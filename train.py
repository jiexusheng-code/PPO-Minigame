"""Minimal training launcher using stable-baselines3 PPO, driven by config file."""

import os
import glob
import yaml
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from src.policies.masked_flatten_policy import MaskedFlattenPolicy, VectorLayerNormExtractor
from src.callbacks.tb_dual_writer import TBDualWriterCallback

DEFAULT_CONFIG_PATH = os.environ.get("TRAIN_CONFIG_PATH", "./configs/ppo_config.yaml")

def make_env_fn(map_name: str, env_kwargs=None):
    from src.envs.pysc2_gym_wrapper import PySC2GymEnv
    env_kwargs = env_kwargs or {}
    def _init():
        return PySC2GymEnv(map_name=map_name, **env_kwargs)
    return _init

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():

    cfg = load_config(DEFAULT_CONFIG_PATH)
    # Strict config access: require keys to be present in the config file.
    def require(key: str):
        if key not in cfg:
            raise KeyError(f"Required config key '{key}' missing from {DEFAULT_CONFIG_PATH}")
        return cfg[key]

    # forbid legacy/ambiguous key 'eval_freq' to avoid silent misconfiguration
    if "eval_freq" in cfg:
        raise ValueError("Config key 'eval_freq' is not supported; use 'eval_every' (optimization-step units) instead.")

    env_name = require("env")
    import datetime
    today_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Use user-provided out_dir as base path (required). Timestamped subfolder will be created inside it.
    out_dir_root = require("out_dir")
    base_dir = os.path.join(out_dir_root, today_str)
    out_dir = base_dir
    n_envs = require("n_envs")
    seed = require("seed")
    total_timesteps = require("total_timesteps")
    policy_cfg = require("policy")
    policy = MaskedFlattenPolicy if policy_cfg == "MaskedFlattenPolicy" else policy_cfg
    policy_kwargs = require("policy_kwargs")
    if policy is MaskedFlattenPolicy and "features_extractor_class" not in policy_kwargs:
        policy_kwargs["features_extractor_class"] = VectorLayerNormExtractor
    env_kwargs = require("env_kwargs")
    os.makedirs(out_dir, exist_ok=True)
    # expose output dir to worker envs before envs are created
    try:
        os.environ['TRAIN_OUT_DIR'] = base_dir
    except Exception:
        pass
    ppo_param_keys = [
        "learning_rate", "ent_coef", "batch_size", "n_epochs", "gamma", "gae_lambda", "n_steps", "clip_range", "vf_coef", "max_grad_norm"
    ]
    # require all PPO params to be explicitly provided in config
    ppo_kwargs = {k: require(k) for k in ppo_param_keys}
    # Ensure numeric PPO params are proper types (yaml may load exponent as string)
    try:
        lr = ppo_kwargs.get("learning_rate")
        if isinstance(lr, str):
            try:
                ppo_kwargs["learning_rate"] = float(lr)
            except Exception:
                pass
    except Exception:
        pass
    # 读取评估/日志相关配置（统一以 optimization-step 为单位）
    save_iters = require("save_iters")
    summary_iters = require("summary_iters")
    # eval_every must be provided explicitly (optimization-step units)
    eval_every = require("eval_every")
    # 将 optimization-step 转换为环境 timestep 供 EvalCallback 使用：timesteps = opt_steps * n_envs * n_steps
    n_steps = int(require("n_steps"))
    eval_freq = int(eval_every) * n_envs * n_steps
    n_eval_episodes = require("n_eval_episodes")
    eval_deterministic = require("eval_deterministic")
    eval_render = require("eval_render")
    tensorboard = require("tensorboard")
    tb_log_dirname = require("tb_log_dir")
    save_best_model = require("save_best_model")
    verbose_level = require("verbose")
    env_fn = make_env_fn(env_name, env_kwargs)
    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, wrapper_class=Monitor)
    tb_log = os.path.join(base_dir, tb_log_dirname) if tensorboard else None
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    logger = logging.getLogger("train")
    logger.info("==== RL训练启动 ====")
    logger.info(f"输出目录: {out_dir}")
    logger.info(f"TensorBoard目录: {tb_log}")
    logger.info(f"日志文件: {log_file}")
    logger.info(f"环境名: {env_name}")
    logger.info(f"环境参数: {env_kwargs}")
    logger.info(f"训练配置: {cfg}")
    # 记录PyTorch设备信息
    try:
        import torch
        if torch.cuda.is_available():
            device_str = f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})"
            device = "cuda"
        else:
            device_str = "cpu"
            device = "cpu"
        logger.info(f"PyTorch 当前设备: {device_str}")
    except Exception as e:
        logger.warning(f"无法检测PyTorch设备: {e}")
        device = "cpu"
    # checkpoint_path is optional: if provided in the config it must be present as a key (can be empty to mean None)
    if "checkpoint_path" in cfg:
        checkpoint_path = cfg["checkpoint_path"]
        if checkpoint_path in (None, ""):
            checkpoint_path = None
    else:
        checkpoint_path = None
    # Disable SB3's internal TensorBoard writer to avoid SB3 creating algorithm-named
    # subfolders (e.g. PPO_1). Our TBDualWriterCallback will create and manage
    # `fundamental` and `extra` directories and write scalars there.
    tb_log_for_sb3 = None
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"[INFO] 从checkpoint加载模型: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=vec_env, tensorboard_log=tb_log_for_sb3, policy=policy, policy_kwargs=policy_kwargs, device=device, **ppo_kwargs)
    else:
        # honor clip_value_loss config: when True, pass clip_range_vf to SB3 if supported
        clip_value_loss_flag = False
        if "clip_value_loss" in cfg:
            clip_value_loss_flag = cfg["clip_value_loss"]
            if clip_value_loss_flag:
                # set clip_range_vf equal to clip_range (user-provided)
                try:
                    ppo_kwargs["clip_range_vf"] = ppo_kwargs["clip_range"]
                except Exception:
                    pass
        model = PPO(policy, vec_env, verbose=verbose_level, tensorboard_log=tb_log_for_sb3, policy_kwargs=policy_kwargs, device=device, **ppo_kwargs)
    # 使用EvalCallback只保存表现最好的模型
    eval_env = make_vec_env(env_fn, n_envs=1, seed=seed+100, wrapper_class=Monitor)
    best_model_save_path = out_dir if save_best_model else None
    class LogEvalCallback(EvalCallback):
        def __init__(self, *args, logger=None, save_best_only: bool = False, checkpoint_dir: str = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._logger = logger or logging.getLogger("train")
            self._save_best_only = bool(save_best_only)
            self._checkpoint_dir = checkpoint_dir

        def _on_step(self) -> bool:
            do_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
            if do_eval:
                self._logger.info(
                    f"[EvalCallback] 触发评估: num_timesteps={self.num_timesteps}, n_calls={self.n_calls}, eval_freq={self.eval_freq}"
                )
            # remember previous best to detect improvement after super
            prev_best = getattr(self, 'best_mean_reward', float('-inf'))
            result = super()._on_step()
            if do_eval:
                self._logger.info(
                    f"[EvalCallback] 评估完成: num_timesteps={self.num_timesteps}, last_mean_reward={self.last_mean_reward}"
                )
                # if configured to keep only best, and a new best was found, remove periodic checkpoints
                try:
                    new_best = getattr(self, 'best_mean_reward', float('-inf'))
                    if self._save_best_only and self._checkpoint_dir is not None and new_best != prev_best:
                        # remove checkpoint_opt_*.zip files in checkpoint_dir
                        pattern = os.path.join(self._checkpoint_dir, 'checkpoint_opt_*.zip')
                        for path in glob.glob(pattern):
                            try:
                                os.remove(path)
                            except Exception:
                                pass
                except Exception:
                    pass
            return result

    eval_callback = LogEvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=out_dir,
        eval_freq=eval_freq,
        deterministic=eval_deterministic,
        render=eval_render,
        n_eval_episodes=n_eval_episodes,
        logger=logger,
        save_best_only=save_best_model,
        checkpoint_dir=out_dir,
    )
    logger.info(
        f"[EvalCallback] 有效评估频率: eval_freq={eval_callback.eval_freq} (n_envs={vec_env.num_envs})"
    )
    logger.info(f"开始训练，总步数: {total_timesteps}")
    logger.info(f"评估频率(eval_freq): {eval_freq}, 每次评估episode数: {n_eval_episodes}, 保存最佳模型: {save_best_model}")
    logger.info(f"TensorBoard: {'启用' if tensorboard and tb_log else '禁用'}, TB目录: {tb_log}")
    # 日志/打印间隔（多少次学习更新写一次日志）
    log_interval = require("log_interval")
    logger.info(f"日志间隔(log_interval): {log_interval}")
    # attach dual-writer callback together with EvalCallback
    callbacks = [eval_callback]
    if tensorboard and tb_log:
        tb_callback = TBDualWriterCallback(
            tb_log,
            write_env_every=summary_iters,
            write_opt_every=summary_iters,
            save_every=save_iters,
            save_dir=out_dir,
        )
        callbacks.append(tb_callback)
    callback_list = CallbackList(callbacks)

    # If total_timesteps <= 0, treat as "run until interrupted"; otherwise run for given timesteps
    try:
        tt = int(total_timesteps) if total_timesteps is not None else None
    except Exception:
        tt = None

    if tt is None:
        # if unspecified, require explicit total_timesteps in config
        raise ValueError("total_timesteps must be specified in config and be a non-negative integer")

    if tt <= 0:
        logger.info("运行模式: total_timesteps <= 0，持续训练直到手动中断 (Ctrl+C)。")
        try:
            while True:
                model.learn(total_timesteps=10**9, callback=callback_list, reset_num_timesteps=False, log_interval=log_interval)
        except KeyboardInterrupt:
            logger.info("检测到 KeyboardInterrupt，停止训练并保存。")
    else:
        model.learn(total_timesteps=tt, callback=callback_list, log_interval=log_interval)
    logger.info("训练完成，保存最终模型...")
    model.save(os.path.join(out_dir, f"final_model_{today_str}"))
    def _sanitize_for_yaml(obj):
        if isinstance(obj, dict):
            return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ _sanitize_for_yaml(v) for v in obj ]
        if isinstance(obj, type):
            return f"{obj.__module__}.{obj.__name__}"
        return obj

    with open(os.path.join(out_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(_sanitize_for_yaml(cfg), f, allow_unicode=True)
    logger.info("配置已保存: config_used.yaml")
    # If configured to only keep best, remove periodic checkpoint files created during training
    try:
        if save_best_model:
            pattern = os.path.join(out_dir, 'checkpoint_opt_*.zip')
            for path in glob.glob(pattern):
                try:
                    os.remove(path)
                except Exception:
                    pass
            logger.info("已清理周期性 checkpoint，只保留最佳/最终模型（save_best_model=True）。")
    except Exception:
        pass

if __name__ == "__main__":
    # 修复 pysc2 的 shuffled_hue 问题，兼容 Python 3.9+
    import sys
    import types

    def shuffled_hue_patch(palette_size):
        import random
        palette = [i for i in range(palette_size)]
        random.shuffle(palette)  # Python 3.9+ 兼容写法
        return palette

    # 猴子补丁 pysc2 的 shuffled_hue
    sys.modules_backup = dict(sys.modules)
    try:
        import pysc2.lib.colors
        pysc2.lib.colors.shuffled_hue = shuffled_hue_patch
    except ImportError:
        pass

    main()
