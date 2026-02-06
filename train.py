"""Minimal training launcher using stable-baselines3 PPO, driven by config file."""

import os
import yaml
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from src.policies.masked_flatten_policy import MaskedFlattenPolicy, VectorLayerNormExtractor

DEFAULT_CONFIG_PATH = "./configs/ppo_config.yaml"

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
    env_name = cfg.get("env", "MoveToBeacon")
    import datetime
    today_str = datetime.datetime.now().strftime("%Y%m%d%H")
    base_dir = os.path.join("models", today_str)
    out_dir = base_dir
    n_envs = cfg.get("n_envs", 1)
    seed = cfg.get("seed", 0)
    total_timesteps = cfg.get("total_timesteps", 100000)
    policy = MaskedFlattenPolicy if cfg.get("policy", "MaskedFlattenPolicy") == "MaskedFlattenPolicy" else cfg["policy"]
    policy_kwargs = cfg.get("policy_kwargs", {})
    if policy is MaskedFlattenPolicy and "features_extractor_class" not in policy_kwargs:
        policy_kwargs["features_extractor_class"] = VectorLayerNormExtractor
    env_kwargs = cfg.get("env_kwargs", {})
    os.makedirs(out_dir, exist_ok=True)
    ppo_param_keys = [
        "learning_rate", "ent_coef", "batch_size", "n_epochs", "gamma", "gae_lambda", "n_steps", "clip_range", "vf_coef", "max_grad_norm"
    ]
    ppo_kwargs = {k: cfg[k] for k in ppo_param_keys if k in cfg}
    # 读取评估/日志相关配置
    eval_freq = cfg.get("eval_freq", 10000)
    n_eval_episodes = cfg.get("n_eval_episodes", 5)
    eval_deterministic = cfg.get("eval_deterministic", True)
    eval_render = cfg.get("eval_render", False)
    tensorboard = cfg.get("tensorboard", True)
    tb_log_dirname = cfg.get("tb_log_dir", "tb_logs")
    save_best_model = cfg.get("save_best_model", True)
    verbose_level = cfg.get("verbose", 1)
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
        else:
            device_str = "cpu"
        logger.info(f"PyTorch 当前设备: {device_str}")
    except Exception as e:
        logger.warning(f"无法检测PyTorch设备: {e}")
    device = "auto"
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"[INFO] 从checkpoint加载模型: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=vec_env, tensorboard_log=tb_log, policy=policy, policy_kwargs=policy_kwargs, device=device, **ppo_kwargs)
    else:
        model = PPO(policy, vec_env, verbose=verbose_level, tensorboard_log=tb_log, policy_kwargs=policy_kwargs, device=device, **ppo_kwargs)
    # 使用EvalCallback只保存表现最好的模型
    eval_env = make_vec_env(env_fn, n_envs=1, seed=seed+100, wrapper_class=Monitor)
    best_model_save_path = out_dir if save_best_model else None
    class LogEvalCallback(EvalCallback):
        def __init__(self, *args, logger=None, log_native_reward=False, **kwargs):
            super().__init__(*args, **kwargs)
            self._logger = logger or logging.getLogger("train")
            self._log_native_reward = log_native_reward

        def _log_rollout_native_reward(self):
            infos = self.locals.get("infos") if isinstance(self.locals, dict) else None
            if not infos:
                return
            native_vals = []
            for info in infos:
                if isinstance(info, dict) and "native_reward" in info:
                    native_vals.append(float(info.get("native_reward", 0.0)))
            if native_vals:
                self.logger.record("rollout/native_reward", float(np.mean(native_vals)))

        def _evaluate_policy(self, model, eval_env, n_eval_episodes, deterministic, render):
            ep_rewards = []
            ep_native_rewards = []

            for _ in range(n_eval_episodes):
                obs = eval_env.reset()
                done = False
                ep_reward = 0.0
                ep_native = 0.0
                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    obs, rewards, dones, infos = eval_env.step(action)
                    reward = float(rewards[0]) if isinstance(rewards, (list, tuple, np.ndarray)) else float(rewards)
                    info = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
                    done = bool(dones[0]) if isinstance(dones, (list, tuple, np.ndarray)) else bool(dones)
                    ep_reward += reward
                    if isinstance(info, dict):
                        ep_native += float(info.get("native_reward", 0.0))
                ep_rewards.append(ep_reward)
                ep_native_rewards.append(ep_native)

            mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            std_reward = float(np.std(ep_rewards)) if ep_rewards else 0.0
            if self._log_native_reward and ep_native_rewards:
                self.logger.record("eval/native_reward", float(np.mean(ep_native_rewards)))
            return mean_reward, std_reward

        def _on_step(self) -> bool:
            if self._log_native_reward:
                self._log_rollout_native_reward()
            do_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
            if do_eval:
                self._logger.info(
                    f"[EvalCallback] 触发评估: num_timesteps={self.num_timesteps}, n_calls={self.n_calls}, eval_freq={self.eval_freq}"
                )
            result = super()._on_step()
            if do_eval:
                self._logger.info(
                    f"[EvalCallback] 评估完成: num_timesteps={self.num_timesteps}, last_mean_reward={self.last_mean_reward}"
                )
            return result

    reward_mode = env_kwargs.get("reward_mode", "native")
    eval_callback = LogEvalCallback(
        eval_env,
        best_model_save_path=best_model_save_path,
        log_path=out_dir,
        eval_freq=eval_freq,
        deterministic=eval_deterministic,
        render=eval_render,
        n_eval_episodes=n_eval_episodes,
        logger=logger,
        log_native_reward=(reward_mode == "custom")
    )
    logger.info(
        f"[EvalCallback] 有效评估频率: eval_freq={eval_callback.eval_freq} (n_envs={vec_env.num_envs})"
    )
    logger.info(f"开始训练，总步数: {total_timesteps}")
    logger.info(f"评估频率(eval_freq): {eval_freq}, 每次评估episode数: {n_eval_episodes}, 保存最佳模型: {save_best_model}")
    logger.info(f"TensorBoard: {'启用' if tensorboard and tb_log else '禁用'}, TB目录: {tb_log}")
    # 日志/打印间隔（多少次学习更新写一次日志）
    log_interval = cfg.get("log_interval", 1)
    logger.info(f"日志间隔(log_interval): {log_interval}")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=log_interval)
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
