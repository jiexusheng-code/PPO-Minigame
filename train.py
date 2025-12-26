"""Minimal training launcher using stable-baselines3 PPO, driven by config file."""

import os
import yaml
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from src.policies.masked_flatten_policy import MaskedFlattenPolicy

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
    env_kwargs = cfg.get("env_kwargs", {})
    os.makedirs(out_dir, exist_ok=True)
    ppo_param_keys = [
        "learning_rate", "ent_coef", "batch_size", "n_epochs", "gamma", "gae_lambda", "n_steps", "clip_range", "vf_coef", "max_grad_norm"
    ]
    ppo_kwargs = {k: cfg[k] for k in ppo_param_keys if k in cfg}
    env_fn = make_env_fn(env_name, env_kwargs)
    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, wrapper_class=Monitor)
    tb_log = os.path.join(base_dir, "tb_logs")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
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
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"[INFO] 从checkpoint加载模型: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=vec_env, tensorboard_log=tb_log, policy=policy, policy_kwargs=policy_kwargs, **ppo_kwargs)
    else:
        model = PPO(policy, vec_env, verbose=1, tensorboard_log=tb_log, policy_kwargs=policy_kwargs, **ppo_kwargs)
    checkpoint_prefix = f"ppo_sc2_{today_str}"
    checkpoint_cb = CheckpointCallback(save_freq=10000, save_path=out_dir, name_prefix=checkpoint_prefix)
    logger.info(f"开始训练，总步数: {total_timesteps}")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    logger.info("训练完成，保存最终模型...")
    model.save(os.path.join(out_dir, f"final_model_{today_str}"))
    with open(os.path.join(out_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    logger.info("配置已保存: config_used.yaml")

if __name__ == "__main__":
    main()
