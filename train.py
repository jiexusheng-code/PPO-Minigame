"""Minimal training launcher using stable-baselines3 PPO.

此脚本为示例：完全由配置文件驱动（无命令行参数）。
"""
import os
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from src.policies.masked_flatten_policy import MaskedFlattenPolicy


DEFAULT_CONFIG_PATH = "./configs/ppo_config.yaml"


def make_env_fn(map_name: str, env_kwargs=None):
    # 延迟导入以避免在未安装 pysc2 时抛错
    try:
        from src.envs.pysc2_gym_wrapper import PySC2GymEnv
    except Exception:
        raise RuntimeError("请先安装并配置 pysc2；或在 src/envs/pysc2_gym_wrapper.py 中替换为你自己的 env 实现。")

    env_kwargs = env_kwargs or {}

    def _init():
        return PySC2GymEnv(map_name=map_name, **env_kwargs)

    return _init


def load_config(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config(DEFAULT_CONFIG_PATH)

    env_name = cfg.get("env", "MoveToBeacon")
    out_dir = cfg.get("out_dir", "./models")
    n_envs = cfg.get("n_envs", 1)
    seed = cfg.get("seed", 0)
    total_timesteps = cfg.get("total_timesteps", 100000)
    policy = cfg.get("policy", MaskedFlattenPolicy)
    if isinstance(policy, str) and policy == "MaskedFlattenPolicy":
        policy = MaskedFlattenPolicy
    policy_kwargs = cfg.get("policy_kwargs", {})
    env_kwargs = cfg.get("env_kwargs", {})

    os.makedirs(out_dir, exist_ok=True)

    # 组装可传递给 PPO 的超参（policy/total_timesteps/env 等不传入 kwargs）
    ppo_param_keys = [
        "learning_rate",
        "ent_coef",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "n_steps",
        "clip_range",
        "vf_coef",
        "max_grad_norm",
    ]
    ppo_kwargs = {k: cfg[k] for k in ppo_param_keys if k in cfg}

    # 创建向量化环境（使用 Monitor 作为 wrapper）
    env_fn = make_env_fn(env_name, env_kwargs)
    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, wrapper_class=Monitor)

    tb_log = os.path.join(out_dir, "tb_logs")
    model = PPO(policy, vec_env, verbose=1, tensorboard_log=tb_log, policy_kwargs=policy_kwargs, **ppo_kwargs)

    checkpoint_cb = CheckpointCallback(save_freq=10000, save_path=out_dir, name_prefix="ppo_sc2")

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    model.save(os.path.join(out_dir, "final_model"))

    # 将实际使用的配置持久化一份，便于复现
    with open(os.path.join(out_dir, "config_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


if __name__ == "__main__":
    main()
