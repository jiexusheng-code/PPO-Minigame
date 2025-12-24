"""Minimal training launcher using stable-baselines3 PPO.

此脚本为示例：在安装并正确配置 PySC2 与 wrapper 后即可运行训练。
"""
import argparse
import os
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


def make_env_fn(map_name: str):
    # 延迟导入以避免在未安装 pysc2 时抛错
    try:
        from src.envs.pysc2_gym_wrapper import PySC2GymEnv
    except Exception:
        raise RuntimeError("请先安装并配置 pysc2；或在 src/envs/pysc2_gym_wrapper.py 中替换为你自己的 env 实现。")

    def _init():
        return PySC2GymEnv(map_name=map_name)

    return _init


def load_config(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MoveToBeacon")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="./models")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, default="./configs/ppo_config.yaml")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_config(args.config)
    total_timesteps = args.total_timesteps or cfg.get("total_timesteps", 100000)
    policy = cfg.get("policy", "CnnPolicy")

    # 创建向量化环境（使用 Monitor 作为 wrapper）
    env_fn = make_env_fn(args.env)
    vec_env = make_vec_env(env_fn, n_envs=args.n_envs, seed=args.seed, wrapper_class=Monitor)

    tb_log = os.path.join(args.out_dir, "tb_logs")
    model = PPO(policy, vec_env, verbose=1, tensorboard_log=tb_log)

    checkpoint_cb = CheckpointCallback(save_freq=10000, save_path=args.out_dir, name_prefix="ppo_sc2")

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    model.save(os.path.join(args.out_dir, "final_model"))


if __name__ == "__main__":
    main()
