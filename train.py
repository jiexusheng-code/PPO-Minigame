"""Minimal training launcher using stable-baselines3 PPO.

此脚本为示例：在安装并正确配置 PySC2 与 wrapper 后即可运行训练。
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


def make_env(env_id: str):
    # 延迟导入以避免在未安装 pysc2 时抛错
    try:
        from src.envs.pysc2_gym_wrapper import PySC2GymEnv
    except Exception:
        raise RuntimeError("请先安装并配置 pysc2；或在 src/envs/pysc2_gym_wrapper.py 中替换为你自己的 env 实现。")
    return PySC2GymEnv(map_name=env_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MoveToBeacon")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--out-dir", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = make_env(args.env)

    model = PPO("CnnPolicy", env, verbose=1)

    checkpoint_cb = CheckpointCallback(save_freq=10000, save_path=args.out_dir, name_prefix="ppo_sc2")

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_cb)
    model.save(os.path.join(args.out_dir, "final_model"))


if __name__ == "__main__":
    main()
