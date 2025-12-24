"""PySC2 -> Gym wrapper (placeholder).

实现说明：PySC2 的接口与 Gym 不完全一致。这里放置一个最小的包装器骨架，后续需要根据选定的 observation / action 配置完善。
"""
from typing import Any
import gymnasium as gym
import numpy as np


class PySC2GymEnv(gym.Env):
    """Minimal wrapper skeleton around a pysc2 environment.

    注意：此文件是占位实现，真实运行前请安装 pysc2 并实现 step/reset/render 等方法。
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, map_name: str = "MoveToBeacon"):
        super().__init__()
        self.map_name = map_name
        # TODO: 初始化 pysc2 环境
        # self._env = ...

        # 示例：使用简单的观测与动作空间占位符
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84, 3), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(8)

    def reset(self, **kwargs) -> tuple:
        # Accept seed and other kwargs from VecEnv/Monitor compatibility.
        # TODO: 调用 pysc2 的 reset 并返回处理后的 observation
        # Gymnasium: return observation, info
        obs = self.observation_space.sample()
        info = {}
        return obs, info

    def step(self, action) -> tuple:
        # TODO: 将 action 转换为 pysc2 action，调用 env.step，并返回
        # Gymnasium: (obs, reward, terminated, truncated, info)
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        pass
