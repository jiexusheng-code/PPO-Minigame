"""PySC2 -> Gymnasium wrapper (generic, minimal assumptions).

行为：
- 不对动作/观测做任务特化；按 PySC2 提供的接口动态构建 observation_space，并给出可用动作列表。
- 动作形式固定为 dict：{"function_id": fn_id, "args": args}；若不可用或参数非法则回退 no_op。
- reset/step 使用 Gymnasium 返回格式：(obs, info) 与 (obs, reward, terminated, truncated, info)。

说明：这是通用包装器，未假设具体 minigame；真实训练前应根据任务调整 observation 处理与动作编码。
"""
from typing import Any, Dict

import gymnasium as gym
import numpy as np

from src.obs_parser import ObsParser


class PySC2GymEnv(gym.Env):
    """Generic PySC2 wrapper compatible with Gymnasium."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        map_name: str = None,
        screen_size: int = 64,
        minimap_size: int = 64,
        step_mul: int = 8,
        visualize: bool = False,
    ):#__init__ 只存参数，不启动 SC2
        super().__init__()
        self.map_name = map_name
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.step_mul = step_mul
        self.visualize = visualize

        try:
            from pysc2.env import sc2_env
            from pysc2.lib import actions, features
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("未找到 PySC2 依赖，请先安装并确保 StarCraft II 已正确安装与授权。") from exc

        self.sc2_env = sc2_env
        self.actions = actions
        self.features = features

        # 观测预处理器（将原始 observation 转换为结构化张量）
        self._parser: ObsParser = ObsParser(H=self.screen_size, W=self.screen_size)

        self._env = None
        self.observation_space = None  # type: ignore
        self.action_space = gym.spaces.Discrete(len(self.actions.FUNCTIONS))

    def _lazy_init(self):
        """延迟初始化 SC2Env 实例"""
        if self._env is not None:
            return
        dims = self.sc2_env.Dimensions(screen=(self.screen_size, self.screen_size), minimap=(self.minimap_size, self.minimap_size))
        interface = self.sc2_env.AgentInterfaceFormat(feature_dimensions=dims, use_feature_units=False)
        self._env = self.sc2_env.SC2Env(
            map_name=self.map_name,
            players=[self.sc2_env.Agent(self.sc2_env.Race.terran)],
            agent_interface_format=interface,
            step_mul=self.step_mul,
            visualize=self.visualize,
        )

    def _build_observation_space_from_parser(self, parsed: Dict[str, object]):
        """根据 ObsParser 输出构建 Gymnasium Dict 空间。"""
        spaces: Dict[str, gym.Space] = {}

        # spatial_fused: (C, H, W)
        spatial = parsed.get("spatial_fused")
        if spatial is not None:
            c, h, w = spatial.shape
            spaces["spatial_fused"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(c, h, w), dtype=np.float32)

        # scalar: {scalar_vec, scalar_mask}
        scalar = parsed.get("scalar", {})
        if scalar:
            vec = scalar.get("scalar_vec")
            mask = scalar.get("scalar_mask")
            if vec is not None:
                spaces["scalar_vec"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=vec.shape, dtype=vec.dtype)
            if mask is not None:
                spaces["scalar_mask"] = gym.spaces.Box(low=0.0, high=1.0, shape=mask.shape, dtype=mask.dtype)

        # entities: fixed sizes based on parser.N_max
        entities = parsed.get("entities", {})
        if entities:
            N = getattr(self._parser, "N_max", 512) if self._parser is not None else 512
            spaces["type_ids"] = gym.spaces.Box(low=0, high=1e6, shape=(N,), dtype=np.int64)
            spaces["owner_ids"] = gym.spaces.Box(low=0, high=10, shape=(N,), dtype=np.int64)
            spaces["ent_feats"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=entities.get("ent_feats", np.zeros((N, 5))).shape, dtype=np.float32)
            spaces["ent_mask"] = gym.spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
            spaces["coords"] = gym.spaces.Box(low=0, high=max(self.screen_size, self.minimap_size), shape=(N, 2), dtype=np.int32)

        # action_mask
        action_mask = parsed.get("action_mask")
        if action_mask is not None:
            spaces["action_mask"] = gym.spaces.Box(low=0.0, high=1.0, shape=action_mask.shape, dtype=np.float32)

        self.observation_space = gym.spaces.Dict(spaces)

    def _obs_to_space(self, timestep) -> Dict[str, Any]:
        raw_obs = timestep.observation

        parsed = self._parser.process_observation(raw_obs)
        if self.observation_space is None:
            self._build_observation_space_from_parser(parsed)
        return parsed

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self._lazy_init()
        timesteps = self._env.reset()
        obs_dict = self._obs_to_space(timesteps[0])
        info = {}
        return obs_dict, info

    def _unwrap_action(self, action):
        # 固定动作格式：dict{"function_id": fn_id, "args": args}
        if not isinstance(action, dict):
            raise ValueError("action 必须是 dict，包含 function_id 与 args")
        fn_id = action.get("function_id")
        args = action.get("args", [])
        return fn_id, args

    def step(self, action) -> tuple:
        assert self._env is not None, "Env not initialized; call reset first"

        fn_id, args = self._unwrap_action(action)
        ts = self._env._obs[0]
        available = ts.observation.get("available_actions", [])

        # 如果不可用或参数不匹配，则 no_op
        if fn_id not in available:
            act = self.actions.FUNCTIONS.no_op()
        else:
            try:
                act = self.actions.FUNCTIONS[fn_id](*args) if args else self.actions.FUNCTIONS[fn_id]()
            except Exception:
                act = self.actions.FUNCTIONS.no_op()

        timesteps = self._env.step([act])
        ts = timesteps[0]

        obs = self._obs_to_space(ts)
        reward = float(ts.reward)
        terminated = bool(ts.last())
        truncated = False
        info = {"available_actions": obs.get("available_actions")}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # PySC2 内置可视化取决于 visualize 选项
        return None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
