"""PySC2 -> Gymnasium wrapper (generic, minimal assumptions).

行为：
- 不对动作/观测做任务特化；按 PySC2 提供的接口动态构建 observation_space，并给出可用动作列表。
- 动作空间为 MultiDiscrete：[fn_id, arg1, arg2, ...]，仅支持此格式；参数维度自动覆盖所有函数的最大需求，若不可用或参数非法则回退/修正。
- reset/step 使用 Gymnasium 返回格式：(obs, info) 与 (obs, reward, terminated, truncated, info)。

说明：这是通用包装器，未假设具体 minigame；真实训练前应根据任务调整 observation 处理与动作编码。
"""
from typing import Any, Dict, Optional

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
            from absl import flags
            # 处理 absl 未解析 flags 导致的异常
            if not flags.FLAGS.is_parsed():
                flags.FLAGS(["pysc2_env"])
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError("未找到 PySC2 依赖，请先安装并确保 StarCraft II 已正确安装与授权。") from exc

        self.sc2_env = sc2_env
        self.actions = actions
        self.features = features

        # 观测预处理器（将原始 observation 转换为结构化张量）
        self._parser: ObsParser = ObsParser(H=self.screen_size, W=self.screen_size)

        self._env = None
        self.observation_space = None  # type: ignore
        self.action_space = None  # type: ignore
        self._max_args = 0
        self._build_action_space()

        # 为了兼容向量化环境，在构造时就确定 observation_space
        self._lazy_init()
        initial_ts = self._env.reset()
        parsed = self._parser.process_observation(initial_ts[0].observation)
        self._build_observation_space_from_parser(parsed)

    def _build_action_space(self):
        """构建 MultiDiscrete 动作空间：第一维是 function_id，后续为参数槽。"""
        n_funcs = len(self.actions.FUNCTIONS)
        max_args = 0
        # 计算每个参数槽需要的最大取值范围
        arg_sizes: list[int] = []
        for fn in self.actions.FUNCTIONS:
            max_args = max(max_args, len(fn.args))
        for arg_idx in range(max_args):
            max_size = 1
            for fn in self.actions.FUNCTIONS:
                if len(fn.args) <= arg_idx:
                    continue
                spec = fn.args[arg_idx]
                name = getattr(spec, "name", "")
                if "screen" in name:
                    size = self.screen_size * self.screen_size
                elif "minimap" in name:
                    size = self.minimap_size * self.minimap_size
                else:
                    size = int(spec.sizes[0]) if getattr(spec, "sizes", None) else 1
                max_size = max(max_size, size)
            arg_sizes.append(max_size)
        self._max_args = max_args
        self.action_space = gym.spaces.MultiDiscrete([n_funcs] + arg_sizes)

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
        # 仅支持 MultiDiscrete 输出 (fn_id, arg1, arg2, ...)
        if not isinstance(action, (list, tuple, np.ndarray)):
            raise ValueError("action 必须是 MultiDiscrete 输出的序列")
        if len(action) == 0:
            return None, []
        fn_id = int(action[0])
        raw_params = list(action[1:])
        return fn_id, raw_params

    def _sanitize_params(self, fn_id: int, raw_params: list) -> tuple[list, bool]:
        """根据函数参数规格将 raw_params 解析为合法值，返回 (args, clipped_flag)。"""
        clipped = False
        fn_id = int(fn_id) if fn_id is not None else 0
        if fn_id < 0 or fn_id >= len(self.actions.FUNCTIONS):
            return [], True
        fn = self.actions.FUNCTIONS[fn_id]
        args = []
        for idx, spec in enumerate(fn.args):
            name = getattr(spec, "name", "")
            sizes = getattr(spec, "sizes", []) or []
            raw_val = raw_params[idx] if idx < len(raw_params) else 0
            try:
                raw_int = int(raw_val)
            except Exception:
                raw_int = 0

            if "screen" in name:
                max_flat = self.screen_size * self.screen_size
                flat = int(np.clip(raw_int, 0, max_flat - 1))
                if flat != raw_int:
                    clipped = True
                y, x = divmod(flat, self.screen_size)
                args.append([x, y])
            elif "minimap" in name:
                max_flat = self.minimap_size * self.minimap_size
                flat = int(np.clip(raw_int, 0, max_flat - 1))
                if flat != raw_int:
                    clipped = True
                y, x = divmod(flat, self.minimap_size)
                args.append([x, y])
            else:
                size = int(sizes[0]) if len(sizes) > 0 else 1
                val = int(np.clip(raw_int, 0, max(size - 1, 0)))
                if val != raw_int:
                    clipped = True
                args.append([val])

        return args, clipped

    def step(self, action) -> tuple:
        assert self._env is not None, "Env not initialized; call reset first"

        fn_id, raw_params = self._unwrap_action(action)
        ts = self._env._obs[0]
        available = ts.observation.get("available_actions", [])

        args, clipped = self._sanitize_params(fn_id, raw_params)

        # 如果不可用则回退 no_op
        if fn_id not in available:
            act = self.actions.FUNCTIONS.no_op()
            clipped = True
        else:
            try:
                act = self.actions.FUNCTIONS[fn_id](*args) if args else self.actions.FUNCTIONS[fn_id]()
            except Exception:
                act = self.actions.FUNCTIONS.no_op()
                clipped = True

        timesteps = self._env.step([act])
        ts = timesteps[0]

        obs = self._obs_to_space(ts)
        reward = float(ts.reward)
        terminated = bool(ts.last())
        truncated = False
        info = {"arg_clipped": clipped, "fn_available": fn_id in available}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # PySC2 内置可视化取决于 visualize 选项
        return None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
