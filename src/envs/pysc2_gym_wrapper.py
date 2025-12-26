"""PySC2 -> Gymnasium minigame wrapper, 兼容精简ObsParser输出。"""
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from src.obs_parser import ObsParser
from pysc2.lib import features


class PySC2GymEnv(gym.Env):
    """Generic PySC2 wrapper compatible with Gymnasium."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        map_name: str = None,
        screen_size: int = 64,
        minimap_size: int = None,
        step_mul: int = 8,
        visualize: bool = False,
        debug_print: bool = True,
    ):
        super().__init__()
        self.map_name = map_name
        self.screen_size = screen_size
        self.minimap_size = minimap_size if minimap_size is not None else screen_size
        self.step_mul = step_mul
        self.visualize = visualize
        self.debug_print = debug_print
        self._step_count = 0  # 记录步数

        try:
            from pysc2.env import sc2_env
            from pysc2.lib import actions, features
            from absl import flags
            if not flags.FLAGS.is_parsed():
                flags.FLAGS(["pysc2_env"])
        except Exception as exc:
            raise RuntimeError("未找到 PySC2 依赖，请先安装并确保 StarCraft II 已正确安装与授权。") from exc

        self.sc2_env = sc2_env
        self.actions = actions
        self.features = features
        self._parser: ObsParser = ObsParser(H=self.screen_size, W=self.screen_size)
        self._env = None
        self.observation_space = None
        self.action_space = None
        self._max_args = 0
        self.timestep = None  # 维护当前timestep
        self._build_action_space()
        self._lazy_init()
        initial_ts = self._env.reset()
        self.timestep = initial_ts[0]
        parsed = self._parser.process_observation(self.timestep.observation)
        parsed_flat = self._flatten_parsed(parsed)
        self._build_observation_space_from_parser(parsed_flat)

    def _build_action_space(self):
        n_funcs = len(self.actions.FUNCTIONS)
        # 1. 收集所有参数语义（唯一槽位）
        param_semantics = []  # 语义唯一的参数名列表
        param_semantics_set = set()
        param_size_dict = {}  # 语义->最大size
        for fn in self.actions.FUNCTIONS:
            for spec in fn.args:
                name = getattr(spec, "name", "")
                # 只保留语义唯一的参数名
                if name not in param_semantics_set:
                    param_semantics.append(name)
                    param_semantics_set.add(name)
                # 统计最大size
                if "screen" in name  or "screen2" in name:
                    size = self.screen_size * self.screen_size
                elif "minimap" in name:
                    size = self.minimap_size * self.minimap_size
                else:
                    size = int(spec.sizes[0]) if getattr(spec, "sizes", None) else 1
                if name not in param_size_dict or size > param_size_dict[name]:
                    param_size_dict[name] = size
        # 2. 构建MultiDiscrete动作空间
        arg_sizes = [param_size_dict[name] for name in param_semantics]
        self._param_semantics = param_semantics  # 参数槽位顺序
        self._param_size_dict = param_size_dict
        self._fn_param_map = {}  # fn_id -> [槽位索引]
        for fn in self.actions.FUNCTIONS:
            slot_indices = []
            for spec in fn.args:
                name = getattr(spec, "name", "")
                slot_indices.append(param_semantics.index(name))
            self._fn_param_map[fn.id] = slot_indices
        self._max_args = len(param_semantics)
        self.action_space = gym.spaces.MultiDiscrete([n_funcs] + arg_sizes)
        # 打印参数槽位分配表，便于调试
        if self.debug_print:
            print("[动作参数槽位分配表]")
            for i, name in enumerate(param_semantics):
                print(f"槽位{i}: {name}, size={param_size_dict[name]}")
            print("[动作类型到槽位映射]")
            for fn in self.actions.FUNCTIONS:
                print(f"fn_id={fn.id}, name={fn.name}, 槽位索引={self._fn_param_map[fn.id]}")

    def _lazy_init(self):
        if self._env is not None:
            return
        interface = self.sc2_env.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
            use_feature_units=True,
            action_space=self.actions.ActionSpace.FEATURES,
            use_raw_actions=False,
        )
        # 与obs_parser.py测试代码保持一致，显式设置game_steps_per_episode=0
        self._env = self.sc2_env.SC2Env(
            map_name=self.map_name,
            players=[self.sc2_env.Agent(self.sc2_env.Race.terran)],
            agent_interface_format=interface,
            step_mul=self.step_mul,
            game_steps_per_episode=0,
            visualize=self.visualize,
            ensure_available_actions=True
        )

    def _build_observation_space_from_parser(self, parsed: Dict[str, object]):
        spaces: Dict[str, gym.Space] = {}
        spatial = parsed.get("spatial_fused")
        if spatial is not None:
            c, h, w = spatial.shape
            spaces["spatial_fused"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(c, h, w), dtype=np.float32)
        scalar = parsed.get("scalar", {}) if isinstance(parsed.get("scalar", {}), dict) else {}
        if "scalar_vec" in scalar:
            vec = scalar["scalar_vec"]
            spaces["scalar_vec"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=vec.shape, dtype=vec.dtype)
        if "scalar_mask" in scalar:
            mask = scalar["scalar_mask"]
            spaces["scalar_mask"] = gym.spaces.Box(low=0.0, high=1.0, shape=mask.shape, dtype=mask.dtype)
        entities = parsed.get("entities", {}) if isinstance(parsed.get("entities", {}), dict) else {}
        if entities:
            N = getattr(self._parser, "N_max", 512)
            spaces["type_ids"] = gym.spaces.Box(low=0, high=1e6, shape=(N,), dtype=np.int64)
            spaces["owner_ids"] = gym.spaces.Box(low=0, high=10, shape=(N,), dtype=np.int64)
            ent_feats_shape = entities.get("ent_feats", np.zeros((N, 5))).shape
            spaces["ent_feats"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=ent_feats_shape, dtype=np.float32)
            spaces["ent_mask"] = gym.spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
            spaces["coords"] = gym.spaces.Box(low=0, high=self.screen_size, shape=(N, 2), dtype=np.int32)
        action_mask = parsed.get("action_mask")
        if action_mask is not None:
            spaces["action_mask"] = gym.spaces.Box(low=0.0, high=1.0, shape=action_mask.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def _obs_to_space(self, timestep) -> Dict[str, Any]:
        raw_obs = timestep.observation
        #print("DEBUG: observation fields:", dir(raw_obs))
        parsed = self._parser.process_observation(raw_obs)
        parsed_flat = self._flatten_parsed(parsed)
        if self.observation_space is None:
            self._build_observation_space_from_parser(parsed_flat)
        return parsed_flat

    def _flatten_parsed(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        def _add(name: str, val: Any, dtype) -> None:
            if val is None:
                return
            try:
                out[name] = np.array(val, dtype=dtype, copy=False)
            except Exception:
                return
        if "spatial_fused" in parsed:
            _add("spatial_fused", parsed["spatial_fused"], np.float32)
        scalar = parsed.get("scalar", {}) if isinstance(parsed.get("scalar", {}), dict) else {}
        if "scalar_vec" in scalar:
            _add("scalar_vec", scalar["scalar_vec"], np.float32)
        if "scalar_mask" in scalar:
            _add("scalar_mask", scalar["scalar_mask"], np.float32)
        entities = parsed.get("entities", {}) if isinstance(parsed.get("entities", {}), dict) else {}
        if entities:
            _add("type_ids", entities.get("type_ids"), np.int64)
            _add("owner_ids", entities.get("owner_ids"), np.int64)
            _add("ent_feats", entities.get("ent_feats"), np.float32)
            _add("ent_mask", entities.get("ent_mask"), np.float32)
            _add("coords", entities.get("coords"), np.int32)
        if "action_mask" in parsed:
            _add("action_mask", parsed["action_mask"], np.float32)
        return out

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self._lazy_init()
        timesteps = self._env.reset()
        self.timestep = timesteps[0]
        obs_dict = self._obs_to_space(self.timestep)
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

            if "screen" in name or "minimap" in name:
                max_flat = self.screen_size * self.screen_size
                if not (0 <= raw_int < max_flat):
                    flat = np.random.randint(0, max_flat)
                    clipped = True
                else:
                    flat = raw_int
                y, x = divmod(flat, self.screen_size)
                args.append([x, y])
            else:
                size = int(sizes[0]) if len(sizes) > 0 else 1
                if not (0 <= raw_int < size):
                    val = np.random.randint(0, max(size, 1))
                    clipped = True
                else:
                    val = raw_int
                args.append([val])
        return args, clipped

    def step(self, action) -> tuple:
        assert self._env is not None, "Env not initialized; call reset first"

        fn_id, raw_params = self._unwrap_action(action)
        # 用self.timestep获取当前观测
        obs_raw = self.timestep.observation
        parsed_now = self._parser.process_observation(obs_raw)
        parsed_flat_now = self._flatten_parsed(parsed_now)
        action_mask = parsed_flat_now.get("action_mask")
        if action_mask is None:
            raise RuntimeError("action_mask missing; cannot determine available actions")
        available = np.nonzero(np.asarray(action_mask).reshape(-1))[0].tolist()

        args, clipped = self._sanitize_params(fn_id, raw_params)

        # 如果不可用则回退 no_op
        if fn_id not in available:
            act = self.actions.FUNCTIONS.no_op()
            clipped = True
            exec_fn_id = self.actions.FUNCTIONS.no_op.id
            exec_args = []
        else:
            try:
                act = self.actions.FUNCTIONS[fn_id](*args) if args else self.actions.FUNCTIONS[fn_id]()
                exec_fn_id = fn_id
                exec_args = args
            except Exception:
                act = self.actions.FUNCTIONS.no_op()
                clipped = True
                exec_fn_id = self.actions.FUNCTIONS.no_op.id
                exec_args = []

        timesteps = self._env.step([act])
        self.timestep = timesteps[0]  # 更新当前timestep
        self._step_count += 1

        obs = self._obs_to_space(self.timestep)
        reward = float(self.timestep.reward)
        terminated = bool(self.timestep.last())
        truncated = False
        info = {"arg_clipped": clipped, "fn_available": fn_id in available}
        if self.debug_print:
            print(f"Step {self._step_count}")
            print(f"  Agent raw action: fn_id={fn_id}, raw_params={raw_params}")
            print(f"  Executed action: fn_id={exec_fn_id}, args={exec_args}")
            print(f"  Available actions: {len(available)}, reward={reward:.3f}, terminated={terminated}, clipped={clipped}")
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
