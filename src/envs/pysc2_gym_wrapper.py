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
        screen_size: int = 32,
        minimap_size: int = 32,
        step_mul: int = 8,
        visualize: bool = False,
        reward_mode: str = "native",
    ):
        import logging, os, datetime
        super().__init__()
        self.map_name = map_name
        self.screen_size = screen_size
        self.minimap_size = minimap_size 
        self.step_mul = step_mul
        self.visualize = visualize
        self.reward_mode = reward_mode
        self._step_count = 0  # 记录步数

        # 日志设置：只用主进程的logging.basicConfig，所有模块共用同一日志文件
        self.logger = logging.getLogger("PySC2GymEnv")

        try:
            from pysc2.env import sc2_env
            from pysc2.lib import actions, features
            from absl import flags
            if not flags.FLAGS.is_parsed():
                run_config = "Linux" if os.name != "nt" else "Windows"
                flags.FLAGS(["pysc2_env", f"--sc2_run_config={run_config}"])
        except Exception as exc:
            self.logger.error(f"无法导入 pysc2，请确保已正确安装。错误详情: {exc}")
            raise RuntimeError(f"无法导入 pysc2，请确保已正确安装。错误详情: {exc}") from exc

        self.sc2_env = sc2_env
        self.actions = actions
        self.features = features
        if self.map_name is None:
            raise ValueError("map_name 不能为空，需与 ObsParser 配置一致")
        self._parser: ObsParser = ObsParser(self.map_name, screen_size=self.screen_size, minimap_size=self.minimap_size)
        self._env = None
        self.observation_space = None
        self.action_space = None
        self._max_args = 0
        self.timestep = None  # 维护当前timestep
        self._reward_state = None
        self._reward_fn = None
        self._lazy_init()
        self._build_action_space()
        initial_ts = self._env.reset()
        self.timestep = initial_ts[0]
        parsed = self._parser.parse(self.timestep.observation)
        parsed_flat = self._flatten_parsed(parsed)
        self._build_observation_space_from_parser(parsed_flat)

        # Reward selection (native/custom)
        if self.reward_mode not in ["native", "custom"]:
            raise ValueError("reward_mode must be 'native' or 'custom'")
        if self.reward_mode == "custom":
            from src.reward_registry import get_reward_fn, RewardState
            self._reward_fn = get_reward_fn(self.map_name)
            if self._reward_fn is None:
                raise ValueError(f"No custom reward registered for map: {self.map_name}")
            self._reward_state = RewardState()

    def _build_action_space(self):
        try:
            if self._env is None:
                raise RuntimeError("Env not initialized before building action space")
            action_spec = self._env.action_spec()[0]
            self._action_spec_types = action_spec.types
            n_funcs = len(self.actions.FUNCTIONS)
            # 1. 收集所有参数语义（唯一槽位）
            param_semantics = []  # 语义唯一的参数名列表
            param_semantics_set = set()
            param_size_dict = {}  # 语义->最大size
            for fn in self.actions.FUNCTIONS:
                for spec in fn.args:
                    name = getattr(spec, "name")
                    # 只保留语义唯一的参数名
                    if name not in param_semantics_set:
                        param_semantics.append(name)
                        param_semantics_set.add(name)
                    if getattr(spec, "sizes", None):
                        if name in self._action_spec_types._fields:
                            sizes = getattr(self._action_spec_types, name).sizes
                            if len(sizes) >= 2:
                                size = int(sizes[0]) * int(sizes[1])
                            else:
                                size = int(sizes[0])
                        else:
                            size = int(spec.sizes[0])
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
                    name = getattr(spec, "name")
                    slot_indices.append(param_semantics.index(name))
                self._fn_param_map[fn.id] = slot_indices
            self._max_args = len(param_semantics)
            self.action_space = gym.spaces.MultiDiscrete([n_funcs] + arg_sizes)
        except Exception as e:
            self.logger.error(f"构建动作空间失败: {e}")
            raise RuntimeError(f"构建动作空间失败: {e}") from e
        

    def _lazy_init(self):
        if self._env is not None:
            return
        interface = self.sc2_env.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
            use_feature_units=True,
            action_space=self.actions.ActionSpace.FEATURES,
            use_raw_actions=False,
        )
        self._env = self.sc2_env.SC2Env(
            map_name=self.map_name,
            players=[self.sc2_env.Agent(self.sc2_env.Race.terran)],
            agent_interface_format=interface,
            step_mul=self.step_mul,
            game_steps_per_episode=0,
            visualize=self.visualize,
            ensure_available_actions=True,
            score_index=0,
        )

    def _build_observation_space_from_parser(self, parsed: Dict[str, object]):
        spaces: Dict[str, gym.Space] = {}
        vector = parsed.get("vector")
        if vector is not None:
            spaces["vector"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=vector.shape, dtype=np.float32)
        screen = parsed.get("screen")
        if screen is not None:
            spaces["screen"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=screen.shape, dtype=np.float32)
        minimap = parsed.get("minimap")
        if minimap is not None:
            spaces["minimap"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=minimap.shape, dtype=np.float32)
        available_actions = parsed.get("available_actions")
        if available_actions is not None:
            spaces["available_actions"] = gym.spaces.Box(low=0.0, high=1.0, shape=available_actions.shape, dtype=np.float32)
        screen_layer_flags = parsed.get("screen_layer_flags")
        if screen_layer_flags is not None:
            spaces["screen_layer_flags"] = gym.spaces.Box(low=0.0, high=1.0, shape=screen_layer_flags.shape, dtype=np.float32)
        minimap_layer_flags = parsed.get("minimap_layer_flags")
        if minimap_layer_flags is not None:
            spaces["minimap_layer_flags"] = gym.spaces.Box(low=0.0, high=1.0, shape=minimap_layer_flags.shape, dtype=np.float32)
        vector_mask = parsed.get("vector_mask")
        if vector_mask is not None:
            spaces["vector_mask"] = gym.spaces.Box(low=0.0, high=1.0, shape=vector_mask.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def _obs_to_space(self, timestep) -> Dict[str, Any]:
        raw_obs = timestep.observation
        #print("DEBUG: observation fields:", dir(raw_obs))
        parsed = self._parser.parse(raw_obs)
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
        _add("vector", parsed.get("vector"), np.float32)
        _add("screen", parsed.get("screen"), np.float32)
        _add("minimap", parsed.get("minimap"), np.float32)
        _add("available_actions", parsed.get("available_actions"), np.float32)
        _add("screen_layer_flags", parsed.get("screen_layer_flags"), np.float32)
        _add("minimap_layer_flags", parsed.get("minimap_layer_flags"), np.float32)
        _add("vector_mask", parsed.get("vector_mask"), np.float32)
        return out

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        self._lazy_init()
        timesteps = self._env.reset()
        self.timestep = timesteps[0]
        obs_dict = self._obs_to_space(self.timestep)
        info = {}
        if self.reward_mode == "custom" and self._reward_fn is not None:
            # Reset reward state and compute initial distance baseline
            from src.reward_registry import RewardState
            self._reward_state = RewardState()
            _ = self._reward_fn(obs_dict, self._reward_state, info)
        return obs_dict, info

    def _unwrap_action(self, action):
        # 仅支持 MultiDiscrete 输出 (fn_id, param_vec)
        if not isinstance(action, (list, tuple, np.ndarray)):
            raise ValueError("action 必须是 MultiDiscrete 输出的序列")
        if len(action) == 0:
            return None, []
        fn_id = int(action[0])
        param_vec = list(action[1:])
        return fn_id, param_vec

    def _sanitize_params(self, fn_id: int, param_vec: list) -> tuple[list, bool]:
        try:
            fn_id = int(fn_id) if fn_id is not None else 0
            if fn_id < 0 or fn_id >= len(self.actions.FUNCTIONS):
                raise ValueError(f"非法fn_id: {fn_id}")
            fn = self.actions.FUNCTIONS[fn_id]
            slot_indices = self._fn_param_map.get(fn_id, [])
            args = []
            for idx, spec in enumerate(fn.args):
                name = getattr(spec, "name", "")
                sizes = getattr(spec, "sizes", []) or []
                slot = slot_indices[idx] if idx < len(slot_indices) else None
                if slot is None or slot >= len(param_vec):
                    raise RuntimeError(f"参数槽位缺失: fn_id={fn_id}, 参数{name}, 槽位={slot}")
                raw_val = param_vec[slot]
                try:
                    raw_int = int(raw_val)
                except Exception:
                    raise RuntimeError(f"参数类型错误: fn_id={fn_id}, 参数{name}, 槽位={slot}, raw_val={raw_val}")

                if name in self._action_spec_types._fields:
                    sizes = getattr(self._action_spec_types, name).sizes
                else:
                    sizes = sizes

                if name in ["minimap", "screen", "screen2"] and len(sizes) >= 2:
                    h, w = int(sizes[0]), int(sizes[1])
                    max_flat = h * w
                    if not (0 <= raw_int < max_flat):
                        raise ValueError(f"参数越界: fn_id={fn_id}, 参数{name}, 槽位={slot}, raw_val={raw_val}")
                    y, x = divmod(raw_int, w)
                    args.append([x, y])
                else:
                    size = int(sizes[0]) if len(sizes) > 0 else 1
                    if not (0 <= raw_int < size):
                        raise ValueError(f"参数越界: fn_id={fn_id}, 参数{name}, 槽位={slot}, raw_val={raw_val}")
                    args.append([raw_int])
            return args, False
        except Exception as e:
            self.logger.error(f"[参数异常] {e}, fn_id={fn_id}, param_vec={param_vec}, 使用 no_op 回退")
            return [], True

    def step(self, action) -> tuple:
        assert self._env is not None, "Env not initialized; call reset first"

        fn_id, raw_params = self._unwrap_action(action)
        # 用self.timestep获取当前观测
        obs_raw = self.timestep.observation
        parsed_now = self._parser.parse(obs_raw)
        parsed_flat_now = self._flatten_parsed(parsed_now)
        action_mask = parsed_flat_now.get("available_actions")
        if action_mask is None:
            self.logger.error("available_actions missing; cannot determine available actions")
            raise RuntimeError("available_actions missing; cannot determine available actions")
        available = np.nonzero(np.asarray(action_mask).reshape(-1))[0].tolist()

        args, clipped = self._sanitize_params(fn_id, raw_params)

        try:
            act = self.actions.FUNCTIONS[fn_id](*args) if args else self.actions.FUNCTIONS[fn_id]()
            exec_fn_id = fn_id
            exec_args = args
        except Exception as e:
            act = self.actions.FUNCTIONS.no_op()
            clipped = True
            exec_fn_id = self.actions.FUNCTIONS.no_op.id
            exec_args = []
            self.logger.error(f"[执行动作异常] {e}, fn_id={fn_id}, args={args}, 使用 no_op 回退")

        timesteps = self._env.step([act])
        self.timestep = timesteps[0]  # 更新当前timestep
        self._step_count += 1

        obs = self._obs_to_space(self.timestep)
        native_reward = float(self.timestep.reward)
        reward = native_reward
        terminated = bool(self.timestep.last())
        truncated = False
        info = {"arg_clipped": clipped, "fn_available": fn_id in available, "native_reward": native_reward}

        if self.reward_mode == "custom" and self._reward_fn is not None:
            reward = float(self._reward_fn(obs, self._reward_state, info))
            info["custom_reward"] = reward
        
        self.logger.debug(f"Step {self._step_count}")
        self.logger.debug(f"  Agent raw action: fn_id={fn_id}, raw_params={raw_params}")
        self.logger.debug(f"  Executed action: fn_id={exec_fn_id}, args={exec_args}")
        self.logger.debug(f"  Available actions: {len(available)}, reward={reward:.3f}, terminated={terminated}, clipped={clipped}")
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
