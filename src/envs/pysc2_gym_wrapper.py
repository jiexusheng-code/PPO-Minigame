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
        minimap_size: int = 64,
        step_mul: int = 8,
        visualize: bool = False,
    ):
        import logging, os, datetime
        super().__init__()
        self.map_name = map_name
        self.screen_size = screen_size
        self.minimap_size = minimap_size 
        self.step_mul = step_mul
        self.visualize = visualize
        self._step_count = 0  # 记录步数

        # 日志设置：只用主进程的logging.basicConfig，所有模块共用同一日志文件
        self.logger = logging.getLogger("PySC2GymEnv")

        try:
            from pysc2.env import sc2_env
            from pysc2.lib import actions, features
            from absl import flags
            if not flags.FLAGS.is_parsed():
                flags.FLAGS(["pysc2_env"])
        except Exception as exc:
            self.logger.error(f"无法导入 pysc2，请确保已正确安装。错误详情: {exc}")
            raise RuntimeError(f"无法导入 pysc2，请确保已正确安装。错误详情: {exc}") from exc

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
        try:
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
                        if any(k in name for k in ["screen", "screen2"]):
                            size = 84*84
                        elif "minimap" in name: 
                            size = 64*64
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
            ensure_available_actions=True
        )

    def _build_observation_space_from_parser(self, parsed: Dict[str, object]):
        spaces: Dict[str, gym.Space] = {}
        # spatial_fused
        spatial = parsed.get("spatial_fused")
        if spatial is not None:
            c, h, w = spatial.shape
            spaces["spatial_fused"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(c, h, w), dtype=np.float32)
        # scalar_vec（obs_parser输出直接是np.ndarray，flatten后key为scalar_vec）
        scalar_vec = parsed.get("scalar_vec")
        if scalar_vec is not None:
            spaces["scalar_vec"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=scalar_vec.shape, dtype=scalar_vec.dtype)
        # entities
        N = getattr(self._parser, "N_max", 512)
        type_ids = parsed.get("type_ids")
        if type_ids is not None:
            spaces["type_ids"] = gym.spaces.Box(low=0, high=1e6, shape=(N,), dtype=np.int64)
        owner_ids = parsed.get("owner_ids")
        if owner_ids is not None:
            spaces["owner_ids"] = gym.spaces.Box(low=0, high=10, shape=(N,), dtype=np.int64)
        ent_feats = parsed.get("ent_feats")
        if ent_feats is not None:
            spaces["ent_feats"] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=ent_feats.shape, dtype=np.float32)
        ent_mask = parsed.get("ent_mask")
        if ent_mask is not None:
            spaces["ent_mask"] = gym.spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)
        coords = parsed.get("coords")
        if coords is not None:
            spaces["coords"] = gym.spaces.Box(low=0, high=self.screen_size, shape=(N, 2), dtype=np.int32)
        # action_mask
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
        # flatten spatial_fused
        if "spatial_fused" in parsed:
            _add("spatial_fused", parsed["spatial_fused"], np.float32)
        # flatten scalar（obs_parser输出直接是np.ndarray，不是dict）
        if "scalar" in parsed and parsed["scalar"] is not None:
            _add("scalar_vec", parsed["scalar"], np.float32)
        # flatten entities
        entities = parsed.get("entities", {}) if isinstance(parsed.get("entities", {}), dict) else {}
        if entities:
            _add("type_ids", entities.get("type_ids"), np.int64)
            _add("owner_ids", entities.get("owner_ids"), np.int64)
            _add("ent_feats", entities.get("ent_feats"), np.float32)
            _add("ent_mask", entities.get("ent_mask"), np.float32)
            _add("coords", entities.get("coords"), np.int32)
        # flatten action_mask
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

                if "minimap" in name:
                    max_flat = 64 * 64
                    if not (0 <= raw_int < max_flat):
                        raise ValueError(f"参数越界: fn_id={fn_id}, 参数{name}, 槽位={slot}, raw_val={raw_val}")
                    y, x = divmod(raw_int, 64)
                    args.append([x, y])
                elif "screen" in name or "screen2" in name:
                    max_flat = 84 * 84
                    if not (0 <= raw_int < max_flat):
                        raise ValueError(f"参数越界: fn_id={fn_id}, 参数{name}, 槽位={slot}, raw_val={raw_val}")
                    y, x = divmod(raw_int, 84)
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
        parsed_now = self._parser.process_observation(obs_raw)
        parsed_flat_now = self._flatten_parsed(parsed_now)
        action_mask = parsed_flat_now.get("action_mask")
        if action_mask is None:
            logger.error("action_mask missing; cannot determine available actions")
            raise RuntimeError("action_mask missing; cannot determine available actions")
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
        reward = float(self.timestep.reward)
        terminated = bool(self.timestep.last())
        truncated = False
        info = {"arg_clipped": clipped, "fn_available": fn_id in available}
        
        self.logger.info(f"Step {self._step_count}")
        self.logger.info(f"  Agent raw action: fn_id={fn_id}, raw_params={raw_params}")
        self.logger.info(f"  Executed action: fn_id={exec_fn_id}, args={exec_args}")
        self.logger.info(f"  Available actions: {len(available)}, reward={reward:.3f}, terminated={terminated}, clipped={clipped}")
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
