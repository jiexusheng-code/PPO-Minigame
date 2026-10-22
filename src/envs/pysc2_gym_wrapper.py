"""PySC2 -> Gymnasium minigame wrapper, 兼容精简ObsParser输出。"""
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from src.obs_parser import ObsParser
from pysc2.lib import features


class PySC2GymEnv(gym.Env):
    """Generic PySC2 wrapper compatible with Gymnasium."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_name: str = None,
        screen_size: int = 32,
        minimap_size: int = 32,
        step_mul: int = 8,
        visualize: bool = False,
        reset_jitter_max_steps: int = 0,
        random_seed: Optional[int] = None,
    ):
        import logging, os, datetime
        super().__init__()
        self.map_name = map_name
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.step_mul = step_mul
        self.visualize = visualize
        self.reset_jitter_max_steps = max(0, int(reset_jitter_max_steps))
        # Seed used to construct SC2Env. train.py passes seed+rank per env;
        # reset(seed=...) remains authoritative if it differs.
        self.random_seed = None if random_seed is None else int(random_seed)
        self._sc2_random_seed = self.random_seed
        self._step_count = 0  # 记录步数
        # whether we've already applied the one-time initial jitter
        self._init_jitter_done = False
        # Episode bookkeeping for the CollectMineralShards trajectory sampler.
        # The constructor's probe reset bypasses reset(), so the first real
        # reset() starts episode 1.
        self._episode_id = 0
        self._episode_step = 0
        self._episode_seed_source = "constructor" if random_seed is not None else "unknown"
        # train.py exposes the VecEnv rank before constructing each env. Only
        # rank 0 builds the trajectory payload, so the other workers do not pay
        # the raster/BFS cost for data that is never persisted.
        try:
            self._vec_env_rank = int(os.environ.get("VEC_ENV_RANK", 0))
        except (TypeError, ValueError):
            self._vec_env_rank = 0

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
        self._lazy_init()
        self._build_action_space()
        initial_ts = self._env.reset()
        self.timestep = initial_ts[0]
        parsed = self._parser.parse(self.timestep.observation)
        parsed_flat = self._flatten_parsed(parsed)
        self._build_observation_space_from_parser(parsed_flat)

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
            # write a one-time snapshot of param semantics to disk for cross-checking
            try:
                import json, os
                snapshot = {
                    'param_semantics': self._param_semantics,
                    'param_size_dict': self._param_size_dict,
                }
                # Parallel envs all share TRAIN_OUT_DIR; only rank 0 may write,
                # otherwise concurrent workers race on the same file.
                rank = int(os.environ.get('VEC_ENV_RANK', 0))
                if rank == 0:
                    out_dir = os.environ.get('TRAIN_OUT_DIR', None)
                    if out_dir is None:
                        out_dir = os.path.join(os.getcwd(), 'models')
                    os.makedirs(out_dir, exist_ok=True)
                    path = os.path.join(out_dir, 'param_semantics_env.json')
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(snapshot, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"Wrote param semantics snapshot to {path}")
            except Exception:
                pass
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
            random_seed=self._sc2_random_seed,
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

    def _apply_sc2_seed(self, seed: Optional[int]) -> None:
        """Rebuild SC2Env if reset(seed) disagrees with the construction seed.

        The constructor probe reset (used to build the spaces) must not make
        the later Gymnasium seed ineffective. When train.py already passes the
        same per-env seed at construction this is a no-op and no extra SC2
        launch occurs.
        """
        if seed is None or int(seed) == self._sc2_random_seed:
            return
        self.logger.info(
            f"reset(seed={seed}) 与构造 seed={self._sc2_random_seed} 不同，重建 SC2Env"
        )
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        self._sc2_random_seed = int(seed)
        self._lazy_init()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Gymnasium seeding: initializes self.np_random without touching the
        # process-global numpy RNG.
        super().reset(seed=seed)
        self._apply_sc2_seed(seed)
        self._lazy_init()
        timesteps = self._env.reset()
        self.timestep = timesteps[0]
        # A successful reset starts a new episode. The sampler only begins on
        # episode_step == 0, so this boundary is observable from the step info.
        self._episode_id = int(getattr(self, "_episode_id", 0)) + 1
        self._episode_step = 0
        if seed is not None:
            self._episode_seed_source = "reset"
        elif getattr(self, "_sc2_random_seed", None) is not None:
            self._episode_seed_source = "constructor"
        else:
            self._episode_seed_source = "unknown"

        # Optional one-time initial jitter: execute random no-op steps only on first reset
        if self.reset_jitter_max_steps > 0 and not getattr(self, '_init_jitter_done', False):
            jitter_steps = int(self.np_random.integers(0, self.reset_jitter_max_steps + 1))
            for _ in range(jitter_steps):
                try:
                    no_op = self.actions.FUNCTIONS.no_op()
                    timesteps = self._env.step([no_op])
                    self.timestep = timesteps[0]
                    if bool(self.timestep.last()):
                        timesteps = self._env.reset()
                        self.timestep = timesteps[0]
                except Exception:
                    break
            # ensure we don't repeat initial jitter on subsequent resets
            try:
                self._init_jitter_done = True
            except Exception:
                pass

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

    def _sanitize_params(self, fn_id: int, param_vec: list) -> tuple[list, bool, bool]:
        """Return ``(args, failed, bounds_violation)``.

        ``failed`` is the legacy "invalid or unusable" flag. ``bounds_violation``
        is True only when an argument value was outside its declared range; it
        distinguishes real bounds problems from unavailable functions, missing
        slots, type errors and low-level step failures. The sanitizer rejects
        invalid values rather than silently clipping them.
        """
        bounds_violation = False
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
                        bounds_violation = True
                        raise ValueError(f"参数越界: fn_id={fn_id}, 参数{name}, 槽位={slot}, raw_val={raw_val}")
                    y, x = divmod(raw_int, w)
                    args.append([x, y])
                else:
                    size = int(sizes[0]) if len(sizes) > 0 else 1
                    if not (0 <= raw_int < size):
                        bounds_violation = True
                        raise ValueError(f"参数越界: fn_id={fn_id}, 参数{name}, 槽位={slot}, raw_val={raw_val}")
                    args.append([raw_int])
            return args, False, False
        except Exception as e:
            self.logger.error(f"[参数异常] {e}, fn_id={fn_id}, param_vec={param_vec}, 使用 no_op 回退")
            return [], True, bounds_violation

    def _step_with_no_op(self, reason: str, fn_id, args) -> tuple:
        """Execute ``no_op`` as the safe fallback.

        Returns ``(timesteps, executed_fn_id, executed_args)``. If even the
        no_op cannot be executed, raise a clear ``RuntimeError`` instead of
        swallowing the failure and leaving the environment out of sync.
        """
        self.logger.error(
            f"[执行动作回退] {reason}, fn_id={fn_id}, args={args}, 使用 no_op"
        )
        try:
            no_op = self.actions.FUNCTIONS.no_op()
            timesteps = self._env.step([no_op])
            return timesteps, self.actions.FUNCTIONS.no_op.id, []
        except Exception as exc:
            raise RuntimeError(
                f"动作执行失败且 no_op 回退也失败: fn_id={fn_id}, args={args}, "
                f"原因={reason}; no_op 错误: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _flat_spatial_to_xy(value, parsed, name):
        """Decode a flattened spatial action index into ``(x, y)`` or None."""
        arr = parsed.get(name) if isinstance(parsed, dict) else None
        if arr is None:
            return None
        shape = np.asarray(arr).shape
        if len(shape) < 2:
            return None
        height, width = int(shape[0]), int(shape[1])
        try:
            flat = int(value)
        except (TypeError, ValueError):
            return None
        if not (0 <= flat < height * width):
            return None
        y, x = divmod(flat, width)
        return x, y

    def _target_cell_from_args(self, arg_names, arg_values, parsed, spatial_name="screen"):
        """Return the ``(y, x)`` screen cell used by an action, or None.

        Raw policy values are flattened indices; sanitized/executed PySC2
        values are ``[x, y]``. Only the requested ``screen`` argument is
        mapped to a screen cell; minimap/screen2 targets are deliberately
        excluded because the raster diagnostics live on the screen map.
        """
        for name, value in zip(arg_names, arg_values):
            if name != spatial_name:
                continue
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) >= 2:
                    return (int(value[1]), int(value[0]))
                return None
            xy = self._flat_spatial_to_xy(value, parsed, spatial_name)
            if xy is None:
                return None
            x, y = xy
            return (y, x)
        return None

    @staticmethod
    def _fallback_reason_code(reason):
        if not reason:
            return "none"
        if reason.startswith("参数清洗失败"):
            return "sanitize_failed"
        if reason.startswith("fn_id 不在"):
            return "fn_unavailable"
        if reason.startswith("构造动作异常"):
            return "construct_error"
        if reason.startswith("底层 step 异常"):
            return "step_error"
        return "other"

    def _collect_mineral_shards_diagnostics(
        self, before, after, requested_target_cell=None, executed_target_cell=None
    ):
        """Return compact spatial diagnostics for CollectMineralShards only.

        ``requested_target_cell`` and ``executed_target_cell`` are ``(y, x)``
        screen cells. The executed distance is emitted only when an executed
        screen target exists, so no_op/fallback never reuses the requested
        target. Both distances are measured against the pre-step raster, the
        state the action was chosen from.
        """
        def screen_metrics(parsed, target_cell=None):
            screen = np.asarray(parsed.get("screen"))
            if screen.ndim != 3 or screen.shape[-1] < 2:
                return 0, float("nan"), float("nan")
            # Canonical ObsParser channels: player_relative is channel 1.
            relative = screen[..., 1]
            neutral_pixels = np.argwhere(relative == 3)
            marine = np.argwhere(relative == 1)
            if neutral_pixels.size == 0:
                return 0, float("nan"), float("nan")
            remaining = {tuple(point) for point in neutral_pixels.tolist()}
            shard_centers = []
            while remaining:
                stack = [remaining.pop()]
                component = []
                while stack:
                    y, x = stack.pop()
                    component.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            neighbor = (y + dy, x + dx)
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                stack.append(neighbor)
                shard_centers.append(np.mean(component, axis=0))
            neutral = np.asarray(shard_centers, dtype=np.float32)
            target_distance = float("nan")
            if target_cell is not None:
                target = np.asarray(target_cell, dtype=np.float32).reshape(1, 2)
                target_distance = float(np.sqrt(((neutral - target) ** 2).sum(axis=1)).min())
            marine_distance = float(np.sqrt(((marine[:, None, :] - neutral[None, :, :]) ** 2).sum(axis=2)).min(axis=1).mean()) if marine.size else float("nan")
            return len(shard_centers), target_distance, marine_distance

        before_count, requested_distance, marine_distance = screen_metrics(before, requested_target_cell)
        after_count, _, _ = screen_metrics(after)
        out = {
            "collect_mineral_shards/raster_neutral_component_count_estimate_before": before_count,
            "collect_mineral_shards/raster_neutral_component_count_estimate_after": after_count,
            "collect_mineral_shards/raster_marine_to_neutral_component_distance_estimate_before_cells": marine_distance,
        }
        if requested_target_cell is not None:
            out["collect_mineral_shards/raster_requested_target_to_neutral_component_distance_estimate_cells"] = requested_distance
        if executed_target_cell is not None:
            _, executed_distance, _ = screen_metrics(before, executed_target_cell)
            out["collect_mineral_shards/raster_executed_target_to_neutral_component_distance_estimate_cells"] = executed_distance
        return out

    @staticmethod
    def _finite_or_none(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _canonical_screen_channels(self):
        """Return ``(player_relative, selected)`` screen channel indices.

        ObsParser emits the canonical screen layers in this order:
        visibility_map, player_relative, unit_type, selected, ... . Resolve
        the indices from the parser config instead of guessing, so a future
        layer-order change cannot silently read the wrong raster.
        """
        names = []
        try:
            names = [name for _, name in self._parser.config.screen_layers]
        except Exception:
            names = []

        def _index(name, default):
            try:
                return names.index(name)
            except ValueError:
                return default

        return _index("player_relative", 1), _index("selected", 3)

    @staticmethod
    def _raster_unit_estimates(screen, relative_ch, selected_ch):
        """Estimate unit/shard locations from canonical screen rasters.

        These are connected-component centroid estimates on the downsampled
        screen raster, not exact unit entities. ``player_relative`` values are
        1 for self and 3 for neutral; ``selected`` is the categorical
        selection layer (1 when selected).
        """
        screen = np.asarray(screen)
        empty = {"marines_raster_estimate": [], "neutral_shards_raster_estimate": []}
        if screen.ndim != 3 or screen.shape[-1] <= max(relative_ch, selected_ch):
            return empty
        relative = screen[..., relative_ch]
        selected = screen[..., selected_ch] > 0.5

        def components(mask):
            remaining = {tuple(point) for point in np.argwhere(mask).tolist()}
            found = []
            while remaining:
                stack = [remaining.pop()]
                pixels = []
                while stack:
                    y, x = stack.pop()
                    pixels.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            neighbor = (y + dy, x + dx)
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                stack.append(neighbor)
                ys = [point[0] for point in pixels]
                xs = [point[1] for point in pixels]
                found.append({
                    "centroid_y_raster_estimate": float(np.mean(ys)),
                    "centroid_x_raster_estimate": float(np.mean(xs)),
                    "pixel_count": len(pixels),
                    "pixels": pixels,
                })
            found.sort(
                key=lambda item: (
                    item["centroid_y_raster_estimate"],
                    item["centroid_x_raster_estimate"],
                )
            )
            return found

        marines = []
        for component in components(relative == 1):
            marines.append({
                "centroid_y_raster_estimate": component["centroid_y_raster_estimate"],
                "centroid_x_raster_estimate": component["centroid_x_raster_estimate"],
                "pixel_count": component["pixel_count"],
                "selected_raster_estimate": bool(
                    any(selected[y, x] for y, x in component["pixels"])
                ),
            })
        shards = [
            {
                "centroid_y_raster_estimate": component["centroid_y_raster_estimate"],
                "centroid_x_raster_estimate": component["centroid_x_raster_estimate"],
                "pixel_count": component["pixel_count"],
            }
            for component in components(relative == 3)
        ]
        return {
            "marines_raster_estimate": marines,
            "neutral_shards_raster_estimate": shards,
        }

    @staticmethod
    def _trajectory_stage_payload(info, stage, pairs, target_cell):
        """Compact requested/sanitized/executed stage payload.

        ``queued_flag`` is the FunctionCall ``queued`` argument (a command
        flag), never a real order-queue length.
        """
        prefix = "collect_mineral_shards/"

        def _value(value):
            if isinstance(value, (list, tuple, np.ndarray)):
                return [int(item) for item in value]
            return int(value)

        queued = None
        for name, value in pairs:
            if name == "queued":
                queued = _value(value)
                break
        return {
            "fn_id": info.get(prefix + stage + "_fn_id"),
            "fn_name": info.get(prefix + stage + "_fn_name"),
            "args": [[name, _value(value)] for name, value in pairs],
            "queued_flag": queued,
            "screen_xy": (
                None if target_cell is None else [int(target_cell[1]), int(target_cell[0])]
            ),
        }

    def _collect_mineral_shards_trajectory(
        self,
        info,
        parsed_before,
        parsed_after,
        reward,
        terminated,
        truncated,
        available,
        requested_pairs,
        sanitized_pairs,
        executed_pairs,
        requested_target_cell,
        sanitized_target_cell,
        executed_target_cell,
        episode_step,
    ):
        """Build one compact, JSON-safe step payload for trajectory sampling."""
        prefix = "collect_mineral_shards/"
        relative_ch, selected_ch = self._canonical_screen_channels()

        def _raster(parsed):
            screen = parsed.get("screen") if isinstance(parsed, dict) else None
            if screen is None:
                return {"marines_raster_estimate": [], "neutral_shards_raster_estimate": []}
            return self._raster_unit_estimates(screen, relative_ch, selected_ch)

        def _distance(key):
            return self._finite_or_none(info.get(prefix + key))

        return {
            "map": getattr(self, "map_name", None),
            "episode_id": int(getattr(self, "_episode_id", 0)),
            "episode_step": int(episode_step),
            "seed": getattr(self, "_sc2_random_seed", None),
            "seed_source": getattr(self, "_episode_seed_source", "unknown"),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": bool(terminated),
            "available_fn_ids": [int(fn) for fn in available],
            "requested": self._trajectory_stage_payload(
                info, "requested", requested_pairs, requested_target_cell
            ),
            "sanitized": self._trajectory_stage_payload(
                info, "sanitized", sanitized_pairs, sanitized_target_cell
            ),
            "executed": self._trajectory_stage_payload(
                info, "executed", executed_pairs, executed_target_cell
            ),
            "fallback": bool(info.get(prefix + "fallback", False)),
            "fallback_reason": info.get(prefix + "fallback_reason", ""),
            "fallback_reason_code": info.get(prefix + "fallback_reason_code", "none"),
            "sanitization_failed": bool(info.get(prefix + "sanitization_failed", False)),
            "args_bounds_violation": bool(info.get(prefix + "args_bounds_violation", False)),
            "neutral_component_count_raster_estimate": {
                "before": info.get(prefix + "raster_neutral_component_count_estimate_before"),
                "after": info.get(prefix + "raster_neutral_component_count_estimate_after"),
            },
            "target_distance_cells_raster_estimate": {
                "requested_to_neutral": _distance(
                    "raster_requested_target_to_neutral_component_distance_estimate_cells"
                ),
                "executed_to_neutral": _distance(
                    "raster_executed_target_to_neutral_component_distance_estimate_cells"
                ),
                "marine_to_neutral_before": _distance(
                    "raster_marine_to_neutral_component_distance_estimate_before_cells"
                ),
            },
            # ``pre`` is the raster the action was chosen from; ``post`` is the
            # terminal/next timestep returned by the same step. Neither is
            # taken from a VecEnv auto-reset observation.
            "pre": _raster(parsed_before),
            "post": _raster(parsed_after),
        }

    def step(self, action) -> tuple:
        assert self._env is not None, "Env not initialized; call reset first"

        episode_step = int(getattr(self, "_episode_step", 0))
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

        args, clipped, bounds_violation = self._sanitize_params(fn_id, raw_params)
        sanitization_failed = clipped
        fallback_used = False

        # Build and execute inside the safety net: PySC2 validates the argument
        # count only in SC2Env.step(), so a bad FunctionCall must never escape
        # before the no_op fallback gets a chance to run.
        act = None
        reason = None
        if clipped:
            reason = "参数清洗失败"
        elif fn_id is None or fn_id not in available:
            clipped = True
            reason = "fn_id 不在 available_actions 中"
        else:
            try:
                act = self.actions.FUNCTIONS[fn_id](*args) if args else self.actions.FUNCTIONS[fn_id]()
                exec_fn_id = fn_id
                exec_args = args
            except Exception as exc:
                clipped = True
                reason = f"构造动作异常: {type(exc).__name__}: {exc}"

        if act is None:
            fallback_used = True
            timesteps, exec_fn_id, exec_args = self._step_with_no_op(reason, fn_id, args)
        else:
            try:
                timesteps = self._env.step([act])
            except Exception as exc:
                clipped = True
                fallback_used = True
                reason = f"底层 step 异常: {type(exc).__name__}: {exc}"
                timesteps, exec_fn_id, exec_args = self._step_with_no_op(reason, fn_id, args)

        self.timestep = timesteps[0]  # 更新当前timestep
        self._step_count += 1

        obs = self._obs_to_space(self.timestep)
        reward = float(self.timestep.reward)
        terminated = bool(self.timestep.last())
        # game_steps_per_episode=0: episodes end via timestep.last() only, so
        # there is no Gymnasium time-limit truncation to report.
        truncated = False
        info = {
            "arg_clipped": clipped,
            "fn_available": fn_id in available,
            "executed_fn_id": exec_fn_id,
        }
        if getattr(self, "map_name", None) == "CollectMineralShards":
            def fn_name(fid):
                if fid is None or not 0 <= int(fid) < len(self.actions.FUNCTIONS):
                    return "invalid"
                return getattr(self.actions.FUNCTIONS[int(fid)], "name", "unknown")

            valid_fn = fn_id is not None and 0 <= int(fn_id) < len(self.actions.FUNCTIONS)
            selected_fn = self.actions.FUNCTIONS[int(fn_id)] if valid_fn else None
            selected_arg_names = [getattr(spec, "name", "arg") for spec in selected_fn.args] if selected_fn is not None else []
            selected_slots = self._fn_param_map.get(int(fn_id), []) if fn_id is not None else []
            selected_arg_values = [raw_params[index] for index in selected_slots if index < len(raw_params)]
            executed_fn = self.actions.FUNCTIONS[int(exec_fn_id)]
            executed_arg_names = [getattr(spec, "name", "arg") for spec in executed_fn.args]
            requested_pairs = tuple(
                (name, tuple(int(value) for value in values) if isinstance(values, (list, tuple, np.ndarray)) else int(values))
                for name, values in zip(selected_arg_names, selected_arg_values)
            )
            sanitized_pairs = tuple(
                (name, tuple(int(value) for value in values) if isinstance(values, (list, tuple, np.ndarray)) else int(values))
                for name, values in zip(selected_arg_names, args)
            )
            executed_pairs = tuple(
                (name, tuple(int(value) for value in values) if isinstance(values, (list, tuple, np.ndarray)) else int(values))
                for name, values in zip(executed_arg_names, exec_args)
            )
            sanitized_ok = not sanitization_failed
            sanitized_fn_id = int(fn_id) if sanitized_ok and fn_id is not None else -1
            sanitized_fn_name = fn_name(fn_id) if sanitized_ok and fn_id is not None else "invalid"
            reason_code = self._fallback_reason_code(reason)
            info.update({
                "collect_mineral_shards/requested_fn_id": -1 if fn_id is None else int(fn_id),
                "collect_mineral_shards/requested_fn_name": fn_name(fn_id),
                "collect_mineral_shards/requested_args": requested_pairs,
                "collect_mineral_shards/sanitized_fn_id": sanitized_fn_id,
                "collect_mineral_shards/sanitized_fn_name": sanitized_fn_name,
                "collect_mineral_shards/sanitized_args": sanitized_pairs if sanitized_ok else (),
                "collect_mineral_shards/selected_fn_id": -1 if fn_id is None else int(fn_id),
                "collect_mineral_shards/selected_fn_name": fn_name(fn_id),
                "collect_mineral_shards/selected_args": requested_pairs,
                "collect_mineral_shards/executed_fn_id": int(exec_fn_id),
                "collect_mineral_shards/executed_fn_name": fn_name(exec_fn_id),
                "collect_mineral_shards/executed_args": executed_pairs,
                "collect_mineral_shards/fn_available": bool(fn_id in available),
                # The sanitizer rejects invalid arguments instead of clipping
                # them, so args_modified is always False. The legacy generic
                # info["arg_clipped"] also covers unavailable functions and
                # step errors; label that honestly here.
                "collect_mineral_shards/args_modified": False,
                "collect_mineral_shards/args_bounds_violation": bool(bounds_violation),
                "collect_mineral_shards/sanitization_failed": bool(sanitization_failed),
                "collect_mineral_shards/legacy_invalid_or_fallback": bool(clipped),
                "collect_mineral_shards/fallback": bool(fallback_used),
                "collect_mineral_shards/fallback_reason": reason or "",
                "collect_mineral_shards/fallback_reason_code": reason_code,
                "collect_mineral_shards/available_fn_count": len(available),
            })
            for stage, pairs in (("requested", requested_pairs), ("sanitized", sanitized_pairs), ("executed", executed_pairs)):
                if stage == "sanitized" and not sanitized_ok:
                    continue
                for name, value in pairs:
                    if isinstance(value, tuple):
                        if len(value) == 1:
                            info[f"collect_mineral_shards/{stage}_{name}"] = value[0]
                        elif name in ("screen", "screen2", "minimap") and len(value) == 2:
                            info[f"collect_mineral_shards/{stage}_{name}_x"] = value[0]
                            info[f"collect_mineral_shards/{stage}_{name}_y"] = value[1]
                    else:
                        if name in ("screen", "screen2", "minimap"):
                            # Raw policy values are flat indices. Keep them
                            # explicitly labeled as flat and also emit decoded
                            # coordinates when the index is in range.
                            info[f"collect_mineral_shards/{stage}_{name}_flat"] = value
                            xy = self._flat_spatial_to_xy(value, parsed_flat_now, name)
                            if xy is not None:
                                info[f"collect_mineral_shards/{stage}_{name}_x"] = xy[0]
                                info[f"collect_mineral_shards/{stage}_{name}_y"] = xy[1]
                        else:
                            info[f"collect_mineral_shards/{stage}_{name}"] = value
            requested_target_cell = self._target_cell_from_args(
                selected_arg_names, selected_arg_values, parsed_flat_now, "screen"
            )
            executed_target_cell = None
            if not fallback_used and int(exec_fn_id) != int(self.actions.FUNCTIONS.no_op.id):
                executed_target_cell = self._target_cell_from_args(
                    executed_arg_names, exec_args, parsed_flat_now, "screen"
                )
            info.update(self._collect_mineral_shards_diagnostics(
                parsed_flat_now,
                obs,
                requested_target_cell=requested_target_cell,
                executed_target_cell=executed_target_cell,
            ))
            if int(getattr(self, "_vec_env_rank", 0)) == 0:
                sanitized_target_cell = None
                if sanitized_ok:
                    sanitized_target_cell = self._target_cell_from_args(
                        selected_arg_names, args, parsed_flat_now, "screen"
                    )
                info["collect_mineral_shards/trajectory"] = self._collect_mineral_shards_trajectory(
                    info=info,
                    parsed_before=parsed_flat_now,
                    parsed_after=obs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    available=available,
                    requested_pairs=requested_pairs,
                    sanitized_pairs=sanitized_pairs if sanitized_ok else (),
                    executed_pairs=executed_pairs,
                    requested_target_cell=requested_target_cell,
                    sanitized_target_cell=sanitized_target_cell,
                    executed_target_cell=executed_target_cell,
                    episode_step=episode_step,
                )

        self.logger.debug(f"Step {self._step_count}")
        self.logger.debug(f"  Agent raw action: fn_id={fn_id}, raw_params={raw_params}")
        self.logger.debug(f"  Executed action: fn_id={exec_fn_id}, args={exec_args}")
        self.logger.debug(f"  Available actions: {len(available)}, reward={reward:.3f}, terminated={terminated}, clipped={clipped}")
        self._episode_step = episode_step + 1
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render method for Gymnasium API compatibility.

        Returns None as rendering is handled by PySC2's visualize flag.
        """
        return None

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
