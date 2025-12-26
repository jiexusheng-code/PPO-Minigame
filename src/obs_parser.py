"""
这份环境解析器是针对PYSC2中的minigame环境的，故在选取环境信息时做了简化处理。
"""

import numpy as np
import cv2
import time
import logging
from typing import Dict, Optional, Any

from pysc2.lib import actions as sc2_actions
from pysc2.lib import units as sc2_units
from pysc2.lib import upgrades as sc2_upgrades
from pysc2 import maps as sc2_maps

# 日志配置：与主程序保持一致，归档到统一目录
logger = logging.getLogger("rl.obs_parser")

def _default_unit_type_vocab() -> Dict[int, int]:
    if sc2_units is None:
        return {}
    vocab: Dict[int, int] = {}
    idx = 1
    for race_name in ("Neutral", "Protoss", "Terran", "Zerg"):
        enum_cls = getattr(sc2_units, race_name, None)
        if enum_cls is None:
            continue
        for entry in enum_cls:
            vocab[int(entry.value)] = idx
            idx += 1
    return vocab


def _default_upgrade_vocab() -> Dict[int, int]:
    if sc2_upgrades is None:
        return {}
    return {int(up.value): i for i, up in enumerate(sc2_upgrades.Upgrades, start=1)}


def _default_map_name_vocab() -> Dict[str, int]:
    if sc2_maps is None:
        return {}
    try:
        available = sc2_maps.get_maps()
    except Exception:  # pragma: no cover - map discovery may fail in some installs
        return {}
    return {name: idx for idx, name in enumerate(sorted(available.keys()), start=1)}


def _default_action_vocab_size() -> int:
    """使用 PySC2 的 FUNCTIONS（非 RAW_FUNCTIONS）动作表。"""
    if sc2_actions is None:
        return 0
    return len(getattr(sc2_actions, "FUNCTIONS", []))


class ObsParser:
    def __init__(
        self,
        unit_type_vocab: Optional[dict] = None,
        map_name_vocab: Optional[dict] = None,
        upgrade_vocab: Optional[dict] = None,
        action_vocab_size: Optional[int] = None,
        H: int = 64,
        W: int = 64,
        N_max: int = 512,
    ):
        """初始化ObsParser"""
        self.unit_type_vocab = (
            unit_type_vocab if unit_type_vocab is not None else _default_unit_type_vocab()
        )
        self.map_name_vocab = (
            map_name_vocab if map_name_vocab is not None else _default_map_name_vocab()
        )
        self.upgrade_vocab = (
            upgrade_vocab if upgrade_vocab is not None else _default_upgrade_vocab()
        )
        self.action_vocab_size = (
            action_vocab_size if action_vocab_size is not None else _default_action_vocab_size()
        )
        self.H = H
        self.W = W
        self.N_max = N_max

        # 仅使用CPU，无需GPU相关逻辑
        logger.info("ObsParser initialized (CPU only)")

    def parse_entities(self, obs: Any) -> dict:
        """
        Process raw entity list into structured tensors:
          - type_ids: [N_max]
          - owner_ids: [N_max]
          - ent_feats: [N_max, D_num]
          - ent_mask: [N_max]
          - coords: [N_max, 2]
        """
        if isinstance(obs, dict):
            if "feature_units" not in obs:
                raise RuntimeError("observation dict missing feature_units")
            units = obs["feature_units"]
        elif hasattr(obs, "feature_units"):
            units = getattr(obs, "feature_units")
        elif hasattr(obs, "raw_data") and getattr(obs, "raw_data") is not None and hasattr(getattr(obs, "raw_data"), "units"):
            units = getattr(getattr(obs, "raw_data"), "units")
        else:
            raise RuntimeError(f"unexpected observation type for entities: {type(obs)}; no feature_units/raw_data.units")
        N = min(len(units), self.N_max)

        type_ids = np.zeros(self.N_max, dtype=np.int64)
        owner_ids = np.zeros(self.N_max, dtype=np.int64)
        hp_norm = np.zeros(self.N_max, dtype=np.float32)
        energy_norm = np.zeros(self.N_max, dtype=np.float32)
        build_p = np.zeros(self.N_max, dtype=np.float32)
        visible = np.zeros(self.N_max, dtype=np.float32)
        sel_flag = np.zeros(self.N_max, dtype=np.float32)
        xs = np.zeros(self.N_max, dtype=np.float32)
        ys = np.zeros(self.N_max, dtype=np.float32)
        ent_mask = np.zeros(self.N_max, dtype=np.float32)

        for i, u in enumerate(units[:N]):
            type_ids[i] = self.unit_type_vocab.get(u.unit_type, 0)
            owner_ids[i] = getattr(u, "alliance", 0)
            hp = getattr(u, "health", 0.0)
            hp_max = max(getattr(u, "health_max", 1.0), 1.0)
            hp_norm[i] = hp / hp_max
            energy = getattr(u, "energy", 0.0)
            energy_max = max(getattr(u, "energy_max", 1.0), 1.0)
            energy_norm[i] = energy / energy_max
            build_p[i] = getattr(u, "build_progress", 0.0)
            visible[i] = float(getattr(u, "is_visible", True))
            sel_flag[i] = float(getattr(u, "is_selected", False))
            xs[i] = getattr(u, "x", 0)
            ys[i] = getattr(u, "y", 0)
            ent_mask[i] = 1.0

        gx = np.clip(
            (xs / (xs.max(initial=1) + 1e-6)) * (self.W - 1), 0, self.W - 1
        ).astype(np.int32)
        gy = np.clip(
            (ys / (ys.max(initial=1) + 1e-6)) * (self.H - 1), 0, self.H - 1
        ).astype(np.int32)

        ent_feats = np.stack([hp_norm, energy_norm, build_p, visible, sel_flag], axis=1)
        return {
            "type_ids": type_ids,
            "owner_ids": owner_ids,
            "ent_feats": ent_feats,
            "ent_mask": ent_mask,
            "coords": np.stack([gx, gy], axis=1),
        }


    @staticmethod
    def _grid_ratio(grid: Any) -> float:
        if grid is None:
            return 0.0
        try:
            arr = np.asarray(grid)
            if arr.size == 0:
                return 0.0
            positive = np.count_nonzero(arr)
            return float(positive) / float(arr.size)
        except Exception:
            return 0.0

    def _encode_upgrades(self, upgrades) -> np.ndarray:
        size = len(self.upgrade_vocab)
        vec = np.zeros(size, dtype=np.float16)
        if upgrades is None:
            return vec
        # 将 numpy array 转为 list，避免空 array 的 bool 语义错误
        try:
            upgrades_iter = upgrades.tolist() if hasattr(upgrades, "tolist") else list(upgrades)
        except TypeError:
            upgrades_iter = []
        for up in upgrades_iter:
            idx = self.upgrade_vocab.get(up)
            if idx is None:
                continue
            slot = max(0, int(idx) - 1)
            if slot < size:
                vec[slot] = 1.0
        return vec

    def _unit_summary(self, units) -> np.ndarray:
        total_units = max(1, len(units))
        orders_presence = orders_intensity = 0.0
        orders_units = 0.0
        harvester_gap_sum = harvester_units = 0.0
        weapon_cd_sum = weapon_cd_units = 0.0
        cargo_units = structure_units = incomplete_structures = 0.0
        for u in units:
            orders = getattr(u, "orders", None) or []
            if orders:
                orders_presence += 1.0
                orders_units += 1.0
                orders_intensity += min(len(orders), 3) / 3.0
            assigned = getattr(u, "assigned_harvesters", None)
            ideal = getattr(u, "ideal_harvesters", None)
            try:
                assigned_val = float(assigned if assigned is not None else 0.0)
                ideal_val = float(ideal if ideal is not None else 0.0)
            except Exception:
                assigned_val = ideal_val = 0.0
            if ideal_val > 0:
                gap = max(ideal_val - assigned_val, 0.0) / ideal_val
                harvester_gap_sum += gap
                harvester_units += 1.0
            weapon_cd = getattr(u, "weapon_cooldown", None)
            if weapon_cd is not None:
                try:
                    cd_norm = min(max(float(weapon_cd) / 50.0, 0.0), 1.0)
                except Exception:
                    cd_norm = 0.0
                weapon_cd_sum += cd_norm
                weapon_cd_units += 1.0
            if getattr(u, "cargo_space_taken", 0) not in (0, None):
                cargo_units += 1.0
            if bool(getattr(u, "is_structure", False)):
                structure_units += 1.0
                build_p = getattr(u, "build_progress", 1.0)
                try:
                    if float(build_p) < 1.0:
                        incomplete_structures += 1.0
                except Exception:
                    pass
        return np.array(
            [
                orders_presence / float(total_units),
                orders_intensity / max(1.0, float(orders_units)),
                harvester_gap_sum / max(1.0, float(harvester_units)),
                weapon_cd_sum / max(1.0, float(weapon_cd_units)),
                cargo_units / float(total_units),
                incomplete_structures / max(1.0, float(structure_units)),
            ],
            dtype=np.float16,
        )

    def _ui_summary(self, obs: Any) -> np.ndarray:
        def _get(path, default=None):
            cur = obs
            for key in path:
                if cur is None:
                    return default
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = getattr(cur, key, None)
            return cur if cur is not None else default

        single_select = _get(["single_select"], [])
        multi_select = _get(["multi_select"], [])
        control_groups = _get(["control_groups"], [])
        alerts = _get(["alerts"], [])
        build_queue = _get(["build_queue"], [])
        production_queue = _get(["production_queue"], [])
        cargo = _get(["cargo"], [])
        last_actions = _get(["last_actions"], [])
        filled_groups = 0
        for group in control_groups:
            try:
                if group and group[0] is not None and group[0] >= 0:
                    filled_groups += 1
            except Exception:
                continue
        cargo_count = 0
        for slot in cargo:
            try:
                cargo_count += len(slot)
            except TypeError:
                cargo_count += 1
        norm_units = max(1.0, float(self.N_max))
        norm_groups = max(1.0, float(len(control_groups) or 1))
        norm_actions = max(1.0, float(self.action_vocab_size - 1))
        last_action_ratio = 0.0
        try:
            if len(last_actions) > 0:
                last_action_ratio = float(last_actions[-1]) / norm_actions
        except Exception:
            last_action_ratio = 0.0
        return np.array(
            [
                len(single_select) / norm_units,
                len(multi_select) / norm_units,
                len(control_groups) / 10.0,
                filled_groups / norm_groups,
                len(alerts) / 10.0,
                (len(build_queue) + len(production_queue)) / 10.0,
                cargo_count / norm_units,
                last_action_ratio,
            ],
            dtype=np.float16,
        )


    def _ensure_scalar_projection(self, input_dim: int, target_dim: int):
        need_new = (
            not hasattr(self, "scalar_projection")
            or getattr(self, "_scalar_proj_in_dim", None) != input_dim
            or getattr(self, "_scalar_proj_dim", None) != target_dim
        )
        if need_new:
            projection_np = np.random.randn(input_dim, target_dim).astype(np.float32)
            q_np, _ = np.linalg.qr(projection_np)
            self.scalar_projection = (q_np * np.sqrt(1.0 / input_dim)).astype(np.float16)
            self._scalar_projection_backend = "cpu"
            self._scalar_proj_in_dim = input_dim
            self._scalar_proj_dim = target_dim
        return self.scalar_projection

    def parse_scalar(self, obs: dict) -> dict:
        """Process global scalar data into vector and mask"""
        def _get(path, default=None):
            cur = obs
            for key in path:
                if cur is None:
                    return default
                if isinstance(cur, dict):
                    cur = cur.get(key)
                else:
                    cur = getattr(cur, key, None)
            return cur if cur is not None else default

        map_name_val = _get(["map_name"], "")
        map_id = self.map_name_vocab.get(map_name_val, 0)

        upgrades_val = _get(["upgrades"], [])
        upgrades_vec = self._encode_upgrades(upgrades_val)

        player_val = _get(["player"], _get(["player_common"], []))
        player_arr = np.asarray(player_val, dtype=np.float16)

        score_val = _get(["score_cumulative"], _get(["score", "score_cumulative"], []))
        score_arr = np.log1p(np.asarray(score_val, dtype=np.float32)).astype(np.float16)

        loop_val = _get(["game_loop"], 0)
        loop_arr = np.array([loop_val], dtype=np.float16) / 1e4

        units_raw = _get(["feature_units"], _get(["raw_data", "units"], None))
        if units_raw is None:
            units = []
        else:
            try:
                units = units_raw.tolist() if hasattr(units_raw, "tolist") else list(units_raw)
            except TypeError:
                units = []

        unit_summary = self._unit_summary(units)
        ui_summary = self._ui_summary(obs)
        scalar_components = [
            np.array([map_id], dtype=np.float16),
            upgrades_vec,
            player_arr,
            score_arr,
            loop_arr,
            unit_summary,
            ui_summary,
        ]

        # 可选：如需详细调试可用logger.debug
        # logger.debug(f"Scalar vector components: {[comp.size for comp in scalar_components]}")

        scalar_vec = (
            np.concatenate([comp.flatten() for comp in scalar_components]).astype(np.float16)
        )
        # logger.debug(f"Original scalar_vec shape: {scalar_vec.shape}, dims: {scalar_vec.size}")

        # 添加线性映射到256维
        target_dim = 256
        input_dim = scalar_vec.size
        projection = self._ensure_scalar_projection(input_dim, target_dim)
        # logger.debug("=== 开始矩阵计算 ===")
        backend = getattr(self, "_scalar_projection_backend", "cpu")
        # 仅使用CPU进行矩阵乘法
        scalar_vec_f32 = scalar_vec.astype(np.float32)
        projection_f32 = self.scalar_projection.astype(np.float32)
        scalar_vec = np.dot(scalar_vec_f32, projection_f32).astype(np.float16)

        # logger.debug(f"Projected scalar_vec shape: {scalar_vec.shape}, dims: {scalar_vec.size}, range: {scalar_vec.min():.3f} to {scalar_vec.max():.3f}")

        scalar_mask = np.ones_like(scalar_vec, dtype=np.float16)
        return {"scalar_vec": scalar_vec.astype(np.float16), "scalar_mask": scalar_mask}

    def resize_spatial(self, S: np.ndarray) -> np.ndarray:
        """Resize spatial feature layers to (H, W)"""
        C, h0, w0 = S.shape
        out = np.zeros((C, self.H, self.W), dtype=S.dtype)
        for i in range(C):
            out[i] = cv2.resize(S[i], (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return out

    def scatter_connections(
        self, coords: np.ndarray, ent_embed: np.ndarray, mode: str = "sum"
    ) -> np.ndarray:
        """Scatter entity embeddings onto spatial grid channels"""
        # logger.debug(f"scatter_connections input shapes: coords={coords.shape}, ent_embed={ent_embed.shape}, coords min/max={coords.min()},{coords.max()}")

        C_e = ent_embed.shape[1]
        grid = np.zeros((C_e, self.H, self.W), dtype=np.float32)
        # logger.debug(f"grid shape: {grid.shape}")

        try:
            for i, (x, y) in enumerate(coords):
                if not (
                    isinstance(x, (int, np.integer))
                    and isinstance(y, (int, np.integer))
                ):
                    logger.warning(f"Non-integer coordinates at index {i}: x={x} ({type(x)}), y={y} ({type(y)})")
                    x, y = int(x), int(y)

                if 0 <= x < self.W and 0 <= y < self.H:
                    if mode == "sum":
                        grid[:, y, x] += ent_embed[i]
                    else:
                        grid[:, y, x] = np.maximum(grid[:, y, x], ent_embed[i])
                else:
                    logger.warning(f"Coordinates out of bounds at index {i}: x={x}, y={y}")

            grid_sum = (grid**2).sum(axis=(1, 2), keepdims=True)
            # logger.debug(f"grid_sum shape: {grid_sum.shape}, min/max: {grid_sum.min()}, {grid_sum.max()}")
            grid_norm = np.sqrt(grid_sum + 1e-8)
            grid = grid / grid_norm
            # logger.debug(f"Final grid shape: {grid.shape}")
            return grid
        except Exception as e:
            logger.error(f"Error in scatter_connections: {str(e)} at coords index {i if 'i' in locals() else '?'}")
            raise

    def _prepare_screen_layers(self, layers: Optional[np.ndarray]) -> np.ndarray:
        """Normalize screen layers to a fixed (3, H, W) tensor."""
        screen_target = 3
        if layers is None or np.size(layers) == 0:
            return np.zeros((screen_target, self.H, self.W), dtype=np.float32)
        layers = np.asarray(layers, dtype=np.float32)
        resized = self.resize_spatial(layers)
        if resized.size > 0:
            resized = (resized - resized.mean(axis=(1, 2), keepdims=True)) / (
                resized.std(axis=(1, 2), keepdims=True) + 1e-8
            )
        if resized.shape[0] < screen_target:
            pad = np.zeros((screen_target - resized.shape[0], self.H, self.W), dtype=resized.dtype)
            resized = np.concatenate([resized, pad], axis=0)
        elif resized.shape[0] > screen_target:
            resized = resized[:screen_target]
        return resized

    def create_entity_embedding(self, entity_dict: dict) -> np.ndarray:
        """将实体特征转换为嵌入向量"""
        N = self.N_max
        C_e = 16
        type_ids_norm = entity_dict["type_ids"].reshape(N, 1) / (
            max(self.unit_type_vocab.values()) + 1
        )
        owner_ids_norm = entity_dict["owner_ids"].reshape(N, 1) / 4.0
        ent_feats = entity_dict["ent_feats"]
        combined_features = np.concatenate(
            [type_ids_norm, owner_ids_norm, ent_feats], axis=1
        )
        W = np.random.randn(combined_features.shape[1], C_e) / np.sqrt(
            combined_features.shape[1]
        )
        combined_features = combined_features.astype(np.float16)
        W = W.astype(np.float16)
        # logger.debug(f"combined_features shape: {combined_features.shape}, W shape: {W.shape}")
        try:
            embeddings = np.matmul(combined_features, W)
        except Exception as e:
            logger.error(f"Error in create_entity_embedding: {str(e)}")
            raise
        mask = entity_dict["ent_mask"].reshape(N, 1)
        return embeddings * mask

    def process_observation(self, obs: object) -> dict:
        """
        Runs full pipeline for one observation (minigame精简版)
        只保留minigame相关观测字段，保证输出格式和形状不变。
        """
        # 兼容 dict 和对象属性两种访问方式，缺失直接报错
        def extract_attr(obs, key):
            # 1. 尝试 _asdict
            if hasattr(obs, "_asdict"):
                d = obs._asdict()
                if key in d:
                    return d[key]
            # 2. 尝试 __dict__
            if hasattr(obs, "__dict__"):
                d = vars(obs)
                if key in d:
                    return d[key]
            # 3. 尝试 keys()/__getitem__
            if hasattr(obs, "keys") and callable(obs.keys):
                try:
                    if key in obs.keys():
                        return obs[key]
                except Exception:
                    pass
            # 4. 尝试 getattr
            try:
                return getattr(obs, key)
            except Exception as e:
                logger.error(f"Failed to extract '{key}' from observation: {e}")
                logger.debug(f"extract_attr调试: type={type(obs)}, dir={dir(obs)}")
                try:
                    logger.debug(f"vars: {vars(obs)}")
                except Exception:
                    pass
                raise RuntimeError(f"Failed to extract '{key}' from observation. getattr error: {e}")

        obs_minigame = {
            "feature_units": extract_attr(obs, "feature_units"),
            "map_name": extract_attr(obs, "map_name"),
            "upgrades": extract_attr(obs, "upgrades"),
            "player": extract_attr(obs, "player"),
            "score_cumulative": extract_attr(obs, "score_cumulative"),
            "game_loop": extract_attr(obs, "game_loop"),
            "feature_screen": extract_attr(obs, "feature_screen"),
            "available_actions": extract_attr(obs, "available_actions"),
        }

        # 实体数据
        e = self.parse_entities(obs_minigame)
        # 标量数据
        s = self.parse_scalar(obs_minigame)
        # 空间融合数据（只用feature_screen）
        screen_layers = obs_minigame["feature_screen"]
        S_base = self._prepare_screen_layers(screen_layers)
        ent_embed = self.create_entity_embedding(e)
        E_grid = self.scatter_connections(e["coords"], ent_embed)
        vis = np.zeros((1, self.H, self.W), dtype=np.float32)
        for i, (x, y) in enumerate(e["coords"]):
            if e["ent_mask"][i] > 0:
                vis[0, y, x] = 1.0
        spatial_fused = np.concatenate([S_base, E_grid, vis], axis=0)

        # 动作掩码（基于 FUNCTIONS），供上层策略使用
        action_mask = np.zeros(self.action_vocab_size, dtype=np.float32)
        avail = obs_minigame["available_actions"]
        if avail is None:
            raise RuntimeError("observation missing available_actions; ensure feature action space is enabled")
        try:
            avail_iter = np.asarray(avail).flatten().tolist()
        except Exception:
            avail_iter = list(avail) if avail is not None else []
        for fn_id in avail_iter:
            try:
                idx = int(fn_id)
            except Exception:
                continue
            if 0 <= idx < self.action_vocab_size:
                action_mask[idx] = 1.0

        return {
            "entities": e,
            "scalar": s,
            "spatial_fused": spatial_fused,
            "action_mask": action_mask,
        }


if __name__ == "__main__":
    from absl import flags
    flags.FLAGS(['run'])
    parser = ObsParser()
    try:
        from pysc2.env import sc2_env
        from pysc2.lib import features
    except ImportError as exc:
        print(f"PySC2 import failed: {exc}. Install pysc2 and set SC2PATH to your game.")
        exit(1)

    def print_obs_fields(obs, label):
        print(f"\n--- {label} ---")
        print(f"type: {type(obs)}")
        print(f"dir: {dir(obs)}")
        try:
            print(f"vars: {vars(obs)}")
        except Exception:
            print("vars: <not available>")
        for key in [
            "feature_units", "map_name", "upgrades", "player", "score_cumulative", "game_loop", "feature_screen", "available_actions"
        ]:
            if isinstance(obs, dict):
                val = obs.get(key, None)
                print(f"dict: {key}: type={type(val)} value={str(val)[:80]}")
            else:
                val = getattr(obs, key, None)
                print(f"obj: {key}: type={type(val)} value={str(val)[:80]}")

    try:
        with sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=64, minimap=64),
                use_feature_units=True,
                action_space=sc2_env.ActionSpace.FEATURES,
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=False,
        ) as env:
            timestep = env.reset()[0]
            obs_live = timestep.observation
            print_obs_fields(obs_live, "obs_live (TimeStep.observation)")
            # 尝试转为dict
            if hasattr(obs_live, "_asdict"):
                obs_dict = obs_live._asdict()
                print_obs_fields(obs_dict, "obs_live._asdict()")
            # 用 process_observation 前后对比
            print("\n=== Running ObsParser.process_observation on minigame live data ===")
            out_live = parser.process_observation(obs_live)
            entities_live = out_live["entities"]
            scalar_live = out_live["scalar"]
            spatial_live = out_live["spatial_fused"]
            action_mask_live = out_live["action_mask"]

            print("entities ent_feats shape:", entities_live["ent_feats"].shape)
            print("scalar_vec shape:", scalar_live["scalar_vec"].shape)
            print("spatial_fused shape:", spatial_live.shape)
            print("action_mask nonzero count:", int(action_mask_live.sum()))
            print("action_mask indices set:", np.nonzero(action_mask_live)[0].tolist())
    except Exception as exc:
        print(f"Failed to collect or process live observation: {exc}")
        print("Make sure StarCraft II is installed and SC2PATH points to the game root.")
