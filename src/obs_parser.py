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
from pysc2 import maps as sc2_maps

# 日志配置：与主程序保持一致，归档到统一目录
logger = logging.getLogger("rl.obs_parser")

def _default_unit_type_vocab() -> Dict[int, int]:
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


def _default_map_name_vocab() -> Dict[str, int]:
    available = sc2_maps.get_maps()
    return {name: idx for idx, name in enumerate(sorted(available.keys()), start=1)}


def _default_action_vocab_size() -> int:
    """为了方便使用动作掩码，这里使用 PySC2 的 FUNCTIONS（非 RAW_FUNCTIONS）动作表。"""
    return len(getattr(sc2_actions, "FUNCTIONS", []))


class ObsParser:
    def __init__(
        self,
        unit_type_vocab: Optional[dict] = None,
        map_name_vocab: Optional[dict] = None,
        action_vocab_size: Optional[int] = None,
        H: int = 64,
        W: int = 64,
        N_max: int = 64,#针对minigame环境，实体数量不多，64足矣
    ):
        #几个表都使用默认排序，减少工作量
        self.unit_type_vocab = (
            unit_type_vocab if unit_type_vocab is not None else _default_unit_type_vocab()
        )
        self.map_name_vocab = (
            map_name_vocab if map_name_vocab is not None else _default_map_name_vocab()
        )
        self.action_vocab_size = (
            action_vocab_size if action_vocab_size is not None else _default_action_vocab_size()
        )
        self.H = H
        self.W = W
        self.N_max = N_max

        logger.info("ObsParser initialized successfully.")

    def parse_entities(self, obs: Any) -> dict:
        """
        Process raw entity list into structured tensors:
          - type_ids: [N_max]
          - owner_ids: [N_max]
          - ent_feats: [N_max, D_num]
          - ent_mask: [N_max]
          - coords: [N_max, 2]
        """
        try:
            units = obs.feature_units
        except Exception as e:
            logger.error(f"无法提取feature_units: {e}")
            raise RuntimeError(f"无法提取feature_units: {e}")
        N = min(len(units), self.N_max)

        type_ids = np.zeros(self.N_max, dtype=np.int64)
        owner_ids = np.zeros(self.N_max, dtype=np.int64)
        health_ratio = np.zeros(self.N_max, dtype=np.float32)
        build_progress = np.zeros(self.N_max, dtype=np.float32)
        facing = np.zeros(self.N_max, dtype=np.float32)
        radius = np.zeros(self.N_max, dtype=np.float32)
        mineral_contents = np.zeros(self.N_max, dtype=np.float32)
        vespene_contents = np.zeros(self.N_max, dtype=np.float32)
        assigned_harvesters = np.zeros(self.N_max, dtype=np.float32)
        ideal_harvesters = np.zeros(self.N_max, dtype=np.float32)
        weapon_cooldown = np.zeros(self.N_max, dtype=np.float32)
        sel_flag = np.zeros(self.N_max, dtype=np.float32)
        xs = np.zeros(self.N_max, dtype=np.float32)
        ys = np.zeros(self.N_max, dtype=np.float32)
        ent_mask = np.zeros(self.N_max, dtype=np.float32)
        owner_ids = np.zeros(self.N_max, dtype=np.int64)

        for i, u in enumerate(units[:N]):
            try:
                type_ids[i] = self.unit_type_vocab.get(getattr(u, "unit_type"), 0)
                owner_ids[i] = getattr(u, "alliance")
                health_ratio[i] = getattr(u, "health_ratio")
                build_progress[i] = getattr(u, "build_progress")
                facing[i] = getattr(u, "facing")
                radius[i] = getattr(u, "radius")
                mineral_contents[i] = getattr(u, "mineral_contents")
                vespene_contents[i] = getattr(u, "vespene_contents")
                assigned_harvesters[i] = getattr(u, "assigned_harvesters")
                ideal_harvesters[i] = getattr(u, "ideal_harvesters")
                weapon_cooldown[i] = getattr(u, "weapon_cooldown")
                sel_flag[i] = float(getattr(u, "is_selected"))
                xs[i] = getattr(u, "x")
                ys[i] = getattr(u, "y")
                ent_mask[i] = 1.0
            except Exception as e:
                logger.error(f"parse_entities特征提取失败: {e}")
                raise

        gx = np.clip(
            (xs / (xs.max(initial=1) + 1e-6)) * (self.W - 1), 0, self.W - 1
        ).astype(np.int32)
        gy = np.clip(
            (ys / (ys.max(initial=1) + 1e-6)) * (self.H - 1), 0, self.H - 1
        ).astype(np.int32)

        ent_feats = np.stack([
            health_ratio, build_progress, facing, radius,
            mineral_contents, vespene_contents,
            assigned_harvesters, ideal_harvesters, weapon_cooldown, sel_flag
        ], axis=1)
        return {
            "type_ids": type_ids,
            "owner_ids": owner_ids,
            "ent_feats": ent_feats,
            "ent_mask": ent_mask,
            "coords": np.stack([gx, gy], axis=1),
        }

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


    def parse_scalar(self, obs: Any) -> np.ndarray:
        """
        提取minigame中有意义的全局标量，输出定长向量。
        选取特征：
        - map_id（地图编号）
        - game_loop（归一化步数）
        - player资源与人口（minerals, vespene, food_used, food_cap, army_count, idle_worker_count）
        - score_cumulative（总分、采集、消灭等）
        - last_action（最近一次动作id，归一化）
        - available_actions数量
        - control_groups数量
        - alerts数量
        """
        try:
            # 地图编号
            map_id = float(self.map_name_vocab.get(obs.map_name))

            # 游戏步数归一化
            game_loop = float(np.asarray(obs.game_loop).flatten()[0])
            game_loop_norm = game_loop / 1e4

            # 玩家资源与人口
            player = np.asarray(obs.player).flatten()
            player_feats =  [float(player[i]) for i in range(1, 9)]

            score = np.asarray(obs.score_cumulative).flatten()
            score_feats = [
                float(score[0]) ,   # score
                float(score[5]) ,   # killed_value_units
                float(score[6]) ,   # killed_value_structures
                float(score[7]) ,   # collected_minerals
                float(score[8]) ,   # collected_vespene
                float(score[11]) , # spent_minerals
                float(score[12]) , # spent_vespene
            ]

            # last_action（最近一次动作id，归一化）
            last_actions = np.asarray(getattr(obs, "last_actions", []))
            if last_actions.size > 0:
                last_action = float(last_actions[-1]) / max(1.0, float(self.action_vocab_size - 1))
            else:
                last_action = 0.0

            # control_groups数量
            control_groups = np.asarray(getattr(obs, "control_groups", []))
            ctrl_group_count = float(control_groups.shape[0]) if control_groups.size > 0 else 0.0

            # alerts数量
            alerts = np.asarray(getattr(obs, "alerts", []))
            alerts_count = float(alerts.size)

            # 拼接所有特征
            scalar_vec = np.array(
                [map_id, game_loop_norm] + player_feats + score_feats +
                [last_action, ctrl_group_count, alerts_count],
                dtype=np.float32
            )
            return scalar_vec
        except Exception as e:
            logger.error(f"parse_scalar特征提取失败: {e}")
            raise

    def resize_spatial(self, S: np.ndarray) -> np.ndarray:
        """Resize spatial feature layers to (H, W)"""
        C, h0, w0 = S.shape
        out = np.zeros((C, self.H, self.W), dtype=S.dtype)
        for i in range(C):
            out[i] = cv2.resize(S[i], (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return out

    def scatter_connections(
        self, coords: np.ndarray, ent_feats: np.ndarray
    ) -> np.ndarray:
        """
        直接将每个实体的特征scatter到空间网格，每个特征为一个通道。
        ent_feats: [N, C_e]
        返回: [C_e, H, W]
        """
        C_e = ent_feats.shape[1]
        grid = np.zeros((C_e, self.H, self.W), dtype=np.float32)
        try:
            for i, (x, y) in enumerate(coords):
                if not (
                    isinstance(x, (int, np.integer))
                    and isinstance(y, (int, np.integer))
                ):
                    x, y = int(x), int(y)
                if 0 <= x < self.W and 0 <= y < self.H:
                    grid[:, y, x] += ent_feats[i]
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
        """
        直接将归一化后的实体特征拼接，作为每个实体的特征向量，无learnable参数。
        输出: [N_max, C_e]，C_e为特征数
        """
        N = self.N_max
        # 归一化 type_ids, owner_ids
        type_ids_norm = entity_dict["type_ids"].reshape(N, 1) / (max(self.unit_type_vocab.values()) + 1)
        owner_ids_norm = entity_dict["owner_ids"].reshape(N, 1) / 4.0
        ent_feats = entity_dict["ent_feats"]  # [N, D]
        # 拼接所有特征
        combined_features = np.concatenate([
            type_ids_norm, owner_ids_norm, ent_feats
        ], axis=1)  # [N, C_e]
        # mask无效实体
        mask = entity_dict["ent_mask"].reshape(N, 1)
        return combined_features * mask  # [N, C_e]

    def process_observation(self, obs: object) -> dict:
        """
        Runs full pipeline for one observation (minigame精简版)
        空间融合采用：实体特征直接scatter到空间网格（每特征一通道），与feature_screen和可视mask拼接。
        """
        # 实体数据
        e = self.parse_entities(obs)
        # 标量数据
        s = self.parse_scalar(obs)
        # 空间融合数据（只用feature_screen）
        screen_layers = obs.feature_screen
        S_base = self._prepare_screen_layers(screen_layers)
        ent_feats = self.create_entity_embedding(e)  # [N, C_e]
        E_grid = self.scatter_connections(e["coords"], ent_feats)  # [C_e, H, W]
        vis = np.zeros((1, self.H, self.W), dtype=np.float32)
        for i, (x, y) in enumerate(e["coords"]):
            if e["ent_mask"][i] > 0:
                vis[0, y, x] = 1.0
        spatial_fused = np.concatenate([S_base, E_grid, vis], axis=0)

        # 动作掩码（基于 FUNCTIONS），供上层策略使用
        action_mask = np.zeros(self.action_vocab_size, dtype=np.float32)
        avail = obs.available_actions
        if avail is None:
            logger.error("observation missing available_actions; ensure feature action space is enabled")
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
    try:
        from pysc2.env import sc2_env
        from pysc2.lib import features
    except ImportError as exc:
        print(f"PySC2 import failed: {exc}. 请安装 pysc2 并设置好 SC2PATH。")
        exit(1)

    parser = ObsParser()
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
            obs = timestep.observation
            result = parser.process_observation(obs)
            print("[ObsParser] process_observation 输出：")
            print("entities['ent_feats'] shape:", result["entities"]["ent_feats"].shape)
            print("entities['coords'] shape:", result["entities"]["coords"].shape)
            print("scalar shape:", result["scalar"].shape)
            print("spatial_fused shape:", result["spatial_fused"].shape)
            print("action_mask sum:", int(result["action_mask"].sum()))
            print("action_mask nonzero idx:", np.nonzero(result["action_mask"])[0][:10], "...")
            # 打印部分内容做 sanity check
            print("scalar (前10):", result["scalar"][:10])
            print("spatial_fused[0, :4, :4]:\n", result["spatial_fused"][0, :4, :4])
    except Exception as exc:
        print(f"[ObsParser] 测试失败: {exc}")
        print("请确保 StarCraft II 已安装且 SC2PATH 设置正确。")
