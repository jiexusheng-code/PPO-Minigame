import numpy as np
import cv2
import time
from typing import Dict, Optional, Any

DEBUG = False  # 调试开关


# GPU支持初始化
def init_gpu():
    try:
        import cupy as cp

        # 尝试初始化GPU并选择设备
        if cp.cuda.is_available():
            device = cp.cuda.get_device_id()
            cp.cuda.Device(device).use()
            if DEBUG:
                print(f"\nGPU computation enabled - Using CUDA device {device}")
                print(
                    f"Device name: {cp.cuda.runtime.getDeviceProperties(device)['name'].decode()}"
                )
            return cp, True
        else:
            if DEBUG:
                print("\nNo CUDA device available")
            return None, False
    except ImportError as e:
        if DEBUG:
            print(f"\n无法导入cupy: {str(e)}")
        return None, False
    except Exception as e:
        if DEBUG:
            print(f"\nGPU初始化失败: {str(e)}")
        return None, False


cp, USE_GPU = init_gpu()


from pysc2.lib import actions as sc2_actions
from pysc2.lib import units as sc2_units
from pysc2.lib import upgrades as sc2_upgrades
from pysc2 import maps as sc2_maps



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
        if not self.unit_type_vocab:
            raise ValueError("unit_type_vocab is empty; please provide a custom mapping or ensure pysc2 is available")
        if not self.map_name_vocab:
            raise ValueError("map_name_vocab is empty; please provide map mappings or ensure pysc2 maps can be discovered")
        if not self.upgrade_vocab:
            raise ValueError("upgrade_vocab is empty; please supply upgrade mappings or enable pysc2")
        if self.action_vocab_size <= 0:
            raise ValueError("action_vocab_size must be positive; provide a value or install pysc2 for RAW_FUNCTIONS")
        self.H = H
        self.W = W
        self.N_max = N_max

        # 用init_gpu()返回的cp和USE_GPU判断GPU状态，移除对CUPY的依赖
        self._gpu_initialized = USE_GPU and (cp is not None)
        if DEBUG:
            if self._gpu_initialized:
                print(f"ObsParser initialized with GPU support")
                print(f"Using CUDA device: {cp.cuda.get_device_id()}")
            else:
                print("ObsParser initialized without GPU support")

    def parse_entities(self, obs: dict) -> dict:
        """
        Process raw entity list into structured tensors:
          - type_ids: [N_max]
          - owner_ids: [N_max]
          - ent_feats: [N_max, D_num]
          - ent_mask: [N_max]
          - coords: [N_max, 2]
        """
        units = obs.get("feature_units", [])
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
        for up in upgrades or []:
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

    def _ui_summary(self, obs: dict) -> np.ndarray:
        single_select = obs.get("single_select", []) 
        multi_select = obs.get("multi_select", []) 
        control_groups = obs.get("control_groups", []) 
        alerts = obs.get("alerts", []) 
        build_queue = obs.get("build_queue", []) 
        production_queue = obs.get("production_queue", []) 
        cargo = obs.get("cargo", []) 
        last_actions = obs.get("last_actions", []) 
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
            if USE_GPU and self._gpu_initialized and cp is not None:
                try:
                    projection = cp.random.randn(input_dim, target_dim).astype(np.float32)
                    q, _ = cp.linalg.qr(projection)
                    self.scalar_projection = (q * cp.sqrt(1.0 / input_dim)).astype(np.float16)
                    self._scalar_projection_backend = "gpu"
                except Exception:
                    self._gpu_initialized = False
                    projection_np = np.random.randn(input_dim, target_dim).astype(np.float32)
                    q_np, _ = np.linalg.qr(projection_np)
                    self.scalar_projection = (q_np * np.sqrt(1.0 / input_dim)).astype(np.float16)
                    self._scalar_projection_backend = "cpu"
            else:
                projection_np = np.random.randn(input_dim, target_dim).astype(np.float32)
                q_np, _ = np.linalg.qr(projection_np)
                self.scalar_projection = (q_np * np.sqrt(1.0 / input_dim)).astype(np.float16)
                self._scalar_projection_backend = "cpu"
            self._scalar_proj_in_dim = input_dim
            self._scalar_proj_dim = target_dim
        return self.scalar_projection

    def parse_scalar(self, obs: dict) -> dict:
        """Process global scalar data into vector and mask"""
        map_id = self.map_name_vocab.get(obs.get("map_name", ""), 0)
        upgrades_vec = self._encode_upgrades(obs.get("upgrades", []))
        player_arr = np.asarray(obs.get("player", []), dtype=np.float16)
        score_arr = np.log1p(
            np.asarray(obs.get("score_cumulative", []), dtype=np.float32)
        ).astype(np.float16)
        loop_arr = np.array([obs.get("game_loop", 0)], dtype=np.float16) / 1e4
        units = obs.get("feature_units", [])

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

        if DEBUG:
            print("\nScalar vector components dimensions:")
            names = [
                "map_id",
                "upgrades_vec",
                "player_arr",
                "score_arr",
                "loop_arr",
                "unit_summary",
                "ui_summary",
                "effects_summary",
                "map_state_summary",
            ]
            for comp, name in zip(scalar_components, names):
                print(f"{name}: {comp.size} dims")

        scalar_vec = (
            np.concatenate([comp.flatten() for comp in scalar_components]).astype(np.float16)
        )
        if DEBUG:
            print(f"\nOriginal scalar_vec shape: {scalar_vec.shape}")
            print(f"Original dimensions: {scalar_vec.size}")

            # 检查GPU状态
            print(f"\nGPU Status:")
            print(f"USE_GPU: {USE_GPU}")
            print(
                f"_gpu_initialized: {hasattr(self, '_gpu_initialized')} {getattr(self, '_gpu_initialized', False)}"
            )

        # 添加线性映射到256维
        target_dim = 256
        input_dim = scalar_vec.size
        projection = self._ensure_scalar_projection(input_dim, target_dim)
        if DEBUG:
            print("\n=== 开始矩阵计算 ===")
        backend = getattr(self, "_scalar_projection_backend", "cpu")
        if USE_GPU and self._gpu_initialized and backend == "gpu" and cp is not None:
            if DEBUG:
                print("使用GPU进行计算...")
            try:
                if DEBUG:
                    print("\n正在进行GPU矩阵乘法...")
                    print(f"输入向量形状: {scalar_vec.shape}")
                    print(f"投影矩阵形状: {self.scalar_projection.shape}")

                    # 将数据转移到GPU
                    print("传输数据到GPU...")
                scalar_vec_gpu = cp.asarray(scalar_vec.astype(np.float32))
                projection_gpu = cp.asarray(self.scalar_projection.astype(np.float32))
                if DEBUG:
                    print(
                        f"数据已传输到GPU - 形状: {scalar_vec_gpu.shape}, {projection_gpu.shape}"
                    )

                # 在GPU上进行矩阵乘法并计时
                start_time = time.time()
                result_gpu = cp.dot(scalar_vec_gpu, projection_gpu)
                cp.cuda.stream.get_current_stream().synchronize()
                gpu_time = (time.time() - start_time) * 1000  # 转换为毫秒

                if DEBUG:
                    print(f"GPU计算完成 - 结果形状: {result_gpu.shape}")
                    print(f"GPU计算时间: {gpu_time:.2f} ms")

                    # 将结果传回CPU
                    print("正在将结果传回CPU...")
                scalar_vec = cp.asnumpy(result_gpu).astype(np.float16)

                # 清理GPU内存
                del scalar_vec_gpu, projection_gpu, result_gpu
                cp.get_default_memory_pool().free_all_blocks()
                if DEBUG:
                    print("GPU内存已清理")

                    print(f"\nGPU计算结果:")
                    print(f"最终向量形状: {scalar_vec.shape}")
                    print(f"数值范围: [{scalar_vec.min():.3f}, {scalar_vec.max():.3f}]")
            except Exception as e:
                # 修复重复except块的语法错误，只保留一个异常捕获
                if DEBUG:
                    print(f"GPU computation failed with error: {str(e)}")
                    print("Falling back to CPU computation...")
                # 回退到CPU计算
                scalar_vec_f32 = scalar_vec.astype(np.float32)
                if getattr(self, "_scalar_projection_backend", "cpu") == "gpu" and cp is not None:
                    projection_f32 = cp.asnumpy(self.scalar_projection).astype(np.float32)
                else:
                    projection_f32 = self.scalar_projection.astype(np.float32)
                scalar_vec = np.dot(scalar_vec_f32, projection_f32).astype(np.float16)
        else:
            # CPU计算
            if DEBUG:
                print("\nUsing CPU for matrix multiplication (GPU not available)")
            scalar_vec_f32 = scalar_vec.astype(np.float32)
            if getattr(self, "_scalar_projection_backend", "cpu") == "gpu" and cp is not None:
                projection_f32 = cp.asnumpy(self.scalar_projection).astype(np.float32)
            else:
                projection_f32 = self.scalar_projection.astype(np.float32)
            scalar_vec = np.dot(scalar_vec_f32, projection_f32).astype(np.float16)

        if DEBUG:
            print(f"Projected scalar_vec shape: {scalar_vec.shape}")
            print(f"Final dimensions: {scalar_vec.size}")
            print(f"Value range: {scalar_vec.min():.3f} to {scalar_vec.max():.3f}")

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
        if DEBUG:
            print(f"scatter_connections input shapes:")
            print(f"coords shape: {coords.shape}")
            print(f"ent_embed shape: {ent_embed.shape}")
            print(f"coords min/max: {coords.min()}, {coords.max()}")

        C_e = ent_embed.shape[1]
        grid = np.zeros((C_e, self.H, self.W), dtype=np.float32)
        if DEBUG:
            print(f"grid shape: {grid.shape}")

        try:
            for i, (x, y) in enumerate(coords):
                if not (
                    isinstance(x, (int, np.integer))
                    and isinstance(y, (int, np.integer))
                ):
                    if DEBUG:
                        print(
                            f"Warning: Non-integer coordinates at index {i}: x={x} ({type(x)}), y={y} ({type(y)})"
                        )
                    x, y = int(x), int(y)

                if 0 <= x < self.W and 0 <= y < self.H:
                    if mode == "sum":
                        grid[:, y, x] += ent_embed[i]
                    else:
                        grid[:, y, x] = np.maximum(grid[:, y, x], ent_embed[i])
                else:
                    if DEBUG:
                        print(
                            f"Warning: Coordinates out of bounds at index {i}: x={x}, y={y}"
                        )

            grid_sum = (grid**2).sum(axis=(1, 2), keepdims=True)
            if DEBUG:
                print(
                    f"grid_sum shape: {grid_sum.shape}, min/max: {grid_sum.min()}, {grid_sum.max()}"
                )

            grid_norm = np.sqrt(grid_sum + 1e-8)
            grid = grid / grid_norm

            if DEBUG:
                print(f"Final grid shape: {grid.shape}")
            return grid

        except Exception as e:
            if DEBUG:
                print(f"Error in scatter_connections: {str(e)}")
                print(f"Error at coords index {i} if available")
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
        if DEBUG:
            print(f"combined_features shape: {combined_features.shape}")
            print(f"W shape: {W.shape}")
        try:
            embeddings = np.matmul(combined_features, W)
        except Exception as e:
            if DEBUG:
                print(f"Error in create_entity_embedding: {str(e)}")
            raise
        mask = entity_dict["ent_mask"].reshape(N, 1)
        return embeddings * mask

    def process_observation(self, obs: dict) -> dict:
        """Runs full pipeline for one observation"""
        e = self.parse_entities(obs)
        s = self.parse_scalar(obs)

        S_base = self._prepare_screen_layers(obs.get("screen_layers"))
        ent_embed = self.create_entity_embedding(e)
        E_grid = self.scatter_connections(e["coords"], ent_embed)
        vis = np.zeros((1, self.H, self.W), dtype=np.float32)
        for i, (x, y) in enumerate(e["coords"]):
            if e["ent_mask"][i] > 0:
                vis[0, y, x] = 1.0
        spatial_fused = np.concatenate([S_base, E_grid, vis], axis=0)

        # 动作掩码（基于 FUNCTIONS），供上层策略使用
        action_mask = np.zeros(self.action_vocab_size, dtype=np.float32)
        for fn_id in obs.get("available_actions", []) or []:
            if 0 <= int(fn_id) < self.action_vocab_size:
                action_mask[int(fn_id)] = 1.0

        return {
            "entities": e,
            "scalar": s,
            "spatial_fused": spatial_fused,
            "action_mask": action_mask,
        }


if __name__ == "__main__":
    from types import SimpleNamespace

    parser = ObsParser()

    marine = SimpleNamespace(
        unit_type="Marine",
        alliance=1,
        health=35.0,
        health_max=45.0,
        energy=20.0,
        energy_max=50.0,
        build_progress=1.0,
        is_visible=True,
        is_selected=True,
        x=25,
        y=40,
        orders=[{"ability_id": 1}],
        assigned_harvesters=None,
        ideal_harvesters=None,
        weapon_cooldown=5.0,
        cargo_space_taken=0,
        is_structure=False,
    )
    depot = SimpleNamespace(
        unit_type="SupplyDepot",
        alliance=1,
        health=300.0,
        health_max=400.0,
        energy=0.0,
        energy_max=0.0,
        build_progress=0.5,
        is_visible=True,
        is_selected=False,
        x=80,
        y=120,
        orders=[],
        assigned_harvesters=0,
        ideal_harvesters=0,
        weapon_cooldown=None,
        cargo_space_taken=0,
        is_structure=True,
    )

    raw_obs = {
        "feature_units": [marine, depot],
        "map_name": "AbyssalReef",
        "upgrades": [101],
        "available_actions": [0, 2, 5],
        "player": [50, 0, 2, 500, 50, 10],
        "score_cumulative": [100, 200, 50],
        "game_loop": 1234,
        "single_select": [(0, 0)],
        "multi_select": [(0, 0), (1, 1)],
        "control_groups": [(0, 5), (1, 3)],
        "alerts": [1, 2],
        "build_queue": [1],
        "production_queue": [2, 3],
        "cargo": [["Marine"], ["SCV"]],
        "last_actions": [0, 2, 5],
        "screen_layers": np.random.rand(3, 64, 64).astype(np.float32),
    }

    out = parser.process_observation(raw_obs)
    scalar_shape = out["scalar"]["scalar_vec"].shape
    spatial_shape = out["spatial_fused"].shape
    print("Processed entities:", out["entities"]["ent_feats"].shape)
    print("Processed scalar:", scalar_shape, "(expect 256 dims)")
    print("Spatial fused shape:", spatial_shape, f"(expect (20, {parser.H}, {parser.W}))")
