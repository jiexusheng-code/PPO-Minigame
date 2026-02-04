"""
Observation Parser: 将 PYSC2 的原始 observation 转化为固定结构的状态表示

设计原则：
1. 输出结构固定，支持跨地图使用
2. 地图差异通过配置驱动，不改变输出形状
3. 使用 mask 标记可用信息，避免缺省值混淆
4. minigame 专用（不处理复杂模式的观测）

输出格式：
{
    "vector": np.array([...]),              # 结构化向量特征（固定长度）
    "screen": np.array([...]),              # 屏幕空间特征 (H, W, C)
    "minimap": np.array([...]),             # 小地图空间特征 (H, W, C)
    "available_actions": np.array([...]),   # one-hot 可用动作
    "screen_layer_flags": np.array([...]),  # 每个 screen 层的 0/1 标志
    "minimap_layer_flags": np.array([...]), # 每个 minimap 层的 0/1 标志
    "vector_mask": np.array([...]),         # 每个 vector 维度的 0/1 标志
}
"""

import numpy as np
from typing import Dict, Any, List
from pysc2.lib import features, actions


# ======================== 配置部分 ========================

class MapConfig:
    """地图配置基类"""
    
    def __init__(self, map_name: str):
        self.map_name = map_name
        # 结构化向量的字段定义（字段名 -> 数据类型 + 处理函数）
        # 默认对所有 minigame 使用相同的 player 8 维字段
        self.vector_fields = {
            "player_vec": {
                "indices": [1, 2, 3, 4, 5, 6, 7, 8],
                "size": 8,
                "dtype": np.float32,
                "normalize": True,
                # 默认全部有效；各地图可仅覆盖 valid
                "valid": [True, True, True, True, True, True, True, True],
            }
        }
        # 所有 minigame 统一输出 7 层（canonical 顺序）
        self.screen_layers = [
            (features.ScreenFeatures.visibility_map, "visibility_map"),
            (features.ScreenFeatures.player_relative, "player_relative"),
            (features.ScreenFeatures.unit_type, "unit_type"),
            (features.ScreenFeatures.selected, "selected"),
            (features.ScreenFeatures.unit_hit_points_ratio, "unit_hit_points_ratio"),
            (features.ScreenFeatures.build_progress, "build_progress"),
            (features.ScreenFeatures.buildable, "buildable"),
        ]
        # 屏幕特征层实际需要的子集（按 name 标记，None 表示全用）
        self.screen_active_layers = None
        # 小地图特征层选择（canonical 顺序）
        self.minimap_layers = [
            (features.MinimapFeatures.visibility_map, "visibility_map"),
            (features.MinimapFeatures.player_relative, "player_relative"),
            (features.MinimapFeatures.selected, "selected"),
            (features.MinimapFeatures.unit_type, "unit_type"),
            (features.MinimapFeatures.alerts, "alerts"),
            (features.MinimapFeatures.buildable, "buildable"),
        ]
        # 小地图特征层实际需要的子集（按 name 标记，None 表示全用）
        self.minimap_active_layers = None
        # 屏幕/小地图的归一化尺寸
        self.screen_size = 32
        self.minimap_size = 32



class MoveToBeaconConfig(MapConfig):
    """MoveToBeacon 配置（仅需简单观测）"""
    
    def __init__(self):
        super().__init__("MoveToBeacon")
        
        # MoveToBeacon 不需要 player 8 维信息，仅覆盖 valid 置为 False
        # NOTE: 仅支持与 indices 等长的布尔列表/数组（每个维度单独有效/无效）
        self.vector_fields["player_vec"]["valid"] = [
            False, False, False, False, False, False, False, False
        ]
        # 对于 MoveToBeacon，只需要 player_relative、selected 两层
        self.screen_active_layers = {"player_relative", "selected"}
        self.minimap_active_layers = {"player_relative", "selected"}


# 地图配置映射表
AVAILABLE_MAPS = {
    "MoveToBeacon": MoveToBeaconConfig,
}


# ======================== ObsParser 类 ========================

class ObsParser:
    """
    PYSC2 observation 解析器
    
    将原始 observation 转化为固定结构、可跨地图使用的状态表示
    """
    
    def __init__(self, map_name: str):
        """
        Args:
            map_name: 地图名称（如 "MoveToBeacon"）
        """
        if map_name not in AVAILABLE_MAPS:
            raise ValueError(
                f"Map '{map_name}' not configured. Available: {list(AVAILABLE_MAPS.keys())}"
            )
        
        self.map_name = map_name
        self.config = AVAILABLE_MAPS[map_name]()
        
        # 从配置中获取尺寸参数
        self.screen_size = self.config.screen_size
        self.minimap_size = self.config.minimap_size
        
        # 向量使用规范长度（canonical），便于跨地图复用与有效性掩码
        # 默认采用 SC2 player 向量中的 11 个字段作为规范长度
        # 优先使用配置中声明的 canonical_vector_size
        self.vector_size = getattr(self.config, "canonical_vector_size", None)
        if self.vector_size is None:
            self.vector_size = sum(field["size"] for field in self.config.vector_fields.values())
        
        # 可用动作的维度（所有可能的 PYSC2 函数）
        self.num_actions = len(actions.FUNCTIONS)
        
        # 屏幕和小地图的通道数
        self.screen_channels = len(self.config.screen_layers)
        self.minimap_channels = len(self.config.minimap_layers)
    
    def get_output_spec(self) -> Dict[str, Any]:
        """
        获取输出状态的规格（用于初始化网络输入层）
        
        Returns:
            包含各部分 shape 和 dtype 的字典
        """
        return {
            "vector": {
                "shape": (self.vector_size,),
                "dtype": np.float32,
                "desc": "结构化向量特征（player信息等）"
            },
            "screen": {
                "shape": (self.screen_size, self.screen_size, self.screen_channels),
                "dtype": np.float32,
                "desc": f"屏幕特征层：{[name for _, name in self.config.screen_layers]}"
            },
            "minimap": {
                "shape": (self.minimap_size, self.minimap_size, self.minimap_channels),
                "dtype": np.float32,
                "desc": f"小地图特征层：{[name for _, name in self.config.minimap_layers]}"
            },
            "available_actions": {
                "shape": (self.num_actions,),
                "dtype": np.float32,
                "desc": "可用动作 one-hot 向量"
            },
            "screen_layer_flags": {
                "shape": (self.screen_channels,),
                "dtype": np.float32,
                "desc": "每个 canonical screen 层的 0/1 标志（顺序与 config.screen_layers 一致）"
            },
            "minimap_layer_flags": {
                "shape": (self.minimap_channels,),
                "dtype": np.float32,
                "desc": "每个 canonical minimap 层的 0/1 标志（顺序与 config.minimap_layers 一致）"
            },
            "vector_mask": {
                "shape": (self.vector_size,),
                "dtype": np.float32,
                "desc": "结构化向量每一维的 0/1 有效性标志（与 vector 对齐）"
            },
        }
    
    def parse(self, obs: Dict) -> Dict[str, np.ndarray]:
        """Parse a single observation and return inputs with explicit masks."""
        # 提取向量特征（不在 parser 内门控，mask 单独输出）
        vector, vector_mask = self._extract_vector(obs)

        # 提取空间特征（不在 parser 内门控，mask 单独输出）
        screen = self._extract_screen(obs)
        minimap = self._extract_minimap(obs)

        # 提取可用动作
        available_actions = self._extract_available_actions(obs)

        # 每层激活标志（独立输出，由模型侧做门控）
        active_set = self.config.screen_active_layers
        if active_set is None:
            screen_layer_flags = np.ones(self.screen_channels, dtype=np.float32)
        else:
            screen_layer_flags = np.array([
                1.0 if name in active_set else 0.0
                for _, name in self.config.screen_layers
            ], dtype=np.float32)

        minimap_active_set = self.config.minimap_active_layers
        if minimap_active_set is None:
            minimap_layer_flags = np.ones(self.minimap_channels, dtype=np.float32)
        else:
            minimap_layer_flags = np.array([
                1.0 if name in minimap_active_set else 0.0
                for _, name in self.config.minimap_layers
            ], dtype=np.float32)

        return {
            "vector": vector,
            "screen": screen,
            "minimap": minimap,
            "available_actions": available_actions,
            "screen_layer_flags": screen_layer_flags,
            "minimap_layer_flags": minimap_layer_flags,
            "vector_mask": vector_mask,
        }
    
    def _extract_vector(self, obs: Dict) -> tuple:
        """
        提取结构化向量特征
        
        【MoveToBeacon】：
        - player 信息（11维）：玩家ID、矿物、瓦斯、人口、军队数等
          用途：让agent知道当前资源状态（虽然MoveToBeacon中这些信息基本不变，
                但保留是为了代码的通用性）
          处理：逐维归一化到 [0, 1]
        """
        # 输出为选取字段的拼接向量（长度 = sum sizes），并返回对应的有效性掩码
        vec_list = []
        mask_list = []

        if "player_vec" in self.config.vector_fields:
            field_cfg = self.config.vector_fields["player_vec"]
            indices = field_cfg["indices"]
            player_vals = obs["player"][indices].astype(np.float32)

            if field_cfg.get("normalize", False):
                player_vals = np.log1p(player_vals)

            # 门控：仅支持与 indices 等长的布尔列表/数组（每维单独有效/无效）
            if "valid" not in field_cfg:
                raise RuntimeError("vector field 'player_vec' must provide 'valid' as a boolean list/array matching indices length")

            valid_cfg = field_cfg["valid"]
            valid_mask = np.asarray(valid_cfg, dtype=np.bool_)
            if valid_mask.shape[0] != len(indices):
                raise RuntimeError(f"'valid' length {valid_mask.shape[0]} doesn't match indices length {len(indices)} for player_vec")

            vec_list.append(player_vals)
            mask_list.append(valid_mask.astype(np.float32))

        if len(vec_list) == 0:
            vector = np.zeros(self.vector_size, dtype=np.float32)
            vector_mask = np.zeros(self.vector_size, dtype=np.float32)
        else:
            vector = np.concatenate(vec_list, axis=0).astype(np.float32)
            vector_mask = np.concatenate(mask_list, axis=0).astype(np.float32)

        # 如果长度不够，补零
        if len(vector) < self.vector_size:
            pad_len = self.vector_size - len(vector)
            vector = np.pad(vector, (0, pad_len), mode='constant', constant_values=0)
            vector_mask = np.pad(vector_mask, (0, pad_len), mode='constant', constant_values=0)

        return vector, vector_mask
    
    def _extract_screen(self, obs: Dict) -> np.ndarray:
        """
        提取屏幕特征层

        处理流程：
        1. 每层分别提取并 reshape 到 (H, W, 1)
        2. 对 SCALAR 层按 scale 归一化到 [0,1]；对 CATEGORICAL 层保留标签值
        3. 拼接成 (H, W, C) 并缩放到目标尺寸
        """
        layers = []
        import re

        def _get_layer_index(layer_feature):
            if hasattr(layer_feature, "index"):
                return layer_feature.index
            try:
                return int(layer_feature)
            except Exception:
                s = str(layer_feature)
                m = re.search(r"(\d+)", s)
                if m:
                    return int(m.group(1))
                raise RuntimeError(f"无法解析 screen layer feature: {layer_feature}")

        for layer_feature, _name in self.config.screen_layers:
            li = _get_layer_index(layer_feature)

            # 如果 obs 中没有 feature_screen，直接报错
            if "feature_screen" not in obs or obs.get("feature_screen") is None:
                raise RuntimeError("observation missing 'feature_screen' while screen layers are requested")

            fs = obs["feature_screen"]
            if li < 0 or li >= fs.shape[0]:
                raise RuntimeError(f"screen layer index {li} out of range for feature_screen with shape {fs.shape}")
            layer_data = fs[li]

            # 识别该层是 SCALAR 还是 CATEGORICAL（尽量使用 pysc2 的元信息）
            f_type = None
            try:
                meta = features.SCREEN_FEATURES[li]
                f_type = getattr(meta, "type", None)
            except Exception:
                f_type = getattr(layer_feature, "type", None)

            is_categorical = False
            if f_type is not None:
                try:
                    name = str(f_type).upper()
                    if "CAT" in name:
                        is_categorical = True
                except Exception:
                    is_categorical = False

            # 处理数据
            if is_categorical:
                # CATEGORICAL: 保留原始标签值，建议在模型侧用 embedding/one-hot 处理
                cdata = layer_data.astype(np.float32)
            else:
                # SCALAR: 按 scale 归一化到 [0,1]
                scale = None
                try:
                    meta = features.SCREEN_FEATURES[li]
                    scale = getattr(meta, "scale", None)
                except Exception:
                    scale = None

                cdata = layer_data.astype(np.float32)
                if scale is not None and scale > 1:
                    cdata = cdata / float(scale - 1)

            # 注意：不在 parser 内置零，保留原始层值；mask 由 screen_layer_flags 提供

            # 添加通道维度并收集
            cdata = np.expand_dims(cdata, axis=-1)
            layers.append(cdata)

        # 拼接所有层（若空则返回全零占位）
        if len(layers) == 0:
            screen = np.zeros((self.screen_size, self.screen_size, self.screen_channels), dtype=np.float32)
        else:
            screen = np.concatenate(layers, axis=-1)  # (H, W, C)

        # 缩放到目标尺寸
        screen = self._resize_spatial(screen, self.screen_size)

        return screen.astype(np.float32)

    # parser no longer exposes per-layer masks; gating is applied inside _extract_screen
    
    def _extract_minimap(self, obs: Dict) -> np.ndarray:
        """
        提取小地图特征层（与屏幕处理逻辑类似）
        """
        layers = []
        import re

        def _get_layer_index(layer_feature):
            if hasattr(layer_feature, "index"):
                return layer_feature.index
            try:
                return int(layer_feature)
            except Exception:
                s = str(layer_feature)
                m = re.search(r"(\d+)", s)
                if m:
                    return int(m.group(1))
                raise RuntimeError(f"无法解析 minimap layer feature: {layer_feature}")

        for layer_feature, _name in self.config.minimap_layers:
            li = _get_layer_index(layer_feature)

            if "feature_minimap" not in obs or obs.get("feature_minimap") is None:
                raise RuntimeError("observation missing 'feature_minimap' while minimap layers are requested")

            mm = obs["feature_minimap"]
            if li < 0 or li >= mm.shape[0]:
                raise RuntimeError(f"minimap layer index {li} out of range for feature_minimap with shape {mm.shape}")
            layer_data = mm[li]

            # 判断类型
            f_type = None
            try:
                meta = features.MINIMAP_FEATURES[li]
                f_type = getattr(meta, "type", None)
            except Exception:
                f_type = getattr(layer_feature, "type", None)

            is_categorical = False
            if f_type is not None:
                try:
                    name = str(f_type).upper()
                    if "CAT" in name:
                        is_categorical = True
                except Exception:
                    is_categorical = False

            if is_categorical:
                cdata = layer_data.astype(np.float32)
            else:
                scale = None
                try:
                    meta = features.MINIMAP_FEATURES[li]
                    scale = getattr(meta, "scale", None)
                except Exception:
                    scale = None

                cdata = layer_data.astype(np.float32)
                if scale is not None and scale > 1:
                    cdata = cdata / float(scale - 1)

            cdata = np.expand_dims(cdata, axis=-1)
            layers.append(cdata)

        if len(layers) == 0:
            minimap = np.zeros((self.minimap_size, self.minimap_size, self.minimap_channels), dtype=np.float32)
        else:
            minimap = np.concatenate(layers, axis=-1)

        minimap = self._resize_spatial(minimap, self.minimap_size)
        return minimap.astype(np.float32)
    
    def _extract_available_actions(self, obs: Dict) -> np.ndarray:
        """
        提取可用动作
        
        【所有地图】：
        - 从 obs["available_actions"] 获取可用的动作函数 ID 列表
        - 转换为 one-hot 向量
        
        处理：创建长度为 len(FUNCTIONS) 的 one-hot 编码
        """
        available = np.zeros(self.num_actions, dtype=np.float32)
        
        if "available_actions" in obs:
            for action_id in obs["available_actions"]:
                if action_id < self.num_actions:
                    available[action_id] = 1.0
        
        return available
    
    @staticmethod
    def _resize_spatial(spatial: np.ndarray, target_size: int) -> np.ndarray:
        """
        缩放空间特征到目标尺寸（双线性插值）
        
        Args:
            spatial: (H, W, C) 数组
            target_size: 目标尺寸
        
        Returns:
            (target_size, target_size, C) 数组
        """
        from scipy import ndimage
        
        h, w, c = spatial.shape
        if h == target_size and w == target_size:
            return spatial
        
        # 对每个通道分别缩放
        scale = target_size / max(h, w)
        resized_layers = []
        
        for i in range(c):
            layer = spatial[:, :, i]
            # 使用 zoom 进行缩放
            zoom_factors = (scale, scale)
            resized = ndimage.zoom(layer, zoom_factors, order=1)  # bilinear
            resized_layers.append(resized)
        
        # 裁剪或补零到精确尺寸
        resized = np.stack(resized_layers, axis=-1)
        h_new, w_new = resized.shape[:2]
        
        if h_new > target_size or w_new > target_size:
            resized = resized[:target_size, :target_size, :]
        elif h_new < target_size or w_new < target_size:
            pad_h = target_size - h_new
            pad_w = target_size - w_new
            resized = np.pad(resized, 
                           ((0, pad_h), (0, pad_w), (0, 0)),
                           mode='constant', constant_values=0)
        
        return resized


# ======================== 便捷函数 ========================

def create_parser(map_name: str) -> ObsParser:
    """创建 observation 解析器"""
    return ObsParser(map_name)


def parse_observations(obs_list: List[Dict], parser: ObsParser) -> Dict[str, np.ndarray]:
    """
    批量解析 observations（用于并行环境）
    
    Args:
        obs_list: observation 列表
        parser: ObsParser 实例
    
    Returns:
        {
            "vector": (B, vector_size),
            "screen": (B, H, W, C),
            "minimap": (B, H, W, C),
            "available_actions": (B, num_actions),
            "minimap_layer_flags": (B, minimap_channels),
            "vector_mask": (B, vector_size),
        }
    """
    batch = {
        "vector": [],
        "screen": [],
        "minimap": [],
        "available_actions": [],
        "screen_layer_flags": [],
        "minimap_layer_flags": [],
        "vector_mask": [],
    }
    
    for obs in obs_list:
        parsed = parser.parse(obs)
        for key in batch.keys():
            batch[key].append(parsed[key])
    
    # 堆叠成 batch
    result = {}
    for key in batch.keys():
        result[key] = np.stack(batch[key], axis=0)
    
    return result


if __name__ == "__main__":
    # 示例：打印 MoveToBeacon 的配置
    parser = create_parser("MoveToBeacon")
    print(f"Map: {parser.map_name}")
    print("\nOutput Specification:")
    for key, spec in parser.get_output_spec().items():
        print(f"  {key}: {spec['shape']} {spec['dtype'].__name__}")
        print(f"    {spec['desc']}")

    # --- 运行一个短的 MoveToBeacon 实例并尝试解析 obs（带防护） ---
    try:
        # 解析 absl flags，避免 pysc2 在未 parse flags 时报错
        from absl import flags
        try:
            flags.FLAGS(['prog', '--sc2_run_config=Windows'])
        except Exception:
            pass
    except Exception:
        pass

    try:
        from pysc2.env import sc2_env
        from pysc2.lib import actions, features as pysc2_features
    except Exception as e:
        print('\npysc2 not available or failed to import:', e)
        print('Skipping runtime demo. If you want to run this, ensure StarCraft II and pysc2 are installed.')
    else:
        print('\nAttempting a short MoveToBeacon episode (32x32)...')
        try:
            with sc2_env.SC2Env(
                map_name="MoveToBeacon",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=sc2_env.AgentInterfaceFormat(
                    feature_dimensions=sc2_env.Dimensions(screen=32, minimap=32)
                ),
                step_mul=8,
                game_steps_per_episode=0,
                visualize=False,
            ) as env:
                timesteps = env.reset()
                for _ in range(5):
                    act = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])
                    timesteps = env.step([act])

                last = timesteps[0].observation
                print('Observation keys:', list(last.keys()))
                try:
                    parsed = parser.parse(last)
                    print('Parsed keys:', list(parsed.keys()))
                    print('vector shape:', parsed['vector'].shape)
                    print('screen shape:', parsed['screen'].shape)
                    print('minimap shape:', parsed['minimap'].shape)
                    print('available_actions sum:', float(parsed['available_actions'].sum()))
                    print('screen_layer_flags:', parsed['screen_layer_flags'])
                    print('minimap_layer_flags:', parsed['minimap_layer_flags'])
                    print('vector_mask:', parsed['vector_mask'])
                except Exception as e:
                    print('Parser error while handling observation:', e)
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print('Error running SC2 env (ensure SC2 and replays are installed):', e)
            import traceback
            traceback.print_exc()
