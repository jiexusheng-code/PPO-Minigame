"""MultiInput policy with action masking on function id and flatten coordinate heads via MultiDiscrete action space."""
import numpy as np
import torch
from torch import nn

from pysc2.lib import features
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo.policies import MultiInputPolicy


class VectorLayerNormExtractor(BaseFeaturesExtractor):
    """Embedding-aware extractor with explicit masks + LayerNorm on vector.

    - vector: apply vector_mask (if provided) then LayerNorm
    - screen/minimap: categorical layers -> embedding, scalar layers -> pass-through
    - apply layer flags to screen/minimap before embedding
    """

    def __init__(self, observation_space, cnn_output_dim: int = 256):
        spaces = getattr(observation_space, "spaces", {})

        self._screen_layers = [
            (features.ScreenFeatures.visibility_map, "visibility_map"),
            (features.ScreenFeatures.player_relative, "player_relative"),
            (features.ScreenFeatures.unit_type, "unit_type"),
            (features.ScreenFeatures.selected, "selected"),
            (features.ScreenFeatures.unit_hit_points_ratio, "unit_hit_points_ratio"),
            (features.ScreenFeatures.build_progress, "build_progress"),
            (features.ScreenFeatures.buildable, "buildable"),
        ]
        self._minimap_layers = [
            (features.MinimapFeatures.visibility_map, "visibility_map"),
            (features.MinimapFeatures.player_relative, "player_relative"),
            (features.MinimapFeatures.selected, "selected"),
            (features.MinimapFeatures.unit_type, "unit_type"),
            (features.MinimapFeatures.alerts, "alerts"),
            (features.MinimapFeatures.buildable, "buildable"),
        ]

        self._screen_cat_info, self._screen_cont_idx, self._screen_in_channels = self._build_layer_info(
            self._screen_layers, features.SCREEN_FEATURES
        )
        self._minimap_cat_info, self._minimap_cont_idx, self._minimap_in_channels = self._build_layer_info(
            self._minimap_layers, features.MINIMAP_FEATURES
        )

        self._flat_keys = []
        flat_dim = 0
        for key in ["available_actions", "screen_layer_flags", "minimap_layer_flags", "vector_mask"]:
            if key in spaces:
                self._flat_keys.append(key)
                flat_dim += get_flattened_obs_dim(spaces[key])

        vector_dim = get_flattened_obs_dim(spaces["vector"]) if "vector" in spaces else 0

        screen_out_dim = self._expected_cnn_out_dim(spaces.get("screen"), cnn_output_dim)
        minimap_out_dim = self._expected_cnn_out_dim(spaces.get("minimap"), cnn_output_dim)

        total_dim = vector_dim + flat_dim + screen_out_dim + minimap_out_dim
        super().__init__(observation_space, features_dim=total_dim)

        self._vector_ln = None
        if "vector" in spaces:
            self._vector_ln = nn.LayerNorm(get_flattened_obs_dim(spaces["vector"]))

        self._screen_embeddings = nn.ModuleList([
            nn.Embedding(num, dim) for _, num, dim in self._screen_cat_info
        ])
        self._minimap_embeddings = nn.ModuleList([
            nn.Embedding(num, dim) for _, num, dim in self._minimap_cat_info
        ])

        self._screen_cnn, _screen_out_dim = self._build_cnn(spaces.get("screen"), self._screen_in_channels, cnn_output_dim)
        self._minimap_cnn, _minimap_out_dim = self._build_cnn(spaces.get("minimap"), self._minimap_in_channels, cnn_output_dim)

        self._flatten = nn.Flatten()

    @staticmethod
    def _layer_index(layer_feature):
        if hasattr(layer_feature, "index"):
            return layer_feature.index
        try:
            return int(layer_feature)
        except Exception:
            return int(getattr(layer_feature, "id", 0))

    @staticmethod
    def _is_categorical(meta) -> bool:
        f_type = getattr(meta, "type", None)
        if f_type is None:
            return False
        try:
            return "CAT" in str(f_type).upper()
        except Exception:
            return False

    def _build_layer_info(self, layers, meta_list):
        cat_info = []
        cont_idx = []
        in_channels = 0
        for i, (layer_feature, _name) in enumerate(layers):
            li = self._layer_index(layer_feature)
            meta = meta_list[li]
            if self._is_categorical(meta):
                num = int(getattr(meta, "scale", 0) or 0)
                if num <= 1:
                    num = 2
                emb_dim = int(min(16, max(2, int(np.ceil(np.sqrt(num))))))
                cat_info.append((i, num, emb_dim))
                in_channels += emb_dim
            else:
                cont_idx.append(i)
                in_channels += 1
        return cat_info, cont_idx, in_channels

    def _build_cnn(self, space, in_channels: int, cnn_output_dim: int):
        if space is None:
            return None, 0
        # infer H,W from (H,W,C) or (C,H,W)
        shape = space.shape
        if len(shape) != 3:
            return None, 0
        if shape[0] == in_channels:
            h, w = shape[1], shape[2]
        else:
            h, w = shape[0], shape[1]

        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            flat_dim = cnn(dummy).shape[1]
        fc = nn.Linear(flat_dim, cnn_output_dim)
        return nn.Sequential(cnn, fc, nn.ReLU()), cnn_output_dim

    @staticmethod
    def _expected_cnn_out_dim(space, cnn_output_dim: int) -> int:
        if space is None:
            return 0
        shape = getattr(space, "shape", None)
        if shape is None or len(shape) != 3:
            return 0
        return cnn_output_dim

    def _apply_channel_flags(self, x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        if flags is None:
            return x
        if x.dim() != 4 or flags.dim() != 2:
            return x
        if x.shape[1] == flags.shape[1]:
            return x * flags[:, :, None, None]
        if x.shape[-1] == flags.shape[1]:
            return x * flags[:, None, None, :]
        return x

    def _embed_spatial(self, x: torch.Tensor, flags: torch.Tensor, cat_info, cont_idx, emb_layers):
        if x.dim() != 4:
            return None
        # convert to (B,H,W,C)
        if x.shape[1] == (len(cat_info) + len(cont_idx)):
            x = x.permute(0, 2, 3, 1)
        x = self._apply_channel_flags(x, flags)

        parts = []
        emb_i = 0
        for i in range(x.shape[-1]):
            is_cat = any(ci[0] == i for ci in cat_info)
            if is_cat:
                idx, num, emb_dim = next(ci for ci in cat_info if ci[0] == i)
                vals = x[..., i].round().clamp(0, num - 1).long()
                emb = emb_layers[emb_i](vals)
                emb_i += 1
                parts.append(emb)
            else:
                parts.append(x[..., i].unsqueeze(-1))
        feat = torch.cat(parts, dim=-1)  # (B,H,W,C')
        feat = feat.permute(0, 3, 1, 2)  # (B,C',H,W)
        return feat

    def forward(self, observations):
        encoded = []

        if "vector" in observations:
            vec = observations["vector"]
            vector_mask = observations.get("vector_mask")
            if vector_mask is not None:
                vec = vec * vector_mask
            if self._vector_ln is not None:
                vec = self._vector_ln(vec)
            encoded.append(vec)

        if self._screen_cnn is not None and "screen" in observations:
            screen = observations["screen"]
            screen_flags = observations.get("screen_layer_flags")
            screen_feat = self._embed_spatial(screen, screen_flags, self._screen_cat_info, self._screen_cont_idx, self._screen_embeddings)
            if screen_feat is not None:
                encoded.append(self._screen_cnn(screen_feat))

        if self._minimap_cnn is not None and "minimap" in observations:
            minimap = observations["minimap"]
            minimap_flags = observations.get("minimap_layer_flags")
            minimap_feat = self._embed_spatial(minimap, minimap_flags, self._minimap_cat_info, self._minimap_cont_idx, self._minimap_embeddings)
            if minimap_feat is not None:
                encoded.append(self._minimap_cnn(minimap_feat))

        for key in self._flat_keys:
            if key in observations:
                encoded.append(self._flatten(observations[key]))

        return torch.cat(encoded, dim=1) if len(encoded) > 1 else encoded[0]


class MaskedFlattenPolicy(MultiInputPolicy):
    """Applies action_mask on the first categorical (function id) of a MultiDiscrete action space."""

    def _apply_action_mask(self, action_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        nvec = self.action_space.nvec
        func_dim = int(nvec[0])
        func_logits = action_logits[:, :func_dim]
        rest_logits = action_logits[:, func_dim:]
        # mask==0 -> set logit to large negative
        mask = mask.to(action_logits.device)
        func_logits = func_logits + (mask <= 0).float() * (-1e9)
        return torch.cat([func_logits, rest_logits], dim=1)

    def forward(self, obs, deterministic: bool = False):
        # 1. 特征提取
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        logits = self.action_net(latent_pi)

        # 2. 只对第一个头(fn_id)做mask，其他参数槽位严格一一对应
        if isinstance(self.action_dist, MultiCategoricalDistribution) and "available_actions" in obs:
            logits = self._apply_action_mask(logits, obs["available_actions"])

        # 3. 构造分布并采样，确保每个头只采样唯一语义参数
        distribution = self.action_dist.proba_distribution(logits)
        actions = distribution.get_actions(deterministic=deterministic)

        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def get_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        if isinstance(self.action_dist, MultiCategoricalDistribution) and "available_actions" in obs:
            logits = self._apply_action_mask(logits, obs["available_actions"])
        return self.action_dist.proba_distribution(logits)
