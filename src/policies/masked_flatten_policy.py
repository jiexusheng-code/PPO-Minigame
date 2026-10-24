"""MultiInput policy with action masking on function id and flatten coordinate heads via MultiDiscrete action space."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pysc2.lib import features
from pysc2.lib import actions as pysc2_actions
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo.policies import MultiInputPolicy


# Output channels of the full-resolution stride-1 spatial trunk. The trunk is
# shared by the main action/value features and the per-map spatial heads.
SPATIAL_FEATURE_CHANNELS = 32
# Function-embedding planes broadcast into the spatial heads.
FN_EMBED_DIM = 16


def _as_action_tensor(actions, device) -> torch.Tensor:
    if isinstance(actions, torch.Tensor):
        return actions.to(device)
    try:
        return torch.as_tensor(actions, device=device)
    except Exception:
        return torch.tensor(actions, device=device)


def _resolve_channel_flags(flags: torch.Tensor, n_channels: int, name: str) -> torch.Tensor:
    """Validate per-channel layer flags and return them as (B, n_channels).

    Layer flags select whole feature planes (one flag per channel); they are
    not element-wise spatial masks.
    """
    if flags.dim() == 1:
        flags = flags.unsqueeze(0)
    if flags.dim() != 2:
        raise ValueError(
            f"'{name}' must be per-channel flags of shape (B, {n_channels}) or "
            f"({n_channels},), got {tuple(flags.shape)}; spatial masks are not supported."
        )
    if flags.shape[1] != n_channels:
        raise ValueError(
            f"'{name}' has {flags.shape[1]} channel flags but the observation has "
            f"{n_channels} channels."
        )
    return flags


class VectorLayerNormExtractor(BaseFeaturesExtractor):
    """Embedding-aware extractor with explicit masks + LayerNorm on vector.

    - vector: apply vector_mask (if provided) then LayerNorm
    - screen/minimap: categorical layers -> embedding, scalar layers -> pass-through
    - apply layer flags to screen/minimap before embedding
    - each map is encoded by a full-resolution stride-1 conv trunk; the trunk
      output is shared with the policy's spatial action heads.

    Canonical layer order (must match ``src.obs_parser``):
      screen : visibility_map, player_relative, unit_type, selected,
               unit_hit_points_ratio, build_progress, buildable, active
      minimap: visibility_map, camera, player_relative, selected, buildable
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
            (features.ScreenFeatures.active, "active"),
        ]
        self._minimap_layers = [
            (features.MinimapFeatures.visibility_map, "visibility_map"),
            (features.MinimapFeatures.camera, "camera"),
            (features.MinimapFeatures.player_relative, "player_relative"),
            (features.MinimapFeatures.selected, "selected"),
            (features.MinimapFeatures.buildable, "buildable"),
        ]
        self._screen_n_layers = len(self._screen_layers)
        self._minimap_n_layers = len(self._minimap_layers)

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

        self._screen_encoder, self._screen_fc = self._build_spatial_tower(
            spaces.get("screen"), self._screen_n_layers, self._screen_in_channels, cnn_output_dim
        )
        self._minimap_encoder, self._minimap_fc = self._build_spatial_tower(
            spaces.get("minimap"), self._minimap_n_layers, self._minimap_in_channels, cnn_output_dim
        )

        self._flatten = nn.Flatten()
        # Spatial trunk outputs from the most recent forward, reused by the
        # policy's spatial heads so the conv trunk runs exactly once per batch.
        self.last_spatial_maps = None

    @staticmethod
    def _layer_index(layer_feature, name=None, meta_list=None):
        idx = getattr(layer_feature, "index", None)
        if isinstance(idx, int):
            return idx
        try:
            return int(layer_feature)
        except Exception:
            pass
        # PySC2 exposes ScreenFeatures/MinimapFeatures fields as namedtuple
        # descriptors (not ints), so fall back to the layer name.
        if name is not None and meta_list is not None:
            for i, meta in enumerate(meta_list):
                if getattr(meta, "name", None) == name:
                    return i
        return 0

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
        for i, (layer_feature, name) in enumerate(layers):
            li = self._layer_index(layer_feature, name, meta_list)
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

    @staticmethod
    def _spatial_hw(space, n_layers: int):
        if space is None:
            return None, None
        shape = getattr(space, "shape", None)
        if shape is None or len(shape) != 3:
            return None, None
        if shape[-1] == n_layers:
            return int(shape[0]), int(shape[1])
        if shape[0] == n_layers:
            return int(shape[1]), int(shape[2])
        return int(shape[0]), int(shape[1])

    def _build_spatial_tower(self, space, n_layers: int, in_channels: int, cnn_output_dim: int):
        """Build a full-resolution stride-1 trunk + projection for one map.

        Two same-padded conv layers (16ch 5x5 followed by 32ch 3x3, both
        stride 1) keep the (H, W) grid so the spatial action head can produce
        one logit per cell. The projection to ``cnn_output_dim`` feeds the
        main action/value network, matching the historical features_dim.
        """
        if space is None:
            return None, None
        h, w = self._spatial_hw(space, n_layers)
        if h is None:
            return None, None
        encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, SPATIAL_FEATURE_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SPATIAL_FEATURE_CHANNELS * h * w, cnn_output_dim),
            nn.ReLU(),
            # Control the main-network input scale to reduce critic Tanh saturation.
            nn.LayerNorm(cnn_output_dim),
        )
        return encoder, fc

    @staticmethod
    def _expected_cnn_out_dim(space, cnn_output_dim: int) -> int:
        if space is None:
            return 0
        shape = getattr(space, "shape", None)
        if shape is None or len(shape) != 3:
            return 0
        return cnn_output_dim

    def _embed_spatial(self, x: torch.Tensor, flags: torch.Tensor, cat_info, cont_idx, emb_layers, flags_name: str = "layer_flags"):
        if x.dim() != 4:
            return None
        # convert to (B,H,W,C)
        if x.shape[1] == (len(cat_info) + len(cont_idx)):
            x = x.permute(0, 2, 3, 1)

        ch_flags = None
        if flags is not None:
            ch_flags = _resolve_channel_flags(flags.float(), x.shape[-1], flags_name)
            x = x * ch_flags[:, None, None, :]

        parts = []
        emb_i = 0
        for i in range(x.shape[-1]):
            is_cat = any(ci[0] == i for ci in cat_info)
            if is_cat:
                idx, num, emb_dim = next(ci for ci in cat_info if ci[0] == i)
                vals = x[..., i].round().clamp(0, num - 1).long()
                emb = emb_layers[emb_i](vals)
                if ch_flags is not None:
                    # embedding(0) is a learned non-zero vector, so zeroing the
                    # raw channel is not enough: kill the embedding output too.
                    emb = emb * ch_flags[:, i, None, None, None]
                emb_i += 1
                parts.append(emb)
            else:
                parts.append(x[..., i].unsqueeze(-1))
        feat = torch.cat(parts, dim=-1)  # (B,H,W,C')
        feat = feat.permute(0, 3, 1, 2)  # (B,C',H,W)
        return feat

    def _layer_flags(self, observations, key: str):
        """Return validated per-layer flags (B, n_layers) for a map, or None."""
        if key == "screen":
            flags = observations.get("screen_layer_flags")
            n_layers = self._screen_n_layers
        else:
            flags = observations.get("minimap_layer_flags")
            n_layers = self._minimap_n_layers
        if flags is None:
            return None
        if not isinstance(flags, torch.Tensor):
            flags = torch.as_tensor(flags, device=observations[key].device)
        return _resolve_channel_flags(flags.float(), n_layers, f"{key}_layer_flags")

    def encode_spatial(self, observations, key: str):
        """Run the shared stride-1 trunk for ``key`` and return (B,C,H,W)."""
        if key == "screen":
            if self._screen_encoder is None or "screen" not in observations:
                return None
            feat = self._embed_spatial(
                observations["screen"],
                observations.get("screen_layer_flags"),
                self._screen_cat_info,
                self._screen_cont_idx,
                self._screen_embeddings,
                "screen_layer_flags",
            )
            return None if feat is None else self._screen_encoder(feat)
        if self._minimap_encoder is None or "minimap" not in observations:
            return None
        feat = self._embed_spatial(
            observations["minimap"],
            observations.get("minimap_layer_flags"),
            self._minimap_cat_info,
            self._minimap_cont_idx,
            self._minimap_embeddings,
            "minimap_layer_flags",
        )
        return None if feat is None else self._minimap_encoder(feat)

    def forward(self, observations):
        encoded = []
        self.last_spatial_maps = {}

        if "vector" in observations:
            vec = observations["vector"]
            vector_mask = observations.get("vector_mask")
            if vector_mask is not None:
                vec = vec * vector_mask
            if self._vector_ln is not None:
                vec = self._vector_ln(vec)
            encoded.append(vec)

        if self._screen_encoder is not None and "screen" in observations:
            screen_feat = self.encode_spatial(observations, "screen")
            if screen_feat is not None:
                self.last_spatial_maps["screen"] = screen_feat
                encoded.append(self._screen_fc(screen_feat))

        if self._minimap_encoder is not None and "minimap" in observations:
            minimap_feat = self.encode_spatial(observations, "minimap")
            if minimap_feat is not None:
                self.last_spatial_maps["minimap"] = minimap_feat
                encoded.append(self._minimap_fc(minimap_feat))

        for key in self._flat_keys:
            if key in observations:
                encoded.append(self._flatten(observations[key]))

        return torch.cat(encoded, dim=1) if len(encoded) > 1 else encoded[0]


class MaskedFlattenPolicy(MultiInputPolicy):
    """Function + spatial action policy over the full MultiDiscrete action space.

    Only the function head and the spatial ``screen``/``minimap`` slots are
    sampled. Every other argument slot is a deterministic zero, excluded from
    log_prob and entropy. For ``select_rect`` the learned ``screen`` value is
    the rectangle center; the environment derives the two corners (see the
    convention documented in ``_sample_joint_actions``).
    """

    def _build(self, lr_schedule) -> None:
        # Spatial + conditioning parameters must be registered before SB3
        # builds the optimizer from ``self.parameters()``.
        self._build_spatial_heads()
        self._build_function_embedding()
        super()._build(lr_schedule)

    def _build_spatial_heads(self) -> None:
        """Create spatial action heads eagerly from observation/action spaces."""
        param_semantics = []
        param_semantics_set = set()
        for fn in pysc2_actions.FUNCTIONS:
            for spec in fn.args:
                name = getattr(spec, "name")
                if name not in param_semantics_set:
                    param_semantics.append(name)
                    param_semantics_set.add(name)

        # fn_id -> list of slot indices (in param_semantics order)
        fn_param_map = {}
        for fn in pysc2_actions.FUNCTIONS:
            fn_param_map[fn.id] = [param_semantics.index(getattr(spec, "name")) for spec in fn.args]

        # slot sizes come from action_space.nvec (skip first entry which is fn head)
        slot_sizes = [int(x) for x in list(self.action_space.nvec)[1:]]
        if len(slot_sizes) != len(param_semantics):
            raise ValueError(
                f"action_space has {len(slot_sizes)} argument slots but PySC2 defines "
                f"{len(param_semantics)} unique argument names; spatial action heads cannot be aligned."
            )

        self._param_semantics = param_semantics
        self._fn_param_map = fn_param_map
        self._slot_sizes = slot_sizes
        self._screen_slot_idx = param_semantics.index("screen") if "screen" in param_semantics else None
        self._minimap_slot_idx = param_semantics.index("minimap") if "minimap" in param_semantics else None
        # Slots sampled from a learned spatial distribution. ``screen2`` is
        # deliberately absent: select_rect's corners are derived from the
        # learned screen center by the environment.
        self._sampled_slot_indices = sorted(
            idx for idx in (self._screen_slot_idx, self._minimap_slot_idx) if idx is not None
        )

        # fn -> binary mask over slots, used for joint log_prob/entropy.
        # Non-persistent buffer: it follows policy.to(device) but is not part of
        # the checkpoint (it is fully derived from the action space).
        n_funcs = int(list(self.action_space.nvec)[0])
        n_slots = len(slot_sizes)
        self._n_funcs = n_funcs
        fn_slot_mask = torch.zeros((n_funcs, n_slots), dtype=torch.float32)
        for fid, slots in fn_param_map.items():
            for s in slots:
                if s < n_slots:
                    fn_slot_mask[fid, s] = 1.0
        self.register_buffer("_fn_slot_mask", fn_slot_mask, persistent=False)

        spaces = getattr(self.observation_space, "spaces", {})
        self._screen_n_layers = self._spatial_layer_count(spaces.get("screen"), "_screen_n_layers")
        self._minimap_n_layers = self._spatial_layer_count(spaces.get("minimap"), "_minimap_n_layers")

        # Base head: trunk output + broadcast layer flags (expensive, cached
        # across function branches during exact entropy). Function conditioning
        # is a cheap per-function channel bias added before the per-map 1x1
        # output conv, so enumerating dozens of available functions stays cheap.
        if self._screen_n_layers is not None and self._screen_slot_idx is not None:
            self._screen_spatial_conv = nn.Conv2d(
                SPATIAL_FEATURE_CHANNELS + self._screen_n_layers,
                16,
                kernel_size=3,
                padding=1,
            )
            self._screen_fn_conditioner = nn.Linear(FN_EMBED_DIM, 16)
            self._screen_spatial_out = nn.Conv2d(16, 1, kernel_size=1)
        else:
            self._screen_spatial_conv = None
            self._screen_fn_conditioner = None
            self._screen_spatial_out = None

        if self._minimap_n_layers is not None and self._minimap_slot_idx is not None:
            self._minimap_spatial_conv = nn.Conv2d(
                SPATIAL_FEATURE_CHANNELS + self._minimap_n_layers,
                16,
                kernel_size=3,
                padding=1,
            )
            self._minimap_fn_conditioner = nn.Linear(FN_EMBED_DIM, 16)
            self._minimap_spatial_out = nn.Conv2d(16, 1, kernel_size=1)
        else:
            self._minimap_spatial_conv = None
            self._minimap_fn_conditioner = None
            self._minimap_spatial_out = None

    def _spatial_layer_count(self, space, extractor_attr):
        n_layers = getattr(self.features_extractor, extractor_attr, None)
        if n_layers is not None:
            return int(n_layers)
        if space is None:
            return None
        shape = getattr(space, "shape", None)
        if shape is None or len(shape) != 3:
            return None
        return int(shape[-1])

    def _build_function_embedding(self) -> None:
        if hasattr(self, "_function_embedding"):
            return
        n_funcs = int(self.action_space.nvec[0])
        self._function_embedding = nn.Embedding(n_funcs, FN_EMBED_DIM)

    def _build_slot_map_if_needed(self):
        # Slot mask is built eagerly in _build_spatial_heads; this method only
        # performs the one-time environment snapshot cross-check.
        if getattr(self, "_slot_map_built", False):
            return
        if not hasattr(self, "_param_semantics"):
            # Safety net for construction paths that bypass _build().
            self._build_spatial_heads()
            self._build_function_embedding()

        self._slot_map_built = True
        # One-time cross-check against environment snapshot (if present)
        try:
            import json, os, logging
            if not getattr(self, '_param_semantics_checked', False):
                candidate_paths = []
                train_out_dir = os.environ.get('TRAIN_OUT_DIR', None)
                if train_out_dir:
                    candidate_paths.append(os.path.join(train_out_dir, 'param_semantics_env.json'))
                candidate_paths.append(os.path.join(os.getcwd(), 'models', 'param_semantics_env.json'))
                # backward compatibility with historical root-level snapshot
                candidate_paths.append(os.path.join(os.getcwd(), 'param_semantics_env.json'))

                env_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
                if env_path is not None:
                    try:
                        with open(env_path, 'r', encoding='utf-8') as f:
                            snap = json.load(f)
                        env_sem = snap.get('param_semantics')
                        if env_sem is not None and env_sem != self._param_semantics:
                            logging.getLogger('train').warning(
                                'param_semantics mismatch between env and policy.\n'
                                f'env (first 10): {env_sem[:10]}\n'
                                f'policy (first 10): {self._param_semantics[:10]}\n'
                                'Saved env snapshot to param_semantics_env.json; please verify mapping.'
                            )
                        # mark checked so we don't spam
                        self._param_semantics_checked = True
                    except Exception:
                        pass
        except Exception:
            pass

    def _clear_last_spatial_maps(self):
        """Drop the extractor's reference to the current spatial maps.

        The maps themselves remain alive through the tensors that use them;
        this only prevents a stale cache (and the autograd graph it holds)
        from being reused or retained across calls, e.g. by ``predict_values``.
        """
        extractor = self.features_extractor
        if hasattr(extractor, "last_spatial_maps"):
            extractor.last_spatial_maps = None

    def _extract_latents(self, obs):
        """Run the feature extractor once and keep its shared spatial maps."""
        self._clear_last_spatial_maps()
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        maps = getattr(self.features_extractor, "last_spatial_maps", None)
        return latent_pi, latent_vf, maps

    def _spatial_maps(self, obs):
        """Compute the shared trunk maps for ``obs``.

        Callers that already ran the feature extractor should pass their maps
        explicitly; this fallback never reuses a possibly-stale cache.
        """
        maps = {}
        encode = getattr(self.features_extractor, "encode_spatial", None)
        if encode is None:
            raise RuntimeError(
                "VectorLayerNormExtractor is required for spatial heads; "
                f"got {type(self.features_extractor).__name__}."
            )
        for key in ("screen", "minimap"):
            enc = encode(obs, key)
            if enc is not None:
                maps[key] = enc
        return maps

    def _spatial_head_logits(self, key, obs, fn_ids, spatial_maps, head_cache=None):
        """Per-cell logits (B, H*W) for one map from the shared trunk output.

        The expensive 3x3 convolution over ``[trunk, flags]`` depends only on
        the observation, so it is cached per map in ``head_cache``. Each
        function branch then costs one linear projection of the embedding plus
        a 1x1 output convolution, which keeps exact entropy cheap even when
        dozens of functions are available.
        """
        if key == "screen":
            conv = self._screen_spatial_conv
            conditioner = self._screen_fn_conditioner
            out_conv = self._screen_spatial_out
            n_layers = self._screen_n_layers
            slot_idx = self._screen_slot_idx
        else:
            conv = self._minimap_spatial_conv
            conditioner = self._minimap_fn_conditioner
            out_conv = self._minimap_spatial_out
            n_layers = self._minimap_n_layers
            slot_idx = self._minimap_slot_idx
        if conv is None or out_conv is None or slot_idx is None:
            return None

        base = head_cache.get(key) if head_cache is not None else None
        if base is None:
            enc = spatial_maps.get(key) if spatial_maps is not None else None
            if enc is None:
                enc = self.features_extractor.encode_spatial(obs, key)
                if enc is None:
                    return None
                if spatial_maps is not None:
                    spatial_maps[key] = enc

            batch, _, height, width = enc.shape
            flags = self.features_extractor._layer_flags(obs, key)
            if flags is None:
                flags_planes = enc.new_zeros(batch, n_layers, height, width)
            else:
                flags = flags.to(device=enc.device, dtype=enc.dtype)
                flags_planes = flags[:, :, None, None].expand(-1, -1, height, width)
            base = conv(torch.cat([enc, flags_planes], dim=1))
            if head_cache is not None:
                head_cache[key] = base

        if fn_ids is None:
            features = F.relu(base)
        else:
            embedding = self._function_embedding(
                fn_ids.to(device=base.device, dtype=torch.long)
            )
            fn_bias = conditioner(embedding)[:, :, None, None]
            features = F.relu(base + fn_bias)

        out = out_conv(features).flatten(1)
        slot_size = self._slot_sizes[slot_idx]
        if out.shape[1] != slot_size:
            raise ValueError(
                f"Spatial logits for '{key}' contain {out.shape[1]} cells, "
                f"but the action slot has size {slot_size}."
            )
        return out

    def _condition_argument_logits(
        self,
        base_logits,
        fn_ids,
        obs,
        used_slots=None,
        spatial_maps=None,
        head_cache=None,
    ):
        """Return action logits with spatial slots conditioned on the function.

        ``used_slots`` optionally restricts which spatial heads are evaluated;
        skipped slots retain their flat logits and must be unused by the caller.
        ``spatial_maps`` carries the shared trunk output so the exact-entropy
        enumeration does not re-run it for every function branch. ``head_cache``
        similarly caches the observation-only base convolution per map.
        """
        self._build_slot_map_if_needed()
        safe_fn_ids = fn_ids.to(device=base_logits.device, dtype=torch.long).clamp(
            min=0, max=self._n_funcs - 1
        )
        if spatial_maps is None:
            spatial_maps = self._spatial_maps(obs)
        return self._inject_spatial_logits(
            base_logits,
            obs,
            fn_ids=safe_fn_ids,
            used_slots=used_slots,
            spatial_maps=spatial_maps,
            head_cache=head_cache,
        )

    def _sample_joint_actions(self, base_logits, obs, deterministic=False, spatial_maps=None):
        """Sample function + spatial slots; all other argument slots are zero.

        select_rect convention: the sampled ``screen`` slot holds the learned
        rectangle center as a flat row-major index (``flat = y * W + x``).
        ``screen2`` is left at 0 and must be ignored by the environment, which
        derives the two corners around the center.
        """
        sizes = [int(x) for x in self.action_space.nvec]
        fn_logits = torch.split(base_logits, sizes, dim=1)[0]
        fn_dist = torch.distributions.Categorical(logits=fn_logits)
        fn_ids = fn_logits.argmax(dim=1) if deterministic else fn_dist.sample()
        conditioned = self._condition_argument_logits(
            base_logits, fn_ids, obs, spatial_maps=spatial_maps
        )
        slot_logits = torch.split(conditioned, sizes, dim=1)[1:]
        actions = torch.zeros((base_logits.shape[0], len(sizes)), dtype=torch.long, device=base_logits.device)
        actions[:, 0] = fn_ids
        used = self._fn_slot_mask[fn_ids]
        for slot_idx in self._sampled_slot_indices:
            slot_logits_i = slot_logits[slot_idx]
            rows = torch.nonzero(used[:, slot_idx] > 0, as_tuple=False).squeeze(1)
            if rows.numel() == 0:
                continue
            if deterministic:
                values = slot_logits_i[rows].argmax(dim=1)
            else:
                values = torch.distributions.Categorical(logits=slot_logits_i[rows]).sample()
            actions[rows, slot_idx + 1] = values
        return actions, conditioned

    def _conditional_joint_entropy(self, base_logits, obs, spatial_maps=None):
        """Exact joint conditional entropy of the hierarchical policy.

        The policy factorizes as ``pi(fn|s) * prod_{slot in sampled(fn)} pi(slot|fn, s)``,
        where ``sampled(fn)`` is the function's used slots intersected with the
        learned spatial slots (screen/minimap). Non-spatial slots are
        deterministic zeros and therefore contribute no entropy.

            H(fn|s) + sum_fn pi(fn|s) * sum_{slot in sampled(fn)} H(slot|fn, s).

        The function sum runs over the union of functions permitted by
        ``obs['available_actions']``. The shared spatial trunk is computed once
        and reused for every function branch.
        """
        sizes = [int(x) for x in self.action_space.nvec]
        fn_logits = torch.split(base_logits, sizes, dim=1)[0]
        fn_logp = F.log_softmax(fn_logits, dim=1)
        fn_probs = fn_logp.exp()
        fn_entropy = -(fn_probs * fn_logp).sum(dim=1)

        if spatial_maps is None:
            spatial_maps = self._spatial_maps(obs)
        sampled = set(self._sampled_slot_indices)

        available = obs.get("available_actions") if isinstance(obs, dict) else None
        if available is not None:
            mask = torch.as_tensor(available, device=base_logits.device)
            # Keep the support identical to sampling/evaluation: an all-zero
            # row is treated as ``no_op``-only rather than uniformly over all
            # functions.
            mask = self._with_no_op_fallback(mask)
            if mask.dim() == 2:
                fn_ids = torch.nonzero(mask.any(dim=0), as_tuple=False).squeeze(1)
            else:
                fn_ids = torch.nonzero(mask > 0, as_tuple=False).squeeze(1)
            fn_ids = fn_ids[fn_ids < self._n_funcs]
            if fn_ids.numel() == 0:
                fn_ids = torch.arange(self._n_funcs, device=base_logits.device)
        else:
            fn_ids = torch.arange(self._n_funcs, device=base_logits.device)

        arg_expectation = torch.zeros_like(fn_entropy)
        head_cache = {}
        for fn_id in fn_ids.tolist():
            weight = fn_probs[:, fn_id]
            if not torch.any(weight > 0):
                continue
            fn_slots = self._fn_slot_mask[fn_id]
            used_slots = [
                slot for slot in self._sampled_slot_indices
                if bool(fn_slots[slot] > 0)
            ]
            if not used_slots:
                # No sampled arguments: this branch contributes no argument entropy.
                continue
            branch_ids = torch.full(
                (base_logits.shape[0],), fn_id, dtype=torch.long, device=base_logits.device
            )
            conditioned = self._condition_argument_logits(
                base_logits,
                branch_ids,
                obs,
                used_slots=used_slots,
                spatial_maps=spatial_maps,
                head_cache=head_cache,
            )
            slot_logits = torch.split(conditioned, sizes, dim=1)[1:]
            used = self._fn_slot_mask[branch_ids]
            branch_entropy = torch.zeros_like(fn_entropy)
            for slot_idx in used_slots:
                if slot_idx not in sampled:
                    continue
                slot_logp = F.log_softmax(slot_logits[slot_idx], dim=1)
                branch_entropy = branch_entropy + (
                    -(slot_logp.exp() * slot_logp).sum(dim=1) * used[:, slot_idx]
                )
            arg_expectation = arg_expectation + weight * branch_entropy
        return fn_entropy + arg_expectation

    def _joint_logprob_and_entropy(self, logits: torch.Tensor, actions: torch.Tensor):
        """Joint log_prob and entropy over the function + sampled spatial slots.

        logits: (B, sum(nvec)), already spatially conditioned for each row's fn
        actions: (B, n_actions)  # first col is fn_id, following are slot values
        Returns: log_prob (B,), entropy (B,)
        """
        self._build_slot_map_if_needed()
        nvec = list(self.action_space.nvec)
        sizes = [int(x) for x in nvec]
        splits = torch.split(logits, sizes, dim=1)
        fn_logits = splits[0]
        slot_logits = splits[1:]

        fn_logp = F.log_softmax(fn_logits, dim=1)
        fn_ids = actions[:, 0].long()
        fn_selected = fn_logp.gather(
            1, fn_ids.clamp(min=0, max=fn_logits.shape[1] - 1).unsqueeze(1)
        ).squeeze(1)

        total_slot_logp = torch.zeros_like(fn_selected)
        total_slot_ent = torch.zeros_like(fn_selected)
        fn_mask = self._fn_slot_mask[fn_ids]  # (B, n_slots)

        # Differentiable statistics for monitoring; only sampled slots count.
        slot_used_means = []
        slot_entropy_means = []
        slot_logprob_means = []
        for i, s_logits in enumerate(slot_logits):
            size = s_logits.shape[1]
            used = fn_mask[:, i]
            if i in self._sampled_slot_indices:
                logp = F.log_softmax(s_logits, dim=1)
                probs = logp.exp()
                a_i = actions[:, i + 1].long().clamp(min=0, max=size - 1)
                picked = logp.gather(1, a_i.unsqueeze(1)).squeeze(1) * used
                ent = -(probs * logp).sum(dim=1) * used
                total_slot_logp = total_slot_logp + picked
                total_slot_ent = total_slot_ent + ent
                raw_ent = -(probs * logp).sum(dim=1)
            else:
                picked = torch.zeros_like(fn_selected)
                raw_ent = torch.zeros_like(fn_selected)
                used = torch.zeros_like(used)

            used_sum = float(used.sum().detach().cpu().item())
            if used_sum > 0:
                slot_used_means.append(float(used.mean().detach().cpu().item()))
                slot_entropy_means.append(
                    float((raw_ent * used).sum().detach().cpu().item() / (used_sum + 1e-8))
                )
                slot_logprob_means.append(
                    float(picked.sum().detach().cpu().item() / (used_sum + 1e-8))
                )
            else:
                slot_used_means.append(0.0)
                slot_entropy_means.append(0.0)
                slot_logprob_means.append(0.0)

        total_logp = fn_selected + total_slot_logp
        fn_probs = F.softmax(fn_logits, dim=1)
        total_ent = -(fn_probs * fn_logp).sum(dim=1) + total_slot_ent

        # Cache per-arg statistics for external callbacks/monitoring
        try:
            fn_entropy_mean = float((-(fn_probs * fn_logp).sum(dim=1)).mean().detach().cpu().item())
            self._last_per_arg_stats = {
                "fn_entropy_mean": fn_entropy_mean,
                "fn_logprob_mean": float(fn_selected.mean().detach().cpu().item()),
                "slot_used_mean": slot_used_means,
                "slot_entropy_mean": slot_entropy_means,
                "slot_logprob_mean": slot_logprob_means,
            }
        except Exception:
            # non-critical: do not break training if stats computation fails
            self._last_per_arg_stats = None

        return total_logp, total_ent

    def _inject_spatial_logits(
        self,
        logits: torch.Tensor,
        obs,
        fn_ids=None,
        used_slots=None,
        spatial_maps=None,
        head_cache=None,
    ):
        """Replace flat slot logits with shared-trunk spatial-head logits.

        Keeps the external MultiDiscrete action interface while producing
        ``screen``/``minimap`` logits from the same full-resolution trunk used
        by the main network. ``used_slots`` restricts which spatial heads are
        evaluated (unused slots keep their flat logits).
        """
        if not isinstance(logits, torch.Tensor):
            return logits
        sizes = [int(x) for x in self.action_space.nvec]
        func_dim = int(sizes[0])
        slot_sizes = [int(x) for x in sizes[1:]]
        used = None if used_slots is None else {int(slot) for slot in used_slots}
        if spatial_maps is None:
            spatial_maps = self._spatial_maps(obs)

        outputs = {}
        for key, slot_idx in (
            ("screen", self._screen_slot_idx),
            ("minimap", self._minimap_slot_idx),
        ):
            if slot_idx is None or key not in obs:
                continue
            if used is not None and slot_idx not in used:
                continue
            outputs[slot_idx] = self._spatial_head_logits(
                key, obs, fn_ids, spatial_maps, head_cache=head_cache
            )

        # Build the result out-of-place: in-place writes into a tensor that is
        # part of the autograd graph are fragile and unnecessary here.
        pieces = [logits[:, :func_dim]]
        cursor = func_dim
        for slot_idx, slot_size in enumerate(slot_sizes):
            end = cursor + slot_size
            flat = outputs.get(slot_idx)
            pieces.append(flat if flat is not None else logits[:, cursor:end])
            cursor = end
        return torch.cat(pieces, dim=1)

    def _base_action_logits(self, latent_pi, obs):
        logits = self.action_net(latent_pi)
        if isinstance(self.action_dist, MultiCategoricalDistribution) and "available_actions" in obs:
            logits = self._apply_action_mask(logits, obs["available_actions"])
        return logits

    @staticmethod
    def _with_no_op_fallback(mask: torch.Tensor) -> torch.Tensor:
        """Allow ``no_op`` for rows whose available-action mask is all zero.

        PySC2 normally guarantees at least one available action, but a malformed
        or synthetic all-zero row would otherwise silently become uniform over
        every PySC2 function. Sampling, evaluation and entropy must share one
        support, so replace that degenerate row with the same legal ``no_op``
        fallback the env uses.
        """
        no_op_id = int(pysc2_actions.FUNCTIONS.no_op.id)
        if mask.dim() == 0 or mask.shape[-1] <= no_op_id:
            return mask
        if mask.dim() == 1:
            if bool((mask > 0).any()):
                return mask
            patched = mask.clone()
            patched[no_op_id] = 1.0
            return patched
        if mask.dim() == 2:
            empty_rows = (mask > 0).sum(dim=1) == 0
            if not bool(empty_rows.any()):
                return mask
            patched = mask.clone()
            patched[empty_rows, no_op_id] = 1.0
            return patched
        return mask

    def _apply_action_mask(self, action_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        nvec = self.action_space.nvec
        func_dim = int(nvec[0])
        func_logits = action_logits[:, :func_dim]
        rest_logits = action_logits[:, func_dim:]
        # mask==0 -> set logit to large negative
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        mask = self._with_no_op_fallback(mask.to(action_logits.device))
        func_logits = func_logits + (mask <= 0).float() * (-1e9)
        return torch.cat([func_logits, rest_logits], dim=1)

    def _get_masked_distribution(self, obs):
        latent_pi, _, spatial_maps = self._extract_latents(obs)
        # The distribution keeps its own reference for log_prob/entropy; do not
        # leave a second, stale reference on the extractor.
        self._clear_last_spatial_maps()
        base_logits = self._base_action_logits(latent_pi, obs)
        policy = self

        class ConditionalJointDistribution:
            def __init__(self):
                self.logits = base_logits
                self.obs = obs
                self.spatial_maps = spatial_maps

            def get_actions(self, deterministic: bool = False):
                actions, _ = policy._sample_joint_actions(
                    self.logits, self.obs, deterministic, spatial_maps=self.spatial_maps
                )
                return actions

            def sample(self):
                return self.get_actions(deterministic=False)

            def mode(self):
                return self.get_actions(deterministic=True)

            def log_prob(self, actions):
                actions_t = _as_action_tensor(actions, self.logits.device).long()
                if actions_t.dim() == 1:
                    actions_t = actions_t.unsqueeze(0)
                conditioned = policy._condition_argument_logits(
                    self.logits,
                    actions_t[:, 0].long(),
                    self.obs,
                    spatial_maps=self.spatial_maps,
                )
                return policy._joint_logprob_and_entropy(conditioned, actions_t)[0]

            def entropy(self):
                # Exact over the available-function support; see _conditional_joint_entropy.
                return policy._conditional_joint_entropy(
                    self.logits, self.obs, spatial_maps=self.spatial_maps
                )

        return ConditionalJointDistribution()

    def forward(self, obs, deterministic: bool = False):
        latent_pi, latent_vf, spatial_maps = self._extract_latents(obs)
        try:
            values = self.value_net(latent_vf)
            base_logits = self._base_action_logits(latent_pi, obs)
            actions, conditioned_logits = self._sample_joint_actions(
                base_logits, obs, deterministic, spatial_maps=spatial_maps
            )
            log_prob, _ = self._joint_logprob_and_entropy(conditioned_logits, actions)
            return actions, values, log_prob
        finally:
            self._clear_last_spatial_maps()

    def predict_values(self, obs):
        try:
            return super().predict_values(obs)
        finally:
            self._clear_last_spatial_maps()

    def get_distribution(self, obs):
        return self._get_masked_distribution(obs)

    def evaluate_actions(self, obs, actions):
        """Evaluate the same function + spatial distribution used for sampling."""
        latent_pi, latent_vf, spatial_maps = self._extract_latents(obs)
        try:
            values = self.value_net(latent_vf)
            base_logits = self._base_action_logits(latent_pi, obs)
            actions_t = torch.as_tensor(actions, device=base_logits.device).long()
            if actions_t.dim() == 1:
                actions_t = actions_t.unsqueeze(0)
            conditioned_logits = self._condition_argument_logits(
                base_logits, actions_t[:, 0].long(), obs, spatial_maps=spatial_maps
            )
            log_prob, _ = self._joint_logprob_and_entropy(conditioned_logits, actions_t)
            entropy = self._conditional_joint_entropy(
                base_logits, obs, spatial_maps=spatial_maps
            )
            return values, log_prob, entropy
        finally:
            self._clear_last_spatial_maps()
