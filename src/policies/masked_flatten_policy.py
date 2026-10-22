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


class _ResumeTolerantAdam(torch.optim.Adam):
    """Adam that can restore checkpoints saved before new parameters existed.

    Only the default ``torch.optim.Adam`` is substituted with this class. The
    padding assumes new parameters are appended after every historical
    parameter in the same parameter group(s); a refactor that reorders modules
    would invalidate historical Adam indices. Unsupported layouts fail loudly
    instead of guessing a mapping.
    """

    def load_state_dict(self, state_dict):
        saved_groups = state_dict.get("param_groups", [])
        if saved_groups and len(saved_groups) != len(self.param_groups):
            raise ValueError(
                "_ResumeTolerantAdam cannot resume optimizer state with "
                f"{len(saved_groups)} parameter groups into "
                f"{len(self.param_groups)} groups; refusing to guess a mapping."
            )
        if saved_groups:
            saved_lens = [len(group["params"]) for group in saved_groups]
            current_lens = [len(group["params"]) for group in self.param_groups]
            if any(s > c for s, c in zip(saved_lens, current_lens)):
                raise ValueError(
                    "Saved optimizer has more parameters than the current policy; "
                    "refusing to guess a mapping."
                )
            needs_padding = any(s < c for s, c in zip(saved_lens, current_lens))
            if needs_padding:
                state_dict = dict(state_dict)
                state_dict["param_groups"] = [dict(group) for group in saved_groups]
                offset = 0
                for group, saved_len, current_len in zip(
                    state_dict["param_groups"], saved_lens, current_lens
                ):
                    group["params"] = list(group["params"]) + list(
                        range(offset + saved_len, offset + current_len)
                    )
                    offset += current_len
        return super().load_state_dict(state_dict)


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
            screen_feat = self._embed_spatial(screen, screen_flags, self._screen_cat_info, self._screen_cont_idx, self._screen_embeddings, "screen_layer_flags")
            if screen_feat is not None:
                encoded.append(self._screen_cnn(screen_feat))

        if self._minimap_cnn is not None and "minimap" in observations:
            minimap = observations["minimap"]
            minimap_flags = observations.get("minimap_layer_flags")
            minimap_feat = self._embed_spatial(minimap, minimap_flags, self._minimap_cat_info, self._minimap_cont_idx, self._minimap_embeddings, "minimap_layer_flags")
            if minimap_feat is not None:
                encoded.append(self._minimap_cnn(minimap_feat))

        for key in self._flat_keys:
            if key in observations:
                encoded.append(self._flatten(observations[key]))

        return torch.cat(encoded, dim=1) if len(encoded) > 1 else encoded[0]


class MaskedFlattenPolicy(MultiInputPolicy):
    """Applies action_mask on the first categorical (function id) of a MultiDiscrete action space."""

    def _build(self, lr_schedule) -> None:
        # Spatial action heads must be registered before SB3 builds the
        # optimizer from ``self.parameters()``, otherwise they would be
        # silently excluded from optimization.
        self._build_spatial_heads()
        super()._build(lr_schedule)
        # The function-conditioned head is appended after every historical
        # parameter, so optimizer state saved by pre-conditioning checkpoints
        # keeps its original parameter indices. The optimizer created by
        # ``super()._build`` is replaced afterwards so the new parameters are
        # optimized from the first step.
        self._build_conditioned_heads()
        self._rebuild_optimizer(lr_schedule)

    def _rebuild_optimizer(self, lr_schedule) -> None:
        optimizer_class = self.optimizer_class
        if optimizer_class is torch.optim.Adam:
            # Only the default Adam gets the append-only resume tolerance.
            # Custom optimizer classes are used as-is; resuming a
            # pre-conditioning checkpoint with them may require a compatible
            # state dict supplied by the caller.
            optimizer_class = _ResumeTolerantAdam
        self.optimizer = optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

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
        screen_in_ch = self._spatial_channels(spaces.get("screen"))
        minimap_in_ch = self._spatial_channels(spaces.get("minimap"))

        # simple two-layer conv producing 1-channel spatial logits
        if screen_in_ch is not None and self._screen_slot_idx is not None:
            self._screen_spatial_net = nn.Sequential(
                nn.Conv2d(screen_in_ch, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),
            )
            self._screen_in_ch = screen_in_ch
            # log-scale keeps the multiplier positive and differentiable;
            # initialized to 0 -> scale == 1, matching the previous behavior.
            self._screen_spatial_log_scale = nn.Parameter(torch.zeros(()))
        else:
            self._screen_spatial_net = None
            self._screen_conditional_net = None
            self._screen_in_ch = None

        if minimap_in_ch is not None and self._minimap_slot_idx is not None:
            self._minimap_spatial_net = nn.Sequential(
                nn.Conv2d(minimap_in_ch, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),
            )
            self._minimap_in_ch = minimap_in_ch
            self._minimap_spatial_log_scale = nn.Parameter(torch.zeros(()))
        else:
            self._minimap_spatial_net = None
            self._minimap_conditional_net = None
            self._minimap_in_ch = None

    def _build_conditioned_heads(self) -> None:
        if hasattr(self, "_function_embedding"):
            return
        n_funcs = int(self.action_space.nvec[0])
        latent_dim = int(self.mlp_extractor.latent_dim_pi)
        self._fn_embedding_dim = 16
        self._function_embedding = nn.Embedding(n_funcs, self._fn_embedding_dim)
        self._argument_conditioner = nn.Sequential(
            nn.Linear(latent_dim + self._fn_embedding_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, sum(self._slot_sizes)),
        )
        # Zero-init the residual outputs so checkpoints saved before the
        # conditional head existed keep exactly their previous behavior when
        # loaded. The last layer's own gradient is non-zero on the first step,
        # after which the conditioning path becomes active.
        nn.init.zeros_(self._argument_conditioner[-1].weight)
        nn.init.zeros_(self._argument_conditioner[-1].bias)
        if self._screen_spatial_net is not None:
            self._screen_conditional_net = nn.Sequential(
                nn.Conv2d(self._screen_in_ch + self._fn_embedding_dim, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),
            )
            nn.init.zeros_(self._screen_conditional_net[-1].weight)
            nn.init.zeros_(self._screen_conditional_net[-1].bias)
        if self._minimap_spatial_net is not None:
            self._minimap_conditional_net = nn.Sequential(
                nn.Conv2d(self._minimap_in_ch + self._fn_embedding_dim, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=1),
            )
            nn.init.zeros_(self._minimap_conditional_net[-1].weight)
            nn.init.zeros_(self._minimap_conditional_net[-1].bias)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load pre-conditioning checkpoints with zero-effect initialized additions."""
        current = self.state_dict()
        extended = state_dict.copy()
        new_prefixes = (
            "_function_embedding.",
            "_argument_conditioner.",
            "_screen_conditional_net.",
            "_minimap_conditional_net.",
        )
        for key, value in current.items():
            if key.startswith(new_prefixes) and key not in extended:
                extended[key] = value
        if hasattr(state_dict, "_metadata"):
            extended._metadata = state_dict._metadata
        return super().load_state_dict(extended, strict=strict, assign=assign)

    @staticmethod
    def _spatial_channels(space):
        if space is None:
            return None
        shape = getattr(space, "shape", None)
        if shape is None or len(shape) != 3:
            return None
        # observation space is (H, W, C)
        return int(shape[2])

    @staticmethod
    def _with_no_op_fallback(mask: torch.Tensor) -> torch.Tensor:
        """Allow ``no_op`` for rows whose available-action mask is all zero.

        PySC2 normally guarantees at least one available action, but a malformed
        or synthetic all-zero row would otherwise silently become uniform over
        every PySC2 function (all function logits are masked equally). Sampling,
        evaluation and entropy must share one support, so replace that
        degenerate row with the same legal ``no_op`` fallback the env uses.
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

    def _build_slot_map_if_needed(self):
        # Slot mask is built eagerly in _build_spatial_heads; this method only
        # performs the one-time environment snapshot cross-check.
        if getattr(self, "_slot_map_built", False):
            return
        if not hasattr(self, "_param_semantics"):
            # Safety net for construction paths that bypass _build().
            self._build_spatial_heads()

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

    def _condition_argument_logits(
        self, base_logits, latent_pi, fn_ids, obs, used_slots=None, cache=None
    ):
        """Return argument logits conditioned on each sample's selected function.

        ``used_slots``/``cache`` are performance hints for the exact-entropy
        path: they skip spatial heads that the function never uses and reuse
        observation features across function branches. Default behavior
        (both ``None``) is unchanged.
        """
        self._build_slot_map_if_needed()
        sizes = [int(x) for x in self.action_space.nvec]
        func_dim = sizes[0]
        safe_fn_ids = fn_ids.to(device=latent_pi.device, dtype=torch.long).clamp(
            min=0, max=self._n_funcs - 1
        )
        embeddings = self._function_embedding(safe_fn_ids)
        conditioned = torch.cat([latent_pi, embeddings], dim=1)
        residual = self._argument_conditioner(conditioned)
        base_args = base_logits[:, func_dim:]
        logits = torch.cat([base_logits[:, :func_dim], base_args + residual], dim=1)
        return self._inject_spatial_logits(
            logits, obs, fn_ids=safe_fn_ids, used_slots=used_slots, cache=cache
        )

    def _sample_joint_actions(self, base_logits, latent_pi, obs, deterministic=False):
        sizes = [int(x) for x in self.action_space.nvec]
        fn_logits = torch.split(base_logits, sizes, dim=1)[0]
        fn_dist = torch.distributions.Categorical(logits=fn_logits)
        fn_ids = fn_logits.argmax(dim=1) if deterministic else fn_dist.sample()
        conditioned = self._condition_argument_logits(base_logits, latent_pi, fn_ids, obs)
        slot_logits = torch.split(conditioned, sizes, dim=1)[1:]
        actions = torch.zeros((base_logits.shape[0], len(sizes)), dtype=torch.long, device=base_logits.device)
        actions[:, 0] = fn_ids
        used = self._fn_slot_mask[fn_ids]
        for i, slot_logits_i in enumerate(slot_logits):
            rows = torch.nonzero(used[:, i] > 0, as_tuple=False).squeeze(1)
            if rows.numel() == 0:
                continue
            if deterministic:
                values = slot_logits_i[rows].argmax(dim=1)
            else:
                values = torch.distributions.Categorical(logits=slot_logits_i[rows]).sample()
            actions[rows, i + 1] = values
        return actions, conditioned

    def _conditional_joint_entropy(self, base_logits, latent_pi, obs):
        """Exact joint conditional entropy of the hierarchical policy.

        The policy factorizes as ``pi(fn|s) * prod_{slot in used(fn)} pi(slot|fn, s)``,
        so its entropy is

            H(fn|s) + sum_fn pi(fn|s) * sum_{slot in used(fn)} H(slot|fn, s).

        The sum over functions is exact: it runs over the union of functions
        permitted by ``obs['available_actions']``. Functions excluded by the
        mask have probability zero after ``_apply_action_mask``, so they cannot
        contribute. Enumerating that (small) support avoids the prohibitive cost
        of running one spatial conv pass for all 573 PySC2 functions.

        To keep the enumeration cheap while remaining exact, observation
        preprocessing and the observation-only base spatial conv are cached
        once per call, and functions that do not use a spatial slot skip that
        slot's conditional head. Functions with no arguments are skipped
        entirely because their argument entropy is zero by definition.
        """
        sizes = [int(x) for x in self.action_space.nvec]
        fn_logits = torch.split(base_logits, sizes, dim=1)[0]
        fn_logp = F.log_softmax(fn_logits, dim=1)
        fn_probs = fn_logp.exp()
        fn_entropy = -(fn_probs * fn_logp).sum(dim=1)

        available = obs.get("available_actions") if isinstance(obs, dict) else None
        if available is not None:
            mask = torch.as_tensor(available, device=base_logits.device)
            # Keep the support identical to sampling/evaluation: an all-zero
            # row is treated as ``no_op``-only rather than uniformly over all
            # functions. The function probs below are already masked the same
            # way, so rows using the fallback contribute zero argument entropy.
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
        # Observation preprocessing and the base spatial conv do not depend on
        # the function id, so compute them once and reuse them for every branch.
        spatial_cache = {}
        for fn_id in fn_ids.tolist():
            weight = fn_probs[:, fn_id]
            if not torch.any(weight > 0):
                continue
            fn_slots = self._fn_slot_mask[fn_id]
            if not bool(fn_slots.any()):
                # No arguments: this branch contributes no argument entropy.
                continue
            branch_ids = torch.full(
                (base_logits.shape[0],), fn_id, dtype=torch.long, device=base_logits.device
            )
            used_slots = torch.nonzero(fn_slots > 0, as_tuple=False).squeeze(1).tolist()
            conditioned = self._condition_argument_logits(
                base_logits,
                latent_pi,
                branch_ids,
                obs,
                used_slots=used_slots,
                cache=spatial_cache,
            )
            slot_logits = torch.split(conditioned, sizes, dim=1)[1:]
            used = self._fn_slot_mask[branch_ids]
            branch_entropy = torch.zeros_like(fn_entropy)
            for slot_idx, slot_logits_i in enumerate(slot_logits):
                slot_logp = F.log_softmax(slot_logits_i, dim=1)
                branch_entropy = branch_entropy + (
                    -(slot_logp.exp() * slot_logp).sum(dim=1) * used[:, slot_idx]
                )
            arg_expectation = arg_expectation + weight * branch_entropy
        return fn_entropy + arg_expectation

    def _joint_logprob_and_entropy(self, logits: torch.Tensor, actions: torch.Tensor):
        """Compute joint log_prob and entropy while ignoring unused arg slots per-sample.

        logits: (B, sum(nvec))
        actions: (B, n_actions)  # first col is fn_id, following are slot values (can be -1)
        Returns: log_prob (B,), entropy (B,)
        """
        self._build_slot_map_if_needed()
        nvec = list(self.action_space.nvec)
        # split logits per head
        sizes = [int(x) for x in nvec]
        splits = torch.split(logits, sizes, dim=1)
        fn_logits = splits[0]
        slot_logits = splits[1:]

        batch_size = logits.shape[0]

        # function log_prob
        fn_logp = F.log_softmax(fn_logits, dim=1)
        fn_ids = actions[:, 0].long()
        fn_selected = fn_logp.gather(1, fn_ids.clamp(min=0, max=fn_logits.shape[1]-1).unsqueeze(1)).squeeze(1)

        # per-slot log probs and entropy
        slot_logps = []
        slot_ents = []
        # actions for slots are in actions[:, 1:]
        slot_actions = actions[:, 1:]

        for i, s_logits in enumerate(slot_logits):
            # s_logits: (B, size)
            size = s_logits.shape[1]
            probs = F.softmax(s_logits, dim=1)
            logp = F.log_softmax(s_logits, dim=1)
            # gather selected indices, clamp to valid range
            a_i = slot_actions[:, i].long()
            gather_idx = a_i.clamp(min=0, max=size-1).unsqueeze(1)
            picked = logp.gather(1, gather_idx).squeeze(1)
            # used mask from fn mapping
            # fn_slot_mask: (n_funcs, n_slots)
            fn_mask = self._fn_slot_mask[fn_ids]  # (B, n_slots)
            used = fn_mask[:, i]
            picked = picked * used
            slot_logps.append(picked)

            ent = - (probs * logp).sum(dim=1)
            ent = ent * used
            slot_ents.append(ent)

        total_logp = fn_selected + sum(slot_logps) if len(slot_logps) > 0 else fn_selected
        total_ent = (- (F.softmax(fn_logits, dim=1) * F.log_softmax(fn_logits, dim=1))).sum(dim=1)
        if len(slot_ents) > 0:
            total_ent = total_ent + sum(slot_ents)

        # Cache per-arg statistics for external callbacks/monitoring
        try:
            # fn entropy mean
            fn_probs = F.softmax(fn_logits, dim=1)
            fn_entropy_per_sample = - (fn_probs * F.log_softmax(fn_logits, dim=1)).sum(dim=1)
            fn_entropy_mean = float(fn_entropy_per_sample.mean().detach().cpu().item())

            slot_used_means = []
            slot_entropy_means = []
            slot_logprob_means = []
            # slot_ents contains ent * used already; need used statistics per slot
            for i, s_logits in enumerate(slot_logits):
                size = s_logits.shape[1]
                # raw entropy per-sample for this slot
                raw_ent = - (F.softmax(s_logits, dim=1) * F.log_softmax(s_logits, dim=1)).sum(dim=1)
                # used mask for this slot across batch
                fn_mask = self._fn_slot_mask[fn_ids]  # (B, n_slots)
                used = fn_mask[:, i]
                used_sum = float(used.sum().detach().cpu().item())
                used_mean = float(used.mean().detach().cpu().item()) if used.numel() > 0 else 0.0
                if used_sum > 0:
                    ent_mean = float((raw_ent * used).sum().detach().cpu().item() / (used_sum + 1e-8))
                else:
                    ent_mean = 0.0
                slot_used_means.append(used_mean)
                slot_entropy_means.append(ent_mean)
                # compute selected log-prob mean for this slot (only over used entries)
                try:
                    # slot_logps list contains per-sample picked*used; recover corresponding tensor
                    picked_vals = slot_logps[i]
                    if used_sum > 0:
                        logp_mean = float(picked_vals.sum().detach().cpu().item() / (used_sum + 1e-8))
                    else:
                        logp_mean = 0.0
                except Exception:
                    logp_mean = 0.0
                slot_logprob_means.append(logp_mean)

            self._last_per_arg_stats = {
                "fn_entropy_mean": fn_entropy_mean,
                "fn_logprob_mean": float(fn_selected.mean().detach().cpu().item()) if isinstance(fn_selected, torch.Tensor) else 0.0,
                "slot_used_mean": slot_used_means,
                "slot_entropy_mean": slot_entropy_means,
                "slot_logprob_mean": slot_logprob_means,
            }
        except Exception:
            # non-critical: do not break training if stats computation fails
            self._last_per_arg_stats = None

        return total_logp, total_ent

    def _joint_entropy_from_logits(self, logits: torch.Tensor, latent_pi=None, obs=None):
        """Entropy from an already-materialized logits tensor.

        Without ``latent_pi``/``obs`` this is the legacy exact formula for a
        fixed logits tensor: H(fn) + sum_i E[used_i] * H(arg_i). With them, the
        logits are treated as the base (unconditioned) logits and the result is
        the exact function-conditioned joint entropy.
        """
        if latent_pi is not None and obs is not None:
            return self._conditional_joint_entropy(logits, latent_pi, obs)
        device = logits.device
        self._build_slot_map_if_needed()
        nvec = list(self.action_space.nvec)
        sizes = [int(x) for x in nvec]
        splits = torch.split(logits, sizes, dim=1)
        fn_logits = splits[0]
        slot_logits = splits[1:]

        fn_probs = F.softmax(fn_logits, dim=1)
        fn_entropy = - (fn_probs * F.log_softmax(fn_logits, dim=1)).sum(dim=1)

        # compute each slot entropy (B,)
        slot_entropies = []
        for s_logits in slot_logits:
            ent = - (F.softmax(s_logits, dim=1) * F.log_softmax(s_logits, dim=1)).sum(dim=1)
            slot_entropies.append(ent)

        if len(slot_entropies) == 0:
            return fn_entropy

        # expected usage per slot: (B, n_slots) = fn_probs (B, n_funcs) @ fn_slot_mask (n_funcs, n_slots)
        expected_usage = fn_probs.matmul(self._fn_slot_mask.to(device))

        # combine: total_ent = fn_entropy + sum_i expected_usage[:,i] * slot_entropies[i]
        total = fn_entropy
        for i, ent in enumerate(slot_entropies):
            total = total + expected_usage[:, i] * ent
        return total

    def _inject_spatial_logits(self, logits: torch.Tensor, obs, fn_ids=None, used_slots=None, cache=None):
        """Replace flat slot logits with spatial-head logits computed from obs['screen']/['minimap'].

        This keeps the external MultiDiscrete action interface but changes how spatial slot
        logits are produced (from a spatial conv map flattened to match slot size).
        When ``fn_ids`` is given, an additional conv conditioned on the selected
        function's embedding is added to each spatial map. ``fn_ids=None`` keeps
        the historical observation-only behavior (used by legacy callers/tests).

        ``used_slots`` optionally restricts which spatial heads are evaluated.
        Skipped slots retain their flat logits and must be unused by the caller
        (the conditional entropy path ignores them). ``cache`` optionally reuses
        preprocessed observations and base spatial maps across per-function
        passes while keeping the conditional conv per function.
        """
        if not isinstance(logits, torch.Tensor):
            return logits
        device = logits.device
        sizes = [int(x) for x in self.action_space.nvec]
        func_dim = int(sizes[0])
        slot_sizes = [int(x) for x in sizes[1:]]
        used = None if used_slots is None else {int(slot) for slot in used_slots}

        # helper: compute the spatial logits for a single head
        def _compute(net, conditional_net, slot_idx, in_ch, key, flags_key, log_scale):
            if net is None or slot_idx is None or key not in obs:
                return None
            if used is not None and slot_idx not in used:
                return None
            cached = cache.get(key) if cache is not None else None
            if cached is None:
                x = obs[key]
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x, device=device)
                x = x.float().to(device)
                if x.dim() != 4:
                    raise ValueError(
                        f"obs['{key}'] must be 4D (B,H,W,C) or (B,C,H,W), got shape {tuple(x.shape)}"
                    )
                # accept (B,H,W,C) or (B,C,H,W)
                if x.shape[-1] == in_ch:
                    x_t = x.permute(0, 3, 1, 2)
                elif x.shape[1] == in_ch:
                    x_t = x
                else:
                    raise ValueError(
                        f"obs['{key}'] channel count does not match head input: "
                        f"leading={x.shape[1]}, trailing={x.shape[-1]}, expected {in_ch}"
                    )
                flags = obs.get(flags_key)
                if flags is not None:
                    if not isinstance(flags, torch.Tensor):
                        flags = torch.as_tensor(flags, device=device)
                    flags = _resolve_channel_flags(flags.float().to(device), in_ch, flags_key)
                    # per-channel layer flags, not an element-wise spatial mask
                    x_t = x_t * flags[:, :, None, None]
                base_map = net(x_t)
                if cache is not None:
                    cache[key] = (x_t, base_map)
            else:
                x_t, base_map = cached
            out_map = base_map
            if conditional_net is not None and fn_ids is not None:
                ids = fn_ids.to(device=device, dtype=torch.long)
                embedding = self._function_embedding(ids)
                embedding_map = embedding[:, :, None, None].expand(
                    -1, -1, x_t.shape[2], x_t.shape[3]
                )
                conditioned_map = conditional_net(torch.cat([x_t, embedding_map], dim=1))
                out_map = base_map + conditioned_map
            out_flat = out_map.flatten(1)
            if log_scale is not None:
                # scale = exp(log_scale): positive, differentiable, initialized to 1.
                out_flat = out_flat * torch.exp(log_scale.clamp(-6.0, 6.0))
            slot_size = slot_sizes[slot_idx]
            if out_flat.shape[1] != slot_size:
                raise ValueError(
                    f"Spatial logits for '{key}' contain {out_flat.shape[1]} cells, "
                    f"but the action slot has size {slot_size}."
                )
            return out_flat

        screen_flat = _compute(
            getattr(self, '_screen_spatial_net', None),
            getattr(self, '_screen_conditional_net', None),
            getattr(self, '_screen_slot_idx', None),
            getattr(self, '_screen_in_ch', None),
            'screen',
            'screen_layer_flags',
            getattr(self, '_screen_spatial_log_scale', None),
        )
        minimap_flat = _compute(
            getattr(self, '_minimap_spatial_net', None),
            getattr(self, '_minimap_conditional_net', None),
            getattr(self, '_minimap_slot_idx', None),
            getattr(self, '_minimap_in_ch', None),
            'minimap',
            'minimap_layer_flags',
            getattr(self, '_minimap_spatial_log_scale', None),
        )

        # Build the result out-of-place: in-place writes into a tensor that is
        # part of the autograd graph are fragile and unnecessary here.
        pieces = [logits[:, :func_dim]]
        cursor = func_dim
        for slot_idx, slot_size in enumerate(slot_sizes):
            end = cursor + slot_size
            if slot_idx == self._screen_slot_idx and screen_flat is not None:
                pieces.append(screen_flat)
            elif slot_idx == self._minimap_slot_idx and minimap_flat is not None:
                pieces.append(minimap_flat)
            else:
                pieces.append(logits[:, cursor:end])
            cursor = end
        return torch.cat(pieces, dim=1)

    def _base_action_logits(self, latent_pi, obs):
        logits = self.action_net(latent_pi)
        if isinstance(self.action_dist, MultiCategoricalDistribution) and "available_actions" in obs:
            logits = self._apply_action_mask(logits, obs["available_actions"])
        return logits

    def _get_masked_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        base_logits = self._base_action_logits(latent_pi, obs)
        policy = self

        class ConditionalJointDistribution:
            def __init__(self):
                self.logits = base_logits
                self.latent_pi = latent_pi
                self.obs = obs

            def get_actions(self, deterministic: bool = False):
                actions, _ = policy._sample_joint_actions(
                    self.logits, self.latent_pi, self.obs, deterministic
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
                    self.logits, self.latent_pi, actions_t[:, 0].long(), self.obs
                )
                return policy._joint_logprob_and_entropy(conditioned, actions_t)[0]

            def entropy(self):
                # Exact over the available-function support; see _conditional_joint_entropy.
                return policy._conditional_joint_entropy(self.logits, self.latent_pi, self.obs)

        return ConditionalJointDistribution()

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        base_logits = self._base_action_logits(latent_pi, obs)
        actions, conditioned_logits = self._sample_joint_actions(
            base_logits, latent_pi, obs, deterministic
        )
        log_prob, _ = self._joint_logprob_and_entropy(conditioned_logits, actions)
        return actions, values, log_prob

    def get_distribution(self, obs):
        return self._get_masked_distribution(obs)

    def evaluate_actions(self, obs, actions):
        """Evaluate the same function-conditioned joint distribution used for sampling."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        base_logits = self._base_action_logits(latent_pi, obs)
        actions_t = torch.as_tensor(actions, device=base_logits.device).long()
        if actions_t.dim() == 1:
            actions_t = actions_t.unsqueeze(0)
        conditioned_logits = self._condition_argument_logits(
            base_logits, latent_pi, actions_t[:, 0].long(), obs
        )
        log_prob, _ = self._joint_logprob_and_entropy(conditioned_logits, actions_t)
        entropy = self._conditional_joint_entropy(base_logits, latent_pi, obs)
        return values, log_prob, entropy
