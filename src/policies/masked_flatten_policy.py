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

    def _build_slot_map_if_needed(self, device=None):
        # Build param semantics and fn->slot mapping consistent with PySC2GymEnv
        if getattr(self, "_slot_map_built", False):
            return
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
            slot_indices = []
            for spec in fn.args:
                name = getattr(spec, "name")
                slot_indices.append(param_semantics.index(name))
            fn_param_map[fn.id] = slot_indices

        # slot sizes come from action_space.nvec (skip first entry which is fn head)
        nvec = list(self.action_space.nvec)
        slot_sizes = nvec[1:]

        # build fn -> binary mask over slots tensor for fast indexing
        n_funcs = int(nvec[0])
        n_slots = len(slot_sizes)
        fn_slot_mask = torch.zeros((n_funcs, n_slots), dtype=torch.float32)
        for fid, slots in fn_param_map.items():
            for s in slots:
                if s < n_slots:
                    fn_slot_mask[fid, s] = 1.0

        self._param_semantics = param_semantics
        self._fn_param_map = fn_param_map
        self._slot_sizes = slot_sizes
        self._fn_slot_mask = fn_slot_mask.to(device) if device is not None else fn_slot_mask
        self._slot_map_built = True
        # One-time cross-check against environment snapshot (if present)
        try:
            import json, os, logging
            if not getattr(self, '_param_semantics_checked', False):
                env_path = os.path.join(os.getcwd(), 'param_semantics_env.json')
                if os.path.isfile(env_path):
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
        # Build spatial action heads (minimal: small conv -> 1-channel spatial logits)
        try:
            obs_space = getattr(self, 'observation_space', None)
            if obs_space is not None and hasattr(obs_space, 'spaces'):
                # screen
                screen_space = obs_space.spaces.get('screen')
                if screen_space is not None:
                    sh = screen_space.shape
                    if len(sh) == 3:
                        # (H, W, C)
                        screen_in_ch = int(sh[2])
                    else:
                        screen_in_ch = None
                else:
                    screen_in_ch = None

                minimap_space = obs_space.spaces.get('minimap')
                if minimap_space is not None:
                    mh = minimap_space.shape
                    if len(mh) == 3:
                        minimap_in_ch = int(mh[2])
                    else:
                        minimap_in_ch = None
                else:
                    minimap_in_ch = None

                # find indices of 'screen'/'minimap' in slot semantics
                try:
                    self._screen_slot_idx = self._param_semantics.index('screen') if 'screen' in self._param_semantics else None
                except Exception:
                    self._screen_slot_idx = None
                try:
                    self._minimap_slot_idx = self._param_semantics.index('minimap') if 'minimap' in self._param_semantics else None
                except Exception:
                    self._minimap_slot_idx = None

                import torch.nn as nn
                # create small conv heads when channels and slot indices are available
                if screen_in_ch is not None and self._screen_slot_idx is not None:
                    # simple two-layer conv producing 1-channel spatial logits
                    self._screen_spatial_net = nn.Sequential(
                        nn.Conv2d(screen_in_ch, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 1, kernel_size=1)
                    )
                    self._screen_in_ch = screen_in_ch
                    try:
                        # learnable scale for spatial logits to control softmax sharpness
                        self._screen_spatial_scale = nn.Parameter(torch.tensor(1.0))
                    except Exception:
                        self._screen_spatial_scale = None
                else:
                    self._screen_spatial_net = None
                    self._screen_in_ch = None

                if minimap_in_ch is not None and self._minimap_slot_idx is not None:
                    self._minimap_spatial_net = nn.Sequential(
                        nn.Conv2d(minimap_in_ch, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(32, 1, kernel_size=1)
                    )
                    self._minimap_in_ch = minimap_in_ch
                    try:
                        self._minimap_spatial_scale = nn.Parameter(torch.tensor(1.0))
                    except Exception:
                        self._minimap_spatial_scale = None
                else:
                    self._minimap_spatial_net = None
                    self._minimap_in_ch = None
        except Exception:
            # non-critical
            self._screen_spatial_net = None
            self._minimap_spatial_net = None
            self._screen_slot_idx = None
            self._minimap_slot_idx = None

    def _joint_logprob_and_entropy(self, logits: torch.Tensor, actions: torch.Tensor):
        """Compute joint log_prob and entropy while ignoring unused arg slots per-sample.

        logits: (B, sum(nvec))
        actions: (B, n_actions)  # first col is fn_id, following are slot values (can be -1)
        Returns: log_prob (B,), entropy (B,)
        """
        device = logits.device
        self._build_slot_map_if_needed(device=device)
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

    def _joint_entropy_from_logits(self, logits: torch.Tensor):
        """Compute per-sample entropy of the joint action distribution given logits.

        Uses expected slot usage under fn distribution: E[entropy] = H(fn) + sum_i E[used_i] * H(arg_i)
        """
        device = logits.device
        self._build_slot_map_if_needed(device=device)
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

    def _inject_spatial_logits(self, logits: torch.Tensor, obs):
        """Replace flat slot logits with spatial-head logits computed from obs['screen']/['minimap'].

        This keeps the external MultiDiscrete action interface but changes how spatial slot
        logits are produced (from a spatial conv map flattened to match slot size).
        """
        import torch
        if not isinstance(logits, torch.Tensor):
            return logits
        device = logits.device
        sizes = list(self.action_space.nvec)
        func_dim = int(sizes[0])
        slot_sizes = [int(x) for x in sizes[1:]]

        # helper: inject for a single head
        def _inject(net, slot_idx, in_ch, key):
            if net is None or slot_idx is None:
                return
            if key not in obs:
                return
            x = obs[key]
            if not isinstance(x, torch.Tensor):
                try:
                    x = torch.as_tensor(x, device=device)
                except Exception:
                    return
            x = x.float().to(device)
            if x.dim() != 4:
                return
            # accept (B,H,W,C) or (B,C,H,W)
            if x.shape[-1] == in_ch:
                x_t = x.permute(0, 3, 1, 2)
            elif x.shape[1] == in_ch:
                x_t = x
            else:
                return
            try:
                out_map = net(x_t)  # (B,1,H,W)
                out_flat = out_map.view(out_map.shape[0], -1)
            except Exception:
                return
            # apply learnable scale if present to control numeric magnitude
            try:
                if slot_idx is not None:
                    if key == 'screen' and getattr(self, '_screen_spatial_scale', None) is not None:
                        scale = getattr(self, '_screen_spatial_scale')
                        out_flat = out_flat * float(scale)
                    if key == 'minimap' and getattr(self, '_minimap_spatial_scale', None) is not None:
                        scale = getattr(self, '_minimap_spatial_scale')
                        out_flat = out_flat * float(scale)
            except Exception:
                pass
            slot_size = slot_sizes[slot_idx]
            if out_flat.shape[1] != slot_size:
                if out_flat.shape[1] > slot_size:
                    out_flat = out_flat[:, :slot_size]
                else:
                    out_flat = F.pad(out_flat, (0, slot_size - out_flat.shape[1]))
            start = func_dim + sum(slot_sizes[:slot_idx])
            end = start + slot_size
            logits[:, start:end] = out_flat

        try:
            _inject(getattr(self, '_screen_spatial_net', None), getattr(self, '_screen_slot_idx', None), getattr(self, '_screen_in_ch', None), 'screen')
            _inject(getattr(self, '_minimap_spatial_net', None), getattr(self, '_minimap_slot_idx', None), getattr(self, '_minimap_in_ch', None), 'minimap')
        except Exception:
            pass
        return logits

    def _get_masked_distribution(self, obs):
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        if isinstance(self.action_dist, MultiCategoricalDistribution) and "available_actions" in obs:
            logits = self._apply_action_mask(logits, obs["available_actions"])
        # inject spatial logits (screen/minimap) if available
        try:
            logits = self._inject_spatial_logits(logits, obs)
        except Exception:
            pass
        base_dist = self.action_dist.proba_distribution(logits)

        # Wrapper to override log_prob and entropy using joint per-slot masking
        policy_ref = self

        class JointMaskedDistribution:
            def __init__(self, base, logits, policy):
                self.base = base
                self.logits = logits
                self.policy = policy

            def get_actions(self, deterministic: bool = False):
                return self.base.get_actions(deterministic=deterministic)

            def log_prob(self, actions):
                # actions may be numpy or tensor
                if not isinstance(actions, torch.Tensor):
                    try:
                        actions_t = torch.as_tensor(actions, device=self.logits.device)
                    except Exception:
                        actions_t = torch.tensor(actions, device=self.logits.device)
                else:
                    actions_t = actions.to(self.logits.device)
                logp, _ = self.policy._joint_logprob_and_entropy(self.logits, actions_t)
                return logp

            def entropy(self):
                return self.policy._joint_entropy_from_logits(self.logits)

        return JointMaskedDistribution(base_dist, logits, policy_ref)

    def forward(self, obs, deterministic: bool = False):
        # 1. 特征提取
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        logits = self.action_net(latent_pi)

        # 2. 只对第一个头(fn_id)做mask，其他参数槽位严格一一对应
        if isinstance(self.action_dist, MultiCategoricalDistribution) and "available_actions" in obs:
            logits = self._apply_action_mask(logits, obs["available_actions"])

        # inject spatial logits (screen/minimap) so flat slots for spatial args come
        # from spatial conv maps instead of learned flat heads
        try:
            logits = self._inject_spatial_logits(logits, obs)
        except Exception:
            pass

        # 3. 构造分布并采样，确保每个头只采样唯一语义参数
        distribution = self.action_dist.proba_distribution(logits)
        actions = distribution.get_actions(deterministic=deterministic)

        # ensure actions is a tensor on the same device
        if not isinstance(actions, torch.Tensor):
            try:
                actions_t = torch.as_tensor(actions, device=logits.device)
            except Exception:
                actions_t = torch.tensor(actions, device=logits.device)
        else:
            actions_t = actions.to(logits.device)

        log_prob, _ = self._joint_logprob_and_entropy(logits, actions_t)
        return actions, values, log_prob

    def get_distribution(self, obs):
        return self._get_masked_distribution(obs)

    def evaluate_actions(self, obs, actions):
        """Ensure train-time log_prob/entropy use the same mask as sampling-time."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)

        logits = self.action_net(latent_pi)
        if isinstance(self.action_dist, MultiCategoricalDistribution) and "available_actions" in obs:
            logits = self._apply_action_mask(logits, obs["available_actions"])

        # ensure evaluation uses same injected spatial logits
        try:
            logits = self._inject_spatial_logits(logits, obs)
        except Exception:
            pass

        # compute joint log_prob and entropy using sample-level mask
        if not isinstance(actions, torch.Tensor):
            try:
                actions_t = torch.as_tensor(actions, device=logits.device)
            except Exception:
                actions_t = torch.tensor(actions, device=logits.device)
        else:
            actions_t = actions.to(logits.device)

        log_prob, entropy = self._joint_logprob_and_entropy(logits, actions_t)
        return values, log_prob, entropy
