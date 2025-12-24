"""MultiInput policy with action masking on function id and flatten coordinate heads via MultiDiscrete action space."""
import torch
from torch import nn

from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.ppo.policies import MultiInputPolicy


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
        features, _ = self.extract_features(obs)
        latent_pi, latent_vf = self._get_latent(features)
        values = self.value_net(latent_vf)
        logits = self.action_net(latent_pi)

        if isinstance(self.action_dist, MultiCategoricalDistribution) and "action_mask" in obs:
            logits = self._apply_action_mask(logits, obs["action_mask"])

        distribution = self._get_action_dist_from_latent(logits, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
