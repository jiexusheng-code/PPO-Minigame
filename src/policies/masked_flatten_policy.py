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
        # 1. 特征提取
        features = self.extract_features(obs)
        # 懒初始化 LayerNorm：按特征维度归一化，减轻不同输入尺度的影响
        if not hasattr(self, "_feat_layernorm"):
            self._feat_layernorm = nn.LayerNorm(features.shape[1])
            try:
                self._feat_layernorm.to(features.device)
            except Exception:
                pass
        features = self._feat_layernorm(features)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        logits = self.action_net(latent_pi)

        # 2. 只对第一个头(fn_id)做mask，其他参数槽位严格一一对应
        if isinstance(self.action_dist, MultiCategoricalDistribution) and "action_mask" in obs:
            logits = self._apply_action_mask(logits, obs["action_mask"])

        # 3. 构造分布并采样，确保每个头只采样唯一语义参数
        distribution = self.action_dist.proba_distribution(logits)
        actions = distribution.get_actions(deterministic=deterministic)

        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob
