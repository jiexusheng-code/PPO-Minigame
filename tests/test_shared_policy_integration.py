"""Offline parser -> policy -> PPO regression checks; no SC2 process is started."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pytest
import torch
from pysc2.lib import actions, features
from stable_baselines3 import PPO

from src.obs_parser import AVAILABLE_MAPS, ObsParser
from src.policies.masked_flatten_policy import MaskedFlattenPolicy, VectorLayerNormExtractor


def _action_space():
    semantics = []
    sizes = []
    for fn in actions.FUNCTIONS:
        for arg in fn.args:
            if arg.name in semantics:
                continue
            semantics.append(arg.name)
            sizes.append(32 * 32 if arg.name in ("screen", "screen2", "minimap") else int(arg.sizes[0]))
    return gym.spaces.MultiDiscrete([len(actions.FUNCTIONS)] + sizes), semantics


def _observation(map_name):
    screen = np.zeros((len(features.SCREEN_FEATURES), 32, 32), dtype=np.int32)
    minimap = np.zeros((len(features.MINIMAP_FEATURES), 32, 32), dtype=np.int32)
    screen[features.SCREEN_FEATURES.player_relative.index, 12:15, 10:13] = 1
    screen[features.SCREEN_FEATURES.selected.index, 12:15, 10:13] = 1
    minimap[features.MINIMAP_FEATURES.selected.index, 12:15, 10:13] = 1
    return ObsParser(map_name).parse({
        "player": np.array([1, 50, 0, 3, 15, 3, 0, 0, 3, 0, 0], dtype=np.int32),
        "feature_screen": screen,
        "feature_minimap": minimap,
        "available_actions": [actions.FUNCTIONS.no_op.id, actions.FUNCTIONS.select_rect.id,
                              actions.FUNCTIONS.Move_screen.id, actions.FUNCTIONS.Move_minimap.id],
    })


class _OfflineEnv(gym.Env):
    def __init__(self, map_name="CollectMineralShards"):
        self.obs = _observation(map_name)
        self.observation_space = gym.spaces.Dict({
            key: gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
            for key, value in self.obs.items()
        })
        self.action_space, self.semantics = _action_space()
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return {key: value.copy() for key, value in self.obs.items()}, {}

    def step(self, action):
        assert self.action_space.contains(action)
        for index, name in enumerate(self.semantics):
            if name not in ("screen", "minimap"):
                assert action[index + 1] == 0
        self.steps += 1
        return ({key: value.copy() for key, value in self.obs.items()},
                float(action[0] == actions.FUNCTIONS.Move_screen.id), False, self.steps >= 4, {})


@pytest.mark.parametrize("map_name", list(AVAILABLE_MAPS))
def test_seven_map_parser_outputs_work_with_shared_policy(map_name):
    env = _OfflineEnv(map_name)
    policy = MaskedFlattenPolicy(
        env.observation_space, env.action_space, lambda _: 3e-4,
        features_extractor_class=VectorLayerNormExtractor,
        features_extractor_kwargs={"cnn_output_dim": 8},
        net_arch=dict(pi=[16], vf=[16]),
    )
    obs = {key: torch.as_tensor(value).unsqueeze(0) for key, value in env.obs.items()}
    with torch.no_grad():
        sampled, values, log_prob = policy(obs)
        evaluated_values, evaluated_log_prob, entropy = policy.evaluate_actions(obs, sampled)
    assert sampled.shape == (1, len(env.action_space.nvec))
    assert torch.isfinite(values).all() and torch.isfinite(entropy).all()
    torch.testing.assert_close(log_prob, evaluated_log_prob)
    torch.testing.assert_close(values, evaluated_values)
    env.step(sampled[0].cpu().numpy())


def test_offline_ppo_updates_and_new_checkpoint_roundtrip(tmp_path):
    env = _OfflineEnv()
    model = PPO(
        MaskedFlattenPolicy, env, device="cpu", seed=7,
        n_steps=8, batch_size=4, n_epochs=1,
        policy_kwargs={"features_extractor_class": VectorLayerNormExtractor,
                       "features_extractor_kwargs": {"cnn_output_dim": 8},
                       "net_arch": dict(pi=[16], vf=[16])},
    )
    before = {name: param.detach().clone() for name, param in model.policy.named_parameters()}
    model.learn(total_timesteps=16)
    assert any(not torch.equal(before[name], param) for name, param in model.policy.named_parameters())
    observation, _ = env.reset()
    predicted, _ = model.predict(observation, deterministic=True)
    path = tmp_path / "shared_policy.zip"
    model.save(path)
    # This archive was generated in this test, not loaded from external training data.
    restored = PPO.load(path, env=env, device="cpu")
    reloaded, _ = restored.predict(observation, deterministic=True)
    np.testing.assert_array_equal(predicted, reloaded)
    assert model.num_timesteps == restored.num_timesteps == 16
    assert all(torch.isfinite(param).all() for param in restored.policy.parameters())
    env.close()
