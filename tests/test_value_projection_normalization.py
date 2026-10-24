"""Value-path regression tests for normalized spatial projections.

These tests do not launch SC2. Observations are produced by the real
``ObsParser`` from synthetic raw PySC2 rasters, so categorical layers only
contain legal integer labels and the CollectMineralShards layer flags/vector
mask follow the project configuration.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from pysc2.lib import actions as pysc2_actions
from pysc2.lib import features

from src.obs_parser import ObsParser
from src.policies.masked_flatten_policy import (
    MaskedFlattenPolicy,
    VectorLayerNormExtractor,
)

SPATIAL_SIZE = 32
SPATIAL_SLOT_SIZE = 1024


def _param_semantics():
    semantics = []
    sizes = []
    seen = set()
    for fn in pysc2_actions.FUNCTIONS:
        for spec in fn.args:
            name = getattr(spec, "name")
            if name in seen:
                continue
            seen.add(name)
            semantics.append(name)
            sizes.append(
                SPATIAL_SLOT_SIZE if name in ("screen", "screen2", "minimap")
                else int(spec.sizes[0])
            )
    return semantics, sizes


def _make_spaces(observation_space_dict):
    _, slot_sizes = _param_semantics()
    n_funcs = len(pysc2_actions.FUNCTIONS)
    action_space = gym.spaces.MultiDiscrete([n_funcs] + slot_sizes)
    observation_space = gym.spaces.Dict(
        {
            key: gym.spaces.Box(-np.inf, np.inf, value.shape[1:], np.float32)
            for key, value in observation_space_dict.items()
        }
    )
    return observation_space, action_space


def _make_policy(observation_space, action_space, cnn_output_dim=256, net_arch=None):
    return MaskedFlattenPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=VectorLayerNormExtractor,
        features_extractor_kwargs={"cnn_output_dim": cnn_output_dim},
        net_arch=net_arch,
    )


def _raw_cms_observation(rng, n_shards, n_marines=2):
    screen = np.zeros((len(features.SCREEN_FEATURES), SPATIAL_SIZE, SPATIAL_SIZE), np.int32)
    minimap = np.zeros((len(features.MINIMAP_FEATURES), SPATIAL_SIZE, SPATIAL_SIZE), np.int32)
    used = set()

    def _cell():
        while True:
            y, x = (int(v) for v in rng.integers(0, SPATIAL_SIZE, 2))
            if (y, x) not in used:
                used.add((y, x))
                return y, x

    for _ in range(n_shards):
        y, x = _cell()
        screen[features.SCREEN_FEATURES.player_relative.index, y, x] = 3
    for i in range(n_marines):
        y, x = _cell()
        screen[features.SCREEN_FEATURES.player_relative.index, y, x] = 1
        is_selected = int(i == 0)
        screen[features.SCREEN_FEATURES.selected.index, y, x] = is_selected
        minimap[features.MINIMAP_FEATURES.selected.index, y, x] = is_selected
    raw = {
        "feature_screen": screen,
        "feature_minimap": minimap,
        "player": np.zeros(11, np.int32),
        # Legal CollectMineralShards function ids only.
        "available_actions": [0, 1, 2, 3, 4, 5, 7, 331],
    }
    return raw


def _parse_cms_batch(n, seed=0, shard_counts=None):
    parser = ObsParser("CollectMineralShards")
    rng = np.random.default_rng(seed)
    parsed = []
    for i in range(n):
        raw = _raw_cms_observation(
            rng, n_shards=12 if shard_counts is None else shard_counts[i]
        )
        parsed.append(parser.parse(raw))
    batch = {key: np.stack([p[key] for p in parsed]) for key in parsed[0]}
    # "vector 全关闭" comes from the CMS parser configuration; assert it
    # instead of overwriting it so the test exercises the real semantics.
    assert np.all(batch["vector_mask"] == 0.0)
    return batch, parsed


def _extractors_for_dim(cnn_output_dim):
    batch, _ = _parse_cms_batch(2)
    observation_space, _ = _make_spaces(batch)
    extractor = VectorLayerNormExtractor(observation_space, cnn_output_dim=cnn_output_dim)
    return extractor


@pytest.mark.parametrize("cnn_output_dim", [8, 17, 256])
def test_spatial_projection_is_layernormed_after_relu(cnn_output_dim):
    extractor = _extractors_for_dim(cnn_output_dim)
    for fc in (extractor._screen_fc, extractor._minimap_fc):
        assert isinstance(fc[0], nn.Flatten)
        assert isinstance(fc[1], nn.Linear)
        assert fc[1].out_features == cnn_output_dim
        assert isinstance(fc[2], nn.ReLU)
        assert isinstance(fc[3], nn.LayerNorm)
        # ReLU must run before normalization so LayerNorm sees non-negative
        # activations and its output scale does not depend on input scale.
        assert tuple(fc[3].normalized_shape) == (cnn_output_dim,)
        assert len(fc) == 4


def _policy_with_large_projections(cnn_output_dim=256):
    torch.manual_seed(0)
    batch, _ = _parse_cms_batch(8)
    observation_space, action_space = _make_spaces(batch)
    policy = _make_policy(
        observation_space,
        action_space,
        cnn_output_dim=cnn_output_dim,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    extractor = policy.features_extractor
    # Reproduce the observed checkpoint pathology: the learned visual
    # projections emitted values up to ~600. Scaling the projection weights
    # emulates that scale without loading any checkpoint.
    with torch.no_grad():
        extractor._screen_fc[1].weight.mul_(500.0)
        extractor._minimap_fc[1].weight.mul_(500.0)
    return policy, batch


def test_large_visual_projection_stays_bounded_and_critic_gradients_flow():
    policy, batch = _policy_with_large_projections(cnn_output_dim=256)
    extractor = policy.features_extractor
    obs = {key: torch.as_tensor(value) for key, value in batch.items()}

    screen_proj = extractor._screen_fc(extractor.encode_spatial(obs, "screen"))
    minimap_proj = extractor._minimap_fc(extractor.encode_spatial(obs, "minimap"))
    for name, projection in (("screen", screen_proj), ("minimap", minimap_proj)):
        assert torch.isfinite(projection).all(), f"{name} projection is not finite"
        # LayerNorm output scale is independent of the pre-normalization scale.
        assert projection.abs().max() < 20.0, f"{name} projection is not scale-controlled"
        assert torch.allclose(
            projection.var(dim=1, unbiased=False),
            torch.ones(projection.shape[0]),
            atol=1e-3,
        ), f"{name} projection is not unit-variance per sample"

    features = policy.extract_features(obs)
    tanh = next(
        module for module in policy.mlp_extractor.value_net.modules()
        if isinstance(module, nn.Tanh)
    )
    captured = {}

    def _capture(_module, _inputs, output):
        captured["out"] = output.detach()

    handle = tanh.register_forward_hook(_capture)
    try:
        _, latent_vf = policy.mlp_extractor(features)
    finally:
        handle.remove()
    assert "out" in captured, "value MLP Tanh was not executed"
    saturation = (captured["out"].abs() >= 0.999).float().mean().item()
    assert saturation < 0.5, f"critic Tanh is saturated ({saturation:.3f})"

    values = policy.value_net(latent_vf).flatten()
    # Observable synthetic target: number of neutral shard cells.
    parsed = _parse_cms_batch(8)[1]
    layer_names = [name for _, name in policy.features_extractor._screen_layers]
    player_relative_idx = layer_names.index("player_relative")
    target = torch.tensor(
        [(p["screen"][..., player_relative_idx] == 3).sum() for p in parsed],
        dtype=torch.float32,
    ) / 20.0
    loss = torch.nn.functional.mse_loss(values, target)
    assert torch.isfinite(loss)
    loss.backward()

    params = dict(policy.named_parameters())
    for name in (
        "mlp_extractor.value_net.0.weight",
        "value_net.weight",
        "features_extractor._screen_fc.1.weight",
        "features_extractor._minimap_fc.1.weight",
    ):
        grad = params[name].grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} has a non-finite gradient"
        assert grad.abs().sum() > 0, f"{name} received an all-zero gradient"


def test_short_supervised_value_fit_reduces_loss_and_tracks_state():
    torch.manual_seed(1)
    shard_counts = [4 + (i % 17) for i in range(48)]
    batch, parsed = _parse_cms_batch(48, seed=3, shard_counts=shard_counts)
    observation_space, action_space = _make_spaces(batch)
    policy = _make_policy(
        observation_space,
        action_space,
        cnn_output_dim=8,
        net_arch=dict(pi=[64], vf=[64, 64]),
    )
    obs = {key: torch.as_tensor(value) for key, value in batch.items()}

    # Freeze the visual trunk and fit only the critic on its features; this is
    # a supervised trainability check on synthetic observations, not an SC2
    # result.
    with torch.no_grad():
        features = policy.extract_features(obs)
        policy._clear_last_spatial_maps()

    # Observable state quantity: number of neutral shard cells (4..20).
    layer_names = [name for _, name in policy.features_extractor._screen_layers]
    player_relative_idx = layer_names.index("player_relative")
    target = torch.tensor(
        [(p["screen"][..., player_relative_idx] == 3).sum() for p in parsed],
        dtype=torch.float32,
    ) / 20.0

    critic_params = list(policy.mlp_extractor.value_net.parameters()) + list(
        policy.value_net.parameters()
    )
    optimizer = torch.optim.Adam(critic_params, lr=3e-4)

    def _predict():
        _, latent_vf = policy.mlp_extractor(features)
        return policy.value_net(latent_vf).flatten()

    with torch.no_grad():
        initial_loss = torch.nn.functional.mse_loss(_predict(), target).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(_predict(), target)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = _predict()
        final_loss = torch.nn.functional.mse_loss(predictions, target).item()
    correlation = float(np.corrcoef(predictions.numpy(), target.numpy())[0, 1])
    assert final_loss < 0.8 * initial_loss, (
        f"value fit did not improve: {initial_loss:.4f} -> {final_loss:.4f}"
    )
    assert correlation > 0.15, f"predictions do not track state (r={correlation:.3f})"
    assert torch.isfinite(predictions).all()
