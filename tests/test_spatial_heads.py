"""Regression tests for eager registration of spatial action heads.

These tests do not launch SC2; they build the policy directly from dummy
observation/action spaces matching the 32x32 PySC2 mini-game configuration.
"""
import os
import sys

# Make `src` importable when pytest is invoked as a console script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pytest
import torch

from pysc2.lib import actions as pysc2_actions

from src.policies.masked_flatten_policy import MaskedFlattenPolicy, VectorLayerNormExtractor

SPATIAL_SIZE = 32
SCREEN_LAYERS = 7
MINIMAP_LAYERS = 6
VECTOR_DIM = 4
BATCH_SIZE = 2


def _param_semantics():
    semantics = []
    seen = set()
    for fn in pysc2_actions.FUNCTIONS:
        for spec in fn.args:
            name = getattr(spec, "name")
            if name not in seen:
                semantics.append(name)
                seen.add(name)
    return semantics


def _make_spaces():
    semantics = _param_semantics()
    n_funcs = len(pysc2_actions.FUNCTIONS)
    # Spatial slots match the 32x32 flattened maps; other slots are irrelevant here.
    slot_sizes = [SPATIAL_SIZE * SPATIAL_SIZE if name in ("screen", "minimap") else 2 for name in semantics]
    action_space = gym.spaces.MultiDiscrete([n_funcs] + slot_sizes)
    observation_space = gym.spaces.Dict(
        {
            "vector": gym.spaces.Box(-np.inf, np.inf, (VECTOR_DIM,), np.float32),
            "vector_mask": gym.spaces.Box(0.0, 1.0, (VECTOR_DIM,), np.float32),
            "screen": gym.spaces.Box(-np.inf, np.inf, (SPATIAL_SIZE, SPATIAL_SIZE, SCREEN_LAYERS), np.float32),
            "minimap": gym.spaces.Box(-np.inf, np.inf, (SPATIAL_SIZE, SPATIAL_SIZE, MINIMAP_LAYERS), np.float32),
            "available_actions": gym.spaces.Box(0.0, 1.0, (n_funcs,), np.float32),
            "screen_layer_flags": gym.spaces.Box(0.0, 1.0, (SCREEN_LAYERS,), np.float32),
            "minimap_layer_flags": gym.spaces.Box(0.0, 1.0, (MINIMAP_LAYERS,), np.float32),
        }
    )
    return observation_space, action_space


def _make_policy():
    observation_space, action_space = _make_spaces()
    return MaskedFlattenPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=VectorLayerNormExtractor,
        features_extractor_kwargs={"cnn_output_dim": 8},
        net_arch=dict(pi=[16], vf=[16]),
    )


def _make_obs():
    observation_space, _ = _make_spaces()
    obs = {}
    for key, space in observation_space.spaces.items():
        if key in ("screen", "minimap"):
            # random input so conv weights receive non-zero gradients
            obs[key] = torch.randn(BATCH_SIZE, *space.shape)
        else:
            obs[key] = torch.ones(BATCH_SIZE, *space.shape)
    return obs


def _spatial_named_params(policy):
    for name, param in policy.named_parameters():
        if name.startswith("_screen_spatial_net.") or name.startswith("_minimap_spatial_net."):
            yield name, param
        elif name in ("_screen_spatial_log_scale", "_minimap_spatial_log_scale"):
            yield name, param


def test_spatial_heads_registered_before_optimizer_and_receive_gradients():
    policy = _make_policy()
    spatial_params = dict(_spatial_named_params(policy))
    assert spatial_params, "spatial heads were not created during policy construction"

    optimizer_ids = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    for name, param in spatial_params.items():
        assert id(param) in optimizer_ids, f"{name} is missing from the optimizer parameter groups"

    obs = _make_obs()
    total_logits = int(sum(policy.action_space.nvec))
    base_logits = torch.zeros(BATCH_SIZE, total_logits)
    logits = policy._inject_spatial_logits(base_logits, obs)
    logits.sum().backward()

    for name, param in spatial_params.items():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"
        assert param.grad.abs().sum() > 0, f"{name} received an all-zero gradient"


def test_cpu_forward_and_evaluate_actions_backward():
    policy = _make_policy()
    obs = _make_obs()

    actions, values, log_prob = policy(obs)
    assert actions.shape[0] == BATCH_SIZE
    assert values.shape == (BATCH_SIZE, 1)
    assert torch.isfinite(log_prob).all()

    eval_values, eval_log_prob, entropy = policy.evaluate_actions(obs, actions.detach())
    assert torch.isfinite(eval_log_prob).all()
    assert torch.isfinite(entropy).all()

    loss = log_prob.sum() + values.sum() + eval_values.sum() + eval_log_prob.sum() + entropy.sum()
    loss.backward()
    assert torch.isfinite(loss)


def test_save_load_then_forward_without_prior_forward(tmp_path):
    policy = _make_policy()
    obs = _make_obs()
    # Exercise the policy before saving: spatial parameters must already exist
    # and must not be added only during the first forward.
    keys_before_forward = set(policy.state_dict())
    policy(obs)
    assert set(policy.state_dict()) == keys_before_forward

    path = tmp_path / "policy.pt"
    policy.save(str(path))

    # BaseModel.load uses strict load_state_dict, so missing/unexpected spatial
    # head keys would fail here.
    loaded = MaskedFlattenPolicy.load(str(path), device="cpu")
    assert set(dict(_spatial_named_params(loaded))) == set(dict(_spatial_named_params(policy)))

    actions, values, log_prob = loaded(obs)
    assert actions.shape[0] == BATCH_SIZE
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(values).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_spatial_heads_follow_policy_device():
    policy = _make_policy().to("cuda")
    for name, param in _spatial_named_params(policy):
        assert param.device.type == "cuda", f"{name} did not follow policy.to('cuda')"

    obs = {key: value.to("cuda") for key, value in _make_obs().items()}
    actions, values, log_prob = policy(obs)
    (log_prob.sum() + values.sum()).backward()

    for name, param in _spatial_named_params(policy):
        assert param.grad is not None, f"{name} received no gradient on CUDA"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient on CUDA"


# ---------------------------------------------------------------------------
# Layer-flag semantics
# ---------------------------------------------------------------------------


def _make_extractor():
    observation_space, _ = _make_spaces()
    return VectorLayerNormExtractor(observation_space, cnn_output_dim=8)


def _cat_output_span(extractor, cat_channel):
    """Return the output-channel slice occupied by one categorical input channel."""
    offset = 0
    for i in range(cat_channel):
        ci = next((c for c in extractor._screen_cat_info if c[0] == i), None)
        offset += ci[2] if ci is not None else 1
    ci = next(c for c in extractor._screen_cat_info if c[0] == cat_channel)
    return offset, offset + ci[2]


def test_inactive_categorical_layers_embed_to_exact_zero():
    extractor = _make_extractor()
    assert extractor._screen_cat_info, "categorical screen layers were not detected"
    screen = torch.randint(0, 4, (BATCH_SIZE, SPATIAL_SIZE, SPATIAL_SIZE, SCREEN_LAYERS)).float()
    flags = torch.zeros(BATCH_SIZE, SCREEN_LAYERS)
    feat = extractor._embed_spatial(
        screen,
        flags,
        extractor._screen_cat_info,
        extractor._screen_cont_idx,
        extractor._screen_embeddings,
        "screen_layer_flags",
    )
    # embedding(0) is a learned non-zero vector: the output must be explicitly
    # zeroed by the flag, so every channel (cat and scalar) is exactly zero.
    assert feat.shape[0] == BATCH_SIZE
    assert torch.count_nonzero(feat).item() == 0


def test_active_categorical_layer_matches_embedding_lookup():
    extractor = _make_extractor()
    assert extractor._screen_cat_info, "categorical screen layers were not detected"
    screen = torch.randint(0, 4, (BATCH_SIZE, SPATIAL_SIZE, SPATIAL_SIZE, SCREEN_LAYERS)).float()
    flags = torch.ones(BATCH_SIZE, SCREEN_LAYERS)
    feat = extractor._embed_spatial(
        screen,
        flags,
        extractor._screen_cat_info,
        extractor._screen_cont_idx,
        extractor._screen_embeddings,
        "screen_layer_flags",
    )
    ch, num, _ = extractor._screen_cat_info[0]
    emb_idx = [ci[0] for ci in extractor._screen_cat_info].index(ch)
    vals = screen[..., ch].round().clamp(0, num - 1).long()
    expected = extractor._screen_embeddings[emb_idx](vals).permute(0, 3, 1, 2)
    start, end = _cat_output_span(extractor, ch)
    torch.testing.assert_close(feat[:, start:end], expected)


def test_inactive_embeddings_do_not_affect_extracted_features():
    observation_space, _ = _make_spaces()
    extractor = VectorLayerNormExtractor(observation_space, cnn_output_dim=8)
    obs = _make_obs()
    obs["screen_layer_flags"] = torch.zeros(BATCH_SIZE, SCREEN_LAYERS)
    obs["minimap_layer_flags"] = torch.zeros(BATCH_SIZE, MINIMAP_LAYERS)

    with torch.no_grad():
        before = extractor(obs)
        for emb in list(extractor._screen_embeddings) + list(extractor._minimap_embeddings):
            emb.weight.add_(1.0)
        after = extractor(obs)

    # Inactive layers must not leak a constant embedding into the features.
    torch.testing.assert_close(before, after)


def _slot_span(policy, slot_idx):
    sizes = [int(x) for x in policy.action_space.nvec]
    start = sizes[0] + sum(sizes[1:1 + slot_idx])
    return slice(start, start + sizes[1 + slot_idx])


def test_spatial_layer_flags_change_spatial_logits():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs()
    total = int(sum(policy.action_space.nvec))

    on = dict(obs)
    on["screen_layer_flags"] = torch.ones(BATCH_SIZE, SCREEN_LAYERS)
    on["minimap_layer_flags"] = torch.ones(BATCH_SIZE, MINIMAP_LAYERS)
    logits_on = policy._inject_spatial_logits(torch.zeros(BATCH_SIZE, total), on)

    off = dict(obs)
    off["screen_layer_flags"] = torch.zeros(BATCH_SIZE, SCREEN_LAYERS)
    off["minimap_layer_flags"] = torch.zeros(BATCH_SIZE, MINIMAP_LAYERS)
    logits_off = policy._inject_spatial_logits(torch.zeros(BATCH_SIZE, total), off)

    for slot_idx in (policy._screen_slot_idx, policy._minimap_slot_idx):
        span = _slot_span(policy, slot_idx)
        assert not torch.allclose(logits_on[:, span], logits_off[:, span])

    # Flags are per-channel and applied before the conv: all-zero flags must
    # feed an all-zero tensor into the screen conv.
    screen_span = _slot_span(policy, policy._screen_slot_idx)
    expected = policy._screen_spatial_net(
        torch.zeros(BATCH_SIZE, policy._screen_in_ch, SPATIAL_SIZE, SPATIAL_SIZE)
    ).flatten(1)
    expected = expected * torch.exp(policy._screen_spatial_log_scale.clamp(-6.0, 6.0))
    torch.testing.assert_close(logits_off[:, screen_span], expected)


def test_spatial_layer_flag_shape_mismatch_raises():
    policy = _make_policy()
    obs = _make_obs()
    total = int(sum(policy.action_space.nvec))

    bad_channels = dict(obs)
    bad_channels["screen_layer_flags"] = torch.ones(BATCH_SIZE, SCREEN_LAYERS - 1)
    with pytest.raises(ValueError, match="screen_layer_flags"):
        policy._inject_spatial_logits(torch.zeros(BATCH_SIZE, total), bad_channels)

    spatial_mask = dict(obs)
    spatial_mask["minimap_layer_flags"] = torch.ones(BATCH_SIZE, SPATIAL_SIZE, SPATIAL_SIZE)
    with pytest.raises(ValueError, match="minimap_layer_flags"):
        policy._inject_spatial_logits(torch.zeros(BATCH_SIZE, total), spatial_mask)


def test_movetobeacon_shapes_forward_with_flags():
    policy = _make_policy()
    actions, values, log_prob = policy(_make_obs())
    assert actions.shape == (BATCH_SIZE, 1 + len(_param_semantics()))
    assert values.shape == (BATCH_SIZE, 1)
    assert log_prob.shape == (BATCH_SIZE,)
    assert torch.isfinite(log_prob).all()
