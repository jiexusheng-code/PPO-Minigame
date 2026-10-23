"""Regression tests for the shared full-resolution spatial policy.

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

from src.policies.masked_flatten_policy import (
    FN_EMBED_DIM,
    SPATIAL_FEATURE_CHANNELS,
    MaskedFlattenPolicy,
    VectorLayerNormExtractor,
)

SPATIAL_SIZE = 32
# Canonical ObsParser layer order: screen has 8 layers, minimap has 5.
SCREEN_LAYERS = 8
MINIMAP_LAYERS = 5
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


def _default_available():
    return [
        pysc2_actions.FUNCTIONS.no_op.id,
        pysc2_actions.FUNCTIONS.Move_screen.id,
        pysc2_actions.FUNCTIONS.select_point.id,
    ]


def _make_obs(available=None):
    observation_space, _ = _make_spaces()
    obs = {}
    for key, space in observation_space.spaces.items():
        if key in ("screen", "minimap"):
            # random input so conv weights receive non-zero gradients
            obs[key] = torch.randn(BATCH_SIZE, *space.shape)
        else:
            obs[key] = torch.ones(BATCH_SIZE, *space.shape)
    obs["available_actions"].zero_()
    for fn_id in _default_available() if available is None else available:
        obs["available_actions"][:, fn_id] = 1.0
    return obs


def _spatial_named_params(policy):
    prefixes = (
        "_screen_spatial_conv.",
        "_screen_fn_conditioner.",
        "_screen_spatial_out.",
        "_minimap_spatial_conv.",
        "_minimap_fn_conditioner.",
        "_minimap_spatial_out.",
        "features_extractor._screen_encoder.",
        "features_extractor._minimap_encoder.",
        "features_extractor._screen_fc.",
        "features_extractor._minimap_fc.",
    )
    for name, param in policy.named_parameters():
        if name.startswith(prefixes):
            yield name, param


def _assert_nonzero_finite_grads(policy, prefixes):
    params = dict(policy.named_parameters())
    for name, param in params.items():
        if not name.startswith(prefixes):
            continue
        grad = param.grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} has a non-finite gradient"
        assert grad.abs().sum() > 0, f"{name} received an all-zero gradient"


def test_spatial_heads_registered_before_optimizer_and_receive_gradients():
    policy = _make_policy()
    spatial_params = dict(_spatial_named_params(policy))
    assert spatial_params, "spatial heads were not created during policy construction"

    optimizer_ids = {id(p) for group in policy.optimizer.param_groups for p in group["params"]}
    for name, param in spatial_params.items():
        assert id(param) in optimizer_ids, f"{name} is missing from the optimizer parameter groups"

    # A screen-using function forces the shared trunk, projection, and screen head.
    obs = _make_obs([pysc2_actions.FUNCTIONS.Move_screen.id])
    actions, values, log_prob = policy(obs, deterministic=True)
    (log_prob.sum() + values.sum()).backward()

    _assert_nonzero_finite_grads(
        policy,
        (
            "features_extractor._screen_encoder.",
            "features_extractor._screen_fc.",
            "_screen_spatial_conv.",
            "_screen_fn_conditioner.",
            "_screen_spatial_out.",
            "_function_embedding.",
        ),
    )
    assert policy._screen_spatial_conv is not None
    assert policy._minimap_spatial_conv is not None


def test_minimap_head_receives_gradients_from_minimap_function():
    policy = _make_policy()
    obs = _make_obs([pysc2_actions.FUNCTIONS.Attack_minimap.id])
    actions, values, log_prob = policy(obs, deterministic=True)
    (log_prob.sum() + values.sum()).backward()
    _assert_nonzero_finite_grads(
        policy,
        (
            "features_extractor._minimap_encoder.",
            "features_extractor._minimap_fc.",
            "_minimap_spatial_conv.",
            "_minimap_fn_conditioner.",
            "_minimap_spatial_out.",
            "_function_embedding.",
        ),
    )


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
    torch.testing.assert_close(eval_log_prob, log_prob)

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

    expected_actions, _, expected_log_prob = policy(obs, deterministic=True)
    path = tmp_path / "policy.pt"
    policy.save(str(path))

    # BaseModel.load uses strict load_state_dict, so missing/unexpected spatial
    # head keys would fail here.
    loaded = MaskedFlattenPolicy.load(str(path), device="cpu")
    assert set(dict(_spatial_named_params(loaded))) == set(dict(_spatial_named_params(policy)))

    actions, values, log_prob = loaded(obs, deterministic=True)
    torch.testing.assert_close(actions, expected_actions)
    torch.testing.assert_close(log_prob, expected_log_prob)


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


def test_entropy_is_exact_over_sampled_spatial_slots():
    torch.manual_seed(0)
    policy = _make_policy()
    available = [
        pysc2_actions.FUNCTIONS.no_op.id,
        pysc2_actions.FUNCTIONS.Move_screen.id,
        pysc2_actions.FUNCTIONS.Attack_minimap.id,
    ]
    obs = _make_obs(available)
    latent, base = _latent_and_base_logits(policy, obs)
    sizes = [int(size) for size in policy.action_space.nvec]
    fn_logits = torch.split(base, sizes, dim=1)[0]
    fn_probs = torch.softmax(fn_logits, dim=1)
    fn_logp = torch.log_softmax(fn_logits, dim=1)
    expected = -(fn_probs * fn_logp).sum(dim=1)
    for fn_id in available:
        fn_slots = policy._fn_slot_mask[fn_id]
        used_slots = [slot for slot in policy._sampled_slot_indices if fn_slots[slot] > 0]
        if not used_slots:
            continue
        branch_ids = torch.full((BATCH_SIZE,), fn_id, dtype=torch.long)
        branch = policy._condition_argument_logits(base, branch_ids, obs)
        branch_splits = torch.split(branch, sizes, dim=1)[1:]
        branch_entropy = torch.zeros(BATCH_SIZE)
        for slot_idx in used_slots:
            slot_logp = torch.log_softmax(branch_splits[slot_idx], dim=1)
            branch_entropy = branch_entropy - (slot_logp.exp() * slot_logp).sum(dim=1)
        expected = expected + fn_probs[:, fn_id] * branch_entropy

    entropy = policy._conditional_joint_entropy(base, obs)
    torch.testing.assert_close(entropy, expected)

    entropy.sum().backward()
    for name in ("_function_embedding.weight", "_screen_spatial_out.weight"):
        grad = dict(policy.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all()


def test_non_sampled_argument_slots_do_not_change_joint_log_prob():
    policy = _make_policy()
    obs = _make_obs([pysc2_actions.FUNCTIONS.select_rect.id])
    latent, base = _latent_and_base_logits(policy, obs)
    actions, _, _ = policy(obs, deterministic=True)
    log_prob, _ = policy._joint_logprob_and_entropy(
        policy._condition_argument_logits(base, actions[:, 0].long(), obs), actions
    )

    sizes = [int(x) for x in policy.action_space.nvec]
    changed = actions.clone()
    for slot_idx in range(len(sizes) - 1):
        if slot_idx not in policy._sampled_slot_indices:
            changed[:, slot_idx + 1] = 1 % sizes[slot_idx + 1]
    changed_log_prob, _ = policy._joint_logprob_and_entropy(
        policy._condition_argument_logits(base, changed[:, 0].long(), obs), changed
    )
    torch.testing.assert_close(changed_log_prob, log_prob)

    # select_rect: screen2 is a deterministic placeholder, never a sampled corner.
    screen2_idx = policy._param_semantics.index("screen2")
    assert changed[:, screen2_idx + 1].abs().sum().item() > 0
    assert screen2_idx not in policy._sampled_slot_indices


def test_spatial_logits_reject_action_slot_dimension_mismatch():
    policy = _make_policy()
    obs = _make_obs()
    obs["screen"] = torch.randn(BATCH_SIZE, SPATIAL_SIZE - 1, SPATIAL_SIZE - 1, SCREEN_LAYERS)
    logits = torch.zeros(BATCH_SIZE, int(sum(policy.action_space.nvec)))

    with pytest.raises(ValueError, match="action slot has size"):
        policy._inject_spatial_logits(logits, obs)


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


def _latent_and_base_logits(policy, obs):
    latent, _, _ = policy._extract_latents(obs)
    return latent, policy._base_action_logits(latent, obs)


def test_spatial_logits_depend_on_function_and_observation():
    policy = _make_policy()
    obs = _make_obs([pysc2_actions.FUNCTIONS.Move_screen.id, pysc2_actions.FUNCTIONS.Attack_screen.id])
    latent, base = _latent_and_base_logits(policy, obs)
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    attack_id = pysc2_actions.FUNCTIONS.Attack_screen.id
    screen_span = _slot_span(policy, policy._screen_slot_idx)

    move = policy._condition_argument_logits(base, torch.full((BATCH_SIZE,), move_id), obs)
    attack = policy._condition_argument_logits(base, torch.full((BATCH_SIZE,), attack_id), obs)
    assert not torch.allclose(move[:, screen_span], attack[:, screen_span])

    # The spatial head sees only the spatial trunk + flags + function id, so a
    # changed screen raster must move its logits ...
    changed_obs = dict(obs)
    changed_obs["screen"] = obs["screen"].clone()
    changed_obs["screen"][:, 0, 0, 1] = 3.0
    changed = policy._condition_argument_logits(base, torch.full((BATCH_SIZE,), move_id), changed_obs)
    assert not torch.allclose(move[:, screen_span], changed[:, screen_span])

    # ... while the vector only affects the main network, not the spatial head.
    vector_obs = dict(obs)
    vector_obs["vector"] = obs["vector"].clone()
    vector_obs["vector"][:, 0] = -2.0
    vector_latent, vector_base = _latent_and_base_logits(policy, vector_obs)
    vector_conditioned = policy._condition_argument_logits(
        vector_base, torch.full((BATCH_SIZE,), move_id), vector_obs
    )
    assert not torch.allclose(base[:, :policy._n_funcs], vector_base[:, :policy._n_funcs])
    torch.testing.assert_close(move[:, screen_span], vector_conditioned[:, screen_span])


def test_conditional_sampling_evaluation_distribution_and_entropy_match():
    policy = _make_policy()
    point_id = pysc2_actions.FUNCTIONS.select_point.id
    obs = _make_obs([pysc2_actions.FUNCTIONS.no_op.id, point_id])

    actions, _, forward_logp = policy(obs)
    _, eval_logp, eval_entropy = policy.evaluate_actions(obs, actions)
    distribution = policy.get_distribution(obs)
    torch.testing.assert_close(eval_logp, forward_logp)
    torch.testing.assert_close(distribution.log_prob(actions), eval_logp)
    torch.testing.assert_close(distribution.entropy(), eval_entropy)

    deterministic_actions, _, deterministic_logp = policy(obs, deterministic=True)
    torch.testing.assert_close(distribution.get_actions(deterministic=True), deterministic_actions)
    torch.testing.assert_close(distribution.log_prob(deterministic_actions), deterministic_logp)
    assert actions.shape == (BATCH_SIZE, len(policy.action_space.nvec))
    for row, fn_id in zip(actions, actions[:, 0].tolist()):
        for slot_idx in range(len(policy._slot_sizes)):
            if slot_idx not in policy._sampled_slot_indices:
                assert row[slot_idx + 1].item() == 0


def test_select_rect_keeps_center_and_zero_screen2_placeholder():
    policy = _make_policy()
    rect_id = pysc2_actions.FUNCTIONS.select_rect.id
    obs = _make_obs([rect_id])

    actions, _, log_prob = policy(obs, deterministic=True)
    screen_idx = policy._param_semantics.index("screen")
    screen2_idx = policy._param_semantics.index("screen2")
    assert (actions[:, 0] == rect_id).all()
    assert ((actions[:, screen_idx + 1] >= 0) & (actions[:, screen_idx + 1] < SPATIAL_SIZE * SPATIAL_SIZE)).all()
    assert (actions[:, screen2_idx + 1] == 0).all()

    # Corners are deterministic, so changing the screen2 placeholder must not
    # change the log probability of the latent center.
    changed = actions.clone()
    changed[:, screen2_idx + 1] = 1
    _, eval_logp, _ = policy.evaluate_actions(obs, actions)
    _, changed_logp, _ = policy.evaluate_actions(obs, changed)
    torch.testing.assert_close(eval_logp, changed_logp)
    torch.testing.assert_close(eval_logp, log_prob)


def test_conditional_entropy_exactly_marginalizes_available_functions_and_backpropagates():
    policy = _make_policy()
    no_op_id = pysc2_actions.FUNCTIONS.no_op.id
    point_id = pysc2_actions.FUNCTIONS.select_point.id
    obs = _make_obs([no_op_id, point_id])
    latent, base = _latent_and_base_logits(policy, obs)

    probs = torch.softmax(torch.split(base, [int(x) for x in policy.action_space.nvec], dim=1)[0], dim=1)
    fn_logp = torch.log_softmax(torch.split(base, [int(x) for x in policy.action_space.nvec], dim=1)[0], dim=1)
    fn_entropy = -(probs * fn_logp).sum(dim=1)
    point_ids = torch.full((BATCH_SIZE,), point_id, dtype=torch.long)
    point_logits = policy._condition_argument_logits(base, point_ids, obs)
    point_slots = torch.split(point_logits, [int(size) for size in policy.action_space.nvec], dim=1)[1:]
    screen_idx = policy._param_semantics.index("screen")
    point_screen_logp = torch.log_softmax(point_slots[screen_idx], dim=1)
    point_entropy = -(point_screen_logp.exp() * point_screen_logp).sum(dim=1)
    expected = fn_entropy + probs[:, point_id] * point_entropy

    entropy = policy._conditional_joint_entropy(base, obs)
    torch.testing.assert_close(entropy, expected)
    entropy.sum().backward()
    assert policy._screen_spatial_out.weight.grad is not None
    assert torch.isfinite(policy._screen_spatial_out.weight.grad).all()


def test_strict_load_rejects_old_architecture_checkpoint():
    policy = _make_policy()
    state = {key: value.clone() for key, value in policy.state_dict().items()}
    # Simulate the previous architecture: no shared-trunk spatial head keys.
    state.pop("_screen_spatial_conv.weight")

    with pytest.raises(RuntimeError, match="Missing key"):
        policy.load_state_dict(state, strict=True)


def test_shared_trunk_dimensions():
    policy = _make_policy()
    extractor = policy.features_extractor
    assert extractor._screen_encoder is not None
    assert extractor._minimap_encoder is not None
    with torch.no_grad():
        screen_map = extractor.encode_spatial(_make_obs(), "screen")
        minimap_map = extractor.encode_spatial(_make_obs(), "minimap")
    assert screen_map.shape == (BATCH_SIZE, SPATIAL_FEATURE_CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE)
    assert minimap_map.shape == (BATCH_SIZE, SPATIAL_FEATURE_CHANNELS, SPATIAL_SIZE, SPATIAL_SIZE)
    # Base head input = trunk channels + broadcast layer flags; the function
    # embedding is added as a channel bias after this convolution.
    assert policy._screen_spatial_conv.in_channels == SPATIAL_FEATURE_CHANNELS + SCREEN_LAYERS
    assert policy._minimap_spatial_conv.in_channels == SPATIAL_FEATURE_CHANNELS + MINIMAP_LAYERS
    assert policy._screen_fn_conditioner.in_features == FN_EMBED_DIM
