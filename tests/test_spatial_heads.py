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
    obs["available_actions"].zero_()
    obs["available_actions"][:, pysc2_actions.FUNCTIONS.no_op.id] = 1.0
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


def test_spatial_entropy_is_expected_over_function_usage_and_differentiable():
    policy = _make_policy()
    logits = torch.randn(BATCH_SIZE, int(sum(policy.action_space.nvec)), requires_grad=True)
    sizes = [int(size) for size in policy.action_space.nvec]
    fn_logits, *slot_logits = torch.split(logits, sizes, dim=1)
    fn_probs = torch.softmax(fn_logits, dim=1)
    fn_entropy = -(fn_probs * torch.log_softmax(fn_logits, dim=1)).sum(dim=1)
    expected = fn_entropy
    usage = fn_probs @ policy._fn_slot_mask
    for i, arg_logits in enumerate(slot_logits):
        probs = torch.softmax(arg_logits, dim=1)
        arg_entropy = -(probs * torch.log_softmax(arg_logits, dim=1)).sum(dim=1)
        expected = expected + usage[:, i] * arg_entropy

    entropy = policy._joint_entropy_from_logits(logits)
    torch.testing.assert_close(entropy, expected)

    offsets = torch.cat([torch.full((size,), float(i + 1)) for i, size in enumerate(sizes)])
    shifted_entropy = policy._joint_entropy_from_logits(logits + offsets)
    torch.testing.assert_close(shifted_entropy, entropy)

    entropy.sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_unused_argument_slots_do_not_change_joint_log_prob():
    policy = _make_policy()
    logits = torch.randn(BATCH_SIZE, int(sum(policy.action_space.nvec)))
    actions = torch.zeros(BATCH_SIZE, 1 + len(_param_semantics()), dtype=torch.long)
    actions[:, 0] = pysc2_actions.FUNCTIONS.no_op.id
    changed = actions.clone()
    changed[:, 1:] = torch.tensor([min(1, int(size) - 1) for size in policy.action_space.nvec[1:]])

    log_prob, _ = policy._joint_logprob_and_entropy(logits, actions)
    changed_log_prob, _ = policy._joint_logprob_and_entropy(logits, changed)
    torch.testing.assert_close(log_prob, changed_log_prob)


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


def _latent_and_base_logits(policy, obs):
    latent, _ = policy.mlp_extractor(policy.extract_features(obs))
    return latent, policy._base_action_logits(latent, obs)


def test_argument_and_spatial_logits_depend_on_function_and_observation():
    policy = _make_policy()
    obs = _make_obs()
    latent, base_logits = _latent_and_base_logits(policy, obs)
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    attack_id = pysc2_actions.FUNCTIONS.Attack_screen.id
    queued_idx = policy._param_semantics.index("queued")
    screen_span = _slot_span(policy, policy._screen_slot_idx)
    with torch.no_grad():
        policy._argument_conditioner[-1].weight.normal_(std=0.01)
        policy._screen_conditional_net[-1].weight.normal_(std=0.01)
        policy._screen_conditional_net[-1].bias.normal_(std=0.01)

    move = policy._condition_argument_logits(
        base_logits, latent, torch.full((BATCH_SIZE,), move_id), obs
    )
    attack = policy._condition_argument_logits(
        base_logits, latent, torch.full((BATCH_SIZE,), attack_id), obs
    )
    assert not torch.allclose(move[:, screen_span], attack[:, screen_span])
    queued_span = _slot_span(policy, queued_idx)
    assert not torch.allclose(move[:, queued_span], attack[:, queued_span])

    changed_obs = dict(obs)
    changed_obs["vector"] = obs["vector"].clone()
    changed_obs["vector"][:, 0] = -2.0
    changed_latent, changed_base = _latent_and_base_logits(policy, changed_obs)
    changed = policy._condition_argument_logits(
        changed_base, changed_latent, torch.full((BATCH_SIZE,), move_id), changed_obs
    )
    assert not torch.allclose(move[:, queued_span], changed[:, queued_span])


def test_conditional_sampling_evaluation_distribution_and_entropy_match():
    policy = _make_policy()
    obs = _make_obs()
    point_id = pysc2_actions.FUNCTIONS.select_point.id
    obs["available_actions"].zero_()
    obs["available_actions"][:, [pysc2_actions.FUNCTIONS.no_op.id, point_id]] = 1.0

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
        active = set(policy._fn_param_map[fn_id])
        for slot_idx in range(len(policy._slot_sizes)):
            if slot_idx not in active:
                assert row[slot_idx + 1].item() == 0


def test_conditional_entropy_exactly_marginalizes_available_functions_and_backpropagates():
    policy = _make_policy()
    obs = _make_obs()
    no_op_id = pysc2_actions.FUNCTIONS.no_op.id
    point_id = pysc2_actions.FUNCTIONS.select_point.id
    obs["available_actions"].zero_()
    obs["available_actions"][:, [no_op_id, point_id]] = 1.0
    latent, logits = _latent_and_base_logits(policy, obs)
    fn_logits = torch.full_like(logits[:, :len(pysc2_actions.FUNCTIONS)], -20.0)
    fn_logits[:, no_op_id] = 0.0
    fn_logits[:, point_id] = 0.0
    logits = torch.cat([fn_logits, logits[:, len(pysc2_actions.FUNCTIONS):]], dim=1)

    probs = torch.softmax(fn_logits, dim=1)
    fn_entropy = -(probs * torch.log_softmax(fn_logits, dim=1)).sum(dim=1)
    point_ids = torch.full((BATCH_SIZE,), point_id, dtype=torch.long)
    point_logits = policy._condition_argument_logits(logits, latent, point_ids, obs)
    point_slots = torch.split(point_logits, [int(size) for size in policy.action_space.nvec], dim=1)[1:]
    point_entropy = torch.zeros(BATCH_SIZE)
    for slot_idx in policy._fn_param_map[point_id]:
        slot_logp = torch.log_softmax(point_slots[slot_idx], dim=1)
        point_entropy = point_entropy - (slot_logp.exp() * slot_logp).sum(dim=1)
    expected = fn_entropy + probs[:, point_id] * point_entropy

    entropy = policy._conditional_joint_entropy(logits, latent, obs)
    torch.testing.assert_close(entropy, expected)
    entropy.sum().backward()
    assert policy._argument_conditioner[-1].weight.grad is not None
    assert torch.isfinite(policy._argument_conditioner[-1].weight.grad).all()


def test_old_checkpoint_loads_with_zero_effect_conditioning_parameters():
    policy = _make_policy()
    old_keys = ("_function_embedding.", "_argument_conditioner.", "_screen_conditional_net.", "_minimap_conditional_net.")
    old_state = {key: value.clone() for key, value in policy.state_dict().items() if not key.startswith(old_keys)}
    loaded = _make_policy()

    result = loaded.load_state_dict(old_state, strict=True)

    assert not result.missing_keys and not result.unexpected_keys
    obs = _make_obs()
    latent, base = _latent_and_base_logits(loaded, obs)
    fn_ids = torch.full((BATCH_SIZE,), pysc2_actions.FUNCTIONS.Move_screen.id)
    conditioned = loaded._condition_argument_logits(base, latent, fn_ids, obs)
    legacy = loaded._inject_spatial_logits(base.clone(), obs)
    torch.testing.assert_close(conditioned, legacy)
