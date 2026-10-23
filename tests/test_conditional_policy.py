"""Focused tests for function-conditioned spatial policy behavior.

These tests build the policy directly from dummy spaces and never launch SC2.
They complement ``test_spatial_heads.py`` with mixed-function batches,
conditioning-gradient checks, and save/load round-trips.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pysc2.lib import actions as pysc2_actions

from src.policies.masked_flatten_policy import MaskedFlattenPolicy
from test_spatial_heads import BATCH_SIZE, _latent_and_base_logits, _make_obs, _make_policy


def _tile_obs(obs, n):
    return {key: value.repeat(n, *([1] * (value.dim() - 1))) for key, value in obs.items()}


def _spatial_action_values(policy, fn_ids, n):
    """Random values for the sampled spatial slots (other slots stay zero)."""
    sizes = [int(x) for x in policy.action_space.nvec]
    actions = torch.zeros(n, len(sizes), dtype=torch.long)
    actions[:, 0] = fn_ids
    used = policy._fn_slot_mask[fn_ids]
    for slot_idx in policy._sampled_slot_indices:
        for row in range(n):
            if used[row, slot_idx] > 0:
                actions[row, slot_idx + 1] = torch.randint(0, sizes[slot_idx + 1], (1,)).item()
    return actions


def _expected_joint_log_prob(policy, conditioned, actions):
    sizes = [int(x) for x in policy.action_space.nvec]
    splits = torch.split(conditioned, sizes, dim=1)
    fn_ids = actions[:, 0].long()
    expected = torch.log_softmax(splits[0], dim=1).gather(1, fn_ids[:, None]).squeeze(1)
    used = policy._fn_slot_mask[fn_ids]
    for slot_idx in policy._sampled_slot_indices:
        slot_logp = torch.log_softmax(splits[slot_idx + 1], dim=1)
        chosen = actions[:, slot_idx + 1]
        expected = expected + used[:, slot_idx] * slot_logp.gather(1, chosen[:, None]).squeeze(1)
    return expected


def _expected_entropy(policy, base, obs, available_ids):
    sizes = [int(x) for x in policy.action_space.nvec]
    fn_logits = torch.split(base, sizes, dim=1)[0]
    fn_probs = torch.softmax(fn_logits, dim=1)
    expected = -(fn_probs * torch.log_softmax(fn_logits, dim=1)).sum(dim=1)
    n = base.shape[0]
    for fn_id in available_ids:
        fn_slots = policy._fn_slot_mask[fn_id]
        used_slots = [slot for slot in policy._sampled_slot_indices if fn_slots[slot] > 0]
        if not used_slots:
            continue
        branch_ids = torch.full((n,), fn_id, dtype=torch.long)
        branch = policy._condition_argument_logits(base, branch_ids, obs)
        branch_splits = torch.split(branch, sizes, dim=1)[1:]
        branch_entropy = torch.zeros(n)
        for slot_idx in used_slots:
            slot_logp = torch.log_softmax(branch_splits[slot_idx], dim=1)
            branch_entropy = branch_entropy - (slot_logp.exp() * slot_logp).sum(dim=1)
        expected = expected + fn_probs[:, fn_id] * branch_entropy
    return expected


def test_mixed_batch_conditions_on_per_sample_function_and_counts():
    torch.manual_seed(0)
    policy = _make_policy()
    n = 2 * BATCH_SIZE
    no_op_id = pysc2_actions.FUNCTIONS.no_op.id
    point_id = pysc2_actions.FUNCTIONS.select_point.id
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    attack_minimap_id = pysc2_actions.FUNCTIONS.Attack_minimap.id
    available_ids = [no_op_id, point_id, move_id, attack_minimap_id]
    obs = _make_obs(available_ids)
    obs = _tile_obs(obs, 2)

    latent, base = _latent_and_base_logits(policy, obs)
    fn_ids = torch.tensor([no_op_id, point_id, move_id, attack_minimap_id])
    conditioned = policy._condition_argument_logits(base, fn_ids, obs)
    actions = _spatial_action_values(policy, fn_ids, n)

    log_prob, _ = policy._joint_logprob_and_entropy(conditioned, actions)
    torch.testing.assert_close(log_prob, _expected_joint_log_prob(policy, conditioned, actions))

    # Changing a non-sampled slot (including screen2) must not matter.
    changed = actions.clone()
    for slot_idx in range(len(policy._slot_sizes)):
        if slot_idx not in policy._sampled_slot_indices:
            changed[:, slot_idx + 1] = 1
    changed_log_prob, _ = policy._joint_logprob_and_entropy(conditioned, changed)
    torch.testing.assert_close(changed_log_prob, log_prob)

    entropy = policy._conditional_joint_entropy(base, obs)
    torch.testing.assert_close(entropy, _expected_entropy(policy, base, obs, available_ids))


def test_conditioning_path_receives_nonzero_gradients():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs([pysc2_actions.FUNCTIONS.Move_screen.id])
    actions, values, log_prob = policy(obs, deterministic=True)
    (log_prob.sum() + values.sum()).backward()

    params = dict(policy.named_parameters())
    for name in (
        "_function_embedding.weight",
        "_screen_spatial_conv.weight",
        "_screen_fn_conditioner.weight",
        "_screen_spatial_out.weight",
        "features_extractor._screen_encoder.0.weight",
    ):
        grad = params[name].grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} has a non-finite gradient"
        assert grad.abs().sum() > 0, f"{name} received an all-zero gradient"


def test_minimap_conditional_head_receives_nonzero_gradients():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs([pysc2_actions.FUNCTIONS.Attack_minimap.id])
    actions, values, log_prob = policy(obs, deterministic=True)
    (log_prob.sum() + values.sum()).backward()

    params = dict(policy.named_parameters())
    for name in (
        "_minimap_spatial_conv.weight",
        "_minimap_spatial_conv.bias",
        "_minimap_fn_conditioner.weight",
        "_minimap_spatial_out.weight",
        "_function_embedding.weight",
        "features_extractor._minimap_encoder.0.weight",
    ):
        grad = params[name].grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} has a non-finite gradient"
        assert grad.abs().sum() > 0, f"{name} received an all-zero gradient"


def test_exact_entropy_backpropagates_to_conditioning_parameters():
    torch.manual_seed(0)
    policy = _make_policy()
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    attack_id = pysc2_actions.FUNCTIONS.Attack_screen.id
    obs = _make_obs([move_id, attack_id])

    latent, base = _latent_and_base_logits(policy, obs)
    entropy = policy._conditional_joint_entropy(base.detach(), obs)
    entropy.sum().backward()

    params = dict(policy.named_parameters())
    for name in (
        "_function_embedding.weight",
        "_screen_spatial_conv.weight",
        "_screen_spatial_out.weight",
    ):
        grad = params[name].grad
        assert grad is not None, f"{name} received no entropy gradient"
        assert torch.isfinite(grad).all(), f"{name} has a non-finite entropy gradient"
        assert grad.abs().sum() > 0, f"{name} received an all-zero entropy gradient"


def test_save_load_preserves_conditioned_distribution_and_state_keys(tmp_path):
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs([pysc2_actions.FUNCTIONS.Move_screen.id, pysc2_actions.FUNCTIONS.Attack_screen.id])

    keys_before_forward = set(policy.state_dict())
    latent, base = _latent_and_base_logits(policy, obs)
    fn_ids = torch.tensor(
        [
            pysc2_actions.FUNCTIONS.Move_screen.id,
            pysc2_actions.FUNCTIONS.Attack_screen.id,
        ]
    )
    conditioned = policy._condition_argument_logits(base, fn_ids, obs)
    entropy = policy._conditional_joint_entropy(base, obs)
    assert set(policy.state_dict()) == keys_before_forward

    path = tmp_path / "policy.pt"
    policy.save(str(path))
    loaded = MaskedFlattenPolicy.load(str(path), device="cpu")
    assert set(loaded.state_dict()) == keys_before_forward

    loaded_latent, loaded_base = _latent_and_base_logits(loaded, obs)
    loaded_conditioned = loaded._condition_argument_logits(loaded_base, fn_ids, obs)
    torch.testing.assert_close(loaded_conditioned, conditioned)
    torch.testing.assert_close(loaded._conditional_joint_entropy(loaded_base, obs), entropy)


def test_batch_varying_masks_entropy_is_exact_and_deterministic():
    torch.manual_seed(0)
    policy = _make_policy()
    no_op_id = pysc2_actions.FUNCTIONS.no_op.id
    point_id = pysc2_actions.FUNCTIONS.select_point.id
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    attack_id = pysc2_actions.FUNCTIONS.Attack_screen.id
    obs = _make_obs([no_op_id])

    # Different rows see different available sets, so the exact entropy must
    # marginalize over the per-batch union without assuming a shared mask.
    obs["available_actions"].zero_()
    obs["available_actions"][0, [no_op_id, point_id, move_id]] = 1.0
    obs["available_actions"][1, [no_op_id, attack_id]] = 1.0

    latent, base = _latent_and_base_logits(policy, obs)
    sizes = [int(x) for x in policy.action_space.nvec]
    fn_logits = torch.split(base, sizes, dim=1)[0]
    fn_logp = torch.log_softmax(fn_logits, dim=1)
    fn_probs = fn_logp.exp()
    assert fn_probs[0, attack_id] < 1e-6
    assert fn_probs[1, move_id] < 1e-6

    expected = _expected_entropy(policy, base, obs, [point_id, move_id, attack_id])
    entropy = policy._conditional_joint_entropy(base, obs)
    torch.testing.assert_close(entropy, expected)

    # Repeated evaluation must be bitwise identical (no sampling in the path).
    assert torch.equal(entropy, policy._conditional_joint_entropy(base, obs))


def test_mixed_batch_empty_available_row_falls_back_to_no_op_consistently():
    torch.manual_seed(0)
    policy = _make_policy()
    no_op_id = pysc2_actions.FUNCTIONS.no_op.id
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    sizes = [int(x) for x in policy.action_space.nvec]
    obs = _make_obs([no_op_id])

    # Row 0 has no legal action at all; row 1 can move. Sampling, evaluation
    # and entropy must all agree that row 0's support is no_op-only.
    obs["available_actions"].zero_()
    obs["available_actions"][1, move_id] = 1.0

    latent, base = _latent_and_base_logits(policy, obs)
    fn_probs = torch.softmax(torch.split(base, sizes, dim=1)[0], dim=1)
    assert int((fn_probs[0] > 0).sum()) == 1
    assert fn_probs[0, no_op_id].item() == pytest.approx(1.0)
    assert int((fn_probs[1] > 0).sum()) == 1
    assert fn_probs[1, move_id].item() == pytest.approx(1.0)

    actions, values, log_prob = policy(obs, deterministic=True)
    assert actions[0, 0].item() == no_op_id
    assert actions[1, 0].item() == move_id
    _, eval_log_prob, entropy = policy.evaluate_actions(obs, actions)
    torch.testing.assert_close(eval_log_prob, log_prob)

    # Row 0 uses no_op so its argument entropy is zero; row 1 contributes
    # Move_screen's screen entropy.
    expected = _expected_entropy(policy, base, obs, [move_id])
    assert entropy[0].item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(entropy, expected)


def test_all_parameters_are_registered_with_the_optimizer():
    policy = _make_policy()
    optimizer_ids = {
        id(param) for group in policy.optimizer.param_groups for param in group["params"]
    }
    assert optimizer_ids == {id(param) for param in policy.parameters()}
