"""Focused tests for the function-conditioned argument head.

These tests build the policy directly from dummy spaces and never launch SC2.
They complement ``test_spatial_heads.py`` with mixed-function batches,
conditioning-gradient checks, save/load round-trips, and loading the saved
CollectMineralShards archive without an environment.
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_ARCHIVE = os.path.join(
    REPO_ROOT, "models", "collect_mineral_shards", "20261022153422", "best_model.zip"
)


def _tile_obs(obs, n):
    return {key: value.repeat(n, *([1] * (value.dim() - 1))) for key, value in obs.items()}


def _randomize_conditioning_outputs(policy):
    """Activate the residual heads so gradients reach the conditioning path."""
    with torch.no_grad():
        policy._argument_conditioner[-1].weight.normal_(std=0.01)
        policy._screen_conditional_net[-1].weight.normal_(std=0.01)
        policy._screen_conditional_net[-1].bias.normal_(std=0.01)
        if getattr(policy, "_minimap_conditional_net", None) is not None:
            policy._minimap_conditional_net[-1].weight.normal_(std=0.01)
            policy._minimap_conditional_net[-1].bias.normal_(std=0.01)


def test_mixed_batch_conditions_on_per_sample_function_and_counts():
    torch.manual_seed(0)
    policy = _make_policy()
    n = 2 * BATCH_SIZE
    obs = _tile_obs(_make_obs(), 2)
    no_op_id = pysc2_actions.FUNCTIONS.no_op.id
    point_id = pysc2_actions.FUNCTIONS.select_point.id
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    attack_id = pysc2_actions.FUNCTIONS.Attack_screen.id
    available_ids = [no_op_id, point_id, move_id, attack_id]
    obs["available_actions"].zero_()
    obs["available_actions"][:, available_ids] = 1.0
    _randomize_conditioning_outputs(policy)

    latent, base = _latent_and_base_logits(policy, obs)
    fn_ids = torch.tensor([no_op_id, point_id, move_id, attack_id])
    conditioned = policy._condition_argument_logits(base, latent, fn_ids, obs)

    sizes = [int(x) for x in policy.action_space.nvec]
    splits = torch.split(conditioned, sizes, dim=1)
    actions = torch.zeros(n, len(sizes), dtype=torch.long)
    actions[:, 0] = fn_ids
    used = policy._fn_slot_mask[fn_ids]
    for slot_idx in range(len(sizes) - 1):
        slot_size = sizes[slot_idx + 1]
        actions[:, slot_idx + 1] = torch.randint(0, min(slot_size, 4), (n,))

    log_prob, _ = policy._joint_logprob_and_entropy(conditioned, actions)
    fn_logp = torch.log_softmax(splits[0], dim=1)
    expected = fn_logp.gather(1, fn_ids[:, None]).squeeze(1)
    for slot_idx in range(len(sizes) - 1):
        slot_logp = torch.log_softmax(splits[slot_idx + 1], dim=1)
        chosen = actions[:, slot_idx + 1]
        expected = expected + used[:, slot_idx] * slot_logp.gather(1, chosen[:, None]).squeeze(1)
    torch.testing.assert_close(log_prob, expected)

    # Changing a slot that the row's function does not use must not matter.
    changed = actions.clone()
    for row in range(n):
        unused = [i for i in range(len(sizes) - 1) if used[row, i] == 0]
        for slot_idx in unused:
            changed[row, slot_idx + 1] = (changed[row, slot_idx + 1] + 1) % sizes[slot_idx + 1]
    changed_log_prob, _ = policy._joint_logprob_and_entropy(conditioned, changed)
    torch.testing.assert_close(changed_log_prob, log_prob)

    # Exact entropy: marginalize the per-function argument entropy over the
    # available support, masking slots the function does not use.
    fn_probs = torch.softmax(splits[0], dim=1)
    expected_entropy = -(fn_probs * torch.log_softmax(splits[0], dim=1)).sum(dim=1)
    for fn_id in available_ids:
        branch_ids = torch.full((n,), fn_id, dtype=torch.long)
        branch = policy._condition_argument_logits(base, latent, branch_ids, obs)
        branch_splits = torch.split(branch, sizes, dim=1)
        branch_used = policy._fn_slot_mask[branch_ids]
        branch_entropy = torch.zeros(n)
        for slot_idx in range(len(sizes) - 1):
            slot_logp = torch.log_softmax(branch_splits[slot_idx + 1], dim=1)
            branch_entropy = branch_entropy + (
                -(slot_logp.exp() * slot_logp).sum(dim=1) * branch_used[:, slot_idx]
            )
        expected_entropy = expected_entropy + fn_probs[:, fn_id] * branch_entropy

    entropy = policy._conditional_joint_entropy(base, latent, obs)
    torch.testing.assert_close(entropy, expected_entropy)


def test_conditioning_path_receives_nonzero_gradients():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs()
    _randomize_conditioning_outputs(policy)
    latent, base = _latent_and_base_logits(policy, obs)
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    attack_id = pysc2_actions.FUNCTIONS.Attack_screen.id

    move = policy._condition_argument_logits(
        base, latent, torch.full((BATCH_SIZE,), move_id), obs
    )
    attack = policy._condition_argument_logits(
        base, latent, torch.full((BATCH_SIZE,), attack_id), obs
    )
    (move.sum() + attack.sum()).backward()

    params = dict(policy.named_parameters())
    for name in (
        "_function_embedding.weight",
        "_argument_conditioner.0.weight",
        "_argument_conditioner.2.weight",
        "_screen_conditional_net.0.weight",
        "_screen_conditional_net.2.weight",
    ):
        grad = params[name].grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} has a non-finite gradient"
        assert grad.abs().sum() > 0, f"{name} received an all-zero gradient"


def test_minimap_conditional_head_receives_nonzero_gradients():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs()
    _randomize_conditioning_outputs(policy)
    latent, base = _latent_and_base_logits(policy, obs)
    attack_minimap_id = pysc2_actions.FUNCTIONS.Attack_minimap.id

    logits = policy._condition_argument_logits(
        base, latent, torch.full((BATCH_SIZE,), attack_minimap_id), obs
    )
    logits.sum().backward()

    params = dict(policy.named_parameters())
    for name in (
        "_minimap_conditional_net.0.weight",
        "_minimap_conditional_net.0.bias",
        "_minimap_conditional_net.2.weight",
        "_function_embedding.weight",
    ):
        grad = params[name].grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} has a non-finite gradient"
        assert grad.abs().sum() > 0, f"{name} received an all-zero gradient"


def test_exact_entropy_backpropagates_to_conditioning_parameters():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs()
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    attack_id = pysc2_actions.FUNCTIONS.Attack_screen.id
    obs["available_actions"].zero_()
    obs["available_actions"][:, [move_id, attack_id]] = 1.0
    _randomize_conditioning_outputs(policy)

    latent, base = _latent_and_base_logits(policy, obs)
    # Detach the observation-only path so the gradient can only come from the
    # function-conditioned branch entropies.
    entropy = policy._conditional_joint_entropy(base.detach(), latent.detach(), obs)
    entropy.sum().backward()

    params = dict(policy.named_parameters())
    for name in (
        "_function_embedding.weight",
        "_argument_conditioner.0.weight",
        "_argument_conditioner.2.weight",
        "_screen_conditional_net.0.weight",
        "_screen_conditional_net.2.weight",
    ):
        grad = params[name].grad
        assert grad is not None, f"{name} received no entropy gradient"
        assert torch.isfinite(grad).all(), f"{name} has a non-finite entropy gradient"
        assert grad.abs().sum() > 0, f"{name} received an all-zero entropy gradient"


def test_save_load_preserves_conditioned_distribution_and_state_keys(tmp_path):
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs()
    _randomize_conditioning_outputs(policy)

    keys_before_forward = set(policy.state_dict())
    latent, base = _latent_and_base_logits(policy, obs)
    fn_ids = torch.tensor(
        [
            pysc2_actions.FUNCTIONS.Move_screen.id,
            pysc2_actions.FUNCTIONS.Attack_screen.id,
        ]
    )
    conditioned = policy._condition_argument_logits(base, latent, fn_ids, obs)
    entropy = policy._conditional_joint_entropy(base, latent, obs)
    assert set(policy.state_dict()) == keys_before_forward

    path = tmp_path / "policy.pt"
    policy.save(str(path))
    loaded = MaskedFlattenPolicy.load(str(path), device="cpu")
    assert set(loaded.state_dict()) == keys_before_forward

    loaded_latent, loaded_base = _latent_and_base_logits(loaded, obs)
    loaded_conditioned = loaded._condition_argument_logits(
        loaded_base, loaded_latent, fn_ids, obs
    )
    torch.testing.assert_close(loaded_conditioned, conditioned)
    torch.testing.assert_close(
        loaded._conditional_joint_entropy(loaded_base, loaded_latent, obs), entropy
    )


def test_batch_varying_masks_entropy_is_exact_and_deterministic():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs()
    _randomize_conditioning_outputs(policy)
    no_op_id = pysc2_actions.FUNCTIONS.no_op.id
    point_id = pysc2_actions.FUNCTIONS.select_point.id
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    attack_id = pysc2_actions.FUNCTIONS.Attack_screen.id

    # Different rows see different available sets, so the exact entropy must
    # marginalize over the per-batch union without assuming a shared mask.
    obs["available_actions"].zero_()
    obs["available_actions"][0, [no_op_id, point_id, move_id]] = 1.0
    obs["available_actions"][1, [no_op_id, attack_id]] = 1.0

    sizes = [int(x) for x in policy.action_space.nvec]
    latent, base = _latent_and_base_logits(policy, obs)
    fn_logits = torch.split(base, sizes, dim=1)[0]
    fn_logp = torch.log_softmax(fn_logits, dim=1)
    fn_probs = fn_logp.exp()
    assert fn_probs[0, attack_id] < 1e-6
    assert fn_probs[1, move_id] < 1e-6

    expected = -(fn_probs * fn_logp).sum(dim=1)
    for fn_id in (point_id, move_id, attack_id):
        branch_ids = torch.full((obs["vector"].shape[0],), fn_id, dtype=torch.long)
        branch = policy._condition_argument_logits(base, latent, branch_ids, obs)
        branch_splits = torch.split(branch, sizes, dim=1)
        branch_used = policy._fn_slot_mask[branch_ids]
        branch_entropy = torch.zeros_like(expected)
        for slot_idx in range(len(sizes) - 1):
            slot_logp = torch.log_softmax(branch_splits[slot_idx + 1], dim=1)
            branch_entropy = branch_entropy + (
                -(slot_logp.exp() * slot_logp).sum(dim=1) * branch_used[:, slot_idx]
            )
        expected = expected + fn_probs[:, fn_id] * branch_entropy

    entropy = policy._conditional_joint_entropy(base, latent, obs)
    torch.testing.assert_close(entropy, expected)

    # Repeated evaluation must be bitwise identical (no sampling in the path).
    assert torch.equal(entropy, policy._conditional_joint_entropy(base, latent, obs))


def test_mixed_batch_empty_available_row_falls_back_to_no_op_consistently():
    torch.manual_seed(0)
    policy = _make_policy()
    obs = _make_obs()
    _randomize_conditioning_outputs(policy)
    no_op_id = pysc2_actions.FUNCTIONS.no_op.id
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    sizes = [int(x) for x in policy.action_space.nvec]

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

    # Function entropy is zero (one-hot support in both rows); row 0 uses
    # no_op so its argument entropy is zero, row 1 contributes Move_screen's.
    branch_ids = torch.full((obs["vector"].shape[0],), move_id, dtype=torch.long)
    branch = policy._condition_argument_logits(base, latent, branch_ids, obs)
    branch_splits = torch.split(branch, sizes, dim=1)
    branch_used = policy._fn_slot_mask[branch_ids]
    branch_entropy = torch.zeros(obs["vector"].shape[0])
    for slot_idx in range(len(sizes) - 1):
        slot_logp = torch.log_softmax(branch_splits[slot_idx + 1], dim=1)
        branch_entropy = branch_entropy + (
            -(slot_logp.exp() * slot_logp).sum(dim=1) * branch_used[:, slot_idx]
        )
    expected = fn_probs[:, move_id] * branch_entropy
    assert entropy[0].item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(entropy, expected)


CONDITIONAL_PREFIXES = (
    "_function_embedding.",
    "_argument_conditioner.",
    "_screen_conditional_net.",
    "_minimap_conditional_net.",
)

# Parameter order from the pre-conditioning architecture. Keeping this literal
# guards the append-only optimizer-resume assumption: historical Adam indices
# are only valid while these parameters keep their positions.
HISTORICAL_PARAMETER_NAMES = (
    "_screen_spatial_log_scale",
    "_minimap_spatial_log_scale",
    "features_extractor._vector_ln.weight",
    "features_extractor._vector_ln.bias",
    "features_extractor._screen_embeddings.0.weight",
    "features_extractor._screen_embeddings.1.weight",
    "features_extractor._screen_embeddings.2.weight",
    "features_extractor._screen_embeddings.3.weight",
    "features_extractor._screen_embeddings.4.weight",
    "features_extractor._minimap_embeddings.0.weight",
    "features_extractor._minimap_embeddings.1.weight",
    "features_extractor._minimap_embeddings.2.weight",
    "features_extractor._minimap_embeddings.3.weight",
    "features_extractor._minimap_embeddings.4.weight",
    "features_extractor._minimap_embeddings.5.weight",
    "features_extractor._screen_cnn.0.0.weight",
    "features_extractor._screen_cnn.0.0.bias",
    "features_extractor._screen_cnn.0.2.weight",
    "features_extractor._screen_cnn.0.2.bias",
    "features_extractor._screen_cnn.1.weight",
    "features_extractor._screen_cnn.1.bias",
    "features_extractor._minimap_cnn.0.0.weight",
    "features_extractor._minimap_cnn.0.0.bias",
    "features_extractor._minimap_cnn.0.2.weight",
    "features_extractor._minimap_cnn.0.2.bias",
    "features_extractor._minimap_cnn.1.weight",
    "features_extractor._minimap_cnn.1.bias",
    "_screen_spatial_net.0.weight",
    "_screen_spatial_net.0.bias",
    "_screen_spatial_net.2.weight",
    "_screen_spatial_net.2.bias",
    "_minimap_spatial_net.0.weight",
    "_minimap_spatial_net.0.bias",
    "_minimap_spatial_net.2.weight",
    "_minimap_spatial_net.2.bias",
    "mlp_extractor.policy_net.0.weight",
    "mlp_extractor.policy_net.0.bias",
    "mlp_extractor.value_net.0.weight",
    "mlp_extractor.value_net.0.bias",
    "action_net.weight",
    "action_net.bias",
    "value_net.weight",
    "value_net.bias",
)


def test_synthetic_legacy_optimizer_resume_pads_without_remapping_moments():
    """Resume a pre-conditioning Adam state with no archive dependency."""
    torch.manual_seed(0)
    reference = _make_policy()
    current_names = [name for name, _ in reference.named_parameters()]
    historical_names = [name for name in current_names if not name.startswith(CONDITIONAL_PREFIXES)]
    conditional_names = [name for name in current_names if name.startswith(CONDITIONAL_PREFIXES)]

    # Regression: historical parameters keep their order and the new heads are
    # appended, which is what makes append-only Adam padding valid.
    assert historical_names == list(HISTORICAL_PARAMETER_NAMES)
    assert current_names == historical_names + conditional_names
    assert len(conditional_names) == 13

    # New heads are registered before optimizer construction.
    reference_optimizer_ids = {
        id(param) for group in reference.optimizer.param_groups for param in group["params"]
    }
    assert reference_optimizer_ids == {id(param) for param in reference.parameters()}
    reference_params = dict(reference.named_parameters())
    for name in conditional_names:
        assert id(reference_params[name]) in reference_optimizer_ids

    current_state = reference.optimizer.state_dict()
    assert len(current_state["param_groups"]) == 1
    legacy_group = dict(current_state["param_groups"][0])
    legacy_group["params"] = list(range(len(historical_names)))
    expected_moments = {}
    legacy_state = {}
    for index, name in enumerate(historical_names):
        exp_avg = torch.full_like(reference_params[name], float(index + 1))
        exp_avg_sq = torch.full_like(reference_params[name], float(index + 2))
        legacy_state[index] = {
            "step": torch.tensor(1.0),
            "exp_avg": exp_avg.clone(),
            "exp_avg_sq": exp_avg_sq.clone(),
        }
        expected_moments[name] = (exp_avg, exp_avg_sq)

    resumed = _make_policy()
    resumed_names = [name for name, _ in resumed.named_parameters()]
    assert resumed_names == current_names
    resumed.optimizer.load_state_dict({"state": legacy_state, "param_groups": [legacy_group]})

    resumed_params = dict(resumed.named_parameters())
    assert len(resumed.optimizer.param_groups) == 1
    group_params = resumed.optimizer.param_groups[0]["params"]
    assert [id(param) for param in group_params] == [id(param) for param in resumed.parameters()]

    # Historical moments are preserved by parameter name (no index remapping).
    for name in historical_names:
        state = resumed.optimizer.state[resumed_params[name]]
        torch.testing.assert_close(state["exp_avg"], expected_moments[name][0])
        torch.testing.assert_close(state["exp_avg_sq"], expected_moments[name][1])
        assert float(state["step"]) == 1.0

    # The newly appended conditional parameters start without Adam state and
    # initialize it lazily on the first step.
    for name in conditional_names:
        assert resumed_params[name] not in resumed.optimizer.state

    obs = _make_obs()
    actions, values, log_prob = resumed(obs)
    assert actions.shape[0] == obs["vector"].shape[0]
    resumed.optimizer.zero_grad()
    (log_prob.sum() + values.sum()).backward()
    resumed.optimizer.step()
    for name in conditional_names:
        state = resumed.optimizer.state[resumed_params[name]]
        assert "exp_avg" in state and "exp_avg_sq" in state
        assert torch.isfinite(state["exp_avg"]).all()
        assert torch.isfinite(state["exp_avg_sq"]).all()


def test_resume_tolerant_adam_rejects_incompatible_layouts():
    policy = _make_policy()
    current_state = policy.optimizer.state_dict()
    group = dict(current_state["param_groups"][0])

    with pytest.raises(ValueError, match="parameter groups"):
        policy.optimizer.load_state_dict(
            {"state": {}, "param_groups": [dict(group), dict(group)]}
        )

    too_many = dict(group)
    too_many["params"] = list(range(len(list(policy.parameters())) + 1))
    with pytest.raises(ValueError, match="more parameters"):
        policy.optimizer.load_state_dict({"state": {}, "param_groups": [too_many]})


def _dummy_obs_from_spaces(observation_space, batch_size=3):
    obs = {}
    for key, space in observation_space.spaces.items():
        obs[key] = torch.zeros(batch_size, *space.shape)
    obs["vector"].normal_()
    obs["screen"].normal_()
    obs["minimap"].normal_()
    obs["vector_mask"].fill_(1.0)
    obs["screen_layer_flags"].fill_(1.0)
    obs["minimap_layer_flags"].fill_(1.0)
    obs["available_actions"].zero_()
    return obs


def test_saved_collect_mineral_shards_archive_loads_without_env():
    if not os.path.isfile(SAVED_ARCHIVE):
        import pytest

        pytest.skip(f"saved archive not found: {SAVED_ARCHIVE}")

    from stable_baselines3 import PPO

    model = PPO.load(SAVED_ARCHIVE, env=None, device="cpu")
    policy = model.policy
    assert isinstance(policy, MaskedFlattenPolicy)
    for name in (
        "_function_embedding",
        "_argument_conditioner",
        "_screen_conditional_net",
        "_minimap_conditional_net",
    ):
        assert hasattr(policy, name), f"missing backward-compatible module {name}"

    # Every policy parameter, including the newly added conditional ones, must
    # be registered in the optimizer after loading an older checkpoint.
    optimizer_ids = {
        id(param) for group in policy.optimizer.param_groups for param in group["params"]
    }
    assert optimizer_ids == {id(param) for param in policy.parameters()}

    obs = _dummy_obs_from_spaces(model.observation_space)
    no_op_id = pysc2_actions.FUNCTIONS.no_op.id
    point_id = pysc2_actions.FUNCTIONS.select_point.id
    move_id = pysc2_actions.FUNCTIONS.Move_screen.id
    obs["available_actions"][:, [no_op_id, point_id, move_id]] = 1.0

    # The conditional output layers are zero-initialized for missing keys, so
    # the loaded archive behaves exactly like its pre-conditioning policy.
    latent, base = _latent_and_base_logits(policy, obs)
    fn_ids = torch.full((obs["vector"].shape[0],), move_id, dtype=torch.long)
    conditioned = policy._condition_argument_logits(base, latent, fn_ids, obs)
    legacy = policy._inject_spatial_logits(base.clone(), obs)
    torch.testing.assert_close(conditioned, legacy)

    keys_before_forward = set(policy.state_dict())
    actions, values, log_prob = policy(obs)
    assert set(policy.state_dict()) == keys_before_forward
    eval_values, eval_log_prob, entropy = policy.evaluate_actions(obs, actions)
    torch.testing.assert_close(eval_log_prob, log_prob)
    assert torch.isfinite(entropy).all()

    (log_prob.sum() + entropy.sum() + values.sum() + eval_values.sum()).backward()
    policy.optimizer.step()
