"""Critic rollout diagnostics emitted by ``TBDualWriterCallback``.

No SC2 and no TensorBoard event files: a recording writer stands in for the
summary writer. One test drives a real (offline) PPO rollout so the callback
ordering and the real ``DictRolloutBuffer`` are exercised end to end.
"""
import copy
import glob
import json
import logging
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO

from pysc2.lib import actions as pysc2_actions

from src.callbacks.critic_diagnostics import (
    CRITIC_DIAG_PREFIX,
    CRITIC_SATURATION_SAMPLE_SIZE,
    DIAG_SAMPLE_INTERVAL_ENV_STEPS,
    DIAGNOSTICS_DIRNAME,
    compute_rollout_stats,
    sample_rollout,
)
from src.callbacks.tb_dual_writer import TBDualWriterCallback
from src.policies.masked_flatten_policy import MaskedFlattenPolicy, VectorLayerNormExtractor
from tests.test_shared_policy_integration import _OfflineEnv

# The whole suite is small; a single CPU thread keeps RNG/threading state
# comparisons meaningful.
torch.set_num_threads(1)

SPATIAL_SIZE = 32
SPATIAL_SLOT_SIZE = 1024
SCREEN_LAYERS = 8
MINIMAP_LAYERS = 5
VECTOR_DIM = 8


class RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[(tag, step)] = value

    def close(self):
        pass


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


def _make_policy(net_arch=None):
    _, slot_sizes = _param_semantics()
    n_funcs = len(pysc2_actions.FUNCTIONS)
    observation_space = gym.spaces.Dict(
        {
            "vector": gym.spaces.Box(-np.inf, np.inf, (VECTOR_DIM,), np.float32),
            "vector_mask": gym.spaces.Box(0.0, 1.0, (VECTOR_DIM,), np.float32),
            "screen": gym.spaces.Box(
                -np.inf, np.inf, (SPATIAL_SIZE, SPATIAL_SIZE, SCREEN_LAYERS), np.float32
            ),
            "minimap": gym.spaces.Box(
                -np.inf, np.inf, (SPATIAL_SIZE, SPATIAL_SIZE, MINIMAP_LAYERS), np.float32
            ),
            "available_actions": gym.spaces.Box(0.0, 1.0, (n_funcs,), np.float32),
            "screen_layer_flags": gym.spaces.Box(0.0, 1.0, (SCREEN_LAYERS,), np.float32),
            "minimap_layer_flags": gym.spaces.Box(0.0, 1.0, (MINIMAP_LAYERS,), np.float32),
        }
    )
    action_space = gym.spaces.MultiDiscrete([n_funcs] + slot_sizes)
    return MaskedFlattenPolicy(
        observation_space,
        action_space,
        lr_schedule=lambda _: 3e-4,
        features_extractor_class=VectorLayerNormExtractor,
        features_extractor_kwargs={"cnn_output_dim": 8},
        net_arch=dict(pi=[16], vf=[16, 16]) if net_arch is None else net_arch,
    )


def _make_buffer(n_steps, n_envs, seed=0):
    rng = np.random.default_rng(seed)
    n_funcs = len(pysc2_actions.FUNCTIONS)
    available = np.zeros((n_steps, n_envs, n_funcs), np.float32)
    available[..., 0] = 1.0
    available[..., 1] = 1.0
    observations = {
        "vector": rng.normal(size=(n_steps, n_envs, VECTOR_DIM)).astype(np.float32),
        "vector_mask": np.ones((n_steps, n_envs, VECTOR_DIM), np.float32),
        "screen": rng.integers(
            0, 4, size=(n_steps, n_envs, SPATIAL_SIZE, SPATIAL_SIZE, SCREEN_LAYERS)
        ).astype(np.float32),
        "minimap": rng.integers(
            0, 3, size=(n_steps, n_envs, SPATIAL_SIZE, SPATIAL_SIZE, MINIMAP_LAYERS)
        ).astype(np.float32),
        "available_actions": available,
        "screen_layer_flags": np.ones((n_steps, n_envs, SCREEN_LAYERS), np.float32),
        "minimap_layer_flags": np.ones((n_steps, n_envs, MINIMAP_LAYERS), np.float32),
    }
    values = rng.normal(size=(n_steps, n_envs, 1)).astype(np.float32)
    returns = values + rng.normal(scale=0.25, size=(n_steps, n_envs, 1)).astype(np.float32)
    rewards = rng.normal(scale=0.5, size=(n_steps, n_envs, 1)).astype(np.float32)
    advantages = returns - values
    episode_starts = np.zeros((n_steps, n_envs, 1), np.float32)
    episode_starts[0] = 1.0
    return SimpleNamespace(
        observations=observations,
        values=values,
        returns=returns,
        rewards=rewards,
        advantages=advantages,
        episode_starts=episode_starts,
        generator_ready=False,
        n_envs=n_envs,
    )


def _make_callback(policy, buffer, num_timesteps=1000, write_opt_every=1, base_log_dir="unused"):
    callback = TBDualWriterCallback(base_log_dir, write_opt_every=write_opt_every)
    callback.model = SimpleNamespace(
        num_timesteps=num_timesteps,
        policy=policy,
        rollout_buffer=buffer,
        device=torch.device("cpu"),
    )
    callback.opt_writer = RecordingWriter()
    return callback


def _saturation_tag(layer_index):
    return (
        f"{CRITIC_DIAG_PREFIX}/vf_tanh_layer{layer_index}_saturated_fraction_sample"
        f"{CRITIC_SATURATION_SAMPLE_SIZE}_abs_ge_0p999"
    )


def _vf_tanh_layers(policy):
    return [
        module for module in policy.mlp_extractor.value_net.modules()
        if isinstance(module, nn.Tanh)
    ]


def _npz_paths(base_log_dir):
    return sorted(glob.glob(os.path.join(str(base_log_dir), DIAGNOSTICS_DIRNAME, "*.npz")))


def _meta_paths(base_log_dir):
    return sorted(
        glob.glob(os.path.join(str(base_log_dir), DIAGNOSTICS_DIRNAME, "*_meta.json"))
    )


def _step_from_npz(path):
    match = re.search(r"critic_rollout_step(\d+)", os.path.basename(path))
    assert match is not None, path
    return int(match.group(1))


def _flatten_obs_buffer(buffer, key):
    return np.asarray(buffer.observations[key]).reshape(-1, *buffer.observations[key].shape[2:])


def _flatten_buffer_array(array):
    return np.asarray(array).reshape(-1)


def _assert_state_dicts_equal(before, after):
    assert set(before.keys()) == set(after.keys())
    for key in before:
        b, a = before[key], after[key]
        if isinstance(b, torch.Tensor):
            assert torch.equal(b, a), key
        elif isinstance(b, dict):
            _assert_state_dicts_equal(b, a)
        elif isinstance(b, (list, tuple)):
            assert len(b) == len(a)
            for i, (bi, ai) in enumerate(zip(b, a)):
                if isinstance(bi, torch.Tensor):
                    assert torch.equal(bi, ai), f"{key}[{i}]"
                else:
                    assert bi == ai, f"{key}[{i}]"
        else:
            assert b == a, key


def test_critic_diagnostics_report_rollout_scale_saturation_and_sample_count(tmp_path):
    torch.manual_seed(0)
    policy = _make_policy()
    buffer = _make_buffer(n_steps=20, n_envs=2)  # 40 flattened observations
    callback = _make_callback(policy, buffer, num_timesteps=1000, base_log_dir=tmp_path)
    tanh_layers = _vf_tanh_layers(policy)
    assert len(tanh_layers) == 2
    assert all(len(layer._forward_hooks) == 0 for layer in tanh_layers)

    callback._on_rollout_end()

    scalars = callback.opt_writer.scalars
    step = 1000  # x-axis is num_timesteps at the rollout end
    values = np.asarray(buffer.values, dtype=np.float64).reshape(-1)
    returns = np.asarray(buffer.returns, dtype=np.float64).reshape(-1)
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/value_mean", step)] == float(values.mean())
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/value_std_ddof0", step)] == float(values.std(ddof=0))
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/return_mean", step)] == float(returns.mean())
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/return_std_ddof0", step)] == float(returns.std(ddof=0))
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/value_min", step)] == float(values.min())
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/value_max", step)] == float(values.max())
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/value_finite_fraction", step)] == 1.0
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/residual_finite_fraction", step)] == 1.0
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/vf_tanh_saturation_sample_count", step)] == 32
    for layer_index in range(len(tanh_layers)):
        fraction = scalars[(_saturation_tag(layer_index), step)]
        assert 0.0 <= fraction <= 1.0
    for (tag, _), value in scalars.items():
        assert np.isfinite(value), f"{tag} is not finite"

    # Temporary hooks must not survive, and the buffer must not be flattened.
    assert all(len(layer._forward_hooks) == 0 for layer in tanh_layers)
    assert buffer.observations["screen"].shape[0] == 20
    assert buffer.generator_ready is False


def test_critic_saturation_sample_is_capped_and_uses_available_observations(tmp_path):
    torch.manual_seed(0)
    policy = _make_policy()
    buffer = _make_buffer(n_steps=6, n_envs=2)  # 12 flattened observations
    callback = _make_callback(policy, buffer, base_log_dir=tmp_path)

    callback._on_rollout_end()

    count = callback.opt_writer.scalars[
        (f"{CRITIC_DIAG_PREFIX}/vf_tanh_saturation_sample_count", 1000)
    ]
    assert count == 12


def test_critic_diagnostics_honour_write_opt_every_and_never_leak_hooks(tmp_path):
    torch.manual_seed(0)
    policy = _make_policy()
    buffer = _make_buffer(n_steps=20, n_envs=2)
    callback = _make_callback(
        policy, buffer, num_timesteps=1000, write_opt_every=2, base_log_dir=tmp_path
    )
    tanh_layers = _vf_tanh_layers(policy)

    callback._on_rollout_end()  # opt_step 0 -> written
    callback.model.num_timesteps = 1008
    callback._on_rollout_end()  # opt_step 1 -> skipped
    callback.model.num_timesteps = 1016
    callback._on_rollout_end()  # opt_step 2 -> written

    steps_with_critic_metrics = sorted(
        {step for (tag, step) in callback.opt_writer.scalars if tag.startswith(CRITIC_DIAG_PREFIX)}
    )
    assert steps_with_critic_metrics == [1000, 1016]
    assert all(len(layer._forward_hooks) == 0 for layer in tanh_layers)


def test_value_std_still_logged_when_value_mlp_has_no_tanh(tmp_path):
    torch.manual_seed(0)
    policy = _make_policy(net_arch=dict(pi=[16], vf=[]))
    assert _vf_tanh_layers(policy) == []
    buffer = _make_buffer(n_steps=4, n_envs=1)
    callback = _make_callback(policy, buffer, base_log_dir=tmp_path)

    callback._on_rollout_end()

    scalars = callback.opt_writer.scalars
    assert (f"{CRITIC_DIAG_PREFIX}/value_std_ddof0", 1000) in scalars
    assert (f"{CRITIC_DIAG_PREFIX}/return_std_ddof0", 1000) in scalars
    assert not any("vf_tanh" in tag for (tag, _) in scalars)


def test_failed_critic_probe_logs_warning_and_cleans_hooks_and_maps(
    tmp_path, monkeypatch, caplog
):
    policy = _make_policy()
    callback = _make_callback(
        policy, _make_buffer(n_steps=4, n_envs=2), base_log_dir=tmp_path
    )

    def fail_after_feature_extraction(_features):
        assert policy.features_extractor.last_spatial_maps
        raise RuntimeError("synthetic critic probe failure")

    monkeypatch.setattr(policy.mlp_extractor, "forward_critic", fail_after_feature_extraction)
    callback._on_rollout_end()

    assert "synthetic critic probe failure" in caplog.text
    assert policy.features_extractor.last_spatial_maps is None
    assert all(len(layer._forward_hooks) == 0 for layer in _vf_tanh_layers(policy))
    assert not any("vf_tanh" in tag for tag, _ in callback.opt_writer.scalars)
    assert (f"{CRITIC_DIAG_PREFIX}/value_std_ddof0", 1000) in callback.opt_writer.scalars


def test_npz_sample_aligns_time_env_and_matches_stored_arrays(tmp_path):
    torch.manual_seed(0)
    policy = _make_policy()
    buffer = _make_buffer(n_steps=20, n_envs=2)  # 40 transitions
    callback = _make_callback(policy, buffer, base_log_dir=tmp_path)

    callback._on_rollout_end()

    npz_paths = _npz_paths(tmp_path)
    assert len(npz_paths) == 1
    with np.load(npz_paths[0], allow_pickle=False) as sample:
        indices = sample["sample_indices_flat"]
        assert indices.size == 32
        assert np.unique(indices).size == indices.size
        assert indices[0] == 0 and indices[-1] == 39  # first and last transition
        assert np.all(np.diff(indices) >= 1)
        assert np.array_equal(sample["sample_indices_time"], indices // 2)
        assert np.array_equal(sample["sample_indices_env"], indices % 2)
        np.testing.assert_array_equal(
            sample["obs_screen"], _flatten_obs_buffer(buffer, "screen")[indices]
        )
        np.testing.assert_array_equal(
            sample["obs_vector"], _flatten_obs_buffer(buffer, "vector")[indices]
        )
        np.testing.assert_allclose(
            sample["stored_values"], _flatten_buffer_array(buffer.values)[indices]
        )
        np.testing.assert_allclose(
            sample["stored_returns"], _flatten_buffer_array(buffer.returns)[indices]
        )
        np.testing.assert_allclose(
            sample["stored_advantages"], _flatten_buffer_array(buffer.advantages)[indices]
        )
        np.testing.assert_allclose(
            sample["stored_rewards"], _flatten_buffer_array(buffer.rewards)[indices]
        )
        np.testing.assert_allclose(
            sample["episode_starts"], _flatten_buffer_array(buffer.episode_starts)[indices]
        )
        assert sample["current_values"].shape == (32,)
        assert np.all(np.isfinite(sample["current_values"]))
        assert sample["buffer_num_steps"] == 20
        assert sample["buffer_num_envs"] == 2
        assert sample["num_timesteps"] == 1000
        # Every array loads without pickle and is numeric.
        for key in sample.files:
            arr = sample[key]
            assert arr.dtype != object, key
            assert np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_, key
    with open(_meta_paths(tmp_path)[0], encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["timing"] == "rollout_end_before_ppo_train_preupdate"
    assert metadata["map"] == "unknown"
    assert metadata["sample"]["count"] == 32
    assert metadata["rollout"]["num_steps"] == 20
    assert metadata["rollout"]["num_envs"] == 2
    assert metadata["gradient"]["writes_param_grad"] is False
    assert metadata["gradient"]["optimizer_step"] is False
    assert metadata["thresholds"]["tanh_saturation_abs"] == 0.999
    groups = metadata["gradient"]["groups"]
    assert {"vf_first_layer", "vf_last_head", "screen_encoder", "minimap_encoder"} <= set(groups)
    assert groups["vf_first_layer"]["grad_l2"] is not None
    assert groups["vf_first_layer"]["grad_finite_fraction"] == 1.0
    assert metadata["sample"]["rule"].endswith("first_to_last_inclusive")


def test_sample_covers_multiple_envs_and_late_rollout_on_64x16():
    buffer = _make_buffer(n_steps=64, n_envs=16)  # 1024 transitions
    sample = sample_rollout(buffer, sample_size=32)

    assert sample.indices_flat.size == 32
    assert sample.indices_flat[0] == 0
    assert sample.indices_flat[-1] == 1023
    assert len(set(sample.indices_env.tolist())) == 16  # every env covered
    assert sample.indices_time.max() == 63  # late rollout is represented
    np.testing.assert_array_equal(sample.indices_time, sample.indices_flat // 16)
    np.testing.assert_array_equal(sample.indices_env, sample.indices_flat % 16)


def test_time_major_flatten_order_matches_buffer_indices():
    buffer = _make_buffer(n_steps=4, n_envs=3)  # 12 transitions < sample cap
    buffer.values = (
        100.0 * np.arange(4)[:, None, None] + np.arange(3)[None, :, None]
    ).astype(np.float32)

    sample = sample_rollout(buffer, sample_size=32)

    assert sample.indices_flat.size == 12
    np.testing.assert_array_equal(sample.indices_time, sample.indices_flat // 3)
    np.testing.assert_array_equal(sample.indices_env, sample.indices_flat % 3)
    expected = (100.0 * sample.indices_time + sample.indices_env).astype(np.float32)
    np.testing.assert_array_equal(sample.values, expected)


def test_flattened_buffer_is_rejected_instead_of_mislabelled(tmp_path, caplog):
    policy = _make_policy()
    buffer = _make_buffer(n_steps=4, n_envs=2)
    # Simulate SB3's get(): swapaxes(0, 1) is env-major, not time-major.
    for name in ("values", "returns", "advantages", "rewards", "episode_starts"):
        setattr(buffer, name, np.swapaxes(getattr(buffer, name), 0, 1).reshape(-1, 1))
    for key, value in buffer.observations.items():
        buffer.observations[key] = np.swapaxes(value, 0, 1).reshape(
            -1, *value.shape[2:]
        )
    buffer.generator_ready = True

    assert sample_rollout(buffer) is None
    assert compute_rollout_stats(buffer) == {}

    callback = _make_callback(policy, buffer, base_log_dir=tmp_path)
    callback._on_rollout_end()
    assert _npz_paths(tmp_path) == []
    assert callback.opt_writer.scalars == {}
    assert any("env-major" in record.message for record in caplog.records)


def test_real_ppo_buffer_after_get_is_rejected(tmp_path, caplog):
    env = _OfflineEnv()
    model = PPO(
        MaskedFlattenPolicy,
        env,
        device="cpu",
        seed=3,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": VectorLayerNormExtractor,
            "features_extractor_kwargs": {"cnn_output_dim": 8},
            "net_arch": dict(pi=[16], vf=[16]),
        },
    )
    model.learn(total_timesteps=8)
    assert model.rollout_buffer.generator_ready is True

    assert sample_rollout(model.rollout_buffer) is None
    assert compute_rollout_stats(model.rollout_buffer) == {}
    assert any("env-major" in record.message for record in caplog.records)
    env.close()


def test_npz_cadence_and_never_overwrites_existing_files(tmp_path):
    policy = _make_policy()
    buffer = _make_buffer(n_steps=4, n_envs=2)
    callback = _make_callback(policy, buffer, num_timesteps=1000, base_log_dir=tmp_path)

    callback._on_rollout_end()  # first complete rollout -> saved
    callback.model.num_timesteps = 1000 + DIAG_SAMPLE_INTERVAL_ENV_STEPS - 1
    callback._on_rollout_end()  # below cadence -> no new file
    assert len(_npz_paths(tmp_path)) == 1
    callback.model.num_timesteps = 1000 + DIAG_SAMPLE_INTERVAL_ENV_STEPS
    callback._on_rollout_end()  # cadence reached -> saved
    steps = [_step_from_npz(path) for path in _npz_paths(tmp_path)]
    assert len(steps) == 2
    # Saves land on the first rollout end at/after the 50k threshold.
    assert DIAG_SAMPLE_INTERVAL_ENV_STEPS <= steps[1] - steps[0] < (
        DIAG_SAMPLE_INTERVAL_ENV_STEPS + 4 * 2
    )

    # A new callback (e.g. after a resume) at the same num_timesteps must not
    # overwrite the earlier run's file.
    second = _make_callback(policy, buffer, num_timesteps=steps[1], base_log_dir=tmp_path)
    second._on_rollout_end()
    all_paths = _npz_paths(tmp_path)
    assert len(all_paths) == 3
    assert len({os.path.basename(path) for path in all_paths}) == 3
    assert sorted(_step_from_npz(path) for path in all_paths) == [steps[0], steps[1], steps[1]]
    assert len(_meta_paths(tmp_path)) == 3


def test_diagnostics_do_not_change_params_grads_rng_optimizer_or_buffer(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    policy = _make_policy()
    buffer = _make_buffer(n_steps=8, n_envs=2)
    callback = _make_callback(policy, buffer, num_timesteps=1000, base_log_dir=tmp_path)

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    policy.optimizer = optimizer
    # Create real Adam moments with one optimizer step, then leave the policy
    # with mixed gradients: half the parameters carry a sentinel gradient and
    # half are None.
    warmup_loss = sum(param.float().sum() for param in policy.parameters())
    warmup_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert optimizer.state, "Adam state should exist after one step"
    for index, param in enumerate(policy.parameters()):
        param.grad = None if index % 2 else torch.full_like(param, 0.25)
    assert any(p.grad is None for p in policy.parameters())
    assert any(p.grad is not None for p in policy.parameters())

    params_before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    grads_before = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in policy.named_parameters()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    torch_rng_before = torch.get_rng_state().clone()
    numpy_rng_before = copy.deepcopy(np.random.get_state())
    training_before = policy.training
    buffer_arrays_before = {
        name: np.array(getattr(buffer, name), copy=True)
        for name in ("values", "returns", "advantages", "rewards", "episode_starts")
    }
    buffer_obs_before = {
        key: np.array(value, copy=True) for key, value in buffer.observations.items()
    }
    generator_ready_before = buffer.generator_ready

    callback._on_rollout_end()

    for name, param in policy.named_parameters():
        assert torch.equal(params_before[name], param.detach()), name
        if grads_before[name] is None:
            assert param.grad is None, name
        else:
            assert param.grad is not None, name
            assert torch.equal(grads_before[name], param.grad.detach()), name
    _assert_state_dicts_equal(optimizer_before, optimizer.state_dict())
    assert torch.equal(torch_rng_before, torch.get_rng_state())
    numpy_rng_after = np.random.get_state()
    assert numpy_rng_before[0] == numpy_rng_after[0]
    np.testing.assert_array_equal(numpy_rng_before[1], numpy_rng_after[1])
    assert numpy_rng_before[2:] == numpy_rng_after[2:]
    assert policy.training is training_before
    assert buffer.generator_ready is generator_ready_before
    for name, before in buffer_arrays_before.items():
        np.testing.assert_array_equal(getattr(buffer, name), before, err_msg=name)
    for key, before in buffer_obs_before.items():
        np.testing.assert_array_equal(buffer.observations[key], before, err_msg=key)
    assert all(len(module._forward_hooks) == 0 for module in policy.modules())
    assert policy.features_extractor.last_spatial_maps is None

    # The sparse gradient probe still reports evidence on the sample cadence.
    scalars = callback.opt_writer.scalars
    grad_tags = [tag for (tag, _) in scalars if "/grad_preupdate/" in tag]
    assert grad_tags
    assert any(tag.endswith("vf_first_layer_grad_l2") for tag in grad_tags)
    assert any(tag.endswith("screen_encoder_grad_l2") for tag in grad_tags)
    assert any(tag.endswith("minimap_encoder_grad_l2") for tag in grad_tags)


def test_saturation_and_nonfinite_activations_are_visible(tmp_path):
    torch.manual_seed(0)
    policy = _make_policy()
    # Saturate the first value-MLP Tanh for every finite sample.
    policy.mlp_extractor.value_net[0].bias.data.fill_(100.0)
    buffer = _make_buffer(n_steps=4, n_envs=2)
    # Channel 4 is the continuous unit_hit_points_ratio layer; a NaN in a
    # categorical layer would fail earlier in the embedding lookup.
    buffer.observations["screen"][0, 0, 0, 0, 4] = np.nan
    callback = _make_callback(policy, buffer, base_log_dir=tmp_path)

    callback._on_rollout_end()

    scalars = callback.opt_writer.scalars
    screen_finite = scalars[
        (f"{CRITIC_DIAG_PREFIX}/act/features_screen_encoder/finite_fraction", 1000)
    ]
    assert 0.0 < screen_finite < 1.0
    vf_finite = scalars[
        (f"{CRITIC_DIAG_PREFIX}/act/vf_linear_0/finite_fraction", 1000)
    ]
    assert vf_finite < 1.0
    saturation = scalars[
        (
            f"{CRITIC_DIAG_PREFIX}/act/vf_tanh_0/"
            "saturated_fraction_abs_ge_0p999",
            1000,
        )
    ]
    assert saturation > 0.9
    assert scalars[(f"{CRITIC_DIAG_PREFIX}/grad_preupdate/vf_first_layer_grad_finite", 1000)] == 0.0

    with np.load(_npz_paths(tmp_path)[0], allow_pickle=False) as sample:
        nonfinite = sample["act_features_screen_encoder_per_sample_nonfinite_fraction"]
        assert nonfinite[0] > 0.0
        assert np.all(nonfinite[1:] == 0.0)
        # Non-finite gradient metrics stay in the npz as NaN.
        assert np.isnan(sample["grad_l2_vf_first_layer"])
        assert sample["grad_params_used_fraction_vf_first_layer"] == 1.0
    with open(_meta_paths(tmp_path)[0], encoding="utf-8") as handle:
        metadata = json.load(handle)
    group = metadata["gradient"]["groups"]["vf_first_layer"]
    assert group["grad_l2"] is None
    assert group["grad_finite_fraction"] == 0.0
    assert group["params_used_fraction"] == 1.0


def test_npz_write_failure_logs_warning_and_training_continues(
    tmp_path, monkeypatch, caplog
):
    torch.manual_seed(0)
    policy = _make_policy()
    callback = _make_callback(
        policy, _make_buffer(n_steps=4, n_envs=2), base_log_dir=tmp_path
    )

    def fail_save(*_args, **_kwargs):
        raise OSError("synthetic disk full")

    import src.callbacks.critic_diagnostics as diagnostics_module

    monkeypatch.setattr(diagnostics_module.np, "savez_compressed", fail_save)
    with caplog.at_level(logging.WARNING):
        callback._on_rollout_end()

    assert "样本写入失败" in caplog.text
    assert _npz_paths(tmp_path) == []
    assert callback._critic_diag.last_sample_step is None
    # Scalars were still written before the disk failure.
    assert (f"{CRITIC_DIAG_PREFIX}/value_mean", 1000) in callback.opt_writer.scalars
    assert not glob.glob(os.path.join(str(tmp_path), DIAGNOSTICS_DIRNAME, "*.tmp.npz"))


def test_callback_rollout_diagnostics_inside_real_ppo_rollout(tmp_path):
    """Real SB3 ordering: returns exist and metrics match the second rollout."""
    env = _OfflineEnv()
    callback = TBDualWriterCallback(str(tmp_path))
    writer = RecordingWriter()
    callback._on_training_start = lambda: setattr(callback, "opt_writer", writer)

    model = PPO(
        MaskedFlattenPolicy,
        env,
        device="cpu",
        seed=7,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": VectorLayerNormExtractor,
            "features_extractor_kwargs": {"cnn_output_dim": 8},
            "net_arch": dict(pi=[16], vf=[16, 16]),
        },
    )
    model.learn(total_timesteps=16, callback=callback)

    scalars = writer.scalars
    steps = {step for (tag, step) in scalars if tag.startswith(CRITIC_DIAG_PREFIX)}
    assert steps == {8, 16}
    for step in (8, 16):
        for tag in (
            f"{CRITIC_DIAG_PREFIX}/value_std_ddof0",
            f"{CRITIC_DIAG_PREFIX}/return_std_ddof0",
            _saturation_tag(0),
            _saturation_tag(1),
        ):
            assert (tag, step) in scalars, f"missing {tag} at step {step}"
        assert scalars[(f"{CRITIC_DIAG_PREFIX}/vf_tanh_saturation_sample_count", step)] == 8
    assert all(np.isfinite(value) for value in scalars.values())
    assert all(len(layer._forward_hooks) == 0 for layer in _vf_tanh_layers(model.policy))

    # The last logged rollout must match the buffer captured before train().
    buffer_values = np.asarray(model.rollout_buffer.values, dtype=np.float64).reshape(-1)
    buffer_returns = np.asarray(model.rollout_buffer.returns, dtype=np.float64).reshape(-1)
    assert np.isclose(
        scalars[(f"{CRITIC_DIAG_PREFIX}/value_std_ddof0", 16)], buffer_values.std(ddof=0)
    )
    assert np.isclose(
        scalars[(f"{CRITIC_DIAG_PREFIX}/return_std_ddof0", 16)], buffer_returns.std(ddof=0)
    )
    env.close()


def test_real_ppo_rollout_writes_npz_with_preupdate_current_values(tmp_path):
    env = _OfflineEnv()
    callback = TBDualWriterCallback(str(tmp_path))
    model = PPO(
        MaskedFlattenPolicy,
        env,
        device="cpu",
        seed=7,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={
            "features_extractor_class": VectorLayerNormExtractor,
            "features_extractor_kwargs": {"cnn_output_dim": 8},
            "net_arch": dict(pi=[16], vf=[16, 16]),
        },
    )
    model.learn(total_timesteps=16, callback=callback)

    # First rollout (step 8) is saved; the second (step 16) is below cadence.
    paths = _npz_paths(tmp_path)
    assert len(paths) == 1
    assert _step_from_npz(paths[0]) == 8
    with np.load(paths[0], allow_pickle=False) as sample:
        assert sample["sample_indices_flat"].size == 8
        assert np.array_equal(sample["sample_indices_time"], np.arange(8))
        assert np.array_equal(sample["sample_indices_env"], np.zeros(8, dtype=np.int64))
        assert sample["obs_screen"].shape == (8, SPATIAL_SIZE, SPATIAL_SIZE, SCREEN_LAYERS)
        assert sample["stored_returns"].shape == (8,)
        assert sample["stored_advantages"].shape == (8,)
        assert sample["stored_rewards"].shape == (8,)
        assert sample["episode_starts"].shape == (8,)
        # Pre-update: current recomputation equals the values stored during
        # the same rollout (no PPO update has run yet).
        np.testing.assert_allclose(
            sample["current_values"], sample["stored_values"], rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            sample["stored_returns"],
            sample["stored_values"] + sample["stored_advantages"],
            rtol=1e-4,
            atol=1e-5,
        )
        for key in sample.files:
            assert sample[key].dtype != object, key
    with open(_meta_paths(tmp_path)[0], encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["num_timesteps"] == 8
    assert metadata["timing"] == "rollout_end_before_ppo_train_preupdate"
    assert metadata["sample"]["count"] == 8
    env.close()
