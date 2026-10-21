"""Pure/seeded tests for reproducibility helpers (no SC2 launch)."""
import logging
import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import train
from src.envs.pysc2_gym_wrapper import PySC2GymEnv
from train import (
    EVAL_SEED_OFFSET,
    MaskedFlattenPolicy,
    VectorLayerNormExtractor,
    build_config_snapshot,
    compute_env_seed,
    compute_eval_seed_offset,
    compute_learn_target,
    compute_resume_offsets,
    resolve_policy_and_kwargs,
    sanitize_for_yaml,
)


# ---------------------------------------------------------------- env seeds

def test_training_env_seeds_are_seed_plus_rank():
    assert [compute_env_seed(7, rank) for rank in range(4)] == [7, 8, 9, 10]


def test_eval_env_seeds_use_an_independent_range():
    train_seeds = {compute_env_seed(7, rank) for rank in range(8)}
    eval_seeds = {
        compute_env_seed(7, rank, offset=EVAL_SEED_OFFSET) for rank in range(4)
    }
    assert eval_seeds == {107, 108, 109, 110}
    assert train_seeds.isdisjoint(eval_seeds)


def test_eval_seed_offset_scales_past_large_training_env_counts():
    n_envs = 200
    offset = compute_eval_seed_offset(n_envs)
    train_seeds = {compute_env_seed(5, rank) for rank in range(n_envs)}
    eval_seeds = {
        compute_env_seed(5, rank, offset=offset) for rank in range(2)
    }
    assert train_seeds.isdisjoint(eval_seeds)


def test_compute_env_seed_rejects_negative_seeds():
    with pytest.raises(ValueError):
        compute_env_seed(-1, 0)
    with pytest.raises(ValueError):
        compute_env_seed(None, 0)


class _FakeGymEnv(gym.Env):
    observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


def test_make_seeded_vec_env_passes_seed_plus_rank_to_constructor(monkeypatch):
    created = []

    def fake_make_env_fn(map_name, env_kwargs=None, random_seed=None):
        def _init():
            created.append(random_seed)
            return _FakeGymEnv()

        return _init

    monkeypatch.setattr(train, "make_env_fn", fake_make_env_fn)

    # DummyVecEnv keeps envs in-process so the recorded constructor seeds and
    # the VecEnv seed list are observable here.
    vec_env = train.make_seeded_vec_env(
        "MoveToBeacon", {}, 3, seed=10, vec_env_cls="dummy"
    )
    assert created == [10, 11, 12]
    assert vec_env._seeds == [10, 11, 12]

    eval_env = train.make_seeded_vec_env(
        "MoveToBeacon", {}, 2, seed=10, offset=EVAL_SEED_OFFSET,
        vec_env_cls="dummy",
    )
    assert created[-2:] == [110, 111]
    assert eval_env._seeds == [110, 111]


# ------------------------------------------------- resuming counter semantics

def test_resume_appends_steps_without_resetting_counter():
    assert compute_learn_target(1000, 500, reset_num_timesteps=False) == 1500
    assert compute_learn_target(1000, 500, reset_num_timesteps=True) == 1000


def test_resume_offsets_continue_eval_and_opt_counters():
    offsets = compute_resume_offsets(16 * 32 * 5, n_envs=32, n_steps=16)
    assert offsets == {"vec_steps": 80, "opt_steps": 5}


# ------------------------------------------------- wrapper seeding (no SC2)

class _FakeTimestep:
    observation = None

    def last(self):
        return False


class _FakeSC2Env:
    def __init__(self):
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self):
        self.reset_calls += 1
        return [_FakeTimestep()]

    def step(self, actions):
        self.step_calls += 1
        return [_FakeTimestep()]

    def close(self):
        pass


def _bare_wrapper(sc2_seed):
    env = object.__new__(PySC2GymEnv)
    env.reset_jitter_max_steps = 0
    env._init_jitter_done = False
    env._sc2_random_seed = sc2_seed
    env._env = _FakeSC2Env()
    env.timestep = None
    env.logger = logging.getLogger("test_wrapper")
    env._obs_to_space = lambda timestep: {}
    return env


def test_wrapper_reset_seeds_np_random_reproducibly():
    first = _bare_wrapper(123)
    second = _bare_wrapper(123)
    other = _bare_wrapper(456)
    first.reset(seed=123)
    second.reset(seed=123)
    other.reset(seed=456)

    seq_first = first.np_random.integers(0, 2**31, size=8)
    seq_second = second.np_random.integers(0, 2**31, size=8)
    seq_other = other.np_random.integers(0, 2**31, size=8)

    assert np.array_equal(seq_first, seq_second)
    assert not np.array_equal(seq_first, seq_other)


def test_wrapper_jitter_is_deterministic_for_same_seed():
    class _NoOp:
        @staticmethod
        def no_op():
            return object()

    import types

    env_a = _bare_wrapper(42)
    env_b = _bare_wrapper(42)
    for env in (env_a, env_b):
        env.reset_jitter_max_steps = 20
        env.actions = types.SimpleNamespace(FUNCTIONS=_NoOp())
        env._obs_to_space = lambda timestep: {}
        env.reset(seed=42)

    assert env_a._env.step_calls == env_b._env.step_calls
    assert env_a._init_jitter_done is True


# ------------------------------------------------- config snapshot

def test_runtime_policy_kwargs_do_not_mutate_the_config():
    cfg_kwargs = {"features_extractor_kwargs": {"cnn_output_dim": 256}}

    policy, runtime_kwargs = resolve_policy_and_kwargs(
        "MaskedFlattenPolicy", cfg_kwargs
    )

    assert policy is MaskedFlattenPolicy
    assert runtime_kwargs["features_extractor_class"] is VectorLayerNormExtractor
    assert "features_extractor_class" not in cfg_kwargs


def test_string_class_name_is_rejected_not_treated_as_callable():
    with pytest.raises(ValueError, match="features_extractor_class"):
        resolve_policy_and_kwargs(
            "MaskedFlattenPolicy",
            {"features_extractor_class": "src.policies.masked_flatten_policy.X"},
        )


def test_config_snapshot_is_yaml_safe_and_rerunnable():
    cfg = {
        "policy": "MaskedFlattenPolicy",
        "policy_kwargs": {"features_extractor_kwargs": {"cnn_output_dim": 256}},
        "n_envs": 1,
    }
    snapshot = build_config_snapshot(
        cfg,
        {
            "policy": MaskedFlattenPolicy,
            "features_extractor_class": VectorLayerNormExtractor,
        },
        "configs/ppo_config.yaml",
    )
    loaded = yaml.safe_load(yaml.safe_dump(snapshot, allow_unicode=True))

    # Runtime objects stay out of the keys train.py feeds to PPO...
    assert loaded["policy_kwargs"] == cfg["policy_kwargs"]
    assert "features_extractor_class" not in loaded["policy_kwargs"]
    # ...and are recorded as stable, readable strings in metadata only.
    resolved = loaded["_snapshot"]["resolved"]
    assert resolved["policy"].endswith("MaskedFlattenPolicy")
    assert resolved["features_extractor_class"].endswith("VectorLayerNormExtractor")


def test_sanitize_for_yaml_converts_classes_to_stable_names():
    assert sanitize_for_yaml(MaskedFlattenPolicy).endswith("MaskedFlattenPolicy")
    assert sanitize_for_yaml({"x": [VectorLayerNormExtractor]}) == {
        "x": ["src.policies.masked_flatten_policy.VectorLayerNormExtractor"]
    }
