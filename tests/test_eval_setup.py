"""Eval/callback assembly tests (no SC2 launch)."""
import logging
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import train
from train import (
    PeriodicCheckpointCallback,
    build_callbacks,
    build_evaluation,
    should_prune_periodic_checkpoints,
)

LOGGER = logging.getLogger("train")


def test_eval_disabled_does_not_create_env_or_callback(caplog):
    created = []

    def make_env():
        created.append("env")
        return object()

    def build_callback(env):
        created.append("callback")
        return object()

    with caplog.at_level(logging.INFO, logger="train"):
        eval_env, eval_callback = build_evaluation(
            0, logger=LOGGER, make_env=make_env, build_callback=build_callback
        )

    assert eval_env is None
    assert eval_callback is None
    assert created == []
    assert "评估已禁用" in caplog.text


@pytest.mark.parametrize("eval_freq", [-16, 0])
def test_non_positive_eval_freq_is_disabled(eval_freq):
    eval_env, eval_callback = build_evaluation(
        eval_freq,
        logger=LOGGER,
        make_env=lambda: pytest.fail("eval env must not be created"),
        build_callback=lambda env: pytest.fail("eval callback must not be created"),
    )
    assert eval_env is None
    assert eval_callback is None


def test_eval_enabled_creates_env_and_callback():
    sentinel_env = object()
    sentinel_callback = object()
    calls = []

    def make_env():
        calls.append("env")
        return sentinel_env

    def build_callback(env):
        calls.append(("callback", env))
        return sentinel_callback

    eval_env, eval_callback = build_evaluation(
        32, logger=LOGGER, make_env=make_env, build_callback=build_callback
    )

    assert eval_env is sentinel_env
    assert eval_callback is sentinel_callback
    assert calls == ["env", ("callback", sentinel_env)]


def test_callback_failure_closes_eval_env_and_reraises():
    class FakeEnv:
        closed = False

        def close(self):
            self.closed = True

    fake_env = FakeEnv()

    def build_callback(env):
        raise RuntimeError("callback construction failed")

    with pytest.raises(RuntimeError, match="callback construction failed"):
        build_evaluation(
            32, logger=LOGGER, make_env=lambda: fake_env, build_callback=build_callback
        )
    assert fake_env.closed is True


def test_checkpoint_and_tb_callbacks_do_not_require_eval(tmp_path):
    callbacks, tb_callback = build_callbacks(
        eval_callback=None,
        out_dir=str(tmp_path),
        n_steps=16,
        n_envs=1,
        save_iters=5,
        summary_iters=10,
        tensorboard=True,
        tb_log=str(tmp_path / "tb"),
    )

    assert any(isinstance(cb, PeriodicCheckpointCallback) for cb in callbacks)
    assert tb_callback is not None
    assert tb_callback in callbacks
    assert len(callbacks) == 2


def test_eval_callback_is_included_when_present(tmp_path):
    sentinel_callback = object()
    callbacks, _ = build_callbacks(
        eval_callback=sentinel_callback,
        out_dir=str(tmp_path),
        n_steps=16,
        n_envs=1,
        save_iters=0,
        summary_iters=10,
        tensorboard=False,
        tb_log=None,
    )
    assert callbacks == [sentinel_callback]


def test_pruning_requires_both_best_model_and_eval():
    # eval disabled: periodic checkpoints are the only intermediate artifacts.
    assert should_prune_periodic_checkpoints(True, False) is False
    assert should_prune_periodic_checkpoints(False, True) is False
    assert should_prune_periodic_checkpoints(True, True) is True
    assert should_prune_periodic_checkpoints(False, False) is False
