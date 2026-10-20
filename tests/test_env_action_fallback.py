"""Action execution fallback tests using fake SC2 objects (no game launch)."""
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.envs.pysc2_gym_wrapper import PySC2GymEnv

NO_OP_ID = 0
POINT_FN_ID = 1


class FakeFunction:
    def __init__(self, fn_id, name, args=()):
        self.id = fn_id
        self.name = name
        self.args = list(args)

    def __call__(self, *args):
        # PySC2's Function.__call__ only wraps the arguments; it does not
        # validate their count, which is why SC2Env.step must be guarded too.
        return SimpleNamespace(fn_id=self.id, args=args)


class FakeFunctions:
    def __init__(self, functions):
        self._by_id = {fn.id: fn for fn in functions}
        self.no_op = self._by_id[NO_OP_ID]

    def __getitem__(self, fn_id):
        return self._by_id[int(fn_id)]

    def __len__(self):
        return len(self._by_id)


class FakeParser:
    def __init__(self, available):
        self.available = list(available)

    def parse(self, observation):
        mask = np.zeros(max(self.available) + 1, dtype=np.float32)
        mask[self.available] = 1.0
        return {"available_actions": mask}


class FakeSC2Env:
    def __init__(self, reject_non_noop=False, fail_all=False):
        self.reject_non_noop = reject_non_noop
        self.fail_all = fail_all
        self.calls = []

    def step(self, actions):
        call = actions[0]
        self.calls.append(call)
        if self.fail_all:
            raise RuntimeError("SC2 process unavailable")
        if call.fn_id != NO_OP_ID and self.reject_non_noop:
            raise ValueError("Wrong number of arguments")
        return [SimpleNamespace(reward=1.0, last=lambda: False, observation=None)]


def make_env(*, available=(NO_OP_ID, POINT_FN_ID), reject_non_noop=False, fail_all=False):
    screen_arg = SimpleNamespace(name="screen", sizes=[4, 4])
    functions = FakeFunctions(
        [
            FakeFunction(NO_OP_ID, "no_op"),
            FakeFunction(POINT_FN_ID, "select_point", [screen_arg]),
        ]
    )

    env = PySC2GymEnv.__new__(PySC2GymEnv)
    env.logger = logging.getLogger("test_wrapper")
    env.actions = SimpleNamespace(FUNCTIONS=functions)
    env._env = FakeSC2Env(reject_non_noop=reject_non_noop, fail_all=fail_all)
    env._parser = FakeParser(available)
    env.observation_space = {}
    env._fn_param_map = {NO_OP_ID: [], POINT_FN_ID: [0]}
    env._action_spec_types = SimpleNamespace(
        _fields=("screen",), screen=SimpleNamespace(sizes=[4, 4])
    )
    env.timestep = SimpleNamespace(observation=None, reward=0.0, last=lambda: True)
    env._step_count = 0
    return env


def test_valid_action_executes_normally():
    env = make_env()
    _, reward, terminated, truncated, info = env.step([POINT_FN_ID, 5])

    assert env._env.calls[-1].fn_id == POINT_FN_ID
    assert info["executed_fn_id"] == POINT_FN_ID
    assert info["arg_clipped"] is False
    assert info["fn_available"] is True
    assert reward == 1.0
    assert terminated is False and truncated is False


def test_illegal_params_fall_back_to_no_op():
    env = make_env(reject_non_noop=True)
    # 99 is outside the 4x4 screen range, so sanitization must clip.
    _, _, _, _, info = env.step([POINT_FN_ID, 99])

    assert env._env.calls[-1].fn_id == NO_OP_ID
    assert info["executed_fn_id"] == NO_OP_ID
    assert info["arg_clipped"] is True
    assert info["fn_available"] is True


def test_unavailable_function_falls_back_to_no_op():
    env = make_env(available=(NO_OP_ID,), reject_non_noop=True)
    _, _, _, _, info = env.step([POINT_FN_ID, 0])

    assert env._env.calls[-1].fn_id == NO_OP_ID
    assert info["executed_fn_id"] == NO_OP_ID
    assert info["arg_clipped"] is True
    assert info["fn_available"] is False


def test_low_level_step_rejection_falls_back_to_no_op():
    # Sanitization passes, but the underlying SC2Env rejects the call
    # ("Wrong number of arguments"); it must not leak to the caller.
    env = make_env(reject_non_noop=True)
    _, _, _, _, info = env.step([POINT_FN_ID, 0])

    assert [call.fn_id for call in env._env.calls] == [POINT_FN_ID, NO_OP_ID]
    assert info["executed_fn_id"] == NO_OP_ID
    assert info["arg_clipped"] is True


def test_no_op_failure_raises_clear_runtime_error():
    env = make_env(fail_all=True)
    with pytest.raises(RuntimeError, match="no_op"):
        env.step([POINT_FN_ID, 99])
