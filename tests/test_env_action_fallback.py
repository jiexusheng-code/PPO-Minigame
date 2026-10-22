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
        parsed = {"available_actions": mask}
        if hasattr(self, "screen"):
            parsed["screen"] = self.screen
        return parsed


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
    _, reward, terminated, truncated, info = env.step([POINT_FN_ID, 6])

    assert env._env.calls[-1].fn_id == POINT_FN_ID
    assert env._env.calls[-1].args == ([2, 1],)
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


def test_collect_mineral_shards_info_is_scoped_and_distinguishes_action_stages():
    generic = make_env()
    _, _, _, _, generic_info = generic.step([POINT_FN_ID, 6])
    assert not any(key.startswith("collect_mineral_shards/") for key in generic_info)

    env = make_env()
    env.map_name = "CollectMineralShards"
    env._parser.screen = np.zeros((4, 4, 2), dtype=np.float32)
    env._parser.screen[1:3, 1:3, 1] = 3
    _, _, _, _, info = env.step([POINT_FN_ID, 6])

    prefix = "collect_mineral_shards/"
    assert info[prefix + "requested_fn_name"] == "select_point"
    assert info[prefix + "requested_args"] == (("screen", 6),)
    assert info[prefix + "sanitized_args"] == (("screen", (2, 1)),)
    assert info[prefix + "executed_args"] == (("screen", (2, 1)),)
    assert info[prefix + "fallback"] is False
    assert info[prefix + "fallback_reason_code"] == "none"
    assert info[prefix + "raster_neutral_component_count_estimate_before"] == 1
    # Requested raw values stay explicitly flat; coordinates are decoded too.
    assert info[prefix + "requested_screen_flat"] == 6
    assert info[prefix + "requested_screen_x"] == 2
    assert info[prefix + "requested_screen_y"] == 1
    assert info[prefix + "sanitized_screen_x"] == 2
    assert info[prefix + "sanitized_screen_y"] == 1
    assert info[prefix + "executed_screen_x"] == 2
    assert info[prefix + "executed_screen_y"] == 1
    assert prefix + "raster_requested_target_to_neutral_component_distance_estimate_cells" in info
    assert prefix + "raster_executed_target_to_neutral_component_distance_estimate_cells" in info


def test_no_op_is_not_reported_as_a_fallback():
    env = make_env()
    env.map_name = "CollectMineralShards"
    env._parser.screen = np.zeros((4, 4, 2), dtype=np.float32)
    _, _, _, _, info = env.step([NO_OP_ID])

    prefix = "collect_mineral_shards/"
    assert info[prefix + "fallback"] is False
    assert info[prefix + "sanitization_failed"] is False
    assert info[prefix + "args_modified"] is False
    assert info[prefix + "args_bounds_violation"] is False
    assert info[prefix + "legacy_invalid_or_fallback"] is False
    assert info[prefix + "fallback_reason_code"] == "none"
    assert info[prefix + "executed_fn_name"] == "no_op"
    # no_op did not execute the requested target, so no executed-target metric.
    assert prefix + "raster_executed_target_to_neutral_component_distance_estimate_cells" not in info


def test_failed_sanitization_uses_invalid_sanitized_sentinel():
    env = make_env()
    env.map_name = "CollectMineralShards"
    env._parser.screen = np.zeros((4, 4, 2), dtype=np.float32)
    _, _, _, _, info = env.step([POINT_FN_ID, 99])

    prefix = "collect_mineral_shards/"
    assert info[prefix + "sanitization_failed"] is True
    assert info[prefix + "args_bounds_violation"] is True
    assert info[prefix + "args_modified"] is False
    assert info[prefix + "legacy_invalid_or_fallback"] is True
    assert info[prefix + "sanitized_fn_id"] == -1
    assert info[prefix + "sanitized_fn_name"] == "invalid"
    assert info[prefix + "sanitized_args"] == ()
    assert info[prefix + "fallback"] is True
    assert info[prefix + "fallback_reason_code"] == "sanitize_failed"
    assert info[prefix + "requested_fn_id"] == POINT_FN_ID
    assert info[prefix + "executed_fn_name"] == "no_op"
    # The raw requested index was out of range, so it has no decoded target.
    assert prefix + "requested_screen_x" not in info
    assert prefix + "raster_requested_target_to_neutral_component_distance_estimate_cells" not in info
    assert prefix + "raster_executed_target_to_neutral_component_distance_estimate_cells" not in info

    # An out-of-range function id also fails sanitization and must not be
    # reported as a valid sanitized function.
    invalid_fn = make_env()
    invalid_fn.map_name = "CollectMineralShards"
    invalid_fn._parser.screen = np.zeros((4, 4, 2), dtype=np.float32)
    _, _, _, _, info = invalid_fn.step([99, 0])

    assert info[prefix + "requested_fn_id"] == 99
    assert info[prefix + "requested_fn_name"] == "invalid"
    assert info[prefix + "sanitization_failed"] is True
    assert info[prefix + "args_bounds_violation"] is False
    assert info[prefix + "legacy_invalid_or_fallback"] is True
    assert info[prefix + "sanitized_fn_id"] == -1
    assert info[prefix + "sanitized_fn_name"] == "invalid"
    assert info[prefix + "sanitized_args"] == ()
    assert info[prefix + "fallback_reason_code"] == "sanitize_failed"
    assert info[prefix + "executed_fn_name"] == "no_op"


def test_unavailable_and_low_level_failures_report_reason_codes_and_scalars():
    unavailable = make_env(available=(NO_OP_ID,), reject_non_noop=True)
    unavailable.map_name = "CollectMineralShards"
    unavailable._parser.screen = np.zeros((4, 4, 2), dtype=np.float32)
    _, _, _, _, info = unavailable.step([POINT_FN_ID, 0])

    prefix = "collect_mineral_shards/"
    assert info[prefix + "sanitization_failed"] is False
    assert info[prefix + "args_bounds_violation"] is False
    assert info[prefix + "args_modified"] is False
    assert info[prefix + "legacy_invalid_or_fallback"] is True
    assert info[prefix + "sanitized_fn_id"] == POINT_FN_ID
    assert info[prefix + "fallback_reason_code"] == "fn_unavailable"
    assert info[prefix + "fallback"] is True
    assert info[prefix + "requested_screen_flat"] == 0
    assert info[prefix + "requested_screen_x"] == 0
    assert info[prefix + "requested_screen_y"] == 0
    assert info[prefix + "executed_fn_name"] == "no_op"
    assert prefix + "raster_executed_target_to_neutral_component_distance_estimate_cells" not in info

    low_level = make_env(reject_non_noop=True)
    low_level.map_name = "CollectMineralShards"
    low_level._parser.screen = np.zeros((4, 4, 2), dtype=np.float32)
    _, _, _, _, info = low_level.step([POINT_FN_ID, 0])

    assert info[prefix + "fallback_reason_code"] == "step_error"
    assert info[prefix + "fallback"] is True
    assert info[prefix + "args_bounds_violation"] is False
    assert info[prefix + "legacy_invalid_or_fallback"] is True
    assert info[prefix + "executed_fn_name"] == "no_op"
    assert prefix + "raster_requested_target_to_neutral_component_distance_estimate_cells" in info
    assert prefix + "raster_executed_target_to_neutral_component_distance_estimate_cells" not in info


def test_collect_mineral_shards_raster_estimates_use_synthetic_positions():
    env = PySC2GymEnv.__new__(PySC2GymEnv)
    before = np.zeros((8, 8, 2), dtype=np.float32)
    before[1:3, 1:3, 1] = 3
    before[5:7, 5:7, 1] = 3
    before[3:5, 2:4, 1] = 1
    after = np.zeros_like(before)
    after[5:7, 5:7, 1] = 3

    stats = env._collect_mineral_shards_diagnostics(
        {"screen": before},
        {"screen": after},
        requested_target_cell=(2, 2),
        executed_target_cell=(6, 6),
    )

    assert stats["collect_mineral_shards/raster_neutral_component_count_estimate_before"] == 2
    assert stats["collect_mineral_shards/raster_neutral_component_count_estimate_after"] == 1
    assert stats[
        "collect_mineral_shards/raster_requested_target_to_neutral_component_distance_estimate_cells"
    ] == pytest.approx(2 ** -0.5)
    assert stats[
        "collect_mineral_shards/raster_executed_target_to_neutral_component_distance_estimate_cells"
    ] == pytest.approx(2 ** -0.5)
    assert stats[
        "collect_mineral_shards/raster_marine_to_neutral_component_distance_estimate_before_cells"
    ] > 0

    requested_only = env._collect_mineral_shards_diagnostics(
        {"screen": before}, {"screen": after}, requested_target_cell=(2, 2)
    )
    assert (
        "collect_mineral_shards/raster_executed_target_to_neutral_component_distance_estimate_cells"
        not in requested_only
    )


def test_no_op_failure_raises_clear_runtime_error():
    env = make_env(fail_all=True)
    with pytest.raises(RuntimeError, match="no_op"):
        env.step([POINT_FN_ID, 99])
