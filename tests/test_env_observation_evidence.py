"""Observation-evidence payload tests for the PySC2 wrapper (no SC2 launch).

These tests use fake SC2 objects and check that the trajectory payload records
what PySC2 actually exposed: feature_units rows, last_actions/action_result
availability, and the active canonical parser rasters.
"""
import json
import logging
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pysc2.lib import features

from src.envs.pysc2_gym_wrapper import PySC2GymEnv

NO_OP_ID = 0
POINT_FN_ID = 1
FIELD_COUNT = len(features.FeatureUnit)


class FakeFunction:
    def __init__(self, fn_id, name, args=()):
        self.id = fn_id
        self.name = name
        self.args = list(args)

    def __call__(self, *args):
        return SimpleNamespace(fn_id=self.id, args=args)


class FakeFunctions:
    def __init__(self, functions):
        self._by_id = {fn.id: fn for fn in functions}
        self.no_op = self._by_id[NO_OP_ID]

    def __getitem__(self, fn_id):
        return self._by_id[int(fn_id)]

    def __len__(self):
        return len(self._by_id)


class TwoPhaseParser:
    """Return the pre-step parse first and the post-step parse second."""

    def __init__(
        self,
        before_screen,
        after_screen,
        *,
        before_minimap=None,
        after_minimap=None,
        flags=True,
        vector=True,
        include_active_config=True,
    ):
        self.before_screen = before_screen
        self.after_screen = after_screen
        self.before_minimap = before_minimap
        self.after_minimap = after_minimap
        self.flags = flags
        self.vector = vector
        self.calls = 0
        self.config = SimpleNamespace(
            screen_layers=[
                (0, "visibility_map"),
                (1, "player_relative"),
                (2, "unit_type"),
                (3, "selected"),
            ],
            minimap_layers=[
                (0, "visibility_map"),
                (1, "camera"),
                (2, "player_relative"),
                (3, "selected"),
                (4, "buildable"),
            ],
        )
        if include_active_config:
            self.config.screen_active_layers = {"player_relative", "selected"}
            self.config.minimap_active_layers = {"selected"}

    def parse(self, observation):
        index = 0 if self.calls == 0 else 1
        self.calls += 1
        parsed = {
            "available_actions": np.array([1.0, 1.0, 0.0], dtype=np.float32),
            "screen": self.before_screen if index == 0 else self.after_screen,
        }
        if self.before_minimap is not None:
            parsed["minimap"] = (
                self.before_minimap if index == 0 else self.after_minimap
            )
        if self.flags:
            parsed["screen_layer_flags"] = np.array(
                [0.0, 1.0, 0.0, 1.0], dtype=np.float32
            )
            parsed["minimap_layer_flags"] = np.array(
                [0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32
            )
        if self.vector:
            parsed["vector"] = np.array([1.0, np.nan, np.inf], dtype=np.float32)
            parsed["vector_mask"] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        return parsed


class FakeSC2Env:
    def __init__(self, post_observation):
        self.post_observation = post_observation
        self.calls = []

    def step(self, actions):
        self.calls.append(actions[0])
        return [
            SimpleNamespace(
                reward=1.5, last=lambda: False, observation=self.post_observation
            )
        ]


def _unit_rows():
    return [
        {
            "unit_type": 48,
            "alliance": 1,
            "x": 3,
            "y": 4,
            "is_selected": 1,
            "health": 45,
            "health_ratio": 255,
            "shield": 0,
            "shield_ratio": 0,
            "energy": 0,
            "energy_ratio": 0,
            "display_type": 1,
            "radius": 4.5,
            "active": 1,
            "is_on_screen": 1,
            "order_length": 1,
            "order_id_0": 7,
            "order_id_1": 8,
            "tag": 0,
        },
        {
            "unit_type": 48,
            "alliance": 1,
            "x": 6,
            "y": 2,
            "is_selected": 0,
            "health": 45,
            "health_ratio": 255,
            "active": 1,
            "is_on_screen": 1,
            "tag": 0,
        },
        {
            "unit_type": 1680,
            "alliance": 3,
            "x": 7,
            "y": 7,
            "is_selected": 0,
            "tag": 0,
        },
    ]


def _feature_units(rows):
    array = np.zeros((len(rows), FIELD_COUNT), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for field_name, value in row.items():
            array[row_index, int(getattr(features.FeatureUnit, field_name))] = value
    return array


def _unit_layer_rows(rows):
    array = np.zeros((len(rows), len(features.UnitLayer)), dtype=np.int32)
    for row_index, row in enumerate(rows):
        for field_name, value in row.items():
            array[row_index, int(getattr(features.UnitLayer, field_name))] = value
    return array


def _raw_observation(
    units,
    *,
    last_actions=None,
    action_result=None,
    action_errors=None,
    single_select=None,
    multi_select=None,
):
    fields = {"feature_units": units}
    if last_actions is not None:
        fields["last_actions"] = np.asarray(last_actions, dtype=np.int32)
    if action_result is not None:
        fields["action_result"] = np.asarray(action_result, dtype=np.int32)
    if action_errors is not None:
        fields["action_errors"] = action_errors
    if single_select is not None:
        fields["single_select"] = _unit_layer_rows(single_select)
    if multi_select is not None:
        fields["multi_select"] = _unit_layer_rows(multi_select)
    return SimpleNamespace(**fields)


def _screens():
    before = np.zeros((8, 8, 4), dtype=np.float32)
    before[1, 1, 1] = 1
    before[1, 1, 3] = 1
    before[6, 6, 1] = 3
    after = np.zeros((8, 8, 4), dtype=np.float32)
    after[2, 2, 1] = 1
    after[2, 2, 3] = 1
    return before, after


def _minimaps():
    before = np.zeros((8, 8, 5), dtype=np.float32)
    before[4, 4, 3] = 1
    after = np.zeros((8, 8, 5), dtype=np.float32)
    after[5, 5, 3] = 1
    return before, after


def make_env(
    parser,
    *,
    raw_before,
    raw_after,
):
    env = PySC2GymEnv.__new__(PySC2GymEnv)
    env.logger = logging.getLogger("test_observation_evidence")
    env.actions = SimpleNamespace(
        FUNCTIONS=FakeFunctions(
            [
                FakeFunction(NO_OP_ID, "no_op"),
                FakeFunction(
                    POINT_FN_ID,
                    "select_point",
                    [SimpleNamespace(name="screen", sizes=[8, 8])],
                ),
            ]
        )
    )
    env._env = FakeSC2Env(raw_after)
    env._parser = parser
    env.observation_space = {}
    env._fn_param_map = {NO_OP_ID: [], POINT_FN_ID: [0]}
    env._action_spec_types = SimpleNamespace(
        _fields=("screen",), screen=SimpleNamespace(sizes=[8, 8])
    )
    env.timestep = SimpleNamespace(
        observation=raw_before, reward=0.0, last=lambda: False
    )
    env._step_count = 0
    env._vec_env_rank = 0
    env.map_name = "CollectMineralShards"
    env._episode_id = 7
    env._episode_step = 0
    env._episode_seed_source = "constructor"
    env._sc2_random_seed = 0
    return env


def run_step(**parser_kwargs):
    before_screen, after_screen = _screens()
    before_minimap, after_minimap = _minimaps()
    parser = TwoPhaseParser(
        before_screen,
        after_screen,
        before_minimap=before_minimap,
        after_minimap=after_minimap,
        **parser_kwargs,
    )
    raw_before = _raw_observation(
        _feature_units(_unit_rows()),
        last_actions=[0],
        action_result=[0],
        single_select=[
            {
                "unit_type": 48,
                "player_relative": 1,
                "health": 45,
                "shields": 0,
                "energy": 0,
                "transport_slots_taken": 0,
                "build_progress": 100,
            }
        ],
    )
    raw_after = _raw_observation(
        _feature_units(_unit_rows()),
        last_actions=[POINT_FN_ID],
        action_result=[0],
        multi_select=[
            {"unit_type": 48, "player_relative": 1, "health": 45},
            {"unit_type": 48, "player_relative": 1, "health": 44},
        ],
    )
    env = make_env(parser, raw_before=raw_before, raw_after=raw_after)
    _, _, _, _, info = env.step([POINT_FN_ID, 10])
    return env, info["collect_mineral_shards/trajectory"]


def test_payload_records_feature_units_and_engine_feedback():
    _, payload = run_step()

    pre_units = payload["pre"]["feature_units"]
    assert pre_units["available"] is True
    assert pre_units["count"] == 2  # neutral shard rows are filtered out
    first = pre_units["units"][0]
    assert first["unit_type"] == 48
    assert first["alliance"] == 1
    assert first["screen_x"] == 3 and first["screen_y"] == 4
    assert first["is_selected"] == 1
    assert first["health"] == 45
    assert first["health_ratio_255"] == 255
    assert first["order_length"] == 1 and first["order_id_0"] == 7
    assert first["order_id_1"] == 8
    # Continuous radius stays float; it is not truncated to int by us.
    assert first["radius"] == 4.5
    assert isinstance(first["radius"], float)
    assert first["tag"] == 0
    assert "raw_units" in pre_units["tag_note"]
    assert "not a stable unit identity" in pre_units["tag_note"]
    assert payload["post"]["feature_units"]["count"] == 2

    panel = payload["pre"]["select_panel"]
    assert panel["single_select"]["available"] is True
    assert panel["single_select"]["count"] == 1
    selected = panel["single_select"]["units"][0]
    assert selected["unit_type"] == 48
    assert selected["player_relative"] == 1
    assert selected["health"] == 45
    assert selected["build_progress_100"] == 100
    # The panel carries no coordinates/tags, so rows cannot be matched.
    assert "cannot be matched" in panel["note"]
    assert panel["multi_select"] == {
        "available": False,
        "count": None,
        "units": None,
    }

    post_panel = payload["post"]["select_panel"]
    assert post_panel["multi_select"]["count"] == 2
    assert post_panel["multi_select"]["units"][1]["health"] == 44
    assert post_panel["single_select"]["available"] is False

    feedback = payload["engine_feedback"]
    assert feedback["last_actions"]["available"] is True
    assert feedback["last_actions"]["value"] == [POINT_FN_ID]
    assert feedback["action_result"]["available"] is True
    assert feedback["action_result"]["value"] == [0]
    assert feedback["action_errors"]["available"] is False
    assert feedback["action_errors"]["value"] is None

    # Existing trajectory schema stays intact.
    assert "marines_raster_estimate" in payload["pre"]
    assert "neutral_shards_raster_estimate" in payload["post"]
    # The payload is self-contained strict JSON (non-finite floats -> null).
    json.loads(json.dumps(payload, allow_nan=False))


def test_active_rasters_reconstruct_the_parser_inputs():
    _, payload = run_step()

    pre_inputs = payload["pre"]["observation_inputs"]
    screen = pre_inputs["screen_active_layers"]
    assert screen["channels"] == ["player_relative", "selected"]
    assert screen["shape"] == [8, 8, 2]
    assert screen["data"][1][1] == [1, 1]
    assert screen["data"][6][6] == [3, 0]
    # Inactive canonical layers are omitted, not copied as zeros.
    assert screen["shape"][-1] == 2

    minimap = pre_inputs["minimap_active_layers"]
    assert minimap["channels"] == ["selected"]
    assert minimap["shape"] == [8, 8, 1]
    assert minimap["data"][4][4] == [1]

    assert pre_inputs["screen_layer_flags"] == {
        "names": ["visibility_map", "player_relative", "unit_type", "selected"],
        "values": [0, 1, 0, 1],
    }
    assert pre_inputs["minimap_layer_flags"] == {
        "names": [
            "visibility_map",
            "camera",
            "player_relative",
            "selected",
            "buildable",
        ],
        "values": [0, 0, 0, 1, 0],
    }
    assert pre_inputs["available_fn_ids"] == [NO_OP_ID, POINT_FN_ID]
    # Non-finite vector entries are explicit nulls, never NaN in JSON.
    assert pre_inputs["vector"] == [1.0, None, None]
    assert pre_inputs["vector_mask"] == [1.0, 1.0, 0.0]

    post_screen = payload["post"]["observation_inputs"]["screen_active_layers"]
    assert post_screen["data"][2][2] == [1, 1]
    assert payload["post"]["observation_inputs"]["minimap_active_layers"]["data"][5][5] == [1]


def test_nonfinite_raster_cells_are_flagged_not_silently_zeroed():
    before_screen, after_screen = _screens()
    before_screen[0, 0, 1] = np.nan
    before_screen[7, 7, 3] = np.inf
    before_minimap, after_minimap = _minimaps()
    parser = TwoPhaseParser(
        before_screen,
        after_screen,
        before_minimap=before_minimap,
        after_minimap=after_minimap,
    )
    env = make_env(
        parser,
        raw_before=_raw_observation(_feature_units(_unit_rows())),
        raw_after=SimpleNamespace(),
    )
    _, _, _, _, info = env.step([POINT_FN_ID, 10])
    payload = info["collect_mineral_shards/trajectory"]
    screen = payload["pre"]["observation_inputs"]["screen_active_layers"]

    assert screen["nonfinite_count"] == 2
    assert screen["finite_fraction"] == pytest.approx(1.0 - 2.0 / (8 * 8 * 2))
    assert [0, 0, "player_relative"] in screen["nonfinite_coords"]
    assert [7, 7, "selected"] in screen["nonfinite_coords"]
    assert screen["nonfinite_coords_truncated"] is False
    # The int encoding still uses 0, but the flag above makes it visible.
    assert screen["data"][0][0] == [0, 0]
    assert screen["data"][7][7] == [0, 0]
    # Strict JSON still holds: non-finite values never leak into the payload.
    json.loads(json.dumps(payload, allow_nan=False))

    clean = payload["post"]["observation_inputs"]["screen_active_layers"]
    assert clean["nonfinite_count"] == 0
    assert clean["finite_fraction"] == 1.0
    assert clean["nonfinite_coords"] is None


def test_missing_observation_fields_are_explicit_nulls():
    parser = TwoPhaseParser(
        *_screens(), flags=False, vector=False, include_active_config=False
    )
    env = make_env(
        parser,
        raw_before=SimpleNamespace(),
        raw_after=SimpleNamespace(),
    )
    _, _, _, _, info = env.step([POINT_FN_ID, 10])
    payload = info["collect_mineral_shards/trajectory"]

    units = payload["pre"]["feature_units"]
    assert units["available"] is False
    assert units["units"] is None
    assert units["count"] is None
    assert "no feature_units" in units["reason"]

    panel = payload["pre"]["select_panel"]
    assert panel["single_select"] == {
        "available": False,
        "count": None,
        "units": None,
    }
    assert panel["multi_select"]["available"] is False

    feedback = payload["engine_feedback"]
    assert feedback["last_actions"]["available"] is False
    assert feedback["action_result"]["available"] is False
    assert feedback["action_errors"]["available"] is False
    assert feedback["last_actions"]["value"] is None

    inputs = payload["pre"]["observation_inputs"]
    assert inputs["vector"] is None
    assert inputs["screen_layer_flags"] is None
    # Without flags the parser config's all-layers default is recorded.
    screen = inputs["screen_active_layers"]
    assert screen["available"] is True
    assert screen["channels"] == [
        "visibility_map",
        "player_relative",
        "unit_type",
        "selected",
    ]
    assert screen["shape"] == [8, 8, 4]
    # No minimap was produced by this parser.
    assert inputs["minimap_active_layers"]["available"] is False
    assert inputs["minimap_active_layers"]["data"] is None
    json.loads(json.dumps(payload, allow_nan=False))


def test_empty_action_result_is_available_but_not_a_confirmation():
    before_screen, after_screen = _screens()
    parser = TwoPhaseParser(before_screen, after_screen)
    raw_after = _raw_observation(
        _feature_units(_unit_rows()),
        last_actions=[],
        action_result=[],
    )
    env = make_env(
        parser,
        raw_before=SimpleNamespace(),
        raw_after=raw_after,
    )
    _, _, _, _, info = env.step([POINT_FN_ID, 10])
    feedback = info["collect_mineral_shards/trajectory"]["engine_feedback"]

    assert feedback["last_actions"] == {
        "available": True,
        "value": [],
        "source": "PySC2 transform_obs: obs.actions (function ids)",
        "note": "Observed by the engine, not an independent execution acknowledgement.",
    }
    assert feedback["action_result"]["available"] is True
    assert feedback["action_result"]["value"] == []
    assert "not an engine success confirmation" in feedback["action_result"]["note"]


def test_nonzero_rank_keeps_diagnostics_but_skips_payload():
    before_screen, after_screen = _screens()
    env = make_env(
        TwoPhaseParser(before_screen, after_screen),
        raw_before=SimpleNamespace(),
        raw_after=SimpleNamespace(),
    )
    env._vec_env_rank = 1
    _, _, _, _, info = env.step([POINT_FN_ID, 10])

    assert "collect_mineral_shards/trajectory" not in info
    assert "collect_mineral_shards/requested_fn_id" in info
