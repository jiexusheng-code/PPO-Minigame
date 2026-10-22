"""CollectMineralShards trajectory payload and streaming persistence tests.

The wrapper tests use fake SC2 objects. The SB3 integration test uses a small
synthetic Gymnasium environment; it exercises the real Monitor/DummyVecEnv/
PPO/callback wiring without launching StarCraft II.
"""
import gzip
import json
import logging
import os
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src.callbacks import tb_dual_writer
from src.callbacks.tb_dual_writer import (
    TRAJECTORY_INFO_KEY,
    TRAJECTORY_SAMPLE_INTERVAL_ENV_STEPS,
    TBDualWriterCallback,
)
from src.envs.pysc2_gym_wrapper import PySC2GymEnv

NO_OP_ID = 0
POINT_FN_ID = 1


# ---------------------------------------------------------------------------
# Wrapper payload
# ---------------------------------------------------------------------------


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


class SequencedParser:
    """Return a before screen on the first parse and an after screen next."""

    def __init__(self, before, after):
        self.before = before
        self.after = after
        self.calls = 0
        self.config = SimpleNamespace(screen_layers=[
            (0, "visibility_map"),
            (1, "player_relative"),
            (2, "unit_type"),
            (3, "selected"),
            (4, "unit_hit_points_ratio"),
            (5, "build_progress"),
            (6, "buildable"),
        ])

    def parse(self, observation):
        screen = self.before if self.calls == 0 else self.after
        self.calls += 1
        mask = np.zeros(2, dtype=np.float32)
        mask[[NO_OP_ID, POINT_FN_ID]] = 1.0
        return {"available_actions": mask, "screen": screen}


class FakeSC2Env:
    def __init__(self):
        self.calls = []

    def step(self, actions):
        self.calls.append(actions[0])
        return [SimpleNamespace(reward=2.5, last=lambda: True, observation=object())]


def make_wrapper_env(before, after):
    env = PySC2GymEnv.__new__(PySC2GymEnv)
    env.logger = logging.getLogger("test_trajectory_wrapper")
    env.actions = SimpleNamespace(FUNCTIONS=FakeFunctions([
        FakeFunction(NO_OP_ID, "no_op"),
        FakeFunction(POINT_FN_ID, "select_point", [
            SimpleNamespace(name="screen", sizes=[8, 8]),
        ]),
    ]))
    env._env = FakeSC2Env()
    env._parser = SequencedParser(before, after)
    env.observation_space = {}
    env._fn_param_map = {NO_OP_ID: [], POINT_FN_ID: [0]}
    env._action_spec_types = SimpleNamespace(
        _fields=("screen",), screen=SimpleNamespace(sizes=[8, 8])
    )
    env.timestep = SimpleNamespace(observation=object(), reward=0.0, last=lambda: False)
    env._step_count = 0
    env.map_name = "CollectMineralShards"
    env._episode_id = 4
    env._episode_step = 2
    env._episode_seed_source = "constructor"
    env._sc2_random_seed = 42
    return env


def _synthetic_screens():
    before = np.zeros((8, 8, 7), dtype=np.float32)
    # Marine A: 2x2 block, selected.
    before[1:3, 1:3, 1] = 1
    before[1:3, 1:3, 3] = 1
    # Marine B: single pixel, not selected.
    before[5, 5, 1] = 1
    # One neutral shard: 2x2 block.
    before[6:8, 6:8, 1] = 3

    after = np.zeros((8, 8, 7), dtype=np.float32)
    # Marine A moved down-right and stays selected.
    after[2:4, 2:4, 1] = 1
    after[2:4, 2:4, 3] = 1
    after[5, 5, 1] = 1
    return before, after


def test_wrapper_payload_contains_stages_coordinates_and_pre_post_rasters():
    before, after = _synthetic_screens()
    env = make_wrapper_env(before, after)

    assert env._canonical_screen_channels() == (1, 3)

    _, reward, terminated, truncated, info = env.step([POINT_FN_ID, 10])
    payload = info[TRAJECTORY_INFO_KEY]

    assert payload["map"] == "CollectMineralShards"
    assert payload["episode_id"] == 4
    assert payload["episode_step"] == 2
    assert payload["seed"] == 42
    assert payload["seed_source"] == "constructor"
    assert payload["reward"] == 2.5
    assert payload["terminated"] is True
    assert payload["done"] is True
    assert payload["truncated"] is False
    assert payload["available_fn_ids"] == [NO_OP_ID, POINT_FN_ID]

    # requested keeps the raw flat index; sanitized/executed are [x, y].
    assert payload["requested"]["fn_id"] == POINT_FN_ID
    assert payload["requested"]["fn_name"] == "select_point"
    assert payload["requested"]["args"] == [["screen", 10]]
    assert payload["requested"]["screen_xy"] == [2, 1]
    assert payload["sanitized"]["args"] == [["screen", [2, 1]]]
    assert payload["sanitized"]["screen_xy"] == [2, 1]
    assert payload["executed"]["args"] == [["screen", [2, 1]]]
    assert payload["executed"]["screen_xy"] == [2, 1]
    assert payload["requested"]["queued_flag"] is None
    assert payload["executed"]["queued_flag"] is None
    assert payload["fallback"] is False
    assert payload["fallback_reason_code"] == "none"

    assert payload["neutral_component_count_raster_estimate"] == {
        "before": 1,
        "after": 0,
    }
    assert payload["target_distance_cells_raster_estimate"][
        "requested_to_neutral"
    ] == pytest.approx(np.hypot(5.5, 4.5))
    assert payload["target_distance_cells_raster_estimate"][
        "executed_to_neutral"
    ] == pytest.approx(np.hypot(5.5, 4.5))
    assert payload["target_distance_cells_raster_estimate"][
        "marine_to_neutral_before"
    ] > 0

    pre_marines = payload["pre"]["marines_raster_estimate"]
    assert len(pre_marines) == 2
    assert pre_marines[0]["centroid_y_raster_estimate"] == pytest.approx(1.5)
    assert pre_marines[0]["centroid_x_raster_estimate"] == pytest.approx(1.5)
    assert pre_marines[0]["pixel_count"] == 4
    assert pre_marines[0]["selected_raster_estimate"] is True
    assert pre_marines[1]["centroid_y_raster_estimate"] == pytest.approx(5.0)
    assert pre_marines[1]["centroid_x_raster_estimate"] == pytest.approx(5.0)
    assert pre_marines[1]["selected_raster_estimate"] is False

    assert payload["pre"]["neutral_shards_raster_estimate"] == [{
        "centroid_y_raster_estimate": pytest.approx(6.5),
        "centroid_x_raster_estimate": pytest.approx(6.5),
        "pixel_count": 4,
    }]

    post_marines = payload["post"]["marines_raster_estimate"]
    assert post_marines[0]["centroid_y_raster_estimate"] == pytest.approx(2.5)
    assert post_marines[0]["centroid_x_raster_estimate"] == pytest.approx(2.5)
    assert post_marines[0]["selected_raster_estimate"] is True
    assert payload["post"]["neutral_shards_raster_estimate"] == []

    # The wrapper advances its own episode step counter, and the trajectory
    # payload is a dict so TB aggregation must ignore it.
    assert env._episode_step == 3
    assert isinstance(info[TRAJECTORY_INFO_KEY], dict)


def test_wrapper_does_not_emit_trajectory_for_other_maps():
    before, after = _synthetic_screens()
    env = make_wrapper_env(before, after)
    env.map_name = "MoveToBeacon"
    _, _, _, _, info = env.step([NO_OP_ID])
    assert TRAJECTORY_INFO_KEY not in info


def test_nonzero_vec_env_rank_skips_payload_but_keeps_tb_diagnostics():
    before, after = _synthetic_screens()
    env = make_wrapper_env(before, after)
    env._vec_env_rank = 1
    _, _, _, _, info = env.step([POINT_FN_ID, 10])
    assert TRAJECTORY_INFO_KEY not in info
    assert "collect_mineral_shards/requested_fn_id" in info


def test_wrapper_payload_roundtrips_through_callback(tmp_path):
    before, after = _synthetic_screens()
    env = make_wrapper_env(before, after)
    env._episode_step = 0
    _, _, _, _, info = env.step([POINT_FN_ID, 10])
    payload = dict(info[TRAJECTORY_INFO_KEY])
    # The fake SC2 timestep reports last()==True; this round-trip test only
    # needs a non-terminal first step followed by a terminal one.
    payload["terminated"] = False
    payload["done"] = False
    assert payload["episode_step"] == 0

    callback = make_callback(tmp_path)
    feed(callback, [payload], 2)
    terminal = dict(payload)
    terminal["episode_step"] = 1
    terminal["terminated"] = True
    feed(callback, [terminal], 4)

    records = read_trajectory(trajectory_files(tmp_path)[0])
    steps = [record for record in records if record["type"] == "step"]
    assert len(steps) == 2
    assert steps[0]["pre"]["marines_raster_estimate"][0][
        "selected_raster_estimate"
    ] is True
    assert steps[0]["requested"]["screen_xy"] == [2, 1]
    assert records[-1]["complete"] is True


# ---------------------------------------------------------------------------
# Callback sampling / persistence
# ---------------------------------------------------------------------------


class RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[(tag, step)] = value

    def close(self):
        pass


def make_payload(
    episode_id,
    episode_step,
    reward=0.5,
    terminated=False,
    source_env=0,
    map_name="CollectMineralShards",
):
    stage = {
        "fn_id": 331,
        "fn_name": "Move_screen",
        "args": [["queued", [1]], ["screen", [3, 4]]],
        "queued_flag": 1,
        "screen_xy": [4, 3],
    }
    return {
        "map": map_name,
        "episode_id": episode_id,
        "episode_step": episode_step,
        "seed": 7,
        "seed_source": "constructor",
        "reward": reward,
        "terminated": terminated,
        "truncated": False,
        "available_fn_ids": [0, 331],
        "requested": dict(stage),
        "sanitized": dict(stage),
        "executed": dict(stage),
        "fallback": False,
        "fallback_reason": "",
        "fallback_reason_code": "none",
        "sanitization_failed": False,
        "args_bounds_violation": False,
        "neutral_component_count_raster_estimate": {"before": 2, "after": 1},
        "target_distance_cells_raster_estimate": {
            "requested_to_neutral": 1.5,
            "executed_to_neutral": 1.5,
            "marine_to_neutral_before": 2.0,
        },
        "pre": {
            "marines_raster_estimate": [{
                "centroid_y_raster_estimate": 1.0,
                "centroid_x_raster_estimate": 2.0,
                "pixel_count": 4,
                "selected_raster_estimate": True,
            }],
            "neutral_shards_raster_estimate": [{
                "centroid_y_raster_estimate": 6.0,
                "centroid_x_raster_estimate": 6.0,
                "pixel_count": 4,
            }],
        },
        "post": {
            "marines_raster_estimate": [{
                "centroid_y_raster_estimate": 2.0,
                "centroid_x_raster_estimate": 3.0,
                "pixel_count": 4,
                "selected_raster_estimate": True,
            }],
            "neutral_shards_raster_estimate": [],
        },
        "source_env": source_env,
    }


def make_callback(tmp_path):
    callback = TBDualWriterCallback(str(tmp_path))
    callback.opt_writer = RecordingWriter()
    return callback


def feed(callback, payloads, global_step):
    callback.model = SimpleNamespace(
        num_timesteps=global_step, policy=SimpleNamespace()
    )
    callback.locals = {
        "infos": [
            ({TRAJECTORY_INFO_KEY: payload} if payload is not None else {})
            for payload in payloads
        ]
    }
    assert callback._on_step() is True


def read_trajectory(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def trajectory_files(tmp_path):
    directory = tmp_path / "trajectories"
    if not directory.exists():
        return []
    return sorted(directory.glob("*.jsonl.gz"))


def test_first_complete_episode_is_streamed_for_env0_only(tmp_path):
    callback = make_callback(tmp_path)

    for step in range(3):
        global_step = (step + 1) * 2  # n_envs == 2
        feed(callback, [
            make_payload(1, step, reward=0.5, terminated=(step == 2), source_env=0),
            make_payload(9, step, reward=99.0, terminated=(step == 2), source_env=1),
        ], global_step)

    files = trajectory_files(tmp_path)
    assert len(files) == 1
    records = read_trajectory(files[0])

    header = records[0]
    assert header["type"] == "header"
    assert header["env_index"] == 0
    assert header["episode_id"] == 1
    assert header["start_global_num_timesteps"] == 2
    assert header["map"] == "CollectMineralShards"
    assert header["seed"] == 7
    assert header["seed_source"] == "constructor"
    assert "not a real order-queue length" in header["queued_note"]

    steps = [record for record in records if record["type"] == "step"]
    assert [record["episode_step"] for record in steps] == [0, 1, 2]
    assert [record["global_num_timesteps"] for record in steps] == [2, 4, 6]
    assert all(record["source_env"] == 0 for record in steps)
    assert all(record["episode_id"] == 1 for record in steps)
    assert steps[0]["requested"]["queued_flag"] == 1
    assert steps[0]["requested"]["screen_xy"] == [4, 3]

    end = records[-1]
    assert end["type"] == "episode_end"
    assert end["complete"] is True
    assert end["steps"] == 3
    assert end["total_reward"] == pytest.approx(1.5)
    assert end["terminated"] is True
    assert end["truncated"] is False


def test_next_sample_waits_for_50000_global_steps_then_episode_boundary(tmp_path):
    callback = make_callback(tmp_path)

    # First complete episode starts at global step 2 and ends at 4.
    feed(callback, [make_payload(1, 0)], 2)
    feed(callback, [make_payload(1, 1, terminated=True)], 4)
    assert len(trajectory_files(tmp_path)) == 1

    # An episode starting before the 50000-step interval is not sampled.
    feed(callback, [make_payload(2, 0)], 100)
    feed(callback, [make_payload(2, 1, terminated=True)], 102)
    assert len(trajectory_files(tmp_path)) == 1

    # 50001 is still one env step short of the 2 + 50000 boundary.
    feed(callback, [make_payload(3, 0)], 50001)
    assert len(trajectory_files(tmp_path)) == 1

    # The next episode boundary at/after the threshold starts a new file.
    feed(callback, [make_payload(4, 0)], 50002)
    feed(callback, [make_payload(4, 1, terminated=True)], 50004)
    files = trajectory_files(tmp_path)
    assert len(files) == 2
    headers = [read_trajectory(path)[0] for path in files]
    assert sorted(header["start_global_num_timesteps"] for header in headers) == [
        2,
        50002,
    ]


def test_multi_env_rhythm_uses_global_num_timesteps(tmp_path, monkeypatch):
    monkeypatch.setattr(tb_dual_writer, "TRAJECTORY_SAMPLE_INTERVAL_ENV_STEPS", 8)
    callback = make_callback(tmp_path)

    # n_envs == 4: each VecEnv step advances the global counter by 4.
    feed(callback, [make_payload(1, 0), None, None, None], 4)
    feed(callback, [make_payload(1, 1, terminated=True), None, None, None], 8)
    assert len(trajectory_files(tmp_path)) == 1

    # start 4 + interval 8 = 12; the next episode begins at global 12.
    feed(callback, [make_payload(2, 0), None, None, None], 12)
    feed(callback, [make_payload(2, 1, terminated=True), None, None, None], 16)
    assert len(trajectory_files(tmp_path)) == 2


def test_two_sampled_episodes_have_continuous_local_steps(tmp_path, monkeypatch):
    monkeypatch.setattr(tb_dual_writer, "TRAJECTORY_SAMPLE_INTERVAL_ENV_STEPS", 4)
    callback = make_callback(tmp_path)

    feed(callback, [make_payload(1, 0, reward=0.1)], 2)
    feed(callback, [make_payload(1, 1, reward=0.2)], 4)
    feed(callback, [make_payload(1, 2, reward=0.3, terminated=True)], 6)

    feed(callback, [make_payload(2, 0, reward=1.0)], 8)
    feed(callback, [make_payload(2, 1, reward=2.0)], 10)
    feed(callback, [make_payload(2, 2, reward=3.0, terminated=True)], 12)

    files = trajectory_files(tmp_path)
    assert len(files) == 2
    for path in files:
        records = read_trajectory(path)
        steps = [record for record in records if record["type"] == "step"]
        end = records[-1]
        assert [record["episode_step"] for record in steps] == [0, 1, 2]
        assert end["type"] == "episode_end"
        assert end["complete"] is True
        assert end["steps"] == 3
        assert end["total_reward"] == pytest.approx(
            sum(record["reward"] for record in steps)
        )


def test_callback_waits_for_next_episode_boundary_when_attached_mid_episode(tmp_path):
    callback = make_callback(tmp_path)

    # First observed step is already mid-episode: do not slice into it.
    feed(callback, [make_payload(1, 5)], 60000)
    feed(callback, [make_payload(1, 6, terminated=True)], 60002)
    assert trajectory_files(tmp_path) == []

    # The next real reset boundary is the first complete episode.
    feed(callback, [make_payload(2, 0)], 60004)
    feed(callback, [make_payload(2, 1, terminated=True)], 60006)
    files = trajectory_files(tmp_path)
    assert len(files) == 1
    assert read_trajectory(files[0])[0]["episode_id"] == 2


def test_lost_episode_boundary_closes_incomplete_and_respects_interval(tmp_path):
    callback = make_callback(tmp_path)
    feed(callback, [make_payload(1, 0)], 2)
    feed(callback, [make_payload(1, 1)], 4)

    # A different episode id appears without a terminal step for episode 1.
    feed(callback, [make_payload(2, 0)], 6)

    files = trajectory_files(tmp_path)
    assert len(files) == 1
    first = read_trajectory(files[0])
    assert first[-1]["type"] == "episode_end"
    assert first[-1]["complete"] is False
    assert first[-1]["incomplete_reason"] == "episode_boundary_lost"
    assert first[-1]["steps"] == 2

    # The incomplete sample still counts as the previous sampling start, so
    # the next episode is only sampled after the interval.
    feed(callback, [make_payload(3, 0)], 50002)
    feed(callback, [make_payload(3, 1, terminated=True)], 50004)
    assert len(trajectory_files(tmp_path)) == 2


def test_training_end_closes_partial_episode_as_incomplete(tmp_path):
    callback = make_callback(tmp_path)
    feed(callback, [make_payload(1, 0)], 2)
    feed(callback, [make_payload(1, 1)], 4)

    callback._on_training_end()

    files = trajectory_files(tmp_path)
    assert len(files) == 1
    records = read_trajectory(files[0])
    end = records[-1]
    assert end["type"] == "episode_end"
    assert end["complete"] is False
    assert end["incomplete_reason"] == "training_end"
    assert end["steps"] == 2


def test_public_interrupt_close_marks_incomplete(tmp_path):
    callback = make_callback(tmp_path)
    feed(callback, [make_payload(1, 0)], 2)

    callback.close_incomplete_trajectory("training_interrupted")

    end = read_trajectory(trajectory_files(tmp_path)[0])[-1]
    assert end["type"] == "episode_end"
    assert end["complete"] is False
    assert end["incomplete_reason"] == "training_interrupted"
    assert end["steps"] == 1


def test_nonfinite_values_become_null_and_json_is_strict(tmp_path):
    callback = make_callback(tmp_path)
    payload = make_payload(1, 0, terminated=True)
    payload["target_distance_cells_raster_estimate"].update({
        "nan": float("nan"),
        "inf": float("inf"),
        "ninf": float("-inf"),
        "np_nan": np.float32("nan"),
        "np_int": np.int64(3),
        "np_bool": np.bool_(True),
    })

    feed(callback, [payload], 2)

    path = trajectory_files(tmp_path)[0]
    raw = gzip.open(path, "rb").read().decode("utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert "-Infinity" not in raw
    records = read_trajectory(path)
    distances = records[1]["target_distance_cells_raster_estimate"]
    assert distances["nan"] is None
    assert distances["inf"] is None
    assert distances["ninf"] is None
    assert distances["np_nan"] is None
    assert distances["np_int"] == 3
    assert distances["np_bool"] is True


def test_existing_trajectory_file_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tb_dual_writer, "time", SimpleNamespace(time_ns=lambda: 123456789)
    )
    directory = tmp_path / "trajectories"
    directory.mkdir(parents=True)
    existing = directory / "collect_mineral_shards_env0_ep000001_g2_123456789.jsonl.gz"
    existing.write_bytes(b"sentinel")

    callback = make_callback(tmp_path)
    feed(callback, [make_payload(1, 0, terminated=True)], 2)

    assert existing.read_bytes() == b"sentinel"
    files = trajectory_files(tmp_path)
    assert len(files) == 2
    assert existing in files
    new_files = [path for path in files if path != existing]
    assert len(new_files) == 1
    assert new_files[0].name.endswith("_1.jsonl.gz")
    assert read_trajectory(new_files[0])[0]["type"] == "header"


def test_non_map_infos_create_no_trajectory_files(tmp_path):
    callback = make_callback(tmp_path)
    callback.model = SimpleNamespace(num_timesteps=2, policy=SimpleNamespace())
    callback.locals = {"infos": [{"arg_clipped": True}, {"arg_clipped": False}]}

    assert callback._on_step() is True
    assert not (tmp_path / "trajectories").exists()


def test_payload_for_other_map_is_ignored(tmp_path):
    callback = make_callback(tmp_path)
    feed(callback, [make_payload(1, 0, terminated=True, map_name="MoveToBeacon")], 2)
    assert not (tmp_path / "trajectories").exists()


def test_tb_aggregation_ignores_trajectory_payload(tmp_path):
    callback = make_callback(tmp_path)
    prefix = "collect_mineral_shards/"
    info = {
        prefix + "requested_fn_id": 331,
        prefix + "requested_fn_name": "Move_screen",
        prefix + "sanitized_fn_id": 331,
        prefix + "sanitized_fn_name": "Move_screen",
        prefix + "executed_fn_id": 331,
        prefix + "executed_fn_name": "Move_screen",
        prefix + "fallback": False,
        prefix + "fallback_reason_code": "none",
        TRAJECTORY_INFO_KEY: make_payload(1, 0, terminated=True),
    }
    callback.model = SimpleNamespace(num_timesteps=8, policy=SimpleNamespace())
    callback.locals = {"infos": [info]}

    assert callback._on_step() is True
    callback._on_rollout_end()

    writer = callback.opt_writer
    assert writer.scalars[(prefix + "fallback_mean", 8)] == 0.0
    assert writer.scalars[(prefix + "requested_function_fraction/331_Move_screen", 8)] == 1.0
    assert not any("trajectory" in tag for tag, _ in writer.scalars)
    assert all("trajectory" not in key for key in callback._collect_diag_sum)


# ---------------------------------------------------------------------------
# Synthetic SB3 integration (no SC2)
# ---------------------------------------------------------------------------


class SyntheticCollectEnv(gym.Env):
    """Minimal environment emitting the wrapper's trajectory payload."""

    metadata = {"render_modes": []}

    def __init__(self, env_index, episode_len=4):
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.env_index = env_index
        self.episode_len = episode_len
        self.episode_id = 0
        self.episode_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_id += 1
        self.episode_step = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        step = self.episode_step
        reward = 1.0
        terminated = step == self.episode_len - 1
        self.episode_step += 1
        payload = make_payload(
            self.episode_id,
            step,
            reward=reward,
            terminated=terminated,
            source_env=self.env_index,
        )
        payload["pre"]["obs_marker"] = (
            "episode_start" if step == 0 else f"after_step_{step - 1}"
        )
        payload["post"]["obs_marker"] = "terminal" if terminated else f"after_step_{step}"
        return (
            np.zeros(2, dtype=np.float32),
            reward,
            terminated,
            False,
            {TRAJECTORY_INFO_KEY: payload},
        )


def test_synthetic_sb3_ppo_dummy_vec_env_writes_env0_episode(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    vec_env = DummyVecEnv([
        lambda: Monitor(SyntheticCollectEnv(0)),
        lambda: Monitor(SyntheticCollectEnv(1)),
    ])
    callback = TBDualWriterCallback(str(tmp_path))
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            learning_rate=1e-3,
            device="cpu",
            seed=0,
            verbose=0,
        )
        model.learn(total_timesteps=16, callback=callback)
    finally:
        vec_env.close()

    files = trajectory_files(tmp_path)
    assert len(files) == 1
    records = read_trajectory(files[0])
    header = records[0]
    assert header["type"] == "header"
    assert header["env_index"] == 0
    assert header["episode_id"] == 1
    # The first VecEnv step advances the global counter by n_envs == 2.
    assert header["start_global_num_timesteps"] == 2

    steps = [record for record in records if record["type"] == "step"]
    assert [record["episode_step"] for record in steps] == [0, 1, 2, 3]
    assert [record["global_num_timesteps"] for record in steps] == [2, 4, 6, 8]
    assert all(record["source_env"] == 0 for record in steps)
    assert steps[0]["pre"]["obs_marker"] == "episode_start"
    # The terminal step's post raster must come from the step result, never
    # from the VecEnv auto-reset observation.
    assert steps[-1]["post"]["obs_marker"] == "terminal"
    assert all(
        "reset" not in record["post"]["obs_marker"] for record in steps
    )

    end = records[-1]
    assert end["type"] == "episode_end"
    assert end["complete"] is True
    assert end["steps"] == 4
    assert end["terminated"] is True

    # The existing TensorBoard writer still produced its event file.
    assert list((tmp_path / "extra").glob("events.out.tfevents.*"))


def test_synthetic_sb3_samples_multiple_episodes_when_interval_is_small(
    tmp_path, monkeypatch
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    monkeypatch.setattr(tb_dual_writer, "TRAJECTORY_SAMPLE_INTERVAL_ENV_STEPS", 8)
    vec_env = DummyVecEnv([
        lambda: Monitor(SyntheticCollectEnv(0)),
        lambda: Monitor(SyntheticCollectEnv(1)),
    ])
    callback = TBDualWriterCallback(str(tmp_path))
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            learning_rate=1e-3,
            device="cpu",
            seed=0,
            verbose=0,
        )
        # Two rollouts of 8 VecEnv steps == 32 env steps == 8 episodes per
        # environment; episode starts land on global steps 2, 10, 18, 26.
        model.learn(total_timesteps=24, callback=callback)
    finally:
        vec_env.close()

    files = trajectory_files(tmp_path)
    assert len(files) == 4
    starts = sorted(read_trajectory(path)[0]["start_global_num_timesteps"] for path in files)
    assert starts == [2, 10, 18, 26]
    for path in files:
        records = read_trajectory(path)
        steps = [record for record in records if record["type"] == "step"]
        assert [record["episode_step"] for record in steps] == [0, 1, 2, 3]
        assert records[-1]["complete"] is True
        assert all(record["source_env"] == 0 for record in steps)


# ---------------------------------------------------------------------------
# Close-path I/O failures must be visible without interrupting training
# ---------------------------------------------------------------------------


class FailingStream:
    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.closed = False

    def write(self, data):
        if self.fail_on == "write":
            raise OSError("simulated episode_end write failure")

    def flush(self):
        if self.fail_on == "flush":
            raise OSError("simulated flush failure")

    def close(self):
        self.closed = True
        if self.fail_on == "close":
            raise OSError("simulated close failure")


def install_fake_trajectory_stream(callback, stream, path):
    callback._traj_file = stream
    callback._traj_path = str(path)
    callback._traj_episode_id = 7
    callback._traj_start_global = 100
    callback._traj_steps = 5
    callback._traj_reward_sum = 1.5
    callback._traj_next_allowed_step = 0


def assert_clean_close_state(callback):
    assert callback._traj_file is None
    assert callback._traj_path is None
    assert callback._traj_episode_id is None
    assert callback._traj_steps == 0
    assert callback._traj_reward_sum == 0.0
    assert callback._traj_next_allowed_step == (
        100 + TRAJECTORY_SAMPLE_INTERVAL_ENV_STEPS
    )


def assert_error_log(caplog, path, fragment):
    matches = [
        record
        for record in caplog.records
        if record.levelno == logging.ERROR
        and fragment in record.getMessage()
        and path in record.getMessage()
    ]
    assert matches
    assert all(record.exc_info is not None for record in matches)


def test_close_trajectory_logs_episode_end_write_failure(tmp_path, caplog):
    callback = TBDualWriterCallback(str(tmp_path))
    path = tmp_path / "trajectories" / "ep000007.jsonl.gz"
    stream = FailingStream("write")
    install_fake_trajectory_stream(callback, stream, path)

    with caplog.at_level(logging.ERROR, logger=tb_dual_writer.__name__):
        callback._close_trajectory(
            complete=True, global_step=110, terminated=True, truncated=False
        )

    assert_error_log(caplog, str(path), "结束标记写入失败")
    assert stream.closed is True
    assert_clean_close_state(callback)


def test_close_incomplete_trajectory_logs_flush_failure(tmp_path, caplog):
    callback = TBDualWriterCallback(str(tmp_path))
    path = tmp_path / "trajectories" / "ep000007.jsonl.gz"
    stream = FailingStream("flush")
    install_fake_trajectory_stream(callback, stream, path)

    with caplog.at_level(logging.ERROR, logger=tb_dual_writer.__name__):
        callback.close_incomplete_trajectory("training_interrupted")

    assert_error_log(caplog, str(path), "flush 失败")
    assert stream.closed is True
    assert_clean_close_state(callback)


def test_training_end_logs_close_failure(tmp_path, caplog):
    callback = TBDualWriterCallback(str(tmp_path))
    path = tmp_path / "trajectories" / "ep000007.jsonl.gz"
    stream = FailingStream("close")
    install_fake_trajectory_stream(callback, stream, path)
    callback.opt_writer = None

    with caplog.at_level(logging.ERROR, logger=tb_dual_writer.__name__):
        callback._on_training_end()

    assert_error_log(caplog, str(path), "close 失败")
    assert stream.closed is True
    assert_clean_close_state(callback)
