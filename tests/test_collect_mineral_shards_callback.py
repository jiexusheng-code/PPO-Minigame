"""CollectMineralShards info aggregation without SC2 or a real event writer."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.callbacks.tb_dual_writer import TBDualWriterCallback


class RecordingWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[(tag, step)] = value


def test_diagnostics_aggregate_flags_arguments_function_frequencies_and_reset():
    callback = TBDualWriterCallback("unused")
    callback.model = SimpleNamespace(num_timesteps=8, policy=SimpleNamespace())
    writer = RecordingWriter()
    callback.opt_writer = writer
    prefix = "collect_mineral_shards/"
    callback.locals = {"infos": [
        {"arg_clipped": True, "executed_fn_id": 7},
        {
            prefix + "requested_fn_id": 331,
            prefix + "requested_fn_name": "Move_screen",
            prefix + "sanitized_fn_id": 331,
            prefix + "sanitized_fn_name": "Move_screen",
            prefix + "executed_fn_id": 331,
            prefix + "executed_fn_name": "Move_screen",
            prefix + "fallback": False,
            prefix + "fallback_reason_code": "none",
            prefix + "requested_queued": 1,
            prefix + "requested_screen_flat": 17,
            prefix + "requested_screen_x": 1,
            prefix + "requested_screen_y": 1,
            prefix + "sanitized_screen_flat": 17,
            prefix + "sanitized_screen_x": 1,
            prefix + "sanitized_screen_y": 1,
            prefix + "raster_neutral_component_count_estimate_before": 2,
            prefix + "raster_requested_target_to_neutral_component_distance_estimate_cells": 3.0,
            prefix + "raster_executed_target_to_neutral_component_distance_estimate_cells": 2.0,
            prefix + "requested_args": (("queued", 1), ("screen", 17)),
        },
        {
            prefix + "requested_fn_id": 331,
            prefix + "requested_fn_name": "Move_screen",
            prefix + "sanitized_fn_id": 331,
            prefix + "sanitized_fn_name": "Move_screen",
            prefix + "executed_fn_id": 0,
            prefix + "executed_fn_name": "no_op",
            prefix + "fallback": True,
            prefix + "fallback_reason_code": "step_error",
            prefix + "requested_queued": 0,
            prefix + "raster_neutral_component_count_estimate_before": 4,
            prefix + "raster_requested_target_to_neutral_component_distance_estimate_cells": float("nan"),
        },
        {
            prefix + "requested_fn_id": 12,
            prefix + "requested_fn_name": "Attack_screen",
            prefix + "sanitized_fn_id": -1,
            prefix + "sanitized_fn_name": "invalid",
            prefix + "executed_fn_id": 0,
            prefix + "executed_fn_name": "no_op",
            prefix + "fallback": True,
            prefix + "fallback_reason_code": "sanitize_failed",
            prefix + "requested_queued": 0,
            prefix + "raster_neutral_component_count_estimate_before": 6,
        },
    ]}

    assert callback._on_step() is True
    callback._on_rollout_end()

    assert writer.scalars[(prefix + "fallback_mean", 8)] == 2 / 3
    assert writer.scalars[(prefix + "requested_queued_mean", 8)] == 1 / 3
    assert writer.scalars[(prefix + "raster_neutral_component_count_estimate_before_mean", 8)] == 4
    assert writer.scalars[
        (prefix + "raster_requested_target_to_neutral_component_distance_estimate_cells_mean", 8)
    ] == 3
    assert writer.scalars[
        (prefix + "raster_executed_target_to_neutral_component_distance_estimate_cells_mean", 8)
    ] == 2
    assert writer.scalars[(prefix + "requested_function_fraction/331_Move_screen", 8)] == 2 / 3
    assert writer.scalars[(prefix + "requested_function_fraction/12_Attack_screen", 8)] == 1 / 3
    assert writer.scalars[(prefix + "sanitized_function_fraction/331_Move_screen", 8)] == 2 / 3
    assert writer.scalars[(prefix + "sanitized_function_fraction/-1_invalid", 8)] == 1 / 3
    assert writer.scalars[(prefix + "executed_function_fraction/0_no_op", 8)] == 2 / 3
    assert writer.scalars[(prefix + "executed_function_fraction/331_Move_screen", 8)] == 1 / 3
    assert writer.scalars[(prefix + "fallback_reason_fraction/none", 8)] == 1 / 3
    assert writer.scalars[(prefix + "fallback_reason_fraction/step_error", 8)] == 1 / 3
    assert writer.scalars[(prefix + "fallback_reason_fraction/sanitize_failed", 8)] == 1 / 3
    assert writer.scalars[(prefix + "requested_screen_x_mean", 8)] == 1
    assert writer.scalars[(prefix + "requested_screen_y_mean", 8)] == 1
    # Categorical function ids and flattened spatial indices must never be
    # averaged as numeric metrics.
    assert not any(tag.endswith("_fn_id_mean") for tag, _ in writer.scalars)
    assert not any(tag.endswith("_flat_mean") for tag, _ in writer.scalars)
    assert not any("requested_args" in tag for tag, _ in writer.scalars)
    assert callback._collect_diag_sum == {}
    assert callback._collect_diag_count == {}
    assert callback._collect_fn_counts == {}
    assert callback._collect_reason_counts == {}
    assert callback._collect_diag_steps == 0


def test_unrelated_map_infos_do_not_emit_collect_mineral_shards_metrics():
    callback = TBDualWriterCallback("unused")
    callback.model = SimpleNamespace(num_timesteps=2, policy=SimpleNamespace())
    writer = RecordingWriter()
    callback.opt_writer = writer
    callback.locals = {"infos": [{"arg_clipped": True, "fn_available": False, "executed_fn_id": 0}]}

    assert callback._on_step() is True
    callback._on_rollout_end()

    assert not any(tag.startswith("collect_mineral_shards/") for tag, _ in writer.scalars)
