"""ObsParser registry/config tests (no SC2 launch)."""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pysc2.lib import features

from src.obs_parser import (
    AVAILABLE_MAPS,
    CollectMineralShardsConfig,
    ObsParser,
)


def test_collect_mineral_shards_is_registered():
    assert "CollectMineralShards" in AVAILABLE_MAPS
    assert AVAILABLE_MAPS["CollectMineralShards"] is CollectMineralShardsConfig


def test_collect_mineral_shards_config_uses_minimal_layers():
    parser = ObsParser("CollectMineralShards")

    assert parser.map_name == "CollectMineralShards"
    assert parser.config.screen_active_layers == {"player_relative", "selected"}
    assert parser.config.minimap_active_layers == {"player_relative", "selected"}

    spec = parser.get_output_spec()
    assert spec["screen"]["shape"] == (32, 32, 7)
    assert spec["minimap"]["shape"] == (32, 32, 6)


def test_collect_mineral_shards_parse_emits_layer_flags():
    """Synthetic observation only: exercises config gating without SC2."""
    parser = ObsParser("CollectMineralShards")
    obs = {
        "player": np.zeros(11, dtype=np.int32),
        "feature_screen": np.zeros(
            (len(features.SCREEN_FEATURES), 64, 64), dtype=np.int32
        ),
        "feature_minimap": np.zeros(
            (len(features.MINIMAP_FEATURES), 64, 64), dtype=np.int32
        ),
        "available_actions": np.array([0], dtype=np.int32),
    }

    parsed = parser.parse(obs)

    assert parsed["screen"].shape == (32, 32, 7)
    assert parsed["minimap"].shape == (32, 32, 6)
    assert parsed["screen_layer_flags"].tolist() == [0, 1, 0, 1, 0, 0, 0]
    assert parsed["minimap_layer_flags"].tolist() == [0, 1, 1, 0, 0, 0]


def test_unknown_map_still_rejected():
    with pytest.raises(ValueError, match="not configured"):
        ObsParser("NotAMap")
