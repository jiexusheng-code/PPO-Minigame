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
    BuildMarinesConfig,
    CollectMineralShardsConfig,
    CollectMineralsAndGasConfig,
    DefeatRoachesConfig,
    DefeatZerglingsAndBanelingsConfig,
    FindAndDefeatZerglingsConfig,
    MoveToBeaconConfig,
    ObsParser,
)


CANONICAL_SCREEN_LAYERS = [
    "visibility_map",
    "player_relative",
    "unit_type",
    "selected",
    "unit_hit_points_ratio",
    "build_progress",
    "buildable",
    "active",
]

CANONICAL_MINIMAP_LAYERS = [
    "visibility_map",
    "camera",
    "player_relative",
    "selected",
    "buildable",
]

# map -> (screen_layer_flags, minimap_layer_flags, vector_mask)
EXPECTED_CONFIGS = {
    "MoveToBeacon": (
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ),
    "CollectMineralShards": (
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ),
    "DefeatRoaches": (
        [0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ),
    "FindAndDefeatZerglings": (
        [1, 1, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ),
    "DefeatZerglingsAndBanelings": (
        [0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ),
    "CollectMineralsAndGas": (
        [0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 0],
    ),
    "BuildMarines": (
        [0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 1, 1, 1],
    ),
}


def _obs():
    """Synthetic observation; exercises config gating without SC2."""
    return {
        "player": np.zeros(11, dtype=np.int32),
        "feature_screen": np.zeros(
            (len(features.SCREEN_FEATURES), 64, 64), dtype=np.int32
        ),
        "feature_minimap": np.zeros(
            (len(features.MINIMAP_FEATURES), 64, 64), dtype=np.int32
        ),
        "available_actions": np.array([0], dtype=np.int32),
    }


def test_all_maps_registered():
    expected_classes = {
        "MoveToBeacon": MoveToBeaconConfig,
        "CollectMineralShards": CollectMineralShardsConfig,
        "DefeatRoaches": DefeatRoachesConfig,
        "FindAndDefeatZerglings": FindAndDefeatZerglingsConfig,
        "DefeatZerglingsAndBanelings": DefeatZerglingsAndBanelingsConfig,
        "CollectMineralsAndGas": CollectMineralsAndGasConfig,
        "BuildMarines": BuildMarinesConfig,
    }
    assert set(AVAILABLE_MAPS) == set(expected_classes)
    for map_name, cls in expected_classes.items():
        assert AVAILABLE_MAPS[map_name] is cls


@pytest.mark.parametrize("map_name", sorted(EXPECTED_CONFIGS))
def test_canonical_layer_order_and_shapes(map_name):
    parser = ObsParser(map_name)

    assert [name for _, name in parser.config.screen_layers] == CANONICAL_SCREEN_LAYERS
    assert [name for _, name in parser.config.minimap_layers] == CANONICAL_MINIMAP_LAYERS

    spec = parser.get_output_spec()
    assert spec["screen"]["shape"] == (32, 32, 8)
    assert spec["minimap"]["shape"] == (32, 32, 5)


@pytest.mark.parametrize("map_name", sorted(EXPECTED_CONFIGS))
def test_parse_emits_expected_flags_and_vector_mask(map_name):
    parser = ObsParser(map_name)
    expected_screen, expected_minimap, expected_vector = EXPECTED_CONFIGS[map_name]

    parsed = parser.parse(_obs())

    assert parsed["screen"].shape == (32, 32, 8)
    assert parsed["minimap"].shape == (32, 32, 5)
    assert parsed["screen_layer_flags"].tolist() == expected_screen
    assert parsed["minimap_layer_flags"].tolist() == expected_minimap
    assert parsed["vector_mask"].tolist() == expected_vector


def test_unknown_map_still_rejected():
    with pytest.raises(ValueError, match="not configured"):
        ObsParser("NotAMap")
