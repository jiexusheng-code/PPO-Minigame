"""Custom reward functions and registry for PySC2 minigames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional, Tuple

import numpy as np


@dataclass
class RewardState:
    prev_distance: Optional[float] = None
    prev_beacon_center: Optional[Tuple[float, float]] = None


def _coords_for_value(layer: np.ndarray, value: int) -> np.ndarray:
    layer_int = np.rint(layer).astype(np.int32)
    return np.argwhere(layer_int == value)


def _min_distance(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size == 0 or b.size == 0:
        return None
    # a, b are (N, 2) arrays of (y, x)
    dists = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return float(dists.min())


def _center(coords: np.ndarray) -> Optional[Tuple[float, float]]:
    if coords.size == 0:
        return None
    cy, cx = coords.mean(axis=0)
    return float(cy), float(cx)


def reward_move_to_beacon(parsed_obs: Dict[str, Any], state: RewardState, info: Dict[str, Any]) -> float:
    screen = parsed_obs.get("screen")
    if screen is None or screen.ndim != 3 or screen.shape[-1] < 2:
        raise RuntimeError("screen missing or invalid; cannot compute custom reward")

    # screen layers follow MapConfig order; player_relative is index 1
    player_relative = screen[:, :, 1]

    # PySC2 player_relative: 1=self, 3=neutral
    self_coords = _coords_for_value(player_relative, 1)
    beacon_coords = _coords_for_value(player_relative, 3)

    dist = _min_distance(self_coords, beacon_coords)
    info["dist_to_beacon"] = dist

    beacon_center = _center(beacon_coords)
    beacon_moved = False
    if beacon_center is not None and state.prev_beacon_center is not None:
        shift = np.sqrt(
            (beacon_center[0] - state.prev_beacon_center[0]) ** 2
            + (beacon_center[1] - state.prev_beacon_center[1]) ** 2
        )
        # Beacon moved: reset distance baseline
        if shift >= 1.0:
            state.prev_distance = None
            beacon_moved = True
    state.prev_beacon_center = beacon_center
    info["beacon_moved"] = beacon_moved

    if dist is None:
        return 0.0

    reward = 0.0
    if state.prev_distance is not None:
        delta = state.prev_distance - dist
        reward += 0.05 * float(delta)

    # Bonus for reaching the beacon: use native reward as the reach signal
    native_reward = info.get("native_reward")
    if native_reward == 1.0:
        reward += 1.0

    state.prev_distance = dist
    return float(reward)


RewardFn = Callable[[Dict[str, Any], RewardState, Dict[str, Any]], float]


def get_reward_fn(map_name: str) -> Optional[RewardFn]:
    if map_name == "MoveToBeacon":
        return reward_move_to_beacon
    return None
