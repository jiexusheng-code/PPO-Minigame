"""
测试PySC2 minigame环境下score_cumulative字段类型与内容，并解释每一项含义。
"""
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import features

from absl import flags
flags.FLAGS(['run'])

def inspect_score_cumulative(map_name="MoveToBeacon", step_mul=8):
    with sc2_env.SC2Env(
        map_name=map_name,
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=64, minimap=64),
            use_feature_units=True,
            action_space=sc2_env.ActionSpace.FEATURES,
        ),
        step_mul=step_mul,
        game_steps_per_episode=0,
        visualize=False,
    ) as env:
        timestep = env.reset()[0]
        obs = timestep.observation
        score = getattr(obs, "score_cumulative", None)
        print(f"type(score_cumulative): {type(score)}")
        if score is not None:
            arr = np.asarray(score)
            print(f"np.asarray(score_cumulative).shape: {arr.shape}")
            print(f"score_cumulative raw: {score}")
            print(f"score_cumulative as array: {arr}")
            # 字段含义参考 pysc2/lib/features.py PlayerScore
            score_fields = [
                "score", "idle_production_time", "idle_worker_time", "total_value_units", "total_value_structures", "killed_value_units", "killed_value_structures", "collected_minerals", "collected_vespene", "collection_rate_minerals", "collection_rate_vespene", "spent_minerals", "spent_vespene"
            ]
            print("Field values:")
            for i, val in enumerate(arr.flatten()):
                name = score_fields[i] if i < len(score_fields) else f"unknown_{i}"
                print(f"  [{i}] {name}: {val}")
        else:
            print("No score_cumulative field found in observation.")

if __name__ == "__main__":
    inspect_score_cumulative()
