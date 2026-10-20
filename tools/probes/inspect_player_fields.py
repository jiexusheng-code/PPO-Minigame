"""
测试PySC2 minigame环境下player字段类型与内容。
"""
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import features

from absl import flags
flags.FLAGS(['run'])

def inspect_player_fields(map_name="MoveToBeacon", step_mul=8):
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
        player = getattr(obs, "player", None)
        print(f"type(player): {type(player)}")
        if player is not None:
            arr = np.asarray(player)
            print(f"np.asarray(player).shape: {arr.shape}")
            print(f"player raw: {player}")
            print(f"player as array: {arr}")
            print("Field values:")
            for i, val in enumerate(arr.flatten()):
                print(f"  [{i}]: {val}")
        else:
            print("No player field found in observation.")

if __name__ == "__main__":
    inspect_player_fields()
