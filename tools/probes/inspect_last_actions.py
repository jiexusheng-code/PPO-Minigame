"""
测试PySC2 minigame环境reset后第一步observation是否包含last_actions字段。
"""
from pysc2.env import sc2_env
from pysc2.lib import features
from absl import flags
flags.FLAGS(['run'])

def inspect_last_actions(map_name="MoveToBeacon", step_mul=8):
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
        # 检查是否有last_actions字段
        has_attr = hasattr(obs, "last_actions")
        as_dict = obs._asdict() if hasattr(obs, "_asdict") else None
        in_dict = as_dict is not None and ("last_actions" in as_dict)
        print(f"hasattr(obs, 'last_actions'): {has_attr}")
        print(f"'last_actions' in obs._asdict(): {in_dict}")
        if has_attr:
            print(f"obs.last_actions: {getattr(obs, 'last_actions')}")
        if in_dict:
            print(f"obs._asdict()['last_actions']: {as_dict['last_actions']}")
        if not has_attr and not in_dict:
            print("last_actions字段确实不存在于reset后的第一步observation中。")

if __name__ == "__main__":
    inspect_last_actions()
