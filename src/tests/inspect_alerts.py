"""
测试PySC2环境下observation的alerts字段内容和含义。
"""
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import features
from absl import flags
flags.FLAGS(['run'])

def main():
    with sc2_env.SC2Env(
        map_name="MoveToBeacon",
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=64, minimap=64),
            use_feature_units=True,
            action_space=sc2_env.ActionSpace.FEATURES,
        ),
        step_mul=8,
        game_steps_per_episode=0,
        visualize=False,
    ) as env:
        timestep = env.reset()[0]
        obs = timestep.observation
        print("\n--- alerts 字段内容 ---")
        if hasattr(obs, "alerts"):
            alerts = getattr(obs, "alerts")
            print(f"type: {type(alerts)}")
            print(f"value: {alerts}")
            print(f"as numpy: {np.asarray(alerts)}")
            print(f"size: {np.asarray(alerts).size}")
        else:
            print("observation无alerts字段")
        # 若alerts非空，尝试解释其含义
        if hasattr(obs, "alerts") and len(getattr(obs, "alerts")) > 0:
            print("\n--- alerts 含义解释 ---")
            print("alerts是一个整数列表，每个元素代表一个警报类型，具体类型可参考pysc2.env.enums.Alert枚举。常见如：")
            try:
                from pysc2.env.enums import Alert
                for a in getattr(obs, "alerts"):
                    try:
                        alert_name = Alert(a).name
                    except Exception:
                        alert_name = "<未知类型>"
                    print(f"alert {a}: {alert_name}")
            except ImportError:
                print("未找到pysc2.env.enums.Alert，无法详细解释。")
        else:
            print("当前时刻无警报事件（alerts为空）。")

if __name__ == "__main__":
    main()
