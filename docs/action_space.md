# Action space in PySC2

**接口类型 (RAW vs FEATURES)**

- **何时使用 RAW**: 需要直接使用世界坐标（`world`）、单位标签（`unit_tags`）或精细的低层控制时使用 RAW 接口。例如需要对原始单位标签或真实世界坐标发出命令，或在研究低级别动作策略时。创建环境时应设置 `AgentInterfaceFormat(use_raw_actions=True, use_raw_units=True)` 或 `action_space=ActionSpace.RAW`；构造动作时请使用 `RAW_FUNCTIONS` 或 `FunctionCall(..., raw=True)`。

- **何时使用 FEATURES（非‑RAW）**: 以 feature 层（`screen`/`minimap`）为基础进行高层交互时使用 FEATURES（默认常用）。适用于大多数基于像素的策略与常见示例，使用 `FUNCTIONS` 或不传 `raw` 的 `FunctionCall()`。

- **注意事项**：最终由环境决定哪类动作被接受 — 即 `AgentInterfaceFormat` / env 中的 `use_raw_actions`（或 `action_space=ActionSpace.RAW`）决定环境处于 RAW 模式还是 FEATURES 模式。`FunctionCall(..., raw=True)` 或使用 `RAW_FUNCTIONS` 仅决定动作在构造时使用哪张函数表进行验证和解释，但它不能绕过环境模式的检查；如果构造的动作与环境不匹配，`Features.transform_action` 会在发送时拒绝或报错。直接传入 `sc_pb.Action` proto 会绕过该封装检查，因此应谨慎使用。

本文档旨在系统性说明 PySC2 的动作空间（非 RAW / RAW）。

# 1. 非 RAW 动作

非 RAW 动作基于“屏幕 / 小地图坐标与 UI 交互逻辑”，动作以 `actions.FunctionCall` 表示：

```
FunctionCall(function_id, arguments)
```

其中 `function_id` 是动作函数 id，`arguments` 是该动作要求的参数列表（通常是若干整数或坐标对）。

## 1.1 非 RAW 动作使用示例

示例 1：在屏幕坐标上移动

```python
from pysc2.lib import actions

# Move_screen 需要两个参数：queued 与 screen
# queued: [0/1]，screen: [x, y]
act = actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], [x, y]])
```

示例 2：选择屏幕某点单位（可切换、添加）

```python
from pysc2.lib import actions

# select_point_act: [0..3] (select/toggle/select_all_type/add_all_type)
# screen: [x, y]
act = actions.FunctionCall(actions.FUNCTIONS.select_point.id, [[0], [x, y]])
```

## 1.2 非 RAW 动作组成部分（含义 / dtype / 取值范围）

非 RAW 动作由两部分组成：

1) **动作函数 id**（`int`）：取自 `actions.FUNCTIONS`。
2) **参数列表**（`List[List[int]]`）：每个参数由对应 `ArgumentType` 定义。


参数类型（来自 `actions.TYPES`）：

| 参数名 | 含义 | dtype | 取值范围 / 说明 |
|---|---|---|---|
| `screen` | 屏幕坐标 | `int32` | `[x, y]`，范围由 `action_spec()` 的 `screen` 尺寸决定。在创建环境时通过 `agent_interface_format` 的 `feature_dimensions` 配置（见下）。 |
| `minimap` | 小地图坐标 | `int32` | `[x, y]`，范围由 `action_spec()` 的 `minimap` 尺寸决定。通常分辨率低于 `screen`，用于全局操作。 |
| `screen2` | 第二屏幕坐标（框选） | `int32` | `[x, y]`，范围同 `screen`；用于线性框选的第二个角坐标或其他第二屏幕坐标用途。 |
| `queued` | 是否排队 | `int32` | `0=now`（立即执行），`1=queued`（加入队列后执行）。 |
| `control_group_act` | 控制组操作 | `int32` | `0..4`，具体含义见表下方。 |
| `control_group_id` | 控制组编号 | `int32` | `0..9`（典型为 10 个组）。 |
| `select_point_act` | 选择方式 | `int32` | `0..3`：`select`（选中）、`toggle`（切换选中状态）、`select_all_type`（按类型全选）、`add_all_type`（按类型追加）。 |
| `select_add` | 选择方式（是否追加） | `int32` | `0=select`（替换选中）、`1=add`（追加到当前选择）。 |
| `select_unit_act` | 多选面板行为 | `int32` | `0..3`（select / deselect / select_all_type / deselect_all_type）。 |
| `select_unit_id` | 多选面板索引 | `int32` | `0..499`。 |
| `select_worker` | 工人选择策略 | `int32` | `0..3`（select / add / select_all / add_all）。 |
| `build_queue_id` | 生产队列索引 | `int32` | `0..9`（依建筑/队列槽数而定）。 |
| `unload_id` | 载具中单位索引 | `int32` | `0..499`（依载具内单位数量变化）。 |


**`control_group_act` 的取值含义**

- `recall`：跳转摄像机/焦点至该控制组（用于查看或快速选中）。
- `set`：将当前选中单位设置为该控制组（覆盖原有控制组成员）。
- `append`：把当前选中单位追加到该控制组（保留原有成员）。
- `set_and_steal`：设置该控制组为当前选中单位，并在必要时从其它控制组中移除这些单位（用于“窃取”单位到该组）。
- `append_and_steal`：追加当前选中单位到该控制组，并在必要时从其它控制组中移除这些单位。

### 如何配置 `screen` / `minimap` / `screen2` 的尺寸

在创建 `SC2Env` 时通过 `agent_interface_format` 指定 `feature_dimensions`（使用 `features.AgentInterfaceFormat` 和 `features.Dimensions`）。示例：

```python
from pysc2.env import sc2_env
from pysc2.lib import features

agent_if = features.AgentInterfaceFormat(
	feature_dimensions=features.Dimensions(
		screen=(64, 64),    # 屏幕分辨率 (height, width)
		minimap=(64, 64),   # 小地图分辨率
	)
)

env = sc2_env.SC2Env(
	map_name='MoveToBeacon',
	agent_interface_format=agent_if,
	step_mul=8,
	game_steps_per_episode=0
)

print('action_spec:', env.action_spec())
```

上述设置将影响 `env.action_spec()` 返回的 `screen` / `minimap` 大小，`screen2` 与 `screen` 使用相同维度。

### 如何配置 `world` 的范围（RAW）

`world` 坐标用于 RAW 动作，其离散取值/尺寸由环境的 `raw_resolution`（通常来自地图的 `map_size` 或者由 `env.action_spec()` 报告）决定。换言之，`world` 的绝对范围由地图大小（world units）确定，而不是由屏幕/相机像素的参数决定。

## 1.3 非 RAW 动作 id 与参数对照表

完整动作表见 **附录 A**。

---

# 2. RAW 动作

RAW 动作直接基于“单位 tag + 世界坐标”进行控制，适合做更精细的底层控制。动作仍使用 `actions.FunctionCall`：

```
FunctionCall(function_id, arguments, raw=True)
```

不同之处在于：函数集合使用 `actions.RAW_FUNCTIONS`，参数类型来自 `actions.RAW_TYPES`。

## 2.1 RAW 动作使用示例

示例 1：对指定单位下达移动命令（世界坐标）

```python
from pysc2.lib import actions

# RAW 模式示例：环境必须以 RAW 模式创建，例如
# AgentInterfaceFormat(use_raw_actions=True, use_raw_units=True)
unit_tag = 123456  # 来自 obs['raw_units'] 的 tag 列
world_target = (30, 40)
# 显式以 raw=True 构造（或使用 RAW_FUNCTIONS helper）
act = actions.FunctionCall(actions.RAW_FUNCTIONS.Move_pt.id,
						   [[0], [unit_tag], list(world_target)],
						   raw=True)
```

示例 2：攻击指定单位

```python
from pysc2.lib import actions

# queued, unit_tags, target_unit_tag
act = actions.FunctionCall(actions.RAW_FUNCTIONS.Attack_unit.id, [[0], [unit_tag], [target_tag]], raw=True)
```

## 2.2 RAW 动作组成部分（含义 / dtype / 取值范围）

RAW 动作由两部分组成：

1) **动作函数 id**（`int`）：取自 `actions.RAW_FUNCTIONS`。
2) **参数列表**（`List[List[int]]`）：由 `RAW_TYPES` 定义。

参数类型（来自 `actions.RAW_TYPES`，详细由 `generated_raw_arguments.json` 生成）：

| 参数名 | 含义 | dtype | 取值范围 / 说明 |
|---|---|---|---|
| `world` | 世界坐标 | `int32` | `[x, y]`，范围由 `action_spec()` 的 `world` 尺寸决定 |
| `queued` | 是否排队 | `int32` | `0=now`，`1=queued` |
| `unit_tags` | 执行动作的单位 tag 列表 | `int64` | tag 为单位唯一标识，长度上限通常为 512 |
| `target_unit_tag` | 目标单位 tag | `int64` | 单个 tag（长度为 1） |

> 说明：`world` 坐标尺度由 `env.action_spec()` 给出，与 `raw_units` 的坐标系一致。

## 2.3 RAW 动作 id 与参数对照表

完整动作表见 **附录 B**。

# 3. 补充说明

`env.action_spec()` 返回当前动作空间的合法参数维度（例如 screen/minimap/world 的大小），这对于确定坐标范围非常关键。

# 附录 A：非 RAW 动作 id → 参数对照表（完整表）

| id | name | args |
|---|---|---|
| 0 | no_op |  |
| 1 | move_camera | minimap |
| 2 | select_point | select_point_act, screen |
| 3 | select_rect | select_add, screen, screen2 |
| 4 | select_control_group | control_group_act, control_group_id |
| 5 | select_unit | select_unit_act, select_unit_id |
| 6 | select_idle_worker | select_worker |
| 7 | select_army | select_add |
| 8 | select_warp_gates | select_add |
| 9 | select_larva |  |
| 10 | unload | unload_id |
| 11 | build_queue | build_queue_id |
| 12 | Attack_screen | queued, screen |
| 13 | Attack_minimap | queued, minimap |
| 14 | Attack_Attack_screen | queued, screen |
| 15 | Attack_Attack_minimap | queued, minimap |
| 16 | Attack_AttackBuilding_screen | queued, screen |
| 17 | Attack_AttackBuilding_minimap | queued, minimap |
| 18 | Attack_Redirect_screen | queued, screen |
| 19 | Scan_Move_screen | queued, screen |
| 20 | Scan_Move_minimap | queued, minimap |
| 21 | Behavior_BuildingAttackOff_quick | queued |
| 22 | Behavior_BuildingAttackOn_quick | queued |
| 23 | Behavior_CloakOff_quick | queued |
| 24 | Behavior_CloakOff_Banshee_quick | queued |
| 25 | Behavior_CloakOff_Ghost_quick | queued |
| 26 | Behavior_CloakOn_quick | queued |
| 27 | Behavior_CloakOn_Banshee_quick | queued |
| 28 | Behavior_CloakOn_Ghost_quick | queued |
| 29 | Behavior_GenerateCreepOff_quick | queued |
| 30 | Behavior_GenerateCreepOn_quick | queued |
| 31 | Behavior_HoldFireOff_quick | queued |
| 32 | Behavior_HoldFireOff_Ghost_quick | queued |
| 33 | Behavior_HoldFireOff_Lurker_quick | queued |
| 34 | Behavior_HoldFireOn_quick | queued |
| 35 | Behavior_HoldFireOn_Ghost_quick | queued |
| 36 | Behavior_HoldFireOn_Lurker_quick | queued |
| 37 | Behavior_PulsarBeamOff_quick | queued |
| 38 | Behavior_PulsarBeamOn_quick | queued |
| 39 | Build_Armory_screen | queued, screen |
| 40 | Build_Assimilator_screen | queued, screen |
| 41 | Build_BanelingNest_screen | queued, screen |
| 42 | Build_Barracks_screen | queued, screen |
| 43 | Build_Bunker_screen | queued, screen |
| 44 | Build_CommandCenter_screen | queued, screen |
| 45 | Build_CreepTumor_screen | queued, screen |
| 46 | Build_CreepTumor_Queen_screen | queued, screen |
| 47 | Build_CreepTumor_Tumor_screen | queued, screen |
| 48 | Build_CyberneticsCore_screen | queued, screen |
| 49 | Build_DarkShrine_screen | queued, screen |
| 50 | Build_EngineeringBay_screen | queued, screen |
| 51 | Build_EvolutionChamber_screen | queued, screen |
| 52 | Build_Extractor_screen | queued, screen |
| 53 | Build_Factory_screen | queued, screen |
| 54 | Build_FleetBeacon_screen | queued, screen |
| 55 | Build_Forge_screen | queued, screen |
| 56 | Build_FusionCore_screen | queued, screen |
| 57 | Build_Gateway_screen | queued, screen |
| 58 | Build_GhostAcademy_screen | queued, screen |
| 59 | Build_Hatchery_screen | queued, screen |
| 60 | Build_HydraliskDen_screen | queued, screen |
| 61 | Build_InfestationPit_screen | queued, screen |
| 62 | Build_Interceptors_quick | queued |
| 63 | Build_Interceptors_autocast |  |
| 64 | Build_MissileTurret_screen | queued, screen |
| 65 | Build_Nexus_screen | queued, screen |
| 66 | Build_Nuke_quick | queued |
| 67 | Build_NydusNetwork_screen | queued, screen |
| 68 | Build_NydusWorm_screen | queued, screen |
| 69 | Build_PhotonCannon_screen | queued, screen |
| 70 | Build_Pylon_screen | queued, screen |
| 71 | Build_Reactor_quick | queued |
| 72 | Build_Reactor_screen | queued, screen |
| 73 | Build_Reactor_Barracks_quick | queued |
| 74 | Build_Reactor_Barracks_screen | queued, screen |
| 75 | Build_Reactor_Factory_quick | queued |
| 76 | Build_Reactor_Factory_screen | queued, screen |
| 77 | Build_Reactor_Starport_quick | queued |
| 78 | Build_Reactor_Starport_screen | queued, screen |
| 79 | Build_Refinery_screen | queued, screen |
| 80 | Build_RoachWarren_screen | queued, screen |
| 81 | Build_RoboticsBay_screen | queued, screen |
| 82 | Build_RoboticsFacility_screen | queued, screen |
| 83 | Build_SensorTower_screen | queued, screen |
| 84 | Build_SpawningPool_screen | queued, screen |
| 85 | Build_SpineCrawler_screen | queued, screen |
| 86 | Build_Spire_screen | queued, screen |
| 87 | Build_SporeCrawler_screen | queued, screen |
| 88 | Build_Stargate_screen | queued, screen |
| 89 | Build_Starport_screen | queued, screen |
| 90 | Build_StasisTrap_screen | queued, screen |
| 91 | Build_SupplyDepot_screen | queued, screen |
| 92 | Build_TechLab_quick | queued |
| 93 | Build_TechLab_screen | queued, screen |
| 94 | Build_TechLab_Barracks_quick | queued |
| 95 | Build_TechLab_Barracks_screen | queued, screen |
| 96 | Build_TechLab_Factory_quick | queued |
| 97 | Build_TechLab_Factory_screen | queued, screen |
| 98 | Build_TechLab_Starport_quick | queued |
| 99 | Build_TechLab_Starport_screen | queued, screen |
| 100 | Build_TemplarArchive_screen | queued, screen |
| 101 | Build_TwilightCouncil_screen | queued, screen |
| 102 | Build_UltraliskCavern_screen | queued, screen |
| 103 | BurrowDown_quick | queued |
| 104 | BurrowDown_Baneling_quick | queued |
| 105 | BurrowDown_Drone_quick | queued |
| 106 | BurrowDown_Hydralisk_quick | queued |
| 107 | BurrowDown_Infestor_quick | queued |
| 108 | BurrowDown_InfestorTerran_quick | queued |
| 109 | BurrowDown_Lurker_quick | queued |
| 110 | BurrowDown_Queen_quick | queued |
| 111 | BurrowDown_Ravager_quick | queued |
| 112 | BurrowDown_Roach_quick | queued |
| 113 | BurrowDown_SwarmHost_quick | queued |
| 114 | BurrowDown_Ultralisk_quick | queued |
| 115 | BurrowDown_WidowMine_quick | queued |
| 116 | BurrowDown_Zergling_quick | queued |
| 117 | BurrowUp_quick | queued |
| 118 | BurrowUp_autocast |  |
| 119 | BurrowUp_Baneling_quick | queued |
| 120 | BurrowUp_Baneling_autocast |  |
| 121 | BurrowUp_Drone_quick | queued |
| 122 | BurrowUp_Hydralisk_quick | queued |
| 123 | BurrowUp_Hydralisk_autocast |  |
| 124 | BurrowUp_Infestor_quick | queued |
| 125 | BurrowUp_InfestorTerran_quick | queued |
| 126 | BurrowUp_InfestorTerran_autocast |  |
| 127 | BurrowUp_Lurker_quick | queued |
| 128 | BurrowUp_Queen_quick | queued |
| 129 | BurrowUp_Queen_autocast |  |
| 130 | BurrowUp_Ravager_quick | queued |
| 131 | BurrowUp_Ravager_autocast |  |
| 132 | BurrowUp_Roach_quick | queued |
| 133 | BurrowUp_Roach_autocast |  |
| 134 | BurrowUp_SwarmHost_quick | queued |
| 135 | BurrowUp_Ultralisk_quick | queued |
| 136 | BurrowUp_Ultralisk_autocast |  |
| 137 | BurrowUp_WidowMine_quick | queued |
| 138 | BurrowUp_Zergling_quick | queued |
| 139 | BurrowUp_Zergling_autocast |  |
| 140 | Cancel_quick | queued |
| 141 | Cancel_AdeptPhaseShift_quick | queued |
| 142 | Cancel_AdeptShadePhaseShift_quick | queued |
| 143 | Cancel_BarracksAddOn_quick | queued |
| 144 | Cancel_BuildInProgress_quick | queued |
| 145 | Cancel_CreepTumor_quick | queued |
| 146 | Cancel_FactoryAddOn_quick | queued |
| 147 | Cancel_GravitonBeam_quick | queued |
| 148 | Cancel_LockOn_quick | queued |
| 149 | Cancel_MorphBroodlord_quick | queued |
| 150 | Cancel_MorphGreaterSpire_quick | queued |
| 151 | Cancel_MorphHive_quick | queued |
| 152 | Cancel_MorphLair_quick | queued |
| 153 | Cancel_MorphLurker_quick | queued |
| 154 | Cancel_MorphLurkerDen_quick | queued |
| 155 | Cancel_MorphMothership_quick | queued |
| 156 | Cancel_MorphOrbital_quick | queued |
| 157 | Cancel_MorphOverlordTransport_quick | queued |
| 158 | Cancel_MorphOverseer_quick | queued |
| 159 | Cancel_MorphPlanetaryFortress_quick | queued |
| 160 | Cancel_MorphRavager_quick | queued |
| 161 | Cancel_MorphThorExplosiveMode_quick | queued |
| 162 | Cancel_NeuralParasite_quick | queued |
| 163 | Cancel_Nuke_quick | queued |
| 164 | Cancel_SpineCrawlerRoot_quick | queued |
| 165 | Cancel_SporeCrawlerRoot_quick | queued |
| 166 | Cancel_StarportAddOn_quick | queued |
| 167 | Cancel_StasisTrap_quick | queued |
| 168 | Cancel_Last_quick | queued |
| 169 | Cancel_HangarQueue5_quick | queued |
| 170 | Cancel_Queue1_quick | queued |
| 171 | Cancel_Queue5_quick | queued |
| 172 | Cancel_QueueAddOn_quick | queued |
| 173 | Cancel_QueueCancelToSelection_quick | queued |
| 174 | Cancel_QueuePassive_quick | queued |
| 175 | Cancel_QueuePassiveCancelToSelection_quick | queued |
| 176 | Effect_Abduct_screen | queued, screen |
| 177 | Effect_AdeptPhaseShift_screen | queued, screen |
| 178 | Effect_AutoTurret_screen | queued, screen |
| 179 | Effect_BlindingCloud_screen | queued, screen |
| 180 | Effect_Blink_screen | queued, screen |
| 181 | Effect_Blink_Stalker_screen | queued, screen |
| 182 | Effect_ShadowStride_screen | queued, screen |
| 183 | Effect_CalldownMULE_screen | queued, screen |
| 184 | Effect_CausticSpray_screen | queued, screen |
| 185 | Effect_Charge_screen | queued, screen |
| 186 | Effect_Charge_autocast |  |
| 187 | Effect_ChronoBoost_screen | queued, screen |
| 188 | Effect_Contaminate_screen | queued, screen |
| 189 | Effect_CorrosiveBile_screen | queued, screen |
| 190 | Effect_EMP_screen | queued, screen |
| 191 | Effect_Explode_quick | queued |
| 192 | Effect_Feedback_screen | queued, screen |
| 193 | Effect_ForceField_screen | queued, screen |
| 194 | Effect_FungalGrowth_screen | queued, screen |
| 195 | Effect_GhostSnipe_screen | queued, screen |
| 196 | Effect_GravitonBeam_screen | queued, screen |
| 197 | Effect_GuardianShield_quick | queued |
| 198 | Effect_Heal_screen | queued, screen |
| 199 | Effect_Heal_autocast |  |
| 200 | Effect_HunterSeekerMissile_screen | queued, screen |
| 201 | Effect_ImmortalBarrier_quick | queued |
| 202 | Effect_ImmortalBarrier_autocast |  |
| 203 | Effect_InfestedTerrans_screen | queued, screen |
| 204 | Effect_InjectLarva_screen | queued, screen |
| 205 | Effect_KD8Charge_screen | queued, screen |
| 206 | Effect_LockOn_screen | queued, screen |
| 207 | Effect_LocustSwoop_screen | queued, screen |
| 208 | Effect_MassRecall_screen | queued, screen |
| 209 | Effect_MassRecall_Mothership_screen | queued, screen |
| 210 | Effect_MassRecall_MothershipCore_screen | queued, screen |
| 211 | Effect_MedivacIgniteAfterburners_quick | queued |
| 212 | Effect_NeuralParasite_screen | queued, screen |
| 213 | Effect_NukeCalldown_screen | queued, screen |
| 214 | Effect_OracleRevelation_screen | queued, screen |
| 215 | Effect_ParasiticBomb_screen | queued, screen |
| 216 | Effect_PhotonOvercharge_screen | queued, screen |
| 217 | Effect_PointDefenseDrone_screen | queued, screen |
| 218 | Effect_PsiStorm_screen | queued, screen |
| 219 | Effect_PurificationNova_screen | queued, screen |
| 220 | Effect_Repair_screen | queued, screen |
| 221 | Effect_Repair_autocast |  |
| 222 | Effect_Repair_Mule_screen | queued, screen |
| 223 | Effect_Repair_Mule_autocast |  |
| 224 | Effect_Repair_SCV_screen | queued, screen |
| 225 | Effect_Repair_SCV_autocast |  |
| 226 | Effect_Salvage_quick | queued |
| 227 | Effect_Scan_screen | queued, screen |
| 228 | Effect_SpawnChangeling_quick | queued |
| 229 | Effect_SpawnLocusts_screen | queued, screen |
| 230 | Effect_Spray_screen | queued, screen |
| 231 | Effect_Spray_Protoss_screen | queued, screen |
| 232 | Effect_Spray_Terran_screen | queued, screen |
| 233 | Effect_Spray_Zerg_screen | queued, screen |
| 234 | Effect_Stim_quick | queued |
| 235 | Effect_Stim_Marauder_quick | queued |
| 236 | Effect_Stim_Marauder_Redirect_quick | queued |
| 237 | Effect_Stim_Marine_quick | queued |
| 238 | Effect_Stim_Marine_Redirect_quick | queued |
| 239 | Effect_SupplyDrop_screen | queued, screen |
| 240 | Effect_TacticalJump_screen | queued, screen |
| 241 | Effect_TimeWarp_screen | queued, screen |
| 242 | Effect_Transfusion_screen | queued, screen |
| 243 | Effect_ViperConsume_screen | queued, screen |
| 244 | Effect_VoidRayPrismaticAlignment_quick | queued |
| 245 | Effect_WidowMineAttack_screen | queued, screen |
| 246 | Effect_WidowMineAttack_autocast |  |
| 247 | Effect_YamatoGun_screen | queued, screen |
| 248 | Hallucination_Adept_quick | queued |
| 249 | Hallucination_Archon_quick | queued |
| 250 | Hallucination_Colossus_quick | queued |
| 251 | Hallucination_Disruptor_quick | queued |
| 252 | Hallucination_HighTemplar_quick | queued |
| 253 | Hallucination_Immortal_quick | queued |
| 254 | Hallucination_Oracle_quick | queued |
| 255 | Hallucination_Phoenix_quick | queued |
| 256 | Hallucination_Probe_quick | queued |
| 257 | Hallucination_Stalker_quick | queued |
| 258 | Hallucination_VoidRay_quick | queued |
| 259 | Hallucination_WarpPrism_quick | queued |
| 260 | Hallucination_Zealot_quick | queued |
| 261 | Halt_quick | queued |
| 262 | Halt_Building_quick | queued |
| 263 | Halt_TerranBuild_quick | queued |
| 264 | Harvest_Gather_screen | queued, screen |
| 265 | Harvest_Gather_Drone_screen | queued, screen |
| 266 | Harvest_Gather_Mule_screen | queued, screen |
| 267 | Harvest_Gather_Probe_screen | queued, screen |
| 268 | Harvest_Gather_SCV_screen | queued, screen |
| 269 | Harvest_Return_quick | queued |
| 270 | Harvest_Return_Drone_quick | queued |
| 271 | Harvest_Return_Mule_quick | queued |
| 272 | Harvest_Return_Probe_quick | queued |
| 273 | Harvest_Return_SCV_quick | queued |
| 274 | HoldPosition_quick | queued |
| 275 | Land_screen | queued, screen |
| 276 | Land_Barracks_screen | queued, screen |
| 277 | Land_CommandCenter_screen | queued, screen |
| 278 | Land_Factory_screen | queued, screen |
| 279 | Land_OrbitalCommand_screen | queued, screen |
| 280 | Land_Starport_screen | queued, screen |
| 281 | Lift_quick | queued |
| 282 | Lift_Barracks_quick | queued |
| 283 | Lift_CommandCenter_quick | queued |
| 284 | Lift_Factory_quick | queued |
| 285 | Lift_OrbitalCommand_quick | queued |
| 286 | Lift_Starport_quick | queued |
| 287 | Load_screen | queued, screen |
| 288 | Load_Bunker_screen | queued, screen |
| 289 | Load_Medivac_screen | queued, screen |
| 290 | Load_NydusNetwork_screen | queued, screen |
| 291 | Load_NydusWorm_screen | queued, screen |
| 292 | Load_Overlord_screen | queued, screen |
| 293 | Load_WarpPrism_screen | queued, screen |
| 294 | LoadAll_quick | queued |
| 295 | LoadAll_CommandCenter_quick | queued |
| 296 | Morph_Archon_quick | queued |
| 297 | Morph_BroodLord_quick | queued |
| 298 | Morph_Gateway_quick | queued |
| 299 | Morph_GreaterSpire_quick | queued |
| 300 | Morph_Hellbat_quick | queued |
| 301 | Morph_Hellion_quick | queued |
| 302 | Morph_Hive_quick | queued |
| 303 | Morph_Lair_quick | queued |
| 304 | Morph_LiberatorAAMode_quick | queued |
| 305 | Morph_LiberatorAGMode_screen | queued, screen |
| 306 | Morph_Lurker_quick | queued |
| 307 | Morph_LurkerDen_quick | queued |
| 308 | Morph_Mothership_quick | queued |
| 309 | Morph_OrbitalCommand_quick | queued |
| 310 | Morph_OverlordTransport_quick | queued |
| 311 | Morph_Overseer_quick | queued |
| 312 | Morph_PlanetaryFortress_quick | queued |
| 313 | Morph_Ravager_quick | queued |
| 314 | Morph_Root_screen | queued, screen |
| 315 | Morph_SpineCrawlerRoot_screen | queued, screen |
| 316 | Morph_SporeCrawlerRoot_screen | queued, screen |
| 317 | Morph_SiegeMode_quick | queued |
| 318 | Morph_SupplyDepot_Lower_quick | queued |
| 319 | Morph_SupplyDepot_Raise_quick | queued |
| 320 | Morph_ThorExplosiveMode_quick | queued |
| 321 | Morph_ThorHighImpactMode_quick | queued |
| 322 | Morph_Unsiege_quick | queued |
| 323 | Morph_Uproot_quick | queued |
| 324 | Morph_SpineCrawlerUproot_quick | queued |
| 325 | Morph_SporeCrawlerUproot_quick | queued |
| 326 | Morph_VikingAssaultMode_quick | queued |
| 327 | Morph_VikingFighterMode_quick | queued |
| 328 | Morph_WarpGate_quick | queued |
| 329 | Morph_WarpPrismPhasingMode_quick | queued |
| 330 | Morph_WarpPrismTransportMode_quick | queued |
| 331 | Move_screen | queued, screen |
| 332 | Move_minimap | queued, minimap |
| 333 | Patrol_screen | queued, screen |
| 334 | Patrol_minimap | queued, minimap |
| 335 | Rally_Units_screen | queued, screen |
| 336 | Rally_Units_minimap | queued, minimap |
| 337 | Rally_Building_screen | queued, screen |
| 338 | Rally_Building_minimap | queued, minimap |
| 339 | Rally_Hatchery_Units_screen | queued, screen |
| 340 | Rally_Hatchery_Units_minimap | queued, minimap |
| 341 | Rally_Morphing_Unit_screen | queued, screen |
| 342 | Rally_Morphing_Unit_minimap | queued, minimap |
| 343 | Rally_Workers_screen | queued, screen |
| 344 | Rally_Workers_minimap | queued, minimap |
| 345 | Rally_CommandCenter_screen | queued, screen |
| 346 | Rally_CommandCenter_minimap | queued, minimap |
| 347 | Rally_Hatchery_Workers_screen | queued, screen |
| 348 | Rally_Hatchery_Workers_minimap | queued, minimap |
| 349 | Rally_Nexus_screen | queued, screen |
| 350 | Rally_Nexus_minimap | queued, minimap |
| 351 | Research_AdeptResonatingGlaives_quick | queued |
| 352 | Research_AdvancedBallistics_quick | queued |
| 353 | Research_BansheeCloakingField_quick | queued |
| 354 | Research_BansheeHyperflightRotors_quick | queued |
| 355 | Research_BattlecruiserWeaponRefit_quick | queued |
| 356 | Research_Blink_quick | queued |
| 357 | Research_Burrow_quick | queued |
| 358 | Research_CentrifugalHooks_quick | queued |
| 359 | Research_Charge_quick | queued |
| 360 | Research_ChitinousPlating_quick | queued |
| 361 | Research_CombatShield_quick | queued |
| 362 | Research_ConcussiveShells_quick | queued |
| 363 | Research_DrillingClaws_quick | queued |
| 364 | Research_ExtendedThermalLance_quick | queued |
| 365 | Research_GlialRegeneration_quick | queued |
| 366 | Research_GraviticBooster_quick | queued |
| 367 | Research_GraviticDrive_quick | queued |
| 368 | Research_GroovedSpines_quick | queued |
| 369 | Research_HiSecAutoTracking_quick | queued |
| 370 | Research_HighCapacityFuelTanks_quick | queued |
| 371 | Research_InfernalPreigniter_quick | queued |
| 372 | Research_InterceptorGravitonCatapult_quick | queued |
| 373 | Research_SmartServos_quick | queued |
| 374 | Research_MuscularAugments_quick | queued |
| 375 | Research_NeosteelFrame_quick | queued |
| 376 | Research_NeuralParasite_quick | queued |
| 377 | Research_PathogenGlands_quick | queued |
| 378 | Research_PersonalCloaking_quick | queued |
| 379 | Research_PhoenixAnionPulseCrystals_quick | queued |
| 380 | Research_PneumatizedCarapace_quick | queued |
| 381 | Research_ProtossAirArmor_quick | queued |
| 382 | Research_ProtossAirArmorLevel1_quick | queued |
| 383 | Research_ProtossAirArmorLevel2_quick | queued |
| 384 | Research_ProtossAirArmorLevel3_quick | queued |
| 385 | Research_ProtossAirWeapons_quick | queued |
| 386 | Research_ProtossAirWeaponsLevel1_quick | queued |
| 387 | Research_ProtossAirWeaponsLevel2_quick | queued |
| 388 | Research_ProtossAirWeaponsLevel3_quick | queued |
| 389 | Research_ProtossGroundArmor_quick | queued |
| 390 | Research_ProtossGroundArmorLevel1_quick | queued |
| 391 | Research_ProtossGroundArmorLevel2_quick | queued |
| 392 | Research_ProtossGroundArmorLevel3_quick | queued |
| 393 | Research_ProtossGroundWeapons_quick | queued |
| 394 | Research_ProtossGroundWeaponsLevel1_quick | queued |
| 395 | Research_ProtossGroundWeaponsLevel2_quick | queued |
| 396 | Research_ProtossGroundWeaponsLevel3_quick | queued |
| 397 | Research_ProtossShields_quick | queued |
| 398 | Research_ProtossShieldsLevel1_quick | queued |
| 399 | Research_ProtossShieldsLevel2_quick | queued |
| 400 | Research_ProtossShieldsLevel3_quick | queued |
| 401 | Research_PsiStorm_quick | queued |
| 402 | Research_RavenCorvidReactor_quick | queued |
| 403 | Research_RavenRecalibratedExplosives_quick | queued |
| 404 | Research_ShadowStrike_quick | queued |
| 405 | Research_Stimpack_quick | queued |
| 406 | Research_TerranInfantryArmor_quick | queued |
| 407 | Research_TerranInfantryArmorLevel1_quick | queued |
| 408 | Research_TerranInfantryArmorLevel2_quick | queued |
| 409 | Research_TerranInfantryArmorLevel3_quick | queued |
| 410 | Research_TerranInfantryWeapons_quick | queued |
| 411 | Research_TerranInfantryWeaponsLevel1_quick | queued |
| 412 | Research_TerranInfantryWeaponsLevel2_quick | queued |
| 413 | Research_TerranInfantryWeaponsLevel3_quick | queued |
| 414 | Research_TerranShipWeapons_quick | queued |
| 415 | Research_TerranShipWeaponsLevel1_quick | queued |
| 416 | Research_TerranShipWeaponsLevel2_quick | queued |
| 417 | Research_TerranShipWeaponsLevel3_quick | queued |
| 418 | Research_TerranStructureArmorUpgrade_quick | queued |
| 419 | Research_TerranVehicleAndShipPlating_quick | queued |
| 420 | Research_TerranVehicleAndShipPlatingLevel1_quick | queued |
| 421 | Research_TerranVehicleAndShipPlatingLevel2_quick | queued |
| 422 | Research_TerranVehicleAndShipPlatingLevel3_quick | queued |
| 423 | Research_TerranVehicleWeapons_quick | queued |
| 424 | Research_TerranVehicleWeaponsLevel1_quick | queued |
| 425 | Research_TerranVehicleWeaponsLevel2_quick | queued |
| 426 | Research_TerranVehicleWeaponsLevel3_quick | queued |
| 427 | Research_TunnelingClaws_quick | queued |
| 428 | Research_WarpGate_quick | queued |
| 429 | Research_ZergFlyerArmor_quick | queued |
| 430 | Research_ZergFlyerArmorLevel1_quick | queued |
| 431 | Research_ZergFlyerArmorLevel2_quick | queued |
| 432 | Research_ZergFlyerArmorLevel3_quick | queued |
| 433 | Research_ZergFlyerAttack_quick | queued |
| 434 | Research_ZergFlyerAttackLevel1_quick | queued |
| 435 | Research_ZergFlyerAttackLevel2_quick | queued |
| 436 | Research_ZergFlyerAttackLevel3_quick | queued |
| 437 | Research_ZergGroundArmor_quick | queued |
| 438 | Research_ZergGroundArmorLevel1_quick | queued |
| 439 | Research_ZergGroundArmorLevel2_quick | queued |
| 440 | Research_ZergGroundArmorLevel3_quick | queued |
| 441 | Research_ZergMeleeWeapons_quick | queued |
| 442 | Research_ZergMeleeWeaponsLevel1_quick | queued |
| 443 | Research_ZergMeleeWeaponsLevel2_quick | queued |
| 444 | Research_ZergMeleeWeaponsLevel3_quick | queued |
| 445 | Research_ZergMissileWeapons_quick | queued |
| 446 | Research_ZergMissileWeaponsLevel1_quick | queued |
| 447 | Research_ZergMissileWeaponsLevel2_quick | queued |
| 448 | Research_ZergMissileWeaponsLevel3_quick | queued |
| 449 | Research_ZerglingAdrenalGlands_quick | queued |
| 450 | Research_ZerglingMetabolicBoost_quick | queued |
| 451 | Smart_screen | queued, screen |
| 452 | Smart_minimap | queued, minimap |
| 453 | Stop_quick | queued |
| 454 | Stop_Building_quick | queued |
| 455 | Stop_Redirect_quick | queued |
| 456 | Stop_Stop_quick | queued |
| 457 | Train_Adept_quick | queued |
| 458 | Train_Baneling_quick | queued |
| 459 | Train_Banshee_quick | queued |
| 460 | Train_Battlecruiser_quick | queued |
| 461 | Train_Carrier_quick | queued |
| 462 | Train_Colossus_quick | queued |
| 463 | Train_Corruptor_quick | queued |
| 464 | Train_Cyclone_quick | queued |
| 465 | Train_DarkTemplar_quick | queued |
| 466 | Train_Disruptor_quick | queued |
| 467 | Train_Drone_quick | queued |
| 468 | Train_Ghost_quick | queued |
| 469 | Train_Hellbat_quick | queued |
| 470 | Train_Hellion_quick | queued |
| 471 | Train_HighTemplar_quick | queued |
| 472 | Train_Hydralisk_quick | queued |
| 473 | Train_Immortal_quick | queued |
| 474 | Train_Infestor_quick | queued |
| 475 | Train_Liberator_quick | queued |
| 476 | Train_Marauder_quick | queued |
| 477 | Train_Marine_quick | queued |
| 478 | Train_Medivac_quick | queued |
| 479 | Train_MothershipCore_quick | queued |
| 480 | Train_Mutalisk_quick | queued |
| 481 | Train_Observer_quick | queued |
| 482 | Train_Oracle_quick | queued |
| 483 | Train_Overlord_quick | queued |
| 484 | Train_Phoenix_quick | queued |
| 485 | Train_Probe_quick | queued |
| 486 | Train_Queen_quick | queued |
| 487 | Train_Raven_quick | queued |
| 488 | Train_Reaper_quick | queued |
| 489 | Train_Roach_quick | queued |
| 490 | Train_SCV_quick | queued |
| 491 | Train_Sentry_quick | queued |
| 492 | Train_SiegeTank_quick | queued |
| 493 | Train_Stalker_quick | queued |
| 494 | Train_SwarmHost_quick | queued |
| 495 | Train_Tempest_quick | queued |
| 496 | Train_Thor_quick | queued |
| 497 | Train_Ultralisk_quick | queued |
| 498 | Train_VikingFighter_quick | queued |
| 499 | Train_Viper_quick | queued |
| 500 | Train_VoidRay_quick | queued |
| 501 | Train_WarpPrism_quick | queued |
| 502 | Train_WidowMine_quick | queued |
| 503 | Train_Zealot_quick | queued |
| 504 | Train_Zergling_quick | queued |
| 505 | TrainWarp_Adept_screen | queued, screen |
| 506 | TrainWarp_DarkTemplar_screen | queued, screen |
| 507 | TrainWarp_HighTemplar_screen | queued, screen |
| 508 | TrainWarp_Sentry_screen | queued, screen |
| 509 | TrainWarp_Stalker_screen | queued, screen |
| 510 | TrainWarp_Zealot_screen | queued, screen |
| 511 | UnloadAll_quick | queued |
| 512 | UnloadAll_Bunker_quick | queued |
| 513 | UnloadAll_CommandCenter_quick | queued |
| 514 | UnloadAll_NydusNetwork_quick | queued |
| 515 | UnloadAll_NydusWorm_quick | queued |
| 516 | UnloadAllAt_screen | queued, screen |
| 517 | UnloadAllAt_minimap | queued, minimap |
| 518 | UnloadAllAt_Medivac_screen | queued, screen |
| 519 | UnloadAllAt_Medivac_minimap | queued, minimap |
| 520 | UnloadAllAt_Overlord_screen | queued, screen |
| 521 | UnloadAllAt_Overlord_minimap | queued, minimap |
| 522 | UnloadAllAt_WarpPrism_screen | queued, screen |
| 523 | UnloadAllAt_WarpPrism_minimap | queued, minimap |
| 524 | Build_LurkerDen_screen | queued, screen |
| 525 | Build_ShieldBattery_screen | queued, screen |
| 526 | Effect_AntiArmorMissile_screen | queued, screen |
| 527 | Effect_ChronoBoostEnergyCost_screen | queued, screen |
| 528 | Effect_InterferenceMatrix_screen | queued, screen |
| 529 | Effect_MassRecall_Nexus_screen | queued, screen |
| 530 | Effect_Repair_RepairDrone_screen | queued, screen |
| 531 | Effect_Repair_RepairDrone_autocast |  |
| 532 | Effect_RepairDrone_screen | queued, screen |
| 533 | Effect_Restore_screen | queued, screen |
| 534 | Effect_Restore_autocast |  |
| 535 | Morph_ObserverMode_quick | queued |
| 536 | Morph_OverseerMode_quick | queued |
| 537 | Morph_OversightMode_quick | queued |
| 538 | Morph_SurveillanceMode_quick | queued |
| 539 | Research_AdaptiveTalons_quick | queued |
| 540 | Research_CycloneRapidFireLaunchers_quick | queued |
| 541 | Train_Mothership_quick | queued |
| 542 | Effect_Scan_minimap | queued, minimap |
| 543 | Effect_Blink_minimap | queued, minimap |
| 544 | Effect_Blink_Stalker_minimap | queued, minimap |
| 545 | Effect_ShadowStride_minimap | queued, minimap |
| 546 | Cancel_VoidRayPrismaticAlignment_quick | queued |
| 547 | Effect_AdeptPhaseShift_minimap | queued, minimap |
| 548 | Effect_MassRecall_StrategicRecall_screen | queued, screen |
| 549 | Effect_Spray_minimap | queued, minimap |
| 550 | Effect_Spray_Protoss_minimap | queued, minimap |
| 551 | Effect_Spray_Terran_minimap | queued, minimap |
| 552 | Effect_Spray_Zerg_minimap | queued, minimap |
| 553 | Effect_TacticalJump_minimap | queued, minimap |
| 554 | Morph_LiberatorAGMode_minimap | queued, minimap |
| 555 | Attack_Battlecruiser_screen | queued, screen |
| 556 | Attack_Battlecruiser_minimap | queued, minimap |
| 557 | Effect_LockOn_autocast |  |
| 558 | HoldPosition_Battlecruiser_quick | queued |
| 559 | HoldPosition_Hold_quick | queued |
| 560 | Morph_WarpGate_autocast |  |
| 561 | Move_Battlecruiser_screen | queued, screen |
| 562 | Move_Battlecruiser_minimap | queued, minimap |
| 563 | Move_Move_screen | queued, screen |
| 564 | Move_Move_minimap | queued, minimap |
| 565 | Patrol_Battlecruiser_screen | queued, screen |
| 566 | Patrol_Battlecruiser_minimap | queued, minimap |
| 567 | Patrol_Patrol_screen | queued, screen |
| 568 | Patrol_Patrol_minimap | queued, minimap |
| 569 | Research_AnabolicSynthesis_quick | queued |
| 570 | Research_CycloneLockOnDamage_quick | queued |
| 571 | Stop_Battlecruiser_quick | queued |
| 572 | Research_EnhancedShockwaves_quick | queued |


# 附录 B：RAW 动作 id → 参数对照表（完整表）

| id | name | args |
|---|---|---|
| 0 | no_op |  |
| 1 | Smart_pt | queued, unit_tags, world |
| 2 | Attack_pt | queued, unit_tags, world |
| 3 | Attack_unit | queued, unit_tags, target_unit_tag |
| 4 | Attack_Attack_pt | queued, unit_tags, world |
| 5 | Attack_Attack_unit | queued, unit_tags, target_unit_tag |
| 6 | Attack_AttackBuilding_pt | queued, unit_tags, world |
| 7 | Attack_AttackBuilding_unit | queued, unit_tags, target_unit_tag |
| 8 | Attack_Redirect_pt | queued, unit_tags, world |
| 9 | Attack_Redirect_unit | queued, unit_tags, target_unit_tag |
| 10 | Scan_Move_pt | queued, unit_tags, world |
| 11 | Scan_Move_unit | queued, unit_tags, target_unit_tag |
| 12 | Smart_unit | queued, unit_tags, target_unit_tag |
| 13 | Move_pt | queued, unit_tags, world |
| 14 | Move_unit | queued, unit_tags, target_unit_tag |
| 15 | Patrol_pt | queued, unit_tags, world |
| 16 | Patrol_unit | queued, unit_tags, target_unit_tag |
| 17 | HoldPosition_quick | queued, unit_tags |
| 18 | Research_InterceptorGravitonCatapult_quick | queued, unit_tags |
| 19 | Research_PhoenixAnionPulseCrystals_quick | queued, unit_tags |
| 20 | Effect_GuardianShield_quick | queued, unit_tags |
| 21 | Train_Mothership_quick | queued, unit_tags |
| 22 | Hallucination_Archon_quick | queued, unit_tags |
| 23 | Hallucination_Colossus_quick | queued, unit_tags |
| 24 | Hallucination_HighTemplar_quick | queued, unit_tags |
| 25 | Hallucination_Immortal_quick | queued, unit_tags |
| 26 | Hallucination_Phoenix_quick | queued, unit_tags |
| 27 | Hallucination_Probe_quick | queued, unit_tags |
| 28 | Hallucination_Stalker_quick | queued, unit_tags |
| 29 | Hallucination_VoidRay_quick | queued, unit_tags |
| 30 | Hallucination_WarpPrism_quick | queued, unit_tags |
| 31 | Hallucination_Zealot_quick | queued, unit_tags |
| 32 | Effect_GravitonBeam_unit | queued, unit_tags, target_unit_tag |
| 33 | Effect_ChronoBoost_unit | queued, unit_tags, target_unit_tag |
| 34 | Build_Nexus_pt | queued, unit_tags, world |
| 35 | Build_Pylon_pt | queued, unit_tags, world |
| 36 | Build_Assimilator_unit | queued, unit_tags, target_unit_tag |
| 37 | Build_Gateway_pt | queued, unit_tags, world |
| 38 | Build_Forge_pt | queued, unit_tags, world |
| 39 | Build_FleetBeacon_pt | queued, unit_tags, world |
| 40 | Build_TwilightCouncil_pt | queued, unit_tags, world |
| 41 | Build_PhotonCannon_pt | queued, unit_tags, world |
| 42 | Build_Stargate_pt | queued, unit_tags, world |
| 43 | Build_TemplarArchive_pt | queued, unit_tags, world |
| 44 | Build_DarkShrine_pt | queued, unit_tags, world |
| 45 | Build_RoboticsBay_pt | queued, unit_tags, world |
| 46 | Build_RoboticsFacility_pt | queued, unit_tags, world |
| 47 | Build_CyberneticsCore_pt | queued, unit_tags, world |
| 48 | Build_ShieldBattery_pt | queued, unit_tags, world |
| 49 | Train_Zealot_quick | queued, unit_tags |
| 50 | Train_Stalker_quick | queued, unit_tags |
| 51 | Train_HighTemplar_quick | queued, unit_tags |
| 52 | Train_DarkTemplar_quick | queued, unit_tags |
| 53 | Train_Sentry_quick | queued, unit_tags |
| 54 | Train_Adept_quick | queued, unit_tags |
| 55 | Train_Phoenix_quick | queued, unit_tags |
| 56 | Train_Carrier_quick | queued, unit_tags |
| 57 | Train_VoidRay_quick | queued, unit_tags |
| 58 | Train_Oracle_quick | queued, unit_tags |
| 59 | Train_Tempest_quick | queued, unit_tags |
| 60 | Train_WarpPrism_quick | queued, unit_tags |
| 61 | Train_Observer_quick | queued, unit_tags |
| 62 | Train_Colossus_quick | queued, unit_tags |
| 63 | Train_Immortal_quick | queued, unit_tags |
| 64 | Train_Probe_quick | queued, unit_tags |
| 65 | Effect_PsiStorm_pt | queued, unit_tags, world |
| 66 | Build_Interceptors_quick | queued, unit_tags |
| 67 | Research_GraviticBooster_quick | queued, unit_tags |
| 68 | Research_GraviticDrive_quick | queued, unit_tags |
| 69 | Research_ExtendedThermalLance_quick | queued, unit_tags |
| 70 | Research_PsiStorm_quick | queued, unit_tags |
| 71 | TrainWarp_Zealot_pt | queued, unit_tags, world |
| 72 | TrainWarp_Stalker_pt | queued, unit_tags, world |
| 73 | TrainWarp_HighTemplar_pt | queued, unit_tags, world |
| 74 | TrainWarp_DarkTemplar_pt | queued, unit_tags, world |
| 75 | TrainWarp_Sentry_pt | queued, unit_tags, world |
| 76 | TrainWarp_Adept_pt | queued, unit_tags, world |
| 77 | Morph_WarpGate_quick | queued, unit_tags |
| 78 | Morph_Gateway_quick | queued, unit_tags |
| 79 | Effect_ForceField_pt | queued, unit_tags, world |
| 80 | Morph_WarpPrismPhasingMode_quick | queued, unit_tags |
| 81 | Morph_WarpPrismTransportMode_quick | queued, unit_tags |
| 82 | Research_WarpGate_quick | queued, unit_tags |
| 83 | Research_Charge_quick | queued, unit_tags |
| 84 | Research_Blink_quick | queued, unit_tags |
| 85 | Research_AdeptResonatingGlaives_quick | queued, unit_tags |
| 86 | Morph_Archon_quick | queued, unit_tags |
| 87 | Behavior_BuildingAttackOn_quick | queued, unit_tags |
| 88 | Behavior_BuildingAttackOff_quick | queued, unit_tags |
| 89 | Hallucination_Oracle_quick | queued, unit_tags |
| 90 | Effect_OracleRevelation_pt | queued, unit_tags, world |
| 91 | Effect_ImmortalBarrier_quick | queued, unit_tags |
| 92 | Hallucination_Disruptor_quick | queued, unit_tags |
| 93 | Hallucination_Adept_quick | queued, unit_tags |
| 94 | Effect_VoidRayPrismaticAlignment_quick | queued, unit_tags |
| 95 | Build_StasisTrap_pt | queued, unit_tags, world |
| 96 | Effect_AdeptPhaseShift_pt | queued, unit_tags, world |
| 97 | Research_ShadowStrike_quick | queued, unit_tags |
| 98 | Cancel_quick | queued, unit_tags |
| 99 | Halt_quick | queued, unit_tags |
| 100 | UnloadAll_quick | queued, unit_tags |
| 101 | Stop_quick | queued, unit_tags |
| 102 | Harvest_Gather_unit | queued, unit_tags, target_unit_tag |
| 103 | Harvest_Return_quick | queued, unit_tags |
| 104 | Load_unit | queued, unit_tags, target_unit_tag |
| 105 | UnloadAllAt_pt | queued, unit_tags, world |
| 106 | Rally_Units_pt | queued, unit_tags, world |
| 107 | Rally_Units_unit | queued, unit_tags, target_unit_tag |
| 108 | Effect_Repair_pt | queued, unit_tags, world |
| 109 | Effect_Repair_unit | queued, unit_tags, target_unit_tag |
| 110 | Effect_MassRecall_pt | queued, unit_tags, world |
| 111 | Effect_Blink_pt | queued, unit_tags, world |
| 112 | Effect_Blink_unit | queued, unit_tags, target_unit_tag |
| 113 | Effect_ShadowStride_pt | queued, unit_tags, world |
| 114 | Rally_Workers_pt | queued, unit_tags, world |
| 115 | Rally_Workers_unit | queued, unit_tags, target_unit_tag |
| 116 | Research_ProtossAirArmor_quick | queued, unit_tags |
| 117 | Research_ProtossAirWeapons_quick | queued, unit_tags |
| 118 | Research_ProtossGroundArmor_quick | queued, unit_tags |
| 119 | Research_ProtossGroundWeapons_quick | queued, unit_tags |
| 120 | Research_ProtossShields_quick | queued, unit_tags |
| 121 | Morph_ObserverMode_quick | queued, unit_tags |
| 122 | Effect_ChronoBoostEnergyCost_unit | queued, unit_tags, target_unit_tag |
| 123 | Cancel_AdeptPhaseShift_quick | queued, unit_tags |
| 124 | Cancel_AdeptShadePhaseShift_quick | queued, unit_tags |
| 125 | Cancel_BuildInProgress_quick | queued, unit_tags |
| 126 | Cancel_GravitonBeam_quick | queued, unit_tags |
| 127 | Cancel_StasisTrap_quick | queued, unit_tags |
| 128 | Cancel_VoidRayPrismaticAlignment_quick | queued, unit_tags |
| 129 | Cancel_Last_quick | queued, unit_tags |
| 130 | Cancel_Queue1_quick | queued, unit_tags |
| 131 | Cancel_Queue5_quick | queued, unit_tags |
| 132 | Cancel_QueueCancelToSelection_quick | queued, unit_tags |
| 133 | Cancel_QueuePassive_quick | queued, unit_tags |
| 134 | Cancel_QueuePassiveCancelToSelection_quick | queued, unit_tags |
| 135 | Effect_Blink_Stalker_pt | queued, unit_tags, world |
| 136 | Effect_MassRecall_Mothership_pt | queued, unit_tags, world |
| 137 | Effect_MassRecall_StrategicRecall_pt | queued, unit_tags, world |
| 138 | Rally_Nexus_pt | queued, unit_tags, world |
| 139 | Research_ProtossAirArmorLevel1_quick | queued, unit_tags |
| 140 | Research_ProtossAirArmorLevel2_quick | queued, unit_tags |
| 141 | Research_ProtossAirArmorLevel3_quick | queued, unit_tags |
| 142 | Research_ProtossAirWeaponsLevel1_quick | queued, unit_tags |
| 143 | Research_ProtossAirWeaponsLevel2_quick | queued, unit_tags |
| 144 | Research_ProtossAirWeaponsLevel3_quick | queued, unit_tags |
| 145 | Research_ProtossGroundArmorLevel1_quick | queued, unit_tags |
| 146 | Research_ProtossGroundArmorLevel2_quick | queued, unit_tags |
| 147 | Research_ProtossGroundArmorLevel3_quick | queued, unit_tags |
| 148 | Research_ProtossGroundWeaponsLevel1_quick | queued, unit_tags |
| 149 | Research_ProtossGroundWeaponsLevel2_quick | queued, unit_tags |
| 150 | Research_ProtossGroundWeaponsLevel3_quick | queued, unit_tags |
| 151 | Research_ProtossShieldsLevel1_quick | queued, unit_tags |
| 152 | Research_ProtossShieldsLevel2_quick | queued, unit_tags |
| 153 | Research_ProtossShieldsLevel3_quick | queued, unit_tags |
| 154 | Harvest_Return_Probe_quick | queued, unit_tags |
| 155 | Stop_Stop_quick | queued, unit_tags |
| 156 | UnloadAllAt_WarpPrism_pt | queued, unit_tags, world |
| 157 | Effect_Feedback_unit | queued, unit_tags, target_unit_tag |
| 158 | Behavior_PulsarBeamOff_quick | queued, unit_tags |
| 159 | Behavior_PulsarBeamOn_quick | queued, unit_tags |
| 160 | Morph_SurveillanceMode_quick | queued, unit_tags |
| 161 | Effect_Restore_unit | queued, unit_tags, target_unit_tag |
| 162 | Effect_MassRecall_Nexus_pt | queued, unit_tags, world |
| 163 | UnloadAllAt_WarpPrism_unit | queued, unit_tags, target_unit_tag |
| 164 | UnloadAllAt_unit | queued, unit_tags, target_unit_tag |
| 165 | Rally_Nexus_unit | queued, unit_tags, target_unit_tag |
| 166 | Train_Disruptor_quick | queued, unit_tags |
| 167 | Effect_PurificationNova_pt | queued, unit_tags, world |
| 168 | raw_move_camera | world |
| 169 | Behavior_CloakOff_quick | queued, unit_tags |
| 170 | Behavior_CloakOff_Banshee_quick | queued, unit_tags |
| 171 | Behavior_CloakOff_Ghost_quick | queued, unit_tags |
| 172 | Behavior_CloakOn_quick | queued, unit_tags |
| 173 | Behavior_CloakOn_Banshee_quick | queued, unit_tags |
| 174 | Behavior_CloakOn_Ghost_quick | queued, unit_tags |
| 175 | Behavior_GenerateCreepOff_quick | queued, unit_tags |
| 176 | Behavior_GenerateCreepOn_quick | queued, unit_tags |
| 177 | Behavior_HoldFireOff_quick | queued, unit_tags |
| 178 | Behavior_HoldFireOff_Ghost_quick | queued, unit_tags |
| 179 | Behavior_HoldFireOff_Lurker_quick | queued, unit_tags |
| 180 | Behavior_HoldFireOn_quick | queued, unit_tags |
| 181 | Behavior_HoldFireOn_Ghost_quick | queued, unit_tags |
| 182 | Behavior_HoldFireOn_Lurker_quick | queued, unit_tags |
| 183 | Build_Armory_pt | queued, unit_tags, world |
| 184 | Build_BanelingNest_pt | queued, unit_tags, world |
| 185 | Build_Barracks_pt | queued, unit_tags, world |
| 186 | Build_Bunker_pt | queued, unit_tags, world |
| 187 | Build_CommandCenter_pt | queued, unit_tags, world |
| 188 | Build_CreepTumor_pt | queued, unit_tags, world |
| 189 | Build_CreepTumor_Queen_pt | queued, unit_tags, world |
| 190 | Build_CreepTumor_Tumor_pt | queued, unit_tags, world |
| 191 | Build_EngineeringBay_pt | queued, unit_tags, world |
| 192 | Build_EvolutionChamber_pt | queued, unit_tags, world |
| 193 | Build_Extractor_unit | queued, unit_tags, target_unit_tag |
| 194 | Build_Factory_pt | queued, unit_tags, world |
| 195 | Build_FusionCore_pt | queued, unit_tags, world |
| 196 | Build_GhostAcademy_pt | queued, unit_tags, world |
| 197 | Build_Hatchery_pt | queued, unit_tags, world |
| 198 | Build_HydraliskDen_pt | queued, unit_tags, world |
| 199 | Build_InfestationPit_pt | queued, unit_tags, world |
| 200 | Build_Interceptors_autocast | unit_tags |
| 201 | Build_LurkerDen_pt | queued, unit_tags, world |
| 202 | Build_MissileTurret_pt | queued, unit_tags, world |
| 203 | Build_Nuke_quick | queued, unit_tags |
| 204 | Build_NydusNetwork_pt | queued, unit_tags, world |
| 205 | Build_NydusWorm_pt | queued, unit_tags, world |
| 206 | Build_Reactor_quick | queued, unit_tags |
| 207 | Build_Reactor_pt | queued, unit_tags, world |
| 208 | Build_Reactor_Barracks_quick | queued, unit_tags |
| 209 | Build_Reactor_Barracks_pt | queued, unit_tags, world |
| 210 | Build_Reactor_Factory_quick | queued, unit_tags |
| 211 | Build_Reactor_Factory_pt | queued, unit_tags, world |
| 212 | Build_Reactor_Starport_quick | queued, unit_tags |
| 213 | Build_Reactor_Starport_pt | queued, unit_tags, world |
| 214 | Build_Refinery_pt | queued, unit_tags, target_unit_tag |
| 215 | Build_RoachWarren_pt | queued, unit_tags, world |
| 216 | Build_SensorTower_pt | queued, unit_tags, world |
| 217 | Build_SpawningPool_pt | queued, unit_tags, world |
| 218 | Build_SpineCrawler_pt | queued, unit_tags, world |
| 219 | Build_Spire_pt | queued, unit_tags, world |
| 220 | Build_SporeCrawler_pt | queued, unit_tags, world |
| 221 | Build_Starport_pt | queued, unit_tags, world |
| 222 | Build_SupplyDepot_pt | queued, unit_tags, world |
| 223 | Build_TechLab_quick | queued, unit_tags |
| 224 | Build_TechLab_pt | queued, unit_tags, world |
| 225 | Build_TechLab_Barracks_quick | queued, unit_tags |
| 226 | Build_TechLab_Barracks_pt | queued, unit_tags, world |
| 227 | Build_TechLab_Factory_quick | queued, unit_tags |
| 228 | Build_TechLab_Factory_pt | queued, unit_tags, world |
| 229 | Build_TechLab_Starport_quick | queued, unit_tags |
| 230 | Build_TechLab_Starport_pt | queued, unit_tags, world |
| 231 | Build_UltraliskCavern_pt | queued, unit_tags, world |
| 232 | BurrowDown_quick | queued, unit_tags |
| 233 | BurrowDown_Baneling_quick | queued, unit_tags |
| 234 | BurrowDown_Drone_quick | queued, unit_tags |
| 235 | BurrowDown_Hydralisk_quick | queued, unit_tags |
| 236 | BurrowDown_Infestor_quick | queued, unit_tags |
| 237 | BurrowDown_InfestorTerran_quick | queued, unit_tags |
| 238 | BurrowDown_Lurker_quick | queued, unit_tags |
| 239 | BurrowDown_Queen_quick | queued, unit_tags |
| 240 | BurrowDown_Ravager_quick | queued, unit_tags |
| 241 | BurrowDown_Roach_quick | queued, unit_tags |
| 242 | BurrowDown_SwarmHost_quick | queued, unit_tags |
| 243 | BurrowDown_Ultralisk_quick | queued, unit_tags |
| 244 | BurrowDown_WidowMine_quick | queued, unit_tags |
| 245 | BurrowDown_Zergling_quick | queued, unit_tags |
| 246 | BurrowUp_quick | queued, unit_tags |
| 247 | BurrowUp_autocast | unit_tags |
| 248 | BurrowUp_Baneling_quick | queued, unit_tags |
| 249 | BurrowUp_Baneling_autocast | unit_tags |
| 250 | BurrowUp_Drone_quick | queued, unit_tags |
| 251 | BurrowUp_Hydralisk_quick | queued, unit_tags |
| 252 | BurrowUp_Hydralisk_autocast | unit_tags |
| 253 | BurrowUp_Infestor_quick | queued, unit_tags |
| 254 | BurrowUp_InfestorTerran_quick | queued, unit_tags |
| 255 | BurrowUp_InfestorTerran_autocast | unit_tags |
| 256 | BurrowUp_Lurker_quick | queued, unit_tags |
| 257 | BurrowUp_Queen_quick | queued, unit_tags |
| 258 | BurrowUp_Queen_autocast | unit_tags |
| 259 | BurrowUp_Ravager_quick | queued, unit_tags |
| 260 | BurrowUp_Ravager_autocast | unit_tags |
| 261 | BurrowUp_Roach_quick | queued, unit_tags |
| 262 | BurrowUp_Roach_autocast | unit_tags |
| 263 | BurrowUp_SwarmHost_quick | queued, unit_tags |
| 264 | BurrowUp_Ultralisk_quick | queued, unit_tags |
| 265 | BurrowUp_Ultralisk_autocast | unit_tags |
| 266 | BurrowUp_WidowMine_quick | queued, unit_tags |
| 267 | BurrowUp_Zergling_quick | queued, unit_tags |
| 268 | BurrowUp_Zergling_autocast | unit_tags |
| 269 | Cancel_BarracksAddOn_quick | queued, unit_tags |
| 270 | Cancel_CreepTumor_quick | queued, unit_tags |
| 271 | Cancel_FactoryAddOn_quick | queued, unit_tags |
| 272 | Cancel_HangarQueue5_quick | queued, unit_tags |
| 273 | Cancel_LockOn_quick | queued, unit_tags |
| 274 | Cancel_MorphBroodlord_quick | queued, unit_tags |
| 275 | Cancel_MorphGreaterSpire_quick | queued, unit_tags |
| 276 | Cancel_MorphHive_quick | queued, unit_tags |
| 277 | Cancel_MorphLair_quick | queued, unit_tags |
| 278 | Cancel_MorphLurker_quick | queued, unit_tags |
| 279 | Cancel_MorphLurkerDen_quick | queued, unit_tags |
| 280 | Cancel_MorphMothership_quick | queued, unit_tags |
| 281 | Cancel_MorphOrbital_quick | queued, unit_tags |
| 282 | Cancel_MorphOverlordTransport_quick | queued, unit_tags |
| 283 | Cancel_MorphOverseer_quick | queued, unit_tags |
| 284 | Cancel_MorphPlanetaryFortress_quick | queued, unit_tags |
| 285 | Cancel_MorphRavager_quick | queued, unit_tags |
| 286 | Cancel_MorphThorExplosiveMode_quick | queued, unit_tags |
| 287 | Cancel_NeuralParasite_quick | queued, unit_tags |
| 288 | Cancel_Nuke_quick | queued, unit_tags |
| 289 | Cancel_QueueAddOn_quick | queued, unit_tags |
| 290 | Cancel_SpineCrawlerRoot_quick | queued, unit_tags |
| 291 | Cancel_SporeCrawlerRoot_quick | queued, unit_tags |
| 292 | Cancel_StarportAddOn_quick | queued, unit_tags |
| 293 | Effect_Abduct_unit | queued, unit_tags, target_unit_tag |
| 294 | Effect_AntiArmorMissile_unit | queued, unit_tags, target_unit_tag |
| 295 | Effect_AutoTurret_pt | queued, unit_tags, world |
| 296 | Effect_BlindingCloud_pt | queued, unit_tags, world |
| 297 | Effect_CalldownMULE_pt | queued, unit_tags, world |
| 298 | Effect_CalldownMULE_unit | queued, unit_tags, target_unit_tag |
| 299 | Effect_CausticSpray_unit | queued, unit_tags, target_unit_tag |
| 300 | Effect_Charge_pt | queued, unit_tags, world |
| 301 | Effect_Charge_unit | queued, unit_tags, target_unit_tag |
| 302 | Effect_Charge_autocast | unit_tags |
| 303 | Effect_Contaminate_unit | queued, unit_tags, target_unit_tag |
| 304 | Effect_CorrosiveBile_pt | queued, unit_tags, world |
| 305 | Effect_EMP_pt | queued, unit_tags, world |
| 306 | Effect_EMP_unit | queued, unit_tags, target_unit_tag |
| 307 | Effect_Explode_quick | queued, unit_tags |
| 308 | Effect_FungalGrowth_pt | queued, unit_tags, world |
| 309 | Effect_FungalGrowth_unit | queued, unit_tags, target_unit_tag |
| 310 | Effect_GhostSnipe_unit | queued, unit_tags, target_unit_tag |
| 311 | Effect_Heal_unit | queued, unit_tags, target_unit_tag |
| 312 | Effect_Heal_autocast | unit_tags |
| 313 | Effect_ImmortalBarrier_autocast | unit_tags |
| 314 | Effect_InfestedTerrans_pt | queued, unit_tags, world |
| 315 | Effect_InjectLarva_unit | queued, unit_tags, target_unit_tag |
| 316 | Effect_InterferenceMatrix_unit | queued, unit_tags, target_unit_tag |
| 317 | Effect_KD8Charge_pt | queued, unit_tags, world |
| 318 | Effect_LockOn_unit | queued, unit_tags, target_unit_tag |
| 319 | Effect_LocustSwoop_pt | queued, unit_tags, world |
| 320 | Effect_MedivacIgniteAfterburners_quick | queued, unit_tags |
| 321 | Effect_NeuralParasite_unit | queued, unit_tags, target_unit_tag |
| 322 | Effect_NukeCalldown_pt | queued, unit_tags, world |
| 323 | Effect_ParasiticBomb_unit | queued, unit_tags, target_unit_tag |
| 324 | Effect_Repair_autocast | unit_tags |
| 325 | Effect_Repair_Mule_unit | queued, unit_tags, target_unit_tag |
| 326 | Effect_Repair_Mule_autocast | unit_tags |
| 327 | Effect_Repair_RepairDrone_unit | queued, unit_tags, target_unit_tag |
| 328 | Effect_Repair_RepairDrone_autocast | unit_tags |
| 329 | Effect_Repair_SCV_unit | queued, unit_tags, target_unit_tag |
| 330 | Effect_Repair_SCV_autocast | unit_tags |
| 331 | Effect_Restore_autocast | unit_tags |
| 332 | Effect_Salvage_quick | queued, unit_tags |
| 333 | Effect_Scan_pt | queued, unit_tags, world |
| 334 | Effect_SpawnChangeling_quick | queued, unit_tags |
| 335 | Effect_SpawnLocusts_pt | queued, unit_tags, world |
| 336 | Effect_SpawnLocusts_unit | queued, unit_tags, target_unit_tag |
| 337 | Effect_Spray_pt | queued, unit_tags, world |
| 338 | Effect_Spray_Protoss_pt | queued, unit_tags, world |
| 339 | Effect_Spray_Terran_pt | queued, unit_tags, world |
| 340 | Effect_Spray_Zerg_pt | queued, unit_tags, world |
| 341 | Effect_Stim_quick | queued, unit_tags |
| 342 | Effect_Stim_Marauder_quick | queued, unit_tags |
| 343 | Effect_Stim_Marauder_Redirect_quick | queued, unit_tags |
| 344 | Effect_Stim_Marine_quick | queued, unit_tags |
| 345 | Effect_Stim_Marine_Redirect_quick | queued, unit_tags |
| 346 | Effect_SupplyDrop_unit | queued, unit_tags, target_unit_tag |
| 347 | Effect_TacticalJump_pt | queued, unit_tags, world |
| 348 | Effect_TimeWarp_pt | queued, unit_tags, world |
| 349 | Effect_Transfusion_unit | queued, unit_tags, target_unit_tag |
| 350 | Effect_ViperConsume_unit | queued, unit_tags, target_unit_tag |
| 351 | Effect_WidowMineAttack_pt | queued, unit_tags, world |
| 352 | Effect_WidowMineAttack_unit | queued, unit_tags, target_unit_tag |
| 353 | Effect_WidowMineAttack_autocast | unit_tags |
| 354 | Halt_Building_quick | queued, unit_tags |
| 355 | Halt_TerranBuild_quick | queued, unit_tags |
| 356 | Harvest_Gather_Drone_unit | queued, unit_tags, target_unit_tag |
| 357 | Harvest_Gather_Mule_unit | queued, unit_tags, target_unit_tag |
| 358 | Harvest_Gather_Probe_unit | queued, unit_tags, target_unit_tag |
| 359 | Harvest_Gather_SCV_unit | queued, unit_tags, target_unit_tag |
| 360 | Harvest_Return_Drone_quick | queued, unit_tags |
| 361 | Harvest_Return_Mule_quick | queued, unit_tags |
| 362 | Harvest_Return_SCV_quick | queued, unit_tags |
| 363 | Land_pt | queued, unit_tags, world |
| 364 | Land_Barracks_pt | queued, unit_tags, world |
| 365 | Land_CommandCenter_pt | queued, unit_tags, world |
| 366 | Land_Factory_pt | queued, unit_tags, world |
| 367 | Land_OrbitalCommand_pt | queued, unit_tags, world |
| 368 | Land_Starport_pt | queued, unit_tags, world |
| 369 | Lift_quick | queued, unit_tags |
| 370 | Lift_Barracks_quick | queued, unit_tags |
| 371 | Lift_CommandCenter_quick | queued, unit_tags |
| 372 | Lift_Factory_quick | queued, unit_tags |
| 373 | Lift_OrbitalCommand_quick | queued, unit_tags |
| 374 | Lift_Starport_quick | queued, unit_tags |
| 375 | LoadAll_quick | queued, unit_tags |
| 376 | LoadAll_CommandCenter_quick | queued, unit_tags |
| 377 | Load_Bunker_unit | queued, unit_tags, target_unit_tag |
| 378 | Load_Medivac_unit | queued, unit_tags, target_unit_tag |
| 379 | Load_NydusNetwork_unit | queued, unit_tags, target_unit_tag |
| 380 | Load_NydusWorm_unit | queued, unit_tags, target_unit_tag |
| 381 | Load_Overlord_unit | queued, unit_tags, target_unit_tag |
| 382 | Load_WarpPrism_unit | queued, unit_tags, target_unit_tag |
| 383 | Morph_BroodLord_quick | queued, unit_tags |
| 384 | Morph_GreaterSpire_quick | queued, unit_tags |
| 385 | Morph_Hellbat_quick | queued, unit_tags |
| 386 | Morph_Hellion_quick | queued, unit_tags |
| 387 | Morph_Hive_quick | queued, unit_tags |
| 388 | Morph_Lair_quick | queued, unit_tags |
| 389 | Morph_LiberatorAAMode_quick | queued, unit_tags |
| 390 | Morph_LiberatorAGMode_pt | queued, unit_tags, world |
| 391 | Morph_Lurker_quick | queued, unit_tags |
| 392 | Morph_LurkerDen_quick | queued, unit_tags |
| 393 | Morph_Mothership_quick | queued, unit_tags |
| 394 | Morph_OrbitalCommand_quick | queued, unit_tags |
| 395 | Morph_OverlordTransport_quick | queued, unit_tags |
| 396 | Morph_Overseer_quick | queued, unit_tags |
| 397 | Morph_OverseerMode_quick | queued, unit_tags |
| 398 | Morph_OversightMode_quick | queued, unit_tags |
| 399 | Morph_PlanetaryFortress_quick | queued, unit_tags |
| 400 | Morph_Ravager_quick | queued, unit_tags |
| 401 | Morph_Root_pt | queued, unit_tags, world |
| 402 | Morph_SiegeMode_quick | queued, unit_tags |
| 403 | Morph_SpineCrawlerRoot_pt | queued, unit_tags, world |
| 404 | Morph_SpineCrawlerUproot_quick | queued, unit_tags |
| 405 | Morph_SporeCrawlerRoot_pt | queued, unit_tags, world |
| 406 | Morph_SporeCrawlerUproot_quick | queued, unit_tags |
| 407 | Morph_SupplyDepot_Lower_quick | queued, unit_tags |
| 408 | Morph_SupplyDepot_Raise_quick | queued, unit_tags |
| 409 | Morph_ThorExplosiveMode_quick | queued, unit_tags |
| 410 | Morph_ThorHighImpactMode_quick | queued, unit_tags |
| 411 | Morph_Unsiege_quick | queued, unit_tags |
| 412 | Morph_Uproot_quick | queued, unit_tags |
| 413 | Morph_VikingAssaultMode_quick | queued, unit_tags |
| 414 | Morph_VikingFighterMode_quick | queued, unit_tags |
| 415 | Rally_Building_pt | queued, unit_tags, world |
| 416 | Rally_Building_unit | queued, unit_tags, target_unit_tag |
| 417 | Rally_CommandCenter_pt | queued, unit_tags, world |
| 418 | Rally_CommandCenter_unit | queued, unit_tags, target_unit_tag |
| 419 | Rally_Hatchery_Units_pt | queued, unit_tags, world |
| 420 | Rally_Hatchery_Units_unit | queued, unit_tags, target_unit_tag |
| 421 | Rally_Hatchery_Workers_pt | queued, unit_tags, world |
| 422 | Rally_Hatchery_Workers_unit | queued, unit_tags, target_unit_tag |
| 423 | Rally_Morphing_Unit_pt | queued, unit_tags, world |
| 424 | Rally_Morphing_Unit_unit | queued, unit_tags, target_unit_tag |
| 425 | Research_AdaptiveTalons_quick | queued, unit_tags |
| 426 | Research_AdvancedBallistics_quick | queued, unit_tags |
| 427 | Research_BansheeCloakingField_quick | queued, unit_tags |
| 428 | Research_BansheeHyperflightRotors_quick | queued, unit_tags |
| 429 | Research_BattlecruiserWeaponRefit_quick | queued, unit_tags |
| 430 | Research_Burrow_quick | queued, unit_tags |
| 431 | Research_CentrifugalHooks_quick | queued, unit_tags |
| 432 | Research_ChitinousPlating_quick | queued, unit_tags |
| 433 | Research_CombatShield_quick | queued, unit_tags |
| 434 | Research_ConcussiveShells_quick | queued, unit_tags |
| 435 | Research_CycloneRapidFireLaunchers_quick | queued, unit_tags |
| 436 | Research_DrillingClaws_quick | queued, unit_tags |
| 437 | Research_GlialRegeneration_quick | queued, unit_tags |
| 438 | Research_GroovedSpines_quick | queued, unit_tags |
| 439 | Research_HiSecAutoTracking_quick | queued, unit_tags |
| 440 | Research_HighCapacityFuelTanks_quick | queued, unit_tags |
| 441 | Research_InfernalPreigniter_quick | queued, unit_tags |
| 442 | Research_MuscularAugments_quick | queued, unit_tags |
| 443 | Research_NeosteelFrame_quick | queued, unit_tags |
| 444 | Research_NeuralParasite_quick | queued, unit_tags |
| 445 | Research_PathogenGlands_quick | queued, unit_tags |
| 446 | Research_PersonalCloaking_quick | queued, unit_tags |
| 447 | Research_PneumatizedCarapace_quick | queued, unit_tags |
| 448 | Research_RavenCorvidReactor_quick | queued, unit_tags |
| 449 | Research_RavenRecalibratedExplosives_quick | queued, unit_tags |
| 450 | Research_SmartServos_quick | queued, unit_tags |
| 451 | Research_Stimpack_quick | queued, unit_tags |
| 452 | Research_TerranInfantryArmor_quick | queued, unit_tags |
| 453 | Research_TerranInfantryArmorLevel1_quick | queued, unit_tags |
| 454 | Research_TerranInfantryArmorLevel2_quick | queued, unit_tags |
| 455 | Research_TerranInfantryArmorLevel3_quick | queued, unit_tags |
| 456 | Research_TerranInfantryWeapons_quick | queued, unit_tags |
| 457 | Research_TerranInfantryWeaponsLevel1_quick | queued, unit_tags |
| 458 | Research_TerranInfantryWeaponsLevel2_quick | queued, unit_tags |
| 459 | Research_TerranInfantryWeaponsLevel3_quick | queued, unit_tags |
| 460 | Research_TerranShipWeapons_quick | queued, unit_tags |
| 461 | Research_TerranShipWeaponsLevel1_quick | queued, unit_tags |
| 462 | Research_TerranShipWeaponsLevel2_quick | queued, unit_tags |
| 463 | Research_TerranShipWeaponsLevel3_quick | queued, unit_tags |
| 464 | Research_TerranStructureArmorUpgrade_quick | queued, unit_tags |
| 465 | Research_TerranVehicleAndShipPlating_quick | queued, unit_tags |
| 466 | Research_TerranVehicleAndShipPlatingLevel1_quick | queued, unit_tags |
| 467 | Research_TerranVehicleAndShipPlatingLevel2_quick | queued, unit_tags |
| 468 | Research_TerranVehicleAndShipPlatingLevel3_quick | queued, unit_tags |
| 469 | Research_TerranVehicleWeapons_quick | queued, unit_tags |
| 470 | Research_TerranVehicleWeaponsLevel1_quick | queued, unit_tags |
| 471 | Research_TerranVehicleWeaponsLevel2_quick | queued, unit_tags |
| 472 | Research_TerranVehicleWeaponsLevel3_quick | queued, unit_tags |
| 473 | Research_TunnelingClaws_quick | queued, unit_tags |
| 474 | Research_ZergFlyerArmor_quick | queued, unit_tags |
| 475 | Research_ZergFlyerArmorLevel1_quick | queued, unit_tags |
| 476 | Research_ZergFlyerArmorLevel2_quick | queued, unit_tags |
| 477 | Research_ZergFlyerArmorLevel3_quick | queued, unit_tags |
| 478 | Research_ZergFlyerAttack_quick | queued, unit_tags |
| 479 | Research_ZergFlyerAttackLevel1_quick | queued, unit_tags |
| 480 | Research_ZergFlyerAttackLevel2_quick | queued, unit_tags |
| 481 | Research_ZergFlyerAttackLevel3_quick | queued, unit_tags |
| 482 | Research_ZergGroundArmor_quick | queued, unit_tags |
| 483 | Research_ZergGroundArmorLevel1_quick | queued, unit_tags |
| 484 | Research_ZergGroundArmorLevel2_quick | queued, unit_tags |
| 485 | Research_ZergGroundArmorLevel3_quick | queued, unit_tags |
| 486 | Research_ZergMeleeWeapons_quick | queued, unit_tags |
| 487 | Research_ZergMeleeWeaponsLevel1_quick | queued, unit_tags |
| 488 | Research_ZergMeleeWeaponsLevel2_quick | queued, unit_tags |
| 489 | Research_ZergMeleeWeaponsLevel3_quick | queued, unit_tags |
| 490 | Research_ZergMissileWeapons_quick | queued, unit_tags |
| 491 | Research_ZergMissileWeaponsLevel1_quick | queued, unit_tags |
| 492 | Research_ZergMissileWeaponsLevel2_quick | queued, unit_tags |
| 493 | Research_ZergMissileWeaponsLevel3_quick | queued, unit_tags |
| 494 | Research_ZerglingAdrenalGlands_quick | queued, unit_tags |
| 495 | Research_ZerglingMetabolicBoost_quick | queued, unit_tags |
| 496 | Stop_Building_quick | queued, unit_tags |
| 497 | Stop_Redirect_quick | queued, unit_tags |
| 498 | Train_Baneling_quick | queued, unit_tags |
| 499 | Train_Banshee_quick | queued, unit_tags |
| 500 | Train_Battlecruiser_quick | queued, unit_tags |
| 501 | Train_Corruptor_quick | queued, unit_tags |
| 502 | Train_Cyclone_quick | queued, unit_tags |
| 503 | Train_Drone_quick | queued, unit_tags |
| 504 | Train_Ghost_quick | queued, unit_tags |
| 505 | Train_Hellbat_quick | queued, unit_tags |
| 506 | Train_Hellion_quick | queued, unit_tags |
| 507 | Train_Hydralisk_quick | queued, unit_tags |
| 508 | Train_Infestor_quick | queued, unit_tags |
| 509 | Train_Liberator_quick | queued, unit_tags |
| 510 | Train_Marauder_quick | queued, unit_tags |
| 511 | Train_Marine_quick | queued, unit_tags |
| 512 | Train_Medivac_quick | queued, unit_tags |
| 513 | Train_MothershipCore_quick | queued, unit_tags |
| 514 | Train_Mutalisk_quick | queued, unit_tags |
| 515 | Train_Overlord_quick | queued, unit_tags |
| 516 | Train_Queen_quick | queued, unit_tags |
| 517 | Train_Raven_quick | queued, unit_tags |
| 518 | Train_Reaper_quick | queued, unit_tags |
| 519 | Train_Roach_quick | queued, unit_tags |
| 520 | Train_SCV_quick | queued, unit_tags |
| 521 | Train_SiegeTank_quick | queued, unit_tags |
| 522 | Train_SwarmHost_quick | queued, unit_tags |
| 523 | Train_Thor_quick | queued, unit_tags |
| 524 | Train_Ultralisk_quick | queued, unit_tags |
| 525 | Train_VikingFighter_quick | queued, unit_tags |
| 526 | Train_Viper_quick | queued, unit_tags |
| 527 | Train_WidowMine_quick | queued, unit_tags |
| 528 | Train_Zergling_quick | queued, unit_tags |
| 529 | UnloadAllAt_Medivac_pt | queued, unit_tags, world |
| 530 | UnloadAllAt_Medivac_unit | queued, unit_tags, target_unit_tag |
| 531 | UnloadAllAt_Overlord_pt | queued, unit_tags, world |
| 532 | UnloadAllAt_Overlord_unit | queued, unit_tags, target_unit_tag |
| 533 | UnloadAll_Bunker_quick | queued, unit_tags |
| 534 | UnloadAll_CommandCenter_quick | queued, unit_tags |
| 535 | UnloadAll_NydusNetwork_quick | queued, unit_tags |
| 536 | UnloadAll_NydusWorm_quick | queued, unit_tags |
| 537 | Effect_YamatoGun_unit | queued, unit_tags, target_unit_tag |
| 538 | Effect_KD8Charge_unit | queued, unit_tags, target_unit_tag |
| 539 | Attack_Battlecruiser_pt | queued, unit_tags, world |
| 540 | Attack_Battlecruiser_unit | queued, unit_tags, target_unit_tag |
| 541 | Effect_LockOn_autocast | unit_tags |
| 542 | HoldPosition_Battlecruiser_quick | queued, unit_tags |
| 543 | HoldPosition_Hold_quick | queued, unit_tags |
| 544 | Morph_WarpGate_autocast | unit_tags |
| 545 | Move_Battlecruiser_pt | queued, unit_tags, world |
| 546 | Move_Battlecruiser_unit | queued, unit_tags, target_unit_tag |
| 547 | Move_Move_pt | queued, unit_tags, world |
| 548 | Move_Move_unit | queued, unit_tags, target_unit_tag |
| 549 | Patrol_Battlecruiser_pt | queued, unit_tags, world |
| 550 | Patrol_Battlecruiser_unit | queued, unit_tags, target_unit_tag |
| 551 | Patrol_Patrol_pt | queued, unit_tags, world |
| 552 | Patrol_Patrol_unit | queued, unit_tags, target_unit_tag |
| 553 | Research_AnabolicSynthesis_quick | queued, unit_tags |
| 554 | Research_CycloneLockOnDamage_quick | queued, unit_tags |
| 555 | Stop_Battlecruiser_quick | queued, unit_tags |
| 556 | UnloadUnit_quick | queued, unit_tags |
| 557 | UnloadUnit_Bunker_quick | queued, unit_tags |
| 558 | UnloadUnit_CommandCenter_quick | queued, unit_tags |
| 559 | UnloadUnit_Medivac_quick | queued, unit_tags |
| 560 | UnloadUnit_NydusNetwork_quick | queued, unit_tags |
| 561 | UnloadUnit_Overlord_quick | queued, unit_tags |
| 562 | UnloadUnit_WarpPrism_quick | queued, unit_tags |
| 563 | Research_EnhancedShockwaves_quick | queued, unit_tags |
||||



