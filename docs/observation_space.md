# Observation space in PYSC2

本文旨在尽可能详细地介绍 `observation` 观测空间。

> 记号说明：
> - `H`/`W` 表示高度/宽度（shape 采用 `(y, x)` 排列），见第3节详细解释
> - `N` 表示可变长度，含义见每个字段的说明
> - 若未注明 dtype，默认为 `np.int32`（详见 `transform_obs()`）

---

# 1. 观测空间概览与获取方式

## 1.1 观测空间由什么决定

`observation` 的字段集合与 shape 由配置环境时的 `AgentInterfaceFormat` 参数影响，在一局游戏中，`AgentInterfaceFormat`不发生变化，故观测空间的字段等信息不发生变化。

## 1.2 观测获取与字段访问示例

下面示例演示：创建环境、获取 `observation`、读取通用字段与条件字段。

```python
from pysc2.env import sc2_env

aif = sc2_env.AgentInterfaceFormat(
	feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
	use_feature_units=True,
	use_raw_units=True,
	use_unit_counts=True,
	use_camera_position=True,
)

env = sc2_env.SC2Env(
	map_name="MoveToBeacon",
	players=[sc2_env.Agent(sc2_env.Race.terran)],
	agent_interface_format=aif,
	step_mul=8,
)

try:
	timesteps = env.reset()
	obs = timesteps[0].observation

	# 通用字段
	player = obs["player"]
	score = obs["score_cumulative"]

	# 条件字段（需对应配置为 True）
	feature_screen = obs.get("feature_screen")
	raw_units = obs.get("raw_units")
	unit_counts = obs.get("unit_counts")
	camera_position = obs.get("camera_position")
finally:
	env.close()
```

如果只想打印观测空间结构，可以直接调用 `env.observation_spec()`。

# 2. `observation` 字段详解

## 2.1 通用字段（默认总是存在）

### 2.1.1 `action_result`
- **shape**：`(N,)`
- **dtype**：`np.int32`
- **N 含义**：本步返回的动作结果数量（通常为 0 或与上一步动作数相关）。
- **含义**：动作执行结果，来源 `obs.action_errors`。
- **注意**：`observation["action_result"]` 仅包含每条错误的整数码（`result` 字段）。要把整数码映射为可读名称，请使用 `s2clientprotocol.error_pb2.ActionResult`。完整映射表见文档末尾的**附录 A**。

### 2.1.2 `alerts`
- **shape**：`(N,)`
- **dtype**：`np.int32`
- **N 含义**：当前触发的告警数量。
- **含义**：告警列表，来源 `obs.observation.alerts`。完整映射表见文档末尾的**附录B**。

### 2.1.3 `build_queue`
- **shape**：`(N, len(UnitLayer))`，`len(UnitLayer)=7`
- **dtype**：`np.int32`
- **N 含义**：生产/建造队列中的单位数量。
- **含义**：生产/建造队列中的单位（UI 数据）。
- **字段含义**：见文档末尾 **附录 C（UnitLayer 列索引）**。

### 2.1.4 `cargo`
- **shape**：`(N, len(UnitLayer))`
- **dtype**：`np.int32`
- **N 含义**：当前选中运输单位的载荷数量。
- **含义**：运输单位的乘客列表（UI 数据）。
- **字段含义**：见文档末尾 **附录 C（UnitLayer 列索引）**。

### 2.1.5 `cargo_slots_available`
- **shape**：`(1,)`
- **dtype**：`np.int32`
- **含义**：可用载荷位数量。

### 2.1.6 `control_groups`
- **shape**：`(10, 2)`
- **dtype**：`np.int32`
- **含义**：控制组信息 `(leader_unit_type, count)`。
- **维度含义**：
	- 第 1 维：控制组编号（0–9）。
	- 第 2 维：
		- 索引 0：`leader_unit_type`（领队单位类型 id）。
		- 索引 1：`count`（该控制组单位数量）。
- **备注**：`leader_unit_type` 的 id ↔ 名称映射方法见文档末尾 **附录 D**。

### 2.1.7 `game_loop`
- **shape**：`(1,)`
- **dtype**：`np.int32`
- **含义**：当前游戏步数（game loop）。

### 2.1.8 `last_actions`
- **shape**：`(N,)`
- **dtype**：`np.int32`
- **N 含义**：上一帧执行的动作数量。
- **含义**：上一帧执行的动作函数 id（非 RAW 模式），具体映射表见action_space.md。

### 2.1.9 `map_name`
- **dtype**：`str`（字符串）
- **含义**：地图名称，来源于 `Features` 初始化参数（例如在创建环境时传入的 `map_name`）。
- **说明**：在 `observation_spec()` 中该字段有时以占位形式显示（如 `(0,)`），但实际值为字符串；直接读取 `obs.get('map_name')` 返回字符串或 None。

### 2.1.10 `multi_select`
- **shape**：`(N, len(UnitLayer))`
- **dtype**：`np.int32`
- **N 含义**：多选单位数量。
- **含义**：多选单位列表（UI 数据）。
- **字段含义**：见文档末尾 **附录 C（UnitLayer 列索引）**。

### 2.1.11 `player`
- **shape**：`(len(Player),)`，`len(Player)=11`
- **dtype**：`np.int32`
- **含义**：玩家通用信息数组。
- **字段含义**：见**附录E**。

### 2.1.12 `production_queue`
- **shape**：`(N, len(ProductionQueue))`，`len(ProductionQueue)=2`
- **dtype**：`np.int32`
- **N 含义**：生产队列条目数量。
- **含义**：生产队列条目 `(ability_id, build_progress)`，进度是 `0-100`。
### 2.1.12 `production_queue`
- **shape**：`(N, len(ProductionQueue))`，`len(ProductionQueue)=2`
- **dtype**：`np.int32`
- **N 含义**：生产队列条目数量。
- **含义**：生产队列条目 `(ability_id, build_progress)`，进度是 `0-100`。
- **维度逐项说明：**
	- 索引 0：`ability_id`（能力 id，整型）
		- 含义：表示该队列条目对应的技能/建造/生产能力的协议 id（SC2 ability id）。
		- 备注：`ability_id` 使用 SC2 协议定义的能力 id 空间，此处只会出现部分ability_id，想要知道其含义可以查看action_space.md。
	- 索引 1：`build_progress`（建造进度，0-100）
		- 含义：该条目当前的完成百分比（整数，0 表示刚开始，100 表示已完成）。

### 2.1.13 `score_cumulative`
- **shape**：`(len(ScoreCumulative),)`，`len(ScoreCumulative)=13`
- **dtype**：`np.int32`
- **含义**：累计分数明细。
- 详细索引说明见文档末尾 **附录 F（ScoreCumulative 索引说明）**。

### 2.1.14 `score_by_category`
- **shape**：`(len(ScoreByCategory), len(ScoreCategories))`
- **dtype**：`np.int32`
- **含义**：按类别分数明细。
- **维度逐项说明：**见**附录G**。

### 2.1.15 `score_by_vital`
- **shape**：`(len(ScoreByVital), len(ScoreVitals))`
- **dtype**：`np.int32`
- **含义**：按生命值/护盾/能量分项分数。
- **维度逐项说明：**见**附录H**。

### 2.1.16 `single_select`
- **shape**：`(N, len(UnitLayer))`
- **dtype**：`np.int32`
- **N 含义**：单选单位数量（通常为 0 或 1）。
- **含义**：当前单选单位（UI 数据）。
- **字段含义**：见文档末尾 **附录 C（UnitLayer 列索引）**。

### 2.1.17 `upgrades`
- **shape**：`(N,)`
- **dtype**：`np.int32`
- **N 含义**：已完成科技升级数量。
- **含义**：已完成科技升级 id 列表。
- **取值含义**：见文档末尾**附录I**。

### 2.1.18 `home_race_requested`
- **shape**：`(1,)`
- **dtype**：`np.int32`
- **含义**：当前环境中“主（home）玩家”请求的种族编码。
- **使用注意**：观测中该字段可能以长度为 1 的数组或类似对象返回；请以稳健方式转换为整数后再做映射（详见附录 J）。

### 2.1.19 `away_race_requested`
- **shape**：`(1,)`
- **dtype**：`np.int32`
- **含义**：当前环境中“客（away）玩家”请求的种族编码。
- **使用注意**：同上——该字段通常为长度为 1 的数组类型，建议先将其转换为 Python 整数再做名称映射。

## 2.2 条件字段：动作空间相关

### 2.2.1 `available_actions`
- **shape**：`(N,)`
- **dtype**：`np.int32`
- **N 含义**：当前可执行动作函数 id 数量。
- **含义**：可执行动作函数 id 列表。详见action_space.md。
- **出现条件**：未启用 RAW 动作接口，即 `use_raw_actions=False`。

**对应配置示例：**

```python
from pysc2.env import sc2_env

aif = sc2_env.AgentInterfaceFormat(
	feature_dimensions=sc2_env.Dimensions(64, 64),
	use_raw_actions=False,
)
```

## 2.3 条件字段：特征层观测

### 2.3.1 `feature_screen`
- **shape**：`(len(SCREEN_FEATURES), H, W)`
- **dtype**：`np.int32`
- **含义**：屏幕特征层堆叠（y,x 顺序）。
- **出现条件**：`feature_dimensions` 非空。
- **详细说明**：见 **附录 K（SCREEN_FEATURES 列表）**

### 2.3.2 `feature_minimap`
- **shape**：`(len(MINIMAP_FEATURES), H, W)`
- **dtype**：`np.int32`
- **含义**：小地图特征层堆叠（y,x 顺序）。
- **出现条件**：`feature_dimensions` 非空。
- **详细说明**：见 **附录 L（MINIMAP_FEATURES 列表）**

**对应配置示例：**

```python
from pysc2.env import sc2_env

aif = sc2_env.AgentInterfaceFormat(
	feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64)
)
```

### 2.3.3 特征层读取示例

```python
from pysc2.lib import features

screen = obs["feature_screen"]
minimap = obs["feature_minimap"]

# 通过枚举索引取某一层
unit_type_layer = screen[features.ScreenFeatures.unit_type]
visibility_layer = minimap[features.MinimapFeatures.visibility_map]
```

## 2.4 条件字段：RGB 观测

### 2.4.1 `rgb_screen`
- **shape**：`(H, W, 3)`
- **dtype**：`np.int32`（由 `Feature.unpack_rgb_image` 转为 int32）
- **含义**：主屏 RGB 图像。
- **出现条件**：`rgb_dimensions` 非空。

### 2.4.2 `rgb_minimap`
- **shape**：`(H, W, 3)`
- **dtype**：`np.int32`
- **含义**：小地图 RGB 图像。
- **出现条件**：`rgb_dimensions` 非空。

**对应配置示例：**

```python
from pysc2.env import sc2_env

aif = sc2_env.AgentInterfaceFormat(
	rgb_dimensions=sc2_env.Dimensions(screen=128, minimap=128)
)
```

## 2.5 条件字段：单位列表（Feature / Raw）

### 2.5.1 `feature_units`
- **shape**：`(N, len(FeatureUnit))`，`len(FeatureUnit)=46`
- **dtype**：`np.int64`
- **N 含义**：屏幕内单位数量。
- **含义**：屏幕内单位表（坐标为屏幕坐标）。
- **出现条件**：`use_feature_units=True`。
- **详细说明**：见**附录M**。

### 2.5.2 `feature_effects`
- **shape**：`(N, len(EffectPos))`，`len(EffectPos)=6`
- **dtype**：`np.int32`
- **N 含义**：屏幕内效果数量。
- **含义**：屏幕内效果位置列表。
- **出现条件**：`use_feature_units=True`。
- **详细说明**：见**附录N**。

### 2.5.3 `raw_units`
- **shape**：`(N, len(FeatureUnit))`
- **dtype**：`np.int64`
- **N 含义**：全图单位数量。
- **含义**：全图单位表（坐标为 raw/minimap 坐标）。
- **出现条件**：`use_raw_units=True`。
- **详细说明**：见**附录M**。

### 2.5.4 `raw_effects`
- **shape**：`(N, len(EffectPos))`
- **dtype**：`np.int32`
- **N 含义**：全图效果数量。
- **含义**：全图效果位置列表。
- **出现条件**：`use_raw_units=True`。
- **详细说明**：见**附录N**。

### 2.5.5 `radar`
- **shape**：`(N, len(Radar))`，`len(Radar)=3`
- **dtype**：`np.int32`
- **N 含义**：侦测塔数量。
- **含义**：侦测塔信息（位置、半径）。
- **出现条件**：`use_feature_units=True` 或 `use_raw_units=True`。
- **详细说明**：见**附录O**。

### 2.5.6 `unit_counts`
- **shape**：`(N, len(UnitCounts))`，`len(UnitCounts)=2`
- **dtype**：`np.int32`
- **N 含义**：自方单位类型数量（去重后的类型数）。
- **含义**：自方单位类型统计 `(unit_type, count)`。
- **出现条件**：`use_unit_counts=True`。
- **详细说明**：unit_type见**附录D**。

**对应配置示例：**

```python
from pysc2.env import sc2_env

aif = sc2_env.AgentInterfaceFormat(
	feature_dimensions=sc2_env.Dimensions(64, 64),
	use_feature_units=True,
	use_raw_units=True,
	use_unit_counts=True,
)
```

## 2.6 条件字段：相机信息

### 2.6.1 `camera_position`
- **shape**：`(2,)`
- **dtype**：`np.int32`
- **含义**：相机中心位置（minimap 坐标）。
- **出现条件**：`use_camera_position=True`。

### 2.6.2 `camera_size`
- **shape**：`(2,)`
- **dtype**：`np.int32`
- **含义**：相机视野大小（minimap 坐标单位）。
- **出现条件**：`use_camera_position=True`。

**对应配置示例：**

```python
from pysc2.env import sc2_env

aif = sc2_env.AgentInterfaceFormat(
	feature_dimensions=sc2_env.Dimensions(64, 64),
	use_camera_position=True,
)
```

## 2.7 条件字段：原始 proto

### 2.7.1 `_response_observation`
- **dtype**：`sc_pb.ResponseObservation`（s2clientprotocol 的 proto 对象，通常通过惰性函数/包装器暴露）
- **含义**：原始 SC2 observation protocol buffer，包含底层观测 proto 信息，供需要访问原始协议数据的调试或高级用途使用。
- **说明**：该字段不是数组类型；在 `observation_spec()` 中可能以占位形式出现。要启用并获取原始 proto，请在 `AgentInterfaceFormat` 中设置 `send_observation_proto=True`，并直接读取 `obs['_response_observation']`（可能是一个惰性加载的对象或函数）。

**对应配置示例：**

```python
from pysc2.env import sc2_env

aif = sc2_env.AgentInterfaceFormat(
	feature_dimensions=sc2_env.Dimensions(64, 64),
	send_observation_proto=True,
)
```

# 3. 坐标与索引约定（(y, x) 与 (x, y)）

在使用 `feature_screen` / `feature_minimap` 等图像型特征层时，数组的索引顺序为 `(y, x)`（行, 列），也就是 `array[row, col]`。但是在构造动作（如点击屏幕、移动到某点）时，许多动作接口期望坐标以 `(x, y)` 的形式给出（先 x 再 y）。下面给出常见情况、要点与示例。

- 规则要点：
  - 特征层数组索引：`feature_screen[layer, y, x]`。
  - `FeatureUnit` 表中单元坐标以 `(x, y)` 分别存储在列索引 `12`（x）和 `13`（y）。
  - 构造动作时通常使用 `(x, y)`（例如 `FUNCTIONS.Move_screen`、`FUNCTIONS.Attack_screen` 的目标参数）。
  - 要从数组索引得到动作坐标，需要交换索引顺序：`(x, y) = (col, row)`。

示例 1：在屏幕特征层中找到“自己”的一个像素位置并构造移动/攻击动作参数

```python
import numpy as np
from pysc2.lib import features, actions

# 假设 obs 为当前帧的 observation
screen = obs['feature_screen']  # shape = (num_layers, H, W)
player_rel = screen[features.ScreenFeatures.player_relative]

# 找到标记为 SELF 的像素位置（可能有多处）
ys, xs = np.where(player_rel == features.PlayerRelative.SELF)
if len(ys) > 0:
	y0, x0 = int(ys[0]), int(xs[0])  # 数组索引 (row, col)

	# 构造动作参数时坐标需为 (x, y)
	target = [x0, y0]
	act = actions.FunctionCall(actions.FUNCTIONS.Move_screen.id, [[0], target])
	# env.step([act])
```

示例 2：读取 `feature_units` 表中第 0 个单位的坐标并在数组上访问该像素

```python
units = obs.get('feature_units')  # shape = (N, len(FeatureUnit))
if units is not None and len(units) > 0:
	unit0 = units[0]
	ux = int(unit0[12])  # FeatureUnit.x 列
	uy = int(unit0[13])  # FeatureUnit.y 列

	# 在屏幕数组上读取该位置的某层值（注意索引顺序）
	if 'feature_screen' in obs:
		layer_val = obs['feature_screen'][features.ScreenFeatures.unit_type, uy, ux]

	# 若要对该位置下达动作，按 (x, y) 传参
	act = actions.FunctionCall(actions.FUNCTIONS.Attack_screen.id, [[0], [ux, uy]])
	# env.step([act])
```

示例 3：判断 `raw_units` / `feature_units` 单位是否在屏幕上

```python
# FeatureUnit 的第 36 列为 is_on_screen
if units is not None:
	on_screen_mask = units[:, 36] == 1
	on_screen_units = units[on_screen_mask]
	# on_screen_units 中的 x,y 可直接用于屏幕动作参数

# raw_units 包含全图坐标，通常需要基于相机/尺度关系映射到屏幕坐标；
# 如果只是判断是否在屏幕内，优先使用 feature_units 的 is_on_screen 字段。
```

注意：不同观测字段使用的坐标系可能不同（`feature_units` 为屏幕坐标，`raw_units` 为全图/原始坐标），因此在对坐标做跨字段映射时要确认坐标来源与尺度。简单且可靠的方法是：当可用时，使用 `feature_units` 的 `is_on_screen` 与其 x/y 直接作为动作目标；若只使用 `raw_units`，则需要基于相机与像素/游戏坐标比例进行映射或借助环境提供的工具函数。

# 附录 A：`ActionResult` 数值 → 名称 映射表

说明：下表来自 `s2clientprotocol.error_pb2.ActionResult` 枚举（可通过下列代码枚举或映射）：

```python
from s2clientprotocol import error_pb2 as sc_err
for v in sc_err.ActionResult.DESCRIPTOR.values:
	print(v.number, v.name)
```

| 数值 | 名称 |
|---:|---|
| 1 | Success |
| 2 | NotSupported |
| 3 | Error |
| 4 | CantQueueThatOrder |
| 5 | Retry |
| 6 | Cooldown |
| 7 | QueueIsFull |
| 8 | RallyQueueIsFull |
| 9 | NotEnoughMinerals |
| 10 | NotEnoughVespene |
| 11 | NotEnoughTerrazine |
| 12 | NotEnoughCustom |
| 13 | NotEnoughFood |
| 14 | FoodUsageImpossible |
| 15 | NotEnoughLife |
| 16 | NotEnoughShields |
| 17 | NotEnoughEnergy |
| 18 | LifeSuppressed |
| 19 | ShieldsSuppressed |
| 20 | EnergySuppressed |
| 21 | NotEnoughCharges |
| 22 | CantAddMoreCharges |
| 23 | TooMuchMinerals |
| 24 | TooMuchVespene |
| 25 | TooMuchTerrazine |
| 26 | TooMuchCustom |
| 27 | TooMuchFood |
| 28 | TooMuchLife |
| 29 | TooMuchShields |
| 30 | TooMuchEnergy |
| 31 | MustTargetUnitWithLife |
| 32 | MustTargetUnitWithShields |
| 33 | MustTargetUnitWithEnergy |
| 34 | CantTrade |
| 35 | CantSpend |
| 36 | CantTargetThatUnit |
| 37 | CouldntAllocateUnit |
| 38 | UnitCantMove |
| 39 | TransportIsHoldingPosition |
| 40 | BuildTechRequirementsNotMet |
| 41 | CantFindPlacementLocation |
| 42 | CantBuildOnThat |
| 43 | CantBuildTooCloseToDropOff |
| 44 | CantBuildLocationInvalid |
| 45 | CantSeeBuildLocation |
| 46 | CantBuildTooCloseToCreepSource |
| 47 | CantBuildTooCloseToResources |
| 48 | CantBuildTooFarFromWater |
| 49 | CantBuildTooFarFromCreepSource |
| 50 | CantBuildTooFarFromBuildPowerSource |
| 51 | CantBuildOnDenseTerrain |
| 52 | CantTrainTooFarFromTrainPowerSource |
| 53 | CantLandLocationInvalid |
| 54 | CantSeeLandLocation |
| 55 | CantLandTooCloseToCreepSource |
| 56 | CantLandTooCloseToResources |
| 57 | CantLandTooFarFromWater |
| 58 | CantLandTooFarFromCreepSource |
| 59 | CantLandTooFarFromBuildPowerSource |
| 60 | CantLandTooFarFromTrainPowerSource |
| 61 | CantLandOnDenseTerrain |
| 62 | AddOnTooFarFromBuilding |
| 63 | MustBuildRefineryFirst |
| 64 | BuildingIsUnderConstruction |
| 65 | CantFindDropOff |
| 66 | CantLoadOtherPlayersUnits |
| 67 | NotEnoughRoomToLoadUnit |
| 68 | CantUnloadUnitsThere |
| 69 | CantWarpInUnitsThere |
| 70 | CantLoadImmobileUnits |
| 71 | CantRechargeImmobileUnits |
| 72 | CantRechargeUnderConstructionUnits |
| 73 | CantLoadThatUnit |
| 74 | NoCargoToUnload |
| 75 | LoadAllNoTargetsFound |
| 76 | NotWhileOccupied |
| 77 | CantAttackWithoutAmmo |
| 78 | CantHoldAnyMoreAmmo |
| 79 | TechRequirementsNotMet |
| 80 | MustLockdownUnitFirst |
| 81 | MustTargetUnit |
| 82 | MustTargetInventory |
| 83 | MustTargetVisibleUnit |
| 84 | MustTargetVisibleLocation |
| 85 | MustTargetWalkableLocation |
| 86 | MustTargetPawnableUnit |
| 87 | YouCantControlThatUnit |
| 88 | YouCantIssueCommandsToThatUnit |
| 89 | MustTargetResources |
| 90 | RequiresHealTarget |
| 91 | RequiresRepairTarget |
| 92 | NoItemsToDrop |
| 93 | CantHoldAnyMoreItems |
| 94 | CantHoldThat |
| 95 | TargetHasNoInventory |
| 96 | CantDropThisItem |
| 97 | CantMoveThisItem |
| 98 | CantPawnThisUnit |
| 99 | MustTargetCaster |
| 100 | CantTargetCaster |
| 101 | MustTargetOuter |
| 102 | CantTargetOuter |
| 103 | MustTargetYourOwnUnits |
| 104 | CantTargetYourOwnUnits |
| 105 | MustTargetFriendlyUnits |
| 106 | CantTargetFriendlyUnits |
| 107 | MustTargetNeutralUnits |
| 108 | CantTargetNeutralUnits |
| 109 | MustTargetEnemyUnits |
| 110 | CantTargetEnemyUnits |
| 111 | MustTargetAirUnits |
| 112 | CantTargetAirUnits |
| 113 | MustTargetGroundUnits |
| 114 | CantTargetGroundUnits |
| 115 | MustTargetStructures |
| 116 | CantTargetStructures |
| 117 | MustTargetLightUnits |
| 118 | CantTargetLightUnits |
| 119 | MustTargetArmoredUnits |
| 120 | CantTargetArmoredUnits |
| 121 | MustTargetBiologicalUnits |
| 122 | CantTargetBiologicalUnits |
| 123 | MustTargetHeroicUnits |
| 124 | CantTargetHeroicUnits |
| 125 | MustTargetRoboticUnits |
| 126 | CantTargetRoboticUnits |
| 127 | MustTargetMechanicalUnits |
| 128 | CantTargetMechanicalUnits |
| 129 | MustTargetPsionicUnits |
| 130 | CantTargetPsionicUnits |
| 131 | MustTargetMassiveUnits |
| 132 | CantTargetMassiveUnits |
| 133 | MustTargetMissile |
| 134 | CantTargetMissile |
| 135 | MustTargetWorkerUnits |
| 136 | CantTargetWorkerUnits |
| 137 | MustTargetEnergyCapableUnits |
| 138 | CantTargetEnergyCapableUnits |
| 139 | MustTargetShieldCapableUnits |
| 140 | CantTargetShieldCapableUnits |
| 141 | MustTargetFlyers |
| 142 | CantTargetFlyers |
| 143 | MustTargetBuriedUnits |
| 144 | CantTargetBuriedUnits |
| 145 | MustTargetCloakedUnits |
| 146 | CantTargetCloakedUnits |
| 147 | MustTargetUnitsInAStasisField |
| 148 | CantTargetUnitsInAStasisField |
| 149 | MustTargetUnderConstructionUnits |
| 150 | CantTargetUnderConstructionUnits |
| 151 | MustTargetDeadUnits |
| 152 | CantTargetDeadUnits |
| 153 | MustTargetRevivableUnits |
| 154 | CantTargetRevivableUnits |
| 155 | MustTargetHiddenUnits |
| 156 | CantTargetHiddenUnits |
| 157 | CantRechargeOtherPlayersUnits |
| 158 | MustTargetHallucinations |
| 159 | CantTargetHallucinations |
| 160 | MustTargetInvulnerableUnits |
| 161 | CantTargetInvulnerableUnits |
| 162 | MustTargetDetectedUnits |
| 163 | CantTargetDetectedUnits |
| 164 | CantTargetUnitWithEnergy |
| 165 | CantTargetUnitWithShields |
| 166 | MustTargetUncommandableUnits |
| 167 | CantTargetUncommandableUnits |
| 168 | MustTargetPreventDefeatUnits |
| 169 | CantTargetPreventDefeatUnits |
| 170 | MustTargetPreventRevealUnits |
| 171 | CantTargetPreventRevealUnits |
| 172 | MustTargetPassiveUnits |
| 173 | CantTargetPassiveUnits |
| 174 | MustTargetStunnedUnits |
| 175 | CantTargetStunnedUnits |
| 176 | MustTargetSummonedUnits |
| 177 | CantTargetSummonedUnits |
| 178 | MustTargetUser1 |
| 179 | CantTargetUser1 |
| 180 | MustTargetUnstoppableUnits |
| 181 | CantTargetUnstoppableUnits |
| 182 | MustTargetResistantUnits |
| 183 | CantTargetResistantUnits |
| 184 | MustTargetDazedUnits |
| 185 | CantTargetDazedUnits |
| 186 | CantLockdown |
| 187 | CantMindControl |
| 188 | MustTargetDestructibles |
| 189 | CantTargetDestructibles |
| 190 | MustTargetItems |
| 191 | CantTargetItems |
| 192 | NoCalldownAvailable |
| 193 | WaypointListFull |
| 194 | MustTargetRace |
| 195 | CantTargetRace |
| 196 | MustTargetSimilarUnits |
| 197 | CantTargetSimilarUnits |
| 198 | CantFindEnoughTargets |
| 199 | AlreadySpawningLarva |
| 200 | CantTargetExhaustedResources |
| 201 | CantUseMinimap |
| 202 | CantUseInfoPanel |
| 203 | OrderQueueIsFull |
| 204 | CantHarvestThatResource |
| 205 | HarvestersNotRequired |
| 206 | AlreadyTargeted |

---

# 附录 B：`Alert` 数值 → 名称 映射表

说明：下表来自 `s2clientprotocol.sc2api_pb2.Alert` 枚举（可通过下列代码枚举或映射）：

```python
from s2clientprotocol import sc2api_pb2 as sc_pb
for v in sc_pb.Alert.DESCRIPTOR.values:
	print(v.number, v.name)
```

| 数值 | 名称 |
|---:|---|
| 1 | NuclearLaunchDetected |
| 2 | NydusWormDetected |
| 3 | AlertError |
| 4 | AddOnComplete |
| 5 | BuildingComplete |
| 6 | BuildingUnderAttack |
| 7 | LarvaHatched |
| 8 | MergeComplete |
| 9 | MineralsExhausted |
| 10 | MorphComplete |
| 11 | MothershipComplete |
| 12 | MULEExpired |
| 13 | NukeComplete |
| 14 | ResearchComplete |
| 15 | TrainError |
| 16 | TrainUnitComplete |
| 17 | TrainWorkerComplete |
| 18 | TransformationComplete |
| 19 | UnitUnderAttack |
| 20 | UpgradeComplete |
| 21 | VespeneExhausted |
| 22 | WarpInComplete |

---

# 附录 C：`UnitLayer` 列索引

说明：`UnitLayer` 定义了 UI 中与单个单位相关的短向量（例如 `single_select` / `multi_select` / `build_queue` / `cargo` 的列）。下面按列顺序列出字段名与含义：

| 索引 | 字段名 | 含义 |
|---:|---|---|
| 0 | unit_type | 单位类型 id |
| 1 | player_relative | 相对阵营 |
| 2 | health | 生命值 |
| 3 | shields | 护盾值 |
| 4 | energy | 能量值 |
| 5 | transport_slots_taken | 占用载荷位 |
| 6 | build_progress | 建造进度（0-100，整数） |

（以上顺序与 `pysc2/pysc2/lib/features.py` 中的 `UnitLayer` 枚举一致。）

## 索引方法（示例）

下面给出两种访问 `UnitLayer` 列的常用方法：

- 方法一：通过枚举索引（推荐）

	使用 `pysc2.lib.features.UnitLayer` 中定义的枚举常量，这样代码更具可读性且不易出错：

	```python
	from pysc2.lib import features

	# 假设 obs 中有 single_select，并取第 0 个条目
	unit_vec = obs['single_select'][0]

	unit_type = unit_vec[features.UnitLayer.unit_type]
	build_progress = unit_vec[features.UnitLayer.build_progress]
	```

- 方法二：直接使用整数列索引

	直接使用表中给出的列索引（在某些简单脚本或快速原型中可能更方便）：

	```python
	# 同样假设 unit_vec 已从 obs['single_select'][0] 获取
	unit_type = unit_vec[0]
	build_progress = unit_vec[6]
	```

	注意：直接使用整数索引可行但可读性差，且当源码或枚举调整时更易出错，因此通常建议使用枚举常量。

---

# 附录 D： id ↔ 单位名称 对照（生成方法）

说明：`leader_unit_type` 与 `unit_type` 使用同一套单位类型 id。该映射由 SC2 的 `RequestData` 返回，随游戏版本变化，无法在静态源码中保证完整一致。推荐在实际运行环境中按以下方法生成并保存对照表：

```python
# 生成 leader_unit_type / unit_type 的 id -> name 映射（取样输出以验证）
from absl import flags
import pysc2.run_configs
try:
    flags.FLAGS(['notebook', '--sc2_run_config=Windows'])
except Exception:
    pass
try:
    from pysc2.env import sc2_env
    aif = sc2_env.AgentInterfaceFormat(
        feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
    )
    env = sc2_env.SC2Env(
        map_name="MoveToBeacon",
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=aif,
        step_mul=8,
        game_steps_per_episode=0,
        visualize=False,
    )
    try:
        data = env._controllers[0].data_raw()  # ResponseData
        unit_id_to_name = {u.unit_id: u.name for u in data.units}
        print("unit_id_to_name size:", len(unit_id_to_name))
        # 打印前 20 个（按 id 排序）
        for unit_id in sorted(unit_id_to_name)[:20]:
            print(unit_id, unit_id_to_name[unit_id])
    finally:
        try:
            env.close()
        except Exception:
            pass
except Exception as e:
    print("Failed to generate unit_id_to_name mapping:", e)
    import traceback
    traceback.print_exc()
```

---

# 附录 E：`Player` 字段索引

`Player` 枚举决定了 `player` 数组每个位置的含义：

| 索引 | 字段名 | 含义 |
|---:|---|---|
| 0 | player_id | 玩家 id |
| 1 | minerals | 矿物 |
| 2 | vespene | 瓦斯 |
| 3 | food_used | 已用人口 |
| 4 | food_cap | 人口上限 |
| 5 | food_army | 军队人口 |
| 6 | food_workers | 工人人口 |
| 7 | idle_worker_count | 空闲工人数量 |
| 8 | army_count | 军队单位数量 |
| 9 | warp_gate_count | 折跃门数量 |
| 10 | larva_count | 幼虫数量 |

## 索引方法（示例）

- 方法一：通过枚举索引（推荐）

	```python
	from pysc2.lib import features

	player = obs['player']
	minerals = player[features.Player.minerals]
	food_cap = player[features.Player.food_cap]
	```

- 方法二：直接使用整数索引

	```python
	player = obs['player']
	minerals = player[1]
	food_cap = player[4]
	```

	注意：直接使用整数索引可行但可读性差，且当源码或枚举调整时更易出错，因此通常建议使用枚举常量。

---

# 附录 F：`ScoreCumulative`（`score_cumulative` 索引说明）

下面列出 `score_cumulative` 数组每个索引的含义。

- 索引 0：`score` — 整体/总分值（可以作为胜负/表现的高层指标）。
- 索引 1：`idle_production_time` — 生产设施空闲时间（累计，表示建筑未产出期间的空闲 ticks）。
- 索引 2：`idle_worker_time` — 空闲采集者时间（累计，表示农民/采集者未工作的时间）。
- 索引 3：`total_value_units` — 单位总价值（所有单位当前价值之和）。
- 索引 4：`total_value_structures` — 建筑总价值（所有建筑当前价值之和）。
- 索引 5：`killed_value_units` — 击杀单位获得的价值（对手单位被击杀累计获得的价值）。
- 索引 6：`killed_value_structures` — 击毁建筑获得的价值（对手建筑被摧毁累计获得的价值）。
- 索引 7：`collected_minerals` — 已采集的矿物总量。
- 索引 8：`collected_vespene` — 已采集的瓦斯总量。
- 索引 9：`collection_rate_minerals` — 矿物采集速率（单位/时间窗口，具体缩放依版本可能不同）。
- 索引 10：`collection_rate_vespene` — 瓦斯采集速率（单位/时间窗口）。
- 索引 11：`spent_minerals` — 已花费的矿物总量。
- 索引 12：`spent_vespene` — 已花费的瓦斯总量。

示例：访问与打印

```python
sc = obs['score_cumulative']
print('total_score:', int(sc[0]))
print('collected_minerals:', int(sc[7]))
```

---

# 附录 G：`ScoreByCategory`（`score_by_category` 索引说明）

`score_by_category` 的 shape 为 `(len(ScoreByCategory), len(ScoreCategories))`，即第一维是下面的 `ScoreByCategory` 项目（行），第二维是 `ScoreCategories`（列）。

**ScoreByCategory（第一维，行）索引表**

| 索引 | 字段名 | 含义 |
|---:|---|---|
| 0 | `food_used` | 使用的人口数 |
| 1 | `killed_minerals` | 击杀单位/结构获得的矿物价值 |
| 2 | `killed_vespene` | 击杀单位/结构获得的瓦斯价值 |
| 3 | `lost_minerals` | 自身单位/结构损失造成的矿物损失 |
| 4 | `lost_vespene` | 自身单位/结构损失造成的瓦斯损失 |
| 5 | `friendly_fire_minerals` | 友军误伤消耗的矿物价值 |
| 6 | `friendly_fire_vespene` | 友军误伤消耗的瓦斯价值 |
| 7 | `used_minerals` | 总计使用的矿物 |
| 8 | `used_vespene` | 总计使用的瓦斯 |
| 9 | `total_used_minerals` | （聚合/历史）已使用矿物总量 |
| 10 | `total_used_vespene` | （聚合/历史）已使用瓦斯总量 |

**ScoreCategories（第二维，列）索引表**

| 索引 | 字段名 | 含义 |
|---:|---|---|
| 0 | `none` | 无特定类别（通用/未分类） |
| 1 | `army` | 与军队作战/单位相关的分项 |
| 2 | `economy` | 经济相关（采集/资源/开销） |
| 3 | `technology` | 技术/科技相关得分 |
| 4 | `upgrade` | 升级相关得分 |

示例：读取“军队类别下的击杀矿物值”

```python
sb = obs['score_by_category']  # shape = (11, 5)
# 直接使用整数索引
val = int(sb[1, 1])  # killed_minerals (行1) 在 army (列1) 下的值

# 或使用枚举（推荐，可读性更好）
from pysc2.lib import features
val2 = int(sb[features.ScoreByCategory.killed_minerals,
			   features.ScoreCategories.army])
print('killed_minerals (army):', val2)
```

---

# 附录 H：`ScoreByVital`（`score_by_vital` 索引说明）

`score_by_vital` 的 shape 为 `(len(ScoreByVital), len(ScoreVitals))`，即第一维是下面的 `ScoreByVital` 项目（行），第二维是 `ScoreVitals`（列）。

**ScoreByVital（第一维，行）索引表**

| 索引 | 字段名 | 含义 |
|---:|---|---|
| 0 | `total_damage_dealt` | 累计造成的总伤害 |
| 1 | `total_damage_taken` | 累计承受的总伤害 |
| 2 | `total_healed` | 累计治愈/恢复的总量 |

**ScoreVitals（第二维，列）索引表**

| 索引 | 字段名 | 含义 |
|---:|---|---|
| 0 | `life` | 与生命值相关的分项 |
| 1 | `shields` | 与护盾相关的分项 |
| 2 | `energy` | 与能量/法力相关的分项 |

示例：读取“生命（life）维度下的总造成伤害”

```python
sbv = obs['score_by_vital']  # shape = (3, 3)
# 直接使用整数索引
val = int(sbv[0, 0])  # total_damage_dealt (行0) 在 life (列0) 下的值

# 或使用枚举（推荐，可读性更好）
from pysc2.lib import features
val2 = int(sbv[features.ScoreByVital.total_damage_dealt,
			   features.ScoreVitals.life])
print('total_damage_dealt (life):', val2)
```

---

# 附录 I：`upgrades`（科技升级 id → 名称 映射 / 生成方法）

说明：SC2 的升级（upgrade）id 随游戏版本变化，不保证在静态源码中始终稳定。建议在运行时从游戏返回的 `RequestData` / `ResponseData` 中生成对照表并保存，以确保与当前运行时版本一致。

下面提供一个可直接在有 SC2 与 `pysc2` 的运行环境中执行的示例脚本，用于生成 `upgrade_id -> name` 的映射并将其保存为 CSV：

```python
from absl import flags
import pysc2.run_configs
try:
    flags.FLAGS(['notebook', '--sc2_run_config=Windows'])
except Exception:
    pass
try:
    from pysc2.env import sc2_env
    aif = sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64))
    env = sc2_env.SC2Env(map_name="MoveToBeacon", players=[sc2_env.Agent(sc2_env.Race.terran)], agent_interface_format=aif, step_mul=8, game_steps_per_episode=0, visualize=False)
    try:
        data = env._controllers[0].data_raw()  # ResponseData
        # 有些版本字段名为 'upgrades' 或 'upgrades_data'，兼容处理
        upgrades = getattr(data, 'upgrades', None) or getattr(data, 'upgrades_data', None) or []
        upgrade_map = {u.upgrade_id: u.name for u in upgrades}
        print('sampled upgrades count:', len(upgrade_map))
        for uid in sorted(upgrade_map)[:50]:
            print(uid, upgrade_map[uid])
    finally:
        try:
            env.close()
        except Exception:
            pass
except Exception as e:
    print('Failed to sample upgrades mapping:', e)
    import traceback; traceback.print_exc()
```

使用说明：
- 生成脚本将在本地与当前 SC2 版本通信以获得升级表，得到的 `upgrade_id` 即可用于与观测字段 `upgrades`（第 2.1.17 节）中的 id 列表做匹配。示例：

```python
# 检查某个升级 id 是否已完成
upgrades = obs.get('upgrades', [])
is_done = int(42) in upgrades  # 假设 42 是某条升级 id
```

---

# 附录 J：`Race` 数值 → 名称 对照表（常见）

说明：不同版本的 `s2clientprotocol` 在 proto 中对枚举的定义通常保持一致，但在极少数版本或自定义构建中可能不同。建议在正式使用前通过运行时从 `s2clientprotocol.common_pb2.Race.DESCRIPTOR.values` 或 `s2clientprotocol.sc2api_pb2` 的相应枚举处读取确认。下面为常见映射，适用于大多数标准发行的 `s2clientprotocol`：

| 数值 | 名称 |
|---:|---|
| 0 | Unknown/None |
| 1 | Terran |
| 2 | Zerg |
| 3 | Protoss |
| 4 | Random |

提示：若需要在代码中映射数值为名称，请在运行时枚举而不要硬编码，以防协议版本差异。例如：

```python
from s2clientprotocol import common_pb2 as sc_common
race_map = {v.number: v.name for v in sc_common.Race.DESCRIPTOR.values}
```

以上静态表仅作文档参考；实际项目中请以运行时枚举结果为准。

---

# 附录 K：`SCREEN_FEATURES` 层级列表与取值范围

说明：下面列表来源于 `pysc2/pysc2/lib/features.py`（不同 SC2 版本可能略有差别），`scale` 的含义为“该层最大值 + 1”。若某层在当前 SC2 版本未实现，环境会返回全零。

- `height_map`：**scale=256**，**类型**=SCALAR。地形高度。
- `visibility_map`：**scale=4**，**类型**=CATEGORICAL。可见性（隐藏/已见/可见等）。
- `creep`：**scale=2**，**类型**=CATEGORICAL。菌毯分布。
- `power`：**scale=2**，**类型**=CATEGORICAL。水晶塔能量覆盖。
- `player_id`：**scale=17**，**类型**=CATEGORICAL。绝对玩家编号。
- `player_relative`：**scale=5**，**类型**=CATEGORICAL。相对阵营（自己/盟友/中立/敌人）。
- `unit_type`：**scale=max(UNIT_TYPES)+1**，**类型**=CATEGORICAL。单位类型 id。
- `selected`：**scale=2**，**类型**=CATEGORICAL。是否选中。
- `unit_hit_points`：**scale=1600**，**类型**=SCALAR。生命值。
- `unit_hit_points_ratio`：**scale=256**，**类型**=SCALAR。生命值比例（0-255）。
- `unit_energy`：**scale=1000**，**类型**=SCALAR。能量值。
- `unit_energy_ratio`：**scale=256**，**类型**=SCALAR。能量比例（0-255）。
- `unit_shields`：**scale=1000**，**类型**=SCALAR。护盾值。
- `unit_shields_ratio`：**scale=256**，**类型**=SCALAR。护盾比例（0-255）。
- `unit_density`：**scale=16**，**类型**=SCALAR。单位密度（聚集程度）。
- `unit_density_aa`：**scale=256**，**类型**=SCALAR。对空密度。
- `effects`：**scale=16**，**类型**=CATEGORICAL。地面效果（如灵能风暴）。
- `hallucinations`：**scale=2**，**类型**=CATEGORICAL。幻象标记。
- `cloaked`：**scale=2**，**类型**=CATEGORICAL。隐形标记。
- `blip`：**scale=2**，**类型**=CATEGORICAL。雷达虚影标记。
- `buffs`：**scale=max(BUFFS)+1**，**类型**=CATEGORICAL。增益/减益 id。
- `buff_duration`：**scale=256**，**类型**=SCALAR。增益剩余时间（0-255）。
- `active`：**scale=2**，**类型**=CATEGORICAL。活跃状态。
- `build_progress`：**scale=256**，**类型**=SCALAR。建造进度（0-255）。
- `pathable`：**scale=2**，**类型**=CATEGORICAL。是否可通行。
- `buildable`：**scale=2**，**类型**=CATEGORICAL。是否可建造。
- `placeholder`：**scale=2**，**类型**=CATEGORICAL。建造占位符。

# 附录 L：`MINIMAP_FEATURES` 层级列表与取值范围

说明：下面列表来源于 `pysc2/pysc2/lib/features.py`（不同 SC2 版本可能略有差别），同样遵循 `scale = 最大值 + 1` 的约定。

- `height_map`：**scale=256**，**类型**=SCALAR。地形高度。
- `visibility_map`：**scale=4**，**类型**=CATEGORICAL。可见性。
- `creep`：**scale=2**，**类型**=CATEGORICAL。菌毯分布。
- `camera`：**scale=2**，**类型**=CATEGORICAL。相机视野覆盖区域。
- `player_id`：**scale=17**，**类型**=CATEGORICAL。绝对玩家编号。
- `player_relative`：**scale=5**，**类型**=CATEGORICAL。相对阵营。
- `selected`：**scale=2**，**类型**=CATEGORICAL。是否选中。
- `unit_type`：**scale=max(UNIT_TYPES)+1**，**类型**=CATEGORICAL。单位类型 id。
- `alerts`：**scale=2**，**类型**=CATEGORICAL。告警标记。
- `pathable`：**scale=2**，**类型**=CATEGORICAL。是否可通行。
- `buildable`：**scale=2**，**类型**=CATEGORICAL。是否可建造。


---

# 附录 M：`FeatureUnit` 列语义（逐项）

以下顺序与 `FeatureUnit` 枚举一致（pysc2/pysc2/lib/features.py）。字段来源主要在 `full_unit_vec()`。

1. `unit_type`：单位类型 id。
2. `alliance`：阵营（Self=1, Ally=2, Neutral=3, Enemy=4）。
3. `health`：生命值。
4. `shield`：护盾值。
5. `energy`：能量值。
6. `cargo_space_taken`：已占用载荷空间。
7. `build_progress`：建造进度百分比（0-100，整数）。
8. `health_ratio`：生命比例（0-255，整数）。
9. `shield_ratio`：护盾比例（0-255，整数）。
10. `energy_ratio`：能量比例（0-255，整数）。
11. `display_type`：显示类型（可见/快照/隐藏，取值由协议定义）。
12. `owner`：所有者玩家 id（1-15，16=中立）。
13. `x`：坐标 x（feature_units 为屏幕坐标；raw_units 为 minimap/raw 坐标）。
14. `y`：坐标 y。
15. `facing`：朝向（弧度/方向值，协议定义）。
16. `radius`：单位半径（与坐标同单位）。
17. `cloak`：隐形状态（Cloaked=1, CloakedDetected=2, NotCloaked=3）。
18. `is_selected`：是否被选中（0/1）。
19. `is_blip`：是否雷达虚影（0/1）。
20. `is_powered`：是否有能量覆盖（0/1）。
21. `mineral_contents`：矿物剩余量（矿物单位）。
22. `vespene_contents`：瓦斯剩余量（瓦斯单位）。
23. `cargo_space_max`：最大载荷空间（敌军/中立通常为 0）。
24. `assigned_harvesters`：已分配采集者数量。
25. `ideal_harvesters`：理想采集者数量。
26. `weapon_cooldown`：武器冷却时间。
27. `order_length`：命令队列长度（0 表示空闲）。
28. `order_id_0`：命令 0（转为函数 id，未定义则 0）。
29. `order_id_1`：命令 1（转为函数 id）。
30. `tag`：单位唯一标识（仅 raw_units 填充，feature_units 为 0）。
31. `hallucination`：是否幻象（0/1）。
32. `buff_id_0`：buff id 0（不存在则 0）。
33. `buff_id_1`：buff id 1（不存在则 0）。
34. `addon_unit_type`：挂件/附属单位类型 id（无则 0）。
35. `active`：是否激活/工作中（0/1）。
36. `is_on_screen`：是否在屏幕内（0/1）。
37. `order_progress_0`：命令 0 进度（0-100）。
38. `order_progress_1`：命令 1 进度（0-100）。
39. `order_id_2`：命令 2（转为函数 id）。
40. `order_id_3`：命令 3（转为函数 id）。
41. `is_in_cargo`：是否在载具中（0/1）。
42. `buff_duration_remain`：buff 剩余时间。
43. `buff_duration_max`：buff 最大持续时间。
44. `attack_upgrade_level`：攻击升级等级。
45. `armor_upgrade_level`：护甲升级等级。
46. `shield_upgrade_level`：护盾升级等级。

> 说明：`order_id_*` 由 `RAW_ABILITY_ID_TO_FUNC_ID` 映射得到；若无法映射则为 0。

---

# 附录 N：`EffectPos`（`feature_effects` / `raw_effects` 列索引）

`EffectPos` 列索引按顺序为：

- `effect`：效果 id。
- `alliance`：阵营。
- `owner`：所有者。
- `radius`：半径。
- `x`：位置 x。
- `y`：位置 y。

说明：该表用于 `feature_effects` 与 `raw_effects` 的每一行表示一个效果实例的简短向量（位置、归属与半径等）。

---

# 附录 O：`radar` 列索引与取值范围

`radar` 字段的 shape 为 `(N, len(Radar))`，`len(Radar)=3`。每一行表示一个侦测塔/探测器实例，其列含义与典型取值如下：

- `x`：位置 x（整数）。
	- 说明：以与 `feature_units` / `raw_units` 使用的坐标系一致的游戏坐标或像素坐标（通常为 minimap/raw 坐标，取决于观测字段来源）。
	- 典型取值：0 ≤ x < 地图宽度（单位：像素或地图坐标）。

- `y`：位置 y（整数）。
	- 说明：与 `x` 同坐标系，表示垂直方向坐标。
	- 典型取值：0 ≤ y < 地图高度（单位：像素或地图坐标）。

- `radius`：侦测半径（整数）。
	- 说明：表示该侦测/影响区域的半径（以相同坐标系的单位衡量）。在某些环境或协议版本中，该值可能为 0（代表点状效果）或以像素/格为单位的整数。
	- 典型取值：非负整数；具体含义/尺度依地图与协议实现而异（建议运行时取样并验证）。

使用建议：`radar` 列的精确单位与量纲依环境/观测字段来源不同而变化。若需严格映射到屏幕像素或游戏逻辑坐标，请在运行时通过 `raw_units` / `feature_units` 的坐标与 `camera_position` 等信息进行对齐与验证。



