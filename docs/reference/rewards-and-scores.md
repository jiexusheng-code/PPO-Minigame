# Rewards and scores

[TOC]

## Reward

### 影响因素

在创建sc2_env.SC2Env时，参数score_index和score_multiplier会影响reward的取值，如果不人为赋值，将分别使用每张地图的默认值。

### 生成机制

- 如果score_index <0:将以游戏的胜负作为标准。
  - 胜利：1*score_multiplier
  - 失败：-1*score_multiplier
  - 平局/还未决出胜负/本局游戏不涉及胜负概念：0
- 如果score_index = N ≥ 0：
  - 第一步的reward被强制设置为0.
  - 原始reward的值为当前步 score_cumulative[N] 减去上一步的score_cumulative[N]。比如说N = 0，则原始reward即为当前步的score减去上一步的score。
  - 最后返回的reward是原始reward乘以score_multiplier。



## Scores（ScoreCumulative、ScoreByCategory、ScoreByVital）

这三个字段的数据都是直接从SC2游戏引擎返回的，PYSC2只对其进行映射与命名，没有进行计算。在通讯协议s2client-proto中，可以看到有简单的文字描述，但是也没有具体展开score的计算公式，而SC2的游戏引擎代码是不公开的。



### 