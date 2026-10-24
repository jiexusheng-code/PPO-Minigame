# 训练诊断与复盘

诊断只记录数据，不改变策略输入、动作、奖励或优化目标。本文不包含运行身份、Git 版本或依赖清单记录。

## 下次需要一起下载什么

下载同一次运行的完整 `tb_logs/`、`config_used.yaml`、`logs/train.log` 和 `evaluations.npz`。其中：

- `tb_logs/PPO_<n>/`：SB3 的 PPO 损失、KL、裁剪比例、评估分数等。
- `tb_logs/extra/`：项目补充的函数频率、非法动作、critic 数值和内部特征诊断。
- `tb_logs/diagnostics/`：周期抽样的实际策略输入、训练目标与模型内部数值，供逐样本复盘。
- `tb_logs/trajectories/`：CollectMineralShards 环境 0 抽样的完整回合，关联动作、执行前后观测及单位状态。

不要只截一张 `explained_variance` 曲线：单个标量无法区分预测不变、目标变化很小、偏置过大和梯度失效。

## 推荐排查顺序

1. **先看任务表现**：评估单局分数及最近多次评估趋势；单次最高分不能代表稳定水平。
2. **看更新是否异常**：NaN/Inf、KL、clip fraction、value loss；跨不同动作参数化版本不要直接比较熵绝对值。
3. **看 critic 在区分局面吗**：同一 rollout 的 value 与 return 标准差、误差、相关性；return 是 GAE 训练目标，不是整局分数。
4. **看信息在哪层消失**：视觉投影归一化前后尺度、各层跨样本变化、Tanh 饱和比例、value-only 梯度。
5. **回看现场**：用抽样文件检查同一条观测、value、return 和 advantage；用轨迹检查选择状态、目标位置与单位行动。

## 证据的解释范围

- rollout 统计与抽样内部数值在当前轮 PPO 更新之前采集，横轴是总环境步数。
- critic 每次内部前向最多使用 32 条观测；首个完整 rollout及此后每累计至少 50,000 环境步保存一次样本与 value-only 梯度证据。轨迹沿用 env 0 的完整回合周期抽样，不保存全部环境的全部帧。
- 抽样 value-only 梯度与 PPO 实际更新的总梯度不同：它不包含 actor/entropy loss，也没有全局梯度裁剪。它用来检查价值损失能否传回指定层。
- 周期抽样无法完整重建所有训练 minibatch；未抽到的瞬时异常可能需要结合连续标量与日志定位。
- 单位行号是当前观测中的行位置，不是稳定单位身份；只有真实可用且有效的单位 tag 才能用于跨帧关联。
- wrapper 的 `executed` 表示提交的动作并获得 step 返回；只有原始观测确实提供的引擎结果字段才能作为额外证据，缺失字段不能解释为成功。
- feature units、选择面板和额外栅格仅用于诊断，不作为新增策略输入。

诊断文件只使用数值数组和 JSON 数据；读取 NumPy 抽样文件时使用 `np.load(path, allow_pickle=False)`。文件字段、采样周期和指标名称以 [TensorBoard 参考](tensorboard.md) 与实现为准。
