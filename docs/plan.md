# Plan

## Week 2

- [x] 完成状态空间的梳理
- [x] 完成动作空间的梳理
- [x] 完成PPO超参数含义的梳理
- [x] 完成tensorboard各图表的横纵坐标的含义的梳理

## Week 5

- [x] 更改动作处理

  - [x] **定义参数头规范**

  - [x] **实现参数掩码构造**

  - [x] **实现联合分布与 log_prob**

  - [x] **训练期忽略未用参数损失**

  - [x] **接入 SB3 策略兼容层**

- [ ] 分离 env-step / optimization-step 写入

  - [ ] **新增两个 TensorBoard 子目录**

    - 在输出目录下创建 `tb_logs/env-step`（以环境交互步为 x 轴）和 `tb_logs/optimization-step`（以优化步或 episode 为 x 轴）。
- [ ] **实现 env-step 写入逻辑**
  
  - 使用 SB3 的 `num_timesteps` 或 `model.num_timesteps` 作为 `env-step` 的 global_step，写入训练中按时间自然产生的标量（policy loss、value loss、总体 entropy、lr、eval mean reward 等）。
  - [ ] **实现 optimization-step 写入逻辑**

    - 定义并维护 `opt_step` 计数器（将其与对照项目的 global_step 等价，建议在每次完整的 PPO 更新后自增一次）。
  - 在每次优化结束时把细粒度指标（按参数类型的 entropy、per-arg used-rate、old/new logprob 统计、被剪切/非法动作比例、mini-batch 平均 loss 等）写入 `optimization-step`，x 轴使用 `opt_step` 或 episode 编号。
  - [ ] **策略/分布层导出指标**
  
  - 在 `MaskedFlattenPolicy` 中暴露 / 缓存 per-arg 统计（entropy、used mask 均值、被剪切计数等），供 Callback 在写入时读取并聚合。
  - [ ] **实现 Callback 与写入聚合**

    - 新增自定义 SB3 Callback：在 `on_rollout_end` 聚合本轮的训练标量并写入 `env-step`（global_step=num_timesteps）和 `optimization-step`（global_step=opt_step）。对 per-arg 指标做 batch 平均后写入，减少 I/O。
- [ ] **性能与稳定性调优**
  
  - 控制写入频率（例如每次 update 或每 K 次 update 写一次），避免过多小文件；仅由主进程写 TensorBoard 日志。
