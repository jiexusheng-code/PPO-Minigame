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

- [ ] 分离两个 TensorBoard 子目录写入语义

  - [ ] **新增两个 TensorBoard 子目录**

    - 在输出目录下创建 `tb_logs/fundamental`（以 optimization-step 为 x 轴）和 `tb_logs/extra`（以 optimization-step 为 x 轴）。
- [ ] **实现 env-step 写入逻辑**
  
  - 使用 SB3 的 `num_timesteps` 或 `model.num_timesteps` 作为 `env-step` 的 global_step，写入训练中按时间自然产生的标量（policy loss、value loss、总体 entropy、lr、eval mean reward 等）。
  - [ ] **实现 optimization-step 写入逻辑**

    - 定义并维护 `opt_step` 计数器（将其与对照项目的 global_step 等价，建议在每次完整的 PPO 更新后自增一次）。
  - 在每次优化结束时把细粒度指标（按参数类型的 entropy、per-arg used-rate、old/new logprob 统计、被剪切/非法动作比例、mini-batch 平均 loss 等）写入 `extra`（或 `fundamental` 中的额外分组），x 轴统一使用 `opt_step`（optimization-step）。
  - [x] **策略/分布层导出指标**
  
  - 在 `MaskedFlattenPolicy` 中暴露 / 缓存 per-arg 统计（entropy、used mask 均值、被剪切计数等），供 Callback 在写入时读取并聚合。
  - [x] **实现 Callback 与写入聚合**

    - 新增自定义 SB3 Callback：在 `on_rollout_end` 聚合本轮的训练标量并写入 `env-step`（global_step=num_timesteps）和 `optimization-step`（global_step=opt_step）。对 per-arg 指标做 batch 平均后写入，减少 I/O。

      - 已实现：

        - 在 `MaskedFlattenPolicy` 中缓存 `self._last_per_arg_stats`（fn_entropy_mean / slot_used_mean / slot_entropy_mean）。
        - 新增 `src/callbacks/tb_dual_writer.py` 实现写入到 `fundamental` 与 `extra` 两个子目录，并已在 `train.py` 中接入 CallbackList。
  - [x] **性能与稳定性调优**

  - 控制写入频率（例如每次 update 或每 K 次 update 写一次），避免过多小文件；仅由主进程写 TensorBoard 日志。

    - 已实现：`TBDualWriterCallback` 增加 `write_env_every` 与 `write_opt_every` 参数，且在写入前检查常见分布式环境变量（`RANK`/`OMPI_COMM_WORLD_RANK`/`LOCAL_RANK`/`SLURM_PROCID`），仅在主进程写入。默认频率为 1（每次写入）。

## 验证

- [x] smoke test

  - 说明：已运行短时 smoke test（`total_timesteps=512`）。结果：`models/<ts>/tb_logs/fundamental` 含 SB3/训练主要标量写入的事件文件（x 轴统一以 optimization-step 为准），`models/<ts>/tb_logs/extra` 含 callback 写入的 per-arg/优化级别事件文件，显示 per-arg 指标已被写入。

- [x] 支持持续训练直到 Ctrl+C（通过 `total_timesteps` 控制）

  - 已实现：不使用独立的 `run_forever` 配置键；当 `total_timesteps` 在配置中设为 <= 0 时，训练会进入持续运行模式（循环调用 `model.learn(..., reset_num_timesteps=False)`），直到手动中断（Ctrl+C），随后自动保存模型和配置。
