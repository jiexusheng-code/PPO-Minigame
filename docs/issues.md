# 待修复问题清单

记录日期：2026-10-20。来源：项目迁移到云端后的环境核对与代码审阅。
状态说明：`P0` 阻塞正式训练，`P1` 影响训练效率或结果可信度，`P2` 代码健壮性/维护性。

> 2026-10-21 更新：P0-1、P0-2、P1-3、P1-4、P1-5、P2-6、P2-7 已修复并提交
> （commits 2a361e0 / 2d8ef0f / bbe1b52）；P2-11 的 `metadata` 已改，`render` 签名仍为旧格式。
> 剩余待办：P2-8（JointMaskedDistribution 接口补全）、P2-9（逐元素 CPU 同步）、
> P2-10（特征复用，框架限制）——均不阻塞训练，待首次正式实验后再评估。
>
> 2026-10-21 补充：验证了 `unset DISPLAY` 下 SubprocVecEnv 训练正常，SC2 以无头方式运行，
> 因此移除 `DISPLAY`/Xvfb 依赖：`scripts/env.sh` 只保留 `SC2PATH`，preflight 将 DISPLAY
> 缺失从 FAIL 降为 SKIP。此前“需要 X 显示”的假设不成立，属于多余依赖。

## P0：阻塞首次正式训练

### 1. ✅ 已修复：新 shell 缺少 `SC2PATH` 与 `DISPLAY`，preflight 直接失败

在云端新开终端执行 `tools/bitahub_preflight.py` 的结果：

```text
[WARN] SC2PATH: 未设置，将使用 PySC2 默认路径 /root/StarCraftII
[FAIL] SC2 目录: 不存在: /root/StarCraftII（需放置 SC2 本体或导出 SC2PATH）
[FAIL] DISPLAY: 未设置；SC2 需要 X 显示。
```

实际 SC2 在 `/mnt/ppo/StarCraftII`，Xvfb 已在 `:99` 运行，但二者都没有导出到 shell 环境。
每次新会话都必须手动 export，否则表现为“环境坏了”。

修复方向：

- 将 `export SC2PATH=/mnt/ppo/StarCraftII`、`export DISPLAY=:99` 写入 `~/.bashrc`，或提供 `scripts/env.sh` 统一 source；
- preflight 对“默认路径不存在但常见自定义路径存在”给出更明确的提示。

### 2. 首次 1M 步实验尚未执行，README 状态描述过期

`README.md` 仍写着“尚未完成正式训练或基准结论”，实际只有一次 1024 步 smoke 产物。
`configs/bitahub_first_run.yaml` 从未真实运行。

修复方向：跑通首次实验后更新 README 状态、补充一条最小结果记录（分数曲线、耗时、设备）。

## P1：影响训练效率与结果可信度

### 3. `DummyVecEnv` 串行推进，增大 `n_envs` 不提速

`train.py:make_seeded_vec_env` 使用 `DummyVecEnv`，各环境在同一进程内依次 `step`。
每个 env 背后是独立的 SC2 进程，单步延迟主要来自 SC2 模拟，串行执行无法重叠，13 核基本闲置。

修复方向：切换到 `SubprocVecEnv`。需要同时处理：

- 每个子进程的 seed 与 `random_seed` 传递（当前在构造期注入 `env_kwargs`）；
- `TRAIN_OUT_DIR` 导致的 `param_semantics_env.json` 多进程写竞争（当前每个 env 都会写同一路径）；
- 日志只由主进程写。

### 4. 吞吐约 14 env-steps/s，1M 步预计约 20 小时

来自 smoke 的 `time/fps`（8 envs，含评估）。评估本身开销显著：
`eval_every=50` 对应约 156 次评估，每次 10 个 episode、单个 episode 约 3 秒，评估合计超过 1 小时。

修复方向：

- 首次实验先跑 1e5 量级确认学习趋势，再决定是否补到 1M；
- 长跑前评估 `n_eval_episodes`、`eval_every` 与 `eval_deterministic` 的成本；
- 记录并对比启用/禁用评估时的 `time/fps`，量化评估占比。

### 5. 周期 checkpoint 磁盘占用

单模型约 33MB，`save_iters=200` 在 1M 步下约 39 次保存，粗算 1.3GB；
`save_best_model=true` 时仅在训练结束/发现更优模型时清理，中途会持续增长。

修复方向：确认云端磁盘配额；必要时保留最近 K 个 checkpoint 而不是全部保留。

## P2：代码健壮性与维护性

### 6. `configs/bitahub_smoke.yaml` 未纳入版本管理

该文件当前为 untracked，但它是可复用的最小 smoke 配置（1024 步）。
修复方向：确认后提交，或在 README 中说明其用途与生成方式。

### 7. `LogEvalCallback` 内大量静默 `except Exception: pass`

`train.py` 的 `LogEvalCallback._on_training_start` / `_on_step` 中，初始评估、统计聚合、checkpoint 清理均用
`try/except: pass` 包裹，出错时没有任何日志，短跑 smoke 会表现为“评估没生效但训练正常”。

修复方向：至少 `logger.warning` 记录异常类型与信息，保留不影响主流程的语义。

### 8. `JointMaskedDistribution` 只实现了部分分布接口

`MaskedFlattenPolicy._get_masked_distribution` 返回的内部类只提供 `get_actions`、`log_prob`、`entropy`，
缺少 `mode()`、`kl_divergence()`、`proba_distribution()` 等 SB3 `Distribution` 成员。
目前 `predict` 路径可用，但任何依赖完整分布接口的调用（自定义回调、算法扩展、SB3 版本升级）会直接报错。

修复方向：改为显式的 `Distribution` 子类并实现完整接口，或在注释中明确其受限用途。

### 9. 联合 log_prob / 熵的实现存在逐元素 CPU 同步

`_joint_logprob_and_entropy` 对 13 个参数槽位做 Python 循环，并在每次调用中多次 `.cpu().item()`
取 per-arg 统计；该函数在 rollout 采样和每次 PPO 更新时都会被调用。
GPU 上会产生大量同步点，削弱并行度。

修复方向：统计量改为张量形式累积，只在写入 TensorBoard 时（按 `summary_iters`）统一同步。

### 10. 采样与训练各算一次前向，特征未复用

`forward()` 与 `evaluate_actions()` 各自执行 `extract_features` + `mlp_extractor` + `_inject_spatial_logits`，
PPO 每轮需要对同一批观测重复计算特征。

修复方向：在保持 SB3 接口的前提下，评估是否值得缓存（SB3 的 `evaluate_actions` 本身不复用 rollout 期特征，
属于框架限制，需评估改动收益后再决定）。

### 11. Gymnasium API 细节

`PySC2GymEnv.metadata` 使用旧版 `render.modes`，`render(mode="human")` 忽略参数并返回 `None`。
当前不影响训练，但 `gymnasium` 0.29 期望 `render_modes`；后续若启用渲染或做 `check_env` 会告警/失败。

修复方向：改为 `metadata = {"render_modes": ["human"]}`，并让 `render` 签名符合 Gymnasium。

## 已验证的基线

- `pytest -q`：58 passed（2026-10-20，`.venv`，Python 3.10.20）
- `tools/bitahub_preflight.py`：仅上述 P0-1 两项 FAIL，其余 PASS
- 设备：RTX 4090 24GB，torch 2.2.2+cu121，`torch.cuda.is_available()=True`
- smoke：`total_timesteps=1024`，产出 `final_model` / `best_model`（约 33MB）、`evaluations.npz`；
  评估分数 0–3，episode 长度 240（`step_mul=8`，即 1920 game steps），不足以判断学习效果
