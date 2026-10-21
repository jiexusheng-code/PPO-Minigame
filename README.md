# PPO-Minigame

基于 PySC2 与 Stable-Baselines3 PPO 的《星际争霸 II》minigame 强化学习项目。当前目标是先把 MoveToBeacon 等小任务训练跑通并验证学习效果，再逐步扩展到更复杂的 minigame。

> **项目状态**：
> - ✅ 本地验证完成（smoke test 通过）
> - ✅ 云端环境配置就绪
> - ⏳ 等待首次 1M 步实验执行
> - 尚无正式基准结论

## 技术栈

- Python 3.10
- PySC2 3.0.0 + Gymnasium 0.29.1
- Stable-Baselines3 2.3.0（PPO）
- PyTorch 2.2（默认 CUDA 12.1 wheel，也可回退 CPU）
- TensorBoard

## 目录结构

```text
train.py                       # 训练入口
src/
  envs/pysc2_gym_wrapper.py    # PySC2 -> Gymnasium 环境包装
  policies/masked_flatten_policy.py  # 自定义策略、动作掩码与参数头
  callbacks/tb_dual_writer.py  # TensorBoard 双目录写入
  obs_parser.py                # observation -> 固定结构特征
configs/
  ppo_config.yaml              # 本地默认配置（无限训练）
  bitahub_first_run.yaml       # Bitahub 有限步数首次云端配置
tools/
  smoke_run.py                 # 本地短时 smoke
  bitahub_preflight.py         # 云端启动前只读检查
  inspect_tb.py                # 查看 TensorBoard 事件文件
  generate_action_slot_map.py  # 生成动作槽位映射参考表
  notebooks/                   # 手动检查 notebook
  probes/                      # 手动 SC2 探针（不会被 pytest 收集）
tests/                         # pytest 正式测试
docs/
  reference/                   # 动作/观测/奖励/超参数/TensorBoard 参考
  archive/plan-2026-02.md      # 历史计划，不代表当前实现
pyproject.toml
uv.lock
```

## 安装（仅使用 uv）

本仓库只维护 `pyproject.toml` + `uv.lock` 一套依赖；`requirements.txt` 与 `environment.yml` 已移除（旧结构保留在 `legacy/pre-uv-migration` 分支）。

需要 Python 3.10：

```bash
uv python install 3.10
uv sync
```

运行针对性测试：

```bash
uv run pytest -q
```

运行前还需自行准备 StarCraft II 客户端及 PySC2 所需地图。

## 云端环境配置

在 Bitahub 或其他云端环境首次使用时，需要配置环境变量：

```bash
# 方式 1：每次会话手动设置
export SC2PATH=/mnt/ppo/StarCraftII
export DISPLAY=:99

# 方式 2：使用项目提供的环境脚本（推荐）
source scripts/env.sh

# 方式 3：写入 ~/.bashrc（永久生效）
echo 'export SC2PATH=/mnt/ppo/StarCraftII' >> ~/.bashrc
echo 'export DISPLAY=:99' >> ~/.bashrc
source ~/.bashrc
```

配置完成后，运行 preflight 检查：

```bash
python tools/bitahub_preflight.py --config configs/bitahub_first_run.yaml
```

## 计算设备

配置项 `device` 默认 `auto`：CUDA 可用时优先使用 GPU，否则回退 CPU。也可显式写 `cpu`、`cuda` 或 `cuda:0`；显式指定 CUDA 但不可用时会直接报错，避免在 GPU 机器上静默用 CPU 长时间训练。

## 本地 smoke

```bash
uv run python tools/smoke_run.py
```

该命令基于 `configs/ppo_config.yaml` 生成 512 步的临时配置，输出写入临时目录，结束后清理，不影响正式模型目录。

## Bitahub 首次云端运行

启动前只读检查（不会启动 SC2）：

```bash
uv run python tools/bitahub_preflight.py --config configs/bitahub_first_run.yaml
```

检查通过后运行有限步数的首次云端配置：

```bash
TRAIN_CONFIG_PATH=configs/bitahub_first_run.yaml uv run python train.py
```

PowerShell：

```powershell
$env:TRAIN_CONFIG_PATH="configs/bitahub_first_run.yaml"; uv run python train.py
```

`configs/bitahub_first_run.yaml` 固定为 1,000,000 步，用于首次云端验证，不是正式基准实验。

## 性能优化配置

### 并行环境

- `vec_env_cls: "subproc"`：使用多进程并行环境，充分利用多核 CPU（推荐用于正式训练）
- `vec_env_cls: "dummy"`：单进程串行环境，便于调试和排查问题

### 评估频率

`eval_every` 控制评估间隔（单位：训练步数）：

- 较小值（如 10000）：更频繁评估，曲线更平滑，但会增加训练时间
- 较大值（如 50000）：减少评估开销，加快训练，但曲线粒度粗

### Checkpoint 策略

`save_iters` 控制模型保存频率（单位：训练步数）：

- 较小值：更频繁保存，便于回滚，但占用更多磁盘空间
- 较大值：减少磁盘占用，但回滚粒度粗

建议首次实验使用默认配置，待稳定后根据实际需求调整。

## 职责边界

- Stable-Baselines3 PPO：负责 rollout、GAE、PPO 裁剪目标、优化器与训练循环。
- `train.py`：负责配置读取与校验、设备解析、环境/评估环境构造、回调装配、保存与续训。
- `src/envs/pysc2_gym_wrapper.py`：把 PySC2 minigame 包装为 Gymnasium 环境，处理 reset/step、观测与动作接口。
- `src/obs_parser.py`：把原始 observation 解析为固定的 vector/screen/minimap 特征及 mask。
- `src/policies/masked_flatten_policy.py`：项目自定义的 MultiInputPolicy，负责动作掩码、函数头与参数头建模。
- `src/callbacks/tb_dual_writer.py`：把项目扩展指标写入 `tb_logs/extra/`。

## TensorBoard

训练日志位于 `<out_dir>/<timestamp>/tb_logs/`，实际子目录取决于运行：

- `extra/`：项目回调 `TBDualWriterCallback` 写入的扩展指标（episode score/length、per-arg 指标等）。
- `PPO_<n>/`（如 `PPO_1/`）：SB3 内置 writer 写入的主指标（`rollout/*`、`eval/*`、`train/*`、`time/*`）。SB3 会自动递增编号。

当前实现没有单独的 `fundamental/` writer，因此不保证该目录存在。

查看方式：

```bash
uv run tensorboard --logdir models/<timestamp>/tb_logs
```

## 参考文档

- [动作空间](docs/reference/action-space.md)
- [观测空间](docs/reference/observation-space.md)
- [奖励与分数](docs/reference/rewards-and-scores.md)
- [SC2 minigames](docs/reference/sc2-minigames.md)
- [超参数](docs/reference/hyperparameters.md)
- [TensorBoard 指标](docs/reference/tensorboard.md)
