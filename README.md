# SC2 Minigame PPO Project

简介：使用 `PySC2` 提供的 StarCraft II minigame 作为环境，采用 `stable-baselines3` 中的 PPO 算法做训练。此仓库为实验和学习用途的骨架。

快速开始：

1. 创建 conda 环境（见 `environment.yml`）：

```powershell
conda env create -f environment.yml
conda activate minigame
```

2.（可选）或使用 `requirements.txt`：

```powershell
pip install -r requirements.txt
```

3. 按需安装并配置 StarCraft II（Windows）并将 minigames 地图放到 PySC2 可识别的位置。

4. 运行训练示例：

```powershell
python train.py --env "MoveToBeacon" --total-timesteps 10000
```

注意：`pysc2` 在 Windows 上有额外依赖，且 StarCraft II 需要手动下载与许可。训练建议在云服务器（有 GPU）上运行。
