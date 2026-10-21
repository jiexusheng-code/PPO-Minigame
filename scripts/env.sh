#!/bin/bash
# 云端训练环境变量配置
# 使用方式：source scripts/env.sh

export SC2PATH=/mnt/ppo/StarCraftII
export DISPLAY=:99

echo "[INFO] 环境变量已设置："
echo "  SC2PATH=$SC2PATH"
echo "  DISPLAY=$DISPLAY"
