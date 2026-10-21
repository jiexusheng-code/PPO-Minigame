#!/bin/bash
# 云端训练环境变量配置
# 使用方式：source scripts/env.sh
#
# 只需要 SC2PATH。本项目以 visualize=false 运行，SC2 无头启动，
# 不需要 X server / DISPLAY（已在本机 unset DISPLAY 验证可正常训练）。

export SC2PATH=/mnt/ppo/StarCraftII

if [ ! -d "$SC2PATH" ]; then
    echo "[WARN] SC2PATH 不存在: $SC2PATH，请修改 scripts/env.sh 中的路径" >&2
fi

echo "[INFO] SC2PATH=$SC2PATH"
