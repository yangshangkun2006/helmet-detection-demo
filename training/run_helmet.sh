#!/usr/bin/env bash
# 安全帽检测模型：数据准备 -> GPU 微调 -> 评估 -> 带框样例图
# 云端后台用法：
#   cd /workspace && setsid nohup bash /workspace/run_helmet.sh \
#       > /workspace/helmet_run.log 2>&1 < /dev/null &
set -euo pipefail
P=/workspace/yolo26-rocm-day1-labs/.venv/bin/python

echo "===== [1/2] 数据准备：COCO json -> YOLO txt 分层子集 ====="
"$P" /workspace/prepare_helmet_data.py 2>&1 | tee /workspace/helmet_prepare.log

echo "===== [2/2] W7900 微调 + 评估 + 推理示例 ====="
"$P" /workspace/train_helmet.py 2>&1 | tee /workspace/helmet_train.log

echo "===== 最终状态 ====="
cat /workspace/helmet_runs/summary.json
echo "HELMET_DONE"
