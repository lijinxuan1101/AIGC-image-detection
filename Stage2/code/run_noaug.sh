#!/bin/bash
# 等 run_rest.sh 结束后，跑 M3 不加增强对照（只用原图视图；来源图抽样序列、步数、种子与 M3 相同）。
cd "$(dirname "$0")/.."
while ! grep -q "REST DONE" logs/rest.log; do sleep 30; done
echo "=== M3_noaug $(date)"
python3 code/train.py --name M3_noaug --stage B --init results/runs/A_proj/last.pt --steps 200 --micro 4 --accum 4 --eval-every 50 --eval-n 150 --lib --gsr --aug none > logs/M3_noaug.log 2>&1
echo "NOAUG DONE $(date)"
