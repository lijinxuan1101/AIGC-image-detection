#!/bin/bash
# 队列收尾：补评 M0（训练已完成，只缺预测）；修复语义头 bug 后重跑 M3_sem0.1。
cd "$(dirname "$0")/.."
echo "=== eval M0 $(date)"
python3 code/eval_run.py --name M0 > logs/M0_eval.log 2>&1
rm -rf results/runs/M3_sem0.1
echo "=== M3_sem0.1 $(date)"
python3 code/train.py --name M3_sem0.1 --stage B --init results/runs/A_proj/last.pt --steps 200 --micro 4 --accum 4 --eval-every 50 --eval-n 150 --lib --gsr --sem 0.1 > logs/M3_sem0.1.log 2>&1
echo "REST DONE $(date)"
