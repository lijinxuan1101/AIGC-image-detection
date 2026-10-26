#!/bin/bash
# M4：对 calib_src / target_dev / closed_test 批量推理（每模型、每划分一个文件，跳过已完成者）
cd "$(dirname "$0")/../.." && source .venv/bin/activate
m=$1
for sp in calib_src target_dev closed_test; do
  out=results/predictions/${sp}_${m}.csv
  [ -s "$out" ] && { echo "skip $out"; continue; }
  python code/common/run_detector.py --model $m --splits $sp --out $out --workers 12 > logs/infer_${sp}_${m}.log 2>&1
done
echo "ALL DONE $m"
