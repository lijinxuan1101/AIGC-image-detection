#!/bin/bash
# 阶段 B/C/D 的顺序实验队列。所有运行共用：阶段 A 投影初始化、相同种子/步数/样本曝光量。
cd "$(dirname "$0")/.."
COMMON="--stage B --init results/runs/A_proj/last.pt --steps 200 --micro 4 --accum 4 --eval-every 50 --eval-n 150"
run() {  # run NAME extra-args...
  name=$1; shift
  if [ -s results/runs/$name/summary.json ] && ls results/runs/$name/pred_target_dev.csv >/dev/null 2>&1; then echo "skip $name"; return; fi
  echo "=== $name $(date)"
  python3 code/train.py --name $name $COMMON "$@" > logs/$name.log 2>&1
}
run M0
run M1_lib --lib
run M2_gsr --gsr
run M3_lib_gsr --lib --gsr
run M3_prop --lib --gsr --aug prop
run M3_sem0.1 --lib --gsr --sem 0.1
run M3_nocap --lib --gsr --no-caption
echo "QUEUE DONE $(date)"
