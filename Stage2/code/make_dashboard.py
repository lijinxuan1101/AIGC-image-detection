#!/usr/bin/env python3
"""把训练日志/结果解析成 Cursor canvas 面板（数据内嵌，需周期性重写文件以刷新）。

用法：
  python make_dashboard.py            # 写一次
  python make_dashboard.py --loop 60  # 每 60 秒检查并刷新（内容变化才写）
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CANVAS = "/root/.cursor/projects/root/canvases/stage2-training-dashboard.canvas.tsx"

# 队列定义（与 run_queue.sh 一致）：name, 说明, 总步数
PLAN = [
    ("A_proj", "阶段 A：只训练视觉投影（共用初始化）", 150),
    ("M0", "M0 基础：SigLIP2+投影（LoRA）", 200),
    ("M1_lib", "M1 = M0 + LIB", 200),
    ("M2_gsr", "M2 = M0 + GSR", 200),
    ("M3_lib_gsr", "M3 = M0 + LIB + GSR", 200),
    ("M3_prop", "M3 + 随机传播增强", 200),
    ("M3_sem0.1", "M3 + 语义对齐 λ=0.1", 200),
    ("M3_nocap", "M3 不用描述", 200),
    ("M3_noaug", "M3 不加增强（仅原图视图）", 200),
]


def clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def parse_log(path):
    steps, evals, err = [], [], None
    if not os.path.exists(path):
        return steps, evals, err
    for line in open(path, errors="ignore"):
        if line.startswith('{"step"'):
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        m = re.match(r"\[eval step (\d+)\] macro AUROC=(\S+)", line)
        if m:
            evals.append((int(m.group(1)), float(m.group(2))))
        if "Traceback" in line:
            err = "训练进程报错（见日志）"
    return steps, evals, err


def running_runs():
    try:
        out = subprocess.run(["pgrep", "-af", "code/train.py"], capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return set()
    return {m.group(1) for m in re.finditer(r"--name (\S+)", out) if "pgrep" not in m.group(0)}


def build():
    run_set = running_runs()
    runs, rows = {}, []
    for name, desc, total in PLAN:
        steps, evals, err = parse_log(os.path.join(ROOT, "logs", f"{name}.log"))
        rd = os.path.join(ROOT, "results", "runs", name)
        has_pred = os.path.exists(os.path.join(rd, "pred_target_dev.csv"))
        has_last = os.path.exists(os.path.join(rd, "last.pt"))
        cur = steps[-1]["step"] if steps else 0
        if name in run_set:
            status = "running"
        elif has_pred:
            status = "done"
        elif has_last:
            status = "trained"       # 已训练但缺评估
        elif steps and err:
            status = "failed"
        else:
            status = "pending"
        last = steps[-1] if steps else {}
        rows.append(dict(name=name, desc=desc, status=status, step=cur, total=total,
                         loss=clean(last.get("loss")), acc=clean(last.get("acc")),
                         elapsed_min=round(last["elapsed"] / 60, 1) if last else None,
                         mem=round(last["mem_gb"], 1) if last else None,
                         alpha=clean(last.get("lib_alpha")),
                         sec_per_step=round(last["elapsed"] / cur, 1) if cur else None))
        runs[name] = dict(steps=[s["step"] for s in steps], loss=[clean(s["loss"]) for s in steps],
                          acc=[clean(s["acc"]) for s in steps], gnorm=[clean(s["gnorm"]) for s in steps],
                          evals=evals)
    data = dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), rows=rows, runs=runs)

    # 评估结果
    tdir = os.path.join(ROOT, "results", "tables")
    for key, fn in [("results", "target_dev_runs.csv"), ("shortcut", "shortcut_baselines.csv"),
                    ("stage1", "stage1_baselines_target_dev.csv"), ("jpeg_only", "target_dev_jpeg_only.csv")]:
        p = os.path.join(tdir, fn)
        data[key] = json.loads(pd.read_csv(p).round(4).to_json(orient="records")) if os.path.exists(p) else []

    # 阶段 A 在 explore 上的评估（analyze 之前就能给出）
    pa = os.path.join(ROOT, "results", "runs", "A_proj", "pred_explore.csv")
    data["explore_A"] = []
    if os.path.exists(pa):
        sys.path.insert(0, HERE)
        from evaluate import rank_table
        t = rank_table(pd.read_csv(pa), ["orig", "transfer", "q75", "q40"])
        data["explore_A"] = json.loads(t[["condition", "n_real", "n_fake", "auroc", "tpr", "fpr", "bacc"]].round(4).to_json(orient="records"))
    return data


TEMPLATE = r'''import {
  Callout, Card, CardBody, CardHeader, Divider, Grid, H1, H2, H3, LineChart, Pill, Row, Stack, Stat, Table, Text,
} from "cursor/canvas";

const D: any = __DATA__;

const STATUS: Record<string, string> = {
  done: "已完成", running: "运行中", pending: "排队中", trained: "已训练，待评估", failed: "失败",
};
const f3 = (v: any) => (v === null || v === undefined ? "-" : Number(v).toFixed(3));
const f2 = (v: any) => (v === null || v === undefined ? "-" : Number(v).toFixed(2));

export default function Dashboard() {
  const rows: any[] = D.rows;
  const running = rows.find((r) => r.status === "running");
  const doneCount = rows.filter((r) => r.status === "done" || r.status === "trained").length;

  // loss / acc 曲线：只画步数网格与 M0 一致且已跑完的阶段 B 运行，避免错位
  const full = rows.filter((r) => r.name !== "A_proj" && r.step >= r.total);
  const ref = full.length ? D.runs[full[0].name].steps : [];
  const comparable = full.filter((r) => D.runs[r.name].steps.length === ref.length);
  const cats = ref.map((s: number) => String(s));

  // 运行中任务的剩余时间
  let eta = "-";
  if (running && running.sec_per_step) {
    const mins = ((running.total - running.step) * running.sec_per_step) / 60;
    eta = mins >= 60 ? (mins / 60).toFixed(1) + " 小时" : Math.round(mins) + " 分钟";
  }
  const remainingRuns = rows.filter((r) => r.status === "pending").length;
  const stepSec = running?.sec_per_step ?? 10;
  const queueEtaH = ((remainingRuns * 200 * stepSec) / 3600 + remainingRuns * 0.15).toFixed(1);

  const finished = D.results.length > 0;
  const runNames = Array.from(new Set(D.results.map((r: any) => r.run)));
  const conds = ["orig", "transfer", "q75", "q40", "prop"];
  const resultRows = runNames.map((n: any) => [n, ...conds.map((c) => {
    const r = D.results.find((x: any) => x.run === n && x.condition === c);
    return r ? f3(r.auroc) + " [" + f3(r.auroc_lo) + ", " + f3(r.auroc_hi) + "]" : "-";
  })]);

  return (
    <Stack gap={20} style={{ padding: 20 }}>
      <Row justify="space-between" align="center">
        <H1>Stage 2 训练面板</H1>
        <Text tone="secondary" size="small">数据快照：{D.updated}（服务器时间，UTC）</Text>
      </Row>

      <Row gap={32} align="start" wrap>
        <Stat value={running ? running.name : "无"} label="当前运行" tone={running ? "info" : undefined} />
        <Stat value={running ? running.step + " / " + running.total : "-"} label="当前步数" />
        <Stat value={eta} label="当前运行预计剩余" />
        <Stat value={doneCount + " / " + rows.length} label="已训练的运行" />
        <Stat value={queueEtaH + " 小时"} label="排队任务预计总耗时（粗估）" />
      </Row>

      {D.rows.some((r: any) => r.status === "trained") && (
        <Callout tone="warning" title="部分运行已训练但还没有评估结果">
          {D.rows.filter((r: any) => r.status === "trained").map((r: any) => r.name).join("、")} 的权重已保存，
          但缺少 explore / target_dev 上的预测文件，需要补评后才能比较 AUROC。
        </Callout>
      )}

      <section>
        <H2>运行队列</H2>
        <Table
          headers={["运行", "说明", "状态", "步数", "最新 loss", "训练准确率", "单步耗时 (s)", "峰值显存 (GB)", "LIB α"]}
          rows={rows.map((r) => [
            r.name, r.desc, STATUS[r.status] ?? r.status, r.step + " / " + r.total,
            f3(r.loss), f3(r.acc), r.sec_per_step ?? "-", r.mem ?? "-", r.alpha === null ? "-" : f3(r.alpha),
          ])}
          rowTone={rows.map((r) => (r.status === "running" ? "info" : r.status === "done" ? "success"
            : r.status === "failed" ? "danger" : r.status === "trained" ? "warning" : "neutral"))}
          columnAlign={["left", "left", "left", "right", "right", "right", "right", "right", "right"]}
        />
        <Text tone="secondary" size="small">
          Loss/准确率为日志中每 10 步的平均值（训练批，标签平衡，含原图与退化图两个视图）。
        </Text>
      </section>

      {comparable.length > 0 && (
        <Grid columns={2} gap={20}>
          <section>
            <H3>训练 loss（每 10 步均值）</H3>
            <LineChart
              categories={cats}
              series={comparable.map((r) => ({ name: r.name, data: D.runs[r.name].loss }))}
              height={240}
            />
            <Text tone="secondary" size="small">横轴：优化器步数；纵轴：真假 BCE 损失（原图与退化图平均）。只显示已跑完且步数网格一致的阶段 B 运行。</Text>
          </section>
          <section>
            <H3>训练准确率（每 10 步均值）</H3>
            <LineChart
              categories={cats}
              series={comparable.map((r) => ({ name: r.name, data: D.runs[r.name].acc }))}
              height={240}
              yMin={0.8}
              yMax={1}
            />
            <Text tone="secondary" size="small">横轴：优化器步数；纵轴：阈值 0（p=0.5）下的训练批准确率，纵轴范围 0.8–1.0。</Text>
          </section>
        </Grid>
      )}

      {running && D.runs[running.name].steps.length > 0 && (
        <section>
          <H3>{running.name} 最近的训练记录</H3>
          <Table
            headers={["步数", "loss", "准确率", "梯度范数"]}
            rows={D.runs[running.name].steps.slice(-6).reverse().map((s: number, i: number, arr: number[]) => {
              const k = D.runs[running.name].steps.length - 1 - i;
              return [s, f3(D.runs[running.name].loss[k]), f3(D.runs[running.name].acc[k]), f2(D.runs[running.name].gnorm[k])];
            })}
            columnAlign={["right", "right", "right", "right"]}
          />
        </section>
      )}

      <Divider />

      {D.explore_A.length > 0 && (
        <section>
          <H2>阶段 A：explore 验证集结果</H2>
          <Table
            headers={["条件", "真实", "生成", "AUROC", "TPR（阈值 0）", "FPR（阈值 0）", "平衡准确率"]}
            rows={D.explore_A.map((r: any) => [r.condition, r.n_real, r.n_fake, f3(r.auroc), f3(r.tpr), f3(r.fpr), f3(r.bacc)])}
            columnAlign={["left", "right", "right", "right", "right", "right", "right"]}
          />
          <Text tone="secondary" size="small">
            explore 用于验证与选阈值；403 对（203 真实 / 200 生成）。数值很高，但见下方捷径基线，不能直接解读为模型更强。
          </Text>
        </section>
      )}

      <section>
        <H2>target_dev 开发测试：AUROC [95% CI]</H2>
        {finished ? (
          <Table
            headers={["运行", "orig", "transfer", "q75", "q40", "prop"]}
            rows={resultRows}
          />
        ) : (
          <Callout tone="neutral" title="还没有 target_dev 结果">
            阶段 B 运行完成评估后，运行 analyze.py 会生成该表。
          </Callout>
        )}
        <Text tone="secondary" size="small">
          target_dev：1004 对，不参与 checkpoint 或阈值选择；置信区间按来源图分组 bootstrap。transfer 为数据集自带的合成传播（JPEG 质量 20–55 + 约 35% 居中裁切），不是真实平台。
        </Text>
      </section>

      {D.shortcut.length > 0 && (
        <section>
          <H2>捷径基线（target_dev，AUROC）</H2>
          <Table
            headers={["条件", "仅文件元数据", "仅描述文本"]}
            rows={D.shortcut.map((r: any) => [r.condition, f3(r.metadata_only_auroc), f3(r.caption_only_auroc)])}
            columnAlign={["left", "right", "right"]}
          />
          <Text tone="secondary" size="small">
            元数据 = 格式/宽高/字节数；描述 = BLIP-2 生成的文字（TF-IDF + 逻辑回归）。两者都在训练集上训练。
          </Text>
        </section>
      )}

      {D.stage1.length > 0 && (
        <section>
          <H2>参照：Stage 1 官方基线（target_dev）</H2>
          <Table
            headers={["模型", "条件", "AUROC"]}
            rows={D.stage1.map((r: any) => [r.model, r.condition, f3(r.auroc)])}
            columnAlign={["left", "left", "right"]}
          />
          <Text tone="secondary" size="small">官方权重、未微调；与本表不同训练数据，只作量级参照。</Text>
        </section>
      )}
    </Stack>
  );
}
'''


def write(data):
    src = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    old = open(CANVAS).read() if os.path.exists(CANVAS) else ""
    # 比较时忽略时间戳
    strip = lambda s: re.sub(r'"updated": "[^"]*"', "", s)  # noqa: E731
    if strip(old) != strip(src):
        os.makedirs(os.path.dirname(CANVAS), exist_ok=True)
        open(CANVAS, "w").write(src)
        return True
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    while True:
        try:
            changed = write(build())
            print(time.strftime("%H:%M:%S"), "updated" if changed else "no change", flush=True)
        except Exception as e:  # noqa: BLE001
            print("error", repr(e), flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)
