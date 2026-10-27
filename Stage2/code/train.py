#!/usr/bin/env python3
"""Stage 2 训练入口。

阶段 A（--stage A）：冻结 SigLIP2 / Phi-3-mini / BLIP-2，只训练视觉投影；无 LIB/GSR/语义教师。
阶段 B（--stage B）：SigLIP2 + Phi-3 的 LoRA + 视觉投影（+ 可选 LIB / GSR / 语义对齐）。
所有 M0–M3 对照使用同一阶段 A 投影初始化、相同的样本曝光量与随机种子。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, ROOT, SEED, set_seed  # noqa: E402
from data import PairSampler, Views  # noqa: E402
from evaluate import DEV_CONDS, macro_auroc, predict  # noqa: E402
from model import Cfg, Detector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--lib", action="store_true")
    ap.add_argument("--lib-kernel", type=int, default=3)
    ap.add_argument("--gsr", action="store_true")
    ap.add_argument("--gsr-radius", type=int, default=2)
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--sem", type=float, default=0.0, help="λ_sem；0 表示不构建语义头")
    ap.add_argument("--split", default="train", help="训练划分；阶段 A 过拟合检查用 debug")
    ap.add_argument("--steps", type=int, default=300, help="优化器步数")
    ap.add_argument("--micro", type=int, default=4, help="微批每类样本数（real 与 ai 各 micro 个；每个样本含原图+退化图两个视图，即一个微批 4*micro 个视图）")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr-new", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--init", default=None, help="加载阶段 A 的投影等可训练参数")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-subset", default="explore")
    ap.add_argument("--eval-n", type=int, default=0, help="周期评估只用前 n 个来源（0=全部）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--aug", choices=["ordinary", "prop", "none"], default="ordinary",
                    help="x′ 来源：ordinary=单次 JPEG(+缩放)；prop=随机参数传播流水线；none=不加增强，只用原图视图")
    ap.add_argument("--no-ckpt", action="store_true")
    ap.add_argument("--final-eval", nargs="*", default=["explore", "target_dev"])
    a = ap.parse_args()

    set_seed(a.seed)
    out_dir = os.path.join(RESULTS, "runs", a.name)
    os.makedirs(out_dir, exist_ok=True)
    cfg = Cfg(use_lib=a.lib, lib_kernel=a.lib_kernel, use_gsr=a.gsr, gsr_radius=a.gsr_radius,
              lora=(a.stage == "B"), use_caption=not a.no_caption, sem=a.sem > 0, grad_ckpt=not a.no_ckpt)
    json.dump({"args": vars(a), "cfg": cfg.to_dict()}, open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    views = Views()
    model = Detector(cfg)
    if a.init:
        sd = torch.load(a.init, map_location="cpu")
        miss = model.load_trainable(sd)
        print("加载初始化参数", len(sd), "未匹配", miss[:5], flush=True)
    model.train()

    # ---- 冻结核验 ----
    train_names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数 {n_tr / 1e6:.2f}M，张量 {len(train_names)}；示例 {train_names[:3]}", flush=True)
    assert all(("lora_" in n) or n.startswith(("proj", "lib", "gsr", "aux")) for n in train_names), "有非预期参数被训练"

    groups = [{"params": model.new_params(), "lr": a.lr_new, "weight_decay": 0.01}]
    if a.stage == "B":
        groups.append({"params": model.lora_params(), "lr": a.lr_lora, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    warm = max(1, int(a.steps * a.warmup))

    def set_lr(step):
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, a.steps - warm)))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * max(f, 0.0)

    sampler = PairSampler(views, a.split, seed=a.seed, deg_prefix="p" if a.aug == "prop" else "d")   # none 时仍抽取但不使用，保证来源图抽样序列与其他运行一致
    batch = a.micro * 2                              # 每个微批 micro 个 real+ai 平衡的样本对
    n_micro = a.steps * a.accum
    gen = sampler.batches(batch, n_micro)
    eff = 2 * a.micro * a.accum                      # 每个微批 micro 个 real + micro 个 ai
    print(f"有效 batch={eff} 对（{2 * eff} 视图），步数={a.steps}，曝光 {a.steps * eff} 个样本对，"
          f"约 {a.steps * eff / (2 * min(len(v) for v in sampler.keys.values())):.2f} 个 epoch", flush=True)

    dev = views.df[(views.df.split == "dev") & (views.df.subset == a.eval_subset)
                   & views.df.condition.isin(DEV_CONDS)]
    if a.eval_n:   # 每类各取前 eval_n//2 个来源（Stage1 排序使 real 在前，直接 head 会只剩一类）
        src = dev[dev.condition == "orig"].drop_duplicates("source_key")
        keep = pd.concat([src[src.label == l].head(a.eval_n // 2) for l in (0, 1)]).source_key
        dev = dev[dev.source_key.isin(keep)]

    log = []
    best = (-1, None)
    t0 = time.time()
    run_loss = run_acc = run_sem = 0.0
    for step in range(a.steps):
        set_lr(step)
        for _ in range(a.accum):
            o, d = next(gen)
            pix_o, cap_o = views.get(o)
            if a.aug == "none":              # 只用原图视图：每步来源图数不变，视图数减半
                pix, caps = pix_o, cap_o
                y = torch.tensor(o.label.values, dtype=torch.float32, device="cuda")
                out = model(pix, caps, with_sem=a.sem > 0)
                det = F.binary_cross_entropy_with_logits(out["logit"], y)
            else:
                pix_d, cap_d = views.get(d)
                pix = torch.cat([pix_o, pix_d])
                caps = cap_o + cap_d
                y = torch.tensor(np.concatenate([o.label.values, d.label.values]), dtype=torch.float32, device="cuda")
                out = model(pix, caps, with_sem=a.sem > 0)
                det = 0.5 * (F.binary_cross_entropy_with_logits(out["logit"][:len(o)], y[:len(o)]) +
                             F.binary_cross_entropy_with_logits(out["logit"][len(o):], y[len(o):]))
            loss = det
            if a.sem > 0:
                sem = out["sem"].mean()
                loss = det + a.sem * sem
                run_sem += sem.item() / a.accum
            (loss / a.accum).backward()
            run_loss += det.item() / a.accum
            run_acc += ((out["logit"] > 0).float() == y).float().mean().item() / a.accum
        gn = torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 10 == 0 or step == 0:
            n = 1 if step == 0 else 10
            rec = dict(step=step + 1, loss=run_loss / n, acc=run_acc / n, sem=run_sem / n, gnorm=gn,
                       lr_new=opt.param_groups[0]["lr"], mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                       elapsed=time.time() - t0)
            if cfg.use_lib:
                rec["lib_alpha"] = float(torch.sigmoid(model.lib.beta))
            print(json.dumps(rec), flush=True)
            log.append(rec)
            run_loss = run_acc = run_sem = 0.0
        if a.eval_every and (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            pred = predict(model, views, dev)
            m, t = macro_auroc(pred)
            print(f"[eval step {step + 1}] macro AUROC={m:.4f}  " +
                  " ".join(f"{r.condition}={r.auroc:.3f}" for r in t.itertuples()), flush=True)
            log.append(dict(step=step + 1, eval_macro_auroc=m, **{f"auroc_{r.condition}": r.auroc for r in t.itertuples()}))
            if m > best[0]:
                best = (m, step + 1)      # 仅记录；所有运行统一使用最后一步的权重（见报告）

    # ---- 收尾：最终评估与保存 ----
    torch.save(model.trainable_state(), os.path.join(out_dir, "last.pt"))
    if a.eval_every:
        pred = predict(model, views, dev)
        m, t = macro_auroc(pred)
        print(f"[eval final step {a.steps}] macro AUROC={m:.4f}", flush=True)
        log.append(dict(step=a.steps, eval_macro_auroc=m, **{f"auroc_{r.condition}": r.auroc for r in t.itertuples()}))
        best = (m, a.steps)
    pd.DataFrame(log).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)
    json.dump({"best_step": best[1], "best_macro_auroc": best[0], "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
               "train_seconds": time.time() - t0}, open(os.path.join(out_dir, "summary.json"), "w"))

    for sub in a.final_eval:
        d = views.df[(views.df.split == "dev") & (views.df.subset == sub) &
                     views.df.condition.isin(DEV_CONDS + ["prop"])]
        if len(d) == 0:
            continue
        pred = predict(model, views, d, log=True)
        pred.drop(columns=["path", "caption"], errors="ignore").to_csv(os.path.join(out_dir, f"pred_{sub}.csv"), index=False)
        m, t = macro_auroc(pred)
        print(f"[final {sub}] macro AUROC={m:.4f}\n{t.round(3).to_string()}", flush=True)


if __name__ == "__main__":
    main()
