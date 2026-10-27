#!/usr/bin/env python3
"""对已保存的运行（last.pt）补做最终评估，输出 pred_explore.csv / pred_target_dev.csv。"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS  # noqa: E402
from data import Views  # noqa: E402
from evaluate import DEV_CONDS, macro_auroc, predict  # noqa: E402
from model import Cfg, Detector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--ckpt", default="last.pt")
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()

d = os.path.join(RESULTS, "runs", a.name)
cfg = Cfg(**json.load(open(os.path.join(d, "config.json")))["cfg"])
model = Detector(cfg)
missing = model.load_trainable(torch.load(os.path.join(d, a.ckpt)), strict=True)
model.eval()
views = Views()
for sub in ("explore", "target_dev"):
    v = views.df[(views.df.split == "dev") & (views.df.subset == sub) & views.df.condition.isin(DEV_CONDS + ["prop"])]
    pred = predict(model, views, v, bs=a.bs, log=True)
    pred.drop(columns=["path", "caption"], errors="ignore").to_csv(os.path.join(d, f"pred_{sub}.csv"), index=False)
    m, t = macro_auroc(pred)
    print(f"[{a.name} {sub}] macro AUROC={m:.4f}\n{t.round(3).to_string()}", flush=True)
