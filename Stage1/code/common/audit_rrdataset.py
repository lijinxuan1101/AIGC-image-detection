#!/usr/bin/env python3
"""M1 数据审计：为 RRDataset 建立逐图清单，并核验原图-传播图配对。

输出（均在 results/audit/）：
  inventory.csv        每张图一行：包、split、condition、label、source_id、场景、格式、尺寸、字节数、sha1、JPEG 量化表摘要
  pairing_check.csv    抽样配对核验：原图与传播图的缩略图相关系数、长宽比差异
"""
import argparse
import hashlib
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(ROOT, "results", "audit")

SCENARIOS = ["Culture_&_Religion", "Medical_&_Public_Health", "Natural_Disasters_&_Accidents",
             "Political_&_Social_Events", "War_&_Conflict_Scenes", "normal", "production", "real"]


def parse_name(fname):
    stem = fname.rsplit(".", 1)[0]
    sid = re.sub(r"^(transfer|redigital)_", "", stem)
    scen = "unknown"
    for s in SCENARIOS:
        if sid.startswith(s):
            scen = s
            break
    return sid, scen


def describe(args):
    path, pkg, split, cond, label = args
    fname = os.path.basename(path)
    sid, scen = parse_name(fname)
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    rec = dict(package=pkg, split=split, condition=cond, label=1 if label == "ai" else 0,
               label_name=label, source_id=sid, scenario=scen, filename=fname,
               relpath=os.path.relpath(path, DATA), bytes=os.path.getsize(path), sha1=h.hexdigest())
    try:
        with Image.open(path) as im:
            rec.update(format=im.format, mode=im.mode, width=im.size[0], height=im.size[1])
            q = getattr(im, "quantization", None)
            rec["jpeg_q0_sum"] = int(sum(q[0])) if q and 0 in q else -1  # 亮度量化表之和：越大压缩越狠
            rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, err=str(e)[:100])
    return rec


def thumb(path, n=32):
    with Image.open(path) as im:
        im = im.convert("L").resize((n, n), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32).ravel(), Image.open(path).size


def pair_check(args):
    po, pt = args
    a, so = thumb(po)
    b, st = thumb(pt)
    a = a - a.mean()
    b = b - b.mean()
    corr = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return dict(orig=po, transfer=pt, corr=corr, orig_w=so[0], orig_h=so[1], tr_w=st[0], tr_h=st[1],
                aspect_diff=abs(so[0] / so[1] - st[0] / st[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--pair-sample", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    jobs = []
    for split in ["train", "val"]:
        for label in ["real", "ai"]:
            d = os.path.join(DATA, "RRDataset_original_train_val", split, label)
            jobs += [(os.path.join(d, f), "train_val", split, "original", label) for f in sorted(os.listdir(d))]
    for cond in ["original", "transfer"]:
        for label in ["real", "ai"]:
            d = os.path.join(DATA, "RRDataset_final", cond, label)
            jobs += [(os.path.join(d, f), "test", "test", cond, label) for f in sorted(os.listdir(d))]
    print("images:", len(jobs), flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(describe, jobs, chunksize=64))
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(OUT, "inventory.csv"), index=False)
    print("inventory saved; unreadable:", int((~df.ok).sum()), flush=True)

    t = df[df.package == "test"]
    o = t[t.condition == "original"].set_index(["label_name", "source_id"])
    tr = t[t.condition == "transfer"].set_index(["label_name", "source_id"])
    common = sorted(set(o.index) & set(tr.index))
    rng = np.random.RandomState(0)
    pick = [common[i] for i in rng.choice(len(common), min(a.pair_sample, len(common)), replace=False)]
    pj = [(os.path.join(DATA, o.loc[k, "relpath"]), os.path.join(DATA, tr.loc[k, "relpath"])) for k in pick]
    with ProcessPoolExecutor(a.workers) as ex:
        pc = pd.DataFrame(list(ex.map(pair_check, pj, chunksize=8)))
    pc["label_name"] = [k[0] for k in pick]
    pc["source_id"] = [k[1] for k in pick]
    # 对照：随机错配的相关系数（负控制）
    shuf = [(pj[i][0], pj[(i + 1) % len(pj)][1]) for i in range(len(pj))]
    with ProcessPoolExecutor(a.workers) as ex:
        nc = pd.DataFrame(list(ex.map(pair_check, shuf, chunksize=8)))
    pc["corr_mismatch_control"] = nc["corr"].values
    pc.to_csv(os.path.join(OUT, "pairing_check.csv"), index=False)
    print(pc.groupby("label_name")[["corr", "corr_mismatch_control", "aspect_diff"]].describe().T.to_string())


if __name__ == "__main__":
    main()
