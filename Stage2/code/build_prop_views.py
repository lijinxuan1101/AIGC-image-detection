#!/usr/bin/env python3
"""用随机参数传播流水线（propagation.py）预生成视图。

  --part trainprop : 与 build_views 的 train/debug 相同的来源图，每张图 K=3 个视图 p0..p2
  --part devprop   : dev（explore+target_dev）每张原图 1 个视图 'prop'（仅用于报告，不参与 checkpoint 选择）
抽样种子只依赖 (SEED, SPEC_VERSION, source_key, k)，与标签无关。每个视图的全部参数写入 csv。
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from build_views import to_pix  # noqa: E402
from common import DATA, IMG_SIZE, S1_MANIFEST, SEED, pix_path, stable_int, views_csv  # noqa: E402
from propagation import SPEC_VERSION, apply, flat, sample_params  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
K_TRAIN_VIEWS = 3


def params_for(source_key, k):
    return sample_params(np.random.RandomState(stable_int(f"{SEED}|{SPEC_VERSION}|{source_key}|{k}")))


def work(job):
    img = Image.open(job["path_in"])
    img.load()
    data = apply(img, job["P"])
    os.makedirs(os.path.dirname(job["out_path"]), exist_ok=True)
    with open(job["out_path"], "wb") as f:
        f.write(data)
    out = Image.open(job["out_path"])
    out.load()
    return job["idx"], to_pix(out), out.size[0], out.size[1], len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["trainprop", "devprop"], required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()

    m = pd.read_csv(S1_MANIFEST)
    rows, jobs = [], []

    def add(split, sk, cond, pin, label, group, fmt, ow, oh, k):
        P = params_for(sk, k)
        idx = len(rows)
        outp = os.path.join(DATA, "views", a.part, f"{sk.replace('/', '_')}_{cond}.jpg")
        rows.append(dict(view_key=f"{split}|{sk}|{cond}", split=split, source_key=sk, group_id=group, label=label,
                         condition=cond, kind="prop", path=outp, q=P["jpeg"]["q"], scale=1.0, orig_format=fmt,
                         orig_w=ow, orig_h=oh, **flat(P)))
        jobs.append(dict(idx=idx, path_in=pin, out_path=outp, P=P))

    if a.part == "trainprop":
        # 与 build_views 相同的 debug 抽样（同一 RandomState 流程）
        cs = m[m.split == "calib_src"].reset_index(drop=True)
        rng = np.random.RandomState(SEED)
        dbg = set()
        for lab in (0, 1):
            dbg |= set(rng.choice(cs[cs.label == lab].source_key.values, 32, replace=False))
        for r in cs.itertuples():
            split = "debug" if r.source_key in dbg else "train"
            pin = os.path.join(DATA, r.relpath)
            for k in range(K_TRAIN_VIEWS):
                add(split, r.source_key, f"p{k}", pin, r.label, r.group_id, r.format, r.width, r.height, k)
    else:
        dv = m[(m.split.isin(["explore", "target_dev"])) & (m.condition == "original")]
        for r in dv.itertuples():
            add("dev", r.source_key, "prop", os.path.join(DATA, r.relpath), r.label, r.group_id, r.format,
                r.width, r.height, 0)

    n = len(rows)
    print("views:", n, flush=True)
    pix = np.lib.format.open_memmap(pix_path(a.part), mode="w+", dtype=np.uint8, shape=(n, IMG_SIZE, IMG_SIZE, 3))
    ws, hs, bs = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for idx, arr, w, h, b in ex.map(work, jobs, chunksize=8):
            pix[idx] = arr
            ws[idx], hs[idx], bs[idx] = w, h, b
            done += 1
            if done % 2000 == 0:
                print(done, "/", n, flush=True)
    pix.flush()
    df = pd.DataFrame(rows)
    df["w"], df["h"], df["bytes"] = ws, hs, bs
    df["pix_idx"] = np.arange(n)
    df.to_csv(views_csv(a.part), index=False)
    print(df.groupby(["split", "condition", "label"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
