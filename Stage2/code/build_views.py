#!/usr/bin/env python3
"""构建 Stage 2 的视图清单与 256x256 像素缓存。

划分（沿用 Stage1 manifest，不改动其分组）：
  train  : calib_src 去掉 debug；每张原图 + K 个预生成的随机退化视图（JPEG 质量/缩放与标签无关）
  debug  : calib_src 中随机抽 32 real + 32 ai，仅用于阶段 A 的接口/过拟合检查，不进入任何报告
  dev    : Stage1 的 explore + target_dev（1407 对原图/传播图）；另加固定强度的 JPEG q75 / q40 模拟视图
  closed_test 不在此处理（已被 Stage1 使用过一次，不再作为未见测试集）

退化视图本身以 JPEG 文件保存在 data/views/，使 BLIP-2 可以对"实际收到的图像"生成描述。
"""
import argparse
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATA, IMG_SIZE, S1_MANIFEST, SEED, pix_path, stable_int,  # noqa: E402
                    views_csv)

Image.MAX_IMAGE_PIXELS = None
K_TRAIN_VIEWS = 3
RESAMPLE = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]


def train_view_params(source_key: str, k: int):
    """与标签无关、仅由 source_key 与 k 决定的退化参数。"""
    rng = np.random.RandomState(stable_int(f"{SEED}|{source_key}|{k}"))
    q = int(rng.randint(30, 96))                       # JPEG 质量 30–95
    scale = 1.0 if rng.rand() < 0.5 else float(rng.uniform(0.5, 0.9))
    rs = int(rng.randint(0, len(RESAMPLE)))
    return q, scale, rs


def degrade_bytes(img: Image.Image, q: int, scale: float = 1.0, rs: int = 1) -> bytes:
    """可选缩小 -> 单次 JPEG 编码，返回文件字节（直接写盘，避免二次压缩）。"""
    img = img.convert("RGB")
    if scale < 1.0:
        w, h = img.size
        img = img.resize((max(8, round(w * scale)), max(8, round(h * scale))), RESAMPLE[rs])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def to_pix(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), dtype=np.uint8)


def work(job):
    """job: dict(path_in, kind, out_path|None, q, scale, rs) -> (idx, pix, out_w, out_h, bytes)"""
    idx = job["idx"]
    img = Image.open(job["path_in"])
    img.load()
    if job["kind"] == "file":
        return idx, to_pix(img), img.size[0], img.size[1], os.path.getsize(job["path_in"])
    data = degrade_bytes(img, job["q"], job["scale"], job["rs"])
    os.makedirs(os.path.dirname(job["out_path"]), exist_ok=True)
    with open(job["out_path"], "wb") as f:
        f.write(data)
    out = Image.open(job["out_path"])
    out.load()
    return idx, to_pix(out), out.size[0], out.size[1], len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--part", choices=["train", "dev"], required=True)
    a = ap.parse_args()

    m = pd.read_csv(S1_MANIFEST)
    rows, jobs = [], []

    def add(split, src, cond, kind, path_in, label, group, orig_fmt, ow, oh, **kw):
        idx = len(rows)
        out_path = kw.get("out_path")
        rows.append(dict(view_key=f"{split}|{src}|{cond}", split=split, source_key=src, group_id=group,
                         label=label, condition=cond, kind=kind,
                         path=(out_path if kind == "degrade" else path_in),
                         q=kw.get("q", -1), scale=kw.get("scale", 1.0), orig_format=orig_fmt,
                         orig_w=ow, orig_h=oh))
        jobs.append(dict(idx=idx, path_in=path_in, kind=kind, out_path=out_path,
                         q=kw.get("q"), scale=kw.get("scale"), rs=kw.get("rs")))

    cs = m[m.split == "calib_src"].reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    dbg_keys = set()
    for lab in (0, 1):
        ks = cs[cs.label == lab].source_key.values
        dbg_keys |= set(rng.choice(ks, 32, replace=False))
    for r in (cs.itertuples() if a.part == "train" else []):
        split = "debug" if r.source_key in dbg_keys else "train"
        pin = os.path.join(DATA, r.relpath)
        add(split, r.source_key, "orig", "file", pin, r.label, r.group_id, r.format, r.width, r.height)
        for k in range(K_TRAIN_VIEWS):
            q, scale, rs = train_view_params(r.source_key, k)
            outp = os.path.join(DATA, "views", split, f"{r.source_key.replace('/', '_')}_d{k}.jpg")
            add(split, r.source_key, f"d{k}", "degrade", pin, r.label, r.group_id, r.format, r.width, r.height,
                out_path=outp, q=q, scale=scale, rs=rs)

    # ---- dev ----
    dv = m[m.split.isin(["explore", "target_dev"])]
    orig = dv[dv.condition == "original"].set_index("source_key")
    trans = dv[dv.condition == "transfer"].set_index("source_key")
    for sk, r in (orig.iterrows() if a.part == "dev" else []):
        g = r.group_id
        pin = os.path.join(DATA, r.relpath)
        add("dev", sk, "orig", "file", pin, r.label, g, r.format, r.width, r.height)
        pt = os.path.join(DATA, trans.loc[sk].relpath)
        add("dev", sk, "transfer", "file", pt, r.label, g, r.format, r.width, r.height)
        for q in (75, 40):
            outp = os.path.join(DATA, "views", "dev", f"{sk.replace('/', '_')}_q{q}.jpg")
            add("dev", sk, f"q{q}", "degrade", pin, r.label, g, r.format, r.width, r.height,
                out_path=outp, q=q, scale=1.0, rs=1)

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
