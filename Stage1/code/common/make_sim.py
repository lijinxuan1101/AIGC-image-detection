#!/usr/bin/env python3
"""受控模拟：对同一批源图的 real 与 ai 原图施加完全相同、参数已知的处理。

由审计得知 transfer 条件可逐像素复现为：[可选：正中心裁切为原图宽高的 ~0.85-0.90] + PIL JPEG(quality=整数 20-55)。
这里用同一套流程做剂量-响应：
  jpeg_q{90,75,55,40,30,20} : 仅 JPEG，固定质量，不裁切
  crop874_q40               : 中心裁切 0.874 + JPEG q=40
范围：explore + target_dev 的全部源图（不用 closed_test 做新选择）。
输出：data/extracted/sim/<variant>/<label>/<id>.jpg 与 manifests/manifest_sim.csv
"""
import io
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from PIL import Image

warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "extracted")
VARIANTS = {"jpeg_q90": (90, 1.0), "jpeg_q75": (75, 1.0), "jpeg_q55": (55, 1.0), "jpeg_q40": (40, 1.0),
            "jpeg_q30": (30, 1.0), "jpeg_q20": (20, 1.0), "crop874_q40": (40, 0.874)}


def job(a):
    src, dst, q, crop = a
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = Image.open(src).convert("RGB")
    if crop < 1.0:
        w, h = im.size
        cw, ch = int(round(w * crop)), int(round(h * crop))
        x, y = (w - cw) // 2, (h - ch) // 2
        im = im.crop((x, y, x + cw, y + ch))
    im.save(dst, "JPEG", quality=q)
    return im.size[0], im.size[1], os.path.getsize(dst)


def main():
    man = pd.read_csv(os.path.join(ROOT, "manifests", "manifest.csv"))
    base = man[man.split.isin(["explore", "target_dev"]) & (man.condition == "original")].copy()
    rows, jobs = [], []
    for v, (q, crop) in VARIANTS.items():
        for _, r in base.iterrows():
            rel = f"sim/{v}/{r.label_name}/{r.source_id}.jpg"
            jobs.append((os.path.join(DATA, r.relpath), os.path.join(DATA, rel), q, crop))
            rows.append(dict(source_key=r.source_key, source_id=r.source_id, variant_id=f"{v}|{r.source_key}", split=r.split,
                             package=r.package, label=r.label, scenario=r.scenario, condition=v, group_id=r.group_id,
                             relpath=rel, format="JPEG"))
    with ProcessPoolExecutor(48) as ex:
        out = list(ex.map(job, jobs, chunksize=16))
    df = pd.DataFrame(rows)
    df["width"], df["height"], df["bytes"] = zip(*out)
    df.to_csv(os.path.join(ROOT, "manifests", "manifest_sim.csv"), index=False)
    print("生成", len(df), "张；各变体平均字节：")
    print(df.groupby("condition").bytes.mean().round(0).to_string())


if __name__ == "__main__":
    main()
