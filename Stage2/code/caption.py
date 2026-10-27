#!/usr/bin/env python3
"""用冻结 BLIP-2 (opt-2.7b) 为每个视图生成内容描述并缓存。

固定配置：无文本提示（纯图像条件）、贪心解码、max_new_tokens=30、fp16。
描述从"实际收到的视图文件"生成（原图/传播图/退化图分别生成），不接触标签、生成器名称。
"""
import argparse
import os
import sys

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
from common import BLIP2_DIR, CAPTIONS_CSV, PARTS, views_csv  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
GEN_CFG = dict(do_sample=False, num_beams=1, max_new_tokens=30)


class DS(Dataset):
    def __init__(self, df, proc):
        self.df, self.proc = df.reset_index(drop=True), proc

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path)
        img = img.convert("RGB")
        px = self.proc.image_processor(img, return_tensors="pt").pixel_values[0]
        return px, r.view_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from transformers import Blip2ForConditionalGeneration, Blip2Processor
    views = pd.concat([pd.read_csv(views_csv(p)) for p in PARTS if os.path.exists(views_csv(p))])
    done = {}
    if os.path.exists(CAPTIONS_CSV):
        old = pd.read_csv(CAPTIONS_CSV, keep_default_na=False)
        done = dict(zip(old.view_key, old.caption))
    todo = views[~views.view_key.isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"总视图 {len(views)}，已有 {len(done)}，待生成 {len(todo)}", flush=True)
    if len(todo) == 0:
        return
    proc = Blip2Processor.from_pretrained(BLIP2_DIR)
    model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_DIR, torch_dtype=torch.float16).cuda().eval()
    dl = DataLoader(DS(todo, proc), batch_size=a.bs, num_workers=a.workers)
    new, n = [], 0
    with torch.no_grad():
        for px, keys in dl:
            out = model.generate(pixel_values=px.cuda().half(), **GEN_CFG)
            caps = proc.batch_decode(out, skip_special_tokens=True)
            new += [(k, c.strip()) for k, c in zip(keys, caps)]
            n += len(keys)
            if (n // a.bs) % 20 == 0:
                print(n, "/", len(todo), new[-1], flush=True)
                pd.DataFrame(list(done.items()) + new, columns=["view_key", "caption"]).to_csv(CAPTIONS_CSV, index=False)
    pd.DataFrame(list(done.items()) + new, columns=["view_key", "caption"]).to_csv(CAPTIONS_CSV, index=False)
    print("完成", n)


if __name__ == "__main__":
    main()
