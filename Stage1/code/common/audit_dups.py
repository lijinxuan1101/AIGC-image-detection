#!/usr/bin/env python3
"""M1：精确重复（sha1）与近重复（dHash 汉明距离）审计。输出 results/audit/dup_*.csv"""
import os, warnings
import numpy as np, pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "extracted"); OUT = os.path.join(ROOT, "results", "audit")

def dhash(p, n=16):
    with Image.open(p) as im:
        im.draft("L", (n * 8, n * 8))
        a = np.asarray(im.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.float32)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())  # 256 bit

if __name__ == "__main__":
    inv = pd.read_csv(os.path.join(OUT, "inventory.csv"))
    o = inv[inv.condition == "original"].reset_index(drop=True)   # 只看原图条件
    print("原图条件图像数", len(o))
    ex = o[o.duplicated("sha1", keep=False)].sort_values("sha1")
    ex.to_csv(os.path.join(OUT, "dup_exact.csv"), index=False)
    print("精确重复（sha1）涉及图像:", len(ex), "| 跨包:", int(ex.groupby("sha1").package.nunique().gt(1).sum()),
          "| 跨标签:", int(ex.groupby("sha1").label.nunique().gt(1).sum()))
    with ProcessPoolExecutor(32) as pool:
        H = np.stack(list(pool.map(dhash, [os.path.join(DATA, r) for r in o.relpath], chunksize=64)))
    import torch
    t = torch.from_numpy(H).cuda()  # [N,32] uint8
    pop = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.uint8, device="cuda")
    N = len(o); pairs = []
    for s in range(0, N, 1024):
        x = t[s:s + 1024, None, :] ^ t[None, :, :]
        d = pop[x.long()].sum(-1, dtype=torch.int32)  # [b,N]
        idx = (d <= 16).nonzero()
        for i, j in idx.tolist():
            if s + i < j: pairs.append((s + i, j, int(d[i, j])))
    P = pd.DataFrame(pairs, columns=["i", "j", "dist"])
    for c in ["package", "label", "scenario", "source_id", "relpath"]:
        P[c + "_a"] = o[c].values[P.i]; P[c + "_b"] = o[c].values[P.j]
    P.to_csv(os.path.join(OUT, "dup_near.csv"), index=False)
    print("近重复对（256bit dHash 距离<=16）:", len(P))
    P["xpkg"] = P.package_a != P.package_b; P["xlabel"] = P.label_a != P.label_b
    print(P.groupby(["xpkg", "xlabel"]).size())
    print(P.groupby(pd.cut(P.dist, [-1, 0, 4, 8, 16])).size())
