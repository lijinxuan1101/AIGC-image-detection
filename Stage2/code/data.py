"""数据读取：视图清单 + 256x256 像素缓存 + BLIP-2 描述缓存。"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CAPTIONS_CSV, PARTS, pix_path, views_csv  # noqa: E402


class Views:
    """把各 part 的 views csv / 像素 memmap 拼成统一索引；(part, pix_idx) -> 像素。"""

    def __init__(self, parts=PARTS):
        dfs, self.pix = [], {}
        for p in parts:
            if not os.path.exists(views_csv(p)):
                continue
            d = pd.read_csv(views_csv(p))
            d["part"] = p
            dfs.append(d)
            self.pix[p] = np.load(pix_path(p), mmap_mode="r")
        self.df = pd.concat(dfs, ignore_index=True)
        cap = pd.read_csv(CAPTIONS_CSV, keep_default_na=False)
        self.cap = dict(zip(cap.view_key, cap.caption))
        self.df["caption"] = self.df.view_key.map(self.cap)
        # dev 分为 explore（验证：选 checkpoint / 阈值）与 target_dev（开发测试，不参与选择）
        from common import S1_MANIFEST
        m = pd.read_csv(S1_MANIFEST, usecols=["source_key", "split"]).drop_duplicates("source_key")
        sub = dict(zip(m.source_key, m.split))
        self.df["subset"] = [sub.get(k, "") if s == "dev" else s
                             for k, s in zip(self.df.source_key, self.df.split)]

    def get(self, rows):
        """rows: DataFrame 行子集 -> (uint8 tensor [B,H,W,3], captions)。"""
        pix = np.stack([self.pix[r.part][r.pix_idx] for r in rows.itertuples()])
        caps = [self.cap.get(k, "") for k in rows.view_key]
        return torch.from_numpy(pix), caps


class PairSampler:
    """标签平衡的训练采样：每个 batch 一半 real 一半 ai；每个样本返回 (原图视图, 随机退化视图)。

    退化视图只由 source 的 d0..dK 随机选择，与标签无关。
    """

    def __init__(self, views: Views, split="train", seed=0, deg_prefix="d"):
        """deg_prefix: 'd' = 普通增强视图 d0..；'p' = 随机传播视图 p0..。"""
        df = views.df[views.df.split == split]
        orig = df[df.condition == "orig"].set_index("source_key")
        self.orig = orig
        dd = df[df.condition.str.match(rf"^{deg_prefix}\d+$")]
        self.deg = {k: g for k, g in dd.groupby("source_key")}
        self.keys = {lab: orig.index[orig.label == lab].tolist() for lab in (0, 1)}
        self.rng = np.random.RandomState(seed)
        self.views = views

    def epoch_len(self, batch):                      # 以数量较少的类为一个 epoch
        return min(len(v) for v in self.keys.values()) // (batch // 2)

    def batches(self, batch, n_batches):
        """生成 n_batches 个 batch（每个 batch 是 (原图行 DataFrame, 退化行 DataFrame)）。"""
        h = batch // 2
        pools = {lab: [] for lab in (0, 1)}
        for _ in range(n_batches):
            sel = []
            for lab in (0, 1):
                if len(pools[lab]) < h:
                    pools[lab] += list(self.rng.permutation(self.keys[lab]))
                sel += [pools[lab].pop() for _ in range(h)]
            o = self.orig.loc[sel].reset_index()
            d = []
            for k in sel:
                g = self.deg[k]
                d.append(g.iloc[self.rng.randint(len(g))])
            yield o, pd.DataFrame(d).reset_index(drop=True)
