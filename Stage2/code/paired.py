#!/usr/bin/env python3
"""配对比较：同一批 target_dev 样本上，两个运行的 AUROC 差（B - A），按来源图分组 bootstrap（配对重采样）。"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS

PAIRS = [("M0", "M1_lib"), ("M0", "M2_gsr"), ("M0", "M3_lib_gsr"),
         ("M3_lib_gsr", "M3_noaug"), ("M3_lib_gsr", "M3_prop"), ("M3_lib_gsr", "M3_sem0.1"), ("M3_lib_gsr", "M3_nocap"),
         ("M3_noaug", "M3_prop")]
CONDS = ["orig", "transfer", "q75", "q40", "prop"]
N = 1000
rows = []
cache = {}
def load(n):
    if n not in cache:
        cache[n] = pd.read_csv(os.path.join(RESULTS, "runs", n, "pred_target_dev.csv"))
    return cache[n]
for a, b in PAIRS:
    pa, pb = load(a), load(b)
    for c in CONDS:
        da = pa[pa.condition == c].set_index("source_key").sort_index()
        db = pb[pb.condition == c].set_index("source_key").sort_index()
        assert (da.index == db.index).all()
        y = da.label.values; sa, sb = da.logit.values, db.logit.values
        groups = da.group_id.values
        ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
        rng = np.random.RandomState(0); d = []
        for _ in range(N):
            ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
            if len(np.unique(y[ii])) == 2:
                d.append(roc_auc_score(y[ii], sb[ii]) - roc_auc_score(y[ii], sa[ii]))
        diff = roc_auc_score(y, sb) - roc_auc_score(y, sa)
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(A=a, B=b, condition=c, auroc_A=roc_auc_score(y, sa), auroc_B=roc_auc_score(y, sb),
                         diff_B_minus_A=diff, lo=lo, hi=hi, excludes_zero=bool(lo > 0 or hi < 0)))
df = pd.DataFrame(rows)
os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
df.round(4).to_csv(os.path.join(RESULTS, "tables", "paired_comparisons.csv"), index=False)
print(df.round(4).to_string(index=False))
