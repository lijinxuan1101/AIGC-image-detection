"""推理与指标。"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

DEV_CONDS = ["orig", "transfer", "q75", "q40"]


@torch.no_grad()
def predict(model, views, df, bs=32, log=False):
    """返回 df 副本，附 logit（= z_No - z_Yes）与 p_fake。"""
    was = model.training
    model.eval()
    df = df.reset_index(drop=True)
    out = np.zeros(len(df), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(df), bs):
        rows = df.iloc[i:i + bs]
        pix, caps = views.get(rows)
        out[i:i + bs] = model(pix, caps)["logit"].float().cpu().numpy()
        if log and (i // bs) % 20 == 0:
            print(f"  predict {i}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    model.train(was)
    r = df.copy()
    r["logit"] = out
    r["p_fake"] = 1 / (1 + np.exp(-out))
    return r


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def tpr_at_fpr(y, s, fpr):
    neg = np.sort(s[y == 0])
    thr = neg[min(len(neg) - 1, int(np.ceil(len(neg) * (1 - fpr))))]
    return float((s[y == 1] > thr).mean())


def op_metrics(y, s, thr):
    pred = s > thr
    tpr, tnr = float(pred[y == 1].mean()), float((~pred[y == 0]).mean())
    f1 = 2 * (pred & (y == 1)).sum() / max(1, (pred.sum() + (y == 1).sum()))
    return dict(thr=float(thr), tpr=tpr, fpr=1 - tnr, bacc=(tpr + tnr) / 2, f1=float(f1))


def rank_table(pred, conds=DEV_CONDS, thr=0.0):
    """逐条件的排序与固定阈值指标；thr 为 logit 阈值（默认 0 即 p=0.5）。"""
    rows = []
    for c in conds:
        d = pred[pred.condition == c]
        if len(d) == 0:
            continue
        y, s = d.label.values, d.logit.values
        rows.append(dict(condition=c, n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()),
                         auroc=safe_auc(y, s), ap=float(average_precision_score(y, s)),
                         tpr_at_1fpr=tpr_at_fpr(y, s, 0.01), tpr_at_5fpr=tpr_at_fpr(y, s, 0.05),
                         **op_metrics(y, s, thr)))
    return pd.DataFrame(rows)


def macro_auroc(pred, conds=DEV_CONDS):
    t = rank_table(pred, conds)
    return float(t.auroc.mean()), t


def bootstrap_auc(pred, cond, n=500, seed=0):
    """按 group_id（来源图）重采样的 AUROC 95% 置信区间。"""
    d = pred[pred.condition == cond]
    groups = d.group_id.unique()
    idx = {g: np.where(d.group_id.values == g)[0] for g in groups}
    rng = np.random.RandomState(seed)
    y, s = d.label.values, d.logit.values
    vals = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])
        if len(np.unique(y[ii])) == 2:
            vals.append(roc_auc_score(y[ii], s[ii]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
