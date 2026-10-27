#!/usr/bin/env python3
"""汇总 Stage 2 结果：

  1. 各运行在 target_dev（开发测试，不参与选择）上的逐条件指标 + 按来源分组 bootstrap CI；
  2. 阈值 T 只在 explore 上选定（跨条件宏平均平衡准确率最大），同时报告默认阈值 0；
  3. 捷径基线：仅元数据（格式/宽高/字节数）与仅描述文本（TF-IDF+LR），在 train 上训练、dev 上测试；
  4. Stage1 三个官方基线在同一 target_dev 上的原图/传播图 AUROC；
  5. 逐格式分层（原图为 JPEG 的子集），观察格式捷径。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, ROOT  # noqa: E402
from data import Views  # noqa: E402
from evaluate import bootstrap_auc, op_metrics, rank_table  # noqa: E402

CONDS = ["orig", "transfer", "q75", "q40", "prop"]
OUT = os.path.join(RESULTS, "tables")
os.makedirs(OUT, exist_ok=True)


def pick_threshold(pred_val):
    """explore 上：使 4 个已知开发条件的平衡准确率均值最大的 logit 阈值。"""
    d = pred_val[pred_val.condition.isin(["orig", "transfer", "q75", "q40"])]
    cand = np.unique(np.quantile(d.logit.values, np.linspace(0.02, 0.98, 97)))
    best = max(cand, key=lambda t: np.mean([op_metrics(g.label.values, g.logit.values, t)["bacc"]
                                            for _, g in d.groupby("condition")]))
    return float(best)


def run_table(name, run_dir):
    pv = pd.read_csv(os.path.join(run_dir, "pred_explore.csv"))
    pt_path = os.path.join(run_dir, "pred_target_dev.csv")
    if not os.path.exists(pt_path):
        return None
    pt = pd.read_csv(pt_path)
    T = pick_threshold(pv)
    rows = []
    for c in CONDS:
        d = pt[pt.condition == c]
        if len(d) == 0:
            continue
        t0 = rank_table(pt, [c], 0.0).iloc[0]
        t1 = rank_table(pt, [c], T).iloc[0]
        lo, hi = bootstrap_auc(pt, c, n=300)
        rows.append(dict(run=name, condition=c, n_real=t0.n_real, n_fake=t0.n_fake, auroc=t0.auroc,
                         auroc_lo=lo, auroc_hi=hi, ap=t0.ap, tpr_at_1fpr=t0.tpr_at_1fpr,
                         tpr_default=t0.tpr, fpr_default=t0.fpr, bacc_default=t0.bacc,
                         T_explore=T, tpr_T=t1.tpr, fpr_T=t1.fpr, bacc_T=t1.bacc))
    return pd.DataFrame(rows)


def shortcut_baselines(views):
    df = views.df.copy()
    df["fmt"] = np.where(df.kind.isin(["degrade", "prop"]) | (df.condition == "transfer"), "JPEG", df.orig_format)
    df["is_png"] = (df.fmt == "PNG").astype(int)
    df["npx"] = df.w * df.h
    df["bpp"] = df.bytes / df.npx
    df["aspect"] = df.w / df.h
    feats = ["is_png", "w", "h", "npx", "bytes", "bpp", "aspect"]
    tr = df[df.split == "train"]
    dv = df[(df.split == "dev") & (df.subset == "target_dev")]
    out = []
    g = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(tr[feats], tr.label)
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    lr = LogisticRegression(max_iter=2000, C=2.0).fit(v.fit_transform(tr.caption.fillna("")), tr.label)
    for c in CONDS:
        d = dv[dv.condition == c]
        if len(d) == 0:
            continue
        out.append(dict(condition=c,
                        metadata_only_auroc=roc_auc_score(d.label, g.predict_proba(d[feats])[:, 1]),
                        caption_only_auroc=roc_auc_score(d.label, lr.decision_function(v.transform(d.caption.fillna(""))))))
    return pd.DataFrame(out)


def stage1_baselines():
    rows = []
    for m in ("clip", "bfree", "npr"):
        p = pd.read_csv(os.path.join(ROOT, "..", "Stage1", "results", "predictions", f"target_dev_{m}.csv"))
        for c in ("original", "transfer"):
            d = p[p.condition == c]
            rows.append(dict(model=f"stage1_{m}", condition="orig" if c == "original" else c,
                             auroc=roc_auc_score(d.label, d.raw_score), n=len(d)))
    return pd.DataFrame(rows)


def jpeg_only_strata(run, run_dir):
    pt = pd.read_csv(os.path.join(run_dir, "pred_target_dev.csv"))
    rows = []
    for c in CONDS:
        d = pt[(pt.condition == c)]
        j = d[(d.orig_format == "JPEG")]
        if j.label.nunique() == 2:
            rows.append(dict(run=run, condition=c, subset="orig_format=JPEG", n_real=int((j.label == 0).sum()),
                             n_fake=int((j.label == 1).sum()), auroc=roc_auc_score(j.label, j.logit)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    a = ap.parse_args()
    runs_dir = os.path.join(RESULTS, "runs")
    names = a.runs or sorted(n for n in os.listdir(runs_dir) if os.path.exists(os.path.join(runs_dir, n, "pred_target_dev.csv")))
    tabs, strata = [], []
    for n in names:
        t = run_table(n, os.path.join(runs_dir, n))
        if t is not None:
            tabs.append(t)
            strata.append(jpeg_only_strata(n, os.path.join(runs_dir, n)))
    if tabs:
        T = pd.concat(tabs)
        T.to_csv(os.path.join(OUT, "target_dev_runs.csv"), index=False)
        print(T.round(3).to_string(index=False))
        pd.concat(strata).to_csv(os.path.join(OUT, "target_dev_jpeg_only.csv"), index=False)
        print(pd.concat(strata).round(3).to_string(index=False))
    views = Views()
    sb = shortcut_baselines(views)
    sb.to_csv(os.path.join(OUT, "shortcut_baselines.csv"), index=False)
    print(sb.round(3).to_string(index=False))
    s1 = stage1_baselines()
    s1.to_csv(os.path.join(OUT, "stage1_baselines_target_dev.csv"), index=False)
    print(s1.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
