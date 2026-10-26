#!/usr/bin/env python3
"""受控模拟分析：同一批源图，real 与 ai 经过完全相同的处理，AUROC 随处理强度如何变化。
对照：真实 transfer 条件（同一批源图的实际传播图）。置信区间按 group_id 重采样 500 次。
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "results", "predictions")
OUT = os.path.join(ROOT, "results", "metrics")
MODELS = ["clip", "bfree", "npr"]
ORDER = ["original", "jpeg_q90", "jpeg_q75", "jpeg_q55", "jpeg_q40", "jpeg_q30", "jpeg_q20", "crop874_q40", "transfer(real)"]
B = 500
rng = np.random.RandomState(20261027)


def auc(y, s):
    return roc_auc_score(y, s)


def main():
    thr = json.load(open(f"{OUT}/locked_thresholds.json"))
    rows, rows_shift = [], []
    for m in MODELS:
        d0 = pd.concat([pd.read_csv(f"{P}/{sp}_{m}.csv") for sp in ["explore", "target_dev"]])
        sim = pd.read_csv(f"{P}/sim_{m}.csv")
        d0 = d0.assign(condition=d0.condition.replace({"transfer": "transfer(real)"}))
        d = pd.concat([d0[["source_key", "label", "group_id", "condition", "raw_score", "format"]],
                       sim[["source_key", "label", "group_id", "condition", "raw_score", "format"]]])
        W = d.pivot(index="source_key", columns="condition", values="raw_score")
        meta = d0[d0.condition == "original"].drop_duplicates("source_key").set_index("source_key").loc[W.index]
        y, g = meta.label.values, meta.group_id.values
        ug, inv = np.unique(g, return_inverse=True)
        mem = [np.where(inv == k)[0] for k in range(len(ug))]
        bidx = [np.concatenate([mem[k] for k in rng.randint(0, len(ug), len(ug))]) for _ in range(B)]
        base = W["original"].values
        for c in ORDER:
            s = W[c].values
            a = auc(y, s)
            ba = [auc(y[i], s[i]) for i in bidx]
            bd = [auc(y[i], s[i]) - auc(y[i], base[i]) for i in bidx]
            r = dict(model=m, condition=c, n_real=int((y == 0).sum()), n_ai=int(y.sum()), AUROC=a, lo=np.percentile(ba, 2.5), hi=np.percentile(ba, 97.5),
                     delta_vs_original=a - auc(y, base), d_lo=np.percentile(bd, 2.5), d_hi=np.percentile(bd, 97.5))
            for tn in ["T_default", "T_fpr5"]:
                r[f"TPR_{tn}"] = float((s[y == 1] > thr[m][tn]).mean())
                r[f"FPR_{tn}"] = float((s[y == 0] > thr[m][tn]).mean())
            rows.append(r)
            for lab, nm in [(0, "real"), (1, "ai")]:
                rows_shift.append(dict(model=m, condition=c, cls=nm, median=float(np.median(s[y == lab]))))
        # 同源配对：真实 transfer 与"同强度模拟"的分数一致性（仅中心裁切者除外，不能一一对应的略去）
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/sim_dose_response.csv", index=False)
    pd.DataFrame(rows_shift).to_csv(f"{OUT}/sim_score_shift.csv", index=False)
    pd.set_option("display.width", 220)
    R["AUROC [95%CI]"] = R.apply(lambda x: f"{x.AUROC:.3f} [{x.lo:.3f},{x.hi:.3f}]", axis=1)
    R["Δ vs 原图"] = R.apply(lambda x: f"{x.delta_vs_original:+.3f} [{x.d_lo:+.3f},{x.d_hi:+.3f}]" if x.condition != "original" else "", axis=1)
    for m in MODELS:
        print(f"\n== {m}  (n_real={int(R[R.model==m].n_real.iloc[0])}, n_ai={int(R[R.model==m].n_ai.iloc[0])})")
        print(R[R.model == m].set_index("condition").loc[ORDER][["AUROC [95%CI]", "Δ vs 原图", "TPR_T_fpr5", "FPR_T_fpr5"]].round(3).to_string())
    sh = pd.DataFrame(rows_shift)
    print("\n分数中位数（real / ai）：")
    print(sh.pivot_table(index=["model", "condition"], columns="cls", values="median").round(2).loc[MODELS, :].to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q = ["original", "jpeg_q90", "jpeg_q75", "jpeg_q55", "jpeg_q40", "jpeg_q30", "jpeg_q20"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for j, m in enumerate(MODELS):
        t = R[R.model == m].set_index("condition")
        x = range(len(q))
        ax[j].errorbar(x, t.loc[q].AUROC, yerr=[t.loc[q].AUROC - t.loc[q].lo, t.loc[q].hi - t.loc[q].AUROC], marker="o", label="same JPEG on real+ai")
        ax[j].axhline(t.loc["transfer(real)"].AUROC, c="r", ls="--", label="actual transfer")
        ax[j].plot([len(q) - 0.7], [t.loc["crop874_q40"].AUROC], "s", c="g", label="crop0.874+q40")
        ax[j].axhline(0.5, c="k", ls=":", lw=.7)
        ax[j].set_xticks(list(x)); ax[j].set_xticklabels(["orig", "q90", "q75", "q55", "q40", "q30", "q20"])
        ax[j].set_title(m); ax[j].set_ylabel("AUROC"); ax[j].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{ROOT}/results/figures/sim_dose_response.png", dpi=130)


if __name__ == "__main__":
    main()
