#!/usr/bin/env python3
"""M5 诊断分析。协议在查看 closed_test 结果前固定（见 manifests/protocol_v1.json 与 docs/M5 协议段）。

阈值：
  T_default   : 0（官方默认，logit>0 判生成）
  T_fpr5/T_fpr1: 在 calib_src 的原图 real 上取 95%/99% 分位（即源域 FPR=5%/1%），锁定
  T_bacc      : calib_src 原图上使平衡准确率最大的阈值，锁定
  T_tdev_fpr5 : E4，仅用 target_dev 传播图 real 的 95% 分位；单列为"目标域校准"，不与无目标域方案混报
评测：closed_test，original 与 transfer。置信区间：按 group_id 重采样 B 次。
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "results", "predictions")
OUT = os.path.join(ROOT, "results", "metrics")
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
MODELS = ["clip", "bfree", "npr"]
B = 500
rng = np.random.RandomState(20261026)


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def tpr_at_fpr(y, s, f):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(f, fpr, tpr))


def op(y, s, t):
    pred = s > t
    tpr = pred[y == 1].mean()
    fpr = pred[y == 0].mean()
    return dict(TPR=tpr, FPR=fpr, BalAcc=(tpr + 1 - fpr) / 2)


def boot_idx(groups, n_boot):
    """按组重采样：返回 n_boot 个行索引数组"""
    ug, inv = np.unique(groups, return_inverse=True)
    members = [[] for _ in ug]
    for i, g in enumerate(inv):
        members[g].append(i)
    members = [np.array(m) for m in members]
    out = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(ug), len(ug))
        out.append(np.concatenate([members[k] for k in pick]))
    return out


def ci(vals):
    v = np.asarray(vals, dtype=float)
    v = v[~np.isnan(v)]
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)


def main():
    thr = {}
    rows_main, rows_op, rows_shift, rows_flip, rows_e4, rows_strat = [], [], [], [], [], []
    store = {}
    for m in MODELS:
        cal = pd.read_csv(f"{P}/calib_src_{m}.csv")
        cal = cal[cal.condition == "original"]
        yc, sc = cal.label.values, cal.raw_score.values
        fpr, tpr, th = roc_curve(yc, sc)
        bacc = (tpr + 1 - fpr) / 2
        thr[m] = dict(T_default=0.0, T_fpr5=float(np.percentile(sc[yc == 0], 95)), T_fpr1=float(np.percentile(sc[yc == 0], 99)),
                      T_bacc=float(th[np.argmax(bacc)]), calib_auc=float(auc(yc, sc)), n_calib_real=int((yc == 0).sum()),
                      n_calib_fake=int((yc == 1).sum()))
        tdev = pd.read_csv(f"{P}/target_dev_{m}.csv")
        tt = tdev[tdev.condition == "transfer"]
        thr[m]["T_tdev_fpr5"] = float(np.percentile(tt[tt.label == 0].raw_score, 95))
        thr[m]["T_tdev_bacc"] = float(roc_curve(tt.label, tt.raw_score)[2][np.argmax(((lambda f, t: (t + 1 - f) / 2)(*roc_curve(tt.label, tt.raw_score)[:2])))])
    json.dump(thr, open(f"{OUT}/locked_thresholds.json", "w"), indent=1)
    print(json.dumps(thr, indent=1))

    for m in MODELS:
        d = pd.read_csv(f"{P}/closed_test_{m}.csv")
        o = d[d.condition == "original"].set_index("source_key")
        t = d[d.condition == "transfer"].set_index("source_key")
        keys = o.index.intersection(t.index)
        o, t = o.loc[keys], t.loc[keys]
        pair = pd.DataFrame(dict(y=o.label.values, s_o=o.raw_score.values, s_t=t.raw_score.values, group=o.group_id.values,
                                 scen=o.scenario.values, w=o.width.values, h=o.height.values, fmt=o.format.values,
                                 tw=t.width.values, th=t.height.values, key=keys))
        pair["maxside"] = pair[["w", "h"]].max(axis=1)
        pair["scaled"] = (pair.tw / pair.w) < 0.98
        store[m] = pair
        y = pair.y.values
        bidx = boot_idx(pair.group.values, B)

        # ---- 排序指标 + CI
        for cond, col in [("original", "s_o"), ("transfer", "s_t")]:
            s = pair[col].values
            bs = dict(AUROC=[], AP=[], TPR_at_FPR1=[], TPR_at_FPR5=[])
            for ix in bidx:
                yy, ss = y[ix], s[ix]
                bs["AUROC"].append(auc(yy, ss))
                bs["AP"].append(average_precision_score(yy, ss))
                bs["TPR_at_FPR1"].append(tpr_at_fpr(yy, ss, 0.01))
                bs["TPR_at_FPR5"].append(tpr_at_fpr(yy, ss, 0.05))
            r = dict(model=m, condition=cond, n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()), prevalence=float(y.mean()))
            for k, f in [("AUROC", auc(y, s)), ("AP", average_precision_score(y, s)), ("TPR_at_FPR1", tpr_at_fpr(y, s, 0.01)),
                         ("TPR_at_FPR5", tpr_at_fpr(y, s, 0.05))]:
                lo, hi = ci(bs[k])
                r[k], r[k + "_lo"], r[k + "_hi"] = f, lo, hi
            rows_main.append(r)
        # AUROC 下降值的 CI（配对 bootstrap）
        dd = [auc(y[ix], pair.s_t.values[ix]) - auc(y[ix], pair.s_o.values[ix]) for ix in bidx]
        rows_main.append(dict(model=m, condition="delta(transfer-original)", AUROC=auc(y, pair.s_t.values) - auc(y, pair.s_o.values),
                              AUROC_lo=ci(dd)[0], AUROC_hi=ci(dd)[1]))

        # ---- 固定阈值运行点
        for tn in ["T_default", "T_fpr5", "T_fpr1", "T_bacc"]:
            for cond, col in [("original", "s_o"), ("transfer", "s_t")]:
                s = pair[col].values
                base = op(y, s, thr[m][tn])
                bs = [op(y[ix], s[ix], thr[m][tn]) for ix in bidx[:200]]
                r = dict(model=m, threshold=tn, thr_value=thr[m][tn], condition=cond, **base)
                for k in ["TPR", "FPR", "BalAcc"]:
                    r[k + "_lo"], r[k + "_hi"] = ci([b[k] for b in bs])
                rows_op.append(r)
            # 翻转率（同一源图）
            po, pt = pair.s_o.values > thr[m][tn], pair.s_t.values > thr[m][tn]
            rows_flip.append(dict(model=m, threshold=tn,
                                  real_to_fake=float((~po & pt)[y == 0].mean()), real_stays_fake=float((po & pt)[y == 0].mean()),
                                  fake_to_real=float((po & ~pt)[y == 1].mean()), fake_stays_real=float((~po & ~pt)[y == 1].mean()),
                                  real_flip_total=float((po != pt)[y == 0].mean()), fake_flip_total=float((po != pt)[y == 1].mean())))

        # ---- 分数位移
        for lab, nm in [(0, "real"), (1, "fake")]:
            so, st = pair.s_o.values[y == lab], pair.s_t.values[y == lab]
            dlt = st - so
            rows_shift.append(dict(model=m, cls=nm, med_orig=np.median(so), med_transfer=np.median(st), iqr_orig=np.subtract(*np.percentile(so, [75, 25])),
                                   iqr_transfer=np.subtract(*np.percentile(st, [75, 25])), med_delta=np.median(dlt),
                                   delta_p10=np.percentile(dlt, 10), delta_p90=np.percentile(dlt, 90),
                                   spearman_orig_vs_transfer=pd.Series(so).corr(pd.Series(st), method="spearman")))

        # ---- E4：目标域校准（target_dev 传播图选阈值，closed_test 传播图评测）
        for tn in ["T_tdev_fpr5", "T_tdev_bacc"]:
            r = dict(model=m, threshold=tn, thr_value=thr[m][tn], **{f"transfer_{k}": v for k, v in op(y, pair.s_t.values, thr[m][tn]).items()})
            rows_e4.append(r)

        # ---- 分层（控制分析）
        def strat(name, mask_fn, levels):
            for lv in levels:
                mk = mask_fn(lv)
                yy = y[mk]
                if yy.sum() < 30 or (len(yy) - yy.sum()) < 30:
                    rows_strat.append(dict(model=m, stratum=name, level=lv, n_real=int((yy == 0).sum()), n_fake=int(yy.sum()), note="样本不足"))
                    continue
                so, st = pair.s_o.values[mk], pair.s_t.values[mk]
                rows_strat.append(dict(model=m, stratum=name, level=lv, n_real=int((yy == 0).sum()), n_fake=int(yy.sum()),
                                       AUROC_orig=auc(yy, so), AUROC_transfer=auc(yy, st), delta=auc(yy, st) - auc(yy, so),
                                       TPR_fpr5_orig=op(yy, so, thr[m]["T_fpr5"])["TPR"], FPR_fpr5_orig=op(yy, so, thr[m]["T_fpr5"])["FPR"],
                                       TPR_fpr5_tr=op(yy, st, thr[m]["T_fpr5"])["TPR"], FPR_fpr5_tr=op(yy, st, thr[m]["T_fpr5"])["FPR"]))
        bins = pd.cut(pair.maxside, [0, 300, 600, 1100, 1600, 1e9], labels=["<=300", "301-600", "601-1100", "1101-1600", ">1600"])
        strat("原图最长边", lambda lv: (bins == lv).values, ["<=300", "301-600", "601-1100", "1101-1600", ">1600"])
        strat("原图格式", lambda lv: (pair.fmt == lv).values, ["JPEG", "PNG"])
        strat("传播是否缩放", lambda lv: (pair.scaled == lv).values, [False, True])
        # 仅 JPEG 原图、同尺寸区间：把"格式"与"尺寸"一并控制
        strat("仅JPEG原图且最长边601-1100", lambda lv: ((pair.fmt == "JPEG") & (bins == "601-1100")).values, ["x"])
        # 场景（real 无场景，用全部 real 作负类，仅作参考）
        for sc in sorted(set(pair.scen[pair.y == 1])):
            mk = ((pair.y == 0) | (pair.scen == sc)).values
            yy = y[mk]
            rows_strat.append(dict(model=m, stratum="ai场景(负类=全部real)", level=sc, n_real=int((yy == 0).sum()), n_fake=int(yy.sum()),
                                   AUROC_orig=auc(yy, pair.s_o.values[mk]), AUROC_transfer=auc(yy, pair.s_t.values[mk]),
                                   delta=auc(yy, pair.s_t.values[mk]) - auc(yy, pair.s_o.values[mk])))

    pd.DataFrame(rows_main).to_csv(f"{OUT}/rank_metrics.csv", index=False)
    pd.DataFrame(rows_op).to_csv(f"{OUT}/operating_points.csv", index=False)
    pd.DataFrame(rows_shift).to_csv(f"{OUT}/score_shift.csv", index=False)
    pd.DataFrame(rows_flip).to_csv(f"{OUT}/flip_rates.csv", index=False)
    pd.DataFrame(rows_e4).to_csv(f"{OUT}/e4_target_calibration.csv", index=False)
    pd.DataFrame(rows_strat).to_csv(f"{OUT}/stratified.csv", index=False)

    # ---- 仅元数据基线（在 calib_src 训练，测 closed_test）：量化"捷径"能到什么程度
    from sklearn.ensemble import GradientBoostingClassifier
    man = pd.read_csv(os.path.join(ROOT, "manifests", "manifest.csv"))
    inv = pd.read_csv(os.path.join(ROOT, "results", "audit", "inventory.csv"))[["variant_id_tmp"]] if False else None
    man["is_png"] = (man.format == "PNG").astype(int)
    man["bpp"] = man.bytes / (man.width * man.height)
    man["aspect"] = man.width / man.height
    feats = ["is_png", "width", "height", "bytes", "bpp", "aspect"]
    tr = man[(man.split == "calib_src")]
    clf = GradientBoostingClassifier(n_estimators=150, random_state=0).fit(tr[feats], tr.label)
    rows = []
    for cond in ["original", "transfer"]:
        te = man[(man.split == "closed_test") & (man.condition == cond)]
        p = clf.predict_proba(te[feats])[:, 1]
        rows.append(dict(condition=cond, metadata_only_AUROC=auc(te.label.values, p)))
    pd.DataFrame(rows).to_csv(f"{OUT}/metadata_only_baseline.csv", index=False)
    print(pd.DataFrame(rows))

    # ---- 图
    lab = dict(clip="CLIP 2024", bfree="B-Free", npr="NPR")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for j, m in enumerate(MODELS):
        pr = store[m]
        for cond, col, c in [("original", "s_o", "tab:blue"), ("transfer", "s_t", "tab:red")]:
            f, t, _ = roc_curve(pr.y, pr[col])
            ax[0, j].plot(f, t, c=c, label=f"{cond} AUROC={auc(pr.y.values, pr[col].values):.3f}")
            ax[1, j].plot(f, t, c=c)
        ax[0, j].plot([0, 1], [0, 1], "k:", lw=.7)
        ax[0, j].set_title(lab[m]); ax[0, j].legend(loc="lower right"); ax[0, j].set_xlabel("FPR"); ax[0, j].set_ylabel("TPR")
        ax[1, j].set_xscale("log"); ax[1, j].set_xlim(1e-3, 1); ax[1, j].set_xlabel("FPR (log)"); ax[1, j].set_ylabel("TPR"); ax[1, j].grid(alpha=.3)
        ax[1, j].set_title("low-FPR region")
    plt.tight_layout(); plt.savefig(f"{FIG}/roc_original_vs_transfer.png", dpi=130); plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for j, m in enumerate(MODELS):
        pr = store[m]
        lo, hi = np.percentile(np.r_[pr.s_o, pr.s_t], [0.5, 99.5])
        bins_ = np.linspace(lo, hi, 70)
        for i, (cond, col) in enumerate([("original", "s_o"), ("transfer", "s_t")]):
            ax[i, j].hist(pr[col][pr.y == 0], bins_, alpha=.55, label="real", density=True)
            ax[i, j].hist(pr[col][pr.y == 1], bins_, alpha=.55, label="fake", density=True)
            for tn, ls in [("T_default", "-"), ("T_fpr5", "--")]:
                ax[i, j].axvline(thr[m][tn], c="k", ls=ls, lw=.8)
            ax[i, j].set_title(f"{lab[m]} / {cond}"); ax[i, j].legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/score_distributions.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for j, m in enumerate(MODELS):
        pr = store[m].sample(4000, random_state=0)
        ax[j].scatter(pr.s_o[pr.y == 0], pr.s_t[pr.y == 0], s=3, alpha=.35, label="real")
        ax[j].scatter(pr.s_o[pr.y == 1], pr.s_t[pr.y == 1], s=3, alpha=.35, label="fake")
        lim = np.percentile(np.r_[pr.s_o, pr.s_t], [0.5, 99.5]); ax[j].plot(lim, lim, "k:", lw=.7)
        ax[j].set_xlim(lim); ax[j].set_ylim(lim); ax[j].set_xlabel("original score"); ax[j].set_ylabel("transfer score"); ax[j].set_title(lab[m]); ax[j].legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/paired_scores.png", dpi=130); plt.close()
    print("done")


if __name__ == "__main__":
    main()
