#!/usr/bin/env python3
"""受控复核：在格式、尺寸、JPEG 压缩强度匹配的子集上，重做原图 vs 传播图诊断。

预先固定的规则（在查看结果前）：
  范围   : 测试包 explore+target_dev+closed_test 的全部源图；仅保留原图为 JPEG 者（real 与 ai 都是 JPEG）。
  匹配变量: 原图最长边分箱 × 原图 JPEG 亮度量化表之和（压缩强度）5 分位箱。
  方法   : 粗化精确匹配；每个格子两类各随机保留 min(n_real, n_ai) 张。种子固定。
  变体   : strict = 尺寸 × 压缩强度；loose = 仅尺寸。
  验证   : 匹配后用"仅元数据"分类器的 AUROC 检查是否已无可用捷径（应接近 0.5）。
  阈值   : 沿用 calib_src 锁定阈值（locked_thresholds.json），不重选。
  置信区间: 按 group_id 重采样 500 次。
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "results", "predictions")
OUT = os.path.join(ROOT, "results", "metrics")
SEED = 20261027
MODELS = ["clip", "bfree", "npr"]
SIZE_BINS = [0, 300, 400, 512, 600, 800, 1024, 1280, 1600, 2048, 1e9]
B = 500


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def boot_idx(groups, rng, n):
    ug, inv = np.unique(groups, return_inverse=True)
    mem = [[] for _ in ug]
    for i, g in enumerate(inv):
        mem[g].append(i)
    mem = [np.array(m) for m in mem]
    return [np.concatenate([mem[k] for k in rng.randint(0, len(ug), len(ug))]) for _ in range(n)]


def main():
    rng = np.random.RandomState(SEED)
    inv = pd.read_csv(os.path.join(ROOT, "results", "audit", "inventory.csv"))[["relpath", "jpeg_q0_sum"]]
    thr = json.load(open(os.path.join(OUT, "locked_thresholds.json")))
    # 合并三个划分的预测
    base = None
    for m in MODELS:
        d = pd.concat([pd.read_csv(f"{P}/{sp}_{m}.csv") for sp in ["explore", "target_dev", "closed_test"]])
        if base is None:
            base = d[d.condition == "original"][["source_key", "label", "group_id", "format", "width", "height", "bytes", "relpath", "scenario"]].copy()
        o = d[d.condition == "original"].set_index("source_key").raw_score.rename(f"{m}_o")
        t = d[d.condition == "transfer"].set_index("source_key")
        base = base.join(o, on="source_key").join(t.raw_score.rename(f"{m}_t"), on="source_key")
        if m == MODELS[0]:
            base = base.join(t[["width", "height", "bytes", "format"]].add_prefix("t_"), on="source_key")
    base = base.merge(inv, on="relpath", how="left")
    base["maxside"] = base[["width", "height"]].max(axis=1)
    base["sbin"] = pd.cut(base.maxside, SIZE_BINS, labels=False)
    print("测试包源图:", len(base))
    jp = base[base.format == "JPEG"].copy()
    print("原图为 JPEG: real", int((jp.label == 0).sum()), "ai", int((jp.label == 1).sum()))
    # 压缩强度分位箱（以 JPEG 原图整体分位）
    jp["qbin"] = pd.qcut(jp.jpeg_q0_sum, 5, labels=False, duplicates="drop")
    print("JPEG 亮度量化表之和（越大压缩越狠）按类别中位数:\n", jp.groupby("label").jpeg_q0_sum.median().to_string())

    def match(df, keys, tag):
        parts = []
        for _, g in df.groupby(keys):
            r, a = g[g.label == 0], g[g.label == 1]
            k = min(len(r), len(a))
            if k == 0:
                continue
            parts += [r.sample(k, random_state=rng), a.sample(k, random_state=rng)]
        out = pd.concat(parts)
        out["variant"] = tag
        return out

    sets = {"strict(尺寸×压缩强度)": match(jp, ["sbin", "qbin"], "strict"), "loose(仅尺寸)": match(jp, ["sbin"], "loose")}

    # 匹配前对照：全部 JPEG 原图（未匹配）
    sets = {"unmatched_JPEG原图": jp.assign(variant="unmatched"), **sets}

    rows_rank, rows_op, rows_meta = [], [], []
    for name, d in sets.items():
        d = d.reset_index(drop=True)
        y = d.label.values
        print(f"\n== {name}: real {int((y == 0).sum())} / ai {int(y.sum())}")
        # 元数据捷径检验（原图与传播图的元数据；分组 5 折）
        for cond, cols in [("original", ["width", "height", "bytes", "jpeg_q0_sum"]), ("transfer", ["t_width", "t_height", "t_bytes"])]:
            X = d[cols].copy()
            X["bpp"] = X[cols[2]] / (X[cols[0]] * X[cols[1]])
            X["aspect"] = X[cols[0]] / X[cols[1]]
            p = cross_val_predict(GradientBoostingClassifier(n_estimators=100, random_state=0), X.values, y,
                                  cv=GroupKFold(5), groups=d.group_id.values, method="predict_proba")[:, 1]
            rows_meta.append(dict(set=name, condition=cond, n_real=int((y == 0).sum()), n_ai=int(y.sum()), metadata_only_AUROC=auc(y, p)))
            print(f"   元数据-only AUROC [{cond}] = {auc(y, p):.3f}")
        bidx = boot_idx(d.group_id.values, rng, B)
        for m in MODELS:
            so, st = d[f"{m}_o"].values, d[f"{m}_t"].values
            ao, at = auc(y, so), auc(y, st)
            bo = [auc(y[ix], so[ix]) for ix in bidx]
            bt = [auc(y[ix], st[ix]) for ix in bidx]
            bd = [t - o for o, t in zip(bo, bt)]
            rows_rank.append(dict(set=name, model=m, n_real=int((y == 0).sum()), n_ai=int(y.sum()), AUROC_orig=ao,
                                  AUROC_orig_lo=ci(bo)[0], AUROC_orig_hi=ci(bo)[1], AUROC_transfer=at,
                                  AUROC_transfer_lo=ci(bt)[0], AUROC_transfer_hi=ci(bt)[1], delta=at - ao,
                                  delta_lo=ci(bd)[0], delta_hi=ci(bd)[1]))
            for tn in ["T_default", "T_fpr5"]:
                t0 = thr[m][tn]
                rr = dict(set=name, model=m, threshold=tn, thr=t0)
                for cond, s in [("orig", so), ("transfer", st)]:
                    rr[f"TPR_{cond}"] = float((s[y == 1] > t0).mean())
                    rr[f"FPR_{cond}"] = float((s[y == 0] > t0).mean())
                rows_op.append(rr)
            print(f"   {m:5s} AUROC orig {ao:.3f} [{ci(bo)[0]:.3f},{ci(bo)[1]:.3f}]  transfer {at:.3f} [{ci(bt)[0]:.3f},{ci(bt)[1]:.3f}]  Δ {at - ao:+.3f} [{ci(bd)[0]:+.3f},{ci(bd)[1]:+.3f}]")
    pd.DataFrame(rows_rank).to_csv(f"{OUT}/controlled_rank.csv", index=False)
    pd.DataFrame(rows_op).to_csv(f"{OUT}/controlled_operating_points.csv", index=False)
    pd.DataFrame(rows_meta).to_csv(f"{OUT}/controlled_metadata_check.csv", index=False)
    sets["strict(尺寸×压缩强度)"][["source_key", "label", "group_id", "maxside", "jpeg_q0_sum"]].to_csv(
        os.path.join(ROOT, "manifests", "controlled_strict_v1.csv"), index=False)


if __name__ == "__main__":
    main()
