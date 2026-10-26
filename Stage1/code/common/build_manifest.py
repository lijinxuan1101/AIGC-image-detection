#!/usr/bin/env python3
"""M2：构建样本 manifest 并冻结划分。

划分单位 = 重复组（精确重复 sha1 + 近重复 dHash<=16 的并查集），同组所有版本同一划分。
  explore      探索集：测试包中 200 real + 200 ai 源图（ai 按场景分层）。可反复查看，不是最终测试。
  target_dev   目标域开发集：测试包中 500 real + 500 ai；仅用于 E4（目标域阈值诊断），与无目标域方案分列。
  closed_test  封闭最终测试集：测试包其余源图。方法/阈值/模型选择不得使用。
  calib_src    源域校准集：训练/验证包原图（剔除与测试包同组者）；只含 original 条件，用于选定并锁定阈值。
输出：manifests/manifest.csv（每行一个图像版本），manifests/split_summary.csv，manifests/protocol_v1.json
"""
import json
import os
import hashlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUD = os.path.join(ROOT, "results", "audit")
OUT = os.path.join(ROOT, "manifests")
SEED = 20261026
N_EXPLORE = {0: 200, 1: 200}
N_TARGET_DEV = {0: 500, 1: 500}
NEAR_DIST = 16


class UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    os.makedirs(OUT, exist_ok=True)
    inv = pd.read_csv(os.path.join(AUD, "inventory.csv"))
    inv["source_key"] = inv.label_name + "/" + inv.source_id
    orig = inv[inv.condition == "original"]

    uf = UF()
    for k in orig.source_key:
        uf.find(k)
    # 精确重复
    for _, g in orig.groupby("sha1"):
        ks = g.source_key.tolist()
        for k in ks[1:]:
            uf.union(ks[0], k)
    # 近重复
    near = pd.read_csv(os.path.join(AUD, "dup_near.csv"))
    near = near[near.dist <= NEAR_DIST]
    for a, b in zip(near.label_a.map({0: "real", 1: "ai"}) + "/" + near.source_id_a,
                    near.label_b.map({0: "real", 1: "ai"}) + "/" + near.source_id_b):
        uf.union(a, b)
    src = orig[["source_key", "package", "label", "scenario"]].drop_duplicates("source_key").copy()
    src["group_id"] = src.source_key.map(uf.find)
    g = src.groupby("group_id").agg(n=("source_key", "size"), pk=("package", lambda s: set(s)), lab=("label", lambda s: set(s)))
    print("重复组:", int((g.n > 1).sum()), "| 跨包组:", int(g.pk.map(len).gt(1).sum()), "| 跨标签组:", int(g.lab.map(len).gt(1).sum()))

    # 以组为单位的包归属：含测试包成员的组整体视为测试组
    grp_test = src.groupby("group_id").package.agg(lambda s: (s == "test").any())
    src["grp_has_test"] = src.group_id.map(grp_test)
    rng = np.random.RandomState(SEED)

    split = {}
    # 源域校准集：训练/验证包且所在组不含测试包成员
    for _, r in src[(src.package == "train_val")].iterrows():
        split[r.source_key] = "calib_src" if not r.grp_has_test else "excluded_overlap_with_test"

    test_src = src[src.package == "test"]
    # 以组为单位抽样：先在 label 内按组打乱，逐组加入直到满足数量（组内所有源图一起进入）
    assigned_groups = {}

    def take(label, n, name, stratify=False):
        pool = test_src[(test_src.label == label) & (~test_src.group_id.isin(assigned_groups))]
        gids = pool.group_id.drop_duplicates().tolist()
        gids = [gids[i] for i in rng.permutation(len(gids))]
        if stratify and label == 1:  # ai 按场景比例分层
            scen_counts = pool.drop_duplicates("group_id").scenario.value_counts(normalize=True)
            quota = (scen_counts * n).round().astype(int).to_dict()
            diff = n - sum(quota.values())
            quota[scen_counts.index[0]] += diff
            got = {s: 0 for s in quota}
            gs = pool.drop_duplicates("group_id").set_index("group_id").scenario
            chosen = []
            for gid in gids:
                s = gs[gid]
                if got[s] < quota[s]:
                    chosen.append(gid)
                    got[s] += 1
        else:
            chosen = gids[:n]
        for gid in chosen:
            assigned_groups[gid] = name
        return chosen

    for lab in (0, 1):
        take(lab, N_EXPLORE[lab], "explore", stratify=True)
    for lab in (0, 1):
        take(lab, N_TARGET_DEV[lab], "target_dev")
    for _, r in test_src.iterrows():
        split[r.source_key] = assigned_groups.get(r.group_id, "closed_test")

    inv["split_name"] = inv.source_key.map(split)
    inv["group_id"] = inv.source_key.map(src.set_index("source_key").group_id)
    # 仅保留 original（所有包）与 transfer（测试包）；redigital 未纳入
    man = inv[inv.split_name.notna()].copy()
    man["variant_id"] = man.condition + "|" + man.source_key
    man["sha1_short"] = man.sha1.str[:12]
    cols = ["source_key", "source_id", "variant_id", "split_name", "package", "label", "label_name", "condition", "scenario",
            "group_id", "relpath", "format", "width", "height", "bytes", "sha1_short"]
    man = man[cols].rename(columns={"split_name": "split"}).sort_values(["split", "label", "source_key", "condition"])
    man.to_csv(os.path.join(OUT, "manifest.csv"), index=False)

    # 验证：同一源图所有版本同一划分；同一组同一划分（排除 excluded）
    chk1 = man.groupby("source_key").split.nunique().max()
    chk2 = man[man.split != "excluded_overlap_with_test"].groupby("group_id").split.nunique().max()
    print("同源图跨划分数(应为1):", chk1, "| 同组跨划分数(应为1):", chk2)
    assert chk1 == 1 and chk2 == 1, "划分泄漏"

    summ = man.groupby(["split", "condition", "label_name"]).size().unstack().fillna(0).astype(int)
    summ.to_csv(os.path.join(OUT, "split_summary.csv"))
    print(summ.to_string())
    print("explore ai 场景分布:\n", man[(man.split == "explore") & (man.label == 1) & (man.condition == "original")].scenario.value_counts().to_string())

    mf = os.path.join(OUT, "manifest.csv")
    proto = dict(version="v1", date="2026-10-26", seed=SEED, near_dist=NEAR_DIST, n_explore=N_EXPLORE, n_target_dev=N_TARGET_DEV,
                 manifest_sha256=hashlib.sha256(open(mf, "rb").read()).hexdigest(),
                 rules=["同源图所有版本同划分", "同重复组同划分", "closed_test 不得用于方法/阈值/模型选择",
                        "阈值只在 calib_src（原图）上选定并锁定；目标域校准仅用 target_dev 并单独列报",
                        "redigital 不纳入 Stage 1"])
    json.dump(proto, open(os.path.join(OUT, "protocol_v1.json"), "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
