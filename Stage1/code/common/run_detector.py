#!/usr/bin/env python3
"""统一推理：按 manifest 对每个图像版本独立推理，保存连续分数。fake=1，分数越高越倾向生成。

每个模型保留官方预处理：
  clip   : clipdet_latent10k_plus；Resize(224, bicubic) + CenterCrop(224) + CLIP 归一化；输出 LLR logit（>0 为合成）
  bfree  : BFREE_dino2reg4；ToTensor + ImageNet 归一化，无缩放；模型内部做中心+四角共 5 个 504 裁切并平均 logit；batch=1
  npr    : NPR.pth（备选 model_epoch_last_3090.pth）；Resize((256,256)) bilinear，无裁切（官方 test.py 对混合尺寸数据集的默认）+ ImageNet 归一化；输出 logit
所有模型保存 logit 作为 raw_score；默认判决阈值 raw_score>0（对应 sigmoid>0.5），仅为默认，不作校准。
失败样本写入失败日志，不静默丢弃。
"""
import argparse
import importlib
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = None  # 超大图（最高约 1.8 亿像素）不触发解压炸弹异常，已在审计报告记录

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "extracted")
BASE = os.path.join(ROOT, "code", "baselines")
W = os.path.join(ROOT, "weights")


def build(model_name, ckpt, device):
    """返回 (model, transform, batch_size, checkpoint_desc)"""
    import torchvision.transforms as T
    if model_name == "clip":
        repo = os.path.join(BASE, "ClipBased-SyntheticImageDetection")
        sys.path.insert(0, repo)
        from networks import create_architecture, load_weights
        from utils.processing import make_normalize
        name = ckpt or "clipdet_latent10k_plus"
        cfg = yaml.safe_load(open(os.path.join(W, "clip2024", name, "config.yaml")))
        model = load_weights(create_architecture(cfg["arch"]), os.path.join(W, "clip2024", name, cfg["weights_file"]))
        tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop((224, 224)),
                        make_normalize(cfg["norm_type"])])
        return model.to(device).eval(), tf, 64, f"clip2024/{name}"
    if model_name == "bfree":
        repo = os.path.join(BASE, "B-Free", "code")
        sys.path.insert(0, repo)
        from networks import get_network, load_weights
        from utils.normalization import get_list_norm
        d = os.path.join(W, "bfree", "BFREE_dino2reg4")
        cfg = yaml.safe_load(open(os.path.join(d, "config.yaml")))
        model = load_weights(get_network(cfg["arch"]), os.path.join(d, cfg["weights_file"]))
        tf = T.Compose(get_list_norm(cfg["norm_type"]))
        return model.to(device).eval(), tf, 1, "bfree/BFREE_dino2reg4"
    if model_name == "npr":
        repo = os.path.join(BASE, "NPR-DeepfakeDetection")
        sys.path.insert(0, repo)
        from networks.resnet import resnet50
        name = ckpt or "NPR.pth"
        model = resnet50(num_classes=1)
        sd = torch.load(os.path.join(W, "npr", name), map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        tf = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return model.to(device).eval(), tf, 128, f"npr/{name}"
    raise ValueError(model_name)


class DS(Dataset):
    def __init__(self, df, tf):
        self.df, self.tf = df.reset_index(drop=True), tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:
            with Image.open(os.path.join(DATA, r.relpath)) as im:
                x = self.tf(im.convert("RGB"))
            return i, x, ""
        except Exception as e:  # noqa: BLE001
            return i, torch.zeros(1), f"load_fail:{type(e).__name__}:{str(e)[:80]}"


def collate(b):
    idx = [t[0] for t in b]
    err = [t[2] for t in b]
    ok = [t[1] for t in b if t[2] == ""]
    return idx, err, (torch.stack(ok) if ok and all(o.shape == ok[0].shape for o in ok) else ok)


def to_score(out):
    out = out.float().cpu().numpy()
    if out.ndim == 1:
        return out
    if out.shape[1] == 2:
        out = out[:, 1] - out[:, 0]
    elif out.shape[1] == 1:
        out = out[:, 0]
    if out.ndim > 1:
        out = out.reshape(out.shape[0], -1).mean(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["clip", "bfree", "npr"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--splits", default="explore")
    ap.add_argument("--conditions", default="original,transfer")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    man = pd.read_csv(os.path.join(ROOT, "manifests", a.manifest))
    df = man[man.split.isin(a.splits.split(",")) & man.condition.isin(a.conditions.split(","))].copy()
    if a.limit:
        df = df.sample(a.limit, random_state=0)
    t0 = time.time()
    model, tf, bs, ckdesc = build(a.model, a.ckpt, a.device)
    print(f"[{a.model}] {ckdesc} 加载完成，用时 {time.time() - t0:.1f}s；待测 {len(df)} 张", flush=True)
    dl = DataLoader(DS(df, tf), batch_size=bs, num_workers=a.workers, collate_fn=collate, shuffle=False)
    scores = np.full(len(df), np.nan)
    errs = [""] * len(df)
    t0 = time.time()
    with torch.no_grad():
        for n, (idx, err, x) in enumerate(dl):
            for i, e in zip(idx, err):
                errs[i] = e
            good = [i for i, e in zip(idx, err) if e == ""]
            if len(good):
                try:
                    xs = x if torch.is_tensor(x) else None
                    if xs is None:  # 形状不一致（不应发生于 clip/npr）
                        raise RuntimeError("batch shape mismatch")
                    s = to_score(model(xs.to(a.device)))
                    scores[good] = s
                except Exception as e:  # noqa: BLE001
                    for i in good:
                        errs[i] = f"infer_fail:{type(e).__name__}:{str(e)[:80]}"
                    torch.cuda.empty_cache()
            if n % 20 == 0:
                print(f"  {n * bs}/{len(df)}  {(n + 1) * bs / (time.time() - t0):.1f} img/s", flush=True)
    el = time.time() - t0
    out = df[["source_key", "source_id", "variant_id", "split", "package", "label", "scenario", "condition", "group_id",
              "relpath", "format", "width", "height", "bytes"]].copy()
    out["generator"] = "unknown"
    out["platform_or_chain"] = np.where(out.condition == "transfer", "unknown_transfer", "none")
    out["model"] = a.model
    out["checkpoint"] = ckdesc
    out["raw_score"] = scores
    out["pred_default"] = (scores > 0).astype(float)
    out.loc[np.isnan(scores), "pred_default"] = np.nan
    out["error"] = errs
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    nf = int((out.error != "").sum())
    print(f"完成：{len(out)} 条，失败 {nf}，推理用时 {el:.1f}s（{len(out) / el:.1f} img/s）→ {a.out}", flush=True)
    if nf:
        print(out[out.error != ""].error.value_counts().to_string())
    # 显存/延迟记录
    print(f"峰值显存 {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
