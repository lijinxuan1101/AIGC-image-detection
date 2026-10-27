"""Stage 2 公共常量与工具。"""
import hashlib
import os
import random

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")
WEIGHTS = os.path.join(ROOT, "weights")
RESULTS = os.path.join(ROOT, "results")
S1_MANIFEST = os.path.join(ROOT, "..", "Stage1", "manifests", "manifest.csv")

SIGLIP_DIR = os.path.join(WEIGHTS, "siglip2-base-patch16-256")
PHI3_DIR = os.path.join(WEIGHTS, "Phi-3-mini-4k-instruct")
BLIP2_DIR = os.path.join(WEIGHTS, "blip2-opt-2.7b")

IMG_SIZE = 256          # SigLIP2 固定输入（与 checkpoint 一致，真假同一预处理）
PARTS = ("train", "dev", "trainprop", "devprop")  # train 含 debug 划分；*prop 为随机传播视图


def views_csv(part):
    return os.path.join(DATA, f"views_{part}.csv")


def pix_path(part):                             # N x 256 x 256 x 3 uint8 npy memmap
    return os.path.join(DATA, f"pix256_{part}.npy")


CAPTIONS_CSV = os.path.join(DATA, "captions.csv")

SEED = 20261027

PROMPT = ("Based on the image description: {caption}, assess the authenticity.\n"
          "Is this image real?")
PROMPT_NO_CAPTION = "Assess the authenticity.\nIs this image real?"


def stable_int(s: str, mod: int = 2**31 - 1) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16) % mod


def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
