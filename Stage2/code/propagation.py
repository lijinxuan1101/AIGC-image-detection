"""随机参数传播流水线（Stage2 §4.1 / 阶段 C）。

按"类别 → 参数"组织。每个类别有启用概率与参数区间；抽样只依赖 (seed, source_key, view_index)，
与标签无关、可复现。它不拟合任何具体社交平台，只是一个带随机参数的确定性算子组合。

参数区间的灵感来自以下工作的常见做法（**仅参考其增强类型，区间是本项目的起始值，并非逐项复现**）：
  - Wang et al., CVPR 2020（CNNDetection）：高斯模糊 + JPEG 的随机增强；
  - Real-ESRGAN, ICCVW 2021：多阶（高阶）退化、随机缩放插值、JPEG 质量 30–95、少量噪声；
  - Corvi et al., ICASSP 2023 / B-Free：随机裁切、缩放、JPEG、模糊等检测器训练增强；
  - Hadwiger & Riess, FODB 论文：Instagram 上传时的正方形中心裁切后缩小（只采用"裁切+缩小"这一现象，
    没有采用任何平台的具体数值）。
参数区间不得根据 target_dev / 任何测试结果调整。改动区间需更新 SPEC_VERSION 并重新生成视图与描述。
"""
import io
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from PIL import Image, ImageFilter

SPEC_VERSION = "prop-v1"

INTERP = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS, "box": Image.BOX}


@dataclass
class Spec:
    # ---- 1. 缩小（只缩小不放大）----
    p_resize: float = 0.7
    resize_modes: tuple = ("long_side", "scale")           # 二选一，等概率
    long_side_caps: tuple = (720, 960, 1080, 1280, 1600, 2048)
    scale_range: tuple = (0.5, 0.95)
    interps: tuple = ("bilinear", "bicubic", "lanczos", "box")
    # ---- 2. 裁切 ----
    p_crop: float = 0.3
    crop_ratios: tuple = (1.0, 4 / 5, 4 / 3, 16 / 9)       # 宽/高；1.0 即正方形
    crop_offset_max: float = 0.10                           # 裁窗中心相对中心偏移占可移动范围的比例上限
    # ---- 3. 模糊 / 锐化（互斥，各占一半）----
    p_filter: float = 0.2
    blur_sigma: tuple = (0.3, 1.0)
    unsharp_percent: tuple = (50, 200)
    unsharp_radius: tuple = (0.5, 1.5)
    # ---- 4. 噪声 ----
    p_noise: float = 0.1
    noise_sigma: tuple = (1.0, 4.0)                         # 像素 0–255 尺度
    # ---- 5. JPEG ----
    jpeg_q: tuple = (40, 95)
    subsampling: tuple = ("4:2:0", "4:4:4")
    # ---- 6. 链：第二轮（缩小 + JPEG）----
    p_second_round: float = 0.2
    second_scale_range: tuple = (0.7, 1.0)
    second_jpeg_q: tuple = (50, 95)


SPEC = Spec()
SUBSAMPLING = {"4:4:4": 0, "4:2:0": 2}


def sample_params(rng: np.random.RandomState, spec: Spec = SPEC) -> Dict:
    """按固定顺序抽取全部随机数（无论算子是否启用都消耗相同数量的随机数，保证参数稳定可比）。"""
    u = rng.rand(12)
    P: Dict = {"spec": SPEC_VERSION}
    # 缩小
    mode = spec.resize_modes[int(rng.randint(len(spec.resize_modes)))]
    cap = int(spec.long_side_caps[int(rng.randint(len(spec.long_side_caps)))])
    scale = float(rng.uniform(*spec.scale_range))
    interp = spec.interps[int(rng.randint(len(spec.interps)))]
    P["resize"] = None if u[0] >= spec.p_resize else dict(mode=mode, cap=cap, scale=scale, interp=interp)
    # 裁切
    ratio = float(spec.crop_ratios[int(rng.randint(len(spec.crop_ratios)))])
    off = rng.uniform(-spec.crop_offset_max, spec.crop_offset_max, size=2)
    P["crop"] = None if u[1] >= spec.p_crop else dict(ratio=ratio, off=[float(off[0]), float(off[1])])
    # 模糊 / 锐化
    sigma = float(rng.uniform(*spec.blur_sigma))
    up = float(rng.uniform(*spec.unsharp_percent))
    ur = float(rng.uniform(*spec.unsharp_radius))
    if u[2] >= spec.p_filter:
        P["filter"] = None
    elif u[3] < 0.5:
        P["filter"] = dict(kind="blur", sigma=sigma)
    else:
        P["filter"] = dict(kind="unsharp", percent=up, radius=ur)
    # 噪声
    ns = float(rng.uniform(*spec.noise_sigma))
    P["noise"] = None if u[4] >= spec.p_noise else dict(sigma=ns, seed=int(rng.randint(2**31 - 1)))
    # JPEG
    q = int(rng.randint(spec.jpeg_q[0], spec.jpeg_q[1] + 1))
    sub = spec.subsampling[int(rng.randint(len(spec.subsampling)))]
    P["jpeg"] = dict(q=q, subsampling=sub)
    # 第二轮
    s2 = float(rng.uniform(*spec.second_scale_range))
    q2 = int(rng.randint(spec.second_jpeg_q[0], spec.second_jpeg_q[1] + 1))
    sub2 = spec.subsampling[int(rng.randint(len(spec.subsampling)))]
    interp2 = spec.interps[int(rng.randint(len(spec.interps)))]
    P["round2"] = None if u[5] >= spec.p_second_round else dict(
        scale=s2, interp=interp2, jpeg=dict(q=q2, subsampling=sub2))
    return P


# ----------------------------------------------------------------------------
def _resize(img, r):
    w, h = img.size
    if r["mode"] == "long_side":
        s = min(1.0, r["cap"] / max(w, h))
    else:
        s = r["scale"]
    if s >= 0.999:
        return img
    return img.resize((max(8, round(w * s)), max(8, round(h * s))), INTERP[r["interp"]])


def _crop(img, c):
    w, h = img.size
    ratio = c["ratio"]
    if w / h > ratio:                      # 太宽：裁宽
        cw, ch = round(h * ratio), h
    else:                                  # 太高：裁高
        cw, ch = w, round(w / ratio)
    cw, ch = max(8, min(cw, w)), max(8, min(ch, h))
    mx, my = (w - cw) / 2, (h - ch) / 2    # 可移动半径
    cx = w / 2 + c["off"][0] * mx * 2
    cy = h / 2 + c["off"][1] * my * 2
    x0 = int(round(min(max(cx - cw / 2, 0), w - cw)))
    y0 = int(round(min(max(cy - ch / 2, 0), h - ch)))
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def _filter(img, f):
    if f["kind"] == "blur":
        return img.filter(ImageFilter.GaussianBlur(f["sigma"]))
    return img.filter(ImageFilter.UnsharpMask(radius=f["radius"], percent=int(f["percent"]), threshold=0))


def _noise(img, n):
    rng = np.random.RandomState(n["seed"])
    a = np.asarray(img, dtype=np.float32)
    a = a + rng.normal(0, n["sigma"], a.shape).astype(np.float32)
    return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8))


def _jpeg(img, j) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(j["q"]), subsampling=SUBSAMPLING[j["subsampling"]])
    return buf.getvalue()


def apply(img: Image.Image, P: Dict) -> bytes:
    """按固定顺序执行：缩小 → 裁切 → 模糊/锐化 → 噪声 → JPEG →（可选）第二轮缩小+JPEG。返回最终 JPEG 字节。"""
    img = img.convert("RGB")
    if P["resize"]:
        img = _resize(img, P["resize"])
    if P["crop"]:
        img = _crop(img, P["crop"])
    if P["filter"]:
        img = _filter(img, P["filter"])
    if P["noise"]:
        img = _noise(img, P["noise"])
    data = _jpeg(img, P["jpeg"])
    if P["round2"]:
        r2 = P["round2"]
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        if r2["scale"] < 0.999:
            img = img.resize((max(8, round(w * r2["scale"])), max(8, round(h * r2["scale"]))), INTERP[r2["interp"]])
        data = _jpeg(img, r2["jpeg"])
    return data


def flat(P: Dict) -> Dict:
    """展平为一行记录（便于存 csv 和统计）。"""
    r = P["resize"] or {}
    c = P["crop"] or {}
    f = P["filter"] or {}
    n = P["noise"] or {}
    r2 = P["round2"] or {}
    return dict(
        spec=P["spec"],
        resize=(r.get("mode", "none")), resize_cap=r.get("cap", -1) if r.get("mode") == "long_side" else -1,
        resize_scale=round(r.get("scale", -1), 4) if r.get("mode") == "scale" else -1,
        resize_interp=r.get("interp", ""),
        crop_ratio=round(c.get("ratio", -1), 4), crop_off_x=round(c["off"][0], 4) if c else 0.0,
        crop_off_y=round(c["off"][1], 4) if c else 0.0,
        filter=f.get("kind", "none"),
        filter_strength=round(f.get("sigma", f.get("percent", -1)), 3),
        noise_sigma=round(n.get("sigma", -1), 3),
        jpeg_q=P["jpeg"]["q"], jpeg_sub=P["jpeg"]["subsampling"],
        round2=int(bool(r2)), round2_scale=round(r2.get("scale", -1), 4),
        round2_q=r2["jpeg"]["q"] if r2 else -1,
    )
