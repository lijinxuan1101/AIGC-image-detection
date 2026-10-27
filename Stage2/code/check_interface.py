#!/usr/bin/env python3
"""阶段 A 接口核验（只使用 debug 划分，不进入任何报告）。

检查项：
  1. Yes/No 答案 token：在完整提示上下文之后恰为 1 个 token，且 Yes≠No
  2. LIB/GSR：形状、初始 α=0.5、GSR 初始为恒等、屏蔽后每个 query 都有合法 key、无 NaN
  3. 冻结/梯度：只有投影有梯度，冻结参数 grad 为 None
  4. 位置/mask：单样本 logit 与"同批、不同长度 caption"下的 logit 一致（容许 bf16 误差）
  5. 方向：label=1（生成）对应 "No"，BCE 对 logit = z_No - z_Yes
  6. 视觉 token 量级与文本 embedding 的比较
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import PHI3_DIR, PROMPT, RESULTS  # noqa: E402
from data import Views  # noqa: E402
from model import GSR, LIB, Cfg, Detector  # noqa: E402

res = {}

# 2. LIB / GSR（不需要大模型）
x = torch.randn(2, 256, 64)
lib = LIB(3)
y = lib(x, 16, 16)
res["lib_shape_ok"] = tuple(y.shape) == tuple(x.shape)
res["lib_alpha0"] = float(torch.sigmoid(lib.beta))
res["lib_kernel_sum"] = float(lib.kernel.sum())
res["lib_no_trainable_kernel"] = not lib.kernel.requires_grad
for r in (1, 2, 3):
    g = GSR(64, 16, heads=4, radius=r)
    o = g(x)
    res[f"gsr_r{r}_min_valid_keys"] = int(g.allowed.sum(-1).min())
    res[f"gsr_r{r}_max_valid_keys"] = int(g.allowed.sum(-1).max())
    res[f"gsr_r{r}_identity_at_init"] = bool(torch.allclose(o, x))
    res[f"gsr_r{r}_finite"] = bool(torch.isfinite(o).all())
    # 屏蔽验证：改变某 query 近邻 token 的值，不应改变该 query 的注意力输出（不含残差）
    g.o.weight.data.normal_(0, 0.1)
    base = g(x)
    x2 = x.clone()
    x2[:, 0] += 5.0                       # 改动 token 0（grid 位置 (0,0)）
    o2 = g(x2)
    # token 1 与 token 0 相邻：其 key 集合不包含 token 0，但自身 x2[1]==x[1]，因此 token 1 输出应不变
    res[f"gsr_r{r}_neighbor_blocked"] = bool(torch.allclose(base[:, 1], o2[:, 1], atol=1e-5))

# 1. 答案 token
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(PHI3_DIR)
ctx = f"<|user|>\n{PROMPT.format(caption='a cat on a sofa')}<|end|>\n<|assistant|>\n"
base_ids = tok(ctx, add_special_tokens=False).input_ids
for ans in ("Yes", "No"):
    full = tok(ctx + ans, add_special_tokens=False).input_ids
    res[f"answer_{ans}_extra_tokens"] = len(full) - len(base_ids)
    res[f"answer_{ans}_prefix_unchanged"] = full[:len(base_ids)] == base_ids
    res[f"answer_{ans}_token"] = tok.convert_ids_to_tokens(full[len(base_ids):])

# 3–6 需要完整模型（阶段 A：全部冻结，只训练投影）
views = Views(("train",))
d = views.df[views.df.split == "debug"]
rows = pd_rows = d[d.condition == "orig"].groupby("label").head(2)       # 2 real + 2 ai
pix, caps = views.get(rows)
y = torch.tensor(rows.label.values, dtype=torch.float32, device="cuda")

model = Detector(Cfg(lora=False, use_caption=True, grad_ckpt=True))
model.train()
res["yes_id"], res["no_id"] = model.yes_id, model.no_id
out = model(pix, caps)
res["logit_finite"] = bool(torch.isfinite(out["logit"]).all())
loss = F.binary_cross_entropy_with_logits(out["logit"], y)
loss.backward()
named = dict(model.named_parameters())
res["trainable_tensors"] = [n for n, p in named.items() if p.requires_grad]
res["proj_grad_norm"] = float(model.proj.weight.grad.norm())
res["frozen_with_grad"] = [n for n, p in named.items() if (not p.requires_grad) and p.grad is not None]
res["n_frozen_params_M"] = round(sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6, 1)
res["n_trainable_params_M"] = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3)

with torch.no_grad():
    V = model.proj(model.visual_tokens(model.normalize(pix.cuda())))
    res["visual_token_rms"] = float(V.pow(2).mean().sqrt())
    res["text_embedding_rms"] = model.emb_rms
    # mask/位置：样本 0 单独 vs 与长 caption 同批
    model.eval()
    one = model(pix[:1], caps[:1])["logit"]
    long_caps = [caps[0], caps[1] + " and many other very long words " * 6, caps[2], caps[3]]
    batched = model(pix, long_caps)["logit"][:1]
    res["pad_invariance_abs_diff"] = float((one - batched).abs())
    res["logit_scale_example"] = [round(v, 3) for v in out["logit"].tolist()]
    # 方向：BCE 的梯度应推动 label=1 的 logit 上升
    z = out["z"]
    res["direction_logit_equals_zNo_minus_zYes"] = bool(torch.allclose(out["logit"], z[:, 1] - z[:, 0]))

res["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

# 4'. fp32 下的 mask/位置不变性（消除 bf16 噪声）：主干与 LLM 暂转 fp32
with torch.no_grad():
    model._raw["llm"].float()
    model._raw["vis"].float()
    one32 = model(pix[:1], caps[:1])["logit"]
    batched32 = model(pix, long_caps)["logit"][:1]
    res["pad_invariance_fp32_abs_diff"] = float((one32 - batched32).abs())
    res["fp32_logit_example"] = float(one32)
os.makedirs(os.path.join(RESULTS, "checks"), exist_ok=True)
json.dump(res, open(os.path.join(RESULTS, "checks", "interface_check.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
