"""Stage 2 模型：SigLIP2 -> (LIB) -> (GSR) -> 视觉投影 -> [BOS; 视觉 tokens; 文本 tokens] -> Phi-3-mini -> Yes/No。

对应 Stage2/模型结构与训练方案.md 第 2、3 节。
"""
import math
import os
import sys
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import PHI3_DIR, PROMPT, PROMPT_NO_CAPTION, SIGLIP_DIR  # noqa: E402


@dataclass
class Cfg:
    use_lib: bool = False
    lib_kernel: int = 3
    use_gsr: bool = False
    gsr_radius: int = 2
    gsr_heads: int = 8
    lora: bool = True              # 阶段 A 为 False：全部冻结，只训练投影
    lora_r: int = 16
    lora_alpha: int = 32
    use_caption: bool = True
    sem: bool = False              # 是否构建辅助语义头与教师读出（仅训练使用）
    grad_ckpt: bool = True

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------
class LIB(nn.Module):
    """固定高斯深度卷积 + 可学习标量混合：F^ = (1-a) F + a Smooth(F)，a = sigmoid(beta)，初始 a=0.5。"""

    def __init__(self, kernel=3, alpha0=0.5):
        super().__init__()
        sigma = (kernel + 1) / 4.0
        ax = torch.arange(kernel, dtype=torch.float32) - (kernel - 1) / 2
        g = torch.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
        self.register_buffer("kernel", (g / g.sum())[None, None])  # 固定，不训练
        self.k = kernel
        self.beta = nn.Parameter(torch.tensor(math.log(alpha0 / (1 - alpha0))))

    def forward(self, x, h, w):
        B, N, D = x.shape
        t = x.transpose(1, 2).reshape(B, D, h, w)
        p = self.k // 2
        t = F.pad(t, (p, p, p, p), mode="replicate")
        k = self.kernel.to(t.dtype).expand(D, 1, self.k, self.k)
        sm = F.conv2d(t, k, groups=D)
        sm = sm.reshape(B, D, N).transpose(1, 2)
        a = torch.sigmoid(self.beta).to(x.dtype)
        return (1 - a) * x + a * sm


class GSR(nn.Module):
    """屏蔽近邻（Chebyshev 距离 <= r，含自身）的自注意力；残差保留原特征。W_o 零初始化，初始为恒等映射。"""

    def __init__(self, dim, grid, heads=8, radius=2):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)
        ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        yy, xx = ys.reshape(-1), xs.reshape(-1)
        cheb = torch.maximum((yy[:, None] - yy[None]).abs(), (xx[:, None] - xx[None]).abs())
        allowed = cheb > radius                                   # True = 允许注意
        assert bool(allowed.any(dim=-1).all()), "存在没有合法 key 的 query，会导致 NaN"
        self.register_buffer("allowed", allowed)
        self.radius = radius

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=self.allowed[None, None])
        o = o.transpose(1, 2).reshape(B, N, D)
        return x + self.o(o)


# ----------------------------------------------------------------------------
class Detector(nn.Module):
    def __init__(self, cfg: Cfg, device="cuda"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel
        self.cfg = cfg
        self.device_ = device
        self.tok = AutoTokenizer.from_pretrained(PHI3_DIR)
        self.yes_id = self.tok("<|assistant|>\nYes", add_special_tokens=False).input_ids[-1]
        self.no_id = self.tok("<|assistant|>\nNo", add_special_tokens=False).input_ids[-1]

        vis = SiglipVisionModel.from_pretrained(SIGLIP_DIR, torch_dtype=torch.bfloat16)
        llm = AutoModelForCausalLM.from_pretrained(PHI3_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        for p in list(vis.parameters()) + list(llm.parameters()):
            p.requires_grad = False
        vc = vis.config
        self.grid = vc.image_size // vc.patch_size
        self.Dv, self.Dh = vc.hidden_size, llm.config.hidden_size

        vis_raw, llm_raw = vis, llm
        if cfg.lora:
            from peft import LoraConfig, get_peft_model
            vis = get_peft_model(vis, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
                target_modules=r".*encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|out_proj)"))
            llm = get_peft_model(llm, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, bias="none",
                target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]))
        self.vis, self.llm = vis, llm
        # peft 原地向 raw 模块注入 LoRA；保留 raw 引用（不注册为子模块）以便直接调用内部模块
        object.__setattr__(self, "_raw", {"vis": vis_raw, "llm": llm_raw})
        if cfg.grad_ckpt:
            llm_raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        llm_raw.config.use_cache = False

        self.lib = LIB(cfg.lib_kernel) if cfg.use_lib else None
        self.gsr = GSR(self.Dv, self.grid, cfg.gsr_heads, cfg.gsr_radius) if cfg.use_gsr else None
        self.proj = nn.Linear(self.Dv, self.Dh)
        emb = self._llm_inner().get_input_embeddings().weight
        emb_rms = emb.float().pow(2).mean().sqrt().item()
        nn.init.normal_(self.proj.weight, std=emb_rms / math.sqrt(self.Dv))   # 使初始视觉 token 的量级接近文本 embedding
        nn.init.zeros_(self.proj.bias)
        self.emb_rms = emb_rms
        self.aux = None
        if cfg.sem:
            self.aux = nn.Linear(self.Dv, self.Dv)
            nn.init.eye_(self.aux.weight)
            nn.init.zeros_(self.aux.bias)
        for m in (self.lib, self.gsr, self.proj, self.aux):
            if m is not None:
                m.float()
        self.to(device)

    # ---- 内部访问 ----
    def _llm_causal(self):
        return self._raw["llm"]

    def _llm_inner(self):
        return self._llm_causal().model

    def vis_dtype(self):
        return self._raw["vis"].embeddings.patch_embedding.weight.dtype if hasattr(self._raw["vis"], "embeddings") \
            else self._raw["vis"].vision_model.embeddings.patch_embedding.weight.dtype

    def llm_dtype(self):
        return self._llm_inner().get_input_embeddings().weight.dtype

    def _vis_core(self):
        return self._raw["vis"]

    # ---- 参数分组 ----
    def new_params(self):
        out = []
        for m in (self.lib, self.gsr, self.proj, self.aux):
            if m is not None:
                out += [p for p in m.parameters() if p.requires_grad]
        return out

    def lora_params(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and "lora_" in n]

    # ---- 视觉 ----
    @staticmethod
    def normalize(pix_u8):
        """uint8 [B,H,W,3] -> SigLIP2 归一化（mean=std=0.5）。"""
        x = pix_u8.permute(0, 3, 1, 2).float() / 255.0
        return (x - 0.5) / 0.5

    def visual_tokens(self, x):
        h = self.vis(pixel_values=x.to(self.vis_dtype())).last_hidden_state   # [B, N, Dv]
        h = h.float()
        if self.lib is not None:
            h = self.lib(h, self.grid, self.grid)
        if self.gsr is not None:
            h = self.gsr(h)
        return h

    def teacher_pool(self, x):
        with torch.no_grad():
            ctx = self.vis.disable_adapter() if self.cfg.lora else torch.no_grad()
            with ctx:
                return self._vis_core().forward(pixel_values=x.to(self.vis_dtype())).pooler_output.float()

    def student_pool(self, tokens):
        core = self._vis_core()
        head = core.vision_model.head if hasattr(core, "vision_model") else core.head   # 原权重冻结的 MAP 读出，仅梯度穿过
        return self.aux(head(tokens.to(self.vis_dtype())).float())

    # ---- 文本与拼接 ----
    def build_text_ids(self, captions):
        outs = []
        for c in captions:
            body = PROMPT.format(caption=c) if self.cfg.use_caption else PROMPT_NO_CAPTION
            s = f"<|user|>\n{body}<|end|>\n<|assistant|>\n"
            outs.append(self.tok(s, add_special_tokens=False).input_ids)
        return outs

    def forward(self, pix_u8, captions, with_sem=False):
        dev = self.device_
        x = self.normalize(pix_u8.to(dev))
        tokens = self.visual_tokens(x)                                  # F_g
        V = self.proj(tokens)                                           # [B, N, Dh]
        B, N, _ = V.shape
        ids = self.build_text_ids(captions)
        L = max(len(i) for i in ids)
        pad_id = self.tok.pad_token_id
        T_ids = torch.full((B, L), pad_id, dtype=torch.long)
        T_mask = torch.zeros(B, L, dtype=torch.long)
        for b, i in enumerate(ids):                                     # 文本内部左填充：[BOS][V][pad][T]
            T_ids[b, L - len(i):] = torch.tensor(i)
            T_mask[b, L - len(i):] = 1
        T_ids, T_mask = T_ids.to(dev), T_mask.to(dev)
        emb = self._llm_inner().get_input_embeddings()
        bos = emb(torch.full((B, 1), self.tok.bos_token_id, device=dev))
        T = emb(T_ids)
        seq = torch.cat([bos.float(), V, T.float()], dim=1).to(self.llm_dtype())
        mask = torch.cat([torch.ones(B, 1 + N, dtype=torch.long, device=dev), T_mask], dim=1)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        hs = self._llm_inner()(inputs_embeds=seq, attention_mask=mask, position_ids=pos,
                               use_cache=False).last_hidden_state[:, -1]  # 回答位置（最后一个有效 token）
        W = self._llm_causal().lm_head.weight[[self.yes_id, self.no_id]]
        z = hs.float() @ W.float().T                                           # [B, 2] = (Yes, No)
        out = {"logit": z[:, 1] - z[:, 0], "z": z}                      # logit(P(fake)) = z_No - z_Yes
        if with_sem and self.aux is not None:
            out["sem"] = 1 - F.cosine_similarity(self.student_pool(tokens), self.teacher_pool(x), dim=-1)
        return out

    # ---- 保存/加载（只存可训练参数） ----
    def trainable_state(self):
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, sd, strict=False):
        own = dict(self.named_parameters())
        missing = [k for k in sd if k not in own]
        for k, v in sd.items():
            if k in own:
                own[k].data.copy_(v.to(own[k].device, own[k].dtype))
        if strict and missing:
            raise KeyError(missing)
        return missing
