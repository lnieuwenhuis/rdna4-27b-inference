"""PyTorch replica of the fork's Qwen3.5/3.8 MTP head (src/models/qwen35.cpp, build_mtp) plus the recursive-draft
("training-time test") forward used for training, and loaders for the GGUF weights.

Inference (common/speculative.cpp, draft-mtp, single head): the head's KV holds, for every position q, the entry
built from (h = target hidden at q-1, token x_q). Draft step 0 at the chain start p uses the target hidden; step k
feeds the head's own output hidden of step k-1 (h_nextn after shared_head_norm) with the drafted token, at position
p+k, and attends to the true entries up to p plus the chain's own entries. Training mirrors that with teacher forcing:
state S[k][q] (step k, chain start q-k) has input (out(S[k-1][q-1]), x_q) and attends to S[0][r] for r <= q-k and
S[j][q-k+j] for j = 1..k.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# gguf-py from the llama.cpp checkout ($LLAMA_CPP_DIR) when set, otherwise the pip `gguf` package
if os.environ.get("LLAMA_CPP_DIR"):
    sys.path.insert(0, os.path.join(os.environ["LLAMA_CPP_DIR"], "gguf-py"))
from gguf import GGUFReader, quants  # noqa: E402

N_EMBD, N_HEAD, N_HEAD_KV, HEAD_DIM, N_FF = 5120, 24, 4, 256, 17408
N_ROT, ROPE_BASE, EPS = 64, 1e7, 1e-6
LINEARS = ["nextn.eh_proj", "attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
NORMS = ["nextn.enorm", "nextn.hnorm", "attn_norm", "attn_q_norm", "attn_k_norm", "post_attention_norm",
         "nextn.shared_head_norm"]


def read_gguf(path, names):
    """Dequantized float32 numpy arrays (shape [out, in] for matrices) of the named tensors."""
    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        if t.name in names:
            out[t.name] = quants.dequantize(t.data, t.tensor_type).astype(np.float32)
    missing = set(names) - set(out)
    assert not missing, missing
    return out


def rms_norm(x, w):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * w).to(x.dtype)


def rope(x, pos):
    """NeoX-style rotation of the first N_ROT dims (IMROPE with equal text positions in every section)."""
    half = N_ROT // 2
    inv = ROPE_BASE ** (-torch.arange(half, device=x.device, dtype=torch.float32) * 2 / N_ROT)
    ang = pos.float()[:, None] * inv[None, :]                       # [L, half]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]          # [L, 1, half]
    x1, x2, rest = x[..., :half].float(), x[..., half:N_ROT].float(), x[..., N_ROT:]
    rot = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)
    return torch.cat([rot, rest], -1)


class LoRALinear(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        self.register_buffer("w", weight)                            # frozen base, bf16 [out, in]
        self.rank = rank
        if rank > 0:
            self.a = nn.Parameter(torch.randn(rank, weight.shape[1], device=weight.device) / math.sqrt(weight.shape[1]))
            self.b = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device))
            self.scale = alpha / rank

    def forward(self, x):
        y = F.linear(x, self.w)
        if self.rank > 0:
            y = y + F.linear(F.linear(x.float(), self.a), self.b).to(y.dtype) * self.scale
        return y

    def merged(self):
        w = self.w.float()
        if self.rank > 0:
            w = w + (self.b.float() @ self.a.float()) * self.scale
        return w


class MTPHead(nn.Module):
    def __init__(self, tensors, device, rank=64, alpha=64):
        super().__init__()
        g = lambda n: torch.from_numpy(tensors[f"blk.64.{n}.weight"]).to(device)
        self.lin = nn.ModuleDict({n.replace(".", "_"): LoRALinear(g(n).to(torch.bfloat16), rank, alpha)
                                  for n in LINEARS})
        self.norm = nn.ParameterDict({n.replace(".", "_"): nn.Parameter(g(n).float()) for n in NORMS})

    def L(self, n, x):
        return self.lin[n.replace(".", "_")](x)

    def N(self, n, x):
        return rms_norm(x, self.norm[n.replace(".", "_")])

    def qkv(self, h, e, pos):
        """Inputs of one state: h [L, n_embd] (hidden to continue from), e [L, n_embd] (token embeddings)."""
        x = self.L("nextn.eh_proj", torch.cat([self.N("nextn.enorm", e), self.N("nextn.hnorm", h)], -1))
        a = self.N("attn_norm", x)
        qf = self.L("attn_q", a).view(-1, N_HEAD, 2 * HEAD_DIM)
        q, gate = qf[..., :HEAD_DIM], qf[..., HEAD_DIM:]
        q = rope(self.N("attn_q_norm", q), pos)
        k = rope(self.N("attn_k_norm", self.L("attn_k", a).view(-1, N_HEAD_KV, HEAD_DIM)), pos)
        v = self.L("attn_v", a).view(-1, N_HEAD_KV, HEAD_DIM)
        return x, q, gate, k, v

    def finish(self, x, attn, gate):
        """Attention output -> residual -> FFN -> shared_head_norm: the h_nextn row (fed to lm_head and the next step)."""
        o = self.L("attn_output", (attn * torch.sigmoid(gate)).reshape(attn.shape[0], -1))
        x = x + o
        f = self.N("post_attention_norm", x)
        x = x + self.L("ffn_down", F.silu(self.L("ffn_gate", f)) * self.L("ffn_up", f))
        return self.N("nextn.shared_head_norm", x)

    def forward_steps(self, h0, emb, pos, n_steps):
        """h0: target hidden rows H[q-1] for q = 1..L (row i belongs to position i+1 of the window), emb: token
        embeddings of x_q for the same rows, pos: positions. Returns the output hidden of every step: list of
        [L, n_embd], where out[k][i] is state S[k][q=i] (meaningful for i >= k)."""
        L = h0.shape[0]
        dev = h0.device
        idx = torch.arange(L, device=dev)
        ks, vs, outs = [], [], []
        h = h0
        for k in range(n_steps):
            x, q, gate, kk, vv = self.qkv(h, emb, pos)
            ks.append(kk)
            vs.append(vv)
            # keys of S[k][q]: S[0][r] for r <= q-k, and S[j][q-k+j] for j = 1..k (one per step, the chain's own)
            rep = N_HEAD // N_HEAD_KV
            qh = q.transpose(0, 1)                                                    # [H, L, D]
            k0 = ks[0].repeat_interleave(rep, 1).transpose(0, 1)                      # [H, L, D]
            s0 = (qh.float() @ k0.float().transpose(1, 2)) / math.sqrt(HEAD_DIM)      # [H, L, L]
            mask0 = idx[None, :] <= (idx[:, None] - k)                                # r <= q-k
            s0 = s0.masked_fill(~mask0[None], float("-inf"))
            sd, vd = [], []
            for j in range(1, k + 1):
                src = (idx - k + j).clamp(min=0)                                      # row q-k+j of step j
                kj = ks[j].repeat_interleave(rep, 1)[src]                             # [L, H, D]
                sd.append((q.float() * kj.float()).sum(-1).transpose(0, 1) / math.sqrt(HEAD_DIM))  # [H, L]
                vd.append(vs[j].repeat_interleave(rep, 1)[src])                       # [L, H, D]
            s = torch.cat([s0] + [x_[..., None] for x_ in sd], -1)                    # [H, L, L+k]
            p = torch.softmax(s, -1)
            p = torch.nan_to_num(p)                                                   # rows q < k have no keys yet
            v0 = vs[0].repeat_interleave(rep, 1).transpose(0, 1).float()              # [H, L, D]
            attn = p[..., :L] @ v0
            for j in range(k):
                attn = attn + p[..., L + j, None] * vd[j].transpose(0, 1).float()
            attn = attn.transpose(0, 1).to(q.dtype)                                   # [L, H, D]
            out = self.finish(x, attn, gate)
            outs.append(out)
            # next step continues from this step's output of the previous position (chain shift)
            h = torch.cat([out[:1] * 0, out[:-1]], 0)
        return outs
