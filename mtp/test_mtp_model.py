"""Equivalence test: the parallel recursive-draft forward (MTPHead.forward_steps) must equal a sequential simulation
of the fork's inference loop (common/speculative.cpp draft-mtp: one KV entry appended per draft step). Small random
dimensions, CPU, float32. run: .venv\\Scripts\\python.exe -I test_mtp_model.py"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_model as M  # noqa: E402

M.N_EMBD, M.N_HEAD, M.N_HEAD_KV, M.HEAD_DIM, M.N_FF, M.N_ROT = 32, 4, 2, 16, 48, 8
E, H, HKV, D, FF = M.N_EMBD, M.N_HEAD, M.N_HEAD_KV, M.HEAD_DIM, M.N_FF
rng = np.random.default_rng(0)
shapes = {"nextn.eh_proj": (E, 2 * E), "attn_q": (2 * H * D, E), "attn_k": (HKV * D, E), "attn_v": (HKV * D, E),
          "attn_output": (E, H * D), "ffn_gate": (FF, E), "ffn_up": (FF, E), "ffn_down": (E, FF)}
t = {f"blk.64.{n}.weight": (rng.standard_normal(s) * 0.2).astype(np.float32) for n, s in shapes.items()}
for n in M.NORMS:
    t[f"blk.64.{n}.weight"] = (1 + 0.1 * rng.standard_normal(D if n in ("attn_q_norm", "attn_k_norm") else E)).astype(np.float32)
head = M.MTPHead(t, "cpu", rank=4, alpha=4)
for lin in head.lin.values():          # bf16 base -> float32 for an exact comparison, and a nonzero LoRA delta
    lin.w = lin.w.float()
    with torch.no_grad():
        lin.b.normal_(0, 0.05)

L, K = 12, 4
h0 = torch.randn(L, E)                 # target hidden H[q-1] per row
emb = torch.randn(L, E)                # token embeddings x_q
pos = torch.arange(100, 100 + L)
with torch.no_grad():
    par = head.forward_steps(h0, emb, pos, K)


def attend(q, keys, vals):
    """q [H, D]; keys/vals: lists of [HKV, D] -> [H, D] (GQA)."""
    kk = torch.stack(keys).repeat_interleave(H // HKV, 1)      # [n, H, D]
    vv = torch.stack(vals).repeat_interleave(H // HKV, 1)
    s = (q[None] * kk).sum(-1) / math.sqrt(D)                  # [n, H]
    p = torch.softmax(s, 0)
    return (p[..., None] * vv).sum(0)                          # [H, D]


with torch.no_grad():
    # step-0 entries for every position (process(): prompt + accepted tokens)
    x0, q0, g0, k0, v0 = head.qkv(h0, emb, pos)
    out0 = torch.stack([head.finish(x0[i:i + 1], attend(q0[i], list(k0[:i + 1]), list(v0[:i + 1]))[None], g0[i:i + 1])[0]
                        for i in range(L)])
    assert torch.allclose(out0, par[0], atol=1e-4), (out0 - par[0]).abs().max()
    worst = 0.0
    for p in range(L):                 # a draft chain starting at p (id_last = x_p, h = H[p-1])
        keys, vals = list(k0[:p + 1]), list(v0[:p + 1])
        h = out0[p]
        for k in range(1, K):
            q_ = p + k
            if q_ >= L:
                break
            x, qq, g, kk, vv = head.qkv(h[None], emb[q_:q_ + 1], pos[q_:q_ + 1])
            keys.append(kk[0])
            vals.append(vv[0])
            h = head.finish(x, attend(qq[0], keys, vals)[None], g)[0]
            worst = max(worst, float((h - par[k][q_]).abs().max()))
    assert worst < 1e-4, worst
print(f"parallel recursive forward == sequential inference simulation (max diff {worst:.2e})")
