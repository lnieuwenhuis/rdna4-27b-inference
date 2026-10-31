"""Retrain the MTP draft head of the served GGUF by self-distillation (LoRA on the head's matrices, norms full).

Data: dumps from llama-dump-hidden (<prefix>.bin + .idx.jsonl): tokens, train mask, keep mask and the target's
final hidden rows. Target distribution at each position = the served model's own next-token distribution, computed
from the dumped hidden state with the GGUF's output matrix and the server's sampler (temp 1.0, top-k 20,
top-p 0.95). Loss: forward KL(target || draft) per recursive draft step, weighted toward early steps. Metric: the
exact speculative-sampling acceptance probability sum(min(p, q)) per step, and the expected accepted draft length.

usage:
  python train_mtp.py eval  --data PREFIX [--max-windows N]                 # original head only
  python train_mtp.py train --data PREFIX --eval-data PREFIX --out DIR [--epochs 2 --rank 64 --lr 1e-4 ...]
  --gguf FILE: the served GGUF with the MTP head (default $MTP_GGUF; required one way or the other)
  --out DIR defaults to runs/mtp under $RDNA4_WORK_DIR, else under work/ in the repository root
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_model as M  # noqa: E402

WORK = os.environ.get("RDNA4_WORK_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                        "work")
TOP_K, TOP_P, TEMP = 20, 0.95, 1.0
STEP_WEIGHTS = [1.0, 0.8, 0.6, 0.4]


class Dump:
    """Windows of up to `win` consecutive kept positions from a llama-dump-hidden dump."""

    def __init__(self, prefix, win, n_embd=M.N_EMBD):
        self.bin = np.memmap(prefix + ".bin", dtype=np.uint8, mode="r")
        self.win, self.n_embd = win, n_embd
        self.windows = []
        for line in open(prefix + ".idx.jsonl", encoding="utf-8"):
            ix = json.loads(line)
            off, n, n_keep = ix["offset"], ix["n_tokens"], ix["n_keep"]
            hdr = np.frombuffer(self.bin, dtype=np.int32, count=2, offset=off)
            assert hdr[0] == n and hdr[1] == n_keep, ix
            o_tok = off + 8
            o_train = o_tok + 4 * n
            o_keep = o_train + n
            o_hid = o_keep + n
            keep = np.frombuffer(self.bin, dtype=np.uint8, count=n, offset=o_keep).astype(bool)
            train = np.frombuffer(self.bin, dtype=np.uint8, count=n, offset=o_train).astype(bool)
            row_of = np.cumsum(keep) - 1                     # hidden row index of each kept position
            # contiguous kept runs, cut into windows; a window needs one leading row (H[q-1] for its first q)
            runs = np.flatnonzero(np.diff(np.concatenate([[0], keep.astype(np.int8), [0]])))
            for s, e in zip(runs[::2], runs[1::2]):
                # a window [a, a+win] predicts tokens a+2 .. a+win, so the next one starts at a+win-1
                for a in range(s, e - 1, win - 1):
                    b = min(e, a + win + 1)
                    if train[a + 1:b].any():
                        self.windows.append((ix["id"], o_tok, o_train, o_hid, n, int(row_of[a]), a, b))

    def get(self, w):
        _id, o_tok, o_train, o_hid, n, row0, a, b = w
        tok = np.frombuffer(self.bin, dtype=np.int32, count=n, offset=o_tok)[a:b]
        train = np.frombuffer(self.bin, dtype=np.uint8, count=n, offset=o_train)[a:b]
        hid = np.frombuffer(self.bin, dtype=np.float16, count=(b - a) * self.n_embd,
                            offset=o_hid + 2 * self.n_embd * row0).reshape(b - a, self.n_embd)
        return tok.astype(np.int64), train.astype(bool), hid, a


def target_probs(logits):
    """The server's sampling distribution: temp, top-k, then top-p over the top-k (llama.cpp sampler order)."""
    v, i = torch.topk(logits.float() / TEMP, TOP_K, dim=-1)
    p = torch.softmax(v, -1)
    keep = (p.cumsum(-1) - p) < TOP_P                          # keep tokens until the mass reaches top-p
    p = p * keep
    return p / p.sum(-1, keepdim=True), i


def step_loss(out_rows, p, pi, lm_head, chunk=256):
    """KL(p || q) and sum(min(p, q)) for draft hidden rows `out_rows` [n, n_embd] against target (p, pi)."""
    kl_sum, acc_sum = 0.0, 0.0
    for c in range(0, out_rows.shape[0], chunk):
        def f(x, pc, pic):
            lq = torch.log_softmax(F.linear(x, lm_head).float(), -1)
            lqi = lq.gather(-1, pic)
            kl = (pc * (torch.log(pc.clamp_min(1e-12)) - lqi)).sum(-1)
            acc = torch.minimum(pc, lqi.exp()).sum(-1)
            return kl.sum(), acc.sum()
        x = out_rows[c:c + chunk]
        kl, acc = checkpoint(f, x, p[c:c + chunk], pi[c:c + chunk], use_reentrant=False) if x.requires_grad \
            else f(x, p[c:c + chunk], pi[c:c + chunk])
        kl_sum = kl_sum + kl
        acc_sum = acc_sum + acc.detach()
    return kl_sum, acc_sum


def window_layout(tok, train, a):
    """Alignment of one window holding positions a .. b-1 (rows 0 .. n-1). The head's state for position q
    (q = a+1 .. b-1, row i = q-a-1) takes h = H[q-1] and token x_q, and predicts x_{q+1}, whose target
    distribution comes from H[q]. Returns (rows of H used as input, rows of H giving the target, input token ids,
    positions, mask of states whose predicted token x_{q+1} is model output and inside the window)."""
    n = len(tok)
    h_in = np.arange(0, n - 1)                  # H[q-1]
    h_tgt = np.arange(1, n)                     # H[q]
    x_in = tok[1:]                              # x_q
    pos = np.arange(a + 1, a + n)
    valid = np.zeros(n - 1, dtype=bool)
    valid[:-1] = train[2:]                      # x_{q+1} is model output (none beyond the window's last token)
    return h_in, h_tgt, x_in, pos, valid


def run_window(head, w_data, emb_table, lm_head, n_steps, device, grad):
    tok, train, hid, a = w_data
    H = torch.from_numpy(np.array(hid)).to(device, torch.bfloat16)      # H[a .. b-1]
    h_in, h_tgt, x_in, pos, valid = window_layout(tok, train, a)
    with torch.no_grad():
        p, pi = target_probs(F.linear(H[h_tgt], lm_head))
        emb = emb_table[torch.from_numpy(x_in)].to(device, torch.bfloat16)
    with torch.set_grad_enabled(grad):
        outs = head.forward_steps(H[h_in], emb, torch.from_numpy(pos).to(device), n_steps)
    valid_next = torch.from_numpy(valid).to(device)
    total, stats = 0.0, []
    for k, out in enumerate(outs):
        m = valid_next.clone()
        m[:k] = False                                                    # chains must start inside the window
        n = int(m.sum())
        if n == 0:
            stats.append((0, 0.0, 0.0))
            continue
        kl, acc = step_loss(out[m], p[m], pi[m], lm_head)
        total = total + STEP_WEIGHTS[k] * kl / n
        stats.append((n, float(kl), float(acc)))
    return total, stats


def summarize(acc):
    """acc[k] = (n, kl_sum, acc_sum): mean acceptance per step and expected accepted length (chained)."""
    alpha = [a / max(n, 1) for n, _, a in acc]
    exp_len, prod = 0.0, 1.0
    for al in alpha:
        prod *= al
        exp_len += prod
    return alpha, exp_len


def evaluate(head, ds, windows, emb_table, lm_head, n_steps, device, per_session=False):
    acc = [[0, 0.0, 0.0] for _ in range(n_steps)]
    sess = {}                                                   # sequence id -> per-step [n, kl, acc] sums
    head.eval()
    with torch.no_grad():
        for w in windows:
            _, stats = run_window(head, ds.get(w), emb_table, lm_head, n_steps, device, False)
            s = sess.setdefault(w[0], [[0, 0.0, 0.0] for _ in range(n_steps)])
            for k, (n, kl, a) in enumerate(stats):
                for tgt in (acc[k], s[k]):
                    tgt[0] += n
                    tgt[1] += kl
                    tgt[2] += a
    alpha, exp_len = summarize(acc)
    kl = [x[1] / max(x[0], 1) for x in acc]
    out = {"alpha": [round(x, 4) for x in alpha], "kl": [round(x, 4) for x in kl],
           "expected_accepted": round(exp_len, 3), "positions": acc[0][0]}
    if per_session:   # for paired comparisons between heads on the same sessions (compare_heads.py)
        out["sessions"] = {i: {"positions": v[0][0], "expected_accepted": round(summarize(v)[1], 4),
                               "alpha": [round(x, 4) for x in summarize(v)[0]]} for i, v in sess.items()}
    return out


def load(gguf, device, rank, alpha):
    names = [f"blk.64.{n}.weight" for n in M.LINEARS + M.NORMS] + ["token_embd.weight", "output.weight"]
    t0 = time.time()
    t = M.read_gguf(gguf, names)
    emb_table = torch.from_numpy(t.pop("token_embd.weight")).to(torch.bfloat16)      # CPU, gathered per window
    lm_head = torch.from_numpy(t.pop("output.weight")).to(device, torch.bfloat16)
    head = M.MTPHead(t, device, rank, alpha)
    print(f"loaded weights in {time.time() - t0:.0f}s", flush=True)
    return head, emb_table, lm_head


def split(ds, frac, seed=0):
    """Per sequence id (a whole session stays on one side): train, validation (watched during training and used to
    pick head_best.pt), test (only ever reported, never used for a decision)."""
    def bucket(i):
        h = int(hashlib.sha256(f"{seed}{i}".encode()).hexdigest(), 16) % 1000
        return "val" if h < frac * 1000 else "test" if h < 2 * frac * 1000 else "train"
    b = {i: bucket(i) for i in {w[0] for w in ds.windows}}
    return tuple([w for w in ds.windows if b[w[0]] == part] for part in ("train", "val", "test"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "train"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--eval-data", default=None, help="separate held-out dump (e.g. the replay benchmark turns)")
    ap.add_argument("--gguf", default=os.environ.get("MTP_GGUF"),
                    help="the served GGUF with the MTP head (default: $MTP_GGUF)")
    ap.add_argument("--out", default=os.path.join(WORK, "runs", "mtp"))
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--final-onpolicy", type=float, default=0.25,
                    help="fraction of the run at the end that trains on Swift's own sessions only")
    ap.add_argument("--own-weight", type=int, default=2,
                    help="times each on-policy capture window (id cap-*) appears per epoch vs online data")
    args = ap.parse_args()
    if not args.gguf:
        ap.error("no model: pass --gguf FILE or set MTP_GGUF")
    device = "cuda"
    torch.manual_seed(0)
    random.seed(0)

    head, emb_table, lm_head = load(args.gguf, device, args.rank if args.mode == "train" else 0, args.alpha)
    ds = Dump(args.data, args.win)
    train_w, held_w, test_w = split(ds, 0.05)
    if args.max_windows:
        train_w, held_w, test_w = (train_w[:args.max_windows], held_w[:max(1, args.max_windows // 10)],
                                   test_w[:max(1, args.max_windows // 10)])
    ev_ds = Dump(args.eval_data, args.win) if args.eval_data else None
    ev_w = ev_ds.windows if ev_ds else []
    print(f"{len(train_w)} train, {len(held_w)} validation, {len(test_w)} test windows, {len(ev_w)} eval-set windows",
          flush=True)

    held_own = [w for w in held_w if w[0].startswith("cap-")]
    held_ext = [w for w in held_w if not w[0].startswith("cap-")]
    test_own = [w for w in test_w if w[0].startswith("cap-")]

    def report(tag, test=False):
        # validation: on-policy (Swift's own sessions) decides, external data is watched for drift only.
        # test: Swift's own held-out sessions that no decision ever looks at, reported for the original head and the
        # two checkpoints that go to the A/B (mixed-phase end, final)
        r = {"held_out": evaluate(head, ds, held_own, emb_table, lm_head, args.steps, device),
             "held_external": evaluate(head, ds, held_ext, emb_table, lm_head, args.steps, device)}
        if test:
            r["test"] = evaluate(head, ds, test_own, emb_table, lm_head, args.steps, device, per_session=True)
        if ev_ds:
            r["eval_set"] = evaluate(head, ev_ds, ev_w, emb_table, lm_head, args.steps, device)
        print(f"[{tag}] {json.dumps(r)}", flush=True)
        return r

    if args.mode == "eval":
        report("original", test=True)
        return

    os.makedirs(args.out, exist_ok=True)
    base = report("original", test=True)
    params = [p for p in head.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    # Swift's own sessions count more than base-Qwen text: repeat their windows within each epoch
    train_w = [w for w in train_w for _ in range(args.own_weight if w[0].startswith("cap-") else 1)]
    total_steps = int(len(train_w) * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps))))
    step, t0, best = 0, time.time(), None
    run = {"args": vars(args), "original": base, "evals": []}
    n_iter = int(len(train_w) * args.epochs)
    # schedule: the mix first, then a final phase on Swift's own sessions only (while the learning rate decays),
    # so the head ends on the served model's on-policy distribution
    n_final = int(n_iter * args.final_onpolicy)
    own = [w for w in train_w if w[0].startswith("cap-")]
    sched_w = []
    while len(sched_w) < n_iter - n_final:
        sched_w += random.sample(train_w, len(train_w))
    sched_w = sched_w[:n_iter - n_final]
    while len(sched_w) < n_iter:
        sched_w += random.sample(own, len(own))
    sched_w = sched_w[:n_iter]
    print(f"{n_iter} windows: {n_iter - n_final} mixed, then {n_final} on-policy only", flush=True)
    for it in range(n_iter):
        head.train()
        loss, stats = run_window(head, ds.get(sched_w[it]), emb_table, lm_head, args.steps, device, True)
        if torch.is_tensor(loss):
            (loss / args.accum).backward()
        if (it + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 20 == 0:
                print(f"step {step}/{total_steps} loss {float(loss):.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / (it + 1):.2f}s/window", flush=True)
            if step % args.eval_every == 0 or it == n_iter - 1:
                r = report(f"step {step}")
                run["evals"].append({"step": step, **r})
                score = r["held_out"]["expected_accepted"]
                if best is None or score > best:
                    best = score
                    save(head, args.out, "head_best.pt")
                json.dump(run, open(os.path.join(args.out, "run.json"), "w"), indent=1)
        if n_final and it == n_iter - n_final - 1:
            # the mixed-data checkpoint, right before the Swift-only phase: A/B'd separately, so the effect of the
            # on-policy specialization can be told apart from the mixed training
            r = report("end of mixed phase", test=True)
            run["mixed_phase_end"] = {"step": step, **r}
            save(head, args.out, "head_mixed.pt")
            json.dump(run, open(os.path.join(args.out, "run.json"), "w"), indent=1)
    save(head, args.out, "head_final.pt")
    run["final"] = {"step": step, **report("final", test=True)}
    json.dump(run, open(os.path.join(args.out, "run.json"), "w"), indent=1)
    print(f"done: best held-out expected accepted length {best} (original {base['held_out']['expected_accepted']})")


def save(head, out, name):
    """Merged float32 weights of every head tensor (export_mtp.py quantizes them into a GGUF copy)."""
    sd = {f"blk.64.{n}.weight": head.lin[n.replace(".", "_")].merged().cpu() for n in M.LINEARS}
    sd.update({f"blk.64.{n}.weight": head.norm[n.replace(".", "_")].detach().float().cpu() for n in M.NORMS})
    torch.save(sd, os.path.join(out, name))


if __name__ == "__main__":
    main()
