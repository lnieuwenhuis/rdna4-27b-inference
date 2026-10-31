"""Speculative-decoding statistics from the llama-server log (needs the server started with -lv 4): per drafter
(draft-mtp, ngram-mod, ...) the drafts, tokens and acceptance per draft position. The server prints CUMULATIVE
counters (since it started) after every request, so the latest line is the total; with a byte offset, the counters
at that point are subtracted (statistics since then).
usage: python -I spec_stats.py [log] [byte_offset=0]   (log defaults to $LLAMA_SERVER_LOG)"""
import os
import re
import sys

log = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LLAMA_SERVER_LOG")
if not log:
    sys.exit(__doc__)
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
pat = re.compile(r"statistics\s+(\S+): #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), #gen drafts =\s*(\d+), "
                 r"#acc drafts =\s*(\d+), #gen tokens =\s*(\d+), #acc tokens =\s*(\d+)"
                 r"(?:, #mean acc len = [\d.]+, #acc rate/pos = \(([^)]*)\))?")


def snapshot(text):
    """Latest cumulative counters per drafter in `text` (accepted-per-position as counts)."""
    out = {}
    for m in pat.finditer(text):
        calls_acc = int(m.group(4))
        out[m.group(1)] = {
            "requests": int(m.group(2)), "calls_gen": int(m.group(3)), "verify": calls_acc, "gen_drafts": int(m.group(5)),
            "acc_drafts": int(m.group(6)), "gen_tokens": int(m.group(7)), "acc_tokens": int(m.group(8)),
            "pos": [float(x) * calls_acc for x in m.group(9).split(",")] if m.group(9) else []}
    return out


with open(log, "rb") as f:
    data = f.read().decode("utf-8", "replace")
cut = len(data.encode("utf-8")[:start].decode("utf-8", "ignore"))
before, now = snapshot(data[:cut]), snapshot(data)
for name, t in now.items():
    b = before.get(name)
    if b and b["requests"] <= t["requests"]:
        for k in ("requests", "calls_gen", "verify", "gen_drafts", "acc_drafts", "gen_tokens", "acc_tokens"):
            t[k] -= b[k]
        t["pos"] = [x - (b["pos"][i] if i < len(b["pos"]) else 0) for i, x in enumerate(t["pos"])]
    rate = t["acc_tokens"] / t["gen_tokens"] if t["gen_tokens"] else 0
    hit = t["gen_drafts"] / t["calls_gen"] if t.get("calls_gen") else 0
    per_draft = t["acc_tokens"] / t["gen_drafts"] if t["gen_drafts"] else 0
    print(f"{name:10s}: {t['requests']} requests, {t['gen_drafts']} drafts from {t.get('calls_gen', 0)} attempts "
          f"(hit rate {hit:.3f}), {per_draft:.2f} accepted tokens per draft, "
          f"{t['acc_tokens']}/{t['gen_tokens']} drafted tokens accepted = {rate:.3f}")
    if t["pos"] and t["verify"]:
        pos = [p / t["verify"] for p in t["pos"]]
        last = max(i for i, p in enumerate(pos) if p > 0) + 1 if any(pos) else 4
        print("            reached per position: " + ", ".join(f"{p:.3f}" for p in pos[:max(last, 4)]))
