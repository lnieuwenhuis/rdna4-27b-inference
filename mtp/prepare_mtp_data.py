"""Build the MTP-head training corpus: rendered conversations plus byte ranges of the model's own output, for
llama-dump-hidden. Rendering uses llama-server's /apply-template, so the text is exactly what the model sees.

Sources (each one is used when its path is given; at least one is needed):
  captures  --captures FILE or $MTP_CAPTURES: a JSONL file written by context_proxy.py --capture (on-policy sessions
            of the served model). Agent steps whose full text (reasoning included, as the template preserves it) is
            a prefix of a later step's prompt are merged into that later sequence, so long sessions are processed once.
  faunix    --faunix DIR or $MTP_FAUNIX: Qwen3.8-27B reasoning traces (medium effort), sampled. The Hugging Face
            dataset faunix/Qwen3.8-27B-Distillation-40K; DIR is its local copy (the parquet files under data/).
  ukisai    --ukisai FILE or $MTP_UKISAI: Qwen3.8-27B terminal-agent steps, sampled. The ukisai agent SFT parquet
            (qwen38-agent-sft-combined-100pct.parquet).
usage: python -E -P prepare_mtp_data.py <out.jsonl> [faunix_rows=1200] [ukisai_rows=350] [max_bytes=120000]
                                        [--captures FILE] [--faunix DIR] [--ukisai FILE]
                                        [--url http://127.0.0.1:18080]
Rendering needs a running llama-server (--url); its API key, if any, comes from $LLAMA_API_KEY."""
import glob
import json
import os
import random
import re
import sys
import urllib.request

import pyarrow.parquet as pq


def _flag(args, name, default):
    """Remove `name VALUE` from args and return VALUE (or default)."""
    if name in args:
        i = args.index(name)
        value = args[i + 1]
        del args[i:i + 2]
        return value
    return default


sys.stdout.reconfigure(encoding="utf-8")
_args = sys.argv[1:]
CAPTURES = _flag(_args, "--captures", os.environ.get("MTP_CAPTURES"))
FAUNIX = _flag(_args, "--faunix", os.environ.get("MTP_FAUNIX"))
UKISAI = _flag(_args, "--ukisai", os.environ.get("MTP_UKISAI"))
URL = _flag(_args, "--url", "http://127.0.0.1:18080").rstrip("/") + "/apply-template"
if not _args or not (CAPTURES or FAUNIX or UKISAI):
    sys.exit(__doc__)
OUT = _args[0]
N_FAUNIX = int(_args[1]) if len(_args) > 1 else 1200
N_UKISAI = int(_args[2]) if len(_args) > 2 else 350
MAX_BYTES = int(_args[3]) if len(_args) > 3 else 120000  # ~30k tokens: the dump context
KEY = os.environ.get("LLAMA_API_KEY", "")
CUT = "⁣mtp-cut⁣"


def render(body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["prompt"]


def with_answer(base, messages, answer):
    """(prompt text, prompt+answer text): the answer as the template writes it into history."""
    prompt = render({**base, "messages": messages})
    full = render({**base, "messages": messages + [answer, {"role": "user", "content": CUT}]})
    full = full[:full.rfind("<|im_start|>", 0, full.rfind(CUT))]
    return prompt, full


def split_think(text):
    """'<think>r</think>\\n\\nanswer' -> (reasoning, answer)"""
    if text.lstrip().startswith("<think>") and "</think>" in text:
        r, a = text.lstrip()[len("<think>"):].split("</think>", 1)
        return r.strip(), a.strip()
    return "", text.strip()


def b(s):
    return len(s.encode("utf-8"))


records = []

# 1. on-policy captures
caps = []
for line in (open(CAPTURES, encoding="utf-8") if CAPTURES else []):
    c = json.loads(line)
    if c["response"]["finish_reason"] not in ("stop", "tool_calls"):
        continue
    req = c["request"]
    base = {k: v for k, v in req.items() if k not in ("messages", "stream", "stream_options")}
    r = c["response"]
    ans = {"role": "assistant", "content": r["content"] or ""}
    if r["reasoning_content"]:
        ans["reasoning_content"] = r["reasoning_content"]
    if r["tool_calls"]:
        ans["tool_calls"] = r["tool_calls"]
    try:
        prompt, full = with_answer(base, req["messages"], ans)
    except Exception as e:
        print("render failed:", type(e).__name__, e)
        continue
    if not full.startswith(prompt):
        # the answer as history differs from the generation prompt's tail (e.g. no think block): train on the tail
        common = next((i for i, (x, y) in enumerate(zip(prompt, full)) if x != y), min(len(prompt), len(full)))
        prompt = full[:common]
    caps.append({"prompt": prompt, "full": full, "spans": [(b(prompt), b(full))]})
print(f"captures: {len(caps)} responses")

# merge steps into the later prompt that contains them verbatim
caps.sort(key=lambda x: len(x["full"]))
covered = set()
for j, cj in enumerate(caps):
    for i in range(len(caps) - 1, j, -1):
        if caps[i]["prompt"].startswith(cj["full"]):
            caps[i]["spans"] += cj["spans"]
            covered.add(j)
            break
for j, c in enumerate(caps):
    if j not in covered:
        if b(c["full"]) > MAX_BYTES:
            print(f"  long sequence: {b(c['full']) // 4} tokens (trainable spans past the dump context are cut)")
        records.append({"id": f"cap-{j:05d}", "text": c["full"], "train": sorted(set(c["spans"]))})
print(f"captures: {len(records)} sequences after merging {len(covered)} steps into later ones")

def mostly_latin(text, max_frac=0.005):
    """The served vocabulary is pruned to ASCII + Latin: text with more foreign script than this turns into byte
    tokens the head would never see in real use, so such external rows are skipped (own captures are kept as-is)."""
    if not text:
        return True
    foreign = sum(1 for ch in text if ord(ch) > 0x24F and not ch.isspace())
    return foreign / len(text) <= max_frac


def cap_middle(text, head=2000, tail=4500):
    """Terminal-agent prompts are mostly long command output: keep the task statement and the latest output only.
    The dumped hidden states and the labels both come from this shortened context, so they stay consistent."""
    return text if len(text) <= head + tail + 500 else text[:head] + "\n[... earlier output omitted ...]\n" + text[-tail:]


# 2. faunix reasoning traces (medium effort, as generated)
rng = random.Random(7)
rows = pq.read_table(glob.glob(os.path.join(FAUNIX, "data", "*.parquet"))).to_pylist() if FAUNIX else []
n_rows = len(rows)
rows = [r for r in rows if r["tokens_total"] <= 6000 and mostly_latin(r["messages"][0]["content"] + r["messages"][1]["content"])]
print(f"faunix: {len(rows)} of {n_rows} rows are <= 6000 tokens and mostly Latin")
rng.shuffle(rows)
for k, r in enumerate(rows[:N_FAUNIX]):
    reasoning, answer = split_think(r["messages"][1]["content"])
    base = {"chat_template_kwargs": {"reasoning_effort": "medium"}}
    prompt, full = with_answer(base, [{"role": "user", "content": r["messages"][0]["content"]}],
                               {"role": "assistant", "content": answer, "reasoning_content": reasoning})
    if full.startswith(prompt):
        records.append({"id": f"faunix-{k:05d}", "text": full, "train": [(b(prompt), b(full))]})

# 3. ukisai terminal-agent steps
u = pq.read_table(UKISAI).to_pylist() if UKISAI else []
u = [r for r in u if sum(len(m["content"]) for m in r["conversations"]) <= 40000
     and mostly_latin("".join(m["content"] for m in r["conversations"]))]
rng.shuffle(u)
ASSISTANT_BLOCK = re.compile(r"<\|im_start\|>assistant\n<think>\n(.*?<\|im_end\|>\n)", re.S)
for k, r in enumerate(u[:N_UKISAI]):
    # an episode: every assistant turn is Qwen3.8 output, so every one is trained on (the history renders with its
    # reasoning, as preserve_thinking does), not just the last
    msgs = []
    for m in r["conversations"]:
        if m["role"] == "assistant":
            reasoning, answer = split_think(m["content"])
            msgs.append({"role": "assistant", "content": answer, "reasoning_content": reasoning})
        else:
            msgs.append({"role": m["role"], "content": cap_middle(m["content"]) if m["role"] == "user" else m["content"]})
    if msgs[-1]["role"] != "assistant":
        continue
    _, full = with_answer({}, msgs[:-1], msgs[-1])
    spans = [(b(full[:mt.start(1)]), b(full[:mt.end(1)])) for mt in ASSISTANT_BLOCK.finditer(full)]
    if spans:
        records.append({"id": f"ukisai-{k:05d}", "text": full, "train": spans})

# decontamination: nothing the A/B evaluates on may be trained on (replay.py = the agentbench Inventory project,
# bench.py prompts and its synthetic service log, sanity.py questions, long_test.py's planted-fact log)
EVAL_MARKERS = [
    "inventory.py", "Inventory.total_value", "test_inventory",                       # replay.py / agent_bench.py
    "LRU cache with per-entry TTL", "how a CPU cache works", "train leaves city A",   # bench.py
    "3-day trip to Lisbon", "svc=auth", "svc=db",
    "17 * 23 + 144 / 12", "Bob is older than Carol", "sum of squares of the even numbers",  # sanity.py
    "ideeën-bus", "letter r appear in the word 'strawberry'",
    "violet-harbor-42", "region=eu-west",                                           # long_test.py
]
markers = [m.lower() for m in EVAL_MARKERS]
kept, dropped = [], {}
for rec in records:
    low = rec["text"].lower()
    hit = next((m for m in markers if m in low), None)
    if hit:
        dropped[hit] = dropped.get(hit, 0) + 1
    else:
        kept.append(rec)
records = kept
print(f"decontamination: dropped {sum(dropped.values())} sequences {dropped}")

with open(OUT, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
by = {}
for rec in records:
    src = rec["id"].split("-")[0]
    n_all, n_train = by.get(src, (0, 0))
    by[src] = (n_all + b(rec["text"]), n_train + sum(e - s for s, e in rec["train"]))
print(f"wrote {len(records)} sequences to {OUT}")
for src, (n_all, n_train) in by.items():
    print(f"  {src}: ~{n_all / 4 / 1e6:.2f}M tokens to read, ~{n_train / 4 / 1e6:.2f}M trainable")
