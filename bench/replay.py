"""Replays captured OpenCode agent requests against the server and reports aggregate decode speed, so
speculative-decoding settings can be compared on a real agent workload.

The captured requests come from --captures PATH (or $REPLAY_CAPTURES), either
  - a JSONL file written by context_proxy.py --capture FILE: one record per chat completion that went through the
    proxy, {"time", "request", "response"}; the "request" (the body as forwarded upstream) is replayed, or
  - a directory of *.json files, each one /chat/completions request body (replayed in file-name order).
The first request is skipped: in an OpenCode session it is the title-generation request, not agent work.
Each request is sent non-streamed with max_tokens and a fixed seed per repeat; the result line is printed and
appended to --out (default: $RDNA4_WORK_DIR/replay-results.jsonl, else work/ in the repository root).
usage: python replay.py <label> [repeats=2] [max_tokens=600] [--captures PATH] [--server http://127.0.0.1:18080]
                        [--out FILE]
API key: $LLAMA_API_KEY (or $QWEN_KEY) if the server needs one."""
import glob
import json
import os
import sys
import urllib.request


def _flag(args, name, default):
    """Remove `name VALUE` from args and return VALUE (or default)."""
    if name in args:
        i = args.index(name)
        value = args[i + 1]
        del args[i:i + 2]
        return value
    return default


args = sys.argv[1:]
captures = _flag(args, "--captures", os.environ.get("REPLAY_CAPTURES"))
server = _flag(args, "--server", "http://127.0.0.1:18080").rstrip("/")
work = os.environ.get("RDNA4_WORK_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                        "work")
out = _flag(args, "--out", os.path.join(work, "replay-results.jsonl"))
if not args or not captures:
    sys.exit(__doc__)
label = args[0]
repeats = int(args[1]) if len(args) > 1 else 2
max_tokens = int(args[2]) if len(args) > 2 else 600
key = os.environ.get("LLAMA_API_KEY") or os.environ.get("QWEN_KEY", "")
if os.path.isdir(captures):
    bodies = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(captures, "*.json")))]
else:
    with open(captures, encoding="utf-8") as fh:
        bodies = [json.loads(line)["request"] for line in fh if line.strip()]
bodies = bodies[1:]  # skip the title-generation request
if not bodies:
    sys.exit(f"no requests to replay in {captures}")

tot_tok = tot_ms = acc = drafted = 0
per = []
for rep in range(repeats):
    for captured in bodies:
        body = dict(captured)
        body.update({"stream": False, "max_tokens": max_tokens, "seed": 1234 + rep})
        body.pop("max_completion_tokens", None)
        body.pop("stream_options", None)
        req = urllib.request.Request(server + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=900) as r:
            t = json.load(r).get("timings", {})
        tot_tok += t.get("predicted_n", 0)
        tot_ms += t.get("predicted_ms", 0)
        acc += t.get("draft_n_accepted", 0) or 0
        drafted += t.get("draft_n", 0) or 0
        per.append(round(t.get("predicted_per_second", 0), 1))
res = {"label": label, "requests": len(per), "gen_tokens": tot_tok, "decode_tps": round(tot_tok / (tot_ms / 1000), 1),
       "accept": round(acc / drafted, 2) if drafted else None, "per_request_tps": per}
print(json.dumps(res))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(res) + "\n")
