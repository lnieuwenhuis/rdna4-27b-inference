"""Long-context check: builds a prompt of roughly N tokens (synthetic service log with a planted fact),
asks for the fact, and reports prefill/decode speed and whether the answer is right.
usage: python long_test.py <approx_tokens> [url] [seed=7]
(a different seed gives a different log, so parallel streams don't share a cached prefix)
API key: $LLAMA_API_KEY (or $QWEN_KEY) if the server needs one."""
import json
import os
import random
import sys
import time
import urllib.request

n_tok = int(sys.argv[1])
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:18080"
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
lines, i = [], 0
needle_at = int(n_tok * 0.37 / 22)  # roughly 37% into the prompt
while len(lines) * 22 < n_tok:  # ~22 tokens per line
    if i == needle_at:
        lines.append("[NOTE] The deployment password for the staging cluster is 'violet-harbor-42'.")
    svc = random.choice(["auth", "db", "api", "cache", "queue", "billing"])
    lines.append(f"[{i:06d}] svc={svc} lat_ms={random.randint(1, 999)} status={random.choice([200, 200, 201, 404, 500])} "
                 f"user=u{random.randint(1, 5000)} region={random.choice(['eu-west', 'us-east', 'ap-south'])}")
    i += 1
prompt = "Service log follows.\n" + "\n".join(lines) + \
    "\n\nQuestion: what is the deployment password for the staging cluster? Answer with just the password."
body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0.6,
        "chat_template_kwargs": {"reasoning_effort": "low"}}
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + (os.environ.get("LLAMA_API_KEY")
                                                                or os.environ.get("QWEN_KEY", ""))})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as r:
    d = json.load(r)
t = d.get("timings", {})
ans = (d["choices"][0]["message"].get("content") or "").strip()
print(json.dumps({"target_tokens": n_tok, "prompt_tokens": t.get("prompt_n"), "prefill_tps": round(t.get("prompt_per_second", 0)),
                  "prefill_s": round(t.get("prompt_ms", 0) / 1000), "gen_tokens": t.get("predicted_n"),
                  "decode_tps": round(t.get("predicted_per_second", 0), 1), "correct": "violet-harbor-42" in ans,
                  "answer": ans[:80], "wall_s": round(time.time() - t0)}))
