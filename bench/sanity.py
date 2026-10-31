"""Quick correctness sanity check against a running llama-server. Prints answers + speed.
usage: python sanity.py [url=http://127.0.0.1:18080]   (API key: $LLAMA_API_KEY or $QWEN_KEY)"""
import json
import os
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"
KEY = os.environ.get("LLAMA_API_KEY") or os.environ.get("QWEN_KEY", "")
CASES = [
    ("math", "What is 17 * 23 + 144 / 12? Answer with just the number.", "403"),
    ("logic", "Alice is older than Bob. Bob is older than Carol. Is Carol older than Alice? Answer yes or no.", "no"),
    ("code", "Write a Python one-liner that returns the sum of squares of the even numbers in a list `xs`. Only the code.", "%"),
    ("dutch", "Leg in twee zinnen uit wat een ideeën-bus is. Antwoord in het Nederlands.", "ideeën"),
    ("trick", "How many times does the letter r appear in the word 'strawberry'? Answer with just the number.", "3"),
]
for name, q, expect in CASES:
    body = {"messages": [{"role": "user", "content": q}], "max_tokens": 4096, "temperature": 1.0,
            "top_p": 0.95, "top_k": 20, "min_p": 0, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    ans = (d["choices"][0]["message"].get("content") or "").strip()
    t = d.get("timings", {})
    ok = expect.lower() in ans.lower()
    print(f"[{name}] {'PASS' if ok else 'CHECK'} {t.get('predicted_per_second', 0):.1f} t/s "
          f"({t.get('predicted_n')} tok)\n   {ans[:300]!r}")
