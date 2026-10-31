"""Decode-speed benchmark against a running llama-server (OpenAI-compatible).

Sends a fixed set of prompts, reads llama-server's own `timings` block, and
reports per-prompt and mean generation tok/s (plus prefill tok/s and the
speculative-decoding acceptance rate when present).

usage: python bench.py [--url http://127.0.0.1:18080] [--max-tokens 512]
                       [--temp 0.6] [--label NAME] [--long] [--out results.jsonl]
API key: $LLAMA_API_KEY (or $QWEN_KEY) if the server needs one.
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python class implementing an LRU cache with per-entry TTL expiry. "
            "Include type hints, docstrings and a small pytest test suite.",
    "prose": "Explain to a curious high-school student how a CPU cache works and why it "
             "makes programs faster. Use an analogy, then go into some real detail.",
    "reasoning": "A train leaves city A at 09:00 going 80 km/h toward city B, 300 km away. "
                 "Another leaves B at 09:30 going 100 km/h toward A. At what time and how far "
                 "from A do they meet? Then: if the second train had left at 10:15 instead, "
                 "how would the answer change? Show your work.",
    "chat": "I'm planning a 3-day trip to Lisbon in November on a moderate budget. "
            "Give me a day-by-day itinerary with food recommendations.",
}


def long_prompt():
    # ~10k-token prompt: synthetic but varied log + a question about it.
    lines = []
    for i in range(900):
        lines.append(f"[{i:04d}] svc={['auth','db','api','cache','queue'][i % 5]} "
                     f"lat_ms={(i * 37) % 997} status={[200, 200, 200, 500, 429][i % 5]} "
                     f"user=u{(i * 13) % 211}")
    return ("Here is a service log:\n" + "\n".join(lines) +
            "\n\nWhich service has the most 500 errors, and what pattern do you see in the "
            "status codes? Then write a short Python script that computes per-service error rates "
            "from a log in this format.")


def run(url, prompt, max_tokens, temp, extra, min_p=0.05):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": min_p,
        "stream": False,
        "seed": 42,
    }
    body.update(extra)
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("LLAMA_API_KEY") or os.environ.get("QWEN_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    wall = time.time() - t0
    t = resp.get("timings", {})
    msg = resp["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    return {
        "gen_tps": t.get("predicted_per_second"),
        "gen_n": t.get("predicted_n"),
        "pp_tps": t.get("prompt_per_second"),
        "pp_n": t.get("prompt_n"),
        "draft_n": t.get("draft_n"),
        "draft_accepted": t.get("draft_n_accepted"),
        "wall_s": round(wall, 2),
        "preview": text[:160].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--min-p", type=float, default=0.05)
    ap.add_argument("--label", default="")
    ap.add_argument("--long", action="store_true", help="also run a ~10k-token prompt")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--extra", default="{}", help="JSON merged into the request body")
    a = ap.parse_args()
    extra = json.loads(a.extra)

    prompts = dict(PROMPTS)
    if a.long:
        prompts["long10k"] = long_prompt()

    run(a.url, "Say hi.", 8, a.temp, extra, a.min_p)  # warm-up
    results = {}
    for name, p in prompts.items():
        for _ in range(a.repeat):
            r = run(a.url, p, a.max_tokens, a.temp, extra, a.min_p)
            results.setdefault(name, []).append(r)
            acc = ""
            if r["draft_n"]:
                acc = f" accept={r['draft_accepted']}/{r['draft_n']} ({r['draft_accepted'] / r['draft_n']:.0%})"
            print(f"{a.label:24s} {name:10s} gen={r['gen_tps']:7.2f} t/s (n={r['gen_n']}) "
                  f"pp={r['pp_tps']:8.1f} t/s (n={r['pp_n']}){acc}", flush=True)
    short = [r["gen_tps"] for k, v in results.items() if k != "long10k" for r in v]
    print(f"{a.label:24s} MEAN(short)  gen={statistics.mean(short):7.2f} t/s  "
          f"min={min(short):.2f} max={max(short):.2f}")
    if a.out:
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"label": a.label, "temp": a.temp, "max_tokens": a.max_tokens,
                                "results": results, "mean_short": statistics.mean(short)}) + "\n")


if __name__ == "__main__":
    main()
