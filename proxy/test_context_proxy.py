"""Offline tests for context_proxy (run: python -I test_context_proxy.py)."""
import copy
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import context_proxy as cp  # noqa: E402

tmp = tempfile.mkdtemp()
cp.LOG = os.path.join(tmp, "proxy.log")
cp.NOTES_FILE = os.path.join(tmp, "notes.json")
big = "x" * 5000


def msgs(*roles):
    out = []
    for r in roles:
        if r == "A+tool":
            out.append({"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function",
                                                                             "function": {"name": "f", "arguments": "{}"}}]})
        elif r == "tool":
            out.append({"role": "tool", "tool_call_id": "t1", "content": big})
        else:
            out.append({"role": r, "content": "hi"})
    return out


cp._notes.clear()
cp._pending.clear()

# 1. current turn's tool loop (no final answer after the tools) is untouched
m = msgs("user", "A+tool", "tool", "A+tool", "tool")
assert cp.rewrite(m, wait=False) == (0, 0, 0) and m[2]["content"] == big

# 2. answered turn without notes -> truncation fallback; the new turn's result stays whole
m = msgs("user", "A+tool", "tool", "assistant", "user", "A+tool", "tool")
c, t, removed = cp.rewrite(m, wait=False)
assert (c, t) == (0, 1) and m[2]["content"].startswith("x" * cp.KEEP) and m[6]["content"] == big

# 3. answered turn with notes -> notes replace the raw result
cp._notes[cp._h(big)] = "- version 1.2.3 released 2026-10-27\n- https://example.com/release"
m = msgs("user", "A+tool", "tool", "assistant", "user")
c, t, removed = cp.rewrite(m, wait=False)
assert (c, t) == (1, 0) and m[2]["content"].startswith("[Notes on") and "1.2.3" in m[2]["content"]
assert removed > 4000

# 4. idempotent (stable prefix for the prompt cache)
again = copy.deepcopy(m)
assert cp.rewrite(again, wait=False) == (0, 0, 0) and again == m

# 5. short results stay whole
m = msgs("user", "A+tool", "tool", "assistant", "user")
m[2]["content"] = "short"
assert cp.rewrite(m, wait=False) == (0, 0, 0)

# 6. response parsing on SSE streams and plain JSON
sse_stop = (b'data: {"choices":[{"delta":{"content":"hi "},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"there"},"finish_reason":"stop"}],"timings":{"cache_n":5}}\n\n'
            b'data: [DONE]\n\n')
sse_tool = b'data: {"choices":[{"delta":{"tool_calls":[]},"finish_reason":"tool_calls"}]}\n\ndata: [DONE]\n\n'
sse_think = (b'data: {"choices":[{"delta":{"reasoning_content":"hm"},"finish_reason":null}]}\n\n'
             b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
assert cp._parse_response(sse_stop, True) == ("stop", "hi there", "", {"cache_n": 5})
assert cp._parse_response(sse_tool, True)[0] == "tool_calls"
assert cp._parse_response(sse_think, True)[:3] == ("stop", "ok", "hm")
plain = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}).encode()
assert cp._parse_response(plain, False) == ("stop", "ok", "", {})
assert cp._parse_response(b"garbage", False) == (None, "", "", {})

# 6a. streamed tool-call deltas are joined for --capture
sse_calls = (b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{\\"pa"}}]}}]}\n\n'
             b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\\": 1}"}}]},"finish_reason":"tool_calls"}]}\n\n'
             b'data: [DONE]\n\n')
assert cp._tool_calls(sse_calls, True) == [{"id": "c1", "type": "function",
                                            "function": {"name": "read", "arguments": '{"path": 1}'}}]

# 6b. notes output cut off by the token limit keeps what was written
assert cp._salvage_json('{"r1": "a", "r2": "b"}') == {"r1": "a", "r2": "b"}
assert cp._salvage_json('{"r1": "a", "r2": "partial no') == {"r1": "a", "r2": "partial no"}
assert cp._salvage_json('{"r1": "a"') == {"r1": "a"}


# 7. background job against a fake llama-server: cache-exact condense prompt, notes per result, prewarm cut
def fake_template(body):
    out = "<|im_start|>system\nsys<|im_end|>\n"
    for x in body["messages"]:
        think = f"<think>{x['reasoning_content']}</think>" if x.get("reasoning_content") else ""
        out += f"<|im_start|>{x['role']}\n{think}{cp._text(x.get('content'))}<|im_end|>\n"
    return out + "<|im_start|>assistant\n<think>\n"


seen = []


class Fake(BaseHTTPRequestHandler):
    def version_string(self):
        return "llama.cpp"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append((self.path, body, self.headers.get("Authorization")))
        if self.path == "/apply-template":
            out = {"prompt": fake_template(body)}
        elif body.get("n_predict"):
            out = {"content": json.dumps({"r1": "note one", "r2": "note two"}), "timings": {"cache_n": 100}}
        else:
            out = {"content": "", "timings": {"cache_n": 90, "prompt_n": 10}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class Other(Fake):  # another program on llama-server's port (e.g. some other local web UI started on it)
    def version_string(self):
        return "other-web-ui/1.0"


other = ThreadingHTTPServer(("127.0.0.1", 0), Other)
threading.Thread(target=other.serve_forever, daemon=True).start()
cp.UPSTREAM = ("127.0.0.1", other.server_address[1])
cp._identity["checked"] = 0
assert not cp._upstream_is_llama()
try:
    cp._post("/completion", {}, "Bearer k", 5)
    raise AssertionError("posted to a non-llama upstream")
except RuntimeError:
    pass
other.shutdown()

srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=srv.serve_forever, daemon=True).start()
cp.UPSTREAM = ("127.0.0.1", srv.server_address[1])
cp._identity["checked"] = 0
assert cp._upstream_is_llama()
cp._notes.clear()
r1, r2 = "a" * 3000, "b" * 4000
request = {"tools": [{"type": "function", "function": {"name": "web_fetch"}}], "stream": True,
           "chat_template_kwargs": {"reasoning_effort": "medium"},
           "messages": [{"role": "user", "content": "q"},
                        {"role": "assistant", "content": "", "tool_calls": [
                            {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{\"q\":1}"}},
                            {"id": "c2", "type": "function", "function": {"name": "web_fetch", "arguments": "{\"u\":2}"}}]},
                        {"role": "tool", "tool_call_id": "c1", "content": r1},
                        # OpenCode injects user-role updates between tool calls; they do not end the turn
                        {"role": "user", "content": "<system-update>catalog changed</system-update>"},
                        {"role": "tool", "tool_call_id": "c2", "content": r2}]}
cp._after_turn(request, "the answer", "my reasoning", "Bearer k")
paths = [p for p, _, _ in seen]
assert paths == ["/apply-template", "/completion", "/apply-template", "/completion"], paths
tpl, comp = seen[0][1], seen[1][1]
assert tpl["chat_template_kwargs"] == request["chat_template_kwargs"] and tpl["tools"] == request["tools"]
assert "stream" not in tpl and all(a == "Bearer k" for _, _, a in seen)
assert comp["prompt"].startswith(fake_template(dict(request))[:-len("<|im_start|>assistant\n<think>\n")])
assert comp["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
assert "r1: web_search" in comp["prompt"] and "r2: web_fetch" in comp["prompt"]
assert comp["json_schema"]["required"] == ["r1", "r2"]
assert cp._notes[cp._h(r1)] == "note one" and cp._notes[cp._h(r2)] == "note two"
warm = seen[3][1]["prompt"]
# this client sent no reasoning back, so the prewarm leaves it out too
assert warm.endswith("<|im_start|>assistant\nthe answer<|im_end|>\n") and cp.CUT_MARKER not in warm
assert "note one" in warm and r1 not in warm and seen[3][1]["n_predict"] == 0
assert json.load(open(cp.NOTES_FILE, encoding="utf-8"))[cp._h(r2)] == "note two"

# the next real request is rewritten to exactly the prewarmed prefix
nxt = copy.deepcopy(request["messages"]) + [{"role": "assistant", "content": "the answer"},
                                            {"role": "user", "content": "follow-up"}]
cp.rewrite(nxt)
assert fake_template({"messages": nxt}).startswith(warm)

# 8. a client that sends reasoning back (OpenCode) gets it in the prewarm; notes already exist, so no condense
seen.clear()
request2 = copy.deepcopy(request)
request2["messages"][1]["reasoning_content"] = "earlier reasoning"
cp._after_turn(request2, "the answer", "my reasoning", "Bearer k")
assert [p for p, _, _ in seen] == ["/apply-template", "/completion"]
warm2 = seen[1][1]["prompt"]
assert warm2.endswith("<|im_start|>assistant\n<think>my reasoning</think>the answer<|im_end|>\n")
nxt2 = copy.deepcopy(request2["messages"]) + [{"role": "assistant", "content": "the answer",
                                               "reasoning_content": "my reasoning"},
                                              {"role": "user", "content": "follow-up"}]
cp.rewrite(nxt2)
assert fake_template({"messages": nxt2}).startswith(warm2)
srv.shutdown()
print("all proxy tests passed")
