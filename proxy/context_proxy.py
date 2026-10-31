"""Context-condensing reverse proxy in front of llama-server, used by both the web chat and the coding agent.

Everything is forwarded unchanged (streaming-safe), except /chat/completions requests:

1. When a response finishes a turn (finish_reason "stop", i.e. the model answered instead of calling a tool), the
   proxy asks the model, in the background, to condense that turn's tool results into notes: every fact relevant
   to the conversation, nothing else. The request is rendered with /apply-template from the exact request the
   server just processed (same messages, tools and template kwargs), so its prefix is byte-identical and the
   prompt cache already holds it: only the instruction and the notes are new work. Thinking is skipped by closing
   the think block in the prompt (changing enable_thinking would change the system prompt and miss the cache),
   and the output is grammar-constrained JSON with one field per tool result.
2. On later requests, tool results of already-answered turns are replaced by their notes. If notes are still
   being written, the request waits for them (the GPU is busy with them anyway), so the context changes only once.
   Results without notes (e.g. from before the proxy ran) fall back to their first KEEP characters.
3. Right after the notes exist, the proxy prefills the condensed history (up to where the next user message will
   start) so the next turn does not pay for re-reading the rewritten context.

4. Mid-turn cleanup: a long agent turn (one user message, many tool calls) is never "answered", so 1-3 never touch
   it, and past about 120k tokens KV streaming makes every decode step slower. When a tool-call response ends at
   CLEANUP_AT tokens or more, the next request of that conversation first gets a one-time cleanup of everything
   before its last CLEANUP_KEEP_STEPS tool-call steps: past reasoning is dropped (the template then writes an empty
   think block, as for a non-thinking turn) and tool results of CLEANUP_MIN_CHARS or more are replaced by notes
   (written by the same grammar-constrained request as in 1, on the cached prefix). The boundary is remembered by
   the id of its first tool call (CLEANUP_FILE), so later requests are rewritten identically and the cache keeps
   hitting; the next cleanup moves the boundary only once enough new material has piled up behind it.

Apart from 4, the current turn's tool loop is never touched, and rewriting is deterministic, so the prompt cache
keeps hitting. Notes are cached in NOTES_FILE by a hash of the original tool result.

usage: python context_proxy.py [listen_port=8090] [upstream_port=18080] [--capture FILE] [--no-condense]
                               [--cleanup-at TOKENS] [--state-dir DIR]
  --capture FILE       append every chat completion (request as sent + response) to FILE as JSONL (training data)
  --no-condense        forward only: no notes, no rewriting, no prefill (for the training-data generation instance)
  --cleanup-at TOKENS  context size that triggers the mid-turn cleanup (default CLEANUP_AT; 0 turns it off)
  --state-dir DIR      where the log, NOTES_FILE, CLEANUP_FILE and the debug dumps live (default: $RDNA4_WORK_DIR,
                       else work/ in the repository root; created if missing)
"""
import copy
import hashlib
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = ("127.0.0.1", 18080)  # set from argv in __main__
CAPTURE = None                   # --capture FILE
CONDENSE = True                  # --no-condense turns it off
KEEP = 600                       # fallback: characters kept when no notes exist
MIN_CONDENSE = 1500              # tool results shorter than this stay as they are
WAIT_FOR_NOTES_S = 180
MAX_NOTES = 3000                 # entries kept in NOTES_FILE (oldest dropped first)
# mid-turn cleanup (4): decode stays ~80 t/s up to ~120k tokens, then drops to ~45-65 t/s as KV streaming starts
CLEANUP_AT = 110000              # tokens (prompt + generated) after a tool-call response; 0 = off
CLEANUP_KEEP_STEPS = 8           # most recent tool-call steps left exactly as they are
CLEANUP_MIN_CHARS = 4000         # smaller tool results stay (most of the volume is in a few large ones)
CLEANUP_MIN_GAIN = 60000         # characters a cleanup must remove (~16k tokens) to be worth re-reading the rest
MAX_ANCHORS = 500
STATE_DIR = os.environ.get("RDNA4_WORK_DIR") or os.path.join(  # --state-dir DIR
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work")


def _state_paths(state_dir):
    """(LOG, NOTES_FILE, CLEANUP_FILE, DEBUG_FLAG, DEBUG_DIR) inside `state_dir`."""
    return tuple(os.path.join(state_dir, n) for n in ("context_proxy.log", "context_notes.json",
                                                      "context_cleanup.json", "proxy-debug.flag", "proxy-debug"))


LOG, NOTES_FILE, CLEANUP_FILE, DEBUG_FLAG, DEBUG_DIR = _state_paths(STATE_DIR)
HOP = {"connection", "keep-alive", "transfer-encoding", "upgrade", "proxy-connection", "te", "trailer", "host",
       "content-length"}
CUT_MARKER = "\u2063context-proxy-cut\u2063"

NOTES_HEADER = ("[Notes on this tool result; the raw output ({n} characters) was removed to save context and "
                "anything not in these notes is gone. If an answer depends on a detail that is not in them, call "
                "the tool again instead of guessing.]\n")
FALLBACK_NOTE = ("\n\n[{n} more characters of this tool result were removed to save context: it belongs to an "
                 "earlier, already-answered question. Call the tool again if you need those details.]")
_NOTES_BODY = (
    "exact names, versions, dates, numbers, URLs, file paths with line numbers, commands, "
    "identifiers, code that may be needed again and short quotes, plus anything the result showed was absent, "
    "wrong or failed. Leave out boilerplate, navigation and unrelated material. Terse bullet points, at most "
    "about 150 words per result.\n\n")
INSTRUCTION = (
    "Context maintenance (not from the user, do not answer the user): the raw tool results of this turn will be "
    "deleted from the conversation to save space and replaced by your notes. For each result listed below, write "
    "compact notes of everything in it that is or could become relevant to the user's requests in this "
    "conversation: " + _NOTES_BODY + "Tool results of this turn, in order:\n{items}")
MIDTURN_INSTRUCTION = (
    "Context maintenance (not from the user, do not answer the user; the task is still in progress and continues "
    "after this): the older raw tool results listed below will be deleted from the conversation to save space and "
    "replaced by your notes. For each result, write compact notes of everything in it that the rest of this task "
    "or the user's requests may still need: " + _NOTES_BODY
    + "Tool results to condense, in order:\n{items}")

_lock = threading.Condition()
_notes = {}      # sha256(original tool result) -> notes text
_pending = set()  # hashes being condensed right now
_gpu = threading.Lock()  # one background job at a time
_anchors = []    # first tool-call id of each mid-turn cleanup boundary (CLEANUP_FILE)
_want = {}       # tool-call id of a response that ended at >= CLEANUP_AT tokens -> that size


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def _debug_dump(kind, data):
    """With DEBUG_FLAG present, keep chat requests and prewarm prompts in DEBUG_DIR (for prompt-cache diagnosis)."""
    if os.path.exists(DEBUG_FLAG):
        os.makedirs(DEBUG_DIR, exist_ok=True)
        name = f"{time.strftime('%H%M%S')}-{time.time_ns() % 10**6:06d}-{kind}.json"
        with open(os.path.join(DEBUG_DIR, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)


def _load_notes():
    try:
        _notes.update(json.load(open(NOTES_FILE, encoding="utf-8")))
    except (OSError, ValueError):
        pass


def _save_notes():
    while len(_notes) > MAX_NOTES:
        del _notes[next(iter(_notes))]
    tmp = NOTES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_notes, f, ensure_ascii=False)
    os.replace(tmp, NOTES_FILE)


def _load_anchors():
    try:
        _anchors.extend(a for a in json.load(open(CLEANUP_FILE, encoding="utf-8")) if isinstance(a, str))
    except (OSError, ValueError, TypeError):
        pass


def _save_anchors():
    del _anchors[:-MAX_ANCHORS]
    tmp = CLEANUP_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_anchors, f)
    os.replace(tmp, CLEANUP_FILE)


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _last_final(messages):
    finals = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and not m.get("tool_calls")]
    return finals[-1] if finals else -1


def rewrite(messages, wait=True):
    """Replace tool results of answered turns by their notes (or the truncation fallback), in place.
    Returns (condensed, truncated, characters removed)."""
    end = _last_final(messages)
    olds = [m for m in messages[:max(end, 0)] if m.get("role") == "tool"
            and len(_text(m.get("content"))) >= MIN_CONDENSE and not _text(m.get("content")).startswith("[Notes on")]
    if not olds:
        return 0, 0, 0
    hashes = [_h(_text(m["content"])) for m in olds]
    if wait:
        deadline = time.time() + WAIT_FOR_NOTES_S
        with _lock:
            while any(h in _pending for h in hashes) and time.time() < deadline:
                _lock.wait(timeout=1)
    condensed = truncated = removed = 0
    for m, h in zip(olds, hashes):
        text = _text(m["content"])
        notes = _notes.get(h)
        if notes:
            m["content"] = NOTES_HEADER.format(n=len(text)) + notes.strip()
            condensed += 1
        else:
            m["content"] = text[:KEEP].rstrip() + FALLBACK_NOTE.format(n=len(text) - KEEP)
            truncated += 1
        removed += len(text) - len(m["content"])
    return condensed, truncated, removed


def _steps(messages):
    """Indices of the assistant messages that call tools."""
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("tool_calls")]


def _boundary(messages):
    """Index of the first message the mid-turn cleanup leaves alone: the latest known anchor in `messages`, or 0."""
    with _lock:
        anchors = set(_anchors)
    found = [i for i in _steps(messages) if any(tc.get("id") in anchors for tc in messages[i]["tool_calls"])]
    return max(found) if found else 0


def _big_results(messages):
    """[(message, hash, size)] of the raw tool results the mid-turn cleanup replaces by notes."""
    out = []
    for m in messages:
        text = _text(m.get("content")) if m.get("role") == "tool" else ""
        if len(text) >= CLEANUP_MIN_CHARS and not text.startswith("[Notes on"):
            out.append((m, _h(text), len(text)))
    return out


def cleanup(messages):
    """Mid-turn cleanup of everything before the boundary, in place: drop past reasoning and replace large tool
    results that have notes. Deterministic for a given boundary and notes. Returns characters removed."""
    end = _boundary(messages)
    removed = 0
    for m in messages[:end]:
        if m.get("role") == "assistant" and m.get("reasoning_content"):
            removed += len(m.pop("reasoning_content"))
    for m, h, size in _big_results(messages[:end]):
        notes = _notes.get(h)
        if notes:  # no notes (condensing failed): the result stays raw rather than being cut
            m["content"] = NOTES_HEADER.format(n=size) + notes.strip()
            removed += size - len(m["content"])
    return removed


def condense_history(messages, wait=True):
    """Everything the proxy rewrites in a chat request: answered turns (rewrite) and the mid-turn cleanup.
    Returns (condensed, truncated, cleanup characters removed, total characters removed)."""
    c, t, removed = rewrite(messages, wait)
    cleaned = cleanup(messages) if CLEANUP_AT else 0
    return c, t, cleaned, removed + cleaned


def prepare(request, auth):
    """Rewrite a chat request in place before it is forwarded. Returns True if anything changed."""
    messages = request["messages"]
    c, t, cleaned, removed = condense_history(messages)
    if CLEANUP_AT and _cleanup_due(messages):
        # holds this request: notes are written on its (cached) prefix, then it is rewritten again
        line = _checkpoint(request, auth)
        if line:
            more = cleanup(messages)  # only the newly covered region is left to do
            cleaned += more
            removed += more
            log(f"{line}, -{more} chars")
    if c or t or cleaned:
        log(f"rewrote request: {c} condensed, {t} truncated, cleanup -{cleaned} chars, -{removed} chars")
    return bool(c or t or cleaned)


def _checkpoint(request, auth):
    """One mid-turn cleanup: move the boundary to the CLEANUP_KEEP_STEPS-th last tool-call step and write notes for
    the large results behind it (on the cached prefix of this request, already rewritten as before). Returns a
    log line, or None if moving the boundary would not remove enough to pay for re-reading the rest."""
    messages = request["messages"]
    steps = _steps(messages)
    if len(steps) <= CLEANUP_KEEP_STEPS:
        return None
    old, new = _boundary(messages), steps[-CLEANUP_KEEP_STEPS]
    if new <= old:
        return None
    region = messages[old:new]
    big = _big_results(region)
    gain = (sum(len(m.get("reasoning_content") or "") for m in region if m.get("role") == "assistant")
            + sum(size - 900 for _, _, size in big))  # notes are ~150 words
    if gain < CLEANUP_MIN_GAIN:
        return None
    t0 = time.time()
    with _lock:
        todo = [(m, h, s) for m, h, s in big if h not in _notes and h not in _pending]
        _pending.update(h for _, h, _ in todo)
    timing = {}
    try:
        if todo:
            with _gpu:
                timing = _condense(request, auth, todo, MIDTURN_INSTRUCTION)
    except Exception as e:  # results without notes stay raw; the reasoning is still dropped
        log(f"cleanup notes failed: {type(e).__name__}: {e}")
    finally:
        with _lock:
            for _, h, _ in todo:
                _pending.discard(h)
            _lock.notify_all()
    with _lock:
        _anchors.append(messages[new]["tool_calls"][0]["id"])
        _save_anchors()
    noted = sum(1 for _, h, _ in big if h in _notes)
    return (f"cleanup: boundary moved from message {old} to {new} of {len(messages)}, {noted}/{len(big)} large "
            f"results noted ({len(todo)} new, {time.time() - t0:.1f}s, gen={timing.get('predicted_n')})")


def arm_cleanup(tool_calls, timings):
    """After a tool-call response: if the context ended at >= CLEANUP_AT tokens, the request that answers these
    tool calls gets a cleanup."""
    total = sum(timings.get(k) or 0 for k in ("prompt_n", "cache_n", "predicted_n"))
    if CLEANUP_AT and total >= CLEANUP_AT:
        with _lock:
            for tc in tool_calls:
                _want[tc.get("id")] = total
            while len(_want) > 100:
                del _want[next(iter(_want))]


def _cleanup_due(messages):
    """True if this request answers tool calls whose response ended at >= CLEANUP_AT tokens."""
    with _lock:
        ids = [m.get("tool_call_id") for m in messages[-8:] if m.get("role") == "tool"]
        return bool([_want.pop(i) for i in ids if i in _want])


def _post(path, body, auth, timeout):
    if not _upstream_is_llama():
        raise RuntimeError("llama-server is not running")
    conn = http.client.HTTPConnection(*UPSTREAM, timeout=timeout)
    try:
        conn.request("POST", path, body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                     headers={"Content-Type": "application/json", "Authorization": auth})
        resp = conn.getresponse()
        data = resp.read()
    finally:
        conn.close()
    if resp.status != 200:
        raise RuntimeError(f"{path}: HTTP {resp.status}: {data[:300]!r}")
    return json.loads(data)


def _render(request, messages, auth):
    """The prompt string the chat endpoint would build for `messages` with the request's tools and kwargs."""
    body = {k: v for k, v in request.items() if k not in ("messages", "stream", "stream_options")}
    body["messages"] = messages
    return _post("/apply-template", body, auth, 60)["prompt"]


def _describe(call):
    fn = (call or {}).get("function") or {}
    args = fn.get("arguments") or ""
    args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return f"{fn.get('name') or 'tool'} {args[:200]}"


def _salvage_json(text):
    """Parse the grammar-constrained notes object; if generation hit the token limit, keep what was written."""
    for suffix in ("", '"}', "}"):
        try:
            return json.loads(text + suffix)
        except ValueError:
            continue
    raise ValueError(f"unparseable notes output: {text[:200]!r}")


def _condense(request, auth, todo, instruction=INSTRUCTION):
    """Write notes for `todo` [(message, hash, size)] with one cache-friendly /completion request."""
    calls = {tc.get("id"): tc for m in request["messages"] if m.get("role") == "assistant"
             for tc in (m.get("tool_calls") or [])}
    items = "\n".join(f"r{i}: {_describe(calls.get(m.get('tool_call_id')))} ({size} characters)"
                      for i, (m, _, size) in enumerate(todo, 1))
    prompt = _render(request, request["messages"] + [{"role": "user", "content": instruction.format(items=items)}],
                     auth)
    if prompt.endswith("<think>\n"):
        prompt += "\n</think>\n\n"  # an empty think block, as the template writes for non-thinking turns
    keys = [f"r{i}" for i in range(1, len(todo) + 1)]
    schema = {"type": "object", "properties": {k: {"type": "string"} for k in keys}, "required": keys,
              "additionalProperties": False}
    resp = _post("/completion", {"prompt": prompt, "n_predict": min(16000, 900 * len(todo) + 300),
                                 "temperature": 0.3, "json_schema": schema, "cache_prompt": True}, auth, 900)
    out = _salvage_json(resp["content"])
    with _lock:
        for k, (_, h, _) in zip(keys, todo):
            if (out.get(k) or "").strip():
                _notes[h] = out[k].strip()
        _save_notes()
    return resp.get("timings", {})


def _prewarm(request, answer, reasoning, auth):
    """Prefill the condensed history up to where the next user message starts, so the next turn hits the cache."""
    final = {"role": "assistant", "content": answer}
    # the template keeps past reasoning (preserve_thinking), so mirror the client: OpenCode sends reasoning back,
    # the web UI does not
    if reasoning and any(m.get("role") == "assistant" and m.get("reasoning_content") for m in request["messages"]):
        final["reasoning_content"] = reasoning
    msgs = copy.deepcopy(request["messages"]) + [final, {"role": "user", "content": CUT_MARKER}]
    if not any(condense_history(msgs, wait=False)[:2]):
        return None
    prompt = _render(request, msgs, auth)
    cut = prompt.rfind(CUT_MARKER)
    start = prompt.rfind("<|im_start|>", 0, cut)
    if cut < 0 or start < 0:
        return None
    _debug_dump("prewarm", {"prompt": prompt[:start], "answer": answer})
    resp = _post("/completion", {"prompt": prompt[:start], "n_predict": 0, "cache_prompt": True}, auth, 900)
    return resp.get("timings", {})


def _after_turn(request, answer, reasoning, auth):
    """Background job after a finished turn: notes for this turn's tool results, then prefill the new context."""
    messages = request["messages"]
    # this turn = everything after the previous final answer (not the last user message: OpenCode injects
    # <system-update> user messages between tool calls)
    todo = []
    with _lock:
        for m in messages[_last_final(messages) + 1:]:
            text = _text(m.get("content")) if m.get("role") == "tool" else ""
            h = _h(text)
            if (len(text) >= MIN_CONDENSE and not text.startswith("[Notes on") and h not in _notes
                    and h not in _pending):
                _pending.add(h)
                todo.append((m, h, len(text)))
    with _gpu:
        if todo:
            t0 = time.time()
            try:
                t = _condense(request, auth, todo)
                kept = sum(len(_notes.get(h, "")) for _, h, _ in todo)
                log(f"condensed {len(todo)} tool results {sum(s for _, _, s in todo)} -> {kept} chars in "
                    f"{time.time() - t0:.1f}s (prompt_n={t.get('prompt_n')}, cached={t.get('cache_n')}, "
                    f"gen={t.get('predicted_n')})")
            except Exception as e:  # keep the proxy alive; affected results fall back to truncation
                log(f"condense failed after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")
            finally:
                with _lock:
                    for _, h, _ in todo:
                        _pending.discard(h)
                    _lock.notify_all()
        t0 = time.time()
        try:
            t = _prewarm(request, answer, reasoning, auth)
            if t is not None:
                log(f"prewarmed condensed context in {time.time() - t0:.1f}s (prompt_n={t.get('prompt_n')}, "
                    f"cached={t.get('cache_n')})")
        except Exception as e:
            log(f"prewarm failed after {time.time() - t0:.1f}s: {type(e).__name__}: {e}")


def _parse_response(raw, streamed):
    """(finish_reason, answer content, reasoning, timings) of a chat completion response, streamed or not."""
    try:
        if not streamed:
            d = json.loads(raw)
            ch = d["choices"][0]
            msg = ch.get("message") or {}
            return (ch.get("finish_reason"), msg.get("content") or "", msg.get("reasoning_content") or "",
                    d.get("timings") or {})
        reason, content, reasoning, timings = None, [], [], {}
        for line in raw.decode("utf-8", "replace").splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            timings = d.get("timings") or timings
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                reason = ch.get("finish_reason") or reason
        return reason, "".join(content), "".join(reasoning), timings
    except (ValueError, KeyError, IndexError, TypeError):
        return None, "", "", {}


def _tool_calls(raw, streamed):
    """Tool calls of a chat completion response (streamed deltas are joined), for --capture."""
    try:
        if not streamed:
            return (json.loads(raw)["choices"][0].get("message") or {}).get("tool_calls") or []
        calls = {}
        for line in raw.decode("utf-8", "replace").splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            for ch in d.get("choices") or []:
                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function",
                                                              "function": {"name": "", "arguments": ""}})
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["function"]["name"] += fn.get("name") or ""
                    c["function"]["arguments"] += fn.get("arguments") or ""
        return [calls[i] for i in sorted(calls)]
    except (ValueError, KeyError, IndexError, TypeError):
        return []


_capture_lock = threading.Lock()
_identity = {"ok": False, "checked": 0.0}


def _upstream_is_llama():
    """True if the upstream port is served by llama-server (it sends `Server: llama.cpp`). Checked without the
    client's credentials, so if llama-server is down and another program takes its port (any local web UI on a
    common port such as 8080 can), neither the API key nor any request reaches that program. Cached for 2 seconds."""
    now = time.time()
    if now - _identity["checked"] < 2:
        return _identity["ok"]
    try:
        conn = http.client.HTTPConnection(*UPSTREAM, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        ok = resp.getheader("Server") == "llama.cpp"
    except OSError:
        ok = False
    _identity.update(ok=ok, checked=now)
    return ok


def _capture(req, raw, streamed):
    reason, content, reasoning, t = _parse_response(raw, streamed)
    rec = {"time": time.time(), "request": req, "response": {
        "finish_reason": reason, "content": content, "reasoning_content": reasoning,
        "tool_calls": _tool_calls(raw, streamed), "timings": t}}
    with _capture_lock, open(CAPTURE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        chat_req = None
        if body and self.command == "POST" and self.path.split("?")[0].endswith("/chat/completions"):
            try:
                req = json.loads(body)
                if isinstance(req.get("messages"), list):
                    if CONDENSE and prepare(req, self.headers.get("Authorization", "")):
                        body = json.dumps(req, ensure_ascii=False).encode("utf-8")
                    chat_req = req
                    _debug_dump("request", req)
            except (ValueError, TypeError, AttributeError):
                pass  # not JSON we understand: forward untouched
        if not _upstream_is_llama():
            self.send_error(502, "llama-server is not running")
            return
        conn = http.client.HTTPConnection(*UPSTREAM, timeout=1800)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as e:
            self.send_error(502, f"upstream unavailable: {e}")
            return
        if resp.getheader("Server") != "llama.cpp":  # the port was taken over between the check and now
            conn.close()
            self.send_error(502, "llama-server is not running")
            return
        self.send_response(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in HOP:
                self.send_header(k, v)
        if self.command == "HEAD":
            self.send_header("Content-Length", resp.getheader("Content-Length", "0"))
            self.end_headers()
            conn.close()
            return
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        captured = bytearray() if chat_req is not None and resp.status == 200 else None
        completed = False
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                if captured is not None:
                    captured += chunk
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            completed = True
        except OSError:
            pass  # client went away (e.g. the user pressed stop)
        finally:
            conn.close()
        if completed and captured is not None:
            streamed = bool(chat_req.get("stream"))
            if CAPTURE:
                _capture(chat_req, bytes(captured), streamed)
            if not CONDENSE:
                return
            reason, answer, reasoning, t = _parse_response(bytes(captured), streamed)
            log(f"chat {reason}: prompt_n={t.get('prompt_n')} cached={t.get('cache_n')} gen={t.get('predicted_n')}")
            if reason == "tool_calls":
                arm_cleanup(_tool_calls(bytes(captured), streamed), t)
            if reason == "stop":
                threading.Thread(target=_after_turn, daemon=True,
                                 args=(chat_req, answer, reasoning, self.headers.get("Authorization", ""))).start()

    do_GET = do_POST = do_PUT = do_DELETE = do_OPTIONS = do_HEAD = do_PATCH = _forward

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--capture" in args:
        i = args.index("--capture")
        CAPTURE = args[i + 1]
        del args[i:i + 2]
    if "--no-condense" in args:
        CONDENSE = False
        args.remove("--no-condense")
    if "--cleanup-at" in args:
        i = args.index("--cleanup-at")
        CLEANUP_AT = int(args[i + 1])
        del args[i:i + 2]
    if "--state-dir" in args:
        i = args.index("--state-dir")
        STATE_DIR = args[i + 1]
        del args[i:i + 2]
    LOG, NOTES_FILE, CLEANUP_FILE, DEBUG_FLAG, DEBUG_DIR = _state_paths(STATE_DIR)
    os.makedirs(STATE_DIR, exist_ok=True)
    listen = int(args[0]) if args else 8090
    UPSTREAM = ("127.0.0.1", int(args[1]) if len(args) > 1 else 18080)
    if CONDENSE:
        _load_notes()
        _load_anchors()

    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
                super().handle_error(request, client_address)  # clients dropping keep-alive sockets are normal

    server = Server(("127.0.0.1", listen), Proxy)
    server.daemon_threads = True
    server.serve_forever()
