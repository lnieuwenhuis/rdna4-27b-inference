# A 27B model at 262k context on one 16 GB AMD card

This is the setup that runs **Swift 1.5**, UkisAI's reasoning-efficient fine-tune of **Qwen3.8-27B**, on an
**RX 9070 XT (RDNA 4, 16 GB) under Windows 11**. It serves the model's full 262,144-token context and decodes
coding-agent turns at about **100–115 tok/s**. On a chat benchmark it runs 66–70 tok/s; stock llama.cpp ran base
Qwen3.8-27B (IQ4_XS) on the same card at about 40 tok/s with ROCm and 21–22 tok/s with Vulkan.

Model files: **[Safasfly/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-P2-MTP1-GGUF](https://huggingface.co/Safasfly/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-P2-MTP1-GGUF)**
(IQ3_S, Latin-pruned vocabulary, self-distilled MTP draft head).

| | |
| --- | --- |
| GPU | RX 9070 XT 16 GB (gfx1201), also driving a 1440p desktop (~2.3 GB of VRAM taken by Windows) |
| CPU, RAM | Ryzen 7 7800X3D, 32 GB DDR5 |
| OS | Windows 11, ROCm 10.1 SDK from AMD's pip wheels (no system ROCm install) |
| Runtime | [achyudh/llama.cpp](https://github.com/achyudh/llama.cpp) `feature/adaptive-kv-stream-rocm` @ `b540a4b` + [`patches/kvwin-rdna4-windows.patch`](patches/) |

## Results

All runs were single-user and used the final build and flags from [`server/launch.ps1`](server/launch.ps1).

| Workload | Decode |
| --- | --- |
| Agent replay: 5 captured OpenCode coding-agent requests (`bench/replay.py`) | ~102 tok/s in the deploy check. A 90-minute soak ranged 100–145 as the prompt cache warmed. |
| Chat: 512-token answers, temperature 1.0 (`bench/bench.py`) | 66–70 tok/s |
| Long context, planted-fact retrieval at 64k / 96k (`bench/long_test.py`) | 69–73 / 68 tok/s, all answers correct. The soak's 96k runs ranged 44–92. |
| Prefill | ~1,300 tok/s at short depth, ~900 at 180k |

Speculative decoding makes speed depend on the content. Code and repetitive agent turns run fastest; free-form prose
at temperature 1.0 runs around 60 tok/s.

Stock references, measured earlier on the same card with base Qwen3.8-27B IQ4_XS on the chat benchmark:

| Runtime | Decode |
| --- | --- |
| llama.cpp ROCm b11936 with MTP | ~40 tok/s |
| llama.cpp ROCm without speculative decoding | 36 tok/s |
| llama.cpp Vulkan | 21–22 tok/s |

## Setup

### 1. Model: Swift 1.5 GSQ-RCO IQ3_S

Of the models that fit a 16 GB card, Qwen3.8-27B scores highest on the Artificial Analysis Intelligence Index: 34,
against 18 for Qwen3.6-35B-A3B. The MoE models that reach 80+ tok/s easily score about half as high.

Swift 1.5 writes far fewer thinking tokens for the same problems. Replaying a real chat, it used 2,015 → 939 thinking tokens on the
hard step (−53%) and 1,079 → 656 on the follow-up (−39%), at the same decode speed and with the same answers.
UkisAI's GSQ-RCO IQ3_S is 11.2 GiB; IQ4_XS doesn't leave room for the KV pool next to a Windows desktop.

### 2. Vocabulary pruning, ASCII + Latin ([`prune/`](prune/))

The Qwen vocabulary has 248k tokens, most of them CJK, Cyrillic and other scripts. Their rows in `token_embd` and
`output` are a large part of a 3-bit file. Policy **P2** keeps ASCII, Latin-1 and Latin Extended-A, and drops
~107k tokens. The file shrinks by 0.49 GB and decodes ~5% faster, because the output projection gets smaller.

Every other tensor stays byte-identical, which `verify_prune.py` checks. The keep-set is closed under substring, so
BPE merges stay reachable, and dropped characters still work through byte fallback at one token per UTF-8 byte.
Perplexity on an emoji-heavy corpus rose from 3.375 to 3.42.

### 3. KV streaming: 262k context on 16 GB

achyudh's fork keeps most of the KV cache in pinned system RAM (~6.8 GB at 262k) and streams it to the GPU each
step, so only a small **resident pool** lives in VRAM. raffidahmad's
[Windows recipe](https://github.com/raffidahmad/llama-cpp-kvstream-windows) builds it against AMD's pip ROCm SDK;
[`build/build.ps1`](build/build.ps1) is that recipe pointed at this repo's patch.

VRAM layout on a 16 GB card that is also running the desktop:

- Embeddings stay on the CPU (`-ot token_embd.weight=CPU`). An embedding lookup is a plain gather, so this costs no
  speed.
- `--load-mode none` reads the weights once instead of memory-mapping them, which keeps ~10 GB out of resident RAM.
- KV cache types are K `q8_0` and V `q4_0`, for the main model and the draft.
- One resident pool of 1355 MiB, with no separate decode pool. In an early version, the recipe's 1800 MiB decode
  pool spilled into shared memory and every prefill/decode switch cost 0.4–0.75 s.
- One slot: two slots with KV streaming hard-reset the machine.

### 4. Speculative decoding: MTP + n-gram

Qwen3.8 ships one multi-token-prediction (MTP) layer. These settings were tuned by replaying captured agent turns:

```
--spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-rs-replay --spec-draft-sampling probabilistic
--spec-ngram-mod-n-min 6 --spec-ngram-mod-n-match 12 --spec-ngram-mod-n-max 16
```

- **MTP draft depth 4, plus n-gram drafts** that start after 6 repeated tokens with a 12-token match: agent decode
  went 85.6 → 97.8 tok/s (+14%), with chat unchanged. Agent turns repeat a lot of text (file contents, diffs,
  tool arguments), and n-gram drafts hit on those repeats.
- **Capping n-gram drafts at 16 tokens** instead of 32 gave another ~+5%. Long drafts were usually rejected partway.

### 5. A self-distilled MTP head ([`mtp/`](mtp/))

The stock MTP head was trained against the full-precision base model, not against a 3-bit quant of a fine-tune. So
it was retrained against this exact model:

1. **Generate** ~1.06M tokens with the deployed model: coding-agent sessions through OpenCode, plus chat answers.
2. **Mix in** 1,600 rows of [faunix/Qwen3.8-27B-Distillation-40K](https://huggingface.co/datasets/faunix/Qwen3.8-27B-Distillation-40K)
   and 300 rows of [ukisai/Qwen3.8-27B-multi-turn-agent-sft](https://huggingface.co/datasets/ukisai/Qwen3.8-27B-multi-turn-agent-sft).
   Own generations are weighted ×2. Anything the A/B evaluates is removed (`prepare_mtp_data.py`).
3. **Dump the target hidden states** from the quantized model with `llama-dump-hidden`
   ([`mtp/dump-hidden/`](mtp/dump-hidden/)), at about 1,300 tok/s.
4. **Train only `blk.64`** in PyTorch ROCm on the same GPU (`train_mtp.py`). The last 25% of training windows are
   on-policy only. Checkpoints are selected on expected accepted draft length for held-out own sessions.
5. **Write the head back** into the GGUF (`export_mtp.py`), then A/B it on the real server.

| Workload | Original head | Retrained head |
| --- | --- | --- |
| Agent replay, warm / cold | 102.2 / 88.0 tok/s | 115.8 / 93.7 tok/s (+13% / +6.5%) |
| Chat | 58.7 tok/s | 66.0 tok/s (+12%) |
| ~75k context | 75.8 tok/s | 85.5 tok/s (+13%) |
| 150k+ context | 34.8 tok/s | 33.3 tok/s (−4%, within restart noise) |

Offline, on held-out sessions, expected accepted length rose +7.9% (95% CI +6.7 to +11.4%), with 26 of 28 sessions
better. A second round with ~2M more tokens didn't beat it, so the head looks saturated at this size.

### 6. Fork patches for Windows ([`patches/`](patches/))

One diff against `b540a4b`, made of 44 commits plus merges (listed in [`patches/COMMITS.md`](patches/COMMITS.md)). The parts
that are on in the measured setup:

- **Copy-deadlock fix.** On Linux the fork streams KV over a dedicated SDMA copy engine. On Windows that engine is
  unavailable, so the copy runs as an ordinary HIP stream. H2D copies could then queue behind a compute-stream copy
  while the paged-decode kernel spun waiting for them. The result was a hard hang at ~143k context with n-gram
  drafting. The fix waits for streamed K/V on the host (one host-wait policy, a host copy of the attention sync tag).
  Long-context decode also went up 4–9%.
- **Fit-gated residency re-plan** (`LLAMA_KV_STREAM_RERESIDE=1`): when the context shrinks, the resident pages are
  re-planned, but only while the context doesn't fit the pool. Long-context decode improved 5–6% per step.
  Re-planning every time (mode 2) cost 6–7% on short agent turns.
- **Split arena and pool in ≤1 GiB chunks** (`--kv-stream-split`, `LLAMA_KV_STREAM_POOL_CHUNK_MIB=1024`). On this
  driver, Windows placed any single allocation above ~1.5 GiB in shared memory, even with VRAM free. Chunking the pool made a larger split budget possible:
  100k-context decode went up 21–54%.

Every change passed the same gates before going live: greedy output byte-identical to the previous build, no
workload regressing beyond noise, and a 90-minute soak (23 iterations of mixed replay, chat and 32k/96k retrieval)
with zero stalls.

Also in the patch, but **off by default**:

- an adaptive pool sized to the live VRAM budget;
- DFlash2 drafter tuning;
- `--spec-switch` to choose between MTP and DFlash drafts per step;
- a DFlash first-draft warm-up.

### 7. Context-condensing proxy ([`proxy/`](proxy/))

`context_proxy.py` sits between clients (a web chat, OpenCode) and llama-server. When a turn ends, it asks the model
in the background to write notes on that turn's large tool results, as grammar-constrained JSON with one field per
result. Later requests carry the notes instead of the raw results; the current turn's tool loop is never touched.
The proxy then prefills the condensed history, so the next message only has to prefill its own tokens.

The notes request is rendered with `/apply-template` from the exact request just served and sent to `/completion`,
so its prefix is byte-identical and is served from the prompt cache. Thinking is skipped by closing the think block
in the prompt. Setting `enable_thinking: false` would change the system prompt and miss the cache.

Long single agent turns get a one-time mid-turn cleanup instead. Past ~110k tokens it drops old reasoning and puts
notes in place of big tool results behind the last 8 tool-call steps, because every decode step slows down at that
depth with KV streaming.

Measured on a real 3-question research chat:

- The third question's context went from 49k → 8.3k tokens, and only 23 tokens needed prefilling.
- 75k characters of web pages became 3.5k characters of notes in 19 s.

## Failed experiments

| Tried | Result |
| --- | --- |
| Vulkan backend | ~40% slower than ROCm on RDNA 4 |
| Two slots with KV streaming | Hard reset of the PC |
| DFlash2 drafter, off the shelf | 8–23% slower than the retrained MTP head on agent and chat; better only on repetitive long context |
| Separate 1800 MiB decode pool | Spilled into shared memory, with a 0.4–0.75 s stall per prefill/decode switch |
| Split arena as one allocation | Spilled (that's why the pool is chunked) |
| Residency re-plan on every shrink (mode 2) | −6–7% on short agent turns |
| MTP draft window limited to 32k | Long-context regression |
| HIP graphs off | Chat 2–6% faster, short agent turns 3–9% slower: mixed |
| Deeper MTP drafts, higher-precision draft embeddings, a cheaper MTP block or draft LM head | No gain |
| Thread count, polling, process priority, HIP environment variables | No gain |
| IQ3_XXS | ~1% faster than IQ3_S, +1.1% perplexity, and it exposed the ~143k n-gram hang that the patch fixes |

## Windows pitfalls

- **Shared-memory spill.** Windows spills to shared memory silently, and decode speed roughly halves. Watch
  "Shared GPU memory" in Task Manager. The desktop compositor alone holds ~2.3 GB of VRAM.
- **Load the right HIP DLLs.** Copy the SDK's `amdhip64_*.dll`, `amd_comgr*.dll` and `hiprtc*.dll` next to
  `llama-server.exe`. Otherwise the older driver copy in System32 loads and fails with "device kernel image is
  invalid". The build script does this.
- **Don't build with `-march=native`.** A native build crashes with `0xC000001D` on other CPUs; build for an AVX2
  baseline.
- **PyTorch ROCm needs the GPU-specific package.** Its gfx1201 kernels live in a separate package
  (`amd-torch-device-gfx1201`); without it you get `hipErrorInvalidKernelFile`.
- **32-bit file offsets.** `ftell()` returns a 32-bit `long` on Windows, so dump files past 2 GB need 64-bit offsets
  (`mtp/fix_dump_index.py` repairs old indexes).
- **Check what's on your port.** Pick a port no other app grabs. The proxy refuses an upstream that doesn't answer
  `Server: llama.cpp`, so a crash can't leave another program exposed behind it.

## Build and run

Requirements: Visual Studio 2022 Build Tools ("Desktop development with C++"), Git, and
[uv](https://github.com/astral-sh/uv).

```powershell
powershell -ExecutionPolicy Bypass -File build\build.ps1          # ROCm SDK venv + fork @ b540a4b + patch + build
hf download Safasfly/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-P2-MTP1-GGUF --local-dir models
powershell -ExecutionPolicy Bypass -File server\launch.ps1 -Model models\Swift-1.5-P2-GSQ-RCO-IQ3_S-mtp1.gguf
```

llama-server then serves the OpenAI API on `http://127.0.0.1:18080/v1`. Reasoning effort is set per request:
`"chat_template_kwargs": {"reasoning_effort": "low" | "medium" | "xhigh"}`, with `xhigh` as the default.

Condensing proxy: `python proxy/context_proxy.py 8090 18080`, then point clients at `http://127.0.0.1:8090/v1`. Add
`--capture work/captures.jsonl` to record requests for `bench/replay.py --captures work/captures.jsonl`.

Other GPUs: `build.ps1 -Gpu gfx1200` builds for other RDNA 4 cards. On a card with less free VRAM, lower
`-KvResidentMiB` until shared memory stops growing.

## Layout

| Path | Contents |
| --- | --- |
| `patches/` | The diff against achyudh's fork and the commit list |
| `build/` | Windows build script (ROCm SDK venv, pinned source, patch, HIP DLLs) |
| `server/` | `launch.ps1` with the measured flags |
| `proxy/` | The context-condensing proxy and its tests |
| `prune/` | Vocabulary audit, prune and verify tools |
| `mtp/` | MTP head data prep, hidden-state dump tool, training, export and offline comparison |
| `bench/` | Agent replay, chat bench, long-context retrieval, speculative-decoding acceptance stats |

## Credits

- [UkisAI](https://huggingface.co/ukisai) for Swift 1.5 and the GSQ-RCO quants, and
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) for the GSQ-RCO allocation.
- [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B.
- [achyudh](https://github.com/achyudh/llama.cpp) for the RDNA4 ROCm KV-streaming fork, which builds on
  [RaymondHuang210129's adaptive KV streaming](https://github.com/RaymondHuang210129/llama.cpp-adaptive-kv-streaming)
  and [stew675's RDNA boosts](https://github.com/stew675/llama-cpp-rdna-boosts).
- [raffidahmad](https://github.com/raffidahmad/llama-cpp-kvstream-windows) for the Windows build recipe.
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

The code here is MIT-licensed (see `LICENSE`); the patch follows llama.cpp's MIT licence. The model weights are
under the Swift Open License v1.0; see the Hugging Face repo.
