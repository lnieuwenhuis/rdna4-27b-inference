# Starts llama-server with the configuration measured in the README: Swift 1.5 (Qwen3.8-27B) IQ3_S, 262k context,
# KV streaming, MTP + n-gram speculative decoding, one slot. Needs the patched build from build\build.ps1.
# usage: powershell -ExecutionPolicy Bypass -File launch.ps1 -Model <file.gguf> [-Bin <dir with llama-server.exe>]
#        [-Port 18080] [-Ctx 262144] [-KvResidentMiB 1355] [-ApiKeyFile <file>] [-Extra '<more flags>']
param(
    [Parameter(Mandatory)][string]$Model,
    [string]$Bin = (Join-Path $PSScriptRoot '..\build\src\build-rocm\bin'),
    [int]$Port = 18080,
    [int]$Ctx = 262144,
    # resident KV pool on the GPU. 1355 MiB fits next to the IQ3_S weights with a Windows desktop holding ~2.3 GB of
    # VRAM. Lower it if your desktop uses more, and watch "Shared GPU memory" in Task Manager: any growth there
    # means Windows is spilling and decode speed roughly halves.
    [int]$KvResidentMiB = 1355,
    [string]$ApiKeyFile = '',
    [string]$Extra = ''
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path "$Bin\llama-server.exe")) { throw "llama-server.exe not found in $Bin (build it with build\build.ps1, or pass -Bin)" }

# patch features (see patches\COMMITS.md): re-plan KV residency only while the context does not fit the pool, and
# allocate the pool in <= 1 GiB chunks (one large allocation spills into shared memory on Windows)
$env:LLAMA_KV_STREAM_RERESIDE = '1'
$env:LLAMA_KV_STREAM_POOL_CHUNK_MIB = '1024'

$a = @(
    '-m', $Model, '--host', '127.0.0.1', '--port', $Port,
    # one slot: two slots with KV streaming hard-reset the test machine
    '-c', $Ctx, '-np', 1, '-ngl', 999, '-fa', 'on', '--jinja', '-fit', 'off',
    # embeddings are a pure gather: keep them in system RAM and save VRAM
    '-ot', 'token_embd.weight=CPU',
    # read the weights once instead of memory-mapping them: ~10 GB less resident RAM
    '--load-mode', 'none',
    '-ctk', 'q8_0', '-ctv', 'q4_0', '-ctkd', 'q8_0', '-ctvd', 'q4_0',
    '-b', 2048, '-ub', 512,
    # host-RAM cache of recent prompt states: switching between conversations restores them instead of re-reading
    '--cache-ram', 4096,
    # MTP drafts 4 tokens; n-gram drafts start after 6 repeated tokens with a 12-token match, capped at 16 tokens
    '--spec-type', 'draft-mtp,ngram-mod', '--spec-rs-replay', '--spec-draft-n-max', 4,
    '--spec-ngram-mod-n-max', 16, '--spec-ngram-mod-n-min', 6, '--spec-ngram-mod-n-match', 12,
    '--spec-draft-sampling', 'probabilistic',
    # Qwen's recommended thinking-mode sampling (clients can override per request)
    '--temp', '1.0', '--top-p', '0.95', '--top-k', 20, '--min-p', '0',
    '--kv-stream', '--kv-resident-mib', $KvResidentMiB, '--kv-stream-split'
)
if ($ApiKeyFile) { $a += @('--api-key-file', $ApiKeyFile) }
$a += @($Extra -split '\s+' | Where-Object { $_ })

& "$Bin\llama-server.exe" @a
