# Hidden-state dump for MTP training. The server must be stopped first (the dump needs the GPU's VRAM); the context
# proxy can keep running: it answers 502 while llama-server is down.
# usage: powershell -ExecutionPolicy Bypass -File run_dump.ps1 -In <corpus.jsonl> -Out <prefix> [-Model <file.gguf>]
#            [-Ctx 32768] [-LlamaDir <llama.cpp checkout>] [-Venv <ROCm SDK venv>]
#   -Model     the served GGUF with the MTP head (default: $env:MTP_GGUF; required one way or the other)
#   -LlamaDir  the fork's checkout with build-rocm\bin\llama-dump-hidden.exe (dump-hidden\build.ps1)
#              (default: $env:LLAMA_CPP_DIR, else build\src in the repository root, as build\build.ps1 lays it out)
#   -Venv      the ROCm SDK venv build\build.ps1 created (default: $env:ROCM_VENV, else build\.venv-rocm)
param([Parameter(Mandatory)][string]$In, [Parameter(Mandatory)][string]$Out, [int]$Ctx = 32768, [int]$Ahead = 2048,
    [string]$Model = $env:MTP_GGUF,
    [string]$LlamaDir = $(if ($env:LLAMA_CPP_DIR) { $env:LLAMA_CPP_DIR } else { Join-Path $PSScriptRoot '..\build\src' }),
    [string]$Venv = $(if ($env:ROCM_VENV) { $env:ROCM_VENV } else { Join-Path $PSScriptRoot '..\build\.venv-rocm' }))
$ErrorActionPreference = 'Stop'
if (-not $Model) { Write-Error 'no model: pass -Model <file.gguf> or set MTP_GGUF' }
# the tool runs in $Bin, so relative paths are made absolute here
$Model = (Resolve-Path $Model).Path
$In = (Resolve-Path $In).Path
$Out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
if (Get-Process llama-server -ErrorAction SilentlyContinue) { Write-Error 'llama-server is running: stop it first' }
$SP = "$Venv\Lib\site-packages"
$Bin = "$LlamaDir\build-rocm\bin"
$env:PATH = "$SP\_rocm_sdk_core\bin;$SP\_rocm_sdk_devel\bin;$SP\_rocm_sdk_libraries\bin;$SP\_rocm_sdk_devel\lib\llvm\bin;$env:PATH"
$env:HIP_PATH = "$SP\_rocm_sdk_devel"
$env:DUMP_IN = $In; $env:DUMP_OUT = $Out; $env:DUMP_AHEAD = "$Ahead"
foreach ($f in "$Out.bin", "$Out.idx.jsonl") { if (Test-Path $f) { Write-Error "$f exists (the tool appends): pick a new prefix" } }
# Same memory layout as the server (KV streaming with a 600 MiB GPU pool, embeddings on the CPU): without streaming,
# 48k of KV on top of the weights would spill into shared memory (on a 16 GB card the desktop compositor, dwm,
# can hold ~2.3 GB)
# Run it as a process with its output in log files: piping a native program's stderr in Windows PowerShell 5.1 turns
# the first stderr line (a harmless warning) into a terminating error under ErrorActionPreference Stop.
$args_ = @('-m', "`"$Model`"", '-ngl', '999', '-fa', 'on',
    '-c', "$Ctx", '-b', '2048', '-ub', '512', '-ctk', 'q8_0', '-ctv', 'q4_0', '-ot', 'token_embd.weight=CPU',
    '--load-mode', 'none', '-dev', 'ROCm0', '-fit', 'off', '--kv-stream', '--kv-resident-mib', '600')
$p = Start-Process -FilePath "$Bin\llama-dump-hidden.exe" -ArgumentList $args_ -WorkingDirectory $Bin -NoNewWindow -PassThru `
    -RedirectStandardError "$Out.log" -RedirectStandardOutput "$Out.out.log"
$p.WaitForExit()
Select-String -Path "$Out.log" -Pattern 'sequences|done:|error|failed|out of memory|ROCm' | Select-Object -Last 5 | ForEach-Object { $_.Line }
exit $p.ExitCode
