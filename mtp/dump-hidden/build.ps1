# Builds llama-dump-hidden against an existing build of the fork (<Src>\build-rocm, from build\build.ps1) without
# touching it: compiles this one file with the same compiler/flags and links the build's import libraries. The exe is
# written next to the DLLs it needs (build-rocm\bin), as a new file only.
# usage: powershell -ExecutionPolicy Bypass -File build.ps1 [-Src <llama.cpp checkout>] [-Venv <ROCm SDK venv>]
#   -Src   default: $env:LLAMA_CPP_DIR, else build\src in the repository root (build\build.ps1's default)
#   -Venv  default: $env:ROCM_VENV, else build\.venv-rocm in the repository root
param(
    [string]$Src = $(if ($env:LLAMA_CPP_DIR) { $env:LLAMA_CPP_DIR } else { Join-Path $PSScriptRoot '..\..\build\src' }),
    [string]$Venv = $(if ($env:ROCM_VENV) { $env:ROCM_VENV } else { Join-Path $PSScriptRoot '..\..\build\.venv-rocm' })
)
$ErrorActionPreference = 'Stop'
$B = "$Src\build-rocm"
$SP = "$Venv\Lib\site-packages"
$vsw = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vsw -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] }
}
$clang = "$SP\_rocm_sdk_devel\lib\llvm\bin\clang++.exe"
$obj = "$PSScriptRoot\dump-hidden.obj"
# -I$Src\src: llama-ext.h (staging API with llama_set/get_embeddings_layer_inp, exported by llama.dll)
& $clang -O3 -DNDEBUG -std=gnu++17 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt `
    -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP -DLLAMA_SHARED -D_CRT_SECURE_NO_WARNINGS `
    "-I$Src\common" "-I$Src\src" "-I$Src\vendor" "-I$Src\include" "-I$Src\ggml\include" `
    -c "$PSScriptRoot\dump-hidden.cpp" -o $obj
if ($LASTEXITCODE) { throw 'compile failed' }
& $clang -fuse-ld=lld -Xlinker /subsystem:console $obj -o "$B\bin\llama-dump-hidden.exe" `
    "$B\common\llama-common.lib" "$B\common\llama-common-base.lib" "$B\src\llama.lib" "$B\ggml\src\ggml.lib" `
    "$B\ggml\src\ggml-base.lib" -lkernel32 -luser32 -ladvapi32 -lshell32
if ($LASTEXITCODE) { throw 'link failed' }
Write-Output "built $B\bin\llama-dump-hidden.exe"
