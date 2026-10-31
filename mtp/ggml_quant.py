"""Quantize float32 rows with the fork's own ggml (ggml_quantize_chunk in ggml-base.dll), so the bytes are exactly
what llama.cpp's quantizer would write. The DLL is loaded from <llama.cpp checkout>/build-rocm/bin, where the
checkout is $LLAMA_CPP_DIR, else build/src in the repository root (build/build.ps1's default)."""
import ctypes
import os

import numpy as np

_SRC = os.environ.get("LLAMA_CPP_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "src")
BIN = os.path.join(_SRC, "build-rocm", "bin")
if not os.path.isfile(os.path.join(BIN, "ggml-base.dll")):
    raise ImportError(f"ggml-base.dll not found in {BIN}: build the fork (build\\build.ps1) or set LLAMA_CPP_DIR")
os.add_dll_directory(BIN)
_ggml = ctypes.CDLL(os.path.join(BIN, "ggml-base.dll"))
_ggml.ggml_quantize_chunk.restype = ctypes.c_size_t
_ggml.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                      ctypes.c_int64, ctypes.c_void_p]
_ggml.ggml_row_size.restype = ctypes.c_size_t
_ggml.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
_ggml.ggml_quantize_init.argtypes = [ctypes.c_int]

GGML_TYPE_Q6_K = 14


def quantize(rows: np.ndarray, ggml_type: int = GGML_TYPE_Q6_K) -> np.ndarray:
    """rows: float32 [n_rows, n_per_row] -> uint8 [n_rows, row_bytes] in ggml's block format."""
    src = np.ascontiguousarray(rows, dtype=np.float32)
    n_rows, n_per_row = src.shape
    row_bytes = _ggml.ggml_row_size(ggml_type, n_per_row)
    dst = np.empty((n_rows, row_bytes), dtype=np.uint8)
    _ggml.ggml_quantize_init(ggml_type)
    written = _ggml.ggml_quantize_chunk(ggml_type, src.ctypes.data, dst.ctypes.data, 0, n_rows, n_per_row, None)
    assert written == dst.nbytes, (written, dst.nbytes)
    return dst
