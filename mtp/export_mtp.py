"""Write a retrained MTP head into a NEW GGUF: a byte copy of the served file with only the blk.64 head tensors
overwritten in place (same types and sizes: Q6_K via the fork's own quantizer, F32 norms). The original file is
only read. usage: python export_mtp.py <head_merged.pt> <out.gguf> [src.gguf]
src.gguf (the served model with the MTP head) defaults to $MTP_GGUF. Quantization goes through the fork's built
ggml-base.dll (ggml_quant.py, found via $LLAMA_CPP_DIR); gguf-py is taken from $LLAMA_CPP_DIR too when set."""
import hashlib
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if os.environ.get("LLAMA_CPP_DIR"):  # else the pip `gguf` package
    sys.path.insert(0, os.path.join(os.environ["LLAMA_CPP_DIR"], "gguf-py"))
import ggml_quant  # noqa: E402
from gguf import GGMLQuantizationType as T, GGUFReader, quants  # noqa: E402


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    head_pt, out = sys.argv[1], sys.argv[2]
    src = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("MTP_GGUF")
    if not src:
        sys.exit("no source GGUF: pass src.gguf or set MTP_GGUF")
    assert os.path.abspath(out) != os.path.abspath(src), "refusing to overwrite the source model"
    sd = torch.load(head_pt)
    r = GGUFReader(src)
    patches = []
    for t in r.tensors:
        if t.name not in sd:
            continue
        w = sd[t.name].detach().float().numpy()
        if t.tensor_type == T.F32:
            data = np.ascontiguousarray(w.astype(np.float32).reshape(-1)).view(np.uint8)
        elif t.tensor_type == T.Q6_K:
            data = ggml_quant.quantize(w.reshape(-1, w.shape[-1]), ggml_quant.GGML_TYPE_Q6_K).reshape(-1)
            err = np.abs(quants.dequantize(data, T.Q6_K).reshape(w.shape) - w).mean() / (np.abs(w).mean() + 1e-12)
            print(f"{t.name}: Q6_K, mean relative quantization error {err:.4f}")
        else:
            raise SystemExit(f"{t.name}: unsupported type {t.tensor_type.name}")
        assert data.nbytes == t.n_bytes, (t.name, data.nbytes, t.n_bytes)
        patches.append((int(t.data_offset), data, t.name))
    missing = set(sd) - {n for _, _, n in patches}
    assert not missing, missing
    del r
    shutil.copyfile(src, out)
    with open(out, "r+b") as f:
        for off, data, _ in patches:
            f.seek(off)
            f.write(data.tobytes())
    print(f"wrote {out}: {len(patches)} head tensors replaced")
    # sanity: re-read the copy and compare one tensor
    r2 = GGUFReader(out)
    t = next(x for x in r2.tensors if x.name == patches[0][2])
    assert np.array_equal(np.asarray(t.data).reshape(-1).view(np.uint8)[:64], patches[0][1][:64])
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    print("sha256", h.hexdigest())


if __name__ == "__main__":
    main()
