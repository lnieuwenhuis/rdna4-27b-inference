"""Rebuild the offsets in a llama-dump-hidden index. The tool wrote ftell() offsets, a 32-bit long on Windows, so every
record past 2 GB had a wrapped offset. The .bin itself is sequential and intact: walk it record by record (layout:
int32 n, int32 n_keep, int32 tokens[n], uint8 train[n], uint8 keep[n], fp16 hidden[n_keep][n_embd]) and give every
index line, in order, the true offset, checking that n_tokens and n_keep match. The old index is kept as .bak.
usage: python -I fix_dump_index.py <prefix> [n_embd=5120]"""
import json
import os
import shutil
import sys

import numpy as np

prefix = sys.argv[1]
n_embd = int(sys.argv[2]) if len(sys.argv) > 2 else 5120
binp, idxp = prefix + ".bin", prefix + ".idx.jsonl"
size = os.path.getsize(binp)
lines = [json.loads(l) for l in open(idxp, encoding="utf-8")]
off, fixed = 0, 0
with open(binp, "rb") as f:
    for i, ix in enumerate(lines):
        f.seek(off)
        n, n_keep = np.frombuffer(f.read(8), dtype=np.int32)
        if n != ix["n_tokens"] or n_keep != ix["n_keep"]:
            raise SystemExit(f"record {i} ({ix['id']}) at {off}: bin says n={n} keep={n_keep}, index says "
                             f"{ix['n_tokens']}/{ix['n_keep']}: not the same order, giving up")
        if ix["offset"] != off:
            fixed += 1
        ix["offset"] = int(off)
        off += 8 + 4 * int(n) + 2 * int(n) + 2 * int(n_keep) * n_embd
if off > size:
    raise SystemExit(f"index runs past the end of the bin ({off} > {size})")
shutil.copyfile(idxp, idxp + ".bak")
with open(idxp, "w", encoding="utf-8") as f:
    for ix in lines:
        f.write(json.dumps(ix) + "\n")
print(f"{len(lines)} records verified, {fixed} offsets corrected; records end at {off} of {size} bytes "
      f"({size - off} trailing bytes)")
