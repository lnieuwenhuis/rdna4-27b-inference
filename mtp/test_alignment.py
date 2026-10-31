"""Alignment test for the training data path: a synthetic llama-dump-hidden file whose hidden rows carry their own
position, with gaps in the kept positions. Checks that every window's input rows are H[q-1], target rows H[q], input
tokens x_q, and that the loss mask marks exactly the states whose next token x_{q+1} is model output.
run: .venv\\Scripts\\python.exe -I test_alignment.py"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mtp as T  # noqa: E402

E = 4
n = 40
tok = np.arange(1000, 1000 + n, dtype=np.int32)                      # token id = 1000 + position
train = np.zeros(n, np.uint8)
train[10:18] = 1                                                    # two model-output spans
train[30:36] = 1
keep = np.zeros(n, np.uint8)
for i in np.flatnonzero(train):                                     # as the dump tool does with DUMP_AHEAD=4
    keep[max(0, i - 5):i + 1] = 1
rows = np.flatnonzero(keep)
hid = np.zeros((len(rows), E), np.float16)
hid[:, 0] = rows                                                    # each hidden row stores its position

d = tempfile.mkdtemp()
prefix = os.path.join(d, "dump")
with open(prefix + ".bin", "wb") as f:
    f.write(np.array([n, len(rows)], np.int32).tobytes())
    f.write(tok.tobytes())
    f.write(train.tobytes())
    f.write(keep.tobytes())
    f.write(hid.tobytes())
with open(prefix + ".idx.jsonl", "w") as f:
    f.write(json.dumps({"id": "cap-00000", "offset": 0, "n_tokens": n, "n_keep": int(len(rows))}) + "\n")

for win in (3, 5, 64):
    ds = T.Dump(prefix, win, n_embd=E)
    assert ds.windows, win
    marked = set()
    for w in ds.windows:
        t, tr, h, a = ds.get(w)
        b = a + len(t)
        assert np.array_equal(h[:, 0].astype(int), np.arange(a, b)), "hidden rows must be positions a..b-1"
        assert keep[a:b].all(), "a window may only span kept positions"
        h_in, h_tgt, x_in, pos, valid = T.window_layout(t, tr, a)
        q = pos
        assert np.array_equal(h[h_in, 0].astype(int), q - 1)        # input hidden H[q-1]
        assert np.array_equal(h[h_tgt, 0].astype(int), q)           # target from H[q] (predicts x_{q+1})
        assert np.array_equal(x_in, 1000 + q)                       # input token x_q
        for i, qq in enumerate(q):
            expect = qq + 1 < b and train[qq + 1] == 1
            assert valid[i] == expect, (win, qq)
            if valid[i]:
                marked.add(qq + 1)
    # every model-output token that has a kept predecessor pair (q-1, q) is predicted by some window
    targets = {i for i in np.flatnonzero(train) if keep[i - 2] and keep[i - 1]}
    assert targets <= marked, (win, sorted(targets - marked))
print("window alignment and loss mask OK")
