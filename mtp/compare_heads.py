"""Paired comparison of MTP heads on the untouched test split (run.json from train_mtp.py): for each pair, the
per-session difference in expected accepted draft length (same sessions for both heads), its mean with a bootstrap
95% interval, wins/losses, and the aggregate relative change with its interval (sessions resampled).
usage: python -I compare_heads.py <run.json>   |   python -I compare_heads.py --selftest"""
import json
import random
import sys


def paired(a, b, n_boot=10000, seed=0):
    """a, b: {session: {"positions", "expected_accepted"}} for the old and new head."""
    ids = sorted(set(a) & set(b))
    d = [b[i]["expected_accepted"] - a[i]["expected_accepted"] for i in ids]
    w = [a[i]["positions"] for i in ids]

    def rel(sample):           # position-weighted aggregate: sum(w*new)/sum(w*old) - 1
        num = sum(w[j] * b[ids[j]]["expected_accepted"] for j in sample)
        den = sum(w[j] * a[ids[j]]["expected_accepted"] for j in sample)
        return num / den - 1 if den else 0.0

    rng = random.Random(seed)
    n = len(ids)
    means, rels = [], []
    for _ in range(n_boot):
        s = [rng.randrange(n) for _ in range(n)]
        means.append(sum(d[j] for j in s) / n)
        rels.append(rel(s))
    means.sort()
    rels.sort()
    ci = lambda xs: (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1])
    return {"sessions": n, "mean_diff": sum(d) / n if n else 0, "mean_diff_ci": ci(means),
            "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d),
            "relative": rel(range(n)), "relative_ci": ci(rels)}


def main(path):
    run = json.load(open(path, encoding="utf-8"))
    heads = {"original": run["original"], "mixed": run.get("mixed_phase_end"), "final": run.get("final")}
    heads = {k: v["test"]["sessions"] for k, v in heads.items() if v and "test" in v and "sessions" in v["test"]}
    for old, new in (("original", "mixed"), ("original", "final"), ("mixed", "final")):
        if old in heads and new in heads:
            r = paired(heads[old], heads[new])
            lo, hi = r["mean_diff_ci"]
            rl, rh = r["relative_ci"]
            print(f"{new:>5} vs {old:<8}: {r['sessions']} sessions, {r['wins']} better / {r['losses']} worse, "
                  f"accepted length {r['mean_diff']:+.3f} per step [95% CI {lo:+.3f}, {hi:+.3f}], "
                  f"relative {100 * r['relative']:+.1f}% [{100 * rl:+.1f}%, {100 * rh:+.1f}%]")


def selftest():
    rng = random.Random(1)
    old = {f"s{i}": {"positions": rng.randint(200, 2000), "expected_accepted": rng.uniform(1.5, 3.0)} for i in range(25)}
    same = {k: dict(v) for k, v in old.items()}
    better = {k: {**v, "expected_accepted": v["expected_accepted"] * 1.05 + rng.gauss(0, 0.01)} for k, v in old.items()}
    r0, r1 = paired(old, same), paired(old, better)
    assert r0["mean_diff"] == 0 and r0["wins"] == r0["losses"] == 0
    assert r1["mean_diff_ci"][0] > 0 and 0.03 < r1["relative"] < 0.07, r1
    print("compare_heads self-test OK")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--selftest"] else main(sys.argv[1])
