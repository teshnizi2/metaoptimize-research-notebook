#!/usr/bin/env python3
"""c244_cvt5_k13_readouts.py <runsdir>   -- CORRECTIONS 244 (b) and (c).  DESCRIPTIVE ONLY; no bar, gate or token reads it.

Reads only <runsdir>/cvt5/probe_cvt5-K13-s{93,94,95}/probe.jsonl (json module, stdlib; imports nothing from the repo).
One record every 100 steps; 500 steps per epoch (50000 / batch 100), so a record's epoch is step / 500.
Group 0 is the complement (every tensor but layer4.1.bn2.weight), group 1 is tensor 50.

(b) DOWN fraction of K13's complement.  A record is DOWN iff  c = pt_b2 * mom_pre[0][0] + (1 - pt_b2) * z_agg[0][0] > 0,
    i.e. the Lion direction the harness applies moves the complement's beta DOWN (step sign == -sign(c); the sign
    convention re-checked below against the applied step on every record where beta moved and did not touch a clamp).
    Windows are [lo, lo+50) epochs by record epoch, 250 records each; per seed, then the three seeds pooled.
(c) r = exp(beta[0] + 15) of K13's complement after its first record with r <= 2, s94: the largest value and EVERY
    record carrying it (exact float equality of beta[0]), plus every record whose r prints as the same 4-sig-fig value.
"""
import json
import math
import os
import sys

SEEDS = (93, 94, 95)
SPE = 500
LO, HI = -15.0, math.log(0.1)


def load(runs, s):
    with open(os.path.join(runs, "cvt5", "probe_cvt5-K13-s%d" % s, "probe.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    runs = sys.argv[1]
    recs = {s: load(runs, s) for s in SEEDS}
    for s in SEEDS:
        assert len(recs[s]) == 1500 and [r["step"] for r in recs[s]] == list(range(0, 150000, 100)), s

    print("(b) K13 complement DOWN fraction, DOWN iff pt_b2*mom_pre[0][0] + (1-pt_b2)*z_agg[0][0] > 0, windows [lo,lo+50) epochs")
    agree = disagree = 0
    for s in SEEDS:
        R = recs[s]
        for b in R:
            c = b["pt_b2"] * b["mom_pre"][0][0] + (1 - b["pt_b2"]) * b["z_agg"][0][0]
            pre = b["beta_pre"][0][0]
            moved = b["beta"][0] - pre
            # meta step size 1e-3: keep records whose pre-update beta is > 2 steps from either clamp and that moved
            if abs(moved) > 5e-4 and c != 0 and LO + 2e-3 < pre < HI - 2e-3:
                if (moved < 0) == (c > 0):
                    agree += 1
                else:
                    disagree += 1
    print("    sign convention check (records where beta moved, off both clamps): DOWN-by-c == beta fell on %d, not on %d"
          % (agree, disagree))
    post = []
    pooled = {}
    for s in SEEDS:
        cells = []
        for lo in range(0, 300, 50):
            sel = [r for r in recs[s] if lo * SPE <= r["step"] < (lo + 50) * SPE]
            dn = sum(1 for r in sel if r["pt_b2"] * r["mom_pre"][0][0] + (1 - r["pt_b2"]) * r["z_agg"][0][0] > 0)
            pooled.setdefault(lo, [0, 0])
            pooled[lo][0] += dn
            pooled[lo][1] += len(sel)
            cells.append("%d-%d:%d/%d=%.3f" % (lo, lo + 50, dn, len(sel), dn / len(sel)))
            if lo >= 100:
                post.append(dn / len(sel))
        print("    s%d  %s" % (s, "  ".join(cells)))
    print("    pooled %s" % "  ".join("%d-%d:%d/%d=%.3f" % (lo, lo + 50, d, n, d / n) for lo, (d, n) in sorted(pooled.items())))
    pp = [d / n for lo, (d, n) in sorted(pooled.items()) if lo >= 100]
    print("    RANGE per seed, the 12 windows from epoch 100: min %.3f  max %.3f" % (min(post), max(post)))
    print("    RANGE seeds pooled, the 4 windows from epoch 100: min %.3f  max %.3f" % (min(pp), max(pp)))

    print("\n(c) K13 s94 complement r = exp(beta[0]+15) after its first record with r <= 2")
    R = recs[94]
    first = next(i for i, r in enumerate(R) if math.exp(r["beta"][0] + 15) <= 2)
    after = R[first + 1:]
    print("    first r <= 2 at epoch %.1f" % (R[first]["step"] / SPE))
    top = max(r["beta"][0] for r in after)
    same = [r for r in after if r["beta"][0] == top]
    print("    max r %.4f (beta %.6f); records with exactly this beta: %s"
          % (math.exp(top + 15), top, ", ".join("%.1f" % (r["step"] / SPE) for r in same)))
    label = "%.3f" % math.exp(top + 15)
    printed = [r for r in after if "%.3f" % math.exp(r["beta"][0] + 15) == label]
    print("    records whose r prints as %s: %s" % (label, ", ".join("%.1f (beta %.6f)" % (r["step"] / SPE, r["beta"][0]) for r in printed)))


if __name__ == "__main__":
    main()
