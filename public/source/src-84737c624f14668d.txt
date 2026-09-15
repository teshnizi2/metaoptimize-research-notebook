#!/usr/bin/env python3
"""cvk1_attack_indep.py <runsdir> -- independent re-derivation of cvk1 (CORRECTIONS 210).

Imports nothing from cVK1/cVK2 or any repo module.  Reads raw .out Epoch lines by string
splitting (no regex).  plateau5 = mean TEST over epochs 95..99; TRAIN beside TEST.
Bars from CORRECTIONS 201/208 as registered: SIGMA_USED = max(0.586232 floor,
0.5862321062215915 NARROW pin, 0.3638879864280581 VGG pin, in-batch sigma),
SE = SIGMA_USED*sqrt(2/3), PEAK_BAR = 2 SE, PEAK_SET = sweep arms within PEAK_BAR of the argmax.
"""
import math
import os
import sys

GRID = [13, 16, 19, 20, 21, 22, 23]
ARMS = ["k01", "kL"] + ["k%d" % k for k in GRID]
SEEDS = [55, 56, 57]


def read(path):
    te, tr, done = {}, {}, False
    for ln in open(path, errors="replace"):
        if ln.startswith("Epoch "):
            p = ln.split(",")
            e = int(p[0].split()[1])
            tr[e] = float(p[1].split(":")[1].replace("%", ""))
            te[e] = float(p[2].split(":")[1].replace("%", ""))
        elif ln.strip() == "RUN_DONE":
            done = True
    assert sorted(te) == list(range(100)) and done, path
    return sum(te[e] for e in range(95, 100)) / 5, sum(tr[e] for e in range(95, 100)) / 5


def main():
    d = sys.argv[1]
    fs = {}
    for fn in os.listdir(d):
        if fn.startswith("cvk1-") and fn.endswith(".out"):
            stem = fn[:-4].rpartition("-")[0]
            fs.setdefault(stem, []).append(fn)
    m, sd, t = {}, {}, {}
    ss = df = 0.0
    for a in ARMS:
        v, w = [], []
        for s in SEEDS:
            (fn,) = fs["cvk1-%s-s%d" % (a, s)]
            x, y = read(os.path.join(d, fn))
            v.append(x)
            w.append(y)
        m[a] = sum(v) / 3
        t[a] = sum(w) / 3
        sd[a] = math.sqrt(sum((x - m[a]) ** 2 for x in v) / 2)
        ss += sum((x - m[a]) ** 2 for x in v)
        df += 2
        print("%-4s TEST %.4f sd %.3f  TRAIN %.4f  seeds %s" % (a, m[a], sd[a], t[a], ["%.3f" % x for x in v]))
    sin = math.sqrt(ss / df)
    sig = max(0.586232, 0.5862321062215915, 0.3638879864280581, sin)
    se = sig * math.sqrt(2.0 / 3.0)
    sweep = ["k%d" % k for k in GRID]
    top = max(sweep, key=lambda a: m[a])
    peak = [int(a[1:]) for a in sweep if m[top] - m[a] <= 2 * se]
    rest = sorted((a for a in sweep if a != top), key=lambda a: -m[a])
    print("sigma in-batch %.6f (df %d) used %.6f SE %.6f PEAK_BAR %.4f" % (sin, df, sig, se, 2 * se))
    print("argmax %s %.4f, runner-up %s %.4f, margin %.4f = %.2f SE" % (top, m[top], rest[0], m[rest[0]],
                                                                         m[top] - m[rest[0]], (m[top] - m[rest[0]]) / se))
    print("PEAK_SET %s" % peak)
    print("D22-16 %+.4f = %+.2f SE | CLIFF 22-23 %+.4f = %+.2f SE (TRAIN %+.4f) | GAP kL-k01 %+.4f"
          % (m["k22"] - m["k16"], (m["k22"] - m["k16"]) / se, m["k22"] - m["k23"], (m["k22"] - m["k23"]) / se,
             t["k22"] - t["k23"], m["kL"] - m["k01"]))
    trtop = max(sweep, key=lambda a: t[a])
    print("TRAIN argmax %s %.4f" % (trtop, t[trtop]))
    print("k23 - k01 %+.4f = %+.2f SE (floored if within 2 SE of k01)" % (m["k23"] - m["k01"], (m["k23"] - m["k01"]) / se))


if __name__ == "__main__":
    main()
