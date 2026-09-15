#!/usr/bin/env python3
"""
RULE 13 (test every guard BOTH directions) for the `complete` column added in
cycle 88 / CORRECTIONS 119.3.

The column exists because `window_ok` was being used as if it meant "this run
finished", which it never did: it means "epochs_done > 20, so `plateau` is a
tail rather than the whole run".  Sixteen runs of 1,960 are window_ok==1 with
epochs_done < 0.9 * epochs_requested, and one of them sits in the PRIMARY
cell's blk6 arm.

Direction A -- the gate ADMITS what it should admit.
Direction B -- the gate REFUSES what it should refuse (8 injected violations).
Direction C -- the corpus-level consequence is exactly what was claimed.

Run:  python3 analysis/c88_complete_selftest.py
"""
import csv, math, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate import complete_of

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "all_runs.csv")

FAILS = []
def check(name, got, want):
    ok = (got == want)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name:52s} got={got!r} want={want!r}")
    if not ok:
        FAILS.append(name)

print("DIRECTION A -- the gate ADMITS a finished run")
check("A1 exact budget           100/100", complete_of(100, "100"), 1)
check("A2 at the 95% boundary     95/100", complete_of(95,  "100"), 1)
check("A3 over budget            101/100", complete_of(101, "100"), 1)
check("A4 short budget             20/20", complete_of(20,  "20"),  1)

print("DIRECTION B -- the gate REFUSES what it must refuse")
check("B1 one epoch below boundary 94/100", complete_of(94, "100"), 0)
check("B2 the real offender        29/100", complete_of(29, "100"), 0)
check("B3 half budget              50/100", complete_of(50, "100"), 0)
check("B4 near-miss                89/100", complete_of(89, "100"), 0)
check("B5 unknown budget      100/missing", complete_of(100, ""),   "")
check("B6 unparseable budget    100/'abc'", complete_of(100, "abc"), "")
check("B7 zero budget              100/0", complete_of(100, "0"),   "")
check("B8 negative budget         100/-1", complete_of(100, "-1"),  "")

print("DIRECTION C -- the corpus-level consequence")
rows = list(csv.DictReader(open(CSV)))
def f(r, k):
    try:    return float(r[k])
    except: return None

leak = [r for r in rows
        if r["window_ok"] == "1"
        and f(r, "epochs_done") is not None and f(r, "epochs_requested")
        and f(r, "epochs_done") < 0.9 * f(r, "epochs_requested")]
check("C1 window_ok=1 but <90% of budget", len(leak), 16)
check("C2 the primary-cell offender is present",
      any(r["run"] == "rs-blk6-1e4-s2" for r in leak), True)

PRI = dict(network="ResNet18", dataset="CIFAR10", base="SGDm", meta="Lion",
           meta_stepsize="1e-4", alpha0="1e-3", gamma="1", augment="1",
           beta_clip="-15:-2.3026", epochs_requested="100")
def cell(r):
    return all(r[k] == v for k, v in PRI.items())

def arm(gran, use_complete):
    v = []
    for r in rows:
        if not (cell(r) and r["granularity"] == gran and r["window_ok"] == "1"):
            continue
        if f(r, "plateau5") is None:
            continue
        if use_complete and complete_of(f(r, "epochs_done"), r["epochs_requested"]) != 1:
            continue
        v.append(f(r, "plateau5"))
    return v

before, after = arm("resnet18_blocks", False), arm("resnet18_blocks", True)
check("C3 blk6 n before/after", (len(before), len(after)), (6, 5))
check("C4 blk6 mean shift >= 1.2 pp",
      round(st.mean(after) - st.mean(before), 3) >= 1.2, True)
check("C5 blk6 mean after == 92.530", round(st.mean(after), 3), 92.530)
check("C6 blk6 sem inflation before >= 15x",
      (st.stdev(before)/math.sqrt(len(before))) /
      (st.stdev(after)/math.sqrt(len(after))) >= 15.0, True)

nb, na = arm("nodewise", False), arm("nodewise", True)
check("C7 nodewise n before/after", (len(nb), len(na)), (21, 20))
check("C8 nodewise mean after == 92.031", round(st.mean(na), 3), 92.031)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL 20 CHECKS PASS -- `complete` gate validated in both directions.")
