#!/usr/bin/env python3
r"""c67_ms_axis.py -- THE META-STEP-SIZE AXIS: DO THE TWO LADDERS MOVE TOGETHER?

WHY THIS EXISTS
---------------
Cycle 65 (FINDINGS 65.2/65.3, CORRECTIONS 94.2) put the campaign's two ladders on one
axis `u` and measured a five-decade separation:

    field   E(u) peaks at u* ~ 0.0017   (1-9 coordinates, the kernel scale), 17/17 arms
    training  peaks at LAYERWISE, u ~ 440, in 16 of 21 budget-matched cells
    K2 SEPARATION (>= 2 decades) fires 20/21, median log10 = 5.41

Cycle 66 (CORRECTIONS 95) then discharged the caveat 94.7 had installed -- the field's
LOCATION does not move across four trajectory classes -- and 95.8 recorded that PROBE7 is
no longer the binding uncertainty in direction C.

**So what IS the binding uncertainty now?**  Read 65.3's own table: every primary cell in
it is `ms1e-3`.  The meta-step-size is the knob that decides how far beta is allowed to
travel, i.e. how much of any structure the partition exposes can actually be exploited.
K2 has never been scored against it.  Under STANDING RULE (12) -- *a statistic aggregated
over a parameter stack must state BOTH its arm/tensor restriction AND its evaluation
window before it is quoted* -- the honest current statement of K2 is

    K2 fires 20/21 [weightwise-probed field; 20-epoch training window; **ms = 1e-3**]

and the third bracket has never been written down.  This module writes it, or removes it.

THE DESIGN, AND WHY IT IS UNUSUALLY CLEAN FOR THIS CAMPAIGN
-----------------------------------------------------------
The corpus already contains a MATCHED ms ladder in which BOTH readouts exist at every
rung: ResNet18 / CIFAR10 / 20 epochs / alpha0=1e-3 / AUGMENT=1 / SGDm base / Lion meta /
batch 100, five rungs spanning 100x --

    rung   ms      training grans (n)              weightwise field probes (n)
    1      1e-4    lay,node,blk6,w  (3 each)       ml5/probe_w_m4_s*   (3)
    2      2e-4    lay,node,w       (3 each)       ns5/probe_w_m2p4_s* (3)
    3      5e-4    lay,node,w       (2 each)       ns5/probe_w_m5p4_s* (2)
    4      1e-3    lay,node,blk6,w  (3 each, ml5)  ml5/probe_w_m3_s*   (3)
    5      1e-2    lay,node,blk6,w  (3 each)       ml5/probe_w_m2_s*   (3)

Rungs 1, 4 and 5 are the SAME FAMILY (`ml5`) varying ONLY `ms`; rungs 2 and 3 come from
`ns5` and carry a different `beta_clip`, which is why C1 below exists.  The field side is
c62's `E(u)` computed by c63's `measure_arm` -- **by reference, not reimplemented** (T1).
The training side is c65's `clean_rows` / `ladder_of_cell` / `u_point` -- also by
reference (T2).  Nothing about either ladder is redefined here; the only new object is
the ms axis they are both indexed on.

PROVENANCE, DECLARED PER TEST BEFORE ANY SCORING (95.5's practice)
-------------------------------------------------------------------
This matters more than usual here, because the tick's DISCOVERY was post-hoc.

**POST-HOC-INFORMED (P-class).**  Before writing this file I ran a plain cross-tabulation
of the CSV and READ the mean plateau of every (ms x beta_clip x granularity) cell in the
matched design.  P1, P2, C1 and C2 are therefore *descriptions of a table already seen*.
They are printed with that label and, per 76(1)/79, they carry NO independent weight.

**BLIND (B-class).**  B1-B5 and B7 are computed from quantities that appear in NO output
that preceded this registration: individual seed values, `best_test`, the field ladder of
14 of the 15 probe arms, and the 100-epoch stratum.

**ONE DECLARED LEAK.**  To check runtime feasibility I ran `measure_arm` on exactly one
arm, `probes_ml5/probe_w_m4_s0`, and saw `u_peak = 1/512, E_peak = 0.1984, res = 0.0090`.
That single arm is 1 of 15 on the field side and is NOT excluded (excluding it would be a
worse distortion than declaring it).  B3 and B4 are labelled `BLIND-1LEAK`.

REGISTERED TESTS -- WRITTEN AND COMMITTED BEFORE ANY SCORE WAS COMPUTED
------------------------------------------------------------------------
P1  TRAINING ARGMAX MOVES WITH ms  [P-class].
    Let `u_train*(ms, clip)` be the argmax of the training ladder in u units.
    REGISTERED: at ms=1e-2 the argmax is strictly FINER (smaller u) than at ms=1e-3, in
    every clip stratum that has n>=2 at both rungs, AND the winning margin at ms=1e-2
    exceeds `2 * sqrt(sem_1^2 + sem_2^2)` of the top two granularities.
    REFUTATION: fails in any such stratum.

P2  THE ORDERED TREND  [P-class].
    Spearman rho between log10(ms) and `-log2(u_train*)` (i.e. FINENESS of the optimum)
    over the five rungs, using the ml5-or-lowest-clip stratum at each rung.
    REGISTERED: rho >= +0.7.
    REFUTATION: rho < +0.7.  (Only 5 points and few distinct argmax values; this is a
    coarse statistic and is reported with its ties.)

B1  PER-SEED PAIRING  [BLIND].
    The ms=1e-2 reversal at the level of MATCHED SEEDS, not means: for each family x seed
    at ms=1e-2, compare nodewise vs layerwise.
    REGISTERED: nodewise > layerwise in >= 5 of the 6 matched pairs (ml5 s0-s2 + wc5
    s0-s2).  REFUTATION: <= 3 of 6.
    NEGATIVE CONTROL, same test at ms=1e-3: layerwise > nodewise in >= 5 of 6.  If the
    negative control fails the seed pairing is uninformative and B1 is VOID.

B2  SECONDARY READOUT  [BLIND].
    The reversal must not be an artifact of the plateau window (METRIC RULES warn that
    `best_test` inflates ~0.37pp; here it is used only as a CONSISTENCY check, never as
    the primary).
    REGISTERED: on `best_test`, nodewise > layerwise at ms=1e-2 in both clip strata, AND
    layerwise > nodewise at ms=1e-3 in all four clip strata.
    REFUTATION: either half fails.

B3  FIELD INVARIANCE ON THE SAME AXIS  [BLIND-1LEAK].
    `u*_E` over the 15 matched weightwise probe arms.
    REGISTERED: `max log2(u*) - min log2(u*) <= 1.0` (one ladder rung) across a 100x ms
    range.  REFUTATION: > 1.0 rung.
    This is 66's B3 generalised from one exception arm to a matched five-rung ladder.

B4  AMPLITUDE -- THE POWER CONTROL  [BLIND-1LEAK].
    95.4's lesson: an invariance is vacuous unless the knob demonstrably does something.
    REGISTERED: the geometric max/min of `E*` (peak amplitude) across the five ms rungs is
    >= 1.5x.
    REFUTATION: < 1.5x  ->  **B3 IS WITHDRAWN** and this tick reports only that a
    manipulation which changed nothing changed nothing.

B5  K2 ACROSS THE ms AXIS -- THE TEST THE PAPER NEEDS  [BLIND, derived].
    At each rung, `sep(ms) = log10(u_train*(ms) / u_field*(ms))`.
    REGISTERED: `sep >= 2.0` at ALL FIVE rungs (K2's own bar, applied rung by rung).
    REFUTATION: any rung < 2.0  ->  K2 is ms-scoped and 65.4/94.2 must say so.
    Censoring is handled exactly as c65 does it: a layerwise argmax is coarser than the
    ladder top, so its separation is a LOWER bound taken at the top rung.

B7  BUDGET TRANSFER  [BLIND].
    At 100 epochs, alpha0=1e-3, layerwise and nodewise coexist at ms = 1e-4, 3e-4, 1e-3.
    REGISTERED: layerwise > nodewise at all three -- i.e. the 20-epoch ordering at those
    same ms rungs survives a 5x budget.
    REFUTATION: nodewise wins any of the three.
    (There is no 100-epoch cell at ms=1e-2 with both granularities, so B7 CANNOT test the
    reversal itself.  It tests only that the pre-reversal half of the trend is not a
    20-epoch artifact.  Stated here so the gap is not discovered later.)

C1  CLIP IS NOT A CONFOUND  [P-class, CALIBRATION ONLY, never a verdict].
    At ms=1e-3 four clip strata exist.  REGISTERED bar: max-min of the LAYERWISE plateau
    across them <= 0.5 pp, and likewise for nodewise and weightwise.  This licenses
    comparing an ml5 rung against an ns5 rung.
    If C1 fails, rungs 2 and 3 are dropped and P2/B5 are rescored on rungs 1/4/5 only.

C2  ms IS NOT DEGENERATE AT THE TOP RUNG  [P-class].
    plateau rises monotonically with ms on layerwise, nodewise and weightwise.  ms=1e-2
    must be the BEST rung, not a diverging one, or the reversal is a collapse artifact.

SCOPE, STATED UP FRONT (STANDING RULE 10 / 12)
----------------------------------------------
Everything here is [ResNet18 / CIFAR10 / 20-epoch window / alpha0=1e-3 / AUGMENT=1 / SGDm
base / Lion meta / batch 100], field measured in the WEIGHTWISE arm (65.7's instrument
limit, unchanged -- 66 bounded it, it did not remove it).  B7 is the only 100-epoch test.
We measure `z`, the META-gradient; Adam-mini / Adalayer / SGG argue about `G`.  **No
document may write "we refuted Adam-mini."**

STANDING RULE (13) STILL BINDS.  Whatever this module finds about where TRAINING peaks,
no sentence of the form "the field has structure at scale X therefore partition at scale
X" is licensed by it.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c62_blocksize_curve as C62          # noqa: E402
import c63_uladder_corpus as C63           # noqa: E402
import c65_field_vs_training as C65        # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
BACKUP = os.path.abspath(os.path.join(ROOT, ".."))
CSV_PATH = os.path.join(ROOT, "results", "all_runs.csv")

# --------------------------------------------------------------------------------------
# The matched design.  (relpath, ms, clip_tag, seed)
# --------------------------------------------------------------------------------------

MS_RUNGS = ["1e-4", "2e-4", "5e-4", "1e-3", "1e-2"]

# Field side: weightwise probes, one entry per (ms, seed).
FIELD_SET = [
    ("probes_ml5/probe_w_m4_s0",   "1e-4", "-15:-2.3026", 0),
    ("probes_ml5/probe_w_m4_s1",   "1e-4", "-15:-2.3026", 1),
    ("probes_ml5/probe_w_m4_s2",   "1e-4", "-15:-2.3026", 2),
    ("probes_ns5/probe_w_m2p4_s0", "2e-4", "-30:0.0",     0),
    ("probes_ns5/probe_w_m2p4_s1", "2e-4", "-30:0.0",     1),
    ("probes_ns5/probe_w_m2p4_s2", "2e-4", "-30:0.0",     2),
    ("probes_ns5/probe_w_m5p4_s0", "5e-4", "-30:0.0",     0),
    ("probes_ns5/probe_w_m5p4_s1", "5e-4", "-30:0.0",     1),
    ("probes_ml5/probe_w_m3_s0",   "1e-3", "-15:-2.3026", 0),
    ("probes_ml5/probe_w_m3_s1",   "1e-3", "-15:-2.3026", 1),
    ("probes_ml5/probe_w_m3_s2",   "1e-3", "-15:-2.3026", 2),
    ("probes_ml5/probe_w_m2_s0",   "1e-2", "-15:-2.3026", 0),
    ("probes_ml5/probe_w_m2_s1",   "1e-2", "-15:-2.3026", 1),
    ("probes_ml5/probe_w_m2_s2",   "1e-2", "-15:-2.3026", 2),
]

# Training side: the fixed cell coordinates every row must match.
FIXED = dict(network="ResNet18", dataset="CIFAR10", base="SGDm", meta="Lion",
             alpha0="1e-3", augment="1", batch_size="100", gamma="1")

GRANS_TRAIN = ["layerwise", "nodewise", "weightwise", "resnet18_blocks", "scalar"]


# --------------------------------------------------------------------------------------
# CSV side
# --------------------------------------------------------------------------------------

def load_rows():
    with open(CSV_PATH) as fh:
        return list(csv.DictReader(fh))


def matched_rows(rows, epochs="20", fixed=None):
    """c65's clean_rows, then the fixed-cell filter.  Selection rule is c65's (T2)."""
    fixed = FIXED if fixed is None else fixed
    out = []
    for r in C65.clean_rows(rows):
        if str(r.get("epochs_requested", "")) != str(epochs):
            continue
        if any(str(r.get(k, "")) != v for k, v in fixed.items()):
            continue
        out.append(r)
    return out


def by_ms_clip(rows):
    """{(ms, clip): {gran: [ (plateau, best_test, seed, run) ... ]}}"""
    acc = {}
    for r in rows:
        k = (r["meta_stepsize"], r["beta_clip"])
        try:
            bt = float(r.get("best_test", ""))
        except (TypeError, ValueError):
            bt = float("nan")
        run = r.get("run", "")
        # A seed label is only unique WITHIN a family: `p3` and `z3` both ship seed 0.
        # T12 caught this.  The pairing key for B1 is therefore (family, seed).
        fam = run.split("-")[0]
        acc.setdefault(k, {}).setdefault(r["granularity"], []).append(
            (r["_plateau"], bt, (fam, r.get("seed", "")), run))
    return acc


def ladder(cell, key=0):
    """{gran: (mean, sem, n)} using c65's summariser."""
    return {g: C65.summarise([t[key] for t in v])
            for g, v in cell.items() if g in GRANS_TRAIN}


def argmax_gran(lad, min_n=2):
    elig = {g: v for g, v in lad.items() if v[2] >= min_n}
    if not elig:
        return None, None
    order = sorted(elig, key=lambda g: -elig[g][0])
    if len(order) == 1:
        return order[0], float("inf")
    (m1, s1, _), (m2, s2, _) = elig[order[0]], elig[order[1]]
    s1 = 0.0 if s1 != s1 else s1
    s2 = 0.0 if s2 != s2 else s2
    pooled = math.sqrt(s1 * s1 + s2 * s2)
    margin = (m1 - m2) / (2.0 * pooled) if pooled > 0 else float("inf")
    return order[0], margin


def spearman(x, y):
    """Spearman rho with average ranks; returns nan if either side is constant."""
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(list(x)), rank(list(y))
    ax, ay = np.array(rx), np.array(ry)
    if ax.std() == 0 or ay.std() == 0:
        return float("nan")
    return float(np.corrcoef(ax, ay)[0, 1])


# --------------------------------------------------------------------------------------
# Field side
# --------------------------------------------------------------------------------------

def measure_field(seeds=(101, 202, 303), limit=None):
    arms = []
    for rel, ms, clip, sd in (FIELD_SET[:limit] if limit else FIELD_SET):
        a = C63.measure_arm(BACKUP, rel, f"r18 ms{ms}", seeds)
        if a is None:
            print(f"  MISSING {rel}")
            continue
        a["ms"] = ms
        a["clip"] = clip
        a["seed"] = sd
        arms.append(a)
    return arms


# ======================================================================================
# SELFTESTS
# ======================================================================================

_N_OK = 0


def ok(cond, msg):
    global _N_OK
    if not cond:
        raise AssertionError(msg)
    _N_OK += 1


def selftest():
    global _N_OK
    _N_OK = 0

    # T1/T2 -- provenance: the two ladders are IMPORTED, not restated.
    ok(C63.C62 is C62, "T1 c63 must be driving the same c62 module object")
    ok(C65.C62 is C62, "T1b c65 must be driving the same c62 module object")
    ok(callable(C63.measure_arm), "T1c measure_arm must come from c63")
    ok(callable(C65.clean_rows) and callable(C65.summarise) and callable(C65.u_point),
       "T2 training-side selection/summary/u-map must come from c65")

    # T3 -- the design table is internally consistent.
    ok(len(FIELD_SET) == 14, f"T3 expected 14 field arms, got {len(FIELD_SET)}")
    got_ms = sorted({m for _r, m, _c, _s in FIELD_SET}, key=float)
    ok(got_ms == sorted(MS_RUNGS, key=float), f"T3b ms rungs mismatch: {got_ms}")
    ok(len({r for r, _m, _c, _s in FIELD_SET}) == 14, "T3c duplicate field arm path")
    # rungs 1/4/5 must be one family varying only ms
    ml5 = {m for r, m, _c, _s in FIELD_SET if r.startswith("probes_ml5/")}
    ok(ml5 == {"1e-4", "1e-3", "1e-2"}, f"T3d ml5 must supply exactly 3 rungs, got {ml5}")

    # T4 -- u_point orders the granularities the way the axis claims.
    sh = C65.shapes_for_family("r18")
    ok(sh is not None, "T4 r18 shapes must load from a real probe")
    uw = C65.u_point(sh, "weightwise")
    un = C65.u_point(sh, "nodewise")
    ul = C65.u_point(sh, "layerwise")
    ok(uw < un < ul, f"T4b expected u(w) < u(node) < u(lay), got {uw}, {un}, {ul}")
    ok(un == 1.0, "T4c nodewise must be exactly u=1 (the literature's `out` partition)")
    ok(math.isinf(C65.u_point(sh, "scalar")), "T4d scalar must be off the axis")

    # T5 -- argmax_gran picks the max and its margin is in units of pooled sem.
    lad = {"layerwise": (70.0, 0.10, 3), "nodewise": (71.0, 0.10, 3),
           "weightwise": (60.0, 0.10, 3)}
    g, m = argmax_gran(lad)
    ok(g == "nodewise", "T5 argmax must be nodewise")
    ok(m is not None and abs(m - 1.0 / (2 * math.sqrt(0.02))) < 1e-9,
       f"T5b margin math wrong: {m}")
    ok(argmax_gran({"layerwise": (70.0, 0.1, 1)})[0] is None,
       "T5c a single n=1 granularity must not yield an argmax")

    # T6 -- spearman is correct on known cases, and ties are averaged.
    ok(abs(spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-12, "T6 rho=+1")
    ok(abs(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-12, "T6b rho=-1")
    ok(spearman([1, 2, 3], [1, 1, 1]) != spearman([1, 2, 3], [1, 1, 1]),
       "T6c constant y must give nan")
    r = spearman([1, 2, 3, 4], [1, 2, 2, 3])
    ok(0.9 < r < 1.0, f"T6d tie handling: {r}")

    # T7 -- the CSV really contains the matched design at the stated n.
    rows = matched_rows(load_rows())
    cells = by_ms_clip(rows)
    for ms in MS_RUNGS:
        strata = [k for k in cells if k[0] == ms]
        ok(strata, f"T7 no cell at ms={ms}")
        best = max(strata, key=lambda k: len(cells[k]))
        for g in ("layerwise", "nodewise", "weightwise"):
            n = len(cells[best].get(g, []))
            ok(n >= 2, f"T7b ms={ms} {best[1]} {g} has n={n} (<2)")

    # T8 -- every field arm path exists on disk (fail loudly, not silently).
    for rel, _ms, _c, _s in FIELD_SET:
        ok(os.path.isdir(os.path.join(BACKUP, rel)), f"T8 missing probe dir {rel}")

    # T9 -- clean_rows really excludes hier/collapsed/superseded from the matched set.
    ok(all(str(r.get("hier", "")).strip() in ("", "na", "none") for r in rows),
       "T9 hier arms must be excluded (CORRECTIONS 21)")
    ok(all(str(r.get("collapsed", "")).lower() not in ("1", "true") for r in rows),
       "T9b collapsed rows must be excluded")

    # T10 -- the ms=1e-3 stratum has the >=2 clip settings C1 needs.
    clips_1e3 = {k[1] for k in cells if k[0] == "1e-3"}
    ok(len(clips_1e3) >= 2, f"T10 C1 needs >=2 clip strata at ms=1e-3, got {clips_1e3}")

    # T11 -- best_test is present (B2 is unscorable without it).
    bt = [t[1] for c in cells.values() for v in c.values() for t in v]
    ok(sum(1 for x in bt if x == x) > 0.9 * len(bt), "T11 best_test mostly missing")

    # T12 -- the (family, seed) key must be UNIQUE inside every (ms, clip, gran) cell,
    # or B1's matched pairing silently joins the wrong runs.  A bare seed label is NOT
    # unique: `p3` and `z3` both ship seed 0 into the ms=1e-3 scalar cell.
    for k, cell in cells.items():
        for g, v in cell.items():
            keys = [t[2] for t in v]
            ok(len(set(keys)) == len(keys),
               f"T12 duplicate (family,seed) key in {k} {g}: {keys}")
    bare = [t[2][1] for t in cells[("1e-3", "-15:-2.3026")]["scalar"]]
    ok(len(set(bare)) < len(bare),
       "T12b the scalar/ms=1e-3 cell must still demonstrate the duplicate BARE seed "
       "label that motivates the (family,seed) key")

    # T13 -- separation arithmetic: a known pair reproduces c65's formula.
    ok(abs(math.log10(440.0 / 0.0017) - 5.4128) < 1e-3, "T13 sep arithmetic")

    # T14 -- the E ladder end-identities hold on a synthetic field (65.1), so a peak
    # this module reads is interior by construction and not a boundary artifact.
    rng = np.random.default_rng(0)
    p = rng.random(4096).astype(np.float64)
    shapes = [("t", (16, 16, 4, 4))]
    rows_e = C62.curve(p, shapes, 100, [C62.U_LADDER[0], C62.U_LADDER[-1]], (101,))
    ok(all(abs(r["E_pp"]) < 1e-6 for r in rows_e),
       f"T14 both ladder ends must be identities, got {[r['E_pp'] for r in rows_e]}")

    print(f"  selftest: {_N_OK}/{_N_OK} PASS")
    return _N_OK


# ======================================================================================
# SCORING
# ======================================================================================

def report(out_json=None, field_limit=None):
    rows = load_rows()
    m20 = matched_rows(rows, "20")
    cells = by_ms_clip(m20)

    print("=" * 112)
    print("C67 -- THE META-STEP-SIZE AXIS.  Do the field ladder and the training ladder")
    print("       move together?   [R18 / CIFAR10 / 20ep / a0=1e-3 / AUG / SGDm / Lion]")
    print("=" * 112)

    # ---------------- PART A: the training ladder, rung by rung ----------------------
    print("\nPART A -- TRAINING LADDER  (plateau, mean +- sem (n))   [P-class: table seen"
          " before registration]")
    hdr = f"{'ms':>6}{'clip':>15}" + "".join(f"{g[:9]:>17}" for g in
                                             ("layerwise", "nodewise", "weightwise",
                                              "blocks"))
    print(hdr + "   argmax   margin")
    train = {}
    for k in sorted(cells, key=lambda k: (float(k[0]), k[1])):
        lad = ladder(cells[k])
        g, marg = argmax_gran(lad)
        train[k] = (lad, g, marg)
        cellstr = ""
        for gg in ("layerwise", "nodewise", "weightwise", "resnet18_blocks"):
            if gg in lad:
                m, s, n = lad[gg]
                cellstr += f"{m:>9.3f}+-{0.0 if s != s else s:>4.2f}({n:>1d})"
            else:
                cellstr += f"{'--':>17}"
        mstr = "inf" if marg == float("inf") else (f"{marg:.2f}" if marg is not None else "-")
        print(f"{k[0]:>6}{k[1]:>15}" + cellstr + f"   {str(g):>10} {mstr:>7}")

    # ---------------- C1 / C2 : calibration ------------------------------------------
    print("\nC1 [P-class, CALIBRATION] clip spread at ms=1e-3 (bar <= 0.5 pp):")
    c1_ok = True
    for g in ("layerwise", "nodewise", "weightwise"):
        vals = [train[k][0][g][0] for k in train if k[0] == "1e-3" and g in train[k][0]]
        if len(vals) >= 2:
            sp = max(vals) - min(vals)
            c1_ok &= sp <= 0.5
            print(f"   {g:>12}: n_clip={len(vals)}  spread = {sp:.3f} pp  "
                  f"-> {'PASS' if sp <= 0.5 else 'FAIL'}")
    print(f"   C1 -> {'PASS -- ml5 and ns5 rungs are comparable' if c1_ok else 'FAIL'}")

    print("\nC2 [P-class] monotone accuracy in ms (ms=1e-2 must be best, not diverging):")
    c2_ok = True
    for g in ("layerwise", "nodewise", "weightwise"):
        seq = []
        for ms in MS_RUNGS:
            ks = [k for k in train if k[0] == ms and g in train[k][0]]
            if ks:
                k = max(ks, key=lambda k: train[k][0][g][2])
                seq.append((ms, train[k][0][g][0]))
        mono = all(seq[i + 1][1] > seq[i][1] for i in range(len(seq) - 1))
        c2_ok &= mono
        print(f"   {g:>12}: " + " -> ".join(f"{v:.2f}" for _m, v in seq)
              + f"   {'MONOTONE' if mono else 'NOT MONOTONE'}")
    print(f"   C2 -> {'PASS' if c2_ok else 'FAIL'}")

    # ---------------- P1 : argmax moves ----------------------------------------------
    print("\nP1 [P-class] argmax at ms=1e-2 strictly FINER than at ms=1e-3, per clip:")
    sh = C65.shapes_for_family("r18")
    p1_hits, p1_tot = 0, 0
    for clip in sorted({k[1] for k in train if k[0] == "1e-2"}):
        k2_, k3_ = ("1e-2", clip), ("1e-3", clip)
        if k3_ not in train:
            print(f"   clip {clip}: no ms=1e-3 counterpart -- not scorable")
            continue
        g2, m2 = train[k2_][1], train[k2_][2]
        g3, _m3 = train[k3_][1], train[k3_][2]
        u2, u3 = C65.u_point(sh, g2), C65.u_point(sh, g3)
        hit = (u2 < u3) and (m2 is not None) and (m2 > 1.0)
        p1_tot += 1
        p1_hits += int(hit)
        print(f"   clip {clip:>15}: 1e-3 -> {g3:<11}(u={u3:.1f})   "
              f"1e-2 -> {g2:<11}(u={u2:.1f})  margin={m2:.2f} sem "
              f"-> {'HIT' if hit else 'MISS'}")
    print(f"   P1 -> {p1_hits}/{p1_tot} "
          f"{'HELD' if p1_tot and p1_hits == p1_tot else 'REFUTED'}")

    # ---------------- P2 : ordered trend ---------------------------------------------
    print("\nP2 [P-class] Spearman(log10 ms, fineness of the training argmax):")
    xs, ys, lab = [], [], []
    for ms in MS_RUNGS:
        ks = [k for k in train if k[0] == ms and train[k][1] is not None]
        if not ks:
            continue
        k = max(ks, key=lambda k: sum(v[2] for v in train[k][0].values()))
        g = train[k][1]
        u = C65.u_point(sh, g)
        xs.append(math.log10(float(ms)))
        ys.append(-math.log2(u) if u > 0 and not math.isinf(u) else -60.0)
        lab.append((ms, k[1], g))
    rho = spearman(xs, ys)
    for (ms, clip, g), y in zip(lab, ys):
        print(f"   ms={ms:>6} clip={clip:>15} argmax={g:<12} fineness=-log2(u)={y:+.2f}")
    print(f"   P2 rho = {rho:+.3f}  -> "
          f"{'HELD' if rho == rho and rho >= 0.7 else 'REFUTED'}")

    # ---------------- B1 : per-seed pairing ------------------------------------------
    print("\nB1 [BLIND] per-seed nodewise-vs-layerwise pairing:")
    b1 = {}
    for ms in ("1e-2", "1e-3"):
        pairs = []
        for k in sorted(cells):
            if k[0] != ms:
                continue
            cell = cells[k]
            if "nodewise" not in cell or "layerwise" not in cell:
                continue
            nd = {t[2]: t[0] for t in cell["nodewise"]}
            ly = {t[2]: t[0] for t in cell["layerwise"]}
            for sd in sorted(set(nd) & set(ly)):
                pairs.append((k[1], sd, nd[sd], ly[sd]))
        b1[ms] = pairs
        wins = sum(1 for _c, _s, n, l in pairs if n > l)
        print(f"   ms={ms}: {len(pairs)} matched pairs, nodewise wins {wins}")
        for c, sd, n, l in pairs:
            print(f"      clip={c:>15} {sd[0]:>4}/s{sd[1]:<2}  node={n:7.3f}  "
                  f"lay={l:7.3f}  diff={n - l:+7.3f}")
    n_hi = sum(1 for _c, _s, n, l in b1.get("1e-2", []) if n > l)
    n_lo = sum(1 for _c, _s, n, l in b1.get("1e-3", []) if l > n)
    tot_hi, tot_lo = len(b1.get("1e-2", [])), len(b1.get("1e-3", []))
    ctrl = tot_lo > 0 and n_lo >= max(1, tot_lo - 1)
    b1_verdict = ("VOID (negative control failed)" if not ctrl else
                  ("HELD" if n_hi >= max(1, tot_hi - 1) else
                   ("REFUTED" if n_hi <= tot_hi // 2 else "UNDECIDED")))
    print(f"   negative control ms=1e-3 layerwise wins {n_lo}/{tot_lo}  "
          f"-> {'PASS' if ctrl else 'FAIL'}")
    print(f"   B1 -> {n_hi}/{tot_hi} at ms=1e-2  -> {b1_verdict}")

    # ---------------- B2 : best_test ---------------------------------------------------
    print("\nB2 [BLIND] same comparison on best_test (consistency only, never primary):")
    b2_hi, b2_hi_t, b2_lo, b2_lo_t = 0, 0, 0, 0
    for k in sorted(cells, key=lambda k: (float(k[0]), k[1])):
        if k[0] not in ("1e-2", "1e-3"):
            continue
        lad_bt = ladder(cells[k], key=1)
        if "nodewise" not in lad_bt or "layerwise" not in lad_bt:
            continue
        n, l = lad_bt["nodewise"][0], lad_bt["layerwise"][0]
        if k[0] == "1e-2":
            b2_hi_t += 1
            b2_hi += int(n > l)
        else:
            b2_lo_t += 1
            b2_lo += int(l > n)
        print(f"   ms={k[0]:>6} clip={k[1]:>15}  node={n:7.3f}  lay={l:7.3f}  "
              f"diff={n - l:+7.3f}")
    b2_ok = (b2_hi_t and b2_hi == b2_hi_t) and (b2_lo_t and b2_lo == b2_lo_t)
    print(f"   B2 -> ms=1e-2 node>lay {b2_hi}/{b2_hi_t}; ms=1e-3 lay>node {b2_lo}/{b2_lo_t}"
          f"  -> {'HELD' if b2_ok else 'REFUTED'}")

    # ---------------- FIELD LADDER ------------------------------------------------------
    print("\nPART B -- FIELD LADDER  E(u) on the SAME ms axis, weightwise probes")
    arms = measure_field(limit=field_limit)
    print(f"{'arm':<34}{'ms':>7}{'peak':>8}{'E*':>10}{'res':>8}{'E/res':>8}{'E(1)':>9}")
    for a in arms:
        rr = a["E_peak"] / a["res_peak"] if a["res_peak"] > 0 else float("nan")
        print(f"{a['arm']:<34}{a['ms']:>7}{a['peak_label']:>8}{a['E_peak']:>10.4f}"
              f"{a['res_peak']:>8.4f}{rr:>8.1f}{a['E1']:>9.4f}")

    per_ms = {}
    for a in arms:
        per_ms.setdefault(a["ms"], []).append(a)

    print(f"\n{'ms':>7}{'n':>3}{'u*_E geo':>12}{'log2 u*':>10}{'E* geo':>10}{'range u*':>18}")
    field = {}
    for ms in MS_RUNGS:
        v = per_ms.get(ms, [])
        if not v:
            continue
        us = [a["u_peak"] for a in v]
        es = [a["E_peak"] for a in v]
        gu = math.exp(sum(math.log(u) for u in us) / len(us))
        ge = math.exp(sum(math.log(e) for e in es) / len(es))
        field[ms] = dict(u=gu, E=ge, n=len(v), lo=min(us), hi=max(us))
        print(f"{ms:>7}{len(v):>3}{gu:>12.6f}{math.log2(gu):>10.2f}{ge:>10.4f}"
              f"{C62.u_label(min(us)) + '..' + C62.u_label(max(us)):>18}")

    # ---------------- B3 / B4 ------------------------------------------------------------
    l2 = [math.log2(field[ms]["u"]) for ms in MS_RUNGS if ms in field]
    spread = (max(l2) - min(l2)) if l2 else float("nan")
    print(f"\nB3 [BLIND-1LEAK] field-peak spread across a 100x ms range = "
          f"{spread:.3f} rungs (bar <= 1.0)  -> "
          f"{'HELD' if spread <= 1.0 else 'REFUTED'}")
    Es = [field[ms]["E"] for ms in MS_RUNGS if ms in field]
    amp = (max(Es) / min(Es)) if Es and min(Es) > 0 else float("nan")
    b4 = amp >= 1.5
    print(f"B4 [BLIND-1LEAK, POWER CONTROL] E* geometric max/min across ms = {amp:.3f}x "
          f"(bar >= 1.5)  -> {'HELD' if b4 else 'REFUTED -- B3 IS WITHDRAWN'}")

    # ---------------- B5 : K2 rung by rung ----------------------------------------------
    print("\nB5 [BLIND] K2's own bar (log10 separation >= 2.0) applied RUNG BY RUNG:")
    u_top = max(r["u"] for r in arms[0]["rows"]) if arms else float("nan")
    b5_hits, b5_tot, seps = 0, 0, []
    for ms in MS_RUNGS:
        if ms not in field:
            continue
        ks = [k for k in train if k[0] == ms and train[k][1] is not None]
        if not ks:
            continue
        k = max(ks, key=lambda k: sum(v[2] for v in train[k][0].values()))
        g = train[k][1]
        ua = C65.u_point(sh, g)
        cens = math.isinf(ua)
        ua_eff = u_top if cens else ua
        sep = math.log10(ua_eff / field[ms]["u"])
        seps.append(sep)
        b5_tot += 1
        b5_hits += int(sep >= 2.0)
        print(f"   ms={ms:>6}  argmax={g:<12} u_train*={ua_eff:>8.2f}"
              f"{' (censored)' if cens else '':<11}  u_field*={field[ms]['u']:.6f}  "
              f"log10 sep = {sep:5.2f}  -> {'>=2' if sep >= 2.0 else 'BELOW BAR'}")
    print(f"   B5 -> {b5_hits}/{b5_tot} rungs at/above K2's bar  -> "
          f"{'HELD -- K2 is not ms-scoped' if b5_tot and b5_hits == b5_tot else 'REFUTED -- K2 IS ms-SCOPED'}")
    if seps:
        print(f"   separation range over the ms axis: {min(seps):.2f} .. {max(seps):.2f} "
              f"decades  (65.4 quotes median 5.41 at ms=1e-3 across all families)")

    # ---------------- B7 : budget transfer ----------------------------------------------
    print("\nB7 [BLIND] 100-epoch budget transfer, layerwise vs nodewise:")
    m100 = matched_rows(rows, "100")
    c100 = by_ms_clip(m100)
    b7_hits, b7_tot = 0, 0
    for k in sorted(c100, key=lambda k: (float(k[0]) if k[0] else 0, k[1])):
        lad100 = ladder(c100[k])
        if "layerwise" not in lad100 or "nodewise" not in lad100:
            continue
        l, n = lad100["layerwise"][0], lad100["nodewise"][0]
        b7_tot += 1
        b7_hits += int(l > n)
        print(f"   ms={k[0]:>7} clip={k[1]:>15}  lay={l:7.3f}(n={lad100['layerwise'][2]})"
              f"  node={n:7.3f}(n={lad100['nodewise'][2]})  diff={l - n:+7.3f}")
    print(f"   B7 -> layerwise wins {b7_hits}/{b7_tot}  -> "
          f"{'HELD' if b7_tot and b7_hits == b7_tot else 'REFUTED' if b7_tot else 'UNSCORABLE'}")

    if out_json:
        with open(out_json, "w") as f:
            json.dump(dict(train={f"{k[0]}|{k[1]}": dict(lad=v[0], argmax=v[1],
                                                         margin=None if v[2] is None or
                                                         math.isinf(v[2]) else v[2])
                                  for k, v in train.items()},
                           field=field, arms=arms,
                           score=dict(P1=[p1_hits, p1_tot], P2=rho,
                                      B1=[n_hi, tot_hi, n_lo, tot_lo],
                                      B2=[b2_hi, b2_hi_t, b2_lo, b2_lo_t],
                                      B3=spread, B4=amp, B5=[b5_hits, b5_tot],
                                      B7=[b7_hits, b7_tot], C1=c1_ok, C2=c2_ok)),
                      f, indent=1, default=float)
        print(f"\nwrote {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.run:
        report(out_json=a.json, field_limit=a.limit)


if __name__ == "__main__":
    main()
