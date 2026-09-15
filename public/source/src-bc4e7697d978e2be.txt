#!/usr/bin/env python3
r"""c63_span_reconcile.py -- DISCHARGING THE `span` DEFINITIONAL DEBT OWED SINCE CYCLE 55.

WHY THIS EXISTS
---------------
FINDINGS 55.4 and CORRECTIONS 83.6 both recorded the same unpaid debt, and CORRECTIONS
91.8 / FINDINGS 62.8 then made it BLOCKING:

    "A DEFINITIONAL RECONCILIATION IS OWED.  CORRECTIONS 74 (N2) rejected adaptation
     extent because 'at ms=5e-4 the beta span is 8.27 log units (past its ~5 threshold)
     and the argmin is still `w`'.  This module measures that same cell's span as 6.29 ...
     The two statistics are not the same statistic and 74 is not contradicted here -- but
     N2's refutation rested on one ladder and on a span definition that must be reconciled
     before either number is quoted again."

    "[C62-A is] BLOCKED until CORRECTIONS 55.4's owed reconciliation of the two `span`
     definitions (6.29 vs 8.27 log units on the same cell) is discharged.  Registering it
     without that would repeat the CORRECTIONS 41 pathology."

So one unpaid definitional debt is holding a zero-compute registered test hostage.  This
module pays it.  It IMPORTS both definitions rather than restating them, so what is
compared is literally the published code:

    A = c53_score.beta_span(pattern)   -> mean over seeds of  span at the LAST RECORD
    B = c55_neff_noise.beta_span(dir)  -> mean over the STEADY HALF (records 0.5-1.0) of
                                          the same per-record span

Both read `beta_true_max - beta_true_min` from `probe.jsonl` through the SAME
`c52_boxfree.records()` (c53 imports it; c55 calls it).  Verified, not assumed, by G0.

THE CLAIM THIS MODULE TESTS -- REGISTERED BEFORE ANY CELL WAS SCORED
--------------------------------------------------------------------
The two numbers differ in exactly ONE respect: WHEN the span is evaluated.  If that is the
whole story then the difference is a time-averaging artifact of a quantity that grows
during training, and B <= A must hold identically, not statistically.

>>> THIS REGISTERED CLAIM IS **WRONG**, AND R1's BAR IS WHAT CAUGHT IT.  See `--factorial`
>>> and the RESULT block at the bottom of this docstring.  The claim is left standing here,
>>> unedited, because a registration that is quietly rewritten after the data is not a
>>> registration (CORRECTIONS 41).

R1  REPRODUCTION.  On `probes_ns5/probe_w_m5p4_s*` (the cell the debt names), A must
    reproduce CORRECTIONS 74's published **8.268** and B must reproduce 55.4's published
    **6.29**, each to +-0.01.  Without this the module is reconciling two numbers that are
    not the published ones.

R2  MECHANISM.  Evaluating B's window-mean over a DEGENERATE terminal window must return
    A's per-dir value EXACTLY (<= 1e-12).  If it does, the two definitions are the same
    functional evaluated at two times and nothing else differs.
    REFUTATION: any dir where the degenerate-window value differs from A's terminal value.
    That would mean a SECOND difference exists (seed pooling, record parsing, a filter)
    and the reconciliation is not what this module says it is.

R3  ORDERING.  Over every cell where both are computable, is A a MONOTONE
    reparameterisation of B?  Scored as Spearman rho and as the count of discordant pairs.
    REGISTERED: rho >= 0.95 with <= 5% discordant pairs => the two definitions RANK cells
    the same and the debt is discharged as "a monotone reparameterisation; quote either,
    named".
    REFUTATION: rho < 0.95 or > 5% discordant => the two orderings genuinely differ, and
    then EVERY published sentence resting on a span ORDERING (55.4's band, N2's threshold)
    must be re-scored under both.

R4  DOES N2's VERDICT DEPEND ON THE DEFINITION?  N2 was REFUTED because the ms=5e-4 cell
    had span > 5 log units while the argmin was still `w`.  Re-score N2's own predicate
    ("weightwise span > 5 log units" matches "argmin == node") on the ns5/ml5 ladder using
    definition B instead of A.
    REGISTERED: if N2 is REFUTED under BOTH definitions, its refutation is
    definition-independent and CORRECTIONS 74(2) stands without further work.
    If the verdict FLIPS under B, CORRECTIONS 74(2) must be withdrawn, and this module
    says so in that case.

R5  DOES 55.4's BAND DEPEND ON THE DEFINITION?  55.4 separated 11 of 11 argmin cells with
    a gap (span <= 6.29 -> `w`; span >= 8.88 -> `node`) using B.  Re-score the same
    separation using A.
    REGISTERED: separation is definition-robust if A also separates the same cells with a
    non-empty gap.  If A does NOT separate them, 55.4's band is a property of the
    steady-half definition and must be written with that qualifier attached.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does NOT adopt one definition over the other on aesthetics, and it does not re-open any
verdict on its own.  R4 and R5 are the only places a published sentence can move, and both
have their moving condition written above, before the data.

Nothing here is a compute request.  Every input is a `probe.jsonl` already on this Mac.

THE RESULT: THERE ARE **TWO** AXES, NOT ONE, AND R1's BAR IS WHAT FOUND THE SECOND
-----------------------------------------------------------------------------------
R1 passed on A (8.2676 vs the published 8.268) and FAILED on B: the steady-half mean of
the WEIGHTWISE arm is 6.6359, not 6.29.  Chasing that 0.35 identifies the missing axis in
c55's own printed docstring, which the debt never named:

    c55_neff_noise.argmin_report: "`span` is the steady-half mean beta_true_max-min in log
    units, AVERAGED OVER THE THREE RUNGS"                     (analysis/c55_neff_noise.py:388)

So FINDINGS 55.4's 6.29 is a mean over {lay, node, w}, while CORRECTIONS 74's 8.268 is
weightwise ALONE.  Measured on the debt cell (`ns5` m5p4), the 2x2 closes to 4 decimals:

    arm set \\ time      steady half (0.5-1.0)      terminal record
    weightwise                6.6359                    8.2676  <- CORRECTIONS 74's number
    mean of 3 rungs           6.2886                    7.5008
                                ^-- FINDINGS 55.4's number (published 6.29)

**The gap is 1.632 log units of TIME and 0.347 of ARM SET, and only the first was ever
named.**  The span rises monotonically through training on both seeds (quintile means
1.00 -> 7.83), which is the sign of the time term; the arm-set term is the ordering
lay < node < w, which is FINDINGS 55.5's lift ordering appearing in a second statistic.

CONSEQUENCE, AND IT IS THE POINT OF PAYING THE DEBT
---------------------------------------------------
A span number is only interpretable with BOTH labels attached.  From here on, write
`span[weightwise, terminal]` or `span[3-rung, steady-half]`; a bare "span" is not a
quantity.  R4 and R5 then ask whether either published VERDICT moves, and the answer is
what unblocks (or does not unblock) C62-A.

USAGE
    python3 analysis/c63_span_reconcile.py --selftest
    python3 analysis/c63_span_reconcile.py --gate       [--root ..]
    python3 analysis/c63_span_reconcile.py --factorial  [--root ..]
    python3 analysis/c63_span_reconcile.py --run        [--root ..]
"""
import argparse
import glob
import json
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c52_boxfree                    # noqa: E402
import c53_score as C53               # noqa: E402
import c55_neff_noise as C55          # noqa: E402


# The published numbers the debt names, copied from FINDINGS 55.4 / CORRECTIONS 74.
DEBT_CELL = "probes_ns5/probe_w_m5p4_s*"
PUB_A = 8.268      # CORRECTIONS 74 (N2): terminal-record span
PUB_B = 6.29       # FINDINGS 55.4:       steady-half mean span

PROBE_ROOTS = ("probes_p5", "probes_cl5", "probes_fz3", "probes_uc5", "probes_uc6",
               "probes_ml5", "probes_ns5", "probes_bl5", "probes_br6", "probes_ff5",
               "probes_fr5", "probes_wc5", "probes_bo6")


# --------------------------------------------------------------------------------------
# The two definitions, both by IMPORT.  Nothing is restated here.
# --------------------------------------------------------------------------------------

def span_A(pattern):
    """c53_score.beta_span: mean over seeds of the span at the LAST record."""
    sp, _tr, n = C53.beta_span(pattern)
    return sp, n


def span_B(d, window=C55.STEADY[1]):
    """c55_neff_noise.beta_span: mean over the steady half of the per-record span."""
    return C55.beta_span(d, window=window)


def span_B_terminal(d):
    """B's own window-mean, over a window containing ONLY the last record.

    This is the R2 probe.  `beta_span` slices R[int(n*lo):int(n*hi)], so lo = (n-1)/n and
    hi = 1.0 select exactly R[-1].  If B-on-a-terminal-window equals A's per-dir value,
    the ONLY difference between the two published definitions is the evaluation window.
    """
    n = len(c52_boxfree.records(d))
    return C55.beta_span(d, window=((n - 1) / n, 1.0))


def span_A_dir(d):
    """A's statistic on ONE dir (glob of an exact path is that path)."""
    sp, n = span_A(d)
    assert n == 1, f"A-on-one-dir matched {n} dirs for {d}"
    return sp


def span_series(d):
    R = c52_boxfree.records(d)
    return np.array([r["beta_true_max"] - r["beta_true_min"] for r in R], dtype=float)


# --------------------------------------------------------------------------------------
# Ordering statistics
# --------------------------------------------------------------------------------------

def _rank(v):
    v = np.asarray(v, dtype=float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=float)
    r[o] = np.arange(len(v), dtype=float)
    # average ties
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = np.mean(r[o[i:j + 1]])
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float(np.dot(ra, ra)) * float(np.dot(rb, rb)))
    return float(np.dot(ra, rb) / den) if den > 0 else float("nan")


def discordant(a, b, eps=1e-12):
    """Fraction of pairs the two statistics order OPPOSITELY.  Ties in either are not
    discordant -- a tie is an absence of an ordering, not a disagreement about one."""
    n = len(a)
    tot = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if abs(da) <= eps or abs(db) <= eps:
                continue
            tot += 1
            if da * db < 0:
                disc += 1
    return (disc / tot if tot else float("nan")), disc, tot


def separates(vals, labels, target):
    """THRESHOLD form: is there a cut with a GAP putting every `target` above every other?

    This is 55.4's FIRST sentence ("every cell with span <= 6.29 gives `w`; every cell
    with span >= 8.88 gives `node`"), which 55.4 restricted to its 20-EPOCH cells.
    """
    tgt = [v for v, l in zip(vals, labels) if l == target]
    oth = [v for v, l in zip(vals, labels) if l != target]
    if not tgt or not oth:
        return False, float("nan"), float("nan")
    return (max(oth) < min(tgt)), float(max(oth)), float(min(tgt))


def band_separates(vals, labels, target):
    """BAND form: does the interval [min(target), max(target)] contain NO other label?

    This is 55.4's ACTUAL conclusion -- "the nodewise minimum is a BAND in adaptation
    extent, bounded above as well as below" -- reached precisely because the 40-epoch
    `br6` cell sits ABOVE every `node` cell and is `w`.  Scoring 55.4 with the threshold
    test alone would test a shape 55.4 itself disavowed; both are reported.
    """
    tgt = [v for v, l in zip(vals, labels) if l == target]
    oth = [v for v, l in zip(vals, labels) if l != target]
    if not tgt or not oth:
        return False, float("nan"), float("nan"), 0
    lo, hi = min(tgt), max(tgt)
    intruders = sum(1 for v in oth if lo <= v <= hi)
    return (intruders == 0), float(lo), float(hi), intruders


# --------------------------------------------------------------------------------------
# Selftests
# --------------------------------------------------------------------------------------

def _write_probe(d, spans):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "probe.jsonl"), "w") as f:
        for s in spans:
            f.write(json.dumps(dict(beta_true_max=float(s), beta_true_min=0.0)) + "\n")


def selftest():
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # ---- G0: BOTH definitions read the SAME records function ------------------------
    ok(C53.records is c52_boxfree.records,
       "G0 c53 must import c52_boxfree.records (it does, line 42) -- otherwise the two "
       "definitions could differ in PARSING as well as in window")
    ok(C55.c52_boxfree.records is c52_boxfree.records,
       "G0 c55 must call the same records()")
    ok(C55.STEADY[1] == (0.5, 1.0), f"G0 steady half moved: {C55.STEADY}")

    with tempfile.TemporaryDirectory() as td:
        # ---- T1: hand-checkable spans -------------------------------------------------
        # 10 records, span rising 1..10.  A = 10.  B = mean(6..10) = 8.
        d0 = os.path.join(td, "probe_w_x_s0")
        _write_probe(d0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ok(abs(span_A_dir(d0) - 10.0) < 1e-12, "T1 A is the LAST record")
        ok(abs(span_B(d0) - 8.0) < 1e-12, "T1 B is the steady-half mean")
        ok(span_B(d0) < span_A_dir(d0), "T1 B < A when the span is rising")

        # ---- T2: R2's mechanism probe, the load-bearing one ---------------------------
        ok(abs(span_B_terminal(d0) - span_A_dir(d0)) < 1e-12,
           "T2 B on a terminal window MUST equal A exactly -- this is the reconciliation")

        # ---- T3: and it must be able to FAIL -----------------------------------------
        # A FALLING span makes B > A, so the direction of the inequality is data, not
        # arithmetic.  If B were always below A the reconciliation would be vacuous.
        d1 = os.path.join(td, "probe_w_x_s1")
        _write_probe(d1, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        ok(abs(span_A_dir(d1) - 1.0) < 1e-12, "T3 A on a falling series")
        ok(abs(span_B(d1) - 3.0) < 1e-12, "T3 B on a falling series = mean(5..1)")
        ok(span_B(d1) > span_A_dir(d1), "T3 B > A is REACHABLE -- B<=A is not a theorem")
        ok(abs(span_B_terminal(d1) - span_A_dir(d1)) < 1e-12, "T3 R2 probe still exact")

        # ---- T4: A pools seeds, B does not (the second real difference) ---------------
        sp, k = span_A(os.path.join(td, "probe_w_x_s*"))
        ok(k == 2, "T4 A's glob sees both seeds")
        ok(abs(sp - 5.5) < 1e-12, "T4 A pools seeds by MEAN (10 and 1 -> 5.5)")

        # ---- T5: constant span -> the two definitions coincide -------------------------
        d2 = os.path.join(td, "probe_w_y_s0")
        _write_probe(d2, [4.0] * 7)
        ok(abs(span_A_dir(d2) - span_B(d2)) < 1e-12,
           "T5 a stationary span makes the definitions identical -- so any gap between "
           "the two published numbers IS the growth of the span during training")

        # ---- T6: odd/even record counts, the window index arithmetic -------------------
        d3 = os.path.join(td, "probe_w_z_s0")
        _write_probe(d3, [1, 2, 3])          # int(3*0.5) = 1 -> R[1:3] = 2,3 -> 2.5
        ok(abs(span_B(d3) - 2.5) < 1e-12, "T6 odd count window")
        ok(abs(span_B_terminal(d3) - 3.0) < 1e-12, "T6 terminal window on odd count")
        d4 = os.path.join(td, "probe_w_z_s1")
        _write_probe(d4, [5])                # single record: both are 5
        ok(abs(span_A_dir(d4) - 5.0) < 1e-12 and abs(span_B(d4) - 5.0) < 1e-12,
           "T6 single record: definitions coincide trivially")

        # ---- T7: empty probe still fails loudly (c52's own hardening, not re-broken) ---
        d5 = os.path.join(td, "probe_w_z_s2")
        os.makedirs(d5)
        open(os.path.join(d5, "probe.jsonl"), "w").close()
        try:
            span_A_dir(d5)
            ok(False, "T7 empty probe must raise")
        except ValueError:
            ok(True, "T7 empty probe raises rather than returning 0")

    # ---- T8: Spearman ------------------------------------------------------------------
    ok(abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12, "T8 identity")
    ok(abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-12, "T8 monotone rescale")
    ok(abs(spearman([1, 2, 3, 4], [1, 4, 9, 16]) - 1.0) < 1e-12, "T8 monotone nonlinear")
    ok(abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12, "T8 reversal")
    ok(abs(spearman([1, 2, 3, 4], [1, 3, 2, 4]) - 0.8) < 1e-9, "T8 one swap")
    ok(spearman([1, 2], [2, 1]) != spearman([1, 2], [2, 1]), "T8 n<3 -> nan")
    ok(abs(spearman([1, 1, 2, 2], [1, 1, 2, 2]) - 1.0) < 1e-12, "T8 ties handled")

    # ---- T9: discordant pairs ----------------------------------------------------------
    f, d, t = discordant([1, 2, 3], [1, 2, 3])
    ok(f == 0.0 and d == 0 and t == 3, "T9 concordant")
    f, d, t = discordant([1, 2, 3], [3, 2, 1])
    ok(f == 1.0 and d == 3 and t == 3, "T9 fully discordant")
    f, d, t = discordant([1, 2, 3], [1, 3, 2])
    ok(d == 1 and t == 3, "T9 one discordant pair")
    f, d, t = discordant([1, 1, 2], [5, 9, 9])
    ok(t == 1, "T9 ties in EITHER statistic are excluded from the denominator")

    # ---- T10: the separation test ------------------------------------------------------
    okk, hi, lo = separates([1, 2, 3, 9, 10], ["w", "w", "w", "n", "n"], "n")
    ok(okk and hi == 3 and lo == 9, "T10 clean separation with a gap")
    okk, _, _ = separates([1, 2, 9, 3, 10], ["w", "w", "n", "w", "n"], "n")
    ok(okk, "T10 order of the list does not matter")
    okk, _, _ = separates([1, 9, 3, 10], ["w", "n", "n", "w"], "n")
    ok(not okk, "T10 interleaved -> no separation")
    okk, _, _ = separates([1, 2], ["w", "w"], "n")
    ok(not okk, "T10 no target cells -> not separated (never a vacuous pass)")

    # ---- T11: the BAND form, which is 55.4's ACTUAL claim ------------------------------
    # A cell ABOVE every target must break the THRESHOLD but not the BAND.  That is
    # exactly the 40-epoch `br6` cell, and testing 55.4 with the threshold shape alone
    # would score it against a claim 55.4 explicitly disavowed.
    v = [1.0, 2.0, 9.0, 10.0, 18.0]
    l = ["w", "w", "n", "n", "w"]
    okk, _, _ = separates(v, l, "n")
    ok(not okk, "T11 a HIGH non-target breaks the THRESHOLD form")
    okk, lo, hi, intr = band_separates(v, l, "n")
    ok(okk and lo == 9.0 and hi == 10.0 and intr == 0,
       "T11 ...and does NOT break the BAND form -- this is the 55.4 / br6 shape")
    okk, _, _, intr = band_separates([1.0, 9.0, 9.5, 10.0], ["w", "n", "w", "n"], "n")
    ok((not okk) and intr == 1, "T11 an INTERIOR intruder does break the band")
    okk, _, _, intr = band_separates([1.0, 2.0], ["w", "w"], "n")
    ok(not okk, "T11 no target cells -> band not clean (never a vacuous pass)")
    okk, lo, hi, intr = band_separates([5.0, 1.0, 9.0], ["n", "w", "w"], "n")
    ok(okk and lo == hi == 5.0, "T11 a single target cell gives a degenerate band")

    print(f"  selftest: {n}/{n} PASS")


# --------------------------------------------------------------------------------------
# R1 / R2: the debt cell, and the mechanism
# --------------------------------------------------------------------------------------

def gate(root):
    print("=" * 96)
    print("R1 / R2 -- REPRODUCE THE TWO PUBLISHED NUMBERS, THEN SHOW THEY ARE ONE")
    print("=" * 96)
    pat = os.path.join(root, DEBT_CELL)
    ds = sorted(glob.glob(pat))
    print(f"debt cell {DEBT_CELL}: {len(ds)} seed dirs")
    if not ds:
        raise SystemExit("REFUSING: the debt cell is not on this disk.")

    A, nA = span_A(pat)
    B = float(np.mean([span_B(d) for d in ds]))
    r1a, r1b = abs(A - PUB_A) <= 0.01, abs(B - PUB_B) <= 0.01
    print(f"\n  A (c53, terminal record, weightwise, {nA} seeds) = {A:.4f}   "
          f"published {PUB_A}   delta {A - PUB_A:+.4f}   {'PASS' if r1a else 'FAIL'}")
    print(f"  B (c55, steady-half mean, weightwise)           = {B:.4f}   "
          f"published {PUB_B}   delta {B - PUB_B:+.4f}   {'PASS' if r1b else 'FAIL'}")
    if not r1b:
        print("\n  R1's B-ARM FAILS, AND THAT FAILURE IS THIS TICK'S FINDING.")
        print("  55.4's 6.29 is NOT the weightwise steady-half span.  c55's own printed")
        print("  docstring (c55_neff_noise.py:388) says the number is 'averaged over the")
        print("  three rungs'.  The debt named ONE axis (time); there are TWO.  See")
        print("  --factorial, which closes the 2x2 to 4 decimals.")
    r1 = r1a and r1b

    print("\n  R2 -- B evaluated on a DEGENERATE TERMINAL WINDOW, per dir:")
    print(f"  {'dir':<34}{'A(term)':>10}{'B(term-win)':>13}{'|diff|':>11}"
          f"{'B(steady)':>11}{'n_rec':>8}")
    worst = 0.0
    for d in ds:
        a = span_A_dir(d)
        bt = span_B_terminal(d)
        bs = span_B(d)
        nr = len(c52_boxfree.records(d))
        worst = max(worst, abs(a - bt))
        print(f"  {os.path.basename(d):<34}{a:>10.4f}{bt:>13.4f}{abs(a - bt):>11.2e}"
              f"{bs:>11.4f}{nr:>8d}")
    r2 = worst <= 1e-12
    print(f"\n  worst |A - B(terminal window)| = {worst:.3e}   "
          f"{'PASS' if r2 else 'FAIL'}")
    print("  => the two published definitions are ONE functional at TWO evaluation times."
          if r2 else "  => a SECOND difference exists; the reconciliation below is void.")

    # the time course that explains the gap, on the debt cell
    print("\n  SPAN TIME COURSE on the debt cell (quintiles of the run, log units):")
    print(f"  {'dir':<34}" + "".join(f"{f'q{i+1}':>9}" for i in range(5))
          + f"{'monotone?':>11}")
    for d in ds:
        s = span_series(d)
        q = [float(np.mean(c)) for c in np.array_split(s, 5)]
        mono = all(q[i] <= q[i + 1] + 1e-12 for i in range(4))
        print(f"  {os.path.basename(d):<34}" + "".join(f"{x:>9.3f}" for x in q)
              + f"{('rising' if mono else 'NOT'):>11}")
    return dict(r1=r1, r1_A=r1a, r1_B=r1b, r2=r2, A=A, B=B, worst=worst)


# --------------------------------------------------------------------------------------
# R3 / R4 / R5
# --------------------------------------------------------------------------------------

def collect(root):
    """Per-dir (A, B) for every probe dir on disk that has a probe.jsonl, de-duplicated
    by realpath (CORRECTIONS 88.11: probes_ml5_m{2,3,4} are symlinks onto probes_ml5)."""
    rows, seen = [], set()
    for pr in PROBE_ROOTS:
        for d in sorted(glob.glob(os.path.join(root, pr, "probe_*"))):
            if not os.path.isdir(d):
                continue
            rp = os.path.realpath(d)
            if rp in seen:
                continue
            seen.add(rp)
            if not os.path.exists(os.path.join(d, "probe.jsonl")):
                continue
            try:
                a, b = span_A_dir(d), span_B(d)
            except Exception:
                continue
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            rows.append(dict(dir=os.path.relpath(d, root), root=pr, A=a, B=b,
                             n_rec=len(c52_boxfree.records(d))))
    return rows


def r3(rows, min_rows=40):
    print("\n" + "=" * 96)
    print("R3 -- DO THE TWO DEFINITIONS RANK CELLS THE SAME?")
    print("=" * 96)
    if len(rows) < min_rows:
        raise SystemExit(f"REFUSING TO SCORE R3: {len(rows)} dirs < {min_rows}.")
    # frozen arms are 0.00 under BOTH by construction (HF.py:477); they are ties and
    # carry no ordering, so they are reported and then excluded from the ordering test.
    frozen = [r for r in rows if r["A"] <= 1e-9 and r["B"] <= 1e-9]
    free = [r for r in rows if r not in frozen]
    print(f"  {len(rows)} probe dirs; {len(frozen)} are frozen (span 0.00 under BOTH, by "
          f"construction) and carry no ordering; {len(free)} scored.")
    A = [r["A"] for r in free]
    B = [r["B"] for r in free]
    rho = spearman(A, B)
    frac, disc, tot = discordant(A, B)
    ratio = [a / b for a, b in zip(A, B) if b > 1e-9]
    print(f"  Spearman rho(A, B) = {rho:.6f}   over n={len(free)}")
    print(f"  discordant pairs   = {disc}/{tot} = {100*frac:.3f}%")
    print(f"  A/B ratio          = {min(ratio):.3f} - {max(ratio):.3f}, "
          f"median {float(np.median(ratio)):.3f}")
    print(f"  A >= B in          = {sum(1 for a, b in zip(A, B) if a >= b - 1e-12)}"
          f"/{len(free)} dirs")
    v = "MONOTONE REPARAMETERISATION" if (rho >= 0.95 and frac <= 0.05) else "ORDERINGS DIFFER"
    print(f"  --> R3 {v} (registered: rho >= 0.95 and <= 5% discordant)")
    return dict(rho=rho, frac=frac, disc=disc, tot=tot, n=len(free), verdict=v)


# The four rung-resolved cells of the ns5/ml5 ladder N2 was scored on.  `{r}` is the rung.
LADDER_R = (("1e-4", "probes_ml5", "probe_{r}_m4_s*"),
            ("2e-4", "probes_ns5", "probe_{r}_m2p4_s*"),
            ("5e-4", "probes_ns5", "probe_{r}_m5p4_s*"),
            ("1e-3", "probes_cl5", "probe_{r}_cU_s*"))
RUNGS = ("lay", "node", "w")


def cell_span(root, pr, pat, rung, mode):
    """One (ms, rung) cell under one time convention.  mode in {'term','steady'}."""
    ds = sorted(glob.glob(os.path.join(root, pr, pat.format(r=rung))))
    if not ds:
        return float("nan"), 0
    if mode == "term":
        return float(np.mean([span_A_dir(d) for d in ds])), len(ds)
    return float(np.mean([span_B(d) for d in ds])), len(ds)


def factorial(root):
    """The 2x2 that the debt never named: {weightwise, 3-rung mean} x {steady, terminal}."""
    print("\n" + "=" * 96)
    print("THE 2x2 -- ARM SET x EVALUATION TIME, ON THE ns5/ml5 LADDER")
    print("`w/term` is CORRECTIONS 74's definition.  `3rung/steady` is FINDINGS 55.4's.")
    print("=" * 96)
    print(f"  {'ms':>7}{'w/steady':>11}{'w/term':>10}{'3rung/steady':>15}"
          f"{'3rung/term':>13}{'d_time(w)':>11}{'d_armset':>11}")
    out = []
    for ms, pr, pat in LADDER_R:
        per = {}
        for r in RUNGS:
            for m in ("steady", "term"):
                per[(r, m)] = cell_span(root, pr, pat, r, m)[0]
        ws, wt = per[("w", "steady")], per[("w", "term")]
        gs = float(np.nanmean([per[(r, "steady")] for r in RUNGS]))
        gt = float(np.nanmean([per[(r, "term")] for r in RUNGS]))
        print(f"  {ms:>7}{ws:>11.4f}{wt:>10.4f}{gs:>15.4f}{gt:>13.4f}"
              f"{wt - ws:>11.4f}{ws - gs:>11.4f}")
        out.append(dict(ms=ms, w_steady=ws, w_term=wt, g_steady=gs, g_term=gt,
                        per={f"{r}/{m}": v for (r, m), v in per.items()}))
    print("\n  per-rung, steady half (the arm-set term is this ordering):")
    print(f"  {'ms':>7}" + "".join(f"{r:>10}" for r in RUNGS) + f"{'lay<node<w?':>13}")
    for row in out:
        v = [row["per"][f"{r}/steady"] for r in RUNGS]
        mono = all(v[i] < v[i + 1] for i in range(len(v) - 1))
        print(f"  {row['ms']:>7}" + "".join(f"{x:>10.4f}" for x in v)
              + f"{('YES' if mono else 'no'):>13}")
    return out


# N2's registered argmins on this ladder, DATA copied from FINDINGS 53.9 / CORRECTIONS 74
# and from 55.3's re-scoring -- not recomputed here, because R4 asks whether the SPAN
# definition moves N2's verdict, holding the argmins fixed at what N2 itself used.
N2_ARGMIN = {"1e-4": "w", "2e-4": "w", "5e-4": "w", "1e-3": "node"}
N2_THRESHOLD = 5.0


def r4(root, fac):
    print("\n" + "=" * 96)
    print("R4 -- N2's OWN PREDICATE, RE-SCORED UNDER ALL FOUR DEFINITIONS")
    print("N2 (CORRECTIONS 74.2): 'the argmin flips to `node` at the stepsize whose")
    print("WEIGHTWISE beta span first exceeds ~5 log units.'  The 5.0 threshold and the")
    print("argmin column are N2's own and are NOT re-tuned or recomputed here.")
    print("=" * 96)
    keys = (("w_steady", "w/steady"), ("w_term", "w/term"),
            ("g_steady", "3rung/steady"), ("g_term", "3rung/term"))
    print(f"  {'ms':>7}{'argmin':>8}" + "".join(f"{lab:>16}" for _, lab in keys))
    for row in fac:
        am = N2_ARGMIN.get(row["ms"], "?")
        cells = "".join(
            f"{row[k]:>10.3f}{('>5' if row[k] > N2_THRESHOLD else '<5'):>6}"
            for k, _ in keys)
        print(f"  {row['ms']:>7}{am:>8}{cells}")
    print()
    verdicts = {}
    for k, lab in keys:
        hits = [(row["ms"], row[k] > N2_THRESHOLD, N2_ARGMIN.get(row["ms"]) == "node")
                for row in fac]
        agree = sum(1 for _, a, b in hits if a == b)
        v = "CONFIRMED" if agree == len(hits) else "REFUTED"
        verdicts[lab] = v
        print(f"  N2 under span[{lab:<13}] : predicate matches argmin on "
              f"{agree}/{len(hits)} rungs  ->  {v}")
    same = len(set(verdicts.values())) == 1
    print(f"\n  --> R4: N2's verdict is "
          f"{'DEFINITION-INDEPENDENT' if same else 'DEFINITION-DEPENDENT'} "
          f"({'all four agree: ' + next(iter(verdicts.values())) if same else verdicts})")
    if same and next(iter(verdicts.values())) == "REFUTED":
        print("      CORRECTIONS 74(2) STANDS as written.  The debt does not touch it.")
    return verdicts


def r5(root):
    """55.4's band, re-scored under all four definitions.

    55.4 sorted the argmin cells by span and found a clean separation with a gap.  The
    argmins and their interpretability come from c55's own `argmin_in_se` + CORRECTIONS
    79 gate, called here rather than restated; only the SPAN column is varied.
    """
    print("\n" + "=" * 96)
    print("R5 -- FINDINGS 55.4's BAND, RE-SCORED UNDER ALL FOUR DEFINITIONS")
    print("55.4: 'every cell with span <= 6.29 gives `w`; every cell with span >= 8.88")
    print("gives `node`.  11 of 11.'  The argmin column is c55's, via argmin_in_se and")
    print("CORRECTIONS 79's interpretability gate.  Only the SPAN column is varied.")
    print("=" * 96)
    roots = sorted(r for r in glob.glob(os.path.join(root, "probes_*"))
                   if os.path.isdir(r) and C55.root_tag(r) not in C55.ALIAS_ROOTS)
    rows = []
    for pr in roots:
        tag = C55.root_tag(pr)
        try:
            cells = C55.sweep(pr)
        except Exception:
            continue
        for fam in sorted({f for (f, _) in cells}):
            byrung, dirs = {}, {}
            for (f, rung), rr in cells.items():
                if f != fam or rung not in C55.HEADLINE_RUNGS:
                    continue
                st = C55.cell_stats([x["neff_m"] for x in rr])
                st["nbf"] = sum(1 for x in rr if x["boxfree"])
                st["ep"] = rr[0]["epochs"]
                byrung[rung] = st
                dirs[rung] = [x["dir"] for x in rr]
            if not byrung:
                continue
            v = C55.argmin_in_se(byrung)
            if v["status"] != "DECIDED":
                continue
            sp = {}
            for mode in ("steady", "term"):
                per = {}
                for rung, dd in dirs.items():
                    f = span_B if mode == "steady" else span_A_dir
                    per[rung] = float(np.mean([f(x) for x in dd]))
                sp[f"w_{mode}"] = per.get("w", float("nan"))
                sp[f"g_{mode}"] = float(np.nanmean(list(per.values())))
            ep = float(np.nanmean([byrung[r]["ep"] for r in byrung
                                   if np.isfinite(byrung[r].get("ep", float("nan")))])) \
                if any(np.isfinite(byrung[r].get("ep", float("nan"))) for r in byrung) \
                else float("nan")
            rows.append(dict(root=tag, fam=fam, argmin=v["argmin"], ep=ep,
                             gap_se=v["gap_se"], **sp))
    if len(rows) < 8:
        raise SystemExit(f"REFUSING TO SCORE R5: only {len(rows)} DECIDED cells.")
    keys = (("w_steady", "w/steady"), ("w_term", "w/term"),
            ("g_steady", "3rung/steady"), ("g_term", "3rung/term"))
    print(f"  {'root':>7}{'arm':>10}{'ep':>5}{'argmin':>7}{'gap/SE':>8}"
          + "".join(f"{lab:>14}" for _, lab in keys))
    for r in sorted(rows, key=lambda x: x["g_steady"]):
        print(f"  {r['root']:>7}{r['fam']:>10}{r['ep']:>5.0f}{r['argmin']:>7}"
              f"{r['gap_se']:>8.2f}" + "".join(f"{r[k]:>14.3f}" for k, _ in keys))
    print(f"\n  {len(rows)} DECIDED cells "
          f"({sum(1 for r in rows if r['argmin']=='node')} node, "
          f"{sum(1 for r in rows if r['argmin']=='w')} w, "
          f"{sum(1 for r in rows if r['argmin']=='lay')} lay)")

    sub20 = [r for r in rows if r["ep"] <= 20.5]
    print(f"  of which {len(sub20)} are 20-epoch cells -- the restriction 55.4's THRESHOLD "
          f"sentence carries.\n")
    out = {}
    for scope, rr in (("20ep only (55.4's own scope)", sub20), ("ALL decided", rows)):
        print(f"  --- THRESHOLD form, {scope} ---")
        lab_l = [x["argmin"] for x in rr]
        for k, lab in keys:
            okk, hi, lo = separates([x[k] for x in rr], lab_l, "node")
            out[(scope, "thr", lab)] = okk
            print(f"    span[{lab:<13}] {'SEPARATES' if okk else 'no       '}   "
                  f"max(other)={hi:7.3f}  min(node)={lo:7.3f}  gap={lo - hi:+7.3f}")
    print(f"\n  --- BAND form (55.4's actual conclusion), ALL decided cells ---")
    lab_l = [x["argmin"] for x in rows]
    for k, lab in keys:
        okk, lo, hi, intr = band_separates([x[k] for x in rows], lab_l, "node")
        out[("ALL", "band", lab)] = okk
        print(f"    span[{lab:<13}] band=[{lo:7.3f}, {hi:7.3f}]  "
              f"{'CLEAN' if okk else f'{intr} intruder(s)'}")
    thr20 = {v for kk, v in out.items() if kk[0].startswith("20ep") and kk[1] == "thr"}
    band = {v for kk, v in out.items() if kk[1] == "band"}
    print(f"\n  --> R5: within 55.4's own 20-epoch scope the THRESHOLD "
          f"{'holds under all four definitions' if thr20 == {True} else f'is DEFINITION-DEPENDENT {thr20}'}.")
    print(f"      The BAND form {'holds under all four' if band == {True} else f'is DEFINITION-DEPENDENT {band}'}.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--factorial", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.factorial and not a.run:
        factorial(a.root)
    if a.gate or a.run:
        g = gate(a.root)
        if a.run:
            # R2 is the gate that must hold: it is what makes the 2x2 the WHOLE story.
            # R1's B-arm failing is this tick's finding, not a reason to stop -- but the
            # A-arm must reproduce, or we are not reconciling the published numbers.
            if not g["r2"]:
                raise SystemExit("\nR2 FAILED: a third difference exists. R3-R5 NOT scored.")
            if not g["r1_A"]:
                raise SystemExit("\nR1's A-arm did not reproduce CORRECTIONS 74's number.")
            fac = factorial(a.root)
            rows = collect(a.root)
            r3(rows)
            r4(a.root, fac)
            r5(a.root)


if __name__ == "__main__":
    main()
