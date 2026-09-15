#!/usr/bin/env python3
r"""c63_c62a_travel.py -- C62-A: DOES F_col/F_row TRACK MEASURED BETA TRAVEL, OR JUST ms?

WHY THIS EXISTS
---------------
FINDINGS 62.7 measured a monotone dose-response in `F_col/F_row` along the R18/CIFAR-10
meta-stepsize ladder (0.700-0.753 frozen < 0.825-0.887 at ms=1e-4 < ... < 1.411-1.578 at
ms=1e-2; whole-corpus Mann-Whitney U=542/544, z=+5.62).  CORRECTIONS 91.8 labelled it
POST-HOC and registered its confirmation test as C62-A, verbatim:

    "C62-A.  If 62.7 is adaptation and not the hyperparameter, `F_col/F_row` must track
     measured beta TRAVEL per arm, not `ms`, and must do so ACROSS families -- the
     r10/r34/c100 arms are available and were not used in the matched ladder.  BLOCKED
     until CORRECTIONS 55.4's owed reconciliation of the two `span` definitions ... is
     discharged.  Registering it without that would repeat the CORRECTIONS 41 pathology."

**That block is discharged this tick (FINDINGS 63.2-63.4).**  So C62-A may now be
registered -- and FINDINGS 63.3 forces it to name a span definition rather than leave one
implicit, because the two published definitions disagree on 6.4% of arm pairs.

THE SPAN DEFINITION IS DERIVED, NOT CHOSEN (FINDINGS 63.5)
-----------------------------------------------------------
* **arm set = weightwise.**  `F_col`/`F_row` are computed from a weightwise arm's
  `neg_counts.npy`.  A 3-rung mean would average in two arms whose coordinates the
  statistic never touches.
* **window = the FULL RUN.**  `p_i = neg_count_i / n_records` is accumulated over EVERY
  record (PATCH_PROBE5 never resets), so the adaptation extent that produced the F
  statistics is the full-run mean span.  Steady-half and terminal both mismatch the
  accumulation window of the quantity being explained.

Primary = `span[weightwise, full-run mean]`.  The other three cells of 63.2's 2x2 are
carried as a sensitivity column and a disagreement between them is REPORTED, not resolved.

PRE-REGISTERED, WRITTEN AND COMMITTED BEFORE ANY ARM WAS SCORED (cycle 63)
---------------------------------------------------------------------------
Clean weightwise arms only (`c62.is_exception` == `c60.EXC_W`, which is FINDINGS 59.4's
set).  The ms=1e-2 and AdamW-base arms stay unquotable (CORRECTIONS 62) and are printed
separately.

A0  **FROZEN ARMS ARE AN ANCHOR, NOT DATA.**  A `--alg-meta fixed` arm has span EXACTLY
    0.0 by construction (HF.py:477).  Including them in a correlation would make any
    zero-versus-nonzero contrast drive the whole statistic.  **They are excluded from A1,
    A2 and A2b and reported as an anchor row.**  Registered here so the exclusion cannot
    be read as chosen after seeing the numbers.

A1  **DOES IT TRACK TRAVEL AT ALL?**  Over clean FREE weightwise arms, Spearman
    rho(span, F_col/F_row) >= +0.50 with a permutation p < 0.01 (10,000 shuffles).
    REFUTATION: rho <= 0.

A2  **PARTIAL, CONTROLLING FOR ms.**  Over the clean free arms whose meta-stepsize is
    verifiable from its submitting script (see MS_MAP), partial Spearman of
    (F_col/F_row, span) given log10(ms) >= +0.40.

A2b **THE CONFOUND-FREE FORM, AND THE ONE THAT DECIDES.**  62.7's ladder varies ms and
    span together, so it cannot separate them.  Restricted to the **ms = 1e-3 stratum
    alone** -- where ms is literally constant and the four architectures / two datasets
    supply the span variation -- Spearman(span, F_col/F_row) >= +0.40.
    REFUTATION: rho <= 0 within the stratum while A1 is strongly positive.  That pattern
    means the driver is the HYPERPARAMETER, 62.7's "adaptation extent" reading is
    REFUTED, and 62.7 must be rewritten as an ms effect.
    A2b is the test C62-A actually asked for ("across families").

A3  **SENSITIVITY.**  A1/A2b are re-scored under all four of FINDINGS 63.2's span
    definitions plus the primary full-run one.  If the verdict flips across them, that is
    reported as the result and no single definition is promoted.

62.7 was POST-HOC and CORRECTIONS 76(1)/79 read symmetrically forbid it overturning a
registered gate.  **Nothing here can overturn a gate either**; A1/A2b can only confirm or
refute an INTERPRETATION of a post-hoc trend.  Said in advance so a positive result is not
later promoted past its class.

SCOPE.  3x3 conv tensors only (1x1 shortcut convs have no kernel).  We measure `z`, the
META-gradient; Adam-mini argues about `G`.  **No document may write "we refuted Adam-mini."**

USAGE
    python3 analysis/c63_c62a_travel.py --selftest
    python3 analysis/c63_c62a_travel.py --run   [--root ..]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c55_neff_noise as C55           # noqa: E402
import c60_exception_mechanism as C60  # noqa: E402
import c62_blocksize_curve as C62      # noqa: E402
import c63_span_reconcile as C63S      # noqa: E402


# --------------------------------------------------------------------------------------
# The meta-stepsize map.  EVERY entry cites the submitting script and line that sets it.
# A family whose ms could not be verified from its own script is ABSENT and is excluded
# from A2/A2b rather than guessed -- that is the whole point of CORRECTIONS 55.4's lesson.
# --------------------------------------------------------------------------------------
MS_MAP = {
    # (probe root, substring of the dir name) -> meta-stepsize
    ("probes_ml5", "_m4_"): 1e-4,    # bin/c49_ms_ladder_p5.sh:119   1e-4:m4
    ("probes_ml5", "_m3_"): 1e-3,    # bin/c49_ms_ladder_p5.sh:119   1e-3:m3
    ("probes_ml5", "_m2_"): 1e-2,    # bin/c49_ms_ladder_p5.sh:119   1e-2:m2  (EXCEPTION set)
    ("probes_ns5", "_m2p4_"): 2e-4,  # bin/c53_ns_thirdseed.sh:114   MST=2e-4
    ("probes_ns5", "_m5p4_"): 5e-4,  # ns5 naming, same ladder
    ("probes_cl5", ""): 1e-3,        # bin/c51_ceiling_ladder.sh:219 --meta-stepsize 1e-3
    ("probes_bl5", ""): 1e-3,        # bin/c52_budget_ladder.sh:266  --meta-stepsize 1e-3
    ("probes_ff5", ""): 1e-3,        # bin/c49_free_family_ladder.sh:181
    ("probes_uc5", ""): 1e-3,        # bin/c51_unclipped_family.sh:192
}
FROZEN_ROOTS = ("probes_fz3", "probes_p5")   # --alg-meta fixed: span == 0 by construction


def ms_for(rel):
    root = rel.split("/")[0]
    base = "/" + os.path.basename(rel) + "_"
    best = None
    for (r, sub), v in MS_MAP.items():
        if r != root:
            continue
        if sub == "":
            best = v if best is None else best
        elif sub in base:
            return v
    return best


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def spearman(a, b):
    return C63S.spearman(a, b)


def perm_p(a, b, n=10000, seed=12345):
    """Two-sided permutation p for Spearman.  Deterministic seed: this is a registered
    test, so the p must be reproducible bit-for-bit."""
    if len(a) < 4:
        return float("nan")
    obs = abs(spearman(a, b))
    rng = np.random.default_rng(seed)
    b = np.asarray(b, dtype=float)
    hits = 0
    for _ in range(n):
        if abs(spearman(a, rng.permutation(b))) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def partial_spearman(x, y, z):
    """Spearman of x,y given z: Pearson of the residuals of the RANKS."""
    if len(x) < 4:
        return float("nan")
    rx, ry, rz = C63S._rank(x), C63S._rank(y), C63S._rank(z)

    def resid(v, u):
        u = u - u.mean()
        d = float(np.dot(u, u))
        if d <= 0:
            return v - v.mean()
        return (v - v.mean()) - u * (float(np.dot(v - v.mean(), u)) / d)

    ex, ey = resid(rx, rz), resid(ry, rz)
    den = math.sqrt(float(np.dot(ex, ex)) * float(np.dot(ey, ey)))
    return float(np.dot(ex, ey) / den) if den > 0 else float("nan")


# --------------------------------------------------------------------------------------
# Per-arm measurement
# --------------------------------------------------------------------------------------

def spans_for(d):
    """All five span variants for one weightwise dir."""
    return dict(
        full=C55.beta_span(d, window=(0.0, 1.0)),
        steady=C55.beta_span(d, window=(0.5, 1.0)),
        term=C63S.span_A_dir(d),
        q4=C55.beta_span(d, window=(0.75, 1.0)),
    )


def measure(root):
    out = []
    for d in sorted(C60.arms(root, "weightwise")):
        rel = os.path.relpath(os.path.realpath(d), os.path.realpath(root))
        L = C62.load_3x3(d)
        if L is None:
            continue
        p3, s3, n_rec, fam, _meta = L
        S = C62.filter_interaction(p3, s3)
        try:
            sp = spans_for(d)
        except Exception:
            continue
        lab = C60.arm_label(d)
        out.append(dict(
            arm=rel, label=lab, fam=fam, n_rec=n_rec,
            exc=C62.is_exception(lab),
            frozen=rel.split("/")[0] in FROZEN_ROOTS,
            ms=ms_for(rel),
            F_row=S["row"]["F"], F_col=S["col"]["F"], F_int=S["inter"]["F"],
            ratio=(S["col"]["F"] / S["row"]["F"]) if S["row"]["F"] > 0 else float("nan"),
            **{f"span_{k}": v for k, v in sp.items()}))
    return out


SPAN_KEYS = (("span_full", "full-run (PRIMARY)"), ("span_steady", "steady-half"),
             ("span_term", "terminal"), ("span_q4", "last quarter"))


def report(root, out_json=None):
    rows = measure(root)
    clean = [r for r in rows if not r["exc"]]
    exc = [r for r in rows if r["exc"]]
    frozen = [r for r in clean if r["frozen"]]
    free = [r for r in clean if not r["frozen"]]

    print("=" * 118)
    print("C62-A  --  DOES F_col/F_row TRACK MEASURED BETA TRAVEL, OR THE HYPERPARAMETER?")
    print("span[weightwise, full-run mean] is PRIMARY and is DERIVED (FINDINGS 63.5).")
    print("=" * 118)
    print(f"{'arm':<32}{'fam':>6}{'ms':>8}{'span_full':>11}{'span_std':>10}"
          f"{'F_row':>10}{'F_col':>10}{'F_col/F_row':>12}")
    for r in sorted(clean, key=lambda x: (x["frozen"] is False, x["span_full"])):
        msl = f"{r['ms']:.0e}" if r["ms"] else "  -"
        print(f"{r['arm']:<32}{r['fam']:>6}{msl:>8}{r['span_full']:>11.4f}"
              f"{r['span_steady']:>10.4f}{r['F_row']:>10.3f}{r['F_col']:>10.3f}"
              f"{r['ratio']:>12.4f}")
    if exc:
        print("\nEXCEPTION ARMS (FINDINGS 59.4 set; ms=1e-2 / AdamW-base). NEVER POOLED:")
        for r in sorted(exc, key=lambda x: x["span_full"]):
            print(f"{r['arm']:<32}{r['fam']:>6}{'':>8}{r['span_full']:>11.4f}"
                  f"{r['span_steady']:>10.4f}{r['F_row']:>10.3f}{r['F_col']:>10.3f}"
                  f"{r['ratio']:>12.4f}")

    print(f"\n--- A0 FROZEN ANCHOR (span == 0.00 by construction; EXCLUDED from A1/A2/A2b) ---")
    if frozen:
        v = [r["ratio"] for r in frozen]
        print(f"  n={len(frozen)}  F_col/F_row = {min(v):.4f} - {max(v):.4f}, "
              f"median {float(np.median(v)):.4f}")
        mx = max(v)
        below = sum(1 for r in free if r["ratio"] <= mx)
        print(f"  free arms at or below the frozen MAXIMUM: {below}/{len(free)}")
    print(f"  {len(free)} clean free arms scored; {len(exc)} exception arms held out.")

    if len(free) < 15:
        raise SystemExit(f"REFUSING TO SCORE: only {len(free)} clean free arms.")

    print("\n--- A1  Spearman(span, F_col/F_row) over clean FREE arms ---")
    a1 = {}
    for k, lab in SPAN_KEYS:
        x = [r[k] for r in free]
        y = [r["ratio"] for r in free]
        rho = spearman(x, y)
        p = perm_p(x, y)
        a1[lab] = rho
        print(f"  span[{lab:<20}] rho = {rho:+.4f}   perm p = {p:.5f}   n = {len(free)}"
              f"   {'PASS' if (rho >= 0.50 and p < 0.01) else 'fail'}")

    msfree = [r for r in free if r["ms"]]
    print(f"\n--- A2  partial Spearman given log10(ms), n = {len(msfree)} "
          f"(arms with a script-verified ms) ---")
    a2 = {}
    for k, lab in SPAN_KEYS:
        if len(msfree) < 4:
            break
        rho = partial_spearman([r[k] for r in msfree], [r["ratio"] for r in msfree],
                               [math.log10(r["ms"]) for r in msfree])
        a2[lab] = rho
        print(f"  span[{lab:<20}] partial rho = {rho:+.4f}   "
              f"{'PASS' if rho >= 0.40 else 'fail'}")

    strat = [r for r in free if r["ms"] == 1e-3]
    fams = sorted({r["fam"] for r in strat})
    print(f"\n--- A2b  THE DECIDING TEST: ms = 1e-3 STRATUM ALONE, n = {len(strat)}, "
          f"families {fams} ---")
    a2b = {}
    for k, lab in SPAN_KEYS:
        if len(strat) < 5:
            print("  stratum too small to score")
            break
        x = [r[k] for r in strat]
        y = [r["ratio"] for r in strat]
        rho = spearman(x, y)
        p = perm_p(x, y)
        a2b[lab] = rho
        print(f"  span[{lab:<20}] rho = {rho:+.4f}   perm p = {p:.5f}   "
              f"{'PASS' if rho >= 0.40 else 'fail'}")
    if strat:
        print(f"\n  the stratum, sorted by span_full:")
        print(f"  {'arm':<32}{'fam':>6}{'span_full':>11}{'F_col/F_row':>12}")
        for r in sorted(strat, key=lambda x: x["span_full"]):
            print(f"  {r['arm']:<32}{r['fam']:>6}{r['span_full']:>11.4f}"
                  f"{r['ratio']:>12.4f}")

    print("\n--- A3  SENSITIVITY ACROSS SPAN DEFINITIONS ---")
    for name, dd, bar in (("A1", a1, 0.50), ("A2", a2, 0.40), ("A2b", a2b, 0.40)):
        if not dd:
            continue
        vs = list(dd.values())
        agree = len({v >= bar for v in vs}) == 1
        print(f"  {name:>4}: rho {min(vs):+.4f} .. {max(vs):+.4f}   "
              f"verdict {'AGREES' if agree else 'FLIPS'} across the four definitions")

    if out_json:
        with open(out_json, "w") as f:
            json.dump(dict(rows=rows, a1=a1, a2=a2, a2b=a2b), f, indent=1)
        print(f"\nwrote {out_json}")
    return rows, a1, a2, a2b


# --------------------------------------------------------------------------------------
# Selftests
# --------------------------------------------------------------------------------------

def selftest():
    n = 0

    def ok(c, m):
        nonlocal n
        assert c, m
        n += 1

    # ---- ms map ---------------------------------------------------------------------
    ok(ms_for("probes_ml5/probe_w_m4_s0") == 1e-4, "ms ml5 m4")
    ok(ms_for("probes_ml5/probe_w_m3_s0") == 1e-3, "ms ml5 m3")
    ok(ms_for("probes_ml5/probe_w_m2_s0") == 1e-2, "ms ml5 m2")
    ok(ms_for("probes_ns5/probe_w_m2p4_s0") == 2e-4, "ms ns5 m2p4")
    ok(ms_for("probes_ns5/probe_w_m5p4_s1") == 5e-4, "ms ns5 m5p4")
    ok(ms_for("probes_cl5/probe_w_cU_s0") == 1e-3, "ms cl5 (root-wide)")
    ok(ms_for("probes_ff5/probe_r34_w_s1") == 1e-3, "ms ff5")
    ok(ms_for("probes_uc5/probe_r10_w_s0") == 1e-3, "ms uc5")
    ok(ms_for("probes_br6/probe_w_c2_s0") is None,
       "an UNVERIFIED family must return None, never a guess")
    ok(ms_for("probes_uc6/probe_c100_w_s0") is None, "uc6 not verified -> None")
    ok(ms_for("probes_fz3/probe_r10_w_s0") is None, "frozen roots carry no ms")
    # m2p4 must not be captured by an "_m2_" rule
    ok(ms_for("probes_ns5/probe_w_m2p4_s0") != 1e-2, "m2p4 is not m2")

    # ---- spearman / permutation ------------------------------------------------------
    ok(abs(spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-12, "rho identity")
    ok(abs(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-12, "rho reversal")
    p = perm_p([1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8], n=2000)
    ok(p < 0.01, f"perm p on a perfect monotone must be small, got {p}")
    rng = np.random.default_rng(7)
    p2 = perm_p(list(range(12)), list(rng.permutation(12)), n=2000)
    ok(p2 > 0.02, f"perm p on noise must not be small, got {p2}")
    ok(perm_p([1, 2], [1, 2]) != perm_p([1, 2], [1, 2]), "perm p undefined below n=4")
    # determinism: a registered p must reproduce bit-for-bit
    ok(perm_p([1, 3, 2, 4, 6, 5, 8, 7], [1, 2, 3, 4, 5, 6, 7, 8], n=1000)
       == perm_p([1, 3, 2, 4, 6, 5, 8, 7], [1, 2, 3, 4, 5, 6, 7, 8], n=1000),
       "perm p must be deterministic")

    # ---- partial spearman -------------------------------------------------------------
    # x an EXACT copy of the control: the residual of x on z is identically 0, so the
    # partial correlation is 0/0 and must come back UNDEFINED.  Returning 0.0 here would
    # silently assert "no partial association" from a degenerate denominator, which is the
    # failure mode A2 exists to avoid.  (This module's first assertion demanded ~0 and was
    # WRONG; the code was right.  Recorded rather than quietly cut.)
    z = [1.0, 2, 3, 4, 5, 6, 7, 8]
    y = [2.0 * v for v in z]
    pr = partial_spearman(list(z), y, z)
    ok(pr != pr, "partial rho must be UNDEFINED (nan) when x IS the control, never 0.0")
    # The non-degenerate version.  Both x and y are z plus ONE independent rank swap, so
    # after z is removed their residuals share nothing: partial rho must be small AND
    # defined.  (y = 2*z exactly is also degenerate -- ITS residual vanishes too -- which
    # this module's second wrong assertion missed as well.)
    x_ind = [2.0, 1, 3, 4, 5, 6, 7, 8]     # swap at the low end
    y_ind = [1.0, 2, 3, 4, 5, 6, 8, 7]     # swap at the high end
    pr2 = partial_spearman(x_ind, y_ind, z)
    ok(pr2 == pr2 and abs(pr2) < 0.35,
       f"partial rho ~ 0 when the two residuals are independent, got {pr2}")
    # positive control: the SAME swap in both -> the residuals coincide -> partial rho = 1
    pr3 = partial_spearman(y_ind, y_ind[:], z)
    ok(abs(pr3 - 1.0) < 1e-9,
       f"partial rho = 1 when the residuals coincide, got {pr3}")
    # y driven ONLY by x, z independent-ish: partial rho stays high
    x = [8.0, 7, 6, 5, 4, 3, 2, 1]
    y2 = [v * 3 for v in x]
    zc = [1.0, 1, 1, 1, 2, 2, 2, 2]
    ok(partial_spearman(x, y2, zc) > 0.8, "partial rho survives an unrelated control")
    ok(partial_spearman([1, 2], [1, 2], [1, 2]) != partial_spearman([1, 2], [1, 2], [1, 2]),
       "partial rho undefined below n=4")
    # a CONSTANT control must leave the plain rho unchanged (A2b's degenerate case)
    ok(abs(partial_spearman(x, y2, [5.0] * 8) - spearman(x, y2)) < 1e-9,
       "a constant control reduces the partial to the plain rho")

    # ---- exception / frozen bookkeeping ------------------------------------------------
    ok(C62.is_exception is not None and callable(C62.is_exception), "exception fn present")
    ok(set(FROZEN_ROOTS) == {"probes_fz3", "probes_p5"}, "frozen roots")

    print(f"  selftest: {n}/{n} PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.run:
        report(a.root, out_json=a.json)


if __name__ == "__main__":
    main()
