#!/usr/bin/env python3
"""IDEA 3 reducer -- alpha0-robustness width, head to head, at TWO budgets.

Written and unit-tested BEFORE the data landed, so the analysis is pre-registered
rather than chosen after seeing the curves (docs/IDEA3-robustness.md).

Reports, per arm:
    (i)   WIDTH of the region within 1pp / 2pp of that arm's OWN best, in decades
    (ii)  WORST-CASE plateau across the whole grid
    (iii) PEAK plateau
and then the tradeoff sentence: "costs X pp of peak, buys N decades of alpha0
insensitivity", where X is measured against the tuned AdamW+cosine reference.

THE CONVERGENCE CONTROL (cycle 46).  Neither arm is converged at 100 epochs --
FINDINGS 44.1 measures 94.417 -> 94.999 -> 95.138 (cosine) and 92.795 -> 93.033 ->
93.201 (meta) at 100/300/600 -- and the under-convergence is DIFFERENTIAL: at
alpha0=1e-6 MetaOptimize must first grow its own step size, costing +11.4 to +19.0
epochs on `ep_to_85` (audit 5's correction to standing rule R5).  So a 100-epoch
budget understates arm B at exactly the extreme the robustness claim is about.
`bin/c46_idea3_budget_control.sh` runs the FOUR EXTREMES {1e-6, 1e-5, 1e-2, 1e-1} of
both arms at 300 epochs, 2 seeds.  This reducer therefore reports every robustness
metric at BOTH budgets and states explicitly whether the SHAPE is budget-stable.

    If the shape is NOT budget-stable, the 100-epoch sweep is not a valid basis for
    the robustness claim, and this reducer says so in those words.

Width convention (stated because it changes the number):
  the grid is 7 points spanning log10 alpha0 in [-6, -1].  A cell is IN-BAND if its
  plateau >= (arm's own best - tol).  Width is the span of the LARGEST CONTIGUOUS run
  of in-band cells, measured in decades between the first and last in-band grid point.
  A single in-band cell has width 0 decades and is reported as such -- it is not
  padded out to half a grid spacing.  Contiguity is required: a non-contiguous
  in-band set means the curve is not a plateau and the width is not meaningful, so
  the reducer prints a WARNING and reports only the largest contiguous run.

  On the 4-point EXTREMES sub-grid the same rule applies, but "contiguous" there can
  jump the unmeasured interior 1e-4..1e-3.  Any width that crosses the 1e-5 -> 1e-2
  gap is flagged and is an UPPER BOUND on the true width at that budget.

Usage:  python3 analysis/idea3_robustness.py [results/all_runs.csv]
        python3 analysis/idea3_robustness.py --selftest
"""
import csv
import math
import sys

# CORRECTIONS 30: the tuned non-meta peak reference, n=5, 100ep, AUGMENT=1, R18/C10.
TUNED_COSINE_REF = 94.417
TUNED_COSINE_SD = 0.113

GRID = [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1]

# The convergence control runs the EXTREMES only -- c46.  Budget-stability is judged
# on this matched sub-grid, never on 7 points at 100ep vs 4 points at 300ep.
EXTREMES = [1e-6, 1e-5, 1e-2, 1e-1]

# audit 5 (FINDINGS 36.2, rule R7): same-config same-seed spread is 0.119pp median /
# 0.243pp p90.  At n=2 per cell a 100-vs-300 delta below this is not resolvable.
DELTA_RESOLUTION = 0.3

# label -> (100-epoch prefix, 300-epoch prefix or None, min epochs at each budget)
ARMS = {
    "A  fixed-lr AdamW": ("i3a-", "i3a300-"),
    "B  MetaOptimize m=6": ("i3b-", "i3b300-"),
    "B' clip control": ("i3bc-", None),
}
BUDGETS = (100, 300)


def load(path):
    """Every row with a parseable plateau.  Epoch filtering is per-arm, not global."""
    out = []
    for r in csv.DictReader(open(path)):
        if r.get("superseded") not in ("0", ""):
            continue
        try:
            r["_epochs"] = int(r["epochs_done"])
            r["_plateau"] = float(r["plateau"])
        except (ValueError, KeyError, TypeError):
            continue
        out.append(r)
    return out


def cells(rows, prefix, min_epochs):
    """{alpha0: [plateau, ...]} for one arm at one budget, keyed by float alpha0.

    `min_epochs` is REQUIRED and is what keeps the two budgets apart.  The prefixes
    are already disjoint (`i3a300-` does not start with `i3a-`), but the epoch filter
    is the thing that makes a mislabelled row impossible to pool across budgets.
    """
    acc = {}
    for r in rows:
        if not r["run"].startswith(prefix):
            continue
        # defensive: never let a longer sibling prefix be swallowed by a shorter one
        if prefix == "i3b-" and r["run"].startswith("i3bc-"):
            continue
        if r["_epochs"] < min_epochs:
            continue
        acc.setdefault(float(r["alpha0"]), []).append(r["_plateau"])
    return acc


# A run at or below this plateau has COLLAPSED (the campaign has seeds at exactly
# 10.000 = chance).  It is a different outcome, not a low score, and averaging it into a
# cell produces a number no seed is near.  `i3b-1e1` = {10.000, 86.858} averaged to
# 48.429 +-54.3 and that flipped the SIGN of this file's own clip-control verdict --
# CORRECTIONS 39.  The `collapsed` column in all_runs.csv is 0 on every row and flags
# nothing, so the filter has to live here.
DIVERGED = 50.0


def split_div(v):
    """(survivors, n_diverged)."""
    return [x for x in v if x > DIVERGED], sum(1 for x in v if x <= DIVERGED)


def mean_sd(v):
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, float("nan")
    var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
    return m, math.sqrt(var)


def band_flags(grid, means, tol, divs=None):
    """In-band status over the PRESENT grid points, vs that arm's own best.

    A grid point with ANY diverged seed is never in band, at any tolerance: a step size
    that collapses on some seeds is not one the method is robust at, whatever the
    surviving seeds averaged to.
    """
    divs = divs or {}
    present = [g for g in grid if g in means]
    if not present:
        return [], []
    best = max(means[g] for g in present)
    return present, [means[g] >= best - tol and not divs.get(g) for g in present]


def width_decades(grid, means, tol, divs=None):
    """Largest contiguous in-band run, in decades.  Returns (decades, lo, hi, gapped)."""
    present, flags = band_flags(grid, means, tol, divs)
    if not present:
        return float("nan"), None, None, False
    gapped = False
    # find longest contiguous True run
    bi = bj = -1
    i = 0
    runs = 0
    while i < len(flags):
        if flags[i]:
            runs += 1
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            if bi < 0 or (j - i) > (bj - bi):
                bi, bj = i, j
            i = j + 1
        else:
            i += 1
    if runs > 1:
        gapped = True
    lo, hi = present[bi], present[bj]
    return math.log10(hi) - math.log10(lo), lo, hi, gapped


def crosses_unmeasured_interior(lo, hi):
    """True if a sub-grid in-band run spans the 1e-5 -> 1e-2 gap the extremes leave open."""
    if lo is None or hi is None:
        return False
    return lo <= 1e-5 and hi >= 1e-2


def arm_stats(acc, grid):
    """Peak / worst / widths / band vectors for one arm at one budget on one grid."""
    surv = {a: split_div(v)[0] for a, v in acc.items() if a in grid}
    divs = {a: split_div(v)[1] for a, v in acc.items() if a in grid}
    # a cell with NO surviving seed has no mean; it is reported as missing, not as a low
    # score.  Arm A at alpha0=1e-1 is exactly this: 3 of 3 seeds collapse.
    means = {a: mean_sd(v)[0] for a, v in surv.items() if v}
    if not means:
        return None
    st = {"means": means, "divs": divs,
          "sds": {a: mean_sd(surv[a])[1] for a in means},
          "ns": {a: len(surv[a]) for a in means},
          "peak": max(means.values()),
          "worst": min(means.values()),
          "peak_a0": max(means, key=lambda k: means[k]),
          "missing": [g for g in grid if g not in means],
          "allgone": [g for g in grid if g in divs and g not in means]}
    for tol in (1.0, 2.0):
        w, lo, hi, gapped = width_decades(grid, means, tol, divs)
        present, flags = band_flags(grid, means, tol, divs)
        key = f"w{int(tol)}"
        st[key] = w
        st[key + "_lo"] = lo
        st[key + "_hi"] = hi
        st[key + "_gapped"] = gapped
        st[key + "_flags"] = tuple(flags)
        st[key + "_present"] = tuple(present)
        st[key + "_spans_gap"] = crosses_unmeasured_interior(lo, hi)
    st["complete"] = not st["missing"]
    return st


def print_arm(label, budget, st):
    print(f"\n=== {label}  @ {budget} epochs ===")
    print(f"{'alpha0':>8} {'n':>3} {'plateau':>9} {'sd':>7} {'diverged':>9}")
    for a in sorted(st["means"]):
        nd = st.get("divs", {}).get(a, 0)
        print(f"{a:>8.0e} {st['ns'][a]:>3} {st['means'][a]:>9.3f} "
              f"{st['sds'][a]:>7.3f} {(nd if nd else '.'):>9}")
    for a in sorted(st.get("allgone", [])):
        print(f"{a:>8.0e} {0:>3} {'  no survivor':>9} {'':>7} "
              f"{st['divs'][a]:>9}   <- EVERY seed collapsed")
    if st["missing"]:
        print("  INCOMPLETE -- missing grid points: "
              + ", ".join(f"{g:.0e}" for g in st["missing"]))
    bad = [g for g in sorted(st.get("divs", {})) if st["divs"][g]]
    if bad:
        print("  DIVERGED seeds at " + ", ".join(f"{g:.0e}" for g in bad)
              + " -- excluded from every width band below.")
    print(f"  (iii) PEAK      {st['peak']:.3f} at alpha0={st['peak_a0']:.0e}")
    print(f"  (ii)  WORST     {st['worst']:.3f}   (grid span {st['peak']-st['worst']:.3f} pp)")
    for tol in (1.0, 2.0):
        k = f"w{int(tol)}"
        flag = "  [WARNING: in-band set is NOT contiguous]" if st[k + "_gapped"] else ""
        if st[k + "_spans_gap"]:
            flag += "  [UPPER BOUND: spans the unmeasured 1e-5..1e-2 interior]"
        print(f"  (i)   WIDTH within {tol:.0f}pp of own best: "
              f"{st[k]:.1f} decades  [{st[k+'_lo']:.0e} .. {st[k+'_hi']:.0e}]{flag}")


def budget_stability(sub):
    """Is the SHAPE the same at 100 and at 300 epochs?

    `sub` is {label: {budget: stats-on-the-EXTREMES-sub-grid}} for arms A and B.
    Three pre-registered checks (c46 header, prediction B1); ALL must hold:
      1. every cell keeps its in-band/out-of-band status at 1pp and 2pp, both arms
      2. the sub-grid width is unchanged at 1pp and 2pp, both arms
      3. the A-vs-B width ORDERING (which arm is flatter) is unchanged, 1pp and 2pp
    Returns (stable: bool, failures: [str]).  A missing budget -> (None, [...]).
    """
    fails = []
    labs = [l for l in sub if l.startswith("A") or l.startswith("B ")]
    for lab in labs:
        for b in BUDGETS:
            if sub[lab].get(b) is None:
                fails.append(f"{lab}: no data at {b} epochs")
    if fails:
        return None, fails

    for lab in labs:
        s100, s300 = sub[lab][100], sub[lab][300]
        for tol in (1, 2):
            k = f"w{tol}"
            if s100[k + "_present"] != s300[k + "_present"]:
                fails.append(f"{lab}: sub-grid cells differ between budgets "
                             f"({tol}pp) -- cannot compare shapes")
                continue
            if s100[k + "_flags"] != s300[k + "_flags"]:
                changed = [f"{a:.0e}" for a, f1, f3 in
                           zip(s100[k + "_present"], s100[k + "_flags"], s300[k + "_flags"])
                           if f1 != f3]
                fails.append(f"{lab}: band membership at {tol}pp CHANGES with budget "
                             f"at alpha0 = {', '.join(changed)}")
            if abs(s100[k] - s300[k]) > 1e-9:
                fails.append(f"{lab}: width at {tol}pp changes "
                             f"{s100[k]:.1f} -> {s300[k]:.1f} decades")

    a, b = sub.get("A  fixed-lr AdamW"), sub.get("B  MetaOptimize m=6")
    if a and b:
        for tol in (1, 2):
            k = f"w{tol}"
            o100 = _sign(b[100][k] - a[100][k])
            o300 = _sign(b[300][k] - a[300][k])
            if o100 != o300:
                fails.append(f"A-vs-B width ORDERING at {tol}pp FLIPS with budget "
                             f"(B-A: {b[100][k]-a[100][k]:+.1f} at 100ep, "
                             f"{b[300][k]-a[300][k]:+.1f} at 300ep)")
    return (not fails), fails


def _sign(x):
    return (x > 0) - (x < 0)


def report(path):
    rows = load(path)
    full, sub = {}, {}

    for label, (p100, p300) in ARMS.items():
        full[label] = {}
        sub[label] = {}
        for budget, prefix in ((100, p100), (300, p300)):
            if prefix is None:
                full[label][budget] = sub[label][budget] = None
                continue
            acc = cells(rows, prefix, budget)
            if not acc:
                full[label][budget] = sub[label][budget] = None
                continue
            full[label][budget] = arm_stats(acc, GRID if budget == 100 else EXTREMES)
            sub[label][budget] = arm_stats(acc, EXTREMES)

    # ---- per-arm tables, both budgets ----
    for label, (p100, p300) in ARMS.items():
        for budget, prefix in ((100, p100), (300, p300)):
            if prefix is None:
                continue
            st = full[label][budget]
            if st is None:
                print(f"\n=== {label}  @ {budget} epochs ({prefix}*) -- NO ROWS YET ===")
                continue
            print_arm(f"{label} ({prefix}*)", budget, st)

    # ---- the 100-epoch tradeoff, unchanged ----
    a = full.get("A  fixed-lr AdamW", {}).get(100)
    b = full.get("B  MetaOptimize m=6", {}).get(100)
    bc = full.get("B' clip control", {}).get(100)

    print("\n=== THE TRADEOFF (100 epochs, full 7-point grid) ===")
    print(f"peak reference: tuned AdamW+cosine {TUNED_COSINE_REF:.3f} "
          f"+-{TUNED_COSINE_SD:.3f} (n=5, CORRECTIONS 30)")
    if not (a and b):
        print("  both arms needed; not computable yet.")
    else:
        if not (a["complete"] and b["complete"]):
            print("  PROVISIONAL -- at least one arm is missing grid points (see above).")
        cost = TUNED_COSINE_REF - b["peak"]
        buys1 = b["w1"] - a["w1"]
        buys2 = b["w2"] - a["w2"]
        print(f"  MetaOptimize peak      {b['peak']:.3f}  -> costs {cost:+.3f} pp "
              f"of peak vs the tuned baseline")
        print(f"  fixed-lr AdamW peak    {a['peak']:.3f}")
        print(f"  width within 1pp:  arm A {a['w1']:.1f} dec   arm B {b['w1']:.1f} dec"
              f"   -> buys {buys1:+.1f} decades")
        print(f"  width within 2pp:  arm A {a['w2']:.1f} dec   arm B {b['w2']:.1f} dec"
              f"   -> buys {buys2:+.1f} decades")
        print(f"  worst case:        arm A {a['worst']:.3f}      arm B {b['worst']:.3f}"
              f"   -> {b['worst'] - a['worst']:+.3f} pp")
        if buys1 > 0:
            print(f"\n  CLAIM: MetaOptimize costs {cost:.2f} pp of peak accuracy and buys "
                  f"{buys1:.1f} decades of alpha0 insensitivity (1pp band).")
        else:
            print("\n  REFUTED: MetaOptimize is NOT flatter than a fixed step size on this "
                  "grid.  The parent paper's own robustness claim fails at m=6 on "
                  "ResNet18/CIFAR-10.  Write it as the negative result it is.")

    # ---- the convergence control ----
    report_budget_stability(sub)

    if bc:
        top = 1e-1
        if b and top in b["means"] and top in bc["means"]:
            d = bc["means"][top] - b["means"][top]
            nb = b["ns"][top]
            nbc = bc["ns"][top]
            db = b.get("divs", {}).get(top, 0)
            dbc = bc.get("divs", {}).get(top, 0)
            print("\n=== CLIP CONTROL (alpha0=1e-1, 100 epochs) ===")
            print("  Means are over SURVIVING seeds only.  CORRECTIONS 39: the previous "
                  "version of")
            print("  this block averaged a collapsed seed in and reported the delta with "
                  "the WRONG SIGN.")
            print(f"  BETA_CLIP=-15:-2.3026 (alpha<=0.1)  {b['means'][top]:8.3f}"
                  f"   n={nb}  diverged={db}")
            print(f"  BETA_CLIP=-15:0       (alpha<=1.0)  {bc['means'][top]:8.3f}"
                  f"   n={nbc}  diverged={dbc}")
            print(f"  delta {d:+.3f} pp on survivors.")
            if db or dbc:
                print(f"  DIVERGENCE RATES DIFFER ({db}/{nb + db} vs {dbc}/{nbc + dbc}). "
                      "That, not the delta, is the")
                print("  result at this alpha0: the two settings fail in DIFFERENT WAYS. "
                      "Do not quote the")
                print("  delta alone.")
            if min(nb, nbc) < 2:
                print(f"  NOT DECIDABLE: n={min(nb, nbc)} surviving seed(s) on one side. "
                      "Re-read when the")
                print("  third seeds land.")
            elif abs(d) < 1.0:
                print("  Guard is NOT doing the work at the top end; arm B's top-end "
                      "flatness is real.")
            else:
                print("  The guard IS part of arm B's top-end behaviour. State it.")


def report_budget_stability(sub):
    """The c46 section: every metric at both budgets, then the shape verdict."""
    print("\n=== CONVERGENCE CONTROL -- IS THE SHAPE BUDGET-STABLE? ===")
    print("  matched 4-point EXTREMES sub-grid {1e-6, 1e-5, 1e-2, 1e-1}; 100ep metrics")
    print("  are RECOMPUTED on this sub-grid so the two budgets are compared like for like.")

    labs = [l for l in ARMS if l.startswith("A") or l.startswith("B ")]
    for lab in labs:
        s100, s300 = sub[lab].get(100), sub[lab].get(300)
        print(f"\n  -- {lab} --")
        if s300 is None:
            print("     300-epoch rows have not landed yet (c46 in flight); "
                  "the 100-epoch width below is UNCONTROLLED.")
        if s100 is None:
            print("     no 100-epoch rows on the sub-grid.")
            continue
        print(f"     {'alpha0':>8} {'100ep':>9} {'n':>3} {'300ep':>9} {'n':>3} {'delta':>8}")
        for g in EXTREMES:
            m1 = s100["means"].get(g)
            m3 = s300["means"].get(g) if s300 is not None else None
            c1 = f"{m1:9.3f}" if m1 is not None else " " * 9
            n1 = f"{s100['ns'][g]:3d}" if m1 is not None else "  -"
            c3 = f"{m3:9.3f}" if m3 is not None else " " * 9
            n3 = f"{s300['ns'][g]:3d}" if (m3 is not None and s300 is not None) else "  -"
            d = f"{m3-m1:+8.3f}" if (m1 is not None and m3 is not None) else " " * 8
            mark = ""
            if m1 is not None and m3 is not None and abs(m3 - m1) < DELTA_RESOLUTION:
                mark = "  (below the 0.3pp resolution floor -- do not interpret)"
            print(f"     {g:>8.0e} {c1} {n1} {c3} {n3} {d}{mark}")
        for tag, st in (("100ep", s100), ("300ep", s300)):
            if st is None:
                continue
            g1 = "  [UPPER BOUND: spans unmeasured interior]" if st["w1_spans_gap"] else ""
            print(f"     {tag}: peak {st['peak']:7.3f}  worst {st['worst']:7.3f}  "
                  f"width1 {st['w1']:.1f} dec  width2 {st['w2']:.1f} dec{g1}")

    # (B3) the startup-tax refund: does arm B gain MORE at the bottom than at the top?
    b = sub.get("B  MetaOptimize m=6", {})
    if b.get(100) and b.get(300):
        lo, hi = 1e-6, 1e-2
        if all(g in b[100]["means"] and g in b[300]["means"] for g in (lo, hi)):
            dlo = b[300]["means"][lo] - b[100]["means"][lo]
            dhi = b[300]["means"][hi] - b[100]["means"][hi]
            print(f"\n  (B3) arm B startup-tax refund: gain at 1e-6 {dlo:+.3f} pp vs "
                  f"at 1e-2 {dhi:+.3f} pp  -> differential {dlo-dhi:+.3f} pp")
            if dlo - dhi > DELTA_RESOLUTION:
                print("       The bottom end gains MORE. The 100-epoch budget UNDERSTATED "
                      "arm B's bottom-end robustness; its 100ep width is a LOWER bound.")
            elif dhi - dlo > DELTA_RESOLUTION:
                print("       The TOP end gains more -- the opposite of B3. The 100-epoch "
                      "width was not depressed by the startup tax.")
            else:
                print("       Differential is inside the 0.3pp resolution floor: "
                      "no startup-tax refund is resolvable at n=2.")

    stable, fails = budget_stability(sub)
    print("\n  VERDICT:")
    if stable is None:
        print("    NOT YET DECIDABLE -- " + "; ".join(fails))
        print("    Until the 300-epoch control lands, the 100-epoch width CANNOT be "
              "separated from a budget artefact, and no robustness claim should be "
              "written off the 100-epoch sweep alone.")
        return
    if stable:
        print("    THE SHAPE IS BUDGET-STABLE.  Band membership, both widths and the "
              "A-vs-B ordering are identical at 100 and 300 epochs on the matched "
              "sub-grid.  The 100-epoch sweep IS a valid basis for the robustness "
              "claim, and the full-grid 100-epoch width above stands.")
    else:
        print("    THE SHAPE IS **NOT** BUDGET-STABLE:")
        for f in fails:
            print(f"      - {f}")
        print("    Therefore the 100-epoch sweep is NOT a valid basis for the "
              "robustness claim.  Do not patch this with a caveat: the width must be "
              "re-measured on the FULL grid at 300 epochs, and every 100-epoch width "
              "number must be reported as budget-confounded.")


# --------------------------------------------------------------------------
def selftest():
    """The width rule and the budget-stability verdict are the non-obvious things
    here, so they are the things tested."""
    g = GRID
    ok = 0

    def chk(name, got, want):
        nonlocal ok
        assert abs(got - want) < 1e-9, f"{name}: got {got}, want {want}"
        ok += 1

    def yes(name, cond):
        nonlocal ok
        assert cond, name
        ok += 1

    # ---- width rule (unchanged from the pre-data version) ----
    # flat everywhere -> full 5-decade span
    m = {a: 92.0 for a in g}
    chk("flat-1pp", width_decades(g, m, 1.0)[0], 5.0)

    # a single peak, everything else far below -> 0 decades, not half a spacing
    m = {a: 10.0 for a in g}
    m[1e-3] = 94.0
    chk("spike", width_decades(g, m, 1.0)[0], 0.0)

    # collapse at the bottom: 1e-6/1e-5 dead, 1e-4..1e-1 within 1pp -> 3 decades
    m = {1e-6: 20.0, 1e-5: 60.0, 1e-4: 93.5, 3e-4: 94.0, 1e-3: 94.2, 1e-2: 93.4, 1e-1: 93.3}
    chk("collapse-lo-1pp", width_decades(g, m, 1.0)[0], 3.0)
    # at 2pp the 1e-5 cell is still 34pp down, so the width does not grow
    chk("collapse-lo-2pp", width_decades(g, m, 2.0)[0], 3.0)

    # non-contiguous in-band set -> flagged, and only the longest run counted
    m = {1e-6: 92.0, 1e-5: 80.0, 1e-4: 92.0, 3e-4: 92.0, 1e-3: 92.0, 1e-2: 80.0, 1e-1: 92.0}
    w, lo, hi, gapped = width_decades(g, m, 1.0)
    assert gapped, "should flag a gapped in-band set"
    chk("gapped-longest-run", w, 1.0)  # 1e-4 .. 1e-3

    # partial grid (mid-batch read) must not crash and must use only present cells
    m = {1e-6: 91.5, 1e-5: 91.8}
    chk("partial", width_decades(g, m, 1.0)[0], 1.0)

    # tolerance boundary is inclusive: exactly 1.00pp down is IN band
    m = {a: 90.0 for a in g}
    m[1e-3] = 91.0
    chk("boundary-inclusive", width_decades(g, m, 1.0)[0], 5.0)

    # mean/sd
    # --- divergence handling (added cycle 47; CORRECTIONS 39) ---
    # `chk` compares numerically, so every assertion below is reduced to a number.
    chk("split_div survivor kept", split_div([10.0, 86.858])[0][0], 86.858)
    chk("split_div survivor count", len(split_div([10.0, 86.858])[0]), 1)
    chk("split_div diverged count", split_div([10.0, 86.858])[1], 1)
    chk("split_div boundary 50 is diverged", split_div([50.0, 91.0])[1], 1)
    chk("split_div boundary 50.001 survives", split_div([50.001, 91.0])[1], 0)
    gd = [1e-6, 1e-5, 1e-4]
    md = {1e-6: 91.0, 1e-5: 91.0, 1e-4: 91.0}
    chk("width ignores divs when none given", width_decades(gd, md, 1.0)[0], 2.0)
    chk("width excludes a diverged cell", width_decades(gd, md, 1.0, {1e-5: 1})[0], 0.0)
    chk("band_flags: diverged cell is out of band",
        int(band_flags(gd, md, 1.0, {1e-5: 1})[1][1]), 0)
    chk("band_flags: neighbours stay in band",
        int(band_flags(gd, md, 1.0, {1e-5: 1})[1][0]), 1)
    stx = arm_stats({1e-6: [91.0, 92.0], 1e-5: [10.0, 10.0], 1e-4: [10.0, 91.0]},
                    [1e-6, 1e-5, 1e-4])
    chk("arm_stats: all-diverged cell has no mean", int(1e-5 in stx["means"]), 0)
    chk("arm_stats: all-diverged cell is in allgone", len(stx["allgone"]), 1)
    chk("arm_stats: partial-diverged mean is survivor-only", stx["means"][1e-4], 91.0)
    chk("arm_stats: divergence counted", stx["divs"][1e-4], 1)
    chk("arm_stats: partial-diverged cell out of band",
        int(band_flags([1e-6, 1e-4], stx["means"], 1.0, stx["divs"])[1][1]), 0)
    chk("arm_stats: sd is over survivors only",
        int(math.isnan(stx["sds"][1e-4])), 1)

    mm, ss = mean_sd([1.0, 2.0, 3.0])
    assert abs(mm - 2.0) < 1e-12 and abs(ss - 1.0) < 1e-12
    ok += 1
    assert math.isnan(mean_sd([5.0])[1])
    ok += 1

    # ---- the EXTREMES sub-grid and its unmeasured interior ----
    # flat across all four extremes -> 5 decades, but it JUMPS the 1e-5..1e-2 gap
    m = {a: 92.0 for a in EXTREMES}
    w, lo, hi, _ = width_decades(EXTREMES, m, 1.0)
    chk("sub-flat", w, 5.0)
    yes("sub-flat flags the unmeasured interior", crosses_unmeasured_interior(lo, hi))
    # a run confined to the bottom two extremes does NOT cross the gap
    m = {1e-6: 92.0, 1e-5: 92.0, 1e-2: 60.0, 1e-1: 60.0}
    w, lo, hi, _ = width_decades(EXTREMES, m, 1.0)
    chk("sub-bottom-only", w, 1.0)
    yes("bottom-only does not flag the gap", not crosses_unmeasured_interior(lo, hi))

    # ---- budget stability ----
    def mk(means):
        return arm_stats({a: [v] for a, v in means.items()}, EXTREMES)

    flatA = {1e-6: 20.0, 1e-5: 60.0, 1e-2: 93.0, 1e-1: 60.0}   # A: only 1e-2 in band
    flatB = {a: 92.0 for a in EXTREMES}                         # B: flat everywhere

    # (a) identical shapes, both arms uniformly shifted up -> STABLE
    up = 0.6
    s = {"A  fixed-lr AdamW": {100: mk(flatA), 300: mk({k: v + up for k, v in flatA.items()})},
         "B  MetaOptimize m=6": {100: mk(flatB), 300: mk({k: v + up for k, v in flatB.items()})}}
    stable, fails = budget_stability(s)
    yes("uniform shift is budget-stable", stable is True and not fails)

    # (b) a cell CHANGES band status at 300 -> NOT stable, and the failure names it
    b300 = dict(flatB)
    b300[1e-6] = 88.0                    # falls out of the 1pp band
    s = {"A  fixed-lr AdamW": {100: mk(flatA), 300: mk(flatA)},
         "B  MetaOptimize m=6": {100: mk(flatB), 300: mk(b300)}}
    stable, fails = budget_stability(s)
    yes("band flip is caught", stable is False)
    yes("band-flip failure names the cell", any("1e-06" in f for f in fails))

    # (c) the A-vs-B width ORDERING flips -> caught even though each arm is internally
    #     consistent in band membership only if widths change; here B collapses to a
    #     spike and A stays a spike, so the ordering goes from B>A to B==A.
    b_spike = {1e-6: 60.0, 1e-5: 60.0, 1e-2: 60.0, 1e-1: 93.0}
    s = {"A  fixed-lr AdamW": {100: mk(flatA), 300: mk(flatA)},
         "B  MetaOptimize m=6": {100: mk(flatB), 300: mk(b_spike)}}
    stable, fails = budget_stability(s)
    yes("width collapse is caught", stable is False)
    yes("width change is reported", any("width at 1pp changes" in f for f in fails))

    # (d) a missing budget is NOT-YET-DECIDABLE, never "stable"
    s = {"A  fixed-lr AdamW": {100: mk(flatA), 300: None},
         "B  MetaOptimize m=6": {100: mk(flatB), 300: None}}
    stable, fails = budget_stability(s)
    yes("missing 300ep is undecidable, not stable", stable is None and len(fails) == 2)

    # ---- the epoch filter is what keeps the budgets apart ----
    rows = [
        {"run": "i3b-1e6-s0", "alpha0": "1e-6", "_epochs": 100, "_plateau": 91.0},
        {"run": "i3b300-1e6-s0", "alpha0": "1e-6", "_epochs": 300, "_plateau": 92.0},
        {"run": "i3bc-1e1-s0", "alpha0": "1e-1", "_epochs": 100, "_plateau": 90.0},
        {"run": "i3b300-1e6-s1", "alpha0": "1e-6", "_epochs": 150, "_plateau": 50.0},  # truncated
    ]
    c100 = cells(rows, "i3b-", 100)
    c300 = cells(rows, "i3b300-", 300)
    yes("100ep cell excludes the 300ep run", c100 == {1e-6: [91.0]})
    yes("300ep cell excludes the 100ep run and the truncated one",
        c300 == {1e-6: [92.0]})
    yes("i3bc- is not swallowed by i3b-", 1e-1 not in c100)

    print(f"selftest: {ok}/{ok} PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        report(sys.argv[1] if len(sys.argv) > 1 else "results/all_runs.csv")
