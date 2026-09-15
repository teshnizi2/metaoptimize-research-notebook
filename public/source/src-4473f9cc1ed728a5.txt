#!/usr/bin/env python3
r"""PROBE5 scale profile WITH ITS WINDOW SCAN -- and per-family n_weights.

WHY THIS EXISTS, and it is a direct application of CORRECTIONS 41(c).

CORRECTIONS 41(c) made it a standing rule that *a width, count, or ranking measured
against an absolute threshold may not be reported until it has been recomputed across a
range of thresholds and shown to be stable*.  CORRECTIONS 33 made the same demand of a
null (state the resolution).  The scale profile has a third free parameter of exactly the
same kind and nobody had scanned it: **the time window**.

`analysis/probe5_floor.py --profile` reduces the FULL run.  Cycle 48 measured that this
inflates rho_s by ~4.2x through the startup transient, and that the STEADY HALF reproduces
FINDINGS 44.3's independently measured frozen value 1.925e-06 to within 3-8% on a
byte-matched configuration (p5-w-a3 steady half: 1.991e-06 / 2.085e-06).  So the two
reducers were not disagreeing about the science -- they were reducing different windows,
and only one of them matches the number the campaign already published.

The verdict must therefore be reported against the whole scan, not one window:

    window          k=1        k=775      k=180,225   span    outcome
    full         1.408e-05   3.428e-06   2.637e-07    53.4x   (b)
    steady .5-1  3.200e-06   1.133e-06   4.620e-08    69.3x   (b)
    startup 0-.25 3.322e-05  6.778e-06   5.882e-07   520.0x   (b)   [blk6 also resolves]

Outcome (b) -- "the correlation length is REAL" -- holds in all three, so no window choice
is a selection.  **Quote the steady half**, because that is the window FINDINGS 44.3 used,
it is the regime the 1/sqrt(N) noise model is a claim about, and it is the most
conservative of the three that resolves every rung.

THE SECOND FIX: n_weights is PER FAMILY.
`probe5_floor.py` hardcodes `N_WEIGHTS_R18 = 11_173_962`, which is correct for the R18
batch and wrong for ResNet10, ResNet34 and any other model.  `k = n_weights / n_tot` is
the profile's x-axis, so a wrong n_weights rescales an entire family's curve.  This reducer
takes each family's n_weights from **that family's own weightwise arm's `neg_counts.json`
n_tot** -- the weightwise partition has exactly one group per parameter, so its n_tot IS
the parameter count.  It never reads `block_sizes.json`, which reports the R18 count for
nodewise arms too (CORRECTIONS 16).  A family with no weightwise arm is REFUSED, not
guessed at.

THE HETEROGENEITY CORRECTION IS FULL-RUN, BY CONSTRUCTION, AND THAT IS SOUND.
`neg_counts.npy` is a cumulative counter overwritten in place, so only the whole-run counts
survive to disk and Var_b(P_b) cannot be windowed.  Pairing it with a windowed rho_s is
legitimate: writing p_b = P_b + fbar + e_b, the common mode `fbar` is identical for every
coordinate, so it moves the MEAN of p_b across coordinates and not the cross-coordinate
VARIANCE.  A startup transient is a common mode.  This is the same algebra probe5_floor.py's
docstring uses to argue the common mode cannot contaminate Var_b(P_b); it is stated here
because the window pairing makes it load-bearing rather than incidental.

Run `--selftest` before trusting any number this prints.
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe5_floor import (corrected_floor, hetvar_from_counts, read_probe5,  # noqa: E402
                          recompute_rho)

try:
    from corr_range import group_rho_from_weight, implied_rho_w
    from twochannel import decompose, load
except ImportError:                                   # --selftest from any cwd
    group_rho_from_weight = implied_rho_w = decompose = load = None

# The windows scanned.  (label, (lo, hi) fractional slice or None for the whole run.)
WINDOWS = (("full", None), ("steady .5-1", (0.5, 1.0)), ("startup 0-.25", (0.0, 0.25)))

RUNG_TOKENS = ("blk6", "lay", "node", "w", "scal")
DEFAULT_FAMILY = "r18c10"

# Pre-registered discrimination thresholds, from bin/c44_probe5_heterogeneity.sh.
FLAT_WITHIN = 3.0      # outcome (a): flat to within 3x  -> profile is an artefact
FALLS_ATLEAST = 10.0   # outcome (b): falls >= 10x       -> the correlation length is real


# ------------------------------------------------------------------ dir name parsing
def parse_dirname(base):
    """('probe_r10_lay_s0') -> ('r10', 'lay', 0);  ('probe_blk6_a3_s1') -> (DEFAULT, 'blk6', 1).

    Two naming conventions are in the tree and both must parse:
      c47 p5 / c48 fr5 :  probe_<rung>_a3_s<seed>       -- one family, R18/CIFAR-10
      c48 fz3          :  probe_<family>_<rung>_s<seed> -- three families
    The rung is identified by membership in RUNG_TOKENS rather than by position, so the
    two layouts cannot be confused with each other.
    """
    parts = base.split("_")
    if parts and parts[0] == "probe":
        parts = parts[1:]
    seed = None
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
        parts = parts[:-1]
    rung = next((p for p in parts if p in RUNG_TOKENS), None)
    fam = next((p for p in parts if p != rung and p != "a3"), DEFAULT_FAMILY)
    return fam, rung, seed


def family_n_weights(rows):
    """n_weights for one family, taken from ITS OWN weightwise arm's n_tot.

    Returns None if the family has no weightwise arm -- callers must REFUSE the family
    rather than substitute another model's parameter count.
    """
    w = [r["n_tot"] for r in rows if r["rung"] == "w"]
    if not w:
        return None
    if len(set(w)) != 1:
        raise ValueError(f"weightwise arms disagree on n_tot: {sorted(set(w))}")
    return int(w[0])


# ------------------------------------------------------------------ per-dir reduction
def reduce_dir(d, windows=WINDOWS):
    """One probe dir -> {'fam','rung','seed','n_tot','win':{label: {...}}} or None."""
    got = read_probe5(d)
    if got is None:
        return None
    counts, meta = got
    T5, n_tot = int(meta["n_records"]), int(meta["n_tot"])
    recs = load(d) if load else []
    if not recs:
        return None
    fn = [r.get("frac_neg") for r in recs]
    fz = [r.get("frac_zero", 0.0) for r in recs]
    if any(v is None for v in fn):
        return None
    fam, rung, seed = parse_dirname(os.path.basename(d))
    out = {}
    for lab, w in windows:
        dec = decompose(fn, fz, n_tot, window=w)
        if not dec:
            continue
        hv, _, _, _ = hetvar_from_counts(counts, T5, tau=dec["tau"])
        v_un, _ = corrected_floor(dec["v_indep"], 0.0, n_tot)
        v_co, H = corrected_floor(dec["v_indep"], hv, n_tot)
        c_un, c_co = recompute_rho(dec, v_un), recompute_rho(dec, v_co)
        out[lab] = dict(T=dec["T"], tau=dec["tau"], H=H,
                        rho_s_unc=c_un["rho_s"], rho_s=c_co["rho_s"],
                        rho_min=c_co["rho_min"], resolved=c_co["resolved"])
    return dict(dir=d, fam=fam, rung=rung, seed=seed, n_tot=n_tot, win=out)


def geo(vals):
    """Geometric mean -- the right average for a quantity spanning decades."""
    v = [x for x in vals if np.isfinite(x) and x > 0]
    return math.exp(float(np.mean(np.log(v)))) if v else float("nan")


def powerlaw_b(ks, rhos):
    """Least-squares slope of log(rho_w) vs log(k), returned as the POSITIVE exponent b
    in rho_w ~ k^-b.  Needs >= 2 distinct k."""
    k = np.asarray([x for x in ks], dtype=float)
    r = np.asarray([x for x in rhos], dtype=float)
    m = np.isfinite(k) & np.isfinite(r) & (k > 0) & (r > 0)
    if m.sum() < 2 or len(set(k[m].tolist())) < 2:
        return float("nan")
    return float(-np.polyfit(np.log(k[m]), np.log(r[m]), 1)[0])


# ------------------------------------------------------------------ profile assembly
def profile(rows, n_weights, window):
    """Rung-level profile for one family in one window.

    Returns (points, excluded) where points is [(k, rho_w_geo, n_seeds), ...] sorted by k
    over the RESOLVED rungs only, and excluded names the rungs dropped for being below
    their own rho_min -- printed, never silently dropped (CORRECTIONS 33).
    """
    by_k, excluded = {}, []
    for r in rows:
        w = r["win"].get(window)
        if not w:
            continue
        k = max(n_weights / r["n_tot"], 1.0)
        if not w["resolved"]:
            excluded.append((os.path.basename(r["dir"]), w["rho_s"], w["rho_min"]))
            continue
        by_k.setdefault(k, []).append(implied_rho_w(w["rho_s"], k))
    pts = [(k, geo(v), len(v)) for k, v in sorted(by_k.items())]
    return pts, excluded


def uniform_model_check(rows, n_weights, window):
    """PARAMETER-FREE falsification of the exchangeable one-factor model.

    WHY THIS AND NOT THE SPAN.  "rho_w^implied falls 69.3x" is a statement about a
    REPARAMETRIZATION: rho_w^implied is just rho_s pushed through the one-factor inversion,
    so a reader can fairly ask whether the fall is an artefact of the model used to define
    the y-axis.  This block removes that objection.  It CALIBRATES the model on the finest
    rung alone -- one number, no fit -- then PREDICTS the measured quantity (rho_s, the
    between-group sign correlation) at every coarser rung and compares.  There is no free
    parameter anywhere, so a large ratio cannot be a fitting artefact.

    The model being tested is the one under which "pool N coordinates, noise falls as
    1/sqrt(N) with a single correlation rho" is stated: coordinates exchangeable, one global
    factor.  The Adam-mini / Adalayer / SGG line assumes the stronger rho = 0.

    Returns None if fewer than two rungs resolve.
    """
    by_k = {}
    for r in rows:
        w = r["win"].get(window)
        if not w or not w["resolved"]:
            continue
        by_k.setdefault(max(n_weights / r["n_tot"], 1.0), []).append(w["rho_s"])
    ks = sorted(by_k)
    if len(ks) < 2:
        return None
    k0 = ks[0]
    rho_s0 = geo(by_k[k0])
    rho_w0 = implied_rho_w(rho_s0, k0)      # the ONE calibrated number
    out = []
    for k in ks[1:]:
        meas = geo(by_k[k])
        pred = group_rho_from_weight(rho_w0, k)
        out.append((k, pred, meas, pred / meas if meas > 0 else float("nan")))
    return dict(k0=k0, rho_s0=rho_s0, rho_w0=rho_w0, rows=out)


def verdict(pts):
    """Score a profile against the pre-registered (a)/(b) of c44_probe5_heterogeneity.sh."""
    vals = [p[1] for p in pts]
    if len(vals) < 2:
        return "UNDECIDABLE", float("nan"), "fewer than 2 resolved rungs"
    span = max(vals) / min(vals)
    if span >= FALLS_ATLEAST:
        return "(b) RANGE IS REAL", span, f"falls {span:.1f}x >= {FALLS_ATLEAST:.0f}x"
    if span <= FLAT_WITHIN:
        return "(a) ARTEFACT", span, f"flat to {span:.1f}x <= {FLAT_WITHIN:.0f}x"
    return "NEITHER", span, f"{span:.1f}x is between the pre-registered {FLAT_WITHIN:.0f}x and {FALLS_ATLEAST:.0f}x"


def main(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "neg_counts.json")):
            r = reduce_dir(d)
            if r:
                rows.append(r)
    if not rows:
        print(f"no reducible PROBE5 dirs under {root}")
        return 1

    fams = sorted({r["fam"] for r in rows})
    print(f"PROBE5 SCALE PROFILE + WINDOW SCAN   ({len(rows)} dirs, "
          f"{len(fams)} famil{'y' if len(fams) == 1 else 'ies'}: {', '.join(fams)})")

    all_ok = True
    for fam in fams:
        frows = [r for r in rows if r["fam"] == fam]
        nw = family_n_weights(frows)
        print(f"\n=== family {fam} ===")
        if nw is None:
            print("  REFUSED: no weightwise arm, so n_weights is unknown for this family.")
            print("  (Never substitute another model's parameter count -- k would be wrong")
            print("   for every rung and the whole curve would shift.)")
            all_ok = False
            continue
        print(f"  n_weights = {nw:,}  (from this family's OWN weightwise n_tot, not block_sizes.json)")
        hdr = f"  {'window':15}{'span':>9}{'b in k^-b':>11}  profile (k: rho_w implied, geo-mean over seeds)"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        verds = {}
        for lab, _ in WINDOWS:
            pts, exc = profile(frows, nw, lab)
            v, span, why = verdict(pts)
            verds[lab] = v
            b = powerlaw_b([p[0] for p in pts], [p[1] for p in pts])
            body = "  ".join(f"k={k:,.0f}: {r:.3e} (n={n})" for k, r, n in pts) or "-- nothing resolved --"
            print(f"  {lab:15}{(f'{span:.1f}x' if np.isfinite(span) else '-'):>9}"
                  f"{(f'{b:.3f}' if np.isfinite(b) else '-'):>11}  {body}")
            if exc:
                names = ", ".join(f"{n} (rho_s={rs:.2e} < rho_min={rm:.2e})" for n, rs, rm in exc)
                print(f"  {'':15}{'':9}{'':11}  below own rho_min, EXCLUDED: {names}")
        # the parameter-free check, on the window we actually quote
        um = uniform_model_check(frows, nw, "steady .5-1")
        if um:
            print(f"  uniform one-factor model, CALIBRATED on k={um['k0']:,.0f} alone "
                  f"(rho_s={um['rho_s0']:.3e} -> rho_w={um['rho_w0']:.3e}), then PREDICTING:")
            print(f"  {'':4}{'k':>12}{'rho_s predicted':>18}{'rho_s measured':>17}{'over-predicts by':>18}")
            for k, pred, meas, ratio in um["rows"]:
                print(f"  {'':4}{k:>12,.0f}{pred:>18.3e}{meas:>17.3e}{ratio:>17.1f}x")
            worst = max((r[3] for r in um["rows"] if np.isfinite(r[3])), default=float("nan"))
            if np.isfinite(worst) and worst >= 3.0:
                print(f"  --> the exchangeable one-factor model is REJECTED: it over-predicts the")
                print(f"      between-group correlation by up to {worst:.0f}x with NO free parameter.")
                print(f"      Correlation between coordinates does not extend across the coarse groups.")
            elif np.isfinite(worst):
                print(f"  --> the exchangeable model is NOT rejected (worst factor {worst:.1f}x).")
        uniq = set(verds.values())
        print(f"  window scan: " + "; ".join(f"{k} -> {v}" for k, v in verds.items()))
        if len(uniq) == 1:
            print(f"  --> STABLE ACROSS WINDOWS: {uniq.pop()} in every window. "
                  f"No window choice is a selection.  Quote the steady half.")
        else:
            all_ok = False
            print("  --> NOT WINDOW-STABLE: the verdict depends on the window.  Report the")
            print("      whole scan, not a row (CORRECTIONS 41(c), applied to the window).")
    print("\nQuote the STEADY HALF: it is the window FINDINGS 44.3 used, it is the regime")
    print("the 1/sqrt(N) noise model is a claim about, and the full run is inflated ~4.2x")
    print("by the startup transient.")
    return 0 if all_ok else 2


# ------------------------------------------------------------------------- selftest
def _selftest():
    ok = fail = 0

    def chk(name, cond, extra=""):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name} {extra}")

    # -- W1: dirname parsing, both conventions, and they must not collide
    chk("W1 p5 layout", parse_dirname("probe_blk6_a3_s1") == (DEFAULT_FAMILY, "blk6", 1),
        str(parse_dirname("probe_blk6_a3_s1")))
    chk("W1 fz3 layout", parse_dirname("probe_r10_lay_s0") == ("r10", "lay", 0),
        str(parse_dirname("probe_r10_lay_s0")))
    chk("W1 fz3 weightwise", parse_dirname("probe_c100_w_s2") == ("c100", "w", 2),
        str(parse_dirname("probe_c100_w_s2")))
    chk("W1 p5 weightwise", parse_dirname("probe_w_a3_s0") == (DEFAULT_FAMILY, "w", 0),
        str(parse_dirname("probe_w_a3_s0")))

    # -- W2: n_weights comes from the family's OWN weightwise arm, and a family without
    #    one is REFUSED rather than silently given R18's count.  This is the bug that
    #    would rescale an entire ResNet34 curve.
    rows = [dict(rung="w", n_tot=21_282_122), dict(rung="lay", n_tot=110)]
    chk("W2 n_weights from weightwise", family_n_weights(rows) == 21_282_122)
    chk("W2 no weightwise -> None", family_n_weights([dict(rung="lay", n_tot=62)]) is None)
    try:
        family_n_weights([dict(rung="w", n_tot=10), dict(rung="w", n_tot=11)])
        chk("W2 inconsistent weightwise raises", False)
    except ValueError:
        chk("W2 inconsistent weightwise raises", True)

    # -- W3: geometric mean, and that it ignores non-positive / non-finite entries
    chk("W3 geo mean", abs(geo([1e-6, 1e-4]) - 1e-5) < 1e-12, f"{geo([1e-6, 1e-4]):.3e}")
    chk("W3 geo skips nan", abs(geo([1e-5, float('nan')]) - 1e-5) < 1e-12)
    chk("W3 geo skips zero", abs(geo([1e-5, 0.0]) - 1e-5) < 1e-12)
    chk("W3 geo of nothing is nan", math.isnan(geo([])))

    # -- W4: power-law fit recovers a KNOWN exponent on exact synthetic data
    ks = [1.0, 775.0, 180225.0]
    for b_true in (0.25, 0.33, 0.5):
        rho = [1e-5 * k ** (-b_true) for k in ks]
        chk(f"W4 recovers b={b_true}", abs(powerlaw_b(ks, rho) - b_true) < 1e-9,
            f"got {powerlaw_b(ks, rho):.6f}")
    chk("W4 single point -> nan", math.isnan(powerlaw_b([1.0], [1e-5])))
    chk("W4 duplicate k -> nan", math.isnan(powerlaw_b([5.0, 5.0], [1e-5, 2e-5])))

    # -- W5: the pre-registered verdict boundaries, checked ON the boundary
    chk("W5 10x is (b)", verdict([(1, 1e-5, 1), (100, 1e-6, 1)])[0] == "(b) RANGE IS REAL")
    chk("W5 3x is (a)", verdict([(1, 3e-6, 1), (100, 1e-6, 1)])[0] == "(a) ARTEFACT")
    chk("W5 5x is neither", verdict([(1, 5e-6, 1), (100, 1e-6, 1)])[0] == "NEITHER")
    chk("W5 one rung undecidable", verdict([(1, 1e-5, 1)])[0] == "UNDECIDABLE")
    chk("W5 empty undecidable", verdict([])[0] == "UNDECIDABLE")

    # -- W6: profile() EXCLUDES unresolved rungs and REPORTS them; a silently dropped
    #    rung is how a null becomes an absence (CORRECTIONS 33).
    def mk(rung, n_tot, rho_s, resolved):
        return dict(dir=f"probe_{rung}_a3_s0", fam=DEFAULT_FAMILY, rung=rung, seed=0,
                    n_tot=n_tot,
                    win={"steady .5-1": dict(T=1000, tau=1.0, H=1.0, rho_s_unc=rho_s,
                                             rho_s=rho_s, rho_min=1e-9 if resolved else 1.0,
                                             resolved=resolved)})
    frows = [mk("w", 11_173_962, 2.0e-6, True), mk("node", 14_420, 5.4e-4, True),
             mk("blk6", 6, -4.6e-2, False)]
    pts, exc = profile(frows, 11_173_962, "steady .5-1")
    chk("W6 resolved rungs kept", len(pts) == 2, f"{len(pts)}")
    chk("W6 unresolved rung excluded and named",
        len(exc) == 1 and exc[0][0] == "probe_blk6_a3_s0", str(exc))
    chk("W6 k ascending", [p[0] for p in pts] == sorted(p[0] for p in pts))
    chk("W6 k=1 for weightwise", abs(pts[0][0] - 1.0) < 1e-9, f"{pts[0][0]}")

    # -- W7: seeds at the same rung are averaged, not treated as separate rungs.  A
    #    2-seed batch must give 3 profile points, not 6.
    two = [mk("w", 11_173_962, 2.0e-6, True), mk("w", 11_173_962, 2.2e-6, True),
           mk("node", 14_420, 5.4e-4, True), mk("node", 14_420, 5.8e-4, True)]
    pts2, _ = profile(two, 11_173_962, "steady .5-1")
    chk("W7 seeds collapse to one point per rung", len(pts2) == 2, f"{len(pts2)}")
    chk("W7 seed count reported", [p[2] for p in pts2] == [2, 2], str([p[2] for p in pts2]))
    chk("W7 geo-mean of the two seeds",
        abs(pts2[0][1] - geo([implied_rho_w(2.0e-6, 1.0), implied_rho_w(2.2e-6, 1.0)])) < 1e-18)

    # -- W8: THE POINT OF THE FILE.  A profile that is (b) in one window and (a) in
    #    another must be reported as NOT window-stable.  Built as data, not asserted.
    v_b = verdict([(1, 1e-5, 1), (1e5, 1e-7, 1)])[0]
    v_a = verdict([(1, 1e-5, 1), (1e5, 5e-6, 1)])[0]
    chk("W8 divergent windows are distinguishable", v_b != v_a, f"{v_b} vs {v_a}")
    chk("W8 identical windows agree",
        verdict([(1, 1e-5, 1), (1e5, 1e-7, 1)])[0] == v_b)

    # -- W9: the window slice itself.  decompose(window=) must take the stated fraction
    #    of records; an off-by-one here silently changes every number in the table.
    if decompose is not None:
        fn = [0.5 + 0.01 * math.sin(i) for i in range(1000)]
        fz = [0.0] * 1000
        d_full = decompose(fn, fz, 100)
        d_half = decompose(fn, fz, 100, window=(0.5, 1.0))
        d_qtr = decompose(fn, fz, 100, window=(0.0, 0.25))
        chk("W9 full window T", d_full["T"] == 1000, str(d_full["T"]))
        chk("W9 steady half T", d_half["T"] == 500, str(d_half["T"]))
        chk("W9 startup quarter T", d_qtr["T"] == 250, str(d_qtr["T"]))

    # ---- U: the parameter-free uniform-model check.  The validation that matters is U2:
    #      on data GENERATED BY the uniform model the check must NOT report a rejection.
    #      An estimator that "rejects" everything proves nothing.
    if group_rho_from_weight is not None:
        for rs, k in ((1.991e-06, 1.0), (5.4e-04, 775.0), (5.186e-03, 180225.0)):
            rt = group_rho_from_weight(implied_rho_w(rs, k), k)
            chk(f"U1 round trip k={k:,.0f}", abs(rt - rs) < 1e-12 * max(rs, 1e-12) + 1e-15, True)

        def urows(pairs):   # pairs: (n_tot, rho_s)
            return [dict(dir=f"probe_x{i}_a3_s0", fam=DEFAULT_FAMILY, rung="w" if n == 11_173_962 else "node",
                         seed=0, n_tot=n,
                         win={"steady .5-1": dict(T=1000, tau=1.0, H=1.0, rho_s_unc=rs,
                                                  rho_s=rs, rho_min=1e-12, resolved=True)})
                    for i, (n, rs) in enumerate(pairs)]

        # U2: TRUE uniform model.  Pick rho_w, generate rho_s at each k from it exactly.
        NW = 11_173_962
        rho_w_true = 3.2e-06
        pairs = [(NW // k, group_rho_from_weight(rho_w_true, float(k))) for k in (1, 775, 180225)]
        um = uniform_model_check(urows(pairs), NW, "steady .5-1")
        ratios = [r[3] for r in um["rows"]]
        chk("U2 uniform data is NOT rejected", all(abs(r - 1.0) < 0.02 for r in ratios),
            f"ratios={[round(r, 4) for r in ratios]}")

        # U3: coarse rung deliberately 50x BELOW the uniform prediction -> must be rejected
        pairs3 = list(pairs)
        pairs3[-1] = (pairs3[-1][0], pairs3[-1][1] / 50.0)
        um3 = uniform_model_check(urows(pairs3), NW, "steady .5-1")
        chk("U3 sub-uniform coarse rung IS rejected", um3["rows"][-1][3] > 40.0,
            f"{um3['rows'][-1][3]:.1f}x")

        # U4: calibration uses the FINEST rung, and a single rung is undecidable
        chk("U4 calibrates on the finest k", abs(um["k0"] - 1.0) < 1e-9, f"{um['k0']}")
        chk("U4 one rung -> None", uniform_model_check(urows(pairs[:1]), NW, "steady .5-1") is None)
        chk("U4 unresolved rungs are not used",
            uniform_model_check([dict(dir="d", fam=DEFAULT_FAMILY, rung="w", seed=0, n_tot=NW,
                                      win={"steady .5-1": dict(T=1, tau=1.0, H=1.0, rho_s_unc=1e-6,
                                                               rho_s=1e-6, rho_min=1.0, resolved=False)})],
                                NW, "steady .5-1") is None)

    print(f"selftest: {ok}/{ok + fail} PASS")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    _root = next((a for a in sys.argv[1:] if not a.startswith("--")), "../probes_p5")
    sys.exit(main(_root))
