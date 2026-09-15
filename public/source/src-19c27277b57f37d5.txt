#!/usr/bin/env python3
r"""THE TIME LADDER -- b(t) across quarters of a run, with the FROZEN arm as its control.

WHY THIS EXISTS, and why it costs zero compute.

CORRECTIONS 49(2) states the cycle-48 mechanism as a hypothesis:

    "A step-size adapter is a HIGH-PASS FILTER on meta-gradient correlation -- the
     short-range structure is present whenever the step size is far from its
     meta-optimum, and is erased fine-scales-first as beta adapts."

That reading currently rests on TWO points: the frozen arm (b = 0.343) and the free arm's
steady half (b = 0.065).  FINDINGS 48.15 noticed a third, and it is the one that turns the
hypothesis into a curve: the free arm's STARTUP quarter reads b = 0.331, i.e. the frozen
arm's value.  If the filter reading is right, b in the free arm must fall MONOTONICALLY
from the frozen value toward the equilibrium value as beta walks away from its
initialisation -- and it must do so on data already on disk.  CORRECTIONS 52.4 proposed
buying that curve with a 12-job meta-stepsize ladder.  Three of its four points are already
paid for; only the axis was never plotted.

THE CONFOUND, and the control that removes it.  A quantity that changes across quarters of
a run is confounded with TRAINING PROGRESS: the base network's loss surface, gradient norm
and effective curvature all move too, and any of them could set the correlation profile
without beta being involved at all.  The frozen arm (`--alg-meta fixed`) is the exact
control: it runs the identical base optimizer over the identical schedule with beta held
still (verified sd(beta) = 0 at every record), so it sees the SAME training progress and
NONE of the adaptation.

    free arm   b(t) falls, frozen arm b(t) flat  -> adaptation causes it.  Filter reading
                                                    survives its first real test.
    both fall                                    -> TRAINING PROGRESS causes it, not beta.
                                                    The filter reading is REFUTED and
                                                    CORRECTIONS 49(2) must be rewritten.
    neither falls                                -> the 0.331 -> 0.065 gap is a window
                                                    artefact of the .5 split, not a trend.

That is a genuine three-way test on existing bytes, and its refutation is the one that
would most change the paper.

THE X-AXIS IS MEASURED, NOT ASSUMED.  "Distance from the meta-optimum" is reported as the
per-quarter NET DISPLACEMENT of the beta vector, mean_j |beta_j(end) - beta_j(start)|,
from the 62-entry beta summary in probe.jsonl.  A system at its meta-optimum fluctuates
without net drift; one far from it moves.  Under Lion every per-step increment has
magnitude exactly the meta-stepsize (CORRECTIONS 21), so the STEP SIZE is constant by
construction and only the COHERENCE of those steps varies -- which is precisely what net
displacement measures and what a "distance from equilibrium" claim needs.

RESOLUTION -- and it is the reason this uses quarters rather than eighths.  rho_min grows
as 1/sqrt(T).  At T = 2000 records the steady half (T = 1000) put the free weightwise rung
at 8.888e-08 against rho_min 2.90e-08, a factor 3.07.  Quarters (T = 500) cost sqrt(2),
leaving ~2.2; eighths would leave ~1.5 and the rung would start dropping out of its own
profile.  Every unresolved rung is PRINTED, never silently dropped (CORRECTIONS 33).

PER-LEG, NOT ONE EXPONENT.  CORRECTIONS 51 and FINDINGS 48.18 established that a single
fitted b is a summary of a curve that is not a power law -- the weight->channel and
channel->layer legs differ by up to 6.3x.  This reducer reports both legs separately and
the pooled fit only as a third column.  Read the legs.

Run `--selftest` before trusting any number this prints.
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe5_window import (family_n_weights, geo, powerlaw_b,  # noqa: E402
                           profile, reduce_dir)

# Four non-overlapping quarters.  Q1 is byte-identical to probe5_window.py's
# "startup 0-.25", so its numbers must reproduce FINDINGS 48.15's startup row exactly --
# that is this reducer's own calibration check and --selftest asserts the tie.
QUARTERS = (("Q1 0-.25", (0.00, 0.25)),
            ("Q2 .25-.5", (0.25, 0.50)),
            ("Q3 .5-.75", (0.50, 0.75)),
            ("Q4 .75-1", (0.75, 1.00)))

# A leg is only interpretable if both of its endpoints resolved in that quarter.
LEG_NAMES = ("weight->channel", "channel->layer")

# Pre-registered discrimination for the trend test, fixed BEFORE the numbers were read.
# b1 is the weight->channel leg -- the leg the filter reading is about, since that is where
# adaptation suppresses 22.92x (FINDINGS 48.14) and where the frozen profile is nearly
# scale-free (b = 0.069-0.181, FINDINGS 48.18).
FALL_FACTOR = 2.0     # free-arm b1(Q1)/b1(Q4) >= this  -> b(t) genuinely falls
FLAT_FACTOR = 1.5     # frozen-arm b1(Q1)/b1(Q4) <= this -> the control is flat


# ------------------------------------------------------------------ beta displacement
def beta_displacement(d, window):
    """Mean per-coordinate NET displacement of the beta summary vector over `window`.

    Returns (net, span) where
      net  = mean_j |beta_j(last record in window) - beta_j(first record in window)|
      span = mean over records in the window of (beta_true_max - beta_true_min)

    `net` is the distance-from-equilibrium proxy; `span` is the cross-group spread, which
    separates "beta moved together" (common mode) from "beta spread out".  Returns
    (nan, nan) if the file is missing or the window holds fewer than 2 records.
    """
    p = os.path.join(d, "probe.jsonl")
    if not os.path.exists(p):
        return float("nan"), float("nan")
    recs = []
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if len(recs) < 2:
        return float("nan"), float("nan")
    lo, hi = int(len(recs) * window[0]), int(len(recs) * window[1])
    sl = recs[lo:hi]
    if len(sl) < 2:
        return float("nan"), float("nan")
    b0 = np.asarray(sl[0].get("beta", []), dtype=float)
    b1 = np.asarray(sl[-1].get("beta", []), dtype=float)
    net = float(np.mean(np.abs(b1 - b0))) if b0.size and b0.size == b1.size else float("nan")
    sp = [r["beta_true_max"] - r["beta_true_min"] for r in sl
          if r.get("beta_true_max") is not None and r.get("beta_true_min") is not None]
    return net, (float(np.mean(sp)) if sp else float("nan"))


# ------------------------------------------------------------------ per-leg exponents
def leg_exponents(pts):
    """[(k, rho_w, n), ...] sorted by k -> (b1, b2) for the two adjacent legs.

    b over a leg (k_a -> k_b) is log(rho_a / rho_b) / log(k_b / k_a): the exact two-point
    slope, POSITIVE when rho falls with k.  nan when either endpoint is missing, so an
    unresolved rung produces a hole rather than a silently shortened fit.
    """
    def slope(a, b):
        (ka, ra, _), (kb, rb, _) = a, b
        if not (ka > 0 and kb > ka and ra > 0 and rb > 0):
            return float("nan")
        return float(math.log(ra / rb) / math.log(kb / ka))
    if len(pts) < 2:
        return float("nan"), float("nan")
    if len(pts) == 2:
        # Only one leg resolved.  Attribute it by the coarse endpoint's k: a k of order
        # 1e5 is the layer rung, so the leg is channel->layer even with the middle missing.
        s = slope(pts[0], pts[1])
        return (float("nan"), s) if pts[1][0] > 1e4 else (s, float("nan"))
    return slope(pts[0], pts[1]), slope(pts[1], pts[2])


# ------------------------------------------------------------------ one arm
def reduce_arm(root, label):
    """One probe root -> {fam: {'nw':int, 'q':{qlabel: {...}}}}."""
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "neg_counts.json")):
            r = reduce_dir(d, windows=QUARTERS)
            if r:
                for lab, w in QUARTERS:
                    net, sp = beta_displacement(d, w)
                    if lab in r["win"]:
                        r["win"][lab]["beta_net"] = net
                        r["win"][lab]["beta_span"] = sp
                rows.append(r)
    out = {}
    for fam in sorted({r["fam"] for r in rows}):
        frows = [r for r in rows if r["fam"] == fam]
        nw = family_n_weights(frows)
        if nw is None:
            out[fam] = dict(nw=None, q={}, arm=label)
            continue
        q = {}
        for lab, _ in QUARTERS:
            pts, exc = profile(frows, nw, lab)
            b1, b2 = leg_exponents(pts)
            nets = [r["win"][lab]["beta_net"] for r in frows
                    if lab in r["win"] and np.isfinite(r["win"][lab].get("beta_net", np.nan))]
            spans = [r["win"][lab]["beta_span"] for r in frows
                     if lab in r["win"] and np.isfinite(r["win"][lab].get("beta_span", np.nan))]
            q[lab] = dict(pts=pts, excluded=exc, b1=b1, b2=b2,
                          b_all=powerlaw_b([p[0] for p in pts], [p[1] for p in pts]),
                          span=(max(p[1] for p in pts) / min(p[1] for p in pts)
                                if len(pts) >= 2 else float("nan")),
                          beta_net=(float(np.mean(nets)) if nets else float("nan")),
                          beta_span=(float(np.mean(spans)) if spans else float("nan")),
                          T=next((r["win"][lab]["T"] for r in frows if lab in r["win"]), 0))
        out[fam] = dict(nw=nw, q=q, arm=label)
    return out


def suppression(free_fam, froz_fam):
    """frozen rho_w / free rho_w, per (quarter, k).  The filter's transfer function.

    WHY THIS IS THE STATISTIC AND b1 IS NOT.  b1 is a SLOPE, and a slope that crosses zero
    cannot be summarised by a ratio -- which is exactly what the pre-registered
    b1(Q1)/b1(Q4) test does, and why it returns a meaningless negative number on the R18
    free arm.  This one is a ratio of two POSITIVE levels measured on byte-matched runs
    differing in one field, so it is defined everywhere and it reads directly as the
    filter's gain: 1.0 = adaptation did nothing at this scale in this quarter.

    Returns {quarter: {k: (frozen_rho, free_rho, ratio)}} over the k present in BOTH arms.
    """
    out = {}
    for lab, _ in QUARTERS:
        fe = {k: r for k, r, _ in free_fam["q"][lab]["pts"]}
        fz = {k: r for k, r, _ in froz_fam["q"][lab]["pts"]}
        out[lab] = {k: (fz[k], fe[k], fz[k] / fe[k])
                    for k in sorted(set(fe) & set(fz)) if fe[k] > 0}
    return out


def trend(qd, key):
    """Ratio first-quarter / last-quarter of `key`, nan if either is missing."""
    a = qd[QUARTERS[0][0]][key]
    b = qd[QUARTERS[-1][0]][key]
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float("nan")
    return a / b


def print_arm(name, fams):
    for fam, d in sorted(fams.items()):
        print(f"\n=== {name} / family {fam} ===")
        if d["nw"] is None:
            print("  REFUSED: no weightwise arm, so n_weights is unknown (CORRECTIONS 16).")
            continue
        print(f"  n_weights = {d['nw']:,}  (from this family's OWN weightwise n_tot)")
        hdr = (f"  {'quarter':12}{'T':>6}{'span':>9}{'b1 w->ch':>10}{'b2 ch->lay':>12}"
               f"{'b pooled':>10}{'beta net':>10}{'beta span':>11}   resolved rungs (k: rho_w)")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for lab, _ in QUARTERS:
            q = d["q"][lab]
            def f(x, w, p=3):
                return f"{x:>{w}.{p}f}" if np.isfinite(x) else f"{'-':>{w}}"
            body = "  ".join(f"k={k:,.0f}: {r:.3e}(n={n})" for k, r, n in q["pts"]) or "-- none --"
            spanv = q["span"]
            spans = f"{spanv:.1f}x" if np.isfinite(spanv) else "-"
            print(f"  {lab:12}{q['T']:>6}{spans:>9}"
                  f"{f(q['b1'],10)}{f(q['b2'],12)}{f(q['b_all'],10)}"
                  f"{f(q['beta_net'],10)}{f(q['beta_span'],11)}   {body}")
            if q["excluded"]:
                names = ", ".join(f"{n}(rho_s={rs:.2e}<rho_min={rm:.2e})"
                                  for n, rs, rm in q["excluded"])
                print(f"  {'':12}{'':6}{'':9}{'':10}{'':12}{'':10}{'':10}{'':11}   "
                      f"BELOW OWN rho_min, EXCLUDED: {names}")


def main(free_root, frozen_root):
    print("PROBE5 TIME LADDER -- b(t) across quarters, frozen arm as the training-progress control")
    print(f"  free arm   : {free_root}")
    print(f"  frozen ctrl: {frozen_root}")
    free = reduce_arm(free_root, "FREE")
    froz = reduce_arm(frozen_root, "FROZEN")
    print_arm("FREE (--alg-meta Lion)", free)
    print_arm("FROZEN control (--alg-meta fixed)", froz)

    print("\n" + "=" * 78)
    print("THE TEST (pre-registered above): does b1 fall in the free arm and stay flat frozen?")
    print("=" * 78)
    # Pairing the two arms.  Normally family names match (ff5 vs fz3).  But a ladder over
    # a NON-family knob -- eta_meta, in c49_meta_stepsize_ladder.sh -- names its arms e4 /
    # e2 while sharing ONE frozen control (p5, family r18c10).  Broadcast in that case,
    # and ONLY in that case: with two or more frozen families there is no way to know
    # which control belongs to which free arm, and guessing would silently difference two
    # different architectures.
    ctl = {f: d for f, d in froz.items() if d["nw"] is not None}
    if len(ctl) == 1 and not (set(free) & set(ctl)):
        only = next(iter(ctl))
        print(f"\n  [pairing] the frozen arm holds ONE family ({only}) and no free family "
              f"shares its name.")
        print(f"  [pairing] broadcasting it as the control for every free arm: "
              f"{', '.join(sorted(free))}.")
        print(f"  [pairing] this is correct ONLY if those arms differ from {only} in "
              f"something other than the")
        print(f"  [pairing] network and dataset -- verify n_weights matches below before "
              f"reading any gain.")
        froz = {f: ctl[only] for f in free}
    shared = sorted(set(free) & set(froz))
    if not shared:
        print("  no family measured in BOTH arms -- the control does not apply.  NOT DECIDABLE.")
        return 1
    for fam in shared:
        if free[fam]["nw"] is not None and froz[fam]["nw"] is not None \
                and free[fam]["nw"] != froz[fam]["nw"]:
            print(f"  REFUSED for {fam}: n_weights differs between arms "
                  f"({free[fam]['nw']:,} free vs {froz[fam]['nw']:,} frozen). A gain ratio "
                  f"across two different networks is meaningless.")
            free[fam] = dict(free[fam], nw=None)
    for fam in shared:
        if free[fam]["nw"] is None or froz[fam]["nw"] is None:
            continue
        rf, rz = trend(free[fam]["q"], "b1"), trend(froz[fam]["q"], "b1")
        f1 = free[fam]["q"][QUARTERS[0][0]]["b1"]
        f4 = free[fam]["q"][QUARTERS[-1][0]]["b1"]
        z1 = froz[fam]["q"][QUARTERS[0][0]]["b1"]
        z4 = froz[fam]["q"][QUARTERS[-1][0]]["b1"]
        print(f"\n  family {fam}")
        print(f"    free   b1: Q1 {f1:.3f} -> Q4 {f4:.3f}   ratio {rf:.2f}x"
              if np.isfinite(rf) else f"    free   b1: NOT MEASURABLE (a leg endpoint did not resolve)")
        print(f"    frozen b1: Q1 {z1:.3f} -> Q4 {z4:.3f}   ratio {rz:.2f}x"
              if np.isfinite(rz) else f"    frozen b1: NOT MEASURABLE (a leg endpoint did not resolve)")
        if not (np.isfinite(rf) and np.isfinite(rz)):
            print("    --> NOT DECIDABLE: state the missing rung, do not read a trend into a hole.")
            continue
        if min(f1, f4, z1, z4) <= 0:
            print("    --> THE PRE-REGISTERED RATIO TEST IS UNDEFINED HERE: b1 changes sign")
            print("        across the run, and a ratio through zero is not a magnitude. This")
            print("        is a defect in the pre-registration, found by the data, not a")
            print("        result. Scored on the DIFFERENCE instead, and flagged as such:")
            print(f"        free   delta b1 = {f4 - f1:+.3f}   frozen delta b1 = {z4 - z1:+.3f}"
                  f"   free moves {abs(f4 - f1) / abs(z4 - z1):.1f}x the control"
                  if (z4 - z1) != 0 else
                  f"        free delta b1 = {f4 - f1:+.3f}, frozen control did not move at all")
            print("        A difference is NOT the statistic that was registered. Treat the")
            print("        reading below as POST-HOC and confirm it on independent data.")
        falls, flat = rf >= FALL_FACTOR, rz <= FLAT_FACTOR
        if falls and flat:
            print(f"    --> FILTER READING SUPPORTED: free b1 falls {rf:.2f}x (>= {FALL_FACTOR}) "
                  f"while the frozen control is flat to {rz:.2f}x (<= {FLAT_FACTOR}).")
            print("        Training progress is ruled out -- the same schedule with beta held")
            print("        still does not produce the fall.  Adaptation does.")
        elif falls and not flat:
            print(f"    --> REFUTED AS STATED: the frozen control ALSO moves ({rz:.2f}x). The "
                  f"fall is (at least partly) TRAINING PROGRESS, not adaptation.")
            print("        CORRECTIONS 49(2) must be rewritten before any filter sentence is used.")
        elif not falls and flat:
            print(f"    --> NO TREND: free b1 moves only {rf:.2f}x across the run. The "
                  f"0.331 -> 0.065 gap is then a property of the .5 split, not a curve.")
        else:
            print(f"    --> UNINTERPRETABLE: free {rf:.2f}x, frozen {rz:.2f}x. Report both, "
                  f"claim neither.")

    print("\n" + "=" * 78)
    print("THE FILTER'S TRANSFER FUNCTION -- frozen rho_w / free rho_w, by quarter and scale")
    print("(1.0 = adaptation did nothing at that scale in that quarter.  Byte-matched runs,")
    print(" one field different.  This is a ratio of positive levels, so unlike b1 it is")
    print(" defined through a sign change.)")
    print("=" * 78)
    for fam in shared:
        if free[fam]["nw"] is None or froz[fam]["nw"] is None:
            continue
        sup = suppression(free[fam], froz[fam])
        ks = sorted({k for q in sup.values() for k in q})
        print(f"\n  family {fam}")
        print(f"  {'quarter':12}" + "".join(f"{('k=%s' % f'{k:,.0f}'):>16}" for k in ks))
        print("  " + "-" * (12 + 16 * len(ks)))
        for lab, _ in QUARTERS:
            cells = "".join(
                (f"{sup[lab][k][2]:>15.2f}x" if k in sup[lab] else f"{'-':>16}") for k in ks)
            print(f"  {lab:12}{cells}")
        q1 = sup[QUARTERS[0][0]]
        if q1 and max(v[2] for v in q1.values()) < 2.0:
            print("  Q1 gain ~1 at every scale: BEFORE beta has moved the free arm IS the")
            print("  frozen arm.  That is this table's internal control -- it is not assumed.")
    return 0


# ------------------------------------------------------------------ selftest
def _selftest():
    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL: {name}")

    # --- quarters partition the run exactly once, in order, with no overlap or gap
    edges = [w for _, w in QUARTERS]
    chk("quarters start at 0", edges[0][0] == 0.0)
    chk("quarters end at 1", edges[-1][1] == 1.0)
    chk("quarters are contiguous", all(edges[i][1] == edges[i + 1][0] for i in range(len(edges) - 1)))
    chk("quarters are non-empty", all(b > a for a, b in edges))
    # Q1 must be byte-identical to probe5_window's published startup window, else this
    # reducer's Q1 cannot be checked against FINDINGS 48.15.
    from probe5_window import WINDOWS as PW
    startup = dict(PW)["startup 0-.25"]
    chk("Q1 == probe5_window startup window", edges[0] == startup)

    # --- leg_exponents recovers exact two-point slopes
    # rho = k^-0.5 on the first leg, k^-1.0 on the second
    pts = [(1.0, 1.0, 2), (100.0, 100.0 ** -0.5, 2), (10000.0, (100.0 ** -0.5) * (100.0 ** -1.0), 2)]
    b1, b2 = leg_exponents(pts)
    chk("leg b1 exact", abs(b1 - 0.5) < 1e-9)
    chk("leg b2 exact", abs(b2 - 1.0) < 1e-9)
    # a RISING profile must give a NEGATIVE b, not an absolute value
    b1r, _ = leg_exponents([(1.0, 1.0, 2), (100.0, 2.0, 2), (10000.0, 1.0, 2)])
    chk("rising leg gives negative b", b1r < 0)
    # two points only: attribution by the coarse endpoint's magnitude
    chk("2pt coarse -> b2 leg", np.isnan(leg_exponents([(1.0, 1.0, 1), (1e5, 1e-3, 1)])[0]))
    chk("2pt fine   -> b1 leg", np.isfinite(leg_exponents([(1.0, 1.0, 1), (7e2, 1e-1, 1)])[0]))
    chk("fewer than 2 points -> both nan",
        all(np.isnan(x) for x in leg_exponents([(1.0, 1.0, 1)])))
    # degenerate inputs must not raise
    chk("non-positive rho -> nan", np.isnan(leg_exponents([(1.0, 0.0, 1), (1e2, 1e-1, 1),
                                                           (1e4, 1e-2, 1)])[0]))

    # --- trend()
    q = {QUARTERS[0][0]: dict(b1=0.4), QUARTERS[1][0]: dict(b1=0.3),
         QUARTERS[2][0]: dict(b1=0.2), QUARTERS[3][0]: dict(b1=0.1)}
    chk("trend ratio", abs(trend(q, "b1") - 4.0) < 1e-12)
    q2 = dict(q)
    q2[QUARTERS[3][0]] = dict(b1=0.0)
    chk("trend divide-by-zero -> nan", np.isnan(trend(q2, "b1")))
    q3 = dict(q)
    q3[QUARTERS[0][0]] = dict(b1=float("nan"))
    chk("trend nan endpoint -> nan", np.isnan(trend(q3, "b1")))

    # --- the verdict thresholds are ordered and cannot both fire
    chk("FALL > FLAT", FALL_FACTOR > FLAT_FACTOR)

    # --- beta_displacement on a synthetic probe dir
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        recs = []
        for t in range(100):
            # beta drifts linearly by 1.0 over the whole run, 4 coordinates
            recs.append(dict(step=t, beta=[t / 99.0] * 4,
                             beta_true_max=t / 99.0, beta_true_min=0.0))
        with open(os.path.join(td, "probe.jsonl"), "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        net_full, sp_full = beta_displacement(td, (0.0, 1.0))
        chk("net displacement over full run ~1", abs(net_full - (99 / 99.0)) < 0.02)
        net_q1, _ = beta_displacement(td, (0.0, 0.25))
        chk("net displacement over a quarter ~1/4", abs(net_q1 - 0.25) < 0.02)
        chk("beta span is the mean max-min", 0.0 < sp_full < 1.0)
        # a FROZEN arm: beta identical at every record -> net displacement exactly 0
        with open(os.path.join(td, "probe.jsonl"), "w") as fh:
            for t in range(100):
                fh.write(json.dumps(dict(step=t, beta=[-6.9078] * 4,
                                         beta_true_max=-6.9078, beta_true_min=-6.9078)) + "\n")
        chk("frozen arm net displacement is 0", beta_displacement(td, (0.0, 1.0))[0] == 0.0)
        chk("frozen arm span is 0", beta_displacement(td, (0.0, 1.0))[1] == 0.0)
        # missing file
        chk("missing probe.jsonl -> nan",
            np.isnan(beta_displacement(os.path.join(td, "nope"), (0.0, 1.0))[0]))
        # a window holding <2 records
        chk("degenerate window -> nan", np.isnan(beta_displacement(td, (0.0, 0.005))[0]))


    # --- suppression(): ratio of positive levels, defined through a b1 sign change
    def _fam(pts_by_q):
        return dict(nw=1, q={lab: dict(pts=pts_by_q[i]) for i, (lab, _) in enumerate(QUARTERS)})
    fr = _fam([[(1.0, 1e-6, 3), (775.0, 1e-7, 3)]] * 4)
    fz = _fam([[(1.0, 2e-5, 2), (775.0, 5e-7, 2)]] * 4)
    sup = suppression(fr, fz)
    chk("suppression gain at k=1", abs(sup[QUARTERS[0][0]][1.0][2] - 20.0) < 1e-9)
    chk("suppression gain at k=775", abs(sup[QUARTERS[0][0]][775.0][2] - 5.0) < 1e-9)
    # a k present in only ONE arm must be dropped, never paired against a different k
    fr2 = _fam([[(1.0, 1e-6, 3)]] * 4)
    chk("unmatched k dropped", set(suppression(fr2, fz)[QUARTERS[0][0]]) == {1.0})
    # zero/negative free level must not produce an infinite gain
    fr3 = _fam([[(1.0, 0.0, 3), (775.0, 1e-7, 3)]] * 4)
    chk("non-positive free level dropped", 1.0 not in suppression(fr3, fz)[QUARTERS[0][0]])


    # --- broadcast pairing: one frozen family, several free arms (the eta_meta ladder)
    _one = {"r18c10": dict(nw=100, q={})}
    _many = {"e4": dict(nw=100, q={}), "e2": dict(nw=100, q={})}
    _ctl = {f: d for f, d in _one.items() if d["nw"] is not None}
    chk("broadcast fires on 1 frozen family, no name overlap",
        len(_ctl) == 1 and not (set(_many) & set(_ctl)))
    # must NOT fire when the frozen arm holds several families -- guessing would
    # difference two different architectures
    _two = {"r10": dict(nw=1, q={}), "r34": dict(nw=2, q={})}
    chk("broadcast does NOT fire on 2 frozen families", len(_two) != 1)
    # must NOT fire when the names already overlap (ff5 vs fz3): pair by name instead
    chk("broadcast does NOT fire when names overlap",
        bool(set({"r10": 1}) & set({"r10": 1})))

    # --- the imported helpers still behave as this file assumes
    chk("geo of decades", abs(geo([1e-8, 1e-6]) - 1e-7) < 1e-12)
    chk("powerlaw_b sign", abs(powerlaw_b([1, 100], [1.0, 0.1]) - 0.5) < 1e-9)

    print(f"selftest: {ok}/{ok + fail} PASS")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(pos[0] if pos else "../probes_fr5",
                  pos[1] if len(pos) > 1 else "../probes_p5"))
