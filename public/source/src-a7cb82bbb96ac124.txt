#!/usr/bin/env python3
r"""WHY THE TWO `s` INSTRUMENTS DISAGREE -- the bias channel, tested directly.

THE OPEN ITEM.  FINDINGS 48.13 recorded, and 48.19 / CORRECTIONS 51 sharpened, a gap
between two estimators of the exponent `s` in N_eff ~ m^s measured ON THE SAME RUNS:

    family   agreement-derived s   variance-derived s (steady)   gap
    c100           0.572                    0.826               +0.254
    r10            0.613                    0.722               +0.109
    r18c10         0.629                    0.765               +0.136
    r34            0.738                    0.823               +0.085

and, worse than the size of the gap, THE TWO REVERSE THE FAMILY ORDERING -- c100 is last by
one instrument and first by the other.  CORRECTIONS 51 therefore forbade using `s` to rank
architectures.  Two candidate causes were named and one was eliminated (it is not the
heterogeneity correction: matching the null moved the mean gap only +0.097 -> +0.090).
CORRECTIONS 52(6) left the window as the next suspect.

THE WINDOW IS NOT IT EITHER, AND THAT IS SETTLED BY READING THE CODE.
`frozen_agreement.arm_stats` defaults to `window=0.5`, i.e. it ALREADY reduces the steady
half -- the same window `probe5_window.py` quotes.  The table above is window-matched, and
matching it makes the gap LARGER (+0.146 mean) than the full-run comparison (+0.097).  So
the window candidate is eliminated, not pending.

THE ACTUAL CANDIDATE, WHICH NOBODY HAD NAMED: THE TWO ESTIMATORS DO NOT MEASURE THE SAME
DEVIATION.  Writing p_t for the fraction of negative meta-gradient signs at record t:

    agreement (frozen_agreement.neff_from_agreement)
        A = mean_t max(p_t, 1-p_t) = 0.5 + mean_t |p_t - 0.5|
        ... deviation measured from the FIXED POINT 0.5

    variance  (twochannel.decompose -> probe5_floor.recompute_rho)
        v_p = Var_t(p_t)  (numpy var, ddof=1)
        ... deviation measured from THE SAMPLE MEAN pbar

The difference is exactly the BIAS CHANNEL b = pbar - 0.5, which twochannel already
isolates as `v_bias` and reports as `share_bias`.  The variance instrument excludes it by
construction.  The agreement instrument folds it in.  And FINDINGS 44.3 measured the frozen
bias share at **85% of total deviation** -- which is the regime every number in the table
above was measured in.  If b dominates, the agreement instrument is largely reading the
bias channel while the variance instrument reads only the fluctuation, and any dependence
of b on granularity or family goes straight into `s` as a spurious difference.

PRE-REGISTERED PREDICTIONS.  Written and committed BEFORE the reducer was first run on real
data, so that a confirmation is not a story fitted afterwards.

  (E1) DEBIASING CLOSES THE GAP.  Replacing mean_t|p_t - 0.5| with mean_t|p_t - pbar| --
       one character of statistics, the same inversion, the same window, the same runs --
       moves the agreement-derived s TOWARD the variance-derived s in 4 of 4 families, and
       cuts the mean gap from +0.146 to below +0.05.
  (E2) THE ORDERING STOPS REVERSING.  Under the debiased agreement instrument, c100 is no
       longer the LOWEST of the four; the debiased ordering matches the variance ordering.
  REFUTATION of both: if the debiased gap is not smaller, or the ordering does not move,
       the bias channel is NOT the cause, and 48.13 stays open on its last candidate --
       that N_eff = m/(1+(m-1)rho_s) is a first-order expansion.  Write it as that.

  WHAT WOULD STILL BE LEFT EVEN IF E1 AND E2 BOTH CONFIRM, and it must be said rather than
  buried: the debiased agreement instrument measures a FIRST absolute moment
  (mean|p - pbar|) while the variance instrument measures a SECOND (Var(p)).  Those two
  agree only if p_t is Gaussian -- E|X - EX| = sqrt(2/pi) * sd(X).  Any residual gap after
  debiasing is therefore a measure of NON-GAUSSIANITY in p_t, not an instrument bug, and it
  is reported as such.  This is why E1's threshold is "below +0.05" and not "zero".

Run `--selftest` before trusting any number this prints.
"""
import glob
import math
import re
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frozen_agreement import neff_from_agreement  # noqa: E402
from probe5_window import parse_dirname, reduce_dir  # noqa: E402
from twochannel import load  # noqa: E402

STEADY = (("steady .5-1", (0.5, 1.0)),)
RUNGS = ("blk6", "lay", "node", "w")
SQ2PI = math.sqrt(2.0 / math.pi)

# E1's threshold: the mean |gap| the debiased instrument must beat.
E1_TARGET = 0.05


def agreement_stats(d, window=(0.5, 1.0)):
    """One probe dir -> the RAW and DEBIASED agreement statistics on `window`.

    A_raw = mean_t max(p_t, 1-p_t)          deviation from the fixed point 0.5
    A_deb = 0.5 + mean_t |p_t - pbar|       deviation from the sample mean

    Both are then pushed through the SAME exact inversion, so any difference between the
    two N_eff values is the bias channel and nothing else.
    """
    recs = load(d)
    if len(recs) < 8:
        return None
    fn = np.asarray([r.get("frac_neg") for r in recs], dtype=float)
    fz = np.asarray([r.get("frac_zero", 0.0) for r in recs], dtype=float)
    if not np.all(np.isfinite(fn)):
        return None
    lo, hi = int(len(fn) * window[0]), int(len(fn) * window[1])
    fn, fz = fn[lo:hi], fz[lo:hi]
    keep = (1.0 - fz) > 0
    fn, fz = fn[keep], fz[keep]
    if len(fn) < 8:
        return None
    p = fn / (1.0 - fz)
    pbar = float(np.mean(p))
    a_raw = float(np.mean(np.maximum(p, 1.0 - p)))
    a_deb = 0.5 + float(np.mean(np.abs(p - pbar)))
    dev_raw = a_raw - 0.5
    dev_deb = a_deb - 0.5
    return dict(T=len(p), pbar=pbar, bias=pbar - 0.5,
                a_raw=a_raw, a_deb=a_deb,
                neff_raw=neff_from_agreement(a_raw),
                neff_deb=neff_from_agreement(a_deb),
                # what fraction of the RAW deviation is the bias channel.  1.0 means the
                # agreement instrument is reading the bias and nothing else.
                bias_share=(1.0 - dev_deb / dev_raw) if dev_raw > 0 else float("nan"))


def neff_from_rho(m, rho_s):
    """N_eff = m / (1 + (m-1) rho_s), the exchangeable-model effective sample size."""
    if not np.isfinite(rho_s) or m < 1:
        return float("nan")
    den = 1.0 + (m - 1.0) * rho_s
    return m / den if den > 0 else float("nan")


def fit_s(ms, neffs):
    """Slope of log10(N_eff) vs log10(m).  Needs >= 3 distinct m, matching the campaign's
    convention in frozen_agreement.powerlaw and neff_ladder."""
    x, y = [], []
    for m, n in zip(ms, neffs):
        if np.isfinite(m) and np.isfinite(n) and m > 0 and n > 0:
            x.append(math.log10(m))
            y.append(math.log10(n))
    if len(set(x)) < 3:
        return float("nan")
    return float(np.polyfit(np.asarray(x), np.asarray(y), 1)[0])


def reduce_root(root, window=None):
    """probe root -> {family: {rung: {...}}}, steady half, seeds averaged.

    `window` is an OPTION, added for the cycle-52 `bl5` budget ladder's B1.5
    replication: a 40-epoch run restricted to records 1000-2000 is the SAME
    ABSOLUTE window of training as the whole steady half of a 20-epoch control,
    and without that arm B1 is a seed contrast rather than a budget contrast.
    The DEFAULT is unchanged (steady half, 0.5-1.0) -- every number the campaign
    has quoted comes from it, and a changed default would silently re-derive them.
    """
    if window is None:
        window, wname = (0.5, 1.0), "steady .5-1"
    else:
        wname = f"win {window[0]:g}-{window[1]:g}"
    wins = ((wname, tuple(window)),)
    acc = {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "neg_counts.json")):
            continue
        fam, rung, _ = parse_dirname(os.path.basename(d))
        if rung not in RUNGS:
            continue
        ag = agreement_stats(d, window=tuple(window))
        if ag is None:
            continue
        r5 = reduce_dir(d, windows=wins)
        w = r5["win"].get(wname) if r5 else None
        acc.setdefault(fam, {}).setdefault(rung, []).append(dict(
            m=float(r5["n_tot"]) if r5 else float("nan"),
            neff_raw=ag["neff_raw"], neff_deb=ag["neff_deb"],
            bias=ag["bias"], bias_share=ag["bias_share"],
            rho_s=(w["rho_s"] if w else float("nan")),
            resolved=(bool(w["resolved"]) if w else False)))
    out = {}
    for fam, rungs in acc.items():
        out[fam] = {}
        for rung, vs in rungs.items():
            m = float(np.mean([v["m"] for v in vs]))
            rho = float(np.mean([v["rho_s"] for v in vs]))
            out[fam][rung] = dict(
                m=m, n=len(vs),
                neff_raw=float(np.mean([v["neff_raw"] for v in vs])),
                neff_deb=float(np.mean([v["neff_deb"] for v in vs])),
                neff_var=neff_from_rho(m, rho),
                bias=float(np.mean([v["bias"] for v in vs])),
                bias_share=float(np.mean([v["bias_share"] for v in vs])),
                rho_s=rho,
                resolved=all(v["resolved"] for v in vs))
    return out


def main(root, window=None):
    fams = reduce_root(root, window=window)
    if not fams:
        print(f"no reducible probe dirs under {root}")
        return 1
    wlabel = ("steady half, matching frozen_agreement.py's default" if window is None
              else f"{window[0]:g}-{window[1]:g} of the run (EXPLICIT --window)")
    print("=" * 96)
    print("WHY THE TWO `s` INSTRUMENTS DISAGREE -- testing the BIAS CHANNEL (FINDINGS 48.13)")
    print(f"  root = {root}   window = {wlabel}")
    print("=" * 96)

    rows = []
    for fam in sorted(fams):
        rg = [g for g in RUNGS if g in fams[fam]]
        if len(rg) < 3:
            print(f"\n=== {fam} ===  SKIPPED: only {len(rg)} rungs, `s` needs >= 3.")
            continue
        print(f"\n=== family {fam} ===")
        hdr = (f"  {'rung':6}{'m':>12}{'n':>3}{'bias b':>11}{'bias share':>12}"
               f"{'N_eff raw':>13}{'N_eff debias':>14}{'N_eff var':>13}{'res':>5}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for g in rg:
            v = fams[fam][g]
            print(f"  {g:6}{v['m']:>12,.0f}{v['n']:>3}{v['bias']:>11.2e}"
                  f"{v['bias_share']:>12.3f}{v['neff_raw']:>13,.1f}"
                  f"{v['neff_deb']:>14,.1f}{v['neff_var']:>13,.1f}"
                  f"{('yes' if v['resolved'] else 'NO'):>5}")
        ms = [fams[fam][g]["m"] for g in rg]
        s_raw = fit_s(ms, [fams[fam][g]["neff_raw"] for g in rg])
        s_deb = fit_s(ms, [fams[fam][g]["neff_deb"] for g in rg])
        s_var = fit_s(ms, [fams[fam][g]["neff_var"] for g in rg])
        print(f"    s (agreement, RAW)      = {s_raw:.3f}"
              f"      gap to variance = {s_raw - s_var:+.3f}")
        print(f"    s (agreement, DEBIASED) = {s_deb:.3f}"
              f"      gap to variance = {s_deb - s_var:+.3f}")
        print(f"    s (variance)            = {s_var:.3f}")
        rows.append((fam, s_raw, s_deb, s_var))

    if len(rows) < 2:
        print("\nfewer than 2 families -- E1/E2 are not scorable here.")
        return 0

    print("\n" + "=" * 96)
    print("SCORING THE PRE-REGISTRATION (E1, E2)")
    print("=" * 96)
    gaps_raw = [abs(r[1] - r[3]) for r in rows]
    gaps_deb = [abs(r[2] - r[3]) for r in rows]
    closer = sum(1 for r in rows if abs(r[2] - r[3]) < abs(r[1] - r[3]))
    print(f"  {'family':10}{'s raw':>9}{'s debiased':>13}{'s variance':>13}"
          f"{'|gap| raw':>12}{'|gap| deb':>12}")
    for fam, sr, sd_, sv in rows:
        print(f"  {fam:10}{sr:>9.3f}{sd_:>13.3f}{sv:>13.3f}"
              f"{abs(sr - sv):>12.3f}{abs(sd_ - sv):>12.3f}")
    mg_raw, mg_deb = float(np.mean(gaps_raw)), float(np.mean(gaps_deb))
    print(f"  {'MEAN':10}{'':>9}{'':>13}{'':>13}{mg_raw:>12.3f}{mg_deb:>12.3f}")

    print(f"\n  (E1) debiasing moves s toward the variance instrument in "
          f"{closer} of {len(rows)} families; mean |gap| {mg_raw:.3f} -> {mg_deb:.3f} "
          f"(target < {E1_TARGET}).")
    e1 = (closer == len(rows)) and (mg_deb < E1_TARGET)
    print(f"       --> E1 {'CONFIRMED' if e1 else 'NOT CONFIRMED'}")

    def order(i):
        return [r[0] for r in sorted(rows, key=lambda r: r[i])]
    o_raw, o_deb, o_var = order(1), order(2), order(3)
    print(f"\n  (E2) family ordering, lowest s first:")
    print(f"       raw agreement : {' < '.join(o_raw)}")
    print(f"       debiased      : {' < '.join(o_deb)}")
    print(f"       variance      : {' < '.join(o_var)}")
    e2 = (o_deb == o_var) and (o_raw != o_var)
    if e2:
        print("       --> E2 CONFIRMED: debiasing reconciles the ordering the raw "
              "instrument reversed.")
    elif o_raw == o_var:
        print("       --> E2 NOT APPLICABLE on this root: the RAW ordering already "
              "matches, so there is no reversal to fix here.")
    else:
        print("       --> E2 NOT CONFIRMED: debiasing did not reconcile the ordering.")

    if e1 and e2:
        print("\n  VERDICT: the gap of FINDINGS 48.13 is the BIAS CHANNEL. The agreement")
        print("  instrument measures deviation from the fixed point 0.5 and so reads")
        print("  b = pbar - 0.5 together with the fluctuation; the variance instrument")
        print("  measures deviation from pbar and excludes b by construction. Any residual")
        print("  gap is non-Gaussianity in p_t (first vs second absolute moment), not a bug.")
    else:
        print("\n  VERDICT: the bias channel does NOT account for the gap on this root.")
        print("  48.13 stays open on its last candidate -- N_eff = m/(1+(m-1)rho_s) being a")
        print("  first-order expansion. Report it as open, not as explained.")
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

    rng = np.random.default_rng(0)

    # --- neff_from_rho
    chk("neff at rho=0 is m", abs(neff_from_rho(100, 0.0) - 100) < 1e-9)
    chk("neff at rho=1 is 1", abs(neff_from_rho(100, 1.0) - 1.0) < 1e-9)
    chk("neff falls with rho", neff_from_rho(100, 0.1) < neff_from_rho(100, 0.01))
    chk("neff nan on nan rho", np.isnan(neff_from_rho(100, float("nan"))))

    # --- fit_s recovers an exact power law
    ms = [10.0, 1000.0, 100000.0]
    chk("fit_s exact on m^0.6", abs(fit_s(ms, [m ** 0.6 for m in ms]) - 0.6) < 1e-9)
    chk("fit_s needs 3 distinct m", np.isnan(fit_s([10.0, 10.0, 10.0], [1.0, 1.0, 1.0])))
    chk("fit_s ignores non-positive", np.isnan(fit_s([10.0, 100.0], [1.0, 2.0])))

    # --- THE CORE CLAIM, on synthetic data with a KNOWN bias.
    # p_t = 0.5 + b + noise.  The debiased statistic must be blind to b; the raw one
    # must not.  This is the whole hypothesis, tested in isolation before any real data.
    import tempfile
    import json as _json

    def _write(dirpath, p_series):
        os.makedirs(dirpath, exist_ok=True)
        with open(os.path.join(dirpath, "probe.jsonl"), "w") as fh:
            for t, p in enumerate(p_series):
                fh.write(_json.dumps(dict(step=t * 5, frac_neg=float(p), frac_zero=0.0,
                                          beta_true_max=-6.9, beta_true_min=-6.9)) + "\n")

    with tempfile.TemporaryDirectory() as td:
        noise = rng.normal(0.0, 0.001, 4000)
        for tag, b in (("nobias", 0.0), ("bias", 0.02)):
            _write(os.path.join(td, tag), 0.5 + b + noise)
        a0 = agreement_stats(os.path.join(td, "nobias"))
        a1 = agreement_stats(os.path.join(td, "bias"))
        chk("debiased deviation is invariant to b",
            abs((a0["a_deb"] - 0.5) - (a1["a_deb"] - 0.5)) < 1e-6)
        chk("raw deviation grows with b", (a1["a_raw"] - 0.5) > 10 * (a0["a_raw"] - 0.5))
        chk("debiased N_eff invariant to b",
            abs(a0["neff_deb"] - a1["neff_deb"]) / a0["neff_deb"] < 1e-3)
        chk("raw N_eff collapses under b", a1["neff_raw"] < a0["neff_raw"] / 10)
        chk("bias_share ~0 with no bias", abs(a0["bias_share"]) < 0.05)
        chk("bias_share ~1 with dominant bias", a1["bias_share"] > 0.9)
        chk("reported bias equals the injected b", abs(a1["bias"] - 0.02) < 1e-3)

        # Gaussian consistency: for Gaussian p, mean|p-pbar| = sqrt(2/pi)*sd, so the
        # debiased agreement N_eff must match the variance-based one built from the SAME
        # series.  This is the residual term the docstring promises to report.
        sd = float(np.std(0.5 + noise, ddof=1))
        neff_gauss = (SQ2PI * 0.5 / (SQ2PI * sd)) ** 2
        chk("debiased agreement == Gaussian variance form",
            abs(a0["neff_deb"] - neff_gauss) / neff_gauss < 0.05)

        # a series with fewer than 8 usable records must be refused, not extrapolated
        _write(os.path.join(td, "tiny"), [0.5] * 4)
        chk("too-few-records refused", agreement_stats(os.path.join(td, "tiny")) is None)
        # frac_zero == 1 on every record must be refused too
        os.makedirs(os.path.join(td, "allzero"), exist_ok=True)
        with open(os.path.join(td, "allzero", "probe.jsonl"), "w") as fh:
            for t in range(100):
                fh.write(_json.dumps(dict(step=t, frac_neg=0.0, frac_zero=1.0)) + "\n")
        chk("all-zero records refused", agreement_stats(os.path.join(td, "allzero")) is None)

    # --- the window default matches frozen_agreement.arm_stats(window=0.5), i.e. the
    #     LAST half.  If these ever diverge the whole comparison is invalid.
    with tempfile.TemporaryDirectory() as td:
        ramp = np.concatenate([np.full(1000, 0.60), np.full(1000, 0.50)])
        _write(os.path.join(td, "ramp"), ramp)
        a = agreement_stats(os.path.join(td, "ramp"))
        chk("default window is the LAST half", abs(a["pbar"] - 0.50) < 1e-9)
        chk("window T is half the records", a["T"] == 1000)

    # --- the B1.5 option.  A 4000-record run sliced 0.25-0.5 must be records
    #     1000-2000 -- the SAME ABSOLUTE window as the whole steady half of a
    #     2000-record control.  Without this, B1 is a seed contrast.
    with tempfile.TemporaryDirectory() as td:
        # four quarters, each a distinct constant, so a slice is identifiable by value
        quarters = np.concatenate([np.full(1000, 0.60), np.full(1000, 0.70),
                                   np.full(1000, 0.80), np.full(1000, 0.90)])
        dq = os.path.join(td, "quarters")
        _write(dq, quarters)
        a25 = agreement_stats(dq, window=(0.25, 0.5))
        chk("--window 0.25-0.5 selects the SECOND quarter", abs(a25["pbar"] - 0.70) < 1e-9)
        chk("--window 0.25-0.5 keeps 1000 records", a25["T"] == 1000)
        a_def = agreement_stats(dq)
        chk("default still the last half (pbar of Q3+Q4)",
            abs(a_def["pbar"] - 0.85) < 1e-9)
        chk("explicit 0.5-1.0 reproduces the default exactly",
            abs(agreement_stats(dq, window=(0.5, 1.0))["pbar"] - a_def["pbar"]) < 1e-12)

    print(f"selftest: {ok}/{ok + fail} PASS")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    win = None
    for a in sys.argv[1:]:
        if a.startswith("--window"):
            spec = a.split("=", 1)[1] if "=" in a else sys.argv[sys.argv.index(a) + 1]
            lo, hi = (float(x) for x in spec.split("-"))
            if not (0.0 <= lo < hi <= 1.0):
                sys.exit(f"--window must satisfy 0 <= lo < hi <= 1, got {spec}")
            win = (lo, hi)
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    pos = [a for a in pos if not re.fullmatch(r"[\d.]+-[\d.]+", a)]
    sys.exit(main(pos[0] if pos else "../probes_fz3", window=win))
