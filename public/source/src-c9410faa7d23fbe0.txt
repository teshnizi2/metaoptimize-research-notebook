"""Drive analysis/twochannel.py over every probe batch on disk.

Answers the question CORRECTIONS 33 left open: is the frozen-beta N_eff deficit
(s = 0.629, N/N_eff up to 200) a MARGINAL-BIAS failure or a COMMON-MODE failure?

Run the validation suite first -- numbers from this script are not to be trusted otherwise:
    python3 tests/test_twochannel.py
"""
import os, sys, glob, collections, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twochannel import ROOT, arm, parse_gran, GRAN_ORDER, GRAN_NAME  # noqa: E402

BATCHES = ["fz", "p7free", "p6free", "mx", "gate3", "kt2"]
WINDOW = (0.5, 1.0)          # steady half; agree2.py's convention (startup is a transient)


def collect(batch, window):
    root = os.path.join(ROOT, batch)
    by = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        g, fam = parse_gran(os.path.basename(d))
        if g is None:
            continue
        try:
            r = arm(d, window=window)
        except Exception:
            continue
        if r:
            by[(fam, g)].append(r)
    return by


def agg(v, k):
    a = np.array([x[k] for x in v], dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0)


def table(batch, by):
    """Significance is reported SEED-LEVEL (t_seed = mean/se over independent runs).  The
    within-run z is autocorrelation-corrected but still assumes the record series is the
    only source of error; the seed-level t absorbs run-to-run variation as well and is the
    number to quote.  n=1 cells cannot carry one and are marked '--'."""
    if not by:
        return
    print("\n" + "=" * 134)
    print(f"BATCH {batch}   window={WINDOW}   (share_* are fractions of E[(p-1/2)^2])")
    print("=" * 134)
    for fam in sorted({k[0] for k in by}):
        gs = [g for g in GRAN_ORDER if by.get((fam, g))]
        if not gs:
            continue
        print(f"\n  family = {fam or 'R18/CIFAR-10'}")
        h = (f"  {'gran':6s} {'n (nonzero)':>13s} {'sd':>3s} {'froz':>5s} "
             f"{'bias pp':>9s} {'rho_s':>11s} {'rho_fast':>11s} {'rho_min':>10s} "
             f"{'t_seed':>7s} {'tau':>5s} {'%bias':>7s} {'%com':>7s} {'%indep':>7s} {'N/Neff':>9s}")
        print(h)
        print("  " + "-" * (len(h) - 2))
        for g in gs:
            v = by[(fam, g)]
            nbar, _ = agg(v, "nbar")
            bias, _ = agg(v, "bias")
            rho, rho_se = agg(v, "rho_s")
            rfast, _ = agg(v, "rho_fast")
            rmin, _ = agg(v, "rho_min")
            tau, _ = agg(v, "tau")
            sb, _ = agg(v, "share_bias")
            sc, _ = agg(v, "share_common")
            si, _ = agg(v, "share_indep")
            noe, _ = agg(v, "n_over_neff")
            froz = "yes" if all(x.get("frozen") for x in v) else "no"
            ts = f"{rho/rho_se:>7.1f}" if len(v) > 1 and rho_se > 0 else "     --"
            print(f"  {g:6s} {nbar:>13,.0f} {len(v):>3d} {froz:>5s} "
                  f"{100*bias:>9.4f} {rho:>11.3e} {rfast:>11.3e} {rmin:>10.2e} "
                  f"{ts} {tau:>5.1f} {100*sb:>7.2f} {100*sc:>7.2f} {100*si:>7.2f} {noe:>9.4g}")


def main():
    for b in BATCHES:
        if not os.path.isdir(os.path.join(ROOT, b)):
            continue
        table(b, collect(b, WINDOW))


if __name__ == "__main__":
    main()
