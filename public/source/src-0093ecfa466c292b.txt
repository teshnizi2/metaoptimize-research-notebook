"""Is the meta-gradient's correlation the part that ADAPTATION CONSUMES?

Setup.  `fz-*-a3` (frozen beta) and `p7-r18-*` (free beta) are identical in every field --
SGDm base, Lion/fixed meta, ms=1e-3, alpha0=1e-3, CIFAR-10, ResNet18, 20 epochs, AUGMENT=1,
BETA_CLIP=-15:-2.3026, PROBE=100 -- except `--alg-meta`.  Their effective-sample-size
exponents disagree sharply:

    frozen beta   N_eff ~ m^0.629 +- 0.013   (n=5)
    free   beta   N_eff ~ m^0.963 +- 0.015   (n=10)

Two readings, and they are NOT separated by those two numbers alone:

  (A) ADAPTATION CONSUMES THE COMMON MODE.  The correlated part of the meta-gradient is
      "every coordinate's step size is wrong in the same direction".  A free beta absorbs
      exactly that part, so what is left at equilibrium is the idiosyncratic residual, which
      is near-independent.
  (B) HETEROGENEOUS ALPHA DECORRELATES MECHANICALLY.  In a free arm each group's alpha has
      drifted to its own value, so the coordinates sit at different points and their
      meta-gradients decorrelate for a reason that has nothing to do with what was consumed.

THE FREE RUNS SEPARATE THEM BY THEMSELVES.  At step 0 every free arm has beta uniform at
ln(alpha0) -- identical to the frozen arm.  Under (A) the exponent should START near the
frozen value and RISE as beta absorbs the common mode.  Under (B) it should start near the
frozen value and rise only as fast as beta SPREADS.  So we report, per record window, both
the exponent and sd(beta) -- the actual spread of the step sizes -- and check whether the
exponent moves before or after the spread does.

Reported per window: the exponent from a per-seed log-log fit over the four granularities.
"""
import os, glob, collections, math
import numpy as np
from frozen_agreement import arm_stats, load, infer_ntot, neff_from_agreement, exact_null

ROOT = os.path.join(os.path.dirname(__file__), "killtest_data")
GRANS = ("blk6", "lay", "node", "w")


def windows(T, k=5):
    e = np.linspace(0, T, k + 1).astype(int)
    return [(e[i], e[i + 1]) for i in range(k) if e[i + 1] - e[i] >= 3]


def arm_window(recs, lo, hi, nt):
    A, SD = [], []
    for r in recs[lo:hi]:
        fz, fn = r["frac_zero"], r["frac_neg"]
        if fz >= 1.0:
            continue
        n = nt * (1.0 - fz)
        if n < 2:
            continue
        p = fn / (1.0 - fz)
        A.append(max(p, 1 - p))
        b = np.array(r["beta"], dtype=float)
        SD.append(float(np.std(b)))
    if not A:
        return None
    return float(np.mean(A)), float(np.mean(SD))


def collect(batch, pat, seeds):
    """-> {gran: [ (recs, n_tot) per seed ]}"""
    out = collections.defaultdict(list)
    for g in GRANS:
        for s in range(seeds):
            d = os.path.join(ROOT, batch, pat.format(g=g, s=s))
            if not os.path.isdir(d):
                continue
            recs = load(d)
            if len(recs) < 20:
                continue
            out[g].append((recs, infer_ntot(recs)))
    return out


def report(label, data):
    gs = [g for g in GRANS if data.get(g)]
    if len(gs) < 3:
        print(f"  {label}: only {len(gs)} granularities present, skipped")
        return
    nseed = min(len(data[g]) for g in gs)
    T = min(len(data[g][k][0]) for g in gs for k in range(nseed))
    x = np.array([math.log10(data[g][0][1]) for g in gs])
    print(f"\n  {label}   (n={nseed} seeds, {T} records, m = "
          f"{', '.join(format(data[g][0][1], ',') for g in gs)})")
    print(f"    {'record window':>16s} {'epochs':>12s} {'exponent s':>14s} "
          f"{'sd(beta) mean':>15s}   " + "  ".join(f"{'agree%('+g+')':>13s}" for g in gs))
    for lo, hi in windows(T, 5):
        sl, sdb, ag = [], [], collections.defaultdict(list)
        for k in range(nseed):
            y, ok = [], True
            for g in gs:
                recs, nt = data[g][k]
                r = arm_window(recs, lo, hi, nt)
                if r is None:
                    ok = False; break
                a, sb = r
                y.append(math.log10(max(neff_from_agreement(a), 1e-9)))
                ag[g].append(a)
                if g == "lay":
                    sdb.append(sb)
            if ok and len(y) == len(gs):
                sl.append(np.polyfit(x, np.array(y), 1)[0])
        if not sl:
            continue
        ep = f"{lo*20/T:.0f}-{hi*20/T:.0f}"
        print(f"    {str(lo)+'-'+str(hi):>16s} {ep:>12s} "
              f"{np.mean(sl):>9.3f} +-{(np.std(sl, ddof=1) if len(sl)>1 else 0):.3f} "
              f"{np.mean(sdb):>15.4f}   "
              + "  ".join(f"{100*np.mean(ag[g]):>13.4f}" for g in gs))


if __name__ == "__main__":
    print("=" * 122)
    print("TIME COURSE of the effective-sample-size exponent  (s=1 <=> independent "
          "coordinates <=> the literature's 1/sqrt(N) assumption)")
    print("=" * 122)
    print("R18 / CIFAR-10 / SGDm / ms=1e-3 / a0=1e-3 / 20 ep / AUGMENT=1.  The two rows "
          "differ ONLY in --alg-meta.")
    report("FREE beta   (p7-r18-*, --alg-meta Lion)",
           collect("p7free", "p7-r18-{g}-s{s}", 10))
    report("FROZEN beta (fz-*-a3, --alg-meta fixed)",
           collect("fz", "fz-{g}-a3-s{s}", 5))
    print("\n  sd(beta) is measured on the LAYERWISE arm and is the spread of the 62 "
          "log-step-sizes.\n  It is identically 0 on every frozen row by construction -- "
          "that is the design check.")
