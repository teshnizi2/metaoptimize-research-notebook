"""How far does meta-gradient correlation REACH?  The granularity ladder, inverted.

MOTIVATION.  twochannel.py measures the intraclass sign correlation rho_s(m) between the
m GROUP meta-gradients of an arm.  A group at granularity m is a SUM of k = N/m per-weight
meta-gradients, so rho_s(m) is a correlation between sums of different widths.  Reading the
four rungs as one number is wrong; reading them as one number PER SCALE is the measurement.

THE INVERSION.  Under a one-factor model with per-weight latent correlation rho_w, two
disjoint sums of k coordinates have latent correlation

    rho_group(k) = rho_w * k / (1 + (k-1) * rho_w)                                    (1)

and, for jointly Gaussian summands, the correlation of their SIGNS is the orthant formula

    rho_sign = (2/pi) * arcsin(rho_group)                                             (2)

Inverting (1)-(2) at each rung gives rho_w^implied(k): the per-weight all-to-all correlation
that WOULD be needed for a single global factor to explain that rung on its own.

WHAT IT DIAGNOSES.  If one global common mode explained everything, rho_w^implied(k) would be
CONSTANT across rungs.  If it FALLS with k, the shared component does not keep growing as
more coordinates are summed -- i.e. correlation is SHORT-RANGE: strong between neighbouring
weights, dying out across a whole tensor.  That is a correlation length, and it is the
quantity the Adam-mini / Adalayer / SGG granularity argument implicitly assumes is zero.

WHY THIS IS THE RIGHT READING OF CORRECTIONS 33.  §33 inverted the SAME formula in the
other direction -- from a tensor-level excess down to the per-weight rho it implies -- to
show kt2's pairwise test was 2x-170x under-powered.  This script does the inversion at every
rung at once, so the scale dependence is measured instead of assumed.

CAVEAT 1.  Equations (1)-(2) assume exchangeable coordinates and Gaussian group sums.  Real
groups differ in width (block_sizes are not uniform) and in variance.  rho_w^implied is a
DIAGNOSTIC of scale dependence, not a structural parameter.  The measured rho_s(m) per rung
is the primary number; the inversion is how to compare rungs.

CAVEAT 2 -- THE ONE THAT BLOCKS THE HEADLINE, READ BEFORE QUOTING ANY "SHORT-RANGE" CLAIM.
twochannel.py estimates the independence floor as pbar(1-pbar)/n, using the CROSS-SECTIONAL
fraction pbar.  If the groups are heterogeneous -- each with its own persistent sign
preference p_b -- the true floor is mean_b[p_b(1-p_b)]/n, which is SMALLER (Jensen).  The
estimator therefore UNDER-states the common mode whenever groups are heterogeneous
(tests/test_twochannel.py VALIDATION 11 confirms it never reports a spurious positive).

That is conservative for the per-rung rho_s, which is why those are quotable as LOWER
BOUNDS.  It is NOT conservative for the scale PROFILE: heterogeneity grows with group size
(a whole tensor has a much more persistent sign than a single weight), so the downward bias
is strongest at the COARSE rungs -- which depresses rho_s(coarse) and makes rho_w^implied
fall with k.  That is precisely the signature of short-range correlation.  A falling profile
here is therefore NOT evidence of a correlation length until per-group marginals are
measured and the floor corrected.  The probe writes only the POOLED frac_neg scalar
(HF_patched.py:360), so the correction cannot be made from data now on disk; it needs a
probe that emits per-group negative fractions (PATCH_PROBE5, prepared in
bin/c44_probe5_heterogeneity.sh).
"""
import os, sys, math, glob, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twochannel import ROOT, arm, parse_gran, GRAN_ORDER  # noqa: E402

WINDOW = (0.5, 1.0)


def sign_to_latent(rho_sign):
    """Invert the orthant formula (2).  Returns nan outside the valid range."""
    if not np.isfinite(rho_sign) or abs(rho_sign) >= 1.0:
        return float("nan")
    return math.sin(math.pi * rho_sign / 2.0)


def implied_rho_w(rho_sign, k):
    """Invert (1)-(2): the per-weight latent correlation a single global factor would need
    to produce `rho_sign` between two disjoint sums of k coordinates."""
    rg = sign_to_latent(rho_sign)
    if not np.isfinite(rg) or k <= 1:
        return rg
    denom = k - (k - 1) * rg
    if denom <= 0:
        return float("nan")
    return rg / denom


def group_rho_from_weight(rho_w_latent, k):
    """Forward direction of (1)-(2): predicted SIGN correlation at group size k."""
    if not np.isfinite(rho_w_latent):
        return float("nan")
    rg = rho_w_latent * k / (1.0 + (k - 1) * rho_w_latent)
    rg = min(max(rg, -0.999999), 0.999999)
    return (2.0 / math.pi) * math.asin(rg)


def ladder(batch, fam_filter=None, window=WINDOW):
    root = os.path.join(ROOT, batch)
    by = collections.defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        g, fam = parse_gran(os.path.basename(d))
        if g is None:
            continue
        if fam_filter is not None and fam != fam_filter:
            continue
        try:
            r = arm(d, window=window)
        except Exception:
            continue
        if r:
            by[g].append(r)
    return by


def report(label, by, n_weights):
    gs = [g for g in GRAN_ORDER if by.get(g)]
    if not gs:
        return None
    print(f"\n  {label}    (N = {n_weights:,} weights)")
    h = (f"    {'gran':6s} {'m groups':>12s} {'k = N/m':>12s} {'sd':>3s} "
         f"{'rho_s(m)':>11s} {'se':>10s} {'rho_min':>10s} {'resolved':>9s} "
         f"{'rho_w implied':>14s}")
    print(h)
    print("    " + "-" * (len(h) - 4))
    out = {}
    for g in gs:
        v = by[g]
        m = float(np.mean([x["nbar"] for x in v]))
        rho = np.array([x["rho_s"] for x in v], float)
        rmin = float(np.mean([x["rho_min"] for x in v]))
        mu = float(rho.mean())
        se = float(rho.std(ddof=1) / math.sqrt(len(rho))) if len(rho) > 1 else float("nan")
        k = n_weights / m if m > 0 else float("nan")
        res = "yes" if mu > rmin else "NO"
        rw = implied_rho_w(mu, k) if res == "yes" else float("nan")
        print(f"    {g:6s} {m:>12,.0f} {k:>12,.0f} {len(v):>3d} "
              f"{mu:>11.3e} {se:>10.2e} {rmin:>10.2e} {res:>9s} "
              f"{rw:>14.3e}")
        out[g] = dict(m=m, k=k, rho=mu, se=se, rmin=rmin, resolved=(res == "yes"), rho_w=rw)
    # scale-dependence verdict
    ok = [(g, o) for g, o in out.items() if o["resolved"] and np.isfinite(o["rho_w"])]
    if len(ok) >= 2:
        ok.sort(key=lambda t: t[1]["k"])
        lo, hi = ok[0], ok[-1]
        ratio = lo[1]["rho_w"] / hi[1]["rho_w"] if hi[1]["rho_w"] else float("nan")
        print(f"    -> rho_w implied falls {ratio:,.1f}x from k={lo[1]['k']:,.0f} "
              f"({lo[0]}) to k={hi[1]['k']:,.0f} ({hi[0]})")
        print(f"       a single GLOBAL factor requires this ratio to be 1.0; "
              f"{'falling profile' if ratio > 3 else 'consistent with global'}"
              f"  [CONFOUNDED -- see CAVEAT 2; not evidence of a correlation length]")
        # what the finest rung predicts at the coarsest, vs what is measured
        pred = group_rho_from_weight(sign_to_latent(lo[1]["rho"]) if lo[1]["k"] <= 1
                                     else lo[1]["rho_w"], hi[1]["k"])
        print(f"       one-factor extrapolation from {lo[0]} predicts rho_s({hi[0]}) = "
              f"{pred:.3e}; measured {hi[1]['rho']:.3e} "
              f"(over-predicts {pred/hi[1]['rho']:,.1f}x)")
    return out


def main():
    print("=" * 108)
    print("CORRELATION RANGE -- per-weight correlation implied by each granularity rung")
    print("window =", WINDOW)
    print("=" * 108)
    print("\nFROZEN beta (--alg-meta fixed), a0=1e-3, R18/CIFAR-10, n=5 -- the CONTROLLED batch")
    report("fz / a3", ladder("fz", "a3"), 11_173_962)
    print("\nFROZEN beta, a0=1e-6, n=5")
    report("fz / a6", ladder("fz", "a6"), 11_173_962)
    print("\nFREE beta (--alg-meta Lion), a0=1e-3, R18/CIFAR-10, n=10 "
          "-- byte-matched to fz/a3 except --alg-meta")
    report("p7free / r18", ladder("p7free", "r18"), 11_173_962)
    print("\nFREE beta, a0=1e-3, CIFAR-100 / R18, n=10")
    report("p7free / c100", ladder("p7free", "c100"), 11_219_984)
    print("\nFREE beta, a0=1e-6, R18/CIFAR-10, 100 epochs, n=3 (mx)")
    report("mx", ladder("mx", ""), 11_125_461)


if __name__ == "__main__":
    main()
