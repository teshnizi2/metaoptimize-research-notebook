"""Two-channel decomposition of meta-gradient sign agreement, at GROUP granularity.

WHAT THIS ANSWERS.  CORRECTIONS 31 split the failure of 1/sqrt(N) noise-averaging into two
channels that break it in different ways:

  * MARGINAL BIAS      -- the population sign fraction sits away from 1/2.  Pooling more
                          coordinates does not help AT ALL, at any N.
  * COMMON MODE        -- all coordinates share a fluctuating component, so the pooled
                          estimate has irreducible variance.  Also does not vanish with N.
  * INDEPENDENT NOISE  -- the only channel that averages down as 1/n.  This is the channel
                          the Adam-mini / Adalayer / SGG line assumes is the whole story.

CORRECTIONS 33 then WITHDREW the inference that the variance channel is exact: kt2's
per-weight PAIRWISE test is under-powered by 2x-170x for the correlation the tensor-level
data already implies, and left the frozen-beta runs explicitly undecomposed --
"the frozen runs have no PATCH_PROBE4 data, so this is an open question, not a settled one".

THIS REDUCER CLOSES THAT WITHOUT NEW COMPUTE, and it has power precisely where kt2 does not.
kt2 asked whether two INDIVIDUAL weights agree, and a shared component of size rho is
invisible per-pair.  The `frac_neg` series is an AVERAGE over all n coordinates, so a shared
component is amplified by ~n while independent noise falls as 1/n.  Aggregation is the
amplifier CORRECTIONS 33 identified; frac_neg already applies it.

THE STATISTIC.  Per probe record t, PATCH_PROBE2 writes frac_neg and frac_zero over the
arm's n_tot coordinates (HF_patched.py:360-361), computed on the INSTANTANEOUS meta-gradient
`zall`, not on the running mean `z_mean` (which is cumulative from step 0 and therefore
useless as a sign series -- CORRECTIONS 18).  Let

    n_t = n_tot * (1 - frac_zero_t)          nonzero coordinates at record t
    p_t = frac_neg_t / (1 - frac_zero_t)     fraction of those that are negative

If the n_t coordinate signs were independent given a latent population fraction P_t, then
Var_t(p_t) = E[P_t(1-P_t)]/n_t + Var(P_t).  So the variance budget of the agreement signal is

    E[(p_t - 1/2)^2]  =  b^2         +  V_common        +  V_indep
                          bias^2        Var_t(P_t)         E[p(1-p)/n_t]

and only the last term is the one that 1/sqrt(N) noise-averaging is allowed to assume.
We report each channel's share, and the intraclass sign correlation

    rho_s = V_common / E[p(1-p)]

SLOW vs FAST COMMON MODE.  Var_t(p_t) counts any slow drift of P_t over training as common
mode, which it is, but it is worth separating.  The successive-difference (Allan) estimator

    V_fast = mean_t (p_{t+1} - p_t)^2 / 2

is insensitive to a smooth trend, so V_fast - V_indep isolates the record-to-record shared
fluctuation, and V_common - (V_fast - V_indep) is the slow drift.

POWER (mandatory -- CORRECTIONS 33's standing rule: a null may not be reported as evidence of
absence until the minimum effect the test could resolve is stated next to it).  Under the
pure-independence null, Var_t(p_t) is a scaled chi-square with T-1 df, so its sampling sd is
sqrt(2/(T-1)) * V_indep.  The smallest common mode resolvable at 2 sd is therefore

    rho_min = 2 * sqrt(2/(T-1)) / n

which FALLS with n.  At T=100 that is rho_min = 4.6e-3 at n=62 and 2.5e-8 at n=11.17M --
the opposite ordering to the per-weight pairwise test, and the reason this instrument can
see structure kt2 provably could not.

CAVEAT.  A shared minibatch is itself a common mode: every coordinate's meta-gradient at
step t is computed from the SAME batch.  This estimator measures the total shared component
and does not attribute it.  It is an upper bound on any *architectural* correlation.
"""
import json, os, math, glob, collections
import numpy as np
from fractions import Fraction

ROOT = os.path.join(os.path.dirname(__file__), "killtest_data")

GRAN_ORDER = ["scal", "blk6", "lay", "node", "w"]
GRAN_NAME = {"scal": "scalar", "blk6": "resnet18_blocks", "lay": "layerwise",
             "node": "nodewise", "w": "weightwise"}


def infer_ntot(recs):
    """Recover n_tot from the rational denominator of frac_neg / frac_zero.
    block_sizes.json reports 11.17M for nodewise arms (CORRECTIONS 16)."""
    nt = 1
    for r in recs[:400]:
        for v in (r.get("frac_neg"), r.get("frac_zero")):
            if v is not None and 0 < v < 1:
                nt = max(nt, Fraction(v).limit_denominator(20_000_000).denominator)
    return nt


def load(d):
    recs = []
    p = os.path.join(d, "probe.jsonl")
    if not os.path.exists(p):
        return recs
    for l in open(p):
        l = l.strip()
        if l:
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    return recs


def _int_autocorr_time(x, max_lag=None):
    """Integrated autocorrelation time tau = 1 + 2*sum_k r_k, truncated at the first
    non-positive r_k (Geyer's initial-positive-sequence rule).  tau = 1 for white noise."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 8:
        return 1.0
    v = float(np.dot(x, x) / T)
    if v <= 0:
        return 1.0
    if max_lag is None:
        max_lag = max(1, min(T // 4, 200))
    tau = 1.0
    for k in range(1, max_lag + 1):
        r = float(np.dot(x[:-k], x[k:]) / (T - k)) / v
        if r <= 0:
            break
        tau += 2.0 * r
    return max(tau, 1.0)


def decompose(frac_neg, frac_zero, n_tot, window=None):
    """Two-channel decomposition from raw per-record frac_neg / frac_zero.

    Returns a dict of channel variances, their shares, rho_s, and the power bound.
    `window` = (lo, hi) fractional slice of the record sequence, e.g. (0.5, 1.0) for the
    steady half.  Records where every coordinate is zero are dropped.
    """
    fn = np.asarray(frac_neg, dtype=float)
    fz = np.asarray(frac_zero, dtype=float)
    if window is not None:
        lo = int(len(fn) * window[0])
        hi = int(len(fn) * window[1])
        fn, fz = fn[lo:hi], fz[lo:hi]
    keep = (1.0 - fz) > 0
    fn, fz = fn[keep], fz[keep]
    T = len(fn)
    if T < 8:
        return None
    n_t = n_tot * (1.0 - fz)
    p_t = fn / (1.0 - fz)

    pbar = float(np.mean(p_t))
    b = pbar - 0.5
    # Independent-sampling floor, averaged over records (n_t varies with frac_zero).
    # The plug-in p(1-p)/n is biased LOW: E[p(1-p)] = P(1-P)(1-1/n), so it understates the
    # floor by a factor (1-1/n) and inflates v_common by that much.  Dividing by (n-1)
    # instead is exactly unbiased for P(1-P)/n.  This matters only at the coarse arms
    # (-1.6% at n=62, -20% at n=6) but those are the arms the granularity ladder turns on.
    v_indep = float(np.mean(p_t * (1.0 - p_t) / np.maximum(n_t - 1.0, 1.0)))
    v_tot_dev = float(np.mean((p_t - 0.5) ** 2))          # b^2 + Var(p) with the 1/T convention
    v_p = float(np.var(p_t, ddof=1))                      # total variance of p_t, unbiased
    v_common = v_p - v_indep                              # may be negative under the null
    # fast (trend-insensitive) component
    d = np.diff(p_t)
    v_fast_tot = float(np.mean(d ** 2) / 2.0) if len(d) else float("nan")
    v_fast_common = v_fast_tot - v_indep
    v_slow_common = v_common - v_fast_common

    # same (1-1/n) correction on the scale factor P(1-P) that rho is expressed against
    pq = float(np.mean(p_t * (1.0 - p_t) * n_t / np.maximum(n_t - 1.0, 1.0)))
    rho_s = v_common / pq if pq > 0 else float("nan")
    rho_fast = v_fast_common / pq if pq > 0 else float("nan")

    nbar = float(np.mean(n_t))
    # Sampling sd of Var_t(p_t) under the pure-independence null.  The chi-square form
    # sqrt(2/(T-1))*v_indep assumes INDEPENDENT records; probe records are 25 optimiser
    # steps apart in a smooth training run, so p_t is autocorrelated and that form
    # understates the sd (i.e. overstates z).  Deflate T by the integrated autocorrelation
    # time of the deviation series.  This only affects significance, never the point
    # estimate of rho_s.
    tau = _int_autocorr_time(p_t - p_t.mean())
    T_eff = max(T / max(tau, 1.0), 2.0)
    sd_v = math.sqrt(2.0 / max(T_eff - 1, 1)) * v_indep
    rho_min = 2.0 * sd_v / pq if pq > 0 else float("nan")
    z_common = v_common / sd_v if sd_v > 0 else float("nan")

    denom = v_tot_dev if v_tot_dev > 0 else float("nan")
    return dict(T=T, T_eff=T_eff, tau=tau, nbar=nbar, pbar=pbar, bias=b,
                v_bias=b * b, v_common=v_common, v_indep=v_indep,
                v_fast_common=v_fast_common, v_slow_common=v_slow_common,
                v_total=v_tot_dev,
                share_bias=b * b / denom, share_common=v_common / denom,
                share_indep=v_indep / denom,
                rho_s=rho_s, rho_fast=rho_fast, rho_min=rho_min, z_common=z_common,
                # what the campaign's N/N_eff would be, from the same budget
                n_over_neff=denom / v_indep if v_indep > 0 else float("nan"))


def arm(d, window=None):
    recs = load(d)
    if len(recs) < 8:
        return None
    n_tot = infer_ntot(recs)
    if n_tot < 2:
        return None                      # scalar arms: frac_neg is a time-fraction, not agreement
    fn = [r.get("frac_neg") for r in recs]
    fz = [r.get("frac_zero", 0.0) for r in recs]
    if any(v is None for v in fn):
        return None
    out = decompose(fn, fz, n_tot, window=window)
    if out is not None:
        out["n_tot"] = n_tot
        out["frozen"] = all(
            abs(r.get("beta_true_max", 0.0) - r.get("beta_true_min", 0.0)) < 1e-9 for r in recs)
    return out


def parse_gran(name):
    t = name.replace("probe_sig_", "").replace("probe_", "")
    t = t.rsplit("-s", 1)[0].rsplit("_s", 1)[0]
    p = t.replace("_", "-").split("-")
    if p and p[0] in ("p7", "p6f", "p9", "fz"):
        p = p[1:]
    canon = {"blk6": "blk6", "resnet18": "blk6", "blocks": "blk6", "lay": "lay",
             "layer": "lay", "layerwise": "lay", "node": "node", "nodewise": "node",
             "w": "w", "ww": "w", "weight": "w", "weightwise": "w",
             "scal": "scal", "scalar": "scal"}
    for tok in reversed(p):
        if tok in canon:
            return canon[tok], "-".join(x for x in p if x != tok)
    return None, None
