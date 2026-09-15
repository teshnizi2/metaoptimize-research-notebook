#!/usr/bin/env python3
r"""PROBE5 reducer -- the HETEROGENEITY-CORRECTED independence floor.

WRITTEN BEFORE THE DATA (cycle 47), per the discipline that produced
`analysis/idea3_robustness.py`: the reducer and its validation exist and pass before
the batch lands, so the analysis cannot be tuned to the result.

WHAT IT FIXES.  `analysis/twochannel.py` estimates the independence floor of the sign
agreement statistic from the POOLED per-record fraction:

    V_indep_pooled = mean_t [ p_t (1 - p_t) / (n_t - 1) ]        (twochannel.py:150)

With heterogeneous coordinates -- each carrying its own persistent sign preference P_b --
that is the WRONG floor, and it is wrong in a direction that manufactures the one structure
FINDINGS 44.5 wanted to claim.

THE ALGEBRA, and it is short enough to check.  Model record t as: coordinate b is negative
with probability P_{b,t} = P_b + f_t, independent across b GIVEN the common mode f_t
(E[f]=0).  Then

    Var_t(p_t) = Var(f)  +  E_f[ mean_b (P_b+f)(1-P_b-f) ] / n
                            \_______________ the TRUE floor ______/

and since  mean_b[P_b(1-P_b)] = Pbar(1-Pbar) - Var_b(P_b),  to leading order in f

    V_indep_true  =  V_indep_pooled  -  Var_b(P_b) / n .                             (*)

So the pooled floor OVERSTATES the true floor by exactly `Var_b(P_b)/n`, and therefore
UNDERSTATES the common mode by the same amount:

    V_common_true = V_common_pooled + Var_b(P_b) / n .

`Var_b(P_b)` is the cross-coordinate spread of marginal bias.  PATCH_PROBE2 cannot see it --
it writes one pooled scalar per record.  PATCH_PROBE5 writes the per-coordinate negative
COUNTS, which is exactly `Var_b(P_b)`.

WHY THE SIGN OF THE BIAS MATTERS SO MUCH HERE.  Heterogeneity grows with group size: a
whole tensor's meta-gradient sum has a far more persistent sign than one weight's.  So the
missing term `Var_b(P_b)/n` is largest at the COARSE rungs, which depresses rho_s(coarse)
and makes the implied per-weight rho FALL with block size -- which is precisely the
signature of short-range correlation.  FINDINGS 44.5 therefore refused to claim the profile.
This reducer is what decides it.

ESTIMATING Var_b(P_b) FROM COUNTS, with the two corrections that matter.
Let  p_b = c_b / T  be coordinate b's observed negative fraction over T records.
Writing  p_b = P_b + fbar + e_b  (fbar is COMMON to every b, so it moves the mean and NOT
the cross-coordinate variance -- this is why the common mode does not contaminate the
estimate):

    Var_b(p_b) = Var_b(P_b) + E_b[ Var_t(p_b) ]
               = Var_b(P_b) + mean_b[P_b(1-P_b)] * tau / T

so the estimator is

    Var_b(P_b)^  =  Var_b(p_b)  -  mean_b[p_b(1-p_b)] * tau / T                      (**)

  1. `tau` is the integrated autocorrelation time.  Probe records are `PROBE` optimiser
     steps apart in a smooth run and are NOT independent -- CORRECTIONS 34 measured tau up
     to 36.3.  Using tau=1 under-subtracts the sampling term and INFLATES Var_b(P_b).
     tau is estimated from the pooled p_t series (the per-coordinate series is not written)
     and BOTH tau=1 and tau=tau_pooled are reported as a sensitivity band, because that
     substitution is the reducer's one uncontrolled approximation.
  2. The result is clipped at 0.  A negative estimate means heterogeneity is below the
     sampling floor at this T -- it is reported as `<res` and NOT as zero, per CORRECTIONS
     33's standing rule that a null needs its resolution stated beside it.

THE HETEROGENEITY RATIO reported per rung is

    H = V_indep_true / V_indep_pooled  =  1 - Var_b(P_b) / (n * V_indep_pooled)   in (0, 1]

H = 1 means homogeneous coordinates and the pooled floor was already right.  H -> 0 means
the pooled floor was almost entirely heterogeneity.

PRE-REGISTERED, from bin/c44_probe5_heterogeneity.sh -- scored against these, not otherwise:
  (0) SANITY, batch is void if it fails:  H <= 1 + 3 s.e. on every rung, and
      H(weightwise) > H(blk6)  -- i.e. the COARSE arm is the MORE heterogeneous one.
  (a) PROFILE IS AN ARTEFACT:  after correction rho_w implied is flat across resolved rungs
      to within 3x.  Then "short-range correlation" is withdrawn.
  (b) CORRELATION HAS A RANGE:  rho_w implied still falls >= 10x after correction.

Run `--selftest` before trusting any number this prints.  The selftest builds synthetic
data with KNOWN Var_b(P_b) and KNOWN common mode and checks recovery -- including the case
that matters most, a TRUE global common mode with homogeneous coordinates, where a broken
estimator would report spurious heterogeneity.
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from twochannel import _int_autocorr_time, decompose, infer_ntot, load
except ImportError:  # allow --selftest from any cwd without the sibling import
    _int_autocorr_time = decompose = infer_ntot = load = None


# --------------------------------------------------------------------- core estimator
def hetvar_from_counts(counts, n_records, tau=1.0):
    """Var_b(P_b) from per-coordinate negative COUNTS, equation (**) in the docstring.

    Returns (hetvar, hetvar_raw, sampling_term, resolution).
      hetvar          -- the estimate, clipped at 0
      hetvar_raw      -- before clipping; negative means below resolution
      sampling_term   -- what was subtracted, mean_b[p_b(1-p_b)] * tau / T
      resolution      -- sd of Var_b(p_b) under the homogeneous null, the smallest
                         Var_b(P_b) this T and this n could distinguish from zero
    """
    c = np.asarray(counts, dtype=np.float64)
    T = float(n_records)
    n = c.size
    if T < 2 or n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    p = c / T
    var_p = float(np.var(p, ddof=1))
    pq = float(np.mean(p * (1.0 - p)))
    sampling = pq * tau / T
    raw = var_p - sampling
    # Under the homogeneous null Var_b(p_b) is a scaled chi-square with n-1 df, so its
    # sampling sd is sqrt(2/(n-1)) * sampling.  Two of those is the resolution.
    res = 2.0 * math.sqrt(2.0 / max(n - 1, 1)) * sampling
    return max(raw, 0.0), raw, sampling, res


def corrected_floor(v_indep_pooled, hetvar, n_tot):
    """Equation (*): the pooled floor minus the heterogeneity it wrongly contains."""
    if not np.isfinite(v_indep_pooled) or n_tot < 2:
        return float("nan"), float("nan")
    v_true = v_indep_pooled - hetvar / float(n_tot)
    # A floor cannot be negative.  If the correction overshoots, the model is wrong
    # (or tau is badly mis-estimated) and the caller must be told, not quietly clipped.
    H = v_true / v_indep_pooled if v_indep_pooled > 0 else float("nan")
    return v_true, H


def recompute_rho(dec, v_indep_true):
    """Re-run twochannel's channel split with the corrected floor.

    Only the floor changes; V_p (the total variance of p_t) is an observable and is
    untouched.  Returns a dict with the corrected common mode, rho_s and rho_min.
    """
    if dec is None or not np.isfinite(v_indep_true):
        return None
    v_p = dec["v_common"] + dec["v_indep"]        # the observed Var_t(p_t), reconstructed
    v_common_true = v_p - v_indep_true
    # rho is expressed against the same P(1-P) scale twochannel uses
    pq = dec["v_indep"] * max(dec["nbar"] - 1.0, 1.0)
    rho_s = v_common_true / pq if pq > 0 else float("nan")
    sd_v = math.sqrt(2.0 / max(dec["T_eff"] - 1, 1)) * v_indep_true
    rho_min = 2.0 * sd_v / pq if pq > 0 else float("nan")
    return dict(v_common=v_common_true, v_indep=v_indep_true, rho_s=rho_s,
                rho_min=rho_min, resolved=bool(rho_s > rho_min),
                z=v_common_true / sd_v if sd_v > 0 else float("nan"))


# --------------------------------------------------------------------- dir handling
def read_probe5(d):
    """Load neg_counts.npy + neg_counts.json from a p5 probe dir. Returns None if absent."""
    npy = os.path.join(d, "neg_counts.npy")
    js = os.path.join(d, "neg_counts.json")
    if not (os.path.exists(npy) and os.path.exists(js)):
        return None
    meta = json.load(open(js))
    counts = np.load(npy)
    if counts.size != int(meta.get("n_tot", counts.size)):
        return None                      # partial/torn write -- refuse it
    return counts, meta


def analyse_dir(d):
    """Full corrected analysis of one probe dir. Returns a row dict or None."""
    got = read_probe5(d)
    if got is None:
        return None
    counts, meta = got
    T5 = int(meta["n_records"])
    n_tot = int(meta["n_tot"])

    recs = load(d) if load else []
    dec = None
    tau = 1.0
    if recs:
        fn = [r.get("frac_neg") for r in recs]
        fz = [r.get("frac_zero", 0.0) for r in recs]
        if not any(v is None for v in fn):
            dec = decompose(fn, fz, n_tot)
            if dec:
                tau = dec["tau"]

    rows = {}
    for lab, tv in (("tau=1", 1.0), ("tau=pooled", tau)):
        hv, raw, samp, res = hetvar_from_counts(counts, T5, tau=tv)
        entry = dict(hetvar=hv, hetvar_raw=raw, sampling=samp, hetres=res,
                     het_resolved=bool(raw > res))
        if dec:
            v_true, H = corrected_floor(dec["v_indep"], hv, n_tot)
            entry.update(v_indep_pooled=dec["v_indep"], v_indep_true=v_true, H=H)
            entry["corr"] = recompute_rho(dec, v_true)
        rows[lab] = entry
    return dict(dir=d, n_tot=n_tot, T5=T5, tau=tau, dec=dec, bands=rows,
                counts_mean=float(np.mean(counts) / T5))


N_WEIGHTS_R18 = 11_173_962   # ResNet18 parameter count (CORRECTIONS 16: do NOT trust
                             # block_sizes.json's n_b, it reports this for nodewise too)


def profile(root, n_weights=N_WEIGHTS_R18):
    """The pre-registered scale-profile verdict, with the CORRECTED floor.

    FINDINGS 44.5 measured rho_w^implied falling 2.7x-85.4x from k=1 to k~1.8e5 and refused
    to call it a correlation length, because the pooled floor is biased in exactly the
    direction that produces a falling profile.  This recomputes the same inversion with the
    heterogeneity-corrected floor and scores the batch's own pre-registration:

      (a) ARTEFACT -- rho_w^implied is flat across resolved rungs to within 3x.
      (b) REAL RANGE -- it still falls >= 10x after correction.

    Rungs below their own rho_min are EXCLUDED and printed, never silently dropped
    (CORRECTIONS 33).  The pre-registered blk6 rung is not expected to resolve.
    """
    try:
        from corr_range import implied_rho_w
    except ImportError:
        print("cannot import corr_range.implied_rho_w -- run from the repo root")
        return
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "neg_counts.json")))
    rows = []
    for d in dirs:
        r = analyse_dir(d)
        if r is None:
            continue
        band = r["bands"]["tau=pooled"]
        c = band.get("corr")
        if not c:
            continue
        k = max(n_weights / r["n_tot"], 1.0)
        rows.append(dict(name=os.path.basename(d), n_tot=r["n_tot"], k=k, H=band.get("H"),
                         rho_s=c["rho_s"], rho_min=c["rho_min"], resolved=c["resolved"],
                         rho_w=implied_rho_w(c["rho_s"], k) if c["resolved"] else float("nan")))
    if not rows:
        print("no reducible PROBE5 dirs found")
        return
    print("\nSCALE PROFILE with the HETEROGENEITY-CORRECTED floor")
    hdr = (f"{'rung':26}{'n_tot':>12}{'k':>12}{'H':>8}{'rho_s':>11}{'rho_min':>11}"
           f"{'res':>5}{'rho_w implied':>15}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: x["k"]):
        print(f"{r['name'][:25]:26}{r['n_tot']:>12,}{r['k']:>12,.0f}"
              f"{(r['H'] if r['H'] is not None else float('nan')):>8.3f}"
              f"{r['rho_s']:>11.3e}{r['rho_min']:>11.3e}"
              f"{('YES' if r['resolved'] else 'NO'):>5}{r['rho_w']:>15.3e}")
    ok = [r for r in rows if r["resolved"] and np.isfinite(r["rho_w"]) and r["rho_w"] > 0]
    unres = [r["name"] for r in rows if not r["resolved"]]
    if unres:
        print(f"  EXCLUDED as below their own rho_min: {', '.join(unres)}")
        print("  (a null at these rungs means 'no correlation above rho_min', not 'none').")
    if len(ok) < 2:
        print("  NOT DECIDABLE -- fewer than 2 resolved rungs; the profile needs at least two.")
        return
    lo = min(ok, key=lambda r: r["k"])
    hi = max(ok, key=lambda r: r["k"])
    ratio = lo["rho_w"] / hi["rho_w"] if hi["rho_w"] else float("inf")
    print(f"\n  rho_w implied changes {ratio:,.2f}x from k={lo['k']:,.0f} to k={hi['k']:,.0f}")
    if ratio < 3.0:
        print("  -> OUTCOME (a): the profile is FLAT after correction to within 3x.")
        print("     FINDINGS 44.5's provisional short-range reading is WITHDRAWN; the honest")
        print("     claim is a single scale-free per-weight rho.")
    elif ratio >= 10.0:
        print("  -> OUTCOME (b): rho_w implied still falls >=10x after correction.")
        print("     The correlation length is REAL and is the paper's structural result.")
    else:
        print(f"  -> NEITHER pre-registered outcome: {ratio:,.2f}x is between 3x and 10x.")
        print("     Report it as an inconclusive middle, NOT as the nearer of the two.")


def main(root):
    dirs = sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "neg_counts.json")))
    if not dirs:
        print(f"no PROBE5 dirs under {root} (looking for neg_counts.json)")
        print("NOTE: a dir with probe.jsonl but no neg_counts.json means PROBE5 was not set,")
        print("      or the run died before the first write (record PROBE5_WRITE_EVERY).")
        return
    print("PROBE5 -- HETEROGENEITY-CORRECTED INDEPENDENCE FLOOR")
    print("H = corrected floor / pooled floor.  H=1 homogeneous; H->0 the pooled floor was")
    print("almost all heterogeneity.  Pre-registered sanity: H<=1 everywhere and H(w)>H(blk6).")
    print()
    hdr = (f"{'dir':28}{'n_tot':>10}{'T':>6}{'tau':>7}{'H tau=1':>10}{'H pooled':>10}"
           f"{'rho_s corr':>12}{'rho_min':>10}{'res':>5}")
    print(hdr)
    print("-" * len(hdr))
    for d in dirs:
        r = analyse_dir(d)
        if r is None:
            print(f"{os.path.basename(d)[:27]:28}  (unreadable / torn write)")
            continue
        b1, bp = r["bands"]["tau=1"], r["bands"]["tau=pooled"]
        h1 = b1.get("H", float("nan"))
        hp = bp.get("H", float("nan"))
        cp = bp.get("corr") or {}
        print(f"{os.path.basename(d)[:27]:28}{r['n_tot']:>10,}{r['T5']:>6}{r['tau']:>7.1f}"
              f"{h1:>10.4f}{hp:>10.4f}"
              f"{cp.get('rho_s', float('nan')):>12.3e}{cp.get('rho_min', float('nan')):>10.3e}"
              f"{('YES' if cp.get('resolved') else 'NO'):>5}")
        if np.isfinite(hp) and hp > 1.0 + 3 * b1["hetres"]:
            print("     ^ SANITY FAIL: H > 1.  The corrected floor exceeds the pooled one, "
                  "which equation (*) forbids.  Do not use this rung.")
        if not b1["het_resolved"]:
            print(f"     ^ heterogeneity BELOW resolution (raw {b1['hetvar_raw']:.3e} vs "
                  f"res {b1['hetres']:.3e}) -- report as '<res', never as zero.")
    print()
    print("Rungs printed NO in `res` are below their own rho_min and are NULLS WITH A "
          "RESOLUTION, not zeros (CORRECTIONS 33).")
    print("The tau=1 / tau=pooled pair is the sensitivity band for this reducer's one")
    print("uncontrolled approximation: per-coordinate tau is not written, so the pooled")
    print("tau is substituted.  If the two columns disagree materially, say so.")


# --------------------------------------------------------------------- selftest
def _selftest():
    ok = fail = 0

    def chk(name, cond, extra=""):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {name} {extra}")

    rng = np.random.default_rng(0)

    # -- V1: homogeneous coordinates, no common mode.  Var_b(P_b) is TRULY zero, so the
    #    estimator must return something below its own resolution -- NOT a positive number.
    n, T = 4000, 2000
    P = np.full(n, 0.5)
    counts = rng.binomial(T, P)
    hv, raw, samp, res = hetvar_from_counts(counts, T, tau=1.0)
    chk("V1 homogeneous -> below resolution", abs(raw) < res, f"raw={raw:.3e} res={res:.3e}")

    # -- V2: known heterogeneity is RECOVERED.  This is the estimator's main job.
    true_var = 0.01
    P = np.clip(rng.normal(0.5, math.sqrt(true_var), n), 0.02, 0.98)
    true_var_actual = float(np.var(P, ddof=1))
    counts = rng.binomial(T, P)
    hv, raw, samp, res = hetvar_from_counts(counts, T, tau=1.0)
    rel = abs(hv - true_var_actual) / true_var_actual
    chk("V2 heterogeneity recovered within 5%", rel < 0.05, f"got {hv:.5f} want {true_var_actual:.5f}")

    # -- V3: THE ONE THAT MATTERS.  A true GLOBAL COMMON MODE with homogeneous coordinates.
    #    f_t shifts every coordinate identically, so it must NOT appear as heterogeneity.
    #    A broken estimator reports spurious Var_b(P_b) here and would then over-correct the
    #    floor, manufacturing exactly the scale dependence FINDINGS 44.5 refused to claim.
    f = rng.normal(0.0, 0.05, T)
    counts = np.zeros(n, dtype=np.int64)
    for t in range(T):
        counts += (rng.random(n) < (0.5 + f[t])).astype(np.int64)
    hv, raw, samp, res = hetvar_from_counts(counts, T, tau=1.0)
    chk("V3 global common mode is NOT read as heterogeneity", abs(raw) < 3 * res,
        f"raw={raw:.3e} res={res:.3e}")

    # -- V4: Jensen direction.  The corrected floor must never EXCEED the pooled one,
    #    and H must land in (0, 1].
    v_true, H = corrected_floor(1e-4, 0.01, 1000)
    chk("V4 corrected floor <= pooled", v_true <= 1e-4)
    chk("V4 H in (0,1]", 0 < H <= 1.0, f"H={H}")
    v_true0, H0 = corrected_floor(1e-4, 0.0, 1000)
    chk("V4 zero heterogeneity -> H == 1", abs(H0 - 1.0) < 1e-12)

    # -- V5: tau does real work.  With autocorrelated records the tau=1 estimate must be
    #    BIASED HIGH (it under-subtracts the sampling term), which is the direction that
    #    would fake heterogeneity.
    P = np.full(n, 0.5)
    tau_true = 10
    counts = np.zeros(n, dtype=np.int64)
    blk = None
    for t in range(T):
        if t % tau_true == 0:
            blk = (rng.random(n) < P).astype(np.int64)
        counts += blk
    hv1, raw1, _, res1 = hetvar_from_counts(counts, T, tau=1.0)
    hvt, rawt, _, rest = hetvar_from_counts(counts, T, tau=float(tau_true))
    chk("V5 tau=1 over-states heterogeneity", raw1 > rawt)
    chk("V5 tau=true recovers ~zero", abs(rawt) < 3 * rest, f"raw={rawt:.3e} res={rest:.3e}")

    # -- V6: coarse arm.  n=6 with near-deterministic block signs is the blk6 rung; the
    #    estimator must survive it and report LOW H (i.e. high heterogeneity).
    n6, T6 = 6, 2000
    P6 = np.array([0.02, 0.05, 0.5, 0.95, 0.98, 0.5])
    c6 = rng.binomial(T6, P6)
    hv6, raw6, _, res6 = hetvar_from_counts(c6, T6, tau=1.0)
    chk("V6 coarse arm heterogeneity is large", hv6 > 0.1, f"hv={hv6:.4f}")
    # pooled floor for this arm, mean_t p(1-p)/(n-1) with pbar ~ 0.5
    v_pool = 0.5 * 0.5 / (n6 - 1)
    v6, H6 = corrected_floor(v_pool, hv6, n6)
    chk("V6 coarse H is much less than 1", H6 < 0.6, f"H={H6:.3f}")

    # -- V7: ordering, which is pre-registered sanity check (0).  A fine arm (each b is one
    #    weight, mild bias) must have H CLOSER to 1 than a coarse arm.
    nw, Tw = 20000, 2000
    Pw = np.clip(rng.normal(0.5, 0.02, nw), 0.01, 0.99)
    cw = rng.binomial(Tw, Pw)
    hvw, _, _, _ = hetvar_from_counts(cw, Tw, tau=1.0)
    v_poolw = 0.25 / (nw - 1)
    _, Hw = corrected_floor(v_poolw, hvw, nw)
    chk("V7 H(fine) > H(coarse)", Hw > H6, f"Hw={Hw:.3f} H6={H6:.3f}")

    # -- V8: torn-write refusal.  A neg_counts.npy whose length disagrees with n_tot must
    #    be REFUSED, not silently analysed -- the file is written every WRITE_EVERY records
    #    and a job killed mid-write leaves a mismatched pair.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "neg_counts.npy"), np.zeros(10))
        json.dump({"n_records": 100, "n_tot": 62, "stepsize_type": "layerwise"},
                  open(os.path.join(d, "neg_counts.json"), "w"))
        chk("V8 torn write refused", read_probe5(d) is None)
        np.save(os.path.join(d, "neg_counts.npy"), np.zeros(62))
        chk("V8 matching write accepted", read_probe5(d) is not None)

    # -- V9: degenerate inputs return nan, never a number
    chk("V9 n<2 -> nan", math.isnan(hetvar_from_counts([5], 100)[0]))
    chk("V9 T<2 -> nan", math.isnan(hetvar_from_counts([5, 3], 1)[0]))

    print(f"selftest: {ok}/{ok + fail} PASS")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    _root = next((a for a in sys.argv[1:] if not a.startswith("--")),
                 "analysis/killtest_data/p5")
    main(_root)
    if "--profile" in sys.argv:
        profile(_root)
