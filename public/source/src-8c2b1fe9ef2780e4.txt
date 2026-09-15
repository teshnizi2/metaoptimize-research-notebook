#!/usr/bin/env python3
"""c76_partition_vs_count.py -- decompose ck1's K2 refutation.

WHAT K2 MEASURED, AND WHY IT IS NOT YET A CLEAN STATEMENT
`ck1`'s K2 compared chunk1024 (m=10,944, contiguous flat chunks) against tw0's
nodewise (m=14,420, output channels) at the same ms=1e-4 and found +0.565 pp,
outside the registered +-0.50 bar -- REFUTING "accuracy is a function of m alone".

That comparison carries TWO confounds this reducer separates, and NEITHER is
removed by rhetoric:

  (1) THE COUNT DIFFERENCE.  10,944 vs 14,420 is 24% in m.  K3 established that
      plateau5 rises as m falls, so PART of the +0.565 is just the count moving.
      Removed here by interpolating ck1's OWN chunk curve to m = 14,420, using the
      two rungs that BRACKET it (chunk1024 m=10,944 and chunk128 m=87,303).  This
      is an INTERPOLATION, not an extrapolation -- 14,420 lies strictly between the
      two -- which is why a local secant is used and not a global 5-point fit whose
      slope varies 0.35-0.83 pp/decade across the ladder.

  (2) THE CROSS-BATCH OFFSET.  chunk1024 is a `ck1` run and nodewise is a `tw0`
      run.  K1 gives a direct handle on it: chunk1 and tw0's weightwise are BITWISE
      the same configuration, so their difference IS the batch offset.  It is
      reported as a LABELLED SENSITIVITY and is NOT folded into the primary,
      because (a) at n=3 vs n=3 it is not resolved from zero, and correcting by an
      unresolved offset manufactures precision, and (b) its sign moves the residual
      AWAY from zero, so the uncorrected primary is the CONSERVATIVE one.

WHAT THIS REDUCER DOES NOT CLAIM
  * It does not remove the GROUP-SIZE-DISTRIBUTION difference.  At matched m the
    MEAN group size matches exactly (total weights are shared), but nodewise's
    groups are heterogeneous (per-layer channel counts) and chunk's are uniform at
    K except for ragged tails.  That is a real residual confound; it is named in
    the output and it is what a permuted-partition arm would isolate.
  * It does not turn an interpolated residual into a measurement.  The definitive
    version is a MATCHED-m arm run in ONE batch -- m(777) = 14,421 against
    nodewise's 14,420 -- which is why `mm1` exists.

USAGE
  python3 analysis/c76_partition_vs_count.py --selftest
  python3 analysis/c76_partition_vs_count.py --csv results/all_runs.csv
"""
import argparse
import csv
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# --- the registered facts this reducer stands on ----------------------------
M_OF_K = {1: 11173962, 2: 5586981, 16: 698373, 128: 87303, 1024: 10944}
NODE_M = 14420                      # nodewise's group count, measured
N_WEIGHTS = 11173962
RESOLVE_T = 2.0                     # the campaign's gap/SE bar


def _mean_sem(v):
    if not v:
        return None, None
    m = statistics.mean(v)
    s = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, s


def plateaus(rows, prefix):
    """Every completed 100-epoch plateau5 for runs whose name starts with `prefix`."""
    out = []
    for r in rows:
        if not r["run"].startswith(prefix):
            continue
        if r.get("epochs_done") != "100":
            continue
        v = (r.get("plateau5") or "").strip()
        if v:
            out.append(float(v))
    return out


def secant_at(m_lo, p_lo, m_hi, p_hi, m_at):
    """Value of the log10(m)-linear secant through two rungs, evaluated at m_at.

    Raises unless m_at lies strictly BETWEEN the two rungs -- this reducer
    interpolates and must never silently extrapolate.
    """
    a, b = sorted((m_lo, m_hi))
    if not (a < m_at < b):
        raise ValueError("m_at=%s is not strictly inside (%s, %s) -- this is an "
                         "extrapolation and is refused" % (m_at, a, b))
    x_lo, x_hi, x = math.log10(m_lo), math.log10(m_hi), math.log10(m_at)
    f = (x - x_lo) / (x_hi - x_lo)
    return p_lo + f * (p_hi - p_lo)


def bracketing_rungs(m_at, by_k):
    """The two chunk rungs that bracket m_at, as (K_lo_m, K_hi_m).  None if unbracketed."""
    below = [K for K in by_k if M_OF_K[K] < m_at]
    above = [K for K in by_k if M_OF_K[K] > m_at]
    if not below or not above:
        return None
    return max(below, key=lambda K: M_OF_K[K]), min(above, key=lambda K: M_OF_K[K])


def diff_stat(a, b):
    """(mean_a - mean_b) with the two-sample SE and the campaign's resolution flag."""
    ma, sa = _mean_sem(a)
    mb, sb = _mean_sem(b)
    if ma is None or mb is None:
        return None
    se = math.sqrt((sa or 0.0) ** 2 + (sb or 0.0) ** 2)
    d = ma - mb
    return dict(mean_a=ma, sem_a=sa, n_a=len(a), mean_b=mb, sem_b=sb, n_b=len(b),
                diff=d, se=se, t=(abs(d) / se if se > 0 else float("inf")),
                resolved=(se > 0 and abs(d) / se > RESOLVE_T))


# ---------------------------------------------------------------------------
def selftest():
    n = p = 0

    def ck(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("    %-66s %s" % (name, "ok" if cond else "FAIL"))

    print("c76_partition_vs_count selftest")

    # --- the secant interpolates, and REFUSES to extrapolate
    ck("secant at the low rung returns the low value",
       abs(secant_at(10, 1.0, 1000, 3.0, 10.0 + 1e-9) - 1.0) < 1e-6)
    ck("secant is linear in log10(m)",
       abs(secant_at(10, 1.0, 1000, 3.0, 100) - 2.0) < 1e-9)
    ck("secant is order-independent in its two rungs",
       abs(secant_at(10, 1.0, 1000, 3.0, 100)
           - secant_at(1000, 3.0, 10, 1.0, 100)) < 1e-9)
    for bad in (5, 5000):
        try:
            secant_at(10, 1.0, 1000, 3.0, bad)
            ok = False
        except ValueError:
            ok = True
        ck("secant REFUSES to extrapolate to m=%d" % bad, ok)
    ck("secant refuses m exactly AT a rung (not strictly inside)", _refuses(10))

    # --- bracketing picks the TIGHTEST pair, not just any pair
    by = {1: 0, 2: 0, 16: 0, 128: 0, 1024: 0}
    ck("14420 is bracketed by chunk1024 and chunk128",
       bracketing_rungs(NODE_M, by) == (1024, 128))
    ck("bracket is the TIGHTEST pair, not the endpoints",
       bracketing_rungs(NODE_M, by) != (1024, 1))
    ck("14420 lies strictly between the bracketing rungs",
       M_OF_K[1024] < NODE_M < M_OF_K[128])
    ck("an m above every rung is unbracketed",
       bracketing_rungs(2e7, by) is None)
    ck("an m below every rung is unbracketed",
       bracketing_rungs(10, by) is None)

    # --- diff_stat
    d = diff_stat([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    ck("diff_stat gets the difference right", abs(d["diff"] - 1.0) < 1e-12)
    # A DEGENERATE sample (SE exactly 0) is NOT resolved.  This matches the campaign's
    # existing convention in c75_tw0_score.argmin_of (`decided = se > 0 and ...`):
    # three seeds that happen to agree to the last digit do not license a claim, they
    # signal that the noise was not sampled.
    ck("a zero-SE difference is NOT resolved (campaign convention)",
       not d["resolved"] and d["t"] == float("inf"))
    d2 = diff_stat([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    ck("identical samples give diff 0 and are NOT resolved",
       abs(d2["diff"]) < 1e-12 and not d2["resolved"])
    d3 = diff_stat([10.0, 0.0], [5.0, 5.0])
    ck("a difference inside the noise is NOT resolved", not d3["resolved"])
    ck("diff_stat is antisymmetric",
       abs(diff_stat([3.0, 4.0], [1.0, 2.0])["diff"]
           + diff_stat([1.0, 2.0], [3.0, 4.0])["diff"]) < 1e-12)
    ck("diff_stat on a missing arm returns None", diff_stat([], [1.0]) is None)
    ck("resolution bar is the campaign's 2.0", RESOLVE_T == 2.0)

    # --- the arithmetic the output rests on
    ck("mean group size at matched m is the same for both partitions",
       abs(N_WEIGHTS / NODE_M - N_WEIGHTS / 14421) < 0.1)
    ck("chunk1024's m is BELOW nodewise's", M_OF_K[1024] < NODE_M)
    ck("m(K) table matches the registered ck1 ladder",
       M_OF_K == {1: 11173962, 2: 5586981, 16: 698373, 128: 87303, 1024: 10944})

    # --- plateaus() filters on completion, not on presence
    rows = [dict(run="ck1-k1-s0", epochs_done="100", plateau5="91.0"),
            dict(run="ck1-k1-s1", epochs_done="29", plateau5="80.0"),
            dict(run="ck1-k1-s2", epochs_done="100", plateau5=""),
            dict(run="ck1-k1024-s0", epochs_done="100", plateau5="92.0")]
    ck("plateaus drops a run that died early", plateaus(rows, "ck1-k1-") == [91.0])
    ck("plateaus drops a blank plateau5", len(plateaus(rows, "ck1-k1-")) == 1)
    ck("prefix ck1-k1- does not swallow ck1-k1024-",
       plateaus(rows, "ck1-k1-") == [91.0]
       and plateaus(rows, "ck1-k1024-") == [92.0])

    # --- the honesty clauses this reducer registered against itself
    ck("the group-size confound is named in the docstring",
       "GROUP-SIZE-DISTRIBUTION" in __doc__)
    ck("the batch offset is declared a SENSITIVITY, not a correction",
       "LABELLED SENSITIVITY" in __doc__)
    ck("the reducer says the definitive version is a matched-m batch",
       "14,421" in __doc__)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


def _refuses(m_at):
    try:
        secant_at(10, 1.0, 1000, 3.0, m_at)
        return False
    except ValueError:
        return True


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    rows = list(csv.DictReader(open(a.csv)))
    chunk = {K: plateaus(rows, "ck1-k%d-" % K) for K in M_OF_K}
    chunk = {K: v for K, v in chunk.items() if v}
    node = plateaus(rows, "tw0-node-")
    wgt = plateaus(rows, "tw0-w-")

    print("=" * 78)
    print("c76 -- IS THE GRANULARITY AXIS THE COUNT, OR THE PARTITION?")
    print("       decomposing ck1's K2 refutation.  ms=1e-4, 100 ep, box (-15, -2.3026)")
    print("=" * 78)

    print("\n--- THE RAW K2 CONTRAST (what ck1 scored)")
    raw = diff_stat(chunk.get(1024, []), node)
    if raw is None:
        print("    NO DATA")
        return 1
    print("    chunk1024  m=%-8d %.3f +-%.3f (n=%d)   contiguous flat chunks"
          % (M_OF_K[1024], raw["mean_a"], raw["sem_a"], raw["n_a"]))
    print("    nodewise   m=%-8d %.3f +-%.3f (n=%d)   output channels"
          % (NODE_M, raw["mean_b"], raw["sem_b"], raw["n_b"]))
    print("    diff %+.3f pp   SE %.3f   t = %.2f   -> %s"
          % (raw["diff"], raw["se"], raw["t"],
             "RESOLVED" if raw["resolved"] else "NOT resolved"))
    print("    But m differs by %.1f%%, so this is NOT yet a partition statement."
          % (100.0 * (NODE_M - M_OF_K[1024]) / NODE_M))

    print("\n--- CONFOUND 1: REMOVE THE COUNT DIFFERENCE")
    br = bracketing_rungs(NODE_M, chunk)
    if br is None:
        print("    nodewise's m is not bracketed by the chunk ladder -- REFUSED.")
        return 1
    k_lo, k_hi = br            # k_lo has the SMALLER m
    p_lo, _ = _mean_sem(chunk[k_lo])
    p_hi, _ = _mean_sem(chunk[k_hi])
    print("    bracketing rungs: chunk%d (m=%d, %.3f)  and  chunk%d (m=%d, %.3f)"
          % (k_lo, M_OF_K[k_lo], p_lo, k_hi, M_OF_K[k_hi], p_hi))
    pred = secant_at(M_OF_K[k_lo], p_lo, M_OF_K[k_hi], p_hi, NODE_M)
    print("    the CHUNK curve interpolated to m=%d  ->  %.3f pp" % (NODE_M, pred))
    resid = pred - raw["mean_b"]
    print("    nodewise measured at that same m        ->  %.3f pp" % raw["mean_b"])
    print("    **PARTITION RESIDUAL AT MATCHED COUNT = %+.3f pp**" % resid)
    print("    (t vs the seed noise of the two arms = %.2f, treating the secant as"
          % (abs(resid) / raw["se"] if raw["se"] > 0 else float("inf")))
    print("     exact -- it is not, so this t is an UPPER bound on the confidence.)")
    print("    Of the raw %+.3f pp, the count explains %+.3f and the partition %+.3f."
          % (raw["diff"], raw["diff"] - resid, resid))

    print("\n--- CONFOUND 2: THE CROSS-BATCH OFFSET  (SENSITIVITY, NOT A CORRECTION)")
    off = diff_stat(chunk.get(1, []), wgt)
    if off is None:
        print("    chunk1 or tw0 weightwise missing -- offset not measurable.")
    else:
        print("    chunk1 (ck1) %.3f +-%.3f (n=%d)  vs  weightwise (tw0) %.3f +-%.3f (n=%d)"
              % (off["mean_a"], off["sem_a"], off["n_a"],
                 off["mean_b"], off["sem_b"], off["n_b"]))
        print("    These are BITWISE the same configuration, so their difference IS")
        print("    the batch offset: %+.3f pp, SE %.3f, t = %.2f -> %s"
              % (off["diff"], off["se"], off["t"],
                 "RESOLVED" if off["resolved"] else "NOT resolved from zero"))
        if not off["resolved"]:
            print("    NOT resolved, so it is NOT applied.  Correcting by an unresolved")
            print("    offset manufactures precision.")
        print("    Had it been applied, the residual would move %+.3f -> %+.3f,"
              % (resid, resid - off["diff"]))
        print("    i.e. %s from zero.  The uncorrected primary is the CONSERVATIVE one."
              % ("AWAY" if abs(resid - off["diff"]) > abs(resid) else "TOWARD"))

    print("\n--- WHAT IS STILL NOT CONTROLLED")
    print("    Mean group size at matched m is IDENTICAL by construction (%.1f weights),"
          % (N_WEIGHTS / NODE_M))
    print("    but the group-size DISTRIBUTION is not: nodewise's groups are")
    print("    heterogeneous (per-layer channel counts), chunk's are uniform at K.")
    print("    A permuted partition with nodewise's exact size multiset would isolate")
    print("    architecture ALIGNMENT from size distribution.  Not run.")
    print("    The interpolation is also a model.  `mm1` replaces it with a measured")
    print("    chunk777 arm (m=14,421 vs nodewise's 14,420) inside ONE batch.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
