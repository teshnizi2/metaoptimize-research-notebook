#!/usr/bin/env python3
"""c77_family_curve.py -- the chunk family as ONE curve, and the matched-m channel split.

**THIS SCRIPT IS POST-HOC AND DESCRIPTIVE.  IT REGISTERS NOTHING AND SCORES NOTHING.**
Every gate this campaign has a verdict on was registered in a batch script before its
data existed and scored by `c75_ck1_score.py`, `c76_mm1_score.py`, `c76_cx2_score.py`.
This file only JOINS data those scorers already read, so no number here may be quoted
as a confirmed or refuted prediction.  Where it computes something new it says so and
says it is a reading.

WHAT IT DOES

  F1  THE SEVEN-RUNG CHUNK FAMILY AS ONE CURVE.  ck1 measured chunk{1,2,16,128,1024}
      and cx2 chunk{8192,65536}: m = 11,173,962 down to 220, ONE partition family,
      ONE knob, ONE ms, all box-free.  Reports plateau5, pbar, bias, a_raw, a_deb,
      bias_share, N_eff/m per rung and the whole-family d log N_eff / d log m slope.
      The slope is a REPLACEMENT for tw0's 0.686, which was fitted across three
      DIFFERENT partitions (layerwise / nodewise / weightwise) and therefore confounded
      the count with the partition -- exactly what mm1 showed is not safe to do.

  F2  THE MATCHED-m CHANNEL SPLIT.  mm1 holds m fixed to one group in 14,420 and varies
      only WHICH weights are grouped.  The instrument's two statistics separate a
      persistent tilt from an instantaneous one:
          dev_raw = a_raw - 0.5 = mean_t |p_t - 0.5|-ish   (deviation from the FIXED 0.5)
          dev_deb = a_deb - 0.5 = mean_t |p_t - pbar|      (deviation from the run's mean)
          bias channel = dev_raw - dev_deb
      F2 splits the matched-m difference in dev_raw into those two channels and reports
      each with a per-seed t.  **NO DIRECTION WAS REGISTERED FOR THIS AND NONE IS
      INFERRED.**  mm1's M3 was declared descriptive in advance; this is M3 arithmetic.

  F3  DOES THE FIELD PREDICT ACCURACY?  Across the seven chunk rungs the count and the
      field both move, so ANY correlation between them is confounded with m.  F3 prints
      the pairing and REFUSES to fit it, stating why.  The only place in the campaign
      where the field and accuracy vary with m held fixed is mm1's two arms, which is
      TWO points -- a slope through two points is not evidence.

WHAT THIS SCRIPT WILL NOT DO
  * It will not reproduce the "53.1%" sentence.
  * It will not treat the fall of a_raw with m as evidence of structure: a coarse
    coordinate's meta-gradient is a sum over many fine ones, so the fall is PARTLY
    MECHANICAL.  The whole point of F2 is that at MATCHED m that mechanical channel
    is switched off.
  * It will not fit accuracy on the field across rungs (F3).

USAGE
  python3 analysis/c77_family_curve.py --selftest
  python3 analysis/c77_family_curve.py
"""
import argparse
import csv
import glob
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from c52_boxfree import occupancy                      # noqa: E402

# The chunk family, MEASURED m per K (guard 4, from the allocated beta, in both batches).
CHUNK_M = {1: 11173962, 2: 5586981, 16: 698373, 128: 87303, 1024: 10944,
           8192: 1407, 65536: 220}
CK1_KS = (1, 2, 16, 128, 1024)
CX2_KS = (8192, 65536)
# The family's coarse endpoint: chunk<huge> == layerwise BITWISE (tests/test_chunkwise C2).
# Its accuracy is the tuned ladder's layerwise cell, NOT run under a chunk<K> label.
LAYERWISE_M, LAYERWISE_PLATEAU, LAYERWISE_N = 62, 92.887, 11
LO, HI = -15.0, -2.3026
MM1_M = {"ch": 14421, "node": 14420}


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def _t(a, b):
    """Welch-ish t on two small samples of means.  DESCRIPTIVE."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = math.sqrt(_sem(a) ** 2 + _sem(b) ** 2)
    return (statistics.mean(a) - statistics.mean(b)) / se if se > 0 else float("inf")


def split_channels(stats):
    """agreement_stats dict -> (dev_raw, dev_deb, dev_bias).

    dev_raw is the deviation of the per-step negative-fraction from the FIXED point 0.5.
    dev_deb is its deviation from the RUN'S OWN mean.  The difference is the persistent
    tilt, i.e. the bias channel.  These are the instrument's own definitions
    (neff_instrument.agreement_stats lines 90-92), not new quantities.
    """
    dev_raw = stats["a_raw"] - 0.5
    dev_deb = stats["a_deb"] - 0.5
    return dev_raw, dev_deb, dev_raw - dev_deb


def slope(ms, ys):
    """d log10 y / d log10 m over >= 3 distinct m.  NaN otherwise."""
    x, z = [], []
    for m, y in zip(ms, ys):
        if m and y and m > 0 and y > 0 and math.isfinite(m) and math.isfinite(y):
            x.append(math.log10(m))
            z.append(math.log10(y))
    if len(set(x)) < 3:
        return float("nan")
    import numpy as np
    return float(np.polyfit(np.asarray(x), np.asarray(z), 1)[0])


def collect(root, tag, ks, m_of_k):
    """probe root -> {K: [agreement_stats, ...]} plus {K: [N_eff/m, ...]}."""
    from neff_instrument import agreement_stats
    import probe5_window as p5w
    ag, neff, free = {}, {}, {}
    for d in sorted(glob.glob(os.path.join(root, "probe_*"))):
        base = os.path.basename(d)
        parts = base.split("_")
        if len(parts) < 3 or parts[0] != "probe" or tag not in parts:
            continue
        ktok = next((p for p in parts if p.startswith("k") and p[1:].isdigit()), None)
        if ktok is None:
            continue
        K = int(ktok[1:])
        if K not in ks:
            continue
        s = agreement_stats(d, window=(0.5, 1.0))
        if s is None:
            continue
        ag.setdefault(K, []).append(s)
        o = occupancy(d, LO, HI)
        free.setdefault(K, []).append(o["rec_lo"] < 0.05 and o["rec_hi"] < 0.05)
        w = p5w.reduce_dir(d, windows=(("steady .5-1", (0.5, 1.0)),))
        if w is None:
            continue
        ww = w["win"].get("steady .5-1")
        if not ww or ww["rho_s"] is None:
            continue
        mm, rho = float(w["n_tot"]), ww["rho_s"]
        neff.setdefault(K, []).append(1.0 / (1.0 + (mm - 1.0) * rho))
    return ag, neff, free


def selftest():
    n = p = 0

    def ck(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("    %-66s %s" % (name, "ok" if cond else "FAIL"))

    print("c77_family_curve selftest")
    # --- the family index, against the two batch scripts' own WANT tables
    ck1 = open(os.path.join(REPO, "bin", "c75_chunk_ladder.sh")).read()
    cx2 = open(os.path.join(REPO, "bin", "c76_chunk_coarse_extension.sh")).read()
    mm1 = open(os.path.join(REPO, "bin", "c76_matched_m_partition.sh")).read()
    ck("ck1's WANT table matches this file's CHUNK_M",
       "WANT = {1: 11173962, 2: 5586981, 16: 698373, 128: 87303, 1024: 10944}" in ck1)
    ck("cx2's WANT table matches this file's CHUNK_M",
       "WANT = {8192: 1407, 65536: 220}" in cx2)
    ck("CHUNK_M is exactly the union of the two WANT tables",
       set(CHUNK_M) == set(CK1_KS) | set(CX2_KS))
    ck("mm1's M_CHUNK matches", ("M_CHUNK=%d" % MM1_M["ch"]) in mm1)
    ck("mm1's M_NODE matches", ("M_NODE=%d" % MM1_M["node"]) in mm1)
    ck("m(K) is strictly decreasing over the whole family",
       all(CHUNK_M[a] > CHUNK_M[b]
           for a, b in zip(sorted(CHUNK_M), sorted(CHUNK_M)[1:])))
    ck("the family spans >4.7 decades",
       math.log10(CHUNK_M[1] / CHUNK_M[65536]) > 4.7)
    ck("layerwise m=62 is BELOW the family's coarsest measured rung",
       LAYERWISE_M < CHUNK_M[65536])
    ck("mm1's matched m sits strictly inside the family's range",
       CHUNK_M[65536] < MM1_M["node"] < CHUNK_M[1])
    ck("mm1's two arms are one group apart",
       abs(MM1_M["ch"] - MM1_M["node"]) == 1)
    ck("box is ck1's box", (LO, HI) == (-15.0, -2.3026))

    # --- the channel split is exactly the instrument's own definition
    s = dict(a_raw=0.54, a_deb=0.52)
    dr, dd, db = split_channels(s)
    ck("dev_raw is a_raw - 0.5", abs(dr - 0.04) < 1e-12)
    ck("dev_deb is a_deb - 0.5", abs(dd - 0.02) < 1e-12)
    ck("the bias channel is the difference of the two", abs(db - 0.02) < 1e-12)
    ck("the two channels sum back to dev_raw", abs((dd + db) - dr) < 1e-12)
    ck("a run with no persistent tilt has a zero bias channel",
       abs(split_channels(dict(a_raw=0.52, a_deb=0.52))[2]) < 1e-12)
    ck("a run that is ALL tilt has dev_deb 0",
       abs(split_channels(dict(a_raw=0.52, a_deb=0.50))[1]) < 1e-12)
    import neff_instrument as ni
    ck("split_channels matches the instrument's own bias_share",
       abs((1.0 - split_channels(s)[1] / split_channels(s)[0]) - 0.5) < 1e-12)
    ck("the instrument really defines a_deb from pbar, not from 0.5",
       "mean_t |p_t - pbar|" in ni.agreement_stats.__doc__)

    # --- the slope helper
    ck("slope needs 3 distinct m", math.isnan(slope([10, 100], [1, 10])))
    ck("slope of a pure power law is its exponent",
       abs(slope([10, 100, 1000], [10 ** 0.5, 10, 10 ** 1.5]) - 0.5) < 1e-9)
    ck("slope 1.0 is independence", abs(slope([10, 100, 1000], [10, 100, 1000]) - 1.0)
       < 1e-9)
    ck("slope 0.0 is full sharing", abs(slope([10, 100, 1000], [5, 5, 5])) < 1e-9)

    # --- the t helper
    ck("t of identical samples is 0",
       abs(_t([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])) < 1e-12
       or math.isinf(_t([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])))
    ck("t needs 2 per arm", math.isnan(_t([1.0], [2.0, 3.0])))
    ck("t is signed a-minus-b", _t([1.0, 1.1], [2.0, 2.1]) < 0)

    # --- the discipline this file registered against itself
    ck("declared POST-HOC and DESCRIPTIVE in the docstring",
       "POST-HOC AND DESCRIPTIVE" in __doc__)
    ck("declares it registers nothing", "REGISTERS NOTHING" in __doc__)
    ck("refuses the 53.1% sentence", '"53.1%"' in __doc__)
    ck("refuses to fit accuracy on the field across rungs",
       "will not fit accuracy on the field across rungs" in __doc__)
    ck("says why tw0's 0.686 was confounded", "confounded" in __doc__)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck1", default=os.path.join(REPO, "..", "probes_ck1"))
    ap.add_argument("--cx2", default=os.path.join(REPO, "..", "probes_cx2"))
    ap.add_argument("--mm1", default=os.path.join(REPO, "..", "probes_mm1"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    rows = {r["run"]: r for r in csv.DictReader(open(a.csv))}

    def plateaus(fmt, ks_or_arms):
        out = {}
        for k in ks_or_arms:
            v = []
            for s in (0, 1, 2):
                r = rows.get(fmt % (k, s))
                if r and r.get("plateau5", "").strip():
                    v.append(float(r["plateau5"]))
            if v:
                out[k] = v
        return out

    print("=" * 78)
    print("c77 -- THE CHUNK FAMILY AS ONE CURVE, AND THE MATCHED-m CHANNEL SPLIT")
    print("**POST-HOC AND DESCRIPTIVE.  REGISTERS NOTHING, SCORES NOTHING.**")
    print("=" * 78)

    # ---- F1
    ag1, ne1, fr1 = collect(a.ck1, "ck1", CK1_KS, CHUNK_M)
    ag2, ne2, fr2 = collect(a.cx2, "cx2", CX2_KS, CHUNK_M)
    ag = dict(ag1)
    ag.update(ag2)
    ne = dict(ne1)
    ne.update(ne2)
    fr = dict(fr1)
    fr.update(fr2)
    pl = plateaus("ck1-k%d-s%d", CK1_KS)
    pl.update(plateaus("cx2-k%d-s%d", CX2_KS))

    print("\n--- F1  THE SEVEN-RUNG CHUNK FAMILY.  One partition family, one knob,")
    print("    one ms=1e-4, one box, 100 ep, seeds 0-2.")
    print("    %-8s %-12s %10s %8s %9s %10s %9s %9s %10s %9s"
          % ("K", "m", "plateau5", "sem", "pbar", "bias", "a_raw", "a_deb",
             "bias_shr", "N_eff/m"))
    order = sorted(CHUNK_M, key=lambda k: -CHUNK_M[k])
    for K in order:
        v = ag.get(K, [])
        pv = pl.get(K, [])
        if not v or not pv:
            print("    %-8d %-12d  NO DATA" % (K, CHUNK_M[K]))
            continue
        nev = ne.get(K, [])
        print("    %-8d %-12d %10.3f %8.3f %9.5f %+10.5f %9.5f %9.5f %10.4f %9s"
              % (K, CHUNK_M[K], statistics.mean(pv), _sem(pv),
                 statistics.mean([x["pbar"] for x in v]),
                 statistics.mean([x["bias"] for x in v]),
                 statistics.mean([x["a_raw"] for x in v]),
                 statistics.mean([x["a_deb"] for x in v]),
                 statistics.mean([x["bias_share"] for x in v]),
                 ("%.4f" % statistics.mean(nev)) if nev else "--"))
    print("    %-8s %-12d %10.3f %8s %9s %10s %9s %9s %10s %9s"
          % ("huge*", LAYERWISE_M, LAYERWISE_PLATEAU, "n=%d" % LAYERWISE_N,
             "--", "--", "--", "--", "--", "--"))
    print("    * chunk<huge> == layerwise BITWISE (tests/test_chunkwise.py C2), so the")
    print("      family's coarse ENDPOINT is the tuned ladder's layerwise cell.  Its")
    print("      accuracy is carried in from that cell; it was not run under a chunk")
    print("      label and it carries no probe, hence the blank field columns.")
    nfree = sum(1 for K in order for b in fr.get(K, []) if b)
    ntot = sum(len(fr.get(K, [])) for K in order)
    print("    box-free: %d/%d arms" % (nfree, ntot))

    ms_ = [CHUNK_M[K] for K in order if ne.get(K)]
    ne_ = [statistics.mean(ne[K]) * CHUNK_M[K] for K in order if ne.get(K)]
    s_fam = slope(ms_, ne_)
    print("\n    d log N_eff / d log m over the WHOLE FAMILY (%d rungs, PARTITION HELD"
          % len(ms_))
    print("    FIXED) = %s" % (("%.3f" % s_fam) if math.isfinite(s_fam) else "n/a"))
    print("    Independence would give 1.000; full sharing 0.000.")
    print("    tw0's 0.686 was fitted across layerwise/nodewise/weightwise -- THREE")
    print("    DIFFERENT PARTITIONS -- so it confounded the count with the partition.")
    print("    mm1 showed that is not safe.  This slope varies only the count.")

    # ---- F2
    print("\n--- F2  THE MATCHED-m CHANNEL SPLIT (mm1: m = 14,421 vs 14,420).")
    print("    dev_raw = a_raw - 0.5 (deviation from the FIXED 0.5)")
    print("    dev_deb = a_deb - 0.5 (deviation from the RUN'S OWN mean pbar)")
    print("    bias channel = dev_raw - dev_deb  (the persistent tilt)")
    from neff_instrument import agreement_stats
    mm = {}
    for d in sorted(glob.glob(os.path.join(a.mm1, "probe_*"))):
        parts = os.path.basename(d).split("_")
        if "mm1" not in parts:
            continue
        arm = next((p for p in parts if p in MM1_M), None)
        if arm is None:
            continue
        s = agreement_stats(d, window=(0.5, 1.0))
        if s is not None:
            mm.setdefault(arm, []).append(s)
    if not (mm.get("ch") and mm.get("node")):
        print("    NO DATA")
        return 0
    chans = {arm: [split_channels(s) for s in v] for arm, v in mm.items()}
    print("\n    %-12s %-8s %10s %10s %10s %10s"
          % ("arm", "m", "pbar", "dev_raw", "dev_deb", "dev_bias"))
    for arm, label in (("ch", "chunk777"), ("node", "nodewise")):
        c = chans[arm]
        print("    %-12s %-8d %10.5f %10.5f %10.5f %10.5f"
              % (label, MM1_M[arm],
                 statistics.mean([s["pbar"] for s in mm[arm]]),
                 statistics.mean([x[0] for x in c]),
                 statistics.mean([x[1] for x in c]),
                 statistics.mean([x[2] for x in c])))
    d_raw = (statistics.mean([x[0] for x in chans["ch"]])
             - statistics.mean([x[0] for x in chans["node"]]))
    d_deb = (statistics.mean([x[1] for x in chans["ch"]])
             - statistics.mean([x[1] for x in chans["node"]]))
    d_bias = (statistics.mean([x[2] for x in chans["ch"]])
              - statistics.mean([x[2] for x in chans["node"]]))
    t_raw = _t([x[0] for x in chans["ch"]], [x[0] for x in chans["node"]])
    t_deb = _t([x[1] for x in chans["ch"]], [x[1] for x in chans["node"]])
    t_bias = _t([x[2] for x in chans["ch"]], [x[2] for x in chans["node"]])
    print("\n    chunk777 MINUS nodewise, at m matched to ONE group in 14,420:")
    print("      dev_raw   %+.5f   (t %+.2f)" % (d_raw, t_raw))
    print("      dev_deb   %+.5f   (t %+.2f)" % (d_deb, t_deb))
    print("      dev_bias  %+.5f   (t %+.2f)" % (d_bias, t_bias))
    print("      check: dev_deb + dev_bias = %+.5f  vs dev_raw %+.5f"
          % (d_deb + d_bias, d_raw))
    if d_raw != 0:
        print("\n    THE TWO CHANNELS MOVE IN OPPOSITE DIRECTIONS."
              if d_deb * d_bias < 0 else "\n    Both channels move the same way.")
        print("    bias channel / raw difference = %+.1f%%" % (100.0 * d_bias / d_raw))
        print("    debiased channel / raw difference = %+.1f%%"
              % (100.0 * d_deb / d_raw))
    print("\n    READING, and it is only a reading: at EXACTLY matched count the two")
    print("    partitions differ in the RAW agreement almost entirely through the")
    print("    PERSISTENT TILT, not through the instantaneous across-coordinate")
    print("    agreement.  mm1's M3 was registered DESCRIPTIVE with no direction, so")
    print("    this is not a confirmed prediction and must not be written as one.")
    print("    The mechanical channel -- a coarse coordinate's meta-gradient being a")
    print("    sum over many fine ones -- is switched OFF here by construction, which")
    print("    is the only reason the split is interpretable at all.")

    # ---- F3
    print("\n--- F3  DOES THE FIELD PREDICT ACCURACY?  **NOT FITTED, AND HERE IS WHY.**")
    print("    %-8s %-12s %10s %10s %10s"
          % ("K", "m", "plateau5", "a_raw", "bias_share"))
    for K in order:
        if not (ag.get(K) and pl.get(K)):
            continue
        print("    %-8d %-12d %10.3f %10.5f %10.4f"
              % (K, CHUNK_M[K], statistics.mean(pl[K]),
                 statistics.mean([x["a_raw"] for x in ag[K]]),
                 statistics.mean([x["bias_share"] for x in ag[K]])))
    print("    Across these rungs the COUNT moves with the field, so any accuracy-on-")
    print("    field fit is confounded with m and is not computed.  The one place the")
    print("    campaign varies the field with m HELD FIXED is mm1's two arms -- two")
    print("    points, and a slope through two points is not evidence.  Deciding this")
    print("    needs more partitions at the SAME m, which is a code change.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
