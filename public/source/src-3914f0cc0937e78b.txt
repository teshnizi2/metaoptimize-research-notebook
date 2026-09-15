#!/usr/bin/env python3
"""c79_field_vs_accuracy.py -- does the sign-agreement FIELD order the arms the way
ACCURACY does, at EXACTLY matched group count?

**POST-HOC.  DESCRIPTIVE.  REGISTERS NOTHING.  PREDICTS NOTHING.**
Written AFTER `analysis/c78_bn1_score.py` printed its T1/T4 tables, on data that already
existed in three batches (mm1, pp1, bn1).  It is a re-reading of measurements already
taken, assembled into one table.  **No verdict below may be quoted as a tested
prediction, and none of it can void or amend a registered gate.**  Its ONLY legitimate
use is to MOTIVATE a future pre-registered batch.

WHY IT EXISTS
  The brief's default direction (C) is to make the sign-agreement measurement the
  project: extend it across granularities, model sizes and datasets, and characterise
  how agreement varies with block size.  That programme presupposes something nobody in
  the Adam-mini / Adalayer / SGG line has checked and this campaign has never registered:
  **that the field is informative about which partition actually optimises better.**
  FINDINGS 77.5 found ONE post-hoc contrast where it is not.  This script asks the same
  question of every matched-count pair the campaign owns.

THE COMPARISON, and why it is restricted to MATCHED COUNT
  N_eff/m is a per-group quantity and moves strongly and mechanically with m (c77's
  seven-rung chunk family, PARTITION HELD FIXED: d log N_eff / d log m = 0.850).  Ranking
  arms of DIFFERENT m by N_eff/m therefore mostly ranks m.  Only pairs whose group COUNT
  is matched isolate the partition.  Three such pairs exist:

    pair          batch  m           contrast
    node/chunk    mm1    14,420 vs 14,421   architecture-aligned vs uniform
    node/perm     pp1    14,420 vs 14,420   alignment ONLY (size multiset identical)
    n1d/chunk     bn1     4,851 vs  4,851   as mm1 but with the size-1 tail removed

  Every pair is WITHIN ONE BATCH, so the +-0.25 pp cross-batch offset (CORRECTIONS 106.3)
  cancels in both columns.

WHAT "CONCORDANT" MEANS HERE
  A sign convention has to be fixed to speak at all.  This script uses the one the
  noise-averaging literature implies -- **more effective independence (higher N_eff/m)
  means more information per meta-step, hence better accuracy** -- and labels a pair
  CONCORDANT when sign(dN_eff/m) == sign(dplateau5).  **This convention was chosen AFTER
  the data were seen and is not a hypothesis this script tests.**  It is a bookkeeping
  device: what matters below is whether ONE fixed convention -- either one -- orders all
  three pairs, not which one.
  A pair is called UNRESOLVED on a channel when |t| < 2.0 on that channel.

WHAT THIS SCRIPT WILL NOT DO
  * It will not re-derive an independence null and will not print "53.1%".
  * It will not compute a rank correlation across UNMATCHED m (see above).
  * It will not touch, reorder or annotate any registered gate in c76/c77/c78.
  * It will not treat bn1's T1 as anything other than the UNDECIDED its own registered
    bands returned.
  * It will not evaluate bn1's T5, which its scorer declared NOT APPLICABLE.

USAGE
  python3 analysis/c79_field_vs_accuracy.py --selftest
  python3 analysis/c79_field_vs_accuracy.py --csv results/all_runs.csv
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

from c52_boxfree import occupancy                                    # noqa: E402
from c55_neff_noise import BOXES                                     # noqa: E402

EPOCHS = 100
UNRESOLVED_T = 2.0

# (label, probe root, probe tag, csv arm prefix, expected m, box key)
ARM = {
    "mm1.node": ("../probes_mm1", "node", "mm1-node", 14420, "mm1"),
    "mm1.ch":   ("../probes_mm1", "ch",   "mm1-ch",   14421, "mm1"),
    "pp1.node": ("../probes_pp1", "node", "pp1-node", 14420, "pp1"),
    "pp1.perm": ("../probes_pp1", "perm", "pp1-perm", 14420, "pp1"),
    "pp1.ch":   ("../probes_pp1", "ch",   "pp1-ch",   14421, "pp1"),
    "bn1.n1d":  ("../probes_bn1", "n1d",  "bn1-n1d",   4851, "bn1"),
    "bn1.c23":  ("../probes_bn1", "c23",  "bn1-c23",   4851, "bn1"),
    "bn1.node": ("../probes_bn1", "node", "bn1-node", 14420, "bn1"),
}

# The three MATCHED-COUNT pairs.  (name, hi arm, lo arm, what the contrast isolates)
PAIRS = [
    ("mm1  chunk777  - nodewise",   "mm1.ch",  "mm1.node", "aligned vs uniform, m=14,420"),
    ("pp1  permnode  - nodewise",   "pp1.perm", "pp1.node", "ALIGNMENT only, size multiset identical"),
    ("bn1  chunk2325 - nodewise1d", "bn1.c23", "bn1.n1d",  "aligned vs uniform, m=4,851, tail removed"),
]


def _sem(v):
    if len(v) < 2:
        return float("nan")
    return statistics.stdev(v) / math.sqrt(len(v))


def _t(a, b):
    """Welch t on two small samples; nan if either is degenerate."""
    sa, sb = _sem(a), _sem(b)
    se = math.sqrt(sa * sa + sb * sb)
    if not (se > 0):
        return float("nan")
    return (statistics.mean(a) - statistics.mean(b)) / se


def plateaus(csv_path, prefix):
    """plateau5 of every 100-epoch run whose name starts with `prefix`-s."""
    out = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if not r["run"].startswith(prefix + "-s"):
                continue
            if int(r["epochs_done"] or 0) != EPOCHS:
                continue
            if not r["plateau5"]:
                continue
            out.append(float(r["plateau5"]))
    return sorted(out)


def neff_of(root, tag):
    """N_eff/m per seed for one arm, by the SAME code path as c77_family_curve."""
    import probe5_window as p5w
    root = os.path.join(REPO, root) if not os.path.isabs(root) else root
    vals, nfree, ntot = [], 0, 0
    key = os.path.basename(root.rstrip("/")).replace("probes_", "")
    lo, hi = BOXES[key][0], BOXES[key][1]
    for d in sorted(glob.glob(os.path.join(root, "probe_*"))):
        parts = os.path.basename(d).split("_")
        if len(parts) < 3 or parts[0] != "probe" or parts[1] != tag:
            continue
        w = p5w.reduce_dir(d, windows=(("steady .5-1", (0.5, 1.0)),))
        if w is None:
            continue
        ww = w["win"].get("steady .5-1")
        if not ww or ww["rho_s"] is None:
            continue
        mm, rho = float(w["n_tot"]), ww["rho_s"]
        vals.append(1.0 / (1.0 + (mm - 1.0) * rho))
        o = occupancy(d, lo, hi)
        ntot += 1
        nfree += int(o["rec_lo"] < 0.05 and o["rec_hi"] < 0.05)
    return sorted(vals), nfree, ntot


def selftest():
    p = n = 0

    def ck(name, cond):
        nonlocal p, n
        n += 1
        p += bool(cond)
        print("    %-70s %s" % (name, "ok" if cond else "FAIL"))

    print("c79_field_vs_accuracy selftest")

    # --- the statistics -----------------------------------------------------
    ck("_sem of a constant sample is 0", _sem([1.0, 1.0, 1.0]) == 0.0)
    ck("_sem of n=1 is nan", math.isnan(_sem([1.0])))
    ck("_t of identical samples is 0",
       _t([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0)
    ck("_t is antisymmetric",
       abs(_t([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) + _t([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])) < 1e-12)
    ck("_t on zero-variance pairs is nan", math.isnan(_t([1.0, 1.0], [2.0, 2.0])))

    # --- the pair table is matched by construction --------------------------
    for name, hi, lo, _ in PAIRS:
        mh, ml = ARM[hi][3], ARM[lo][3]
        ck("PAIR %-30s counts matched to <=1 group (%d vs %d)" % (name.split()[0] + "/" + hi.split(".")[1], mh, ml),
           abs(mh - ml) <= 1)
    ck("exactly 3 matched-count pairs are declared", len(PAIRS) == 3)
    ck("every pair is WITHIN one batch",
       all(h.split(".")[0] == l.split(".")[0] for _, h, l, _ in PAIRS))
    ck("bn1.node is carried but is in NO pair (it is count-mismatched)",
       "bn1.node" in ARM and all("bn1.node" not in (h, l) for _, h, l, _ in PAIRS))

    # --- every arm's box is registered --------------------------------------
    for k, (root, tag, pref, m, box) in ARM.items():
        ck("ARM %-10s box %-4s is registered in c55 BOXES" % (k, box), box in BOXES)

    # --- the honesty guards, asserted against THIS FILE'S OWN TEXT ----------
    src = open(os.path.abspath(__file__)).read()
    ck("declares itself POST-HOC in the docstring", "**POST-HOC." in src)
    ck("declares it REGISTERS NOTHING", "REGISTERS NOTHING" in src)
    ck("declares the sign convention was chosen AFTER the data",
       "chosen AFTER the data were seen" in src)
    ck("states why UNMATCHED m is excluded", "mostly ranks m" in src)
    ck("cites the 0.850 within-family exponent as the reason",
       "0.850" in src)
    ck("refuses to reproduce the 53.1% sentence",
       'will not print "53.1%"' in src)
    ck("states it cannot void a registered gate", "cannot void or amend a registered gate" in src)
    ck("records that bn1 T1 stays UNDECIDED", "the UNDECIDED its own registered" in src)
    ck("records that bn1 T5 stays NOT APPLICABLE", "NOT APPLICABLE" in src)
    ck("names the cross-batch offset it cancels", "106.3" in src)
    ck("names the prior-art line the question bears on", "Adam-mini" in src)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    print("=" * 78)
    print("c79 -- DOES THE FIELD ORDER THE ARMS THE WAY ACCURACY DOES?")
    print("**POST-HOC, DESCRIPTIVE, REGISTERS NOTHING.**  Matched group count only.")
    print("=" * 78)

    acc, fld = {}, {}
    print("\n--- the arms")
    print("    %-12s %-8s %-7s  %-22s %-24s" % ("arm", "m", "seeds", "plateau5", "N_eff/m"))
    for k, (root, tag, pref, m, box) in ARM.items():
        pl = plateaus(a.csv, pref)
        nf, free, tot = neff_of(root, tag)
        acc[k], fld[k] = pl, nf
        print("    %-12s %-8d %-7s  %8.3f +-%-11.3f %8.4f +-%-8.4f  box-free %d/%d" % (
            k, m, "%d/%d" % (len(pl), len(nf)),
            statistics.mean(pl) if pl else float("nan"), _sem(pl) if len(pl) > 1 else float("nan"),
            statistics.mean(nf) if nf else float("nan"), _sem(nf) if len(nf) > 1 else float("nan"),
            free, tot))

    print("\n--- THE THREE MATCHED-COUNT PAIRS")
    print("    convention (chosen post-hoc, see docstring): higher N_eff/m -> higher plateau = CONCORDANT")
    rows = []
    for name, hi, lo, what in PAIRS:
        da = statistics.mean(acc[hi]) - statistics.mean(acc[lo])
        df = statistics.mean(fld[hi]) - statistics.mean(fld[lo])
        ta, tf = _t(acc[hi], acc[lo]), _t(fld[hi], fld[lo])
        a_res = abs(ta) >= UNRESOLVED_T
        f_res = abs(tf) >= UNRESOLVED_T
        if not a_res:
            lab = "ACCURACY UNRESOLVED"
        elif not f_res:
            lab = "FIELD UNRESOLVED"
        elif (df > 0) == (da > 0):
            lab = "CONCORDANT"
        else:
            lab = "ANTI-CONCORDANT"
        rows.append((name, da, ta, df, tf, lab, what))
        print("\n    %s" % name)
        print("      isolates : %s" % what)
        print("      dplateau5 = %+7.3f pp   (t %6.2f)  %s" % (da, ta, "resolved" if a_res else "UNRESOLVED"))
        print("      dN_eff/m  = %+7.4f      (t %6.2f)  %s" % (df, tf, "resolved" if f_res else "UNRESOLVED"))
        print("      -> **%s**" % lab)

    print("\n--- READING (POST-HOC; motivates a batch, settles nothing)")
    conc = sum(1 for r in rows if r[5] == "CONCORDANT")
    anti = sum(1 for r in rows if r[5] == "ANTI-CONCORDANT")
    unres = sum(1 for r in rows if "UNRESOLVED" in r[5])
    print("    CONCORDANT %d   ANTI-CONCORDANT %d   UNRESOLVED-on-a-channel %d   of %d pairs"
          % (conc, anti, unres, len(rows)))
    if conc and anti:
        print("    The two resolved pairs point OPPOSITE ways under ONE fixed convention,")
        print("    so NO monotone function of N_eff/m orders these arms by accuracy.")
    elif anti and not conc:
        print("    Every resolved pair is ANTI-concordant under the literature-facing")
        print("    convention; the field's sign is systematically WRONG about accuracy here.")
    elif conc and not anti:
        print("    Every resolved pair is concordant; the field is NOT contradicted.")
    print("    In ALL cases this is 2-3 pairs at n=3 on ONE architecture, ONE dataset,")
    print("    ONE meta stepsize.  It is a REASON TO REGISTER A TEST, not a result.")

    print("\n--- WHAT WOULD MAKE THIS A RESULT")
    print("    All pairs in ONE batch (these span three), with the concordance rule and")
    print("    its sign convention fixed BEFORE the data exist, and with at least one")
    print("    matched-count pair whose FIELD values have never been read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
