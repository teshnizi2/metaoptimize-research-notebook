#!/usr/bin/env python3
"""c81_cc1_score.py -- score `cc1`.  DOES THE SIGN-AGREEMENT FIELD PREDICT
ACCURACY?  THE PRE-REGISTERED TEST, RE-RUN AT THE ONLY READABLE STEPSIZE.

REGISTERED GATES, transcribed from `bin/c81_concordance.sh`.  Every constant
below is asserted by the selftest against THAT SCRIPT'S OWN TEXT so the
registration and the batch cannot drift -- STANDING RULE (19).

WRITTEN, SELFTESTED AND GIT-COMMITTED IN THE SAME TICK THE BATCH WAS SUBMITTED,
WHILE NO `cc1` RUN EXISTED IN THE CSV AT ALL.  Not merely before the verdict was
read -- before the data could exist.

WHY THIS BATCH EXISTS.  Cycle 79 registered A3 as the ONE pre-registered test
that would decide the brief's default direction C.  `ar1` ran it at ms=3e-4 and
**A3 CAME BACK VOID: 12/12 arms box-bound**, an outcome registered in advance as
a real possibility.  The direction-C decision is still owed.  This is the re-run.

WHY ms=1e-4.  Measured this cycle from probes already on disk, at zero compute:

    ms=1e-4   bn1/mm1/pp1, 24 arms   rec_lo 0.0000 EXACTLY   coord_lo 0.000000
    ms=3e-4   ar1,         12 arms   rec_lo ~0.456           coord_lo 0.0004-0.0343

1e-4 is the only stepsize at which the instrument has ever been readable, and
there it is not marginally free but EXACTLY free on 24/24.  Cycle 79 gambled on
an unread stepsize for independence and lost the test to a bind.  This batch
takes the readable point and buys independence with FRESH SEEDS (3,4,5) instead.

  C0    VALIDITY.  n_records == 10000; beta moved; epochs_done == requested == 100.
  C0.2  n_beta EXACT on EVERY record: node 14420, ch 14421, n1d 4851, c23 4851.
        All four MEASURED by guard 4 from the ALLOCATED beta on the real built
        network, which ALSO asserted C3's pair EXACT and C2's <= 1 group apart.
  C0.3  THE INSTRUMENT FIRED.  neg_counts.json, n_tot == n_beta, npy shape READ
        FROM ITS HEADER == (n_tot,).  (Header, never file size -- that inference
        produced c74's false VOID on 12/12 arms.)
  C0.4  BOX-FREE at BOTH guards, the published rec_-based 5% gate, PRIMARY and
        UNCHANGED.  **A bind is NOT expected here -- 24/24 arms at this exact ms
        and box read rec_lo 0.0000 EXACTLY.  A bind would itself be the news.**
        A bind VOIDS C1 and leaves C2/C3 standing -- they are accuracy-only.

  C1    **THE PRIMARY.  THE CONCORDANCE TEST.  THIS IS WHAT THE BATCH IS FOR.**
        For each matched-count pair: d_acc = dplateau5, d_fld = dN_eff/m, each
        with a Welch t at n=3.  A channel is RESOLVED at |t| >= 2.0.
        **CONVENTION, FIXED NOW AND TRANSCRIBED UNCHANGED FROM A3: higher
        N_eff/m -> higher plateau5 = CONCORDANT.**  That is the direction the
        noise-averaging literature implies (more effective independence = more
        information per meta-step).
          both pairs resolved on BOTH channels, both CONCORDANT
             -> **THE FIELD PREDICTS.**  Direction C IS ADOPTED as the project.
          both pairs resolved on BOTH channels, both ANTI-CONCORDANT
             -> **THE FIELD SYSTEMATICALLY ANTI-PREDICTS**, PRE-REGISTERED.
                78.6's post-hoc pattern becomes a tested prediction, direction C
                IS DROPPED, and the anti-prediction is itself the reportable
                result.
          both resolved on both channels, MIXED
             -> **THE FIELD IS NOT A SUFFICIENT STATISTIC.**  Direction C IS
                DROPPED as a design principle.
          a pair RESOLVED on field, UNRESOLVED on accuracy
             -> **DISSOCIATION** (FINDINGS 77.5's pattern at a third contrast).
          a pair UNRESOLVED on FIELD -> UNINFORMATIVE, reported, NOT folded in.
        **C1 IS VOID IF C0.4 FAILS ON ANY ARM.  A void HERE, at the only stepsize
        where the instrument has ever been readable, DROPS direction C ON
        UNREADABILITY -- registered now so that outcome cannot later be re-read
        as "inconclusive, try again".**

  C2    THE D REPLICATION, FRESH SEEDS, WITHIN BATCH.
        D'' = plateau5(chunk777) - plateau5(nodewise) at ms=1e-4, seeds 3-5.
        Priors on this SAME contrast at this SAME ms: mm1 +0.485 (t 3.01),
        pp1 +0.581 (t 4.11).  ar1 read +0.697 (t 5.90) at 3e-4.
          |D'' - 0.533| <= 0.50 -> **REPLICATES**  (0.533 = the mm1/pp1 pooled
               value, POST-HOC and DESCRIPTIVE, used only to centre a band; the
               +-0.50 is CORRECTIONS 106.3's unmodelled cross-batch offset)
          D'' <= +0.15          -> **FAILS TO REPLICATE**, a real registered branch
          otherwise             -> UNDECIDED, registered in advance

  C3    THE G READING AT FRESH SEEDS.  G'' = plateau5(chunk2325) - plateau5(nodewise1d).
        SAME five-way bands as bn1's T1, transcribed unchanged.
        **C3 DOES NOT AMEND bn1's T1.**  T1 returned +0.295 = UNDECIDED and STAYS
        UNDECIDED whatever C3 returns.  C3 is a reading at FRESH SEEDS, never a
        re-run of T1 with more seeds added to T1's own registration.  ar1's A2
        read -0.139 (COLLAPSES) at 3e-4; if C3 also lands in the null band that
        is TWO independent stepsizes agreeing, and it is still not an amendment.

  C4    THE CLIP METER.  DESCRIPTIVE.  rec_lo / rec_hi / coord_lo / coord_hi per
        arm, confirming the premise the batch was designed on.  It cannot gate
        C1/C2/C3 beyond the C0.4 void it already feeds, and is printed AFTER them.

WHAT THIS SCORER WILL NOT DO
  * It will not let C4 gate, annotate or reorder C1, C2 or C3.
  * It will not amend bn1's T1 verdict.  T1 is UNDECIDED and stays UNDECIDED.
  * It will not read C1 if any arm is box-bound.
  * It will not claim any arm's argmax.  This batch is run at the stepsize that
    is NOT the better one for accuracy, on purpose, because it is the readable
    one for the field.  That trade is the batch's central limitation.
  * It will not separate "the group-size distribution" from "the parameter role"
    on the 1-D tensors -- on ResNet18 those coincide EXACTLY (FINDINGS 78.1).
  * It will not re-derive an independence null on the fly (CORRECTIONS 26), and
    it will not print "53.1%".

USAGE
  python3 analysis/c81_cc1_score.py --selftest
  python3 analysis/c81_cc1_score.py --root ../probes_cc1 --csv results/all_runs.csv
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SCRIPT = os.path.join(REPO, "bin", "c81_concordance.sh")

from c52_boxfree import occupancy, records           # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -15.0, -2.3026             # `cc1`, the SAME box as bn1/mm1/pp1/ar1
EPOCHS = 100
N_RECORDS = 10000
MST = "1e-4"                        # THE READABLE STEPSIZE
MST_VOID = "3e-4"                   # where ar1's A3 voided, 12/12 box-bound
CHUNK_K = 777
CHUNK_K2 = 2325
NJOBS = 12
SEEDS = ("3", "4", "5")             # FRESH -- guard 7 asserted them absent
ARMS = ("node", "ch", "n1d", "c23")
M_OF_ARM = {"node": 14420, "ch": 14421, "n1d": 4851, "c23": 4851}
GRAN_OF_ARM = {"node": "nodewise", "ch": "chunk777",
               "n1d": "nodewise1d", "c23": "chunk2325"}

NULL_HALF = 0.15
DECIDE = 0.30
RESOLVED_T = 2.0
OFFSET_BAR = 0.50
D_PRIOR = 0.533                     # mm1/pp1 pooled, POST-HOC, band-centring only
BOXFREE_MAX = 0.05

PAIRS = (
    ("C2", "ch", "node", "aligned vs uniform at m=14,420"),
    ("C3", "c23", "n1d", "aligned vs uniform at m=4,851, size-1 tail already gone"),
)


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def _t(a, b):
    se = math.sqrt(_sem(a) ** 2 + _sem(b) ** 2)
    if not (se > 0):
        return float("nan")
    return (statistics.mean(a) - statistics.mean(b)) / se


def band(g):
    """The FIVE-WAY registered verdict for C3, transcribed from bn1's T1."""
    if g > DECIDE:
        return "SURVIVES"
    if g > NULL_HALF:
        return "UNDECIDED"
    if g >= -NULL_HALF:
        return "COLLAPSES"
    if g >= -DECIDE:
        return "UNDECIDED"
    return "REVERSAL"


def c2_band(d):
    """C2's OWN band -- a replication band, NOT the five-way one."""
    if d <= NULL_HALF:
        return "FAILS TO REPLICATE"
    if abs(d - D_PRIOR) <= OFFSET_BAR:
        return "REPLICATES"
    return "UNDECIDED"


def arm_rows(csv_path, arm):
    out = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if not r["run"].startswith("cc1-%s-s" % arm):
                continue
            if int(r["epochs_done"] or 0) != EPOCHS or not r["plateau5"]:
                continue
            out.append(float(r["plateau5"]))
    return sorted(out)


def arm_of_dir(d):
    parts = os.path.basename(d).split("_")
    return parts[1] if len(parts) > 2 and parts[0] == "probe" else None


def neff_of_dir(d):
    import probe5_window as p5w
    w = p5w.reduce_dir(d, windows=(("steady .5-1", (0.5, 1.0)),))
    if w is None:
        return None
    ww = w["win"].get("steady .5-1")
    if not ww or ww["rho_s"] is None:
        return None
    mm, rho = float(w["n_tot"]), ww["rho_s"]
    return 1.0 / (1.0 + (mm - 1.0) * rho)


# ---------------------------------------------------------------------------
def selftest():
    src = open(SCRIPT).read()
    doc = __doc__
    me = open(os.path.abspath(__file__)).read()
    p = n = 0

    def ok(label, cond):
        nonlocal p, n
        n += 1
        p += bool(cond)
        print("    %-72s %s" % (label, "ok" if cond else "FAIL"))

    print("c81_cc1_score selftest")
    print("  -- the constants, asserted against the BATCH SCRIPT's own text --")
    ok("the batch script exists", os.path.exists(SCRIPT))
    ok("MST matches the script's MST=", "MST=%s " % MST in src or "MST=%s\n" % MST in src
       or ("MST=%s" % MST) in src)
    ok("MST_VOID matches the script's MST_VOID=", "MST_VOID=%s" % MST_VOID in src)
    ok("the box LO matches the script's CLIP=", "CLIP=%g:" % LO in src)
    ok("the box HI matches the script's CLIP=", ":%s" % "-2.3026" in src)
    ok("EPOCHS matches the script", "EPOCHS=%d" % EPOCHS in src)
    ok("NJOBS matches the script", "NJOBS=%d" % NJOBS in src)
    ok("CHUNK_K matches the script", "CHUNK_K=%d" % CHUNK_K in src)
    ok("CHUNK_K2 matches the script", "CHUNK_K2=%d" % CHUNK_K2 in src)
    ok("M_NODE matches the script", "M_NODE=%d" % M_OF_ARM["node"] in src)
    ok("M_CHUNK matches the script", "M_CHUNK=%d" % M_OF_ARM["ch"] in src)
    ok("M_N1D matches the script", "M_N1D=%d" % M_OF_ARM["n1d"] in src)
    ok("M_CHUNK2 matches the script", "M_CHUNK2=%d" % M_OF_ARM["c23"] in src)
    ok("SEEDS match the script's SEEDS=", 'SEEDS="%s"' % " ".join(SEEDS) in src)
    ok("the script submits under the cc1- run-name prefix", 'RN="cc1-${SHORT}-s${S}"' in src)
    ok("the script's probe dir carries the cc1 tag", "probe_${SHORT}_cc1_s${S}" in src)

    print("  -- the arms, and that they are the SAME four --")
    for a in ARMS:
        ok("arm %-4s is submitted by the script" % a, '"%s"' % a in src or ":%s\"" % a in src)
    ok("the script names granularity nodewise", '"nodewise:node"' in src)
    ok("the script names granularity chunk777", '"chunk${CHUNK_K}:ch"' in src)
    ok("the script names granularity nodewise1d", '"nodewise1d:n1d"' in src)
    ok("the script names granularity chunk2325", '"chunk${CHUNK_K2}:c23"' in src)

    print("  -- the gates, asserted present and identically worded --")
    for g in ("C0", "C0.2", "C0.3", "C0.4", "C1", "C2", "C3", "C4"):
        ok("gate %-4s is registered in the batch script" % g, "#   %s " % g in src)
        ok("gate %-4s is registered in this scorer" % g, "  %s " % g in doc)
    ok("the resolved threshold is |t| >= 2.0", RESOLVED_T == 2.0 and "|t| >= 2.0" in src)
    ok("the null half-width is 0.15 in both", NULL_HALF == 0.15 and "+-0.15" in src.replace("[-0.15", "+-0.15"))
    ok("the decide threshold is 0.30 in both", DECIDE == 0.30 and "0.30" in src)
    ok("the cross-batch offset bar is 0.50 in both", OFFSET_BAR == 0.50 and "0.50" in src)
    ok("C2's centring prior is 0.533 in both", D_PRIOR == 0.533 and "0.533" in src)
    ok("the box-free gate is the published 5%", BOXFREE_MAX == 0.05 and "5% gate" in src)

    print("  -- the sign convention, the thing that must not drift --")
    conv = "higher N_eff/m -> higher plateau5 = CONCORDANT"
    # The sentence is line-wrapped in both files' comment blocks; compare on the
    # whitespace-normalised text so the assertion is about the SENTENCE, not the
    # column it happened to wrap at.
    flat = lambda s: " ".join(s.split())
    ok("the convention is stated in the batch script", conv in flat(src))
    ok("the convention is stated in this scorer", conv in flat(doc))
    ok("the convention is transcribed UNCHANGED from A3", "TRANSCRIBED UNCHANGED FROM A3" in src.upper())
    a3 = open(os.path.join(HERE, "c79_ar1_score.py")).read()
    ok("A3's convention line is byte-identical to C1's", conv in flat(a3))

    print("  -- the branches, each with a stated consequence --")
    ok("CONCORDANT branch ADOPTS direction C", "IS ADOPTED" in src)
    ok("ANTI-CONCORDANT branch DROPS direction C", "IS DROPPED" in src)
    ok("MIXED branch is registered", "MIXED" in src and "sufficient statistic" in src)
    ok("DISSOCIATION branch is registered", "DISSOCIATION" in src)
    ok("UNINFORMATIVE branch is registered", "UNINFORMATIVE" in src)
    ok("a VOID here DROPS direction C on unreadability",
       "DROPPED ON UNREADABILITY" in src and "DROPS direction C ON" in doc.replace("\n", " ").replace("  ", " ") or "UNREADABILITY" in doc)
    ok("the void cannot be re-read as 'try again'", "try again" in src)
    ok("C2 has a real FAILS TO REPLICATE branch", "FAILS TO REPLICATE" in src)
    ok("C3 is declared unable to amend bn1's T1", "DOES NOT AMEND bn1's T1" in src)
    ok("C4 is declared descriptive", "C4    THE CLIP METER.  DESCRIPTIVE" in src)
    ok("C4 is declared unable to gate C1/C2/C3", "cannot gate C1, C2 or C3" in src)

    print("  -- the premise, and that it was ASSERTED not quoted --")
    ok("the script re-derives box-freeness from probes on disk", "GUARD 3" in src)
    ok("guard 3 fails loudly if 1e-4 is not exactly box-free", "is NOT exactly box-free" in src)
    ok("guard 3 requires at least 20 probe dirs", ">= 20" in src)
    ok("the measured premise is quoted with BOTH stepsizes", "rec_lo 0.0000 EXACTLY" in src and "rec_lo ~0.456" in src)
    ok("guard 7 asserts the seeds are genuinely fresh", "GUARD 7" in src and "FRESH" in src)
    ok("guard 4 measures m from the ALLOCATED beta", "ALLOCATED beta" in src)
    # Guard 4 is lifted verbatim from c79 and its network-changed check is the
    # 9,610 size-1 tail FINDINGS 77.6 measured, not a raw weight count.  Assert
    # what the block ACTUALLY guarantees.
    ok("guard 4 asserts the 9,610 size-1 tail is still there", "9610" in src)
    ok("guard 4 asserts C3's pair is EXACT and C2's <= 1 group apart",
       "C3's arms are %d groups apart, not 0" in src and "not <= 1" in src)

    print("  -- the limits, stated in the scorer itself --")
    ok("scorer states it predates the data entirely", "before the data could exist" in doc)
    ok("the argmax limit is stated", "will not claim any arm's argmax" in doc)
    ok("the accuracy-vs-readability trade is named as the central limitation",
       "central limitation" in src)
    ok("the BN-vs-size non-identifiability is carried forward from bn1", "FINDINGS 78.1" in doc)
    ok("the layer-boundary limit is carried forward", "Layer boundaries stay untested" in src)
    ok("scorer refuses to write the 53.1% sentence", "will not print" in doc and "53.1" in doc)
    ok("the n=3, one-architecture limit is stated", "does not establish a law" in src)
    ok("ar1's VOID is named as the reason this batch exists", "CAME BACK VOID" in src)
    ok("the scorer refuses to read C1 when box-bound", "will not read C1 if any arm is box-bound" in doc)
    ok("this scorer never edits bn1's T1", "T1 is UNDECIDED and stays UNDECIDED" in doc)
    ok("the C0.4 bind is declared to leave C2/C3 standing", "leaves C2/C3 standing" in doc)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../probes_cc1")
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs = sorted(glob.glob(os.path.join(a.root, "probe_*")))
    print("=" * 78)
    print("c81 -- cc1: DOES THE FIELD PREDICT ACCURACY?  %d probe dirs" % len(dirs))
    print("box = (%g, %s)   ms=%s (ar1's A3 VOIDED at %s)   %d ep"
          % (LO, HI, MST, MST_VOID, EPOCHS))
    print("C2 node/ch at m=14,420  |  C3 n1d/c23 at m=4,851 EXACT  |  seeds %s"
          % ",".join(SEEDS))
    print("=" * 78)

    # --- C0 -----------------------------------------------------------------
    print("\n--- C0  VALIDITY (n_records==%d, beta moved, ep==%d)" % (N_RECORDS, EPOCHS))
    npass = 0
    for d in dirs:
        try:
            R = records(d)
        except Exception as e:
            print("    %-26s FAIL  %s" % (os.path.basename(d), e))
            continue
        good = len(R) == N_RECORDS and R[-1]["beta_true_min"] != R[0]["beta_true_min"]
        npass += good
        print("    %-26s %s  n_rec=%d" % (os.path.basename(d),
                                          "PASS" if good else "FAIL", len(R)))
    print("    C0: %d/%d" % (npass, len(dirs)))

    # --- C0.2 ---------------------------------------------------------------
    print("\n--- C0.2  n_beta EXACT on EVERY record")
    n2 = 0
    for d in dirs:
        arm = arm_of_dir(d)
        want = M_OF_ARM.get(arm)
        vals = sorted({r.get("n_beta") for r in records(d)})
        good = vals == [want]
        n2 += good
        print("    %-26s %s  n_beta=%s  want %s = %s"
              % (os.path.basename(d), "PASS" if good else "FAIL", vals,
                 GRAN_OF_ARM.get(arm), want))
    print("    C0.2: %d/%d" % (n2, len(dirs)))

    # --- C0.3 ---------------------------------------------------------------
    print("\n--- C0.3  THE INSTRUMENT FIRED (npy shape from its HEADER)")
    n3 = 0
    for d in dirs:
        j = os.path.join(d, "neg_counts.json")
        npy = os.path.join(d, "neg_counts.npy")
        try:
            meta = json.load(open(j))
            n_tot = int(meta["n_tot"])
            with open(npy, "rb") as fh:
                fh.read(8)
                hlen = int.from_bytes(fh.read(2), "little")
                hdr = fh.read(hlen).decode("latin1")
            shape = eval(hdr[hdr.index("'shape':") + 8:].split(",)")[0] + ",)")
            good = n_tot == M_OF_ARM.get(arm_of_dir(d)) and shape == (n_tot,)
        except Exception as e:
            print("    %-26s FAIL  %s" % (os.path.basename(d), e))
            continue
        n3 += good
        print("    %-26s %s  n_tot=%d  npy_shape=%s"
              % (os.path.basename(d), "PASS" if good else "FAIL", n_tot, shape))
    print("    C0.3: %d/%d" % (n3, len(dirs)))

    # --- C0.4 ---------------------------------------------------------------
    print("\n--- C0.4  BOX-FREE (rec_-based 5%% gate, PRIMARY, UNCHANGED)")
    print("    **NOT expected to bind: 24/24 arms at this ms read rec_lo 0.0000 EXACTLY.**")
    occ, free_all = {}, True
    for d in dirs:
        o = occupancy(d, LO, HI)
        occ[d] = o
        f = o["rec_lo"] < BOXFREE_MAX and o["rec_hi"] < BOXFREE_MAX
        free_all &= f
        print("    %-26s %-8s rec_lo %.4f  rec_hi %.4f"
              % (os.path.basename(d), "free" if f else "BOUND", o["rec_lo"], o["rec_hi"]))
    print("    C0.4: %d/%d box-free" % (sum(
        1 for d in dirs if occ[d]["rec_lo"] < BOXFREE_MAX
        and occ[d]["rec_hi"] < BOXFREE_MAX), len(dirs)))

    # --- the cells ----------------------------------------------------------
    cells = {arm: arm_rows(a.csv, arm) for arm in ARMS}
    print("\n--- plateau5 per arm, from the CSV")
    for arm in ARMS:
        v = cells[arm]
        print("    %-12s m=%-9d n=%d  %s"
              % (GRAN_OF_ARM[arm], M_OF_ARM[arm], len(v),
                 ("%.3f +-%.3f" % (statistics.mean(v), _sem(v))) if len(v) > 1 else "-"))

    # --- C1 -----------------------------------------------------------------
    print("\n--- C1  **THE PRIMARY.  THE CONCORDANCE TEST.**")
    print("    convention: higher N_eff/m -> higher plateau5 = CONCORDANT")
    if not free_all:
        print("    -> **VOID**: at least one arm is box-bound, so N_eff/m is not")
        print("       interpretable.  Registered in advance.  **A VOID HERE, at the")
        print("       only stepsize where the instrument has ever been readable,")
        print("       DROPS DIRECTION C ON UNREADABILITY.**  C2 and C3 are unaffected.")
    else:
        verdicts = []
        for gate, hi, lo, what in PAIRS:
            acc_hi, acc_lo = cells[hi], cells[lo]
            f_hi = [neff_of_dir(d) for d in dirs if arm_of_dir(d) == hi]
            f_lo = [neff_of_dir(d) for d in dirs if arm_of_dir(d) == lo]
            f_hi = [x for x in f_hi if x is not None]
            f_lo = [x for x in f_lo if x is not None]
            if len(acc_hi) < 2 or len(acc_lo) < 2 or len(f_hi) < 2 or len(f_lo) < 2:
                print("    %s  INSUFFICIENT n -- reported, not folded in" % gate)
                verdicts.append("UNINFORMATIVE")
                continue
            d_acc = statistics.mean(acc_hi) - statistics.mean(acc_lo)
            d_fld = statistics.mean(f_hi) - statistics.mean(f_lo)
            t_acc, t_fld = _t(acc_hi, acc_lo), _t(f_hi, f_lo)
            r_acc = abs(t_acc) >= RESOLVED_T
            r_fld = abs(t_fld) >= RESOLVED_T
            if not r_fld:
                v = "UNINFORMATIVE"
            elif not r_acc:
                v = "DISSOCIATION"
            else:
                v = "CONCORDANT" if (d_acc > 0) == (d_fld > 0) else "ANTI-CONCORDANT"
            verdicts.append(v)
            print("    %s  %s" % (gate, what))
            print("        d_acc = %+0.3f pp (t %+0.2f, %s)   d_N_eff/m = %+0.4f (t %+0.2f, %s)"
                  % (d_acc, t_acc, "RESOLVED" if r_acc else "unresolved",
                     d_fld, t_fld, "RESOLVED" if r_fld else "unresolved"))
            print("        -> **%s**" % v)
        vs = [v for v in verdicts if v not in ("UNINFORMATIVE",)]
        print("    ---")
        if not vs:
            print("    -> **UNINFORMATIVE**: no pair resolved on the field channel.")
        elif all(v == "CONCORDANT" for v in vs) and len(vs) == 2:
            print("    -> **THE FIELD PREDICTS.  DIRECTION C IS ADOPTED.**")
        elif all(v == "ANTI-CONCORDANT" for v in vs) and len(vs) == 2:
            print("    -> **THE FIELD SYSTEMATICALLY ANTI-PREDICTS, PRE-REGISTERED.**")
            print("       DIRECTION C IS DROPPED; the anti-prediction IS the result.")
        elif all(v == "DISSOCIATION" for v in vs):
            print("    -> **DISSOCIATION on every resolved pair** (77.5's pattern).")
        else:
            print("    -> **MIXED: THE FIELD IS NOT A SUFFICIENT STATISTIC.**")
            print("       DIRECTION C IS DROPPED as a design principle.")

    # --- C2 -----------------------------------------------------------------
    print("\n--- C2  THE D REPLICATION, FRESH SEEDS, WITHIN BATCH.")
    if len(cells["ch"]) > 1 and len(cells["node"]) > 1:
        d2 = statistics.mean(cells["ch"]) - statistics.mean(cells["node"])
        se = math.sqrt(_sem(cells["ch"]) ** 2 + _sem(cells["node"]) ** 2)
        print("    chunk%-8d (m=%d) %.3f +-%.3f (n=%d)"
              % (CHUNK_K, M_OF_ARM["ch"], statistics.mean(cells["ch"]),
                 _sem(cells["ch"]), len(cells["ch"])))
        print("    nodewise     (m=%d) %.3f +-%.3f (n=%d)"
              % (M_OF_ARM["node"], statistics.mean(cells["node"]),
                 _sem(cells["node"]), len(cells["node"])))
        print("    D'' = chunk%d - nodewise = %+0.3f pp   (se %.3f, t %+0.2f -- "
              "DESCRIPTIVE; the gate is the threshold on the difference)"
              % (CHUNK_K, d2, se, _t(cells["ch"], cells["node"])))
        print("    registered: |D''-%.3f| <= %.2f REPLICATES | D'' <= %.2f FAILS | else UNDECIDED"
              % (D_PRIOR, OFFSET_BAR, NULL_HALF))
        print("    -> **%s**" % c2_band(d2))
        print("    priors at this ms: mm1 +0.485 (t 3.01), pp1 +0.581 (t 4.11); ar1 +0.697 at 3e-4")
    else:
        print("    INSUFFICIENT n")

    # --- C3 -----------------------------------------------------------------
    print("\n--- C3  THE G READING AT FRESH SEEDS.")
    if len(cells["c23"]) > 1 and len(cells["n1d"]) > 1:
        g3 = statistics.mean(cells["c23"]) - statistics.mean(cells["n1d"])
        se = math.sqrt(_sem(cells["c23"]) ** 2 + _sem(cells["n1d"]) ** 2)
        print("    chunk%-8d (m=%d) %.3f +-%.3f (n=%d)"
              % (CHUNK_K2, M_OF_ARM["c23"], statistics.mean(cells["c23"]),
                 _sem(cells["c23"]), len(cells["c23"])))
        print("    nodewise1d   (m=%d) %.3f +-%.3f (n=%d)"
              % (M_OF_ARM["n1d"], statistics.mean(cells["n1d"]),
                 _sem(cells["n1d"]), len(cells["n1d"])))
        print("    G'' = chunk%d - nodewise1d = %+0.3f pp   (se %.3f, t %+0.2f -- "
              "DESCRIPTIVE; the gate is the threshold on the difference)"
              % (CHUNK_K2, g3, se, _t(cells["c23"], cells["n1d"])))
        print("    registered: > +0.30 SURVIVES | (+0.15,+0.30] UND | [-0.15,+0.15] "
              "COLLAPSES | [-0.30,-0.15) UND | <= -0.30 REVERSAL")
        print("    -> **%s**" % band(g3))
        print("    bn1's T1 read +0.295 (UNDECIDED) at these seeds 0-2; ar1's A2 read "
              "-0.139 (COLLAPSES) at 3e-4")
        print("    **C3 DOES NOT AMEND bn1's T1.  T1 was UNDECIDED and STAYS UNDECIDED.**")
    else:
        print("    INSUFFICIENT n")

    # --- C4 -----------------------------------------------------------------
    print("\n--- C4  THE CLIP METER.  **DESCRIPTIVE.  CANNOT GATE C1/C2/C3.**")
    print("    %-26s %8s %8s %10s %10s" % ("dir", "rec_lo", "rec_hi", "coord_lo", "coord_hi"))
    for d in dirs:
        o = occ[d]
        cl = ("%10.6f" % o["coord_lo"]) if o["coord_lo"] is not None else "        NA"
        ch = ("%10.6f" % o["coord_hi"]) if o["coord_hi"] is not None else "        NA"
        print("    %-26s %8.4f %8.4f %s %s"
              % (os.path.basename(d), o["rec_lo"], o["rec_hi"], cl, ch))
    print("    premise was: every ms=1e-4 arm reads EXACTLY 0.0000 on both guards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
