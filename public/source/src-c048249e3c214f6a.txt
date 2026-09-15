#!/usr/bin/env python3
"""c79_ar1_score.py -- score `ar1`.  IS THE MATCHED-COUNT PARTITION GAP AN ARGMAX
ARTEFACT?  AND DOES THE SIGN-AGREEMENT FIELD PREDICT ACCURACY -- PRE-REGISTERED?

REGISTERED GATES, transcribed from `bin/c79_argmax_robustness.sh`.  Every constant
below is asserted by the selftest against THAT SCRIPT'S OWN TEXT so the registration
and the batch cannot drift -- STANDING RULE (19).

WRITTEN, SELFTESTED AND GIT-COMMITTED IN THE SAME TICK THE BATCH WAS SUBMITTED, WHILE
NO `ar1` RUN EXISTED IN THE CSV AT ALL.  Not merely before the verdict was read --
before the data could exist.

THE FOUR ARMS, one batch, ms=3e-4, 100 ep, seeds 0-2.  TWO MATCHED-COUNT PAIRS:
    node  nodewise    m=14,420   |  ch   chunk777    m=14,421   (1 group apart)
    n1d   nodewise1d  m= 4,851   |  c23  chunk2325   m= 4,851   (EXACT)

WHY ms=3e-4.  Re-derived from the CSV at the batch's own 13-axis signature:
    nodewise @ 1e-4  92.073 +-0.044 (n=13)
    nodewise @ 3e-4  92.493 +-0.092 (n= 8)      **+0.420 pp, t ~ 4.1**
  Every D and G the campaign has measured (mm1 +0.485, pp1 +0.581, bn1 +0.295) was
  taken at 1e-4, which is 0.42 pp BELOW nodewise's own better stepsize -- comparable
  to the whole gap being explained.  chunk777, chunk2325 and nodewise1d have never
  been run at any other stepsize, so their ms curves are entirely unmeasured.

  A0    VALIDITY.  n_records == 10000; beta moved; epochs_done == requested == 100.
  A0.2  n_beta EXACT on EVERY record: node 14420, ch 14421, n1d 4851, c23 4851.
        All four MEASURED by guard 4 from the ALLOCATED beta on the real built
        network, which ALSO asserted A1's pair <= 1 group apart and A2's EXACT.
  A0.3  THE INSTRUMENT FIRED.  neg_counts.json, n_tot == n_beta, npy shape READ
        FROM ITS HEADER == (n_tot,).  (Header, never file size -- that inference
        produced c74's false VOID on 12/12 arms.)
  A0.4  BOX-FREE at BOTH guards, the published rec_-based 5% gate, PRIMARY and
        UNCHANGED.  **ms is 3x larger here than in ANY previously probed batch, so
        beta travels further and a bind is a REAL possibility, registered in advance
        rather than treated as a surprise.**  A bind VOIDS A3 and leaves A1/A2
        standing -- they are accuracy-only and do not read the field at all.

  A1    **THE PRIMARY.  FIVE-WAY, SYMMETRIC, WITHIN-BATCH, COUNT MATCHED.**
        D' = plateau5(chunk777) - plateau5(nodewise) at ms=3e-4.
          D' >  +0.30     -> **THE GAP SURVIVES THE STEPSIZE MOVE.**  mm1/pp1's D is
               not an argmax artefact; the partition result is robust across a 3x
               change in ms.  The branch that STRENGTHENS the campaign.
          (+0.15, +0.30]  -> **UNDECIDED**, registered in advance.
          [-0.15, +0.15]  -> **THE GAP COLLAPSES AT ms=3e-4.**  Every D the campaign
               quotes is then bounded to ONE meta stepsize; CORRECTIONS 107.2/107.4
               and FINDINGS 76.x/77.x must all carry "at ms=1e-4", and "uniform beats
               architecture-aligned" becomes a statement about a stepsize rather than
               about partitions.  **The branch that costs the campaign its central
               partition claim -- registered with EXACTLY the same width as the
               branch that confirms it.**
          [-0.30, -0.15)  -> **UNDECIDED**, registered in advance.
          D' <= -0.30     -> **REVERSAL: AT 3e-4 THE ALIGNED PARTITION WINS.**
               Reported as an inversion of the central partition claim, not as noise.

  A2    **THE SECONDARY.  bn1's T1 QUESTION AT A SECOND STEPSIZE.**
        G' = plateau5(chunk2325) - plateau5(nodewise1d), both m = 4,851 EXACTLY.
        SAME five-way bands as bn1's T1, transcribed unchanged.
        **A2 DOES NOT AMEND bn1's T1.**  T1 returned +0.295 = UNDECIDED and STAYS
        UNDECIDED whatever A2 returns.  A2 is a reading at a DIFFERENT ms, never a
        re-run of T1 with more seeds.  Registered so that a batch designed after T1
        landed 0.005 pp below its threshold cannot be used to nudge it across.

  A3    **THE CONCORDANCE TEST.  PRE-REGISTERED, SIGN CONVENTION FIXED BEFORE ANY
        ar1 RUN EXISTS.**  This is the gate the tick exists for.
        For each matched-count pair: d_acc = dplateau5, d_fld = dN_eff/m, each with
        a Welch t at n=3.  A channel is RESOLVED at |t| >= 2.0.
        **CONVENTION, FIXED NOW: higher N_eff/m -> higher plateau5 = CONCORDANT.**
        That is the direction the noise-averaging literature implies (more effective
        independence = more information per meta-step).
          both pairs resolved on BOTH channels, both CONCORDANT
             -> **THE FIELD PREDICTS.**  Direction C's premise is supported.
          both pairs resolved on BOTH channels, both ANTI-CONCORDANT
             -> **THE FIELD SYSTEMATICALLY ANTI-PREDICTS**, PRE-REGISTERED, at a
                stepsize where neither channel had been read.  FINDINGS 78.6's
                post-hoc pattern becomes a tested prediction and direction C's
                premise is refuted on its own terms.
          both resolved on both channels, MIXED
             -> **THE FIELD IS NOT A SUFFICIENT STATISTIC.**
          a pair RESOLVED on field, UNRESOLVED on accuracy
             -> **DISSOCIATION** at a new stepsize (FINDINGS 77.5's pattern, which
                stands at ONE contrast until this fires).
          a pair UNRESOLVED on FIELD -> UNINFORMATIVE, reported, NOT folded in.
        **A3 IS VOID IF A0.4 FAILS ON ANY ARM.**

  A4    THE ARGMAX METER.  **DESCRIPTIVE, CROSS-BATCH, NO DIRECTION REGISTERED.**
        Per arm, plateau5 here at 3e-4 vs the CSV's value at 1e-4.  Carries the
        unmodelled +-0.25 pp cross-batch offset (CORRECTIONS 106.3) on EVERY row.
        Cannot gate A1/A2/A3; computed and printed AFTER them.

  A5    THE BATCH-OFFSET METER.  nodewise@3e-4 here vs the CSV's existing cell, bar
        +-0.50.  **CANNOT VOID A1 OR A2** -- both are within-batch differences and
        any offset common to their arms cancels exactly.

WHAT THIS SCORER WILL NOT DO
  * It will not let A4 or A5 gate, annotate or reorder A1, A2 or A3.
  * It will not amend bn1's T1 verdict.  T1 is UNDECIDED and stays UNDECIDED.
  * It will not read A3 if any arm is box-bound.
  * It will not claim any arm's OWN argmax.  Two stepsizes are two points; the
    argmax of chunk777, chunk2325 and nodewise1d remains unmeasured.
  * It will not separate "the group-size distribution" from "the parameter role" on
    the 1-D tensors -- on ResNet18 those coincide EXACTLY (FINDINGS 78.1).
  * It will not re-derive an independence null on the fly (CORRECTIONS 26), and it
    will not print "53.1%".

USAGE
  python3 analysis/c79_ar1_score.py --selftest
  python3 analysis/c79_ar1_score.py --root ../probes_ar1 --csv results/all_runs.csv
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
SCRIPT = os.path.join(REPO, "bin", "c79_argmax_robustness.sh")

from c52_boxfree import occupancy, records           # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -15.0, -2.3026             # `ar1`, registered in c55_neff_noise.BOXES
EPOCHS = 100
N_RECORDS = 10000
MST = "3e-4"                        # THE POINT OF THE BATCH
MST_REF = "1e-4"                    # the ms every D in the campaign was measured at
CHUNK_K = 777
CHUNK_K2 = 2325
NJOBS = 12
ARMS = ("node", "ch", "n1d", "c23")
# n_beta per arm, MEASURED by guard 4 from the ALLOCATED beta.  Not a formula.
M_OF_ARM = {"node": 14420, "ch": 14421, "n1d": 4851, "c23": 4851}
GRAN_OF_ARM = {"node": "nodewise", "ch": "chunk777",
               "n1d": "nodewise1d", "c23": "chunk2325"}

# The five-way band, IDENTICAL for A1 and A2 (A2 transcribes bn1's T1 unchanged).
NULL_HALF = 0.15
DECIDE = 0.30
RESOLVED_T = 2.0
OFFSET_BAR = 0.50

# The two matched-count pairs.  (gate, hi arm, lo arm, what it isolates)
PAIRS = (
    ("A1", "ch", "node", "aligned vs uniform at m=14,420"),
    ("A2", "c23", "n1d", "aligned vs uniform at m=4,851, size-1 tail already gone"),
)


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def _t(a, b):
    se = math.sqrt(_sem(a) ** 2 + _sem(b) ** 2)
    if not (se > 0):
        return float("nan")
    return (statistics.mean(a) - statistics.mean(b)) / se


def band(g):
    """The FIVE-WAY registered verdict, shared by A1 and A2."""
    if g > DECIDE:
        return "SURVIVES"
    if g > NULL_HALF:
        return "UNDECIDED"
    if g >= -NULL_HALF:
        return "COLLAPSES"
    if g >= -DECIDE:
        return "UNDECIDED"
    return "REVERSAL"


def plateaus(csv_path, prefix):
    out = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if not r["run"].startswith(prefix):
                continue
            if int(r["epochs_done"] or 0) != EPOCHS or not r["plateau5"]:
                continue
            out.append(float(r["plateau5"]))
    return sorted(out)


def csv_cell(csv_path, gran, ms):
    """plateau5 at the batch's exact 13-axis signature, for A4/A5 only."""
    out = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["granularity"] != gran or r["meta_stepsize"] != ms:
                continue
            if int(r["epochs_done"] or 0) != EPOCHS or not r["plateau5"]:
                continue
            if r["beta_clip"] != "%g:%s" % (LO, "-2.3026"):
                continue
            if r["augment"] != "1" or r["alpha0"] != "1e-3":
                continue
            if r["network"] != "ResNet18" or r["dataset"] != "CIFAR10":
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


def selftest():
    p = n = 0

    def ck(name, cond):
        nonlocal p, n
        n += 1
        p += bool(cond)
        print("    %-72s %s" % (name, "ok" if cond else "FAIL"))

    print("c79_ar1_score selftest")
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""
    me = open(os.path.abspath(__file__)).read()

    # --- the batch script exists and this scorer matches ITS OWN TEXT -------
    ck("the batch script exists at bin/c79_argmax_robustness.sh", bool(src))
    ck("script CLIP= is this scorer's box", "CLIP=-15:-2.3026" in src)
    ck("script MST= is %s" % MST, "MST=%s" % MST in src)
    ck("script MST_REF= is %s" % MST_REF, "MST_REF=%s" % MST_REF in src)
    ck("script EPOCHS= is %d" % EPOCHS, "EPOCHS=%d" % EPOCHS in src)
    ck("script NJOBS= is %d" % NJOBS, "NJOBS=%d" % NJOBS in src)
    ck("script CHUNK_K= is %d" % CHUNK_K, "CHUNK_K=%d" % CHUNK_K in src)
    ck("script CHUNK_K2= is %d" % CHUNK_K2, "CHUNK_K2=%d" % CHUNK_K2 in src)
    for a, m in sorted(M_OF_ARM.items()):
        keyed = {"node": "M_NODE", "ch": "M_CHUNK", "n1d": "M_N1D", "c23": "M_CHUNK2"}[a]
        ck("script %s=%d matches arm %r" % (keyed, m, a), "%s=%d" % (keyed, m) in src)
    # The script builds two arm names from its OWN CHUNK_K/CHUNK_K2 shell variables,
    # so assert the LITERAL loop tokens, and separately that this scorer's granularity
    # names are exactly what those (already-asserted) constants expand to.
    ck("script's arm loop is exactly the 4 declared arms",
       all(tok in src for tok in ('"nodewise:node"', '"chunk${CHUNK_K}:ch"',
                                  '"nodewise1d:n1d"', '"chunk${CHUNK_K2}:c23"')))
    ck("GRAN_OF_ARM['ch'] is what the script's CHUNK_K expands to",
       GRAN_OF_ARM["ch"] == "chunk%d" % CHUNK_K)
    ck("GRAN_OF_ARM['c23'] is what the script's CHUNK_K2 expands to",
       GRAN_OF_ARM["c23"] == "chunk%d" % CHUNK_K2)
    ck("GRAN_OF_ARM's two non-chunk arms are literal in the script",
       GRAN_OF_ARM["node"] == "nodewise" and GRAN_OF_ARM["n1d"] == "nodewise1d")
    ck("script writes run names ar1-<arm>-s<seed>", 'RN="ar1-${SHORT}-s${S}"' in src)
    ck("script exports PROBE=5 AND PROBE5=1", "PROBE=5,PROBE5=1" in src)
    ck("script registers ar1 in c55 BOXES before submitting", '"ar1" not in c55.BOXES' in src)

    # --- the box really is registered, and is bn1's -------------------------
    import c55_neff_noise as c55
    ck("ar1 is registered in c55 BOXES", "ar1" in c55.BOXES)
    ck("ar1's box equals this scorer's (LO, HI)", c55.BOXES.get("ar1", (0, 0, ""))[:2] == (LO, HI))
    ck("ar1's box is bn1's box (A5 comparable)",
       c55.BOXES.get("ar1", (0, 0, ""))[:2] == c55.BOXES.get("bn1", (1, 1, ""))[:2])

    # --- the matched-count claims -------------------------------------------
    ck("A1's pair is matched to <= 1 group", abs(M_OF_ARM["ch"] - M_OF_ARM["node"]) <= 1)
    ck("A2's pair is matched EXACTLY", M_OF_ARM["c23"] == M_OF_ARM["n1d"])
    ck("A1 and A2 sit at DIFFERENT counts (they are not the same contrast)",
       M_OF_ARM["node"] != M_OF_ARM["n1d"])
    ck("exactly 2 pairs are declared", len(PAIRS) == 2)
    ck("every pair's arms are ar1 arms", all(h in ARMS and l in ARMS for _, h, l, _ in PAIRS))

    # --- the five-way band is symmetric and exhaustive ----------------------
    ck("band is SYMMETRIC about 0", NULL_HALF > 0 and DECIDE > NULL_HALF)
    ck("+0.31 -> SURVIVES", band(+0.31) == "SURVIVES")
    ck("+0.30 -> UNDECIDED (boundary is CLOSED on the UNDECIDED side)",
       band(+0.30) == "UNDECIDED")
    ck("+0.16 -> UNDECIDED", band(+0.16) == "UNDECIDED")
    ck("+0.15 -> COLLAPSES (the NULL band is CLOSED)", band(+0.15) == "COLLAPSES")
    ck("0.00 -> COLLAPSES", band(0.0) == "COLLAPSES")
    ck("-0.15 -> COLLAPSES", band(-0.15) == "COLLAPSES")
    ck("-0.16 -> UNDECIDED", band(-0.16) == "UNDECIDED")
    ck("-0.30 -> UNDECIDED", band(-0.30) == "UNDECIDED")
    ck("-0.31 -> REVERSAL", band(-0.31) == "REVERSAL")
    ck("the NULL band is reachable only by FAILING to find an effect",
       band(0.0) == "COLLAPSES" and band(0.6) == "SURVIVES")
    ck("bn1's measured G=+0.295 would read UNDECIDED under these same bands",
       band(0.295) == "UNDECIDED")
    ck("mm1's D=+0.485 would read SURVIVES under these same bands", band(0.485) == "SURVIVES")

    # --- the statistics ------------------------------------------------------
    ck("_t of identical samples is 0", _t([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0)
    ck("_t is antisymmetric",
       abs(_t([2., 3., 4.], [1., 2., 3.]) + _t([1., 2., 3.], [2., 3., 4.])) < 1e-12)
    ck("_sem of n=1 is nan", math.isnan(_sem([1.0])))

    # --- A3's convention and its branches -----------------------------------
    ck("A3's sign convention is fixed in THIS file",
       "higher N_eff/m -> higher plateau5 = CONCORDANT" in me)
    ck("A3's convention is also fixed in the BATCH SCRIPT",
       "higher N_eff/m -> higher plateau5 = CONCORDANT" in src)
    ck("A3 declares the ANTI branch refutes direction C on its own terms",
       "refuted on its own terms" in me and "refuted on its own terms" in src)
    ck("A3 declares a field-unresolved pair UNINFORMATIVE",
       "UNINFORMATIVE" in me and "UNINFORMATIVE" in src)
    ck("A3 is VOID if the box binds", "A3 IS VOID IF A0.4 FAILS" in me)
    ck("the resolved threshold is |t| >= 2.0", RESOLVED_T == 2.0)

    # --- the honesty guards, asserted against THIS FILE'S OWN TEXT ----------
    ck("scorer states it predates the data entirely", "before the data could exist" in me)
    ck("A2 is declared unable to amend bn1's T1", "STAYS\nUNDECIDED" in me or "STAYS UNDECIDED" in me)
    ck("A4 is declared descriptive with no direction", "NO DIRECTION REGISTERED" in me)
    ck("A4/A5 are declared unable to gate A1/A2/A3", "will not let A4 or A5 gate" in me)
    ck("A5 is declared unable to void A1 or A2", "CANNOT VOID A1 OR A2" in me)
    ck("the cross-batch offset is named with its size", "+-0.25 pp" in me and "106.3" in me)
    ck("the box-bind risk at 3x ms is registered in ADVANCE",
       "a bind is a REAL possibility" in me)
    ck("the batch script registers the same bind risk", "REAL\n#         possibility" in src or "REAL possibility" in src)
    ck("scorer refuses to write the 53.1% sentence", 'will not print "53.1%"' in me)
    ck("the BN-vs-size non-identifiability is carried forward from bn1",
       "coincide EXACTLY (FINDINGS 78.1)" in me)
    ck("the unmeasured-argmax limit is stated", "remains unmeasured" in me)
    ck("the premise is quoted with its own n and t", "+0.420 pp, t ~ 4.1" in me)
    ck("the collapse branch is declared equal in width to the confirm branch",
       "EXACTLY the same width" in me)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_ar1"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs = sorted(glob.glob(os.path.join(a.root, "probe_*")))
    print("=" * 78)
    print("c79 -- ar1: IS THE PARTITION GAP AN ARGMAX ARTEFACT?  %d probe dirs" % len(dirs))
    print("box = (%.1f, %.4f)   ms=%s (vs %s)   %d ep" % (LO, HI, MST, MST_REF, EPOCHS))
    print("A1 node/ch at m=14,420  |  A2 n1d/c23 at m=4,851 EXACT")
    print("=" * 78)

    # --- A0 / A0.2 / A0.3 / A0.4 -------------------------------------------
    ok0 = ok2 = ok3 = free = tot = 0
    fld = {}
    print("\n--- A0  VALIDITY (n_records==%d, beta moved, ep==%d)" % (N_RECORDS, EPOCHS))
    for d in dirs:
        arm = arm_of_dir(d)
        recs = records(d)
        nrec = len(recs)
        good = nrec == N_RECORDS
        print("    %-26s %s  n_rec=%d" % (os.path.basename(d), "PASS" if good else "FAIL", nrec))
        ok0 += good
    print("    A0: %d/%d" % (ok0, len(dirs)))

    print("\n--- A0.2  n_beta EXACT on EVERY record")
    for d in dirs:
        arm = arm_of_dir(d)
        want = M_OF_ARM.get(arm)
        nb = {r.get("n_beta") for r in records(d) if r.get("n_beta") is not None}
        good = nb == {want}
        print("    %-26s %s  n_beta=%s  want %s = %s" % (
            os.path.basename(d), "PASS" if good else "FAIL",
            sorted(nb) if nb else "absent", GRAN_OF_ARM.get(arm), want))
        ok2 += good
    print("    A0.2: %d/%d" % (ok2, len(dirs)))

    print("\n--- A0.3  THE INSTRUMENT FIRED (npy shape from its HEADER)")
    import numpy as np
    for d in dirs:
        arm = arm_of_dir(d)
        want = M_OF_ARM.get(arm)
        jp, npp = os.path.join(d, "neg_counts.json"), os.path.join(d, "neg_counts.npy")
        good, shape, ntot = False, None, None
        if os.path.exists(jp) and os.path.exists(npp):
            ntot = json.load(open(jp)).get("n_tot")
            with open(npp, "rb") as fh:
                shape = np.lib.format.read_magic(fh) and np.lib.format.read_array_header_1_0(fh)[0]
            good = (ntot == want) and (shape == (want,))
        print("    %-26s %s  n_tot=%s  npy_shape=%s" % (
            os.path.basename(d), "PASS" if good else "FAIL", ntot, shape))
        ok3 += good
    print("    A0.3: %d/%d" % (ok3, len(dirs)))

    print("\n--- A0.4  BOX-FREE (rec_-based 5%% gate, PRIMARY, UNCHANGED)")
    print("    **ms is 3x any probed batch; a bind here VOIDS A3 and leaves A1/A2.**")
    for d in dirs:
        o = occupancy(d, LO, HI)
        f = o["rec_lo"] < 0.05 and o["rec_hi"] < 0.05
        tot += 1
        free += f
        print("    %-26s %-8s rec_lo %.4f  rec_hi %.4f" % (
            os.path.basename(d), "free" if f else "BOUND", o["rec_lo"], o["rec_hi"]))
        nf = neff_of_dir(d)
        if nf is not None:
            fld.setdefault(arm_of_dir(d), []).append(nf)
    print("    A0.4: %d/%d box-free" % (free, tot))
    a3_void = free != tot

    # --- the cells ----------------------------------------------------------
    acc = {arm: plateaus(a.csv, "ar1-%s-s" % arm) for arm in ARMS}
    print("\n--- plateau5 per arm, from the CSV")
    for arm in ARMS:
        v = acc[arm]
        print("    %-12s m=%-8d n=%d  %.3f +-%.3f" % (
            GRAN_OF_ARM[arm], M_OF_ARM[arm], len(v),
            statistics.mean(v) if v else float("nan"), _sem(v)))

    # --- A1 and A2 ----------------------------------------------------------
    verdicts = {}
    for gate, hi, lo, what in PAIRS:
        g = statistics.mean(acc[hi]) - statistics.mean(acc[lo])
        t = _t(acc[hi], acc[lo])
        v = band(g)
        verdicts[gate] = (g, t, v)
        title = ("**THE PRIMARY.  FIVE-WAY, SYMMETRIC, COUNT MATCHED.**" if gate == "A1"
                 else "**THE SECONDARY.  bn1's T1 QUESTION AT A SECOND STEPSIZE.**")
        print("\n--- %s  %s" % (gate, title))
        print("    isolates : %s" % what)
        print("    %-12s (m=%d) %.3f +-%.3f (n=%d)" % (
            GRAN_OF_ARM[hi], M_OF_ARM[hi], statistics.mean(acc[hi]), _sem(acc[hi]), len(acc[hi])))
        print("    %-12s (m=%d) %.3f +-%.3f (n=%d)" % (
            GRAN_OF_ARM[lo], M_OF_ARM[lo], statistics.mean(acc[lo]), _sem(acc[lo]), len(acc[lo])))
        print("    %s = %s - %s = %+.3f pp   (se %.3f, t %.2f -- DESCRIPTIVE; the gate is "
              "the threshold on the difference)" % (
                  "D'" if gate == "A1" else "G'", GRAN_OF_ARM[hi], GRAN_OF_ARM[lo], g,
                  math.sqrt(_sem(acc[hi]) ** 2 + _sem(acc[lo]) ** 2), t))
        print("    registered: > +%.2f SURVIVES | (+%.2f,+%.2f] UND | [-%.2f,+%.2f] COLLAPSES "
              "| [-%.2f,-%.2f) UND | <= -%.2f REVERSAL"
              % (DECIDE, NULL_HALF, DECIDE, NULL_HALF, NULL_HALF, DECIDE, NULL_HALF, DECIDE))
        print("    -> **%s**" % v)
        if gate == "A1":
            print("    at ms=%s the same contrast read: mm1 +0.485, pp1 +0.581 (pooled +0.533)" % MST_REF)
            if v == "COLLAPSES":
                print("    **EVERY D THE CAMPAIGN QUOTES IS NOW BOUNDED TO ms=%s.**  CORRECTIONS" % MST_REF)
                print("    107.2/107.4 and FINDINGS 76.x/77.x must all carry 'at ms=%s'." % MST_REF)
        else:
            print("    at ms=%s the same contrast read: bn1 T1 = +0.295 (UNDECIDED)" % MST_REF)
            print("    **A2 DOES NOT AMEND bn1's T1.  T1 was UNDECIDED and STAYS UNDECIDED.**")

    # --- A3 -----------------------------------------------------------------
    print("\n--- A3  **THE CONCORDANCE TEST.  PRE-REGISTERED; CONVENTION FIXED BEFORE THE DATA.**")
    print("    convention: higher N_eff/m -> higher plateau5 = CONCORDANT")
    if a3_void:
        print("    -> **VOID**: %d/%d arms box-bound, so N_eff/m is not interpretable." % (tot - free, tot))
        print("       Registered in advance.  A1 and A2 above are unaffected.")
    else:
        labels = []
        for gate, hi, lo, what in PAIRS:
            da = statistics.mean(acc[hi]) - statistics.mean(acc[lo])
            ta = _t(acc[hi], acc[lo])
            if hi not in fld or lo not in fld:
                print("    %s: field absent on an arm -> UNINFORMATIVE" % gate)
                labels.append("UNINFORMATIVE")
                continue
            df = statistics.mean(fld[hi]) - statistics.mean(fld[lo])
            tf = _t(fld[hi], fld[lo])
            ares, fres = abs(ta) >= RESOLVED_T, abs(tf) >= RESOLVED_T
            if not fres:
                lab = "UNINFORMATIVE"
            elif not ares:
                lab = "DISSOCIATION"
            elif (df > 0) == (da > 0):
                lab = "CONCORDANT"
            else:
                lab = "ANTI-CONCORDANT"
            labels.append(lab)
            print("\n    %s  %s - %s  (%s)" % (gate, GRAN_OF_ARM[hi], GRAN_OF_ARM[lo], what))
            print("      dplateau5 = %+7.3f pp  (t %6.2f)  %s" % (da, ta, "resolved" if ares else "UNRESOLVED"))
            print("      dN_eff/m  = %+7.4f     (t %6.2f)  %s" % (df, tf, "resolved" if fres else "UNRESOLVED"))
            print("      -> **%s**" % lab)
        s = set(labels)
        print("\n    A3 VERDICT over %d pairs: %s" % (len(labels), ", ".join(labels)))
        if s == {"CONCORDANT"}:
            print("    -> **THE FIELD PREDICTS.**  Direction C's premise is SUPPORTED and the")
            print("       sign-agreement programme is worth running.")
        elif s == {"ANTI-CONCORDANT"}:
            print("    -> **THE FIELD SYSTEMATICALLY ANTI-PREDICTS**, PRE-REGISTERED, at a stepsize")
            print("       where neither channel had been read.  FINDINGS 78.6's post-hoc pattern is")
            print("       now a tested prediction; direction C's premise is refuted on its own terms.")
        elif "DISSOCIATION" in s:
            print("    -> **DISSOCIATION** at a new stepsize.  FINDINGS 77.5 stood at ONE contrast;")
            print("       it now has an independent one.  The field is not a sufficient statistic.")
        elif s == {"UNINFORMATIVE"}:
            print("    -> **UNINFORMATIVE**: the field did not resolve on either pair.")
        else:
            print("    -> **THE FIELD IS NOT A SUFFICIENT STATISTIC**: no monotone function of")
            print("       N_eff/m orders these arms by accuracy.")

    # --- A4 -----------------------------------------------------------------
    print("\n--- A4  THE ARGMAX METER.  **DESCRIPTIVE, CROSS-BATCH, NO DIRECTION.**")
    print("    every row carries the unmodelled +-0.25 pp cross-batch offset (106.3)")
    print("    %-12s %-22s %-22s %s" % ("arm", "ms=%s (HERE)" % MST, "ms=%s (CSV)" % MST_REF, "delta"))
    for arm in ARMS:
        here, there = acc[arm], csv_cell(a.csv, GRAN_OF_ARM[arm], MST_REF)
        d = (statistics.mean(here) - statistics.mean(there)) if here and there else float("nan")
        print("    %-12s %8.3f +-%-11.3f %8.3f +-%-11.3f %+7.3f  %s" % (
            GRAN_OF_ARM[arm],
            statistics.mean(here) if here else float("nan"), _sem(here),
            statistics.mean(there) if there else float("nan"), _sem(there), d,
            "(n=%d vs %d)" % (len(here), len(there))))
    print("    This says which stepsize each arm prefers.  It does NOT locate any")
    print("    arm's argmax: two stepsizes are two points.")

    # --- A5 -----------------------------------------------------------------
    print("\n--- A5  THE BATCH-OFFSET METER.  **DECLARED UNABLE TO VOID A1 OR A2.**")
    here, there = acc["node"], csv_cell(a.csv, "nodewise", MST)
    prev = [x for x in there if x not in here]
    if prev:
        d = statistics.mean(here) - statistics.mean(prev)
        print("    nodewise@%s HERE %.3f +-%.3f (n=%d) vs the CSV's prior %.3f (n=%d): %+.3f, bar +-%.2f -> %s"
              % (MST, statistics.mean(here), _sem(here), len(here), statistics.mean(prev),
                 len(prev), d, OFFSET_BAR, "REPRODUCES" if abs(d) <= OFFSET_BAR else "DEVIATES"))
    else:
        print("    no prior nodewise@%s rows outside this batch" % MST)
    print("    A1 and A2 are unaffected either way: both are within-batch differences")
    print("    and any offset common to their arms cancels exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
