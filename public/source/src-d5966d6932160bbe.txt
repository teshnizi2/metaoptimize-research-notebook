#!/usr/bin/env python3
"""c82_fa1_score.py -- score `fa1`.  THE ms/BOX DECONFOUND: the partition contrasts
at ms=3e-4 under a FLOOR THAT CANNOT BIND and a CEILING HELD FIXED AT ar1's.

REGISTERED GATES, transcribed from `bin/c82_field_wideclip.sh`.  Every constant
below is asserted by the selftest against THAT SCRIPT'S OWN TEXT so the
registration and the batch cannot drift -- STANDING RULE (19).

WRITTEN, SELFTESTED AND GIT-COMMITTED BEFORE ANY `fa1` JOB WAS SUBMITTED, WHILE NO
`fa1` RUN EXISTED IN THE CSV AT ALL.  Not merely before the verdict was read --
before the data could exist.

REVISION NOTE -- WHAT CHANGED FROM THE FIRST REGISTRATION (commit c72d3f7), AND
WHY.  All of it is a change to gates that had not yet seen data.
  (1) **THE BOX IS -25:-2.3026, NOT -25:+9.0.**  The first version budgeted BOTH
      guards from the Lion identity and argued that margin above an unreachable
      bound is free.  That is right on the FLOOR and wrong on the CEILING, for
      three reasons, each re-derived from disk here:
      (a) Its headline HI evidence came from two DIVERGED runs.  `bd7-w-c6-s0/s1`
          (BETA_CLIP=-30:6.0) read collapsed=1, plateau5=10.000, best_test
          88.82/88.94, and in `probes_bd7` their beta_true_min AND beta_true_max
          are bit-identical for 4,042 and 3,303 consecutive records.  The
          "free maximum +3.436" was a dead network's frozen odometer.  The largest
          HEALTHY free HI excursion at a coarse granularity is +2.096
          (probe_node_c6_s1, collapsed=0, plateau5 92.176).
      (b) The ceiling is a STABILITY DEVICE, not an instrument.  bd7-w-c2 (HI=+2.0)
          clamped hard (rec_hi up to 0.6209) and SURVIVED 2/2; bd7-w-c6 (HI=+6.0)
          never bound and DIED 2/2.  Of 14 collapses in 1,867 CSV rows, 10 are
          weightwise UNBOXED at ms=1e-3, 2 unattributed, and exactly 2 boxed -- the
          released-ceiling pair.  ZERO coarse-granularity collapses under any box.
      (c) Only the FLOOR is confounded with the stepsize.  In `probes_ar1` the LO
          guard binds 12/12 (rec_lo 0.4521-0.4597); the HI guard binds 1/12 (node
          s2, rec_hi 0.1196, at most 3 of 14,420 coordinates).
      Direct validation of the fix: re-running occupancy() over ar1's own probes
      with the floor moved to -25 gives rec_lo = 0.0000 on 12/12, rec_hi unchanged.
  (2) **F3 IS UNPAIRED, AND THE PAIRED-VARIANCE CLAIM IS WITHDRAWN.**  Measured
      here over 356 same-config SAME-SEED replicate pairs (ResNet18/CIFAR10/100 ep/
      not collapsed/plateau5>80): median |difference| 0.1710 pp, per-run sd 0.179.
      The across-seed sd within a config (156 configs) has median 0.159 pp.  Those
      are the same number: **seed explains essentially none of the run-to-run
      variance in plateau5 on this cluster** (PARTS spans five GPU classes; ar1
      alone scattered over five nodes with wallclocks 44-114 min).  Unpaired 6 v 3
      has se 0.127 pp and resolves 0.25 pp.
  (3) **n GOES 3 -> 6** (24 jobs, two waves of 12 for the FairShare budget rule).
  (4) **F3 IS THE PRIMARY, F2 SECONDARY, F1 DESCRIPTIVE.**  Every F1 branch ends in
      "direction C stays DROPPED", so F1 cannot change a conclusion and may not
      hold the headline.  F3 is the test that retires the STANDING RULE 10 breach.
  (5) **F0.1, A HEALTH GATE, IS ADDED.**  The first version had none: `collapsed`
      appeared nowhere in either file, and a diverged run that still completes its
      epochs would have passed F0 and dragged D_w ~80 pp into INVERTS.
  (6) **F2's BAND ACQUIRES A SIGNIFICANCE CLAUSE** (see F2).

THE FOUR ARMS, ms=3e-4, 100 ep, seeds 0-5.  TWO MATCHED-COUNT PAIRS:
    node  nodewise    m=14,420   |  ch   chunk777    m=14,421   (1 group apart)
    n1d   nodewise1d  m= 4,851   |  c23  chunk2325   m= 4,851   (EXACT)
ONE FIELD CHANGES FROM `ar1`: BETA_CLIP -15:-2.3026 -> **-25:-2.3026**, and within
that field ONLY THE FLOOR MOVES.

WHY, AND THE DEFLATION FIRST.  `ar1` VOIDED A3 with 12/12 arms on the LOW guard
(rec_lo 0.4521-0.4597, first pin at step 27,020-27,395 of 50,000).  **But the
scientific question A3 asked is already CLOSED**: `cc1` ran the same test at
ms=1e-4, 12/12 box-free, and C1 returned MIXED -- ANTI-CONCORDANT on chunk777 -
nodewise (d_acc +0.727 t +3.63, d_N_eff/m -0.0237 t -11.14) and DISSOCIATION on
chunk2325 - nodewise1d (d_acc +0.011 t +0.08, d_N_eff/m -0.0529 t -23.26).
**DIRECTION C IS DROPPED (CORRECTIONS 110.2, 110.5(1)) AND NOTHING BELOW REOPENS
IT.**  What fa1 uniquely buys is narrower: every 100-ep ResNet18 probe at ms=1e-4
in the old box is box-FREE (39/39) and every one at ms=3e-4 is LO-BOUND (15/15), so
the campaign's stepsize axis is perfectly confounded with its floor-binding axis --
a STANDING RULE 10 violation sitting under a headline.  fa1 changes one field and
retires it.

**HOW BIG IS THE EFFECT F3 LOOKS FOR?  SMALL, AND THAT IS REGISTERED IN ADVANCE.**
Releasing the floor from -15 to -25 moves a pinned coordinate's step size from
exp(-15) = 3.06e-7 to at worst exp(-21.908) = 3.06e-10 -- both inert against base
weights of order 1e-1 -- and a coordinate pinned at -15 by step 27,020 cannot climb
back to even -8 in the 22,980 steps left (it needs 23,333).  So |Delta_arm| is
EXPECTED below 0.10 pp, i.e. BELOW what n=6 resolves.  **An unresolved F3 is the
EXPECTED outcome and is nearly uninformative on its own.  The deliverable is the
box-free ms=3e-4 CELL, not F3's t-statistic.**  Said here so nobody prices this
batch as though its primary were powered.

**CAN fa1 RE-READ ar1's A1/A2?  NO.  THIS BATCH IS INSTRUMENT-ONLY.**  The guard
bound DIFFERENTIALLY across exactly the arms A1 compares: at the final record
nodewise has 11.25-11.74% of its 14,420 coordinates on the floor against chunk777's
0.159-0.173% of 14,421, a 68-71x difference (85x over all records); the clamp was
live for the final 45.2-46.0% of training, which is the half plateau5 is read from;
and in n1d/ch/c23 15-17 of the 41 one-dimensional tensors had their single group
pinned -- the very size-1 tail A1-A2 exists to isolate.  Therefore **F2's D_w may
NOT be appended to the mm1/pp1/ar1/cc1 D series as a fifth replication**, and every
fa1 number is quoted with its box.  Whether freeing the floor helps or hurts
nodewise is **UNSURE** -- that is why F3 is mandatory, not optional.

  F0    VALIDITY.  n_records == 10000; beta moved; epochs_done == requested == 100.
  F0.1  **HEALTH.  THE GATE THE FIRST VERSION DID NOT HAVE.**  Three checks, all of
        which `bd7-w-c6-s0` would fail and F0 would not:
          (a) CSV `collapsed` is 0 or empty;
          (b) best_test - plateau5 <= 2.0 pp -- the LATE-CRASH tripwire (healthy ar1
              arms run 0.27-0.42 pp; bd7-w-c6-s0 reads 78.8);
          (c) a BETA-FREEZE detector on probe.jsonl: VOID any arm whose
              beta_true_min AND beta_true_max are both bit-identical, **and both
              strictly INSIDE the box**, across >= 1,000 consecutive records
              (bd7-w-c6-s0/s1 read 4,042 and 3,303).
        A failing arm is VOIDED and excluded from F1, F2 and F3 -- never banded.
        **THE INTERIORITY CLAUSE IN (c) WAS ADDED AFTER SMOKE-TESTING THE DETECTOR
        ON REAL PROBES AND BEFORE ANY fa1 DATA EXISTED.**  The naive form is not a
        divergence test but a CLAMP test: ar1-node-s2 sat on BOTH guards at once
        (rec_lo 0.4521, rec_hi 0.1196), so both extremes were constant by
        construction and the naive detector reported a 1,194-record "freeze" on a
        healthy run.  With interiority: ar1-node-s2 -> 0, bd7-w-c6-s0/s1 -> 4,042 /
        3,303, bd7-node-c6-s1 -> 0.  Frozen ON the guards is clamped; frozen INSIDE
        the box is dead.
  F0.2  n_beta EXACT on EVERY record: node 14420, ch 14421, n1d 4851, c23 4851.
        All four MEASURED by the batch script's guard 4 from the ALLOCATED beta on
        the real built network, which ALSO asserted D_w's pair <= 1 group apart and
        G_w's EXACT.
  F0.3  THE INSTRUMENT FIRED.  neg_counts.json, n_tot == n_beta, npy shape READ FROM
        ITS HEADER == (n_tot,).  (Header, never file size -- that inference produced
        c74's false VOID on 12/12 arms.)
  F0.4  **THE BOX-OCCUPANCY GATE, SCORED PER SEED, SPLIT BY GUARD** because the two
        guards now have DIFFERENT logical status.
    F0.4a **LO -- ALGEBRAIC.**  rec_lo == 0.0000 EXACTLY on every arm.  HF.py's Lion
          meta update is beta <- (1 - ms*wd_meta)*beta - ms*sign(.) and every fa1 job
          carries --weight-decay-meta 0, so each update moves each coordinate by
          exactly 0 or +-ms and beta is confined to [ln(alpha0) - ms*T,
          ln(alpha0) + ms*T] = [-21.907755, +8.092245] at alpha0=1e-3, ms=3e-4,
          T = 100 ep x 500 = 50,000.  A floor of -25 lies strictly below that, so a
          record at the floor is **ALGEBRAICALLY IMPOSSIBLE and means the config is
          not what the header says** (wd_meta != 0, wrong ms, wrong step count, a
          resumed run, or HIER != none).  If F0.4a fires the batch is **VOID -- do
          not rescore, debug the config.**  Checked with the documented one-update
          offset: HF.py clamps (PATCH_CLIP) before it probes (PATCH_PROBE), so the
          record labelled step 0 has already taken one update, and on ar1's 120,000
          records the worst slack is exactly -0.000300 = one ms step, zero
          violations beyond it.
    F0.4b **HI -- EMPIRICAL.**  The ceiling is DELIBERATELY UNCHANGED from ar1 and IS
          reachable (+8.092245).  ar1 bound there on 1/12 (node s2, rec_hi 0.1196),
          so a bind on the nodewise arm is EXPECTED and is not a defect of this
          batch.  Registered: an arm with rec_hi >= 5% is UNINTERPRETABLE on the
          FIELD (rule 5) and its PAIR is dropped from F1; if both pairs drop, F1 is
          VOID.  F2 and F3 are accuracy-only, stand, and carry the occupancy beside
          every number.  Because ar1 carried the SAME ceiling, an HI bind is a
          condition SHARED by both halves of F3 rather than a difference between
          them.  Occupancy is printed for every arm whatever the outcome.

  F3    **THE PRIMARY.  THE BOX EFFECT, PER ARM -- THE STANDING-RULE-10 REPAIR.**
        Delta_arm = mean plateau5(fa1 arm, n=6) - mean plateau5(ar1 arm, n=3),
        **UNPAIRED Welch t, RESOLVED at |t| >= 2.0.**  se ~0.127 pp -> resolves
        0.25 pp.
          resolved on ANY arm -> **THE BOX CHANGED THE OPTIMISER, NOT MERELY THE
             INSTRUMENT.**  D_w may not be pooled with ar1's D under any
             circumstances and every sentence about D carries its box thereafter.
          no arm resolved -> **UNRESOLVED at 0.25 pp -- the EXPECTED outcome.**
             **This may NOT be written as "the box is accuracy-neutral."**  Declared
             now, because the campaign has previously written exactly that sentence
             off an underpowered null.
        What F3 delivers regardless of its t: a box-free ms=3e-4 cell, which does
        not exist anywhere in the corpus today.

  F2    SECONDARY.  THE ACCURACY CONTRASTS **IN THIS BOX, AND ONLY IN THIS BOX.**
          D_w = plateau5(chunk777)  - plateau5(nodewise)     m 14,421 vs 14,420
          G_w = plateau5(chunk2325) - plateau5(nodewise1d)   m 4,851 EXACT
        FOUR BANDS, FIXED NOW, IDENTICAL FOR D_w AND G_w, calibrated against the
        three BOX-FREE readings of D the campaign owns (mm1 +0.485, pp1 +0.581,
        cc1 +0.727) so that SURVIVES means "indistinguishable from the box-free
        evidence":
          >= +0.45       -> SURVIVES     (not a guard artefact)
          [+0.15, +0.45) -> ATTENUATED   (box-dependent; ar1's +0.697 acquires an
                                          "at BETA_CLIP=-15:-2.3026" qualifier,
                                          permanently)
          (-0.15, +0.15) -> COLLAPSES    (the ms=3e-4 cell is a guard artefact and is
                                          dropped from the D series)
          <= -0.15       -> INVERTS      (the strongest available refutation; never
                                          softened to "collapses")
        The negative side is deliberately NOT subdivided.
        **THE SECOND CLAUSE, ADDED IN THIS REVISION AND FIXED HERE.**  The band is a
        threshold on a POINT ESTIMATE and the ATTENUATED verdict carries a PERMANENT
        registered cost.  Inverse-variance pooling the three box-free anchors gives
        **+0.5805 +- 0.0939** (Q = 0.88 on 2 df -- homogeneous), the SURVIVES line at
        +0.45 sits only 0.13 pp below that pool, and se(D_w - pool) at n=6 is 0.140,
        so a point estimate alone would mis-band a truly box-free batch at a rate of
        order 10-25%.  **REGISTERED: ATTENUATED, COLLAPSES or INVERTS may be
        DECLARED only if D_w is ALSO below the pooled box-free anchor with |t| >= 2
        against it.  Otherwise the reported verdict is CONSISTENT-WITH-BOX-FREE
        (UNDERPOWERED) and the band label is printed only as an unconfirmed point
        reading beside it.**  SURVIVES needs no second clause -- it asserts nothing
        new.  The same two-clause rule applies to G_w, against ITS registered
        expectation rather than the D pool: ar1 read -0.139 and cc1 read +0.011, so
        G_w is EXPECTED to COLLAPSE; a G_w outside the null band with |t| >= 2 is the
        informative outcome and would UNDERCUT the tail interpretation of A1-A2.

  F1    **DESCRIPTIVE.  DEMOTED FROM PRIMARY IN THIS REVISION.**  The concordance
        reading at ms=3e-4 under a floor that cannot bind.  Every branch it
        enumerates ends in "direction C stays DROPPED", so it cannot change a
        conclusion.  Convention and bands are nonetheless FIXED HERE.
        For each matched-count pair: d_acc = dplateau5, d_fld = dN_eff/m, each with a
        Welch t.  A channel is RESOLVED at |t| >= 2.0.
        **CONVENTION, TRANSCRIBED UNCHANGED FROM `analysis/c81_cc1_score.py`'s C1
        (itself unchanged from `c79_ar1_score.py`'s A3): higher N_eff/m -> higher
        plateau5 = CONCORDANT.**  That is the direction the noise-averaging
        literature implies (more effective independence = more information per
        meta-step).
          both pairs resolved on BOTH channels, both ANTI-CONCORDANT
             -> REPLICATES cc1's ANTI-CONCORDANT leg at a second stepsize and a
                non-binding floor.  Direction C stays DROPPED.
          MIXED (any combination of CONCORDANT / ANTI-CONCORDANT / DISSOCIATION)
             -> REPLICATES cc1's own MIXED verdict.  Direction C stays DROPPED.
          both pairs resolved on BOTH channels, both CONCORDANT
             -> **DISCREPANCY WITH cc1, LOGGED FOR THE RECORD.**  It does NOT reopen
                direction C: it would say the field's sign is stepsize- or
                box-dependent, which makes it LESS of a design variable, not more.
          a pair RESOLVED on field, UNRESOLVED on accuracy -> DISSOCIATION.
          a pair UNRESOLVED on FIELD -> UNINFORMATIVE, reported, NOT folded in.
        **A PAIR IS DROPPED IF EITHER ARM BINDS OR FAILS F0.1.**  If both pairs
        drop, F1 is VOID.  A void here is not another owed re-run -- the question is
        closed; a second void would only say this box binds too.

  F4    THE CLIP METER.  DESCRIPTIVE.  rec_lo / rec_hi / coord_lo / coord_hi and
        beta_true_min/max per arm, printed AFTER F1-F3.  Its only job is to document
        how much headroom the floor bound left and how hard the unchanged ceiling
        was pressed.  It cannot gate F1, F2 or F3 beyond the F0.4 void it feeds.

THE STOPPING RULE
  24 jobs, submitted in two waves of 12 only because the FairShare rule budgets a
  batch at ~20 jobs of ~40 min.  **THE REGISTERED n IS 6.  F2's band verdict and
  F3's resolution verdict MAY BE READ ONCE, AT n=6.**  With fewer than 6 usable
  seeds per arm this scorer prints INTERIM-UNDERPOWERED, reports occupancy and cell
  means, and REFUSES to emit a band label.  Stopping after wave a because it "looks
  clear" is a stopping-rule violation, forbidden here in advance.

WHAT THIS SCORER WILL NOT DO
  * It will not reopen direction C under ANY F1 outcome.  cc1's five-way
    pre-registered test decided it and the question is CLOSED.
  * It will not pool D_w with ar1's D, and it will not print D_w as a fifth
    replication of the D series.
  * It will not emit a band label below the registered n.
  * It will not declare ATTENUATED / COLLAPSES / INVERTS on a point estimate alone.
  * It will not let F4 gate, annotate or reorder F1, F2 or F3.
  * It will not read F1 on a pair whose arm is box-bound or unhealthy.
  * It will not write "the box is accuracy-neutral" off an unresolved F3.
  * It will not quote a beta statistic from a run without joining it against
    `collapsed` first -- that is how the withdrawn +3.436 ceiling argument happened.
  * It will not claim any arm's argmax.  Two stepsizes are two points.
  * It will not say anything about a RELEASED CEILING; this batch holds one fixed.
  * It will not separate "the group-size distribution" from "the parameter role" on
    the 1-D tensors -- on ResNet18 those coincide EXACTLY (FINDINGS 78.1).
  * It will not re-derive an independence null on the fly (CORRECTIONS 26), and it
    will not print "53.1%".

USAGE
  python3 analysis/c82_fa1_score.py --selftest
  python3 analysis/c82_fa1_score.py --root ../probes_fa1 --csv results/all_runs.csv
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
SCRIPT = os.path.join(REPO, "bin", "c82_field_wideclip.sh")

from c52_boxfree import occupancy, records           # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -25.0, -2.3026             # `fa1`, registered in c55_neff_noise.BOXES.
                                    # ONLY THE FLOOR MOVES; HI == ar1's ceiling.
CLIP_REF = "-15:-2.3026"            # ar1's box: LO-bound 12/12, HI-bound 1/12
EPOCHS = 100
N_RECORDS = 10000
MST = "3e-4"                        # ar1's stepsize, UNCHANGED -- the box is the axis
ALPHA0 = 1e-3
BATCH = 100
STEPS_PER_EPOCH = 500               # MEASURED by the script's guard 4d, not assumed
CHUNK_K = 777
CHUNK_K2 = 2325
NJOBS = 12                          # PER WAVE
NJOBS_TOTAL = 24                    # 4 arms x 6 seeds
SEEDS = ("0", "1", "2", "3", "4", "5")
N_SEEDS_REGISTERED = 6              # THE STOPPING RULE: a band verdict at n=6, once
REF_FAM = "ar1"
ARMS = ("node", "ch", "n1d", "c23")
# n_beta per arm, MEASURED from the ALLOCATED beta by the script's guard 4.
M_OF_ARM = {"node": 14420, "ch": 14421, "n1d": 4851, "c23": 4851}
GRAN_OF_ARM = {"node": "nodewise", "ch": "chunk777",
               "n1d": "nodewise1d", "c23": "chunk2325"}

# The Lion identity the box was BUDGETED from (F0.5).  Not a fit, not a trace.
T_UPDATES = EPOCHS * STEPS_PER_EPOCH             # 50,000
MS = float(MST)
BETA0 = math.log(ALPHA0)                         # -6.907755
SPAN = MS * T_UPDATES                            # 15.0
REACH_LO, REACH_HI = BETA0 - SPAN, BETA0 + SPAN  # -21.907755, +8.092245

# The FOUR-WAY accuracy band, IDENTICAL for D_w and G_w.
SURVIVES_AT = 0.45          # the lowest BOX-FREE reading of D is mm1's +0.485
NULL_HALF = 0.15            # the campaign's null half-width, unchanged
RESOLVED_T = 2.0            # a channel / a Delta is RESOLVED at |t| >= 2.0
BOXFREE_MAX = 0.05          # the published rec_-based 5% gate, unchanged

# --- F0.1, THE HEALTH GATE.  Registered thresholds. -------------------------
LATE_CRASH_PP = 2.0         # best_test - plateau5 above this = a late crash.
                            # healthy ar1 arms run 0.27-0.42; bd7-w-c6-s0 reads 78.8
FREEZE_RECORDS = 1000       # beta bit-identical on BOTH sides for this many
                            # consecutive records = a dead network.
                            # bd7-w-c6-s0/s1 read 4,042 and 3,303.

# The three BOX-FREE anchors F2's bands are calibrated against.  POST-HOC and
# DESCRIPTIVE: used to place a threshold, never quoted as a prediction.
D_BOXFREE_ANCHORS = {"mm1": 0.485, "pp1": 0.581, "cc1": 0.727}
# Inverse-variance pool of the three, from their own Welch se (0.1613/0.1414/
# 0.2001), re-derived from the CSV: +0.5805 +- 0.0939, Q = 0.88 on 2 df.
# THE SECOND CLAUSE of F2's band tests against THIS, not against the threshold.
D_POOL, D_POOL_SE = 0.5805, 0.0939
D_AR1_BOUND = 0.697         # ar1's own D, measured with 12/12 arms box-bound
G_AR1_BOUND = -0.139
G_CC1_FREE = 0.011
# G_w's registered expectation is the null itself, so its second clause tests
# against 0.0 with the campaign's null half-width as the effect of interest.
G_EXPECT, G_EXPECT_SE = 0.0, 0.0

# Measured run-to-run scale, re-derived from the CSV this tick over 356
# same-config SAME-SEED replicate pairs (ResNet18/CIFAR10/100 ep/not collapsed/
# plateau5>80): median |difference| 0.1710 pp -> per-run sd 0.179 pp.  The
# across-seed sd within a config (156 configs) has median 0.159 pp.  SAME NUMBER
# -> seed carries no reproducibility here -> F3 IS UNPAIRED.
PER_RUN_SD = 0.179
SEED_IS_REPRODUCIBLE = False

# The two matched-count pairs.  (gate, hi arm, lo arm, what it isolates)
PAIRS = (
    ("D_w", "ch", "node", "aligned vs uniform at m=14,420"),
    ("G_w", "c23", "n1d", "aligned vs uniform at m=4,851, size-1 tail already gone"),
)


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def _t(a, b):
    se = math.sqrt(_sem(a) ** 2 + _sem(b) ** 2)
    if not (se > 0):
        return float("nan")
    return (statistics.mean(a) - statistics.mean(b)) / se


def _t_paired(d):
    """One-sample t on the paired differences -- F3's statistic."""
    if len(d) < 2:
        return float("nan")
    s = _sem(d)
    if not (s > 0):
        return float("nan")
    return statistics.mean(d) / s


def band(g):
    """The FOUR-WAY registered POINT-ESTIMATE band, shared by D_w and G_w.

    This is HALF of F2's verdict.  It selects a label; `verdict()` decides
    whether that label may be DECLARED.  Registered, unchanged from the first
    registration: the thresholds themselves did not move.
    """
    if g >= SURVIVES_AT:
        return "SURVIVES"
    if g >= NULL_HALF:
        return "ATTENUATED"
    if g > -NULL_HALF:
        return "COLLAPSES"
    return "INVERTS"


def verdict(g, se, ref, ref_se):
    """F2's TWO-CLAUSE registered verdict.  Returns (declared, label, t_vs_ref).

    Clause 1: the point estimate selects a band (`band`).
    Clause 2: ATTENUATED / COLLAPSES / INVERTS -- the three labels that impose a
    PERMANENT cost on how the campaign may quote ar1's +0.697 -- may be DECLARED
    only if `g` is ALSO below the reference (the pooled box-free anchor for D_w)
    with |t| >= RESOLVED_T against it.  Otherwise the declared verdict is
    CONSISTENT-WITH-BOX-FREE (UNDERPOWERED) and the band label is reported only as
    an unconfirmed point reading.

    SURVIVES needs no second clause: it asserts nothing new.

    WHY.  se(D_w - pool) at n=6 is sqrt(0.104^2 + 0.0939^2) = 0.140 pp while the
    SURVIVES line sits only 0.13 pp below the pool, so a threshold on the point
    estimate alone mis-bands a genuinely box-free batch at a rate of order 10-25%.
    """
    lab = band(g)
    if lab == "SURVIVES":
        return lab, lab, float("nan")
    sed = math.sqrt(se ** 2 + ref_se ** 2)
    t = (g - ref) / sed if sed > 0 else float("nan")
    if (not math.isnan(t)) and t <= -RESOLVED_T:
        return lab, lab, t
    return "CONSISTENT-WITH-BOX-FREE (UNDERPOWERED)", lab, t


def freeze_run(dirpath, lo=LO, hi=HI, tol=1e-6):
    """Longest run of consecutive records with beta_true_min AND beta_true_max
    both bit-identical **AND both strictly INSIDE the box** -- F0.1(c).

    THE INTERIORITY CLAUSE IS LOAD-BEARING AND WAS ADDED AFTER SMOKE-TESTING THE
    DETECTOR ON REAL PROBES, BEFORE ANY fa1 DATA EXISTED.  Without it the detector
    is not a divergence test, it is a CLAMP test: `ar1-node-s2` sat on the floor
    (rec_lo 0.4521) and on the ceiling (rec_hi 0.1196) simultaneously, so both
    extremes were constant by construction and the naive detector reported a
    1,194-record 'freeze' on a perfectly healthy run.  A beta frozen ON THE GUARDS
    is a clamped network; a beta frozen STRICTLY INSIDE the box has a zero or NaN
    meta-gradient and is a DEAD one.

    Verified on disk, both directions:
      ar1-node-s2      naive 1,194  ->  with interiority 0   (healthy, clamped)
      bd7-w-c6-s0      4,042        ->  4,042                (collapsed=1)
      bd7-w-c6-s1      3,303        ->  3,303                (collapsed=1)
      bd7-node-c6-s1   0            ->  0                    (healthy, free)
    """
    R = records(dirpath)
    best = cur = 0
    for i in range(1, len(R)):
        same = (R[i]["beta_true_max"] == R[i - 1]["beta_true_max"]
                and R[i]["beta_true_min"] == R[i - 1]["beta_true_min"])
        interior = (R[i]["beta_true_min"] > lo + tol
                    and R[i]["beta_true_max"] < hi - tol)
        if same and interior:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def arm_rows(csv_path, fam, arm, health=None):
    """{seed: plateau5} for one arm of one family.

    **F0.1 IS ENFORCED HERE, NOT AFTERWARDS.**  A run that carries collapsed=1, or
    whose best_test - plateau5 exceeds LATE_CRASH_PP, never enters a cell.  The
    first registration filtered on epochs_done and plateau5 alone, which
    bd7-w-c6-s0 (epochs_done == requested, plateau5 = 10.000 present, probe full)
    would have passed -- dragging D_w ~80 pp into the INVERTS band.
    `health`, if given, collects (run, reason) for every row excluded.
    """
    out = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if not r["run"].startswith("%s-%s-s" % (fam, arm)):
                continue
            if int(r["epochs_done"] or 0) != EPOCHS or not r["plateau5"]:
                continue
            if (r.get("collapsed") or "0").strip() not in ("", "0"):
                if health is not None:
                    health.append((r["run"], "collapsed=1"))
                continue
            p5 = float(r["plateau5"])
            bt = r.get("best_test")
            if bt:
                gap = float(bt) - p5
                if gap > LATE_CRASH_PP:
                    if health is not None:
                        health.append((r["run"],
                                       "late crash: best_test - plateau5 = %.1f pp"
                                       % gap))
                    continue
            out[r["seed"]] = p5
    return out


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


def identity_slack(d):
    """min over records of the distance to the Lion bound, with the one-update
    offset.  Negative by more than one ms step == the config is not the header."""
    worst = float("inf")
    n = 0
    for r in records(d):
        st = int(r.get("step", 0))
        worst = min(worst,
                    r["beta_true_min"] - (BETA0 - MS * st),
                    (BETA0 + MS * st) - r["beta_true_max"])
        n += 1
    return worst, n


# ---------------------------------------------------------------------------
def selftest():
    p = n = 0

    def ck(name, cond):
        nonlocal p, n
        n += 1
        p += bool(cond)
        print("    %-72s %s" % (name, "ok" if cond else "FAIL"))

    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""
    doc = __doc__
    me = open(os.path.abspath(__file__)).read()
    flat = lambda s: " ".join(s.split())
    # The batch script's prose lives in a COMMENT BLOCK, so a sentence that wraps
    # carries a leading "#" on its continuation line.  flat_sh strips the comment
    # markers before normalising whitespace, so an assertion is about the SENTENCE
    # rather than the column it happened to wrap at.
    flat_sh = lambda s: " ".join(
        l.lstrip().lstrip("#").strip() for l in s.splitlines()).replace("  ", " ")

    print("c82_fa1_score selftest")
    print("  -- the constants, asserted against the BATCH SCRIPT's own text --")
    ck("the batch script exists at bin/c82_field_wideclip.sh", bool(src))
    ck("script CLIP= is this scorer's box", "CLIP=-25:-2.3026" in src)
    ck("script CLIP_REF= is ar1's box", "CLIP_REF=%s" % CLIP_REF in src)
    ck("script MST= is %s" % MST, "MST=%s" % MST in src)
    ck("script ALPHA0= is 1e-3", "ALPHA0=1e-3" in src)
    ck("script EPOCHS= is %d" % EPOCHS, "EPOCHS=%d" % EPOCHS in src)
    ck("script BATCH= is %d" % BATCH, "BATCH=%d" % BATCH in src)
    ck("script STEPS_PER_EPOCH= is %d" % STEPS_PER_EPOCH,
       "STEPS_PER_EPOCH=%d" % STEPS_PER_EPOCH in src)
    ck("script NJOBS= is %d per wave" % NJOBS, "NJOBS=%d " % NJOBS in src)
    ck("script NJOBS_TOTAL= is %d" % NJOBS_TOTAL,
       "NJOBS_TOTAL=%d" % NJOBS_TOTAL in src)
    ck("script submits in TWO WAVES", 'SEEDS_A="0 1 2"' in src and 'SEEDS_B="3 4 5"' in src)
    ck("the scorer's SEEDS are the union of the script's two waves",
       set(SEEDS) == {"0", "1", "2", "3", "4", "5"})
    ck("the registered n is %d" % N_SEEDS_REGISTERED,
       N_SEEDS_REGISTERED == len(SEEDS) and "REGISTERED n = 6" in src)
    ck("script CHUNK_K= is %d" % CHUNK_K, "CHUNK_K=%d" % CHUNK_K in src)
    ck("script CHUNK_K2= is %d" % CHUNK_K2, "CHUNK_K2=%d" % CHUNK_K2 in src)
    ck("script pins NET/DSET as shell vars the emitter really uses",
       "NET=ResNet18" in src and "DSET=CIFAR10" in src
       and '--dataset "$DSET" --NN-name "$NET"' in src)
    ck("script REF_FAM= is %s" % REF_FAM, "REF_FAM=%s" % REF_FAM in src)
    for a, m in sorted(M_OF_ARM.items()):
        keyed = {"node": "M_NODE", "ch": "M_CHUNK", "n1d": "M_N1D", "c23": "M_CHUNK2"}[a]
        ck("script %s=%d matches arm %r" % (keyed, m, a), "%s=%d" % (keyed, m) in src)
    ck("script's arm loop is exactly the 4 declared arms",
       all(tok in src for tok in ('"nodewise:node"', '"chunk${CHUNK_K}:ch"',
                                  '"nodewise1d:n1d"', '"chunk${CHUNK_K2}:c23"')))
    ck("GRAN_OF_ARM['ch'] is what the script's CHUNK_K expands to",
       GRAN_OF_ARM["ch"] == "chunk%d" % CHUNK_K)
    ck("GRAN_OF_ARM['c23'] is what the script's CHUNK_K2 expands to",
       GRAN_OF_ARM["c23"] == "chunk%d" % CHUNK_K2)
    ck("GRAN_OF_ARM's two non-chunk arms are literal in the script",
       GRAN_OF_ARM["node"] == "nodewise" and GRAN_OF_ARM["n1d"] == "nodewise1d")
    ck("script writes run names fa1-<arm>-s<seed>", 'RN="fa1-${SHORT}-s${S}"' in src)
    ck("script's probe dir carries the fa1 tag", "probe_${SHORT}_fa1_s${S}" in src)
    ck("script exports PROBE=5 AND PROBE5=1", "PROBE=5,PROBE5=1" in src)
    ck("script registers fa1 in c55 BOXES before submitting",
       '"fa1" not in c55.BOXES' in src)
    ck("script passes --weight-decay-meta 0 (the bound's precondition)",
       "--weight-decay-meta 0" in src)

    print("  -- the box, really registered, and really DIFFERENT from ar1's --")
    import c55_neff_noise as c55
    ck("fa1 is registered in c55 BOXES", "fa1" in c55.BOXES)
    ck("fa1's box equals this scorer's (LO, HI)",
       c55.BOXES.get("fa1", (0, 0, ""))[:2] == (LO, HI))
    ck("ar1 is registered in c55 BOXES", "ar1" in c55.BOXES)
    ck("fa1's box is NOT ar1's (F3 is a real contrast)",
       c55.BOXES.get("fa1", (0, 0, ""))[:2] != c55.BOXES.get("ar1", (1, 1, ""))[:2])
    ck("ar1's registered box is the CLIP_REF this scorer pairs against",
       c55.BOXES.get("ar1", (0, 0, ""))[:2]
       == tuple(float(x) for x in CLIP_REF.split(":")))

    print("  -- THE IDENTITY.  The whole ceiling rests on this arithmetic --")
    ck("T = EPOCHS * STEPS_PER_EPOCH = 50,000", T_UPDATES == 50000)
    ck("beta_0 = ln(alpha0) = -6.907755", abs(BETA0 + 6.907755279) < 1e-6)
    ck("ms*T = 15.0 exactly", abs(SPAN - 15.0) < 1e-9)
    ck("the reachable floor is -21.907755", abs(REACH_LO + 21.907755279) < 1e-6)
    ck("the reachable ceiling is +8.092245", abs(REACH_HI - 8.092245) < 1e-6)
    ck("the registered floor is STRICTLY below the reachable floor", LO < REACH_LO)
    ck("the CEILING is NOT budgeted from the identity -- it is BELOW the "
       "reachable maximum, deliberately", HI < REACH_HI)
    ck("the CEILING is byte-identical to ar1's",
       HI == float(CLIP_REF.split(":")[1]))
    ck("the script asserts the ceiling did NOT move (guard H2b)",
       "guard H2b" in src and "the CEILING moved" in src)
    ck("the script registers the -25:+9.0 box as WITHDRAWN",
       "-25:+9.0" in src and "Withdrawn" in src)
    ck("floor headroom is more than the batch's own horizon",
       (BETA0 - LO) / (MS * STEPS_PER_EPOCH) > EPOCHS)
    ck("the FLOOR is the only guard that moves",
       LO < float(CLIP_REF.split(":")[0]) and HI == float(CLIP_REF.split(":")[1]))
    ck("ar1's OLD floor was NOT reachable-proof (it is above the reachable floor)",
       float(CLIP_REF.split(":")[0]) > REACH_LO)
    ck("the script derives the bound rather than asserting it (GUARD H)",
       "GUARD H" in src and "HARD BOUND" in src)
    ck("the script validates the identity against ar1's own records",
       "guard H3" in src and "identity validated" in src)
    ck("the script asserts the clamp still runs BEFORE the probe",
       "clamps before PATCH_PROBE probes" in src)
    ck("the script MEASURES len(trainloader) rather than assuming T",
       "len(trainloader)" in src and "guard 4d" in src)
    ck("the script VOIDS the ceiling if Lion_meta_update changed",
       "THE BOUND IS VOID" in src)

    print("  -- the matched-count claims --")
    ck("D_w's pair is matched to <= 1 group",
       abs(M_OF_ARM["ch"] - M_OF_ARM["node"]) <= 1)
    ck("G_w's pair is matched EXACTLY", M_OF_ARM["c23"] == M_OF_ARM["n1d"])
    ck("the two pairs sit at DIFFERENT counts", M_OF_ARM["node"] != M_OF_ARM["n1d"])
    ck("exactly 2 pairs are declared", len(PAIRS) == 2)
    ck("every pair's arms are fa1 arms",
       all(h in ARMS and l in ARMS for _, h, l, _ in PAIRS))

    print("  -- the FOUR-WAY accuracy band, fixed here and in the script --")
    ck("+0.46 -> SURVIVES", band(+0.46) == "SURVIVES")
    ck("+0.45 -> SURVIVES (the boundary is CLOSED on the SURVIVES side)",
       band(+0.45) == "SURVIVES")
    ck("+0.44 -> ATTENUATED", band(+0.44) == "ATTENUATED")
    ck("+0.15 -> ATTENUATED", band(+0.15) == "ATTENUATED")
    ck("+0.14 -> COLLAPSES", band(+0.14) == "COLLAPSES")
    ck("0.00 -> COLLAPSES", band(0.0) == "COLLAPSES")
    ck("-0.14 -> COLLAPSES", band(-0.14) == "COLLAPSES")
    ck("-0.15 -> INVERTS", band(-0.15) == "INVERTS")
    ck("-0.60 -> INVERTS", band(-0.60) == "INVERTS")
    ck("the null band is SYMMETRIC about 0 at +-0.15",
       band(+0.149) == "COLLAPSES" and band(-0.149) == "COLLAPSES")
    ck("every box-free D anchor would read SURVIVES under these bands",
       all(band(v) == "SURVIVES" for v in D_BOXFREE_ANCHORS.values()))
    ck("the SURVIVES line sits BELOW the lowest box-free anchor",
       SURVIVES_AT < min(D_BOXFREE_ANCHORS.values()))
    ck("ar1's own bound D=+0.697 would read SURVIVES", band(D_AR1_BOUND) == "SURVIVES")
    ck("ar1's G=-0.139 would read COLLAPSES", band(G_AR1_BOUND) == "COLLAPSES")
    ck("cc1's G=+0.011 would read COLLAPSES", band(G_CC1_FREE) == "COLLAPSES")
    ck("the script states the same four bands",
       all(w in src for w in ("SURVIVES", "ATTENUATED", "COLLAPSES", "INVERTS")))
    ck("the script fixes the SURVIVES line at +0.45", ">= +0.45" in src)
    ck("the script declares the negative side deliberately not subdivided",
       "deliberately NOT subdivided" in src)

    print("  -- the statistics --")
    ck("_t of identical samples is 0", _t([1., 2., 3.], [1., 2., 3.]) == 0.0)
    ck("_t is antisymmetric",
       abs(_t([2., 3., 4.], [1., 2., 3.]) + _t([1., 2., 3.], [2., 3., 4.])) < 1e-12)
    ck("_sem of n=1 is nan", math.isnan(_sem([1.0])))
    ck("_t_paired of all-zero differences is nan (no spread, no claim)",
       math.isnan(_t_paired([0.0, 0.0, 0.0])))
    ck("_t_paired of a constant offset with spread is finite and signed",
       _t_paired([0.4, 0.5, 0.6]) > 0 and _t_paired([-0.4, -0.5, -0.6]) < 0)

    print("  -- the sign convention, the thing that must not drift --")
    conv = "higher N_eff/m -> higher plateau5 = CONCORDANT"
    ck("the convention is stated in the batch script", conv in flat(src))
    ck("the convention is stated in this scorer", conv in flat(doc))
    cc1 = open(os.path.join(HERE, "c81_cc1_score.py")).read()
    a3 = open(os.path.join(HERE, "c79_ar1_score.py")).read()
    ck("cc1's C1 convention line is byte-identical to F1's", conv in flat(cc1))
    ck("ar1's A3 convention line is byte-identical to F1's", conv in flat(a3))
    ck("the convention is declared transcribed unchanged",
       "TRANSCRIBED UNCHANGED" in src.upper() and "TRANSCRIBED UNCHANGED" in doc.upper())
    ck("the resolved threshold is |t| >= 2.0", RESOLVED_T == 2.0 and "|t| >= 2.0" in src)

    print("  -- F1 is a REPLICATION of a CLOSED question, and says so --")
    ck("the script names direction C as DROPPED",
       "DIRECTION C IS DROPPED" in src.upper())
    ck("the script forbids reopening it", "may reopen" in src or "REOPEN" in src.upper())
    ck("this scorer refuses to reopen direction C under ANY outcome",
       "will not reopen direction C under ANY F1 outcome" in flat(doc))
    ck("a CONCORDANT reading is registered as a DISCREPANCY, not a re-opening",
       "DISCREPANCY WITH cc1" in src and "DISCREPANCY WITH cc1" in doc)
    ck("cc1's own verdict is quoted with both channels' t",
       "-11.14" in doc and "-23.26" in doc and "-11.14" in src and "-23.26" in src)
    ck("the UNINFORMATIVE branch is registered",
       "UNINFORMATIVE" in src and "UNINFORMATIVE" in doc)
    ck("the DISSOCIATION branch is registered",
       "DISSOCIATION" in src and "DISSOCIATION" in doc)

    print("  -- THE BOX-OCCUPANCY GATE, and that it VOIDS the batch's OWN primary --")
    ck("the box gate is scored PER SEED in the script", "SCORED PER SEED" in src)
    ck("the script splits the box gate into F0.4a (LO) and F0.4b (HI)",
       "F0.4a" in src and "F0.4b" in src and "ALGEBRAIC" in src.upper()
       and "EMPIRICAL" in src.upper())
    ck("this scorer splits it the same way",
       "F0.4a" in doc and "F0.4b" in doc)
    ck("a bind DROPS the PAIR from F1 in the script",
       "PAIR IS DROPPED" in src.upper())
    ck("a bind DROPS the PAIR from F1 in this scorer",
       "PAIR IS DROPPED" in doc.upper())
    ck("the scorer will not read F1 on a bound or unhealthy pair",
       "will not read F1 on a pair whose arm is box-bound or unhealthy" in doc)
    ck("an HI bind is registered IN ADVANCE as expected on nodewise",
       "bound on 1/12" in flat_sh(src) and "1/12" in doc)
    ck("the box-free gate is the published 5%", BOXFREE_MAX == 0.05)
    ck("occupancy is reported for all 12 arms regardless of outcome",
       "whatever the outcome" in src and "whatever the outcome" in doc)
    ck("F0.5 declares a bind ALGEBRAICALLY IMPOSSIBLE",
       "ALGEBRAICALLY IMPOSSIBLE" in doc.upper() or "algebraically impossible" in doc)
    ck("a FLOOR touch sends the batch to VOID-and-debug, never to a rescore",
       # the docstring wraps this sentence across a line break, so the assertion
       # is about the SENTENCE (flat), not the column it wrapped at -- the c77
       # precedent: correct the SCORER's assertion, never the batch script.
       "do not rescore" in flat(doc).lower() and "do not rescore" in src.lower())

    print("  -- INSTRUMENT-ONLY: the comparability finding, stated in advance --")
    ck("the script states the batch is INSTRUMENT-ONLY", "INSTRUMENT-ONLY" in src)
    ck("this scorer states the batch is INSTRUMENT-ONLY", "INSTRUMENT-ONLY" in doc)
    ck("the script answers 'can fa1 re-read A1/A2?' with NO",
       "CAN fa1 RE-READ ar1's A1 / A2?  **NO." in src)
    # Both files wrap these sentences across comment/docstring line breaks, so the
    # assertions compare WHITESPACE-NORMALISED text: the check is about the
    # SENTENCE, not the column it happened to wrap at (the c77 precedent -- correct
    # the SCORER's assertion, never the batch script).
    ck("this scorer answers the same question with NO",
       flat("CAN fa1 RE-READ ar1's A1/A2?  NO.") in flat(doc))
    ck("the differential bind is quoted WITH its denominator (rule 21)",
       "of its 14,420 coordinates" in flat(src) and "14,420 coordinates" in flat(doc))
    ck("D_w is declared NOT a fifth replication",
       "fifth replication" in src and "fifth replication" in doc)
    ck("the scorer refuses to pool D_w with ar1's D",
       "will not pool D_w with ar1's D" in doc)
    ck("the sign of the box effect is declared UNSURE in advance",
       "UNSURE" in src and "UNSURE" in doc)
    ck("F3 is declared mandatory BECAUSE the sign is unpredictable",
       "why F3 is mandatory" in flat(src) or "that is why F3 is mandatory" in flat(doc))
    ck("F3's null is pre-emptively forbidden from becoming 'accuracy-neutral'",
       "accuracy-neutral" in src and "accuracy-neutral" in doc)

    print("  -- the premise, ASSERTED by the script rather than quoted --")
    ck("the script re-derives ar1's bind from the probes (guard 7)",
       "GUARD 7" in src and "the premise holds" in src)
    ck("guard 7 fails loudly if ar1 is not 12/12 bound", "only %d/12" in src)
    ck("the script re-derives the box-free D anchors from the CSV (guard 2d)",
       "guard 2d" in src and "0.485" in src and "0.581" in src and "0.727" in src)
    ck("guard 2 asserts beta_clip is the ONLY axis that moves",
       "the ONLY axis that moves" in src)
    # The guard's message is assembled from two adjacent Python string literals, so
    # the sentence is broken by quote characters that no whitespace normalisation
    # removes.  Assert the distinctive fragment plus the condition it guards.
    ck("guard 2c asserts the pairing partner exists at the SAME ms",
       "box+stepsize contrast" in src and "rows are not at ms=%s" in src)
    ck("the ms/box confound this batch retires is named with its counts",
       "39/39" in src and "15/15" in src and "39/39" in doc and "15/15" in doc)
    ck("STANDING RULE 10 is named as what the confound violates",
       "STANDING RULE 10" in src and "STANDING RULE 10" in doc)

    print("  -- the limits, stated in the scorer itself --")
    ck("scorer states it predates the data entirely", "before the data could exist" in doc)
    ck("the argmax limit is stated", "will not claim any arm's argmax" in doc)
    ck("the BN-vs-size non-identifiability is carried forward",
       "coincide EXACTLY (FINDINGS 78.1)" in doc)
    ck("the layer-boundary limit is carried forward in the script",
       "Layer boundaries stay untested" in src)
    ck("the n=3, one-architecture limit is stated in the script",
       "does not establish a law" in src)
    ck("scorer refuses to write the 53.1% sentence", 'will not print "53.1%"' in doc)
    ck("F4 is declared descriptive and unable to gate",
       flat("cannot gate F1, F2 or F3") in flat_sh(src)
       and "will not let F4 gate" in doc)
    ck("plateau5 is declared PRIMARY over best_test in the script",
       "plateau5 is PRIMARY" in src)

    print("  -- F0.1, THE HEALTH GATE, AND THE COLLAPSED-RUN LESSON --")
    ck("this scorer has an F0.1 health gate", "F0.1" in doc and "HEALTH" in doc)
    ck("the batch script has an F0.1 health gate", "F0.1" in src and "HEALTH" in src)
    ck("arm_rows() excludes collapsed=1 rows", 'r.get("collapsed")' in me)
    ck("arm_rows() excludes late crashes", "LATE_CRASH_PP" in me and "late crash" in me)
    ck("the late-crash threshold is %.1f pp" % LATE_CRASH_PP, LATE_CRASH_PP == 2.0)
    ck("a beta-freeze detector exists and is used",
       "def freeze_run" in me and "freeze_run(d)" in me)
    ck("the freeze threshold is %d records" % FREEZE_RECORDS, FREEZE_RECORDS == 1000)
    ck("the freeze detector requires BOTH sides frozen (a dead network)",
       'beta_true_max"] == R[i - 1]["beta_true_max"]' in me
       and 'beta_true_min"] == R[i - 1]["beta_true_min"]' in me)
    ck("the freeze detector ALSO requires both sides strictly INSIDE the box",
       "interior = (" in me and "> lo + tol" in me and "< hi - tol" in me)
    ck("the interiority clause is justified in the docstring by ar1-node-s2",
       "ar1-node-s2" in doc and "1,194" in doc)
    ck("both files carry the interiority clause",
       "INTERIORITY" in doc.upper() and "strictly INSIDE the box" in src)
    ck("the detector's smoke test is recorded in the code",
       "ar1-node-s2      naive 1,194" in me)
    ck("both files name bd7-w-c6 as the collapsed pair whose +3.436 was withdrawn",
       "bd7-w-c6" in src and "bd7-w-c6" in doc)
    ck("both files quote the freeze lengths 4,042 and 3,303",
       "4,042" in src and "3,303" in src and "4,042" in doc and "3,303" in doc)
    ck("the batch script GATES on bd7-w-c6 still reading collapsed=1 (guard H4)",
       "guard H4" in src and 'collapsed") == "1"' in src)
    ck("the corpus rule is stated in both files",
       "JOIN EVERY BETA STATISTIC" in src.upper()
       and "joining it against" in doc)
    ck("the scorer refuses to quote a beta statistic without the collapsed join",
       "will not quote a beta statistic from a run without joining it against"
       in flat(doc))

    print("  -- THE TWO-CLAUSE F2 VERDICT --")
    ck("verdict() exists and is two-clause", "def verdict" in me)
    ck("SURVIVES is declared without a second clause",
       verdict(+0.60, 0.10, D_POOL, D_POOL_SE)[0] == "SURVIVES")
    ck("a point estimate in ATTENUATED that is NOT resolved below the pool is "
       "NOT declared",
       verdict(+0.40, 0.10, D_POOL, D_POOL_SE)[0].startswith("CONSISTENT-WITH-BOX-FREE"))
    ck("a point estimate in ATTENUATED that IS resolved below the pool IS declared",
       verdict(+0.16, 0.02, D_POOL, D_POOL_SE)[0] == "ATTENUATED")
    ck("an unresolved COLLAPSES is not declared either",
       verdict(0.0, 0.30, D_POOL, D_POOL_SE)[0].startswith("CONSISTENT-WITH-BOX-FREE"))
    ck("a resolved INVERTS IS declared",
       verdict(-0.60, 0.05, D_POOL, D_POOL_SE)[0] == "INVERTS")
    ck("verdict() always reports the point-estimate band alongside",
       verdict(+0.40, 0.10, D_POOL, D_POOL_SE)[1] == "ATTENUATED")
    ck("the pooled box-free anchor is +0.5805 +- 0.0939",
       abs(D_POOL - 0.5805) < 1e-9 and abs(D_POOL_SE - 0.0939) < 1e-9)
    ck("the pool lies between the lowest and highest box-free anchor",
       min(D_BOXFREE_ANCHORS.values()) < D_POOL < max(D_BOXFREE_ANCHORS.values()))
    ck("the SURVIVES line still sits BELOW the pool", SURVIVES_AT < D_POOL)
    ck("the batch script states the second clause",
       "SECOND CLAUSE" in src.upper() and "0.5805" in src)
    ck("this scorer states the second clause",
       "SECOND CLAUSE" in doc.upper() and "0.5805" in doc)

    print("  -- F3 IS UNPAIRED, AND THE PAIRED CLAIM IS WITHDRAWN --")
    ck("seed is registered as NON-reproducible", SEED_IS_REPRODUCIBLE is False)
    ck("the measured per-run sd is 0.179 pp", abs(PER_RUN_SD - 0.179) < 1e-9)
    ck("both files quote the 356 same-seed replicate pairs",
       "356 same-config" in flat_sh(src) and "356 same-config" in flat(doc))
    ck("both files quote median |diff| 0.1710 pp",
       "0.1710" in src and "0.1710" in doc)
    ck("both files quote the across-seed sd 0.159 pp",
       "0.159" in src and "0.159" in doc)
    ck("both files call F3 UNPAIRED",
       "UNPAIRED" in src.upper() and "UNPAIRED" in doc.upper())
    ck("the paired-variance claim is explicitly WITHDRAWN in both",
       "WITHDRAWN" in src.upper() and "WITHDRAWN" in doc.upper())
    ck("F3 uses the two-sample _t, not _t_paired",
       "tt = _t(av, aw)" in me)
    ck("F3's resolution is registered at 0.25 pp",
       "0.25 pp" in src and "0.25 pp" in doc)
    ck("the EXPECTED |Delta| is registered BELOW what F3 resolves",
       "|Delta| < 0.10 pp" in flat_sh(src) and "below 0.10 pp" in flat(doc).lower()
       or "EXPECTED below 0.10 pp" in flat(doc))
    ck("an unresolved F3 is registered as the EXPECTED outcome",
       "EXPECTED outcome" in src and "EXPECTED outcome" in doc)

    print("  -- THE RE-RANKING, AND THE STOPPING RULE --")
    ck("F3 is named the PRIMARY in the script", "F3    **THE PRIMARY" in src)
    ck("F3 is named the PRIMARY in this scorer", "F3    **THE PRIMARY" in doc)
    ck("F2 is named SECONDARY in both", "F2    SECONDARY" in src and "F2    SECONDARY" in doc)
    ck("F1 is named DESCRIPTIVE in both",
       "F1    **DESCRIPTIVE" in src and "F1    **DESCRIPTIVE" in doc)
    ck("the demotion is justified by F1's own branches",
       "cannot change" in flat_sh(src) and "cannot change a conclusion" in flat(doc))
    ck("the stopping rule is stated in the script",
       "STOPPING RULE" in src.upper() and "READ ONCE, AT n=6" in src)
    ck("the stopping rule is stated in this scorer",
       "STOPPING RULE" in doc.upper() and "READ ONCE, AT n=6" in doc)
    ck("the scorer refuses a band label below the registered n",
       "will not emit a band label below the registered n" in doc
       and "interim" in me)
    ck("the script warns that stopping after wave a is a violation",
       "stopping-rule violation" in src)

    print("  -- THE NETWORK / DATASET JOIN (the CIFAR-100 head mismatch) --")
    ck("guard H1d parses --NN-name and --dataset off the emitter",
       "guard H1d" in src and "--NN-name" in src and "--dataset" in src)
    ck("guard 2 reads network/dataset from those variables, not a literal",
       "'network': net, 'dataset': dset" in src)
    ck("guard 4 builds the EMITTER's network", 'build_network(NET, "cpu")' in src)
    ck("guard 4 loads the EMITTER's dataset", "load_data(DSET, BATCH, 0)" in src)
    ck("guard 4c2 joins the classifier head to the dataset",
       "guard 4c2" in src and "HEAD MISMATCH" in src.upper())
    ck("the script records that len(trainloader)==500 for BOTH CIFAR-10 and -100",
       "CIFAR-100 (both are 50,000 train images)" in flat_sh(src))

    print("  -- THE HIER PRECONDITION ON THE IDENTITY --")
    ck("the export line pins HIER=none and SCHED=none",
       "HIER=none,SCHED=none" in src)
    ck("guard H1e asserts the pin", "guard H1e" in src)
    ck("the script names _apply_hier's additive branch as the threat",
       "_apply_hier" in src and "additive" in src)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_fa1"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs = sorted(glob.glob(os.path.join(a.root, "probe_*")))
    print("=" * 78)
    print("c82 -- fa1: THE ms/BOX DECONFOUND.  %d probe dirs" % len(dirs))
    print("box = (%.1f, %+.4f)   ms=%s   %d ep   seeds %s   (ar1's box was %s)"
          % (LO, HI, MST, EPOCHS, ",".join(SEEDS), CLIP_REF))
    print("ONE FIELD, TWO GUARDS -- AND ONLY THE FLOOR MOVES.  The ceiling %+.4f is"
          % HI)
    print("byte-identical to ar1/mm1/pp1/cc1, IS reachable, and bound on 1/12 ar1")
    print("arms.  F0.4a treats a FLOOR touch as algebraically impossible; F0.4b")
    print("treats a CEILING touch as an empirical fact to report.")
    print("PRIMARY = F3 (the box effect).  F2 secondary.  F1 DESCRIPTIVE.")
    print("D_w node/ch at m=14,420  |  G_w n1d/c23 at m=4,851 EXACT")
    print("**INSTRUMENT-ONLY: this batch does NOT re-read ar1's A1/A2.**")
    print("=" * 78)
    if not dirs:
        print("\n0 probe dirs.  **THAT IS ALWAYS A SYNC FAULT, NEVER A FINDING**")
        print("(CORRECTIONS 110.1): a probe-reading scorer needs the FULL probe_*")
        print("dirs including probe.jsonl, not just neg_counts.*.  Nothing scored.")
        return 1

    # --- F0 -----------------------------------------------------------------
    print("\n--- F0  VALIDITY (n_records==%d, beta moved, ep==%d)" % (N_RECORDS, EPOCHS))
    ok0 = 0
    for d in dirs:
        R = records(d)
        good = len(R) == N_RECORDS and R[-1]["beta_true_min"] != R[0]["beta_true_min"]
        ok0 += good
        print("    %-26s %s  n_rec=%d" % (os.path.basename(d),
                                          "PASS" if good else "FAIL", len(R)))
    print("    F0: %d/%d" % (ok0, len(dirs)))

    # --- F0.1  HEALTH -------------------------------------------------------
    print("\n--- F0.1  **HEALTH.**  beta-freeze detector on the probes; the CSV")
    print("    `collapsed` and late-crash checks are enforced inside arm_rows().")
    print("    A beta frozen on BOTH sides AND STRICTLY INSIDE the box for >= %d"
          % FREEZE_RECORDS)
    print("    consecutive records is a DEAD network (bd7-w-c6-s0/s1: 4,042 and")
    print("    3,303).  Frozen ON a guard is merely CLAMPED -- without that clause")
    print("    the detector misreads ar1-node-s2 (both guards at once) as a freeze.")
    frozen = {}
    ok1 = 0
    for d in dirs:
        fr = freeze_run(d)
        frozen[d] = fr
        good = fr < FREEZE_RECORDS
        ok1 += good
        print("    %-26s %s  longest frozen run = %d records"
              % (os.path.basename(d), "PASS" if good else "**VOID**", fr))
    print("    F0.1: %d/%d" % (ok1, len(dirs)))
    if ok1 != len(dirs):
        print("    **VOIDED ARMS ARE EXCLUDED FROM F1, F2 AND F3 -- never banded.**")

    # --- F0.2 ---------------------------------------------------------------
    print("\n--- F0.2  n_beta EXACT on EVERY record")
    ok2 = 0
    for d in dirs:
        arm = arm_of_dir(d)
        want = M_OF_ARM.get(arm)
        nb = sorted({r.get("n_beta") for r in records(d)})
        good = nb == [want]
        ok2 += good
        print("    %-26s %s  n_beta=%s  want %s = %s"
              % (os.path.basename(d), "PASS" if good else "FAIL", nb,
                 GRAN_OF_ARM.get(arm), want))
    print("    F0.2: %d/%d" % (ok2, len(dirs)))

    # --- F0.3 ---------------------------------------------------------------
    print("\n--- F0.3  THE INSTRUMENT FIRED (npy shape from its HEADER)")
    import numpy as np
    ok3 = 0
    for d in dirs:
        want = M_OF_ARM.get(arm_of_dir(d))
        jp, npp = os.path.join(d, "neg_counts.json"), os.path.join(d, "neg_counts.npy")
        good, shape, ntot = False, None, None
        if os.path.exists(jp) and os.path.exists(npp):
            ntot = json.load(open(jp)).get("n_tot")
            with open(npp, "rb") as fh:
                np.lib.format.read_magic(fh)
                shape = np.lib.format.read_array_header_1_0(fh)[0]
            good = (ntot == want) and (shape == (want,))
        ok3 += good
        print("    %-26s %s  n_tot=%s  npy_shape=%s"
              % (os.path.basename(d), "PASS" if good else "FAIL", ntot, shape))
    print("    F0.3: %d/%d" % (ok3, len(dirs)))

    # --- F0.4 ---------------------------------------------------------------
    print("\n--- F0.4  **THE BOX GATE, PER SEED, SPLIT BY GUARD.**")
    print("    F0.4a  LO -- ALGEBRAIC.  rec_lo MUST be 0.0000; the Lion identity")
    print("           forces beta >= %.6f and the floor is %.1f.  A touch means the"
          % (REACH_LO, LO))
    print("           CONFIG IS NOT THE HEADER -> VOID, debug, do not rescore.")
    print("    F0.4b  HI -- EMPIRICAL.  The ceiling %+.4f is UNCHANGED from ar1, IS"
          % HI)
    print("           reachable, and bound on 1/12 ar1 arms.  An arm at >= %.0f%% is"
          % (100 * BOXFREE_MAX))
    print("           dropped from F1 only; F2/F3 stand and carry its occupancy.")
    occ, fld = {}, {}
    lo_touch, hi_bound = [], []
    for d in dirs:
        o = occupancy(d, LO, HI)
        occ[d] = o
        lo_bad = o["rec_lo"] > 0.0
        hi_bad = o["rec_hi"] >= BOXFREE_MAX
        if lo_bad:
            lo_touch.append(d)
        if hi_bad:
            hi_bound.append(d)
        state = ("**LO TOUCH -- IMPOSSIBLE**" if lo_bad
                 else ("HI-BOUND" if hi_bad else "free"))
        print("    %-26s %-26s rec_lo %.4f  rec_hi %.4f  bmin %8.3f  bmax %+8.3f"
              % (os.path.basename(d), state,
                 o["rec_lo"], o["rec_hi"], o["min_final"], o["max_final"]))
        nf = neff_of_dir(d)
        if nf is not None:
            fld.setdefault(arm_of_dir(d), []).append(nf)
    print("    F0.4a: %d/%d arms with rec_lo == 0.0000 EXACTLY"
          % (len(dirs) - len(lo_touch), len(dirs)))
    print("    F0.4b: %d/%d arms with rec_hi < %.2f at the UNCHANGED ceiling"
          % (len(dirs) - len(hi_bound), len(dirs), BOXFREE_MAX))
    if lo_touch:
        print("    **THE BATCH IS VOID.**  A floor touch is algebraically impossible")
        print("    under wd_meta=0, ms=%s, T=%d and HIER=none.  DEBUG THE CONFIG."
              % (MST, T_UPDATES))
    if hi_bound:
        print("    NOTE: an HI bind was EXPECTED on nodewise -- ar1 carried the SAME")
        print("    ceiling and bound there on 1/12 (node s2, rec_hi 0.1196).  It is a")
        print("    condition SHARED by both halves of F3, not a difference.")
    # an arm is UNUSABLE for the FIELD if it touched either guard or froze
    bad_dirs = set(lo_touch) | set(hi_bound) | {d for d in dirs
                                               if frozen[d] >= FREEZE_RECORDS}
    bad_arms = {arm_of_dir(d) for d in bad_dirs}
    f1_void = False

    # --- F0.5 ---------------------------------------------------------------
    print("\n--- F0.5  **THE IDENTITY GATE.**  beta in [%.6f, %+.6f] is FORCED by"
          % (REACH_LO, REACH_HI))
    print("    beta <- (1-ms*wd)*beta - ms*sign(.) with wd_meta=0, ms=%s, T=%d."
          % (MST, T_UPDATES))
    print("    A guard touch here is ALGEBRAICALLY IMPOSSIBLE -> the config is not")
    print("    what the header says.  If this fires: VOID, do not rescore, debug.")
    ok5 = 0
    for d in dirs:
        slack, nrec = identity_slack(d)
        o = occ[d]
        # ONLY the FLOOR is algebraically impossible.  The ceiling is reachable
        # by the same identity (+8.092245) and is deliberately unchanged from ar1, so a
        # HI touch is F0.4b's business, not a violation of the bound.
        touched = o["rec_lo"] > 0
        good = (slack >= -1.5 * MS) and not touched
        ok5 += good
        print("    %-26s %s  worst slack %+.6f = %.2f ms steps  guard-touch %s"
              % (os.path.basename(d), "PASS" if good else "**VOID**", slack,
                 abs(slack) / MS, "YES" if touched else "no"))
    print("    F0.5: %d/%d" % (ok5, len(dirs)))
    if ok5 != len(dirs):
        print("    **THE BATCH IS VOID.**  The Lion identity forces |beta - beta_0|")
        print("    <= %.1f; a violation means wd_meta != 0, the wrong ms, the wrong" % SPAN)
        print("    step count or a resumed run.  DEBUG THE CONFIG.  Do not rescore.")

    # --- the cells ----------------------------------------------------------
    health = []
    cells = {arm: arm_rows(a.csv, "fa1", arm, health) for arm in ARMS}
    ref = {arm: arm_rows(a.csv, REF_FAM, arm, health) for arm in ARMS}
    if health:
        print("\n--- F0.1  ROWS EXCLUDED BY THE HEALTH GATE (never banded)")
        for run, why in health:
            print("    %-22s %s" % (run, why))
    # the beta-freeze veto from F0.1(c) removes whole arms from every gate
    for d in dirs:
        if frozen[d] >= FREEZE_RECORDS:
            print("    %-22s beta frozen for %d records -> arm %r VOIDED"
                  % (os.path.basename(d), frozen[d], arm_of_dir(d)))
            cells.pop(arm_of_dir(d), None)

    # --- THE STOPPING RULE ---------------------------------------------------
    n_min = min((len(cells.get(arm, {})) for arm in ARMS), default=0)
    interim = n_min < N_SEEDS_REGISTERED
    print("\n--- plateau5 per arm, from the CSV (per seed)")
    for arm in ARMS:
        v = cells.get(arm, {})
        vals = [v[s] for s in sorted(v)]
        print("    %-12s m=%-8d n=%d  %s  %s"
              % (GRAN_OF_ARM[arm], M_OF_ARM[arm], len(vals),
                 ("%.3f +-%.3f" % (statistics.mean(vals), _sem(vals)))
                 if len(vals) > 1 else "-",
                 " ".join("s%s=%.3f" % (s, v[s]) for s in sorted(v))))
    if interim:
        print("\n    **INTERIM-UNDERPOWERED: the thinnest arm has n=%d against the"
              % n_min)
        print("    REGISTERED n=%d.  NO BAND LABEL IS EMITTED BELOW THE REGISTERED n.**"
              % N_SEEDS_REGISTERED)
        print("    Occupancy and cell means above are the whole of what may be")
        print("    reported.  Stopping here because it 'looks clear' is a")
        print("    stopping-rule violation, forbidden in the registration.")

    # --- F3  ** THE PRIMARY ** ----------------------------------------------
    # Registered order: F3 first because it is the only gate that can change a
    # conclusion.  F1 held this slot in the first registration and was demoted --
    # every one of its branches ends in "direction C stays DROPPED".
    print("\n--- F3  **THE PRIMARY.  THE BOX EFFECT, PER ARM.  fa1 (%g:%g) vs %s (%s).**"
          % (LO, HI, REF_FAM, CLIP_REF))
    print("    **UNPAIRED Welch, n=%d vs n=3.**  The first registration called this a"
          % N_SEEDS_REGISTERED)
    print("    PAIRED test on shared seed labels; that claim is WITHDRAWN.  Measured")
    print("    over 356 same-config SAME-SEED replicate pairs: median |difference|")
    print("    0.1710 pp (per-run sd %.3f) against an across-seed sd of 0.159 pp."
          % PER_RUN_SD)
    print("    Seed carries no reproducibility on this cluster, so a same-seed")
    print("    difference is NOT a paired statistic.  se ~0.127 pp, resolves 0.25 pp.")
    print("    REGISTERED EXPECTATION: |Delta| < 0.10 pp, i.e. BELOW what this")
    print("    resolves.  **AN UNRESOLVED F3 IS THE EXPECTED OUTCOME.**  The")
    print("    deliverable is the box-free ms=3e-4 CELL, not this t-statistic.")
    print("    %-12s %10s %10s %10s %8s   %s"
          % ("arm", "fa1", REF_FAM, "Delta", "t", "verdict"))
    resolved_any = False
    for arm in ARMS:
        v, w = cells.get(arm, {}), ref.get(arm, {})
        av = [v[s] for s in sorted(v)]
        aw = [w[s] for s in sorted(w)]
        if len(av) < 2 or len(aw) < 2:
            print("    %-12s INSUFFICIENT n (fa1 %d, %s %d)"
                  % (GRAN_OF_ARM[arm], len(av), REF_FAM, len(aw)))
            continue
        dd = statistics.mean(av) - statistics.mean(aw)
        tt = _t(av, aw)
        res = (not math.isnan(tt)) and abs(tt) >= RESOLVED_T
        resolved_any |= res
        print("    %-12s %10.3f %10.3f %+10.3f %8s   %s  (n=%d v %d)"
              % (GRAN_OF_ARM[arm], statistics.mean(av), statistics.mean(aw), dd,
                 ("%+.2f" % tt) if not math.isnan(tt) else "nan",
                 "**RESOLVED**" if res else "unresolved", len(av), len(aw)))
    if interim:
        print("    -> **INTERIM-UNDERPOWERED.**  Below the registered n; no verdict.")
    elif resolved_any:
        print("    -> **THE BOX CHANGED THE OPTIMISER, NOT MERELY THE INSTRUMENT.**")
        print("       D_w may NOT be pooled with ar1's D, and every prose sentence")
        print("       about D must carry its box from here on.")
    else:
        print("    -> **UNRESOLVED at this test's own 0.25 pp floor -- THE EXPECTED")
        print("       OUTCOME, registered in advance.**  **This is NOT 'the box is")
        print("       accuracy-neutral'** -- declared before the data, because that")
        print("       sentence has been written off an underpowered null here before.")
        print("       WHAT IS DELIVERED ANYWAY: a box-free ms=3e-4 cell, which retires")
        print("       the STANDING RULE 10 breach (39/39 free at ms=1e-4 vs 15/15")
        print("       LO-bound at ms=3e-4) whatever this t says.")

    # --- F2  SECONDARY ------------------------------------------------------
    print("\n--- F2  SECONDARY.  THE ACCURACY CONTRASTS **IN THIS BOX ONLY.**")
    print("    registered band: >= +%.2f SURVIVES | [+%.2f,+%.2f) ATTENUATED | "
          "(-%.2f,+%.2f) COLLAPSES | <= -%.2f INVERTS"
          % (SURVIVES_AT, NULL_HALF, SURVIVES_AT, NULL_HALF, NULL_HALF, NULL_HALF))
    print("    **TWO-CLAUSE.**  ATTENUATED / COLLAPSES / INVERTS may be DECLARED only")
    print("    if the estimate is ALSO below its reference with |t| >= %.1f."
          % RESOLVED_T)
    print("    D_w's reference is the inverse-variance pool of the three BOX-FREE D")
    print("    readings: %+.4f +- %.4f (mm1 %.3f/0.1613, pp1 %.3f/0.1414,"
          % (D_POOL, D_POOL_SE, D_BOXFREE_ANCHORS["mm1"], D_BOXFREE_ANCHORS["pp1"]))
    print("    cc1 %.3f/0.2001; Q = 0.88 on 2 df, homogeneous).  G_w's reference is"
          % D_BOXFREE_ANCHORS["cc1"])
    print("    0.0 -- its registered expectation is the null itself.")
    got = {}
    for gate, hi, lo, what in PAIRS:
        vh, vl = cells.get(hi, {}), cells.get(lo, {})
        ah = [vh[s] for s in sorted(vh)]
        al = [vl[s] for s in sorted(vl)]
        if len(ah) < 2 or len(al) < 2:
            print("    %s: INSUFFICIENT n (%d v %d)" % (gate, len(ah), len(al)))
            continue
        g = statistics.mean(ah) - statistics.mean(al)
        se = math.sqrt(_sem(ah) ** 2 + _sem(al) ** 2)
        got[gate] = g
        ref_v, ref_se = ((D_POOL, D_POOL_SE) if gate == "D_w"
                         else (G_EXPECT, G_EXPECT_SE))
        decl, lab, tref = verdict(g, se, ref_v, ref_se)
        print("\n    %s  isolates: %s" % (gate, what))
        print("    %-12s (m=%d) %.3f +-%.3f (n=%d)"
              % (GRAN_OF_ARM[hi], M_OF_ARM[hi], statistics.mean(ah), _sem(ah), len(ah)))
        print("    %-12s (m=%d) %.3f +-%.3f (n=%d)"
              % (GRAN_OF_ARM[lo], M_OF_ARM[lo], statistics.mean(al), _sem(al), len(al)))
        print("    %s = %+.3f pp  (se %.3f, within-batch t %.2f)"
              % (gate, g, se, _t(ah, al)))
        if interim:
            print("    -> **INTERIM-UNDERPOWERED.  NO BAND LABEL EMITTED** (thinnest")
            print("       arm n=%d against the registered n=%d)." % (n_min, N_SEEDS_REGISTERED))
        else:
            print("    point-estimate band: %s;  vs reference %+.4f: t %s"
                  % (lab, ref_v, ("%+.2f" % tref) if not math.isnan(tref) else "n/a"))
            print("    -> **%s**  [at BETA_CLIP=%g:%g]" % (decl, LO, HI))
            if decl != lab:
                print("       The point estimate reads %s, but it is NOT resolved" % lab)
                print("       below the reference, so %s MAY NOT BE DECLARED." % lab)
        if gate == "D_w":
            print("    box-free anchors: %s;  ar1 (BOX-BOUND 12/12) read %+.3f"
                  % (", ".join("%s %+.3f" % kv for kv in
                               sorted(D_BOXFREE_ANCHORS.items())), D_AR1_BOUND))
            print("    **NOT a fifth replication of D.  fa1 is a different box and")
            print("    pooling across boxes is the error F3 exists to detect.**")
        else:
            print("    prior expectation, registered in advance: ar1 %+.3f, cc1 %+.3f,"
                  % (G_AR1_BOUND, G_CC1_FREE))
            print("    so COLLAPSES was expected; a resolved reading OUTSIDE the null")
            print("    band UNDERCUTS the tail interpretation of A1-A2.")

    # --- F1  DESCRIPTIVE ----------------------------------------------------
    print("\n--- F1  **DESCRIPTIVE (demoted from PRIMARY in this revision).**")
    print("    The concordance reading at ms=%s under a floor that cannot bind." % MST)
    print("    convention: higher N_eff/m -> higher plateau5 = CONCORDANT")
    print("    **REPLICATION ONLY.  Direction C is CLOSED (cc1's C1 was MIXED);")
    print("    no outcome below reopens it.  F1 cannot change any conclusion, which")
    print("    is exactly why it no longer holds the headline slot.**")
    labels = []
    for gate, hi, lo, what in PAIRS:
        vh, vl = cells.get(hi, {}), cells.get(lo, {})
        ah = [vh[s] for s in sorted(vh)]
        al = [vl[s] for s in sorted(vl)]
        if hi in bad_arms or lo in bad_arms:
            print("    %s: an arm binds at a guard or failed F0.1 -> **PAIR DROPPED**"
                  % gate)
            print("       (%s bound/unhealthy: %s)"
                  % (gate, ", ".join(sorted({x for x in (hi, lo) if x in bad_arms}))))
            continue
        if len(ah) < 2 or len(al) < 2 or hi not in fld or lo not in fld:
            print("    %s: accuracy or field absent on an arm -> UNINFORMATIVE" % gate)
            labels.append("UNINFORMATIVE")
            continue
        da, ta = statistics.mean(ah) - statistics.mean(al), _t(ah, al)
        df, tf = (statistics.mean(fld[hi]) - statistics.mean(fld[lo]),
                  _t(fld[hi], fld[lo]))
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
        print("\n    %s  %s - %s  (%s)"
              % (gate, GRAN_OF_ARM[hi], GRAN_OF_ARM[lo], what))
        print("      dplateau5 = %+7.3f pp  (t %6.2f)  %s"
              % (da, ta, "resolved" if ares else "UNRESOLVED"))
        print("      dN_eff/m  = %+7.4f     (t %6.2f)  %s"
              % (df, tf, "resolved" if fres else "UNRESOLVED"))
        print("      -> **%s**" % lab)
    f1_void = not labels
    lset = set(l for l in labels if l != "UNINFORMATIVE")
    print("\n    F1 VERDICT over %d readable pairs: %s"
          % (len(labels), ", ".join(labels) if labels else "none"))
    if interim:
        print("    -> **INTERIM-UNDERPOWERED**: below the registered n; logged only.")
    elif f1_void:
        print("    -> **VOID**: both pairs dropped for a guard bind or an F0.1")
        print("       failure, so N_eff/m is not interpretable (rule 5).  Registered")
        print("       in advance.  This is NOT another owed re-run: the question is")
        print("       closed and a second void would only say this box binds too.")
    elif not lset:
        print("    -> **UNINFORMATIVE**: the field did not resolve on either pair.")
    elif lset == {"ANTI-CONCORDANT"} and labels.count("ANTI-CONCORDANT") == 2:
        print("    -> **REPLICATES cc1's ANTI-CONCORDANT LEG** at a second stepsize")
        print("       and a non-binding floor.  Direction C stays DROPPED.")
    elif lset == {"CONCORDANT"} and labels.count("CONCORDANT") == 2:
        print("    -> **DISCREPANCY WITH cc1, LOGGED FOR THE RECORD.**  This does")
        print("       NOT reopen direction C.  A field whose sign depends on the")
        print("       stepsize or the box is LESS of a design variable, not more.")
    else:
        print("    -> **REPLICATES cc1's MIXED VERDICT.**  The field is not a")
        print("       sufficient statistic.  Direction C stays DROPPED.")

    # --- F4 -----------------------------------------------------------------
    print("\n--- F4  THE CLIP METER.  **DESCRIPTIVE.  CANNOT GATE F1, F2 OR F3.**")
    print("    %-26s %8s %8s %10s %10s %10s %10s"
          % ("dir", "rec_lo", "rec_hi", "coord_lo", "coord_hi", "bmin", "bmax"))
    for d in dirs:
        o = occ[d]
        cl = ("%10.6f" % o["coord_lo"]) if o["coord_lo"] is not None else "        NA"
        ch = ("%10.6f" % o["coord_hi"]) if o["coord_hi"] is not None else "        NA"
        print("    %-26s %8.4f %8.4f %s %s %10.3f %+10.3f"
              % (os.path.basename(d), o["rec_lo"], o["rec_hi"], cl, ch,
                 o["min_final"], o["max_final"]))
    print("    FLOOR headroom the identity left: %.3f below the reachable %.3f"
          % (REACH_LO - LO, REACH_LO))
    print("    CEILING: %+.4f, UNCHANGED from ar1.  It is %+.3f BELOW the reachable"
          % (HI, HI - REACH_HI))
    print("    maximum %+.3f, i.e. it CAN bind -- deliberately, because it is a"
          % REACH_HI)
    print("    stability device and not an instrument.  The corpus's only")
    print("    released-ceiling runs (bd7-w-c6, HI +6.0) are 2/2 collapsed=1.")
    print("    JOIN EVERY BETA STATISTIC ABOVE AGAINST `collapsed` BEFORE QUOTING IT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
