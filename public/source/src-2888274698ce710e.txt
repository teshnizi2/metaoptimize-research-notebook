#!/usr/bin/env python3
"""c84_gn1_score.py -- score `gn1`.  DOES THE PARTITION GAP SURVIVE A CHANGE OF
NORMALISER AT BYTE-IDENTICAL SINGLETON STRUCTURE?

REGISTERED GATES, transcribed from `bin/c84_normaliser_transfer.sh` and
`docs/REGISTER-ideas-ABC.md` section 4.  Every constant below is asserted by the
selftest against THAT SCRIPT'S OWN TEXT so the registration and the batch cannot
drift -- STANDING RULE 19.

WRITTEN, SELFTESTED AND COMMITTED WHILE **NO `gn1` RUN EXISTED IN THE CSV AT ALL**,
and while the batch had not been submitted.  Not merely before the verdict was read --
before the data could exist.

THE FOUR ARMS, ONE BATCH, ms=1e-4, BETA_CLIP -25:-2.3026, 100 ep, seeds 0-5:
    bn-node  ResNet18     nodewise   m = 14,420      |  bn-ch  ResNet18     chunk777  m = 14,421
    gn-node  ResNet18_gn  nodewise   m = 14,420      |  gn-ch  ResNet18_gn  chunk777  m = 14,421
Count mismatch +1 group in 14,420 = 0.007%, identical to every previous reading of D.

THE QUESTION.  Is the partition gap carried by the degenerate size-1 tail PER SE, or by
BATCHNORM specifically?  `nn.GroupNorm(32, C)` has weight and bias of shape (C,) exactly
as `nn.BatchNorm2d(C)` does; BN's running_mean/running_var are BUFFERS that never enter
named_parameters(); and neither train.py nor HF.py contains any module-type logic.  So
the two networks present the optimiser with the SAME partition object -- same 9,610
size-1 groups on the same 41 one-D tensors in the same positions -- and exactly one
thing changes.  `bin/c84_normaliser_transfer.sh` guard 4 MEASURES that identity from the
allocated beta before anything is submitted.

  T0    **TRAINS-AT-ALL AND BOX GATE, EVALUATED FIRST, PER RUN.**
        T0.1 epochs_done == epochs_requested == 100.
        T0.2 collapsed == 0 (aggregate.py:155 verbatim).
        T0.3 final_test > 60.0 AND plateau5 >= 85.0.
        T0.4 NO LATE BLOW-UP: best_test - plateau5 <= 2.0 pp (the bd7-w-c6 phenotype:
             best 88.82, final 10.0).
        T0.5 BOX OCCUPANCY IN BOTH DIRECTIONS (RULE 13), rec_lo <= 0.05 AND
             rec_hi <= 0.05, **MEASURED ON THE gn ARMS AND NEVER INHERITED FROM THE bn
             ANCHOR** -- the GN net's beta trajectory is not the BN net's.
        FIRING: an arm failing T0.1-T0.4 on >= 2 of its 6 seeds is VOID; an arm failing
        T0.5 is UNINTERPRETABLE (RULE 5) and drops.  >= 2 of the 4 arms void or dropped
        => THE WHOLE BATCH IS VOID.

  T1    **THE POSITIVE-CONTROL GATE.**  D_BN = plateau5(bn-ch) - plateau5(bn-node).
        If D_BN < +0.30 OR t(D_BN) < 2, THE BATCH IS VOID AS A TEST OF GROUPNORM.  You
        cannot ask whether an effect transfers in a batch where your own control does
        not show it.  D_BN is then recorded as a failed sixth replication of D -- a
        serious finding in its own right -- but NOTHING ABOUT GROUPNORM MAY BE CONCLUDED.

  T2    **THE PRIMARY.**  D_GN = plateau5(gn-ch) - plateau5(gn-node), Welch 6v6.
          TRANSFERS          D_GN >= +0.30 AND t(D_GN) >= 2.
          DOES NOT TRANSFER  D_GN in [-0.15, +0.15] AND se(D_GN) <= 0.13.
          UNDECIDED          everything else, including D_GN in (0.15, 0.30) and
                             **including ANY NULL whose se exceeds 0.13**.  An
                             underpowered null is UNDECIDED and MAY NOT be reported as
                             "no gap".
        se(D_GN) <= 0.13 is FIXED IN ADVANCE at 1.3x the planning se of 0.100 (sigma =
        0.174 at this cell) and is NOT to be recomputed post hoc.
        THE TAIL MECHANISM AS CURRENTLY WRITTEN IS **REFUTED** if D_GN lands in the null
        band, POWERED, while D_BN passes T1 -- a within-batch dissociation at
        byte-identical partition structure.  Consequence, written before the data: the
        paper's prescriptive sentence narrows from "do not give normalisation scalars
        their own step sizes" to "do not give BATCHNORM scalars their own step sizes",
        with NO LICENCE TO GENERALISE to normalisation layers at large.

  T2b   **SECONDARY.**  dD = D_GN - D_BN (planning se 0.151).
          |dD| <= 0.30            -> the two normalisers agree within resolution.
          dD <= -0.30 AND t <= -2 -> RESOLVED ATTENUATION under GroupNorm.  Reported as
                                     ATTENUATION and NOT as collapse unless D_GN
                                     independently lands in the null band.

  T2c   **ERROR-BUDGET-NORMALISED, DESCRIPTIVE ONLY.**  D / (100 - level) per net,
        printed beside the pp numbers so a scale artefact is visible rather than
        hidden.  It gates nothing: registering two primaries would move the argmax
        problem up a level rather than solve it.

  T0.6  **THE COMMENSURABILITY GATE** (added at cycle-84 review, evaluated after
        T0.1-T0.5 and BEFORE T2).  |mean plateau5 over the GN arms - mean over the BN
        arms| <= 2.0 pp.  T0.3's floor of 85 does not protect this batch's arithmetic:
        BN plateaus near 92.2 (budget 7.8 pp) and a GN arm at 88.0 would pass every
        other gate on a 12.0 pp budget -- a 1.6x change in the scale the effect lives
        on.  If it fires the batch reports "GroupNorm sits in a different accuracy
        regime; D_GN and D_BN are not commensurable in pp" and issues NO transfer
        verdict.  **THAT IS NOT A NULL.**

  T2's AMBIGUITY, REGISTERED IN ADVANCE.  A T2 refutation DOES NOT IDENTIFY THE
  REPLACEMENT CARRIER.  nn.GroupNorm(32, C) takes num_groups=32, so it forms **32
  GROUPS OF C/32 CHANNELS** -- 2, 4, 8 and 16 at ResNet18's widths 64/128/256/512.
  (An earlier draft said "blocks of 32 channels".  THAT WAS BACKWARDS, and since it was
  the sole basis of this paragraph it is corrected here, BEFORE any run.)  Per-channel
  scale invariance -- which BatchNorm provides, making gamma_c the sole controller of a
  channel's effective magnitude -- therefore does NOT hold, and the departure is GRADED
  BY DEPTH: nearly per-channel at layer1, 16-to-a-group at layer4.  Only the PARTITION
  role of a singleton is preserved, not its FUNCTIONAL role.  The test is ASYMMETRIC:
  **a POSITIVE IS CLEAN; A NULL IS AMBIGUOUS BETWEEN THREE READINGS** -- (i) batch
  statistics, (ii) per-channel scale invariance (depth-graded), (iii) MISTUNING, since
  this runs a different network at a BatchNorm-derived cell with no argmax located on
  either net (RULE 11).  This batch separates NONE of them, so **a null from gn1 alone
  may NOT be written as a mechanism narrowing**; it requires gn2a (the alpha0=3e-4
  tuning bracket, provably box-free in both directions) and gn2b (GroupNorm(num_groups
  == C)), both specified in advance in bin/c84_normaliser_transfer.sh.

  T3    **ANTI-NARRATION.**  T3.1 a VOID GN cell is not evidence of anything and may not
        be salvaged by adding seeds after seeing which side it fell on.  T3.2 NO
        REBRANDING -- this batch may not be written up as a no-BatchNorm test, a
        norm-free test, or a falsification of the singleton mechanism by removal; IT
        REMOVES ZERO SINGLETONS.  T3.3 NO CROSS-BOX POOLING -- every D sentence carries
        its cell.  T3.4 no argmax harvesting; the four-arm table is reported whole.

WHAT THIS SCORER WILL NOT DO
  * It will not read D_GN if T0 voids the batch or if T1 fails.  A void cell is not a null.
  * It will not compute ANY field statistic.  At this cell no coordinate reaches either
    guard, so coord_lo is identically 0 and the clip-severity channel yields nothing;
    rec_lo/rec_hi are the T0.5 gate only.  Recovering that channel would require a
    BINDING box, which would trade an interpretable accuracy contrast for an
    uninterpretable one.
  * It will not pool D_GN or D_BN with mm1/pp1/cc1/ar1/fa1.  Those are a cross-batch
    comparison and are printed as a DESCRIPTIVE anchor row that cannot gate.
  * It will not report an underpowered null as "no gap".
  * It will not claim anything about GroupNorm as a training recipe.

USAGE
  python3 analysis/c84_gn1_score.py --selftest
  python3 analysis/c84_gn1_score.py --root ../probes_gn1 --csv results/all_runs.csv
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
SCRIPT = os.path.join(REPO, "bin", "c84_normaliser_transfer.sh")

# `occupancy` counts, per probe dir, the fraction of RECORDS with any beta at each
# guard (rec_lo / rec_hi) and returns the record count T -- the A0 validity check
# and the T0.5 gate come from the same trusted reader every batch uses.
from c52_boxfree import occupancy                     # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
TAG = "gn1"
LO, HI = -15.0, -2.3026             # `gn1`, registered in c55_neff_noise.BOXES.
                                    # MOVED FROM -25 AT CYCLE-84 REVIEW: (-25, ms=1e-4)
                                    # exists in ZERO corpus rows, while every threshold
                                    # below was computed from -15 data.  The Lion
                                    # identity proves the two floors are identical in
                                    # effect here (reach [-11.908, -1.908]), so the move
                                    # is free and it removes a cross-box import.
EPOCHS = 100
N_RECORDS = 10000
MST = "1e-4"
CHUNK_K = 777
NJOBS = 24
# **SEEDS ARE NOT EQUAL ACROSS ARMS.**  The BN control only has to CLEAR a gate
# (t >= 2 against +0.5805), which n=4 does at t = 4.72; the GN arms carry the PRIMARY.
# 6/6/6/6 -> 4/4/8/8 keeps the batch at 24 jobs and moves se(D_GN) 0.100 -> 0.0870.
SEEDS_BN = (0, 1, 2, 3)
SEEDS_GN = (0, 1, 2, 3, 4, 5, 6, 7)
SEEDS = tuple(sorted(set(SEEDS_BN) | set(SEEDS_GN)))
ARMS = ("bn-node", "bn-ch", "gn-node", "gn-ch")
SEEDS_OF_ARM = {"bn-node": SEEDS_BN, "bn-ch": SEEDS_BN,
                "gn-node": SEEDS_GN, "gn-ch": SEEDS_GN}
# n_beta per arm, MEASURED by guard 4 from the ALLOCATED beta on BOTH built networks.
M_OF_ARM = {"bn-node": 14420, "bn-ch": 14421, "gn-node": 14420, "gn-ch": 14421}
NET_OF_ARM = {"bn-node": "ResNet18", "bn-ch": "ResNet18",
              "gn-node": "ResNet18_gn", "gn-ch": "ResNet18_gn"}
GRAN_OF_ARM = {"bn-node": "nodewise", "bn-ch": "chunk777",
               "gn-node": "nodewise", "gn-ch": "chunk777"}

# THE BANDS, FIXED BEFORE THE DATA EXISTS.
NULL_HALF = 0.15                    # the null band, identical to bn1's T1 and ar1's A1
DECIDE = 0.30                       # the TRANSFERS line, ~half the pooled anchor
SE_MAX = 0.113                      # THE POWERED-NULL THRESHOLD.  1.3 x the planning
                                    # se of 0.0870 (n=8).  NOT recomputed post hoc.  At
                                    # p90 corpus noise (sd 0.24) the realised se would be
                                    # 0.120 > 0.113 -> UNDECIDED.  Intended conservatism.
RESOLVED_T = 2.0
DD_BAND = 0.30                      # |dD| <= this => the normalisers agree
SIGMA = 0.174                       # pooled within-cell sd of plateau5 at this cell
SE_D_PLAN = 0.0870                  # SIGMA*sqrt(2/8), the PRIMARY at n=8
SE_D_BN_PLAN = 0.1230               # SIGMA*sqrt(2/4), the CONTROL at n=4
SE_DD_PLAN = 0.1507                 # sqrt(SE_D_PLAN^2 + SE_D_BN_PLAN^2)
COMM_MAX = 2.0                      # **THE COMMENSURABILITY GATE (T0.6).**  |mean
                                    # plateau5 over the GN arms - mean over the BN arms|.
                                    # D_GN and D_BN are compared in PERCENTAGE POINTS,
                                    # which only means something if the two nets sit on
                                    # comparable error budgets.  A GN arm at 88.0 clears
                                    # every other T0 gate while sitting on a 12.0 pp
                                    # budget against BN's 7.8 -- a 1.6x change in the
                                    # scale the effect lives on.  If the gap is even
                                    # partly MULTIPLICATIVE in the budget that distorts
                                    # D_GN by ~60% = ~7x se, straddling BOTH the +0.30
                                    # line and the null band, so a T2 firing could be a
                                    # pure accuracy-regime artefact.
ANCHOR = 0.5805                     # IV pool of D at ms=1e-4 (mm1 +0.485 / pp1 +0.581 /
ANCHOR_SE = 0.0939                  # cc1 +0.727).  DESCRIPTIVE here: cross-batch.

# T0 thresholds.
FINAL_FLOOR = 60.0
PLATEAU_FLOOR = 85.0
LATE_BLOWUP = 2.0
REC_MAX = 0.05
VOID_SEEDS = 2                      # >= this many bad seeds voids an arm
VOID_ARMS = 2                       # >= this many void/dropped arms voids the batch

PAIRS = (
    ("D_BN", "bn-ch", "bn-node", "the POSITIVE CONTROL -- a GATE (T1), not a result"),
    ("D_GN", "gn-ch", "gn-node", "THE PRIMARY -- does the gap survive the normaliser?"),
)


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def _welch(a, b):
    """(difference, se, t).  Unpaired -- seeds do not pair, RULE 14."""
    d = statistics.mean(a) - statistics.mean(b)
    se = math.sqrt(_sem(a) ** 2 + _sem(b) ** 2)
    return d, se, (d / se if se > 0 else float("nan"))


def band(d, se, t):
    """THE THREE REGISTERED OUTCOMES for D_GN.  No reinterpretable middle.

    An underpowered null is UNDECIDED, never "no gap" -- that is the whole point of
    carrying `se` into this function.
    """
    if d >= DECIDE and t >= RESOLVED_T:
        return "TRANSFERS"
    if -NULL_HALF <= d <= NULL_HALF and se <= SE_MAX:
        return "DOES NOT TRANSFER"
    return "UNDECIDED"


def dd_band(dd, t):
    if dd <= -DD_BAND and t <= -RESOLVED_T:
        return "RESOLVED ATTENUATION"
    if abs(dd) <= DD_BAND:
        return "AGREE WITHIN RESOLUTION"
    return "UNRESOLVED"


def rows(csv_path, prefix):
    """Every CSV row whose run name starts with the prefix, per seed."""
    out = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["run"].startswith(prefix):
                out.append(r)
    return sorted(out, key=lambda r: r["run"])


def _f(r, k, default=float("nan")):
    v = (r.get(k) or "").strip()
    try:
        return float(v)
    except ValueError:
        return default


def t0_of_row(r):
    """T0.1-T0.4 per RUN.  Returns (ok, [reasons])."""
    bad = []
    ed, er = (r.get("epochs_done") or "").strip(), (r.get("epochs_requested") or "").strip()
    if ed != str(EPOCHS) or (er and er != str(EPOCHS)):
        bad.append("T0.1 epochs %s/%s" % (ed or "?", er or "?"))
    if (r.get("collapsed") or "0").strip() not in ("0", ""):
        bad.append("T0.2 collapsed")
    ft, p5, bt = _f(r, "final_test"), _f(r, "plateau5"), _f(r, "best_test")
    if not (ft > FINAL_FLOOR):
        bad.append("T0.3 final_test %.3f" % ft)
    if not (p5 >= PLATEAU_FLOOR):
        bad.append("T0.3 plateau5 %.3f" % p5)
    if not (bt - p5 <= LATE_BLOWUP):
        bad.append("T0.4 best-plateau5 %.3f" % (bt - p5))
    return (not bad), bad


def arm_of_dir(d):
    """probe_<arm>_gn1_s<seed>  ->  <arm>.  Arm names carry hyphens, never underscores."""
    parts = os.path.basename(d).split("_")
    return parts[1] if len(parts) > 2 and parts[0] == "probe" else None


def csv_cell(csv_path, gran, net, ms, clip):
    """plateau5 at a full axis signature.  DESCRIPTIVE anchor rows only."""
    out = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["granularity"] != gran or r["network"] != net:
                continue
            if r["meta_stepsize"] != ms or r["beta_clip"] != clip:
                continue
            if (r.get("epochs_done") or "") != str(EPOCHS) or not (r.get("plateau5") or "").strip():
                continue
            if r["augment"] != "1" or r["alpha0"] != "1e-3" or r["dataset"] != "CIFAR10":
                continue
            out.append(float(r["plateau5"]))
    return sorted(out)


def selftest():
    p = n = 0

    def ck(name, cond):
        nonlocal p, n
        n += 1
        p += bool(cond)
        print("    %-74s %s" % (name, "ok" if cond else "FAIL"))

    print("c84_gn1_score selftest")
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""
    me = open(os.path.abspath(__file__)).read()

    # --- the batch script exists and this scorer matches ITS OWN TEXT -------
    ck("the batch script exists at bin/c84_normaliser_transfer.sh", bool(src))
    ck("script CLIP= is this scorer's box", "CLIP=-25:-2.3026" in src)
    ck("script MST= is %s" % MST, "MST=%s" % MST in src)
    ck("script EPOCHS= is %d" % EPOCHS, "EPOCHS=%d" % EPOCHS in src)
    ck("script NJOBS= is %d" % NJOBS, "NJOBS=%d" % NJOBS in src)
    ck("script CHUNK_K= is %d" % CHUNK_K, "CHUNK_K=%d" % CHUNK_K in src)
    ck("script TAG= is %s" % TAG, "TAG=%s" % TAG in src)
    ck("script SEEDS_BN= is the 4 control seeds this scorer scores",
       'SEEDS_BN="%s"' % " ".join(str(x) for x in SEEDS_BN) in src)
    ck("script SEEDS_GN= is the 8 primary seeds this scorer scores",
       'SEEDS_GN="%s"' % " ".join(str(x) for x in SEEDS_GN) in src)
    ck("the control carries FEWER seeds than the primary -- deliberate, not a typo",
       len(SEEDS_BN) < len(SEEDS_GN))
    ck("the emitter selects a PER-ARM seed list (the 4/4/8/8 rebalance is in force)",
       'ARM_SEEDS="$SEEDS_BN"' in src and 'ARM_SEEDS="$SEEDS_GN"' in src
       and "for S in $ARM_SEEDS" in src)
    ck("script M_NODE= matches the two node arms", "M_NODE=%d" % M_OF_ARM["bn-node"] in src)
    ck("script M_CHUNK= matches the two chunk arms", "M_CHUNK=%d" % M_OF_ARM["bn-ch"] in src)
    ck("script NET_BN= is ResNet18", "NET_BN=ResNet18" in src)
    ck("script NET_GN= is ResNet18_gn", "NET_GN=ResNet18_gn" in src)
    ck("script's arm table is exactly this scorer's four arms",
       all(t in src for t in ("ResNet18:nodewise:bn-node", "ResNet18:chunk${CHUNK_K}:bn-ch",
                              "ResNet18_gn:nodewise:gn-node",
                              "ResNet18_gn:chunk${CHUNK_K}:gn-ch")))
    ck("script writes run names gn1-<arm>-s<seed>", 'RN="${TAG}-${SHORT}-s${S}"' in src)
    ck("script exports PROBE=5-style probing AND PROBE5=1",
       "PROBE=${PROBE_EVERY}" in src and "PROBE5=1" in src and "PROBE_EVERY=5" in src)
    ck("script pins HIER=none and SCHED=none in --export",
       "HIER=none" in src and "SCHED=none" in src)
    ck("script pins --max-time 999:00:00", "--max-time 999:00:00" in src)
    ck("script's BETA_CLIP carries a COLON", "CLIP=-25:-2.3026" in src and ":" in "-25:-2.3026")
    ck("script defaults to a DRY RUN", "SUBMIT=0" in src and "--submit) SUBMIT=1" in src)
    ck("script registers gn1 in c55 BOXES before submitting", 'tag not in c55.BOXES' in src)
    ck("script MEASURES m per arm from the ALLOCATED beta",
       "ALLOCATED beta m=" in src and "m_of(o) != want" in src)
    ck("script asserts the two nets present IDENTICAL named_parameters lists",
       "named[NET_BN] != named[NET_GN]" in src)

    # --- the thresholds, asserted against the script's own registered values
    for label, val in (("NULL_HALF", NULL_HALF), ("DECIDE", DECIDE), ("SE_MAX", SE_MAX)):
        ck("script %s=%s matches this scorer" % (label, val),
           "%s=%s" % (label, val) in src)
    ck("script SIGMA= matches", "SIGMA=%s" % SIGMA in src)
    ck("script SE_D= matches the planning se", "SE_D=%s" % SE_D_PLAN in src)
    ck("script SE_DD= matches the planning se", "SE_DD=%s" % SE_DD_PLAN in src)
    ck("script ANCHOR= matches", "ANCHOR=%s" % ANCHOR in src)
    ck("script ANCHOR_SE= matches", "ANCHOR_SE=%s" % ANCHOR_SE in src)
    ck("SE_MAX is exactly 1.3x the planning se of the PRIMARY, as registered",
       abs(SE_MAX - 1.3 * SE_D_PLAN) < 1e-3)
    ck("the planning se's follow from SIGMA and the ACTUAL per-arm n",
       abs(SE_D_PLAN - SIGMA * math.sqrt(2 / len(SEEDS_GN))) < 0.002
       and abs(SE_D_BN_PLAN - SIGMA * math.sqrt(2 / len(SEEDS_BN))) < 0.002
       and abs(SE_DD_PLAN - math.sqrt(SE_D_PLAN ** 2 + SE_D_BN_PLAN ** 2)) < 0.002)
    ck("the rebalance BUYS power on the primary: se(D_GN) < the old 6/6 se of 0.100",
       SE_D_PLAN < SIGMA * math.sqrt(2 / 6))
    ck("the control still clears the T1 gate at its smaller n",
       ANCHOR / SE_D_BN_PLAN >= RESOLVED_T)
    ck("the TRANSFERS line is about half the pooled anchor",
       abs(DECIDE - ANCHOR / 2) < 0.02)

    # --- the box, registered before any run --------------------------------
    import c55_neff_noise as c55
    ck("gn1 is registered in c55 BOXES", TAG in c55.BOXES)
    ck("gn1's box equals this scorer's (LO, HI)",
       c55.BOXES.get(TAG, (0, 0, ""))[:2] == (LO, HI))
    # NOT fa1's box.  fa1 is (-25, ms=3e-4); this batch's thresholds all come from the
    # ms=1e-4 anchors, which live at -15, so THAT is the box that must match.
    ck("gn1's box is the ms=1e-4 ANCHOR box, so no threshold is a cross-box import",
       (LO, HI) == (-15.0, -2.3026))
    ck("gn1's box is NOT fa1's -- fa1 is a different stepsize entirely",
       c55.BOXES.get(TAG, (0, 0, ""))[:2] != c55.BOXES.get("fa1", (1, 1, ""))[:2])
    ck("sl1 is NOT registered -- the ladder was killed pre-data at cycle-84 review",
       "sl1" not in c55.BOXES)

    # --- the matched-count and same-partition claims ------------------------
    ck("both pairs are matched to <= 1 group",
       abs(M_OF_ARM["bn-ch"] - M_OF_ARM["bn-node"]) <= 1
       and abs(M_OF_ARM["gn-ch"] - M_OF_ARM["gn-node"]) <= 1)
    ck("the BN and GN arms sit at the SAME two counts (the partition is the same object)",
       M_OF_ARM["bn-node"] == M_OF_ARM["gn-node"]
       and M_OF_ARM["bn-ch"] == M_OF_ARM["gn-ch"])
    ck("the two nets differ, so the batch is not four copies of one arm",
       NET_OF_ARM["bn-node"] != NET_OF_ARM["gn-node"])
    ck("each pair varies GRANULARITY at fixed network",
       GRAN_OF_ARM["bn-ch"] != GRAN_OF_ARM["bn-node"]
       and NET_OF_ARM["bn-ch"] == NET_OF_ARM["bn-node"])
    ck("exactly 2 pairs are declared, control first", len(PAIRS) == 2 and PAIRS[0][0] == "D_BN")
    ck("every pair's arms are gn1 arms", all(h in ARMS and l in ARMS for _, h, l, _ in PAIRS))
    ck("2 control arms x n_BN + 2 primary arms x n_GN == NJOBS",
       2 * len(SEEDS_BN) + 2 * len(SEEDS_GN) == NJOBS)
    ck("every arm has a seed list", all(a in SEEDS_OF_ARM for a in ARMS))

    # --- the band, exhaustive and asymmetric-by-design ---------------------
    ck("+0.60 with t 4 -> TRANSFERS", band(0.60, 0.10, 4.0) == "TRANSFERS")
    ck("+0.30 with t 3 -> TRANSFERS (the line is CLOSED on the transfer side)",
       band(0.30, 0.10, 3.0) == "TRANSFERS")
    ck("+0.30 with t 1.5 -> UNDECIDED (the t requirement really binds)",
       band(0.30, 0.20, 1.5) == "UNDECIDED")
    ck("+0.295 -> UNDECIDED (bn1's exact number stays UNDECIDED here too)",
       band(0.295, 0.10, 3.0) == "UNDECIDED")
    ck("+0.20 -> UNDECIDED", band(0.20, 0.10, 2.0) == "UNDECIDED")
    ck("+0.15 powered -> DOES NOT TRANSFER (the null band is CLOSED)",
       band(0.15, 0.11, 1.4) == "DOES NOT TRANSFER")
    ck("+0.15 at se 0.12 -> UNDECIDED: 0.12 now EXCEEDS the tightened SE_MAX",
       band(0.15, 0.12, 1.2) == "UNDECIDED")
    ck("0.00 powered -> DOES NOT TRANSFER", band(0.0, 0.10, 0.0) == "DOES NOT TRANSFER")
    ck("-0.15 powered -> DOES NOT TRANSFER", band(-0.15, 0.10, -1.5) == "DOES NOT TRANSFER")
    ck("-0.40 -> UNDECIDED (a large NEGATIVE is not a null)",
       band(-0.40, 0.10, -4.0) == "UNDECIDED")
    ck("**0.00 with se 0.14 -> UNDECIDED: AN UNDERPOWERED NULL IS NOT 'NO GAP'**",
       band(0.0, 0.14, 0.0) == "UNDECIDED")
    ck("0.00 with se exactly 0.13 -> DOES NOT TRANSFER (the se gate is CLOSED)",
       band(0.0, SE_MAX, 0.0) == "DOES NOT TRANSFER")
    ck("the anchor +0.5805 would read TRANSFERS at the planning se",
       band(ANCHOR, SE_D_PLAN, ANCHOR / SE_D_PLAN) == "TRANSFERS")
    ck("a FULL COLLAPSE is >= 5 sigma on D_GN at the planning se",
       ANCHOR / SE_D_PLAN >= 5.0)
    ck("dD -0.60 with t -4 -> RESOLVED ATTENUATION", dd_band(-0.60, -4.0) == "RESOLVED ATTENUATION")
    ck("dD -0.10 -> AGREE WITHIN RESOLUTION", dd_band(-0.10, -0.7) == "AGREE WITHIN RESOLUTION")
    ck("dD -0.50 with t -1 -> UNRESOLVED (the t requirement binds here too)",
       dd_band(-0.50, -1.0) == "UNRESOLVED")

    # --- T0's thresholds ----------------------------------------------------
    ck("T0.3 floors are final_test > 60 and plateau5 >= 85",
       FINAL_FLOOR == 60.0 and PLATEAU_FLOOR == 85.0)
    ck("T0.4 late-blow-up bar is 2.0 pp", LATE_BLOWUP == 2.0)
    ck("T0.5 occupancy bar is the published 5% gate, BOTH directions", REC_MAX == 0.05)
    ck("an arm voids at >= 2 bad seeds; the batch voids at >= 2 bad arms",
       VOID_SEEDS == 2 and VOID_ARMS == 2)
    _good = {"epochs_done": "100", "epochs_requested": "100", "collapsed": "0",
             "final_test": "92.1", "plateau5": "92.0", "best_test": "92.4"}
    ck("T0 passes a healthy row", t0_of_row(dict(_good))[0])
    ck("T0.2 catches collapsed=1", not t0_of_row(dict(_good, collapsed="1"))[0])
    ck("T0.4 catches the bd7-w-c6 phenotype (best 88.82, final 10.0)",
       not t0_of_row(dict(_good, best_test="88.82", final_test="10.0",
                          plateau5="10.0"))[0])
    ck("T0.3 catches a plateau below the 85 floor",
       not t0_of_row(dict(_good, plateau5="84.9", best_test="85.0"))[0])
    ck("T0.1 catches a truncated run", not t0_of_row(dict(_good, epochs_done="61"))[0])

    # --- the statistics ------------------------------------------------------
    ck("_welch of identical samples is 0", _welch([1., 2., 3.], [1., 2., 3.])[0] == 0.0)
    ck("_welch is antisymmetric",
       abs(_welch([2., 3., 4.], [1., 2., 3.])[2] + _welch([1., 2., 3.], [2., 3., 4.])[2]) < 1e-12)
    ck("_sem of n=1 is nan", math.isnan(_sem([1.0])))
    ck("arm_of_dir parses the hyphenated arm names",
       arm_of_dir("/x/probe_gn-node_gn1_s3") == "gn-node"
       and arm_of_dir("/x/probe_bn-ch_gn1_s0") == "bn-ch")
    ck("every arm name is parseable (no underscores in arm names)",
       all("_" not in a for a in ARMS))

    # --- the honesty guards, asserted against THIS FILE'S OWN TEXT ----------
    ck("scorer states it predates the data entirely", "before the data could exist" in me)
    ck("scorer refuses to read D_GN when T1 fails",
       "will not read D_GN if T0 voids the batch or if T1 fails" in me)
    ck("T1 is declared a GATE and not a result",
       "a GATE (T1), not a result" in me and "GATE" in src)
    ck("the batch-void consequence of a failed control is stated",
       "NOTHING ABOUT GROUPNORM MAY BE CONCLUDED" in me
       and "NOTHING ABOUT GROUPNORM MAY BE CONCLUDED" in src)
    ck("the underpowered-null prohibition is stated in BOTH files",
       "may not be reported as" in me.lower().replace("\n", " ")
       and "MAY NOT be reported as" in src)
    ck("T3.2 NO REBRANDING is carried in BOTH files",
       "NO REBRANDING" in me and "NO REBRANDING" in src)
    ck("both files say the batch removes ZERO singletons",
       "REMOVES ZERO SINGLETONS" in me.upper() and "REMOVES ZERO SINGLETONS" in src.upper())
    ck("the asymmetry (positive clean / null ambiguous) is registered in BOTH files",
       "A NULL IS AMBIGUOUS" in me and "A NULL IS AMBIGUOUS" in src)
    ck("the disambiguator is named in advance in BOTH files",
       "GroupNorm(num_groups==C)" in me.replace(" ", "")
       and "GroupNorm(num_groups == C)" in src)
    ck("the field channel is declared UNAVAILABLE at this cell, in BOTH files",
       "coord_lo is identically 0" in me and "coord_lo is identically 0" in src)
    ck("no cross-box pooling is declared", "NO CROSS-BOX POOLING" in me and "NO CROSS-BOX POOLING" in src)
    ck("the anchor row is declared DESCRIPTIVE and unable to gate",
       "cannot gate" in me and "DESCRIPTIVE" in me)
    ck("RULE 14 (seeds do not pair across batches) is why all four arms are one batch",
       "RULE 14" in me and "RULE 14" in src)
    ck("the scorer refuses to compute a field statistic",
       "will not compute ANY field statistic" in me)

    # --- THE CYCLE-84 REVIEW CORRECTIONS, ASSERTED IN BOTH FILES ------------
    # Each of these is a REGISTERED INTERPRETATION.  They are asserted from the files'
    # own text so that a later edit that softens one fails the selftest rather than
    # passing silently.
    ck("the GroupNorm arithmetic is stated CORRECTLY in both files",
       "32 GROUPS OF C/32 CHANNELS" in me and "32 GROUPS OF C/32 CHANNELS" in src)
    # **THE LITERAL IS SPLIT SO THIS CHECK CANNOT MATCH ITSELF.**  CORRECTIONS 112
    # withdrew fa1's guard H1e for exactly this: a check whose own check-list literal
    # satisfied the condition it was testing.  Built out here rather than commented on.
    _bad = "blocks of " + "32"
    _corr = "THAT WAS " + "BACKWARDS"
    ck("the old backwards wording survives ONLY as a flagged self-correction",
       me.count(_bad) <= 1 and src.count(_bad) <= 1
       and _corr in me and _corr in src)
    ck("the depth-grading of the GN caveat is registered in both files",
       "GRADED BY DEPTH" in me and "GRADED BY DEPTH" in src)
    ck("a null is declared ambiguous between THREE readings, not two",
       "THREE READINGS" in me.upper() and "THREE READINGS" in src.upper())
    ck("the TUNING reading (RULE 11) is named in both files",
       "RULE 11" in me and "RULE 11" in src)
    ck("a null from gn1 ALONE may not be written as a mechanism narrowing",
       "may NOT be written as a" in me.replace("MAY NOT BE WRITTEN AS A",
                                               "may NOT be written as a")
       or "MAY NOT BE WRITTEN AS A" in me.upper())
    ck("the conditional follow-ups gn2a and gn2b are specified in advance, both files",
       "gn2a" in me and "gn2b" in me and "gn2a" in src and "gn2b" in src)
    ck("the bracket axis is alpha0, and WHY (no free box at ms=3e-4) is stated",
       "alpha0=3e-4" in me and "alpha0=3e-4" in src
       and "0.2192" in me and "0.2192" in src)
    ck("T0.6 the commensurability gate is registered in BOTH files",
       "COMMENSURABILITY GATE" in me and "COMMENSURABILITY GATE" in src)
    ck("T0.6's bar is 2.0 pp and it is the same constant in both files",
       abs(COMM_MAX - 2.0) < 1e-9 and "COMM_MAX=2.0" in src)
    ck("a T0.6 firing is declared NOT A NULL, in both files",
       "NOT A NULL" in me.upper() and "NOT A NULL" in src.upper())
    ck("the error-budget-normalised statistic is declared DESCRIPTIVE and gates nothing",
       "GATES NOTHING" in me.upper() and "gates nothing" in me.lower())
    # The guard DISCOVERS the generalisation batches by glob rather than by name -- they
    # were restructured mid-cycle -- so what is asserted here is the glob and the token,
    # not a filename that a later rename would silently invalidate.
    ck("the sequencing interlock exists and DISCOVERS the batches by glob, not by name",
       "GN1_GEN_OK" in src and 'glob.glob(os.path.join(repo, "bin", "c83_gen_*.sh"))' in src
       and "hard-code" in src)
    ck("at least one generalisation batch is actually on disk to be run first",
       bool(glob.glob(os.path.join(REPO, "bin", "c83_gen_*.sh"))))
    ck("the sequencing rationale (rank 1 for four cycles) is stated in the script",
       "rank 1 for" in src and "REFINEMENT" in src)
    ck("box occupancy is measured on the GN arms and never inherited",
       "NEVER INHERITED" in me and "NEVER INHERITED" in src)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_%s" % TAG))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    clip = "%g:%s" % (LO, "-2.3026")
    dirs = sorted(glob.glob(os.path.join(a.root, "probe_*")))
    print("=" * 78)
    print("c84 -- %s: DOES THE PARTITION GAP SURVIVE A CHANGE OF NORMALISER" % TAG)
    print("       AT BYTE-IDENTICAL SINGLETON STRUCTURE?   %d probe dirs" % len(dirs))
    print("box = (%.1f, %.4f)   ms=%s   %d ep   ResNet18 vs ResNet18_gn (GroupNorm(32,C))"
          % (LO, HI, MST, EPOCHS))
    print("**NOT a no-BatchNorm test.  It removes ZERO singletons (T3.2).**")
    print("=" * 78)

    # =====================================================================
    # T0 -- THE GATE, EVALUATED FIRST.  BOX OCCUPANCY BEFORE ANY ACCURACY.
    # =====================================================================
    print("\n--- T0.5  BOX OCCUPANCY, BOTH DIRECTIONS (RULE 13), MEASURED PER ARM")
    print("    **never inherited from the BN anchor -- the GN net's beta trajectory is")
    print("    not the BN net's.**  bar: rec_lo <= %.2f AND rec_hi <= %.2f" % (REC_MAX, REC_MAX))
    occ = {}
    for d in dirs:
        arm = arm_of_dir(d)
        try:
            o = occupancy(d, LO, HI)
        except Exception as e:                                    # pragma: no cover
            print("    %-30s UNREADABLE (%s)" % (os.path.basename(d), e))
            continue
        free = o["rec_lo"] <= REC_MAX and o["rec_hi"] <= REC_MAX
        occ.setdefault(arm, []).append((os.path.basename(d), free, o))
        print("    %-30s %-6s rec_lo %.4f  rec_hi %.4f  (T=%d)"
              % (os.path.basename(d), "free" if free else "BOUND",
                 o["rec_lo"], o["rec_hi"], o["T"]))
        if o["T"] != N_RECORDS:
            print("        !! %d records, not the registered %d -- the instrument did "
                  "not run to length" % (o["T"], N_RECORDS))
    box_ok = {}
    for arm in ARMS:
        seen = occ.get(arm, [])
        box_ok[arm] = bool(seen) and all(f for _, f, _ in seen)
        print("    %-9s %d probe dirs, box-free on %d -> %s"
              % (arm, len(seen), sum(1 for _, f, _ in seen if f),
                 "OK" if box_ok[arm] else ("UNINTERPRETABLE (RULE 5) -- ARM DROPS"
                                           if seen else "NO PROBE -- cannot be gated")))

    print("\n--- T0.1-T0.4  TRAINS-AT-ALL, PER SEED (the per-run table, reported whole)")
    per_arm = {}
    for arm in ARMS:
        rs = rows(a.csv, "%s-%s-s" % (TAG, arm))
        good, bad_seeds = [], []
        for r in rs:
            ok, why = t0_of_row(r)
            print("    %-16s %-6s ep %-4s coll %-2s final %7.3f best %7.3f plateau5 %7.3f  %s"
                  % (r["run"], "PASS" if ok else "FAIL", r.get("epochs_done"),
                     r.get("collapsed"), _f(r, "final_test"), _f(r, "best_test"),
                     _f(r, "plateau5"), "" if ok else "; ".join(why)))
            (good if ok else bad_seeds).append(r)
        if len(rs) != len(SEEDS_OF_ARM[arm]):
            print("    !! %s has %d rows, not the registered %d"
                  % (arm, len(rs), len(SEEDS_OF_ARM[arm])))
        per_arm[arm] = [_f(r, "plateau5") for r in good]
        print("    %-9s %d/%d seeds pass T0.1-T0.4 -> %s"
              % (arm, len(good), len(rs),
                 "VOID" if len(bad_seeds) >= VOID_SEEDS else "OK"))

    void = [arm for arm in ARMS
            if (len(rows(a.csv, "%s-%s-s" % (TAG, arm))) - len(per_arm[arm])) >= VOID_SEEDS
            or not box_ok[arm]]
    print("\n--- T0 VERDICT: %d of %d arms void or dropped (%s)"
          % (len(void), len(ARMS), ", ".join(void) if void else "none"))
    if len(void) >= VOID_ARMS:
        print("    -> **THE WHOLE BATCH IS VOID.**  Registered in advance.")
        print("    T3.1: a VOID GN cell is NOT EVIDENCE OF ANYTHING.  If the GN arms are")
        print("    the void ones, the batch reports 'GroupNorm did not train to a usable")
        print("    plateau at this cell' and STOPS.  It may NOT be reported as an absent")
        print("    gap, and it may NOT be salvaged by adding seeds after seeing which")
        print("    side it fell on.")
        return 1

    # --- the four-arm table, MANDATORY, reported whole (T3.4) ---------------
    print("\n--- THE FOUR ARMS (mandatory: means, sd, n and both occupancy channels)")
    print("    %-9s %-12s %-10s %-8s %8s %8s %5s %10s %10s"
          % ("arm", "network", "granularity", "m", "plateau5", "sd", "n", "rec_lo", "rec_hi"))
    for arm in ARMS:
        v = per_arm[arm]
        o = occ.get(arm, [])
        rl = max([x[2]["rec_lo"] for x in o], default=float("nan"))
        rh = max([x[2]["rec_hi"] for x in o], default=float("nan"))
        print("    %-9s %-12s %-10s %-8d %8.3f %8.3f %5d %10.4f %10.4f"
              % (arm, NET_OF_ARM[arm], GRAN_OF_ARM[arm], M_OF_ARM[arm],
                 statistics.mean(v) if v else float("nan"),
                 statistics.stdev(v) if len(v) > 1 else float("nan"), len(v), rl, rh))
    print("    The ABSOLUTE GN level is itself the headroom check: a GN plateau near")
    print("    91-92 against the T0.3 floor of 85 is the expected 0.5-1.5 pp cost.")

    # =====================================================================
    # T0.6 -- **THE COMMENSURABILITY GATE.**  Added at cycle-84 review.
    # T0.3's floor of 85 does not protect this batch's arithmetic on its own: the
    # whole test compares two differences in PERCENTAGE POINTS on two networks that
    # need not sit at the same accuracy.
    # =====================================================================
    bn_lvl = [x for arm in ("bn-node", "bn-ch") for x in per_arm[arm]]
    gn_lvl = [x for arm in ("gn-node", "gn-ch") for x in per_arm[arm]]
    lvl_bn = statistics.mean(bn_lvl) if bn_lvl else float("nan")
    lvl_gn = statistics.mean(gn_lvl) if gn_lvl else float("nan")
    gap_lvl = lvl_gn - lvl_bn
    bud_bn, bud_gn = 100.0 - lvl_bn, 100.0 - lvl_gn
    print("\n--- T0.6  **THE COMMENSURABILITY GATE.**  D_GN and D_BN are compared in")
    print("    PERCENTAGE POINTS, which only means something if the two nets sit on")
    print("    comparable ERROR BUDGETS.")
    print("    BN level %.3f (budget %.3f pp)   GN level %.3f (budget %.3f pp)"
          % (lvl_bn, bud_bn, lvl_gn, bud_gn))
    print("    difference %+.3f pp;  budget ratio %.2fx;  registered bar |diff| <= %.1f"
          % (gap_lvl, (bud_gn / bud_bn) if bud_bn else float("nan"), COMM_MAX))
    comm_ok = abs(gap_lvl) <= COMM_MAX
    print("    -> **%s**" % ("PASS -- the two nets are in the same accuracy regime"
                             if comm_ok else
                             "FAIL -- GroupNorm sits in a DIFFERENT ACCURACY REGIME"))
    if not comm_ok:
        print("    **NO TRANSFER VERDICT IS ISSUED.**  D_GN and D_BN are not")
        print("    commensurable in pp at a %.2fx difference in error budget: if the gap"
              % ((bud_gn / bud_bn) if bud_bn else float("nan")))
        print("    is even partly MULTIPLICATIVE in the budget, D_GN is distorted by")
        print("    several times its own se, straddling both the +%.2f line and the"
              % DECIDE)
        print("    [-%.2f,+%.2f] null band.  **THIS IS NOT A NULL** and it may not be"
              % (NULL_HALF, NULL_HALF))
        print("    written as one.  Registered in advance at cycle-84 review.")
        return 1

    # =====================================================================
    # T1 -- THE POSITIVE-CONTROL GATE
    # =====================================================================
    D = {}
    for name, hi_arm, lo_arm, what in PAIRS:
        v_hi, v_lo = per_arm[hi_arm], per_arm[lo_arm]
        if len(v_hi) < 2 or len(v_lo) < 2:
            D[name] = (float("nan"),) * 3
            continue
        D[name] = _welch(v_hi, v_lo)

    d_bn, se_bn, t_bn = D["D_BN"]
    print("\n--- T1  **THE POSITIVE-CONTROL GATE.**  %s" % PAIRS[0][3])
    print("    D_BN = plateau5(bn-ch) - plateau5(bn-node) = %+.3f pp  (se %.3f, t %+.2f)"
          % (d_bn, se_bn, t_bn))
    print("    registered gate: D_BN >= +%.2f AND t >= %.1f" % (DECIDE, RESOLVED_T))
    t1_ok = (d_bn >= DECIDE) and (t_bn >= RESOLVED_T)
    print("    -> **%s**" % ("PASS -- the control reproduces D in this batch" if t1_ok
                             else "FAIL -- THE BATCH IS VOID AS A TEST OF GROUPNORM"))
    if not t1_ok:
        print("    D_BN is recorded as a FAILED SIXTH REPLICATION of D at ms=%s, box"
              % MST)
        print("    (%.1f, %.4f).  That is a serious finding in its own right.  But you" % (LO, HI))
        print("    cannot ask whether an effect transfers in a batch where your own")
        print("    control does not show it: **NOTHING ABOUT GROUPNORM MAY BE CONCLUDED.**")
        print("    D_GN is NOT computed.  Registered in advance.")
        return 1

    # =====================================================================
    # T2 -- THE PRIMARY
    # =====================================================================
    d_gn, se_gn, t_gn = D["D_GN"]
    v = band(d_gn, se_gn, t_gn)
    print("\n--- T2  **THE PRIMARY.**  %s" % PAIRS[1][3])
    print("    gn-ch   (m=%d) %.3f +-%.3f (n=%d)"
          % (M_OF_ARM["gn-ch"], statistics.mean(per_arm["gn-ch"]),
             _sem(per_arm["gn-ch"]), len(per_arm["gn-ch"])))
    print("    gn-node (m=%d) %.3f +-%.3f (n=%d)"
          % (M_OF_ARM["gn-node"], statistics.mean(per_arm["gn-node"]),
             _sem(per_arm["gn-node"]), len(per_arm["gn-node"])))
    print("    D_GN = %+.3f pp   (se %.3f, t %+.2f)" % (d_gn, se_gn, t_gn))
    print("    registered: >= +%.2f with t >= %.1f TRANSFERS | [-%.2f,+%.2f] with "
          "se <= %.2f DOES NOT TRANSFER | else UNDECIDED"
          % (DECIDE, RESOLVED_T, NULL_HALF, NULL_HALF, SE_MAX))
    print("    -> **%s**" % v)
    if v == "TRANSFERS":
        print("    **A POSITIVE IS CLEAN AND NEEDS NO FOLLOW-UP: all three readings that")
        print("    make a null ambiguous (batch statistics, per-channel scale invariance,")
        print("    mistuning) can only SUPPRESS the gap, never manufacture it.**")
        print("    The gap survives a change of normaliser at BYTE-IDENTICAL singleton")
        print("    structure -- same 9,610 size-1 groups on the same 41 one-D tensors,")
        print("    same chunk777 count-match.  **A POSITIVE IS CLEAN: there is no")
        print("    alternative reading.**  The prescription may be written about")
        print("    NORMALISATION scalars, not only BatchNorm ones, on this architecture")
        print("    and dataset at this cell.")
    elif v == "DOES NOT TRANSFER":
        print("    **T2 FIRES: THE TAIL MECHANISM AS CURRENTLY WRITTEN IS REFUTED.**")
        print("    A within-batch dissociation at byte-identical partition structure,")
        print("    with the control passing T1 at %+.3f (t %+.2f)." % (d_bn, t_bn))
        print("    CONSEQUENCE, WRITTEN BEFORE THE DATA: the paper's prescriptive")
        print("    sentence narrows from 'do not give normalisation scalars their own")
        print("    step sizes' to 'do not give BATCHNORM scalars their own step sizes',")
        print("    with NO LICENCE TO GENERALISE to normalisation layers at large.")
        print("    **AND THE AMBIGUITY, REGISTERED IN ADVANCE: THIS DOES NOT IDENTIFY")
        print("    THE REPLACEMENT CARRIER.**  nn.GroupNorm(32, C) takes num_groups=32,")
        print("    so it forms 32 GROUPS OF C/32 CHANNELS -- 2, 4, 8 and 16 at ResNet18's")
        print("    widths 64/128/256/512.  Per-channel scale invariance therefore does")
        print("    NOT hold, and the departure is GRADED BY DEPTH: nearly per-channel at")
        print("    layer1, 16-to-a-group at layer4.  A NULL IS AMBIGUOUS BETWEEN THREE")
        print("    READINGS AND THIS BATCH SEPARATES NONE OF THEM:")
        print("      (i)   the carrier is batch statistics;")
        print("      (ii)  the carrier is per-channel scale invariance (depth-graded, so")
        print("            it cannot even be attributed uniformly across the net);")
        print("      (iii) THE TUNING READING (RULE 11) -- this batch runs a DIFFERENT")
        print("            NETWORK at a cell inherited wholesale from BatchNorm, and no")
        print("            arm's argmax is located on either net.  A null at a MISTUNED")
        print("            cell is observationally identical to 'BatchNorm-specific'.")
        print("    **BINDING CONSEQUENCE: a null from gn1 ALONE MAY NOT BE WRITTEN AS A")
        print("    MECHANISM NARROWING.**  What is reported today is 'DOES NOT TRANSFER")
        print("    AT THIS CELL, three unseparated readings'.  The prescriptive sentence")
        print("    stays as written until gn2 has run:")
        print("      gn2a  the TUNING BRACKET -- all four arms at alpha0=3e-4, n=4, 16")
        print("            jobs.  That cell is PROVABLY box-free in BOTH directions")
        print("            (reach 5.0 nats from beta0=-8.1117 gives [-13.112,-3.112], so")
        print("            neither the -15 floor nor the -2.3026 ceiling is reachable).")
        print("            alpha0, not ms, is the bracket axis because at ms=3e-4 there")
        print("            is NO free box for the singleton arm: fa1's nodewise arm reads")
        print("            rec_hi 0.0000/0.1672/0.0023/0.0137/0.2192/0.0600, 3 of 6 seeds")
        print("            OVER the 5%% T0.5 threshold.  BOTH nets are re-run so a null")
        print("            there cannot be confused with 'that cell kills it for anyone'.")
        print("      gn2b  the CARRIER DISAMBIGUATOR, separating (i) from (ii):")
        print("            GroupNorm(num_groups == C).")
    else:
        print("    UNDECIDED is a registered outcome, not a failure of the design.")
        if -NULL_HALF <= d_gn <= NULL_HALF and se_gn > SE_MAX:
            print("    **THIS IS AN UNDERPOWERED NULL: se %.3f > the frozen %.2f.  IT MAY"
                  % (se_gn, SE_MAX))
            print("    NOT BE REPORTED AS 'NO GAP'.**  The threshold was fixed in advance")
            print("    at 1.3x the planning se and is NOT recomputed post hoc.")
        if NULL_HALF < d_gn < DECIDE:
            print("    D_GN sits in the (%.2f, %.2f) corridor -- the band bn1's +0.295"
                  % (NULL_HALF, DECIDE))
            print("    fell into.  It stays UNDECIDED; it is not nudged across.")

    # --- T2b  the secondary -------------------------------------------------
    dd = d_gn - d_bn
    se_dd = math.sqrt(se_gn ** 2 + se_bn ** 2)
    t_dd = dd / se_dd if se_dd > 0 else float("nan")
    print("\n--- T2b  SECONDARY.  dD = D_GN - D_BN = %+.3f  (se %.3f, t %+.2f; planning "
          "se %.3f)" % (dd, se_dd, t_dd, SE_DD_PLAN))
    print("    registered: |dD| <= %.2f AGREE | dD <= -%.2f with t <= -%.1f RESOLVED "
          "ATTENUATION | else UNRESOLVED" % (DD_BAND, DD_BAND, RESOLVED_T))
    ddv = dd_band(dd, t_dd)
    print("    -> **%s**" % ddv)
    if ddv == "RESOLVED ATTENUATION" and v != "DOES NOT TRANSFER":
        print("    Reported as ATTENUATION and **NOT as collapse** -- D_GN did not land")
        print("    in the null band, so the gap is smaller under GroupNorm, not absent.")

    # --- T2c  THE ERROR-BUDGET-NORMALISED STATISTIC.  DESCRIPTIVE, GATES NOTHING.
    # Registered before the data so that a reader can see for themselves whether a
    # verdict is a scale artefact.  It is NOT a second primary: registering two would
    # just move the argmax problem one level up.
    nb = d_bn / bud_bn if bud_bn else float("nan")
    ng = d_gn / bud_gn if bud_gn else float("nan")
    print("\n--- T2c  ERROR-BUDGET-NORMALISED (DESCRIPTIVE ONLY -- GATES NOTHING)")
    print("    D_BN / (100 - level_BN) = %+.3f / %.3f = %+.4f" % (d_bn, bud_bn, nb))
    print("    D_GN / (100 - level_GN) = %+.3f / %.3f = %+.4f" % (d_gn, bud_gn, ng))
    print("    ratio (GN/BN) = %s" % ("%.2f" % (ng / nb) if nb else "n/a"))
    print("    If the pp verdict and this one disagree in DIRECTION, or on which side of")
    print("    the band they fall, SAY SO EXPLICITLY in the write-up.  The registered")
    print("    verdict is still the pp one; this row exists so a scale artefact is")
    print("    visible rather than hidden.")

    # --- the anchor row: DESCRIPTIVE, cross-batch, cannot gate --------------
    print("\n--- ANCHOR ROW.  **DESCRIPTIVE, CROSS-BATCH, CANNOT GATE ANYTHING.**")
    print("    The ms=%s IV pool of D (mm1 +0.485 / pp1 +0.581 / cc1 +0.727) is %+.4f "
          "(se %.4f)." % (MST, ANCHOR, ANCHOR_SE))
    print("    D_BN here = %+.3f (se %.3f).  Cross-batch, so it carries the RULE 14"
          % (d_bn, se_bn))
    print("    floor: seeds do not pair across batches (356 pairs, median |diff| 0.171")
    print("    vs across-seed sd 0.159).  T3.3: NO CROSS-BOX POOLING -- these numbers")
    print("    come from THIS batch at ms=%s, box (%.1f, %.4f), and every prose sentence"
          % (MST, LO, HI))
    print("    about a D carries its cell.")
    for arm in ARMS:
        prev = csv_cell(a.csv, GRAN_OF_ARM[arm], NET_OF_ARM[arm], MST, clip)
        prev = [x for x in prev if x not in per_arm[arm]]
        print("    %-9s here %.3f (n=%d)   prior rows at this exact cell: %s"
              % (arm, statistics.mean(per_arm[arm]) if per_arm[arm] else float("nan"),
                 len(per_arm[arm]),
                 ("%.3f (n=%d)" % (statistics.mean(prev), len(prev))) if prev else "none"))

    # --- what may NOT be said ----------------------------------------------
    print("\n--- T3  WHAT THIS BATCH MAY NOT BE WRITTEN UP AS")
    print("    T3.2 NO REBRANDING: not a no-BatchNorm test, not a norm-free test, not a")
    print("         falsification of the singleton mechanism by removal.  IT REMOVES")
    print("         ZERO SINGLETONS.  Headline: does the gap survive a change of")
    print("         normaliser at FIXED singleton structure?")
    print("    T3.4 NO ARGMAX HARVESTING: the four-arm table above is the result.")
    print("    NO FIELD STATISTIC was computed.  At this cell no coordinate reaches")
    print("         either guard, so coord_lo is identically 0 and the clip-severity")
    print("         channel yields nothing; rec_lo/rec_hi are the T0.5 gate only.")
    print("    STILL OWED: architecture and dataset.  gn1 discharges the NORMALISATION")
    print("         axis of the generalisation debt and nothing else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
