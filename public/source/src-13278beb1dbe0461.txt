#!/usr/bin/env python3
# =============================================================================
# cR1_cpk2_score.py -- THE REGISTERED SCORER FOR BATCH `cpk2`.
#
# COMMITTED BEFORE ANY cpk2 RUN EXISTS (STANDING RULE 21).  Run it UNEDITED
# (RULE 16).  Documented command-line arguments are not edits; nothing below
# this line may be changed once a cpk2 run exists on disk.  If this file turns
# out to be broken it is FROZEN and a NEW file is registered (precedent
# cN1/cN2, CORRECTIONS 149).
#
# -----------------------------------------------------------------------------
# WHY THIS BATCH EXISTS -- THE GAP CORRECTIONS 156 OPENED
# -----------------------------------------------------------------------------
# `cpk1` (CORRECTIONS 146) measured the cut-position curve at fixed m = 2 over
# k in {17,24,31,38,42,45,47,49,52,55,60}, found it single-peaked, and located
# the argmax at k* = 49.  EVERY ONE OF THOSE 39 RUNS WAS 100 EPOCHS.
#
# `cts3` (CORRECTIONS 156) then ran k = 49 and k = 50 to 772 epochs and found
# BOTH verdicts at once: the k49->k50 cliff SURVIVES the horizon (DE 19.3427 pp
# = 26.00 SE) and SHRINKS on it (dD -5.6747 pp = -5.39 SE_dD, as predicted at
# registration).  The decomposition is what forces this batch: d(k49) = +1.0133
# pp but d(k50) = +6.6880 pp.  ARMS EITHER SIDE OF cpk1's PEAK CONVERGE AT VERY
# DIFFERENT RATES.  CORRECTIONS 156.10 therefore records k* = 49 as a
# 100-EPOCH OBJECT and states in terms that a two-point contrast cannot
# relocate an argmax.
#
# cpk2 is that measurement: a short ladder around the peak, at cts3's horizon,
# on FRESH SEEDS, with the 100-epoch control read off epochs 95-99 of the same
# runs exactly as cts3 did.
#
# -----------------------------------------------------------------------------
# THE DESIGN.  6 arms x 3 seeds = 18 jobs, ONE submission, 772 epochs.
# -----------------------------------------------------------------------------
#   SWEEP ARMS (all m = 2, contiguous split of named_parameters()):
#       k45 = [45,17]   k47 = [47,15]   k49 = [49,13]
#       k50 = [50,12]   k52 = [52,10]
#   FLOOR ANCHOR (m = 1):
#       k01 = scalar
#   SEEDS {3,4,5}.  HORIZON 772.  Clamp held at the campaign standard
#   BETA_CLIP = -15:-2.3026.  Everything else identical to cpk1/cts1/cts3.
#
# WHY THIS GRID AND NOT THE ONE PROPOSED AT CORRECTIONS 156's CLOSE.  156
# proposed k in {45,47,49,51,53}.  Two of those five arms are replaced, and the
# reason for each is measured from data ALREADY in the corpus, not argued:
#
#   * k = 53 IS REPLACED BY k = 52 BECAUSE k = 53 FAILS THIS FILE'S OWN FLOOR
#     GATE BEFORE IT IS RUN.  `scl1` measured [53,9] at plateau5 = 22.0753 pp
#     at 100 epochs.  The pooled same-cell m=1 scalar baseline is 22.749176 pp
#     (17 runs, 6 batches -- re-derived by --selftest).  k = 53 sits 0.6739 pp
#     BELOW the scalar floor, i.e. it has already saturated to the m=1
#     baseline.  Gate R-FLOOR below would score it NOT-INFORMATIVE, so running
#     it at 772 epochs would spend ~15.6 GPU-hours to re-measure a floor.
#     (Same for k = 54, 22.2133, and k = 55, 23.3720 -- 0.6228 pp above the
#     baseline, inside the 1.1488 pp bar.)  k = 52 is the LAST non-saturated
#     cut position on the right flank: 38.1667 pp pooled over three batches,
#     15.5928 pp = 27.1 SE_FLOOR clear of the floor.
#
#   * k = 52 IS ALSO THE ARM MOST LIKELY TO MOVE THE ARGMAX RIGHT, WHICH IS THE
#     WHOLE POINT.  cts3's mechanism is that the arms furthest from convergence
#     gain the most.  k = 52's terminal 20-epoch OLS slope at epoch 100 is
#     +0.06072 pp/epoch pooled over cpk1 (+0.05858), cts1 (+0.06600) and scl1
#     (+0.05757) -- the LARGEST of any arm anywhere near the peak, larger than
#     the k = 50 arm cts3 measured (+0.04293) and 140x k = 49's.  If any arm
#     can overtake k = 49 once the budget is bought, it is this one.
#
#   * k = 51 IS REPLACED BY k = 50 BECAUSE k = 51 IS A NEAR-DUPLICATE OF AN ARM
#     ALREADY MEASURED AT 772.  cts1 ran [51,11]: plateau5 30.5680 pp, terminal
#     slope +0.04670.  cts1 also ran [50,12]: 30.3347 pp, slope +0.04620.  The
#     two differ by 0.2333 pp = 0.31 SE_ARM_DIFF and by 0.0005 pp/epoch of
#     slope.  k = 50 at 772 epochs is measured (cts3, seeds {0,1,2}); k = 51
#     would buy a copy of it.  Running k = 50 on seeds {3,4,5} instead buys a
#     FRESH-SEED replication of cts3's own headline arm and gives this batch
#     its cliff premise (R1) in batch.
#
#   * THE SCALAR ANCHOR IS ADDED, AGAINST cts3's "NO ANCHORS" STANCE, BECAUSE
#     THE FLOOR GATE NEEDS A BASELINE AT THE HORIZON IT IS APPLIED AT.  Every
#     m=1 scalar row in this cell anywhere in the corpus is 100 epochs.
#     Applying a 100-epoch floor to a 772-epoch arm would assume the floor does
#     not move with budget -- exactly the class of unmeasured carry-over that
#     cost this thread 5.67 pp at CORRECTIONS 156.  cpk1's scalar arm is
#     converged at 100 (slope -0.00053 +- 0.00464), so the prediction is that
#     it barely moves; that makes this a cheap, refutable check rather than an
#     open-ended addition.  COST OF THE ANCHOR: 3 runs, ~15.6 GPU-hours.
#     NO CAPTURE.  There is no layerwise anchor in this batch, so CAPTURE is
#     not computable, is not computed here, and may not be computed from cpk2
#     afterwards.  The scalar arm is a FLOOR REFERENCE only.
#
# THE PRICE OF FRESH SEEDS, STATED.  cpk1, cts1, cts2, scl1 and cts3 all use
# seeds {0,1,2}; CORRECTIONS 152 found six of scl1's twelve runs were
# exact-configuration RE-EXECUTIONS and had to withdraw an "independent
# submission" claim.  cpk2 uses {3,4,5} and is therefore the first batch in
# this thread that SAMPLES the seed nuisance rather than re-executing it.  What
# that costs:
#   * k45 / k47 / k49 must be RE-RUN at 100 epochs rather than reused from
#     cpk1.  That cost is zero here only because the 100-epoch readout rides
#     inside the 772-epoch runs.
#   * cpk2's 100-epoch control is NOT poolable with cpk1's numbers.  Batch is
#     the unit of replication (F(62,85) = 5.47, p 6.9e-13), so every comparison
#     to cpk1 is CROSS-BATCH and descriptive.  Every gate below is WITHIN cpk2.
#   * ONE DISCLOSED OVERLAP.  `c100b-1e6-scal-s3` (22.504) and
#     `c100b-1e6-scal-s4` (23.426) are exact-configuration 100-epoch runs of
#     THIS batch's k01 arm at seeds 3 and 4.  cpk2's k01 100-epoch readout is
#     therefore a re-execution at two of its three seeds.  This is recorded as
#     a DESCRIPTIVE cross-check, never as a gate and never as independence.
#
# -----------------------------------------------------------------------------
# THE PRE-REGISTERED POINT PREDICTION.  WRITTEN DOWN BEFORE ANY RUN EXISTS.
# -----------------------------------------------------------------------------
# MODEL, stated with its weakness first.  cts3 gives exactly two (slope@100,
# gain from 100 to 772) pairs:
#       k49  slope +0.00239  gain +1.0134
#       k50  slope +0.03966  gain +6.6880
# A straight line through two points has ZERO residual degrees of freedom.  It
# is therefore NOT a tested model and is not claimed to be one; it is the
# arithmetic that turns each arm's own measured 100-epoch slope into a number
# that the batch can refute.  --selftest re-derives:
#       gain(k) = GAIN_A + GAIN_B * slope100(k),  GAIN_A 0.649507 pp,
#       GAIN_B 152.2565 epochs.
#
# INPUT SLOPES, each MEASURED on the .out files of the batches named, as the
# OLS slope of test accuracy on epoch index over epochs 80-99 (the same
# 20-epoch terminal window this file uses at score time):
#       k01  -0.00053   cpk1
#       k45  -0.00042   cpk1
#       k47  +0.00496   cpk1
#       k49  -0.00043   mean of cpk1 +0.00005, cts1 -0.00374, cts3 +0.00239
#       k50  +0.04293   mean of cts1 +0.04620, cts3 +0.03966
#       k52  +0.06072   mean of cpk1 +0.05858, cts1 +0.06600, scl1 +0.05757
# INPUT 100-EPOCH LEVELS.  The mean of the per-BATCH mean plateau5 at 100
# epochs over every batch in the corpus that ran that granularity in this cell
# -- batch, not run, is the unit of replication (F(62,85) = 5.47, p 6.9e-13) --
# PLUS, for k49 and k50, cts3's own in-run 100-epoch readout, which is a bona
# fide 100-epoch measurement of the identical configuration but carries
# epochs_done=772 in the CSV and so is invisible to a corpus query.
# --selftest re-derives every one of these six.  NOTE that LEVEL100["k01"]
# (22.722778, a mean of 6 batch means) is NOT the same estimator as
# SCALAR_BASELINE (22.749176, a mean of 17 runs, which the FLOOR bar uses);
# they differ by 0.0264 pp because batch `c100` contributed only 2 runs.  Both
# are stated rather than one being quietly reused for the other:
#       k01  22.722778  (c100, c100b, cbl1, cpk1, cts1, hb1)
#       k45  42.345667  (cbl1, cpk1)
#       k47  45.688000  (cpk1)
#       k49  55.338325  (cpk1, cts1, cts2, cts3)
#       k50  30.297333  (cts1, cts2, cts3)
#       k52  38.166667  (cpk1, cts1, scl1)
#
# THE PREDICTION AT E = 772, in pp of plateau5:
#       k01  23.292     k45  42.931     k47  47.093
#       k49  55.922     k50  37.483     k52  48.061
# => PREDICTED BRANCH: **k* UNMOVED, k* = 49**, and the predicted margin over
#    the runner-up (k = 52) is 7.8613 pp = 10.50 SE_ARM_DIFF.
# WHAT WOULD REFUTE IT, stated as a number rather than a mood:
#   * k = 52 takes the argmax only if it gains >= 17.756 pp, i.e. 1.795x its
#     predicted gain of 9.894 pp.
#   * k = 47 takes the argmax only if it gains >= 10.234 pp, i.e. 7.29x its
#     predicted gain of 1.405 pp.
# TWO INDEPENDENT SANITY CHECKS OF THE SAME ARITHMETIC, both re-derived by
# --selftest: the model returns k49 at 55.922 against cts3's measured 56.2487
# (-0.327 pp) and k50 at 37.483 against cts3's measured 36.9060 (+0.577 pp).
# Those two are NOT free -- the model was fitted on cts3's GAINS, not on its
# levels, and the levels come from a four-batch and three-batch pooled mean.
#
# THE DIRECTIONAL REASONING, so the forecast is not a bare number.  cts3's
# mechanism is that arms not converged at 100 gain most from the extra 672
# epochs.  The arms LEFT of the peak are converged at 100: k45 -0.00042 and
# k49 -0.00043 are INSIDE the +-0.00426 slope at which CORRECTIONS 150
# declared an arm converged, and k47 +0.00496 is marginally OUTSIDE it -- by
# 0.00070 pp/epoch, i.e. 1.16x that bar and 0.20x CONV_BAR.  That 0.0007 is
# recorded rather than rounded away: it is the ONLY left-flank arm with any
# room at all, and the prediction gives it the largest left-flank gain
# (+1.405 pp) because of it.  The arms RIGHT of the peak are not converged on
# any bar (k50 +0.04293, k52 +0.06072, 1.8x and 2.5x CONV_BAR).  So the budget
# moves the RIGHT flank up and leaves the LEFT flank essentially where it is:
#   * k* CANNOT MOVE LEFT on this mechanism -- the left arms have nothing to
#     gain.  A MOVES-LEFT verdict would REFUTE the mechanism, not confirm it.
#   * k* CAN ONLY MOVE RIGHT, and only if the right flank's larger gains are
#     big enough to close a 17.8 pp gap.  On the calibrated arithmetic they are
#     not, by a factor of 1.8.
#   * The peak therefore stays at 49 and the curve becomes MORE asymmetric, not
#     less: the right flank rises toward the peak while the left flank does not.
#
# -----------------------------------------------------------------------------
# THE GATES.  They fire IN ORDER; the first failure decides.
# -----------------------------------------------------------------------------
# G0 PROVENANCE.  18 runs, 6 cells x 3 seeds, 18 distinct job ids, E/E epochs
#    each, no repeated flag on any ARGS line, every run's NAME agreeing with
#    its own ARGS, and BETA_CLIP / AUGMENT / PROBE audited from the ENV line
#    because they cannot ride the ARGS line.
# R2 DIVERGENCE.  Any run with fewer than E epoch lines, a non-finite metric,
#    or plateau5 at E <= 5.00 pp is DEAD -> UNRESOLVED-DIVERGED.  Registered so
#    that "the long runs fell over" can never be scored as "the peak moved".
# R3 NOISE.  Any cell SD at E above 3 * SIGMA_W = 2.7518 pp ->
#    UNRESOLVED-NOISY.
# R1 PREMISE -- THE CLIFF, IN BATCH, AT 100 EPOCHS.  D100 = M(k49, ep95-99) -
#    M(k50, ep95-99) must be >= 12.4170 pp, half of cts1's own in-batch cliff.
#    Otherwise UNRESOLVED-PREMISE.  cts1's, cts2's or cts3's arms may NOT be
#    spliced in to rescue it.
# R-CTRL PREMISE -- THE PEAK, IN BATCH, AT 100 EPOCHS.  This is the gate that
#    makes any movement attributable to the BUDGET rather than to the seeds.
#    At the 100-epoch readout of these same runs, the argmax over the five
#    sweep arms must be k = 49, and it must beat the second-best sweep arm by
#    more than ARGMAX_BAR = 1.4979 pp.  If it does not, cpk1's peak has not
#    reproduced on fresh seeds and NO relocation may be attributed to the
#    horizon -> UNRESOLVED-CONTROL.
# R-FLOOR SATURATION, per arm, at BOTH horizons.  An arm is NOT-INFORMATIVE if
#    its mean sits within FLOOR_BAR = 1.1488 pp (= 2 * SE_FLOOR) of the pooled
#    same-cell m=1 scalar baseline.  AT E the baseline used is THIS BATCH's OWN
#    k01 arm at E; at 100 it is this batch's own k01 arm at 100.  A
#    NOT-INFORMATIVE arm is EXCLUDED from the argmax set.  If k49 itself is
#    NOT-INFORMATIVE at E -> UNRESOLVED-SATURATED.  (CORRECTIONS 152 found
#    scl1's control unidentified because its arms had saturated at the scalar
#    baseline; this gate exists so that cannot recur silently.)
# R-CONV HORIZON EFFECTIVENESS, per arm.  Two bars, both printed for every arm:
#    CONV_BAR      0.024235 pp/epoch -- half the k50 arm's own measured
#                  100-epoch slope, the identical bar cts3 registered as its
#                  R4.  An arm whose terminal 20-epoch OLS slope at E exceeds
#                  this is NOT CONVERGED.
#    TIGHT_BAR     0.00426 pp/epoch -- the s* that DEFINED E = 772 (the slope
#                  at which CORRECTIONS 150 declared the k49 arm converged).
#                  An arm inside this is TIGHT-CONVERGED.  cts3's own arms
#                  landed at +0.00287 (k49) and -0.00199 (k50), so both cleared
#                  it; a 20-epoch slope has a seed SD of about 0.005, so the
#                  gap between the two bars is the honest region.
#    If ANY arm fails CONV_BAR the batch is UNRESOLVED-NOT-CONVERGED: the
#    772-epoch argmax is still reported, but it MAY NOT be called the converged
#    argmax and MUST NOT be reported as final.
#
# THE VERDICT ON THE ARGMAX.  Over the INFORMATIVE sweep arms at E, with
# best = the largest arm mean and second = the next largest:
#    PEAK-DISSOLVES   best - second <= ARGMAX_BAR (1.4979 pp = 2 SE_ARM_DIFF).
#                     No argmax may be named at this horizon.
#    k*-UNMOVED       best is k = 49, by more than ARGMAX_BAR.
#    k*-MOVES-LEFT    best is k < 49, by more than ARGMAX_BAR.
#    k*-MOVES-RIGHT   best is k > 49, by more than ARGMAX_BAR.
# Reported in EVERY branch, so a movement driven by the winner decaying is
# never retold as one driven by a rival catching up: the per-arm gain
# d(arm) = M(arm,E) - M(arm,100), each arm's measured gain against its
# PREDICTED gain, and the full 100-epoch and 772-epoch curves side by side.
#
# MAY NOT CLAIM.  Anything about m (every sweep arm is m = 2 and the anchor is
# an anchor, not a rung).  Any cut position off this five-point grid -- in
# particular cpk1's own k = 17/24/31/38/42/55/60 are NOT re-run here and their
# 772-epoch behaviour stays unmeasured, so cpk2 cannot restore
# single-peakedness over cpk1's grid and does not try.  Note that the cpk2
# grid CONTAINS k = 50, which cpk1's grid did not, and the 100-epoch curve is
# already non-monotone across 49 -> 50 -> 52 in cts1's own data; that shape
# fact predates cpk2 and is not a cpk2 finding.  Any horizon but 100 and 772.
# Any CAPTURE quantity.  The asymptote.  The released floor at long budget.
# Any other dataset, network, optimiser pair, meta-stepsize, alpha0, batch
# size, box or hierarchical operator.  Which TENSOR carries any step (cts1 and
# scl1 own that).  And the MECHANISM of any gain: this file measures gains and
# compares them to a prediction; it explains nothing.
# =============================================================================

from __future__ import annotations

import argparse
import collections
import csv
import glob
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV = os.path.join(REPO, "results", "all_runs.csv")

# ---- FROZEN REGISTRATION ----------------------------------------------------
PREFIX = "cpk2-"
NET, DSET = "ResNet18_c100", "CIFAR100"
MS, ALPHA0, AUG = "1e-3", "1e-6", "1"
EPOCHS = 772                     # E, cts3's registered horizon, kept identical
CONTROL_EPOCHS = 100             # the in-batch, in-run control readout
BATCH, GAMMA = 100, "1"
BASE_ALG, META_ALG = "SGDm", "Lion"
SEEDS = (3, 4, 5)                # FRESH.  cpk1/cts1/cts2/scl1/cts3 all use 0,1,2
N_TENSORS = 62
CLIP_C = "-15:-2.3026"           # the campaign standard; the only level run
KGRID = (45, 47, 49, 50, 52)     # the m=2 sweep
ANCHOR = "k01"                   # m=1 scalar, the FLOOR reference only
SWEEP_ARMS = tuple("k%02d" % k for k in KGRID)
ARMS = (ANCHOR,) + SWEEP_ARMS
N_RUNS = len(ARMS) * len(SEEDS)  # 18
SLOPE_WINDOW = 20                # epochs in the terminal OLS slope window

# ---- the noise floor, RE-DERIVED FROM THE LIVE CORPUS AT REGISTRATION TIME.
# NOT copied from cts3: CORRECTIONS 156.9 found cts3's frozen 0.9111 had drifted
# to 0.9173 because scl1's rows landed 2.5 h after cts3 was registered.  The
# triple below was re-derived from results/all_runs.csv as committed at this
# file's registration commit; --selftest re-derives it and prints the live
# value.  EVERY BAR IN THIS FILE DERIVES FROM THE FROZEN LITERAL, never from
# the live corpus, so no verdict here can depend on the order in which
# unrelated batches are ingested.
SIGMA_W = 0.917280
SIGMA_DF = 58
SIGMA_CELLS = 29
SIGMA_MEMBERS = 87

# ---- the scalar floor, RE-DERIVED FROM THE LIVE CORPUS AT REGISTRATION TIME.
# Pooled mean plateau5 of every m=1 `scalar` row in the cts1/cpk1 cell at 100
# epochs.  Used ONLY to set FLOOR_BAR and to justify the grid; the gate itself
# runs against THIS BATCH's own k01 arm at the matching horizon.
SCALAR_BASELINE = 22.749176
SCALAR_N = 17
SCALAR_BATCHES = 6

# ---- cts1's own in-batch cliff.  Enters ONLY through PREMISE_BAR. -----------
CTS1_PREFIX = "cts1-"
CTS1_K49 = 55.168667
CTS1_K50 = 30.334667
CTS1_CLIFF = 24.834000

# ---- cts3's two measured arms.  They calibrate the prediction and set the
# convergence bars.  Frozen from CORRECTIONS 156.5 and re-derivable from the
# cts3 .out files with this file's own tail5()/ols_slope().
CTS3_P100 = {"k49": 55.2353, "k50": 30.2180}
CTS3_PE = {"k49": 56.2487, "k50": 36.9060}
CTS3_S100 = {"k49": 0.00239, "k50": 0.03966}
CTS3_SE_ = {"k49": 0.00287, "k50": -0.00199}
SLOPE_K50_AT_100 = 0.04847       # cts2's measurement, which set cts3's R4 bar
CONVERGED_SLOPE = 0.00426        # s*, the slope that DEFINED E = 772

# ---- the per-arm inputs to the prediction.  MEASURED, see the header. -------
SLOPE100 = {"k01": -0.00053, "k45": -0.00042, "k47": 0.00496,
            "k49": -0.00043, "k50": 0.04293, "k52": 0.06072}
LEVEL100 = {"k01": 22.722778, "k45": 42.345667, "k47": 45.688000,
            "k49": 55.338325, "k50": 30.297333, "k52": 38.166667}
PRED_ARGMAX = "k49"
PRED_MARGIN = 7.861173           # pp, over the predicted runner-up k52
PRED_BRANCH = "k*-UNMOVED"

# ---- the SE arithmetic.  Each contrast gets the SE of ITS OWN combination. ---
def se_of_combo(coefs, sigma=SIGMA_W, n=len(SEEDS)):
    return sigma * math.sqrt(sum(c * c for c in coefs) / float(n))


SE_ARM_DIFF = se_of_combo((1, -1))            # any two-arm difference
SE_DD = se_of_combo((1, -1, -1, 1))           # dD, treated as UNPAIRED
# the floor contrast pools a 3-seed arm mean against the 17-run scalar corpus
SE_FLOOR = SIGMA_W * math.sqrt(1.0 / len(SEEDS) + 1.0 / SCALAR_N)

# ---- BARS.  --selftest quotes every one in SE units. ------------------------
PREMISE_BAR = 12.4170     # R1: half of cts1's own in-batch cliff
ARGMAX_BAR = 1.497910     # the argmax verdict: 2 * SE_ARM_DIFF
FLOOR_BAR = 1.148848      # R-FLOOR: 2 * SE_FLOOR
NOISY_BAR = 2.751840      # R3: 3 * SIGMA_W
DEAD_BAR = 5.00           # R2
CONV_BAR = 0.024235       # R-CONV: half SLOPE_K50_AT_100; cts3's own R4 bar
TIGHT_BAR = 0.00426       # R-CONV: s*, the slope that defined E

EP_RE = re.compile(r"Epoch\s+(\d+).*?Test Accuracy:\s*([0-9.]+)", re.I)
EPTR_RE = re.compile(r"Epoch\s+(\d+).*?Train Accuracy:\s*([0-9.]+)", re.I)
ARGS_RE = re.compile(r"^\s*ARGS:\s*(.*)$")
ENV_RE = re.compile(r"^\s*ENV:\s*(.*)$")
NAME_RE = re.compile(r"^cpk2-(k01|k45|k47|k49|k50|k52)-s([345])-(\d+)\.out$")


# =============================================================================
# the registered grid, spelled out so nothing is reconstructed at score time
# =============================================================================
def kspec(k):
    return "[%d,%d]" % (k, N_TENSORS - k)


def arm_k(arm):
    return int(arm[1:])


def arm_spec(arm):
    return "scalar" if arm == ANCHOR else kspec(arm_k(arm))


# =============================================================================
# the corpus.  Used ONLY by --selftest, to re-derive the noise floor, the
# scalar baseline, cts1's cliff, the RULE 21 premise and the horizon census.
# score() never opens the CSV.
# =============================================================================
def _csv_rows(path=CSV):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if (r.get("superseded") or "0") != "1"]


def _batch_of(run):
    return re.split(r"[-_]", str(run))[0]


def _in_cell(r, epochs=CONTROL_EPOCHS):
    """The cts1/cpk1 cell, EXACTLY, at the stated horizon."""
    return (r["network"] == NET and r["dataset"] == DSET
            and r["base"] == BASE_ALG and r["meta"] == META_ALG
            and r["meta_stepsize"] == MS and r["alpha0"] == ALPHA0
            and r["gamma"] == GAMMA and r["augment"] == AUG
            and r["beta_clip"] == CLIP_C and not (r.get("hier") or "").strip()
            and r["batch_size"] == str(BATCH)
            and r["epochs_done"] == str(epochs)
            and r["collapsed"] == "0")


M2_RE = re.compile(r"^\[\d+,\d+\]$")


def noise_floor_from_csv(path=CSV):
    """Pooled WITHIN (batch x granularity) SD of plateau5, m = 2 arms only."""
    cells = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not _in_cell(r) or not M2_RE.match(str(r["granularity"] or "")):
            continue
        try:
            cells[(_batch_of(r["run"]), r["granularity"])].append(float(r["plateau5"]))
        except (TypeError, ValueError):
            pass
    ss, df, nc, nm = 0.0, 0, 0, 0
    for v in cells.values():
        nm += len(v)
        if len(v) >= 2:
            m = statistics.mean(v)
            ss += sum((x - m) ** 2 for x in v)
            df += len(v) - 1
            nc += 1
    return (math.sqrt(ss / df) if df else float("nan")), df, nc, nm


def scalar_baseline_from_csv(path=CSV):
    """Pooled mean plateau5 of every m=1 scalar row in the cell at 100 epochs."""
    per_batch = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not _in_cell(r) or str(r["granularity"] or "") != "scalar":
            continue
        try:
            per_batch[_batch_of(r["run"])].append(float(r["plateau5"]))
        except (TypeError, ValueError):
            pass
    vals = [x for v in per_batch.values() for x in v]
    if not vals:
        return None
    return statistics.mean(vals), len(vals), len(per_batch), per_batch


def arm_levels_from_csv(path=CSV):
    """Pooled cell mean of plateau5 per granularity at 100 epochs, per batch."""
    g = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not _in_cell(r):
            continue
        try:
            g[(str(r["granularity"] or ""), _batch_of(r["run"]))].append(
                float(r["plateau5"]))
        except (TypeError, ValueError):
            pass
    return g


def level100_from_corpus(path=CSV):
    """RE-DERIVE the frozen LEVEL100 table.

    For each cpk2 arm, the mean of the per-BATCH mean plateau5 at 100 epochs
    (batch, not run, is the unit of replication: F(62,85) = 5.47, p 6.9e-13),
    over every batch in the corpus that ran that granularity in this cell --
    PLUS, for k49 and k50 only, cts3's own in-run 100-epoch readout, which is
    a bona fide 100-epoch measurement of the identical configuration but does
    NOT appear in the CSV as a 100-epoch row (cts3's rows carry 772).  That
    inclusion is the one place this table is not purely a CSV query, so it is
    made explicit here rather than left to be discovered.
    """
    g = arm_levels_from_csv(path)
    out = {}
    for a in ARMS:
        spec = arm_spec(a)
        bm = {kk[1]: statistics.mean(v) for kk, v in g.items() if kk[0] == spec}
        if a in CTS3_P100:
            bm["cts3"] = CTS3_P100[a]
        if not bm:
            continue
        out[a] = (statistics.mean(bm.values()), sorted(bm))
    return out


def cts1_cliff_from_csv(path=CSV):
    g = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not str(r.get("run") or "").startswith(CTS1_PREFIX):
            continue
        try:
            g[r["granularity"]].append(float(r["plateau5"]))
        except (TypeError, ValueError):
            pass
    a, b = kspec(49), kspec(50)
    if a not in g or b not in g:
        return None
    return (statistics.mean(g[a]), statistics.mean(g[b]),
            statistics.mean(g[a]) - statistics.mean(g[b]))


def count_cpk2_rows(path=CSV):
    return sum(1 for r in _csv_rows(path)
               if str(r.get("run") or "").startswith(PREFIX))


def horizons_on_net(path=CSV):
    """Every distinct epochs_done above the control horizon on THIS network."""
    out = set()
    for r in _csv_rows(path):
        if r.get("network") != NET:
            continue
        e = str(r.get("epochs_done") or "")
        if e.isdigit() and int(e) > CONTROL_EPOCHS:
            out.add(int(e))
    return sorted(out)


# =============================================================================
# the prediction, re-derived by --selftest from cts3's two measured arms
# =============================================================================
def calibrate():
    """gain = A + B*slope100, the line through cts3's two (slope, gain) points."""
    d49 = CTS3_PE["k49"] - CTS3_P100["k49"]
    d50 = CTS3_PE["k50"] - CTS3_P100["k50"]
    B = (d50 - d49) / (CTS3_S100["k50"] - CTS3_S100["k49"])
    A = d49 - B * CTS3_S100["k49"]
    return A, B


def predicted_gain(arm, A=None, B=None):
    if A is None:
        A, B = calibrate()
    return A + B * SLOPE100[arm]


def predicted_level(arm, A=None, B=None):
    return LEVEL100[arm] + predicted_gain(arm, A, B)


# =============================================================================
# reading the runs.  Nothing is taken from the launcher's header.
# =============================================================================
def _tokens(line):
    import shlex
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def parse_args_line(line):
    """{flag: value} plus the list of flags that appear more than once."""
    toks = _tokens(line)
    seen, order = {}, []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:]
            v = ""
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                v = toks[i + 1]
                i += 1
            order.append(k)
            seen[k] = v
        i += 1
    dup = sorted({k for k in order if order.count(k) > 1})
    return seen, dup


def parse_env_line(line):
    env = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
    return env


def read_out(path):
    args = env = None
    test, train = {}, {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if args is None:
                    m = ARGS_RE.match(line)
                    if m:
                        args = m.group(1).strip()
                        continue
                if env is None:
                    m = ENV_RE.match(line)
                    if m:
                        env = m.group(1).strip()
                        continue
                m = EP_RE.search(line)
                if m:
                    test[int(m.group(1))] = float(m.group(2))
                m = EPTR_RE.search(line)
                if m:
                    train[int(m.group(1))] = float(m.group(2))
    except IOError:
        return None
    if args is None:
        return None
    eff, dup = parse_args_line(args)
    return {"args": eff, "dup": dup, "argsline": args,
            "env": parse_env_line(env or ""), "envline": env or "",
            "test": test, "train": train, "path": path}


def tail5(series, budget, w=5):
    """PRIMARY WINDOW, and the ONLY one this file scores.

    Mean of the last 5 epoch lines BEFORE `budget`.  train.py prints 0-indexed
    epochs, so budget=772 reads epochs 767..771 and budget=100 reads 95..99.
    WHY THIS AND NOTHING ELSE, at BOTH horizons: the CSV `plateau` column is
    BANNED as a primary (STANDING RULE); `best_test` is a maximum over the run
    and is not a plateau at all; and using the IDENTICAL 5-epoch tail estimator
    at 100 and at 772 is what makes the within-run gain d = M(E) - M(100) free
    of any estimator difference.  At 772 epochs a 5-epoch tail is 0.65% of the
    run, and R-CONV certifies separately that the curve is flat there, so the
    window is not doing any smoothing work the gate has not checked."""
    v = [series[e] for e in range(budget - w, budget) if e in series]
    return sum(v) / len(v) if len(v) == w else None


def ols_slope(series, budget, w=SLOPE_WINDOW):
    """OLS slope of the series on the epoch index over the w epochs ending at
    budget-1, in pp/epoch.  None if any epoch in the window is missing."""
    xs = list(range(budget - w, budget))
    if any(e not in series for e in xs):
        return None
    ys = [series[e] for e in xs]
    mx, my = sum(xs) / float(w), sum(ys) / float(w)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def norm_num(s):
    try:
        return "%g" % float(s)
    except (TypeError, ValueError):
        return None


def collect(runsdir):
    """Every cpk2 .out file, with its cell MEASURED from its own ARGS and ENV."""
    recs = []
    for p in sorted(glob.glob(os.path.join(runsdir, PREFIX + "*.out"))):
        base = os.path.basename(p)
        m = NAME_RE.match(base)
        r = read_out(p)
        if r is None:
            recs.append({"path": p, "base": base, "bad": "no ARGS line",
                         "job_id": None, "meas_arm": None, "dup": [],
                         "name_arm": None, "name_seed": None, "n_ep": 0,
                         "p5_E": None, "p5_100": None, "t5_E": None,
                         "t5_100": None, "slope_E": None, "slope_100": None,
                         "env": {}, "args": {}, "clip": None, "seed": None})
            continue
        r["base"] = base
        r["job_id"] = m.group(3) if m else None
        r["name_arm"] = m.group(1) if m else None
        r["name_seed"] = int(m.group(2)) if m else None
        spec = r["args"].get("stepsize-groups")
        r["meas_arm"] = None
        for a in ARMS:
            if spec == arm_spec(a):
                r["meas_arm"] = a
        r["clip"] = r["env"].get("BETA_CLIP")
        try:
            r["seed"] = int(r["args"].get("seed"))
        except (TypeError, ValueError):
            r["seed"] = None
        r["p5_E"] = tail5(r["test"], EPOCHS)
        r["t5_E"] = tail5(r["train"], EPOCHS)
        r["p5_100"] = tail5(r["test"], CONTROL_EPOCHS)
        r["t5_100"] = tail5(r["train"], CONTROL_EPOCHS)
        r["slope_E"] = ols_slope(r["test"], EPOCHS)
        r["slope_100"] = ols_slope(r["test"], CONTROL_EPOCHS)
        r["n_ep"] = len(r["test"])
        recs.append(r)
    return recs


# =============================================================================
def fmt(x, w=9, p=4):
    return (" " * w) if x is None else ("%*.*f" % (w, p, x))


# =============================================================================
# --selftest.  Everything the registration asserts is RE-DERIVED here, from
# the corpus and from arithmetic, and checked against THIS FILE'S constants.
#
# TWO CHECKS ARE DELIBERATELY CORPUS-CONDITIONAL, AND THAT IS A DESIGN
# DECISION, NOT A LOOPHOLE.  CORRECTIONS 156.9 recorded that cfr1's and cts3's
# RULE 21 premise checks become KNOWN-FALSE assertions the moment their own
# rows land -- a scorer that must FAIL after a successful ingest teaches the
# next reader to ignore its own FAILs.  So:
#   * the RULE 21 premise admits exactly TWO states, 0 rows (pre-registration)
#     or exactly N_RUNS rows (post-ingest), and FAILS on anything else -- which
#     is the state that actually matters, a partial ingest, a duplicate, or a
#     name collision before launch.  The ORDERING claim RULE 21 really makes is
#     not checkable from inside a python file at all; it is proved by wall
#     clock, git commit time against the earliest sacct Submit, and that proof
#     lives in the CORRECTIONS entry, not here.
#   * the horizon census admits {} (pre-cts3) or {772} (post-cts3) as the set
#     of above-100 horizons on this network, and will admit {772} again once
#     cpk2 itself lands, because cpk2 runs at exactly cts3's E.
# The corpus-derived CONSTANTS (SIGMA_W, the scalar baseline, cts1's cliff) are
# checked as hard equalities against frozen literals.  If corpus growth moves
# them the check FAILS, and that FAIL is the audit working: every bar in this
# file derives from the FROZEN literal, never from the live value, so a drift
# cannot move a verdict -- it can only be reported.
# =============================================================================
def selftest():
    ok = True

    def chk(cond, label, got="", want=""):
        nonlocal ok
        print("  %-4s %-56s %s%s" % ("PASS" if cond else "FAIL", label,
                                     got, (" (want %s)" % want) if want else ""))
        if not cond:
            ok = False

    print("cR1_cpk2_score.py --selftest")
    print("-" * 78)

    print("A. the grid, the arms and the run count")
    chk(SWEEP_ARMS == ("k45", "k47", "k49", "k50", "k52"), "sweep arms",
        " ".join(SWEEP_ARMS))
    for k in KGRID:
        chk(arm_spec("k%02d" % k) == kspec(k), "spec for k=%d" % k, kspec(k))
    chk(arm_spec(ANCHOR) == "scalar", "the anchor is m=1 scalar", arm_spec(ANCHOR))
    chk(all(sum(int(x) for x in kspec(k)[1:-1].split(",")) == N_TENSORS
            for k in KGRID), "every spec covers all %d tensors" % N_TENSORS)
    chk(N_RUNS == 18, "run count", str(N_RUNS), "18")
    chk(SEEDS == (3, 4, 5), "seeds are FRESH (cpk1/cts1/cts2/scl1/cts3 use 0,1,2)",
        str(SEEDS))
    chk(EPOCHS == 772 and CONTROL_EPOCHS == 100, "horizons",
        "%d / %d" % (EPOCHS, CONTROL_EPOCHS))
    chk(list(KGRID) == sorted(set(KGRID)), "the grid is strictly increasing")
    chk(49 in KGRID, "the incumbent argmax k*=49 is on the grid")
    chk(sum(1 for k in KGRID if k < 49) == 2 and sum(1 for k in KGRID if k > 49) == 2,
        "the grid brackets the incumbent 2 left / 2 right",
        "left %s right %s" % ([k for k in KGRID if k < 49],
                              [k for k in KGRID if k > 49]))

    print("B. the noise floor and the scalar floor, re-derived from the corpus")
    if not os.path.exists(CSV):
        chk(False, "results/all_runs.csv present", CSV)
    else:
        s, df, nc, nm = noise_floor_from_csv()
        chk(abs(s - SIGMA_W) < 5e-5, "SIGMA_W reproduces", "%.6f" % s,
            "%.6f" % SIGMA_W)
        chk(df == SIGMA_DF, "SIGMA df reproduces", str(df), str(SIGMA_DF))
        chk(nc == SIGMA_CELLS, "SIGMA cells reproduce", str(nc), str(SIGMA_CELLS))
        chk(nm == SIGMA_MEMBERS, "SIGMA stratum members reproduce", str(nm),
            str(SIGMA_MEMBERS))
        sb = scalar_baseline_from_csv()
        chk(sb is not None, "scalar rows present in the cell")
        if sb:
            mu, n, nb, per = sb
            chk(abs(mu - SCALAR_BASELINE) < 5e-5, "scalar baseline reproduces",
                "%.6f" % mu, "%.6f" % SCALAR_BASELINE)
            chk(n == SCALAR_N and nb == SCALAR_BATCHES,
                "scalar baseline n / batches reproduce", "%d / %d" % (n, nb),
                "%d / %d" % (SCALAR_N, SCALAR_BATCHES))
            bm = sorted(statistics.mean(v) for v in per.values())
            print("       scalar batch means: %s"
                  % "  ".join("%.4f" % x for x in bm))
            print("       between-batch SD of those means: %.6f"
                  % statistics.stdev(bm))
        c = cts1_cliff_from_csv()
        chk(c is not None, "cts1 arms present in the corpus")
        if c:
            chk(abs(c[0] - CTS1_K49) < 5e-5, "cts1 k49 reproduces", "%.6f" % c[0])
            chk(abs(c[1] - CTS1_K50) < 5e-5, "cts1 k50 reproduces", "%.6f" % c[1])
            chk(abs(c[2] - CTS1_CLIFF) < 5e-5, "cts1 cliff reproduces", "%.6f" % c[2])
            chk(abs(PREMISE_BAR - c[2] / 2.0) < 5e-4,
                "PREMISE_BAR is half cts1's own cliff", "%.4f" % (c[2] / 2.0))

        print("C. the grid choice, justified against the corpus and not prose")
        g = arm_levels_from_csv()
        rows = []
        for gran in sorted({k[0] for k in g}, key=lambda z: (len(z), z)):
            if not M2_RE.match(gran):
                continue
            k = int(gran[1:-1].split(",")[0])
            if not 42 <= k <= 55:
                continue
            vals = [x for kk, v in g.items() if kk[0] == gran for x in v]
            batches = sorted({kk[1] for kk in g if kk[0] == gran})
            mu = statistics.mean(vals)
            rows.append((k, gran, mu, len(vals), batches))
        rows.sort()
        print("       k    spec       n  batches                  p5@100   "
              "above floor   informative?")
        for k, gran, mu, n, batches in rows:
            above = mu - SCALAR_BASELINE
            good = above > FLOOR_BAR
            print("       %-4d %-10s %2d  %-24s %8.4f  %+8.4f pp   %s%s"
                  % (k, gran, n, ",".join(batches), mu, above,
                     "YES" if good else "NO ",
                     "   <-- ON THE cpk2 GRID" if k in KGRID else ""))
        onfloor = {k for k, _, mu, _, _ in rows if mu - SCALAR_BASELINE <= FLOOR_BAR}
        chk(53 in onfloor, "k=53 is AT THE FLOOR in the corpus (so it is NOT run)",
            "%.4f vs baseline %.4f" % (
                [r[2] for r in rows if r[0] == 53][0] if any(r[0] == 53 for r in rows)
                else float("nan"), SCALAR_BASELINE))
        chk(not (set(KGRID) & onfloor),
            "no cpk2 sweep arm is at the floor in the corpus",
            "grid %s, on-floor %s" % (list(KGRID), sorted(onfloor)))

        print("D. RULE 21 premise and the horizon census -- CORPUS-CONDITIONAL")
        n = count_cpk2_rows()
        chk(n in (0, N_RUNS),
            "cpk2 rows in the corpus are 0 (pre-reg) or exactly %d" % N_RUNS,
            "%d found -> %s" % (n, "PRE-REGISTRATION" if n == 0
                                else ("POST-INGEST" if n == N_RUNS else "PARTIAL")),
            "0 or %d" % N_RUNS)
        hz = horizons_on_net()
        chk(hz in ([], [EPOCHS]),
            "above-100 horizons on %s are {} or {%d}" % (NET, EPOCHS),
            str(hz), "[] or [%d]" % EPOCHS)

    print("E. the SE arithmetic and every bar in SE units")
    chk(abs(SE_ARM_DIFF - SIGMA_W * math.sqrt(2.0 / 3.0)) < 1e-9,
        "SE_ARM_DIFF", "%.6f" % SE_ARM_DIFF)
    chk(abs(SE_DD - SIGMA_W * math.sqrt(4.0 / 3.0)) < 1e-9,
        "SE_dD (unpaired, conservative)", "%.6f" % SE_DD)
    chk(abs(SE_FLOOR - SIGMA_W * math.sqrt(1.0 / 3.0 + 1.0 / SCALAR_N)) < 1e-9,
        "SE_FLOOR (3-seed arm vs the %d-run scalar corpus)" % SCALAR_N,
        "%.6f" % SE_FLOOR)
    chk(abs(ARGMAX_BAR - 2 * SE_ARM_DIFF) < 5e-6, "ARGMAX_BAR = 2*SE_ARM_DIFF",
        "%.6f" % (2 * SE_ARM_DIFF))
    chk(abs(FLOOR_BAR - 2 * SE_FLOOR) < 5e-6, "FLOOR_BAR = 2*SE_FLOOR",
        "%.6f" % (2 * SE_FLOOR))
    chk(abs(NOISY_BAR - 3 * SIGMA_W) < 5e-6, "NOISY_BAR = 3*SIGMA_W",
        "%.6f" % (3 * SIGMA_W))
    chk(abs(CONV_BAR - SLOPE_K50_AT_100 / 2.0) < 1e-9,
        "CONV_BAR is exactly half cts2's measured k50 slope at 100",
        "%.6f" % (SLOPE_K50_AT_100 / 2.0))
    chk(abs(TIGHT_BAR - CONVERGED_SLOPE) < 1e-12,
        "TIGHT_BAR is s*, the slope that defined E = 772", "%.5f" % TIGHT_BAR)
    chk(abs(CTS3_SE_["k49"]) < TIGHT_BAR and abs(CTS3_SE_["k50"]) < TIGHT_BAR,
        "cts3's own two arms cleared TIGHT_BAR at E",
        "%.5f / %.5f" % (CTS3_SE_["k49"], CTS3_SE_["k50"]))
    print("       PREMISE  %8.4f pp = %6.2f SE_ARM_DIFF" %
          (PREMISE_BAR, PREMISE_BAR / SE_ARM_DIFF))
    print("       ARGMAX   %8.4f pp = %6.2f SE_ARM_DIFF" %
          (ARGMAX_BAR, ARGMAX_BAR / SE_ARM_DIFF))
    print("       FLOOR    %8.4f pp = %6.2f SE_FLOOR" %
          (FLOOR_BAR, FLOOR_BAR / SE_FLOOR))
    print("       NOISY    %8.4f pp = %6.2f SIGMA_W" %
          (NOISY_BAR, NOISY_BAR / SIGMA_W))

    print("F. THE PRE-REGISTERED POINT PREDICTION, re-derived from cts3")
    A, B = calibrate()
    chk(abs(A - 0.649507) < 5e-6, "GAIN_A reproduces", "%.6f" % A, "0.649507")
    chk(abs(B - 152.256507) < 5e-6, "GAIN_B reproduces", "%.6f" % B,
        "152.256507")
    print("       gain(k) = %.6f + %.4f * slope100(k)   "
          "(2 points, ZERO residual df)" % (A, B))
    if os.path.exists(CSV):
        lv = level100_from_corpus()
        for a in ARMS:
            if a not in lv:
                chk(False, "LEVEL100[%s] re-derives from the corpus" % a, "absent")
                continue
            mu, batches = lv[a]
            chk(abs(mu - LEVEL100[a]) < 5e-5,
                "LEVEL100[%s] re-derives (%s)" % (a, ",".join(batches)),
                "%.6f" % mu, "%.6f" % LEVEL100[a])
    print("       arm   slope@100   level@100   pred gain   pred @E=772")
    preds = {}
    for a in ARMS:
        preds[a] = predicted_level(a, A, B)
        print("       %-5s %+10.5f %11.4f %11.4f %13.4f"
              % (a, SLOPE100[a], LEVEL100[a], predicted_gain(a, A, B), preds[a]))
    sweep = {a: preds[a] for a in SWEEP_ARMS}
    best = max(sweep, key=sweep.get)
    second = sorted(sweep, key=sweep.get)[-2]
    chk(best == PRED_ARGMAX, "the predicted argmax", best, PRED_ARGMAX)
    chk(abs((sweep[best] - sweep[second]) - PRED_MARGIN) < 5e-6,
        "the predicted margin over the runner-up",
        "%.4f pp over %s" % (sweep[best] - sweep[second], second),
        "%.6f" % PRED_MARGIN)
    print("       => PREDICTED BRANCH: %s  (%.2f SE_ARM_DIFF of margin)"
          % (PRED_BRANCH, (sweep[best] - sweep[second]) / SE_ARM_DIFF))
    for a in SWEEP_ARMS:
        if a == PRED_ARGMAX:
            continue
        need = preds[PRED_ARGMAX] - LEVEL100[a]
        got = predicted_gain(a, A, B)
        print("       to take the argmax %s must gain %8.4f pp = %5.2fx its "
              "predicted %6.4f" % (a, need, need / got if got > 0 else float("inf"),
                                   got))
    print("       SANITY, against cts3's OWN measured levels at E "
          "(the model was fitted on its GAINS, not its levels):")
    for a in ("k49", "k50"):
        print("         %-4s predicted %8.4f   cts3 measured %8.4f   "
              "residual %+7.4f pp"
              % (a, preds[a], CTS3_PE[a], preds[a] - CTS3_PE[a]))
    chk(all(abs(preds[a] - CTS3_PE[a]) < 1.0 for a in ("k49", "k50")),
        "both sanity residuals are under 1 pp")
    print("       the LEFT flank is converged at 100 and has nothing to gain:")
    for a in ("k45", "k47", "k49"):
        print("         %-4s slope@100 %+9.5f   %s TIGHT_BAR %.5f"
              % (a, SLOPE100[a],
                 "inside" if abs(SLOPE100[a]) < TIGHT_BAR else "OUTSIDE", TIGHT_BAR))
    chk(all(abs(SLOPE100[a]) < CONV_BAR for a in ("k45", "k47", "k49")),
        "k45/k47/k49 are all inside CONV_BAR at 100 -> MOVES-LEFT would "
        "REFUTE the mechanism",
        " ".join("%s %.5f" % (a, SLOPE100[a]) for a in ("k45", "k47", "k49")))
    chk(abs(SLOPE100["k45"]) < TIGHT_BAR and abs(SLOPE100["k49"]) < TIGHT_BAR
        and abs(SLOPE100["k47"]) > TIGHT_BAR,
        "k45/k49 inside TIGHT_BAR, k47 marginally OUTSIDE it -- recorded, not "
        "rounded", "k47 exceeds by %.5f pp/epoch" % (SLOPE100["k47"] - TIGHT_BAR))
    chk(SLOPE100["k52"] > SLOPE100["k50"] > CONV_BAR,
        "k52 is the least-converged arm on the grid, k50 next",
        "%.5f > %.5f > %.5f" % (SLOPE100["k52"], SLOPE100["k50"], CONV_BAR))

    print("-" * 78)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# =============================================================================
def score(runsdir):
    recs = collect(runsdir)
    print("=" * 78)
    print("cpk2 -- DOES THE CUT-POSITION ARGMAX k* = 49 SURVIVE THE BUDGET?")
    print("        E = %d epochs, control at %d, seeds %s"
          % (EPOCHS, CONTROL_EPOCHS, ",".join(str(s) for s in SEEDS)))
    print("scorer: %s   (registered before any cpk2 run existed)"
          % os.path.basename(__file__))
    print("=" * 78)

    # ---- G0 provenance ------------------------------------------------------
    print("\nG0 PROVENANCE")
    fail = []
    if len(recs) != N_RUNS:
        fail.append("expected %d runs, found %d" % (N_RUNS, len(recs)))
    jobs = {r.get("job_id") for r in recs if r.get("job_id")}
    if len(jobs) != len(recs):
        fail.append("job ids not distinct: %d ids for %d runs" % (len(jobs), len(recs)))
    for r in recs:
        b = r.get("base", r.get("path"))
        if r.get("bad"):
            fail.append("%s: %s" % (b, r["bad"]))
            continue
        if r["dup"]:
            fail.append("%s: repeated flag(s) %s" % (b, ",".join(r["dup"])))
        if r["meas_arm"] is None:
            fail.append("%s: stepsize-groups %r is not a registered arm"
                        % (b, r["args"].get("stepsize-groups")))
        elif r["meas_arm"] != r["name_arm"]:
            fail.append("%s: NAME says %s, ARGS say %s"
                        % (b, r["name_arm"], r["meas_arm"]))
        if r["clip"] != CLIP_C:
            fail.append("%s: ENV BETA_CLIP is %r, registered %r"
                        % (b, r["clip"], CLIP_C))
        if r["env"].get("AUGMENT") != AUG:
            fail.append("%s: ENV AUGMENT %r, registered %r"
                        % (b, r["env"].get("AUGMENT"), AUG))
        if r["env"].get("PROBE") not in (None, "0"):
            fail.append("%s: ENV PROBE %r, registered 0" % (b, r["env"].get("PROBE")))
        if r["seed"] != r["name_seed"]:
            fail.append("%s: NAME seed %s, ARGS seed %s"
                        % (b, r["name_seed"], r["seed"]))
        if r["seed"] not in SEEDS:
            fail.append("%s: seed %s is not a registered seed %s"
                        % (b, r["seed"], SEEDS))
        if norm_num(r["args"].get("num-epochs")) != norm_num(EPOCHS):
            fail.append("%s: ARGS num-epochs %r, registered %d"
                        % (b, r["args"].get("num-epochs"), EPOCHS))
        for flag, want in (("meta-stepsize", MS), ("alpha0", ALPHA0),
                           ("dataset", DSET), ("NN-name", NET),
                           ("batch-size", str(BATCH)), ("gamma", GAMMA),
                           ("alg-base", BASE_ALG), ("alg-meta", META_ALG)):
            got = r["args"].get(flag)
            if norm_num(got) is not None and norm_num(want) is not None:
                same = norm_num(got) == norm_num(want)
            else:
                same = str(got) == str(want)
            if not same:
                fail.append("%s: ARGS %s = %r, registered %r" % (b, flag, got, want))
        if r["n_ep"] != EPOCHS:
            fail.append("%s: %d epoch lines, registered %d" % (b, r["n_ep"], EPOCHS))
    for f in fail:
        print("  FAIL  %s" % f)
    if fail:
        print("\nVERDICT: UNRESOLVED-PROVENANCE")
        return 3
    print("  PASS  %d runs, %d job ids, %d/%d epochs, NAME == ARGS == ENV, no "
          "repeated flag" % (len(recs), len(jobs), EPOCHS, EPOCHS))

    cell = collections.defaultdict(list)
    for r in recs:
        cell[r["meas_arm"]].append(r)
    for a in ARMS:
        if len(cell[a]) != len(SEEDS):
            print("  FAIL  cell %s has %d runs" % (a, len(cell[a])))
            print("\nVERDICT: UNRESOLVED-PROVENANCE")
            return 3

    def M(a, key):
        v = [r[key] for r in cell[a] if r[key] is not None]
        return statistics.mean(v) if len(v) == len(SEEDS) else None

    def SD(a, key):
        v = [r[key] for r in cell[a] if r[key] is not None]
        return statistics.stdev(v) if len(v) > 1 else None

    # ---- the per-run table --------------------------------------------------
    print("\nPER RUN.  plateau5 = mean of the last 5 epoch lines of the stated "
          "horizon\n          (%d..%d at E, %d..%d at the control)."
          % (EPOCHS - 5, EPOCHS - 1, CONTROL_EPOCHS - 5, CONTROL_EPOCHS - 1))
    print("  %-5s %-5s %-9s %10s %10s %10s %10s %11s"
          % ("arm", "seed", "spec", "test@100", "test@E", "train@100", "train@E",
             "slope@E"))
    for a in ARMS:
        for r in sorted(cell[a], key=lambda x: x["seed"]):
            print("  %-5s %-5d %-9s %s %s %s %s %s"
                  % (a, r["seed"], arm_spec(a), fmt(r["p5_100"], 10),
                     fmt(r["p5_E"], 10), fmt(r["t5_100"], 10), fmt(r["t5_E"], 10),
                     fmt(r["slope_E"], 11, 5)))

    # ---- R2 divergence ------------------------------------------------------
    print("\nR2 DIVERGENCE")
    dead = [r["base"] for r in recs
            if r["p5_E"] is None or not math.isfinite(r["p5_E"])
            or r["p5_E"] <= DEAD_BAR or r["n_ep"] != EPOCHS]
    if dead:
        for d in dead:
            print("  FAIL  %s is DEAD (plateau5 <= %.2f, non-finite, or short)"
                  % (d, DEAD_BAR))
        print("\nVERDICT: UNRESOLVED-DIVERGED")
        return 4
    print("  PASS  every run finished %d epochs with plateau5 > %.2f pp"
          % (EPOCHS, DEAD_BAR))

    # ---- R3 noise -----------------------------------------------------------
    print("\nR3 NOISE   bar = 3*SIGMA_W = %.4f pp   (SIGMA_W %.6f, df %d, %d cells)"
          % (NOISY_BAR, SIGMA_W, SIGMA_DF, SIGMA_CELLS))
    noisy = []
    for a in ARMS:
        sd = SD(a, "p5_E")
        print("  %-5s SD at E %8.4f pp %s" % (a, sd, "" if sd <= NOISY_BAR
                                              else "  <<< OVER BAR"))
        if sd > NOISY_BAR:
            noisy.append(a)
    if noisy:
        print("\nVERDICT: UNRESOLVED-NOISY (%s)" % ",".join(noisy))
        return 5
    print("  PASS")

    # ---- the two curves -----------------------------------------------------
    print("\nTHE CURVES, both read off the SAME runs.")
    print("  %-5s %-9s %10s %10s %10s %10s %10s %10s"
          % ("arm", "spec", "test@100", "test@E", "gain d", "pred d",
             "train@100", "train@E"))
    A_, B_ = calibrate()
    for a in ARMS:
        d = M(a, "p5_E") - M(a, "p5_100")
        print("  %-5s %-9s %s %s %s %s %s %s"
              % (a, arm_spec(a), fmt(M(a, "p5_100"), 10), fmt(M(a, "p5_E"), 10),
                 fmt(d, 10), fmt(predicted_gain(a, A_, B_), 10),
                 fmt(M(a, "t5_100"), 10), fmt(M(a, "t5_E"), 10)))

    # ---- R1 premise: the CLIFF, in batch, at 100 ---------------------------
    d100 = M("k49", "p5_100") - M("k50", "p5_100")
    de = M("k49", "p5_E") - M("k50", "p5_E")
    dd = de - d100
    print("\nR1 PREMISE -- THE CLIFF, in batch, at %d epochs" % CONTROL_EPOCHS)
    print("  D100 = %.4f pp = %.2f SE_ARM_DIFF   bar %.4f pp"
          % (d100, d100 / SE_ARM_DIFF, PREMISE_BAR))
    print("  cts1's own 100-epoch cliff, DESCRIPTIVE only: %.4f pp" % CTS1_CLIFF)
    print("  DE   = %.4f pp   dD = %.4f pp = %.2f SE_dD "
          "(cts3 measured dD -5.6747 on seeds 0,1,2 -- CROSS-BATCH, descriptive)"
          % (de, dd, dd / SE_DD))
    if d100 < PREMISE_BAR:
        print("  FAIL  the cliff is not present in this batch at 100 epochs.")
        print("        cts1/cts2/cts3 arms may NOT be spliced in to rescue it.")
        print("\nVERDICT: UNRESOLVED-PREMISE")
        return 6
    print("  PASS")

    # ---- R-CTRL premise: the PEAK, in batch, at 100 ------------------------
    print("\nR-CTRL PREMISE -- THE PEAK, in batch, at %d epochs" % CONTROL_EPOCHS)
    s100 = {a: M(a, "p5_100") for a in SWEEP_ARMS}
    b100 = max(s100, key=s100.get)
    r100 = sorted(s100, key=s100.get)[-2]
    m100 = s100[b100] - s100[r100]
    print("  argmax at 100 = %s (%.4f), runner-up %s (%.4f), margin %.4f pp "
          "= %.2f SE   bar %.4f"
          % (b100, s100[b100], r100, s100[r100], m100, m100 / SE_ARM_DIFF,
             ARGMAX_BAR))
    if b100 != "k49" or m100 <= ARGMAX_BAR:
        print("  FAIL  cpk1's peak does NOT reproduce on fresh seeds at 100 "
              "epochs, so no relocation can be attributed to the horizon.")
        print("\nVERDICT: UNRESOLVED-CONTROL")
        return 7
    print("  PASS  cpk1's argmax reproduces in batch, on fresh seeds")

    # ---- R-FLOOR ------------------------------------------------------------
    print("\nR-FLOOR SATURATION   bar = 2*SE_FLOOR = %.4f pp" % FLOOR_BAR)
    print("  the baseline at each horizon is THIS BATCH's OWN %s arm at that "
          "horizon" % ANCHOR)
    print("  (corpus scalar baseline at 100, DESCRIPTIVE: %.4f pp over %d runs "
          "in %d batches)" % (SCALAR_BASELINE, SCALAR_N, SCALAR_BATCHES))
    fl_E, fl_100 = M(ANCHOR, "p5_E"), M(ANCHOR, "p5_100")
    informative = []
    print("  %-5s %10s %12s %6s   %10s %12s %6s"
          % ("arm", "test@100", "above floor", "info", "test@E", "above floor",
             "info"))
    for a in SWEEP_ARMS:
        a100 = M(a, "p5_100") - fl_100
        aE = M(a, "p5_E") - fl_E
        i100, iE = a100 > FLOOR_BAR, aE > FLOOR_BAR
        print("  %-5s %10.4f %+12.4f %6s   %10.4f %+12.4f %6s"
              % (a, M(a, "p5_100"), a100, "YES" if i100 else "NO ",
                 M(a, "p5_E"), aE, "YES" if iE else "NO "))
        if iE:
            informative.append(a)
    if "k49" not in informative:
        print("  FAIL  the incumbent argmax has itself saturated to the m=1 "
              "baseline at E.")
        print("\nVERDICT: UNRESOLVED-SATURATED")
        return 8
    print("  %d of %d sweep arms are INFORMATIVE at E: %s"
          % (len(informative), len(SWEEP_ARMS), " ".join(informative)))

    # ---- R-CONV -------------------------------------------------------------
    print("\nR-CONV HORIZON EFFECTIVENESS   CONV_BAR %.6f   TIGHT_BAR %.5f"
          % (CONV_BAR, TIGHT_BAR))
    print("  %-5s %11s %11s %11s   %s"
          % ("arm", "slope@100", "slope@E", "SD(slope@E)", "state"))
    notconv = []
    for a in ARMS:
        sE, s1 = M(a, "slope_E"), M(a, "slope_100")
        sd = SD(a, "slope_E")
        if sE is None:
            state = "UNMEASURED"
        elif sE > CONV_BAR:
            state = "NOT CONVERGED"
            notconv.append(a)
        elif abs(sE) <= TIGHT_BAR:
            state = "TIGHT-CONVERGED"
        else:
            state = "converged (loose)"
        print("  %-5s %s %s %s   %s"
              % (a, fmt(s1, 11, 5), fmt(sE, 11, 5), fmt(sd, 11, 5), state))
    if notconv:
        print("  FAIL  %s exceed CONV_BAR at E." % ",".join(notconv))
        print("        The 772-epoch argmax below is REPORTED but is NOT the "
              "converged argmax and may not be quoted as final.")
    else:
        print("  PASS  every arm is converged at E by the registered bar")

    # ---- THE VERDICT --------------------------------------------------------
    sE = {a: M(a, "p5_E") for a in informative}
    best = max(sE, key=sE.get)
    runner = sorted(sE, key=sE.get)[-2] if len(sE) > 1 else None
    margin = (sE[best] - sE[runner]) if runner else float("inf")
    print("\nTHE ARGMAX AT E = %d" % EPOCHS)
    for a in sorted(sE, key=lambda z: -sE[z]):
        print("  %-5s %10.4f pp   %+8.4f vs k49   %+8.4f vs its own 100-epoch"
              % (a, sE[a], sE[a] - sE.get("k49", float("nan")),
                 sE[a] - M(a, "p5_100")))
    print("  best %s, runner-up %s, margin %.4f pp = %.2f SE_ARM_DIFF   "
          "bar %.4f" % (best, runner, margin, margin / SE_ARM_DIFF, ARGMAX_BAR))

    if margin <= ARGMAX_BAR:
        v = "PEAK-DISSOLVES"
    elif best == "k49":
        v = "k*-UNMOVED"
    elif arm_k(best) < 49:
        v = "k*-MOVES-LEFT"
    else:
        v = "k*-MOVES-RIGHT"

    print("\nPRE-REGISTERED PREDICTION vs MEASURED")
    print("  predicted branch %s, predicted argmax %s, predicted margin "
          "%.4f pp" % (PRED_BRANCH, PRED_ARGMAX, PRED_MARGIN))
    print("  measured  branch %s, measured  argmax %s, measured  margin "
          "%.4f pp" % (v, best, margin))
    print("  %-5s %10s %10s %10s" % ("arm", "pred @E", "meas @E", "residual"))
    for a in ARMS:
        p = predicted_level(a, A_, B_)
        print("  %-5s %10.4f %10.4f %+10.4f" % (a, p, M(a, "p5_E"), M(a, "p5_E") - p))

    print("\n" + "=" * 78)
    print("VERDICT (the ARGMAX at %d epochs): %s" % (EPOCHS, v))
    if notconv:
        print("VERDICT (convergence):            UNRESOLVED-NOT-CONVERGED (%s)"
              % ",".join(notconv))
        print("  => the batch says NOTHING about the CONVERGED argmax.")
        print("FINAL: %s AT %d EPOCHS | UNRESOLVED-NOT-CONVERGED" % (v, EPOCHS))
    else:
        print("VERDICT (convergence):            ALL ARMS CONVERGED at E")
        print("FINAL: %s | CONVERGED" % v)
    print("=" * 78)
    return 0


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runsdir", nargs="?",
                    help="directory holding the cpk2-*.out files")
    ap.add_argument("--selftest", action="store_true",
                    help="re-derive every registered constant and exit")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.runsdir:
        ap.error("runsdir is required unless --selftest is given")
    return score(a.runsdir)


if __name__ == "__main__":
    sys.exit(main())
