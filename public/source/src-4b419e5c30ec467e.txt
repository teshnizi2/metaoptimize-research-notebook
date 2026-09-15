#!/usr/bin/env python3
# =============================================================================
# cS2_cpk3_score.py -- THE REGISTERED SCORER FOR BATCH `cpk3`.
#
# COMMITTED BEFORE ANY cpk3 RUN EXISTS (STANDING RULE 21).  Run it UNEDITED
# (RULE 16).  Documented command-line arguments are not edits; nothing below
# this line may be changed once a cpk3 run exists on disk.  If this file turns
# out to be broken it is FROZEN and a NEW file is registered (precedent
# cN1/cN2, CORRECTIONS 149).
#
# -----------------------------------------------------------------------------
# THE GAP THIS BATCH CLOSES
# -----------------------------------------------------------------------------
# `cpk2` (CORRECTIONS 158) measured the cut-position argmax at 772 epochs on
# fresh seeds and returned k*-UNMOVED | CONVERGED, with k = 49 ahead of the
# runner-up by 10.5140 pp = 14.04 SE.  The adversarial round that followed
# established the limit of that sentence exactly:
#
#   **THE PEAK'S IDENTITY IS SETTLED AMONG THE SAMPLED CUTS.  ITS LOCATION IS
#   NOT.**  k = 44, 46 and 48 have NEVER been run at ANY horizon in this cell
#   -- the corpus holds ZERO rows with granularity [44,18], [46,16] or [48,14]
#   anywhere, at any horizon, on any network.  k is a 1-BASED INTEGER TENSOR
#   INDEX, so there is no interpolation between 49 and 50: the claim must read
#   "k* = 49 among {45,47,49,50,52}" until the gaps are MEASURED.
#
# cpk3 measures them.
#
# -----------------------------------------------------------------------------
# WHAT THE LIVE MODEL SAYS, AND WHY IT MAKES THIS BATCH MORE THAN AN INFILL
# -----------------------------------------------------------------------------
# k is 1-BASED and is the SIZE of the first group: group 1 is
# named_parameters()[0:k].  On the LIVE ResNet18_c100 (62 tensors, 11,220,132
# parameters, measured by the launcher's guard 4 and written to
# runs/cpk3/PARTITION-MANIFEST.txt, not taken from any document):
#
#     45  layer3.1.bn2.bias                256
#     46  layer4.0.conv1.weight      1,179,648
#     47  layer4.0.bn1.weight              512
#     48  layer4.0.bn1.bias                512
#     49  layer4.0.conv2.weight      2,359,296
#     50  layer4.0.bn2.weight              512
#
# So the two cuts the corpus has never run, k = 46 and k = 48, are exactly the
# two indices that TURN cpk2's TWO-TENSOR STEPS INTO SINGLE-TENSOR STEPS.  With
# the consecutive grid 45..50 in ONE batch, cpk3 measures FIVE CONSECUTIVE
# SINGLE-TENSOR STEPS, every one of them WITHIN batch:
#
#     45 -> 46   layer4.0.conv1.weight   1,179,648 params
#     46 -> 47   layer4.0.bn1.weight           512
#     47 -> 48   layer4.0.bn1.bias             512
#     48 -> 49   layer4.0.conv2.weight   2,359,296
#     49 -> 50   layer4.0.bn2.weight           512
#
# That is a complete single-tensor decomposition of the whole rise into the
# peak AND of the cliff off it.  cpk2 could only measure the two-tensor
# aggregates 45->47 (+2.3847 pp) and 47->49 (+10.5140 pp).
#
# -----------------------------------------------------------------------------
# THE DESIGN.  7 arms x 3 seeds = 21 jobs, ONE submission, 772 epochs.
# -----------------------------------------------------------------------------
#   SWEEP (all m = 2, contiguous split of named_parameters()):
#       k45 [45,17]  k46 [46,16]  k47 [47,15]  k48 [48,14]  k49 [49,13]
#       k50 [50,12]
#   FLOOR ANCHOR (m = 1):
#       k01 scalar
#   SEEDS {6, 7, 8}.  HORIZON 772.  Clamp at the campaign standard
#   BETA_CLIP = -15:-2.3026.  Every non-axis flag byte-matched to cpk2's
#   launcher, which was read rather than reconstructed.
#
# WHY 21 RUNS AND NOT THE 6 THE CYCLE-133 CLOSE PROPOSED.  Each addition is
# justified separately, with what is LOST if it is dropped:
#
#   * k46, k48 (the proposed core, 6 runs).  The two unrun cuts.
#   * + k49 IN BATCH (3 runs).  Without it "k48 > k49 relocates the peak" is a
#     CROSS-BATCH comparison, and BATCH is the unit of replication in this
#     corpus (F(62,85) = 5.47, p 6.9e-13).  The batch could not decide its own
#     headline branch.  With it, the branch is decided WITHIN cpk3.
#   * + k47 (3 runs).  Makes 47 -> 48 a single-tensor step IN BATCH.  Without
#     it the decomposition of the +10.5140 pp step is cross-batch.
#   * + k45 (3 runs).  Makes 45 -> 46 a single-tensor step in batch -- the ONLY
#     place on this grid where a 1,179,648-parameter conv moves ALONE, and
#     therefore the sharpest available test of the mass hypothesis below.
#   * + k50 (3 runs).  WITHOUT IT k49 SITS AT THE RIGHT EDGE OF THE GRID, which
#     is the exact defect cpk2 was criticised for.  With it the argmax is
#     strictly INTERIOR to a consecutive grid, the cliff premise (R1) is in
#     batch, and the k49->k50 cliff gets a THIRD independent seed set.
#   * + k01 scalar floor (3 runs).  The informativeness gate must run against a
#     baseline AT THE HORIZON IT IS APPLIED AT and IN THE BATCH IT JUDGES.
#     cpk2 measured the 772-epoch floor once (23.3680) and found it had not
#     moved from the 100-epoch corpus floor (+0.3513 pp = 0.47 SE).  ONE
#     measurement on ONE seed set is not a replication.  Splicing it into cpk3
#     would make the only gate that decides whether an arm is INFORMATIVE a
#     cross-batch splice -- weakening precisely the sentence this batch exists
#     to strengthen.  IF THIS ARM WERE DROPPED, cpk3 could not claim in batch
#     that any arm is above the m=1 floor, and every FLOOR verdict would carry
#     a cross-batch qualifier.  It is kept.
#
# WHY k = 44 IS **DECLINED**, stated so the omission is a decision and not an
# oversight.  WHAT IT WOULD BUY: it would extend the contiguous measured range
# from 45..50 to 44..50 and give the 44 -> 45 single-tensor step
# (layer3.1.bn2.bias).  WHAT IT COSTS: 3 runs, ~16.5 GPU-hours, a 16% larger
# batch.  WHY THAT TRADE IS DECLINED: it closes NO gap inside the bracket --
# 46 and 48 are the only unsampled integers between 45 and 50 -- and it cannot
# move the peak.  k45 measured 43.5140 at 772, i.e. 12.8987 pp below k49, and
# EVERY left-flank arm is converged at 100 epochs and gains about 1 pp over the
# whole 672 extra epochs (cpk2: k45 +1.0180, k47 +0.8180, k49 +0.9247).  For
# k44 to take the argmax it would have to exceed k45 by ~13 pp, which is an
# order of magnitude more than any left-flank arm moves.
#   **WHAT cpk3 THEREFORE MAY NOT CLAIM:** that the curve is monotone on
#   44 -> 49, or that k* = 49 over any contiguous range wider than 45..50.
#   The left edge of the measured bracket is k = 45 and k <= 44 is UNMEASURED
#   at this horizon.
#
# WHY SEEDS {6,7,8} AND NOT {3,4,5}.  cpk3 re-runs k45, k47, k49 and k50, all
# of which cpk2 ran at 772 on seeds {3,4,5}.  At {3,4,5} TWELVE of these
# twenty-one runs would be EXACT-CONFIGURATION RE-EXECUTIONS of existing corpus
# rows -- the defect CORRECTIONS 152 found in scl1 and which cost that batch
# its independence claim.  Seeds {6,7,8} are used nowhere in this cell: the
# only ResNet18_c100 rows carrying them are batch `p7` at alpha0 = 1e-3, 20
# epochs, named granularities, which is a different cell on three axes.  So
# cpk3 produces ZERO duplicate rows AND buys a THIRD DISJOINT SEED SET for
# k49 at 772 (cts3 {0,1,2}, cpk2 {3,4,5}, cpk3 {6,7,8}).
#   THE PRICE, STATED: every level comparison to cpk2 is cross-batch AND
#   cross-seed, so it is descriptive only.  Every GATE below is within cpk3.
#
# -----------------------------------------------------------------------------
# THE PRE-REGISTERED POINT PREDICTION.  WRITTEN BEFORE ANY RUN EXISTS.
# -----------------------------------------------------------------------------
# THE ONE THING THE CORPUS ALREADY PROVES ABOUT THIS NEIGHBOURHOOD:
# PARAMETER MASS IS NOT A COMPLETE LAW.  Group 1's parameter share is IDENTICAL
# at k = 49 and k = 50 to within 512 of 11,220,132 parameters (56.28% both),
# and those two arms are 20.4993 pp apart.  A 512-parameter batch-norm weight,
# moved by one index, is the largest single effect anywhere on this grid.
#
# BUT THE CORPUS ALSO SHOWS MASS IS NOT IRRELEVANT: the 45 -> 47 step moves
# 1,180,160 parameters and buys +2.3847 pp, while the 47 -> 49 step moves
# 2,359,808 parameters and buys +10.5140 pp -- 2.00x the mass, 4.41x the gain.
#
# THE REGISTERED PREDICTION IS THEREFORE A SHARP, FALSIFIABLE HYPOTHESIS RATHER
# THAN AN INTERPOLATION.  Call it **H-MASS**: within a multi-tensor step, each
# tensor carries a share of the step PROPORTIONAL TO ITS PARAMETER COUNT.
# Applied to cpk2's two measured steps, with the tensor sizes read off the LIVE
# model, it predicts:
#
#     k46 = 43.5140 + 2.3847 * 1179648/1180160  = 45.8976
#     k48 = 45.8987 + 10.5140 *     512/2359808 = 45.9009
#
# i.e. **k46 AND k48 SHOULD BOTH LAND ON TOP OF k47 (45.8987)**, the entire
# +2.3847 arriving at 45 -> 46 and the entire +10.5140 arriving at 48 -> 49.
# EQUIVALENTLY, IN THE FORM THE BATCH CAN TEST WITHIN ITSELF:
#     H-MASS predicts |k48 - k46| ~ 0 and k46 ~ k47 ~ k48, all three separated
#     from k49 by the FULL 10.5 pp step.
#
# THE NAMED ALTERNATIVE, printed beside it and re-derived by --selftest, is
# **H-EQUI**: each tensor in a step carries HALF of it.  H-EQUI predicts
#     k46 = 44.7063     k48 = 51.1557
# and k48 - k46 = 6.4494 pp.  The two hypotheses are 5.25 pp apart on k48 --
# 6.60 SE_ARM_DIFF -- so this batch SEPARATES THEM, which is why the point
# prediction is worth registering at all.
#
# H-MASS IS FAVOURED, and the reason is stated rather than asserted: the only
# large-mass move in the neighbourhood (2,359,296 params at 48 -> 49) coincides
# with the only large positive step (+10.5140), and the only two 512-parameter
# batch-norm moves whose effect is separately known are tensor 47 (inside the
# small +2.3847 step) and tensor 50 (the -20.4993 cliff).  H-MASS says tensors
# 47 and 48 are ordinary; the cliff says tensor 50 is not.  **cpk3 IS EXACTLY
# THE TEST OF WHETHER THE OTHER TWO layer4.0 BATCH-NORM PARAMETERS ARE SPECIAL
# LIKE TENSOR 50 IS.**  H-MASS says no.  A large |k47 - k48| or |k46 - k47|
# would say yes and would be the first mechanistic handle this thread has had.
#
# PREDICTED BRANCH: **PEAK-CONFIRMED-AT-49**, predicted margin over the
# runner-up 56.4127 - 45.9009 = 10.5118 pp = 13.20 SE_ARM_DIFF.
# THE STANDING SHARP PREDICTION ON RECORD -- "k48 lands between 45.8987 and
# 56.4127, and k48 > k49 RELOCATES THE PEAK" -- is SATISFIED AT ITS LOWER
# ENDPOINT by H-MASS, and this file records that it is being satisfied at an
# endpoint rather than in the middle, so nobody may later retell H-MASS as an
# unremarkable interval prediction.
#   TO RELOCATE THE PEAK, k48 must gain 10.5118 pp over its predicted level,
#   and k46 must gain 10.5151 pp over its predicted level.
#
# THE ANCHOR LEVELS ARE CROSS-BATCH.  43.5140 / 45.8987 / 56.4127 / 35.9133 are
# cpk2's own 772-epoch arm means on seeds {3,4,5}; --selftest re-derives all
# four from the corpus.  THE PREDICTION IS THEREFORE A LEVEL FORECAST ACROSS A
# BATCH BOUNDARY AND ACROSS A SEED SET, and any residual mixes the hypothesis
# with a batch effect.  THE BRANCH VERDICT DOES NOT: it is computed entirely
# from cpk3's own arms.
#
# -----------------------------------------------------------------------------
# THE NOISE FLOOR.  RE-DERIVED FROM THE LIVE CORPUS AT REGISTRATION TIME.
# -----------------------------------------------------------------------------
# CORRECTIONS 156 found cts3's sigma_w had drifted 0.9111 -> 0.9173 between its
# registration and its scoring because scl1's rows landed in between, so NO
# sigma is copied forward here.  Both horizons are re-derived, by the estimator
# cR1 registered (pooled within (batch x granularity) SD of plateau5, m = 2
# arms only, in this exact cell), and the FROZEN LITERAL is
#       SIGMA_W = max(SIGMA_100, SIGMA_772)
# so that the bar can only ever be the WIDER of the two.  At registration:
#       SIGMA_100 = 0.917280  df 58  cells 29  members 87
#       SIGMA_772 = 0.975496  df 14  cells  7  members 21   <-- the max, USED
# The 772 stratum is the horizon-matched one AND the larger, so the choice is
# conservative on both readings.  EVERY BAR BELOW DERIVES FROM THE FROZEN
# LITERAL, never from the live value, so no verdict here can depend on the
# order in which unrelated batches are ingested; --selftest re-derives the live
# values and FAILS loudly if they have moved, and that FAIL is the audit
# working, not a broken gate.
#
# -----------------------------------------------------------------------------
# THE GATES.  They fire IN ORDER; the first failure decides.
# -----------------------------------------------------------------------------
# G0 PROVENANCE.  21 runs, 7 cells x 3 seeds, 21 distinct job ids, E/E epoch
#    lines each, no repeated flag on any ARGS line, every run's NAME agreeing
#    with its own ARGS, and BETA_CLIP / AUGMENT / PROBE audited from the ENV
#    line because they cannot ride the ARGS line (RULE 20).
# R2 DIVERGENCE.  Any run with fewer than E epoch lines, a non-finite metric,
#    or plateau5 at E <= DEAD_BAR = 5.00 pp is DEAD -> UNRESOLVED-DIVERGED.
# R3 NOISE.  Any cell SD at E above NOISY_BAR = 3 * SIGMA_W = 2.9265 pp ->
#    UNRESOLVED-NOISY.
# R1 PREMISE -- THE CLIFF, IN BATCH, AT 100 EPOCHS.  D100 = M(k49, ep 95-99) -
#    M(k50, ep 95-99) must be >= PREMISE_BAR = 12.4170 pp, half of cts1's own
#    in-batch cliff (24.8340, re-derived by --selftest).  Otherwise
#    UNRESOLVED-PREMISE.  cts1's, cts2's, cts3's or cpk2's arms may NOT be
#    spliced in to rescue it.
# R-CTRL PREMISE -- THE SEED-SET REPRODUCTION CONTROL, IN BATCH, AT 100 EPOCHS.
#    NOTE THIS IS **DELIBERATELY NOT cpk2's R-CTRL**, and the change is
#    justified rather than silent.  cpk2 asked whether the 100-epoch argmax was
#    still k49, because cpk2's question was whether the BUDGET moved the peak.
#    cpk3's question is the peak's LOCATION at a fixed budget, and asserting the
#    100-epoch argmax over a grid that CONTAINS THE TWO NEVER-RUN CUTS would
#    prejudge the very thing being measured.  So R-CTRL here asserts only what
#    is already known about the arms cpk2 also ran, on this third seed set:
#        (a) the 100-epoch order M(k49) > M(k47) > M(k45) holds, and
#        (b) M(k49) - M(k47) > PEAK_BAR.
#    Failure -> UNRESOLVED-CONTROL: seeds {6,7,8} do not reproduce the cell and
#    no location claim may be made from them.  k46 and k48 are NOT in this
#    gate, by construction.
# R-FLOOR SATURATION, per arm, at BOTH horizons, ENTIRELY IN BATCH.  An arm is
#    NOT-INFORMATIVE if its mean sits within FLOOR_BAR = 2 * SE_FLOOR = 1.5930
#    pp of THIS BATCH's OWN k01 arm at the MATCHING horizon.  A NOT-INFORMATIVE
#    arm is EXCLUDED from the argmax set.  If k49 itself is NOT-INFORMATIVE at
#    E -> UNRESOLVED-SATURATED.  SE_FLOOR is a 3-seed vs 3-seed contrast here
#    (cR1's was a 3-seed arm against a 17-run corpus baseline), so this bar is
#    WIDER than cR1's 1.1488 -- that is the honest price of an in-batch floor
#    and it is not disguised.
# R-CONV HORIZON EFFECTIVENESS, per arm, both bars printed for every arm:
#    CONV_BAR   0.024235 pp/epoch -- half cts2's own measured k50 slope at 100
#               (0.04847); the identical bar cts3 registered as R4 and cR1 as
#               R-CONV.  RE-USED, not re-invented.  An arm whose terminal
#               20-epoch OLS slope at E exceeds this is NOT CONVERGED.
#    TIGHT_BAR  0.00426 pp/epoch -- s*, the slope that DEFINED E = 772.
#    If ANY INFORMATIVE sweep arm fails CONV_BAR the batch is
#    UNRESOLVED-NOT-CONVERGED: the 772-epoch argmax is still reported but MAY
#    NOT be called the converged argmax and MUST NOT be reported as final.
#
# THE VERDICT ON THE PEAK.  Over the INFORMATIVE sweep arms at E, best = the
# largest arm mean, second = the next largest:
#    PEAK-PLATEAU            best - second <= PEAK_BAR (1.5930 pp = 2 SE).
#                            No peak location may be named at this horizon.
#    PEAK-CONFIRMED-AT-49    best is k49, by more than PEAK_BAR.
#    PEAK-RELOCATES-TO-48    best is k48, by more than PEAK_BAR.
#    PEAK-RELOCATES-TO-46    best is k46, by more than PEAK_BAR.
#    PEAK-RELOCATES-TO-45 / -47 / -50   the remaining possibilities, named
#                            explicitly so the scorer can never be unable to
#                            report what it measured.
#    UNRESOLVED-NOT-CONVERGED  as above, reported ALONGSIDE the peak branch.
#
# REPORTED IN EVERY BRANCH, so nothing can be retold selectively:
#    * the full 100-epoch and 772-epoch curves, TEST AND TRAIN, at every arm;
#    * the per-arm gain d = M(E) - M(100);
#    * the FIVE consecutive single-tensor steps at both horizons, each with its
#      SE_ARM_DIFF, and each labelled with the tensor NAME and PARAMETER COUNT
#      taken from the launcher's live manifest if --manifest is given;
#    * each arm's measured level against BOTH H-MASS and H-EQUI;
#    * the first 100-epoch measurement of k46 and k48 that has ever existed.
#
# MAY NOT CLAIM.  Anything about m (every sweep arm is m = 2; the anchor is an
# anchor, not a rung).  Any cut position off the grid 45..50 -- in particular
# k <= 44 and k >= 51 are NOT measured here and cpk1's k = 17/24/31/38/42/55/60
# stay unmeasured above 100 epochs, so cpk3 cannot restore single-peakedness
# over cpk1's grid and does not try.  Any horizon but 100 and 772.  Any CAPTURE
# quantity -- there is NO layerwise anchor in this batch, CAPTURE is not
# computable from it and may not be computed from it afterwards.  The
# asymptote.  The released floor at long budget.  A POOLED estimate across
# cpk2/cts3/cpk3 of anything.  Any other dataset, network, optimiser pair,
# meta-stepsize, alpha0, batch size, box or hierarchical operator.  And the
# MECHANISM of any step: this file MEASURES single-tensor steps and compares
# them to two named hypotheses; it does not explain why a tensor matters.
# The `scalar` anchor still takes a DIFFERENT CODE PATH (exact string match in
# init_meta, proven on the live source by the launcher's guard 4g), so it is a
# FLOOR REFERENCE and nothing else; that confound is NOT lifted here.
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
PREFIX = "cpk3-"
NET, DSET = "ResNet18_c100", "CIFAR100"
MS, ALPHA0, AUG = "1e-3", "1e-6", "1"
EPOCHS = 772                     # E, cts3's and cpk2's horizon, kept identical
CONTROL_EPOCHS = 100             # the in-batch, in-run control readout
BATCH, GAMMA = 100, "1"
BASE_ALG, META_ALG = "SGDm", "Lion"
SEEDS = (6, 7, 8)                # unused in this cell; see the header
N_TENSORS = 62
CLIP_C = "-15:-2.3026"           # the campaign standard; the only level run
KGRID = (45, 46, 47, 48, 49, 50)  # CONSECUTIVE; k49 strictly interior
ANCHOR = "k01"                   # m=1 scalar, the FLOOR reference only
SWEEP_ARMS = tuple("k%02d" % k for k in KGRID)
ARMS = (ANCHOR,) + SWEEP_ARMS
N_RUNS = len(ARMS) * len(SEEDS)  # 21
SLOPE_WINDOW = 20                # epochs in the terminal OLS slope window

# ---- the live-model tensor sizes.  MEASURED by the launcher's guard 4 on the
# live ResNet18_c100 before this batch was composed, and re-checked against the
# manifest at score time when --manifest is given.  They enter the PREDICTION.
TENSOR_NAME = {45: "layer3.1.bn2.bias", 46: "layer4.0.conv1.weight",
               47: "layer4.0.bn1.weight", 48: "layer4.0.bn1.bias",
               49: "layer4.0.conv2.weight", 50: "layer4.0.bn2.weight"}
TENSOR_NUMEL = {45: 256, 46: 1179648, 47: 512, 48: 512, 49: 2359296, 50: 512}
TOTAL_PARAMS = 11220132

# ---- the noise floor, RE-DERIVED FROM THE LIVE CORPUS AT REGISTRATION TIME.
SIGMA_100 = 0.917280
SIGMA_100_DF, SIGMA_100_CELLS, SIGMA_100_MEMBERS = 58, 29, 87
SIGMA_772 = 0.975496
SIGMA_772_DF, SIGMA_772_CELLS, SIGMA_772_MEMBERS = 14, 7, 21
SIGMA_W = 0.975496               # = max(SIGMA_100, SIGMA_772), the frozen rule

# ---- cts1's own in-batch cliff.  Enters ONLY through PREMISE_BAR. -----------
CTS1_PREFIX = "cts1-"
CTS1_K49 = 55.168667
CTS1_K50 = 30.334667
CTS1_CLIFF = 24.834000

# ---- cpk2's own 772-epoch arm means.  CROSS-BATCH ANCHORS for the prediction
# only; no gate uses them.  --selftest re-derives every one from the corpus.
CPK2_PREFIX = "cpk2-"
CPK2_PE = {"k01": 23.368000, "k45": 43.514000, "k47": 45.898667,
           "k49": 56.412667, "k50": 35.913333, "k52": 39.838667}

# ---- the two named hypotheses, re-derived by --selftest from CPK2_PE and
# TENSOR_NUMEL.  H-MASS is the registered point prediction.
PRED_BRANCH = "PEAK-CONFIRMED-AT-49"
PRED_ARGMAX = "k49"

# ---- the SE arithmetic.  Each contrast gets the SE of ITS OWN combination. ---
def se_of_combo(coefs, sigma=SIGMA_W, n=len(SEEDS)):
    return sigma * math.sqrt(sum(c * c for c in coefs) / float(n))


SE_ARM_DIFF = se_of_combo((1, -1))        # any two-arm difference, in batch
SE_FLOOR = se_of_combo((1, -1))           # in batch the floor IS a two-arm diff

# ---- BARS.  --selftest quotes every one in SE units. ------------------------
PREMISE_BAR = 12.4170     # R1: half of cts1's own in-batch cliff
PEAK_BAR = 1.592974       # the peak verdict: 2 * SE_ARM_DIFF
FLOOR_BAR = 1.592974      # R-FLOOR: 2 * SE_FLOOR (in batch, 3 vs 3)
NOISY_BAR = 2.926488      # R3: 3 * SIGMA_W
DEAD_BAR = 5.00           # R2
SLOPE_K50_AT_100 = 0.04847  # cts2's measurement, which set cts3's R4 bar
CONV_BAR = 0.024235       # R-CONV: half SLOPE_K50_AT_100; cts3's and cR1's bar
CONVERGED_SLOPE = 0.00426
TIGHT_BAR = 0.00426       # R-CONV: s*, the slope that defined E

EP_RE = re.compile(r"Epoch\s+(\d+).*?Test Accuracy:\s*([0-9.]+)", re.I)
EPTR_RE = re.compile(r"Epoch\s+(\d+).*?Train Accuracy:\s*([0-9.]+)", re.I)
ARGS_RE = re.compile(r"^\s*ARGS:\s*(.*)$")
ENV_RE = re.compile(r"^\s*ENV:\s*(.*)$")
NAME_RE = re.compile(r"^cpk3-(k01|k45|k46|k47|k48|k49|k50)-s([678])-(\d+)\.out$")


# =============================================================================
def kspec(k):
    return "[%d,%d]" % (k, N_TENSORS - k)


def arm_k(arm):
    return int(arm[1:])


def arm_spec(arm):
    return "scalar" if arm == ANCHOR else kspec(arm_k(arm))


# =============================================================================
# THE PREDICTION.  Re-derived by --selftest; never hand-typed as a level.
# =============================================================================
def predict():
    """H-MASS and H-EQUI, from cpk2's measured steps and the live tensor sizes.

    A multi-tensor step from cut a to cut b is attributed to the tensors
    a+1 .. b.  H-MASS gives each a share proportional to its parameter count;
    H-EQUI gives each an equal share.
    """
    s4547 = CPK2_PE["k47"] - CPK2_PE["k45"]      # tensors 46, 47
    s4749 = CPK2_PE["k49"] - CPK2_PE["k47"]      # tensors 48, 49
    m4547 = TENSOR_NUMEL[46] + TENSOR_NUMEL[47]
    m4749 = TENSOR_NUMEL[48] + TENSOR_NUMEL[49]
    mass = {"k46": CPK2_PE["k45"] + s4547 * TENSOR_NUMEL[46] / float(m4547),
            "k48": CPK2_PE["k47"] + s4749 * TENSOR_NUMEL[48] / float(m4749)}
    equi = {"k46": CPK2_PE["k45"] + s4547 / 2.0,
            "k48": CPK2_PE["k47"] + s4749 / 2.0}
    for a in ("k45", "k47", "k49", "k50"):
        mass[a] = CPK2_PE[a]
        equi[a] = CPK2_PE[a]
    mass["k01"] = equi["k01"] = CPK2_PE["k01"]
    return mass, equi, s4547, s4749


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
    epochs, so budget=772 reads 767..771 and budget=100 reads 95..99.  The CSV
    `plateau` column is BANNED as a primary (standing rule) and is never read;
    `best_test` is a maximum over a run and is not a plateau at all.  Using the
    IDENTICAL estimator at both horizons is what makes the within-run gain
    d = M(E) - M(100) free of any estimator difference, and it is byte-for-byte
    the estimator cts3, cR1 and cS1 all used."""
    v = [series[e] for e in range(budget - w, budget) if e in series]
    return sum(v) / len(v) if len(v) == w else None


def ols_slope(series, budget, w=SLOPE_WINDOW):
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
                         "env": {}, "args": {}, "clip": None, "seed": None,
                         "envline": ""})
            continue
        r["base"] = base
        r["bad"] = None
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
# the corpus.  Used ONLY by --selftest.  score() never opens the CSV.
# =============================================================================
def _csv_rows(path=CSV):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if (r.get("superseded") or "0") != "1"]


def _batch_of(run):
    return re.split(r"[-_]", str(run))[0]


def _in_cell(r, epochs=CONTROL_EPOCHS):
    return (r["network"] == NET and r["dataset"] == DSET
            and r["base"] == BASE_ALG and r["meta"] == META_ALG
            and r["meta_stepsize"] == MS and r["alpha0"] == ALPHA0
            and r["gamma"] == GAMMA and r["augment"] == AUG
            and r["beta_clip"] == CLIP_C and not (r.get("hier") or "").strip()
            and r["batch_size"] == str(BATCH)
            and r["epochs_done"] == str(epochs)
            and r["collapsed"] == "0")


M2_RE = re.compile(r"^\[\d+,\d+\]$")


def noise_floor_from_csv(path=CSV, epochs=CONTROL_EPOCHS):
    cells = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not _in_cell(r, epochs) or not M2_RE.match(str(r["granularity"] or "")):
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


def cpk2_levels_from_csv(path=CSV):
    g = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not str(r.get("run") or "").startswith(CPK2_PREFIX):
            continue
        try:
            g[str(r["granularity"] or "")].append(float(r["plateau5"]))
        except (TypeError, ValueError):
            pass
    out = {}
    for a, spec in (("k01", "scalar"), ("k45", kspec(45)), ("k47", kspec(47)),
                    ("k49", kspec(49)), ("k50", kspec(50)), ("k52", kspec(52))):
        if spec in g:
            out[a] = (statistics.mean(g[spec]), len(g[spec]))
    return out


def count_cpk3_rows(path=CSV):
    return sum(1 for r in _csv_rows(path)
               if str(r.get("run") or "").startswith(PREFIX))


def unrun_cuts_in_corpus(path=CSV):
    """Every corpus row anywhere, any cell, any horizon, at [k, 62-k] for the
    two cuts this batch exists to measure."""
    out = {}
    rows = _csv_rows(path)
    for k in (44, 46, 48):
        out[k] = [r["run"] for r in rows
                  if str(r.get("granularity") or "") == kspec(k)]
    return out


def seeds_in_cell(path=CSV):
    s = set()
    for r in _csv_rows(path):
        if _in_cell(r, CONTROL_EPOCHS) or _in_cell(r, EPOCHS):
            s.add(str(r.get("seed")))
    return sorted(s)


def horizons_on_net(path=CSV):
    out = set()
    for r in _csv_rows(path):
        if r.get("network") != NET:
            continue
        e = str(r.get("epochs_done") or "")
        if e.isdigit() and int(e) > CONTROL_EPOCHS:
            out.add(int(e))
    return sorted(out)


def read_manifest(path):
    """The launcher's live-model manifest.  A DOCUMENTED optional argument."""
    sizes, names = {}, {}
    total = None
    try:
        for line in open(path):
            t = line.split()
            if not t:
                continue
            if t[0] == "TOTAL_PARAMS":
                total = int(t[1])
            if t[0] == "TENSOR" and len(t) >= 4:
                sizes[int(t[1])] = int(t[3])
                names[int(t[1])] = t[2]
    except IOError:
        return None
    return {"numel": sizes, "name": names, "total": total}


# =============================================================================
def fmt(x, w=9, p=4):
    return (" " * w) if x is None else ("%*.*f" % (w, p, x))


# =============================================================================
# --selftest.  Everything the registration asserts is RE-DERIVED here, from the
# corpus and from arithmetic, and checked against THIS FILE'S constants.
#
# THE RULE 21 PREMISE IS CORPUS-CONDITIONAL, DELIBERATELY.  CORRECTIONS 156.9
# recorded that cfr1's and cts3's premise checks become KNOWN-FALSE assertions
# the moment their own rows land -- a scorer that must FAIL after a successful
# ingest teaches the next reader to ignore its own FAILs.  cR1 was the first
# scorer in this campaign that did NOT acquire one on its own ingest, and THAT
# is the pattern copied here: the premise admits exactly TWO states, 0 rows
# (pre-registration) or exactly N_RUNS rows (post-ingest), and FAILS on
# anything else -- a partial ingest, a duplicate, or a name collision before
# launch, which are the states that actually matter.  The ORDERING claim RULE
# 21 really makes is not checkable from inside a python file at all; it is
# proved by wall clock, git commit time against the earliest sacct Submit, and
# that proof lives in the CORRECTIONS entry, not here.
# =============================================================================
def selftest():
    ok = True

    def chk(cond, label, got="", want=""):
        nonlocal ok
        print("  %-4s %-58s %s%s" % ("PASS" if cond else "FAIL", label,
                                     got, (" (want %s)" % want) if want else ""))
        if not cond:
            ok = False

    print("cS2_cpk3_score.py --selftest")
    print("-" * 78)

    print("A. the grid, the arms and the run count")
    chk(SWEEP_ARMS == ("k45", "k46", "k47", "k48", "k49", "k50"),
        "sweep arms", " ".join(SWEEP_ARMS))
    for k in KGRID:
        chk(arm_spec("k%02d" % k) == kspec(k), "spec for k=%d" % k, kspec(k))
    chk(arm_spec(ANCHOR) == "scalar", "the anchor is m=1 scalar", arm_spec(ANCHOR))
    chk(all(sum(int(x) for x in kspec(k)[1:-1].split(",")) == N_TENSORS
            for k in KGRID), "every spec covers all %d tensors" % N_TENSORS)
    chk(N_RUNS == 21, "run count", str(N_RUNS), "21")
    chk(SEEDS == (6, 7, 8), "seeds", str(SEEDS))
    chk(EPOCHS == 772 and CONTROL_EPOCHS == 100, "horizons",
        "%d / %d" % (EPOCHS, CONTROL_EPOCHS))
    chk(list(KGRID) == list(range(min(KGRID), max(KGRID) + 1)),
        "the grid is CONSECUTIVE -- every step moves exactly ONE tensor",
        str(list(KGRID)))
    chk(49 in KGRID and min(KGRID) < 49 < max(KGRID),
        "the incumbent argmax k*=49 is STRICTLY INTERIOR (not at a grid edge)",
        "left %s right %s" % ([k for k in KGRID if k < 49],
                              [k for k in KGRID if k > 49]))

    print("B. the live-model tensor sizes that enter the PREDICTION")
    chk(TENSOR_NUMEL[46] == 1179648 and TENSOR_NUMEL[49] == 2359296,
        "the two large convs", "t46 %d  t49 %d" % (TENSOR_NUMEL[46],
                                                   TENSOR_NUMEL[49]))
    chk(TENSOR_NUMEL[47] == 512 and TENSOR_NUMEL[48] == 512
        and TENSOR_NUMEL[50] == 512,
        "the three layer4.0 batch-norm parameters are 512 each",
        "t47/t48/t50 = %d/%d/%d" % (TENSOR_NUMEL[47], TENSOR_NUMEL[48],
                                    TENSOR_NUMEL[50]))
    for k in sorted(TENSOR_NAME):
        print("       tensor %2d  %-26s %9d params  (%.4f%% of the model)"
              % (k, TENSOR_NAME[k], TENSOR_NUMEL[k],
                 100.0 * TENSOR_NUMEL[k] / TOTAL_PARAMS))
    print("       MASS IS NOT A COMPLETE LAW, and the corpus already proves it:")
    print("       group 1 at k=49 and k=50 differ by %d of %d parameters"
          % (TENSOR_NUMEL[50], TOTAL_PARAMS))
    print("       (%.5f%%) and the two arms are %.4f pp apart."
          % (100.0 * TENSOR_NUMEL[50] / TOTAL_PARAMS,
             CPK2_PE["k49"] - CPK2_PE["k50"]))

    print("C. the noise floor and cts1's cliff, re-derived from the corpus")
    if not os.path.exists(CSV):
        chk(False, "results/all_runs.csv present", CSV)
    else:
        s1, d1, c1, m1 = noise_floor_from_csv(CSV, CONTROL_EPOCHS)
        chk(abs(s1 - SIGMA_100) < 5e-5, "SIGMA_100 reproduces", "%.6f" % s1,
            "%.6f" % SIGMA_100)
        chk(d1 == SIGMA_100_DF and c1 == SIGMA_100_CELLS
            and m1 == SIGMA_100_MEMBERS,
            "SIGMA_100 df / cells / members reproduce",
            "%d / %d / %d" % (d1, c1, m1),
            "%d / %d / %d" % (SIGMA_100_DF, SIGMA_100_CELLS, SIGMA_100_MEMBERS))
        s2, d2, c2, m2 = noise_floor_from_csv(CSV, EPOCHS)
        chk(abs(s2 - SIGMA_772) < 5e-5, "SIGMA_772 reproduces", "%.6f" % s2,
            "%.6f" % SIGMA_772)
        chk(d2 == SIGMA_772_DF and c2 == SIGMA_772_CELLS
            and m2 == SIGMA_772_MEMBERS,
            "SIGMA_772 df / cells / members reproduce",
            "%d / %d / %d" % (d2, c2, m2),
            "%d / %d / %d" % (SIGMA_772_DF, SIGMA_772_CELLS, SIGMA_772_MEMBERS))
        chk(abs(SIGMA_W - max(s1, s2)) < 5e-5,
            "SIGMA_W IS the max of the two, by the frozen rule",
            "%.6f" % max(s1, s2), "%.6f" % SIGMA_W)
        print("       the 772 stratum is the horizon-matched one AND the larger")
        print("       (%.6f > %.6f), so the choice is conservative either way"
              % (s2, s1))
        c = cts1_cliff_from_csv()
        chk(c is not None, "cts1 arms present in the corpus")
        if c:
            chk(abs(c[0] - CTS1_K49) < 5e-5, "cts1 k49 reproduces", "%.6f" % c[0])
            chk(abs(c[1] - CTS1_K50) < 5e-5, "cts1 k50 reproduces", "%.6f" % c[1])
            chk(abs(c[2] - CTS1_CLIFF) < 5e-5, "cts1 cliff reproduces",
                "%.6f" % c[2])
            chk(abs(PREMISE_BAR - c[2] / 2.0) < 5e-4,
                "PREMISE_BAR is half cts1's own cliff", "%.4f" % (c[2] / 2.0))

        print("D. the cross-batch anchors, re-derived from cpk2's own rows")
        lv = cpk2_levels_from_csv()
        for a in sorted(CPK2_PE):
            if a not in lv:
                chk(False, "cpk2 %s present" % a, "absent")
                continue
            mu, n = lv[a]
            chk(abs(mu - CPK2_PE[a]) < 5e-5,
                "cpk2 %s at 772 reproduces (n=%d)" % (a, n), "%.6f" % mu,
                "%.6f" % CPK2_PE[a])

        print("E. THE PREMISE OF THE WHOLE BATCH: k=46 and k=48 are UNRUN")
        u = unrun_cuts_in_corpus()
        for k in (46, 48):
            chk(len(u[k]) == 0,
                "the corpus holds ZERO rows at %s, ANY cell, ANY horizon"
                % kspec(k), "%d found" % len(u[k]), "0")
        chk(len(u[44]) == 0,
            "k=44 is also unrun (DECLINED, see the header, not an oversight)",
            "%d found" % len(u[44]), "0")
        sd = seeds_in_cell()
        chk(all(str(s) not in sd for s in SEEDS),
            "seeds %s appear NOWHERE in this cell -> zero duplicate rows"
            % str(SEEDS), "cell seeds: %s" % ",".join(sd))

        print("F. RULE 21 premise and the horizon census -- CORPUS-CONDITIONAL")
        n = count_cpk3_rows()
        chk(n in (0, N_RUNS),
            "cpk3 rows in the corpus are 0 (pre-reg) or exactly %d" % N_RUNS,
            "%d found -> %s" % (n, "PRE-REGISTRATION" if n == 0
                                else ("POST-INGEST" if n == N_RUNS else "PARTIAL")),
            "0 or %d" % N_RUNS)
        hz = horizons_on_net()
        chk(hz in ([], [EPOCHS]),
            "above-100 horizons on %s are {} or {%d}" % (NET, EPOCHS),
            str(hz), "[] or [%d]" % EPOCHS)

    print("G. the SE arithmetic and every bar in SE units")
    chk(abs(SE_ARM_DIFF - SIGMA_W * math.sqrt(2.0 / 3.0)) < 1e-9,
        "SE_ARM_DIFF", "%.6f" % SE_ARM_DIFF)
    chk(abs(SE_FLOOR - SE_ARM_DIFF) < 1e-12,
        "SE_FLOOR == SE_ARM_DIFF: the floor is an IN-BATCH 3-vs-3 contrast, "
        "so it is WIDER than cR1's 1.1488", "%.6f" % SE_FLOOR)
    chk(abs(PEAK_BAR - 2 * SE_ARM_DIFF) < 5e-6, "PEAK_BAR = 2*SE_ARM_DIFF",
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
    print("       PREMISE  %8.4f pp = %6.2f SE_ARM_DIFF"
          % (PREMISE_BAR, PREMISE_BAR / SE_ARM_DIFF))
    print("       PEAK     %8.4f pp = %6.2f SE_ARM_DIFF"
          % (PEAK_BAR, PEAK_BAR / SE_ARM_DIFF))
    print("       FLOOR    %8.4f pp = %6.2f SE_FLOOR"
          % (FLOOR_BAR, FLOOR_BAR / SE_FLOOR))
    print("       NOISY    %8.4f pp = %6.2f SIGMA_W"
          % (NOISY_BAR, NOISY_BAR / SIGMA_W))

    print("H. THE PRE-REGISTERED POINT PREDICTION, re-derived from arithmetic")
    mass, equi, s4547, s4749 = predict()
    print("       cpk2's measured two-tensor steps at 772:")
    print("         45 -> 47  %+8.4f pp   over tensors 46 (%d) + 47 (%d)"
          % (s4547, TENSOR_NUMEL[46], TENSOR_NUMEL[47]))
    print("         47 -> 49  %+8.4f pp   over tensors 48 (%d) + 49 (%d)"
          % (s4749, TENSOR_NUMEL[48], TENSOR_NUMEL[49]))
    print("       arm     H-MASS (registered)   H-EQUI (the named alternative)")
    for a in ARMS:
        print("       %-5s %14.4f %22.4f" % (a, mass[a], equi[a]))
    chk(abs(mass["k46"] - 45.897633) < 5e-5, "H-MASS k46",
        "%.6f" % mass["k46"], "45.897633")
    chk(abs(mass["k48"] - 45.900948) < 5e-5, "H-MASS k48",
        "%.6f" % mass["k48"], "45.900948")
    chk(abs(equi["k46"] - 44.706333) < 5e-5, "H-EQUI k46",
        "%.6f" % equi["k46"], "44.706333")
    chk(abs(equi["k48"] - 51.155667) < 5e-5, "H-EQUI k48",
        "%.6f" % equi["k48"], "51.155667")
    sep = abs(equi["k48"] - mass["k48"])
    chk(sep > PEAK_BAR,
        "the two hypotheses are SEPARABLE by this batch on k48",
        "%.4f pp = %.2f SE_ARM_DIFF" % (sep, sep / SE_ARM_DIFF))
    sw = {a: mass[a] for a in SWEEP_ARMS}
    best = max(sw, key=sw.get)
    second = sorted(sw, key=sw.get)[-2]
    chk(best == PRED_ARGMAX, "the predicted argmax under H-MASS", best,
        PRED_ARGMAX)
    marg = sw[best] - sw[second]
    print("       => PREDICTED BRANCH %s, margin %.4f pp = %.2f SE_ARM_DIFF"
          % (PRED_BRANCH, marg, marg / SE_ARM_DIFF))
    print("       to RELOCATE the peak an arm must exceed k49; that requires")
    for a in ("k46", "k48"):
        print("         %-4s to gain %8.4f pp over its H-MASS prediction"
              % (a, sw[PRED_ARGMAX] - mass[a]))
    print("       H-MASS satisfies the standing 'k48 in [45.8987, 56.4127]'")
    print("       prediction AT ITS LOWER ENDPOINT, not in the middle.")

    print("I. what the FIVE single-tensor steps will be measured against")
    print("       step     tensor moved into group 1                 params")
    for a, b in zip(KGRID, KGRID[1:]):
        print("       %2d -> %2d  %-34s %9d" % (a, b, TENSOR_NAME[b],
                                                TENSOR_NUMEL[b]))
    print("       and, CROSS-BATCH and descriptive only, cpk2 measured the")
    print("       49 -> 50 aggregate at %+8.4f pp"
          % (CPK2_PE["k50"] - CPK2_PE["k49"]))

    print("-" * 78)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# =============================================================================
def score(runsdir, manifest=None):
    recs = collect(runsdir)
    mass, equi, s4547, s4749 = predict()
    print("=" * 78)
    print("cpk3 -- WHERE IS THE CUT-POSITION PEAK?  THE CONSECUTIVE GRID 45..50")
    print("        E = %d epochs, control at %d, seeds %s"
          % (EPOCHS, CONTROL_EPOCHS, ",".join(str(s) for s in SEEDS)))
    print("scorer: %s   (registered before any cpk3 run existed)"
          % os.path.basename(__file__))
    print("=" * 78)

    # ---- G0 provenance ------------------------------------------------------
    print("\nG0 PROVENANCE")
    fail = []
    if len(recs) != N_RUNS:
        fail.append("expected %d runs, found %d" % (N_RUNS, len(recs)))
    jobs = {r.get("job_id") for r in recs if r.get("job_id")}
    if len(jobs) != len(recs):
        fail.append("job ids not distinct: %d ids for %d runs"
                    % (len(jobs), len(recs)))
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
            fail.append("%s: ENV PROBE %r, registered 0"
                        % (b, r["env"].get("PROBE")))
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
                fail.append("%s: ARGS %s = %r, registered %r"
                            % (b, flag, got, want))
        if r["n_ep"] != EPOCHS:
            fail.append("%s: %d epoch lines, registered %d" % (b, r["n_ep"], EPOCHS))
    for f in fail:
        print("  FAIL  %s" % f)
    if fail:
        print("\nVERDICT: UNRESOLVED-PROVENANCE")
        return 3
    print("  PASS  %d runs, %d job ids, %d/%d epochs, NAME == ARGS == ENV, no "
          "repeated flag" % (len(recs), len(jobs), EPOCHS, EPOCHS))
    print("  PASS  %d distinct ENV line(s) across the batch"
          % len({r["envline"] for r in recs}))

    if manifest:
        mf = read_manifest(manifest)
        print("\nTHE LIVE-MODEL MANIFEST (--manifest, a documented argument)")
        if mf is None:
            print("  NOTE  manifest unreadable at %s" % manifest)
        else:
            bad = [k for k in TENSOR_NUMEL
                   if k in mf["numel"] and mf["numel"][k] != TENSOR_NUMEL[k]]
            print("  TOTAL_PARAMS %s (registered %d)"
                  % (mf["total"], TOTAL_PARAMS))
            for k in sorted(TENSOR_NAME):
                print("  tensor %2d  %-26s %9s  registered %9d  %s"
                      % (k, mf["name"].get(k, "?"), mf["numel"].get(k, "?"),
                         TENSOR_NUMEL[k],
                         "MISMATCH" if k in bad else "ok"))
            if bad:
                print("  FAIL  the live model does not match the registration")
                print("\nVERDICT: UNRESOLVED-PROVENANCE")
                return 3

    cell = collections.defaultdict(list)
    for r in recs:
        cell[r["meas_arm"]].append(r)
    for a in ARMS:
        if len(cell[a]) != len(SEEDS):
            print("  FAIL  arm %s has %d runs, want %d" % (a, len(cell[a]),
                                                           len(SEEDS)))
            print("\nVERDICT: UNRESOLVED-PROVENANCE")
            return 3

    def M(a, k):
        v = [r[k] for r in cell[a] if r.get(k) is not None]
        return statistics.mean(v) if v else None

    def SD(a, k):
        v = [r[k] for r in cell[a] if r.get(k) is not None]
        return statistics.stdev(v) if len(v) > 1 else None

    # ---- R2 divergence ------------------------------------------------------
    print("\nR2 DIVERGENCE")
    dead = [r["base"] for r in recs
            if r["p5_E"] is None or not math.isfinite(r["p5_E"])
            or r["p5_E"] <= DEAD_BAR]
    if dead:
        for d in dead:
            print("  FAIL  %s is DEAD (<= %.2f pp at E)" % (d, DEAD_BAR))
        print("\nVERDICT: UNRESOLVED-DIVERGED")
        return 1
    print("  PASS  %d/%d runs alive at E (bar %.2f pp)" % (len(recs), N_RUNS,
                                                           DEAD_BAR))

    # ---- R3 noise -----------------------------------------------------------
    print("\nR3 NOISE  (bar %.4f pp = 3 * SIGMA_W)" % NOISY_BAR)
    worst, worsta = -1.0, None
    for a in ARMS:
        s = SD(a, "p5_E")
        if s is not None and s > worst:
            worst, worsta = s, a
        print("  %-5s SD at E %8.4f   SD at 100 %8.4f" % (a, s or float("nan"),
                                                          SD(a, "p5_100") or float("nan")))
    if worst > NOISY_BAR:
        print("  FAIL  worst cell SD %.4f (%s) exceeds %.4f" % (worst, worsta,
                                                                NOISY_BAR))
        print("\nVERDICT: UNRESOLVED-NOISY")
        return 1
    print("  PASS  worst cell SD %.4f (%s) <= %.4f" % (worst, worsta, NOISY_BAR))

    # ---- the curves ---------------------------------------------------------
    print("\nTHE CURVES.  TEST AND TRAIN, BOTH HORIZONS, EVERY ARM.")
    print("  arm  spec        test@100    test@E      gain   train@100"
          "   train@E     tgain   slope@100    slope@E")
    for a in ARMS:
        print("  %-5s %-10s %s %s %s %s %s %s %s %s"
              % (a, arm_spec(a), fmt(M(a, "p5_100"), 10), fmt(M(a, "p5_E"), 10),
                 fmt((M(a, "p5_E") - M(a, "p5_100")), 9),
                 fmt(M(a, "t5_100"), 10), fmt(M(a, "t5_E"), 10),
                 fmt((M(a, "t5_E") - M(a, "t5_100")), 9),
                 fmt(M(a, "slope_100"), 11, 5), fmt(M(a, "slope_E"), 11, 5)))
    print("  k46 and k48 above are the FIRST measurements of those cuts at ANY")
    print("  horizon anywhere in this corpus.")

    # ---- R1 premise: the cliff, in batch, at 100 ----------------------------
    print("\nR1 PREMISE -- THE CLIFF, IN BATCH, AT %d EPOCHS" % CONTROL_EPOCHS)
    d100 = M("k49", "p5_100") - M("k50", "p5_100")
    print("  D100 = M(k49) - M(k50) = %.4f pp = %.2f SE_ARM_DIFF   bar %.4f"
          % (d100, d100 / SE_ARM_DIFF, PREMISE_BAR))
    if d100 < PREMISE_BAR:
        print("  FAIL  the cliff is not present in this batch's own control")
        print("\nVERDICT: UNRESOLVED-PREMISE")
        return 1
    print("  PASS")
    dE = M("k49", "p5_E") - M("k50", "p5_E")
    print("  (reported, not gated) the same cliff at E = %.4f pp = %.2f SE;"
          % (dE, dE / SE_ARM_DIFF))
    print("   dD = %.4f pp.  cpk2 measured DE 20.4993 and cts3 19.3427 -- "
          "CROSS-BATCH," % (dE - d100))
    print("   DESCRIPTIVE, NOT POOLED.  BATCH is the unit of replication.")

    # ---- R-CTRL: the seed-set reproduction control --------------------------
    print("\nR-CTRL PREMISE -- SEED-SET REPRODUCTION, IN BATCH, AT %d EPOCHS"
          % CONTROL_EPOCHS)
    print("  (k46 and k48 are NOT in this gate, by construction -- asserting")
    print("   anything about them at 100 would prejudge the measurement.)")
    a49, a47, a45 = M("k49", "p5_100"), M("k47", "p5_100"), M("k45", "p5_100")
    print("  k49 %.4f  >  k47 %.4f  >  k45 %.4f" % (a49, a47, a45))
    print("  k49 - k47 = %.4f pp = %.2f SE_ARM_DIFF   bar %.4f"
          % (a49 - a47, (a49 - a47) / SE_ARM_DIFF, PEAK_BAR))
    if not (a49 > a47 > a45 and (a49 - a47) > PEAK_BAR):
        print("  FAIL  seeds %s do not reproduce the cell's known 100-epoch "
              "order" % str(SEEDS))
        print("\nVERDICT: UNRESOLVED-CONTROL")
        return 1
    print("  PASS")

    # ---- R-FLOOR ------------------------------------------------------------
    print("\nR-FLOOR SATURATION -- IN BATCH, against THIS batch's own k01,")
    print("  at BOTH horizons.  bar %.4f pp = 2 * SE_FLOOR (3-seed vs 3-seed)"
          % FLOOR_BAR)
    fl_E, fl_100 = M(ANCHOR, "p5_E"), M(ANCHOR, "p5_100")
    print("  this batch's own m=1 floor:  at 100 %.4f   at E %.4f   "
          "(moved %+.4f pp = %.2f SE)"
          % (fl_100, fl_E, fl_E - fl_100, (fl_E - fl_100) / SE_ARM_DIFF))
    print("  arm    above floor @100   informative?   above floor @E   "
          "informative?")
    informative = []
    for a in SWEEP_ARMS:
        a1 = M(a, "p5_100") - fl_100
        aE = M(a, "p5_E") - fl_E
        i1, iE = a1 > FLOOR_BAR, aE > FLOOR_BAR
        if iE:
            informative.append(a)
        print("  %-5s %+16.4f   %-13s %+15.4f   %s"
              % (a, a1, "YES" if i1 else "NO ", aE, "YES" if iE else "NO "))
    if PRED_ARGMAX not in informative:
        print("  FAIL  %s itself is NOT informative at E" % PRED_ARGMAX)
        print("\nVERDICT: UNRESOLVED-SATURATED")
        return 1
    print("  INFORMATIVE AT E: %s" % ", ".join(informative))
    if len(informative) < len(SWEEP_ARMS):
        print("  EXCLUDED from the argmax set: %s"
              % ", ".join(a for a in SWEEP_ARMS if a not in informative))

    # ---- R-CONV -------------------------------------------------------------
    print("\nR-CONV HORIZON EFFECTIVENESS  (CONV_BAR %.6f, TIGHT_BAR %.5f)"
          % (CONV_BAR, TIGHT_BAR))
    print("  arm    slope@100     slope@E    converged?   tight?")
    notconv = []
    for a in ARMS:
        se = M(a, "slope_E")
        c = abs(se) <= CONV_BAR
        t = abs(se) <= TIGHT_BAR
        if not c and a in informative:
            notconv.append(a)
        print("  %-5s %+11.5f %+11.5f   %-11s %s"
              % (a, M(a, "slope_100"), se, "YES" if c else "**NO**",
                 "YES" if t else "no"))

    # ---- the FIVE single-tensor steps --------------------------------------
    print("\nTHE FIVE CONSECUTIVE SINGLE-TENSOR STEPS, ALL IN BATCH")
    print("  step     tensor moved into group 1              params"
          "     step@100     step@E    step@E in SE")
    for x, y in zip(KGRID, KGRID[1:]):
        ax, ay = "k%02d" % x, "k%02d" % y
        s1 = M(ay, "p5_100") - M(ax, "p5_100")
        sE = M(ay, "p5_E") - M(ax, "p5_E")
        print("  %2d -> %2d  %-30s %9d %+12.4f %+10.4f %12.2f"
              % (x, y, TENSOR_NAME[y], TENSOR_NUMEL[y], s1, sE,
                 sE / SE_ARM_DIFF))
    print("  cpk2's two-tensor aggregates at 772, CROSS-BATCH and DESCRIPTIVE:")
    print("    45 -> 47 %+8.4f   47 -> 49 %+8.4f   49 -> 50 %+8.4f"
          % (s4547, s4749, CPK2_PE["k50"] - CPK2_PE["k49"]))

    # ---- the verdict --------------------------------------------------------
    sw = {a: M(a, "p5_E") for a in informative}
    best = max(sw, key=sw.get)
    second = sorted(sw, key=sw.get)[-2] if len(sw) > 1 else None
    margin = (sw[best] - sw[second]) if second else float("inf")
    if second is not None and margin <= PEAK_BAR:
        v = "PEAK-PLATEAU"
    elif best == "k49":
        v = "PEAK-CONFIRMED-AT-49"
    else:
        v = "PEAK-RELOCATES-TO-%d" % arm_k(best)

    print("\nTHE PEAK AT E, over the INFORMATIVE sweep arms")
    for a in sorted(sw, key=sw.get, reverse=True):
        print("  %-5s %10.4f" % (a, sw[a]))
    print("  best %s over %s by %.4f pp = %.2f SE_ARM_DIFF   bar %.4f"
          % (best, second, margin, margin / SE_ARM_DIFF if second else 0.0,
             PEAK_BAR))

    print("\nPRE-REGISTERED PREDICTION vs MEASURED")
    print("  predicted branch %s, predicted argmax %s" % (PRED_BRANCH,
                                                          PRED_ARGMAX))
    print("  measured  branch %s, measured  argmax %s" % (v, best))
    print("  NOTE the predicted LEVELS are cross-batch and cross-seed anchors")
    print("  (cpk2, seeds 3/4/5); the BRANCH above is decided entirely in batch.")
    print("  %-5s %10s %10s %10s %10s %10s"
          % ("arm", "H-MASS", "H-EQUI", "measured", "res(MASS)", "res(EQUI)"))
    for a in ARMS:
        m = M(a, "p5_E")
        print("  %-5s %10.4f %10.4f %10.4f %+10.4f %+10.4f"
              % (a, mass[a], equi[a], m, m - mass[a], m - equi[a]))
    dm = sum(abs(M(a, "p5_E") - mass[a]) for a in SWEEP_ARMS) / len(SWEEP_ARMS)
    de = sum(abs(M(a, "p5_E") - equi[a]) for a in SWEEP_ARMS) / len(SWEEP_ARMS)
    print("  mean |residual| over the sweep arms:  H-MASS %.4f   H-EQUI %.4f"
          % (dm, de))
    print("  -> the closer hypothesis is %s.  DESCRIPTIVE: both carry a batch"
          % ("H-MASS" if dm < de else "H-EQUI"))
    print("     effect, and neither was fitted on cpk3.")

    print("\n" + "=" * 78)
    print("VERDICT (the PEAK at %d epochs): %s" % (EPOCHS, v))
    if notconv:
        print("VERDICT (convergence):           UNRESOLVED-NOT-CONVERGED (%s)"
              % ",".join(notconv))
        print("  => the batch says NOTHING about the CONVERGED peak location.")
        print("FINAL: %s AT %d EPOCHS | UNRESOLVED-NOT-CONVERGED" % (v, EPOCHS))
    else:
        print("VERDICT (convergence):           ALL INFORMATIVE ARMS CONVERGED")
        print("FINAL: %s | CONVERGED" % v)
    print("=" * 78)
    print("SCOPE: the measured bracket is k = 45..50 ONLY.  k <= 44 and k >= 51")
    print("are UNMEASURED at this horizon, so no claim is made that the curve is")
    print("monotone on 44 -> 49 or that k* is the argmax over any wider range.")
    print("NO m claim, NO CAPTURE, NO asymptote, NO pooled estimate across")
    print("cpk2/cts3/cpk3, NO mechanism for any step, and the `scalar` anchor's")
    print("separate code path is NOT lifted.")
    return 0


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="cS2 cpk3 scorer")
    ap.add_argument("runsdir", nargs="?",
                    help="directory holding the cpk3-*.out files")
    ap.add_argument("--manifest", default=None,
                    help="runs/cpk3/PARTITION-MANIFEST.txt, to re-check the "
                         "live tensor sizes against the registration.  A "
                         "DOCUMENTED argument: passing it is not an edit.")
    ap.add_argument("--selftest", action="store_true",
                    help="re-derive every registered constant and exit")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.runsdir:
        ap.error("runsdir is required unless --selftest is given")
    return score(a.runsdir, a.manifest)


if __name__ == "__main__":
    sys.exit(main())
