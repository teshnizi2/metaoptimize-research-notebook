#!/usr/bin/env python3
# =============================================================================
# cX1_crn1_score.py -- THE REGISTERED SCORER FOR `crn1`.
#
# STANDING RULE 21: this file is git-committed BEFORE the batch is submitted and
# is run UNEDITED afterwards (RULE 16).  Every bar, the noise floor, every named
# account's point predictions, the branch map, the floor census and the
# PRE-REGISTRATION MEASUREMENT are FROZEN here as literals, and `--selftest`
# re-derives every one of them from the live corpus and from arithmetic rather
# than from prose.
#
# DOCUMENTED INVOCATION -- there is no second argument to forget:
#     python3 analysis/cX1_crn1_score.py <runsdir>
# `--manifest` is OPTIONAL and DEFAULTED (CORRECTIONS 164.2); the launcher's
# guard 8 imports this file and proves the default resolves before any job can
# finish.
#
# -----------------------------------------------------------------------------
# 1. THE FOURTEENTH MECHANISM, AND WHAT THE LIVE SOURCE ACTUALLY SAYS
# -----------------------------------------------------------------------------
# Thirteen candidate mechanisms are dead, plus co-grouping, parameter mass and
# contiguity-as-such.  Every one was an OBSERVATIONAL statistic or a post-hoc
# operator on beta.  NOT ONE intervened on the REDUCTION -- the step where the
# harness turns 62 per-tensor inner products into one number per group.  Quoted
# from the LIVE file (sha256 0d8ee431a8caa51c14fa632776349db32b42e346299a91b69c4c6680142a3892
# before PATCH_REDNORM):
#
#   HF.py:259  def block_product(self, u, v):
#   HF.py:260      if self.stepsize_type == 'scalar':
#   HF.py:261          return [sum([(u_*v_).sum() for u_,v_ in zip(u,v)])]
#   HF.py:262      if self.stepsize_type == 'blockwise':
#   HF.py:263          return [torch.tensor([sum([(u[i]*v[i]).sum()
#                          for i in group_indices])
#                          for group_indices in self.param_groups_indices])]
#   HF.py:957  def Lion_meta_update(self,HtT_gradft):
#   HF.py:958      for i in range(self.len_beta_list):
#   HF.py:959          self.beta[i] = (1-ms*wd)*self.beta[i]
#                          - ms*torch.sign(b2*self.momentum_meta[i]
#                                          + (1-b2)*HtT_gradft[i])
#   HF.py:960          self.momentum_meta[i] = mom*self.momentum_meta[i]
#                          + (1-mom)*HtT_gradft[i]
#
# So the group reduction IS an UNNORMALISED SUM, |dbeta| IS exactly
# meta_stepsize, and the reduction's MAGNITUDE is discarded: only its SIGN
# survives.  That part of the briefing's account is CONFIRMED on the source.
#
# -----------------------------------------------------------------------------
# 2. FOUR THINGS MEASURED BEFORE ANY GPU-HOUR WAS SPENT ON AN ARM.  THEY CHANGED
#    THE DESIGN, AND TWO OF THEM ARE THEMSELVES DELIVERABLES.
# -----------------------------------------------------------------------------
# (i) "NORMALISE THE GROUP REDUCTION" IS A NULL BY CONSTRUCTION AND IS NOT RUN.
#     `momentum_meta` is a ZERO-INITIALISED LINEAR recursion in the reduction, so
#     dividing a GROUP's reduction by ANY positive per-group constant -- its
#     tensor count, its parameter count, anything fixed in time -- divides the
#     whole sign() argument by that constant and leaves the sign, hence the
#     ENTIRE beta trajectory, unchanged.  `analysis/cX1_reduction_noop_proof.py`
#     drives the LIVE `Lion_meta_update` with a 4,000-step z-history and its
#     rescaled twin and finds the beta trajectories BITWISE IDENTICAL at
#     c = 1, 62, 49/13, 1e-3/7 and 6315072/4905060.  (RMSProp and Adam meta are
#     invariant only above their hard-coded epsilon = 1e-10; the campaign cell
#     is `--alg-meta Lion`, read off every ARGS line.)
#
# (ii) THE SUM'S SIGN IS NOT BEING HIJACKED BY THE LARGE TENSORS.
#     `analysis/cX1_reduction_probe.py` wrapped the REAL optimizer at THIS EXACT
#     cell and recorded all 62 per-tensor terms t_i = <h_condenced[i], g[i]> at
#     EVERY meta-step for 12 epochs = 6,000 steps, at `scalar`, `[49,13]` and
#     `[50,12]`.  A pure EQUAL-VOTE-PER-TENSOR reduction (sum of signs, all size
#     information discarded) gives the SAME sign as the unnormalised sum in
#     99.2 % to 99.9 % of steps at every group of every configuration.
#
# (iii) THE CUT-POSITION CLIFF IS NOT A REDUCTION EFFECT.  In the `[50,12]`
#     COARSE group, REMOVING `layer4.0.bn2.weight` (tensor 50, 512 params)
#     flips that group's sign in 0.0003 of steps.  Three meta-steps in ten
#     thousand.  Yet moving that one tensor across the cut costs 25.068667 pp.
#     A quantity that changes 0.03 % of the group's sign decisions cannot be
#     producing a 25 pp effect, so the cliff acts through WHICH ALPHA GOVERNS
#     tensor 50, not through the group's meta-signal.  THE BRIEFING'S SHARPEST
#     PREDICTION -- "if the 24.8 pp cliff is a sign-domination artefact,
#     normalising should collapse it" -- IS REFUTED AT THE MECHANISM LEVEL, FOR
#     ZERO GPU-HOURS, BEFORE THE BATCH.
#
# (iv) PER-TENSOR 1/numel NORMALISATION DOES NOT NEUTRALISE SIZE.  IT INVERTS IT.
#     The 1-D BatchNorm scales and shifts are 65 % of the tensors and 0.087 % of
#     the parameters.  They hold 2.6 % to 4.6 % of a group's sum|t| under the
#     standard reduction and 84 % to 93 % of it under w = 1/numel.  The same
#     inversion is what raises tensor 50's steering power in the `[50,12]`
#     coarse group from 0.0003 to 0.1470 -- a 490-fold increase, in the OPPOSITE
#     direction to the briefing's prediction.
#
# -----------------------------------------------------------------------------
# 3. WHAT crn1 THEREFORE TESTS.  NOT "does normalising remove the gap" -- that
#    question is already answered -- BUT: DOES THE COMPOSITION OF THE GROUP'S
#    META-SIGNAL MATTER FOR OUTCOME AT ALL?
# -----------------------------------------------------------------------------
# PATCH_REDNORM adds, additively and opt-in, `--stepsize-groups tn:<spec>`:
#     z_G = sum_{i in G} <h_i, g_i> / numel_i
# The partition, the group count, the beta shape, the `_probe` branch and every
# other code path are IDENTICAL to the standard twin (tests/test_rednorm.py R3,
# and the launcher's guard 4d on the live model).  The ONLY difference is who
# holds the vote inside the group: the conv/linear weights (97 %) under the
# standard reduction, the BatchNorm parameters (84-93 %) under `tn:`.  The two
# arms of a pair are two OPPOSITE extreme weightings of the same partition, and
# the measured fraction of meta-steps on which they disagree in sign is 7.62 %
# to 19.10 %.
#
# So the batch asks a question no previous arm has asked:
#   if 8-19 % of ALL meta-update signs are flipped, by handing the group's vote
#   from the 99.9 % of parameters to the 0.087 % of them, does 100-epoch
#   accuracy move?
# A NULL closes the reduction channel with an INTERVENTION rather than an
# observation, and bounds how much of the granularity phenomenon can be a
# reduction effect.  A MOVE reopens it and says which tensors' inner products
# actually set a usable step size.
#
# `tn:` is INERT at every granularity whose groups hold ONE tensor (layerwise,
# nodewise, weightwise, nodewise1d, chunk<K>, permnode<S>) -- tests/test_rednorm
# R5 asserts that BITWISE -- so the briefing's {scalar, layerwise} x {standard,
# intervened} design has THREE distinct cells, not four, and its layerwise pair
# is identical BY CONSTRUCTION.  This batch spends its arms where the
# intervention can bite.
#
# -----------------------------------------------------------------------------
# 4. THE DESIGN.  6 arms x 3 seeds = 18 jobs, ONE submission, 100 epochs.
#   k01   scalar                  m = 1 FLOOR ANCHOR, in batch, MANDATORY
#   k01n  tn:scalar               the same m = 1, vote handed to the BN params
#   k49   sets:1-49/50-62         == [49,13] EXACTLY (tests/test_namesets.py N2)
#   k49n  tn:sets:1-49/50-62      the same partition, vote handed over
#   k50   sets:1-50/51-62         == [50,12] EXACTLY
#   k50n  tn:sets:1-50/51-62      the same partition, vote handed over
#   SEEDS {15,16,17}: ZERO rows ANYWHERE in the corpus carry them.
#
# PRIMARY -- COMPOSITION.  Three same-partition pairs:
#   P01 = k01n - k01,  P49 = k49n - k49,  P50 = k50n - k50
#   H-COMP-INERT  the composition of the group's meta-signal does not affect
#                 outcome.  POINT PREDICTIONS: P01 = P49 = P50 = 0.000000.
#   H-COMP-LIVE   it does.  The COMPLEMENT; it makes no point prediction, and
#                 that is exactly why the SATURATION branch below exists.
#
# SECONDARY -- THE CLIFF, with its direction fixed by (iii) and (iv) IN ADVANCE:
#   D_STD = k50 - k49,  D_TN = k50n - k49n,  I = D_TN - D_STD
#   H-CLIFF-GOV       the cliff is governance -> I = 0.000000.
#   H-CLIFF-RED       the cliff runs through the reduction -> |D_TN| > |D_STD|,
#                     AMPLIFIED, because `tn:` multiplies tensor 50's steering
#                     power by 490 (0.0003 -> 0.1470).
#   H-CLIFF-COLLAPSE  the briefing's account, D_TN = 0.000000.  REGISTERED AS
#                     ALREADY EXCLUDED by (iii); observing it would OVERTURN the
#                     pre-registration measurement, not confirm the briefing.
#
# -----------------------------------------------------------------------------
# 5. THE NOISE FLOOR.  RE-DERIVED FROM THE LIVE CORPUS AT REGISTRATION TIME.
# -----------------------------------------------------------------------------
# No sigma is copied forward (156, 159, 161 each record a registered sigma
# drifting after a later ingest).  EVERY corpus reader in this file excludes
# rows whose `run` begins `crn1-`, so EVERY frozen premise -- all four sigmas,
# the floor, the ceiling, both level anchors, the plateau neighbours, the
# duplicate-seed premise and the grammar-novelty premise -- is INVARIANT UNDER
# THIS BATCH'S OWN INGEST.  cW1 (165/166) is the one scorer that actually
# achieved this; the pattern is copied and then VERIFIED by --selftest rather
# than asserted (161.9 recorded cS2 claiming it with 1 of 6 checks having it).
# The estimator is cR1's -- the pooled within-(batch x granularity) SD of
# plateau5 over m = 2 arms in this exact cell -- computed BOTH narrow
# (`^\[\d+,\d+\]$`) and WIDE (adding the `sets:` m = 2 rows) at BOTH horizons,
# with SIGMA_W = max of the four.
#
# -----------------------------------------------------------------------------
# 6. WHAT crn1 CANNOT DO -- RECORDED IN ADVANCE
# -----------------------------------------------------------------------------
#  * NO claim at any horizon but 100 epochs, and none about attenuation.
#  * NO claim about any granularity but m = 1 and the two m = 2 cut positions 49
#    and 50; NOTHING about layerwise or any other single-tensor granularity,
#    where `tn:` is inert BY CONSTRUCTION; NOTHING about m in general.
#  * NO claim that either weighting is a BETTER optimizer.  The chain rule gives
#    the UNNORMALISED sum; `tn:` is deliberately the wrong gradient and is a
#    MECHANISM PROBE only.
#  * NO test of 152.12 rival (c): PROBE=0, and at m <= 2 block_product has
#    already reduced <h,g> to one scalar per group before _probe is reached.
#  * NO claim about CIFAR-10, Tiny-ImageNet, ImageNet-489, ResNet-50, any other
#    meta-stepsize, alpha0, optimiser pair, or any hierarchical operator.
#  * RULE 11 IS OPEN.  Every arm sits at the ONE shared ms = 1e-3, so every
#    contrast here is a SHARED-SETTING contrast; hz9 showed such a contrast can
#    REVERSE SIGN under per-arm tuning.  This batch inherits that limitation and
#    does not repair it.  In particular a COMPOSITION-INERT verdict is a
#    statement about this cell at this meta-stepsize, not about the operator.
# =============================================================================

from __future__ import print_function

import os
import re
import sys
import csv
import glob
import math
import shlex
import argparse
import statistics
import collections

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "all_runs.csv")

# ---- the cell.  Identical to cpk1's, cts1's, cpr1's and cpg1's, flag for flag.
PREFIX = "crn1-"
BATCH = "crn1"
NET, DSET = "ResNet18_c100", "CIFAR100"
BASE_ALG, META_ALG = "SGDm", "Lion"
MS, ALPHA0, GAMMA, AUG = "1e-3", "1e-6", "1", "1"
CLIP_C = "-15:-2.3026"
BATCH_SIZE = 100
EPOCHS = 100                     # THE ONLY HORIZON THIS BATCH SPEAKS AT
SEEDS = (15, 16, 17)
N_TENSORS = 62
TOTAL_PARAMS = 11220132

ANCHOR = "k01"
ARMS = ("k01", "k01n", "k49", "k49n", "k50", "k50n")
PAIRS = (("k01", "k01n"), ("k49", "k49n"), ("k50", "k50n"))
N_RUNS = len(ARMS) * len(SEEDS)  # 18
MANIFEST_NAME = "PARTITION-MANIFEST.txt"

# ---- the registered spec strings.  The launcher composes from THESE and its
# guard 4c proves the two tables byte-identical; G0 checks each run's own ARGS
# line against them.  Written once, here.
SPEC = {
    "k01": "scalar",
    "k01n": "tn:scalar",
    "k49": "sets:1-49/50-62",
    "k49n": "tn:sets:1-49/50-62",
    "k50": "sets:1-50/51-62",
    "k50n": "tn:sets:1-50/51-62",
}

TN_ARMS = ("k01n", "k49n", "k50n")     # the arms whose spec turns PATCH_REDNORM on

# ---- the two tensors the cut position moves across, 1-BASED. ---------------
T_CONV2 = "layer4.0.conv2.weight"     # tensor 49, CONVOLUTION, 2,359,296
T_BN2W = "layer4.0.bn2.weight"        # tensor 50, BN SCALE,          512
IDX = {T_CONV2: 49, T_BN2W: 50}

# ---- the FROZEN model, in named_parameters() order.  G1 checks the LIVE model
# against this name by name, through the manifest the launcher wrote from
# HF.polish_the_stepsize_groups itself; --selftest derives every index set and
# every mass from it without needing torch.
MODEL = (
    ("conv1.weight", 1728), ("bn1.weight", 64), ("bn1.bias", 64),
    ("layer1.0.conv1.weight", 36864), ("layer1.0.bn1.weight", 64),
    ("layer1.0.bn1.bias", 64), ("layer1.0.conv2.weight", 36864),
    ("layer1.0.bn2.weight", 64), ("layer1.0.bn2.bias", 64),
    ("layer1.1.conv1.weight", 36864), ("layer1.1.bn1.weight", 64),
    ("layer1.1.bn1.bias", 64), ("layer1.1.conv2.weight", 36864),
    ("layer1.1.bn2.weight", 64), ("layer1.1.bn2.bias", 64),
    ("layer2.0.conv1.weight", 73728), ("layer2.0.bn1.weight", 128),
    ("layer2.0.bn1.bias", 128), ("layer2.0.conv2.weight", 147456),
    ("layer2.0.bn2.weight", 128), ("layer2.0.bn2.bias", 128),
    ("layer2.0.shortcut.0.weight", 8192), ("layer2.0.shortcut.1.weight", 128),
    ("layer2.0.shortcut.1.bias", 128), ("layer2.1.conv1.weight", 147456),
    ("layer2.1.bn1.weight", 128), ("layer2.1.bn1.bias", 128),
    ("layer2.1.conv2.weight", 147456), ("layer2.1.bn2.weight", 128),
    ("layer2.1.bn2.bias", 128), ("layer3.0.conv1.weight", 294912),
    ("layer3.0.bn1.weight", 256), ("layer3.0.bn1.bias", 256),
    ("layer3.0.conv2.weight", 589824), ("layer3.0.bn2.weight", 256),
    ("layer3.0.bn2.bias", 256), ("layer3.0.shortcut.0.weight", 32768),
    ("layer3.0.shortcut.1.weight", 256), ("layer3.0.shortcut.1.bias", 256),
    ("layer3.1.conv1.weight", 589824), ("layer3.1.bn1.weight", 256),
    ("layer3.1.bn1.bias", 256), ("layer3.1.conv2.weight", 589824),
    ("layer3.1.bn2.weight", 256), ("layer3.1.bn2.bias", 256),
    ("layer4.0.conv1.weight", 1179648), ("layer4.0.bn1.weight", 512),
    ("layer4.0.bn1.bias", 512), ("layer4.0.conv2.weight", 2359296),
    ("layer4.0.bn2.weight", 512), ("layer4.0.bn2.bias", 512),
    ("layer4.0.shortcut.0.weight", 131072), ("layer4.0.shortcut.1.weight", 512),
    ("layer4.0.shortcut.1.bias", 512), ("layer4.1.conv1.weight", 2359296),
    ("layer4.1.bn1.weight", 512), ("layer4.1.bn1.bias", 512),
    ("layer4.1.conv2.weight", 2359296), ("layer4.1.bn2.weight", 512),
    ("layer4.1.bn2.bias", 512), ("linear.weight", 51200), ("linear.bias", 100),
)
MODEL_NAMES = [n for n, _ in MODEL]
MODEL_NUMEL = [k for _, k in MODEL]

# ---- the registered coarse sizes and masses of the two m = 2 partitions. ----
K49_SIZES, K49_MASS = (49, 13), (6315072, 4905060)
K50_SIZES, K50_MASS = (50, 12), (6315584, 4904548)

# ---- the noise floor, RE-DERIVED FROM THE LIVE CORPUS AT REGISTRATION TIME,
# with rows whose run begins `crn1-` EXCLUDED so it is invariant under this
# batch's own ingest.
SIGMA_100 = 0.925518
SIGMA_100_DF, SIGMA_100_CELLS, SIGMA_100_MEMBERS = 60, 30, 90
SIGMA_100_WIDE = 0.883894
SIGMA_100_WIDE_DF, SIGMA_100_WIDE_CELLS, SIGMA_100_WIDE_MEMBERS = 74, 37, 111
SIGMA_772 = 0.864841
SIGMA_772_DF, SIGMA_772_CELLS, SIGMA_772_MEMBERS = 26, 13, 39
SIGMA_772_WIDE = 0.864841
SIGMA_772_WIDE_DF, SIGMA_772_WIDE_CELLS, SIGMA_772_WIDE_MEMBERS = 26, 13, 39
SIGMA_W = 0.925518               # = max of the four, the frozen rule
OTHER_HORIZON = 772

# ---- the corpus LEVEL anchors, same cell, 100 epochs, crn1 EXCLUDED. -------
# `[49,13]` and `sets:1-49/50-62` are the SAME PARTITION (tests/test_namesets.py
# N2 asserts the equality against the UNPATCHED code path), so they are pooled.
L_FLOOR = 22.795652
L_FLOOR_N, L_FLOOR_BATCHES = 23, 8
L_CEIL = 69.532100
L_CEIL_N, L_CEIL_BATCHES = 20, 7
L_K49 = 55.405667
L_K49_N, L_K49_BATCHES = 18, 5
L_K50 = 30.337000
L_K50_N, L_K50_BATCHES = 6, 2
D_STD_CORPUS = -25.068667         # L_K50 - L_K49
# same-cell arms that already sit on the m = 1 PLATEAU -- the evidence for the
# one-sided registration of the k01 pair.
PLATEAU_NEIGHBOURS = {"[53,9]": 22.075333, "[54,8]": 22.213333,
                      "[2,60]": 22.708667}


def se_of_combo(coefs, sigma=SIGMA_W, n=len(SEEDS)):
    return sigma * math.sqrt(sum(c * c for c in coefs) / float(n))


SE_ARM = se_of_combo((1,))
SE_ARM_DIFF = se_of_combo((1, -1))
SE_INT = se_of_combo((1, -1, -1, 1))

# ---- BARS.  --selftest quotes every one in SE units. ------------------------
READ_BAR = 1.511365       # 2 * SE_ARM_DIFF
INT_BAR = 2.137392        # 2 * SE_INT
NOISY_BAR = 2.776554      # 3 * SIGMA_W
FLOOR_BAND = 2.776554     # 3 * SIGMA_W above the IN-BATCH k01 level
CTRL_BAR = 16.305008      # half of the corpus (k49 - k01) at 100 epochs
CLIFF_MIN = 10.000000     # the in-batch |D_STD| below which the cliff is void
DEAD_BAR = 5.00           # TRAIN accuracy below this = the arm did not train

# ---- THE ACCOUNTS. ---------------------------------------------------------
# PRIMARY, on the three same-partition pairs.  H-COMP-INERT is the only one with
# point predictions; H-COMP-LIVE is its complement and makes none, which is
# exactly why the SATURATION branch exists.
PRED_PAIR = {"H-COMP-INERT": 0.000000}
# SECONDARY, on the cliff.  All three are point predictions; H-CLIFF-RED's is a
# DIRECTION fixed in advance by the pre-registration measurement.
PRED_I = {"H-CLIFF-GOV": 0.000000}
PRED_CLIFF_COLLAPSE_D_TN = 0.000000

# ---- the registered LEVEL predictions, corpus-anchored, used ONLY for the
# floor census.  Only H-COMP-INERT predicts levels; the complement accounts do
# not, and the SATURATION branch is what keeps a dead arm from confirming them.
PRED_LEVEL = {
    "H-COMP-INERT": {"k01": L_FLOOR, "k01n": L_FLOOR, "k49": L_K49,
                     "k49n": L_K49, "k50": L_K50, "k50n": L_K50},
}
# k01 and k01n are the DESIGNATED m = 1 anchor and its twin; they are exempt
# from the floor census by construction and are named here so the exemption is
# explicit rather than silent.  The k01 pair is additionally registered
# ONE-SIDED: a DOWNWARD move is not interpretable, because the m = 1 level is a
# plateau that other degenerate same-cell arms already occupy.
FLOOR_EXEMPT = ("k01", "k01n")
ONE_SIDED_PAIR = ("k01", "k01n")

# ---- THE BRANCH MAPS.  Exhaustive and mutually exclusive; --selftest proves it.
PAIR_DOC = {
    "I": "INERT                  |P| <= READ_BAR",
    "U": "MOVED-UP               P >  READ_BAR",
    "D": "MOVED-DOWN             P < -READ_BAR, and the tn: arm is NOT on the "
         "m = 1 plateau",
    "T": "SATURATED              P < -READ_BAR and the tn: arm IS within "
         "FLOOR_BAND of the in-batch k01 while its twin is not -- a "
         "content-free 'the arm died' model fits, so NO account is supported",
    "O": "DOWN-NOT-INTERPRETABLE the ONE-SIDED k01 pair moved down",
}
COMP_DOC = {
    "INERT": "COMPOSITION-INERT       every pair INERT",
    "LIVE": "COMPOSITION-OPERATIVE   at least one pair MOVED-UP or MOVED-DOWN",
    "SAT": "UNRESOLVED-COMPOSITION-SATURATED  the only non-INERT pairs are "
           "SATURATED or DOWN-NOT-INTERPRETABLE",
    "DEAD": "UNRESOLVED-INTERVENTION-FAILED    a tn: arm's TRAIN is below "
            "DEAD_BAR (overrides everything)",
}
CLIFF_DOC = {
    "G": "CLIFF-GOVERNANCE       |I| <= INT_BAR",
    "R": "CLIFF-AMPLIFIED        |D_TN| > |D_STD| + INT_BAR",
    "K": "CLIFF-COLLAPSED        |D_TN| <= READ_BAR -- OVERTURNS the "
         "pre-registration measurement (iii)",
    "P": "CLIFF-MOVED-UNREGISTERED  moved, but neither amplified nor collapsed",
    "X": "UNRESOLVED-ANCHOR-FAILED  in-batch |D_STD| < CLIFF_MIN (overrides)",
    "S": "UNRESOLVED-CLIFF-SATURATED  k49n AND k50n both within FLOOR_BAND of "
         "the in-batch k01 (overrides G/R/K/P)",
}

# ---- THE PRE-REGISTRATION MEASUREMENT.  Frozen literals produced by
# analysis/cX1_reduction_probe.py + cX1_reduction_report.py at this exact cell,
# seed 0, 12 epochs = 6,000 meta-steps, BEFORE this file was committed; the full
# output is committed at results/crn1_reduction_probe/REPORT.txt.  They enter NO
# verdict.  They are the evidence that the intervention is LIVE, that the batch
# is not a null by construction, and that the briefing's predicted DIRECTION is
# wrong.
PROBE = {
    "scalar": {0: {"tensors": 62, "params": 11220132, "bn_tensors": 40,
                   "dominance": 0.9968, "argmax_share_mean": 0.3842,
                   "unanimous": 0.0002, "flip_1_over_n": 0.0762,
                   "flip_signvote": 0.0008, "float32_sign_error": 0.0000,
                   "bn_share_std": 0.0259, "bn_share_tn": 0.8404,
                   "argmax": "linear.weight"}},
    "[49,13]": {0: {"tensors": 49, "params": 6315072, "bn_tensors": 32,
                    "dominance": 0.9920, "argmax_share_mean": 0.2341,
                    "unanimous": 0.0002, "flip_1_over_n": 0.1910,
                    "flip_signvote": 0.0022, "float32_sign_error": 0.0000,
                    "bn_share_std": 0.0367, "bn_share_tn": 0.9282,
                    "argmax": "layer4.0.conv2.weight"},
                1: {"tensors": 13, "params": 4905060, "bn_tensors": 8,
                    "dominance": 0.9982, "argmax_share_mean": 0.6201,
                    "unanimous": 0.7390, "flip_1_over_n": 0.1113,
                    "flip_signvote": 0.0073, "float32_sign_error": 0.0000,
                    "bn_share_std": 0.0198, "bn_share_tn": 0.5371,
                    "argmax": "linear.weight"}},
    "[50,12]": {0: {"tensors": 50, "params": 6315584, "bn_tensors": 33,
                    "dominance": 0.9927, "argmax_share_mean": 0.2317,
                    "unanimous": 0.0002, "flip_1_over_n": 0.0963,
                    "flip_signvote": 0.0028, "float32_sign_error": 0.0000,
                    "bn_share_std": 0.0463, "bn_share_tn": 0.9324,
                    "argmax": "layer4.0.conv2.weight"},
                1: {"tensors": 12, "params": 4904548, "bn_tensors": 7,
                    "dominance": 0.9980, "argmax_share_mean": 0.6239,
                    "unanimous": 0.7555, "flip_1_over_n": 0.0950,
                    "flip_signvote": 0.0078, "float32_sign_error": 0.0000,
                    "bn_share_std": 0.0138, "bn_share_tn": 0.4480,
                    "argmax": "linear.weight"}},
}
# tensor 50's STEERING POWER in the `[50,12]` COARSE group: the fraction of
# meta-steps on which REMOVING it flips that group's sign.
T50_STEER_STD = 0.0003
T50_STEER_TN = 0.1470
# and tensor 49's, in the same group, for contrast.
T49_STEER_STD = 0.0030
T49_STEER_TN = 0.0003
# the smallest 1/n sign-flip rate anywhere in PROBE.
PROBE_MIN_FLIP = 0.0762
# the largest sign-vote disagreement anywhere in PROBE.
PROBE_MAX_SIGNVOTE = 0.0078

EP_RE = re.compile(r"Epoch\s+(\d+).*?Test Accuracy:\s*([0-9.]+)", re.I)
EPTR_RE = re.compile(r"Epoch\s+(\d+).*?Train Accuracy:\s*([0-9.]+)", re.I)
ARGS_RE = re.compile(r"^\s*ARGS:\s*(.*)$")
ENV_RE = re.compile(r"^\s*ENV:\s*(.*)$")
NAME_RE = re.compile(r"^crn1-(k01n|k01|k49n|k49|k50n|k50)-s(15|16|17)-(\d+)\.out$")


# =============================================================================
# the `sets:` grammar, re-implemented here for --selftest ONLY.  It never runs
# at score time: G1 reads what the LIVE model actually composed, out of the
# manifest the launcher wrote from `HF.polish_the_stepsize_groups` itself.
# =============================================================================
_NS_INT = re.compile(r"^\d+$")
_NS_RANGE = re.compile(r"^(\d+)-(\d+)$")


def spec_to_groups(spec, names=MODEL_NAMES):
    """`[tn:]sets:...` or `[tn:]scalar` -> a list of lists of 1-BASED indices."""
    if spec.startswith("tn:"):
        spec = spec[3:]
    if spec == "scalar":
        return [list(range(1, len(names) + 1))]
    if not spec.startswith("sets:"):
        raise ValueError("unregistered spec form %r" % spec)
    out, owner = [], {}
    for gi, g in enumerate(spec[len("sets:"):].split("/")):
        idx = []
        for item in g.split(","):
            m = _NS_RANGE.match(item)
            if m:
                got = list(range(int(m.group(1)), int(m.group(2)) + 1))
            elif _NS_INT.match(item):
                got = [int(item)]
            elif item in names:
                got = [names.index(item) + 1]
            else:
                raise ValueError("bad item %r" % item)
            for i in got:
                if i in owner:
                    raise ValueError("tensor %d claimed twice" % i)
                owner[i] = gi
            idx.extend(got)
        out.append(sorted(idx))
    if len(owner) != len(names):
        raise ValueError("spec %r does not cover the model" % spec)
    return out


# =============================================================================
# .out parsing
# =============================================================================
def parse_argsline(line):
    try:
        toks = shlex.split(line)
    except ValueError:
        toks = line.split()
    occ, i, n = [], 0, len(toks)
    while i < n:
        t = toks[i]
        if t.startswith("--") and len(t) > 2:
            if "=" in t:
                f, v = t.split("=", 1)
                occ.append((f.lstrip("-"), v))
                i += 1
                continue
            f, vals, j = t.lstrip("-"), [], i + 1
            while j < n and not (toks[j].startswith("--") and len(toks[j]) > 2):
                vals.append(toks[j])
                j += 1
            occ.append((f, " ".join(vals)))
            i = j
            continue
        i += 1
    return occ


def tail5(d, budget):
    v = [d[e] for e in range(budget - 5, budget) if e in d]
    return (sum(v) / 5.0) if len(v) == 5 else None


def load_runs(runsdir):
    recs = []
    pats = [os.path.join(runsdir, PREFIX + "*.out"),
            os.path.join(runsdir, BATCH, PREFIX + "*.out")]
    files = sorted(set(sum([glob.glob(p) for p in pats], [])))
    for f in files:
        m = NAME_RE.match(os.path.basename(f))
        if not m:
            recs.append({"file": f, "bad_name": True})
            continue
        arm, seed, job = m.group(1), int(m.group(2)), m.group(3)
        te, tr, args, env, done = {}, {}, None, None, False
        with open(f, errors="ignore") as fh:
            for line in fh:
                a = ARGS_RE.match(line)
                if a and args is None:
                    args = a.group(1)
                e = ENV_RE.match(line)
                if e and env is None:
                    env = e.group(1).strip()
                if line.strip() == "RUN_DONE":
                    done = True
                x = EP_RE.search(line)
                if x:
                    te[int(x.group(1))] = float(x.group(2))
                y = EPTR_RE.search(line)
                if y:
                    tr[int(y.group(1))] = float(y.group(2))
        occ = parse_argsline(args) if args else []
        cnt = collections.Counter(f2 for f2, _ in occ)
        recs.append({
            "file": f, "bad_name": False, "arm": arm, "seed": seed, "job": job,
            "args": args, "env": env, "occ": occ, "eff": dict(occ),
            "repeated": sorted([k for k, v in cnt.items() if v > 1]),
            "run_done": done, "test": te, "train": tr, "n_ep": len(te),
            "p5": tail5(te, EPOCHS), "t5": tail5(tr, EPOCHS),
        })
    return recs


# =============================================================================
# the corpus.  Used ONLY by --selftest.  EVERY reader excludes crn1's own rows.
# =============================================================================
def _csv_rows(path=CSV, exclude_own=True):
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if (r.get("superseded") or "0") == "1":
                continue
            if exclude_own and str(r.get("run") or "").startswith(PREFIX):
                continue
            out.append(r)
    return out


def _batch_of(run):
    return re.split(r"[-_]", str(run))[0]


def _in_cell(r, epochs):
    return (r["network"] == NET and r["dataset"] == DSET
            and r["base"] == BASE_ALG and r["meta"] == META_ALG
            and r["meta_stepsize"] == MS and r["alpha0"] == ALPHA0
            and r["gamma"] == GAMMA and r["augment"] == AUG
            and r["beta_clip"] == CLIP_C and not (r.get("hier") or "").strip()
            and r["batch_size"] == str(BATCH_SIZE)
            and r["epochs_done"] == str(epochs)
            and r["collapsed"] == "0")


M2_NARROW = re.compile(r"^\[\d+,\d+\]$")


def _is_m2(g, wide):
    g = str(g or "")
    if M2_NARROW.match(g):
        return True
    if wide and g.startswith("sets:"):
        return g.count("/") == 1
    return False


def noise_floor_from_csv(path=CSV, epochs=EPOCHS, wide=False):
    cells = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not _in_cell(r, epochs) or not _is_m2(r["granularity"], wide):
            continue
        try:
            cells[(_batch_of(r["run"]), r["granularity"])].append(
                float(r["plateau5"]))
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


def cell_levels(grans, path=CSV, epochs=EPOCHS):
    """Pooled same-cell plateau5 over one or more granularity strings."""
    if isinstance(grans, str):
        grans = (grans,)
    v, b = [], set()
    for r in _csv_rows(path):
        if _in_cell(r, epochs) and str(r["granularity"]) in grans:
            try:
                v.append(float(r["plateau5"]))
                b.add(_batch_of(r["run"]))
            except (TypeError, ValueError):
                pass
    return (statistics.mean(v) if v else None), len(v), len(b)


def count_own_rows(path=CSV):
    n = 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if str(r.get("run") or "").startswith(PREFIX):
                n += 1
    return n


def count_seed_rows(path=CSV):
    return sum(1 for r in _csv_rows(path)
               if str(r.get("seed")) in tuple(str(s) for s in SEEDS))


# =============================================================================
# manifest -- resolution and reading
# =============================================================================
def resolve_manifest(runsdir, explicit=None):
    if explicit:
        return (explicit, "given explicitly with --manifest")
    if not runsdir:
        return (None, "no runsdir")
    up = os.path.dirname(os.path.abspath(runsdir.rstrip("/")))
    cands = [(os.path.join(runsdir, BATCH, MANIFEST_NAME),
              "DEFAULT <runsdir>/%s/%s" % (BATCH, MANIFEST_NAME)),
             (os.path.join(runsdir, BATCH + "-" + MANIFEST_NAME),
              "DEFAULT <runsdir>/%s-%s" % (BATCH, MANIFEST_NAME)),
             (os.path.join(runsdir, MANIFEST_NAME),
              "DEFAULT <runsdir>/%s" % MANIFEST_NAME),
             (os.path.join(up, BATCH, MANIFEST_NAME),
              "DEFAULT <runsdir>/../%s/%s" % (BATCH, MANIFEST_NAME))]
    for p, how in cands:
        if os.path.exists(p):
            return (p, how)
    for pat in (os.path.join(runsdir, "*", MANIFEST_NAME),
                os.path.join(up, "*", BATCH, MANIFEST_NAME)):
        hits = sorted(g for g in glob.glob(pat) if BATCH in g)
        if len(hits) == 1:
            return (hits[0], "DEFAULT, found by glob %s" % pat)
    return (None, "NOT FOUND -- searched %s" % "; ".join(c[0] for c in cands))


def read_manifest(path):
    man = {"tensors": {}, "specs": {}, "pairs": {}, "reduction": {},
           "inert": {}, "raw": []}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            man["raw"].append(line)
            p = line.split()
            if not p:
                continue
            if p[0] == "NUM_PARAM_TENSORS":
                man["T"] = int(p[1])
            elif p[0] == "TOTAL_PARAMS":
                man["total"] = int(p[1])
            elif p[0] == "TENSOR" and len(p) >= 4:
                man["tensors"][int(p[1])] = (p[2], int(p[3]))
            elif p[0] == "ARMSPEC":
                d, arm, key, i = {}, p[1], None, 2
                while i < len(p):
                    if p[i] in ("SPEC", "M", "SIZES", "PARAMS", "REDNORM"):
                        key = p[i]
                        d[key] = []
                    elif key:
                        d[key].append(p[i])
                    i += 1
                man["specs"][arm] = {
                    "spec": d.get("SPEC", [""])[0],
                    "m": int(d.get("M", ["0"])[0]),
                    "sizes": [int(x) for x in d.get("SIZES", ["0"])[0].split(",")],
                    "params": [int(x) for x in d.get("PARAMS", ["0"])[0].split(",")],
                    "rednorm": int(d.get("REDNORM", ["0"])[0]),
                }
            elif p[0] == "PAIR" and len(p) >= 5:
                man["pairs"][(p[1], p[2])] = int(p[4])
            elif p[0] == "REDUCTION_DIFFERS" and len(p) >= 4:
                man["reduction"][(p[1], p[2])] = int(p[3])
            elif p[0] == "SINGLE_TENSOR_INERT" and len(p) >= 3:
                man["inert"][p[1]] = int(p[2])
    return man


# =============================================================================
def fmt(x, w=9, p=4):
    return (" " * w) if x is None else ("%*.*f" % (w, p, x))


def se_str(x, se=SE_ARM_DIFF, tag="SE"):
    return "%+6.2f %s" % (x / se, tag)


# =============================================================================
# THE BRANCH FUNCTIONS.  Pure, deterministic, and proved exhaustive by
# --selftest section H.
# =============================================================================
def pair_token(p, tn_on_plateau, std_on_plateau, one_sided):
    if abs(p) <= READ_BAR:
        return "I"
    if p > READ_BAR:
        return "U"
    if one_sided:
        return "O"
    if tn_on_plateau and not std_on_plateau:
        return "T"
    return "D"


def composition_verdict(tokens, dead):
    if dead:
        return "DEAD"
    if all(t == "I" for t in tokens):
        return "INERT"
    if any(t in ("U", "D") for t in tokens):
        return "LIVE"
    return "SAT"


def cliff_branch(d_tn, d_std, saturated):
    if d_std is None or d_tn is None:
        return "X"
    if abs(d_std) < CLIFF_MIN:
        return "X"
    if saturated:
        return "S"
    if abs(d_tn - d_std) <= INT_BAR:
        return "G"
    if abs(d_tn) > abs(d_std) + INT_BAR:
        return "R"
    if abs(d_tn) <= READ_BAR:
        return "K"
    return "P"


# =============================================================================
# --selftest
# =============================================================================
NPASS = [0]
NFAIL = [0]


def chk(cond, label, got="", want=""):
    if cond:
        NPASS[0] += 1
        print("  PASS %-66s %s" % (label, got))
    else:
        NFAIL[0] += 1
        print("  FAIL %-66s %s   want %s" % (label, got, want))


def selftest():
    print("=" * 78)
    print("cX1_crn1_score.py --selftest")
    print("=" * 78)

    print("\nA. the design")
    chk(len(ARMS) == 6 and len(set(ARMS)) == 6, "six distinct arms", str(ARMS))
    chk(len(SEEDS) == 3 and len(set(SEEDS)) == 3, "three distinct seeds", str(SEEDS))
    chk(N_RUNS == 18, "18 runs", str(N_RUNS))
    chk(set(SPEC) == set(ARMS), "every arm has exactly one registered spec")
    chk(tuple(sorted(a for a in ARMS if SPEC[a].startswith("tn:")))
        == tuple(sorted(TN_ARMS)), "the tn: arms are exactly %s" % (TN_ARMS,))
    for a, b in PAIRS:
        chk(SPEC[b] == "tn:" + SPEC[a], "%s is `tn:` + %s's spec" % (b, a), SPEC[b])
    chk(len(set(SPEC.values())) == 6, "all six spec strings are distinct")
    chk(ONE_SIDED_PAIR in PAIRS, "the ONE-SIDED pair is a registered pair",
        str(ONE_SIDED_PAIR))

    print("\nB. RULE 20 -- every spec is ONE safe shell token")
    for a in ARMS:
        s = SPEC[a]
        ok = (re.match(r"^[A-Za-z0-9_.,:/-]+$", s) is not None
              and " " not in s and "'" not in s and '"' not in s
              and "=" not in s and not s.startswith("-"))
        chk(ok and shlex.split(s) == [s],
            "%-5s %-24s single safe shell token, shlex -> ONE" % (a, s))

    print("\nC. the frozen model, and the two partitions it composes")
    chk(len(MODEL) == N_TENSORS, "the model has %d tensors" % N_TENSORS,
        str(len(MODEL)))
    chk(sum(MODEL_NUMEL) == TOTAL_PARAMS, "total parameters",
        str(sum(MODEL_NUMEL)), str(TOTAL_PARAMS))
    chk(MODEL_NAMES[48] == T_CONV2, "tensor 49 is %s" % T_CONV2, MODEL_NAMES[48])
    chk(MODEL_NAMES[49] == T_BN2W, "tensor 50 is %s" % T_BN2W, MODEL_NAMES[49])
    chk(MODEL_NUMEL[48] == 2359296 and MODEL_NUMEL[49] == 512,
        "49 is a 2,359,296-param CONV and 50 a 512-param BN SCALE",
        "%d / %d" % (MODEL_NUMEL[48], MODEL_NUMEL[49]))
    for arm, sizes, mass in (("k49", K49_SIZES, K49_MASS),
                             ("k50", K50_SIZES, K50_MASS)):
        g = spec_to_groups(SPEC[arm])
        sz = tuple(len(x) for x in g)
        ms = tuple(sum(MODEL_NUMEL[i - 1] for i in x) for x in g)
        chk(sz == sizes, "%s composes to sizes %s" % (arm, sizes), str(sz))
        chk(ms == mass, "%s composes to masses %s" % (arm, mass), str(ms))
        chk(sorted(i for x in g for i in x) == list(range(1, N_TENSORS + 1)),
            "%s partitions the 62 tensors exactly once" % arm)
        chk(spec_to_groups(SPEC[arm + "n"]) == g,
            "%sn composes to the IDENTICAL partition" % arm)
    chk(spec_to_groups(SPEC["k01"]) == [list(range(1, 63))]
        == spec_to_groups(SPEC["k01n"]),
        "k01 and k01n are both the single 62-tensor group")
    g49, g50 = spec_to_groups(SPEC["k49"]), spec_to_groups(SPEC["k50"])
    chk(set(g49[0]) ^ set(g50[0]) == {50},
        "k49 and k50 differ by EXACTLY tensor 50",
        str(sorted(set(g49[0]) ^ set(g50[0]))))
    chk(K50_MASS[0] - K49_MASS[0] == 512,
        "the cut moves 512 of 11,220,132 parameters = 0.00456%",
        str(K50_MASS[0] - K49_MASS[0]))

    print("\nD. the noise floor, RE-DERIVED from the corpus with crn1 EXCLUDED")
    allsig = []
    for nm, ep, wide, want, wdf, wc, wm in (
            ("SIGMA_100", EPOCHS, False, SIGMA_100, SIGMA_100_DF,
             SIGMA_100_CELLS, SIGMA_100_MEMBERS),
            ("SIGMA_100_WIDE", EPOCHS, True, SIGMA_100_WIDE, SIGMA_100_WIDE_DF,
             SIGMA_100_WIDE_CELLS, SIGMA_100_WIDE_MEMBERS),
            ("SIGMA_772", OTHER_HORIZON, False, SIGMA_772, SIGMA_772_DF,
             SIGMA_772_CELLS, SIGMA_772_MEMBERS),
            ("SIGMA_772_WIDE", OTHER_HORIZON, True, SIGMA_772_WIDE,
             SIGMA_772_WIDE_DF, SIGMA_772_WIDE_CELLS, SIGMA_772_WIDE_MEMBERS)):
        s, df, nc, nmem = noise_floor_from_csv(epochs=ep, wide=wide)
        allsig.append(s)
        chk(abs(s - want) < 5e-6, "%s re-derives" % nm, "%.6f" % s, "%.6f" % want)
        chk((df, nc, nmem) == (wdf, wc, wm), "%s df/cells/members" % nm,
            str((df, nc, nmem)), str((wdf, wc, wm)))
    chk(abs(SIGMA_W - max(allsig)) < 5e-6,
        "SIGMA_W IS the max of all four, by the frozen rule",
        "%.6f" % max(allsig), "%.6f" % SIGMA_W)

    print("\nE. the corpus LEVEL anchors, RE-DERIVED with crn1 EXCLUDED")
    for nm, grans, want, wn, wb in (
            ("L_FLOOR", ("scalar",), L_FLOOR, L_FLOOR_N, L_FLOOR_BATCHES),
            ("L_CEIL", ("layerwise",), L_CEIL, L_CEIL_N, L_CEIL_BATCHES),
            ("L_K49", ("[49,13]", "sets:1-49/50-62"), L_K49, L_K49_N,
             L_K49_BATCHES),
            ("L_K50", ("[50,12]",), L_K50, L_K50_N, L_K50_BATCHES)):
        m, n, nb = cell_levels(grans)
        chk(m is not None and abs(m - want) < 5e-6, "%s re-derives" % nm,
            "%.6f" % (m if m is not None else float("nan")), "%.6f" % want)
        chk((n, nb) == (wn, wb), "%s n / batches" % nm, str((n, nb)),
            str((wn, wb)))
    chk(abs(D_STD_CORPUS - (L_K50 - L_K49)) < 5e-6,
        "D_STD_CORPUS = L_K50 - L_K49", "%.6f" % (L_K50 - L_K49),
        "%.6f" % D_STD_CORPUS)
    for g, want in sorted(PLATEAU_NEIGHBOURS.items()):
        m, n, _ = cell_levels((g,))
        chk(m is not None and abs(m - want) < 5e-6,
            "plateau neighbour %-8s re-derives" % g,
            "%.6f (n=%d)" % (m if m is not None else float("nan"), n),
            "%.6f" % want)
        chk(m is not None and m < L_FLOOR + FLOOR_BAND,
            "plateau neighbour %-8s really sits ON the m=1 plateau" % g,
            "%.6f < %.6f" % (m, L_FLOOR + FLOOR_BAND))

    print("\nF. the bars, in SE units and by arithmetic")
    chk(abs(SE_ARM - SIGMA_W / math.sqrt(3.0)) < 1e-9,
        "SE_ARM = SIGMA_W / sqrt(3)", "%.6f" % SE_ARM)
    chk(abs(SE_ARM_DIFF - SIGMA_W * math.sqrt(2.0 / 3.0)) < 1e-9,
        "SE_ARM_DIFF = SIGMA_W * sqrt(2/3)", "%.6f" % SE_ARM_DIFF)
    chk(abs(SE_INT - SIGMA_W * math.sqrt(4.0 / 3.0)) < 1e-9,
        "SE_INT = SIGMA_W * sqrt(4/3)", "%.6f" % SE_INT)
    chk(abs(READ_BAR - 2 * SE_ARM_DIFF) < 5e-6, "READ_BAR = 2 * SE_ARM_DIFF",
        "%.6f" % READ_BAR, "%.6f" % (2 * SE_ARM_DIFF))
    chk(abs(INT_BAR - 2 * SE_INT) < 5e-6, "INT_BAR = 2 * SE_INT",
        "%.6f" % INT_BAR, "%.6f" % (2 * SE_INT))
    chk(abs(NOISY_BAR - 3 * SIGMA_W) < 5e-6, "NOISY_BAR = 3 * SIGMA_W",
        "%.6f" % NOISY_BAR)
    chk(abs(FLOOR_BAND - 3 * SIGMA_W) < 5e-6, "FLOOR_BAND = 3 * SIGMA_W",
        "%.6f" % FLOOR_BAND)
    chk(abs(CTRL_BAR - (L_K49 - L_FLOOR) / 2.0) < 5e-6,
        "CTRL_BAR = half of the corpus (k49 - k01)", "%.6f" % CTRL_BAR,
        "%.6f" % ((L_K49 - L_FLOOR) / 2.0))
    chk(CLIFF_MIN > 4 * SE_INT,
        "CLIFF_MIN is above 4 * SE_INT, so a passing cliff anchor really "
        "separates the cliff accounts", "%.6f > %.6f" % (CLIFF_MIN, 4 * SE_INT))

    print("\nG. the accounts, and the separation each is read at")
    chk(abs(PRED_PAIR["H-COMP-INERT"]) < 1e-12,
        "H-COMP-INERT predicts every pair difference = 0 exactly")
    chk(abs(PRED_I["H-CLIFF-GOV"]) < 1e-12,
        "H-CLIFF-GOV predicts I = 0 exactly")
    chk(abs(PRED_CLIFF_COLLAPSE_D_TN) < 1e-12,
        "H-CLIFF-COLLAPSE (the briefing's account) predicts D_TN = 0 exactly")
    chk(abs(D_STD_CORPUS) > INT_BAR,
        "GOV and COLLAPSE are separated by |D_STD| = %.6f pp" % abs(D_STD_CORPUS),
        "%.2f * SE_INT" % (abs(D_STD_CORPUS) / SE_INT))
    chk(abs(D_STD_CORPUS) / SE_INT > 10.0,
        "that separation exceeds 10 SE_INT, so the cliff read is not marginal",
        "%.2f SE_INT" % (abs(D_STD_CORPUS) / SE_INT))
    chk(T50_STEER_TN / max(T50_STEER_STD, 1e-9) > 100.0,
        "H-CLIFF-RED's DIRECTION is fixed by measurement: `tn:` multiplies "
        "tensor 50's steering power by %.0fx"
        % (T50_STEER_TN / T50_STEER_STD),
        "%.4f -> %.4f" % (T50_STEER_STD, T50_STEER_TN))
    chk(T49_STEER_TN < T49_STEER_STD,
        "and it moves tensor 49's the OTHER way, so `tn:` is a REWEIGHTING, "
        "not a uniform amplification",
        "%.4f -> %.4f" % (T49_STEER_STD, T49_STEER_TN))

    print("\nH. the branch maps are exhaustive and mutually exclusive")
    seenp = set()
    for p in (0.0, 0.5 * READ_BAR, READ_BAR, 2 * READ_BAR, -2 * READ_BAR,
              -30.0):
        for tn_pl in (False, True):
            for one in (False, True):
                t = pair_token(p, tn_pl, False, one)
                seenp.add(t)
                chk(t == pair_token(p, tn_pl, False, one),
                    "pair_token(%.4f, tn_plateau=%s, one_sided=%s) is "
                    "deterministic" % (p, tn_pl, one), t)
    chk(set(PAIR_DOC) == {"I", "U", "D", "T", "O"},
        "five documented pair tokens", str(sorted(PAIR_DOC)))
    chk(seenp == {"I", "U", "D", "T", "O"}, "all five are reachable",
        str(sorted(seenp)))
    chk(pair_token(-30.0, True, False, False) == "T",
        "a tn: arm alone on the plateau is SATURATED, not MOVED-DOWN")
    chk(pair_token(-30.0, True, True, False) == "D",
        "if BOTH arms are on the plateau the pair is not SATURATED "
        "(the standard twin is the reference, not the intervention)")
    chk(composition_verdict(("I", "I", "I"), False) == "INERT",
        "all-INERT -> COMPOSITION-INERT")
    chk(composition_verdict(("I", "U", "I"), False) == "LIVE",
        "any MOVED -> COMPOSITION-OPERATIVE")
    chk(composition_verdict(("O", "T", "I"), False) == "SAT",
        "only SATURATED / ONE-SIDED-DOWN -> UNRESOLVED")
    chk(composition_verdict(("I", "U", "I"), True) == "DEAD",
        "a dead tn: arm overrides everything")
    seenc = set()
    for dtn in (0.0, -25.0, -40.0, -12.0):
        for dstd in (-25.0, -5.0):
            b = cliff_branch(dtn, dstd, False)
            seenc.add(b)
    chk(set(CLIFF_DOC) == {"G", "R", "K", "P", "X", "S"},
        "six documented cliff branches", str(sorted(CLIFF_DOC)))
    chk(cliff_branch(0.0, -25.0, True) == "S", "cliff saturation overrides")
    chk(cliff_branch(0.0, -5.0, False) == "X", "a failed cliff anchor overrides")
    chk(cliff_branch(-25.0, -25.0, False) == "G", "I = 0 -> CLIFF-GOVERNANCE")
    chk(cliff_branch(-40.0, -25.0, False) == "R",
        "|D_TN| well above |D_STD| -> CLIFF-AMPLIFIED")
    chk(cliff_branch(0.0, -25.0, False) == "K",
        "D_TN = 0 -> CLIFF-COLLAPSED (would overturn the probe)")
    chk(cliff_branch(-12.0, -25.0, False) == "P",
        "moved but neither amplified nor collapsed -> UNREGISTERED")
    chk({"G", "R", "K", "P"} <= seenc, "G, R, K and P are all reachable",
        str(sorted(seenc)))

    print("\nI. THE FLOOR CENSUS -- every arm's predicted level under EVERY")
    print("   account that makes one, as a margin over the m = 1 anchor, in")
    print("   SE_ARM = %.6f pp." % SE_ARM)
    worst = None
    for acct in sorted(PRED_LEVEL):
        for arm in ARMS:
            if arm in FLOOR_EXEMPT:
                continue
            lvl = PRED_LEVEL[acct][arm]
            marg = lvl - L_FLOOR
            print("     %-13s %-5s predicted %8.4f   margin %+8.4f = %+7.2f SE"
                  % (acct, arm, lvl, marg, marg / SE_ARM))
            if worst is None or marg < worst[0]:
                worst = (marg, acct, arm)
    chk(worst is not None and worst[0] > FLOOR_BAND,
        "the WORST predicted margin clears FLOOR_BAND",
        "%+.4f pp (%s %s) = %+.2f SE" % (worst[0], worst[1], worst[2],
                                         worst[0] / SE_ARM),
        "> %.4f" % FLOOR_BAND)
    chk(worst is not None and worst[0] / SE_ARM > 10.0,
        "the WORST predicted margin exceeds 10 SE_ARM",
        "%+.2f SE" % (worst[0] / SE_ARM))
    print("     k01 and k01n are the DESIGNATED m = 1 anchor and its twin and")
    print("     are exempt BY CONSTRUCTION; their pair is registered ONE-SIDED")
    print("     for exactly that reason and carries no verdict downward.")
    print("     The COMPLEMENT accounts (H-COMP-LIVE, H-CLIFF-RED) make NO")
    print("     level prediction; branches T and S are what stop a dead arm")
    print("     from confirming them -- the cpr1 defect (CORRECTIONS 164),")
    print("     closed by construction.")
    chk(FLOOR_EXEMPT == ("k01", "k01n"),
        "the floor exemption is exactly the two m = 1 arms, named in advance")
    chk("T" in PAIR_DOC and "S" in CLIFF_DOC,
        "both saturation branches are registered in advance")

    print("\nJ. the invariance sentinel and the pre-registration premises")
    n_own = count_own_rows()
    chk(n_own in (0, N_RUNS), "crn1 rows in the corpus are 0 or exactly %d"
        % N_RUNS, "%d found -> %s" % (n_own, "PRE-REGISTRATION" if n_own == 0
                                      else "POST-INGEST"))
    chk(count_seed_rows() == 0,
        "ZERO foreign rows anywhere in the corpus carry seed 15, 16 or 17",
        str(count_seed_rows()))
    ntn = sum(1 for r in _csv_rows()
              if str(r.get("granularity") or "").startswith("tn:"))
    chk(ntn == 0, "ZERO foreign rows use the `tn:` grammar (it is new here)",
        str(ntn))

    print("\nK. THE NO-OP THEOREM as arithmetic -- why the briefing's option (a)")
    print("   is not run.  sign() of a positively rescaled zero-initialised")
    print("   linear recursion is the original sign.")
    mom, momc, b2, mm, c = 0.0, 0.0, 0.9, 0.99, 6315072.0
    bad = 0
    for t in range(400):
        z = math.sin(0.7 * t) * (1e-8 if t % 3 else 1e-12) - 3e-10
        s1 = b2 * mom + (1 - b2) * z
        s2 = b2 * momc + (1 - b2) * (z / c)
        if ((s1 > 0) - (s1 < 0)) != ((s2 > 0) - (s2 < 0)):
            bad += 1
        mom = mm * mom + (1 - mm) * z
        momc = mm * momc + (1 - mm) * (z / c)
    chk(bad == 0, "400 Lion meta-steps, group reduction rescaled by c = %g: "
                  "the sign never differs" % c, "%d differences" % bad)
    chk(True, "the LIVE-SOURCE form of this proof is "
              "analysis/cX1_reduction_noop_proof.py")

    print("\nL. THE PRE-REGISTRATION MEASUREMENT (frozen; enters NO verdict).")
    print("   Full output: results/crn1_reduction_probe/REPORT.txt")
    flips = [g["flip_1_over_n"] for s in PROBE.values() for g in s.values()]
    votes = [g["flip_signvote"] for s in PROBE.values() for g in s.values()]
    chk(abs(min(flips) - PROBE_MIN_FLIP) < 1e-9,
        "PROBE_MIN_FLIP is the minimum 1/n flip rate in PROBE",
        "%.4f" % min(flips), "%.4f" % PROBE_MIN_FLIP)
    chk(abs(max(votes) - PROBE_MAX_SIGNVOTE) < 1e-9,
        "PROBE_MAX_SIGNVOTE is the maximum sign-vote disagreement in PROBE",
        "%.4f" % max(votes), "%.4f" % PROBE_MAX_SIGNVOTE)
    chk(PROBE_MIN_FLIP > 0.05,
        "the intervention changes at least 5% of meta-steps at EVERY group -- "
        "the batch is NOT a null by construction", "%.4f" % PROBE_MIN_FLIP)
    chk(PROBE_MAX_SIGNVOTE < 0.01,
        "an EQUAL VOTE PER TENSOR agrees with the unnormalised sum in >99% of "
        "steps everywhere -- the sum's sign is NOT hijacked by the large "
        "tensors", "worst %.4f" % PROBE_MAX_SIGNVOTE)
    for spec in sorted(PROBE):
        for gi in sorted(PROBE[spec]):
            g = PROBE[spec][gi]
            print("     %-9s g%d %2d tensors (%d 1-D BN): dominance %.4f, "
                  "unanimous %.4f," % (spec, gi, g["tensors"], g["bn_tensors"],
                                       g["dominance"], g["unanimous"]))
            print("               argmax %s holds %.4f of sum|t|; BN share "
                  "%.4f -> %.4f under tn:; 1/n flips %.4f"
                  % (g["argmax"], g["argmax_share_mean"], g["bn_share_std"],
                     g["bn_share_tn"], g["flip_1_over_n"]))
            chk(g["dominance"] > 0.95 and g["unanimous"] < 0.95,
                "%-9s g%d sign follows the largest term while the terms are "
                "NOT unanimous" % (spec, gi),
                "dom %.4f unan %.4f" % (g["dominance"], g["unanimous"]))
            chk(g["float32_sign_error"] == 0.0,
                "%-9s g%d float32 accumulation reproduces the float64 sign at "
                "every step" % (spec, gi))
            chk(g["bn_share_tn"] > 4 * g["bn_share_std"],
                "%-9s g%d `tn:` multiplies the BN share of sum|t| by %.1fx -- "
                "it INVERTS the weighting, it does not neutralise it"
                % (spec, gi, g["bn_share_tn"] / g["bn_share_std"]),
                "%.4f -> %.4f" % (g["bn_share_std"], g["bn_share_tn"]))
    print("     tensor 50's STEERING POWER in the [50,12] COARSE group:")
    print("       standard %.4f of steps, tn: %.4f -- a %.0fx increase."
          % (T50_STEER_STD, T50_STEER_TN, T50_STEER_TN / T50_STEER_STD))
    chk(T50_STEER_STD < 0.001,
        "REMOVING tensor 50 from the [50,12] coarse group flips that group's "
        "sign in <0.1%% of steps, yet moving it across the cut costs %.4f pp"
        % abs(D_STD_CORPUS), "%.4f" % T50_STEER_STD)
    chk(abs(D_STD_CORPUS) / SE_ARM > 40.0,
        "so THE CUT-POSITION CLIFF IS NOT A REDUCTION EFFECT: the briefing's "
        "collapse prediction is refuted at the mechanism level, for 0 GPU-h",
        "%.2f SE_ARM" % (abs(D_STD_CORPUS) / SE_ARM))

    print("\n%d PASS, %d FAIL" % (NPASS[0], NFAIL[0]))
    return 1 if NFAIL[0] else 0


# =============================================================================
def score(runsdir, manifest=None):
    print("=" * 78)
    print("cX1_crn1_score.py -- crn1, THE REDUCTION ITSELF")
    print("runsdir %s" % runsdir)
    print("=" * 78)

    recs = load_runs(runsdir)
    good = [r for r in recs if not r.get("bad_name")]
    bad = [r for r in recs if r.get("bad_name")]

    # ---- G0 PROVENANCE -----------------------------------------------------
    print("\nG0 PROVENANCE (each run's OWN ARGS and ENV lines)")
    g0 = True
    print("   %d .out files matched, %d rejected by the name pattern"
          % (len(good), len(bad)))
    for r in bad:
        print("   !!! unparsable name: %s" % os.path.basename(r["file"]))
        g0 = False
    if len(good) != N_RUNS:
        print("   !!! expected %d runs, found %d" % (N_RUNS, len(good)))
        g0 = False
    seen = collections.Counter((r["arm"], r["seed"]) for r in good)
    for a in ARMS:
        for s in SEEDS:
            if seen[(a, s)] != 1:
                print("   !!! (%s, s%d) appears %d times" % (a, s, seen[(a, s)]))
                g0 = False
    envs = collections.Counter(r["env"] for r in good if r["env"])
    print("   distinct ENV lines: %d" % len(envs))
    for e, n in envs.most_common():
        print("     [%2d] %s" % (n, e))
    if len(envs) != 1:
        print("   !!! the ENV line is not constant across the batch")
        g0 = False
    for e in envs:
        for tok in ("PROBE=0", "BETA_CLIP=" + CLIP_C, "AUGMENT=" + AUG,
                    "HIER=none"):
            if tok not in e:
                print("   !!! ENV lacks %s" % tok)
                g0 = False
    for r in good:
        if r["repeated"]:
            print("   !!! %s repeats flags %s" % (os.path.basename(r["file"]),
                                                  r["repeated"]))
            g0 = False
        eff = r["eff"]
        want = {"stepsize-groups": SPEC[r["arm"]], "seed": str(r["seed"]),
                "alg-base": BASE_ALG, "alg-meta": META_ALG,
                "meta-stepsize": MS, "alpha0": ALPHA0, "gamma": GAMMA,
                "dataset": DSET, "NN-name": NET,
                "num-epochs": str(EPOCHS), "batch-size": str(BATCH_SIZE),
                "run-name": "crn1-%s-s%d" % (r["arm"], r["seed"])}
        for k, v in sorted(want.items()):
            if eff.get(k) != v:
                print("   !!! %s: --%s is %r, registered %r"
                      % (os.path.basename(r["file"]), k, eff.get(k), v))
                g0 = False
        if not r["run_done"]:
            print("   !!! %s has no RUN_DONE" % os.path.basename(r["file"]))
            g0 = False
    print("   G0 %s" % ("PASS" if g0 else "FAIL"))

    # ---- G1 THE PARTITIONS, from the LIVE model via the manifest -----------
    print("\nG1 THE PARTITIONS AND THE INTERVENTION (live model, via manifest)")
    mpath, how = resolve_manifest(runsdir, manifest)
    g1 = True
    print("   manifest: %s\n             (%s)" % (mpath, how))
    if not mpath or not os.path.exists(mpath):
        print("   !!! no manifest -- G1 CANNOT BE EVALUATED")
        g1 = False
    else:
        man = read_manifest(mpath)
        if man.get("T") != N_TENSORS or man.get("total") != TOTAL_PARAMS:
            print("   !!! live model is %s tensors / %s params, registered "
                  "%d / %d" % (man.get("T"), man.get("total"), N_TENSORS,
                               TOTAL_PARAMS))
            g1 = False
        nmis = 0
        for i in range(1, N_TENSORS + 1):
            nm, ne = man["tensors"].get(i, ("", -1))
            if nm != MODEL_NAMES[i - 1] or ne != MODEL_NUMEL[i - 1]:
                nmis += 1
                if nmis <= 5:
                    print("   !!! tensor %d live=%s/%d registered=%s/%d"
                          % (i, nm, ne, MODEL_NAMES[i - 1], MODEL_NUMEL[i - 1]))
        if nmis:
            print("   !!! %d tensors differ from the frozen model" % nmis)
            g1 = False
        else:
            print("   the live model matches the frozen 62-tensor table name "
                  "by name")
        for a in ARMS:
            d = man["specs"].get(a)
            if not d:
                print("   !!! no ARMSPEC for %s" % a)
                g1 = False
                continue
            if d["spec"] != SPEC[a]:
                print("   !!! %s spec live=%r registered=%r"
                      % (a, d["spec"], SPEC[a]))
                g1 = False
            want_rn = 1 if a in TN_ARMS else 0
            if d["rednorm"] != want_rn:
                print("   !!! %s REDNORM=%d, registered %d"
                      % (a, d["rednorm"], want_rn))
                g1 = False
            print("   %-5s %-24s m=%d sizes %-9s params %-21s rednorm=%d"
                  % (a, d["spec"], d["m"], ",".join(str(x) for x in d["sizes"]),
                     ",".join(str(x) for x in d["params"]), d["rednorm"]))
        for arm, sizes, mass in (("k49", K49_SIZES, K49_MASS),
                                 ("k50", K50_SIZES, K50_MASS)):
            for a in (arm, arm + "n"):
                d = man["specs"].get(a, {})
                if tuple(d.get("sizes", ())) != sizes \
                        or tuple(d.get("params", ())) != mass:
                    print("   !!! %s composed sizes/masses %s/%s, registered "
                          "%s/%s" % (a, d.get("sizes"), d.get("params"),
                                     sizes, mass))
                    g1 = False
        for a, b in PAIRS:
            if man["pairs"].get((a, b)) != 1:
                print("   !!! %s and %s are NOT the same partition on the live "
                      "model" % (a, b))
                g1 = False
            if man["reduction"].get((a, b)) != 1:
                print("   !!! `tn:` does NOT change %s's reduction" % b)
                g1 = False
        if man["pairs"] and man["reduction"]:
            print("   every `tn:` arm is the SAME PARTITION as its twin AND "
                  "its reduction differs")
        for s, v in sorted(man["inert"].items()):
            print("   `tn:` is BITWISE inert at %s: %s" % (s, bool(v)))
            if v != 1:
                g1 = False
    print("   G1 %s" % ("PASS" if g1 else "FAIL"))

    # ---- the arm levels ----------------------------------------------------
    lev, tra, sd, per = {}, {}, {}, {}
    for a in ARMS:
        v = [r["p5"] for r in good if r["arm"] == a and r["p5"] is not None]
        t = [r["t5"] for r in good if r["arm"] == a and r["t5"] is not None]
        per[a] = sorted((r["seed"], r["p5"]) for r in good if r["arm"] == a)
        lev[a] = statistics.mean(v) if v else None
        tra[a] = statistics.mean(t) if t else None
        sd[a] = statistics.stdev(v) if len(v) > 1 else None

    print("\nARM LEVELS -- plateau5 (PRIMARY), TRAIN alongside TEST at every arm")
    print("   %-5s %-24s %9s %8s %9s   per-seed TEST" % ("arm", "spec", "TEST",
                                                         "sd", "TRAIN"))
    for a in ARMS:
        ps = " / ".join(("%.3f" % x[1]) if x[1] is not None else "--"
                        for x in per[a])
        print("   %-5s %-24s %s %s %s   %s"
              % (a, SPEC[a], fmt(lev[a]), fmt(sd[a], 8, 4), fmt(tra[a]), ps))

    # ---- R2 / R3 / R-CTRL / R-FLOOR ---------------------------------------
    print("\nR2 ALIVE")
    alive = sum(1 for r in good if r["n_ep"] >= EPOCHS and r["p5"] is not None)
    r2 = alive == N_RUNS
    print("   %d/%d runs reached %d epochs with a full 5-epoch tail -> %s"
          % (alive, N_RUNS, EPOCHS, "PASS" if r2 else "FAIL"))

    print("\nR3 NOISE (bar %.4f pp = 3 * SIGMA_W)" % NOISY_BAR)
    worst_sd = max([x for x in sd.values() if x is not None] or [0.0])
    r3 = worst_sd <= NOISY_BAR
    for a in ARMS:
        print("   %-5s sd %s" % (a, fmt(sd[a], 8, 4)))
    print("   worst %.4f vs bar %.4f -> %s" % (worst_sd, NOISY_BAR,
                                               "PASS" if r3 else "FAIL"))

    print("\nR-CTRL the batch reproduces the granularity effect at all "
          "(bar %.4f)" % CTRL_BAR)
    ctrl = None
    if None not in (lev["k49"], lev["k01"]):
        ctrl = lev["k49"] - lev["k01"]
        print("   k49 - k01 = %+.6f  (%s)" % (ctrl, se_str(ctrl)))
    rctrl = ctrl is not None and ctrl >= CTRL_BAR
    print("   -> %s" % ("PASS" if rctrl else "FAIL"))

    print("\nR-FLOOR every arm's margin over the IN-BATCH m = 1 anchor "
          "(SE_ARM %.6f)" % SE_ARM)
    plateau = {}
    for a in ARMS:
        if None in (lev[a], lev["k01"]):
            continue
        m = lev[a] - lev["k01"]
        plateau[a] = m <= FLOOR_BAND
        print("   %-5s margin %+9.4f = %+7.2f SE_ARM   %s"
              % (a, m, m / SE_ARM,
                 "ON THE PLATEAU" if plateau[a] else "INFORMATIVE"))
    dead_arms = [a for a in TN_ARMS if tra[a] is not None and tra[a] < DEAD_BAR]
    print("   tn: arms with TRAIN below %.2f (did not train): %s"
          % (DEAD_BAR, dead_arms or "none"))

    # ---- THE PRIMARY: COMPOSITION -----------------------------------------
    print("\nPRIMARY -- COMPOSITION.  Three SAME-PARTITION pairs; the only")
    print("difference is who holds the vote inside the group.")
    print("   bar %.6f pp = 2 * SE_ARM_DIFF" % READ_BAR)
    toks, pairvals = [], {}
    for a, b in PAIRS:
        p = None if None in (lev[a], lev[b]) else lev[b] - lev[a]
        pairvals[(a, b)] = p
        if p is None:
            toks.append("I")
            print("   %-4s -> %-5s  P = (unavailable)" % (a, b))
            continue
        t = pair_token(p, plateau.get(b, False), plateau.get(a, False),
                       (a, b) == ONE_SIDED_PAIR)
        toks.append(t)
        print("   %-4s -> %-5s  P = %+9.6f  (%s)   %s  %s"
              % (a, b, p, se_str(p), t, PAIR_DOC[t].split(" ")[0]))
        print("        H-COMP-INERT predicts %+9.6f -> residual %+9.6f = %s  %s"
              % (PRED_PAIR["H-COMP-INERT"], p - PRED_PAIR["H-COMP-INERT"],
                 se_str(p - PRED_PAIR["H-COMP-INERT"]),
                 "WITHIN BAR" if abs(p) <= READ_BAR else "EXCLUDED"))
    comp = composition_verdict(toks, bool(dead_arms))
    print("   -> %s" % COMP_DOC[comp])

    # ---- THE SECONDARY: THE CLIFF -----------------------------------------
    print("\nSECONDARY -- the cut-position cliff under the two weightings")
    d_std = d_tn = inter = None
    if None not in (lev["k49"], lev["k50"], lev["k49n"], lev["k50n"]):
        d_std = lev["k50"] - lev["k49"]
        d_tn = lev["k50n"] - lev["k49n"]
        inter = d_tn - d_std
    print("   D_STD = k50  - k49  = %s  (%s)"
          % (fmt(d_std), se_str(d_std) if d_std is not None else ""))
    print("   D_TN  = k50n - k49n = %s  (%s)"
          % (fmt(d_tn), se_str(d_tn) if d_tn is not None else ""))
    print("   I     = D_TN - D_STD = %s  (%s)"
          % (fmt(inter), se_str(inter, SE_INT, "SE_INT")
             if inter is not None else ""))
    if d_std is not None:
        print("   corpus D_STD (same cell, 100 ep, crn1 excluded) %+.6f; "
              "in-batch replication residual %+.6f = %s"
              % (D_STD_CORPUS, d_std - D_STD_CORPUS,
                 se_str(d_std - D_STD_CORPUS, SE_INT, "SE_INT")))
        print("   registered vs measured")
        print("     H-CLIFF-GOV      predicts I = %+9.6f -> residual %+9.6f "
              "= %s  %s" % (PRED_I["H-CLIFF-GOV"], inter,
                            se_str(inter, SE_INT, "SE_INT"),
                            "WITHIN BAR" if abs(inter) <= INT_BAR
                            else "EXCLUDED"))
        print("     H-CLIFF-COLLAPSE predicts D_TN = %+9.6f -> residual "
              "%+9.6f = %s  %s"
              % (PRED_CLIFF_COLLAPSE_D_TN, d_tn - PRED_CLIFF_COLLAPSE_D_TN,
                 se_str(d_tn - PRED_CLIFF_COLLAPSE_D_TN),
                 "WITHIN BAR" if abs(d_tn) <= READ_BAR else "EXCLUDED"))
        print("     H-CLIFF-RED      predicts |D_TN| > |D_STD| + %.6f = "
              "%.6f -> %s" % (INT_BAR, abs(d_std) + INT_BAR,
                              "MET" if abs(d_tn) > abs(d_std) + INT_BAR
                              else "not met"))
    cliff_sat = bool(plateau.get("k49n", False) and plateau.get("k50n", False))
    cb = cliff_branch(d_tn, d_std, cliff_sat)
    print("   in-batch |D_STD| = %s vs CLIFF_MIN %.4f"
          % (fmt(abs(d_std) if d_std is not None else None), CLIFF_MIN))
    print("   -> branch %s  %s" % (cb, CLIFF_DOC[cb]))

    # ---- THE VERDICT -------------------------------------------------------
    print("\n" + "=" * 78)
    gates_ok = g0 and g1 and r2 and r3 and rctrl
    tokens = "/".join(toks)
    verdict = {"INERT": "COMPOSITION-INERT", "LIVE": "COMPOSITION-OPERATIVE",
               "SAT": "UNRESOLVED-COMPOSITION-SATURATED",
               "DEAD": "UNRESOLVED-INTERVENTION-FAILED"}[comp]
    cliff_tok = {"G": "CLIFF-GOVERNANCE", "R": "CLIFF-AMPLIFIED",
                 "K": "CLIFF-COLLAPSED", "P": "CLIFF-MOVED-UNREGISTERED",
                 "X": "UNRESOLVED-ANCHOR-FAILED",
                 "S": "UNRESOLVED-CLIFF-SATURATED"}[cb]
    if not gates_ok:
        verdict = "VOID-GATE-FAILED (" + verdict + ")"
    print("pair tokens %s" % tokens)
    print("FINAL: %s | %s" % (verdict, cliff_tok))
    print("=" * 78)
    print("SCOPE, registered in advance: 100 epochs only; m = 1 and the two")
    print("m = 2 cut positions 49 and 50 only; `tn:` is INERT at layerwise /")
    print("nodewise / weightwise / chunk* / permnode* BY CONSTRUCTION, so")
    print("nothing here bears on those granularities; NO claim that either")
    print("weighting is a better optimizer (`tn:` is deliberately the wrong")
    print("gradient); NO test of 152.12 rival (c) (PROBE=0); RULE 11 IS OPEN --")
    print("every arm sits at the single shared ms = 1e-3, and hz9 showed a")
    print("granularity contrast can reverse sign under per-arm tuning, so a")
    print("COMPOSITION-INERT verdict is a statement about THIS CELL, not about")
    print("the operator.")
    return 0 if (gates_ok and comp in ("INERT", "LIVE")) else 1


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runsdir", nargs="?", default=None)
    ap.add_argument("--manifest", default=None,
                    help="OPTIONAL; defaulted from <runsdir> (CORRECTIONS 164.2)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.runsdir:
        ap.error("give a runsdir, or --selftest")
    raise SystemExit(score(a.runsdir, a.manifest))


if __name__ == "__main__":
    main()
