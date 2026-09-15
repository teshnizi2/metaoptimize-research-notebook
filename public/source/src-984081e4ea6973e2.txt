#!/usr/bin/env python3
# =============================================================================
# cVK1_vggcut_score.py -- THE REGISTERED SCORER FOR `cvk1`, THE CUT-POSITION
#   LADDER ON VGG11_bn_c100 / CIFAR-100.  ONE SUBMISSION, 27 JOBS (9 arms x 3
#   seeds), 100 EPOCHS, SEEDS {55,56,57}, PROBE=0.
#
# REGISTERED ONLY.  NOTHING OF THIS BATCH HAS BEEN SUBMITTED.  The operator
# launches it, or does not; this file is committed BEFORE any cvk1 run exists
# (STANDING RULE 21) and is RUN UNEDITED (RULE 16).  Documented invocation,
# ONE argument:
#
#     python3 analysis/cVK1_vggcut_score.py <runsdir>
#
# --manifest and --csv are OPTIONAL and DEFAULTED (the 164.2 pattern).
# --selftest re-derives every frozen premise: the corpus noise floor, the
# ResNet premise from cpk1/cpk2/cpk3's RAW .out files, the VGG manifest
# arithmetic, the prefix-sign map from cvg1's probe records, the per-account
# predicted levels and the floor table, and it drives the SAME decide() the
# scoring path uses over synthetic curves to prove the branch map is total.
#
# =============================================================================
# THE PREDICTION UNDER TEST, AND THE PREMISE IT RESTS ON (re-derived, never
# quoted -- see --selftest section RESNET PREMISE)
# =============================================================================
# On ResNet18_c100 / CIFAR-100 the two-group ladder [k, 62-k] peaks at k* = 49
# in all five curves on disk: cpk1 @100 (55.4473), cpk2 @100 (55.4880) and
# @772 (56.4127), cpk3 @100 (55.2120) and @772 (56.0453).  The three nominated
# carriers are 1-based 50 layer4.0.bn2.weight, 53 layer4.0.shortcut.1.weight,
# 59 layer4.1.bn2.weight (185, by name on the live model).  So k* = 49 IS the
# last cut before the first carrier.  The curve's drops sit at the carriers:
#   49 -> 50  (moves carrier 50 alone)   cpk3 -24.4593 @100, -18.2707 @772
#                                        cpk2 -25.5127 @100, -20.4993 @772
#   49 -> 52  (moves 50,51,52)           cpk1 -17.1053 @100
#   52 -> 55  (moves carrier 53 + 2)     cpk1 -14.9700 @100
#   55 -> 60  (moves carrier 59 + 4)     cpk1  +1.3467 @100  (both arms floored)
# The one drop NOT at a carrier is 46 -> 47 (non-carrier layer4.0.bn1.weight),
# -2.5773 @100 / -2.7220 @772 -- a class-sized step, 7-9x smaller.
#
# A BRIEFING CORRECTION: "the -18.3 pp drop three tensors past the peak" joins
# two different measurements.  -18.2707 is cpk3's ONE-tensor step 49 -> 50 at
# 772 epochs; the THREE-tensor step 49 -> 52 is cpk1's -17.1053 at 100 epochs.
# The drop sits ONE tensor past the peak, at the first carrier itself.
#
# THE CONFOUND THE RESNET RECORD CANNOT BREAK.  k = 49 is ALSO the exact
# argmin over ALL k of |share(1..k) - 0.5| on ResNet18_c100 (share 0.562834;
# k = 50 is 0.562880, 4.6e-5 further).  cpk1's pre-registered PARAMETER-BALANCE
# rival (P5) predicted 49 and WON (146.2).  On ResNet the carrier-cut account
# and the balance account name the SAME k; nothing measured on ResNet can tell
# them apart.  VGG CAN: its balance argmin is k = 18 (16/17/18 within 1.1e-4),
# four cuts from the carrier cut at 22.
#
# THE PREDICTION (A1, CARRIER-CUT).  VGG11_bn_c100 has ONE nominated carrier,
# bn8.weight at 1-based 23 (194.5, re-derived at 198.4).  If "last cut before
# the first carrier" is not a ResNet coincidence, the ladder [k, 26-k] peaks at
#
#                               k* = 22
#
# A1 has two halves.  A1a CLIFF: moving the carrier into the big group (22 ->
# 23) collapses the arm.  A1b PURITY: for k < 22 the level RISES with k as the
# carrier's group sheds non-carrier mass (the ResNet left flank rises
# 17 -> 49 on every cpk1 step).  cvg1's own probe records support A1a and are
# SILENT on A1b: on the scalar arm's pinned records the prefix sum over 1..k is
# negative on 1.0000 of records for EVERY k in 4..22 and on 0.0000 at k = 23
# and 24 -- the sign boundary sits exactly at 22|23, but the map is FLAT over
# 4..22 and names no peak (H-DISAGREE, a sign-based peak locator, already died
# on ResNet at 179).
#
# =============================================================================
# THE ALIAS (161.7c(ii), 189 V2-V6): CLASS == INDEX MOD 3 ON BOTH NETS
# =============================================================================
# conv/linear weight at k = 1 mod 3, BN scale at 2, BN shift at 0, at every
# index 1..24 on VGG and 1..61 on ResNet; each net breaks it only at its last
# tensor (linear.bias).  Carriers are BN scales, so "the last cut before the
# first carrier" is ALWAYS a conv|BN-scale junction (k = 1 mod 3): 49 on
# ResNet, 22 on VGG.  What that does to the prediction:
#   (i)  PEAK-AT-22 cannot separate bn8.weight's IDENTITY from the CLASS of
#        the junction it closes.  No prefix cut can; only a name-list swap can
#        (cpr1's design, 162).  Stamped unconditionally.
#   (ii) But a pure class account names NO location -- every junction
#        k = 13, 16, 19, 22 is structurally identical -- so it needs a selection
#        rule, and the ladder CAN separate selection rules: carrier (22),
#        balance-with-parity (16), the previous junction (19).
#   (iii) A peak at 20 or 21 (k = 2, 0 mod 3) breaks the parity pattern
#        itself, independent of any carrier.
#   (iv) The window 19..23 moves conv7-terminal -> bn7.weight -> bn7.bias ->
#        conv8.weight -> bn8.weight: the exact class sequence of cpk3's window
#        46..50 (conv1, bn1.w, bn1.b, conv2, bn2.w-carrier).  Its four steps are
#        printed beside their ResNet analogues as a SIGN CENSUS ONLY (161's
#        rule); nothing branches on them.
#
# =============================================================================
# THE REGISTERED ACCOUNTS AND WHERE EACH PUTS THE PEAK
# =============================================================================
#   A1 CARRIER-CUT     k* = first carrier - 1 = 22.   ResNet: 49.
#   A2 PARAM-BALANCE   k* = argmin |share(1..k) - 0.5| = 18 (16, 17 within
#                      1.1e-4); 16 on this grid.  ResNet: 49.  cpk1's P5.
#   A4 DEPTH-FRACTION  k*/N = 49/62 -> 26 * 49/62 = 20.55: 20 or 21.
#   A5 NO-PEAK         monotone or flat in k; no interior optimum.
#   (A stage-boundary account -- cpk1's P1, the entry of the last stage,
#   18 on VGG_CFG11 -- was REFUTED on ResNet at 146.2 and is not re-registered;
#   on VGG it would sit beside A2's plateau.)
#
# PREDICTED LEVELS, capture units then pp at cvg1's measured anchors (k01
# 35.0173, kL 66.2860, gap 31.2687).  A1 maps each VGG cut by the non-carrier
# mass sharing the carrier's group onto cpk1's left flank (plus the isolation
# limit capture 1.0 at zero mass), with cpk3's class steps inside 19..21; A2
# maps |share - 0.5| onto cpk1's left flank; A4 re-indexes the whole cpk1
# curve by tensor fraction.  These are MODELS FROM A DIFFERENT ARCHITECTURE and
# are used for ONE thing only: the floor table.  No branch reads them.
#
#          k13     k16     k19     k20     k21     k22     k23    peak
#   A1   0.4076  0.6200  0.8194  0.7646  0.7669  0.9961  0.0425   22
#   A2   0.4076  0.8147  0.4243  0.4243  0.4242  0.0367  0.0366   16
#   A4   0.2190  0.2857  0.4267  0.5607  0.5658  0.2834  0.0302   21
#   A5   representative: INC 0.30 + 0.05(k-13); DEC the mirror; FLAT 0.50.
#
# =============================================================================
# THE FLOOR GATE (164.6) -- AND WHERE THE BRIEF'S RULE CANNOT BE MET LITERALLY
# =============================================================================
# The floor here is 164.6's floor: THIS batch's own m = 1 anchor k01, not
# chance.  "At or near" = within NEAR_FLOOR_PP = 3.0 pp of k01, the width of
# the only floor-saturation band this campaign has measured (164.6: 3.004 pp).
#
#   predicted AT/NEAR the floor:   A1 {k23}   A2 {k22, k23}   A4 {k23}   A5 {}
#   predicted PEAK above k01:      A1 k22 +31.15   A2 k16 +25.48   A4 k21 +17.69
#   lowest NON-floored prediction under any account: A4 k13 +6.85 pp
#   highest prediction under any account: A1 k22 66.16 pp, 33.8 pp below 100
#
# "Every arm's predicted level under every account must clear the floor" is
# UNSATISFIABLE for any design that can tell PEAK-AT-22 from NO-PEAK: the
# right flank of 22 is k in {23,24,25}, every one of which puts the ONLY
# nominated carrier in the big group, which A1 itself predicts collapses.
# Delete k23 and 22 becomes the grid's edge, where a peak and a monotone rise
# are the same measurement -- a primary bounded in one direction, the worse
# violation.  So the rule is honoured in the form 164.6 actually states it:
#   F1 EVERY account's PEAK arm is predicted >= 15 pp above k01, so every
#      account can FAIL through the arm that carries its claim.
#   F2 NO account's floored arm is that account's peak arm.
#   F3 The primary is an ARGMAX.  A floored arm can never BE the argmax unless
#      every arm is floored (-> ALL-CUTS-FLOORED, a registered outcome no
#      account predicts).  A floored arm enters a verdict ONLY by SIGN and
#      BOUND -- 164.6's "what may be written" -- never by agreeing with a point.
#   F4 k23's floor bounds it DOWNWARD only; every claim that k23 can refute
#      (A1a's cliff, NO-PEAK-INCREASING) is refuted by k23 landing HIGH.
#
# =============================================================================
# THE NOISE FLOOR, RE-DERIVED AT REGISTRATION WITH `cvk1-` EXCLUDED FROM EVERY
# READER (the 165.4 pattern)
# =============================================================================
# Pooled within-cell SD of plateau5 over scalar/layerwise rows at the standard
# cell (100 ep, AUGMENT=1, BETA_CLIP -15:-2.3026, ms 1e-3, alpha0 1e-6, batch
# 100), superseded/collapsed/incomplete dropped, cell = the 15 design columns:
#   SIGMA_VGG     0.400658  df   4    <- THIS architecture (cvg1's 6 runs)
#   SIGMA_NARROW  0.586232  df  77    <- ResNet18_c100 / CIFAR-100
#   SIGMA_WIDE    NOT USED (98.84 % of its SS is one bimodal cell; 189.5, 198.5)
#   SIGMA_PRIOR := max(SIGMA_VGG, SIGMA_NARROW) = 0.586232
#   SE_PRIOR     = SIGMA_PRIOR * sqrt(2/3) = 0.478656 pp
# At score time SIGMA_USED = max(SIGMA_PRIOR, SIGMA_NARROW_live, SIGMA_VGG_live,
# SIGMA_INBATCH) -- monotone, so corpus drift (cvi1's VGG anchors WILL land
# first) can only WIDEN the bars in SE.  SE_ARM_DIFF = SIGMA_USED*sqrt(2/3).
#
# =============================================================================
# THE PRIMARY -- THE ARGMAX LOCATION, BOUNDED IN NEITHER DIRECTION
# =============================================================================
#   k_hat = argmax over the SEVEN sweep arms of plateau5 (3-seed in-batch mean)
#   PEAK_SET = { k : M(k_hat) - M(k) <= PEAK_BAR },  PEAK_BAR = 2 SE_ARM_DIFF
# It is categorical over the whole grid.  Every grid position is a reachable
# verdict (the selftest drives a spike at each), so it can disagree with k* =
# 22 in both directions; and 22 is INTERIOR (21 and 23 both measured), so a
# peak at 22 and a monotone rise are different outcomes.
#
# BARS (frozen; SE at the prior):
#   PEAK_BAR      2 SE   = 0.957 pp   argmax separation (cpk3's R-CTRL rule)
#   FLOOR_BAR     2 SE   = 0.957 pp   an arm within it of k01 is FLOORED
#   FLAT_BAR      max(0.10 * D_GAP, 4 SE) = 3.127 pp at cvg1's gap -- cpk1's
#                 X1 NO-POSITION-EFFECT rule (0.10 of the gap)
#   GAP_MIN       10.0 pp  G-GAP: the anchors must reproduce a gap, or the
#                 ladder has no range to localise in
#   DIVERGED_BAR  5.0 pp   within-arm seed range; a bimodal arm suspends
#   FLOOR_MIN 15.0 / CEIL_MAX 90.0   harness gates (chance 1.00)
#
# =============================================================================
# THE BRANCH MAP, FROZEN.  First match wins.  Gates suspend; a failed gate is
# NOT an adverse result.
# =============================================================================
#   HARNESS-UNSOUND      max arm < 15.00 pp.
#   UNRESOLVED-DIVERGED  an arm's seed range > 5.0 pp.
#   UNRESOLVED-NO-GAP    kL - k01 < 10.0 pp in batch.
#   ALL-CUTS-FLOORED     every sweep arm within FLOOR_BAR of k01.  No two-group
#     contiguous cut recovers anything on VGG.  A1, A2 and A4 are all REFUTED
#     (each predicts its peak >= 17 pp above the floor); every level is a BOUND.
#   NO-PEAK-FLAT         sweep RANGE < FLAT_BAR.  Moving the cut across the
#     whole grid buys < a tenth of the gap: cut position is NOT a localising
#     variable on VGG.  A SCOPE LIMIT on 199.10(4), graded publishable-as-is.
#   (then k_hat, PEAK_SET)
#   UNRESOLVED-TIE-WITH-22  |PEAK_SET| > 1 and 22 in it.  A1 neither supported
#     nor refuted; the tie set is printed.
#   PEAK-ELSEWHERE-TIE      |PEAK_SET| > 1 and 22 NOT in it.  A1 REFUTED (22 is
#     more than PEAK_BAR below the best); a tie set within {20,21} is A4's own
#     prediction and is stamped DEPTH-FRACTION-FAVOURED.
#   PEAK-AT-22           PEAK_SET == {22}.  A1 SUPPORTED: the argmax is the
#     last cut before the carrier and the right flank (23) sits below it.
#     A2 and A4 are refuted by the same inequality.
#   NO-PEAK-INCREASING   PEAK_SET == {23} and no adjacent step falls by more
#     than PEAK_BAR.  Moving the ONLY carrier into the big group does not
#     hurt: A1a is refuted at its core, and position is not localising.
#   PEAK-AT-RIGHT-EDGE-23  PEAK_SET == {23}, not monotone.  A1 refuted; the
#     optimum may lie at 24/25, unmeasured.
#   NO-PEAK-DECREASING   PEAK_SET == {13}, no adjacent step rises by more than
#     PEAK_BAR.  The smaller the first group the better: no account here
#     predicts it; position is not localising in the carrier's sense.
#   PEAK-AT-LEFT-EDGE-13 PEAK_SET == {13}, not monotone; optimum may be < 13.
#   PEAK-ELSEWHERE-k     PEAK_SET == {k}, k interior and != 22.  A1 REFUTED:
#     k = 21  conv8 belongs WITH the carrier; the conv step's sign (+9.0 on
#             ResNet, 15/15 in 161's census) does NOT transfer; parity broken;
#             A4 (20.55) is the surviving account.
#     k = 20  the peak closes a BN scale (k = 2 mod 3): parity itself broken;
#             A4 survives.
#     k = 19  the PREVIOUS conv|BN-scale junction: parity holds, the carrier
#             rule fails by one junction -- bn7 (0.0021 of mean|L|, no sign
#             weight on the record) would have to travel with bn8.
#     k = 16  A2 WINS: the pre-registered ResNet rival (cpk1 P5) locates VGG's
#             peak too, and ResNet's k* = 49 was the BALANCE point that
#             happened to sit before a carrier -- the carrier-cut reading of
#             199.10(4)'s localisation dies.
#
# STAMPS, NEVER THE BRANCH.  Unconditional: ALIAS-CLASS-IS-INDEX-MOD-3,
# MAGNITUDE-NOT-SEPARATED (bn8 holds 0.5783 of mean|L|), HORIZON-100-ONLY,
# SHARED-MS-1E-3, NOT-A-ONE-VARIABLE-ABLATION.  Conditional: GATES-CLEAN,
# SIGMA-*, TRAIN-ARGMAX-AGREES/DISAGREES, D22-16 sign (the A1-vs-A2
# discriminator, bounded in neither direction), CLIFF-POINT / CLIFF-IS-A-BOUND
# (k23 floored), GAP-REPLICATES-CVG1 (BETWEEN-BATCH, NON-GATING).
#
# =============================================================================
# WHAT cvk1 CANNOT LICENSE, WRITTEN BEFORE ANY RUN EXISTS
# =============================================================================
# * NOT carrier IDENTITY vs junction CLASS (the alias), and NOT identity vs
#   MAGNITUDE (bn8 holds 0.5783 of the mean|L| mass, bn7 0.0021; 198.9).
# * NOT the ISOLATION result.  A peak at 22 is a cut-position fact.  It may not
#   travel with the ResNet isolation (197.8 / 199.10(3)), and cvi1's verdict,
#   whatever it is, is BETWEEN-BATCH and enters no contrast here.
# * NOTHING about residual connections (189.2, 194.6), about any horizon but
#   100 epochs (156's rescoping applies: ResNet's k* needed cpk2/cpk3 at 772),
#   or about any ms but 1e-3 (199.10(2)).  RULE 11 is untested on VGG.
# * NOTHING about group COUNT: every sweep arm is m = 2 (160).
#
# DISCLOSURE.  Visible when this file was written: results/all_runs.csv at
# 2,797 rows (HEAD 5d31aea); cpk1 (39), cpk2 (18) and cpk3 (21) raw .out files
# and cdep1's and cvg1's PARTITION-MANIFEST.txt; cvg1's probe records;
# CORRECTIONS 146-199; analysis/cVI1_vggiso_identity_score.py and
# bin/cVI1_vgg_isolate.sh as convention templates.  cvi1 had been SUBMITTED by
# another track (12 jobs, 2 running, 0 landed, 0 rows); no cvi1 number is used
# anywhere here.  NO cvk1 run existed anywhere and NO cvk1 job was submitted.
# plateau5 is PRIMARY; the CSV `plateau` column is read NOWHERE; best_test is
# used NOWHERE; TRAIN is printed beside TEST at every arm.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

# ---- the batch design, LITERAL.  The launcher holds the same strings and its
# ---- guard 4c' proves the two agree, so neither can drift. -------------------
PREFIX = "cvk1"
NET = "VGG11_bn_c100"
DSET = "CIFAR100"
EPOCHS = 100
BATCH = 100
SEEDS = (55, 56, 57)
CLIP = "-15:-2.3026"
MST = "1e-3"
A0 = "1e-6"
AUG = "1"
PROBE = 0
GRID = (13, 16, 19, 20, 21, 22, 23)
KSTAR_PRED = 22
SPEC = {
    "k01": "scalar",
    "kL": "layerwise",
    "k13": "[13,13]",
    "k16": "[16,10]",
    "k19": "[19,7]",
    "k20": "[20,6]",
    "k21": "[21,5]",
    "k22": "[22,4]",
    "k23": "[23,3]",
}
ARMS = ("k01", "kL") + tuple("k%d" % k for k in GRID)

# ---- the live manifest this batch assumes, re-derived by name and numel ------
NTENS = 26
TOTPAR = 9274532
CARRIER = ("bn8.weight", 23, 512)
BN512_IDX = (14, 17, 20, 23)
# parameters in group 1 = tensors 1..k (contiguous, named_parameters() order)
WANT_G1 = {13: 2141248, 16: 4501568, 17: 4502080, 18: 4502592, 19: 6861888,
           20: 6862400, 21: 6862912, 22: 9222208, 23: 9222720}
A2_ARGMIN = 18
A4_POINT = 26.0 * 49.0 / 62.0            # 20.548387

# ---- the ResNet premise, frozen from an independent parse of the RAW .out ----
RES_ARGMAX = 49
RES_CURVES = {("cpk1", "100"): 55.4473, ("cpk2", "100"): 55.4880,
              ("cpk2", "E"): 56.4127, ("cpk3", "100"): 55.2120,
              ("cpk3", "E"): 56.0453}
RES_CARRIERS = {50: "layer4.0.bn2.weight", 53: "layer4.0.shortcut.1.weight",
                59: "layer4.1.bn2.weight"}
RES_STEP_49_50 = {("cpk2", "100"): -25.5127, ("cpk2", "E"): -20.4993,
                  ("cpk3", "100"): -24.4593, ("cpk3", "E"): -18.2707}
RES_CPK1_STEPS = {(49, 52): -17.1053, (52, 55): -14.9700, (55, 60): 1.3467}
RES_SHARE_ARGMIN = 49
# cpk3 @100, the class-matched window; VGG step (a -> b) <-> ResNet step
WINDOW = ((19, 20, "bn7.weight", 46, 47, "layer4.0.bn1.weight", -2.5773),
          (20, 21, "bn7.bias", 47, 48, "layer4.0.bn1.bias", 0.1087),
          (21, 22, "conv8.weight", 48, 49, "layer4.0.conv2.weight", 9.0020),
          (22, 23, "bn8.weight", 49, 50, "layer4.0.bn2.weight", -24.4593))
# cpk1 in-batch plateau5 levels (its own anchors k01 / k62), used by the
# selftest to RE-DERIVE the predicted-level tables below
CPK1_LEVELS = {1: 22.7207, 17: 25.9493, 24: 27.4160, 31: 33.0093, 38: 35.9507,
               42: 41.0720, 45: 42.2393, 47: 45.6880, 49: 55.4473, 52: 38.3420,
               55: 23.3720, 60: 24.7187, 62: 69.7107}
RES_SHARE1 = {17: 0.019936, 24: 0.033865, 31: 0.086478, 38: 0.142081,
              42: 0.194719, 45: 0.247333, 47: 0.352515, 49: 0.562834,
              52: 0.574607, 55: 0.784972, 60: 0.995428}

# ---- the prefix-sign map on cvg1's scalar pinned records, frozen -------------
PREFIX_NEG = {13: 1.0, 16: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 0.0}
PREFIX_TOL = 5e-3

# ---- predicted captures per account (models; used for the FLOOR TABLE only) --
PRED = {
    "A1": {13: 0.4076, 16: 0.6200, 19: 0.8194, 20: 0.7646, 21: 0.7669, 22: 0.9961, 23: 0.0425},
    "A2": {13: 0.4076, 16: 0.8147, 19: 0.4243, 20: 0.4243, 21: 0.4242, 22: 0.0367, 23: 0.0366},
    "A4": {13: 0.2190, 16: 0.2857, 19: 0.4267, 20: 0.5607, 21: 0.5658, 22: 0.2834, 23: 0.0302},
    "A5-INC": dict((k, 0.30 + 0.05 * (k - 13)) for k in GRID),
    "A5-DEC": dict((k, 0.80 - 0.05 * (k - 13)) for k in GRID),
    "A5-FLAT": dict((k, 0.50) for k in GRID),
}
PRED_TOL = 5e-3
AT_FLOOR_DECLARED = {"A1": {23}, "A2": {22, 23}, "A4": {23},
                     "A5-INC": set(), "A5-DEC": set(), "A5-FLAT": set()}
PRED_BRANCH = {"A1": "PEAK-AT-22", "A2": "PEAK-ELSEWHERE-16",
               "A4": "PEAK-ELSEWHERE-TIE", "A5-INC": "NO-PEAK-INCREASING",
               "A5-DEC": "NO-PEAK-DECREASING", "A5-FLAT": "NO-PEAK-FLAT"}
NEAR_FLOOR_PP = 3.0          # 164.6's measured saturation-band width, 3.004 pp
PEAK_MARGIN_MIN = 15.0       # F1: every account's peak arm >= this above k01

# ---- the noise floor and the bars, frozen ------------------------------------
SIGMA_VGG = 0.400658
SIGMA_NARROW = 0.586232
SIGMA_PRIOR = SIGMA_NARROW   # = max(SIGMA_VGG, SIGMA_NARROW)
SE_PRIOR = 0.478656          # SIGMA_PRIOR * sqrt(2/3)
PEAK_SE = 2.0
FLOOR_SE = 2.0
FLAT_FRAC = 0.10
FLAT_MIN_SE = 4.0
GAP_MIN = 10.0
DIVERGED_BAR = 5.0
FLOOR_MIN = 15.0
CEIL_MAX = 90.0
CHANCE = 1.0

# ---- this cell's measured levels (cvg1, seeds 31-33), frozen; BETWEEN-BATCH --
CVG1_K01 = 35.0173
CVG1_KL = 66.2860
CVG1_D = 31.2687
NOM_SHARE_BN8 = 0.5783
NOM_SHARE_BN7 = 0.0021

CELLKEYS = ["network", "dataset", "granularity", "base", "meta", "meta_stepsize",
            "alpha0", "gamma", "augment", "beta_clip", "batch_size",
            "epochs_requested", "hier", "lam", "eta_ratio"]

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "all_runs.csv")

EPTR_RE = re.compile(r"Epoch\s+(\d+),\s*Train Accuracy:\s*([0-9.]+)\s*%,"
                     r"\s*Test Accuracy:\s*([0-9.]+)")
OUT_RE = re.compile(r"^cvk1-(k01|kL|k\d\d)-s(\d+)-(\d+)\.out$")
CPK_RE = re.compile(r"^(cpk[123])-k(\d+)-s(\d+)-(\d+)\.out$")
ENV_RE = re.compile(r"^ENV:")
PT_ON_RE = re.compile(r"^PROBE_TENSOR: on ")

GATE_FAIL = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   " + extra) if extra else ""))
    if not cond:
        GATE_FAIL.append(label)
    return bool(cond)


def note(label, extra=""):
    print("  NOTE %s%s" % (label, ("   " + extra) if extra else ""))


def skip(label, extra=""):
    print("  SKIP %s%s" % (label, ("   " + extra) if extra else ""))


def fmt(x, nd=4):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def interp(x, pts):
    """piecewise-linear through sorted (x, y), linear extrapolation at ends"""
    pts = sorted(pts)
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    elif x >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
    else:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                break
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def tensor_class(name):
    if name.endswith(".bias"):
        return "shift" if ("bn" in name or "shortcut.1" in name) else "linbias"
    if "bn" in name or "shortcut.1" in name:
        return "scale"
    return "weight"


def alias_violations(tens):
    want = {1: "weight", 2: "scale", 0: "shift"}
    return [i for i, n, _ in tens if tensor_class(n) != want[i % 3]]


# =============================================================================
# THE DECISION -- one pure function, used by score() AND by the selftest's
# synthetic curves, so the branch map that is proved total is the one that runs
# =============================================================================
def decide(M, k01, se, d_gap):
    """M: {k: mean plateau5} over GRID.  -> (branch, k_hat, peak_set, info)"""
    peak_bar = PEAK_SE * se
    floor_bar = FLOOR_SE * se
    flat_bar = max(FLAT_FRAC * d_gap, FLAT_MIN_SE * se)
    info = {"peak_bar": peak_bar, "floor_bar": floor_bar, "flat_bar": flat_bar}
    floored = [k for k in GRID if M[k] - k01 < floor_bar]
    info["floored"] = floored
    if len(floored) == len(GRID):
        return "ALL-CUTS-FLOORED", None, [], info
    rng = max(M[k] for k in GRID) - min(M[k] for k in GRID)
    info["range"] = rng
    if rng < flat_bar:
        return "NO-PEAK-FLAT", None, [], info
    k_hat = max(GRID, key=lambda k: M[k])
    peak_set = [k for k in GRID if M[k_hat] - M[k] <= peak_bar]
    if len(peak_set) > 1:
        if KSTAR_PRED in peak_set:
            return "UNRESOLVED-TIE-WITH-22", k_hat, peak_set, info
        return "PEAK-ELSEWHERE-TIE", k_hat, peak_set, info
    if k_hat == KSTAR_PRED:
        return "PEAK-AT-22", k_hat, peak_set, info
    steps = [M[b] - M[a] for a, b in zip(GRID, GRID[1:])]
    if k_hat == GRID[-1]:
        mono = all(s >= -peak_bar for s in steps)
        return ("NO-PEAK-INCREASING" if mono else "PEAK-AT-RIGHT-EDGE-23"), k_hat, peak_set, info
    if k_hat == GRID[0]:
        mono = all(s <= peak_bar for s in steps)
        return ("NO-PEAK-DECREASING" if mono else "PEAK-AT-LEFT-EDGE-13"), k_hat, peak_set, info
    return "PEAK-ELSEWHERE-%d" % k_hat, k_hat, peak_set, info


def carrier_token(branch):
    if branch == "PEAK-AT-22":
        return "CARRIER-CUT-SUPPORTED"
    if branch == "UNRESOLVED-TIE-WITH-22":
        return "CARRIER-CUT-UNRESOLVED"
    if branch in ("HARNESS-UNSOUND", "UNRESOLVED-DIVERGED", "UNRESOLVED-NO-GAP"):
        return "CARRIER-CUT-NOT-TESTED"
    return "CARRIER-CUT-REFUTED"


def rival_token(branch, peak_set):
    if branch == "PEAK-AT-22":
        return "A2-A4-REFUTED"
    if branch == "PEAK-ELSEWHERE-16":
        return "PARAM-BALANCE-FAVOURED"
    if branch in ("PEAK-ELSEWHERE-20", "PEAK-ELSEWHERE-21"):
        return "DEPTH-FRACTION-FAVOURED"
    if branch == "PEAK-ELSEWHERE-TIE":
        return ("DEPTH-FRACTION-FAVOURED" if set(peak_set) <= {20, 21}
                else "NO-SINGLE-RIVAL")
    if branch == "PEAK-ELSEWHERE-19":
        return "EARLIER-JUNCTION"
    if branch.startswith("NO-PEAK"):
        return "POSITION-NOT-LOCALISING"
    if branch == "ALL-CUTS-FLOORED":
        return "NO-CUT-RESCUES"
    if branch.startswith("PEAK-AT-") and "EDGE" in branch:
        return "OPTIMUM-OFF-GRID"
    if branch == "UNRESOLVED-TIE-WITH-22":
        return "TIE-" + "-".join(str(k) for k in peak_set)
    return "NONE"


# =============================================================================
# corpus readers -- `cvk1-` EXCLUDED FROM EVERY ONE
# =============================================================================
def read_corpus(csvpath):
    if not csvpath or not os.path.exists(csvpath):
        return None
    return [r for r in csv.DictReader(open(csvpath))
            if not r.get("run", "").startswith(PREFIX + "-")]


def usable(r):
    return (r.get("superseded") == "0" and r.get("collapsed") == "0"
            and r.get("complete") == "1" and r.get("plateau5", "").strip() != "")


def std_cell(r):
    return (r.get("granularity") in ("scalar", "layerwise")
            and r.get("epochs_requested") == str(EPOCHS)
            and r.get("augment") == AUG and r.get("beta_clip") == CLIP
            and r.get("meta_stepsize") == MST and r.get("alpha0") == A0
            and r.get("batch_size") == str(BATCH))


def pooled_sigma(rows, extra=lambda r: True):
    cells = {}
    for r in rows:
        if usable(r) and std_cell(r) and extra(r):
            cells.setdefault(tuple(r[k] for k in CELLKEYS), []).append(float(r["plateau5"]))
    ss, df, nc, nm = 0.0, 0, 0, 0
    for v in cells.values():
        if len(v) < 2:
            continue
        m = mean(v)
        ss += sum((x - m) ** 2 for x in v)
        df += len(v) - 1
        nc += 1
        nm += len(v)
    return (math.sqrt(ss / df) if df else None), df, nc, nm


def corpus_gap(rows, net, dset):
    s, l = [], []
    for r in rows:
        if usable(r) and std_cell(r) and r["network"] == net and r["dataset"] == dset:
            (s if r["granularity"] == "scalar" else l).append(float(r["plateau5"]))
    if not s or not l:
        return None
    return mean(s), mean(l), mean(l) - mean(s)


# =============================================================================
# run readers -- the RAW .out files are PRIMARY
# =============================================================================
def parse_out(path):
    txt = open(path, errors="replace").read()
    eps = dict((int(a), (float(b), float(c))) for a, b, c in EPTR_RE.findall(txt))
    return txt, eps


def plateau(eps, last):
    ks = range(last - 4, last + 1)
    if not all(k in eps for k in ks):
        return None, None
    return mean([eps[k][1] for k in ks]), mean([eps[k][0] for k in ks])


def read_runs(runsdir):
    got = {}
    if not runsdir or not os.path.isdir(runsdir):
        return got
    for fn in sorted(os.listdir(runsdir)):
        m = OUT_RE.match(fn)
        if not m:
            continue
        arm, seed, jid = m.group(1), int(m.group(2)), m.group(3)
        txt, eps = parse_out(os.path.join(runsdir, fn))
        te, tr = plateau(eps, EPOCHS - 1)
        rec = {"file": fn, "job": jid, "arm": arm, "seed": seed,
               "n_epochs": len(eps),
               "traceback": "Traceback (most recent call last)" in txt,
               "run_done": "RUN_DONE" in txt,
               "env": [ln for ln in txt.splitlines() if ENV_RE.match(ln)],
               "pt_on": [ln for ln in txt.splitlines() if PT_ON_RE.match(ln)],
               "args": [ln for ln in txt.splitlines() if ln.startswith("ARGS:")],
               "plateau5": te, "train5": tr}
        prev = got.get((arm, seed))
        if prev is None or int(jid) > int(prev["job"]):
            got[(arm, seed)] = rec
    return got


def read_manifest(path):
    if not path or not os.path.exists(path):
        return None
    d = {"tensors": [], "armspec": {}, "raw": {}}
    for ln in open(path):
        p = ln.split()
        if not p:
            continue
        if p[0] == "TENSOR" and len(p) >= 4:
            d["tensors"].append((int(p[1]), p[2], int(p[3])))
        elif p[0] == "ARMSPEC" and len(p) >= 3:
            d["armspec"][p[1]] = " ".join(p[2:])
        elif len(p) >= 2:
            d["raw"][p[0]] = " ".join(p[1:])
    return d


def resolve_manifest(runsdir, explicit):
    if explicit:
        return explicit, "explicit --manifest"
    for cand, how in (
        (os.path.join(runsdir, PREFIX, "PARTITION-MANIFEST.txt"), "<runsdir>/cvk1/"),
        (os.path.join(runsdir, PREFIX + "-PARTITION-MANIFEST.txt"), "<runsdir>/cvk1-"),
    ):
        if os.path.exists(cand):
            return cand, how
    return None, "not found"


# =============================================================================
# premise re-derivations (selftest) -- no code shared with any other scorer
# =============================================================================
def rederive_resnet(runsdir):
    """-> dict or None.  Reads cpk1/cpk2/cpk3 RAW .out under runsdir and
    cdep1's live-model manifest."""
    if not runsdir or not os.path.isdir(runsdir):
        return None
    man = read_manifest(os.path.join(runsdir, "cdep1", "PARTITION-MANIFEST.txt"))
    runs = {}
    for fn in os.listdir(runsdir):
        m = CPK_RE.match(fn)
        if not m:
            continue
        b, k, s = m.group(1), int(m.group(2)), int(m.group(3))
        txt, eps = parse_out(os.path.join(runsdir, fn))
        if "RUN_DONE" not in txt or not eps:
            continue
        runs.setdefault(b, {}).setdefault(k, []).append(eps)
    if not runs or man is None:
        return None
    out = {"curves": {}, "levels": {}, "man": man, "nruns": sum(
        len(v) for d in runs.values() for v in d.values())}
    for b, d in runs.items():
        lastE = max(max(e) for v in d.values() for e in v)
        for hz, last in (("100", 99), ("E", lastE)):
            if hz == "E" and lastE == 99:
                continue
            lv = {}
            for k, v in d.items():
                vals = [plateau(e, last)[0] for e in v]
                if None in vals or len(vals) != 3:
                    continue
                lv[k] = mean(vals)
            if lv:
                out["levels"][(b, hz)] = lv
    return out


def rederive_prefix_sign(runsdir):
    base = None
    for cand in (os.path.join(runsdir, "cvg1"), runsdir):
        if cand and os.path.isdir(cand) and any(
                d.startswith("probe_cvg1-k01-s") for d in os.listdir(cand)):
            base = cand
            break
    if base is None:
        return None
    acc, nseeds, npin = dict((k, 0.0) for k in PREFIX_NEG), 0, 0
    for d in sorted(os.listdir(base)):
        if not d.startswith("probe_cvg1-k01-s"):
            continue
        p = os.path.join(base, d, "probe.jsonl")
        if not os.path.exists(p):
            continue
        pin = []
        for ln in open(p):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if (r.get("n_beta", 0) > 0 and r.get("n_at_lo", 0) == r.get("n_beta")
                    and isinstance(r.get("m_tensor"), list) and len(r["m_tensor"]) == NTENS):
                pin.append(r)
        if not pin:
            continue
        nseeds += 1
        npin += len(pin)
        for k in PREFIX_NEG:
            neg = 0
            for r in pin:
                b2 = r.get("pt_b2", 0.9)
                L = [b2 * m + (1.0 - b2) * z for m, z in zip(r["m_tensor"], r["z_tensor"])]
                if sum(L[:k]) < 0:
                    neg += 1
            acc[k] += neg / float(len(pin))
    if not nseeds:
        return None
    return dict((k, v / nseeds) for k, v in acc.items()), npin, nseeds


def model_predictions(levels=CPK1_LEVELS, res_share=RES_SHARE1, vgg_share=None,
                      cpk3_step_47=-2.5773, cpk3_step_48=0.1087):
    """Re-derive PRED[A1/A2/A4] from a cpk1 curve and the two share tables."""
    k01, k62 = levels[1], levels[62]
    cap = dict((k, (levels[k] - k01) / (k62 - k01)) for k in res_share)
    vs = vgg_share or dict((k, WANT_G1[k] / float(TOTPAR)) for k in GRID)
    left = (17, 24, 31, 38, 42, 45, 47, 49)
    a1pts = [(1.0 - res_share[k], cap[k]) for k in left] + [(0.0, 1.0)]
    gap = k62 - k01
    A1 = {}
    for k in GRID:
        if k <= 22:
            A1[k] = interp(1.0 - vs[k] - 512.0 / TOTPAR, a1pts)
    A1[20] = A1[19] + cpk3_step_47 / gap
    A1[21] = A1[20] + cpk3_step_48 / gap
    A1[23] = cap[60]
    a2pts = [(abs(res_share[k] - 0.5), cap[k]) for k in left]
    A2 = dict((k, interp(abs(vs[k] - 0.5), a2pts)) for k in GRID)
    a4pts = sorted((k, cap[k]) for k in res_share)
    A4 = dict((k, interp(k * 62.0 / 26.0, a4pts)) for k in GRID)
    return {"A1": A1, "A2": A2, "A4": A4}


# =============================================================================
def score(runsdir, csvpath, manifest_path):
    print("=" * 78)
    print(" cvk1 -- THE CUT-POSITION LADDER ON VGG11_bn_c100: DOES THE PEAK SIT AT")
    print("         THE LAST CUT BEFORE THE CARRIER (k* = 22)?")
    print(" %s / %s   %d epochs   seeds %s   AUGMENT=%s  BETA_CLIP=%s  PROBE=%d"
          % (NET, DSET, EPOCHS, ",".join(str(s) for s in SEEDS), AUG, CLIP, PROBE))
    for a in ARMS:
        print("   %-4s %s" % (a, SPEC[a]))
    print(" PRIMARY: the ARGMAX LOCATION over the 7 sweep arms of plateau5 (TEST,")
    print("          mean of epochs %d..%d), 3-seed IN-BATCH means from the runs' OWN"
          % (EPOCHS - 5, EPOCHS - 1))
    print("          .out files.  TRAIN is printed beside TEST at every arm.")
    print("=" * 78)

    rows = read_corpus(csvpath)
    runs = read_runs(runsdir)

    print("\nG-STRUCT  the live model and every partition are what the batch claims")
    mp, how = resolve_manifest(runsdir, manifest_path)
    man = read_manifest(mp)
    if man is None:
        chk(False, "G-STRUCT PARTITION-MANIFEST.txt found (%s)" % how)
    else:
        print("  manifest: %s (%s)" % (mp, how))
        tens = man["tensors"]
        numel = dict((t[0], t[2]) for t in tens)
        idxof = dict((t[1], t[0]) for t in tens)
        chk(len(tens) == NTENS, "G-STRUCT %d parameter tensors" % NTENS, str(len(tens)))
        chk(sum(numel.values()) == TOTPAR, "G-STRUCT %d parameters" % TOTPAR,
            str(sum(numel.values())))
        chk(idxof.get(CARRIER[0]) == CARRIER[1] and numel.get(CARRIER[1]) == CARRIER[2],
            "G-STRUCT the nominated carrier %s sits at 1-based %d, numel %d" % CARRIER)
        chk(alias_violations(tens) == [NTENS],
            "G-STRUCT class == index mod 3 at every index 1..%d (only linear.bias breaks it)"
            % (NTENS - 2), str(alias_violations(tens)))
        for k in GRID:
            a = "k%d" % k
            spec = man["armspec"].get(a, "")
            m = re.search(r"SIZES (\S+) PARAMS (\S+)", spec)
            if not m:
                chk(False, "G-STRUCT %s ARMSPEC line present" % a)
                continue
            sizes = [int(x) for x in m.group(1).split(",")]
            pars = [int(x) for x in m.group(2).split(",")]
            chk(sizes == [k, NTENS - k] and pars == [WANT_G1[k], TOTPAR - WANT_G1[k]]
                and SPEC[a] in spec,
                "G-STRUCT %s composes to [%d,%d], params [%d,%d]"
                % (a, k, NTENS - k, WANT_G1[k], TOTPAR - WANT_G1[k]),
                "sizes %s params %s" % (sizes, pars))

    print("\nG-SOUND   27 runs, %d epochs each, RUN_DONE, no traceback" % EPOCHS)
    for s in SEEDS:
        for a in ARMS:
            r = runs.get((a, s))
            if r is None:
                chk(False, "G-SOUND %s-s%d present" % (a, s), "missing")
                continue
            chk(r["n_epochs"] == EPOCHS and not r["traceback"]
                and r["plateau5"] is not None and r["run_done"],
                "G-SOUND %s-s%d" % (a, s),
                "%d epoch lines, RUN_DONE=%s, plateau5 %s (train %s)"
                % (r["n_epochs"], r["run_done"], fmt(r["plateau5"]), fmt(r["train5"])))

    print("\nG-ENV     the environment rode the ENV line on every run")
    envs = sorted(set(re.sub(r" PROBE_DIR=\S*", "", ln)
                      for r in runs.values() for ln in r["env"]))
    chk(len(envs) == 1, "G-ENV one distinct ENV line across all runs", "%d distinct" % len(envs))
    for ln in envs:
        print("        %s" % ln)
        for tok in ("AUGMENT=%s" % AUG, "BETA_CLIP=%s" % CLIP, "PROBE=%d" % PROBE):
            chk((" " + tok + " ") in (" " + ln + " "), "G-ENV ENV line carries %s" % tok)
    chk(not any(r["pt_on"] for r in runs.values()),
        "G-ENV no run printed `PROBE_TENSOR: on` (PROBE_TENSOR unset in this batch)")
    for (a, s), r in sorted(runs.items()):
        want = "--stepsize-groups %s " % SPEC.get(a, "?")
        if not any(want in (ln + " ") for ln in r["args"]):
            chk(False, "G-ENV %s-s%d ARGS line carries %s" % (a, s, want.strip()))

    print("\nLEVELS    plateau5 (TEST) and TRAIN, IN BATCH")
    arm_v, arm_t = {}, {}
    for a in ARMS:
        v = [runs[(a, s)]["plateau5"] for s in SEEDS
             if (a, s) in runs and runs[(a, s)]["plateau5"] is not None]
        t = [runs[(a, s)]["train5"] for s in SEEDS
             if (a, s) in runs and runs[(a, s)]["train5"] is not None]
        arm_v[a], arm_t[a] = v, t
        if v:
            print("  %-4s %-10s TEST %.4f  sd %s  range %.3f  n=%d   TRAIN %s"
                  % (a, SPEC[a], mean(v), fmt(sd(v), 3), max(v) - min(v), len(v),
                     fmt(mean(t))))
        else:
            print("  %-4s %-10s NO USABLE RUN" % (a, SPEC[a]))
    if not all(len(arm_v[a]) == len(SEEDS) for a in ARMS):
        print("\nFINAL: INCOMPLETE -- %s"
              % ", ".join("%s %d/%d" % (a, len(arm_v[a]), len(SEEDS)) for a in ARMS))
        print("No branch is taken and NO NUMBER HERE IS QUOTABLE.")
        return 2

    print("\nFLOOR     SIGMA, monotone-conservative under corpus drift")
    ss = sum(sum((x - mean(arm_v[a])) ** 2 for x in arm_v[a]) for a in ARMS)
    df = sum(len(arm_v[a]) - 1 for a in ARMS)
    sigma_in = math.sqrt(ss / df)
    cands = [("SIGMA_PRIOR_frozen", SIGMA_PRIOR), ("SIGMA_INBATCH", sigma_in)]
    if rows:
        sn = pooled_sigma(rows, lambda r: r["network"] == "ResNet18_c100"
                          and r["dataset"] == "CIFAR100")[0]
        sv = pooled_sigma(rows, lambda r: r["network"] == NET)[0]
        if sn:
            cands.append(("SIGMA_NARROW_live", sn))
        if sv:
            cands.append(("SIGMA_VGG_live", sv))
    for nm, v in cands:
        print("  %-20s %.6f" % (nm, v))
    which, sigma_used = max(cands, key=lambda kv: kv[1])
    se = sigma_used * math.sqrt(2.0 / 3.0)
    print("  SIGMA_USED           %.6f  (= %s)" % (sigma_used, which))
    print("  SE_ARM_DIFF          %.6f  (df_inbatch %d)" % (se, df))

    print("\nG-FLOOR / G-CEIL / G-DIVERGE / G-GAP")
    M = dict((a, mean(arm_v[a])) for a in ARMS)
    T = dict((a, mean(arm_t[a])) for a in ARMS)
    hi = max(M.values())
    harness_ok = chk(hi >= FLOOR_MIN, "G-FLOOR max arm mean >= %.2f pp (chance %.2f)"
                     % (FLOOR_MIN, CHANCE), "max %.4f" % hi)
    chk(hi <= CEIL_MAX, "G-CEIL max arm mean <= %.2f pp" % CEIL_MAX, "max %.4f" % hi)
    div = []
    for a in ARMS:
        rng = max(arm_v[a]) - min(arm_v[a])
        if not chk(rng <= DIVERGED_BAR, "G-DIVERGE %s seed range <= %.2f pp" % (a, DIVERGED_BAR),
                   "range %.4f" % rng):
            div.append(a)
    D_GAP = M["kL"] - M["k01"]
    gap_ok = chk(D_GAP >= GAP_MIN, "G-GAP in-batch kL - k01 >= %.1f pp" % GAP_MIN,
                 "%+.4f pp = %+.2f SE" % (D_GAP, D_GAP / se))

    Mk = dict((k, M["k%d" % k]) for k in GRID)
    Tk = dict((k, T["k%d" % k]) for k in GRID)
    print("\nTHE LADDER (every contrast WITHIN batch)")
    print("   k  spec      TEST      above k01 (SE)     TRAIN    CAPTURE(desc.)")
    for k in GRID:
        cap = (Mk[k] - M["k01"]) / D_GAP if abs(D_GAP) > 1e-9 else float("nan")
        print("  %2d  %-8s %8.4f   %+8.4f (%+6.2f)  %8.4f   %+.4f"
              % (k, SPEC["k%d" % k], Mk[k], Mk[k] - M["k01"], (Mk[k] - M["k01"]) / se,
                 Tk[k], cap))
    print("  anchors: k01 %.4f (TRAIN %.4f)   kL %.4f (TRAIN %.4f)   D_GAP %+.4f"
          % (M["k01"], T["k01"], M["kL"], T["kL"], D_GAP))

    print("\n" + "=" * 78)
    if not harness_ok:
        branch, k_hat, peak_set, info = "HARNESS-UNSOUND", None, [], {}
    elif div:
        branch, k_hat, peak_set, info = "UNRESOLVED-DIVERGED", None, [], {}
    elif not gap_ok:
        branch, k_hat, peak_set, info = "UNRESOLVED-NO-GAP", None, [], {}
    else:
        branch, k_hat, peak_set, info = decide(Mk, M["k01"], se, D_GAP)
    if info:
        print("PRIMARY   bars: PEAK_BAR %.4f pp (2 SE) | FLOOR_BAR %.4f pp (2 SE) | "
              "FLAT_BAR %.4f pp" % (info["peak_bar"], info["floor_bar"], info["flat_bar"]))
        print("          floored sweep arms: %s" % (info.get("floored") or "none"))
        if "range" in info:
            print("          sweep RANGE %.4f pp" % info["range"])
    if k_hat is not None:
        order = sorted(GRID, key=lambda k: -Mk[k])
        print("          argmax k_hat = %d (%.4f); runner-up k%d (%.4f), margin %.4f pp = %.2f SE"
              % (k_hat, Mk[k_hat], order[1], Mk[order[1]], Mk[k_hat] - Mk[order[1]],
                 (Mk[k_hat] - Mk[order[1]]) / se))
        print("          PEAK_SET %s" % peak_set)

    # the A1-vs-A2 discriminator and the cliff, sign-and-bound discipline
    d2216 = Mk[22] - Mk[16]
    cliff = Mk[22] - Mk[23]
    k23_floored = Mk[23] - M["k01"] < FLOOR_SE * se
    print("\nSECONDARIES (descriptive; nothing below branches)")
    print("  D22-16 = M(22) - M(16) = %+.4f pp = %+.2f SE   (A1 predicts +, A2 predicts -)"
          % (d2216, d2216 / se))
    print("  CLIFF  = M(22) - M(23) = %+.4f pp = %+.2f SE%s"
          % (cliff, cliff / se, "   -- k23 is FLOORED: this is a BOUND, not an effect size"
             if k23_floored else ""))
    print("  THE CLASS-MATCHED WINDOW, beside cpk3's ResNet steps @100 (SIGN CENSUS ONLY):")
    for a, b, nm, ra, rb, rnm, rstep in WINDOW:
        d, dt = Mk[b] - Mk[a], Tk[b] - Tk[a]
        print("    %2d->%2d moves %-13s TEST %+8.4f (%+6.2f SE) TRAIN %+8.4f | ResNet %d->%d %-22s %+8.4f"
              % (a, b, nm, d, d / se, dt, ra, rb, rnm, rstep))

    t_hat = max(GRID, key=lambda k: Tk[k])
    stamps = ["GATES-CLEAN" if not GATE_FAIL else "GATES-%d-FAIL" % len(GATE_FAIL),
              "ALIAS-CLASS-IS-INDEX-MOD-3", "MAGNITUDE-NOT-SEPARATED", "HORIZON-100-ONLY",
              "SHARED-MS-1E-3", "NOT-A-ONE-VARIABLE-ABLATION",
              "SIGMA-%s" % which.replace("SIGMA_", "").replace("_", "-").upper(),
              "TRAIN-ARGMAX-AGREES" if (k_hat is None or t_hat == k_hat) else
              "TRAIN-ARGMAX-DISAGREES(%d)" % t_hat,
              "D22-16-%s" % ("POSITIVE" if d2216 > PEAK_SE * se else
                             "NEGATIVE" if d2216 < -PEAK_SE * se else "TIED"),
              "CLIFF-IS-A-BOUND" if k23_floored else "CLIFF-POINT",
              "GAP-REPLICATES-CVG1" if abs(D_GAP - CVG1_D) <= 5.0 else "GAP-DIFFERS-FROM-CVG1"]
    print("\nFINAL: %s | %s | %s | %s" % (branch, carrier_token(branch),
                                          rival_token(branch, peak_set), " | ".join(stamps)))
    print("=" * 78)
    print("BETWEEN-BATCH, NON-GATING: cvg1 measured k01 %.4f / kL %.4f / D %+.4f at this cell"
          % (CVG1_K01, CVG1_KL, CVG1_D))
    print("  on seeds {31,32,33}; cvi1 isolates bn8 at [25,1] on {41,42,43}.  NEITHER enters")
    print("  any cvk1 contrast.  BATCH is the unit of replication (F(62,85)=5.47).")
    if GATE_FAIL:
        print("\nGATE FAILURES (%d): %s" % (len(GATE_FAIL), "; ".join(GATE_FAIL)))
        print("A failed gate SUSPENDS the branch; it is NOT an adverse result.")
    print("\nWHAT IT DOES NOT LICENSE, UNCONDITIONALLY: carrier identity vs junction class")
    print("  (class == index mod 3); identity vs magnitude (bn8 %.4f of mean|L|, bn7 %.4f);"
          % (NOM_SHARE_BN8, NOM_SHARE_BN7))
    print("  anything about the ISOLATION line, residual connections, other horizons,")
    print("  other ms, or group COUNT.")
    return 1 if GATE_FAIL else 0


# =============================================================================
def selftest(csvpath, runsdir):
    print("cVK1 --selftest: the design, the ResNet premise, the VGG arithmetic, the")
    print("prefix-sign map, the corpus noise floor, the floor table and the branch map.")
    print("Every corpus reader EXCLUDES `cvk1-`.\n")

    print("ARITHMETIC AND REGEX")
    chk(abs(SIGMA_PRIOR * math.sqrt(2 / 3) - SE_PRIOR) < 1e-6, "SE_PRIOR = SIGMA_PRIOR*sqrt(2/3)")
    chk(SIGMA_PRIOR == max(SIGMA_VGG, SIGMA_NARROW), "SIGMA_PRIOR = max(SIGMA_VGG, SIGMA_NARROW)")
    chk(FLOOR_MIN > CHANCE and CEIL_MAX < 100.0, "harness gates sit inside (chance, 100)")
    chk(EPTR_RE.search("Epoch 99, Train Accuracy: 62.59 %, Test Accuracy: 55.52 %") is not None,
        "epoch-line regex matches the harness's print")
    chk(OUT_RE.match("cvk1-k22-s55-1234567.out") is not None
        and OUT_RE.match("cvk1-kL-s57-1.out") is not None
        and OUT_RE.match("cvi1-ISO-s41-1.out") is None
        and OUT_RE.match("cpk1-k49-s0-1.out") is None,
        "run-file regex matches cvk1 ONLY (not cvi1, not cpk1)")

    print("\nTHE DESIGN")
    chk(len(ARMS) == 9 and set(SPEC) == set(ARMS), "nine arms: two anchors + seven cuts")
    chk(SPEC["k01"] == "scalar" and SPEC["kL"] == "layerwise", "anchors scalar and layerwise, IN BATCH")
    for k in GRID:
        chk(SPEC["k%d" % k] == "[%d,%d]" % (k, NTENS - k), "k%d spec is [%d,%d]" % (k, k, NTENS - k))
    chk(KSTAR_PRED in GRID and KSTAR_PRED - 1 in GRID and KSTAR_PRED + 1 in GRID,
        "the predicted peak 22 is INTERIOR: 21 and 23 both measured")
    chk(KSTAR_PRED == CARRIER[1] - 1, "k* = carrier index - 1 = %d" % (CARRIER[1] - 1))
    chk(16 in GRID and 21 in GRID and 20 in GRID and 19 in GRID,
        "every rival's peak is on the grid: A2 16, A4 20/21, previous junction 19")
    chk(GRID[0] < 16 and GRID[-1] > KSTAR_PRED, "16 and 22 are both interior (13 and 23 bound them)")
    chk(list(range(19, 24)) == [k for k in GRID if k >= 19],
        "19..23 is CONSECUTIVE -- the class sequence of cpk3's 46..50")
    chk(len(set(SEEDS)) == 3 and all(55 <= s <= 59 for s in SEEDS),
        "three seeds from the pre-assigned block 55-59: %s" % (SEEDS,))
    chk(PROBE == 0, "PROBE=0, PROBE_TENSOR unset -- the ENV of cpk1/cpk2/cpk3, the ladder replicated")

    print("\nTHE PRIMARY IS BOUNDED IN NEITHER DIRECTION -- every grid position is reachable")
    # A spike over a FLAT curve is monotone when it sits at an edge (flat-then-up
    # never falls), so the edge verdicts are the NO-PEAK tokens; the argmax
    # LOCATION itself is asserted separately for every k.
    for k in GRID:
        M = dict((j, 50.0) for j in GRID)
        M[k] = 60.0
        br, k_hat = decide(M, 35.0, SE_PRIOR, 31.0)[:2]
        want = ("PEAK-AT-22" if k == 22 else "NO-PEAK-INCREASING" if k == 23 else
                "NO-PEAK-DECREASING" if k == 13 else "PEAK-ELSEWHERE-%d" % k)
        chk(br == want and k_hat == k, "a spike at k=%d -> argmax %d, %s" % (k, k, want),
            "%s k_hat=%s" % (br, k_hat))

    print("\nTHE BRANCH MAP IS TOTAL (the SAME decide() the scoring path calls)")
    k01 = CVG1_K01
    lv = lambda cap: dict((k, k01 + c * CVG1_D) for k, c in cap.items())
    cases = [
        ("all floored", dict((k, k01 + 0.3) for k in GRID), "ALL-CUTS-FLOORED"),
        ("flat and high", dict((k, 50.0 + 0.2 * (k % 2)) for k in GRID), "NO-PEAK-FLAT"),
        ("monotone rise", dict((k, 40.0 + 1.5 * i) for i, k in enumerate(GRID)), "NO-PEAK-INCREASING"),
        ("monotone fall", dict((k, 58.0 - 1.5 * i) for i, k in enumerate(GRID)), "NO-PEAK-DECREASING"),
        ("22 and 21 tied", dict(list(lv(PRED["A1"]).items()) + [(21, lv(PRED["A1"])[22] - 0.3)]),
         "UNRESOLVED-TIE-WITH-22"),
        ("16 and 19 tied", dict([(13, 45.0), (16, 55.0), (19, 54.6), (20, 50.0), (21, 50.0),
                                 (22, 40.0), (23, 36.0)]), "PEAK-ELSEWHERE-TIE"),
        ("rise with a sawtooth to 23", dict([(13, 40.0), (16, 45.0), (19, 50.0), (20, 47.0),
                                             (21, 50.0), (22, 55.0), (23, 58.0)]),
         "PEAK-AT-RIGHT-EDGE-23"),
    ]
    for nm, M, want in cases:
        br = decide(M, k01, SE_PRIOR, CVG1_D)[0]
        chk(br == want, "branch(%s) -> %s" % (nm, want), br)
    chk(carrier_token("PEAK-AT-22") == "CARRIER-CUT-SUPPORTED"
        and carrier_token("PEAK-ELSEWHERE-16") == "CARRIER-CUT-REFUTED"
        and carrier_token("NO-PEAK-FLAT") == "CARRIER-CUT-REFUTED"
        and carrier_token("UNRESOLVED-TIE-WITH-22") == "CARRIER-CUT-UNRESOLVED",
        "the carrier token maps every branch")

    print("\nTHE FLOOR TABLE (164.6) -- F1..F4, each account's predicted levels at cvg1's anchors")
    worst_peak = None
    for acct, cap in PRED.items():
        L = lv(cap)
        peak = max(GRID, key=lambda k: L[k])
        near = set(k for k in GRID if L[k] - k01 < NEAR_FLOOR_PP)
        print("  %-8s %s   peak k%d" % (acct, " ".join("k%d %.2f" % (k, L[k]) for k in GRID), peak))
        chk(near == AT_FLOOR_DECLARED[acct],
            "%s: the arms predicted AT/NEAR the floor are exactly the declared %s"
            % (acct, sorted(AT_FLOOR_DECLARED[acct]) or "{}"), str(sorted(near)))
        chk(peak not in near, "F2 %s: its peak arm k%d is NOT among its floored arms" % (acct, peak))
        chk(L[peak] - k01 >= PEAK_MARGIN_MIN,
            "F1 %s: its peak arm clears the floor by >= %.0f pp" % (acct, PEAK_MARGIN_MIN),
            "%+.2f pp = %.1f SE" % (L[peak] - k01, (L[peak] - k01) / SE_PRIOR))
        chk(max(L.values()) <= CEIL_MAX, "%s: highest prediction <= %.0f" % (acct, CEIL_MAX))
        br = decide(L, k01, SE_PRIOR, CVG1_D)
        chk(br[0] == PRED_BRANCH[acct], "%s's own predicted curve scores as %s"
            % (acct, PRED_BRANCH[acct]), "%s %s" % (br[0], br[2]))
        if acct in ("A1", "A2", "A4"):
            m = L[peak] - k01
            worst_peak = m if worst_peak is None else min(worst_peak, m)
    nonfloor = [lv(c)[k] - k01 for a, c in PRED.items() for k in GRID
                if k not in AT_FLOOR_DECLARED[a]]
    note("lowest NON-floored predicted arm over all accounts: %+.2f pp above k01" % min(nonfloor))
    note("lowest PEAK arm over A1/A2/A4: %+.2f pp above k01" % worst_peak)
    chk(decide(lv(PRED["A4"]), k01, SE_PRIOR, CVG1_D)[2] == [20, 21],
        "A4's own curve ties 20 and 21 -- 20.55 falls between them, as the account says")
    note("F3/F4: k23 is predicted floored under A1, A2 and A4 and is the TOP arm under A5-INC;")
    note("its floor binds it only DOWNWARD, and every claim it can refute is refuted by it")
    note("landing HIGH.  The brief's literal rule (no arm floored under any account) cannot")
    note("be met by ANY design that separates PEAK-AT-22 from NO-PEAK: see the header.")

    print("\nTHE VGG ARITHMETIC (live-model manifest, cvg1's copy or this batch's own)")
    vman = None
    for cand in (os.path.join(runsdir, PREFIX, "PARTITION-MANIFEST.txt"),
                 os.path.join(runsdir, "cvg1", "PARTITION-MANIFEST.txt")):
        vman = read_manifest(cand)
        if vman and len(vman["tensors"]) == NTENS:
            print("       manifest: %s" % cand)
            break
        vman = None
    if vman is None:
        skip("VGG arithmetic -- no 26-tensor manifest under %s" % runsdir)
    else:
        tens = vman["tensors"]
        numel = [t[2] for t in tens]
        cum = [sum(numel[:k]) for k in range(NTENS + 1)]
        chk(cum[NTENS] == TOTPAR, "total %d" % TOTPAR, str(cum[NTENS]))
        chk(all(cum[k] == WANT_G1[k] for k in WANT_G1),
            "group-1 parameter counts re-derive for k in %s" % sorted(WANT_G1),
            str(dict((k, cum[k]) for k in WANT_G1)))
        chk(tens[CARRIER[1] - 1][1] == CARRIER[0] and tens[CARRIER[1] - 1][2] == CARRIER[2],
            "tensor %d is %s (numel %d)" % (CARRIER[1], CARRIER[0], CARRIER[2]))
        chk([i for i, n, q in tens if q == 512 and n.endswith(".weight")] == list(BN512_IDX),
            "the four 512-wide BN scales are at %s" % (BN512_IDX,))
        shares = dict((k, cum[k] / float(TOTPAR)) for k in range(1, NTENS))
        amin = min(range(1, NTENS), key=lambda k: abs(shares[k] - 0.5))
        chk(amin == A2_ARGMIN, "A2: argmin |share(1..k) - 0.5| over k=1..25 is %d" % A2_ARGMIN,
            "k=%d share %.6f" % (amin, shares[amin]))
        chk(abs(abs(shares[16] - 0.5) - abs(shares[18] - 0.5)) < 2e-4,
            "A2: 16, 17, 18 are within 2e-4 in |share - 0.5| -- 16 represents the plateau")
        chk(abs(shares[19] - 0.5) > 0.2 and abs(shares[22] - 0.5) > 0.49,
            "A2: 19..21 sit 0.24 and 22 sits 0.49 from balance -- A2 puts 22 near the floor")
        chk(abs(A4_POINT - 20.548387) < 1e-5, "A4: 26 * 49/62 = 20.5484, between 20 and 21")
        chk(alias_violations(tens) == [NTENS],
            "ALIAS: class == index mod 3 at every index 1..24; only 26 (linear.bias) breaks it",
            str(alias_violations(tens)))
        chk(all(k % 3 == 1 for k in (13, 16, 19, 22)) and 20 % 3 == 2 and 21 % 3 == 0,
            "ALIAS: 13/16/19/22 are conv|BN-scale junctions; 20 closes a scale, 21 a shift")

    print("\nRESNET PREMISE -- re-derived from cpk1/cpk2/cpk3 RAW .out and cdep1's manifest")
    res = rederive_resnet(runsdir)
    if res is None:
        skip("ResNet premise -- cpk*.out or cdep1/PARTITION-MANIFEST.txt not under %s" % runsdir,
             "(the launcher REQUIRES this section to have RUN on the cluster host)")
    else:
        rt = res["man"]["tensors"]
        print("       %d cpk runs with RUN_DONE; manifest %d tensors" % (res["nruns"], len(rt)))
        chk(len(rt) == 62, "ResNet18_c100 manifest has 62 tensors")
        for i, nm in sorted(RES_CARRIERS.items()):
            chk(rt[i - 1][1] == nm and rt[i - 1][2] == 512,
                "carrier %d is %s (numel 512)" % (i, nm), rt[i - 1][1])
        for key, lvl in sorted(RES_CURVES.items()):
            lv_ = res["levels"].get(key)
            if not lv_:
                chk(False, "%s @%s curve present" % key)
                continue
            sweep = dict((k, v) for k, v in lv_.items() if k not in (1, 62))
            am = max(sweep, key=lambda k: sweep[k])
            chk(am == RES_ARGMAX and abs(sweep[am] - lvl) < 5e-4,
                "%s @%s: argmax over m=2 arms is k=%d at %.4f" % (key[0], key[1], RES_ARGMAX, lvl),
                "k=%d %.4f" % (am, sweep[am]))
        chk(min(RES_CARRIERS) - 1 == RES_ARGMAX,
            "k* = 49 IS the last cut before the first carrier (tensor %d)" % min(RES_CARRIERS))
        for key, st in sorted(RES_STEP_49_50.items()):
            lv_ = res["levels"].get(key, {})
            if 49 in lv_ and 50 in lv_:
                chk(abs((lv_[50] - lv_[49]) - st) < 5e-4,
                    "%s @%s: the ONE-tensor step 49->50 (carrier 50) = %+.4f" % (key[0], key[1], st),
                    "%+.4f" % (lv_[50] - lv_[49]))
        c1 = res["levels"].get(("cpk1", "100"), {})
        for (a, b), st in sorted(RES_CPK1_STEPS.items()):
            if a in c1 and b in c1:
                chk(abs((c1[b] - c1[a]) - st) < 5e-4,
                    "cpk1 @100 step %d->%d = %+.4f" % (a, b, st), "%+.4f" % (c1[b] - c1[a]))
        note("BRIEFING CORRECTION: -18.2707 is cpk3's ONE-tensor 49->50 at 772 epochs; the")
        note("THREE-tensor 49->52 is cpk1's -17.1053 at 100.  The drop is AT the first carrier.")
        c3 = res["levels"].get(("cpk3", "100"), {})
        for a, b, nm, ra, rb, rnm, rst in WINDOW:
            if ra in c3 and rb in c3:
                chk(abs((c3[rb] - c3[ra]) - rst) < 5e-4,
                    "cpk3 @100 window analogue %d->%d (%s) = %+.4f" % (ra, rb, rnm, rst),
                    "%+.4f" % (c3[rb] - c3[ra]))
        rnum = [t[2] for t in rt]
        rcum = [sum(rnum[:k]) for k in range(63)]
        rsh = dict((k, rcum[k] / float(rcum[62])) for k in range(1, 62))
        ramin = min(range(1, 62), key=lambda k: abs(rsh[k] - 0.5))
        chk(ramin == RES_SHARE_ARGMIN,
            "THE CONFOUND: on ResNet, argmin |share(1..k) - 0.5| over ALL k is ALSO 49 -- the "
            "balance rival and the carrier cut name the SAME k", "k=%d share %.6f" % (ramin, rsh[ramin]))
        chk(all(abs(rsh[k] - RES_SHARE1[k]) < 5e-6 for k in RES_SHARE1),
            "ResNet group-1 shares re-derive for cpk1's eleven cuts")
        chk(alias_violations(rt) == [62], "ALIAS on ResNet: class == index mod 3 on 1..61; 62 breaks it",
            str(alias_violations(rt)))
        if c1 and all(k in c1 for k in CPK1_LEVELS):
            chk(all(abs(c1[k] - CPK1_LEVELS[k]) < 5e-4 for k in CPK1_LEVELS),
                "cpk1's thirteen in-batch levels re-derive (they feed the floor-table models)")
            mp = model_predictions(dict((k, c1[k]) for k in CPK1_LEVELS), RES_SHARE1)
            for acct in ("A1", "A2", "A4"):
                chk(all(abs(mp[acct][k] - PRED[acct][k]) < PRED_TOL for k in GRID),
                    "PRED[%s] re-derives from cpk1's raw curve" % acct,
                    " ".join("%.4f" % mp[acct][k] for k in GRID))

    print("\nPREFIX-SIGN MAP -- cvg1's scalar pinned records (A1a's cliff location)")
    ps = rederive_prefix_sign(runsdir)
    if ps is None:
        skip("prefix-sign map -- cvg1 probe records not under %s" % runsdir,
             "(the launcher REQUIRES this section to have RUN on the cluster host)")
    else:
        neg, npin, nseeds = ps
        print("       %d pinned scalar records over %d seeds" % (npin, nseeds))
        print("       frac(sum_{i<=k} L_i < 0): %s" % " ".join("k%d %.4f" % (k, neg[k]) for k in GRID))
        chk(all(abs(neg[k] - PREFIX_NEG[k]) < PREFIX_TOL for k in GRID),
            "PREFIX-SIGN: the group-1 sum is negative on ALL records for every k<=22 and on NONE at "
            "k=23 -- the sign boundary sits at 22|23, and the map is FLAT below it")

    print("\nCORPUS PREMISES, re-derived (never quoted):")
    rows = read_corpus(csvpath)
    if rows is None:
        skip("corpus premises -- %s not readable" % csvpath)
    else:
        note("corpus rows with `cvk1-` excluded", str(len(rows)))
        chk(not any(r["run"].startswith(PREFIX + "-") for r in rows), "ZERO cvk1 rows survive the exclusion")
        e250 = [r for r in rows if r.get("epochs_requested") != str(EPOCHS)]
        chk(all(not std_cell(r) for r in e250),
            "CONSTRUCTIVE: all %d non-100-epoch rows are EXCLUDED by std_cell" % len(e250))
        sV, dV, cV, mV = pooled_sigma(rows, lambda r: r["network"] == NET)
        sN, dN, cN, mN = pooled_sigma(rows, lambda r: r["network"] == "ResNet18_c100"
                                      and r["dataset"] == "CIFAR100")
        print("       SIGMA_VGG    %s df %s cells %s members %s" % (fmt(sV, 6), dV, cV, mV))
        print("       SIGMA_NARROW %s df %s cells %s members %s" % (fmt(sN, 6), dN, cN, mN))
        for nm, live, frozen in (("SIGMA_VGG", sV, SIGMA_VGG), ("SIGMA_NARROW", sN, SIGMA_NARROW)):
            if live is not None and abs(live - frozen) < 1e-6:
                chk(True, "%s re-derived == frozen %.6f" % (nm, frozen))
            elif live is not None and live > frozen:
                note("%s HAS DRIFTED UP (frozen %.6f, live %.6f) -- NOT a failure: SIGMA_USED "
                     "takes the max, so the bars can only WIDEN" % (nm, frozen, live))
            else:
                chk(False, "%s re-derived (frozen %.6f)" % (nm, frozen), fmt(live, 6))
        g = corpus_gap(rows, NET, DSET)
        if g:
            if abs(g[0] - CVG1_K01) < 5e-3 and abs(g[1] - CVG1_KL) < 5e-3:
                chk(True, "THIS CELL's levels re-derive == frozen (k01 %.4f, kL %.4f, D %+.4f)"
                    % (CVG1_K01, CVG1_KL, CVG1_D))
            else:
                note("THIS CELL's corpus levels have MOVED (cvi1's anchors landed?): %.4f / %.4f / %+.4f "
                     "-- they enter NO bar and NO branch; the frozen cvg1 numbers are the "
                     "floor-table anchors only" % g)
            chk(GAP_MIN <= CVG1_D / 3.0, "GAP_MIN (%.1f) <= a third of this cell's cvg1 gap" % GAP_MIN)
            chk(FLAT_FRAC * CVG1_D >= FLAT_MIN_SE * SE_PRIOR,
                "FLAT_BAR at cvg1's gap is the 0.10-of-gap rule (%.3f pp = %.2f SE), not the SE floor"
                % (FLAT_FRAC * CVG1_D, FLAT_FRAC * CVG1_D / SE_PRIOR))
        used = sorted(set(int(r["seed"]) for r in rows if str(r.get("seed", "")).strip().isdigit()))
        print("       seeds present anywhere: %s" % used)
        chk(not (set(SEEDS) & set(used)),
            "ZERO rows ANYWHERE carry seed %s" % ", ".join(str(s) for s in SEEDS))

    print("\nRULE 21 PREMISE: no cvk1 run may exist yet")
    got = read_runs(runsdir) if runsdir else {}
    chk(not got, "no cvk1-*.out under %s" % runsdir, "%d found" % len(got))

    print("\n%s" % ("ALL PASS" if not GATE_FAIL else "FAILURES: %d" % len(GATE_FAIL)))
    for f in GATE_FAIL:
        print("   FAIL", f)
    return 1 if GATE_FAIL else 0


def main():
    ap = argparse.ArgumentParser(
        description="cvk1 registered scorer.  Documented invocation: "
                    "python3 analysis/cVK1_vggcut_score.py <runsdir>")
    ap.add_argument("runsdir", nargs="?", default="")
    ap.add_argument("--runsdir", dest="runsdir_kw", default="")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--manifest", default="",
                    help="defaults to <runsdir>/cvk1/PARTITION-MANIFEST.txt")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    runsdir = a.runsdir or a.runsdir_kw
    if a.selftest:
        return selftest(a.csv, runsdir or ".")
    if not runsdir:
        ap.error("a runsdir is required: python3 %s <runsdir>" % os.path.basename(__file__))
    return score(runsdir, a.csv, a.manifest)


if __name__ == "__main__":
    sys.exit(main())
