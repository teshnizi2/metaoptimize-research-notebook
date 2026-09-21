#!/usr/bin/env python3
# =============================================================================
# cwd5_attack_indep.py -- THE INDEPENDENT PARSER OF `cwd5` (CORRECTIONS 281, the coupled weight-decay ladder).
#
# It re-derives, FROM THE RAW RECORDS ALONE, every line the REGISTERED scorer
# analysis/cWD5_wdladder_score.py prints, and compares its rebuild with a committed scorer log, line by line, in order.
#
#   python3 analysis/cwd5_attack_indep.py <runsdir> <scorer-log> [--csv P] [--tsv P]
#
# INDEPENDENCE, deliberately (the cwd2 / cwd3 / cwd4 pattern, CORRECTIONS 261 / 275 / 280, carried to cwd5):
#   * it imports NOTHING from cWD5_wdladder_score.py, cwd5_design.py, cwd_design.py, cwd_common.py,
#     corpus_exclusions.py, argsline_guard.py or any other repo module.  Stdlib only: hashlib, json, math, os,
#     struct, sys.  `csv` is NOT used either -- the corpus is split by hand.
#   * NO regular expressions anywhere.  The `.out` lines are split on commas / colons / whitespace, the TSV on tabs,
#     the file names on hyphens.
#   * every bar, sigma literal, witness string, tree sha, rung token, licence sentence and stamp name below is
#     RE-TYPED from CORRECTIONS 281, not read from any module.
#   * the ResNet18_c100 tensor table is rebuilt FROM THE ARCHITECTURE (stem, four stages of two BasicBlocks, the three
#     downsample shortcuts, the linear head), and CARW2's masked set is resolved against that rebuild BY NAME.
#   * float32 rounding is re-derived through `struct`, and the Lion natural step is recomputed from the records' own
#     beta_pre / mom_pre / z_agg -- on EACH ARM'S OWN GROUP COUNT (scalar 1, layerwise 62) -- so G-BITE's every printed
#     counter is independent of the scorer's implementation.
#
# HOST INDEPENDENCE.  Three kinds of line in the scorer's output are host-dependent and are matched, not rebuilt byte
# for byte: (1) the two path disclosures (`manifest:` / `provenance:`), matched by their host-independent suffix;
# (2) the corpus disclosure, whose row count depends on which corpus commit the host's tree carries -- this parser
# re-filters the corpus IT IS GIVEN and requires the log's count to equal its own, WITHOUT printing either number, so
# that THIS parser's own stdout is byte-identical on both hosts; (3) the G-PROV `SCORER_SHA256` row, whose printed
# prefix is the scorer file's own sha -- RE-TYPED here from CORRECTIONS 281.5, so it is rebuilt, not waived.
#
# WHAT IT ADDS BEYOND A REPLAY (the six independent attacks):
#   [1]  the design rebuilt from the architecture: 62 tensors / 11,220,132 params, the 20 normscale indices, and
#        CARW2's name list resolved to the registered idx 50/53/59, numel 1,536.  It also re-derives that the batch's
#        EIGHT LADDER ARMS ARE ALL UNMASKED -- the structural fact that makes a patch defect unable to forge a
#        threshold (281's registered G-BITE split).
#   [1b] the RUNG SEPARATION: each run's own ARGS line is read for `--weight-decay-base`, the four tokens are required
#        to be PAIRWISE DISTINCT and to partition the 27 runs 6/9/6/6, and every run is required to carry ITS OWN
#        rung's token.  This is the ONLY in-batch readout that tells the rungs apart -- the ENV line carries no weight
#        decay, a limit re-derived here and NOT hidden.
#   [1c] the WITNESS re-derivation: CARW2's `DECAY_MASK: on ... wd=0.01 ...` line rebuilt from the architecture table
#        AND from ITS OWN RUNG'S weight decay, and required to equal, byte for byte, what the run itself printed --
#        the `wd=` field is what separates a carrier mask at 1e-2 from the same mask at the anchor.
#   [2]  every level, sigma, gap, ratio, rung state and stamp recomputed from the raw `.out` epoch lines.
#   [3]  line-by-line agreement with the committed scorer log.
#   [4]  the FINAL line rebuilt token by token.
#   [5]  THE CARRIER-READABILITY AUDIT: over all 4^4 = 256 rung-state vectors, which accounts are reachable, and with
#        which carrier words -- so that a CAR-* licence paragraph is never quoted at a rung that did not collapse.
#        The registration pre-registered CAR-UNREADABLE for exactly that case (281.3 / 281.9).
#   [6]  THE FLOOR/BOUND AUDIT: which of this batch's printed readings are BOUNDS rather than point effects (164.6).
# =============================================================================

import hashlib
import json
import math
import os
import struct
import sys

# ---------------------------------------------------------------------------------------------------------------------
# RE-TYPED LITERALS (CORRECTIONS 281; the cell from 260 / 271 / 278, the ENV line from cvt1...cvt9 / cwd1 / cwd3 / cwd4)
# ---------------------------------------------------------------------------------------------------------------------
PREFIX = "cwd5"
NET = "ResNet18_c100"
DSET = "CIFAR100"
EPOCHS = 100
BATCH = 100
CLIP = "-15:-2.3026"
MST = "1e-3"
A0 = "1e-6"
AUG = "1"
PROBE = 100
STEPS_PER_EPOCH = 50000 // BATCH                         # 500
N_RECORDS = EPOCHS * STEPS_PER_EPOCH // PROBE            # 500
NTENS = 62
TOTPAR = 11220132
SEEDS = (146, 147, 148)

# THE LADDER, RE-TYPED (281.3).  The CLI token is what the ARGS line echoes; the float is what the harness sees.
RUNG_IDS = ("W1", "W2", "W3", "W4")
WD_TOKEN = {"W1": "0.1", "W2": "1e-2", "W3": "1e-3", "W4": "5e-4"}
WD_VALUE = {"W1": 0.1, "W2": 0.01, "W3": 0.001, "W4": 0.0005}
ANCHOR_RUNG = "W1"
CARRIER_RUNG = "W2"
SCALAR_OF = {"W1": "k01W1", "W2": "k01W2", "W3": "k01W3", "W4": "k01W4"}
LAYER_OF = {"W1": "kLW1", "W2": "kLW2", "W3": "kLW3", "W4": "kLW4"}
CAR = "CARW2"
ARMS = ("k01W1", "kLW1", "k01W2", "kLW2", "k01W3", "kLW3", "k01W4", "kLW4", "CARW2")
MASKED = ("CARW2",)
LADDER_ARMS = tuple(a for a in ARMS if a != CAR)
RUNG_OF = {"k01W1": "W1", "kLW1": "W1", "k01W2": "W2", "kLW2": "W2",
           "k01W3": "W3", "kLW3": "W3", "k01W4": "W4", "kLW4": "W4", "CARW2": "W2"}
SPEC = {"k01W1": "scalar", "kLW1": "layerwise", "k01W2": "scalar", "kLW2": "layerwise",
        "k01W3": "scalar", "kLW3": "layerwise", "k01W4": "scalar", "kLW4": "layerwise", "CARW2": "scalar"}
PT_TYPE = dict(SPEC)
WD = dict((a, WD_TOKEN[RUNG_OF[a]]) for a in ARMS)
WDF = dict((a, WD_VALUE[RUNG_OF[a]]) for a in ARMS)
NG = dict((a, 1 if SPEC[a] == "scalar" else NTENS) for a in ARMS)

# the three ctd1 carriers (275.2 / 188.1), RE-TYPED by name
C50 = "layer4.0.bn2.weight"            # idx 50  Kim gamma_last(layer4.0)
C53 = "layer4.0.shortcut.1.weight"     # idx 53  Kim gamma_down(layer4.0)
C59 = "layer4.1.bn2.weight"            # idx 59  Kim gamma_last(layer4.1)
CARRIERS = (C50, C53, C59)
DMASK = dict((a, "") for a in ARMS)
DMASK[CAR] = "+".join(CARRIERS)
K_MASKED = dict((a, 3 if a == CAR else 0) for a in ARMS)
REG_SET_IDX = {CAR: (50, 53, 59)}
REG_SET_NUMEL = {CAR: 1536}
REG_NORMSCALE_IDX = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59)
NORMSCALE_NUMEL = 4800
W512_IDX = (47, 50, 53, 56, 59)

ENV_EXPECTED = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na "
                "COS_TOTAL=default COS_WARMUP=default SCHED=none SCHED_TOTAL=none "
                "SCHED_WARMUP=none SCHED_MIN=none PROBE=100 EB_RHO=na EB_LOG=0")
OFF_WITNESS = {"VOTE_W": "VOTE_W: off", "BETA_HOLD": "BETA_HOLD: off", "GROUP_HOLD": "GROUP_HOLD: off",
               "REST_HOLD": "REST_HOLD: off"}
KINDS = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD", "COMP_HOLD", "WINDOW_HOLD", "DECAY_MASK")
DM_KEYS = ("dm_n", "dm_masked", "dm_skipped", "dm_wdterm", "dm_norm", "dm_absmin", "dm_small")
HOLD_PREFIXES = ("bh_", "gh_", "rh_", "ch_", "wh_")

# frozen bars (281.4), RE-TYPED
SIGMA_R18ALL = 0.6451413439065666
SIGMA_R18ALL_NAIVE = 12.482176983761198
SIGMA_PRIOR = 0.6451413439065666
SE_PRIOR = 0.526756
R50 = 0.50
GAP_BAR = 10.0
REF_MIN = 55.0
K01_MAX = 30.0
MATCH_BAR = 5.0
NULL_BAR = 2.0
DIVERGED_BAR = 5.0
FLOOR_MIN = 15.0
CEIL_MAX = 90.0
CHANCE = 1.0
BH_TOL = 1e-5
K01_CWD3 = 22.9853                     # cwd3's k01 at wd 0.1 (278.3) -- ONE non-gating replication stamp only

# BETWEEN-BATCH, NON-GATING disclosure block (281's LANDED_DISCLOSURE), RE-TYPED IN ITS PRINTED ORDER
R_BETWEEN_ORDER = ("cwd3 k01 (wd 0.1, scalar)", "cwd3 CARWD0 (wd 0.1, scalar, carriers masked)",
                   "cmo1 W0k01 (wd 0, scalar)", "cmo1 k01 (wd 0.1, scalar)",
                   "cell layerwise mean (n=32, census)")
R_BETWEEN = {"cwd3 k01 (wd 0.1, scalar)": 22.9853,
             "cwd3 CARWD0 (wd 0.1, scalar, carriers masked)": 70.2640,
             "cmo1 W0k01 (wd 0, scalar)": 71.5600,
             "cmo1 k01 (wd 0.1, scalar)": 22.7887,
             "cell layerwise mean (n=32, census)": 69.42}

# Lion, RE-TYPED (the meta optimiser of the cell: ms 1e-3, beta2 0.9, clip -15 .. -2.3026)
MS = 1e-3
B2 = 0.9
LO, HI = -15.0, -2.3026
TIE_REL = 1e-12

# provenance, RE-TYPED (281.5; the tree / runner / build_network shas from 271.2 / cwd1's registration)
BN_SHA = "c79988833c4399a06cc84d15c0830f54d86f59feeec345f47a2287c1cf2ba77d"
HF_POST_SHA = "94aedc33046afb12782ffb7f2eb729bafc7fcba5f42227fec373fa67b6f58ebf"
HF_PARENT_SHA = "5197dc2eecbbcb6feb815f3798b3455171e6ad614deac913be5df48dd9ede1a9"
RUNNER_SHA = "34a8c90ee3a0bee00aa2d0b4a3ad5d16d461fbf5a1f0cda5797e44e44c158371"
SCORER_SHA = "51c28608b4786e2f90cd3ceb54b6ed709dca127eedeb8dbd83ecf3eaf068866c"
DESIGN_SHA = "dd3ab158eab1080406cd608982cde0ff7d20ccd52b48d41131e690ff84dbd74d"
CWD_DESIGN_SHA = "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195"
COMMON_SHA = "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"
PROV_WANT = {"MODE": "submit", "BUILD_NETWORK_SHA256": BN_SHA, "HF_SHA256": HF_POST_SHA,
             "HF_PRE_DECAYMASK_SHA256": HF_PARENT_SHA, "RUNNER_SHA256": RUNNER_SHA, "SCORER_SHA256": SCORER_SHA,
             "DESIGN_SHA256": DESIGN_SHA, "CWD_DESIGN_SHA256": CWD_DESIGN_SHA, "COMMON_SHA256": COMMON_SHA}

# ---------------------------------------------------------------------------------------------------------------------
# THE REGISTERED LICENCE PROSE, RE-TYPED FROM analysis/cWD5_wdladder_score.py's LICENSE / CARRIER_LICENSE dicts
# (CORRECTIONS 281.4 / 281.9; registration commits 1d2a8b7 / fdb16ac).  Re-typed, not imported: the bars inside the
# sentences are this file's own literals.
# ---------------------------------------------------------------------------------------------------------------------
READING = ("the audit's own SCALAR arm (scalar ~22.8 vs layerwise ~69.1 at the CIFAR-100 cells) and the mechanism "
           "line's configuration scope (LIMITS-PREP L1, CORRECTIONS 264 / 271 / 273 / 278)")
FLOOR_NOTE = ("A COLLAPSE state whose scalar arm sits at the floor is a LOCATION, a BOUND on the size of the gap, "
              "never a point effect (164.6); a NOGAP state is a bound in the other direction, set by GAP_BAR.")
NOT_DECOUPLED = ("Not licensed, unconditionally: anything about DECOUPLED weight decay.  The base optimiser has no "
                 "decoupled path (SGDm_base_update multiplies the decay by the LEARNED step size in the weight update "
                 "AND bakes (1 - wd*a) into the meta trace), and no lambda can match the coupled dose, which is wd*a "
                 "with `a` learned and clamped to exp([-15,-2.3026]).  DESCOPED WITH REASONS at 281.2.")
NOT_FINER = ("Not licensed: any weight-decay threshold finer than the bracketing PAIR of rungs.  The ladder has FOUR "
             "points; nothing between two adjacent rungs was run.")
NOT_ELSE = ("Not licensed: any other network, dataset, meta step size, alpha0, momentum, horizon or grouping; the "
            "carrier account at any rung other than W2; which ROUTE the decay acts through.")

LICENSE = {
    "INCOMPLETE": ["No branch.  No number is quotable."],
    "HARNESS-UNSOUND": ["A harness gate failed (or every arm is under %.0f / over %.0f pp).  A statement about the"
                        % (FLOOR_MIN, CEIL_MAX), "harness only; " + READING + " untouched."],
    "PATCH-NOT-VERIFIED": [
        "The LADDER's G-BITE gate failed: on at least one of the EIGHT UNMASKED arms the meta update does not recompute",
        "as Lion on that arm's own group count, a hold key appears, a dm_* key sits on an unmasked arm, or records are",
        "missing / short.  NO level is read.  (A failure on CARW2 alone does NOT reach here: it suppresses the carrier",
        "word only -- see CARRIER-PATCH-NOT-VERIFIED.)  " + READING + " untouched."],
    "UNRESOLVED-DIVERGED": ["A bimodal arm must be reported, not averaged.  No branch; " + READING + " untouched."],
    "UNRESOLVED-REFERENCE": [
        "At least one rung's LAYERWISE arm is below %.0f pp, so at that rung there is no healthy reference against" % REF_MIN,
        "which a collapse can be defined, and the ladder cannot be read end to end.  This is a GATE, not a finding: it",
        "says the experiment did not deliver a usable reference at that weight decay, which is itself worth reporting",
        "as a fact about the configuration.  No threshold sentence, no corner-case verdict either way."],
    "ANCHOR-NOT-COLLAPSED": [
        "At the ANCHOR rung (wd 0.1) the batch did not reproduce the mechanism cell -- either k01W1 is above %.0f pp" % K01_MAX,
        "or the scalar arm is not at or below %.2f x its own layerwise arm.  The ladder has no baseline, so NOTHING" % R50,
        "about weight-decay dependence is licensed.  This questions the in-batch reproduction, not the landed corpus",
        "cells (between-batch); report it as a failed replication and do not read any other rung."],
    "COLLAPSE-AT-ALL-VALUES": [
        "The scalar arm sits at or below %.2f x its own layerwise arm at ALL FOUR weight decays, including %s -- the" % (R50, WD_TOKEN["W4"]),
        "standard CIFAR SGD value.  SENTENCE LICENSED, at this cell: 'the shared-step-size collapse is not an artefact",
        "of the campaign's large coupled weight decay: at a standard CIFAR weight decay the scalar grouping still falls",
        "to at most half the layerwise level.'  EFFECT ON " + READING + ": the CORNER-CASE CONFIGURATION CHARGE IS",
        "REFUTED at this cell, and the audit's scalar arm may be reported as a granularity result rather than a",
        "configuration artefact -- with G_W4 quoted, since the SIZE of the gap may still vary along the ladder.",
        FLOOR_NOTE, NOT_FINER, NOT_DECOUPLED, NOT_ELSE],
    "THRESHOLD-W1-W2": [
        "The collapse exists at the anchor (wd %s) and at NO lower rung: it dies between %s and %s." % (WD_TOKEN["W1"], WD_TOKEN["W1"], WD_TOKEN["W2"]),
        "SENTENCE LICENSED: 'the shared-step-size collapse at this cell requires a coupled weight decay within a decade",
        "of 0.1; at 1e-2 and below the two grains do not differ by the collapse bar.'  EFFECT ON " + READING + ": THIS IS",
        "THE REFEREE'S CORNER-CASE CHARGE LANDING.  The mechanism line must be written as a diagnostic of ONE extreme",
        "configuration, and the audit's scalar cells must be reported with the disclosure that they sit at a weight",
        "decay ~200x the standard value, at which the scalar configuration is broken.  The honest framing is then that",
        "the subsection PROTECTS the audit's scalar arm by naming the configuration, not that it explains a general",
        "granularity effect.  " + FLOOR_NOTE, NOT_FINER, NOT_DECOUPLED, NOT_ELSE],
    "THRESHOLD-W2-W3": [
        "The collapse survives one decade down (wd %s) and dies between %s and %s." % (WD_TOKEN["W2"], WD_TOKEN["W2"], WD_TOKEN["W3"]),
        "SENTENCE LICENSED: 'the collapse has a coupled weight-decay threshold between 1e-2 and 1e-3 at this cell; it is",
        "not confined to 0.1, and it is not present at the standard CIFAR value.'  EFFECT ON " + READING + ": the charge",
        "is PARTLY answered -- the phenomenon is a decade wide, not a single point -- but the audit's scalar cells still",
        "sit above the threshold and that must be disclosed.  " + FLOOR_NOTE, NOT_FINER, NOT_DECOUPLED, NOT_ELSE],
    "THRESHOLD-W3-W4": [
        "The collapse survives to wd %s and dies between %s and %s." % (WD_TOKEN["W3"], WD_TOKEN["W3"], WD_TOKEN["W4"]),
        "SENTENCE LICENSED: 'the collapse persists across two decades of coupled weight decay and is absent only at the",
        "standard CIFAR value.'  EFFECT ON " + READING + ": the corner-case charge is LARGELY answered -- the effect is",
        "not a knife edge -- but the boundary sits exactly at the value a referee will name, and the subsection must say",
        "so in the same breath.  " + FLOOR_NOTE, NOT_FINER, NOT_DECOUPLED, NOT_ELSE],
    "GRADED-ABOVE-THRESHOLD-W1": [
        "The COLLAPSE bar is crossed only at the anchor, but at least one higher rung still shows a PARTIAL gap (above",
        "%.0f pp, below the %.2f ratio bar).  SENTENCE LICENSED: 'the granularity gap is GRADED in the coupled weight" % (GAP_BAR, R50),
        "decay, not a threshold: it is total at 0.1 and shrinks with the decay, remaining substantial at lower values.'",
        "EFFECT ON " + READING + ": the corner-case charge is answered in PART -- the gap does not vanish outside the",
        "campaign's configuration -- but the word 'collapse' is licensed only at the anchor.  Quote G_W2..G_W4 as the",
        "finding; do not report a threshold.  " + FLOOR_NOTE, NOT_FINER, NOT_DECOUPLED, NOT_ELSE],
    "GRADED-ABOVE-THRESHOLD-W2": [
        "The COLLAPSE bar is crossed at wd %s and above, and at least one lower rung still shows a PARTIAL gap." % WD_TOKEN["W2"],
        "SENTENCE LICENSED: 'the collapse is complete down to 1e-2 and the granularity gap then decays smoothly rather",
        "than vanishing.'  EFFECT ON " + READING + ": the charge is answered in PART; report the graded curve, not a",
        "threshold.  " + FLOOR_NOTE, NOT_FINER, NOT_DECOUPLED, NOT_ELSE],
    "GRADED-ABOVE-THRESHOLD-W3": [
        "The COLLAPSE bar is crossed at wd %s and above, and the lowest rung still shows a PARTIAL gap." % WD_TOKEN["W3"],
        "SENTENCE LICENSED: 'the collapse persists across two decades and, at the standard CIFAR value, becomes a",
        "substantial but sub-bar granularity gap.'  EFFECT ON " + READING + ": the corner-case charge is answered -- the",
        "gap exists at the standard value too, though below the collapse bar.  Quote G_W4 as the number that answers the",
        "referee.  " + FLOOR_NOTE, NOT_FINER, NOT_DECOUPLED, NOT_ELSE],
    "NON-MONOTONE": [
        "A LOWER coupled weight decay collapses while a HIGHER one does not.  NO threshold sentence and NO corner-case",
        "verdict is licensed in either direction: the intervention is not described by 'how much weight decay' alone at",
        "this cell, and the result must be reported as the non-monotonicity it is, with all four rungs' levels and the",
        "seed spreads.  This outcome is registered BEFORE any run precisely so it cannot be re-described afterwards as a",
        "threshold with one rung explained away.  " + FLOOR_NOTE, NOT_DECOUPLED, NOT_ELSE],
}

CARRIER_LICENSE = {
    "CAR-REC": ("the carrier account HOLDS at this rung: masking the three ctd1 carriers' coupled decay brings the "
                "scalar arm within %.0f pp of its own layerwise arm at wd %s, as it does at 0.1 (cwd3).  ONE rung, "
                "ONE network, between-batch only against cwd3." % (MATCH_BAR, WD_TOKEN[CARRIER_RUNG])),
    "CAR-PART": ("the carrier mask does PART of the work at this rung: it lifts the scalar arm off the floor but not "
                 "to the layerwise level.  Read P_CARW2 and D_CARW2 together; neither is a share of anything."),
    "CAR-NULL": ("the carrier mask does NOT lift the run at this rung -- a LOCATION at k01W2's floor, a BOUND, not a "
                 "measured zero (164.6).  The carrier account, which holds at wd 0.1 (cwd3), does NOT transfer to wd "
                 "%s at this cell.  That is an ADVERSE reading for the carrier account and is registered as such."
                 % WD_TOKEN[CARRIER_RUNG]),
    "CAR-UNREADABLE": ("rung %s is NOT in the COLLAPSE state, so there is no collapse for the carrier mask to remove "
                       "and NOTHING about the carrier account is licensed here.  This is cmo1's ISO-UNREADABLE at wd 0 "
                       "(CORRECTIONS 264) repeating one rung higher, and it was registered as a reachable outcome "
                       "before any run." % CARRIER_RUNG),
    "CAR-PATCH-NOT-VERIFIED": ("CARW2's DECAY_MASK record audit failed, so the carrier arm is NOT READ AT ALL.  A mask "
                               "that never bit would read as CAR-NULL -- 'the carrier account fails at 1e-2' -- and "
                               "only this gate separates the two."),
}

ACCOUNT_TOKENS = ("COLLAPSE-AT-ALL-VALUES", "THRESHOLD-W1-W2", "THRESHOLD-W2-W3", "THRESHOLD-W3-W4",
                  "GRADED-ABOVE-THRESHOLD-W1", "GRADED-ABOVE-THRESHOLD-W2", "GRADED-ABOVE-THRESHOLD-W3",
                  "NON-MONOTONE")


def f32(x):
    """the float32 value of a python float, re-derived through struct."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ---------------------------------------------------------------------------------------------------------------------
# THE NETWORK, REBUILT FROM THE ARCHITECTURE (no manifest, no repo module)
#   ResNet18 for CIFAR-100: 3x3 stem (no maxpool), four stages of two BasicBlocks with widths 64 / 128 / 256 / 512,
#   a 1x1 convolutional downsample + BatchNorm on the first block of stages 2..4, and a 512 -> 100 linear head.
# ---------------------------------------------------------------------------------------------------------------------
def r18_c100_tensors():
    t = [("conv1.weight", 64 * 3 * 3 * 3, "Conv2d"), ("bn1.weight", 64, "BatchNorm2d"), ("bn1.bias", 64, "BatchNorm2d")]
    cin = 64
    for li, w in enumerate((64, 128, 256, 512), 1):
        for b in (0, 1):
            stride = 2 if (li > 1 and b == 0) else 1
            fan = cin if b == 0 else w
            pre = "layer%d.%d." % (li, b)
            t += [(pre + "conv1.weight", w * fan * 3 * 3, "Conv2d"),
                  (pre + "bn1.weight", w, "BatchNorm2d"), (pre + "bn1.bias", w, "BatchNorm2d"),
                  (pre + "conv2.weight", w * w * 3 * 3, "Conv2d"),
                  (pre + "bn2.weight", w, "BatchNorm2d"), (pre + "bn2.bias", w, "BatchNorm2d")]
            if stride != 1 or fan != w:
                t += [(pre + "shortcut.0.weight", w * fan, "Conv2d"),
                      (pre + "shortcut.1.weight", w, "BatchNorm2d"), (pre + "shortcut.1.bias", w, "BatchNorm2d")]
        cin = w
    t += [("linear.weight", 512 * 100, "Linear"), ("linear.bias", 100, "Linear")]
    out = []
    for n, q, o in t:
        nd = 1 if (o in ("BatchNorm2d", "GroupNorm") or n.endswith(".bias")) else (4 if o == "Conv2d" else 2)
        out.append((n, q, o, nd))
    return out


TENSORS = r18_c100_tensors()
NAMES = [x[0] for x in TENSORS]


def dm_indices(spec):
    """PATCH_DECAYMASK._dm_resolve, re-derived: `normscale` = every 1-dim `.weight`; otherwise a `+` name list."""
    if spec == "normscale":
        return sorted(i for i, t in enumerate(TENSORS) if t[0].endswith(".weight") and t[3] == 1)
    return sorted(NAMES.index(x) for x in spec.split("+"))


def dm_witness(spec, wd):
    """the witness line, built with THE ARM'S OWN weight decay -- CARW2 runs at 1e-2, so it says wd=0.01."""
    if not spec:
        return "DECAY_MASK: off"
    idx = dm_indices(spec)
    return ("DECAY_MASK: on base=SGDm wd=%r spec=%s masked=%d of=%d numel=%d idx=%s names=%s"
            % (wd, spec, len(idx), len(TENSORS), sum(TENSORS[i][1] for i in idx),
               ",".join("%d" % (i + 1) for i in idx), ",".join(TENSORS[i][0] for i in idx)))


WITNESS_DM = dict((a, dm_witness(DMASK[a], WDF[a])) for a in ARMS)
SET_IDX = dict((a, tuple(i + 1 for i in dm_indices(DMASK[a]))) for a in MASKED)


def manifest_text():
    """cwd5_design.manifest_text, re-derived.  NOT cwd_design's: this batch prints a WD_LADDER line instead of one
    global WD_BASE, and an ARMWD line per arm, because the weight decay is the AXIS."""
    lines = ["NETWORK %s" % NET, "NUM_PARAM_TENSORS %d" % len(TENSORS), "TOTAL_PARAMS %d" % sum(x[1] for x in TENSORS),
             "CLIP_C %s" % CLIP, "EPOCHS %d" % EPOCHS, "META_STEPS %d" % (EPOCHS * STEPS_PER_EPOCH)]
    lines.append("WD_LADDER %s" % ",".join("%s=%s" % (r, WD_TOKEN[r]) for r in RUNG_IDS))
    for i, (n, q, o, nd) in enumerate(TENSORS, 1):
        lines.append("TENSOR %d %s %d %s ndim=%d" % (i, n, q, o, nd))
    lines.append("NORMSCALE %s" % ",".join("%d:%s" % (i + 1, TENSORS[i][0]) for i in dm_indices("normscale")))
    for a in ARMS:
        lines.append("ARMSPEC %s SPEC %s TYPE %s" % (a, SPEC[a], PT_TYPE[a]))
        lines.append("ARMWD %s RUNG %s WD %s" % (a, RUNG_OF[a], WD[a]))
        lines.append("DECAYMASK %s %s" % (a, DMASK[a] or "off"))
        lines.append("DWITNESS %s %s" % (a, WITNESS_DM[a]))
    for a in MASKED:
        idx = dm_indices(DMASK[a])
        lines.append("SET %s idx=%s numel=%d owners=%s widths=%s classes=%s" % (
            a, ",".join("%d" % (i + 1) for i in idx), sum(TENSORS[i][1] for i in idx),
            ",".join(TENSORS[i][2] for i in idx), ",".join("%d" % TENSORS[i][1] for i in idx),
            ",".join("scale" if TENSORS[i][0].endswith(".weight") else "shift" for i in idx)))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
# arithmetic
# ---------------------------------------------------------------------------------------------------------------------
def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def fmt(x, nd=4):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---------------------------------------------------------------------------------------------------------------------
# raw `.out` reading -- no regex
# ---------------------------------------------------------------------------------------------------------------------
def _num(tok):
    tok = tok.strip()
    return float("nan") if tok == "nan" else float(tok)


def parse_out(path):
    rec = {"eps": {}, "args": [], "env": [], "pt": [], "device": None, "traceback": False, "run_done": False}
    for kind in KINDS:
        rec[kind] = []
    for ln in open(path, errors="replace"):
        ln = ln.rstrip("\n")
        s = ln.strip()
        if s == "RUN_DONE":
            rec["run_done"] = True
        if "Traceback (most recent call last)" in ln:
            rec["traceback"] = True
        if ln.startswith("ARGS:"):
            rec["args"].append(ln)
        elif ln.startswith("ENV:"):
            rec["env"].append(ln)
        elif ln.startswith("PROBE_TENSOR:"):
            rec["pt"].append(ln)
        else:
            for kind in KINDS:
                if ln.startswith(kind):
                    rec[kind].append(ln)
        if ln.startswith("Epoch "):
            parts = ln.split(",")
            if len(parts) >= 3 and "Train Accuracy" in parts[1] and "Test Accuracy" in parts[2]:
                head = parts[0].split()
                if len(head) == 2 and all(c.isdigit() for c in head[1]):
                    rec["eps"][int(head[1])] = (_num(parts[1].split(":")[1].split("%")[0]),
                                                _num(parts[2].split(":")[1].split("%")[0]))
        if rec["device"] is None and s.endswith(" MiB") and ", " in s and not s.startswith(("ARGS:", "ENV:", "Epoch")):
            rec["device"] = s.split(",")[0].strip()
    tail = [rec["eps"].get(e) for e in range(EPOCHS - 5, EPOCHS)]
    if all(x is not None for x in tail) and all(finite(x[0]) and finite(x[1]) for x in tail):
        rec["plateau5"] = mean([x[1] for x in tail])
        rec["train5"] = mean([x[0] for x in tail])
    else:
        rec["plateau5"] = rec["train5"] = None
    rec["n_epochs"] = len(rec["eps"])
    rec["complete"] = (sorted(rec["eps"]) == list(range(EPOCHS)) and rec["run_done"] and not rec["traceback"]
                       and rec["plateau5"] is not None)
    return rec


def parse_args_line(line):
    toks = line[len("ARGS:"):].split()
    flags, cur = {}, None
    for t in toks:
        if t.startswith("--"):
            cur = t[2:]
            flags.setdefault(cur, []).append(None)
        elif cur is not None:
            v = flags[cur][-1]
            flags[cur][-1] = t if v is None else v + " " + t
    return flags


def read_runs(runsdir):
    cand = {}
    for fn in sorted(os.listdir(runsdir)):
        if not fn.endswith(".out") or not fn.startswith(PREFIX + "-"):
            continue
        bits = fn[:-len(".out")].split("-")
        if len(bits) != 4 or bits[0] != PREFIX or bits[1] not in ARMS:
            continue
        if not bits[2].startswith("s") or not bits[2][1:].isdigit() or not bits[3].isdigit():
            continue
        arm, seed, jid = bits[1], int(bits[2][1:]), int(bits[3])
        r = parse_out(os.path.join(runsdir, fn))
        r.update({"file": fn, "job": jid, "arm": arm, "seed": seed})
        cand.setdefault((arm, seed), []).append(r)
    got, notes = {}, []
    for k in sorted(cand):
        lst = cand[k]
        comp = [r for r in lst if r["complete"]]
        pick = max(comp or lst, key=lambda r: r["job"])
        if len(lst) > 1:
            notes.append("%s-s%d has %d files (%d complete); using %s" % (k[0], k[1], len(lst), len(comp), pick["file"]))
        got[k] = pick
    return got, notes


def load_probe(runsdir, arm, seed):
    for base in (os.path.join(runsdir, PREFIX), runsdir):
        p = os.path.join(base, "probe_%s-%s-s%d" % (PREFIX, arm, seed), "probe.jsonl")
        if os.path.exists(p):
            out = []
            for ln in open(p, errors="replace"):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    out.append(None)
                    continue
                out.append(r if isinstance(r, dict) else None)
            return out
    return None


def read_kv(path):
    d = {}
    for ln in open(path):
        p = ln.split()
        if len(p) >= 2:
            d.setdefault(p[0], " ".join(p[1:]))
    return d


def tail_slope(eps, lo=80, hi=99):
    xs = [e for e in range(lo, hi + 1) if e in eps]
    if len(xs) < 3:
        return None
    ys = [eps[e][1] for e in xs]
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return None if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


# ---------------------------------------------------------------------------------------------------------------------
# G-BITE, re-derived: Lion on EACH ARM'S OWN group count; no hold key; the DECAY_MASK record audit
# ---------------------------------------------------------------------------------------------------------------------
def lion_natural(beta_pre, mom_pre, z):
    a, b = B2 * mom_pre, (1 - B2) * z
    L = a + b
    tie = abs(L) <= TIE_REL * (abs(a) + abs(b))
    s = (L > 0) - (L < 0)
    return max(LO, min(HI, beta_pre - MS * s)), tie


def _groupvec(r, key, ng):
    v = r.get(key)
    if not (isinstance(v, list) and len(v) == 1 and isinstance(v[0], list) and len(v[0]) == ng):
        return None
    try:
        return [float(x) for x in v[0]]
    except (TypeError, ValueError):
        return None


def lion_check(recs, ng):
    d = {"n": 0, "bad_rec": 0, "bad_step": 0, "bad_lion": 0, "ties": 0, "nonfinite": 0, "worst": 0.0}
    if recs is None:
        d["why"] = "no probe.jsonl"
        return False, d
    for k, r in enumerate(recs):
        if r is None or not isinstance(r.get("step"), int) or not isinstance(r.get("beta"), list) or len(r["beta"]) != ng:
            d["bad_rec"] += 1
            continue
        try:
            beta = [float(x) for x in r["beta"]]
        except (TypeError, ValueError):
            d["bad_rec"] += 1
            continue
        bpre, mom, z = _groupvec(r, "beta_pre", ng), _groupvec(r, "mom_pre", ng), _groupvec(r, "z_agg", ng)
        if bpre is None or mom is None or z is None:
            d["bad_rec"] += 1
            continue
        if not all(math.isfinite(x) for x in beta + bpre + mom + z):
            d["nonfinite"] += 1
            continue
        if r["step"] != PROBE * k:
            d["bad_step"] += 1
        d["n"] += 1
        for g in range(ng):
            nat, tie = lion_natural(bpre[g], mom[g], z[g])
            if tie:
                d["ties"] += 1
                continue
            e = abs(nat - beta[g])
            d["worst"] = max(d["worst"], e)
            d["bad_lion"] += e > BH_TOL
    ok = (len(recs) == N_RECORDS and d["bad_rec"] == 0 and d["n"] + d["nonfinite"] == N_RECORDS
          and d["bad_step"] == 0 and d["bad_lion"] == 0)
    return ok, d


def dm_audit(recs, k):
    """PATCH_DECAYMASK's own record on ONE run, re-derived.  k = the arm's number of masked tensors (0 = unmasked)."""
    d = {"n": 0, "bad_rec": 0, "bad_n": 0, "bad_masked": 0, "bad_skipped": 0, "bad_types": 0, "keys_on_unmasked": 0,
         "nonfinite": 0, "pos_wdterm": 0, "last": None}
    if recs is None:
        d["why"] = "no probe.jsonl"
        return False, d
    prev = -1
    for r in recs:
        if r is None or not isinstance(r.get("step"), int):
            d["bad_rec"] += 1
            continue
        d["n"] += 1
        has = [key for key in r if key.startswith("dm_")]
        if k == 0:
            d["keys_on_unmasked"] += bool(has)
            continue
        dn = r.get("dm_n")
        if not (isinstance(dn, int) and not isinstance(dn, bool) and dn == r["step"] + 2 and dn > prev):
            d["bad_n"] += 1
        prev = dn if (isinstance(dn, int) and not isinstance(dn, bool)) else prev
        if r.get("dm_masked") != k:
            d["bad_masked"] += 1
        if not (isinstance(dn, int) and not isinstance(dn, bool) and r.get("dm_skipped") == dn * k):
            d["bad_skipped"] += 1
        wt, nr, am, sm = r.get("dm_wdterm"), r.get("dm_norm"), r.get("dm_absmin"), r.get("dm_small")
        types_ok = (isinstance(wt, (int, float)) and not isinstance(wt, bool) and isinstance(nr, list) and len(nr) == k
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in nr)
                    and isinstance(am, (int, float)) and not isinstance(am, bool)
                    and isinstance(sm, int) and not isinstance(sm, bool) and sm >= 0
                    and sorted(has) == sorted(DM_KEYS))
        if not types_ok:
            d["bad_types"] += 1
            continue
        if not (finite(wt) and all(finite(x) for x in nr) and finite(am)):
            d["nonfinite"] += 1
            continue
        if wt < 0 or am < 0:
            d["bad_types"] += 1
            continue
        d["pos_wdterm"] += wt > 0
        d["last"] = {"dm_n": dn, "dm_skipped": r.get("dm_skipped"), "dm_wdterm": wt, "dm_absmin": am, "dm_small": sm}
    ok = (len(recs) == N_RECORDS and d["bad_rec"] == 0 and d["n"] == N_RECORDS)
    if k == 0:
        ok = ok and d["keys_on_unmasked"] == 0
    else:
        ok = (ok and d["bad_n"] == 0 and d["bad_masked"] == 0 and d["bad_skipped"] == 0 and d["bad_types"] == 0
              and d["pos_wdterm"] > 0)
    return ok, d


def hold_keys_present(recs):
    if recs is None:
        return None
    return sum(1 for r in recs if isinstance(r, dict) and any(key.startswith(HOLD_PREFIXES) for key in r))


def bite(arm, recs):
    okl, dl = lion_check(recs, NG[arm])
    hk = hold_keys_present(recs)
    okd, dd = dm_audit(recs, K_MASKED[arm])
    return (okl and okd and hk == 0), dl, dd, hk


# ---------------------------------------------------------------------------------------------------------------------
# the rung states and the branch, re-derived (CORRECTIONS 281.4)
# ---------------------------------------------------------------------------------------------------------------------
def rung_state(M, r):
    k, L = M[SCALAR_OF[r]], M[LAYER_OF[r]]
    if L < REF_MIN:
        return "UNREADABLE"
    if L > 0 and k <= R50 * L:
        return "COLLAPSE"
    if L - k < GAP_BAR:
        return "NOGAP"
    return "PARTIAL"


def carrier_state(M, states):
    if states[RUNG_IDS.index(CARRIER_RUNG)] != "COLLAPSE":
        return "CAR-UNREADABLE"
    if M[CAR] >= M[LAYER_OF[CARRIER_RUNG]] - MATCH_BAR:
        return "CAR-REC"
    if M[CAR] <= M[SCALAR_OF[CARRIER_RUNG]] + NULL_BAR:
        return "CAR-NULL"
    return "CAR-PART"


def account_of(states):
    c = 0
    while c < len(states) and states[c] == "COLLAPSE":
        c += 1
    if "COLLAPSE" in states[c:]:
        return "NON-MONOTONE"
    if c == len(states):
        return "COLLAPSE-AT-ALL-VALUES"
    if "PARTIAL" in states[c:]:
        return "GRADED-ABOVE-THRESHOLD-%s" % RUNG_IDS[c - 1]
    return "THRESHOLD-%s-%s" % (RUNG_IDS[c - 1], RUNG_IDS[c])


def decide(M, diverged=False, carrier_ok=True):
    if diverged:
        return "UNRESOLVED-DIVERGED", None
    bad_ref = [r for r in RUNG_IDS if M[LAYER_OF[r]] < REF_MIN]
    if bad_ref:
        return "UNRESOLVED-REFERENCE", "REF-BELOW-%.0f-AT-%s" % (REF_MIN, "-".join(bad_ref))
    states = [rung_state(M, r) for r in RUNG_IDS]
    if M[SCALAR_OF["W1"]] > K01_MAX or states[0] != "COLLAPSE":
        return "ANCHOR-NOT-COLLAPSED", "W1-%s" % states[0]
    tok = account_of(states)
    cw = carrier_state(M, states) if carrier_ok else "CAR-PATCH-NOT-VERIFIED"
    return tok, "+".join("%s-%s" % (r, s) for r, s in zip(RUNG_IDS, states)) + "+" + cw


# ---------------------------------------------------------------------------------------------------------------------
# the corpus, split by hand (no `csv`, no corpus_exclusions import)
# ---------------------------------------------------------------------------------------------------------------------
def _split_csv(ln):
    out, cur, q = [], "", False
    for ch in ln:
        if ch == '"':
            q = not q
        elif ch == "," and not q:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def corpus_filtered_count(csvpath, tsvpath):
    """the number of corpus rows corpus_exclusions.filter_rows keeps, re-derived by hand: the exclusions TSV is a
    tab-separated table with `#` comment lines, and a row is dropped iff its (run, job_id) PAIR is listed.  cwd5's own
    rows never enter (the scorer drops them before filtering)."""
    if not csvpath or not os.path.exists(csvpath):
        return None
    excl = set()
    if tsvpath and os.path.exists(tsvpath):
        lines = [ln.rstrip("\n") for ln in open(tsvpath, errors="replace") if ln.strip() and not ln.startswith("#")]
        if not lines:
            return None
        head = lines[0].split("\t")
        if "run" not in head or "job_id" not in head:
            return None
        ir, ij = head.index("run"), head.index("job_id")
        for ln in lines[1:]:
            f = ln.split("\t")
            if len(f) != len(head):
                return None
            excl.add((f[ir], f[ij]))
    n = 0
    head = None
    for ln in open(csvpath, errors="replace"):
        ln = ln.rstrip("\n")
        if not ln.strip():
            continue
        cells = _split_csv(ln)
        if head is None:
            head = [c.strip() for c in cells]
            if "run" not in head or "job_id" not in head:
                return None
            continue
        ir, ij = head.index("run"), head.index("job_id")
        run = cells[ir] if len(cells) > ir else ""
        jid = cells[ij] if len(cells) > ij else ""
        if run.startswith(PREFIX + "-"):
            continue
        if (run, jid) in excl:
            continue
        n += 1
    return n


# ---------------------------------------------------------------------------------------------------------------------
# REBUILD the scorer's whole stdout
# ---------------------------------------------------------------------------------------------------------------------
def rebuild(runsdir):
    L = []
    P = L.append
    bar = "=" * 78
    P(bar)
    P(" cwd5 -- ResNet18_c100: THE COUPLED WEIGHT-DECAY LADDER.  Does the shared-step-size collapse exist at")
    P("         NORMAL weight-decay values, or only at the campaign's 0.1?")
    P(" %s / %s   %d epochs   seeds %s   AUGMENT=%s  BETA_CLIP=%s  PROBE=%d  PROBE_TENSOR=1"
      % (NET, DSET, EPOCHS, ",".join(str(s) for s in SEEDS), AUG, CLIP, PROBE))
    for r in RUNG_IDS:
        P("   rung %s  --weight-decay-base %-5s   %-7s (scalar) | %-7s (layerwise)%s"
          % (r, WD_TOKEN[r], SCALAR_OF[r], LAYER_OF[r],
             "   + %s (scalar, DECAY_MASK = the three ctd1 carriers)" % CAR if r == CARRIER_RUNG else ""))
    P(" CO-PRIMARY: G_W4 = %s - %s, the granularity gap at the STANDARD CIFAR weight decay."
      % (LAYER_OF["W4"], SCALAR_OF["W4"]))
    P(" Every state is WITHIN its rung.  plateau5 = mean TEST over epochs %d..%d of each run's own .out."
      % (EPOCHS - 5, EPOCHS - 1))
    P(" BARS ARE FROZEN LITERALS (O2).  DECOUPLED WEIGHT DECAY IS NOT TESTED (281.2) and is stamped on every FINAL.")
    P(bar)
    P("@CORPUS@")                                          # host-dependent; checked separately

    runs, rnotes = read_runs(runsdir)
    P("")
    P("COMPLETE  %d runs, epochs 0..%d, RUN_DONE, no traceback, finite plateau5" % (len(ARMS) * len(SEEDS), EPOCHS - 1))
    for n_ in rnotes:
        P("  NOTE %s" % n_)
    for s in SEEDS:
        for a in ARMS:
            r = runs.get((a, s))
            ok = r is not None and r["complete"]
            P("  %-4s %s-%s-s%d  %s" % ("OK" if ok else "MISS", PREFIX, a, s,
              "absent" if r is None else "%d epoch lines, RUN_DONE=%s, traceback=%s, plateau5 %s (train %s)"
              % (r["n_epochs"], r["run_done"], r["traceback"], fmt(r["plateau5"]), fmt(r["train5"]))))
    missing = [(a, s) for s in SEEDS for a in ARMS if not (runs.get((a, s)) and runs[(a, s)]["complete"])]
    if missing:
        return L, {"incomplete": missing, "runs": runs, "manifest": manifest_text()}

    P("")
    P("G-ARGS / G-WDPATH   every run's OWN ARGS line says what its file name registers, INCLUDING its rung's")
    P("          --weight-decay-base token and its grain.  That the flag reaches the update is proved on the real")
    P("          GPU path by the bite job (bitwise w'/h'/m' at all four values) and in batch by %s's wd=0.01 witness." % CAR)
    for s in SEEDS:
        for a in ARMS:
            r = runs[(a, s)]
            if len(r["args"]) != 1:
                P("  FAIL G-ARGS %s-s%d exactly one ARGS line   %d" % (a, s, len(r["args"])))
                continue
            f = parse_args_line(r["args"][0])
            dup = [k for k, v in f.items() if len(v) > 1]
            want = {"optimizer": "HF", "alg-base": "SGDm", "alg-meta": "Lion", "dataset": DSET,
                    "NN-name": NET, "batch-size": str(BATCH), "num-epochs": str(EPOCHS),
                    "meta-stepsize": MST, "alpha0": A0, "stepsize-groups": SPEC[a],
                    "seed": str(s), "run-name": "%s-%s-s%d" % (PREFIX, a, s),
                    "momentum-param-base": "0.99", "weight-decay-base": WD[a], "momentum-param-meta": "0.99",
                    "Lion-beta2-meta": "0.9", "weight-decay-meta": "0", "gamma": "1"}
            bad = [k for k, v in want.items() if f.get(k, [None])[-1] != v]
            extra = ("   repeated %s; wrong %s" % (dup, ["%s=%s" % (k, f.get(k, [None])[-1]) for k in bad])) if (dup or bad) else ""
            P("  %-4s G-ARGS %s-s%d (wd %s, %s)%s" % ("PASS" if (not dup and not bad) else "FAIL", a, s, WD[a], SPEC[a], extra))
    wds = sorted(set(parse_args_line(runs[(a, s)]["args"][0]).get("weight-decay-base", [None])[-1]
                     for a in ARMS for s in SEEDS if len(runs[(a, s)]["args"]) == 1))
    okw = wds == sorted(set(WD_TOKEN[r] for r in RUNG_IDS))
    P("  %-4s G-WDPATH the batch carries EXACTLY the four registered weight-decay tokens and no other   %s"
      % ("PASS" if okw else "FAIL", str(wds)))

    P("")
    P("G-ENV     the environment rode the ENV line on every run")
    envs = sorted(set(" ".join(t for t in ln.split() if not t.startswith("PROBE_DIR=")) for r in runs.values() for ln in r["env"]))
    ok1 = len(envs) == 1 and all(len(r["env"]) == 1 for r in runs.values())
    P("  %-4s G-ENV one distinct ENV line (PROBE_DIR stripped), one per run   %d distinct"
      % ("PASS" if ok1 else "FAIL", len(envs)))
    for ln in envs:
        P("        %s" % ln)
    P("  %-4s G-ENV the ENV line is cvt1's ... cvt9's / cwd1's / cwd3's / cwd4's, byte for byte"
      % ("PASS" if envs == [ENV_EXPECTED] else "FAIL"))
    P("  DECLARED: the runner's ENV line carries NO weight decay, so the ENV half cannot discriminate the rungs.")
    P("  The rungs are discriminated by each run's OWN ARGS line (G-ARGS above and RULE 20's per-run --expect) and")
    P("  the grains by the PROBE_TENSOR type below.  Stated at registration, not discovered afterwards.")
    badpt = [(a, s) for (a, s), r in sorted(runs.items())
             if len(r["pt"]) != 1 or not r["pt"][0].startswith("PROBE_TENSOR: on every=%d type=%s tensors=%d "
                                                               % (PROBE, PT_TYPE[a], NTENS))]
    P("  %-4s G-ENV every run printed ONE `PROBE_TENSOR: on every=%d type=<its own grain> tensors=%d`   %s"
      % ("PASS" if not badpt else "FAIL", PROBE, NTENS,
         ("bad: %s" % badpt[:5]) if badpt else "%d runs (%d scalar, %d layerwise)"
         % (len(runs), sum(1 for a in ARMS if SPEC[a] == "scalar") * len(SEEDS),
            sum(1 for a in ARMS if SPEC[a] == "layerwise") * len(SEEDS))))

    P("")
    P("G-WITNESS one line of each kind per run == the registered witness (VOTE_W / BETA_HOLD / GROUP_HOLD / REST_HOLD off; DECAY_MASK)")
    for s in SEEDS:
        for a in ARMS:
            r = runs[(a, s)]
            for kind, want in list(OFF_WITNESS.items()) + [("DECAY_MASK", WITNESS_DM[a])]:
                good = r[kind] == [want]
                P("  %-4s G-WITNESS %s %s-s%d%s" % ("PASS" if good else "FAIL", kind, a, s,
                  "" if good else "   got %r" % ([x[:170] for x in r[kind][:2]],)))
            others = [k for k in ("COMP_HOLD", "WINDOW_HOLD") if r[k]]
            P("  %-4s G-WITNESS %s-s%d prints no COMP_HOLD / WINDOW_HOLD line (cvt8 lineage)%s"
              % ("PASS" if not others else "FAIL", a, s, ("   " + str(others)) if others else ""))

    P("")
    P("G-STRUCT  the batch's own PARTITION-MANIFEST.txt == the frozen manifest, byte for byte")
    P("@MANIFESTPATH@")
    mt = manifest_text()
    mp = os.path.join(runsdir, PREFIX, "PARTITION-MANIFEST.txt")
    txt = open(mp).read() if os.path.exists(mp) else ""
    P("  %-4s G-STRUCT byte-identical to cwd5_design.manifest_text(CWD5)   %d vs %d bytes"
      % ("PASS" if txt == mt else "FAIL", len(txt), len(mt)))

    P("")
    P("G-PROV    PROVENANCE.txt")
    P("@PROVPATH@")
    pp = os.path.join(runsdir, PREFIX, "PROVENANCE.txt")
    prov = read_kv(pp) if os.path.exists(pp) else {}
    for key, lab in (("MODE", "MODE submit"),
                     ("BUILD_NETWORK_SHA256", "build_network.py == cvt8's"),
                     ("HF_SHA256", "HF.py == cwd1's PATCH_DECAYMASK tree, UNCHANGED"),
                     ("HF_PRE_DECAYMASK_SHA256", "HF.py.pre_decaymask == cvt8's HF.py"),
                     ("RUNNER_SHA256", "runner == run_cifar_cwd1.sh, UNCHANGED"),
                     ("SCORER_SHA256", "SCORER == this file"),
                     ("DESIGN_SHA256", "DESIGN == the cwd5_design.py this scorer loaded"),
                     ("CWD_DESIGN_SHA256", "the reused cwd_design.py this scorer loaded"),
                     ("COMMON_SHA256", "COMMON == the cwd_common.py this scorer loaded")):
        P("  %-4s G-PROV %s   %s=%s" % ("PASS" if prov.get(key) == PROV_WANT[key] else "FAIL", lab, key,
                                        str(prov.get(key))[:16]))

    P("")
    P("LEVELS    plateau5 (TEST) and TRAIN, IN BATCH; tail slope = TEST OLS over epochs 80-99 (descriptive)")
    arm_v = dict((a, [runs[(a, s)]["plateau5"] for s in SEEDS]) for a in ARMS)
    arm_t = dict((a, [runs[(a, s)]["train5"] for s in SEEDS]) for a in ARMS)
    for a in ARMS:
        v, t = arm_v[a], arm_t[a]
        sl = [tail_slope(runs[(a, s)]["eps"]) for s in SEEDS]
        P("  %-7s wd %-5s %-9s TEST %.4f  sd %s  range %.4f   TRAIN %.4f   tail slope %s pp/ep"
          % (a, WD[a], SPEC[a], mean(v), fmt(sd(v), 4), max(v) - min(v), mean(t), "/".join(fmt(x, 3) for x in sl)))
        P("           seeds: %s" % ", ".join("s%d %.4f (train %.4f)" % (s, x, y) for s, x, y in zip(SEEDS, v, t)))
    M = dict((a, mean(arm_v[a])) for a in ARMS)
    T = dict((a, mean(arm_t[a])) for a in ARMS)
    hi = max(M.values())
    P("")
    P("G-FLOOR / G-CEIL")
    P("  %-4s G-FLOOR max arm mean >= %.2f pp (chance %.2f)   max %.4f"
      % ("PASS" if hi >= FLOOR_MIN else "FAIL", FLOOR_MIN, CHANCE, hi))
    P("  %-4s G-CEIL max arm mean <= %.2f pp   max %.4f" % ("PASS" if hi <= CEIL_MAX else "FAIL", CEIL_MAX, hi))

    P("")
    P("G-BITE    SPLIT, as registered.  LADDER gate (the 8 unmasked arms) is HARD; CARRIER gate (%s) suppresses" % CAR)
    P("          only the carrier word.  Lion is recomputed on each arm's OWN group count (scalar 1, layerwise %d)." % NTENS)
    ladder_fail, carrier_fail = [], []
    nonfin = {}
    for s in SEEDS:
        for a in ARMS:
            recs = load_probe(runsdir, a, s)
            ok, dl, dd, hk = bite(a, recs)
            nonfin[(a, s)] = dl["nonfinite"] + dd["nonfinite"]
            P("  %-4s G-BITE %-7s s%d  records %s  groups %d  lion-mismatch %d (ties %d, worst %.1e)  bad %d  "
              "nonfinite %d  hold-key records %s | dm: k=%d bad_n %d bad_skipped %d bad_masked %d bad_types %d "
              "on-unmasked %d pos_wdterm %d nonfinite %d last %s%s"
              % ("PASS" if ok else "FAIL", a, s, "absent" if recs is None else len(recs), NG[a], dl["bad_lion"],
                 dl["ties"], dl["worst"], dl["bad_rec"], dl["nonfinite"], hk, K_MASKED[a], dd["bad_n"],
                 dd["bad_skipped"], dd["bad_masked"], dd["bad_types"], dd["keys_on_unmasked"], dd["pos_wdterm"],
                 dd["nonfinite"], dd["last"],
                 ("  (%s)" % (dl.get("why") or dd.get("why"))) if (dl.get("why") or dd.get("why")) else ""))
            if not ok:
                (carrier_fail if a == CAR else ladder_fail).append("%s-s%d" % (a, s))
    carrier_ok = not carrier_fail
    if carrier_fail:
        P("  CARRIER GATE FAILED on %s -- the carrier word is SUPPRESSED; the ladder is read as registered."
          % ",".join(carrier_fail))

    P("")
    P("G-HW      DISCLOSURE ONLY (271.4(5)): the GPU each run's own nvidia-smi line names -- NO gate reads it")
    hw = dict(((a, s), runs[(a, s)]["device"]) for s in SEEDS for a in ARMS)
    for a in ARMS:
        P("  %-7s %s" % (a, "  ".join("s%d %s" % (s, hw[(a, s)]) for s in SEEDS)))
    hwset = sorted(set(str(v) for v in hw.values()))
    P("  distinct devices: %s" % hwset)

    P("")
    P("SIGMA     O2: max(frozen floor, in-batch) -- nothing from the corpus")
    ss = sum(sum((x - M[a]) ** 2 for x in arm_v[a]) for a in ARMS)
    df = sum(len(arm_v[a]) - 1 for a in ARMS)
    sigma_in = math.sqrt(ss / df)
    which, sigma_used = ("SIGMA_PRIOR_frozen", SIGMA_PRIOR) if SIGMA_PRIOR >= sigma_in else ("SIGMA_INBATCH", sigma_in)
    P("  %-20s %.6f" % ("SIGMA_PRIOR_frozen", SIGMA_PRIOR))
    P("  %-20s %.6f" % ("SIGMA_INBATCH", sigma_in))
    se = sigma_used * math.sqrt(2.0 / len(SEEDS))
    P("  SIGMA_USED           %.6f  (= %s)" % (sigma_used, which))
    P("  SE_ARM_DIFF          %.6f  = SIGMA_USED * sqrt(2/%d)  (df_inbatch %d)" % (se, len(SEEDS), df))
    P("  2 SE (the HALF-WIDTH of a +/-2 SE interval, NOT a bound on any effect -- 278.6 C1): %.6f" % (2 * se))

    P("")
    P("THE LADDER (every contrast WITHIN its rung)")
    states = [rung_state(M, r) for r in RUNG_IDS]
    CP = {}
    for r, st in zip(RUNG_IDS, states):
        k, Lw = SCALAR_OF[r], LAYER_OF[r]
        g = M[Lw] - M[k]
        CP["G_" + r] = (g, T[Lw] - T[k])
        ratio = (M[k] / M[Lw]) if M[Lw] else float("nan")
        P("  %-3s wd %-5s  scalar %7.4f  layerwise %7.4f   G_%s = %+.4f pp = %+.2f SE   R_%s = %.4f   -> %s"
          % (r, WD_TOKEN[r], M[k], M[Lw], r, g, g / se, r, ratio, st))
        P("           +/-2 SE interval [%+.4f, %+.4f]   TRAIN gap %+.4f   seeds %s"
          % (g - 2 * se, g + 2 * se, T[Lw] - T[k],
             " / ".join("%+.3f" % (runs[(Lw, s)]["plateau5"] - runs[(k, s)]["plateau5"]) for s in SEEDS)))
        P("           bars at this rung: COLLAPSE iff scalar <= %.4f (%.2f x layerwise) ; NOGAP iff G < %.1f pp ;"
          " reference healthy iff layerwise >= %.1f" % (R50 * M[Lw], R50, GAP_BAR, REF_MIN))
    P("  CO-PRIMARY G_W4 = %+.4f pp = %+.2f SE -- the granularity gap at the STANDARD CIFAR weight decay %s"
      % (CP["G_W4"][0], CP["G_W4"][0] / se, WD_TOKEN["W4"]))
    P("  DESCRIPTIVE ladders (levels, not effects): scalar %s"
      % "  ".join("%s %.4f" % (WD_TOKEN[r], M[SCALAR_OF[r]]) for r in RUNG_IDS))
    P("                                            layerwise %s"
      % "  ".join("%s %.4f" % (WD_TOKEN[r], M[LAYER_OF[r]]) for r in RUNG_IDS))

    cw = carrier_state(M, states) if carrier_ok else "CAR-PATCH-NOT-VERIFIED"
    P("")
    P("THE CARRIER COMPANION at rung %s (wd %s), read ONLY if that rung is COLLAPSE" % (CARRIER_RUNG, WD_TOKEN[CARRIER_RUNG]))
    if not carrier_ok:
        P("  %s -- G-BITE's carrier gate failed; NO carrier number is read." % cw)
    elif states[RUNG_IDS.index(CARRIER_RUNG)] != "COLLAPSE":
        P("  %s -- rung %s is %s, so there is no collapse for the mask to remove.  %s %.4f is PRINTED as a level"
          % (cw, CARRIER_RUNG, states[RUNG_IDS.index(CARRIER_RUNG)], CAR, M[CAR]))
        P("  and NOTHING is read from it (cmo1's ISO-UNREADABLE at wd 0, CORRECTIONS 264).")
    else:
        k2, l2 = SCALAR_OF[CARRIER_RUNG], LAYER_OF[CARRIER_RUNG]
        CP["P_CARW2"] = (M[CAR] - M[k2], T[CAR] - T[k2])
        CP["D_CARW2"] = (M[l2] - M[CAR], T[l2] - T[CAR])
        P("  %-8s %-9s = %-7s - %-7s = %+.4f pp = %+.2f SE   (TRAIN %+.4f)   seeds %s"
          % ("KEY", "P_CARW2", CAR, k2, CP["P_CARW2"][0], CP["P_CARW2"][0] / se, CP["P_CARW2"][1],
             " / ".join("%+.3f" % (runs[(CAR, s)]["plateau5"] - runs[(k2, s)]["plateau5"]) for s in SEEDS)))
        P("  %-8s %-9s = %-7s - %-7s = %+.4f pp = %+.2f SE   (TRAIN %+.4f)"
          % ("KEY", "D_CARW2", l2, CAR, CP["D_CARW2"][0], CP["D_CARW2"][0] / se, CP["D_CARW2"][1]))
        P("  state bars: CAR-REC iff %s >= %.4f (%s - %.0f) ; CAR-NULL iff %s <= %.4f (%s + %.0f)  -> %s"
          % (CAR, M[l2] - MATCH_BAR, l2, MATCH_BAR, CAR, M[k2] + NULL_BAR, k2, NULL_BAR, cw))

    P("")
    P("G-DIVERGE (a branch condition, not a harness gate)")
    div = [a for a in ARMS if max(arm_v[a]) - min(arm_v[a]) > DIVERGED_BAR]
    P("  %s every arm's seed range <= %.1f pp%s" % ("PASS" if not div else "FAIL", DIVERGED_BAR,
                                                    "" if not div else "  diverged: %s" % div))

    branch, word = decide(M, bool(div), carrier_ok)
    stamps = ([word] if word else []) + [
        "HARNESS-CLEAN", "LADDER-BITES", "ALL-RUNGS-IN-BATCH", "BOTH-GRAINS-AT-EVERY-RUNG",
        "REFERENCE-IS-IN-BATCH-LAYERWISE", "CARRIER-AGAINST-KLW2", "COUPLED-DECAY-ONLY", "DECOUPLED-NOT-TESTED",
        "LADDER-IS-FOUR-POINTS", "ONE-NETWORK-RESNET18", "ONE-CELL-OTHERWISE", "HORIZON-100-ONLY", "THREE-SEEDS",
        "WD-IS-AN-ARGS-DEVIATION", "FLOOR-READINGS-ARE-BOUNDS",
        "SIGMA-PRIOR-FROZEN" if which == "SIGMA_PRIOR_frozen" else "SIGMA-INBATCH"]
    if div:
        stamps.append("DIVERGED-%s" % "-".join(div))
    if carrier_fail:
        stamps.append("CARRIER-PATCH-NOT-VERIFIED")
    else:
        stamps.append("MASK-UPDATE-AND-TRACE")
    if word and "UNREADABLE" not in (word or "") and branch in ACCOUNT_TOKENS:
        gg = [CP["G_" + r][0] for r in RUNG_IDS]
        if all(gg[i] >= gg[i + 1] - 2 * se for i in range(len(gg) - 1)):
            stamps.append("GAP-MONOTONE-IN-WD")
        else:
            stamps.append("GAP-NOT-MONOTONE-IN-WD")
    if abs(M[SCALAR_OF["W1"]] - K01_CWD3) > MATCH_BAR:
        stamps.append("ANCHOR-DIFFERS-FROM-CWD3")
    for a in ARMS:
        if any(nonfin[(a, s)] for s in SEEDS):
            stamps.append("NONFINITE-%s" % a)
    stamps.append(("HW-UNIFORM-%s" % hwset[0].replace(" ", "_")) if len(hwset) == 1 else "HW-MIXED")
    agree = all((t > 0) == (p > 0) for p, t in CP.values() if abs(p) >= GAP_BAR)
    stamps.append("TRAIN-AGREES" if agree else "TRAIN-DISAGREES")
    final = "FINAL: %s | %s" % (branch, " | ".join(stamps))
    P("")
    P(bar)
    P(final)
    P(bar)

    P("")
    P("BETWEEN-BATCH, NON-GATING (frozen literals; NO other batch enters any cwd5 bar, state or contrast):")
    for b_ in R_BETWEEN_ORDER:
        P("  %-48s %.4f" % (b_, R_BETWEEN[b_]))
    P("  cwd5 {%s}: %s" % (",".join(str(s) for s in SEEDS), "  ".join("%s %.4f" % (a, M[a]) for a in ARMS)))
    P("  k01W1 - cwd3's k01 = %+.4f pp  (the anchor's in-batch replication of the mechanism cell)"
      % (M[SCALAR_OF["W1"]] - K01_CWD3))

    P("")
    P("WHAT THIS DECIDES -- as registered, before any run existed:")
    for ln in LICENSE[branch]:
        P("  " + ln)
    P("")
    P("THE CARRIER HALF (rung %s only):" % CARRIER_RUNG)
    P("  %s -- %s" % (cw, CARRIER_LICENSE[cw]))

    P("")
    P("WHAT IT DOES NOT LICENSE, UNCONDITIONALLY:")
    P("  * DECOUPLED weight decay, in either direction.  The base optimiser has no decoupled path and no lambda")
    P("    matches the coupled dose wd*a with `a` learned; the arm was DESCOPED WITH REASONS at 281.2, not run.")
    P("  * Any weight-decay threshold finer than the bracketing PAIR of rungs: the ladder has FOUR points.")
    P("  * Any other network (PlainNet / VGG / GroupNorm / ResNet10 / 34 / 50), dataset, meta step size, alpha0,")
    P("    momentum, horizon beyond 100 epochs, or grouping other than scalar and layerwise.")
    P("  * The carrier account at any rung other than %s, or the ROUTE the decay acts through (weight shrink vs" % CARRIER_RUNG)
    P("    meta trace -- one flag changes both).")
    P("  * Anything about the corpus's landed cells: every cwd5 reading is WITHIN this batch.  The 21 runs at a")
    P("    non-standard weight decay are ARGS-value deviations and owe CORPUS-EXCLUSIONS rows of kind ARGS_WD_BASE")
    P("    at the landing (CORRECTIONS 263's kind; corpus_exclusions.py is NOT edited).")

    X = {"M": M, "T": T, "branch": branch, "word": word, "final": final, "runs": runs, "se": se, "sigma_in": sigma_in,
         "CP": CP, "arm_v": arm_v, "arm_t": arm_t, "div": div, "stamps": stamps, "hwset": hwset, "states": states,
         "cw": cw, "manifest": mt, "df": df, "which": which, "nonfin": nonfin, "incomplete": None,
         "ladder_fail": ladder_fail, "carrier_fail": carrier_fail}
    return L, X


# ---------------------------------------------------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: cwd5_attack_indep.py <runsdir> <scorer-log> [--csv P] [--tsv P]")
        return 2
    runsdir, logpath = args[0], args[1]
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    opt = {"--csv": os.path.join(repo, "results", "all_runs.csv"),
           "--tsv": os.path.join(repo, "results", "CORPUS-EXCLUSIONS.tsv")}
    i = 2
    while i < len(args):
        if args[i] in opt and i + 1 < len(args):
            opt[args[i]] = args[i + 1]
            i += 2
        else:
            i += 1

    got = [ln.rstrip("\n") for ln in open(logpath, errors="replace")]
    pp = os.path.join(runsdir, PREFIX, "PROVENANCE.txt")
    prov_file = read_kv(pp) if os.path.exists(pp) else {}

    rebuilt, X = rebuild(runsdir)

    bar = "=" * 78
    print(bar)
    print(" cwd5_attack_indep -- AN INDEPENDENT RE-DERIVATION OF THE REGISTERED cwd5 SCORER'S OUTPUT")
    print(" CORRECTIONS 281.  No repo module imported, no regular expression, every literal re-typed.")
    print(bar)

    v = []

    def ck(cond, label, extra=""):
        print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
        if not cond:
            v.append(label)

    print("")
    print("[1] WHAT WAS REBUILT FROM SCRATCH, BEFORE THE LOG WAS OPENED")
    ck(len(TENSORS) == NTENS and sum(x[1] for x in TENSORS) == TOTPAR,
       "ResNet18_c100 rebuilt from the architecture",
       "%d tensors, %d params" % (len(TENSORS), sum(x[1] for x in TENSORS)))
    ns = dm_indices("normscale")
    ck(tuple(i + 1 for i in ns) == REG_NORMSCALE_IDX and sum(TENSORS[i][1] for i in ns) == NORMSCALE_NUMEL,
       "the 20 normscale tensors resolved by ndim==1 and name.endswith('.weight')",
       "%d tensors, numel %d" % (len(ns), sum(TENSORS[i][1] for i in ns)))
    idx = dm_indices(DMASK[CAR])
    ck(tuple(i + 1 for i in idx) == REG_SET_IDX[CAR] and sum(TENSORS[i][1] for i in idx) == REG_SET_NUMEL[CAR]
       and len(idx) == K_MASKED[CAR],
       "CARW2's name list resolves to the registered set",
       "idx %s numel %d k %d" % (",".join("%d" % (i + 1) for i in idx), sum(TENSORS[i][1] for i in idx), len(idx)))
    w512 = tuple(i + 1 for i in ns if TENSORS[i][1] == 512)
    ck(w512 == W512_IDX, "every 512-wide BN scale in the model is one of {47,50,53,56,59} (188.1 / 275.2)",
       "512-wide scales %s" % (w512,))
    ck(all(K_MASKED[a] == 0 for a in LADDER_ARMS) and len(LADDER_ARMS) == 8,
       "ALL EIGHT LADDER ARMS ARE UNMASKED -- so no patch defect can forge a threshold (281's G-BITE split)",
       "%d ladder arms, %d masked arms" % (len(LADDER_ARMS), len(MASKED)))
    ck(sorted(set(SPEC[a] for a in ARMS)) == ["layerwise", "scalar"]
       and all(SPEC[SCALAR_OF[r]] == "scalar" and SPEC[LAYER_OF[r]] == "layerwise" for r in RUNG_IDS),
       "BOTH GRAINS EXIST AT EVERY RUNG -- every gap is a WITHIN-rung contrast, never a cross-batch one",
       "%d scalar / %d layerwise arms" % (sum(1 for a in ARMS if SPEC[a] == "scalar"),
                                          sum(1 for a in ARMS if SPEC[a] == "layerwise")))
    ck(len(set(WD_TOKEN[r] for r in RUNG_IDS)) == 4
       and sorted(WD_VALUE[r] for r in RUNG_IDS) == [0.0005, 0.001, 0.01, 0.1],
       "the four rung tokens are PAIRWISE DISTINCT and span 200x (0.1 -> 5e-4)",
       "%s" % ",".join("%s=%s" % (r, WD_TOKEN[r]) for r in RUNG_IDS))
    ck(abs(f32(math.log(1e-6)) - (-13.815510749816895)) == 0.0,
       "alpha0 1e-6 -> beta0 re-derived through float32", "%r" % f32(math.log(1e-6)))

    print("")
    print("[1b] THE RUNG SEPARATION, READ OFF THE RUNS' OWN ARGS LINES")
    print("     This is the ONLY in-batch readout that tells the four rungs apart.  DECLARED, not hidden: the")
    print("     runner's ENV line carries no weight decay, and PROBE_TENSOR carries only the grain, so neither can")
    print("     discriminate a rung.  The bite job's RW2 is what proves, on the real GPU path, that the flag")
    print("     CHANGES THE UPDATE at each of the four values -- this parser checks only that each run carried its own.")
    cover, wrong = {}, []
    for s in SEEDS:
        for a in ARMS:
            r = X["runs"].get((a, s))
            if r is None or len(r["args"]) != 1:
                wrong.append("%s-s%d (no ARGS)" % (a, s))
                continue
            tok = parse_args_line(r["args"][0]).get("weight-decay-base", [None])[-1]
            gr = parse_args_line(r["args"][0]).get("stepsize-groups", [None])[-1]
            cover[tok] = cover.get(tok, 0) + 1
            if tok != WD[a] or gr != SPEC[a]:
                wrong.append("%s-s%d carries wd=%s grain=%s, wants wd=%s grain=%s" % (a, s, tok, gr, WD[a], SPEC[a]))
    for r in RUNG_IDS:
        print("      rung %s  --weight-decay-base %-5s  %d runs  (%s scalar + %s layerwise%s)"
              % (r, WD_TOKEN[r], cover.get(WD_TOKEN[r], 0), len(SEEDS), len(SEEDS),
                 " + %d %s" % (len(SEEDS), CAR) if r == CARRIER_RUNG else ""))
    ck(not wrong, "every one of the %d runs carries ITS OWN rung's weight decay AND its own grain"
       % (len(ARMS) * len(SEEDS)), "; ".join(wrong[:4]) if wrong else "0 wrong")
    want_cov = {}
    for a in ARMS:
        want_cov[WD[a]] = want_cov.get(WD[a], 0) + len(SEEDS)
    ck(cover == want_cov, "the axis coverage is exactly as registered",
       " ".join("%s x%d" % (k, cover.get(k, 0)) for k in (WD_TOKEN[r] for r in RUNG_IDS)))

    print("")
    print("[1c] THE WITNESS, REBUILT FROM THE ARCHITECTURE *AND FROM ITS OWN RUNG'S WEIGHT DECAY*")
    print("     CARW2 runs at 1e-2, so its witness must say wd=0.01 -- a string PATCH_DECAYMASK had never printed")
    print("     before this batch.  The `wd=` field is what separates a carrier mask at 1e-2 from the same mask at")
    print("     the anchor, where cwd3 and cwd4 already ran it.")
    wbad, swaps = [], []
    for s in SEEDS:
        for a in ARMS:
            r = X["runs"].get((a, s))
            if r is None or r["DECAY_MASK"] != [WITNESS_DM[a]]:
                wbad.append("%s-s%d" % (a, s))
    ck(not wbad, "all %d runs printed exactly the witness this parser rebuilt" % (len(ARMS) * len(SEEDS)),
       "bad: %s" % wbad if wbad else "%d/%d" % (len(ARMS) * len(SEEDS), len(ARMS) * len(SEEDS)))
    anchor_witness = dm_witness(DMASK[CAR], WD_VALUE[ANCHOR_RUNG])
    for s in SEEDS:
        r = X["runs"].get((CAR, s))
        if r is not None and r["DECAY_MASK"] == [anchor_witness]:
            swaps.append("%s-s%d carries the ANCHOR's wd=0.1 witness" % (CAR, s))
    ck(not swaps and WITNESS_DM[CAR] != anchor_witness,
       "the carrier witness at 1e-2 DIFFERS from the same mask at 0.1, and no run carries the anchor's",
       "%d swaps" % len(swaps))
    print("      %-7s %s" % (CAR, WITNESS_DM[CAR]))
    print("      (the same mask at the anchor would read `wd=0.1`, and no cwd5 run printed that)")
    n_masked_runs = n_masked_rec = n_pos = n_plain_runs = n_plain_rec = n_dmkey_on_plain = 0
    for s in SEEDS:
        for a in ARMS:
            recs = load_probe(runsdir, a, s) or []
            hit = sum(1 for r in recs if isinstance(r, dict) and "dm_wdterm" in r and r["dm_wdterm"] > 0)
            if a in MASKED:
                n_masked_runs += 1
                n_masked_rec += len(recs)
                n_pos += hit
            else:
                n_plain_runs += 1
                n_plain_rec += len(recs)
                n_dmkey_on_plain += sum(1 for r in recs if isinstance(r, dict)
                                        and any(str(kk).startswith("dm_") for kk in r))
    print("      DENOMINATOR (281.10, 274's lesson): %d masked runs x %d records = %d MASKED records, %d with a"
          % (n_masked_runs, n_masked_rec // max(n_masked_runs, 1), n_masked_rec, n_pos))
    print("      positive dm_wdterm.  The %d UNMASKED runs contribute %d records and carry NO dm_* key (%d found):"
          % (n_plain_runs, n_plain_rec, n_dmkey_on_plain))
    print("      they are NOT part of this denominator.")
    ck(n_pos == n_masked_rec and n_dmkey_on_plain == 0,
       "every masked record carries a positive dm_wdterm and no unmasked record carries a dm_* key",
       "%d/%d masked, %d dm_* keys on the ladder" % (n_pos, n_masked_rec, n_dmkey_on_plain))

    print("")
    print("[1d] THE MANIFEST AND THE PROVENANCE, REBUILT")
    mp = os.path.join(runsdir, PREFIX, "PARTITION-MANIFEST.txt")
    livetxt = open(mp).read() if os.path.exists(mp) else ""
    ck(livetxt == X["manifest"], "the batch's PARTITION-MANIFEST.txt == the manifest rebuilt here, byte for byte",
       "%d bytes, sha256 %s" % (len(X["manifest"]), hashlib.sha256(X["manifest"].encode()).hexdigest()[:16]))
    badprov = [k for k in sorted(PROV_WANT) if prov_file.get(k) != PROV_WANT[k]]
    ck(not badprov, "every PROVENANCE.txt value equals the literal RE-TYPED here (281.5 / 271.2), scorer sha included",
       "bad: %s" % badprov if badprov else "%d keys" % len(PROV_WANT))

    if X.get("incomplete"):
        print("")
        print("INCOMPLETE: %s" % X["incomplete"])
        return 1

    print("")
    print("[2] THE NUMBERS, RE-DERIVED FROM THE RAW RECORDS ONLY")
    for a in ARMS:
        print("  %-7s wd %-5s %-9s TEST %.4f  TRAIN %.4f  seeds %s"
              % (a, WD[a], SPEC[a], X["M"][a], X["T"][a], " / ".join("%.4f" % x for x in X["arm_v"][a])))
    print("  SIGMA_INBATCH %.6f   SIGMA_USED %s   SE_ARM_DIFF %.6f   df %d"
          % (X["sigma_in"], X["which"], X["se"], X["df"]))
    for r, st in zip(RUNG_IDS, X["states"]):
        g = X["CP"]["G_" + r][0]
        print("  %-3s wd %-5s  G_%s %+.4f pp = %+.2f SE   R_%s %.4f   -> %-9s (COLLAPSE bar scalar <= %.4f)"
              % (r, WD_TOKEN[r], r, g, g / X["se"], r, X["M"][SCALAR_OF[r]] / X["M"][LAYER_OF[r]], st,
                 R50 * X["M"][LAYER_OF[r]]))
    print("  carrier word  %s" % X["cw"])
    print("  branch        %s" % X["branch"])

    print("")
    print("[3] LINE-BY-LINE AGREEMENT WITH THE SCORER LOG")
    n_lines = min(len(rebuilt), len(got))
    ck(len(rebuilt) == len(got), "the rebuild has the scorer log's line count",
       "%d rebuilt vs %d in the log" % (len(rebuilt), len(got)))
    verbatim = host = mism = 0
    first = []
    for i in range(n_lines):
        mine, theirs = rebuilt[i], got[i]
        if mine == "@CORPUS@":
            want_n = corpus_filtered_count(opt["--csv"], opt["--tsv"])
            okc = want_n is not None and theirs == (
                "corpus: %d rows after corpus_exclusions.filter_rows, %s- excluded  "
                "(DISCLOSURE ONLY -- no bar, sigma, branch or stamp reads it)" % (want_n, PREFIX))
            host += 1 if okc else 0
            if not okc:
                mism += 1
                first.append(i + 1)
            continue
        if mine == "@MANIFESTPATH@":
            ok = theirs.startswith("  manifest: ") and theirs.endswith("/cwd5/PARTITION-MANIFEST.txt (<runsdir>/cwd5/)")
            host += 1 if ok else 0
            if not ok:
                mism += 1
                first.append(i + 1)
            continue
        if mine == "@PROVPATH@":
            ok = theirs.startswith("  provenance: ") and theirs.endswith("/cwd5/PROVENANCE.txt (<runsdir>/cwd5/)")
            host += 1 if ok else 0
            if not ok:
                mism += 1
                first.append(i + 1)
            continue
        if mine == theirs:
            verbatim += 1
        else:
            mism += 1
            first.append(i + 1)
    print("  lines in the scorer log            %d" % len(got))
    print("  rebuilt and matched VERBATIM       %d" % verbatim)
    print("  matched host-independently          %d   (2 path disclosures by suffix; 1 corpus count re-filtered here)" % host)
    print("  MISMATCHES                          %d%s" % (mism, ("   first at lines %s" % first[:10]) if first else ""))
    if mism:
        for i in first[:5]:
            print("    line %d" % i)
            print("      log      %r" % got[i - 1])
            print("      rebuilt  %r" % rebuilt[i - 1])
        v.append("line-by-line agreement")

    print("")
    print("[4] THE FINAL LINE")
    logfinal = [ln for ln in got if ln.startswith("FINAL:")]
    ck(len(logfinal) == 1 and logfinal[0] == X["final"], "the FINAL line is rebuilt token by token")
    print("  %s" % X["final"])

    print("")
    print("[5] THE CARRIER-READABILITY AUDIT (all 4^4 = 256 rung-state vectors, re-derived)")
    print("     The carrier companion sits at rung W2 ALONE and may be void: if W2 does not COLLAPSE there is no")
    print("     collapse for the mask to remove.  CAR-UNREADABLE was pre-registered for exactly that case (281.3).")
    print("     This enumerates which accounts are reachable and which of them can carry a READ carrier word.")
    words = ("COLLAPSE", "NOGAP", "PARTIAL", "UNREADABLE")
    reach = {}
    for a1 in words:
        for a2 in words:
            for a3 in words:
                for a4 in words:
                    st = (a1, a2, a3, a4)
                    if "UNREADABLE" in st:
                        continue                       # UNRESOLVED-REFERENCE gates before any account
                    if st[0] != "COLLAPSE":
                        continue                       # ANCHOR-NOT-COLLAPSED gates before any account
                    tok = account_of(list(st))
                    reach.setdefault(tok, set()).add(st[RUNG_IDS.index(CARRIER_RUNG)] == "COLLAPSE")
    for tok in sorted(reach):
        cws = sorted("READ" if x else "CAR-UNREADABLE" for x in reach[tok])
        print("      %-27s carrier word reachable there: %s" % (tok, ",".join(cws)))
    print("      THIS batch: rung %s is %s, so the carrier word is %s"
          % (CARRIER_RUNG, X["states"][RUNG_IDS.index(CARRIER_RUNG)], X["cw"]))
    ck((X["cw"] == "CAR-UNREADABLE") == (X["states"][RUNG_IDS.index(CARRIER_RUNG)] != "COLLAPSE"),
       "the carrier word is CAR-UNREADABLE exactly when rung W2 is not COLLAPSE")
    if X["cw"] == "CAR-UNREADABLE":
        print("      QUOTE-BLOCK: NO CAR-REC / CAR-PART / CAR-NULL sentence may be written, and CARW2 %.4f is a"
              % X["M"][CAR])
        print("      LEVEL ONLY.  P_CARW2 and D_CARW2 were not computed by the scorer and are not computed here.")
    else:
        print("      QUOTE-OK: rung %s COLLAPSED, so the carrier contrasts are defined and %s is quotable."
              % (CARRIER_RUNG, X["cw"]))

    print("")
    print("[6] WHICH PRINTED READINGS ARE BOUNDS RATHER THAN POINT EFFECTS (164.6)")
    for r, st in zip(RUNG_IDS, X["states"]):
        g = X["CP"]["G_" + r][0]
        if st == "COLLAPSE":
            print("      %s (wd %-5s) COLLAPSE: G_%s = %+.4f pp is a BOUND -- the scalar arm sits at its floor, so the"
                  % (r, WD_TOKEN[r], r, g))
            print("      gap's SIZE is a location, not a measured effect.")
        elif st == "NOGAP":
            print("      %s (wd %-5s) NOGAP: G_%s = %+.4f pp is a BOUND in the other direction -- the reading is"
                  % (r, WD_TOKEN[r], r, g))
            print("      'below the %.0f pp bar', not 'the two grains are equal'." % GAP_BAR)
        else:
            print("      %s (wd %-5s) %s: G_%s = %+.4f pp is read against the bars, DESCRIPTIVELY."
                  % (r, WD_TOKEN[r], st, r, g))
    print("      The ladder has FOUR points: nothing between two adjacent rungs was run, so the threshold this")
    print("      batch reports is a BRACKETING PAIR and never a value.")
    print("      The DECOUPLED arm was never run (281.2): no sentence about it is licensed in either direction.")

    print("")
    print("[7] RULE 16 DEFECTS FOUND IN THE REGISTERED SCORER BY THIS REPLAY (REPORTED, NOT FIXED)")
    print("     F1 (cwd5, NEW).  analysis/cWD5_wdladder_score.py guards the DESCRIPTIVE gap-monotonicity stamp")
    print("     with `if word and \'UNREADABLE\' not in (word or \'\') and branch in ACCOUNT_TOKENS:`.  The")
    print("     substring test was plainly meant to catch a RUNG state of UNREADABLE -- but a rung is UNREADABLE")
    print("     exactly when its layerwise arm is below REF_MIN, and decide() returns UNRESOLVED-REFERENCE in that")
    print("     case, which is NOT in ACCOUNT_TOKENS.  So the rung reading of the test is DEAD, and the only string")
    print("     that can trip it is the CARRIER word CAR-UNREADABLE.  CONSEQUENCE, MEASURED ON THIS BATCH: the four")
    print("     gaps are monotone non-increasing in the weight decay within 2 SE, so GAP-MONOTONE-IN-WD would have")
    print("     fired -- and it is SUPPRESSED, because rung W2 did not collapse and the carrier companion is void.")
    print("     Gap monotonicity and carrier readability are unrelated facts, and one is silencing the other.")
    print("     COSMETIC IN SCOPE: the clause appends a DESCRIPTIVE stamp only.  It touches no bar, sigma, level,")
    print("     contrast, rung state, carrier word, branch or licence paragraph, and the FINAL's account token is")
    print("     unchanged either way.  The registered file is NOT edited (RULE 16); this parser RE-TYPES the clause")
    print("     as written and reproduces the bytes the scorer actually emits, so the byte-identical claim above is")
    print("     honest rather than flattering.  A landing that wants to state the monotonicity must re-derive it")
    print("     from the four G values (this section does) and must NOT quote a stamp the scorer did not print.")

    print("")
    print("VIOLATIONS %d" % len(v))
    return 0 if not v else 1


if __name__ == "__main__":
    raise SystemExit(main())
