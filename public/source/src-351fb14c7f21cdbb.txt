#!/usr/bin/env python3
# =============================================================================
# cwd4_attack_indep.py -- THE INDEPENDENT PARSER OF `cwd4` (CORRECTIONS 280, landing entry).
#
# It re-derives, FROM THE RAW RECORDS ALONE, every line the REGISTERED scorer
# analysis/cWD4_countwd_score.py prints, and compares its rebuild with a committed scorer log, line by line, in order.
#
#   python3 analysis/cwd4_attack_indep.py <runsdir> <scorer-log> [--csv P] [--tsv P]
#
# INDEPENDENCE, deliberately (the cwd2 / cwd3 pattern, CORRECTIONS 261 / 275, carried to cwd4):
#   * it imports NOTHING from cWD4_countwd_score.py, cwd4_design.py, cwd_design.py, cwd_common.py,
#     corpus_exclusions.py, argsline_guard.py or any other repo module.  Stdlib only: hashlib, json, math, os,
#     struct, sys.  `csv` is NOT used either -- the corpus is split by hand.
#   * NO regular expressions anywhere.  The `.out` lines are split on commas / colons / whitespace, the TSV on tabs,
#     the file names on hyphens.
#   * every bar, sigma literal, witness string, tree sha, licence sentence and stamp name below is RE-TYPED from
#     CORRECTIONS 280, not read from any module.
#   * the ResNet18_c100 tensor table is rebuilt FROM THE ARCHITECTURE (stem, four stages of two BasicBlocks, the three
#     downsample shortcuts, the linear head), not read from a manifest, and the six masked sets are resolved against
#     that rebuild by NAME.
#   * float32 rounding is re-derived through `struct`, and the Lion natural step is recomputed from the records' own
#     beta_pre / mom_pre / z_agg, so G-BITE's every printed counter is independent of the scorer's implementation.
#
# HOST INDEPENDENCE.  Three kinds of line in the scorer's output are host-dependent and are matched, not rebuilt byte
# for byte: (1) the two path disclosures (`manifest:` / `provenance:`), matched by their host-independent suffix;
# (2) the corpus disclosure, whose row count depends on which corpus commit the host's tree carries -- this parser
# re-filters the corpus IT IS GIVEN and requires the log's count to equal its own, WITHOUT printing either number, so
# that THIS parser's own stdout is byte-identical on both hosts; (3) the G-PROV `SCORER_SHA256` row, whose printed
# prefix is the scorer file's own sha -- re-typed here from CORRECTIONS 280.5, so it is rebuilt, not waived.
#
# WHAT IT ADDS BEYOND A REPLAY (the five independent attacks):
#   [1]  the design rebuilt from the architecture: 62 tensors / 11,220,132 params, the 20 normscale indices, and each
#        arm's masked set resolved by name -> the registered idx / numel.  It also re-derives, from the rebuild alone,
#        that the 512-wide BN scales are EXACTLY {47,50,53,56,59} and that the carrier-free ones are EXACTLY {47,56}
#        -- the structural fact the whole count-matched design rests on.
#   [1b] the CROSS-READ: every run's probe records are read as its OWN arm's k AND as every OTHER k in the batch.
#        This separates CORRECTIONS 280.4's null (i) -- a mask that bit on CARWD0 but on NO subset arm, which predicts
#        EXACTLY ALL-THREE-NEEDED -- from a mask that really bit.  ITS DECLARED LIMIT IS RE-DERIVED HERE, NOT HIDDEN:
#        k=2 is shared by TWOWD0 and CTL2WD0 and k=1 by the three singles, so the cross-read CANNOT refuse a swap
#        within those groups.  [1c] is what refuses it.
#   [1c] the WITNESS re-derivation: each arm's `DECAY_MASK: on ...` line rebuilt from the architecture table and
#        required to equal, byte for byte, the line the run's own `.out` printed -- idx and names included, which is
#        the ONLY readout that tells the two k=2 arms and the three k=1 arms apart.
#   [2]  every level, sigma, contrast, state and stamp recomputed from the raw `.out` epoch lines.
#   [3]  line-by-line agreement with the committed scorer log.
#   [4]  the FINAL line rebuilt token by token.
#   [5]  THE REACHABILITY AUDIT OF THE FLOOR CLAUSE (278.6 D1's cwd4 analogue): over all 3^5 = 243 state combinations,
#        which accounts are reachable with CTL2WD0 in state PART -- i.e. with the carrier-free pair NOT at k01's
#        floor -- so that a licence paragraph asserting the pair "stays at k01" is never quoted when it did not.
# =============================================================================

import hashlib
import json
import math
import os
import struct
import sys

# ---------------------------------------------------------------------------------------------------------------------
# RE-TYPED LITERALS (CORRECTIONS 280; the cell from 260 / 271, the ENV line from cvt1...cvt9 / cwd1 / cwd3)
# ---------------------------------------------------------------------------------------------------------------------
PREFIX = "cwd4"
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
WD_BASE = 0.1
NTENS = 62
TOTPAR = 11220132
SEEDS = (143, 144, 145)
ARMS = ("k01", "CARWD0", "TWOWD0", "CTL2WD0", "ONE50", "ONE53", "ONE59")
MASKED = ("CARWD0", "TWOWD0", "CTL2WD0", "ONE50", "ONE53", "ONE59")
SINGLES = ("ONE50", "ONE53", "ONE59")
STATEARMS = ("TWOWD0", "CTL2WD0", "ONE50", "ONE53", "ONE59")
REF = "CARWD0"

C50 = "layer4.0.bn2.weight"            # idx 50  carrier  Kim gamma_last(layer4.0)
C53 = "layer4.0.shortcut.1.weight"     # idx 53  carrier  Kim gamma_down(layer4.0)
C59 = "layer4.1.bn2.weight"            # idx 59  carrier  Kim gamma_last(layer4.1)
N47 = "layer4.0.bn1.weight"            # idx 47  NON-carrier  Kim gamma_others(layer4.0)
N56 = "layer4.1.bn1.weight"            # idx 56  NON-carrier  Kim gamma_others(layer4.1)
CARRIERS = (C50, C53, C59)
TWO_CAR = (C50, C53)
CTL_DEPTH2 = (N47, N56)
DMASK = {"k01": "", "CARWD0": "+".join(CARRIERS), "TWOWD0": "+".join(TWO_CAR), "CTL2WD0": "+".join(CTL_DEPTH2),
         "ONE50": C50, "ONE53": C53, "ONE59": C59}
SPEC = dict((a, "scalar") for a in ARMS)
PT_TYPE = dict((a, "scalar") for a in ARMS)
K_MASKED = {"k01": 0, "CARWD0": 3, "TWOWD0": 2, "CTL2WD0": 2, "ONE50": 1, "ONE53": 1, "ONE59": 1}
NG = dict((a, 1) for a in ARMS)                           # every arm is SCALAR: exactly one step-size group
REG_SET_IDX = {"CARWD0": (50, 53, 59), "TWOWD0": (50, 53), "CTL2WD0": (47, 56),
               "ONE50": (50,), "ONE53": (53,), "ONE59": (59,)}
REG_SET_NUMEL = {"CARWD0": 1536, "TWOWD0": 1024, "CTL2WD0": 1024, "ONE50": 512, "ONE53": 512, "ONE59": 512}
REG_NORMSCALE_IDX = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59)
NORMSCALE_NUMEL = 4800
KIM_CLASS = {50: "gamma_last", 53: "gamma_down", 59: "gamma_last", 47: "gamma_others", 56: "gamma_others"}

# the runner's ENV line (PROBE_DIR stripped), cvt1 ... cvt9's / cwd1's / cwd3's -- RE-TYPED
ENV_EXPECTED = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na "
                "COS_TOTAL=default COS_WARMUP=default SCHED=none SCHED_TOTAL=none "
                "SCHED_WARMUP=none SCHED_MIN=none PROBE=100 EB_RHO=na EB_LOG=0")
OFF_WITNESS = {"VOTE_W": "VOTE_W: off", "BETA_HOLD": "BETA_HOLD: off", "GROUP_HOLD": "GROUP_HOLD: off",
               "REST_HOLD": "REST_HOLD: off"}
KINDS = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD", "COMP_HOLD", "WINDOW_HOLD", "DECAY_MASK")
DM_KEYS = ("dm_n", "dm_masked", "dm_skipped", "dm_wdterm", "dm_norm", "dm_absmin", "dm_small")
HOLD_PREFIXES = ("bh_", "gh_", "rh_", "ch_", "wh_")

# frozen bars (CORRECTIONS 280.4), RE-TYPED
SIGMA_R18ALL = 0.6451413439065666
SIGMA_R18ALL_NAIVE = 12.482176983761198
SIGMA_PRIOR = 0.6451413439065666
GAP_MIN = 20.0
K01_MAX = 30.0
RESCUE_BAR = 10.0
NULL_BAR = 2.0
MATCH_BAR = 5.0
DIVERGED_BAR = 5.0
FLOOR_MIN = 15.0
CEIL_MAX = 90.0
CHANCE = 1.0
BH_TOL = 1e-5
# cwd3's landed levels (CORRECTIONS 278.3): BETWEEN-BATCH, NON-GATING, disclosure and two stamps only
CWD3_LEVELS = (("k01", 22.9853), ("CARWD0", 70.2640), ("CTLWD0", 23.0013), ("CTL2WD0", 22.9333), ("NWD", 70.7227))
CAR_CWD3 = 70.2640
CTL2_CWD3 = 22.9333
K01_CWD3 = 22.9853

# Lion, RE-TYPED (the meta optimiser of the cell: ms 1e-3, beta2 0.9, clip -15 .. -2.3026)
MS = 1e-3
B2 = 0.9
LO, HI = -15.0, -2.3026
TIE_REL = 1e-12

# provenance, RE-TYPED (CORRECTIONS 280.5; the tree / runner / build_network shas from 271.2 / cwd1's registration)
BN_SHA = "c79988833c4399a06cc84d15c0830f54d86f59feeec345f47a2287c1cf2ba77d"        # cvt8's build_network.py
HF_POST_SHA = "94aedc33046afb12782ffb7f2eb729bafc7fcba5f42227fec373fa67b6f58ebf"   # cvt8's HF.py + PATCH_DECAYMASK
HF_PARENT_SHA = "5197dc2eecbbcb6feb815f3798b3455171e6ad614deac913be5df48dd9ede1a9"  # cvt8's HF.py
RUNNER_SHA = "34a8c90ee3a0bee00aa2d0b4a3ad5d16d461fbf5a1f0cda5797e44e44c158371"    # jobs/run_cifar_cwd1.sh
SCORER_SHA = "475e1278bce0eb0eb61b78b7491237466b70f03f9fb1d1a6f67029c0bc799fbc"    # analysis/cWD4_countwd_score.py
DESIGN_SHA = "a6f2c8754524a56e8e1117882348fb5b159c19ee07bb4d9143038367f6155610"    # analysis/cwd4_design.py (280.5)
CWD_DESIGN_SHA = "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195"
COMMON_SHA = "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"
PROV_WANT = {"MODE": "submit", "BUILD_NETWORK_SHA256": BN_SHA, "HF_SHA256": HF_POST_SHA,
             "HF_PRE_DECAYMASK_SHA256": HF_PARENT_SHA, "RUNNER_SHA256": RUNNER_SHA, "SCORER_SHA256": SCORER_SHA,
             "DESIGN_SHA256": DESIGN_SHA, "CWD_DESIGN_SHA256": CWD_DESIGN_SHA, "COMMON_SHA256": COMMON_SHA}

# ---------------------------------------------------------------------------------------------------------------------
# THE REGISTERED LICENCE PROSE, RE-TYPED FROM analysis/cWD4_countwd_score.py's LICENSE dict (CORRECTIONS 280.4 / the
# registration commits c0abca1 / 209d9eb).  Re-typed, not imported: the bars inside the sentences are this file's own
# literals.
# ---------------------------------------------------------------------------------------------------------------------
READING = ("the carrier account of the scalar collapse at the mechanism cell, and specifically cwd3's "
           "CARRIER-DECAY-SUFFICES (CORRECTIONS 278, MASTER-TABLE row 236)")
NOT_ROUTE = "Not licensed: which ROUTE acts (weight shrink vs meta trace -- the mask changes both), decoupled WD, other cells."
FLOOR_NOTE = "A NULL state is a location at k01's floor (164.6), a BOUND on the effect, not a measured zero."
NOT_MAG = ("Not licensed either way: IDENTITY vs TERM MAGNITUDE.  On this network no carrier-free control can be "
           "magnitude-matched.  Mean |L| on the scalar anchor's PINNED records (SCORE-cdep1 [P]): the three carriers "
           "are ranks 1/2/3 of 62 (59 3.2984e-01, 50 2.9155e-01, 53 1.9526e-01); the ONLY carrier-free 512-wide "
           "layer4 BN scales are ranks 33 and 36 (56 1.5166e-03, 47 1.1298e-03).  The SMALLEST carrier exceeds the "
           "LARGEST admissible control tensor by 128.8x, and SCORE-cdep1's registered sum ratio is 308.4.  cmg1's "
           "MAGNITUDE-NOT-SEPARATED stands: this batch matches COUNT, class, width, depth and numel, never magnitude.")
NOT_KIM = ("Not licensed: ctd1 IDENTITY vs Kim et al.'s POSITION CLASS (arXiv:2205.07260) -- {50,59} are gamma_last, "
           "53 is gamma_down, {47,56} are gamma_others, and no carrier-free gamma_last/gamma_down exists at this depth.")
LICENSE = {
    "ANY-ONE-SUFFICES": [
        "Each of the three carrier scales, masked ALONE, reaches within %.0f pp of the three-carrier mask, while the" % MATCH_BAR,
        "class-pure count-matched carrier-free pair {47,56} stays at k01.  SENTENCE LICENSED, at this cell: 'removing the",
        "coupled weight decay from ANY ONE of the three carrier BatchNorm scales alone removes the collapse; the effect is",
        "NOT graded in the NUMBER of masked scales, and removing it from two matched non-carrier scales does not lift the",
        "run off the floor.'  EFFECT ON " + READING + ": 278's COUNT / DOSE rival (T19) is REFUTED at this cell -- one",
        "genuine 512-wide layer4 BN scale suffices if it is a carrier, and two do not if they are not.  " + FLOOR_NOTE,
        NOT_MAG, NOT_KIM, NOT_ROUTE],
    "ONE-SUFFICES-PARTIAL": [
        "At least one carrier scale masked ALONE reaches within %.0f pp of the three-carrier mask, but NOT all three do" % MATCH_BAR,
        "(read the state word and the per-single contrasts).  SENTENCE LICENSED: 'at this cell one named carrier scale's",
        "own decay suffices, and the effect is carrier-SPECIFIC WITHIN the three carriers -- it is not a function of the",
        "number of masked scales alone.'  EFFECT ON " + READING + ": 278's COUNT / DOSE rival (T19) is REFUTED at this",
        "cell, and a new within-carrier asymmetry is opened.  WHICH single sufficed is read from the state word only;",
        "any ordering among the singles is DESCRIPTIVE (D_MAG / D_CLASS) and is a BOUND wherever they saturate.",
        FLOOR_NOTE, NOT_MAG, NOT_KIM, NOT_ROUTE],
    "TWO-SUFFICES": [
        "No single carrier scale lifts the run, the carrier PAIR {50,53} reaches within %.0f pp of the three-carrier" % MATCH_BAR,
        "mask, and the COUNT-MATCHED, class-pure, width-, depth- and numel-matched carrier-free pair {47,56} stays at",
        "k01.  SENTENCE LICENSED, at this cell: 'two of the three carrier scales suffice and one does not, while two",
        "matched NON-carrier 512-wide layer4 BN scales do not lift the run off the floor: at count two the effect is",
        "specific to the carriers.'  EFFECT ON " + READING + ": this is the contrast 278 could not build at count three",
        "-- the COUNT / DOSE rival (T19) is REFUTED AT COUNT TWO, and the dose ladder is resolved as 1 < 2 = 3.",
        FLOOR_NOTE, NOT_MAG, NOT_KIM, NOT_ROUTE],
    "COUNT-GRADED": [
        "No single lifts the run and the carrier pair reaches only PART of the way to the three-carrier mask: the effect",
        "GROWS WITH THE NUMBER of masked carrier scales and is not complete at two.  SENTENCE LICENSED: 'the rescue is",
        "graded in the number of carrier scales exempted from the decay; all three are needed for the full effect.'",
        "EFFECT ON " + READING + ": 278's COUNT / DOSE rival (T19) is SUPPORTED on the carrier side.  Whether the count",
        "or the identity is what matters is then decided by CTL2WD0's state: with CTL2WD0 at k01 the count account still",
        "needs the scales to be carriers (P_2SPEC), and that pair of facts must be written together.",
        FLOOR_NOTE, NOT_MAG, NOT_KIM, NOT_ROUTE],
    "PARTIAL-NON-MONOTONE": [
        "A single carrier scale reaches PART of the way while the PAIR containing carriers does not lift the run at all:",
        "the dose ladder is NON-MONOTONE.  NO count sentence and NO identity sentence is licensed; the result must be",
        "reported as a non-monotonicity with the per-arm levels, and it questions whether a single number ('how many",
        "scales') describes this intervention at all.  " + FLOOR_NOTE, NOT_MAG, NOT_KIM, NOT_ROUTE],
    "ALL-THREE-NEEDED": [
        "Neither any single carrier scale nor the carrier pair lifts the run off k01 (all within %.0f pp), while the" % NULL_BAR,
        "three-carrier mask recovers.  SENTENCE LICENSED: 'at this cell the rescue needs ALL THREE carrier scales",
        "exempted; no one of them and no two of them suffice.'  EFFECT ON " + READING + ": cwd3's sentence stands but",
        "becomes a statement about the SET, not about any member, and 278's COUNT / DOSE rival (T19) is SUPPORTED at",
        "count 1 and 2 -- the campaign may NOT write that a named single scale carries the precondition on ResNet.",
        "IDENTICAL IN LEVEL TO THE BROKEN-SUBSET-MASK NULL, which G-BITE excluded.  " + FLOOR_NOTE, NOT_MAG, NOT_KIM],
    "CONTROL-EXCEEDS-AT-TWO": [
        "The class-pure, count-, width-, depth- and numel-matched carrier-free pair {47,56} reaches AT LEAST AS HIGH A",
        "STATE as the carrier pair {50,53}, and is not itself at k01's floor.  AT MATCHED COUNT TWO THE CARRIER IDENTITY",
        "IS NOT WHAT MATTERS.  SENTENCE LICENSED: 'exempting any two 512-wide layer4 BatchNorm scales from the coupled",
        "decay does at least as much as exempting two carriers: at the decay grain the effect follows the COUNT (or the",
        "class or the depth), not the ctd1 carrier identity.'  EFFECT ON " + READING + ": 278's COUNT / DOSE rival (T19)",
        "is SUPPORTED and cwd3's specificity reading is CONTRADICTED at count two -- the mechanism line must be rewritten",
        "as a statement about last-block normalisation scales in general, not about the three nominated tensors.",
        "This is an ADVERSE outcome for the campaign's own account and is registered as such BEFORE any run.  " + NOT_ROUTE],
    "UNRESOLVED-DIVERGED": ["A bimodal arm must be reported, not averaged.  No branch; " + READING + " untouched."],
    "SCALAR-NOT-COLLAPSED": ["k01 (no mask) did not collapse in batch (> %.0f pp).  Nothing to explain; the batch cannot" % K01_MAX,
                             "speak to any mask."],
    "REPLICATE-FAILED": [
        "CARWD0 (WD 0 on the three ctd1 carriers) is not %.0f pp above k01: cwd3's CARRIER-DECAY-SUFFICES did NOT" % GAP_MIN,
        "replicate in batch, so there is no recovered level to read the subset arms against.  IDENTICAL IN LEVEL to the",
        "null in which NO mask bites (G-BITE excluded it).  No count or identity sentence; cwd3's result is QUESTIONED,",
        "not overturned (between-batch)."],
}

# the three licence paragraphs that assert the carrier-free pair is at k01's floor (the 278.6 D1 analogue, [5])
FLOOR_CLAUSE_ACCOUNTS = ("ANY-ONE-SUFFICES", "TWO-SUFFICES", "COUNT-GRADED")

R_BETWEEN_ORDER = ("corpus mechanism cell (LIMITS-PREP 2.2)", "cwd3 (278.3)")
R_BETWEEN = {"corpus mechanism cell (LIMITS-PREP 2.2)": (("k01 (n44)", 22.96), ("layerwise (n32)", 69.42)),
             "cwd3 (278.3)": CWD3_LEVELS}


def f32(x):
    """the float32 value of a python float, re-derived through struct."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ---------------------------------------------------------------------------------------------------------------------
# THE NETWORK, REBUILT FROM THE ARCHITECTURE (no manifest, no repo module)
#   ResNet18 for CIFAR-100: 3x3 stem (no maxpool), four stages of two BasicBlocks with widths 64 / 128 / 256 / 512,
#   a 1x1 convolutional downsample + BatchNorm on the first block of stages 2..4, and a 512 -> 100 linear head.
#   `ndim` follows the harness's own rule: 1 for a normalisation parameter or any `.bias`, 4 for a conv weight,
#   2 for the linear weight.
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


def dm_witness(spec):
    if not spec:
        return "DECAY_MASK: off"
    idx = dm_indices(spec)
    return ("DECAY_MASK: on base=SGDm wd=%r spec=%s masked=%d of=%d numel=%d idx=%s names=%s"
            % (WD_BASE, spec, len(idx), len(TENSORS), sum(TENSORS[i][1] for i in idx),
               ",".join("%d" % (i + 1) for i in idx), ",".join(TENSORS[i][0] for i in idx)))


WITNESS_DM = dict((a, dm_witness(DMASK[a])) for a in ARMS)
SET_IDX = dict((a, tuple(i + 1 for i in dm_indices(DMASK[a]))) for a in MASKED)


def manifest_text():
    lines = ["NETWORK %s" % NET, "NUM_PARAM_TENSORS %d" % len(TENSORS), "TOTAL_PARAMS %d" % sum(x[1] for x in TENSORS),
             "CLIP_C %s" % CLIP, "EPOCHS %d" % EPOCHS, "META_STEPS %d" % (EPOCHS * STEPS_PER_EPOCH),
             "WD_BASE %r" % WD_BASE]
    for i, (n, q, o, nd) in enumerate(TENSORS, 1):
        lines.append("TENSOR %d %s %d %s ndim=%d" % (i, n, q, o, nd))
    lines.append("NORMSCALE %s" % ",".join("%d:%s" % (i + 1, TENSORS[i][0]) for i in dm_indices("normscale")))
    for a in ARMS:
        lines.append("ARMSPEC %s SPEC %s TYPE %s" % (a, SPEC[a], PT_TYPE[a]))
        lines.append("DECAYMASK %s %s" % (a, DMASK[a] or "off"))
        lines.append("DWITNESS %s %s" % (a, WITNESS_DM[a]))
    txt = "\n".join(lines) + "\n"
    for a in MASKED:
        idx = dm_indices(DMASK[a])
        txt += "SET %s idx=%s numel=%d owners=%s widths=%s classes=%s\n" % (
            a, ",".join("%d" % (i + 1) for i in idx), sum(TENSORS[i][1] for i in idx),
            ",".join(TENSORS[i][2] for i in idx), ",".join("%d" % TENSORS[i][1] for i in idx),
            ",".join("scale" if TENSORS[i][0].endswith(".weight") else "shift" for i in idx))
    return txt


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


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


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
        st = ln.strip()
        if rec["device"] is None and st.endswith(" MiB") and ", " in st and not st.startswith(("ARGS:", "ENV:", "Epoch")):
            rec["device"] = st.split(",")[0].strip()
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
# G-BITE, re-derived: Lion on the ONE group of every arm; no hold key; the DECAY_MASK record audit
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
# the branch, re-derived (CORRECTIONS 280.4)
# ---------------------------------------------------------------------------------------------------------------------
RANK = {"NULL": 0, "PART": 1, "REC": 2}


def state(M, x):
    if M[x] >= M[REF] - MATCH_BAR:
        return "REC"
    if M[x] <= M["k01"] + NULL_BAR:
        return "NULL"
    return "PART"


def account_of(two, ctl2, singles):
    if RANK[ctl2] >= RANK[two] and RANK[ctl2] >= 1:
        return "CONTROL-EXCEEDS-AT-TWO"
    m1 = max(RANK[s] for s in singles)
    r2 = RANK[two]
    if m1 == 2:
        return "ANY-ONE-SUFFICES" if all(RANK[s] == 2 for s in singles) else "ONE-SUFFICES-PARTIAL"
    if r2 == 2:
        return "TWO-SUFFICES"
    if r2 == 1:
        return "COUNT-GRADED"
    return "PARTIAL-NON-MONOTONE" if m1 == 1 else "ALL-THREE-NEEDED"


def decide(M, diverged=False):
    if diverged:
        return "UNRESOLVED-DIVERGED", None
    if M["k01"] > K01_MAX:
        return "SCALAR-NOT-COLLAPSED", None
    if M[REF] - M["k01"] < GAP_MIN:
        return "REPLICATE-FAILED", None
    st = dict((a, state(M, a)) for a in STATEARMS)
    tok = account_of(st["TWOWD0"], st["CTL2WD0"], [st[a] for a in SINGLES])
    return tok, "TWO-%s+CTL2-%s+ONE50-%s+ONE53-%s+ONE59-%s" % tuple(st[a] for a in STATEARMS)


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
    tab-separated table with `#` comment lines, and a row is dropped iff its (run, job_id) PAIR is listed.  cwd4's own
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
    P(" cwd4 -- ResNet18_c100: IDENTITY vs COUNT (and, as far as this network allows, vs MAGNITUDE) IN THE DECAY MASK")
    P(" %s / %s   %d epochs   seeds %s   AUGMENT=%s  BETA_CLIP=%s  PROBE=%d  PROBE_TENSOR=1  every arm scalar"
      % (NET, DSET, EPOCHS, ",".join(str(s) for s in SEEDS), AUG, CLIP, PROBE))
    for a in ARMS:
        P("   %-8s k=%d  idx %-9s DECAY_MASK %s"
          % (a, K_MASKED[a], ",".join("%d" % i for i in SET_IDX.get(a, ())) or "-", DMASK[a] or "(unset)"))
    P(" CO-PRIMARY: P_2SPEC = TWOWD0 - CTL2WD0; D_TWO = CARWD0 - TWOWD0; P_ONE53 = ONE53 - k01 (bounded in neither direction).")
    P(" RECOVERY IS AGAINST THE IN-BATCH CARWD0.  plateau5 = mean TEST over epochs %d..%d of each run's own .out."
      % (EPOCHS - 5, EPOCHS - 1))
    P(" BARS ARE FROZEN LITERALS (O2).")
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
    P("G-ARGS    every run's OWN ARGS line says what its file name registers")
    for s in SEEDS:
        for a in ARMS:
            r = runs[(a, s)]
            if len(r["args"]) != 1:
                P("  FAIL G-ARGS %s-s%d exactly one ARGS line   %d" % (a, s, len(r["args"])))
                continue
            f = parse_args_line(r["args"][0])
            dup = [k for k, v in f.items() if len(v) > 1]
            want = {"optimizer": "HF", "alg-base": "SGDm", "alg-meta": "Lion", "dataset": DSET, "NN-name": NET,
                    "batch-size": str(BATCH), "num-epochs": str(EPOCHS), "meta-stepsize": MST, "alpha0": A0,
                    "stepsize-groups": SPEC[a], "seed": str(s), "run-name": "%s-%s-s%d" % (PREFIX, a, s),
                    "momentum-param-base": "0.99", "weight-decay-base": "0.1", "momentum-param-meta": "0.99",
                    "Lion-beta2-meta": "0.9", "weight-decay-meta": "0", "gamma": "1"}
            bad = [k for k, v in want.items() if f.get(k, [None])[-1] != v]
            extra = ("   repeated %s; wrong %s" % (dup, ["%s=%s" % (k, f.get(k, [None])[-1]) for k in bad])) if (dup or bad) else ""
            P("  %-4s G-ARGS %s-s%d%s" % ("PASS" if (not dup and not bad) else "FAIL", a, s, extra))

    P("")
    P("G-ENV     the environment rode the ENV line on every run")
    envs = sorted(set(" ".join(t for t in ln.split() if not t.startswith("PROBE_DIR=")) for r in runs.values() for ln in r["env"]))
    ok1 = len(envs) == 1 and all(len(r["env"]) == 1 for r in runs.values())
    P("  %-4s G-ENV one distinct ENV line (PROBE_DIR stripped), one per run   %d distinct"
      % ("PASS" if ok1 else "FAIL", len(envs)))
    for ln in envs:
        P("        %s" % ln)
    P("  %-4s G-ENV the ENV line is cvt1's ... cvt9's / cwd1's / cwd3's, byte for byte (PROBE_DIR stripped)"
      % ("PASS" if envs == [ENV_EXPECTED] else "FAIL"))
    badpt = [(a, s) for (a, s), r in sorted(runs.items())
             if len(r["pt"]) != 1 or not r["pt"][0].startswith("PROBE_TENSOR: on every=%d type=%s tensors=%d "
                                                               % (PROBE, PT_TYPE[a], NTENS))]
    P("  %-4s G-ENV every run printed ONE `PROBE_TENSOR: on every=%d type=scalar tensors=%d`   %s"
      % ("PASS" if not badpt else "FAIL", PROBE, NTENS, ("bad: %s" % badpt[:5]) if badpt else "%d runs" % len(runs)))

    P("")
    P("G-SETSEP  the structural precondition for G-WITNESS to discriminate arms that G-BITE's k cannot (280.4)")
    wits = dict((a, WITNESS_DM[a]) for a in ARMS)
    idxs = dict((a, SET_IDX[a]) for a in MASKED)
    kcoll = sorted(set(K_MASKED[a] for a in MASKED if sum(K_MASKED[x] == K_MASKED[a] for x in MASKED) > 1))
    P("  %-4s G-SETSEP the %d registered DECAY_MASK witnesses are PAIRWISE DISTINCT   %d distinct"
      % ("PASS" if len(set(wits.values())) == len(ARMS) else "FAIL", len(ARMS), len(set(wits.values()))))
    P("  %-4s G-SETSEP the %d masked idx tuples are PAIRWISE DISTINCT"
      % ("PASS" if len(set(idxs.values())) == len(MASKED) else "FAIL", len(MASKED)))
    P("  DECLARED: k collides on %s -- TWOWD0 vs CTL2WD0 (both k=2) and the three singles (all k=1) are NOT separated"
      % (kcoll,))
    P("  by G-BITE's k.  They are separated by G-WITNESS (idx + names in each run's own line) and by the bite job's")
    P("  RR4 set discrimination on the real GPU path.  This limit is registered, not discovered afterwards.")

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
    P("  %-4s G-STRUCT byte-identical to cwd4_design.manifest_text(CWD4)   %d vs %d bytes"
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
                     ("DESIGN_SHA256", "DESIGN == the cwd4_design.py this scorer loaded"),
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
        P("  %-8s TEST %.4f  sd %s  range %.4f   TRAIN %.4f   tail slope %s pp/ep"
          % (a, mean(v), fmt(sd(v), 4), max(v) - min(v), mean(t), "/".join(fmt(x, 3) for x in sl)))
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
    P("G-BITE    Lion recomputed on the one group of every arm; no hold key; the DECAY_MASK record audit (k per arm)")
    nonfin = {}
    for s in SEEDS:
        for a in ARMS:
            recs = load_probe(runsdir, a, s)
            ok, dl, dd, hk = bite(a, recs)
            nonfin[(a, s)] = dl["nonfinite"] + dd["nonfinite"]
            P("  %-4s G-BITE %-8s s%d  records %s  groups %d  lion-mismatch %d (ties %d, worst %.1e)  bad %d  nonfinite %d  "
              "hold-key records %s | dm: k=%d bad_n %d bad_skipped %d bad_masked %d bad_types %d on-unmasked %d pos_wdterm %d "
              "nonfinite %d last %s%s"
              % ("PASS" if ok else "FAIL", a, s, "absent" if recs is None else len(recs), NG[a], dl["bad_lion"],
                 dl["ties"], dl["worst"], dl["bad_rec"], dl["nonfinite"], hk, K_MASKED[a], dd["bad_n"],
                 dd["bad_skipped"], dd["bad_masked"], dd["bad_types"], dd["keys_on_unmasked"], dd["pos_wdterm"],
                 dd["nonfinite"], dd["last"],
                 ("  (%s)" % (dl.get("why") or dd.get("why"))) if (dl.get("why") or dd.get("why")) else ""))

    P("")
    P("G-HW      DISCLOSURE ONLY (271.4(5)): the GPU each run's own nvidia-smi line names -- NO gate reads it")
    hw = dict(((a, s), runs[(a, s)]["device"]) for s in SEEDS for a in ARMS)
    for a in ARMS:
        P("  %-8s %s" % (a, "  ".join("s%d %s" % (s, hw[(a, s)]) for s in SEEDS)))
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
    P("CONTRASTS (every one WITHIN batch)")
    CP = {}
    for kind, nm, (x, y) in (("PRIMARY", "P_2SPEC", ("TWOWD0", "CTL2WD0")), ("PRIMARY", "D_TWO", ("CARWD0", "TWOWD0")),
                             ("PRIMARY", "P_ONE53", ("ONE53", "k01")), ("KEY", "P_CAR", ("CARWD0", "k01")),
                             ("KEY", "P_TWO", ("TWOWD0", "k01")), ("KEY", "P_CTL2", ("CTL2WD0", "k01")),
                             ("KEY", "P_ONE50", ("ONE50", "k01")), ("KEY", "P_ONE59", ("ONE59", "k01"))):
        CP[nm] = (M[x] - M[y], T[x] - T[y])
        P("  %-8s %-8s = %-7s - %-7s = %+.4f pp = %+.2f SE   (TRAIN %+.4f)   seeds %s"
          % (kind, nm, x, y, CP[nm][0], CP[nm][0] / se, CP[nm][1],
             " / ".join("%+.3f" % (runs[(x, s)]["plateau5"] - runs[(y, s)]["plateau5"]) for s in SEEDS)))
        P("           +/-2 SE interval [%+.4f, %+.4f]" % (CP[nm][0] - 2 * se, CP[nm][0] + 2 * se))
    P("  DESCR    D_MAG    = ONE59 - ONE53 = %+.4f pp   (mean |L| 3.2984e-01 vs 1.9526e-01, ratio 1.69 -- but Kim"
      % (M["ONE59"] - M["ONE53"]))
    P("           class ALSO differs, gamma_last vs gamma_down: this contrast does NOT isolate magnitude)")
    P("  DESCR    D_CLASS  = ONE59 - ONE50 = %+.4f pp   (BOTH Kim gamma_last; mean |L| 3.2984e-01 vs 2.9155e-01,"
      % (M["ONE59"] - M["ONE50"]))
    P("           ratio 1.13 -- the ONLY within-class magnitude contrast this network offers, and it is small)")
    P("  DESCR    D_ONEMEAN = CARWD0 - mean(ONE50,ONE53,ONE59) = %+.4f pp"
      % (M["CARWD0"] - mean([M[a] for a in SINGLES])))
    P("  bars: GAP_MIN %.1f (%.2f SE) | MATCH %.1f pp (%.2f SE) | NULL %.1f pp (%.2f SE) | RESCUE %.1f pp (%.2f SE)"
      % (GAP_MIN, GAP_MIN / se, MATCH_BAR, MATCH_BAR / se, NULL_BAR, NULL_BAR / se, RESCUE_BAR, RESCUE_BAR / se))
    P("  state bars: REC iff arm >= %.4f (CARWD0 - %.0f) ; NULL iff arm <= %.4f (k01 + %.0f)"
      % (M[REF] - MATCH_BAR, MATCH_BAR, M["k01"] + NULL_BAR, NULL_BAR))
    for a in STATEARMS:
        P("  STATE    %-8s %.4f -> %-4s  (to REC bar %+.4f pp, to NULL bar %+.4f pp)   k=%d  idx %s  Kim %s"
          % (a, M[a], state(M, a), M[a] - (M[REF] - MATCH_BAR), M[a] - (M["k01"] + NULL_BAR), K_MASKED[a],
             ",".join("%d" % i for i in SET_IDX[a]), ",".join(KIM_CLASS[i] for i in SET_IDX[a])))

    P("")
    P("G-DIVERGE (a branch condition, not a harness gate)")
    div = [a for a in ARMS if max(arm_v[a]) - min(arm_v[a]) > DIVERGED_BAR]
    P("  %s every arm's seed range <= %.1f pp%s" % ("PASS" if not div else "FAIL", DIVERGED_BAR,
                                                    "" if not div else "  diverged: %s" % div))

    branch, word = decide(M, bool(div))
    st = dict((a, state(M, a)) for a in STATEARMS)
    stamps = ([word] if word else []) + [
        "HARNESS-CLEAN", "PATCH-BITES", "MASK-UPDATE-AND-TRACE", "CONV-LINEAR-WD-KEPT", "ALL-ARMS-SCALAR",
        "RECOVERY-AGAINST-CARWD0", "CTL2-CLASS-PURE-COUNT-MATCHED", "CTL-NOT-MAGNITUDE-MATCHED", "MAGNITUDE-NOT-SEPARATED",
        "POSITION-CLASS-NOT-SEPARATED", "ONE-NETWORK-RESNET18", "ONE-CELL", "HORIZON-100-ONLY",
        "SIGMA-PRIOR-FROZEN" if which == "SIGMA_PRIOR_frozen" else "SIGMA-INBATCH"]
    if div:
        stamps.append("DIVERGED-%s" % "-".join(div))
    if word:
        if st["TWOWD0"] == "REC" and st["CTL2WD0"] == "NULL":
            stamps.append("TWO-SPECIFIC")
        if st["CTL2WD0"] == "PART":
            stamps.append("CTL2-PARTIAL")
        if any(RANK[st[a]] > RANK[st["TWOWD0"]] for a in SINGLES):
            stamps.append("NON-MONOTONE-IN-COUNT")
        if M["TWOWD0"] > M[REF] + MATCH_BAR:
            stamps.append("TWO-ABOVE-CAR")
        if len(set(st[a] for a in SINGLES)) == 1:
            stamps.append("SINGLES-SATURATED")
        else:
            stamps.append("ONE-ORDER-" + "-".join(sorted(SINGLES, key=lambda a: -M[a])))
        if any(st[a] == "NULL" for a in STATEARMS):
            stamps.append("FLOOR-READINGS-ARE-BOUNDS")
    for a in ARMS[1:]:
        if M[a] < M["k01"] - NULL_BAR:
            stamps.append("%s-BELOW-K01" % a)
    if abs(M["CARWD0"] - CAR_CWD3) > MATCH_BAR:
        stamps.append("CARWD0-DIFFERS-FROM-CWD3")
    if abs(M["CTL2WD0"] - CTL2_CWD3) > MATCH_BAR:
        stamps.append("CTL2-DIFFERS-FROM-CWD3")
    for a in ARMS:
        if any(nonfin[(a, s)] for s in SEEDS):
            stamps.append("NONFINITE-%s" % a)
    stamps.append(("HW-UNIFORM-%s" % hwset[0].replace(" ", "_")) if len(hwset) == 1 else "HW-MIXED")
    agree = all((t > 0) == (p > 0) for p, t in CP.values() if abs(p) >= RESCUE_BAR)
    stamps.append("TRAIN-AGREES" if agree else "TRAIN-DISAGREES")
    final = "FINAL: %s | %s" % (branch, " | ".join(stamps))
    P("")
    P(bar)
    P(final)
    P(bar)

    P("")
    P("BETWEEN-BATCH, NON-GATING (frozen literals; NO other batch enters any cwd4 bar or contrast):")
    for b_ in R_BETWEEN_ORDER:
        P("  %s: %s" % (b_, "  ".join("%s %.4f" % kv for kv in R_BETWEEN[b_])))
    P("  cwd4 {%s}: %s" % (",".join(str(s) for s in SEEDS), "  ".join("%s %.4f" % (a, M[a]) for a in ARMS)))
    P("  CARWD0 - cwd3's CARWD0 = %+.4f pp; CTL2WD0 - cwd3's CTL2WD0 = %+.4f pp; k01 - cwd3's k01 = %+.4f pp"
      % (M["CARWD0"] - CAR_CWD3, M["CTL2WD0"] - CTL2_CWD3, M["k01"] - K01_CWD3))

    P("")
    P("DESCRIPTIVE (non-gating) -- PATCH_DECAYMASK's readout on the masked arms: ||w|| per masked tensor (seed means)")
    P("and min |w|, entries |w| < 1e-3, at records 0 / 50 / 200 / 499.  k01 carries no readout (the mask is off there).")
    for a in MASKED:
        rows = [load_probe(runsdir, a, s) or [] for s in SEEDS]
        parts = []
        k = K_MASKED[a]
        for r_ in (0, 50, 200, 499):
            v = [r[r_] for r in rows if len(r) > r_ and isinstance(r[r_], dict) and isinstance(r[r_].get("dm_norm"), list)
                 and len(r[r_]["dm_norm"]) == k]
            if not v:
                parts.append("rec %d n/a" % r_)
                continue
            nrm = "/".join("%.3g" % mean([x["dm_norm"][p] for x in v]) for p in range(k))
            parts.append("rec %d: %s absmin %.3g small %.1f"
                         % (r_, nrm, mean([x["dm_absmin"] for x in v]), mean([x["dm_small"] for x in v])))
        P("  %-8s idx %-8s %s" % (a, ",".join("%d" % i for i in SET_IDX[a]), " | ".join(parts)))
    P("  (the median of a dm_norm vector is NOT printed here: 278.6 D1 reported cWD3's upper-middle median as a")
    P("   RULE 16 defect in a registered file; this scorer prints the per-tensor norms and no median at all.)")

    P("")
    P("WHAT THIS DECIDES -- as registered, before any run existed:")
    for ln in LICENSE[branch]:
        P("  " + ln)
    P("")
    P("WHAT IT DOES NOT LICENSE, UNCONDITIONALLY:")
    P("  * Anything about PlainNet / VGG / GroupNorm, other cells, momentum 0.9, or WD on conv / linear weights.")
    P("  * Which ROUTE (weight shrink vs meta trace -- one switch changes both), decoupled weight decay, other WD values.")
    P("  * IDENTITY vs TERM MAGNITUDE: no magnitude-matched carrier-free control exists on this network.  The three")
    P("    carriers are mean-|L| ranks 1/2/3 of 62 on the scalar anchor's pinned records AND the only tensors that")
    # RULE 16 DEFECT F1 (cwd4), REPRODUCED HERE RATHER THAN CORRECTED.  analysis/cWD4_countwd_score.py lines 779
    # and 781 write `100 %%` inside a BARE print() -- no `%` operator is applied to those two strings, so the
    # `%%` escape is never consumed and the scorer emits a LITERAL double percent.  CORRECTIONS 280's own prose
    # says `100 %`.  The registered scorer is NOT edited (RULE 16); this parser re-derives the bytes the scorer
    # ACTUALLY prints, and the defect is reported in [6] below so it is visible rather than papered over.
    P("    vote DOWN on 100 %% of them; the only carrier-free 512-wide layer4 BN scales are ranks 33 and 36, so the")
    P("    SMALLEST carrier exceeds the LARGEST admissible control tensor by 128.8x.  The next-largest terms in the")
    P("    whole model (linear.weight rank 4, layer4.0.conv2.weight rank 5) vote UP on 100 %% of pinned records and")  # RULE 16 defect F1, see above
    P("    are neither normalisation scales nor numel-matched, so they are not an admissible control either.")
    P("    D_MAG / D_CLASS are DESCRIPTIVE and are BOUNDS wherever the singles saturate (stamp SINGLES-SATURATED).")
    P("  * ctd1 IDENTITY vs Kim et al.'s POSITION CLASS (arXiv:2205.07260) at count 3 or 1; at count TWO the carrier")
    P("    pair is {gamma_last, gamma_down} and the control is {gamma_others, gamma_others}, so class and carrier")
    P("    status remain confounded there too -- this batch separates COUNT, not CLASS.")
    P("  * Any statement about the layerwise grouping (no layerwise arm) or beyond 100 epochs.")

    X = {"M": M, "T": T, "branch": branch, "word": word, "final": final, "runs": runs, "se": se, "sigma_in": sigma_in,
         "CP": CP, "arm_v": arm_v, "arm_t": arm_t, "div": div, "stamps": stamps, "hwset": hwset, "st": st,
         "manifest": mt, "df": df, "which": which, "nonfin": nonfin, "incomplete": None}
    return L, X


# ---------------------------------------------------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: cwd4_attack_indep.py <runsdir> <scorer-log> [--csv P] [--tsv P]")
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
    print(" cwd4_attack_indep -- AN INDEPENDENT RE-DERIVATION OF THE REGISTERED cwd4 SCORER'S OUTPUT")
    print(" CORRECTIONS 280.  No repo module imported, no regular expression, every literal re-typed.")
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
    for a in MASKED:
        idx = dm_indices(DMASK[a])
        ck(tuple(i + 1 for i in idx) == REG_SET_IDX[a] and sum(TENSORS[i][1] for i in idx) == REG_SET_NUMEL[a]
           and len(idx) == K_MASKED[a],
           "%s's name list resolves to the registered set" % a,
           "idx %s numel %d k %d" % (",".join("%d" % (i + 1) for i in idx), sum(TENSORS[i][1] for i in idx), len(idx)))
    w512 = tuple(i + 1 for i in ns if TENSORS[i][1] == 512)
    ck(w512 == (47, 50, 53, 56, 59),
       "every 512-wide BN scale in the model is one of {47,50,53,56,59} (188.1 / 275.2)",
       "512-wide scales %s" % (w512,))
    carrier_idx = tuple(i + 1 for i in dm_indices(DMASK["CARWD0"]))
    free512 = tuple(i for i in w512 if i not in carrier_idx)
    ck(free512 == (47, 56) and free512 == REG_SET_IDX["CTL2WD0"],
       "the carrier-free 512-wide BN scales are EXACTLY {47,56}: TWO -- so a count-matched class-pure control",
       "exists at count 2 and CANNOT exist at count 3 (280's whole design insight), free512 %s" % (free512,))
    ck(set(REG_SET_IDX["TWOWD0"]) < set(REG_SET_IDX["CARWD0"])
       and not (set(REG_SET_IDX["CTL2WD0"]) & set(REG_SET_IDX["CARWD0"])),
       "TWOWD0 is a PROPER SUBSET of CARWD0 and CTL2WD0 is DISJOINT from it")
    ck(REG_SET_NUMEL["TWOWD0"] == REG_SET_NUMEL["CTL2WD0"] == 1024,
       "the two count-2 arms are numel-matched", "%d = %d" % (REG_SET_NUMEL["TWOWD0"], REG_SET_NUMEL["CTL2WD0"]))
    ck(all(TENSORS[i - 1][0].endswith(".weight") and TENSORS[i - 1][3] == 1
           for i in REG_SET_IDX["TWOWD0"] + REG_SET_IDX["CTL2WD0"]),
       "both count-2 sets are class-pure: every member is a 1-dim BatchNorm SCALE, no shift")
    ck(abs(f32(math.log(1e-6)) - (-13.815510749816895)) == 0.0,
       "alpha0 1e-6 -> beta0 re-derived through float32", "%r" % f32(math.log(1e-6)))

    print("")
    print("[1b] COULD THE DECAY_MASK HALF OF G-BITE HAVE FAILED?  every run's records read as EVERY k in the batch")
    print("     (its own, and every other): this separates 280.4's null (i) -- a mask that bit on CARWD0 but on NO")
    print("     subset arm, which predicts EXACTLY ALL-THREE-NEEDED -- from a mask that really bit.  ITS LIMIT IS")
    print("     RE-DERIVED, NOT HIDDEN: k=2 is shared by TWOWD0 and CTL2WD0, k=1 by the three singles, so a swap")
    print("     INSIDE either group is invisible here.  [1c] is what refuses it.")
    KS = sorted(set(K_MASKED.values()))
    cross = []
    for s in SEEDS:
        for a in ARMS:
            recs = load_probe(runsdir, a, s)
            res = [(k, dm_audit(recs, k)[0]) for k in KS]
            own = dict(res)[K_MASKED[a]]
            others = [k for k, o in res if k != K_MASKED[a] and o]
            cross.append((a, s, own, others))
            print("      %-8s s%d  as-registered(k=%d) %-4s   cross-reads that also pass: %s"
                  % (a, s, K_MASKED[a], "PASS" if own else "FAIL", others if others else "none"))
    ck(all(o and not x for _a, _s, o, x in cross),
       "every run passes its OWN k and FAILS every other k in the batch",
       "%d runs, %d cross-reads refused" % (len(cross), sum(1 for c in cross if not c[3])))
    samek = sorted(set((K_MASKED[a], tuple(sorted(x for x in MASKED if K_MASKED[x] == K_MASKED[a]))) for a in MASKED
                       if sum(K_MASKED[x] == K_MASKED[a] for x in MASKED) > 1))
    for k, grp in samek:
        print("      DECLARED LIMIT: k=%d is shared by %s -- the record audit alone cannot tell them apart" % (k, list(grp)))
    # 280.10's OWED ITEM, discharged here: "the dm_* readout stated with the RIGHT DENOMINATOR (274's lesson:
    # k01 carries none)".  Counted from the records themselves, not quoted from the registration.
    n_masked_runs = n_masked_rec = n_pos = n_plain_runs = n_plain_rec = n_dmkey_on_k01 = 0
    for s in SEEDS:
        for a in ARMS:
            recs = load_probe(runsdir, a, s)
            hit = sum(1 for r in recs if "dm_wdterm" in r and r["dm_wdterm"] > 0)
            if a in MASKED:
                n_masked_runs += 1
                n_masked_rec += len(recs)
                n_pos += hit
            else:
                n_plain_runs += 1
                n_plain_rec += len(recs)
                n_dmkey_on_k01 += sum(1 for r in recs if any(str(kk).startswith("dm_") for kk in r))
    print("      DENOMINATOR (280.10): %d masked runs x %d records = %d MASKED records, %d with a positive"
          % (n_masked_runs, n_masked_rec // max(n_masked_runs, 1), n_masked_rec, n_pos))
    print("      dm_wdterm.  The %d k01 runs contribute %d records and carry NO dm_* key (%d found): they are"
          % (n_plain_runs, n_plain_rec, n_dmkey_on_k01))
    print("      NOT part of this denominator, which is exactly 274's lesson.")
    ck(n_pos == n_masked_rec and n_dmkey_on_k01 == 0,
       "every masked record carries a positive dm_wdterm and no k01 record carries a dm_* key",
       "%d/%d masked, %d dm_* keys on k01" % (n_pos, n_masked_rec, n_dmkey_on_k01))

    print("")
    print("[1c] THE WITNESS, RE-DERIVED FROM THE ARCHITECTURE AND MATCHED TO WHAT THE RUN ITSELF PRINTED")
    print("     (this is the ONLY readout that separates the two k=2 arms and the three k=1 arms from each other)")
    wbad = []
    for s in SEEDS:
        for a in ARMS:
            r = X["runs"].get((a, s))
            if r is None or r["DECAY_MASK"] != [WITNESS_DM[a]]:
                wbad.append("%s-s%d" % (a, s))
    ck(not wbad, "all %d runs printed exactly the witness this parser rebuilt from the architecture"
       % (len(ARMS) * len(SEEDS)), "bad: %s" % wbad if wbad else "%d/%d" % (len(ARMS) * len(SEEDS), len(ARMS) * len(SEEDS)))
    ck(len(set(WITNESS_DM.values())) == len(ARMS),
       "the %d rebuilt witnesses are PAIRWISE DISTINCT (so a swap inside a k group is detectable)" % len(ARMS))
    swaps = []
    for s in SEEDS:
        for a in ARMS:
            r = X["runs"].get((a, s))
            if r is None:
                continue
            for b in ARMS:
                if b != a and r["DECAY_MASK"] == [WITNESS_DM[b]]:
                    swaps.append("%s-s%d carries %s's witness" % (a, s, b))
    ck(not swaps, "no run carries ANOTHER arm's registered witness", "; ".join(swaps) if swaps else "0 swaps")
    for a in ARMS:
        print("      %-8s %s" % (a, WITNESS_DM[a]))

    print("")
    print("[1d] THE MANIFEST AND THE PROVENANCE, REBUILT")
    mp = os.path.join(runsdir, PREFIX, "PARTITION-MANIFEST.txt")
    livetxt = open(mp).read() if os.path.exists(mp) else ""
    ck(livetxt == X["manifest"], "the batch's PARTITION-MANIFEST.txt == the manifest rebuilt here, byte for byte",
       "%d bytes, sha256 %s" % (len(X["manifest"]), hashlib.sha256(X["manifest"].encode()).hexdigest()[:16]))
    badprov = [k for k in sorted(PROV_WANT) if prov_file.get(k) != PROV_WANT[k]]
    ck(not badprov,
       "every PROVENANCE.txt value equals the literal RE-TYPED here (280.5 / 271.2), scorer sha included",
       "bad: %s" % badprov if badprov else "%d keys" % len(PROV_WANT))

    if X.get("incomplete"):
        print("")
        print("INCOMPLETE: %s" % X["incomplete"])
        return 1

    print("")
    print("[2] THE NUMBERS, RE-DERIVED FROM THE RAW RECORDS ONLY")
    for a in ARMS:
        print("  %-8s TEST %.4f  TRAIN %.4f  seeds %s"
              % (a, X["M"][a], X["T"][a], " / ".join("%.4f" % x for x in X["arm_v"][a])))
    print("  SIGMA_INBATCH %.6f   SIGMA_USED %s   SE_ARM_DIFF %.6f   df %d"
          % (X["sigma_in"], X["which"], X["se"], X["df"]))
    for nm in ("P_2SPEC", "D_TWO", "P_ONE53", "P_CAR", "P_TWO", "P_CTL2", "P_ONE50", "P_ONE59"):
        print("  %-8s %+.4f pp = %+.2f SE   (TRAIN %+.4f)" % (nm, X["CP"][nm][0], X["CP"][nm][0] / X["se"],
                                                              X["CP"][nm][1]))
    for a in STATEARMS:
        print("  STATE %-8s %s" % (a, X["st"][a]))
    print("  branch  %s" % X["branch"])

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
            ok = theirs.startswith("  manifest: ") and theirs.endswith("/cwd4/PARTITION-MANIFEST.txt (<runsdir>/cwd4/)")
            host += 1 if ok else 0
            if not ok:
                mism += 1
                first.append(i + 1)
            continue
        if mine == "@PROVPATH@":
            ok = theirs.startswith("  provenance: ") and theirs.endswith("/cwd4/PROVENANCE.txt (<runsdir>/cwd4/)")
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
    print("[5] THE FLOOR-CLAUSE REACHABILITY AUDIT (the 278.6 D1 analogue, re-derived over all 3^5 = 243 states)")
    print("     Three registered licence paragraphs -- ANY-ONE-SUFFICES, TWO-SUFFICES, COUNT-GRADED -- assert in")
    print("     terms that the carrier-free pair {47,56} 'stays at k01' / is 'at k01'.  A PART control is NOT at the")
    print("     floor.  This enumerates which accounts are reachable with CTL2WD0 in state PART.")
    words = ("NULL", "PART", "REC")
    reach = {}
    for two in words:
        for ctl2 in words:
            for o50 in words:
                for o53 in words:
                    for o59 in words:
                        tok = account_of(two, ctl2, (o50, o53, o59))
                        reach.setdefault(tok, set()).add(ctl2)
    for tok in sorted(reach):
        print("      %-24s CTL2WD0 states that reach it: %s" % (tok, ",".join(sorted(reach[tok]))))
    bad_floor = sorted(t for t in FLOOR_CLAUSE_ACCOUNTS if "PART" in reach.get(t, set()))
    print("      floor-clause accounts REACHABLE with CTL2WD0 PART: %s" % (bad_floor or "none"))
    here_ctl2 = X["st"]["CTL2WD0"]
    print("      THIS batch: CTL2WD0 is %s, and the FINAL %s the CTL2-PARTIAL stamp"
          % (here_ctl2, "carries" if "CTL2-PARTIAL" in X["stamps"] else "does not carry"))
    ck(("CTL2-PARTIAL" in X["stamps"]) == (here_ctl2 == "PART"),
       "the CTL2-PARTIAL stamp is present exactly when CTL2WD0 is PART")
    # NOT a violation of the scorer -- a WRITE-UP instruction this parser emits for the landing entry to obey.
    if X["branch"] in FLOOR_CLAUSE_ACCOUNTS and here_ctl2 == "PART":
        print("      QUOTE-BLOCK: branch %s is a floor-clause account AND CTL2WD0 is PART.  THE LICENCE PARAGRAPH"
              % X["branch"])
        print("      MAY NOT BE QUOTED VERBATIM: strike its floor clause and state CTL2WD0's own level instead.")
    else:
        print("      QUOTE-OK: branch %s with CTL2WD0 %s -- no floor clause is asserted against a non-floor control."
              % (X["branch"], here_ctl2))

    print("")
    print("[6] RULE 16 DEFECTS FOUND IN THE REGISTERED SCORER BY THIS REPLAY (REPORTED, NOT FIXED)")
    print("     F1 (cwd4, NEW).  analysis/cWD4_countwd_score.py lines 779 and 781 print the literal string")
    print("     `100 %%` from a BARE print() -- no `%` operator is applied, so the `%%` escape is never consumed")
    print("     and the scorer's stdout carries a DOUBLE percent sign where CORRECTIONS 280's own prose has one.")
    print("     COSMETIC: it is in the WHAT-IT-DOES-NOT-LICENSE block and touches no bar, sigma, level, contrast,")
    print("     branch or stamp.  The registered file is NOT edited (RULE 16); this parser reproduces the bytes")
    print("     the scorer actually emits, so the byte-identical claim above is honest rather than flattering.")
    print("     A landing that QUOTES that sentence must write `100 %` -- the scorer's rendering is the defect.")
    print("")
    print("VIOLATIONS %d" % len(v))
    return 0 if not v else 1


if __name__ == "__main__":
    raise SystemExit(main())
