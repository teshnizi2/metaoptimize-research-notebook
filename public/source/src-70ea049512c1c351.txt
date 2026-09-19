#!/usr/bin/env python3
# =============================================================================
# cwd3_attack_indep.py -- THE INDEPENDENT PARSER OF `cwd3` (CORRECTIONS 275, landing entry).
#
# It re-derives, FROM THE RAW RECORDS ALONE, every line the REGISTERED scorer
# analysis/cWD3_carrierwd_score.py prints, and compares its rebuild with a committed scorer log, line by line, in order.
#
#   python3 analysis/cwd3_attack_indep.py <runsdir> <scorer-log> [--csv P] [--tsv P]
#
# INDEPENDENCE, deliberately (the cwd2 pattern, CORRECTIONS 261, carried to cwd3):
#   * it imports NOTHING from cWD3_carrierwd_score.py, cwd3_design.py, cwd_design.py, cwd_common.py,
#     cVT9_dosewindow_score.py, cVT4_betahold_score.py, corpus_exclusions.py, argsline_guard.py or any other repo
#     module.  Stdlib only: hashlib, json, math, os, struct, sys.  `csv` is NOT used either -- the corpus is split by
#     hand.
#   * NO regular expressions anywhere.  The `.out` lines are split on commas / colons / whitespace, the TSV on tabs,
#     the file names on hyphens.
#   * every bar, sigma literal, witness string, tree sha, licence sentence and stamp name below is RE-TYPED from
#     CORRECTIONS 275, not read from any module.
#   * the ResNet18_c100 tensor table is rebuilt FROM THE ARCHITECTURE (stem, four stages of two BasicBlocks, the three
#     downsample shortcuts, the linear head), not read from a manifest, and the three masked sets are resolved against
#     that rebuild by NAME.
#   * float32 rounding is re-derived through `struct`, and the Lion natural step is recomputed from the records' own
#     beta_pre / mom_pre / z_agg, so G-BITE's every printed counter is independent of the scorer's implementation.
#
# HOST INDEPENDENCE.  Three kinds of line in the scorer's output are host-dependent and are matched, not rebuilt byte
# for byte: (1) the two path disclosures (`manifest:` / `provenance:`), matched by their host-independent suffix;
# (2) the corpus disclosure, whose row count depends on which corpus commit the host's tree carries -- this parser
# re-filters the corpus IT IS GIVEN and requires the log's count to equal its own, WITHOUT printing either number, so
# that THIS parser's own stdout is byte-identical on both hosts; (3) the G-PROV `SCORER_SHA256` row, whose printed
# prefix is the scorer file's own sha -- re-typed here from CORRECTIONS 275.5, so it is rebuilt, not waived.
#
# WHAT IT ADDS BEYOND A REPLAY (the four independent attacks):
#   [1]  the design rebuilt from the architecture: 62 tensors / 11,220,132 params, the 20 normscale indices, and each
#        arm's masked set resolved by name -> the registered idx / numel.
#   [1b] the CROSS-READ: every run's probe records are read as its OWN arm's k AND as every OTHER arm's k.  A run may
#        pass only its own.  This is the gate that separates CORRECTIONS 275.4's null (i) -- a name-list mask that
#        never bit, which predicts EXACTLY NEEDS-MORE-THAN-THREE -- from a mask that really bit.
#   [1c] the WITNESS re-derivation: each arm's `DECAY_MASK: on ...` line rebuilt from the architecture table and
#        required to equal, byte for byte, the line the run's own `.out` printed.
#   [2]  every level, sigma, contrast, state and stamp recomputed from the raw `.out` epoch lines.
#   [3]  line-by-line agreement with the committed scorer log.
#   [4]  the FINAL line rebuilt token by token.
# =============================================================================

import hashlib
import json
import math
import os
import struct
import sys

# ---------------------------------------------------------------------------------------------------------------------
# RE-TYPED LITERALS (CORRECTIONS 275; the cell from 260 / 271, the ENV line from cvt1...cvt9 / cwd1)
# ---------------------------------------------------------------------------------------------------------------------
PREFIX = "cwd3"
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
SEEDS = (140, 141, 142)
ARMS = ("k01", "CARWD0", "CTLWD0", "CTL2WD0", "NWD")
SETARMS = ("CARWD0", "CTLWD0", "CTL2WD0")
MASKED = ("CARWD0", "CTLWD0", "CTL2WD0", "NWD")

CARRIERS = ("layer4.0.bn2.weight", "layer4.0.shortcut.1.weight", "layer4.1.bn2.weight")     # ctd1, idx 50 / 53 / 59
CTL_DEPTH = ("layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.1.bn1.weight")             # cdep1 DEPTH, 47 / 48 / 56
CTL_DEPTH2 = ("layer4.0.bn1.weight", "layer4.1.bn1.weight")                                 # cdep1 DEPTH2, 47 / 56
DMASK = {"k01": "", "CARWD0": "+".join(CARRIERS), "CTLWD0": "+".join(CTL_DEPTH),
         "CTL2WD0": "+".join(CTL_DEPTH2), "NWD": "normscale"}
SPEC = dict((a, "scalar") for a in ARMS)
PT_TYPE = dict((a, "scalar") for a in ARMS)
K_MASKED = {"k01": 0, "CARWD0": 3, "CTLWD0": 3, "CTL2WD0": 2, "NWD": 20}
NG = dict((a, 1) for a in ARMS)                           # every arm is SCALAR: exactly one step-size group
REG_SET_IDX = {"CARWD0": (50, 53, 59), "CTLWD0": (47, 48, 56), "CTL2WD0": (47, 56)}
REG_SET_NUMEL = {"CARWD0": 1536, "CTLWD0": 1536, "CTL2WD0": 1024, "NWD": 4800}
REG_NORMSCALE_IDX = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59)

# the runner's ENV line (PROBE_DIR stripped), cvt1 ... cvt9's / cwd1's -- RE-TYPED
ENV_EXPECTED = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na "
                "COS_TOTAL=default COS_WARMUP=default SCHED=none SCHED_TOTAL=none "
                "SCHED_WARMUP=none SCHED_MIN=none PROBE=100 EB_RHO=na EB_LOG=0")
OFF_WITNESS = {"VOTE_W": "VOTE_W: off", "BETA_HOLD": "BETA_HOLD: off", "GROUP_HOLD": "GROUP_HOLD: off",
               "REST_HOLD": "REST_HOLD: off"}
KINDS = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD", "COMP_HOLD", "WINDOW_HOLD", "DECAY_MASK")
DM_KEYS = ("dm_n", "dm_masked", "dm_skipped", "dm_wdterm", "dm_norm", "dm_absmin", "dm_small")
HOLD_PREFIXES = ("bh_", "gh_", "rh_", "ch_", "wh_")

# frozen bars (CORRECTIONS 275.4), RE-TYPED
SIGMA_PRIOR = 0.6481128684085689
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
NWD_CWD1 = 70.7760

# Lion, RE-TYPED (the meta optimiser of the cell: ms 1e-3, beta2 0.9, clip -15 .. -2.3026)
MS = 1e-3
B2 = 0.9
LO, HI = -15.0, -2.3026
TIE_REL = 1e-12

# provenance, RE-TYPED (CORRECTIONS 275.5; the tree / runner / build_network shas from 271.2 / cwd1's registration)
BN_SHA = "c79988833c4399a06cc84d15c0830f54d86f59feeec345f47a2287c1cf2ba77d"        # cvt8's build_network.py
HF_POST_SHA = "94aedc33046afb12782ffb7f2eb729bafc7fcba5f42227fec373fa67b6f58ebf"   # cvt8's HF.py + PATCH_DECAYMASK
HF_PARENT_SHA = "5197dc2eecbbcb6feb815f3798b3455171e6ad614deac913be5df48dd9ede1a9"  # cvt8's HF.py
RUNNER_SHA = "34a8c90ee3a0bee00aa2d0b4a3ad5d16d461fbf5a1f0cda5797e44e44c158371"    # jobs/run_cifar_cwd1.sh
SCORER_SHA = "bade3827b0261b5d3044f5df6344f53b257b1bac1f142699f6d34b5ad4c849a6"    # analysis/cWD3_carrierwd_score.py
DESIGN_SHA = "f8afdfa24cac6c4e7d34aa9378a0bc6e315a4802ed83d05bd060163787377318"   # analysis/cwd3_design.py (275.5)
CWD_DESIGN_SHA = "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195"
COMMON_SHA = "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"
PROV_WANT = {"MODE": "submit", "BUILD_NETWORK_SHA256": BN_SHA, "HF_SHA256": HF_POST_SHA,
             "HF_PRE_DECAYMASK_SHA256": HF_PARENT_SHA, "RUNNER_SHA256": RUNNER_SHA, "SCORER_SHA256": SCORER_SHA,
             "DESIGN_SHA256": DESIGN_SHA, "CWD_DESIGN_SHA256": CWD_DESIGN_SHA, "COMMON_SHA256": COMMON_SHA}

# ---------------------------------------------------------------------------------------------------------------------
# THE REGISTERED LICENCE PROSE, RE-TYPED FROM analysis/cWD3_carrierwd_score.py's LICENSE dict (CORRECTIONS 275.4 / the
# registration commit 3c22eda).  Re-typed, not imported: the bars inside the sentences are this file's own literals.
# ---------------------------------------------------------------------------------------------------------------------
READING = "the carrier account of the scalar collapse at the mechanism cell (MASTER-TABLE rows 212-227, cwd1 271)"
NOT_ROUTE = "Not licensed: which ROUTE acts (weight shrink vs meta trace -- the mask changes both), decoupled WD, other cells."
FLOOR_NOTE = "A NULL state is a location at k01's floor (164.6), a BOUND on the effect, not a measured zero."
LICENSE = {
    "CARRIER-DECAY-SUFFICES": [
        "CARWD0 sits within %.0f pp of NWD while CTLWD0 AND CTL2WD0 sit within %.0f pp of k01: removing coupled WD from the" % (MATCH_BAR, NULL_BAR),
        "three ctd1 carrier BatchNorm scales ALONE (1,536 of 11,220,132 parameters; update AND meta trace) removes the scalar",
        "collapse as fully as removing it from all 20 scales, while removing it from cdep1's count-, numel-, width- and",
        "depth-matched non-carrier triple {47,48,56} and from the class-pure non-carrier pair {47,56} does not lift the run.",
        "SENTENCE LICENSED, at this cell: 'the collapse's weight-decay precondition is carried by the three carriers' own",
        "decay: removing it from those three scales alone removes the collapse, and removing it from matched non-carrier",
        "last-block normalisation parameters does not.'  EFFECT ON " + READING + ": the sentence 271 / 274 could NOT write",
        "is EARNED at this cell.  " + FLOOR_NOTE,
        "Not licensed: which ONE of the three; identity vs dynamical MAGNITUDE (188: the carriers' vote mass is x225 any",
        "admissible control's); " + NOT_ROUTE],
    "CARRIER-DECAY-SUFFICES-CLASS-OPEN": [
        "CARWD0 recovers (within %.0f pp of NWD) and the count-matched triple CTLWD0 stays at k01, but the class-pure pair" % MATCH_BAR,
        "CTL2WD0 does NOT stay at k01 -- a SUBSET of the null control moves while the control does not (non-monotone).",
        "SENTENCE LICENSED: 'removing the three carriers' decay alone removes the collapse, and removing it from a count-",
        "and numel-matched non-carrier triple does not'.  NOT LICENSED: 'only the carriers' -- the class-pure pair moved.",
        FLOOR_NOTE, NOT_ROUTE],
    "SPECIFICITY-PARTIAL": [
        "CARWD0 recovers but CTLWD0 sits BETWEEN k01 and NWD: the carriers' decay suffices, and the matched non-carrier",
        "triple's decay carries part of the collapse too.  Sentence licensed: 'removing the carriers' decay alone removes",
        "the collapse; the matched triple's removes part of it' -- identity is favoured, not established.  Read P_SPEC.",
        NOT_ROUTE],
    "ANY-THREE-SCALES": [
        "CARWD0 AND CTLWD0 both recover to within %.0f pp of NWD: removing coupled WD from a few layer4 normalisation" % MATCH_BAR,
        "parameters suffices, WHICHEVER they are.  The carrier IDENTITY is NOT earned at the decay grain.  Sentence",
        "licensed: 'removing coupled WD from a few last-block normalisation parameters removes the collapse; the ctd1",
        "carriers are not special in this respect'.  EFFECT ON " + READING + ": the decay precondition is LOCAL (layer4), not",
        "carrier-specific; ctd1 / ciso1's identity claims stand on the STEP-SIZE grain only.  " + NOT_ROUTE],
    "CARRIER-DECAY-PARTIAL": [
        "CARWD0 sits BETWEEN k01 and NWD while CTLWD0 stays at k01: the carriers' own decay carries PART of the collapse,",
        "specifically, and the other 17 scales carry the rest (read P_CAR, P_SET, F_CAR).  Sentence licensed: 'removing",
        "the carriers' decay alone removes part of the collapse; a matched non-carrier set removes none'.  NOT licensed:",
        "'removing their decay removes it'.  " + FLOOR_NOTE, NOT_ROUTE],
    "PARTIAL-BOTH": [
        "CARWD0 and CTLWD0 both sit BETWEEN k01 and NWD: each three-tensor mask removes part of the collapse; read P_SPEC",
        "for identity.  Neither the carrier sentence nor 'any three' is licensed; the precondition is distributed.", NOT_ROUTE],
    "NEEDS-MORE-THAN-THREE": [
        "Neither CARWD0 nor CTLWD0 lifts the run off k01 (both within %.0f pp), while NWD recovers: the WD precondition is" % NULL_BAR,
        "NOT carried by the three carriers' decay alone; more of the 20 scales are needed.  IDENTICAL IN LEVEL TO THE BROKEN",
        "NAME-LIST-MASK NULL, which G-BITE excluded.  Sentence licensed: 'removing coupled WD from the three carriers alone",
        "does not remove the collapse; removing it from all 20 scales does'.  EFFECT ON " + READING + ": the sentence 271 / 274",
        "could not write is REFUTED at this cell; cwd1's 20-scale form is the one to write.  " + FLOOR_NOTE],
    "CONTROL-EXCEEDS": [
        "The matched non-carrier mask CTLWD0 lifts the run at least as far as the carrier mask CARWD0 (state word): at the",
        "decay grain the ctd1 nomination is CONTRADICTED -- a non-carrier triple's decay matters as much or more.  Sentence",
        "licensed: 'the collapse's WD precondition is not specific to the ctd1 carriers'.  " + NOT_ROUTE],
    "UNRESOLVED-DIVERGED": ["A bimodal arm must be reported, not averaged.  No branch; " + READING + " untouched."],
    "SCALAR-NOT-COLLAPSED": ["k01 (no mask) did not collapse in batch (> %.0f pp).  Nothing to explain; the batch cannot" % K01_MAX,
                             "speak to any mask."],
    "REPLICATE-FAILED": [
        "NWD (WD 0 on all 20 BN scales) is not %.0f pp above k01: cwd1's COLLAPSE-VANISHES did not replicate in batch, so" % GAP_MIN,
        "there is no recovered level to read the three-set arms against.  IDENTICAL IN LEVEL to the null in which NO mask",
        "bites (G-BITE excluded it).  No carrier sentence; cwd1's result is QUESTIONED, not overturned (between-batch)."],
}

R_BETWEEN_ORDER = ("corpus mechanism cell (LIMITS-PREP 2.2)", "cwd1 (271)")
R_BETWEEN = {"corpus mechanism cell (LIMITS-PREP 2.2)": (("k01 (n44)", 22.96), ("layerwise (n32)", 69.42)),
             "cwd1 (271)": (("k01", 22.9513), ("k01NWD", 70.7760), ("kLNWD", 69.3420))}


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
    for a in SETARMS:
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
# the branch, re-derived (CORRECTIONS 275.4)
# ---------------------------------------------------------------------------------------------------------------------
def state(M, x):
    if M[x] >= M["NWD"] - MATCH_BAR:
        return "REC"
    if M[x] <= M["k01"] + NULL_BAR:
        return "NULL"
    return "PART"


def account_of(c, t, t2):
    if c == "REC" and t == "NULL":
        return "CARRIER-DECAY-SUFFICES" if t2 == "NULL" else "CARRIER-DECAY-SUFFICES-CLASS-OPEN"
    if c == "REC" and t == "PART":
        return "SPECIFICITY-PARTIAL"
    if c == "REC" and t == "REC":
        return "ANY-THREE-SCALES"
    if c == "PART" and t == "NULL":
        return "CARRIER-DECAY-PARTIAL"
    if c == "PART" and t == "PART":
        return "PARTIAL-BOTH"
    if c == "NULL" and t == "NULL":
        return "NEEDS-MORE-THAN-THREE"
    return "CONTROL-EXCEEDS"


def decide(M, diverged=False):
    if diverged:
        return "UNRESOLVED-DIVERGED", None
    if M["k01"] > K01_MAX:
        return "SCALAR-NOT-COLLAPSED", None
    if M["NWD"] - M["k01"] < GAP_MIN:
        return "REPLICATE-FAILED", None
    c, t, t2 = state(M, "CARWD0"), state(M, "CTLWD0"), state(M, "CTL2WD0")
    return account_of(c, t, t2), "CAR-%s+CTL-%s+CTL2-%s" % (c, t, t2)


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
    tab-separated table with `#` comment lines, and a row is dropped iff its (run, job_id) PAIR is listed.  cwd3's own
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
    P(" cwd3 -- ResNet18_c100: IS THE COLLAPSE's WD PRECONDITION CARRIED BY THE THREE CARRIERS' OWN DECAY?  (PATCH_DECAYMASK)")
    P(" %s / %s   %d epochs   seeds %s   AUGMENT=%s  BETA_CLIP=%s  PROBE=%d  PROBE_TENSOR=1  every arm scalar"
      % (NET, DSET, EPOCHS, ",".join(str(s) for s in SEEDS), AUG, CLIP, PROBE))
    for a in ARMS:
        P("   %-8s DECAY_MASK %s" % (a, DMASK[a] or "(unset)"))
    P(" CO-PRIMARY: P_CAR = CARWD0 - k01; P_SPEC = CARWD0 - CTLWD0; P_SET = NWD - CARWD0 (bounded in neither direction).")
    P(" plateau5 = mean TEST over epochs %d..%d of each run's own .out.  BARS ARE FROZEN LITERALS (O2)." % (EPOCHS - 5, EPOCHS - 1))
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
    P("  %-4s G-ENV one distinct ENV line, one per run   %d distinct" % ("PASS" if ok1 else "FAIL", len(envs)))
    for ln in envs:
        P("        %s" % ln)
    P("  %-4s G-ENV the ENV line is cvt1's ... cvt9's / cwd1's, byte for byte (PROBE_DIR stripped)"
      % ("PASS" if envs == [ENV_EXPECTED] else "FAIL"))
    badpt = [(a, s) for (a, s), r in sorted(runs.items())
             if len(r["pt"]) != 1 or not r["pt"][0].startswith("PROBE_TENSOR: on every=%d type=%s tensors=%d "
                                                               % (PROBE, PT_TYPE[a], NTENS))]
    P("  %-4s G-ENV every run printed ONE `PROBE_TENSOR: on every=%d type=scalar tensors=%d`   %s"
      % ("PASS" if not badpt else "FAIL", PROBE, NTENS, ("bad: %s" % badpt[:5]) if badpt else "%d runs" % len(runs)))

    P("")
    P("G-WITNESS one line of each kind per run == the registered witness (VOTE_W / BETA_HOLD / GROUP_HOLD / REST_HOLD off; DECAY_MASK)")
    for s in SEEDS:
        for a in ARMS:
            r = runs[(a, s)]
            for kind, want in list(OFF_WITNESS.items()) + [("DECAY_MASK", WITNESS_DM[a])]:
                good = r[kind] == [want]
                P("  %-4s G-WITNESS %s %s-s%d%s" % ("PASS" if good else "FAIL", kind, a, s,
                  "" if good else "   got %r" % ([x[:160] for x in r[kind][:2]],)))
            others = [k for k in ("COMP_HOLD", "WINDOW_HOLD") if r[k]]
            P("  %-4s G-WITNESS %s-s%d prints no COMP_HOLD / WINDOW_HOLD line (cvt8 lineage)%s"
              % ("PASS" if not others else "FAIL", a, s, ("   " + str(others)) if others else ""))

    P("")
    P("G-STRUCT  the batch's own PARTITION-MANIFEST.txt == the frozen manifest, byte for byte")
    P("@MANIFESTPATH@")
    mt = manifest_text()
    mp = os.path.join(runsdir, PREFIX, "PARTITION-MANIFEST.txt")
    txt = open(mp).read() if os.path.exists(mp) else ""
    P("  %-4s G-STRUCT byte-identical to cwd3_design.manifest_text(CWD3)   %d vs %d bytes"
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
                     ("DESIGN_SHA256", "DESIGN == the cwd3_design.py this scorer loaded"),
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

    P("")
    P("CONTRASTS (every one WITHIN batch)")
    CP = {}
    for kind, nm, (x, y) in (("PRIMARY", "P_CAR", ("CARWD0", "k01")), ("PRIMARY", "P_SPEC", ("CARWD0", "CTLWD0")),
                             ("PRIMARY", "P_SET", ("NWD", "CARWD0")), ("KEY", "P_NWD", ("NWD", "k01")),
                             ("KEY", "P_CTL", ("CTLWD0", "k01")), ("KEY", "P_CTL2", ("CTL2WD0", "k01"))):
        CP[nm] = (M[x] - M[y], T[x] - T[y])
        P("  %-8s %-6s = %-7s - %-7s = %+.4f pp = %+.2f SE   (TRAIN %+.4f)   seeds %s"
          % (kind, nm, x, y, CP[nm][0], CP[nm][0] / se, CP[nm][1],
             " / ".join("%+.3f" % (runs[(x, s)]["plateau5"] - runs[(y, s)]["plateau5"]) for s in SEEDS)))
    F_CAR = CP["P_CAR"][0] / CP["P_NWD"][0] if CP["P_NWD"][0] else float("nan")
    P("  DESCR    F_CAR  = P_CAR / P_NWD = %.4f" % F_CAR)
    P("  DESCR    D_SHIFT = CTLWD0 - CTL2WD0 = %+.4f pp (the one BN shift in CTLWD0's triple, and the count/numel difference)"
      % (M["CTLWD0"] - M["CTL2WD0"]))
    P("  bars: GAP_MIN %.1f (%.2f SE) | MATCH %.1f pp (%.2f SE) | NULL %.1f pp (%.2f SE) | RESCUE %.1f pp (%.2f SE)"
      % (GAP_MIN, GAP_MIN / se, MATCH_BAR, MATCH_BAR / se, NULL_BAR, NULL_BAR / se, RESCUE_BAR, RESCUE_BAR / se))
    P("  state bars: REC iff arm >= %.4f (NWD - %.0f) ; NULL iff arm <= %.4f (k01 + %.0f)"
      % (M["NWD"] - MATCH_BAR, MATCH_BAR, M["k01"] + NULL_BAR, NULL_BAR))
    for a in SETARMS:
        P("  STATE    %-8s %.4f -> %s   (to REC bar %+.4f pp, to NULL bar %+.4f pp)"
          % (a, M[a], state(M, a), M[a] - (M["NWD"] - MATCH_BAR), M[a] - (M["k01"] + NULL_BAR)))

    P("")
    P("G-DIVERGE (a branch condition, not a harness gate)")
    div = [a for a in ARMS if max(arm_v[a]) - min(arm_v[a]) > DIVERGED_BAR]
    P("  %s every arm's seed range <= %.1f pp%s" % ("PASS" if not div else "FAIL", DIVERGED_BAR,
                                                    "" if not div else "  diverged: %s" % div))

    branch, word = decide(M, bool(div))
    stamps = ([word] if word else []) + [
        "HARNESS-CLEAN", "PATCH-BITES", "MASK-UPDATE-AND-TRACE", "CONV-LINEAR-WD-KEPT", "ALL-ARMS-SCALAR",
        "RECOVERY-AGAINST-NWD", "CTL-HAS-ONE-SHIFT", "CTL-DEPTH-MATCHED-NOT-MAGNITUDE", "ONE-NETWORK-RESNET18",
        "ONE-CELL", "HORIZON-100-ONLY", "SIGMA-PRIOR-FROZEN" if which == "SIGMA_PRIOR_frozen" else "SIGMA-INBATCH"]
    if div:
        stamps.append("DIVERGED-%s" % "-".join(div))
    if word:
        rank = {"NULL": 0, "PART": 1, "REC": 2}
        if rank[state(M, "CTL2WD0")] > rank[state(M, "CTLWD0")]:
            stamps.append("CTL2-EXCEEDS-CTL")
        if any(state(M, a) == "NULL" for a in SETARMS):
            stamps.append("FLOOR-READINGS-ARE-BOUNDS")
        if M["CARWD0"] > M["NWD"] + MATCH_BAR:
            stamps.append("CAR-ABOVE-NWD")
    for a in ARMS[1:]:
        if M[a] < M["k01"] - NULL_BAR:
            stamps.append("%s-BELOW-K01" % a)
    if abs(M["NWD"] - NWD_CWD1) > MATCH_BAR:
        stamps.append("NWD-DIFFERS-FROM-CWD1")
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
    P("BETWEEN-BATCH, NON-GATING (frozen literals; NO other batch enters any cwd3 bar or contrast):")
    for b_ in R_BETWEEN_ORDER:
        P("  %s: %s" % (b_, "  ".join("%s %.4f" % kv for kv in R_BETWEEN[b_])))
    P("  cwd3 {%s}: %s" % (",".join(str(s) for s in SEEDS), "  ".join("%s %.4f" % (a, M[a]) for a in ARMS)))

    P("")
    P("DESCRIPTIVE (non-gating) -- PATCH_DECAYMASK's readout on the masked arms: ||w|| per masked tensor (seed means) and")
    P("min |w|, entries |w| < 1e-3, at records 0 / 50 / 200 / 499.  k01 carries no readout (the mask is off there).")
    normnames = [TENSORS[i][0] for i in dm_indices("normscale")]
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
            if k <= 3:
                nrm = "/".join("%.3g" % mean([x["dm_norm"][p] for x in v]) for p in range(k))
            else:
                pos = [normnames.index(c) for c in CARRIERS]
                nrm = "carriers %s median %.3g" % ("/".join("%.3g" % mean([x["dm_norm"][p] for x in v]) for p in pos),
                                                   mean([sorted(x["dm_norm"])[len(x["dm_norm"]) // 2] for x in v]))
            parts.append("rec %d: %s absmin %.3g small %.1f"
                         % (r_, nrm, mean([x["dm_absmin"] for x in v]), mean([x["dm_small"] for x in v])))
        P("  %-8s %s" % (a, " | ".join(parts)))

    P("")
    P("WHAT THIS DECIDES -- as registered, before any run existed:")
    for ln in LICENSE[branch]:
        P("  " + ln)
    P("")
    P("WHAT IT DOES NOT LICENSE, UNCONDITIONALLY:")
    P("  * Anything about PlainNet / VGG / GroupNorm, other cells, momentum 0.9, or WD on conv / linear weights.")
    P("  * Which ROUTE (weight shrink vs meta trace -- one switch changes both), decoupled weight decay, other WD values.")
    P("  * Which ONE of the three carriers (they are masked together); identity vs dynamical magnitude (188.1).")
    P("  * Any statement about the layerwise grouping (no layerwise arm) or beyond 100 epochs.")

    X = {"M": M, "T": T, "branch": branch, "word": word, "final": final, "runs": runs, "se": se, "sigma_in": sigma_in,
         "CP": CP, "F_CAR": F_CAR, "arm_v": arm_v, "arm_t": arm_t, "div": div, "stamps": stamps, "hwset": hwset,
         "manifest": mt, "df": df, "which": which, "nonfin": nonfin, "incomplete": None}
    return L, X


# ---------------------------------------------------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: cwd3_attack_indep.py <runsdir> <scorer-log> [--csv P] [--tsv P]")
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
    print(" cwd3_attack_indep -- AN INDEPENDENT RE-DERIVATION OF THE REGISTERED cwd3 SCORER'S OUTPUT")
    print(" CORRECTIONS 275.  No repo module imported, no regular expression, every literal re-typed.")
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
    ck(tuple(i + 1 for i in ns) == REG_NORMSCALE_IDX and sum(TENSORS[i][1] for i in ns) == REG_SET_NUMEL["NWD"],
       "the 20 normscale tensors resolved by ndim==1 and name.endswith('.weight')",
       "%d tensors, numel %d" % (len(ns), sum(TENSORS[i][1] for i in ns)))
    for a in SETARMS:
        idx = dm_indices(DMASK[a])
        ck(tuple(i + 1 for i in idx) == REG_SET_IDX[a] and sum(TENSORS[i][1] for i in idx) == REG_SET_NUMEL[a]
           and len(idx) == K_MASKED[a],
           "%s's name list resolves to the registered set" % a,
           "idx %s numel %d k %d" % (",".join("%d" % (i + 1) for i in idx), sum(TENSORS[i][1] for i in idx), len(idx)))
    w512 = tuple(i + 1 for i in ns if TENSORS[i][1] == 512)
    ck(w512 == (47, 50, 53, 56, 59),
       "every 512-wide BN scale in the model is one of {47,50,53,56,59} (188.1 / 275.2: no carrier-free class-pure triple)",
       "512-wide scales %s" % (w512,))
    ck(TENSORS[47][0] == "layer4.0.bn1.bias" and TENSORS[47][3] == 1,
       "CTLWD0's idx 48 is a BatchNorm SHIFT, not a scale (the CTL2WD0 contrast exists for this reason)",
       "%s" % TENSORS[47][0])
    ck(abs(f32(math.log(1e-6)) - (-13.815510749816895)) == 0.0,
       "alpha0 1e-6 -> beta0 re-derived through float32", "%r" % f32(math.log(1e-6)))

    print("")
    print("[1b] COULD THE DECAY_MASK HALF OF G-BITE HAVE FAILED?  every run's records read as EVERY arm's k")
    print("     (the registered k, and every other k in the batch): exactly one may pass.  This is the gate that")
    print("     separates a name-list mask that never bit (275.4 null (i)) from one that really bit.")
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
       "every run passes its OWN k and FAILS every other arm's k",
       "%d runs, %d cross-reads refused" % (len(cross), sum(1 for c in cross if not c[3])))

    print("")
    print("[1c] THE WITNESS, RE-DERIVED FROM THE ARCHITECTURE AND MATCHED TO WHAT THE RUN ITSELF PRINTED")
    wbad = []
    for s in SEEDS:
        for a in ARMS:
            r = X["runs"].get((a, s))
            if r is None or r["DECAY_MASK"] != [WITNESS_DM[a]]:
                wbad.append("%s-s%d" % (a, s))
    ck(not wbad, "all 15 runs printed exactly the witness this parser rebuilt from the architecture",
       "bad: %s" % wbad if wbad else "15/15")
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
       "every PROVENANCE.txt value equals the literal RE-TYPED here (275.5 / 271.2), scorer sha included",
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
    for nm in ("P_CAR", "P_SPEC", "P_SET", "P_NWD", "P_CTL", "P_CTL2"):
        print("  %-7s %+.4f pp = %+.2f SE   (TRAIN %+.4f)" % (nm, X["CP"][nm][0], X["CP"][nm][0] / X["se"],
                                                              X["CP"][nm][1]))
    print("  F_CAR %.4f   D_SHIFT %+.4f pp" % (X["F_CAR"], X["M"]["CTLWD0"] - X["M"]["CTL2WD0"]))
    for a in SETARMS:
        print("  STATE %-8s %s" % (a, state(X["M"], a)))
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
            ok = theirs.startswith("  manifest: ") and theirs.endswith("/cwd3/PARTITION-MANIFEST.txt (<runsdir>/cwd3/)")
            host += 1 if ok else 0
            if not ok:
                mism += 1
                first.append(i + 1)
            continue
        if mine == "@PROVPATH@":
            ok = theirs.startswith("  provenance: ") and theirs.endswith("/cwd3/PROVENANCE.txt (<runsdir>/cwd3/)")
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
    print("VIOLATIONS %d" % len(v))
    return 0 if not v else 1


if __name__ == "__main__":
    raise SystemExit(main())
