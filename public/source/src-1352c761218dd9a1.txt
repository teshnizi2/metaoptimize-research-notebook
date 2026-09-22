#!/usr/bin/env python3
"""crt2_design.py -- THE FROZEN DESIGN TABLE OF `crt2` (CORRECTIONS 313): THE RETUNE PAST `crt1`'s GRID EDGE.

QUESTION.  After tuning each grain until its optimum is LOCATED (not at a grid edge), does plain scalar TIE, BEAT, or
FALL BELOW the better count-matched partition at `crt1`'s cell (ResNet18 / CIFAR-10 / SGDm 0.99 + Lion, alpha-scaled
weight decay 5e-4)?

WHY THIS BATCH EXISTS.  `crt1` (309) landed `WEAKENED-TO-TIE`: every grain selected A2 (ms 1e-4, alpha0 1e-2), the TOP of
a two-point alpha0 axis (`GRID-EDGE-*-A2` x3), so no grain's optimum was located; from M2 to A2 the partitions gained
+3.59 / +3.50 pp and scalar +1.28 (309.3, descriptive), and whether an alpha0 above 1e-2 closes the tie further or
reverses it is untested (311 A3).  309's branch also rested on sigma: the rule max(floor, in-batch) chose the in-batch
0.2498 (df 24), and at the floor 0.1782 the branch would have been the opposite licence (break-even 0.2014, 309.4 item 2).

THE GRID (a five-point PLUS centred on (ms 1e-4, alpha0 3e-2); 313.3):
    A2  ms 1e-4  alpha0 1e-2   crt1's selection, REPLICATED on fresh seeds (the alpha0 lower arm of the plus)
    A3  ms 1e-4  alpha0 3e-2   the centre: the one point that is interior on BOTH axes
    A4  ms 1e-4  alpha0 1e-1   the alpha0 upper arm; ln(1e-1) = -2.302585 IS the box ceiling -2.3026 (to 1.5e-5), so every
                               group starts at the ceiling and the upper alpha0 edge and the ceiling coincide (313.4)
    H3  ms 3e-4  alpha0 3e-2   the meta step size crossed at the new alpha0, upward
    L3  ms 3e-5  alpha0 3e-2   ... and downward
  alpha0 axis at ms 1e-4: {1e-2, 3e-2, 1e-1} -- BRACKETS 3e-2 in batch; 1e-2 is bracketed from below by `crt1`'s landed
  IN-BATCH contrast A2 - M2 (alpha0 1e-2 vs 1e-3 at ms 1e-4: +3.5887 / +3.5040 / +1.2780 pp, each > 2 SE 0.4080; 309.3).
  ms axis at alpha0 3e-2: {3e-5, 1e-4, 3e-4}.
  Same grid for every grain (a fair tuning protocol gives every grain the same search space: Choi et al. 2019,
  Sivaprasad et al. 2020; PRIOR-ART 2026-09-22 crt2 sweep).

THE CEILING (313.4) is NOT an axis.  It is the runner's ENV var BETA_CLIP=-15:-2.3026, parsed in HF.__init__ (PATCH_CLIP,
HF.py 4732b74a lines 117-122) and applied as a clamp of every log step size AFTER each meta update (lines 191-193).  From
`crt1`'s probe records: at the selected A2 arms it holds, in the plateau window, a MEDIAN of 4-5 of 14,421 chunk777 groups
(max 9; window-median fraction 2.8e-4 to 3.5e-4) -- all single-chunk BatchNorm weight / bias tensors of layers 3-4
(probe tensors 26-56, 128-512 parameters each; time-averaged about 1.9-2.2 thousand of 11,173,962 parameters, 0.02 %) --
and ZERO groups of nodewise and scalar.  `crt1`'s "box-bound on 49 % of records" is the record-level stamp (ANY one
coordinate at the edge), not bulk binding.  So the ceiling is not what separated scalar from the partitions at A2, and it
does not earn one of 15 configs.  Because the new alpha0 values push toward it, crt2 REGISTERS a bulk-occupancy stamp
(`CEIL-BULK-<arm>`: window-median fraction of groups at the ceiling >= 0.05, the audit's 5 % box number) and makes a
ceiling-bound selection NOT LOCATED (the scorer's `located`).

THE CELL is `crt1`'s VERBATIM (cgw1's W4 ARGS line, 20 flags; ENV AUGMENT=1 BETA_CLIP=-15:-2.3026 PROBE=5), only
`--meta-stepsize`, `--alpha0` and `--stepsize-groups` vary.  THE TREE is a fresh isolated byte copy (`harness_crt2`) of the
SAME pinned bytes `crt1` and `cgw1` ran.  Seeds {200, 201, 202} (this track's block 200-207; 203-207 unused).

Stdlib only.  Imported, UNEDITED, by analysis/cRT2_retune_score.py, analysis/crt2_live_check.py and bin/cRT2_retune.sh.
    python3 analysis/crt2_design.py      # prints the manifest
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PREFIX = "crt2"
NET = "ResNet18"
DSET = "CIFAR10"
EPOCHS = 100
BATCH = 100
STEPS_PER_EPOCH = 50000 // BATCH             # 500
META_STEPS = EPOCHS * STEPS_PER_EPOCH        # 50,000
PROBE = 5
N_RECORDS = META_STEPS // PROBE              # 10,000
CLIP = "-15:-2.3026"
LO, HI = -15.0, -2.3026
WD_TOKEN = "5e-4"
WD_VALUE = 0.0005
AUG = "1"
GAMMA = "1"
WALL = "02:00:00"
GPU_CONSTRAINT = "L4"
GPU_NAME = "NVIDIA L4"
NTENS = 62
TOTPAR = 11173962

# the tree: an isolated byte copy -- the SAME pinned bytes as crt1 / cgw1 (analysis/crt1_design.py)
HF_LIVE_SHA = "4732b74aa3a10508896e92eccced5c89051c353fa8a01a3af21aab9aeda0cecd"
TRAIN_SHA = "3fea309e172609c1ee1be143d5f1847404426df31869a63c4cefbb1ebddfcab7"
BN_SHA = "c79988833c4399a06cc84d15c0830f54d86f59feeec345f47a2287c1cf2ba77d"
BUILDOPT_SHA = "25a899b3745e9d66fbf630795a1202c6075e2067daffa077e4a54c28e54c7ec2"
LOADDATA_SHA = "b52b58a35dcecd80ce6d8f8033b09c20dad9f4827b808f4ea2a828ce88138c46"
TINDATA_SHA = "9e0f3322e5025e966aa18e6ac39c1d1b347c9ce2bf6e50423f301e965a6eecb7"
RUNNER_SRC_SHA = "a0d0a1b904e64ea5a3de8c0649b58454beff0322419c97ea9e6d77481cfd9f3a"   # $WS/jobs/run_cifar.sh
TREE = "harness_crt2"
RUNNER_NAME = "run_cifar_crt2.sh"

# ---- THE GRID.  (config id, --meta-stepsize token, value, --alpha0 token, value, role).  Order = the tie-break order
# ---- of the selection (an EXACT tie goes to the earlier config: crt1's selection first, then the centre). -----------
CONFIGS = (("A2", "1e-4", 1e-4, "1e-2", 1e-2, "crt1's selection, replicated on fresh seeds; alpha0 lower arm"),
           ("A3", "1e-4", 1e-4, "3e-2", 3e-2, "the centre of the plus; interior on both axes"),
           ("A4", "1e-4", 1e-4, "1e-1", 1e-1, "alpha0 upper arm; starts AT the box ceiling"),
           ("H3", "3e-4", 3e-4, "3e-2", 3e-2, "ms crossed upward at the new alpha0"),
           ("L3", "3e-5", 3e-5, "3e-2", 3e-2, "ms crossed downward at the new alpha0"))
CONFIG_IDS = tuple(c[0] for c in CONFIGS)
MS_TOKEN = dict((c[0], c[1]) for c in CONFIGS)
MS_VALUE = dict((c[0], c[2]) for c in CONFIGS)
A0_TOKEN = dict((c[0], c[3]) for c in CONFIGS)
A0_VALUE = dict((c[0], c[4]) for c in CONFIGS)
REPLICATE = "A2"
CENTRE = "A3"
# where a selected config sits in the plus (the scorer's `located`; GRID-EDGE stamps are kept, 313.5)
EDGE_OF = {"A2": "A0LO", "A4": "A0HI-CEILING", "H3": "MSHI", "L3": "MSLO"}
GRID_EDGE = tuple(sorted(EDGE_OF))

# ---- THE GRAINS: crt1's (layerwise not re-run) --------------------------------------------------------------------
GRAINS = (("ch", "chunk777", 14421),     # the audit's uniform partition
          ("nd", "nodewise", 14420),     # the audit's architecture-aligned partition
          ("k01", "scalar", 1))          # plain shared step size
GRAIN_IDS = tuple(g[0] for g in GRAINS)
SPEC_OF_GRAIN = dict((g[0], g[1]) for g in GRAINS)
M_OF_GRAIN = dict((g[0], g[2]) for g in GRAINS)
PARTITIONS = ("ch", "nd")

# ---- SEEDS: this track's block is 200-207; three are used on every arm (313.6).  No seed may be added after any run is
# ---- read (108.6a).
SEED_BLOCK = tuple(range(200, 208))
SEEDS_ALL = (200, 201, 202)

ARMS = tuple("%s%s" % (g, c) for c in CONFIG_IDS for g in GRAIN_IDS)
CONFIG_OF = dict(("%s%s" % (g, c), c) for c in CONFIG_IDS for g in GRAIN_IDS)
GRAIN_OF = dict(("%s%s" % (g, c), g) for c in CONFIG_IDS for g in GRAIN_IDS)
SPEC = dict((a, SPEC_OF_GRAIN[GRAIN_OF[a]]) for a in ARMS)
M_OF_ARM = dict((a, M_OF_GRAIN[GRAIN_OF[a]]) for a in ARMS)
MS = dict((a, MS_TOKEN[CONFIG_OF[a]]) for a in ARMS)
MSF = dict((a, MS_VALUE[CONFIG_OF[a]]) for a in ARMS)
A0 = dict((a, A0_TOKEN[CONFIG_OF[a]]) for a in ARMS)
A0F = dict((a, A0_VALUE[CONFIG_OF[a]]) for a in ARMS)
SEEDS = dict((a, SEEDS_ALL) for a in ARMS)
RUNS = tuple((a, s) for a in ARMS for s in SEEDS[a])
NJOBS = len(RUNS)                                            # 15 arms x 3 = 45
N_EXCLUSION_ROWS = NJOBS                                     # every run is at wd 5e-4 != 0.1: ONE ARGS_WD_BASE row each


def arm_of(grain, config):
    return "%s%s" % (grain, config)


def run_name(arm, seed):
    return "%s-%s-s%d" % (PREFIX, arm, seed)


def expected_args(arm, seed, save_dir):
    """The run's ARGS line as (flag, value) pairs IN ORDER -- crt1's (= cgw1's) 20 flags, every value ONE shell token."""
    return [("optimizer", "HF"), ("alg-base", "SGDm"), ("momentum-param-base", "0.99"),
            ("weight-decay-base", WD_TOKEN), ("alg-meta", "Lion"), ("momentum-param-meta", "0.99"),
            ("Lion-beta2-meta", "0.9"), ("weight-decay-meta", "0"), ("dataset", DSET), ("NN-name", NET),
            ("batch-size", str(BATCH)), ("max-time", "999:00:00"), ("gamma", GAMMA), ("meta-stepsize", MS[arm]),
            ("alpha0", A0[arm]), ("num-epochs", str(EPOCHS)), ("stepsize-groups", SPEC[arm]), ("seed", str(seed)),
            ("save-directory", save_dir), ("run-name", run_name(arm, seed))]


def args_line(arm, seed, save_dir):
    return "ARGS: " + " ".join("--%s %s" % kv for kv in expected_args(arm, seed, save_dir))


# the runner's ENV line with the PROBE_DIR VALUE replaced by <P>; identical on every arm (crt1's, verbatim)
ENV_TEMPLATE = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
                "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=5 "
                "PROBE_DIR=<P> EB_RHO=na EB_LOG=0")


def probe_dir_name(arm, seed):
    return "probe_%s" % run_name(arm, seed)


# ---- crt1's LANDED values at THIS cell (309.3, results/crt1_retune_score_mac.txt).  They feed (i) ONE non-gating
# ---- replicate stamp and a descriptive line (A2 levels), and (ii) the cross-batch LOWER alpha0 bracket of an A2
# ---- selection (crt1's IN-BATCH contrast A2 - M2, a literal).  NEVER pooled; no bar, sigma, selection or reading
# ---- reads them.
CRT1_A2 = {"ch": 91.3400, "nd": 91.1007, "k01": 91.7887}         # n 3 each, seeds 176-178
CRT1_N = 3
CRT1_A2_MINUS_M2 = {"ch": 3.5887, "nd": 3.5040, "k01": 1.2780}  # crt1 in-batch, alpha0 1e-2 minus 1e-3 at ms 1e-4
CRT1_2SE = 0.4080                                                 # crt1's in-batch 2 SE (309.3)
CRT1_T_TUNED = {"ch": 0.4487, "nd": 0.6880}
CRT1_SHRINK_A2 = {"ch": 1.1556e-5, "nd": 5.0457e-6, "k01": 5.2467e-6}   # 309.3; reproduced by selftest B
STANDARD_RECIPE_SHRINK = 5e-4    # lr 0.1 x wd 5e-4 / (1 - 0.9) (296.4)

# ---- COST (313.8).  Per-run L4 minutes at THIS cell from crt1's own landed `minutes` lines (36 runs): chunk777 52-58,
# ---- nodewise 45-51, scalar 47-49; max + 1 min start-up.
MIN_PER_RUN_L4 = {"ch": 59, "nd": 52, "k01": 50}
EXPECTED_GPU_H = sum(MIN_PER_RUN_L4[GRAIN_OF[a]] for a, _s in RUNS) / 60.0      # 40.25
HARD_BOUND_GPU_H = NJOBS * 2.0                                                 # WALL x NJOBS = 90

# ---- PREDICTED LEVELS PER ARM (313.9).  JUDGEMENTS, UNSURE, written before any run exists.  A2 at crt1's A2 +/- 0.5;
# ---- nothing on this network has run alpha0 3e-2 or 1e-1, so those bands are wide.  (lo, hi) plateau5.
PRED = {
    "chA2": (90.8, 91.9), "ndA2": (90.6, 91.6), "k01A2": (91.3, 92.3),
    "chA3": (90.3, 92.8), "ndA3": (90.1, 92.6), "k01A3": (90.6, 92.6),
    "chA4": (88.0, 92.6), "ndA4": (88.0, 92.4), "k01A4": (88.5, 92.4),
    "chH3": (89.8, 92.6), "ndH3": (89.6, 92.4), "k01H3": (90.3, 92.4),
    "chL3": (89.8, 92.7), "ndL3": (89.6, 92.5), "k01L3": (90.0, 92.4),
}
BRANCHES = ("NOT-A-TUNING-ARTEFACT", "SELECTION-DEPENDENT", "WEAKENED-TO-TIE", "TIE-SCALAR-LEAD-BOUNDED",
            "RETUNE-UNDECIDED", "HEADLINE-REFUTED-BY-RETUNE", "BRANCH-UNREADABLE", "BRANCH-SIGMA-INCONSISTENT")
PRIOR_P = {"NOT-A-TUNING-ARTEFACT": 0.15, "SELECTION-DEPENDENT": 0.05, "WEAKENED-TO-TIE": 0.25,
           "TIE-SCALAR-LEAD-BOUNDED": 0.12, "RETUNE-UNDECIDED": 0.25, "HEADLINE-REFUTED-BY-RETUNE": 0.10,
           "BRANCH-UNREADABLE": 0.04, "BRANCH-SIGMA-INCONSISTENT": 0.04}     # judgements
PRIOR_LOC_ALL = 0.45    # judgement: every grain's selection located (LOC-ALL)
CHANCE = 10.0


def manifest_text():
    L = ["BATCH %s" % PREFIX, "NETWORK %s" % NET, "DATASET %s" % DSET, "NUM_PARAM_TENSORS %d" % NTENS,
         "TOTAL_PARAMS %d" % TOTPAR, "CLIP_C %s" % CLIP, "WD %s" % WD_TOKEN, "EPOCHS %d" % EPOCHS,
         "META_STEPS %d" % META_STEPS, "PROBE %d N_RECORDS %d" % (PROBE, N_RECORDS),
         "GRID %s" % ",".join("%s=ms%s/a0%s" % (c, MS_TOKEN[c], A0_TOKEN[c]) for c in CONFIG_IDS)]
    for g, spec, m in GRAINS:
        L.append("GRAIN %s SPEC %s M %d" % (g, spec, m))
    for a in ARMS:
        L.append("ARM %s CONFIG %s MS %s ALPHA0 %s SPEC %s M %d SEEDS %s" % (
            a, CONFIG_OF[a], MS[a], A0[a], SPEC[a], M_OF_ARM[a], ",".join(str(s) for s in SEEDS[a])))
    L.append("NJOBS %d" % NJOBS)
    return "\n".join(L) + "\n"


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    print(manifest_text())
    print("every run owes ONE ARGS_WD_BASE exclusion row at landing (%d rows); expected %.2f GPU-h, hard bound %.0f"
          % (N_EXCLUSION_ROWS, EXPECTED_GPU_H, HARD_BOUND_GPU_H))
