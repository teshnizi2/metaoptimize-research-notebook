#!/usr/bin/env python3
"""cvl1_design.py -- THE FROZEN DESIGN TABLE OF `cvl1` (CORRECTIONS 303; row 1.14 of docs/ICML-PLAN.md):
DO THE AUDIT'S RANKINGS, REPORTED ON TEST, HOLD ON A HELD-OUT VALIDATION SPLIT -- AND WOULD SELECTING ON VALIDATION
HAVE PICKED THE SAME CONFIGURATION?

WHY THIS BATCH EXISTS.  Every number of the count-matched partition audit is plateau5 = mean TEST accuracy over epochs
95-99, and the harness has no validation split (CORRECTIONS 135.1; PLAN.md B3).  What was SELECTED on test is recorded
at CORRECTIONS 302.1: the core cell's meta step size ms 1e-4 was taken as the plateau5-TEST argmax of a meta-step ladder
(FINDINGS 38.1 / 72.4; `tw0`'s guard 2b "ms=1e-4 is the argmax for weightwise, layerwise AND scalar"), and nodewise's
own test argmax is 3e-4, not 1e-4 (CORRECTIONS 108).  The partitions (chunk777's K is fixed by COUNT-MATCHING, m 14,421
v 14,420) and the plateau window (fixed a priori by PLAN.md B3) were not selected on test.  The TMLR headline now rests on
`cgw1` (CORRECTIONS 296), whose three load-bearing readings are TEST readings:
    D_W1 = chunk777 - nodewise at wd 0.1   +0.3693  SURVIVES
    D_W4 = chunk777 - nodewise at wd 5e-4  +0.2410  UNDECIDED
    T_W4 = scalar - both partitions at 5e-4  +2.559 / +2.800  SCALAR-BEATS-BEST
`cvl1` re-runs that cell at wd 0.1 and 5e-4 with PATCH_VALSPLIT on (VAL_SPLIT=5000:302: 5,000 class-stratified training
images held out by a fixed split seed independent of the run seed; the model trains on the other 45,000) and reads EVERY
contrast twice from the SAME runs: on TEST (10,000 images) and on VAL (the 5,000 held out).  Nothing is selected on VAL
before the scorer runs; the scorer's selection reading is an argmax it computes itself.

THE CELL is cgw1's VERBATIM (ResNet18 / CIFAR-10 / bs 100 / SGDm 0.99 / Lion 0.99 / 0.9 / meta wd 0 / gamma 1 / ms 1e-4 /
alpha0 1e-3 / 100 epochs / AUGMENT=1 / BETA_CLIP=-15:-2.3026 / PROBE=5), the same 20 ARGS flags in the same order, at
two of cgw1's rungs (W1 0.1, W4 5e-4) and cgw1's four grains; the ONLY additions are the ENV item VAL_SPLIT=5000:302 and
the tree harness_cvl1 (cgw1's bytes + PATCH_VALSPLIT on load_data.py / train.py; HF.py untouched, 4732b74a...).

WHAT IT CANNOT SAY (bounds printed on every FINAL): ONE-CELL; TWO-RUNGS; MS-ALPHA0-NOT-RESELECTED (the hyperparameter
that WAS chosen on test, ms, is not re-chosen here -- that needs an ms ladder, ICML-PLAN 4a rank 2); TRAIN-45K (every
model here saw 45,000 images and 45,000 steps, so in-batch TEST levels are not cgw1's and are compared with them only
through a non-gating stamp); ONE-SPLIT (one fixed 5,000-image split: split-to-split variance is not measured);
ALPHA-INDEPENDENT-DECAY-NOT-TESTED; FLOOR-READINGS-ARE-BOUNDS.

Stdlib only (no torch).  Imported, UNEDITED, by analysis/cVL1_valsplit_score.py, bin/cVL1_valsplit.sh and
analysis/cvl1_live_check.py.   python3 analysis/cvl1_design.py  # prints the manifest
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PREFIX = "cvl1"
NET = "ResNet18"
DSET = "CIFAR10"
EPOCHS = 100
BATCH = 100
N_TRAIN_FULL = 50000
N_VAL = 5000
SPLIT_SEED = 302
VAL_SPLIT = "%d:%d" % (N_VAL, SPLIT_SEED)     # "5000:302" -- the registered ENV value, ONE token
N_TRAIN = N_TRAIN_FULL - N_VAL                # 45,000
N_CLASSES = 10
PER_CLASS = N_VAL // N_CLASSES                # 500
STEPS_PER_EPOCH = N_TRAIN // BATCH            # 450
PROBE = 5
N_RECORDS = EPOCHS * STEPS_PER_EPOCH // PROBE   # 9,000 (records at counter % 5 == 0)
CLIP = "-15:-2.3026"
LO, HI = -15.0, -2.3026
MST = "1e-4"
A0 = "1e-3"
AUG = "1"
GAMMA = "1"
WALL = "02:00:00"
GPU_CONSTRAINT = "L4"
GPU_NAME = "NVIDIA L4"
NTENS = 62
TOTPAR = 11173962

# ---- the tree: cgw1's bytes + PATCH_VALSPLIT (bin/cVL1_stage_harness.sh; proof job CORRECTIONS 302) ----------------------
HF_LIVE_SHA = "4732b74aa3a10508896e92eccced5c89051c353fa8a01a3af21aab9aeda0cecd"      # UNPATCHED (== cgw1's)
TRAIN_PRE_SHA = "3fea309e172609c1ee1be143d5f1847404426df31869a63c4cefbb1ebddfcab7"
LOADDATA_PRE_SHA = "b52b58a35dcecd80ce6d8f8033b09c20dad9f4827b808f4ea2a828ce88138c46"
TRAIN_POST_SHA = "8706d8c752ff8971818cb3b6625af1d77dbeed9a79f0560027f628f19be8b0f5"      # PATCH_VALSPLIT
LOADDATA_POST_SHA = "c8eb83859312487162f8a41a132388f4f560403ae99b68865882f5ff00ad2c03"   # PATCH_VALSPLIT
BN_SHA = "c79988833c4399a06cc84d15c0830f54d86f59feeec345f47a2287c1cf2ba77d"
BUILDOPT_SHA = "25a899b3745e9d66fbf630795a1202c6075e2067daffa077e4a54c28e54c7ec2"
TINDATA_SHA = "9e0f3322e5025e966aa18e6ac39c1d1b347c9ce2bf6e50423f301e965a6eecb7"
RUNNER_SRC_SHA = "a0d0a1b904e64ea5a3de8c0649b58454beff0322419c97ea9e6d77481cfd9f3a"   # $WS/jobs/run_cifar.sh
TREE = "harness_cvl1"
RUNNER_NAME = "run_cifar_cvl1.sh"
# the split every run must print (the REAL CIFAR-10 labels): val_sha from proof job 5080195's V3 (CPU) AND RR2 (GPU, run
# seeds 184 / 185), train_sha's first 16 hex from RR2, the full value re-derived from the tree's own helper on the
# data_batch_1..5 labels (CORRECTIONS 302.4)
VAL_SHA = "7d3a1489390161d637ad0b526ac32a10723210722879f8deead4462e4f69bb0e"
TRAIN_SPLIT_SHA = "2733a990cf7a76d8e92014cdd6aceeb8c055f7cd49cc7df913923b066c1e5e91"


def witness_on():
    return ("VAL_SPLIT: on dataset=%s n_val=%d n_train=%d classes=%d per_class=%d split_seed=%d val_sha=%s train_sha=%s"
            % (DSET, N_VAL, N_TRAIN, N_CLASSES, PER_CLASS, SPLIT_SEED, VAL_SHA, TRAIN_SPLIT_SHA))


# ---- THE RUNGS (cgw1's W1 and W4 tokens, verbatim) ---------------------------------------------------------------------
RUNGS = (("W1", "0.1", 0.1, "the audit's own value; cgw1 D_W1 SURVIVES on TEST"),
         ("W4", "5e-4", 0.0005, "the standard CIFAR value; cgw1 D_W4 UNDECIDED and SCALAR-BEATS-BEST on TEST"))
RUNG_IDS = tuple(r[0] for r in RUNGS)
WD_TOKEN = dict((r[0], r[1]) for r in RUNGS)
WD_VALUE = dict((r[0], r[2]) for r in RUNGS)
STANDARD_WD_TOKEN = "0.1"      # corpus_exclusions ARGS_KINDS standard for ARGS_WD_BASE (CORRECTIONS 263)

GRAINS = (("ch", "chunk777", 14421), ("nd", "nodewise", 14420), ("k01", "scalar", 1), ("kL", "layerwise", 62))
GRAIN_IDS = tuple(g[0] for g in GRAINS)
SPEC_OF_GRAIN = dict((g[0], g[1]) for g in GRAINS)
M_OF_GRAIN = dict((g[0], g[2]) for g in GRAINS)

# ---- SEEDS: Track D's assigned block {184..191} (verified free at 303.3: 0 corpus rows and 0 .out ARGS lines under
# ---- $WS/runs carry any of 184-191; max corpus seed 162).  4 per arm (32 = the task's ceiling); 188-191 UNUSED.
# ---- No seed may be added after any run is read (108.6a).
SEED_BLOCK = tuple(range(184, 192))
SEEDS_OF_RUNG = {"W1": (184, 185, 186, 187), "W4": (184, 185, 186, 187)}
UNUSED_SEEDS = (188, 189, 190, 191)

ARMS = tuple("%s%s" % (g, r) for r in RUNG_IDS for g in GRAIN_IDS)
RUNG_OF = dict(("%s%s" % (g, r), r) for r in RUNG_IDS for g in GRAIN_IDS)
GRAIN_OF = dict(("%s%s" % (g, r), g) for r in RUNG_IDS for g in GRAIN_IDS)
SPEC = dict((a, SPEC_OF_GRAIN[GRAIN_OF[a]]) for a in ARMS)
M_OF_ARM = dict((a, M_OF_GRAIN[GRAIN_OF[a]]) for a in ARMS)
WD = dict((a, WD_TOKEN[RUNG_OF[a]]) for a in ARMS)
WDF = dict((a, WD_VALUE[RUNG_OF[a]]) for a in ARMS)
SEEDS = dict((a, SEEDS_OF_RUNG[RUNG_OF[a]]) for a in ARMS)
RUNS = tuple((a, s) for a in ARMS for s in SEEDS[a])
NJOBS = len(RUNS)                                            # 16 + 16 = 32
ARGS_DEVIATING = tuple(a for a in ARMS if WD[a] != STANDARD_WD_TOKEN)
# EVERY run prints `VAL_SPLIT: on` (a train-set intervention no CSV column carries): all 32 owe an exclusion row at
# landing; the 16 W4 runs are TWO-AXIS (ON line + ARGS_WD_BASE).  corpus_exclusions.py has NO VAL_SPLIT kind yet:
# that code gap is OWED before the ingest (303.9) and is NOT closed here.
N_EXCLUSION_ROWS = NJOBS


def run_name(arm, seed):
    return "%s-%s-s%d" % (PREFIX, arm, seed)


def expected_args(arm, seed, save_dir):
    """The run's ARGS line as (flag, value) pairs IN ORDER -- cgw1's 20 flags, every value ONE shell token."""
    return [("optimizer", "HF"), ("alg-base", "SGDm"), ("momentum-param-base", "0.99"),
            ("weight-decay-base", WD[arm]), ("alg-meta", "Lion"), ("momentum-param-meta", "0.99"),
            ("Lion-beta2-meta", "0.9"), ("weight-decay-meta", "0"), ("dataset", DSET), ("NN-name", NET),
            ("batch-size", str(BATCH)), ("max-time", "999:00:00"), ("gamma", GAMMA), ("meta-stepsize", MST),
            ("alpha0", A0), ("num-epochs", str(EPOCHS)), ("stepsize-groups", SPEC[arm]), ("seed", str(seed)),
            ("save-directory", save_dir), ("run-name", run_name(arm, seed))]


def args_line(arm, seed, save_dir):
    return "ARGS: " + " ".join("--%s %s" % kv for kv in expected_args(arm, seed, save_dir))


# the runner's ENV line (cgw1's template; the runner cannot print VAL_SPLIT -- the patch's witness line carries it)
ENV_TEMPLATE = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
                "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=5 "
                "PROBE_DIR=<P> EB_RHO=na EB_LOG=0")


def probe_dir_name(arm, seed):
    return "probe_%s" % run_name(arm, seed)


# ---- cgw1's LANDED TEST readings (CORRECTIONS 296.3; trained on 50,000).  ONE non-gating stamp reads them. ---------------
CGW1_TEST = {"chW1": 92.4287, "ndW1": 92.0593, "k01W1": 92.1907, "kLW1": 92.7933,
             "chW4": 87.9060, "ndW4": 87.6650, "k01W4": 90.4650, "kLW4": 89.8110}
CGW1_STATES = {"W1": "SURVIVES", "W4": "UNDECIDED", "SCALAR": "SCALAR-BEATS-BEST"}

# ---- COST.  cgw1's per-grain L4 minutes at this cell (chunk777 53, nodewise 45, scalar 43, layerwise 35, incl. start-up)
# ---- x 0.9 (450 of 500 steps per epoch) + 1 min per run for 100 VAL passes over 5,000 images (a test pass over 10,000
# ---- is ~1-2 s on L4; 100 x 5,000 is ~1 min, a generous allowance).
MIN_PER_RUN_L4 = {"ch": 53 * 0.9 + 1, "nd": 45 * 0.9 + 1, "k01": 43 * 0.9 + 1, "kL": 35 * 0.9 + 1}
EXPECTED_GPU_H = sum(MIN_PER_RUN_L4[GRAIN_OF[a]] for a, _s in RUNS) / 60.0      # 21.65
HARD_BOUND_GPU_H = NJOBS * 2.0                                                 # WALL x NJOBS = 64

# ---- PREDICTIONS (303.5), written before any run exists.  PRIORS, UNSURE.  (lo, hi) plateau5 per arm, TEST and VAL.
# ---- Anchored on cgw1's landed TEST levels, lowered 0.2-0.6 pp for 45,000 training images; VAL predicted within
# ---- +/- 0.8 pp of TEST (a different 5,000-image sample of the same distribution; not a harder or easier set by design).
PRED = {
    "chW1": (91.6, 92.7), "ndW1": (91.2, 92.3), "k01W1": (91.3, 92.5), "kLW1": (92.0, 93.0),
    "chW4": (86.6, 88.4), "ndW4": (86.3, 88.2), "k01W4": (89.5, 90.9), "kLW4": (88.8, 90.3),
}
PRED_VAL_MINUS_TEST = (-0.8, 0.8)
ACCOUNTS = ("VAL-AGREES", "VAL-SHIFTS", "VAL-DIFFERS-UNRESOLVED", "VAL-FLIPS", "VAL-UNREADABLE")
# judgements (303.5).  VAL-DIFFERS-UNRESOLVED is the likeliest single outcome BY CONSTRUCTION: the state bands are
# 0.15 / 0.30 pp wide and D_W1 (+0.37 on cgw1's test) sits near the SURVIVES edge, so one reader can cross it on noise.
PRIOR_P = {"VAL-AGREES": 0.30, "VAL-SHIFTS": 0.13, "VAL-DIFFERS-UNRESOLVED": 0.45, "VAL-FLIPS": 0.04, "VAL-UNREADABLE": 0.08}
PRIOR_SELECT = {"SELECT-SAME": 0.75, "SELECT-DIFFERS-UNRESOLVED": 0.22, "SELECT-DIFFERS-RESOLVED": 0.03}
CHANCE = 10.0


def manifest_text():
    L = ["BATCH %s" % PREFIX, "NETWORK %s" % NET, "DATASET %s" % DSET, "NUM_PARAM_TENSORS %d" % NTENS,
         "TOTAL_PARAMS %d" % TOTPAR, "CLIP_C %s" % CLIP, "MS %s ALPHA0 %s" % (MST, A0), "EPOCHS %d" % EPOCHS,
         "VAL_SPLIT %s N_TRAIN %d N_VAL %d PER_CLASS %d" % (VAL_SPLIT, N_TRAIN, N_VAL, PER_CLASS),
         "META_STEPS %d" % (EPOCHS * STEPS_PER_EPOCH), "PROBE %d N_RECORDS %d" % (PROBE, N_RECORDS),
         "WD_LADDER %s" % ",".join("%s=%s" % (r, WD_TOKEN[r]) for r in RUNG_IDS)]
    for g, spec, m in GRAINS:
        L.append("GRAIN %s SPEC %s M %d" % (g, spec, m))
    for a in ARMS:
        L.append("ARM %s RUNG %s WD %s SPEC %s M %d SEEDS %s" % (a, RUNG_OF[a], WD[a], SPEC[a], M_OF_ARM[a],
                                                                ",".join(str(s) for s in SEEDS[a])))
    L.append("NJOBS %d" % NJOBS)
    return "\n".join(L) + "\n"


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    print(manifest_text())
    print("EXPECTED_GPU_H %.2f  HARD_BOUND_GPU_H %.0f" % (EXPECTED_GPU_H, HARD_BOUND_GPU_H))
    print("every run owes an exclusion row (%d; %d of them two-axis at wd 5e-4)" % (N_EXCLUSION_ROWS, len(ARGS_DEVIATING) * 4))
