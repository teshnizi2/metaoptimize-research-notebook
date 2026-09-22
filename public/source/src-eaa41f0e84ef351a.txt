#!/usr/bin/env python3
"""cai1_design.py -- THE FROZEN DESIGN TABLE OF `cai1` (CORRECTIONS 306; row 1.6's alpha-independent half, and the
stamp DECOUPLED-NOT-TESTED, of docs/ICML-PLAN.md):  THE COUNT-MATCHED PARTITION AUDIT UNDER ALPHA-INDEPENDENT DECAY.

WHY THIS BATCH EXISTS.  Every campaign run -- every one of the audit's 20 count-matched cells, and `cgw1`'s three rungs
(CORRECTIONS 296) -- uses the harness's ALPHA-SCALED decay:  delta = a*(m + wd*w);  w <- w - delta;
h <- gamma*(1 - wd*a)*h - delta  (a = exp(beta), the LEARNED step size).  Its realised per-step shrink a*wd moves with
the learned a, and at cgw1's 5e-4 it is 115-883x below a standard SGD recipe's (296.4).  The TMLR paper's biggest open
referee point is therefore: is the audit's sign (uniform chunk777 beats aligned nodewise at matched count) -- and
cgw1's SCALAR-BEATS-BEST at 5e-4 -- a property of that non-standard decay?  `cai1` re-runs cgw1's cell with the
alpha-scaled decay REMOVED (--weight-decay-base 0) and REPLACED by alpha-INDEPENDENT decay (PATCH_DECAYROUTE,
CORRECTIONS 305: DECAY_ROUTE=alpha_indep:<LAMBDA>, w <- w - LAMBDA*w - a*u, trace h <- gamma*(1 - LAMBDA)*h - a*u,
Loshchilov & Hutter's SGDW form, arXiv:1711.05101, with a constant schedule multiplier), at two LAMBDA levels taken from
a STANDARD RECIPE, with the four grains cgw1 ran.

LAMBDA, CHOSEN FROM A STANDARD RECIPE (306.3).  The standard CIFAR SGD recipe (lr 0.1, wd 5e-4, momentum 0.9) shrinks
the weights per step by lr*wd = 5e-5 before momentum amplification, and by about lr*wd/(1 - 0.9) = 5e-4 in steady state
once the 0.9 buffer carries the decay term (the two bases 296.4 / 298 quote side by side).  The campaign has itself
used BOTH bases for the same caveat, so a referee may pick either; and the two differ 10x in shrink and ~3.2x in the
equilibrium rotation rate sqrt(2*LAMBDA) of a normalised tensor (Kosson et al. arXiv:2305.17212, the mechanism only;
the number is this design's own arithmetic).  ONE level cannot decide the question: a null at one end could be answered
"wrong LAMBDA" from the other.  So BOTH endpoints are registered, as a BRACKET -- `I5` = 5e-5 and `I4` = 5e-4 -- and
32 runs, not 16.  NOT RUN: LAMBDA = alpha0*wd = 1e-3 * 5e-4 = 5e-7 (the match on INITIAL shrink to cgw1's W4, row 1.6's
wording): its horizon 1/LAMBDA = 2e6 steps is 40x the 50,000-step run, so it is essentially undecayed training and
answers the route question, not the standard-recipe question the referee asks (bound SHRINK-MATCHED-NOT-RATIO-MATCHED).

THE CELL is cgw1's VERBATIM (ResNet18 / CIFAR-10 / bs 100 / SGDm 0.99 / Lion 0.99 / 0.9 / meta wd 0 / gamma 1 /
ms 1e-4 / alpha0 1e-3 / 100 epochs / AUGMENT=1 / BETA_CLIP=-15:-2.3026 / PROBE=5), the same 20 ARGS flags in the same
order, EXCEPT `--weight-decay-base 0` on every arm (the patch REFUSES alpha_indep at any other value: the alpha-scaled
decay is replaced, never stacked).  The ONLY other differences: the tree harness_cdr1 (cgw1's bytes + PATCH_DECAYROUTE
on HF.py only; train.py / load_data.py / build_network.py / build_optimizer.py unchanged) and ONE --export item
DECAY_ROUTE=alpha_indep:<LAMBDA> per rung.  The two rungs' ARGS lines are therefore IDENTICAL at the same grain and
seed except `--run-name`: the rung is witnessed by the harness's own `DECAY_ROUTE: on ... lambda=<repr>` line (printed
at construction on every run) and by every probe record's `dr_lam` and MEASURED shrink `dr_shrink_meas`, which the
scorer gates run by run.

WHAT IT CANNOT SAY (bounds printed on every FINAL): ONE-CELL; TWO-LAMBDAS-BRACKET (they bracket the standard recipe;
they locate nothing between); ALPHA-SCALED-REFERENCE-BETWEEN-BATCH (no alpha-scaled arm is in this batch: "the effect
exists under alpha-scaled decay" is cgw1's in-batch W1 / W2 SURVIVES, 296, read between batches; the two batches are
never pooled); ROTATIONAL-EQUILIBRIUM-CONFOUND (on normalised tensors alpha-independent decay drives the effective
step toward a rate set by LAMBDA rather than by the learned step size (arXiv:2305.17212), so a VANISHES / TIES here can
mean "the partition no longer controls the effective step", not "the audit effect was a harness artefact"; the
alpha-independent arm is not a pure route control, 305 bound 4); SHRINK-MATCHED-NOT-RATIO-MATCHED (LAMBDA matches a
standard recipe's per-step SHRINK; the learned step size a ~ 1e-3 is ~100x below the recipe's lr, so LAMBDA/a is
~100-1000x the recipe's decay-to-step ratio); CONSTANT-SCHEDULE (no lr schedule, no warm-up, as every campaign run);
FLOOR-READINGS-ARE-BOUNDS.

Stdlib only (no torch).  Imported, UNEDITED, by analysis/cAI1_alphaindep_score.py, bin/cAI1_alphaindep.sh and
analysis/cai1_live_check.py.   python3 analysis/cai1_design.py  # prints the manifest
"""
import hashlib
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PREFIX = "cai1"
NET = "ResNet18"
DSET = "CIFAR10"
EPOCHS = 100
BATCH = 100
STEPS_PER_EPOCH = 50000 // BATCH             # 500
PROBE = 5
N_RECORDS = EPOCHS * STEPS_PER_EPOCH // PROBE   # 10,000 (records at counter % 5 == 0)
CLIP = "-15:-2.3026"
LO, HI = -15.0, -2.3026
MST = "1e-4"
A0 = "1e-3"
AUG = "1"
GAMMA = "1"
WD_TOKEN = "0"          # --weight-decay-base on EVERY arm: the alpha-scaled decay is REMOVED (the patch requires it)
WDF = 0.0
WALL = "02:00:00"
GPU_CONSTRAINT = "L4"
GPU_NAME = "NVIDIA L4"
NTENS = 62
TOTPAR = 11173962

# ---- the tree: harness_cdr1 (CORRECTIONS 305), cgw1's pinned bytes + PATCH_DECAYROUTE on HF.py only ----------------------
HF_POST_SHA = "17ee0a2862b31b48c4196a75e88d60cf19f49d0f993cecc2750af832fe44b020"    # PATCH_DECAYROUTE
HF_PRE_SHA = "4732b74aa3a10508896e92eccced5c89051c353fa8a01a3af21aab9aeda0cecd"     # HF.py.pre_decayroute == cgw1's HF
TRAIN_SHA = "3fea309e172609c1ee1be143d5f1847404426df31869a63c4cefbb1ebddfcab7"      # unpatched, == cgw1's
BN_SHA = "c79988833c4399a06cc84d15c0830f54d86f59feeec345f47a2287c1cf2ba77d"
BUILDOPT_SHA = "25a899b3745e9d66fbf630795a1202c6075e2067daffa077e4a54c28e54c7ec2"
LOADDATA_SHA = "b52b58a35dcecd80ce6d8f8033b09c20dad9f4827b808f4ea2a828ce88138c46"
TINDATA_SHA = "9e0f3322e5025e966aa18e6ac39c1d1b347c9ce2bf6e50423f301e965a6eecb7"
RUNNER_SRC_SHA = "a0d0a1b904e64ea5a3de8c0649b58454beff0322419c97ea9e6d77481cfd9f3a"   # $WS/jobs/run_cifar.sh
RUNNER_SHA = "d3d3bdd781282416316be2eab91a70cf796e2d725a24df4c153696146b81083c"      # $WS/jobs/run_cifar_cdr1.sh
STAGE_MANIFEST_SHA = "7f82f0a1c25240d6fe395c576f393a33d00a2cd1d5a385af75e47bbffea752ff"
PROOF_LOG_SHA = "dc73dfbabf25fa01f2d0669a46fdb3202502e0e4a5851d4b4cce99146c8f25b1"   # job 5081090, 410 PASS / 0 FAIL
PATCH_SHA = "12c107884db4c9285a012ef06c4bdcb3f5f6528c6c3da0c16421944718be4c95"
TEST_SHA = "7044b1c76153077fe2a01884f07b97707081bceb62c4e55e70b8c876d2175d69"
DRIVER_SHA = "466d2b4ad86508122fe31e821ada8ab4e25dccf1a16786ba7643524c96b6c474"
TREE = "harness_cdr1"
RUNNER_NAME = "run_cifar_cdr1.sh"

# ---- THE TWO LAMBDA RUNGS (306.3).  (rung id, the --export token's LAMBDA text, its float, why) ------------------------
RUNGS = (("I5", "5e-5", 5e-05, "lr*wd of the standard recipe (0.1 x 5e-4) before momentum amplification"),
         ("I4", "5e-4", 5e-04, "lr*wd/(1 - 0.9): the standard recipe's steady-state shrink with its 0.9 momentum buffer"))
RUNG_IDS = tuple(r[0] for r in RUNGS)
LAM_TEXT = dict((r[0], r[1]) for r in RUNGS)
LAM = dict((r[0], r[2]) for r in RUNGS)


def route_value(rung):
    """The ONE --export token's value: DECAY_ROUTE=<this>."""
    return "alpha_indep:%s" % LAM_TEXT[rung]


def witness_on(rung):
    """The line PATCH_DECAYROUTE's _dr_init prints on every run of this rung (format: patches/patch_decayroute.py)."""
    lam = LAM[rung]
    return ("DECAY_ROUTE: on mode=alpha_indep base=SGDm wd=%r lambda=%r lambda_f32=%r gamma=%r"
            % (WDF, lam, struct.unpack("f", struct.pack("f", lam))[0], float(GAMMA)))


GRAINS = (("ch", "chunk777", 14421), ("nd", "nodewise", 14420), ("k01", "scalar", 1), ("kL", "layerwise", 62))
GRAIN_IDS = tuple(g[0] for g in GRAINS)
SPEC_OF_GRAIN = dict((g[0], g[1]) for g in GRAINS)
M_OF_GRAIN = dict((g[0], g[2]) for g in GRAINS)

# ---- SEEDS: Track F2's assigned block {192..195} (verified free at 306.4: 0 corpus rows, 0 .out ARGS lines under
# ---- $WS/runs, 0 sacct / squeue jobs carry any of them; max corpus seed 162).  4 per arm, all 4 used.
# ---- No seed may be added after any run is read (108.6a).
SEED_BLOCK = (192, 193, 194, 195)
SEEDS_OF_RUNG = {"I5": (192, 193, 194, 195), "I4": (192, 193, 194, 195)}
UNUSED_SEEDS = ()

ARMS = tuple("%s%s" % (g, r) for r in RUNG_IDS for g in GRAIN_IDS)
RUNG_OF = dict(("%s%s" % (g, r), r) for r in RUNG_IDS for g in GRAIN_IDS)
GRAIN_OF = dict(("%s%s" % (g, r), g) for r in RUNG_IDS for g in GRAIN_IDS)
SPEC = dict((a, SPEC_OF_GRAIN[GRAIN_OF[a]]) for a in ARMS)
M_OF_ARM = dict((a, M_OF_GRAIN[GRAIN_OF[a]]) for a in ARMS)
LAM_OF_ARM = dict((a, LAM[RUNG_OF[a]]) for a in ARMS)
SEEDS = dict((a, SEEDS_OF_RUNG[RUNG_OF[a]]) for a in ARMS)
RUNS = tuple((a, s) for a in ARMS for s in SEEDS[a])
NJOBS = len(RUNS)                                            # 16 + 16 = 32
# EVERY run prints `DECAY_ROUTE: on` (a decay-route intervention no CSV column carries) AND runs --weight-decay-base 0
# (an ARGS_WD_BASE deviation from 263's standard 0.1): all 32 owe a TWO-AXIS exclusion row at landing.
# corpus_exclusions.py has NO DECAY_ROUTE kind yet (305.6 d): that code gap is OWED before the ingest and is NOT
# closed here.
N_EXCLUSION_ROWS = NJOBS


def run_name(arm, seed):
    return "%s-%s-s%d" % (PREFIX, arm, seed)


def expected_args(arm, seed, save_dir):
    """The run's ARGS line as (flag, value) pairs IN ORDER -- cgw1's 20 flags, --weight-decay-base 0."""
    return [("optimizer", "HF"), ("alg-base", "SGDm"), ("momentum-param-base", "0.99"),
            ("weight-decay-base", WD_TOKEN), ("alg-meta", "Lion"), ("momentum-param-meta", "0.99"),
            ("Lion-beta2-meta", "0.9"), ("weight-decay-meta", "0"), ("dataset", DSET), ("NN-name", NET),
            ("batch-size", str(BATCH)), ("max-time", "999:00:00"), ("gamma", GAMMA), ("meta-stepsize", MST),
            ("alpha0", A0), ("num-epochs", str(EPOCHS)), ("stepsize-groups", SPEC[arm]), ("seed", str(seed)),
            ("save-directory", save_dir), ("run-name", run_name(arm, seed))]


def args_line(arm, seed, save_dir):
    return "ARGS: " + " ".join("--%s %s" % kv for kv in expected_args(arm, seed, save_dir))


# the runner's ENV line (cgw1's template; the runner cannot print DECAY_ROUTE -- the patch's witness line carries it)
ENV_TEMPLATE = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
                "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=5 "
                "PROBE_DIR=<P> EB_RHO=na EB_LOG=0")


def probe_dir_name(arm, seed):
    return "probe_%s" % run_name(arm, seed)


# ---- cgw1's LANDED readings at alpha-scaled decay (CORRECTIONS 296.3), the BETWEEN-BATCH reference.  They are quoted in
# ---- licence text and feed non-gating stamps only; no bar, sigma, state or contrast reads them. -------------------------
CGW1_D = {"W1": 0.3693, "W2": 0.4533, "W4": 0.2410}
CGW1_STATES = {"W1": "SURVIVES", "W2": "SURVIVES", "W4": "UNDECIDED", "SCALAR_W4": "SCALAR-BEATS-BEST"}
CGW1_T_W4 = {"ch": 2.5590, "nd": 2.8000}
SGDM_POOL_D = 0.5556            # MASTER-TABLE line 175 / CORRECTIONS 211.2, fixed-effect, 8 cells

# ---- COST.  cgw1's per-grain L4 minutes at this cell (53 / 45 / 43 / 35, incl. start-up) x 1.15: an ALLOWANCE for the
# ---- patch's float64 shrink measurement at every probe step (every 5th step, 62 tensors); UNSURE, not measured at 100 ep.
MIN_PER_RUN_L4 = {"ch": 53 * 1.15, "nd": 45 * 1.15, "k01": 43 * 1.15, "kL": 35 * 1.15}
EXPECTED_GPU_H = sum(MIN_PER_RUN_L4[GRAIN_OF[a]] for a, _s in RUNS) / 60.0      # 26.99
HARD_BOUND_GPU_H = NJOBS * 2.0                                                 # WALL x NJOBS = 64

# ---- PREDICTIONS (306.6), written before any run exists.  PRIORS, UNSURE: no MetaOptimize run has EVER used
# ---- alpha-independent decay.  (lo, hi) plateau5 TEST per arm.  Anchors: (i) cgw1's W1 levels (92.06-92.79, alpha-scaled
# ---- 0.1, realised shrink 1e-4 at init falling to ~2e-6) and W4 levels (87.7-90.5, nearly undecayed); (ii) rotational
# ---- equilibrium: under alpha-independent decay a normalised tensor settles at a rotation of ~sqrt(2*LAMBDA) per step
# ---- whatever its step size -- ~0.010 at I5 (reached in ~1/(2*LAMBDA) = 10,000 steps = 20 epochs) and ~0.032 at I4
# ---- (1,000 steps), with no schedule to anneal it, so I4 is predicted noisier and lower, and its low edge is BELOW
# ---- HEALTH_MIN 85 (disclosed: UNHEALTHY at I4 is a reachable, registered outcome, inside the 0.12 one-rung prior).
PRED = {
    "chI5": (89.0, 92.5), "ndI5": (89.0, 92.5), "k01I5": (89.0, 92.5), "kLI5": (89.0, 92.5),
    "chI4": (84.0, 91.0), "ndI4": (84.0, 91.0), "k01I4": (84.0, 91.0), "kLI4": (84.0, 91.0),
}
PRED_LO_BELOW_HEALTH = ("chI4", "ndI4", "k01I4", "kLI4")      # disclosed, not hidden (the 164.6 gate's form)
# the predicted D band per rung under each account (the scorer's own bars)
PRED_D = {"SURVIVES": (0.30, 0.80), "VANISHES": (-0.15, 0.15), "REVERSES": (-0.80, -0.15)}
ACCOUNTS = ("AI-SURVIVES", "AI-ARTEFACT-VANISHES", "AI-ARTEFACT-REVERSES", "AI-LAMBDA-DEPENDENT",
            "AI-SURVIVES-ONE-LAMBDA", "AI-ABSENT-ONE-LAMBDA", "AI-UNDECIDED", "AI-ONE-RUNG-OR-UNREADABLE")
# judgements (306.6).  Rotational equilibrium predicts the partition contrast SHRINKS under alpha-independent decay
# (normalised layers are driven to one gradient-to-weight ratio whatever their step size), so ABSENT is favoured over
# SURVIVES -- and is exactly the reading ROTATIONAL-EQUILIBRIUM-CONFOUND bounds.
PRIOR_P = {"AI-SURVIVES": 0.15, "AI-ARTEFACT-VANISHES": 0.25, "AI-ARTEFACT-REVERSES": 0.05, "AI-LAMBDA-DEPENDENT": 0.10,
           "AI-SURVIVES-ONE-LAMBDA": 0.08, "AI-ABSENT-ONE-LAMBDA": 0.15, "AI-UNDECIDED": 0.10,
           "AI-ONE-RUNG-OR-UNREADABLE": 0.12}
PRIOR_SC = {"SC-BEATS-BOTH": 0.15, "SC-TIES-OR-BEATS-BOTH": 0.35, "SC-BELOW-BOTH": 0.10, "SC-LAMBDA-DEPENDENT": 0.10,
            "SC-PARTIAL": 0.22, "SC-UNREADABLE": 0.08}
CHANCE = 10.0


def manifest_text():
    L = ["BATCH %s" % PREFIX, "NETWORK %s" % NET, "DATASET %s" % DSET, "NUM_PARAM_TENSORS %d" % NTENS,
         "TOTAL_PARAMS %d" % TOTPAR, "CLIP_C %s" % CLIP, "MS %s ALPHA0 %s" % (MST, A0), "EPOCHS %d" % EPOCHS,
         "WEIGHT_DECAY_BASE %s" % WD_TOKEN,
         "META_STEPS %d" % (EPOCHS * STEPS_PER_EPOCH), "PROBE %d N_RECORDS %d" % (PROBE, N_RECORDS),
         "DECAY_ROUTE %s" % ",".join("%s=%s" % (r, route_value(r)) for r in RUNG_IDS)]
    for g, spec, m in GRAINS:
        L.append("GRAIN %s SPEC %s M %d" % (g, spec, m))
    for a in ARMS:
        L.append("ARM %s RUNG %s ROUTE %s SPEC %s M %d SEEDS %s" % (a, RUNG_OF[a], route_value(RUNG_OF[a]), SPEC[a],
                                                                   M_OF_ARM[a], ",".join(str(s) for s in SEEDS[a])))
    L.append("NJOBS %d" % NJOBS)
    return "\n".join(L) + "\n"


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    print(manifest_text())
    for r in RUNG_IDS:
        print("WITNESS %s  %s" % (r, witness_on(r)))
    print("EXPECTED_GPU_H %.2f  HARD_BOUND_GPU_H %.0f" % (EXPECTED_GPU_H, HARD_BOUND_GPU_H))
    print("every run owes a TWO-AXIS exclusion row (%d: DECAY_ROUTE on + ARGS_WD_BASE weight-decay-base=0)" % N_EXCLUSION_ROWS)