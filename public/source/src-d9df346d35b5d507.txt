#!/usr/bin/env python3
"""caw2_design.py -- THE FROZEN DESIGN TABLE OF `caw2` (CORRECTIONS 290): THE STANDARD-RECIPE GATE, REDESIGNED SO A
NULL CAN BE READ.  `caw2` is the SUCCESSOR of `caw1` (cad9fba / 16faa5f), which is WITHDRAWN BEFORE LAUNCH (no caw1
job was ever submitted; its registered files stay in the repo UNEDITED, RULE 16).

WHY caw1 WAS WITHDRAWN (the verifier's V2, binding).  caw1 compared the campaign's collapsing cell -- SGDm(0.99) base +
Lion meta -- with AdamW base + Adam meta.  That swaps TWO things at once (the base AND the meta optimiser), and, because
the harness's AdamW first moment is an UNNORMALISED sum (m <- b1*m + g, no (1 - b1)), it also shrinks the ratio of the
decay step to the gradient step by up to 1/(1 - b1) = 10x at the same wd.  A caw1 NO-COLLAPSE could therefore have
been caused by the meta change, by the base change, or by a ~10x smaller decay dose, and caw1 could not say which:
it could not decide "hazard vs corner case".  caw2 gives each confound its own in-batch cell.

THE QUESTION.  Is the shared-step-size collapse a hazard of the standard AdamW recipe, or a corner case of SGDm(0.99)
+ Lion at alpha-scaled decay 0.1 -- and if the standard recipe does not collapse, is that because of the META swap,
the BASE swap, or the DOSE?

THE HARNESS's DECAY, READ (caw1's reading, re-used; the proof job re-verifies it BITWISE on the real GPU path).
$WS/harness_cwd1/cifar10/Optimizers/HF.py:
    SGDm_base_update:  delta = a * (m + wd * w)                        (m the (1 - mp)-normalised momentum)
    AdamW_base_update: lambda_t *= b2 ; mu = (1 - b2) / (1 - lambda_t)
                       m <- b1 * m + g   (UNNORMALISED sum)  ;  v <- b2 * v + g**2
                       delta = a * ( m / sqrt(mu * v + eps) + wd * w )
    both:              w <- w - delta ;  h <- gamma * (1 - wd * a) * h - delta
So in BOTH bases the decay term is multiplied by the LEARNED step size a: the per-step shrink is a*wd.  That is the
form torch.optim.AdamW uses (it multiplies the decay by lr).  It is NOT Loshchilov & Hutter's decoupled decay
(arXiv:1711.05101): in their AdamW the decay term lambda*theta is scaled by the SCHEDULE MULTIPLIER eta_t only, not by
the step size alpha -- which is exactly the property ("decoupled from alpha") that the harness's form lacks.  This file
therefore calls the harness's decay ALPHA-SCALED (CORRECTIONS 292's name), and never "decoupled".  (V3: caw1's design
prose said the harness's decay was "decoupled in Loshchilov & Hutter's sense"; that sentence was wrong, and caw1's
files are NOT edited -- this successor carries the corrected sentence.)

THE ARMS (9 arms x seeds {160, 161, 162} = 27 jobs, ONE submission).  The cell is cwd5's byte for byte (ResNet18_c100,
CIFAR-100, 100 epochs, batch 100, AUGMENT=1, BETA_CLIP=-15:-2.3026, meta step 1e-3, alpha0 1e-6, gamma 1, PROBE=100,
PROBE_TENSOR=1, cwd1's tree and runner UNCHANGED, every patch OFF) except the base / meta pairing and the decay:
  cell K  K01          SGDm(0.99) + Lion(0.99, 0.9)       wd 0.1  scalar      the mechanism cell: IN-BATCH POSITIVE CONTROL
  cell M  MS | ML      SGDm(0.99) + Adam(0.9, 0.999)      wd 0.1  both grains the META swap ALONE (base held at K's)
  cell L  LS | LL      AdamW(0.9, 0.999) + Lion(0.99, 0.9) wd 0.1 both grains the BASE swap ALONE (meta held at K's)
  cell A  AS | AL      AdamW(0.9, 0.999) + Adam(0.9, 0.999) wd 0.1 both grains the STANDARD RECIPE (both swaps)
  cell X  XS | XL      AdamW(0.9, 0.999) + Adam(0.9, 0.999) wd 1.0 both grains the DOSE arm: A at 10x the decay
cell X is the decay-to-gradient ratio's arm.  In the harness, AdamW(a, wd) with the unnormalised first moment takes a
gradient step of up to 10x torch.optim.AdamW's at the same a (steady state, fully correlated gradients; ~2.3x for pure
noise, 1/sqrt(1 - b1**2)), so harness wd 1.0 is the harness value whose decay-to-gradient ratio is torch.optim.AdamW's
at wd 0.1 at the UPPER end of that mapping.  1.0 is NOT a standard harness value and is not presented as one: X is a
dose arm.  Whether the doses actually matched is NOT ASSUMED: the scorer reports the REALISED per-step shrink a*wd of
every arm from its own probe records (peak over the run, and peak over the pre-divergence window, steps <= 8,500) and
its ratio to the in-batch control's, and the NO-COLLAPSE branches read that ratio.

WHAT WAS DROPPED, WITH REASONS.  (i) caw1's wd-1e-2 AdamW cell (A2): cwd5 bracketed the SGDm(0.99)+Lion transition
between 0.1 and 1e-2 (CORRECTIONS 285), so an AdamW cell at 1e-2 -- a SMALLER dose than A's -- is predicted no-collapse
under every account this batch distinguishes and would not change which branch is taken; its 6 runs (~4.1 GPU-h) buy
the X cell instead.  If A or X collapses, 1e-2 is the natural follow-up.  (ii) a layerwise arm for cell K: the
control only has to REPRODUCE the collapse (K01 <= 30), which it did on four landed batches (22.79-23.22); its
layerwise reference has landed four times (67.9-69.3) and is not re-bought.  (iii) more seeds: 3 per arm, as every
cwd batch; SPLIT (1 or 2 of 3 scalar seeds collapsed) is a registered state, not a failure.

NO NEW HARNESS CODE: every flag is an existing registered CLI flag of train.py; SGDm / AdamW / Adam / Lion are existing
algs of HF.py; cwd1's tree ($WS/harness_cwd1, HF.py 94aedc33...) and runner (run_cifar_cwd1.sh, 34a8c90e...) run
UNCHANGED.  Stdlib only (no torch).  Imports analysis/cwd_design.py UNEDITED.

    python3 analysis/caw2_design.py          # prints the arm table, the ARGS strings and the witnesses
"""
import hashlib
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W = _load("cwd_design_for_caw2", os.path.join(HERE, "cwd_design.py"))
CWD_DESIGN_SHA = "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195"   # analysis/cwd_design.py (RULE 16)
CWD_COMMON_SHA = "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"   # analysis/cwd_common.py (RULE 16)
# the WITHDRAWN predecessor's registered files, pinned so the successor's selftest proves they were NOT edited (RULE 16)
CAW1_FILES_SHA = {
    "analysis/caw1_design.py": "0795aedd68be22a6741241c667e20e9b9ca92a23399cc4a16fc59fb441a9bfac",
    "analysis/cAW1_stdrecipe_score.py": "bd259052a1a738cad71ff565dd8cc94c7888087f6d860c5ed7925a6f26e94793",
    "tests/test_caw1_realrun.py": "72bcb3e17ab7a8dfd41e2057671d237c46e1c3f5c4a5b09e0cbdac9959483f5a",
    "bin/cAW1_realrun_proof.sbatch": "61455a0265653814815466b238a05032e9258be6b63a6ad803dfe76ce30e159b",
}

# ---- the common cell: cwd5's, byte for byte ------------------------------------------------------------------------
DSET, EPOCHS, BATCH, CLIP, MST, A0, AUG, PROBE = W.DSET, W.EPOCHS, W.BATCH, W.CLIP, W.MST, W.A0, W.AUG, W.PROBE
STEPS_PER_EPOCH, N_RECORDS, ENV_EXPECTED = W.STEPS_PER_EPOCH, W.N_RECORDS, W.ENV_EXPECTED
WALL = "03:00:00"
NET = "ResNet18_c100"
NTENS = 62
TOTPAR = 11220132
TENSORS = W.tensors_of(False)
NAMES = [t[0] for t in TENSORS]
SEEDS = (160, 161, 162)
SEED_BLOCK = tuple(range(160, 166))   # the fresh block this track took (cgw1 holds 152..159; caw1 held 149..151)
PREFIX = "caw2"

# the campaign's three ctd1 carriers (0-based 49 / 52 / 58; 1-based 50 / 53 / 59) -- DECLARED, not fitted
CARRIERS = ("layer4.0.bn2.weight", "layer4.0.shortcut.1.weight", "layer4.1.bn2.weight")
CARRIER_IDX0 = tuple(NAMES.index(n) for n in CARRIERS)

# ---- the base and meta halves, as ARGS fragments in the campaign's own flag order -----------------------------------
BASE = {
    "SGDm": (("alg-base", "SGDm"), ("momentum-param-base", "0.99"), ("weight-decay-base", None)),
    "AdamW": (("alg-base", "AdamW"), ("normalizer-param-base", "0.999"), ("momentum-param-base", "0.9"),
              ("weight-decay-base", None)),
}
META = {
    "Lion": (("alg-meta", "Lion"), ("momentum-param-meta", "0.99"), ("Lion-beta2-meta", "0.9"),
             ("weight-decay-meta", "0")),
    "Adam": (("alg-meta", "Adam"), ("normalizer-param-meta", "0.999"), ("momentum-param-meta", "0.9"),
             ("weight-decay-meta", "0")),
}
# pairing code -> (base alg, meta alg)
PAIRS = {"SL": ("SGDm", "Lion"), "SA": ("SGDm", "Adam"), "WL": ("AdamW", "Lion"), "AW": ("AdamW", "Adam")}
COMMON = (("dataset", DSET), ("NN-name", NET), ("batch-size", str(BATCH)), ("max-time", "999:00:00"),
          ("gamma", "1"), ("meta-stepsize", MST), ("alpha0", A0), ("num-epochs", str(EPOCHS)))
# what the PROBE_TENSOR line prints after `type=<grain> tensors=62 ` (HF._pt_init: the META alg, args_meta's
# momentum_param and args_meta's Lion_beta2, `%g`).  caw1's proof job 5079156 MEASURED that Adam-meta runs print
# Lion_beta2=0.9 (train.py's --Lion-beta2-meta defaults to 0.9 and build_optimizer passes it to every meta alg; the Adam
# update never reads it).  It depends on the META half only; caw2's proof job re-measures it on all four pairings.
PT_TAIL_META = {"Adam": "meta_alg=Adam momentum_param=0.9 Lion_beta2=0.9",
                "Lion": "meta_alg=Lion momentum_param=0.99 Lion_beta2=0.9"}

# ---- the arms ---------------------------------------------------------------------------------------------------------
# (arm, pairing, grain, wd token, wd float, cell, role)
ARM_TABLE = (
    ("K01", "SL", "scalar",    "0.1", 0.1, "K", "SGDm(0.99)+Lion at wd 0.1: the mechanism cell, POSITIVE CONTROL"),
    ("MS",  "SA", "scalar",    "0.1", 0.1, "M", "SGDm(0.99)+Adam at wd 0.1 (META swap alone), scalar"),
    ("ML",  "SA", "layerwise", "0.1", 0.1, "M", "SGDm(0.99)+Adam at wd 0.1, layerwise -- M's in-batch reference"),
    ("LS",  "WL", "scalar",    "0.1", 0.1, "L", "AdamW+Lion at wd 0.1 (BASE swap alone), scalar"),
    ("LL",  "WL", "layerwise", "0.1", 0.1, "L", "AdamW+Lion at wd 0.1, layerwise -- L's in-batch reference"),
    ("AS",  "AW", "scalar",    "0.1", 0.1, "A", "AdamW+Adam at wd 0.1 (the STANDARD RECIPE), scalar"),
    ("AL",  "AW", "layerwise", "0.1", 0.1, "A", "AdamW+Adam at wd 0.1, layerwise -- A's in-batch reference"),
    ("XS",  "AW", "scalar",    "1.0", 1.0, "X", "AdamW+Adam at wd 1.0 (the DOSE arm, 10x A's decay), scalar"),
    ("XL",  "AW", "layerwise", "1.0", 1.0, "X", "AdamW+Adam at wd 1.0, layerwise -- X's in-batch reference"),
)
ARMS = tuple(r[0] for r in ARM_TABLE)
PAIR_OF = dict((r[0], r[1]) for r in ARM_TABLE)
SPEC = dict((r[0], r[2]) for r in ARM_TABLE)
WD = dict((r[0], r[3]) for r in ARM_TABLE)
WDF = dict((r[0], r[4]) for r in ARM_TABLE)
CELL_OF = dict((r[0], r[5]) for r in ARM_TABLE)
ROLE = dict((r[0], r[6]) for r in ARM_TABLE)
BASE_OF = dict((a, PAIRS[PAIR_OF[a]][0]) for a in ARMS)
META_OF = dict((a, PAIRS[PAIR_OF[a]][1]) for a in ARMS)
CELLS = ("M", "L", "A", "X")
SCALAR_OF = {"M": "MS", "L": "LS", "A": "AS", "X": "XS"}
LAYER_OF = {"M": "ML", "L": "LL", "A": "AL", "X": "XL"}
CONTROL = "K01"
SCALAR_ARMS = ("K01", "MS", "LS", "AS", "XS")

OFF_LINES = ("VOTE_W: off", "BETA_HOLD: off", "GROUP_HOLD: off", "REST_HOLD: off", "DECAY_MASK: off")

# cwd1's tree and runner, UNCHANGED (cwd3 / cwd4 / cwd5 ran on the same bytes)
HF_SHA = W.HF_CWD1_SHA
HF_PARENT_SHA = W.HF_CVT8_SHA
BN_SHA = W.BN_CVT8_SHA
RUNNER_SHA = W.RUNNER_CWD1_SHA
TREE = "harness_cwd1"
RUNNER_NAME = "run_cifar_cwd1.sh"
GPU_CONSTRAINT = "L4"
GPU_NAME = "NVIDIA L4"
PER_RUN_GPUH = 0.6855                    # cwd5's sacct mean over 27 runs of THIS cell on L4 (CORRECTIONS 285.1)
NJOBS = len(ARMS) * len(SEEDS)           # 27


def args_pairs(arm, seed, save):
    """-> the ordered (flag, value) pairs of one run's train.py command line (the ARGS line's payload)."""
    out = [("optimizer", "HF")]
    for f, v in BASE[BASE_OF[arm]]:
        out.append((f, WD[arm] if f == "weight-decay-base" else v))
    out += list(META[META_OF[arm]])
    out += list(COMMON)
    out += [("stepsize-groups", SPEC[arm]), ("seed", str(seed)), ("save-directory", save),
            ("run-name", "%s-%s-s%d" % (PREFIX, arm, seed))]
    return out


def args_string(arm, seed, save):
    return " ".join("--%s %s" % (f, v) for f, v in args_pairs(arm, seed, save))


def pt_witness_prefix(arm):
    """the PROBE_TENSOR line up to (not including) ` dir=`: grain, tensor count and the META alg's own constants."""
    return "PROBE_TENSOR: on every=%d type=%s tensors=%d %s" % (PROBE, SPEC[arm], NTENS, PT_TAIL_META[META_OF[arm]])


def manifest_text():
    lines = ["BATCH %s" % PREFIX, "NETWORK %s" % NET, "NUM_PARAM_TENSORS %d" % len(TENSORS),
             "TOTAL_PARAMS %d" % sum(t[1] for t in TENSORS), "CLIP_C %s" % CLIP, "EPOCHS %d" % EPOCHS,
             "META_STEPS %d" % (EPOCHS * STEPS_PER_EPOCH), "SEEDS %s" % ",".join(str(s) for s in SEEDS)]
    for i, (n, q, o, nd) in enumerate(TENSORS, 1):
        lines.append("TENSOR %d %s %d %s ndim=%d" % (i, n, q, o, nd))
    lines.append("CARRIERS %s" % ",".join("%d:%s" % (i + 1, NAMES[i]) for i in CARRIER_IDX0))
    for a in ARMS:
        lines.append("ARM %s PAIR %s BASE %s META %s GRAIN %s WD %s CELL %s"
                     % (a, PAIR_OF[a], BASE_OF[a], META_OF[a], SPEC[a], WD[a], CELL_OF[a]))
        lines.append("ARGS %s %s" % (a, args_string(a, 0, "<save>")))
        lines.append("PTWITNESS %s %s" % (a, pt_witness_prefix(a)))
    return "\n".join(lines) + "\n"


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---- the exclusion rows owed at landing, as corpus_exclusions.py's ARGS kinds define them (CORRECTIONS 263) ----------
# standard values: momentum-param-base 0.99, weight-decay-base 0.1.  An AdamW arm carries momentum-param-base 0.9; the
# X arms also carry weight-decay-base 1.0.  The meta alg is a CSV cell-key column (`meta`), so a meta swap owes nothing.
ARGS_STD = {"momentum-param-base": "0.99", "weight-decay-base": "0.1"}


def args_deviating_kinds(arm):
    fl = dict(args_pairs(arm, 0, "x"))
    out = []
    if float(fl["momentum-param-base"]) != float(ARGS_STD["momentum-param-base"]):
        out.append("ARGS_MOMENTUM_BASE")
    if float(fl["weight-decay-base"]) != float(ARGS_STD["weight-decay-base"]):
        out.append("ARGS_WD_BASE")
    return tuple(out)


if __name__ == "__main__":
    print(manifest_text())
    for a in ARMS:
        print("%-4s %-2s %-5s+%-4s %-9s wd %-4s cell %s  deviates on %s" % (
            a, PAIR_OF[a], BASE_OF[a], META_OF[a], SPEC[a], WD[a], CELL_OF[a], args_deviating_kinds(a) or "nothing"))
    print("%d jobs, expected %.1f GPU-h (%.4f x %d), hard bound %d GPU-h (WALL %s x %d)"
          % (NJOBS, NJOBS * PER_RUN_GPUH, PER_RUN_GPUH, NJOBS, 3 * NJOBS, WALL, NJOBS))
