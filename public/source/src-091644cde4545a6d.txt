#!/usr/bin/env python3
"""caw1_design.py -- THE FROZEN DESIGN TABLE OF `caw1` (CORRECTIONS 290): THE STANDARD-RECIPE GATE.

THE QUESTION.  Does the shared-step-size collapse appear under a STANDARD recipe?  `cwd5` (CORRECTIONS 285) showed that
at the campaign's cell (SGDm 0.99 base + Lion meta) the scalar step size collapses at alpha-scaled base weight decay 0.1
(k01W1 23.2240 vs kLW1 69.2940) and does not at 1e-2 / 1e-3 / 5e-4.  But 0.1 is a corner case ONLY for an SGDm base:
for an AdamW base, the harness's decay has exactly PyTorch AdamW's form -- the decay term is multiplied by the (learned)
step size -- and 0.1 is a common AdamW value (and the parent's), 1e-2 is PyTorch's own AdamW default.  `caw1` runs the
parent's headline pairing -- AdamW base + Adam meta -- on ResNet18_c100 / CIFAR-100 at BOTH of those standard decays,
scalar vs layerwise in batch, plus the campaign's own mechanism cell as an in-batch POSITIVE CONTROL.

THE HARNESS's AdamW, READ (not assumed; CORRECTIONS 290.2 quotes it).  $WS/harness_cwd1/cifar10/Optimizers/HF.py,
`AdamW_base_update`:
    lambda_t *= b2 ;  mu = (1 - b2) / (1 - lambda_t)
    m  <- b1 * m + g                      (NOTE: an UNNORMALISED sum -- no (1 - b1) factor, no first-moment bias fix)
    v  <- b2 * v + g**2
    delta = a * ( m / sqrt(mu * v + eps) + wd * w )
    w  <- w - delta ;   h <- gamma * (1 - wd * a) * h - delta
So the decay is ALPHA-SCALED (decoupled in Loshchilov & Hutter's sense, arXiv:1711.05101): the per-step shrink is
`a * wd`, exactly as in torch.optim.AdamW with lr = a.  What differs from torch.optim.AdamW is the GRADIENT term: with
the unnormalised first moment its steady-state size is up to 1/(1 - b1) = 10x torch's for the same `a`, so the ratio
of the decay step to the gradient step is up to 10x SMALLER here than in torch.optim.AdamW at the same wd.  The proof
job (tests/test_caw1_realrun.py) verifies this formula BITWISE on the real GPU path at both registered decays.

THE ARMS (5 x seeds {149, 150, 151} = 15 jobs, ONE submission).  The cell is `cwd5`'s byte for byte (ResNet18_c100,
CIFAR-100, 100 epochs, batch 100, AUGMENT=1, BETA_CLIP=-15:-2.3026, meta step 1e-3, alpha0 1e-6, gamma 1, PROBE=100,
PROBE_TENSOR=1, cwd1's tree and runner UNCHANGED) except the optimiser pairing and the decay:
    AS1  AdamW(0.9, 0.999) + Adam(0.9, 0.999)  scalar     wd 0.1     cell A1 -- the parent's pairing at the parent's decay
    AL1  AdamW(0.9, 0.999) + Adam(0.9, 0.999)  layerwise  wd 0.1
    AS2  AdamW(0.9, 0.999) + Adam(0.9, 0.999)  scalar     wd 1e-2    cell A2 -- PyTorch AdamW's default decay
    AL2  AdamW(0.9, 0.999) + Adam(0.9, 0.999)  layerwise  wd 1e-2
    K01  SGDm(0.99) + Lion(0.99, 0.9)          scalar     wd 0.1     the campaign's mechanism cell: IN-BATCH POSITIVE
                                                                     CONTROL (landed 4x at 22.79-23.22)
The AdamW ARGS are the campaign's own AdamW+Adam ARGS (a0-scal-1e6 / g2_mAdam / i3b / cau1 on CIFAR-10), byte for
byte except dataset, network, decay and seed.

NO NEW HARNESS CODE: every flag is an existing registered CLI flag of train.py; AdamW / Adam are existing algs of
HF.py; cwd1's tree ($WS/harness_cwd1, HF.py 94aedc33...) and runner (run_cifar_cwd1.sh, 34a8c90e...) run UNCHANGED, with
every patch OFF (its witnesses print `off`).  Stdlib only (no torch).  Imports analysis/cwd_design.py UNEDITED.

    python3 analysis/caw1_design.py          # prints the arm table, the ARGS strings and the witnesses
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


W = _load("cwd_design_for_caw1", os.path.join(HERE, "cwd_design.py"))
CWD_DESIGN_SHA = "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195"   # analysis/cwd_design.py (RULE 16)
CWD_COMMON_SHA = "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"   # analysis/cwd_common.py (RULE 16)

# ---- the common cell: cwd5's, byte for byte (the optimiser pairing and the decay are this batch's axes) -------------
DSET, EPOCHS, BATCH, CLIP, MST, A0, AUG, PROBE = W.DSET, W.EPOCHS, W.BATCH, W.CLIP, W.MST, W.A0, W.AUG, W.PROBE
STEPS_PER_EPOCH, N_RECORDS, ENV_EXPECTED = W.STEPS_PER_EPOCH, W.N_RECORDS, W.ENV_EXPECTED
WALL = "03:00:00"
NET = "ResNet18_c100"
NTENS = 62
TOTPAR = 11220132
TENSORS = W.tensors_of(False)
NAMES = [t[0] for t in TENSORS]
SEEDS = (149, 150, 151)
PREFIX = "caw1"

# the campaign's three ctd1 carriers (0-based 49 / 52 / 58; 1-based 50 / 53 / 59) -- DECLARED, not fitted
CARRIERS = ("layer4.0.bn2.weight", "layer4.0.shortcut.1.weight", "layer4.1.bn2.weight")
CARRIER_IDX0 = tuple(NAMES.index(n) for n in CARRIERS)

# ---- the two optimiser pairings, as ARGS fragments in the campaign's own flag order ----------------------------------
PAIR = {
    "AW": (("alg-base", "AdamW"), ("normalizer-param-base", "0.999"), ("momentum-param-base", "0.9"),
           ("weight-decay-base", None),
           ("alg-meta", "Adam"), ("normalizer-param-meta", "0.999"), ("momentum-param-meta", "0.9"),
           ("weight-decay-meta", "0")),
    "SL": (("alg-base", "SGDm"), ("momentum-param-base", "0.99"),
           ("weight-decay-base", None),
           ("alg-meta", "Lion"), ("momentum-param-meta", "0.99"), ("Lion-beta2-meta", "0.9"),
           ("weight-decay-meta", "0")),
}
COMMON = (("dataset", DSET), ("NN-name", NET), ("batch-size", str(BATCH)), ("max-time", "999:00:00"),
          ("gamma", "1"), ("meta-stepsize", MST), ("alpha0", A0), ("num-epochs", str(EPOCHS)))
# what each pairing's PROBE_TENSOR line prints after `type=<grain> tensors=62 ` (HF._pt_init: meta alg, the META
# momentum, and args_meta's Lion_beta2; `%g` formatting).  MEASURED, NOT ASSUMED (CORRECTIONS 290.6): the first proof
# job 5079156 showed the Adam arms print `Lion_beta2=0.9`, not the `=1` first written here -- train.py's
# --Lion-beta2-meta defaults to 0.9 (not -1), so build_optimizer puts 0.9 into args_meta for EVERY meta alg, and the
# Adam meta update never reads it.  The value is inert for Adam and is pinned as printed.
PT_TAIL = {"AW": "meta_alg=Adam momentum_param=0.9 Lion_beta2=0.9",
           "SL": "meta_alg=Lion momentum_param=0.99 Lion_beta2=0.9"}
# the pairing's meta momentum and how the per-tensor vote term is formed from the probe's own EMA (m_tensor, which
# HF._pt_capture keeps as mp*m + (1-mp)*z for EVERY meta alg):
#   Adam meta: the harness's momentum is an UNNORMALISED sum M' = mp*M + z, so M_i = m_tensor_i / (1 - mp) and the
#              applied term is L_i = mp*M_i + z_i = mp/(1-mp) * m_tensor_i + z_i; applied sign = sign(mp*mom_pre + z_agg)
#   Lion meta: L_i = b2*m_tensor_i + (1-b2)*z_i; applied sign = sign(b2*mom_pre + (1-b2)*z_agg)   (CORRECTIONS 256)
META_ALG = {"AW": "Adam", "SL": "Lion"}

# ---- the arms ---------------------------------------------------------------------------------------------------------
# (arm, pairing, grain, wd token, wd float, cell, role)
ARM_TABLE = (
    ("AS1", "AW", "scalar",    "0.1",  0.1,  "A1", "AdamW+Adam at wd 0.1 (the parent's pairing and decay), scalar"),
    ("AL1", "AW", "layerwise", "0.1",  0.1,  "A1", "AdamW+Adam at wd 0.1, layerwise -- A1's in-batch reference"),
    ("AS2", "AW", "scalar",    "1e-2", 0.01, "A2", "AdamW+Adam at wd 1e-2 (torch.optim.AdamW's default), scalar"),
    ("AL2", "AW", "layerwise", "1e-2", 0.01, "A2", "AdamW+Adam at wd 1e-2, layerwise -- A2's in-batch reference"),
    ("K01", "SL", "scalar",    "0.1",  0.1,  "K",  "SGDm(0.99)+Lion at wd 0.1: the mechanism cell, POSITIVE CONTROL"),
)
ARMS = tuple(r[0] for r in ARM_TABLE)
PAIR_OF = dict((r[0], r[1]) for r in ARM_TABLE)
SPEC = dict((r[0], r[2]) for r in ARM_TABLE)
WD = dict((r[0], r[3]) for r in ARM_TABLE)
WDF = dict((r[0], r[4]) for r in ARM_TABLE)
CELL_OF = dict((r[0], r[5]) for r in ARM_TABLE)
CELLS = ("A1", "A2")
SCALAR_OF = {"A1": "AS1", "A2": "AS2"}
LAYER_OF = {"A1": "AL1", "A2": "AL2"}
CONTROL = "K01"
SCALAR_ARMS = ("AS1", "AS2", "K01")

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


def args_pairs(arm, seed, save):
    """-> the ordered (flag, value) pairs of one run's train.py command line (the ARGS line's payload)."""
    out = [("optimizer", "HF")]
    for f, v in PAIR[PAIR_OF[arm]]:
        out.append((f, WD[arm] if f == "weight-decay-base" else v))
    out += list(COMMON)
    out += [("stepsize-groups", SPEC[arm]), ("seed", str(seed)), ("save-directory", save),
            ("run-name", "%s-%s-s%d" % (PREFIX, arm, seed))]
    return out


def args_string(arm, seed, save):
    return " ".join("--%s %s" % (f, v) for f, v in args_pairs(arm, seed, save))


def pt_witness_prefix(arm):
    """the PROBE_TENSOR line up to (not including) ` dir=`: grain, tensor count and the META alg's own constants."""
    return "PROBE_TENSOR: on every=%d type=%s tensors=%d %s" % (PROBE, SPEC[arm], NTENS, PT_TAIL[PAIR_OF[arm]])


def manifest_text():
    lines = ["BATCH %s" % PREFIX, "NETWORK %s" % NET, "NUM_PARAM_TENSORS %d" % len(TENSORS),
             "TOTAL_PARAMS %d" % sum(t[1] for t in TENSORS), "CLIP_C %s" % CLIP, "EPOCHS %d" % EPOCHS,
             "META_STEPS %d" % (EPOCHS * STEPS_PER_EPOCH), "SEEDS %s" % ",".join(str(s) for s in SEEDS)]
    for i, (n, q, o, nd) in enumerate(TENSORS, 1):
        lines.append("TENSOR %d %s %d %s ndim=%d" % (i, n, q, o, nd))
    lines.append("CARRIERS %s" % ",".join("%d:%s" % (i + 1, NAMES[i]) for i in CARRIER_IDX0))
    for a in ARMS:
        lines.append("ARM %s PAIR %s GRAIN %s WD %s CELL %s" % (a, PAIR_OF[a], SPEC[a], WD[a], CELL_OF[a]))
        lines.append("ARGS %s %s" % (a, args_string(a, 0, "<save>")))
        lines.append("PTWITNESS %s %s" % (a, pt_witness_prefix(a)))
    return "\n".join(lines) + "\n"


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---- the exclusion rows owed at landing, as corpus_exclusions.py's ARGS kinds define them (CORRECTIONS 263) ----------
# standard values: momentum-param-base 0.99, weight-decay-base 0.1.  An AdamW arm carries momentum-param-base 0.9.
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
        print("%-4s %-3s %-9s wd %-5s cell %-2s deviates on %s" % (a, PAIR_OF[a], SPEC[a], WD[a], CELL_OF[a],
                                                                  args_deviating_kinds(a) or "nothing"))
