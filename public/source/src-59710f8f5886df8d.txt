#!/usr/bin/env python3
"""cwd5_design.py -- THE FROZEN DESIGN TABLE OF `cwd5` (CORRECTIONS 281): THE COUPLED WEIGHT-DECAY LADDER ON
`ResNet18_c100` AT THE MECHANISM CELL.  DOES THE SHARED-STEP-SIZE COLLAPSE EXIST AT NORMAL WEIGHT-DECAY VALUES?

WHY THIS BATCH EXISTS.  Every number in the mechanism line -- and every scalar cell of the audit's own granularity
table -- sits at coupled base weight decay **0.1**, roughly 200x the value a CIFAR ResNet is normally trained with
(5e-4).  `cmo1` (CORRECTIONS 264) established the two ENDPOINTS and nothing between them: at wd **0.1** the scalar arm
collapses (`k01` 22.79) and at wd **0** it does not (`W0k01` 71.56, ABOVE its own layerwise arm).  So the campaign
currently cannot answer the first question any referee will ask: *is the scalar-vs-layerwise gap a GRANULARITY effect,
or a broken scalar CONFIGURATION that exists only at one extreme weight decay?*  `cwd5` puts four rungs between the
endpoints and reads the gap at each.

THE DELIVERABLE IS ONE CURVE: `G(w)` = layerwise - scalar at coupled base weight decay `w`, at
w = 0.1 (the campaign's anchor), 1e-2, 1e-3 and 5e-4 (the standard CIFAR SGD value), three seeds per point,
BOTH GRAINS RUN IN BATCH AT EVERY RUNG so the gap is a within-rung contrast and never a cross-batch one.

WHAT IS **NOT** HERE, AND WHY -- THE DECOUPLED ARM IS **DESCOPED**, NOT FORGOTTEN (281.2).  The base optimiser's
update is COUPLED L2 by construction and offers no decoupled path: `HF.SGDm_base_update` is
    delta = a * (m + wd * w);   w <- w - delta;   h <- gamma * (1 - wd * a) * h - delta
so the decay term is multiplied by the LEARNED step size `a` in the weight update AND its Jacobian factor `(1 - wd*a)`
is baked into the meta trace.  A decoupled arm therefore needs a NEW harness patch touching BOTH, i.e. a full patch
cycle (structural delete-back, a real-GPU bite job with a bitwise formula verifier, loudness tests) -- and it has no
well-defined "same effective strength", because the coupled per-step dose is `wd * a` while `a` is learned, clamped to
exp([-15, -2.3026]) = [3.06e-07, 0.1], and differs by orders of magnitude between the collapsing scalar arm (large `a`
on the carriers -- the mechanism itself) and the healthy layerwise arm.  Any single lambda matches one arm and not the
other, so the contrast would be confounded by construction.  `DECOUPLED-NOT-TESTED` is stamped on every FINAL.

NO NEW HARNESS CODE.  `--weight-decay-base` is an EXISTING registered CLI float of `train.py` (default 0.1; `cmo1`
already ran it at 0).  The one masked arm (`CARW2`) uses `PATCH_DECAYMASK` UNEDITED from `cwd1`'s isolated tree
$WS/harness_cwd1/cifar10 (HF.py 94aedc33..., the bytes cwd3 and cwd4 ran on) with `cwd1`'s runner (34a8c90e...).
The patch has never run at a weight decay other than 0.1, so the bite job re-proves the witness and the BITWISE update
formula at wd 1e-2, and proves the LADDER'S OWN PREMISE on the real GPU path: that changing the flag changes the
update at every rung.

This module IMPORTS analysis/cwd_design.py UNEDITED (a registered file of cwd1 / cwd2, RULE 16): the tensor table, the
`DECAY_MASK` witness function and the cell constants come from there, so cwd5's witness strings are cwd1's by
construction.  Stdlib only (no torch).

Imported, UNEDITED, by
    analysis/cWD5_wdladder_score.py          (the registered scorer)
    bin/cWD5_wdladder.sh                     (the launcher's guards 4 / 4c')
    tests/test_decaymask_realrun_cwd5.py     (the bite proof on the real GPU path)

    python3 analysis/cwd5_design.py          # prints the manifest and the witness table
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


W = _load("cwd_design_for_cwd5", os.path.join(HERE, "cwd_design.py"))
S9 = W.S9
# the REUSED registered files, pinned (cwd1's G-PROV values, 271.2; identical to cwd3_design's / cwd4_design's pins)
CWD_DESIGN_SHA = "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195"   # analysis/cwd_design.py
CWD_COMMON_SHA = "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"   # analysis/cwd_common.py
PATCH_DECAYMASK_SHA = "cf88e7d25793b73c27e7a3bafcabd68e5499fe9007e803d871f25aaef75b2379"   # patches/patch_decaymask.py

# ---- the common cell (cwd1's = cwd3's = cwd4's, byte for byte EXCEPT the weight decay, which is this batch's axis) ----
DSET, EPOCHS, BATCH, CLIP, MST, A0, AUG, PROBE = W.DSET, W.EPOCHS, W.BATCH, W.CLIP, W.MST, W.A0, W.AUG, W.PROBE
STEPS_PER_EPOCH, N_RECORDS, ENV_EXPECTED, WALL = W.STEPS_PER_EPOCH, W.N_RECORDS, W.ENV_EXPECTED, "03:00:00"
WD_ANCHOR = W.WD_BASE                 # 0.1 -- the campaign's value, and rung W1
dm_indices, dm_witness = W.dm_indices, W.dm_witness

# ---- THE LADDER.  The CLI token is what the ARGS line echoes and what --expect / corpus_exclusions compare. ----------
# (rung, CLI token, float value, what the rung is)
RUNGS = (("W1", "0.1",  0.1,    "the campaign's anchor -- every mechanism-line and audit-scalar number sits here"),
         ("W2", "1e-2", 0.01,   "one decade down"),
         ("W3", "1e-3", 0.001,  "two decades down"),
         ("W4", "5e-4", 0.0005, "the standard CIFAR SGD value -- 200x below the anchor"))
RUNG_IDS = tuple(r[0] for r in RUNGS)
WD_TOKEN = dict((r[0], r[1]) for r in RUNGS)
WD_VALUE = dict((r[0], r[2]) for r in RUNGS)
ANCHOR_RUNG = "W1"
CARRIER_RUNG = "W2"                   # the ONE rung that also gets a carrier-mask companion (281.2)

# ---- the three ctd1 carriers (275.2 / 188.1; the launcher's guard 4h re-derives them from the LIVE model) ------------
C50 = "layer4.0.bn2.weight"           # idx 50
C53 = "layer4.0.shortcut.1.weight"    # idx 53
C59 = "layer4.1.bn2.weight"           # idx 59
CARRIERS = (C50, C53, C59)
CARRIER_SPEC = "+".join(CARRIERS)

# ---- landed levels quoted for DISCLOSURE only.  BETWEEN-BATCH, NON-GATING: no bar, level, sigma, state or contrast
# ---- of this batch reads any of them.  The scorer's selftest proves it by reading its own source. --------------------
LANDED_DISCLOSURE = {
    "cwd3 k01 (wd 0.1, scalar)": 22.9853,          # CORRECTIONS 278.3
    "cwd3 CARWD0 (wd 0.1, scalar, carriers masked)": 70.2640,
    "cmo1 W0k01 (wd 0, scalar)": 71.5600,          # CORRECTIONS 264 -- the LOWER endpoint this ladder interpolates to
    "cmo1 k01 (wd 0.1, scalar)": 22.7887,
    "cell layerwise mean (n=32, census)": 69.42,   # LIMITS-PREP 2.2 -- a CENSUS figure, not a scorer figure
}


class CWD5:
    PREFIX = "cwd5"
    NET = "ResNet18_c100"
    TENSORS = W.tensors_of(False)
    NAMES = [t[0] for t in TENSORS]
    NTENS = 62
    TOTPAR = 11220132
    SEEDS = (146, 147, 148)
    # 9 arms: scalar + layerwise at each of the four rungs, plus ONE carrier-mask companion at rung W2
    ARMS = ("k01W1", "kLW1", "k01W2", "kLW2", "k01W3", "kLW3", "k01W4", "kLW4", "CARW2")
    SCALAR_OF = {"W1": "k01W1", "W2": "k01W2", "W3": "k01W3", "W4": "k01W4"}
    LAYER_OF = {"W1": "kLW1", "W2": "kLW2", "W3": "kLW3", "W4": "kLW4"}
    CARRIER_ARM = "CARW2"
    RUNG_OF = {"k01W1": "W1", "kLW1": "W1", "k01W2": "W2", "kLW2": "W2",
               "k01W3": "W3", "kLW3": "W3", "k01W4": "W4", "kLW4": "W4", "CARW2": "W2"}
    SPEC = {"k01W1": "scalar", "kLW1": "layerwise", "k01W2": "scalar", "kLW2": "layerwise",
            "k01W3": "scalar", "kLW3": "layerwise", "k01W4": "scalar", "kLW4": "layerwise", "CARW2": "scalar"}
    PT_TYPE = dict(SPEC)
    DMASK = {"k01W1": "", "kLW1": "", "k01W2": "", "kLW2": "", "k01W3": "", "kLW3": "",
             "k01W4": "", "kLW4": "", "CARW2": CARRIER_SPEC}
    MASKED = ("CARW2",)
    K_MASKED = {"k01W1": 0, "kLW1": 0, "k01W2": 0, "kLW2": 0, "k01W3": 0, "kLW3": 0,
                "k01W4": 0, "kLW4": 0, "CARW2": 3}
    SET_IDX = {"CARW2": (50, 53, 59)}
    SET_NUMEL = {"CARW2": 1536}
    SETS = {"CARW2": CARRIERS}
    OFF_LINES = ("VOTE_W: off", "BETA_HOLD: off", "GROUP_HOLD: off", "REST_HOLD: off")
    OFF_KINDS = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD")
    CARRIERS = CARRIERS
    NORMSCALE_IDX = W.CWD1.NORMSCALE_IDX
    W512_IDX = (47, 50, 53, 56, 59)
    # the tree, the runner and the network: cwd1's, UNCHANGED (cwd3 and cwd4 ran the same ones)
    HF_PARENT_SHA = W.HF_CVT8_SHA
    HF_POST_SHA = W.HF_CWD1_SHA
    RUNNER_SHA = W.RUNNER_CWD1_SHA
    BN_SHA = W.BN_CVT8_SHA
    RUNNER_PARENT_SHA = W.RUNNER_CVT8_SHA
    PARENT_TREE = "harness_cvt8"
    TREE = "harness_cwd1"
    RUNNER_NAME = "run_cifar_cwd1.sh"
    GPU_CONSTRAINT = "L4"              # 271.4(5) / 273 bound (1): pinned at the LAUNCHER, never in the scorer
    GPU_NAME = "NVIDIA L4"


CWD5.WD = dict((a, WD_TOKEN[CWD5.RUNG_OF[a]]) for a in CWD5.ARMS)          # the CLI token, verbatim
CWD5.WDF = dict((a, WD_VALUE[CWD5.RUNG_OF[a]]) for a in CWD5.ARMS)         # the float the harness sees
# the runs that DEVIATE from the standard cell's ARGS value 0.1 and therefore owe an ARGS_WD_BASE exclusion row
CWD5.ARGS_DEVIATING = tuple(a for a in CWD5.ARMS if CWD5.WD[a] != WD_TOKEN[ANCHOR_RUNG])

# the DECAY_MASK witness is built with THE ARM'S OWN weight decay -- CARW2 runs at 1e-2, so its witness says wd=0.01,
# a string PATCH_DECAYMASK has never printed before.  That is exactly what the bite job proves on the real GPU path.
CWD5.WITNESS_DM = dict((a, dm_witness(CWD5.DMASK[a], CWD5.TENSORS, CWD5.WDF[a])) for a in CWD5.ARMS)


def manifest_text(D=CWD5):
    """The live-model manifest.  NOT cwd_design.manifest_text: that prints ONE global `WD_BASE` line and builds every
    DWITNESS at wd 0.1, and in this batch the weight decay is the AXIS.  The TENSOR / NORMSCALE lines are built from
    the same tables, so they are cwd1's by construction."""
    t = D.TENSORS
    lines = ["NETWORK %s" % D.NET, "NUM_PARAM_TENSORS %d" % len(t), "TOTAL_PARAMS %d" % sum(x[1] for x in t),
             "CLIP_C %s" % CLIP, "EPOCHS %d" % EPOCHS, "META_STEPS %d" % (EPOCHS * STEPS_PER_EPOCH)]
    lines.append("WD_LADDER %s" % ",".join("%s=%s" % (r, WD_TOKEN[r]) for r in RUNG_IDS))
    for i, (n, q, o, nd) in enumerate(t, 1):
        lines.append("TENSOR %d %s %d %s ndim=%d" % (i, n, q, o, nd))
    lines.append("NORMSCALE %s" % ",".join("%d:%s" % (i + 1, t[i][0]) for i in dm_indices("normscale", t)))
    for a in D.ARMS:
        lines.append("ARMSPEC %s SPEC %s TYPE %s" % (a, D.SPEC[a], D.PT_TYPE[a]))
        lines.append("ARMWD %s RUNG %s WD %s" % (a, D.RUNG_OF[a], D.WD[a]))
        lines.append("DECAYMASK %s %s" % (a, D.DMASK[a] or "off"))
        lines.append("DWITNESS %s %s" % (a, D.WITNESS_DM[a]))
    for a in D.MASKED:
        idx = dm_indices(D.DMASK[a], t)
        lines.append("SET %s idx=%s numel=%d owners=%s widths=%s classes=%s" % (
            a, ",".join("%d" % (i + 1) for i in idx), sum(t[i][1] for i in idx),
            ",".join(t[i][2] for i in idx), ",".join("%d" % t[i][1] for i in idx),
            ",".join("scale" if t[i][0].endswith(".weight") else "shift" for i in idx)))
    return "\n".join(lines) + "\n"


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    print(manifest_text())
    print("RUNGS")
    for r, tok, val, why in RUNGS:
        print("  %s  --weight-decay-base %-5s (%r)  %s" % (r, tok, val, why))
    print("ARMS")
    for a in CWD5.ARMS:
        print("  %-7s rung=%s wd=%-5s grain=%-9s k=%d  %s"
              % (a, CWD5.RUNG_OF[a], CWD5.WD[a], CWD5.SPEC[a], CWD5.K_MASKED[a], CWD5.WITNESS_DM[a]))
    print("ARGS-DEVIATING RUNS (owe an ARGS_WD_BASE exclusion row per seed): %s" % (CWD5.ARGS_DEVIATING,))
