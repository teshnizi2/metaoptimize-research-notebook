#!/usr/bin/env python3
"""cwd3_design.py -- THE FROZEN DESIGN TABLE OF `cwd3` (CORRECTIONS 275): ON ResNet18_c100, IS IT THE CARRIERS' OWN
COUPLED WEIGHT DECAY?

`cwd1` (260 / 271) removed coupled weight decay from ALL 20 BatchNorm scales at once and the scalar collapse vanished.
It could not say WHICH of the 20 matters.  `cwd3` masks the three `ctd1` carriers ALONE, a count / numel / width /
depth-matched non-carrier triple (`cdep1`'s DEPTH set, 188.1), a class-pure non-carrier pair (`cdep1`'s DEPTH2 set),
and all 20 (the `cwd1` replicate, in batch), every arm in the SCALAR grouping.

NO NEW HARNESS CODE.  PATCH_DECAYMASK's registered grammar (`DECAY_MASK=normscale | <name>(+<name>)*`,
patches/patch_decaymask.py) already admits a name list, so `cwd3` runs from `cwd1`'s EXISTING isolated tree
$WS/harness_cwd1/cifar10 (HF.py 94aedc33..., UNCHANGED) with `cwd1`'s EXISTING runner $WS/jobs/run_cifar_cwd1.sh
(34a8c90e..., archived as jobs/run_cifar_cwd1.sh, UNCHANGED).  What is new is the ARM SET only.

This module IMPORTS analysis/cwd_design.py and analysis/cwd_common.py UNEDITED (both are registered files of cwd1 /
cwd2, RULE 16): the tensor table, the DECAY_MASK witness function, the cell constants and the tree / runner hashes all
come from there, so cwd3's witness strings are cwd1's by construction.  Stdlib only (no torch).

Imported, UNEDITED, by
    analysis/cWD3_carrierwd_score.py         (the registered scorer)
    bin/cWD3_carrierwd.sh                    (the launcher's guards 4 / 4c')
    tests/test_decaymask_realrun_cwd3.py     (the bite proof on the real GPU path)

    python3 analysis/cwd3_design.py          # prints the manifest and the witness table
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


W = _load("cwd_design_for_cwd3", os.path.join(HERE, "cwd_design.py"))
S9 = W.S9
# the two REUSED registered files, pinned (cwd1's G-PROV values, 271.2); the scorer's selftest and the launcher check them
CWD_DESIGN_SHA = "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195"   # analysis/cwd_design.py
CWD_COMMON_SHA = "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"   # analysis/cwd_common.py
PATCH_DECAYMASK_SHA = "cf88e7d25793b73c27e7a3bafcabd68e5499fe9007e803d871f25aaef75b2379"   # patches/patch_decaymask.py

# ---- common cell (cwd1's, byte for byte) -----------------------------------------------------------------------------
DSET, EPOCHS, BATCH, CLIP, MST, A0, AUG, PROBE = W.DSET, W.EPOCHS, W.BATCH, W.CLIP, W.MST, W.A0, W.AUG, W.PROBE
STEPS_PER_EPOCH, N_RECORDS, WD_BASE, ENV_EXPECTED, WALL = W.STEPS_PER_EPOCH, W.N_RECORDS, W.WD_BASE, W.ENV_EXPECTED, "03:00:00"
dm_indices, dm_witness = W.dm_indices, W.dm_witness

# the three sets, by NAME (re-derived on the LIVE model at registration, CORRECTIONS 275.2; the launcher's guard 4
# re-derives them again at submit time and aborts on any disagreement)
CARRIERS = ("layer4.0.bn2.weight", "layer4.0.shortcut.1.weight", "layer4.1.bn2.weight")        # 50, 53, 59 (ctd1)
CTL_DEPTH = ("layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.1.bn1.weight")                 # 47, 48, 56 (cdep1 DEPTH)
CTL_DEPTH2 = ("layer4.0.bn1.weight", "layer4.1.bn1.weight")                                     # 47, 56     (cdep1 DEPTH2)


class CWD3:
    PREFIX = "cwd3"
    NET = "ResNet18_c100"
    TENSORS = W.tensors_of(False)
    NAMES = [t[0] for t in TENSORS]
    NTENS = 62
    TOTPAR = 11220132
    SEEDS = (140, 141, 142)
    ARMS = ("k01", "CARWD0", "CTLWD0", "CTL2WD0", "NWD")
    SPEC = dict((a, "scalar") for a in ARMS)
    PT_TYPE = dict((a, "scalar") for a in ARMS)
    DMASK = {"k01": "", "CARWD0": "+".join(CARRIERS), "CTLWD0": "+".join(CTL_DEPTH),
             "CTL2WD0": "+".join(CTL_DEPTH2), "NWD": "normscale"}
    MASKED = ("CARWD0", "CTLWD0", "CTL2WD0", "NWD")
    SETS = {"CARWD0": CARRIERS, "CTLWD0": CTL_DEPTH, "CTL2WD0": CTL_DEPTH2}
    SET_IDX = {"CARWD0": (50, 53, 59), "CTLWD0": (47, 48, 56), "CTL2WD0": (47, 56),
               "NWD": (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59)}
    SET_NUMEL = {"CARWD0": 1536, "CTLWD0": 1536, "CTL2WD0": 1024, "NWD": 4800}
    K_MASKED = {"k01": 0, "CARWD0": 3, "CTLWD0": 3, "CTL2WD0": 2, "NWD": 20}
    OFF_LINES = ("VOTE_W: off", "BETA_HOLD: off", "GROUP_HOLD: off", "REST_HOLD: off")
    OFF_KINDS = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD")
    CARRIERS = CARRIERS
    # the tree, the runner and the network: cwd1's, UNCHANGED
    HF_PARENT_SHA = W.HF_CVT8_SHA
    HF_POST_SHA = W.HF_CWD1_SHA
    RUNNER_SHA = W.RUNNER_CWD1_SHA
    BN_SHA = W.BN_CVT8_SHA
    RUNNER_PARENT_SHA = W.RUNNER_CVT8_SHA
    PARENT_TREE = "harness_cvt8"
    TREE = "harness_cwd1"
    RUNNER_NAME = "run_cifar_cwd1.sh"
    GPU_CONSTRAINT = "L4"                       # 271.4(5) / 273 bound (1): pinned at the LAUNCHER, never in the scorer
    GPU_NAME = "NVIDIA L4"


CWD3.WITNESS_DM = dict((a, dm_witness(CWD3.DMASK[a], CWD3.TENSORS)) for a in CWD3.ARMS)


def set_rows(D=CWD3):
    """-> [(arm, idx1 tuple, numel, owners, widths, classes)] for every masked name-list arm, from the frozen table."""
    out = []
    for a in ("CARWD0", "CTLWD0", "CTL2WD0"):
        idx = dm_indices(D.DMASK[a], D.TENSORS)
        out.append((a, tuple(i + 1 for i in idx), sum(D.TENSORS[i][1] for i in idx),
                    tuple(D.TENSORS[i][2] for i in idx), tuple(D.TENSORS[i][1] for i in idx),
                    tuple("scale" if D.TENSORS[i][0].endswith(".weight") else "shift" for i in idx)))
    return out


def manifest_text(D=CWD3):
    """cwd_design.manifest_text (UNEDITED; the CWD2-only branch is not taken) plus one SET line per name-list arm."""
    txt = W.manifest_text(D)
    for a, idx, numel, owners, widths, classes in set_rows(D):
        txt += "SET %s idx=%s numel=%d owners=%s widths=%s classes=%s\n" % (
            a, ",".join("%d" % i for i in idx), numel, ",".join(owners), ",".join("%d" % w for w in widths), ",".join(classes))
    return txt


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    print(manifest_text())
    for a in CWD3.ARMS:
        print("%-8s %s" % (a, CWD3.WITNESS_DM[a]))
