#!/usr/bin/env python3
"""cwd4_design.py -- THE FROZEN DESIGN TABLE OF `cwd4` (CORRECTIONS 280): ON ResNet18_c100, SEPARATE CARRIER IDENTITY
FROM COUNT / DOSE (and, as far as this network allows, FROM TERM MAGNITUDE) IN THE DECAY MASK.

`cwd3` (275 / 278) removed coupled weight decay from the THREE ctd1 carriers alone and the scalar collapse went
(`P_CAR` +47.2787 pp = +89.34 SE), while `cdep1`'s count-matched triple {47,48,56} and its class-pure pair {47,56}
stayed on `k01`'s floor.  278's bound (5) named the rival it could NOT exclude: *"exempting ANY three genuine 512-wide
`layer4` BN scales suffices, and two does not"* fits every number of that batch exactly as well as the carrier account,
because at COUNT THREE no class-pure, carrier-free control can exist (only five 512-wide BN scales exist -- 47, 50, 53,
56, 59 -- and three are carriers; 275.2 / 188.1).  278's open item **O-12** proposed the arm that separates them.

THE DESIGN INSIGHT THIS BATCH RESTS ON: **AT COUNT TWO THE MISSING CONTROL EXISTS.**  {50, 53} are two carriers;
{47, 56} are the only two carrier-free 512-wide `layer4` BN scales -- class-pure (both `.weight` BatchNorm scales),
width-matched (512), depth-matched (`layer4`), numel-matched (1,024) and COUNT-matched (2).  So `TWOWD0` vs `CTL2WD0`
is the count-matched, class-pure, carrier-free contrast that `cwd3` could not build at three.  Adding the three
SINGLE-carrier masks puts the carrier side on a 1 / 2 / 3 dose ladder at the same time.

NO NEW HARNESS CODE.  PATCH_DECAYMASK's REGISTERED grammar is `DECAY_MASK=normscale | <name>(+<name>)*`
(patches/patch_decaymask.py, `_dm_resolve`): `raw.split("+")` over a ONE-token value is the k=1 case and over a
TWO-token value the k=2 case, with the same empty-token, duplicate and unknown-name ValueErrors.  `cwd2` already ran a
ONE-name list on PlainNet's tree.  So `cwd4` runs from `cwd1`'s EXISTING isolated tree $WS/harness_cwd1/cifar10
(HF.py 94aedc33..., UNCHANGED) with `cwd1`'s EXISTING runner $WS/jobs/run_cifar_cwd1.sh (34a8c90e..., UNCHANGED).
What is new is the ARM SET only.  The bite job re-proves the k=1 and k=2 forms on the real GPU path anyway.

This module IMPORTS analysis/cwd_design.py and analysis/cwd_common.py UNEDITED (registered files of cwd1 / cwd2,
RULE 16): the tensor table, the DECAY_MASK witness function and the cell constants come from there, so cwd4's witness
strings are cwd1's by construction.  Stdlib only (no torch).

Imported, UNEDITED, by
    analysis/cWD4_countwd_score.py           (the registered scorer)
    bin/cWD4_countwd.sh                      (the launcher's guards 4 / 4c')
    tests/test_decaymask_realrun_cwd4.py     (the bite proof on the real GPU path)

    python3 analysis/cwd4_design.py          # prints the manifest and the witness table
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


W = _load("cwd_design_for_cwd4", os.path.join(HERE, "cwd_design.py"))
S9 = W.S9
# the two REUSED registered files, pinned (cwd1's G-PROV values, 271.2; identical to cwd3_design's pins)
CWD_DESIGN_SHA = "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195"   # analysis/cwd_design.py
CWD_COMMON_SHA = "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"   # analysis/cwd_common.py
PATCH_DECAYMASK_SHA = "cf88e7d25793b73c27e7a3bafcabd68e5499fe9007e803d871f25aaef75b2379"   # patches/patch_decaymask.py

# ---- common cell (cwd1's = cwd3's, byte for byte) ---------------------------------------------------------------------
DSET, EPOCHS, BATCH, CLIP, MST, A0, AUG, PROBE = W.DSET, W.EPOCHS, W.BATCH, W.CLIP, W.MST, W.A0, W.AUG, W.PROBE
STEPS_PER_EPOCH, N_RECORDS, WD_BASE, ENV_EXPECTED, WALL = W.STEPS_PER_EPOCH, W.N_RECORDS, W.WD_BASE, W.ENV_EXPECTED, "03:00:00"
dm_indices, dm_witness = W.dm_indices, W.dm_witness

# ---- the named tensors (re-derived on the LIVE model at registration, CORRECTIONS 280.2; the launcher's guard 4
# re-derives them again at submit time from cwd1's own tree and aborts on any disagreement) -----------------------------
C50 = "layer4.0.bn2.weight"            # idx 50  carrier  Kim gamma_last(layer4.0)   |L| rank 1..3
C53 = "layer4.0.shortcut.1.weight"     # idx 53  carrier  Kim gamma_down(layer4.0)
C59 = "layer4.1.bn2.weight"            # idx 59  carrier  Kim gamma_last(layer4.1)
N47 = "layer4.0.bn1.weight"            # idx 47  NON-carrier  Kim gamma_others(layer4.0)
N56 = "layer4.1.bn1.weight"            # idx 56  NON-carrier  Kim gamma_others(layer4.1)
CARRIERS = (C50, C53, C59)                                                    # 50, 53, 59  (ctd1)
TWO_CAR = (C50, C53)                                                          # 50, 53      (278's O-12)
CTL_DEPTH2 = (N47, N56)                                                       # 47, 56      (cdep1 DEPTH2, class-pure)

# ---- DESCRIPTIVE, FROZEN, NON-GATING: the landed per-tensor term magnitudes this design was chosen against ------------
# mean |L_i| over the SCALAR anchor's PINNED probe records, with the rank among all 62 tensors, quoted verbatim from
# results/SCORE-cdep1.txt section [P] (cdep1, landed).  NO BAR, LEVEL, CONTRAST, STATE, BRANCH OR STAMP READS THESE.
MEAN_ABS_L_PINNED = {50: (2.9155e-01, 2), 53: (1.9526e-01, 3), 59: (3.2984e-01, 1),
                     47: (1.1298e-03, 36), 48: (1.7399e-06, 62), 56: (1.5166e-03, 33)}
SUM_L_CARRIERS = 8.1665e-01       # SCORE-cdep1 [P]: SUM ISO
SUM_L_DEPTH = 2.6482e-03          # SCORE-cdep1 [P]: SUM DEPTH {47,48,56}
L_RATIO_ISO_DEPTH = 308.4         # SCORE-cdep1 [P]: "the magnitude match this design CANNOT make"
# results/ctd1_tensor_dominate/ATTACK_REPORT.txt [A1], SCALAR L PINNED, median |X| and rank among 62, plus the sign
# census: the three carriers are ranks 1/2/3 AND vote DOWN on 100 % of pinned records; the next-largest terms
# (linear.weight rank 4, layer4.0.conv2.weight rank 5) vote UP on 100 % of them.
MEDIAN_ABS_L_PINNED_TOP = (("layer4.1.bn2.weight", 3.192e-01, 1), ("layer4.0.bn2.weight", 2.830e-01, 2),
                           ("layer4.0.shortcut.1.weight", 1.877e-01, 3), ("linear.weight", 1.345e-01, 4),
                           ("layer4.0.conv2.weight", 6.023e-02, 5))
P_VOTES_DOWN_PINNED = {"carriers": 1.000, "linear.weight": 0.000, "layer4.0.conv2.weight": 0.000,
                       "layer4.1.conv2.weight": 0.000}

# ---- cwd3's landed levels (CORRECTIONS 278.3).  BETWEEN-BATCH, NON-GATING, DISCLOSURE ONLY -----------------------------
CWD3_LEVELS = {"k01": 22.9853, "CARWD0": 70.2640, "CTLWD0": 23.0013, "CTL2WD0": 22.9333, "NWD": 70.7227}


class CWD4:
    PREFIX = "cwd4"
    NET = "ResNet18_c100"
    TENSORS = W.tensors_of(False)
    NAMES = [t[0] for t in TENSORS]
    NTENS = 62
    TOTPAR = 11220132
    SEEDS = (143, 144, 145)
    ARMS = ("k01", "CARWD0", "TWOWD0", "CTL2WD0", "ONE50", "ONE53", "ONE59")
    SPEC = dict((a, "scalar") for a in ARMS)
    PT_TYPE = dict((a, "scalar") for a in ARMS)
    DMASK = {"k01": "", "CARWD0": "+".join(CARRIERS), "TWOWD0": "+".join(TWO_CAR), "CTL2WD0": "+".join(CTL_DEPTH2),
             "ONE50": C50, "ONE53": C53, "ONE59": C59}
    MASKED = ("CARWD0", "TWOWD0", "CTL2WD0", "ONE50", "ONE53", "ONE59")
    SINGLES = ("ONE50", "ONE53", "ONE59")
    REF = "CARWD0"                      # recovery is defined against the IN-BATCH three-carrier mask (RECOVERY-AGAINST-CARWD0)
    SETS = {"CARWD0": CARRIERS, "TWOWD0": TWO_CAR, "CTL2WD0": CTL_DEPTH2,
            "ONE50": (C50,), "ONE53": (C53,), "ONE59": (C59,)}
    SET_IDX = {"CARWD0": (50, 53, 59), "TWOWD0": (50, 53), "CTL2WD0": (47, 56),
               "ONE50": (50,), "ONE53": (53,), "ONE59": (59,)}
    SET_NUMEL = {"CARWD0": 1536, "TWOWD0": 1024, "CTL2WD0": 1024, "ONE50": 512, "ONE53": 512, "ONE59": 512}
    K_MASKED = {"k01": 0, "CARWD0": 3, "TWOWD0": 2, "CTL2WD0": 2, "ONE50": 1, "ONE53": 1, "ONE59": 1}
    # 278's Kim et al. (arXiv:2205.07260) position class of every masked tensor -- DESCRIPTIVE, printed, never gating
    KIM_CLASS = {50: "gamma_last", 53: "gamma_down", 59: "gamma_last", 47: "gamma_others", 56: "gamma_others"}
    OFF_LINES = ("VOTE_W: off", "BETA_HOLD: off", "GROUP_HOLD: off", "REST_HOLD: off")
    OFF_KINDS = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD")
    CARRIERS = CARRIERS
    NORMSCALE_IDX = W.CWD1.NORMSCALE_IDX
    W512_IDX = (47, 50, 53, 56, 59)     # EVERY 512-wide BN scale in the model (275.2 / 188.1; guard 4h re-derives it)
    # the tree, the runner and the network: cwd1's, UNCHANGED (cwd3 ran the same ones)
    HF_PARENT_SHA = W.HF_CVT8_SHA
    HF_POST_SHA = W.HF_CWD1_SHA
    RUNNER_SHA = W.RUNNER_CWD1_SHA
    BN_SHA = W.BN_CVT8_SHA
    RUNNER_PARENT_SHA = W.RUNNER_CVT8_SHA
    PARENT_TREE = "harness_cvt8"
    TREE = "harness_cwd1"
    RUNNER_NAME = "run_cifar_cwd1.sh"
    GPU_CONSTRAINT = "L4"               # 271.4(5) / 273 bound (1): pinned at the LAUNCHER, never in the scorer
    GPU_NAME = "NVIDIA L4"


CWD4.WITNESS_DM = dict((a, dm_witness(CWD4.DMASK[a], CWD4.TENSORS)) for a in CWD4.ARMS)

# THE PAIRS the bite proof must tell apart (RR4).  G-BITE's k cannot separate TWOWD0 from CTL2WD0 (both k=2) nor the
# three singles from each other (all k=1): that separation rests ENTIRELY on G-WITNESS (each run's own DECAY_MASK line
# carries idx and names) and on these real-GPU discriminations.  DECLARED, not papered over -- CORRECTIONS 280.4.
DISCRIMINATE = (("TWOWD0", "CTL2WD0"), ("CTL2WD0", "TWOWD0"), ("CARWD0", "TWOWD0"),
                ("ONE50", "ONE53"), ("ONE53", "ONE59"), ("ONE59", "ONE50"))


def set_rows(D=CWD4):
    """-> [(arm, idx1 tuple, numel, owners, widths, classes)] for every masked name-list arm, from the frozen table."""
    out = []
    for a in D.MASKED:
        idx = dm_indices(D.DMASK[a], D.TENSORS)
        out.append((a, tuple(i + 1 for i in idx), sum(D.TENSORS[i][1] for i in idx),
                    tuple(D.TENSORS[i][2] for i in idx), tuple(D.TENSORS[i][1] for i in idx),
                    tuple("scale" if D.TENSORS[i][0].endswith(".weight") else "shift" for i in idx)))
    return out


def manifest_text(D=CWD4):
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
    for a in CWD4.ARMS:
        print("%-8s k=%d  %s" % (a, CWD4.K_MASKED[a], CWD4.WITNESS_DM[a]))
