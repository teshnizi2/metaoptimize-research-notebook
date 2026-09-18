#!/usr/bin/env python3
"""cwd_design.py -- THE FROZEN DESIGN TABLES OF `cwd1` (CORRECTIONS 260) AND `cwd2` (CORRECTIONS 261).

One source for the arm tables, spec strings, DECAY_MASK strings, seeds, witness lines, tree / runner / patch hashes and
the synthetic partition manifests.  Imported, UNEDITED, by
    analysis/cWD1_normwd_score.py      (the registered scorer of cwd1)
    analysis/cWD2_carrierwd_score.py   (the registered scorer of cwd2)
    bin/cWD1_normwd.sh, bin/cWD2_carrierwd.sh  (the launchers' guard 4 / 4c')
    tests/test_decaymask.py, tests/test_decaymask_realrun.py  (the patch proofs)
Stdlib only (no torch).  It imports the registered cvt9 scorer (analysis/cVT9_dosewindow_score.py, UNEDITED) for the
ResNet18 / PlainNet18 tensor tables and cvt9's BETA_HOLD / COMP_HOLD witness lines, so the replicate strings are cvt9's by
construction.

    python3 analysis/cwd_design.py            # prints both manifests and the witness tables
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


S9 = _load("cvt9_scorer_for_cwd", os.path.join(HERE, "cVT9_dosewindow_score.py"))
S9_SHA = "5c53a40dff7f8eebb80525f4d8ad38b682054b6185699b85620f13903588c2d7"   # the registered cvt9 scorer (249); checked in the selftests

# ---- common cell (the mechanism cell of every carrier batch) ---------------------------------------------------------
DSET = "CIFAR100"
EPOCHS = 100
BATCH = 100
CLIP = "-15:-2.3026"
MST = "1e-3"
A0 = "1e-6"
AUG = "1"
PROBE = 100
STEPS_PER_EPOCH = 50000 // BATCH
N_RECORDS = EPOCHS * STEPS_PER_EPOCH // PROBE          # 500
WD_BASE = 0.1
ENV_EXPECTED = S9.ENV_EXPECTED                          # the runner's ENV line (PROBE_DIR stripped), cvt1 ... cvt9's
WALL = "03:00:00"
HF_LIVE_SHA = "4732b74aa3a10508896e92eccced5c89051c353fa8a01a3af21aab9aeda0cecd"
RUNNER_SRC_SHA = "a0d0a1b904e64ea5a3de8c0649b58454beff0322419c97ea9e6d77481cfd9f3a"   # $WS/jobs/run_cifar.sh

# ---- the two parent trees (READ ONLY) --------------------------------------------------------------------------------
HF_CVT8_SHA = "5197dc2eecbbcb6feb815f3798b3455171e6ad614deac913be5df48dd9ede1a9"   # cvt8's HF.py (+ PATCH_RESTHOLD)
HF_CVT9_SHA = "816e335777f7c0bb52747949dff81aa57fe0695bab2d61d984ae6ce9daab3a9d"   # cvt9's HF.py (+ PATCH_WINDOWHOLD)
BN_CVT8_SHA = "c79988833c4399a06cc84d15c0830f54d86f59feeec345f47a2287c1cf2ba77d"   # cvt8's build_network.py (live ResNet)
BN_CVT9_SHA = "e65e67738e45418045ccc7f737fb0ea5b2523a9294a4c3a288e543e95508a18b"   # cvt9's build_network.py (cpl1 POST)
RUNNER_CVT8_SHA = "1dbe4ff52eccbbb1c9936f4a687ed88172dd41c48b14ae60eb353b45f766f5e0"
RUNNER_CVT9_SHA = "0fe45b1d1d2cea28740c72e08193af7d5bab3819b397f59f580c71bb9fbf7849"

# ---- the two cwd trees (pinned from a local application of the patch to the parents' HF.py bytes; verified at staging)
HF_CWD1_SHA = "94aedc33046afb12782ffb7f2eb729bafc7fcba5f42227fec373fa67b6f58ebf"   # cvt8's HF.py + PATCH_DECAYMASK
HF_CWD2_SHA = "133c12296b4b9e4428cfe8ba882413c85fc0ec7e5372f3043e6aaa99b623a4b4"   # cvt9's HF.py + PATCH_DECAYMASK
RUNNER_CWD1_SHA = "34a8c90ee3a0bee00aa2d0b4a3ad5d16d461fbf5a1f0cda5797e44e44c158371"   # $WS/jobs/run_cifar_cwd1.sh (run_cifar.sh, cd line)
RUNNER_CWD2_SHA = "f3dfa97fb93e3d749aab156ca2562dd24e9b97e77664cb1537ad1c16a6df73ef"   # $WS/jobs/run_cifar_cwd2.sh (run_cifar.sh, cd line)


# ---- the DECAY_MASK witness ------------------------------------------------------------------------------------------
def dm_indices(spec, tensors):
    """tensors: [(name, numel, owner, ndim)] -> sorted 0-based indices, exactly as PATCH_DECAYMASK._dm_resolve."""
    names = [t[0] for t in tensors]
    if spec == "normscale":
        return sorted(i for i, t in enumerate(tensors) if t[0].endswith(".weight") and t[3] == 1)
    return sorted(names.index(x) for x in spec.split("+"))


def dm_witness(spec, tensors, wd=WD_BASE):
    if not spec:
        return "DECAY_MASK: off"
    idx = dm_indices(spec, tensors)
    return ("DECAY_MASK: on base=SGDm wd=%r spec=%s masked=%d of=%d numel=%d idx=%s names=%s"
            % (wd, spec, len(idx), len(tensors), sum(tensors[i][1] for i in idx), ",".join("%d" % (i + 1) for i in idx),
               ",".join(tensors[i][0] for i in idx)))


def tensors_of(plain):
    """[(name, numel, owner, ndim)] in parameter order, from cvt9's registered r18_tensors (guard 4 checks the live model)."""
    out = []
    for n, q, o in S9.r18_tensors(plain):
        nd = 1 if (o in ("BatchNorm2d", "GroupNorm") or n.endswith(".bias")) else (4 if o == "Conv2d" else 2)
        out.append((n, q, o, nd))
    return out


# =====================================================================================================================
# cwd1 -- ResNet18_c100: does the scalar collapse need coupled WD on the normalisation scales?   (CORRECTIONS 260)
# =====================================================================================================================
class CWD1:
    PREFIX = "cwd1"
    NET = "ResNet18_c100"
    TENSORS = tensors_of(False)
    NAMES = [t[0] for t in TENSORS]
    NTENS = 62
    TOTPAR = 11220132
    SEEDS = (128, 129, 130)                             # 131 of the pre-assigned block NOT used (260.2)
    ARMS = ("k01", "k01NWD", "kLNWD")
    SPEC = {"k01": "scalar", "k01NWD": "scalar", "kLNWD": "layerwise"}
    PT_TYPE = {"k01": "scalar", "k01NWD": "scalar", "kLNWD": "layerwise"}
    DMASK = {"k01": "", "k01NWD": "normscale", "kLNWD": "normscale"}
    MASKED = ("k01NWD", "kLNWD")
    OFF_LINES = ("VOTE_W: off", "BETA_HOLD: off", "GROUP_HOLD: off", "REST_HOLD: off")
    OFF_KINDS = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD")
    NORMSCALE_IDX = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59)
    NORMSCALE_NUMEL = 4800
    CARRIERS = ("layer4.0.bn2.weight", "layer4.0.shortcut.1.weight", "layer4.1.bn2.weight")   # 50, 53, 59 (ctd1)
    HF_PARENT_SHA = HF_CVT8_SHA
    HF_POST_SHA = HF_CWD1_SHA
    RUNNER_SHA = RUNNER_CWD1_SHA
    BN_SHA = BN_CVT8_SHA
    RUNNER_PARENT_SHA = RUNNER_CVT8_SHA
    PARENT_TREE = "harness_cvt8"
    TREE = "harness_cwd1"


CWD1.WITNESS_DM = dict((a, dm_witness(CWD1.DMASK[a], CWD1.TENSORS)) for a in CWD1.ARMS)


# =====================================================================================================================
# cwd2 -- PlainNet18_c100: does the held large step on idx 50 do its damage through WD on that scale?   (CORRECTIONS 261)
# =====================================================================================================================
class CWD2:
    PREFIX = "cwd2"
    NET = "PlainNet18_c100"
    TENSORS = tensors_of(True)
    NAMES = [t[0] for t in TENSORS]
    NTENS = 53
    TOTPAR = 11046308
    SEEDS = (132, 133, 134)                             # 135 of the pre-assigned block NOT used (261.2)
    CARRIER = "layer4.1.bn2.weight"                     # idx 50 (HEAD)
    HEADSPEC = S9.HEADSPEC                              # sets:1-49,51-53/layer4.1.bn2.weight
    ARMS = ("k01", "k01WD0", "HIGHHEADPATH", "HIGHWD0", "LOWWD0")
    CTL = "LOWWD0"
    SPEC = {"k01": "scalar", "k01WD0": "scalar", "HIGHHEADPATH": S9.HEADSPEC, "HIGHWD0": S9.HEADSPEC, "LOWWD0": S9.HEADSPEC}
    PT_TYPE = {"k01": "scalar", "k01WD0": "scalar", "HIGHHEADPATH": "blockwise", "HIGHWD0": "blockwise", "LOWWD0": "blockwise"}
    # the cvt9 arm each cwd2 arm's hold schedule IS (so cvt9's registered bite_check audits it unchanged)
    AS_CVT9 = {"k01": "k01", "k01WD0": "k01", "HIGHHEADPATH": "HIGHHEADPATH", "HIGHWD0": "HIGHHEADPATH", "LOWWD0": "LOWHEADPATH"}
    WITNESS_VW = "VOTE_W: off"
    DMASK = {"k01": "", "k01WD0": CARRIER, "HIGHHEADPATH": "", "HIGHWD0": CARRIER, "LOWWD0": CARRIER}
    MASKED = ("k01WD0", "HIGHWD0", "LOWWD0")
    HELD = ("HIGHHEADPATH", "HIGHWD0", "LOWWD0")
    HF_PARENT_SHA = HF_CVT9_SHA
    HF_POST_SHA = HF_CWD2_SHA
    RUNNER_SHA = RUNNER_CWD2_SHA
    BN_SHA = BN_CVT9_SHA
    RUNNER_PARENT_SHA = RUNNER_CVT9_SHA
    PARENT_TREE = "harness_cvt9"
    TREE = "harness_cwd2"
    HEADPATH_ID = S9.HEADPATH_ID
    HEADPATH_SHA = S9.HEADPATH_SHA


CWD2.BETAHOLD = dict((a, S9.BETAHOLD[CWD2.AS_CVT9[a]]) for a in CWD2.ARMS)
CWD2.COMPHOLD = dict((a, S9.COMPHOLD[CWD2.AS_CVT9[a]]) for a in CWD2.ARMS)
CWD2.WITNESS_BH = dict((a, S9.WITNESS_BH[CWD2.AS_CVT9[a]]) for a in CWD2.ARMS)
CWD2.WITNESS_CH = dict((a, S9.WITNESS_CH[CWD2.AS_CVT9[a]]) for a in CWD2.ARMS)
CWD2.WITNESS_WH = dict((a, "WINDOW_HOLD: off") for a in CWD2.ARMS)
CWD2.WITNESS_DM = dict((a, dm_witness(CWD2.DMASK[a], CWD2.TENSORS)) for a in CWD2.ARMS)


# ---- synthetic partition manifests (the launchers' guard 4d proves the LIVE manifest byte-identical) -----------------
def manifest_text(D):
    t = D.TENSORS
    lines = ["NETWORK %s" % D.NET, "NUM_PARAM_TENSORS %d" % len(t), "TOTAL_PARAMS %d" % sum(x[1] for x in t),
             "CLIP_C %s" % CLIP, "EPOCHS %d" % EPOCHS, "META_STEPS %d" % (EPOCHS * STEPS_PER_EPOCH), "WD_BASE %r" % WD_BASE]
    for i, (n, q, o, nd) in enumerate(t, 1):
        lines.append("TENSOR %d %s %d %s ndim=%d" % (i, n, q, o, nd))
    lines.append("NORMSCALE %s" % ",".join("%d:%s" % (i + 1, t[i][0]) for i in dm_indices("normscale", t)))
    for a in D.ARMS:
        lines.append("ARMSPEC %s SPEC %s TYPE %s" % (a, D.SPEC[a], D.PT_TYPE[a]))
        lines.append("DECAYMASK %s %s" % (a, D.DMASK[a] or "off"))
        lines.append("DWITNESS %s %s" % (a, D.WITNESS_DM[a]))
        if D is CWD2:
            lines.append("BETAHOLD %s %s" % (a, D.BETAHOLD[a] or "off"))
            lines.append("COMPHOLD %s %s" % (a, D.COMPHOLD[a] or "off"))
    return "\n".join(lines) + "\n"


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


if __name__ == "__main__":
    for D in (CWD1, CWD2):
        print("=" * 100)
        print(manifest_text(D))
