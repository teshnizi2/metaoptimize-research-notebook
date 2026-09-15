#!/usr/bin/env python3
# =============================================================================
# cVG1_vggbn_gap_score.py -- THE REGISTERED SCORER FOR `cvg1`, THE CAMPAIGN'S
#   FIRST NON-ResNet BATCH.  ONE SUBMISSION, 6 JOBS (2 arms x 3 seeds),
#   100 EPOCHS, SEEDS {31,32,33}, PROBE=100, PROBE_TENSOR=1 ON EVERY ARM.
#
# Committed BEFORE any cvg1 run exists (STANDING RULE 21).  RUN IT UNEDITED
# (RULE 16); the documented invocation takes ONE argument:
#
#     python3 analysis/cVG1_vggbn_gap_score.py <runsdir>
#
# =============================================================================
# THE SCOPE LIMIT THIS BATCH ATTACKS
# =============================================================================
# Re-derived at registration from results/all_runs.csv at 2,761 rows with the
# `cvg1-` prefix excluded from every reader (the 165.4 pattern -- every frozen
# premise is invariant under this batch's own ingest): the `network` column
# holds ELEVEN distinct values --
#   ResNet18 1854, ResNet18_c100 518, ResNet34 141, ResNet10 110, ResNet50 67,
#   resnet18 26, ResNet18_gn 17, ResNet10_c100 9, ResNet18_tin 9,
#   ResNet34_c100 9, ResNet101 1
# -- and ZERO rows whose network does not match `^[Rr]es[Nn]et`.  Every number
# the campaign owns is a ResNet number.
#
# (The cycle-147 briefing named only the first SIX of those eleven, 2,716 of
# 2,761 rows.  Its headline claim -- all 2,761 rows are ResNet -- is TRUE; its
# LIST is incomplete by five names and 45 rows.  Recorded here because this
# file re-derives the census rather than inheriting it.)
#
# =============================================================================
# THE STRUCTURAL READING THAT PICKS THIS ARCHITECTURE, RE-DERIVED, NOT QUOTED
# =============================================================================
# On the LIVE ResNet18_c100 (62 tensors, 11,220,132 parameters) the five
# 512-wide 1-D `*.weight` tensors -- every 512-wide BatchNorm scale in the
# model -- are at 1-based 47, 50, 53, 56, 59.  CORRECTIONS 187's carriers are
# 50, 53, 59; its non-carriers are 47 and 56.  Against the live
# `BasicBlock.forward`
#     out = F.relu(self.bn1(self.conv1(x)));  out = self.bn2(self.conv2(out))
#     out += self.shortcut(x);                out = F.relu(out)
# the three CARRIERS are EXACTLY the 512-wide BN scales whose output is a
# SUMMAND OF A RESIDUAL ADDITION (bn2 of each layer4 block, and shortcut.1 of
# the block with a projection); the two NON-CARRIERS are EXACTLY the two bn1
# scales, which sit mid-branch behind a ReLU and feed no addition.  3/2, no
# exceptions.
#
# So "the carriers are the BN scales that feed a residual add" and "the
# carriers are the deepest, widest BN scales" are on ResNet THE SAME
# MEASUREMENT SEEN TWICE.  A net with no residual addition separates them.
# THIS IS A HYPOTHESIS ABOUT THE CARRIER SET, NOT A RESULT.  cvg1 does not
# test it -- cvg1 tests the PRECONDITION for testing it.
#
# =============================================================================
# WHAT cvg1 ASKS -- AND WHAT IT DELIBERATELY DOES NOT
# =============================================================================
# THE QUESTION: does the scalar-to-layerwise plateau5 gap -- the gap the whole
# granularity programme is about -- EXIST AT ALL on VGG11_bn_c100, a plain
# BatchNorm conv stack with no residual addition anywhere?
#
# NOT ASKED HERE: which tensors carry it on this architecture.  That needs an
# isolation batch, and registering isolation arms on an architecture whose arm
# levels have never been measured would violate the standing prohibition on
# predicting an arm at or near the floor.  cvg1 measures the levels; the
# isolation batch is the follow-on.  cvg1 nonetheless carries PROBE=100 and
# PROBE_TENSOR=1 on every run at ~zero GPU cost, so the per-tensor z /
# momentum / beta records that let ctd1 nominate ResNet's carriers exist for
# VGG from this batch alone -- the follow-on needs no extra batch to nominate.
#
# NOT A ONE-VARIABLE ABLATION.  VGG11_bn differs from ResNet18 in depth, in
# channel schedule and in downsampling as well as in the residual connection.
# A null here does NOT license "residuals cause the gap"; a positive here does
# NOT license "residuals are irrelevant".  The clean one-variable object is
# ResNet18 with `out += self.shortcut(x)` deleted, and it is NOT built.
#
# =============================================================================
# THE TWO ACCOUNTS, AND THEIR PREDICTIONS, FROZEN
# =============================================================================
# (A) ARCHITECTURE-GENERAL.  The granularity gap is a property of deep conv
#     nets under MetaOptimize at this cell, not of residual structure.
#     PREDICTS: D = plateau5(kL) - plateau5(k01) is large and positive.
# (B) RESIDUAL-SPECIFIC.  The gap is carried by BN scales that feed residual
#     additions.  VGG11_bn has none.
#     PREDICTS: D is small or absent; both arms sit at a common level.
#
# =============================================================================
# THE NOISE FLOOR, RE-DERIVED AT REGISTRATION, cvg1's OWN ROWS EXCLUDED
# =============================================================================
# Pooled within-cell SD of plateau5 over `scalar` and `layerwise` rows at the
# standard hyperparameters (100 epochs, AUGMENT=1, BETA_CLIP -15:-2.3026,
# meta_stepsize 1e-3, alpha0 1e-6, batch 100), superseded / collapsed /
# incomplete rows dropped, cell = the 15 design columns:
#
#   SIGMA_NARROW  (ResNet18_c100 / CIFAR100 only)   0.585420  df 71  11 cells  82 members
#   SIGMA_WIDE    (any network)                     2.971730  df 384 105 cells 489 members
#
# THE CAMPAIGN'S PRECEDENT (149, 156, 159, 187) IS SIGMA_W = max(NARROW, WIDE).
# THAT PRECEDENT IS DEPARTED FROM HERE, AND THIS IS THE DISCLOSURE.  98.92 % of
# SIGMA_WIDE's total sum of squares (3354.568 of 3391.173) comes from ONE cell:
# ResNet18 / scalar, n = 4, within-cell SD 33.439 -- a BIMODAL cell in which
# some seeds escape the floor and some do not.  SIGMA_WIDE therefore estimates
# BIMODALITY, not seed noise; adopting it would set a floor of 2.97 pp and make
# every bar below it vacuously easy.  The next two contributors are 0.33 % and
# 0.22 %.  (The median of the 105 per-cell SDs is 0.168297.)
#
# So: SIGMA_PRIOR := SIGMA_NARROW = 0.585420, the cell family this contrast
# actually lives in; bimodality is handled SEPARATELY and explicitly by the
# DIVERGED gate below rather than by inflating sigma.  AND, because cvg1 runs
# an architecture whose seed noise has NEVER been measured, the floor is not
# trusted from the corpus alone:
#
#   SIGMA_INBATCH = pooled within-arm SD of plateau5 over cvg1's own 2 arms
#                   x 3 seeds (df 4), computed AT SCORE TIME
#   SIGMA_USED    = max(SIGMA_PRIOR, SIGMA_INBATCH)          <-- conservative
#   SE_ARM_DIFF   = SIGMA_USED * sqrt(2/3)
#
# SE_ARM_DIFF at the prior alone is 0.585420 * sqrt(2/3) = 0.477993 pp.
#
# =============================================================================
# THE BARS, FROZEN, IN pp AND IN SE-AT-THE-PRIOR
# =============================================================================
#   D = plateau5(kL) - plateau5(k01), both 3-seed in-batch means.
#
#   GAP_BAR  = +10.0 pp = 20.92 SE_prior  ->  GAP-REPLICATES
#     RATIONALE, from the corpus at registration (cvg1 excluded), scalar vs
#     layerwise at the standard cell, 100 epochs:
#         ResNet18_c100 / CIFAR100      22.914 -> 63.285   D +40.371
#         ResNet18_tin  / TinyImageNet   9.859 -> 50.835   D +40.975
#         ResNet10      / CIFAR10       70.734 -> 90.319   D +19.585
#         ResNet18      / CIFAR10       86.051 -> 91.463   D  +5.412
#         ResNet34      / CIFAR10       89.431 -> 92.570   D  +3.139
#         ResNet50      / CIFAR10       89.510 -> 90.795   D  +1.285
#     The CIFAR-10 rows are CEILING-COMPRESSED (both arms in the high 80s /
#     low 90s), which is precisely why cvg1 runs CIFAR-100.  On the two
#     uncompressed tasks the gap is +40.4 and +41.0.  GAP_BAR = +10.0 is a
#     QUARTER of the smaller of those.  A quarter of the effect is not
#     "the same effect", but nothing below a quarter could be called the
#     phenomenon appearing on this architecture.
#
#   NULL_BAR = +2.0 pp = 4.18 SE_prior   ->  |D| <= NULL_BAR is GAP-ABSENT
#     RATIONALE: 4.18 SE at the prior is comfortably outside seed noise, and
#     2.0 pp is 5 % of the uncompressed ResNet gap.  If moving from one shared
#     step size to 26 per-tensor step sizes buys under 2 pp on this net, the
#     scalar-to-layerwise phenomenon did not appear on it.
#
#   Between them: GAP-PARTIAL.   D <= -NULL_BAR: GAP-REVERSED.
#   DIVERGED_BAR = 5.0 pp: a within-arm seed RANGE above it stamps that arm
#     UNRESOLVED-DIVERGED and SUSPENDS the branch (a bimodal arm must be
#     reported, not averaged -- see the SIGMA_WIDE disclosure above).
#
# =============================================================================
# GATES.  A FAILED GATE SUSPENDS THE BRANCH; IT IS NOT AN ADVERSE RESULT.
# =============================================================================
#   G-SOUND   6/6 runs RUN_DONE, 100 epoch lines each, no traceback.
#   G-FLOOR   max(arm mean) >= FLOOR_MIN = 15.00 pp.  CIFAR-100 chance is
#             1.00 pp.  If BOTH arms land under 15.00 the batch is a statement
#             about the harness, not about architecture, and NO gap claim may
#             be made: stamp HARNESS-UNSOUND and stop.  This is the one way
#             cvg1 can fail to decide anything, and it is registered in
#             advance rather than discovered.
#   G-CEIL    max(arm mean) <= CEIL_MAX = 90.00 pp, so the contrast is not
#             ceiling-compressed the way every CIFAR-10 row above is.
#   G-DIVERGE within-arm seed range <= 5.0 pp in BOTH arms.
#   G-STRUCT  from the batch's own PARTITION-MANIFEST.txt: 26 parameter
#             tensors, 9,274,532 parameters, >= 3 512-wide 1-D `*.weight`
#             tensors (without which the follow-on isolation is not statable),
#             and NO parameter name containing `shortcut`.
#   G-ENV     every run's ENV line carries AUGMENT=1, BETA_CLIP=-15:-2.3026,
#             PROBE=100, and every run prints `PROBE_TENSOR: on`.
#
# NO ARM IS PREDICTED AT OR NEAR THE FLOOR OR THE CEILING, UNDER EITHER
# ACCOUNT.  Chance on CIFAR-100 is 1.00 pp; the ceiling is 100.
#   Under (A): k01 ~ 20-25 (ResNet18_c100's scalar arm is 22.914), kL ~ 55-70.
#              Margins over chance: >= 19 pp and >= 54 pp.  Neither near 100.
#   Under (B): both arms at a common L.  A VGG11-BN trained by SGD-with-
#              momentum at a step size bounded above by exp(-2.3026) = 0.1 on
#              augmented CIFAR-100 for 100 epochs is a well-understood object;
#              L is predicted in 45-68, margin over chance >= 44 pp.
#   The ONLY way both arms approach the floor is total training failure, which
#   is exactly what G-FLOOR catches, and under which no claim is made.
#
# =============================================================================
# WHAT THE RESULT DECIDES FOR THE CAMPAIGN
# =============================================================================
#   GAP-REPLICATES -> the granularity gap is NOT a ResNet artefact.  The
#     campaign's central phenomenon has external validity across architecture
#     families, the scope objection is answered for the GAP, and the
#     three-BN-scale isolation becomes a well-posed question on a second
#     family -- which the follow-on isolation batch asks, using THIS batch's
#     PROBE_TENSOR records to nominate VGG's candidate carriers.  It also
#     makes account (B) hard to hold in its strong form: a net with no
#     residual addition cannot have residual-fed carriers, so whatever carries
#     the gap there is something else, and the ResNet carrier set would then
#     be one architecture's way of expressing a general mechanism.
#   GAP-ABSENT -> the gap, and a fortiori the three-BN-scale result, is
#     ARCHITECTURE-SPECIFIC as far as anything measured can tell.  That is a
#     genuine and publishable scope limit on the campaign's headline, and it
#     promotes the residual reading from a hypothesis to the leading
#     candidate -- WITHOUT establishing it, because of confound (i) (family,
#     not variable) and confound (ii) (RULE 11 mistuning: no argmax is located
#     on VGG by this batch, and mistuning can suppress a gap).  The registered
#     next step under this branch is the tuning bracket, NOT a claim.
#   GAP-PARTIAL -> the phenomenon is present but attenuated; family and
#     tuning are not separated; the tuning bracket is required before any
#     architecture claim.
#   GAP-REVERSED -> unexpected under both accounts; report and stop.
#   HARNESS-UNSOUND -> no architecture claim; the batch reports on the harness.
#
# =============================================================================
# DISCLOSURE.  Visible when this file was written: results/all_runs.csv at
# 2,761 rows; the live build_network.py / train.py / load_data.py /
# Optimizers/HF.py / build_optimizer.py; the live ResNet18_c100 manifest;
# CORRECTIONS 187 and 187.4; bin/cIS1_isolate_carriers.sh and
# analysis/cIS1_ciso1_isolate_score.py as convention templates.  NO cvg1 run
# existed anywhere -- no VGG run of any kind has ever been made in this
# campaign -- when this file was committed.  plateau5 is PRIMARY; the CSV
# `plateau` column is read NOWHERE; best_test is used NOWHERE; TRAIN is
# printed beside TEST at every arm.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys

# ---- the batch design, LITERAL.  The launcher holds the same strings and its
# ---- guard 4c' proves the two agree, so neither can drift. -------------------
NET = "VGG11_bn_c100"
DSET = "CIFAR100"
EPOCHS = 100
BATCH = 100
SEEDS = (31, 32, 33)
CLIP = "-15:-2.3026"
MST = "1e-3"
A0 = "1e-6"
AUG = "1"
PROBE = 100
SPEC = {
    "k01": "scalar",
    "kL": "layerwise",
}
ARMS = ("k01", "kL")

# ---- the live manifest this batch assumes (proved by tests/test_vggbn.py V2
# ---- and re-proved on the live model by the launcher's guard 4) -------------
NTENS = 26
TOTPAR = 9274532
BN512_IDX = (14, 17, 20, 23)
BN512_NAMES = ("bn5.weight", "bn6.weight", "bn7.weight", "bn8.weight")

# ---- the noise floor and the bars, frozen ------------------------------------
SIGMA_NARROW = 0.585420      # df 71, 11 cells, 82 members
SIGMA_WIDE = 2.971730        # df 384, 105 cells, 489 members -- 98.92% ONE cell
SIGMA_WIDE_TOP_SS_FRAC = 0.9892
SIGMA_PRIOR = SIGMA_NARROW   # see the disclosure in the header
SE_PRIOR = 0.477993          # SIGMA_PRIOR * sqrt(2/3)

GAP_BAR = 10.0
NULL_BAR = 2.0
DIVERGED_BAR = 5.0
FLOOR_MIN = 15.0
CEIL_MAX = 90.0
CHANCE = 1.0                 # CIFAR-100

# ---- the corpus premises, frozen at registration (cvg1 excluded) -------------
CORPUS_ROWS = 2761
CORPUS_NETWORKS = {
    "ResNet18": 1854, "ResNet18_c100": 518, "ResNet34": 141, "ResNet10": 110,
    "ResNet50": 67, "resnet18": 26, "ResNet18_gn": 17, "ResNet10_c100": 9,
    "ResNet18_tin": 9, "ResNet34_c100": 9, "ResNet101": 1,
}
CORPUS_GAPS = {                      # (network, dataset): (scalar, layerwise, D)
    ("ResNet18_c100", "CIFAR100"): (22.914, 63.285, 40.371),
    ("ResNet18_tin", "TinyImageNet"): (9.859, 50.835, 40.975),
    ("ResNet10", "CIFAR10"): (70.734, 90.319, 19.585),
    ("ResNet18", "CIFAR10"): (86.051, 91.463, 5.412),
    ("ResNet34", "CIFAR10"): (89.431, 92.570, 3.139),
    ("ResNet50", "CIFAR10"): (89.510, 90.795, 1.285),
}

CELLKEYS = ["network", "dataset", "granularity", "base", "meta", "meta_stepsize",
            "alpha0", "gamma", "augment", "beta_clip", "batch_size",
            "epochs_requested", "hier", "lam", "eta_ratio"]

EPTR_RE = re.compile(r"Epoch\s+(\d+),\s*Train Accuracy:\s*([0-9.]+)\s*%,"
                     r"\s*Test Accuracy:\s*([0-9.]+)")
OUT_RE = re.compile(r"^cvg1-(k01|kL)-s(\d+)-(\d+)\.out$")
PT_RE = re.compile(r"^PROBE_TENSOR: on every=%d type=(scalar|layerwise) tensors=%d "
                   % (PROBE, NTENS))
ENV_RE = re.compile(r"^ENV:")

GATE_FAIL = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   " + extra) if extra else ""))
    if not cond:
        GATE_FAIL.append(label)
    return bool(cond)


def skip(label, extra=""):
    print("  SKIP %s%s" % (label, ("   " + extra) if extra else ""))


def fmt(x, nd=4):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


# =============================================================================
# corpus readers -- cvg1 EXCLUDED FROM EVERY ONE, so every premise is
# invariant under this batch's own ingest
# =============================================================================
def read_corpus(csvpath):
    if not os.path.exists(csvpath):
        return None
    rows = [r for r in csv.DictReader(open(csvpath))
            if not r.get("run", "").startswith("cvg1-")]
    return rows


def usable(r):
    return (r.get("superseded") == "0" and r.get("collapsed") == "0"
            and r.get("complete") == "1" and r.get("plateau5", "").strip() != "")


def std_cell(r):
    return (r.get("granularity") in ("scalar", "layerwise")
            and r.get("epochs_requested") == str(EPOCHS)
            and r.get("augment") == AUG and r.get("beta_clip") == CLIP
            and r.get("meta_stepsize") == MST and r.get("alpha0") == A0
            and r.get("batch_size") == str(BATCH))


def pooled_sigma(rows, extra=lambda r: True):
    cells = {}
    for r in rows:
        if usable(r) and std_cell(r) and extra(r):
            cells.setdefault(tuple(r[k] for k in CELLKEYS), []).append(float(r["plateau5"]))
    ss = 0.0
    df = 0
    nc = 0
    nm = 0
    top = []
    for k, v in cells.items():
        if len(v) < 2:
            continue
        m = mean(v)
        s = sum((x - m) ** 2 for x in v)
        ss += s
        df += len(v) - 1
        nc += 1
        nm += len(v)
        top.append((s, k[0], k[2], len(v)))
    top.sort(reverse=True)
    return (math.sqrt(ss / df) if df else None), df, nc, nm, ss, top


def corpus_gaps(rows):
    by = {}
    for r in rows:
        if usable(r) and std_cell(r):
            by.setdefault((r["network"], r["dataset"]), {}) \
              .setdefault(r["granularity"], []).append(float(r["plateau5"]))
    out = {}
    for k, v in by.items():
        if "scalar" in v and "layerwise" in v:
            s, l = mean(v["scalar"]), mean(v["layerwise"])
            out[k] = (s, l, l - s, len(v["scalar"]), len(v["layerwise"]))
    return out


# =============================================================================
# run readers -- the RAW .out files are PRIMARY; the CSV is never the source
# of a cvg1 level
# =============================================================================
def read_runs(runsdir):
    """-> {(arm, seed): dict} from cvg1-<arm>-s<seed>-<jobid>.out"""
    got = {}
    if not os.path.isdir(runsdir):
        return got
    for fn in sorted(os.listdir(runsdir)):
        m = OUT_RE.match(fn)
        if not m:
            continue
        arm, seed, jid = m.group(1), int(m.group(2)), m.group(3)
        txt = open(os.path.join(runsdir, fn), errors="replace").read()
        eps = [(int(a), float(b), float(c)) for a, b, c in EPTR_RE.findall(txt)]
        eps.sort()
        rec = {
            "file": fn, "job": jid, "arm": arm, "seed": seed,
            "n_epochs": len(eps),
            "traceback": "Traceback (most recent call last)" in txt,
            "env": [ln for ln in txt.splitlines() if ENV_RE.match(ln)],
            "pt": [ln for ln in txt.splitlines() if ln.startswith("PROBE_TENSOR:")],
            "plateau5": None, "train5": None,
        }
        tail = [e for e in eps if e[0] >= EPOCHS - 5]
        if len(tail) == 5:
            rec["plateau5"] = mean([e[2] for e in tail])
            rec["train5"] = mean([e[1] for e in tail])
        prev = got.get((arm, seed))
        if prev is None or jid > prev["job"]:
            got[(arm, seed)] = rec
    return got


def read_manifest(path):
    if not path or not os.path.exists(path):
        return None
    d = {"tensors": [], "raw": {}}
    for ln in open(path):
        p = ln.split()
        if not p:
            continue
        if p[0] == "TENSOR" and len(p) >= 4:
            d["tensors"].append((int(p[1]), p[2], int(p[3])))
        elif len(p) >= 2:
            d["raw"][p[0]] = " ".join(p[1:])
    return d


def resolve_manifest(runsdir, explicit):
    """DEFAULT resolution, so the documented one-argument invocation cannot be
    got wrong (the 164.2 pattern).  Returns (path, how)."""
    if explicit:
        return explicit, "explicit --manifest"
    for cand, how in (
        (os.path.join(runsdir, "cvg1", "PARTITION-MANIFEST.txt"), "<runsdir>/cvg1/"),
        (os.path.join(runsdir, "cvg1-PARTITION-MANIFEST.txt"), "<runsdir>/cvg1-"),
    ):
        if os.path.exists(cand):
            return cand, how
    return None, "not found"


# =============================================================================
def score(runsdir, csvpath, manifest_path):
    print("=" * 78)
    print(" cvg1 -- THE CAMPAIGN'S FIRST NON-ResNet BATCH")
    print(" %s / %s   %d epochs   seeds %s   AUGMENT=%s  BETA_CLIP=%s  PROBE=%d"
          % (NET, DSET, EPOCHS, ",".join(str(s) for s in SEEDS), AUG, CLIP, PROBE))
    print(" arms: k01 = %s   kL = %s" % (SPEC["k01"], SPEC["kL"]))
    print(" PRIMARY: plateau5 = mean TEST accuracy over epochs %d..%d, from the"
          % (EPOCHS - 5, EPOCHS - 1))
    print("          runs' own .out files.  TRAIN printed beside TEST everywhere.")
    print("=" * 78)

    rows = read_corpus(csvpath)
    runs = read_runs(runsdir)

    # ---- G-STRUCT ------------------------------------------------------------
    print("\nG-STRUCT  the live model is what the batch claims")
    mp, how = resolve_manifest(runsdir, manifest_path)
    man = read_manifest(mp)
    if man is None:
        skip("G-STRUCT: no PARTITION-MANIFEST.txt (%s)" % how)
    else:
        print("  manifest: %s (%s)" % (mp, how))
        names = [t[1] for t in man["tensors"]]
        numel = {t[0]: t[2] for t in man["tensors"]}
        chk(len(man["tensors"]) == NTENS, "G-STRUCT %d parameter tensors" % NTENS,
            str(len(man["tensors"])))
        chk(sum(numel.values()) == TOTPAR, "G-STRUCT %d parameters" % TOTPAR,
            str(sum(numel.values())))
        w512 = [(i, n) for i, n, q in man["tensors"]
                if q == 512 and n.endswith(".weight")]
        chk(len(w512) >= 3,
            "G-STRUCT >= 3 512-wide *.weight tensors (isolation is statable)", str(w512))
        chk(tuple(i for i, _n in w512) == BN512_IDX
            and tuple(n for _i, n in w512) == BN512_NAMES,
            "G-STRUCT they are %s at %s" % (",".join(BN512_NAMES),
                                            ",".join(str(i) for i in BN512_IDX)))
        chk(not any("shortcut" in n for n in names),
            "G-STRUCT NO parameter name contains `shortcut` (no residual projection)")

    # ---- G-SOUND / G-ENV -----------------------------------------------------
    print("\nG-SOUND   6 runs, 100 epochs each, no traceback")
    need = [(a, s) for s in SEEDS for a in ARMS]
    for a, s in need:
        r = runs.get((a, s))
        if r is None:
            chk(False, "G-SOUND %s-s%d present" % (a, s), "missing")
            continue
        chk(r["n_epochs"] == EPOCHS and not r["traceback"] and r["plateau5"] is not None,
            "G-SOUND %s-s%d" % (a, s),
            "%d epoch lines, traceback=%s, plateau5 %s (train %s)"
            % (r["n_epochs"], r["traceback"], fmt(r["plateau5"]), fmt(r["train5"])))
    print("\nG-ENV     the environment rode the ENV line on every run")
    envs = sorted({re.sub(r" PROBE_DIR=\S*", "", ln)
                   for r in runs.values() for ln in r["env"]})
    chk(len(envs) == 1, "G-ENV one distinct ENV line across all runs",
        "%d distinct" % len(envs))
    for ln in envs:
        print("        %s" % ln)
        for tok in ("AUGMENT=%s" % AUG, "BETA_CLIP=%s" % CLIP, "PROBE=%d" % PROBE):
            chk(tok in ln, "G-ENV ENV line carries %s" % tok)
    pts = [ln for r in runs.values() for ln in r["pt"]]
    chk(len(pts) == len(runs) and len(runs) > 0 and all(PT_RE.match(p) for p in pts),
        "G-ENV every run printed a matching `PROBE_TENSOR: on` line",
        "%d lines / %d runs" % (len(pts), len(runs)))
    for t in sorted({re.sub(r" dir=.*", "", p) for p in pts}):
        print("        %s" % t)

    # ---- the levels -----------------------------------------------------------
    print("\nLEVELS    plateau5 (TEST) and TRAIN, in batch")
    arm_v, arm_t = {}, {}
    for a in ARMS:
        v = [runs[(a, s)]["plateau5"] for s in SEEDS
             if (a, s) in runs and runs[(a, s)]["plateau5"] is not None]
        t = [runs[(a, s)]["train5"] for s in SEEDS
             if (a, s) in runs and runs[(a, s)]["train5"] is not None]
        arm_v[a], arm_t[a] = v, t
        if v:
            print("  %-4s (%s)  TEST %.4f  sd %s  range %.3f  n=%d   TRAIN %s"
                  % (a, SPEC[a], mean(v), fmt(sd(v), 3),
                     max(v) - min(v), len(v), fmt(mean(t))))
            print("        seeds: %s" % ", ".join("s%d %.4f" % (s, runs[(a, s)]["plateau5"])
                                                  for s in SEEDS if (a, s) in runs
                                                  and runs[(a, s)]["plateau5"] is not None))
        else:
            print("  %-4s (%s)  NO USABLE RUN" % (a, SPEC[a]))

    complete = all(len(arm_v[a]) == len(SEEDS) for a in ARMS)
    if not complete:
        print("\nFINAL: INCOMPLETE -- %s"
              % ", ".join("%s %d/%d" % (a, len(arm_v[a]), len(SEEDS)) for a in ARMS))
        print("No branch is taken and no number here is quotable.")
        return 2

    # ---- the noise floor, prior vs in-batch -----------------------------------
    print("\nFLOOR     SIGMA_PRIOR vs cvg1's OWN seed noise")
    ss = sum(sum((x - mean(arm_v[a])) ** 2 for x in arm_v[a]) for a in ARMS)
    df = sum(len(arm_v[a]) - 1 for a in ARMS)
    sigma_in = math.sqrt(ss / df)
    sigma_used = max(SIGMA_PRIOR, sigma_in)
    se = sigma_used * math.sqrt(2.0 / 3.0)
    print("  SIGMA_PRIOR   %.6f  (corpus, ResNet18_c100/CIFAR100, df 71, cvg1 excluded)"
          % SIGMA_PRIOR)
    print("  SIGMA_INBATCH %.6f  (cvg1's own %d arms x %d seeds, df %d)"
          % (sigma_in, len(ARMS), len(SEEDS), df))
    print("  SIGMA_USED    %.6f  = max of the two (registered, conservative)" % sigma_used)
    print("  SE_ARM_DIFF   %.6f  = SIGMA_USED * sqrt(2/3)" % se)

    # ---- G-FLOOR / G-CEIL / G-DIVERGE -----------------------------------------
    print("\nG-FLOOR / G-CEIL / G-DIVERGE")
    hi = max(mean(arm_v[a]) for a in ARMS)
    lo = min(mean(arm_v[a]) for a in ARMS)
    floor_ok = chk(hi >= FLOOR_MIN,
                   "G-FLOOR max arm mean >= %.2f pp (chance is %.2f)" % (FLOOR_MIN, CHANCE),
                   "max %.4f, margin over chance %+.4f pp" % (hi, hi - CHANCE))
    chk(hi <= CEIL_MAX, "G-CEIL max arm mean <= %.2f pp (not ceiling-compressed)" % CEIL_MAX,
        "max %.4f" % hi)
    div = []
    for a in ARMS:
        rng = max(arm_v[a]) - min(arm_v[a])
        if not chk(rng <= DIVERGED_BAR, "G-DIVERGE %s seed range <= %.2f pp" % (a, DIVERGED_BAR),
                   "range %.4f" % rng):
            div.append(a)
    print("  (both arms' margin over chance: k01 %+.4f, kL %+.4f)"
          % (mean(arm_v["k01"]) - CHANCE, mean(arm_v["kL"]) - CHANCE))
    print("  (lowest arm mean %.4f)" % lo)

    # ---- the contrast ----------------------------------------------------------
    D = mean(arm_v["kL"]) - mean(arm_v["k01"])
    Dtr = mean(arm_t["kL"]) - mean(arm_t["k01"])
    print("\nCONTRAST  D = plateau5(kL) - plateau5(k01), both in-batch 3-seed means")
    print("  D(TEST)  = %.4f - %.4f = %+.4f pp = %+.2f SE"
          % (mean(arm_v["kL"]), mean(arm_v["k01"]), D, D / se))
    print("  D(TRAIN) = %.4f - %.4f = %+.4f pp"
          % (mean(arm_t["kL"]), mean(arm_t["k01"]), Dtr))
    print("  bars: GAP_BAR %+.2f pp (%.2f SE) | NULL_BAR %+.2f pp (%.2f SE)"
          % (GAP_BAR, GAP_BAR / se, NULL_BAR, NULL_BAR / se))
    if rows:
        g = corpus_gaps(rows)
        ref = g.get(("ResNet18_c100", "CIFAR100"))
        if ref:
            print("  reference, re-derived from the corpus (cvg1 excluded): "
                  "ResNet18_c100/CIFAR100 D %+.3f pp (scalar n%d, layerwise n%d)"
                  % (ref[2], ref[3], ref[4]))
            print("  cvg1's D is %.1f%% of that reference." % (100.0 * D / ref[2])
                  if ref[2] else "")

    # ---- the branch -------------------------------------------------------------
    print("\n" + "=" * 78)
    if GATE_FAIL and not floor_ok:
        branch = "HARNESS-UNSOUND"
    elif div:
        branch = "UNRESOLVED-DIVERGED"
    elif D >= GAP_BAR:
        branch = "GAP-REPLICATES"
    elif D <= -NULL_BAR:
        branch = "GAP-REVERSED"
    elif abs(D) <= NULL_BAR:
        branch = "GAP-ABSENT"
    else:
        branch = "GAP-PARTIAL"
    stamps = []
    stamps.append("GATES-CLEAN" if not GATE_FAIL else "GATES-%d-FAIL" % len(GATE_FAIL))
    stamps.append("SIGMA-INBATCH-DOMINATES" if sigma_in > SIGMA_PRIOR
                  else "SIGMA-PRIOR-DOMINATES")
    stamps.append("TRAIN-AGREES" if (Dtr > 0) == (D > 0) else "TRAIN-DISAGREES")
    if branch in ("GAP-ABSENT", "GAP-PARTIAL"):
        stamps.append("MISTUNING-NOT-EXCLUDED")   # confound (ii), RULE 11
    stamps.append("FAMILY-NOT-VARIABLE")          # confound (i), always true here
    print("FINAL: %s | %s" % (branch, " | ".join(stamps)))
    print("=" * 78)
    if GATE_FAIL:
        print("GATE FAILURES (%d): %s" % (len(GATE_FAIL), "; ".join(GATE_FAIL)))
        print("A failed gate SUSPENDS the branch; it is not an adverse result.")
    print("\nWHAT THIS DECIDES -- as registered, before any run existed:")
    if branch == "GAP-REPLICATES":
        print("  The scalar-to-layerwise gap is NOT a ResNet artefact.  The campaign's")
        print("  central phenomenon has external validity across architecture families;")
        print("  the scope objection is answered FOR THE GAP.  The three-BN-scale")
        print("  isolation becomes a well-posed question on a second family: the")
        print("  follow-on isolation batch nominates VGG's candidate carriers from THIS")
        print("  batch's PROBE_TENSOR records.  Account (B) in its strong form is hard")
        print("  to hold -- this net has no residual-fed BN scales at all.")
    elif branch == "GAP-ABSENT":
        print("  The gap -- and a fortiori the three-BN-scale result -- is")
        print("  ARCHITECTURE-SPECIFIC as far as anything measured can tell.  That is a")
        print("  real scope limit on the campaign's headline.  It PROMOTES the residual")
        print("  reading to leading candidate WITHOUT establishing it: confound (i)")
        print("  (family, not variable) and confound (ii) (RULE 11 mistuning -- no")
        print("  argmax is located on VGG here, and mistuning can only suppress a gap)")
        print("  are both live.  REGISTERED NEXT STEP: the tuning bracket, not a claim.")
    elif branch == "GAP-PARTIAL":
        print("  Present but attenuated.  Family and tuning are not separated.")
        print("  REGISTERED NEXT STEP: the tuning bracket before any architecture claim.")
    elif branch == "GAP-REVERSED":
        print("  Unexpected under BOTH registered accounts.  Report and stop.")
    elif branch == "HARNESS-UNSOUND":
        print("  Both arms under %.2f pp: this batch is a statement about the harness," % FLOOR_MIN)
        print("  not about architecture.  NO gap claim may be made from it.")
    else:
        print("  A bimodal arm must be reported, not averaged.  No branch is taken.")
    return 1 if GATE_FAIL else 0


# =============================================================================
def selftest(csvpath, runsdir):
    print("cVG1 --selftest: the frozen premises, the noise floor, the bars, the design.")
    print("Every corpus reader EXCLUDES the `cvg1-` prefix, so each premise is")
    print("invariant under this batch's own ingest.\n")
    chk(abs(SIGMA_PRIOR * math.sqrt(2 / 3) - SE_PRIOR) < 1e-6,
        "SE_PRIOR = SIGMA_PRIOR*sqrt(2/3)", "%.6f" % (SIGMA_PRIOR * math.sqrt(2 / 3)))
    chk(SIGMA_PRIOR == SIGMA_NARROW, "SIGMA_PRIOR is the NARROW estimator (disclosed departure)")
    chk(GAP_BAR > NULL_BAR > 0, "GAP_BAR > NULL_BAR > 0")
    chk(FLOOR_MIN > CHANCE, "FLOOR_MIN is above chance", "%.2f > %.2f" % (FLOOR_MIN, CHANCE))
    chk(EPTR_RE.search("Epoch 99, Train Accuracy: 23.65 %, Test Accuracy: 23.68 %") is not None,
        "epoch-line regex matches the harness's print")
    chk(OUT_RE.match("cvg1-k01-s31-1234567.out") is not None
        and OUT_RE.match("cvg1-kL-s33-1.out") is not None
        and OUT_RE.match("ciso1-k01-s21-1.out") is None,
        "run-file regex matches cvg1 only")
    chk(PT_RE.match("PROBE_TENSOR: on every=100 type=layerwise tensors=26 meta_alg=Lion "
                    "momentum_param=0.99 Lion_beta2=0.9 dir=/x") is not None,
        "PROBE_TENSOR regex matches a 26-tensor VGG line")
    chk(PT_RE.match("PROBE_TENSOR: on every=100 type=layerwise tensors=62 meta_alg=Lion "
                    "momentum_param=0.99 Lion_beta2=0.9 dir=/x") is None,
        "PROBE_TENSOR regex REJECTS a 62-tensor ResNet line")
    chk(set(SPEC) == set(ARMS) and SPEC["k01"] == "scalar" and SPEC["kL"] == "layerwise",
        "the two arms are scalar and layerwise")

    print("\nCORPUS PREMISES, re-derived (never quoted):")
    rows = read_corpus(csvpath)
    if rows is None:
        skip("corpus premises -- %s not readable" % csvpath)
    else:
        chk(len(rows) == CORPUS_ROWS, "corpus has %d rows with cvg1 excluded" % CORPUS_ROWS,
            str(len(rows)))
        cnt = {}
        for r in rows:
            cnt[r["network"]] = cnt.get(r["network"], 0) + 1
        chk(cnt == CORPUS_NETWORKS, "the `network` census matches the frozen census",
            "%d distinct" % len(cnt))
        nonres = [n for n in cnt if not re.match(r"^[Rr]es[Nn]et", n)]
        chk(not nonres, "ZERO non-ResNet rows in the corpus -- the scope limit is real",
            str(nonres))
        chk(NET not in cnt, "%s appears NOWHERE in the corpus (RULE 21 premise)" % NET)

        sn, dfn, cn, mn, _ssn, _tn = pooled_sigma(
            rows, lambda r: r["network"] == "ResNet18_c100" and r["dataset"] == "CIFAR100")
        sw, dfw, cw, mw, ssw, tw = pooled_sigma(rows)
        print("       SIGMA_NARROW %.6f df %d cells %d members %d" % (sn, dfn, cn, mn))
        print("       SIGMA_WIDE   %.6f df %d cells %d members %d" % (sw, dfw, cw, mw))
        chk(abs(sn - SIGMA_NARROW) < 1e-6, "SIGMA_NARROW re-derived == frozen %.6f" % SIGMA_NARROW,
            "%.6f" % sn)
        chk(abs(sw - SIGMA_WIDE) < 1e-6, "SIGMA_WIDE re-derived == frozen %.6f" % SIGMA_WIDE,
            "%.6f" % sw)
        if tw:
            frac = tw[0][0] / ssw
            print("       SIGMA_WIDE's top cell: %s / %s n=%d, %.2f%% of total SS"
                  % (tw[0][1], tw[0][2], tw[0][3], 100 * frac))
            chk(frac > 0.95,
                "SIGMA_WIDE is >95%% ONE bimodal cell -- the disclosed reason it is NOT used",
                "%.4f" % frac)
        chk(sw > sn, "the precedent max() would have picked WIDE -- departure is material")

        g = corpus_gaps(rows)
        print("       scalar -> layerwise gaps at the standard cell:")
        for k in sorted(g):
            s, l, d, ns, nl = g[k]
            print("         %-14s %-12s %7.3f (n%3d) -> %7.3f (n%3d)   D %+8.3f"
                  % (k[0], k[1], s, ns, l, nl, d))
            if k in CORPUS_GAPS:
                fs, fl, fd = CORPUS_GAPS[k]
                chk(abs(s - fs) < 5e-3 and abs(l - fl) < 5e-3 and abs(d - fd) < 5e-3,
                    "gap %s/%s re-derived == frozen" % k, "%+.3f vs %+.3f" % (d, fd))
        unc = [v[2] for k, v in g.items() if k[1] != "CIFAR10"]
        chk(unc and GAP_BAR <= min(unc) / 4.0 + 1e-9,
            "GAP_BAR (%.1f) <= a quarter of the smallest UNCOMPRESSED corpus gap" % GAP_BAR,
            "smallest uncompressed %.3f, quarter %.3f" % (min(unc), min(unc) / 4.0))

    print("\nRULE 21 PREMISE: no cvg1 run may exist yet")
    runs = read_runs(runsdir)
    chk(not runs, "no cvg1-*.out under %s" % runsdir, "%d found" % len(runs))
    if rows is not None:
        chk(not any(r["run"].startswith("cvg1-") for r in csv.DictReader(open(csvpath))),
            "no cvg1- row in the CSV")

    print("\nFLOOR/CEILING PROOF, both accounts, stated before any run:")
    print("  chance = %.2f pp; ceiling = 100 pp; FLOOR_MIN = %.2f; CEIL_MAX = %.2f"
          % (CHANCE, FLOOR_MIN, CEIL_MAX))
    print("  (A) architecture-general: k01 ~ 20-25, kL ~ 55-70")
    print("      margins over chance >= %.2f and >= %.2f pp; neither near 100"
          % (20 - CHANCE, 55 - CHANCE))
    print("  (B) residual-specific:    both arms at a common L in 45-68")
    print("      margin over chance >= %.2f pp" % (45 - CHANCE))
    print("  NO ARM IS PREDICTED AT OR NEAR THE FLOOR OR CEILING UNDER EITHER ACCOUNT.")
    print("  The only route to a floored batch is total training failure -> G-FLOOR.")

    print("\n%s" % ("ALL PASS" if not GATE_FAIL else "FAILURES: %d" % len(GATE_FAIL)))
    for f in GATE_FAIL:
        print("   FAIL", f)
    return 1 if GATE_FAIL else 0


def main():
    ap = argparse.ArgumentParser(
        description="cvg1 registered scorer.  Documented invocation: "
                    "python3 analysis/cVG1_vggbn_gap_score.py <runsdir>")
    ap.add_argument("runsdir", nargs="?", default="")
    ap.add_argument("--runsdir", dest="runsdir_kw", default="")
    ap.add_argument("--csv", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "all_runs.csv"))
    ap.add_argument("--manifest", default="",
                    help="defaults to <runsdir>/cvg1/PARTITION-MANIFEST.txt")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    runsdir = a.runsdir or a.runsdir_kw
    if a.selftest:
        return selftest(a.csv, runsdir or ".")
    if not runsdir:
        ap.error("a runsdir is required: python3 %s <runsdir>" % os.path.basename(__file__))
    return score(runsdir, a.csv, a.manifest)


if __name__ == "__main__":
    sys.exit(main())
