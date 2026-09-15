#!/usr/bin/env python3
# =============================================================================
# cAU1_unaug_denominator_score.py -- THE REGISTERED SCORER FOR `cau1`:
#   DOES THE DENOMINATOR SURVIVE IN THE PARENT PAPER'S OWN, UNAUGMENTED SETTING?
#   ONE SUBMISSION, 27 JOBS (7 baseline rungs + 2 MetaOptimize arms, x 3 seeds),
#   ResNet18 / CIFAR-10 / batch 100 / 100 epochs / AUGMENT=0, SEEDS {50,51,52}.
#
# Committed BEFORE any cau1 run exists (STANDING RULE 21).  RUN IT UNEDITED
# (RULE 16).  The documented invocation takes ONE argument:
#
#     python3 analysis/cAU1_unaug_denominator_score.py <runsdir>
#
# --csv is OPTIONAL and DEFAULTED (the 164.2 pattern).  --selftest re-derives
# every frozen premise from the corpus (cau1- EXCLUDED from every reader),
# re-derives the record-based reasons for the choice of PRIMARY, checks the
# floor gate, and drives the REAL score() over synthetic run directories so
# that a green selftest means the scorer can score (171.9 item 2).
#
# =============================================================================
# THE QUESTION, AND WHY IT IS THE RIGHT ONE
# =============================================================================
# The campaign's strongest publishable result is the DENOMINATOR: MetaOptimize
# does not beat a properly tuned non-meta baseline (CIFAR-100 GAP_in +5.699 pp,
# cdn2 at 175; CIFAR-10 -1.807 pp, bl-sgd-01 vs i3b-3e4).  EVERY run behind it
# is AUGMENT=1.  The parent paper never augments and its released code has
# none.  A referee can therefore say the deficit is an artefact of moving the
# method out of its own setting.  CLOSEOUT item 5b DECLINED this batch as
# "re-measures the dead method claim"; that rationale is now circular -- the
# "dead method claim" IS the denominator, and the objection is precisely that
# it might be alive in the parent's setting.  What the corpus holds at
# AUGMENT=0 is 27 MetaOptimize rows (pp 9, PP 9, ub9 9: a GRANULARITY contrast,
# CORRECTIONS 10's flaw-3 "Running:" line executed three times) and ONE
# non-meta row (gate0_adamw_s1, AdamW at the parent's own lr=1e-5, n=1,
# augment unrecorded).  No TUNED non-meta baseline has ever run unaugmented.
#
# =============================================================================
# THE ARMS (all AUGMENT=0; everything else per arm is a byte copy of an
# existing, cited configuration -- nothing is invented)
# =============================================================================
#   lr{0005,001,002,005,01,02,04}   plain SGD + momentum 0.9 + wd 5e-4 +
#       cosine-to-zero, COS_TOTAL=50000 (= 100 ep x 500 steps), warmup 1000
#       steps: cdn1's arm-A recipe (175), i.e. the DENOMINATOR's baseline
#       family.  LR ladder {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4}.  The two
#       end rungs extend cdn1's {0.01..0.3} on BOTH sides: 0.4 because an
#       unaugmented optimum may move up, 0.005 because the record's only
#       CIFAR-10 cosine ladder puts its AUC argmax at its LOWEST rung (B2).
#   m6   MetaOptimize, THE PARENT'S OWN CONFIG: AdamW base + Adam meta,
#        alpha0 1e-6, eta 1e-3, gamma 1, the parent's 6-block partition
#        (resnet18_blocks), BETA_CLIP=-60:6.0, PROBE=5.  Byte-identical to
#        `ub9-b6` (c88, 119.10) except seed / run-name / save-dir / probe-dir.
#        -60:6.0 because it is the nearest the harness gets to the parent's
#        NO box while still measuring the walls: ub9-b6 touched neither wall
#        on 0/3 seeds (149.4, full streaming pass).
#   m6t  MetaOptimize, THE CIFAR-10 DENOMINATOR'S OWN META CELL: the same with
#        alpha0 3e-4 and the canonical box -15:-2.3026 -- byte-identical to
#        `i3b-3e4` (c45) except AUGMENT, PROBE=5 (instrument; inert by
#        FINDINGS 74.2) and seed / names.  It is there because i3b at AUG=1
#        puts alpha0 3e-4 +1.726 pp above the parent's 1e-6 (91.591 -> 93.317),
#        so a deficit measured against 1e-6 alone could be blamed on an
#        untuned alpha0.  The method gets the better of its two cells.
#
# =============================================================================
# THE PRIMARY, AND WHY IT IS AN ENDPOINT DESPITE CORRECTIONS 10
# =============================================================================
#   GAP_END = plateau5(S*) - plateau5(M*)            [pp, and in SE units]
#     S* = the ladder rung with the highest 3-seed mean plateau5
#     M* = the better of {m6, m6t} by 3-seed mean plateau5
#     plateau5 = mean TEST accuracy over epochs 95..99 from the RAW .out.
#   Positive = the tuned baseline beats the method (the denominator holds).
#   Bounded in NEITHER direction: every arm is predicted at >= 55 pp and <= 95
#   pp on a 10-class problem (section C).
#
# CORRECTIONS 10 (flaw 3) wrote "their evidence is Fig. 1 learning curves, so
# the comparison must be on convergence, not the endpoint".  That was about the
# PARENT'S claim (MetaOptimize vs a fixed-step baseline).  This batch tests
# the DENOMINATOR, whose statistic is plateau5, and three facts on the record
# decide against a curve statistic as PRIMARY here (re-derived in section B):
#   B1  At AUGMENT=0 the method's curve IS its endpoint: on all 9 ub9 runs
#       |AUC - plateau5| <= 0.70 pp and TRAIN >= 99 % by epoch 5-6.  A curve
#       statistic would move only through the BASELINE's curve shape.
#   B2  On the record's only CIFAR-10 cosine ladder (bl-sgd, AUG=1) AUC falls
#       MONOTONICALLY with lr (90.10 > 89.52 > 86.21 > 77.60) while plateau5
#       peaks INTERIOR at 0.1.  An AUC primary on a cosine ladder has its
#       optimum at the ladder's lower edge by construction, and it measures the
#       anneal's high-lr phase -- which MASTER-TABLE row 15 already names as the
#       source of the gap.  The interior-optimum requirement cannot be met.
#   B3  TRAIN plateau is at its CEILING at AUGMENT=0 (ub9 final_train
#       99.81-100.00), so a TRAIN endpoint is bounded and is barred as a
#       primary; TRAIN is reported as a convergence statistic instead.
# The parent's evidence type is NOT dropped: it is the registered SECONDARY,
# with its own bars and branches, read against the baseline rung that is best
# ON THAT STATISTIC (the fairest reading on curves), and that is exactly where
# a reversal in the method's favour is most reachable.
#
#   CURVE (secondary): GAP_AUC = AUC(S^auc) - AUC(M^auc),  AUC = mean TEST
#     accuracy over all 100 epochs from the raw .out; S^auc / M^auc are the
#     AUC-argmax rung and meta arm.  GAP_AUC at S* is printed beside it.
#
# =============================================================================
# BARS (SE units; noise floor re-derived at registration, cau1- excluded)
# =============================================================================
# Pooled within-cell SD, cell = the 15 design columns + BATCH PREFIX (so a
# cross-batch offset can never masquerade as seed noise), usable rows only,
# ResNet18 / CIFAR-10 / batch 100 / 100 epochs:
#   plateau5  AUGMENT=0 MetaOptimize (pp, PP, ub9)  0.446775  df 18, 9 cells
#             non-meta SGD+cosine (bl-sgd, AUG=1)    0.120142  df 16, 4 cells
#   AUC       AUGMENT=0 MetaOptimize                  0.385791  df 18
#             non-meta SGD+cosine                     0.162216  df 16
# SIGMA_PRIOR = the max of each pair.  At scoring, SIGMA_USED = max(prior,
# this batch's own pooled within-arm SD) -- the conservative of the two, and
# the choice is stamped.  SE = SIGMA_USED * sqrt(2/3)  (3 v 3, within batch).
# Every branch boundary is 2 SE.
#
# =============================================================================
# THE BRANCH MAP (in order; the first that applies is the token)
# =============================================================================
#   INCOMPLETE                   not every one of the 27 (arm, seed) has a
#                                100-epoch .out with RUN_DONE and no traceback
#   GATED-<why>                  an ENV or ARGS line departs from the design
#                                (AUGMENT=0 is checked on EVERY run)
#   GATED-METHOD-AT-FLOOR        neither meta arm trains (mean <= 40, or a
#                                diverged seed) -- the floor gate predicted
#                                this impossible; if it happens it is reported
#                                and nothing is branched on
#   GATED-BASELINE-AT-FLOOR      the argmax rung does not train
#   DEFICIT-HOLDS                GAP_END >= +2 SE      (+ LOWER-BOUND if the
#                                ladder argmax is at an edge: the baseline can
#                                only get better, so the deficit only grows)
#   DEFICIT-CLOSES               |GAP_END| < 2 SE and the ladder is INTERIOR:
#                                in its own setting the method TIES a tuned
#                                baseline.  "Does not beat" survives; "loses"
#                                does not.
#   UNRESOLVED-TIE-LADDER-EDGE   |GAP_END| < 2 SE with an edge argmax
#   REVERSES-AT-PAPER-CONFIG     GAP_END <= -2 SE, ladder INTERIOR, and the
#                                parent's own config m6 ALSO clears -2 SE:
#                                the denominator is an augmentation artefact
#                                and the method WINS in its own setting.  A
#                                real finding in the method's favour.
#   REVERSES-AT-TUNED-ALPHA0-ONLY  the same with only m6t clearing
#   UNRESOLVED-REVERSAL-LADDER-EDGE  GAP_END <= -2 SE but the baseline's
#                                argmax is at an edge -- it may be under-tuned,
#                                so no reversal is claimed
# CURVE tokens, same bars on GAP_AUC: CURVE-DEFICIT (+LOWER-BOUND at an AUC
# edge), CURVE-TIE / CURVE-UNRESOLVED-TIE-EDGE, CURVE-REVERSES /
# CURVE-UNRESOLVED-REVERSAL-EDGE.  The curve token NEVER moves the primary.
#
# Selection: M* is the max of two arms and S* the max of seven, both chosen
# on the same data they are scored on.  The expected upward bias of a max of
# k 3-seed means, in the worst case where the k cells are truly EQUAL, is
# ~E[max of k N(0,1)] * sigma/sqrt(3) (0.564 for k=2, 1.352 for k=7); both are
# printed in pp.  They push in OPPOSITE directions on GAP_END: picking S*
# inflates GAP_END (toward DEFICIT, against the method, <= 0.349 pp at the
# prior sigma), picking M* deflates it (toward REVERSAL, <= 0.146 pp).  Each is
# under the 2 SE bar (0.730 pp); neither is branched on.  A ladder whose rungs
# really differ (the record's do, by ~1 pp: bl-sgd, cdn1) carries far less.
#
# =============================================================================
# THE FLOOR GATE (164.6) -- NO ARM MAY BE PREDICTED AT OR NEAR THE FLOOR
# =============================================================================
# Chance on CIFAR-10 is 10 %.  Registered predictions, under EITHER account --
#   D (the deficit is a property of the method; setting-independent) or
#   A (the deficit is an augmentation artefact):
#     m6    74.797  -- ub9-b6's in-cell mean at the IDENTICAL spec, n=3
#     m6t   in [72, 80] -- unmeasured unaugmented; the lowest AUGMENT=0 meta
#           cell in the corpus is ub9-sc at 72.422
#     every ladder rung >= 55 -- an ARGUMENT, not a measurement: the only
#           non-meta optimiser ever run unaugmented here (AdamW at lr=1e-5,
#           ~3 orders of magnitude smaller steps than any rung) reached
#           58.426 in 100 epochs
#   D predicts GAP_END >= +2 SE; A predicts GAP_END <= +2 SE.
# Lowest level any account predicts for any arm: 55 pp = 45 pp = ~123 SE above
# chance.  Highest: well under the 100 % TEST ceiling.  PASS by construction,
# re-checked in section C.
#
# =============================================================================
# WHAT cau1 CANNOT DO, WRITTEN BEFORE ANY RUN EXISTS (stamped unconditionally)
# =============================================================================
#  * ONE baseline family (SGDm + cosine).  Not AdamW-cosine, not constant-LR.
#  * The COUNTERWEIGHT -- "against the fixed-step baseline the parent used,
#    MetaOptimize WINS (+1.551)" -- is NOT re-measured unaugmented.
#  * alpha0 gets TWO cells (the parent's 1e-6 and the AUG=1 optimum 3e-4), not
#    a ladder; the meta arm's own unaugmented optimum is not located.
#  * CIFAR-10 / ResNet18 only.  The CIFAR-100 GAP_in +5.699 stays AUGMENT=1.
#  * AT AUGMENT=0 TRAIN IS AT ITS CEILING, so the endpoint compares the
#    generalisation of interpolating solutions.  That is the regime the parent
#    chose; it is stated beside every number, not argued away.
#  * The comparison with the augmented deficit (+1.807) is CROSS-BATCH and
#    CROSS-SETTING.  It is printed; it is never a token.
# =============================================================================
import argparse
import contextlib
import csv
import io
import math
import os
import re
import struct
import sys
import tempfile

PREFIX = "cau1-"
NET = "ResNet18"
DSET = "CIFAR10"
EPOCHS = 100
BATCH = 100
AUG = "0"
SEEDS = (50, 51, 52)
LADDER = [("0.005", "0005"), ("0.01", "001"), ("0.02", "002"), ("0.05", "005"),
          ("0.1", "01"), ("0.2", "02"), ("0.4", "04")]
COS_TOTAL = "50000"
COS_WARMUP = "1000"
META = {
    "m6": {"alpha0": "1e-6", "clip": "-60:6.0",
           "role": "the PARENT'S OWN config (ub9-b6 spec)"},
    "m6t": {"alpha0": "3e-4", "clip": "-15:-2.3026",
            "role": "the CIFAR-10 denominator's OWN meta cell (i3b-3e4 spec)"},
}
META_ARMS = ("m6", "m6t")
LADDER_ARMS = tuple("lr" + t for _, t in LADDER)
ARMS = LADDER_ARMS + META_ARMS
PROBE_META = "5"

# ---- the noise floor, frozen at registration (cau1- excluded) --------------
SIGMA_P5_AUG0 = 0.446775
SIGMA_P5_AUG0_DF = 18
SIGMA_P5_SGD = 0.120142
SIGMA_P5_SGD_DF = 16
SIGMA_AUC_AUG0 = 0.385791
SIGMA_AUC_AUG0_DF = 18
SIGMA_AUC_SGD = 0.162216
SIGMA_AUC_SGD_DF = 16
SIGMA_P5_PRIOR = max(SIGMA_P5_AUG0, SIGMA_P5_SGD)
SIGMA_AUC_PRIOR = max(SIGMA_AUC_AUG0, SIGMA_AUC_SGD)
SE_FACTOR = math.sqrt(2.0 / 3.0)
SIGMA_TOL = 0.02

# ---- reference levels, frozen (cau1- excluded) -------------------------------
REF = {
    "ub9-b6 (m6's spec, AUG=0, n=3)": 74.7967,
    "i3b-3e4 (m6t's spec, AUG=1, n=3)": 93.3173,
    "i3b-1e6 (paper alpha0, AUG=1, n=3)": 91.5907,
    "bl-sgd-01 (augmented denominator baseline, n=5)": 95.1244,
    "gate0_adamw_s1 (only unaugmented non-meta row, n=1)": 58.426,
}
REF_SD = {"bl-sgd-01": 0.105256, "i3b-3e4": 0.142945}
GAP_AUG = 95.1244 - 93.3173          # +1.8071, CROSS-BATCH, AUG=1
REF_TOL = 0.001

# ---- gates and bars -----------------------------------------------------------
Z_BAR = 2.0
CHANCE = 10.0
FLOOR_MIN = 40.0                     # trains-at-all (cdn1's V1 threshold)
DIVERGED_BAR = CHANCE + 5.0          # a seed at or below this has diverged
PRED_MIN_ANY_ARM = 55.0
PRED_MAX_ANY_ARM = 95.0
CEIL_TEST = 100.0
FLOOR_SE_MIN = 20.0
E_MAX = {2: 0.5642, 7: 1.3522}       # E[max of k iid N(0,1)]

CELLKEYS = ["network", "dataset", "granularity", "base", "meta", "meta_stepsize",
            "alpha0", "gamma", "augment", "beta_clip", "batch_size",
            "epochs_requested", "hier", "lam", "eta_ratio"]

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "all_runs.csv")

EPTR_RE = re.compile(r"Epoch\s+(\d+),\s*Train Accuracy:\s*([0-9.]+)\s*%,"
                     r"\s*Test Accuracy:\s*([0-9.]+)")
OUT_RE = re.compile(r"^cau1-(lr\d+|m6t|m6)-s(\d+)-(\d+)\.out$")

GATE_FAIL = []


def chk(cond, label, extra=""):
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label,
                           ("   " + extra) if extra else ""))
    if not cond:
        GATE_FAIL.append(label)
    return bool(cond)


def note(label, extra=""):
    print("  NOTE %s%s" % (label, ("   " + extra) if extra else ""))


def skip(label, extra=""):
    print("  SKIP %s%s" % (label, ("   " + extra) if extra else ""))


def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def f32(x):
    return struct.unpack("f", struct.pack("f", float(x)))[0]


# =============================================================================
# the design, as ONE function each, so the launcher can import and compare
# =============================================================================
def run_name(arm, seed):
    return "%s%s-s%d" % (PREFIX, arm, seed)


def arm_names():
    return [run_name(a, s) for s in SEEDS for a in ARMS]


def lr_of(arm):
    for v, t in LADDER:
        if arm == "lr" + t:
            return v
    return None


def expected_args(arm, seed):
    """flag -> value for the train.py half of the command.  --save-directory
    is checked by suffix (the workspace is host-specific)."""
    if arm in META:
        d = {"--optimizer": "HF", "--alg-base": "AdamW",
             "--normalizer-param-base": "0.999", "--momentum-param-base": "0.9",
             "--weight-decay-base": "0.1", "--alg-meta": "Adam",
             "--normalizer-param-meta": "0.999", "--momentum-param-meta": "0.9",
             "--weight-decay-meta": "0", "--dataset": DSET, "--NN-name": NET,
             "--batch-size": str(BATCH), "--max-time": "999:00:00", "--gamma": "1",
             "--meta-stepsize": "1e-3", "--alpha0": META[arm]["alpha0"],
             "--num-epochs": str(EPOCHS), "--stepsize-groups": "resnet18_blocks"}
    else:
        d = {"--optimizer": "SGD", "--dataset": DSET, "--NN-name": NET,
             "--batch-size": str(BATCH), "--max-time": "999:00:00",
             "--num-epochs": str(EPOCHS), "--alpha0": lr_of(arm)}
    d["--seed"] = str(seed)
    d["--run-name"] = run_name(arm, seed)
    d["--save-directory"] = "*/runs/cau1"
    return d


def expected_env(arm, seed):
    """KEY -> value on the runner's ENV line.  PROBE_DIR is checked by suffix."""
    d = {"AUGMENT": AUG, "HIER": "none", "LAM": "na", "ETA_RATIO": "na",
         "SCHED": "none", "SCHED_TOTAL": "none", "SCHED_WARMUP": "none",
         "SCHED_MIN": "none"}
    if arm in META:
        d.update({"BETA_CLIP": META[arm]["clip"], "COS_TOTAL": "default",
                  "COS_WARMUP": "default", "PROBE": PROBE_META,
                  "PROBE_DIR": "*/cau1/probe_" + run_name(arm, seed)})
    else:
        d.update({"BETA_CLIP": "none", "COS_TOTAL": COS_TOTAL,
                  "COS_WARMUP": COS_WARMUP, "PROBE": "0", "PROBE_DIR": "none"})
    return d


OPTIONAL_ENV = {"EB_RHO": "na", "EB_LOG": "0"}   # echoed by the live runner only


def parse_args_line(line):
    toks = line.split()[1:]
    flags, cur, dup = {}, None, []
    for t in toks:
        if t.startswith("--"):
            k, v = (t.split("=", 1) + [None])[:2] if "=" in t else (t, None)
            if k in flags:
                dup.append(k)
            flags[k] = v
            cur = k if v is None else None
        elif cur is not None:
            flags[cur] = t if flags[cur] is None else flags[cur] + " " + t
    return flags, dup


def parse_env_line(line):
    d = {}
    for t in line.split()[1:]:
        if "=" in t:
            k, v = t.split("=", 1)
            d[k] = v
    return d


def _match(want, got):
    if want.startswith("*"):
        return got is not None and got.endswith(want[1:])
    return got == want


# =============================================================================
# corpus readers -- `cau1-` EXCLUDED FROM EVERY ONE
# =============================================================================
def read_corpus(csvpath):
    if not csvpath or not os.path.exists(csvpath):
        return None
    return [r for r in csv.DictReader(open(csvpath))
            if not r.get("run", "").startswith(PREFIX)]


def usable(r):
    return (r.get("superseded") == "0" and r.get("collapsed") == "0"
            and r.get("complete") == "1" and r.get("plateau5", "").strip() != "")


def batch_of(r):
    return re.split(r"[-_]", r.get("run", ""))[0]


def c10_std(r):
    return (r.get("network") == NET and r.get("dataset") == DSET
            and r.get("epochs_requested") == str(EPOCHS)
            and r.get("batch_size") == str(BATCH))


def is_aug0_meta(r):
    return c10_std(r) and r.get("augment") == "0"


def is_sgd_nonmeta(r):
    return c10_std(r) and r.get("base") == "SGD" and r.get("meta") == "?"


def pooled_sigma(rows, pred, field="plateau5"):
    cells = {}
    for r in rows:
        if usable(r) and pred(r) and r.get(field, "").strip():
            k = tuple(r[c] for c in CELLKEYS) + (batch_of(r),)
            cells.setdefault(k, []).append(float(r[field]))
    ss, df, nc = 0.0, 0, 0
    for v in cells.values():
        if len(v) < 2:
            continue
        m = mean(v)
        ss += sum((x - m) ** 2 for x in v)
        df += len(v) - 1
        nc += 1
    return (math.sqrt(ss / df) if df else None), df, nc


def cell_mean(rows, pred, field="plateau5"):
    v = [float(r[field]) for r in rows if usable(r) and pred(r)
         and r.get(field, "").strip()]
    return mean(v), sd(v), len(v)


# =============================================================================
# run readers -- the RAW .out files are PRIMARY; the CSV is never the source of
# a cau1 level
# =============================================================================
def parse_out(path):
    txt = open(path, errors="replace").read()
    eps = {}
    for a, b, c in EPTR_RE.findall(txt):
        eps[int(a)] = (float(b), float(c))
    lines = txt.splitlines()
    args = [ln for ln in lines if ln.startswith("ARGS:")]
    env = [ln for ln in lines if ln.startswith("ENV:")]
    node = [ln for ln in lines if ln.startswith("NODE=")]
    rec = {"n_epochs": len(eps), "epochs": eps,
           "traceback": "Traceback (most recent call last)" in txt,
           "run_done": "RUN_DONE" in txt, "args": args, "env": env, "node": node,
           "plateau5": None, "train5": None, "auc": None, "auc_train": None,
           "ep_train99": None}
    if all(e in eps for e in range(EPOCHS)):
        te = [eps[e][1] for e in range(EPOCHS)]
        tr = [eps[e][0] for e in range(EPOCHS)]
        rec["plateau5"] = mean(te[-5:])
        rec["train5"] = mean(tr[-5:])
        rec["auc"] = mean(te)
        rec["auc_train"] = mean(tr)
        rec["ep_train99"] = next((e for e in range(EPOCHS) if tr[e] >= 99.0), None)
    return rec


def complete(rec):
    return (rec["n_epochs"] == EPOCHS and all(e in rec["epochs"] for e in range(EPOCHS))
            and rec["run_done"] and not rec["traceback"])


def read_runs(runsdir):
    """-> {(arm, seed): rec}.  When a run name has several .out files (an
    abandoned resubmit shadowing the real run, 149.2), a COMPLETE file beats an
    incomplete one, and the higher job id breaks ties.  Every duplicate is
    reported."""
    got, dups = {}, []
    if not runsdir or not os.path.isdir(runsdir):
        return got, dups
    for fn in sorted(os.listdir(runsdir)):
        m = OUT_RE.match(fn)
        if not m:
            continue
        arm, seed, jid = m.group(1), int(m.group(2)), int(m.group(3))
        if arm not in ARMS:
            continue
        rec = parse_out(os.path.join(runsdir, fn))
        rec.update({"file": fn, "job": jid, "arm": arm, "seed": seed})
        prev = got.get((arm, seed))
        if prev is not None:
            dups.append((arm, seed, prev["file"], fn))
            if (complete(prev), prev["job"]) >= (complete(rec), rec["job"]):
                continue
        got[(arm, seed)] = rec
    return got, dups


def read_occupancy(runsdir, rn, clip):
    """Wall contact on the meta arms' own probe records.  A DISCLOSURE ONLY --
    never a gate, and deliberately NOT a dwell-time 'freedom' statistic (the
    ciso2 defect, 196): it answers only 'did the box ever hold beta'."""
    p = os.path.join(runsdir, "cau1", "probe_" + rn, "probe.jsonl")
    if not os.path.exists(p):
        return None
    import json
    lo, hi = [f32(x) for x in clip.split(":")]
    n = n_lo = n_hi = 0
    first = None
    last = None
    for ln in open(p):
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        n += 1
        a_lo = r.get("n_at_lo")
        a_hi = r.get("n_at_hi")
        if a_lo is None and "beta_true_min" in r:
            a_lo = 1 if f32(r["beta_true_min"]) <= lo else 0
        if a_hi is None and "beta_true_max" in r:
            a_hi = 1 if f32(r["beta_true_max"]) >= hi else 0
        t_lo, t_hi = (a_lo or 0) > 0, (a_hi or 0) > 0
        n_lo += t_lo
        n_hi += t_hi
        if (t_lo or t_hi) and first is None:
            first = r.get("step")
        last = (t_lo, t_hi)
    return {"records": n, "at_lo": n_lo, "at_hi": n_hi, "first_step": first,
            "terminal": last}


# =============================================================================
PRIMARY_TOKENS = {("hi", True): "DEFICIT-HOLDS", ("hi", False): "DEFICIT-HOLDS",
                  ("mid", True): "DEFICIT-CLOSES", ("mid", False): "UNRESOLVED-TIE-LADDER-EDGE",
                  ("lo", True): "REVERSES", ("lo", False): "UNRESOLVED-REVERSAL-LADDER-EDGE"}
CURVE_TOKENS = {("hi", True): "CURVE-DEFICIT", ("hi", False): "CURVE-DEFICIT",
                ("mid", True): "CURVE-TIE", ("mid", False): "CURVE-UNRESOLVED-TIE-EDGE",
                ("lo", True): "CURVE-REVERSES", ("lo", False): "CURVE-UNRESOLVED-REVERSAL-EDGE"}


def _gap_token(z, interior, table):
    """z in SE units; interior = the relevant ladder argmax is not an end rung."""
    band = "hi" if z >= Z_BAR else ("mid" if z > -Z_BAR else "lo")
    return table[(band, bool(interior))]


def score(runsdir, csvpath=None):
    del GATE_FAIL[:]
    print("=" * 78)
    print(" cau1 -- DOES THE DENOMINATOR SURVIVE IN THE PARENT'S OWN SETTING?")
    print(" %s / %s / batch %d / %d epochs / AUGMENT=%s / seeds %s"
          % (NET, DSET, BATCH, EPOCHS, AUG, ",".join(str(s) for s in SEEDS)))
    print(" baseline: plain SGD+momentum+cosine, COS_TOTAL=%s COS_WARMUP=%s, lr ladder %s"
          % (COS_TOTAL, COS_WARMUP, "/".join(v for v, _ in LADDER)))
    for a in META_ARMS:
        print(" %-4s AdamW+Adam blk6, alpha0 %s, eta 1e-3, gamma 1, box %s -- %s"
              % (a, META[a]["alpha0"], META[a]["clip"], META[a]["role"]))
    print(" PRIMARY  GAP_END = plateau5(S*) - plateau5(M*), both 3-seed IN-BATCH means")
    print("          from the runs' OWN .out; plateau5 = mean TEST over epochs %d..%d."
          % (EPOCHS - 5, EPOCHS - 1))
    print(" SECONDARY GAP_AUC = AUC(S^auc) - AUC(M^auc), AUC = mean TEST over all %d epochs."
          % EPOCHS)
    print("=" * 78)

    runs, dups = read_runs(runsdir)
    for d in dups:
        note("duplicate .out for %s s%d: %s and %s -- the COMPLETE / later one is used"
             % d)

    # ---- G0 COMPLETENESS ------------------------------------------------------
    print("\nG0  COMPLETENESS (%d arms x %d seeds = %d runs)"
          % (len(ARMS), len(SEEDS), len(ARMS) * len(SEEDS)))
    missing, bad = [], []
    for s in SEEDS:
        for a in ARMS:
            rec = runs.get((a, s))
            if rec is None:
                missing.append(run_name(a, s))
            elif not complete(rec):
                bad.append("%s (%d epochs, RUN_DONE %s, traceback %s)"
                           % (rec["file"], rec["n_epochs"], rec["run_done"],
                              rec["traceback"]))
    have = len(ARMS) * len(SEEDS) - len(missing) - len(bad)
    print("    %d/%d complete" % (have, len(ARMS) * len(SEEDS)))
    for m in missing:
        print("    MISSING: %s" % m)
    for b in bad:
        print("    NOT COMPLETE: %s" % b)
    if missing or bad:
        print("\nFINAL: INCOMPLETE -- %d missing, %d not complete; NO number from this"
              " batch may be read" % (len(missing), len(bad)))
        return 1

    # ---- G1 ENV and G2 ARGS ---------------------------------------------------
    print("\nG1  ENV LINE (AUGMENT=0 on EVERY run; box, schedule, probe per family)")
    env_bad = 0
    for (a, s), rec in sorted(runs.items()):
        if len(rec["env"]) != 1:
            print("    %s: %d ENV lines" % (rec["file"], len(rec["env"])))
            env_bad += 1
            continue
        got = parse_env_line(rec["env"][0])
        for k, want in expected_env(a, s).items():
            if not _match(want, got.get(k)):
                print("    %s: %s=%s, want %s" % (rec["file"], k, got.get(k), want))
                env_bad += 1
        for k, want in OPTIONAL_ENV.items():
            if k in got and got[k] != want:
                print("    %s: %s=%s, want %s" % (rec["file"], k, got[k], want))
                env_bad += 1
        if not rec["node"] or "AUGMENT=0" not in rec["node"][0]:
            print("    %s: NODE header does not read AUGMENT=0" % rec["file"])
            env_bad += 1
    print("    G1: %s (%d violations)" % ("PASS" if env_bad == 0 else "FAIL", env_bad))

    print("\nG2  ARGS LINE (no repeated flag; every declared flag, nothing else)")
    args_bad = 0
    for (a, s), rec in sorted(runs.items()):
        if len(rec["args"]) != 1:
            print("    %s: %d ARGS lines" % (rec["file"], len(rec["args"])))
            args_bad += 1
            continue
        got, dup = parse_args_line(rec["args"][0])
        if dup:
            print("    %s: REPEATED FLAG(S) %s -- argparse keeps the LAST" % (rec["file"], dup))
            args_bad += 1
        want = expected_args(a, s)
        for k, w in want.items():
            if not _match(w, got.get(k)):
                print("    %s: %s=%s, want %s" % (rec["file"], k, got.get(k), w))
                args_bad += 1
        extra = sorted(set(got) - set(want))
        if extra:
            print("    %s: UNDECLARED flag(s) %s" % (rec["file"], extra))
            args_bad += 1
    print("    G2: %s (%d violations)" % ("PASS" if args_bad == 0 else "FAIL", args_bad))
    if env_bad or args_bad:
        why = []
        if env_bad:
            why.append("ENV")
        if args_bad:
            why.append("ARGS")
        print("\nFINAL: GATED-%s -- the runs are not the registered design; nothing is"
              " branched on" % "-".join(why))
        return 1

    # ---- per-arm table ------------------------------------------------------------
    arm = {}
    for a in ARMS:
        rs = [runs[(a, s)] for s in SEEDS]
        arm[a] = {
            "p5": [r["plateau5"] for r in rs], "tr5": [r["train5"] for r in rs],
            "auc": [r["auc"] for r in rs], "auct": [r["auc_train"] for r in rs],
            "e99": [r["ep_train99"] for r in rs],
            "div": [r["plateau5"] <= DIVERGED_BAR for r in rs]}
        for k in ("p5", "tr5", "auc", "auct"):
            arm[a]["m_" + k] = mean(arm[a][k])
        arm[a]["sd_p5"] = sd(arm[a]["p5"])
        arm[a]["sd_auc"] = sd(arm[a]["auc"])
        arm[a]["trains"] = arm[a]["m_p5"] > FLOOR_MIN and not any(arm[a]["div"])

    print("\n    ARM     n  TEST plateau5     TEST AUC    TRAIN plateau5  TRAIN AUC  ep(train>=99)")
    for a in ARMS:
        r = arm[a]
        e = ",".join("-" if x is None else str(x) for x in r["e99"])
        print("    %-6s %2d  %7.3f sd %5.3f  %7.3f    %7.3f        %7.3f    %s%s"
              % (a, len(SEEDS), r["m_p5"], r["sd_p5"], r["m_auc"], r["m_tr5"],
                 r["m_auct"], e, "   DIVERGED SEED" if any(r["div"]) else ""))

    # ---- sigma ------------------------------------------------------------------
    ss = df = 0
    ssa = 0.0
    for a in ARMS:
        if any(arm[a]["div"]):
            continue
        m, ma = arm[a]["m_p5"], arm[a]["m_auc"]
        ss += sum((x - m) ** 2 for x in arm[a]["p5"])
        ssa += sum((x - ma) ** 2 for x in arm[a]["auc"])
        df += len(SEEDS) - 1
    sig_in = math.sqrt(ss / df) if df else 0.0
    sig_in_auc = math.sqrt(ssa / df) if df else 0.0
    sig = max(SIGMA_P5_PRIOR, sig_in)
    sig_auc = max(SIGMA_AUC_PRIOR, sig_in_auc)
    se, se_auc = sig * SE_FACTOR, sig_auc * SE_FACTOR
    sig_stamp = "SIGMA-PRIOR-DOMINATES" if SIGMA_P5_PRIOR >= sig_in else "SIGMA-INBATCH-DOMINATES"
    print("\nSIGMA  prior %.4f (df %d) | in-batch %.4f (df %d, diverged arms excluded)"
          " -> USED %.4f, SE = %.4f pp  [%s]" % (SIGMA_P5_PRIOR, SIGMA_P5_AUG0_DF,
                                                 sig_in, df, sig, se, sig_stamp))
    print("       AUC: prior %.4f | in-batch %.4f -> USED %.4f, SE_AUC = %.4f pp"
          % (SIGMA_AUC_PRIOR, sig_in_auc, sig_auc, se_auc))

    # ---- V1 trains-at-all, V2 bracketing -------------------------------------------
    print("\nV1  TRAINS-AT-ALL (mean plateau5 > %.0f and no seed <= %.0f)" % (FLOOR_MIN, DIVERGED_BAR))
    ok_meta = [a for a in META_ARMS if arm[a]["trains"]]
    for a in ARMS:
        print("    %-6s %s" % (a, "trains" if arm[a]["trains"] else "DOES NOT TRAIN"))
    if not ok_meta:
        print("\nFINAL: GATED-METHOD-AT-FLOOR -- neither MetaOptimize arm trains; the"
              " registered floor gate predicted this impossible.  Nothing is branched on.")
        return 1
    means = [arm[a]["m_p5"] for a in LADDER_ARMS]
    i_star = max(range(len(LADDER_ARMS)), key=lambda i: means[i])
    s_star = LADDER_ARMS[i_star]
    if not arm[s_star]["trains"]:
        print("\nFINAL: GATED-BASELINE-AT-FLOOR -- the best rung (%s) does not train." % s_star)
        return 1
    interior = 0 < i_star < len(LADDER_ARMS) - 1
    ladder_tok = ("INTERIOR" if interior else
                  ("EDGE-LOW" if i_star == 0 else "EDGE-HIGH"))
    print("\nV2  BRACKETING on plateau5: argmax %s (lr=%s) at %.3f -- %s"
          % (s_star, LADDER[i_star][0], means[i_star], ladder_tok))
    for i, a in enumerate(LADDER_ARMS):
        print("      %s lr=%-6s %.3f%s" % (a, LADDER[i][0], means[i],
                                          "   <- S*" if i == i_star else ""))

    # ---- the PRIMARY --------------------------------------------------------------
    m_star = max(ok_meta, key=lambda a: arm[a]["m_p5"])
    gap = arm[s_star]["m_p5"] - arm[m_star]["m_p5"]
    z = gap / se
    print("\n" + "=" * 78)
    print("THE PRIMARY: GAP_END = plateau5(S*) - plateau5(M*), WITHIN batch")
    print("=" * 78)
    print("    S*  %-6s (lr=%s)   %.4f pp" % (s_star, LADDER[i_star][0], arm[s_star]["m_p5"]))
    print("    M*  %-6s            %.4f pp   (%s)" % (m_star, arm[m_star]["m_p5"], META[m_star]["role"]))
    print("    GAP_END             %+.4f pp = %+.2f SE   (SE %.4f, bar +/- %.1f SE = %.4f pp)"
          % (gap, z, se, Z_BAR, Z_BAR * se))
    zs = {}
    for a in META_ARMS:
        if arm[a]["trains"]:
            g = arm[s_star]["m_p5"] - arm[a]["m_p5"]
            zs[a] = g / se
            print("      vs %-4s  %+.4f pp = %+.2f SE" % (a, g, zs[a]))
        else:
            print("      vs %-4s  arm does not train -- excluded from M*" % a)
    bias_m = E_MAX[2] * sig / math.sqrt(3.0)
    bias_s = E_MAX[7] * sig / math.sqrt(3.0)
    print("    selection: max-of-2 meta bias ~%.3f pp, max-of-7 rung bias ~%.3f pp"
          " (opposite signs on GAP_END)" % (bias_m, bias_s))

    tok = _gap_token(z, interior, PRIMARY_TOKENS)
    if tok == "DEFICIT-HOLDS" and not interior:
        tok = "DEFICIT-HOLDS | LOWER-BOUND"
    if tok == "REVERSES":
        tok = ("REVERSES-AT-PAPER-CONFIG" if zs.get("m6", 0.0) <= -Z_BAR
               else "REVERSES-AT-TUNED-ALPHA0-ONLY")
    print("    BRANCH: %s" % tok)

    # ---- the SECONDARY (curves) ----------------------------------------------------
    aucs = [arm[a]["m_auc"] for a in LADDER_ARMS]
    j_star = max(range(len(LADDER_ARMS)), key=lambda i: aucs[i])
    s_auc = LADDER_ARMS[j_star]
    interior_auc = 0 < j_star < len(LADDER_ARMS) - 1
    m_auc = max(ok_meta, key=lambda a: arm[a]["m_auc"])
    gc = arm[s_auc]["m_auc"] - arm[m_auc]["m_auc"]
    zc = gc / se_auc
    gc_at_s = arm[s_star]["m_auc"] - arm[m_auc]["m_auc"]
    print("\nSECONDARY (the parent's evidence type): GAP_AUC = AUC(S^auc) - AUC(M^auc)")
    print("    S^auc %-6s (lr=%s)  %.4f   [%s]" % (s_auc, LADDER[j_star][0], arm[s_auc]["m_auc"],
                                             "INTERIOR" if interior_auc else "AT AN EDGE"))
    print("    M^auc %-6s           %.4f" % (m_auc, arm[m_auc]["m_auc"]))
    print("    GAP_AUC  %+.4f pp = %+.2f SE_AUC;   at S* instead: %+.4f pp = %+.2f SE_AUC"
          % (gc, zc, gc_at_s, gc_at_s / se_auc))
    ctok = _gap_token(zc, interior_auc, CURVE_TOKENS)
    if ctok == "CURVE-DEFICIT" and not interior_auc:
        ctok = "CURVE-DEFICIT-LOWER-BOUND"
    print("    CURVE: %s" % ctok)

    # ---- TRAIN, the ceiling ----------------------------------------------------------
    ceil = all(arm[a]["m_tr5"] >= 99.5 for a in ARMS)
    print("\nTRAIN  plateau at ceiling on every arm: %s  -> TRAIN is read as AUC and"
          " epoch-to-99%%, never as an endpoint" % ceil)
    print("       TRAIN-AUC gap (S* - M*): %+.4f pp" % (arm[s_star]["m_auct"] - arm[m_star]["m_auct"]))

    # ---- disclosures, non-gating ------------------------------------------------------
    print("\nDISCLOSURES (none gates, none moves a token)")
    for a in META_ARMS:
        for s in SEEDS:
            rn = run_name(a, s)
            oc = read_occupancy(runsdir, rn, META[a]["clip"])
            if oc is None:
                print("    box occupancy %-14s NOT AVAILABLE (no probe.jsonl) -- not 'not binding'" % rn)
            else:
                print("    box occupancy %-14s %d records: floor %d, ceiling %d, first step %s, terminal %s"
                      % (rn, oc["records"], oc["at_lo"], oc["at_hi"], oc["first_step"], oc["terminal"]))
    print("    BETWEEN-BATCH concordance, m6 vs ub9-b6 at the identical spec: %.4f vs %.4f = %+.4f pp"
          " (%+.2f SE)" % (arm["m6"]["m_p5"], REF["ub9-b6 (m6's spec, AUG=0, n=3)"],
                           arm["m6"]["m_p5"] - REF["ub9-b6 (m6's spec, AUG=0, n=3)"],
                           (arm["m6"]["m_p5"] - REF["ub9-b6 (m6's spec, AUG=0, n=3)"]) / se))
    print("    CROSS-SETTING, m6t (AUG=0) vs i3b-3e4 (AUG=1): %.4f vs %.4f = %+.4f pp"
          % (arm["m6t"]["m_p5"], REF["i3b-3e4 (m6t's spec, AUG=1, n=3)"],
             arm["m6t"]["m_p5"] - REF["i3b-3e4 (m6t's spec, AUG=1, n=3)"]))
    se_aug = math.sqrt(REF_SD["bl-sgd-01"] ** 2 / 5 + REF_SD["i3b-3e4"] ** 2 / 3)
    print("    CROSS-BATCH, CROSS-SETTING: GAP_END %+.4f (SE %.4f) vs the augmented CIFAR-10 deficit"
          " %+.4f (seed SE %.4f, and its two cells are DIFFERENT batches) -- difference %+.4f pp."
          "  PRINTED, NEVER A TOKEN." % (gap, se, GAP_AUG, se_aug, gap - GAP_AUG))

    stamps = ["LADDER:" + ladder_tok, "M*=" + m_star, sig_stamp, ctok]
    if ceil:
        stamps.append("TRAIN-AT-CEILING")
    stamps += ["ONE-BASELINE-FAMILY", "COUNTERWEIGHT-NOT-TESTED", "ALPHA0-TWO-CELLS",
               "CIFAR10-ONLY"]
    print("\n    SCOPE.  In-batch, AUGMENT=0, ResNet18/CIFAR-10, 100 epochs, one baseline")
    print("    family.  It licenses a sentence about the DENOMINATOR in the parent's own")
    print("    setting and nothing about granularity, CIFAR-100, or the fixed-step")
    print("    counterweight.")
    print("\nFINAL: %s | %s" % (tok, " | ".join(stamps)))
    return 0


# =============================================================================
# SELFTEST
# =============================================================================
def _synth_out(d, arm, seed, jid, test, train, env_over=None, args_extra="",
               run_done=True, n_ep=EPOCHS, node_aug="0"):
    rn = run_name(arm, seed)
    a = expected_args(arm, seed)
    toks = []
    for k, v in a.items():
        if k == "--save-directory":
            v = "/ws/runs/cau1"
        toks += [k, v]
    args = "ARGS: " + " ".join(toks) + args_extra
    e = expected_env(arm, seed)
    if arm in META:
        e["PROBE_DIR"] = "/ws/runs/cau1/probe_" + rn
    e.update(OPTIONAL_ENV)
    if env_over:
        e.update(env_over)
    env = "ENV: " + " ".join("%s=%s" % (k, v) for k, v in e.items())
    lines = ["NODE=nodeX | JOB=%s %d | AUGMENT=%s" % (rn, jid, node_aug), args, env,
             "NVIDIA L4, 23034 MiB"]
    for ep in range(n_ep):
        lines.append("Epoch %d, Train Accuracy: %.2f %%, Test Accuracy: %.2f %%"
                     % (ep, train[ep], test[ep]))
    lines.append("43  minutes")
    if run_done:
        lines.append("RUN_DONE")
    open(os.path.join(d, "%s-%d.out" % (rn, jid)), "w").write("\n".join(lines) + "\n")


def _curve_meta(level, jitter):
    te = [min(level, 55.0 + 10.0 * e) for e in range(EPOCHS)]
    te = [t + (jitter if e >= 20 else 0.0) for e, t in enumerate(te)]
    tr = [min(100.0, 50.0 + 12.0 * e) for e in range(EPOCHS)]
    return te, tr


def _curve_sgd(level, mid, jitter):
    """a cosine-like curve: depressed at `mid` through the high-lr phase, rising
    to `level` over the last 20 epochs."""
    te = []
    for e in range(EPOCHS):
        if e < 80:
            te.append(mid + jitter)
        else:
            te.append(mid + (level - mid) * (e - 79) / 20.0 + jitter)
    te[-5:] = [level + jitter] * 5
    tr = [min(100.0, 40.0 + 2.0 * e) for e in range(EPOCHS)]
    return te, tr


def _build(d, ladder_levels, meta_levels, ladder_mid=None, jit=(0.0, 0.2, -0.2),
           drop=None, env_break=None, args_break=None, shadow=None):
    jid = 5000000
    for s_i, s in enumerate(SEEDS):
        for i, a in enumerate(LADDER_ARMS):
            if drop == (a, s):
                continue
            mid = ladder_mid[i] if ladder_mid else ladder_levels[i] - 8.0
            te, tr = _curve_sgd(ladder_levels[i], mid, jit[s_i])
            jid += 1
            eo = env_break[1] if env_break and env_break[0] == (a, s) else None
            ax = args_break[1] if args_break and args_break[0] == (a, s) else ""
            _synth_out(d, a, s, jid, te, tr, env_over=eo, args_extra=ax)
        for a in META_ARMS:
            if drop == (a, s):
                continue
            te, tr = _curve_meta(meta_levels[a], jit[s_i])
            jid += 1
            eo = env_break[1] if env_break and env_break[0] == (a, s) else None
            ax = args_break[1] if args_break and args_break[0] == (a, s) else ""
            _synth_out(d, a, s, jid, te, tr, env_over=eo, args_extra=ax)
    if shadow:
        a, s = shadow
        te, tr = _curve_meta(10.0, 0.0)
        _synth_out(d, a, s, 9999999, te, tr, run_done=False, n_ep=40)


def _run_score(d):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = score(d, None)
    except Exception as ex:          # a crash is a FAIL line, never a traceback
        return None, "EXCEPTION %s: %s" % (type(ex).__name__, ex)
    out = buf.getvalue()
    fin = [ln for ln in out.splitlines() if ln.startswith("FINAL:")]
    return rc, (fin[-1] if fin else "(no FINAL line)"), out


def selftest(csvpath, runsdir):
    print("cAU1_unaug_denominator_score.py --selftest")
    fails = []
    passes = [0]

    def T(cond, label, extra=""):
        print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
        if not cond:
            fails.append(label)
        else:
            passes[0] += 1

    skips = [0]

    def S(label, extra=""):
        skip(label, extra)
        skips[0] += 1

    # ---- A: premises vs the live corpus ------------------------------------------
    print("\nA. FROZEN PREMISES RE-DERIVED FROM THE CORPUS (cau1- excluded from every reader)")
    rows = read_corpus(csvpath)
    if rows is None:
        S("no corpus at %s -- section A and B not run" % csvpath)
    else:
        allrows = list(csv.DictReader(open(csvpath)))
        own = [r for r in allrows if r.get("run", "").startswith(PREFIX)]
        note("corpus %d rows; %d are cau1- (%s) and are excluded from every reader"
             % (len(allrows), len(own), "POST-INGEST" if own else "PRE-INGEST"))
        for lbl, pred, fld, want, wdf in (
                ("sigma plateau5 AUGMENT=0 MetaOptimize", is_aug0_meta, "plateau5", SIGMA_P5_AUG0, SIGMA_P5_AUG0_DF),
                ("sigma plateau5 non-meta SGD+cosine", is_sgd_nonmeta, "plateau5", SIGMA_P5_SGD, SIGMA_P5_SGD_DF),
                ("sigma AUC AUGMENT=0 MetaOptimize", is_aug0_meta, "auc", SIGMA_AUC_AUG0, SIGMA_AUC_AUG0_DF),
                ("sigma AUC non-meta SGD+cosine", is_sgd_nonmeta, "auc", SIGMA_AUC_SGD, SIGMA_AUC_SGD_DF)):
            s_, df_, nc_ = pooled_sigma(rows, pred, fld)
            T(s_ is not None and abs(s_ - want) <= SIGMA_TOL and df_ == wdf,
              "%s = %s over df %d, %d cells (registered %.6f, df %d)"
              % (lbl, "n/a" if s_ is None else "%.6f" % s_, df_, nc_, want, wdf))
        refs = (
            ("ub9-b6 (m6's spec, AUG=0, n=3)", lambda r: r["run"].startswith("ub9-b6")),
            ("i3b-3e4 (m6t's spec, AUG=1, n=3)", lambda r: r["run"].startswith("i3b-") and r["alpha0"] == "3e-4"),
            ("i3b-1e6 (paper alpha0, AUG=1, n=3)", lambda r: r["run"].startswith("i3b-") and r["alpha0"] == "1e-6"),
            ("bl-sgd-01 (augmented denominator baseline, n=5)", lambda r: r["run"].startswith("bl-sgd") and r["alpha0"] == "0.1"),
            ("gate0_adamw_s1 (only unaugmented non-meta row, n=1)", lambda r: r["run"] == "gate0_adamw_s1"))
        for k, pred in refs:
            m_, s_, n_ = cell_mean(rows, pred)
            T(m_ is not None and abs(m_ - REF[k]) <= REF_TOL,
              "%s = %s (registered %.4f)" % (k, "n/a" if m_ is None else "%.4f" % m_, REF[k]))
        T(abs(GAP_AUG - 1.8071) < 1e-4, "GAP_AUG = bl-sgd-01 - i3b-3e4 = %+.4f pp (cross-batch)" % GAP_AUG)
        nseed = sum(1 for r in rows if str(r.get("seed")) in ("50", "51", "52", "53", "54"))
        T(nseed == 0, "ZERO rows outside cau1- carry seed 50-54 (the block pre-assigned to this track)",
          "found %d" % nseed)
        # the AUGMENT=0 census: what the record actually holds
        a0 = [r for r in rows if r.get("augment") == "0"]
        fams = sorted(set(batch_of(r) for r in a0))
        nonmeta0 = [r for r in rows if r.get("meta") == "?" and r.get("augment") in ("0",)]
        T(len(a0) == 27 and fams == ["PP", "pp", "ub9"],
          "the corpus holds %d AUGMENT=0 rows in batches %s (199.10 said only PP-*)" % (len(a0), fams))
        T(all(r.get("base") == "AdamW" and r.get("meta") == "Adam" for r in a0),
          "every one of them is MetaOptimize (AdamW+Adam) -- a granularity contrast, no baseline")
        T(len(nonmeta0) == 0, "ZERO non-meta rows are recorded AUGMENT=0 (gate0_adamw_s1 is augment='?')",
          "found %d" % len(nonmeta0))

        # ---- B: the record-based reasons for the PRIMARY -----------------------------
        print("\nB. WHY THE PRIMARY IS AN ENDPOINT -- re-derived from the record")
        ub = [r for r in rows if r["run"].startswith("ub9-") and usable(r)]
        dif = [abs(float(r["auc"]) - float(r["plateau5"])) for r in ub]
        T(len(ub) == 9 and max(dif) <= 0.70 + 1e-9,
          "B1 at AUGMENT=0 the method's curve IS its endpoint: |AUC - plateau5| <= 0.70 on all %d ub9 runs"
          % len(ub), "max %.3f" % (max(dif) if dif else float("nan")))
        tr = [float(r["final_train"]) for r in ub]
        T(len(tr) == 9 and min(tr) >= 99.8,
          "B3 TRAIN is at its CEILING at AUGMENT=0: ub9 final_train in [%.2f, %.2f]"
          % (min(tr) if tr else 0, max(tr) if tr else 0))
        lad = {}
        for r in rows:
            if r["run"].startswith("bl-sgd") and usable(r):
                lad.setdefault(float(r["alpha0"]), []).append((float(r["plateau5"]), float(r["auc"])))
        ks = sorted(lad)
        p5m = [mean([x[0] for x in lad[k]]) for k in ks]
        aum = [mean([x[1] for x in lad[k]]) for k in ks]
        ip = max(range(len(ks)), key=lambda i: p5m[i]) if ks else -1
        ia = max(range(len(ks)), key=lambda i: aum[i]) if ks else -1
        T(len(ks) == 4 and 0 < ip < len(ks) - 1 and ia == 0
          and all(aum[i] > aum[i + 1] for i in range(len(ks) - 1)),
          "B2 on bl-sgd (AUG=1) AUC falls monotonically with lr (%s) -- argmax at the LOWER EDGE --"
          " while plateau5 peaks INTERIOR at lr=%s" % (" > ".join("%.2f" % x for x in aum),
                                                       ks[ip] if ks else "?"))
        # raw-.out re-derivation of B1, when this host can see ub9's files
        outs = []
        if runsdir and os.path.isdir(runsdir):
            outs = [f for f in os.listdir(runsdir) if re.match(r"^ub9-(sc|b6|ly)-s\d-\d+\.out$", f)]
        if len(outs) == 9:
            e99, e95 = [], []
            for f in outs:
                rec = parse_out(os.path.join(runsdir, f))
                e99.append(rec["ep_train99"])
                te = [rec["epochs"][e][1] for e in range(EPOCHS)]
                e95.append(next(e for e in range(EPOCHS) if te[e] >= 0.95 * rec["plateau5"]))
            T(all(x is not None and x <= 6 for x in e99) and max(e95) <= 2,
              "B1' from the RAW ub9 .out: TRAIN >= 99%% by epoch %s, TEST >= 95%% of plateau5 by epoch %d"
              % (max(e99), max(e95)))
        else:
            S("B1' raw ub9 .out not reachable under %s (%d found) -- CSV reading above stands" % (runsdir, len(outs)))

    # ---- C: the floor gate -----------------------------------------------------------
    print("\nC. THE FLOOR GATE (164.6) AND BOUNDEDNESS")
    se_p = SIGMA_P5_PRIOR * SE_FACTOR
    preds = {"m6": 74.7967, "m6t-low": 72.0, "m6t-high": 80.0, "any-rung-low": PRED_MIN_ANY_ARM}
    lo = min(preds.values())
    T(lo >= PRED_MIN_ANY_ARM and (lo - CHANCE) / se_p >= FLOOR_SE_MIN,
      "lowest level ANY account predicts for ANY arm = %.1f pp = %.1f pp = %.0f SE above chance"
      % (lo, lo - CHANCE, (lo - CHANCE) / se_p))
    T(PRED_MAX_ANY_ARM < CEIL_TEST - 2.0 * Z_BAR * se_p,
      "highest predicted level %.0f pp is %.0f SE under the TEST ceiling" % (PRED_MAX_ANY_ARM, (CEIL_TEST - PRED_MAX_ANY_ARM) / se_p))
    T(abs(SIGMA_P5_PRIOR * SE_FACTOR - 0.364790) < 1e-5, "SE_PRIOR = %.6f pp (3 v 3)" % (SIGMA_P5_PRIOR * SE_FACTOR))
    b_s = E_MAX[7] * SIGMA_P5_PRIOR / math.sqrt(3.0)
    b_m = E_MAX[2] * SIGMA_P5_PRIOR / math.sqrt(3.0)
    T(b_s < Z_BAR * se_p and b_m < Z_BAR * se_p,
      "worst-case selection: S* pushes toward DEFICIT <= %.3f pp, M* toward REVERSAL <= %.3f pp;"
      " both under the 2 SE bar %.3f" % (b_s, b_m, Z_BAR * se_p))

    # ---- D: design constants ---------------------------------------------------------
    print("\nD. DESIGN")
    T(len(ARMS) == 9 and len(arm_names()) == 27 and len(set(arm_names())) == 27,
      "9 arms x 3 seeds = 27 distinct run names")
    lrs = [float(v) for v, _ in LADDER]
    T(lrs == sorted(lrs) and len(lrs) == 7, "ladder ascending, 7 rungs: %s" % lrs)
    T(all(OUT_RE.match(run_name(a, s) + "-123.out").group(1) == a for a in ARMS for s in SEEDS),
      "every run name round-trips through the .out regex (m6 vs m6t disambiguated)")
    T(set(SEEDS) <= set(range(50, 55)), "seeds %s lie in the pre-assigned block 50-54" % (SEEDS,))

    # ---- E: the REAL score() on synthetic run directories ---------------------------
    print("\nE. score() ON SYNTHETIC RUNS (invented numbers; nothing about the real batch)")
    L_int = [80.0, 83.0, 85.0, 86.5, 86.0, 84.0, 81.0]        # interior peak at 0.05
    L_mid = [60.0, 66.0, 68.0, 66.0, 60.0, 50.0, 40.0]         # cosine high-lr phase, AUC peak interior
    cases = [
        ("E1 deficit, interior", dict(ladder_levels=L_int, meta_levels={"m6": 74.8, "m6t": 75.5}),
         0, ["FINAL: DEFICIT-HOLDS |", "LADDER:INTERIOR", "M*=m6t", "TRAIN-AT-CEILING",
             "COUNTERWEIGHT-NOT-TESTED"]),
        ("E2 tie, interior", dict(ladder_levels=[70.0, 72.0, 74.0, 75.0, 74.5, 73.0, 71.0],
                                  meta_levels={"m6": 74.9, "m6t": 74.2}),
         0, ["FINAL: DEFICIT-CLOSES |", "M*=m6"]),
        ("E3 reversal at the paper config", dict(ladder_levels=[68.0, 70.0, 71.5, 72.0, 71.0, 70.0, 69.0],
                                                 meta_levels={"m6": 75.0, "m6t": 74.0}),
         0, ["FINAL: REVERSES-AT-PAPER-CONFIG |", "M*=m6"]),
        ("E4 reversal only at tuned alpha0", dict(ladder_levels=[68.0, 70.0, 71.5, 73.5, 71.0, 70.0, 69.0],
                                                  meta_levels={"m6": 73.4, "m6t": 76.0}),
         0, ["FINAL: REVERSES-AT-TUNED-ALPHA0-ONLY |", "M*=m6t"]),
        ("E5 deficit, ladder edge high", dict(ladder_levels=[78.0, 80.0, 82.0, 84.0, 85.0, 86.0, 87.0],
                                              meta_levels={"m6": 74.8, "m6t": 74.0}),
         0, ["FINAL: DEFICIT-HOLDS | LOWER-BOUND |", "LADDER:EDGE-HIGH"]),
        ("E6 reversal but ladder edge", dict(ladder_levels=[72.0, 71.0, 70.0, 69.0, 68.0, 67.0, 66.0],
                                             meta_levels={"m6": 75.0, "m6t": 74.0}),
         0, ["FINAL: UNRESOLVED-REVERSAL-LADDER-EDGE |", "LADDER:EDGE-LOW"]),
        ("E7 missing run", dict(ladder_levels=L_int, meta_levels={"m6": 74.8, "m6t": 75.5},
                                drop=("m6", 51)),
         1, ["FINAL: INCOMPLETE"]),
        ("E8 AUGMENT=1 on one run", dict(ladder_levels=L_int, meta_levels={"m6": 74.8, "m6t": 75.5},
                                         env_break=(("m6t", 52), {"AUGMENT": "1"})),
         1, ["FINAL: GATED-ENV"]),
        ("E9 repeated flag", dict(ladder_levels=L_int, meta_levels={"m6": 74.8, "m6t": 75.5},
                                  args_break=(("m6", 50), " --alg-meta Lion")),
         1, ["FINAL: GATED-ARGS"]),
        ("E10 method at floor", dict(ladder_levels=L_int, meta_levels={"m6": 10.0, "m6t": 10.0}),
         1, ["FINAL: GATED-METHOD-AT-FLOOR"]),
        ("E11 endpoint deficit, curve reversal", dict(ladder_levels=L_int, ladder_mid=L_mid,
                                                      meta_levels={"m6": 74.8, "m6t": 75.5}),
         0, ["FINAL: DEFICIT-HOLDS |", "CURVE-REVERSES"]),
        ("E12 noisy seeds", dict(ladder_levels=L_int, meta_levels={"m6": 74.8, "m6t": 75.5},
                                 jit=(0.0, 1.5, -1.5)),
         0, ["SIGMA-INBATCH-DOMINATES"]),
        ("E13 truncated resubmit shadows a real run (149.2)",
         dict(ladder_levels=L_int, meta_levels={"m6": 74.8, "m6t": 75.5}, shadow=("m6", 50)),
         0, ["FINAL: DEFICIT-HOLDS |"]),
    ]
    for label, kw, want_rc, want_sub in cases:
        with tempfile.TemporaryDirectory() as d:
            _build(d, **kw)
            res = _run_score(d)
            if res[0] is None:
                T(False, label, res[1])
                continue
            rc, fin, out = res
            ok = rc == want_rc and all(w in fin for w in want_sub)
            T(ok, "%s -> %s (rc %s)" % (label, fin[:150], rc))
    # the curve deficit case is the default synthetic shape; check its token too
    with tempfile.TemporaryDirectory() as d:
        _build(d, ladder_levels=L_int, meta_levels={"m6": 74.8, "m6t": 75.5},
               ladder_mid=[x - 1.0 for x in L_int])
        rc, fin, out = _run_score(d)
        T("CURVE-DEFICIT" in fin, "E14 curve deficit when the baseline curve is never depressed -> %s"
          % [t for t in fin.split(" | ") if t.startswith("CURVE")])

    print("\n%d checks, %d failures, %d skipped" % (passes[0] + len(fails), len(fails), skips[0]))
    if fails:
        print("FAILED: " + " ;; ".join(fails))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runsdir", nargs="?", default=None,
                    help="directory holding the cau1-*.out files (the runner writes"
                         " <WS>/runs/%%x-%%j.out)")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runsdir", dest="runsdir_opt", default=None,
                    help="for --selftest: where ub9's raw .out files live")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a.csv, a.runsdir_opt or a.runsdir)
    if not a.runsdir:
        ap.error("the documented invocation is: %s <runsdir>" % os.path.basename(sys.argv[0]))
    return score(a.runsdir, a.csv)


if __name__ == "__main__":
    sys.exit(main())
