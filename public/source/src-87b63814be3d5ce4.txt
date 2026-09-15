#!/usr/bin/env python3
# =============================================================================
# cdn2_denominator_score.py -- THE CIFAR-100 DENOMINATOR.
# THE SUCCESSOR TO analysis/cdn1_denominator_score.py, WHICH CANNOT RUN.
#
# =============================================================================
# WHY THIS FILE IS cdn2 AND NOT cdn1 -- A DEFECT, DISCLOSED IN FULL
#
# analysis/cdn1_denominator_score.py was registered at commit 1e8e538
# (2026-09-08T08:27:34+02:00, epoch 1788848854), 72 s before the single sbatch
# submission of all 24 `cdn1-*` jobs, with --selftest 34/34 PASS.  Run
# unedited against the ingested corpus (CORRECTIONS 171.3b) it CRASHES
# unconditionally in score():
#
#     File "analysis/cdn1_denominator_score.py", line 524, in cell
#       rs = [have[n] for n in want if _cell_of(n) == PREFIX + tag and n in have]
#     File "analysis/cdn1_denominator_score.py", line 292, in _cell_of
#       return re.sub(r"-s\d+$", "", str(r.get("run") or ""))
#     AttributeError: 'str' object has no attribute 'get'
#
# THE DEFECT.  _cell_of() takes a CSV ROW dict and calls .get("run") on it;
# the list comprehension in cell() hands it the run-name STRING `n`.  The
# crash is corpus-independent (171.3b re-ran it against the pre-ingest CSV and
# it died at the same line), it fires on the FIRST ladder rung, and it fires
# AFTER V0 has printed and BEFORE any arm mean, gate, gap, branch or verdict
# is computed.  cdn1's --selftest is green because --selftest never calls
# score(): the two paths share constants and readers but not one line of the
# tabulation.  171.9 named the lesson: A GREEN SELFTEST IS NOT EVIDENCE THAT A
# SCORER CAN SCORE.  Section J below is this file's answer to that.
#
# ONE CORRECTION TO 171.3b.  It also named "a second latent fault": that the
# same line "indexes have[n] BEFORE the `n in have` guard".  It does not.  A
# list comprehension evaluates its `if` clause before its element expression,
# so have[n] is only ever reached for an n that passed `n in have`.  There was
# ONE fault, not two.  The fix below nonetheless puts `n in have` FIRST in the
# filter chain, because _cell_of(have[n]) -- the row, which is what the
# function was written for -- must not be evaluated for an absent n.
#
# RULE 16 IS HONOURED, NOT WORKED AROUND.  cdn1 is NOT edited; it stays in the
# tree frozen and broken (precedent cN1 -> cN2 at CORRECTIONS 149).  This file
# is a NEW registration whose non-comment diff against cdn1 is:
#   (1) the one-line fix in cell();
#   (2) three import lines (io, contextlib, tempfile) that section J needs;
#   (3) the tool name in the two banner strings;
#   (4) section J of --selftest, which exercises score() on SYNTHETIC corpora.
# `diff cdn1 cdn2` with comment lines stripped is the audit, and it must show
# NO change to any registered constant, threshold, stratum, gate, bar, branch
# boundary, ladder, estimand, SE formula or corpus reader.  PREFIX stays
# "cdn1-": the BATCH is still cdn1; only the scorer is cdn2.
#
# WHAT THIS FILE MAY CLAIM, AND WHAT IT MAY NOT.  This file has NO runs of its
# own -- every `cdn1-*` row it scores was on disk and ingested (corpus 2,662
# rows at 171; 2,740 rows at 174) before this file existed.  It therefore does
# NOT carry the RULE 21 label.  It claims COMMIT-BEFORE-FIRST-EXECUTION only
# (precedent cQ1 at 149, cS1 at 159, cZ1 at 170): this file is committed
# before its score() is run against results/all_runs.csv for the first time,
# and the margin is stated in docs/CORRECTIONS.md in seconds.
#
# DISCLOSURE OF WHAT THE REGISTERING AGENT HAD SEEN.  Everything.  CORRECTIONS
# 171.4-171.6 tabulate every per-seed plateau5 and final_train of all 24 rows
# and the hand-derived GAP_in (+5.6993 pp = +18.80 SE), branch, V2 argmax and
# DELTA_H, and the cycle brief restated those targets.  During design the
# agent also ran cdn1's --selftest on the live 2,740-row corpus (30/34: the D
# df, the two E floor checks and I fail by census drift, as 174.12 recorded)
# and cdn1's default invocation (which printed V0 24/24 and crashed).  What
# this registration buys is therefore NOT that the numbers were unknown; it is
# that the SCORER THAT EMITS THEM is fixed and committed before it emits them,
# so that its verdict is machine output and not a hand derivation, and that a
# discrepancy against 171 -- should one appear -- is a finding and not
# something to adjust away.  Section J was shaken out on synthetic corpora
# before this commit; no real quantity was produced by cdn2 before it.
# =============================================================================
#
#   python3 analysis/cdn2_denominator_score.py --selftest
#   python3 analysis/cdn2_denominator_score.py                 # DEFAULT, scores
#
# THE DEFAULT INVOCATION TAKES NO ARGUMENTS.  Every optional flag defaults to
# this batch's own path (164.2: a missing --manifest published a WRONG verdict
# for an hour).  `--csv` defaults to the repo's results/all_runs.csv resolved
# from THIS FILE's location, so the scorer cannot be pointed at the wrong
# corpus by being run from the wrong directory.  (bin/cdn1_c100_denominator.sh
# imported cdn1, not this file, at submit time; nothing imports cdn2.)
#
# =============================================================================
# WHAT THIS BATCH IS FOR
#
# There is no tuned non-meta baseline anywhere in this corpus outside CIFAR-10.
# On CIFAR-10 the denominator exists and is quoted in the abstract:
# `bl-sgd-01` = 95.124 pp (n=5, sem 0.047) against a best MetaOptimize cell of
# 93.317 (n=3), a deficit of -1.807 pp (119.11).  On CIFAR-100 there is
# NOTHING.  Every CIFAR-100 number in the corpus is a MetaOptimize number
# compared to another MetaOptimize number, so the whole family may sit below
# the plain optimiser it is implicitly measured against and nobody has looked.
#
# This batch looks.  It is deliberately cheap.  Its value is that it exists.
#
# =============================================================================
# THE ARMS  (24 jobs, ONE submission, seeds {0,1,2} throughout)
#
#   A  `cdn1-lr<TAG>-s<S>`   6 rungs x 3 seeds = 18.  --optimizer SGD,
#      CIFAR-100 / ResNet18_c100 / batch 100 / AUGMENT=1 / 100 epochs,
#      momentum 0.9, weight decay 5e-4, cosine-to-zero with a 1000-step warmup
#      (COS_TOTAL=50000 = 500 steps/epoch x 100 epochs, COS_WARMUP=1000).
#      LR ladder {0.01, 0.02, 0.05, 0.1, 0.2, 0.3}.
#      This is the DENOMINATOR.  Its best rung is the comparison quantity.
#
#   M  `cdn1-m-s<S>`         3 jobs.  An IN-BATCH replicate of the corpus's
#      best CIFAR-100 cell -- `gm2-ch`, chunk771, ms=1e-4, alpha0=1e-3,
#      box -15:-2.3026, SGDm base / Lion meta, same 100-epoch cell.  Byte-
#      identical submit line to bin/c91_c100_mech.sh's `gm2-ch` arm except the
#      run name and the save directory.
#      WHY IT IS HERE: BATCH is this campaign's unit of replication
#      (F(62,85)=5.47), so a cross-batch gap is exposed to a batch offset that
#      a within-batch gap is not.  Arm M makes the headline gap WITHIN-BATCH
#      and simultaneously MEASURES this batch's offset against `gm2`.
#
#   C  `cdn1-h2-s<S>`        3 jobs.  Arm A's recipe at lr=0.1 run for 200
#      epochs with the cosine horizon rescaled to match (COS_TOTAL=100000,
#      COS_WARMUP unchanged at 1000 steps).  SECONDARY.  Published
#      ResNet-18/CIFAR-100 numbers are usually quoted at 200 epochs; the
#      100-epoch cell is what makes the comparison like-for-like with this
#      corpus, so BOTH are run and the primary stays at 100.
#      REGISTERED LIMITATION: arm C's lr is fixed a priori at 0.1 (the standard
#      recipe, and the CIFAR-10 ladder's argmax in this same harness) because
#      one submission cannot condition on arm A's argmax.  If arm A's argmax is
#      not 0.1, arm C is a LOWER BOUND on the 200-epoch tuned baseline and is
#      reported as a lower bound, never as a tuned number.
#
# WHAT DIFFERS FROM THE CAMPAIGN'S STANDARD CIFAR-100 CELL, AND WHY
#   MATCHED:   dataset CIFAR100, network ResNet18_c100, batch 100, AUGMENT=1,
#              100 epochs, one GPU, the same harness, the same aggregator,
#              plateau5 as the primary metric, seeds {0,1,2}.
#   DIFFERS:   the optimiser.  That is the entire point of the batch -- it is
#              the one axis being introduced.  Arms A and C carry no meta
#              optimiser at all, therefore no meta-stepsize, no alpha0-as-beta0,
#              no BETA_CLIP box and no step-size groups; `--alpha0` carries the
#              SGD learning rate because that is the flag build_optimizer.py
#              routes to torch.optim.SGD's lr.  Arm M carries all of them and
#              is the anchor that absorbs the difference.
#   DIFFERS:   arm C's horizon (200 epochs).  Declared SECONDARY for exactly
#              that reason.
#
# =============================================================================
# THE PRE-REGISTRATION
#
# PRIMARY QUANTITY
#     GAP_in = mean plateau5 of arm A's BEST rung  -  mean plateau5 of arm M
# both n=3, both in this batch, so every batch offset cancels exactly.
#
# SECONDARY QUANTITY (reported, and the reason arm M exists)
#     GAP_corpus = arm A's best rung  -  the corpus's best CIFAR-100 cell,
# re-derived by section B of --selftest with `cdn1-*` EXCLUDED.
#
# THE NOISE FLOOR, RE-DERIVED AT REGISTRATION TIME AND AGAIN AT SCORE TIME
#     sigma_seed = 0.3713 pp -- the pooled within-cell seed sd of plateau5 over
#     every CIFAR-100, 100-epoch, complete, window_ok cell whose mean is >= 60
#     pp (df 49, 28 cells).  Restricted to the >= 60 regime ON PURPOSE: the
#     unrestricted pool is 0.7624 and is dominated by the collapsed
#     cut-position arms in the 20s, which are not the regime any arm of this
#     batch lives in.  DO NOT copy this literal into a later batch: sigma_w has
#     drifted twice after a registration already (156, 159).  Section D
#     re-derives it and FAILS if it has moved by more than 0.05 pp.
#     For reference and NOT used as the bar: the CIFAR-10 tuned-SGD ladder
#     `bl-sgd-*` has a pooled seed sd of 0.1201 pp (df 16), so treating the
#     meta-arm sigma as the SGD arms' sigma is conservative by ~3x.
#
#     se(one n=3 cell mean)      = 0.3713/sqrt(3)      = 0.2144 pp
#     SE  = se(difference of two n=3 cell means) = 0.3713*sqrt(2/3) = 0.3031 pp
#     Every bar below is quoted in units of SE = 0.3031 pp.
#
# POINT PREDICTIONS (the predictions are not the test; the GAP is the test)
#     arm A best rung   76.0 pp   arm M   72.05 pp   arm C   77.5 pp
#     -> GAP_in predicted +3.95 pp = 13.0 SE.
#
# THE REGISTERED BRANCHES ON GAP_in.  Boundaries at 1.0 pp (3.30 SE) and
# 3.0 pp (9.90 SE); the branches are 3.30 SE apart at the closest, well over
# the 1.5-SE separation this campaign requires of adjacent accounts.
#
#   BELOW-BY-A-LOT      GAP_in >= +3.0
#       MetaOptimize on CIFAR-100 sits materially below a tuned plain
#       SGD+cosine baseline, and by MORE than it does on CIFAR-10 (-1.807).
#       The abstract's scope sentence must carry the CIFAR-100 deficit
#       explicitly and must not generalise the CIFAR-10 figure.
#   CIFAR-10-LIKE       +1.0 <= GAP_in < +3.0
#       The CIFAR-100 deficit is the same order as the CIFAR-10 one; a single
#       scope sentence covers both datasets.
#   NO-RESOLVABLE-DEFICIT   |GAP_in| < 1.0
#       At this cell the family is not resolvably below tuned SGD on
#       CIFAR-100, and the CIFAR-10 deficit does NOT generalise.  Reported as
#       such -- this is the branch that would most help the paper and it is
#       registered with the same bar as the others.
#   ABOVE               GAP_in <= -1.0
#       MetaOptimize beats tuned SGD+cosine at this cell.  A genuinely
#       positive result; report it as one, and re-check arm A's ladder for a
#       tuning failure before believing it.
#
# THE SCOPE REGISTRATION, FIXED NOW SO IT CANNOT BE RE-INTERPRETED LATER
#   A gap of ANY size is a SCOPE fact about where the MetaOptimize FAMILY sits
#   on CIFAR-100 against a tuned plain optimiser.  It is NOT a refutation of
#   any granularity finding in this corpus, and it is not evidence for one
#   either.  Every granularity contrast the campaign owns (D, G, U, T, the
#   count-matched partition audit, the cut-position ladder) is a WITHIN-
#   MetaOptimize, WITHIN-batch difference between two arms that share the base
#   optimiser, the meta optimiser, the horizon and the box.  A common additive
#   offset between the family and SGD cancels EXACTLY out of every one of
#   those differences.  The gap therefore constrains one sentence in the
#   abstract and nothing else.  Symmetrically: a SMALL gap would not
#   strengthen any granularity finding.
#
# GATES, SCORED IN THIS ORDER, BEFORE ANY GAP IS QUOTED
#   V0  COMPLETENESS.  24/24 rows present, every one complete==1 and
#       window_ok==1, epochs_done == epochs_requested.
#   V1  TRAINS-AT-ALL.  every arm's mean plateau5 > 40.0 and collapsed false.
#   V2  BRACKETING.  arm A's argmax over the 6 rungs is INTERIOR (not 0.01 and
#       not 0.3).  IF IT IS AT AN ENDPOINT the baseline is NOT tuned, RULE 11
#       is not satisfied, and GAP_in is reported as a LOWER BOUND on the gap,
#       never as a tuned comparison.
#   V3  OFFSET.  |arm M - corpus gm2-ch (72.054, n=3)| <= 2 SE = 0.606 pp.
#       PASS -> GAP_corpus may be quoted alongside GAP_in.  FAIL -> only
#       GAP_in is quoted, and the offset is reported as the finding it is.
#
# NO ARM IS PREDICTED AT OR NEAR THE FLOOR (the cpr1 lesson, 164)
#   Two floors are registered.  (a) CHANCE on CIFAR-100 = 1.000 pp.  (b) the
#   corpus's own m=1 floor for this network: the unweighted mean of the ten
#   ResNet18_c100 `scalar` 100-epoch cells = 22.727 pp (section E re-derives
#   it with `cdn1-*` excluded).  The lowest level ANY arm is predicted to
#   reach under ANY registered account is arm A's lr=0.01 rung at ~73.0 pp
#   under the central account, and ~69.7 pp (the corpus's best CIFAR-100
#   layerwise cell) under the most adverse account in which the SGD recipe
#   underperforms badly.  That is 46.97 pp = 155.0 SE above the m=1 floor and
#   68.7 pp = 226.6 SE above chance.  The content-free 'the arm just died'
#   model predicts ~1.0 pp and is therefore >= 226 SE from EVERY registered
#   account -- it cannot be mistaken for one, which is exactly what cpr1's
#   headline could not say.  Section F enforces this and FAILS the selftest if
#   any registered prediction comes within 10 SE of either floor.
#
# WHAT THIS BATCH MAY NOT CLAIM
#   * Nothing about ImageNet, TinyImageNet, ResNet34/50, or CIFAR-10.
#   * Nothing about any granularity contrast (see THE SCOPE REGISTRATION).
#   * No claim that arm A is THE optimal plain baseline: it is tuned over one
#     axis (lr) on a 6-point ladder at fixed momentum 0.9 and wd 5e-4.  A
#     wd/momentum sweep is a different batch and is not funded here.
#   * Arm C is not a tuned 200-epoch number unless arm A's argmax is 0.1.
# =============================================================================

from __future__ import print_function

import os
import re
import csv
import sys
import math
import argparse
import collections
import io
import contextlib
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV_DEFAULT = os.path.join(REPO, "results", "all_runs.csv")

PREFIX = "cdn1-"              # THIS batch.  Excluded from every corpus reader.

# ---------------------------------------------------------------------------
# REGISTERED CONSTANTS.  Every one is RE-DERIVED by --selftest from the corpus
# with `cdn1-*` excluded, so each is invariant under this batch's own ingest.
# ---------------------------------------------------------------------------

SIGMA_SEED   = 0.3713         # pp, pooled within-cell seed sd, C100/100ep/>=60
SIGMA_DF     = 49
SE           = SIGMA_SEED * math.sqrt(2.0 / 3.0)     # 0.30313 pp
SE_CELL      = SIGMA_SEED / math.sqrt(3.0)           # 0.21437 pp
SIGMA_DRIFT_BAR = 0.05        # section D fails if sigma_seed moved by more

CORPUS_BEST_NAME = "gm2-ch"
CORPUS_BEST      = 72.054
CORPUS_BEST_N    = 3
CORPUS_BEST_CELL = ("ResNet18_c100", "chunk771", "1e-4", "1e-3",
                    "-15:-2.3026", "100", "1", "SGDm", "Lion")

SCALAR_FLOOR = 22.727         # pp, mean of the 10 ResNet18_c100 scalar cells
CHANCE_FLOOR = 1.000          # pp, 1/100 classes

# branch boundaries on GAP_in, in pp
B_LARGE = 3.0
B_SOME  = 1.0

# gates
V1_TRAINS   = 40.0
V3_OFFSET   = 2.0 * SE        # 0.60627 pp

# point predictions, pp
PRED = {"A_best": 76.0, "M": 72.05, "C": 77.5}
PRED_LADDER = {"0.01": 73.0, "0.02": 74.3, "0.05": 75.5,
               "0.1": 76.0, "0.2": 75.3, "0.3": 74.5}
PRED_ADVERSE_MIN = 69.7       # most adverse credible level for any arm

# the ladder, and the arm names it produces.  The submit script derives its
# own tags from THIS table by importing this module, so the two cannot drift.
LADDER = [("0.01", "001"), ("0.02", "002"), ("0.05", "005"),
          ("0.1", "01"), ("0.2", "02"), ("0.3", "03")]
INTERIOR = set(t for _, t in LADDER[1:-1])
SEEDS = [0, 1, 2]

EPOCHS_A = 100
EPOCHS_C = 200
COS_TOTAL_A = 50000           # 500 steps/epoch (50,000 train / batch 100) x 100
COS_TOTAL_C = 100000
COS_WARMUP = 1000


def arm_names():
    """Every run name this batch submits.  24 of them."""
    out = []
    for _, tag in LADDER:
        for s in SEEDS:
            out.append("%slr%s-s%d" % (PREFIX, tag, s))
    for s in SEEDS:
        out.append("%sm-s%d" % (PREFIX, s))
    for s in SEEDS:
        out.append("%sh2-s%d" % (PREFIX, s))
    return out


# ---------------------------------------------------------------------------
# corpus readers.  EVERY one excludes `cdn1-*`.
# ---------------------------------------------------------------------------

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def csv_rows(path=None, exclude_own=True):
    path = path or CSV_DEFAULT
    out = []
    with open(path, "r") as fh:
        for r in csv.DictReader(fh):
            if exclude_own and str(r.get("run") or "").startswith(PREFIX):
                continue
            out.append(r)
    return out


def own_rows(path=None):
    """This batch's rows.  The ONLY reader that does not exclude the prefix."""
    path = path or CSV_DEFAULT
    with open(path, "r") as fh:
        return [r for r in csv.DictReader(fh)
                if str(r.get("run") or "").startswith(PREFIX)]


def _cell_of(r):
    return re.sub(r"-s\d+$", "", str(r.get("run") or ""))


def _cfg_of(r):
    return (r.get("network"), r.get("granularity"), r.get("meta_stepsize"),
            r.get("alpha0"), r.get("beta_clip"), r.get("batch_size"),
            r.get("augment"), r.get("base"), r.get("meta"))


def c100_cells(rows, epochs="100", min_mean=None):
    """{cell name: [plateau5, ...]} over complete, window_ok CIFAR-100 rows."""
    g = collections.defaultdict(list)
    meta = {}
    for r in rows:
        if r.get("dataset") != "CIFAR100":
            continue
        if r.get("complete") != "1" or r.get("window_ok") != "1":
            continue
        if epochs is not None and r.get("epochs_requested") != epochs:
            continue
        p = _f(r.get("plateau5"))
        if p is None:
            continue
        k = _cell_of(r)
        g[k].append(p)
        meta[k] = _cfg_of(r)
    if min_mean is not None:
        g = dict((k, v) for k, v in g.items() if sum(v) / len(v) >= min_mean)
    return g, meta


def pooled_sigma(cells):
    ss = 0.0
    df = 0
    for v in cells.values():
        if len(v) >= 2:
            m = sum(v) / len(v)
            ss += sum((x - m) ** 2 for x in v)
            df += len(v) - 1
    return (math.sqrt(ss / df) if df else float("nan")), df


def mean_sd(v):
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


# ---------------------------------------------------------------------------
# --selftest.  Every registered premise is re-derived here from the corpus,
# with `cdn1-*` excluded, so it is invariant under this batch's own ingest.
# ---------------------------------------------------------------------------

_N_OK = [0]
_N_BAD = [0]


def chk(cond, msg):
    if cond:
        _N_OK[0] += 1
        print("    ok   %s" % msg)
    else:
        _N_BAD[0] += 1
        print("    FAIL %s" % msg)


def selftest(csvpath=None):
    csvpath = csvpath or CSV_DEFAULT
    print("cdn2_denominator_score.py --selftest")
    print("corpus: %s" % csvpath)

    print("\nA. shape of the batch")
    names = arm_names()
    chk(len(names) == 24, "24 run names composed (%d)" % len(names))
    chk(len(set(names)) == 24, "every run name unique")
    chk(len([n for n in names if "-lr" in n]) == 18, "arm A is 18 runs")
    chk(len([n for n in names if n.endswith(("m-s0", "m-s1", "m-s2"))]) == 3,
        "arm M is 3 runs")
    chk(len([n for n in names if "-h2-" in n]) == 3, "arm C is 3 runs")
    chk(all(n.startswith(PREFIX) for n in names), "every name carries %s" % PREFIX)
    chk(COS_TOTAL_A == 500 * EPOCHS_A,
        "arm A's cosine horizon matches its epoch count (500 steps/epoch)")
    chk(COS_TOTAL_C == 500 * EPOCHS_C,
        "arm C's cosine horizon matches its epoch count")
    chk(INTERIOR == set(["002", "005", "01", "02"]),
        "V2's interior set is the 4 middle rungs")

    print("\nB. the comparison quantity: the corpus's best CIFAR-100 cell,")
    print("   RE-DERIVED with cdn1-* EXCLUDED")
    rows = csv_rows(csvpath)
    cells, meta = c100_cells(rows)
    ranked = sorted(((sum(v) / len(v), len(v), k) for k, v in cells.items()),
                    reverse=True)
    top_m, top_n, top_k = ranked[0]
    for m, n, k in ranked[:4]:
        print("       %-16s mean %7.3f  n=%d" % (k, m, n))
    chk(top_k == CORPUS_BEST_NAME,
        "best CIFAR-100 cell is %s (got %s)" % (CORPUS_BEST_NAME, top_k))
    chk(abs(top_m - CORPUS_BEST) < 5e-3,
        "its mean is %.3f (registered %.3f)" % (top_m, CORPUS_BEST))
    chk(top_n == CORPUS_BEST_N, "its n is %d" % top_n)
    chk(meta[top_k] == CORPUS_BEST_CELL,
        "its cell is the registered one: %s" % (meta[top_k],))

    print("\nC. arm M reproduces that cell's SUBMIT LINE, not just its name")
    chk(CORPUS_BEST_CELL[1] == "chunk771", "granularity chunk771")
    chk(CORPUS_BEST_CELL[2] == "1e-4" and CORPUS_BEST_CELL[3] == "1e-3",
        "ms=1e-4, alpha0=1e-3")
    chk(CORPUS_BEST_CELL[4] == "-15:-2.3026", "box -15:-2.3026")
    chk(CORPUS_BEST_CELL[7] == "SGDm" and CORPUS_BEST_CELL[8] == "Lion",
        "SGDm base / Lion meta")

    print("\nD. the noise floor, RE-DERIVED with cdn1-* EXCLUDED")
    hi, _ = c100_cells(rows, min_mean=60.0)
    sig, df = pooled_sigma(hi)
    print("       sigma_seed = %.4f pp over df=%d, %d cells (registered %.4f, df %d)"
          % (sig, df, len(hi), SIGMA_SEED, SIGMA_DF))
    chk(abs(sig - SIGMA_SEED) <= SIGMA_DRIFT_BAR,
        "sigma_seed has not drifted by more than %.2f pp" % SIGMA_DRIFT_BAR)
    chk(df == SIGMA_DF, "df is %d (registered %d)" % (df, SIGMA_DF))
    allc, _ = c100_cells(rows)
    sig_all, df_all = pooled_sigma(allc)
    print("       (unrestricted pool, for contrast: %.4f over df=%d -- NOT the bar)"
          % (sig_all, df_all))
    chk(sig_all > sig, "the unrestricted pool is larger, as documented")
    chk(abs(SE - SIGMA_SEED * math.sqrt(2.0 / 3.0)) < 1e-12,
        "SE = sigma*sqrt(2/3) = %.5f pp" % SE)

    print("\nE. the floors, RE-DERIVED with cdn1-* EXCLUDED")
    scal = {}
    for r in rows:
        if (r.get("dataset") == "CIFAR100" and r.get("granularity") == "scalar"
                and r.get("network") == "ResNet18_c100"
                and r.get("complete") == "1"
                and r.get("epochs_requested") == "100"):
            p = _f(r.get("plateau5"))
            if p is not None:
                scal.setdefault(_cell_of(r), []).append(p)
    cellmeans = [sum(v) / len(v) for v in scal.values()]
    floor = sum(cellmeans) / len(cellmeans)
    print("       m=1 floor = %.3f pp over %d ResNet18_c100 scalar cells "
          "(registered %.3f)" % (floor, len(cellmeans), SCALAR_FLOOR))
    chk(len(cellmeans) == 10, "10 scalar cells (%d)" % len(cellmeans))
    chk(abs(floor - SCALAR_FLOOR) < 5e-3, "the m=1 floor is the registered one")

    print("\nF. NO ARM IS PREDICTED AT OR NEAR EITHER FLOOR")
    worst = min(list(PRED_LADDER.values()) + list(PRED.values()) + [PRED_ADVERSE_MIN])
    for lab, fl in (("m=1 scalar", SCALAR_FLOOR), ("chance", CHANCE_FLOOR)):
        marg = worst - fl
        print("       worst registered level %.2f pp is %+.2f pp = %.1f SE above "
              "the %s floor" % (worst, marg, marg / SE, lab))
        chk(marg / SE >= 10.0, "clears the %s floor by >= 10 SE" % lab)
    dead = CHANCE_FLOOR
    seps = [abs(p - dead) / SE for p in list(PRED.values()) + [PRED_ADVERSE_MIN]]
    chk(min(seps) >= 10.0,
        "the content-free 'arm just died' model (%.1f pp) is >= 10 SE from "
        "every registered account (min %.1f SE)" % (dead, min(seps)))

    print("\nG. the branches are separable at this n")
    chk(B_LARGE > B_SOME, "the two boundaries are ordered")
    chk((B_LARGE - B_SOME) / SE >= 1.5,
        "adjacent branch boundaries are %.2f SE apart (>= 1.5)"
        % ((B_LARGE - B_SOME) / SE))
    chk(B_SOME / SE >= 1.5,
        "the null band's edge is %.2f SE from zero" % (B_SOME / SE))
    gap_pred = PRED["A_best"] - PRED["M"]
    print("       predicted GAP_in = %.3f pp = %.1f SE" % (gap_pred, gap_pred / SE))
    chk(gap_pred >= B_LARGE, "the central prediction lands in BELOW-BY-A-LOT")

    print("\nH. the CIFAR-10 precedent this batch is the CIFAR-100 twin of")
    bl = {}
    for r in rows:
        if str(r.get("run") or "").startswith("bl-sgd-"):
            p = _f(r.get("plateau5"))
            if p is not None:
                bl.setdefault(_cell_of(r), []).append(p)
    chk(len(bl) == 4, "the CIFAR-10 tuned-SGD ladder has 4 rungs (%d)" % len(bl))
    best10 = max((sum(v) / len(v), k, len(v)) for k, v in bl.items())
    print("       CIFAR-10 denominator: %s = %.3f pp (n=%d)"
          % (best10[1], best10[0], best10[2]))
    chk(best10[1] == "bl-sgd-01", "its argmax is lr=0.1, INTERIOR to that ladder")
    sig10, df10 = pooled_sigma(bl)
    print("       CIFAR-10 tuned-SGD pooled seed sd = %.4f pp (df %d) -- "
          "%.1fx tighter than the registered bar" % (sig10, df10, SIGMA_SEED / sig10))
    chk(sig10 < SIGMA_SEED, "using the meta-arm sigma for the SGD arms is conservative")

    print("\nI. this batch is absent from the corpus at registration time")
    chk(len(own_rows(csvpath)) == 0,
        "no cdn1-* row exists yet (%d found)" % len(own_rows(csvpath)))

    print("\nJ. score() ACTUALLY RUNS -- exercised on SYNTHETIC corpora, never the")
    print("   real one (171.9: a green selftest is not evidence a scorer can score)")
    _selftest_score_path()

    print("\n%d/%d checks passed" % (_N_OK[0], _N_OK[0] + _N_BAD[0]))
    return 0 if _N_BAD[0] == 0 else 1


# ---------------------------------------------------------------------------
# section J helpers.  Synthetic data computes NOTHING about the real corpus.
# Every value below is invented; the only things asserted are that score()
# returns instead of raising, and that what it prints is what the REGISTERED
# rules say it must print for the invented inputs.
# ---------------------------------------------------------------------------

_SYN_COLS = ["run", "job_id", "account", "network", "dataset", "batch_size",
             "granularity", "base", "meta", "meta_stepsize", "alpha0", "gamma",
             "augment", "beta_clip", "hier", "lam", "eta_ratio", "seed",
             "epochs_done", "epochs_requested", "best_test", "final_test",
             "final_train", "collapsed", "node", "wallclock_min", "provenance",
             "dup_group", "superseded", "ep_to_85", "ep_to_88", "ep_to_90",
             "plateau", "ep_in_band_90", "plateau5", "auc", "window_ok",
             "complete"]


def _syn_ladder(peak, shape=None):
    """Cell means per rung tag.  Default shape peaks at lr=0.1 (interior)."""
    shape = shape or {"001": -3.0, "002": -2.0, "005": -1.0, "01": 0.0,
                      "02": -0.5, "03": -1.5}
    return dict((t, peak + d) for t, d in shape.items())


def _syn_corpus(path, ladder, m_mean, c_mean, drop=(), override=None):
    """Write a corpus holding ONLY this batch's 24 rows (score() reads nothing
    else).  Each cell's three seeds are mean + {-0.1, 0, +0.1}, so the cell
    mean is the requested value to floating-point rounding."""
    override = override or {}
    rows = []
    for n in arm_names():
        if n in drop:
            continue
        tag = re.sub(r"^%s" % re.escape(PREFIX), "", _cell_of({"run": n}))
        s = int(n.rsplit("-s", 1)[1])
        if tag.startswith("lr"):
            mean, ep = ladder[tag[2:]], str(EPOCHS_A)
        elif tag == "m":
            mean, ep = m_mean, str(EPOCHS_A)
        else:
            mean, ep = c_mean, str(EPOCHS_C)
        r = dict((c, "") for c in _SYN_COLS)
        r.update({"run": n, "job_id": "0", "dataset": "CIFAR100", "seed": str(s),
                  "epochs_done": ep, "epochs_requested": ep, "complete": "1",
                  "window_ok": "1", "collapsed": "0", "final_train": "99.0",
                  "plateau5": "%.3f" % (mean + 0.1 * (s - 1))})
        r.update(override.get(n, {}))
        rows.append(r)
    with open(path, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=_SYN_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _run_score(path):
    """(rc, stdout) of score() on `path`; an exception becomes rc='RAISED'."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = score(path)
    except Exception as e:   # the whole point: a crash must be a FAIL line
        return "RAISED %s: %s" % (type(e).__name__, e), buf.getvalue()
    return rc, buf.getvalue()


def _grab(text, pattern):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def _selftest_score_path():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "syn.csv")

        # J1  nominal: interior argmax, M near the registered corpus best,
        #     every gate PASS, branch BELOW-BY-A-LOT, DELTA_H resolved.
        _syn_corpus(p, _syn_ladder(76.3), 72.1, 78.2)
        rc, out = _run_score(p)
        chk(rc == 0, "J1 nominal synthetic corpus: score() returns 0 (got %r)" % (rc,))
        chk("V0: PASS" in out and "V1: PASS" in out and "V2: PASS" in out
            and "V3: PASS" in out, "J1 all four gates PASS")
        chk("argmax lr=0.1 at 76.300 pp -- INTERIOR" in out,
            "J1 V2 finds the interior argmax at the invented peak")
        g = _grab(out, r"GAP_in\s+([+-]\d+\.\d+) pp")
        chk(g is not None and abs(float(g) - (76.3 - 72.1)) < 5e-4,
            "J1 GAP_in = invented A_best - invented M = +4.200 (printed %s)" % g)
        gse = _grab(out, r"GAP_in\s+[+-]\d+\.\d+ pp\s+= ([+-]\d+\.\d+) SE")
        chk(gse is not None and abs(float(gse) - 4.2 / SE) < 0.01,
            "J1 GAP_in in SE uses the registered SE (%.4f): printed %s" % (SE, gse))
        chk("BRANCH: BELOW-BY-A-LOT" in out, "J1 branch token BELOW-BY-A-LOT")
        gc = _grab(out, r"GAP_corpus\s+([+-]\d+\.\d+) pp")
        chk(gc is not None and abs(float(gc) - (76.3 - CORPUS_BEST)) < 5e-4,
            "J1 GAP_corpus against the registered corpus best (printed %s)" % gc)
        chk("(quotable, V3 passed)" in out, "J1 GAP_corpus marked quotable under V3 PASS")
        d = _grab(out, r"arm A\(lr=0\.1\)\s+\d+\.\d+\s+=\s+([+-]\d+\.\d+) pp")
        chk(d is not None and abs(float(d) - (78.2 - 76.3)) < 5e-4,
            "J1 DELTA_H = invented C - invented A(lr=0.1) = +1.900 (printed %s)" % d)
        chk("RESOLVED: the 100-epoch cell understates" in out,
            "J1 DELTA_H >= 2 SE is reported RESOLVED")
        chk("NOTE: arm A's argmax" not in out,
            "J1 no lower-bound note on arm C when the argmax is 0.1")
        chk("VERDICT: USABLE" in out, "J1 VERDICT: USABLE")
        chk("TEST plateau5" in out and out.count("\n    lr=") == 6,
            "J1 the arm table prints all six ladder rungs")

        # J2  every branch token, driven by the invented peak alone
        for peak, want_br in ((74.1, "CIFAR-10-LIKE"),
                              (72.1, "NO-RESOLVABLE-DEFICIT"),
                              (70.1, "ABOVE")):
            _syn_corpus(p, _syn_ladder(peak), 72.1, peak + 1.0)
            rc, out = _run_score(p)
            chk(rc == 0 and ("BRANCH: %s" % want_br) in out,
                "J2 peak %.1f vs M 72.1 -> %s (rc %r)" % (peak, want_br, rc))

        # J3  argmax at a ladder ENDPOINT: V2 FAIL, lower-bound wording twice
        mono = {"001": -5.0, "002": -4.0, "005": -3.0, "01": -2.0, "02": -1.0, "03": 0.0}
        _syn_corpus(p, _syn_ladder(77.0, mono), 72.1, 78.0)
        rc, out = _run_score(p)
        chk(rc == 0 and "AT AN ENDPOINT" in out and "V2: FAIL" in out,
            "J3 endpoint argmax -> V2 FAIL (rc %r)" % (rc,))
        chk("this is a LOWER BOUND, not a tuned gap" in out,
            "J3 GAP_in carries the lower-bound caveat")
        chk("NOTE: arm A's argmax is lr=0.3, not 0.1" in out,
            "J3 arm C carries the lower-bound note")
        chk("VERDICT: USABLE" in out,
            "J3 V2 does not gate the verdict (registered rule is V0 and V1 and V3)")

        # J4  V0: a missing row and an incomplete row
        _syn_corpus(p, _syn_ladder(76.3), 72.1, 78.2, drop=("cdn1-h2-s2",),
                    override={"cdn1-lr01-s0": {"complete": "0"}})
        rc, out = _run_score(p)
        chk(rc == 0 and "23/24 rows present" in out and "MISSING: cdn1-h2-s2" in out,
            "J4 a dropped row is named under V0 (rc %r)" % (rc,))
        chk("NOT COMPLETE: cdn1-lr01-s0 complete=0" in out and "V0: FAIL" in out,
            "J4 an incomplete row is named and V0 FAILs")
        chk("VERDICT: GATED" in out, "J4 VERDICT: GATED under V0 FAIL")

        # J5  V1: a dead arm below the trains-at-all bar
        _syn_corpus(p, _syn_ladder(76.3), 30.0, 78.2)
        rc, out = _run_score(p)
        chk(rc == 0 and "m: 30.000 <= 40.0" in out and "V1: FAIL" in out,
            "J5 an arm at 30 pp FAILs V1 by name (rc %r)" % (rc,))
        chk("V3: FAIL" in out and "(V3 FAILED -- do not quote)" in out,
            "J5 the same arm FAILs V3 and GAP_corpus is marked unquotable")
        chk("VERDICT: GATED" in out, "J5 VERDICT: GATED")

        # J6  the pre-ingest state: no cdn1-* row at all
        with open(p, "w") as fh:
            csv.DictWriter(fh, fieldnames=_SYN_COLS).writeheader()
        rc, out = _run_score(p)
        chk(rc == 1 and "NOT COMPUTABLE -- arms missing" in out,
            "J6 an empty corpus returns 1 with NOT COMPUTABLE, no crash (rc %r)" % (rc,))

        # J7  the row->name confusion that killed cdn1 cannot recur silently:
        #     _cell_of on a run-name string must raise, and cell() must not
        #     be calling it that way (J1-J6 would have been RAISED otherwise).
        try:
            _cell_of("cdn1-lr01-s0")
            raised = False
        except AttributeError:
            raised = True
        chk(raised, "J7 _cell_of(str) still raises -- the contract is a ROW dict")
        chk(_cell_of({"run": "cdn1-lr01-s0"}) == "cdn1-lr01",
            "J7 _cell_of(row) strips the seed suffix")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(csvpath=None):
    csvpath = csvpath or CSV_DEFAULT
    print("=" * 72)
    print("cdn2 -- THE CIFAR-100 DENOMINATOR (scoring batch cdn1)")
    print("corpus: %s" % csvpath)
    print("=" * 72)

    mine = own_rows(csvpath)
    want = arm_names()
    have = dict((str(r.get("run")), r) for r in mine)

    print("\nV0  COMPLETENESS")
    missing = [n for n in want if n not in have]
    print("    %d/%d rows present" % (len(want) - len(missing), len(want)))
    if missing:
        print("    MISSING: %s" % ", ".join(missing))
    bad = []
    for n in want:
        r = have.get(n)
        if r is None:
            continue
        if r.get("complete") != "1" or r.get("window_ok") != "1":
            bad.append("%s complete=%s window_ok=%s"
                       % (n, r.get("complete"), r.get("window_ok")))
        elif r.get("epochs_done") != r.get("epochs_requested"):
            bad.append("%s %s/%s epochs"
                       % (n, r.get("epochs_done"), r.get("epochs_requested")))
    for b in bad:
        print("    NOT COMPLETE: %s" % b)
    v0 = (not missing) and (not bad)
    print("    V0: %s" % ("PASS" if v0 else "FAIL -- no gap may be quoted"))

    def cell(tag):
        rs = [have[n] for n in want if n in have and _cell_of(have[n]) == PREFIX + tag]
        te = [_f(r.get("plateau5")) for r in rs]
        tr = [_f(r.get("final_train")) for r in rs]
        te = [x for x in te if x is not None]
        tr = [x for x in tr if x is not None]
        col = [str(r.get("collapsed")) for r in rs]
        return te, tr, col

    print("\n    ARM      n   TEST plateau5      TRAIN final     collapsed")
    table = {}
    for lr, tag in LADDER:
        te, tr, col = cell("lr" + tag)
        table["lr" + tag] = te
        if te:
            m, s = mean_sd(te)
            mt = (sum(tr) / len(tr)) if tr else float("nan")
            print("    lr=%-5s  %d   %7.3f  sd %.3f   %7.3f        %s"
                  % (lr, len(te), m, s, mt, ",".join(col)))
    for tag, lab in (("m", "META (chunk771)"), ("h2", "SGD 200ep lr=0.1")):
        te, tr, col = cell(tag)
        table[tag] = te
        if te:
            m, s = mean_sd(te)
            mt = (sum(tr) / len(tr)) if tr else float("nan")
            print("    %-8s %d   %7.3f  sd %.3f   %7.3f        %s"
                  % (lab[:8], len(te), m, s, mt, ",".join(col)))

    print("\nV1  TRAINS-AT-ALL (mean plateau5 > %.1f, collapsed false)" % V1_TRAINS)
    v1 = True
    for k, v in table.items():
        if not v:
            v1 = False
            print("    %s: NO DATA" % k)
        elif sum(v) / len(v) <= V1_TRAINS:
            v1 = False
            print("    %s: %.3f <= %.1f" % (k, sum(v) / len(v), V1_TRAINS))
    print("    V1: %s" % ("PASS" if v1 else "FAIL"))

    print("\nV2  BRACKETING (arm A's argmax must be interior)")
    lad = [(sum(table["lr" + t]) / len(table["lr" + t]), t, lr)
           for lr, t in LADDER if table.get("lr" + t)]
    v2 = False
    best_lr = None
    if lad:
        bm, bt, blr = max(lad)
        best_lr = blr
        v2 = bt in INTERIOR
        print("    argmax lr=%s at %.3f pp -- %s"
              % (blr, bm, "INTERIOR, the ladder brackets" if v2
                 else "AT AN ENDPOINT: the baseline is NOT tuned"))
    print("    V2: %s" % ("PASS" if v2
                          else "FAIL -- every gap below is a LOWER BOUND"))

    print("\nV3  OFFSET (|arm M - %s| <= 2 SE = %.3f pp)"
          % (CORPUS_BEST_NAME, V3_OFFSET))
    v3 = False
    if table.get("m"):
        mm = sum(table["m"]) / len(table["m"])
        off = mm - CORPUS_BEST
        v3 = abs(off) <= V3_OFFSET
        print("    arm M %.3f  -  %s %.3f  =  %+.3f pp (%.2f SE)  %s"
              % (mm, CORPUS_BEST_NAME, CORPUS_BEST, off, off / SE,
                 "PASS" if v3 else "FAIL"))
    print("    V3: %s" % ("PASS" if v3 else
                          "FAIL -- quote GAP_in only, and report the offset"))

    print("\n" + "=" * 72)
    print("THE PRIMARY: GAP_in = arm A best rung - arm M, both n=3, WITHIN batch")
    print("=" * 72)
    if not (lad and table.get("m")):
        print("    NOT COMPUTABLE -- arms missing.")
        return 1
    a_best = max(lad)[0]
    m_mean = sum(table["m"]) / len(table["m"])
    gap = a_best - m_mean
    print("    arm A best (lr=%s)  %7.3f pp   n=%d" % (best_lr, a_best, 3))
    print("    arm M               %7.3f pp   n=%d" % (m_mean, 3))
    print("    GAP_in            %+8.3f pp   = %+.2f SE   (SE = %.4f pp)"
          % (gap, gap / SE, SE))
    print("    predicted         %+8.3f pp" % (PRED["A_best"] - PRED["M"]))
    if gap >= B_LARGE:
        br = "BELOW-BY-A-LOT"
    elif gap >= B_SOME:
        br = "CIFAR-10-LIKE"
    elif gap > -B_SOME:
        br = "NO-RESOLVABLE-DEFICIT"
    else:
        br = "ABOVE"
    print("    BRANCH: %s" % br)
    if not v2:
        print("    ...but V2 FAILED, so this is a LOWER BOUND, not a tuned gap.")

    print("\n    SECONDARY: GAP_corpus = arm A best - %s (%.3f, n=%d), cross-batch"
          % (CORPUS_BEST_NAME, CORPUS_BEST, CORPUS_BEST_N))
    gc = a_best - CORPUS_BEST
    print("    GAP_corpus        %+8.3f pp   = %+.2f SE   %s"
          % (gc, gc / SE, "(quotable, V3 passed)" if v3 else "(V3 FAILED -- do not quote)"))

    print("\n    SECONDARY: DELTA_H = arm C - arm A at lr=0.1 (horizon, 200 vs 100 ep)")
    if table.get("h2") and table.get("lr01"):
        c = sum(table["h2"]) / len(table["h2"])
        a01 = sum(table["lr01"]) / len(table["lr01"])
        d = c - a01
        print("    arm C %7.3f  -  arm A(lr=0.1) %7.3f  =  %+.3f pp (%.2f SE)"
              % (c, a01, d, d / SE))
        print("    %s" % ("RESOLVED: the 100-epoch cell understates the plain-SGD "
                          "ceiling; quote both horizons."
                          if d >= 2 * SE else
                          "NOT RESOLVED at 2 SE: the 100-epoch cell is adequate."))
        if best_lr != "0.1":
            print("    NOTE: arm A's argmax is lr=%s, not 0.1, so arm C is a LOWER "
                  "BOUND on the tuned 200-epoch baseline." % best_lr)

    print("\n    SCOPE.  This gap is a fact about where the MetaOptimize FAMILY")
    print("    sits on CIFAR-100 against a tuned plain optimiser.  It refutes NO")
    print("    granularity finding: every granularity contrast in this corpus is")
    print("    a within-MetaOptimize, within-batch difference, and a common")
    print("    additive offset cancels out of every one of them.")
    print("\n    VERDICT: %s" % ("USABLE" if (v0 and v1 and v3) else
                                 "GATED -- see V0/V1/V3 above"))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(
        description="Score batch cdn1: the CIFAR-100 non-meta denominator.")
    ap.add_argument("--csv", default=CSV_DEFAULT,
                    help="corpus CSV (default: the repo's results/all_runs.csv)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(a.csv)
    return score(a.csv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
