#!/usr/bin/env python3
# =============================================================================
# cO1_cts3_score.py -- THE REGISTERED SCORER FOR BATCH `cts3`.
#
# COMMITTED BEFORE ANY cts3 RUN EXISTS (STANDING RULE 21).  Run it UNEDITED
# (RULE 16).  Documented command-line arguments are not edits; nothing below
# this line may be changed once a cts3 run exists on disk.  If this file turns
# out to be broken, it is FROZEN and a NEW file is registered (precedent
# cN1/cN2, CORRECTIONS 149).
#
# -----------------------------------------------------------------------------
# WHY THIS BATCH EXISTS -- THE STANDING RISK AT CORRECTIONS 150
# -----------------------------------------------------------------------------
# cts1 (CORRECTIONS 147) measured, IN BATCH, on plateau5 at 100 epochs:
#     k49 = [49,13]  55.168667 pp    k50 = [50,12]  30.334667 pp
#     cliff = 24.834000 pp
# and cts2 (CORRECTIONS 150) reproduced it under both clamp levels
# (D_C 25.1627, D_R 24.7673) and showed it does not depend on the BETA_CLIP
# floor.  Both batches are 100 epochs.  CORRECTIONS 150 then recorded, from
# cts2's OWN epoch lines, terminal OLS slopes of test accuracy over epochs
# 80-99 (pp/epoch, mean of 3 seeds):
#
#     k49-C  +0.00426     k50-C  +0.04847
#     k49-R  -0.00310     k50-R  +0.04758
#
# THE WINNING ARM HAS CONVERGED AND THE LOSING ARM HAS NOT.  plateau5 at the
# k50 arms is a snapshot of a still-rising curve, so 24.83 / 25.16 pp is a
# LOWER BOUND on a converged cliff -- or, read the other way, the measured
# cliff could be an artefact of stopping the clock at 100 epochs.  No
# ResNet18_c100 run anywhere in the corpus exceeds 100 epochs (verified by
# --selftest against results/all_runs.csv), so this cannot be settled from
# existing data.  hz3 already cost the programme a finding to exactly this
# failure.  cts3 buys the horizon and nothing else.
#
# -----------------------------------------------------------------------------
# HOW THE HORIZON WAS CHOSEN -- FROM THE MEASURED SLOPES, NOT ROUNDED
# -----------------------------------------------------------------------------
# ASSUMPTION, STATED: the k50 arm's terminal 20-epoch OLS slope decays as a
# power law in the epoch index, s(t) = A * t^-p.  This is the weakest model
# that fits the six rolling windows cts2 actually produced, and it is fitted
# only to the five windows whose midpoint is at or beyond 49.5 -- the earlier
# windows are still in the fast phase and are excluded by registration.
#
# MEASURED on the twelve cts2 .out files (k50-C, mean over seeds 0,1,2), OLS
# slope of test accuracy on epoch index, 20-epoch windows every 10 epochs:
#     mid 39.5 +0.13824   mid 49.5 +0.09706   mid 59.5 +0.08062
#     mid 69.5 +0.05999   mid 79.5 +0.06083   mid 89.5 +0.04847
# Log-log OLS on the five windows with mid >= 49.5 gives
#     A = 8.1380   p = 1.1362
# CONVERGENCE CRITERION, registered: the k50 arm is "converged" when its own
# terminal 20-epoch slope has fallen to the slope at which the k49 arm was
# declared converged by CORRECTIONS 150, namely s* = +0.00426 pp/epoch.  This
# is a data-anchored bar -- the winning arm's own measured convergence -- not
# a chosen tolerance.  Solving A * t^-p = s* gives
#     E = (A / s*)^(1/p) = (8.1380 / 0.00426)^(1/1.1362) = 772 epochs.
# SENSITIVITY, WRITTEN DOWN IN FULL SO IT CANNOT BE DISCOVERED LATER.  The fit
# is not stable to the choice of window subset, and pretending otherwise would
# be the whole failure mode this batch exists to fix.  Re-derived by
# --selftest over six defensible subsets of the SAME measured windows:
#     all six windows                A 12.4603  p 1.2353  E  640
#     mid >= 49.5  (REGISTERED)      A  8.1380  p 1.1362  E  772
#     mid >= 59.5                    A  7.4831  p 1.1170  E  803
#     mid >= 69.5                    A  2.0389  p 0.8220  E 1821
#     4 non-overlapping windows      A 40.5360  p 1.5199  E  414
#     last 3 non-overlapping         A  9.6968  p 1.1860  E  677
# and the target itself moves E from 671 (s* = 0.005) to 772 (s* = 0.00426).
# So E is bracketed by [414, 1821], five of the six variants fall in
# [414, 803], and the outlier is the three-point tail fit whose p = 0.8220 is
# BELOW 1 -- a regime in which the accumulated gain diverges and no finite
# convergence epoch exists at all.  THE REGISTERED E = 772 IS THEREFORE NOT
# CLAIMED TO REACH CONVERGENCE.  Whether it did is settled EMPIRICALLY by gate
# R4 below, on the runs' own terminal slope, and R4 has headroom under every
# one of the six variants (the largest predicted terminal slope at E across
# all of them is 0.008724, against an R4 bar of 0.024235).  772 is 7.72x the
# horizon every existing ResNet18_c100 run has had.
#
# HONEST LIMIT, registered in advance: the ASYMPTOTE is NOT identifiable from
# 60 epochs of curve.  A direct saturating fit y = yinf - b*t^-c to k50-C over
# epochs 40-99 returns yinf = 53.05, and the same fit to k50-R returns
# yinf = 72.06 -- above the k49 arm's own asymptote, which is impossible.  So
# cts3 MAY NOT claim anything about the cliff at infinite budget.  It answers
# one question: what the cliff is at 772 epochs, and how it moved from 100.
#
# -----------------------------------------------------------------------------
# THE DESIGN.  2 cuts x 3 seeds = 6 jobs, ONE submission, 772 epochs.
#
#   factor A  cut  k49 = [49,13]  |  k50 = [50,12]
#   clamp     C = BETA_CLIP -15:-2.3026, the CAMPAIGN STANDARD, held fixed.
#
# WHY ONE CLAMP AND WHY THE CLAMPED ONE.  cts2 measured the clamp x cut
# interaction at 0.3953 pp = 0.38 SE_INT: the clamp axis is inert for this
# cliff.  Carrying it would double the cost to answer a question already
# answered.  The level kept is the CAMPAIGN STANDARD, which is the regime
# CORRECTIONS 147's cliff is stated in and the regime the whole cts1/cpk1
# corpus sits in.  It is also the only arithmetically safe level at this
# horizon: 772 epochs x 500 iterations = 386000 meta-steps, and |dbeta| <= ms
# = 1e-3 per step, so the worst-case reachable beta from the init -13.815511
# is -399.815511.  cts2's RELEASED floor of -80 was justified as unreachable
# at 50000 meta-steps and IS REACHABLE at 386000, so the released arm would
# clamp anyway at this horizon; and a floor set low enough to stay clear is
# useless, because alpha = exp(beta) is float32 and underflows to EXACTLY ZERO
# for every beta below ln(1.401298e-45) = -103.278930.  There is no clamp-free
# arm available at this horizon -- either the clamp binds or the stepsize
# becomes exactly 0.  cts3 therefore claims NOTHING about the released floor
# at long budget, and that gap is stated, not hidden.
#
# THE 100-EPOCH CONTROL IS IN BATCH, IN THE SAME RUNS, AT ZERO COST.
# train.py uses args.num_epochs in exactly two places: the argparse
# declaration (line 42) and the loop bound `for epoch in range(...)` (line
# 135).  There is no schedule, no total-step count, no warmup and no
# num_epochs-dependent quantity anywhere else; SCHED=none, and --max-time
# 999:00:00 keeps train.py's own time-based break from ever firing.  The
# epoch-99 readout of a 772-epoch run is therefore a bona fide 100-epoch run
# of an identical configuration.  So:
#   * D100 -- the cliff at 100 epochs -- is measured WITHIN THIS BATCH, from
#     epochs 95-99 of these same six runs.  No cross-batch splice is needed
#     for the premise gate, and none is permitted.
#   * The horizon contrast D(E) - D100 is PAIRED WITHIN RUN, so run-to-run and
#     seed variance drop out of it entirely.  The SE used below is
#     nevertheless the UNPAIRED one; the paired SE can only be smaller, so
#     every bar quoted here is conservative.
# WHAT THIS IS NOT: the epoch-99 readout is a fresh draw, not a bit-replicate
# of cts2's own C arms -- different job, possibly different GPU, cuDNN
# nondeterminism.  And cts3 shares seeds {0,1,2} and every non-horizon flag
# with cts1/cts2/cpk1, so it is a FOURTH CORRELATED BATCH.  "Reproduces in an
# independent batch" may not be written.
#
# NO ANCHORS, NO CAPTURE.  The estimand is a difference of two arm means in
# percentage points.  This file computes no CAPTURE and forbids one being
# computed from cts3 afterwards.
#
# -----------------------------------------------------------------------------
# THE PRE-REGISTRATION.  Gates fire IN ORDER; the first failure decides.
# -----------------------------------------------------------------------------
# G0 PROVENANCE.  6 runs, 2 cells x 3 seeds, 6 distinct job ids, E/E epochs
#    each, no repeated flag on any ARGS line, and every run's NAME agreeing
#    with its own ARGS and ENV.  BETA_CLIP is audited from the ENV line
#    because it cannot ride the ARGS line.
# R1 PREMISE, IN BATCH.  D100 = M(k49, ep95-99) - M(k50, ep95-99) must be
#    >= 12.4170 pp, half of cts1's own 24.834000 pp cliff.  Otherwise
#    UNRESOLVED-PREMISE, and cts1's or cts2's arms may NOT be spliced in to
#    rescue it.
# R2 DIVERGENCE.  Any run with fewer than E epoch lines, a non-finite metric,
#    or plateau5 at E <= 5.00 pp is DEAD -> UNRESOLVED-DIVERGED.  Registered
#    in advance precisely so that "the long runs fell over" can NEVER be
#    scored as "the cliff collapsed".
# R3 NOISE.  Any cell SD at E above 3 * SIGMA_W = 2.7333 pp ->
#    UNRESOLVED-NOISY.  SIGMA_W is re-derived from the corpus by --selftest,
#    from 100-epoch runs, which are the only ones that exist; R3 is exactly
#    the check that the long-horizon spread has not outgrown it.
# R4 HORIZON EFFECTIVENESS.  The k50 arm's terminal 20-epoch OLS slope at E
#    must be at most HALF its measured 100-epoch value, i.e.
#    <= 0.024235 pp/epoch.  If it is not, the horizon bought epochs but not
#    convergence, and the batch is UNRESOLVED-HORIZON-INEFFECTIVE: the cliff
#    at E is then still a snapshot of a rising curve and the budget question
#    is NOT settled either way.  (The registered fit predicts the window
#    [752,771] slope at 0.0058 pp/epoch, an 8.4x fall, so this gate has
#    real headroom and is not a formality.)
#
# THE TWO VERDICTS.  Both are reported; neither may be quoted without the
# other.
#
#  (a) ON THE SIZE AT LONG BUDGET.  DE = M(k49, E) - M(k50, E):
#      SURVIVED   DE >= 12.4170 pp -- the cliff keeps at least half of cts1's
#                 own measured cliff at 7.72x the budget.  The standing risk
#                 at CORRECTIONS 150 is DISCHARGED AT THIS HORIZON, and only
#                 at this horizon.
#      COLLAPSED  DE <= 1.4878 pp (= 2 * SE_ARM_DIFF) -- the cliff is gone.
#                 CORRECTIONS 147's cliff is then a TRUNCATION ARTEFACT and
#                 must be WITHDRAWN as a partition finding, not re-scoped.
#      ATTENUATED in between -- neither branch claimed.
#
#  (b) ON THE DIRECTION.  dD = DE - D100, both measured in these same runs:
#      GROWS   dD >= +2.1040 pp (= 2 * SE_dD)
#      HOLDS   |dD| <  2.1040 pp -- the cliff is budget-invariant over
#              100 -> 772 epochs on this cell.
#      SHRINKS dD <= -2.1040 pp
#
# WHAT `SHRINKS` MEANS, WRITTEN DOWN BEFORE THE DATA EXIST SO IT CANNOT BE
# EXPLAINED AWAY AFTERWARDS.  ***SHRINKS IS THE PREDICTED OUTCOME.***  Two
# independent extrapolations from cts2's own curves -- integrating the fitted
# slope law, and the direct saturating fit -- agree to 0.3 pp that the k50 arm
# gains about +4.4 pp by epoch 300 and about +7.3 pp by epoch 772 while k49
# stays flat, i.e. dD is predicted at roughly -7.3 pp = -6.9 SE_dD.  A SHRINKS
# verdict is therefore NOT a surprise, NOT a null, and may NOT be reported as
# "the cliff held up".  If SHRINKS fires:
#   * CORRECTIONS 147's cliff numeral is BUDGET-DEPENDENT.  Every future
#     quotation of 24.834 / 25.163 / 24.767 pp carries the scope
#     "at 100 epochs", in the same sentence, permanently.
#   * cpk1's capture curve and its argmax k* = 49 (CORRECTIONS 146) are
#     100-epoch measurements throughout and INHERIT THAT SCOPE.  The single
#     peak may not be quoted as a property of the partition without it.
#   * If SHRINKS fires AND DE < 12.4170 pp, CORRECTIONS 147's headline -- the
#     largest single-tensor partition effect the campaign has measured -- is
#     DOWNGRADED to a statement about a 100-epoch budget, and FINDINGS must
#     record that the effect more than halves when the budget is bought.
#   * If SHRINKS fires AND DE <= 1.4878 pp, verdict (a) is COLLAPSED and the
#     finding is withdrawn outright, per (a).
# The decomposition dD = d(k49) - d(k50) is computed and reported in ALL
# branches, so a shrink driven by the WINNER decaying is never reported as a
# shrink driven by the loser catching up, or the reverse.
#
# WHAT A `HOLDS` VERDICT WOULD MEAN.  It would REFUTE the extrapolation above,
# which is a real and registered prediction, and it would say the k50 arm's
# rise between epochs 100 and 772 is matched by the k49 arm's.  That is the
# outcome that most strengthens CORRECTIONS 147, and precisely because it is
# the flattering one it is held to the same 2-SE bar.
#
# MAY NOT CLAIM: anything about m, about any cut but 49/50, about CAPTURE or
# either anchor (none were run), about the RELEASED floor at long budget (not
# runnable at this horizon, see above), about any other dataset, network,
# alpha0, meta-stepsize or optimiser pair, about which TENSOR carries the step
# (cts1 owns that), about the cliff at any horizon other than 100 and 772, or
# about the asymptote.
# =============================================================================

from __future__ import annotations

import argparse
import collections
import csv
import glob
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV = os.path.join(REPO, "results", "all_runs.csv")

# ---- FROZEN REGISTRATION ----------------------------------------------------
PREFIX = "cts3-"
NET, DSET = "ResNet18_c100", "CIFAR100"
MS, ALPHA0, AUG = "1e-3", "1e-6", "1"
EPOCHS = 772                     # E, derived above from the measured slopes
CONTROL_EPOCHS = 100             # the in-batch, in-run control readout
BATCH, GAMMA = 100, "1"
BASE_ALG, META_ALG = "SGDm", "Lion"
SEEDS = (0, 1, 2)
N_TENSORS = 62
CLIP_C = "-15:-2.3026"           # the campaign standard; the only level run
CLAMP = "C"
KGRID = (49, 50)
ARMS = ("k49", "k50")
CELLS = ARMS
N_RUNS = len(CELLS) * len(SEEDS)          # 6

# ---- the horizon arithmetic, re-derived by --selftest ------------------------
# (midpoint, mean 20-epoch OLS slope) measured on cts2's k50-C .out files.
CTS2_K50C_WINDOWS = ((39.5, 0.13824), (49.5, 0.09706), (59.5, 0.08062),
                     (69.5, 0.05999), (79.5, 0.06083), (89.5, 0.04847))
FIT_FROM_MID = 49.5              # windows earlier than this are excluded
FIT_A = 8.1380
FIT_P = 1.1362
CONVERGED_SLOPE = 0.00426        # k49-C's own measured terminal slope
SLOPE_K50_AT_100 = 0.04847       # k50-C's measured terminal slope at 100 ep
META_STEPS = EPOCHS * 500        # 386000; one meta-step per training iteration
BETA_INIT = -13.815511           # float32 log(1e-6)
FLOOR_C = -15.0
CEIL = -2.3026
REACHABLE_MIN_AT_E = BETA_INIT - 1e-3 * META_STEPS   # -399.815511
F32_MIN_SUBNORMAL = 1.401298464324817e-45           # below this alpha == 0
F32_UNDERFLOW_BETA = -103.278930                    # ln(F32_MIN_SUBNORMAL)
# every defensible window subset of the SAME measured windows, and the E each
# implies at the anchored target.  --selftest re-derives all of them.
FIT_VARIANTS = (
    ("all six windows",           0.0,  None),
    ("mid >= 49.5 (REGISTERED)", 49.5,  None),
    ("mid >= 59.5",              59.5,  None),
    ("mid >= 69.5",              69.5,  None),
)
NONOVERLAP_WINDOWS = ((29.5, 0.25395), (49.5, 0.09706),
                      (69.5, 0.05999), (89.5, 0.04847))
E_BRACKET = (414.0, 1821.0)
MAX_PRED_SLOPE_AT_E = 0.008724

# ---- the noise floor, re-derived from the corpus by --selftest ---------------
SIGMA_W = 0.9111
SIGMA_DF = 50
SIGMA_CELLS = 25

# ---- cts1's own in-batch cliff, re-derived from the corpus.  It enters ONLY
# through PREMISE_BAR and SURVIVE_BAR, never as a comparator arm. -------------
CTS1_PREFIX = "cts1-"
CTS1_K49 = 55.168667
CTS1_K50 = 30.334667
CTS1_CLIFF = 24.834000

# ---- BARS.  --selftest quotes every one in SE units. ------------------------
PREMISE_BAR = 12.4170     # R1: half of cts1's own in-batch cliff
SURVIVE_BAR = 12.4170     # verdict (a)
COLLAPSE_BAR = 1.4878     # verdict (a): 2 * SE_ARM_DIFF
DIRECTION_BAR = 2.1040    # verdict (b): 2 * SE_dD
NOISY_BAR = 2.7333        # R3: 3 * SIGMA_W
DEAD_BAR = 5.00           # R2
R4_SLOPE_BAR = 0.024235   # R4: half of SLOPE_K50_AT_100
SLOPE_WINDOW = 20         # epochs in the terminal OLS slope window

EP_RE = re.compile(r"Epoch\s+(\d+).*?Test Accuracy:\s*([0-9.]+)", re.I)
EPTR_RE = re.compile(r"Epoch\s+(\d+).*?Train Accuracy:\s*([0-9.]+)", re.I)
ARGS_RE = re.compile(r"^\s*ARGS:\s*(.*)$")
ENV_RE = re.compile(r"^\s*ENV:\s*(.*)$")
NAME_RE = re.compile(r"^cts3-(k49|k50)-C-s([012])-(\d+)\.out$")


# =============================================================================
# the registered grid, spelled out so nothing is reconstructed at score time
# =============================================================================
def kspec(k):
    return "[%d,%d]" % (k, N_TENSORS - k)


def arm_spec(arm):
    return {"k49": kspec(49), "k50": kspec(50)}[arm]


# =============================================================================
# the corpus.  Used ONLY by --selftest, to re-derive the noise floor, cts1's
# cliff, the no-cts3-row premise, and the max-horizon claim.  score() never
# opens the CSV.
# =============================================================================
def _csv_rows(path=CSV):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if (r.get("superseded") or "0") != "1"]


def _batch_of(run):
    return re.split(r"[-_]", str(run))[0]


def _in_cell(r):
    """The cts1 cell, EXACTLY: the clamped 100-epoch regime the corpus is in."""
    return (r["network"] == NET and r["dataset"] == DSET
            and r["base"] == BASE_ALG and r["meta"] == META_ALG
            and r["meta_stepsize"] == MS and r["alpha0"] == ALPHA0
            and r["gamma"] == GAMMA and r["augment"] == AUG
            and r["beta_clip"] == CLIP_C and not (r.get("hier") or "").strip()
            and r["batch_size"] == str(BATCH)
            and r["epochs_done"] == str(CONTROL_EPOCHS)
            and r["collapsed"] == "0")


M2_RE = re.compile(r"^\[\d+,\d+\]$")


def noise_floor_from_csv(path=CSV):
    """Pooled WITHIN (batch x granularity) SD of plateau5, m = 2 arms only."""
    cells = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not _in_cell(r) or not M2_RE.match(str(r["granularity"] or "")):
            continue
        try:
            cells[(_batch_of(r["run"]), r["granularity"])].append(float(r["plateau5"]))
        except (TypeError, ValueError):
            pass
    ss, df, nc = 0.0, 0, 0
    for v in cells.values():
        if len(v) >= 2:
            m = statistics.mean(v)
            ss += sum((x - m) ** 2 for x in v)
            df += len(v) - 1
            nc += 1
    return (math.sqrt(ss / df) if df else float("nan")), df, nc


def cts1_cliff_from_csv(path=CSV):
    g = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not str(r.get("run") or "").startswith(CTS1_PREFIX):
            continue
        try:
            g[r["granularity"]].append(float(r["plateau5"]))
        except (TypeError, ValueError):
            pass
    a, b = kspec(49), kspec(50)
    if a not in g or b not in g:
        return None
    return (statistics.mean(g[a]), statistics.mean(g[b]),
            statistics.mean(g[a]) - statistics.mean(g[b]), len(g[a]), len(g[b]))


def premise_no_cts3_rows(path=CSV):
    return sum(1 for r in _csv_rows(path)
               if str(r.get("run") or "").startswith(PREFIX))


def max_epochs_on_net(path=CSV):
    """The horizon claim: nothing on this network has ever exceeded 100 ep."""
    best = 0
    for r in _csv_rows(path):
        if r.get("network") != NET:
            continue
        e = str(r.get("epochs_done") or "")
        if e.isdigit():
            best = max(best, int(e))
    return best


# =============================================================================
# the SE arithmetic.  Each contrast gets the SE of ITS OWN linear combination.
# =============================================================================
def se_of_combo(coefs, sigma=SIGMA_W, n=3):
    return sigma * math.sqrt(sum(c * c for c in coefs) / float(n))


SE_ARM_DIFF = se_of_combo((1, -1))            # D100, DE
SE_DD = se_of_combo((1, -1, -1, 1))           # dD, treated as UNPAIRED


# =============================================================================
# the horizon fit, re-derived by --selftest from the frozen windows
# =============================================================================
def fit_power(windows, from_mid=FIT_FROM_MID):
    xs = [math.log(m) for m, s in windows if m >= from_mid]
    ys = [math.log(s) for m, s in windows if m >= from_mid]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sxy / sxx
    return math.exp(my - slope * mx), -slope, n


def horizon_for(target, A=FIT_A, p=FIT_P):
    return (A / target) ** (1.0 / p)


def predicted_slope(t, A=FIT_A, p=FIT_P):
    return A * t ** (-p)


# =============================================================================
# reading the runs.  Nothing is taken from the launcher's header.
# =============================================================================
def _tokens(line):
    import shlex
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def parse_args_line(line):
    """{flag: value} plus the list of flags that appear more than once."""
    toks = _tokens(line)
    seen, order = {}, []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:]
            v = ""
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                v = toks[i + 1]
                i += 1
            order.append(k)
            seen[k] = v
        i += 1
    dup = sorted({k for k in order if order.count(k) > 1})
    return seen, dup


def parse_env_line(line):
    env = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            env[k] = v
    return env


def read_out(path):
    args = env = None
    test, train = {}, {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if args is None:
                    m = ARGS_RE.match(line)
                    if m:
                        args = m.group(1).strip()
                        continue
                if env is None:
                    m = ENV_RE.match(line)
                    if m:
                        env = m.group(1).strip()
                        continue
                m = EP_RE.search(line)
                if m:
                    test[int(m.group(1))] = float(m.group(2))
                m = EPTR_RE.search(line)
                if m:
                    train[int(m.group(1))] = float(m.group(2))
    except IOError:
        return None
    if args is None:
        return None
    eff, dup = parse_args_line(args)
    return {"args": eff, "dup": dup, "argsline": args,
            "env": parse_env_line(env or ""), "envline": env or "",
            "test": test, "train": train, "path": path}


def tail5(series, budget, w=5):
    """PRIMARY window.  Mean of the last 5 epochs BEFORE `budget`.
    train.py prints 0-indexed epochs, so budget=100 reads epochs 95..99."""
    v = [series[e] for e in range(budget - w, budget) if e in series]
    return sum(v) / len(v) if len(v) == w else None


def ols_slope(series, budget, w=SLOPE_WINDOW):
    """OLS slope of the series on the epoch index over the w epochs ending at
    budget-1, in pp/epoch.  None if any epoch in the window is missing."""
    xs = list(range(budget - w, budget))
    if any(e not in series for e in xs):
        return None
    ys = [series[e] for e in xs]
    mx, my = sum(xs) / float(w), sum(ys) / float(w)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def norm_num(s):
    try:
        return "%g" % float(s)
    except (TypeError, ValueError):
        return None


def collect(runsdir):
    """Every cts3 .out file, with its cell MEASURED from its own ARGS and ENV."""
    recs = []
    for p in sorted(glob.glob(os.path.join(runsdir, PREFIX + "*.out"))):
        base = os.path.basename(p)
        m = NAME_RE.match(base)
        r = read_out(p)
        if r is None:
            recs.append({"path": p, "base": base, "bad": "no ARGS line",
                         "job_id": None, "meas_arm": None})
            continue
        r["base"] = base
        r["job_id"] = m.group(3) if m else None
        r["name_arm"] = m.group(1) if m else None
        r["name_seed"] = int(m.group(2)) if m else None
        spec = r["args"].get("stepsize-groups")
        r["meas_arm"] = None
        for a in ARMS:
            if spec == arm_spec(a):
                r["meas_arm"] = a
        r["clip"] = r["env"].get("BETA_CLIP")
        r["meas_clamp"] = "C" if r["clip"] == CLIP_C else None
        try:
            r["seed"] = int(r["args"].get("seed"))
        except (TypeError, ValueError):
            r["seed"] = None
        r["p5_E"] = tail5(r["test"], EPOCHS)
        r["t5_E"] = tail5(r["train"], EPOCHS)
        r["p5_100"] = tail5(r["test"], CONTROL_EPOCHS)
        r["t5_100"] = tail5(r["train"], CONTROL_EPOCHS)
        r["slope_E"] = ols_slope(r["test"], EPOCHS)
        r["slope_100"] = ols_slope(r["test"], CONTROL_EPOCHS)
        r["n_ep"] = len(r["test"])
        recs.append(r)
    return recs


# =============================================================================
# the clamp-occupancy DIAGNOSTIC.  REPORTED, NEVER A GATE.
# At the campaign-standard floor the k49 arm is known to pin (cts1: both
# groups reach -15.0000).  The open question this batch cannot control is
# whether the k50 arm's first group ALSO drifts onto the floor during the
# extra 672 epochs.  If it does, the long-horizon comparison is again inside
# the pinned regime -- which is the campaign standard, not a manipulation
# failure, so it is a SCOPE NOTE and is reported in every branch.
# =============================================================================
def beta_from_tb(tbroot):
    try:
        from tensorboard.backend.event_processing import event_accumulator as EA
    except ImportError:
        return None
    out = {}
    if not tbroot or not os.path.isdir(tbroot):
        return None
    for d in sorted(os.listdir(tbroot)):
        if not d.startswith(PREFIX):
            continue
        p = os.path.join(tbroot, d)
        try:
            acc = EA.EventAccumulator(p, size_guidance={"scalars": 0})
            acc.Reload()
            tags = [t for t in acc.Tags().get("scalars", []) if "beta" in t.lower()]
            out[d] = {t: [(e.step, e.value) for e in acc.Scalars(t)] for t in tags}
        except Exception:
            out[d] = None
    return out or None


def beta_summary(trace, floor=FLOOR_C, tol=1e-4):
    """min beta, terminal beta, and the fraction of logged points ON the floor."""
    if not trace:
        return None
    out = {}
    for tag, pts in trace.items():
        if not pts:
            continue
        vals = [v for _, v in pts]
        on = sum(1 for v in vals if v <= floor + tol)
        out[tag] = {"min": min(vals), "last": vals[-1],
                    "n": len(vals), "frac_on_floor": on / float(len(vals))}
    return out or None


# =============================================================================
def fmt(x, w=9, p=4):
    return (" " * w) if x is None else ("%*.*f" % (w, p, x))


# =============================================================================
# --selftest.  Everything the registration asserts is RE-DERIVED here, from
# the corpus and from arithmetic, and checked against THIS FILE'S constants.
# It is a PRE-REGISTRATION self-check: once cts3 rows land, the "no cts3 row
# exists" check and the corpus-dependent noise floor may legitimately move,
# and the scorer is NOT edited to make them green (RULE 16, precedent
# CORRECTIONS 150.7).
# =============================================================================
def selftest():
    ok = True

    def chk(cond, label, got="", want=""):
        nonlocal ok
        print("  %-4s %-58s %s%s" % ("PASS" if cond else "FAIL", label,
                                     got, (" (want %s)" % want) if want else ""))
        if not cond:
            ok = False

    print("cO1_cts3_score.py --selftest")
    print("-" * 78)
    print("A. the horizon, re-derived from the frozen measured windows")
    A, p, n = fit_power(CTS2_K50C_WINDOWS)
    chk(abs(A - FIT_A) < 5e-4, "power fit A reproduces", "%.6f" % A, "%.4f" % FIT_A)
    chk(abs(p - FIT_P) < 5e-4, "power fit p reproduces", "%.6f" % p, "%.4f" % FIT_P)
    chk(n == 5, "windows used in the fit", str(n), "5")
    E = horizon_for(CONVERGED_SLOPE, A, p)
    chk(abs(E - EPOCHS) < 1.0, "E = (A/s*)^(1/p) reproduces the registered horizon",
        "%.2f" % E, str(EPOCHS))
    # THE SENSITIVITY, IN FULL.  E is NOT stable to the window subset, and the
    # registered design does not pretend it is: R4 settles convergence
    # empirically.  What must hold is that R4 has headroom under EVERY variant.
    tmid = EPOCHS - SLOPE_WINDOW / 2.0 - 0.5
    Es, worst_slope = [], 0.0
    variants = [(lab, fit_power(CTS2_K50C_WINDOWS, from_mid=fm))
                for lab, fm, _ in FIT_VARIANTS]
    variants.append(("4 non-overlapping windows", fit_power(NONOVERLAP_WINDOWS, 0.0)))
    variants.append(("last 3 non-overlapping",
                     fit_power(NONOVERLAP_WINDOWS[1:], 0.0)))
    for lab, (Av, pv, nv) in variants:
        Ev = horizon_for(CONVERGED_SLOPE, Av, pv)
        sv = predicted_slope(tmid, Av, pv)
        Es.append(Ev)
        worst_slope = max(worst_slope, sv)
        print("       %-28s n=%d A=%8.4f p=%.4f  E=%7.0f  slope@E=%.6f"
              % (lab, nv, Av, pv, Ev, sv))
    chk(abs(min(Es) - E_BRACKET[0]) < 1.0 and abs(max(Es) - E_BRACKET[1]) < 1.0,
        "the E bracket across all six variants reproduces",
        "[%.0f, %.0f]" % (min(Es), max(Es)),
        "[%.0f, %.0f]" % E_BRACKET)
    chk(E_BRACKET[0] <= EPOCHS <= E_BRACKET[1],
        "the registered E lies inside its own bracket", str(EPOCHS))
    chk(abs(horizon_for(0.005) - 671) < 2, "s*=0.005 gives the stated 671",
        "%.1f" % horizon_for(0.005), "671")
    chk(abs(worst_slope - MAX_PRED_SLOPE_AT_E) < 5e-6,
        "worst predicted terminal slope at E across all variants",
        "%.6f" % worst_slope, "%.6f" % MAX_PRED_SLOPE_AT_E)
    chk(worst_slope < R4_SLOPE_BAR,
        "R4 has headroom under EVERY variant, not just the registered one",
        "%.6f" % worst_slope, "< %.6f" % R4_SLOPE_BAR)
    sE = predicted_slope(tmid)
    chk(sE < R4_SLOPE_BAR, "R4 headroom under the registered fit",
        "%.6f" % sE, "< %.6f" % R4_SLOPE_BAR)
    chk(abs(R4_SLOPE_BAR - SLOPE_K50_AT_100 / 2.0) < 1e-6,
        "R4 bar is exactly half the measured 100-epoch slope",
        "%.6f" % R4_SLOPE_BAR)

    print("B. the clamp arithmetic at this horizon")
    chk(META_STEPS == 386000, "meta-steps at E", str(META_STEPS), "386000")
    chk(REACHABLE_MIN_AT_E < -399.0,
        "worst-case reachable beta at E", "%.6f" % REACHABLE_MIN_AT_E, "< -399")
    chk(REACHABLE_MIN_AT_E < -80.0,
        "cts2's RELEASED floor -80 IS reachable at E, so it is not clamp-free",
        "%.3f < -80" % REACHABLE_MIN_AT_E)
    chk(abs(F32_UNDERFLOW_BETA - math.log(F32_MIN_SUBNORMAL)) < 5e-6,
        "the float32 underflow beta reproduces",
        "%.6f" % math.log(F32_MIN_SUBNORMAL), "%.6f" % F32_UNDERFLOW_BETA)
    chk(REACHABLE_MIN_AT_E < F32_UNDERFLOW_BETA,
        "at E, beta can reach BELOW the float32 underflow point, so no floor "
        "low enough to stay non-binding leaves alpha > 0",
        "%.3f < %.3f" % (REACHABLE_MIN_AT_E, F32_UNDERFLOW_BETA))
    chk(FLOOR_C > F32_UNDERFLOW_BETA,
        "the standard floor keeps alpha a normal float32 at every epoch",
        "exp(%.1f) = %.3e" % (FLOOR_C, math.exp(FLOOR_C)))
    chk(FLOOR_C < BETA_INIT < CEIL, "the standard clamp brackets the init",
        "%.6f in (%.1f, %.4f)" % (BETA_INIT, FLOOR_C, CEIL))

    print("C. the noise floor and the bars, re-derived from the corpus")
    if not os.path.exists(CSV):
        chk(False, "results/all_runs.csv present", CSV)
    else:
        s, df, nc = noise_floor_from_csv()
        chk(abs(s - SIGMA_W) < 5e-4, "SIGMA_W reproduces", "%.4f" % s, "%.4f" % SIGMA_W)
        chk(df == SIGMA_DF, "SIGMA df reproduces", str(df), str(SIGMA_DF))
        chk(nc == SIGMA_CELLS, "SIGMA cells reproduce", str(nc), str(SIGMA_CELLS))
        c = cts1_cliff_from_csv()
        chk(c is not None, "cts1 arms present in the corpus")
        if c:
            chk(abs(c[0] - CTS1_K49) < 5e-5, "cts1 k49 reproduces", "%.6f" % c[0])
            chk(abs(c[1] - CTS1_K50) < 5e-5, "cts1 k50 reproduces", "%.6f" % c[1])
            chk(abs(c[2] - CTS1_CLIFF) < 5e-5, "cts1 cliff reproduces", "%.6f" % c[2])
            chk(abs(PREMISE_BAR - c[2] / 2.0) < 5e-4,
                "PREMISE_BAR is half cts1's own cliff", "%.4f" % (c[2] / 2.0))
        nrows = premise_no_cts3_rows()
        chk(nrows == 0, "RULE 21: no cts3- row exists in the corpus yet",
            "%d found" % nrows, "0")
        mx = max_epochs_on_net()
        chk(mx == CONTROL_EPOCHS,
            "no %s run in the corpus exceeds 100 epochs" % NET, str(mx), "100")

    print("D. the SE arithmetic and every bar in SE units")
    chk(abs(SE_ARM_DIFF - SIGMA_W * math.sqrt(2.0 / 3.0)) < 1e-9,
        "SE_ARM_DIFF", "%.4f" % SE_ARM_DIFF)
    chk(abs(SE_DD - SIGMA_W * math.sqrt(4.0 / 3.0)) < 1e-9,
        "SE_dD (unpaired, conservative)", "%.4f" % SE_DD)
    chk(abs(COLLAPSE_BAR - 2 * SE_ARM_DIFF) < 5e-4,
        "COLLAPSE_BAR = 2*SE_ARM_DIFF", "%.4f" % (2 * SE_ARM_DIFF))
    chk(abs(DIRECTION_BAR - 2 * SE_DD) < 5e-4,
        "DIRECTION_BAR = 2*SE_dD", "%.4f" % (2 * SE_DD))
    chk(abs(NOISY_BAR - 3 * SIGMA_W) < 5e-4, "NOISY_BAR = 3*SIGMA_W",
        "%.4f" % (3 * SIGMA_W))
    print("       PREMISE/SURVIVE %8.4f pp = %6.2f SE_ARM_DIFF"
          % (SURVIVE_BAR, SURVIVE_BAR / SE_ARM_DIFF))
    print("       COLLAPSE        %8.4f pp = %6.2f SE_ARM_DIFF"
          % (COLLAPSE_BAR, COLLAPSE_BAR / SE_ARM_DIFF))
    print("       DIRECTION       %8.4f pp = %6.2f SE_dD"
          % (DIRECTION_BAR, DIRECTION_BAR / SE_DD))
    print("       predicted dD    %8.4f pp = %6.2f SE_dD   <-- SHRINKS is PREDICTED"
          % (-7.332, -7.332 / SE_DD))

    print("E. the grid")
    chk(kspec(49) == "[49,13]" and kspec(50) == "[50,12]", "the two specs",
        "%s %s" % (kspec(49), kspec(50)))
    chk(N_RUNS == 6, "run count", str(N_RUNS), "6")
    chk(EPOCHS == 772 and CONTROL_EPOCHS == 100, "horizons", "%d / %d"
        % (EPOCHS, CONTROL_EPOCHS))
    print("-" * 78)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# =============================================================================
def verdict_size(de):
    if de >= SURVIVE_BAR:
        return "SURVIVED"
    if de <= COLLAPSE_BAR:
        return "COLLAPSED"
    return "ATTENUATED"


def verdict_direction(dd):
    if dd >= DIRECTION_BAR:
        return "GROWS"
    if dd <= -DIRECTION_BAR:
        return "SHRINKS"
    return "HOLDS"


def score(runsdir, tbroot):
    recs = collect(runsdir)
    print("=" * 78)
    print("cts3 -- DOES THE k=49 -> k=50 CLIFF SURVIVE BUDGET?  E = %d epochs"
          % EPOCHS)
    print("scorer: %s   (registered before any cts3 run existed)" % __file__)
    print("=" * 78)

    # ---- G0 provenance ------------------------------------------------------
    print("\nG0 PROVENANCE")
    fail = []
    if len(recs) != N_RUNS:
        fail.append("expected %d runs, found %d" % (N_RUNS, len(recs)))
    jobs = {r.get("job_id") for r in recs if r.get("job_id")}
    if len(jobs) != len(recs):
        fail.append("job ids not distinct: %d ids for %d runs" % (len(jobs), len(recs)))
    for r in recs:
        b = r.get("base", r.get("path"))
        if r.get("bad"):
            fail.append("%s: %s" % (b, r["bad"]))
            continue
        if r["dup"]:
            fail.append("%s: repeated flag(s) %s" % (b, ",".join(r["dup"])))
        if r["meas_arm"] is None:
            fail.append("%s: stepsize-groups %r is not a registered arm"
                        % (b, r["args"].get("stepsize-groups")))
        elif r["meas_arm"] != r["name_arm"]:
            fail.append("%s: NAME says %s, ARGS say %s"
                        % (b, r["name_arm"], r["meas_arm"]))
        if r["meas_clamp"] != CLAMP:
            fail.append("%s: ENV BETA_CLIP is %r, registered %r"
                        % (b, r["clip"], CLIP_C))
        if r["seed"] != r["name_seed"]:
            fail.append("%s: NAME seed %s, ARGS seed %s"
                        % (b, r["name_seed"], r["seed"]))
        if norm_num(r["args"].get("num-epochs")) != norm_num(EPOCHS):
            fail.append("%s: ARGS num-epochs %r, registered %d"
                        % (b, r["args"].get("num-epochs"), EPOCHS))
        for flag, want in (("meta-stepsize", MS), ("alpha0", ALPHA0),
                           ("dataset", DSET), ("NN-name", NET),
                           ("batch-size", str(BATCH)), ("gamma", GAMMA),
                           ("alg-base", BASE_ALG), ("alg-meta", META_ALG)):
            got = r["args"].get(flag)
            if norm_num(got) is not None and norm_num(want) is not None:
                same = norm_num(got) == norm_num(want)
            else:
                same = str(got) == str(want)
            if not same:
                fail.append("%s: ARGS %s = %r, registered %r" % (b, flag, got, want))
        if r["env"].get("AUGMENT") != AUG:
            fail.append("%s: ENV AUGMENT %r, registered %r"
                        % (b, r["env"].get("AUGMENT"), AUG))
        if r["n_ep"] != EPOCHS:
            fail.append("%s: %d epoch lines, registered %d" % (b, r["n_ep"], EPOCHS))
    for f in fail:
        print("  FAIL  %s" % f)
    if fail:
        print("\nVERDICT: UNRESOLVED-PROVENANCE")
        return 3
    print("  PASS  %d runs, %d job ids, %d/%d epochs, NAME == ARGS == ENV, no "
          "repeated flag" % (len(recs), len(jobs), EPOCHS, EPOCHS))

    cell = collections.defaultdict(list)
    for r in recs:
        cell[r["meas_arm"]].append(r)
    for a in ARMS:
        if len(cell[a]) != len(SEEDS):
            print("  FAIL  cell %s has %d runs" % (a, len(cell[a])))
            print("\nVERDICT: UNRESOLVED-PROVENANCE")
            return 3

    # ---- the table ----------------------------------------------------------
    print("\nARMS.  plateau5 = mean of the last 5 epochs of the stated horizon.")
    print("  %-5s %-6s %10s %10s %10s %10s %12s"
          % ("arm", "seed", "test@100", "train@100", "test@E", "train@E", "slope@E"))
    for a in ARMS:
        for r in sorted(cell[a], key=lambda x: x["seed"]):
            print("  %-5s %-6d %s %s %s %s %s"
                  % (a, r["seed"], fmt(r["p5_100"], 10), fmt(r["t5_100"], 10),
                     fmt(r["p5_E"], 10), fmt(r["t5_E"], 10), fmt(r["slope_E"], 12, 5)))

    def M(a, key):
        v = [r[key] for r in cell[a] if r[key] is not None]
        return statistics.mean(v) if len(v) == len(SEEDS) else None

    def SD(a, key):
        v = [r[key] for r in cell[a] if r[key] is not None]
        return statistics.pstdev(v) * math.sqrt(len(v) / (len(v) - 1.0)) if len(v) > 1 else None

    print("\n  %-5s %10s %10s %10s %10s" % ("mean", "test@100", "test@E",
                                            "train@100", "train@E"))
    for a in ARMS:
        print("  %-5s %s %s %s %s" % (a, fmt(M(a, "p5_100"), 10), fmt(M(a, "p5_E"), 10),
                                      fmt(M(a, "t5_100"), 10), fmt(M(a, "t5_E"), 10)))

    # ---- R2 divergence ------------------------------------------------------
    print("\nR2 DIVERGENCE")
    dead = [r["base"] for r in recs
            if r["p5_E"] is None or not math.isfinite(r["p5_E"])
            or r["p5_E"] <= DEAD_BAR or r["n_ep"] != EPOCHS]
    if dead:
        for d in dead:
            print("  FAIL  %s is DEAD (plateau5 <= %.2f, non-finite, or short)"
                  % (d, DEAD_BAR))
        print("\nVERDICT: UNRESOLVED-DIVERGED")
        return 4
    print("  PASS  every run finished %d epochs with plateau5 > %.2f pp"
          % (EPOCHS, DEAD_BAR))

    # ---- R1 premise, IN BATCH ----------------------------------------------
    d100 = M("k49", "p5_100") - M("k50", "p5_100")
    de = M("k49", "p5_E") - M("k50", "p5_E")
    dd = de - d100
    print("\nR1 PREMISE (in batch, from epochs %d-%d of these same runs)"
          % (CONTROL_EPOCHS - 5, CONTROL_EPOCHS - 1))
    print("  D100 = %.4f pp = %.2f SE_ARM_DIFF   bar %.4f pp"
          % (d100, d100 / SE_ARM_DIFF, PREMISE_BAR))
    print("  cts1's own 100-epoch cliff, DESCRIPTIVE only: %.4f pp" % CTS1_CLIFF)
    if d100 < PREMISE_BAR:
        print("  FAIL  the cliff is not present in this batch at 100 epochs.")
        print("        cts1's and cts2's arms may NOT be spliced in to rescue it.")
        print("\nVERDICT: UNRESOLVED-PREMISE")
        return 5
    print("  PASS")

    # ---- R3 noise -----------------------------------------------------------
    print("\nR3 NOISE   sigma_w %.4f (df %d, %d cells), 3*sigma = %.4f pp"
          % (SIGMA_W, SIGMA_DF, SIGMA_CELLS, NOISY_BAR))
    worst, worst_a = -1.0, None
    for a in ARMS:
        s = SD(a, "p5_E")
        print("    cell %-5s SD@E %s" % (a, fmt(s, 9)))
        if s is not None and s > worst:
            worst, worst_a = s, a
    if worst > NOISY_BAR:
        print("  FAIL  cell %s SD %.4f > %.4f" % (worst_a, worst, NOISY_BAR))
        print("\nVERDICT: UNRESOLVED-NOISY")
        return 6
    print("  PASS  worst cell SD %.4f <= %.4f" % (worst, NOISY_BAR))

    # ---- R4 horizon effectiveness ------------------------------------------
    s50E = statistics.mean([r["slope_E"] for r in cell["k50"]])
    s49E = statistics.mean([r["slope_E"] for r in cell["k49"]])
    s50_100 = statistics.mean([r["slope_100"] for r in cell["k50"]])
    s49_100 = statistics.mean([r["slope_100"] for r in cell["k49"]])
    print("\nR4 HORIZON EFFECTIVENESS   terminal %d-epoch OLS slope, pp/epoch"
          % SLOPE_WINDOW)
    print("    k50  at 100 %s   at E %s   bar %.6f   (cts2 measured %.5f)"
          % (fmt(s50_100, 9, 5), fmt(s50E, 9, 5), R4_SLOPE_BAR, SLOPE_K50_AT_100))
    print("    k49  at 100 %s   at E %s   (reported; k49 was already converged)"
          % (fmt(s49_100, 9, 5), fmt(s49E, 9, 5)))
    print("    registered prediction of the k50 slope at E: %.6f"
          % predicted_slope(EPOCHS - SLOPE_WINDOW / 2.0 - 0.5))
    if s50E > R4_SLOPE_BAR:
        print("  FAIL  the k50 arm is still climbing at %.5f pp/epoch, more than "
              "half its 100-epoch rate." % s50E)
        print("        The horizon bought epochs but not convergence; the cliff at "
              "E is still a snapshot of a rising curve.")
        print("\nVERDICT: UNRESOLVED-HORIZON-INEFFECTIVE")
        return 7
    print("  PASS")

    # ---- the two verdicts ---------------------------------------------------
    v_size = verdict_size(de)
    v_dir = verdict_direction(dd)
    d49 = M("k49", "p5_E") - M("k49", "p5_100")
    d50 = M("k50", "p5_E") - M("k50", "p5_100")

    print("\n" + "=" * 78)
    print("PRIMARY")
    print("  D100 (in batch)  %9.4f pp = %6.2f SE_ARM_DIFF" % (d100, d100 / SE_ARM_DIFF))
    print("  DE   (in batch)  %9.4f pp = %6.2f SE_ARM_DIFF" % (de, de / SE_ARM_DIFF))
    print("  dD = DE - D100   %9.4f pp = %6.2f SE_dD  (SE unpaired, conservative)"
          % (dd, dd / SE_DD))
    print("  DECOMPOSITION    d(k49) %+9.4f   d(k50) %+9.4f   dD = d49 - d50 = %+.4f"
          % (d49, d50, d49 - d50))
    print("\n  (a) SIZE AT LONG BUDGET : %s   (survive %.4f, collapse %.4f)"
          % (v_size, SURVIVE_BAR, COLLAPSE_BAR))
    print("  (b) DIRECTION           : %s   (bar +/- %.4f pp)" % (v_dir, DIRECTION_BAR))

    if v_dir == "SHRINKS":
        print("\n  SHRINKS FIRED.  This was the REGISTERED PREDICTION (about -7.3 pp),")
        print("  not a surprise and not a null.  By registration:")
        print("   * CORRECTIONS 147's cliff numeral is BUDGET-DEPENDENT; every")
        print("     quotation of 24.834 / 25.163 / 24.767 pp now carries")
        print("     'at 100 epochs' in the same sentence.")
        print("   * cpk1's capture curve and its argmax k* = 49 are 100-epoch")
        print("     measurements and INHERIT that scope.")
        if de < SURVIVE_BAR:
            print("   * DE < %.4f pp: CORRECTIONS 147's headline is DOWNGRADED to a"
                  % SURVIVE_BAR)
            print("     statement about a 100-epoch budget.")
        if v_size == "COLLAPSED":
            print("   * DE <= %.4f pp: the cliff is a TRUNCATION ARTEFACT and the"
                  % COLLAPSE_BAR)
            print("     finding is WITHDRAWN, not re-scoped.")
    elif v_dir == "HOLDS":
        print("\n  HOLDS FIRED.  This REFUTES the registered extrapolation, which")
        print("  predicted about -7.3 pp = -6.9 SE_dD.  The cliff is budget-invariant")
        print("  over 100 -> %d epochs on this cell, and nowhere else." % EPOCHS)
    else:
        print("\n  GROWS FIRED.  The 100-epoch cliff was a LOWER BOUND, as")
        print("  CORRECTIONS 150 allowed for.  The larger numeral may be quoted only")
        print("  with its horizon attached.")

    # ---- the clamp-occupancy SCOPE NOTE, reported in every branch -----------
    print("\nSCOPE NOTE -- clamp occupancy at the campaign-standard floor %.1f"
          % FLOOR_C)
    tb = beta_from_tb(tbroot) if tbroot else None
    if not tb:
        print("  UNREAD.  TensorBoard traces were not readable at %r." % tbroot)
        print("  This does NOT change any gate or verdict -- it is a scope note --")
        print("  but the long-horizon result is then reported WITHOUT knowing")
        print("  whether the k50 arm's first group drifted onto the floor.")
    else:
        for name in sorted(tb):
            s = beta_summary(tb[name])
            if not s:
                print("  %-24s unreadable" % name)
                continue
            for tag in sorted(s):
                d = s[tag]
                print("  %-24s %-22s min %9.4f  last %9.4f  on-floor %6.1f%%"
                      % (name, tag, d["min"], d["last"], 100 * d["frac_on_floor"]))
        print("  Read this as scope, not as a gate: the standard clamp binding is")
        print("  the campaign-standard condition, and cts2 already showed the cliff")
        print("  does not depend on it AT 100 EPOCHS.  cts3 does not re-test that.")

    print("\nMAY NOT CLAIM: the asymptote; any horizon but 100 and %d; the released"
          % EPOCHS)
    print("floor at long budget; any cut but 49/50; any CAPTURE; any other cell.")
    print("=" * 78)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runsdir", nargs="?", help="directory holding cts3-*.out")
    ap.add_argument("--tb", default=None,
                    help="Tensorboard_outputs root, for the clamp scope note")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.runsdir:
        ap.error("give a runs directory, or --selftest")
    return score(a.runsdir, a.tb)


if __name__ == "__main__":
    sys.exit(main())
