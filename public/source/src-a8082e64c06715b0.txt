#!/usr/bin/env python3
# =============================================================================
# cS1_ratelaw_shape.py -- THE REGISTERED SCORER FOR THE TRAJECTORY ANALYSIS
#                         `cS1`.  ZERO GPU.  IT BUYS NO RUNS.
#
# -----------------------------------------------------------------------------
# THE RULE-21 CAVEAT, STATED FIRST BECAUSE IT LIMITS WHAT MAY BE CLAIMED
# -----------------------------------------------------------------------------
# STANDING RULE 21 requires a scorer to be committed BEFORE any run of its
# batch exists, and the proof is a wall-clock comparison of the registration
# commit against the earliest `sacct` Submit stamp of that batch.
#
# **THIS FILE HAS NO BATCH.**  It scores `.out` files that were produced by
# `cpk2` and `cts3`, both of which LANDED before this file was written
# (CORRECTIONS 156 and 158).  The "commit before earliest Submit" proof
# THEREFORE DOES NOT EXIST FOR THIS FILE AND IS NOT CLAIMED.  This is the same
# situation CORRECTIONS 149 recorded for `cQ1`.
#
# WHAT IS CLAIMED, AND IT IS THE ONLY THING CLAIMED:
#   **COMMIT BEFORE FIRST EXECUTION.**  This file is committed to git BEFORE it
#   is executed for the first time on any real `.out` file, and the CORRECTIONS
#   entry states the commit hash, the commit time and the first-execution time
#   with the margin between them.  Its `--selftest` runs only on SYNTHETIC data
#   and on the corpus CSV, never on a cpk2/cts3 trajectory, so `--selftest` may
#   be run at any time without spending the claim.
#
#   **THIS FILE DOES NOT CARRY THE RULE 21 LABEL.**  Nothing below may be
#   reported as "RULE 21 registered".  It is "registered before first
#   execution", which is a weaker and honest thing.
#
# RULE 16 STILL APPLIES IN FULL.  Run it UNEDITED.  Documented command-line
# arguments are not edits.  If it turns out to be broken it is FROZEN and a
# NEW file is registered (precedent cN1/cN2, CORRECTIONS 149).
#
# -----------------------------------------------------------------------------
# THE QUESTION.  WHY THIS ANALYSIS EXISTS.
# -----------------------------------------------------------------------------
# `cts3` (CORRECTIONS 156) produced a RATE LAW: over its two arms, the gain
# from 100 to 772 epochs is an affine function of the arm's own terminal
# 20-epoch OLS slope at epoch 100,
#       gain = 0.649507 + 152.256507 * slope100 ,
# fitted through exactly two points and therefore with ZERO residual degrees of
# freedom.  `cpk2` (CORRECTIONS 158) refuted it on the right flank and the
# refutation is sharp:
#       arm   slope@100    gain(100->772)
#       k50    +0.03750        +5.9380
#       k52    +0.06763        +2.4420
# k52's slope is 1.80x k50's while its gain is 0.41x k50's.  Gain is NOT
# monotone in slope@100 on the right flank.  CORRECTIONS 158 reports this as a
# measured fact and explicitly declines to explain it.
#
# CORRECTIONS 158 also established that the 772-epoch rank order of cpk2's arms
# is IDENTICAL to their 100-epoch rank order -- ZERO inversions -- so whatever
# is wrong is not a re-ordering of the arms.
#
# THE QUESTION THIS FILE ASKS, AND THE ONLY ONE:
#   **Do k50 and k52 differ in KIND or only in RATE?**
#   * DIFFER IN RATE  = the two arms' approaches to their own asymptotes are
#     TIME-RESCALINGS of one another.  One number per arm (a time constant)
#     describes each, the two curves have the SAME SHAPE, and cts3's rate law
#     then fails for a reason that is NOT about curve shape -- it must be the
#     map from slope100 to total gain that is mis-specified.
#   * DIFFER IN KIND  = no time-rescaling relates them.  Then a ONE-PARAMETER
#     rate law of cts3's form CANNOT fit both arms even in principle, and the
#     failure is structural rather than a bad fit.
#
# -----------------------------------------------------------------------------
# THE STRATUM.  DECLARED IN ADVANCE.
# -----------------------------------------------------------------------------
# PRIMARY, and the only stratum any verdict is taken from:
#   batch `cpk2` -- 18 runs, 6 arms {k01,k45,k47,k49,k50,k52} x seeds {3,4,5},
#   772 epochs, CIFAR-100 / ResNet18_c100 / SGDm+Lion / ms 1e-3 / alpha0 1e-6 /
#   batch 100 / AUGMENT=1 / BETA_CLIP -15:-2.3026 / PROBE=0.
#   EVERY COMPARISON THAT DECIDES ANYTHING IS WITHIN THIS ONE BATCH.
#
# SECONDARY, DESCRIPTIVE ONLY, NEVER A GATE:
#   batch `cts3` -- 6 runs, k49/k50 x seeds {0,1,2}, same cell, same horizon.
#   Reported so that the k50 shape has a disjoint-seed, disjoint-batch reading.
#   BATCH is the unit of replication (F(62,85) = 5.47, p 6.9e-13), so cts3's
#   numbers are NEVER pooled with cpk2's and never enter a bar.
#
# -----------------------------------------------------------------------------
# THE PRIMARY COLUMN.  DECLARED IN ADVANCE.
# -----------------------------------------------------------------------------
#   W(e) = the mean TEST accuracy over epoch lines e-4 .. e inclusive, read
#   directly from the run's own `.out` file with the identical regex
#   `cR1_cpk2_score.py` uses.  A trailing 5-epoch window -- a plateau5-style
#   window evaluated at EVERY epoch rather than only at the two horizons.
#
#   IT AGREES WITH cR1 BY CONSTRUCTION, and that is why this window and no
#   other:  cR1's tail5(series, 772) is the mean over epochs 767..771, which is
#   exactly W(771);  cR1's tail5(series, 100) is the mean over 95..99, which is
#   exactly W(99).  So this file's endpoints ARE cR1's endpoints, to the last
#   digit, and no estimator change can be smuggled in through the trajectory.
#
#   THE CSV `plateau` COLUMN IS BANNED AS A PRIMARY (standing rule) AND THIS
#   FILE NEVER READS IT.  `best_test` is a maximum over a run, is not a plateau
#   at all, and is never read either.  score() does not open the CSV: the CSV
#   is opened ONLY by --selftest, and only to re-derive the noise floor.
#
#   TRAIN IS READ AND REPORTED ALONGSIDE TEST AT EVERY ARM (standing rule),
#   with the identical window.  No verdict is taken from train.
#
# -----------------------------------------------------------------------------
# THE QUANTITY.  DECLARED IN ADVANCE.
# -----------------------------------------------------------------------------
# (1) TAU, the time constant of each RUN's approach to its own asymptote.
#     Model, over the fit window e in [100, 771]:
#           W(e) = A - D * exp(-(e - 100) / tau)
#     A = the asymptote, D = the amplitude of the remaining approach at e=100,
#     tau = the time constant in epochs.  THREE parameters, 672 data points.
#     FITTED WITHOUT ANY DEPENDENCY: for a FIXED tau the model is LINEAR in
#     (A, D), so the fit is a registered deterministic 1-D scan of tau over the
#     geometric grid TAU_MIN=1 .. TAU_MAX=4000 in NTAU=600 points (ratio
#     1.0139), solving the 2x2 normal equations in closed form at each grid
#     point and taking the least SSE.  No scipy, no numpy, no random seed, no
#     iteration count -- the same input gives the same output on any machine.
#     A fit whose argmin lands on either end of the grid is FLAGGED
#     (TAU-AT-GRID-EDGE) and its arm becomes UNRESOLVED.
#
# (2) RHO, THE SHAPE FINGERPRINT, AND IT IS THE DECIDING QUANTITY.
#     For each run, using ITS OWN fitted A and D, define the level that
#     corresponds to covering a fraction f of the approach:
#           L(f) = A - D * (1 - f)
#     and read off the DURABLE crossing time from the DATA, not from the fit:
#           e(f) = the smallest e in [100, 771] such that W(e') >= L(f) for
#                  EVERY e' in [e, 771].
#     "Durable" -- a suffix-minimum crossing -- so that a single noisy epoch
#     that touches a level and falls back cannot be scored as the arrival.
#     Then, with F_REF = 1 - 1/e = 0.632120558829,
#           rho(f) = (e(f) - 100) / (e(F_REF) - 100).
#
#     WHY RHO AND NOT TAU IS THE DECIDING QUANTITY.  tau is a RATE: two arms
#     may differ in tau by any factor and still have identical shape.  rho is
#     tau-FREE by construction -- it is a ratio of two crossing times of the
#     same run, so the run's own time constant divides out of it.  Under a
#     single exponential of ANY tau,
#           rho(f) = ln(1/(1-f)) / ln(1/(1-F_REF)) = ln(1/(1-f)) ,
#     a PURE NUMBER carrying no fitted quantity at all:
#           f = 0.25 -> 0.287682   f = 0.50 -> 0.693147
#           f = 0.75 -> 1.386294   f = 0.90 -> 2.302585
#     --selftest re-derives all four from arithmetic.  So rho gives a null with
#     no parameters in it, and a between-arm comparison with no fitted rate in
#     it.  That is exactly the KIND-vs-RATE separation this file was built for.
#
# (3) PI, THE PERSISTENCE RATIO -- a SECONDARY, always printed, never a gate.
#           PI = [ W(199) - W(99) ] / [ 100 * slope100 ]
#     where slope100 is the arm's own terminal 20-epoch OLS slope at epoch 100,
#     computed by the identical estimator cR1 registered (epochs 80..99).
#     Under a locally valid linear extrapolation PI = 1.  PI LOCALISES THE
#     RATE-LAW FAILURE IN TIME: if k52's PI is near 1, its slope@100 described
#     the very next 100 epochs correctly and the failure is a long-horizon one;
#     if PI is far below 1, the slope@100 was a transient and the failure is
#     immediate.  This distinction cannot be read off the two endpoints cts3
#     and cpk2 report, which is the whole reason the trajectory is worth
#     opening.
#
# -----------------------------------------------------------------------------
# ARMS THAT WERE ALREADY CONVERGED AT 100.  DECLARED IN ADVANCE.
# -----------------------------------------------------------------------------
# An arm that had already converged by epoch 100 has essentially no approach
# left to shape, so its D is small, its crossing times are noise, and its rho
# is meaningless.  Such an arm is NOT dropped silently -- it is measured,
# printed in full (W at every checkpoint, tau, D, the fit residual, train
# alongside test) and then EXCLUDED FROM THE SHAPE VERDICT by two gates that
# must BOTH hold for an arm to be SHAPE-ELIGIBLE:
#
#   E1  NOT-CONVERGED-AT-100.  The arm's mean terminal 20-epoch OLS slope at
#       epoch 100 must exceed CONV_BAR = 0.024235 pp/epoch, which is half
#       cts2's measured k50 slope at 100 (0.04847) -- the identical bar cts3
#       registered as its R4 and cR1 registered as its R-CONV.  RE-USED, not
#       re-invented.  This is the gate that names the already-converged arms.
#
#   E2  AMPLITUDE-READABLE.  The arm's mean fitted amplitude D must exceed
#       AMP_BAR = AMP_K * SIGMA_WIN, where SIGMA_WIN is the noise on a single
#       W(e) value, MEASURED IN BATCH and re-derived at run time, never copied:
#       SIGMA_TAIL = the pooled within-run SD of the per-epoch test accuracy
#       over the last SIGMA_TAIL_W = 50 epochs of every run in the batch (a
#       region cR1's R-CONV already certified is flat, so the SD there is
#       jitter and not trend), and SIGMA_WIN = SIGMA_TAIL / sqrt(5) because
#       W(e) averages 5 epochs.  AMP_K = 10: an approach must be at least ten
#       window-noise units tall before its crossing times mean anything.
#
# BOTH BARS ARE RE-DERIVED AT RUN TIME AND PRINTED WITH THEIR df.  Neither is a
# literal copied from another file.  If the derived AMP_BAR excludes an arm the
# verdict says so and the branch UNRESOLVED-ELIGIBILITY fires rather than a
# shape claim being made on one arm.
#
# -----------------------------------------------------------------------------
# THE VERDICT.  WHAT RESULT SHOWS *KIND* RATHER THAN *RATE*.  IN ADVANCE.
# -----------------------------------------------------------------------------
# Let the eligible arms be compared pairwise on the primary pair (the two arms
# with the largest slope100, which the design expects to be k50 and k52).  Let
#     RHO_BAR(f) = 2 * sigma_rho(f) * sqrt(2/3)
# where sigma_rho(f) is the POOLED WITHIN-ARM BETWEEN-SEED SD of rho(f) over
# the eligible arms (df = sum over eligible arms of n_seeds - 1), and the
# sqrt(2/3) is the SE of a difference of two 3-seed means.  Everything in this
# bar is measured inside cpk2.
#
#   DIFFERENT-KIND        at least one f in F_GRID has
#                         |rho_A(f) - rho_B(f)| > RHO_BAR(f).
#                         ==> the two arms' approaches are NOT time-rescalings
#                         of one another.  A one-parameter rate law of cts3's
#                         form cannot fit both, and the k52 miss is STRUCTURAL.
#
#   SAME-KIND-DIFFERENT-RATE   every f has |rho_A(f) - rho_B(f)| <= RHO_BAR(f).
#                         ==> the two arms have the same normalised approach
#                         shape and differ only in tau (and in A and D).  Then
#                         cts3's rate law does NOT fail because the curves have
#                         different shapes; it fails because the map from
#                         slope100 to TOTAL GAIN is mis-specified.  The
#                         sub-branch EXPONENTIAL / NON-EXPONENTIAL is reported
#                         alongside, from |rho(f) - RHO_EXP(f)| vs RHO_BAR(f),
#                         but it does NOT change the KIND-vs-RATE verdict:
#                         two arms can share a shape that is not exponential.
#
#   UNRESOLVED-ELIGIBILITY     fewer than 2 SHAPE-ELIGIBLE arms.
#   UNRESOLVED-NONDURABLE      an eligible run has no durable crossing for some
#                              registered f, or its e(F_REF) is not > 100.
#   UNRESOLVED-FIT             an eligible run's tau lands on a grid edge.
#   UNRESOLVED-PROVENANCE      the 18 cpk2 files are not all present, complete
#                              at 772 epoch lines, and self-consistent on their
#                              own ARGS and ENV lines.
#
# -----------------------------------------------------------------------------
# MAY NOT CLAIM.
# -----------------------------------------------------------------------------
# * NO MECHANISM.  This file measures the SHAPE of two approaches.  It does not
#   explain why either arm has the shape it has, and a KIND verdict is not an
#   explanation of the k52 miss -- it is a statement about what class of law
#   could ever fit it.
# * NO REPAIRED RATE LAW.  Nothing here licenses a new gain(slope) formula.  A
#   two-arm shape comparison cannot fit a law, and any refit would have the
#   same zero-residual-df defect that made cts3's line untested.
# * NO ASYMPTOTE.  A is a FITTED asymptote of a 672-epoch window, not a
#   measured limit.  cpk2 and cts3 each measured TWO budgets; CORRECTIONS 156
#   and 157 both forbid reading a limit into that and this file does not.
# * NOT POOLED ACROSS BATCHES.  cts3's runs are printed and never pooled.
# * NO OTHER CELL, HORIZON, CUT POSITION, GRANULARITY, m, BOX OR OPERATOR.
# * NO CAPTURE.  There is no layerwise anchor in cpk2 and none is manufactured.
# * NOTHING ABOUT k46 OR k48, which have never been run at any horizon.
# * NO CLAIM ABOUT WHICH TENSOR CARRIES ANYTHING.  cts1 and scl1 own that.
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
PRIMARY_PREFIX = "cpk2-"
SECONDARY_PREFIX = "cts3-"
NET, DSET = "ResNet18_c100", "CIFAR100"
MS, ALPHA0, AUG = "1e-3", "1e-6", "1"
BASE_ALG, META_ALG = "SGDm", "Lion"
BATCH, GAMMA = 100, "1"
CLIP_C = "-15:-2.3026"
EPOCHS = 772                       # the horizon; epoch lines are 0..771
CONTROL_EPOCHS = 100               # the in-run control readout
PRIMARY_ARMS = ("k01", "k45", "k47", "k49", "k50", "k52")
PRIMARY_SEEDS = (3, 4, 5)
N_PRIMARY = len(PRIMARY_ARMS) * len(PRIMARY_SEEDS)      # 18
SECONDARY_ARMS = ("k49", "k50")
SECONDARY_SEEDS = (0, 1, 2)
N_SECONDARY = len(SECONDARY_ARMS) * len(SECONDARY_SEEDS)  # 6
N_TENSORS = 62

WIN = 5                            # the plateau5-style window, = cR1's tail5
SLOPE_WINDOW = 20                  # cR1's terminal OLS window
FIT_LO, FIT_HI = 100, 771          # the fit window, inclusive, in W(e) index
PERSIST_AT = 199                   # W(199) - W(99) is the "next 100 epochs"

# ---- the tau scan.  Deterministic, dependency-free, registered. -------------
TAU_MIN, TAU_MAX, NTAU = 1.0, 4000.0, 600

# ---- the shape fingerprint --------------------------------------------------
F_GRID = (0.25, 0.50, 0.75, 0.90)
F_REF = 1.0 - 1.0 / math.e         # 0.632120558829...

# ---- the eligibility bars ---------------------------------------------------
SLOPE_K50_AT_100 = 0.04847         # cts2's measurement, which set cts3's R4 bar
CONV_BAR = 0.024235                # E1: half of it; cR1's R-CONV bar, re-used
AMP_K = 10.0                       # E2: an approach must be 10 window-noises tall
SIGMA_TAIL_W = 50                  # epochs of certified-flat tail used for jitter

# ---- corpus constants, re-derived by --selftest, used for CONTEXT ONLY.
# NO BAR IN THIS FILE DERIVES FROM THEM.  They are printed so the reader can see
# the between-seed level noise beside the within-run jitter this file uses.
SIGMA_W_REG = 0.917280             # cR1's registered 100-epoch pooled within-cell SD
SIGMA_DF_REG = 58
SIGMA_CELLS_REG = 29
SIGMA_MEMBERS_REG = 87

# ---- cts3's rate law, quoted so the residuals can be printed ----------------
GAIN_A = 0.649507
GAIN_B = 152.256507

EP_RE = re.compile(r"Epoch\s+(\d+).*?Test Accuracy:\s*([0-9.]+)", re.I)
EPTR_RE = re.compile(r"Epoch\s+(\d+).*?Train Accuracy:\s*([0-9.]+)", re.I)
ARGS_RE = re.compile(r"^\s*ARGS:\s*(.*)$")
ENV_RE = re.compile(r"^\s*ENV:\s*(.*)$")
PRIM_NAME_RE = re.compile(r"^cpk2-(k01|k45|k47|k49|k50|k52)-s([345])-(\d+)\.out$")
SEC_NAME_RE = re.compile(r"^cts3-(k49|k50)-C-s([012])-(\d+)\.out$")

CHECKPOINTS = (99, 149, 199, 299, 399, 499, 599, 699, 771)


# =============================================================================
# reading the runs.  Nothing is taken from any launcher header or any doc.
# =============================================================================
def _tokens(line):
    import shlex
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def parse_args_line(line):
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


def window_series(series, lo, hi, w=WIN):
    """W(e) for e in [lo, hi], the mean of epochs e-w+1..e.  None if short."""
    out = {}
    for e in range(lo, hi + 1):
        xs = [series[i] for i in range(e - w + 1, e + 1) if i in series]
        out[e] = (sum(xs) / w) if len(xs) == w else None
    return out


def ols_slope(series, budget, w=SLOPE_WINDOW):
    """cR1's estimator, byte-for-byte: OLS slope over the w epochs ending at
    budget-1, in pp/epoch."""
    xs = list(range(budget - w, budget))
    if any(e not in series for e in xs):
        return None
    ys = [series[e] for e in xs]
    mx, my = sum(xs) / float(w), sum(ys) / float(w)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


# =============================================================================
# THE FIT.  W(e) = A - D * exp(-(e - FIT_LO)/tau), by a registered 1-D scan.
# =============================================================================
def tau_grid():
    r = (TAU_MAX / TAU_MIN) ** (1.0 / (NTAU - 1))
    return [TAU_MIN * (r ** j) for j in range(NTAU)]


def _linear_at_tau(es, ws, tau):
    """For fixed tau solve min_{A,D} sum (W - A + D*x)^2 with x=exp(-(e-lo)/tau).
    Returns (A, D, sse) or None if the design is degenerate."""
    n = len(es)
    s_x = s_xx = s_w = s_wx = 0.0
    xs = []
    for e, w in zip(es, ws):
        x = math.exp(-(e - FIT_LO) / tau)
        xs.append(x)
        s_x += x
        s_xx += x * x
        s_w += w
        s_wx += w * x
    # W = A*1 + D*(-x)  =>  basis u=1, v=-x
    S11, S12, S22 = float(n), -s_x, s_xx
    b1, b2 = s_w, -s_wx
    det = S11 * S22 - S12 * S12
    if abs(det) < 1e-12:
        return None
    A = (b1 * S22 - b2 * S12) / det
    D = (S11 * b2 - S12 * b1) / det
    sse = 0.0
    for w, x in zip(ws, xs):
        r = w - (A - D * x)
        sse += r * r
    return A, D, sse


def fit_exponential(wser):
    """Scan tau over the registered geometric grid; keep the least SSE."""
    es = [e for e in range(FIT_LO, FIT_HI + 1) if wser.get(e) is not None]
    if len(es) != (FIT_HI - FIT_LO + 1):
        return None
    ws = [wser[e] for e in es]
    grid = tau_grid()
    best = None
    for j, tau in enumerate(grid):
        r = _linear_at_tau(es, ws, tau)
        if r is None:
            continue
        A, D, sse = r
        if best is None or sse < best[3]:
            best = (A, D, tau, sse, j)
    if best is None:
        return None
    A, D, tau, sse, j = best
    mw = sum(ws) / len(ws)
    sst = sum((w - mw) ** 2 for w in ws)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rmse = math.sqrt(sse / len(ws))
    # longest same-sign residual run, as a fraction of the window: a
    # mis-specified form shows itself as long stretches of one-signed residual.
    longest = cur = 0
    prev = 0
    for e, w in zip(es, ws):
        r = w - (A - D * math.exp(-(e - FIT_LO) / tau))
        s = 1 if r >= 0 else -1
        cur = cur + 1 if s == prev else 1
        prev = s
        longest = max(longest, cur)
    return {"A": A, "D": D, "tau": tau, "sse": sse, "r2": r2, "rmse": rmse,
            "edge": (j == 0 or j == NTAU - 1), "signrun": longest / float(len(es))}


def durable_crossing(wser, level):
    """Smallest e in [FIT_LO, FIT_HI] with W(e') >= level for ALL e' >= e."""
    e = FIT_HI
    if wser.get(e) is None or wser[e] < level:
        return None
    best = e
    while e >= FIT_LO:
        v = wser.get(e)
        if v is None or v < level:
            break
        best = e
        e -= 1
    return best


def rho_vector(wser, A, D):
    """rho(f) for f in F_GRID, referenced to F_REF.  None if not readable."""
    def ef(f):
        return durable_crossing(wser, A - D * (1.0 - f))
    eref = ef(F_REF)
    if eref is None or eref <= FIT_LO:
        return None, eref, {}
    raw = {}
    out = {}
    for f in F_GRID:
        e = ef(f)
        if e is None:
            return None, eref, raw
        raw[f] = e
        out[f] = (e - FIT_LO) / float(eref - FIT_LO)
    return out, eref, raw


def rho_exponential(f):
    """The tau-free signature of a single exponential.  A PURE NUMBER."""
    return math.log(1.0 / (1.0 - f)) / math.log(1.0 / (1.0 - F_REF))


# =============================================================================
# collecting a batch
# =============================================================================
def collect(runsdir, prefix, name_re, arms, seeds):
    recs = []
    for p in sorted(glob.glob(os.path.join(runsdir, prefix + "*.out"))):
        base = os.path.basename(p)
        m = name_re.match(base)
        if m is None:
            recs.append({"path": p, "base": base, "bad": "name does not parse"})
            continue
        r = read_out(p)
        if r is None:
            recs.append({"path": p, "base": base, "bad": "no ARGS line"})
            continue
        r["base"] = base
        r["arm"] = m.group(1)
        r["seed_name"] = int(m.group(2))
        r["job_id"] = m.group(3)
        r["bad"] = None
        try:
            r["seed"] = int(r["args"].get("seed"))
        except (TypeError, ValueError):
            r["seed"] = None
        r["n_ep"] = len(r["test"])
        r["Wt"] = window_series(r["test"], WIN - 1, EPOCHS - 1)
        r["Wr"] = window_series(r["train"], WIN - 1, EPOCHS - 1)
        r["w99"] = r["Wt"].get(CONTROL_EPOCHS - 1)
        r["wE"] = r["Wt"].get(EPOCHS - 1)
        r["r99"] = r["Wr"].get(CONTROL_EPOCHS - 1)
        r["rE"] = r["Wr"].get(EPOCHS - 1)
        r["gain"] = (r["wE"] - r["w99"]) if (r["wE"] is not None
                                             and r["w99"] is not None) else None
        r["tgain"] = (r["rE"] - r["r99"]) if (r["rE"] is not None
                                              and r["r99"] is not None) else None
        r["slope100"] = ols_slope(r["test"], CONTROL_EPOCHS)
        r["slopeE"] = ols_slope(r["test"], EPOCHS)
        w2 = r["Wt"].get(PERSIST_AT)
        r["persist_num"] = (w2 - r["w99"]) if (w2 is not None
                                               and r["w99"] is not None) else None
        r["fit"] = fit_exponential(r["Wt"])
        if r["fit"]:
            r["rho"], r["eref"], r["ecross"] = rho_vector(
                r["Wt"], r["fit"]["A"], r["fit"]["D"])
        else:
            r["rho"], r["eref"], r["ecross"] = None, None, {}
        # per-run jitter over the certified-flat tail
        tailv = [r["test"][e] for e in range(EPOCHS - SIGMA_TAIL_W, EPOCHS)
                 if e in r["test"]]
        r["tail_sd"] = statistics.stdev(tailv) if len(tailv) >= 2 else None
        r["tail_n"] = len(tailv)
        recs.append(r)
    return recs


def pooled_sd(groups):
    """Pooled within-group SD and its df.  groups = iterable of value lists."""
    ss, df = 0.0, 0
    for v in groups:
        v = [x for x in v if x is not None]
        if len(v) < 2:
            continue
        m = statistics.mean(v)
        ss += sum((x - m) ** 2 for x in v)
        df += len(v) - 1
    return (math.sqrt(ss / df) if df else None), df


# =============================================================================
# the corpus.  --selftest ONLY.  score() never opens it.
# =============================================================================
def _csv_rows(path=CSV):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if (r.get("superseded") or "0") != "1"]


def _batch_of(run):
    return re.split(r"[-_]", str(run))[0]


def _in_cell(r, epochs=CONTROL_EPOCHS):
    return (r["network"] == NET and r["dataset"] == DSET
            and r["base"] == BASE_ALG and r["meta"] == META_ALG
            and r["meta_stepsize"] == MS and r["alpha0"] == ALPHA0
            and r["gamma"] == GAMMA and r["augment"] == AUG
            and r["beta_clip"] == CLIP_C and not (r.get("hier") or "").strip()
            and r["batch_size"] == str(BATCH)
            and r["epochs_done"] == str(epochs)
            and r["collapsed"] == "0")


M2_RE = re.compile(r"^\[\d+,\d+\]$")


def noise_floor_from_csv(path=CSV):
    cells = collections.defaultdict(list)
    for r in _csv_rows(path):
        if not _in_cell(r) or not M2_RE.match(str(r["granularity"] or "")):
            continue
        try:
            cells[(_batch_of(r["run"]), r["granularity"])].append(float(r["plateau5"]))
        except (TypeError, ValueError):
            pass
    s, df = pooled_sd(cells.values())
    nm = sum(len(v) for v in cells.values())
    nc = sum(1 for v in cells.values() if len(v) >= 2)
    return s, df, nc, nm


def count_rows(path, prefix):
    return sum(1 for r in _csv_rows(path)
               if str(r.get("run") or "").startswith(prefix))


# =============================================================================
def fmt(x, w=9, p=4):
    return (" " * w) if x is None else ("%*.*f" % (w, p, x))


# =============================================================================
# --selftest.  SYNTHETIC DATA AND THE CORPUS ONLY.  It NEVER opens a cpk2 or
# cts3 .out file, so running it does not spend the commit-before-first-
# execution claim this file makes about score().
# =============================================================================
def selftest():
    ok = True

    def chk(cond, label, got="", want=""):
        nonlocal ok
        print("  %-4s %-58s %s%s" % ("PASS" if cond else "FAIL", label,
                                     got, (" (want %s)" % want) if want else ""))
        if not cond:
            ok = False

    print("cS1_ratelaw_shape.py --selftest")
    print("  (registered BEFORE FIRST EXECUTION; this file does NOT carry the")
    print("   RULE 21 label -- it has no batch and no Submit stamp to beat)")
    print("-" * 78)

    print("A. the registration itself")
    chk(PRIMARY_ARMS == ("k01", "k45", "k47", "k49", "k50", "k52"),
        "primary arms", " ".join(PRIMARY_ARMS))
    chk(PRIMARY_SEEDS == (3, 4, 5) and N_PRIMARY == 18,
        "primary stratum is cpk2, 6 arms x seeds {3,4,5} = 18 runs",
        "%s %d" % (str(PRIMARY_SEEDS), N_PRIMARY))
    chk(SECONDARY_SEEDS == (0, 1, 2) and N_SECONDARY == 6,
        "secondary stratum is cts3, k49/k50 x seeds {0,1,2} = 6 runs, "
        "DESCRIPTIVE", "%s %d" % (str(SECONDARY_SEEDS), N_SECONDARY))
    chk(WIN == 5, "the primary column is a plateau5-style 5-epoch window",
        "W = mean of e-4..e")
    chk((EPOCHS - 1) == 771 and (CONTROL_EPOCHS - 1) == 99,
        "W(771) IS cR1's tail5(772) and W(99) IS cR1's tail5(100)",
        "767..771 and 95..99")
    chk(FIT_LO == 100 and FIT_HI == 771,
        "the fit window", "[%d, %d] = %d points" % (FIT_LO, FIT_HI,
                                                    FIT_HI - FIT_LO + 1))

    print("B. the tau-free exponential signature, re-derived from arithmetic")
    chk(abs(F_REF - (1.0 - 1.0 / math.e)) < 1e-15, "F_REF = 1 - 1/e",
        "%.12f" % F_REF)
    chk(abs(math.log(1.0 / (1.0 - F_REF)) - 1.0) < 1e-12,
        "ln(1/(1-F_REF)) = 1 exactly, so rho is a pure log",
        "%.12f" % math.log(1.0 / (1.0 - F_REF)))
    want = {0.25: 0.287682, 0.50: 0.693147, 0.75: 1.386294, 0.90: 2.302585}
    for f in F_GRID:
        chk(abs(rho_exponential(f) - want[f]) < 5e-6,
            "RHO_EXP(%.2f)" % f, "%.6f" % rho_exponential(f), "%.6f" % want[f])
    chk(all(rho_exponential(F_GRID[i]) < rho_exponential(F_GRID[i + 1])
            for i in range(len(F_GRID) - 1)),
        "RHO_EXP is strictly increasing in f")

    print("C. the tau scan grid")
    g = tau_grid()
    chk(len(g) == NTAU, "grid points", str(len(g)), str(NTAU))
    chk(abs(g[0] - TAU_MIN) < 1e-12 and abs(g[-1] - TAU_MAX) < 1e-9,
        "grid spans [%.0f, %.0f] inclusive" % (TAU_MIN, TAU_MAX),
        "%.4f .. %.4f" % (g[0], g[-1]))
    chk(abs(g[1] / g[0] - (TAU_MAX / TAU_MIN) ** (1.0 / (NTAU - 1))) < 1e-12,
        "grid is geometric", "ratio %.6f" % (g[1] / g[0]))

    print("D. THE FIT AND THE FINGERPRINT, ON SYNTHETIC DATA WITH KNOWN ANSWERS")
    # tau must satisfy 671/tau >= ln(10) = 2.3026 (i.e. tau <= 291) or the
    # f = 0.90 crossing falls outside the window; that TRUNCATION LIMIT is a
    # property of the analysis, not of the fit, and is checked explicitly below.
    for tau_true, A_true, D_true in ((40.0, 56.0, 6.0), (120.0, 40.0, 2.5),
                                     (250.0, 45.0, 12.0)):
        syn = {}
        for e in range(WIN - 1, EPOCHS):
            ee = max(e, FIT_LO)
            syn[e] = A_true - D_true * math.exp(-(ee - FIT_LO) / tau_true)
        f = fit_exponential(syn)
        chk(f is not None, "synthetic tau=%.0f fits" % tau_true)
        if f:
            chk(abs(f["tau"] / tau_true - 1.0) < 0.02,
                "  tau recovered to 2% (grid resolution 1.4%)",
                "%.3f vs %.3f" % (f["tau"], tau_true))
            chk(abs(f["A"] - A_true) < 0.005 * D_true,
                "  A recovered to 0.5% of D (tau grid resolves only 1.4%)",
                "%.5f" % f["A"], "%.5f" % A_true)
            chk(abs(f["D"] - D_true) < 0.01 * D_true,
                "  D recovered to 1% of D", "%.5f" % f["D"], "%.5f" % D_true)
            chk(not f["edge"], "  tau is not on a grid edge",
                "%.3f in [%.0f,%.0f]" % (f["tau"], TAU_MIN, TAU_MAX))
            rho, eref, _ = rho_vector(syn, f["A"], f["D"])
            chk(rho is not None, "  rho readable")
            if rho:
                worst = max(abs(rho[x] - rho_exponential(x)) for x in F_GRID)
                chk(worst < 0.05,
                    "  rho MATCHES the exponential signature (max dev)",
                    "%.5f" % worst)
                print("       rho = %s"
                      % "  ".join("%.4f" % rho[x] for x in F_GRID))

    print("   a NON-exponential control: a TWO-TIMESCALE sum must NOT match")
    print("   (this is the literal DIFFERENT-KIND alternative, not a strawman)")
    two = {}
    for e in range(WIN - 1, EPOCHS):
        t = max(e, FIT_LO) - FIT_LO
        two[e] = 45.0 - 8.0 * (0.5 * math.exp(-t / 20.0)
                               + 0.5 * math.exp(-t / 200.0))
    f2 = fit_exponential(two)
    chk(f2 is not None, "  two-timescale curve fits a single exponential")
    if f2:
        chk(not f2["edge"], "  its tau is not on a grid edge", "%.2f" % f2["tau"])
        r2v, _, _ = rho_vector(two, f2["A"], f2["D"])
        chk(r2v is not None, "  rho readable on the two-timescale curve")
        if r2v:
            worst = max(abs(r2v[x] - rho_exponential(x)) for x in F_GRID)
            chk(worst > 0.20,
                "  its rho DEPARTS from the exponential signature",
                "max dev %.5f" % worst)
            print("       rho = %s"
                  % "  ".join("%.4f" % r2v[x] for x in F_GRID))
        print("       signrun %.3f (a mis-specified form holds one sign longer)"
              % f2["signrun"])

    print("   the durable-crossing rule ignores a single spike that falls back")
    base = {}
    for e in range(WIN - 1, EPOCHS):
        t = max(e, FIT_LO) - FIT_LO
        base[e] = 50.0 - 10.0 * math.exp(-t / 100.0)
    lvl = 45.0                       # inside [40.0, 50.0], so it IS crossed
    c0 = durable_crossing(base, lvl)
    chk(c0 is not None and c0 > FIT_LO, "  the clean curve HAS a crossing",
        str(c0))
    spike = dict(base)
    spike[FIT_LO + 3] = 999.0        # a lone huge value BEFORE the true crossing
    chk(durable_crossing(spike, lvl) == c0,
        "  a lone 999 before the crossing does not move it",
        "%s vs %s" % (durable_crossing(spike, lvl), c0))
    dip = dict(base)
    dip[FIT_HI - 2] = 0.0            # a late collapse must PUSH the crossing out
    chk(durable_crossing(dip, lvl) == FIT_HI - 1,
        "  a late fall-back PUSHES the durable crossing to just after it",
        "%s" % str(durable_crossing(dip, lvl)), str(FIT_HI - 1))
    late = dict(base)
    late[FIT_HI] = 0.0               # the LAST epoch below the level: no crossing
    chk(durable_crossing(late, lvl) is None,
        "  a fall-back at the LAST epoch abolishes the crossing entirely",
        str(durable_crossing(late, lvl)))

    print("E. the eligibility bars, and where each one comes from")
    chk(abs(CONV_BAR - SLOPE_K50_AT_100 / 2.0) < 1e-9,
        "E1 CONV_BAR is exactly half cts2's measured k50 slope at 100",
        "%.6f" % (SLOPE_K50_AT_100 / 2.0))
    chk(AMP_K == 10.0, "E2 AMP_K", "%.1f window-noise units" % AMP_K)
    print("       E2's SIGMA_WIN is MEASURED IN BATCH at score time from the")
    print("       last %d epochs of every run and is NOT a literal here." % SIGMA_TAIL_W)

    print("F. the corpus, for CONTEXT ONLY -- no bar in this file uses it")
    if not os.path.exists(CSV):
        chk(False, "results/all_runs.csv present", CSV)
    else:
        s, df, nc, nm = noise_floor_from_csv()
        chk(s is not None, "the 100-epoch within-cell SD re-derives")
        if s is not None:
            print("       live  sigma_w %.6f  df %d  cells %d  members %d"
                  % (s, df, nc, nm))
            print("       cR1's %.6f  df %d  cells %d  members %d (registered)"
                  % (SIGMA_W_REG, SIGMA_DF_REG, SIGMA_CELLS_REG,
                     SIGMA_MEMBERS_REG))
            chk(abs(s - SIGMA_W_REG) < 5e-5,
                "it still agrees with cR1's registered value "
                "(cpk2's rows carry 772, so they are outside this stratum)",
                "%.6f" % s, "%.6f" % SIGMA_W_REG)
        # CORPUS-CONDITIONAL, the cR1 pattern: 0 rows or exactly the batch size.
        n1 = count_rows(CSV, PRIMARY_PREFIX)
        chk(n1 in (0, N_PRIMARY),
            "cpk2 rows in the corpus are 0 or exactly %d" % N_PRIMARY,
            "%d found -> %s" % (n1, "NOT-INGESTED" if n1 == 0 else "INGESTED"),
            "0 or %d" % N_PRIMARY)
        n2 = count_rows(CSV, SECONDARY_PREFIX)
        chk(n2 in (0, N_SECONDARY),
            "cts3 rows in the corpus are 0 or exactly %d" % N_SECONDARY,
            "%d found -> %s" % (n2, "NOT-INGESTED" if n2 == 0 else "INGESTED"),
            "0 or %d" % N_SECONDARY)

    print("G. cts3's rate law, quoted so residuals can be printed")
    chk(abs(GAIN_A - 0.649507) < 1e-9 and abs(GAIN_B - 152.256507) < 1e-9,
        "gain = A + B*slope100", "A %.6f  B %.6f" % (GAIN_A, GAIN_B))
    print("       NOTE: 2 points, ZERO residual df.  This file does NOT refit")
    print("       it and does not license a replacement.")

    print("-" * 78)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# =============================================================================
def _provenance(recs, arms, seeds, epochs, label):
    """Returns a list of failure strings.  Empty list = PASS."""
    fail = []
    want = len(arms) * len(seeds)
    if len(recs) != want:
        fail.append("%s: expected %d runs, found %d" % (label, want, len(recs)))
    jobs = {r.get("job_id") for r in recs if r.get("job_id")}
    if len(jobs) != len(recs):
        fail.append("%s: %d job ids for %d runs" % (label, len(jobs), len(recs)))
    for r in recs:
        b = r.get("base", r.get("path"))
        if r.get("bad"):
            fail.append("%s: %s" % (b, r["bad"]))
            continue
        if r["dup"]:
            fail.append("%s: repeated flag(s) %s" % (b, ",".join(r["dup"])))
        if r["seed"] != r["seed_name"]:
            fail.append("%s: NAME seed %s, ARGS seed %s"
                        % (b, r["seed_name"], r["seed"]))
        if r["seed"] not in seeds:
            fail.append("%s: seed %s not in %s" % (b, r["seed"], str(seeds)))
        if r["n_ep"] != epochs:
            fail.append("%s: %d epoch lines, want %d" % (b, r["n_ep"], epochs))
        if r["env"].get("BETA_CLIP") != CLIP_C:
            fail.append("%s: ENV BETA_CLIP %r" % (b, r["env"].get("BETA_CLIP")))
        if r["env"].get("AUGMENT") != AUG:
            fail.append("%s: ENV AUGMENT %r" % (b, r["env"].get("AUGMENT")))
        for flag, wv in (("meta-stepsize", MS), ("alpha0", ALPHA0),
                         ("dataset", DSET), ("NN-name", NET),
                         ("batch-size", str(BATCH)), ("gamma", GAMMA),
                         ("alg-base", BASE_ALG), ("alg-meta", META_ALG),
                         ("num-epochs", str(epochs))):
            if str(r["args"].get(flag)) != str(wv):
                fail.append("%s: ARGS %s = %r, want %r"
                            % (b, flag, r["args"].get(flag), wv))
    return fail


def _arm_table(recs, arms, title):
    by = collections.defaultdict(list)
    for r in recs:
        if not r.get("bad"):
            by[r["arm"]].append(r)
    print("\n%s" % title)
    print("  arm   n   test@100    test@E      gain    train@100   train@E"
          "     tgain   slope@100   slope@E")
    for a in arms:
        v = by.get(a) or []
        if not v:
            continue
        def M(k):
            xs = [r[k] for r in v if r.get(k) is not None]
            return statistics.mean(xs) if xs else None
        print("  %-5s %d %s %s %s %s %s %s %s %s"
              % (a, len(v), fmt(M("w99"), 10), fmt(M("wE"), 10),
                 fmt(M("gain"), 9), fmt(M("r99"), 10), fmt(M("rE"), 10),
                 fmt(M("tgain"), 9), fmt(M("slope100"), 11, 5),
                 fmt(M("slopeE"), 10, 5)))
    return by


def _trajectory_table(by, arms, title):
    print("\n%s" % title)
    hdr = "  arm  " + "".join("%9s" % ("W(%d)" % e) for e in CHECKPOINTS)
    print(hdr)
    for a in arms:
        v = by.get(a) or []
        if not v:
            continue
        row = "  %-5s" % a
        for e in CHECKPOINTS:
            xs = [r["Wt"].get(e) for r in v if r["Wt"].get(e) is not None]
            row += "%9.4f" % statistics.mean(xs) if xs else " " * 9
        print(row)
    print("  (train, same window)")
    for a in arms:
        v = by.get(a) or []
        if not v:
            continue
        row = "  %-5s" % a
        for e in CHECKPOINTS:
            xs = [r["Wr"].get(e) for r in v if r["Wr"].get(e) is not None]
            row += "%9.4f" % statistics.mean(xs) if xs else " " * 9
        print(row)


def score(runsdir, cts3dir=None):
    print("=" * 78)
    print("cS1 -- DO k50 AND k52 DIFFER IN *KIND* OR ONLY IN *RATE*?")
    print("       the 100->772 trajectory, from .out files already on disk")
    print("scorer: %s" % os.path.basename(__file__))
    print("       registered BEFORE FIRST EXECUTION.  NOT a RULE 21 label:")
    print("       this file has no batch, so there is no Submit stamp to beat.")
    print("=" * 78)

    recs = collect(runsdir, PRIMARY_PREFIX, PRIM_NAME_RE, PRIMARY_ARMS,
                   PRIMARY_SEEDS)

    print("\nG0 PROVENANCE (PRIMARY stratum: cpk2)")
    fail = _provenance(recs, PRIMARY_ARMS, PRIMARY_SEEDS, EPOCHS, "cpk2")
    for f in fail:
        print("  FAIL  %s" % f)
    if fail:
        print("\nVERDICT: UNRESOLVED-PROVENANCE")
        return 3
    envs = {r["envline"] for r in recs}
    print("  PASS  %d runs, %d job ids, %d/%d epoch lines, NAME == ARGS == ENV,"
          " no repeated flag" % (len(recs), len({r["job_id"] for r in recs}),
                                 EPOCHS, EPOCHS))
    print("  PASS  %d distinct ENV line(s) across the batch" % len(envs))

    by = _arm_table(recs, PRIMARY_ARMS,
                    "THE ENDPOINTS (they must reproduce cR1 exactly)")
    _trajectory_table(by, PRIMARY_ARMS, "THE TRAJECTORY, W(e) at checkpoints")

    # ---- the in-batch noise scales, RE-DERIVED, never copied ----------------
    print("\nTHE NOISE SCALES, RE-DERIVED IN BATCH AT RUN TIME")
    tails = [[r["test"][e] for e in range(EPOCHS - SIGMA_TAIL_W, EPOCHS)
              if e in r["test"]] for r in recs]
    sig_tail, df_tail = pooled_sd(tails)
    if sig_tail is None:
        print("  FAIL  the flat-tail jitter could not be derived")
        print("\nVERDICT: UNRESOLVED-PROVENANCE")
        return 3
    sigma_win = sig_tail / math.sqrt(WIN)
    amp_bar = AMP_K * sigma_win
    print("  SIGMA_TAIL  %.6f pp  (per-epoch test jitter, last %d epochs, "
          "df %d, %d runs)" % (sig_tail, SIGMA_TAIL_W, df_tail, len(recs)))
    print("  SIGMA_WIN   %.6f pp  (= SIGMA_TAIL / sqrt(%d), the noise on one "
          "W(e))" % (sigma_win, WIN))
    print("  AMP_BAR     %.6f pp  (= %.0f * SIGMA_WIN)   E2 threshold"
          % (amp_bar, AMP_K))
    print("  CONV_BAR    %.6f pp/epoch                  E1 threshold"
          % CONV_BAR)
    sig_lvl, df_lvl = pooled_sd([[r["wE"] for r in v] for v in by.values()])
    sig_gain, df_gain = pooled_sd([[r["gain"] for r in v] for v in by.values()])
    print("  for context only, NOT a bar here: between-seed SD of W(771) "
          "%.6f (df %d)" % (sig_lvl, df_lvl))
    print("                                    between-seed SD of the gain "
          "%.6f (df %d)" % (sig_gain, df_gain))

    # ---- the per-run fit ----------------------------------------------------
    print("\nTHE FIT  W(e) = A - D*exp(-(e-100)/tau)  over e in [%d, %d]"
          % (FIT_LO, FIT_HI))
    print("  run                    tau        A          D      A-D    "
          "W(99)      R2     RMSE  signrun  edge")
    for a in PRIMARY_ARMS:
        for r in sorted(by.get(a) or [], key=lambda z: z["seed"]):
            f = r["fit"]
            if f is None:
                print("  %-22s  FIT FAILED" % r["base"].split(".")[0])
                continue
            print("  %-22s %7.2f %8.4f %8.4f %8.4f %8.4f %7.4f %8.4f %7.3f  %s"
                  % (r["base"].split(".")[0], f["tau"], f["A"], f["D"],
                     f["A"] - f["D"], r["w99"], f["r2"], f["rmse"],
                     f["signrun"], "EDGE" if f["edge"] else "-"))

    # ---- E1 / E2 eligibility ------------------------------------------------
    print("\nSHAPE-ELIGIBILITY.  BOTH gates must hold.")
    print("  arm    mean slope@100   E1 not-conv?    mean D    E2 amp>=bar?"
          "   ELIGIBLE")
    eligible = []
    for a in PRIMARY_ARMS:
        v = by.get(a) or []
        s = statistics.mean([r["slope100"] for r in v if r["slope100"] is not None])
        ds = [r["fit"]["D"] for r in v if r["fit"]]
        d = statistics.mean(ds) if ds else None
        e1 = s > CONV_BAR
        e2 = (d is not None) and (d >= amp_bar)
        el = e1 and e2
        if el:
            eligible.append(a)
        print("  %-5s  %+13.5f   %-13s %8s   %-13s  %s"
              % (a, s, "YES" if e1 else "no (converged)",
                 ("%.4f" % d) if d is not None else "n/a",
                 "YES" if e2 else "no", "**YES**" if el else "no"))
    print("  ELIGIBLE ARMS: %s" % (", ".join(eligible) if eligible else "NONE"))
    print("  EXCLUDED, and each is excluded by a bar that was fixed before the")
    print("  data were read -- they are reported above in full, not dropped:")
    for a in PRIMARY_ARMS:
        if a not in eligible:
            print("    %-5s" % a)

    if len(eligible) < 2:
        print("\nVERDICT: UNRESOLVED-ELIGIBILITY (%d eligible arm(s), need 2)"
              % len(eligible))
        return 1

    # ---- the fingerprint ----------------------------------------------------
    print("\nTHE SHAPE FINGERPRINT rho(f) = (e(f)-100) / (e(F_REF)-100)")
    print("  tau-free by construction.  A single exponential of ANY tau gives")
    print("  rho = %s" % "  ".join("%.6f" % rho_exponential(f) for f in F_GRID))
    print("  run                  e(.25) e(.50) e(.632) e(.75) e(.90)"
          "     rho(.25)  rho(.50)  rho(.75)  rho(.90)")
    nondurable = []
    edgefit = []
    for a in eligible:
        for r in sorted(by[a], key=lambda z: z["seed"]):
            if r["fit"] and r["fit"]["edge"]:
                edgefit.append(r["base"])
            if r["rho"] is None:
                nondurable.append(r["base"])
                print("  %-22s NO DURABLE CROSSING" % r["base"].split(".")[0])
                continue
            ec = r["ecross"]
            print("  %-22s %6d %6d %7d %6d %6d   %9.5f %9.5f %9.5f %9.5f"
                  % (r["base"].split(".")[0], ec[0.25], ec[0.50], r["eref"],
                     ec[0.75], ec[0.90],
                     r["rho"][0.25], r["rho"][0.50], r["rho"][0.75],
                     r["rho"][0.90]))
    if edgefit:
        print("\nVERDICT: UNRESOLVED-FIT (tau on a grid edge: %s)"
              % ", ".join(edgefit))
        return 1
    if nondurable:
        print("\nVERDICT: UNRESOLVED-NONDURABLE (%s)" % ", ".join(nondurable))
        return 1

    # ---- the bar, pooled within arm across seeds ----------------------------
    print("\nTHE BAR.  RHO_BAR(f) = 2 * sigma_rho(f) * sqrt(2/3), sigma_rho")
    print("  pooled WITHIN arm ACROSS seeds over the eligible arms.")
    sigma_rho, df_rho = {}, {}
    for f in F_GRID:
        s, df = pooled_sd([[r["rho"][f] for r in by[a]] for a in eligible])
        sigma_rho[f], df_rho[f] = s, df
    print("  f      sigma_rho      df     RHO_BAR")
    bars = {}
    for f in F_GRID:
        bars[f] = 2.0 * sigma_rho[f] * math.sqrt(2.0 / 3.0)
        print("  %.2f   %9.5f   %4d   %9.5f" % (f, sigma_rho[f], df_rho[f],
                                                bars[f]))

    # ---- the primary pair: the two least-converged eligible arms ------------
    order = sorted(eligible,
                   key=lambda a: statistics.mean(
                       [r["slope100"] for r in by[a]]), reverse=True)
    A_, B_ = order[0], order[1]
    print("\nTHE PRIMARY PAIR: %s vs %s (the two eligible arms with the largest"
          " slope@100)" % (A_, B_))

    def armrho(a, f):
        return statistics.mean([r["rho"][f] for r in by[a]])

    print("  f      rho(%s)   rho(%s)      diff   RHO_BAR   |diff|/bar  "
          "SEPARATED?" % (A_, B_))
    diffs, sep = {}, []
    for f in F_GRID:
        d = armrho(A_, f) - armrho(B_, f)
        diffs[f] = d
        s = abs(d) > bars[f]
        if s:
            sep.append(f)
        print("  %.2f  %9.5f %9.5f %9.5f %9.5f %11.2f  %s"
              % (f, armrho(A_, f), armrho(B_, f), d, bars[f],
                 abs(d) / bars[f] if bars[f] > 0 else float("inf"),
                 "**YES**" if s else "no"))

    print("\n  AND, separately, how far each arm is from the EXPONENTIAL null:")
    print("  f      RHO_EXP   dev(%s)  /bar   dev(%s)  /bar" % (A_, B_))
    offexp = []
    for f in F_GRID:
        da = armrho(A_, f) - rho_exponential(f)
        db = armrho(B_, f) - rho_exponential(f)
        if abs(da) > bars[f] or abs(db) > bars[f]:
            offexp.append(f)
        print("  %.2f  %9.6f %9.5f %6.2f %9.5f %6.2f"
              % (f, rho_exponential(f), da,
                 abs(da) / bars[f] if bars[f] > 0 else float("inf"), db,
                 abs(db) / bars[f] if bars[f] > 0 else float("inf")))

    # ---- tau, the RATE, reported for every arm ------------------------------
    print("\nTAU -- THE RATE.  Two arms may differ here by any factor and still")
    print("  have the SAME KIND; that is exactly what rho separates out.")
    print("  arm     mean tau     sd      mean A      mean D    gain(meas)"
          "   gain(cts3 law)   residual")
    for a in PRIMARY_ARMS:
        v = [r for r in (by.get(a) or []) if r["fit"]]
        if not v:
            continue
        taus = [r["fit"]["tau"] for r in v]
        s100 = statistics.mean([r["slope100"] for r in v])
        gm = statistics.mean([r["gain"] for r in v])
        gp = GAIN_A + GAIN_B * s100
        print("  %-5s %10.2f %7.2f %11.4f %11.4f %11.4f %14.4f %10.4f"
              % (a, statistics.mean(taus),
                 statistics.stdev(taus) if len(taus) > 1 else 0.0,
                 statistics.mean([r["fit"]["A"] for r in v]),
                 statistics.mean([r["fit"]["D"] for r in v]), gm, gp, gm - gp))

    # ---- PI, the persistence ratio, SECONDARY -------------------------------
    print("\nPI -- PERSISTENCE.  [W(199)-W(99)] / [100 * slope@100].")
    print("  PI ~ 1 means the slope@100 described the NEXT 100 epochs; PI << 1")
    print("  means it was a transient.  SECONDARY, never a gate.")
    print("  arm    slope@100   100*slope   W(199)-W(99)      PI      gain@E")
    for a in PRIMARY_ARMS:
        v = [r for r in (by.get(a) or []) if r["persist_num"] is not None]
        if not v:
            continue
        s = statistics.mean([r["slope100"] for r in v])
        n = statistics.mean([r["persist_num"] for r in v])
        g = statistics.mean([r["gain"] for r in v])
        pi = n / (100.0 * s) if abs(s) > 1e-9 else float("nan")
        print("  %-5s %+10.5f %11.4f %14.4f %8.3f %11.4f" % (a, s, 100.0 * s,
                                                             n, pi, g))

    # ---- the secondary stratum: cts3, DESCRIPTIVE ---------------------------
    if cts3dir:
        print("\n" + "-" * 78)
        print("SECONDARY STRATUM: cts3 (seeds {0,1,2}).  CROSS-BATCH,")
        print("DESCRIPTIVE ONLY.  Never pooled with cpk2, never a bar, never a")
        print("gate.  BATCH is the unit of replication (F(62,85) = 5.47).")
        s3 = collect(cts3dir, SECONDARY_PREFIX, SEC_NAME_RE, SECONDARY_ARMS,
                     SECONDARY_SEEDS)
        f3 = _provenance(s3, SECONDARY_ARMS, SECONDARY_SEEDS, EPOCHS, "cts3")
        for x in f3:
            print("  NOTE  %s" % x)
        by3 = _arm_table(s3, SECONDARY_ARMS, "cts3 endpoints")
        print("\n  cts3 fingerprint (same pipeline, reported not scored)")
        print("  run                  tau        D    rho(.25)  rho(.50)"
              "  rho(.75)  rho(.90)")
        for a in SECONDARY_ARMS:
            for r in sorted(by3.get(a) or [], key=lambda z: z["seed"]):
                if not r.get("fit") or r.get("rho") is None:
                    print("  %-22s  (not readable)" % r["base"].split(".")[0])
                    continue
                print("  %-22s %7.2f %8.4f %9.5f %9.5f %9.5f %9.5f"
                      % (r["base"].split(".")[0], r["fit"]["tau"],
                         r["fit"]["D"], r["rho"][0.25], r["rho"][0.50],
                         r["rho"][0.75], r["rho"][0.90]))
        for a in SECONDARY_ARMS:
            v = [r for r in (by3.get(a) or []) if r.get("rho")]
            if len(v) >= 2 and a in by:
                print("  DESCRIPTIVE cross-batch rho(%s): cts3 %s | cpk2 %s"
                      % (a, "  ".join("%.4f" % statistics.mean(
                          [r["rho"][f] for r in v]) for f in F_GRID),
                         "  ".join("%.4f" % armrho(a, f) for f in F_GRID)
                         if a in eligible else "(not eligible)"))

    # ---- the verdict --------------------------------------------------------
    print("\n" + "=" * 78)
    if sep:
        v = "DIFFERENT-KIND"
        why = ("rho differs between %s and %s beyond the bar at f = %s"
               % (A_, B_, ", ".join("%.2f" % f for f in sep)))
    else:
        v = "SAME-KIND-DIFFERENT-RATE"
        why = ("no f separates %s from %s; the two approaches are "
               "time-rescalings of one another" % (A_, B_))
    sub = "NON-EXPONENTIAL" if offexp else "EXPONENTIAL"
    print("VERDICT (KIND vs RATE): %s" % v)
    print("  %s" % why)
    print("VERDICT (the shared form): %s" % sub)
    if offexp:
        print("  at least one arm departs from the exponential null at "
              "f = %s" % ", ".join("%.2f" % f for f in offexp))
        print("  This does NOT change the KIND verdict: two arms can share a")
        print("  shape that is not exponential.")
    print("FINAL: %s | %s" % (v, sub))
    print("=" * 78)
    print("WHAT THIS DOES *NOT* SAY: no mechanism, no repaired rate law, no")
    print("asymptote, nothing pooled across batches, nothing about any other")
    print("cut, cell or horizon, and nothing about which tensor carries what.")
    return 0


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="cS1 trajectory shape scorer")
    ap.add_argument("runsdir", nargs="?",
                    help="directory holding the cpk2-*.out files (PRIMARY)")
    ap.add_argument("--cts3", dest="cts3dir", default=None,
                    help="directory holding the cts3-*.out files "
                         "(SECONDARY, descriptive only).  A DOCUMENTED "
                         "argument: passing it is not an edit.")
    ap.add_argument("--selftest", action="store_true",
                    help="re-derive every registered constant on synthetic "
                         "data and the corpus, and exit.  Opens no cpk2 or "
                         "cts3 .out file.")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.runsdir:
        ap.error("runsdir is required unless --selftest is given")
    return score(a.runsdir, a.cts3dir)


if __name__ == "__main__":
    sys.exit(main())
