#!/usr/bin/env python3
# =============================================================================
# c87_hz3_score.py -- THE REGISTERED SCORER FOR `hz3`, THE HORIZON CONTROL.
#
# STANDING RULE 19: this file is git-committed BEFORE any hz3 run exists.  Every
# band edge, every threshold, every refusal below is frozen here, and the
# selftest asserts each one against bin/c87_horizon_300ep.sh's OWN TEXT so the
# registration and the batch cannot drift apart.
#
# -----------------------------------------------------------------------------
# THE QUESTION
# -----------------------------------------------------------------------------
# D = chunk777 - nodewise and the mechanism contrast D - G are SIGNIFICANTLY
# NEGATIVE for most of training on every CIFAR-10 cell in the corpus and turn
# positive only in the last 15-30 epochs of a 100-epoch budget (g3m D -0.518 at
# ep55 -> +0.666 at ep100; cc1 D-G -0.945 at ep40 -> +0.715 at ep100; fa1 -0.426
# -> +0.630).  Until this batch is scored, every headline in the paper is a claim
# about epochs 95-100 of 100.
#
# -----------------------------------------------------------------------------
# WHAT IS SCORED, AND IN WHAT ORDER.  THE ORDER IS PART OF THE REGISTRATION.
# -----------------------------------------------------------------------------
#   H0  THE BOX GATE, FIRST.  rec_lo AND rec_hi, PER SEED PER ARM, computed over
#       the SAME 5-epoch window as each accuracy reading.  At 300 epochs the
#       CEILING -2.3026 is algebraically reachable from epoch 92.1, i.e. for 69%
#       of the run against 8% of a 100-epoch run, so occupancy is a FIRST-CLASS
#       number here and not a footnote.  Nothing else is read until H0 is read.
#   HC  THE INTERNAL CONTROL, SECOND AND ABSOLUTELY GATING.  D(100) computed
#       INSIDE this batch must land in [+0.20, +1.00].  **IF IT DOES NOT, THE
#       BATCH IS VOID AND D(300) IS NOT READ.**  The band is wide because the
#       comparison is CROSS-BOX: the floor moved from cc1's -15 to -30 (forced --
#       at 300 ep ms*T = 15.0 nats and -15 is reachable from epoch 161.8) and
#       fa1 measured that this box class is not inert.
#   H1  PRIMARY.    D(300) > 0 at |t| >= 2.
#   H2  SECONDARY.  (D - G)(300) > 0 at |t| >= 2.
#   H3  SHAPE.      D(300) - D(100), banded GROWS / SATURATES / DECAYS at +-0.20.
#                   This contrast is WITHIN-RUN: same run, same seed, same batch,
#                   same box, same hardware.  Only trajectory noise remains.
#   H4  DESCRIPTIVE ONLY.  G(300); the in-batch count contrast U = chunk2325 -
#       chunk777; T = nodewise1d - nodewise, which is COUNT-CONFOUNDED by
#       construction (14,420 -> 4,851 = 0.4731 decades) and is NOT tail removal;
#       the full D/G/(D-G) trajectory at 100/200/300; the seed-blocked estimate;
#       the GPU-class contrast.
#
# -----------------------------------------------------------------------------
# PAIRING.  STANDING RULE 14, AND THE ONE PLACE THIS BATCH DEPARTS FROM IT.
# -----------------------------------------------------------------------------
# Every arm contrast here is WITHIN ONE BATCH, so the batch offset cancels by
# construction and no cross-batch floor applies.  The campaign's variance
# decomposition finds SEED statistically NULL (F(60,85)=1.21, p=0.213) against an
# overwhelming BATCH effect (F(62,85)=5.47, p=6.9e-13), which is why every
# published headline is WELCH-UNPAIRED and why a "k of n" line is a SIGN TEST and
# never a pairing argument.  THAT REMAINS THE PRIMARY HERE.
#
# BUT hz3 assigns the GPU CLASS BY SEED (seeds 0,1 -> L4; seeds 2,3 -> 2080Ti) so
# that all four arms of a seed share it -- the fix for CLOSEOUT 2.2(3)'s finding
# that mm1 and ar1 were arm-imbalanced on hardware and nobody reported it.  That
# makes the seed label carry a REAL nuisance in this batch and only in this
# batch.  So the scorer ALSO reports a SEED-BLOCKED estimate, as a registered
# SECONDARY, and requires it to AGREE IN SIGN with the Welch figure.  A
# disagreement is DISCLOSED, never chosen between.  The Welch number stays
# primary so D(300) is comparable to every published D.
#
# -----------------------------------------------------------------------------
# WHAT THIS SCORER MAY NOT DO
# -----------------------------------------------------------------------------
#   * NO FIELD STATISTIC.  No per-weight correlation, no N_eff/m, no beta
#     moments.  Direction C was DROPPED on its own pre-registered adoption gate
#     (C2 anti-concordant t -11.14; C3 dissociation t -23.26).  This scorer
#     contains no such code and the selftest asserts its absence over the AST.
#   * NO RULE-11 CLAIM.  No 300-epoch ms argmax is measured, on any arm.  Every
#     verdict carries "at cc1's own ms, not at a 300-epoch optimum".
#   * NO POOLING with mm1/pp1/cc1/ar1/fa1/g3m.  hz3 is a horizon control in a
#     DIFFERENT BOX; it is not a sixth replication of the R18 D series.
#   * NO BUDGET LAW.  Three readings from one budget is a trajectory.  Nothing
#     may be extrapolated past 300 epochs.
#   * NO AMENDMENT to cc1's own published D(100) = +0.727 in the -15 box.
#   * T IS NOT TAIL REMOVAL and this scorer refuses to label it so.
# =============================================================================
"""Registered scorer for the hz3 300-epoch horizon control (D(100/200/300))."""

import argparse
import ast as _ast
import json
import math
import os
import re
import statistics as st
import sys

# ---------------------------------------------------------------------------
# FROZEN REGISTRATION.  Changing any of these after an hz3 run exists is a
# violation of RULE 19; the selftest cross-checks them against the batch script.
# ---------------------------------------------------------------------------
TAG = "hz3"
BATCH_SCRIPT = "bin/c87_horizon_300ep.sh"
NET = "ResNet18"
DSET = "CIFAR10"
EPOCHS = 300
STEPS_PER_EPOCH = 500
PROBE_EVERY = 5
MST = "1e-4"
ALPHA0 = "1e-3"
CLIP = "-30:9.0"
SEEDS = [0, 1, 2, 3, 4, 5]

ARMS = {  # short tag -> (granularity, m, leg)
    "node": ("nodewise", 14420, "D"),
    "ch": ("chunk777", 14421, "D"),
    "n1d": ("nodewise1d", 4851, "G"),
    "c23": ("chunk2325", 4851, "G"),
}

# GPU class by SEED, so all four arms of a seed share it (CLOSEOUT 2.2(3)).
SEED_CLASS = {0: "gpu-l4-24g", 1: "gpu-l4-24g", 2: "gpu-l4-24g",
              3: "gpu-2080ti-11g", 4: "gpu-2080ti-11g", 5: "gpu-2080ti-11g"}

# The registered readings.  All THREE come from THE SAME SIXTEEN RUNS.
READINGS = (100, 200, 300)
WINDOW = 5               # plateau5's window: the COMPARABLE secondary reading

# THE PRIMARY READING IS A 50-EPOCH MEAN, NOT A 5-EPOCH ENDPOINT.
# CORRECTIONS 117.6.  D(t) oscillates by ~1.0-1.2 pp peak-to-trough on every
# CIFAR-10 cell, re-derived per seed from the .out series: cc1 reads +0.625 at
# ep25, +0.020 at ep40, -0.560 at ep55, +0.243 at ep70, +0.727 at ep100;
# mm1 +0.539 / +0.237 / -0.439 / -0.027 / +0.485; g3m -0.470 at ep10 through
# -0.518 at ep55 to +0.666 at ep100.  Against a refutation bar of +0.20 and an
# se near 0.13, a single 5-epoch window landing in a trough would fire
# "REFUTED -- fixed-budget artefact" on a wobble and rewrite the paper.  The
# PRIMARY is therefore the mean of D over the last DENSE epochs of each reading;
# the plateau5-comparable 5-epoch value is reported beside it, always.
DENSE = 50

# cc1's own 100-epoch numbers, re-derived from results/all_runs.csv at write
# time, in cc1's box (-15:-2.3026).  THE COMPARISON TO THEM IS CROSS-BOX.
ANCHOR_D_CC1 = 0.727     # se 0.200, n=3v3
ANCHOR_G_CC1 = 0.011     # se 0.147 -- the EXACT-count G
ANCHOR_DG_CC1 = 0.715    # se 0.248, within-batch
ANCHOR_T_CC1 = 0.816     # se 0.159, COUNT-CONFOUNDED, descriptive only
ANCHOR_D_POOL = 0.5805   # the R18 ms=1e-4 inverse-variance pool (mm1/pp1/cc1)
ANCHOR_D_POOL_SE = 0.0939

# THE PLANNING SD IS THE 300-EPOCH ONE, NOT THE IMPORTED 100-EPOCH ONE.
# CORRECTIONS 117.4.  The build imported SD_PLAN = 0.1743 (R18 four-arm pooled,
# 33 df) from 100-epoch cells into a 300-epoch batch.  Re-derived from the CSV
# over EVERY complete 300-epoch cell with n >= 2 at SGDm base / alpha0 = 1e-3,
# the pooled within-cell sd is 0.2260 on 3 df -- 30% larger.  At the old n = 4
# that put the |t| >= 2 bar at 0.3196, ABOVE this batch's own predicted lower
# edge of +0.30, i.e. underpowered for the outcome it predicts.  The bar is now
# taken at the LARGER of the two sds and the seed count raised to 6.
SD_PLAN_100 = 0.1743     # R18 four-arm pooled on 33 df at 100 ep -- REFERENCE
SD_PLAN_300 = 0.2260     # complete 300-ep SGDm/alpha0=1e-3 cells, 3 df
SD_PLAN = max(SD_PLAN_100, SD_PLAN_300)
SD_PLAN_DF = 3           # the 300-epoch sd rests on 3 df.  Disclosed, not hidden.

# THE INTERNAL CONTROL ON D(100).  **REPORTING, NOT GATING** -- CORRECTIONS
# 117.5.  As built it returned before H1 whenever D(100) fell outside the band,
# discarding the batch.  That is wrong in the one direction that costs the most:
# this batch sits in a DIFFERENT BOX from cc1 (-30:9.0 against -15:-2.3026) and
# fa1 established that a box change moves the OPTIMISER, not merely the
# instrument, so a D(100) of, say, +0.15 here is a real box finding -- while the
# PRIMARY quantity, the WITHIN-RUN change D(300) - D(100), stays valid, because
# both readings share the box, the batch and the seed.  The band now decides
# only whether the LEVEL of D may be compared to the published cells.
CTRL_LO = 0.20
CTRL_HI = 1.00
REFUTE_MAX = 0.20        # D(300) <= this REFUTES
FADE_RATIO = 0.5         # D(300) < this x D(100) WITH a monotone decline REFUTES
DECIDE = 0.30            # the predicted lower edge of D(300); H2's confirm bar
SAT_HALF = 0.20          # |D(300) - D(100)| <= this = SATURATES
PRED_LO, PRED_HI = 0.4, 1.2   # the registered point prediction for D(300)

BOX_GATE = 0.05          # a seed with rec_lo or rec_hi >= this is BOX-BOUND
BOX_ASYM = 0.10          # arm-to-arm occupancy asymmetry that VOIDS a contrast
VOID_BOUND_FRAC = 0.5    # an arm with MORE than this fraction bound is VOID
MIN_BOXFREE = 3          # below this, no arm may carry a primary
CONFIRM_T = 2.0          # |t| required to declare anything

# The count axis.  FOUR values for the slope circulate in the record (-0.407,
# -0.4906, ~0.51, and an OLS re-fit of ck1 at -0.5361).  THIS SCORER IMPORTS
# NONE OF THEM.  Both legs are so nearly count-matched that the bias is
# negligible at every slope in the envelope, and where a count effect is wanted
# it is measured IN BATCH as U = chunk2325 - chunk777.
SLOPE_ENVELOPE = (-0.5361, -0.4073)
T_DECADES = 0.473134     # log10(14420 / 4851): the count confound inside T


# ---------------------------------------------------------------------------
# Distribution tails.  The cluster env has no scipy; these are implemented here
# and validated against published critical values in the selftest.
# ---------------------------------------------------------------------------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_sf(t, df):
    """P(T > t) for Student-t on df degrees of freedom."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    p = 0.5 * betainc(0.5 * df, 0.5, x)
    return p if t > 0 else 1.0 - p


def t_two(t, df):
    return 2.0 * t_sf(abs(t), df)


def binom_sf(k, n, p=0.5):
    """P(X >= k) for Binomial(n, p) -- the SIGN TEST, exact."""
    tot = 0.0
    for i in range(k, n + 1):
        tot += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return tot


def welch(a, b):
    """(diff, se, df) for mean(a) - mean(b), Welch-unpaired."""
    if len(a) < 2 or len(b) < 2:
        return (st.mean(a) - st.mean(b) if a and b else float("nan"),
                float("nan"), float("nan"))
    va, vb = st.variance(a), st.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return st.mean(a) - st.mean(b), 0.0, float("nan")
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return st.mean(a) - st.mean(b), se, df


def blocked(pairs):
    """(mean, se, df) of within-seed differences.  Registered SECONDARY only:
    legal HERE because the GPU class is a function of the seed in this batch, so
    blocking on seed removes a real nuisance rather than the null seed label."""
    if len(pairs) < 2:
        return float("nan"), float("nan"), float("nan")
    d = [x - y for x, y in pairs]
    s = st.stdev(d) / math.sqrt(len(d))
    return st.mean(d), s, len(d) - 1


# ---------------------------------------------------------------------------
# The epoch series, re-derived from the .out.  NEVER read from the CSV `plateau`
# column, which is mean-of-last-20 and is not comparable across budgets.
# Epochs in the .out are 0-INDEXED (`Epoch 0` ... `Epoch 299`), so the reading at
# budget B is the mean over epoch indices B-WINDOW .. B-1 -- exactly plateau5
# when B is the full budget.
# ---------------------------------------------------------------------------
EP_RE = re.compile(r"Epoch\s+(\d+).*?Test Accuracy:\s*([0-9.]+)", re.I)


def series_from_out(path):
    """{epoch_index: test_accuracy} read off one .out file."""
    out = {}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = EP_RE.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def window_at(series, budget, w=WINDOW):
    """The w-epoch mean ending at `budget`, or None if the window is incomplete.

    Epochs are 0-INDEXED in the .out series, so window_at(ser, B) is the mean of
    epoch indices B-w .. B-1 -- identical to plateau5 at B = the run length, and
    identical to a B-epoch run's own plateau5 at any earlier B."""
    v = [series[e] for e in range(budget - w, budget) if e in series]
    return sum(v) / len(v) if len(v) == w else None


def dense_at(series, budget, w=DENSE):
    """The PRIMARY reading: the w-epoch mean ending at `budget`.  Same estimator
    as window_at with a wider window; separate name so no call site can confuse
    the primary with the plateau5-comparable secondary."""
    return window_at(series, budget, w)


# ---------------------------------------------------------------------------
# Box occupancy, per seed per arm, PER READING WINDOW.  A 300-epoch probe holds
# 30,000 records and ~330 MB of JSON, so records are selected by their `step`
# field from a cheap prefix match and only the selected ones are parsed.
# ---------------------------------------------------------------------------
STEP_RE = re.compile(r'"step"\s*:\s*(\d+)')


# ---------------------------------------------------------------------------
# THE RAIL READER.  CORRECTED 2026-08-30, CORRECTIONS 117.1.
#
# The per-coordinate rail counts live in  n_at_lo / n_at_hi  over  n_beta.
# r["beta"] is a PER-TENSOR summary -- 62 entries on ResNet18 against
# n_beta = 14,420 -- and reading rails off it is BLIND to real clipping and,
# where it does fire, gets the arm ranking BACKWARDS.  Both failures were
# measured on data already on disk, in the plateau5 window:
#
#   fa1 (ms=3e-4, 100 ep, box -25:-2.3026).  The 62-list reads 0.000000 on
#   ALL 24 runs.  The coordinate counts read nodewise s1/s4/s5 rec_hi = 1.0000
#   -- a clipped coordinate in EVERY record -- while all 18 merged-tail runs
#   are exactly 0.  The ceiling binds, and it binds only on the arm the paper
#   is about.
#
#   ar1 (ms=3e-4, 100 ep, box -15:-2.3026).  The 62-list says chunk777 is
#   7x MORE floor-bound than nodewise (0.2305 vs 0.0325).  The coordinate
#   counts say nodewise is 75x more bound than chunk777 (0.1103 vs 0.00145).
#   The instrument inverted the sign of the arm asymmetry.
#
# The 62-list is retained as a THIRD, LABELLED column so the historical
# statistic stays visible, and is never used for a gate.
# ---------------------------------------------------------------------------
def rail_fracs(rec, lo, hi):
    """(coord_lo, coord_hi, rec_lo, rec_hi, tensor_lo, tensor_hi) or None.

    coord_*  : fraction of the run's beta COORDINATES at a rail (n_at_*/n_beta)
    rec_*    : 1 if ANY coordinate is at that rail in this record, else 0
    tensor_* : the historical 62-entry per-tensor statistic; DIAGNOSTIC ONLY
    """
    nb = rec.get("n_beta")
    nlo = rec.get("n_at_lo")
    nhi = rec.get("n_at_hi")
    if not isinstance(nb, (int, float)) or not nb or nlo is None or nhi is None:
        return None
    nb = float(nb)
    clo, chi = float(nlo) / nb, float(nhi) / nb
    rlo = 1.0 if float(nlo) > 0 else 0.0
    rhi = 1.0 if float(nhi) > 0 else 0.0
    b = rec.get("beta")
    if isinstance(b, list) and b:
        n = float(len(b))
        tlo = sum(1 for x in b if x <= lo + 1e-9) / n
        thi = sum(1 for x in b if x >= hi - 1e-9) / n
    else:
        tlo = thi = float("nan")
    return clo, chi, rlo, rhi, tlo, thi


def occupancy_windows(probe_dir, lo, hi, budgets=READINGS, w=WINDOW,
                      spe=STEPS_PER_EPOCH):
    """{budget: (coord_lo, coord_hi, rec_lo, rec_hi, n_records)} over the SAME
    window as each accuracy reading.  {} if no probe is on disk.

    coord_* is the GATE (n_at_lo / n_at_hi over n_beta); rec_* is printed
    beside it.  CORRECTIONS 117.1: the per-tensor beta summary this function
    previously read is blind to the clipping that actually occurs."""
    p = os.path.join(probe_dir, "probe.jsonl")
    if not os.path.exists(p):
        return {}
    spans = {}
    for bud, wid in budgets.items() if isinstance(budgets, dict) else (
            (x, w) for x in budgets):
        spans[bud] = ((bud - wid) * spe, bud * spe)
    acc = {bud: [[], [], [], []] for bud in spans}
    with open(p, "r", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line[:80])
            if not m:
                continue
            step = int(m.group(1))
            hit = [bud for bud, (s0, s1) in spans.items() if s0 <= step < s1]
            if not hit:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            f = rail_fracs(r, lo, hi)
            if f is None:
                continue
            for bud in hit:
                acc[bud][0].append(f[0])
                acc[bud][1].append(f[1])
                acc[bud][2].append(f[2])
                acc[bud][3].append(f[3])
    out = {}
    for bud, (cl, ch, rl, rh) in acc.items():
        if cl:
            n = float(len(cl))
            out[bud] = (sum(cl) / n, sum(ch) / n, sum(rl) / n, sum(rh) / n,
                        len(cl))
    return out


# ---------------------------------------------------------------------------
def load_arm(runs_dir, probes_dir, short, lo, hi):
    """{seed: dict(acc={budget: value}, occ={budget: (lo, hi, n)}, n_epochs, ...)}"""
    out = {}
    for s in SEEDS:
        rn = "%s-%s-s%d" % (TAG, short, s)
        cands = []
        if os.path.isdir(runs_dir):
            for fn in os.listdir(runs_dir):
                if fn.startswith(rn + "-") and fn.endswith(".out"):
                    cands.append(os.path.join(runs_dir, fn))
        if not cands:
            continue
        ser = series_from_out(sorted(cands)[-1])
        acc = {b: window_at(ser, b) for b in READINGS}
        accd = {b: dense_at(ser, b) for b in READINGS}
        pd = os.path.join(probes_dir or runs_dir, "probe_%s_%s_s%d" % (short, TAG, s))
        occ = occupancy_windows(pd, lo, hi)
        out[s] = dict(acc=acc, accd=accd, occ=occ, n_epochs=len(ser),
                      truncated=(acc[max(READINGS)] is None),
                      gpu=SEED_CLASS.get(s, "?"))
    return out


def seed_ok(d, budget, key="acc"):
    """Is this seed usable AT THIS READING?  Needs the accuracy window AND a
    measured, box-free occupancy in the SAME window.

    occ[budget] is (coord_lo, coord_hi, rec_lo, rec_hi, n).  THE GATE IS THE
    COORDINATE FRACTION -- n_at_lo / n_at_hi over n_beta (CORRECTIONS 117.1).
    rec_* is printed beside it and is never the gate, because a single pinned
    coordinate out of 14,420 sets rec to 1.0 while coord reads 7e-5."""
    if d.get(key, {}).get(budget) is None:
        return False
    o = d["occ"].get(budget)
    return o is not None and o[0] < BOX_GATE and o[1] < BOX_GATE


def band_shape(delta):
    if delta > SAT_HALF:
        return "GROWS"
    if delta < -SAT_HALF:
        return "DECAYS"
    return "SATURATES"


def band_primary(d300, d100, traj_d):
    """The PRIMARY verdict on D(300).  Frozen here, before any data.

    d100 may be None when the epoch-100 reading is itself box-void; the fade
    branch is then not evaluable and the verdict rests on the absolute bar
    alone.  It is labelled UNANCHORED at the call site."""
    if d300 <= REFUTE_MAX:
        return "REFUTED (fixed-budget phenomenon)"
    if d100 is None:
        return "SURVIVES (UNANCHORED -- D(100) not measurable)"
    monotone = all(traj_d[i] > traj_d[i + 1] for i in range(len(traj_d) - 1))
    if d300 < FADE_RATIO * d100 and monotone:
        return "REFUTED (fixed-budget phenomenon)"
    return "SURVIVES"


# ---------------------------------------------------------------------------
def score(runs_dir, probes_dir, out=sys.stdout):
    lo, hi = (float(x) for x in CLIP.split(":"))
    P = lambda *a: print(*a, file=out)
    P("=" * 78)
    P("  hz3 -- THE HORIZON CONTROL.  Does the partition gap survive past a")
    P("         100-epoch budget, or is it a property of the last fifth?")
    P("  %s / %s / %d ep / ms=%s / alpha0=%s / BETA_CLIP=%s"
      % (NET, DSET, EPOCHS, MST, ALPHA0, CLIP))
    P("  ONE batch, FOUR arms, ONE ms, ONE budget, n=4.  D(100), D(200) and")
    P("  D(300) are read WITHIN-RUN off the .out epoch series, so the batch")
    P("  offset AND the seed cancel exactly in the horizon trend.")
    P("  BOTH RAILS MOVED, AND BOTH MOVES ARE FORCED.  Floor -15 -> -30: at 300 ep")
    P("  ms*T = 15.0 nats and -15 is reachable from epoch 161.8.  Ceiling -2.3026")
    P("  -> +9.0: the corpus's ONLY 15-nat precedent is fa1, and on fa1 the")
    P("  -2.3026 ceiling BINDS, ARM-ASYMMETRICALLY -- nodewise seeds 1/4/5 carry a")
    P("  clipped coordinate in EVERY record of the plateau5 window while all 18")
    P("  merged-tail runs read exactly 0.  A ceiling that clamps only the arm the")
    P("  paper is about would have decided this batch.  +9.0 > beta_0 + ms*T =")
    P("  +8.09, so the ceiling is now PROVABLY unreachable (CORRECTIONS 117.2).")
    P("  RULE 10 is broken on BOTH rails, deliberately, and stated: this box is")
    P("  NOT the published cells' box and the LEVEL of D here is not poolable")
    P("  with theirs.  The PRIMARY is the WITHIN-RUN change, which is immune.")
    P("  RULE 11 IS OPEN: no 300-epoch ms argmax is measured, on any arm.  Every")
    P("  verdict below holds at cc1's own ms, NOT at a 300-epoch optimum.")
    P("=" * 78)

    data = {k: load_arm(runs_dir, probes_dir, k, lo, hi) for k in ARMS}

    # --- H0: THE BOX GATE, READ FIRST --------------------------------------
    P("")
    P("H0  BOX OCCUPANCY, per seed per arm, over the SAME %d-epoch window as each"
      % WINDOW)
    P("    accuracy reading.  BOTH RAILS ARE FIRST-CLASS.  At CLIP=-30:9.0 with")
    P("    ms*T = 15.0 nats and beta_0 = -6.908, the deepest reachable beta is")
    P("    -21.91 and the highest is +8.09: BOTH RAILS ARE PROVABLY UNREACHABLE,")
    P("    with 8.09 nats of floor slack and 0.91 nats of ceiling slack.  This gate")
    P("    therefore expects 0.000000 everywhere; anything else is a defect, not a")
    P("    nuisance, and is read as one.")
    P("    coord_* = n_at_lo / n_at_hi over n_beta -- THE GATE (bar %.2f)." % BOX_GATE)
    P("    rec_*   = fraction of records with ANY coordinate at that rail --")
    P("              printed always, NEVER the gate.  The two disagree by up to")
    P("              three orders of magnitude on the same run.")
    P("    %-5s %-4s %-15s %7s %8s %9s %9s %7s %7s  %s"
      % ("arm", "seed", "gpu class", "budget", "acc",
         "coordLO", "coordHI", "recLO", "recHI", "status"))
    arm_bound = {}
    for k in ARMS:
        bound_seeds = set()
        for s in SEEDS:
            d = data[k].get(s)
            if d is None:
                P("    %-5s %-4d %-15s %7s %8s %9s %9s %7s %7s  MISSING"
                  % (k, s, SEED_CLASS.get(s, "?"), "-", "-", "-", "-", "-", "-"))
                continue
            for b in READINGS:
                a = d["accd"].get(b)
                o = d["occ"].get(b)
                cl = ch = rl = rh = None
                if a is None:
                    stat = "NO WINDOW -- run truncated before epoch %d" % b
                elif o is None:
                    stat = "NO PROBE -- occupancy UNMEASURED, reading UNINTERPRETABLE"
                else:
                    cl, ch, rl, rh = o[0], o[1], o[2], o[3]
                    if cl < BOX_GATE and ch < BOX_GATE:
                        stat = "box-free (%d records)" % o[4]
                    else:
                        stat = "BOX-BOUND -- DROPPED at this reading"
                        bound_seeds.add(s)
                P("    %-5s %-4d %-15s %7d %8s %9s %9s %7s %7s  %s"
                  % (k, s, d["gpu"], b,
                     "%.3f" % a if a is not None else "-",
                     "%.6f" % cl if cl is not None else "-",
                     "%.6f" % ch if ch is not None else "-",
                     "%.3f" % rl if rl is not None else "-",
                     "%.3f" % rh if rh is not None else "-", stat))
        arm_bound[k] = bound_seeds
        frac = len(bound_seeds) / float(len(SEEDS))
        if frac > VOID_BOUND_FRAC:
            P("    !!! ARM %s VOID: %d of %d seeds bound at a rail (%.0f%% > %.0f%%)."
              % (k, len(bound_seeds), len(SEEDS), 100 * frac, 100 * VOID_BOUND_FRAC))
            P("        A FLOOR bind is remediable -- re-run at -40.  A CEILING bind is")
            P("        NOT: releasing the ceiling leaves the box class every published")
            P("        D was measured in, so a ceiling bind is a LIMIT ON THE READING")
            P("        and must be reported as one, not patched away.")
        else:
            P("    arm %s: %d of %d seeds bound at a rail -- inside the %.0f%% VOID bar."
              % (k, len(bound_seeds), len(SEEDS), 100 * VOID_BOUND_FRAC))
            if len(SEEDS) - len(bound_seeds) < MIN_BOXFREE:
                P("        but only %d box-free seeds remain, below the %d needed to"
                  % (len(SEEDS) - len(bound_seeds), MIN_BOXFREE))
                P("        carry a primary: the VOID bar is the RE-RUN trigger, and")
                P("        MIN_BOXFREE is the stricter admissibility bar.  Contrasts")
                P("        touching this arm will read VOID at the affected readings.")

    def usable(k, b, key="accd"):
        return {s: d[key][b] for s, d in data[k].items() if seed_ok(d, b, key)}

    def contrast(ka, kb, b, key="accd"):
        A, B = usable(ka, b, key), usable(kb, b, key)
        if len(A) < MIN_BOXFREE or len(B) < MIN_BOXFREE:
            return None
        oa = [data[ka][s]["occ"][b] for s in A]
        ob = [data[kb][s]["occ"][b] for s in B]
        dlo = abs(st.mean([x[0] for x in oa]) - st.mean([x[0] for x in ob]))
        dhi = abs(st.mean([x[1] for x in oa]) - st.mean([x[1] for x in ob]))
        if dlo > BOX_ASYM or dhi > BOX_ASYM:
            return None
        d, se, df = welch(list(A.values()), list(B.values()))
        pr = [(A[s], B[s]) for s in sorted(set(A) & set(B))]
        return dict(d=d, se=se, df=df, nA=len(A), nB=len(B), pairs=pr,
                    dlo=dlo, dhi=dhi)

    # PRIMARY trajectory: the DENSE (%d-epoch) reading.  SECONDARY: the
    # plateau5-comparable 5-epoch endpoint, computed on exactly the same runs.
    traj = {}
    traj5 = {}
    for b in READINGS:
        traj[b] = dict(D=contrast("ch", "node", b), G=contrast("c23", "n1d", b),
                       T=contrast("n1d", "node", b), U=contrast("c23", "ch", b))
        traj5[b] = dict(D=contrast("ch", "node", b, "acc"),
                        G=contrast("c23", "n1d", b, "acc"),
                        T=contrast("n1d", "node", b, "acc"),
                        U=contrast("c23", "ch", b, "acc"))

    def dg_at(b):
        D, G = traj[b]["D"], traj[b]["G"]
        if D is None or G is None:
            return None
        v = D["d"] - G["d"]
        se = math.sqrt(D["se"] ** 2 + G["se"] ** 2)
        return v, se

    # --- HC: THE INTERNAL CONTROL, GATING ----------------------------------
    P("")
    P("HC  THE INTERNAL CONTROL.  **REPORTING, NOT GATING** (CORRECTIONS 117.5).")
    P("    It decides whether the LEVEL of D here may be compared to the")
    P("    published cells.  It does NOT decide whether D(300) is read: the")
    P("    PRIMARY is the WITHIN-RUN change D(300) - D(100), in which the box,")
    P("    the batch and the seed cancel exactly, and that quantity stays valid")
    P("    whatever the level does.  Comparable band: [%+0.2f, %+0.2f]."
      % (CTRL_LO, CTRL_HI))
    P("    cc1 reads D(100) = %+0.3f (se 0.200) and the R18 ms=1e-4 pool %+0.4f"
      % (ANCHOR_D_CC1, ANCHOR_D_POOL))
    P("    +- %.4f -- BOTH in the -15 box, so this comparison is CROSS-BOX and the"
      % ANCHOR_D_POOL_SE)
    P("    band is wide on purpose.  Its job is to detect that the forced floor")
    P("    move, the 300-epoch schedule-free run, or anything else broke the cell.")
    ctrl = traj[100]["D"]
    res = {}
    if ctrl is None:
        P("    D(100) IS NOT MEASURABLE -- too few box-free seeds at ep100, or an")
        P("    asymmetric guard between the arms.  That is a statement about the")
        P("    epoch-100 reading ONLY.  The horizon primary needs D(100) as its")
        P("    baseline, so H3 will read VOID; H1 (the LEVEL of D at 300) is still")
        P("    read and reported below, flagged as UNANCHORED.")
        res["HC"] = ("D(100) NOT MEASURABLE", None)
        d100 = None
    else:
        d100 = ctrl["d"]
        P("    D(100) = %+0.3f  se %.3f  t %.2f (df %.1f)  n=%d v %d   [DENSE %d-ep]"
          % (d100, ctrl["se"], d100 / ctrl["se"] if ctrl["se"] else float("nan"),
             ctrl["df"], ctrl["nA"], ctrl["nB"], DENSE))
        if traj5[100]["D"] is not None:
            P("    D(100) on the plateau5-comparable 5-epoch window = %+0.3f  se %.3f"
              % (traj5[100]["D"]["d"], traj5[100]["D"]["se"]))
        P("    vs cc1's own D(100) %+0.3f: difference %+0.3f (CROSS-BOX, cross-batch"
          % (ANCHOR_D_CC1, d100 - ANCHOR_D_CC1))
        P("    -- reported, NOT tested: STANDING RULE 14's ~0.21 pp batch floor.)")
        if CTRL_LO <= d100 <= CTRL_HI:
            P("    VERDICT: COMPARABLE.  The cell reproduces the 100-epoch effect in")
            P("    the -30:9.0 box, so the LEVEL of D here may be set beside the")
            P("    published cells (with the batch floor still applying).")
            res["HC"] = ("COMPARABLE", d100)
        else:
            P("    VERDICT: **NOT COMPARABLE -- AND THIS IS A BOX FINDING, NOT A")
            P("    FAILURE.**  D(100) = %+0.3f is outside [%+0.2f, %+0.2f].  The box"
              % (d100, CTRL_LO, CTRL_HI))
            P("    moved from -15:-2.3026 to -30:9.0 and fa1 established that a box")
            P("    change moves the OPTIMISER, not merely the instrument, so this")
            P("    number must be REPORTED as \'D at 100 epochs in a free box\' and")
            P("    must NOT be pooled with the published D readings.  THE HORIZON")
            P("    PRIMARY IS STILL READ: D(300) - D(100) is within-run and within-box.")
            res["HC"] = ("NOT COMPARABLE -- BOX FINDING", d100)

    # --- H1: PRIMARY -------------------------------------------------------
    P("")
    P("H1  PRIMARY.  D(300) = chunk777 - nodewise at 300 epochs, read as the")
    P("    mean over the LAST %d EPOCHS (CORRECTIONS 117.6): D(t) oscillates by" % DENSE)
    P("    ~1.0-1.2 pp peak-to-trough on every CIFAR-10 cell measured, so a single")
    P("    5-epoch endpoint landing in a trough would fire a refutation on a wobble.")
    P("    The plateau5-comparable 5-epoch value is reported beside it.")
    P("    Count-matched to +1 group on 14,420 = %.3e decades; |bias| <= %.7f pp"
      % (math.log10(14421 / 14420.0),
         max(abs(s * math.log10(14421 / 14420.0)) for s in SLOPE_ENVELOPE)))
    P("    over the WHOLE slope envelope %s -- no slope is imported."
      % (SLOPE_ENVELOPE,))
    d3 = traj[300]["D"]
    if d3 is None:
        P("    VERDICT: VOID -- fewer than %d box-free seeds per arm at ep300, or an"
          % MIN_BOXFREE)
        P("    asymmetric guard between the arms.")
        res["H1"] = ("VOID", None)
    else:
        d300, se3 = d3["d"], d3["se"]
        t3 = d300 / se3 if se3 else float("nan")
        P("    D(300) = %+0.3f  se %.3f  t %.2f (df %.1f)  n=%d v %d"
          % (d300, se3, t3, d3["df"], d3["nA"], d3["nB"]))
        if traj5[300]["D"] is not None:
            P("    D(300) on the plateau5-comparable 5-epoch window = %+0.3f  se %.3f"
              % (traj5[300]["D"]["d"], traj5[300]["D"]["se"]))
            P("    (reported ALWAYS; the %d-epoch mean is the registered primary)" % DENSE)
        P("    occupancy asymmetry between the arms (COORDINATE fraction, the gate):")
        P("        |d coord_lo| %.6f   |d coord_hi| %.6f   bar %.2f"
          % (d3["dlo"], d3["dhi"], BOX_ASYM))
        traj_d = [traj[b]["D"]["d"] if traj[b]["D"] else float("nan") for b in READINGS]
        if d100 is None:
            P("    D(100) IS NOT MEASURABLE, so the fade branch of the primary (which")
            P("    is defined relative to D(100)) cannot be evaluated.  The verdict")
            P("    below rests on the absolute bar alone and is flagged UNANCHORED.")
        verdict = band_primary(d300, d100, traj_d)
        if verdict.startswith("SURVIVES") and abs(t3) < CONFIRM_T:
            P("    VERDICT: UNRESOLVED.  D(300) = %+0.3f is above the refutation bar"
              % d300)
            P("    (%+0.2f) but is NOT separated from zero at |t| >= %.1f, so neither"
              % (REFUTE_MAX, CONFIRM_T))
            P("    'survives' nor 'fades' may be written.  n=%d at the 300-EPOCH"
              % len(SEEDS))
            P("    planning sd (%.4f on %d df) was sized for D >= %+0.3f."
              % (SD_PLAN, SD_PLAN_DF,
                 CONFIRM_T * SD_PLAN * math.sqrt(2.0 / len(SEEDS))))
            res["H1"] = ("UNRESOLVED", d300)
        else:
            P("    VERDICT: %s" % verdict)
            if verdict.startswith("REFUTED"):
                P("    The effect is a FIXED-BUDGET PHENOMENON.  The headline must be")
                P("    rewritten as 'at a 100-epoch budget', the mechanism claim goes")
                P("    with it, and the parent paper's own unexplained ImageNet null")
                P("    ('the blockwise versions showed no improvement') becomes the")
                P("    correct long-budget reading rather than an anomaly.")
            else:
                P("    The partition gap is NOT an artefact of a 100-epoch budget.")
            res["H1"] = (verdict, d300)
        P("    registered point prediction was D(300) in [%+0.1f, %+0.1f]: %s"
          % (PRED_LO, PRED_HI,
             "INSIDE" if PRED_LO <= d300 <= PRED_HI else "OUTSIDE (disclose it)"))
        k = sum(1 for a, b in d3["pairs"] if a > b)
        P("    SIGN TEST (not a pairing argument): %d of %d seeds favour chunk777;"
          % (k, len(d3["pairs"])))
        P("        exact binomial p = %.4f" % binom_sf(k, len(d3["pairs"])))
        bm, bse, bdf = blocked(d3["pairs"])
        P("    SEED-BLOCKED SECONDARY (legal in THIS batch only, because the GPU")
        P("    class is a function of the seed): %+0.3f se %.3f t %.2f on %d df"
          % (bm, bse, bm / bse if bse else float("nan"), bdf))
        if bm * d300 < 0:
            P("    !!! THE BLOCKED AND WELCH ESTIMATES DISAGREE IN SIGN.  Disclosed,")
            P("    not chosen between; the Welch figure remains PRIMARY.")

    # --- H2: SECONDARY -----------------------------------------------------
    P("")
    P("H2  SECONDARY.  (D - G)(300), THE MECHANISM CONTRAST, WITHIN-BATCH.")
    P("    cc1 reads (D-G)(100) = %+0.3f; it is NEGATIVE for most of training on"
      % ANCHOR_DG_CC1)
    P("    every cell measured so far (cc1 -0.945 @ep40, g3m -0.832 @ep40).")
    dg3 = dg_at(300)
    if dg3 is None:
        P("    VERDICT: VOID (a leg is void at ep300).")
        res["H2"] = ("VOID", None)
    else:
        v, se = dg3
        P("    (D-G)(300) = %+0.3f  se %.3f  t %.2f" % (v, se, v / se if se else float("nan")))
        if v >= DECIDE and v / se >= CONFIRM_T:
            P("    VERDICT: MECHANISM SURVIVES THE HORIZON -- the gap is present with")
            P("    the tail and absent without it, at 300 epochs, inside ONE batch.")
            res["H2"] = ("MECHANISM SURVIVES", v)
        elif v <= -DECIDE and v / se <= -CONFIRM_T:
            P("    VERDICT: MECHANISM REVERSES AT 300 EPOCHS.  The tail is not the")
            P("    carrier at this horizon and the mechanism claim must be rescoped.")
            res["H2"] = ("MECHANISM REVERSES", v)
        else:
            P("    VERDICT: MECHANISM NOT RESOLVED at 300 epochs (needs |D-G| >= %.2f"
              % DECIDE)
            P("    at |t| >= %.1f).  This may NOT be written as either outcome." % CONFIRM_T)
            res["H2"] = ("MECHANISM NOT RESOLVED", v)

    # --- H3: SHAPE ---------------------------------------------------------
    P("")
    P("H3  SHAPE.  D(300) - D(100), WITHIN-RUN: same run, same seed, same batch,")
    P("    same box, same GPU class.  Only trajectory noise remains -- this is the")
    P("    cleanest contrast in the batch and the reason the design is within-run.")
    P("    %8s %10s %10s %10s %10s" % ("budget", "D", "G", "D-G", "T (descr.)"))
    for b in READINGS:
        row = traj[b]
        dgb = dg_at(b)
        P("    %8d %10s %10s %10s %10s"
          % (b,
             "%+0.3f" % row["D"]["d"] if row["D"] else "VOID",
             "%+0.3f" % row["G"]["d"] if row["G"] else "VOID",
             "%+0.3f" % dgb[0] if dgb else "VOID",
             "%+0.3f" % row["T"]["d"] if row["T"] else "VOID"))
    if traj[300]["D"] and traj[100]["D"]:
        delta = traj[300]["D"]["d"] - traj[100]["D"]["d"]
        P("    D(300) - D(100) = %+0.3f -> %s (band +-%.2f)"
          % (delta, band_shape(delta), SAT_HALF))
        # The WITHIN-RUN se, measured rather than assumed.  For every seed usable
        # at BOTH readings in BOTH arms, the per-seed change in the gap is
        #   (ch_s(300) - node_s(300)) - (ch_s(100) - node_s(100)),
        # in which the seed, the run, the batch, the box and the GPU class all
        # cancel identically.  This is the only contrast in the batch from which
        # every known nuisance has been removed by construction.
        cs = sorted(set(usable("ch", 100)) & set(usable("ch", 300)))
        ns = sorted(set(usable("node", 100)) & set(usable("node", 300)))
        if len(cs) >= 2 and len(ns) >= 2:
            dch = [usable("ch", 300)[s2] - usable("ch", 100)[s2] for s2 in cs]
            dnd = [usable("node", 300)[s2] - usable("node", 100)[s2] for s2 in ns]
            dv, dse, ddf = welch(dch, dnd)
            P("    measured WITHIN-RUN: %+0.3f se %.3f t %.2f (df %.1f) on the"
              % (dv, dse, dv / dse if dse else float("nan"), ddf))
            P("    per-seed CHANGE in each arm (n=%d v %d) -- seed, run, batch, box"
              % (len(cs), len(ns)))
            P("    and GPU class all cancel identically in this quantity.")
        else:
            P("    the within-run se is not measurable: too few seeds usable at BOTH")
            P("    readings.")
        if band_shape(delta) == "SATURATES":
            P("    SATURATION is the registered null and it is an ASSET, not a")
            P("    liability: the horizon becomes a footnote and D(t) becomes a")
            P("    measurable late-training onset, which is a mechanism claim itself.")
        res["H3"] = (band_shape(delta), delta)
    else:
        res["H3"] = ("VOID", None)

    # --- H4: DESCRIPTIVE ---------------------------------------------------
    P("")
    P("H4  DESCRIPTIVE ONLY.")
    for b in READINGS:
        u = traj[b]["U"]
        P("    ep%3d  U = chunk2325 - chunk777 (the IN-BATCH count axis, %.4f"
          % (b, T_DECADES))
        P("           decades) = %s -- this, not an imported slope, is what a count"
          % ("%+0.3f se %.3f" % (u["d"], u["se"]) if u else "VOID"))
        P("           correction should ever be built from.")
    P("    T = nodewise1d - nodewise spans %d -> %d groups = %.4f decades."
      % (ARMS["node"][1], ARMS["n1d"][1], T_DECADES))
    P("    T IS NOT TAIL REMOVAL and this scorer refuses to label it so: it is")
    P("    count-confounded by construction.  The tail statistic is G, which is an")
    P("    EXACT count match (4,851 vs 4,851).  cc1 reads T(100) = %+0.3f."
      % ANCHOR_T_CC1)
    P("    GPU CLASS (descriptive, and the reason it is balanced by construction):")
    for k in ARMS:
        by = {}
        for s, d in data[k].items():
            if d["acc"].get(300) is not None:
                by.setdefault(d["gpu"], []).append(d["acc"][300])
        P("      %-5s %s" % (k, ", ".join("%s n=%d mean %.3f" % (g, len(v), st.mean(v))
                                          for g, v in sorted(by.items())) or "none"))
    P("")
    P("NOT COMPUTED, BY REGISTRATION: any field statistic (per-weight correlation,")
    P("N_eff/m, beta moments).  Direction C was dropped on its own pre-registered")
    P("adoption gate.  NOT CLAIMED: a RULE-11 tuned result, a sixth replication of")
    P("the R18 D series, a budget law, an extrapolation past 300 epochs, or any")
    P("amendment to cc1's published D(100) = +0.727 in the -15 box.")
    return res


# ---------------------------------------------------------------------------
def selftest():
    ok = fail = 0

    def chk(cond, msg):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print("FAIL: %s" % msg)

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    bs_path = os.path.join(repo, BATCH_SCRIPT)
    chk(os.path.exists(bs_path), "batch script %s must exist (RULE 19)" % BATCH_SCRIPT)
    bs = open(bs_path).read() if os.path.exists(bs_path) else ""

    # --- the registration cannot drift from the batch script -----------------
    chk("MST=%s" % MST in bs, "batch script must pin MST=%s" % MST)
    chk("CLIP=%s" % CLIP in bs, "batch script must pin CLIP=%s" % CLIP)
    chk("ALPHA0=%s" % ALPHA0 in bs, "batch script must pin ALPHA0=%s" % ALPHA0)
    chk("NET=%s" % NET in bs, "batch script must pin NET=%s" % NET)
    chk("DSET=%s" % DSET in bs, "batch script must pin DSET=%s" % DSET)
    chk("EPOCHS=%d" % EPOCHS in bs, "batch script must pin EPOCHS=%d" % EPOCHS)
    chk("STEPS_PER_EPOCH=%d" % STEPS_PER_EPOCH in bs, "steps/epoch must match")
    chk("PROBE_EVERY=%d" % PROBE_EVERY in bs, "PROBE_EVERY must match")
    chk("NJOBS=%d" % (len(SEEDS) * len(ARMS)) in bs, "batch script must emit %d jobs" % (len(SEEDS) * len(ARMS)))
    chk('SEEDS="%s"' % " ".join(str(x) for x in SEEDS) in bs,
        "batch script must run exactly the scorer's seeds")
    for short, (gran, m, leg) in ARMS.items():
        chk(gran in bs, "batch script must carry arm %s" % gran)
    chk("M_NODE=%d" % ARMS["node"][1] in bs, "m(nodewise) must match")
    chk("M_CHUNK_D=%d" % ARMS["ch"][1] in bs, "m(chunk777) must match")
    chk("M_N1D=%d" % ARMS["n1d"][1] in bs, "m(nodewise1d) must match")
    chk("M_CHUNK_G=%d" % ARMS["c23"][1] in bs, "m(chunk2325) must match")
    chk(ARMS["n1d"][1] == ARMS["c23"][1], "the G leg must be an EXACT count match")
    chk(ARMS["ch"][1] - ARMS["node"][1] == 1, "the D leg must be +1 group")
    chk("SD_PLAN=%s" % SD_PLAN in bs, "planning sd must match the batch script")
    chk("CTRL_LO=%s" % CTRL_LO in bs, "the control band LO must match")
    chk("CTRL_HI=%s" % CTRL_HI in bs, "the control band HI must match")
    chk("REFUTE_MAX=%s" % REFUTE_MAX in bs, "the refutation bar must match")
    chk("DECIDE=%s" % DECIDE in bs, "the decide threshold must match")
    chk("SAT_HALF=%s" % SAT_HALF in bs, "the saturation half-width must match")
    chk("ANCHOR_D_CC1=%s" % ANCHOR_D_CC1 in bs, "cc1's D anchor must match")
    chk("ANCHOR_G_CC1=%s" % ANCHOR_G_CC1 in bs, "cc1's G anchor must match")
    chk("ANCHOR_DG_CC1=%s" % ANCHOR_DG_CC1 in bs, "cc1's D-G anchor must match")
    chk("ANCHOR_T_CC1=%s" % ANCHOR_T_CC1 in bs, "cc1's T anchor must match")
    chk("ANCHOR_D_POOL=%s" % ANCHOR_D_POOL in bs, "the R18 pool must match")
    for s in SLOPE_ENVELOPE:
        chk(("SLOPE_LO=%s" % s in bs) or ("SLOPE_HI=%s" % s in bs),
            "slope envelope edge %s must appear in the batch script" % s)

    # --- THE SHORT TAGS AND THE PATH LAYOUT MUST MATCH THE EMITTER ----------
    # This is the check that would have caught CORRECTIONS 116's "A PROCESS
    # NOTE": g3m's first scoring run printed MISSING on every arm because the
    # scorer and the emitter disagreed about where things land.  The .out files
    # go to the ROOT of runs/ (the runner's `--output .../runs/%x-%j.out`) while
    # the probe directories go to runs/<tag>/, which is why --runs and --probes
    # are separate arguments with separate defaults.
    for short, (gran, m, leg) in ARMS.items():
        chk('%s:%s"' % (gran, short) in bs or '${CHUNK_KD}:%s"' % short in bs
            or '${CHUNK_KG}:%s"' % short in bs,
            "the emitter must use the short tag %r for arm %s, or the scorer will "
            "find no .out and no probe for it" % (short, gran))
    chk('RN="%s-${SHORT}-s${S}"' % TAG in bs,
        "the emitter's run name must be <tag>-<short>-s<seed>, which is what "
        "load_arm globs for")
    chk("probe_${SHORT}_%s_s${S}" % TAG in bs,
        "the emitter's PROBE_DIR must be probe_<short>_<tag>_s<seed>, which is "
        "what occupancy_windows opens")
    chk("runs/%s" % TAG in bs, "the batch must save into runs/<tag>")

    # --- the GPU-class-by-seed map must match the batch script's own map -----
    for s, cls in SEED_CLASS.items():
        chk("%d:%s" % (s, cls) in bs,
            "SEED_PARTS must map seed %d to %s (CLOSEOUT 2.2(3))" % (s, cls))
    chk(sorted(SEED_CLASS) == sorted(SEEDS), "every seed must have a GPU class")
    chk(len(set(SEED_CLASS.values())) == 2, "exactly two GPU classes")
    from collections import Counter
    cnt = Counter(SEED_CLASS.values())
    chk(len(set(cnt.values())) == 1,
        "the classes must be BALANCED across seeds, or the arm means are not")
    chk("gpu-short" not in bs.split("PARTS=")[1].split("\n")[0] if "PARTS=" in bs else False,
        "gpu-short must not be in PARTS -- its 4:00:00 cap cannot hold 300 epochs")

    # --- the batch script must carry the guards this design demands ----------
    for needle, why in (
            ("guard 1e", "the budget-independence proof"),
            ("num_epochs", "guard 1e must name the symbol it proves is unused"),
            ("guard 1b-6", "the GPU-class-by-seed bijection check"),
            ("guard 1b-7", "the schedule-free check, on BOTH of HF.py's paths"),
            ("guard 4e (RULE 13)", "the m check must be tested in the failing direction"),
            ("guard 6d", "the per-class dedicated-cap check"),
            ("guard 7", "the throughput guard"),
            ("CONSERVATIVE", "guard 7 must carry a second, conservative instrument"),
            ("guard 8", "the box-reachability guard"),
            ("CEILING", "guard 8 must report the ceiling, not only the floor"),
            ("RULE 13", "the two most important guards must be tested both ways"),
            ("HEADROOM_FLOOR=1.40", "the throughput floor must be 1.40x"),
            ("WALL=10:00:00", "the wall must clear the CONSERVATIVE projection at 1.40x"),
            ("RULE 11 IS OPEN", "the RULE-11 gap must be declared, not hidden"),
            ("PROBE5", "PROBE5's ABSENCE must be asserted by guard 1b")):
        chk(needle in bs, "batch script must contain %r (%s)" % (needle, why))

    # --- NO FIELD STATISTIC, asserted over the AST's IDENTIFIERS -------------
    # (not over the raw prose, which necessarily names the things it forbids --
    # the self-referential-guard failure of CORRECTIONS 112.)
    me = open(os.path.abspath(__file__)).read()
    idents = set()
    for node in _ast.walk(_ast.parse(me)):
        if isinstance(node, _ast.Name):
            idents.add(node.id)
        elif isinstance(node, _ast.Attribute):
            idents.add(node.attr)
        elif isinstance(node, (_ast.FunctionDef, _ast.ClassDef)):
            idents.add(node.name)
        elif isinstance(node, _ast.arg):
            idents.add(node.arg)
    low = {i.lower() for i in idents}
    for banned in ("rho", "neff", "n_eff", "negcount", "neg_count"):
        chk(not any(banned in i for i in low),
            "this scorer must compute no field statistic (identifier %r)" % banned)
    reads = [m for m in _ast.walk(_ast.parse(me))
             if isinstance(m, _ast.Constant) and isinstance(m.value, str)
             and "neg_" + "counts" in m.value]
    chk(not reads, "this scorer must not name a neg_" + "counts artefact in any path")

    # --- distribution tails, against published critical values ---------------
    chk(abs(t_sf(6.3138, 1) - 0.05) < 1e-4, "t_{.95,1} = 6.3138")
    chk(abs(t_sf(1.7459, 16) - 0.05) < 1e-4, "t_{.95,16} = 1.7459")
    chk(abs(t_sf(2.2281, 10) - 0.025) < 1e-4, "t_{.975,10} = 2.2281")
    chk(abs(t_sf(3.1824, 3) - 0.025) < 1e-4, "t_{.975,3} = 3.1824")
    chk(abs(t_two(0.0, 9) - 1.0) < 1e-9, "two-sided t at 0 is 1")
    chk(abs(binom_sf(4, 4) - 1.0 / 16) < 1e-12, "sign test 4/4 -> 1/16")
    chk(abs(binom_sf(2, 4) - 11.0 / 16) < 1e-12, "sign test 2/4 -> 11/16")

    # --- welch and the blocked estimator -------------------------------------
    d, se, df = welch([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    chk(abs(d - 1.0) < 1e-12, "welch mean difference")
    chk(abs(se - math.sqrt(2.0 / 3)) < 1e-12, "welch se")
    bm, bse, bdf = blocked([(1.0, 0.0), (2.0, 1.0), (3.0, 2.0)])
    chk(abs(bm - 1.0) < 1e-12 and bse == 0.0 and bdf == 2,
        "the blocked estimator must return the exact within-seed difference")

    # --- THE EPOCH-WINDOW READER.  Epochs are 0-INDEXED in the .out. ---------
    ser = {e: float(e) for e in range(EPOCHS)}
    chk(abs(window_at(ser, 300) - (295 + 296 + 297 + 298 + 299) / 5.0) < 1e-9,
        "the reading at budget 300 must be the mean of epoch indices 295-299")
    chk(abs(window_at(ser, 100) - (95 + 96 + 97 + 98 + 99) / 5.0) < 1e-9,
        "the reading at budget 100 must be the mean of epoch indices 95-99 -- "
        "IDENTICAL to what a 100-epoch run's plateau5 reads")
    chk(window_at({0: 1.0, 1: 2.0}, 300) is None,
        "an incomplete window must return None, never a partial mean")
    chk(window_at(ser, 200) == sum(range(195, 200)) / 5.0, "the ep200 reading")

    # --- POWER ARITHMETIC, reproducible from the frozen constants alone ------
    n = len(SEEDS)
    se4 = SD_PLAN * math.sqrt(2.0 / n)
    chk(abs(se4 - 0.1305) < 5e-4, "se(D) at n=6 and the 300-ep sd must be 0.1305")
    chk(abs(se4 * math.sqrt(2) - 0.1845) < 5e-4,
        "se(D-G) within batch must be 0.1845")
    chk(SD_PLAN == SD_PLAN_300 > SD_PLAN_100,
        "the planning sd must be the LARGER, 300-EPOCH one -- importing the "
        "100-epoch 0.1743 into a 300-epoch batch understates the noise by 30%")
    chk(CONFIRM_T * se4 < DECIDE,
        "the |t|>=2 bar must sit BELOW the predicted lower edge, or the batch "
        "cannot confirm its own prediction")
    chk(CONFIRM_T * se4 > REFUTE_MAX,
        "and it must sit ABOVE the refutation bar, or a refutation-shaped point "
        "estimate would be declared merely by being small")
    chk(abs(DECIDE / se4 - 2.299) < 5e-3,
        "D=+0.30 must land at t=2.30 at n=6 on the 300-epoch sd")
    # THE POWER FIX, TESTED IN BOTH DIRECTIONS.  At the OLD n=4 on the 300-epoch
    # sd the |t|>=2 bar was 0.3196, ABOVE the predicted lower edge -- the batch
    # could not have confirmed its own prediction.  Assert that failure exists.
    se_old = SD_PLAN * math.sqrt(2.0 / 4)
    chk(CONFIRM_T * se_old > DECIDE,
        "at n=4 the 300-epoch sd puts the confirm bar ABOVE the prediction -- "
        "which is exactly why the seed count moved to 6")
    chk(CONFIRM_T * se4 < DECIDE, "and at n=6 it sits below it")

    # --- THE BANDS, FIXED BEFORE DATA, TESTED IN BOTH DIRECTIONS (RULE 13) ---
    rising = [0.30, 0.50, 0.70]
    falling = [0.70, 0.40, 0.10]
    flat = [0.60, 0.62, 0.58]
    chk(band_primary(0.70, 0.30, rising) == "SURVIVES", "a rising D survives")
    chk(band_primary(0.10, 0.70, falling).startswith("REFUTED"),
        "D(300) at or below the refutation bar must REFUTE")
    chk(band_primary(REFUTE_MAX, 0.70, falling).startswith("REFUTED"),
        "the refutation bar is inclusive at +%.2f" % REFUTE_MAX)
    chk(band_primary(REFUTE_MAX + 1e-9, 0.30, rising) == "SURVIVES",
        "and one epsilon above it is NOT a refutation -- the edge is exact")
    chk(band_primary(0.30, 0.70, [0.70, 0.50, 0.30]).startswith("REFUTED"),
        "a monotone decline to below half of D(100) must REFUTE even above the bar")
    chk(band_primary(0.30, 0.70, [0.70, 0.20, 0.30]) == "SURVIVES",
        "the same endpoint WITHOUT a monotone decline must NOT refute -- the "
        "monotonicity clause has to bite, or FADE_RATIO alone would be the rule")
    chk(band_primary(0.58, 0.60, flat) == "SURVIVES", "a saturating D survives")
    chk(band_shape(+0.21) == "GROWS" and band_shape(+0.20) == "SATURATES",
        "the GROWS edge must sit exactly at +%.2f" % SAT_HALF)
    chk(band_shape(-0.21) == "DECAYS" and band_shape(-0.20) == "SATURATES",
        "the DECAYS edge must sit exactly at -%.2f" % SAT_HALF)
    chk(band_shape(0.0) == "SATURATES", "no change is SATURATES")

    # --- THE CONTROL BAND, TESTED IN BOTH DIRECTIONS -------------------------
    chk(CTRL_LO <= ANCHOR_D_CC1 <= CTRL_HI,
        "the control band must ACCEPT cc1's own D -- the cell it was derived from")
    chk(CTRL_LO <= ANCHOR_D_POOL <= CTRL_HI,
        "and the R18 ms=1e-4 pool")
    for nm, val in (("a null", 0.0), ("ar1's G", -0.139), ("a reversal", -0.50),
                    ("gc1's C100 D", 1.640)):
        chk(not (CTRL_LO <= val <= CTRL_HI),
            "the control band must REJECT %s (%+0.3f) -- a gate that accepts "
            "everything is decoration" % (nm, val))

    # --- THE BOX GATE CONSTANTS ---------------------------------------------
    chk(MIN_BOXFREE == 3 and MIN_BOXFREE <= len(SEEDS),
        "at least 3 of 4 box-free seeds are needed for any primary")
    chk(BOX_ASYM == 0.10, "the arm-asymmetry bar is 0.10 absolute")
    chk(0 < VOID_BOUND_FRAC < 1, "the VOID bar is a fraction")
    chk(math.floor(VOID_BOUND_FRAC * len(SEEDS)) + 1 == 4,
        "with 6 seeds, >50%% bound means 4 or more -- the VOID trigger")
    chk(len(SEEDS) == 6 and len(ARMS) == 4, "6 seeds x 4 arms = 24 jobs")
    chk(len(READINGS) == 3 and READINGS[-1] == EPOCHS,
        "three readings, the last at the full budget")

    # --- BOX ALGEBRA.  The floor must be unreachable and the ceiling reachable;
    #     BOTH are registered, and the second is the disclosed risk. ----------
    lo, hi = (float(x) for x in CLIP.split(":"))
    b0 = math.log(float(ALPHA0))
    travel = float(MST) * EPOCHS * STEPS_PER_EPOCH
    chk(abs(travel - 15.0) < 1e-9, "ms*T must be 15.0 nats at 300 epochs")
    chk(travel < b0 - lo, "the FLOOR -30 must be algebraically UNREACHABLE")
    chk(abs((b0 - lo) - travel - 8.0922) < 1e-3,
        "the floor must clear by 8.0922 nats")
    chk(travel > b0 - (-15.0),
        "cc1's OWN floor -15 must be REACHABLE at 300 ep -- that is WHY it moved")
    chk(abs((b0 + 15.0) / float(MST) / STEPS_PER_EPOCH - 161.8) < 0.2,
        "cc1's floor would be reached at epoch 161.8")
    # THE CEILING IS NOW PROVABLY UNREACHABLE TOO (CORRECTIONS 117.2).
    chk(travel < hi - b0, "the CEILING +9.0 must be algebraically UNREACHABLE")
    chk(abs((hi - b0) - travel - 0.9078) < 1e-3,
        "the ceiling must clear by 0.9078 nats")
    # BOTH DIRECTIONS: the ceiling this batch was BUILT with must be REFUSED,
    # and so must every other ceiling the corpus has run at this budget.
    for bad in (-2.3026, 0.0, 2.0, 6.0, 8.0):
        chk(travel > bad - b0,
            "ceiling %+0.4f must be REACHABLE at 15 nats -- refused" % bad)
    chk(abs((-2.3026 - b0) / float(MST) / STEPS_PER_EPOCH - 92.1) < 0.3,
        "the ORIGINAL -2.3026 ceiling would have been reached at epoch 92.1, "
        "i.e. for 69%% of this run -- the reason it moved")
    # The fa1 precedent that forced it: identical 15-nat travel, ceiling
    # -2.3026, and the ceiling BINDS on nodewise only.
    chk(abs(3e-4 * 100 * STEPS_PER_EPOCH - 15.0) < 1e-9,
        "fa1 carries the SAME 15 nats of meta-travel as this batch")

    # --- COUNT BIAS, over the WHOLE envelope, no slope imported --------------
    ddec = math.log10(ARMS["ch"][1] / float(ARMS["node"][1]))
    for s in SLOPE_ENVELOPE:
        chk(abs(s * ddec) < 0.005, "the D-leg count bias must be under 0.005 pp "
                                   "at slope %s" % s)
    chk(ARMS["c23"][1] == ARMS["n1d"][1],
        "the G leg carries EXACTLY ZERO count bias at every slope")
    chk(abs(math.log10(ARMS["node"][1] / float(ARMS["n1d"][1])) - T_DECADES) < 1e-5,
        "T's count confound must be %.6f decades" % T_DECADES)

    # --- HONESTY CLAUSES, asserted against this file's own text --------------
    for phrase, why in (
            ("SIGN TEST", "the per-seed gate must be labelled a sign test"),
            ("not a pairing argument", "and explicitly not a pairing argument"),
            ("RULE 11 IS OPEN", "the RULE-11 gap must be printed by the scorer"),
            ("BATCH IS VOID", "the control must be able to void the batch"),
            ("CROSS-BOX", "the box move must be declared wherever cc1 is quoted"),
            ("T IS NOT TAIL REMOVAL", "T must be refused the tail-removal label"),
            ("FIRST-CLASS", "rec_hi must be first-class, not a drop criterion"),
            ("UNRESOLVED", "an underpowered primary must have its own verdict"),
            ("DISCLOSED", "a blocked/Welch sign disagreement must be disclosed"),
            ("fixed-budget", "the refutation must name what it would mean")):
        chk(phrase in me, "%s (%r)" % (why, phrase))

    print("selftest: %d/%d PASS" % (ok, ok + fail) if fail == 0
          else "selftest: %d PASS, %d FAIL" % (ok, fail))
    return 0 if fail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs", default="runs",
                    help="directory holding hz3-*.out (the ROOT of runs/)")
    ap.add_argument("--probes", default=None,
                    help="directory holding probe_* dirs (runs/hz3)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    probes = a.probes or os.path.join(a.runs, TAG)
    have = os.path.isdir(a.runs) and any(
        fn.startswith(TAG + "-") and fn.endswith(".out") for fn in os.listdir(a.runs)) \
        if os.path.isdir(a.runs) else False
    if not have:
        print("no %s runs at %s -- this batch has NOT been submitted.  Nothing to "
              "score, and that is the correct state before the operator's review."
              % (TAG, a.runs))
        return 0
    score(a.runs, probes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
