#!/usr/bin/env python3
# =============================================================================
# c83_gen_score.py -- THE REGISTERED SCORER FOR `g3m`, the ResNet34
# generalisation of the tail prescription.  D AND G IN ONE BATCH.
#
# STANDING RULE 19: this file is git-committed BEFORE any g3m run exists.  Every
# band edge, every threshold, every refusal below is frozen here, and the
# selftest asserts each one against bin/c83_gen_r34_merged.sh's OWN TEXT so the
# registration and the batch cannot drift apart.
#
# -----------------------------------------------------------------------------
# THIS FILE REPLACES TWO DELETED SCORERS
# -----------------------------------------------------------------------------
# cycle 83 first wrote analysis/c83_gen_score.py (for `g3d`, the D leg) and
# analysis/c83_g34_score.py (for `g34`, the G leg).  Adversarial review killed
# the two-batch design pre-data; both scorers and both batch scripts are DELETED
# and this single scorer replaces them.  The reason is stated in full in
# bin/c83_gen_r34_merged.sh and CORRECTIONS 115; the short form is that the
# paper's load-bearing sentence is "D is large AND G is null", which makes
# D-minus-G the mechanism contrast, and splitting it across two batches put
# 2*sd_batch^2 (STANDING RULE 14's measured 0.21 pp batch random effect) inside
# the very number the replication exists to measure.
#
# -----------------------------------------------------------------------------
# WHAT IS SCORED
# -----------------------------------------------------------------------------
#   H1  D = chunk835 - nodewise          (the tail prescription, count-matched +6)
#   H2  G = chunk2500 - nodewise1d       (the tail REMOVED, count-matched -8)
#   H3  D - G                            (THE MECHANISM CONTRAST, within-batch)
#   H1b/H2b  per-seed sign checks -- reported as SIGN TESTS on independent
#            replicates, NEVER as a pairing argument (see PAIRING, below).
#   H0  the box gate: rec_lo AND rec_hi, PER SEED PER ARM, first-class.
#   H4  descriptive only: T2 = nodewise1d - nodewise, COUNT-CONFOUNDED by
#       construction (+0.232 pp of pure count effect) and NOT tail removal.
#
# -----------------------------------------------------------------------------
# PAIRING.  STANDING RULE 14, AND ITS LIMIT.
# -----------------------------------------------------------------------------
# Every contrast here is WITHIN ONE BATCH, so the batch offset cancels by
# construction and no cross-batch floor applies.  Within a batch, pairing on seed
# is legal.  BUT the campaign's own variance decomposition says seed carries
# almost nothing even there: sd_seed ~ 0.044 against sd_resid ~ 0.152, i.e.
# pairing removes ~8% of variance, and the two-way ANOVA finds SEED statistically
# NULL (F(60,85)=1.21, p=0.213) while BATCH is overwhelming (F(62,85)=5.47,
# p=6.9e-13).  THEREFORE:
#   * every headline statistic here is WELCH-UNPAIRED;
#   * H1b/H2b are reported as SIGN TESTS on independent replicates and their
#     power is quoted as such.  A "k of 9" line is NOT a pairing argument and is
#     NOT evidence that the seed label carries anything.  This is registered in
#     advance so nobody reads a 9/9 as if it were a paired 9/9.
#
# -----------------------------------------------------------------------------
# THE ANTI-OVERCLAIM CLAUSE
# -----------------------------------------------------------------------------
# A point estimate alone must not be banded.  NULL / ATTENUATED / REVERSES may be
# DECLARED for H1 only if D is ALSO separated from the ResNet18 ms=1e-4 pool at
# |t| >= 2; otherwise the verdict printed is CONSISTENT-WITH-R18 (UNDERPOWERED).
# At the registered planning sd (0.1742 on 33 df, the R18 FOUR-ARM pooled sd --
# not ResNet34's layerwise/scalar sd, which measures arms this batch does not
# run) and n=9, se(D) = 0.0821 and se_diff vs the anchor = 0.1248, so a
# refutation needs D <= +0.331.  The selftest asserts BOTH sides of that
# threshold so the clause is a real cut and not a blanket veto.
#
# Symmetrically for H2: "THE NULL REPLICATES" requires a TOST at delta = +-0.30
# passing at alpha = 0.05.  delta is 0.30 and NOT the +-0.15 band used for
# banding, because a TOST at 0.15 needs |G| < 0.007 at this n -- arithmetically
# unreachable at any n this campaign can afford.  That is stated here rather than
# discovered later, and it is the honest deliverable: equivalence to +-0.30.
#
# -----------------------------------------------------------------------------
# WHAT THIS SCORER MAY NOT DO
# -----------------------------------------------------------------------------
#   * NO FIELD STATISTIC.  No rho_w, no N_eff/m, no beta moments.  Direction C
#     was DROPPED on its own pre-registered adoption gate (C2 anti-concordant
#     t -11.14; C3 dissociation t -23.26).  This scorer contains no such code and
#     the selftest asserts its absence.
#   * NO RULE-11 CLAIM.  No ResNet34 ms argmax is measured by g3m.  Every verdict
#     line carries "at the R18 anchor's ms, not at ResNet34's optimum".
#   * NO POOLING with mm1/pp1/cc1.  A different architecture is a generalisation
#     test, not a fifth replication.
#   * NO CROSS-ARCHITECTURE LAW.  Two architectures is two points.
#   * T2 IS NOT TAIL REMOVAL and this scorer refuses to label it so.
# =============================================================================
"""Registered scorer for the g3m ResNet34 generalisation batch (D, G, D-G)."""

import argparse
import csv
import json
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# FROZEN REGISTRATION.  Changing any of these after a g3m run exists is a
# violation of RULE 19; the selftest cross-checks them against the batch script.
# ---------------------------------------------------------------------------
TAG = "g3m"
BATCH_SCRIPT = "bin/c83_gen_r34_merged.sh"
NET = "ResNet34"
DSET = "CIFAR10"
EPOCHS = 100
MST = "1e-4"
ALPHA0 = "1e-3"
CLIP = "-15:-2.3026"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

ARMS = {  # short tag -> (granularity, m, leg)
    "node": ("nodewise", 25556, "D"),
    "chd": ("chunk835", 25562, "D"),
    "n1d": ("nodewise1d", 8595, "G"),
    "chg": ("chunk2500", 8587, "G"),
}

# The ResNet18 anchors, re-derived from results/all_runs.csv at write time.
ANCHOR_D = 0.5805        # inverse-variance pool of mm1/pp1/cc1 at ms=1e-4
ANCHOR_D_SE = 0.0939     # Q = 0.88 on 2 df -- homogeneous, so pooling is legal
ANCHOR_G = 0.011         # cc1's EXACT-count G.  The ONLY legal G anchor:
ANCHOR_G_SE = 0.147      # the four R18 G readings are HETEROGENEOUS, Q 26.2/2df,
                         # p 2e-6, so they may NOT be pooled.
ANCHOR_DG = 0.716        # cc1's within-batch D - G

SD_PLAN = 0.1742         # R18 FOUR-ARM pooled sd on 33 df at this ms/alpha0/box
NULL_HALF = 0.15         # H1's five-way band, identical to cc1's / bn1's T1
DECIDE = 0.30            # H1 REVERSES threshold, H2 TOST delta, H3 confirm bar
BOX_GATE = 0.05          # a seed with rec_lo or rec_hi >= this is BOX-BOUND
BOX_ASYM = 0.10          # arm-to-arm occupancy asymmetry that VOIDS a contrast
MIN_BOXFREE = 6          # below this, no equivalence claim is admissible
ANTI_OVERCLAIM_T = 2.0   # |t| vs the R18 anchor required to DECLARE a band

COUNT_SLOPE = -0.4906    # pp per decade of m
T2_COUNT_BIAS = 0.232    # the pure count effect inside T2 (0.4732 decades)


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


# ---------------------------------------------------------------------------
# plateau5, re-derived from the .out epoch series.  NEVER read from the CSV
# `plateau` column, which is mean-of-last-20 and is not comparable across budgets.
# ---------------------------------------------------------------------------
EP_RE = re.compile(r"Epoch\s+(\d+).*?Test Accuracy:\s*([0-9.]+)", re.I)


def plateau5_from_out(path):
    tests = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = EP_RE.search(line)
            if m:
                tests.append((int(m.group(1)), float(m.group(2))))
    if len(tests) < 5:
        return None, len(tests)
    tests.sort()
    return sum(v for _, v in tests[-5:]) / 5.0, len(tests)


def read_probe_occupancy(probe_dir, lo, hi, last_frac=0.25):
    """rec_lo / rec_hi: the fraction of recorded beta coordinates sitting at each
    guard, averaged over the LAST quarter of the run.  Returns (rec_lo, rec_hi,
    n_records) or (None, None, 0) if no probe is on disk."""
    p = os.path.join(probe_dir, "probe.jsonl")
    if not os.path.exists(p):
        return None, None, 0
    recs = []
    with open(p, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    if not recs:
        return None, None, 0
    tail = recs[int(len(recs) * (1.0 - last_frac)):]
    lo_f, hi_f = [], []
    for r in tail:
        b = r.get("beta")
        if isinstance(b, list) and b:
            n = len(b)
            lo_f.append(sum(1 for x in b if x <= lo + 1e-9) / n)
            hi_f.append(sum(1 for x in b if x >= hi - 1e-9) / n)
        else:
            for k_lo, k_hi in (("rec_lo", "rec_hi"), ("frac_lo", "frac_hi")):
                if k_lo in r and k_hi in r:
                    lo_f.append(float(r[k_lo]))
                    hi_f.append(float(r[k_hi]))
                    break
    if not lo_f:
        return None, None, len(recs)
    return sum(lo_f) / len(lo_f), sum(hi_f) / len(hi_f), len(recs)


# ---------------------------------------------------------------------------
def load_arm(runs_dir, probes_dir, short, lo, hi):
    """Return {seed: dict(plateau5, rec_lo, rec_hi, boxfree, epochs)}."""
    out = {}
    for s in SEEDS:
        rn = "%s-%s-s%d" % (TAG, short, s)
        cands = []
        if os.path.isdir(runs_dir):
            for fn in os.listdir(runs_dir):
                if fn.startswith(rn) and fn.endswith(".out"):
                    cands.append(os.path.join(runs_dir, fn))
        if not cands:
            continue
        p5, nep = plateau5_from_out(sorted(cands)[-1])
        if p5 is None or nep < EPOCHS:
            out[s] = dict(plateau5=p5, epochs=nep, rec_lo=None, rec_hi=None,
                          boxfree=False, truncated=True)
            continue
        pd = os.path.join(probes_dir or runs_dir, "probe_%s_%s_s%d" % (short, TAG, s))
        rlo, rhi, nrec = read_probe_occupancy(pd, lo, hi)
        boxfree = (rlo is not None and rhi is not None
                   and rlo < BOX_GATE and rhi < BOX_GATE)
        out[s] = dict(plateau5=p5, epochs=nep, rec_lo=rlo, rec_hi=rhi,
                      n_records=nrec, boxfree=boxfree, truncated=False)
    return out


def band_h1(d):
    if d > DECIDE:
        return "GENERALISES"
    if -NULL_HALF <= d <= NULL_HALF:
        return "NULL ON R34"
    if d <= -DECIDE:
        return "REVERSES"
    if d < -NULL_HALF:
        return "ATTENUATED-NEGATIVE"
    return "ATTENUATED"


def score(runs_dir, probes_dir, out=sys.stdout):
    lo, hi = (float(x) for x in CLIP.split(":"))
    P = lambda *a: print(*a, file=out)
    P("=" * 78)
    P("  g3m -- ResNet34 generalisation of the tail prescription (D, G, D-G)")
    P("  %s / %s / %d ep / ms=%s / alpha0=%s / BETA_CLIP=%s"
      % (NET, DSET, EPOCHS, MST, ALPHA0, CLIP))
    P("  ONE batch, FOUR arms, ONE ms, n=9 -- every contrast is WITHIN-BATCH,")
    P("  so STANDING RULE 14's cross-batch floor does NOT apply to any of them.")
    P("  RULE 11 IS OPEN: no ResNet34 ms argmax is measured.  Every verdict below")
    P("  holds at the ResNet18 anchor's own ms, NOT at ResNet34's optimum.")
    P("=" * 78)

    data = {k: load_arm(runs_dir, probes_dir, k, lo, hi) for k in ARMS}

    # --- H0: THE BOX GATE.  rec_lo AND rec_hi, PER SEED PER ARM, FIRST-CLASS. --
    P("")
    P("H0  BOX OCCUPANCY, per seed per arm.  rec_hi is reported as a FIRST-CLASS")
    P("    number, not merely as a drop criterion: at ms=%s the CEILING is" % MST)
    P("    algebraically reachable from epoch 92.1, INSIDE the plateau5 window.")
    P("    (The FLOOR is unreachable by 3.09 nats -- that is why this ms is pinned.)")
    P("    %-6s %-5s %9s %9s %9s  %s" % ("arm", "seed", "plateau5", "rec_lo", "rec_hi", "status"))
    occ = {}
    for k, (gran, m, leg) in ARMS.items():
        los, his = [], []
        for s in SEEDS:
            d = data[k].get(s)
            if d is None:
                P("    %-6s %-5d %9s %9s %9s  MISSING" % (k, s, "-", "-", "-"))
                continue
            if d.get("truncated"):
                P("    %-6s %-5d %9s %9s %9s  TRUNCATED (%s ep) -- DROPPED"
                  % (k, s, "-", "-", "-", d["epochs"]))
                continue
            rl, rh = d["rec_lo"], d["rec_hi"]
            if rl is None:
                stat = "NO PROBE -- occupancy UNMEASURED, seed UNINTERPRETABLE"
            elif d["boxfree"]:
                stat = "box-free"
                los.append(rl)
                his.append(rh)
            else:
                stat = "BOX-BOUND -- DROPPED"
                los.append(rl)
                his.append(rh)
            P("    %-6s %-5d %9.3f %9s %9s  %s"
              % (k, s, d["plateau5"],
                 "%.4f" % rl if rl is not None else "-",
                 "%.4f" % rh if rh is not None else "-", stat))
        occ[k] = (st.mean(los) if los else None, st.mean(his) if his else None)

    voids = []
    for leg, a, b in (("D", "chd", "node"), ("G", "chg", "n1d")):
        (la, ha), (lb, hb) = occ.get(a, (None, None)), occ.get(b, (None, None))
        if None in (la, ha, lb, hb):
            P("    %s leg: occupancy UNMEASURED on at least one arm -- the standing"
              " rule makes the contrast UNINTERPRETABLE." % leg)
            voids.append(leg)
            continue
        dlo, dhi = abs(la - lb), abs(ha - hb)
        P("    %s leg asymmetry: |d rec_lo| %.4f  |d rec_hi| %.4f  (bar %.2f)"
          % (leg, dlo, dhi, BOX_ASYM))
        if dlo > BOX_ASYM or dhi > BOX_ASYM:
            P("    !!! %s leg VOID: the two arms do not press the guard equally, so"
              " the contrast is confounded with the guard." % leg)
            voids.append(leg)
        else:
            P("    %s leg: occupancy is SYMMETRIC.  A symmetric bind does NOT void an"
              " accuracy contrast -- both arms sit in the same box, which is the"
              " condition under which every published D was measured." % leg)

    def usable(k):
        return {s: d["plateau5"] for s, d in data[k].items()
                if not d.get("truncated") and d.get("boxfree")}

    res = {}
    # --- H1: D -------------------------------------------------------------
    P("")
    P("H1  D = chunk835 - nodewise   (count-matched, +6 groups = %+0.6f pp of bias"
      % (COUNT_SLOPE * math.log10(25562.0 / 25556.0)))
    P("    at the measured %+0.4f pp/decade count slope)" % COUNT_SLOPE)
    A, B = usable("chd"), usable("node")
    if "D" in voids or len(A) < MIN_BOXFREE or len(B) < MIN_BOXFREE:
        P("    VERDICT: VOID -- %d/%d box-free seeds (need >= %d per arm) or an"
          " asymmetric guard." % (len(A), len(B), MIN_BOXFREE))
        res["H1"] = ("VOID", None)
    else:
        d, se, df = welch(list(A.values()), list(B.values()))
        t0 = d / se
        td = (d - ANCHOR_D) / math.sqrt(se * se + ANCHOR_D_SE * ANCHOR_D_SE)
        P("    D = %+0.3f  se %.3f  t %.2f (df %.1f)  n=%d v %d"
          % (d, se, t0, df, len(A), len(B)))
        P("    vs the R18 ms=1e-4 pool %+0.4f +- %.4f:  t = %.2f"
          % (ANCHOR_D, ANCHOR_D_SE, td))
        b = band_h1(d)
        if b in ("NULL ON R34", "REVERSES", "ATTENUATED", "ATTENUATED-NEGATIVE") \
                and abs(td) < ANTI_OVERCLAIM_T:
            P("    VERDICT: CONSISTENT-WITH-R18 (UNDERPOWERED).  The point estimate")
            P("    bands %s, but it is NOT separated from the ResNet18 evidence at" % b)
            P("    |t| >= %.1f, so THIS MAY NOT BE WRITTEN UP AS A REFUTATION."
              % ANTI_OVERCLAIM_T)
            res["H1"] = ("CONSISTENT-WITH-R18 (UNDERPOWERED)", d)
        else:
            P("    VERDICT: %s" % b)
            res["H1"] = (b, d)
        k9 = sum(1 for s in A if s in B and A[s] > B[s])
        n9 = sum(1 for s in A if s in B)
        P("    H1b SIGN TEST (not a pairing argument -- seed is statistically NULL on")
        P("        this cluster, sd_seed 0.044 vs sd_resid 0.152): %d of %d seeds"
          % (k9, n9))
        P("        favour chunk835; exact binomial p = %.4f" % binom_sf(k9, n9))

    # --- H2: G -------------------------------------------------------------
    P("")
    P("H2  G = chunk2500 - nodewise1d   (the TAIL REMOVED, count-matched -8 groups)")
    A, B = usable("chg"), usable("n1d")
    if "G" in voids or len(A) < MIN_BOXFREE or len(B) < MIN_BOXFREE:
        P("    VERDICT: VOID -- %d/%d box-free seeds (need >= %d per arm) or an"
          " asymmetric guard." % (len(A), len(B), MIN_BOXFREE))
        res["H2"] = ("VOID", None)
    else:
        g, se, df = welch(list(A.values()), list(B.values()))
        tg = (g - ANCHOR_G) / math.sqrt(se * se + ANCHOR_G_SE * ANCHOR_G_SE)
        P("    G = %+0.3f  se %.3f  t %.2f (df %.1f)  n=%d v %d"
          % (g, se, g / se, df, len(A), len(B)))
        P("    vs cc1's EXACT-count G %+0.3f +- %.3f: t = %.2f   (the four R18 G"
          % (ANCHOR_G, ANCHOR_G_SE, tg))
        P("    readings are HETEROGENEOUS, Q 26.2 on 2 df, so they are NOT pooled.)")
        # TOST at +-DECIDE
        t_lo = (g + DECIDE) / se
        t_hi = (g - DECIDE) / se
        p_tost = max(t_sf(t_lo, df), t_sf(-t_hi, df))
        P("    TOST at delta = +-%.2f: p = %.4f" % (DECIDE, p_tost))
        P("    (delta is %.2f and NOT %.2f: at this n a TOST at %.2f needs |G| < %.4f,"
          % (DECIDE, NULL_HALF, NULL_HALF, max(0.0, NULL_HALF - 1.7459 * se)))
        P("     which is arithmetically unreachable.  The honest deliverable is")
        P("     equivalence to +-%.2f, and that is what was registered.)" % DECIDE)
        if p_tost < 0.05:
            P("    VERDICT: NULL REPLICATES -- equivalent to zero within +-%.2f."
              % DECIDE)
            P("    The mechanism claim (the gap lives in the size-1 tail) SURVIVES")
            P("    on a second architecture, at n=9 per arm -- a STRONGER null than")
            P("    ResNet18's own, which was declared at n=3 (cc1 se 0.147).")
            res["H2"] = ("NULL REPLICATES", g)
        elif abs(g) >= DECIDE and abs(tg) >= ANTI_OVERCLAIM_T:
            P("    VERDICT: NULL DOES NOT REPLICATE -- G is resolvably non-zero and")
            P("    separated from cc1.  THE MECHANISM CLAIM IS REFUTED ON ResNet34.")
            res["H2"] = ("NULL DOES NOT REPLICATE", g)
        else:
            P("    VERDICT: NULL (UNDERPOWERED) -- the TOST does not pass and G is")
            P("    not resolvably non-zero either.  This is NOT a replication of the")
            P("    null and may not be reported as one.")
            res["H2"] = ("NULL (UNDERPOWERED)", g)

    # --- H3: D - G, THE MECHANISM CONTRAST ---------------------------------
    P("")
    P("H3  D - G   THE MECHANISM CONTRAST.  WITHIN-BATCH: no cross-batch floor.")
    P("    ResNet18/cc1 reads D - G = %+0.3f." % ANCHOR_DG)
    if res.get("H1", ("VOID",))[0] == "VOID" or res.get("H2", ("VOID",))[0] == "VOID":
        P("    VERDICT: VOID (a leg is void).")
        res["H3"] = ("VOID", None)
    else:
        dv = list(usable("chd").values()), list(usable("node").values())
        gv = list(usable("chg").values()), list(usable("n1d").values())
        d, sed, _ = welch(*dv)
        g, seg, _ = welch(*gv)
        dg = d - g
        se = math.sqrt(sed * sed + seg * seg)
        P("    D - G = %+0.3f  se %.3f  t %.2f" % (dg, se, dg / se))
        if dg >= DECIDE and abs(dg / se) >= ANTI_OVERCLAIM_T:
            P("    VERDICT: MECHANISM CONFIRMED ON ResNet34 -- the gap is present with")
            P("    the tail and absent without it, measured inside ONE batch.")
            res["H3"] = ("MECHANISM CONFIRMED", dg)
        else:
            P("    VERDICT: MECHANISM NOT RESOLVED (D - G = %+0.3f, t %.2f)."
              % (dg, dg / se))
            res["H3"] = ("MECHANISM NOT RESOLVED", dg)

    # --- H4: descriptive ---------------------------------------------------
    P("")
    P("H4  DESCRIPTIVE ONLY.  T2 = nodewise1d - nodewise spans %d -> %d groups"
      % (ARMS["node"][1], ARMS["n1d"][1]))
    P("    = %.4f decades = %+0.3f pp of PURE COUNT EFFECT at the measured slope."
      % (math.log10(25556.0 / 8595.0), T2_COUNT_BIAS))
    P("    T2 IS NOT TAIL REMOVAL and this scorer refuses to label it so.  The tail")
    P("    removal statistic is H2 (G), which is COUNT-MATCHED.")
    P("")
    P("NOT COMPUTED, BY REGISTRATION: any field statistic (rho_w, N_eff/m, beta")
    P("moments).  Direction C was dropped on its own pre-registered adoption gate.")
    P("NOT CLAIMED: a RULE-11 tuned result, a fifth replication of the R18 D series,")
    P("a cross-architecture law, or anything about CIFAR-100 or ResNet50 (whose")
    P("largest 1-D tensor is 2,048, above every count-matching K, so a chunk arm")
    P("there would SPLIT the tensors whose merging is the prescription).")
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
    chk("NJOBS=36" in bs, "batch script must emit 36 jobs")
    chk('SEEDS="0 1 2 3 4 5 6 7 8"' in bs, "batch script must run seeds 0-8")
    chk("EPOCHS=%d" % EPOCHS in bs, "batch script must pin EPOCHS=%d" % EPOCHS)
    for short, (gran, m, leg) in ARMS.items():
        chk(gran in bs, "batch script must carry arm %s" % gran)
    chk("M_NODE=%d" % ARMS["node"][1] in bs, "m(nodewise) must match")
    chk("M_CHUNK_D=%d" % ARMS["chd"][1] in bs, "m(chunk835) must match")
    chk("M_N1D=%d" % ARMS["n1d"][1] in bs, "m(nodewise1d) must match")
    chk("M_CHUNK_G=%d" % ARMS["chg"][1] in bs, "m(chunk2500) must match")
    chk("SD_PLAN=%s" % SD_PLAN in bs, "planning sd must match the batch script")
    chk("NULL_HALF=%s" % NULL_HALF in bs, "null band must match")
    chk("DECIDE=%s" % DECIDE in bs, "decide threshold must match")
    chk("ANCHOR_D=%s" % ANCHOR_D in bs, "D anchor must match")
    chk("ANCHOR_G=%s" % ANCHOR_G in bs, "G anchor must match")

    # --- the two deleted batches must NOT be resurrected ---------------------
    chk(not os.path.exists(os.path.join(repo, "bin/c83_gen_r34_dleg.sh")),
        "the fragmented D-leg batch must stay DELETED")
    chk(not os.path.exists(os.path.join(repo, "bin/c83_gen_r34_tail.sh")),
        "the fragmented G-leg batch must stay DELETED")
    chk(not os.path.exists(os.path.join(repo, "analysis/c83_g34_score.py")),
        "the fragmented G-leg scorer must stay DELETED")

    # --- the batch script must carry the guards the review demanded ----------
    for needle, why in (
            ("guard 7", "the throughput guard"),
            ("BINDING ARM", "guard 7 must take the MAX ratio over the SUBMITTED arms"),
            ("guard 8", "the box-reachability guard"),
            ("CEILING", "guard 8 must report the ceiling, not only the floor"),
            ("1.40", "the throughput floor must be 1.40x"),
            ("WALL=06:00:00", "the wall must clear the binding arm at 1.40x"),
            ("RULE 11 IS OPEN", "the RULE-11 gap must be declared, not hidden"),
            ("PROBE5", "PROBE5's ABSENCE must be asserted by guard 1b")):
        chk(needle in bs, "batch script must contain %r (%s)" % (needle, why))
    # gpu-short must be excluded: it caps at 4:00:00 and the binding arm needs 6h.
    chk("PARTS=gpu-l4" in bs and "PARTS=gpu-short" not in bs,
        "gpu-short must be excluded -- its 4h cap gives 1.11x headroom on chunk2500")

    # --- NO FIELD STATISTIC, asserted against this file's own source ----------
    me = open(os.path.abspath(__file__)).read()
    # NO FIELD STATISTIC, asserted over the AST's IDENTIFIERS -- not over the raw
    # prose, which necessarily names the things it forbids.  An earlier version of
    # this check scanned the text and flagged its OWN HEADER: the same
    # self-referential-guard failure as CORRECTIONS 112, caught here by the selftest.
    import ast as _ast
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
    # and it must never READ a neg_counts artefact off disk
    reads = [m for m in _ast.walk(_ast.parse(me))
             if isinstance(m, _ast.Constant) and isinstance(m.value, str)
             and "neg_" + "counts" in m.value]
    # (the message is itself built by concatenation so this check, like guard 1b in
    # the batch script, cannot match its own source -- CORRECTIONS 112's lesson)
    chk(not reads, "this scorer must not name a neg_" + "counts artefact in any path")

    # --- distribution tails, against published critical values ---------------
    chk(abs(t_sf(6.3138, 1) - 0.05) < 1e-4, "t_{.95,1} = 6.3138")
    chk(abs(t_sf(1.7459, 16) - 0.05) < 1e-4, "t_{.95,16} = 1.7459")
    chk(abs(t_sf(2.2281, 10) - 0.025) < 1e-4, "t_{.975,10} = 2.2281")
    chk(abs(t_two(0.0, 9) - 1.0) < 1e-9, "two-sided t at 0 is 1")
    chk(abs(binom_sf(9, 9) - 1.0 / 512) < 1e-12, "sign test 9/9 -> 1/512")
    chk(abs(binom_sf(5, 9) - 0.5) < 1e-12, "sign test 5/9 -> 0.5")

    # --- welch ---------------------------------------------------------------
    d, se, df = welch([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
    chk(abs(d - 1.0) < 1e-12, "welch mean difference")
    chk(abs(se - math.sqrt(2.0 / 3)) < 1e-12, "welch se")

    # --- POWER ARITHMETIC.  These are the numbers the batch was sized on and
    #     they must be reproducible from the frozen constants alone. -----------
    se9 = SD_PLAN * math.sqrt(2.0 / 9)
    chk(abs(se9 - 0.0821) < 5e-4, "se(D) at n=9 must be 0.0821")
    sed = math.sqrt(se9 ** 2 + ANCHOR_D_SE ** 2)
    chk(abs(sed - 0.1248) < 5e-4, "se_diff vs the R18 pool must be 0.1248")
    refut = ANCHOR_D - ANTI_OVERCLAIM_T * sed
    chk(abs(refut - 0.331) < 2e-3, "the refutation window must be D <= +0.331")
    chk(DECIDE - 1.7459 * se9 > 0.10, "a TOST at +-0.30 must be REACHABLE at n=9")
    chk(NULL_HALF - 1.7459 * se9 < 0.02,
        "a TOST at +-0.15 must be UNREACHABLE -- that is why DECIDE is the delta")
    chk(abs(se9 * math.sqrt(2) - 0.1162) < 5e-4, "se(D-G) within batch must be 0.1162")

    # --- THE ANTI-OVERCLAIM CLAUSE, TESTED IN BOTH DIRECTIONS (RULE 13) ------
    # AT THE FULL n=9 THE CLAUSE IS VACUOUS, AND THAT IS THE POINT OF SIZING THE
    # BATCH THIS WAY.  The refutation window is D <= +0.331, which sits ABOVE the
    # top of the REVERSES/NULL/ATTENUATED bands (all d <= +0.30), so every
    # refutation-shaped point estimate is ALREADY separated from ResNet18 and is
    # declarable.  The clause exists for the DEGRADED case -- seeds dropped for box
    # binding -- where n falls and the window closes below the band edge.  The
    # deleted g3d sat at n=3, where the window was +0.240 and the clause vetoed
    # most of its own ATTENUATED band; that is a symptom of an underpowered batch,
    # not a virtue of the clause.
    t_at = lambda d, s=sed: (d - ANCHOR_D) / s
    chk(band_h1(0.20) == "ATTENUATED" and abs(t_at(0.20)) >= ANTI_OVERCLAIM_T,
        "D=+0.20 must band ATTENUATED and BE declarable at n=9")
    chk(band_h1(0.30) == "ATTENUATED" and abs(t_at(0.30)) >= ANTI_OVERCLAIM_T,
        "the TOP of the ATTENUATED band must still be declarable at n=9 -- the "
        "clause must not veto anything the batch was sized to resolve")
    chk(refut > DECIDE,
        "at n=9 the refutation window must sit ABOVE the ATTENUATED band edge, i.e. "
        "the anti-overclaim clause must be VACUOUS at full n")
    # THE OTHER DIRECTION: at n=4 (five of nine seeds dropped for box binding) the
    # window closes to +0.271 and the top of the ATTENUATED band is no longer
    # declarable.  The clause MUST bite there or it is decoration.
    se4 = SD_PLAN * math.sqrt(2.0 / 4)
    sed4 = math.sqrt(se4 ** 2 + ANCHOR_D_SE ** 2)
    chk(ANCHOR_D - ANTI_OVERCLAIM_T * sed4 < DECIDE,
        "at n=4 the refutation window must close BELOW the band edge")
    chk(abs(t_at(0.29, sed4)) < ANTI_OVERCLAIM_T,
        "at n=4 a D of +0.29 must be SUPPRESSED to CONSISTENT-WITH-R18")
    chk(abs(t_at(0.10, sed4)) >= ANTI_OVERCLAIM_T,
        "even at n=4 a genuine NULL (D=+0.10) must remain declarable")
    chk(band_h1(0.10) == "NULL ON R34", "D=+0.10 bands NULL")
    chk(band_h1(-0.40) == "REVERSES", "D=-0.40 bands REVERSES")
    chk(band_h1(0.60) == "GENERALISES", "D=+0.60 bands GENERALISES")
    chk(band_h1(0.15) == "NULL ON R34" and band_h1(0.1501) == "ATTENUATED",
        "the NULL band edge must sit exactly at +-0.15")
    chk(band_h1(-0.30) == "REVERSES" and band_h1(-0.2999) == "ATTENUATED-NEGATIVE",
        "the REVERSES edge must sit exactly at -0.30")

    # --- COUNT BIAS -- both legs, from the frozen m ---------------------------
    bd = COUNT_SLOPE * math.log10(ARMS["chd"][1] / float(ARMS["node"][1]))
    bg = COUNT_SLOPE * math.log10(ARMS["chg"][1] / float(ARMS["n1d"][1]))
    chk(abs(bd) < 0.005, "D-leg count bias must be under 0.005 pp")
    chk(abs(bg) < 0.005, "G-leg count bias must be under 0.005 pp")
    chk(abs(abs(COUNT_SLOPE * math.log10(25556.0 / 8595.0)) - T2_COUNT_BIAS) < 0.005,
        "T2's registered count confound must be +0.232 pp")

    # --- BOX ALGEBRA.  The floor must be unreachable and the ceiling reachable;
    #     both are registered, and the second is the disclosed risk. -----------
    lo, hi = (float(x) for x in CLIP.split(":"))
    b0 = math.log(float(ALPHA0))
    travel = float(MST) * EPOCHS * 500
    chk(travel < b0 - lo, "the FLOOR must be algebraically UNREACHABLE at ms=%s" % MST)
    chk(travel > hi - b0, "the CEILING must be algebraically REACHABLE -- disclosed")
    chk(abs((hi - b0) / float(MST) / 500 - 92.1) < 0.2,
        "the ceiling must become reachable at epoch 92.1, inside the plateau5 window")

    # --- HONESTY CLAUSES, asserted against this file's own text ---------------
    for phrase, why in (
            ("SIGN TEST", "the per-seed gate must be labelled a sign test"),
            ("NOT a pairing argument", "and explicitly not a pairing argument"),
            ("RULE 11 IS OPEN", "the RULE-11 gap must be printed by the scorer"),
            ("CONSISTENT-WITH-R18 (UNDERPOWERED)", "the anti-overclaim verdict"),
            ("NULL (UNDERPOWERED)", "the underpowered-null verdict must exist"),
            ("HETEROGENEOUS", "the R18 G readings must be declared unpoolable"),
            ("T2 IS NOT TAIL REMOVAL", "T2 must be refused the tail-removal label"),
            ("FIRST-CLASS", "rec_hi must be first-class, not a drop criterion"),
            ("ResNet50", "the ResNet50 structural exclusion must be stated")):
        chk(phrase in me, "%s (%r)" % (why, phrase))

    # --- the box gate constants ---------------------------------------------
    chk(MIN_BOXFREE == 6, "at least 6 box-free seeds are needed for any null")
    chk(BOX_ASYM == 0.10, "the arm-asymmetry bar is 0.10 absolute")
    chk(len(SEEDS) == 9 and len(ARMS) == 4, "9 seeds x 4 arms = 36 jobs")

    print("selftest: %d/%d PASS" % (ok, ok + fail) if fail == 0
          else "selftest: %d PASS, %d FAIL" % (ok, fail))
    return 0 if fail == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs", default="runs/g3m", help="directory holding g3m-*.out")
    ap.add_argument("--probes", default=None, help="directory holding probe_* dirs")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not os.path.isdir(a.runs):
        print("no g3m runs at %s -- this batch has NOT been submitted.  Nothing to "
              "score, and that is the correct state before the operator's review."
              % a.runs)
        return 0
    score(a.runs, a.probes or a.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
