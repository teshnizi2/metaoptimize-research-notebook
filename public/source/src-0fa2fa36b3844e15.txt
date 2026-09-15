#!/usr/bin/env python3
"""c77_pp1_score.py -- score `pp1`, THE THREE-WAY DECOMPOSITION OF mm1's +0.485 pp.

REGISTERED GATES, transcribed from `bin/c77_permuted_partition.sh:41-113`.  Every
constant below is asserted by the selftest against THAT SCRIPT'S OWN TEXT so the
registration and the batch cannot drift -- STANDING RULE (19).

WRITTEN AND SELFTESTED AND GIT-COMMITTED BEFORE ANY pp1 VERDICT WAS READ.

THE THREE ARMS, all at m ~ 14,420, one batch, ms=1e-4, 100 ep, seeds 0-2:
    node  nodewise      groups = output channels;  sizes HETEROGENEOUS
    perm  permnode<S>   nodewise's group COUNT and per-tensor size MULTISET EXACTLY,
                        membership randomised WITHIN each tensor
    ch    chunk777      flat contiguous chunks;    sizes UNIFORM at 777
THE LEGS:
    node -> perm  = A = ALIGNMENT          (sizes IDENTICAL, only membership moves)
    perm -> ch    = B = SIZE DISTRIBUTION  (both memberships arbitrary)
    node -> ch    = D = both at once       = mm1's +0.485, re-measured here
A + B == D holds ALGEBRAICALLY.  It is a receipt, never a result.

  P0    VALIDITY.  n_records == 10000; beta moved; epochs_done == requested == 100.
  P0.2  n_beta EXACT on EVERY record: node 14420, perm 14420, ch 14421.  All three
        MEASURED by guard 4 from the ALLOCATED beta on the real built network, and
        guard 4 ALSO asserted perm's per-tensor size multiset equals nodewise's.
        Neither number is inherited from a comment.
  P0.3  THE INSTRUMENT FIRED.  neg_counts.json present, n_tot == n_beta, npy shape
        READ FROM ITS HEADER == (n_tot,).  (Header, never file size -- that
        inference produced c74's false VOID on 12/12 arms.)
  P0.4  BOX-FREE at BOTH guards, the PUBLISHED rec_-based 5% gate, PRIMARY and
        UNCHANGED.  All 6 mm1 arms read 0.0000 on all four columns in this SAME
        box, so a bind on the permnode arm would be NEW and is reported as such.

  P1    **THE REPLICATION.  REGISTERED.**  D = plateau5(ch) - plateau5(node),
        measured HERE.  mm1 measured +0.485.  PREDICTS |D - 0.485| <= 0.50.
        This is the FIRST out-of-batch replication of M1.
        **P1 DOES NOT GATE P2** -- P2 is a within-batch contrast between two arms
        that both run here, so it stands whatever P1 does.  Saying so in advance
        stops a failed replication being used to discard an inconvenient P2.

  P1.5  **THE DECOMPOSABILITY PRECONDITION.**  If D is not RESOLVABLY POSITIVE in
        THIS batch, the quantity being decomposed does not exist here and P3/P4
        must NOT be reported as a decomposition of anything.  The batch script
        registers this condition in words but fixes no number for "resolvably
        positive"; THIS SCORER FIXES IT, and fixed it before any pp1 number was
        read: D > +0.15 (M1's own refute line, so no new threshold is invented)
        AND t = D/se_D >= 2.0.  Both must hold.  The scorer PRINTS the condition
        and suppresses the decomposition language when it fails; it does not
        suppress the NUMBERS, which stay on the record either way.

  P2    **THE ALIGNMENT LEG.  THE PRIMARY.  FIVE-WAY, SYMMETRIC, REGISTERED.**
        A = plateau5(perm) - plateau5(node).  Same m, same per-tensor size
        multiset, ONLY membership randomised.  The bands are mm1's M1 bands,
        deliberately, so the two legs are read on one scale.
          A >  +0.30      -> **ALIGNMENT HURTS.**  Grouping by output channel is
               WORSE than grouping the same number of same-sized ARBITRARY subsets
               of the same layer.  The first claim this campaign would have about
               the Adam-mini / Adalayer / SGG line rather than about MetaOptimize.
          (+0.15, +0.30] -> **UNDECIDED**, registered in advance.
          [-0.15, +0.15] -> **NULL.**  Alignment is not the carrier; nodewise's
               deficit against chunk777 is about the SIZE DISTRIBUTION.  A clean
               negative, worth as much as the positive, and it would redirect the
               campaign toward group-size HOMOGENEITY as the design variable.
          [-0.30, -0.15) -> **UNDECIDED**, registered in advance.
          A <= -0.30      -> **ALIGNMENT HELPS.**  Channel structure is a genuine
               benefit, and B must then exceed D for the arithmetic to close.
        SYMMETRIC because no prior favours either sign.

  P3    THE SIZE-DISTRIBUTION LEG.  B = plateau5(ch) - plateau5(perm).
        **NO INDEPENDENT VERDICT, AND THAT IS DELIBERATE:** B = D - A
        algebraically, so registering a direction for B as well as for P1 and P2
        would be scoring one degree of freedom twice.  B is REPORTED with its sem
        and read as the remainder.
  P4    THE ADDITIVITY RECEIPT.  A + B == D to floating point.  An arithmetic
        check on the reduction, NOT a finding.

  P5    THE FIELD, THREE WAYS AT MATCHED m.  **DESCRIPTIVE, NO DIRECTION.**
        a_raw, a_deb, pbar, bias_share, N_eff/m for all three arms.
        **The "53.1%" sentence is NOT reproduced and must NOT be written.**

  P5b   **THE CHANNEL DISCRIMINANT.  REGISTERED, AND ITS ORIGIN IS DECLARED.**
        c77's F2 was POST-HOC: at matched m the chunk777-vs-nodewise RAW agreement
        gap is +113% bias channel and -13% debiased channel.  Measured there:
        dev_bias(chunk777) = 0.00370, dev_bias(nodewise) = 0.02346.  A prediction
        DERIVED from a post-hoc finding and tested on NEW data is a legitimate
        registration, which is why it lives in the batch script rather than being
        scored off the old data.  Let x = dev_bias(permnode), M = 0.01358.
          x < M - 0.25*(0.02346-0.00370)  -> the bias channel is an ALIGNMENT effect
          x > M + 0.25*(0.02346-0.00370)  -> it is a SIZE-DISTRIBUTION effect
          within +-25% of the midpoint     -> **UNDECIDED**, registered in advance
        **P5b and P2 are LOGICALLY INDEPENDENT** -- the field could track alignment
        while accuracy tracks size.  If they agree, that is a mechanism; **if they
        DISAGREE that is the more interesting result and it must not be buried.**

  P6    **POOLED D ACROSS mm1 AND pp1.  POST-HOC, DESCRIPTIVE, REGISTERS NOTHING.**
        D is a WITHIN-batch difference in both batches, so the unmodelled +-0.25 pp
        cross-batch offset cancels inside each and the two estimates may be
        averaged.  n=6 on mm1's exact contrast, which CORRECTIONS 106.7 ranked
        second among open items and which this batch delivers for free.  It
        carries NO verdict and no threshold is applied to it.

WHAT THIS SCORER WILL NOT DO
  * It will not let P1 gate, annotate or reorder P2.  P2 is computed and printed
    from its own two arms and its return value carries no reference to P1.
  * It will not call P3/P4 a decomposition when P1.5 fails.
  * It will not report a confirmed P2 as "architecture is irrelevant".  permnode
    permutes WITHIN each tensor, so pp1 asks whether an output channel is special
    among the same-sized subsets OF ITS OWN LAYER.  It does NOT test whether LAYER
    boundaries matter.  A confirmed P2 is a WITHIN-LAYER statement.
  * It will not present any verdict as holding at each partition's own argmax.
    nodewise is read at ms=1e-4, NOT at its own 3e-4 -- inherited from mm1
    deliberately, so P1 is M1's contrast re-measured rather than a different one.
  * It will not separate permutation variance from seed variance.  permnode<S>
    uses the run seed as its permutation seed.  That is the conservative direction
    (it can only widen the sem) but it is a limit, and it is printed with P2.
  * It will not re-derive an independence null on the fly (CORRECTIONS 26).

USAGE
  python3 analysis/c77_pp1_score.py --selftest
  python3 analysis/c77_pp1_score.py --root ../probes_pp1 --csv results/all_runs.csv
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SCRIPT = os.path.join(REPO, "bin", "c77_permuted_partition.sh")

from c52_boxfree import occupancy, records           # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -15.0, -2.3026             # `pp1`, registered in c55_neff_noise.BOXES
EPOCHS = 100
N_RECORDS = 10000
MST = "1e-4"
CHUNK_K = 777
NJOBS = 9
ARMS = ("node", "perm", "ch")       # short tokens as the batch script names them
# n_beta per arm, MEASURED by guard 4 from the ALLOCATED beta.  Not a formula.
M_OF_ARM = {"node": 14420, "perm": 14420, "ch": 14421}
GRAN_OF_ARM = {"node": "nodewise", "perm": "permnode<S>", "ch": "chunk777"}
SEEDS = (0, 1, 2)
BOXFREE_MAX = 0.05                  # the PUBLISHED rec_-based 5% gate, UNCHANGED

# A BINARY-REPRESENTATION tolerance, NOT a widening of any registered band.
# 91.15 - 91.00 evaluates to 0.15000000000000568 in IEEE754, which would put an
# exactly-on-the-line result on the wrong side of a registered threshold for a
# reason that has nothing to do with the experiment.  1e-9 pp is SEVEN orders below
# the campaign's +-0.02 pp reproducibility floor, so it can only decide cases that
# are already ties, and it is fixed here BEFORE any pp1 number was read.
EPS = 1e-9

# P1 -- the replication of mm1's M1
P1_REF = 0.485                      # mm1's MEASURED D, re-derived by guard 2b
P1_BAR = 0.50                       # |D - 0.485| <= 0.50 REPLICATES

# P1.5 -- the decomposability precondition.  THE NUMBER IS FIXED BY THIS SCORER,
# not by the batch script, and it was fixed before any pp1 number was read.
# +0.15 is M1's own refute line, so no new threshold is invented.
P15_D_MIN = 0.15
P15_T_MIN = 2.0

# P2 -- THE PRIMARY.  mm1's M1 bands, made symmetric.
P2_NULL = 0.15                      # |A| <= this -> NULL
P2_STRONG = 0.30                    # |A| >  this -> HURTS (+) / HELPS (-)

# P5b -- the channel discriminant.  All three constants are c77's MEASURED values.
P5B_CH = 0.00370                    # dev_bias(chunk777) at matched m
P5B_NODE = 0.02346                  # dev_bias(nodewise)  at matched m
P5B_MID = (P5B_CH + P5B_NODE) / 2.0                 # 0.01358
P5B_HALFBAND = 0.25 * (P5B_NODE - P5B_CH)           # +-25% of the gap

MM1_D = 0.485                       # mm1's D, for P6's pooling
MM1_D_N = 3                         # seeds per arm in mm1


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def parse_pp1_dirname(base):
    """'probe_perm_pp1_s0' -> ('perm', 0).  Returns (None, None) on anything else.

    probe5_window.parse_dirname cannot see chunk or permnode rungs and IT IS NOT
    PATCHED -- six registered scorers depend on it.  Parsed here instead, and the
    arm token is required to be a REGISTERED pp1 arm so a stray directory, or an
    mm1 directory with the same arm token, can never enter.
    """
    parts = base.split("_")
    if not parts or parts[0] != "probe":
        return None, None
    parts = parts[1:]
    seed = None
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
        parts = parts[:-1]
    if "pp1" not in parts:
        return None, None
    arm = next((p for p in parts if p in M_OF_ARM), None)
    return arm, seed


# --- P0 ---------------------------------------------------------------------
def gate_P0(d, arm, row=None):
    R = records(d)
    want = M_OF_ARM.get(arm)
    n_ok = len(R) == N_RECORDS
    nb = {r.get("n_beta") for r in R}
    nb_ok = (nb == {want}) if want else False
    mins = [r["beta_true_min"] for r in R]
    maxs = [r["beta_true_max"] for r in R]
    moved = (max(maxs) - min(mins)) > 1e-9
    ep_ok, ep_d, ep_r = None, None, None
    if row is not None:
        ep_d, ep_r = row.get("epochs_done", ""), row.get("epochs_requested", "")
        ep_ok = (ep_d == str(EPOCHS) and ep_r == str(EPOCHS))
    return dict(arm=arm, n_records=len(R),
                n_beta=sorted(x for x in nb if x is not None),
                n_ok=n_ok, nb_ok=nb_ok, moved=moved, ep_ok=ep_ok,
                epochs_done=ep_d, epochs_requested=ep_r,
                span=max(maxs) - min(mins),
                ok=(n_ok and nb_ok and moved and (ep_ok is not False)))


# --- P0.3 -------------------------------------------------------------------
def gate_P03(d, arm):
    j = os.path.join(d, "neg_counts.json")
    npy = os.path.join(d, "neg_counts.npy")
    if not os.path.exists(j):
        return dict(ok=False, why="neg_counts.json ABSENT -- instrument never fired",
                    n_tot=None, npy_ok=False, shape=None)
    meta = json.load(open(j))
    want = M_OF_ARM.get(arm)
    n_tot = meta.get("n_tot")
    tot_ok = (n_tot == want)
    npy_ok, shape = False, None
    if os.path.exists(npy):
        import numpy as np
        try:
            shape = tuple(np.load(npy, mmap_mode="r").shape)
            npy_ok = shape == (n_tot,)
        except Exception as e:
            shape = "unreadable: %s" % e
    return dict(ok=(tot_ok and npy_ok), n_tot=n_tot, npy_ok=npy_ok, shape=shape,
                why="" if (tot_ok and npy_ok) else
                    ("n_tot %s != %s" % (n_tot, want) if not tot_ok
                     else "npy shape %s != (%s,)" % (shape, n_tot)))


# --- P0.4 -------------------------------------------------------------------
def gate_P04(d, lo=LO, hi=HI):
    o = occupancy(d, lo, hi)
    boxfree = (o["rec_lo"] < BOXFREE_MAX) and (o["rec_hi"] < BOXFREE_MAX)
    return dict(boxfree=boxfree, rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                coord_lo=o["coord_lo"], coord_hi=o["coord_hi"],
                which=("hi" if o["rec_hi"] >= BOXFREE_MAX else
                       ("lo" if o["rec_lo"] >= BOXFREE_MAX else "")))


# --- P1 ---------------------------------------------------------------------
def score_P1(ch, node):
    """THE REPLICATION.  Two-way against a +-0.50 bar registered in advance.
    Carries NO reference to P2 and cannot gate it."""
    if not ch or not node:
        return dict(verdict="NO DATA", D=None, ref=P1_REF, bar=P1_BAR,
                    n_ch=len(ch), n_node=len(node), reading="")
    mc, mn = statistics.mean(ch), statistics.mean(node)
    D = mc - mn
    se = math.sqrt(_sem(ch) ** 2 + _sem(node) ** 2)
    t = (D / se) if se > 0 else float("inf")
    ok = abs(D - P1_REF) <= P1_BAR + EPS
    return dict(verdict=("REPLICATES" if ok else "DEVIATES"), D=D, ch=mc, node=mn,
                sem_ch=_sem(ch), sem_node=_sem(node), n_ch=len(ch), n_node=len(node),
                se_D=se, t=t, ref=P1_REF, bar=P1_BAR, dev=D - P1_REF,
                reading=("mm1's +%.3f pp is reproduced out of batch within the "
                         "registered +-%.2f pp bar" % (P1_REF, P1_BAR) if ok else
                         "D departs from mm1's +%.3f by %+.3f pp, outside the "
                         "registered +-%.2f bar -- the contrast does NOT reproduce "
                         "out of batch, and that is itself a result about how far "
                         "any single-batch partition number can be trusted"
                         % (P1_REF, D - P1_REF, P1_BAR)))


# --- P1.5 -------------------------------------------------------------------
def score_P15(p1):
    """THE DECOMPOSABILITY PRECONDITION.  D must be RESOLVABLY POSITIVE in THIS
    batch or P3/P4 are not a decomposition of anything."""
    if p1.get("D") is None:
        return dict(ok=False, why="NO DATA", D=None, t=None)
    D, t = p1["D"], p1["t"]
    big = D > P15_D_MIN + EPS
    res = t >= P15_T_MIN - EPS
    return dict(ok=(big and res), D=D, t=t, d_min=P15_D_MIN, t_min=P15_T_MIN,
                big=big, res=res,
                why=("" if (big and res) else
                     ("D=%+.3f is not > %+.2f" % (D, P15_D_MIN) if not big
                      else "t=%.2f is under %.1f" % (t, P15_T_MIN))))


# --- P2 ---------------------------------------------------------------------
def score_P2(perm, node):
    """**THE PRIMARY.**  FIVE-WAY, SYMMETRIC, thresholds fixed before the data
    existed.  Within-batch, so no cross-batch offset can enter it.  Computed from
    its own two arms only; P1's result is not consulted."""
    if not perm or not node:
        return dict(verdict="NO DATA", A=None, perm=None, node=None,
                    n_perm=len(perm), n_node=len(node),
                    null_at=P2_NULL, strong_at=P2_STRONG, reading="")
    mp, mn = statistics.mean(perm), statistics.mean(node)
    A = mp - mn
    if A > P2_STRONG + EPS:
        v = "ALIGNMENT HURTS"
        reading = ("grouping by OUTPUT CHANNEL is WORSE than grouping the same "
                   "number of same-sized ARBITRARY subsets of the same layer.  "
                   "This is the campaign's first claim about the Adam-mini / "
                   "Adalayer / SGG line rather than about MetaOptimize.  IT IS A "
                   "WITHIN-LAYER STATEMENT: permnode permutes inside each tensor "
                   "and says nothing about LAYER boundaries.")
    elif A > P2_NULL + EPS:
        v = "UNDECIDED"
        reading = ("landed in the band registered UNDECIDED IN ADVANCE "
                   "(+%.2f, +%.2f]; it is NOT read as either verdict"
                   % (P2_NULL, P2_STRONG))
    elif A >= -P2_NULL - EPS:
        v = "NULL"
        reading = ("ALIGNMENT IS NOT THE CARRIER.  Randomising membership while "
                   "holding the per-tensor size multiset EXACTLY changes nothing, "
                   "so nodewise's deficit against chunk777 is about the SIZE "
                   "DISTRIBUTION.  A clean negative, worth as much as the "
                   "positive, and it redirects the design variable from "
                   "architecture alignment to group-size HOMOGENEITY.")
    elif A > -P2_STRONG + EPS:
        v = "UNDECIDED"
        reading = ("landed in the band registered UNDECIDED IN ADVANCE "
                   "[-%.2f, -%.2f); it is NOT read as either verdict"
                   % (P2_STRONG, P2_NULL))
    else:
        v = "ALIGNMENT HELPS"
        reading = ("channel structure is a genuine benefit -- randomising "
                   "membership at identical sizes COSTS accuracy.  B must then "
                   "exceed D for the arithmetic to close, and that is checked.")
    se = math.sqrt(_sem(perm) ** 2 + _sem(node) ** 2)
    t = (A / se) if se > 0 else float("inf")
    return dict(verdict=v, A=A, perm=mp, node=mn, sem_perm=_sem(perm),
                sem_node=_sem(node), n_perm=len(perm), n_node=len(node),
                se_A=se, t=t, null_at=P2_NULL, strong_at=P2_STRONG, reading=reading)


# --- P3 ---------------------------------------------------------------------
def score_P3(ch, perm):
    """THE REMAINDER.  NO verdict: B = D - A algebraically."""
    if not ch or not perm:
        return dict(reading="NO DATA", B=None)
    mc, mp = statistics.mean(ch), statistics.mean(perm)
    B = mc - mp
    se = math.sqrt(_sem(ch) ** 2 + _sem(perm) ** 2)
    return dict(B=B, ch=mc, perm=mp, se_B=se, n_ch=len(ch), n_perm=len(perm),
                t=(B / se) if se > 0 else float("inf"),
                reading=("B = chunk777 - permnode = %+.3f pp.  REPORTED, NOT "
                         "SCORED: B = D - A algebraically, so registering a "
                         "direction for it would score one degree of freedom "
                         "twice." % B))


# --- P4 ---------------------------------------------------------------------
def score_P4(A, B, D):
    """AN ARITHMETIC RECEIPT.  Never a finding."""
    if A is None or B is None or D is None:
        return dict(ok=None, resid=None,
                    reading="NO DATA -- the receipt cannot be taken")
    resid = (A + B) - D
    return dict(ok=abs(resid) < 1e-9, A=A, B=B, D=D, resid=resid,
                reading=("A + B - D = %.3e.  This is arithmetic on three means of "
                         "the same three arms and is a RECEIPT ON THE REDUCTION, "
                         "not a finding." % resid))


# --- P5b --------------------------------------------------------------------
def score_P5b(x):
    """THE CHANNEL DISCRIMINANT.  THREE-WAY, registered, post-hoc ORIGIN DECLARED."""
    if x is None:
        return dict(verdict="NO DATA", x=None, mid=P5B_MID, half=P5B_HALFBAND,
                    reading="")
    if x < P5B_MID - P5B_HALFBAND - EPS:
        v = "BIAS CHANNEL IS AN ALIGNMENT EFFECT"
        reading = ("permnode's persistent tilt sits with chunk777's %.5f, not with "
                   "nodewise's %.5f, even though it carries nodewise's exact size "
                   "multiset.  What lowers the tilt is breaking the CHANNEL "
                   "grouping." % (P5B_CH, P5B_NODE))
    elif x > P5B_MID + P5B_HALFBAND + EPS:
        v = "BIAS CHANNEL IS A SIZE-DISTRIBUTION EFFECT"
        reading = ("permnode's persistent tilt sits with nodewise's %.5f, not with "
                   "chunk777's %.5f, even though its membership is arbitrary.  "
                   "What lowers the tilt is the UNIFORM group size."
                   % (P5B_NODE, P5B_CH))
    else:
        v = "UNDECIDED"
        reading = ("x sits within +-25%% of the midpoint %.5f, the band registered "
                   "UNDECIDED IN ADVANCE; it is NOT read as either channel."
                   % P5B_MID)
    return dict(verdict=v, x=x, mid=P5B_MID, half=P5B_HALFBAND,
                lo=P5B_MID - P5B_HALFBAND, hi=P5B_MID + P5B_HALFBAND,
                ch_ref=P5B_CH, node_ref=P5B_NODE, reading=reading)


def agree_or_not(p2_verdict, p5b_verdict):
    """P2 and P5b are LOGICALLY INDEPENDENT.  A DISAGREEMENT IS THE MORE
    INTERESTING RESULT AND MUST NOT BE BURIED."""
    if p2_verdict in ("NO DATA", "UNDECIDED") or p5b_verdict in ("NO DATA",
                                                                 "UNDECIDED"):
        return ("INCONCLUSIVE", "at least one of the two is UNDECIDED or absent; "
                                "no cross-reading is taken")
    p2_align = (p2_verdict in ("ALIGNMENT HURTS", "ALIGNMENT HELPS"))
    p5_align = (p5b_verdict == "BIAS CHANNEL IS AN ALIGNMENT EFFECT")
    if p2_align == p5_align:
        return ("AGREE", "accuracy and the field point at the same carrier -- that "
                         "is a MECHANISM, and it is the weaker of the two possible "
                         "outcomes only because it is the expected one")
    return ("DISAGREE", "**THE FIELD AND THE ACCURACY POINT AT DIFFERENT CARRIERS.** "
                        "The instrument tracks one thing and plateau accuracy "
                        "tracks another.  This was declared in advance to be the "
                        "MORE INTERESTING result and it is printed first, not "
                        "buried: it means the sign-agreement field is NOT a "
                        "sufficient statistic for the partition's effect on "
                        "optimisation.")


# --- P6 ---------------------------------------------------------------------
def score_P6(D_here, n_here):
    """POST-HOC, DESCRIPTIVE, REGISTERS NOTHING.  n=6 on mm1's exact contrast."""
    if D_here is None:
        return dict(pooled=None, reading="NO DATA")
    pooled = (MM1_D * MM1_D_N + D_here * n_here) / (MM1_D_N + n_here)
    return dict(pooled=pooled, mm1=MM1_D, here=D_here, n=MM1_D_N + n_here,
                reading=("POST-HOC AND DESCRIPTIVE.  D is a WITHIN-batch difference "
                         "in both batches, so the unmodelled +-0.25 pp cross-batch "
                         "offset cancels inside each and the estimates may be "
                         "averaged.  NO threshold is applied to this number and it "
                         "carries NO verdict."))


# ---------------------------------------------------------------------------
def selftest():
    n = p = 0
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""

    def ck(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("    %-70s %s" % (name, "ok" if cond else "FAIL"))

    print("c77_pp1_score selftest")
    # --- every constant against the batch script's own text (STANDING RULE 19)
    ck("batch script exists", bool(src))
    ck("script CLIP= is the registered box", "CLIP=-15:-2.3026" in src)
    ck("LO/HI match the script's CLIP", (LO, HI) == (-15.0, -2.3026))
    ck("script EPOCHS= matches", ("EPOCHS=%d" % EPOCHS) in src)
    ck("script MST= matches", ("MST=%s" % MST) in src)
    ck("script NJOBS is 9 = 3 arms x 3 seeds", ("NJOBS=%d" % NJOBS) in src)
    ck("NJOBS == len(ARMS) * len(SEEDS)", NJOBS == len(ARMS) * len(SEEDS))
    ck("script CHUNK_K= matches", ("CHUNK_K=%d" % CHUNK_K) in src)
    ck("script M_CHUNK= is this scorer's chunk count",
       ("M_CHUNK=%d" % M_OF_ARM["ch"]) in src)
    ck("script M_NODE= is this scorer's nodewise count",
       ("M_NODE=%d" % M_OF_ARM["node"]) in src)
    ck("script M_PERM= is this scorer's permnode count",
       ("M_PERM=%d" % M_OF_ARM["perm"]) in src)
    ck("N_RECORDS = epochs*500/stride 5", N_RECORDS == EPOCHS * 500 // 5)
    ck("script exports PROBE=5", "PROBE=5" in src)
    ck("script exports PROBE5=1 (instrument, not stride)", "PROBE5=1" in src)
    ck("script's arm loop is nodewise:node permnode:perm chunk<K>:ch",
       ('"nodewise:node" "permnode${S}:perm" "chunk${CHUNK_K}:ch"') in src)
    ck("script's run-name stem is pp1-<short>-s<seed>", 'RN="pp1-${SHORT}-s${S}"' in src)
    ck("perm and node counts are EQUAL -- that is the whole point of P2",
       M_OF_ARM["perm"] == M_OF_ARM["node"])
    ck("chunk is exactly ONE group from the other two",
       abs(M_OF_ARM["ch"] - M_OF_ARM["node"]) == 1)
    ck("chunk777 is the LARGER count, so P1 is not m-favoured",
       M_OF_ARM["ch"] > M_OF_ARM["node"])
    ck("5% gate UNCHANGED", BOXFREE_MAX == 0.05)
    ck("seeds 0-2", SEEDS == (0, 1, 2))
    ck("EPS is a rounding tolerance, orders below the 0.02 pp noise floor",
       0 < EPS < 1e-6)
    ck("EPS cannot move a result across the UNDECIDED band",
       EPS < (P2_STRONG - P2_NULL) / 1e6)

    # --- the registered thresholds, against the script's own text
    ck("P1 reference +0.485 is in the script", "0.485" in src)
    ck("P1 bar is written as |D - 0.485| <= 0.50", "|D - 0.485| <= 0.50" in src)
    ck("P1 is declared NOT to gate P2", "P1 DOES NOT GATE P2" in src)
    ck("the decomposability precondition is in the script",
       "if D is not resolvably positive in THIS batch" in src)
    ck("P1.5's D floor reuses M1's refute line, inventing no new threshold",
       P15_D_MIN == 0.15)
    ck("P1.5 requires BOTH a size and a resolution condition", P15_T_MIN >= 2.0)
    ck("P2 HURTS threshold +0.30 is in the script", "A >  +0.30" in src)
    ck("P2 HELPS threshold -0.30 is in the script", "A <= -0.30" in src)
    ck("P2's upper UNDECIDED band is in the script", "(+0.15, +0.30] -> **UNDECIDED**" in src)
    ck("P2's lower UNDECIDED band is in the script", "[-0.30, -0.15) -> **UNDECIDED**" in src)
    ck("P2's NULL band is in the script", "[-0.15, +0.15] -> **NULL.**" in src)
    ck("P2's bands are symmetric", P2_NULL == 0.15 and P2_STRONG == 0.30)
    ck("P3 is declared to carry NO independent verdict",
       "NO INDEPENDENT VERDICT" in src)
    ck("P4 is declared a receipt and not a finding", "NOT a finding" in src)
    ck("P5b's chunk reference 0.00370 is in the script", "0.00370" in src)
    ck("P5b's nodewise reference 0.02346 is in the script", "0.02346" in src)
    ck("P5b's midpoint 0.01358 is in the script", "0.01358" in src)
    ck("P5b's midpoint is the arithmetic mean of the two references",
       abs(P5B_MID - 0.01358) < 5e-6)
    ck("P5b's half-band is 25% of the gap",
       abs(P5B_HALFBAND - 0.25 * (P5B_NODE - P5B_CH)) < 1e-12)
    ck("P5b's post-hoc origin is DECLARED in the script", "was POST-HOC" in src)
    ck("P2 and P5b are declared LOGICALLY INDEPENDENT in the script",
       "LOGICALLY INDEPENDENT" in src)
    ck("a P2/P5b disagreement is declared the MORE INTERESTING result",
       "the more interesting result" in src)
    # the script wraps this sentence across a comment line break, so both halves
    # are asserted rather than the joined form
    ck("the within-layer limit is stated in the script",
       "AMONG THE SAME-SIZED SUBSETS OF ITS OWN LAYER" in src
       and "does NOT test whether LAYER" in src
       and "boundaries matter" in src)
    ck("the off-argmax risk is stated in the script", "not at its own argmax" in src)
    ck("the permutation-vs-seed confound is stated in the script",
       "CANNOT separate the two" in src)

    import c55_neff_noise as c55
    ck("pp1 registered in c55_neff_noise.BOXES", "pp1" in c55.BOXES)
    ck("c55's pp1 box == this scorer's box",
       c55.BOXES.get("pp1", (None,) * 3)[:2] == (LO, HI))
    ck("pp1's box is mm1's box, not a new one",
       c55.BOXES.get("pp1", (None,) * 3)[:2]
       == c55.BOXES.get("mm1", (None,) * 3)[:2])

    # --- P2: every one of the five branches, at and around every boundary
    def A(a):
        return score_P2([91.0 + a], [91.0])["verdict"]
    ck("P2 A=+1.00 -> ALIGNMENT HURTS", A(+1.00) == "ALIGNMENT HURTS")
    ck("P2 A=+0.31 -> ALIGNMENT HURTS (just over)", A(+0.31) == "ALIGNMENT HURTS")
    ck("P2 A=+0.30 -> UNDECIDED (strict >, band is inclusive at +0.30)",
       A(+0.30) == "UNDECIDED")
    ck("P2 A=+0.20 -> UNDECIDED", A(+0.20) == "UNDECIDED")
    ck("P2 A=+0.16 -> UNDECIDED (just over the NULL line)", A(+0.16) == "UNDECIDED")
    ck("P2 A=+0.15 -> NULL (inclusive)", A(+0.15) == "NULL")
    ck("P2 A=0 -> NULL", A(0.0) == "NULL")
    ck("P2 A=-0.15 -> NULL (inclusive, symmetric)", A(-0.15) == "NULL")
    ck("P2 A=-0.16 -> UNDECIDED", A(-0.16) == "UNDECIDED")
    ck("P2 A=-0.29 -> UNDECIDED", A(-0.29) == "UNDECIDED")
    ck("P2 A=-0.30 -> ALIGNMENT HELPS (inclusive <=)", A(-0.30) == "ALIGNMENT HELPS")
    ck("P2 A=-1.00 -> ALIGNMENT HELPS", A(-1.00) == "ALIGNMENT HELPS")
    ck("P2 is exactly symmetric about zero",
       A(+0.31) != A(-0.31) and A(+0.15) == A(-0.15) == "NULL")
    ck("P2 A is perm MINUS node, in that order", score_P2([90.0], [91.0])["A"] < 0)
    ck("P2 averages its seeds on both arms",
       abs(score_P2([92.0, 92.5], [91.0, 91.5])["A"] - 1.0) < 1e-9)
    ck("P2 with an empty arm is NO DATA", score_P2([], [91.0])["verdict"] == "NO DATA")
    ck("P2 with an empty reference arm is NO DATA",
       score_P2([92.0], [])["verdict"] == "NO DATA")
    ck("P2's HURTS reading names the SGG line",
       "SGG" in score_P2([91.5], [91.0])["reading"])
    ck("P2's HURTS reading states the WITHIN-LAYER limit with the claim",
       "WITHIN-LAYER STATEMENT" in score_P2([91.5], [91.0])["reading"])
    ck("P2's NULL reading redirects to group-size HOMOGENEITY",
       "HOMOGENEITY" in score_P2([91.0], [91.0])["reading"])
    ck("P2's NULL reading calls itself a clean negative",
       "clean negative" in score_P2([91.0], [91.0])["reading"])
    ck("P2's return carries no reference to P1",
       "P1" not in " ".join(str(k) for k in score_P2([92.0], [91.0])))

    # --- P1
    ck("P1 exactly at mm1's D REPLICATES",
       score_P1([91.0 + P1_REF], [91.0])["verdict"] == "REPLICATES")
    ck("P1 D=0 REPLICATES (|0-0.485| = 0.485 <= 0.50)",
       score_P1([91.0], [91.0])["verdict"] == "REPLICATES")
    ck("P1 D=+0.985 REPLICATES (exactly on the upper bar)",
       score_P1([91.985], [91.0])["verdict"] == "REPLICATES")
    ck("P1 D=+1.10 DEVIATES", score_P1([92.10], [91.0])["verdict"] == "DEVIATES")
    ck("P1 D=-0.10 DEVIATES (below the lower bar)",
       score_P1([90.90], [91.0])["verdict"] == "DEVIATES")
    ck("P1's DEVIATES reading is written as a result, not a failure",
       "itself a result" in score_P1([92.10], [91.0])["reading"])
    ck("P1 empty is NO DATA", score_P1([], [91.0])["verdict"] == "NO DATA")
    ck("P1 D is chunk MINUS node", score_P1([90.0], [91.0])["D"] < 0)

    # --- P1.5, the precondition
    strong = score_P1([92.0, 92.0, 92.0], [91.0, 91.0, 91.0])
    ck("P1.5 passes on a large, resolved D", score_P15(strong)["ok"])
    ck("P1.5 FAILS on a D of zero", not score_P15(score_P1([91.0], [91.0]))["ok"])
    ck("P1.5 FAILS on a NEGATIVE D",
       not score_P15(score_P1([90.0], [91.0]))["ok"])
    noisy = score_P1([92.0, 90.0, 93.0], [91.0, 91.0, 91.0])
    ck("P1.5 FAILS when D is big but t is under 2",
       (not score_P15(noisy)["ok"]) or noisy["t"] >= P15_T_MIN)
    ck("P1.5 reports WHICH condition failed",
       bool(score_P15(score_P1([91.0], [91.0]))["why"]))
    ck("P1.5 does not touch P2", score_P2([91.0], [91.0])["verdict"] == "NULL")

    # --- P3 / P4
    r3 = score_P3([92.0], [91.5])
    ck("P3 returns a reading, not a verdict", "verdict" not in r3)
    ck("P3 B is chunk MINUS perm", r3["B"] > 0)
    ck("P3 declares itself unscored", "REPORTED, NOT" in r3["reading"])
    ck("P4 closes exactly on consistent means",
       score_P4(0.2, 0.3, 0.5)["ok"])
    ck("P4 detects an inconsistent triple", not score_P4(0.2, 0.3, 0.9)["ok"])
    ck("P4 calls itself a receipt", "RECEIPT" in score_P4(0.2, 0.3, 0.5)["reading"])
    # the algebra the receipt checks, exercised on real means
    pn, pp, pc = [91.0, 91.2], [91.4, 91.5], [92.0, 92.1]
    _a = score_P2(pp, pn)["A"]
    _b = score_P3(pc, pp)["B"]
    _d = score_P1(pc, pn)["D"]
    ck("A + B == D on real per-seed means", score_P4(_a, _b, _d)["ok"])

    # --- P5b: all three branches, at and around both boundaries
    ck("P5b at chunk's own value -> ALIGNMENT effect",
       score_P5b(P5B_CH)["verdict"] == "BIAS CHANNEL IS AN ALIGNMENT EFFECT")
    ck("P5b at nodewise's own value -> SIZE-DISTRIBUTION effect",
       score_P5b(P5B_NODE)["verdict"] ==
       "BIAS CHANNEL IS A SIZE-DISTRIBUTION EFFECT")
    ck("P5b at the exact midpoint -> UNDECIDED",
       score_P5b(P5B_MID)["verdict"] == "UNDECIDED")
    ck("P5b just inside the lower band edge -> UNDECIDED",
       score_P5b(P5B_MID - P5B_HALFBAND + 1e-6)["verdict"] == "UNDECIDED")
    ck("P5b just outside the lower band edge -> ALIGNMENT effect",
       score_P5b(P5B_MID - P5B_HALFBAND - 1e-6)["verdict"] ==
       "BIAS CHANNEL IS AN ALIGNMENT EFFECT")
    ck("P5b just outside the upper band edge -> SIZE-DISTRIBUTION effect",
       score_P5b(P5B_MID + P5B_HALFBAND + 1e-6)["verdict"] ==
       "BIAS CHANNEL IS A SIZE-DISTRIBUTION EFFECT")
    ck("P5b's band is 25% of the gap on each side",
       abs((P5B_MID + P5B_HALFBAND) - (P5B_MID - P5B_HALFBAND)
           - 0.5 * (P5B_NODE - P5B_CH)) < 1e-12)
    ck("P5b empty is NO DATA", score_P5b(None)["verdict"] == "NO DATA")

    # --- the P2/P5b cross-reading
    ck("HURTS + ALIGNMENT-channel AGREE",
       agree_or_not("ALIGNMENT HURTS",
                    "BIAS CHANNEL IS AN ALIGNMENT EFFECT")[0] == "AGREE")
    ck("HELPS + ALIGNMENT-channel AGREE",
       agree_or_not("ALIGNMENT HELPS",
                    "BIAS CHANNEL IS AN ALIGNMENT EFFECT")[0] == "AGREE")
    ck("NULL + SIZE-channel AGREE",
       agree_or_not("NULL",
                    "BIAS CHANNEL IS A SIZE-DISTRIBUTION EFFECT")[0] == "AGREE")
    ck("HURTS + SIZE-channel DISAGREE",
       agree_or_not("ALIGNMENT HURTS",
                    "BIAS CHANNEL IS A SIZE-DISTRIBUTION EFFECT")[0] == "DISAGREE")
    ck("NULL + ALIGNMENT-channel DISAGREE",
       agree_or_not("NULL",
                    "BIAS CHANNEL IS AN ALIGNMENT EFFECT")[0] == "DISAGREE")
    ck("either UNDECIDED -> INCONCLUSIVE",
       agree_or_not("UNDECIDED",
                    "BIAS CHANNEL IS AN ALIGNMENT EFFECT")[0] == "INCONCLUSIVE")
    ck("a DISAGREEMENT is flagged as the more interesting result",
       "MORE INTERESTING" in agree_or_not(
           "ALIGNMENT HURTS", "BIAS CHANNEL IS A SIZE-DISTRIBUTION EFFECT")[1])

    # --- P6 pooling
    r6 = score_P6(0.485, 3)
    ck("P6 pooling of two equal Ds returns that D", abs(r6["pooled"] - 0.485) < 1e-12)
    ck("P6 pooling weights by seed count",
       abs(score_P6(0.0, 3)["pooled"] - 0.485 / 2) < 1e-12)
    ck("P6 n is 6 at three seeds here", score_P6(0.4, 3)["n"] == 6)
    ck("P6 declares itself POST-HOC and DESCRIPTIVE",
       "POST-HOC AND DESCRIPTIVE" in r6["reading"])
    ck("P6 declares that no threshold is applied", "NO threshold" in r6["reading"])
    ck("P6 empty is NO DATA", score_P6(None, 3)["reading"] == "NO DATA")

    # --- the dirname parser, including what it must REFUSE
    ck("probe_node_pp1_s0 -> ('node', 0)",
       parse_pp1_dirname("probe_node_pp1_s0") == ("node", 0))
    ck("probe_perm_pp1_s1 -> ('perm', 1)",
       parse_pp1_dirname("probe_perm_pp1_s1") == ("perm", 1))
    ck("probe_ch_pp1_s2 -> ('ch', 2)",
       parse_pp1_dirname("probe_ch_pp1_s2") == ("ch", 2))
    ck("an mm1 dir with the SAME arm token is refused",
       parse_pp1_dirname("probe_ch_mm1_s0") == (None, None))
    ck("an mm1 nodewise dir is refused",
       parse_pp1_dirname("probe_node_mm1_s0") == (None, None))
    ck("a ck1 chunk dir is refused",
       parse_pp1_dirname("probe_k1024_ck1_s0") == (None, None))
    ck("a tw0 dir is refused", parse_pp1_dirname("probe_w_tw0_s0") == (None, None))
    ck("an UNREGISTERED arm token is refused",
       parse_pp1_dirname("probe_lay_pp1_s0")[0] is None)
    ck("a non-probe directory is refused",
       parse_pp1_dirname("Tensorboard_outputs") == (None, None))

    # --- P0.3 shape from the HEADER (the c74 false-VOID regression)
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        def mk(arm, n_tot, dtype, arr_n=None):
            d = os.path.join(td, "probe_%s_pp1_s0" % arm)
            os.makedirs(d, exist_ok=True)
            json.dump({"n_tot": n_tot}, open(os.path.join(d, "neg_counts.json"), "w"))
            np.save(os.path.join(d, "neg_counts.npy"),
                    np.zeros(arr_n if arr_n is not None else n_tot, dtype=dtype))
            return d
        ck("int32 array of the right length PASSES P0.3",
           gate_P03(mk("ch", M_OF_ARM["ch"], np.int32), "ch")["ok"])
        ck("P0.3 reads the true shape from the header",
           gate_P03(mk("perm", M_OF_ARM["perm"], np.int32), "perm")["shape"]
           == (M_OF_ARM["perm"],))
        ck("int64 also passes on length",
           gate_P03(mk("node", M_OF_ARM["node"], np.int64), "node")["ok"])
        ck("a SHORT array FAILS P0.3",
           not gate_P03(mk("ch", M_OF_ARM["ch"], np.int32,
                           arr_n=M_OF_ARM["ch"] - 1), "ch")["ok"])
        ck("chunk's n_tot on the perm arm FAILS -- one group apart is a mismatch",
           not gate_P03(mk("perm", M_OF_ARM["ch"], np.int32), "perm")["ok"])
        dm = os.path.join(td, "probe_node_pp1_s1")
        os.makedirs(dm)
        ck("absent neg_counts.json FAILS P0.3 (bf8's failure)",
           not gate_P03(dm, "node")["ok"])

    # --- the discipline this scorer registered against itself
    ck("scorer refuses to write the 53.1% sentence", '"53.1%"' in __doc__)
    ck("P5 carries no registered direction", "DESCRIPTIVE, NO DIRECTION" in __doc__)
    ck("P1's inability to gate P2 is documented", "DOES NOT GATE P2" in __doc__)
    ck("the within-layer limit is documented",
       "WITHIN-LAYER statement" in __doc__)
    ck("the architecture-is-irrelevant sentence is explicitly refused",
       "architecture is irrelevant" in __doc__)
    ck("the off-argmax risk on nodewise is documented", "own 3e-4" in __doc__)
    ck("the permutation-vs-seed confound is documented",
       "permutation variance from seed variance" in __doc__)
    ck("P1.5's number is declared to be the SCORER's, not the script's",
       "THIS SCORER FIXES IT" in __doc__)
    ck("P6 is documented as registering nothing", "REGISTERS NOTHING" in __doc__)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_pp1"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs, refused = [], []
    for d in sorted(glob.glob(os.path.join(a.root, "probe_*"))):
        arm, seed = parse_pp1_dirname(os.path.basename(d))
        if arm is None or seed is None:
            refused.append(os.path.basename(d))
            continue
        dirs.append((d, arm, seed))
    rows = {r["run"]: r for r in csv.DictReader(open(a.csv))}

    print("=" * 78)
    print("c77 -- pp1: THE THREE-WAY DECOMPOSITION OF mm1's +0.485.  %d probe dirs"
          % len(dirs))
    print("box = (%.1f, %.4f)   ms=%s   %d ep   node(m=%d) / perm(m=%d) / ch(m=%d)"
          % (LO, HI, MST, EPOCHS, M_OF_ARM["node"], M_OF_ARM["perm"], M_OF_ARM["ch"]))
    print("legs: node->perm = A ALIGNMENT | perm->ch = B SIZE DIST | node->ch = D")
    print("=" * 78)
    if refused:
        print("REFUSED (not a registered pp1 dir): %s" % ", ".join(refused))

    # ---- P0
    print("\n--- P0  VALIDITY (n_records==%d, beta moved, ep==%d)"
          % (N_RECORDS, EPOCHS))
    p0 = {}
    for d, arm, seed in dirs:
        r = gate_P0(d, arm, rows.get("pp1-%s-s%d" % (arm, seed)))
        p0[d] = r
        print("    %-24s %-4s  n_rec=%-6d ep=%s/%s span=%.2f"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_records"],
                 r["epochs_done"], r["epochs_requested"], r["span"]))
    n0 = sum(1 for r in p0.values() if r["ok"])
    print("    P0: %d/%d" % (n0, len(dirs)))

    # ---- P0.2
    print("\n--- P0.2  n_beta EXACT on EVERY record  (guard 4 measured all three from "
          "the ALLOCATED beta)")
    n02 = 0
    for d, arm, seed in dirs:
        r = p0[d]
        ok = r["nb_ok"]
        n02 += ok
        print("    %-24s %-4s  n_beta=%-12s want %s = %d"
              % (os.path.basename(d), "PASS" if ok else "FAIL",
                 ",".join(str(x) for x in r["n_beta"]), GRAN_OF_ARM[arm],
                 M_OF_ARM[arm]))
    print("    P0.2: %d/%d" % (n02, len(dirs)))
    if n0 != len(dirs) or n02 != len(dirs) or len(dirs) != NJOBS:
        print("    **P0/P0.2 INCOMPLETE (%d dirs, expected %d) -- NOTHING BELOW IS "
              "SCORED.**" % (len(dirs), NJOBS))
        if n0 != len(dirs) or n02 != len(dirs):
            return 1

    # ---- P0.3
    print("\n--- P0.3  THE INSTRUMENT FIRED")
    n03 = 0
    for d, arm, seed in dirs:
        r = gate_P03(d, arm)
        n03 += r["ok"]
        print("    %-24s %-4s  n_tot=%-10s npy_shape=%-14s %s"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_tot"],
                 r.get("shape"), r["why"]))
    print("    P0.3: %d/%d" % (n03, len(dirs)))

    # ---- P0.4
    print("\n--- P0.4  BOX-FREE AT BOTH GUARDS (rec_-based 5%% gate PRIMARY, UNCHANGED)")
    print("    %-24s %-9s %8s %8s %10s %10s"
          % ("dir", "verdict", "rec_lo", "rec_hi", "coord_lo", "coord_hi"))
    p04 = {}
    for d, arm, seed in dirs:
        r = gate_P04(d)
        p04[d] = r
        cl = "%.5f" % r["coord_lo"] if r["coord_lo"] is not None else "None"
        chh = "%.5f" % r["coord_hi"] if r["coord_hi"] is not None else "None"
        print("    %-24s %-9s %8.4f %8.4f %10s %10s"
              % (os.path.basename(d), "free" if r["boxfree"] else "BOUND:" + r["which"],
                 r["rec_lo"], r["rec_hi"], cl, chh))
    nfree = sum(1 for r in p04.values() if r["boxfree"])
    print("    P0.4: %d/%d box-free" % (nfree, len(dirs)))
    if nfree < len(dirs):
        print("    **A BIND HERE IS NEW** -- all 6 mm1 arms read 0.0000 on all four")
        print("    columns in this SAME box.  Reported as such, and the permnode arm")
        print("    is the one that has never been run before.")

    # ---- plateau5 per arm, from the CSV
    pl = {}
    for d, arm, seed in dirs:
        row = rows.get("pp1-%s-s%d" % (arm, seed))
        if row and row.get("plateau5", "").strip():
            pl.setdefault(arm, []).append(float(row["plateau5"]))
    print("\n--- plateau5 per arm, from the CSV")
    for arm in ARMS:
        v = pl.get(arm, [])
        print("    %-12s m=%-8d n=%d  %s"
              % (GRAN_OF_ARM[arm], M_OF_ARM[arm], len(v),
                 ("%.3f +-%.3f" % (statistics.mean(v), _sem(v))) if v else "NO DATA"))

    # ---- P1  THE REPLICATION.  Computed and printed BEFORE P2, but it CANNOT gate it.
    print("\n--- P1  **THE REPLICATION.  REGISTERED.  IT DOES NOT GATE P2.**")
    s1 = score_P1(pl.get("ch", []), pl.get("node", []))
    if s1["D"] is None:
        print("    NO DATA on at least one arm -- P1 cannot be scored.")
    else:
        print("    D = chunk777 - nodewise = %+.3f pp   (se %.3f, t %.2f)"
              % (s1["D"], s1["se_D"], s1["t"]))
        print("    mm1 measured %+.3f;  deviation %+.3f;  bar +-%.2f  -> **%s**"
              % (s1["ref"], s1["dev"], s1["bar"], s1["verdict"]))
        print("    reading: %s" % s1["reading"])

    # ---- P1.5  THE DECOMPOSABILITY PRECONDITION
    print("\n--- P1.5  DECOMPOSABILITY PRECONDITION  (D > %+.2f AND t >= %.1f; the "
          "NUMBERS ARE THIS SCORER'S," % (P15_D_MIN, P15_T_MIN))
    print("           fixed before any pp1 value was read, reusing M1's refute line)")
    s15 = score_P15(s1)
    print("    -> %s%s" % ("SATISFIED -- P3/P4 may be read as a decomposition"
                           if s15["ok"] else
                           "NOT SATISFIED -- P3/P4 are NUMBERS ONLY, not a "
                           "decomposition of anything",
                           ("  (%s)" % s15["why"]) if s15["why"] else ""))

    # ---- P2  THE PRIMARY
    print("\n--- P2  **THE PRIMARY.  THE ALIGNMENT LEG.  FIVE-WAY, SYMMETRIC, "
          "REGISTERED IN ADVANCE.**")
    s2 = score_P2(pl.get("perm", []), pl.get("node", []))
    if s2["A"] is None:
        print("    NO DATA on at least one arm -- P2 cannot be scored.")
    else:
        print("    permnode<S> (m=%d, nodewise's EXACT size multiset, membership "
              "randomised)  %.3f +-%.3f (n=%d)"
              % (M_OF_ARM["perm"], s2["perm"], s2["sem_perm"], s2["n_perm"]))
        print("    nodewise    (m=%d, output channels)                              "
              "        %.3f +-%.3f (n=%d)"
              % (M_OF_ARM["node"], s2["node"], s2["sem_node"], s2["n_node"]))
        print("    A = permnode - nodewise = %+.3f pp   (se %.3f, t %.2f "
              "-- DESCRIPTIVE, the gate is the threshold on A)"
              % (s2["A"], s2["se_A"], s2["t"]))
        print("    registered: A > %+.2f HURTS | (%+.2f, %+.2f] UND | [%+.2f, %+.2f] "
              "NULL | [%+.2f, %+.2f) UND | A <= %+.2f HELPS"
              % (P2_STRONG, P2_NULL, P2_STRONG, -P2_NULL, P2_NULL,
                 -P2_STRONG, -P2_NULL, -P2_STRONG))
        print("    -> **%s**" % s2["verdict"])
        print("    reading: %s" % s2["reading"])
        print("    **LIMITS, PRINTED WITH THE VERDICT, NOT BELOW IT:**")
        print("      * permnode permutes WITHIN each tensor.  This is a WITHIN-LAYER")
        print("        statement.  It does NOT test whether LAYER boundaries matter.")
        print("      * ms is held at %s, which is nodewise's INHERITED ms and not its"
              % MST)
        print("        own argmax (3e-4).  Every verdict here is a statement at %s."
              % MST)
        print("      * permnode<S> uses the run seed as its permutation seed, so this")
        print("        batch CANNOT separate permutation variance from seed variance.")
        print("        That widens the sem rather than narrowing it, but it is a limit.")

    # ---- P3 / P4
    print("\n--- P3  THE SIZE-DISTRIBUTION LEG.  **REPORTED, NOT SCORED.**")
    s3 = score_P3(pl.get("ch", []), pl.get("perm", []))
    if s3.get("B") is None:
        print("    NO DATA")
    else:
        print("    B = chunk777 - permnode = %+.3f pp   (se %.3f, t %.2f)"
              % (s3["B"], s3["se_B"], s3["t"]))
        print("    %s" % s3["reading"])
        if not s15["ok"]:
            print("    **P1.5 NOT SATISFIED: this number is NOT a share of anything.**")

    print("\n--- P4  THE ADDITIVITY RECEIPT")
    s4 = score_P4(s2.get("A"), s3.get("B"), s1.get("D"))
    print("    %s" % s4["reading"])
    if s4["ok"] is False:
        print("    **RECEIPT FAILED -- the reduction is inconsistent.  Stop and "
              "find the bug before reading anything above.**")
    if s15["ok"] and s4["ok"]:
        print("    shares of D:  ALIGNMENT %+.1f%%   SIZE DISTRIBUTION %+.1f%%"
              % (100.0 * s2["A"] / s1["D"], 100.0 * s3["B"] / s1["D"]))

    # ---- P5  THE FIELD
    print("\n--- P5  THE FIELD, THREE WAYS AT MATCHED m.  **DESCRIPTIVE, NO DIRECTION.**")
    from neff_instrument import agreement_stats
    from c77_family_curve import split_channels
    import probe5_window as p5w
    ag, neff = {}, {}
    print("    %-24s %8s %10s %9s %9s %11s"
          % ("dir", "pbar", "bias", "a_raw", "a_deb", "bias_share"))
    for d, arm, seed in dirs:
        s = agreement_stats(d, window=(0.5, 1.0))
        if s is None:
            print("    %-24s  -- agreement_stats returned None" % os.path.basename(d))
            continue
        ag.setdefault(arm, []).append(s)
        print("    %-24s %8.5f %+10.5f %9.5f %9.5f %11.4f"
              % (os.path.basename(d), s["pbar"], s["bias"], s["a_raw"],
                 s["a_deb"], s["bias_share"]))
        w = p5w.reduce_dir(d, windows=(("steady .5-1", (0.5, 1.0)),))
        if w is None:
            continue
        ww = w["win"].get("steady .5-1")
        if not ww or ww["rho_s"] is None:
            continue
        mm, rho = float(w["n_tot"]), ww["rho_s"]
        neff.setdefault(arm, []).append(1.0 / (1.0 + (mm - 1.0) * rho))

    print("\n    per arm (seeds averaged):")
    print("      %-13s %-10s %9s %9s %9s %10s %10s %10s %12s"
          % ("arm", "m", "pbar", "a_raw", "a_deb", "dev_raw", "dev_deb", "dev_bias",
             "N_eff/m"))
    devbias = {}
    for arm in ARMS:
        v = ag.get(arm, [])
        if not v:
            continue
        chans = [split_channels(x) for x in v]
        d_raw = statistics.mean([c[0] for c in chans])
        d_deb = statistics.mean([c[1] for c in chans])
        d_bia = statistics.mean([c[2] for c in chans])
        devbias[arm] = d_bia
        ne = neff.get(arm, [])
        print("      %-13s %-10d %9.5f %9.5f %9.5f %10.5f %10.5f %10.5f %12s"
              % (GRAN_OF_ARM[arm], M_OF_ARM[arm],
                 statistics.mean([x["pbar"] for x in v]),
                 statistics.mean([x["a_raw"] for x in v]),
                 statistics.mean([x["a_deb"] for x in v]),
                 d_raw, d_deb, d_bia,
                 ("%.4f +-%.4f" % (statistics.mean(ne), _sem(ne))) if ne else "--"))
    print("    No independence null is derived here, and the \"53.1%\" sentence is")
    print("    NOT reproduced.")

    # ---- P5b  THE CHANNEL DISCRIMINANT
    print("\n--- P5b  **THE CHANNEL DISCRIMINANT.  REGISTERED; POST-HOC ORIGIN "
          "DECLARED.**")
    s5b = score_P5b(devbias.get("perm"))
    if s5b["x"] is None:
        print("    NO DATA on the permnode arm -- P5b cannot be scored.")
    else:
        print("    references (c77, mm1 data):  dev_bias(chunk777)=%.5f  "
              "dev_bias(nodewise)=%.5f" % (P5B_CH, P5B_NODE))
        print("    midpoint %.5f, band +-%.5f  ->  [%.5f, %.5f] is UNDECIDED"
              % (P5B_MID, P5B_HALFBAND, s5b["lo"], s5b["hi"]))
        print("    MEASURED HERE: dev_bias(permnode) = %.5f" % s5b["x"])
        print("    -> **%s**" % s5b["verdict"])
        print("    reading: %s" % s5b["reading"])
        if "node" in devbias and "ch" in devbias:
            print("    IN-BATCH CONTEXT (descriptive, NOT the registered gate): this")
            print("    batch's own dev_bias reads node %.5f / ch %.5f, midpoint %.5f."
                  % (devbias["node"], devbias["ch"],
                     (devbias["node"] + devbias["ch"]) / 2))
            print("    The registered gate uses c77's mm1-measured references, not")
            print("    these, because that is what was written down in advance.")

    # ---- THE CROSS-READING.  Printed before anything else can bury it.
    print("\n--- P2 x P5b  **LOGICALLY INDEPENDENT.  A DISAGREEMENT IS THE MORE "
          "INTERESTING RESULT.**")
    tag, why = agree_or_not(s2.get("verdict", "NO DATA"),
                            s5b.get("verdict", "NO DATA"))
    print("    P2  = %s" % s2.get("verdict"))
    print("    P5b = %s" % s5b.get("verdict"))
    print("    -> **%s** -- %s" % (tag, why))

    # ---- P6  POOLED D
    print("\n--- P6  POOLED D ACROSS mm1 AND pp1.  **POST-HOC, DESCRIPTIVE, "
          "REGISTERS NOTHING.**")
    s6 = score_P6(s1.get("D"), s1.get("n_ch") or 0)
    if s6["pooled"] is None:
        print("    NO DATA")
    else:
        print("    mm1 D = %+.3f (n=%d per arm);  pp1 D = %+.3f (n=%d per arm)"
              % (s6["mm1"], MM1_D_N, s6["here"], s1["n_ch"]))
        print("    pooled D = %+.3f pp over n=%d seeds per arm"
              % (s6["pooled"], s6["n"]))
        print("    %s" % s6["reading"])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
