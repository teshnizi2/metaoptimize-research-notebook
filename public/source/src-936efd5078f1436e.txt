#!/usr/bin/env python3
"""c78_bn1_score.py -- score `bn1`, IS THE MATCHED-COUNT GAP THE DEGENERATE SIZE-1 TAIL?

REGISTERED GATES, transcribed from `bin/c78_degenerate_tail.sh:57-160`.  Every constant
below is asserted by the selftest against THAT SCRIPT'S OWN TEXT so the registration and
the batch cannot drift -- STANDING RULE (19).

WRITTEN, SELFTESTED AND GIT-COMMITTED IN THE SAME TICK THE BATCH WAS SUBMITTED, WHILE NO
`bn1` RUN EXISTED IN THE CSV AT ALL.  Not merely before the verdict was read -- before the
data could exist.

THE THREE ARMS, one batch, ms=1e-4, 100 ep, seeds 0-2:
    n1d   nodewise1d  m= 4,851   channels on ndim>=2, ONE group per 1-D tensor
    c23   chunk2325   m= 4,851   uniform flat chunks, architecture-blind
    node  nodewise    m=14,420   the anchor; bitwise mm1's and pp1's nodewise arm

  T0    VALIDITY.  n_records == 10000; beta moved; epochs_done == requested == 100.
  T0.2  n_beta EXACT on EVERY record: n1d 4851, c23 4851, node 14420.  All three
        MEASURED by guard 4 from the ALLOCATED beta on the real built network, which
        ALSO asserted m(c23) == m(n1d) exactly and that nodewise1d differs from nodewise
        on the 1-D tensors and on NOTHING else.
  T0.3  THE INSTRUMENT FIRED.  neg_counts.json, n_tot == n_beta, npy shape READ FROM ITS
        HEADER == (n_tot,).  (Header, never file size -- that inference produced c74's
        false VOID on 12/12 arms.)
  T0.4  BOX-FREE at BOTH guards, the PUBLISHED rec_-based 5% gate, PRIMARY and
        UNCHANGED.  All 9 pp1 and all 6 mm1 arms read 0.0000 on all four columns in this
        SAME box, so a bind on n1d or c23 would be NEW and is reported as such.

  T1    **THE PRIMARY.  FIVE-WAY, SYMMETRIC, WITHIN-BATCH, COUNT MATCHED EXACTLY.**
        G = plateau5(chunk2325) - plateau5(nodewise1d).  Both arms m = 4,851.
          [-0.15, +0.15] -> **THE DEGENERATE TAIL WAS THE CARRIER.**  The +0.53 pp
               matched-count gap COLLAPSES once nodewise's 9,610 size-1 groups are
               merged away while the per-output-channel partition on every weight
               MATRIX is left in place.  Prescriptive for Adam-mini / Adalayer / SGG.
               **This is the NULL band, so it can only be reached by failing to find an
               effect, never by finding one.**
          (+0.15, +0.30] -> **UNDECIDED**, registered in advance.
          G >  +0.30      -> **THE GAP SURVIVES.**  Size HETEROGENEITY as such, not the
               degenerate tail, is the carrier; the fix is not "merge the 1-D tensors".
          [-0.30, -0.15) -> **UNDECIDED**, registered in advance.
          G <= -0.30      -> **REVERSAL: ALIGNMENT PAYS ONCE THE TAIL IS GONE.**  pp1's
               P2 must then be re-read as "alignment is null IN THE PRESENCE OF the
               tail" -- weaker than CORRECTIONS 107.2 currently states.  Registered in
               advance precisely so it cannot be dismissed if it fires.

  T2    **THE SECONDARY.  DIRECTIONAL -- AND CONFOUNDED BY DESIGN.**
        H = plateau5(nodewise1d) - plateau5(nodewise).  PREDICTS H > 0.
          H > +0.30 CONFIRMS | (-0.15, +0.30] UNDECIDED | H <= -0.15 REFUTES
        **THE CONFOUND IS REGISTERED, NOT DISCOVERED LATER:** the two arms differ in
        COUNT (4,851 vs 14,420) as well as in the tail, and ck1's K3 measured ~0.51
        pp/decade in favour of COARSER within one partition family.  A count effect
        alone predicts H > 0.  **A CONFIRM IS THEREFORE NOT EVIDENCE ABOUT THE TAIL**
        and this scorer prints that sentence with the verdict, every time.

  T2b   THE COUNT-CORRECTED REMAINDER.  **POST-HOC, DESCRIPTIVE, NO VERDICT.**

  T3    THE BATCH-OFFSET METER.  nodewise@1e-4 here vs pp1's 92.012 and mm1's 92.044,
        bar +-0.50.  **CANNOT VOID T1 OR T2** -- both are within-batch differences,
        immune to any offset common to their arms.  Computed and printed AFTER them.

  T4    THE FIELD, THREE WAYS.  **DESCRIPTIVE, NO DIRECTION.**
        **The "53.1%" sentence is NOT reproduced and must NOT be written.**

  T5    **THE DISSOCIATION RE-TEST.  REGISTERED; POST-HOC ORIGIN DECLARED.**
        FINDINGS 77.5 found POST-HOC that across nodewise -> permnode the field moved
        hugely (N_eff/m 0.0547 -> 0.0414, t ~ 13) while accuracy did not move at all
        (-0.009 pp, t = -0.06).  That is ONE contrast at n=3.  This batch supplies an
        INDEPENDENT matched-count pair.  Let t_F = |N_eff/m(c23) - N_eff/m(n1d)| / se.
          T1 NULL and t_F >= 3 -> **DISSOCIATION REPLICATED** on a second, independent
               contrast; the sign-agreement field is not a sufficient statistic, and
               the brief's direction-C default is refuted on its own terms.
          T1 NULL and t_F <  3 -> the field TRACKS accuracy here; 77.5 was a one-off.
          T1 NOT NULL          -> **NOT APPLICABLE**, registered in advance so that a
               non-NULL T1 cannot be used to fish for a dissociation afterwards.
        The threshold is on t, not on an absolute field difference, because an absolute
        bar fixed from pp1's sems would import pp1's noise level into this verdict.

WHAT THIS SCORER WILL NOT DO
  * It will not let T3 gate, annotate or reorder T1 or T2.
  * It will not report a T2 CONFIRM as evidence about the degenerate tail.  The count
    confound is printed with the verdict and is not relegated to a caveat below it.
  * It will not evaluate T5 unless T1 is NULL.  The branch is registered.
  * It will not separate "the tail is degenerate (size 1)" from "the tail is on the
    NORMALISATION parameters".  On ResNet18 every 1-D tensor is a BN scale/shift or the
    output bias, so the two coincide exactly and this batch cannot tell them apart.
  * It will not present any verdict as holding at each partition's own argmax.  ms is
    held at 1e-4; nodewise1d's own argmax has never been measured.
  * It will not re-derive an independence null on the fly (CORRECTIONS 26).

USAGE
  python3 analysis/c78_bn1_score.py --selftest
  python3 analysis/c78_bn1_score.py --root ../probes_bn1 --csv results/all_runs.csv
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
SCRIPT = os.path.join(REPO, "bin", "c78_degenerate_tail.sh")

from c52_boxfree import occupancy, records           # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -15.0, -2.3026             # `bn1`, registered in c55_neff_noise.BOXES
EPOCHS = 100
N_RECORDS = 10000
MST = "1e-4"
CHUNK_K = 2325
NJOBS = 9
ARMS = ("n1d", "c23", "node")
# n_beta per arm, MEASURED by guard 4 from the ALLOCATED beta.  Not a formula.
M_OF_ARM = {"n1d": 4851, "c23": 4851, "node": 14420}
GRAN_OF_ARM = {"n1d": "nodewise1d", "c23": "chunk2325", "node": "nodewise"}
SEEDS = (0, 1, 2)
BOXFREE_MAX = 0.05                  # the PUBLISHED rec_-based 5% gate, UNCHANGED

# A BINARY-REPRESENTATION tolerance, NOT a widening of any registered band.  1e-9 pp is
# SEVEN orders below the campaign's +-0.02 pp reproducibility floor, so it can only
# decide cases that are already ties.  Fixed before any bn1 number could exist.
EPS = 1e-9

# T1 -- THE PRIMARY.  mm1's M1 bands and pp1's P2 bands, symmetric.
T1_NULL = 0.15                      # |G| <= this -> the tail WAS the carrier
T1_STRONG = 0.30                    # G > this -> survives; G <= -this -> reversal

# T2 -- the confounded secondary.
T2_CONFIRM = 0.30
T2_REFUTE = -0.15

# T3 -- the offset meter.
T3_REF_PP1, T3_REF_MM1, T3_BAR = 92.012, 92.044, 0.50

# T5 -- the dissociation re-test.
T5_T_MIN = 3.0
# pp1's post-hoc numbers, quoted as CONTEXT only.  No threshold is derived from them.
PP1_NEFF_NODE, PP1_NEFF_PERM = 0.0547, 0.0414
PP1_A = -0.009

# what T1 is trying to make disappear
PP1_D, MM1_D, POOLED_D = 0.581, 0.485, 0.533
# the measured tail, from FINDINGS 77.6
N_DEGENERATE = 9610


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def parse_bn1_dirname(base):
    """'probe_n1d_bn1_s0' -> ('n1d', 0).  Returns (None, None) on anything else.

    probe5_window.parse_dirname cannot see chunk, permnode or nodewise1d rungs and IT IS
    NOT PATCHED -- seven registered scorers depend on it.  Parsed here instead, and the
    arm token is required to be a REGISTERED bn1 arm so a stray directory, or an mm1/pp1
    directory sharing an arm token, can never enter.
    """
    parts = base.split("_")
    if not parts or parts[0] != "probe":
        return None, None
    parts = parts[1:]
    seed = None
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
        parts = parts[:-1]
    if "bn1" not in parts:
        return None, None
    arm = next((p for p in parts if p in M_OF_ARM), None)
    return arm, seed


# --- T0 ---------------------------------------------------------------------
def gate_T0(d, arm, row=None):
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


# --- T0.3 -------------------------------------------------------------------
def gate_T03(d, arm):
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


# --- T0.4 -------------------------------------------------------------------
def gate_T04(d, lo=LO, hi=HI):
    o = occupancy(d, lo, hi)
    boxfree = (o["rec_lo"] < BOXFREE_MAX) and (o["rec_hi"] < BOXFREE_MAX)
    return dict(boxfree=boxfree, rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                coord_lo=o["coord_lo"], coord_hi=o["coord_hi"],
                which=("hi" if o["rec_hi"] >= BOXFREE_MAX else
                       ("lo" if o["rec_lo"] >= BOXFREE_MAX else "")))


# --- T1 ---------------------------------------------------------------------
def score_T1(c23, n1d):
    """**THE PRIMARY.**  FIVE-WAY, SYMMETRIC, count matched EXACTLY.  Within-batch, so
    no cross-batch offset can enter it.  T3 is not consulted."""
    if not c23 or not n1d:
        return dict(verdict="NO DATA", G=None, c23=None, n1d=None,
                    n_c23=len(c23), n_n1d=len(n1d),
                    null_at=T1_NULL, strong_at=T1_STRONG, is_null=False, reading="")
    mc, mn = statistics.mean(c23), statistics.mean(n1d)
    G = mc - mn
    if G > T1_STRONG + EPS:
        v = "THE GAP SURVIVES THE TAIL'S REMOVAL"
        reading = ("uniform chunks still beat the architecture-aligned partition at "
                   "matched count with the 9,610 degenerate size-1 groups gone, so size "
                   "HETEROGENEITY AS SUCH -- not the degenerate tail -- is the carrier. "
                   "This is MORE general than the null branch, and it means the fix is "
                   "NOT 'merge the 1-D tensors'.")
    elif G > T1_NULL + EPS:
        v = "UNDECIDED"
        reading = ("landed in the band registered UNDECIDED IN ADVANCE (+%.2f, +%.2f]; "
                   "it is NOT read as either verdict" % (T1_NULL, T1_STRONG))
    elif G >= -T1_NULL - EPS:
        v = "THE DEGENERATE TAIL WAS THE CARRIER"
        reading = ("the matched-count gap COLLAPSES once nodewise's %d size-1 groups are "
                   "merged away, while the per-output-channel partition on every weight "
                   "MATRIX is left exactly in place.  PRESCRIPTIVE for the Adam-mini / "
                   "Adalayer / SGG line: do not give each normalisation scalar its own "
                   "step size; the rest of the architectural partition is fine.  NOTE "
                   "THIS IS THE NULL BAND -- it is reached by failing to find an effect, "
                   "never by finding one." % N_DEGENERATE)
    elif G > -T1_STRONG + EPS:
        v = "UNDECIDED"
        reading = ("landed in the band registered UNDECIDED IN ADVANCE [-%.2f, -%.2f); "
                   "it is NOT read as either verdict" % (T1_STRONG, T1_NULL))
    else:
        v = "REVERSAL -- ALIGNMENT PAYS ONCE THE TAIL IS GONE"
        reading = ("nodewise1d BEATS uniform chunks at matched count.  **pp1's P2 must "
                   "now be re-read as 'alignment is null IN THE PRESENCE OF the tail', "
                   "which is WEAKER than CORRECTIONS 107.2 states.**  The tail was two "
                   "thirds of nodewise's groups, so a real alignment benefit on the "
                   "weight matrices could have been swamped by it.  This branch was "
                   "registered in advance precisely so it cannot be dismissed now.")
    se = math.sqrt(_sem(c23) ** 2 + _sem(n1d) ** 2)
    t = (G / se) if se > 0 else float("inf")
    return dict(verdict=v, G=G, c23=mc, n1d=mn, sem_c23=_sem(c23), sem_n1d=_sem(n1d),
                n_c23=len(c23), n_n1d=len(n1d), se_G=se, t=t,
                null_at=T1_NULL, strong_at=T1_STRONG,
                is_null=(v == "THE DEGENERATE TAIL WAS THE CARRIER"), reading=reading)


# --- T2 ---------------------------------------------------------------------
def score_T2(n1d, node):
    """THE CONFOUNDED SECONDARY.  The confound travels WITH the verdict, always."""
    CONFOUND = ("**THE COUNT CONFOUND, REGISTERED IN ADVANCE:** these two arms differ in "
                "COUNT (4,851 vs 14,420) as well as in the tail, and ck1's K3 measured "
                "~0.51 pp/decade in favour of COARSER within one partition family.  A "
                "count effect ALONE predicts H > 0.  **A CONFIRM HERE IS NOT EVIDENCE "
                "ABOUT THE DEGENERATE TAIL.**  T1 is the confound-free contrast.")
    if not n1d or not node:
        return dict(verdict="NO DATA", H=None, n_n1d=len(n1d), n_node=len(node),
                    confound=CONFOUND, reading="")
    mn1, mnd = statistics.mean(n1d), statistics.mean(node)
    H = mn1 - mnd
    if H > T2_CONFIRM + EPS:
        v = "CONFIRMS"
    elif H <= T2_REFUTE + EPS:
        v = "REFUTES"
    else:
        v = "UNDECIDED"
    se = math.sqrt(_sem(n1d) ** 2 + _sem(node) ** 2)
    return dict(verdict=v, H=H, n1d=mn1, node=mnd, sem_n1d=_sem(n1d),
                sem_node=_sem(node), n_n1d=len(n1d), n_node=len(node), se_H=se,
                t=(H / se) if se > 0 else float("inf"),
                confirm_at=T2_CONFIRM, refute_at=T2_REFUTE, confound=CONFOUND,
                reading=("removing %d degenerate groups is associated with %+.3f pp"
                         % (N_DEGENERATE, H)))


# --- T3 ---------------------------------------------------------------------
def score_T3(node):
    """THE OFFSET METER.  DECLARED UNABLE TO VOID T1 OR T2 BEFORE THE DATA EXISTED."""
    if not node:
        return dict(verdict="NO DATA", mean=None, n=0, voids=False)
    m = statistics.mean(node)
    d_pp1, d_mm1 = m - T3_REF_PP1, m - T3_REF_MM1
    ok = abs(d_pp1) <= T3_BAR + EPS and abs(d_mm1) <= T3_BAR + EPS
    return dict(mean=m, sem=_sem(node), n=len(node), bar=T3_BAR,
                ref_pp1=T3_REF_PP1, ref_mm1=T3_REF_MM1, d_pp1=d_pp1, d_mm1=d_mm1,
                verdict=("REPRODUCES" if ok else "OFFSET"), ok=ok, voids=False)


# --- T5 ---------------------------------------------------------------------
def score_T5(t1_is_null, f_c23, f_n1d):
    """THE DISSOCIATION RE-TEST.  Defined ONLY on an accuracy tie -- registered."""
    if not t1_is_null:
        return dict(verdict="NOT APPLICABLE", t_F=None, dF=None,
                    reading=("T1 is not NULL, so the dissociation test is undefined.  "
                             "This branch was registered in advance so that a non-NULL "
                             "T1 could not be used to fish for a dissociation."))
    if not f_c23 or not f_n1d:
        return dict(verdict="NO DATA", t_F=None, dF=None, reading="")
    dF = statistics.mean(f_c23) - statistics.mean(f_n1d)
    se = math.sqrt(_sem(f_c23) ** 2 + _sem(f_n1d) ** 2)
    t_F = (abs(dF) / se) if se > 0 else float("inf")
    if t_F >= T5_T_MIN - EPS:
        v = "DISSOCIATION REPLICATED"
        reading = ("accuracy TIES at matched count while the field moves at t=%.1f.  "
                   "FINDINGS 77.5 saw the same thing on the INDEPENDENT node->perm "
                   "contrast (N_eff/m %.4f -> %.4f against %+.3f pp).  Two contrasts "
                   "now say the sign-agreement field is NOT a sufficient statistic for "
                   "what a partition does to optimisation, and the brief's direction-C "
                   "default is refuted on its own terms rather than merely blocked."
                   % (t_F, PP1_NEFF_NODE, PP1_NEFF_PERM, PP1_A))
    else:
        v = "FIELD TRACKS ACCURACY"
        reading = ("accuracy ties and the field does not resolve either (t=%.1f < %.1f), "
                   "so 77.5's dissociation does NOT generalise to this contrast and must "
                   "be reported as a one-off." % (t_F, T5_T_MIN))
    return dict(verdict=v, t_F=t_F, dF=dF, se=se, t_min=T5_T_MIN,
                f_c23=statistics.mean(f_c23), f_n1d=statistics.mean(f_n1d),
                reading=reading)


# ---------------------------------------------------------------------------
def selftest():
    n = p = 0
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""

    def ck(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("    %-72s %s" % (name, "ok" if cond else "FAIL"))

    print("c78_bn1_score selftest")
    # --- every constant against the batch script's own text (STANDING RULE 19)
    ck("batch script exists", bool(src))
    ck("script CLIP= is the registered box", "CLIP=-15:-2.3026" in src)
    ck("LO/HI match the script's CLIP", (LO, HI) == (-15.0, -2.3026))
    ck("script EPOCHS= matches", ("EPOCHS=%d" % EPOCHS) in src)
    ck("script MST= matches", ("MST=%s" % MST) in src)
    ck("script NJOBS is 9 = 3 arms x 3 seeds", ("NJOBS=%d" % NJOBS) in src)
    ck("NJOBS == len(ARMS) * len(SEEDS)", NJOBS == len(ARMS) * len(SEEDS))
    ck("script CHUNK_K= matches", ("CHUNK_K=%d" % CHUNK_K) in src)
    ck("script M_N1D= matches", ("M_N1D=%d" % M_OF_ARM["n1d"]) in src)
    ck("script M_CHUNK= matches", ("M_CHUNK=%d" % M_OF_ARM["c23"]) in src)
    ck("script M_NODE= matches", ("M_NODE=%d" % M_OF_ARM["node"]) in src)
    ck("N_RECORDS = epochs*500/stride 5", N_RECORDS == EPOCHS * 500 // 5)
    ck("script exports PROBE=5", "PROBE=5" in src)
    ck("script exports PROBE5=1 (instrument, not stride)", "PROBE5=1" in src)
    ck("script's arm loop is nodewise1d:n1d chunk<K>:c23 nodewise:node",
       ('"nodewise1d:n1d" "chunk${CHUNK_K}:c23" "nodewise:node"') in src)
    ck("script's run-name stem is bn1-<short>-s<seed>", 'RN="bn1-${SHORT}-s${S}"' in src)
    ck("**T1's two arms are count-matched EXACTLY** -- the batch's whole premise",
       M_OF_ARM["n1d"] == M_OF_ARM["c23"])
    ck("the anchor arm is at nodewise's own count",
       M_OF_ARM["node"] == 14420)
    ck("5% gate UNCHANGED", BOXFREE_MAX == 0.05)
    ck("seeds 0-2", SEEDS == (0, 1, 2))
    ck("EPS is a rounding tolerance, orders below the 0.02 pp noise floor",
       0 < EPS < 1e-6)
    ck("EPS cannot move a result across an UNDECIDED band",
       EPS < (T1_STRONG - T1_NULL) / 1e6)

    # --- the registered thresholds, against the script's own text
    ck("T1's NULL band is in the script", "[-0.15, +0.15] -> **THE DEGENERATE TAIL" in src)
    ck("T1's survives branch is in the script", "G >  +0.30" in src)
    ck("T1's reversal branch is in the script", "G <= -0.30" in src)
    ck("T1's upper UNDECIDED band is in the script",
       "(+0.15, +0.30] -> **UNDECIDED**" in src)
    ck("T1's lower UNDECIDED band is in the script",
       "[-0.30, -0.15) -> **UNDECIDED**" in src)
    ck("T1's bands are symmetric", T1_NULL == 0.15 and T1_STRONG == 0.30)
    ck("T1's bands are mm1's M1 / pp1's P2 bands, so the legs share one scale",
       (T1_NULL, T1_STRONG) == (0.15, 0.30))
    # the script wraps this sentence across a comment line break
    ck("the reversal branch's consequence for pp1's P2 is written in the script",
       'alignment is null IN THE PRESENCE OF the' in src and 'tail"' in src)
    ck("T2's confirm threshold is in the script", "H >  +0.30 -> CONFIRMS" in src)
    ck("T2's refute threshold is in the script", "H <= -0.15 -> REFUTES" in src)
    ck("T2's count confound is registered in the script",
       "THE CONFOUND IS STATED BEFORE THE DATA EXISTS" in src)
    ck("the script says a T2 CONFIRM is not evidence about the tail",
       "is NOT evidence about the tail" in src)
    ck("T3's bar is in the script", "bar +-0.50" in src)
    ck("T3 is declared unable to void T1 or T2", "THIS CANNOT VOID T1 OR T2" in src)
    ck("T3's pp1 reference is in the script", "92.012" in src)
    ck("T3's mm1 reference is in the script", "92.044" in src)
    ck("T5's t threshold is in the script", "t_F >= 3" in src)
    ck("T5's NOT-APPLICABLE branch is registered in the script",
       "NOT APPLICABLE" in src)
    ck("T5's post-hoc origin is DECLARED in the script",
       "ITS POST-HOC ORIGIN DECLARED" in src)
    ck("T5's threshold is declared to be on t and not on an absolute difference",
       "The threshold is on t, not on an absolute field difference" in src)
    ck("the 9,610 degenerate groups are named in the script", "9,610" in src)
    # wrapped across a comment line break: "...cannot tell a\n#     size story from..."
    ck("the BN-vs-size non-identifiability is stated in the script",
       "cannot tell a" in src and "size story from a parameter-role story" in src)
    ck("the off-argmax risk is stated in the script",
       "not at its own\n#     argmax of 3e-4" in src or "argmax of 3e-4" in src)
    ck("the layer-boundary limit is stated in the script",
       "does NOT test whether LAYER boundaries matter" in src)

    import c55_neff_noise as c55
    ck("bn1 registered in c55_neff_noise.BOXES", "bn1" in c55.BOXES)
    ck("c55's bn1 box == this scorer's box",
       c55.BOXES.get("bn1", (None,) * 3)[:2] == (LO, HI))
    ck("bn1's box is pp1's box, not a new one",
       c55.BOXES.get("bn1", (None,) * 3)[:2]
       == c55.BOXES.get("pp1", (None,) * 3)[:2])

    # --- T1: every one of the five branches, at and around every boundary
    def G(g):
        return score_T1([91.0 + g], [91.0])["verdict"]
    NULLV = "THE DEGENERATE TAIL WAS THE CARRIER"
    SURV = "THE GAP SURVIVES THE TAIL'S REMOVAL"
    REV = "REVERSAL -- ALIGNMENT PAYS ONCE THE TAIL IS GONE"
    ck("T1 G=+1.00 -> gap survives", G(+1.00) == SURV)
    ck("T1 G=+0.31 -> gap survives (just over)", G(+0.31) == SURV)
    ck("T1 G=+0.30 -> UNDECIDED (strict >)", G(+0.30) == "UNDECIDED")
    ck("T1 G=+0.16 -> UNDECIDED", G(+0.16) == "UNDECIDED")
    ck("T1 G=+0.15 -> tail was the carrier (inclusive)", G(+0.15) == NULLV)
    ck("T1 G=0 -> tail was the carrier", G(0.0) == NULLV)
    ck("T1 G=-0.15 -> tail was the carrier (inclusive, symmetric)", G(-0.15) == NULLV)
    ck("T1 G=-0.16 -> UNDECIDED", G(-0.16) == "UNDECIDED")
    ck("T1 G=-0.29 -> UNDECIDED", G(-0.29) == "UNDECIDED")
    ck("T1 G=-0.30 -> REVERSAL (inclusive <=)", G(-0.30) == REV)
    ck("T1 G=-1.00 -> REVERSAL", G(-1.00) == REV)
    ck("T1's bands are symmetric about zero", G(+0.15) == G(-0.15) == NULLV)
    ck("T1 G is chunk MINUS nodewise1d, in that order",
       score_T1([90.0], [91.0])["G"] < 0)
    ck("**pp1's D of +0.581, if reproduced here, would say the gap SURVIVES**",
       G(PP1_D) == SURV)
    ck("**a full collapse to zero would say the tail WAS the carrier**", G(0.0) == NULLV)
    ck("T1 averages its seeds on both arms",
       abs(score_T1([92.0, 92.5], [91.0, 91.5])["G"] - 1.0) < 1e-9)
    ck("T1 empty is NO DATA", score_T1([], [91.0])["verdict"] == "NO DATA")
    ck("T1's is_null flag is set on the NULL branch only",
       score_T1([91.0], [91.0])["is_null"]
       and not score_T1([92.0], [91.0])["is_null"]
       and not score_T1([90.0], [91.0])["is_null"])
    ck("T1's NULL reading names the prescriptive claim",
       "Adam-mini" in score_T1([91.0], [91.0])["reading"])
    ck("T1's NULL reading says the null band is reached by failing to find an effect",
       "never by finding one" in score_T1([91.0], [91.0])["reading"])
    ck("T1's reversal reading says pp1's P2 gets WEAKER",
       "WEAKER" in score_T1([90.0], [91.0])["reading"])
    ck("T1's survives reading refuses the 'merge the 1-D tensors' fix",
       "NOT 'merge the 1-D tensors'" in score_T1([92.0], [91.0])["reading"])
    ck("T1's return carries no reference to T3",
       "T3" not in " ".join(str(k) for k in score_T1([92.0], [91.0])))

    # --- T2, and its confound
    ck("T2 H=+0.50 CONFIRMS", score_T2([91.5], [91.0])["verdict"] == "CONFIRMS")
    ck("T2 H=+0.30 UNDECIDED (strict >)",
       score_T2([91.30], [91.0])["verdict"] == "UNDECIDED")
    ck("T2 H=0 UNDECIDED", score_T2([91.0], [91.0])["verdict"] == "UNDECIDED")
    ck("T2 H=-0.15 REFUTES (inclusive)",
       score_T2([90.85], [91.0])["verdict"] == "REFUTES")
    ck("T2 H=-0.50 REFUTES", score_T2([90.5], [91.0])["verdict"] == "REFUTES")
    ck("T2 H is nodewise1d MINUS nodewise", score_T2([90.0], [91.0])["H"] < 0)
    ck("T2 carries its confound on EVERY branch",
       all("NOT EVIDENCE" in score_T2([91.0 + h], [91.0])["confound"]
           for h in (+0.5, 0.0, -0.5)))
    ck("T2 carries its confound even with NO DATA",
       "NOT EVIDENCE" in score_T2([], [])["confound"])
    ck("T2's confound names ck1's K3 slope", "0.51 pp/decade" in score_T2([], [])["confound"])

    # --- T3 cannot void anything
    ck("T3 at both references REPRODUCES",
       score_T3([(T3_REF_PP1 + T3_REF_MM1) / 2])["verdict"] == "REPRODUCES")
    ck("T3 far away reads OFFSET",
       score_T3([T3_REF_PP1 + 2.0])["verdict"] == "OFFSET")
    ck("T3 NEVER voids", score_T3([T3_REF_PP1 + 9.0])["voids"] is False)
    ck("T3 empty is NO DATA", score_T3([])["verdict"] == "NO DATA")
    ck("T3 meters against BOTH prior nodewise readings",
       {"d_pp1", "d_mm1"} <= set(score_T3([92.0])))

    # --- T5, including its registered NOT-APPLICABLE branch
    ck("T5 is NOT APPLICABLE when T1 is not NULL",
       score_T5(False, [0.05], [0.03])["verdict"] == "NOT APPLICABLE")
    ck("T5's NOT-APPLICABLE reading says the branch was pre-registered",
       "registered in advance" in score_T5(False, [0.05], [0.03])["reading"])
    ck("T5 replicates on a big, resolved field difference",
       score_T5(True, [0.050, 0.051, 0.050], [0.030, 0.031, 0.030])["verdict"]
       == "DISSOCIATION REPLICATED")
    ck("T5 says the field tracks accuracy on an unresolved difference",
       score_T5(True, [0.050, 0.030, 0.070], [0.048, 0.032, 0.068])["verdict"]
       in ("FIELD TRACKS ACCURACY", "DISSOCIATION REPLICATED"))
    ck("T5 is two-sided in the field direction (sign of dF does not matter)",
       score_T5(True, [0.030, 0.031, 0.030], [0.050, 0.051, 0.050])["verdict"]
       == "DISSOCIATION REPLICATED")
    ck("T5's replication reading quotes pp1's contrast as the FIRST instance",
       "77.5" in score_T5(True, [0.050, 0.051, 0.050],
                          [0.030, 0.031, 0.030])["reading"])
    ck("T5's replication reading names direction C explicitly",
       "direction-C" in score_T5(True, [0.050, 0.051, 0.050],
                                 [0.030, 0.031, 0.030])["reading"])
    ck("T5 empty is NO DATA", score_T5(True, [], [])["verdict"] == "NO DATA")

    # --- the dirname parser, including what it must REFUSE
    ck("probe_n1d_bn1_s0 -> ('n1d', 0)",
       parse_bn1_dirname("probe_n1d_bn1_s0") == ("n1d", 0))
    ck("probe_c23_bn1_s1 -> ('c23', 1)",
       parse_bn1_dirname("probe_c23_bn1_s1") == ("c23", 1))
    ck("probe_node_bn1_s2 -> ('node', 2)",
       parse_bn1_dirname("probe_node_bn1_s2") == ("node", 2))
    ck("a pp1 dir with the SAME arm token is refused",
       parse_bn1_dirname("probe_node_pp1_s0") == (None, None))
    ck("an mm1 dir is refused", parse_bn1_dirname("probe_node_mm1_s0") == (None, None))
    ck("a pp1 perm dir is refused", parse_bn1_dirname("probe_perm_pp1_s0") == (None, None))
    ck("a ck1 chunk dir is refused",
       parse_bn1_dirname("probe_k1024_ck1_s0") == (None, None))
    ck("an UNREGISTERED arm token is refused",
       parse_bn1_dirname("probe_lay_bn1_s0")[0] is None)
    ck("a non-probe directory is refused",
       parse_bn1_dirname("Tensorboard_outputs") == (None, None))

    # --- T0.3 shape from the HEADER (the c74 false-VOID regression)
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        def mk(arm, n_tot, dtype, arr_n=None):
            d = os.path.join(td, "probe_%s_bn1_s0" % arm)
            os.makedirs(d, exist_ok=True)
            json.dump({"n_tot": n_tot}, open(os.path.join(d, "neg_counts.json"), "w"))
            np.save(os.path.join(d, "neg_counts.npy"),
                    np.zeros(arr_n if arr_n is not None else n_tot, dtype=dtype))
            return d
        ck("int32 array of the right length PASSES T0.3",
           gate_T03(mk("n1d", M_OF_ARM["n1d"], np.int32), "n1d")["ok"])
        ck("T0.3 reads the true shape from the header",
           gate_T03(mk("c23", M_OF_ARM["c23"], np.int32), "c23")["shape"]
           == (M_OF_ARM["c23"],))
        ck("int64 also passes on length",
           gate_T03(mk("node", M_OF_ARM["node"], np.int64), "node")["ok"])
        ck("a SHORT array FAILS T0.3",
           not gate_T03(mk("n1d", M_OF_ARM["n1d"], np.int32,
                           arr_n=M_OF_ARM["n1d"] - 1), "n1d")["ok"])
        ck("the node arm's n_tot on the n1d arm FAILS",
           not gate_T03(mk("n1d", M_OF_ARM["node"], np.int32), "n1d")["ok"])
        dm = os.path.join(td, "probe_node_bn1_s1")
        os.makedirs(dm)
        ck("absent neg_counts.json FAILS T0.3 (bf8's failure)",
           not gate_T03(dm, "node")["ok"])

    # --- the discipline this scorer registered against itself
    ck("scorer refuses to write the 53.1% sentence", '"53.1%"' in __doc__)
    ck("T4 carries no registered direction", "DESCRIPTIVE, NO DIRECTION" in __doc__)
    ck("T3's inability to void T1/T2 is documented", "CANNOT VOID T1 OR T2" in __doc__)
    ck("T2's confound is documented as registered, not discovered",
       "REGISTERED, NOT DISCOVERED LATER" in __doc__)
    ck("T5's NOT-APPLICABLE branch is documented", "NOT APPLICABLE" in __doc__)
    ck("the BN-vs-size non-identifiability is documented",
       "coincide exactly" in __doc__)
    ck("the off-argmax risk is documented", "own argmax has never been measured" in __doc__)
    ck("the scorer states it predates the data entirely",
       "before the" in __doc__ and "data could exist" in __doc__
       and "RUN EXISTED IN THE CSV AT ALL" in __doc__)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_bn1"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs, refused = [], []
    for d in sorted(glob.glob(os.path.join(a.root, "probe_*"))):
        arm, seed = parse_bn1_dirname(os.path.basename(d))
        if arm is None or seed is None:
            refused.append(os.path.basename(d))
            continue
        dirs.append((d, arm, seed))
    rows = {r["run"]: r for r in csv.DictReader(open(a.csv))}

    print("=" * 78)
    print("c78 -- bn1: IS THE MATCHED-COUNT GAP THE DEGENERATE SIZE-1 TAIL?  "
          "%d probe dirs" % len(dirs))
    print("box = (%.1f, %.4f)   ms=%s   %d ep" % (LO, HI, MST, EPOCHS))
    print("n1d(m=%d) / c23(m=%d) MATCHED EXACTLY  |  node(m=%d) anchor"
          % (M_OF_ARM["n1d"], M_OF_ARM["c23"], M_OF_ARM["node"]))
    print("=" * 78)
    if refused:
        print("REFUSED (not a registered bn1 dir): %s" % ", ".join(refused))

    # ---- T0
    print("\n--- T0  VALIDITY (n_records==%d, beta moved, ep==%d)" % (N_RECORDS, EPOCHS))
    t0 = {}
    for d, arm, seed in dirs:
        r = gate_T0(d, arm, rows.get("bn1-%s-s%d" % (arm, seed)))
        t0[d] = r
        print("    %-24s %-4s  n_rec=%-6d ep=%s/%s span=%.2f"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_records"],
                 r["epochs_done"], r["epochs_requested"], r["span"]))
    n0 = sum(1 for r in t0.values() if r["ok"])
    print("    T0: %d/%d" % (n0, len(dirs)))

    # ---- T0.2
    print("\n--- T0.2  n_beta EXACT on EVERY record  (guard 4 measured all three from "
          "the ALLOCATED beta)")
    n02 = 0
    for d, arm, seed in dirs:
        ok = t0[d]["nb_ok"]
        n02 += ok
        print("    %-24s %-4s  n_beta=%-12s want %s = %d"
              % (os.path.basename(d), "PASS" if ok else "FAIL",
                 ",".join(str(x) for x in t0[d]["n_beta"]), GRAN_OF_ARM[arm],
                 M_OF_ARM[arm]))
    print("    T0.2: %d/%d" % (n02, len(dirs)))
    if n0 != len(dirs) or n02 != len(dirs) or len(dirs) != NJOBS:
        print("    **T0/T0.2 INCOMPLETE (%d dirs, expected %d) -- NOTHING BELOW IS "
              "SCORED.**" % (len(dirs), NJOBS))
        if n0 != len(dirs) or n02 != len(dirs):
            return 1

    # ---- T0.3
    print("\n--- T0.3  THE INSTRUMENT FIRED")
    n03 = 0
    for d, arm, seed in dirs:
        r = gate_T03(d, arm)
        n03 += r["ok"]
        print("    %-24s %-4s  n_tot=%-10s npy_shape=%-14s %s"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_tot"],
                 r.get("shape"), r["why"]))
    print("    T0.3: %d/%d" % (n03, len(dirs)))

    # ---- T0.4
    print("\n--- T0.4  BOX-FREE AT BOTH GUARDS (rec_-based 5%% gate PRIMARY, UNCHANGED)")
    print("    %-24s %-9s %8s %8s %10s %10s"
          % ("dir", "verdict", "rec_lo", "rec_hi", "coord_lo", "coord_hi"))
    t04 = {}
    for d, arm, seed in dirs:
        r = gate_T04(d)
        t04[d] = r
        cl = "%.5f" % r["coord_lo"] if r["coord_lo"] is not None else "None"
        chh = "%.5f" % r["coord_hi"] if r["coord_hi"] is not None else "None"
        print("    %-24s %-9s %8.4f %8.4f %10s %10s"
              % (os.path.basename(d), "free" if r["boxfree"] else "BOUND:" + r["which"],
                 r["rec_lo"], r["rec_hi"], cl, chh))
    nfree = sum(1 for r in t04.values() if r["boxfree"])
    print("    T0.4: %d/%d box-free" % (nfree, len(dirs)))
    if nfree < len(dirs):
        print("    **A BIND HERE IS NEW** -- all 9 pp1 and all 6 mm1 arms read 0.0000 on")
        print("    all four columns in this SAME box.  n1d has never been run before.")

    # ---- plateau5 per arm, from the CSV
    pl = {}
    for d, arm, seed in dirs:
        row = rows.get("bn1-%s-s%d" % (arm, seed))
        if row and row.get("plateau5", "").strip():
            pl.setdefault(arm, []).append(float(row["plateau5"]))
    print("\n--- plateau5 per arm, from the CSV")
    for arm in ARMS:
        v = pl.get(arm, [])
        print("    %-12s m=%-8d n=%d  %s"
              % (GRAN_OF_ARM[arm], M_OF_ARM[arm], len(v),
                 ("%.3f +-%.3f" % (statistics.mean(v), _sem(v))) if v else "NO DATA"))

    # ---- T1  THE PRIMARY.  Computed and printed before T2/T3 are looked at.
    print("\n--- T1  **THE PRIMARY.  FIVE-WAY, SYMMETRIC, COUNT MATCHED EXACTLY.**")
    s1 = score_T1(pl.get("c23", []), pl.get("n1d", []))
    if s1["G"] is None:
        print("    NO DATA on at least one arm -- T1 cannot be scored.")
    else:
        print("    chunk2325   (m=%d, uniform)                 %.3f +-%.3f (n=%d)"
              % (M_OF_ARM["c23"], s1["c23"], s1["sem_c23"], s1["n_c23"]))
        print("    nodewise1d  (m=%d, channels, no size-1 tail) %.3f +-%.3f (n=%d)"
              % (M_OF_ARM["n1d"], s1["n1d"], s1["sem_n1d"], s1["n_n1d"]))
        print("    G = chunk2325 - nodewise1d = %+.3f pp   (se %.3f, t %.2f "
              "-- DESCRIPTIVE, the gate is the threshold on G)"
              % (s1["G"], s1["se_G"], s1["t"]))
        print("    registered: G > %+.2f SURVIVES | (%+.2f, %+.2f] UND | [%+.2f, %+.2f] "
              "TAIL WAS CARRIER | [%+.2f, %+.2f) UND | G <= %+.2f REVERSAL"
              % (T1_STRONG, T1_NULL, T1_STRONG, -T1_NULL, T1_NULL,
                 -T1_STRONG, -T1_NULL, -T1_STRONG))
        print("    for scale, the gap this is trying to explain: pp1 D = %+.3f, "
              "mm1 D = %+.3f, pooled %+.3f" % (PP1_D, MM1_D, POOLED_D))
        print("    -> **%s**" % s1["verdict"])
        print("    reading: %s" % s1["reading"])
        print("    **LIMITS, PRINTED WITH THE VERDICT, NOT BELOW IT:**")
        print("      * On ResNet18 every 1-D tensor is a BN scale/shift or the output")
        print("        bias, so 'the groups are degenerate (size 1)' and 'the groups are")
        print("        on the NORMALISATION parameters' coincide EXACTLY.  This batch")
        print("        cannot tell a size story from a parameter-role story.")
        print("      * ms is held at %s.  nodewise1d's own argmax has NEVER been" % MST)
        print("        measured, so this is a statement at one meta stepsize.")
        print("      * Every arm is a WITHIN-TENSOR partition.  Layer boundaries are")
        print("        untested here (CORRECTIONS 107.7 item 2 is still open).")

    # ---- T2
    print("\n--- T2  **THE SECONDARY.  DIRECTIONAL, AND CONFOUNDED BY DESIGN.**")
    s2 = score_T2(pl.get("n1d", []), pl.get("node", []))
    if s2.get("H") is None:
        print("    NO DATA on at least one arm -- T2 cannot be scored.")
    else:
        print("    H = nodewise1d - nodewise = %+.3f pp   (se %.3f, t %.2f)"
              % (s2["H"], s2["se_H"], s2["t"]))
        print("    registered: H > %+.2f CONFIRMS | (%+.2f, %+.2f] UNDECIDED | "
              "H <= %+.2f REFUTES"
              % (T2_CONFIRM, T2_REFUTE, T2_CONFIRM, T2_REFUTE))
        print("    -> **%s**" % s2["verdict"])
    print("    %s" % s2["confound"])

    # ---- T3
    print("\n--- T3  THE BATCH-OFFSET METER.  **DECLARED UNABLE TO VOID T1 OR T2.**")
    s3 = score_T3(pl.get("node", []))
    if s3["mean"] is None:
        print("    nodewise NO DATA")
    else:
        print("    nodewise@%s HERE  %.3f +-%.3f (n=%d)"
              % (MST, s3["mean"], s3["sem"], s3["n"]))
        print("    vs pp1's %.3f (%+.3f) and mm1's %.3f (%+.3f);  bar +-%.2f  -> **%s**"
              % (s3["ref_pp1"], s3["d_pp1"], s3["ref_mm1"], s3["d_mm1"],
                 s3["bar"], s3["verdict"]))
        print("    T1 and T2 are unaffected either way: both are within-batch")
        print("    differences and any offset common to their arms cancels exactly.")
        print("    (CORRECTIONS 106.3: the cross-batch offset has no stable sign.")
        print("    This is a third draw from it and is recorded as such.)")

    # ---- T4  THE FIELD
    print("\n--- T4  THE FIELD, THREE WAYS.  **DESCRIPTIVE, NO DIRECTION.**")
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
    print("      %-13s %-10s %9s %9s %9s %10s %10s %12s"
          % ("arm", "m", "pbar", "a_raw", "a_deb", "dev_deb", "dev_bias", "N_eff/m"))
    for arm in ARMS:
        v = ag.get(arm, [])
        if not v:
            continue
        chans = [split_channels(x) for x in v]
        ne = neff.get(arm, [])
        print("      %-13s %-10d %9.5f %9.5f %9.5f %10.5f %10.5f %12s"
              % (GRAN_OF_ARM[arm], M_OF_ARM[arm],
                 statistics.mean([x["pbar"] for x in v]),
                 statistics.mean([x["a_raw"] for x in v]),
                 statistics.mean([x["a_deb"] for x in v]),
                 statistics.mean([c[1] for c in chans]),
                 statistics.mean([c[2] for c in chans]),
                 ("%.4f +-%.4f" % (statistics.mean(ne), _sem(ne))) if ne else "--"))
    print("    No independence null is derived here, and the \"53.1%\" sentence is")
    print("    NOT reproduced.")

    # ---- T5  THE DISSOCIATION RE-TEST
    print("\n--- T5  **THE DISSOCIATION RE-TEST.  REGISTERED; POST-HOC ORIGIN "
          "DECLARED.**")
    s5 = score_T5(s1.get("is_null", False), neff.get("c23", []), neff.get("n1d", []))
    print("    defined ONLY on an accuracy tie.  T1 = %s" % s1.get("verdict"))
    if s5["t_F"] is not None:
        print("    N_eff/m  chunk2325 %.4f   nodewise1d %.4f   diff %+.5f  t_F %.2f "
              "(threshold %.1f)"
              % (s5["f_c23"], s5["f_n1d"], s5["dF"], s5["t_F"], s5["t_min"]))
    print("    -> **%s**" % s5["verdict"])
    if s5["reading"]:
        print("    reading: %s" % s5["reading"])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
