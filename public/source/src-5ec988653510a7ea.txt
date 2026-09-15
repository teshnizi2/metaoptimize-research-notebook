#!/usr/bin/env python3
"""c76_mm1_score.py -- score `mm1`, THE MATCHED-COUNT PARTITION CONTRAST.

REGISTERED GATES, transcribed from `bin/c76_matched_m_partition.sh:26-83`.  Every
constant below is asserted by the selftest against THAT SCRIPT'S OWN TEXT so the
registration and the batch cannot drift -- STANDING RULE (19).

WRITTEN AND SELFTESTED BEFORE ANY mm1 VERDICT WAS READ.

  M0    VALIDITY.  n_records == 10000; beta moved; epochs_done == requested == 100.
  M0.2  n_beta EXACT on EVERY record: chunk777 14421, nodewise 14420.  These are
        MEASURED values -- guard 4 read both from the ALLOCATED beta on the real
        built network before anything was submitted.  Neither is a comment.
  M0.3  THE INSTRUMENT FIRED.  neg_counts.json present, n_tot == n_beta, npy shape
        READ FROM ITS HEADER == (n_tot,).  (Header, never file size -- that
        inference produced c74's false VOID on 12/12 arms.)
  M0.4  BOX-FREE at BOTH guards, the PUBLISHED rec_-based 5% gate, PRIMARY and
        UNCHANGED.  coord_* reported ALONGSIDE and NEVER as the gate.  tw0's
        nodewise@1e-4 was box-free 3/3 and all 15 ck1 arms were box-free in this
        SAME box, so a bind here would be NEW and is reported as such.

  M1    **THE PRIMARY, AND IT IS WITHIN-BATCH.  THREE-WAY, REGISTERED.**
        D = plateau5(chunk777) - plateau5(nodewise), BOTH measured in this batch.
          D >  +0.30      -> **CONFIRMS**.  Partition structure matters at matched
                             count; c76's +0.517 pp residual is real and is not an
                             artifact of the interpolation or the cross-batch
                             offset.  The architecture-AGNOSTIC partition beats the
                             architecture-ALIGNED one.
          D <= +0.15      -> **REFUTES** (INCLUDING NEGATIVE).  K2's +0.565 was
                             carried by the 24% count difference and/or the batch
                             offset, and "granularity == the group count" survives.
          +0.15 < D <= +0.30 -> **UNDECIDED, REGISTERED IN ADVANCE**, so a result
                             landing in the gap cannot be squeezed into either
                             verdict after the fact.
        The band is not symmetric and that is deliberate; it is written down here
        exactly as the batch script wrote it before the data existed.

  M2    THE BATCH-OFFSET METER.  nodewise@1e-4 measured HERE vs tw0's 91.961,
        bar +-0.50.  **THIS CANNOT VOID M1** -- M1 is a within-batch difference and
        is immune to any offset common to both arms.  M2 measures the offset for
        the RECORD; a failure voids only the comparability of this batch's ABSOLUTE
        levels with tw0's, which M1 does not use.  This scorer therefore computes
        M1 BEFORE M2 and never consults M2's result when reporting M1.

  M3    THE FIELD AT MATCHED COUNT.  **DESCRIPTIVE, NO DIRECTION REGISTERED.**
        a_raw(chunk777) vs a_raw(nodewise) with the count difference removed
        entirely (one group in 14,420).  ck1's K4b saw the field move with the
        partition (-0.01556 at nearly fixed m); this is that comparison with the
        confound gone.  No prior measurement at EXACTLY matched m exists, so no
        direction is registered and none may be inferred afterwards.
        **The "53.1%" sentence is NOT reproduced and must NOT be written.**

WHAT THIS SCORER WILL NOT DO
  * It will not let M2 alter, gate or annotate M1's verdict.
  * It will not re-derive an independence null on the fly (CORRECTIONS 26).
  * It will not report a confirmed M1 as "architecture alignment matters".  At
    matched m the MEAN group size is identical by construction but the size
    DISTRIBUTION is not (nodewise heterogeneous, chunk uniform).  The supported
    sentence is "the partition matters"; separating the two needs a permuted
    partition carrying nodewise's exact size multiset.
  * It will not present a confirmed M1 as holding at each partition's own argmax.
    nodewise is read at ms=1e-4, NOT at its own 3e-4 -- deliberately, so M1 is
    K2's contrast de-confounded rather than a different contrast.

USAGE
  python3 analysis/c76_mm1_score.py --selftest
  python3 analysis/c76_mm1_score.py --root ../probes_mm1 --csv results/all_runs.csv
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
SCRIPT = os.path.join(REPO, "bin", "c76_matched_m_partition.sh")

from c52_boxfree import occupancy, records           # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -15.0, -2.3026             # `mm1`, registered in c55_neff_noise.BOXES
EPOCHS = 100
N_RECORDS = 10000
MST = "1e-4"
CHUNK_K = 777
ARMS = ("ch", "node")               # short tokens as the batch script names them
# n_beta per arm, MEASURED by guard 4 from the ALLOCATED beta.  Not a formula.
M_OF_ARM = {"ch": 14421, "node": 14420}
GRAN_OF_ARM = {"ch": "chunk777", "node": "nodewise"}
SEEDS = (0, 1, 2)
BOXFREE_MAX = 0.05                  # the PUBLISHED rec_-based 5% gate, UNCHANGED

M1_REFUTE_AT = 0.15                 # D <= this REFUTES, including negative
M1_CONFIRM_AT = 0.30                # D > this CONFIRMS; (0.15, 0.30] is UNDECIDED
# A BINARY-REPRESENTATION tolerance, NOT a widening of the registered band.
# 91.15 - 91.00 evaluates to 0.15000000000000568 in IEEE754, which would put an
# exactly-on-the-line result on the wrong side of a registered threshold for a
# reason that has nothing to do with the experiment.  1e-9 pp is SEVEN orders below
# the campaign's +-0.02 pp reproducibility floor, so it can only decide cases that
# are already ties, and it is fixed here BEFORE any mm1 number was read.
M1_EPS = 1e-9
M2_REF, M2_REF_N, M2_BAR = 91.961, 3, 0.50      # tw0 nodewise, 100 ep, ms=1e-4
K4B_PRIOR = -0.01556                # ck1's K4b, at NEARLY fixed m.  Context only.
C76_RESIDUAL = 0.517                # the interpolated partition residual M1 tests


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def parse_mm1_dirname(base):
    """'probe_ch_mm1_s0' -> ('ch', 0).  Returns (None, None) on anything else.

    probe5_window.parse_dirname cannot see chunk rungs and IT IS NOT PATCHED --
    five registered scorers depend on it.  Parsed here instead, and the arm token
    is required to be a REGISTERED arm so a stray directory can never enter.
    """
    parts = base.split("_")
    if not parts or parts[0] != "probe":
        return None, None
    parts = parts[1:]
    seed = None
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
        parts = parts[:-1]
    if "mm1" not in parts:
        return None, None
    arm = next((p for p in parts if p in M_OF_ARM), None)
    return arm, seed


# --- M0 ---------------------------------------------------------------------
def gate_M0(d, arm, row=None):
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


# --- M0.3 -------------------------------------------------------------------
def gate_M03(d, arm):
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


# --- M0.4 -------------------------------------------------------------------
def gate_M04(d, lo=LO, hi=HI):
    o = occupancy(d, lo, hi)
    boxfree = (o["rec_lo"] < BOXFREE_MAX) and (o["rec_hi"] < BOXFREE_MAX)
    return dict(boxfree=boxfree, rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                coord_lo=o["coord_lo"], coord_hi=o["coord_hi"],
                which=("hi" if o["rec_hi"] >= BOXFREE_MAX else
                       ("lo" if o["rec_lo"] >= BOXFREE_MAX else "")))


# --- M1 ---------------------------------------------------------------------
def score_M1(ch, node):
    """THE PRIMARY.  THREE-WAY: CONFIRMS / UNDECIDED / REFUTES, thresholds fixed
    before the data existed.  Within-batch, so no offset can enter it."""
    if not ch or not node:
        return dict(verdict="NO DATA", D=None, ch=None, node=None,
                    n_ch=len(ch), n_node=len(node),
                    refute_at=M1_REFUTE_AT, confirm_at=M1_CONFIRM_AT, reading="")
    mc, mn = statistics.mean(ch), statistics.mean(node)
    D = mc - mn
    if D > M1_CONFIRM_AT + M1_EPS:
        v = "CONFIRMS"
        reading = ("PARTITION STRUCTURE MATTERS AT MATCHED COUNT -- the +0.517 pp "
                   "residual survives a MEASURED matched-m contrast, with the "
                   "interpolation and the cross-batch offset both removed")
    elif D <= M1_REFUTE_AT + M1_EPS:
        v = "REFUTES"
        reading = ("K2's +0.565 was carried by the 24% count difference and/or the "
                   "cross-batch offset; 'granularity == the group count' survives "
                   "and 105.2-105.3 come off the board")
    else:
        v = "UNDECIDED"
        reading = ("landed in the band registered UNDECIDED IN ADVANCE (%+.2f, "
                   "%+.2f]; it is NOT read as either verdict"
                   % (M1_REFUTE_AT, M1_CONFIRM_AT))
    # descriptive only -- the gate is the threshold on D, not on t
    se = math.sqrt(_sem(ch) ** 2 + _sem(node) ** 2)
    t = (D / se) if se > 0 else float("inf")
    return dict(verdict=v, D=D, ch=mc, node=mn, sem_ch=_sem(ch), sem_node=_sem(node),
                n_ch=len(ch), n_node=len(node), se_D=se, t=t,
                refute_at=M1_REFUTE_AT, confirm_at=M1_CONFIRM_AT, reading=reading)


# --- M2 ---------------------------------------------------------------------
def score_M2(node):
    """THE OFFSET METER.  DECLARED UNABLE TO VOID M1 BEFORE THE DATA EXISTED."""
    if not node:
        return dict(verdict="NO DATA", mean=None, ref=M2_REF, diff=None, n=0,
                    sem=None, bar=M2_BAR, voids_M1=False)
    m = statistics.mean(node)
    diff = m - M2_REF
    return dict(mean=m, sem=_sem(node), n=len(node), ref=M2_REF, bar=M2_BAR,
                diff=diff, ok=(abs(diff) <= M2_BAR),
                verdict=("REPRODUCES" if abs(diff) <= M2_BAR else "OFFSET"),
                voids_M1=False)


# --- M3 ---------------------------------------------------------------------
def score_M3(a_ch, a_node):
    """NO confirm/refute verdict.  A reading at EXACTLY matched count."""
    if a_ch is None or a_node is None:
        return dict(reading="NO DATA", diff=None, prior=K4B_PRIOR)
    diff = a_ch - a_node
    return dict(a_ch=a_ch, a_node=a_node, diff=diff, prior=K4B_PRIOR,
                reading=("a_raw(chunk777) - a_raw(nodewise) = %+.5f at m matched to "
                         "one group in 14,420.  DESCRIPTIVE: no direction was "
                         "registered and none may be inferred now.  ck1's K4b read "
                         "%+.5f at NEARLY fixed m." % (diff, K4B_PRIOR)))


# ---------------------------------------------------------------------------
def selftest():
    n = p = 0
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""

    def ck(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("    %-66s %s" % (name, "ok" if cond else "FAIL"))

    print("c76_mm1_score selftest")
    # --- every constant against the batch script's own text (STANDING RULE 19)
    ck("batch script exists", bool(src))
    ck("script CLIP= is the registered box", "CLIP=-15:-2.3026" in src)
    ck("LO/HI match the script's CLIP", (LO, HI) == (-15.0, -2.3026))
    ck("script EPOCHS= matches", ("EPOCHS=%d" % EPOCHS) in src)
    ck("script MST= matches", ("MST=%s" % MST) in src)
    ck("script NJOBS is 6 = 2 arms x 3 seeds", "NJOBS=6" in src)
    ck("script CHUNK_K= matches", ("CHUNK_K=%d" % CHUNK_K) in src)
    ck("script M_CHUNK= is this scorer's chunk count",
       ("M_CHUNK=%d" % M_OF_ARM["ch"]) in src)
    ck("script M_NODE= is this scorer's nodewise count",
       ("M_NODE=%d" % M_OF_ARM["node"]) in src)
    ck("N_RECORDS = epochs*500/stride 5", N_RECORDS == EPOCHS * 500 // 5)
    ck("script exports PROBE=5", "PROBE=5" in src)
    ck("script exports PROBE5=1 (instrument, not stride)", "PROBE5=1" in src)
    ck("script's arm loop is chunk<K>:ch and nodewise:node",
       ('"chunk${CHUNK_K}:ch" "nodewise:node"') in src)
    ck("M1 confirm threshold +0.30 is in the script", "D > +0.30" in src)
    ck("M1 refute threshold +0.15 is in the script", "D <= +0.15" in src)
    ck("M1's UNDECIDED band is written in the script",
       "(+0.15, +0.30] IS REGISTERED IN ADVANCE AS UNDECIDED" in src)
    ck("M1's refute branch is declared to INCLUDE NEGATIVE",
       "INCLUDING NEGATIVE" in src)
    ck("M2 reference 91.961 is in the script", "91.961" in src)
    ck("M2 bar +-0.50 is in the script", "bar +-0.50" in src)
    ck("M2 is declared UNABLE TO VOID M1 in the script",
       "THIS CANNOT VOID M1" in src)
    ck("ck1's K4b prior -0.01556 is in the script", "-0.01556" in src)
    ck("the counts differ by exactly ONE group",
       abs(M_OF_ARM["ch"] - M_OF_ARM["node"]) == 1)
    ck("chunk777 is the LARGER count, so M1 is not m-favoured",
       M_OF_ARM["ch"] > M_OF_ARM["node"])
    ck("5% gate UNCHANGED", BOXFREE_MAX == 0.05)
    ck("seeds 0-2", SEEDS == (0, 1, 2))
    ck("M1 thresholds are ordered and asymmetric",
       M1_REFUTE_AT < M1_CONFIRM_AT)
    ck("M1_EPS is a rounding tolerance, orders below the 0.02 pp noise floor",
       0 < M1_EPS < 1e-6)
    ck("M1_EPS cannot move a result across the UNDECIDED band",
       M1_EPS < (M1_CONFIRM_AT - M1_REFUTE_AT) / 1e6)

    import c55_neff_noise as c55
    ck("mm1 registered in c55_neff_noise.BOXES", "mm1" in c55.BOXES)
    ck("c55's mm1 box == this scorer's box",
       c55.BOXES.get("mm1", (None,) * 3)[:2] == (LO, HI))
    ck("mm1's box is ck1's box, not a new one",
       c55.BOXES.get("mm1", (None,) * 3)[:2]
       == c55.BOXES.get("ck1", (None,) * 3)[:2])

    # --- M1: all three branches, at and around every boundary
    hi = score_M1([92.0], [91.0])                       # D = +1.00
    ck("M1 D=+1.00 CONFIRMS", hi["verdict"] == "CONFIRMS")
    ck("M1 confirming reading names the PARTITION",
       "PARTITION STRUCTURE MATTERS" in hi["reading"])
    ck("M1 D=+0.31 CONFIRMS (just over)",
       score_M1([91.31], [91.0])["verdict"] == "CONFIRMS")
    ck("M1 D=+0.30 is UNDECIDED, not a confirm (strict >)",
       score_M1([91.30], [91.0])["verdict"] == "UNDECIDED")
    ck("M1 D=+0.20 is UNDECIDED",
       score_M1([91.20], [91.0])["verdict"] == "UNDECIDED")
    ck("M1 D=+0.16 is UNDECIDED (just over the refute line)",
       score_M1([91.16], [91.0])["verdict"] == "UNDECIDED")
    ck("M1 D=+0.15 REFUTES (inclusive <=)",
       score_M1([91.15], [91.0])["verdict"] == "REFUTES")
    ck("M1 D=+0.05 REFUTES", score_M1([91.05], [91.0])["verdict"] == "REFUTES")
    ck("M1 D=0 REFUTES", score_M1([91.0], [91.0])["verdict"] == "REFUTES")
    ck("M1 D NEGATIVE REFUTES", score_M1([90.0], [91.0])["verdict"] == "REFUTES")
    ck("M1's UNDECIDED reading says so plainly",
       "UNDECIDED IN ADVANCE" in score_M1([91.2], [91.0])["reading"])
    ck("M1 refuting reading says 105.2-105.3 come off the board",
       "off the board" in score_M1([91.0], [91.0])["reading"])
    ck("M1 with an empty arm is NO DATA",
       score_M1([], [91.0])["verdict"] == "NO DATA")
    ck("M1 with an empty reference arm is NO DATA",
       score_M1([92.0], [])["verdict"] == "NO DATA")
    ck("M1 averages its seeds on both arms",
       abs(score_M1([92.0, 92.5], [91.0, 91.5])["D"] - 1.0) < 1e-9)
    ck("M1 D is chunk MINUS nodewise, in that order",
       score_M1([90.0], [91.0])["D"] < 0)
    ck("M1's t is descriptive and does not appear in the verdict",
       score_M1([92.0, 92.0, 92.0], [91.0, 91.0, 91.0])["verdict"] == "CONFIRMS")
    ck("M1 c76's +0.517 residual would CONFIRM if reproduced",
       score_M1([91.0 + C76_RESIDUAL], [91.0])["verdict"] == "CONFIRMS")

    # --- M2 cannot void M1, structurally
    ck("M2 exact reference reproduces", score_M2([M2_REF])["verdict"] == "REPRODUCES")
    ck("M2 +0.49 reproduces", score_M2([M2_REF + 0.49])["verdict"] == "REPRODUCES")
    ck("M2 +0.51 reads OFFSET", score_M2([M2_REF + 0.51])["verdict"] == "OFFSET")
    ck("M2 -0.51 reads OFFSET (two-sided)",
       score_M2([M2_REF - 0.51])["verdict"] == "OFFSET")
    ck("M2 NEVER voids M1", score_M2([M2_REF + 9.0])["voids_M1"] is False)
    ck("M2 empty is NO DATA", score_M2([])["verdict"] == "NO DATA")
    ck("M1's return carries no reference to M2",
       "M2" not in " ".join(str(k) for k in score_M1([92.0], [91.0])))

    # --- M3 carries NO verdict
    r = score_M3(0.52, 0.53)
    ck("M3 returns a reading, not a verdict", "verdict" not in r)
    ck("M3 diff is chunk MINUS nodewise", r["diff"] < 0)
    ck("M3 reading declares itself DESCRIPTIVE", "DESCRIPTIVE" in r["reading"])
    ck("M3 quotes ck1's K4b as context, not as a prediction",
       "NEARLY fixed m" in r["reading"])
    ck("M3 handles missing data", score_M3(None, 0.5)["reading"] == "NO DATA")

    # --- the dirname parser, including what it must REFUSE
    ck("probe_ch_mm1_s0 -> ('ch', 0)", parse_mm1_dirname("probe_ch_mm1_s0") == ("ch", 0))
    ck("probe_node_mm1_s2 -> ('node', 2)",
       parse_mm1_dirname("probe_node_mm1_s2") == ("node", 2))
    ck("a ck1 chunk dir is refused",
       parse_mm1_dirname("probe_k1024_ck1_s0") == (None, None))
    ck("a tw0 dir is refused", parse_mm1_dirname("probe_w_tw0_s0") == (None, None))
    ck("an at1 nodewise dir cannot enter this batch",
       parse_mm1_dirname("probe_node_at1_s0") == (None, None))
    ck("an UNREGISTERED arm token is refused",
       parse_mm1_dirname("probe_lay_mm1_s0")[0] is None)
    ck("a non-probe directory is refused",
       parse_mm1_dirname("Tensorboard_outputs") == (None, None))

    # --- M0.3 shape from the HEADER (the c74 false-VOID regression)
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        def mk(arm, n_tot, dtype, arr_n=None):
            d = os.path.join(td, "probe_%s_mm1_s0" % arm)
            os.makedirs(d, exist_ok=True)
            json.dump({"n_tot": n_tot}, open(os.path.join(d, "neg_counts.json"), "w"))
            np.save(os.path.join(d, "neg_counts.npy"),
                    np.zeros(arr_n if arr_n is not None else n_tot, dtype=dtype))
            return d
        ck("int32 array of the right length PASSES M0.3",
           gate_M03(mk("ch", M_OF_ARM["ch"], np.int32), "ch")["ok"])
        ck("M0.3 reads the true shape from the header",
           gate_M03(mk("ch", M_OF_ARM["ch"], np.int32), "ch")["shape"]
           == (M_OF_ARM["ch"],))
        ck("int64 also passes on length",
           gate_M03(mk("ch", M_OF_ARM["ch"], np.int64), "ch")["ok"])
        ck("a SHORT array FAILS M0.3",
           not gate_M03(mk("ch", M_OF_ARM["ch"], np.int32,
                           arr_n=M_OF_ARM["ch"] - 1), "ch")["ok"])
        ck("the OTHER arm's n_tot fails -- one group apart is still a mismatch",
           not gate_M03(mk("node", M_OF_ARM["ch"], np.int32), "node")["ok"])
        dm = os.path.join(td, "probe_node_mm1_s1")
        os.makedirs(dm)
        ck("absent neg_counts.json FAILS M0.3 (bf8's failure)",
           not gate_M03(dm, "node")["ok"])

    # --- the discipline this scorer registered against itself
    ck("scorer refuses to write the 53.1% sentence", '"53.1%"' in __doc__)
    ck("M3 carries no registered direction", "NO DIRECTION REGISTERED" in __doc__)
    ck("M2's inability to void M1 is documented", "CANNOT VOID M1" in __doc__)
    ck("the alignment claim is explicitly refused",
       "architecture alignment matters" in __doc__)
    ck("the off-argmax risk on nodewise is documented", "own 3e-4" in __doc__)
    ck("M1 is documented as WITHIN-BATCH", "WITHIN-BATCH" in __doc__)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_mm1"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs, refused = [], []
    for d in sorted(glob.glob(os.path.join(a.root, "probe_*"))):
        arm, seed = parse_mm1_dirname(os.path.basename(d))
        if arm is None or seed is None:
            refused.append(os.path.basename(d))
            continue
        dirs.append((d, arm, seed))
    rows = {r["run"]: r for r in csv.DictReader(open(a.csv))}

    print("=" * 78)
    print("c76 -- mm1: THE MATCHED-COUNT PARTITION CONTRAST.  %d probe dirs" % len(dirs))
    print("box = (%.1f, %.4f)   ms=%s   %d ep   chunk%d (m=%d) vs nodewise (m=%d)"
          % (LO, HI, MST, EPOCHS, CHUNK_K, M_OF_ARM["ch"], M_OF_ARM["node"]))
    print("=" * 78)
    if refused:
        print("REFUSED (not a registered mm1 dir): %s" % ", ".join(refused))

    # ---- M0
    print("\n--- M0  VALIDITY (n_records==%d, beta moved, ep==%d)"
          % (N_RECORDS, EPOCHS))
    m0 = {}
    for d, arm, seed in dirs:
        r = gate_M0(d, arm, rows.get("mm1-%s-s%d" % (arm, seed)))
        m0[d] = r
        print("    %-22s %-4s  n_rec=%-6d ep=%s/%s span=%.2f"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_records"],
                 r["epochs_done"], r["epochs_requested"], r["span"]))
    n0 = sum(1 for r in m0.values() if r["ok"])
    print("    M0: %d/%d" % (n0, len(dirs)))

    # ---- M0.2
    print("\n--- M0.2  n_beta EXACT on EVERY record  (guard 4 measured both from the "
          "ALLOCATED beta)")
    n02 = 0
    for d, arm, seed in dirs:
        r = m0[d]
        ok = r["nb_ok"]
        n02 += ok
        print("    %-22s %-4s  n_beta=%-12s want %s = %d"
              % (os.path.basename(d), "PASS" if ok else "FAIL",
                 ",".join(str(x) for x in r["n_beta"]), GRAN_OF_ARM[arm],
                 M_OF_ARM[arm]))
    print("    M0.2: %d/%d" % (n02, len(dirs)))
    if n0 != len(dirs) or n02 != len(dirs):
        print("    **M0/M0.2 FAILED -- NOTHING BELOW IS SCORED.**")
        return 1

    # ---- M0.3
    print("\n--- M0.3  THE INSTRUMENT FIRED")
    n03 = 0
    for d, arm, seed in dirs:
        r = gate_M03(d, arm)
        n03 += r["ok"]
        print("    %-22s %-4s  n_tot=%-10s npy_shape=%-14s %s"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_tot"],
                 r.get("shape"), r["why"]))
    print("    M0.3: %d/%d" % (n03, len(dirs)))

    # ---- M0.4
    print("\n--- M0.4  BOX-FREE AT BOTH GUARDS (rec_-based 5%% gate PRIMARY, UNCHANGED)")
    print("    %-22s %-9s %8s %8s %10s %10s"
          % ("dir", "verdict", "rec_lo", "rec_hi", "coord_lo", "coord_hi"))
    m04 = {}
    for d, arm, seed in dirs:
        r = gate_M04(d)
        m04[d] = r
        cl = "%.5f" % r["coord_lo"] if r["coord_lo"] is not None else "None"
        chh = "%.5f" % r["coord_hi"] if r["coord_hi"] is not None else "None"
        print("    %-22s %-9s %8.4f %8.4f %10s %10s"
              % (os.path.basename(d), "free" if r["boxfree"] else "BOUND:" + r["which"],
                 r["rec_lo"], r["rec_hi"], cl, chh))
    nfree = sum(1 for r in m04.values() if r["boxfree"])
    print("    M0.4: %d/%d box-free" % (nfree, len(dirs)))
    if nfree < len(dirs):
        print("    **A BIND HERE IS NEW** -- tw0's nodewise@1e-4 was box-free 3/3 and")
        print("    all 15 ck1 arms were box-free in this SAME box.  Reported as such.")

    # ---- plateau5 per arm, from the CSV
    pl = {}
    for d, arm, seed in dirs:
        row = rows.get("mm1-%s-s%d" % (arm, seed))
        if row and row.get("plateau5", "").strip():
            pl.setdefault(arm, []).append(float(row["plateau5"]))

    # ---- M1  THE PRIMARY.  Computed and printed BEFORE M2 is looked at.
    print("\n--- M1  **THE PRIMARY, WITHIN-BATCH.  THREE-WAY, REGISTERED IN ADVANCE.**")
    s1 = score_M1(pl.get("ch", []), pl.get("node", []))
    if s1["D"] is None:
        print("    NO DATA on at least one arm -- M1 cannot be scored.")
    else:
        print("    chunk777  (m=%d, uniform flat chunks)      %.3f +-%.3f (n=%d)"
              % (M_OF_ARM["ch"], s1["ch"], s1["sem_ch"], s1["n_ch"]))
        print("    nodewise  (m=%d, output channels)          %.3f +-%.3f (n=%d)"
              % (M_OF_ARM["node"], s1["node"], s1["sem_node"], s1["n_node"]))
        print("    D = chunk777 - nodewise = %+.3f pp     (se %.3f, t %.2f "
              "-- DESCRIPTIVE, the gate is the threshold on D)"
              % (s1["D"], s1["se_D"], s1["t"]))
        print("    registered: D > %+.2f CONFIRMS | (%+.2f, %+.2f] UNDECIDED | "
              "D <= %+.2f REFUTES"
              % (M1_CONFIRM_AT, M1_REFUTE_AT, M1_CONFIRM_AT, M1_REFUTE_AT))
        print("    -> **%s**" % s1["verdict"])
        print("    reading: %s" % s1["reading"])
        print("    (for scale: c76's interpolated partition residual was %+.3f pp)"
              % C76_RESIDUAL)
        if s1["verdict"] == "CONFIRMS":
            print("    **THE LIMIT, STATED WITH THE RESULT:** at matched m the MEAN")
            print("    group size is identical by construction, the size DISTRIBUTION")
            print("    is not.  Supported: 'the partition matters'.  NOT supported:")
            print("    'architecture ALIGNMENT matters'.  And this is a statement at")
            print("    ms=1e-4, not at each partition's own argmax (nodewise's is 3e-4).")

    # ---- M2  THE OFFSET METER.  Cannot and does not touch M1.
    print("\n--- M2  THE BATCH-OFFSET METER.  **DECLARED UNABLE TO VOID M1 IN ADVANCE.**")
    s2 = score_M2(pl.get("node", []))
    if s2["mean"] is None:
        print("    nodewise NO DATA")
    else:
        print("    nodewise@%s HERE  %.3f +-%.3f (n=%d)"
              % (MST, s2["mean"], s2["sem"], s2["n"]))
        print("    tw0's nodewise@%s  %.3f (n=%d)   diff %+.3f pp   bar +-%.2f  -> **%s**"
              % (MST, M2_REF, M2_REF_N, s2["diff"], M2_BAR, s2["verdict"]))
        print("    M1 is unaffected either way: it is a within-batch difference and")
        print("    any offset common to both arms cancels out of it exactly.")

    # ---- M3  THE FIELD AT MATCHED COUNT
    print("\n--- M3  THE FIELD AT MATCHED COUNT.  **DESCRIPTIVE, NO DIRECTION.**")
    from neff_instrument import agreement_stats
    import probe5_window as p5w
    ag, neff = {}, {}
    print("    %-22s %8s %10s %9s %9s %11s"
          % ("dir", "pbar", "bias", "a_raw", "a_deb", "bias_share"))
    for d, arm, seed in dirs:
        s = agreement_stats(d, window=(0.5, 1.0))
        if s is None:
            print("    %-22s  -- agreement_stats returned None" % os.path.basename(d))
            continue
        ag.setdefault(arm, []).append(s)
        print("    %-22s %8.5f %+10.5f %9.5f %9.5f %11.4f"
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
    print("      %-10s %-12s %9s %9s %9s %11s %12s"
          % ("arm", "m", "pbar", "a_raw", "a_deb", "bias_share", "N_eff/m"))
    for arm in ARMS:
        v = ag.get(arm, [])
        if not v:
            continue
        ne = neff.get(arm, [])
        print("      %-10s %-12d %9.5f %9.5f %9.5f %11.4f %12s"
              % (GRAN_OF_ARM[arm], M_OF_ARM[arm],
                 statistics.mean([x["pbar"] for x in v]),
                 statistics.mean([x["a_raw"] for x in v]),
                 statistics.mean([x["a_deb"] for x in v]),
                 statistics.mean([x["bias_share"] for x in v]),
                 ("%.4f +-%.4f" % (statistics.mean(ne), _sem(ne))) if ne else "--"))

    s3 = score_M3(
        statistics.mean([x["a_raw"] for x in ag["ch"]]) if ag.get("ch") else None,
        statistics.mean([x["a_raw"] for x in ag["node"]]) if ag.get("node") else None)
    print("\n    %s" % s3["reading"])
    print("    No independence null is derived here, and the \"53.1%\" sentence is")
    print("    NOT reproduced.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
