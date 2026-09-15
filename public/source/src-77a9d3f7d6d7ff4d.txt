#!/usr/bin/env python3
"""c76_cx2_score.py -- score `cx2`, THE CHUNK LADDER'S COARSE END.

REGISTERED GATES, transcribed from `bin/c76_chunk_coarse_extension.sh:23-63`.  Every
constant below is asserted by the selftest against THAT SCRIPT'S OWN TEXT so the
registration and the batch cannot drift -- STANDING RULE (19).

WRITTEN AND SELFTESTED BEFORE ANY cx2 VERDICT WAS READ.

  X0    VALIDITY.  n_records == 10000; beta moved; epochs_done == requested == 100.
  X0.2  n_beta == m(K) EXACTLY on EVERY record.  m(8192)=1,407 and m(65536)=220 were
        MEASURED by guard 4 from the ALLOCATED beta on the real built network before
        submission, and RE-measured rather than inherited from ck1's comment.
  X0.3  THE INSTRUMENT FIRED.  neg_counts.json present, n_tot == n_beta, npy shape
        READ FROM ITS HEADER == (n_tot,).
  X0.4  BOX-FREE at BOTH guards, the PUBLISHED rec_-based 5% gate, PRIMARY and
        UNCHANGED.  coord_* reported ALONGSIDE and NEVER as the gate.

  X1    **K3 EXTENDED.  REGISTERED.**  plateau5 monotone NON-DECREASING as K rises
        across 1024 -> 8192 -> 65536, ties within +-0.15 pp, CHAINED onto ck1's
        measured chunk1024 = 92.526.  REFUTES if an adjacent pair inverts by more
        than 0.15 pp -> the monotone rise K3 confirmed does NOT continue to the
        coarse end, and the chunk curve turns over somewhere inside 62..10,944.

  X2    **DOES THE CHUNK FAMILY OVERSHOOT LAYERWISE?  REGISTERED, ONE-SIDED.**
        Layerwise is the chunk family's own coarse endpoint (chunk<huge> ==
        layerwise BITWISE, m=62) and is the best rung of the five-rung tuned ladder
        at 92.887 (n=11, FINDINGS 74.3).
          both <= 92.887 + 0.15 -> **CONFIRMS**.  This is the ORDINARY outcome: the
                chunk curve converges to layerwise from below and the tuned ladder's
                interior optimum survives inside the chunk family.
                **This must NOT be reported as a positive finding.**
          either > 92.887 + 0.15 -> **REFUTES** -> the chunk family has an interior
                optimum ABOVE the best rung of the entire tuned ladder, reached by a
                partition knob that ignores architecture completely.  That would be a
                METHOD result and not only a measurement one, which is why the
                threshold is written down here before the data exists.
        The asymmetry is deliberate and is preserved by this scorer: score_X2
        returns `interesting=False` on the confirming branch.

  X3    THE FIELD OVER THE WHOLE RANGE.  **DESCRIPTIVE, NO DIRECTION REGISTERED.**
        a_raw, a_deb, pbar, bias_share and N_eff/m at m=1,407 and m=220, completing
        the chunk family's field curve over 62..11,173,962 with the partition held
        FIXED -- the first agreement curve in this campaign that varies only the
        group count within a single partition family.
        **The "53.1%" sentence is NOT reproduced and must NOT be written.**

WHAT THIS SCORER WILL NOT DO
  * It will not report a confirming X2 as a finding.  X2's confirming branch is
    declared uninteresting IN ADVANCE and the scorer prints that alongside it.
  * It will not re-derive an independence null on the fly (CORRECTIONS 26).
  * It will not treat the fall of a_raw with m as evidence of structure.  A coarse
    coordinate's meta-gradient is a sum over many fine ones, so the fall is PARTLY
    MECHANICAL; the curve is reported as a curve.
  * It will not hide the ms confound X1 inherits: ms=1e-4 is held across the whole
    ladder, and if an interior K's own argmax is away from 1e-4 then X1's SHAPE is
    confounded with distance-from-optimum.  ck1 stated this risk and it is unchanged.

USAGE
  python3 analysis/c76_cx2_score.py --selftest
  python3 analysis/c76_cx2_score.py --root ../probes_cx2 --csv results/all_runs.csv
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
SCRIPT = os.path.join(REPO, "bin", "c76_chunk_coarse_extension.sh")

from c52_boxfree import occupancy, records           # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -15.0, -2.3026             # `cx2`, registered in c55_neff_noise.BOXES
EPOCHS = 100
N_RECORDS = 10000
MST = "1e-4"
KS = (8192, 65536)                  # the rungs THIS batch runs
# m(K), MEASURED from the allocated beta on the built network.  Not a formula.
M_OF_K = {8192: 1407, 65536: 220}
SEEDS = (0, 1, 2)
BOXFREE_MAX = 0.05                  # the PUBLISHED rec_-based 5% gate, UNCHANGED

# X1's chained anchor: ck1's MEASURED chunk1024, the rung this ladder continues from.
X1_ANCHOR_K, X1_ANCHOR_M, X1_ANCHOR = 1024, 10944, 92.526
X1_TIE = 0.15

# X2's reference: layerwise, the chunk family's OWN coarse endpoint (m=62), and the
# best rung of the five-rung tuned ladder (FINDINGS 74.3).
X2_REF, X2_REF_N, X2_REF_M, X2_MARGIN = 92.887, 11, 62, 0.15

# A BINARY-REPRESENTATION tolerance, NOT a widening of any registered band.  1e-9 pp
# is seven orders below the campaign's +-0.02 pp reproducibility floor, so it can only
# decide cases that are already exact ties.  Fixed before any cx2 number was read.
EPS = 1e-9

# ck1's full chunk ladder, for X3's seven-rung continuum.  RE-DERIVED from the CSV at
# report time; this table is only the m(K) index, which guard 4 measured.
CK1_M_OF_K = {1: 11173962, 2: 5586981, 16: 698373, 128: 87303, 1024: 10944}


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def parse_cx2_dirname(base):
    """'probe_k8192_cx2_s0' -> (8192, 0).  (None, None) on anything else.

    K is required to be a REGISTERED cx2 K, so neither a ck1 chunk dir nor a stray
    directory can enter this ladder silently.
    """
    parts = base.split("_")
    if not parts or parts[0] != "probe":
        return None, None
    parts = parts[1:]
    seed = None
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
        parts = parts[:-1]
    if "cx2" not in parts:
        return None, None
    ktok = next((p for p in parts if p.startswith("k") and p[1:].isdigit()), None)
    if ktok is None:
        return None, None
    K = int(ktok[1:])
    return (K if K in M_OF_K else None), seed


# --- X0 ---------------------------------------------------------------------
def gate_X0(d, K, row=None):
    R = records(d)
    want = M_OF_K.get(K)
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
    return dict(K=K, n_records=len(R),
                n_beta=sorted(x for x in nb if x is not None),
                n_ok=n_ok, nb_ok=nb_ok, moved=moved, ep_ok=ep_ok,
                epochs_done=ep_d, epochs_requested=ep_r,
                span=max(maxs) - min(mins),
                ok=(n_ok and nb_ok and moved and (ep_ok is not False)))


# --- X0.3 -------------------------------------------------------------------
def gate_X03(d, K):
    j = os.path.join(d, "neg_counts.json")
    npy = os.path.join(d, "neg_counts.npy")
    if not os.path.exists(j):
        return dict(ok=False, why="neg_counts.json ABSENT -- instrument never fired",
                    n_tot=None, npy_ok=False, shape=None)
    meta = json.load(open(j))
    want = M_OF_K.get(K)
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


# --- X0.4 -------------------------------------------------------------------
def gate_X04(d, lo=LO, hi=HI):
    o = occupancy(d, lo, hi)
    boxfree = (o["rec_lo"] < BOXFREE_MAX) and (o["rec_hi"] < BOXFREE_MAX)
    return dict(boxfree=boxfree, rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                coord_lo=o["coord_lo"], coord_hi=o["coord_hi"],
                which=("hi" if o["rec_hi"] >= BOXFREE_MAX else
                       ("lo" if o["rec_lo"] >= BOXFREE_MAX else "")))


# --- X1 ---------------------------------------------------------------------
def score_X1(by_k, anchor=X1_ANCHOR):
    """{K: mean plateau5} for THIS batch, chained onto ck1's chunk1024.
    Monotone NON-DECREASING as K rises, ties within +-X1_TIE."""
    chain = [(X1_ANCHOR_K, anchor)] + [(k, by_k[k]) for k in KS
                                       if by_k.get(k) is not None]
    if len(chain) < 2:
        return dict(verdict="NO DATA", pairs=[], chain=chain, complete=False)
    pairs, bad = [], []
    for (ka, va), (kb, vb) in zip(chain, chain[1:]):
        d = vb - va
        inv = d < -(X1_TIE + EPS)
        pairs.append(dict(lo=ka, hi=kb, delta=d, inverts=inv))
        if inv:
            bad.append("chunk%d->chunk%d %+.3f" % (ka, kb, d))
    return dict(verdict=("CONFIRMS" if not bad else "REFUTES"), pairs=pairs,
                chain=chain, tie=X1_TIE, why="; ".join(bad),
                complete=(len(chain) == len(KS) + 1),
                anchored_on=X1_ANCHOR_K)


# --- X2 ---------------------------------------------------------------------
def score_X2(by_k):
    """ONE-SIDED.  CONFIRMS is the ORDINARY outcome and is flagged uninteresting."""
    have = {k: by_k[k] for k in KS if by_k.get(k) is not None}
    if not have:
        return dict(verdict="NO DATA", over=[], ceiling=X2_REF + X2_MARGIN,
                    interesting=False, reading="")
    ceiling = X2_REF + X2_MARGIN
    over = [k for k, v in have.items() if v > ceiling + EPS]
    if over:
        return dict(verdict="REFUTES", over=sorted(over), ceiling=ceiling,
                    have=have, interesting=True,
                    reading=("THE CHUNK FAMILY HAS AN INTERIOR OPTIMUM ABOVE THE BEST "
                             "RUNG OF THE WHOLE TUNED LADDER, reached by a partition "
                             "knob that ignores architecture -- a METHOD result, not "
                             "only a measurement one"))
    return dict(verdict="CONFIRMS", over=[], ceiling=ceiling, have=have,
                interesting=False,
                reading=("the chunk curve converges to layerwise from BELOW and the "
                         "tuned ladder's interior optimum at layerwise survives "
                         "inside the chunk family.  **THIS IS THE ORDINARY OUTCOME, "
                         "REGISTERED AS UNINTERESTING IN ADVANCE, AND IS NOT A "
                         "POSITIVE FINDING.**"))


def slope_log_neff(ms, neffs):
    """d log10 N_eff / d log10 m over >= 3 distinct m.  NaN otherwise."""
    x, y = [], []
    for m, n in zip(ms, neffs):
        if m and n and m > 0 and n > 0 and math.isfinite(m) and math.isfinite(n):
            x.append(math.log10(m))
            y.append(math.log10(n))
    if len(set(x)) < 3:
        return float("nan")
    import numpy as np
    return float(np.polyfit(np.asarray(x), np.asarray(y), 1)[0])


# ---------------------------------------------------------------------------
def selftest():
    n = p = 0
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""

    def ck(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("    %-66s %s" % (name, "ok" if cond else "FAIL"))

    print("c76_cx2_score selftest")
    # --- every constant against the batch script's own text (STANDING RULE 19)
    ck("batch script exists", bool(src))
    ck("script CLIP= is the registered box", "CLIP=-15:-2.3026" in src)
    ck("LO/HI match the script's CLIP", (LO, HI) == (-15.0, -2.3026))
    ck("script EPOCHS= matches", ("EPOCHS=%d" % EPOCHS) in src)
    ck("script MST= matches", ("MST=%s" % MST) in src)
    ck("script NJOBS is 6 = 2 K x 3 seeds", "NJOBS=6" in src)
    ck("script KS= is this scorer's rungs",
       ('KS="%s"' % " ".join(str(k) for k in KS)) in src)
    ck("N_RECORDS = epochs*500/stride 5", N_RECORDS == EPOCHS * 500 // 5)
    ck("script exports PROBE=5", "PROBE=5" in src)
    ck("script exports PROBE5=1 (instrument, not stride)", "PROBE5=1" in src)
    ck("script's WANT table is this scorer's M_OF_K",
       "WANT = {8192: 1407, 65536: 220}" in src)
    ck("M_OF_K matches that table literally", M_OF_K == {8192: 1407, 65536: 220})
    ck("X1 anchor 92.526 is in the script", "92.526" in src)
    ck("X1 tie band 0.15 is in the script", "+-0.15 pp" in src)
    ck("X2 reference 92.887 is in the script", "92.887" in src)
    ck("X2 reference n=11 is in the script", "92.887 (n=11" in src)
    ck("X2's margin is written as 92.887 + 0.15 in the script",
       "92.887 + 0.15" in src)
    ck("X2 is declared ONE-SIDED in the script", "X2 is ONE-SIDED" in src)
    ck("the script declares the confirming branch uninteresting",
       "must not be reported as a positive finding" in src)
    ck("layerwise is the chunk family's own endpoint, per the script",
       "chunk<huge> == layerwise BITWISE" in src)
    ck("5% gate UNCHANGED", BOXFREE_MAX == 0.05)
    ck("seeds 0-2", SEEDS == (0, 1, 2))
    ck("m(K) is strictly decreasing in K",
       all(M_OF_K[a] > M_OF_K[b] for a, b in zip(KS, KS[1:])))
    ck("both cx2 rungs are COARSER than ck1's chunk1024",
       all(M_OF_K[k] < X1_ANCHOR_M for k in KS))
    ck("both cx2 rungs are FINER than layerwise's 62",
       all(M_OF_K[k] > X2_REF_M for k in KS))
    ck("the anchor's m is ck1's chunk1024 m", X1_ANCHOR_M == CK1_M_OF_K[1024])
    ck("EPS is a rounding tolerance, orders below the 0.02 pp noise floor",
       0 < EPS < 1e-6)

    import c55_neff_noise as c55
    ck("cx2 registered in c55_neff_noise.BOXES", "cx2" in c55.BOXES)
    ck("c55's cx2 box == this scorer's box",
       c55.BOXES.get("cx2", (None,) * 3)[:2] == (LO, HI))
    ck("cx2's box is ck1's box, not a new one",
       c55.BOXES.get("cx2", (None,) * 3)[:2]
       == c55.BOXES.get("ck1", (None,) * 3)[:2])

    # --- X1: chained, monotone in K, with the tie band
    up = {8192: 92.7, 65536: 92.9}
    ck("X1 rising confirms", score_X1(up)["verdict"] == "CONFIRMS")
    ck("X1 flat at the anchor confirms",
       score_X1({8192: X1_ANCHOR, 65536: X1_ANCHOR})["verdict"] == "CONFIRMS")
    ck("X1 a 0.14 dip is inside the tie band",
       score_X1({8192: X1_ANCHOR - 0.14, 65536: X1_ANCHOR})["verdict"] == "CONFIRMS")
    ck("X1 a 0.16 dip from the ANCHOR refutes",
       score_X1({8192: X1_ANCHOR - 0.16, 65536: X1_ANCHOR})["verdict"] == "REFUTES")
    ck("X1 a 0.16 dip BETWEEN the new rungs refutes",
       score_X1({8192: 92.9, 65536: 92.74})["verdict"] == "REFUTES")
    ck("X1 names the inverting pair",
       "chunk8192->chunk65536" in score_X1({8192: 92.9, 65536: 92.74})["why"])
    ck("X1 really is CHAINED onto ck1's chunk1024",
       score_X1(up)["pairs"][0]["lo"] == X1_ANCHOR_K)
    ck("X1 checks in K order, not dict order",
       [(q["lo"], q["hi"]) for q in score_X1(up)["pairs"]]
       == [(1024, 8192), (8192, 65536)])
    ck("X1 flags an incomplete ladder",
       score_X1({8192: 92.7})["complete"] is False)
    ck("X1 on both rungs is complete", score_X1(up)["complete"] is True)
    ck("X1 with no rungs at all is NO DATA", score_X1({})["verdict"] == "NO DATA")
    ck("X1's anchor is a plain rung of the SAME family, not a foreign reference",
       X1_ANCHOR_K not in KS and X1_ANCHOR_M in CK1_M_OF_K.values())

    # --- X2: one-sided, and its confirming branch is flagged uninteresting
    lowb = {8192: 92.5, 65536: 92.7}
    ck("X2 both below the ceiling CONFIRMS", score_X2(lowb)["verdict"] == "CONFIRMS")
    ck("X2's confirming branch is flagged UNINTERESTING",
       score_X2(lowb)["interesting"] is False)
    ck("X2's confirming reading says it is not a positive finding",
       "NOT A POSITIVE FINDING" in score_X2(lowb)["reading"])
    ck("X2 exactly AT the ceiling still CONFIRMS (<= is inclusive)",
       score_X2({8192: X2_REF + X2_MARGIN, 65536: 92.0})["verdict"] == "CONFIRMS")
    ck("X2 one rung 0.01 over the ceiling REFUTES",
       score_X2({8192: X2_REF + X2_MARGIN + 0.01, 65536: 92.0})["verdict"] == "REFUTES")
    ck("X2 refutes on EITHER rung, not only the coarsest",
       score_X2({8192: 92.0, 65536: X2_REF + X2_MARGIN + 0.01})["verdict"] == "REFUTES")
    ck("X2 names which rungs went over",
       score_X2({8192: 94.0, 65536: 92.0})["over"] == [8192])
    ck("X2's refuting branch IS flagged interesting",
       score_X2({8192: 94.0, 65536: 92.0})["interesting"] is True)
    ck("X2's refuting reading calls it a METHOD result",
       "METHOD result" in score_X2({8192: 94.0, 65536: 92.0})["reading"])
    ck("X2 is NOT two-sided -- far BELOW the ceiling still confirms",
       score_X2({8192: 80.0, 65536: 80.0})["verdict"] == "CONFIRMS")
    ck("X2 with no data is NO DATA", score_X2({})["verdict"] == "NO DATA")
    ck("X2 scores the rungs it has, not the ones it lacks",
       score_X2({8192: 94.0})["verdict"] == "REFUTES")
    ck("X1's anchor alone would CONFIRM X2 (the anchor is below layerwise)",
       X1_ANCHOR <= X2_REF + X2_MARGIN)

    # --- the dirname parser, including what it must REFUSE
    ck("probe_k8192_cx2_s0 -> (8192, 0)",
       parse_cx2_dirname("probe_k8192_cx2_s0") == (8192, 0))
    ck("probe_k65536_cx2_s2 -> (65536, 2)",
       parse_cx2_dirname("probe_k65536_cx2_s2") == (65536, 2))
    ck("a ck1 chunk dir is refused, even though it is a chunk dir",
       parse_cx2_dirname("probe_k1024_ck1_s0") == (None, None))
    ck("an UNREGISTERED K in this family is refused",
       parse_cx2_dirname("probe_k4096_cx2_s0")[0] is None)
    ck("an mm1 dir is refused", parse_cx2_dirname("probe_ch_mm1_s0") == (None, None))
    ck("a non-probe directory is refused",
       parse_cx2_dirname("Tensorboard_outputs") == (None, None))
    ck("k8192 and k65536 are not confused",
       parse_cx2_dirname("probe_k8192_cx2_s0")[0]
       != parse_cx2_dirname("probe_k65536_cx2_s0")[0])

    # --- X0.3 shape from the HEADER (the c74 false-VOID regression)
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        def mk(K, n_tot, dtype, arr_n=None):
            d = os.path.join(td, "probe_k%d_cx2_s0" % K)
            os.makedirs(d, exist_ok=True)
            json.dump({"n_tot": n_tot}, open(os.path.join(d, "neg_counts.json"), "w"))
            np.save(os.path.join(d, "neg_counts.npy"),
                    np.zeros(arr_n if arr_n is not None else n_tot, dtype=dtype))
            return d
        ck("int32 array of the right length PASSES X0.3",
           gate_X03(mk(8192, M_OF_K[8192], np.int32), 8192)["ok"])
        ck("X0.3 reads the true shape from the header",
           gate_X03(mk(8192, M_OF_K[8192], np.int32), 8192)["shape"]
           == (M_OF_K[8192],))
        ck("int64 also passes on length",
           gate_X03(mk(65536, M_OF_K[65536], np.int64), 65536)["ok"])
        ck("a SHORT array FAILS X0.3",
           not gate_X03(mk(8192, M_OF_K[8192], np.int32,
                           arr_n=M_OF_K[8192] - 1), 8192)["ok"])
        ck("n_tot of the WRONG K fails X0.3",
           not gate_X03(mk(65536, M_OF_K[8192], np.int32), 65536)["ok"])
        dm = os.path.join(td, "probe_k65536_cx2_s1")
        os.makedirs(dm)
        ck("absent neg_counts.json FAILS X0.3 (bf8's failure)",
           not gate_X03(dm, 65536)["ok"])

    # --- the slope helper
    ck("slope needs 3 distinct m", math.isnan(slope_log_neff([10, 100], [1, 10])))
    ck("slope of a pure power law is its exponent",
       abs(slope_log_neff([10, 100, 1000], [10 ** 0.5, 10 ** 1.0, 10 ** 1.5]) - 0.5)
       < 1e-9)

    # --- the discipline this scorer registered against itself
    ck("scorer refuses to write the 53.1% sentence", '"53.1%"' in __doc__)
    ck("X3 carries no registered direction", "NO DIRECTION REGISTERED" in __doc__)
    ck("X2's confirming branch is documented as ordinary", "ORDINARY outcome" in __doc__)
    ck("the ms confound X1 inherits is documented",
       "confounded with distance-from-optimum" in __doc__)
    ck("the partly-mechanical caveat on a_raw is documented",
       "PARTLY\n    MECHANICAL" in __doc__)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_cx2"))
    ap.add_argument("--ck1-root", default=os.path.join(REPO, "..", "probes_ck1"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs, refused = [], []
    for d in sorted(glob.glob(os.path.join(a.root, "probe_*"))):
        K, seed = parse_cx2_dirname(os.path.basename(d))
        if K is None or seed is None:
            refused.append(os.path.basename(d))
            continue
        dirs.append((d, K, seed))
    rows = {r["run"]: r for r in csv.DictReader(open(a.csv))}

    print("=" * 78)
    print("c76 -- cx2: THE CHUNK LADDER'S COARSE END.  %d probe dirs" % len(dirs))
    print("box = (%.1f, %.4f)   ms=%s   %d ep   K = %s   m = %s"
          % (LO, HI, MST, EPOCHS, ", ".join(str(k) for k in KS),
             ", ".join(str(M_OF_K[k]) for k in KS)))
    print("=" * 78)
    if refused:
        print("REFUSED (not a registered cx2 dir): %s" % ", ".join(refused))

    # ---- X0
    print("\n--- X0  VALIDITY (n_records==%d, beta moved, ep==%d)"
          % (N_RECORDS, EPOCHS))
    x0 = {}
    for d, K, seed in dirs:
        r = gate_X0(d, K, rows.get("cx2-k%d-s%d" % (K, seed)))
        x0[d] = r
        print("    %-24s %-4s  n_rec=%-6d ep=%s/%s span=%.2f"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_records"],
                 r["epochs_done"], r["epochs_requested"], r["span"]))
    n0 = sum(1 for r in x0.values() if r["ok"])
    print("    X0: %d/%d" % (n0, len(dirs)))

    # ---- X0.2
    print("\n--- X0.2  n_beta == m(K) EXACTLY, on EVERY record")
    n02 = 0
    for d, K, seed in dirs:
        r = x0[d]
        ok = r["nb_ok"]
        n02 += ok
        print("    %-24s %-4s  n_beta=%-10s want m(%d)=%d"
              % (os.path.basename(d), "PASS" if ok else "FAIL",
                 ",".join(str(x) for x in r["n_beta"]), K, M_OF_K[K]))
    print("    X0.2: %d/%d" % (n02, len(dirs)))
    if n0 != len(dirs) or n02 != len(dirs):
        print("    **X0/X0.2 FAILED -- NOTHING BELOW IS SCORED.**")
        return 1

    # ---- X0.3
    print("\n--- X0.3  THE INSTRUMENT FIRED")
    n03 = 0
    for d, K, seed in dirs:
        r = gate_X03(d, K)
        n03 += r["ok"]
        print("    %-24s %-4s  n_tot=%-8s npy_shape=%-12s %s"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_tot"],
                 r.get("shape"), r["why"]))
    print("    X0.3: %d/%d" % (n03, len(dirs)))

    # ---- X0.4
    print("\n--- X0.4  BOX-FREE AT BOTH GUARDS (rec_-based 5%% gate PRIMARY, UNCHANGED)")
    print("    %-24s %-9s %8s %8s %10s %10s"
          % ("dir", "verdict", "rec_lo", "rec_hi", "coord_lo", "coord_hi"))
    x04 = {}
    for d, K, seed in dirs:
        r = gate_X04(d)
        x04[d] = r
        cl = "%.5f" % r["coord_lo"] if r["coord_lo"] is not None else "None"
        chh = "%.5f" % r["coord_hi"] if r["coord_hi"] is not None else "None"
        print("    %-24s %-9s %8.4f %8.4f %10s %10s"
              % (os.path.basename(d), "free" if r["boxfree"] else "BOUND:" + r["which"],
                 r["rec_lo"], r["rec_hi"], cl, chh))
    print("    X0.4: %d/%d box-free"
          % (sum(1 for r in x04.values() if r["boxfree"]), len(dirs)))

    # ---- plateau5 per K, from the CSV
    pl = {}
    for d, K, seed in dirs:
        row = rows.get("cx2-k%d-s%d" % (K, seed))
        if row and row.get("plateau5", "").strip():
            pl.setdefault(K, []).append(float(row["plateau5"]))
    by_k = {k: statistics.mean(v) for k, v in pl.items() if v}

    # ---- X1
    print("\n--- X1  **REGISTERED**: K3 extended.  plateau5 monotone NON-DECREASING")
    print("    as K rises, chained onto ck1's chunk1024 = %.3f (ties +-%.2f)"
          % (X1_ANCHOR, X1_TIE))
    print("    %-10s %-12s %10s %8s %8s" % ("K", "m", "plateau5", "sem", "n"))
    print("    %-10s %-12d %10.3f %8s %8s"
          % ("1024*", X1_ANCHOR_M, X1_ANCHOR, "--", "ck1"))
    for K in KS:
        v = pl.get(K, [])
        if not v:
            print("    %-10d %-12d %10s" % (K, M_OF_K[K], "NO DATA"))
            continue
        print("    %-10d %-12d %10.3f %8.3f %8d"
              % (K, M_OF_K[K], statistics.mean(v), _sem(v), len(v)))
    s1 = score_X1(by_k)
    for q in s1.get("pairs", []):
        print("      chunk%-6d -> chunk%-6d  %+.3f pp   %s"
              % (q["lo"], q["hi"], q["delta"],
                 "**INVERTS**" if q["inverts"] else "ok"))
    print("    X1 -> **%s**%s"
          % (s1["verdict"], ("   " + s1["why"]) if s1.get("why") else ""))
    if not s1.get("complete", False):
        print("    (ladder INCOMPLETE -- the verdict covers only the rungs present)")
    print("    * chunk1024 is ck1's MEASURED rung, carried across batches.  A")
    print("      cross-batch offset can enter the FIRST link only; ck1 measured that")
    print("      offset at -0.255 pp (t=1.74, NOT resolved) and it is NOT applied.")

    # ---- X2
    print("\n--- X2  **REGISTERED, ONE-SIDED**: does the chunk family overshoot layerwise?")
    s2 = score_X2(by_k)
    print("    layerwise (m=%d, the chunk family's OWN coarse endpoint) %.3f (n=%d)"
          % (X2_REF_M, X2_REF, X2_REF_N))
    print("    ceiling = %.3f + %.2f = %.3f" % (X2_REF, X2_MARGIN, X2_REF + X2_MARGIN))
    for K in KS:
        if by_k.get(K) is None:
            continue
        print("      chunk%-6d (m=%-6d) %10.3f   %s"
              % (K, M_OF_K[K], by_k[K],
                 "**OVER**" if by_k[K] > X2_REF + X2_MARGIN + EPS else "under"))
    print("    X2 -> **%s**" % s2["verdict"])
    print("    reading: %s" % s2["reading"])
    if not s2.get("interesting", False) and s2["verdict"] == "CONFIRMS":
        print("    Recorded as the ORDINARY outcome.  It is NOT written up as a finding.")

    # ---- X3
    print("\n--- X3  THE FIELD OVER THE WHOLE RANGE.  **DESCRIPTIVE, NO DIRECTION.**")
    from neff_instrument import agreement_stats
    import probe5_window as p5w
    ag_by_k, neff_by_k = {}, {}
    print("    %-24s %8s %10s %9s %9s %11s"
          % ("dir", "pbar", "bias", "a_raw", "a_deb", "bias_share"))
    for d, K, seed in dirs:
        s = agreement_stats(d, window=(0.5, 1.0))
        if s is None:
            print("    %-24s  -- agreement_stats returned None" % os.path.basename(d))
            continue
        ag_by_k.setdefault(K, []).append(s)
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
        neff_by_k.setdefault(K, []).append(1.0 / (1.0 + (mm - 1.0) * rho))

    print("\n    per K (seeds averaged):")
    print("      %-10s %-12s %9s %9s %9s %11s %12s"
          % ("K", "m", "pbar", "a_raw", "a_deb", "bias_share", "N_eff/m"))
    for K in KS:
        v = ag_by_k.get(K, [])
        if not v:
            continue
        ne = neff_by_k.get(K, [])
        print("      %-10d %-12d %9.5f %9.5f %9.5f %11.4f %12s"
              % (K, M_OF_K[K],
                 statistics.mean([x["pbar"] for x in v]),
                 statistics.mean([x["a_raw"] for x in v]),
                 statistics.mean([x["a_deb"] for x in v]),
                 statistics.mean([x["bias_share"] for x in v]),
                 ("%.4f +-%.4f" % (statistics.mean(ne), _sem(ne))) if ne else "--"))

    ms_ = [M_OF_K[K] for K in KS if neff_by_k.get(K)]
    ne_ = [statistics.mean(neff_by_k[K]) * M_OF_K[K] for K in KS if neff_by_k.get(K)]
    sl = slope_log_neff(ms_, ne_)
    print("\n    d log N_eff / d log m over THESE rungs alone = %s (needs >=3 rungs)"
          % (("%.3f" % sl) if math.isfinite(sl) else "n/a -- only 2 rungs here"))
    print("    The whole-family slope is computed in the FINDINGS write-up by joining")
    print("    these two rungs to ck1's five; it is not computed from 2 points here.")
    print("    The fall of a_raw with m is PARTLY MECHANICAL (a coarse coordinate's")
    print("    meta-gradient is a sum over many fine ones).  No independence null is")
    print("    derived here, and the \"53.1%\" sentence is NOT reproduced.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
