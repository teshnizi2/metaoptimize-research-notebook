#!/usr/bin/env python3
"""c75_ck1_score.py -- score `ck1`, THE CHUNK LADDER.

REGISTERED GATES, transcribed from `bin/c75_chunk_ladder.sh:34-81`.  Every constant
below is asserted by the selftest against THAT SCRIPT'S OWN TEXT so the registration
and the batch cannot drift -- STANDING RULE (19).

WRITTEN AND SELFTESTED BEFORE ANY ck1 VERDICT WAS READ.

  K0    VALIDITY.  n_records == 10000; beta moved; epochs_done == requested == 100.
  K0.2  n_beta == m(K) EXACTLY on EVERY record, per K.  These are MEASURED values
        (from the allocated beta on the built network), not formulas evaluated here.
        A mismatch means the partition is not the one the ladder claims.
  K0.3  THE INSTRUMENT FIRED.  neg_counts.json present, n_tot == n_beta, npy shape
        READ FROM ITS HEADER == (n_tot,).  (The header is read, never inferred from
        the file size -- that inference produced c74's false VOID on 12/12 arms.)
  K0.4  BOX-FREE at BOTH guards, the PUBLISHED rec_-based 5% gate, PRIMARY and
        UNCHANGED.  coord_* reported ALONGSIDE and NEVER as the gate.

  K1    **THE ANCHOR.  CAN ONLY VOID, AND IT VOIDS THE WHOLE BATCH.**
        chunk1 is weightwise BITWISE (tests/test_chunkwise.py C1), so its plateau5
        must reproduce tw0's weightwise 91.234 (n=3) within +-0.50 pp.  FAILS ->
        PATCH_CHUNKWISE changes training in a real 100-epoch run despite the bitwise
        unit test, and NOTHING in this batch may be read.

  K2    **IS "GRANULARITY" m, OR IS IT THE PARTITION?  REGISTERED PREDICTION.**
        chunk1024 (m=10,944) and nodewise (m=14,420) are within 24% in m but are
        structurally different partitions -- contiguous flat chunks vs output
        channels.  At the SAME ms=1e-4, tw0 measured nodewise plateau5 91.961.
        **PREDICTS: chunk1024 within +-0.50 pp of 91.961.**
        CONFIRMS -> accuracy at this scale is a function of m and the campaign may
             keep calling the axis "granularity" and meaning the group count.
        REFUTES  -> **PARTITION STRUCTURE MATTERS AT FIXED m**; the ladder is a
             curve in two variables and every "granularity" statement needs the
             partition named alongside the count.

  K3    THE SHAPE.  plateau5 monotone NON-DECREASING as K rises (m falls) across
        1 -> 2 -> 16 -> 128 -> 1024, ties allowed within +-0.15 pp.  REFUTES if any
        adjacent pair inverts by MORE than 0.15 pp -> structure inside the hole.

  K4    THE BLOCK-SIZE CURVE OF THE FIELD.  **DESCRIPTIVE, NO DIRECTION REGISTERED.**
        Report pbar, bias, a_raw, a_deb, bias_share and N_eff/m per K, plus the
        d log N_eff / d log m slope against tw0's 0.686 over the OUTER range.
        **THE "53.1%" SENTENCE IS NOT REPRODUCED HERE AND MUST NOT BE WRITTEN.**
  K4b   chunk1024's a_raw vs nodewise@1e-4's 0.53836 -- K2's question for the field.
        The 0.01 "close" tolerance is INHERITED from c75_at1_score.score_A2b's
        default; it was NOT chosen against this data.

WHAT THIS SCORER WILL NOT DO
  * It will not re-derive an independence null on the fly (CORRECTIONS 26).
  * It will not read K2-K4 if K1 VOIDs.
  * The monotone fall of a_raw with m is PARTLY MECHANICAL (a coarse coordinate's
    meta-gradient is a sum over many fine ones), so K4's curve is reported as a
    curve and never as evidence of structure on its own.

USAGE
  python3 analysis/c75_ck1_score.py --selftest
  python3 analysis/c75_ck1_score.py --root ../probes_ck1 --csv results/all_runs.csv
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
SCRIPT = os.path.join(REPO, "bin", "c75_chunk_ladder.sh")

from c52_boxfree import occupancy, records           # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -15.0, -2.3026             # `ck1`, registered in c55_neff_noise.BOXES
EPOCHS = 100
N_RECORDS = 10000
MST = "1e-4"
KS = (1, 2, 16, 128, 1024)
# m(K), MEASURED from the allocated beta on the built network.  Not a formula.
M_OF_K = {1: 11173962, 2: 5586981, 16: 698373, 128: 87303, 1024: 10944}
SEEDS = (0, 1, 2)
BOXFREE_MAX = 0.05                  # the PUBLISHED rec_-based 5% gate, UNCHANGED

K1_REF, K1_REF_N, K1_BAR = 91.234, 3, 0.50      # tw0 weightwise, 100 ep, ms=1e-4
K2_REF, K2_REF_N, K2_BAR = 91.961, 3, 0.50      # tw0 nodewise,   100 ep, ms=1e-4
K2_REF_M = 14420                                 # nodewise's group count
K3_TIE = 0.15
K4B_REF = 0.53836                                # tw0 nodewise@1e-4 a_raw
K4B_TOL = 0.01                                   # INHERITED from score_A2b
TW0_OUTER_SLOPE = 0.686                          # d log N_eff / d log m, outer range


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def parse_ck1_dirname(base):
    """'probe_k1024_ck1_s0' -> (1024, 0).  Returns (None, None) on anything else.

    probe5_window.parse_dirname cannot see chunk rungs (they are not in its
    RUNG_TOKENS) and IT IS NOT PATCHED -- five registered scorers depend on it.  The
    chunk naming is parsed here instead, and K is required to be a REGISTERED K so a
    stray directory can never enter the ladder silently.
    """
    parts = base.split("_")
    if not parts or parts[0] != "probe":
        return None, None
    parts = parts[1:]
    seed = None
    if parts and parts[-1].startswith("s") and parts[-1][1:].isdigit():
        seed = int(parts[-1][1:])
        parts = parts[:-1]
    if "ck1" not in parts:
        return None, None
    ktok = next((p for p in parts
                 if p.startswith("k") and p[1:].isdigit()), None)
    if ktok is None:
        return None, None
    K = int(ktok[1:])
    return (K if K in M_OF_K else None), seed


# --- K0 ---------------------------------------------------------------------
def gate_K0(d, K, row=None):
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


# --- K0.3 -------------------------------------------------------------------
def gate_K03(d, K):
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


# --- K0.4 -------------------------------------------------------------------
def gate_K04(d, lo=LO, hi=HI):
    o = occupancy(d, lo, hi)
    boxfree = (o["rec_lo"] < BOXFREE_MAX) and (o["rec_hi"] < BOXFREE_MAX)
    return dict(boxfree=boxfree, rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                coord_lo=o["coord_lo"], coord_hi=o["coord_hi"],
                which=("hi" if o["rec_hi"] >= BOXFREE_MAX else
                       ("lo" if o["rec_lo"] >= BOXFREE_MAX else "")))


# --- K1 ---------------------------------------------------------------------
def score_K1(plateaus):
    """THE ANCHOR.  CAN ONLY VOID.  A VOID voids the whole batch."""
    if not plateaus:
        return dict(ok=False, verdict="NO DATA", mean=None, ref=K1_REF, diff=None,
                    n=0, sem=None, bar=K1_BAR)
    m = statistics.mean(plateaus)
    diff = m - K1_REF
    return dict(mean=m, sem=_sem(plateaus), n=len(plateaus), ref=K1_REF, bar=K1_BAR,
                diff=diff, ok=(abs(diff) <= K1_BAR),
                verdict=("REPRODUCES" if abs(diff) <= K1_BAR else "VOID"))


# --- K2 ---------------------------------------------------------------------
def score_K2(plateaus):
    """REGISTERED, TWO-SIDED.  CONFIRMS -> accuracy is a function of m.
    REFUTES -> partition structure matters at fixed m."""
    if not plateaus:
        return dict(verdict="NO DATA", mean=None, ref=K2_REF, diff=None, n=0,
                    sem=None, bar=K2_BAR, reading="")
    m = statistics.mean(plateaus)
    diff = m - K2_REF
    conf = abs(diff) <= K2_BAR
    return dict(mean=m, sem=_sem(plateaus), n=len(plateaus), ref=K2_REF, bar=K2_BAR,
                diff=diff, ok=conf,
                verdict=("CONFIRMS" if conf else "REFUTES"),
                reading=("accuracy at this scale is a function of m; the axis may be "
                         "called granularity and mean the group count"
                         if conf else
                         "PARTITION STRUCTURE MATTERS AT FIXED m -- the ladder is a "
                         "curve in two variables and every granularity statement "
                         "needs the partition named alongside the count"))


# --- K3 ---------------------------------------------------------------------
def score_K3(by_k):
    """{K: mean plateau5} -> monotone non-decreasing in K, ties within +-K3_TIE."""
    ks = [k for k in KS if by_k.get(k) is not None]
    if len(ks) < 2:
        return dict(verdict="NO DATA", pairs=[], ks=ks)
    pairs, bad = [], []
    for a, b in zip(ks, ks[1:]):
        d = by_k[b] - by_k[a]
        inv = d < -K3_TIE
        pairs.append(dict(lo=a, hi=b, delta=d, inverts=inv))
        if inv:
            bad.append("%d->%d %+.3f" % (a, b, d))
    return dict(verdict=("CONFIRMS" if not bad else "REFUTES"), pairs=pairs,
                ks=ks, tie=K3_TIE, why="; ".join(bad),
                complete=(len(ks) == len(KS)))


# --- K4b --------------------------------------------------------------------
def score_K4b(a_raw_1024):
    """NO confirm/refute verdict.  A reading, with an INHERITED tolerance."""
    if a_raw_1024 is None:
        return dict(reading="NO DATA", diff=None, ref=K4B_REF)
    diff = a_raw_1024 - K4B_REF
    return dict(a_raw=a_raw_1024, ref=K4B_REF, diff=diff, tol=K4B_TOL,
                reading=("the FIELD too looks like a function of m, not of the "
                         "partition" if abs(diff) <= K4B_TOL else
                         "the FIELD moves with the PARTITION at nearly fixed m"))


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

    print("c75_ck1_score selftest")
    # --- every constant against the batch script's own text (STANDING RULE 19)
    ck("batch script exists", bool(src))
    ck("script CLIP= is the registered box", "CLIP=-15:-2.3026" in src)
    ck("LO/HI match the script's CLIP", (LO, HI) == (-15.0, -2.3026))
    ck("script EPOCHS= matches", ("EPOCHS=%d" % EPOCHS) in src)
    ck("script MST= matches", ("MST=%s" % MST) in src)
    ck("script NJOBS is 15 = 5 K x 3 seeds", "NJOBS=15" in src)
    ck("script KS= is this scorer's ladder",
       ('KS="%s"' % " ".join(str(k) for k in KS)) in src)
    ck("N_RECORDS = epochs*500/stride 5", N_RECORDS == EPOCHS * 500 // 5)
    ck("script exports PROBE=5", "PROBE=5" in src)
    ck("script exports PROBE5=1 (instrument, not stride)", "PROBE5=1" in src)
    ck("script's WANT table is this scorer's M_OF_K",
       ("WANT = {1: 11173962, 2: 5586981, 16: 698373, 128: 87303, 1024: 10944}")
       in src)
    ck("M_OF_K matches that table literally",
       M_OF_K == {1: 11173962, 2: 5586981, 16: 698373, 128: 87303, 1024: 10944})
    ck("K1 anchor 91.234 is in the script", "91.234" in src)
    ck("K1 anchor n=3 is in the script", "91.234 (n=3)" in src)
    ck("K2 reference 91.961 is in the script", "91.961" in src)
    ck("K2 reference m 14,420 is in the script", "14,420" in src)
    ck("K3 tie band 0.15 is in the script", "+-0.15 pp" in src)
    ck("K4b reference 0.53836 is in the script", "0.53836" in src)
    ck("tw0 outer slope 0.686 is in the script", "0.686" in src)
    ck("K1/K2 bars are STANDING RULE 9's level bar",
       K1_BAR == 0.50 and K2_BAR == 0.50)
    ck("5% gate UNCHANGED", BOXFREE_MAX == 0.05)
    ck("seeds 0-2", SEEDS == (0, 1, 2))
    ck("chunk1 m == weightwise's 11,173,962", M_OF_K[1] == 11173962)
    ck("m(K) is strictly decreasing in K",
       all(M_OF_K[a] > M_OF_K[b] for a, b in zip(KS, KS[1:])))
    ck("chunk1024's m is BELOW nodewise's, not equal to it",
       M_OF_K[1024] < K2_REF_M)

    import c55_neff_noise as c55
    ck("ck1 registered in c55_neff_noise.BOXES", "ck1" in c55.BOXES)
    ck("c55's ck1 box == this scorer's box",
       c55.BOXES.get("ck1", (None,) * 3)[:2] == (LO, HI))

    # --- K1 can only VOID, two-sided
    ck("K1 exact reference reproduces", score_K1([K1_REF])["verdict"] == "REPRODUCES")
    ck("K1 +0.49 reproduces", score_K1([K1_REF + 0.49])["verdict"] == "REPRODUCES")
    ck("K1 -0.49 reproduces", score_K1([K1_REF - 0.49])["verdict"] == "REPRODUCES")
    ck("K1 +0.51 VOIDs", score_K1([K1_REF + 0.51])["verdict"] == "VOID")
    ck("K1 -0.51 VOIDs (two-sided)", score_K1([K1_REF - 0.51])["verdict"] == "VOID")
    ck("K1 empty is not a pass", score_K1([])["ok"] is False)
    ck("K1 averages its seeds", abs(score_K1([91.0, 91.468])["mean"] - 91.234) < 1e-9)

    # --- K2 both directions, and it is NOT K1's reference
    ck("K2 exact reference confirms", score_K2([K2_REF])["verdict"] == "CONFIRMS")
    ck("K2 +0.49 confirms", score_K2([K2_REF + 0.49])["verdict"] == "CONFIRMS")
    ck("K2 -0.51 refutes (two-sided)", score_K2([K2_REF - 0.51])["verdict"] == "REFUTES")
    ck("K2 +0.51 refutes", score_K2([K2_REF + 0.51])["verdict"] == "REFUTES")
    ck("K2 refuting reading names the partition",
       "PARTITION STRUCTURE MATTERS" in score_K2([K2_REF + 0.9])["reading"])
    ck("K2 uses its OWN reference, not K1's", K2_REF != K1_REF)
    ck("K1's anchor value would REFUTE K2",
       score_K2([K1_REF])["verdict"] == "REFUTES")
    ck("K2 empty is NO DATA", score_K2([])["verdict"] == "NO DATA")

    # --- K3 monotone-in-K with a tie band
    up = {1: 91.0, 2: 91.2, 16: 91.5, 128: 91.8, 1024: 92.0}
    ck("K3 strictly rising confirms", score_K3(up)["verdict"] == "CONFIRMS")
    flat = {k: 91.5 for k in KS}
    ck("K3 flat confirms (ties allowed)", score_K3(flat)["verdict"] == "CONFIRMS")
    tie = dict(up)
    tie[128] = up[16] - 0.14
    ck("K3 a 0.14 dip is inside the tie band", score_K3(tie)["verdict"] == "CONFIRMS")
    inv = dict(up)
    inv[128] = up[16] - 0.16
    ck("K3 a 0.16 dip REFUTES", score_K3(inv)["verdict"] == "REFUTES")
    ck("K3 names the inverting pair", "16->128" in score_K3(inv)["why"])
    ck("K3 checks in K order, not dict order",
       [(q["lo"], q["hi"]) for q in score_K3(up)["pairs"]]
       == [(1, 2), (2, 16), (16, 128), (128, 1024)])
    ck("K3 flags an incomplete ladder",
       score_K3({1: 91.0, 2: 91.2})["complete"] is False)
    ck("K3 on a full ladder is complete", score_K3(up)["complete"] is True)
    ck("K3 with one rung is NO DATA", score_K3({1: 91.0})["verdict"] == "NO DATA")

    # --- K4b carries NO verdict
    r = score_K4b(K4B_REF)
    ck("K4b returns a reading, not a verdict", "verdict" not in r)
    ck("K4b close => function of m", "function of m" in score_K4b(K4B_REF)["reading"])
    ck("K4b far => moves with the PARTITION",
       "PARTITION" in score_K4b(K4B_REF + 0.05)["reading"])
    ck("K4b handles missing data", score_K4b(None)["reading"] == "NO DATA")

    # --- the dirname parser, including what it must REFUSE
    ck("probe_k1_ck1_s0 -> (1, 0)", parse_ck1_dirname("probe_k1_ck1_s0") == (1, 0))
    ck("probe_k1024_ck1_s2 -> (1024, 2)",
       parse_ck1_dirname("probe_k1024_ck1_s2") == (1024, 2))
    ck("probe_k128_ck1_s1 -> (128, 1)",
       parse_ck1_dirname("probe_k128_ck1_s1") == (128, 1))
    ck("an UNREGISTERED K is refused, not admitted",
       parse_ck1_dirname("probe_k4096_ck1_s0")[0] is None)
    ck("another family's dir is refused",
       parse_ck1_dirname("probe_w_tw0_s0") == (None, None))
    ck("a nodewise dir cannot enter the chunk ladder",
       parse_ck1_dirname("probe_node_at1_s0") == (None, None))
    ck("k1 and k1024 are not confused",
       parse_ck1_dirname("probe_k1_ck1_s0")[0]
       != parse_ck1_dirname("probe_k1024_ck1_s0")[0])

    # --- K0.3 shape from the HEADER (the c74 false-VOID regression)
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        def mk(K, n_tot, dtype, arr_n=None):
            d = os.path.join(td, "probe_k%d_ck1_s0" % K)
            os.makedirs(d, exist_ok=True)
            json.dump({"n_tot": n_tot}, open(os.path.join(d, "neg_counts.json"), "w"))
            np.save(os.path.join(d, "neg_counts.npy"),
                    np.zeros(arr_n if arr_n is not None else n_tot, dtype=dtype))
            return d
        ck("int32 array of the right length PASSES K0.3",
           gate_K03(mk(1024, M_OF_K[1024], np.int32), 1024)["ok"])
        ck("K0.3 reads the true shape from the header",
           gate_K03(mk(1024, M_OF_K[1024], np.int32), 1024)["shape"]
           == (M_OF_K[1024],))
        ck("int64 also passes on length",
           gate_K03(mk(1024, M_OF_K[1024], np.int64), 1024)["ok"])
        ck("a SHORT array FAILS K0.3",
           not gate_K03(mk(1024, M_OF_K[1024], np.int32,
                           arr_n=M_OF_K[1024] - 1), 1024)["ok"])
        ck("n_tot of the WRONG K fails K0.3",
           not gate_K03(mk(128, M_OF_K[1024], np.int32), 128)["ok"])
        dm = os.path.join(td, "probe_k16_ck1_s1")
        os.makedirs(dm)
        ck("absent neg_counts.json FAILS K0.3 (bf8's failure)",
           not gate_K03(dm, 16)["ok"])

    # --- the slope helper
    ck("slope needs 3 distinct m", math.isnan(slope_log_neff([10, 100], [1, 10])))
    ck("slope of a pure power law is its exponent",
       abs(slope_log_neff([10, 100, 1000], [10 ** 0.5, 10 ** 1.0, 10 ** 1.5]) - 0.5)
       < 1e-9)

    # --- the discipline this scorer registered against itself
    ck("scorer refuses to write the 53.1% sentence", '"53.1%"' in __doc__)
    ck("K4 carries no registered direction", "NO DIRECTION REGISTERED" in __doc__)
    ck("K4b's tolerance is declared INHERITED", "INHERITED" in __doc__)
    ck("K1 is documented as batch-voiding", "VOIDS THE WHOLE BATCH" in __doc__)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_ck1"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    allduse = sorted(glob.glob(os.path.join(a.root, "probe_*")))
    dirs, refused = [], []
    for d in allduse:
        K, seed = parse_ck1_dirname(os.path.basename(d))
        if K is None or seed is None:
            refused.append(os.path.basename(d))
            continue
        dirs.append((d, K, seed))
    rows = {r["run"]: r for r in csv.DictReader(open(a.csv))}

    print("=" * 78)
    print("c75 -- ck1: THE CHUNK LADDER.  %d probe dirs" % len(dirs))
    print("box = (%.1f, %.4f)   ms=%s   %d ep   K = %s"
          % (LO, HI, MST, EPOCHS, ", ".join(str(k) for k in KS)))
    print("=" * 78)
    if refused:
        print("REFUSED (not a registered ck1 chunk dir): %s" % ", ".join(refused))

    # ---- K0
    print("\n--- K0  VALIDITY (n_records==%d, beta moved, ep==%d)"
          % (N_RECORDS, EPOCHS))
    k0 = {}
    for d, K, seed in dirs:
        r = gate_K0(d, K, rows.get("ck1-k%d-s%d" % (K, seed)))
        k0[d] = r
        print("    %-22s %-4s  n_rec=%-6d ep=%s/%s span=%.2f"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_records"],
                 r["epochs_done"], r["epochs_requested"], r["span"]))
    n0 = sum(1 for r in k0.values() if r["ok"])
    print("    K0: %d/%d" % (n0, len(dirs)))

    # ---- K0.2
    print("\n--- K0.2  n_beta == m(K) EXACTLY, on EVERY record")
    n02 = 0
    for d, K, seed in dirs:
        r = k0[d]
        ok = r["nb_ok"]
        n02 += ok
        print("    %-22s %-4s  n_beta=%-12s want m(%d)=%d"
              % (os.path.basename(d), "PASS" if ok else "FAIL",
                 ",".join(str(x) for x in r["n_beta"]), K, M_OF_K[K]))
    print("    K0.2: %d/%d" % (n02, len(dirs)))
    if n0 != len(dirs) or n02 != len(dirs):
        print("    **K0/K0.2 FAILED -- NOTHING BELOW IS SCORED.**")
        return 1

    # ---- K0.3
    print("\n--- K0.3  THE INSTRUMENT FIRED")
    n03 = 0
    for d, K, seed in dirs:
        r = gate_K03(d, K)
        n03 += r["ok"]
        print("    %-22s %-4s  n_tot=%-10s npy_shape=%-14s %s"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_tot"],
                 r.get("shape"), r["why"]))
    print("    K0.3: %d/%d" % (n03, len(dirs)))

    # ---- K0.4
    print("\n--- K0.4  BOX-FREE AT BOTH GUARDS (rec_-based 5%% gate PRIMARY, UNCHANGED)")
    print("    %-22s %-9s %8s %8s %10s %10s"
          % ("dir", "verdict", "rec_lo", "rec_hi", "coord_lo", "coord_hi"))
    k04 = {}
    for d, K, seed in dirs:
        r = gate_K04(d)
        k04[d] = r
        cl = "%.5f" % r["coord_lo"] if r["coord_lo"] is not None else "None"
        ch = "%.5f" % r["coord_hi"] if r["coord_hi"] is not None else "None"
        print("    %-22s %-9s %8.4f %8.4f %10s %10s"
              % (os.path.basename(d), "free" if r["boxfree"] else "BOUND:" + r["which"],
                 r["rec_lo"], r["rec_hi"], cl, ch))
    print("    K0.4: %d/%d box-free"
          % (sum(1 for r in k04.values() if r["boxfree"]), len(dirs)))

    # ---- plateau5 per K, from the CSV
    pl = {}
    for d, K, seed in dirs:
        row = rows.get("ck1-k%d-s%d" % (K, seed))
        if row and row.get("plateau5", "").strip():
            pl.setdefault(K, []).append(float(row["plateau5"]))

    # ---- K1  THE ANCHOR
    print("\n--- K1  **THE ANCHOR.  CAN ONLY VOID, AND IT VOIDS THE WHOLE BATCH.**")
    s1 = score_K1(pl.get(1, []))
    if s1["mean"] is None:
        print("    chunk1 NO DATA -- the anchor cannot be scored, so nothing is read.")
        return 1
    print("    chunk1 (m=%d, weightwise BITWISE)  %.3f +-%.3f (n=%d)"
          % (M_OF_K[1], s1["mean"], s1["sem"], s1["n"]))
    print("    tw0 weightwise reference %.3f (n=%d)   diff %+.3f pp   bar +-%.2f  -> **%s**"
          % (K1_REF, K1_REF_N, s1["diff"], K1_BAR, s1["verdict"]))
    if not s1["ok"]:
        print("    **K1 VOIDED.  PATCH_CHUNKWISE changes training in a real 100-epoch")
        print("    run despite the bitwise unit test.  K2-K4 ARE NOT READ.**")
        return 1
    print("    The patch is inert at the endpoint it is bitwise-equal to.  K2-K4 may be read.")

    # ---- K2
    print("\n--- K2  **REGISTERED**: is 'granularity' m, or is it the PARTITION?")
    s2 = score_K2(pl.get(1024, []))
    if s2["mean"] is None:
        print("    chunk1024 NO DATA")
    else:
        print("    chunk1024 (m=%d, contiguous flat chunks)  %.3f +-%.3f (n=%d)"
              % (M_OF_K[1024], s2["mean"], s2["sem"], s2["n"]))
        print("    tw0 nodewise (m=%d, output channels) @ the SAME ms  %.3f (n=%d)"
              % (K2_REF_M, K2_REF, K2_REF_N))
        print("    diff %+.3f pp   bar +-%.2f  -> **%s**"
              % (s2["diff"], K2_BAR, s2["verdict"]))
        print("    reading: %s" % s2["reading"])

    # ---- K3
    print("\n--- K3  **REGISTERED**: plateau5 monotone NON-DECREASING as K rises "
          "(ties +-%.2f)" % K3_TIE)
    by_k = {k: statistics.mean(v) for k, v in pl.items() if v}
    print("    %-8s %-12s %10s %8s %8s" % ("K", "m", "plateau5", "sem", "n"))
    for K in KS:
        v = pl.get(K, [])
        if not v:
            print("    %-8d %-12d %10s" % (K, M_OF_K[K], "NO DATA"))
            continue
        print("    %-8d %-12d %10.3f %8.3f %8d"
              % (K, M_OF_K[K], statistics.mean(v), _sem(v), len(v)))
    s3 = score_K3(by_k)
    for q in s3.get("pairs", []):
        print("      chunk%-5d -> chunk%-5d  %+.3f pp   %s"
              % (q["lo"], q["hi"], q["delta"],
                 "**INVERTS**" if q["inverts"] else "ok"))
    print("    K3 -> **%s**%s"
          % (s3["verdict"], ("   " + s3["why"]) if s3.get("why") else ""))
    if not s3.get("complete", False):
        print("    (ladder INCOMPLETE -- the verdict covers only the rungs present)")

    # ---- K4
    print("\n--- K4  THE BLOCK-SIZE CURVE OF THE FIELD.  **DESCRIPTIVE, NO DIRECTION.**")
    from neff_instrument import agreement_stats
    import probe5_window as p5w
    ag_by_k, neff_by_k, neff_free_by_k = {}, {}, {}
    print("    %-22s %8s %10s %9s %9s %11s"
          % ("dir", "pbar", "bias", "a_raw", "a_deb", "bias_share"))
    for d, K, seed in dirs:
        ag = agreement_stats(d, window=(0.5, 1.0))
        if ag is None:
            print("    %-22s  -- agreement_stats returned None" % os.path.basename(d))
            continue
        ag_by_k.setdefault(K, []).append(ag)
        print("    %-22s %8.5f %+10.5f %9.5f %9.5f %11.4f"
              % (os.path.basename(d), ag["pbar"], ag["bias"], ag["a_raw"],
                 ag["a_deb"], ag["bias_share"]))
        w = p5w.reduce_dir(d, windows=(("steady .5-1", (0.5, 1.0)),))
        if w is None:
            continue
        ww = w["win"].get("steady .5-1")
        if not ww or ww["rho_s"] is None:
            continue
        mm, rho = float(w["n_tot"]), ww["rho_s"]
        neff_m = 1.0 / (1.0 + (mm - 1.0) * rho)
        neff_by_k.setdefault(K, []).append(neff_m)
        if k04[d]["boxfree"]:
            neff_free_by_k.setdefault(K, []).append(neff_m)

    print("\n    per K (seeds averaged):")
    print("      %-8s %-12s %9s %9s %9s %11s %12s"
          % ("K", "m", "pbar", "a_raw", "a_deb", "bias_share", "N_eff/m"))
    for K in KS:
        v = ag_by_k.get(K, [])
        if not v:
            continue
        ne = neff_by_k.get(K, [])
        print("      %-8d %-12d %9.5f %9.5f %9.5f %11.4f %12s"
              % (K, M_OF_K[K],
                 statistics.mean([x["pbar"] for x in v]),
                 statistics.mean([x["a_raw"] for x in v]),
                 statistics.mean([x["a_deb"] for x in v]),
                 statistics.mean([x["bias_share"] for x in v]),
                 ("%.4f +-%.4f" % (statistics.mean(ne), _sem(ne))) if ne else "--"))

    ms_ = [M_OF_K[K] for K in KS if neff_by_k.get(K)]
    ne_ = [statistics.mean(neff_by_k[K]) * M_OF_K[K] for K in KS if neff_by_k.get(K)]
    sl = slope_log_neff(ms_, ne_)
    print("\n    d log N_eff / d log m over the CHUNK range = %s   (tw0's OUTER-range "
          "slope %.3f)" % (("%.3f" % sl) if math.isfinite(sl) else "n/a",
                           TW0_OUTER_SLOPE))
    print("    Independence would give 1.000; full sharing 0.000.  DESCRIPTIVE.")
    print("    The fall of a_raw with m is PARTLY MECHANICAL (a coarse coordinate's")
    print("    meta-gradient is a sum over many fine ones).  No independence null is")
    print("    derived here, and the \"53.1%\" sentence is NOT reproduced.")

    # ---- K4b
    print("\n--- K4b  chunk1024's a_raw vs nodewise@1e-4's %.5f  (K2's question for "
          "the field)" % K4B_REF)
    v = ag_by_k.get(1024, [])
    s4b = score_K4b(statistics.mean([x["a_raw"] for x in v]) if v else None)
    if s4b["diff"] is None:
        print("    NO DATA")
    else:
        print("    chunk1024 a_raw %.5f  vs  %.5f   diff %+.5f   (tol %.2f, INHERITED)"
              % (s4b["a_raw"], s4b["ref"], s4b["diff"], s4b["tol"]))
        print("    reading: %s" % s4b["reading"])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
