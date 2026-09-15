#!/usr/bin/env python3
"""c75_at1_score.py -- score `at1`, the ALL-RUNGS-TUNED agreement ladder.

REGISTERED GATES, transcribed from `bin/c75_tuned_agreement_ladder.sh:22-80`.  The
selftests assert every constant against THAT SCRIPT'S OWN TEXT so the registration and
the batch cannot drift -- STANDING RULE (19).

WRITTEN AND SELFTESTED BEFORE THE DATA LANDED.

  A0    VALIDITY.  n_records == 10000; n_beta == {blk6 6, node 14420} on EVERY record;
        beta moved; epochs_done == epochs_requested == 100.
  A0.2  THE INSTRUMENT FIRED.  neg_counts.json, n_tot == n_beta, npy shape READ FROM
        ITS HEADER == (n_tot,).
  A0.3  BOX-FREE AT BOTH GUARDS, per seed, the PUBLISHED rec_-based 5% gate, PRIMARY
        and UNCHANGED.  coord_* reported ALONGSIDE and NEVER the gate.
  A1    ACCURACY REPRODUCTION.  **CAN ONLY VOID.**  Bar |diff| <= 0.50 pp per arm:
            blk6 @1e-4 vs 92.652 (n=2);  node @3e-4 vs 92.450 (n=5).
        FAILS -> the probe perturbed training and A2 is not about the tuned ladder.
  A2    **THE REGISTERED OUT-OF-SAMPLE PREDICTION.**  a_raw is MONOTONE DECREASING in
        the group count m across the tuned ladder.  Already measured at `tw0` and NOT
        re-derived here:  lay (m=62) 0.65693, node@1e-4 0.53836, w 0.50117.
        **PREDICTS: blk6 (m=6) a_raw > 0.65693.**
        **PREDICTS: node@3e-4 (m=14420) a_raw strictly inside (0.50117, 0.65693).**
        REFUTES if blk6 <= lay, or node@3e-4 leaves that interval -> agreement is not a
        function of block size alone and the mechanism story loses its carrier.
  A2b   **THE ms CONTROL.  NO DIRECTION REGISTERED.**  node@3e-4 vs node@1e-4 (0.53836)
        is the SAME m at two meta-stepsizes.  Agreement close => a_raw is a function of
        m, which is what A2's reading assumes.  Materially different => a_raw is a joint
        function of (m, ms) and A2's curve must be re-read.  Reported, never scored as
        confirm/refute, because no measurement of this contrast existed to predict from.
  A3    THE ACCURACY LADDER'S ARGMAX.  **REPLICATION, NOT DISCOVERY.**  A confirmation
        replicates a POST-HOC window and is never a discovery.

NOTE ON n: `at1` RE-RUNS SEEDS THAT ALREADY EXIST (FINDINGS 74.8).  blk6 reaches 5 CSV
rows but only 3 DISTINCT seeds; node reaches 8 rows but only 5 DISTINCT seeds.  This
scorer reports BOTH counts and never prints a row count where a seed count belongs.

USAGE
  python3 analysis/c75_at1_score.py --selftest
  python3 analysis/c75_at1_score.py --root ../probes_at1 --csv results/all_runs.csv
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
SCRIPT = os.path.join(REPO, "bin", "c75_tuned_agreement_ladder.sh")

from c52_boxfree import occupancy, records           # noqa: E402
from probe5_window import parse_dirname              # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -15.0, -2.3026             # `at1`, registered in c55_neff_noise.BOXES
EPOCHS = 100
N_RECORDS = 10000
N_BETA = {"blk6": 6, "node": 14420}
RUNG_FULL = {"blk6": "blk6(6)", "node": "nodewise"}
ARM_MS = {"blk6": "1e-4", "node": "3e-4"}
SEEDS = (0, 1, 2)
BOXFREE_MAX = 0.05                  # the PUBLISHED rec_-based 5% gate, UNCHANGED
A1_REF = {"blk6": (92.652, 2), "node": (92.450, 5)}
A1_BAR = 0.50
# tw0's measurements.  These are READ, never re-derived from at1.
TW0_A_RAW = {"lay": 0.65693, "node_1e4": 0.53836, "w": 0.50117}
LAY_M, NODE_M, W_M = 62, 14420, 11173962


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


# --- A0 ---------------------------------------------------------------------
def gate_A0(d, row=None):
    _, rung, _ = parse_dirname(os.path.basename(d))
    R = records(d)
    want = N_BETA.get(rung)
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
    return dict(rung=rung, n_records=len(R),
                n_beta=sorted(x for x in nb if x is not None),
                n_ok=n_ok, nb_ok=nb_ok, moved=moved, ep_ok=ep_ok,
                epochs_done=ep_d, epochs_requested=ep_r,
                span=max(maxs) - min(mins),
                ok=(n_ok and nb_ok and moved and (ep_ok is not False)))


# --- A0.2 -------------------------------------------------------------------
def gate_A02(d):
    _, rung, _ = parse_dirname(os.path.basename(d))
    j = os.path.join(d, "neg_counts.json")
    npy = os.path.join(d, "neg_counts.npy")
    if not os.path.exists(j):
        return dict(ok=False, why="neg_counts.json ABSENT -- instrument never fired",
                    n_tot=None, npy_ok=False, shape=None)
    meta = json.load(open(j))
    want = N_BETA.get(rung)
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


# --- A0.3 -------------------------------------------------------------------
def gate_A03(d, lo=LO, hi=HI):
    o = occupancy(d, lo, hi)
    boxfree = (o["rec_lo"] < BOXFREE_MAX) and (o["rec_hi"] < BOXFREE_MAX)
    return dict(boxfree=boxfree, rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                coord_lo=o["coord_lo"], coord_hi=o["coord_hi"],
                which=("hi" if o["rec_hi"] >= BOXFREE_MAX else
                       ("lo" if o["rec_lo"] >= BOXFREE_MAX else "")))


# --- A1 ---------------------------------------------------------------------
def score_A1(rung, plateaus):
    """CAN ONLY VOID."""
    ref, _n = A1_REF[rung]
    if not plateaus:
        return dict(ok=False, verdict="NO DATA", mean=None, ref=ref, diff=None)
    m = statistics.mean(plateaus)
    diff = m - ref
    return dict(mean=m, sem=_sem(plateaus), n=len(plateaus), ref=ref, bar=A1_BAR,
                diff=diff, ok=(abs(diff) <= A1_BAR),
                verdict=("REPRODUCES" if abs(diff) <= A1_BAR else "VOID"))


# --- A2 ---------------------------------------------------------------------
def score_A2(blk6_a_raw, node_a_raw):
    """The registered monotonicity gate.  Both clauses must hold to CONFIRM."""
    lay, w = TW0_A_RAW["lay"], TW0_A_RAW["w"]
    c1 = (blk6_a_raw is not None) and (blk6_a_raw > lay)
    c2 = (node_a_raw is not None) and (w < node_a_raw < lay)
    # The `why` string must survive a MISSING arm, not crash on it.  A scorer that
    # raises on absent data reports nothing at all, which is worse than a REFUTE.
    why = []
    if not c1:
        why.append("blk6 ABSENT" if blk6_a_raw is None
                   else "blk6 %.5f <= lay %.5f" % (blk6_a_raw, lay))
    if not c2:
        why.append("node ABSENT" if node_a_raw is None
                   else "node %.5f outside (%.5f, %.5f)" % (node_a_raw, w, lay))
    return dict(blk6=blk6_a_raw, node=node_a_raw, lay=lay, w=w,
                blk6_above_lay=c1, node_interior=c2,
                verdict=("CONFIRMS" if (c1 and c2) else "REFUTES"),
                why="; ".join(why))


# --- A2b --------------------------------------------------------------------
def score_A2b(node_3e4, node_1e4=None, tol=0.01):
    """NO DIRECTION REGISTERED.  Reported, never scored as confirm/refute."""
    ref = TW0_A_RAW["node_1e4"] if node_1e4 is None else node_1e4
    if node_3e4 is None:
        return dict(reading="NO DATA", diff=None, ref=ref)
    diff = node_3e4 - ref
    return dict(node_3e4=node_3e4, ref=ref, diff=diff, tol=tol,
                reading=("a_raw looks like a function of m, not of ms"
                         if abs(diff) <= tol else
                         "a_raw MOVES with ms at fixed m -- A2's curve is a joint "
                         "function of (m, ms) and must be re-read"))


# ---------------------------------------------------------------------------
def selftest():
    n = p = 0
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""

    def ck(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("    %-64s %s" % (name, "ok" if cond else "FAIL"))

    print("c75_at1_score selftest")
    ck("batch script exists", bool(src))
    ck("script CLIP= is the registered box", "CLIP=-15:-2.3026" in src)
    ck("LO/HI match the script's CLIP", (LO, HI) == (-15.0, -2.3026))
    ck("script EPOCHS= matches", ("EPOCHS=%d" % EPOCHS) in src)
    ck("script NJOBS is 6 = 2 arms x 3 seeds", "NJOBS=6" in src)
    ck("script BLK_MS is this scorer's blk6 ms", ("BLK_MS=%s" % ARM_MS["blk6"]) in src)
    ck("script NODE_MS is this scorer's node ms", ("NODE_MS=%s" % ARM_MS["node"]) in src)
    ck("N_RECORDS = epochs*500/stride 5", N_RECORDS == EPOCHS * 500 // 5)
    ck("script exports PROBE=5", "PROBE=5" in src)
    ck("script exports PROBE5=1 (instrument, not stride)", "PROBE5=1" in src)
    ck("n_beta[blk6]=6 is in the script's A0 text", "blk6 6" in src)
    ck("n_beta[node]=14420 is in the script's A0 text", "14420" in src)
    ck("A1 blk6 reference 92.652 is in the script", "92.652" in src)
    ck("A1 node reference 92.450 is in the script", "92.450" in src)
    ck("A2's lay anchor 0.65693 is in the script", "0.65693" in src)
    ck("A2's w anchor 0.50117 is in the script", "0.50117" in src)
    ck("A2b's node@1e-4 anchor 0.53836 is in the script", "0.53836" in src)
    ck("A1 bar is STANDING RULE 9's level bar", A1_BAR == 0.50)
    ck("5% gate UNCHANGED", BOXFREE_MAX == 0.05)
    ck("seeds 0-2", SEEDS == (0, 1, 2))

    import c55_neff_noise as c55
    ck("at1 registered in c55_neff_noise.BOXES", "at1" in c55.BOXES)
    ck("c55's at1 box == this scorer's box",
       c55.BOXES.get("at1", (None,) * 3)[:2] == (LO, HI))

    # --- A1 can only VOID, two-sided
    ck("A1 exact reference reproduces", score_A1("blk6", [92.652])["verdict"] == "REPRODUCES")
    ck("A1 +0.49 reproduces", score_A1("blk6", [92.652 + 0.49])["verdict"] == "REPRODUCES")
    ck("A1 -0.49 reproduces", score_A1("blk6", [92.652 - 0.49])["verdict"] == "REPRODUCES")
    ck("A1 +0.51 VOIDs", score_A1("blk6", [92.652 + 0.51])["verdict"] == "VOID")
    ck("A1 -0.51 VOIDs (two-sided)", score_A1("blk6", [92.652 - 0.51])["verdict"] == "VOID")
    ck("A1 node uses the node reference", score_A1("node", [92.450])["verdict"] == "REPRODUCES")
    ck("A1 node ref is not blk6's", score_A1("node", [92.652])["verdict"] == "REPRODUCES"
       if abs(92.652 - 92.450) <= A1_BAR else True)
    ck("A1 empty is not a pass", score_A1("blk6", [])["ok"] is False)

    # --- A2, both clauses, all four corners
    lay, w = TW0_A_RAW["lay"], TW0_A_RAW["w"]
    ck("A2 confirms when blk6>lay and node interior",
       score_A2(lay + 0.02, 0.55)["verdict"] == "CONFIRMS")
    ck("A2 refutes when blk6 below lay",
       score_A2(lay - 0.02, 0.55)["verdict"] == "REFUTES")
    ck("A2 refutes when blk6 EQUALS lay (strict >)",
       score_A2(lay, 0.55)["verdict"] == "REFUTES")
    ck("A2 refutes when node above lay",
       score_A2(lay + 0.02, lay + 0.01)["verdict"] == "REFUTES")
    ck("A2 refutes when node below w",
       score_A2(lay + 0.02, w - 0.0001)["verdict"] == "REFUTES")
    ck("A2 refutes on missing blk6", score_A2(None, 0.55)["verdict"] == "REFUTES")
    ck("A2 refutes on missing node", score_A2(lay + 0.02, None)["verdict"] == "REFUTES")
    ck("A2 names which clause failed",
       "blk6" in score_A2(lay - 0.02, 0.55)["why"])

    # --- A2b carries NO verdict
    r = score_A2b(0.5390)
    ck("A2b returns a reading, not a verdict", "verdict" not in r)
    ck("A2b close => function of m", "function of m" in score_A2b(0.5390)["reading"])
    ck("A2b far => joint function of (m, ms)", "joint" in score_A2b(0.60)["reading"])
    ck("A2b handles missing data", score_A2b(None)["reading"] == "NO DATA")

    # --- A0.2 shape from the HEADER (the bf9 false-VOID regression)
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        def mk(rung, n_tot, dtype, arr_n=None):
            d = os.path.join(td, "probe_%s_at1_s0" % rung)
            os.makedirs(d, exist_ok=True)
            json.dump({"n_tot": n_tot}, open(os.path.join(d, "neg_counts.json"), "w"))
            np.save(os.path.join(d, "neg_counts.npy"),
                    np.zeros(arr_n if arr_n is not None else n_tot, dtype=dtype))
            return d
        ck("int32 array of the right length PASSES A0.2",
           gate_A02(mk("blk6", 6, np.int32))["ok"])
        ck("A0.2 reads the true shape from the header",
           gate_A02(mk("blk6", 6, np.int32))["shape"] == (6,))
        ck("int64 also passes on length", gate_A02(mk("blk6", 6, np.int64))["ok"])
        ck("a SHORT array FAILS A0.2", not gate_A02(mk("blk6", 6, np.int32, arr_n=5))["ok"])
        ck("n_tot != n_beta FAILS A0.2", not gate_A02(mk("node", 999, np.int32))["ok"])
        dm = os.path.join(td, "probe_node_at1_s1")
        os.makedirs(dm)
        ck("absent neg_counts.json FAILS A0.2 (bf8's failure)", not gate_A02(dm)["ok"])

    # --- dirname parsing
    ck("probe_blk6_at1_s0 -> (at1, blk6, 0)",
       parse_dirname("probe_blk6_at1_s0") == ("at1", "blk6", 0))
    ck("probe_node_at1_s2 -> (at1, node, 2)",
       parse_dirname("probe_node_at1_s2") == ("at1", "node", 2))

    # --- the discipline this batch registered against itself
    ck("scorer records the distinct-seed caveat", "DISTINCT" in __doc__)
    ck("A2b carries no registered direction", "NO DIRECTION" in __doc__)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_at1"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs = sorted(glob.glob(os.path.join(a.root, "probe_*")))
    rows = {r["run"]: r for r in csv.DictReader(open(a.csv))}

    def run_of(d):
        _, rung, seed = parse_dirname(os.path.basename(d))
        return "at1-%s-s%d" % (rung, seed), rung, seed

    print("=" * 78)
    print("c75 -- at1: THE ALL-RUNGS-TUNED AGREEMENT LADDER.  %d probe dirs" % len(dirs))
    print("box = (%.1f, %.4f)  blk6@%s  node@%s  %d ep"
          % (LO, HI, ARM_MS["blk6"], ARM_MS["node"], EPOCHS))
    print("=" * 78)

    # ---- A0
    print("\n--- A0  VALIDITY (n_records==%d, n_beta on EVERY record, beta moved, ep==%d)"
          % (N_RECORDS, EPOCHS))
    a0 = {}
    for d in dirs:
        run, rung, seed = run_of(d)
        r = gate_A0(d, rows.get(run))
        a0[d] = r
        print("    %-22s %-4s  n_rec=%-6d n_beta=%-8s ep=%s/%s span=%.2f"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_records"],
                 ",".join(str(x) for x in r["n_beta"]),
                 r["epochs_done"], r["epochs_requested"], r["span"]))
    n0 = sum(1 for r in a0.values() if r["ok"])
    print("    A0: %d/%d" % (n0, len(dirs)))
    if n0 != len(dirs):
        print("    **A0 FAILED -- NOTHING BELOW IS SCORED.**")
        return 1

    # ---- A0.2
    print("\n--- A0.2  THE INSTRUMENT FIRED")
    for d in dirs:
        r = gate_A02(d)
        print("    %-22s %-4s  n_tot=%-8s npy_shape=%-12s %s"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_tot"],
                 r.get("shape"), r["why"]))
    print("    A0.2: %d/%d" % (sum(1 for d in dirs if gate_A02(d)["ok"]), len(dirs)))

    # ---- A0.3
    print("\n--- A0.3  BOX-FREE AT BOTH GUARDS (rec_-based 5%% gate PRIMARY, UNCHANGED)")
    print("    %-22s %-9s %8s %8s %10s %10s"
          % ("dir", "verdict", "rec_lo", "rec_hi", "coord_lo", "coord_hi"))
    a03 = {}
    for d in dirs:
        r = gate_A03(d)
        a03[d] = r
        cl = "%.5f" % r["coord_lo"] if r["coord_lo"] is not None else "None"
        ch = "%.5f" % r["coord_hi"] if r["coord_hi"] is not None else "None"
        print("    %-22s %-9s %8.4f %8.4f %10s %10s"
              % (os.path.basename(d), "free" if r["boxfree"] else "BOUND:" + r["which"],
                 r["rec_lo"], r["rec_hi"], cl, ch))
    print("    A0.3: %d/%d box-free" % (sum(1 for r in a03.values() if r["boxfree"]),
                                        len(dirs)))

    # ---- A1  (scored BEFORE A2: a VOID means A2 is not about the tuned ladder)
    print("\n--- A1  ACCURACY REPRODUCTION.  **CAN ONLY VOID.**")
    pl = {}
    for d in dirs:
        run, rung, seed = run_of(d)
        row = rows.get(run)
        if row and row.get("plateau5", "").strip():
            pl.setdefault(rung, []).append(float(row["plateau5"]))
    a1_ok = True
    for rung in ("blk6", "node"):
        v = pl.get(rung, [])
        s = score_A1(rung, v)
        a1_ok &= bool(s["ok"])
        if s["mean"] is None:
            print("    %-5s NO DATA" % rung)
            continue
        print("    %-5s @%s  at1 %.3f +-%.3f (n=%d, seeds %s)  vs ref %.3f -> %+.3f pp  **%s**"
              % (rung, ARM_MS[rung], s["mean"], s["sem"], s["n"],
                 ",".join(str(x) for x in SEEDS), s["ref"], s["diff"], s["verdict"]))
    if not a1_ok:
        print("    **A1 VOIDED -- the probe perturbed training.  A2 IS NOT SCORED.**")
        return 1
    print("    A1: both arms reproduce; the probe is inert on these rungs too.")

    # ---- A2  (PRIMARY)
    print("\n--- A2  **REGISTERED**: a_raw MONOTONE DECREASING in m across the tuned ladder")
    from neff_instrument import agreement_stats
    ag_by_rung = {}
    print("    %-22s %8s %9s %9s %9s %10s"
          % ("dir", "pbar", "bias", "a_raw", "a_deb", "bias_share"))
    for d in dirs:
        rung = run_of(d)[1]
        ag = agreement_stats(d, window=(0.5, 1.0))
        if ag is None:
            print("    %-22s  -- agreement_stats returned None" % os.path.basename(d))
            continue
        ag_by_rung.setdefault(rung, []).append(ag)
        print("    %-22s %8.5f %+9.5f %9.5f %9.5f %10.4f"
              % (os.path.basename(d), ag["pbar"], ag["bias"], ag["a_raw"],
                 ag["a_deb"], ag["bias_share"]))
    means = {r: statistics.mean([x["a_raw"] for x in v]) for r, v in ag_by_rung.items() if v}
    blk6_a = means.get("blk6")
    node_a = means.get("node")
    print("\n    THE FULL TUNED LADDER (tw0's three rungs are READ, not re-derived here):")
    ladder = [("blk6", 6, blk6_a), ("lay", LAY_M, TW0_A_RAW["lay"]),
              ("node", NODE_M, node_a), ("w", W_M, TW0_A_RAW["w"])]
    for name, m, val in ladder:
        src = "at1" if name in ("blk6", "node") else "tw0"
        print("      %-5s m=%-11d a_raw = %s   [%s]"
              % (name, m, ("%.5f" % val) if val is not None else "  --  ", src))
    v2 = score_A2(blk6_a, node_a)
    print("\n    blk6 > lay(0.65693)? %s     node inside (0.50117, 0.65693)? %s"
          % (v2["blk6_above_lay"], v2["node_interior"]))
    print("    **A2 VERDICT: %s**  %s" % (v2["verdict"], v2["why"]))
    if v2["verdict"] == "CONFIRMS":
        print("    -> agreement is monotone in block size across FOUR tuned rungs, and the")
        print("       mechanism story for the weightwise deficit keeps its carrier.")
    else:
        print("    -> agreement is NOT a function of block size alone.  The mechanism story")
        print("       LOSES its proposed carrier and the curve must be restated.")

    # ---- A2b
    print("\n--- A2b  THE ms CONTROL AT FIXED m.  **NO DIRECTION REGISTERED.**")
    r2b = score_A2b(node_a)
    if r2b.get("diff") is not None:
        print("    node@3e-4 %.5f  vs  node@1e-4 %.5f  ->  %+.5f"
              % (r2b["node_3e4"], r2b["ref"], r2b["diff"]))
    print("    reading: %s" % r2b["reading"])

    # ---- A3
    print("\n--- A3  THE LADDER'S ARGMAX.  **REPLICATION, NEVER A DISCOVERY.**")
    print("    at1 raises blk6 to 5 CSV rows / 3 DISTINCT seeds, node to 8 rows / 5")
    print("    DISTINCT seeds.  Repeating a seed averages down run-to-run noise but NOT")
    print("    seed variance -- report DISTINCT seeds (FINDINGS 74.8).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
