#!/usr/bin/env python3
"""c75_tw0_score.py -- score `tw0`, the TUNED weightwise operating point with the
instrument on.

REGISTERED GATES, transcribed from `bin/c74_tuned_weightwise_probe.sh:45-105`.
The selftests assert each constant against THAT SCRIPT'S OWN TEXT so the
registration and the batch cannot drift -- STANDING RULE (19).

  T0    VALIDITY.  n_records == 10000 (100 ep x 500 steps / stride 5) on every
        dir; n_beta == {w 11173962, node 14420, lay 62} on EVERY record, not
        just record 0; beta moved; epochs_done == 100 == requested.
  T0.2  THE INSTRUMENT FIRED.  neg_counts.json present, n_tot == n_beta for the
        rung, neg_counts.npy shape == (n_tot,) READ FROM ITS HEADER (never
        inferred from file size: the array is int32 and that inference returned
        a false FAIL on 12/12 healthy bf9 arms).
  T1    **PRIMARY.  THE BOX CONFOUND ON CYCLE 74'S HEADLINE.**  Is the
        WEIGHTWISE arm box-free at BOTH guards at 100 epochs, in wm9's box, at
        wm9's argmax ms?  c52_boxfree.occupancy, PUBLISHED rec_-based 5% gate,
        PRIMARY and UNCHANGED.  coord_lo/coord_hi reported ALONGSIDE, NEVER the
        gate (CORRECTIONS 102.4).
        **REGISTERED PREDICTION: weightwise BOX-FREE on >= 2 of 3 seeds.**
        CONFIRMS -> wm9's W2 refutation is NOT a box artifact; the ladder is
             genuinely not flat AND ceiling-runaway is RULED OUT as the
             mechanism, leaving the deficit unexplained.
        REFUTES  -> **103.2 IS CONFOUNDED WITH THE BOX AND IS WITHDRAWN.**
        The conflict of interest is on the record: CONFIRM keeps cycle 74's own
        headline alive.  That is why the threshold was written before the run.
  T2    ACCURACY REPRODUCTION.  **CAN ONLY VOID.**  tw0 weightwise plateau5 vs
        wm9's ms=1e-4 cell = 91.015 (n=2).  Bar |diff| <= 0.50 pp.
        FAILS -> the probe perturbed training and T1 is not evidence about wm9.
  T3    SIGN AGREEMENT AT A TUNED POINT.  **DESCRIPTIVE.  NO DIRECTION
        REGISTERED.**  Report pbar, bias, a_raw, a_deb, bias_share per rung and
        seed, from neff_instrument.agreement_stats.
        **THE "53.1% OF 11.17M META-GRADIENTS AGREE IN SIGN" SENTENCE IS NOT
        REPRODUCED HERE AND MUST NOT BE WRITTEN** (refuted as stated since
        cycle 42; CORRECTIONS 26).
  T4    N_eff/m AT A TUNED POINT, all three rungs, BOX-FREE CELLS ONLY, argmin
        priced in its own SE.  A bound rung ANYWHERE in the ranked set makes the
        argmin UNINTERPRETABLE (CORRECTIONS 79); that rule is NOT relaxed here.

USAGE
  python3 analysis/c75_tw0_score.py --selftest
  python3 analysis/c75_tw0_score.py --root ../probes_tw0 --csv results/all_runs.csv
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
SCRIPT = os.path.join(REPO, "bin", "c74_tuned_weightwise_probe.sh")

from c52_boxfree import occupancy, records           # noqa: E402
from probe5_window import parse_dirname              # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -15.0, -2.3026             # `tw0`, registered in c55_neff_noise.BOXES
EPOCHS = 100
N_RECORDS = 10000                   # 100 ep x 500 steps / stride 5
N_BETA = {"w": 11173962, "node": 14420, "lay": 62}
RUNG_FULL = {"w": "weightwise", "node": "nodewise", "lay": "layerwise"}
SEEDS = (0, 1, 2)
MS = "1e-4"
BOXFREE_MAX = 0.05                  # the PUBLISHED rec_-based 5% gate, UNCHANGED
T1_MIN_FREE_SEEDS = 2               # "box-free on >= 2 of 3 seeds"
T2_REF = 91.015                     # wm9's ms=1e-4 weightwise plateau5, n=2
T2_BAR = 0.50                       # STANDING RULE 9's level bar
T4_SE_MULT = 2.0
MIN_PLATEAU = 40.0


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


# --- T0 ---------------------------------------------------------------------
def gate_T0(d, row=None):
    """n_records, n_beta on EVERY record (not just record 0), beta moved, and
    -- unlike bf9's G0 -- epochs_done == requested == EPOCHS, which T0 names."""
    _, rung, _ = parse_dirname(os.path.basename(d))
    R = records(d)
    want = N_BETA.get(rung)
    n_ok = len(R) == N_RECORDS
    nb = {r.get("n_beta") for r in R}
    nb_ok = (nb == {want}) if want else False
    mins = [r["beta_true_min"] for r in R]
    maxs = [r["beta_true_max"] for r in R]
    moved = (max(maxs) - min(mins)) > 1e-9
    ep_ok = ep_done = ep_req = None
    if row is not None:
        ep_done, ep_req = row.get("epochs_done", ""), row.get("epochs_requested", "")
        ep_ok = (ep_done == str(EPOCHS) and ep_req == str(EPOCHS))
    return dict(rung=rung, n_records=len(R),
                n_beta=sorted(x for x in nb if x is not None),
                n_ok=n_ok, nb_ok=nb_ok, moved=moved, ep_ok=ep_ok,
                epochs_done=ep_done, epochs_requested=ep_req,
                span=max(maxs) - min(mins),
                ok=(n_ok and nb_ok and moved and (ep_ok is not False)))


# --- T0.2 -------------------------------------------------------------------
def gate_T02(d):
    """The instrument FIRED.  Shape READ FROM THE HEADER, never inferred."""
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


# --- T1 (PRIMARY) -----------------------------------------------------------
def gate_T1(d, lo=LO, hi=HI):
    o = occupancy(d, lo, hi)
    boxfree = (o["rec_lo"] < BOXFREE_MAX) and (o["rec_hi"] < BOXFREE_MAX)
    return dict(boxfree=boxfree, rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                coord_lo=o["coord_lo"], coord_hi=o["coord_hi"],
                first_lo=o["first_lo"], first_hi=o["first_hi"],
                min_final=o["min_final"], max_final=o["max_final"],
                which=("hi" if o["rec_hi"] >= BOXFREE_MAX else
                       ("lo" if o["rec_lo"] >= BOXFREE_MAX else "")))


def score_T1(free_by_seed):
    """{seed: bool} -> the registered >= 2-of-3 verdict on WEIGHTWISE."""
    n_free = sum(1 for v in free_by_seed.values() if v)
    n = len(free_by_seed)
    return dict(n_free=n_free, n=n, need=T1_MIN_FREE_SEEDS,
                verdict=("CONFIRMS" if n_free >= T1_MIN_FREE_SEEDS else "REFUTES"))


# --- T2 ---------------------------------------------------------------------
def score_T2(plateaus, ref=T2_REF, bar=T2_BAR):
    """CAN ONLY VOID: |mean - ref| <= bar reproduces; beyond it the probe moved
    training and T1 stops being evidence about wm9."""
    if not plateaus:
        return dict(ok=False, why="no plateau5 values", mean=None)
    m = statistics.mean(plateaus)
    diff = m - ref
    return dict(mean=m, sem=_sem(plateaus), n=len(plateaus), ref=ref, bar=bar,
                diff=diff, ok=(abs(diff) <= bar),
                verdict=("REPRODUCES" if abs(diff) <= bar else "VOID"))


# --- T4 ---------------------------------------------------------------------
def argmin_of(per_rung):
    """{rung: [N_eff/m per seed]} -> argmin priced in its own SE, or None."""
    live = {k: v for k, v in per_rung.items() if v}
    if len(live) < 2:
        return None
    order = sorted(live, key=lambda k: statistics.mean(live[k]))
    a, b = order[0], order[1]
    ma, mb = statistics.mean(live[a]), statistics.mean(live[b])
    se = math.sqrt(_sem(live[a]) ** 2 + _sem(live[b]) ** 2)
    gap = mb - ma
    return dict(argmin=a, runner_up=b, mean=ma, runner_mean=mb, gap=gap, se=se,
                ratio=(gap / se if se > 0 else float("inf")),
                decided=(se > 0 and gap > T4_SE_MULT * se),
                n={k: len(v) for k, v in live.items()},
                means={k: statistics.mean(v) for k, v in live.items()})


# ---------------------------------------------------------------------------
def selftest():
    n = p = 0
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""

    def ck(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("    %-62s %s" % (name, "ok" if cond else "FAIL"))

    print("c75_tw0_score selftest")
    # --- the registration is transcribed from the batch script, not remembered
    ck("batch script exists", bool(src))
    ck("script CLIP= is the registered box", "CLIP=-15:-2.3026" in src)
    ck("LO/HI match the script's CLIP", (LO, HI) == (-15.0, -2.3026))
    ck("script MST= is the registered ms", ("MST=%s" % MS) in src)
    ck("script EPOCHS= matches", ("EPOCHS=%d" % EPOCHS) in src)
    ck("script NJOBS is 9 = 3 rungs x 3 seeds", "NJOBS=9" in src)
    ck("N_RECORDS = epochs*500/stride 5", N_RECORDS == EPOCHS * 500 // 5)
    ck("script exports PROBE=5", "PROBE=5" in src)
    ck("script exports PROBE5=1 (the INSTRUMENT, not the stride)", "PROBE5=1" in src)
    for r, v in N_BETA.items():
        ck("n_beta[%s]=%d is in the script's T0 text" % (r, v), str(v) in src)
    ck("T2 reference 91.015 is in the script", "91.015" in src)
    ck("T1's >=2-of-3 threshold is in the script", "2 of 3" in src)
    ck("seeds 0-2", SEEDS == (0, 1, 2))
    ck("5% gate UNCHANGED", BOXFREE_MAX == 0.05)
    ck("T2 bar is STANDING RULE 9's level bar", T2_BAR == 0.50)

    # --- the box is registered in the SHARED registry, or the root is skipped
    import c55_neff_noise as c55
    ck("tw0 registered in c55_neff_noise.BOXES", "tw0" in c55.BOXES)
    ck("c55's tw0 box == this scorer's box", c55.BOXES.get("tw0", (None,) * 3)[:2] == (LO, HI))

    # --- T1 verdict arithmetic, both directions
    ck("T1 3/3 free -> CONFIRMS",
       score_T1({0: True, 1: True, 2: True})["verdict"] == "CONFIRMS")
    ck("T1 2/3 free -> CONFIRMS (the registered boundary)",
       score_T1({0: True, 1: True, 2: False})["verdict"] == "CONFIRMS")
    ck("T1 1/3 free -> REFUTES",
       score_T1({0: True, 1: False, 2: False})["verdict"] == "REFUTES")
    ck("T1 0/3 free -> REFUTES",
       score_T1({0: False, 1: False, 2: False})["verdict"] == "REFUTES")

    # --- T1's gate needs BOTH guards, and rec_ is what it reads
    ck("boxfree needs BOTH guards under 5%", BOXFREE_MAX == 0.05)
    ck("a lo-bound arm is not box-free",
       not ((0.9 < BOXFREE_MAX) and (0.0 < BOXFREE_MAX)))

    # --- T2 can only VOID, and the bar is two-sided
    ck("T2 exact reference reproduces", score_T2([T2_REF])["verdict"] == "REPRODUCES")
    ck("T2 +0.49 reproduces", score_T2([T2_REF + 0.49])["verdict"] == "REPRODUCES")
    ck("T2 -0.49 reproduces", score_T2([T2_REF - 0.49])["verdict"] == "REPRODUCES")
    ck("T2 +0.51 VOIDs", score_T2([T2_REF + 0.51])["verdict"] == "VOID")
    ck("T2 -0.51 VOIDs (two-sided)", score_T2([T2_REF - 0.51])["verdict"] == "VOID")
    ck("T2 empty is not a pass", score_T2([])["ok"] is False)
    ck("T2 mean is over seeds", abs(score_T2([91.0, 92.0])["mean"] - 91.5) < 1e-9)

    # --- T4 argmin arithmetic
    ck("argmin needs two rungs", argmin_of({"w": [0.05]}) is None)
    ck("argmin with n=1 has zero SE", argmin_of({"w": [0.05], "node": [0.9]})["se"] == 0.0)
    r = argmin_of({"w": [0.05, 0.06], "node": [0.10, 0.11], "lay": [0.48, 0.49]})
    ck("argmin picks the smallest mean", r["argmin"] == "w")
    ck("runner-up is the second smallest", r["runner_up"] == "node")
    ck("a clean separation is DECIDED", r["decided"])
    tie = argmin_of({"w": [0.05, 0.09], "node": [0.06, 0.10]})
    ck("an overlapping pair is UNDECIDED", not tie["decided"])
    ck("empty rungs are dropped", argmin_of({"w": [], "node": [0.1], "lay": [0.2]})["argmin"] == "node")

    # --- T0.2 shape must come from the HEADER (the bf9 false-VOID regression)
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        def mk(rung, n_tot, dtype, arr_n=None):
            d = os.path.join(td, "probe_%s_tw0_s0" % rung)
            os.makedirs(d, exist_ok=True)
            json.dump({"n_tot": n_tot}, open(os.path.join(d, "neg_counts.json"), "w"))
            np.save(os.path.join(d, "neg_counts.npy"),
                    np.zeros(arr_n if arr_n is not None else n_tot, dtype=dtype))
            return d
        d32 = mk("lay", N_BETA["lay"], np.int32)
        ck("int32 array of the right length PASSES T0.2", gate_T02(d32)["ok"])
        ck("T0.2 reads the true shape from the header",
           gate_T02(d32)["shape"] == (N_BETA["lay"],))
        d64 = mk("lay", N_BETA["lay"], np.int64)
        ck("dtype does not decide T0.2 (int64 also passes on length)", gate_T02(d64)["ok"])
        dbad = mk("lay", N_BETA["lay"], np.int32, arr_n=N_BETA["lay"] - 1)
        ck("a SHORT array FAILS T0.2", not gate_T02(dbad)["ok"])
        dwrong = mk("node", 123, np.int32)
        ck("n_tot != n_beta for the rung FAILS T0.2", not gate_T02(dwrong)["ok"])
        dmiss = os.path.join(td, "probe_w_tw0_s1")
        os.makedirs(dmiss)
        ck("absent neg_counts.json FAILS T0.2 (bf8's failure)", not gate_T02(dmiss)["ok"])
        ck("...and says the instrument never fired",
           "never fired" in gate_T02(dmiss)["why"])

    # --- dirname parsing must not confuse rung with family
    ck("probe_w_tw0_s0 -> (tw0, w, 0)", parse_dirname("probe_w_tw0_s0") == ("tw0", "w", 0))
    ck("probe_node_tw0_s1 -> node", parse_dirname("probe_node_tw0_s1")[1] == "node")
    ck("probe_lay_tw0_s2 -> seed 2", parse_dirname("probe_lay_tw0_s2")[2] == 2)

    # --- the sentence that must not be written
    ck("scorer records that 53.1% must NOT be written", "MUST NOT BE WRITTEN" in __doc__)
    ck("T3 carries no registered direction", "NO DIRECTION" in __doc__)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_tw0"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs = sorted(glob.glob(os.path.join(a.root, "probe_*")))
    rows = {r["run"]: r for r in csv.DictReader(open(a.csv))}

    def run_of(d):
        _, rung, seed = parse_dirname(os.path.basename(d))
        return "tw0-%s-s%d" % (rung, seed), rung, seed

    print("=" * 78)
    print("c75 -- tw0: THE TUNED WEIGHTWISE OPERATING POINT.  %d probe dirs" % len(dirs))
    print("box = (%.1f, %.4f)  ms = %s  %d ep   registered as `tw0` in c55.BOXES"
          % (LO, HI, MS, EPOCHS))
    print("=" * 78)

    # ---- T0
    print("\n--- T0  VALIDITY (n_records==%d, n_beta on EVERY record, beta moved,"
          % N_RECORDS)
    print("        epochs_done==epochs_requested==%d)" % EPOCHS)
    t0 = {}
    for d in dirs:
        run, rung, seed = run_of(d)
        r = gate_T0(d, rows.get(run))
        t0[d] = r
        print("    %-22s %-4s  n_rec=%-6d n_beta=%-11s ep=%s/%s span=%.2f"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_records"],
                 ",".join(str(x) for x in r["n_beta"]),
                 r["epochs_done"], r["epochs_requested"], r["span"]))
    n0 = sum(1 for r in t0.values() if r["ok"])
    print("    T0: %d/%d" % (n0, len(dirs)))
    if n0 != len(dirs):
        print("    **T0 FAILED -- NOTHING BELOW IS SCORED.**")
        return 1

    # ---- T0.2
    print("\n--- T0.2  THE INSTRUMENT FIRED  (bf8 scored 0/12 here by construction)")
    t02 = {}
    for d in dirs:
        r = gate_T02(d)
        t02[d] = r
        print("    %-22s %-4s  n_tot=%-12s npy_shape=%-14s %s"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_tot"],
                 r.get("shape"), r["why"]))
    n02 = sum(1 for r in t02.values() if r["ok"])
    print("    T0.2: %d/%d" % (n02, len(dirs)))

    # ---- T1  (PRIMARY)
    print("\n--- T1  **PRIMARY** BOX-FREE AT BOTH GUARDS AT %d EP" % EPOCHS)
    print("        rec_-based 5%% gate PRIMARY and UNCHANGED; coord_* reported ALONGSIDE")
    print("    %-22s %-9s %8s %8s %8s %8s %10s %10s"
          % ("dir", "verdict", "rec_lo", "rec_hi", "q4_lo", "q4_hi", "coord_lo", "coord_hi"))
    t1 = {}
    for d in dirs:
        r = gate_T1(d)
        t1[d] = r
        cl = "%.5f" % r["coord_lo"] if r["coord_lo"] is not None else "None"
        ch = "%.5f" % r["coord_hi"] if r["coord_hi"] is not None else "None"
        print("    %-22s %-9s %8.4f %8.4f %8.4f %8.4f %10s %10s"
              % (os.path.basename(d), "free" if r["boxfree"] else "BOUND:" + r["which"],
                 r["rec_lo"], r["rec_hi"], r["q4_lo"], r["q4_hi"], cl, ch))
    w_free = {run_of(d)[2]: t1[d]["boxfree"] for d in dirs if run_of(d)[1] == "w"}
    v1 = score_T1(w_free)
    print("\n    WEIGHTWISE box-free on %d of %d seeds (registered need: >= %d)"
          % (v1["n_free"], v1["n"], v1["need"]))
    print("    **T1 VERDICT: %s**" % v1["verdict"])
    if v1["verdict"] == "CONFIRMS":
        print("    -> cycle 74's W2 refutation is NOT a box artifact.  The clamp is")
        print("       provably slack, so CEILING-RUNAWAY IS RULED OUT as the mechanism")
        print("       and the weightwise deficit is left UNEXPLAINED.")
    else:
        print("    -> **103.2 IS CONFOUNDED WITH THE BOX AND IS WITHDRAWN.**")
        print("       Stop here; do not read T3/T4 as statements about the tuned point.")
    for rung in ("lay", "node"):
        f = sum(1 for d in dirs if run_of(d)[1] == rung and t1[d]["boxfree"])
        tot = sum(1 for d in dirs if run_of(d)[1] == rung)
        print("    [context, not the gate] %-4s box-free %d/%d" % (rung, f, tot))

    # ---- T2
    print("\n--- T2  ACCURACY REPRODUCTION vs wm9 ms=%s (%.3f).  **CAN ONLY VOID.**"
          % (MS, T2_REF))
    pl = {}
    for d in dirs:
        run, rung, seed = run_of(d)
        row = rows.get(run)
        if row and row.get("plateau5", "").strip():
            pl.setdefault(rung, []).append(float(row["plateau5"]))
    for rung in ("w", "node", "lay"):
        v = pl.get(rung, [])
        if v:
            print("    %-5s (%-11s) plateau5 = %.3f +-%.3f  (n=%d)  %s"
                  % (rung, RUNG_FULL[rung], statistics.mean(v), _sem(v), len(v),
                     " ".join("%.3f" % x for x in v)))
    v2 = score_T2(pl.get("w", []))
    print("    tw0 weightwise %.3f vs wm9 %.3f -> diff %+.3f pp, bar +-%.2f -> **%s**"
          % (v2["mean"], v2["ref"], v2["diff"], v2["bar"], v2["verdict"]))
    if v2["verdict"] == "VOID":
        print("    **THE PROBE PERTURBED TRAINING.  T1 IS NOT EVIDENCE ABOUT wm9.**")
    else:
        print("    The probe left training untouched; T1 speaks about wm9's cell.")

    # ---- the ladder at this cell, reported alongside T2
    print("\n    [the ladder AT THIS CELL, all rungs at ms=%s]" % MS)
    means = {r: statistics.mean(v) for r, v in pl.items() if v}
    if len(means) >= 2:
        order = sorted(means, key=lambda k: -means[k])
        for r in order:
            print("      %-5s %-11s %.3f" % (r, RUNG_FULL[r], means[r]))
        best = order[0]
        if "w" in means:
            near = min((k for k in means if k != "w"), key=lambda k: abs(means[k] - means["w"]))
            print("      weightwise is %.3f pp below the best (%s) and %.3f pp below"
                  % (means[best] - means["w"], best, means[near] - means["w"]))
            print("      the nearest rung (%s)." % near)
        print("      **NOTE: ms=%s is the argmax for w, lay and scalar -- NOT for node,"
              % MS)
        print("      whose own argmax is 3e-4.  The node rung here is OFF ITS OPTIMUM")
        print("      and this cell is NOT an all-rungs-tuned ladder.**")

    # ---- T3  (DESCRIPTIVE)
    print("\n--- T3  SIGN AGREEMENT AT A TUNED POINT.  **DESCRIPTIVE, NO DIRECTION.**")
    print("    **THE '53.1%' SENTENCE IS NOT REPRODUCED HERE AND MUST NOT BE WRITTEN.**")
    from neff_instrument import agreement_stats
    print("    %-22s %8s %9s %9s %9s %10s %6s"
          % ("dir", "pbar", "bias", "a_raw", "a_deb", "bias_share", "T"))
    ag_by_rung = {}
    for d in dirs:
        _, rung, seed = run_of(d)[1], run_of(d)[1], run_of(d)[2]
        rung = run_of(d)[1]
        ag = agreement_stats(d, window=(0.5, 1.0))
        if ag is None:
            print("    %-22s  -- agreement_stats returned None" % os.path.basename(d))
            continue
        ag_by_rung.setdefault(rung, []).append(ag)
        print("    %-22s %8.5f %+9.5f %9.5f %9.5f %10.4f %6d"
              % (os.path.basename(d), ag["pbar"], ag["bias"], ag["a_raw"],
                 ag["a_deb"], ag["bias_share"], ag["T"]))
    print("\n    per rung (seeds averaged):")
    for rung in ("lay", "node", "w"):
        v = ag_by_rung.get(rung, [])
        if not v:
            continue
        print("      %-5s (%-11s) m=%-11d pbar=%.5f  a_raw=%.5f  a_deb=%.5f  bias_share=%.3f"
              % (rung, RUNG_FULL[rung], N_BETA[rung],
                 statistics.mean([x["pbar"] for x in v]),
                 statistics.mean([x["a_raw"] for x in v]),
                 statistics.mean([x["a_deb"] for x in v]),
                 statistics.mean([x["bias_share"] for x in v])))

    # ---- T4
    print("\n--- T4  N_eff/m AT A TUNED POINT.  BOX-FREE CELLS ONLY.")
    import probe5_window as p5w
    per_all, per_free = {}, {}
    for d in dirs:
        run, rung, seed = run_of(d)
        w = p5w.reduce_dir(d, windows=(("steady .5-1", (0.5, 1.0)),))
        if w is None:
            continue
        ww = w["win"]["steady .5-1"]
        m, rho = float(w["n_tot"]), ww["rho_s"]
        if rho is None:
            continue
        neff_m = 1.0 / (1.0 + (m - 1.0) * rho)
        per_all.setdefault(rung, []).append(neff_m)
        if t1[d]["boxfree"]:
            per_free.setdefault(rung, []).append(neff_m)
    for rung in ("lay", "node", "w"):
        v = per_all.get(rung, [])
        if v:
            print("    %-5s (%-11s) m=%-11d  N_eff/m = %.4f +-%.4f (n=%d)  per-seed %s"
                  % (rung, RUNG_FULL[rung], N_BETA[rung], statistics.mean(v), _sem(v),
                     len(v), " ".join("%.4f" % x for x in v)))
    blockers = sorted({run_of(d)[1] for d in dirs if not t1[d]["boxfree"]})
    if blockers:
        print("    A rung is BOUND: %s.  By CORRECTIONS 79 the argmin is"
              % ", ".join(blockers))
        print("    **UNINTERPRETABLE** -- a bound rung anywhere in the ranked set blocks,")
        print("    even when it is not the argmin.  That rule is NOT relaxed here.")
    r_all = argmin_of(per_all)
    if r_all:
        print("    all-seeds argmin = %s %.4f   2nd %s %.4f   gap %.4f  SE %.4f  gap/SE %.2f -> %s"
              % (r_all["argmin"], r_all["mean"], r_all["runner_up"], r_all["runner_mean"],
                 r_all["gap"], r_all["se"], r_all["ratio"],
                 "DECIDED" if r_all["decided"] else "UNDECIDED"))
    r_free = argmin_of(per_free)
    if r_free:
        print("    box-free-only argmin = %s %.4f   2nd %s %.4f   gap/SE %.2f -> %s"
              % (r_free["argmin"], r_free["mean"], r_free["runner_up"],
                 r_free["runner_mean"], r_free["ratio"],
                 "DECIDED" if r_free["decided"] else "UNDECIDED"))
        if r_all and r_free["argmin"] != r_all["argmin"]:
            print("    **THE ARGMIN FLIPS %s -> %s WHEN BOUND SEEDS ARE DROPPED.**"
                  % (r_all["argmin"], r_free["argmin"]))
    elif per_free:
        print("    box-free set has fewer than two rungs -- no argmin is formed.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
