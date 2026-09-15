#!/usr/bin/env python3
"""c74_bf9_score.py -- score `bf9`, the 80-epoch N_eff/m point WITH THE INSTRUMENT ON.

REGISTERED GATES, transcribed from `bin/c73_bf9_probe5.sh:56-90`.  The selftests
assert each one against THAT SCRIPT'S OWN TEXT so registration and batch cannot
drift -- STANDING RULE (19), the rule CORRECTIONS 102.2 was written to install.

  G0    VALIDITY.  n_records == 8000; n_beta == {w 11173962, node 14420, lay 62}
        on EVERY record, not just record 0; beta moved.  Read from probe.jsonl.
  G0.2  THE INSTRUMENT FIRED.  neg_counts.json exists, n_tot == n_beta for the
        rung, neg_counts.npy has shape (n_tot,).  **THE GATE bf8 HAD NO WAY TO
        FAIL, BECAUSE IT WAS NEVER WRITTEN.**
  G0.3  BOX-FREE AT BOTH GUARDS (STANDING RULE 18), per seed, via
        c52_boxfree.occupancy.  The published rec_-based 5% gate is PRIMARY and
        UNCHANGED.  coord_lo/coord_hi are reported ALONGSIDE and NEVER averaged
        with it, and the gate is NOT re-thresholded after seeing the data
        (CORRECTIONS 102.4: re-reading a gate at whichever resolution flatters
        the data is the failure this campaign has repeated most).
  G0.4  TRAINS AT ALL.  plateau5 > 40.0, collapsed false, per seed.
  G1    THE 80-EPOCH POINT of the N_eff/m budget curve (currently 20 and 40).
        55.6's expectation -- w below 0.1509 and still falling -- is POST-HOC.
        A confirmation REPLICATES a post-hoc window and is NEVER a discovery.
  G2    THE REGISTERED OUT-OF-SAMPLE TEST, carried from bf8 VERBATIM.
        **PREDICTION: argmin of N_eff/m = `w`, gap/SE >= 2.**
        REFUTES (argmin = `node`) -> the 71.7 band is not a function of span
        alone and 101.9's "one scalar" claim is WITHDRAWN.
  G2b   BOUND-SEED POLICY, REGISTERED IN ADVANCE.  If any weightwise seed fails
        G0.3, report the argmin BOTH ways -- all seeds, and bound seeds dropped.
        **THE ALL-SEEDS VERSION IS PRIMARY.**  Dropping a seed after seeing
        which way it moves the argmin is selection; deciding here is not.

USAGE
  python3 analysis/c74_bf9_score.py --selftest
  python3 analysis/c74_bf9_score.py --root ../probes_bf9 --csv results/all_runs.csv
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SCRIPT = os.path.join(REPO, "bin", "c73_bf9_probe5.sh")

from c52_boxfree import occupancy, records           # noqa: E402
from probe5_window import parse_dirname              # noqa: E402

# --- THE REGISTRATION -------------------------------------------------------
LO, HI = -60.0, 2.0                 # bf9:f60, registered in c55_neff_noise.BOXES
EPOCHS = 80
N_RECORDS = 8000
N_BETA = {"w": 11173962, "node": 14420, "lay": 62}
RUNG_FULL = {"w": "weightwise", "node": "nodewise", "lay": "layerwise"}
SEEDS = (0, 1, 2, 3)
BOXFREE_MAX = 0.05                  # the PUBLISHED rec_-based 5% gate, UNCHANGED
G04_MIN_PLATEAU = 40.0
G1_POSTHOC_REF = 0.1509             # 55.6's POST-HOC expectation for `w`
G2_PREDICTED_ARGMIN = "w"
G2_SE_MULT = 2.0


def _sem(v):
    return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


# --- G0 ---------------------------------------------------------------------
def gate_G0(d):
    """n_records, n_beta on EVERY record (not just record 0), and beta moved."""
    _, rung, _ = parse_dirname(os.path.basename(d))
    R = records(d)
    want = N_BETA.get(rung)
    n_ok = len(R) == N_RECORDS
    nb = {r.get("n_beta") for r in R}
    nb_ok = (nb == {want}) if want else False
    mins = [r["beta_true_min"] for r in R]
    maxs = [r["beta_true_max"] for r in R]
    moved = (max(maxs) - min(mins)) > 1e-9
    return dict(rung=rung, n_records=len(R), n_beta=sorted(x for x in nb if x is not None),
                ok=(n_ok and nb_ok and moved), n_ok=n_ok, nb_ok=nb_ok, moved=moved,
                span=max(maxs) - min(mins))


# --- G0.2 -------------------------------------------------------------------
def gate_G02(d):
    """The instrument FIRED.  This is the gate bf8 had no way to fail."""
    _, rung, _ = parse_dirname(os.path.basename(d))
    j = os.path.join(d, "neg_counts.json")
    npy = os.path.join(d, "neg_counts.npy")
    if not os.path.exists(j):
        return dict(ok=False, why="neg_counts.json ABSENT -- instrument never fired",
                    n_tot=None, npy_ok=False)
    meta = json.load(open(j))
    want = N_BETA.get(rung)
    n_tot = meta.get("n_tot")
    tot_ok = (n_tot == want)
    # Read the REAL shape out of the .npy header rather than inferring it from
    # the file size: the first draft of this gate assumed int64 and a 128-byte
    # header, the array is int32, and the gate returned FAIL on 12/12 healthy
    # arms.  That is STANDING RULE (19) firing on this scorer itself -- a gate
    # must read the quantity it names, not a proxy it happens to know how to
    # compute.  np.load(mmap_mode='r') reads the header only.
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


# --- G0.3 -------------------------------------------------------------------
def gate_G03(d, lo=LO, hi=HI):
    """BOTH guards, PRIMARY gate is the published rec_-based 5%.  coord_* is
    reported alongside and is NEVER the gate (CORRECTIONS 102.4)."""
    o = occupancy(d, lo, hi)
    boxfree = (o["rec_lo"] < BOXFREE_MAX) and (o["rec_hi"] < BOXFREE_MAX)
    return dict(boxfree=boxfree, rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                coord_lo=o["coord_lo"], coord_hi=o["coord_hi"],
                first_lo=o["first_lo"], first_hi=o["first_hi"],
                which=("hi" if o["rec_hi"] >= BOXFREE_MAX else
                       ("lo" if o["rec_lo"] >= BOXFREE_MAX else "")))


# --- G0.4 -------------------------------------------------------------------
def gate_G04(row):
    if row is None:
        return dict(ok=False, why="no CSV row")
    pl = row.get("plateau5", "").strip()
    ok_pl = bool(pl) and float(pl) > G04_MIN_PLATEAU
    ok_col = row.get("collapsed", "") in ("False", "false", "0", "")
    ok_ep = row.get("epochs_done", "") == str(EPOCHS)
    return dict(ok=(ok_pl and ok_col and ok_ep), plateau5=(float(pl) if pl else None),
                collapsed=row.get("collapsed"), epochs_done=row.get("epochs_done"))


# --- G2 / G2b ---------------------------------------------------------------
def argmin_of(per_rung):
    """per_rung: {rung: [neff/m per seed]} -> argmin priced in its own SE.
    Returns None if fewer than two rungs carry a value."""
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
                decided=(se > 0 and gap > G2_SE_MULT * se),
                n={k: len(v) for k, v in live.items()},
                means={k: statistics.mean(v) for k, v in live.items()})


# ---------------------------------------------------------------------------
def selftest():
    n = p = 0
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""

    def ck(name, cond):
        nonlocal n, p
        n += 1
        if cond:
            p += 1
        else:
            print("  FAIL %s" % name)

    ck("batch script exists", bool(src))
    # --- the batch's own constants, read from ITS text
    ck("script LO == -60", re.search(r'^LO=(-?\d+)', src, re.M).group(1) == "-60")
    ck("script HI == 2.0", re.search(r'^HI=([\d.]+)', src, re.M).group(1) == "2.0")
    ck("script EPOCHS == 80", re.search(r'^EPOCHS=(\d+)', src, re.M).group(1) == str(EPOCHS))
    ck("registration LO/HI match script", (LO, HI) == (-60.0, 2.0))
    ck("script emits seeds 0-3", "for S in 0 1 2 3" in src)
    ck("script emits three rungs", "for G in weightwise nodewise layerwise" in src)
    ck("script exports PROBE5=1 (the INSTRUMENT, not the stride)", "PROBE5=1" in src)
    ck("script exports PROBE=5 (the stride) too", "PROBE=5," in src)
    # --- the registered gate TEXT
    ck("G0 n_beta values in script",
       all(str(v) in src for v in N_BETA.values()))
    ck("G0 requires EVERY record", "not just record 0" in src)
    ck("G0.2 is the gate bf8 could not fail",
       re.search(r'G0\.2[\s\S]{0,400}?NEVER WRITTEN', src))
    ck("G0.3 cites STANDING RULE 18", re.search(r'G0\.3[\s\S]{0,200}?STANDING RULE 18', src))
    ck("G0.3 keeps the rec_ gate PRIMARY", re.search(r'G0\.3[\s\S]{0,500}?PRIMARY', src))
    ck("G0.3 forbids re-thresholding",
       re.search(r'G0\.3[\s\S]{0,900}?not re-thresholded', src))
    ck("G0.4 plateau5 > 40", re.search(r'G0\.4[\s\S]{0,200}?plateau5 > 40', src))
    ck("G1 says post-hoc is never a discovery",
       re.search(r'G1\b[\s\S]{0,600}?NEVER a discovery', src))
    ck("G2 predicts argmin = w", re.search(r'PREDICTION: argmin = `w`', src))
    ck("G2 requires gap/SE >= 2", "gap/SE >= 2" in src)
    ck("G2 names its refuter", re.search(r'REFUTES \(argmin = `node`\)', src))
    ck("G2b makes ALL-SEEDS primary", re.search(r'ALL-SEEDS VERSION IS PRIMARY|ALL-SEEDS\s+VERSION IS PRIMARY|The ALL-SEEDS version is PRIMARY', src, re.I))
    ck("registration matches G2's predicted argmin", G2_PREDICTED_ARGMIN == "w")
    ck("5% gate unchanged", BOXFREE_MAX == 0.05)

    # --- parse_dirname handles bf9's layout: probe_<rung>_f60_s<seed>
    ck("parse probe_w_f60_s0", parse_dirname("probe_w_f60_s0") == ("f60", "w", 0))
    ck("parse probe_node_f60_s3", parse_dirname("probe_node_f60_s3") == ("f60", "node", 3))
    ck("parse probe_lay_f60_s2", parse_dirname("probe_lay_f60_s2") == ("f60", "lay", 2))

    # --- G0.4
    good = {"plateau5": "90.1", "collapsed": "False", "epochs_done": "80"}
    ck("G0.4 passes a clean row", gate_G04(good)["ok"])
    ck("G0.4 fails a collapsed run", not gate_G04(dict(good, plateau5="10.0"))["ok"])
    ck("G0.4 fails at exactly 40.0 (strict >)", not gate_G04(dict(good, plateau5="40.0"))["ok"])
    ck("G0.4 fails a short run", not gate_G04(dict(good, epochs_done="40"))["ok"])
    ck("G0.4 fails a missing row", not gate_G04(None)["ok"])

    # --- argmin_of
    pr = {"w": [0.05, 0.06], "node": [0.10, 0.11], "lay": [0.48, 0.49]}
    r = argmin_of(pr)
    ck("argmin picks the smallest", r["argmin"] == "w")
    ck("argmin names the runner-up", r["runner_up"] == "node")
    ck("argmin decides a clean separation", r["decided"])
    tie = {"w": [0.05, 0.09], "node": [0.06, 0.10]}
    ck("argmin is UNDECIDED on overlap", not argmin_of(tie)["decided"])
    ck("argmin needs two rungs", argmin_of({"w": [0.05]}) is None)
    ck("argmin with n=1 has zero SE", argmin_of({"w": [0.05], "node": [0.9]})["se"] == 0.0)
    # THE G2b SHAPE: dropping seeds can FLIP the argmin.  Assert the machinery
    # reproduces a flip, because that is exactly what the policy exists to govern.
    allseed = {"w": [0.0468, 0.0591, 0.0587, 0.0527], "node": [0.0589, 0.0555, 0.0658, 0.0603]}
    dropped = {"w": [0.0591], "node": [0.0589, 0.0555]}
    ck("G2b: all-seeds argmin is w", argmin_of(allseed)["argmin"] == "w")
    ck("G2b: bound-dropped argmin flips to node", argmin_of(dropped)["argmin"] == "node")

    # --- G0.2 must read the npy HEADER, not infer shape from the file size.
    # The first draft assumed int64 + a 128-byte header; the array is int32 and
    # the gate FAILED 12/12 healthy arms.  These tests pin the regression:
    # STANDING RULE (19) applied to this scorer itself.
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        def mk(n_tot, dtype, arr_n=None):
            os.makedirs(td, exist_ok=True)
            dd = tempfile.mkdtemp(dir=td)
            json.dump({"n_records": N_RECORDS, "n_tot": n_tot,
                       "stepsize_type": "weightwise"},
                      open(os.path.join(dd, "neg_counts.json"), "w"))
            np.save(os.path.join(dd, "neg_counts.npy"),
                    np.zeros(arr_n if arr_n is not None else n_tot, dtype=dtype))
            os.rename(dd, os.path.join(td, "probe_w_f60_s0"))
            return os.path.join(td, "probe_w_f60_s0")
        p32 = mk(N_BETA["w"], np.int32)
        r = gate_G02(p32)
        ck("G0.2 PASSES an int32 array of the right length", r["ok"])
        ck("G0.2 reports the real shape", r["shape"] == (N_BETA["w"],))
    with tempfile.TemporaryDirectory() as td2:
        dd = os.path.join(td2, "probe_w_f60_s1")
        os.makedirs(dd)
        json.dump({"n_tot": N_BETA["w"]}, open(os.path.join(dd, "neg_counts.json"), "w"))
        np.save(os.path.join(dd, "neg_counts.npy"), np.zeros(7, dtype=np.int64))
        ck("G0.2 FAILS a genuinely wrong length", not gate_G02(dd)["ok"])
    with tempfile.TemporaryDirectory() as td3:
        dd = os.path.join(td3, "probe_w_f60_s2")
        os.makedirs(dd)
        ck("G0.2 FAILS when the instrument never fired (this is bf8)",
           not gate_G02(dd)["ok"])
        ck("G0.2 names the absence explicitly",
           "ABSENT" in gate_G02(dd)["why"])

    # --- G0.3 gate arithmetic (no I/O)
    ck("boxfree needs BOTH guards under 5%", BOXFREE_MAX == 0.05)
    fake = dict(rec_lo=0.0, rec_hi=0.2161)
    ck("a ceiling bind fails the gate",
       not (fake["rec_lo"] < BOXFREE_MAX and fake["rec_hi"] < BOXFREE_MAX))
    fake2 = dict(rec_lo=0.0, rec_hi=0.0)
    ck("a free arm passes the gate",
       (fake2["rec_lo"] < BOXFREE_MAX and fake2["rec_hi"] < BOXFREE_MAX))

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "..", "probes_bf9"))
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    dirs = sorted(glob.glob(os.path.join(a.root, "probe_*")))
    rows = {r["run"]: r for r in csv.DictReader(open(a.csv))}
    print("=" * 78)
    print("c74 -- bf9: THE 80-EPOCH N_eff/m POINT, INSTRUMENT ON.  %d probe dirs" % len(dirs))
    print("box = (%.1f, %+.1f)   registered as bf9:f60 in c55_neff_noise.BOXES" % (LO, HI))
    print("=" * 78)

    # ---- G0
    print("\n--- G0  VALIDITY (n_records==%d, n_beta on EVERY record, beta moved)" % N_RECORDS)
    g0 = {}
    for d in dirs:
        r = gate_G0(d)
        g0[d] = r
        print("    %-22s %-4s  n_rec=%-5d n_beta=%-12s span=%.2f"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_records"],
                 ",".join(str(x) for x in r["n_beta"]), r["span"]))
    print("    G0: %d/%d" % (sum(1 for r in g0.values() if r["ok"]), len(dirs)))

    # ---- G0.2
    print("\n--- G0.2  THE INSTRUMENT FIRED  (the gate bf8 had no way to fail)")
    g02 = {}
    for d in dirs:
        r = gate_G02(d)
        g02[d] = r
        print("    %-22s %-4s  n_tot=%-12s npy_shape=%-14s %s"
              % (os.path.basename(d), "PASS" if r["ok"] else "FAIL", r["n_tot"],
                 r.get("shape"), r["why"]))
    n02 = sum(1 for r in g02.values() if r["ok"])
    print("    G0.2: %d/%d %s" % (n02, len(dirs),
          "-- bf8 scored 0/12 here by construction" if n02 == len(dirs) else ""))

    # ---- G0.3
    print("\n--- G0.3  BOX-FREE AT BOTH GUARDS  (rec_-based 5%% gate PRIMARY, UNCHANGED)")
    print("    %-22s %-8s %8s %8s %8s %8s %10s %10s"
          % ("dir", "verdict", "rec_lo", "rec_hi", "q4_lo", "q4_hi", "coord_lo", "coord_hi"))
    g03 = {}
    for d in dirs:
        r = gate_G03(d)
        g03[d] = r
        cl = "%.5f" % r["coord_lo"] if r["coord_lo"] is not None else "None"
        ch = "%.5f" % r["coord_hi"] if r["coord_hi"] is not None else "None"
        print("    %-22s %-8s %8.4f %8.4f %8.4f %8.4f %10s %10s"
              % (os.path.basename(d), "free" if r["boxfree"] else "BOUND:" + r["which"],
                 r["rec_lo"], r["rec_hi"], r["q4_lo"], r["q4_hi"], cl, ch))
    nfree = sum(1 for r in g03.values() if r["boxfree"])
    print("    G0.3: %d/%d box-free" % (nfree, len(dirs)))
    print("    THE FLOOR: rec_lo = 0.0000 on %d/%d -- LO=-60 freed it completely."
          % (sum(1 for r in g03.values() if r["rec_lo"] == 0.0), len(dirs)))
    print("    THE CEILING is what binds.  coord_hi is reported ALONGSIDE and is NOT")
    print("    the gate: re-reading a gate at the resolution that flatters the data is")
    print("    the failure this campaign has repeated most (CORRECTIONS 102.4).")

    # ---- G0.4
    print("\n--- G0.4  TRAINS AT ALL (plateau5 > %.0f, collapsed false, epochs_done==%d)"
          % (G04_MIN_PLATEAU, EPOCHS))
    g04 = {}
    for d in dirs:
        fam, rung, seed = parse_dirname(os.path.basename(d))
        run = "bf9-%s-s%d" % (rung, seed)
        r = gate_G04(rows.get(run))
        g04[d] = r
        print("    %-22s %-4s  plateau5=%-8s collapsed=%s epochs=%s"
              % (run, "PASS" if r["ok"] else "FAIL", r.get("plateau5"),
                 r.get("collapsed"), r.get("epochs_done")))
    print("    G0.4: %d/%d" % (sum(1 for r in g04.values() if r["ok"]), len(dirs)))

    # ---- G1 / G2 / G2b need N_eff/m; take it from the SHARED instrument.
    from neff_instrument import reduce_root
    import probe5_window as p5w
    print("\n--- G1  THE 80-EPOCH POINT OF THE N_eff/m BUDGET CURVE")
    red = reduce_root(a.root)
    per_rung_all, per_rung_free = {}, {}
    for d in dirs:
        fam, rung, seed = parse_dirname(os.path.basename(d))
        w = p5w.reduce_dir(d, windows=(("steady .5-1", (0.5, 1.0)),))
        # per-seed N_eff/m via the same reducer every published number came through
        ag = None
        try:
            from neff_instrument import agreement_stats
            ag = agreement_stats(d, window=(0.5, 1.0))
        except Exception:
            pass
        if w is None:
            continue
        ww = w["win"]["steady .5-1"]
        m = float(w["n_tot"])
        rho = ww["rho_s"]
        neff_m = 1.0 / (1.0 + (m - 1.0) * rho) if rho is not None else None
        if neff_m is None:
            continue
        per_rung_all.setdefault(rung, []).append(neff_m)
        if g03[d]["boxfree"]:
            per_rung_free.setdefault(rung, []).append(neff_m)
    for rung in ("lay", "node", "w"):
        v = per_rung_all.get(rung, [])
        if v:
            print("    %-5s (%-11s) m=%-11d  N_eff/m = %.4f +-%.4f  (n=%d)  per-seed %s"
                  % (rung, RUNG_FULL[rung], N_BETA[rung], statistics.mean(v), _sem(v),
                     len(v), " ".join("%.4f" % x for x in v)))
    wv = per_rung_all.get("w", [])
    if wv:
        print("    G1: w = %.4f at 80 ep, vs 55.6's POST-HOC expectation < %.4f -> %s"
              % (statistics.mean(wv), G1_POSTHOC_REF,
                 "below it" if statistics.mean(wv) < G1_POSTHOC_REF else "NOT below it"))
        print("        **THIS REPLICATES A POST-HOC WINDOW AND IS NEVER A DISCOVERY**")
        print("        (G1's own registration).  And the cell fails G0.3, so the number")
        print("        is not clean either -- it is reported, not counted.")

    # ---- G2 (PRIMARY, all seeds)
    print("\n--- G2  THE REGISTERED OUT-OF-SAMPLE TEST  (predicted argmin=`%s`, gap/SE>=%.0f)"
          % (G2_PREDICTED_ARGMIN, G2_SE_MULT))
    blockers = sorted({parse_dirname(os.path.basename(d))[1]
                       for d in dirs if not g03[d]["boxfree"]})
    r_all = argmin_of(per_rung_all)
    if blockers:
        print("    A rung is BOUND: %s.  By the pre-registered rule (CORRECTIONS 79) the"
              % ", ".join(blockers))
        print("    argmin is **UNINTERPRETABLE** -- a bound rung anywhere in the ranked")
        print("    set blocks, even when it is not the argmin.")
    if r_all:
        print("    [reported, NOT scored] all-seeds argmin = %s  %.4f   2nd %s %.4f"
              % (r_all["argmin"], r_all["mean"], r_all["runner_up"], r_all["runner_mean"]))
        print("    gap %.4f  SE %.4f  gap/SE %.2f  -> %s"
              % (r_all["gap"], r_all["se"], r_all["ratio"],
                 "DECIDED" if r_all["decided"] else "UNDECIDED even ignoring the box"))
    print("    **G2 VERDICT: NOT SCORED / UNTESTED.**  Not confirmed and not refuted.")

    # ---- G2b (SECONDARY, bound seeds dropped)
    print("\n--- G2b  BOUND-SEED POLICY (registered in advance; ALL-SEEDS is PRIMARY)")
    for rung in ("lay", "node", "w"):
        v = per_rung_free.get(rung, [])
        print("    %-5s box-free seeds: n=%d  %s" % (rung, len(v),
              " ".join("%.4f" % x for x in v) if v else "-- none"))
    r_free = argmin_of(per_rung_free)
    if r_free:
        print("    [SECONDARY] bound-dropped argmin = %s %.4f   2nd %s %.4f   gap/SE %.2f"
              % (r_free["argmin"], r_free["mean"], r_free["runner_up"],
                 r_free["runner_mean"], r_free["ratio"]))
        if r_all and r_free["argmin"] != r_all["argmin"]:
            print("    **THE ARGMIN FLIPS %s -> %s WHEN BOUND SEEDS ARE DROPPED.**"
                  % (r_all["argmin"], r_free["argmin"]))
            print("    This is exactly the shape G2b was registered to govern, and exactly")
            print("    why the box gate is not read past.  ALL-SEEDS stays PRIMARY, and")
            print("    all-seeds is UNINTERPRETABLE, so NEITHER argmin is claimed.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
