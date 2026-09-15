#!/usr/bin/env python3
"""
c72_bf8_score.py -- score `bf8` against the gates REGISTERED in
bin/c71_floor_budget.sh (cycle 71), in the order that script prescribes:

    F0    VALIDITY               n_records == 8000, n_beta byte-match, beta moved
    F0.3  THE BOX-FREE GATE      per seed, per FLOOR, AT BOTH GUARDS (RULE 18).
                                 < 5% of records at LO **and** < 5% at HI.
                                 THE LO COLUMN IS THE ONE bd7 FAILED.
    F0.4  TRAINS-AT-ALL          plateau5 > 40.0 and `collapsed` false, per seed.
    F0.5  FLOOR-INVARIANCE       on rungs box-free at BOTH floors,
                                 |N_eff/m(-60) - N_eff/m(-90)| <= per-rung bar
                                 (w 0.02 / node 0.06 / lay 0.11, measured on br6).
                                 FAIL is a BIGGER result than the budget curve.
    F1    THE 80-EPOCH POINT     branches (a) keeps falling (b) floor ~0.15
                                 (c) rebounds.  55.6's written-down expectation:
                                 weightwise below 0.1509 and still falling -- a
                                 confirmation REPLICATES that curve, never discovers.
    F2    THE SPAN BAND, OUT OF SAMPLE.  PREDICTION argmin = `w`, gap/SE >= 2.
                                 argmin = `node` REFUTES the band as a function of
                                 span alone and WITHDRAWS 101.9's "one scalar".

NOTHING IS RESTATED.  Occupancy comes from `c52_boxfree` (27/27), N_eff/m and the
argmin-in-SE pricing from `c55_neff_noise` (51/51), and the per-arm gate helpers are
IMPORTED from `c71_bd7_bo7_score` (14/14) -- the same modules that scored bd7/bo7 and
every published N_eff number.  A disagreement is a bug HERE.

    python3 analysis/c72_bf8_score.py --selftest
    python3 analysis/c72_bf8_score.py --score
"""
import csv
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(HERE)
PARENT = os.path.dirname(REPO)

import c52_boxfree as BF            # noqa: E402
import c55_neff_noise as C55        # noqa: E402
import c71_bd7_bo7_score as C71     # noqa: E402

CSV = os.path.join(REPO, "results", "all_runs.csv")
ROOT = os.path.join(PARENT, "probes_bf8")
SCRIPT = os.path.join(REPO, "bin", "c71_floor_budget.sh")

N_RECORDS = 8000                   # 80 ep x 100 records/epoch at PROBE=5
NBETA = C71.NBETA                  # w 11,173,962 / node 14,420 / lay 62
FLOORS = {"f60": (-60.0, 2.0), "f90": (-90.0, 2.0)}
RUNGS = ["w", "node", "lay"]
BARS = {"w": 0.02, "node": 0.06, "lay": 0.11}   # br6 n=4 40ep, same bars as bd7's D0.5
D1_PRIOR = 0.1509                  # br6 weightwise at 40 ep -- 55.6's expectation
BR6_SPAN = 17.52                   # br6 40ep, the cell F2 says bf8 must sit ABOVE


def arm_dirs(rung, floor):
    return sorted(glob.glob(os.path.join(ROOT, f"probe_{rung}_{floor}_s*")))


def csv_rows():
    with open(CSV) as f:
        return [r for r in csv.DictReader(f) if r["superseded"] != "1"]


# ------------------------------------------------------------------ F0

def f0():
    out = []
    for rung in RUNGS:
        for fl in FLOORS:
            for d in arm_dirs(rung, fl):
                recs = BF.records(d)
                nb = None
                npy = os.path.join(d, "neg_counts.npy")
                if os.path.exists(npy):
                    import numpy as np
                    nb = int(np.load(npy, mmap_mode="r").shape[0])
                lo = [r.get("beta_true_min") for r in recs
                      if r.get("beta_true_min") is not None]
                hi = [r.get("beta_true_max") for r in recs
                      if r.get("beta_true_max") is not None]
                moved = bool(lo and hi and (max(hi) - min(lo)) > 1e-9)
                out.append(dict(arm=os.path.basename(d), rung=rung, floor=fl,
                                n=len(recs), n_ok=(len(recs) == N_RECORDS),
                                nbeta=nb, nbeta_ok=(nb == NBETA[rung]), moved=moved))
    return out


# ------------------------------------------------------------------ F0.3

def f03():
    """BOX-FREE, per seed, per FLOOR, at BOTH guards. STANDING RULE (18)."""
    out = []
    for rung in RUNGS:
        for fl, (lo, hi) in FLOORS.items():
            for d in arm_dirs(rung, fl):
                o = BF.occupancy(d, lo, hi)
                worst = max(o["rec_lo"], o["rec_hi"])
                out.append(dict(arm=os.path.basename(d), rung=rung, floor=fl,
                                rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                                q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                                first_lo=o["first_lo"], first_hi=o["first_hi"],
                                boxfree=worst < 0.05))
    return out


# ------------------------------------------------------------------ F0.4

def f04():
    out = []
    for r in csv_rows():
        if not r["run"].startswith("bf8-"):
            continue
        p5 = float(r["plateau5"]) if r["plateau5"] else float("nan")
        out.append(dict(run=r["run"], gran=r["granularity"], clip=r["beta_clip"],
                        seed=r["seed"], plateau5=p5, collapsed=r["collapsed"],
                        passes=(p5 > 40.0 and
                                r["collapsed"] not in ("1", "True", "true"))))
    return out


# ------------------------------------------------------------------ N_eff

def neff_cells(window=None):
    """{(floor, rung): cell_stats} from c55's sweep -- the published instrument."""
    window = C55.STEADY[1] if window is None else window
    cells = C55.sweep(ROOT, window=window)
    out = {}
    for (fam, rung), rows in cells.items():
        if rung not in RUNGS:
            continue
        st = C55.cell_stats([r["neff_m"] for r in rows])
        st["nbf"] = sum(1 for r in rows if r["boxfree"])
        st["n_arms"] = len(rows)
        st["span"] = float(sum(r["span"] for r in rows) / len(rows))
        st["ep"] = rows[0]["epochs"]
        out[(fam, rung)] = st
    return out


# ------------------------------------------------------------------ F0.5

def f05(cells):
    """FLOOR-INVARIANCE on rungs box-free at BOTH floors."""
    out = []
    for rung in RUNGS:
        a, b = cells.get(("f60", rung)), cells.get(("f90", rung))
        if a is None or b is None:
            out.append(dict(rung=rung, status="MISSING"))
            continue
        both_bf = (a["nbf"] == a["n_arms"] and b["nbf"] == b["n_arms"])
        d = abs(a["mean"] - b["mean"])
        out.append(dict(rung=rung, f60=a["mean"], f90=b["mean"], delta=d,
                        bar=BARS[rung], both_boxfree=both_bf,
                        status=("NOT SCORED (not box-free at both floors)" if not both_bf
                                else ("PASS" if d <= BARS[rung] else "FAIL"))))
    return out


# ------------------------------------------------------------------ F2

def f2(cells, fam):
    byrung = {r: cells[(fam, r)] for r in RUNGS if (fam, r) in cells}
    if len(byrung) < len(RUNGS):
        return dict(status="MISSING", fam=fam)
    v = C55.argmin_in_se(byrung)
    v["fam"] = fam
    v["span"] = float(sum(c["span"] for c in byrung.values()) / len(byrung))
    return v


# ------------------------------------------------------------------ selftest

def selftest():
    p = n = 0

    def chk(name, cond):
        nonlocal p, n
        n += 1
        if cond:
            p += 1
        else:
            print("  FAIL:", name)

    # --- the box was REGISTERED before scoring, and matches the SCRIPT, not the data
    chk("bf8:f60 registered", C55.BOXES.get("bf8:f60", (None,))[0] == -60.0)
    chk("bf8:f90 registered", C55.BOXES.get("bf8:f90", (None,))[0] == -90.0)
    chk("both floors share HI=+2.0",
        C55.BOXES["bf8:f60"][1] == 2.0 and C55.BOXES["bf8:f90"][1] == 2.0)
    chk("box_for resolves the arm-specific key",
        C55.box_for("bf8", "f60") == (-60.0, 2.0) and
        C55.box_for("bf8", "f90") == (-90.0, 2.0))

    # --- the registered floors are the ones the SCRIPT emits (no drift)
    src = open(SCRIPT).read()
    chk("script emits LO in {-60,-90}", "for LO in -60 -90" in src)
    chk("script fixes HI=2.0", "\nHI=2.0" in src)
    chk("script requests 80 epochs", "\nEPOCHS=80" in src)
    chk("script tags floors f60/f90", "echo f60" in src and "echo f90" in src)

    # --- imported instruments are the published ones
    chk("NBETA imported from c71", NBETA is C71.NBETA)
    chk("n_beta weightwise is the campaign constant", NBETA["w"] == 11173962)
    chk("100 records per epoch", C55.RECORDS_PER_EPOCH == 100)
    chk("8000 records == 80 epochs", N_RECORDS / C55.RECORDS_PER_EPOCH == 80)
    chk("bars match bd7's registered D0.5 bars", BARS == C71.BARS_BD7)
    chk("D1 prior matches c71's", abs(D1_PRIOR - C71.D1_PRIOR) < 1e-12)

    # --- F0.5 branch logic on synthetic cells (no data dependence)
    def cell(mean, sd, na, nbf=None):
        return dict(mean=mean, sd=sd, n=na, se=sd / math.sqrt(na),
                    mrd=C55.mrd(sd, na), nbf=(na if nbf is None else nbf),
                    n_arms=na, span=1.0, ep=80)
    syn = {("f60", "w"): cell(0.10, 0.001, 2), ("f90", "w"): cell(0.11, 0.001, 2),
           ("f60", "node"): cell(0.20, 0.001, 2), ("f90", "node"): cell(0.40, 0.001, 2),
           ("f60", "lay"): cell(0.50, 0.001, 2, nbf=0), ("f90", "lay"): cell(0.50, 0.001, 2)}
    r = {x["rung"]: x for x in f05(syn)}
    chk("F0.5 PASSes inside the w bar (0.01 <= 0.02)", r["w"]["status"] == "PASS")
    chk("F0.5 FAILs outside the node bar (0.20 > 0.06)", r["node"]["status"] == "FAIL")
    chk("F0.5 refuses to score a non-box-free rung",
        r["lay"]["status"].startswith("NOT SCORED"))
    chk("F0.5 reports MISSING when a floor is absent",
        f05({("f60", "w"): cell(0.1, 0.001, 2)})[0]["status"] == "MISSING")

    # --- F2 delegates to the published argmin pricing, and its verdicts are the
    #     ones cycle 71 already argued about (regression against c55's own cases)
    cl5 = {"lay": cell(0.7262, 0.0227, 3), "node": cell(0.4056, 0.0492, 3),
           "w": cell(0.5064, 0.0377, 3)}
    v = C55.argmin_in_se(cl5)
    chk("argmin pricing still reproduces cl5/cU node DECIDED",
        v["argmin"] == "node" and v["status"] == "DECIDED")
    bound = {"lay": cell(0.43, 0.005, 2), "node": cell(0.12, 0.026, 2, nbf=0),
             "w": cell(0.30, 0.010, 2)}
    chk("a bound rung ANYWHERE makes the argmin UNINTERPRETABLE",
        C55.argmin_in_se(bound)["status"] == "UNINTERPRETABLE")
    chk("F2 reports MISSING on an incomplete rung set",
        f2({("f60", "w"): cell(0.1, 0.001, 2)}, "f60")["status"] == "MISSING")

    # --- corpus presence: 12 arms on disk, 12 rows in the CSV
    chk("12 probe arms on disk", sum(len(arm_dirs(g, f))
                                     for g in RUNGS for f in FLOORS) == 12)
    chk("12 bf8 rows in the CSV", len(f04()) == 12)

    print("selftest: %d/%d passed" % (p, n))
    return p == n


# ------------------------------------------------------------------ score

def score():
    print("=" * 78)
    print("bf8 -- SCORED AGAINST bin/c71_floor_budget.sh, IN THE REGISTERED ORDER")
    print("=" * 78)

    # -------- F0
    rows = f0()
    ok = sum(1 for r in rows if r["n_ok"] and r["nbeta_ok"] and r["moved"])
    print("\n[F0] VALIDITY -- n_records == %d, n_beta byte-match, beta moved" % N_RECORDS)
    print("     pass %d/%d" % (ok, len(rows)))
    for r in rows:
        if not (r["n_ok"] and r["nbeta_ok"] and r["moved"]):
            print("     VOID %-22s n=%s nbeta=%s moved=%s"
                  % (r["arm"], r["n"], r["nbeta"], r["moved"]))
    if ok != len(rows):
        print("     >>> a n_beta mismatch means the NETWORK changed: the batch is VOID.")

    # -------- F0.3
    print("\n[F0.3] THE BOX-FREE GATE -- per seed, per FLOOR, AT BOTH GUARDS (RULE 18)")
    print("     bd7 comparison: at LO=-30, 10/12 arms bound (37-42%% of records, 100%% of Q4)")
    print("     %-22s %8s %8s %8s %8s %10s  %s"
          % ("arm", "rec_lo%", "rec_hi%", "q4_lo%", "q4_hi%", "first_lo", "verdict"))
    bf = f03()
    for r in sorted(bf, key=lambda x: (x["floor"], x["rung"], x["arm"])):
        fl = r["first_lo"]
        fl_s = ("%d (ep %.1f)" % (fl, fl / 100.0)) if fl is not None else "-"
        print("     %-22s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %10s  %s"
              % (r["arm"], 100 * r["rec_lo"], 100 * r["rec_hi"],
                 100 * r["q4_lo"], 100 * r["q4_hi"], fl_s,
                 "box-free" if r["boxfree"] else "BOUND"))
    nbf = sum(1 for r in bf if r["boxfree"])
    print("     => box-free %d/%d" % (nbf, len(bf)))
    if nbf < len(bf):
        print("     >>> AN ARM BINDS AT LO EVEN AT THIS FLOOR.  Per the registration,")
        print("     >>> `first_lo` and its epoch IS the result: beta is not confinable at")
        print("     >>> this budget, and the next batch is an `ms` ladder, not a third floor.")

    # -------- F0.4
    print("\n[F0.4] THE TRAINS-AT-ALL GATE -- plateau5 > 40.0, collapsed false")
    acc = f04()
    for r in sorted(acc, key=lambda x: x["run"]):
        print("     %-22s %-16s clip %-10s plateau5 %7.3f  %s"
              % (r["run"], r["gran"], r["clip"], r["plateau5"],
                 "OK" if r["passes"] else "FAIL"))
    print("     => pass %d/%d" % (sum(1 for r in acc if r["passes"]), len(acc)))

    # -------- N_eff table
    cells = neff_cells()
    print("\n[N_eff/m] VARIANCE instrument, STEADY half -- neff_instrument, unchanged")
    print("     %-6s %-6s %5s %10s %10s %8s %7s %6s"
          % ("floor", "rung", "m", "N_eff/m", "sd", "box-free", "span", "ep"))
    for fl in sorted(FLOORS):
        for rung in RUNGS:
            c = cells.get((fl, rung))
            if c is None:
                print("     %-6s %-6s  MISSING" % (fl, rung))
                continue
            print("     %-6s %-6s %5s %10.5f %10.5f %6d/%-2d %7.2f %6.0f"
                  % (fl, rung, NBETA[rung], c["mean"], c["sd"], c["nbf"],
                     c["n_arms"], c["span"], c["ep"]))

    # -------- F0.5
    print("\n[F0.5] THE FLOOR-INVARIANCE GATE -- bars w %.2f / node %.2f / lay %.2f (br6, n=4)"
          % (BARS["w"], BARS["node"], BARS["lay"]))
    inv = f05(cells)
    for r in inv:
        if r["status"] == "MISSING":
            print("     %-6s MISSING" % r["rung"])
            continue
        print("     %-6s f60 %8.5f   f90 %8.5f   |delta| %8.5f   bar %.2f   %s"
              % (r["rung"], r["f60"], r["f90"], r["delta"], r["bar"], r["status"]))
    scored = [r for r in inv if r["status"] in ("PASS", "FAIL")]
    poolable = bool(scored) and all(r["status"] == "PASS" for r in scored)
    print("     => %s" % ("THE TWO FLOORS POOL; the 80-epoch point is a budget point."
                          if poolable else
                          "FLOOR-DEPENDENCE -- report it, do NOT patch it."))

    # -------- F1
    print("\n[F1] THE 80-EPOCH POINT -- what bd7's D1 was meant to be")
    print("     55.6's written-down expectation: weightwise below %.4f and still falling."
          % D1_PRIOR)
    for fl in sorted(FLOORS):
        c = cells.get((fl, "w"))
        if c is None:
            continue
        state = "UNRESOLVED" if c["nbf"] < c["n_arms"] else "%.5f" % c["mean"]
        print("     weightwise @ %s : %s   (box-free %d/%d)"
              % (fl, state, c["nbf"], c["n_arms"]))
    ws = [cells[(fl, "w")] for fl in sorted(FLOORS)
          if (fl, "w") in cells and cells[(fl, "w")]["nbf"] == cells[(fl, "w")]["n_arms"]]
    if ws:
        pooled = sum(c["mean"] for c in ws) / len(ws)
        print("     pooled weightwise N_eff/m @ 80 ep = %.5f" % pooled)
        if pooled < D1_PRIOR:
            print("     BRANCH (a) KEEPS FALLING -- below 55.6's %.4f." % D1_PRIOR)
            print("     THIS IS A REPLICATION of 55.6's curve, NEVER a discovery.")
        elif abs(pooled - 0.15) < 0.03:
            print("     BRANCH (b) FLOOR ~0.15 -- 'at long budgets ~15%' is the sentence.")
        else:
            print("     BRANCH (c) REBOUNDS -- the 20->40 fall is not a trend.")
    else:
        print("     NOT SCORED -- no weightwise arm passes the box-free gate.")

    # -------- F2
    print("\n[F2] THE SPAN BAND, OUT OF SAMPLE -- REGISTERED PREDICTION: argmin = w, gap/SE >= 2")
    print("     br6 (same ms, HALF the budget) sits at span %.2f and reads `w`." % BR6_SPAN)
    for fl in sorted(FLOORS):
        v = f2(cells, fl)
        if v.get("status") == "MISSING":
            print("     %-4s MISSING a rung" % fl)
            continue
        if v["status"] == "UNINTERPRETABLE":
            why = (("bound: " + ",".join(v["blockers"])) if v["blockers"]
                   else ("missing: " + ",".join(v["missing"])))
            print("     %-4s span %6.2f  UNINTERPRETABLE (%s)" % (fl, v["span"], why))
            continue
        print("     %-4s span %6.2f  argmin %-5s 2nd %-5s gap %.4f  SE %.4f  gap/SE %.2f  %s"
              % (fl, v["span"], v["argmin"], v["runner_up"], v["gap"], v["se"],
                 v["gap_se"], v["status"]))
        if v["status"] == "DECIDED":
            if v["argmin"] == "w":
                print("          CONFIRMS the band out of sample at a budget no fit cell had.")
            elif v["argmin"] == "node":
                print("          REFUTES the band as a function of span alone.")
                print("          101.9's 'one scalar' claim is WITHDRAWN.")
    print()


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if a[0] == "--score":
        score()
    else:
        print(__doc__)
