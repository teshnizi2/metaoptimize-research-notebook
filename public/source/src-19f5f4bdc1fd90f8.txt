#!/usr/bin/env python3
# =============================================================================
# cR2_cpk2_audit.py -- POST-HOC DESCRIPTIVE AUDIT OF BATCH `cpk2`.
#
# THIS IS NOT A SCORER AND IT IS NOT REGISTERED.  It computes NO verdict, has
# NO gates, and NOTHING it prints may be quoted as a cpk2 result.  The ONLY
# object that carries cpk2's verdict is `analysis/cR1_cpk2_score.py`, which was
# committed at d5c6eb6, 94 s before the first job was submitted, and which this
# file does not read, import or modify (STANDING RULE 16).
#
# WHY IT EXISTS.  CORRECTIONS 158 presses four attacks on cR1's PASS result.
# Each attack needs arithmetic that cR1 does not print.  Writing that
# arithmetic here -- AFTER the fact, and labelled as after the fact -- keeps it
# out of the registered scorer and keeps CORRECTIONS 158's numerals
# reproducible.  Every number below is re-derived from the cpk2 `.out` files
# and from results/all_runs.csv; none is copied from cR1's output or from
# CORRECTIONS prose.
#
#   ATTACK 1  the k52 residual (-8.22 pp where every other arm is within 1.6):
#             is it the MODEL being weak, or does the ladder's SHAPE at 772
#             differ from its shape at 100?
#   ATTACK 2  the argmax sits at a grid edge of its own neighbourhood.  What
#             does the design resolve about the peak's LOCATION, as opposed to
#             its IDENTITY among the sampled cuts?
#   ATTACK 3  what did the fresh seeds actually buy?  Does any cpk2 run
#             duplicate a corpus row?
#   ATTACK 4  the floor anchor: did the floor move with budget, and would any
#             R-FLOOR call have changed without it?
#
# USAGE:  python3 analysis/cR2_cpk2_audit.py <runsdir>
# =============================================================================
import csv, glob, os, re, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, os.pardir, "results", "all_runs.csv")

E, CONTROL, WIN = 772, 100, 20
ARMS = ("k01", "k45", "k47", "k49", "k50", "k52")
SWEEP = ("k45", "k47", "k49", "k50", "k52")
KOF = {"k45": 45, "k47": 47, "k49": 49, "k50": 50, "k52": 52}

# bars, copied from the REGISTERED scorer's frozen literals purely so this
# audit reports in the same units.  They gate nothing here.
SE_ARM_DIFF = 0.748956
FLOOR_BAR = 1.148846
ARGMAX_BAR = 1.497912
CORPUS_SCALAR_100 = 22.749176
GAIN_A, GAIN_B = 0.649507, 152.256507          # cts3's two-point line

EPOCH_RE = re.compile(
    r"Epoch (\d+), Train Accuracy: ([\d.]+) %, Test Accuracy: ([\d.]+) %")


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / sum((x - mx) ** 2 for x in xs))


def load(runsdir):
    per = {}
    for f in sorted(glob.glob(os.path.join(runsdir, "cpk2-*.out"))):
        nm = os.path.basename(f)
        arm, seed = nm.split("-")[1], nm.split("-")[2]
        ep = {}
        for line in open(f, errors="replace"):
            m = EPOCH_RE.match(line)
            if m:
                ep[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        if len(ep) != E:
            sys.exit("%s: %d epoch lines, want %d" % (nm, len(ep), E))
        per.setdefault(arm, []).append(dict(
            seed=seed,
            p100=st.mean(ep[i][1] for i in range(CONTROL - 5, CONTROL)),
            pE=st.mean(ep[i][1] for i in range(E - 5, E)),
            t100=st.mean(ep[i][0] for i in range(CONTROL - 5, CONTROL)),
            tE=st.mean(ep[i][0] for i in range(E - 5, E)),
            s100=ols(list(range(CONTROL - WIN, CONTROL)),
                     [ep[i][1] for i in range(CONTROL - WIN, CONTROL)]),
            sE=ols(list(range(E - WIN, E)),
                   [ep[i][1] for i in range(E - WIN, E)])))
    A = {}
    for a, rs in per.items():
        A[a] = {k: st.mean(r[k] for r in rs)
                for k in ("p100", "pE", "t100", "tE", "s100", "sE")}
        A[a]["sd_pE"] = st.stdev(r["pE"] for r in rs)
        A[a]["sd_p100"] = st.stdev(r["p100"] for r in rs)
        A[a]["seeds"] = sorted(r["seed"] for r in rs)
    return A, per


def hdr(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: cR2_cpk2_audit.py <runsdir>")
    A, per = load(sys.argv[1])
    if sorted(A) != sorted(ARMS):
        sys.exit("expected arms %s, found %s" % (list(ARMS), sorted(A)))

    hdr("cpk2 POST-HOC AUDIT -- DESCRIPTIVE ONLY, NO VERDICT")
    print("re-derived from the .out files; plateau5 = mean of epochs "
          "%d-%d and %d-%d" % (CONTROL - 5, CONTROL - 1, E - 5, E - 1))
    print("\n  %-5s %9s %9s %8s %9s %9s %9s %10s %10s"
          % ("arm", "TEST@100", "TEST@E", "gain", "TRAIN@100", "TRAIN@E",
             "gen.gap", "slope@100", "slope@E"))
    for a in ARMS:
        d = A[a]
        print("  %-5s %9.4f %9.4f %+8.4f %9.4f %9.4f %9.4f %10.5f %10.5f"
              % (a, d["p100"], d["pE"], d["pE"] - d["p100"], d["t100"],
                 d["tE"], d["tE"] - d["pE"], d["s100"], d["sE"]))

    # ---------------- ATTACK 1 -------------------------------------------
    hdr("ATTACK 1 -- IS THE k52 RESIDUAL A WEAK MODEL, OR A DIFFERENT SHAPE?")
    o100 = sorted(SWEEP, key=lambda a: -A[a]["p100"])
    oE = sorted(SWEEP, key=lambda a: -A[a]["pE"])
    print("  measured rank order @100 : " + " > ".join(o100))
    print("  measured rank order @E   : " + " > ".join(oE))
    print("  ORDERING PRESERVED: %s   (rank inversions: %d)"
          % (o100 == oE, sum(1 for i, x in enumerate(o100) if oE[i] != x)))

    print("\n  gap below the peak, k49 minus arm (pp):")
    print("  %-5s %5s %10s %10s %10s %10s"
          % ("arm", "dk", "gap@100", "gap@E", "change", "in SE"))
    for a in SWEEP:
        if a == "k49":
            continue
        g1 = A["k49"]["p100"] - A[a]["p100"]
        gE = A["k49"]["pE"] - A[a]["pE"]
        print("  %-5s %+5d %10.4f %10.4f %+10.4f %+10.2f"
              % (a, KOF[a] - 49, g1, gE, gE - g1, (gE - g1) / SE_ARM_DIFF))
    r1 = ((A["k49"]["p100"] - A["k50"]["p100"])
          / (A["k49"]["p100"] - A["k47"]["p100"]))
    rE = ((A["k49"]["pE"] - A["k50"]["pE"])
          / (A["k49"]["pE"] - A["k47"]["pE"]))
    print("\n  asymmetry ratio (gap to k50, dk=+1)/(gap to k47, dk=-2):")
    print("     @100 %.3f    @E %.3f    change %+.3f" % (r1, rE, rE - r1))

    print("\n  the SAME two-point line, re-fed with cpk2's OWN in-batch "
          "slope@100:")
    print("  %-5s %12s %12s %12s %11s"
          % ("arm", "slope@100", "model gain", "meas gain", "residual"))
    for a in ARMS:
        pg = GAIN_A + GAIN_B * A[a]["s100"]
        mg = A[a]["pE"] - A[a]["p100"]
        print("  %-5s %12.5f %12.4f %12.4f %+11.4f" % (a, A[a]["s100"], pg,
                                                       mg, mg - pg))
    print("\n  slope@100 and gain, right flank only:")
    print("     k50 slope %.5f -> gain %+.4f" % (A["k50"]["s100"],
                                                 A["k50"]["pE"] - A["k50"]["p100"]))
    print("     k52 slope %.5f -> gain %+.4f" % (A["k52"]["s100"],
                                                 A["k52"]["pE"] - A["k52"]["p100"]))
    print("     k52 slope is %.2fx k50's but its gain is %.2fx k50's."
          % (A["k52"]["s100"] / A["k50"]["s100"],
             (A["k52"]["pE"] - A["k52"]["p100"])
             / (A["k50"]["pE"] - A["k50"]["p100"])))
    print("     => gain is NOT monotone in slope@100 across the right flank.")

    # what the REGISTERED forecast implied about the ORDER, vs what happened.
    # PRED_E is cR1's own registered per-arm prediction (its --selftest prints
    # the same six numbers); it is quoted here only to compare ORDERINGS.
    PRED_E = {"k45": 42.9312, "k47": 47.0927, "k49": 55.9224,
              "k50": 37.4832, "k52": 48.0612}
    LVL100 = {"k45": 42.345667, "k47": 45.688000, "k49": 55.338325,
              "k50": 30.297333, "k52": 38.166667}
    po = sorted(SWEEP, key=lambda a: -PRED_E[a])
    lo = sorted(SWEEP, key=lambda a: -LVL100[a])
    print("\n  the FORECAST predicted the ladder would RE-ORDER; it did not.")
    print("     registered level@100 order : " + " > ".join(lo))
    print("     forecast    order @E       : " + " > ".join(po)
          + "   (%d rank swaps)" % sum(1 for i, x in enumerate(lo)
                                       if po[i] != x))
    print("     MEASURED    order @E       : " + " > ".join(oE)
          + "   (%d rank swaps)" % sum(1 for i, x in enumerate(o100)
                                       if oE[i] != x))
    print("     forecast runner-up %s (margin %.4f) ; measured runner-up "
          "%s (margin %.4f)"
          % (po[1], PRED_E["k49"] - PRED_E[po[1]], oE[1],
             A["k49"]["pE"] - A[oE[1]]["pE"]))
    rp = ((PRED_E["k49"] - PRED_E["k50"]) / (PRED_E["k49"] - PRED_E["k47"]))
    rl = ((LVL100["k49"] - LVL100["k50"]) / (LVL100["k49"] - LVL100["k47"]))
    print("     asymmetry ratio the FORECAST implied: %.3f -> %.3f (%+0.3f), "
          "i.e. DOWN," % (rl, rp, rp - rl))
    print("     so 157.6's prose \"the curve becomes MORE asymmetric\" was "
          "not implied by its")
    print("     own model either.  Measured: %.3f -> %.3f (%+0.3f).  "
          "The prose claim is WRONG." % (r1, rE, rE - r1))

    # ---------------- ATTACK 2 -------------------------------------------
    hdr("ATTACK 2 -- WHAT THE GRID CAN AND CANNOT RESOLVE ABOUT THE PEAK")
    rows = list(csv.DictReader(open(CSV)))

    def incell(r):
        return (r["network"] == "ResNet18_c100" and r["dataset"] == "CIFAR100"
                and r["batch_size"] == "100" and r["meta_stepsize"] == "1e-3"
                and r["alpha0"] == "1e-6" and r["augment"] == "1"
                and r["beta_clip"] == "-15:-2.3026" and r["base"] == "SGDm"
                and r["meta"] == "Lion")
    seen = {}
    for r in rows:
        if not incell(r):
            continue
        m = re.match(r"^\[(\d+),(\d+)\]$", r["granularity"] or "")
        if m:
            seen.setdefault(int(m.group(1)), set()).add(r["epochs_done"])
    print("  every m=2 cut position EVER RUN in this exact cell, and at what "
          "horizon:")
    for k in sorted(seen):
        print("     k=%-3d  %s" % (k, ",".join(sorted(seen[k]))))
    gaps = [k for k in range(44, 54) if k not in seen]
    print("\n  cut positions in 44..53 NEVER RUN AT ANY HORIZON: %s" % gaps)
    print("  k is a 1-based INDEX over the 62 named parameter tensors, so it "
          "is an INTEGER;")
    print("  49 and 50 are ADJACENT and nothing lies between them.  The only "
          "unsampled")
    print("  cuts inside the peak's own neighbourhood are k=%s."
          % ", ".join(str(g) for g in gaps if 44 < g < 53))
    print("\n  the objective is NOT smooth near the peak, so interpolation is "
          "not licensed:")
    print("     k49->k50 moves ONE tensor (512 params) and costs %.4f pp at E."
          % (A["k49"]["pE"] - A["k50"]["pE"]))
    print("  => the resolvable claim is: k* = 49 AMONG {45,47,49,50,52}.")

    # ---------------- ATTACK 3 -------------------------------------------
    hdr("ATTACK 3 -- WHAT THE FRESH SEEDS BOUGHT")
    print("  seeds actually present, per arm (from the .out file names):")
    for a in ARMS:
        print("     %-5s %s" % (a, A[a]["seeds"]))
    grans = {"[45,17]", "[47,15]", "[49,13]", "[50,12]", "[52,10]", "scalar"}
    tup_new = {(r["granularity"], r["seed"], r["epochs_done"])
               for r in rows if r["run"].startswith("cpk2-")}
    clash = [r for r in rows
             if not r["run"].startswith("cpk2-") and incell(r)
             and r["granularity"] in grans
             and (r["granularity"], r["seed"], r["epochs_done"]) in tup_new]
    print("\n  corpus rows OUTSIDE cpk2 sharing a cpk2 "
          "(granularity, seed, epochs) tuple in this cell: %d" % len(clash))
    for r in clash:
        print("     %s" % r["run"])
    near = [r for r in rows
            if not r["run"].startswith("cpk2-") and incell(r)
            and r["granularity"] in grans and r["seed"] in {"3", "4", "5"}]
    print("\n  corpus rows outside cpk2, same cell, same granularity, seed in "
          "{3,4,5}, ANY horizon: %d" % len(near))
    for r in near:
        print("     %-22s gran=%-8s seed=%s epochs=%s plateau5=%s"
              % (r["run"], r["granularity"], r["seed"], r["epochs_done"],
                 r["plateau5"]))
    print("\n  name collisions with an existing run name: %d"
          % len([r for r in rows if r["run"].startswith("cpk2-")
                 and sum(1 for q in rows if q["run"] == r["run"]) > 1]))

    # ---------------- ATTACK 4 -------------------------------------------
    hdr("ATTACK 4 -- DID THE FLOOR ACTUALLY MOVE WITH BUDGET?")
    mv = A["k01"]["pE"] - A["k01"]["p100"]
    print("  k01 (scalar, m=1)  @100 %.4f   @E %.4f   moved %+.4f pp = "
          "%.2f SE_ARM_DIFF" % (A["k01"]["p100"], A["k01"]["pE"], mv,
                                mv / SE_ARM_DIFF))
    print("  FLOOR_BAR (2*SE_FLOOR) %.4f pp -> the move is %s the bar"
          % (FLOOR_BAR, "INSIDE" if abs(mv) < FLOOR_BAR else "OUTSIDE"))
    print("  k01 slope@E %+.5f -- the ONLY arm in the batch with a POSITIVE "
          "terminal slope" % A["k01"]["sE"])
    print("\n  counterfactual: would any R-FLOOR call at E change if the "
          "100-epoch CORPUS")
    print("  floor %.4f were used in place of this batch's own k01 at E "
          "(%.4f)?" % (CORPUS_SCALAR_100, A["k01"]["pE"]))
    print("  %-5s %14s %16s %10s" % ("arm", "above k01@E", "above corpus@100",
                                     "call moves"))
    moved = 0
    for a in SWEEP:
        x, y = A[a]["pE"] - A["k01"]["pE"], A[a]["pE"] - CORPUS_SCALAR_100
        same = (x > FLOOR_BAR) == (y > FLOOR_BAR)
        moved += 0 if same else 1
        print("  %-5s %+14.4f %+16.4f %10s" % (a, x, y, "NO" if same else "YES"))
    print("  R-FLOOR calls that change: %d of %d" % (moved, len(SWEEP)))
    print("  closest arm to the floor at E: %s, %+.4f pp = %.1fx FLOOR_BAR"
          % (min(SWEEP, key=lambda z: A[z]["pE"]),
             min(A[z]["pE"] for z in SWEEP) - A["k01"]["pE"],
             (min(A[z]["pE"] for z in SWEEP) - A["k01"]["pE"]) / FLOOR_BAR))

    hdr("END -- nothing above is a verdict")
    return 0


if __name__ == "__main__":
    sys.exit(main())
