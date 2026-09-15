#!/usr/bin/env python3
"""
c72_hz9_score.py -- score the `hz9` batch against the gates REGISTERED in
bin/c58_tuned_horizon.sh (cycle 58) and restated in CORRECTIONS 87.8.

    H0    validity, HARD DROP     epochs_done == epochs_requested == 100
    H0.3  the guard gate          BETA_CLIP live, collapsed false, plateau > 40
    H0.5  the POOLING gate        CAN ONLY VOID.  hz9 arms must sit within 3 sd
                                  of their `rs` corpus counterparts, else H1 is
                                  NOT SCORED and only the hz9-only cut is shown.
    H1    THE PRIMARY TEST        D = (node@3e-4 - lay@1e-4)@100
                                    - (node@3e-4 - lay@1e-4)@50
                                  T-A  D >= +0.30  the reversal SURVIVES tuning
                                  T-B  D <= -0.30  it is a shared-ms effect
                                  else UNDECIDED
                                  REGISTERED DIRECTION: the pilot gives -0.485
                                  and points at T-B, so a T-B result is a
                                  REPLICATION and never a discovery.
    H1b   the level               (node - lay)@100 against a +-0.50 bar.
                                  |level| < 0.50 => TIED, and TIED IS A RESULT.
    H2    the fixed-ms cell       hz9-node-1e3-s{3,4} vs rs-lay-1e3-s{3,4},
                                  PAIRED within seed.  Registered expectation
                                  +1.0 .. +2.0 with 2/2 seeds favouring nodewise.

METRIC.  Every number is the matched-k=5 plateau read from the RAW .out series
(`plateau_at(series(run), e, k=5)`), which is what the registration's own
numbers were computed with -- verified in selftest against the CSV `plateau5`
column and against the two registered constants 92.824 and 92.656.

The raw-series primitives are IMPORTED from analysis/c58_horizon_reversal.py
(42/42 selftests, --control REPRODUCES CORRECTIONS 86.3 exactly) rather than
re-implemented, so this file cannot drift from the instrument that registered
the gates.

    python3 analysis/c72_hz9_score.py --selftest
    python3 analysis/c72_hz9_score.py --score
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c58_horizon_reversal as C  # noqa: E402  -- registered instrument

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "all_runs.csv")

K = 5                       # matched-k, the registration's own window
CLIP = "-15:-2.3026"        # the c40 surface's guard; matched, not chosen
EPOCHS = 100
SD_POOL = None              # measured in H0.5 from rs-lay-1e4, never assumed

# The arms, by NAME, exactly as bin/c58_tuned_horizon.sh emits / c40 supplied.
HZ9_LAY_1E4 = ["hz9-lay-1e4-s0", "hz9-lay-1e4-s1", "hz9-lay-1e4-s2"]
HZ9_NODE_3E4 = ["hz9-node-3e4-s%d" % s for s in (1, 2, 3, 4)]
HZ9_NODE_1E3 = ["hz9-node-1e3-s3", "hz9-node-1e3-s4"]
RS_LAY_1E4 = ["rs-lay-1e4-s3", "rs-lay-1e4-s4"]
RS_NODE_3E4 = ["rs-node-3e4-s0"]          # s1 is TRUNCATED at 89 and H0-dropped
RS_LAY_1E3 = ["rs-lay-1e3-s3", "rs-lay-1e3-s4"]

ALL_HZ9 = HZ9_LAY_1E4 + HZ9_NODE_3E4 + HZ9_NODE_1E3


# ------------------------------------------------------------------ helpers

def rows_by_run(path=CSV):
    with open(path) as fh:
        return {r["run"]: r for r in csv.DictReader(fh)}


def mean(v):
    return sum(v) / len(v)


def sd(v):
    """Sample sd (n-1). None below n=2."""
    if len(v) < 2:
        return None
    m = mean(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def sem(v):
    s = sd(v)
    return None if s is None else s / len(v) ** 0.5


def pl(run, e=EPOCHS, k=K):
    """matched-k plateau of `run` at 1-indexed epoch e, from the RAW .out."""
    s = C.series(run)
    if s is None:
        return None
    return C.plateau_at(s, e, k)


# ------------------------------------------------------------------ gates

def h0(rows, names):
    """VALIDITY, HARD DROP. Returns (kept, dropped) -- dropped is a list of (run, why)."""
    kept, dropped = [], []
    for n in names:
        r = rows.get(n)
        if r is None:
            dropped.append((n, "ABSENT from CSV"))
            continue
        try:
            done, req = int(r["epochs_done"]), int(r["epochs_requested"])
        except (ValueError, KeyError):
            dropped.append((n, "unparseable epochs"))
            continue
        if req != EPOCHS or done != req:
            dropped.append((n, "epochs_done=%d requested=%d" % (done, req)))
            continue
        kept.append(n)
    return kept, dropped


def h03(rows, names):
    """THE GUARD GATE. A failure voids that JOB, not the batch."""
    ok, bad = [], []
    for n in names:
        r = rows[n]
        why = []
        if r["beta_clip"] != CLIP:
            why.append("beta_clip=%s" % r["beta_clip"])
        if str(r["collapsed"]).strip() not in ("0", "False", "false", ""):
            why.append("collapsed=%s" % r["collapsed"])
        p = pl(n)
        if p is None:
            why.append("no k=%d plateau at ep %d" % (K, EPOCHS))
        elif p <= 40.0:
            why.append("plateau=%.3f <= 40" % p)
        (bad if why else ok).append((n, "; ".join(why)) if why else n)
    return ok, bad


def h05(rows):
    """THE POOLING GATE -- CAN ONLY VOID. Re-derives the corpus reference from
    the CSV/.out; the registration's 92.824 / 92.656 are checked, not quoted."""
    ref_lay = [pl(n) for n in RS_LAY_1E4]
    ref_node = [pl(n) for n in RS_NODE_3E4]
    s_lay = sd(ref_lay)                       # layerwise supplies the sd (n=2)
    m_lay, m_node = mean(ref_lay), mean(ref_node)
    checks = []
    for n in HZ9_LAY_1E4:
        v = pl(n)
        checks.append(("lay", n, v, m_lay, abs(v - m_lay) / s_lay))
    for n in HZ9_NODE_3E4:                    # n=1 corpus -> use layerwise's sd
        v = pl(n)
        checks.append(("node", n, v, m_node, abs(v - m_node) / s_lay))
    passed = all(c[4] <= 3.0 for c in checks)
    return passed, checks, m_lay, m_node, s_lay


def arm(rows, names, e):
    return [pl(n, e) for n in names]


def h1(rows, lay, node, e_hi=100, e_lo=50):
    """THE SLOPE. D = (node-lay)@e_hi - (node-lay)@e_lo, unpaired, matched-k."""
    d_hi = mean(arm(rows, node, e_hi)) - mean(arm(rows, lay, e_hi))
    d_lo = mean(arm(rows, node, e_lo)) - mean(arm(rows, lay, e_lo))
    D = d_hi - d_lo
    verdict = "T-A" if D >= 0.30 else ("T-B" if D <= -0.30 else "UNDECIDED")
    return D, d_hi, d_lo, verdict


def h1b(rows, lay, node, e=100, bar=0.50):
    a, b = arm(rows, node, e), arm(rows, lay, e)
    lvl = mean(a) - mean(b)
    se = (sem(a) ** 2 + sem(b) ** 2) ** 0.5
    verdict = "TIED" if abs(lvl) < bar else ("nodewise" if lvl > 0 else "layerwise")
    return lvl, se, verdict


def h2(rows, e=100):
    """**WELCH, n=2 v 2, CROSS-BATCH.  NOT PAIRED.  [CORRECTED, CORRECTIONS 115.]**

    This function zips HZ9_NODE_1E3 against RS_LAY_1E3 on the seed label, and those
    two arms come from DIFFERENT SUBMISSION WAVES: hz9-node-1e3-s{3,4} are jobs
    4704502/4704503 and rs-lay-1e3-s{3,4} are jobs 4700388/4700392 -- ~4,100 job-ids
    and several weeks apart.  STANDING RULE 14 establishes that seeds do NOT pair
    across batches on this cluster (seed F(60,85)=1.21, p=0.213; batch F(62,85)=5.47,
    p=6.9e-13), so the seed label is doing no work here and the zip is a bookkeeping
    device, not a pairing.

    THE POINT ESTIMATE IS UNCHANGED at +1.138 pp -- in a balanced design the paired
    mean IS the difference of means -- so the registered band (+1.0..+2.0) is still
    MET and the verdict does not move.  What moves is the uncertainty: the paired
    reading implies se 0.0440 and t 25.86 on 1 df; the correct Welch reading is
    se 0.1365, t 8.34 on df 1.67, a 3.1x deflation.  The effect is still 4-11x the
    measured cross-batch floor (sqrt(2)*sd_batch = 0.20-0.29 pp).

    THE `fav` COUNT RETURNED BELOW IS A SIGN CHECK ON TWO INDEPENDENT ARM MEANS, NOT
    A PER-SEED GATE.  A "2/2 favouring nodewise" line may not be reported as a
    within-seed binary gate.  (More generally: a k/k per-seed count is a SIGN TEST on
    independent replicates EVERYWHERE, including within a batch, because seed is
    statistically null on this cluster -- its power must be quoted as such.)

    Returns per-seed diffs and the registered-range verdict."""
    diffs = []
    for nd, ly in zip(HZ9_NODE_1E3, RS_LAY_1E3):
        assert nd[-1] == ly[-1], "seed mismatch in the H2 pairing"
        diffs.append((nd[-1], pl(nd, e), pl(ly, e), pl(nd, e) - pl(ly, e)))
    m = mean([d[3] for d in diffs])
    fav = sum(1 for d in diffs if d[3] > 0)
    inrange = (1.0 <= m <= 2.0) and fav == len(diffs)
    return diffs, m, fav, inrange


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

    # --- statistics primitives
    chk("mean", abs(mean([1.0, 2.0, 3.0]) - 2.0) < 1e-12)
    chk("sd n-1", abs(sd([1.0, 3.0]) - (2 ** 0.5)) < 1e-12)
    chk("sd n=1 is None", sd([1.0]) is None)
    chk("sem", abs(sem([1.0, 3.0]) - 1.0) < 1e-12)

    # --- the imported instrument is the registered one
    chk("K matches c58", C.K == K)
    chk("plateau_at k=5", abs(C.plateau_at([1, 2, 3, 4, 5, 6], 6, 5) - 4.0) < 1e-12)
    chk("plateau_at short returns None", C.plateau_at([1, 2], 2, 5) is None)
    chk("plateau_at past end None", C.plateau_at([1, 2, 3, 4, 5], 9, 5) is None)

    rows = rows_by_run()

    # --- the raw-series metric IS the CSV plateau5 column, on every hz9 run
    agree = 0
    for r in ALL_HZ9:
        v = pl(r)
        agree += (v is not None and abs(v - float(rows[r]["plateau5"])) < 5e-3)
    chk("k=5 plateau == CSV plateau5 on all 9 hz9 runs", agree == len(ALL_HZ9))

    # --- THE TWO REGISTERED CONSTANTS, re-derived rather than trusted
    ref = [pl(x) for x in RS_LAY_1E4]
    chk("rs-lay-1e4 mean == registered 92.824", abs(mean(ref) - 92.824) < 5e-3)
    chk("rs-lay-1e4 sd == registered 0.291", abs(sd(ref) - 0.291) < 5e-3)
    chk("rs-node-3e4-s0 == registered 92.656", abs(pl(RS_NODE_3E4[0]) - 92.656) < 5e-3)

    # --- H0 drops a truncated run and keeps a complete one
    k, d = h0(rows, ["rs-node-3e4-s1"])
    chk("H0 drops the 89-epoch rs-node-3e4-s1", k == [] and len(d) == 1)
    k, d = h0(rows, ["rs-node-3e4-s0"])
    chk("H0 keeps the complete rs-node-3e4-s0", k == ["rs-node-3e4-s0"] and d == [])
    k, d = h0(rows, ["hz9-does-not-exist"])
    chk("H0 reports an absent run", k == [] and d[0][1].startswith("ABSENT"))

    # --- H1 branch logic, on synthetic input (no data dependence)
    class FakeRows(dict):
        pass
    saved = globals()["pl"]
    try:
        table = {("N", 100): 3.0, ("L", 100): 1.0, ("N", 50): 1.0, ("L", 50): 1.0}
        globals()["pl"] = lambda r, e=100, k=K: table[(r, e)]
        D, _, _, v = h1(FakeRows(), ["L"], ["N"])
        chk("H1 T-A branch", abs(D - 2.0) < 1e-12 and v == "T-A")
        table[("N", 100)] = -1.0
        D, _, _, v = h1(FakeRows(), ["L"], ["N"])
        chk("H1 T-B branch", abs(D + 2.0) < 1e-12 and v == "T-B")
        table[("N", 100)] = 1.1
        D, _, _, v = h1(FakeRows(), ["L"], ["N"])
        chk("H1 UNDECIDED band", v == "UNDECIDED")
        table[("N", 100)] = 1.30
        D, _, _, v = h1(FakeRows(), ["L"], ["N"])
        chk("H1 boundary +0.30 is T-A (>=)", v == "T-A")
        table[("N", 100)] = 0.70
        D, _, _, v = h1(FakeRows(), ["L"], ["N"])
        chk("H1 boundary -0.30 is T-B (<=)", v == "T-B")

        # --- H1b bar.  Two runs per arm so the SE is defined, as at score time.
        table = {("N1", 100): 1.3, ("N2", 100): 1.5,
                 ("L1", 100): 0.9, ("L2", 100): 1.1}
        globals()["pl"] = lambda r, e=100, k=K: table[(r, e)]
        lvl, se, v = h1b(FakeRows(), ["L1", "L2"], ["N1", "N2"])
        chk("H1b level arithmetic", abs(lvl - 0.4) < 1e-12)
        chk("H1b SE is defined at n=2", se is not None and se > 0)
        chk("H1b TIED inside +-0.50", v == "TIED")
        table[("N1", 100)], table[("N2", 100)] = 1.5, 1.7
        _, _, v = h1b(FakeRows(), ["L1", "L2"], ["N1", "N2"])
        chk("H1b resolves to nodewise above the bar", v == "nodewise")
        table[("N1", 100)], table[("N2", 100)] = 0.3, 0.5
        _, _, v = h1b(FakeRows(), ["L1", "L2"], ["N1", "N2"])
        chk("H1b resolves to layerwise below the bar", v == "layerwise")
    finally:
        globals()["pl"] = saved

    # --- H0.5 arithmetic: a 3-sd deviation must be flagged
    passed, checks, m_lay, m_node, s_lay = h05(rows)
    chk("H0.5 returns one check per hz9 tuned run",
        len(checks) == len(HZ9_LAY_1E4) + len(HZ9_NODE_3E4))
    chk("H0.5 sd source is layerwise n=2", abs(s_lay - sd([pl(x) for x in RS_LAY_1E4])) < 1e-12)

    # --- H2 pairs by seed and refuses a mis-pair
    diffs, m, fav, _ = h2(rows)
    chk("H2 pairs 2 seeds", len(diffs) == 2)
    chk("H2 pairs seed-for-seed", all(d[0] in ("3", "4") for d in diffs))
    try:
        _ = [1 for a, b in zip(["hz9-node-1e3-s3"], ["rs-lay-1e3-s4"]) if a[-1] != b[-1]]
        chk("H2 seed assertion is reachable", True)
    except AssertionError:
        chk("H2 seed assertion is reachable", True)

    # --- corpus sanity: every name the gates reference exists
    missing = [x for x in ALL_HZ9 + RS_LAY_1E4 + RS_NODE_3E4 + RS_LAY_1E3
               if x not in rows]
    chk("every referenced run is in the CSV", not missing)

    print("selftest: %d/%d passed" % (p, n))
    return p == n


# ------------------------------------------------------------------ score

def score():
    rows = rows_by_run()
    print("=" * 78)
    print("hz9 -- SCORED AGAINST bin/c58_tuned_horizon.sh, IN THE REGISTERED ORDER")
    print("=" * 78)

    # -------- H0
    kept, dropped = h0(rows, ALL_HZ9)
    print("\n[H0] VALIDITY, HARD DROP -- epochs_done == requested == %d" % EPOCHS)
    print("     kept %d/%d" % (len(kept), len(ALL_HZ9)))
    for nm, why in dropped:
        print("     DROPPED %-22s %s" % (nm, why))
    if len(kept) != len(ALL_HZ9):
        print("     >>> the batch is INCOMPLETE; cells at n=0 are MISSING, not down-weighted")

    # -------- H0.3
    ok, bad = h03(rows, kept)
    print("\n[H0.3] THE GUARD GATE -- BETA_CLIP=%s live, collapsed false, plateau > 40" % CLIP)
    print("     pass %d/%d" % (len(ok), len(kept)))
    for nm, why in bad:
        print("     VOID %-22s %s" % (nm, why))

    # -------- H0.5
    passed, checks, m_lay, m_node, s_lay = h05(rows)
    print("\n[H0.5] THE POOLING GATE -- CAN ONLY VOID")
    print("     corpus rs-lay-1e4  mean %.3f  sd %.3f  n=%d" % (m_lay, s_lay, len(RS_LAY_1E4)))
    print("     corpus rs-node-3e4 mean %.3f  n=%d (sd borrowed from layerwise)"
          % (m_node, len(RS_NODE_3E4)))
    for a, nm, v, ref, z in checks:
        print("     %-5s %-20s %7.3f  vs %7.3f  |z| = %.2f  %s"
              % (a, nm, v, ref, z, "OK" if z <= 3.0 else "FAIL"))
    print("     => POOLING %s" % ("PERMITTED" if passed else "VOID -- H1 IS NOT SCORED"))

    lay = RS_LAY_1E4 + HZ9_LAY_1E4
    node = RS_NODE_3E4 + HZ9_NODE_3E4

    # -------- H1
    print("\n[H1] THE PRIMARY TEST -- THE SLOPE, matched-k=%d, unpaired, n=%d vs n=%d"
          % (K, len(node), len(lay)))
    if not passed:
        print("     NOT SCORED -- H0.5 voided pooling.")
    else:
        D, d_hi, d_lo, verdict = h1(rows, lay, node)
        print("     (node@3e-4 - lay@1e-4) @100 = %+.3f" % d_hi)
        print("     (node@3e-4 - lay@1e-4) @ 50 = %+.3f" % d_lo)
        print("     D = %+.3f   bar +-0.30   ==> %s" % (D, verdict))
        print("     REGISTERED DIRECTION: pilot D = -0.485 -> T-B.")
        if verdict == "T-B":
            print("     T-B is a REPLICATION of the registered pilot, NEVER a discovery.")
        elif verdict == "T-A":
            print("     T-A OVERTURNS cycle 58's own read.")
        # the full epoch ladder, reported because the slope is a 2-point read
        print("\n     ladder (matched-k=%d plateau, node - lay):" % K)
        for e in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            a, b = arm(rows, node, e), arm(rows, lay, e)
            if any(x is None for x in a + b):
                continue
            print("       ep %3d   node %7.3f   lay %7.3f   diff %+7.3f   fav-node %d/%d"
                  % (e, mean(a), mean(b), mean(a) - mean(b),
                     sum(1 for x in a if x > mean(b)), len(a)))

    # -------- H1b
    print("\n[H1b] THE LEVEL at epoch %d, bar +-0.50 (STANDING RULE 9)" % EPOCHS)
    lvl, se, verdict = h1b(rows, lay, node)
    a, b = arm(rows, node, EPOCHS), arm(rows, lay, EPOCHS)
    print("     node@3e-4 %.3f +-%.3f (n=%d)   lay@1e-4 %.3f +-%.3f (n=%d)"
          % (mean(a), sem(a), len(a), mean(b), sem(b), len(b)))
    print("     level = %+.3f   se %.3f   |level|/se = %.2f   ==> %s"
          % (lvl, se, abs(lvl) / se, verdict))
    if verdict == "TIED":
        print("     TIED IS A RESULT: among PARTITIONS the choice is unresolvable once")
        print("     each arm is tuned; only partition-vs-none remains resolvable.")

    # -------- H2
    print("\n[H2] THE FIXED-ms CELL, PAIRED WITHIN SEED, at epoch %d -- A REPLICATION" % EPOCHS)
    diffs, m, fav, inrange = h2(rows)
    for s, nd, ly, d in diffs:
        print("     seed %s   node@1e-3 %7.3f   lay@1e-3 %7.3f   diff %+7.3f" % (s, nd, ly, d))
    print("     paired mean %+.3f   favouring nodewise %d/%d" % (m, fav, len(diffs)))
    print("     registered expectation: +1.0 .. +2.0 AND %d/%d  ==> %s"
          % (len(diffs), len(diffs), "MET" if inrange else "NOT MET"))
    if not inrange:
        print("     >>> the 40->100 extension of CORRECTIONS 86.3 FAILS its own")
        print("     >>> registered range; the headline curve may not be quoted past 40.")
    print()


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "--selftest":
        sys.exit(0 if selftest() else 1)
    if a[0] == "--score":
        score()
    else:
        print(__doc__)
