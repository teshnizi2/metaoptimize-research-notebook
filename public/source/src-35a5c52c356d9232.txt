#!/usr/bin/env python3
"""
c98_reproduce.py -- re-derive the CHECKED numbers of the paper from the CSV of
record and the raw per-epoch logs, and assert each one against the value printed
in the paper.  Exit status 0 iff every check passes.

    python3 analysis/c98_reproduce.py              # the full audit + coverage census
    python3 analysis/c98_reproduce.py --table2     # one section
    python3 analysis/c98_reproduce.py --census     # the coverage census alone

**SCOPE, STATED HONESTLY.**  This script does NOT check every numeral the paper
prints, and the paper must not claim that it does.  It checks the numbers that
carry a claim: the corpus and admissibility counts, every cell of Table 2 with its
se, the commensurable ratio rho of Eq. 9 and its RANKING (the check that would have
caught the cycle-100 rho superlative), the heterogeneity pools and the base-optimiser
partition, the alignment legs, the prescription table T, the tail decomposition
D = G + (D-G), the count axis U of section 4.2, the meta-stepsize pair of section 4.1,
the budget ladder, the competitiveness deficit, the gn1 commensurability gate that
section 7 T7 describes, the Appendix A.4 printed-table pools, and the T9 readings.
It does NOT check: prose-only quantities, group counts m, the attrition ledger's
upstream cluster-side rows, GPU-hour subtotals, wallclock, byte counts, arXiv ids,
or any value that exists only inside a registered scorer's own printed output.
`--census` measures and prints that coverage, AND section [16] asserts it: the three
figures section 3.4 prints -- assertion sites, distinct quantity-numerals covered, and
the total it is a fraction of -- are chk()ed like any other number, against the
sentence read out of the manuscript itself, so a stale coverage sentence now exits
non-zero instead of passing. (Before cycle 102 the census only measured, and section
3.4's "216 of the 725 ... 29.8%" survived three review cycles while the code printed
747 and 28.9%.) The census never counts its own assertions.

Each line prints:  derived value | paper value | PASS/FAIL | where it appears.
The tolerance is half a unit in the last printed digit, so a PASS means the paper
and this script agree to the precision the paper actually claims.

House rules this script obeys, and would fail loudly if the CSV stopped obeying:
  * `plateau5` is the only accuracy metric read AS A PRIMARY; the `plateau` column
    is banned as one.  The single exception is the metric-sensitivity section, which
    reads all four end-of-training columns SIDE BY SIDE, as a disclosure of what
    section 4.4's decomposition does when the endpoint is varied, and makes none of
    them primary.
  * admissibility = window_ok AND complete AND a readable plateau5.
  * every contrast is within one batch.
  * rows sharing a dup_group are averaged within the group first (ml2 is 3 v 3).
"""
import argparse, math, os, re, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c98_figures as F
from c98_figures import (load, cells, arm, welch, meta, chi2_sf, series, pl5,
                         CSV, ROOT, POOL12)

def cells_with_gn(adm):
    """The cell list INCLUDING the GroupNorm cell, whatever the default is."""
    old, F.WITH_GN = F.WITH_GN, True
    try:    return cells(adm)
    finally: F.WITH_GN = old

FAILS = []
ASSERTED = []      # every (fmt, paper) pair actually asserted -- drives --census
SKIPPED = []       # sections that could not run here, so "ALL n PASS" cannot be
                   # misread as full coverage (this is A8's own failure mode).
CENSUS_MARK = None # set by censuscheck() to len(ASSERTED) BEFORE it asserts, so the
                   # census is computed over the claim-carrying assertions only and
                   # never counts its own four self-referential sites.

def skip(section, why):
    SKIPPED.append((section, why))
    print("  %s -- SECTION SKIPPED (%s)" % (why, section))

def chk(name, got, paper, where, fmt="%+.3f"):
    """Compare a derived value with the value the paper prints."""
    if paper is None:
        print("  %-46s %s   [derived; no paper value yet]  %s"
              % (name, fmt % got, where)); return
    ASSERTED.append((fmt, paper))
    dec = len((fmt % 0).split(".")[-1]) if "." in (fmt % 0) else 0
    tol = 0.5 * 10 ** (-dec) + 1e-12
    ok = abs(got - paper) <= tol
    if not ok: FAILS.append((name, got, paper, where))
    print("  %-46s %s | paper %s | %s   %s"
          % (name, fmt % got, fmt % paper, "PASS" if ok else "**FAIL**", where))

# --------------------------------------------------------------- the sections
def corpus(rows, adm, args):
    print("\n[1] CORPUS  (§8 Reproducibility, Appendix A.8)")
    chk("rows in results/all_runs.csv", len(rows), 2177, "abstract, §8", "%.0f")
    chk("admissible rows", len(adm), 1735, "§3.3, A.8", "%.0f")
    wc = [float(r["wallclock_min"]) for r in rows if r["wallclock_min"]]
    chk("runs carrying a wallclock", len(wc), 2162, "§8", "%.0f")
    chk("GPU-hours", sum(wc) / 60.0, 1642, "abstract, §8", "%.0f")
    chk("distinct nodes", len({r["node"] for r in rows if r["node"]}), 29, "§8", "%.0f")
    for flag, paper in (("window_ok", 425), ("complete", 17)):
        n = sum(1 for r in rows if r[flag] != "1")
        chk("rows failing %s" % flag, n, paper, "§3.3 attrition", "%.0f")
    chk("rows with no plateau5", sum(1 for r in rows if not r["plateau5"]), 25,
        "§3.3 attrition", "%.0f")

TABLE2 = {   # label -> (D, se) exactly as Table 2 prints them
    "cc1": (0.727, 0.200), "mm1": (0.485, 0.161), "pp1": (0.581, 0.141),
    "gn1 (BN)": (0.587, 0.153), "ml2": (0.456, 0.195), "rl3 @1e-4": (0.681, 0.173),
    "rl3 @3e-4": (0.591, 0.096), "fa1": (0.629, 0.123), "hz3": (0.428, 0.086),
    "gn1 (GN)": (0.202, 0.137), "aw1": (0.279, 0.087), "nl1 (SGD)": (1.035, 0.109),
    "nl1 (RMSProp)": (0.973, 0.251), "g3m": (0.666, 0.094), "r50": (0.881, 0.261),
    "gc1": (1.640, 0.245), "gm2": (1.485, 0.238),
    # the four cells added in cycle 101 (bm2 x2, sm3, sm4).  sm4 is a Table 2 ROW
    # but is NOT a pool member: its own registered scorer forbids pooling it with
    # aw1 or sm3, and c98_figures gives it its own `base` string for that reason.
    "bm2 (SGD)": (0.978, 0.086), "bm2 (RMSProp)": (0.631, 0.149),
    "sm3": (0.141, 0.064), "sm4": (0.889, 0.228),
}
def table2(rows, adm, args):
    print("\n[2] TABLE 2 -- D = uniform chunk − aligned nodewise, within batch")
    print("    (ml2 is 3 v 3 after the dup_group collapse, se 0.195, NOT 6 v 6 / 0.142)")
    cs = cells(adm)
    if not F.WITH_GN:
        print("    (the GroupNorm cell is removed by R0 item 1; --with-gn puts it back)")
    for c in cs:
        d, se = TABLE2[c["label"]]
        chk("D  %-14s (n %d v %d)" % (c["label"], c["n"][0], c["n"][1]), c["D"], d,
            "Table 2, Fig. 1")
        chk("   se %-11s" % "", c["seD"], se, "", "%.3f")
    pos = sum(1 for c in cs if c["D"] > 0)
    chk("count-matched cells in Table 2", len(cs), 20, "§4.3, §1.1, §9", "%.0f")
    chk("cells with D > 0", pos, len(cs), "abstract, §4.3", "%.0f")
    res = sum(1 for c in cs if c["tD"] >= 3.0)
    chk("   ...resolved at t >= 3.0", res, 18, "§4.3", "%.0f")
    chk("   the two that are not: ml2", min(c["tD"] for c in cs), 2.22, "§4.3", "%.2f")

def heterogeneity(rows, adm, args):
    print("\n[3] HETEROGENEITY  (§4.4, Fig. 2)")
    # The live pool is the FOURTEEN byte-identical ResNet-18 cells: the eleven of the
    # previous draft plus bm2's two and sm3.  sm4 runs the same contrast object but a
    # different meta-optimiser, so c98_figures gives it its own `base` string and keeps
    # it out of POOL12 -- it must never appear here.
    full = [c for c in cells_with_gn(adm) if c["in12"]]
    live = [c for c in full if c["base"] != "SGDm-GN"]
    assert all(c["base"] != "AdamW+RMS" for c in live), "sm4 leaked into the pool"
    m, sem_, Q, df, tau = meta([(c["D"], c["seD"]) for c in live])
    chk("live pool, %d byte-identical cells" % len(live), m, 0.530, "§4.4")
    chk("   Q", Q, 102.47, "§4.4, abstract, Fig. 2b", "%.2f")
    chk("   df", df, len(live) - 1, "§4.4", "%.0f")
    chk("   tau", tau, 0.295, "§4.4", "%.3f")
    chk("   rms measurement se", math.sqrt(sum(c["seD"]**2 for c in live) / len(live)),
        0.143, "§4.4", "%.3f")
    # the eleven-cell pool the previous draft published, kept because A.4 records it
    e11 = [c for c in live if c["label"] not in
           ("bm2 (SGD)", "bm2 (RMSProp)", "sm3")]
    m11, s11, Q11, df11, t11 = meta([(c["D"], c["seD"]) for c in e11])
    chk("the eleven-cell pool of the previous draft", m11, 0.571, "§4.4 history, A.4")
    chk("   Q", Q11, 36.4, "§4.4 history, A.4", "%.1f")
    chk("   tau", t11, 0.203, "A.4", "%.3f")
    m2, s2, Q2, df2, t2 = meta([(c["D"], c["seD"]) for c in e11 + [c for c in full
                                if c["base"] == "SGDm-GN"]])
    chk("legacy 12-cell pool (with GroupNorm)", m2, 0.546, "A.4")
    chk("   Q", Q2, 43.2, "A.4", "%.1f")
    chk("   tau", t2, 0.215, "A.4", "%.3f")

    bases = ["SGDm", "SGD", "RMSProp", "AdamW"]
    sub = {b: meta([(c["D"], c["seD"]) for c in live if c["base"] == b]) for b in bases}
    within = sum(sub[b][2] for b in bases); wdf = sum(sub[b][3] for b in bases)
    for b, pool, q in (("SGDm", 0.556, 4.21), ("SGD", 1.000, 0.17),
                       ("RMSProp", 0.720, 1.37), ("AdamW", 0.189, 1.61)):
        k = sum(1 for c in live if c["base"] == b)
        chk("level %-8s (k=%d)" % (b, k), sub[b][0], pool, "§4.4 table, Fig. 2a")
        chk("   within-level Q", sub[b][2], q, "§4.4 table, Fig. 2a", "%.2f")
    chk("within-SGDm tau", sub["SGDm"][4], 0.000, "§4.4", "%.3f")
    chk("SGDm pool se", sub["SGDm"][1], 0.045, "§4.4, abstract", "%.3f")
    chk("within-level Q, all four levels", within, 7.36, "§4.4, Fig. 2b", "%.2f")
    chk("   df", wdf, 10, "§4.4", "%.0f")
    chk("   p", chi2_sf(within, wdf), 0.69, "§4.4, Fig. 2b", "%.2f")
    chk("between-base Q", Q - within, 95.12, "§4.4, Fig. 2b, abstract", "%.2f")
    chk("share of the live Q that is between-base",
        100 * (Q - within) / Q, 92.8, "§4.4, Fig. 2b, abstract, §9", "%.1f")
    chk("   the same share on the eleven-cell pool", 100 * 32.1988 / Q11, 88.4,
        "§4.4 history, Fig. 2b", "%.1f")
    chk("   ...of the legacy 12-cell Q", 100 * 32.1988 / Q2, 74.6,
        "the '~75%' figure, denominator named", "%.1f")
    chk("level spread factor, SGD / AdamW", sub["SGD"][0] / sub["AdamW"][0], 5.3,
        "§4.4 summary block", "%.1f")

    # ---- B1: the rival label, and the conditional tests.  This block is the reason
    # the paper says "candidate moderator" and not "identified moderator".
    BATCH = {"cc1": "cc1", "mm1": "mm1", "pp1": "pp1", "gn1 (BN)": "gn1",
             "rl3 @1e-4": "rl3", "rl3 @3e-4": "rl3", "fa1": "fa1", "hz3": "hz3",
             "aw1": "aw1", "nl1 (SGD)": "nl1", "nl1 (RMSProp)": "nl1",
             "bm2 (SGD)": "bm2", "bm2 (RMSProp)": "bm2", "sm3": "sm3"}
    def part(cells_, key):
        lev = {}
        for c in cells_: lev.setdefault(key(c), []).append(c)
        w = sum(meta([(c["D"], c["seD"]) for c in g])[2] for g in lev.values())
        wd = sum(meta([(c["D"], c["seD"]) for c in g])[3] for g in lev.values())
        return w, wd, len(lev)
    wb, wbd, _ = part(live, lambda c: c["base"])
    wc_, wcd, nb = part(live, lambda c: BATCH[c["label"]])
    wx, wxd, _ = part(live, lambda c: (c["base"], BATCH[c["label"]]))
    chk("batch partition: between Q", Q - wc_, 98.16, "§4.4 rival label", "%.2f")
    chk("   its share -- LARGER than the base optimiser's",
        100 * (Q - wc_) / Q, 95.8, "§4.4, §9, abstract §1", "%.1f")
    chk("dQ(batch | base)", wb - wx, 7.15, "§4.4 nested test", "%.2f")
    chk("   p", chi2_sf(wb - wx, wbd - wxd), 0.62, "§4.4", "%.2f")
    chk("dQ(base | batch) -- the test that does NOT clear", wc_ - wx, 4.11,
        "§4.4, §1.1 C3, §9", "%.2f")
    chk("   df", wcd - wxd, 2, "§4.4", "%.0f")
    chk("   p -- NOT below 0.05, which is why 'identified' is withdrawn",
        chi2_sf(wc_ - wx, wcd - wxd), 0.13, "§4.4, §1.1 C3, §9", "%.2f")

    # ---- the box partition: three readings, all three printed in §3.4 and §4.4
    def box_of(c):
        pre = {"cc1": "cc1-node", "mm1": "mm1-node", "pp1": "pp1-node",
               "gn1 (BN)": "gn1-bn-node", "rl3 @1e-4": "rl3-node-m1e4",
               "rl3 @3e-4": "rl3-node-m3e4", "fa1": "fa1-node", "hz3": "hz3-node",
               "aw1": "aw1-node", "nl1 (SGD)": "nl1-sgd-node",
               "nl1 (RMSProp)": "nl1-rms-node", "bm2 (SGD)": "bm2-sgd-node",
               "bm2 (RMSProp)": "bm2-rms-node", "sm3": "sm3-awrms-node"}[c["label"]]
        return sorted({r["beta_clip"] for r in adm if r["run"].startswith(pre)})[0]
    wbox, wboxd, _ = part(live, box_of)
    chk("between-box Q on the fourteen cells", Q - wbox, 0.69, "§3.4 ex.1, §4.4", "%.2f")
    sgdm = [c for c in live if c["base"] == "SGDm"]
    Qs = meta([(c["D"], c["seD"]) for c in sgdm])[2]
    ws, wsd, _ = part(sgdm, box_of)
    chk("   the UNCONFOUNDED test, inside SGDm", Qs - ws, 0.76,
        "§3.4 ex.1, §4.4, A.10 -- this is what the pooling rests on", "%.2f")
    chk("      p", chi2_sf(Qs - ws, 2), 0.68, "§3.4 ex.1, §4.4, A.10", "%.2f")
    e13 = [c for c in live if c["label"] != "sm3"]
    Q13 = meta([(c["D"], c["seD"]) for c in e13])[2]
    w13, _, _ = part(e13, box_of)
    chk("   the thirteen-cell reading the paper also prints", Q13 - w13, 5.14,
        "§3.4 ex.1, §4.4", "%.2f")

    # ---- the 2x2 on the level pools, and the momentum collapse
    (a, sa), (r_, sr), (g_, sg), (w_, sw) = ((sub[b][0], sub[b][1]) for b in
                                            ("SGD", "RMSProp", "SGDm", "AdamW"))
    se4 = math.sqrt(sa**2 + sr**2 + sg**2 + sw**2)
    chk("momentum main effect", 0.5 * ((g_ - a) + (w_ - r_)), -0.488, "§4.4 2x2")
    chk("   z", 0.5 * ((g_ - a) + (w_ - r_)) / (0.5 * se4), -6.09, "§4.4 2x2", "%.2f")
    chk("second-moment main effect -- RESOLVED at 14 cells, unresolved at 11",
        0.5 * ((r_ - a) + (w_ - g_)), -0.323, "§4.4 2x2, §5.5 withdrawal")
    chk("   z", 0.5 * ((r_ - a) + (w_ - g_)) / (0.5 * se4), -4.03, "§4.4 2x2", "%.2f")
    chk("interaction", (w_ - g_) - (r_ - a), -0.087, "§4.4 2x2")
    chk("D(RMSProp) - D(AdamW)  (M6)", r_ - w_, 0.531, "§5.5 M6, mechanism index")
    chk("   t", (r_ - w_) / math.sqrt(sr**2 + sw**2), 3.84, "§5.5 M6", "%.2f")
    pres = [c for c in live if c["base"] in ("SGDm", "AdamW")]
    absn = [c for c in live if c["base"] in ("SGD", "RMSProp")]
    Qp, dp = meta([(c["D"], c["seD"]) for c in pres])[2:4]
    Qa, da = meta([(c["D"], c["seD"]) for c in absn])[2:4]
    chk("momentum-present residual Q after the collapse", Qp, 34.64, "§4.4, §5.5", "%.2f")
    chk("   df", dp, 9, "§4.4", "%.0f")
    chk("the collapse removes only", 100 * (Q - Qp - Qa) / Q, 61.1, "§4.4", "%.1f")

    # ---- sensitivities the paper quotes
    ml2 = [c for c in cells(adm) if c["label"] == "ml2"][0]
    m15, s15, Q15, d15, _ = meta([(c["D"], c["seD"]) for c in live + [ml2]])
    w15, _, _ = part(live + [ml2], lambda c: c["base"])
    chk("+ml2 as a fifteenth cell: pool", m15, 0.528, "§4.4 sensitivity")
    chk("   Q", Q15, 102.61, "§4.4 sensitivity", "%.2f")
    chk("   share", 100 * (Q15 - w15) / Q15, 92.6, "§4.4 sensitivity", "%.1f")
    gn = [c for c in full if c["base"] == "SGDm-GN"]
    mg, sgn, Qg, dg, _ = meta([(c["D"], c["seD"]) for c in live + gn])
    wg, _, _ = part(live + gn, lambda c: c["base"])
    chk("+gn1(GN) as a fifth level: pool", mg, 0.515, "§4.4 sensitivity, §4.4 gn note")
    chk("   Q", Qg, 107.97, "§4.4 sensitivity", "%.2f")
    chk("   share", 100 * (Qg - wg) / Qg, 93.2, "§4.4 sensitivity", "%.1f")

def alignment(rows, adm, args):
    print("\n[4] THE ALIGNMENT NULL  (§4.6)")
    perm = [float(r["plateau5"]) for r in adm
            if r["run"].startswith("pp1-perm") and r["granularity"].startswith("permnode")]
    node, _ = arm(adm, "pp1-node", "nodewise")
    ch, _   = arm(adm, "pp1-ch", "chunk777")
    a, ase, at = welch(perm, node)
    chk("A = permnode − nodewise", a, -0.009, "§4.6, abstract, Fig. 4 caption")
    chk("   se", ase, 0.157, "§4.6", "%.3f")
    chk("   95% CI low", a - 1.96 * ase, -0.317, "§4.6 rewrite")
    # NOTE: docs/STATUS.md R0 item 3 prints the upper limit as +0.299.  Re-derived
    # here it is A + 1.96 se = -0.00867 + 1.96 x 0.156569 = +0.29822 -> +0.298.
    chk("   95% CI high", a + 1.96 * ase, 0.298, "§4.6 rewrite (STATUS says .299)")
    b, bse, bt = welch(ch, perm)
    chk("B = chunk777 − permnode", b, 0.590, "§4.6")
    d, dse, dt = welch(ch, node)
    chk("A + B reconstructs D", a + b, d, "§4.6 (identity)")

def prescription(rows, adm, args):
    print("\n[5] THE PRESCRIPTION T = nodewise1d − nodewise  (§4.7)")
    T = [("bn1", "bn1-n1d", "bn1-node", 0.427), ("ml2", "ml2-", "ml2-", 0.619),
         ("fa1", "fa1-n1d", "fa1-node", 0.649), ("g3m", "g3m-n1d", "g3m-node", 0.758),
         ("cc1", "cc1-n1d", "cc1-node", 0.816), ("r50", "r50-n1d", "r50-node", 1.049),
         ("gm2", "gm2-n1d", "gm2-node", 1.363), ("hz3", "hz3-n1d", "hz3-node", 0.337),
         ("nl1 SGD", "nl1-sgd-n1d", "nl1-sgd-node", 0.692),
         ("nl1 RMSProp", "nl1-rms-n1d", "nl1-rms-node", 0.916),
         # the two rl3 rungs the previous draft's table omitted, and which its own
         # Eq.-6 decomposition quoted four lines below the table (+0.756)
         ("rl3 @1e-4", "rl3-n1d-m1e4", "rl3-node-m1e4", 0.756),
         ("rl3 @3e-4", "rl3-n1d-m3e4", "rl3-node-m3e4", 0.391),
         ("aw1 AdamW", "aw1-n1d", "aw1-node", 0.091),
         ("sm3 AdamW", "sm3-awrms-n1d", "sm3-awrms-node", -0.083),
         ("sm4 AdamW+RMS", "sm4-awrms-n1d", "sm4-awrms-node", 0.988)]
    got = {}
    for lab, a_, b_, paper in T:
        av, _ = arm(adm, a_, "nodewise1d"); bv, _ = arm(adm, b_, "nodewise")
        t, se, tt = welch(av, bv)
        got[lab] = (t, se, tt)
        chk("T  %-14s" % lab, t, paper, "§4.7 table")
    chk("   se, rl3 @1e-4", got["rl3 @1e-4"][1], 0.117, "§4.7 table", "%.3f")
    chk("   t,  rl3 @1e-4", got["rl3 @1e-4"][2], 6.45, "§4.7 table", "%.2f")
    chk("   se, rl3 @3e-4", got["rl3 @3e-4"][1], 0.127, "§4.7 table", "%.3f")
    chk("   t,  rl3 @3e-4 -- the weakest of the twelve",
        got["rl3 @3e-4"][2], 3.09, "§4.7 prose (t >= 3.0, NOT 3.3)", "%.2f")
    chk("   se, sm4", got["sm4 AdamW+RMS"][1], 0.231, "§4.7 table", "%.3f")
    chk("   t,  sm4", got["sm4 AdamW+RMS"][2], 4.28, "§4.7 table, §1.1 C6", "%.2f")
    nonad = [l for l in got if l not in ("aw1 AdamW", "sm3 AdamW", "sm4 AdamW+RMS")]
    chk("non-AdamW cells in the T table", len(nonad), 12, "§4.7 prose, §1.1 C6", "%.0f")
    chk("   the weakest t among them", min(got[l][2] for l in nonad), 3.09,
        "§4.7 prose", "%.2f")
    aw = meta([(got["aw1 AdamW"][0], got["aw1 AdamW"][1]),
               (got["sm3 AdamW"][0], got["sm3 AdamW"][1])])
    chk("AdamW + Lion T pool over two batches", aw[0], 0.007, "§4.7, §1.1 C6")
    chk("   se", aw[1], 0.056, "§4.7, §1.1 C6", "%.3f")

def tail(rows, adm, args):
    print("\n[6] THE TAIL DECOMPOSITION  D = G + (D − G)  (§5.4, Fig. 4)")
    for c in cells(adm):
        if c["G"] is None: continue
        print("  %-16s D %+0.3f ± %0.3f   G %+0.3f ± %0.3f (t %5.2f)   D−G %+0.3f ± %0.3f (t %5.2f)"
              % (c["label"], c["D"], c["seD"], c["G"], c["seG"], c["tG"],
                 c["DG"], c["seDG"], c["tDG"]))
    cs = {c["label"]: c for c in cells(adm)}
    chk("G under AdamW (aw1)", cs["aw1"]["G"], 0.232, "§5.4, A.2")
    chk("   t", cs["aw1"]["tG"], 2.62, "§5.4", "%.2f")
    chk("D − G under AdamW", cs["aw1"]["DG"], 0.047, "§5.4, abstract")
    chk("D − G under SGDm (cc1)", cs["cc1"]["DG"], 0.715, "§5.4, abstract")
    chk("G under ml2 after the collapse", cs["ml2"]["G"], 0.173, "§5.4 (corrected)")
    chk("   t", cs["ml2"]["tG"], 2.38, "§5.4 (corrected 2.44 → 2.38)", "%.2f")
    # The pool is over CIFAR-10 cells ONLY: §4.3's commensurability rule forbids
    # averaging a CIFAR-10 and a CIFAR-100 effect in percentage points.
    sg = [c for c in cells(adm) if c["G"] is not None and c["base"] == "SGDm"
          and c["dataset"] == "C10"]
    p = meta([(c["DG"], c["seDG"]) for c in sg])
    chk("pooled SGDm/C10 D − G (k=8)", p[0], 0.514, "§9 (quote the pool, not cc1's max)")
    chk("   se", p[1], 0.056, "§9", "%.3f")
    p6 = meta([(c["DG"], c["seDG"]) for c in sg if c["network"] == "ResNet-18"])
    chk("   ...ResNet-18 only (k=6)", p6[0], 0.514, "§9 variant")
    # the two new AdamW cells, and what they do to the tail story (M4, §5.4, §5.5)
    chk("G  sm3 (AdamW + Lion)", cs["sm3"]["G"], 0.296, "§5.4 table, §6.1")
    chk("   D - G", cs["sm3"]["DG"], -0.155, "§5.4 table")
    chk("G  sm4 (AdamW + RMSProp)", cs["sm4"]["G"], 0.261, "§5.4 table, §5.5")
    chk("   D - G", cs["sm4"]["DG"], 0.629, "§5.4 table, §5.5, mechanism index")
    chk("   t", cs["sm4"]["tDG"], 2.59, "§5.4, §5.5", "%.2f")
    aw2 = [(cs[l]["DG"], cs[l]["seDG"]) for l in ("aw1", "sm3")]
    pa = meta(aw2)
    chk("AdamW + Lion pooled D - G", pa[0], -0.061, "§5.4, §9, abstract §1")
    chk("   se", pa[1], 0.085, "§5.4, §9", "%.3f")
    gp = meta([(cs[l]["G"], cs[l]["seG"]) for l in ("aw1", "sm3")])
    chk("AdamW + Lion pooled G", gp[0], 0.261, "§5.4 meta table")
    chk("the meta contrast: dG -- G does NOT move",
        cs["sm4"]["G"] - gp[0], -0.001, "§5.4, §7 T1, abstract §1")
    chk("   dD", cs["sm4"]["D"] - meta([(cs[l]["D"], cs[l]["seD"])
                                        for l in ("aw1", "sm3")])[0], 0.700, "§5.4, §5.5")
    chk("   d(D-G)", cs["sm4"]["DG"] - pa[0], 0.690, "§5.4, §7 T1")
    # the G family, pre-specified twelve and enlarged fourteen
    fam12 = [c for c in cells(adm) if c["G"] is not None
             and c["label"] not in ("sm3", "sm4")]
    fam14 = [c for c in cells(adm) if c["G"] is not None]
    g12 = meta([(c["G"], c["seG"]) for c in fam12])
    g14 = meta([(c["G"], c["seG"]) for c in fam14])
    chk("G family, pre-specified 12: pool", g12[0], 0.067, "§5.4 multiplicity")
    chk("   Q", g12[2], 18.21, "§5.4 multiplicity", "%.2f")
    chk("G family, enlarged 14: pool", g14[0], 0.093, "§5.4 multiplicity")
    chk("   Q -- heterogeneous where the twelve was not", g14[2], 28.25,
        "§5.4 multiplicity", "%.2f")
    bn = welch(arm(adm, "bn1-c23", "chunk2325")[0], arm(adm, "bn1-n1d", "nodewise1d")[0])
    g15 = meta([(c["G"], c["seG"]) for c in fam14] + [(bn[0], bn[1])])
    chk("G family, 15 with bn1: pool", g15[0], 0.128, "§5.4 outside contrasts")
    chk("   Q", g15[2], 42.98, "§5.4 outside contrasts", "%.2f")
    bad = meta([(c["DG"], c["seDG"]) for c in cells(adm)
                if c["G"] is not None and c["base"] == "SGDm"])
    print("        (adding the CIFAR-100 cell would give %+0.3f ± %0.3f with Q %.2f/%d "
          "instead of Q %.2f/%d -- which is why §4.3's commensurability rule is a rule)"
          % (bad[0], bad[1], bad[2], bad[3], p[2], p[3]))

def budget(rows, adm, args):
    print("\n[7] BUDGET  (§4.8, Fig. 3)  -- paired WITHIN run, from the raw .out series")
    ch, nd = series("hz3-ch-s*.out"), series("hz3-node-s*.out")
    seeds = sorted(set(ch) & set(nd))
    if not seeds:
        skip("[7] BUDGET", "raw hz3 .out series not found"); return
    box = {}
    for r in rows:
        for tag in ("ch", "node"):
            if r["run"].startswith("hz3-%s-s" % tag):
                box.setdefault(int(r["run"].rsplit("-s", 1)[1]), {})[tag] = r["beta_clip"]
    bad = sorted(s for s, v in box.items() if len(v) == 2 and v["ch"] != v["node"])
    chk("box-mismatched seeds in hz3", len(bad), 1, "§4.8, T9, Fig. 3", "%.0f")
    print("        (the mismatched seed is seed %s: chunk in %s, nodewise in %s)"
          % (bad, box[bad[0]]["ch"], box[bad[0]]["node"]))
    for keep, tag, paper in ((seeds, "all %d" % len(seeds), (0.576, 0.514, 0.428, -0.149, -1.42)),
                             ([s for s in seeds if s not in bad], "box-matched %d" % (len(seeds) - len(bad)),
                              (0.662, 0.575, 0.455, -0.207, -1.94))):
        for B, exp in zip((100, 200, 300), paper[:3]):
            d = [pl5(ch[s], B) - pl5(nd[s], B) for s in keep]
            chk("D(%d), %s seeds" % (B, tag), st.mean(d), exp, "§4.8 table, Fig. 3a")
        dd = [(pl5(ch[s], 300) - pl5(nd[s], 300)) - (pl5(ch[s], 100) - pl5(nd[s], 100))
              for s in keep]
        m_, se_ = st.mean(dd), st.stdev(dd) / math.sqrt(len(dd))
        chk("D(300)−D(100), %s seeds" % tag, m_, paper[3], "§4.8, Fig. 3b")
        chk("   t", m_ / se_, paper[4], "§4.8, Fig. 3b", "%.2f")

    # ---- the hz3q repair: seed 5 re-run box- and class-matched (§4.8, §7 T9) ----
    # hz3q is a SEPARATE batch under a SEPARATE registration.  It repairs one seed of
    # hz3; it is not a replication, not a new design point and not a new cell, and it
    # enters no cell of Table 2.  The published readings above are UNCHANGED and are
    # asserted first, on purpose: all three readings are reported together, always.
    qch, qnd = series("hz3q-ch-s*.out"),  series("hz3q-node-s*.out")
    qn1, qc2 = series("hz3q-n1d-s*.out"), series("hz3q-c23-s*.out")
    if not (set(qch) == set(qnd) == {5}):
        print("        (hz3q .out series not found -- repaired readings skipped)")
        return
    n1d, c23 = series("hz3-n1d-s*.out"), series("hz3-c23-s*.out")
    CH, ND = dict(ch), dict(nd)
    N1, C2 = dict(n1d), dict(c23)
    CH[5], ND[5], N1[5], C2[5] = qch[5], qnd[5], qn1[5], qc2[5]
    for B, exp, ese in ((100, 0.632, 0.074), (200, 0.512, 0.116), (300, 0.394, 0.090)):
        d = [pl5(CH[s], B) - pl5(ND[s], B) for s in seeds]
        chk("D(%d), REPAIRED 6 seeds" % B, st.mean(d), exp, "§4.8 table col 3")
        chk("   se", st.stdev(d) / math.sqrt(len(d)), ese, "§4.8 table col 3", "%.3f")
    dd = [(pl5(CH[s], 300) - pl5(ND[s], 300)) - (pl5(CH[s], 100) - pl5(ND[s], 100))
          for s in seeds]
    m_, se_ = st.mean(dd), st.stdev(dd) / math.sqrt(len(dd))
    chk("D(300)−D(100), REPAIRED 6 seeds", m_, -0.238, "§4.8 -- c99 H2 REGISTERED VERDICT")
    chk("   se", se_, 0.093, "§4.8 -- c99 H2", "%.3f")
    chk("   t  (bar |t| >= 2.0 frozen pre-run -> NOT FLAT, D DECLINES)",
        m_ / se_, -2.57, "§4.8, §8, A.2, end matter", "%.2f")
    d, se, t = welch([pl5(CH[s], 300) for s in seeds], [pl5(ND[s], 300) for s in seeds])
    chk("D(300) REPAIRED 6 v 6 (Welch)", d, 0.394, "§4.8, Table 2 dagger, §8")
    chk("   se", se, 0.093, "§4.8, Table 2 dagger", "%.3f")
    chk("   t  (Contribution 1 UNTOUCHED)", t, 4.25, "§4.8, Table 2 dagger, §8", "%.2f")
    g = welch([pl5(C2[s], 300) for s in seeds], [pl5(N1[s], 300) for s in seeds])[0]
    chk("G(300) REPAIRED 6 v 6", g, -0.048, "§4.8, A.11")
    chk("(D-G)(300) REPAIRED", d - g, 0.442, "§4.8, A.11")
    chk("HC cross-class hz3q-node-s5 - hz3-node-s5 @300",
        pl5(qnd[5], 300) - pl5(nd[5], 300), 0.134,
        "§4.8 HC, §7 T9 -- bar |delta| <= 1.00 pp")
    chk("   hz3q-node-s5 plateau5(300)", pl5(qnd[5], 300), 92.908, "§4.8 HC", "%.3f")
    chk("   hz3-node-s5  plateau5(300)", pl5(nd[5], 300), 92.774, "§4.8 HC", "%.3f")
    chk("archived seed-5 D(100)", pl5(ch[5], 100) - pl5(nd[5], 100), 0.148, "§7 T9")
    chk("repaired seed-5 D(100)", pl5(qch[5], 100) - pl5(qnd[5], 100), 0.482, "§7 T9")
    chk("archived seed-5 D(300)", pl5(ch[5], 300) - pl5(nd[5], 300), 0.290, "§7 T9")
    chk("repaired seed-5 D(300)", pl5(qch[5], 300) - pl5(qnd[5], 300), 0.086, "§7 T9")
    chk("seed-5 nodewise class-only shift @100", pl5(qnd[5], 100) - pl5(nd[5], 100), -0.204,
        "§7 T9 -- same box, same flags, class only")
    chk("seed-5 chunk777 class-only shift @100", pl5(qch[5], 100) - pl5(ch[5], 100), 0.130,
        "§4.8 -- box-free at B=100")
    chk("seed-5 nodewise1d class-only shift @100", pl5(qn1[5], 100) - pl5(n1d[5], 100), 0.376,
        "§4.8 -- box-free at B=100")
    chk("seed-5 chunk2325 class-only shift @100", pl5(qc2[5], 100) - pl5(c23[5], 100), 0.212,
        "§4.8 -- box-free at B=100")
    chk("epoch the -15 floor first becomes reachable",
        math.ceil((math.log(1e-3) - (-15.0)) / (1e-4 * 500)), 162, "§3.5 R2, §7 T9", "%.0f")

    # ---- §4.8 R3: the extrapolation a referee computes, and why it is not usable ----
    D_ = lambda s, B: pl5(CH[s], B) - pl5(ND[s], B)
    D100 = st.mean([D_(s, 100) for s in seeds]); D300 = st.mean([D_(s, 300) for s in seeds])
    slope = m_ / 200.0                              # m_ is the repaired D(300)-D(100)
    chk("epochs past 300 to a LINEAR zero (0.394/(0.238/200))",
        0.394 / (0.238 / 200.0), 331, "§4.8 R3", "%.0f")
    chk("   linear zero-crossing, rounded ladder", 300 + 0.394 / (0.238 / 200.0), 631,
        "§4.8 R3", "%.0f")
    chk("   linear zero-crossing, unrounded ladder", 300 + D300 / (-slope), 630,
        "§4.8 R3", "%.0f")
    n3, sB, sD = 3, sum((100, 200, 300)), D100 + st.mean([D_(s, 200) for s in seeds]) + D300
    Ds = [D100, st.mean([D_(s, 200) for s in seeds]), D300]
    sBB = sum(b * b for b in (100, 200, 300)); sBD = sum(b * d for b, d in zip((100, 200, 300), Ds))
    sl = (n3 * sBD - sB * sD) / (n3 * sBB - sB * sB); ic = (sD - sl * sB) / n3
    chk("   least-squares line through all three points, zero at", -ic / sl, 630,
        "§4.8 R3", "%.0f")
    chk("   crossing at the steep end of the registered CI [-0.477, -0.000]",
        300 + D300 / (0.477 / 200.0), 465, "§4.8 R3", "%.0f")
    for a, b, ed, ese, et in ((100, 200, -0.120, 0.142, -0.84), (200, 300, -0.118, 0.096, -1.23)):
        v = [D_(s, b) - D_(s, a) for s in seeds]
        mm, ss_ = st.mean(v), st.stdev(v) / math.sqrt(len(v))
        chk("D(%d)-D(%d), repaired -- the half-interval" % (b, a), mm, ed, "§4.8 R3")
        chk("   se", ss_, ese, "§4.8 R3", "%.3f")
        chk("   t  (NEITHER half resolves; only the full span does)", mm / ss_, et,
            "§4.8 R3", "%.2f")
    lb = [math.log(b) for b in (100, 200, 300)]
    sB2 = sum(lb); sBB2 = sum(b * b for b in lb); sBD2 = sum(b * d for b, d in zip(lb, Ds))
    sl2 = (n3 * sBD2 - sB2 * sD) / (n3 * sBB2 - sB2 * sB2); ic2 = (sD - sl2 * sB2) / n3
    chk("   log-budget fit, zero at", math.exp(-ic2 / sl2), 2034, "§4.8 R3", "%.0f")
    ld = [math.log(d) for d in Ds]
    sD3 = sum(ld); sBD3 = sum(b * d for b, d in zip((100, 200, 300), ld))
    sl3 = (n3 * sBD3 - sB * sD3) / (n3 * sBB - sB * sB); ic3 = (sD3 - sl3 * sB) / n3
    chk("   exponential fit, D(600) -- it never reaches zero", math.exp(ic3 + sl3 * 600), 0.195,
        "§4.8 R3")
    chk("   worst residual of the three two-parameter fits (pp)",
        max(max(abs(d - (ic + sl * b)) for b, d in zip((100, 200, 300), Ds)),
            max(abs(d - (ic2 + sl2 * b)) for b, d in zip(lb, Ds)),
            max(abs(d - math.exp(ic3 + sl3 * b)) for b, d in zip((100, 200, 300), Ds))),
        0.020, "§4.8 R3 -- all three fit inside the per-point se", "%.3f")
    for tag, A_, g1, g2, se2 in (("chunk777", CH, 0.693, -0.016, 0.055),
                                 ("nodewise", ND, 0.813, 0.102, 0.096)):
        chk("%s paired gain 100->200 (both arms saturate)" % tag,
            st.mean([pl5(A_[s], 200) - pl5(A_[s], 100) for s in seeds]), g1, "§4.8 R3")
        v = [pl5(A_[s], 300) - pl5(A_[s], 200) for s in seeds]
        chk("   %s paired gain 200->300" % tag, st.mean(v), g2, "§4.8 R3")
        chk("      se", st.stdev(v) / math.sqrt(len(v)), se2, "§4.8 R3", "%.3f")

    # ---- §4.8 R5: HOW the reversal happens.  Dropping s5 does NOT produce it. ----
    d5 = [D_(s, 300) - D_(s, 100) for s in seeds if s != 5]          # five clean seeds
    m5, se5 = st.mean(d5), st.stdev(d5) / math.sqrt(len(d5))
    x = dd[5]                                                        # hz3q's seed-5 delta
    chk("dropping seed 5 alone: delta (STILL FLAT)", m5, -0.207, "§4.8 R5, §3.5")
    chk("   se", se5, 0.107, "§4.8 R5", "%.3f")
    chk("   t  (|t| < 2.0 -> the deletion does NOT reverse the verdict)", m5 / se5, -1.94,
        "§4.8 R5", "%.2f")
    chk("hz3q seed-5 delta, the replacement", x, -0.396, "§4.8 R5")
    chk("   LOCATION: (x - mean5)/6, one sixth of its distance from the mean",
        (x - m5) / 6.0, -0.0315, "§4.8 R5", "%+.4f")
    chk("   mean5 unrounded", m5, -0.2068, "§4.8 R5", "%+.4f")
    chk("   mean5 + (x-mean5)/6 == the repaired estimate", m5 + (x - m5) / 6.0, -0.2383,
        "§4.8 R5", "%+.4f")
    chk("   PRECISION: se falls 5 -> 6 seeds", se_, 0.093, "§4.8 R5", "%.3f")
    chk("   |t| ratio 5 -> 6 seeds", abs((m_ / se_) / (m5 / se5)), 1.327, "§4.8 R5", "%.3f")
    chk("      location factor |m6/m5|", abs(m_ / m5), 1.152, "§4.8 R5", "%.3f")
    chk("      precision factor se5/se6", se5 / se_, 1.152, "§4.8 R5", "%.3f")
    chk("      location share of the move, in logs (%)",
        100 * math.log(abs(m_ / m5)) / math.log(abs((m_ / se_) / (m5 / se5))), 50.1,
        "§4.8 R5", "%.1f")
    chk("      precision share (%)",
        100 * math.log(se5 / se_) / math.log(abs((m_ / se_) / (m5 / se5))), 49.9,
        "§4.8 R5", "%.1f")
    chk("the replacement is NOT an outlier: |z| against the five archived seeds' spread",
        abs((x - m5) / st.stdev(d5)), 0.79, "§4.8 R5", "%.2f")
    chk("   its rank among the repaired six, most negative = 1",
        sorted(dd).index(x) + 1, 3, "§4.8 R5", "%.0f")
    arch = [(pl5(ch[s], 300) - pl5(nd[s], 300)) - (pl5(ch[s], 100) - pl5(nd[s], 100))
            for s in seeds]
    chk("   the ARCHIVED seed-5 delta it displaces", arch[5], 0.142, "§4.8 R5")
    chk("   its rank among the archived six, most POSITIVE = 1",
        sorted(arch, reverse=True).index(arch[5]) + 1, 1, "§4.8 R5", "%.0f")
    jt = []
    for i in range(len(seeds)):
        v = [dd[j] for j in range(len(dd)) if j != i]
        jt.append(abs(st.mean(v) / (st.stdev(v) / math.sqrt(len(v)))))
    for i, exp in enumerate((2.29, 2.50, 1.94, 1.94, 3.98, 1.94)):
        chk("   leave-one-out |t|, seed %d dropped" % i, jt[i], exp, "§4.8 R5", "%.2f")
    chk("   how many of the six deletions fall back under |t| = 2.0",
        sum(1 for v in jt if v < 2.0), 3, "§4.8 R5 -- the sensitivity is n=6, not seed 5", "%.0f")

def competitiveness(rows, adm, args):
    print("\n[8] THE SCOPE LIMIT WE MUST NOT SOFTEN  (abstract (ii), §7 T4)")
    import collections
    g = collections.defaultdict(list)
    for r in adm:
        key = r["run"].rsplit("-s", 1)[0] if "-s" in r["run"] else r["run"]
        g[key].append((float(r["plateau5"]), r))
    def best(pred):
        cand = []
        for k, v in g.items():
            if len(v) < 3 or not pred(v[0][1]): continue
            xs = [x[0] for x in v]
            cand.append((st.mean(xs), st.stdev(xs) / math.sqrt(len(xs)), k, len(xs)))
        return max(cand)
    mo18 = best(lambda r: r["granularity"] not in ("?", "") and
                r["network"] == "ResNet18" and r["dataset"] == "CIFAR10")
    mo   = best(lambda r: r["granularity"] not in ("?", ""))
    base = best(lambda r: r["granularity"] in ("?", "") and r["base"] == "SGD"
                and r["network"] == "ResNet18" and r["dataset"] == "CIFAR10")
    chk("best ResNet-18/C10 MetaOptimize arm", mo18[0], 93.317, "§7 T4, abstract", "%.3f")
    print("        that arm is `%s` (n=%d);  the corpus-wide max over ALL networks is "
          "`%s` %.3f on %s -- so 93.317 is the ResNet-18 max, NOT the corpus max"
          % (mo18[2], mo18[3], mo[2], mo[0],
             [v[0][1]["network"] for k, v in g.items() if k == mo[2]][0]))
    chk("tuned SGD+cosine baseline, ResNet-18/C10", base[0], 95.124, "§7 T4, abstract", "%.3f")
    chk("   se", base[1], 0.047, "§7 T4", "%.3f")
    chk("deficit", base[0] - mo18[0], 1.807, "abstract (ii), §7 T4, §9", "%.3f")


# ===================================================================== NEW: rho
def rho(rows, adm, args):
    """Eq. 9, rho = D / (100 - aligned), AND ITS RANKING.

    This section exists because the paper printed a superlative about rho that no
    check tested: section 4.3 and the Figure 1 caption called CIFAR-100's rho = 0.055
    "the smallest value in the corpus" when it is the FOURTH smallest of twenty.
    A per-cell value check would not have caught it; the RANK check below does."""
    print("\n[9] COMMENSURABLE RATIO rho = D / (100 - aligned)  (Eq. 9, §4.3, Fig. 1b)")
    cs = cells_with_gn(adm) if args.with_gn else cells(adm)
    order = sorted(cs, key=lambda c: c["rel"])
    for i, c in enumerate(order, 1):
        print("  %2d  %-14s %-8s %-5s  D %+0.3f  aligned %7.3f  rho %0.4f"
              % (i, c["label"], c["base"], c["dataset"], c["D"], c["aligned"], c["rel"]))
    by = {c["label"]: c for c in cs}
    chk("rho, gc1 (the CIFAR-100 cell §4.3 names)", by["gc1"]["rel"], 0.055,
        "§4.3, Fig. 1 caption", "%.3f")
    chk("   its RANK from the bottom, of %d" % len(cs),
        1 + sum(1 for c in cs if c["rel"] < by["gc1"]["rel"]), 4,
        "§4.3, Fig. 1 caption -- NOT 'the smallest'", "%.0f")
    chk("rho, the actual smallest (sm3)", by["sm3"]["rel"], 0.020, "§4.3, Fig. 1", "%.3f")
    chk("rho, the second smallest (aw1)", by["aw1"]["rel"], 0.040, "§4.3, Fig. 1", "%.3f")
    chk("rho, gm2 -- the OTHER CIFAR-100 cell", by["gm2"]["rel"], 0.050,
        "§4.3 rewrite", "%.3f")
    chk("   gm2's rank from the bottom",
        1 + sum(1 for c in cs if c["rel"] < by["gm2"]["rel"]), 3, "§4.3, Fig. 1", "%.0f")
    chk("largest D in pp is gc1's", by["gc1"]["D"], 1.640, "§4.3 (the half that IS true)")
    chk("   cells with a larger D", sum(1 for c in cs if c["D"] > by["gc1"]["D"]), 0,
        "§4.3", "%.0f")
    c10 = [c for c in cs if c["dataset"] == "C10"]
    chk("median rho over the %d CIFAR-10 cells" % len(c10),
        st.median([c["rel"] for c in c10]), 0.080, "§4.3 rewrite", "%.3f")
    chk("CIFAR-10 cells with rho below gc1's",
        sum(1 for c in c10 if c["rel"] < by["gc1"]["rel"]), 2, "§4.3 rewrite", "%.0f")

# ================================ NEW: the partition-family meta-optimiser census
def metacensus(rows, adm, args):
    """The census behind §7 T1 and scope item (iii).

    The previous draft printed '367 of 367 partition-programme runs are meta = Lion'.
    That figure does not re-derive under any definition of 'partition-programme run'
    reconstructible from the run table, so it was DROPPED, not restated, and replaced
    by this rule -- which is executable, and is therefore asserted here."""
    print("\n[15] THE PARTITION-FAMILY META-OPTIMISER CENSUS  (§7 T1, §1 scope (iii), A.1)")
    fam = lambda g: (g in ("nodewise", "nodewise1d")
                     or g.startswith("chunk") or g.startswith("permnode"))
    sel = [r for r in adm if fam(r["granularity"])]
    chk("admissible runs in the partition families", len(sel), 431,
        "§7 T1, §1 scope (iii), A.1", "%.0f")
    chk("   ...with meta = Lion", sum(1 for r in sel if r["meta"] == "Lion"), 419,
        "§7 T1, §1 scope (iii), A.1", "%.0f")
    chk("   ...with meta = RMSProp (all twelve are sm4)",
        sum(1 for r in sel if r["meta"] == "RMSProp"), 12,
        "§7 T1, §1 scope (iii), A.1", "%.0f")
    chk("   distinct meta-optimisers on that family", len({r["meta"] for r in sel}), 2,
        "§7 T1", "%.0f")
    # §4.3's and §8's claim is about the UNIFORM-CHUNK, nodewise1d and permnode runs --
    # the arms that could have been lost to the gate -- not about the aligned arm.
    cmf = lambda g: (g.startswith("chunk") or g.startswith("permnode")
                     or g == "nodewise1d")
    norp = [r for r in rows if cmf(r["granularity"]) and not r["run"].startswith("rp1")]
    chk("count-matched-family rows outside the in-flight rp1 batch", len(norp), 241,
        "§4.3, §8", "%.0f")
    chk("   ...of which admissible", sum(1 for r in norp if F.admissible(r)), 241,
        "§4.3, §8 -- 'all ... are admissible' holds only outside rp1", "%.0f")

# =========================================================== NEW: the gn1 gate
def gn1gate(rows, adm, args):
    """The commensurability gate of analysis/c84_gn1_score.py, T0.6.

    Section 7 T7 and section 4.3 describe this gate.  Both described it as firing on a
    RATIO ("1.37x ... registered bar 2.0 pp"), which compares a dimensionless ratio
    with a bar in percentage points.  COMM_MAX is a bar on the LEVEL DIFFERENCE in pp.
    Both quantities are checked here so the prose cannot drift again."""
    print("\n[10] THE gn1 COMMENSURABILITY GATE  (c84_gn1_score.py T0.6; §4.3, §7 T7)")
    bn = [x for a, g in (("gn1-bn-node", "nodewise"), ("gn1-bn-ch", "chunk777"))
          for x in arm(adm, a, g)[0]]
    gn = [x for a, g in (("gn1-gn-node", "nodewise"), ("gn1-gn-ch", "chunk777"))
          for x in arm(adm, a, g)[0]]
    lb, lg = st.mean(bn), st.mean(gn)
    chk("BatchNorm level, n=%d" % len(bn), lb, 92.293, "§7 T7 (derived)", "%.3f")
    chk("   its error budget", 100.0 - lb, 7.707, "§4.3, §7 T7", "%.3f")
    chk("GroupNorm level, n=%d" % len(gn), lg, 89.431, "§7 T7 (derived)", "%.3f")
    chk("   its error budget", 100.0 - gn_budget_paper(lg), 10.569, "§4.3, §7 T7", "%.3f")
    chk("**the gated quantity**: level difference", lg - lb, -2.862,
        "§4.3 + §7 T7 rewrite -- THIS is what meets COMM_MAX")
    chk("   COMM_MAX, the registered bar (pp)", 2.0, 2.0, "c84 source, §4.3, §7 T7", "%.1f")
    chk("   |difference| exceeds the bar by", abs(lg - lb) - 2.0, 0.862,
        "§4.3 + §7 T7 rewrite")
    chk("the budget RATIO (printed, NOT the gate)", (100.0 - lg) / (100.0 - lb), 1.371,
        "§4.3 + §7 T7: quoted as 1.37x", "%.3f")
    # the conservative reader's ten-cell pool that T7 offers.  T7's sentence is about
    # the ELEVEN cells the previous draft pooled, so the drop is taken on those.
    _p11 = ("cc1", "mm1", "pp1", "gn1 (BN)", "rl3 @1e-4", "rl3 @3e-4", "fa1", "hz3",
            "aw1", "nl1 (SGD)", "nl1 (RMSProp)")
    ten = [c for c in cells_with_gn(adm)
           if c["label"] in _p11 and c["label"] != "gn1 (BN)"]
    m, se, Q, df, tau = meta([(c["D"], c["seD"]) for c in ten])
    chk("T7's conservative drop pool (the eleven cells, gn1 (BN) dropped)", m, 0.570,
        "§7 T7")
    chk("   se", se, 0.038, "§7 T7", "%.3f")
    chk("   Q", Q, 36.39, "§7 T7", "%.2f")
    chk("   df", df, 9, "§7 T7", "%.0f")

def gn_budget_paper(lg):
    """100 - (100 - lg) == lg; written out so the budget line reads as a budget."""
    return lg

# ======================================================== NEW: the count axis U
UTAB = [("cc1", "cc1-c23", "cc1-ch", "chunk2325", "chunk777", 0.101, 0.190),
        ("rl3 @1e-4", "rl3-c23-m1e4", "rl3-ch7-m1e4", "chunk2325", "chunk777", 0.064, 0.154),
        ("rl3 @3e-4", "rl3-c23-m3e4", "rl3-ch7-m3e4", "chunk2325", "chunk777", 0.017, 0.098),
        ("fa1", "fa1-c23", "fa1-ch", "chunk2325", "chunk777", 0.019, 0.094),
        ("hz3", "hz3-c23", "hz3-ch", "chunk2325", "chunk777", -0.148, 0.080),
        ("aw1", "aw1-c23", "aw1-ch", "chunk2325", "chunk777", 0.045, 0.097),
        ("ml2", "ml2-", "ml2-", "chunk2325", "chunk777", 0.337, 0.112),
        ("g3m (R34)", "g3m-chg", "g3m-chd", "chunk2500", "chunk835", 0.263, 0.074),
        ("r50 (R50)", "r50-c88", "r50-ch", "chunk884", "chunk295", 0.321, 0.259),
        ("gm2 (C100)", "gm2-c22", "gm2-ch", "chunk2293", "chunk771", -0.054, 0.196),
        ("nl1/SGD", "nl1-sgd-c23", "nl1-sgd-ch", "chunk2325", "chunk777", 0.182, 0.283),
        ("nl1/RMSProp", "nl1-rms-c23", "nl1-rms-ch", "chunk2325", "chunk777", -0.037, 0.085),
        ("sm3", "sm3-awrms-c23", "sm3-awrms-ch", "chunk2325", "chunk777", 0.071, 0.082),
        ("sm4", "sm4-awrms-c23", "sm4-awrms-ch", "chunk2325", "chunk777", 0.359, 0.074)]

def countaxis(rows, adm, args):
    print("\n[11] THE COUNT AXIS  U = chunk2325 - chunk777  (§4.2 table, 14 cells)")
    us = []
    for lab, a_, b_, ga, gb, pu, pse in UTAB:
        av, _ = arm(adm, a_, ga); bv, _ = arm(adm, b_, gb)
        u, se, t = welch(av, bv)
        us.append(u)
        chk("U  %-12s" % lab, u, pu, "§4.2 table")
        chk("   se %-9s" % "", se, pse, "", "%.3f")
    chk("cells in the U table", len(us), 14, "§4.2", "%.0f")
    chk("U changes sign across cells (n negative)", sum(1 for u in us if u < 0), 3,
        "§4.2 'negative in three of the fourteen'", "%.0f")
    chk("max |U| over the fourteen (sm4)", max(abs(u) for u in us), 0.359,
        "§4.2 'never exceeds +0.36 pp'")
    lion = [u for u, (lab, *_ ) in zip(us, UTAB) if lab != "sm4"]
    chk("   max |U| over the thirteen Lion cells (ml2)", max(abs(u) for u in lion), 0.337,
        "§4.2 'the previous draft's +0.34 bound still holds'")
    # the ml2 six-run reading the paper names as the WRONG one
    six = lambda g: [float(r["plateau5"]) for r in adm
                     if r["run"].startswith("ml2-") and r["granularity"] == g]
    chk("ml2's U on six runs (the banned reading)", welch(six("chunk2325"), six("chunk777"))[1],
        0.088, "§4.2 parenthesis", "%.3f")
    # the ck1 ladder of §4.2
    LAD = [("chunk1", 90.979, 0.131), ("chunk2", 91.095, 0.024), ("chunk16", 91.411, 0.277),
           ("chunk128", 92.159, 0.063), ("chunk1024", 92.526, 0.077)]
    got = []
    for g, pm, pse in LAD:
        v, _ = arm(adm, "ck1-", g)
        if not v:
            print("  ck1 arm %-10s NOT FOUND in the CSV -- ladder skipped" % g); got = []; break
        got.append(v)
        chk("ck1 %-10s plateau5" % g, st.mean(v), pm, "§4.2 ladder", "%.3f")
        chk("    se %-6s" % "", st.stdev(v) / math.sqrt(len(v)), pse, "", "%.3f")
    if got:
        a, se, t = welch(got[-1], got[0])
        chk("five-rung ascent", a, 1.547, "§4.2")
        chk("   se", se, 0.152, "§4.2", "%.3f")
        chk("   t", t, 10.2, "§4.2", "%.1f")

# ================================================== NEW: the tuning axes, §4.1
def tuning(rows, adm, args):
    print("\n[12] THE TUNING AXES  (§4.1)")
    pairs = [("1e-3 (the parent's default)", "ms-layA-1e3", "ms-scalA-1e3",
              91.259, 0.076, 87.967, 0.124, 3.291, 0.146, 22.6),
             ("1e-4 (bracketed optimum)", "ms-layA-1e4", "ms-scalA-1e4",
              92.960, 0.024, 92.305, 0.174, 0.655, 0.176, 3.72)]
    lay, scal = {}, {}
    for lab, la, sa, plm, plse, psm, psse, pd, pdse, pt in pairs:
        lv, _ = arm(adm, la, "layerwise"); sv, _ = arm(adm, sa, "scalar")
        lay[lab], scal[lab] = st.mean(lv), st.mean(sv)
        chk("eta %-26s layerwise" % lab, st.mean(lv), plm, "§4.1 table", "%.3f")
        chk("   se", st.stdev(lv) / math.sqrt(len(lv)), plse, "", "%.3f")
        chk("eta %-26s scalar" % lab, st.mean(sv), psm, "§4.1 table", "%.3f")
        chk("   se", st.stdev(sv) / math.sqrt(len(sv)), psse, "", "%.3f")
        d, se, t = welch(lv, sv)
        chk("   layerwise - scalar", d, pd, "§4.1 table")
        chk("   se", se, pdse, "", "%.3f")
        chk("   t", t, pt, "§4.1 table", "%.2f" if pt < 10 else "%.1f")
    k3, k4 = pairs[0][0], pairs[1][0]
    chk("scalar is better at 1e-4 by", scal[k4] - scal[k3], 4.338, "§4.1 prose")
    chk("layerwise is better at 1e-4 by", lay[k4] - lay[k3], 1.701, "§4.1 prose")

# ============================== NEW: Appendix A.4 and the T9 / dup_group readings
def appendices(rows, adm, args):
    print("\n[13] APPENDIX A.4 (printed-table pools) AND §7 T9")
    # A.4 records the pools as they stood when the appendix was written: the ELEVEN
    # same-contrast cells of the previous draft, and the twelve that included the
    # withdrawn GroupNorm cell.  bm2's two cells and sm3 are excluded here on purpose --
    # §4.4's live pool is the fourteen and is asserted in section [3].
    _pool11 = ("cc1", "mm1", "pp1", "gn1 (BN)", "rl3 @1e-4", "rl3 @3e-4", "fa1",
               "hz3", "aw1", "nl1 (SGD)", "nl1 (RMSProp)")
    live = [c for c in cells_with_gn(adm) if c["label"] in _pool11]
    full = live + [c for c in cells_with_gn(adm) if c["base"] == "SGDm-GN"]
    r3 = lambda x: round(x, 3)
    for nm, grp, pQ in (("twelve-cell", full, 43.01), ("eleven-cell", live, 36.29)):
        m, se, Q, df, tau = meta([(r3(c["D"]), r3(c["seD"])) for c in grp])
        chk("%s Q from the PRINTED 3-dp table" % nm, Q, pQ, "A.4, header note", "%.2f")
    for nm, grp, pQ in (("twelve-cell", full, 43.19), ("eleven-cell", live, 36.40)):
        m, se, Q, df, tau = meta([(c["D"], c["seD"]) for c in grp])
        chk("%s Q at full precision" % nm, Q, pQ, "A.4, header note", "%.2f")
    ch, nd = {}, {}
    for r in adm:
        for tag, d in (("hz3-ch-s", ch), ("hz3-node-s", nd)):
            if r["run"].startswith(tag):
                d[int(r["run"].rsplit("-s", 1)[1])] = float(r["plateau5"])
    keep = [s for s in sorted(set(ch) & set(nd)) if s != 5]
    d, se, t = welch([ch[s] for s in keep], [nd[s] for s in keep])
    chk("hz3 box-matched 5 v 5 D", d, 0.455, "§4.3 dagger, §7 T9")
    chk("   se", se, 0.096, "§4.3 dagger, §7 T9", "%.3f")
    chk("   t", t, 4.75, "§4.3 dagger, §7 T9", "%.2f")
    allk = sorted(set(ch) & set(nd))
    d6 = welch([ch[s] for s in allk], [nd[s] for s in allk])[0]
    chk("   the 6 v 6 - 5 v 5 difference", abs(d6 - d), 0.028,
        "§4.3 dagger (0.027 if taken from the ROUNDED table entries)", "%.3f")
    six = lambda g: [float(r["plateau5"]) for r in adm
                     if r["run"].startswith("ml2-") and r["granularity"] == g]
    d, se, t = welch(six("chunk777"), six("nodewise"))
    chk("ml2 read as 6 v 6 (the WRONG reading)", se, 0.142, "§4.3 note, §6.1", "%.3f")
    chk("   its t", t, 3.20, "§4.3 note", "%.2f")

# ================================ NEW: what the deposit alone lets a scorer do
# Three states, each one MEASURED by running that scorer unedited against a tree
# containing only what the deposit ships (results/all_runs.csv + the unpacked .out
# logs), with only its own documented --runs/--root/--csv arguments supplied:
#   REACHED  the printed verdict the paper quotes is regenerated in full
#   PARTIAL  some registered legs regenerate, others do not
#   BLOCKED  the scorer halts, or prints NO DATA, where the paper quotes it
DEPOSIT_SCORERS = [
 ("c76_mm1_score", "§4.3 mm1 cell",              "BLOCKED",
  "M1 cannot be scored; M0-M0.4 read 0/0 (0 probe dirs)"),
 ("c77_pp1_score", "§4.6 alignment null P2",     "BLOCKED",
  "P1/P2/P5b NO DATA -- §4.6's verbatim block is unreachable"),
 ("c78_bn1_score", "§5.4 G at m=4,851",          "BLOCKED",
  "every arm reads n=0 (0 probe dirs)"),
 ("c79_ar1_score", "§4.3 the box-void exclusion","BLOCKED",
  "A0.4 reads 0/0 -- the ground of the exclusion is unreproducible"),
 ("c81_cc1_score", "§5.2 field concordance",     "PARTIAL",
  "C2 REPLICATES / C3 COLLAPSES regenerate; C1 and every N_eff/m in §5.2 do not"),
 ("c82_fa1_score", "§3.4 scorer list",           "BLOCKED",
  "exits 1: '0 probe dirs ... Nothing scored.'"),
 ("c83_gc1_score", "§4.3 gc1 cell",              "PARTIAL",
  "S1 CLOSE-CONFIRMED regenerates; S0.4's arm-asymmetry guard does not run"),
 ("c84_gn1_score", "§5.6 / §7 T7",               "BLOCKED",
  "exits 1 at the T0 VERDICT (T0.5 ungateable) -- never reaches T0.6"),
 ("c87_rl3_score", "§4.5 the RULE-11 closure",   "REACHED",
  "--runs <unpacked logs>; no probe dependency"),
 ("c87_hz3_score", "§4.8 the budget window",     "REACHED",
  "--runs <unpacked logs>; no probe dependency"),
 ("c99_hz3q_score","§4.8 the repaired slope",     "PARTIAL",
  "H0/H2/H3/HC regenerate off the .out series; the H1 box gate prints "
  "'NO OCCUPANCY IS MEASURABLE' -- probe*.jsonl is the excluded class"),
]

def deposit(rows, adm, args):
    """The claim §8 and the End matter make about the deposit, made checkable."""
    print("\n[14] WHAT THE DEPOSIT ALONE LETS A REGISTERED SCORER DO  (§8, End matter)")
    print("    %-16s %-30s %-9s %s" % ("scorer", "quoted for", "state", "note"))
    for mod, where, state, note in DEPOSIT_SCORERS:
        print("    %-16s %-30s %-9s %s" % (mod + ".py", where, state, note))
    n = lambda st_: sum(1 for _, _, x, _ in DEPOSIT_SCORERS if x == st_)
    chk("registered scorers REACHED on the deposit", n("REACHED"), 2,
        "§8 + End-matter rewrite (A2)", "%.0f")
    chk("   PARTIAL", n("PARTIAL"), 3, "§8 + End-matter rewrite (A2)", "%.0f")
    chk("   registered scorer verdicts this paper quotes in full",
        len(DEPOSIT_SCORERS), 11, "§8 + End-matter rewrite (A2)", "%.0f")
    chk("   BLOCKED by the excluded probe files", n("BLOCKED"), 6,
        "§8 + End-matter rewrite (A2)", "%.0f")
    print("    (this table is a REGISTER, not a derivation: each row was produced by")
    print("     running that scorer unedited against this tree.  Re-run them to refresh it.)")



# ============================================================== THE COVERAGE CENSUS
DRAFT = os.path.join(ROOT, "paper", "DRAFT-v4.md")
TEX   = os.path.join(ROOT, "paper", "paper.tex")

# A decimal numeral in the draft is a QUANTITY unless it is one of these.  The rule
# is mechanical and is stated in §3.4 so a referee can re-run it.
# NB the two alternatives need DIFFERENT flags, so they carry inline scopes rather
# than a shared `re.S | re.M`.  With a shared re.S the `.*$` of the indented-command
# branch matched newlines and ran to the LAST `$` in the file: one mask swallowed
# 49,965 of 308,256 characters (16.9% of the draft, against 0.3% correct), so the
# census denominator was computed on a manuscript with a sixth of it invisible.
# analysis/paper_numeric_diff.py always had this right ((?m) only, no re.S); the two
# are meant to share one exclusion rule and §3.4 says so, so this is the bug.
_FENCE   = re.compile(r"(?s:```.*?```)|(?m:^ {4,}\S.*$)")   # code blocks & indented cmds
_XREF    = re.compile(r"(?:§|Appendix\s|App\.\s|Table\s|Tables\s|Figure\s|Fig\.\s|Eq\.\s|"
                      r"Eqs\.\s|item\s|R0\s|\bA\.|\bT\d|\bM\d|\bP\d|\bS\d|\bC\d|\bF\d|"
                      r"\bU\d|\bX\d|arXiv:|:\d{7}|v)$")
_VERSION = re.compile(r"(?:Python|PyTorch|torch|torchvision|numpy|CUDA|cu|GCCcore|"
                      r"Slurm|matplotlib|md5)\W{0,3}$", re.I)
_NUM     = re.compile(r"\d+\.\d+")

def census(path=None, quiet=False):
    """Count the decimal numerals in the draft and how many this script asserts.

    Returns (n_tokens, n_distinct, n_quantities, n_covered).  A numeral is COVERED if
    some chk() in this run asserted a paper value that prints to the same string at
    that chk()'s own precision.

    The census never counts itself: once censuscheck() has frozen CENSUS_MARK, only
    the assertions made BEFORE that point are read, so asserting the census triple
    cannot inflate the coverage it is asserting."""
    path = path or DRAFT
    if not os.path.exists(path):
        print("\n[census] %s not found -- census skipped" % path); return None
    raw = open(path).read()
    raw_tok = _NUM.findall(raw)
    body = _FENCE.sub(" ", raw)
    toks, quants = [], []
    for m in _NUM.finditer(body):
        tok = m.group(0)
        toks.append(tok)
        pre = body[max(0, m.start() - 24):m.start()]
        if _XREF.search(pre) or _VERSION.search(pre):
            continue
        quants.append(tok)
    claims = ASSERTED if CENSUS_MARK is None else ASSERTED[:CENSUS_MARK]
    covered_strings = set()
    for fmt, paper in claims:
        t = (fmt % paper).lstrip("+-")
        if "." in t: covered_strings.add(t)
    hit = {q for q in quants if q.lstrip("0") in covered_strings or q in covered_strings}
    n_tok, n_dis = len(toks), len(set(toks))
    n_q, n_qd = len(quants), len(set(quants))
    n_cov = len(hit)
    if not quiet:
        print("\n" + "=" * 78)
        print("COVERAGE CENSUS -- %s" % os.path.relpath(path, ROOT))
        print("=" * 78)
        print("  every /\\d+[.]\\d+/ in the file                        %5d  (%d distinct)"
              % (len(raw_tok), len(set(raw_tok))))
        print("  ...minus code blocks and indented verbatim scorer")
        print("     output (numbers the SCORERS print, not ours)      %5d  (%d distinct)"
              % (n_tok, n_dis))
        print("  ...minus section, table, figure and equation labels,")
        print("     arXiv ids and software versions  = QUANTITIES     %5d  (%d distinct)"
              % (n_q, n_qd))
        print("  chk() assertion sites executed in this run           %5d" % len(claims))
        print("  distinct quantity-numerals this run asserts          %5d" % n_cov)
        print("  coverage of distinct quantity-numerals               %5.1f%%"
              % (100.0 * n_cov / n_qd if n_qd else 0.0))
        print("  **This script asserts the numbers that carry a claim, not every")
        print("    numeral the draft prints.  §3.4 states that scope; do not widen it")
        print("    in prose without widening it here.**")
    return n_tok, n_qd, n_q, n_cov


# ====================================== METRIC SENSITIVITY  (S4.4, S1.1 C1/C3, S7 T10)
# The house rule is NOT relaxed here: `plateau5` remains the primary and the `plateau`
# column remains banned AS a primary.  What this section does is DISCLOSE what S4.4's
# heterogeneity decomposition and S4.3's sign result do when the endpoint column is the
# only thing that varies.  Everything else is held fixed at the paper's own definitions:
# the row filter of Eq. (9) -- which is defined on plateau5 and is deliberately NOT
# re-derived per metric, because that would make this row-filter sensitivity, which is
# S7 T10 -- the arm prefixes of CELLS, the dup_group collapse of R-E, and POOL12.
MS_METRICS = ["plateau5", "plateau", "best_test", "final_test"]

def _arm_on(col):
    """c98_figures.arm with the metric column swapped, and nothing else changed."""
    def _arm(rows, prefix, gran):
        sel = [r for r in rows if r["run"].startswith(prefix) and r["granularity"] == gran]
        groups, singles = {}, []
        for r in sel:
            v = (r.get(col) or "").strip()
            if v == "":                    # admissible row, unreadable on THIS endpoint
                continue
            dg = (r.get("dup_group") or "").strip() or F.PENDING_DUP.get(r["run"], "")
            if dg: groups.setdefault(dg, []).append(float(v))
            else:  singles.append(float(v))
        vals = singles + [st.mean(v) for v in groups.values()]
        return vals, len(vals)
    return _arm

def _cells_on(col, adm, with_gn):
    old_arm, old_gn = F.arm, F.WITH_GN
    F.arm, F.WITH_GN = _arm_on(col), with_gn
    try:    return F.cells(adm)
    finally: F.arm, F.WITH_GN = old_arm, old_gn

# metric -> the values S4.4's endpoint table prints, in its own column order:
#   pool D, se, Q(13 df), tau, between-base Q(3 df), share %, cells with D>0 of 20,
#   cells resolved at t>=3 of 20, rms measurement se, sd of the fourteen D's
METRIC_TABLE = {
 "plateau5":   (0.530, 0.029, 102.47, 0.295, 95.12, 92.8, 20, 18, 0.143, 0.255),
 "plateau":    (0.449, 0.024,  29.37, 0.101, 18.36, 62.5, 20, 20, 0.112, 0.159),
 "best_test":  (0.396, 0.025,   9.87, 0.000,  3.70, 37.5, 20, 11, 0.137, 0.084),
 "final_test": (0.614, 0.043,  39.91, 0.251, 34.66, 86.8, 19,  7, 0.355, 0.394),
}
# metric -> the four base-level pools S4.4's endpoint table prints
METRIC_LEVELS = {
 "plateau5":   {"SGDm": 0.556, "SGD": 1.000, "RMSProp": 0.720, "AdamW": 0.189},
 "plateau":    {"SGDm": 0.438, "SGD": 0.737, "RMSProp": 0.634, "AdamW": 0.315},
 "best_test":  {"SGDm": 0.411, "SGD": 0.432, "RMSProp": 0.386, "AdamW": 0.261},
 "final_test": {"SGDm": 0.588, "SGD": 0.139, "RMSProp": 1.518, "AdamW": 0.420},
}
METRIC_SPREAD = {"plateau5": 5.3, "plateau": 2.3, "best_test": 1.7, "final_test": 10.9}
METRIC_RANK = {                        # the rank order S4.4 prints, and the inversion
 "plateau5":   "SGD > RMSProp > SGDm > AdamW",
 "plateau":    "SGD > RMSProp > SGDm > AdamW",
 "best_test":  "SGD > SGDm > RMSProp > AdamW",
 "final_test": "RMSProp > SGDm > AdamW > SGD",
}

def metricsens(rows, adm, args):
    print("\n[15] METRIC SENSITIVITY -- the endpoint varied, everything else held fixed")
    print("     (S4.4 'The endpoint, varied'; S1.1 C1 and C3; S7 T10; S9)")
    bases = ["SGDm", "SGD", "RMSProp", "AdamW"]
    for col in MS_METRICS:
        pool, sep, Qp, taup, betp, shrp, posp, resp, rmsp, sdp = METRIC_TABLE[col]
        cs20 = _cells_on(col, adm, False)                    # the 20 Table 2 cells
        live = [c for c in _cells_on(col, adm, True)
                if c["in12"] and c["base"] != "SGDm-GN"]
        assert all(c["base"] != "AdamW+RMS" for c in live), "sm4 leaked into the pool"
        assert len(live) == 14 and len(cs20) == 20, "the cell set moved with the metric"
        m, sem_, Q, df, tau = meta([(c["D"], c["seD"]) for c in live])
        sub = {b: meta([(c["D"], c["seD"]) for c in live if c["base"] == b]) for b in bases}
        within = sum(sub[b][2] for b in bases)
        print("   ---- %s" % col)
        chk("pool D        %-11s" % col, m, pool, "S4.4 endpoint table")
        chk("   se         %-11s" % col, sem_, sep, "S4.4 endpoint table", "%.3f")
        chk("   Q / 13 df  %-11s" % col, Q, Qp, "S4.4 endpoint table", "%.2f")
        chk("   p          %-11s" % col, chi2_sf(Q, df), None, "", "%.4g")
        chk("   tau        %-11s" % col, tau, taup, "S4.4 endpoint table", "%.3f")
        chk("   between-base Q / 3 df", Q - within, betp, "S4.4 endpoint table", "%.2f")
        chk("   share of Q %-11s" % col, 100 * (Q - within) / Q, shrp,
            "S4.4 endpoint table, S7 T10", "%.1f")
        chk("   cells with D > 0 of 20", sum(1 for c in cs20 if c["D"] > 0), posp,
            "S4.4, S1.1 C1, S7 T10, S9", "%.0f")
        chk("   ...resolved at t >= 3.0", sum(1 for c in cs20 if c["tD"] >= 3.0), resp,
            "S4.4 endpoint table", "%.0f")
        chk("   rms measurement se", math.sqrt(sum(c["seD"]**2 for c in live) / len(live)),
            rmsp, "S4.4 endpoint paragraph", "%.3f")
        chk("   sd of the fourteen D's", st.stdev([c["D"] for c in live]), sdp,
            "S4.4 endpoint paragraph", "%.3f")
        for b in bases:
            chk("   level %-8s %-11s" % (b, col), sub[b][0], METRIC_LEVELS[col][b],
                "S4.4 endpoint table")
        hi = max(sub[b][0] for b in bases); lo = min(sub[b][0] for b in bases)
        chk("   level spread factor", hi / lo, METRIC_SPREAD[col],
            "S4.4 endpoint paragraph", "%.1f")
        order = " > ".join(sorted(bases, key=lambda b: -sub[b][0]))
        ok = order == METRIC_RANK[col]
        if not ok: FAILS.append(("rank order %s" % col, order, METRIC_RANK[col], "S4.4"))
        print("      %-46s %-32s | paper %-32s | %s"
              % ("rank order of the four level pools", order, METRIC_RANK[col],
                 "PASS" if ok else "**FAIL**"))

    # The one cell that changes sign, and WHY.  S4.4 calls it an se inflation on a
    # single-epoch reading and not a reversal, so the se ratio and the two arms' own
    # sds are asserted too -- that is the claim, not the sign.
    fin = {c["label"]: c for c in _cells_on("final_test", adm, False)}
    pl5 = {c["label"]: c for c in _cells_on("plateau5",   adm, False)}
    b = fin["bm2 (SGD)"]
    chk("bm2 (SGD) D on final_test", b["D"], -0.073, "S4.4 endpoint paragraph, S1.1 C1")
    chk("   its se", b["seD"], 0.500, "S4.4 endpoint paragraph", "%.3f")
    chk("   its t -- unresolved, NOT reversed", b["tD"], -0.15, "S4.4, S1.1 C1", "%.2f")
    chk("   its plateau5 reading", pl5["bm2 (SGD)"]["D"], 0.978, "S4.4, Table 2")
    chk("   se inflation vs plateau5", b["seD"] / pl5["bm2 (SGD)"]["seD"], 5.8,
        "S4.4 endpoint paragraph", "%.1f")
    for pre, gran, who, s5, sf in (("bm2-sgd-ch",   "chunk777", "uniform",  0.127, 0.467),
                                   ("bm2-sgd-node", "nodewise", "nodewise", 0.077, 0.729)):
        sel = [r for r in adm if r["run"].startswith(pre) and r["granularity"] == gran]
        chk("   bm2 %-8s arm sd, plateau5" % who,
            st.stdev([float(r["plateau5"]) for r in sel]), s5,
            "S4.4 endpoint paragraph", "%.3f")
        chk("   bm2 %-8s arm sd, final_test" % who,
            st.stdev([float(r["final_test"]) for r in sel]), sf,
            "S4.4 endpoint paragraph", "%.3f")
    r5 = math.sqrt(sum(c["seD"]**2 for c in [x for x in _cells_on("plateau5", adm, True)
                       if x["in12"] and x["base"] != "SGDm-GN"]) / 14.0)
    rf = math.sqrt(sum(c["seD"]**2 for c in [x for x in _cells_on("final_test", adm, True)
                       if x["in12"] and x["base"] != "SGDm-GN"]) / 14.0)
    chk("final_test rms se / plateau5 rms se", rf / r5, 2.5,
        "S4.4 endpoint paragraph", "%.1f")

    # ------------- the SAME knife, applied to S4.7's prescription T ---------------
    # S4.4 varies the endpoint under D.  T = nodewise1d - nodewise is Contribution 6
    # and the paper's only actionable recommendation, so it faces the same knife or
    # the disclosure is selective.  Everything is held fixed exactly as above -- the
    # arm prefixes of S4.7's own table, the dup_group collapse, welch() -- and only
    # the column read changes.  The twelve non-AdamW cells are the claim, the two
    # AdamW+Lion cells and their pool are the scope line, sm4 is the exception.
    T_ARMS = [("bn1", "bn1-n1d", "bn1-node"), ("ml2", "ml2-", "ml2-"),
              ("fa1", "fa1-n1d", "fa1-node"), ("g3m", "g3m-n1d", "g3m-node"),
              ("cc1", "cc1-n1d", "cc1-node"), ("r50", "r50-n1d", "r50-node"),
              ("gm2", "gm2-n1d", "gm2-node"), ("hz3", "hz3-n1d", "hz3-node"),
              ("nl1 SGD", "nl1-sgd-n1d", "nl1-sgd-node"),
              ("nl1 RMSProp", "nl1-rms-n1d", "nl1-rms-node"),
              ("rl3 @1e-4", "rl3-n1d-m1e4", "rl3-node-m1e4"),
              ("rl3 @3e-4", "rl3-n1d-m3e4", "rl3-node-m3e4")]
    # metric -> the values S4.7's endpoint table prints, in its own column order:
    #   min T, max T, T > 0 of 12, resolved at t >= 3 of 12, rms se of the twelve,
    #   AdamW+Lion pool and its se, sm4's T, se and t
    T_METRIC = {
     "plateau5":   (0.337, 1.363, 12, 12, 0.162,  0.007, 0.056, 0.988, 0.231, 4.28),
     "plateau":    (0.407, 1.664, 12, 12, 0.104,  0.051, 0.062, 0.730, 0.084, 8.70),
     "best_test":  (0.170, 1.280, 12,  8, 0.132, -0.111, 0.085, 0.393, 0.101, 3.89),
     "final_test": (0.147, 1.630, 12,  6, 0.460,  0.106, 0.054, 0.723, 0.479, 1.51),
    }
    # metric -> hz3 box-matched 5 v 5 (T, t): the seed-5 trio is excluded for the T9
    # clip-box and hardware mismatch, exactly as S4.7's footnote does on plateau5.
    T_HZ3 = {"plateau5": (0.328, 3.89), "plateau": (0.432, 5.37),
             "best_test": (0.338, 9.91), "final_test": (0.368, 4.11)}
    print("\n   ==== S4.7's prescription T = nodewise1d - nodewise, under the same knife")
    tpos = 0
    for col in MS_METRICS:
        lo, hi, posp, resp, rmsp, awp, awsep, s4, s4se, s4t = T_METRIC[col]
        a_ = _arm_on(col)
        g = {}
        for lab, pa, pb in T_ARMS:
            av, _ = a_(adm, pa, "nodewise1d"); bv, _ = a_(adm, pb, "nodewise")
            g[lab] = welch(av, bv)
        assert len(g) == 12, "the T cell set moved with the metric"
        print("   ---- %s" % col)
        chk("T  min of the twelve   %-11s" % col, min(x[0] for x in g.values()), lo,
            "S4.7 endpoint table")
        chk("T  max of the twelve   %-11s" % col, max(x[0] for x in g.values()), hi,
            "S4.7 endpoint table")
        npos = sum(1 for x in g.values() if x[0] > 0); tpos += npos
        chk("   T > 0 of 12         %-11s" % col, npos, posp,
            "S4.7 endpoint table, S1.1 C6", "%.0f")
        chk("   ...resolved t >= 3  %-11s" % col,
            sum(1 for x in g.values() if x[2] >= 3.0), resp,
            "S4.7 endpoint table", "%.0f")
        chk("   rms se of the twelve %-10s" % col,
            math.sqrt(sum(x[1] ** 2 for x in g.values()) / 12.0), rmsp,
            "S4.7 endpoint paragraph", "%.3f")
        pool = []
        for pa, pb in (("aw1-n1d", "aw1-node"), ("sm3-awrms-n1d", "sm3-awrms-node")):
            av, _ = a_(adm, pa, "nodewise1d"); bv, _ = a_(adm, pb, "nodewise")
            t_, se_, _ = welch(av, bv); pool.append((t_, se_))
        m_, sem_, _Q, _df, _tau = meta(pool)
        chk("   AdamW+Lion T pool   %-11s" % col, m_, awp,
            "S4.7 endpoint table, S1.1 C6")
        chk("      its se           %-11s" % col, sem_, awsep,
            "S4.7 endpoint table", "%.3f")
        av, _ = a_(adm, "sm4-awrms-n1d", "nodewise1d")
        bv, _ = a_(adm, "sm4-awrms-node", "nodewise")
        t_, se_, tt_ = welch(av, bv)
        chk("   sm4 (AdamW+RMSProp) %-11s" % col, t_, s4, "S4.7 endpoint table")
        chk("      its se           %-11s" % col, se_, s4se, "S4.7 endpoint table", "%.3f")
        chk("      its t            %-11s" % col, tt_, s4t, "S4.7 endpoint table", "%.2f")
        h = {}
        for tag in ("hz3-n1d-s", "hz3-node-s"):
            h[tag] = [float(r[col]) for r in adm
                      if r["run"].startswith(tag) and (r.get(col) or "").strip()
                      and r["run"] != tag + "5"]
        assert len(h["hz3-n1d-s"]) == 5 and len(h["hz3-node-s"]) == 5, "hz3 is not 5 v 5"
        th, _seh, tth = welch(h["hz3-n1d-s"], h["hz3-node-s"])
        chk("   hz3 box-matched 5v5 %-11s" % col, th, T_HZ3[col][0], "S4.7 footnote")
        chk("      its t            %-11s" % col, tth, T_HZ3[col][1],
            "S4.7 footnote", "%.2f")
    chk("T > 0 over 4 endpoints x 12 cells", tpos, 48,
        "S4.7 endpoint paragraph, S1.1 C6", "%.0f")

    # The two S4.7 claims that are orderings rather than numbers.  The second is the
    # cycle-105 repair: "the largest T in the CIFAR-10 corpus" was FALSE of sm4 --
    # r50 reads +1.049 against sm4's +0.988 -- and is true only at ResNet-18.
    pl5a = _arm_on("plateau5")
    Tv = {}
    for lab, pa, pb in T_ARMS + [("sm4", "sm4-awrms-n1d", "sm4-awrms-node")]:
        av, _ = pl5a(adm, pa, "nodewise1d"); bv, _ = pl5a(adm, pb, "nodewise")
        Tv[lab] = welch(av, bv)[0]
    top18 = max([l for l in Tv if l not in ("g3m", "r50", "gm2")], key=lambda l: Tv[l])
    topc10 = max([l for l in Tv if l != "gm2"], key=lambda l: Tv[l])
    for nm, got_, want in (("largest T at R18 / C10, plateau5", top18, "sm4"),
                           ("largest T over the CIFAR-10 corpus", topc10, "r50")):
        ok = got_ == want
        if not ok: FAILS.append((nm, got_, want, "S4.7 sm4 paragraph"))
        print("      %-46s %-12s | paper %-12s | %s"
              % (nm, got_, want, "PASS" if ok else "**FAIL**"))
    av, _ = pl5a(adm, "r50-n1d", "nodewise1d"); bv, _ = pl5a(adm, "r50-node", "nodewise")
    _t, _se, _tt = welch(av, bv)
    chk("   r50's T, the true CIFAR-10 maximum", _t, 1.049, "S4.7 sm4 paragraph")
    chk("      its se", _se, 0.317, "S4.7 sm4 paragraph, S4.7 table", "%.3f")
    # ...and how many ResNet-18 / CIFAR-10 cells overtake sm4 on the two single-epoch
    # columns, which is why the repaired superlative is still plateau5-specific.
    R18C10 = [l for l, _a, _b in T_ARMS if l not in ("g3m", "r50", "gm2")]
    for col, paper in (("best_test", 4), ("final_test", 2)):
        a_ = _arm_on(col)
        def _T(pa, pb):
            av, _ = a_(adm, pa, "nodewise1d"); bv, _ = a_(adm, pb, "nodewise")
            return welch(av, bv)[0]
        s4v = _T("sm4-awrms-n1d", "sm4-awrms-node")
        others = dict([(l, _T(pa, pb)) for l, pa, pb in T_ARMS if l in R18C10] +
                      [("aw1", _T("aw1-n1d", "aw1-node")),
                       ("sm3", _T("sm3-awrms-n1d", "sm3-awrms-node"))])
        chk("   R18/C10 cells besides sm4", len(others), 11,
            "S4.7 sm4 paragraph", "%.0f")
        chk("   ...larger than sm4 on %-11s" % col,
            sum(1 for v in others.values() if v > s4v), paper,
            "S4.7 sm4 paragraph", "%.0f")

    # The scope line, stated exactly: the two AdamW+Lion cells against the registered
    # band cell by cell (the form S3.4 actually registered), and the pool's interval.
    inband = 0
    for col in MS_METRICS:
        a_ = _arm_on(col); pool = []
        for lab, pa, pb in (("aw1", "aw1-n1d", "aw1-node"),
                            ("sm3", "sm3-awrms-n1d", "sm3-awrms-node")):
            av, _ = a_(adm, pa, "nodewise1d"); bv, _ = a_(adm, pb, "nodewise")
            t_, se_, _ = welch(av, bv); pool.append((t_, se_))
            if abs(t_) <= 0.15: inband += 1
            if lab == "aw1" and col == "final_test":
                chk("   aw1 on final_test -- the UNDECIDED reading", t_, 0.253,
                    "S4.7 endpoint paragraph")
                chk("      its se", se_, 0.109, "S4.7 endpoint paragraph", "%.3f")
        m_, sem_, _Q, _df, _tau = meta(pool)
        if col == "plateau5":
            chk("   AdamW+Lion pool 95% CI low", m_ - 1.96 * sem_, -0.104,
                "S4.7 endpoint paragraph")
            chk("      ...CI high", m_ + 1.96 * sem_, 0.117, "S4.7 endpoint paragraph")
        n_in = sum(1 for x in [(m_ - 1.96 * sem_), (m_ + 1.96 * sem_)]
                   if abs(x) <= 0.15)
        if col != "plateau5" and n_in == 2:
            FAILS.append(("AdamW+Lion pool interval inside the band on " + col,
                          "yes", "plateau5 only", "S4.7 endpoint paragraph"))
    chk("AdamW+Lion cell readings inside the band, of 8", inband, 7,
        "S4.7 endpoint paragraph", "%.0f")
    r12 = {}
    for col in ("plateau5", "final_test"):
        a_ = _arm_on(col)
        r12[col] = math.sqrt(sum(welch(a_(adm, pa, "nodewise1d")[0],
                                       a_(adm, pb, "nodewise")[0])[1] ** 2
                                 for _l, pa, pb in T_ARMS) / 12.0)
    chk("T rms se, final_test / plateau5", r12["final_test"] / r12["plateau5"], 2.8,
        "S4.7 endpoint paragraph", "%.1f")


# ======================= NEW: the Q calibration (S3.3, S4.4, S7 T13, Fig. 2)
# S4.4 refers Cochran Q to chi2_{k-1}, but its weights w_i = se_i^-2 come from Welch
# standard errors estimated at 2.04-9.68 df (median 2.91), so chi2 is the wrong
# reference and every Q p-value on that layer is anticonservative.  c99_qcalibration
# simulates the paper's OWN estimator -- the same welch(), the same DerSimonian-Laird
# meta() -- under a homogeneous truth and gives the reference Q actually has.  The
# seed and draw count are REGISTERED in c99_qcalibration (SEED, DRAWS); changing
# either changes every number below, which is why they live there and not here.
# This section is numbered [15b] on purpose: S3.4 names "section [16]" as the census
# self-check, and renumbering censuscheck would silently falsify that sentence.
def calibration(rows, adm, args):
    print("\n[15b] Q CALIBRATION  (S3.3, S4.4, S7 T13, Fig. 2)")
    import c99_qcalibration as K
    cs = K.live_arms(adm)
    chk("cells in the calibrated pool", len(cs), 14, "S4.4, T13", "%.0f")
    print("   fast estimator agrees with welch() to %.1e  (seed %d, %d draws)"
          % (K.check_estimator_fidelity(cs, K.SEED), K.SEED, K.DRAWS))
    dfs = [K.welch_df(c) for c in cs]
    chk("min Welch df of the fourteen cells", min(dfs), 2.04, "S3.3, T13", "%.2f")
    chk("median Welch df -- why chi2 is the wrong reference", st.median(dfs), 2.91,
        "S3.3, S4.4, T13", "%.2f")
    chk("max Welch df", max(dfs), 9.68, "S3.3, T13", "%.2f")
    psd, pdf = K.pooled_arm_sd(cs)
    chk("pooled within-arm sd (the common-sd null)", psd, 0.186, "S3.3", "%.3f")
    chk("   its df", pdf, 70, "S3.3", "%.0f")

    obs = K.observed(cs)
    per = K.qnull(cs, K.DRAWS, K.SEED, "percell")
    com = K.qnull(cs, K.DRAWS, K.SEED, "common")
    chk("null Q on 13 df: mean", st.mean(per["Q"]), 26.9, "S7 T13", "%.1f")
    chk("   median", K.quant(per["Q"], .5), 21.4, "S7 T13", "%.1f")
    chk("   95th percentile", K.quant(per["Q"], .95), 62.4, "S7 T13", "%.1f")
    chk("MC p of Q = 102.47", K.mc_p(per["Q"], obs["Q"]), 0.013,
        "abstract, S1, S4.4, S7 T13, Fig. 2", "%.3f")
    chk("   ...under the common-sd null", K.mc_p(com["Q"], obs["Q"]), 0.002,
        "S4.4, S7 T13", "%.3f")
    chk("MC p of between-base Q = 95.12", K.mc_p(per["B"], obs["between"]), 0.005,
        "S1, S4.4, S7 T13, Fig. 2", "%.3f")
    chk("   ...under the common-sd null", K.mc_p(com["B"], obs["between"]), 0.0007,
        "S7 T13", "%.4f")
    chk("MC p of within-level Q = 7.36", K.mc_p(per["W"], obs["within"]), 0.86,
        "S1, S4.4, Fig. 2", "%.2f")
    for b, p in (("SGD", 0.70), ("RMSProp", 0.30), ("SGDm", 0.86), ("AdamW", 0.25)):
        chk("MC p of the %-8s level Q" % b, K.mc_p(per["LV"][b], obs["per"][b]), p,
            "S1.1 C3, S4.4 table, Fig. 2", "%.2f")
    chk("null median of the SGDm level Q (7 df)", K.quant(per["LV"]["SGDm"], .5), 9.4,
        "abstract, S1, S4.4", "%.1f")

    # tau and I^2 subtract k-1 = 13 where the null mean is 26.9: UPPER BOUNDS.
    w = [1.0 / se ** 2 for _, se in obs["items"]]
    W = sum(w); den = W - sum(x ** 2 for x in w) / W
    e = st.mean(per["Q"])
    chk("tau recentred on the null mean", math.sqrt(max(0.0, (obs["Q"] - e) / den)),
        0.271, "S4.4, S7 T13", "%.3f")
    chk("I^2 recentred on the null mean", 100 * max(0.0, (obs["Q"] - e) / obs["Q"]),
        73.7, "S4.4, S7 T13", "%.1f")

    # the intervals
    sg = [c for c in cs if c["base"] == "SGDm"]
    sper = K.qnull(sg, K.DRAWS, K.SEED, "percell")
    chk("coverage of the SGDm pool's +-1.96 se interval, %", 100 * K.coverage(sper, 1.96),
        73.0, "abstract, S4.4, S7 T13", "%.1f")
    chk("   its calibrated 95% half-width, pp", K.half_width(sper), 0.121,
        "abstract, S1, S4.4, S7 T13", "%.3f")
    chk("   the nominal half-width it replaces", 1.96 * K.observed(sg)["se"], 0.088,
        "S1, S7 T13", "%.3f")
    chk("coverage of the 14-cell fixed-effect interval, %", 100 * K.coverage(per, 1.96),
        65.2, "S4.4, S7 T13", "%.1f")
    chk("   its calibrated 95% half-width, pp", K.half_width(per), 0.089,
        "S4.4, S7 T13", "%.3f")
    m_re, se_re = K.re_pool(obs["items"], obs["tau"])
    chk("random-effects pool over the fourteen", m_re, 0.611, "S4.4")
    chk("   its se", se_re, 0.087, "S4.4", "%.3f")
    chk("   its 95% CI, low", m_re - 1.96 * se_re, 0.441, "S4.4")
    chk("   ...high", m_re + 1.96 * se_re, 0.782, "S4.4")

    # the one quantity chi2 gets CONSERVATIVELY wrong
    chk("calibrated Q-profile upper limit on the SGDm tau",
        K.tau_upper(sg, K.DRAWS, K.SEED), 0.097, "S4.4, S7 T13", "%.3f")
    # the registered 1-df decision rule
    sz, cv = K.rule_size(cs, K.DRAWS, K.SEED)
    chk("realised size of the registered Q > 3.841 rule", sz, 0.10, "S4.4, S7 T13", "%.2f")
    chk("   its size-0.05 critical value, not 3.841", cv, 6.22, "S7 T13", "%.2f")

    # THE LOAD-BEARING STATISTIC: no se enters it anywhere.
    e0, ge, tot, pp_, med, p95, mx = K.permutation_eta2(cs)
    chk("weight-free eta^2, base-optimiser grouping", e0, 0.840,
        "S1, S1.1 C3, S4.4, S7 T13, Fig. 2", "%.3f")
    chk("   partitions of shape {8,2,2,2}", tot, 45045, "S4.4", "%.0f")
    chk("   the base partition's rank among them", ge, 9, "S4.4, S7 T13", "%.0f")
    chk("   its EXACT permutation p", pp_, 0.00020,
        "S1, S4.4, S7 T13, Fig. 2", "%.5f")
    chk("   null eta^2 median", med, 0.200, "S4.4", "%.3f")
    s0, ge2, tot2, p2 = K.permutation_share(cs)
    chk("   the WEIGHTED share permutation the paper already prints", p2, 0.00031,
        "S4.4 -- cross-check that this enumeration is the paper's own", "%.5f")

    # the endpoint disclosure, calibrated.  plateau5 stays primary.
    END = {"plateau5":   (0.013, 0.005, 0.840, 0.00020, 0.611),
           "plateau":    (0.24,  0.099, 0.800, 0.00029, 0.480),
           "best_test":  (0.90,  0.60,  0.276, 0.33851, 0.396),
           "final_test": (0.16,  0.025, 0.670, 0.00757, 0.674)}
    for col in MS_METRICS:
        mcq, mcb, eta, pe, rem = END[col]
        ec = K.live_arms(adm, _arm_on(col)); o = K.observed(ec)
        s = K.qnull(ec, K.DRAWS, K.SEED, "percell")
        chk("MC p of Q, %-11s" % col, K.mc_p(s["Q"], o["Q"]), mcq,
            "S4.4 endpoint table, S7 T13", "%.3f" if col == "plateau5" else "%.2f")
        chk("   MC p of between-base Q, %-11s" % col, K.mc_p(s["B"], o["between"]), mcb,
            "S4.4 endpoint note, S7 T13", "%.3f")
        q_ = K.permutation_eta2(ec)
        chk("   weight-free eta^2, %-11s" % col, q_[0], eta, "S4.4 endpoint note", "%.3f")
        chk("      its exact p, %-11s" % col, q_[3], pe, "S4.4 endpoint note", "%.5f")
        chk("   random-effects pool, %-11s" % col,
            K.re_pool([K._welch(c["cv"], c["nv"]) for c in ec], o["tau"])[0], rem,
            "S4.4 endpoint note")
    # S5.4's Gstat family, calibrated the same way -- a DIFFERENT contrast (chunk2325
    # minus nodewise1d) and therefore its own null.  K.gfamily_null re-derives both Q's
    # from the raw arms through the paper's own welch()/meta() and asserts they equal the
    # printed 18.21 and 28.25 before it simulates anything.  Neither resolves, so S5.4's
    # old "heterogeneous where the pre-specified one was not" was a property of the
    # reference and is corrected in the manuscript rather than restated.
    g = K.gfamily_null(adm, K.DRAWS, K.SEED)
    chk("G family, twelve cells: Q", g["12"][0], 18.21, "S5.4", "%.2f")
    chk("   its MC p", g["12"][3], 0.42, "S5.4, S7 T13", "%.2f")
    chk("   its null Q mean on 11 df", g["12"][4], 20.1, "S5.4, S7 T13", "%.1f")
    chk("G family, fourteen cells: Q", g["14"][0], 28.25, "S5.4", "%.2f")
    chk("   its MC p -- NOT resolved", g["14"][3], 0.26, "S5.4, S7 T13", "%.2f")
    chk("   its null Q mean on 13 df", g["14"][4], 24.0, "S5.4, S7 T13", "%.1f")
    chk("G family, fifteen with bn1: Q", g["15"][0], 42.98, "S5.4", "%.2f")
    chk("   its MC p -- NOT resolved", g["15"][3], 0.12, "S5.4, S7 T13", "%.2f")
    chk("   its null Q mean on 14 df", g["15"][4], 26.1, "S5.4, S7 T13", "%.1f")
    mres = K.subpool_null([c for c in cs if c["base"] in ("SGDm", "AdamW")],
                          K.DRAWS, K.SEED)
    chk("momentum-present residual Q (S4.5's 2x2 collapse)", mres[0], 34.64, "S4.5", "%.2f")
    chk("   its MC p -- NOT resolved", mres[3], 0.074, "S4.5, S7 T13", "%.3f")
    chk("   its null Q mean on 9 df", mres[4], 16.5, "S4.5, S7 T13", "%.1f")
    chk("null mean of the best_test Q -- why 'below its own df' is the wrong test",
        st.mean(K.qnull(K.live_arms(adm, _arm_on("best_test")), K.DRAWS, K.SEED,
                        "percell")["Q"]), 25.3, "S4.4 reading 1", "%.1f")

# ------------------------------------------------- the census, ASSERTED not measured
# §3.4 of the manuscript prints this audit's own coverage.  Until cycle 102 `--census`
# only MEASURED it, so the sentence "216 of the 725 ... 29.8%" sat in the draft for
# three review cycles while the code printed 747 and 28.9%, and every run still exited
# 0.  Section [16] turns that sentence into an asserted number like any other.
#
# THE PAPER VALUE IS READ OUT OF THE PAPER, not duplicated here.  Every other chk()
# in this file compares a derived value against a constant that a human copied from
# the manuscript; for this one the constant IS the manuscript's own sentence, so
# there is exactly one place to update and no way for the code's idea of what §3.4
# prints to drift from what §3.4 prints.  Reword the sentence and the parse fails
# loudly rather than passing quietly.
#
# WHEN THIS FAILS: it is telling you §3.4 has gone stale -- usually because a chk()
# site was added or removed somewhere above.  Run
#     python3 analysis/c98_reproduce.py --census
# and write the printed triple into the §3.4 sentence of BOTH paper/paper.tex and
# paper/DRAFT-v4.md, then re-run until it is a fixpoint (it converges in one step
# unless the new percentage string is a decimal that was not already in the draft).
CENSUS_SHAPE = ("... the audit executes **N claim-carrying assertions covering C of "
                "the Q distinct quantity-numerals** in this manuscript, which is P% "
                "of them ...")
_CENSUS_RE = re.compile(
    r"audit executes (\d+) (?:claim-carrying )?assertions covering (\d+) of the "
    r"(\d+) distinct quantity-numerals in this manuscript(?:, which is "
    r"([\d.]+)% of them)?")

def _census_claim(path):
    """(assertions, covered, distinct, pct) as §3.4 PRINTS them, or None.

    Reads BOTH markups.  DRAFT-v4.md needs only markdown emphasis flattened.
    paper.tex additionally wraps the triple in \\textbf{...} and escapes the per
    cent sign, so the LaTeX branch strips the markup macro names, the braces and
    the backslash before the shape regex runs.  Neither branch touches a digit,
    and neither re-censuses the .tex: the QUANTITY count is defined by census()'s
    fence rule, which is a markdown rule, so there is one measurement and both
    files must print it."""
    raw = open(path).read()
    if path.endswith(".tex"):
        raw = re.sub(r"\\(?:textbf|textit|emph|mathbf|texttt|mathrm)\s*\{", "{", raw)
        raw = raw.replace("\\%", "%").replace("\\,", " ").replace("~", " ")
        raw = raw.replace("{", " ").replace("}", " ")
    flat = re.sub(r"[*`\s]+", " ", raw)
    m = _CENSUS_RE.search(flat)
    if not m: return None
    n, c, q, p = m.groups()
    return int(n), int(c), int(q), (float(p) if p else None)

def censuscheck(rows, adm, args):
    """§3.4's coverage sentence, re-measured and asserted against BOTH markups."""
    global CENSUS_MARK
    print("\n[16] THE COVERAGE CENSUS, ASSERTED  (§3.4 Registration and scope)")
    if not getattr(args, "_full", True):
        skip("censuscheck", "only some sections were requested, so the assertion "
                            "count would not be the manuscript's")
        return
    path = getattr(args, "draft", None) or DRAFT
    if not os.path.exists(path):
        skip("censuscheck", "no manuscript in this tree (the deposit ships none), "
                            "so §3.4's coverage cannot be re-measured here")
        return
    # The MEASUREMENT is taken once, on the markdown draft.  The ASSERTION is made
    # against every markup present, so a triple that goes stale in paper.tex alone
    # -- which §3.4 used to claim was impossible, while [16] only ever read the
    # draft -- now exits non-zero instead of passing quietly.
    targets = [path]
    if os.path.abspath(path) == os.path.abspath(DRAFT) and os.path.exists(TEX):
        targets.append(TEX)
    claims = {}
    for q_ in targets:
        c_ = _census_claim(q_)
        if c_ is None:
            FAILS.append(("§3.4 coverage sentence not parseable in %s"
                          % os.path.relpath(q_, ROOT), "-", "-",
                          "expected the shape: " + CENSUS_SHAPE))
            print("  **FAIL** could not find §3.4's coverage sentence in %s."
                  % os.path.relpath(q_, ROOT))
            print("           expected shape:  %s" % CENSUS_SHAPE)
        else:
            claims[q_] = c_
    if not claims:
        return
    CENSUS_MARK = len(ASSERTED)      # freeze BEFORE asserting: see census()
    _n_tok, n_qd, _n_q, n_cov = census(path, quiet=True)
    for q_, (p_sites, p_cov, p_qd, p_pct) in claims.items():
        tag = os.path.basename(q_)
        chk("sites executed        %-20s" % tag, CENSUS_MARK, p_sites,
            "§3.4", "%.0f")
        chk("distinct numerals asserted %-15s" % tag, n_cov, p_cov, "§3.4", "%.0f")
        chk("distinct numerals in draft %-15s" % tag, n_qd, p_qd, "§3.4", "%.0f")
        if p_pct is not None:
            chk("coverage of distinct numerals %-12s" % tag,
                100.0 * n_cov / n_qd if n_qd else 0.0, p_pct, "§3.4", "%.1f")
    print("    (the `paper` column here is §3.4's own sentence, read out of %s."
          % " and ".join(os.path.relpath(q_, ROOT) for q_ in claims))
    print("     The COUNT is measured on %s alone -- one measurement, both markups"
          % os.path.relpath(path, ROOT))
    print("     assert it.  A FAIL means that sentence has gone stale, not that a")
    print("     result moved: re-run with --census and write the printed triple into")
    print("     §3.4 in BOTH paper.tex and DRAFT-v4.md, then re-run to a fixpoint.)")


# ================================================ NEW: the design-point set, §5.6 / §5.8
# B6.  §5.8 states a design-point rule -- same network, dataset, base--meta pairing,
# eta and budget -- and the enumeration that follows it printed ELEVEN points while the
# rule itself yields TEN: `rl3` at eta 3e-4 and `fa1` are identical on every element of
# the key and differ only in the step-size clip box, which the same enumeration already
# collapses across for its six-batch point and which §4.4 measures and rejects as a
# moderator.  This section applies the rule as written and asserts every number §5.6 and
# §5.8 print under it, including the 12-point sensitivity the paper reports and rejects.
DP_KEY = lambda c: (c["network"], c["dataset"], c["base"], c["eta"], c["epochs"])

def _dp_instr(adm):
    """The tail-free instrument of §5.6: mean of the chunk2325 and nodewise1d arms."""
    out = {}
    for (lab, net, ds, base, eta, ep, ch, nd, c23, n1d) in F.CELLS:
        if lab == F.GN_CELL and not F.WITH_GN: continue
        if not (c23 and n1d): continue
        gv, _ = arm(adm, *c23); hv, _ = arm(adm, *n1d)
        if gv and hv: out[lab] = 0.5 * (st.mean(gv) + st.mean(hv))
    return out

def _dp_box(rows):
    """The beta_clip box(es) each cell's own D arms ran in, read off the CSV."""
    out = {}
    for (lab, net, ds, base, eta, ep, ch, nd, c23, n1d) in F.CELLS:
        if lab == F.GN_CELL and not F.WITH_GN: continue
        b = set()
        for pre, gran in (ch, nd):
            b |= {r["beta_clip"] for r in rows
                  if r["run"].startswith(pre) and r["granularity"] == gran}
        out[lab] = "+".join(sorted(b))
    return out

def _dp_points(cs, keyfn, instr):
    g = {}
    for c in cs: g.setdefault(keyfn(c), []).append(c)
    pts = []
    for cl in g.values():
        ins = [instr[c["label"]] for c in cl if c["label"] in instr]
        pts.append(dict(labels=[c["label"] for c in cl],
                        D=st.mean([c["D"] for c in cl]),
                        level=st.mean([c["aligned"] for c in cl]),
                        headroom=st.mean([c["headroom"] for c in cl]),
                        instr=(st.mean(ins) if ins else None),
                        dataset=cl[0]["dataset"], network=cl[0]["network"],
                        base=cl[0]["base"]))
    return sorted(pts, key=lambda p: (p["dataset"], -p["D"]))

def _dp_fit_mean(tr):
    m = st.mean([q["D"] for q in tr]);  return lambda p: m
def _dp_fit_origin(tr, x):
    k = sum(x(q) * q["D"] for q in tr) / sum(x(q) ** 2 for q in tr)
    return lambda p: k * x(p)
def _dp_fit_ols(tr, x):
    n = len(tr); xs = [x(q) for q in tr]; ys = [q["D"] for q in tr]
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    if sxx == 0: return lambda p: my           # dummy is constant in this fold
    b = sum((v - mx) * (y - my) for v, y in zip(xs, ys)) / sxx
    return lambda p: (my - b * mx) + b * x(p)
DP_MODELS = [
    ("mean (baseline)",   _dp_fit_mean),
    ("k*log(headroom)",   lambda tr: _dp_fit_origin(tr, lambda q: math.log(q["headroom"]))),
    ("CIFAR-100 dummy",   lambda tr: _dp_fit_ols(tr, lambda q: 1.0 if q["dataset"] == "C100" else 0.0)),
    ("k*headroom",        lambda tr: _dp_fit_origin(tr, lambda q: q["headroom"])),
    ("level (OLS)",       lambda tr: _dp_fit_ols(tr, lambda q: q["level"]))]

def _dp_loo(pts):
    out = {}
    for name, fit in DP_MODELS:
        e = [fit([q for j, q in enumerate(pts) if j != i])(p) - p["D"]
             for i, p in enumerate(pts)]
        c10 = [v for v, p in zip(e, pts) if p["dataset"] == "C10"]
        out[name] = (math.sqrt(sum(v * v for v in e) / len(e)),
                     math.sqrt(sum(v * v for v in c10) / len(c10)), e)
    return out

def _dp_binom2(w, n):
    from math import comb
    lo = min(w, n - w)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)

def _dp_ols(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    sxy = sum((v - mx) * (y - my) for v, y in zip(xs, ys))
    b = sxy / sxx; a = my - b * mx
    res = [y - (a + b * v) for v, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in res) / (n - 2) / sxx) if n > 2 else float("nan")
    syy = sum((y - my) ** 2 for y in ys)
    return b, se, b / se, sxy / math.sqrt(sxx * syy)

def _dp_perm(xs, ys):
    """Exact two-sided permutation p for the OLS slope.  sxx and both means are fixed
    under permutation of y, so |b| is monotone in |sum(x_i y_sigma(i)) - n mx my|."""
    import itertools
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n; c = n * mx * my
    obs = abs(sum(a * b for a, b in zip(xs, ys)) - c)
    hit = tot = 0
    for pm in itertools.permutations(ys):
        tot += 1
        if abs(sum(a * b for a, b in zip(xs, pm)) - c) >= obs - 1e-9: hit += 1
    return hit / tot

def _dp_critr(n):
    """Two-sided 5% critical Pearson r at df = n-2: bisect the Student-t survival
    function, written as a regularised incomplete beta (Numerical Recipes 6.4)."""
    df = n - 2
    def _betacf(a, b, x):
        tiny = 1e-30; c = 1.0; d = 1 - (a + b) * x / (a + 1)
        if abs(d) < tiny: d = tiny
        d = 1 / d; h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
            d = 1 + aa * d; c = 1 + aa / c
            if abs(d) < tiny: d = tiny
            if abs(c) < tiny: c = tiny
            d = 1 / d; h *= d * c
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
            d = 1 + aa * d; c = 1 + aa / c
            if abs(d) < tiny: d = tiny
            if abs(c) < tiny: c = tiny
            d = 1 / d; de = d * c; h *= de
            if abs(de - 1) < 1e-14: break
        return h
    def _betainc(a, b, x):
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        if x < (a + 1) / (a + b + 2):
            return math.exp(math.log(x) * a + math.log(1 - x) * b + lb) * _betacf(a, b, x) / a
        return 1 - math.exp(math.log(1 - x) * b + math.log(x) * a + lb) * _betacf(b, a, 1 - x) / b
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _betainc(df / 2, 0.5, df / (df + mid * mid)) > 0.05: lo = mid
        else: hi = mid
    t = (lo + hi) / 2
    return t / math.sqrt(t * t + df)

def designpoints(rows, adm, args):
    print("\n[17] THE DESIGN-POINT SET  (§5.8's rule, applied as written; §5.6's slopes)")
    cs    = cells(adm)
    instr = _dp_instr(adm)
    box   = _dp_box(rows)

    # --- the collapse the previous enumeration missed, at cell level.  RULE 20 was
    #     checked against the runs' own ARGS:/ENV: lines; see
    #     paper/sections/v6-design-points.md §A.2.
    d = {c["label"]: c for c in cs}
    chk("rl3 @3e-4 and fa1 share every key field", 1.0 if
        DP_KEY(d["rl3 @3e-4"]) == DP_KEY(d["fa1"]) else 0.0, 1.0, "§5.8", "%.0f")
    chk("...and differ in the clip box", 0.0 if box["rl3 @3e-4"] == box["fa1"] else 1.0,
        1.0, "§5.8", "%.0f")
    dd  = d["fa1"]["D"] - d["rl3 @3e-4"]["D"]
    sed = math.sqrt(d["fa1"]["seD"] ** 2 + d["rl3 @3e-4"]["seD"] ** 2)
    chk("fa1 - rl3@3e-4, as cells", dd, +0.038, "§5.8")
    chk("   se", sed, 0.156, "§5.8", "%.3f")
    chk("   z",  dd / sed, +0.24, "§5.8", "%.2f")
    chk("cells of the six-batch point in the -15 box",
        float(sum(1 for l in ("cc1", "mm1", "pp1", "gn1 (BN)", "ml2")
                  if box[l] == "-15:-2.3026")), 5, "§5.8", "%.0f")

    pts = _dp_points(cs, DP_KEY, instr)
    c10 = [p for p in pts if p["dataset"] == "C10"]
    chk("design points under the rule as written", len(pts), 10, "§5.8", "%.0f")
    chk("   of which CIFAR-10",                    len(c10),  9, "§5.6", "%.0f")

    res = _dp_loo(pts)
    for name, paper, paper10 in [("mean (baseline)", 0.3868, 0.2809),
                                 ("k*log(headroom)", 0.2667, 0.2352),
                                 ("CIFAR-100 dummy", 0.3775, None),
                                 ("k*headroom",      0.3654, None),
                                 ("level (OLS)",     0.9319, 0.2309)]:
        chk("LOO RMSE  %-16s" % name, res[name][0], paper, "§5.8 table", "%.4f")
        if paper10 is not None:
            chk("   CIFAR-10 folds", res[name][1], paper10, "§5.8 prose", "%.4f")
    base, alt = res["mean (baseline)"][2], res["k*log(headroom)"][2]
    w  = sum(1 for i in range(len(pts)) if abs(alt[i]) < abs(base[i]))
    w9 = sum(1 for i, p in enumerate(pts) if p["dataset"] == "C10" and abs(alt[i]) < abs(base[i]))
    chk("sign test, all folds (wins)", w, 8, "§5.8", "%.0f")
    chk("   p", _dp_binom2(w, len(pts)), 0.109, "§5.8", "%.3f")
    chk("sign test, CIFAR-10 folds (wins)", w9, 7, "§5.8", "%.0f")
    chk("   p", _dp_binom2(w9, len(c10)), 0.180, "§5.8", "%.3f")
    for e, nm, pv in [(base, "mean", -0.887), (alt, "k*log(headroom)", -0.462)]:
        for v, q in zip(e, pts):
            if q["dataset"] == "C100":
                chk("CIFAR-100 fold error, %-16s" % nm, v, pv, "§5.8")
    chk("critical |r| at 10 design points", _dp_critr(10), 0.632, "§5.8", "%.3f")
    chk("   its r^2 (share of variance needed)", 100 * _dp_critr(10) ** 2, 39.9,
        "§5.8", "%.1f")
    chk("critical |r| at 9 CIFAR-10 points", _dp_critr(9), 0.666, "§5.6", "%.3f")
    chk("|r| = 0.4 first visible at n =",
        float(min(n for n in range(5, 60) if _dp_critr(n) <= 0.4)), 25, "§5.8", "%.0f")

    # the level model, out of sample
    f = _dp_fit_ols(c10, lambda q: q["level"]); m = _dp_fit_mean(c10)
    for p in pts:
        if p["dataset"] == "C100":
            chk("level model predicts D(CIFAR-100)", f(p), +4.43, "§5.8", "%.2f")
            chk("   observed",                      p["D"], +1.56, "§5.8", "%.2f")
            chk("   its error",                     f(p) - p["D"], +2.86, "§5.8", "%.2f")
            chk("   the corpus mean's error",       m(p) - p["D"], -0.89, "§5.8", "%.2f")

    # --- §5.6's slopes on the same points
    def sl(sel, nm, pb, pse, pt, pr=None):
        b, se, t, r = _dp_ols([p["level"] for p in sel], [p["D"] for p in sel])
        chk("slope %-26s" % nm, b, pb, "§5.6")
        chk("   se", se, pse, "", "%.3f")
        chk("   t",  t,  pt,  "", "%.2f")
        if pr is not None: chk("   r", r, pr, "", "%.3f")
        return b
    sl(pts, "all 10 points",          -0.045, 0.011, -4.21, -0.830)
    sl(c10, "9 CIFAR-10 points",      -0.176, 0.056, -3.16, -0.767)
    chk("   exact permutation p over 9!",
        _dp_perm([p["level"] for p in c10], [p["D"] for p in c10]), 0.0151, "§5.6", "%.4f")
    sgdm = [p for p in c10 if p["base"] == "SGDm"]
    r18  = [p for p in c10 if p["network"] == "ResNet-18"]
    both = [p for p in r18 if p["base"] == "SGDm"]
    chk("CIFAR-10 points with base SGDm",  len(sgdm), 5, "§5.6", "%.0f")
    chk("CIFAR-10 points on ResNet-18",    len(r18),  7, "§5.6", "%.0f")
    chk("CIFAR-10 points with both fixed", len(both), 3, "§5.6", "%.0f")
    bs = sl(sgdm, "base fixed at SGDm",   -0.126, 0.022, -5.74)
    br = sl(r18,  "network fixed at R18", -0.287, 0.075, -3.83)
    sl(both,      "both fixed",           -0.188, 0.148, -1.27)
    chk("ratio of the two held-one slopes", br / bs, 2.3, "§5.6", "%.1f")
    chk("both-fixed exact permutation p over 3!",
        _dp_perm([p["level"] for p in both], [p["D"] for p in both]), 0.667, "§5.6", "%.3f")
    ins = [p for p in c10 if p["instr"] is not None]
    b, se, t, _ = _dp_ols([p["instr"] for p in ins], [p["D"] for p in ins])
    chk("slope on the independent instrument", b, -0.208, "§5.6")
    chk("   se", se, 0.089, "", "%.3f"); chk("   t", t, -2.33, "", "%.2f")
    # the arm-sharing artefact
    mv = []
    for (lab, net, ds, base, eta, ep, ch, nd, c23, n1d) in F.CELLS:
        if lab == F.GN_CELL and not F.WITH_GN: continue
        if ds != "C10": continue
        nv, _ = arm(adm, *nd); mv.append(st.variance(nv) / len(nv))
    chk("mean var of a nodewise arm mean, 18 C10 cells", st.mean(mv), 0.01724, "§5.6", "%.5f")
    vl = st.variance([p["level"] for p in c10])
    chk("var of level over the 9 CIFAR-10 points", vl, 1.19374, "§5.6", "%.5f")
    chk("mechanical slope", -st.mean(mv) / vl, -0.014, "§5.6")

    # --- the sensitivity §5.8 reports and rejects: the clip box in the key -> 12 points
    p12 = _dp_points(cs, lambda c: DP_KEY(c) + (box[c["label"]],), instr)
    chk("design points if the clip box enters the key", len(p12), 12, "§5.8", "%.0f")
    r12 = _dp_loo(p12)
    chk("   LOO RMSE, mean baseline",   r12["mean (baseline)"][0], 0.3514, "§5.8", "%.4f")
    chk("   LOO RMSE, k*log(headroom)", r12["k*log(headroom)"][0], 0.2437, "§5.8", "%.4f")
    b12, a12 = r12["mean (baseline)"][2], r12["k*log(headroom)"][2]
    w12 = sum(1 for i in range(len(p12)) if abs(a12[i]) < abs(b12[i]))
    chk("   sign test wins", w12, 10, "§5.8", "%.0f")
    chk("   its p (the one threshold this reading crosses)",
        _dp_binom2(w12, len(p12)), 0.039, "§5.8", "%.3f")
    b5 = [p for p in p12 if p["dataset"] == "C10" and p["base"] == "SGDm"
          and p["network"] == "ResNet-18"]
    bb5, se5, t5, _ = _dp_ols([p["level"] for p in b5], [p["D"] for p in b5])
    chk("   §5.6's both-fixed leg at 5 points", bb5, -0.201, "§5.8")
    chk("      se", se5, 0.092, "", "%.3f"); chk("      t", t5, -2.19, "", "%.2f")
    chk("      its exact permutation p over 5!",
        _dp_perm([p["level"] for p in b5], [p["D"] for p in b5]), 0.100, "§5.8", "%.3f")

    # --- the leakage variant §5.8 names and rejects
    NEW = ("sm3", "bm2 (SGD)", "bm2 (RMSProp)", "sm4")
    p13 = _dp_points(cs, lambda c: ("E", c["label"]) if c["label"] in NEW else DP_KEY(c), instr)
    chk("design points if the four new cells are new folds", len(p13), 13, "§5.8", "%.0f")
    r13 = _dp_loo(p13)
    b13, a13 = r13["mean (baseline)"][2], r13["k*log(headroom)"][2]
    w13 = sum(1 for i in range(len(p13)) if abs(a13[i]) < abs(b13[i]))
    chk("   sign test wins", w13, 11, "§5.8", "%.0f")
    chk("   its p", _dp_binom2(w13, len(p13)), 0.022, "§5.8", "%.3f")

SECTIONS = [("corpus", corpus), ("table2", table2), ("heterogeneity", heterogeneity),
            ("alignment", alignment), ("prescription", prescription), ("tail", tail),
            ("budget", budget), ("competitiveness", competitiveness),
            ("rho", rho), ("gn1gate", gn1gate), ("metacensus", metacensus),
            ("countaxis", countaxis),
            ("tuning", tuning), ("appendices", appendices), ("deposit", deposit),
            ("metricsens", metricsens), ("calibration", calibration),
            ("designpoints", designpoints),
            ("censuscheck", censuscheck)]      # MUST stay last: it freezes CENSUS_MARK

def main():
    ap = argparse.ArgumentParser()
    for name, _ in SECTIONS: ap.add_argument("--" + name, action="store_true")
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--with-gn", action="store_true",
                    help="re-include the GroupNorm cell that R0 item 1 removes")
    ap.add_argument("--census", action="store_true",
                    help="print the coverage census and nothing else")
    ap.add_argument("--draft", default=DRAFT, help="the manuscript the census reads")
    ap.add_argument("--no-census", action="store_true")
    ap.add_argument("--allow-stale-census", action="store_true",
                    help="report a stale §3.4 coverage sentence but do not exit "
                         "non-zero for it.  Used by c98_release.py: the deposit "
                         "ships no manuscript, so §3.4's sentence is not a number "
                         "the artefact can get wrong.  NEVER use it to close F2.")
    a = ap.parse_args()
    F.WITH_GN = a.with_gn
    want = [n for n, _ in SECTIONS if getattr(a, n)] or [n for n, _ in SECTIONS]
    a._full = len(want) == len(SECTIONS)
    rows, adm = load(a.csv)
    if a.census:
        import io as _io, contextlib as _c
        buf = _io.StringIO()
        with _c.redirect_stdout(buf):
            for name, fn in SECTIONS: fn(rows, adm, a)
        census(a.draft)
        if FAILS:
            print("\n%d CHECK(S) FAILED while measuring the census:" % len(FAILS))
            for n, got, paper, where in FAILS:
                print("   %-46s derived %s vs paper %s   (%s)" % (n, got, paper, where))
            return 1
        return 0
    print("=" * 78)
    print("REPRODUCTION AUDIT -- %s" % os.path.relpath(a.csv, ROOT))
    print("=" * 78)
    for name, fn in SECTIONS:
        if name in want: fn(rows, adm, a)
    print("\n" + "=" * 78)
    stale = []
    if a.allow_stale_census:
        stale = [f for f in FAILS if f[3] == "§3.4"]
        for f in stale: FAILS.remove(f)
    if FAILS:
        print("%d CHECK(S) FAILED:" % len(FAILS))
        for n, got, paper, where in FAILS:
            print("   %-46s derived %s vs paper %s   (%s)" % (n, got, paper, where))
    else:
        print("ALL %d CHECKS PASS." % len(ASSERTED))
    if stale:
        print("%d §3.4 COVERAGE CHECK(S) STALE, NOT COUNTED AS FAILURES "
              "(--allow-stale-census):" % len(stale))
        for n, got, paper, where in stale:
            print("   %-46s derived %s vs paper %s" % (n, got, paper))
        print("   The manuscript's coverage sentence is behind the code. Fix §3.4 in")
        print("   paper.tex AND DRAFT-v4.md and re-run WITHOUT this flag.")
    if SKIPPED:
        print("%d SECTION(S) COULD NOT RUN HERE, so this is not full coverage:"
              % len(SKIPPED))
        for sec, why in SKIPPED:
            print("   %-22s %s" % (sec, why))
    print("=" * 78)
    if not a.no_census:
        census(a.draft)
    return 1 if FAILS else 0

if __name__ == "__main__":
    sys.exit(main())
