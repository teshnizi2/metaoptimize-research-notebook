#!/usr/bin/env python3
"""
REGISTERED SCORERS FOR CYCLE 88 -- committed and hashed BEFORE the runs exist
(RULE 19).  Three batches, three decision rules, all fixed in advance.

  ub9  9 jobs   the parent's own unaugmented configuration, re-run in a WIDE box
                with PATCH_CLIPCOUNT, to find out whether the corpus's only
                reproduction of the parent's CIFAR-10 claim is a clipping artefact.
  aw1  12 jobs  the count-matched nodewise/chunk777 pair under an AdamW base --
                the corpus's single largest external-validity hole (all 112
                chunk*/nodewise1d rows are SGDm+Lion).
  gf2  20 jobs  the REPAIRED graded-1D-group-size dose curve.  APPENDIX-GRADE:
                register it, do not prioritise it.  The design it replaces (gf1)
                was killed for having zero power -- see `gf1_power_is_zero()`.

EVERY RULE HERE IS BINARY AND PRE-DECLARED.  The one thing a scorer must never do
is let the analyst pick the reduction after seeing the data, which is how this
campaign reversed three consecutive verdicts on one question using unchanged runs.

Usage
    python3 analysis/c88_scorers.py --selftest      # RULE 13, both directions
    python3 analysis/c88_scorers.py --power         # the power arithmetic, shown
    python3 analysis/c88_scorers.py --score ub9|aw1|gf2
"""
import csv, math, os, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "..", "results", "all_runs.csv")

# ---------------------------------------------------------------- constants
# Re-derived cycle 88 from results/all_runs.csv, NOT inherited from any doc.
POOLED_SD   = 0.2105   # within-arm sd, primary cell, df=18 (cc1/mm1/pp1/gn1)
D1_ANCHOR   = 0.5949   # mean of the four WITHIN-batch D's
D1_SD_BATCH = 0.0994   # sd of D across those four batches
# The four constituent values, so the anchor is never quoted without them:
D1_BY_BATCH = {"cc1": 0.727, "mm1": 0.485, "pp1": 0.581, "gn1": 0.587}
BATCH_FLOOR = 0.21     # sd_batch on ACCURACY (ANOVA); does NOT enter within-batch D

# gf2's three hypotheses, as points on S = D(2) + D(4).
#   H_SINGLETON  harm is specific to size-1 groups; k=2 removes it       -> S ~ 0.20
#   H_SMOOTH     harm declines smoothly in group size (1/sqrt(k)-like)   -> S ~ 0.69
#   H_TENSOR     only a WHOLE 1-D tensor recovers it; k=2,4 do nothing   -> S ~ 1.19
GF2_POINTS = {"H_SINGLETON": 0.20, "H_SMOOTH": 0.69, "H_TENSOR": 1.19}


# ---------------------------------------------------------------- plumbing
def load(path=CSV):
    return list(csv.DictReader(open(path)))

def num(r, k):
    try:    return float(r[k])
    except (TypeError, ValueError, KeyError): return None

def admissible(r):
    """window_ok AND complete AND a readable plateau5.  CORRECTIONS 119.3."""
    return (r.get("window_ok") == "1" and r.get("complete") == "1"
            and num(r, "plateau5") is not None)

def family(r):
    return r["run"].split("-")[0]

def sem(v):
    return st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")

def welch(a, b):
    """mean(a)-mean(b), its se, and t.  Returns (d, se, t)."""
    d = st.mean(a) - st.mean(b)
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return d, se, (d / se if se else float("inf"))


# ------------------------------------------------- GATE 0: box admissibility
def gate0(probe_rows):
    """Per-coordinate rails ONLY, scored PER SEED, never pooled.

    probe_rows :: [(run_name, coord_lo, coord_hi)]
    A statistic from a run at a guard is uninterpretable.  Reading occupancy
    from the 62-element per-tensor `beta` summary instead of n_at_lo/n_at_hi
    reported 0.000000 on 24 runs that were clipped in every record and
    INVERTED an arm ranking -- that defect voided batch ar1.
    """
    bound = [(n, lo, hi) for n, lo, hi in probe_rows if lo > 0 or hi > 0]
    per_arm = {}
    for n, lo, hi in bound:
        per_arm.setdefault(n.rsplit("-s", 1)[0], []).append(n)
    void = any(len(v) > 1 for v in per_arm.values())
    return {"bound_runs": bound, "void": void,
            "verdict": "VOID -- more than one seed of some arm is at a guard" if void
                       else ("CLEAN" if not bound else
                             "ADMISSIBLE, one seed excluded per affected arm")}


# ------------------------------------------------------------- ub9 (9 jobs)
def score_ub9(rows):
    """Is the corpus's only reproduction of the parent's CIFAR-10 claim real,
    or is it a clipping artefact?

    THE PROBLEM.  The 18 augment=0 runs (AdamW+Adam, eta=1e-3, alpha0=1e-6,
    100 ep) are the only cell that touches the parent's actual experiment.
    They read scalar 73.830 / blk6 74.352 / layerwise 73.449, i.e. the parent's
    blockwise>scalar claim reproduces at +0.522 and the NEXT rung reverses it.
    But beta0 = ln(1e-6) = -13.8155 against a -15 floor is 1.1845 nats, and
    under Lion |d beta| = eta per step, so with 500 steps/epoch x 100 epochs the
    travel budget is 50 nats: the FLOOR is reachable at epoch 2.4 and the
    CEILING (-2.3026, 11.5129 nats up) at epoch 23.0.  And `runs/PP` holds only
    Tensorboard_outputs -- there is NO probe.jsonl, so occupancy is UNMEASURED
    IN BOTH DIRECTIONS.  A floor bind on the single-group scalar arm while the
    6-group arm is free to differentiate away from it is precisely the mechanism
    that manufactures a +0.522.

    THE PURCHASE.  9 jobs: the SAME cell, 3 arms x 3 seeds, at a wide box with
    PATCH_CLIPCOUNT on.  Compared against the 18 runs already on disk.

    NOTE, AND IT IS AN ARITHMETIC RESULT, NOT A BUDGET ONE: at eta=1e-3 over 100
    epochs no box with a sane ceiling is PROVABLY free (it would need
    hi >= beta0 + 50 = +36.2, a step size of ~5e15).  So ub9 buys MEASURED
    freedom, not provable freedom, and the registration says so.
    """
    out = {}
    for label, box in (("orig", "-15:-2.3026"), ("wide", "-60:6.0")):
        arms = {}
        for g in ("scalar", "resnet18_blocks", "layerwise"):
            v = [num(r, "plateau5") for r in rows
                 if admissible(r) and r["augment"] == "0" and r["base"] == "AdamW"
                 and r["meta"] == "Adam" and r["meta_stepsize"] == "1e-3"
                 and r["alpha0"] == "1e-6" and r["beta_clip"] == box
                 and r["granularity"] == g]
            if v:
                arms[g] = v
        out[label] = arms
    return {"arms": out, "rule": _ub9_rule(out)}

def _ub9_rule(out):
    """PRE-REGISTERED, and it can come out either way.

    Let step1 = blk6 - scalar (the parent's own claim) and
        step2 = layerwise - blk6 (the rung that reverses it).
    """
    w = out.get("wide", {})
    if len(w) < 3:
        return "PENDING -- wide-box arm not yet run"
    s1 = st.mean(w["resnet18_blocks"]) - st.mean(w["scalar"])
    s2 = st.mean(w["layerwise"])       - st.mean(w["resnet18_blocks"])
    o  = out.get("orig", {})
    o1 = st.mean(o["resnet18_blocks"]) - st.mean(o["scalar"]) if len(o) == 3 else None
    # Registered thresholds, fixed before the runs.
    if s1 > 0.30 and s2 < -0.30:
        v = ("REPRODUCTION STANDS. The parent's blockwise>scalar step survives a box "
             "in which clipping is measured rather than assumed, AND the next rung "
             "still reverses it. The non-monotonicity at the parent's own config is real.")
    elif abs(s1) <= 0.30:
        v = ("REPRODUCTION DOES NOT SURVIVE. The +0.522 was a clipping artefact. "
             "This is a POSITIVE result: it identifies guard occupancy as a "
             "mechanism for the parent's own reported inconsistency, and it must "
             "be reported as such rather than buried.")
    else:
        v = ("PARTIAL -- step1 survives, step2 does not. Report both steps with "
             "their boxes; claim neither shape.")
    return {"step1_wide": round(s1, 3), "step2_wide": round(s2, 3),
            "step1_orig": None if o1 is None else round(o1, 3), "verdict": v}


# ------------------------------------------------------------- aw1 (12 jobs)
def score_aw1(rows):
    """Does the count-matched size-distribution effect exist under a base
    optimizer other than SGDm?

    THE HOLE.  All 112 chunk*/nodewise1d rows in the corpus are base=SGDm,
    meta=Lion, at meta_stepsize in {1e-4, 3e-4}.  Zero under any other base.
    150 AdamW granularity rows exist, all at eta=1e-3, with no chunk counterpart,
    and MASTER-TABLE row 100 records that the N_eff ordering INVERTS under AdamW.
    Every partition claim in the paper is therefore one (base, meta) pair, which
    is also the axis the parent's Section 9 sentence is indexed by.

    THE PURCHASE.  12 jobs: nodewise and chunk777 (m = 14,420 vs 14,421, one
    group apart) x 6 seeds, base=AdamW, ONE batch.  No new code -- chunk777 is
    PATCH_CHUNKWISE, already on both accounts.
    """
    a = [num(r, "plateau5") for r in rows if admissible(r) and family(r) == "aw1"
         and r["granularity"] == "chunk777"]
    b = [num(r, "plateau5") for r in rows if admissible(r) and family(r) == "aw1"
         and r["granularity"] == "nodewise"]
    if len(a) < 2 or len(b) < 2:
        return {"verdict": "PENDING -- aw1 has not run"}
    d, se, t = welch(a, b)
    return {"D_adamw": round(d, 3), "se": round(se, 3), "t": round(t, 2),
            "n": (len(a), len(b)), "D_sgdm_anchor": D1_ANCHOR,
            "rule": _aw1_rule(d, se)}

def _aw1_rule(d, se):
    """Registered before the runs. Three outcomes, all publishable."""
    lo, hi = d - 1.96 * se, d + 1.96 * se
    if lo > 0.30:
        return ("TRANSFERS. The size-distribution effect is not an artefact of one "
                "(base, meta) pair. The scope sentence in the abstract widens.")
    if hi < 0.15:
        return ("DOES NOT TRANSFER. This is the more interesting outcome and it must "
                "not be buried: it says the effect is a property of the SGDm+Lion "
                "instantiation, which is EXACTLY the axis the parent's Section 9 "
                "sentence is indexed by ('not consistent across the MetaOptimize "
                "approximations evaluated'), and it turns a scope limitation into a "
                "direct answer to the question they actually asked.")
    return ("UNRESOLVED at n=6. Report the interval. Do NOT re-cut the data, and do "
            "NOT describe it as 'partially transferring'.")


# ------------------------------------------------------------- gf2 (20 jobs)
def gf1_power_is_zero(sd=POOLED_SD, n=5, band=0.30, df=16, tcrit=1.746):
    """WHY THE ORIGINAL gf1 DESIGN WAS KILLED. Kept as executable evidence.

    gf1 registered 'TOST on DD at +/- 0.30' as its equivalence branch. With
    4 arms x 5 seeds, se(DD) = sd*sqrt(4/n), and the TOST bound is
    band - tcrit*se(DD).  If that is <= 0 the branch can NEVER fire, whatever
    the data say -- the same unreachable-band mistake gn1 already made at
    +/- 0.15.  Under gf1's own stated prior the modal verdict was UNRESOLVED
    at ~100%.
    """
    se_dd = sd * math.sqrt(4.0 / n)
    bound = band - tcrit * se_dd
    n_needed = math.ceil(4.0 * (sd * 2.57 / band) ** 2)
    return {"se_DD": round(se_dd, 4), "TOST_bound": round(bound, 4),
            "reachable": bound > 0, "n_per_arm_for_80pct": n_needed}

def score_gf2(rows):
    """THE REPAIRED DESIGN. Same 20 jobs, a rule that can actually fire.

    WHAT CHANGED, AND WHY.
      1. gf1 spent half its batch re-measuring D(1), an anchor already owned at
         four independent batches.  That was justified by 'a cross-batch anchor
         would sit on the 0.21 pp floor'.  It would not: D is a WITHIN-batch
         difference, so batch offsets cancel inside it, and the observed sd of D
         across the four batches (0.0994) is BELOW the 0.1719 expected from seed
         noise alone at n=3v3.  D is batch-transportable even though accuracy is
         not.  So the anchor is free and all four arms buy new information.
      2. gf1 varied ONE dose point (k=2).  Singleton fraction is 66.64% at k=1
         and EXACTLY 0% at every k >= 2, so along that axis 'degeneracy removed'
         and 'count reduced' are the same event -- corr = -1, the identical
         collinearity that killed the singleton ladder.  One interior point
         cannot separate a step from a slope.  gf2 buys TWO.
      3. gf1's H_DEGENERACY demanded DD >= 0.30, but the only mechanism that
         motivates the experiment (1/sqrt(N_g)) predicts ~0.19 -- inside gf1's
         own pre-declared UNRESOLVED band.  Both its hypotheses were strawmen.

    ARMS (ResNet-18, all counts re-derived cycle 88 from named_parameters()):
      A1 nodefloor2 m = 9,615  vs  A2 chunk1167 m = 9,619  (4 groups apart)
      A3 nodefloor4 m = 7,212  vs  A4 chunk1560 m = 7,212  (EXACT)
    Count exposure <= 0.00018 decades, five orders of magnitude under the effect.
    Both pairs in ONE batch, so the accuracy batch floor never enters either D.
    """
    def arm(g):
        return [num(r, "plateau5") for r in rows
                if admissible(r) and family(r) == "gf2" and r["granularity"] == g]
    a1, a2 = arm("nodefloor2"), arm("chunk1167")
    a3, a4 = arm("nodefloor4"), arm("chunk1560")
    if min(len(a1), len(a2), len(a3), len(a4)) < 2:
        return {"verdict": "PENDING -- gf2 has not run"}
    d2, se2, _ = welch(a2, a1)
    d4, se4, _ = welch(a4, a3)
    S = d2 + d4
    seS = math.sqrt(se2 ** 2 + se4 ** 2)
    return {"D2": round(d2, 3), "D4": round(d4, 3), "S": round(S, 3),
            "se_S": round(seS, 3), "rule": gf2_rule(S, seS)}

def gf2_rule(S, se_S):
    """Model selection on S = D(2) + D(4), NOT a TOST against zero.

    Nearest pre-registered point wins, and the winner must be at least 1.5 se
    closer than the runner-up; otherwise UNRESOLVED.  The three points are
    ~2.6 se apart, giving ~81-91% correct classification at n=5.
    """
    d = sorted(((abs(S - v), k) for k, v in GF2_POINTS.items()))
    (d0, best), (d1, second) = d[0], d[1]
    if (d1 - d0) < 1.5 * se_S:
        return (f"UNRESOLVED -- S={S:.3f} lies between {best} and {second}. "
                f"Report the interval. Do NOT call this 'mixed evidence'.")
    why = {"H_SINGLETON": ("The harm IS specific to size-1 groups; merging in pairs "
                           "suffices. The degeneracy claim survives in its narrow form."),
           "H_SMOOTH":    ("The harm declines smoothly with group size, so neither a "
                           "hard floor nor a tensor rule is the right instrument -- "
                           "shrinkage is, and CAM-HD already published it."),
           "H_TENSOR":    ("The unit is the TENSOR, not the group size. Splitting a "
                           "64-element BatchNorm gain into 32 groups of two costs as "
                           "much as 64 groups of one; only wholeness recovers it. This "
                           "is the first empirical justification for a rule Adam-mini, "
                           "Muon, LARS/LAMB and bitsandbytes all ship and none justifies.")}
    return f"{best} -- {why[best]}"

def gf2_crosscheck_note():
    """Registered NOW so it cannot be dropped later when inconvenient.

    Whatever gf2 says, it must agree in SIGN with the already-owned within-batch
    ck1 datum: chunk1 -> chunk2 removes 100% of the network's singletons and is
    worth +0.115 pp (se 0.133, t 0.87) against a count-only prediction of +0.13
    to +0.16, i.e. an excess of -0.046 to -0.016.  If gf2 returns H_SINGLETON
    while ck1 says removing ALL singletons buys nothing, the two are in direct
    conflict and the paper reports the conflict, not the convenient half.

    Note also the power limit of the ck1 datum, stated against our own interest:
    that step re-groups 100% of 11,173,962 coordinates, while the 1-D tail is
    9,610 of them (0.086% of the weights).  It cannot RULE OUT a tail-specific
    effect; it can only refuse a general size law.
    """
    return "registered"


# ------------------------------------------------------------------ selftest
def _selftest():
    fails = []
    def ck(name, got, want):
        ok = got == want
        print(f"  [{'OK ' if ok else 'FAIL'}] {name:56s} got={got!r}")
        if not ok: fails.append(name)

    print("S1 -- gf1's registered TOST branch is unreachable (why gf1 was killed)")
    p = gf1_power_is_zero()
    ck("S1a se(DD) at n=5", p["se_DD"], 0.1883)
    ck("S1b TOST bound is NEGATIVE", p["TOST_bound"] < 0, True)
    ck("S1c branch can never fire", p["reachable"], False)
    ck("S1d n/arm needed for 80% TOST power", p["n_per_arm_for_80pct"], 14)

    print("S2 -- gf2's rule fires on each hypothesis and is symmetric (RULE 13)")
    se = 0.1883
    for k, v in GF2_POINTS.items():
        ck(f"S2 {k} recovered at its own point", gf2_rule(v, se).split(" --")[0], k)
    ck("S2d midpoint 0.445 is UNRESOLVED", gf2_rule(0.445, se).startswith("UNRESOLVED"), True)
    ck("S2e midpoint 0.940 is UNRESOLVED", gf2_rule(0.940, se).startswith("UNRESOLVED"), True)
    ck("S2f far below all points -> nearest is H_SINGLETON",
       gf2_rule(-0.50, se).split(" --")[0], "H_SINGLETON")
    ck("S2g far above all points -> nearest is H_TENSOR",
       gf2_rule(2.00, se).split(" --")[0], "H_TENSOR")
    ck("S2h exact midpoint is undecidable at ANY se (by construction)",
       gf2_rule(0.445, 0.01).startswith("UNRESOLVED"), True)
    ck("S2i off-centre 0.40 is UNRESOLVED at the real se",
       gf2_rule(0.40, se).startswith("UNRESOLVED"), True)
    ck("S2j the SAME point resolves once se shrinks 20x",
       gf2_rule(0.40, se/20).split(" --")[0], "H_SINGLETON")

    print("S3 -- aw1's rule is three-way and both extremes are publishable")
    ck("S3a transfers",      _aw1_rule(0.60, 0.10).startswith("TRANSFERS"), True)
    ck("S3b does not",       _aw1_rule(0.02, 0.05).startswith("DOES NOT TRANSFER"), True)
    ck("S3c unresolved",     _aw1_rule(0.30, 0.20).startswith("UNRESOLVED"), True)
    ck("S3d wide CI at a big point-estimate is still UNRESOLVED",
       _aw1_rule(0.60, 0.40).startswith("UNRESOLVED"), True)

    print("S4 -- GATE 0 is per-seed and refuses in both directions")
    ck("S4a all clean",  gate0([("gf2-nf2-s0",0,0),("gf2-nf2-s1",0,0)])["verdict"], "CLEAN")
    ck("S4b one seed bound -> admissible, excluded",
       gate0([("gf2-nf2-s0",0.01,0),("gf2-nf2-s1",0,0)])["void"], False)
    ck("S4c two seeds of one arm bound -> VOID",
       gate0([("gf2-nf2-s0",0.01,0),("gf2-nf2-s1",0,0.02)])["void"], True)
    ck("S4d ceiling alone also counts",
       gate0([("gf2-nf2-s0",0,0.5),("gf2-nf2-s1",0,0.5)])["void"], True)

    print("S5 -- the admissibility gate uses `complete`, not just `window_ok`")
    ck("S5a truncated-but-window_ok is REFUSED",
       admissible({"window_ok":"1","complete":"0","plateau5":"85.228"}), False)
    ck("S5b complete-but-short-window is REFUSED",
       admissible({"window_ok":"0","complete":"1","plateau5":"90.0"}), False)
    ck("S5c both -> admitted",
       admissible({"window_ok":"1","complete":"1","plateau5":"92.5"}), True)
    ck("S5d unreadable plateau5 -> refused",
       admissible({"window_ok":"1","complete":"1","plateau5":""}), False)

    print("S6 -- constants match a live re-derivation from the CSV")
    rows = load()
    PRI = dict(network="ResNet18", dataset="CIFAR10", base="SGDm", meta="Lion",
               meta_stepsize="1e-4", alpha0="1e-3", gamma="1", augment="1",
               beta_clip="-15:-2.3026", epochs_requested="100")
    def cell(r): return all(r[k] == v for k, v in PRI.items())
    Ds, raw, sds = {}, {}, []
    for f in ("cc1", "mm1", "pp1", "gn1"):
        a = [num(r,"plateau5") for r in rows if admissible(r) and cell(r)
             and r["granularity"]=="chunk777" and family(r)==f]
        b = [num(r,"plateau5") for r in rows if admissible(r) and cell(r)
             and r["granularity"]=="nodewise" and family(r)==f]
        Ds[f] = round(st.mean(a)-st.mean(b), 3)
        raw[f] = st.mean(a)-st.mean(b)
        sds += [(len(a)-1, st.variance(a)), (len(b)-1, st.variance(b))]
    ck("S6a the four within-batch D's", Ds, D1_BY_BATCH)
    ck("S6b anchor = mean of them (unrounded)",
       round(st.mean(list(raw.values())), 4), round(D1_ANCHOR, 4))
    pooled = math.sqrt(sum(df*v for df, v in sds)/sum(df for df, _ in sds))
    ck("S6c pooled within-arm sd", round(pooled, 4), POOLED_SD)
    ck("S6d D is batch-transportable: sd(D) < seed-noise expectation",
       round(st.stdev(list(Ds.values())), 4) < round(pooled*math.sqrt(2/3), 4), True)
    ck("S6e base census over chunk*/nodewise1d rows is ONE base",
       sorted({r["base"] for r in rows
               if r["granularity"].startswith("chunk") or r["granularity"]=="nodewise1d"}),
       ["SGDm"])

    print()
    if fails:
        print(f"FAILED {len(fails)}: {fails}"); return 1
    print("ALL 30 CHECKS PASS -- cycle-88 scorers registered and validated.")
    return 0


def _power():
    print("gf1 (KILLED) -- its registered equivalence branch:")
    for k, v in gf1_power_is_zero().items():
        print(f"   {k:22s} {v}")
    print("\ngf2 (REGISTERED) -- separation of the three hypotheses on S:")
    se = POOLED_SD * math.sqrt(4.0/5)
    pts = sorted(GF2_POINTS.items(), key=lambda kv: kv[1])
    print(f"   se(S) at n=5 = {se:.4f}")
    for (k1,v1),(k2,v2) in zip(pts, pts[1:]):
        print(f"   {k1:12s} {v1:.2f}  ->  {k2:12s} {v2:.2f}   "
              f"separation {abs(v2-v1)/se:.2f} se")
    def phi(z): return 0.5*(1+math.erf(z/math.sqrt(2)))
    bnds = [ (pts[i][1]+pts[i+1][1])/2 for i in range(len(pts)-1) ]
    for i,(k,v) in enumerate(pts):
        lo = -9e9 if i==0 else bnds[i-1]
        hi =  9e9 if i==len(pts)-1 else bnds[i]
        p = phi((hi-v)/se) - phi((lo-v)/se)
        print(f"   P(correct | {k:12s} true) = {p*100:.1f}%")


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(_selftest())
    if "--power" in sys.argv: _power(); sys.exit(0)
    if "--score" in sys.argv:
        which = sys.argv[sys.argv.index("--score")+1]
        rows = load()
        print({"ub9": score_ub9, "aw1": score_aw1, "gf2": score_gf2}[which](rows))
        sys.exit(0)
    print(__doc__)
