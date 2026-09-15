#!/usr/bin/env python3
r"""cQ1_row24_falloff_score.py -- MASTER-TABLE ROW 24's FOUR FALLOFF NUMERALS,
RE-DERIVED ON plateau5 BY A COMMITTED SCORER.

ZERO GPU.  Nothing is submitted.  results/all_runs.csv is READ, never written.

=============================================================================
0.  WHY THIS FILE EXISTS
=============================================================================
MASTER-TABLE row 24 prints

    "Falloff above the peak: scalar 5.871 vs blk6 1.060, layerwise 1.801,
     nodewise 2.593 pp/decade."

CORRECTIONS 153 recorded that those four numerals DO NOT RE-DERIVE and that no
script in analysis/ computes them.  This file is the replacement instrument.
It is registered and COMMITTED BEFORE IT IS RUN (RULE 21, in the only form
available to a zero-GPU re-derivation -- see section 6).

=============================================================================
1.  PROVENANCE OF THE PRINTED FOUR -- ESTABLISHED BEFORE THIS FILE WAS WRITTEN
=============================================================================
The forensic pass is recorded here so the recipe below is not a guess.

(a) NO COMMITTED SCRIPT COMPUTES THEM.  `grep -rn '5\.871'` over the repo hits
    only docs/{MASTER-TABLE,FINDINGS,CORRECTIONS,STATUS}.md, CONTINUE-HERE.md,
    bin/c57_rsw_peak.sh (which QUOTES the row in its P2 pre-registration), and
    analysis/{cN1,cN2,cO1}_*.py (which quote it in comments).  No analysis/*.py
    that reads `meta_stepsize` computes a peak-relative slope in log10(ms):
    the only log10(ms) users are c65/c66/c67 (the field-vs-training `u` axis).
    `git log -S'5.871' --all -- docs/ analysis/*.py bin/*.sh` first introduces
    the row at 775eb6b (cycle 57, 2026-08-22), in docs/FINDINGS.md 57.2 and in
    bin/c57_rsw_peak.sh, in the SAME commit; that commit adds only
    analysis/c57_surface_truncation.py, which contains neither "decade" nor
    "falloff".  `git log --diff-filter=D` shows the repo has never deleted a
    .py.  CONCLUSION: the row was computed ad hoc in cycle 57 and the code was
    never committed.

(b) THE COLUMN.  results/all_runs.csv at 775eb6b HAS NO plateau5 COLUMN -- it
    ends at `ep_in_band_90`.  The four numerals therefore cannot rest on
    plateau5.  They rest on the k=20 `plateau` column, which is BANNED as
    primary.

(c) THE STRATUM.  FINDINGS 57.2's own table (scalar 90.190/90.699/91.555/
    92.231/88.743/87.758 at ms=1e-8/1e-5/3e-5/1e-4/3e-4/1e-3; layerwise
    90.081/90.874/91.658/92.795/91.885/91.097) re-derives EXACTLY, to the
    printed 3 dp, from the cycle-57 CSV under: ResNet18 / CIFAR10 / SGDm base /
    Lion meta / alpha0=1e-3 / AUGMENT=1 / beta_clip=-15:-2.3026 / hier empty /
    epochs_requested=100 / epochs_done>=100 / plateau>50, no run-name prefix
    filter.  The stratum in section 2 is that stratum.

(d) THE FIT.  Enumerating every two-point chord, every OLS window and every
    segment average over the cycle-57 `plateau` surface, exactly ONE recipe
    reproduces the printed row:

        falloff(arm) = MEAN over all rungs strictly above the arm's peak of
                       (peak_mean - rung_mean) / log10(ms_rung / ms_peak)

    i.e. the AVERAGE CHORD SLOPE from the peak to each higher rung.  On the
    cycle-57 `plateau` surface it gives

        resnet18_blocks  1.0603   (printed 1.060, residual +0.0003)
        layerwise        1.8020   (printed 1.801, residual +0.0010)
        nodewise         2.5926   (printed 2.593, residual -0.0004)
        scalar           5.8919   (printed 5.871, residual +0.0209)  <-- ODD

    and the scalar entry is the odd one out by 20x the other three residuals.
    On `best_test`, the SAME recipe over the SAME stratum gives scalar 5.8706
    (residual -0.0004) while blk6/layerwise/nodewise go to 1.2378 / 1.8489 /
    0.9295 (residuals +0.178 / +0.048 / -1.664).  THE MOST PLAUSIBLE READING,
    stated as such and not as certainty: row 24's four numerals are a
    MIXED-COLUMN row -- three cells from `plateau`, the scalar cell from
    `best_test`.  An alternative reading is a single hand-arithmetic slip in
    the scalar cell; either way the row is off-metric and the scalar entry,
    which is the entry that CARRIES THE SENTENCE, is the one that is wrong.

    INDEPENDENT CORROBORATION OF THE RECIPE: FINDINGS 58.8(a) re-derives the
    nodewise entry as `(92.547-91.310)/log10(3) = 2.592` -- the single chord
    that recipe reduces to when an arm has exactly one rung above its peak.

(e) THE TWO RECIPES CORRECTIONS 153 TRIED AND REJECTED.  "peak-to-next-rung on
    plateau" (3.778/1.119/1.942/0.285) and "peak-to-highest-rung on plateau5"
    (4.427/0.761/1.693/1.125) are this file's SECONDARY_1 and SECONDARY_2.
    They are reported, and they are NOT the recipe the printed row used.

=============================================================================
2.  THE PRE-REGISTRATION.  Everything below is fixed before the scorer runs.
=============================================================================

STRATUM (the canonical box; identical to cO1_cfr1_score.py's `_in_stratum`
except that granularity is not restricted and no batch prefix is excluded):

    network=ResNet18, dataset=CIFAR10, base=SGDm, meta=Lion,
    alpha0=1e-3, gamma=1, batch_size=100, augment=1,
    beta_clip=-15:-2.3026, hier empty (plain -- no hierarchical operator),
    epochs_requested=100 AND epochs_done=100, collapsed=0, superseded!=1,
    window_ok=1 AND complete=1 (asserted, not assumed -- METRIC RULES),
    plateau5 > 50 (57.2's own sanity floor, applied to the primary column).

ARMS (row 24's own four, in row 24's own order):
    scalar, resnet18_blocks, layerwise, nodewise.
    weightwise is reported DESCRIPTIVELY and is not part of any test.

MS RUNGS.  PRIMARY = the canonical ladder
    1e-8, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2
matched on the CSV's meta_stepsize STRING, so 3.947e-4 / 1.277e-4 / 3.307e-7 /
1.520e-5 / 5e-4 / 2e-4 / 3e-2 are EXCLUDED.  Registered reason: 57.2's table is
the canonical ladder, and admitting off-ladder rungs changes which rung is
"above the peak" and so changes the estimand rather than the estimate.
SECONDARY_ALL = every rung present in the stratum, reported alongside; the
ORDERING verdict must hold on BOTH to be called ROBUST.

THE FIT -- what "per decade" means, precisely:
    cell mean  m(arm, ms) = arithmetic mean of plateau5 over surviving runs.
    peak       ms*(arm)   = argmax_ms m(arm, ms).  TIES: the SMALLEST ms wins.
    chord_j               = (m(peak) - m(ms_j)) / log10(ms_j / ms*)  for every
                            rung ms_j > ms* present for that arm.
    falloff(arm)          = arithmetic mean of the chord_j.
    An arm with NO rung above its peak has falloff UNDEFINED and is reported
    as UNDEFINED -- never as 0, never dropped silently.
    MISSING RUNGS are simply absent from that arm's set; nothing is imputed
    and no rung is borrowed from another arm.

NOISE.  SE(falloff) is propagated from the cell means:
    falloff = m(peak) * (1/J) * sum_j 1/d_j  -  (1/J) * sum_j m_j / d_j,
    Var = Var(m_peak) * [(1/J) sum_j 1/d_j]^2 + sum_j Var(m_j) / (J d_j)^2,
    Var(m) = sigma_w^2 / n, with sigma_w the POOLED WITHIN-CELL sd of plateau5
    over every cell of the stratum with n >= 2, re-derived at run time.
    n=1 cells use the same sigma_w; every arm whose peak or whose chords touch
    an n=1 cell is flagged N1-DEPENDENT and its numeral carries that flag.

TEST A -- THE NUMERALS (per arm, independent of Test B):
    AGREE with the printed numeral iff
        |recomputed - printed| <= 2 * SE(recomputed)   AND
        |recomputed - printed| <= 0.50 pp/decade.
    Both conditions are required, so a large SE cannot manufacture agreement
    and a tiny SE cannot manufacture disagreement over a scientifically
    irrelevant gap.  0.50 pp/decade is registered as the "does the sentence
    change" scale: it is under a fifth of the smallest printed numeral and
    under a tenth of the largest.
    Anything else is DISAGREE.

TEST B -- THE ORDERING (registered SEPARATELY; it can pass while every
numeral in Test A fails, and that is the expected outcome):
    The SENTENCE row 24 asserts is "partitioning buys tolerance to an
    over-large meta-stepsize", i.e.
        falloff(scalar) > falloff(a) for EVERY partitioned arm a with a
        defined falloff.
    B-POINT   holds iff that is true of the point estimates.
    B-2SE     holds iff, in addition, scalar - a > 2 * SE(difference) for
              every such a (independent cells, so Var adds).
    B-ALT (the peak-location-free restatement, after FINDINGS 58.8): the loss
          each arm takes at the FIXED over-large ms=1e-3 relative to its own
          tuned peak, loss(a) = m(a, peak) - m(a, 1e-3); B-ALT holds iff
          loss(scalar) > loss(a) for every partitioned arm.

WHAT HAPPENS IF THE ORDERING DOES NOT REPRODUCE -- registered in advance:
    (i)   B-POINT holds, Test A fails on some arms
          -> the SENTENCE survives; the four numerals are STALE and must be
             replaced by this file's plateau5 values.  Row 24 stays CONFIRMED
             on its sentence; its numerals are corrected.  RECOMMENDED, NOT
             APPLIED: this scorer does not edit MASTER-TABLE.
    (ii)  B-POINT holds but B-2SE fails for some arm
          -> that arm is reported UNRESOLVED.  It is NOT rounded into either
             branch and the sentence is reported as holding only over the arms
             that resolve.
    (iii) B-POINT fails for any arm, but B-ALT holds
          -> the sentence survives ONLY in its peak-location-free form; the
             recommendation is that row 24's claim be restated as the loss at
             a fixed over-large ms and the falloff numerals be struck.
    (iv)  B-POINT and B-ALT both fail
          -> row 24's sentence is REFUTED AS PRINTED.  Recommend the verdict
             move off CONFIRMED and hand it to the landing agent.  This file
             still does not edit MASTER-TABLE.

REPORTED ALONGSIDE, per discipline: TRAIN (final_train) cell means at every
arm and rung, and the per-cell n and sd.

=============================================================================
3.  WHAT THIS FILE MAY NOT BE READ AS SAYING
=============================================================================
* It re-derives ROW 24's FALLOFF ROW ONLY.  It says nothing about CORRECTIONS
  153's clamp scope: the -15 floor census still binds ms=3e-4 (100%, n=15),
  blk6 (68.8%) and nodewise, so two of the four falloff arms and the rung that
  carries 94.7% of the ms=1e-3 gap are STILL NOT DISCHARGED.  A recomputed
  falloff is a recomputation of the SAME possibly-clamped surface on a better
  column; it is NOT a release of the box.
* The nodewise arm is n=1 at some rungs and its PEAK LOCATION was already
  recorded as unresolved (FINDINGS 58.8: k=20 puts it at 1e-3, k=5 at 3e-4,
  0.098 pp between two single runs).  Any nodewise numeral this file prints
  inherits that.
* This is a REANALYSIS of runs that all predate it.  It is P-class in the
  sense that the surface has been looked at before -- by 42.2, 57.2 and 58.8.
  What is NEW and blind is only the column swap (plateau -> plateau5) under a
  recipe fixed by (d) above, and the ordering test, which no document has
  stated as a testable proposition before.

=============================================================================
4.  USAGE
=============================================================================
    python3 analysis/cQ1_row24_falloff_score.py --selftest
    python3 analysis/cQ1_row24_falloff_score.py --score
    python3 analysis/cQ1_row24_falloff_score.py            # both

=============================================================================
5.  RULE 16
=============================================================================
Run UNEDITED.  If it is broken, FREEZE it and register a NEW file (cQ2),
precedent cN1/cN2 at CORRECTIONS 149.  Documented arguments are not edits.

=============================================================================
6.  RULE 21, IN THE ONLY FORM A ZERO-GPU RE-DERIVATION ADMITS
=============================================================================
RULE 21's usual proof -- scorer commit time strictly before the earliest sacct
Submit of its own runs -- CANNOT be constructed here, because this scorer has
NO runs of its own: every row it reads was written weeks before it, and the
surface it scores has been looked at by three prior findings.  Claiming a
RULE 21 margin against those runs would be a lie by construction.
WHAT IS PROVEN INSTEAD, and what must be quoted: this file is committed BEFORE
ITS FIRST EXECUTION, and the margin reported is `git commit time` vs the
wall-clock of the first invocation.  That is a weaker guarantee than RULE 21
and it is labelled as such wherever this file's numbers are used.
"""

import argparse
import collections
import csv
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(os.path.dirname(HERE), "results", "all_runs.csv")

# --- the stratum -------------------------------------------------------------
NET, DSET = "ResNet18", "CIFAR10"
BASE_ALG, META_ALG = "SGDm", "Lion"
ALPHA0, GAMMA, BATCH, AUG = "1e-3", "1", "100", "1"
CLIP = "-15:-2.3026"
EPOCHS = 100
P5_FLOOR = 50.0

ARMS = ["scalar", "resnet18_blocks", "layerwise", "nodewise"]
DESCRIPTIVE_ARMS = ["weightwise"]

# the canonical ladder, matched on the CSV's meta_stepsize STRING
LADDER = ["1e-8", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]

# --- row 24 as printed -------------------------------------------------------
PRINTED = {"scalar": 5.871, "resnet18_blocks": 1.060,
           "layerwise": 1.801, "nodewise": 2.593}

# --- Test A bars -------------------------------------------------------------
ABS_BAR = 0.50          # pp/decade; the "does the sentence change" scale
SE_MULT = 2.0

# --- the cycle-57 provenance constants --------------------------------------
# Cell means of the BANNED k=20 `plateau` column on the 57.2 stratum, computed
# from `git show 775eb6b:results/all_runs.csv` (the 1707-row corpus as it stood
# when FINDINGS 57.2 was written).  FROZEN INPUTS TO --selftest, never used by
# --score.
C57_PLATEAU = {
    "scalar":          {"1e-8": 90.1897, "1e-5": 90.6994, "3e-5": 91.5550,
                        "1e-4": 92.2314, "3e-4": 88.7434, "1e-3": 87.7581},
    "resnet18_blocks": {"3e-5": 91.4440, "1e-4": 92.5805, "3e-4": 91.9390,
                        "1e-3": 91.5602, "3e-3": 91.3750},
    "layerwise":       {"1e-8": 90.0810, "1e-5": 90.8744, "3e-5": 91.6576,
                        "1e-4": 92.7948, "3e-4": 91.8852, "1e-3": 91.0971},
    "nodewise":        {"1e-4": 92.0880, "3e-4": 92.5150, "1e-3": 92.5470,
                        "3e-3": 91.3100},
}
# the same stratum on `best_test`, cycle-57 corpus -- the scalar cell's likely
# actual source (section 1(d)).
C57_BEST = {
    "scalar":          {"1e-8": 91.3567, "1e-5": 91.8280, "3e-5": 92.1400,
                        "1e-4": 92.4200, "3e-4": 88.9160, "1e-3": 88.0229},
}


# =============================================================================
# corpus
# =============================================================================
def _rows(path=CSV):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if (r.get("superseded") or "0") != "1"]


def in_stratum(r):
    """The canonical box row 24's surface was measured in."""
    return (r["network"] == NET and r["dataset"] == DSET
            and r["base"] == BASE_ALG and r["meta"] == META_ALG
            and r["alpha0"] == ALPHA0 and r["gamma"] == GAMMA
            and r["batch_size"] == BATCH and r["augment"] == AUG
            and r["beta_clip"] == CLIP
            and not (r.get("hier") or "").strip()
            and r["epochs_requested"] == str(EPOCHS)
            and r["epochs_done"] == str(EPOCHS)
            and r["collapsed"] == "0"
            and r.get("window_ok") == "1" and r.get("complete") == "1")


def surface(path=CSV, ladder_only=True, col="plateau5"):
    """{(arm, ms_string): [values]} plus the matching train values."""
    test = collections.defaultdict(list)
    train = collections.defaultdict(list)
    for r in _rows(path):
        if not in_stratum(r):
            continue
        ms = r["meta_stepsize"]
        if ladder_only and ms not in LADDER:
            continue
        try:
            v = float(r[col])
        except (TypeError, ValueError):
            continue
        if v <= P5_FLOOR:
            continue
        test[(r["granularity"], ms)].append(v)
        try:
            train[(r["granularity"], ms)].append(float(r["final_train"]))
        except (TypeError, ValueError):
            pass
    return test, train


def sigma_within(test_cells):
    """Pooled within-cell sd of the primary column over the stratum."""
    ss, df, nc = 0.0, 0, 0
    for v in test_cells.values():
        if len(v) >= 2:
            m = statistics.mean(v)
            ss += sum((x - m) ** 2 for x in v)
            df += len(v) - 1
            nc += 1
    return (math.sqrt(ss / df) if df else float("nan")), df, nc


# =============================================================================
# the fit
# =============================================================================
def ms_val(s):
    return float(s)


def peak_of(means):
    """argmax of the cell mean; TIES -> the SMALLEST ms."""
    best = None
    for ms in sorted(means, key=ms_val):
        if best is None or means[ms] > means[best] + 1e-12:
            best = ms
    return best


def falloff(means, counts, sigma):
    """MEAN CHORD SLOPE above the peak.  Returns a dict, or None if UNDEFINED."""
    if not means:
        return None
    pk = peak_of(means)
    above = [ms for ms in sorted(means, key=ms_val) if ms_val(ms) > ms_val(pk)]
    if not above:
        return {"peak": pk, "peak_mean": means[pk], "above": [], "value": None,
                "se": None, "chords": [], "n1": counts.get(pk, 0) < 2}
    d = [math.log10(ms_val(ms) / ms_val(pk)) for ms in above]
    chords = [(means[pk] - means[ms]) / dj for ms, dj in zip(above, d)]
    J = len(above)
    val = sum(chords) / J
    a = sum(1.0 / dj for dj in d) / J
    var = (sigma ** 2 / max(counts[pk], 1)) * a * a
    for ms, dj in zip(above, d):
        var += (sigma ** 2 / max(counts[ms], 1)) / (J * dj) ** 2
    n1 = counts.get(pk, 0) < 2 or any(counts.get(ms, 0) < 2 for ms in above)
    return {"peak": pk, "peak_mean": means[pk], "above": above, "value": val,
            "se": math.sqrt(var), "chords": list(zip(above, chords)), "n1": n1}


def secondary_next(means):
    """SECONDARY_1: peak-to-NEXT-rung chord."""
    if not means:
        return None
    pk = peak_of(means)
    above = [ms for ms in sorted(means, key=ms_val) if ms_val(ms) > ms_val(pk)]
    if not above:
        return None
    nxt = above[0]
    return (means[pk] - means[nxt]) / math.log10(ms_val(nxt) / ms_val(pk))


def secondary_top(means):
    """SECONDARY_2: peak-to-HIGHEST-rung chord."""
    if not means:
        return None
    pk = peak_of(means)
    above = [ms for ms in sorted(means, key=ms_val) if ms_val(ms) > ms_val(pk)]
    if not above:
        return None
    top = above[-1]
    return (means[pk] - means[top]) / math.log10(ms_val(top) / ms_val(pk))


def cellstats(test, arm):
    means, counts, sds = {}, {}, {}
    for (a, ms), v in test.items():
        if a != arm:
            continue
        means[ms] = statistics.mean(v)
        counts[ms] = len(v)
        sds[ms] = statistics.stdev(v) if len(v) > 1 else float("nan")
    return means, counts, sds


# =============================================================================
# selftest
# =============================================================================
def selftest():
    P = print
    ok = [0, 0]

    def ck(cond, msg):
        ok[1] += 1
        if cond:
            ok[0] += 1
            P("    ok   %s" % msg)
        else:
            P("    FAIL %s" % msg)

    P("=" * 78)
    P("cQ1 --selftest")
    P("=" * 78)

    P("\nT1  the recipe reproduces the PRINTED row on the cycle-57 `plateau` surface")
    got = {}
    for arm, means in C57_PLATEAU.items():
        counts = dict((ms, 1) for ms in means)
        f = falloff(means, counts, 0.0)
        got[arm] = f["value"]
        P("      %-16s peak %.4f @ %-5s  falloff %.4f  printed %.3f  resid %+.4f"
          % (arm, f["peak_mean"], f["peak"], f["value"], PRINTED[arm],
             f["value"] - PRINTED[arm]))
    ck(abs(got["resnet18_blocks"] - 1.060) < 0.002, "blk6 reproduces 1.060")
    ck(abs(got["layerwise"] - 1.801) < 0.002, "layerwise reproduces 1.801")
    ck(abs(got["nodewise"] - 2.593) < 0.002, "nodewise reproduces 2.593")
    ck(abs(got["scalar"] - 5.871) > 0.010,
       "scalar does NOT reproduce 5.871 on `plateau` (resid %+.4f) -- the row's"
       " scalar cell is off-column" % (got["scalar"] - 5.871))

    P("\nT2  the SAME recipe on cycle-57 `best_test` reproduces the scalar cell")
    counts = dict((ms, 1) for ms in C57_BEST["scalar"])
    fb = falloff(C57_BEST["scalar"], counts, 0.0)
    P("      scalar/best_test falloff %.4f  printed %.3f  resid %+.4f"
      % (fb["value"], PRINTED["scalar"], fb["value"] - PRINTED["scalar"]))
    ck(abs(fb["value"] - 5.871) < 0.002,
       "scalar reproduces 5.871 on best_test -- row 24 is a MIXED-COLUMN row")

    P("\nT3  FINDINGS 58.8(a)'s own re-derivation is the one-chord special case")
    v = (92.547 - 91.310) / math.log10(3.0)
    P("      (92.547-91.310)/log10(3) = %.4f  (58.8 published 2.592)" % v)
    ck(abs(v - 2.592) < 0.001, "58.8's nodewise chord")

    P("\nT4  peak selection, ties and UNDEFINED")
    ck(peak_of({"1e-4": 1.0, "1e-3": 2.0}) == "1e-3", "argmax picks the max")
    ck(peak_of({"1e-4": 2.0, "1e-3": 2.0}) == "1e-4", "ties -> smallest ms")
    ck(falloff({"1e-3": 90.0}, {"1e-3": 3}, 0.2)["value"] is None,
       "no rung above the peak -> falloff UNDEFINED, not 0")
    ck(falloff({}, {}, 0.2) is None, "empty arm -> None")

    P("\nT5  the chord arithmetic, on numbers with a closed form")
    m = {"1e-4": 100.0, "1e-3": 90.0, "1e-2": 70.0}
    f = falloff(m, {"1e-4": 1, "1e-3": 1, "1e-2": 1}, 0.0)
    ck(abs(f["value"] - (10.0 + 15.0) / 2) < 1e-9,
       "mean of chords 10.0 and 15.0 = 12.5 (got %.4f)" % f["value"])
    ck(abs(secondary_next(m) - 10.0) < 1e-9, "SECONDARY_1 = 10.0")
    ck(abs(secondary_top(m) - 15.0) < 1e-9, "SECONDARY_2 = 15.0")

    P("\nT6  SE propagation degenerates correctly")
    f0 = falloff(m, {"1e-4": 1, "1e-3": 1, "1e-2": 1}, 0.0)
    ck(f0["se"] == 0.0, "sigma=0 -> SE=0")
    f1 = falloff(m, {"1e-4": 1, "1e-3": 1, "1e-2": 1}, 1.0)
    f4 = falloff(m, {"1e-4": 4, "1e-3": 4, "1e-2": 4}, 1.0)
    ck(abs(f1["se"] / f4["se"] - 2.0) < 1e-9, "SE scales as 1/sqrt(n)")

    P("\nT7  the stratum predicate is exclusive in BOTH directions")
    base = {"network": NET, "dataset": DSET, "base": BASE_ALG, "meta": META_ALG,
            "alpha0": ALPHA0, "gamma": GAMMA, "batch_size": BATCH, "augment": AUG,
            "beta_clip": CLIP, "hier": "", "epochs_requested": "100",
            "epochs_done": "100", "collapsed": "0", "window_ok": "1",
            "complete": "1"}
    ck(in_stratum(dict(base)), "the canonical row is IN")
    for k, bad in [("beta_clip", "-80:-2.3026"), ("alpha0", "1e-6"),
                   ("hier", "shrink"), ("epochs_done", "76"),
                   ("collapsed", "1"), ("augment", "0"), ("window_ok", "0"),
                   ("complete", "0"), ("dataset", "CIFAR100")]:
        b = dict(base)
        b[k] = bad
        ck(not in_stratum(b), "%s=%s is OUT" % (k, bad))

    P("\nT8  the ladder excludes the off-ladder rungs by construction")
    for bad in ["3.947e-4", "1.277e-4", "3.307e-7", "1.520e-5", "5e-4",
                "2e-4", "3e-2"]:
        ck(bad not in LADDER, "%s is off-ladder" % bad)
    for good in ["1e-4", "3e-4", "1e-3", "3e-3"]:
        ck(good in LADDER, "%s is on the ladder" % good)

    P("\nT9  the corpus is readable and the stratum is non-empty")
    t, _ = surface()
    ck(len(t) > 0, "stratum has cells (%d)" % len(t))
    s, df, nc = sigma_within(t)
    ck(df > 20, "pooled sigma_w has df=%d over %d cells (sigma=%.4f)" % (df, nc, s))

    P("\n%d/%d selftests passed" % (ok[0], ok[1]))
    return ok[0] == ok[1]


# =============================================================================
# score
# =============================================================================
def score():
    P = print
    P("=" * 78)
    P("cQ1 --score   MASTER-TABLE row 24's falloff row, re-derived on plateau5")
    P("=" * 78)

    for ladder_only in (True, False):
        tag = "PRIMARY (canonical ladder)" if ladder_only else "SECONDARY_ALL (every rung)"
        test, train = surface(ladder_only=ladder_only)
        sigma, df, nc = sigma_within(test)
        P("\n" + "-" * 78)
        P("%s" % tag)
        P("-" * 78)
        P("pooled within-cell sigma_w(plateau5) = %.4f  (df %d, %d cells)"
          % (sigma, df, nc))

        P("\n-- THE SURFACE (plateau5 PRIMARY; final_train alongside) --")
        P("%-16s %-6s %8s %7s %6s %9s" % ("arm", "ms", "plateau5", "sd", "n", "train"))
        for arm in ARMS + DESCRIPTIVE_ARMS:
            means, counts, sds = cellstats(test, arm)
            for ms in sorted(means, key=ms_val):
                tr = train.get((arm, ms), [])
                P("%-16s %-6s %8.4f %7s %6d %9s"
                  % (arm, ms, means[ms],
                     ("%.4f" % sds[ms]) if counts[ms] > 1 else "  --",
                     counts[ms],
                     ("%.3f" % statistics.mean(tr)) if tr else "   --"))

        res = {}
        P("\n-- THE FALLOFF (mean chord slope above the peak) --")
        for arm in ARMS:
            means, counts, sds = cellstats(test, arm)
            f = falloff(means, counts, sigma)
            res[arm] = f
            if f is None:
                P("  %-16s NO CELLS" % arm)
                continue
            if f["value"] is None:
                P("  %-16s peak %.4f @ %-5s -- NO RUNG ABOVE THE PEAK -> UNDEFINED"
                  % (arm, f["peak_mean"], f["peak"]))
                continue
            P("  %-16s peak %.4f @ %-5s (n=%d)  falloff %7.4f +- %.4f%s"
              % (arm, f["peak_mean"], f["peak"], counts[f["peak"]],
                 f["value"], f["se"], "   [N1-DEPENDENT]" if f["n1"] else ""))
            for ms, c in f["chords"]:
                P("        chord peak->%-5s (n=%d, %.4f dec): %7.4f"
                  % (ms, counts[ms], math.log10(ms_val(ms) / ms_val(f["peak"])), c))
            P("        SECONDARY_1 peak->next   %s"
              % ("%.4f" % secondary_next(means)))
            P("        SECONDARY_2 peak->top    %s"
              % ("%.4f" % secondary_top(means)))

        P("\n-- TEST A: THE NUMERALS  (bar: |d| <= 2*SE AND |d| <= %.2f pp/dec) --"
          % ABS_BAR)
        P("%-16s %9s %9s %9s %8s   %s"
          % ("arm", "printed", "plateau5", "delta", "2*SE", "verdict"))
        a_pass = 0
        a_n = 0
        for arm in ARMS:
            f = res.get(arm)
            if f is None or f["value"] is None:
                P("%-16s %9.3f %9s %9s %8s   UNDEFINED"
                  % (arm, PRINTED[arm], "--", "--", "--"))
                continue
            a_n += 1
            d = f["value"] - PRINTED[arm]
            v = (abs(d) <= SE_MULT * f["se"]) and (abs(d) <= ABS_BAR)
            a_pass += 1 if v else 0
            P("%-16s %9.3f %9.4f %+9.4f %8.4f   %s%s"
              % (arm, PRINTED[arm], f["value"], d, SE_MULT * f["se"],
                 "AGREE" if v else "DISAGREE",
                 "  [N1]" if f["n1"] else ""))
        P("    -> Test A: %d of %d arms AGREE with the printed numeral" % (a_pass, a_n))

        P("\n-- TEST B: THE ORDERING  (scalar's falloff EXCEEDS every partition's) --")
        sc = res.get("scalar")
        if sc is None or sc["value"] is None:
            P("    scalar UNDEFINED -- Test B cannot be run")
        else:
            bp, b2 = True, True
            for arm in ARMS:
                if arm == "scalar":
                    continue
                f = res.get(arm)
                if f is None or f["value"] is None:
                    P("    %-16s UNDEFINED -- excluded from Test B" % arm)
                    continue
                diff = sc["value"] - f["value"]
                se = math.sqrt(sc["se"] ** 2 + f["se"] ** 2)
                point = diff > 0
                resolved = diff > SE_MULT * se
                bp = bp and point
                b2 = b2 and resolved
                P("    scalar - %-16s = %+8.4f  (2*SE %.4f)  %s / %s"
                  % (arm, diff, SE_MULT * se,
                     "POINT-OK" if point else "POINT-FAILS",
                     "RESOLVED" if resolved else "UNRESOLVED"))
            P("    -> B-POINT %s ; B-2SE %s"
              % ("HOLDS" if bp else "FAILS", "HOLDS" if b2 else "FAILS"))

        P("\n-- TEST B-ALT: loss at the FIXED over-large ms=1e-3 vs own peak --")
        losses = {}
        for arm in ARMS:
            means, counts, sds = cellstats(test, arm)
            if not means or "1e-3" not in means:
                P("    %-16s no ms=1e-3 cell -- excluded" % arm)
                continue
            pk = peak_of(means)
            losses[arm] = means[pk] - means["1e-3"]
            P("    %-16s peak %.4f @ %-5s  ms1e-3 %.4f (n=%d)  loss %7.4f"
              % (arm, means[pk], pk, means["1e-3"], counts["1e-3"], losses[arm]))
        if "scalar" in losses:
            balt = all(losses["scalar"] > v for a, v in losses.items() if a != "scalar")
            P("    -> B-ALT %s" % ("HOLDS" if balt else "FAILS"))

    P("\n" + "=" * 78)
    P("REMINDER: this file does NOT edit docs/MASTER-TABLE.md.  The recommended")
    P("replacement and its provenance go in the report; the landing agent decides.")
    P("=" * 78)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    run_all = not (a.selftest or a.score)
    good = True
    if a.selftest or run_all:
        good = selftest() and good
    if a.score or run_all:
        good = score() and good
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
