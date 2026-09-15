#!/usr/bin/env python3
r"""c66_trajectory_invariance.py -- IS THE FIELD'S SCALE A PROPERTY OF THE FIELD, OR OF THE
TRAINING GRANULARITY THAT WAS ALLOWED TO ADAPT?

WHY THIS EXISTS
---------------
CORRECTIONS 94.7 closed cycle 65 by naming its own largest caveat, and named it as
UNTESTABLE:

    "the per-coordinate field is observable ONLY under weightwise training -- coarse probes
     store per-GROUP counts (`lay` n_tot 62, `node` 14,600, `blk6` 6) against weightwise
     11,220,132.  So every `E` in cycles 62-65 is measured in the weightwise arm ... The
     comparison assumes the field's SHAPE is not created by the training granularity; **that
     assumption is untested and untestable with recorded data**, and it is now the largest
     caveat in direction C."

The instrument limit in the first half is CORRECT and is re-verified here (selftest T6 reads
the actual `n_tot` out of the coarse probes' own json).  **The last clause is wrong**, and
this module is the correction.  It is wrong for a reason that is visible in the SOURCE, not
in the data:

THE PREMISE, WHICH IS A CODE IDENTITY AND NOT AN EMPIRICAL CLAIM
-----------------------------------------------------------------
In `HF_patched.py`, `meta='fixed'` binds `self.meta_update = self.no_meta_update`, and
`no_meta_update` is `return None`.  `init_meta` initialises `self.beta` to
`log(alpha0) * ones(...)` at EVERY `stepsize_type` -- scalar, layerwise, nodewise,
weightwise -- so beta starts UNIFORM and, with `meta='fixed'`, never moves.  Therefore
`alpha = beta_to_alpha(beta)` is the same constant `alpha0` for every coordinate at every
granularity, and `base_update(net, g)` receives an identical step size in all four arms.
`block_product`'s output (`HtT_gradft`) is consumed by exactly two callees: `meta_update`
(a no-op here) and `_probe` (read-only).

    => WITH BETA FROZEN, THE GRANULARITY DOES NOT TOUCH THE TRAJECTORY AT ALL.
       A frozen weightwise run and a frozen layerwise run traverse the SAME trajectory.

So a frozen WEIGHTWISE probe records the per-coordinate field along the very trajectory a
frozen LAYERWISE (or nodewise, or scalar) run would traverse.  For the frozen stratum the
instrument limit does not bind, and 94.7's assumption is directly testable on data already
on this Mac.  Selftests T1 (source) and T7 (the CSV's own frozen granularity span) assert
both halves of the premise rather than asking the reader to accept it.

WHAT IS THEREFORE COMPARABLE, AND WHAT IS NOT
----------------------------------------------
The 19-arm corpus of `results/c63_uladder.json` contains FOUR trajectory classes, all
probed per-coordinate, all at a 20-epoch budget:

    frozen        beta never moves; granularity is a no-op BY CONSTRUCTION   (8 arms)
    free          Lion meta, ms=1e-3, per-coordinate adaptation ON           (9 arms)
    EXC ms=1e-2   10x meta stepsize -- much stronger adaptation              (1 arm)
    EXC AdamW     a different BASE optimizer entirely                        (1 arm)

These span a wide range of training outcome, which is the point: if the field's scale were
manufactured by the optimiser's own adaptation, four optimisers landing 20+ pp apart should
not agree about where it sits.

WHAT THIS DOES NOT DO, STATED BEFORE THE RESULT (STANDING RULE 10)
-------------------------------------------------------------------
No arm here has beta partitioned COARSELY *and freely*.  The free-layerwise trajectory is
still unobserved at coordinate resolution and only PROBE7 (CORRECTIONS 94.12) can reach it.
This module therefore CANNOT discharge 94.7 in full and never claims to.  What it can do is
replace "untested and untestable" with a measured bound, and say exactly which single cell
is left.

PRE-REGISTRATION, WITH ITS PROVENANCE DECLARED ARM BY ARM
----------------------------------------------------------
HONESTY DECLARATION, because it changes what these numbers are worth.  Before writing this
block I had read `results/c65_field_vs_training.txt` PART C, which prints `u*_E` and `E*_pp`
per arm WITH a frozen/free stratum column, for the 17 CLEAN arms.  So:

  * A1 and A3 below are POST-HOC-INFORMED.  They are confirmatory restatements of a table I
    had already seen, and they are labelled as such in the output.  They are reported
    because they are the direct answer to 94.7, not because they are evidence in their own
    right.
  * B1, B2, B4 are BLIND: the full 17-rung ladder SHAPE, the log-slope against accuracy, and
    the frozen/free amplitude ratio are printed nowhere I have read, and c65's PART C prints
    only the peak rung.
  * B3 is BLIND for its scored quantity: c65 EXCLUDES the two exception arms from PART C
    (`if arm["exc"]: continue`), so their `u*` had never been printed in any output I read.

A1  LOCATION INVARIANCE, FROZEN vs FREE  [post-hoc-informed]
    Per family, the geometric-mean `u*` of the frozen arms and of the free arms differ by at
    most ONE ladder rung (factor 2): |log2(u*_frozen / u*_free)| <= 1.0 in 4 of 4 families.
    REFUTATION: >= 2 families differ by more than one rung.

A3  POOLED LOCATION SPREAD  [post-hoc-informed]
    Over all 17 clean arms, every arm's `u*` lies within one rung of the corpus geometric
    mean.  REFUTATION: any clean arm is more than 2 rungs away.

B1  FULL-LADDER SHAPE INVARIANCE  [BLIND]
    Normalise each arm's ladder to its own peak, e(u) = E(u)/E*, then average within
    stratum.  The frozen and free mean curves agree at every rung:
        REGISTERED: max_u |e_frozen(u) - e_free(u)| <= 0.25.
        REFUTATION: the gap exceeds 0.50 at any rung.
    B1 is much stronger than A1: two curves can share an argmax and be shaped nothing alike.

B2  ACCURACY REGRESSION, WITHIN ResNet18  [BLIND]
    r18 is the only family with four trajectory classes probed at coordinate resolution.
    Regress log2(u*) on the arm's OWN 20-epoch plateau (its own run, not a cell mean):
        REGISTERED: |slope| <= 0.10 rung/pp AND the peak span over the arms is <= 1 rung.
        REFUTATION: |slope| >= 0.25 rung/pp.

B3  THE TWO MOST DIFFERENT TRAJECTORIES IN THE CORPUS  [BLIND]
    `ml5/probe_w_m2_s0` (ms=1e-2) and `bo6/probe_w_adw_s0` (AdamW base) are FINDINGS 59.4's
    exception set.  Per CORRECTIONS 62 they are NEVER POOLED with the clean arms and ms=1e-2
    stays unquotable as an accuracy claim; they are scored here SEPARATELY and only for
    LOCATION, which is a statement about the field and not about their accuracy.
        REGISTERED: both `u*` lie within 2 rungs of the clean-arm geometric mean.
        REFUTATION: either is more than 3 rungs away.

B4  AMPLITUDE IS *NOT* INVARIANT -- THE POWER CONTROL, AND IT CAN KILL A1/B1  [BLIND]
    If frozen and free fields were identical in every respect, then A1 and B1 would be
    vacuous: they would report that a manipulation which did nothing changed nothing.  So
    the amplitude must move even though the location does not:
        REGISTERED: geometric-mean E*_free / E*_frozen >= 1.5 over the four matched families.
        REFUTATION: the ratio is <= 1.2 -- in which case the frozen/free contrast is not a
        real manipulation of the field and A1/B1 ARE NOT INTERPRETABLE and must be withdrawn.
    B4 is the test this module can most easily fail, and it is registered for that reason.

B5  HOW MUCH OF THE GAP IS LEFT  [BLIND, descriptive -- reported, not scored]
    Locate the free-LAYERWISE r18 plateau (the training optimum of CORRECTIONS 94.2) inside
    the accuracy range spanned by the r18 probed arms.  If it falls INSIDE that range, the
    step from "what we measured" to "the layerwise trajectory" is an interpolation in
    outcome and the residual gap is specifically the PARTITION, not the outcome.

METRIC RULE: plateau (mean of last 5 epochs) only.  `best_test` is never read; selftest T9
asserts the string never appears on a scored path.
"""

import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BACKUP = os.path.dirname(ROOT)
sys.path.insert(0, HERE)

import c65_field_vs_training as C65          # noqa: E402  (loaders reused, never re-implemented)

ULADDER_PATH = os.path.join(ROOT, "results", "c63_uladder.json")
HF_PATH = os.path.join(ROOT, "patches", "HF_patched.py")

FROZEN_DIRS = ("probes_fz3", "probes_p5")    # c65's own rule, line-for-line
RUNG = math.log(2.0)                         # one ladder rung = factor 2 in u


# ======================================================================================
# loading
# ======================================================================================

def arm_to_run(arm_path):
    """`probes_<F>/probe_<x>` -> the CSV run name `<F>-<x with _ -> ->`.  Asserted by T2."""
    fam_dir, probe = arm_path.split("/")
    assert fam_dir.startswith("probes_") and probe.startswith("probe_"), arm_path
    return fam_dir[len("probes_"):] + "-" + probe[len("probe_"):].replace("_", "-")


def stratum(arm_path):
    return "frozen" if arm_path.split("/")[0] in FROZEN_DIRS else "free"


def load_arms():
    with open(ULADDER_PATH) as fh:
        arms = json.load(fh)["arms"]
    for a in arms:
        a["stratum"] = "EXC" if a["exc"] else stratum(a["arm"])
        a["run"] = arm_to_run(a["arm"])
    return arms


def peak(arm, unclamped_only=True):
    """Re-derive the peak from the arm's own rows.  Never trust the stored `u_peak`."""
    rows = [r for r in arm["rows"] if not (unclamped_only and r["clamped"])]
    if not rows:
        return None
    b = max(rows, key=lambda r: r["E_pp"])
    return b["u"], b["E_pp"], b["res_pp"], b["label"]


def norm_ladder(arm):
    """e(u) = E(u)/E*, on the common 17-rung grid, clamped rungs carried as-is."""
    _u, ep, _r, _l = peak(arm)
    return [r["E_pp"] / ep for r in arm["rows"]]


def geo(xs):
    return float(np.exp(np.mean(np.log(np.asarray(xs, dtype=float)))))


def run_plateau(rows_by_run, run):
    r = rows_by_run.get(run)
    return None if r is None else r["_plateau"]


# ======================================================================================
# selftests
# ======================================================================================

def selftests(arms, rows_by_run):
    ok, fails = 0, []

    def chk(cond, msg):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fails.append(msg)

    # ---- T1  THE PREMISE, READ OUT OF THE SOURCE ------------------------------------
    src = open(HF_PATH).read()
    chk("self.meta_update = self.no_meta_update" in src,
        "T1a meta='fixed' binds no_meta_update")
    nmu = src.split("def no_meta_update")[1].split("def ")[0]
    chk("return None" in nmu and "self.beta" not in nmu,
        "T1b no_meta_update returns None and never touches beta")
    chk(src.count("torch.log(torch.tensor(alpha0") >= 3,
        "T1c beta initialised to log(alpha0) uniformly at each granularity")
    # block_product's result must reach only meta_update and _probe
    step = src.split("def step(")[1].split("\n    def ")[0]
    chk("HtT_gradft = self.block_product" in step
        and "self.meta_update(HtT_gradft)" in step
        and "self._probe(HtT_gradft)" in step
        and "self.base_update(net,g)" in step
        and "base_update(net,g)" not in step.split("HtT_gradft = self.block_product")[1].split("self.base_update")[1],
        "T1d block_product feeds only meta_update and _probe; base_update takes (net,g)")

    # ---- T2  arm -> run mapping resolves for every arm ------------------------------
    miss = [a["arm"] for a in arms if a["run"] not in rows_by_run]
    chk(not miss, f"T2 every arm maps to a CSV run (missing: {miss})")

    # ---- T3  every mapped run is weightwise, 20 epochs -------------------------------
    bad = [(a["run"], rows_by_run[a["run"]]["granularity"],
            rows_by_run[a["run"]]["epochs_requested"])
           for a in arms if a["run"] in rows_by_run
           and (rows_by_run[a["run"]]["granularity"] != "weightwise"
                or rows_by_run[a["run"]]["epochs_requested"] != "20")]
    chk(not bad, f"T3 all probed arms are weightwise@20 ({bad})")

    # ---- T4  stratum from the run's own `meta` field, not from the directory ---------
    mism = [(a["arm"], a["stratum"], rows_by_run[a["run"]]["meta"])
            for a in arms if a["run"] in rows_by_run and not a["exc"]
            and ((rows_by_run[a["run"]]["meta"] == "fixed") != (a["stratum"] == "frozen"))]
    chk(not mism, f"T4 directory stratum agrees with the CSV `meta` field ({mism})")

    # ---- T5  common ladder grid, so rung-wise averaging is legal ---------------------
    grids = {tuple(r["label"] for r in a["rows"]) for a in arms}
    chk(len(grids) == 1, f"T5 one common u grid across all arms ({len(grids)})")
    chk(len(next(iter(grids))) == 17, "T5b grid has 17 rungs")

    # ---- T6  94.7's instrument limit is REAL: coarse probes have no coord resolution --
    ntot = {}
    for d, key in (("probes_bl5/probe_w_e40_s0", "weightwise"),
                   ("probes_bl5/probe_node_e40_s0", "nodewise"),
                   ("probes_bl5/probe_lay_e40_s0", "layerwise")):
        p = os.path.join(BACKUP, d, "neg_counts.json")
        if os.path.isfile(p):
            j = json.load(open(p))
            ntot[key] = (j["n_tot"], j["stepsize_type"])
    chk(len(ntot) == 3 and ntot["weightwise"][0] > 1e6
        and ntot["layerwise"][0] < 1e3 and ntot["nodewise"][0] < 1e5,
        f"T6 coarse probes really do store per-GROUP counts ({ntot})")

    # ---- T7  the premise's empirical half: frozen granularity span at the noise floor -
    spans = {"frozen": [], "free": []}
    for k, cell in C65.cells(C65.clean_rows(C65.load_rows()), "20").items():
        if C65.fam_of_cell(k) is None:
            continue
        lad = {g: v for g, v in C65.ladder_of_cell(cell).items() if v[2] >= 2}
        if len(lad) < 3:
            continue
        st = C65.stratum_of_cell(k)
        if st in spans:
            spans[st].append(max(v[0] for v in lad.values()) - min(v[0] for v in lad.values()))
    chk(spans["frozen"] and np.median(spans["frozen"]) < 0.30,
        f"T7a frozen granularity span at the noise floor "
        f"({np.median(spans['frozen']):.3f} pp)")
    chk(spans["free"] and np.median(spans["free"]) > 1.0,
        f"T7b free granularity span is large ({np.median(spans['free']):.3f} pp)")

    # ---- T8  peak re-derivation matches the stored peak on unclamped arms ------------
    dis = [(a["arm"], peak(a)[3], a["peak_label"]) for a in arms
           if a["n_clamped"] == 0 and peak(a)[3] != a["peak_label"]]
    chk(not dis, f"T8 re-derived peak == stored peak on unclamped arms ({dis})")

    # ---- T9  metric rule: the banned column never appears on an EXECUTABLE line -------
    self_src = open(os.path.abspath(__file__)).read()
    body = self_src.split('"""', 2)[2]              # everything after the module docstring
    banned = "best" + "_test"
    code_lines = [ln.split("#")[0] for ln in body.splitlines()]
    hits = [ln.strip() for ln in code_lines if banned in ln and "T9" not in ln]
    chk(not hits, f"T9 the banned column never appears on an executable line ({hits})")
    # T9b: c65's `_plateau` is a faithful copy of the CSV's own `plateau` column, so the
    # accuracy every test below reads is the registered PRIMARY metric and nothing else.
    bad9 = [a["run"] for a in arms if a["run"] in rows_by_run
            and abs(rows_by_run[a["run"]]["_plateau"]
                    - float(rows_by_run[a["run"]]["plateau"])) > 1e-12]
    chk(not bad9, f"T9b the scored accuracy is the CSV plateau column verbatim ({bad9})")

    # ---- T10  normalised ladders are normalised --------------------------------------
    chk(all(abs(max(norm_ladder(a)) - 1.0) < 1e-12 for a in arms if a["n_clamped"] == 0),
        "T10 e(u) peaks at exactly 1 on unclamped arms")

    # ---- T11  the corpus is matched: same four families on both sides ----------------
    ff = {a["fam"] for a in arms if a["stratum"] == "frozen"}
    fr = {a["fam"] for a in arms if a["stratum"] == "free"}
    chk(ff == fr and len(ff) == 4, f"T11 frozen and free cover the same 4 families ({ff}|{fr})")

    print(f"selftests: {ok} passed, {len(fails)} failed")
    for f in fails:
        print("  FAIL", f)
    return not fails


# ======================================================================================
# scoring
# ======================================================================================

def report():
    arms = load_arms()
    rows = C65.clean_rows(C65.load_rows())
    rows_by_run = {r["run"]: r for r in rows}

    if not selftests(arms, rows_by_run):
        print("\nSELFTESTS FAILED -- nothing is scored.")
        return 1
    print()

    grid = [r["label"] for r in arms[0]["rows"]]
    us = [r["u"] for r in arms[0]["rows"]]

    # ---------------- the per-arm table ----------------------------------------------
    print("=" * 116)
    print("C66 PART 1 -- THE FIELD'S PEAK AND THE ARM'S OWN TRAINING OUTCOME, 19 arms, "
          "per-coordinate, 20 epochs")
    print("=" * 116)
    print(f"{'arm':<32}{'fam':>6}{'stratum':>9}{'run':<18}"
          f"{'plateau':>9}{'u*':>10}{'rung':>9}{'E*_pp':>10}{'res':>8}{'clamp':>7}")
    for a in sorted(arms, key=lambda x: (x["fam"], x["stratum"], x["arm"])):
        u, e, r, lab = peak(a)
        pl = run_plateau(rows_by_run, a["run"])
        a["_u"], a["_E"], a["_pl"] = u, e, pl
        print(f"{a['arm']:<32}{a['fam']:>6}{a['stratum']:>9}{a['run']:<18}"
              f"{pl:>9.3f}{u:>10.5f}{lab:>9}{e:>10.4f}{r:>8.4f}{a['n_clamped']:>7}")
    clean = [a for a in arms if not a["exc"]]
    exc = [a for a in arms if a["exc"]]
    g_clean = geo([a["_u"] for a in clean])
    print(f"\n  clean-arm geometric mean u* = {g_clean:.5f}   (n={len(clean)})")

    # ---------------- A1 / A3 ---------------------------------------------------------
    print()
    print("=" * 116)
    print("A1 [post-hoc-informed] LOCATION INVARIANCE, frozen vs free, per family")
    print("   registered: |log2(u*_frozen / u*_free)| <= 1.00 rung in 4 of 4 families")
    print("=" * 116)
    print(f"{'fam':>6}{'n_fz':>6}{'n_fr':>6}{'u*_frozen':>12}{'u*_free':>12}"
          f"{'d rungs':>10}   verdict")
    a1_hits = 0
    fams = sorted({a["fam"] for a in clean})
    for fam in fams:
        fz = [a["_u"] for a in clean if a["fam"] == fam and a["stratum"] == "frozen"]
        fr = [a["_u"] for a in clean if a["fam"] == fam and a["stratum"] == "free"]
        d = abs(math.log(geo(fz) / geo(fr)) / RUNG)
        hit = d <= 1.0
        a1_hits += hit
        print(f"{fam:>6}{len(fz):>6}{len(fr):>6}{geo(fz):>12.5f}{geo(fr):>12.5f}"
              f"{d:>10.3f}   {'within' if hit else 'MOVED'}")
    print(f"\n  A1: {a1_hits}/4 families within one rung -> "
          f"{'HELD' if a1_hits == 4 else 'REFUTED' if a1_hits <= 2 else 'PARTIAL'}")

    worst = max(clean, key=lambda a: abs(math.log(a["_u"] / g_clean) / RUNG))
    wd = abs(math.log(worst["_u"] / g_clean) / RUNG)
    print(f"\n  A3: furthest clean arm from the corpus mean = {worst['arm']} at "
          f"{wd:.3f} rungs -> {'HELD' if wd <= 1.0 else 'REFUTED' if wd > 2.0 else 'PARTIAL'}")

    # ---------------- B1 ---------------------------------------------------------------
    print()
    print("=" * 116)
    print("B1 [BLIND] FULL-LADDER SHAPE INVARIANCE.  e(u) = E(u)/E*, averaged within stratum")
    print("   registered: max_u |e_frozen - e_free| <= 0.25   |   refuted if > 0.50")
    print("=" * 116)
    ef = np.mean([norm_ladder(a) for a in clean if a["stratum"] == "frozen"], axis=0)
    er = np.mean([norm_ladder(a) for a in clean if a["stratum"] == "free"], axis=0)
    print(f"  {'rung':<9}{'u':>11}{'e_frozen':>11}{'e_free':>11}{'|diff|':>9}")
    for i, lab in enumerate(grid):
        print(f"  {lab:<9}{us[i]:>11.5f}{ef[i]:>11.4f}{er[i]:>11.4f}{abs(ef[i]-er[i]):>9.4f}")
    gap = float(np.max(np.abs(ef - er)))
    print(f"\n  B1: max rung-wise gap = {gap:.4f} at rung "
          f"{grid[int(np.argmax(np.abs(ef - er)))]} -> "
          f"{'HELD' if gap <= 0.25 else 'REFUTED' if gap > 0.50 else 'PARTIAL'}")

    # ---------------- B2 ---------------------------------------------------------------
    print()
    print("=" * 116)
    print("B2 [BLIND] ACCURACY REGRESSION WITHIN ResNet18 -- log2(u*) vs the arm's own plateau")
    print("   registered: |slope| <= 0.10 rung/pp and peak span <= 1 rung  |  refuted if "
          "|slope| >= 0.25")
    print("=" * 116)
    r18 = [a for a in arms if a["fam"] == "r18"]
    r18.sort(key=lambda a: a["_pl"])
    print(f"  {'arm':<32}{'class':>14}{'plateau':>9}{'u*':>10}{'log2 u*':>10}")
    for a in r18:
        cls = ("frozen" if a["stratum"] == "frozen"
               else "EXC " + ("ms=1e-2" if "ml5" in a["arm"] else "AdamW") if a["exc"]
               else "free ms=1e-3")
        print(f"  {a['arm']:<32}{cls:>14}{a['_pl']:>9.3f}{a['_u']:>10.5f}"
              f"{math.log(a['_u'])/RUNG:>10.3f}")
    x = np.array([a["_pl"] for a in r18])
    y = np.array([math.log(a["_u"]) / RUNG for a in r18])
    slope, intercept = np.polyfit(x, y, 1)
    span = (max(y) - min(y))
    print(f"\n  accuracy range spanned  = {x.min():.3f} .. {x.max():.3f} pp  ({float(x.max()-x.min()):.3f} pp)")
    print(f"  slope                   = {slope:+.4f} rungs per pp")
    print(f"  peak span               = {span:.3f} rungs")
    print(f"  B2 -> {'HELD' if abs(slope) <= 0.10 and span <= 1.0 else 'REFUTED' if abs(slope) >= 0.25 else 'PARTIAL'}")

    # ---------------- B3 ---------------------------------------------------------------
    print()
    print("=" * 116)
    print("B3 [BLIND] THE TWO EXCEPTION ARMS -- never pooled (CORRECTIONS 62), LOCATION only")
    print("   registered: both u* within 2 rungs of the clean-arm geometric mean")
    print("=" * 116)
    print(f"  {'arm':<32}{'what':<18}{'plateau':>9}{'u*':>10}{'rungs from clean':>19}")
    b3 = True
    for a in exc:
        d = abs(math.log(a["_u"] / g_clean) / RUNG)
        b3 &= d <= 2.0
        what = "ms=1e-2 (10x)" if "ml5" in a["arm"] else "AdamW base"
        print(f"  {a['arm']:<32}{what:<18}{a['_pl']:>9.3f}{a['_u']:>10.5f}{d:>19.3f}")
    print(f"\n  B3 -> {'HELD' if b3 else 'REFUTED'}")

    # ---------------- B4 ---------------------------------------------------------------
    print()
    print("=" * 116)
    print("B4 [BLIND] POWER CONTROL -- amplitude MUST move even though location does not")
    print("   registered: geo-mean E*_free / E*_frozen >= 1.50  |  refuted (and A1/B1 "
          "withdrawn) if <= 1.20")
    print("=" * 116)
    print(f"{'fam':>6}{'E*_frozen':>12}{'E*_free':>12}{'ratio':>10}")
    rats = []
    for fam in fams:
        fz = geo([a["_E"] for a in clean if a["fam"] == fam and a["stratum"] == "frozen"])
        fr = geo([a["_E"] for a in clean if a["fam"] == fam and a["stratum"] == "free"])
        rats.append(fr / fz)
        print(f"{fam:>6}{fz:>12.4f}{fr:>12.4f}{fr/fz:>10.3f}")
    R = geo(rats)
    print(f"\n  geometric-mean amplitude ratio = {R:.3f}x")
    print(f"  full corpus E* range           = {min(a['_E'] for a in arms):.4f} .. "
          f"{max(a['_E'] for a in arms):.4f} pp  "
          f"({max(a['_E'] for a in arms)/min(a['_E'] for a in arms):.0f}x)")
    print(f"  B4 -> {'HELD' if R >= 1.50 else 'REFUTED' if R <= 1.20 else 'PARTIAL'}")

    # ---------------- B5 ---------------------------------------------------------------
    print()
    print("=" * 116)
    print("B5 [BLIND, descriptive] WHERE THE UNOBSERVED free-LAYERWISE CELL SITS")
    print("=" * 116)
    cs = C65.cells(rows, "20")
    lay = []
    for k, cell in cs.items():
        if C65.fam_of_cell(k) != "r18" or C65.stratum_of_cell(k) != "free":
            continue
        if "layerwise" in cell and len(cell["layerwise"]) >= 2:
            m, s, n = C65.summarise(cell["layerwise"])
            lay.append((m, s, n, dict(zip(C65.CELL_KEYS, k))))
    lay.sort(key=lambda t: -t[0])
    lo, hi = min(a["_pl"] for a in r18), max(a["_pl"] for a in r18)
    print(f"  r18 probed-arm plateau range: {lo:.3f} .. {hi:.3f}")
    for m, s, n, d in lay[:5]:
        inside = "INSIDE" if lo <= m <= hi else "outside"
        print(f"    free layerwise ms={d['meta_stepsize']:<5} clip={d['beta_clip']:<12} "
              f"plateau {m:7.3f} +-{s:5.3f} (n={n})   -> {inside} the probed range")
    print("\n  Reading: where the layerwise optimum falls INSIDE the probed range, the step")
    print("  from measured to unmeasured is not an extrapolation in OUTCOME.  The residual")
    print("  gap is the PARTITION itself, and only PROBE7 (CORRECTIONS 94.12) closes it.")
    return 0


if __name__ == "__main__":
    sys.exit(report())
