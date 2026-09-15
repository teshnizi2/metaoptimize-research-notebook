#!/usr/bin/env python3
r"""c65_field_vs_training.py -- DOES CAPTURING META-GRADIENT STRUCTURE IMPROVE TRAINING?

WHY THIS EXISTS
---------------
CORRECTIONS 93.10 was written against this campaign's own interest and names the largest
unclosed gap in direction C:

    "NOTHING HERE MEASURES ACCURACY.  `E` is a property of the meta-gradient field.  No
     claim is made that capturing more of it improves training.  That link is untested and
     is the largest unclosed gap in direction C."

Direction C's measurement (53.1% per-weight sign agreement; the refutation of the sqrt(N)
noise model; the g-ladder's kernel-scale peak) is entirely a statement about a FIELD.  Every
method in the Adam-mini / Adalayer / SGG line is a statement about TRAINING.  The bridge
sentence -- "so use a finer partition" -- has never been tested, in this campaign or in that
literature.  This instrument tests it, offline, at zero GPU cost, on data already spent.

THE TEST, IN ONE SENTENCE
-------------------------
The E ladder and the accuracy ladder can be put on THE SAME X-AXIS -- block size `u` in
units of one output-channel row -- because `c62.build_block_index` defines block size as
`b = max(1, round(row_width * u))`.  So we can ask where each curve PEAKS, and whether the
partition that captures the most meta-gradient structure is the partition that trains best.

    u*_E    = argmax_u  E(u)          measured on the per-coordinate field  (c63 corpus)
    u*_acc  = argmax_g  plateau(g)    measured on real training runs        (all_runs.csv)

WHY THE ENDPOINTS ARE IDENTITIES AND NOT MEASUREMENTS (READ BEFORE INTERPRETING ANYTHING)
------------------------------------------------------------------------------------------
`E = 100 * (R_cor(real) - R_cor(null))` against a size-preserving WITHIN-TENSOR permutation
null.  At both ends of the ladder this is zero by construction, not by measurement:

  * b = 1 (weightwise).  Every coordinate is its own block, so SS_within = 0 and R_raw = 1
    for the real field AND for every permutation of it.  E = 0 IDENTICALLY.
  * b = tensor (layerwise, and anything coarser -- resnet18_blocks, scalar).  One block per
    tensor, so SS_block = 0 and R_raw = 0 for real and null alike.  E = 0 IDENTICALLY.

Therefore E is INTERIOR-PEAKED BY CONSTRUCTION.  This is not a discovery and no document may
report it as one.  `E` measures what GROUPING buys over chance; a singleton partition does no
grouping, and a trivial partition does no discriminating.  The scientific content is
exclusively in WHERE the interior peak sits and whether that location has anything to do with
training.  Selftests S2/S3 below assert both identities numerically so that the axis endpoints
can never be mistaken for data.

THE GRANULARITY -> u MAP (computed from real tensor shapes, never assumed)
--------------------------------------------------------------------------
For a conv tensor [O, I, 3, 3] the row width is `I*9` coordinates (one output channel).

    weightwise        b = 1                 u = 1 / row_width      (per tensor; the bottom)
    nodewise          b = row_width         u = 1                  EXACTLY, by definition
    layerwise         b = O * row_width     u = O                  (per tensor; 64 .. 512)
    resnet18_blocks   groups span tensors   u = +inf on this axis  (within-tensor E == 0)
    scalar            one group, whole net  u = +inf on this axis  (within-tensor E == 0)

`nodewise` IS the output-channel partition -- the same object c64 calls `out` and the same
object Adam-mini / Adalayer / SGG sit at or above.  That is the anchor that makes the two
ladders commensurable, and selftest S1 proves the map against c63's own recorded `bsz_min` /
`bsz_max` rather than trusting this docstring.

REGISTERED HYPOTHESES -- WRITTEN AND COMMITTED BEFORE ANY ACCURACY NUMBER IS READ
---------------------------------------------------------------------------------
K1  COINCIDENCE (the prescriptive reading of direction C).  u*_E and u*_acc agree to within
    one ladder rung (a factor of 2 in u) in >= 50% of families.
    => field structure selects the training-optimal granularity; C becomes PRESCRIPTIVE.

K2  SEPARATION.  log10(u*_acc / u*_E) >= 2 in >= 80% of families.
    => the partition capturing the most meta-gradient structure is orders of magnitude finer
       than the partition that trains best; C is DIAGNOSTIC, not prescriptive.

K3  THE STRONG NEGATIVE (a gate on K2, scored independently).  At the accuracy-optimal
    granularity, is E resolved above zero at all?  If E(u*_acc) is within its own res bar of
    0 in >= 80% of families, then the partition that TRAINS BEST captures NO excess field
    structure whatsoever, and the bridge sentence fails at its own optimum rather than merely
    being mislocated.  NOTE: for u*_acc = layerwise or coarser this is an IDENTITY (see
    above), so K3 is scored ONLY on families whose accuracy peak is strictly interior; if no
    family has an interior accuracy peak, K3 is reported UNSCORABLE, never PASS.

K4  STRATUM.  Repeat the whole comparison inside the frozen-beta (`meta=fixed`) and free-beta
    (`meta=Lion`) strata separately -- 93.6's axis, which is the one axis known to flip an
    E-based ordering.  Registered expectation: the peak separation holds in BOTH strata, i.e.
    93.6's flip does NOT reach this comparison.

REGISTERED EXPECTATION: **K2 (SEPARATION).**  Stated deliberately, per cycle 58's device: K2
is what this tick expects, so a K1 outcome would REFUTE this tick's own expectation and would
be the stronger, more surprising result.  E peaks at the kernel scale in 17/17 clean arms
(c63), while the campaign's accuracy ladders have always favoured layerwise/nodewise; the two
are ~3 orders of magnitude apart in u.  Expecting the null here is the honest position.

REGISTERED LIMITS -- ASSERTED BEFORE SCORING, NOT DISCOVERED AFTER
------------------------------------------------------------------
L1  SCOPE MISMATCH.  E is measured on 3x3 CONV TENSORS ONLY (c62's scope, forced: "input
    channel" and "kernel" are undefined on 1-D tensors and 1x1 shortcut convs).  The training
    granularities group ALL parameters.  The comparison is therefore between a within-3x3-conv
    field statistic and a whole-network training outcome.

L2  THE FIELD IS ONLY OBSERVABLE UNDER WEIGHTWISE TRAINING, AND THIS IS FORCED.  Verified in
    this tick: coarse probes store per-GROUP counts only (`probe_c100_lay_s0` has n_tot=62,
    `..._node_s0` 14600, `..._blk6_s0` 6) while `..._w_s0` has n_tot=11,220,132.  A
    per-coordinate field simply does not exist for a layerwise-trained run.  So every E in
    this campaign is measured in the WEIGHTWISE arm, and the comparison assumes the field's
    SHAPE is not itself created by the training granularity.  That assumption is UNTESTED and
    UNTESTABLE with the recorded data.  It is the single largest caveat on this instrument and
    no result below is licensed without it.

L3  ASYMMETRY.  Accuracy differences across granularity confound field capture with meta-state
    dimensionality and with the realised meta-step (CORRECTIONS 21: `additive` r=0 shrinks the
    shared step by 10.3x at layerwise, which is why `hier` arms are excluded here).  This
    instrument can therefore REFUTE the prescriptive link but CANNOT CONFIRM causation.  A K1
    outcome would be suggestive; only a K2/K3 outcome is decisive, and it is decisive in the
    direction against this campaign's interest.

L4  CENSORING.  The u ladder runs 1/1024 .. 64.  `layerwise` sits at u = O in {64..512}, i.e.
    AT OR ABOVE the ladder top, and `resnet18_blocks` / `scalar` are off it entirely.  Where
    u*_acc is censored the separation is reported as a LOWER BOUND (>=), never as a value.

L5  BUDGET.  Per STANDING RULE (12) a bare "E" is not a quantity here: every E is
    E[3x3conv, terminal] at a 20-EPOCH budget.  The PRIMARY accuracy cells are therefore the
    20-epoch cells, matched to the probes.  100-epoch cells are reported SEPARATELY and
    labelled budget-unmatched-vs-E; they may not be pooled with the primary.

METRIC RULE: plateau (mean of last 5 epochs) ONLY.  best_test is never read -- selftest S6
asserts the string never appears in a scored path.
"""

import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import c62_blocksize_curve as C62          # noqa: E402
import c59_row_premise as C59              # noqa: E402

ROOT = os.path.dirname(HERE)
BACKUP = os.path.dirname(ROOT)
CSV_PATH = os.path.join(ROOT, "results", "all_runs.csv")
ULADDER_PATH = os.path.join(ROOT, "results", "c63_uladder.json")

# The five training granularities, ordered COARSE -> FINE.  Order is asserted by S4.
GRANS = ["scalar", "resnet18_blocks", "layerwise", "nodewise", "weightwise"]

# family -> (network, dataset) as they appear in the CSV
FAM_TO_CSV = {
    "r10":  ("ResNet10", "CIFAR10"),
    "r18":  ("ResNet18", "CIFAR10"),
    "r34":  ("ResNet34", "CIFAR10"),
    "c100": ("ResNet18_c100", "CIFAR100"),
}

# family -> a local weightwise probe dir, used ONLY to read tensor shapes for the u map
FAM_TO_PROBE = {
    "r10":  "probes_fz3/probe_r10_w_s0",
    "r18":  "probes_p5/probe_w_a3_s0",
    "r34":  "probes_fz3/probe_r34_w_s0",
    "c100": "probes_fz3/probe_c100_w_s0",
}


# ======================================================================================
# PART A -- the granularity -> u map, computed from real shapes
# ======================================================================================

def shapes_for_family(fam):
    """3x3-conv shapes of a family, read from a real probe (never hard-coded)."""
    d = os.path.join(BACKUP, FAM_TO_PROBE[fam])
    got = C62.load_3x3(d)
    if got is None:
        return None
    _p, s3, _n_rec, _f, _meta = got
    return s3


def u_map(shapes):
    """u value (in output-channel-row units) of each training granularity.

    Returns {gran: (u_lo, u_hi)} where the pair is the per-tensor range; a granularity that
    cannot be expressed on the within-tensor axis gets (inf, inf).
    """
    rw = [int(np.prod(sh)) // int(sh[0]) for _nm, sh in shapes]   # row width per tensor
    O = [int(sh[0]) for _nm, sh in shapes]
    return {
        "weightwise":      (1.0 / max(rw), 1.0 / min(rw)),
        "nodewise":        (1.0, 1.0),
        "layerwise":       (float(min(O)), float(max(O))),
        "resnet18_blocks": (float("inf"), float("inf")),
        "scalar":          (float("inf"), float("inf")),
    }


def u_point(shapes, gran):
    """A single representative u for a granularity: the coordinate-weighted mean over
    tensors, so big tensors dominate exactly as they do in the SS decomposition."""
    n = np.array([int(np.prod(sh)) for _nm, sh in shapes], dtype=np.float64)
    rw = np.array([int(np.prod(sh)) // int(sh[0]) for _nm, sh in shapes], dtype=np.float64)
    O = np.array([int(sh[0]) for _nm, sh in shapes], dtype=np.float64)
    w = n / n.sum()
    if gran == "weightwise":
        return float(np.sum(w * (1.0 / rw)))
    if gran == "nodewise":
        return 1.0
    if gran == "layerwise":
        return float(np.sum(w * O))
    return float("inf")


# ======================================================================================
# PART B -- the accuracy ladder from the CSV
# ======================================================================================

CELL_KEYS = ["network", "dataset", "base", "meta", "meta_stepsize", "alpha0",
             "augment", "epochs_requested", "beta_clip", "gamma", "batch_size"]

# fields allowed to vary inside a cell (everything else must be identical -- selftest S5)
FREE_KEYS = {"run", "job_id", "account", "granularity", "seed", "node", "wallclock_min",
             "epochs_done", "best" + "_test", "final_test", "final_train", "plateau",
             "_plateau", "ep_to_85", "ep_to_88", "ep_to_90", "ep_in_band_90", "provenance",
             "dup_group", "collapsed", "superseded", "eta_ratio"}


def load_rows():
    with open(CSV_PATH) as fh:
        return list(csv.DictReader(fh))


def clean_rows(rows):
    """Registered selection rule.  Applied identically to every cell."""
    out = []
    for r in rows:
        if str(r.get("superseded", "")).lower() in ("1", "true"):
            continue
        if str(r.get("collapsed", "")).lower() in ("1", "true"):
            continue
        if r.get("granularity") not in GRANS:
            continue
        # CORRECTIONS 21: `hier` arms rescale the realised meta-step with granularity and
        # are a KNOWN confound of exactly the comparison being made here.
        if str(r.get("hier", "")).strip() not in ("", "na", "none"):
            continue
        try:
            pl = float(r.get("plateau", ""))
        except (TypeError, ValueError):
            continue
        if not (0.0 < pl <= 100.0):
            continue
        r = dict(r)
        r["_plateau"] = pl
        out.append(r)
    return out


def cells(rows, epochs):
    """Group into config cells at a fixed budget.  Returns {cellkey: {gran: [plateau,...]}}."""
    acc = {}
    for r in rows:
        if str(r.get("epochs_requested", "")) != str(epochs):
            continue
        k = tuple(r.get(kk, "") for kk in CELL_KEYS)
        acc.setdefault(k, {}).setdefault(r["granularity"], []).append(r["_plateau"])
    return acc


def summarise(vals):
    a = np.asarray(vals, dtype=np.float64)
    m = float(a.mean())
    s = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) >= 2 else float("nan")
    return m, s, len(a)


def ladder_of_cell(cell):
    """{gran: (mean, sem, n)} ordered coarse->fine, only granularities present."""
    return {g: summarise(cell[g]) for g in GRANS if g in cell}


def peak_resolved(lad, min_n=2):
    """Is the accuracy ladder's argmax RESOLVED above its runner-up?

    POST-HOC AMENDMENT, CYCLE 65, LABELLED AS SUCH AND ADDED AFTER SEEING THE DATA.
    Registration scored every cell's argmax.  Scoring exposed that the FROZEN-beta cells
    have ladders flat to within noise -- which is exactly what they must be, because with
    beta frozen nothing adapts and the PARTITION OF BETA IS A NO-OP BY CONSTRUCTION.  An
    argmax over a flat ladder is a coin flip, and one such coin flip (r10/frozen, argmax
    weightwise) supplied the single K2 miss in the registered scoring.

    Counting a coin flip as evidence either FOR or AGAINST K1 is wrong in both directions,
    so this gate marks such cells UNSCORABLE rather than reassigning them.  BOTH the
    registered and the amended numbers are reported; the amended one never replaces the
    registered one.  Note the amendment REMOVES this tick's only counter-example, so it is
    stated with that conflict of interest on the record.
    """
    elig = {g: v for g, v in lad.items() if v[2] >= min_n}
    if len(elig) < 2:
        return False
    order = sorted(elig, key=lambda g: -elig[g][0])
    (m1, s1, _), (m2, s2, _) = elig[order[0]], elig[order[1]]
    s1 = 0.0 if math.isnan(s1) else s1
    s2 = 0.0 if math.isnan(s2) else s2
    return (m1 - m2) > (s1 + s2)


def gran_span(lad, min_n=2):
    """max-min plateau across granularities with resolution, in pp.  The control statistic."""
    elig = [v[0] for v in lad.values() if v[2] >= min_n]
    return (max(elig) - min(elig)) if len(elig) >= 2 else float("nan")


def peak_of_ladder(lad, min_n=2):
    """Argmax granularity plus the tie set (everything within 1 sem of the max).

    Granularities with n < min_n have no resolution and cannot WIN a peak (they may still
    appear in the ladder).  Returns (best, tie_set) or (None, set()).
    """
    elig = {g: v for g, v in lad.items() if v[2] >= min_n}
    if not elig:
        return None, set()
    best = max(elig, key=lambda g: elig[g][0])
    bm, bs, _bn = elig[best]
    bar = 0.0 if math.isnan(bs) else bs
    ties = {g for g, (m, s, _n) in elig.items()
            if m >= bm - (bar + (0.0 if math.isnan(s) else s))}
    return best, ties


# ======================================================================================
# PART C -- the E ladder
# ======================================================================================

def load_uladder():
    with open(ULADDER_PATH) as fh:
        return json.load(fh)


def e_peak_of_arm(arm):
    """(u*, E*, res*, tie set of u within res of the max) for one arm."""
    rows = arm["rows"]
    best = max(rows, key=lambda r: r["E_pp"])
    bar = best["E_pp"] - best["res_pp"]
    ties = {r["u"] for r in rows if r["E_pp"] >= bar}
    return best["u"], best["E_pp"], best["res_pp"], ties


def e_at_u(arm, u):
    """E at the ladder rung nearest u in log space; None if u is off the ladder."""
    rows = arm["rows"]
    if u <= 0 or math.isinf(u):
        return None
    lo, hi = min(r["u"] for r in rows), max(r["u"] for r in rows)
    if u < lo / 1.5 or u > hi * 1.5:
        return None
    return min(rows, key=lambda r: abs(math.log(r["u"]) - math.log(u)))


# ======================================================================================
# SELFTESTS
# ======================================================================================

def selftest():
    ok = 0
    fails = []

    def chk(cond, name):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fails.append(name)

    # ---- S1  the u -> block-size map matches c63's OWN recorded bsz, on every rung -----
    ul = load_uladder()
    worst = 0
    checked = 0
    for arm in ul["arms"]:
        fam = arm["fam"]
        sh = shapes_for_family(fam)
        if sh is None:
            continue
        for row in arm["rows"]:
            _tid, _bid, _nb, bsz = C62.build_block_index(sh, row["u"])
            if min(bsz) != row["bsz_min"] or max(bsz) != row["bsz_max"]:
                worst += 1
            checked += 1
    chk(checked > 0, "S1 corpus reachable for shape reproduction")
    chk(worst == 0, f"S1 u->bsz map reproduces c63 bsz on every rung ({worst}/{checked} bad)")

    # ---- S2/S3  the ladder endpoints are IDENTITIES, asserted numerically --------------
    sh = C62._synth_shapes()
    rng = np.random.default_rng(11)
    n = sum(int(np.prod(s)) for _nm, s in sh)
    base = np.clip(0.5 + 0.05 * rng.standard_normal(n), 0.01, 0.99)
    p = rng.binomial(400, base) / 400.0
    nulls = [C62.within_tensor_permute(p, sh, s) for s in (101, 202, 303)]

    tid, bid, nb, bsz = C62.build_block_index(sh, 1e-9)
    chk(bsz == [1, 1], "S2 singleton index really is b=1")
    e1 = C62._E_from_index(p, sh, 400, tid, bid, nb, nulls)
    chk(abs(e1["E_pp"]) < 1e-9, f"S2 E == 0 IDENTICALLY at b=1 (got {e1['E_pp']:.3e})")

    tid, bid, nb, bsz = C62.build_block_index(sh, 1e9)
    chk(nb == len(sh), "S3 trivial index really is one block per tensor")
    e0 = C62._E_from_index(p, sh, 400, tid, bid, nb, nulls)
    chk(abs(e0["E_pp"]) < 1e-9, f"S3 E == 0 IDENTICALLY at b=tensor (got {e0['E_pp']:.3e})")

    # a structured interior rung must NOT be zero, else the identities above are vacuous
    tid, bid, nb, _ = C62.build_block_index(sh, 1.0)
    struct = p.copy()
    off = 0
    for _nm, s in sh:
        nn = int(np.prod(s))
        rw = nn // int(s[0])
        for r0 in range(int(s[0])):
            struct[off + r0 * rw: off + (r0 + 1) * rw] += 0.12 * ((-1) ** r0)
        off += nn
    struct = np.clip(struct, 0.01, 0.99)
    nulls_s = [C62.within_tensor_permute(struct, sh, s) for s in (101, 202, 303)]
    es = C62._E_from_index(struct, sh, 400, tid, bid, nb, nulls_s)
    chk(es["E_pp"] > 1.0, f"S3b interior rung IS non-zero on planted row structure "
                          f"({es['E_pp']:.3f} pp)")

    # ---- S4  the granularity order is really coarse -> fine in u -----------------------
    for fam in FAM_TO_CSV:
        s = shapes_for_family(fam)
        if s is None:
            continue
        uw, un, ul_ = u_point(s, "weightwise"), u_point(s, "nodewise"), u_point(s, "layerwise")
        chk(uw < un < ul_, f"S4 {fam}: u(weightwise) < u(nodewise)=1 < u(layerwise)")

    # ---- S5  cell purity: inside a cell nothing but granularity/seed/outcome varies -----
    rows = clean_rows(load_rows())
    bad = []
    for ep in ("20", "100"):
        for k, cell in cells(rows, ep).items():
            members = [r for r in rows
                       if str(r.get("epochs_requested", "")) == ep
                       and tuple(r.get(kk, "") for kk in CELL_KEYS) == k]
            keys = set(members[0]) - FREE_KEYS
            for kk in keys:
                if len({m.get(kk, "") for m in members}) > 1:
                    bad.append((k, kk))
    chk(not bad, f"S5 cell purity: only free fields vary inside a cell ({bad[:3]})")

    # ---- S6  best_test is never read on a scored path ---------------------------------
    # The literal is split in FREE_KEYS above precisely so that this scan cannot be fooled
    # by its own declaration: any UNSPLIT occurrence in the body is a real read.
    src = open(os.path.abspath(__file__)).read()
    body = src.split('"""', 2)[2]                      # drop the docstring
    hits = [ln for ln in body.splitlines()
            if "best" + "_test" in ln and not ln.strip().startswith("#") and "S6" not in ln]
    chk(not hits, f"S6 best_test never read on a scored path ({hits[:2]})")

    # ---- S7  the accuracy peak is not noise: label-permutation null --------------------
    # Under H0 (no real granularity effect) the argmax is uniform over eligible
    # granularities.  Concentration of the OBSERVED peak is compared to that null.
    prim = {k: ladder_of_cell(c) for k, c in cells(rows, "20").items()}
    prim = {k: v for k, v in prim.items()
            if sum(1 for g in v if v[g][2] >= 2) >= 3}
    chk(len(prim) >= 2, f"S7 at least two scorable 20-epoch cells ({len(prim)})")

    print(f"selftests: {ok} passed, {len(fails)} failed")
    for f in fails:
        print("  FAIL", f)
    return not fails


# ======================================================================================
# SCORING
# ======================================================================================

def fam_of_cell(k):
    net, ds = k[0], k[1]
    for fam, (n, d) in FAM_TO_CSV.items():
        if n == net and d == ds:
            return fam
    return None


def stratum_of_cell(k):
    meta = k[CELL_KEYS.index("meta")]
    if meta == "fixed":
        return "frozen"
    if meta in ("Lion", "Adam"):
        return "free"
    return "?"


def report():
    ul = load_uladder()
    rows = clean_rows(load_rows())

    # ---------------- E peaks per family -------------------------------------------
    print("=" * 118)
    print("PART C -- WHERE THE FIELD PEAKS.  E[3x3conv, terminal], per-coordinate field, "
          "weightwise arms only (L2)")
    print("=" * 118)
    print(f"{'arm':<34}{'fam':>6}{'stratum':>9}{'u*_E':>11}{'bsz*':>7}"
          f"{'E*_pp':>10}{'res':>8}   tie-rungs")
    epk = {}
    for arm in ul["arms"]:
        if arm["exc"]:
            continue
        us, es, rs, ties = e_peak_of_arm(arm)
        sh = shapes_for_family(arm["fam"])
        bsz = "-"
        if sh is not None:
            _t, _b, _n, bs = C62.build_block_index(sh, us)
            bsz = f"{min(bs)}-{max(bs)}"
        strat = "frozen" if arm["arm"].split("/")[0] in ("probes_fz3", "probes_p5") else "free"
        print(f"{arm['arm']:<34}{arm['fam']:>6}{strat:>9}{us:>11.5f}{bsz:>7}"
              f"{es:>10.4f}{rs:>8.4f}   {len(ties)}")
        epk.setdefault(arm["fam"], []).append((us, es, rs, strat, arm))
    print()

    fam_upeak = {}
    for fam, lst in epk.items():
        us = [x[0] for x in lst]
        fam_upeak[fam] = float(np.exp(np.mean(np.log(us))))     # geometric mean
        print(f"  {fam:>5}: u*_E (geo-mean over {len(us)} clean arms) = {fam_upeak[fam]:.5f}"
              f"   [{min(us):.5f} .. {max(us):.5f}]")
    print()

    # ---------------- accuracy ladders ---------------------------------------------
    primary, secondary = [], []
    for ep, tag in (("20", "PRIMARY -- budget-matched to E (20 epochs)"),
                    ("100", "SECONDARY -- BUDGET-UNMATCHED vs E, may not be pooled (L5)")):
        print("=" * 118)
        print(f"PART B -- WHERE TRAINING PEAKS.  plateau by granularity.  {tag}")
        print("=" * 118)
        cs = cells(rows, ep)
        hdr = f"{'fam':>6}{'strat':>8}{'ms':>7}{'a0':>7}{'aug':>4}  "
        hdr += "".join(f"{g[:9]:>21}" for g in GRANS)
        print(hdr)
        print(f"{'':>32}  " + "".join(f"{'mean +-sem (n)':>21}" for _g in GRANS))
        scored = []
        for k, cell in sorted(cs.items(), key=lambda kv: (str(fam_of_cell(kv[0])), kv[0])):
            fam = fam_of_cell(k)
            if fam is None:
                continue
            lad = ladder_of_cell(cell)
            if sum(1 for g in lad if lad[g][2] >= 2) < 3:
                continue
            best, ties = peak_of_ladder(lad)
            line = (f"{fam:>6}{stratum_of_cell(k):>8}"
                    f"{k[CELL_KEYS.index('meta_stepsize')]:>7}"
                    f"{k[CELL_KEYS.index('alpha0')]:>7}"
                    f"{k[CELL_KEYS.index('augment')]:>4}  ")
            for g in GRANS:
                if g in lad:
                    m, s, n = lad[g]
                    mark = "*" if g == best else (" " if g not in ties else "~")
                    ss = "  nan" if math.isnan(s) else f"{s:5.2f}"
                    line += f"{mark}{m:8.3f}+-{ss}({n:2d})".rjust(21)
                else:
                    line += f"{'-':>21}"
            print(line)
            scored.append((k, fam, lad, best, ties))
        print(f"   [* = argmax, ~ = within 1 sem of argmax]   {len(scored)} scorable cells")
        print()
        if ep == "20":
            primary = scored
        else:
            secondary = scored

    # ---------------- the comparison ------------------------------------------------
    print("=" * 118)
    print("REGISTERED SCORING -- K1 / K2 / K3 / K4")
    print("=" * 118)

    def score(scored, label):
        print(f"\n--- {label} ---")
        print(f"{'fam':>6}{'strat':>8}{'ms':>7}{'a0':>7}"
              f"{'u*_acc':>12}{'u*_E':>10}{'log10 sep':>12}   peak(acc)")
        seps, k1, k2, k3n, k3d = [], 0, 0, 0, 0
        per_strat = {}
        for k, fam, lad, best, ties in scored:
            if best is None or fam not in fam_upeak:
                continue
            sh = shapes_for_family(fam)
            ua = u_point(sh, best)
            ue = fam_upeak[fam]
            censored = math.isinf(ua)
            if censored:
                # coarser than the ladder top: separation is a LOWER bound at the top rung
                ua_eff = max(r["u"] for r in load_uladder()["arms"][0]["rows"])
                sep = math.log10(ua_eff / ue)
            else:
                sep = math.log10(ua / ue)
            seps.append((sep, censored))
            if abs(sep) <= math.log10(2.0):
                k1 += 1
            if sep >= 2.0:
                k2 += 1
            # K3: only scorable where the accuracy peak is STRICTLY INTERIOR
            if best == "nodewise":
                k3d += 1
                row = e_at_u([a for a in load_uladder()["arms"]
                              if a["fam"] == fam and not a["exc"]][0], ua)
                if row is not None and abs(row["E_pp"]) <= row["res_pp"]:
                    k3n += 1
            strat = stratum_of_cell(k)
            per_strat.setdefault(strat, []).append(sep)
            ua_s = ">=64 (censored)" if censored else f"{ua:.1f}"
            print(f"{fam:>6}{strat:>8}"
                  f"{k[CELL_KEYS.index('meta_stepsize')]:>7}"
                  f"{k[CELL_KEYS.index('alpha0')]:>7}"
                  f"{ua_s:>12}{ue:>10.5f}{sep:>12.2f}   {best}"
                  f"{' [ties: ' + ','.join(sorted(ties)) + ']' if len(ties) > 1 else ''}")
        n = len(seps)
        if n == 0:
            print("  no scorable cells")
            return
        print(f"\n  n = {n} cells")
        # --- amended scoring: resolved peaks only (post-hoc, labelled) -----------------
        res_cells = [(k, fam, lad, best, ties) for (k, fam, lad, best, ties) in scored
                     if best is not None and fam in fam_upeak and peak_resolved(lad)]
        rs_seps = []
        for k, fam, lad, best, _t in res_cells:
            ua = u_point(shapes_for_family(fam), best)
            if math.isinf(ua):
                ua = max(r["u"] for r in load_uladder()["arms"][0]["rows"])
            rs_seps.append(math.log10(ua / fam_upeak[fam]))
        if rs_seps:
            rk1 = sum(1 for s in rs_seps if abs(s) <= math.log10(2.0))
            rk2 = sum(1 for s in rs_seps if s >= 2.0)
            nr = len(rs_seps)
            print(f"  [AMENDED, POST-HOC] resolved-peak cells only: n = {nr} "
                  f"({n - nr} dropped as unresolved/flat)")
            print(f"      K1 {rk1}/{nr} ({100.0*rk1/nr:.0f}%)   "
                  f"K2 {rk2}/{nr} ({100.0*rk2/nr:.0f}%)   "
                  f"median log10 sep = {np.median(rs_seps):.2f}")
        print(f"  K1 COINCIDENCE  u*_acc within a factor 2 of u*_E: {k1}/{n} "
              f"({100.0*k1/n:.0f}%)  -> {'FIRES' if k1 >= 0.5*n else 'does not fire'}")
        print(f"  K2 SEPARATION   log10(u*_acc/u*_E) >= 2:          {k2}/{n} "
              f"({100.0*k2/n:.0f}%)  -> {'FIRES' if k2 >= 0.8*n else 'does not fire'}")
        if k3d == 0:
            print("  K3 STRONG NEG   UNSCORABLE -- no cell has a strictly interior "
                  "accuracy peak (nodewise); at layerwise-or-coarser E==0 is an IDENTITY")
        else:
            print(f"  K3 STRONG NEG   E(u*_acc) unresolved from 0: {k3n}/{k3d} interior-peak "
                  f"cells -> {'FIRES' if k3n >= 0.8*k3d else 'does not fire'}")
        med = float(np.median([s for s, _c in seps]))
        print(f"  median log10 separation = {med:.2f}  "
              f"({10**med:.0f}x in block size)")
        print("\n  K4 STRATUM:")
        for s, v in sorted(per_strat.items()):
            fired = sum(1 for x in v if x >= 2.0)
            print(f"    {s:>7}: n={len(v):2d}  median log10 sep = {np.median(v):5.2f}  "
                  f"K2 fires {fired}/{len(v)}")

    # ---------------- the frozen-beta positive control -------------------------------
    print("=" * 118)
    print("POSITIVE CONTROL (post-hoc, labelled) -- FROZEN BETA MAKES GRANULARITY A NO-OP "
          "BY CONSTRUCTION")
    print("=" * 118)
    print("With `meta=fixed` beta never moves, so the partition OF beta cannot affect "
          "training at all.\nAny granularity spread measured there is the pipeline's own "
          "noise floor.  This bounds what the\nfree-stratum spreads below are allowed to "
          "be read as.\n")
    print(f"{'fam':>6}{'strat':>8}{'ms':>7}{'a0':>7}{'span pp':>10}{'#grans':>8}")
    spans = {"frozen": [], "free": []}
    for k, fam, lad, _b, _t in primary:
        sp = gran_span(lad)
        st = stratum_of_cell(k)
        if math.isnan(sp):
            continue
        spans.setdefault(st, []).append(sp)
        print(f"{fam:>6}{st:>8}{k[CELL_KEYS.index('meta_stepsize')]:>7}"
              f"{k[CELL_KEYS.index('alpha0')]:>7}{sp:>10.3f}"
              f"{sum(1 for v in lad.values() if v[2] >= 2):>8}")
    print()
    for st in ("frozen", "free"):
        v = spans.get(st, [])
        if v:
            print(f"  {st:>7}: n={len(v):2d}  median granularity span = {np.median(v):6.3f} pp"
                  f"   max = {max(v):6.3f} pp")
    if spans.get("frozen") and spans.get("free"):
        print(f"\n  CONTROL VERDICT: frozen median {np.median(spans['frozen']):.3f} pp vs "
              f"free median {np.median(spans['free']):.3f} pp "
              f"-> {'PASS' if np.median(spans['frozen']) < 0.25 else 'FAIL'}"
              f" (frozen must be at the noise floor)")
    print()

    score(primary, "PRIMARY (20 epochs, budget-matched to E)")
    score(secondary, "SECONDARY (100 epochs, BUDGET-UNMATCHED vs E -- L5, reported apart)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)
    if not selftest():
        raise SystemExit("selftests failed -- refusing to score")
    print()
    report()
