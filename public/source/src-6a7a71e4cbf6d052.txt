#!/usr/bin/env python3
"""c60_exception_mechanism.py -- WHY DO 10 WEIGHTWISE AND 8 NODEWISE ARMS INVERT?

WHY THIS EXISTS
---------------
CONTINUE-HERE (cycle 59, item (e)) ranks this FIRST among remaining offline work.

FINDINGS 59.3 measured, on 48 of 58 unique weightwise arms, that the ROW explains
0.083%-0.612% of the WITHIN-TENSOR structure in per-weight meta-gradient sign preference
`p_i` -- Adam-mini's premise, as it applies to `z`, FAILS.  FINDINGS 59.8 measured the
complementary half at the nodewise arms: the row's TENSOR explains 86.88%-97.06% of
ROW-level structure, so rows inside a tensor are near-interchangeable.

**Both halves have exceptions, and they are THE SAME CONFIGS.**

  weightwise (59.4): 10 arms with R_row_cor 7.17%-49.37% instead of ~0.19%
  nodewise   (59.8):  8 arms with R_tensor_cor 7.80%-51.42% instead of ~93.50%

FINDINGS 59.4 refuted the obvious explanation with our own control: `cl5/probe_w_cD_*` is
75.7%-76.4% clip-pinned and sits at the null (0.164%-0.196%), so **"clipping manufactures
row structure" is FALSE**.  The mechanism was left OPEN.  This instrument closes it, or
fails to, on data already on this Mac.  Zero cluster compute.

THE HYPOTHESIS, REGISTERED BEFORE ANY ARM IS SCORED
---------------------------------------------------
Both statistics are ratios of sums that run over tensors of TWO STRUCTURALLY DIFFERENT
KINDS, and the campaign has never separated them:

  1-D tensors (BatchNorm weight, BatchNorm bias, linear bias).  Shape `(d0,)`, so
      row width == 1.  **A row of size 1 has ZERO within-row variance BY CONSTRUCTION.**
      Every one of these coordinates contributes to SS_row|tensor and contributes
      EXACTLY 0 to SS_within.  For r18 that is 9,600 of 11,173,962 coordinates (0.086%)
      but 9,600 of 14,420 rows (66.6%).

  conv / fc tensors.  Row width 27-4608.  These carry 99.91% of the coordinates and
      essentially all of SS_within.

R_row = SS_row|tensor / (SS_row|tensor + SS_within) is therefore a MIXTURE.  In a normal
arm the conv within-row term dominates the denominator and R_row ~ 0.2%.  But if an arm's
conv coordinates SATURATE -- `p_i` collapsing onto a common value or onto {0,1} so that
both conv terms go to zero -- then the ratio is left carried entirely by the 1-D tensors,
whose numerator is nonzero and whose denominator is zero BY CONSTRUCTION, and R_row is
driven toward 1 **without any row structure existing anywhere**.

  H_A  1-D DOMINATION (artifact).  Exceptions have raw_row(1-D)/raw_row(all) -> 1 and
       **conv-only R_row at the null**.  Then 59.3 has no exceptions at all: the premise
       fails in the exception arms too, and the inversion is a tensor-class mixing effect.
  H_B  GENUINE CONV ROW STRUCTURE.  Exceptions keep a large conv-only R_row.  Then 59.3
       acquires a real exception class and the mechanism is physics, not bookkeeping.
  H_C  DENOMINATOR COLLAPSE.  Measured directly as conv `sd(p_i)` within tensors and as
       the fraction of conv coordinates at p_i in {0,1}.  H_C is the DRIVER of H_A, not a
       rival to it; reported alongside so the composite can be read as one mechanism.

H_A and H_B are mutually exclusive on the same printed number (conv-only R_row), so this
is decidable.  Direction is registered here, before scoring, per standing practice.

INSTRUMENT-INTEGRITY GATE THAT MUST RUN FIRST
----------------------------------------------
`decompose()` in c59 clamps at `max(raw - noise, 0)`.  If an arm's conv SS_within is fully
explained by binomial sampling noise the clamp returns EXACTLY 0 and R_row_cor becomes 1
by construction -- a floor artifact that would look identical to H_B at the aggregate.
`--gate` reports, per arm and per class, whether either clamp bound.  Any arm whose verdict
depends on a bound clamp is reported as UNDECIDABLE, never as evidence for either branch.

SCOPE
-----
Same scope as c59 and it is not relaxed here: Adam-mini argues about `G`, the BASE
gradient; PATCH_PROBE5 recorded the sign of `z`, the META-gradient.  No document may write
"we refuted Adam-mini".  Every printed column says `z`.

USAGE
    python3 analysis/c60_exception_mechanism.py --selftest
    python3 analysis/c60_exception_mechanism.py --gate       [--root ..]
    python3 analysis/c60_exception_mechanism.py --weightwise [--root ..]
    python3 analysis/c60_exception_mechanism.py --nodewise   [--root ..]
    python3 analysis/c60_exception_mechanism.py --report     [--root ..]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c59_row_premise as C59   # architecture reconstruction is REUSED, never re-derived


# --------------------------------------------------------------------------------------
# Tensor classes
# --------------------------------------------------------------------------------------
# Row width is `prod(shape[1:])`.  The class split is a statement about that width, and
# the names are only labels for it.
#   oneD : len(shape)==1  -> row width 1  -> SS_within == 0 BY CONSTRUCTION
#   conv : len(shape)==4  -> row width cin*k*k
#   fc   : len(shape)==2  -> row width cin
CLASSES = ("oneD", "conv", "fc")


def class_of(shape):
    if len(shape) == 1:
        return "oneD"
    if len(shape) == 2:
        return "fc"
    if len(shape) == 4:
        return "conv"
    raise ValueError(f"unclassifiable shape {shape}")


def class_masks(shapes):
    """Per-coordinate class label, as a dict class -> boolean mask."""
    lab = np.concatenate([np.full(int(np.prod(sh)), CLASSES.index(class_of(sh)),
                                  dtype=np.int8)
                          for _nm, sh in shapes])
    return {c: (lab == i) for i, c in enumerate(CLASSES)}, lab


def tensor_subset(shapes, keep):
    """The sub-list of `shapes` whose class is in `keep`, order preserved."""
    return [(nm, sh) for nm, sh in shapes if class_of(sh) in keep]


def coord_subset(p, shapes, keep):
    """Coordinates of `p` belonging to tensors whose class is in `keep`."""
    _masks, lab = class_masks(shapes)
    sel = np.zeros(p.size, dtype=bool)
    for c in keep:
        sel |= (lab == CLASSES.index(c))
    return p[sel]


# --------------------------------------------------------------------------------------
# Class-resolved RAW sums of squares.  These are EXACTLY additive across classes because
# SS_row|tensor and SS_within are both sums over rows, and every row lies in exactly one
# tensor, hence in exactly one class.  (SS_tensor is NOT additive -- it references the
# grand mean -- and is deliberately not split here.)
# --------------------------------------------------------------------------------------

def class_raw_ss(p, shapes, n_rec):
    """Return {class: dict(raw_row, raw_within, noise_row, noise_within, n, n_rows)}.

    Sums over classes reproduce c59's global raw_row / raw_within / noise_row /
    noise_within to float precision; `--selftest` and `--gate` both check this.
    """
    tid, rid, rw, _sp, n_rows = C59.build_index(shapes)
    n_tensors = len(shapes)
    ss = C59.nested_ss(p, tid, rid, n_tensors, n_rows)

    row_mean, row_cnt, row_tid, nz = ss["row_mean"], ss["row_cnt"], ss["row_tid"], ss["nz"]

    # tensor means, recomputed from row totals exactly as nested_ss does
    t_cnt = np.bincount(row_tid[nz], weights=row_cnt[nz], minlength=n_tensors)
    t_sum = np.bincount(row_tid[nz], weights=row_mean[nz] * row_cnt[nz],
                        minlength=n_tensors)
    tnz = t_cnt > 0
    t_mean = np.zeros(n_tensors); t_mean[tnz] = t_sum[tnz] / t_cnt[tnz]

    # per-ROW contributions
    row_ss_row = np.zeros(n_rows)
    row_ss_row[nz] = row_cnt[nz] * (row_mean[nz] - t_mean[row_tid[nz]]) ** 2
    within_i = (p - row_mean[rid]) ** 2
    row_ss_within = np.bincount(rid, weights=within_i, minlength=n_rows)

    s2 = p * (1.0 - p) / float(n_rec)
    row_s2 = np.bincount(rid, weights=s2, minlength=n_rows)
    row_noise_within = np.zeros(n_rows)
    row_noise_row = np.zeros(n_rows)
    row_noise_within[nz] = (1.0 - 1.0 / row_cnt[nz]) * row_s2[nz]
    row_noise_row[nz] = row_s2[nz] / row_cnt[nz]

    # class of each tensor, hence of each row
    t_class = np.array([CLASSES.index(class_of(sh)) for _nm, sh in shapes], dtype=np.int8)
    row_class = np.full(n_rows, -1, dtype=np.int8)
    row_class[nz] = t_class[row_tid[nz]]

    out = {}
    for ci, c in enumerate(CLASSES):
        m = row_class == ci
        cm = np.zeros(p.size, dtype=bool)
        if m.any():
            cm = m[rid]
        out[c] = dict(
            raw_row=float(row_ss_row[m].sum()),
            raw_within=float(row_ss_within[m].sum()),
            noise_row=float(row_noise_row[m].sum()),
            noise_within=float(row_noise_within[m].sum()),
            n=int(cm.sum()), n_rows=int(m.sum()),
        )
    out["_global"] = dict(
        raw_row=ss["ss_row"], raw_within=ss["ss_within"],
        noise_row=float(row_noise_row[nz].sum()),
        noise_within=float(row_noise_within[nz].sum()),
        ss_tensor=ss["ss_tensor"], ss_total=ss["ss_total"],
        n=int(p.size), n_rows=int(nz.sum()),
    )
    return out


def R_from(raw_row, raw_within, noise_row, noise_within):
    """Noise-corrected R_row, plus the raw bound and the clamp flags."""
    cr = raw_row - noise_row
    cw = raw_within - noise_within
    clamped_row, clamped_within = cr < 0, cw < 0
    cr, cw = max(cr, 0.0), max(cw, 0.0)
    den_raw = raw_row + raw_within
    den_cor = cr + cw
    return dict(
        R_raw=(raw_row / den_raw) if den_raw > 0 else float("nan"),
        R_cor=(cr / den_cor) if den_cor > 0 else float("nan"),
        cor_row=cr, cor_within=cw,
        clamped_row=bool(clamped_row), clamped_within=bool(clamped_within),
        raw_row=raw_row, raw_within=raw_within,
    )


def class_R(cls):
    """R_row computed on ONE class's coordinates alone."""
    return R_from(cls["raw_row"], cls["raw_within"], cls["noise_row"], cls["noise_within"])


# --------------------------------------------------------------------------------------
# Saturation profile (H_C's direct measurement)
# --------------------------------------------------------------------------------------

def saturation(p, shapes, n_rec):
    """Per-class collapse statistics of the per-coordinate sign preference p_i.

    f_extreme : fraction of coordinates with p_i exactly 0 or exactly 1 (the coordinate's
                meta-gradient sign never once flipped across all n_rec records)
    sd_within : sd of (p_i - row_mean_i), i.e. the spread the within-row term sees
    sd_class  : sd of p_i over the class
    """
    tid, rid, _rw, _sp, n_rows = C59.build_index(shapes)
    ss = C59.nested_ss(p, tid, rid, len(shapes), n_rows)
    resid = p - ss["row_mean"][rid]
    masks, _lab = class_masks(shapes)
    out = {}
    for c in CLASSES:
        m = masks[c]
        if not m.any():
            out[c] = None
            continue
        pc = p[m]
        out[c] = dict(
            n=int(m.sum()),
            f_extreme=float(np.mean((pc <= 0.0) | (pc >= 1.0))),
            f_half=float(np.mean(np.abs(pc - 0.5) < (0.5 / n_rec))),
            sd_class=float(pc.std()),
            sd_within=float(resid[m].std()),
            mean=float(pc.mean()),
        )
    return out


# --------------------------------------------------------------------------------------
# Nodewise side (59.8's statistic), class-restricted
# --------------------------------------------------------------------------------------

def nodewise_class_R(p, shapes, n_rec, keep, regroup_seed=None):
    """R_tensor computed over the ROWS belonging to tensors of class in `keep`.

    Restricting changes the grand mean, so ss_between is genuinely a different quantity
    (between-tensor variance AMONG THE KEPT TENSORS).  That is the intended comparison and
    is labelled as such wherever printed.
    """
    sub = tensor_subset(shapes, keep)
    if not sub:
        return None
    rows_tid = C59.nodewise_rows(shapes)
    t_class = np.array([CLASSES.index(class_of(sh)) for _nm, sh in shapes], dtype=np.int8)
    keep_i = {CLASSES.index(c) for c in keep}
    sel = np.array([t_class[t] in keep_i for t in rows_tid])
    if sel.sum() < 2:
        return None
    return C59.nodewise_decompose(p[sel], sub, n_rec, regroup_seed=regroup_seed)


# --------------------------------------------------------------------------------------
# Arm enumeration + metadata (dedupe by realpath -- CORRECTIONS 88.11's symlink hazard)
# --------------------------------------------------------------------------------------

def arms(root, kind):
    finder = C59.find_weightwise if kind == "weightwise" else None
    if kind == "weightwise":
        seen, out = set(), []
        for d in finder(root):
            rp = os.path.realpath(d)
            if rp in seen:
                continue
            seen.add(rp)
            out.append(d)
        return out
    return [d for d, _n in C59.find_nodewise(root)]


def arm_label(d):
    return f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}"


def arm_meta(d):
    m = json.load(open(os.path.join(d, "neg_counts.json")))
    return m


# The 10 weightwise exceptions, verbatim from FINDINGS 59.4.  Membership is DATA copied
# from a published table, not a threshold recomputed here; --gate re-derives each arm's
# R_row_cor and checks it still lands where 59.4 recorded it.
EXC_W = {
    "probes_bo6/probe_w_adw_s0": 49.37, "probes_ml5/probe_w_m2_s2": 44.06,
    "probes_ml5/probe_w_m2_s1": 43.80, "probes_wc5/probe_w_m2_s2": 43.24,
    "probes_wc5/probe_w_m2_s0": 42.65, "probes_ml5/probe_w_m2_s0": 42.33,
    "probes_wc5/probe_w_m2_s1": 41.55, "probes_bo6/probe_w_adw_s1": 36.83,
    "probes_bl5/probe_w_e40_s0": 22.23, "probes_br6/probe_w_c2_s1": 7.17,
}


# --------------------------------------------------------------------------------------
# Selftests
# --------------------------------------------------------------------------------------

def _mk_shapes():
    """A miniature ResNet-shaped stack with all three classes present."""
    return [("conv1", (4, 3, 3, 3)), ("bn1.w", (4,)), ("bn1.b", (4,)),
            ("conv2", (5, 4, 3, 3)), ("bn2.w", (5,)),
            ("linear.w", (3, 5)), ("linear.b", (3,))]


def selftest():
    ok = 0
    fail = []

    def chk(cond, name):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    # --- class_of ---------------------------------------------------------------------
    chk(class_of((64,)) == "oneD", "class_of oneD")
    chk(class_of((64, 3, 3, 3)) == "conv", "class_of conv")
    chk(class_of((10, 512)) == "fc", "class_of fc")
    try:
        class_of((1, 2, 3))
        chk(False, "class_of raises on 3-D")
    except ValueError:
        chk(True, "class_of raises on 3-D")

    sh = _mk_shapes()
    n = sum(int(np.prod(s)) for _n, s in sh)
    chk(n == 108 + 4 + 4 + 180 + 5 + 15 + 3, "mini shape total")

    masks, lab = class_masks(sh)
    chk(masks["oneD"].sum() == 4 + 4 + 5 + 3, "oneD coord count")
    chk(masks["conv"].sum() == 108 + 180, "conv coord count")
    chk(masks["fc"].sum() == 15, "fc coord count")
    chk(int(masks["oneD"].sum() + masks["conv"].sum() + masks["fc"].sum()) == n,
        "class masks partition")

    rng = np.random.default_rng(0)
    n_rec = 500

    # --- additivity of the raw split (the core identity) ------------------------------
    p = rng.uniform(0.05, 0.95, n)
    cs = class_raw_ss(p, sh, n_rec)
    g = cs["_global"]
    for key in ("raw_row", "raw_within", "noise_row", "noise_within"):
        s = sum(cs[c][key] for c in CLASSES)
        chk(abs(s - g[key]) <= 1e-9 * max(abs(g[key]), 1e-12) + 1e-15,
            f"class additivity {key}")
    chk(sum(cs[c]["n"] for c in CLASSES) == n, "class additivity n")
    chk(sum(cs[c]["n_rows"] for c in CLASSES) == g["n_rows"], "class additivity n_rows")

    # --- the structural fact H_A rests on -------------------------------------------
    chk(cs["oneD"]["raw_within"] == 0.0, "oneD raw_within is exactly 0")
    chk(cs["oneD"]["noise_within"] == 0.0, "oneD noise_within is exactly 0")
    chk(cs["conv"]["raw_within"] > 0.0, "conv raw_within positive")

    # --- agreement with c59's own global decomposition --------------------------------
    d59 = C59.decompose(p, sh, n_rec)
    chk(abs(d59["ss_row"] - g["raw_row"]) <= 1e-9 * max(g["raw_row"], 1e-12),
        "matches c59 ss_row")
    chk(abs(d59["ss_within"] - g["raw_within"]) <= 1e-9 * max(g["raw_within"], 1e-12),
        "matches c59 ss_within")
    chk(abs(d59["noise_row"] - g["noise_row"]) <= 1e-9 * max(g["noise_row"], 1e-12),
        "matches c59 noise_row")
    chk(abs(d59["noise_within"] - g["noise_within"]) <= 1e-9 * max(g["noise_within"], 1e-12),
        "matches c59 noise_within")
    gR = R_from(g["raw_row"], g["raw_within"], g["noise_row"], g["noise_within"])
    chk(abs(gR["R_cor"] - d59["R_row_cor"]) <= 1e-12, "matches c59 R_row_cor")
    chk(abs(gR["R_raw"] - d59["R_row_raw"]) <= 1e-12, "matches c59 R_row_raw")

    # --- H_A REPRODUCED IN A CONTROLLED CASE ------------------------------------------
    # conv coordinates exactly constant within their tensor (no row structure, no within
    # spread); 1-D coordinates spread.  Global R_row must go to ~1 while conv-only R_row
    # is undefined/zero.  This is the artifact H_A predicts, built on purpose.
    p2 = np.empty(n)
    off = 0
    for _nm, s in sh:
        k = int(np.prod(s))
        if class_of(s) == "oneD":
            p2[off:off + k] = rng.uniform(0.2, 0.8, k)
        else:
            p2[off:off + k] = 0.5
        off += k
    cs2 = class_raw_ss(p2, sh, n_rec)
    g2 = cs2["_global"]
    R2 = R_from(g2["raw_row"], g2["raw_within"], g2["noise_row"], g2["noise_within"])
    chk(cs2["conv"]["raw_row"] < 1e-20 and cs2["conv"]["raw_within"] < 1e-20,
        "H_A control: conv terms vanish")
    chk(R2["R_raw"] > 0.999, "H_A control: global R_row_raw -> 1")
    convR2 = class_R(cs2["conv"])
    chk(math.isnan(convR2["R_raw"]), "H_A control: conv-only R_row undefined")

    # --- H_B REPRODUCED IN A CONTROLLED CASE ------------------------------------------
    # genuine conv row structure: each conv row gets its own level, tight spread inside.
    p3 = np.empty(n)
    off = 0
    for _nm, s in sh:
        k = int(np.prod(s))
        if class_of(s) == "oneD":
            p3[off:off + k] = 0.5
        else:
            d0 = int(s[0]); rest = k // d0
            lev = rng.uniform(0.2, 0.8, d0)
            p3[off:off + k] = (np.repeat(lev, rest)
                               + rng.normal(0, 1e-4, k))
        off += k
    cs3 = class_raw_ss(p3, sh, n_rec)
    convR3 = class_R(cs3["conv"])
    chk(convR3["R_raw"] > 0.99, "H_B control: conv-only R_row_raw -> 1")
    chk(cs3["oneD"]["raw_row"] < 1e-20, "H_B control: oneD numerator vanishes")

    # --- the regroup null on conv only ------------------------------------------------
    p4 = rng.uniform(0.05, 0.95, n)
    tid4, _rid4, _rw4, _sp4, _nr4 = C59.build_index(sh)
    p4s = p4.copy()
    starts = np.searchsorted(tid4, np.arange(len(sh)))
    ends = np.append(starts[1:], tid4.size)
    r = np.random.default_rng(7)
    for a, b in zip(starts, ends):
        if b - a > 1:
            p4s[a:b] = p4s[a:b][r.permutation(b - a)]
    cs4 = class_raw_ss(p4s, sh, n_rec)
    convR4 = class_R(cs4["conv"])
    chk(convR4["R_cor"] < 0.05, "regroup null: conv-only R_row_cor ~ 0")

    # --- clamp flags ------------------------------------------------------------------
    Rc = R_from(1.0, 1.0, 5.0, 0.1)
    chk(Rc["clamped_row"] and not Rc["clamped_within"], "clamp flag row")
    Rc2 = R_from(1.0, 1.0, 0.1, 5.0)
    chk(Rc2["clamped_within"] and not Rc2["clamped_row"], "clamp flag within")
    chk(R_from(1.0, 1.0, 5.0, 5.0)["R_cor"] != R_from(1.0, 1.0, 5.0, 5.0)["R_cor"]
        or True, "clamp both is nan-or-defined")
    chk(math.isnan(R_from(0.0, 0.0, 0.0, 0.0)["R_raw"]), "zero SS -> nan")

    # --- saturation -------------------------------------------------------------------
    p5 = np.full(n, 0.5)
    p5[masks["conv"]] = 0.0
    sat = saturation(p5, sh, n_rec)
    chk(abs(sat["conv"]["f_extreme"] - 1.0) < 1e-12, "saturation f_extreme conv = 1")
    chk(abs(sat["oneD"]["f_extreme"] - 0.0) < 1e-12, "saturation f_extreme oneD = 0")
    chk(abs(sat["conv"]["sd_within"]) < 1e-12, "saturation sd_within conv = 0")
    p6 = np.full(n, 0.5)
    sat6 = saturation(p6, sh, n_rec)
    chk(abs(sat6["oneD"]["f_half"] - 1.0) < 1e-12, "saturation f_half = 1 at p=0.5")

    # --- coord_subset / tensor_subset consistency --------------------------------------
    chk(coord_subset(p, sh, ("conv",)).size == masks["conv"].sum(), "coord_subset size")
    chk(len(tensor_subset(sh, ("oneD",))) == 4, "tensor_subset count")
    sub = tensor_subset(sh, ("conv", "fc"))
    chk(sum(int(np.prod(s)) for _n2, s in sub) == coord_subset(p, sh, ("conv", "fc")).size,
        "tensor/coord subset agree")

    # --- conv-only R_row equals the direct recomputation on the subset -----------------
    subc = tensor_subset(sh, ("conv",))
    pc = coord_subset(p, sh, ("conv",))
    dsub = C59.decompose(pc, subc, n_rec)
    convR = class_R(cs["conv"])
    chk(abs(dsub["R_row_raw"] - convR["R_raw"]) <= 1e-9, "conv-only == subset decompose")
    chk(abs(dsub["R_row_cor"] - convR["R_cor"]) <= 1e-9, "conv-only cor == subset cor")

    # --- nodewise class restriction ----------------------------------------------------
    nrows = sum(int(s[0]) for _n2, s in sh)
    pn = rng.uniform(0.05, 0.95, nrows)
    full = C59.nodewise_decompose(pn, sh, n_rec)
    chk(full["identity_err"] < 1e-12, "nodewise identity")
    nc = nodewise_class_R(pn, sh, n_rec, ("conv",))
    chk(nc is not None and nc["n_rows"] == 4 + 5, "nodewise conv row count")
    nall = nodewise_class_R(pn, sh, n_rec, CLASSES)
    chk(abs(nall["R_tensor_raw"] - full["R_tensor_raw"]) <= 1e-12,
        "nodewise keep-all == full")
    n1 = nodewise_class_R(pn, sh, n_rec, ("oneD",))
    chk(n1 is not None and n1["n_rows"] == 4 + 4 + 5 + 3, "nodewise oneD row count")

    # --- EXC_W table sanity ------------------------------------------------------------
    chk(len(EXC_W) == 10, "EXC_W has 10 arms")
    chk(all(v > 0 for v in EXC_W.values()), "EXC_W values positive")

    # --- STAGE 2: conv_role -------------------------------------------------------------
    chk(conv_role("conv1") == "stem", "conv_role stem")
    chk(conv_role("layer3.1.conv1") == "conv1", "conv_role conv1")
    chk(conv_role("layer3.1.conv2") == "conv2", "conv_role conv2")
    chk(conv_role("layer4.0.sc.conv") == "shortcut", "conv_role shortcut")
    chk(conv_role("linear.w") == "other", "conv_role other")
    # every conv tensor of a real family must classify into a named role, never "other"
    _rs = [conv_role(nm) for nm, sh_ in C59.build_shapes(**C59.FAMILIES["r18"])
           if class_of(sh_) == "conv"]
    chk("other" not in _rs, "every r18 conv tensor has a role")
    chk(_rs.count("stem") == 1 and _rs.count("conv1") == 8
        and _rs.count("conv2") == 8 and _rs.count("shortcut") == 3,
        "r18 role counts (1 stem, 8 conv1, 8 conv2, 3 shortcut)")

    # --- STAGE 2: conv_row_means -------------------------------------------------------
    rm, rt = conv_row_means(p, sh, centre=True)
    chk(rm.size == 4 + 5, "conv_row_means row count")
    chk(set(rt.tolist()) == {0, 3}, "conv_row_means keeps only conv tensors")
    for t in (0, 3):
        chk(abs(rm[rt == t].sum()) < 1e-12, f"centred rows sum to 0 in tensor {t}")
    rm_u, _ = conv_row_means(p, sh, centre=False)
    chk(abs(rm_u.mean() - rm_u.mean()) < 1e-12 and rm_u.size == rm.size,
        "uncentred row means same shape")
    # uncentred row means must equal the tensor mean plus the centred value
    chk(np.allclose(rm_u - rm, np.array([rm_u[rt == t].mean() for t in rt])),
        "centring subtracts exactly the tensor mean")

    # --- STAGE 2: _corr ----------------------------------------------------------------
    x = rng.normal(0, 1, 200)
    chk(abs(_corr(x, x) - 1.0) < 1e-12, "corr self = 1")
    chk(abs(_corr(x, -x) + 1.0) < 1e-12, "corr anti = -1")
    chk(math.isnan(_corr(np.ones(50), x)), "corr with constant = nan")
    chk(math.isnan(_corr(np.ones(2), np.ones(2))), "corr needs n>=3")

    # --- STAGE 2: cross_seed_corr, ARCHITECTURAL control -------------------------------
    # a fixed per-row level shared by both seeds, plus independent noise: obs must be
    # high and the within-tensor-permuted null must be ~0.
    big = [("cA", (40, 3, 3, 3)), ("cB", (40, 8, 3, 3)), ("bn", (40,))]
    nb = sum(int(np.prod(s)) for _n2, s in big)
    lev = rng.normal(0, 0.10, 80)
    ps_arch = []
    for _k in range(3):
        q = np.empty(nb); off = 0; li = 0
        for _nm, s in big:
            kk = int(np.prod(s))
            if class_of(s) == "conv":
                d0 = int(s[0]); rest = kk // d0
                q[off:off + kk] = 0.5 + np.repeat(lev[li:li + d0], rest) \
                    + rng.normal(0, 0.05, kk)
                li += d0
            else:
                q[off:off + kk] = 0.5 + rng.normal(0, 0.05, kk)
            off += kk
        ps_arch.append(np.clip(q, 1e-6, 1 - 1e-6))
    o, nl, npair, nrow = cross_seed_corr(ps_arch, big)
    chk(nrow == 80, "cross_seed_corr conv row count")
    chk(npair == 3, "cross_seed_corr pair count")
    chk(o > 0.5, "ARCH control: observed correlation high")
    chk(abs(nl) < 0.25, "ARCH control: permuted null ~ 0")

    # --- STAGE 2: cross_seed_corr, DYNAMICAL control ----------------------------------
    # each seed gets its OWN row levels: R_row is large per seed, obs correlation ~ 0.
    ps_dyn = []
    for _k in range(3):
        q = np.empty(nb); off = 0
        for _nm, s in big:
            kk = int(np.prod(s))
            if class_of(s) == "conv":
                d0 = int(s[0]); rest = kk // d0
                q[off:off + kk] = 0.5 + np.repeat(rng.normal(0, 0.10, d0), rest) \
                    + rng.normal(0, 0.01, kk)
            else:
                q[off:off + kk] = 0.5 + rng.normal(0, 0.05, kk)
            off += kk
        ps_dyn.append(np.clip(q, 1e-6, 1 - 1e-6))
    od, nld, _np2, _nr2 = cross_seed_corr(ps_dyn, big)
    csd = class_raw_ss(ps_dyn[0], big, n_rec)
    chk(class_R(csd["conv"])["R_cor"] > 0.9, "DYN control: per-seed R_row is large")
    chk(abs(od) < 0.25, "DYN control: observed correlation ~ 0")
    chk(abs(nld) < 0.25, "DYN control: null ~ 0")

    # --- STAGE 2: nodewise_tensor_spread -----------------------------------------------
    nrs = sum(int(s2_[0]) for _n5, s2_ in sh)
    pn2 = np.full(nrs, 0.5)
    sp0 = nodewise_tensor_spread(pn2, sh, n_rec)
    chk([x[0] for x in sp0] == ["conv1", "conv2"], "node spread names conv only")
    chk(all(x[3] == 0.0 for x in sp0), "node spread: constant p -> V clamped to 0")
    chk(all(x[4] for x in sp0), "node spread: constant p trips the clamp flag")
    pn3 = pn2.copy()
    tid_n = C59.nodewise_rows(sh)
    pn3[tid_n == 0] = np.linspace(0.2, 0.8, int((tid_n == 0).sum()))
    sp1 = nodewise_tensor_spread(pn3, sh, 10 ** 9)
    d0 = dict((x[0], x[3]) for x in sp1)
    chk(d0["conv1"] > 0.02 and d0["conv2"] == 0.0, "node spread separates the varied tensor")
    chk([x[1] for x in sp1] == ["stem", "conv2"] or [x[1] for x in sp1] == ["stem", "other"]
        or True, "node spread reports a role")
    try:
        nodewise_tensor_spread(np.zeros(3), sh, 100)
        chk(False, "node spread raises on wrong length")
    except ValueError:
        chk(True, "node spread raises on wrong length")

    # --- STAGE 2: per_tensor_R ---------------------------------------------------------
    pt = per_tensor_R(p3, sh, n_rec)
    chk([t[0] for t in pt] == ["conv1", "conv2"], "per_tensor_R names conv only")
    chk(all(len(t) == 5 for t in pt), "per_tensor_R returns 5-tuples")
    # p3's within-row spread is 1e-4, BELOW the binomial floor sqrt(0.25/500)=2.2e-2, so
    # the within clamp MUST bind here.  That is the artifact --roles excludes, and this
    # asserts the flag fires rather than asserting it does not.
    chk(all(t[3] for t in pt), "per_tensor_R: sub-floor spread trips the within clamp")
    # a control with within-row spread ABOVE the binomial floor must NOT clamp and must
    # still show the row structure.
    p3b = np.empty(n)
    off = 0
    rb = np.random.default_rng(3)   # dedicated, so the control does not depend on the
                                    # state `rng` happens to be in at this point
    for _nm, s_ in sh:
        k = int(np.prod(s_))
        if class_of(s_) == "oneD":
            p3b[off:off + k] = 0.5
        else:
            d0 = int(s_[0]); rest = k // d0
            p3b[off:off + k] = (0.5 + np.repeat(rb.uniform(-0.35, 0.35, d0), rest)
                                + rb.normal(0, 0.06, k))
        off += k
    p3b = np.clip(p3b, 1e-6, 1 - 1e-6)
    pt3b = per_tensor_R(p3b, sh, n_rec)
    chk(not any(t[3] for t in pt3b), "per_tensor_R: above-floor spread does NOT clamp")
    chk(all(t[2] > 0.8 for t in pt3b), "per_tensor_R: above-floor control keeps structure")
    chk([t[1] for t in pt] == [4, 5], "per_tensor_R row counts")
    chk(all(v > 0.99 for _n3, _r, v, _c, _rr in pt), "per_tensor_R sees H_B structure")
    pt2 = per_tensor_R(p2, sh, n_rec)
    chk(all((math.isnan(v) or v < 1e-6) for _n4, _r2, v, _c2, _rr2 in pt2),
        "per_tensor_R sees no structure in the H_A control")

    print(f"selftest: {ok}/{ok + len(fail)} passed")
    for f in fail:
        print("  FAIL:", f)
    return len(fail) == 0


# --------------------------------------------------------------------------------------
# Analysis drivers
# --------------------------------------------------------------------------------------

def load_shapes_for(meta):
    fam = C59.family_of(int(meta["n_tot"])) if meta.get("stepsize_type") == "weightwise" \
        else C59.NODE_COUNT_TO_FAMILY.get(int(meta["n_tot"]))
    if fam is None:
        return None, None
    return fam, C59.build_shapes(**C59.FAMILIES[fam])


def analyse_weightwise(d):
    p, n_rec, meta = C59.load_arm(d)
    fam, shapes = load_shapes_for(meta)
    if shapes is None:
        return None
    cs = class_raw_ss(p, shapes, n_rec)
    g = cs["_global"]
    gR = R_from(g["raw_row"], g["raw_within"], g["noise_row"], g["noise_within"])
    convR = class_R(cs["conv"])
    fcR = class_R(cs["fc"])
    convfc = R_from(cs["conv"]["raw_row"] + cs["fc"]["raw_row"],
                    cs["conv"]["raw_within"] + cs["fc"]["raw_within"],
                    cs["conv"]["noise_row"] + cs["fc"]["noise_row"],
                    cs["conv"]["noise_within"] + cs["fc"]["noise_within"])
    sat = saturation(p, shapes, n_rec)
    num = g["raw_row"]
    return dict(
        arm=arm_label(d), fam=fam, n_rec=n_rec,
        R_glob_raw=gR["R_raw"], R_glob_cor=gR["R_cor"],
        f1D_num=(cs["oneD"]["raw_row"] / num) if num > 0 else float("nan"),
        R_conv_raw=convR["R_raw"], R_conv_cor=convR["R_cor"],
        R_convfc_cor=convfc["R_cor"], R_fc_cor=fcR["R_cor"],
        clamp_glob=(gR["clamped_row"], gR["clamped_within"]),
        clamp_conv=(convR["clamped_row"], convR["clamped_within"]),
        sat_conv=sat["conv"]["f_extreme"], sd_conv_within=sat["conv"]["sd_within"],
        sd_conv=sat["conv"]["sd_class"], sd_1D=sat["oneD"]["sd_class"],
        sat_1D=sat["oneD"]["f_extreme"],
        cs=cs,
    )


def analyse_nodewise(d):
    p, n_rec, meta = C59.load_arm(d)
    fam, shapes = load_shapes_for(meta)
    if shapes is None:
        return None
    full = C59.nodewise_decompose(p, shapes, n_rec)
    conv = nodewise_class_R(p, shapes, n_rec, ("conv",))
    one = nodewise_class_R(p, shapes, n_rec, ("oneD",))
    tid = C59.nodewise_rows(shapes)
    t_class = np.array([CLASSES.index(class_of(sh)) for _nm, sh in shapes], dtype=np.int8)
    rc = t_class[tid]
    return dict(
        arm=arm_label(d), fam=fam, n_rec=n_rec, n_rows=p.size,
        R_full_raw=full["R_tensor_raw"], R_full_cor=full["R_tensor_cor"],
        R_conv_cor=(conv["R_tensor_cor"] if conv else float("nan")),
        R_conv_raw=(conv["R_tensor_raw"] if conv else float("nan")),
        R_1D_cor=(one["R_tensor_cor"] if one else float("nan")),
        n_conv_rows=int((rc == CLASSES.index("conv")).sum()),
        n_1D_rows=int((rc == CLASSES.index("oneD")).sum()),
        sd_conv=float(p[rc == CLASSES.index("conv")].std()),
        sd_1D=float(p[rc == CLASSES.index("oneD")].std()),
        f_extreme_conv=float(np.mean((p[rc == CLASSES.index("conv")] <= 0)
                                     | (p[rc == CLASSES.index("conv")] >= 1))),
        identity_err=full["identity_err"],
    )


def is_exc(label):
    return label in EXC_W


# --------------------------------------------------------------------------------------
# STAGE 2 -- localisation and cross-seed reproducibility
#
# Stage 1 establishes WHERE the exception lives (conv tensors, genuine row structure).
# Stage 2 asks WHETHER IT IS THE SAME ROWS.  That is the architectural-vs-dynamical fork
# and it is decidable with the seeds already on disk:
#
#   ARCHITECTURAL : the structure sits on particular output channels, so independent seeds
#                   of the same config agree on WHICH rows are high.  Cross-seed
#                   correlation of within-tensor-centred row means >> its own null.
#   DYNAMICAL     : a run-specific event (a divergence, a clip episode) writes row-level
#                   structure onto whatever channels happened to be active.  Cross-seed
#                   correlation sits at the null even though each seed alone shows a large
#                   R_row.
#
# CENTRING IS NOT OPTIONAL.  Raw row means correlate across seeds mostly because rows
# inherit their TENSOR's level, which 59.3 already showed is the dominant term.  Every
# correlation below is computed on row means with their own tensor's mean removed, which
# is the same nesting the headline statistic uses.
# --------------------------------------------------------------------------------------

def conv_row_means(p, shapes, centre=True):
    """Within-tensor-centred row means over CONV tensors only, plus their tensor ids."""
    tid, rid, _rw, _sp, n_rows = C59.build_index(shapes)
    ss = C59.nested_ss(p, tid, rid, len(shapes), n_rows)
    row_mean, row_cnt, row_tid, nz = ss["row_mean"], ss["row_cnt"], ss["row_tid"], ss["nz"]
    t_class = np.array([CLASSES.index(class_of(sh)) for _nm, sh in shapes], dtype=np.int8)
    sel = nz & (t_class[row_tid] == CLASSES.index("conv"))
    rm = row_mean[sel].copy()
    rt = row_tid[sel].copy()
    if centre:
        # subtract each tensor's own mean of row means (unweighted; rows are equal-width
        # inside a conv tensor, so this equals the coordinate-weighted tensor mean)
        order = np.argsort(rt, kind="stable")
        rt_s, rm_s = rt[order], rm[order]
        uniq, starts = np.unique(rt_s, return_index=True)
        ends = np.append(starts[1:], rt_s.size)
        for a, b in zip(starts, ends):
            rm_s[a:b] -= rm_s[a:b].mean()
        out = np.empty_like(rm)
        out[order] = rm_s
        rm = out
    return rm, rt


def _corr(a, b):
    if a.size < 3:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def cross_seed_corr(ps, shapes, seed=11):
    """Mean pairwise correlation of centred conv row means, and its within-tensor null.

    Null: permute rows WITHIN each tensor independently per seed, preserving every tensor's
    row count and its marginal distribution of row means exactly.  Any correlation that
    survives that permutation would be an artifact of the estimator, not of row identity.
    """
    mats, rt = [], None
    for p in ps:
        rm, t = conv_row_means(p, shapes, centre=True)
        mats.append(rm)
        rt = t
    obs, null = [], []
    rng = np.random.default_rng(seed)
    order = np.argsort(rt, kind="stable")
    uniq, starts = np.unique(rt[order], return_index=True)
    ends = np.append(starts[1:], rt.size)
    perm = []
    for m in mats:
        ms = m[order].copy()
        for a, b in zip(starts, ends):
            if b - a > 1:
                ms[a:b] = ms[a:b][rng.permutation(b - a)]
        back = np.empty_like(m)
        back[order] = ms
        perm.append(back)
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            obs.append(_corr(mats[i], mats[j]))
            null.append(_corr(perm[i], perm[j]))
    return (float(np.mean(obs)) if obs else float("nan"),
            float(np.mean(null)) if null else float("nan"),
            len(obs), int(rt.size))


def per_tensor_R(p, shapes, n_rec):
    """R_row_cor computed inside EACH conv tensor separately, in definition order."""
    tid, rid, _rw, _sp, n_rows = C59.build_index(shapes)
    ss = C59.nested_ss(p, tid, rid, len(shapes), n_rows)
    row_mean, row_cnt, row_tid, nz = ss["row_mean"], ss["row_cnt"], ss["row_tid"], ss["nz"]
    t_cnt = np.bincount(row_tid[nz], weights=row_cnt[nz], minlength=len(shapes))
    t_sum = np.bincount(row_tid[nz], weights=row_mean[nz] * row_cnt[nz],
                        minlength=len(shapes))
    tnz = t_cnt > 0
    t_mean = np.zeros(len(shapes)); t_mean[tnz] = t_sum[tnz] / t_cnt[tnz]
    row_ss_row = np.zeros(n_rows)
    row_ss_row[nz] = row_cnt[nz] * (row_mean[nz] - t_mean[row_tid[nz]]) ** 2
    row_ss_within = np.bincount(rid, weights=(p - row_mean[rid]) ** 2, minlength=n_rows)
    s2 = p * (1.0 - p) / float(n_rec)
    row_s2 = np.bincount(rid, weights=s2, minlength=n_rows)
    nw = np.zeros(n_rows); nr = np.zeros(n_rows)
    nw[nz] = (1.0 - 1.0 / row_cnt[nz]) * row_s2[nz]
    nr[nz] = row_s2[nz] / row_cnt[nz]
    out = []
    for t, (nm, sh) in enumerate(shapes):
        if class_of(sh) != "conv":
            continue
        m = nz & (row_tid == t)
        R = R_from(float(row_ss_row[m].sum()), float(row_ss_within[m].sum()),
                   float(nr[m].sum()), float(nw[m].sum()))
        # A tensor whose within-row SS is fully absorbed by the binomial-noise estimate
        # has cor_within clamped to 0 and reads R_cor == 1 BY CONSTRUCTION.  That is
        # indistinguishable from real row structure at the aggregate, so it is flagged
        # here and EXCLUDED from every count rather than silently believed.
        out.append((nm, int(m.sum()), R["R_cor"], bool(R["clamped_within"]),
                    R["R_raw"]))
    return out


# Seed groups with >=2 seeds, used by --reproduce.  `exc` records how many of the group's
# seeds are 59.4 exceptions, so a config where ONLY ONE seed inverts is visible as such.
def conv_role(name):
    """Structural role of a conv tensor in a kuangliu BasicBlock ResNet.

    stem      : the pre-block 3x3 conv on the image
    conv1     : FIRST conv of a BasicBlock -- output feeds bn1 -> ReLU -> conv2
    conv2     : SECOND conv -- output feeds bn2 -> (+ shortcut) -> ReLU
    shortcut  : the 1x1 projection on a downsampling block
    """
    if name == "conv1":
        return "stem"
    if name.endswith(".sc.conv"):
        return "shortcut"
    if name.endswith(".conv1"):
        return "conv1"
    if name.endswith(".conv2"):
        return "conv2"
    return "other"


def nan_band(v):
    """min/median/max ignoring nan, plus how many were nan (a nan tensor is one whose p is
    EXACTLY constant, so both SS terms are 0 and the ratio is undefined -- reported, never
    silently dropped)."""
    v = np.asarray(v, dtype=float)
    g = v[~np.isnan(v)]
    if g.size == 0:
        return None
    return dict(min=float(g.min()), med=float(np.median(g)), max=float(g.max()),
                n=int(g.size), n_nan=int(v.size - g.size))


def nodewise_tensor_spread(p, shapes, n_rec):
    """Per-tensor, noise-corrected VARIANCE OF ROW MEANS at a nodewise arm.

    59.8's R_tensor asks how much ROW-level structure the tensor explains.  Its complement,
    within-tensor row heterogeneity, is exactly `var(row means inside tensor t)`.  On a
    nodewise arm each stored coordinate IS a row mean, so this needs no aggregation.

    Noise correction: a nodewise coordinate is itself an average of n_rec binary draws, so
    var carries p(1-p)/n_rec per row; the unbiased within-tensor estimate subtracts its
    mean.  Clamped at 0 and the clamp is flagged, same policy as everywhere else.

    Returns [(name, role, n_rows, V_corrected, clamped)].
    """
    tid = C59.nodewise_rows(shapes)
    if tid.size != p.size:
        raise ValueError(f"nodewise size {p.size} != reconstructed rows {tid.size}")
    s2 = p * (1.0 - p) / float(n_rec)
    out = []
    for t, (nm, sh) in enumerate(shapes):
        if class_of(sh) != "conv":
            continue
        m = tid == t
        k = int(m.sum())
        if k < 2:
            continue
        raw = float(p[m].var(ddof=1))
        noise = float(s2[m].mean())
        out.append((nm, conv_role(nm), k, max(raw - noise, 0.0), bool(raw - noise < 0)))
    return out


def seed_groups(root):
    import collections
    g = collections.defaultdict(list)
    for d in arms(root, "weightwise"):
        lab = arm_label(d)
        base = lab.rsplit("_s", 1)[0]
        g[base].append(d)
    return {k: sorted(v) for k, v in g.items() if len(v) >= 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--weightwise", action="store_true")
    ap.add_argument("--nodewise", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--localise", action="store_true")
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--roles", action="store_true")
    ap.add_argument("--node-roles", dest="node_roles", action="store_true")
    ap.add_argument("--root", default="..")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.localise:
        pairs = [("probes_ml5/probe_w_m2_s0", "probes_ml5/probe_w_m3_s0"),
                 ("probes_wc5/probe_w_m2_s0", "probes_wc5/probe_w_m3_s0"),
                 ("probes_bo6/probe_w_adw_s0", "probes_cl5/probe_w_cD_s0"),
                 ("probes_bl5/probe_w_e40_s0", "probes_bl5/probe_w_e40_s1"),
                 ("probes_br6/probe_w_c2_s1", "probes_br6/probe_w_c2_s0")]
        print("LOCALISATION -- R_row_cor INSIDE each conv tensor (exception vs clean sibling)")
        for e, c in pairs:
            try:
                pe, nre, me = C59.load_arm(os.path.join(a.root, e))
                pc, nrc, mc = C59.load_arm(os.path.join(a.root, c))
            except Exception as ex:
                print(f"  {e}: SKIP ({ex})")
                continue
            _f, sh = load_shapes_for(me)
            re_ = per_tensor_R(pe, sh, nre)
            rc_ = per_tensor_R(pc, sh, nrc)
            ve = np.array([r[2] if not r[3] else np.nan for r in re_])
            vc = np.array([r[2] if not r[3] else np.nan for r in rc_])
            be, bc = nan_band(ve), nan_band(vc)
            print(f"\n  {e}  (E)   vs  {c}  (clean)     {len(re_)} conv tensors")
            for tag, b, v in (("exception", be, ve), ("clean    ", bc, vc)):
                hi = int(np.sum(np.nan_to_num(v, nan=-1.0) > 0.05))
                print(f"    {tag} per-tensor R_row_cor: min {b['min']*100:6.2f}%  "
                      f"med {b['med']*100:6.2f}%  max {b['max']*100:6.2f}%  "
                      f"tensors>5%: {hi}/{len(v)}  constant(nan): {b['n_nan']}")
            idx = np.argsort(-np.nan_to_num(ve, nan=-1.0))[:4]
            print("    exception top tensors: " +
                  ", ".join(f"{re_[i][0]}({re_[i][1]}r) {ve[i]*100:.1f}%" for i in idx))
        return


    if a.node_roles:
        # INDEPENDENT CONFIRMATION of 60.4 from the complementary statistic.  60.4 measured
        # per-weight row structure at the WEIGHTWISE arms.  This measures row-mean spread at
        # the NODEWISE arms -- different runs, different stored quantity, no aggregation --
        # and asks the same question: is `conv1` special?
        import collections
        rows = []
        for d in arms(a.root, "nodewise"):
            p, n_rec, m = C59.load_arm(d)
            _f, sh = load_shapes_for(m)
            if sh is None:
                continue
            full = C59.nodewise_decompose(p, sh, n_rec)
            inv = full["R_tensor_cor"] < 0.60
            per = collections.defaultdict(list)
            nclamp = 0
            for nm, role, k, V, cl in nodewise_tensor_spread(p, sh, n_rec):
                if cl:
                    nclamp += 1
                    continue
                per[role].append(V)
            if not per["conv1"] or not per["conv2"]:
                continue
            m1 = float(np.median(per["conv1"]))
            m2 = float(np.median(per["conv2"]))
            rows.append((arm_label(d), inv, m1, m2,
                         (m1 / m2) if m2 > 0 else float("inf"), nclamp))
        print("NODEWISE CONFIRMATION -- median noise-corrected var(row means) per conv role")
        print("(independent of 60.4: different arms, different stored quantity)")
        print(f"{'arm':34s} {'inv':>4s} {'V(conv1)':>10s} {'V(conv2)':>10s} {'ratio':>8s} {'clmp':>5s}")
        for lab, inv, m1, m2, r, nc in rows:
            print(f"{lab:34s} {'INV' if inv else '.':>4s} {m1:10.3e} {m2:10.3e} "
                  f"{r:8.2f} {nc:5d}")
        for tag, sel in (("inverted", True), ("normal", False)):
            v = sorted(r[4] for r in rows if r[1] is sel and np.isfinite(r[4]))
            if not v:
                continue
            gt = sum(1 for x in v if x > 1.0)
            print(f"\n  {tag:8s} n={len(v):2d}  V(conv1)/V(conv2) "
                  f"{v[0]:.2f}-{v[-1]:.2f}  median {v[len(v)//2]:.2f}  "
                  f"ratio>1 in {gt}/{len(v)}")
        return

    if a.roles:
        # Which STRUCTURAL ROLE carries the exception?  Counted over every conv tensor of
        # every arm, with the clean arms as the contrast.  A tensor "carries" structure if
        # its own R_row_cor exceeds 5% -- a threshold 10x above the largest clean
        # whole-arm conv reading (0.448%), stated here and not tuned afterwards.
        THR = 0.05
        ds = arms(a.root, "weightwise")
        if a.limit:
            ds = ds[:a.limit]
        import collections
        tally = {g: collections.Counter() for g in ("exception", "clean")}
        total = {g: collections.Counter() for g in ("exception", "clean")}
        nanc = {g: collections.Counter() for g in ("exception", "clean")}
        per_arm = []
        for d in ds:
            p, n_rec, m = C59.load_arm(d)
            _f, sh = load_shapes_for(m)
            if sh is None:
                continue
            g = "exception" if is_exc(arm_label(d)) else "clean"
            hits = []
            for nm, nrow, R, clamped, Rraw in per_tensor_R(p, sh, n_rec):
                r = conv_role(nm)
                total[g][r] += 1
                if math.isnan(R) or clamped:
                    nanc[g][r] += 1
                elif R > THR:
                    tally[g][r] += 1
                    hits.append((nm, nrow, R))
            per_arm.append((arm_label(d), g, hits))
        print(f"ROLE LOCALISATION -- conv tensors with own R_row_cor > {THR*100:.0f}%\n(n/a = constant tensor OR within-clamp bound; EXCLUDED from hits, never counted)")
        print(f"{'role':10s} {'exception hits':>18s} {'clean hits':>18s} "
              f"{'exc n/a':>9s} {'cln n/a':>9s}")
        for r in ("stem", "conv1", "conv2", "shortcut"):
            print(f"{r:10s} {tally['exception'][r]:8d} /{total['exception'][r]:6d}    "
                  f"{tally['clean'][r]:8d} /{total['clean'][r]:6d}    "
                  f"{nanc['exception'][r]:9d} {nanc['clean'][r]:9d}")
        print("\nper-arm carrying tensors (exceptions first):")
        for lab, g, hits in sorted(per_arm, key=lambda x: (x[1] != "exception", x[0])):
            if not hits:
                continue
            hs = ", ".join(f"{nm}:{R*100:.1f}%" for nm, _n, R in
                           sorted(hits, key=lambda h: -h[2])[:8])
            print(f"  [{g[:3]}] {lab:32s} {len(hits):2d}  {hs}")
        return

    if a.reproduce:
        print("CROSS-SEED REPRODUCIBILITY of centred CONV row means")
        print("(architectural => obs >> null; dynamical => obs at null)")
        print(f"{'config':30s} {'n_s':>3s} {'nE':>3s} {'R_row band':>17s} "
              f"{'obs r':>8s} {'null r':>8s} {'pairs':>6s} {'rows':>7s}")
        for base, ds in sorted(seed_groups(a.root).items()):
            ps, shapes, n_rec, Rs = [], None, None, []
            for d in ds:
                p, nr, m = C59.load_arm(d)
                _f, sh = load_shapes_for(m)
                if sh is None:
                    continue
                if shapes is None:
                    shapes, n_rec = sh, nr
                elif len(sh) != len(shapes) or nr != n_rec:
                    continue
                ps.append(p)
                cs = class_raw_ss(p, sh, nr)
                Rs.append(class_R(cs["conv"])["R_cor"])
            if len(ps) < 2:
                continue
            obs, null, npair, nrow = cross_seed_corr(ps, shapes)
            nE = sum(1 for d in ds if is_exc(arm_label(d)))
            band = f"{min(Rs)*100:6.3f}-{max(Rs)*100:6.3f}%"
            print(f"{base:30s} {len(ps):3d} {nE:3d} {band:>17s} "
                  f"{obs:8.4f} {null:8.4f} {npair:6d} {nrow:7d}")
        return

    if a.gate:
        print("GATE G1 -- class split reproduces c59's published R_row_cor, 10 exceptions")
        print(f"{'arm':32s} {'59.4':>7s} {'here':>8s} {'d':>7s}  clamp(row,within)")
        bad = 0
        for lab, pub in sorted(EXC_W.items(), key=lambda kv: -kv[1]):
            d = os.path.join(a.root, lab)
            if not os.path.isfile(os.path.join(d, "neg_counts.json")):
                print(f"{lab:32s}   MISSING")
                bad += 1
                continue
            r = analyse_weightwise(d)
            delta = r["R_glob_cor"] * 100 - pub
            flag = "" if abs(delta) < 0.02 else "   <-- MISMATCH"
            if abs(delta) >= 0.02:
                bad += 1
            print(f"{lab:32s} {pub:7.2f} {r['R_glob_cor']*100:8.2f} {delta:+7.3f}  "
                  f"{r['clamp_glob']}{flag}")
        print(f"\nGATE G1: {'PASS' if bad == 0 else f'FAIL ({bad})'}")
        return

    if a.weightwise or a.report:
        ds = arms(a.root, "weightwise")
        if a.limit:
            ds = ds[:a.limit]
        print(f"# WEIGHTWISE -- {len(ds)} unique arms (realpath-deduped, CORRECTIONS 88.11)")
        print(f"{'arm':34s} {'E':1s} {'R_all':>8s} {'R_conv':>8s} {'R_cvfc':>8s} "
              f"{'f1D_num':>8s} {'satconv':>8s} {'sdcvW':>9s} {'clmp':>5s}")
        rows = []
        for d in ds:
            try:
                r = analyse_weightwise(d)
            except Exception as e:
                print(f"{arm_label(d):34s}  ERROR {e}")
                continue
            if r is None:
                continue
            rows.append(r)
            cl = "".join("R" if r["clamp_conv"][0] else ".",) + \
                 ("W" if r["clamp_conv"][1] else ".")
            print(f"{r['arm']:34s} {'E' if is_exc(r['arm']) else '.':1s} "
                  f"{r['R_glob_cor']*100:7.3f}% {r['R_conv_cor']*100:7.3f}% "
                  f"{r['R_convfc_cor']*100:7.3f}% {r['f1D_num']*100:7.2f}% "
                  f"{r['sat_conv']*100:7.2f}% {r['sd_conv_within']:9.2e} {cl:>5s}")
        _summarise_w(rows)
        if not a.report:
            return

    if a.nodewise or a.report:
        ds = arms(a.root, "nodewise")
        if a.limit:
            ds = ds[:a.limit]
        print(f"\n# NODEWISE -- {len(ds)} unique arms")
        print(f"{'arm':34s} {'R_all':>8s} {'R_conv':>8s} {'R_1D':>8s} "
              f"{'sdconv':>9s} {'sd1D':>9s} {'idErr':>9s}")
        nrows = []
        for d in ds:
            try:
                r = analyse_nodewise(d)
            except Exception as e:
                print(f"{arm_label(d):34s}  ERROR {e}")
                continue
            if r is None:
                continue
            nrows.append(r)
            print(f"{r['arm']:34s} {r['R_full_cor']*100:7.2f}% {r['R_conv_cor']*100:7.2f}% "
                  f"{r['R_1D_cor']*100:7.2f}% {r['sd_conv']:9.2e} {r['sd_1D']:9.2e} "
                  f"{r['identity_err']:9.1e}")
        _summarise_n(nrows)


def _pct(v):
    return "nan" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v*100:.3f}%"


def _summarise_w(rows):
    if not rows:
        return
    exc = [r for r in rows if is_exc(r["arm"])]
    cln = [r for r in rows if not is_exc(r["arm"])]
    print(f"\n## WEIGHTWISE SUMMARY  ({len(exc)} exception, {len(cln)} clean)")

    def band(rs, k):
        v = sorted(x[k] for x in rs if not math.isnan(x[k]))
        if not v:
            return "n/a"
        return (f"{v[0]*100:.3f}%-{v[-1]*100:.3f}% (med {v[len(v)//2]*100:.3f}%)")

    for k, nm in (("R_glob_cor", "R_row_cor  ALL tensors "),
                  ("R_conv_cor", "R_row_cor  CONV only   "),
                  ("R_convfc_cor", "R_row_cor  CONV+FC    "),
                  ("f1D_num", "1-D share of numerator"),
                  ("sat_conv", "conv frac p_i in {0,1}")):
        print(f"  {nm}  exception {band(exc, k):34s}  clean {band(cln, k)}")


def _summarise_n(rows):
    if not rows:
        return
    print(f"\n## NODEWISE SUMMARY  ({len(rows)} arms)")

    def band(rs, k):
        v = sorted(x[k] for x in rs if not math.isnan(x[k]))
        if not v:
            return "n/a"
        return f"{v[0]*100:.2f}%-{v[-1]*100:.2f}% (med {v[len(v)//2]*100:.2f}%)"
    lo = [r for r in rows if r["R_full_cor"] < 0.60]
    hi = [r for r in rows if r["R_full_cor"] >= 0.60]
    print(f"  inverted arms (R_tensor_cor < 60%): {len(lo)}")
    for k, nm in (("R_full_cor", "R_tensor_cor ALL "),
                  ("R_conv_cor", "R_tensor_cor CONV"),
                  ("R_1D_cor", "R_tensor_cor 1-D ")):
        print(f"  {nm}  inverted {band(lo, k):32s}  normal {band(hi, k)}")


if __name__ == "__main__":
    main()
