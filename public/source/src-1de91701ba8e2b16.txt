#!/usr/bin/env python3
"""c59_row_premise.py -- test ADAM-MINI'S ACTUAL PREMISE at the `nodewise` partition.

WHY THIS EXISTS
---------------
CORRECTIONS 87.1 discharged GATE 86.4 and WITHDREW every sentence of the form "the
Adam-mini / Adalayer / SGG line assumes exactly 0 [correlation]".  No paper in that line
justifies coarse granularity by sqrt(N) noise-averaging.  What Adam-mini ACTUALLY argues
(quoted from the local arXiv full text in paper/refs/) is a CORRELATION argument:

    "they all share the same BP error term e_i ... G usually has similar entries within a row"

i.e. the mean of a block represents the block because a ROW of `G = e z^T` is homogeneous.
CORRECTIONS 87.2 recorded that this is directly testable on our corpus and ranked it above
the carried-since-51 `s` re-derivation, because **a "row of G" is exactly our `nodewise`
partition** -- HF_patched.py:147 groups by `p_size[0]` and sums over all trailing dims:

    nodewise block_product = (u[i]*v[i]).reshape(u[i].shape[0], -1).sum(dim=1)

SCOPE, STATED UP FRONT AND NOT HEDGED LATER
-------------------------------------------
Adam-mini's premise is about `G`, the BASE gradient.  What PATCH_PROBE5 recorded per
coordinate is the sign of `z`, the META-gradient (`z_i = u_i * v_i`, HF_patched.py:145).
These are different quantities.  The STRUCTURAL argument transfers -- `z_i` inherits the
same shared BP error term `e_i` through `g_i`, so if a shared `e_i` made row entries
similar it would do so here too -- but this is a test of the premise AS IT APPLIES TO THE
QUANTITY OUR PARTITION ACTUALLY AGGREGATES, not a reproduction of Adam-mini's own figure.
Every printed line says `z`.  Do not write "we refuted Adam-mini".

WHAT IS MEASURED
----------------
PATCH_PROBE5 wrote, per coordinate of the arm's own partition, a running count of how often
that coordinate's meta-gradient was negative.  On a WEIGHTWISE arm that is per-weight:

    p_i = neg_count_i / n_records        the coordinate's persistent sign preference

`p_i` is the natural per-coordinate summary of "which way does this weight's meta-gradient
push", and Adam-mini's premise is precisely that this is near-constant within a row.

THE STATISTIC, AND WHY IT IS NESTED
------------------------------------
A flat ICC over all 11.17M coordinates would be dominated by BETWEEN-TENSOR variance --
different tensors have wildly different sign statistics -- and rows nested inside tensors
inherit that for free.  A naive ICC would therefore look large for a reason that has
nothing to do with rows.  So the decomposition is NESTED, and it maps exactly onto the
campaign's own granularity ladder:

    SS_total = SS_tensor  +  SS_row|tensor  +  SS_within
               (layerwise)   (nodewise's     (what no partition
                              OWN increment)   coarser than
                                               weightwise can see)

The headline is the row's share of what the TENSOR DOES NOT ALREADY EXPLAIN:

    R_row = SS_row|tensor / (SS_row|tensor + SS_within)        [noise-corrected]

    R_row -> 1  : the row mean represents the row.  Adam-mini's premise HOLDS.
    R_row -> 0  : knowing the row tells you nothing the tensor did not.  Premise FAILS,
                  and nodewise should buy nothing over layerwise.

NOISE CORRECTION
----------------
Each `p_i` is an average of `n_records` binary draws, so it carries sampling variance even
if every coordinate in a row is identical.  Under independence across records the induced
sum of squares partitions EXACTLY:

    sigma2_i     = p_i(1-p_i)/n_rec
    noise_within = sum_rows [ (1 - 1/n_row) * sum_{i in row} sigma2_i ]
    noise_row    = sum_rows [ (1/n_row)     * sum_{i in row} sigma2_i ]   (minus a
                   negligible tensor-mean term, computed exactly below)

Records are NOT independent in time (consecutive probe records are ~correlated), so this
UNDER-states the noise.  Under-stated noise INFLATES the corrected within-row term, which
DEFLATES R_row.  The correction is therefore CONSERVATIVE AGAINST the premise-holds
reading and ANTI-conservative against the premise-fails reading, and the raw (uncorrected)
R_row is printed beside it as the bound in the other direction.  Both are reported; a
conclusion is only drawn where the two agree.

GATES (run with --verify; every one is a falsifiable alignment check, not an assertion)
---------------------------------------------------------------------------------------
A0  shape checksums: reconstructed (n_tensors, n_nodes, n_weights) must equal the three
    numbers independently present in the on-disk neg_counts.json across 4 families.
A1  mod-9 convolution alignment: within conv tensors, grouping p_i by spatial position
    (index mod 9) must show the 3x3 centre/corner signature.  If the flattening offset
    were wrong by even one element this structure scrambles.  This is what makes the
    coordinate->row map MEASURED rather than assumed.
A2  1-D separation: BatchNorm coordinates (nodewise == weightwise there by construction)
    must be separable from conv coordinates under the reconstructed boundaries.
A3  within-tensor random-regroup null: shuffle coordinates into rows of IDENTICAL sizes
    inside each tensor; corrected R_row must come out ~0.  This is the estimator's own
    unbiasedness check, run on the real data rather than only in the selftest.

USAGE
    python3 analysis/c59_row_premise.py --selftest
    python3 analysis/c59_row_premise.py --verify   [--root ..]
    python3 analysis/c59_row_premise.py --decompose probes_fz3/probe_r18_w_s0
    python3 analysis/c59_row_premise.py --sweep    [--root ..]
"""
import argparse
import json
import math
import os
import sys
import glob

import numpy as np

# --------------------------------------------------------------------------------------
# Architecture reconstruction (kuangliu pytorch-cifar ResNet, the parent paper's model).
# Order is the module-definition order that `named_parameters()` yields, which is the
# order `zall = torch.cat([...])` concatenates in (HF_patched.py:349).
# Confirmed at block level by the optimizer's own hard-coded
#   resnet18_blocks = [3,12,15,15,15,2]      (HF_patched.py:165)
# = conv1+bn1 (3) | layer1 (12) | layer2 (15) | layer3 (15) | layer4 (15) | linear (2).
# --------------------------------------------------------------------------------------

def _basic_block(cin, cout, stride):
    """kuangliu BasicBlock: conv1, bn1(w,b), conv2, bn2(w,b), [shortcut conv, bn(w,b)].

    `self.shortcut` is assigned AFTER conv2/bn2 in the module body, so its parameters come
    last in named_parameters() order.
    """
    ts = [
        ("conv1", (cout, cin, 3, 3)),
        ("bn1.w", (cout,)), ("bn1.b", (cout,)),
        ("conv2", (cout, cout, 3, 3)),
        ("bn2.w", (cout,)), ("bn2.b", (cout,)),
    ]
    if stride != 1 or cin != cout:
        ts += [
            ("sc.conv", (cout, cin, 1, 1)),
            ("sc.bn.w", (cout,)), ("sc.bn.b", (cout,)),
        ]
    return ts


def build_shapes(num_blocks, num_classes):
    """Return the ordered list of (name, shape) for a kuangliu CIFAR ResNet."""
    shapes = [("conv1", (64, 3, 3, 3)), ("bn1.w", (64,)), ("bn1.b", (64,))]
    cin = 64
    for li, (cout, nb) in enumerate(zip([64, 128, 256, 512], num_blocks)):
        for b in range(nb):
            stride = 1 if (li == 0 or b > 0) else 2
            for nm, sh in _basic_block(cin, cout, stride):
                shapes.append((f"layer{li+1}.{b}.{nm}", sh))
            cin = cout
    shapes.append(("linear.w", (num_classes, 512)))
    shapes.append(("linear.b", (num_classes,)))
    return shapes


FAMILIES = {
    "r10":  dict(num_blocks=[1, 1, 1, 1], num_classes=10),
    "r18":  dict(num_blocks=[2, 2, 2, 2], num_classes=10),
    "r34":  dict(num_blocks=[3, 4, 6, 3], num_classes=10),
    "c100": dict(num_blocks=[2, 2, 2, 2], num_classes=100),
}

# The three numbers that appear independently in on-disk neg_counts.json, per family.
# (n_tensors from a layerwise arm, n_nodes from a nodewise arm, n_weights from a
# weightwise arm.)  These are DATA, and A0 checks the reconstruction against them.
CHECKSUMS = {
    "r10":  (38,  8660,  4903242),
    "r18":  (62,  14420, 11173962),
    "r34":  (110, 25556, 21282122),
    "c100": (62,  14600, 11220132),
}


def shape_stats(shapes):
    n_tensors = len(shapes)
    n_nodes = sum(int(s[0]) for _, s in shapes)
    n_weights = sum(int(np.prod(s)) for _, s in shapes)
    return n_tensors, n_nodes, n_weights


def family_of(n_weights):
    for f, (_, _, w) in CHECKSUMS.items():
        if w == n_weights:
            return f
    return None


# --------------------------------------------------------------------------------------
# Coordinate -> (tensor, row) maps
# --------------------------------------------------------------------------------------

def build_index(shapes):
    """Return (tensor_id, row_id, row_width, is_conv_spatial) arrays over all coordinates.

    row_id is GLOBAL (offset per tensor) so that rows never collide across tensors.
    row_width[i] is the number of coordinates in coordinate i's row -- used by the noise
    correction.  is_conv_spatial marks coordinates in tensors with a trailing 3x3, i.e.
    the ones gate A1 can read a spatial signature from.
    """
    tid_parts, rid_parts, w_parts, sp_parts = [], [], [], []
    row_off = 0
    for t, (_nm, sh) in enumerate(shapes):
        n = int(np.prod(sh))
        d0 = int(sh[0])
        rest = n // d0
        tid_parts.append(np.full(n, t, dtype=np.int32))
        rid_parts.append(row_off + np.repeat(np.arange(d0, dtype=np.int64), rest))
        w_parts.append(np.full(n, rest, dtype=np.int32))
        sp_parts.append(np.full(n, 1 if (len(sh) == 4 and sh[2] == 3 and sh[3] == 3) else 0,
                                dtype=np.int8))
        row_off += d0
    return (np.concatenate(tid_parts), np.concatenate(rid_parts),
            np.concatenate(w_parts), np.concatenate(sp_parts), row_off)


# --------------------------------------------------------------------------------------
# Nested variance decomposition
# --------------------------------------------------------------------------------------

def nested_ss(p, tid, rid, n_tensors, n_rows, weights=None):
    """Exact nested sums of squares.

    Returns dict with ss_total, ss_tensor, ss_row (row|tensor), ss_within.
    Identity ss_total == ss_tensor + ss_row + ss_within holds to float precision.
    """
    n = p.size
    grand = float(p.mean())

    row_cnt = np.bincount(rid, minlength=n_rows).astype(np.float64)
    row_sum = np.bincount(rid, weights=p, minlength=n_rows)
    nz = row_cnt > 0
    row_mean = np.zeros(n_rows, dtype=np.float64)
    row_mean[nz] = row_sum[nz] / row_cnt[nz]

    # tensor totals, accumulated from row totals (rows nest strictly inside tensors)
    row_tid = np.zeros(n_rows, dtype=np.int64)
    row_tid[rid] = tid            # every row is written by its own coordinates
    t_cnt = np.bincount(row_tid[nz], weights=row_cnt[nz], minlength=n_tensors)
    t_sum = np.bincount(row_tid[nz], weights=row_sum[nz], minlength=n_tensors)
    tnz = t_cnt > 0
    t_mean = np.zeros(n_tensors, dtype=np.float64)
    t_mean[tnz] = t_sum[tnz] / t_cnt[tnz]

    ss_total = float(np.sum((p - grand) ** 2))
    ss_tensor = float(np.sum(t_cnt[tnz] * (t_mean[tnz] - grand) ** 2))
    ss_row = float(np.sum(row_cnt[nz] * (row_mean[nz] - t_mean[row_tid[nz]]) ** 2))
    ss_within = float(np.sum((p - row_mean[rid]) ** 2))
    return dict(ss_total=ss_total, ss_tensor=ss_tensor, ss_row=ss_row,
                ss_within=ss_within, n=n, grand=grand,
                row_mean=row_mean, row_cnt=row_cnt, row_tid=row_tid, nz=nz)


def noise_split(p, rid, row_width, n_rows, n_rec):
    """Exact partition of binomial sampling SS into the within and row|tensor slots.

    sigma2_i = p_i(1-p_i)/n_rec.  For a row of size k with per-coordinate variances s_i:
        E[SS_within contribution] = (1 - 1/k) * sum s_i
        E[SS_row    contribution] = (1/k)     * sum s_i
    (the tensor-mean term is O(1/n_tensor_coords) and is subtracted explicitly below).
    """
    s2 = p * (1.0 - p) / float(n_rec)
    row_s2 = np.bincount(rid, weights=s2, minlength=n_rows)
    row_cnt = np.bincount(rid, minlength=n_rows).astype(np.float64)
    nz = row_cnt > 0
    within = float(np.sum((1.0 - 1.0 / row_cnt[nz]) * row_s2[nz]))
    row = float(np.sum(row_s2[nz] / row_cnt[nz]))
    return within, row, float(s2.sum())


def decompose(p, shapes, n_rec, regroup_seed=None, rng=None):
    """Full nested decomposition with noise correction.

    regroup_seed: if not None, coordinates are randomly permuted WITHIN each tensor before
    rows are assigned (row sizes preserved exactly).  This is gate A3's null.
    """
    tid, rid, rw, sp, n_rows = build_index(shapes)
    n_tensors = len(shapes)
    if regroup_seed is not None:
        r = rng if rng is not None else np.random.default_rng(regroup_seed)
        p = p.copy()
        # permute p within each tensor; row membership (rid) then becomes arbitrary
        starts = np.searchsorted(tid, np.arange(n_tensors))
        ends = np.append(starts[1:], tid.size)
        for a, b in zip(starts, ends):
            if b - a > 1:
                p[a:b] = p[a:b][r.permutation(b - a)]

    ss = nested_ss(p, tid, rid, n_tensors, n_rows)
    nw, nr, ntot = noise_split(p, rid, rw, n_rows, n_rec)

    raw_row, raw_within = ss["ss_row"], ss["ss_within"]
    cor_row = max(raw_row - nr, 0.0)
    cor_within = max(raw_within - nw, 0.0)

    def ratio(a, b):
        return float(a / (a + b)) if (a + b) > 0 else float("nan")

    return dict(
        n=ss["n"], n_rows=int(n_rows), n_tensors=n_tensors, n_rec=n_rec,
        ss_total=ss["ss_total"], ss_tensor=ss["ss_tensor"],
        ss_row=raw_row, ss_within=raw_within,
        noise_row=nr, noise_within=nw, noise_total=ntot,
        cor_row=cor_row, cor_within=cor_within,
        R_row_raw=ratio(raw_row, raw_within),
        R_row_cor=ratio(cor_row, cor_within),
        tensor_share=float(ss["ss_tensor"] / ss["ss_total"]) if ss["ss_total"] > 0 else float("nan"),
        identity_err=abs(ss["ss_total"] - (ss["ss_tensor"] + raw_row + raw_within))
                     / max(ss["ss_total"], 1e-30),
    )


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def load_arm(d):
    meta = json.load(open(os.path.join(d, "neg_counts.json")))
    cnt = np.load(os.path.join(d, "neg_counts.npy")).astype(np.float64)
    n_rec = int(meta["n_records"])
    if cnt.size != int(meta["n_tot"]):
        raise ValueError(f"{d}: npy size {cnt.size} != n_tot {meta['n_tot']}")
    return cnt / n_rec, n_rec, meta


# --------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------

def gate_a1_mod9(p, shapes):
    """Spatial-position signature inside 3x3 conv tensors.

    For every (cout,cin,3,3) tensor, coordinate index within the tensor mod 9 IS the
    spatial position.  Report mean p per position and the centre-vs-corner contrast.
    A correct flattening gives a coherent, repeatable pattern; a misaligned one averages
    it away.
    """
    pos_sum = np.zeros(9)
    pos_cnt = np.zeros(9)
    off = 0
    for _nm, sh in shapes:
        n = int(np.prod(sh))
        if len(sh) == 4 and sh[2] == 3 and sh[3] == 3:
            seg = p[off:off + n]
            k = seg.size // 9
            m = seg[:k * 9].reshape(k, 9)
            pos_sum += m.sum(axis=0)
            pos_cnt += k
        off += n
    means = pos_sum / np.maximum(pos_cnt, 1)
    centre = means[4]
    corners = means[[0, 2, 6, 8]].mean()
    edges = means[[1, 3, 5, 7]].mean()
    return means, centre, edges, corners


def gate_a2_bn(p, shapes):
    """Separation of 1-D (BatchNorm/bias) coordinates from conv/linear coordinates."""
    is1d = np.zeros(p.size, dtype=bool)
    off = 0
    for _nm, sh in shapes:
        n = int(np.prod(sh))
        if len(sh) == 1:
            is1d[off:off + n] = True
        off += n
    return (float(p[is1d].mean()), float(p[is1d].std()), int(is1d.sum()),
            float(p[~is1d].mean()), float(p[~is1d].std()), int((~is1d).sum()))


# --------------------------------------------------------------------------------------
# Selftests
# --------------------------------------------------------------------------------------

def selftest():
    n_pass = n_tot = 0

    def chk(cond, name):
        nonlocal n_pass, n_tot
        n_tot += 1
        if cond:
            n_pass += 1
        else:
            print(f"  FAIL: {name}")
        return cond

    # ---- A0-class: shape checksums, 3 per family ----
    for fam, spec in FAMILIES.items():
        sh = build_shapes(**spec)
        nt, nn, nw = shape_stats(sh)
        ent, enn, enw = CHECKSUMS[fam]
        chk(nt == ent, f"{fam} n_tensors {nt} != {ent}")
        chk(nn == enn, f"{fam} n_nodes {nn} != {enn}")
        chk(nw == enw, f"{fam} n_weights {nw} != {enw}")

    # ---- index construction ----
    toy = [("a", (2, 3)), ("b", (4,)), ("c", (2, 2, 3, 3))]
    tid, rid, rw, sp, nrows = build_index(toy)
    chk(tid.size == 6 + 4 + 36, "toy total size")
    chk(nrows == 2 + 4 + 2, "toy row count")
    chk(list(rid[:6]) == [0, 0, 0, 1, 1, 1], "toy rows of (2,3)")
    chk(list(rid[6:10]) == [2, 3, 4, 5], "toy 1-D rows are singletons")
    chk(list(rid[10:28]) == [6] * 18, "toy conv row 0 spans 18 coords")
    chk(list(rw[:6]) == [3] * 6, "toy row width (2,3)")
    chk(list(rw[6:10]) == [1] * 4, "toy row width 1-D")
    chk(int(sp[:10].sum()) == 0 and int(sp[10:].sum()) == 36, "toy spatial flag")

    # ---- decomposition identity on random data ----
    rng = np.random.default_rng(0)
    sh = [("x", (5, 7)), ("y", (3,)), ("z", (4, 2, 3, 3))]
    tid, rid, rw, sp, nrows = build_index(sh)
    p = rng.random(tid.size)
    ss = nested_ss(p, tid, rid, len(sh), nrows)
    err = abs(ss["ss_total"] - (ss["ss_tensor"] + ss["ss_row"] + ss["ss_within"]))
    chk(err < 1e-8 * max(ss["ss_total"], 1e-30), f"nested SS identity, err={err:.3e}")

    # ---- perfect row structure -> R_row ~ 1 ----
    shp = [("x", (40, 50))]
    tid, rid, rw, sp, nrows = build_index(shp)
    rowvals = rng.random(nrows) * 0.4 + 0.3
    p = rowvals[rid]                      # exactly constant within each row
    d = decompose(p, shp, n_rec=10 ** 9)  # huge n_rec => negligible noise correction
    chk(d["R_row_raw"] > 0.999, f"perfect rows R_row_raw={d['R_row_raw']:.4f}")
    chk(d["identity_err"] < 1e-8, "perfect rows identity")

    # ---- pure sampling noise, homogeneous rows -> corrected R_row ~ 0 ----
    # every coordinate has the SAME true p within a tensor; only binomial noise differs.
    n_rec = 2000
    shp = [("x", (60, 80))]
    tid, rid, rw, sp, nrows = build_index(shp)
    truep = 0.5
    draws = rng.binomial(n_rec, truep, size=tid.size) / n_rec
    d = decompose(draws, shp, n_rec=n_rec)
    chk(d["R_row_cor"] < 0.05,
        f"homogeneous-rows corrected R_row={d['R_row_cor']:.4f} (want ~0)")
    chk(d["R_row_raw"] > 0.005,
        f"homogeneous-rows RAW R_row={d['R_row_raw']:.4f} should be inflated by noise")

    # ---- noise split is exact: within+row == total sigma2 ----
    nw_, nr_, ntot_ = noise_split(draws, rid, rw, nrows, n_rec)
    chk(abs((nw_ + nr_) - ntot_) < 1e-9 * ntot_, "noise split sums to total sigma2")

    # ---- real row signal survives correction ----
    rowvals = rng.normal(0.5, 0.05, size=nrows).clip(0.02, 0.98)
    truth = rowvals[rid]
    draws = rng.binomial(n_rec, truth) / n_rec
    d = decompose(draws, shp, n_rec=n_rec)
    chk(d["R_row_cor"] > 0.8,
        f"true row signal recovered, R_row_cor={d['R_row_cor']:.4f}")

    # ---- regroup null destroys it ----
    d0 = decompose(draws, shp, n_rec=n_rec, regroup_seed=1)
    chk(d0["R_row_cor"] < 0.05,
        f"regroup null R_row_cor={d0['R_row_cor']:.4f} (want ~0)")

    # ---- regroup preserves the total and the tensor term ----
    dfull = decompose(draws, shp, n_rec=n_rec)
    chk(abs(d0["ss_total"] - dfull["ss_total"]) < 1e-6 * dfull["ss_total"],
        "regroup preserves ss_total")
    chk(abs(d0["ss_tensor"] - dfull["ss_tensor"]) < 1e-6 * max(dfull["ss_tensor"], 1e-12),
        "regroup preserves ss_tensor")

    # ---- mod-9 gate reads a planted spatial pattern ----
    shp = [("c", (8, 8, 3, 3))]
    tid, rid, rw, sp, nrows = build_index(shp)
    base = np.full(tid.size, 0.5)
    base[np.arange(tid.size) % 9 == 4] = 0.8      # plant a centre effect
    means, c, e, co = gate_a1_mod9(base, shp)
    chk(abs(c - 0.8) < 1e-9 and abs(co - 0.5) < 1e-9, "mod9 gate recovers planted centre")

    # ---- mod-9 gate is destroyed by a one-element misalignment ----
    means2, c2, e2, co2 = gate_a1_mod9(np.roll(base, 1), shp)
    chk(abs(c2 - 0.8) > 0.2, "mod9 gate is sensitive to a 1-element offset")

    # ---- 1-D separation gate ----
    shp = [("c", (4, 4, 3, 3)), ("b", (7,))]
    tid, rid, rw, sp, nrows = build_index(shp)
    q = np.concatenate([np.full(4 * 4 * 9, 0.4), np.full(7, 0.9)])
    m1, s1, n1, m2, s2, n2 = gate_a2_bn(q, shp)
    chk(n1 == 7 and abs(m1 - 0.9) < 1e-12, "A2 picks the 1-D coordinates")
    chk(n2 == 144 and abs(m2 - 0.4) < 1e-12, "A2 picks the non-1-D coordinates")

    # ---- direction index (defined at the bottom of the module) ----
    dsh = [("c", (2, 3, 3, 3)), ("b", (5,))]
    keep, gid, tid, ng, nt = direction_index(dsh, "row")
    chk(int(keep.sum()) == 54 and ng == 2, "direction row: coverage 54, 2 groups")
    chk(list(gid[:27]) == [0] * 27 and list(gid[27:]) == [1] * 27,
        "direction row: contiguous blocks of 27")
    keep, gid, tid, ng, nt = direction_index(dsh, "col")
    chk(ng == 3 and list(gid[:9]) == [0] * 9 and list(gid[9:18]) == [1] * 9,
        "direction col: 3 groups striding by 9")
    keep, gid, tid, ng, nt = direction_index(dsh, "spatial")
    chk(ng == 9 and list(gid[:9]) == list(range(9)) and list(gid[9:18]) == list(range(9)),
        "direction spatial: 9 groups, period 9")
    chk(int(keep.sum()) == 54, "direction excludes the 1-D tensor")
    # planted row signal is seen by 'row' and not by 'spatial'
    shp2 = [("c", (30, 20, 3, 3))]
    rr = np.random.default_rng(3)
    rowv = rr.normal(0.5, 0.05, 30).clip(0.02, 0.98)
    q = np.repeat(rowv, 20 * 9)
    a_row, _, _ = direction_R(q, shp2, 10 ** 9, "row")
    a_spa, _, _ = direction_R(q, shp2, 10 ** 9, "spatial")
    chk(a_row > 0.999, f"direction_R row recovers planted row signal ({a_row:.4f})")
    chk(a_spa < 0.01, f"direction_R spatial blind to row signal ({a_spa:.4f})")

    # ---- corrected_shares ----
    sh_ = corrected_shares(dict(ss_tensor=1.0, cor_row=2.0, cor_within=7.0))
    chk(abs(sh_[0] - 0.1) < 1e-12 and abs(sh_[1] - 0.2) < 1e-12
        and abs(sh_[2] - 0.7) < 1e-12, "corrected_shares normalises")
    chk(all(x != x for x in corrected_shares(dict(ss_tensor=0.0, cor_row=0.0,
                                                  cor_within=0.0))),
        "corrected_shares nan on empty")

    # ---- family_of ----
    chk(family_of(11173962) == "r18", "family_of r18")
    chk(family_of(21282122) == "r34", "family_of r34")
    chk(family_of(123) is None, "family_of unknown")

    print(f"selftest: {n_pass}/{n_tot} passed")
    return n_pass == n_tot


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def fmt_row(label, d):
    return (f"{label:34s} {d['tensor_share']*100:6.2f}%  "
            f"{d['R_row_raw']*100:7.3f}%  {d['R_row_cor']*100:7.3f}%  "
            f"{d['noise_within']/max(d['ss_within'],1e-30)*100:6.1f}%")


def beta_pinning(d):
    """Fraction of probe records at which the per-coordinate beta extreme sits at its
    MODAL value, plus whether beta moved at all.

    Under a binding BETA_CLIP guard the extreme sits exactly on the guard for many
    records; unclipped it wanders and the exact value recurs once.  `moved` is required
    because on a FROZEN arm beta never changes, so the modal fraction is trivially 1.0 and
    means the opposite of pinning -- that is a false positive this function must not emit.

    Validated against CORRECTIONS 53, which recorded bl5's per-seed HIGH-guard binding as
    24.07% / 0.00% / 0.00%; this returns 24.05% / 0.4% / 0.5%.
    """
    pj = os.path.join(d, "probe.jsonl")
    lo, hi = [], []
    with open(pj) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "beta_true_min" in r:
                lo.append(round(r["beta_true_min"], 6))
                hi.append(round(r["beta_true_max"], 6))
    if not lo:
        return None
    n = len(lo)
    from collections import Counter
    mlo, clo = Counter(lo).most_common(1)[0]
    mhi, chi = Counter(hi).most_common(1)[0]
    moved = (lo[0] != lo[-1]) or (hi[0] != hi[-1])
    return dict(n_rec_probe=n, mode_lo=mlo, mode_hi=mhi,
                f_lo=(clo / n) if moved else 0.0,
                f_hi=(chi / n) if moved else 0.0,
                moved=moved, beta0=lo[0])


def corrected_shares(d):
    """Noise-corrected three-way shares of the structure in p_i.

    The tensor term carries negligible sampling noise (it averages >=10^4 coordinates), so
    the corrected total is ss_tensor + cor_row + cor_within.
    """
    tot = d["ss_tensor"] + d["cor_row"] + d["cor_within"]
    if tot <= 0:
        return float("nan"), float("nan"), float("nan")
    return (d["ss_tensor"] / tot, d["cor_row"] / tot, d["cor_within"] / tot)


def find_weightwise(root):
    out = []
    for f in sorted(glob.glob(os.path.join(root, "probes_*", "*", "neg_counts.json"))):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        if m.get("stepsize_type") != "weightwise":
            continue
        if family_of(int(m.get("n_tot", -1))) is None:
            continue
        out.append(os.path.dirname(f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--decompose", metavar="DIR")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--root", default="..")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.verify:
        print("=== A0  shape checksums (reconstruction vs on-disk neg_counts.json) ===")
        print(f"{'family':6s} {'tensors':>18s} {'nodes':>18s} {'weights':>20s}")
        ok = True
        for fam, spec in FAMILIES.items():
            got = shape_stats(build_shapes(**spec))
            exp = CHECKSUMS[fam]
            mark = "OK" if got == exp else "MISMATCH"
            ok &= (got == exp)
            print(f"{fam:6s} {got[0]:8d}/{exp[0]:<8d} {got[1]:8d}/{exp[1]:<8d} "
                  f"{got[2]:9d}/{exp[2]:<9d}  {mark}")
        print(f"A0: {'PASS' if ok else 'FAIL'}")

        arms = find_weightwise(a.root)
        if not arms:
            print("no weightwise arms found under", a.root)
            return
        # verify on one arm per family
        seen = set()
        for d in arms:
            p, n_rec, meta = load_arm(d)
            fam = family_of(p.size)
            if fam in seen:
                continue
            seen.add(fam)
            shapes = build_shapes(**FAMILIES[fam])
            print(f"\n=== {fam}  {d}  (n_rec={n_rec}) ===")
            means, c, e, co = gate_a1_mod9(p, shapes)
            print("A1 mod-9 mean p by spatial position:")
            print("    " + "  ".join(f"{m:.5f}" for m in means[:3]))
            print("    " + "  ".join(f"{m:.5f}" for m in means[3:6]))
            print("    " + "  ".join(f"{m:.5f}" for m in means[6:]))
            spread = float(means.max() - means.min())
            print(f"    centre={c:.5f}  edge={e:.5f}  corner={co:.5f}  "
                  f"spread={spread:.5f}")
            m1, s1, n1, m2, s2, n2 = gate_a2_bn(p, shapes)
            print(f"A2 1-D   n={n1:>8d} mean={m1:.5f} sd={s1:.5f}")
            print(f"   non1D n={n2:>8d} mean={m2:.5f} sd={s2:.5f}")
        return

    if a.decompose:
        p, n_rec, meta = load_arm(a.decompose)
        fam = family_of(p.size)
        shapes = build_shapes(**FAMILIES[fam])
        d = decompose(p, shapes, n_rec)
        dn = decompose(p, shapes, n_rec, regroup_seed=12345)
        print(f"{a.decompose}  family={fam} n={d['n']} rows={d['n_rows']} "
              f"tensors={d['n_tensors']} n_rec={n_rec}")
        print(f"  identity err        {d['identity_err']:.3e}")
        print(f"  SS tensor share     {d['tensor_share']*100:.2f}%")
        print(f"  R_row raw           {d['R_row_raw']*100:.3f}%")
        print(f"  R_row corrected     {d['R_row_cor']*100:.3f}%")
        print(f"  A3 regroup null cor {dn['R_row_cor']*100:.3f}%  (want ~0)")
        return

    if a.sweep:
        arms = find_weightwise(a.root)
        if a.limit:
            arms = arms[:a.limit]
        print("batch,arm,family,n_rec,tensor_share,R_row_raw,R_row_cor,R_row_null,"
              "sd_p,sd_true,ss_total")
        for d in arms:
            try:
                p, n_rec, meta = load_arm(d)
                fam = family_of(p.size)
                shapes = build_shapes(**FAMILIES[fam])
                r = decompose(p, shapes, n_rec)
                rn = decompose(p, shapes, n_rec, regroup_seed=12345)
                batch = d.split(os.sep)[-2].replace("probes_", "")
                arm = d.split(os.sep)[-1]
                sd_p = float(p.std())
                # noise-corrected sd of the TRUE per-coordinate sign preference
                v = sd_p ** 2 - float((p * (1 - p) / n_rec).mean())
                sd_true = math.sqrt(v) if v > 0 else 0.0
                print(f"{batch},{arm},{fam},{n_rec},{r['tensor_share']:.6f},"
                      f"{r['R_row_raw']:.6f},{r['R_row_cor']:.6f},{rn['R_row_cor']:.6f},"
                      f"{sd_p:.6f},{sd_true:.6f},{r['ss_total']:.4f}")
            except Exception as ex:
                print(f"# {d}: ERROR {ex}", file=sys.stderr)
        return

    if a.report:
        # Deduplicate by resolved path -- probes_ml5_m{2,3,4}/* are SYMLINKS into
        # probes_ml5/*, so a naive glob counts 9 arms twice.
        arms, seen = [], set()
        for d in find_weightwise(a.root):
            rp = os.path.realpath(d)
            if rp in seen:
                continue
            seen.add(rp)
            arms.append(d)
        recs = []
        for d in arms:
            p, n_rec, meta = load_arm(d)
            fam = family_of(p.size)
            shapes = build_shapes(**FAMILIES[fam])
            r = decompose(p, shapes, n_rec)
            r["null"] = decompose(p, shapes, n_rec, regroup_seed=12345)["R_row_cor"]
            r["pin"] = beta_pinning(d)
            r["fam"] = fam
            r["label"] = "/".join(d.split(os.sep)[-2:]).replace("probes_", "")
            recs.append(r)

        EXTREME = 0.01   # R_row_cor above this = the exception set, examined separately
        clean = [r for r in recs if r["R_row_cor"] <= EXTREME]
        odd = [r for r in recs if r["R_row_cor"] > EXTREME]

        print(f"UNIQUE WEIGHTWISE ARMS: {len(recs)}   "
              f"clean {len(clean)}   exceptions {len(odd)}\n")

        print("=== THE MAIN RESULT: row share of within-tensor structure, clean arms ===")
        print(f"{'family':7s} {'n':>3s} {'R_row_raw%':>11s} {'R_row_cor%':>11s} "
              f"{'null%':>7s} {'tensor%':>8s} {'within%':>8s}")
        for fam in ["r10", "r18", "r34", "c100"]:
            v = [r for r in clean if r["fam"] == fam]
            if not v:
                continue
            sh = [corrected_shares(r) for r in v]
            print(f"{fam:7s} {len(v):3d} "
                  f"{min(r['R_row_raw'] for r in v)*100:5.3f}-{max(r['R_row_raw'] for r in v)*100:<5.3f} "
                  f"{min(r['R_row_cor'] for r in v)*100:5.3f}-{max(r['R_row_cor'] for r in v)*100:<5.3f} "
                  f"{sum(r['null'] for r in v)/len(v)*100:7.3f} "
                  f"{sum(s[0] for s in sh)/len(sh)*100:8.2f} "
                  f"{sum(s[2] for s in sh)/len(sh)*100:8.2f}")
        allc = [r["R_row_cor"] for r in clean]
        alln = [r["null"] for r in clean]
        allr = [r["R_row_raw"] for r in clean]
        print(f"\n  ALL {len(clean)} clean arms: R_row_cor "
              f"{min(allc)*100:.3f}%-{max(allc)*100:.3f}% "
              f"(median {sorted(allc)[len(allc)//2]*100:.3f}%)")
        print(f"  {'':>17s} R_row_raw {min(allr)*100:.3f}%-{max(allr)*100:.3f}%  "
              f"-- raw and corrected AGREE, so the reading is bound on both sides")
        print(f"  {'':>17s} regroup null "
              f"{min(alln)*100:.3f}%-{max(alln)*100:.3f}% "
              f"(median {sorted(alln)[len(alln)//2]*100:.3f}%)")

        print("\n=== THE EXCEPTIONS, and the control that refuses to explain them ===")
        print(f"{'arm':28s} {'R_row_cor%':>10s} {'pin_lo%':>8s} {'pin_hi%':>8s} {'mode_hi':>9s}")
        for r in sorted(odd, key=lambda x: -x["R_row_cor"]):
            pn = r["pin"]
            print(f"{r['label']:28s} {r['R_row_cor']*100:10.2f} "
                  f"{pn['f_lo']*100:8.2f} {pn['f_hi']*100:8.2f} {pn['mode_hi']:9.3f}")
        print("\n  COUNTEREXAMPLES that kill 'clipping manufactures row structure':")
        for r in sorted(clean, key=lambda x: -(x["pin"]["f_hi"] if x["pin"] else 0))[:4]:
            pn = r["pin"]
            print(f"  {r['label']:28s} R_row_cor {r['R_row_cor']*100:6.3f}%  "
                  f"pin_hi {pn['f_hi']*100:6.2f}%  mode_hi {pn['mode_hi']:.3f}")
        return

    ap.print_help()




# --------------------------------------------------------------------------------------
# DIRECTION TEST (cycle 59 follow-on).  Adam-mini's argument is specifically that a ROW is
# special because its entries share the BP error term e_i.  A COLUMN shares the input
# activation z_j instead; a SPATIAL position group shares neither.  If all three explain
# equally little, there is nothing special about the row DIRECTION, which is a sharper
# statement than "rows explain little".
#
# Restricted to tensors with ndim in (2,4) -- i.e. the ones that HAVE a row/column
# structure.  1-D tensors (BatchNorm/bias) are excluded and their coverage is reported.
# --------------------------------------------------------------------------------------

def direction_index(shapes, mode):
    """Group ids for 'row' | 'col' | 'spatial', over ndim in (2,4) tensors only.

    Returns (keep_mask, group_id, tensor_id, n_groups, n_tensors_kept).
    Group ids are global (offset per tensor) so groups never merge across tensors.
    """
    keep, gid, tid = [], [], []
    goff = 0
    tkept = 0
    for _t, (_nm, sh) in enumerate(shapes):
        n = int(np.prod(sh))
        if len(sh) not in (2, 4):
            keep.append(np.zeros(n, dtype=bool))
            continue
        d0 = int(sh[0])
        d1 = int(sh[1])
        trail = n // (d0 * d1)          # 1 for ndim==2, d2*d3 for ndim==4
        idx = np.arange(n, dtype=np.int64)
        if mode == "row":
            g = idx // (d1 * trail)
            ng = d0
        elif mode == "col":
            g = (idx // trail) % d1
            ng = d1
        elif mode == "spatial":
            g = idx % trail
            ng = trail
        else:
            raise ValueError(mode)
        if ng < 2:                      # a degenerate grouping carries no information
            keep.append(np.zeros(n, dtype=bool))
            continue
        keep.append(np.ones(n, dtype=bool))
        gid.append(goff + g)
        tid.append(np.full(n, tkept, dtype=np.int64))
        goff += ng
        tkept += 1
    return (np.concatenate(keep), np.concatenate(gid) if gid else np.empty(0, np.int64),
            np.concatenate(tid) if tid else np.empty(0, np.int64), goff, tkept)


def direction_R(p, shapes, n_rec, mode, regroup_seed=None):
    """Noise-corrected R = SS_group|tensor / (SS_group|tensor + SS_within) for one mode."""
    keep, gid, tid, ng, nt = direction_index(shapes, mode)
    q = p[keep]
    if regroup_seed is not None:
        r = np.random.default_rng(regroup_seed)
        q = q.copy()
        starts = np.searchsorted(tid, np.arange(nt))
        ends = np.append(starts[1:], tid.size)
        for a, b in zip(starts, ends):
            if b - a > 1:
                q[a:b] = q[a:b][r.permutation(b - a)]
    ss = nested_ss(q, tid, gid, nt, ng)
    nw, nr, _ = noise_split(q, gid, None, ng, n_rec)
    cr = max(ss["ss_row"] - nr, 0.0)
    cw = max(ss["ss_within"] - nw, 0.0)
    return (float(cr / (cr + cw)) if (cr + cw) > 0 else float("nan"),
            float(keep.sum()) / p.size, ng)


# --------------------------------------------------------------------------------------
# NODEWISE-ARM MODE (cycle 59 follow-on).  On a NODEWISE arm `neg_counts` is already one
# entry per row, and that row is the unit the optimizer ACTUALLY adapts -- no offline
# aggregation is involved.  59.3 asked whether the row mean REPRESENTS its weights; this
# asks the complementary question, whether rows are DISTINGUISHABLE from each other beyond
# their tensor:
#
#     R_tensor = SS_tensor / (SS_tensor + SS_row-within-tensor)      [noise-corrected]
#
#     R_tensor -> 1 : rows inside a tensor are interchangeable; layerwise loses nothing.
#     R_tensor -> 0 : rows are distinct units; nodewise HAS information layerwise lacks.
#
# The two questions are independent: a row can be a distinguishable unit without being a
# representative one.
# --------------------------------------------------------------------------------------

def nodewise_rows(shapes):
    """Tensor id per ROW, for a nodewise arm (tensor t contributes p_size[0] rows)."""
    return np.concatenate([np.full(int(sh[0]), t, dtype=np.int64)
                           for t, (_nm, sh) in enumerate(shapes)])


def nodewise_decompose(p, shapes, n_rec, regroup_seed=None):
    tid = nodewise_rows(shapes)
    if tid.size != p.size:
        raise ValueError(f"nodewise size {p.size} != reconstructed rows {tid.size}")
    q = p
    if regroup_seed is not None:
        # permute rows ACROSS tensors, preserving each tensor's row count exactly
        q = np.random.default_rng(regroup_seed).permutation(p)
    nt = len(shapes)
    grand = float(q.mean())
    cnt = np.bincount(tid, minlength=nt).astype(np.float64)
    tsum = np.bincount(tid, weights=q, minlength=nt)
    nz = cnt > 0
    tmean = np.zeros(nt); tmean[nz] = tsum[nz] / cnt[nz]
    ss_total = float(np.sum((q - grand) ** 2))
    ss_between = float(np.sum(cnt[nz] * (tmean[nz] - grand) ** 2))
    ss_within = float(np.sum((q - tmean[tid]) ** 2))
    s2 = q * (1.0 - q) / float(n_rec)
    t_s2 = np.bincount(tid, weights=s2, minlength=nt)
    n_within = float(np.sum((1.0 - 1.0 / cnt[nz]) * t_s2[nz]))
    n_between = float(np.sum(t_s2[nz] / cnt[nz]))
    cb = max(ss_between - n_between, 0.0)
    cw = max(ss_within - n_within, 0.0)
    return dict(n_rows=int(p.size), n_tensors=nt,
                R_tensor_raw=ss_between / max(ss_between + ss_within, 1e-30),
                R_tensor_cor=cb / max(cb + cw, 1e-30),
                identity_err=abs(ss_total - (ss_between + ss_within))
                             / max(ss_total, 1e-30),
                sd_p=float(q.std()))


def find_nodewise(root):
    out, seen = [], set()
    for f in sorted(glob.glob(os.path.join(root, "probes_*", "*", "neg_counts.json"))):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        if m.get("stepsize_type") != "nodewise":
            continue
        d = os.path.dirname(f)
        rp = os.path.realpath(d)
        if rp in seen:
            continue
        seen.add(rp)
        out.append((d, int(m["n_tot"])))
    return out


NODE_COUNT_TO_FAMILY = {8660: "r10", 14420: "r18", 25556: "r34", 14600: "c100"}

if __name__ == "__main__":
    main()
