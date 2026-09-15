#!/usr/bin/env python3
r"""c62_blocksize_curve.py -- THE BLOCK-SIZE CURVE, at ARBITRARY block size, offline.

WHY THIS EXISTS
---------------
The operator's DEFAULT direction (C) is: *"characterise how agreement varies with block
size."*  FINDINGS 52.12 recorded that this cannot be done:

    "Nor can the curve be recovered offline: `probe5_window`'s k-profile takes one k PER
     ARM, from `n_tot/m` of the granularity that was actually run, and `rho_s` is derived
     from that arm's `frac_neg` time series -- it is not a post-hoc re-blocking of stored
     per-coordinate counts.  So filling the layerwise<->weightwise interval requires a new
     `stepsize_type` in `HF.py` ... flagged here as the one place where the C-direction
     programme needs an operator decision rather than another batch."

STANDING RULE (11), installed last cycle after exactly this kind of error (CORRECTIONS
90.1): *before recording a question as unanswerable from stored data, name the stored field
and the reason it cannot answer it.*  Applying it to 52.12:

  * The named field for 52.12's statistic is `frac_neg`, a per-record POOLED scalar.  It has
    no coordinate resolution, so `rho_s` at an arbitrary block size is genuinely
    unrecoverable.  **52.12 IS CORRECT ABOUT `rho_s`, AND IS NOT WITHDRAWN.**
  * But `rho_s` is not the only channel.  FINDINGS 43.3 established that the agreement
    excess decomposes into a CORRELATION channel (needs joint/temporal signs -> `frac_neg`,
    -> 52.12 binds) and a MARGINAL-BIAS channel (each coordinate's own persistent sign
    preference).  PATCH_PROBE5 writes `neg_counts.npy`: a per-coordinate negative count,
    11.17M of them on an R18 weightwise arm.  **The marginal channel HAS coordinate
    resolution, and can therefore be re-blocked post hoc at ANY block size.**

So the sentence that must replace 52.12's last clause is a SPLIT, not a reversal:

    The CORRELATION-channel curve needs a new `stepsize_type` and an operator decision.
    The MARGINAL-channel curve is recoverable from data already on this Mac, at every
    block size from 1 coordinate to a whole tensor, at zero compute.

This instrument computes the second one.  It does NOT compute the first and never claims to.

WHAT IS MEASURED, AND WHY IT IS THE QUANTITY A BLOCK METHOD DESTROYS
--------------------------------------------------------------------
Per coordinate i, `p_i = neg_count_i / n_records` is its persistent meta-gradient sign
preference (c59's quantity, unchanged).  A block method replaces the per-coordinate step
size with ONE step size per block, i.e. it replaces `p_i` by its block mean.  The question
"how much is destroyed at block size g" is therefore exactly a nested variance
decomposition of `p` at block size g, nested inside the tensor:

    SS_total = SS_tensor + SS_block|tensor + SS_within

    R(g) = SS_block|tensor / (SS_block|tensor + SS_within)      [noise-corrected]

R(1) = 1 and R(n_t) = 0 BY CONSTRUCTION, so R(g) alone is mechanical.  The informative
quantity is the excess over a SIZE-PRESERVING WITHIN-TENSOR REGROUP NULL, which has the
identical mechanical shape:

    E(g) = R(g) - R_null(g)          reported in pp

E(g) -> 0 at BOTH ends by construction (at g=1 both are exactly 1; at g>=n_t both are
exactly 0), so E has an interior maximum by construction and **its LOCATION, not its
existence, is the measurement.**  Gate G3 validates that readout against planted structure.

THE LADDER IS IN UNITS OF THE OUTPUT CHANNEL, AND THAT IS DELIBERATE
--------------------------------------------------------------------
Conv tensors have different row widths (9*cin = 27 .. 4608 here), so a single ABSOLUTE g
sits at a different architectural scale in each tensor and would blur any channel-scale
feature.  The primary ladder is therefore in units `u` of one output channel:

    block size in tensor t  =  max(1, round(row_width_t * u)),  contiguous within tensor

    u = 1     -> exactly one output channel = exactly c59's `nodewise` row  [GATE G1]
    u = 1/9   -> approximately one input channel's 3x3 kernel stack
    u < 1     -> sub-channel; blocks lie inside a row
    u > 1     -> supra-channel; contiguous runs of u rows

Absolute-g is reported as a secondary diagnostic (`--absolute`).

SCOPE, STATED UP FRONT (STANDING RULE 10)
-----------------------------------------
The primary curve is restricted to **conv tensors**.  On 1-D tensors (BN/bias) "row" and
"weight" are the same object and the statistic is vacuous -- CORRECTIONS 89.3 measured that
these supply a median 50.6% of the all-tensor numerator while being 0.086% of coordinates.
The all-tensor curve is printed beside it and labelled.

We measure `z`, the META-gradient.  Adam-mini argues about `G`.  **No document may write
"we refuted Adam-mini."**

PRE-REGISTERED HYPOTHESES -- WRITTEN BEFORE ANY ARM WAS SCORED
--------------------------------------------------------------
H1  CHANNEL KNEE.  argmax_u E(u) sits at u = 1 within one ladder rung (factor 2), i.e. the
    marginal channel has the same correlation length 48.18 found in the correlation channel
    ("the correlation length is approximately the channel, in 4 of 4 families").  This would
    be an independent confirmation of the campaign's sharpest result through a different
    statistic and a disjoint data path.
    REFUTATION: argmax is more than 3 rungs (8x) from u = 1 in a majority of clean arms.

H2  SUB-CHANNEL.  argmax_u E(u) sits BELOW u = 1 by more than 3 rungs -- the structure a
    block method destroys lives inside the output channel, so every partition in the
    Adam-mini / Adalayer / SGG line (all at channel-or-coarser) sits entirely above it.
    REFUTATION: E(1/8) <= E(1) within the null's MC resolution in a majority of clean arms.

H3  NULL EVERYWHERE.  E(u) is at the MC resolution at every u -- 59.3's one-point result
    (row excess 0.008-0.448 pp) generalises to the whole ladder and NO block size at ANY
    scale captures within-tensor marginal structure.  Registered as the most likely outcome.
    REFUTATION: any clean arm exceeds 5x its own MC resolution at any u.

H1, H2 and H3 are mutually exclusive at the stated thresholds.  H3 subsumes the others: if
H3 holds, the argmax of a curve that is everywhere at resolution is not interpretable and
H1/H2 are NOT scored.  That order is fixed here, before the data.

SECOND-STAGE HYPOTHESES -- REGISTERED AFTER SEEING ONE ARM'S CURVE, BEFORE ANY OTHER RUN
-----------------------------------------------------------------------------------------
This block is HONESTLY LABELLED as second-stage.  The first R18 arm scored (p5/probe_w_a3_s0,
frozen) refuted H3 (E/res = 9.5-169 at every rung) and H1 (no peak at u=1); E was flat at
~0.043 pp from u=1 down to u=1/16 and then rose to 0.152 pp at u=1/1024.  H2 therefore
survived on that arm, and the OBVIOUS candidate for a fine scale in a conv tensor is the 3x3
spatial kernel: a contiguous run of 9 coordinates is exactly one (output channel, input
channel) filter.  The channel-unit ladder BLURS that scale, because block size 9 falls at
u = 1/cin and cin ranges 3..512 across the stack.  So:

S1  ABSOLUTE-g LADDER.  On a ladder in ABSOLUTE coordinates (where g = 9 is one 3x3 kernel
    in EVERY tensor at once), E(g) has a local maximum at g = 9.
    REFUTATION: E(9) is not the maximum of {E(6), E(8), E(9), E(12), E(16)}.

S2  THE PHASE TEST -- the one that cannot be a size artifact.  Compare g = 9 blocks ALIGNED
    to the kernel boundary (offset 0) against g = 9 blocks OFFSET by 4 coordinates, so each
    block straddles two kernels.  IDENTICAL block size, near-identical block count, identical
    null: only the ALIGNMENT differs.  If the structure is the kernel, aligned >> offset.
    REGISTERED: E_aligned(9) / E_offset(9) >= 1.5.
    REFUTATION: the ratio is <= 1.2, i.e. phase does not matter and g=9 is just "small".
    S2 is the decisive one.  S1 alone can be produced by any monotone fine-scale rise that
    happens to pass through 9; S2 cannot.

S4  THE RIVAL S2 CANNOT SEPARATE, REGISTERED BEFORE S4 WAS RUN.  A contiguous run of 9 is
    one (output channel o, input channel i) FILTER -- but a filter is also entirely inside
    input channel i, and FINDINGS 59.7 measured a COLUMN (input-channel) excess of up to
    +0.788 pp, LARGER than the row's.  Pure column structure ALSO predicts aligned > offset
    at g = 9, because an offset block straddles columns i and i+1.  **S2 alone therefore
    does not license the word "filter".**
    The separator is an interaction test on the cout x cin matrix of filter means
    m[o,i] = mean of that filter's 9 coordinates:
        m = grand + a[o] + b[i] + e[o,i]
    and the question is whether `e` (the o-by-i INTERACTION) carries structure beyond the
    additive row and column effects.  Scored as F = SS_real / SS_null against the same
    within-tensor permutation null, per component.
    REGISTERED: if F_int <= 1.10, the g=9 effect is ADDITIVE row+column marginals and the
    result must be written as "row and input-channel structure", NOT as "filter structure".
    If F_int > 1.10 AND F_int is comparable to F_row / F_col, the filter is a real object.

S3  c59.7 CONSISTENCY.  FINDINGS 59.7 measured the SPATIAL DIRECTION (group all coordinates
    sharing index mod 9, i.e. 9 groups per tensor) at an excess of ~0.000-0.010 pp -- a null.
    S1/S2 are a DIFFERENT grouping (contiguous runs of 9 = one filter, indexed by the
    (cout,cin) PAIR), and the two are not in conflict: 59.7 asks whether the nine spatial
    TAPS differ on average, S1/S2 ask whether one filter's nine taps move TOGETHER.
    REGISTERED: the mod-9 direction must REPRODUCE 59.7's null here (<= 0.02 pp) on the same
    arms.  If it does not, this instrument disagrees with a published number and S1/S2 are
    NOT reported until that is resolved.

GATES (`--gate`; every one is falsifiable, none is an assertion about the answer)
--------------------------------------------------------------------------------
G1  REPRODUCTION.  At u = 1 the conv-only R_cor must equal c60's `R_conv_cor` for the same
    arm, computed by the independent c59/c60 code path, to <= 1e-9 absolute.  This is what
    makes the ladder a generalisation of a published number rather than a new statistic.
G2  NULL CALIBRATION.  On synthetic i.i.d. binomial coordinates with NO block structure,
    |E(u)| must be within MC resolution at every u.
G3  RECOVERY.  On synthetic data with structure planted at a known scale g0, argmax_u E(u)
    must land within one rung of g0.  Without G3 the argmax readout is uninterpreted.
G4  IDENTITY.  SS_total == SS_tensor + SS_block + SS_within to float precision at every u.
G5  CLAMP.  `max(raw - noise, 0)` reads R = 1 by construction when it binds (CORRECTIONS
    89.6).  Any (arm, u) cell whose clamp binds is excluded and COUNTED, never reported.

USAGE
    python3 analysis/c62_blocksize_curve.py --selftest
    python3 analysis/c62_blocksize_curve.py --gate      [--root ..]
    python3 analysis/c62_blocksize_curve.py --curve probes_fz3/probe_r18_w_s0
    python3 analysis/c62_blocksize_curve.py --report    [--root ..]
    python3 analysis/c62_blocksize_curve.py --families  [--root ..]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c59_row_premise as C59
import c60_exception_mechanism as C60


# --------------------------------------------------------------------------------------
# The ladder.  Powers of two in units of one output channel.
# --------------------------------------------------------------------------------------
U_LADDER = [2.0 ** e for e in range(-10, 7)]     # 1/1024 .. 64


def u_label(u):
    if u >= 1:
        return f"{int(round(u))}"
    return f"1/{int(round(1.0 / u))}"


# --------------------------------------------------------------------------------------
# Block index at a given u
# --------------------------------------------------------------------------------------

def build_block_index(shapes, u):
    """Coordinate -> (tensor_id, block_id) with blocks CONTIGUOUS inside each tensor.

    Block size in tensor t is max(1, round(row_width_t * u)) where row_width_t is the
    number of coordinates in one output-channel row (prod(shape[1:])).  Block ids are
    global (offset per tensor) so they never collide across tensors.

    Returns (tid, bid, n_blocks_total, per_tensor_block_size).
    """
    tid_parts, bid_parts, bsz = [], [], []
    off = 0
    for t, (_nm, sh) in enumerate(shapes):
        n = int(np.prod(sh))
        d0 = int(sh[0])
        rest = n // d0                       # row width = coordinates per output channel
        b = max(1, int(round(rest * u)))
        b = min(b, n)                        # a block never exceeds the tensor
        idx = np.arange(n, dtype=np.int64) // b
        nb = int(idx[-1]) + 1
        tid_parts.append(np.full(n, t, dtype=np.int32))
        bid_parts.append(off + idx)
        bsz.append(b)
        off += nb
    return (np.concatenate(tid_parts), np.concatenate(bid_parts), off, bsz)


# --------------------------------------------------------------------------------------
# ABSOLUTE-g machinery (S1/S2).  Same block size in every tensor, so g = 9 is exactly one
# 3x3 kernel everywhere at once.  `offset` shifts the block boundaries WITHIN each tensor,
# which is the whole point of S2: identical size, identical count, different phase.
# --------------------------------------------------------------------------------------
G_LADDER = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 27, 32, 36, 64, 72, 128, 144, 288, 576]


def build_abs_index(shapes, g, offset=0):
    """Coordinate -> (tensor_id, block_id) with ABSOLUTE contiguous blocks of size g.

    `offset` in [0, g) shifts where the first boundary falls inside each tensor; the
    leading partial block is a block of its own.  Restricted to 3x3 conv tensors by the
    caller (offset only has a kernel meaning there).
    """
    tid_parts, bid_parts = [], []
    off = 0
    for t, (_nm, sh) in enumerate(shapes):
        n = int(np.prod(sh))
        gg = min(g, n)
        idx = (np.arange(n, dtype=np.int64) + (gg - offset % gg)) // gg
        idx = idx - idx[0]
        nb = int(idx[-1]) + 1
        tid_parts.append(np.full(n, t, dtype=np.int32))
        bid_parts.append(off + idx)
        off += nb
    return np.concatenate(tid_parts), np.concatenate(bid_parts), off


def mod9_index(shapes):
    """Coordinate -> (tensor_id, block_id) grouping by SPATIAL POSITION (index mod 9).

    This is FINDINGS 59.7's `spatial` direction: 9 groups per tensor, each holding every
    coordinate at one of the nine 3x3 taps.  S3 checks it reproduces 59.7's null.
    """
    tid_parts, bid_parts = [], []
    off = 0
    for t, (_nm, sh) in enumerate(shapes):
        n = int(np.prod(sh))
        tid_parts.append(np.full(n, t, dtype=np.int32))
        bid_parts.append(off + (np.arange(n, dtype=np.int64) % 9))
        off += 9
    return np.concatenate(tid_parts), np.concatenate(bid_parts), off


def _E_from_index(p, shapes, n_rec, tid, bid, nb, nulls):
    real = R_of(ss_at_u(p, shapes, n_rec, None, tid, bid, nb))
    nv, cl = [], bool(real["clamped_row"] or real["clamped_within"])
    for q in nulls:
        rn = R_of(ss_at_u(q, shapes, n_rec, None, tid, bid, nb))
        nv.append(rn["R_cor"])
        cl = cl or bool(rn["clamped_row"] or rn["clamped_within"])
    nm = float(np.mean(nv))
    sd = float(np.std(nv, ddof=1) / math.sqrt(len(nv))) if len(nv) >= 2 else float("nan")
    return dict(R=real["R_cor"], R_null=nm, E_pp=100.0 * (real["R_cor"] - nm),
                res_pp=100.0 * sd, n_blocks=nb, clamped=cl)


def _rc_ss(m):
    """Additive row/column/interaction sums of squares of a 2-D matrix of filter means."""
    g = m.mean()
    a = m.mean(axis=1) - g            # row (output channel) effect
    b = m.mean(axis=0) - g            # column (input channel) effect
    e = m - g - a[:, None] - b[None, :]
    cout, cin = m.shape
    return dict(row=float(cin * np.sum(a ** 2)),
                col=float(cout * np.sum(b ** 2)),
                inter=float(np.sum(e ** 2)))


def filter_interaction(p, shapes, null_seeds=(101, 202, 303)):
    """S4.  Row / column / interaction structure in the cout x cin filter-mean matrix,
    scored as F = SS_real / SS_null against the SAME within-tensor permutation null.

    Ratios are used rather than noise-corrected excesses on purpose: the null has the
    identical shape and the identical sampling noise, so binomial noise divides out and no
    clamp can bind.  Aggregated over 3x3 conv tensors by summing SS (they are additive).
    """
    def totals(vec):
        acc = dict(row=0.0, col=0.0, inter=0.0)
        off = 0
        for _nm, sh in shapes:
            n = int(np.prod(sh))
            if is3x3(sh):
                m = vec[off:off + n].reshape(int(sh[0]), int(sh[1]), 9).mean(axis=2)
                if m.shape[0] > 1 and m.shape[1] > 1:
                    s = _rc_ss(m)
                    for k in acc:
                        acc[k] += s[k]
            off += n
        return acc

    real = totals(p)
    nulls = [totals(within_tensor_permute(p, shapes, s)) for s in null_seeds]
    out = {}
    for k in ("row", "col", "inter"):
        nv = np.array([x[k] for x in nulls])
        nm = float(nv.mean())
        sd = float(nv.std(ddof=1) / math.sqrt(len(nv))) if len(nv) >= 2 else float("nan")
        out[k] = dict(real=real[k], null=nm, F=(real[k] / nm if nm > 0 else float("inf")),
                      F_res=(sd / nm if nm > 0 else float("nan")))
    return out


def spatial_suite(p, shapes, n_rec, null_seeds=(101, 202, 303)):
    """S1 (absolute-g ladder), S2 (phase test at g=9), S3 (mod-9 direction)."""
    nulls = [within_tensor_permute(p, shapes, s) for s in null_seeds]
    lad = []
    for g in G_LADDER:
        tid, bid, nb = build_abs_index(shapes, g, 0)
        r = _E_from_index(p, shapes, n_rec, tid, bid, nb, nulls)
        r["g"] = g
        lad.append(r)
    phase = {}
    for offs in (0, 1, 2, 3, 4):
        tid, bid, nb = build_abs_index(shapes, 9, offs)
        phase[offs] = _E_from_index(p, shapes, n_rec, tid, bid, nb, nulls)
    tid, bid, nb = mod9_index(shapes)
    mod9 = _E_from_index(p, shapes, n_rec, tid, bid, nb, nulls)
    s4 = filter_interaction(p, shapes, null_seeds)
    return dict(ladder=lad, phase=phase, mod9=mod9, s4=s4)


def within_tensor_permute(p, shapes, seed):
    """Size-preserving null: permute coordinates WITHIN each tensor.

    One permutation serves the whole ladder -- exchangeability within the tensor is what
    the null asserts, and it is a property of the permuted vector, not of u.  Reusing it
    across u correlates the null estimates across rungs, which is desirable (it makes the
    null curve smooth) and is stated rather than hidden.
    """
    r = np.random.default_rng(seed)
    q = p.copy()
    off = 0
    for _nm, sh in shapes:
        n = int(np.prod(sh))
        if n > 1:
            q[off:off + n] = q[off:off + n][r.permutation(n)]
        off += n
    return q


# --------------------------------------------------------------------------------------
# The decomposition at one u.  Reuses c59's validated nested_ss / noise_split verbatim --
# both are generic over (tid, group_id) and know nothing about rows.
# --------------------------------------------------------------------------------------

def ss_at_u(p, shapes, n_rec, u, tid=None, bid=None, n_blocks=None):
    if tid is None:
        tid, bid, n_blocks, _ = build_block_index(shapes, u)
    ss = C59.nested_ss(p, tid, bid, len(shapes), n_blocks)
    nb, nr, _tot = C59.noise_split(p, bid, None, n_blocks, n_rec)
    ident = abs(ss["ss_total"] - (ss["ss_tensor"] + ss["ss_row"] + ss["ss_within"])) \
        / max(ss["ss_total"], 1e-30)
    return dict(raw_block=ss["ss_row"], raw_within=ss["ss_within"],
                noise_block=nr, noise_within=nb,
                ss_total=ss["ss_total"], ss_tensor=ss["ss_tensor"],
                n_blocks=n_blocks, identity_err=ident)


def R_of(d):
    """Noise-corrected R with c60's clamp flags."""
    return C60.R_from(d["raw_block"], d["raw_within"], d["noise_block"], d["noise_within"])


def curve(p, shapes, n_rec, ladder=U_LADDER, null_seeds=(101, 202, 303)):
    """E(u) over the ladder, with a size-preserving within-tensor null.

    Returns a list of per-u dicts.  Everything is in the caller's tensor subset -- pass a
    conv-only (p, shapes) pair for the primary curve.
    """
    nulls = [within_tensor_permute(p, shapes, s) for s in null_seeds]
    out = []
    for u in ladder:
        tid, bid, nb, bsz = build_block_index(shapes, u)
        real = ss_at_u(p, shapes, n_rec, u, tid, bid, nb)
        Rr = R_of(real)
        Rn = []
        for q in nulls:
            dn = ss_at_u(q, shapes, n_rec, u, tid, bid, nb)
            Rn.append(R_of(dn))
        nv = [x["R_cor"] for x in Rn]
        null_mean = float(np.mean(nv))
        # MC resolution of the null mean (sd of the mean over seeds); with <3 seeds this is
        # not defined and the cell is marked unresolved rather than given a fake bar.
        null_sd = float(np.std(nv, ddof=1) / math.sqrt(len(nv))) if len(nv) >= 2 else float("nan")
        clamped = bool(Rr["clamped_row"] or Rr["clamped_within"]
                       or any(x["clamped_row"] or x["clamped_within"] for x in Rn))
        out.append(dict(
            u=u, label=u_label(u), n_blocks=nb,
            bsz_min=int(min(bsz)), bsz_max=int(max(bsz)),
            R=Rr["R_cor"], R_raw=Rr["R_raw"], R_null=null_mean, null_sd=null_sd,
            E_pp=100.0 * (Rr["R_cor"] - null_mean),
            res_pp=100.0 * null_sd,
            clamped=clamped, identity_err=real["identity_err"],
        ))
    return out


def peak(rows, require_unclamped=True):
    """(row, index) of max E among usable rungs; None if none usable."""
    usable = [(i, r) for i, r in enumerate(rows)
              if (not require_unclamped or not r["clamped"]) and np.isfinite(r["E_pp"])]
    if not usable:
        return None, None
    i, r = max(usable, key=lambda ir: ir[1]["E_pp"])
    return r, i


# --------------------------------------------------------------------------------------
# Arm loading -- conv-only subset (STANDING RULE 10) plus the all-tensor comparator
# --------------------------------------------------------------------------------------

def load_conv(d):
    """-> (p_conv, shapes_conv, p_all, shapes_all, n_rec, fam, meta)."""
    p, n_rec, meta = C59.load_arm(d)
    fam, shapes = C60.load_shapes_for(meta)
    if shapes is None:
        return None
    keep = ("conv",)
    pc = C60.coord_subset(p, shapes, keep)
    sc = C60.tensor_subset(shapes, keep)
    return pc, sc, p, shapes, n_rec, fam, meta


def is3x3(sh):
    return len(sh) == 4 and sh[2] == 3 and sh[3] == 3


def load_3x3(d):
    """3x3-conv-only subset.  1x1 shortcut convs have no kernel and are excluded from
    S1/S2/S3 by construction, not by choice of result."""
    p, n_rec, meta = C59.load_arm(d)
    fam, shapes = C60.load_shapes_for(meta)
    if shapes is None:
        return None
    sel, s3 = [], []
    off = 0
    for nm, sh in shapes:
        n = int(np.prod(sh))
        if is3x3(sh):
            sel.append(np.arange(off, off + n))
            s3.append((nm, sh))
        off += n
    if not s3:
        return None
    return p[np.concatenate(sel)], s3, n_rec, fam, meta


# --------------------------------------------------------------------------------------
# Selftests
# --------------------------------------------------------------------------------------

def _synth_shapes():
    """Conv-only miniature stack: row widths 27 and 36, so u=1 is a real channel."""
    return [("c1", (8, 3, 3, 3)), ("c2", (6, 4, 3, 3))]


def _binom(rng, P, n_rec):
    return rng.binomial(n_rec, P) / float(n_rec)


def selftest():
    ok = 0

    def chk(cond, name):
        nonlocal ok
        if not cond:
            print(f"  FAIL {name}")
            raise SystemExit(1)
        ok += 1

    # ---- block index -----------------------------------------------------------------
    sh = _synth_shapes()
    tid, bid, nb, bsz = build_block_index(sh, 1.0)
    chk(bsz == [27, 36], "u=1 block size equals row width")
    chk(nb == 8 + 6, "u=1 block count equals row count")
    # u=1 must reproduce c59's row index EXACTLY
    tid59, rid59, _rw, _sp, nrows59 = C59.build_index(sh)
    chk(nb == nrows59, "u=1 n_blocks == c59 n_rows")
    chk(np.array_equal(bid, rid59), "u=1 block ids == c59 row ids")
    chk(np.array_equal(tid, tid59), "tensor ids match c59")

    tid2, bid2, nb2, bsz2 = build_block_index(sh, 0.5)
    chk(bsz2 == [14, 18], "u=1/2 rounds row width down the half")
    chk(nb2 == math.ceil(216 / 14) + math.ceil(216 / 18), "u=1/2 block count")

    tid3, bid3, nb3, bsz3 = build_block_index(sh, 2.0)
    chk(bsz3 == [54, 72], "u=2 block size is two rows")
    chk(nb3 == 4 + 3, "u=2 block count halves")

    tid4, bid4, nb4, bsz4 = build_block_index(sh, 1e-9)
    chk(bsz4 == [1, 1], "tiny u floors at one coordinate")
    tidB, bidB, nbB, bszB = build_block_index(sh, 1e9)
    chk(bszB == [216, 216], "huge u caps at the tensor")
    chk(nbB == 2, "huge u gives one block per tensor")

    # ---- degenerate ends -------------------------------------------------------------
    rng = np.random.default_rng(7)
    n = sum(int(np.prod(s)) for _nm, s in sh)
    p = _binom(rng, np.full(n, 0.5), 200)
    d1 = ss_at_u(p, sh, 200, 1e-9)
    chk(d1["raw_within"] == 0.0, "g=1: SS_within is exactly 0")
    chk(R_of(d1)["R_raw"] == 1.0, "g=1: R_raw is exactly 1")
    dB = ss_at_u(p, sh, 200, 1e9)
    chk(dB["raw_block"] == 0.0, "g=tensor: SS_block is exactly 0")
    chk(R_of(dB)["R_raw"] == 0.0, "g=tensor: R_raw is exactly 0")

    # ---- G4 identity at every rung ---------------------------------------------------
    worst = 0.0
    for u in U_LADDER:
        worst = max(worst, ss_at_u(p, sh, 200, u)["identity_err"])
    chk(worst < 1e-10, f"G4 identity holds at every rung (worst {worst:.2e})")

    # ---- G1 reproduction against the c59/c60 code path -------------------------------
    def same(a, b, tol=1e-12):
        if np.isnan(a) and np.isnan(b):
            return True
        return abs(a - b) < tol

    # (i) on pure binomial noise BOTH clamps bind and both paths must read nan.  A naive
    #     abs(a-b) comparison silently FAILS on nan==nan, which is how this check was
    #     written first -- the harness caught it, so the nan branch is explicit.
    d_u1 = ss_at_u(p, sh, 200, 1.0)
    R_u1, R59 = R_of(d_u1), C60.class_R(C60.class_raw_ss(p, sh, 200)["conv"])
    chk(np.isnan(R_u1["R_cor"]) and np.isnan(R59["R_cor"]),
        "G1a pure-noise synthetic clamps on BOTH paths (nan)")
    chk(same(R_u1["R_raw"], R59["R_raw"]), "G1a u=1 R_raw == c60 conv R_raw")
    # (ii) the check that matters: structure above the binomial floor, so nothing clamps
    rgs = np.random.default_rng(1234)
    lev = 0.5 + 0.10 * rgs.standard_normal(14)
    Ps = np.clip(np.repeat(lev, [27] * 8 + [36] * 6)
                 + 0.05 * rgs.standard_normal(n), 0.02, 0.98)
    ps = rgs.binomial(600, Ps) / 600.0
    d_s = ss_at_u(ps, sh, 600, 1.0)
    R_s, R59s = R_of(d_s), C60.class_R(C60.class_raw_ss(ps, sh, 600)["conv"])
    chk(not (R_s["clamped_row"] or R_s["clamped_within"]), "G1b structured case does not clamp")
    chk(same(R_s["R_cor"], R59s["R_cor"]), "G1b u=1 R_cor == c60 conv R_cor")
    chk(same(R_s["R_raw"], R59s["R_raw"]), "G1b u=1 R_raw == c60 conv R_raw")

    # ---- permutation preserves the multiset within each tensor -----------------------
    q = within_tensor_permute(p, sh, 5)
    off = 0
    for _nm, s in sh:
        m = int(np.prod(s))
        chk(np.array_equal(np.sort(p[off:off + m]), np.sort(q[off:off + m])),
            f"null permutes within tensor {_nm}")
        off += m
    chk(not np.array_equal(p, q), "null actually moves coordinates")

    # ---- G2 null calibration: i.i.d. binomial, no block structure ---------------------
    rng = np.random.default_rng(11)
    P = np.full(n, 0.5)
    p2 = _binom(rng, P, 400)
    rows = curve(p2, sh, 400, null_seeds=(1, 2, 3, 4, 5))
    bad = [r for r in rows if (not r["clamped"]) and np.isfinite(r["res_pp"])
           and r["res_pp"] > 0 and abs(r["E_pp"]) > 8.0 * r["res_pp"]]
    chk(len(bad) == 0, f"G2 iid-binomial null flat (offenders {[b['label'] for b in bad]})")

    # heterogeneous-but-unstructured control: per-coordinate P varies, but assigned at
    # RANDOM within the tensor, so there is marginal bias and NO block structure.
    rng = np.random.default_rng(13)
    Ph = np.clip(0.5 + 0.05 * rng.standard_normal(n), 0.02, 0.98)
    p3 = _binom(rng, Ph, 400)
    rows3 = curve(p3, sh, 400, null_seeds=(1, 2, 3, 4, 5))
    bad3 = [r for r in rows3 if (not r["clamped"]) and r["res_pp"] > 0
            and abs(r["E_pp"]) > 8.0 * r["res_pp"]]
    chk(len(bad3) == 0, f"G2b heterogeneous-unstructured null flat ({[b['label'] for b in bad3]})")

    # ---- G3 recovery: structure planted at a KNOWN scale, IN LADDER UNITS -------------
    # Two things this test had to be rewritten for, both caught by the harness:
    #  (i) planting a fixed ABSOLUTE block size is not a fixed `u` when tensors have
    #      different row widths, so the "truth" rung was ill-defined.  Plant in u.
    #  (ii) a PERFECTLY constant block makes SS_within pure sampling noise, so the clamp
    #      binds at exactly the planted rung and the excluded cell IS the true peak
    #      (measured: u0=1 planted, true rung E=97.9 pp, clamped, peak misread as 1/2).
    #      Real coordinates carry idiosyncratic variation; the plant now carries jitter.
    #      **A perfectly homogeneous block is the one case this readout cannot locate,
    #      and that is a property of the clamp, recorded here rather than discovered later.**
    shL = [("cL1", (64, 16, 3, 3)), ("cL2", (32, 32, 3, 3))]     # rows 144 / 288
    nL = sum(int(np.prod(s)) for _nm, s in shL)

    def planted_u(u0, amp, jit, seed, n_rec=600):
        r = np.random.default_rng(seed)
        parts = []
        for _nm, s in shL:
            m = int(np.prod(s))
            rest = m // int(s[0])
            b = max(1, int(round(rest * u0)))
            nblk = int(math.ceil(m / b))
            parts.append(np.repeat(0.5 + amp * r.standard_normal(nblk), b)[:m])
        P = np.concatenate(parts)
        P = np.clip(P + jit * r.standard_normal(P.size), 0.02, 0.98)
        return _binom(r, P, n_rec)

    for u0 in (1.0 / 16, 0.25, 1.0, 4.0):
        pp = planted_u(u0, 0.08, 0.05, 4000 + int(u0 * 1000))
        rows_g = curve(pp, shL, 600, null_seeds=(1, 2, 3))
        pk, pi = peak(rows_g)
        truth = U_LADDER.index(u0)
        chk(pk is not None, f"G3 u0={u_label(u0)}: a peak exists")
        chk(abs(pi - truth) <= 1,
            f"G3 u0={u_label(u0)}: peak at {pk['label']} within one rung of the plant")
        chk(pk["E_pp"] > 5.0 * max(pk["res_pp"], 1e-12),
            f"G3 u0={u_label(u0)}: planted structure detected above resolution")

    # planted structure must be INVISIBLE once permuted within the tensor
    pperm = within_tensor_permute(planted_u(1.0, 0.08, 0.05, 99), shL, 4242)
    rows_p = curve(pperm, shL, 600, null_seeds=(1, 2, 3))
    badp = [r for r in rows_p if (not r["clamped"]) and r["res_pp"] > 0
            and r["E_pp"] > 8.0 * r["res_pp"]]
    chk(len(badp) == 0, f"G3b permuting planted structure erases it ({[b['label'] for b in badp]})")

    # ---- G5 clamp flag fires when the correction over-subtracts ----------------------
    #    a within-block spread far below the binomial floor must trip the within clamp
    r = np.random.default_rng(3)
    Pc = np.repeat(0.5 + 0.2 * r.standard_normal(14), [27] * 8 + [36] * 6)
    Pc = np.clip(Pc, 0.05, 0.95)
    pc = Pc.copy()                      # NO sampling noise at all -> raw_within == 0
    dcl = ss_at_u(pc, sh, 50, 1.0)
    Rcl = R_of(dcl)
    chk(Rcl["clamped_within"], "G5 zero within-spread trips the within clamp")
    chk(Rcl["R_cor"] == 1.0, "G5 a bound within-clamp reads R_cor == 1 by construction")

    # ---- S1/S2/S3 machinery ----------------------------------------------------------
    tA, bA, nA = build_abs_index(sh, 9, 0)
    chk(nA == 432 // 9, "abs g=9 block count on a 432-coordinate conv-only stack")
    chk(np.array_equal(bA[:9], np.zeros(9, dtype=np.int64)), "abs g=9 offset 0 first block")
    tO, bO, nO = build_abs_index(sh, 9, 4)
    chk(int(bO[:4].max()) == 0 and int(bO[4]) == 1, "abs g=9 offset 4 breaks after 4 coords")
    chk(nO == nA + len(sh), "offset adds exactly one leading partial block PER TENSOR")
    for offs in range(9):
        _t, _b, _n = build_abs_index(sh, 9, offs)
        szs = np.bincount(_b)
        chk(szs.max() <= 9, f"offset {offs}: no block exceeds g")
        chk(int(szs.sum()) == 432, f"offset {offs}: blocks partition the stack")
    tM, bM, nM = mod9_index(sh)
    chk(nM == 9 * len(sh), "mod9 gives 9 groups per tensor")
    chk(np.array_equal(bM[:9], np.arange(9)), "mod9 assigns consecutive taps to taps")
    chk(np.all(np.bincount(bM) == np.array([216 // 9] * 9 + [216 // 9] * 9)),
        "mod9 groups are equal-sized here")

    # S2 recovery: plant KERNEL-scale structure (each run of 9 shares a level) and check
    # the aligned/offset ratio fires; then plant NON-kernel structure of the same size and
    # check it does NOT.  Both directions, so the test can fail either way.
    def plant_runs(shapes_, run, amp, jit, seed, n_rec=800, phase=0):
        r = np.random.default_rng(seed)
        parts = []
        for _nm, s in shapes_:
            m = int(np.prod(s))
            nblk = int(math.ceil((m + phase) / run))
            parts.append(np.repeat(0.5 + amp * r.standard_normal(nblk), run)[phase:phase + m])
        P = np.clip(np.concatenate(parts) + jit * r.standard_normal(sum(
            int(np.prod(s)) for _nm, s in shapes_)), 0.02, 0.98)
        return _binom(r, P, n_rec)

    pk9 = plant_runs(sh, 9, 0.10, 0.04, 777)
    nl = [within_tensor_permute(pk9, sh, s) for s in (1, 2, 3)]
    tA, bA, nA = build_abs_index(sh, 9, 0)
    tO, bO, nO = build_abs_index(sh, 9, 4)
    Ea = _E_from_index(pk9, sh, 800, tA, bA, nA, nl)
    Eo = _E_from_index(pk9, sh, 800, tO, bO, nO, nl)
    chk(Ea["E_pp"] / max(Eo["E_pp"], 1e-9) >= 1.5,
        f"S2 recovery: kernel-aligned/offset = {Ea['E_pp'] / max(Eo['E_pp'], 1e-9):.2f} >= 1.5")
    # negative control: structure at run length 8 must NOT favour the g=9 aligned cut
    pk8 = plant_runs(sh, 8, 0.10, 0.04, 778)
    nl8 = [within_tensor_permute(pk8, sh, s) for s in (1, 2, 3)]
    Ea8 = _E_from_index(pk8, sh, 800, tA, bA, nA, nl8)
    Eo8 = _E_from_index(pk8, sh, 800, tO, bO, nO, nl8)
    chk(Ea8["E_pp"] / max(Eo8["E_pp"], 1e-9) < 1.5,
        f"S2 negative control: run-8 structure gives {Ea8['E_pp'] / max(Eo8['E_pp'], 1e-9):.2f} < 1.5")

    # S3 machinery: planted MOD-9 TAP structure must show up in the mod9 direction and a
    # kernel plant must not, so the two groupings are demonstrably different objects.
    r = np.random.default_rng(31)
    tap = 0.10 * r.standard_normal(9)
    n_sh = sum(int(np.prod(s)) for _nm, s in sh)
    Pt = np.clip(0.5 + tap[np.arange(n_sh) % 9] + 0.04 * r.standard_normal(n_sh), 0.02, 0.98)
    ptap = _binom(r, Pt, 800)
    nlt = [within_tensor_permute(ptap, sh, s) for s in (1, 2, 3)]
    Em = _E_from_index(ptap, sh, 800, tM, bM, nM, nlt)
    chk(Em["E_pp"] > 5.0 * max(Em["res_pp"], 1e-12), "S3 mod-9 grouping detects tap structure")
    nl9 = [within_tensor_permute(pk9, sh, s) for s in (1, 2, 3)]
    Em2 = _E_from_index(pk9, sh, 800, tM, bM, nM, nl9)
    chk(Em2["E_pp"] < Em["E_pp"] / 5.0,
        "S3 mod-9 grouping is nearly blind to kernel structure (the groupings differ)")

    # ---- S4: the row / column / interaction separator ---------------------------------
    # _rc_ss algebra first: an exactly additive matrix must have ZERO interaction.
    aa = np.arange(5.0)[:, None]
    bb = np.arange(7.0)[None, :]
    chk(abs(_rc_ss(aa + bb)["inter"]) < 1e-20, "S4 additive matrix has zero interaction")
    chk(_rc_ss(aa + bb)["row"] > 0 and _rc_ss(aa + bb)["col"] > 0,
        "S4 additive matrix has nonzero row and column")
    rr = np.random.default_rng(5)
    mm = rr.standard_normal((5, 7))
    s = _rc_ss(mm)
    tot = float(np.sum((mm - mm.mean()) ** 2))
    chk(abs(s["row"] + s["col"] + s["inter"] - tot) < 1e-9,
        "S4 row+col+interaction is an exact decomposition")

    shS = [("cS", (12, 10, 3, 3))]
    nS = 12 * 10 * 9

    def plant_matrix(kind, amp, jit, seed, n_rec=800):
        r = np.random.default_rng(seed)
        if kind == "row":
            M = np.repeat(amp * r.standard_normal((12, 1)), 10, axis=1)
        elif kind == "col":
            M = np.repeat(amp * r.standard_normal((1, 10)), 12, axis=0)
        else:
            M = amp * r.standard_normal((12, 10))
        P = np.clip(0.5 + np.repeat(M.reshape(-1), 9) + jit * r.standard_normal(nS),
                    0.02, 0.98)
        return _binom(r, P, n_rec)

    F_row = filter_interaction(plant_matrix("row", 0.10, 0.03, 61), shS, (1, 2, 3))
    chk(F_row["row"]["F"] > 3.0, f"S4 row plant lifts F_row ({F_row['row']['F']:.2f})")
    chk(F_row["inter"]["F"] < 1.10,
        f"S4 row plant leaves F_int flat ({F_row['inter']['F']:.3f} < 1.10)")
    F_col = filter_interaction(plant_matrix("col", 0.10, 0.03, 62), shS, (1, 2, 3))
    chk(F_col["col"]["F"] > 3.0, f"S4 col plant lifts F_col ({F_col['col']['F']:.2f})")
    chk(F_col["inter"]["F"] < 1.10,
        f"S4 col plant leaves F_int flat ({F_col['inter']['F']:.3f} < 1.10)")
    F_int = filter_interaction(plant_matrix("inter", 0.10, 0.03, 63), shS, (1, 2, 3))
    chk(F_int["inter"]["F"] > 3.0, f"S4 filter plant lifts F_int ({F_int['inter']['F']:.2f})")

    # ---- ladder monotonicity of the MECHANICAL part ----------------------------------
    # The mechanically monotone quantity is R_RAW.  R_cor can be nan wherever the clamp
    # binds (pure-noise data clamps at every coarse rung), so asserting monotonicity on
    # the corrected null was wrong -- the assertion, not the code.
    qm = within_tensor_permute(p2, sh, 7)
    nullseq = [R_of(ss_at_u(qm, sh, 400, u))["R_raw"] for u in U_LADDER]
    chk(all(nullseq[i] >= nullseq[i + 1] - 1e-9 for i in range(len(nullseq) - 1)),
        "null R decreases monotonically with block size")
    chk(abs(nullseq[0] - 1.0) < 1e-9, "null R starts at 1")
    chk(abs(nullseq[-1]) < 1e-9, "null R ends at 0")

    # ---- u_label round-trip ----------------------------------------------------------
    chk(u_label(1.0) == "1" and u_label(0.25) == "1/4" and u_label(64) == "64",
        "u_label formats")

    print(f"  selftest: {ok}/{ok} PASS")
    return ok


# --------------------------------------------------------------------------------------
# Real-data gates
# --------------------------------------------------------------------------------------

def gate(root, only=None, limit=None):
    print("=" * 94)
    print("GATES ON REAL DATA")
    print("=" * 94)
    ds = sorted(C60.arms(root, "weightwise"))
    if only:
        ds = [d for d in ds if only in C60.arm_label(d)]
    if limit:
        ds = ds[:limit]
    if not ds:
        print("  no weightwise arms found under", root)
        return
    n_g1 = n_g4 = 0
    worst_g1 = worst_g4 = 0.0
    for d in ds:
        L = load_conv(d)
        if L is None:
            continue
        pc, sc, pall, shall, n_rec, fam, meta = L
        # G1: u=1 conv-only == c60's R_conv_cor, via the independent c59/c60 path
        d_u1 = ss_at_u(pc, sc, n_rec, 1.0)
        mine = R_of(d_u1)["R_cor"]
        theirs = C60.class_R(C60.class_raw_ss(pall, shall, n_rec)["conv"])["R_cor"]
        e1 = abs(mine - theirs)
        worst_g1 = max(worst_g1, e1)
        n_g1 += 1
        # G4: identity at every rung, conv-only
        for u in U_LADDER:
            worst_g4 = max(worst_g4, ss_at_u(pc, sc, n_rec, u)["identity_err"])
            n_g4 += 1
    print(f"G1 reproduction  : {n_g1} arms, worst |c62(u=1) - c60 R_conv_cor| = {worst_g1:.3e}"
          f"   {'PASS' if worst_g1 < 1e-9 else 'FAIL'}")
    print(f"G4 identity      : {n_g4} (arm,rung) cells, worst relative error = {worst_g4:.3e}"
          f"   {'PASS' if worst_g4 < 1e-9 else 'FAIL'}")
    print()
    print("G2/G3/G5 are synthetic and run under --selftest (they need ground truth).")


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def fmt_curve(rows, title):
    print(title)
    print(f"  {'u':>7} {'blk(min-max)':>16} {'n_blk':>9} {'R':>9} {'R_null':>9} "
          f"{'E (pp)':>10} {'res (pp)':>9} {'E/res':>7}  flag")
    for r in rows:
        ratio = (r["E_pp"] / r["res_pp"]) if r["res_pp"] > 0 else float("nan")
        flag = "CLAMP" if r["clamped"] else ""
        print(f"  {r['label']:>7} {str(r['bsz_min'])+'-'+str(r['bsz_max']):>16} "
              f"{r['n_blocks']:>9d} {r['R']:>9.5f} {r['R_null']:>9.5f} "
              f"{r['E_pp']:>10.4f} {r['res_pp']:>9.4f} {ratio:>7.2f}  {flag}")


def one_arm(root, rel, ladder=U_LADDER, seeds=(101, 202, 303)):
    d = os.path.join(root, rel) if not os.path.isabs(rel) else rel
    L = load_conv(d)
    if L is None:
        print("unrecognised arm", d)
        return
    pc, sc, pall, shall, n_rec, fam, meta = L
    print(f"arm {rel}   family={fam}  n_rec={n_rec}  "
          f"conv coords={pc.size:,} in {len(sc)} tensors "
          f"(all: {pall.size:,} in {len(shall)})")
    rows = curve(pc, sc, n_rec, ladder, seeds)
    fmt_curve(rows, "\nCONV-ONLY (primary, STANDING RULE 10)")
    pk, pi = peak(rows)
    if pk:
        print(f"\n  peak: u={pk['label']}  E={pk['E_pp']:.4f} pp  "
              f"res={pk['res_pp']:.4f} pp  E/res={pk['E_pp']/max(pk['res_pp'],1e-12):.1f}")
    rows_a = curve(pall, shall, n_rec, ladder, seeds)
    fmt_curve(rows_a, "\nALL TENSORS (secondary, labelled per STANDING RULE 10)")
    return rows, rows_a


# The clean/exception split is DATA copied from FINDINGS 59.4 via c60.EXC_W, not a
# threshold recomputed here.
def is_exception(label):
    return label in C60.EXC_W


def report(root, ladder=U_LADDER, seeds=(101, 202, 303), only=None):
    ds = sorted(C60.arms(root, "weightwise"))
    out = []
    for d in ds:
        lab = C60.arm_label(d)
        if only and only not in lab:
            continue
        L = load_conv(d)
        if L is None:
            continue
        pc, sc, pall, shall, n_rec, fam, meta = L
        rows = curve(pc, sc, n_rec, ladder, seeds)
        pk, pi = peak(rows)
        out.append(dict(arm=lab, fam=fam, n_rec=n_rec, exc=is_exception(lab),
                        rows=rows, peak=pk, pi=pi))
        e1 = next(r for r in rows if r["u"] == 1.0)
        e8 = next(r for r in rows if abs(r["u"] - 0.125) < 1e-12)
        mx = max((r["E_pp"] / r["res_pp"]) for r in rows
                 if not r["clamped"] and r["res_pp"] > 0)
        print(f"{lab:<34} {fam:>5} {'EXC' if is_exception(lab) else 'cln':>4} "
              f"E(1)={e1['E_pp']:>9.4f}+-{e1['res_pp']:.4f}  "
              f"E(1/8)={e8['E_pp']:>9.4f}+-{e8['res_pp']:.4f}  "
              f"peak@{(pk['label'] if pk else '-'):>6} {(pk['E_pp'] if pk else float('nan')):>9.4f}pp"
              f"  maxE/res={mx:>6.2f}")
    return out


SPATIAL_SET = [
    # matched frozen / free pairs across the four families -- the operator's stated axis
    ("probes_fz3/probe_r10_w_s0", "r10  frozen"), ("probes_fz3/probe_r10_w_s1", "r10  frozen"),
    ("probes_fz3/probe_r34_w_s0", "r34  frozen"), ("probes_fz3/probe_r34_w_s1", "r34  frozen"),
    ("probes_fz3/probe_c100_w_s0", "c100 frozen"), ("probes_fz3/probe_c100_w_s1", "c100 frozen"),
    ("probes_p5/probe_w_a3_s0", "r18  frozen"), ("probes_p5/probe_w_a3_s1", "r18  frozen"),
    ("probes_ff5/probe_r10_w_s0", "r10  free"), ("probes_ff5/probe_r10_w_s1", "r10  free"),
    ("probes_ff5/probe_r34_w_s0", "r34  free"), ("probes_ff5/probe_r34_w_s1", "r34  free"),
    ("probes_ff5/probe_c100_w_s0", "c100 free"), ("probes_ff5/probe_c100_w_s1", "c100 free"),
    ("probes_cl5/probe_w_cU_s0", "r18  free"), ("probes_cl5/probe_w_cU_s1", "r18  free"),
    ("probes_cl5/probe_w_cU_s2", "r18  free"),
    # two ms=1e-2 EXCEPTION arms (FINDINGS 59.4), reported SEPARATELY and never pooled
    ("probes_ml5/probe_w_m2_s0", "r18  EXC ms1e-2"),
    ("probes_bo6/probe_w_adw_s0", "r18  EXC adamw"),
]


def spatial_report(root, seeds=(101, 202, 303), which=None):
    print("=" * 108)
    print("S1 / S2 / S3 -- ABSOLUTE-g LADDER, THE PHASE TEST AT g=9, AND THE mod-9 DIRECTION")
    print("3x3 CONV TENSORS ONLY (1x1 shortcut convs have no kernel and are excluded by construction)")
    print("=" * 108)
    hdr = (f"{'arm':<32}{'tag':<16}{'E(8)':>9}{'E(9)':>9}{'E(12)':>9}{'E(16)':>9}"
           f"{'S1':>5}{'E9_al':>9}{'E9_off4':>9}{'S2 ratio':>10}{'mod9':>9}{'res':>8}")
    print(hdr)
    rows = []
    for rel, tag in (which or SPATIAL_SET):
        d = os.path.join(root, rel)
        if not os.path.isdir(d):
            print(f"{rel:<32}{tag:<16}  MISSING")
            continue
        L = load_3x3(d)
        if L is None:
            print(f"{rel:<32}{tag:<16}  UNRECOGNISED")
            continue
        p3, s3, n_rec, fam, meta = L
        S = spatial_suite(p3, s3, n_rec, seeds)
        by = {r["g"]: r for r in S["ladder"]}
        neigh = [by[g]["E_pp"] for g in (6, 8, 9, 12, 16)]
        s1 = "PASS" if by[9]["E_pp"] == max(neigh) else "fail"
        ea, eo = S["phase"][0]["E_pp"], S["phase"][4]["E_pp"]
        ratio = ea / eo if eo > 0 else float("inf")
        print(f"{rel:<32}{tag:<16}{by[8]['E_pp']:>9.4f}{by[9]['E_pp']:>9.4f}"
              f"{by[12]['E_pp']:>9.4f}{by[16]['E_pp']:>9.4f}{s1:>5}"
              f"{ea:>9.4f}{eo:>9.4f}{ratio:>10.2f}{S['mod9']['E_pp']:>9.4f}"
              f"{S['phase'][0]['res_pp']:>8.4f}")
        rows.append(dict(arm=rel, tag=tag, fam=fam, S=S, s1=s1, ratio=ratio,
                         ea=ea, eo=eo, mod9=S["mod9"]["E_pp"],
                         res=S["phase"][0]["res_pp"]))
    if rows:
        print()
        print("FULL ABSOLUTE-g LADDER (E in pp), all arms:")
        print(f"  {'arm':<32}" + "".join(f"{g:>8}" for g in G_LADDER))
        for r in rows:
            by = {x["g"]: x for x in r["S"]["ladder"]}
            print(f"  {r['arm']:<32}" + "".join(f"{by[g]['E_pp']:>8.4f}" for g in G_LADDER))
        print()
        print("PHASE SCAN AT g=9 (E in pp) -- offset 0 is kernel-aligned:")
        print(f"  {'arm':<32}" + "".join(f"{o:>10}" for o in (0, 1, 2, 3, 4)))
        for r in rows:
            print(f"  {r['arm']:<32}"
                  + "".join(f"{r['S']['phase'][o]['E_pp']:>10.4f}" for o in (0, 1, 2, 3, 4)))
    return rows


def s4_report(root, seeds=(101, 202, 303), which=None):
    print("=" * 100)
    print("S4 -- ROW / COLUMN / INTERACTION IN THE cout x cin FILTER-MEAN MATRIX")
    print("F = SS_real / SS_null against the same within-tensor permutation null.")
    print("Registered before running: F_int <= 1.10 => the g=9 effect is ADDITIVE row+column")
    print("marginals and the word 'filter' may NOT be used.")
    print("=" * 100)
    print(f"{'arm':<32}{'tag':<16}{'F_row':>10}{'F_col':>10}{'F_int':>10}"
          f"{'res(F)':>9}  verdict")
    out = []
    for rel, tag in (which or SPATIAL_SET):
        d = os.path.join(root, rel)
        if not os.path.isdir(d):
            print(f"{rel:<32}{tag:<16}  MISSING")
            continue
        L = load_3x3(d)
        if L is None:
            continue
        p3, s3, n_rec, fam, meta = L
        S = filter_interaction(p3, s3, seeds)
        v = "FILTER" if S["inter"]["F"] > 1.10 else "additive"
        print(f"{rel:<32}{tag:<16}{S['row']['F']:>10.3f}{S['col']['F']:>10.3f}"
              f"{S['inter']['F']:>10.3f}{S['inter']['F_res']:>9.4f}  {v}")
        out.append(dict(arm=rel, tag=tag, fam=fam, S=S, verdict=v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--curve", default=None)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--spatial", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--s4", action="store_true")
    ap.add_argument("--s4all", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    seeds = tuple(101 * (i + 1) for i in range(a.seeds))
    if a.selftest:
        selftest()
    if a.gate:
        gate(a.root, only=a.only, limit=a.limit)
    if a.curve:
        one_arm(a.root, a.curve, seeds=seeds)
    if a.report:
        report(a.root, seeds=seeds, only=a.only)
    if a.spatial:
        which = [(a.only, "cli")] if a.only else None
        spatial_report(a.root, seeds=seeds, which=which)
    if a.s4:
        which = [(a.only, "cli")] if a.only else None
        s4_report(a.root, seeds=seeds, which=which)
    if a.s4all:
        ds = sorted(C60.arms(a.root, "weightwise"))
        which = [(os.path.relpath(d, a.root), C60.arm_label(d)) for d in ds]
        s4_report(a.root, seeds=seeds, which=which)


if __name__ == "__main__":
    main()
