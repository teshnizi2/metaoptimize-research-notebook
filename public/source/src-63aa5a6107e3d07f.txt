#!/usr/bin/env python3
r"""c64_partition_pareto.py -- IS THE OUTPUT-CHANNEL PARTITION ON THE PARETO FRONTIER?

WHY THIS EXISTS
---------------
FINDINGS 62.6 / CORRECTIONS 91.7 produced the C direction's first PRESCRIPTIVE sentence:

    "The structure a block method destroys is additive row (output channel) + input
     channel.  Every partition in the Adam-mini / Adalayer / SGG line is at the output
     channel or coarser, so it captures `a[o]` exactly and `b[i]` NOT AT ALL, by
     construction."

That sentence is STRUCTURAL.  It says the input-channel term is uncaptured; it never says
HOW MUCH is uncaptured, and it never asks the question a practitioner actually faces:

    Among partitions costing the SAME NUMBER OF STORED SCALARS, is the output channel the
    best available choice?

Every method in that line answers "output channel" by assumption -- Adam-mini from Hessian
block structure, Adalayer from second-moment storage coarseness, SGG from intra-correlated
groups (CORRECTIONS 87.1 established that none of them argues from sqrt(N) noise averaging;
that whole framing is WITHDRAWN and is not revived here).  None of them compares against the
TRANSPOSE of their own partition, which costs `cin` scalars instead of `cout` and is exactly
as easy to implement.

This instrument measures, for each IMPLEMENTABLE within-tensor partition, the fraction of
the per-coordinate meta-gradient sign-preference field that the partition can represent,
against that partition's own size-preserving null -- and puts it beside the partition's
COST in stored scalars.  It is the design-space version of 62.9's abstract g-ladder.

WHAT IS NEW HERE, STATED PRECISELY (so this is not a re-run of a closed question)
---------------------------------------------------------------------------------
CORRECTIONS 92.14(f) lists the filter hypothesis (62.5 / 62.9) as DONE.  This is not it.

  * 62.9 scored CONTIGUOUS ABSOLUTE-g blocks -- runs of g coordinates in flattened order.
    A contiguous run of 9 happens to coincide with one (o,i) kernel, but `out` and `in` are
    NOT contiguous runs at any g and CANNOT appear on that ladder.  The input-channel
    partition -- the one 91.7 names as structurally uncaptured -- has never been scored.
  * `F_row` / `F_col` (62.6, 62.7) are SS ratios against a null on the cout x cin matrix of
    FILTER MEANS, with the nine within-kernel taps averaged AWAY.  They answer "is there
    structure along this axis".  They are not a fraction of the field, they carry no cost
    axis, and a ratio of two SS terms on a reduced matrix cannot be compared to a stored-
    scalar budget.  CORRECTIONS 92.9b additionally REFUTED the reading of `F_col/F_row` as
    adaptation extent, and no sentence here revives it.
  * The quantity below is `E` -- the SAME noise-corrected, clamp-gated, null-subtracted
    statistic the published ladder uses (`c62._E_from_index`), imported BY REFERENCE, not
    restated.  Only the INDEX MAP is new.

SCOPE, STATED UP FRONT (STANDING RULES 10 AND 12)
-------------------------------------------------
  * 3x3 CONV TENSORS ONLY.  "Input channel" and "kernel" are undefined on 1-D tensors and
    on 1x1 shortcut convs, so they are excluded BY CONSTRUCTION, not by choice of result
    (this is 62.5's scope, reused verbatim via `c62.load_3x3`).
  * EVALUATION WINDOW: whole-run terminal `neg_counts`, i.e. `E[3x3conv, terminal]`.  Per
    STANDING RULE (12) a bare "E" is not a quantity in this campaign.
  * We measure `z`, the META-gradient.  Adam-mini / Adalayer / SGG argue about `G`, the base
    gradient.  The structural argument transfers; the quantities differ.  **No document may
    write "we refuted Adam-mini."**

THE PARTITIONS, AND THEIR COST IN STORED SCALARS PER TENSOR [O, I, 3, 3]
------------------------------------------------------------------------
    out       group by o                       O groups      <- Adam-mini / Adalayer / SGG
    in        group by i                       I groups      <- THE COST-MATCHED RIVAL
    oi        group by (o, i) = one 3x3 kernel O*I groups     <- 62.9's measured peak scale
    spatial   group by tap (index mod 9)        9 groups      <- 59.7's null direction
    tensor    one group                         1 group       <- R = 0 by construction

`out` and `in` are the cost-matched pair: on this stack O and I are equal or within 2x on
almost every tensor, and neither requires touching the training loop.

REGISTERED HYPOTHESES -- WRITTEN AND COMMITTED BEFORE ANY REAL ARM IS SCORED
----------------------------------------------------------------------------
H_A  PRIMARY, THE COST-MATCHED AXIS TEST.  Across clean arms, compare E(in) vs E(out).
       "out dominates"  if E(out) > E(in) in >= 80% of clean arms
       "in dominates"   if E(in) > E(out) in >= 80% of clean arms
       "CONFIG-DEPENDENT" otherwise.
     REGISTERED EXPECTATION: **CONFIG-DEPENDENT.**  62.7's F_col/F_row ladder crosses 1.0
     between ms=5e-4 and ms=1e-3, so the two axes are expected to trade places with config.
     A direction is registered DELIBERATELY (cycle 58's device): because CONFIG-DEPENDENT is
     what this tick expects, either uniform outcome REFUTES this tick's own expectation and
     is the stronger result.  If CONFIG-DEPENDENT fires, the prescriptive sentence is
     "no fixed axis is right", NOT "input channel is better".

H_B  THE PARETO TEST.  `out` is DOMINATED on an arm if some partition has n_blocks <=
     n_blocks(out) AND E > E(out) + 2*res.  Scored as the fraction of clean arms where `out`
     is dominated.
       DOMINATED   >= 50% of clean arms
       ON-FRONTIER <= 20% of clean arms
     Anything between is reported as neither and named as such.

H_C  SCALE REPLICATION THROUGH A NON-CONTIGUOUS PARTITION.  62.9 found the argmax of E at
     g <= 9 in 17 of 17 arms using CONTIGUOUS blocks.  If that is a statement about the
     KERNEL and not about contiguity, then the semantic (o,i) partition must also beat the
     output-channel partition: E(oi) > E(out) in >= 90% of clean arms.
     REFUTATION: if E(oi) does not beat E(out), 62.9's peak is a contiguity artifact and
     this instrument says so.  (Note `oi` costs I x more than `out`; H_C is a STRUCTURE
     claim, not a recommendation.  H_B is where cost is adjudicated.)

H_D  PRECONDITION, NOT A RESULT.  The `spatial` partition must reproduce FINDINGS 59.7's
     null: |E(spatial)| <= 0.02 pp on clean arms.  If it does not, this instrument disagrees
     with a published number and NOTHING in H_A / H_B / H_C is reported until that is
     resolved.  Scored FIRST.

GATES (`--gate`; every one falsifiable, none an assertion about the answer)
---------------------------------------------------------------------------
G1  CROSS-INSTRUMENT IDENTITY.  `semantic_index(shapes,"oi")` must induce the SAME partition
    as the published `c62.build_abs_index(shapes, 9, 0)` on a 3x3-only stack, and
    `semantic_index(shapes,"spatial")` the same as `c62.mod9_index(shapes)`.  This makes the
    new index maps a RELABELLING of validated ones, not a parallel implementation.
G2  NULL CALIBRATION.  On i.i.d. binomial coordinates with no structure, every partition's
    E must be within MC resolution.
G3  RECOVERY, IN BOTH DIRECTIONS.  Structure planted on `o` must be recovered by `out`;
    structure planted on `i` must be recovered by `in`.  **G3's second half is what makes a
    null result on H_A informative** -- without it, "we found no input-channel advantage"
    could just mean the readout cannot see input-channel structure at all.
G4  IDENTITY.  SS_total == SS_tensor + SS_block + SS_within to float precision, per
    partition.
G5  CLAMP.  CORRECTIONS 89.6 / 91.4: `max(raw - noise, 0)` reads R = 1 by construction when
    it binds.  Any clamp-binding cell is EXCLUDED and COUNTED, never reported.

A MEASURED LIMIT, RECORDED BEFORE ANY REAL ARM IS SCORED (selftest T13b/T13c)
-----------------------------------------------------------------------------
`oi` groups only NINE coordinates.  On structureless synthetic data at a 110k-coordinate
stack its E wanders over **[-0.72, +0.13] pp across data realisations while `res_pp` reads
0.06-0.38 pp** -- so for this partition the reported resolution UNDERSTATES the cell's true
uncertainty.  (The same 9-coordinate grouping is the published ladder's g=9 rung, so this is
a property of the readout at fine block sizes, not of this instrument.)  Consequences, fixed
here before the data:

  * **H_B is unaffected.**  `oi` costs O*I and `out` costs O, so `oi` can NEVER be a COST
    dominator of `out` (asserted, T13c).  H_B's only candidates are `in`, `spatial`,
    `tensor`, all of which calibrate to within resolution (T13).
  * **H_C is a SIGN TEST across independent arms, never a per-arm threshold**, and the
    measured wander leans NEGATIVE -- i.e. CONSERVATIVE for H_C.
  * **No per-arm E(oi) magnitude claim is licensed by this instrument.**

USAGE
    python3 analysis/c64_partition_pareto.py --selftest
    python3 analysis/c64_partition_pareto.py --gate   [--root ..]
    python3 analysis/c64_partition_pareto.py --report [--root ..]
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c59_row_premise as C59          # noqa: E402
import c60_exception_mechanism as C60  # noqa: E402
import c62_blocksize_curve as C62      # noqa: E402


KINDS = ("out", "in", "oi", "spatial", "tensor")


# --------------------------------------------------------------------------------------
# The only new machinery: semantic index maps.  Everything downstream is c62 by reference.
# --------------------------------------------------------------------------------------

def semantic_index(shapes, kind):
    """Coordinate -> (tensor_id, group_id, n_groups) for a SEMANTIC partition.

    `shapes` must be 3x3 conv tensors [O, I, 3, 3] in flattened C order, so the coordinate
    at (o, i, a, b) sits at ((o*I + i)*3 + a)*3 + b within its tensor.
    """
    tid_parts, bid_parts = [], []
    off = 0
    for t, (_nm, sh) in enumerate(shapes):
        O, I = int(sh[0]), int(sh[1])
        n = int(np.prod(sh))
        idx = np.arange(n, dtype=np.int64)
        if kind == "out":
            g, k = idx // (I * 9), O
        elif kind == "in":
            g, k = (idx // 9) % I, I
        elif kind == "oi":
            g, k = idx // 9, O * I
        elif kind == "spatial":
            g, k = idx % 9, 9
        elif kind == "tensor":
            g, k = np.zeros(n, dtype=np.int64), 1
        else:
            raise ValueError(kind)
        tid_parts.append(np.full(n, t, dtype=np.int32))
        bid_parts.append(off + g)
        off += k
    return np.concatenate(tid_parts), np.concatenate(bid_parts), off


def partition_cost(shapes, kind):
    """Stored scalars a `kind`-partitioned optimizer keeps for this tensor stack.

    Derived independently of `semantic_index` so that selftest T5 is a real cross-check
    and not a restatement.
    """
    tot = 0
    for _nm, sh in shapes:
        O, I = int(sh[0]), int(sh[1])
        tot += {"out": O, "in": I, "oi": O * I, "spatial": 9, "tensor": 1}[kind]
    return tot


def score_arm(p, shapes, n_rec, null_seeds=(101, 202, 303), kinds=KINDS):
    """-> {kind: dict(E_pp, res_pp, n_blocks, clamped, R, R_null)} via c62._E_from_index.

    The null, the noise correction and the clamp flags are the PUBLISHED ones, imported by
    reference; only the index map differs between kinds.
    """
    nulls = [C62.within_tensor_permute(p, shapes, s) for s in null_seeds]
    out = {}
    for k in kinds:
        tid, bid, nb = semantic_index(shapes, k)
        out[k] = C62._E_from_index(p, shapes, n_rec, tid, bid, nb, nulls)
    return out


def out_dominated(row, margin=2.0):
    """H_B.  True if some partition is no more expensive than `out` and beats it by
    `margin` x the MC resolution.  Returns (bool, winner_kind_or_None).

    G5: a clamp-binding cell reads R = 1 by construction (CORRECTIONS 89.6) and can neither
    dominate nor be dominated.
    """
    o = row.get("out")
    if o is None or o.get("clamped"):
        return False, None
    best, best_E = None, o["E_pp"]
    for k, v in row.items():
        if k == "out" or v.get("clamped"):
            continue
        if v["n_blocks"] > o["n_blocks"]:
            continue
        bar = o["E_pp"] + margin * max(o["res_pp"], v["res_pp"])
        if v["E_pp"] > bar and v["E_pp"] > best_E:
            best, best_E = k, v["E_pp"]
    return (best is not None), best


# --------------------------------------------------------------------------------------
# Selftests
# --------------------------------------------------------------------------------------

def _shapes():
    """3x3-only miniature stack with O != I in both tensors, so `out` and `in` differ in
    both cost and grouping and cannot be confused."""
    return [("c1", (8, 4, 3, 3)), ("c2", (6, 5, 3, 3))]


def _plant(kind, amp, seed, shapes, n_rec=2000, base=0.5, jitter=0.01):
    """Binomial-sampled p with structure planted on exactly one axis."""
    rng = np.random.default_rng(seed)
    parts = []
    for _nm, sh in shapes:
        O, I = int(sh[0]), int(sh[1])
        P = np.full((O, I, 9), base)
        if kind == "out":
            P += rng.normal(0, amp, size=(O, 1, 1))
        elif kind == "in":
            P += rng.normal(0, amp, size=(1, I, 1))
        elif kind == "oi":
            e = rng.normal(0, amp, size=(O, I))
            e = e - e.mean(axis=0, keepdims=True) - e.mean(axis=1, keepdims=True) + e.mean()
            P += e[:, :, None]
        elif kind == "spatial":
            P += rng.normal(0, amp, size=(1, 1, 9))
        elif kind == "none":
            pass
        else:
            raise ValueError(kind)
        P += rng.normal(0, jitter, size=P.shape)
        parts.append(np.clip(P, 0.02, 0.98).reshape(-1))
    P = np.concatenate(parts)
    return rng.binomial(n_rec, P) / float(n_rec)


def _groups_equal(bid_a, bid_b):
    """Two label vectors induce the same partition iff each is a function of the other."""
    a = np.asarray(bid_a)
    b = np.asarray(bid_b)
    if a.shape != b.shape:
        return False
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    return bool(np.array_equal(ia, ib))


def selftest():
    ok = 0

    def chk(cond, name):
        nonlocal ok
        if not cond:
            print(f"  FAIL {name}")
            raise SystemExit(1)
        ok += 1
        print(f"  ok   {name}")

    sh = _shapes()
    ntot = sum(int(np.prod(s)) for _n, s in sh)

    # ---- T1-T5  index-map structure -------------------------------------------------
    for kind in KINDS:
        tid, bid, nb = semantic_index(sh, kind)
        chk(len(tid) == ntot and len(bid) == ntot, f"T1 {kind}: covers all {ntot} coords")
        chk(bid.min() >= 0 and bid.max() == nb - 1 and len(np.unique(bid)) == nb,
            f"T2 {kind}: group ids are dense in [0,{nb})")
        # groups never straddle tensors
        chk(all(len(np.unique(tid[bid == g])) == 1 for g in np.unique(bid)),
            f"T3 {kind}: no group straddles a tensor")

    counts = {k: semantic_index(sh, k)[2] for k in KINDS}
    chk(counts["out"] == 8 + 6, f"T4 out cost = sum(O) = 14 (got {counts['out']})")
    chk(counts["in"] == 4 + 5, f"T4 in cost = sum(I) = 9 (got {counts['in']})")
    chk(counts["oi"] == 8 * 4 + 6 * 5, f"T4 oi cost = sum(O*I) = 62 (got {counts['oi']})")
    chk(counts["spatial"] == 9 + 9, f"T4 spatial cost = 9 per tensor (got {counts['spatial']})")
    chk(counts["tensor"] == 2, f"T4 tensor cost = 1 per tensor (got {counts['tensor']})")
    chk(all(partition_cost(sh, k) == counts[k] for k in KINDS),
        "T5 partition_cost agrees with the index map it describes")

    # ---- T6-T7  G1 cross-instrument identity against the PUBLISHED index maps --------
    _t, bid_oi, _n = semantic_index(sh, "oi")
    _t2, bid_abs, _n2 = C62.build_abs_index(sh, 9, 0)
    chk(_groups_equal(bid_oi, bid_abs),
        "T6 G1: semantic 'oi' == published c62.build_abs_index(g=9, offset=0)")
    _t3, bid_sp, _n3 = semantic_index(sh, "spatial")
    _t4, bid_m9, _n4 = C62.mod9_index(sh)
    chk(_groups_equal(bid_sp, bid_m9),
        "T7 G1: semantic 'spatial' == published c62.mod9_index")

    # ---- T8  'out' and 'in' are genuinely different partitions -----------------------
    chk(not _groups_equal(semantic_index(sh, "out")[1], semantic_index(sh, "in")[1]),
        "T8 'out' and 'in' induce different partitions")

    # ---- T9-T12  G3 RECOVERY, in every planted direction ----------------------------
    # The partitions form a REFINEMENT LATTICE: `oi` refines both `out` and `in` (every 3x3
    # kernel lies inside one output channel AND inside one input channel), so `oi` captures
    # by construction whatever `out` or `in` captures.  A recovery test may therefore only
    # demand separation from partitions that do NOT refine the planted axis.  Demanding it
    # of `oi` asserts something false about the lattice, not about the code.
    MUST_NOT_FIRE = {"out": ("in", "spatial"), "in": ("out", "spatial"),
                     "oi": ("out", "in", "spatial"), "spatial": ("out", "in", "oi")}
    for planted, blind in MUST_NOT_FIRE.items():
        p = _plant(planted, 0.10, 4200 + len(planted), sh)
        r = score_arm(p, sh, 2000)
        worst = max(r[k]["E_pp"] for k in blind)
        chk(r[planted]["E_pp"] > 5.0 * max(worst, 1e-6),
            f"T9 G3 recovery: planted '{planted}' -> E({planted})="
            f"{r[planted]['E_pp']:.3f}pp > 5x best NON-REFINING ({worst:.3f}pp via {blind})")

    # ---- T13  G2 NULL CALIBRATION -- ON A REALISTIC STACK, NOT THE TOY ----------------
    # MEASURED, and the reason the toy stack is not used here: at 558 coordinates the null
    # is simply unresolvable (structureless E swings +-1 to +-16 pp with res of the same
    # order).  That is a property of the stack size, not of the statistic.  Calibration is
    # therefore asserted where the real arms live -- 10^5..10^7 coordinates.
    big = [("c1", (64, 64, 3, 3)), ("c2", (128, 64, 3, 3))]
    p0 = _plant("none", 0.0, 777, big)
    r0 = score_arm(p0, big, 2000, null_seeds=(1, 2, 3, 4, 5, 6))
    for k in ("out", "in", "spatial"):
        chk(abs(r0[k]["E_pp"]) <= max(4.0 * r0[k]["res_pp"], 0.10),
            f"T13 G2 null: E({k})={r0[k]['E_pp']:+.4f}pp within resolution "
            f"({r0[k]['res_pp']:.4f}pp)")

    # ---- T13b  A MEASURED LIMIT ON THE FINE PARTITION, RECORDED BEFORE USE ------------
    # `oi` groups only 9 coordinates.  Across structureless data realisations its E wanders
    # over [-0.72, +0.13] pp on a 110k stack while `res_pp` reads 0.06-0.38 pp, so res
    # UNDERSTATES this cell's uncertainty.  Consequences, all stated before any real arm is
    # scored:  (a) H_B is UNAFFECTED -- `oi` costs O*I and can never be a COST dominator of
    # `out` (asserted in T13c);  (b) H_C is a SIGN TEST across independent arms, never a
    # per-arm threshold, and the measured bias leans NEGATIVE, i.e. CONSERVATIVE for H_C;
    # (c) no per-arm E(oi) magnitude claim is licensed by this instrument.
    chk(abs(r0["oi"]["E_pp"]) <= 1.0,
        f"T13b oi structureless |E|={abs(r0['oi']['E_pp']):.4f}pp <= 1.0pp "
        f"(res={r0['oi']['res_pp']:.4f}pp UNDERSTATES it -- limit recorded, not a bar)")
    chk(partition_cost(big, "oi") > partition_cost(big, "out"),
        "T13c 'oi' is strictly more expensive than 'out' -- it can never dominate on cost, "
        "so T13b's limit cannot reach H_B")

    # ---- T14  G4 IDENTITY -------------------------------------------------------------
    for k in ("out", "in", "oi", "spatial", "tensor"):
        tid, bid, nb = semantic_index(big, k)
        d = C62.ss_at_u(p0, big, 2000, None, tid, bid, nb)
        chk(d["identity_err"] < 1e-10,
            f"T14 G4 identity {k}: err={d['identity_err']:.3e}")

    # ---- T15  'tensor' is R = 0 by construction --------------------------------------
    chk(abs(r0["tensor"]["R"]) < 1e-12,
        f"T15 'tensor' partition has R = 0 by construction (got {r0['tensor']['R']:.3e})")

    # ---- T16-T18  H_B dominance logic on SYNTHETIC rows (no real data) ---------------
    mk = lambda E, n, res=0.01: dict(E_pp=E, res_pp=res, n_blocks=n, clamped=False)
    dom, win = out_dominated({"out": mk(1.0, 100), "in": mk(2.0, 90), "spatial": mk(0.1, 9)})
    chk(dom and win == "in", "T16 cheaper-and-better partition dominates 'out'")
    dom, win = out_dominated({"out": mk(2.0, 100), "in": mk(1.0, 90), "spatial": mk(0.1, 9)})
    chk((not dom) and win is None, "T17 'out' best at its cost tier is NOT dominated")
    dom, win = out_dominated({"out": mk(1.0, 100), "in": mk(3.0, 900), "spatial": mk(0.1, 9)})
    chk(not dom, "T18 a MORE EXPENSIVE better partition does not dominate 'out'")
    dom, win = out_dominated({"out": mk(1.0, 100), "in": mk(1.005, 90, res=0.01)})
    chk(not dom, "T19 a win inside 2x resolution does not count as dominance")
    dom, win = out_dominated({"out": mk(1.0, 100), "in": mk(2.0, 90, res=0.01),
                              "oi": mk(9.0, 90)}, )
    chk(dom and win == "oi", "T20 the STRONGEST qualifying dominator is reported")

    # ---- T21  clamped cells are excluded from dominance ------------------------------
    rows = {"out": mk(1.0, 100), "in": dict(E_pp=9.0, res_pp=0.01, n_blocks=90, clamped=True)}
    dom, win = out_dominated(rows)
    chk(not dom, "T21 G5: a clamped cell cannot dominate")

    # ---- T22  the CSV join key (G6 depends on it) ------------------------------------
    for rel, want in [("probes_fz3/probe_r10_w_s0", "fz3-r10-w-s0"),
                      ("probes_ff5/probe_c100_w_s1", "ff5-c100-w-s1"),
                      ("probes_p5/probe_w_a3_s0", "p5-w-a3-s0"),
                      ("probes_cl5/probe_w_cU_s2", "cl5-w-cU-s2"),
                      ("probes_bo6/probe_w_adw_s0", "bo6-w-adw-s0")]:
        chk(arm_to_run(rel) == want, f"T22 join key {rel} -> {want}")

    print(f"\n{ok}/{ok} selftests pass")
    return ok


# --------------------------------------------------------------------------------------
# Real-data gate + report
# --------------------------------------------------------------------------------------

def gate(root):
    print("=" * 100)
    print("G1 CROSS-INSTRUMENT IDENTITY ON REAL ARCHITECTURES")
    print("=" * 100)
    n = 0
    for rel, tag in C62.SPATIAL_SET:
        d = os.path.join(root, rel)
        if not os.path.isdir(d):
            continue
        L = C62.load_3x3(d)
        if L is None:
            continue
        _p, s3, _nrec, fam, _meta = L
        a = semantic_index(s3, "oi")[1]
        b = C62.build_abs_index(s3, 9, 0)[1]
        c = semantic_index(s3, "spatial")[1]
        e = C62.mod9_index(s3)[1]
        ok1, ok2 = _groups_equal(a, b), _groups_equal(c, e)
        print(f"  {rel:<32} {fam:>5}  oi==abs9 {ok1}   spatial==mod9 {ok2}   "
              f"cost out={partition_cost(s3,'out'):>5} in={partition_cost(s3,'in'):>5} "
              f"oi={partition_cost(s3,'oi'):>7}")
        if not (ok1 and ok2):
            print("  GATE FAILED")
            return False
        n += 1
    print(f"\nG1 passes on {n} real architectures")
    return True


def report(root, null_seeds=(101, 202, 303)):
    rows = []
    print("=" * 118)
    print("E[3x3conv, terminal] BY SEMANTIC PARTITION, WITH COST IN STORED SCALARS")
    print("scope: 3x3 conv tensors only; we measure z (meta-gradient), not G")
    print("=" * 118)
    hdr = (f"{'arm':<32}{'fam':>5}{'cls':>4}"
           f"{'E(out)':>10}{'E(in)':>10}{'E(oi)':>10}{'E(spat)':>9}"
           f"{'res':>7}{'#out':>7}{'#in':>7}{'dominated?':>12}")
    print(hdr)
    print("-" * 118)
    for rel, tag in C62.SPATIAL_SET:
        d = os.path.join(root, rel)
        if not os.path.isdir(d):
            continue
        L = C62.load_3x3(d)
        if L is None:
            continue
        p, s3, n_rec, fam, _meta = L
        lab = C60.arm_label(d)
        exc = C62.is_exception(lab)
        r = score_arm(p, s3, n_rec, null_seeds)
        dom, win = out_dominated(r)
        res = max(r[k]["res_pp"] for k in ("out", "in", "oi", "spatial"))
        print(f"{rel.split('/')[-1]:<32}{fam:>5}{'EXC' if exc else 'cln':>4}"
              f"{r['out']['E_pp']:>10.4f}{r['in']['E_pp']:>10.4f}"
              f"{r['oi']['E_pp']:>10.4f}{r['spatial']['E_pp']:>9.4f}"
              f"{res:>7.4f}{r['out']['n_blocks']:>7}{r['in']['n_blocks']:>7}"
              f"{(win if dom else '-'):>12}")
        rows.append(dict(arm=lab, rel=rel, fam=fam, tag=tag, exc=exc, r=r,
                         dominated=dom, winner=win))
    return rows


def score(rows):
    """H_D first (precondition), then H_A, H_B, H_C on CLEAN arms only."""
    clean = [x for x in rows if not x["exc"]]
    n = len(clean)
    print("\n" + "=" * 100)
    print(f"REGISTERED SCORING -- {n} clean arms ({len(rows)-n} exception arms excluded)")
    print("=" * 100)
    if n == 0:
        print("no clean arms; nothing scored")
        return

    # H_D -- precondition
    sp = [abs(x["r"]["spatial"]["E_pp"]) for x in clean]
    hd = max(sp) <= 0.02
    print(f"\nH_D PRECONDITION  |E(spatial)| max = {max(sp):.4f} pp (bar <= 0.02) -> "
          f"{'PASS' if hd else 'FAIL'}")
    if not hd:
        print("  H_D FAILED: this instrument disagrees with FINDINGS 59.7.")
        print("  NOTHING in H_A / H_B / H_C is reported until that is resolved.")
        return

    def frac(pred):
        k = sum(1 for x in clean if pred(x))
        return k, 100.0 * k / n

    k_out, p_out = frac(lambda x: x["r"]["out"]["E_pp"] > x["r"]["in"]["E_pp"])
    k_in, p_in = frac(lambda x: x["r"]["in"]["E_pp"] > x["r"]["out"]["E_pp"])
    verdict = ("out dominates" if p_out >= 80 else
               "in dominates" if p_in >= 80 else "CONFIG-DEPENDENT")
    print(f"\nH_A PRIMARY       E(out)>E(in) in {k_out}/{n} ({p_out:.0f}%), "
          f"E(in)>E(out) in {k_in}/{n} ({p_in:.0f}%)  ->  {verdict}")
    print(f"   registered expectation was CONFIG-DEPENDENT; this "
          f"{'CONFIRMS' if verdict=='CONFIG-DEPENDENT' else 'REFUTES'} it")

    k_d, p_d = frac(lambda x: x["dominated"])
    vb = ("DOMINATED" if p_d >= 50 else "ON-FRONTIER" if p_d <= 20 else "NEITHER")
    print(f"\nH_B PARETO        'out' dominated in {k_d}/{n} ({p_d:.0f}%)  ->  {vb}")
    from collections import Counter
    wins = Counter(x["winner"] for x in clean if x["dominated"])
    if wins:
        print(f"   dominating partitions: {dict(wins)}")

    k_c, p_c = frac(lambda x: x["r"]["oi"]["E_pp"] > x["r"]["out"]["E_pp"])
    vc = "REPLICATES 62.9" if p_c >= 90 else "FAILS -- 62.9's peak may be a contiguity artifact"
    print(f"\nH_C SCALE         E(oi)>E(out) in {k_c}/{n} ({p_c:.0f}%)  ->  {vc}")


def arm_to_run(rel):
    """`probes_fz3/probe_r10_w_s0` -> `fz3-r10-w-s0`, the CSV `run` key."""
    fam = os.path.dirname(rel).replace("probes_", "")
    rest = os.path.basename(rel).replace("probe_", "").replace("_", "-")
    return f"{fam}-{rest}"


def csv_index(repo):
    import csv as _csv
    path = os.path.join(repo, "results", "all_runs.csv")
    out = {}
    with open(path) as fh:
        for r in _csv.DictReader(fh):
            out.setdefault(r["run"], r)
    return out


def strata(rows, repo, verbose=True):
    """POST-HOC stratification of H_A by whether beta ADAPTS, joined to the CSV.

    LABELLED POST-HOC AND IT STAYS POST-HOC.  H_A was registered as a whole-corpus
    fraction; this names the config axis behind that verdict.  Per CORRECTIONS 76(1)/79 a
    post-hoc reading may not overturn a registered gate, and this one does not -- H_A's
    verdict (CONFIG-DEPENDENT) is what it explains, not what it replaces.

    G6 (join gate): every scored arm must resolve to exactly ONE CSV row, and the hand
    written frozen/free tag in `c62.SPATIAL_SET` must AGREE with that row's `meta` column.
    The stratum is taken from the CSV, never from the tag.
    """
    idx = csv_index(repo)
    recs, bad = [], []
    for x in rows:
        if x["exc"]:
            continue
        run = arm_to_run(x["rel"])
        cr = idx.get(run)
        if cr is None:
            bad.append((run, "no CSV row"))
            continue
        stratum = "frozen" if cr["meta"] == "fixed" else "free"
        tagged = "frozen" if "frozen" in x["tag"] else "free"
        if stratum != tagged:
            bad.append((run, f"tag says {tagged}, CSV meta={cr['meta']}"))
            continue
        recs.append(dict(run=run, stratum=stratum, cr=cr, r=x["r"],
                         dominated=x["dominated"], winner=x["winner"],
                         arch=cr["network"], ds=cr["dataset"], seed=cr["seed"],
                         ms=cr["meta_stepsize"], ep=cr["epochs_done"],
                         clip=cr["beta_clip"], a0=cr["alpha0"]))
    if bad:
        print("\nG6 JOIN GATE FAILED:")
        for b in bad:
            print("   ", b)
        return None
    if verbose:
        print("\n" + "=" * 112)
        print("POST-HOC (LABELLED): H_A's config axis is whether BETA ADAPTS.  Stratum from "
              "the CSV `meta` column, G6-gated.")
        print("=" * 112)
        print(f"{'run':<20}{'stratum':>8}{'arch':>16}{'ds':>10}{'ms':>7}{'ep':>4}"
              f"{'E(out)':>10}{'E(in)':>10}{'res':>8}  winner")
        for m in sorted(recs, key=lambda z: (z["stratum"], z["arch"], z["seed"])):
            eo, ei = m["r"]["out"]["E_pp"], m["r"]["in"]["E_pp"]
            res = max(m["r"]["out"]["res_pp"], m["r"]["in"]["res_pp"])
            w = "out" if eo > ei else "in"
            tie = " (within res)" if abs(eo - ei) <= res else ""
            print(f"{m['run']:<20}{m['stratum']:>8}{m['arch']:>16}{m['ds']:>10}"
                  f"{m['ms']:>7}{m['ep']:>4}{eo:>10.4f}{ei:>10.4f}{res:>8.4f}  {w}{tie}")
    return recs


def strata_score(recs):
    from collections import defaultdict
    by = defaultdict(list)
    for m in recs:
        by[m["stratum"]].append(m)
    print("\n" + "-" * 112)
    for st in ("frozen", "free"):
        g = by.get(st, [])
        if not g:
            continue
        k = sum(1 for m in g if m["r"]["out"]["E_pp"] > m["r"]["in"]["E_pp"])
        res_k = sum(1 for m in g
                    if abs(m["r"]["out"]["E_pp"] - m["r"]["in"]["E_pp"])
                    > max(m["r"]["out"]["res_pp"], m["r"]["in"]["res_pp"]))
        dom_k = sum(1 for m in g if m["dominated"])
        print(f"{st:>6}:  out>in in {k}/{len(g)}   "
              f"(resolved beyond res in {res_k}/{len(g)})   "
              f"'out' PARETO-DOMINATED in {dom_k}/{len(g)}")

    # the MATCHED-PAIR design: fz3 (frozen) vs ff5 (free) differ ONLY in `meta`
    fz = {(m["arch"], m["seed"]): m for m in recs if m["run"].startswith("fz3-")}
    ff = {(m["arch"], m["seed"]): m for m in recs if m["run"].startswith("ff5-")}
    keys = sorted(set(fz) & set(ff))
    print(f"\nMATCHED PAIRS fz3(frozen) vs ff5(free) -- identical arch/dataset/ms/alpha0/"
          f"clip/augment/epochs, ONLY `meta` differs:  n={len(keys)}")
    flips = 0
    for k in keys:
        a, b = fz[k], ff[k]
        wa = "out" if a["r"]["out"]["E_pp"] > a["r"]["in"]["E_pp"] else "in"
        wb = "out" if b["r"]["out"]["E_pp"] > b["r"]["in"]["E_pp"] else "in"
        same_budget = a["ep"] == b["ep"] and a["ms"] == b["ms"] and a["a0"] == b["a0"]
        flips += int(wa == "out" and wb == "in")
        print(f"   {k[0]:<16} s{k[1]}  frozen->{wa:<3}  free->{wb:<3}  "
              f"budget/ms/a0 matched={same_budget}  ep={a['ep']}/{b['ep']}")
    p = 2.0 * (0.5 ** len(keys)) if flips == len(keys) else float("nan")
    print(f"\n   out->in flip in {flips}/{len(keys)} matched pairs"
          + (f"   (sign test, two-sided p={p:.3f}, treating (arch,seed) as the unit; "
             f"at the ARCHITECTURE level n=3)" if flips == len(keys) else ""))
    print("\n   THIS IS POST-HOC.  It does not overturn H_A, which stands as "
          "CONFIG-DEPENDENT; it names the axis.")
    print("   It is NOT a revival of the adaptation-extent dose-response that "
          "CORRECTIONS 92.9b REFUTED:")
    print("   that was a CONTINUOUS magnitude claim with budget uncontrolled; this is a "
          "BINARY contrast at matched budget.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--strata", action="store_true")
    ap.add_argument("--root", default="..")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.gate:
        if not gate(a.root):
            raise SystemExit(1)
    if a.report or a.strata:
        rows = report(a.root)
        score(rows)
        if a.strata:
            recs = strata(rows, a.repo)
            if recs is None:
                raise SystemExit(1)
            strata_score(recs)


if __name__ == "__main__":
    main()
