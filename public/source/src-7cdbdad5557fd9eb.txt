#!/usr/bin/env python3
r"""c63_uladder_corpus.py -- C62-C: THE CONV-ONLY u-LADDER ON ALL FOUR FAMILIES.

WHY THIS EXISTS
---------------
FINDINGS 62.8 registered **C62-C** verbatim:

    "The u-ladder was scored on ONE arm (62.4).  The corpus sweep of S1/S2/S3 was launched
     this tick; whatever it returns, H2's *location* claim needs the conv-only u-ladder on
     all four families before it is written as anything but a single-arm observation."

CORRECTIONS 91.10(g) then ranked C62-C FIRST in the offline queue.  This module discharges
it.  It is a DRIVER over `c62_blocksize_curve.py`, not a reimplementation: the ladder, the
block index, the nested SS, the noise split, the within-tensor null and the clamp rule are
all c62's (and through c62, c59's and c60's) code, unchanged.  The arm set is c62's
`SPATIAL_SET` **object**, not a copy of it, so the u-ladder is scored on exactly the arms
62.9's absolute-g sweep used and the two ladders are comparable arm-by-arm.  Selftest T1
asserts that identity rather than trusting it.

WHAT IS NEW HERE IS ONE CROSS-LADDER CONSISTENCY TEST (P3), AND IT CAN FAIL
---------------------------------------------------------------------------
62.9 located the structure on an ABSOLUTE-g ladder: argmax at g <= 9 in 17 of 17 arms.
62.4 located it on a CHANNEL-UNIT ladder on one arm: peak at u = 1/1024.  Those are two
readouts of the same `neg_counts.npy` through two different block indices.  If they disagree
about scale, one of the two published numbers is wrong.  They have never been compared,
because the u-ladder existed on one arm only.

P3 makes the comparison quantitative.  In a conv tensor of shape (cout, cin, 3, 3) the row
width is `cin*9`, so one 3x3 kernel is `u = 9/row_width` of a channel.  Row width varies
27..4608 ACROSS the stack, so a single u is a different absolute g in every tensor and the
channel ladder BLURS the kernel scale -- which is exactly why 62.5 registered the absolute
ladder in the first place.  The blur has a direction, though: if the structure lives at
g <= 9 absolute, then a channel-unit block only stays inside it when u <= 9/row_width, and
the coordinate-weighted median row width says where that is for the arm as a whole.

PRE-REGISTERED SCORING -- WRITTEN AND COMMITTED BEFORE ANY ARM BEYOND
`probes_p5/probe_w_a3_s0` WAS SCORED ON THE u-LADDER (2026-08-23, cycle 63)
---------------------------------------------------------------------------
Clean arms only (17).  The two exception arms (`ml5/probe_w_m2_s0` ms=1e-2,
`bo6/probe_w_adw_s0` AdamW-base) are FINDINGS 59.4's set, are reported separately and are
NEVER pooled -- CORRECTIONS 62 keeps ms=1e-2 unquotable.

P1  **H2's own registered refutation, copied verbatim from c62's docstring, now scored on
    the corpus instead of on one arm:**
        "H2 SUB-CHANNEL. argmax_u E(u) sits BELOW u = 1 by more than 3 rungs ...
         REFUTATION: E(1/8) <= E(1) within the null's MC resolution in a majority of clean
         arms."
    Scored as: refuted if `E(1/8) - E(1) <= res` in >= 9 of 17 clean arms, where
    `res = max(res_pp(1/8), res_pp(1))`.

P2  **THE ARGMAX FORM OF H2.** argmax_u E(u) <= 1/16 (i.e. more than 3 rungs below u=1) in
    >= 15 of 17 clean arms.
    REFUTATION: argmax >= 1/8 in >= 3 arms.  P2 is stricter than P1 and can fail while P1
    holds; both are reported.

P3  **CROSS-LADDER CONSISTENCY -- THE ONE THAT CAN OVERTURN A PUBLISHED NUMBER.**
    For each clean arm let `w50` be the coordinate-weighted median row width over its 3x3
    conv tensors and `u_k = 9 / w50` the channel-unit size of one kernel.  62.9's absolute
    result predicts the channel-unit peak sits AT OR BELOW that scale:
        REGISTERED: `u_peak <= 2 * u_k` (at or below one kernel, allowing one ladder rung)
        in >= 15 of 17 clean arms.
        REFUTATION: `u_peak > 2 * u_k` in >= 6 of 17 arms -- the two ladders would then be
        locating different scales and NEITHER 62.4's peak NOR 62.9's knee could be quoted
        until that is resolved.
    P3 is a consistency test between two of this project's own readouts.  It is registered
    here precisely because it has a failure mode that costs us a published sentence.

P4  **H1 AT CORPUS SCALE.** c62 registered H1 (channel knee: argmax within a factor 2 of
    u = 1) and refuted it on one arm.  Scored here as: H1 refuted if argmax is more than 3
    rungs from u = 1 in >= 9 of 17 clean arms.

READOUT LIMIT, CARRIED FORWARD FROM 62.3 / CORRECTIONS 91.4 AND NOT RE-DISCOVERED
---------------------------------------------------------------------------------
`max(raw - noise, 0)` reads R = 1 by construction where it binds, so a PERFECTLY homogeneous
block is the one structure this readout cannot locate and the peak is then misread one rung
low.  Clamped rungs are excluded from the argmax by c62's `peak(require_unclamped=True)` and
are COUNTED here per arm.  Any arm with a clamped rung at or adjacent to its peak is flagged.

SCOPE (STANDING RULE 10, and it is not optional)
------------------------------------------------
Conv tensors only for the ladder; 3x3 conv tensors only for `w50` (1x1 shortcut convs have
no kernel, exactly as in c62's `load_3x3`).  We measure `z`, the META-gradient; Adam-mini
argues about `G`.  **No document may write "we refuted Adam-mini."**

USAGE
    python3 analysis/c63_uladder_corpus.py --selftest
    python3 analysis/c63_uladder_corpus.py --run            [--root ..] [--limit N]
    python3 analysis/c63_uladder_corpus.py --score results/c63_uladder.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c59_row_premise as C59          # noqa: E402
import c60_exception_mechanism as C60  # noqa: E402
import c62_blocksize_curve as C62      # noqa: E402


# --------------------------------------------------------------------------------------
# Geometry: row width, and the channel-unit size of one 3x3 kernel.
# --------------------------------------------------------------------------------------

def row_widths(shapes3):
    """(width, n_coords) per 3x3 conv tensor.  Row = one output channel = prod(sh[1:]),
    which is c59's `p_size[0]` grouping verbatim (HF_patched.py:147)."""
    out = []
    for _nm, sh in shapes3:
        w = int(np.prod(sh[1:]))
        out.append((w, int(np.prod(sh))))
    return out


def coord_weighted_median(pairs):
    """Median row width weighted by how many COORDINATES sit at that width.

    Tensor-weighted would let a 27-wide first conv (432 coords) count as much as a
    4608-wide late conv (2.36M coords).  Coordinate weighting is the one that says where
    the arm's mass is.  Reported alongside the tensor-weighted value, never instead of it.
    """
    if not pairs:
        return float("nan")
    ws = np.array([w for w, _ in pairs], dtype=float)
    ns = np.array([n for _, n in pairs], dtype=float)
    o = np.argsort(ws)
    ws, ns = ws[o], ns[o]
    c = np.cumsum(ns)
    half = c[-1] / 2.0
    return float(ws[int(np.searchsorted(c, half, side="left"))])


def rung_distance(u_a, u_b):
    """Distance in ladder rungs (the ladder is powers of two)."""
    return abs(math.log(u_a, 2.0) - math.log(u_b, 2.0))


# --------------------------------------------------------------------------------------
# Per-arm measurement.
# --------------------------------------------------------------------------------------

def measure_arm(root, rel, tag, seeds=(101, 202, 303)):
    d = os.path.join(root, rel)
    if not os.path.isdir(d):
        return None
    L = C62.load_conv(d)
    if L is None:
        return None
    pc, sc, _pall, _shall, n_rec, fam, meta = L
    rows = C62.curve(pc, sc, n_rec, C62.U_LADDER, seeds)
    pk, pi = C62.peak(rows)
    by = {r["label"]: r for r in rows}
    e1, e8 = by["1"], by["1/8"]

    L3 = C62.load_3x3(d)
    pairs = row_widths(L3[1]) if L3 is not None else []
    w50 = coord_weighted_median(pairs)
    w50_t = float(np.median([w for w, _ in pairs])) if pairs else float("nan")

    n_clamped = sum(1 for r in rows if r["clamped"])
    peak_adj_clamped = False
    if pi is not None:
        for j in (pi - 1, pi, pi + 1):
            if 0 <= j < len(rows) and rows[j]["clamped"]:
                peak_adj_clamped = True

    return dict(
        arm=rel, tag=tag, fam=fam, n_rec=n_rec,
        exc=C62.is_exception(C60.arm_label(d)),
        n_conv_coords=int(pc.size), n_conv_tensors=len(sc),
        w50=w50, w50_tensorwt=w50_t, n3x3=len(pairs),
        u_kernel=(9.0 / w50) if w50 == w50 else float("nan"),
        E1=e1["E_pp"], res1=e1["res_pp"],
        E8=e8["E_pp"], res8=e8["res_pp"],
        u_peak=(pk["u"] if pk else float("nan")),
        peak_label=(pk["label"] if pk else "-"),
        E_peak=(pk["E_pp"] if pk else float("nan")),
        res_peak=(pk["res_pp"] if pk else float("nan")),
        n_clamped=n_clamped, peak_adj_clamped=peak_adj_clamped,
        rows=[{k: r[k] for k in ("label", "u", "E_pp", "res_pp", "clamped",
                                 "n_blocks", "bsz_min", "bsz_max")} for r in rows],
    )


# --------------------------------------------------------------------------------------
# Scoring -- the registered rules, and nothing else.
# --------------------------------------------------------------------------------------

def score_p1(a):
    """True == H2 REFUTED on this arm (E(1/8) not above E(1) beyond MC resolution)."""
    res = max(a["res8"], a["res1"])
    return (a["E8"] - a["E1"]) <= res


def score_p2(a):
    return a["u_peak"] <= 0.0625 + 1e-12          # <= 1/16


def score_p3(a):
    if not (a["u_kernel"] == a["u_kernel"]):
        return None
    return a["u_peak"] <= 2.0 * a["u_kernel"] * (1 + 1e-12)


def score_p4(a):
    return rung_distance(a["u_peak"], 1.0) > 3.0   # H1 refuted on this arm


def score(arms, min_arms=15):
    clean = [a for a in arms if not a["exc"]]
    exc = [a for a in arms if a["exc"]]
    if len(clean) < min_arms:
        raise SystemExit(
            f"REFUSING TO SCORE: only {len(clean)} clean arms loaded, need >= {min_arms}. "
            "A coverage test over too few items is the silent-zero failure CORRECTIONS "
            "55.1(8) hardened against.")
    n = len(clean)
    p1 = sum(1 for a in clean if score_p1(a))
    p2 = sum(1 for a in clean if score_p2(a))
    p3v = [score_p3(a) for a in clean]
    p3 = sum(1 for v in p3v if v is True)
    p3n = sum(1 for v in p3v if v is not None)
    p4 = sum(1 for a in clean if score_p4(a))
    return dict(n_clean=n, n_exc=len(exc),
                p1_refuted=p1, p1_verdict=("REFUTED" if p1 * 2 >= n else "SURVIVES"),
                p2_hits=p2, p2_verdict=("HOLDS" if p2 >= 15 else
                                        ("REFUTED" if (n - p2) >= 3 else "UNDECIDED")),
                p3_hits=p3, p3_scored=p3n,
                p3_verdict=("CONSISTENT" if p3 >= 15 else
                            ("INCONSISTENT" if (p3n - p3) >= 6 else "UNDECIDED")),
                p4_hits=p4, p4_verdict=("H1 REFUTED" if p4 * 2 >= n else "H1 SURVIVES"))


# --------------------------------------------------------------------------------------
# Selftests
# --------------------------------------------------------------------------------------

def _mk(**kw):
    d = dict(arm="a", tag="t", fam="r18", exc=False, u_kernel=1.0 / 64,
             E1=0.04, res1=0.001, E8=0.04, res8=0.001, u_peak=1.0)
    d.update(kw)
    return d


def selftest():
    n = 0

    def ok(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # ---- T1: the arm set is c62's OBJECT, not a copy -------------------------------
    ok(SPATIAL_SET is C62.SPATIAL_SET, "T1 arm set must BE c62.SPATIAL_SET")
    ok(len(SPATIAL_SET) == 19, f"T1 expected 19 arms, got {len(SPATIAL_SET)}")
    ok(sum(1 for _, t in SPATIAL_SET if "EXC" in t) == 2, "T1 expected 2 EXC arms")

    # ---- T2: row width == prod(sh[1:]) == c59's row grouping -----------------------
    rw = row_widths([("c1", (8, 3, 3, 3)), ("c2", (6, 4, 3, 3))])
    ok(rw == [(27, 216), (36, 216)], f"T2 row_widths wrong: {rw}")
    ok(row_widths([("c", (64, 512, 3, 3))]) == [(4608, 64 * 4608)], "T2 wide conv")

    # ---- T3: coordinate-weighted median ---------------------------------------------
    ok(coord_weighted_median([(27, 10), (100, 1000)]) == 100.0, "T3 mass at the wide one")
    ok(coord_weighted_median([(27, 1000), (100, 10)]) == 27.0, "T3 mass at the narrow one")
    ok(coord_weighted_median([]) != coord_weighted_median([]), "T3 empty -> nan")
    # and it must DIFFER from the tensor-weighted median when the mass is lopsided
    pairs = [(27, 10), (36, 10), (4608, 100000)]
    ok(coord_weighted_median(pairs) == 4608.0, "T3 coord-weighted follows mass")
    ok(float(np.median([w for w, _ in pairs])) == 36.0, "T3 tensor-weighted does not")

    # ---- T4: rung distance -----------------------------------------------------------
    ok(abs(rung_distance(1.0, 1.0)) < 1e-12, "T4 zero")
    ok(abs(rung_distance(1.0 / 8, 1.0) - 3.0) < 1e-12, "T4 three rungs")
    ok(abs(rung_distance(1.0 / 1024, 1.0) - 10.0) < 1e-12, "T4 ten rungs")
    ok(rung_distance(4.0, 1.0) == rung_distance(0.25, 1.0), "T4 symmetric in direction")

    # ---- T5: P1 (H2's registered refutation) ----------------------------------------
    ok(score_p1(_mk(E8=0.04, E1=0.04)) is True, "T5 equal -> refuted")
    ok(score_p1(_mk(E8=0.15, E1=0.04)) is False, "T5 clearly above -> not refuted")
    ok(score_p1(_mk(E8=0.0405, E1=0.04, res1=0.001, res8=0.001)) is True,
       "T5 rise inside MC resolution -> refuted (this is the whole point of the bar)")
    # The bar is max(res8, res1).  0.0008 clears the SMALLER resolution and not the LARGER,
    # so this cell distinguishes max from min and would flip if the bar were min.
    ok(score_p1(_mk(E8=0.0408, E1=0.04, res1=0.0005, res8=0.001)) is True,
       "T5 bar uses the MAX of the two resolutions")
    ok(score_p1(_mk(E8=0.0408, E1=0.04, res1=0.0005, res8=0.0005)) is False,
       "T5 same rise against the SMALLER bar alone would NOT be refuted")
    ok(score_p1(_mk(E8=0.03, E1=0.04)) is True, "T5 below -> refuted")

    # ---- T6: P2 / P4 argmax forms ----------------------------------------------------
    ok(score_p2(_mk(u_peak=1.0 / 16)) is True, "T6 1/16 is inside (>3 rungs is 1/16)")
    ok(score_p2(_mk(u_peak=1.0 / 8)) is False, "T6 1/8 is exactly 3 rungs, NOT more")
    ok(score_p2(_mk(u_peak=1.0 / 1024)) is True, "T6 fine peak")
    ok(score_p4(_mk(u_peak=1.0 / 16)) is True, "T6 H1 refuted at 4 rungs")
    ok(score_p4(_mk(u_peak=1.0 / 8)) is False, "T6 H1 survives at exactly 3 rungs")
    ok(score_p4(_mk(u_peak=1.0)) is False, "T6 H1 survives at the channel")
    ok(score_p4(_mk(u_peak=64.0)) is True, "T6 H1 refuted ABOVE the channel too")

    # ---- T7: P3 cross-ladder rule ----------------------------------------------------
    ok(score_p3(_mk(u_peak=1.0 / 64, u_kernel=1.0 / 64)) is True, "T7 exactly at kernel")
    ok(score_p3(_mk(u_peak=1.0 / 32, u_kernel=1.0 / 64)) is True, "T7 one rung above ok")
    ok(score_p3(_mk(u_peak=1.0 / 16, u_kernel=1.0 / 64)) is False, "T7 two rungs above fails")
    ok(score_p3(_mk(u_peak=1.0 / 1024, u_kernel=1.0 / 64)) is True, "T7 far below ok")
    ok(score_p3(_mk(u_kernel=float("nan"))) is None, "T7 no 3x3 tensors -> unscorable")

    # ---- T8: the corpus scorer, on planted arm sets -----------------------------------
    allref = [_mk(arm=f"a{i}", E8=0.04, E1=0.04, u_peak=1.0 / 1024, u_kernel=1.0 / 64)
              for i in range(17)]
    s = score(allref)
    ok(s["n_clean"] == 17, "T8 clean count")
    ok(s["p1_verdict"] == "REFUTED", "T8 all-flat -> H2 refuted")
    ok(s["p2_verdict"] == "HOLDS" and s["p2_hits"] == 17, "T8 P2 holds")
    ok(s["p3_verdict"] == "CONSISTENT", "T8 P3 consistent")
    ok(s["p4_verdict"] == "H1 REFUTED", "T8 P4")

    alive = [_mk(arm=f"a{i}", E8=0.30, E1=0.04, u_peak=1.0 / 1024, u_kernel=1.0 / 64)
             for i in range(17)]
    s2 = score(alive)
    ok(s2["p1_verdict"] == "SURVIVES", "T8 clear rise -> H2 survives")

    # exactly at the majority boundary: 9 of 17 refute -> REFUTED (>= half)
    mixed = ([_mk(arm=f"r{i}", E8=0.04, E1=0.04) for i in range(9)]
             + [_mk(arm=f"s{i}", E8=0.30, E1=0.04) for i in range(8)])
    ok(score(mixed)["p1_verdict"] == "REFUTED", "T8 9/17 is a majority")
    mixed2 = ([_mk(arm=f"r{i}", E8=0.04, E1=0.04) for i in range(8)]
              + [_mk(arm=f"s{i}", E8=0.30, E1=0.04) for i in range(9)])
    ok(score(mixed2)["p1_verdict"] == "SURVIVES", "T8 8/17 is not")

    # P3 inconsistency must be reachable, or the test is decorative
    bad = [_mk(arm=f"b{i}", u_peak=1.0 / 4, u_kernel=1.0 / 64) for i in range(17)]
    ok(score(bad)["p3_verdict"] == "INCONSISTENT", "T8 P3 CAN fail")
    part = ([_mk(arm=f"g{i}", u_peak=1.0 / 128, u_kernel=1.0 / 64) for i in range(13)]
            + [_mk(arm=f"b{i}", u_peak=1.0 / 4, u_kernel=1.0 / 64) for i in range(4)])
    ok(score(part)["p3_verdict"] == "UNDECIDED", "T8 P3 undecided band exists")

    # ---- T9: the coverage guard actually refuses -------------------------------------
    try:
        score([_mk(arm=f"a{i}") for i in range(14)])
        ok(False, "T9 coverage guard did not fire")
    except SystemExit:
        ok(True, "T9 coverage guard fires below 15 clean arms")
    try:
        score([])
        ok(False, "T9 empty did not fire")
    except SystemExit:
        ok(True, "T9 empty refuses (silent-zero hardening)")

    # ---- T10: exception arms are never pooled ----------------------------------------
    with_exc = ([_mk(arm=f"a{i}", E8=0.04, E1=0.04) for i in range(17)]
                + [_mk(arm="e0", exc=True, E8=99.0, E1=0.04),
                   _mk(arm="e1", exc=True, E8=99.0, E1=0.04)])
    s3 = score(with_exc)
    ok(s3["n_clean"] == 17 and s3["n_exc"] == 2, "T10 split counts")
    ok(s3["p1_verdict"] == "REFUTED",
       "T10 two huge exception arms must not move a clean verdict")

    # ---- T11: END-TO-END through c62 on synthetic data with structure at a KNOWN u ----
    # This re-exercises c62's G3 recovery through THIS module's argmax path, so the
    # readout used below is validated here and not merely inherited.
    rng = np.random.default_rng(4242)
    shapes = [("c1", (32, 8, 3, 3)), ("c2", (32, 8, 3, 3))]      # row width 72
    n_t = 32 * 72
    n_rec = 2000
    for u0, want in ((1.0 / 8, 1.0 / 8), (1.0 / 2, 1.0 / 2)):
        g0 = max(1, int(round(72 * u0)))
        P = []
        for _nm, sh in shapes:
            nt = int(np.prod(sh))
            nb = int(math.ceil(nt / g0))
            lev = 0.5 + rng.normal(0, 0.06, nb)
            P.append(np.repeat(lev, g0)[:nt])
        P = np.clip(np.concatenate(P), 0.02, 0.98)
        p = rng.binomial(n_rec, P) / float(n_rec)
        rows = C62.curve(p, shapes, n_rec, C62.U_LADDER, (101, 202, 303))
        pk, _ = C62.peak(rows)
        ok(pk is not None, "T11 a peak exists")
        assert pk is not None
        ok(rung_distance(pk["u"], want) <= 1.0,
           f"T11 planted u={u0} recovered at {pk['label']} (>1 rung away)")
    # And a NO-STRUCTURE control must not produce a large E on any USABLE rung.
    # The restriction to unclamped rungs is not a convenience: with pure-noise coordinates
    # SS_within IS the noise, so `max(raw - noise, 0)` binds on most rungs and reads
    # |E| up to 63 pp BY CONSTRUCTION (CORRECTIONS 89.6 / 91.4).  Those cells are excluded
    # from the argmax by c62's peak(require_unclamped=True), so the control that matters is
    # the one over exactly the cells the readout can use.  Asserting over ALL rungs was this
    # module's own wrong assertion, caught here, and it is recorded rather than quietly cut.
    p0 = rng.binomial(n_rec, np.full(n_t * 2, 0.5)) / float(n_rec)
    rows0 = C62.curve(p0, shapes, n_rec, C62.U_LADDER, (101, 202, 303))
    usable0 = [r for r in rows0 if not r["clamped"]]
    ok(len(usable0) >= 3, "T11 the flat control must leave SOME usable rung")
    ok(max(abs(r["E_pp"]) for r in usable0) < 1.0,
       "T11 flat control must stay near zero on usable rungs (null calibration)")
    pk0, _ = C62.peak(rows0)
    ok(pk0 is not None and abs(pk0["E_pp"]) < 1.0,
       "T11 the argmax readout itself must return ~0 on a structureless arm")

    print(f"  selftest: {n}/{n} PASS")


# c62's arm set, by reference.  T1 asserts the identity.
SPATIAL_SET = C62.SPATIAL_SET


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def run(root, limit=None, seeds=(101, 202, 303), out_json=None):
    print("=" * 118)
    print("C62-C  --  CONV-ONLY u-LADDER ON ALL FOUR FAMILIES (17 clean + 2 exception arms)")
    print("u = block size in units of ONE OUTPUT CHANNEL.  E = excess over a size-preserving")
    print("within-tensor regroup null, in pp.  u_k = 9/w50 = one 3x3 kernel in channel units.")
    print("=" * 118)
    hdr = (f"{'arm':<32}{'tag':<16}{'w50':>7}{'u_k':>9}{'E(1)':>9}{'E(1/8)':>9}"
           f"{'peak':>8}{'E_peak':>9}{'res':>8}{'E/res':>8}{'clmp':>5}")
    print(hdr)
    arms = []
    for rel, tag in (SPATIAL_SET[:limit] if limit else SPATIAL_SET):
        a = measure_arm(root, rel, tag, seeds)
        if a is None:
            print(f"{rel:<32}{tag:<16}  MISSING/UNRECOGNISED")
            continue
        ratio = a["E_peak"] / a["res_peak"] if a["res_peak"] > 0 else float("nan")
        print(f"{rel:<32}{tag:<16}{a['w50']:>7.0f}{C62.u_label(a['u_kernel']):>9}"
              f"{a['E1']:>9.4f}{a['E8']:>9.4f}{a['peak_label']:>8}{a['E_peak']:>9.4f}"
              f"{a['res_peak']:>8.4f}{ratio:>8.1f}{a['n_clamped']:>5d}")
        arms.append(a)
    print()
    print("FULL u-LADDER (E in pp), clean arms then exception arms:")
    labs = [C62.u_label(u) for u in C62.U_LADDER]
    print(f"  {'arm':<32}" + "".join(f"{l:>9}" for l in labs))
    for a in sorted(arms, key=lambda x: (x["exc"], x["tag"])):
        print(f"  {a['arm']:<32}"
              + "".join(f"{r['E_pp']:>9.4f}" for r in a["rows"]))
    print()
    s = score(arms)
    print("REGISTERED SCORING (clean arms only, n=%d):" % s["n_clean"])
    print(f"  P1  H2's own refutation (E(1/8) <= E(1) + res): fires in "
          f"{s['p1_refuted']}/{s['n_clean']}  ->  H2 {s['p1_verdict']}")
    print(f"  P2  argmax <= 1/16 (>3 rungs below the channel): "
          f"{s['p2_hits']}/{s['n_clean']}  ->  {s['p2_verdict']}")
    print(f"  P3  cross-ladder: u_peak <= 2*u_kernel: "
          f"{s['p3_hits']}/{s['p3_scored']}  ->  {s['p3_verdict']}")
    print(f"  P4  H1 (channel knee) refuted per arm: "
          f"{s['p4_hits']}/{s['n_clean']}  ->  {s['p4_verdict']}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(dict(arms=arms, score=s), f, indent=1)
        print(f"\nwrote {out_json}")
    return arms, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = tuple(101 * (i + 1) for i in range(a.seeds))
    if a.selftest:
        selftest()
    if a.run:
        run(a.root, limit=a.limit, seeds=seeds, out_json=a.json)


if __name__ == "__main__":
    main()
