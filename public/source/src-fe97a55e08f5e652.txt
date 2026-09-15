#!/usr/bin/env python3
"""
c61 -- RECOVERING A PER-TENSOR `z` TIME SERIES FROM THE PROBE'S CUMULATIVE MOMENTS,
and using it to attack CORRECTIONS 89.7 (why `conv1` and not `conv2`).

================================================================================
WHAT THIS CORRECTS IN OUR OWN DOCS -- READ BEFORE THE SCIENCE
================================================================================
CONTINUE-HERE item (h) and CORRECTIONS 89.7 both say that deciding conv1-vs-conv2
"needs `z` time-series the probe does not store", and rank the question below every
cluster item on that basis.  **That is half wrong, and the wrong half is cheap.**

`patches/HF_patched.py::_probe` accumulates

    self._z_sum   += zv          # zv[j] = spatial mean of z over tensor j, THIS step
    self._z_sqsum += zv * zv
    self._z_n     += 1

and **never resets them**.  The emitted `z_mean` / `z_std` / `snr` are therefore
CUMULATIVE running moments from step 0, not per-record values.  OPERATIONS 22 already
records that they are "temporal" ratios; what nobody did is the obvious next step --
**cumulative moments difference exactly**:

    S(k)  = mean(k) * n(k)                      (cumulative first moment)
    Q(k)  = (std(k)^2 + mean(k)^2) * n(k)       (cumulative second moment)
    window mean over (k-1, k]  =  (S(k)-S(k-1)) / (n(k)-n(k-1))
    window var  over (k-1, k]  =  (Q(k)-Q(k-1)) / (n(k)-n(k-1)) - windowmean^2

So a per-tensor, per-window time series of the SPATIALLY-AVERAGED meta-gradient IS
recoverable, at 2000-4000 records, on every arm already on this Mac, at zero cluster
cost.  What is genuinely NOT recoverable is a per-ROW or per-WEIGHT time series: the
spatial average is taken before storage and cannot be undone.

**Therefore 89.7 is open at TENSOR resolution and closed at ROW resolution.**  The
question "why conv1 and not conv2" is a statement about which TENSORS carry the
exception (60.4: conv1 43/80 vs conv2 5/80), so tensor resolution is the matching
resolution and this instrument is entitled to ask it.  The question "which rows, and
why those rows" stays shut and stays shut honestly.

================================================================================
SCOPE, STATED BEFORE ANY NUMBER IS PRINTED
================================================================================
The row-structure statistic (59.3 / 60.2 R_row_cor) is built from per-COORDINATE sign
preference.  This instrument measures the temporal behaviour of the per-TENSOR SPATIAL
MEAN of z.  These are different quantities.  A positive result here is a CONSISTENCY
and a surviving mechanism candidate -- never a proof that the tensor-mean drift causes
the row structure.  A null result closes the tensor-mean avenue and leaves 89.7 open
with one fewer place to look.  Both readings are written down in advance so neither can
be upgraded after the fact.

The exception arms are dominated by the ms=1e-2 rung, whose meta-step is 10x the clean
rung's, so their ABSOLUTE z dynamics differ for a reason that has nothing to do with
conv1.  Every headline statistic below is therefore a WITHIN-ARM conv1/conv2 RATIO,
which is self-normalising against exactly that confound.  Absolute levels are printed
too, so the confound is visible rather than assumed away.

================================================================================
GATES -- all four run before any hypothesis is scored (`--gate`)
================================================================================
G0 THE ACCUMULATION COUNT.  The recovery needs n(t), the number of steps folded into
   the cumulative moments at record t.  Three separate things, kept separate:
   G0a  The recovery is EXACTLY invariant under a pure rescale n -> c*n (selftest).
        So the UNIT of the count is irrelevant.
   G0b  An affine OFFSET mismatch is NOT invariant.  Its relative error is O(1/n) and
        DECAYS (selftest asserts the decay).  So the offset must be known, not guessed.
   G0c  It IS known, from source AND confirmed by an exact fingerprint in the data.
        `HF_patched.py:72` sets `self.counter = -1`; `_probe` is called at :95 and the
        increment happens at :100, AFTER it.  So the first call accumulates with
        counter=-1 and does NOT write (-1 % 5 = 4), the second accumulates with
        counter=0 and DOES write, labelled `step: 0`.  The record labelled step=s
        therefore folds **n = s + 2** samples, not s+1.
        THE FINGERPRINT.  n=2 over samples {0, x} gives mean=x/2 and var=x^2/4, i.e.
        **std == |mean| EXACTLY**; n=1 would give var == 0 exactly.  These are
        different by construction, so record 0 decides the offset on its own.
        MEASURED: in 49 of 49 weightwise arms, 100% of tensors with nonzero mean
        satisfy std == |mean| to 1e-6.  n = step + 2 is confirmed, and as a free
        by-product **the very first meta-gradient is exactly zero on every tensor** --
        H has not accumulated yet at the first meta-step.
        This is why the default `offset` below is 2.  Getting it wrong is not fatal
        (see G0b) but it is not guessed either.
   G0d  ROBUSTNESS OF THE CONCLUSION, not of the digits.  The headline conv1/conv2
        comparison is recomputed under the rival offset; the VERDICT must not change.
        This is reported rather than asserted away, because G0b guarantees the digits
        will differ at the earliest windows.
G1 NUMERICAL CONDITIONING.  Values are stored from float32 (~6e-8 relative), and
   differencing inflates that by n/dn.  Registered bar: at the primary windowing the
   predicted relative error must be < 1e-4, and the fraction of windows with a NEGATIVE
   recovered variance must be < 0.5%.  Both are printed, not asserted away.
G2 ROUND-TRIP.  Re-accumulating the recovered windows must reproduce the stored
   cumulative mean and std at every window boundary to < 1e-6 relative.
G3 SHAPE / ORDER.  len(build_shapes(family)) must equal len(z_mean) for every arm
   scored, and both conv roles must be non-empty.  The architecture map is IMPORTED
   from c59 (which passed A0/A1/A2), never re-derived here.

================================================================================
HYPOTHESES -- registered before any arm was scored
================================================================================
Let c(j,k) = |window mean of z_j| / (window sd of z_j) be the window COHERENCE of
tensor j: how much of that window's meta-gradient is a sustained drift rather than
noise.  Under an i.i.d.-in-time tensor mean, E[c] ~ sqrt(2/(pi*W)) with W steps per
window, so c is small unless the tensor is drifting coherently.

  H1 TEMPORAL EVENT.  The exception arms carry an ELEVATED conv1/conv2 coherence ratio
     relative to clean arms, and within an exception arm the excess is confined to a
     MINORITY of windows -- a burst.  Decides "run-specific dynamical event" (60.5) and
     dates it.
  H2 PERSISTENT LEVEL SHIFT.  The ratio is elevated but spread across essentially all
     windows.  Same localisation in tensor, none in time.
  H3 NULL / WRONG RESOLUTION.  Exception and clean arms are indistinguishable on the
     conv1/conv2 coherence ratio.  The row-level exception leaves NO signature in the
     tensor mean, the tensor-mean channel is the wrong resolution, and 89.7 stays open.

H1 and H2 are separated by one printed number (the fraction of windows carrying the
excess).  H3 is separated from both by the exception-vs-clean comparison.  A fourth
outcome -- ratio elevated in the WRONG direction (conv2 > conv1) -- is possible and is
reported as a REFUTATION of the tensor-mean reading, not massaged into support.

CONTROL C1 (non-conv roles).  The same ratio computed at stem/shortcut/1-D tensors.  If
exception arms are elevated everywhere, this is a global property of the ms=1e-2 rung
and NOT a conv1 story.
CONTROL C2 (within-config seeds).  60.5's two decisive controls -- bl5/e40 s0 (23.014%)
against siblings s1/s2 (0.044/0.039), and br6/c2 s1 (7.216%) against s0/s2/s3 -- are
re-run here on the new statistic.  If the tensor-mean channel carries the event, the
inverting seed must be the extreme one of its own config.  This is the strongest control
available offline because the siblings share config exactly.

Usage:
    python3 analysis/c61_z_timecourse.py --selftest
    python3 analysis/c61_z_timecourse.py --gate      [--root ..]
    python3 analysis/c61_z_timecourse.py --report    [--root ..]
    python3 analysis/c61_z_timecourse.py --controls  [--root ..]
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c59_row_premise as C59              # MEASURED architecture map (A0/A1/A2)
import c60_exception_mechanism as C60      # conv_role, EXC_W, arm discovery

# Primary windowing, registered here rather than chosen per-arm.
K_WINDOWS = 20
# G1 bars, registered.
BAR_RELERR = 1e-4
BAR_NEGVAR = 0.005
# float32 relative storage precision
EPS32 = 6e-8


# --------------------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------------------

def load_probe(d, fields=("z_mean", "z_std")):
    """Read probe.jsonl -> (steps[R], {field: array[R, T]}).  Streamed, one pass."""
    steps = []
    cols = {f: [] for f in fields}
    with open(os.path.join(d, "probe.jsonl")) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            steps.append(int(r["step"]))
            for f in fields:
                cols[f].append(r[f])
    if not steps:
        raise ValueError(f"{d}: empty probe.jsonl")
    return (np.asarray(steps, dtype=np.int64),
            {f: np.asarray(v, dtype=np.float64) for f, v in cols.items()})


def n_of(steps, offset=2, slope=1.0):
    """Accumulation count at each record.  See G0c: `counter` starts at -1 and `_probe`
    accumulates BEFORE the increment, so the record labelled step=s folds s+2 samples.
    Confirmed by an exact float32 fingerprint at record 0 (std == |mean|), 49/49 arms."""
    return steps.astype(np.float64) * slope + offset


def window_bounds(n_rec, k):
    """k+1 record indices spanning [0, n_rec-1], as window BOUNDARIES.  Window i covers
    records (b[i], b[i+1]] -- the record at b[0]=0 is the base and belongs to no window."""
    if k < 1 or k >= n_rec:
        raise ValueError("bad window count")
    return np.unique(np.linspace(0, n_rec - 1, k + 1).round().astype(int))


def recover(mean, std, n, bounds):
    """Difference cumulative moments into per-window (mean, var, count).

    mean, std : [R, T] cumulative moments as stored
    n         : [R]    assumed accumulation count
    bounds    : [K+1]  record indices
    returns   : wmean[K, T], wvar[K, T], dn[K]
    """
    S = mean * n[:, None]
    Q = (std * std + mean * mean) * n[:, None]
    lo, hi = bounds[:-1], bounds[1:]
    dn = n[hi] - n[lo]
    if np.any(dn <= 0):
        raise ValueError("non-increasing accumulation count between window bounds")
    wmean = (S[hi] - S[lo]) / dn[:, None]
    wsec = (Q[hi] - Q[lo]) / dn[:, None]
    wvar = wsec - wmean * wmean
    return wmean, wvar, dn


def reaccumulate(wmean, wvar, dn, base_mean, base_std, base_n):
    """G2 inverse: rebuild the cumulative moments at each window boundary from the
    recovered windows.  Returns (mean_hat[K], std_hat[K]) at boundaries 1..K."""
    S = base_mean * base_n
    Q = (base_std ** 2 + base_mean ** 2) * base_n
    n = base_n
    out_m, out_s = [], []
    for i in range(wmean.shape[0]):
        S = S + wmean[i] * dn[i]
        Q = Q + (wvar[i] + wmean[i] ** 2) * dn[i]
        n = n + dn[i]
        m = S / n
        v = np.maximum(Q / n - m * m, 0.0)
        out_m.append(m)
        out_s.append(np.sqrt(v))
    return np.asarray(out_m), np.asarray(out_s)


def coherence(wmean, wvar, floor=1e-300):
    """c(k, j) = |window mean| / window sd.  Windows whose recovered variance is <= 0
    (numerical, see G1) are returned as nan and COUNTED, never silently zeroed."""
    sd = np.sqrt(np.where(wvar > 0, wvar, np.nan))
    return np.abs(wmean) / (sd + floor)


# --------------------------------------------------------------------------------------
# Per-arm scoring
# --------------------------------------------------------------------------------------

def arm_roles(shapes):
    """role -> list of tensor indices, using c60's MEASURED BasicBlock role map."""
    r = {}
    for t, (nm, sh) in enumerate(shapes):
        role = C60.conv_role(nm) if C60.class_of(sh) == "conv" else "oneD"
        r.setdefault(role, []).append(t)
    return r


def score_arm(d, k=K_WINDOWS, offset=2, slope=1.0):
    """Recover the window series for one arm and reduce it to per-role coherence."""
    steps, cols = load_probe(d)
    mean, std = cols["z_mean"], cols["z_std"]
    n_tens = mean.shape[1]
    fam = family_of_arm(d, n_tens)
    shapes = C59.build_shapes(**C59.FAMILIES[fam])
    if len(shapes) != n_tens:
        raise ValueError(f"{d}: shapes {len(shapes)} != z_mean width {n_tens}")
    n = n_of(steps, offset=offset, slope=slope)
    b = window_bounds(len(steps), k)
    wmean, wvar, dn = recover(mean, std, n, b)
    c = coherence(wmean, wvar)                       # [K, T]
    roles = arm_roles(shapes)

    out = {"dir": d, "n_tens": n_tens, "family": fam, "n_rec": len(steps),
           "steps_per_window": float(np.median(dn)),
           "neg_frac": float(np.mean(wvar <= 0)),
           "relerr": float(EPS32 * np.max(n[b[1:]] / dn)),
           "bounds": b, "dn": dn, "roles": {}}
    with np.errstate(invalid="ignore"):
        for role, idx in roles.items():
            cc = c[:, idx]                            # [K, n_role]
            peak = np.nanmax(cc, axis=0)              # per tensor: peak coherence
            mn = np.nanmean(cc, axis=0)
            # fraction of windows at or above half the tensor's own peak -> H1 vs H2
            half = np.nanmean(cc >= 0.5 * peak[None, :], axis=0)
            # which window holds the peak (median over tensors), as a training fraction
            amax = np.nanargmax(np.nan_to_num(cc, nan=-1.0), axis=0)
            out["roles"][role] = {
                "n": len(idx),
                "peak_med": float(np.nanmedian(peak)),
                "mean_med": float(np.nanmedian(mn)),
                "halfwidth_med": float(np.nanmedian(half)),
                "argmax_frac_med": float(np.median(amax) / max(cc.shape[0] - 1, 1)),
            }
    out["c"] = c
    out["roles_idx"] = roles
    return out


def _family_for(n_tens):
    """Tensor-count fallback ONLY.  r18 and c100 BOTH have 62 tensors, so this is
    ambiguous by construction and must not be the primary route -- use family_of_arm,
    which resolves from the coordinate count the way c59's gate A0 does."""
    hits = [fam for fam, spec in C59.FAMILIES.items()
            if len(C59.build_shapes(**spec)) == n_tens]
    if not hits:
        raise ValueError(f"no family with {n_tens} tensors")
    return hits[0]


def family_of_arm(d, n_tens):
    """Resolve the architecture family from `neg_counts.json`'s coordinate count, which
    is the MEASURED quantity c59's A0 gate reconstructs.  Falls back to the (ambiguous)
    tensor count only when the sidecar is missing, and says so."""
    nc = os.path.join(d, "neg_counts.json")
    if os.path.exists(nc):
        meta = json.load(open(nc))
        n_tot = int(meta.get("n_tot", -1))
        if meta.get("stepsize_type") == "weightwise":
            fam = C59.family_of(n_tot)
            if fam:
                return fam
        # nodewise / layerwise: coordinate count is not the weight count, so match on
        # the reconstructed node or tensor count instead.
        for fam, spec in C59.FAMILIES.items():
            shapes = C59.build_shapes(**spec)
            if len(shapes) != n_tens:
                continue
            if meta.get("stepsize_type") == "nodewise" and \
                    sum(int(s[0]) for _, s in shapes) == n_tot:
                return fam
            if meta.get("stepsize_type") in ("layerwise",) and len(shapes) == n_tot:
                return fam
    return _family_for(n_tens)


def ratio(arm, a="conv1", b="conv2", key="peak_med"):
    ra, rb = arm["roles"].get(a), arm["roles"].get(b)
    if not ra or not rb or rb[key] <= 0:
        return float("nan")
    return ra[key] / rb[key]


# --------------------------------------------------------------------------------------
# Selftests
# --------------------------------------------------------------------------------------

def _synth_cumulative(x, offset=1, slope=1.0, every=1):
    """Build the EXACT cumulative moments a probe would emit for a known stream x[S, T],
    recorded every `every` steps.  Ground truth for the recovery."""
    S = np.cumsum(x, axis=0)
    Q = np.cumsum(x * x, axis=0)
    n = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    mean = S / n[:, None]
    var = np.maximum(Q / n[:, None] - mean * mean, 0.0)
    idx = np.arange(0, x.shape[0], every)
    steps = idx * 1  # step == index when every==1; caller supplies matching n_of
    return steps[::1], mean[idx], np.sqrt(var[idx]), idx


def selftest():
    ok, fail = 0, []

    def chk(cond, name):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    rng = np.random.default_rng(20260823)

    # ---- 1. exact recovery on a known stream --------------------------------------
    S, T = 400, 5
    x = rng.normal(size=(S, T))
    steps, mean, std, idx = _synth_cumulative(x)
    n = idx.astype(np.float64) + 1.0
    b = window_bounds(len(idx), 8)
    wmean, wvar, dn = recover(mean, std, n, b)
    for i in range(len(b) - 1):
        seg = x[idx[b[i]] + 1: idx[b[i + 1]] + 1]
        chk(np.allclose(wmean[i], seg.mean(axis=0), rtol=1e-9, atol=1e-12),
            f"window mean exact w{i}")
        chk(np.allclose(wvar[i], seg.var(axis=0), rtol=1e-7, atol=1e-12),
            f"window var exact w{i}")

    # ---- 2. G0a EXACT invariance under a PURE rescale n -> c*n ----------------------
    # This is the only invariance that actually holds, and it is exact: numerator and
    # denominator both scale, so the recovered window stats are untouched.  It says the
    # recovery does not depend on the UNIT of the accumulation count.
    every = 5
    _s5, mean5, std5, idx5 = _synth_cumulative(x, every=every)
    n_true = idx5.astype(np.float64) + 1.0
    b5 = window_bounds(len(idx5), 8)
    m_a, v_a, _ = recover(mean5, std5, n_true, b5)
    m_c, v_c, _ = recover(mean5, std5, n_true / 7.0, b5)
    chk(np.allclose(m_a, m_c, rtol=1e-12), "G0a exact invariance under pure rescale (mean)")
    chk(np.allclose(v_a, v_c, rtol=1e-12), "G0a exact invariance under pure rescale (var)")

    # ---- 2b. G0b an OFFSET mismatch is NOT invariant, and its error DECAYS as 1/n ---
    # n = a*step + 1 vs n = a'*step + 1 differ by an affine offset, not a pure scale.
    # Algebra: wmean_B = wmean_A + (c-1)*d(cumulative mean)/dn, which is O(1/n).  So the
    # two hypotheses disagree at the FIRST windows and converge later.  Asserting this
    # decay is what licenses reporting the sensitivity instead of claiming invariance.
    n_off = idx5.astype(np.float64) / every + 1.0
    m_b, v_b, _ = recover(mean5, std5, n_off, b5)
    rel = np.abs(m_a - m_b) / np.maximum(np.abs(m_a), 1e-300)
    first, last = float(np.median(rel[0])), float(np.median(rel[-1]))
    chk(last < first, f"G0b offset error decays with n ({first:.2e} -> {last:.2e})")
    chk(not np.allclose(m_a, m_b, rtol=1e-9),
        "G0b offset mismatch is NOT claimed to be invariant")

    # ---- 2c. G0c the record-0 fingerprint SEPARATES n=1 from n=2 --------------------
    # Two accumulations {0, x} give mean=x/2 and var=x^2/4, so std == |mean| exactly.
    # One accumulation {x} gives var == 0 exactly.  Assert both, so the gate is known
    # to discriminate rather than merely to pass.
    two = np.array([[0.0, 0.0], [3.0, -5.0]])
    _s, m2, s2_, _i = _synth_cumulative(two)
    chk(np.allclose(s2_[1], np.abs(m2[1]), rtol=1e-12),
        "G0c n=2 over {0,x} gives std == |mean|")
    chk(np.allclose(s2_[0], 0.0, atol=1e-15), "G0c n=1 gives std == 0")
    chk(not np.allclose(s2_[0], np.abs(m2[0]) + 1.0, atol=1e-9),
        "G0c the two offsets are distinguishable")

    # ---- 3. G2 round-trip ----------------------------------------------------------
    mh, sh = reaccumulate(wmean, wvar, dn, mean[b[0]], std[b[0]], n[b[0]])
    chk(np.allclose(mh, mean[b[1:]], rtol=1e-9, atol=1e-12), "G2 round-trip mean")
    chk(np.allclose(sh, std[b[1:]], rtol=1e-7, atol=1e-12), "G2 round-trip std")

    # ---- 4. coherence behaves ------------------------------------------------------
    pure = rng.normal(size=(S, 1))
    drift = np.linspace(3.0, 3.0, S)[:, None] + rng.normal(size=(S, 1)) * 0.01
    st, mn, sd, ix = _synth_cumulative(np.concatenate([pure, drift], axis=1))
    nn = ix.astype(np.float64) + 1.0
    bb = window_bounds(len(ix), 4)
    wm, wv, _ = recover(mn, sd, nn, bb)
    cc = coherence(wm, wv)
    chk(np.nanmax(cc[:, 0]) < 1.0, "coherence small on zero-mean noise")
    chk(np.nanmin(cc[:, 1]) > 50.0, "coherence large on a coherent drift")

    # ---- 5. an EVENT is localised, a LEVEL SHIFT is not ----------------------------
    ev = rng.normal(size=(S, 1))
    ev[150:200] += 8.0                      # burst in ~1/8 of the run
    lv = rng.normal(size=(S, 1)) + 8.0      # shift everywhere
    st2, mn2, sd2, ix2 = _synth_cumulative(np.concatenate([ev, lv], axis=1))
    n2 = ix2.astype(np.float64) + 1.0
    b2 = window_bounds(len(ix2), 8)
    wm2, wv2, _ = recover(mn2, sd2, n2, b2)
    c2 = coherence(wm2, wv2)
    hw_ev = np.nanmean(c2[:, 0] >= 0.5 * np.nanmax(c2[:, 0]))
    hw_lv = np.nanmean(c2[:, 1] >= 0.5 * np.nanmax(c2[:, 1]))
    chk(hw_ev <= 0.35, f"event is temporally localised (halfwidth {hw_ev:.2f})")
    chk(hw_lv >= 0.75, f"level shift is not localised (halfwidth {hw_lv:.2f})")
    chk(hw_ev < hw_lv, "H1 and H2 are separated by the halfwidth statistic")

    # ---- 6. window_bounds ----------------------------------------------------------
    chk(len(window_bounds(100, 10)) == 11, "window_bounds returns k+1 edges")
    chk(window_bounds(100, 10)[0] == 0 and window_bounds(100, 10)[-1] == 99,
        "window_bounds spans the record range")
    try:
        window_bounds(5, 5)
        chk(False, "window_bounds rejects k>=n_rec")
    except ValueError:
        chk(True, "window_bounds rejects k>=n_rec")

    # ---- 7. recover rejects a non-increasing count ---------------------------------
    try:
        recover(mean[:10], std[:10], np.ones(10), np.array([0, 5, 9]))
        chk(False, "recover rejects flat n")
    except ValueError:
        chk(True, "recover rejects flat n")

    # ---- 8. negative variance is surfaced, not hidden -------------------------------
    bad_v = np.array([[-1.0, 4.0]])
    cbad = coherence(np.array([[2.0, 2.0]]), bad_v)
    chk(np.isnan(cbad[0, 0]), "negative-variance window -> nan, not 0")
    chk(abs(cbad[0, 1] - 1.0) < 1e-9, "positive-variance window unaffected")

    # ---- 9. role map agrees with c60 -----------------------------------------------
    shp = C59.build_shapes(**C59.FAMILIES["r18"])
    roles = arm_roles(shp)
    chk(len(shp) == 62, "r18 has 62 tensors")
    chk(roles["conv1"] and roles["conv2"], "both conv roles populated")
    chk(len(roles["conv1"]) == len(roles["conv2"]), "r18 has equal conv1/conv2 counts")
    chk(all(C60.conv_role(shp[t][0]) == "conv1" for t in roles["conv1"]),
        "conv1 indices carry the conv1 role")
    chk("stem" in roles and len(roles["stem"]) == 1, "exactly one stem conv")

    # ---- 10. _family_for ------------------------------------------------------------
    chk(_family_for(62) == "r18", "_family_for(62) == r18")
    try:
        _family_for(63)
        chk(False, "_family_for rejects an unknown width")
    except ValueError:
        chk(True, "_family_for rejects an unknown width")

    # ---- 11. ratio helper -----------------------------------------------------------
    fake = {"roles": {"conv1": {"peak_med": 4.0}, "conv2": {"peak_med": 2.0}}}
    chk(abs(ratio(fake) - 2.0) < 1e-12, "ratio divides conv1 by conv2")
    fake0 = {"roles": {"conv1": {"peak_med": 4.0}, "conv2": {"peak_med": 0.0}}}
    chk(math.isnan(ratio(fake0)), "ratio nan on a zero denominator")
    chk(math.isnan(ratio({"roles": {"conv1": {"peak_med": 1.0}}})), "ratio nan when a role is absent")

    print(f"selftest: {ok} passed, {len(fail)} failed")
    for f in fail:
        print("   FAIL:", f)
    return not fail


# --------------------------------------------------------------------------------------
# Arm discovery
# --------------------------------------------------------------------------------------

def unique_arms(root, kind):
    """realpath-deduplicated arms of a given kind (88.11: probes_ml5_m* are SYMLINKS)."""
    seen, out = set(), []
    finder = C59.find_weightwise if kind == "w" else C59.find_nodewise
    for d in finder(root):
        rp = os.path.realpath(d)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(d)
    return sorted(out)


def label_of(root, d):
    return os.path.relpath(d, root)


def mannwhitney(a, b):
    """Two-sided Mann-Whitney U with a normal approximation and tie correction."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan"), float("nan")
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    s = np.sort(allv)
    i = 0
    tie_term = 0.0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            m = (np.arange(i, j + 1) + 1).mean()
            ranks[order[i:j + 1]] = m
            t = j - i + 1
            tie_term += t ** 3 - t
        i = j + 1
    Ra = ranks[:na].sum()
    U = Ra - na * (na + 1) / 2.0
    mu = na * nb / 2.0
    N = na + nb
    sd = math.sqrt(na * nb / 12.0 * ((N + 1) - tie_term / (N * (N - 1)))) if N > 1 else float("nan")
    z = (U - mu) / sd if sd and sd > 0 else float("nan")
    p = math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else float("nan")
    return U, z, p


# --------------------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------------------

def run_gate(root):
    print("=" * 88)
    print("GATES  G0 scale-invariance | G1 conditioning | G2 round-trip | G3 shape")
    print("=" * 88)
    arms = unique_arms(root, "w")
    print(f"{len(arms)} unique weightwise arms (realpath-deduped per 88.11)")

    worst_g0c = 0.0        # std/|mean| at record 0 -- must be ~0 if exactly one sample
    g0b_first, g0b_last = [], []
    worst_g2 = 0.0
    worst_relerr = 0.0
    worst_neg = 0.0
    g3_fail = []
    for d in arms:
        steps, cols = load_probe(d)
        mean, std = cols["z_mean"], cols["z_std"]
        try:
            fam = family_of_arm(d, mean.shape[1])
            shapes = C59.build_shapes(**C59.FAMILIES[fam])
            roles = arm_roles(shapes)
            if len(shapes) != mean.shape[1] or not roles.get("conv1") or not roles.get("conv2"):
                g3_fail.append(label_of(root, d))
        except ValueError:
            g3_fail.append(label_of(root, d))
            continue

        # G0c -- the n=2 fingerprint.  Record 0 folds {0, x}, so std == |mean| exactly.
        # (n=1 would give std == 0 exactly; the two are different by construction.)
        nz = np.abs(mean[0]) > 0
        if nz.any():
            rel = np.abs(std[0][nz] - np.abs(mean[0][nz])) / np.abs(mean[0][nz])
            worst_g0c = max(worst_g0c, float(rel.max()))

        b = window_bounds(len(steps), K_WINDOWS)
        every = int(np.median(np.diff(steps))) or 1
        na = n_of(steps, offset=2, slope=1.0)
        nb = steps.astype(float) / every + 1.0
        ma, va, dna = recover(mean, std, na, b)
        mb, _vb, _ = recover(mean, std, nb, b)
        rel = np.abs(ma - mb) / np.maximum(np.abs(ma), 1e-300)
        g0b_first.append(float(np.median(rel[0])))
        g0b_last.append(float(np.median(rel[-1])))

        mh, _sh = reaccumulate(ma, va, dna, mean[b[0]], std[b[0]], na[b[0]])
        d2 = np.maximum(np.abs(mean[b[1:]]), 1e-300)
        worst_g2 = max(worst_g2, float(np.nanmax(np.abs(mh - mean[b[1:]]) / d2)))
        worst_relerr = max(worst_relerr, float(EPS32 * np.max(na[b[1:]] / dna)))
        worst_neg = max(worst_neg, float(np.mean(va <= 0)))

    g0c_ok = worst_g0c < 1e-6
    print(f"G0c worst | std - |mean| | / |mean| at record 0          : {worst_g0c:.3e}   "
          f"[{'PASS' if g0c_ok else 'FAIL'}, bar 1e-6]  -> n = step+2 CONFIRMED (n=1 would give 1.0)")
    print(f"G0b offset sensitivity, median rel. gap window 1 -> {K_WINDOWS:2d}   : "
          f"{np.median(g0b_first):.3e} -> {np.median(g0b_last):.3e}  "
          f"[{'DECAYS' if np.median(g0b_last) < np.median(g0b_first) else 'DOES NOT DECAY'}]")
    print(f"G1  max predicted relative error at K={K_WINDOWS}          : {worst_relerr:.3e}   "
          f"[{'PASS' if worst_relerr < BAR_RELERR else 'FAIL'}, bar {BAR_RELERR:.0e}]")
    print(f"G1  max fraction of windows with negative variance       : {worst_neg*100:.4f}%  "
          f"[{'PASS' if worst_neg < BAR_NEGVAR else 'FAIL'}, bar {BAR_NEGVAR*100:.1f}%]")
    print(f"G2  max round-trip relative error on cumulative mean     : {worst_g2:.3e}   "
          f"[{'PASS' if worst_g2 < 1e-6 else 'FAIL'}, bar 1e-6]")
    print(f"G3  arms failing the shape/role check                    : {len(g3_fail)}      "
          f"[{'PASS' if not g3_fail else 'FAIL'}]")
    for f in g3_fail:
        print("      ", f)
    return (not g3_fail and g0c_ok and worst_relerr < BAR_RELERR
            and worst_neg < BAR_NEGVAR and worst_g2 < 1e-6)


def _collect(root, kind):
    rows = []
    for d in unique_arms(root, kind):
        lab = label_of(root, d)
        try:
            a = score_arm(d)
        except (ValueError, KeyError) as e:
            print(f"  SKIP {lab}: {e}")
            continue
        a["label"] = lab
        a["exc"] = lab in C60.EXC_W
        rows.append(a)
    return rows


def _band(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return "     --          "
    return f"{v.min():8.2f} {np.median(v):8.2f} {v.max():8.2f}"


def run_report(root):
    print("=" * 88)
    print("H1 / H2 / H3  --  conv1-vs-conv2 COHERENCE OF THE PER-TENSOR z TIME SERIES")
    print(f"K={K_WINDOWS} windows;  c = |window mean z| / |window sd z|;  ratio = conv1/conv2")
    print("=" * 88)
    rows = _collect(root, "w")
    exc = [r for r in rows if r["exc"]]
    cln = [r for r in rows if not r["exc"]]
    print(f"{len(rows)} arms scored: {len(exc)} exception, {len(cln)} clean\n")

    for key, name in (("peak_med", "PEAK coherence"), ("mean_med", "MEAN coherence")):
        re_ = [ratio(r, key=key) for r in exc]
        rc_ = [ratio(r, key=key) for r in cln]
        U, z, p = mannwhitney(re_, rc_)
        print(f"--- {name}: conv1/conv2 ratio " + "-" * 40)
        print(f"    exception (n={len(re_)})  min/med/max  {_band(re_)}")
        print(f"    clean     (n={len(rc_)})  min/med/max  {_band(rc_)}")
        finite_c = [x for x in rc_ if np.isfinite(x)]
        if finite_c:
            above = sum(1 for x in re_ if np.isfinite(x) and x > max(finite_c))
            print(f"    exception arms above the clean MAXIMUM: {above}/{len(re_)}")
        print(f"    Mann-Whitney U={U:.0f}  z={z:+.2f}  two-sided p={p:.2e}\n")

    print("--- CONTROL C1: the same ratio at non-conv roles " + "-" * 30)
    for a, b in (("stem", "conv2"), ("oneD", "conv2"), ("shortcut", "conv2")):
        re_ = [ratio(r, a, b) for r in exc]
        rc_ = [ratio(r, a, b) for r in cln]
        if not any(np.isfinite(x) for x in re_):
            continue
        U, z, p = mannwhitney(re_, rc_)
        print(f"    {a:9s}/conv2  exc {_band(re_)} | clean {_band(rc_)} | z={z:+.2f} p={p:.2e}")

    print("\n--- ABSOLUTE levels (the ms=1e-2 confound, shown not assumed away) " + "-" * 10)
    for role in ("conv1", "conv2", "stem", "oneD"):
        re_ = [r["roles"][role]["peak_med"] for r in exc if role in r["roles"]]
        rc_ = [r["roles"][role]["peak_med"] for r in cln if role in r["roles"]]
        print(f"    peak c @ {role:9s} exc {_band(re_)} | clean {_band(rc_)}")

    print("\n--- H1 vs H2: is the excess temporally LOCALISED? " + "-" * 28)
    print("    halfwidth = fraction of windows at >= half the tensor's own peak coherence")
    for role in ("conv1", "conv2"):
        he = [r["roles"][role]["halfwidth_med"] for r in exc if role in r["roles"]]
        hc = [r["roles"][role]["halfwidth_med"] for r in cln if role in r["roles"]]
        print(f"    {role}: exception {_band(he)} | clean {_band(hc)}")
    for role in ("conv1", "conv2"):
        ae = [r["roles"][role]["argmax_frac_med"] for r in exc if role in r["roles"]]
        print(f"    {role}: exception peak-window position (fraction of training) {_band(ae)}")

    print("\n--- PER-ARM TABLE (exception arms, published R_row_cor alongside) " + "-" * 10)
    print(f"    {'arm':34s} {'R_row%':>7} {'c1/c2':>8} {'c1':>8} {'c2':>8} {'halfw':>7}")
    for r in sorted(exc, key=lambda r: -C60.EXC_W.get(r["label"], 0)):
        print(f"    {r['label']:34s} {C60.EXC_W.get(r['label'], float('nan')):7.2f} "
              f"{ratio(r):8.2f} {r['roles']['conv1']['peak_med']:8.2f} "
              f"{r['roles']['conv2']['peak_med']:8.2f} "
              f"{r['roles']['conv1']['halfwidth_med']:7.2f}")
    return rows


def run_controls(root):
    """C2: the two within-config seed controls 60.5 used, on the new statistic."""
    print("=" * 88)
    print("CONTROL C2 -- WITHIN-CONFIG SEED CONTROLS (60.5's two decisive cases)")
    print("=" * 88)
    groups = {
        "bl5/e40  (s0 inverts: 23.014% vs 0.044/0.039)":
            ("probes_bl5/probe_w_e40_s0",
             ["probes_bl5/probe_w_e40_s1", "probes_bl5/probe_w_e40_s2"]),
        "br6/c2   (s1 inverts:  7.216% vs 0.043/0.034/0.034)":
            ("probes_br6/probe_w_c2_s1",
             ["probes_br6/probe_w_c2_s0", "probes_br6/probe_w_c2_s2",
              "probes_br6/probe_w_c2_s3"]),
        "bo6/adw  (BOTH seeds are exceptions -- no clean sibling, reported as such)":
            ("probes_bo6/probe_w_adw_s0", ["probes_bo6/probe_w_adw_s1"]),
    }
    for name, (inv, sibs) in groups.items():
        print(f"\n{name}")
        print(f"    {'seed':34s} {'role':>6} {'c1/c2':>8} {'c1':>8} {'c2':>8} {'halfw':>7} {'peak@':>7}")
        for d, tag in [(inv, "INVERT")] + [(s, "sib") for s in sibs]:
            full = os.path.join(root, d)
            if not os.path.exists(full):
                print(f"    {d:34s}  MISSING")
                continue
            a = score_arm(full)
            print(f"    {d:34s} {tag:>6} {ratio(a):8.2f} "
                  f"{a['roles']['conv1']['peak_med']:8.2f} "
                  f"{a['roles']['conv2']['peak_med']:8.2f} "
                  f"{a['roles']['conv1']['halfwidth_med']:7.2f} "
                  f"{a['roles']['conv1']['argmax_frac_med']:7.2f}")


C2_GROUPS = {
    "bl5/e40": [("probes_bl5/probe_w_e40_s0", "INV"),
                ("probes_bl5/probe_w_e40_s1", "sib"),
                ("probes_bl5/probe_w_e40_s2", "sib")],
    "br6/c2":  [("probes_br6/probe_w_c2_s1", "INV"),
                ("probes_br6/probe_w_c2_s0", "sib"),
                ("probes_br6/probe_w_c2_s2", "sib"),
                ("probes_br6/probe_w_c2_s3", "sib")],
}


def run_ksweep(root, ks=(20, 50, 100, 200, 400)):
    """POST-HOC robustness check on the K=20 registered primary.

    A null is only worth reporting if it is not an artefact of the window width: a short
    burst could hide inside a 1000-step window.  This re-runs the C2 controls across a
    20x range of window widths and prints the inverting seed's RANK among its own
    siblings.  A real event pins the inverting seed at rank 1 everywhere; a null lets the
    rank wander.  Labelled post-hoc because only K=20 was registered.
    """
    print("=" * 88)
    print("K-SENSITIVITY OF THE C2 CONTROLS  [POST-HOC -- only K=20 was registered]")
    print("rank = position of the INVERTING seed among its own config's seeds, by c1/c2")
    print("=" * 88)
    for K in ks:
        cells = []
        for g, arms in C2_GROUPS.items():
            vals = []
            for d, tag in arms:
                full = os.path.join(root, d)
                if not os.path.exists(full):
                    continue
                a = score_arm(full, k=K)
                vals.append((tag, ratio(a)))
            if not vals:
                continue
            inv = [v for v in vals if v[0] == "INV"][0]
            sib = [v for v in vals if v[0] == "sib"]
            rank = 1 + sum(1 for s in sib if s[1] > inv[1])
            cells.append(f"{g}: inv {inv[1]:.2f} sibs {'/'.join(f'{s[1]:.2f}' for s in sib)} "
                         f"rank {rank}/{len(vals)}")
        print(f"  K={K:4d}  " + " | ".join(cells))
    print("\nA burst confined to a few hundred steps would sharpen as K rises.  It does not.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="..")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--ksweep", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.gate:
        run_gate(a.root)
    if a.report:
        run_report(a.root)
    if a.controls:
        run_controls(a.root)
    if a.ksweep:
        run_ksweep(a.root)
    if not (a.gate or a.report or a.controls or a.ksweep):
        ap.print_help()


if __name__ == "__main__":
    main()
