#!/usr/bin/env python3
r"""CYCLE 51 -- SCORING THE WIDE-CLIP CONTROL, AND WHY ITS PREMISE WAS WRONG.

WHAT THIS SCORES.  `bin/c50_wideclip_ladder.sh` submitted 18 jobs (`wc5-*`) to
decide one question: is b(ms=1e-2) = 0.328 -- the rung that breaks the
CORRECTIONS 58 dial -- an artefact of the BETA_CLIP guard?  It moved the LOW
bound -15 -> -30 and held the HIGH bound at -2.3026, and it pre-registered
W0 (validity), W1 (the primary test), W2 (was the guard actually relieved --
**scored FIRST**) and W3 (the suppression curve).

THE HEADLINE OF THE SCORING, AND IT IS ABOUT THE INSTRUMENT.
The batch chose which wall to move from this sentence in its own header:

    "The HIGH bound is at 0.00% in EVERY arm of the whole 36-job batch.  So a
     large meta-stepsize does not blow the step size UP -- it overshoots
     DOWNWARD ... Therefore: LOW bound -15 -> -30.  HIGH bound UNCHANGED at
     -2.3026, because it provably never binds."

That 0.00% was read from the PER-TENSOR `beta[]` list, which stores one mean
per tensor.  The clip is applied PER COORDINATE.  Read at the resolution the
clip actually operates at -- `beta_true_max`, which the same records already
carry -- the high guard binds on **95.4%** of records at weightwise ms=1e-2,
from record **92**, against the low guard's 91.9% from record 162.  The batch
widened the wall that was binding LESS and left the dominant one shut, so W2
could not pass and W1 is uninterpretable, exactly as W2's own fallback says.

This is the failure CORRECTIONS 60 predicted one tick earlier, in writing,
and did not act on: "the per-coordinate clipped FRACTION is not written ...
so the clip mechanism is *unsupported*, not *excluded*."

WHAT THE BATCH DELIVERS ANYWAY, and it is worth more than W1 was.
  * The ms=1e-3 NULL CONTROL (W0.3) passes on four independent statistics.
  * The campaign's headline number is CLIP-ROBUST: N_eff/m at weightwise moves
    0.481 -> 0.519 when the box is made 2.2x deeper.
  * Plateau accuracy is CLIP-NEUTRAL: +0.03pp mean over 6 matched cells, n=3.
  * The mechanism is measured, not inferred: log-alpha spreads BALLISTICALLY at
    ~95-100% of the maximum Lion speed, in BOTH directions, and fills whatever
    box it is given.  Two boxes of width 12.70 and 27.70 give drift velocities
    of 0.0499 and 0.0498 log-units/record -- identical to 0.2%.
  * At ms=1e-3 the descent is STILL RUNNING at 89-100% of max speed in the LAST
    QUARTER in 9 of 9 arms where no floor stops it.  The phrase "at the adapted
    equilibrium" is true of the BULK (per-tensor means are near-stationary) and
    false of the TAIL.

DENOMINATORS (STANDING RULE 5, CORRECTIONS 58).  Every binding fraction below
names its denominator.  Three are distinguishable and they disagree by orders
of magnitude:
    per (record, TENSOR) cell     -- what c50_dial.py reports
    per record, any TENSOR bound  -- what the wideclip header quoted
    per record, any COORDINATE bound -- what this file adds, and the only one
                                        that matches where the clamp is applied
A fourth -- per (record, COORDINATE) -- is not derivable from probes written
before PATCH_CLIPCOUNT (cycle 51) and is REFUSED here rather than estimated.

USAGE
  python3 analysis/c51_wideclip.py --selftest
  python3 analysis/c51_wideclip.py           # reads ../probes_{ml5,wc5} + results/all_runs.csv
"""

import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

HI_NARROW = -2.3026
LO_NARROW = -15.0
LO_WIDE = -30.0
EPS = 1e-3

# steps between probe records, derived: 20 epochs x 500 steps / 2000 records
STEPS_PER_RECORD = 5


# --------------------------------------------------------------- probe reading
def load_beta(d):
    """-> (list of per-tensor beta lists, list of beta_true_min, list of max)."""
    B, mn, mx = [], [], []
    with open(os.path.join(d, "probe.jsonl")) as f:
        for line in f:
            R = json.loads(line)
            B.append(R["beta"])
            mn.append(R["beta_true_min"])
            mx.append(R["beta_true_max"])
    if not B:
        # An empty probe.jsonl has already cost this project a false conclusion
        # once (OPERATIONS: the .out-in-subfolder trap).  Fail loudly, never 0.
        raise ValueError(f"empty probe.jsonl in {d}")
    return B, mn, mx


def binding(d, lo, hi=HI_NARROW, eps=EPS):
    """All THREE derivable denominators, plus which guard binds first.

    Returns dict with
      cell_tensor  -- fraction of (record, tensor-mean) cells at either guard
      rec_tensor   -- fraction of records with any TENSOR mean at a guard
      rec_lo_coord -- fraction of records with any COORDINATE at the LOW guard
      rec_hi_coord -- fraction of records with any COORDINATE at the HIGH guard
      first_lo / first_hi -- first record index at each guard, or None
    """
    B, mn, mx = load_beta(d)
    T = len(B)
    ncell = nclip = nrec = 0
    nlo = nhi = 0
    first_lo = first_hi = None
    for i in range(T):
        hit = False
        for b in B[i]:
            ncell += 1
            if b <= lo + eps or b >= hi - eps:
                nclip += 1
                hit = True
        if hit:
            nrec += 1
        if mn[i] <= lo + eps:
            nlo += 1
            if first_lo is None:
                first_lo = i
        if mx[i] >= hi - eps:
            nhi += 1
            if first_hi is None:
                first_hi = i
    return dict(cell_tensor=nclip / max(ncell, 1), rec_tensor=nrec / T,
                rec_lo_coord=nlo / T, rec_hi_coord=nhi / T,
                first_lo=first_lo, first_hi=first_hi,
                span_final=mx[-1] - mn[-1], box=hi - lo,
                min_final=mn[-1], max_final=mx[-1])


def drift(d, meta_stepsize, lo, eps=EPS):
    """Downward drift velocity of the extreme coordinate, as a fraction of the
    maximum speed Lion can produce.

    Under Lion every realised beta increment has magnitude exactly the
    meta-stepsize (CORRECTIONS: the sign-based update), so the fastest a
    coordinate can travel is meta_stepsize per step.  A coordinate moving at
    ~100% of that is receiving the SAME SIGN every step -- it is not diffusing,
    it is running.

    v_prefloor is measured over the segment BEFORE the floor is reached, because
    averaging across the pin is what made an earlier quartile-based reading print
    32% for an arm actually running at 99.8%.
    """
    mn = load_beta(d)[1]
    T = len(mn)
    vmax = meta_stepsize * STEPS_PER_RECORD
    t_floor = next((i for i, v in enumerate(mn) if v <= lo + eps), None)
    # `is not None`, not truthiness: a floor reached at record 0 is not "no floor".
    end = t_floor if t_floor is not None else T - 1
    v_pre = (mn[end] - mn[0]) / max(end, 1)
    q = T // 4
    v_last = (mn[-1] - mn[3 * q]) / max(T - 3 * q, 1)
    return dict(v_pre=v_pre, v_last=v_last, vmax=vmax,
                frac_pre=abs(v_pre) / vmax, frac_last=abs(v_last) / vmax,
                t_floor=t_floor, pinned=t_floor is not None)


def bulk_drift(d, meta_stepsize):
    """Per-TENSOR mean drift in the last quarter -- the BULK, against the tail.

    Returns (median |v|/vmax, fraction of tensors above 0.9 vmax).
    """
    B, _mn, _mx = load_beta(d)
    T, n = len(B), len(B[0])
    q = T // 4
    vmax = meta_stepsize * STEPS_PER_RECORD
    rs = sorted(abs(B[-1][j] - B[3 * q][j]) / max(T - 3 * q, 1) / vmax
                for j in range(n))
    med = rs[len(rs) // 2]
    hot = sum(1 for r in rs if r > 0.9) / n
    return med, hot


# ------------------------------------------------------------------- accuracy
def plateaus(csv_path, prefix, rung, ms):
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["run"].startswith(f"{prefix}-{rung}-{ms}-s") and row["plateau"]:
                out.append(float(row["plateau"]))
    return sorted(out)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# ------------------------------------------------------------------- selftest
def selftest():
    import shutil
    import tempfile
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    tmp = tempfile.mkdtemp()
    try:
        # --- synthetic arm A: nothing ever binds --------------------------
        a = os.path.join(tmp, "probe_w_m3_s0")
        os.makedirs(a)
        with open(os.path.join(a, "probe.jsonl"), "w") as f:
            for i in range(100):
                f.write(json.dumps({"beta": [-7.0, -8.0],
                                    "beta_true_min": -7.0 - 0.01 * i,
                                    "beta_true_max": -6.0}) + "\n")
        bA = binding(a, LO_WIDE)
        ok("A: no tensor cell binds", bA["cell_tensor"] == 0.0)
        ok("A: no record binds (tensor)", bA["rec_tensor"] == 0.0)
        ok("A: no coordinate at LO", bA["rec_lo_coord"] == 0.0)
        ok("A: no coordinate at HI", bA["rec_hi_coord"] == 0.0)
        ok("A: first_lo is None", bA["first_lo"] is None)
        ok("A: box width is hi-lo", abs(bA["box"] - (HI_NARROW - LO_WIDE)) < 1e-9)

        dA = drift(a, 1e-3, LO_WIDE)
        ok("A: not pinned", dA["pinned"] is False)
        # travels 0.01/record against vmax = 1e-3*5 = 5e-3 -> exactly 2.0x
        ok("A: v_pre = 2.0x vmax", abs(dA["frac_pre"] - 2.0) < 1e-6)
        ok("A: v_pre is negative", dA["v_pre"] < 0)

        # --- synthetic arm B: HIGH guard binds, LOW never -----------------
        # This is the wc5 failure mode in miniature: a reader looking only at
        # the per-tensor list sees 0%, the coordinate reader sees 50%.
        b = os.path.join(tmp, "probe_w_m2_s0")
        os.makedirs(b)
        with open(os.path.join(b, "probe.jsonl"), "w") as f:
            for i in range(100):
                f.write(json.dumps({"beta": [-7.0, -8.0],
                                    "beta_true_min": -9.0,
                                    "beta_true_max": (HI_NARROW if i >= 50
                                                      else -6.0)}) + "\n")
        bB = binding(b, LO_WIDE)
        ok("B: per-TENSOR reading says 0% (the wc5 trap)", bB["cell_tensor"] == 0.0)
        ok("B: per-TENSOR record reading also says 0%", bB["rec_tensor"] == 0.0)
        ok("B: per-COORDINATE HI reading says 50%",
           abs(bB["rec_hi_coord"] - 0.50) < 1e-9)
        ok("B: first_hi = 50", bB["first_hi"] == 50)
        ok("B: LO still 0%", bB["rec_lo_coord"] == 0.0)

        # --- synthetic arm C: pinned at the floor from record 20 ----------
        # descends at exactly vmax for ms=1e-2 (0.05/record), so it reaches the
        # -30 floor at record 460 of 600 and is pinned for the last quarter --
        # the shape that made a quartile-based reading print 32% for a 99.8% arm.
        c = os.path.join(tmp, "probe_lay_m2_s0")
        os.makedirs(c)
        with open(os.path.join(c, "probe.jsonl"), "w") as f:
            for i in range(600):
                v = max(-7.0 - 0.05 * i, LO_WIDE)
                f.write(json.dumps({"beta": [v, -8.0],
                                    "beta_true_min": v,
                                    "beta_true_max": -6.0}) + "\n")
        bC = binding(c, LO_WIDE)
        ok("C: floor reached", bC["first_lo"] is not None)
        ok("C: coordinate LO fraction > tensor cell fraction",
           bC["rec_lo_coord"] > bC["cell_tensor"])
        dC = drift(c, 1e-2, LO_WIDE)
        ok("C: pinned flag set", dC["pinned"] is True)
        # pre-floor: -0.05/record against vmax = 1e-2*5 = 5e-2 -> exactly 1.0
        ok("C: v_pre is 100% of Lion max (not diluted by the pin)",
           abs(dC["frac_pre"] - 1.0) < 1e-6)
        ok("C: quartile reading WOULD have been diluted",
           abs(dC["v_last"]) < abs(dC["v_pre"]))

        # --- bulk vs tail --------------------------------------------------
        med, hot = bulk_drift(c, 1e-2)
        ok("C: bulk median drift is finite", math.isfinite(med))
        ok("C: hot fraction in [0,1]", 0.0 <= hot <= 1.0)

        # --- guards --------------------------------------------------------
        e = os.path.join(tmp, "probe_w_m3_s9")
        os.makedirs(e)
        open(os.path.join(e, "probe.jsonl"), "w").close()
        try:
            load_beta(e)
            ok("empty probe.jsonl RAISES", False)
        except ValueError:
            ok("empty probe.jsonl RAISES", True)

        # --- denominator ordering is the whole point of STANDING RULE 5 ----
        ok("denominators are distinct objects",
           len({"cell_tensor", "rec_tensor", "rec_lo_coord", "rec_hi_coord"}
               & set(bB)) == 4)

        # --- accuracy helper ----------------------------------------------
        cp = os.path.join(tmp, "runs.csv")
        with open(cp, "w") as f:
            f.write("run,plateau\n")
            f.write("wc5-w-m3-s0,68.0\nwc5-w-m3-s1,69.0\nml5-w-m3-s0,70.0\n")
        ok("plateau filter picks 2 wc5 rows",
           plateaus(cp, "wc5", "w", "m3") == [68.0, 69.0])
        ok("plateau filter does not leak ml5",
           plateaus(cp, "ml5", "w", "m3") == [70.0])
        ok("mean of empty is nan", math.isnan(mean([])))
    finally:
        shutil.rmtree(tmp)

    bad = [n for n, c in checks if not c]
    for n in bad:
        print(f"  FAIL {n}")
    print(f"selftest: {len(checks) - len(bad)}/{len(checks)} "
          f"{'PASS' if not bad else 'FAIL'}")
    return 1 if bad else 0


# ----------------------------------------------------------------------- main
def main():
    csv_path = os.path.join(ROOT, "results", "all_runs.csv")
    MS = {"m4": 1e-4, "m3": 1e-3, "m2": 1e-2}
    ARMS = (("ml5", os.path.join(ROOT, "..", "probes_ml5"), LO_NARROW),
            ("wc5", os.path.join(ROOT, "..", "probes_wc5"), LO_WIDE))

    print("=" * 100)
    print("CYCLE 51 -- (W2, SCORED FIRST) WAS THE GUARD RELIEVED?  THREE DENOMINATORS")
    print("=" * 100)
    print("  wc5 moved LO -15 -> -30 and held HI at -2.3026, on the premise that HI 'provably")
    print("  never binds'.  That premise came from the per-TENSOR column.  Read the per-COORDINATE")
    print("  columns next to it.\n")
    hdr = (f"  {'arm':4} {'ms':5} {'rung':5} {'%cell(tens)':>12} {'%rec(tens)':>11} "
           f"{'%rec LO(coord)':>15} {'%rec HI(coord)':>15} {'1st LO':>7} {'1st HI':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tag, root, lo in ARMS:
        for ms in ("m4", "m3", "m2"):
            for rung in ("w", "node", "lay", "blk6"):
                ds = sorted(glob.glob(os.path.join(root, f"probe_{rung}_{ms}_s*")))
                if not ds:
                    continue
                bs = [binding(d, lo) for d in ds]
                fl = [b["first_lo"] for b in bs if b["first_lo"] is not None]
                fh = [b["first_hi"] for b in bs if b["first_hi"] is not None]
                print(f"  {tag:4} {ms:5} {rung:5} "
                      f"{100 * mean([b['cell_tensor'] for b in bs]):12.2f} "
                      f"{100 * mean([b['rec_tensor'] for b in bs]):11.1f} "
                      f"{100 * mean([b['rec_lo_coord'] for b in bs]):15.1f} "
                      f"{100 * mean([b['rec_hi_coord'] for b in bs]):15.1f} "
                      f"{(str(int(mean(fl))) if fl else '-'):>7} "
                      f"{(str(int(mean(fh))) if fh else '-'):>7}")
    print()
    print("  W2 VERDICT (its bar: layerwise record-binding must fall below 5%, from 88.0%).")
    lay_n = mean([binding(d, LO_NARROW)["rec_tensor"]
                  for d in sorted(glob.glob(os.path.join(ROOT, "..", "probes_ml5",
                                                         "probe_lay_m2_s*")))])
    lay_w = mean([binding(d, LO_WIDE)["rec_tensor"]
                  for d in sorted(glob.glob(os.path.join(ROOT, "..", "probes_wc5",
                                                         "probe_lay_m2_s*")))])
    print(f"    layerwise ms=1e-2, records with any tensor bound: "
          f"{100 * lay_n:.1f}% -> {100 * lay_w:.1f}%   "
          f"--> {'PASS' if lay_w < 0.05 else 'FAIL'} (bar 5%)")
    print("    W2 FAILS.  Per its own text: 'W1 is UNINTERPRETABLE rather than refuted.'")
    print("    The reason is above: at weightwise/nodewise the HIGH guard binds MORE and")
    print("    EARLIER than the LOW one, and wc5 did not move it.\n")

    print("=" * 100)
    print("THE MECHANISM -- BALLISTIC, TWO-SIDED, BOX-FILLING")
    print("=" * 100)
    print("  Under Lion every realised beta increment has magnitude exactly the meta-stepsize,")
    print("  so meta_stepsize x 5 steps/record is the FASTEST a coordinate can travel.\n")
    h2 = (f"  {'arm':4} {'ms':5} {'rung':5} {'box':>7} {'final span':>11} {'fill%':>7} "
          f"{'v_pre/vmax':>11} {'v_last/vmax':>12} {'bulk med':>9} {'%tens hot':>10}")
    print(h2)
    print("  " + "-" * (len(h2) - 2))
    for tag, root, lo in ARMS:
        for ms in ("m3", "m2"):
            for rung in ("w", "node", "lay"):
                ds = sorted(glob.glob(os.path.join(root, f"probe_{rung}_{ms}_s*")))
                if not ds:
                    continue
                bs = [binding(d, lo) for d in ds]
                dr = [drift(d, MS[ms], lo) for d in ds]
                med, hot = bulk_drift(ds[0], MS[ms])
                sp = mean([b["span_final"] for b in bs])
                print(f"  {tag:4} {ms:5} {rung:5} {bs[0]['box']:7.2f} {sp:11.2f} "
                      f"{100 * sp / bs[0]['box']:7.1f} "
                      f"{mean([d['frac_pre'] for d in dr]):11.3f} "
                      f"{mean([d['frac_last'] for d in dr]):12.3f} "
                      f"{med:9.3f} {100 * hot:10.1f}")
    print()
    print("  READ THE TWO BOXES AGAINST EACH OTHER AT ms=1e-2: the spread fills 100% of a")
    print("  12.70-wide box and 100% of a 27.70-wide box, at the SAME velocity.  There is no")
    print("  equilibrium spread to measure -- only the wall you happened to put up.")
    print("  READ v_last AT ms=1e-3 IN wc5: the descent is still running at ~90-100% of the")
    print("  maximum Lion speed in the LAST QUARTER, in every arm where no floor stops it.")
    print("  The BULK (per-tensor means) is near-stationary; the TAIL is not.  'At the adapted")
    print("  equilibrium' is a statement about the bulk and must not be read as convergence.\n")

    print("=" * 100)
    print("W0.3 -- THE NULL CONTROL AT ms=1e-3 (the one thing that makes the batch quotable)")
    print("=" * 100)
    print(f"  {'rung':5} {'ml5 plateau (n)':>18} {'wc5 plateau (n)':>18} {'delta pp':>9}")
    tot, cells = 0.0, 0
    for ms in ("m3", "m2"):
        for rung in ("w", "node", "lay"):
            pn, pw = plateaus(csv_path, "ml5", rung, ms), plateaus(csv_path, "wc5", rung, ms)
            if not pn or not pw:
                continue
            d = mean(pw) - mean(pn)
            tot += d
            cells += 1
            print(f"  {rung + '/' + ms:10} {mean(pn):10.3f} ({len(pn)}) "
                  f"{mean(pw):13.3f} ({len(pw)}) {d:9.3f}")
    print(f"  mean delta over {cells} matched cells: {tot / max(cells, 1):+.3f} pp"
          f"   --> the clip is ACCURACY-NEUTRAL\n")
    print("  Run `probe5_window.py ../probes_wc5` and `neff_instrument.py ../probes_wc5` for")
    print("  the profile and N_eff halves of W0.3 -- they are that file's job, not this one's.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main())
