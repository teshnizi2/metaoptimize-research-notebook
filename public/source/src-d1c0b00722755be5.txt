#!/usr/bin/env python3
"""c73 -- bf8's F0 RE-SCORED FROM probe.jsonl, AND THE CEILING BIND CHARACTERISED.

WHY THIS EXISTS (cycle 73)
--------------------------
`analysis/c72_bf8_score.py` reported **F0 pass 0/12, "the batch is VOID"**, on the
grounds of an `n_beta` mismatch, printing `nbeta=None` for all twelve arms.  That
verdict is WRONG, and this file establishes it from the data rather than asserting it.

Two independent facts, each checked below:

  1. `n_beta` is a field of EVERY probe record (`probe.jsonl`), and it is present and
     correct on all twelve bf8 arms.  c72's F0 read it instead from `neg_counts.npy`,
     a file bf8 never wrote, so `nb` was `None` and `None == 11173962` was False.
     F0 therefore measured FILE PRESENCE, not the network.  Re-scored against the
     field it names, **F0 passes 12/12.**

  2. `neg_counts.npy` is missing because the submit script never turned the
     instrument on.  `patches/patch_probe5.py:55` gates the whole writer on
     `os.environ['PROBE5'] == '1'`; `bin/c71_floor_budget.sh:306` exports
     `PROBE=5` (the probe STRIDE) and never `PROBE5=1` (the sign-count INSTRUMENT).
     `bd7`'s script exports `PROBE=5,PROBE5=1,PROBE5_WRITE_EVERY=500` and its dirs
     do contain `neg_counts.npy`.  Two different names one character apart.

  N_eff/m is NOT recoverable from what bf8 wrote: `neff_instrument.reduce_root`
  skips any dir without `neg_counts.json` (line 162), and the per-COORDINATE running
  sign counts it needs are not derivable from the per-STEP `frac_neg` scalar that
  probe.jsonl does carry.  So F0.5 / F1 / F2 are genuinely unscored -- the batch did
  not produce that measurement.  It is neither confirmed nor refuted; it is ABSENT.

WHAT IS RECOVERABLE, AND IS SCORED HERE
---------------------------------------
Everything that lives in probe.jsonl or the CSV.  In particular the guard census,
which is what bf8 was FOR: bd7 fixed LO=-30 and 10/12 arms bound at the FLOOR;
bf8 lowered the floor to -60/-90.  Did that free it, and at what cost at the ceiling?

The ceiling column is reported at BOTH available resolutions, because STANDING RULE 6
says a fraction must be measured where the mechanism acts.  The clamp is applied
per COORDINATE, so `coord_hi` (fraction of (record, coordinate) cells at the guard)
is the honest number and `rec_hi` (fraction of records with ANY coordinate there)
is an upper bound that a single stuck weight can drive to 1.0.  c72 printed only
`rec_hi`.  On a weightwise arm with 11.17M coordinates the gap between the two is
the whole question, so both are printed and never averaged.

Run:  python3 analysis/c73_bf8_ceiling.py --selftest
      python3 analysis/c73_bf8_ceiling.py
"""

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PARENT = os.path.dirname(REPO)
sys.path.insert(0, HERE)

import c52_boxfree as BF            # noqa: E402
import c55_neff_noise as C55        # noqa: E402
import c71_bd7_bo7_score as C71     # noqa: E402

ROOT = os.path.join(PARENT, "probes_bf8")
SCRIPT = os.path.join(REPO, "bin", "c71_floor_budget.sh")
PATCH = os.path.join(REPO, "patches", "patch_probe5.py")

N_RECORDS = 8000                   # 80 ep x 100 records/epoch
RECORDS_PER_EPOCH = 100
NBETA = C71.NBETA                  # w 11,173,962 / node 14,420 / lay 62
FLOORS = {"f60": (-60.0, 2.0), "f90": (-90.0, 2.0)}
RUNGS = ["w", "node", "lay"]

# bd7's floor census, quoted from c72's own registered comparison line so the two
# cannot drift: at LO=-30, 10/12 arms bound, 37-42% of records, 100% of Q4.
BD7_BOUND_ARMS = 10
BD7_ARMS = 12


def arm_dirs(rung, floor):
    return sorted(glob.glob(os.path.join(ROOT, f"probe_{rung}_{floor}_s*")))


def all_arms():
    for rung in RUNGS:
        for fl in FLOORS:
            for d in arm_dirs(rung, fl):
                yield rung, fl, d


# ------------------------------------------------------------------ F0, re-scored

def f0_from_probe():
    """F0 as it was WRITTEN -- n_records, n_beta, beta moved -- read from probe.jsonl.

    `occupancy()` already returns `n_beta` off record 0, so c72 had this value in
    hand inside the very next gate it ran.
    """
    out = []
    for rung, fl, d in all_arms():
        R = BF.records(d)
        nb = R[0].get("n_beta")
        # every record, not just the first: a mid-run change is the thing F0 fears
        stable = all(r.get("n_beta") == nb for r in R)
        lo = [r["beta_true_min"] for r in R]
        hi = [r["beta_true_max"] for r in R]
        out.append(dict(arm=os.path.basename(d), rung=rung, floor=fl,
                        n=len(R), n_ok=(len(R) == N_RECORDS),
                        nbeta=nb, nbeta_ok=(nb == NBETA[rung]), stable=stable,
                        moved=bool((max(hi) - min(lo)) > 1e-9),
                        b0=hi[0], bmax=max(hi), bmin=min(lo)))
    return out


# ------------------------------------------------------------------ the ceiling

def ceiling_census():
    """Per arm: where the box binds, at BOTH resolutions, plus the first-touch epoch."""
    out = []
    for rung, fl, d in all_arms():
        lo, hi = FLOORS[fl]
        o = BF.occupancy(d, lo, hi)
        out.append(dict(arm=os.path.basename(d), rung=rung, floor=fl, lo=lo, hi=hi,
                        rec_lo=o["rec_lo"], rec_hi=o["rec_hi"],
                        q4_lo=o["q4_lo"], q4_hi=o["q4_hi"],
                        coord_lo=o["coord_lo"], coord_hi=o["coord_hi"],
                        first_lo=o["first_lo"], first_hi=o["first_hi"],
                        max_final=o["max_final"], min_final=o["min_final"]))
    return out


def trajectory(d, every=1000):
    """beta_true_max at a coarse grid of epochs -- the shape of the approach."""
    R = BF.records(d)
    pts = []
    for i in range(0, len(R), every):
        pts.append((i // RECORDS_PER_EPOCH, R[i]["beta_true_max"]))
    pts.append((len(R) // RECORDS_PER_EPOCH, R[-1]["beta_true_max"]))
    return pts


# ------------------------------------------------------------------ selftest

def selftest():
    p = n = 0

    def chk(name, cond):
        nonlocal p, n
        n += 1
        if cond:
            p += 1
        else:
            print("  FAIL:", name)

    # -- the DIAGNOSIS is asserted against the two source files, not remembered
    patch = open(PATCH).read()
    chk("the writer is gated on the PROBE5 env var",
        "os.environ.get('PROBE5', '') == '1'" in patch)
    chk("the writer is what emits neg_counts.npy",
        "'neg_counts.npy'" in patch and "'neg_counts.json'" in patch)
    src = open(SCRIPT).read()
    chk("bf8's export sets the probe STRIDE", "PROBE=5," in src)
    chk("bf8's export never sets the INSTRUMENT", "PROBE5=1" not in src)
    chk("so no bf8 dir can hold neg_counts.json",
        not any(os.path.exists(os.path.join(d, "neg_counts.json"))
                for _, _, d in all_arms()))

    # -- and N_eff is genuinely gone, from the instrument's OWN skip condition
    neff = open(os.path.join(HERE, "neff_instrument.py")).read()
    chk("reduce_root skips a dir without neg_counts.json",
        'neg_counts.json' in neff and 'continue' in neff)

    # -- the box is the one c72 registered; this file re-uses it, never re-guesses
    chk("f60 box matches the registered box", C55.box_for("bf8", "f60") == (-60.0, 2.0))
    chk("f90 box matches the registered box", C55.box_for("bf8", "f90") == (-90.0, 2.0))
    chk("FLOORS agrees with the registration",
        FLOORS["f60"] == C55.box_for("bf8", "f60") and
        FLOORS["f90"] == C55.box_for("bf8", "f90"))
    chk("NBETA is the imported published constant", NBETA is C71.NBETA)
    chk("weightwise n_beta is the campaign constant", NBETA["w"] == 11173962)

    # -- corpus
    chk("12 arms on disk", sum(1 for _ in all_arms()) == 12)
    chk("8000 records == 80 epochs", N_RECORDS // RECORDS_PER_EPOCH == 80)

    # -- occupancy's two resolutions really are different quantities
    syn = BF.occupancy(arm_dirs("w", "f60")[0], *FLOORS["f60"])
    chk("coord_hi is available (PATCH_CLIPCOUNT present)", syn["coord_hi"] is not None)
    chk("coord_hi <= rec_hi by construction", syn["coord_hi"] <= syn["rec_hi"] + 1e-12)

    print("selftest: %d/%d passed" % (p, n))
    return p == n


# ------------------------------------------------------------------ report

def main():
    print("=" * 78)
    print("c73 -- bf8 F0 RE-SCORED FROM probe.jsonl, AND THE CEILING BIND")
    print("=" * 78)

    print("\n[A] WHY c72 SAID VOID -- the instrument was never switched on")
    print("     patch_probe5.py gates the writer on   PROBE5 == '1'")
    print("     c71_floor_budget.sh:306 exports       PROBE=5        (the STRIDE)")
    print("     bd7's c54_budget_curve.sh:287 exports PROBE=5,PROBE5=1,PROBE5_WRITE_EVERY=500")
    print("     => bf8 wrote no neg_counts.npy, so c72's F0 compared None to 11173962.")
    print("     F0 as WRITTEN says 'n_beta byte-match'.  n_beta is a probe.jsonl FIELD.")

    print("\n[F0 RE-SCORED] n_records == 8000, n_beta == NBETA[rung] on EVERY record, beta moved")
    rows = f0_from_probe()
    ok = sum(1 for r in rows if r["n_ok"] and r["nbeta_ok"] and r["stable"] and r["moved"])
    print("     pass %d/%d" % (ok, len(rows)))
    print("     %-22s %6s %12s %8s %10s %10s" %
          ("arm", "n", "n_beta", "stable", "beta_min", "beta_max"))
    for r in rows:
        print("     %-22s %6d %12d %8s %10.3f %10.3f"
              % (r["arm"], r["n"], r["nbeta"], r["stable"], r["bmin"], r["bmax"]))
    if ok == len(rows):
        print("     >>> THE NETWORK DID NOT CHANGE.  c72's VOID was a FILE-PRESENCE test")
        print("     >>> wearing an n_beta label.  The batch is VALID.")

    print("\n[B] THE GUARD CENSUS -- bd7 bound %d/%d arms at LO=-30.  Did -60/-90 free it?"
          % (BD7_BOUND_ARMS, BD7_ARMS))
    print("     coord_* is the honest column (STANDING RULE 6): the clamp acts per")
    print("     COORDINATE, and rec_* goes to 1.0 if ONE of 11.17M weights sticks.")
    print("     %-22s %8s %8s %10s %10s %9s %9s"
          % ("arm", "rec_lo%", "rec_hi%", "coord_lo%", "coord_hi%", "1st_lo_ep", "1st_hi_ep"))
    cen = ceiling_census()
    for r in cen:
        fl_ep = "-" if r["first_lo"] is None else "%.1f" % (r["first_lo"] / RECORDS_PER_EPOCH)
        fh_ep = "-" if r["first_hi"] is None else "%.1f" % (r["first_hi"] / RECORDS_PER_EPOCH)
        print("     %-22s %7.2f%% %7.2f%% %9.4f%% %9.4f%% %9s %9s"
              % (r["arm"], 100 * r["rec_lo"], 100 * r["rec_hi"],
                 100 * r["coord_lo"], 100 * r["coord_hi"], fl_ep, fh_ep))

    nlo = sum(1 for r in cen if r["rec_lo"] > 0)
    nhi = sum(1 for r in cen if r["rec_hi"] > 0)
    print("\n     arms touching the FLOOR: %d/12   arms touching the CEILING: %d/12"
          % (nlo, nhi))
    if nlo == 0:
        print("     >>> THE FLOOR IS FULLY FREED.  bd7's %d/%d LO binds go to 0/12 at LO=-60."
              % (BD7_BOUND_ARMS, BD7_ARMS))
        print("     >>> c72's F0.3 printed 'AN ARM BINDS AT LO EVEN AT THIS FLOOR' and")
        print("     >>> pointed at `first_lo`.  first_lo is None on all 12.  The residual")
        print("     >>> binding is at the CEILING, which is the OTHER guard, and is exactly")
        print("     >>> the case STANDING RULE (18) was written to catch.")

    print("\n[C] THE CEILING, ARM BY ARM -- beta_true_max toward HI=+2.0")
    for rung in RUNGS:
        for fl in FLOORS:
            for d in arm_dirs(rung, fl):
                pts = trajectory(d, every=1000)
                s = "  ".join("ep%d %.3f" % (e, v) for e, v in pts)
                print("     %-22s %s" % (os.path.basename(d), s))

    print("\n[D] FLOOR INVARIANCE OF THE GUARD VERDICT -- f60 vs f90, per rung and seed")
    by = {}
    for r in cen:
        key = (r["rung"], r["arm"].rsplit("_", 1)[1])
        by.setdefault(key, {})[r["floor"]] = r
    print("     %-12s %6s %14s %14s  %s" % ("rung", "seed", "f60 coord_hi%", "f90 coord_hi%", "same verdict"))
    for (rung, seed), v in sorted(by.items()):
        if len(v) != 2:
            continue
        a, b = v["f60"], v["f90"]
        same = (a["rec_hi"] > 0) == (b["rec_hi"] > 0)
        print("     %-12s %6s %13.4f%% %13.4f%%  %s"
              % (rung, seed, 100 * a["coord_hi"], 100 * b["coord_hi"], "yes" if same else "NO"))
    print("     >>> if every verdict matches, the -90 floor bought nothing the -60 floor")
    print("     >>> did not already buy, and a THIRD floor is not the next experiment.")

    print("\n[E] WHAT IS STILL MISSING, AND WHY IT NEEDS COMPUTE")
    print("     N_eff/m  -- needs the per-COORDINATE running sign counts in neg_counts.npy.")
    print("                 probe.jsonl carries only the per-STEP scalar frac_neg, which")
    print("                 cannot be un-pooled into per-coordinate counts.  F0.5, F1 and")
    print("                 F2 are ABSENT, not failed: bf8 never took the measurement.")
    print("     F2       -- the registered out-of-sample band prediction (argmin = w) is")
    print("                 UNTESTED.  It is not evidence for or against the span band.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    main()
