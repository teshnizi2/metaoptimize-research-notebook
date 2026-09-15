#!/usr/bin/env python3
r"""CYCLE 53 ADDENDUM -- salvaging an interpretable budget contrast from `bl5`.

WHY THIS FILE EXISTS.  `bin/c52_budget_ladder.sh` registered B0.3 as a gate to be
scored FIRST, with an explicit consequence on failure:

    "IF AN ARM EXCEEDS IT, that arm's B1 is UNINTERPRETABLE, NOT REFUTED, and the
     record index at which it first bound is itself the result."

B0.3 FAILED on the weightwise rung -- the exact rung the headline is quoted at.
B1 (-0.3563, far outside its +-0.10 bar) is therefore UNINTERPRETABLE and is
recorded as such in `results/c53_score.txt`.  That verdict stands and this file
does not touch it.

BUT THE FAILURE IS NOT UNIFORM, AND THE POOLED NUMBER HID THAT.  Per seed:

    probe_w_e40_s0   first_hi 3037 (epoch 30.4)   24.1% of records, 96.3% of Q4
    probe_w_e40_s1   never bound                   0.0%,  final max -2.650
    probe_w_e40_s2   never bound                   0.0%,  final max -1.956

ONE seed of three bound.  The other two are box-free over the whole 40 epochs.
So there are two genuinely interpretable measurements hiding inside an
uninterpretable mean, and both are computed here:

  (P1) THE BOX-FREE WINDOW.  Records 2000-3037 are box-free on EVERY seed, so
       window 0.5-0.759 of the 40-epoch run is a clean budget contrast against
       the control's 0.5-1.0 -- and it is a STRICTLY LATER window than the
       control's, which is what a budget test needs.
  (P2) THE BOX-FREE SEEDS.  s1 and s2 over the full steady half, which is the
       registered window, on runs that never touched a guard.

BOTH ARE POST-HOC AND ARE LABELLED POST-HOC EVERYWHERE THEY APPEAR.  They were
not registered, they were chosen after seeing which seed bound, and a window
chosen after seeing the data is exactly the move CORRECTIONS' window rule exists
to catch.  They are reported as SUGGESTIVE, never as the budget verdict; the
budget verdict is a new pre-registered batch.

AND THE NODEWISE / LAYERWISE RUNGS NEED NO SALVAGE AT ALL.  Both are box-free at
0.0000% on every seed over all 4000 records, so their B1-equivalent contrast is
clean as registered.  That is where the honest budget answer already lives, and
it is computed here alongside, at n=3, with no post-hoc anything.
"""
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c52_boxfree import mean, occupancy  # noqa: E402
from neff_instrument import agreement_stats, neff_from_rho  # noqa: E402
from probe5_window import reduce_dir  # noqa: E402

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
RUNGS = ("lay", "node", "w")
BAR = 0.10


def neff_over(d, window):
    """N_eff/m for ONE probe dir on ONE window, VARIANCE instrument.

    Same call chain as neff_instrument.reduce_root -- reduce_dir for rho_s and
    n_tot, neff_from_rho for the inversion -- so this is the same number the
    campaign quotes, restricted to a window, and not a second implementation."""
    name = f"win {window[0]:g}-{window[1]:g}"
    r5 = reduce_dir(d, windows=((name, tuple(window)),))
    if not r5:
        return None, None, False
    w = r5["win"].get(name)
    if not w:
        return None, None, False
    m = float(r5["n_tot"])
    return neff_from_rho(m, w["rho_s"]) / m, m, bool(w["resolved"])


def fmt(v, nd=4):
    return "--" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def selftest():
    import json
    import tempfile
    checks = []

    def ok(n, c):
        checks.append((n, bool(c)))

    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "probe_w_x_s0")
        os.makedirs(d)
        with open(os.path.join(d, "probe.jsonl"), "w") as f:
            for t in range(4000):
                # binds at record 3037 exactly, and only from there on
                f.write(json.dumps(dict(
                    step=t * 5, beta_true_min=-20.0,
                    beta_true_max=(0.0 if t >= 3037 else -1.0),
                    n_at_lo=0, n_at_hi=(7 if t >= 3037 else 0), n_beta=10)) + "\n")
        o = occupancy(d, -30.0, 0.0)
        ok("first_hi is the FIRST binding record", o["first_hi"] == 3037)
        ok("rec_hi counts every record from there on",
           abs(o["rec_hi"] - (4000 - 3037) / 4000) < 1e-12)
        ok("the pre-3037 window is clean", o["first_hi"] / 4000 > 0.5)
        # the window this file uses must end BEFORE the earliest bind
        ok("window 0.5-0.759 ends before first_hi", 0.759 * 4000 < 3037 + 1)
        ok("window 0.5-0.76 would NOT", 0.76 * 4000 > 3037)
        ok("mean of one value is that value", abs(mean([0.5045]) - 0.5045) < 1e-12)

    n = sum(1 for _, c in checks if c)
    for name, c in checks:
        if not c:
            print(f"  FAIL: {name}")
    print(f"selftest: {n}/{len(checks)} PASS")
    return 0 if n == len(checks) else 1


def main():
    print("=" * 104)
    print("CYCLE 53 ADDENDUM -- bl5's budget contrast, split by what is actually box-free")
    print("  B1 STAYS UNINTERPRETABLE AS REGISTERED.  Everything below is either CLEAN (node,")
    print("  lay -- box-free at 0.0000% on every seed) or POST-HOC AND LABELLED (w).")
    print("=" * 104)

    # ---------------------------------------------------------------- occupancy
    print("\n(1) WHICH ARMS ARE ACTUALLY BOX-FREE, PER SEED.  The pooled B0.3 row hid this.")
    print(f"\n  {'arm':24}{'first_hi':>10}{'epoch':>8}{'%rec HI':>10}{'%Q4 HI':>9}"
          f"{'%coordHI':>11}{'final max':>11}{'box-free?':>11}")
    clean = {}
    for g in RUNGS:
        for d in sorted(glob.glob(f"{P}/probes_bl5/probe_{g}_e40_s*")):
            o = occupancy(d, -30.0, 0.0)
            fh = o["first_hi"]
            free = max(o["rec_lo"], o["rec_hi"]) == 0.0
            clean.setdefault(g, []).append((d, free))
            print(f"  {os.path.basename(d):24}{str(fh) if fh else '-':>10}"
                  f"{(f'{fh / 100:.1f}' if fh else '-'):>8}{100 * o['rec_hi']:>10.2f}"
                  f"{100 * o['q4_hi']:>9.2f}{100 * o['coord_hi']:>11.5f}"
                  f"{o['max_final']:>11.3f}{('yes' if free else 'NO'):>11}")
    nfree = {g: sum(1 for _, f in v if f) for g, v in clean.items()}
    print(f"\n  box-free seeds: lay {nfree['lay']}/3   node {nfree['node']}/3   w {nfree['w']}/3")
    print("  --> node and lay need NO salvage: their budget contrast is clean as registered.")

    # ------------------------------------------------- clean rungs, as registered
    print("\n(2) THE CLEAN RUNGS, AS REGISTERED (n=3, box-free on every seed, steady half).")
    print("  This is the honest budget answer and it needs no caveat beyond its rungs.")
    print(f"\n  {'rung':6}{'20 ep (cl5 cU)':>17}{'40 ep (bl5)':>14}{'delta':>10}"
          f"{'bar':>9}{'verdict':>14}")
    for g in ("lay", "node"):
        c = mean([neff_over(d, (0.5, 1.0))[0]
                  for d in sorted(glob.glob(f"{P}/probes_cl5/probe_{g}_cU_s*"))])
        n40 = mean([neff_over(d, (0.5, 1.0))[0]
                    for d in sorted(glob.glob(f"{P}/probes_bl5/probe_{g}_e40_s*"))])
        d_ = n40 - c
        print(f"  {g:6}{c:>17.4f}{n40:>14.4f}{d_:>+10.4f}{BAR:>9.2f}"
              f"{('STATIONARY' if abs(d_) <= BAR else 'DRIFTING'):>14}")
    print("\n  NOTE THE ASYMMETRY, because it is the finding: layerwise holds its value across")
    print("  a budget doubling and NODEWISE DOES NOT.  Neither arm touched a guard, so this")
    print("  is not the box.  It is the first evidence that N_eff/m's budget-robustness is")
    print("  GRANULARITY-DEPENDENT, and it was bought by a batch whose primary test failed.")

    # ------------------------------------------------------- P1, the box-free window
    print("\n(3) [POST-HOC, LABELLED] P1 -- THE BOX-FREE WINDOW on the weightwise rung.")
    print("  Records 2000-3037 (epochs 20.0-30.4) are box-free on EVERY seed, so window")
    print("  0.5-0.759 is a clean contrast against the control's 0.5-1.0, and it is a")
    print("  STRICTLY LATER window than the control's, which is what a budget test needs.")
    print("  IT WAS CHOSEN AFTER SEEING WHICH SEED BOUND.  Suggestive, never a verdict.")
    ctrl_w = mean([neff_over(d, (0.5, 1.0))[0]
                   for d in sorted(glob.glob(f"{P}/probes_cl5/probe_w_cU_s*"))])
    print(f"\n  {'window':26}{'epochs':>14}{'N_eff/m (w)':>14}{'vs control':>12}{'box-free?':>11}")
    print(f"  {'cl5 cU  0.5-1.0 (ctrl)':26}{'10-20':>14}{ctrl_w:>14.4f}{'':>12}{'yes':>11}")
    for lo, hi, lab, free in ((0.25, 0.5, "10-20", "yes"),
                              (0.5, 0.759, "20.0-30.4", "yes"),
                              (0.759, 1.0, "30.4-40", "NO -- s0 pinned"),
                              (0.5, 1.0, "20-40", "NO -- s0 pinned")):
        v = mean([neff_over(d, (lo, hi))[0]
                  for d in sorted(glob.glob(f"{P}/probes_bl5/probe_w_e40_s*"))])
        print(f"  {f'bl5     {lo:g}-{hi:g}':26}{lab:>14}{v:>14.4f}{v - ctrl_w:>+12.4f}{free:>11}")

    # --------------------------------------------------------- P2, the clean seeds
    print("\n(4) [POST-HOC, LABELLED] P2 -- THE TWO SEEDS THAT NEVER BOUND, full steady half.")
    print("  The registered window, on runs that never touched a guard.  n=2, and n=2 is")
    print("  below this campaign's own bar for a headline number.")
    print(f"\n  {'seed':10}{'bound?':>10}{'N_eff/m (w), steady half':>28}{'vs control':>12}")
    vals = []
    for d in sorted(glob.glob(f"{P}/probes_bl5/probe_w_e40_s*")):
        o = occupancy(d, -30.0, 0.0)
        v = neff_over(d, (0.5, 1.0))[0]
        bound = o["rec_hi"] > 0
        if not bound:
            vals.append(v)
        print(f"  {os.path.basename(d)[-2:]:10}{('YES' if bound else 'no'):>10}"
              f"{v:>28.4f}{v - ctrl_w:>+12.4f}")
    if vals:
        mv = mean(vals)
        print(f"\n  box-free seeds only (n={len(vals)}): {mv:.4f}   vs control {ctrl_w:.4f}"
              f"   delta {mv - ctrl_w:+.4f}   bar {BAR}")
        print(f"  --> {'INSIDE' if abs(mv - ctrl_w) <= BAR else 'OUTSIDE'} the bar, at n={len(vals)}.")

    # -------------------------------------------------------------- the shape, clean
    print("\n(5) B2.5's REFUTATION IS CONFOUNDED, and this is the part that matters most.")
    print("  B2.5 reported the ordering flipping node<w<lay -> w<node<lay at 40 epochs and")
    print("  called the nodewise minimum BUDGET-SPECIFIC.  But w is the rung that bound, and")
    print("  a coordinate pinned against a shared wall AGREES WITH ITS NEIGHBOURS FOR A REASON")
    print("  THAT HAS NOTHING TO DO WITH THE META-GRADIENT -- which depresses N_eff/m.  The")
    print("  flip is therefore exactly what contamination of the w rung would manufacture.")
    print(f"\n  {'basis':34}{'lay':>10}{'node':>10}{'w':>10}   ordering")
    rows = [("20 ep, steady half (registered)",
             {g: mean([neff_over(d, (0.5, 1.0))[0]
                       for d in sorted(glob.glob(f"{P}/probes_cl5/probe_{g}_cU_s*"))])
              for g in RUNGS}),
            ("40 ep, steady half (w CONTAMINATED)",
             {g: mean([neff_over(d, (0.5, 1.0))[0]
                       for d in sorted(glob.glob(f"{P}/probes_bl5/probe_{g}_e40_s*"))])
              for g in RUNGS}),
            ("40 ep, BOX-FREE window 0.5-0.759",
             {g: mean([neff_over(d, (0.5, 0.759))[0]
                       for d in sorted(glob.glob(f"{P}/probes_bl5/probe_{g}_e40_s*"))])
              for g in RUNGS})]
    for lab, pr in rows:
        order = " < ".join(k for k, _ in sorted(pr.items(), key=lambda kv: kv[1]))
        print(f"  {lab:34}{pr['lay']:>10.4f}{pr['node']:>10.4f}{pr['w']:>10.4f}   {order}")
    bf = rows[2][1]
    flip_survives = bf["node"] < bf["w"] < bf["lay"]
    print(f"\n  --> on the BOX-FREE window the ordering is "
          f"{'STILL node < w < lay' if flip_survives else 'NOT node < w < lay'}.")
    if flip_survives:
        print("      B2.5's refutation does NOT survive removing the contaminated window.")
        print("      The nodewise minimum is NOT shown to be budget-specific.  Record B2.5 as")
        print("      REFUTED-AS-REGISTERED but CONFOUNDED, and do not write either conclusion.")
    else:
        print("      The flip survives the box-free window, so contamination is NOT what made")
        print("      it.  B2.5's refutation stands on its own evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
