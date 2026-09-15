#!/usr/bin/env python3
"""c75_frozen_free.py -- the frozen-vs-free agreement comparison across families.

WHY THIS FILE EXISTS.  FINDINGS 74.6 quotes a 7-condition table showing `a_raw` falling
monotonically with the group count m in every family, frozen AND free.  Those numbers
were first produced by an ad-hoc heredoc, which Rule 4 does not allow to stand: a number
that reaches a document must come from a registered reducer with selftests.  This is that
reducer.  It computes NOTHING new -- it calls `neff_instrument.agreement_stats`, the same
function every published agreement number in this campaign came through -- and it exists
so 74.6 is reproducible rather than retyped.

WHAT IT READS.  Two already-paid-for probe roots, at ZERO GPU cost:
  ../probes_fz3   FROZEN ladder + PROBE5  (bin/c48_frozen_ladder_p5.sh)
  ../probes_ff5   FREE   ladder + PROBE5  (bin/c49_free_family_ladder.sh)
Both cover families r10 / r34 / c100 at rungs lay / node / w (fz3 also carries c100 blk6),
2 seeds each, 20 epochs, box (-15, -2.3026).

WHAT IT IS AND IS NOT EVIDENCE FOR.

  * IS: that the monotone fall of `a_raw` with m reproduces across two datasets, four
    architectures, frozen and free -- i.e. it is not an artefact of one setup.
  * IS NOT: evidence of correlation structure.  **The fall is PARTLY MECHANICAL** -- a
    coarse coordinate's meta-gradient is a sum over many fine ones, and the sign of a sum
    is more consistent than the sign of a summand, so SOME monotone fall is arithmetic.
    This module therefore reports the curve and REFUSES to call it a finding on its own.
  * NO INDEPENDENCE NULL IS DERIVED HERE.  Deriving one on the fly is the error
    CORRECTIONS 26 records, and the "53.1 % of 11.17M meta-gradients agree in sign"
    sentence is NOT reproduced by this module and MUST NOT BE WRITTEN from it.
  * n=2 at 20 epochs.  Cited for DIRECTION only, never as an effect size.

USAGE
  python3 analysis/c75_frozen_free.py --selftest
  python3 analysis/c75_frozen_free.py
"""
import argparse
import glob
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from neff_instrument import agreement_stats          # noqa: E402
from probe5_window import parse_dirname              # noqa: E402
import probe5_window as p5w                          # noqa: E402

ROOTS = (("frozen", os.path.join(REPO, "..", "probes_fz3")),
         ("free", os.path.join(REPO, "..", "probes_ff5")))
RUNG_ORDER = ("blk6", "lay", "node", "w")
WINDOW = (0.5, 1.0)                 # the campaign's DEFAULT steady half, unchanged


def is_monotone_decreasing(pairs):
    """pairs: [(m, a_raw)].  True iff a_raw strictly falls as m rises."""
    s = sorted(pairs)
    return all(s[i][1] > s[i + 1][1] for i in range(len(s) - 1))


def reduce_root(root):
    """-> {family: {rung: dict(m, a_raw, a_deb, pbar, bias, bias_share, neff_m, n)}}"""
    acc = {}
    for d in sorted(glob.glob(os.path.join(root, "probe_*"))):
        if not os.path.isdir(d):
            continue
        fam, rung, _seed = parse_dirname(os.path.basename(d))
        if rung not in RUNG_ORDER:
            continue
        ag = agreement_stats(d, window=WINDOW)
        if ag is None:
            continue
        w = p5w.reduce_dir(d, windows=(("steady .5-1", WINDOW),))
        m = float(w["n_tot"]) if w else float("nan")
        rho = w["win"]["steady .5-1"]["rho_s"] if w else None
        neff_m = (1.0 / (1.0 + (m - 1.0) * rho)) if rho is not None else float("nan")
        acc.setdefault(fam, {}).setdefault(rung, []).append((ag, m, neff_m))
    out = {}
    for fam, rungs in acc.items():
        out[fam] = {}
        for rung, vs in rungs.items():
            out[fam][rung] = dict(
                n=len(vs), m=int(vs[0][1]),
                pbar=statistics.mean([x[0]["pbar"] for x in vs]),
                bias=statistics.mean([x[0]["bias"] for x in vs]),
                a_raw=statistics.mean([x[0]["a_raw"] for x in vs]),
                a_deb=statistics.mean([x[0]["a_deb"] for x in vs]),
                bias_share=statistics.mean([x[0]["bias_share"] for x in vs]),
                neff_m=statistics.mean([x[2] for x in vs]))
    return out


def selftest():
    n = p = 0

    def ck(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("    %-62s %s" % (name, "ok" if cond else "FAIL"))

    print("c75_frozen_free selftest")
    ck("window is the campaign DEFAULT steady half", WINDOW == (0.5, 1.0))
    ck("rung order is coarse -> fine", RUNG_ORDER == ("blk6", "lay", "node", "w"))

    # monotonicity helper, both directions and the edge cases
    ck("falling series is monotone",
       is_monotone_decreasing([(6, .66), (62, .62), (14600, .53), (11e6, .50)]))
    ck("rising series is NOT monotone",
       not is_monotone_decreasing([(6, .50), (62, .62)]))
    ck("a tie is NOT strictly monotone",
       not is_monotone_decreasing([(6, .60), (62, .60)]))
    ck("unsorted input is sorted by m first",
       is_monotone_decreasing([(11e6, .50), (6, .66), (62, .62)]))
    ck("single point is trivially monotone", is_monotone_decreasing([(6, .66)]))
    ck("empty is trivially monotone", is_monotone_decreasing([]))

    # the reducer must be the SHARED one, not a local reimplementation
    import neff_instrument
    ck("agreement_stats comes from neff_instrument",
       agreement_stats is neff_instrument.agreement_stats)

    # the deflations this module is required to carry
    ck("module records the fall is PARTLY MECHANICAL", "PARTLY MECHANICAL" in __doc__)
    ck("module forbids the 53.1% sentence", "MUST NOT BE WRITTEN" in __doc__)
    ck("module records no independence null is derived",
       "NO INDEPENDENCE NULL" in __doc__)
    ck("module records n=2 / direction-only", "DIRECTION only" in __doc__)

    ck("both roots are named", len(ROOTS) == 2)
    ck("fz3 is the frozen root", ROOTS[0][0] == "frozen" and "fz3" in ROOTS[0][1])
    ck("ff5 is the free root", ROOTS[1][0] == "free" and "ff5" in ROOTS[1][1])

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    print("=" * 78)
    print("c75 -- FROZEN vs FREE agreement across families (zero GPU cost)")
    print("steady half %s, agreement_stats, n=2 per cell, 20 ep, box (-15, -2.3026)" % (WINDOW,))
    print("=" * 78)

    n_mono = n_tot = 0
    for label, root in ROOTS:
        if not os.path.isdir(root):
            print("\n%s: root %s ABSENT -- skipped" % (label, root))
            continue
        red = reduce_root(root)
        print("\n--- %s   (%s)" % (label.upper(), os.path.basename(root)))
        print("    %-6s %-5s %12s %8s %9s %9s %9s %10s %9s"
              % ("family", "rung", "m", "pbar", "bias", "a_raw", "a_deb",
                 "bias_share", "Neff/m"))
        for fam in sorted(red):
            pairs = []
            for rung in RUNG_ORDER:
                r = red[fam].get(rung)
                if not r:
                    continue
                pairs.append((r["m"], r["a_raw"]))
                print("    %-6s %-5s %12d %8.5f %+9.5f %9.5f %9.5f %10.3f %9.4f"
                      % (fam, rung, r["m"], r["pbar"], r["bias"], r["a_raw"],
                         r["a_deb"], r["bias_share"], r["neff_m"]))
            mono = is_monotone_decreasing(pairs)
            n_tot += 1
            n_mono += mono
            print("      -> a_raw monotone DECREASING in m: %s   (%d rungs)"
                  % (mono, len(pairs)))

    print("\n" + "=" * 78)
    print("a_raw falls monotonically with m in %d of %d conditions." % (n_mono, n_tot))
    print("**THIS IS PARTLY MECHANICAL AND IS NOT A FINDING ON ITS OWN.**  A coarse")
    print("coordinate's meta-gradient is a SUM over many fine ones, and the sign of a sum")
    print("is more consistent than the sign of a summand, so some monotone fall is")
    print("arithmetic.  No independence null is derived here, and the '53.1%' sentence is")
    print("NOT reproduced by this module.  n=2 at 20 ep: DIRECTION only, never an effect size.")
    print("What is NOT mechanical is the EXPONENT d log N_eff / d log m (FINDINGS 74.4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
