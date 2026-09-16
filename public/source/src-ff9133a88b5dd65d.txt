#!/usr/bin/env python3
"""cvt2 REGISTRATION-TIME DERIVATIONS (CORRECTIONS 233).  Descriptive; reads ONLY cvt1's
landed probe records (HEAD, INJECT, k01; seeds 78/79/80); feeds no bar of cvt2's scorer
(the K ladder is a set of frozen literals there).

L_i = b2*m_i + (1-b2)*z_i per tensor (raw, as PROBE_TENSOR logs it; b2 = pt_b2 = 0.9).
PlainNet 1-based: 50 = layer4.1.bn2.weight (HEAD's isolated tensor), 47 = layer4.1.bn1.weight
(the twin).  The complement is group 0 of HEAD/INJECT's [52,1] grouping (every index but 50).
DOWN = the group's Lion direction sum > 0 (beta <- beta - ms*sign(sum)).

  E1  SHARE(K): the twin's mean per-record share of the complement's |L| mass when its
      term is weighted by K, as a function of K, by phase -- on INJECT's own records
      (the K = 691 trajectory; the share is recomputed at every K from the raw terms)
      and on HEAD's records (the unintervened trajectory).  Inverted for the critical
      shares 0.50 and 0.63 (and the share at 691 is printed, cvt1's 0.81-0.83).
  E2  THE CRITICAL SHARE: idx 50's mean share of k01's shared-sum |L| mass, by phase
      (where 0.5-0.63 comes from).
  E3  K*: on HEAD's UP-voting complement records where the twin votes DOWN, the
      smallest K that turns the complement DOWN (K* = 1 + (-A_c)/L_47), by 2-epoch
      bin; and, for each ladder K, the fraction of HEAD's complement records K flips
      DOWN and the first epoch from which the twin at K DOMINATES the complement
      (DOM: |K L_47| > |A_c - L_47| with the same sign, on >= 0.8 of the next 25
      records) -- a counterfactual on HEAD's trajectory, not a prediction of the arm's.
  E4  the same DOM epoch read on INJECT's own records at K = 691 (the registered
      readout definition cvt2's scorer uses, applied to cvt1), and the complement pin.
  E5  K_sustain: on INJECT's OWN records (the collapsed trajectory), where the other 51
      terms' net vote is UP and the twin's is DOWN, the smallest K that still keeps the
      record DOWN (-rest_net / L_47), by phase; and the DOWN fraction at every probe K.
  E6  THE LADDER, derived from E3 and E5 (printed; the scorer freezes it as literals).

  python3 analysis/cvt2_registration_derivations.py $METAOPT_WS/runs/cvt1
"""
import json
import math
import os
import sys

I50, I47 = 49, 46                     # 0-based
SEEDS = (78, 79, 80)
STEPS_PER_EPOCH = 500
PROBE_K = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 152.0, 258.0, 691.0, 2000.0, 5000.0, 10000.0, 30000.0)
DOM_WIN = 25
DOM_FRAC = 0.8


def load(d, arm, s):
    return [json.loads(l) for l in open(os.path.join(d, "probe_cvt1-%s-s%d" % (arm, s), "probe.jsonl")) if l.strip()]


def q(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def Ls(r):
    b2 = r.get("pt_b2", 0.9)
    return [b2 * m + (1 - b2) * z for m, z in zip(r["m_tensor"], r["z_tensor"])]


def ep(r):
    return r["step"] / float(STEPS_PER_EPOCH)


def comp_terms(r):
    """raw complement terms: (L_47 raw, sum of |L_i| over the other 51, net sum of the other 51)."""
    L = Ls(r)
    others = [L[i] for i in range(53) if i not in (I50, I47)]
    return L[I47], math.fsum(abs(x) for x in others), math.fsum(others)


def share_at(recs, K):
    v = []
    for r in recs:
        t, ra, _rn = comp_terms(r)
        den = K * abs(t) + ra
        if den > 0:
            v.append(K * abs(t) / den)
    return sum(v) / len(v) if v else float("nan")


def invert_share(recs, target):
    lo, hi = 1e-3, 1e7
    if share_at(recs, hi) < target:
        return float("inf")
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if share_at(recs, mid) < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def dom_flags(recs, K):
    """record dominated at K: the twin's weighted term alone outweighs the rest's net vote."""
    out = []
    for r in recs:
        t, _ra, rn = comp_terms(r)
        out.append(abs(K * t) > abs(rn) and ((K * t > 0) == (K * t + rn > 0)))
    return out


def dom_epoch(recs, K):
    f = dom_flags(recs, K)
    for j in range(len(f) - DOM_WIN + 1):
        if sum(f[j:j + DOM_WIN]) >= DOM_FRAC * DOM_WIN:
            return ep(recs[j])
    return None


def pin_epoch(recs, k=0):
    for j in range(len(recs)):
        if all(x["beta"][k] <= -15 + 1e-3 for x in recs[j:]):
            return ep(recs[j])
    return None


def main():
    d = sys.argv[1]
    R = dict(((a, s), load(d, a, s)) for a in ("k01", "HEAD", "INJECT") for s in SEEDS)
    phases = ((0, 17), (17, 36), (36, 100.01))

    print("E1  SHARE(K) of the twin in the complement's |L| mass, and K at share 0.50 / 0.63")
    res = {}
    for a in ("INJECT", "HEAD"):
        for lo, hi in phases:
            row = []
            pooled = []
            for s in SEEDS:
                sel = [r for r in R[(a, s)] if lo <= ep(r) < hi]
                pooled += sel
                row.append((s, share_at(sel, 691.0), invert_share(sel, 0.50), invert_share(sel, 0.63)))
            k50, k63 = invert_share(pooled, 0.50), invert_share(pooled, 0.63)
            res[(a, lo)] = (k50, k63)
            print("    %-6s epochs [%3d,%3d)  " % (a, lo, hi) + "  ".join(
                "s%d share@691 %.4f K(0.50) %7.1f K(0.63) %7.1f" % x for x in row)
                  + "   POOLED K(0.50) %.1f K(0.63) %.1f share@2000 %.4f" % (k50, k63, share_at(pooled, 2000.0)))

    print("\nE2  idx 50's mean share of k01's shared-sum |L| mass (the critical-share reference)")
    for lo, hi in phases:
        v = []
        for s in SEEDS:
            sh = []
            for r in R[("k01", s)]:
                if lo <= ep(r) < hi:
                    L = Ls(r)
                    tot = math.fsum(abs(x) for x in L)
                    sh.append(abs(L[I50]) / tot if tot else 0.0)
            v.append(sum(sh) / len(sh))
        print("    k01 epochs [%3d,%3d)  share50 %s" % (lo, hi, " / ".join("%.4f" % x for x in v)))

    print("\nE3  HEAD's complement (counterfactual): K* by 2-epoch bin, and per ladder K the fraction of records"
          " flipped DOWN and the DOM epoch")
    bins = [(e, e + 2) for e in range(0, 30, 2)]
    for s in SEEDS:
        recs = R[("HEAD", s)]
        cells = []
        for lo, hi in bins:
            ks = []
            n = 0
            for r in recs:
                if not (lo <= ep(r) < hi):
                    continue
                n += 1
                b2 = r.get("pt_b2", 0.9)
                Ac = b2 * r["mom_pre"][0][0] + (1 - b2) * r["z_agg"][0][0]
                t = Ls(r)[I47]
                if Ac < 0 and t > 0:
                    ks.append(1 + (-Ac) / t)
            up = len(ks)
            cells.append("%d-%d:%s" % (lo, hi, ("%.0f(%d/%d)" % (q(ks, .5), up, n)) if ks else "-(0/%d)" % n))
        print("    s%d K* median (UP-voting&twin-DOWN / records): %s" % (s, " ".join(cells)))
    for K in PROBE_K:
        parts = []
        for s in SEEDS:
            recs = R[("HEAD", s)]
            flips = {}
            for r in recs:
                e = ep(r)
                ph = "0-18" if e < 18 else ("18-36" if e < 36 else "36-100")
                b2 = r.get("pt_b2", 0.9)
                Ac = b2 * r["mom_pre"][0][0] + (1 - b2) * r["z_agg"][0][0]
                t = Ls(r)[I47]
                AK = Ac - t + K * t
                c = flips.setdefault(ph, [0, 0])
                c[0] += 1
                c[1] += AK > 0
            de = dom_epoch(recs, K)
            parts.append("s%d DOWN 0-18 %.2f 18-36 %.2f 36-100 %.2f DOM %s" % (
                s, flips["0-18"][1] / float(flips["0-18"][0]), flips["18-36"][1] / float(flips["18-36"][0]),
                flips["36-100"][1] / float(flips["36-100"][0]), "never" if de is None else "%.1f" % de))
        print("    K %7.0f  %s" % (K, " | ".join(parts)))

    print("\nE4  on cvt1's OWN records (the registered readout definition): DOM epoch of the twin at the arm's weight, and pins")
    for a, K in (("INJECT", 691.0), ("HEAD", 1.0), ("k01", 1.0)):
        for s in SEEDS:
            recs = R[(a, s)]
            if a == "k01":
                # k01's shared group holds 50 too: report the twin's DOM (it never should) and 50's own
                f50 = []
                for r in recs:
                    L = Ls(r)
                    rest = math.fsum(L) - L[I50]
                    f50.append(abs(L[I50]) > abs(rest) and ((L[I50] > 0) == (L[I50] + rest > 0)))
                d50 = None
                for j in range(len(f50) - DOM_WIN + 1):
                    if sum(f50[j:j + DOM_WIN]) >= DOM_FRAC * DOM_WIN:
                        d50 = ep(recs[j])
                        break
                print("    %-6s s%d  (shared group) DOM of idx 50 at w=1 from epoch %s   pin beta[0] %s"
                      % (a, s, "never" if d50 is None else "%.1f" % d50, pin_epoch(recs, 0)))
                continue
            de = dom_epoch(recs, K)
            print("    %-6s s%d  DOM of the twin at K=%g from epoch %s   pin complement beta[0] %s   pin beta[1] %s"
                  % (a, s, K, "never" if de is None else "%.1f" % de, pin_epoch(recs, 0), pin_epoch(recs, 1)))

    print("\nE5  K_sustain on INJECT's OWN (collapsed) records: rest net UP, twin DOWN -> smallest K keeping the record DOWN")
    ksus_seed_med, ksus_max = [], 0.0
    for s in SEEDS:
        recs = R[("INJECT", s)]
        for lo, hi in ((16, 20), (20, 38), (38, 100.01)):
            sel = [r for r in recs if lo <= ep(r) < hi]
            ks = []
            for r in sel:
                t, _ra, rn = comp_terms(r)
                if t > 0 and rn < 0:
                    ks.append(-rn / t)
            down = []
            for K in PROBE_K:
                down.append("%g:%.2f" % (K, sum((K * comp_terms(r)[0] + comp_terms(r)[2]) > 0 for r in sel) / float(len(sel))))
            print("    s%d epochs [%3d,%3d)  n %3d  rest-UP&twin-DOWN %3d  K_sustain median %.1f q90 %.1f max %.1f | DOWN frac %s"
                  % (s, lo, hi, len(sel), len(ks), q(ks, .5), q(ks, .9), max(ks) if ks else float("nan"), " ".join(down)))
            if lo >= 20:
                ksus_max = max([ksus_max] + ks)
        post = []
        for r in recs:
            if ep(r) >= 38:
                t, _ra, rn = comp_terms(r)
                if t > 0 and rn < 0:
                    post.append(-rn / t)
        ksus_seed_med.append(q(post, .5))

    print("\nE6  THE LADDER")
    k_init = None
    for K in PROBE_K:
        fr_all = []
        for s in SEEDS:
            sel = [r for r in R[("HEAD", s)] if 18 <= ep(r) < 36]
            fr = 0
            for r in sel:
                b2 = r.get("pt_b2", 0.9)
                Ac = b2 * r["mom_pre"][0][0] + (1 - b2) * r["z_agg"][0][0]
                t = Ls(r)[I47]
                fr += (Ac - t + K * t) > 0
            fr_all.append(fr / float(len(sel)))
        if min(fr_all) >= 0.95:
            k_init = K
            print("    K_INIT = smallest probe K turning >= 0.95 of HEAD's epoch-18..36 complement records DOWN on EVERY seed"
                  " = %g (fractions %s)" % (K, " / ".join("%.4f" % x for x in fr_all)))
            break
    lo_med = min(ksus_seed_med)
    print("    K_SUSTAIN  = per-seed median over INJECT's epoch >= 38 records %s; min %.1f; max over every record from epoch 20 %.1f"
          % (" / ".join("%.1f" % x for x in ksus_seed_med), lo_med, ksus_max))
    r1 = k_init * (lo_med / k_init) ** (1.0 / 3.0)
    r2 = k_init * (lo_med / k_init) ** (2.0 / 3.0)
    print("    the INITIATE-BUT-NOT-SUSTAIN interval [K_INIT, min median K_SUSTAIN] = [%g, %.1f], split in three equal LOG parts:" % (k_init, lo_med))
    print("    rung 1  K_INIT * ratio^(1/3)  = %.2f -> %d" % (r1, round(r1)))
    print("    rung 2  K_INIT * ratio^(2/3)  = %.2f -> %d" % (r2, round(r2)))
    print("    rung 3  152 (retained from the brief): = max K_SUSTAIN x %.3f -> above EVERY collapsed record's sustain K" % (152.0 / ksus_max))
    print("    rung 4  691 (cvt1's INJECT, replicated)")
    print("    rung 5  2000 (the brief's top dose): its onset shift vs 691 on HEAD's counterfactual is printed in E3")
    print("    DROPPED 258: sits x%.2f above every collapsed record's sustain K, like 152 and 691 -- no shape information 152 lacks" % (258.0 / ksus_max))

    print("\nSUMMARY (pooled seeds, epochs 17-36): INJECT records K(0.50) %.1f K(0.63) %.1f | HEAD records K(0.50) %.1f K(0.63) %.1f"
          % (res[("INJECT", 17)] + res[("HEAD", 17)]))


if __name__ == "__main__":
    main()
