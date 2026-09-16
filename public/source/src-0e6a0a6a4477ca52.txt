#!/usr/bin/env python3
"""cvt1 REGISTRATION-TIME DERIVATIONS (CORRECTIONS 227.3).  Descriptive; reads ONLY
cpl2's landed probe records; feeds no bar of cvt1's scorer (its K = 691 and the w*
q01 0.51 are frozen literals there, re-checked by its selftest where reachable).

Why the INJECT factor is a FIXED K and not a per-step match to the live term, and
what DOSE = 0.1 can show.  L_i = b2*m_i + (1-b2)*z_i per tensor (b2 = pt_b2 = 0.9);
indices 1-based PlainNet: 50 = layer4.1.bn2.weight, 47 = layer4.1.bn1.weight.

  D1  K: pooled mean-|L| ratio 50/47 on k01's PINNED records (226.4(3)'s x691), and
      the per-record median |L_50|/|L_47| by phase of k01's trajectory.
  D2  sign agreement of z_50 and z_47 by phase.
  D3  w*: the weight at which 50's weighted term stops flipping k01's shared sign
      (records from epoch 17, where removing 50 flips).
  D4  on HEAD's records: the LIVE |z_50|/|z_47| by phase (what a per-step match would
      use), and the fraction of the complement's records whose Lion sign flips under
      (a) a per-step live match (twin term -> sign(own)*|term_50|, termwise on z and m)
      and (b) the fixed K = 691.
  D5  K*: the smallest factor on the twin that turns an UP-voting HEAD complement DOWN.

  python3 analysis/cvt1_registration_derivations.py $METAOPT_WS/runs/cpl2
"""
import json, os, sys

B50, B47 = 49, 46
PHASES = ((0, 5), (5, 17), (17, 36), (36, 100.01))


def load(d, arm, s):
    return [json.loads(l) for l in open(os.path.join(d, "probe_cpl2-%s-s%d" % (arm, s), "probe.jsonl"))]


def q(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def L(r, i):
    b2 = r.get("pt_b2", 0.9)
    return b2 * r["m_tensor"][i] + (1 - b2) * r["z_tensor"][i]


def main():
    d = sys.argv[1]
    seeds = (75, 76, 77)
    print("D1  K on k01's PINNED records (pooled over seeds as the mean of per-seed means)")
    m50, m47 = [], []
    for s in seeds:
        P = [r for r in load(d, "k01", s) if r["beta"][0] <= -15 + 1e-4]
        a = sum(abs(L(r, B50)) for r in P) / len(P)
        b = sum(abs(L(r, B47)) for r in P) / len(P)
        m50.append(a)
        m47.append(b)
        print("    s%d  %d pinned records  mean|L50|/mean|L47| = %.2f" % (s, len(P), a / b))
    K = (sum(m50) / 3) / (sum(m47) / 3)
    print("    POOLED %.4f  -> K = %d" % (K, round(K)))
    print("    per-record median |L50|/|L47| by epoch phase (pooled seeds), and D2 sign agreement of z50, z47")
    for lo, hi in PHASES:
        rat, agree, n = [], 0, 0
        for s in seeds:
            for r in load(d, "k01", s):
                e = r["step"] / 500.0
                if lo <= e < hi and abs(L(r, B47)) > 0:
                    rat.append(abs(L(r, B50)) / abs(L(r, B47)))
                    agree += (r["z_tensor"][B50] > 0) == (r["z_tensor"][B47] > 0)
                    n += 1
        print("    epochs [%5.1f,%5.1f)  n %4d  median %8.1f  q10 %8.1f  q90 %9.1f   sign(z50)==sign(z47) %.3f"
              % (lo, hi, n, q(rat, .5), q(rat, .1), q(rat, .9), agree / float(n)))
    print("\nD3  w* on k01's records from epoch 17 where removing 50 flips the shared sign")
    for s in seeds:
        ws = []
        for r in load(d, "k01", s):
            if r["step"] < 17 * 500:
                continue
            A = sum(L(r, i) for i in range(53))
            t = L(r, B50)
            if (A > 0) != (A - t > 0):
                ws.append(1 - A / t)
        print("    s%d  n %d  w* q01 %.3f  q10 %.3f  median %.3f  q90 %.3f  max %.3f"
              % (s, len(ws), q(ws, .01), q(ws, .1), q(ws, .5), q(ws, .9), max(ws)))
    print("\nD4  HEAD's records: LIVE |z50|/|z47| and the complement's Lion-sign flip fraction")
    for s in seeds:
        R = load(d, "HEAD", s)
        for lo, hi in ((0, 18), (18, 22), (22, 50), (50, 100.01)):
            sel = [r for r in R if lo <= r["step"] / 500.0 < hi]
            rat, f_live, f_fix, up = [], 0, 0, 0
            for r in sel:
                b2 = r.get("pt_b2", 0.9)
                Ac = b2 * r["mom_pre"][0][0] + (1 - b2) * r["z_agg"][0][0]
                z50, z47 = r["z_tensor"][B50], r["z_tensor"][B47]
                m50_, m47_ = r["m_tensor"][B50], r["m_tensor"][B47]
                t47 = b2 * m47_ + (1 - b2) * z47
                sg = lambda x: (x > 0) - (x < 0)
                live = b2 * sg(m47_) * abs(m50_) + (1 - b2) * sg(z47) * abs(z50)
                A_live = Ac - t47 + live
                A_fix = Ac - t47 + 691.0 * t47
                rat.append(abs(z50) / max(abs(z47), 1e-300))
                f_live += sg(A_live) != sg(Ac)
                f_fix += sg(A_fix) != sg(Ac)
                up += Ac < 0
            print("    s%d epochs [%3d,%3d)  n %3d  live |z50|/|z47| median %8.3f  | UP-voting %.2f | flips: live-match %.3f  fixed-691 %.3f"
                  % (s, lo, hi, len(sel), q(rat, .5), up / float(len(sel)), f_live / float(len(sel)), f_fix / float(len(sel))))
    print("\nD5  K* = 1 + (-A_c)/L47 on UP-voting HEAD complement records where the twin votes DOWN")
    for s in seeds:
        for lo, hi in ((0, 18), (18, 50), (50, 100.01)):
            ks = []
            for r in load(d, "HEAD", s):
                if not (lo <= r["step"] / 500.0 < hi):
                    continue
                b2 = r.get("pt_b2", 0.9)
                Ac = b2 * r["mom_pre"][0][0] + (1 - b2) * r["z_agg"][0][0]
                t = L(r, B47)
                if Ac < 0 and t > 0:
                    ks.append(1 + (-Ac) / t)
            if ks:
                print("    s%d epochs [%3d,%3d)  n %3d  K* median %.1f  q90 %.1f  max %.1f  frac K* <= 691 %.3f"
                      % (s, lo, hi, len(ks), q(ks, .5), q(ks, .9), max(ks), sum(k <= 691 for k in ks) / float(len(ks))))


if __name__ == "__main__":
    main()
