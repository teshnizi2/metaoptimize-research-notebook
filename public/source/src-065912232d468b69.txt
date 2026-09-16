#!/usr/bin/env python3
"""cgn1_nominate_gn_carriers.py <runsdir> [pinned|postpeak] -- GroupNorm carrier NOMINATION for cgn1
(CORRECTIONS 217).

A HYPOTHESIS FOR A FUTURE BATCH, NOT A RESULT.  Zero GPU.  Nothing is submitted by this file.
The cvg1 / 194.5 method, re-implemented independently of every registered scorer:
  L_i = b2 * m_tensor_i + (1 - b2) * z_tensor_i   (b2 = the record's pt_b2, Lion beta1 0.9),
ranked on the SCALAR arm's records, mean over the 3 seeds:
  (a) each tensor's share of sum_i mean|L_i| and the fraction of records with L_i > 0;
  (b) 187's OPERATIVE statistic: for a candidate set S, the fraction of records on which removing S
      FLIPS the sign of the remainder sum (and on which S DOMINATES the remainder);
  (c) the same tensors' rank and mean|L| under the LAYERWISE arm (the 184 signature: a carrier's
      |L| collapses under layerwise dynamics).
RECORD SET.  `pinned` (default) is 194.5's: the scalar beta at the -15 clamp.  `postpeak` is a
DEVIATION from 194.5, for a batch whose scalar beta never reaches the clamp inside the budget: every
record after the scalar beta's own maximum (the descending, collapse phase).  Whatever is printed
under `postpeak` is NOT the 194.5 statistic and must be labelled so wherever it is quoted.
Reads only <runsdir>/cgn1/PARTITION-MANIFEST.txt and <runsdir>/cgn1/probe_cgn1-*/probe.jsonl.
"""
import json
import os
import sys

R = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "pinned"
assert MODE in ("pinned", "postpeak"), MODE
NT = 62
SEEDS = (60, 61, 62)
names, numel = {}, {}
for ln in open(os.path.join(R, "cgn1", "PARTITION-MANIFEST.txt")):
    p = ln.split()
    if p and p[0] == "TENSOR":
        names[int(p[1])] = p[2]
        numel[int(p[1])] = int(p[3])
assert len(names) == NT


def load(arm, seed):
    return [json.loads(x) for x in open(os.path.join(R, "cgn1", "probe_cgn1-%s-s%d" % (arm, seed),
                                                     "probe.jsonl")) if x.strip()]


def Lvec(r):
    b2 = r.get("pt_b2", 0.9)
    return [b2 * m + (1 - b2) * z for m, z in zip(r["m_tensor"], r["z_tensor"])]


def select(recs):
    if MODE == "pinned":
        return [r for r in recs if min(r["beta"]) <= -14.999]
    k = max(range(len(recs)), key=lambda i: recs[i]["beta"][0])
    return recs[k + 1:]


def shares(arm, use_select):
    sh = [0.0] * NT
    pos = [0.0] * NT
    meanabs = [0.0] * NT
    used = []
    for s in SEEDS:
        recs = load(arm, s)
        use = select(recs) if use_select else recs
        used.append((len(recs), len(use)))
        if not use:
            continue
        mass = [0.0] * NT
        npos = [0] * NT
        for r in use:
            L = Lvec(r)
            for i in range(NT):
                mass[i] += abs(L[i])
                npos[i] += L[i] > 0
        tot = sum(mass)
        for i in range(NT):
            sh[i] += mass[i] / tot / len(SEEDS)
            pos[i] += npos[i] / len(use) / len(SEEDS)
            meanabs[i] += mass[i] / len(use) / len(SEEDS)
    return sh, pos, meanabs, used


print("RECORD SET: %s%s" % (MODE, "" if MODE == "pinned" else
                             "   *** DEVIATION FROM 194.5: post-peak records, NOT pinned records ***"))
for s in SEEDS:
    recs = load("k01", s)
    b = [r["beta"][0] for r in recs]
    k = max(range(len(b)), key=lambda i: b[i])
    print("  k01-s%d: %d records; scalar beta start %.3f, peak %.3f at record %d (step %d), last %.3f, min %.3f;"
          " records at the -15 clamp: %d"
          % (s, len(b), b[0], b[k], k, recs[k]["step"], b[-1], min(b), sum(1 for x in b if x <= -14.999)))

sh, pos, mabs, used = shares("k01", True)
print("SCALAR arm (k01), (records, selected) per seed: %s" % used)
if not any(u[1] for u in used):
    print("NO RECORD SELECTED ON ANY SEED -- no nomination under this record set.")
    sys.exit(3)
order = sorted(range(NT), key=lambda i: -sh[i])
shL, _posL, mabsL, usedL = shares("kL", False)
orderL = sorted(range(NT), key=lambda i: -shL[i])
rankL = dict((i, r + 1) for r, i in enumerate(orderL))
print("rank  idx  name                          numel      share   frac L>0   | kL rank  kL mean|L| / k01 mean|L|")
for r, i in enumerate(order[:12], 1):
    print("%4d %4d  %-28s %9d   %.4f   %.3f      | %5d    %.4g"
          % (r, i + 1, names[i + 1], numel[i + 1], sh[i], pos[i], rankL[i],
             (mabsL[i] / mabs[i]) if mabs[i] else float("nan")))
for i in (46, 49, 52, 55, 58):          # 0-based; = 1-based 47/50/53/56/59, the 512-wide GN scales
    print("  512-wide scale %2d %-28s rank %2d share %.4f  frac L>0 %.3f  | kL rank %d"
          % (i + 1, names[i + 1], order.index(i) + 1, sh[i], pos[i], rankL[i]))

top = [i + 1 for i in order[:3]]
CANDS = [("{50,53,59} ResNet's BN carriers (187)", [50, 53, 59]), ("{50}", [50]), ("{53}", [53]),
         ("{59}", [59]), ("{47,56} the two bn1 512-wide scales", [47, 56]),
         ("{47,50,53,56,59} all five 512-wide scales", [47, 50, 53, 56, 59]),
         ("{top1 %d}" % top[0], top[:1]), ("{top1,top2 %d,%d}" % tuple(top[:2]), top[:2]),
         ("{top1..3 %s}" % top, top), ("{61} linear.weight", [61])]
print("\nremainder-sign FLIP / DOMINATE fractions on the k01 %s records (TOTAL sign first)" % MODE)
acc = dict((lab, [0.0, 0.0]) for lab, _ in CANDS)
nseed = 0
for s in SEEDS:
    sel = select(load("k01", s))
    if not sel:
        print("  s%d: NO SELECTED RECORD" % s)
        continue
    nseed += 1
    down = sum(1 for r in sel if sum(Lvec(r)) > 0) / len(sel)
    print("  s%d: %d records, sum L > 0 (beta DOWN) on %.4f" % (s, len(sel), down))
    for lab, S in CANDS:
        f = d = 0
        for r in sel:
            L = Lvec(r)
            tot = sum(L)
            ss = sum(L[i - 1] for i in S)
            rem = tot - ss
            f += (rem > 0) != (tot > 0)
            d += abs(ss) > abs(rem) and (ss > 0) != (rem > 0)
        acc[lab][0] += f / len(sel)
        acc[lab][1] += d / len(sel)
        print("      %-44s flip %.4f  dom %.4f" % (lab, f / len(sel), d / len(sel)))
print("  MEAN over %d seeds:" % nseed)
for lab, _ in CANDS:
    print("      %-44s flip %.4f  dom %.4f" % (lab, acc[lab][0] / nseed, acc[lab][1] / nseed))
