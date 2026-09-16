#!/usr/bin/env python3
"""cvt2_attack_indep.py <runsdir> [<scorer.py>] [<score.log>] -- independent re-derivation of `cvt2` (CORRECTIONS 236).

Imports NOTHING from analysis/cVT2_injectladder_score.py or any other repo module and uses no regex.
Reads, by string splitting only, every raw cvt2-<arm>-s<seed>-<jobid>.out in <runsdir> (ARGS / ENV /
PROBE_TENSOR / VOTE_W lines, `Epoch` lines, RUN_DONE, Traceback), <runsdir>/cvt2/PARTITION-MANIFEST.txt
(TENSOR lines) and PROVENANCE.txt, and each run's probe dir (probe_tensor.json, block_sizes.json, probe.jsonl).
plateau5 = mean TEST over the run's own epochs 95..99; TRAIN5 the same on the Train column.  The CSV is NOT read.
Bars and branch order are RE-TYPED from the registration text (233.6); witnesses are built here from K.

Sections:
  [1] completeness, ARGS / ENV / PROBE_TENSOR / VOTE_W per run
  [2] levels: per-run plateau5 + TRAIN5, per-arm mean / sd / range, TEST OLS tail slope 80-99
  [3] sigma (max(frozen prior, in-batch)), SE, every contrast, TEST beside TRAIN; per-seed paired steps along K
  [4] SHAPE and TOP, the gates before them, the conditional stamps, and every bar margin
  [5] THE WEIGHTS BIT: decomposition per group (fsum), and THE APPLIED STEP against the weighted and the
      unweighted Lion directions
  [6] complement beta[0] trajectories: peak and its epoch, first epoch after the peak at or below -8 / -10 / -12 /
      -14, pinned-from, clamp fraction, arm-mean beta at epochs 10..100; isolated beta[1] pin
  [7] the twin's DOM epoch (233.3's definition, re-typed), and by phase the fraction of complement records voting
      DOWN and the twin's weighted share of the complement's |L| mass
  [8] SUSTAIN on each arm's OWN records from epoch 20: on records where the other 51 terms net UP and the twin DOWN,
      the K needed to keep the sum DOWN (-rest / L_47); fraction of those records the arm's own K holds
  [9] arm-mean TEST (TRAIN) trajectory and settle epoch
  [10] level beside the complement's step size in the freeze window: mean exp(beta[0]+15) over epochs 15-30,
      per run, with the rank agreement across the 18 blockwise runs (descriptive)
"""
import hashlib
import json
import math
import os
import sys

PREFIX = "cvt2"
NET = "PlainNet18_c100"
SEEDS = [87, 88, 89]
ARMS = ["k01", "HEAD", "K13", "K33", "K152", "K691", "K2000"]
KOF = {"HEAD": 1, "K13": 13, "K33": 33, "K152": 152, "K691": 691, "K2000": 2000}
HEADSPEC = "sets:1-49,51-53/layer4.1.bn2.weight"
SPEC = dict((a, HEADSPEC) for a in ARMS)
SPEC["k01"] = "scalar"
TWIN, HEAD_IDX = 47, 50
ENV_WANT = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
            "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 "
            "EB_RHO=na EB_LOG=0")
FIXED = {"optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99", "weight-decay-base": "0.1",
         "alg-meta": "Lion", "momentum-param-meta": "0.99", "Lion-beta2-meta": "0.9",
         "weight-decay-meta": "0", "dataset": "CIFAR100", "batch-size": "100", "meta-stepsize": "1e-3",
         "alpha0": "1e-6", "num-epochs": "100", "NN-name": NET}
# 233.6, re-typed
SIGMA_PRIOR = 0.694442846939599
GAP_MIN, K01_MAX, SHAPE_GAP, MATCH, COLLAPSE, NULL, DIVERGED = 20.0, 20.0, 15.0, 5.0, 10.0, 2.0, 5.0
FLOOR_MIN, CEIL_MAX = 15.0, 90.0
BITE_TOL, INFORM, MIN_INF, N_REC = 1e-4, 1e-2, 50, 500
DOM_WIN, DOM_FRAC = 25, 0.8
LADDER = ["K13", "K33", "K152", "K691", "K2000"]
SHAPE_RUNGS = ["K13", "K33", "K152"]
CHAIN = ["HEAD", "K13", "K33", "K152", "K691", "K2000"]
MS = 1e-3
LO, HI = -15.0, -2.3026
SPE = 500


def mean(v):
    return math.fsum(v) / len(v)


def sdev(v):
    m = mean(v)
    return math.sqrt(math.fsum((x - m) ** 2 for x in v) / (len(v) - 1))


def ols(xs, ys):
    xm, ym = mean(xs), mean(ys)
    return math.fsum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / math.fsum((x - xm) ** 2 for x in xs)


def sgn(x):
    return (x > 0) - (x < 0)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(order):
        r[i] = float(k)
    return r


def parse_out(path):
    r = {"args": [], "env": [], "pt": [], "vw": [], "ep": {}, "done": False, "tb": 0, "dup": 0}
    for ln in open(path, errors="replace"):
        s = ln.rstrip("\n")
        if s.startswith("ARGS:"):
            r["args"].append(s)
        elif s.startswith("ENV:"):
            r["env"].append(s)
        elif s.startswith("PROBE_TENSOR:"):
            r["pt"].append(s)
        elif s.startswith("VOTE_W"):
            r["vw"].append(s)
        elif s.startswith("Epoch "):
            p = s.split(",")
            e = int(p[0].split()[1])
            tr = float(p[1].split(":")[1].replace("%", "").strip())
            te = float(p[2].split(":")[1].replace("%", "").strip())
            r["dup"] += e in r["ep"]
            r["ep"][e] = (tr, te)
        elif s.strip() == "RUN_DONE":
            r["done"] = True
        elif s.startswith("Traceback"):
            r["tb"] += 1
    return r


def flags_of(line):
    out = []
    for t in line[len("ARGS:"):].split():
        if t.startswith("--"):
            out.append([t[2:], None])
        elif out:
            out[-1][1] = t if out[-1][1] is None else out[-1][1] + " " + t
    return out


def groups_of(spec, names):
    if spec == "scalar":
        return [list(range(1, len(names) + 1))]
    gs = []
    for g in spec[len("sets:"):].split("/"):
        idx = []
        for tok in g.split(","):
            if tok.isdigit():
                idx.append(int(tok))
            elif "-" in tok and all(x.isdigit() for x in tok.split("-")):
                a, b = tok.split("-")
                idx.extend(range(int(a), int(b) + 1))
            else:
                idx.append(names.index(tok) + 1)
        gs.append(sorted(idx))
    return gs


def weights(arm):
    return {TWIN: float(KOF[arm])} if arm in LADDER else {}


def witness(arm, names):
    if arm not in LADDER:
        return "VOTE_W: off"
    return "VOTE_W: on type=blockwise items=%d:%s:w=%.1f:group=0:groupsize=52" % (TWIN, names[TWIN - 1], float(KOF[arm]))


def main():
    runsdir = sys.argv[1]
    scorer = sys.argv[2] if len(sys.argv) > 2 else None
    slog = sys.argv[3] if len(sys.argv) > 3 else None
    bad = []

    def viol(msg):
        bad.append(msg)
        print("  VIOLATION " + msg)

    base = os.path.join(runsdir, PREFIX)
    names, numel = [], []
    for ln in open(os.path.join(base, "PARTITION-MANIFEST.txt")):
        p = ln.split()
        if p and p[0] == "TENSOR":
            names.append(p[2])
            numel.append(int(p[3]))
    print("[0] manifest: %d tensors, %d params; idx50 %s; idx47 %s"
          % (len(names), sum(numel), names[HEAD_IDX - 1], names[TWIN - 1]))
    if len(names) != 53 or names[HEAD_IDX - 1] != "layer4.1.bn2.weight" or names[TWIN - 1] != "layer4.1.bn1.weight":
        viol("manifest names")
    prov = {}
    for ln in open(os.path.join(base, "PROVENANCE.txt")):
        p = ln.split(None, 1)
        if len(p) == 2:
            prov.setdefault(p[0], p[1].strip())
    print("    PROVENANCE MODE %s  HF %s…  BUILD %s…  RUNNER %s…  SCORER %s…" % (
        prov.get("MODE"), prov.get("HF_SHA256", "")[:8], prov.get("BUILD_NETWORK_SHA256", "")[:8],
        prov.get("RUNNER_SHA256", "")[:8], prov.get("SCORER_SHA256", "")[:8]))
    if scorer:
        s_sha = sha(scorer)
        print("    scorer file sha256 %s == PROVENANCE: %s" % (s_sha[:16], s_sha == prov.get("SCORER_SHA256")))
        if s_sha != prov.get("SCORER_SHA256"):
            viol("scorer sha")

    # ---------------------------------------------------------------- [1]
    print("\n[1] completeness and header lines (witnesses built from K and the manifest names)")
    files = {}
    for fn in sorted(os.listdir(runsdir)):
        if not (fn.startswith(PREFIX + "-") and fn.endswith(".out")):
            continue
        stem = fn[:-4].split("-")
        if len(stem) != 4 or stem[1] not in ARMS:
            continue
        files.setdefault((stem[1], int(stem[2][1:])), []).append((int(stem[3]), fn))
    R = {}
    for a in ARMS:
        wit = witness(a, names)
        for s in SEEDS:
            lst = files.get((a, s), [])
            if len(lst) != 1:
                viol("%s-s%d has %d .out files" % (a, s, len(lst)))
                continue
            jid, fn = lst[0]
            r = parse_out(os.path.join(runsdir, fn))
            r["jid"] = jid
            R[(a, s)] = r
            ok_ep = sorted(r["ep"]) == list(range(100)) and r["dup"] == 0
            f = flags_of(r["args"][0]) if len(r["args"]) == 1 else []
            keys = [k for k, _ in f]
            fd = dict(f)
            want = dict(FIXED)
            want.update({"stepsize-groups": SPEC[a], "seed": str(s), "run-name": "%s-%s-s%d" % (PREFIX, a, s)})
            wrong = [k for k, v in want.items() if fd.get(k) != v]
            rep = sorted(set(k for k in keys if keys.count(k) > 1))
            env = [" ".join(t for t in e.split() if not t.startswith("PROBE_DIR=")) for e in r["env"]]
            ptw = "PROBE_TENSOR: on every=100 type=%s tensors=53" % ("scalar" if SPEC[a] == "scalar" else "blockwise")
            pt_ok = len(r["pt"]) == 1 and r["pt"][0].startswith(ptw)
            line_ok = (ok_ep and r["done"] and r["tb"] == 0 and len(r["args"]) == 1 and not wrong and not rep
                       and env == [ENV_WANT] and pt_ok and r["vw"] == [wit])
            print("  %-4s %-5s s%d job %d  epochs %d  RUN_DONE %s  tb %d  ARGS wrong %s rep %s  ENV %s  PT %s  VOTE_W %s"
                  % ("OK" if line_ok else "BAD", a, s, jid, len(r["ep"]), r["done"], r["tb"], wrong or "-", rep or "-",
                     env == [ENV_WANT], pt_ok, r["vw"] == [wit]))
            if not line_ok:
                viol("%s-s%d header/completeness" % (a, s))

    # ---------------------------------------------------------------- [2]
    print("\n[2] levels (plateau5 = mean TEST epochs 95..99; TRAIN5 beside)")
    P, T = {}, {}
    for a in ARMS:
        v, t, sl = [], [], []
        for s in SEEDS:
            ep = R[(a, s)]["ep"]
            v.append(mean([ep[e][1] for e in range(95, 100)]))
            t.append(mean([ep[e][0] for e in range(95, 100)]))
            xs = list(range(80, 100))
            sl.append(ols(xs, [ep[e][1] for e in xs]))
        P[a], T[a] = v, t
        print("  %-5s TEST %.4f (sd %.4f, range %.4f)  TRAIN %.4f  seeds %s  train %s  slope80-99 %s"
              % (a, mean(v), sdev(v), max(v) - min(v), mean(t),
                 " / ".join("%.4f" % x for x in v), " / ".join("%.4f" % x for x in t),
                 " / ".join("%+.3f" % x for x in sl)))
    M = dict((a, mean(P[a])) for a in ARMS)
    TM = dict((a, mean(T[a])) for a in ARMS)

    # ---------------------------------------------------------------- [3]
    print("\n[3] sigma and contrasts (within batch)")
    ss = math.fsum((x - M[a]) ** 2 for a in ARMS for x in P[a])
    sig_in = math.sqrt(ss / (len(ARMS) * (len(SEEDS) - 1)))
    sig = max(SIGMA_PRIOR, sig_in)
    se = sig * math.sqrt(2.0 / 3.0)
    print("  SIGMA_INBATCH %.6f (df %d)  SIGMA_PRIOR %.6f  -> used %.6f (%s)  SE %.6f"
          % (sig_in, len(ARMS) * 2, SIGMA_PRIOR, sig, "prior" if sig == SIGMA_PRIOR else "in-batch", se))
    gap = M["HEAD"] - M["k01"]
    rows = [("D_HEAD", "HEAD", "k01"), ("P691", "HEAD", "K691")]
    for r_ in SHAPE_RUNGS:
        rows += [("HEAD-" + r_, "HEAD", r_), (r_ + "-K691", r_, "K691")]
    rows += [("TOP", "K691", "K2000")]
    for i in range(len(CHAIN) - 1):
        rows.append(("step %s->%s" % (CHAIN[i], CHAIN[i + 1]), CHAIN[i], CHAIN[i + 1]))
    rows += [("D_" + a, a, "k01") for a in LADDER]
    for lab, x, y in rows:
        d, dt = M[x] - M[y], TM[x] - TM[y]
        extra = ("   SHARE_KEPT %.4f" % (d / gap)) if lab.startswith("D_K") else ""
        print("  %-18s TEST %+9.4f pp = %+8.2f SE   TRAIN %+9.4f pp%s" % (lab, d, d / se, dt, extra))
    for i in range(len(CHAIN) - 1):
        x, y = CHAIN[i], CHAIN[i + 1]
        print("  per-seed paired %-5s - %-5s  %s" % (x, y, " / ".join("%+.4f" % (P[x][j] - P[y][j]) for j in range(3))))
    print("  level per log10(K) step (descriptive): %s" % "  ".join(
        "%s->%s %.3f pp/decade" % (CHAIN[i], CHAIN[i + 1], (M[CHAIN[i]] - M[CHAIN[i + 1]]) /
                                   (math.log10(KOF[CHAIN[i + 1]]) - math.log10(KOF[CHAIN[i]])))
        for i in range(len(CHAIN) - 1)))

    # ---------------------------------------------------------------- [4]
    print("\n[4] SHAPE and TOP, re-typed from 233.6")
    hi = max(M.values())
    div = [a for a in ARMS if max(P[a]) - min(P[a]) > DIVERGED]

    def state(a):
        if M["HEAD"] - M[a] <= MATCH:
            return "AT-HEAD"
        if abs(M[a] - M["K691"]) <= MATCH:
            return "AT-PLATEAU"
        return "BETWEEN"

    shape = top = knee = None
    if not (FLOOR_MIN <= hi <= CEIL_MAX):
        shape = "HARNESS-UNSOUND"
    elif div:
        shape = "UNRESOLVED-DIVERGED"
    elif M["k01"] > K01_MAX:
        shape = "SCALAR-NOT-COLLAPSED"
    elif gap < GAP_MIN:
        shape = "POSITIVE-CONTROL-FAILED"
    elif M["HEAD"] - M["K691"] < SHAPE_GAP:
        shape = "REPLICATE-FAILED"
    elif any(M[a] - M["HEAD"] > MATCH for a in LADDER):
        shape = "LADDER-ABOVE-HEAD"
    elif any(M[CHAIN[i + 1]] - M[CHAIN[i]] > MATCH for i in range(len(CHAIN) - 1)):
        shape = "NON-MONOTONE"
    else:
        sts = [state(a) for a in SHAPE_RUNGS]
        if "BETWEEN" in sts:
            shape = "GRADED"
        elif all(x == "AT-PLATEAU" for x in sts):
            shape = "KNEE-BELOW-LADDER"
        else:
            shape = "THRESHOLD"
            nh = sum(1 for x in sts if x == "AT-HEAD")
            b = (1, 13, 33, 152, 691)
            knee = "KNEE-IN-(%d,%d]" % (b[nh], b[nh + 1])
        tv = M["K691"] - M["K2000"]
        top = "TOP-FLAT" if abs(tv) <= MATCH else ("TOP-DROPS" if tv >= COLLAPSE else "TOP-ATTENUATED")
    st = ["POSITIVE-CONTROL-REPRODUCES" if gap >= GAP_MIN else "POSITIVE-CONTROL-FAILED"]
    st += ["%s-%s" % (a, state(a)) for a in SHAPE_RUNGS]
    if knee:
        st.append(knee)
    if M["K2000"] - M["k01"] <= NULL:
        st.append("TOP-AT-K01")
    if any(M[a] - M["k01"] <= NULL for a in ["HEAD"] + LADDER):
        st.append("FLOOR-READINGS-ARE-BOUNDS")
    tr_mono = all(TM[CHAIN[i + 1]] - TM[CHAIN[i]] <= MATCH for i in range(len(CHAIN) - 1))
    st.append("TRAIN-AGREES" if ((TM["HEAD"] - TM["K691"] > 0) == (M["HEAD"] - M["K691"] > 0)) and tr_mono
              else "TRAIN-DISAGREES")
    st.append("SIGMA-PRIOR-FROZEN" if sig == SIGMA_PRIOR else "SIGMA-INBATCH")
    print("  SHAPE %s | TOP %s | stamps %s | diverged %s" % (shape, top, " | ".join(st), div or "none"))
    mg = [("HEAD-K691 over SHAPE_GAP", M["HEAD"] - M["K691"] - SHAPE_GAP)]
    for a in SHAPE_RUNGS:
        mg.append(("%s: (HEAD-L) - MATCH (>0 = not AT-HEAD)" % a, M["HEAD"] - M[a] - MATCH))
        mg.append(("%s: |L-K691| - MATCH (>0 = not AT-PLATEAU)" % a, abs(M[a] - M["K691"]) - MATCH))
    tv = M["K691"] - M["K2000"]
    mg += [("TOP over MATCH (TOP-FLAT edge)", tv - MATCH), ("TOP under COLLAPSE (TOP-DROPS edge)", COLLAPSE - tv),
           ("worst seed range under DIVERGED", DIVERGED - max(max(P[a]) - min(P[a]) for a in ARMS)),
           ("largest rise along the chain under MATCH (NON-MONOTONE edge)",
            MATCH - max(M[CHAIN[i + 1]] - M[CHAIN[i]] for i in range(len(CHAIN) - 1)))]
    for k, v in mg:
        print("    margin %-58s %+9.4f pp = %+.2f SE" % (k, v, v / se))
    if slog:
        fl = [ln.rstrip("\n") for ln in open(slog) if ln.startswith("FINAL:")]
        print("  scorer log FINAL: %s" % (fl[-1] if fl else "(none)"))
        if fl:
            toks = [t.strip() for t in fl[-1][len("FINAL:"):].split("|")]
            if toks[0] != shape or (top and toks[1] != top):
                viol("shape/top %s/%s != scorer %s/%s" % (shape, top, toks[0], toks[1]))
            miss = [x for x in st if x not in toks]
            if miss:
                viol("stamps not in scorer FINAL: %s" % miss)
            else:
                print("  SHAPE, TOP and all %d re-derived conditional stamps appear in the scorer's FINAL" % len(st))

    # ---------------------------------------------------------------- [5]-[8]
    print("\n[5] THE WEIGHTS BIT -- decomposition (fsum) and the APPLIED step, from the raw probe records")
    traj, vote, sus, win = {}, {}, {}, {}
    for a in ARMS:
        gs = groups_of(SPEC[a], names)
        w = weights(a)
        for s in SEEDS:
            pdir = os.path.join(base, "probe_%s-%s-s%d" % (PREFIX, a, s))
            ptj = json.load(open(os.path.join(pdir, "probe_tensor.json")))
            bsz = json.load(open(os.path.join(pdir, "block_sizes.json")))
            recs = [json.loads(x) for x in open(os.path.join(pdir, "probe.jsonl")) if x.strip()]
            nb_want = [sum(numel[i - 1] for i in g) for g in gs]
            meta_ok = (ptj.get("n_tensors") == 53 and ptj.get("param_numels") == numel
                       and bsz.get("n_b") == nb_want and len(recs) == N_REC)
            worst_z = worst_m = 0.0
            n_inf = n_um = n_dis = n_fw = n_fu = n_mov = n_movw = 0
            b2 = recs[0].get("pt_b2")
            for x in recs:
                zt, mt = x["z_tensor"], x["m_tensor"]
                rec_inf, rec_um = False, True
                for k, g in enumerate(gs):
                    sz = math.fsum(abs(w.get(i, 1.0) * zt[i - 1]) for i in g) + 1e-30
                    sm = math.fsum(abs(w.get(i, 1.0) * mt[i - 1]) for i in g) + 1e-30
                    zw = math.fsum(w.get(i, 1.0) * zt[i - 1] for i in g)
                    mw = math.fsum(w.get(i, 1.0) * mt[i - 1] for i in g)
                    zu = math.fsum(zt[i - 1] for i in g)
                    mu = math.fsum(mt[i - 1] for i in g)
                    za, ma = x["z_agg"][0][k], x["mom_pre"][0][k]
                    worst_z = max(worst_z, abs(zw - za) / sz)
                    worst_m = max(worst_m, abs(mw - ma) / sm)
                    if w and abs(zw - zu) / sz > INFORM:
                        rec_inf = True
                        rec_um = rec_um and abs(zu - za) / sz <= BITE_TOL
                    cw = b2 * ma + (1 - b2) * za
                    cu = b2 * mu + (1 - b2) * zu
                    bp, bn = x["beta_pre"][0][k], x["beta"][k]
                    step = bn - bp
                    if LO + 2 * MS < bp < HI - 2 * MS and abs(step) > 0.5 * MS:
                        n_mov += 1
                        n_movw += sgn(step) == -sgn(cw)
                        if sgn(cw) != sgn(cu):
                            n_dis += 1
                            n_fw += sgn(step) == -sgn(cw)
                            n_fu += sgn(step) == -sgn(cu)
                n_inf += rec_inf
                n_um += rec_inf and rec_um
            ok = meta_ok and worst_z <= BITE_TOL and worst_m <= BITE_TOL and n_movw == n_mov
            if w:
                ok = ok and n_inf >= MIN_INF and n_um == 0 and n_fw == n_dis and n_fu == 0
            else:
                ok = ok and n_inf == 0 and n_dis == 0
            print("  %-4s %-5s s%d recs %d meta %s worst z %.1e m %.1e | informative %d unweighted-match %d"
                  " | moved interior %d follow-weighted %d | sign DISAGREE %d: follow weighted %d, unweighted %d"
                  % ("OK" if ok else "BAD", a, s, len(recs), meta_ok, worst_z, worst_m, n_inf, n_um, n_mov, n_movw,
                     n_dis, n_fw, n_fu))
            if not ok:
                viol("%s-s%d weights-bit check" % (a, s))
            # trajectories
            for k in range(len(gs)):
                seq = [(x["step"] / float(SPE), x["beta"][k]) for x in recs]
                pin = None
                for j in range(len(seq)):
                    if all(b <= LO + 1e-3 for _e, b in seq[j:]):
                        pin = seq[j][0]
                        break
                pk = max(seq, key=lambda eb: eb[1])
                first = dict((thr, next((e for e, b in seq if e > pk[0] and b <= thr), None))
                             for thr in (-8.0, -10.0, -12.0, -14.0))
                traj.setdefault((a, k), []).append({"seq": seq, "pin": pin, "peak": pk, "first": first,
                                                    "clamp": sum(b <= LO + 1e-3 for _e, b in seq) / float(len(seq))})
            # freeze window step size
            win[(a, s)] = mean([math.exp(x["beta"][0] + 15.0) for x in recs if 15 * SPE <= x["step"] < 30 * SPE])
            # the vote in group 0
            g0 = gs[0]
            wt = w.get(TWIN, 1.0)
            flags, eps, ph = [], [], {}
            ks = []
            for x in recs:
                e = x["step"] / float(SPE)
                L = dict((i, b2 * x["m_tensor"][i - 1] + (1 - b2) * x["z_tensor"][i - 1]) for i in g0)
                t = wt * L[TWIN]
                rest = math.fsum(v for i, v in L.items() if i != TWIN)
                tot = t + rest
                flags.append(abs(t) > abs(rest) and ((t > 0) == (tot > 0)))
                eps.append(e)
                mass = abs(t) + math.fsum(abs(v) for i, v in L.items() if i != TWIN) + 1e-30
                phase = ("0-15" if e < 15 else "15-20" if e < 20 else "20-30" if e < 30 else "30-40" if e < 40
                         else "40-100")
                d = ph.setdefault(phase, [0, 0, 0.0])
                d[0] += 1
                d[1] += tot > 0
                d[2] += abs(t) / mass
                if e >= 20 and rest < 0 and L[TWIN] > 0:
                    ks.append(-rest / L[TWIN])
            dom = None
            for j in range(len(flags) - DOM_WIN + 1):
                if sum(flags[j:j + DOM_WIN]) >= DOM_FRAC * DOM_WIN:
                    dom = eps[j]
                    break
            vote[(a, s)] = (dom, ph)
            sus[(a, s)] = ks

    print("\n[6] beta trajectories (probe.jsonl).  after-peak first <= -8 / -10 / -12 / -14; pinned-from = first record after which beta <= -14.999")
    for a in ARMS:
        ng = 1 if SPEC[a] == "scalar" else 2
        for k in range(ng):
            lab = "beta[0]" if ng == 1 else ("complement beta[0]" if k == 0 else "isolated beta[1]")
            T6 = traj[(a, k)]
            if k == 1:
                print("  %-5s %-19s pinned-from %s | peak %s" % (a, lab, " / ".join(
                    "never" if t["pin"] is None else "%.1f" % t["pin"] for t in T6),
                    " / ".join("%.3f" % t["peak"][1] for t in T6)))
                continue
            print("  %-5s %-19s peak %s @ep %s | <=-8 %s | <=-10 %s | <=-12 %s | <=-14 %s | pinned-from %s | clamp %s"
                  % (a, lab, " / ".join("%.3f" % t["peak"][1] for t in T6),
                     " / ".join("%.1f" % t["peak"][0] for t in T6),
                     *[" / ".join("-" if t["first"][thr] is None else "%.1f" % t["first"][thr] for t in T6)
                       for thr in (-8.0, -10.0, -12.0, -14.0)],
                     " / ".join("never" if t["pin"] is None else "%.1f" % t["pin"] for t in T6),
                     " / ".join("%.3f" % t["clamp"] for t in T6)))
            row = []
            for ep in range(10, 101, 10):
                vals = [min(t["seq"], key=lambda eb: abs(eb[0] - ep))[1] for t in T6]
                row.append("%d:%.2f" % (ep, mean(vals)))
            print("        arm-mean beta  " + "  ".join(row))

    print("\n[7] DOM epoch (twin's weighted term outvotes the net of the rest with the total's sign on >= 80%% of the next 25"
          " records), and by phase: DOWN fraction of the complement sum / the twin's weighted |L| share (descriptive)")
    for a in ARMS:
        if a == "k01":
            continue
        for s in SEEDS:
            dom, ph = vote[(a, s)]
            parts = []
            for p in ("0-15", "15-20", "20-30", "30-40", "40-100"):
                n, dn, sh = ph[p]
                parts.append("%s: %.2f/%.3f" % (p, dn / float(n), sh / n))
            print("  %-5s s%d DOM %-5s | %s" % (a, s, "never" if dom is None else "%.1f" % dom, "  ".join(parts)))

    print("\n[8] SUSTAIN on each arm's OWN records from epoch 20: records where the other 51 terms net UP and the twin"
          " votes DOWN; K needed = -rest/L_47 (median, 90th pct, max); fraction of those records held at the arm's own K")
    for a in ARMS:
        if a == "k01":
            continue
        for s in SEEDS:
            ks = sus[(a, s)]
            if not ks:
                print("  %-5s s%d  0 such records" % (a, s))
                continue
            sk = sorted(ks)
            print("  %-5s s%d  records %3d  K-needed median %8.2f  p90 %8.2f  max %9.2f  held at K=%d: %.3f"
                  % (a, s, len(ks), median(ks), sk[int(0.9 * (len(sk) - 1))], sk[-1], KOF[a],
                     sum(1 for v in ks if KOF[a] > v) / float(len(ks))))

    print("\n[9] arm-mean TEST (TRAIN) from the raw .out, settle epoch (arm-mean TEST within 0.5 pp of plateau5 for good)")
    for a in ARMS:
        cells = []
        for e in (9, 14, 17, 19, 21, 24, 29, 39, 49, 69, 99):
            cells.append("%d:%.1f(%.1f)" % (e, mean([R[(a, s)]["ep"][e][1] for s in SEEDS]),
                                            mean([R[(a, s)]["ep"][e][0] for s in SEEDS])))
        am = [mean([R[(a, s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
        settle = next(e for e in range(100) if all(abs(v - M[a]) <= 0.5 for v in am[e:]))
        print("  %-5s %s | settle %d" % (a, "  ".join(cells), settle))

    print("\n[10] level beside the complement's mean step-size multiplier exp(beta[0]+15) over epochs 15-30 (descriptive)")
    xs, ys = [], []
    for a in ARMS:
        vals = [win[(a, s)] for s in SEEDS]
        print("  %-5s  window mean r %s  (arm mean %.1f)  level %s" % (
            a, " / ".join("%.1f" % v for v in vals), mean(vals), " / ".join("%.2f" % v for v in P[a])))
        if a != "k01":
            xs += vals
            ys += P[a]
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    rho = 1 - 6 * math.fsum((p - q) ** 2 for p, q in zip(rx, ry)) / (n * (n * n - 1))
    print("  Spearman rank agreement over the %d blockwise runs (HEAD + ladder): rho %.4f" % (n, rho))
    xl = [math.log(v) for v in xs[3:]]
    print("  ladder only (15 runs): OLS level on ln(window r) slope %.3f pp per e-fold" % ols(xl, ys[3:]))

    print("\nVIOLATIONS %d" % len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
