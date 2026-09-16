#!/usr/bin/env python3
"""cvt3_attack_indep.py <runsdir> [<scorer.py>] [<score.log>] -- independent re-derivation of `cvt3` (CORRECTIONS 235).

Imports NOTHING from analysis/cVT3_downcoalition_score.py or any other repo module and uses no regex.
Reads, by string splitting only:
  * every raw cvt3-<arm>-s<seed>-<jobid>.out in <runsdir>: ARGS / ENV / PROBE_TENSOR / VOTE_W lines,
    every `Epoch` line, RUN_DONE, Traceback;
  * <runsdir>/cvt3/PARTITION-MANIFEST.txt (TENSOR lines give the 53 names and numels) and PROVENANCE.txt;
  * each run's probe dir: probe_tensor.json, block_sizes.json, probe.jsonl (no accuracy is read there).
plateau5 = mean TEST over the run's own epochs 95..99; TRAIN5 the same on the Train column.  The CSV is NOT read.
The two sets C and K, the bars and the branch order are RE-TYPED from the registration text (232.3, 232.5).
The VOTE_W witness lines are BUILT here from the manifest's names, not copied from the scorer.

Sections:
  [1] completeness, ARGS / ENV / PROBE_TENSOR / VOTE_W per run
  [2] levels: per-run plateau5 + TRAIN5, per-arm mean / sd / range, TEST OLS tail slope 80-99
  [3] sigma (max(frozen prior, in-batch)), SE, the primary and every key contrast, TEST beside TRAIN
  [4] G-RUNAWAY re-derived, then the branch and conditional stamps, re-typed from 232.5; every bar margin
  [5] THE WEIGHTS BIT: (a) z_agg == fsum(w z_tensor), mom_pre == fsum(w m_tensor); (b) the set level (registered
      set vs the 50-only set vs the OTHER arm's set); (c) THE APPLIED STEP follows the weighted Lion direction
      on every interior record, and on records where the weighted and unweighted directions disagree
  [6] beta[0] trajectories: peak, first epoch AFTER the peak at or below -10 / -14, pinned-from epoch, clamp fraction,
      arm-mean beta at epochs 10..100
  [7] the vote in the shared sum by phase: fraction of records voting DOWN (weighted, as run), the silenced
      tensors' share of the RAW |L| mass, and the fraction of records where the silence flipped the sign
  [8] top 6 unsilenced voters from epoch 36, and the DOWN mass left outside the silenced set
  [9] arm-mean TEST (TRAIN) trajectory and the epoch each arm's TEST settles within 0.5 pp of its plateau5
  [10] the registration's runaway prior, re-read on THIS batch's k01, MUTE50, MUTECTL and MUTEDOWN records
  [11] level beside the shared step-size multiplier in epochs 15-30 and 30-40 (descriptive)
"""
import hashlib
import json
import math
import os
import sys

PREFIX = "cvt3"
NET = "PlainNet18_c100"
SEEDS = [84, 85, 86]
ARMS = ["k01", "HEAD", "MUTE50", "MUTEDOWN", "MUTECTL"]
HEADSPEC = "sets:1-49,51-53/layer4.1.bn2.weight"
SPEC = {"k01": "scalar", "HEAD": HEADSPEC, "MUTE50": "scalar", "MUTEDOWN": "scalar", "MUTECTL": "scalar"}
HEAD_IDX = 50
# 232.3, re-typed
C_SET = [47, 44, 41, 38, 35, 32, 29, 26, 23, 20, 17, 14, 40, 46, 43, 11, 37, 8, 5, 49]
K_SET = [48, 45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 19, 13, 10, 12, 7, 9, 6, 4]
SILENCED = {"k01": [], "HEAD": [], "MUTE50": [HEAD_IDX], "MUTEDOWN": sorted([HEAD_IDX] + C_SET),
            "MUTECTL": sorted([HEAD_IDX] + K_SET)}
ENV_WANT = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
            "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 "
            "EB_RHO=na EB_LOG=0")
FIXED = {"optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99", "weight-decay-base": "0.1",
         "alg-meta": "Lion", "momentum-param-meta": "0.99", "Lion-beta2-meta": "0.9",
         "weight-decay-meta": "0", "dataset": "CIFAR100", "batch-size": "100", "meta-stepsize": "1e-3",
         "alpha0": "1e-6", "num-epochs": "100", "NN-name": NET}
# 232.5, re-typed
SIGMA_PRIOR = 0.694442846939599
GAP_MIN, K01_MAX, RESCUE, NULL, MATCH, DIVERGED = 20.0, 20.0, 10.0, 2.0, 5.0, 5.0
FLOOR_MIN, CEIL_MAX = 15.0, 90.0
BITE_TOL, INFORM, MIN_INF, N_REC = 1e-4, 1e-2, 50, 500
CEIL, CEIL_TOL, RUN_FROM_EP, RUN_FRAC, RUN_SEEDS, DRIFT = -2.3026, 0.01, 17, 0.10, 2, -4.0
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


def witness(arm, names):
    if not SILENCED[arm]:
        return "VOTE_W: off"
    return "VOTE_W: on type=scalar items=" + ",".join(
        "%d:%s:w=0.0:group=0:groupsize=53" % (i, names[i - 1]) for i in SILENCED[arm])


def main():
    runsdir = sys.argv[1]
    scorer = sys.argv[2] if len(sys.argv) > 2 else None
    slog = sys.argv[3] if len(sys.argv) > 3 else None
    bad = []

    def viol(msg):
        bad.append(msg)
        print("  VIOLATION " + msg)

    base = os.path.join(runsdir, PREFIX)
    names, numel, cls = [], [], []
    man_c = man_k = None
    for ln in open(os.path.join(base, "PARTITION-MANIFEST.txt")):
        p = ln.split()
        if p and p[0] == "TENSOR":
            names.append(p[2])
            numel.append(int(p[3]))
            cls.append(p[4])
        elif p and p[0] == "COALITION_RANKED":
            man_c = [int(x) for x in p[1:]]
        elif p and p[0] == "CONTROL_PAIRED":
            man_k = [int(x) for x in p[1:]]
    print("[0] manifest: %d tensors, %d params; idx50 %s; C == 232.3 %s; K == 232.3 %s"
          % (len(names), sum(numel), names[HEAD_IDX - 1], man_c == C_SET, man_k == K_SET))
    if len(names) != 53 or names[HEAD_IDX - 1] != "layer4.1.bn2.weight" or man_c != C_SET or man_k != K_SET:
        viol("manifest names / sets")
    cc = [cls[i - 1] for i in C_SET]
    kc = [cls[i - 1] for i in K_SET]
    print("    C classes %s ; K classes %s ; pairwise class-equal %s ; disjoint %s ; 50 in neither %s"
          % (dict((c, cc.count(c)) for c in sorted(set(cc))), dict((c, kc.count(c)) for c in sorted(set(kc))),
             cc == kc, not set(C_SET) & set(K_SET), HEAD_IDX not in C_SET + K_SET))
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
    print("\n[1] completeness and header lines (witnesses built from the manifest names)")
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
            print("  %-4s %-8s s%d job %d  epochs %d  RUN_DONE %s  tb %d  ARGS wrong %s rep %s  ENV %s  PT %s  VOTE_W %s (%d chars)"
                  % ("OK" if line_ok else "BAD", a, s, jid, len(r["ep"]), r["done"], r["tb"], wrong or "-", rep or "-",
                     env == [ENV_WANT], pt_ok, r["vw"] == [wit], len(wit)))
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
        print("  %-8s TEST %.4f (sd %.4f, range %.4f)  TRAIN %.4f  seeds %s  train %s  slope80-99 %s"
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
    C = [("P_COAL", "MUTEDOWN", "MUTECTL"), ("D_DOWN", "MUTEDOWN", "k01"), ("D_CTL", "MUTECTL", "k01"),
         ("D_HEAD", "HEAD", "k01"), ("D_MUTE50", "MUTE50", "k01"), ("G_DOWN", "HEAD", "MUTEDOWN"),
         ("DOWN-MUTE50", "MUTEDOWN", "MUTE50"), ("CTL-MUTE50", "MUTECTL", "MUTE50")]
    D = {}
    for lab, x, y in C:
        d, dt = M[x] - M[y], TM[x] - TM[y]
        D[lab] = (d, dt)
        print("  %-12s %-9s - %-9s TEST %+9.4f pp = %+8.2f SE   TRAIN %+9.4f pp" % (lab, x, y, d, d / se, dt))
    print("  per-seed paired (same seed) MUTEDOWN-MUTECTL: %s ; MUTECTL-k01: %s ; MUTEDOWN-k01: %s" % (
        " / ".join("%+.4f" % (P["MUTEDOWN"][i] - P["MUTECTL"][i]) for i in range(3)),
        " / ".join("%+.4f" % (P["MUTECTL"][i] - P["k01"][i]) for i in range(3)),
        " / ".join("%+.4f" % (P["MUTEDOWN"][i] - P["k01"][i]) for i in range(3))))

    # ---------------------------------------------------------------- probe records (read once)
    REC = {}
    for a in ARMS:
        for s in SEEDS:
            pdir = os.path.join(base, "probe_%s-%s-s%d" % (PREFIX, a, s))
            REC[(a, s)] = (json.load(open(os.path.join(pdir, "probe_tensor.json"))),
                           json.load(open(os.path.join(pdir, "block_sizes.json"))),
                           [json.loads(x) for x in open(os.path.join(pdir, "probe.jsonl")) if x.strip()])

    # ---------------------------------------------------------------- [4]
    print("\n[4] G-RUNAWAY (re-derived from beta[0]), then the branch and stamps re-typed from 232.5")
    rw_seeds = dict((a, 0) for a in ARMS)
    for a in ARMS:
        for s in SEEDS:
            recs = REC[(a, s)][2]
            post = [x for x in recs if x["step"] >= RUN_FROM_EP * SPE]
            nonfin = sum(1 for x in recs if not math.isfinite(x["beta"][0]))
            fin = [x["beta"][0] for x in recs if math.isfinite(x["beta"][0])]
            fc = sum(1 for x in post if math.isfinite(x["beta"][0]) and x["beta"][0] >= CEIL - CEIL_TOL) / float(len(post))
            fd = sum(1 for x in post if math.isfinite(x["beta"][0]) and x["beta"][0] >= DRIFT) / float(len(post))
            run = nonfin > 0 or fc >= RUN_FRAC
            rw_seeds[a] += run
            print("  %-8s s%d  post-17 records %d  at-ceiling %.3f  drift %.3f  nonfinite %d  max beta[0] %.3f  runaway %s"
                  % (a, s, len(post), fc, fd, nonfin, max(fin), run))
    print("  runaway seeds per arm: %s" % rw_seeds)
    hi = max(M.values())
    div = [a for a in ARMS if max(P[a]) - min(P[a]) > DIVERGED]
    dh, dm, dd, dc, pc, gd = (D["D_HEAD"][0], D["D_MUTE50"][0], D["D_DOWN"][0], D["D_CTL"][0], D["P_COAL"][0],
                              D["G_DOWN"][0])
    if not (FLOOR_MIN <= hi <= CEIL_MAX):
        br = "HARNESS-UNSOUND"
    elif rw_seeds["MUTEDOWN"] >= RUN_SEEDS:
        br = "UP-RUNAWAY"
    elif rw_seeds["MUTEDOWN"] >= 1:
        br = "UNRESOLVED-RUNAWAY-SPLIT"
    elif any(rw_seeds[a] >= 1 for a in ARMS if a != "MUTEDOWN"):
        br = "CONTROL-RUNAWAY"
    elif div:
        br = "UNRESOLVED-DIVERGED"
    elif M["k01"] > K01_MAX:
        br = "SCALAR-NOT-COLLAPSED"
    elif dh < GAP_MIN:
        br = "POSITIVE-CONTROL-FAILED"
    elif dm >= RESCUE:
        br = "MUTE50-RESCUES"
    elif dd < -NULL:
        br = "MUTEDOWN-BELOW-K01"
    elif dc >= RESCUE:
        br = "COALITION-OVER-NONSPECIFIC" if pc >= RESCUE else "NONSPECIFIC"
    elif dc > NULL:
        br = "CONTROL-WEAK"
    elif dd >= RESCUE:
        br = "COALITION" if gd <= MATCH else "COALITION-PARTIAL"
    elif dd <= NULL:
        br = "OWN-STEP-NECESSARY"
    else:
        br = "DOWN-WEAK"
    st = ["POSITIVE-CONTROL-REPRODUCES" if dh >= GAP_MIN else "POSITIVE-CONTROL-FAILED",
          "MUTE50-AT-K01" if dm <= NULL else ("MUTE50-WEAK" if dm < RESCUE else "MUTE50-RESCUES")]
    if M["MUTEDOWN"] - M["HEAD"] > MATCH:
        st.append("DOWN-ABOVE-HEAD")
    if dc < -NULL:
        st.append("CTL-BELOW-K01")
    if any(abs(M[a] - M["k01"]) <= NULL for a in ("MUTE50", "MUTEDOWN", "MUTECTL")):
        st.append("FLOOR-READINGS-ARE-BOUNDS")
    st.append("TRAIN-AGREES" if (D["P_COAL"][1] > 0) == (pc > 0) else "TRAIN-DISAGREES")
    st.append("SIGMA-PRIOR-FROZEN" if sig == SIGMA_PRIOR else "SIGMA-INBATCH")
    print("  branch %s   stamps %s   diverged %s" % (br, " | ".join(st), div or "none"))
    margins = [("D_CTL under NULL (branch 11/14 edge)", NULL - dc), ("D_CTL under -NULL (CTL-BELOW-K01 edge)", -NULL - dc),
               ("D_DOWN under NULL (OWN-STEP-NECESSARY edge)", NULL - dd), ("D_DOWN over -NULL (BELOW-K01 edge)", dd + NULL),
               ("D_MUTE50 under NULL", NULL - dm), ("D_HEAD over GAP_MIN", dh - GAP_MIN),
               ("worst seed range under DIVERGED", DIVERGED - max(max(P[a]) - min(P[a]) for a in ARMS))]
    for k, v in margins:
        print("    margin %-46s %+9.4f pp" % (k, v))
    if slog:
        fl = [ln.rstrip("\n") for ln in open(slog) if ln.startswith("FINAL:")]
        print("  scorer log FINAL: %s" % (fl[-1] if fl else "(none)"))
        if fl:
            toks = [t.strip() for t in fl[-1][len("FINAL:"):].split("|")]
            if toks[0] != br:
                viol("branch %s != scorer %s" % (br, toks[0]))
            miss = [x for x in st if x not in toks]
            if miss:
                viol("stamps not in scorer FINAL: %s" % miss)
            else:
                print("  branch and all %d re-derived conditional stamps appear in the scorer's FINAL" % len(st))

    # ---------------------------------------------------------------- [5]
    print("\n[5] THE WEIGHTS BIT -- decomposition (fsum), the SET level, and the APPLIED step, from the raw probe records")
    traj, vote = {}, {}
    only50 = {HEAD_IDX: 0.0}
    for a in ARMS:
        gs = groups_of(SPEC[a], names)
        w = dict((i, 0.0) for i in SILENCED[a])
        other = None
        if a == "MUTEDOWN":
            other = dict((i, 0.0) for i in SILENCED["MUTECTL"])
        elif a == "MUTECTL":
            other = dict((i, 0.0) for i in SILENCED["MUTEDOWN"])
        for s in SEEDS:
            ptj, bsz, recs = REC[(a, s)]
            nb_want = [sum(numel[i - 1] for i in g) for g in gs]
            meta_ok = (ptj.get("n_tensors") == 53 and ptj.get("param_numels") == numel
                       and bsz.get("n_b") == nb_want and len(recs) == N_REC)
            worst_z = worst_m = 0.0
            n_inf = n_um = n_set = n_50m = n_sw = n_om = 0
            n_dis = n_fw = n_fu = n_mov = n_movw = 0
            b2 = recs[0].get("pt_b2")
            for x in recs:
                zt, mt = x["z_tensor"], x["m_tensor"]
                for k, g in enumerate(gs):
                    sz = math.fsum(abs(zt[i - 1]) for i in g) + 1e-30
                    sm = math.fsum(abs(mt[i - 1]) for i in g) + 1e-30
                    zw = math.fsum(w.get(i, 1.0) * zt[i - 1] for i in g)
                    mw = math.fsum(w.get(i, 1.0) * mt[i - 1] for i in g)
                    zu = math.fsum(zt[i - 1] for i in g)
                    mu = math.fsum(mt[i - 1] for i in g)
                    za, ma = x["z_agg"][0][k], x["mom_pre"][0][k]
                    worst_z = max(worst_z, abs(zw - za) / sz)
                    worst_m = max(worst_m, abs(mw - ma) / sm)
                    if w and abs(zw - zu) / sz > INFORM:
                        n_inf += 1
                        n_um += abs(zu - za) / sz <= BITE_TOL
                    if other is not None:
                        zo = math.fsum(only50.get(i, 1.0) * zt[i - 1] for i in g)
                        if abs(zw - zo) / sz > INFORM:
                            n_set += 1
                            n_50m += abs(zo - za) / sz <= BITE_TOL
                        zx = math.fsum(other.get(i, 1.0) * zt[i - 1] for i in g)
                        if abs(zw - zx) / sz > INFORM:
                            n_sw += 1
                            n_om += abs(zx - za) / sz <= BITE_TOL
                    cw = b2 * ma + (1 - b2) * za
                    cu = b2 * mu + (1 - b2) * zu
                    bp, bn = x["beta_pre"][0][k], x["beta"][k]
                    step = bn - bp
                    interior = LO + 2 * MS < bp < HI - 2 * MS
                    if interior and abs(step) > 0.5 * MS:
                        n_mov += 1
                        n_movw += sgn(step) == -sgn(cw)
                        if sgn(cw) != sgn(cu):
                            n_dis += 1
                            n_fw += sgn(step) == -sgn(cw)
                            n_fu += sgn(step) == -sgn(cu)
            ok = meta_ok and worst_z <= BITE_TOL and worst_m <= BITE_TOL and n_movw == n_mov
            if w:
                ok = ok and n_inf >= MIN_INF and n_um == 0 and n_fw == n_dis and n_fu == 0
            else:
                ok = ok and n_inf == 0 and n_dis == 0
            if other is not None:
                ok = ok and n_set >= MIN_INF and n_50m == 0 and n_om == 0
            print("  %-4s %-8s s%d recs %d meta %s worst z %.1e m %.1e | informative %d unweighted-match %d%s"
                  " | moved interior %d follow-weighted %d | sign DISAGREE %d: follow weighted %d, unweighted %d"
                  % ("OK" if ok else "BAD", a, s, len(recs), meta_ok, worst_z, worst_m, n_inf, n_um,
                     (" | set-inf %d 50only-match %d swap-inf %d other-set-match %d" % (n_set, n_50m, n_sw, n_om))
                     if other is not None else "", n_mov, n_movw, n_dis, n_fw, n_fu))
            if not ok:
                viol("%s-s%d weights-bit check" % (a, s))
            # trajectories
            seq = [(x["step"] / float(SPE), x["beta"][0]) for x in recs]
            pin = None
            for j in range(len(seq)):
                if all(b <= LO + 1e-3 for _e, b in seq[j:]):
                    pin = seq[j][0]
                    break
            pk = max(seq, key=lambda eb: eb[1])
            first = {}
            for thr in (-10.0, -14.0):
                first[thr] = next((e for e, b in seq if e > pk[0] and b <= thr), None)
            traj.setdefault(a, []).append({"seq": seq, "pin": pin, "peak": pk, "first": first,
                                           "clamp": sum(b <= LO + 1e-3 for _e, b in seq) / float(len(seq)),
                                           "near40": sum(1 for e, b in seq if e >= 40 and b <= -14.0) /
                                           float(sum(1 for e, _b in seq if e >= 40))})
            # the vote in group 0
            g0 = gs[0]
            ph = {}
            for x in recs:
                e = x["step"] / float(SPE)
                phase = "0-17" if e < 17 else ("17-36" if e < 36 else "36-100")
                Lr = dict((i, b2 * x["m_tensor"][i - 1] + (1 - b2) * x["z_tensor"][i - 1]) for i in g0)
                Lw = dict((i, w.get(i, 1.0) * Lr[i]) for i in g0)
                totr = math.fsum(abs(v) for v in Lr.values()) + 1e-30
                totw = math.fsum(abs(v) for v in Lw.values()) + 1e-30
                sw, su = math.fsum(Lw.values()), math.fsum(Lr.values())
                d = ph.setdefault(phase, [0, 0, 0.0, 0, 0.0])
                d[0] += 1
                d[1] += sw > 0
                d[2] += math.fsum(abs(Lr[i]) for i in SILENCED[a] if i in Lr) / totr
                d[3] += (sw > 0) != (su > 0)
                if phase == "36-100":
                    acc = ph.setdefault("top", {})
                    for i in g0:
                        if i in w:
                            continue
                        t_ = acc.setdefault(i, [0.0, 0])
                        t_[0] += abs(Lw[i]) / totw
                        t_[1] += Lw[i] > 0
                    d[4] += math.fsum(v for i, v in Lw.items() if v > 0 and i not in C_SET and i != HEAD_IDX) / totw
            vote[(a, s)] = ph

    print("\n[6] beta[0] trajectories (shared beta; HEAD: complement).  pinned-from = first record after which beta <= -14.999")
    for a in ARMS:
        T6 = traj[a]
        print("  %-8s peak %s @ep %s | after peak first <=-10 @ep %s | <=-14 @ep %s | pinned-from %s | clamp frac %s | ep>=40 frac <=-14 %s"
              % (a, " / ".join("%.3f" % t["peak"][1] for t in T6), " / ".join("%.1f" % t["peak"][0] for t in T6),
                 " / ".join("-" if t["first"][-10.0] is None else "%.1f" % t["first"][-10.0] for t in T6),
                 " / ".join("-" if t["first"][-14.0] is None else "%.1f" % t["first"][-14.0] for t in T6),
                 " / ".join("never" if t["pin"] is None else "%.1f" % t["pin"] for t in T6),
                 " / ".join("%.3f" % t["clamp"] for t in T6), " / ".join("%.3f" % t["near40"] for t in T6)))
        row = []
        for ep in list(range(10, 101, 10)):
            vals = [min(t["seq"], key=lambda eb: abs(eb[0] - ep))[1] for t in T6]
            row.append("%d:%.2f" % (ep, mean(vals)))
        print("           arm-mean beta  " + "  ".join(row))

    print("\n[7] the vote in the shared sum (HEAD: complement), L_i = w_i (b2 m_i + (1-b2) z_i), by phase (descriptive)")
    print("    DOWN = weighted sum > 0 (beta moves down); silenced-raw = silenced tensors' share of the RAW |L| mass;")
    print("    flipped = records where the weighted and unweighted sums disagree in sign")
    for a in ARMS:
        for s in SEEDS:
            ph = vote[(a, s)]
            parts = []
            for p in ("0-17", "17-36", "36-100"):
                n, dn, silr, fl, _r = ph[p]
                parts.append("ep %s: DOWN %.3f silenced-raw %.3f flipped %.3f" % (p, dn / float(n), silr / n, fl / float(n)))
            print("  %-8s s%d  %s" % (a, s, " | ".join(parts)))

    print("\n[8] top 6 UNSILENCED voters from epoch 36 (weighted share of |L|, DOWN fraction) and DOWN mass outside {50} u C")
    for a in ARMS:
        for s in SEEDS:
            ph = vote[(a, s)]
            n = ph["36-100"][0]
            top = sorted(ph["top"].items(), key=lambda kv: -kv[1][0])[:6]
            print("  %-8s s%d  outside-C DOWN mass %.3f | %s" % (
                a, s, ph["36-100"][4] / n, "  ".join("%d %s %.3f D%.2f" % (i, names[i - 1], v[0] / n, v[1] / float(n))
                                                    for i, v in top)))

    print("\n[9] arm-mean TEST (TRAIN) from the raw .out, and the settle epoch (first epoch from which arm-mean TEST stays within 0.5 pp of plateau5)")
    for a in ARMS:
        cells = []
        for e in (9, 14, 19, 24, 29, 39, 49, 69, 99):
            cells.append("%d:%.1f(%.1f)" % (e, mean([R[(a, s)]["ep"][e][1] for s in SEEDS]),
                                            mean([R[(a, s)]["ep"][e][0] for s in SEEDS])))
        am = [mean([R[(a, s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
        settle = next(e for e in range(100) if all(abs(v - M[a]) <= 0.5 for v in am[e:]))
        pk = max(range(100), key=lambda e: am[e])
        print("  %-8s %s | settle %d | arm-mean TEST peak %.2f @ep %d" % (a, "  ".join(cells), settle, am[pk], pk))

    print("\n[10] the registration's runaway prior re-read on THIS batch: fraction of epoch>=17 records whose sum would vote UP"
          " if {50} u C (resp. {50} u K) were silenced (descriptive counterfactual on the as-run trajectory)")
    for a in ("k01", "MUTE50", "MUTECTL", "MUTEDOWN"):
        for s in SEEDS:
            recs = REC[(a, s)][2]
            b2 = recs[0].get("pt_b2")
            up_c = up_k = n = 0
            for x in recs:
                if x["step"] < 17 * SPE:
                    continue
                L = [b2 * m + (1 - b2) * z for m, z in zip(x["m_tensor"], x["z_tensor"])]
                n += 1
                up_c += math.fsum(v for i, v in enumerate(L, 1) if i not in SILENCED["MUTEDOWN"]) < 0
                up_k += math.fsum(v for i, v in enumerate(L, 1) if i not in SILENCED["MUTECTL"]) < 0
            print("  %-8s s%d  records %d  UP-if-C %.3f  UP-if-K %.3f" % (a, s, n, up_c / float(n), up_k / float(n)))

    print("\n[11] level beside the shared (HEAD: complement) step-size multiplier exp(beta[0]+15), mean over epochs 15-30"
          " and 30-40 (descriptive; the same window cvt2's parser [10] reads)")
    for a in ARMS:
        w1 = [mean([math.exp(x["beta"][0] + 15.0) for x in REC[(a, s)][2] if 15 * SPE <= x["step"] < 30 * SPE]) for s in SEEDS]
        w2 = [mean([math.exp(x["beta"][0] + 15.0) for x in REC[(a, s)][2] if 30 * SPE <= x["step"] < 40 * SPE]) for s in SEEDS]
        print("  %-8s ep15-30 r %s (arm mean %.1f) | ep30-40 r %s (arm mean %.1f) | level %s" % (
            a, " / ".join("%.1f" % v for v in w1), mean(w1), " / ".join("%.1f" % v for v in w2), mean(w2),
            " / ".join("%.2f" % v for v in P[a])))

    print("\nVIOLATIONS %d" % len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
