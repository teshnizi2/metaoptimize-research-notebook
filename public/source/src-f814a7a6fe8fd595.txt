#!/usr/bin/env python3
"""cvt1_attack_indep.py <runsdir> [<scorer.py>] [<score.log>] -- independent re-derivation of `cvt1` (CORRECTIONS 230).

Imports NOTHING from analysis/cVT1_voteweight_score.py or any other repo module and uses no regex.
Reads, by string splitting only:
  * every raw cvt1-<arm>-s<seed>-<jobid>.out in <runsdir>: ARGS / ENV / PROBE_TENSOR / VOTE_W lines,
    every `Epoch` line, RUN_DONE, Traceback;
  * <runsdir>/cvt1/PARTITION-MANIFEST.txt (TENSOR lines give the 53 names, independently of the
    scorer's r18_tensors()) and PROVENANCE.txt;
  * each run's probe dir: probe_tensor.json, block_sizes.json, probe.jsonl (NO accuracy is read there).
plateau5 = mean TEST over the run's own epochs 95..99; TRAIN5 the same on the Train column.
The CSV is NOT read.  Bars and branch order are RE-TYPED from the registration text (227.6).

Sections:
  [1] completeness, ARGS / ENV / PROBE_TENSOR / VOTE_W per run
  [2] levels: per-run plateau5 + TRAIN5, per-arm mean / sd / seed range, TEST OLS tail slope 80-99
  [3] sigma (max(frozen prior, in-batch)), SE, both co-primaries and every key contrast, TEST beside TRAIN
  [4] the branch and stamps, re-typed from 227.6
  [5] THE WEIGHTS BIT -- an independent probe-level check, stronger than G-BITE's decomposition:
      (a) per record, per group, z_agg == fsum(w_i z_tensor_i) and mom_pre == fsum(w_i m_tensor_i);
      (b) THE APPLIED STEP: beta - beta_pre against -ms*sign(b2*mom_pre + (1-b2)*z_agg) (the weighted
          Lion direction) and against the same direction built from the UNWEIGHTED terms, on every
          record where the two directions disagree in sign and the coordinate is off the clamp.
          If the harness summed the weighted terms, the applied step follows the weighted direction
          on all such records and the unweighted direction on none.
  [6] beta trajectories: per arm, per seed, per group: peak and its epoch, pinned-from epoch, last,
      and the arm-mean beta at epochs 10, 20, ..., 100
  [7] the vote, descriptive: share of |L| cast by idx 50 (and the x691 twin on INJECT) in the group
      that shares the step size, and the fraction of records voting beta DOWN, by phase
"""
import hashlib
import json
import math
import os
import sys

PREFIX = "cvt1"
NET = "PlainNet18_c100"
SEEDS = [78, 79, 80]
ARMS = ["k01", "HEAD", "MUTE", "DOSE", "INJECT"]
HEADSPEC = "sets:1-49,51-53/layer4.1.bn2.weight"
SPEC = {"k01": "scalar", "HEAD": HEADSPEC, "MUTE": "scalar", "DOSE": "scalar", "INJECT": HEADSPEC}
W = {"k01": {}, "HEAD": {}, "MUTE": {50: 0.0}, "DOSE": {50: 0.1}, "INJECT": {47: 691.0}}
WIT = {"k01": "VOTE_W: off", "HEAD": "VOTE_W: off",
       "MUTE": "VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.0:group=0:groupsize=53",
       "DOSE": "VOTE_W: on type=scalar items=50:layer4.1.bn2.weight:w=0.1:group=0:groupsize=53",
       "INJECT": "VOTE_W: on type=blockwise items=47:layer4.1.bn1.weight:w=691.0:group=0:groupsize=52"}
ENV_WANT = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
            "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 "
            "EB_RHO=na EB_LOG=0")
FIXED = {"optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99", "weight-decay-base": "0.1",
         "alg-meta": "Lion", "momentum-param-meta": "0.99", "Lion-beta2-meta": "0.9",
         "weight-decay-meta": "0", "dataset": "CIFAR100", "batch-size": "100", "meta-stepsize": "1e-3",
         "alpha0": "1e-6", "num-epochs": "100", "NN-name": NET}
# 227.6, re-typed
SIGMA_PRIOR = 0.694442846939599
GAP_MIN, K01_MAX, RESCUE, NULL, MATCH, COLLAPSE, DIVERGED = 20.0, 20.0, 10.0, 2.0, 5.0, 10.0, 5.0
FLOOR_MIN, CEIL_MAX = 15.0, 90.0
MS = 1e-3
LO, HI = -15.0, -2.3026
STEPS_PER_EPOCH = 500
HEAD_IDX, TWIN_IDX = 50, 47


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
    n = len(names)
    if spec == "scalar":
        return [list(range(1, n + 1))]
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
    print("[0] manifest: %d tensors, %d params; idx50 %s (%d), idx47 %s (%d)"
          % (len(names), sum(numel), names[HEAD_IDX - 1], numel[HEAD_IDX - 1], names[TWIN_IDX - 1], numel[TWIN_IDX - 1]))
    if len(names) != 53 or names[HEAD_IDX - 1] != "layer4.1.bn2.weight" or names[TWIN_IDX - 1] != "layer4.1.bn1.weight":
        viol("manifest names")
    prov = {}
    for ln in open(os.path.join(base, "PROVENANCE.txt")):
        p = ln.split(None, 1)
        if len(p) == 2:
            prov.setdefault(p[0], p[1].strip())
    print("    PROVENANCE MODE %s  HF %s…  BUILD %s…  SCORER %s…" % (
        prov.get("MODE"), prov.get("HF_SHA256", "")[:8], prov.get("BUILD_NETWORK_SHA256", "")[:8],
        prov.get("SCORER_SHA256", "")[:8]))
    if scorer:
        s_sha = sha(scorer)
        print("    scorer file sha256 %s == PROVENANCE: %s" % (s_sha[:16], s_sha == prov.get("SCORER_SHA256")))
        if s_sha != prov.get("SCORER_SHA256"):
            viol("scorer sha")

    # ---------------------------------------------------------------- [1]
    print("\n[1] completeness and header lines")
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
                       and env == [ENV_WANT] and pt_ok and r["vw"] == [WIT[a]])
            print("  %-4s %-6s s%d job %d  epochs %d  RUN_DONE %s  tb %d  ARGS wrong %s rep %s  ENV %s  PT %s  VOTE_W %s"
                  % ("OK" if line_ok else "BAD", a, s, jid, len(r["ep"]), r["done"], r["tb"], wrong or "-", rep or "-",
                     env == [ENV_WANT], pt_ok, r["vw"] == [WIT[a]]))
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
        print("  %-6s TEST %.4f (sd %.4f, range %.4f)  TRAIN %.4f (range %.4f)  seeds %s  train %s  slope80-99 %s"
              % (a, mean(v), sdev(v), max(v) - min(v), mean(t), max(t) - min(t),
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
    C = [("P_VOTE   HEAD-MUTE", "HEAD", "MUTE"), ("P_INJECT HEAD-INJECT", "HEAD", "INJECT"),
         ("D_HEAD   HEAD-k01", "HEAD", "k01"), ("D_MUTE   MUTE-k01", "MUTE", "k01"),
         ("D_DOSE   DOSE-k01", "DOSE", "k01"), ("D_INJ    INJECT-k01", "INJECT", "k01"),
         ("         MUTE-DOSE", "MUTE", "DOSE")]
    D = {}
    for lab, x, y in C:
        d, dt = M[x] - M[y], TM[x] - TM[y]
        D[lab.split()[0] if lab.split()[0][0] in "PD" else "MUTE-DOSE"] = (d, dt)
        print("  %-22s TEST %+9.4f pp = %+8.2f SE   TRAIN %+9.4f pp" % (lab, d, d / se, dt))
    dm = D["D_MUTE"][0]
    print("  DOSE_FRAC (descriptive) = D_DOSE/D_MUTE = %s" % ("%.4f" % (D["D_DOSE"][0] / dm) if abs(dm) > 1e-9 else "n/a"))

    # ---------------------------------------------------------------- [4]
    print("\n[4] branch and stamps, re-typed from 227.6")
    hi = max(M.values())
    div = [a for a in ARMS if max(P[a]) - min(P[a]) > DIVERGED]
    pv, pi = D["P_VOTE"][0], D["P_INJECT"][0]
    dh, dmu, dd, di = D["D_HEAD"][0], D["D_MUTE"][0], D["D_DOSE"][0], D["D_INJ"][0]
    if not (FLOOR_MIN <= hi <= CEIL_MAX):
        br = "HARNESS-UNSOUND"
    elif div:
        br = "UNRESOLVED-DIVERGED"
    elif M["k01"] > K01_MAX:
        br = "SCALAR-NOT-COLLAPSED"
    elif dh < GAP_MIN:
        br = "POSITIVE-CONTROL-FAILED"
    elif -pi > MATCH:
        br = "INJECT-ABOVE-HEAD"
    elif dmu >= RESCUE:
        if pv > MATCH:
            br = "VOTE-REMOVAL-PARTIAL"
        elif pi >= COLLAPSE:
            br = "VOTE-MAGNITUDE"
        elif abs(pi) <= MATCH:
            br = "IDENTITY-BEYOND-VOTE"
        else:
            br = "VOTE-MAGNITUDE-ATTENUATED"
    elif dmu <= NULL:
        if abs(pi) <= MATCH:
            br = "OWN-STEP-SIZE"
        elif pi >= COLLAPSE:
            br = "STEP-SIZE-NEEDED-VOTE-SUFFICES"
        else:
            br = "OWN-STEP-SIZE-ATTENUATED"
    else:
        br = "MUTE-WEAK"
    st = ["DOSE-AT-K01" if dd <= NULL else ("DOSE-AT-MUTE" if abs(M["MUTE"] - M["DOSE"]) <= MATCH else "DOSE-PARTIAL"),
          "INJECT-AT-K01" if di <= NULL else ("INJECT-AT-HEAD" if abs(pi) <= MATCH else "INJECT-PARTIAL"),
          "MUTE-TRACKS-HEAD" if abs(pv) <= NULL else ("MUTE-BELOW-HEAD" if pv > 0 else "MUTE-ABOVE-HEAD")]
    if any(abs(M[a] - M["k01"]) <= NULL for a in ("HEAD", "MUTE", "DOSE", "INJECT")):
        st.append("FLOOR-READINGS-ARE-BOUNDS")
    st.append("TRAIN-AGREES" if (sgn(D["P_VOTE"][1]) == sgn(pv) and sgn(D["P_INJECT"][1]) == sgn(pi)) else "TRAIN-DISAGREES")
    st.append("SIGMA-PRIOR-FROZEN" if sig == SIGMA_PRIOR else "SIGMA-INBATCH")
    print("  branch %s   stamps %s   diverged %s" % (br, " | ".join(st), div or "none"))
    margins = {"D_MUTE vs RESCUE %.1f" % RESCUE: dmu - RESCUE, "D_MUTE vs NULL %.1f" % NULL: dmu - NULL,
               "P_VOTE vs MATCH %.1f" % MATCH: MATCH - pv, "P_INJECT vs COLLAPSE %.1f" % COLLAPSE: pi - COLLAPSE,
               "|P_INJECT| vs MATCH %.1f" % MATCH: MATCH - abs(pi), "D_HEAD vs GAP_MIN %.1f" % GAP_MIN: dh - GAP_MIN}
    for k, v in margins.items():
        print("    margin %-26s %+9.4f pp" % (k, v))
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

    # ---------------------------------------------------------------- [5] + [6] + [7]
    print("\n[5] THE WEIGHTS BIT -- decomposition (fsum) and the APPLIED step, from the raw probe records")
    traj = {}
    vote = {}
    for a in ARMS:
        gs = groups_of(SPEC[a], names)
        w = W[a]
        for s in SEEDS:
            pdir = os.path.join(base, "probe_%s-%s-s%d" % (PREFIX, a, s))
            ptj = json.load(open(os.path.join(pdir, "probe_tensor.json")))
            bsz = json.load(open(os.path.join(pdir, "block_sizes.json")))
            recs = [json.loads(x) for x in open(os.path.join(pdir, "probe.jsonl")) if x.strip()]
            nb_want = [sum(numel[i - 1] for i in g) for g in gs]
            meta_ok = (ptj.get("n_tensors") == 53 and ptj.get("param_numels") == numel
                       and bsz.get("n_b") == nb_want and len(recs) == 500)
            worst_z = worst_m = 0.0
            n_dis = n_follow_w = n_follow_u = n_moved = n_moved_w = 0
            b2 = recs[0].get("pt_b2")
            for x in recs:
                zt, mt = x["z_tensor"], x["m_tensor"]
                for k, g in enumerate(gs):
                    zw = math.fsum(w.get(i, 1.0) * zt[i - 1] for i in g)
                    mw = math.fsum(w.get(i, 1.0) * mt[i - 1] for i in g)
                    zu = math.fsum(zt[i - 1] for i in g)
                    mu = math.fsum(mt[i - 1] for i in g)
                    scale_z = math.fsum(abs(zt[i - 1]) for i in g) + 1e-30
                    scale_m = math.fsum(abs(mt[i - 1]) for i in g) + 1e-30
                    worst_z = max(worst_z, abs(zw - x["z_agg"][0][k]) / scale_z)
                    worst_m = max(worst_m, abs(mw - x["mom_pre"][0][k]) / scale_m)
                    cw = b2 * x["mom_pre"][0][k] + (1 - b2) * x["z_agg"][0][k]
                    cu = b2 * mu + (1 - b2) * zu
                    bp, bn = x["beta_pre"][0][k], x["beta"][k]
                    step = bn - bp
                    interior = LO + 2 * MS < bp < HI - 2 * MS
                    if interior and abs(step) > 0.5 * MS:
                        n_moved += 1
                        n_moved_w += sgn(step) == -sgn(cw)
                    if interior and sgn(cw) != sgn(cu) and abs(step) > 0.5 * MS:
                        n_dis += 1
                        n_follow_w += sgn(step) == -sgn(cw)
                        n_follow_u += sgn(step) == -sgn(cu)
            ok = meta_ok and worst_z <= 1e-4 and worst_m <= 1e-4 and n_moved_w == n_moved
            if w:
                ok = ok and n_dis > 0 and n_follow_w == n_dis and n_follow_u == 0
            else:
                ok = ok and n_dis == 0
            print("  %-4s %-6s s%d  recs %d meta %s  worst rel z %.1e m %.1e | moved interior %d, follow weighted dir %d"
                  " | weighted/unweighted sign DISAGREE %d: step follows weighted %d, unweighted %d"
                  % ("OK" if ok else "BAD", a, s, len(recs), meta_ok, worst_z, worst_m, n_moved, n_moved_w,
                     n_dis, n_follow_w, n_follow_u))
            if not ok:
                viol("%s-s%d weights-bit check" % (a, s))
            # trajectories
            for k in range(len(gs)):
                seq = [(x["step"] / float(STEPS_PER_EPOCH), x["beta"][k]) for x in recs]
                pin = None
                for j in range(len(seq)):
                    if all(b <= LO + 1e-3 for _e, b in seq[j:]):
                        pin = seq[j][0]
                        break
                pk = max(seq, key=lambda eb: eb[1])
                traj.setdefault((a, k), []).append({"seq": seq, "pin": pin, "peak": pk, "last": seq[-1][1],
                                                    "clampfrac": sum(b <= LO + 1e-3 for _e, b in seq) / float(len(seq))})
            # vote shares in group 0 (the group idx 50 or the x691 twin shares)
            g0 = gs[0]
            ph = {}
            for x in recs:
                e = x["step"] / float(STEPS_PER_EPOCH)
                phase = "0-17" if e < 17 else ("17-36" if e < 36 else "36-100")
                L = [w.get(i, 1.0) * (b2 * x["m_tensor"][i - 1] + (1 - b2) * x["z_tensor"][i - 1]) for i in g0]
                Lraw50 = b2 * x["m_tensor"][HEAD_IDX - 1] + (1 - b2) * x["z_tensor"][HEAD_IDX - 1]
                tot = math.fsum(abs(v) for v in L) + 1e-30
                sh50 = abs(L[g0.index(HEAD_IDX)]) / tot if HEAD_IDX in g0 else float("nan")
                sh47 = abs(L[g0.index(TWIN_IDX)]) / tot
                d = ph.setdefault(phase, [0, 0.0, 0.0, 0, 0.0])
                d[0] += 1
                d[1] += 0 if sh50 != sh50 else sh50
                d[2] += sh47
                d[3] += math.fsum(L) > 0
                d[4] += abs(Lraw50)
                if e >= 36:
                    acc = ph.setdefault("top", {})
                    for i, v in zip(g0, L):
                        t_ = acc.setdefault(i, [0.0, 0])
                        t_[0] += abs(v) / tot
                        t_[1] += v > 0
            vote[(a, s)] = ph

    print("\n[6] beta trajectories (from probe.jsonl; pinned-from = first record after which beta stays <= -14.999)")
    for a in ARMS:
        ng = 1 if SPEC[a] == "scalar" else 2
        for k in range(ng):
            lab = "beta[0]" if ng == 1 else ("complement beta[0]" if k == 0 else "isolated beta[1] (idx 50)")
            T6 = traj[(a, k)]
            print("  %-6s %-26s peak %s @ep %s | pinned-from %s | at-clamp frac %s | last %s"
                  % (a, lab, " / ".join("%.3f" % t["peak"][1] for t in T6),
                     " / ".join("%.1f" % t["peak"][0] for t in T6),
                     " / ".join("never" if t["pin"] is None else "%.1f" % t["pin"] for t in T6),
                     " / ".join("%.3f" % t["clampfrac"] for t in T6),
                     " / ".join("%.3f" % t["last"] for t in T6)))
            row = []
            for ep in range(10, 101, 10):
                vals = []
                for t in T6:
                    near = min(t["seq"], key=lambda eb: abs(eb[0] - ep))
                    vals.append(near[1])
                row.append("%d:%.2f" % (ep, mean(vals)))
            print("         arm-mean beta  " + "  ".join(row))

    print("\n[7] the vote in the group that shares the step size (L_i = w_i (b2 m_i + (1-b2) z_i)), by phase (descriptive)")
    for a in ARMS:
        for s in SEEDS:
            ph = vote[(a, s)]
            parts = []
            for p in ("0-17", "17-36", "36-100"):
                if p in ph:
                    n, s50, s47, down, raw50 = ph[p]
                    parts.append("ep %s: share50 %s share47 %.4f DOWN %.3f" % (
                        p, "  n/a " if a in ("HEAD", "INJECT") else "%.4f" % (s50 / n), s47 / n, down / float(n)))
            print("  %-6s s%d  %s" % (a, s, " | ".join(parts)))

    # added after the first pass (sections [0]-[7] printed identically before and after)
    print("\n[8] top 5 voters in that group, epochs 36-100: mean share of |L| (weighted), fraction of records voting DOWN (descriptive)")
    for a in ARMS:
        for s in SEEDS:
            ph = vote[(a, s)]
            n = ph["36-100"][0]
            top = sorted(ph["top"].items(), key=lambda kv: -kv[1][0])[:5]
            print("  %-6s s%d  %s" % (a, s, "  ".join("%d %s %.3f DOWN %.2f" % (i, names[i - 1], v[0] / n, v[1] / float(n))
                                                   for i, v in top)))

    # added after the second pass (sections [0]-[8] printed identically before and after)
    print("\n[9] arm-mean TEST (TRAIN) at epochs 9, 19, 29, 39, 49, 69, 99 from the raw .out (descriptive)")
    for a in ARMS:
        cells = []
        for e in (9, 19, 29, 39, 49, 69, 99):
            cells.append("%d:%.1f(%.1f)" % (e, mean([R[(a, s)]["ep"][e][1] for s in SEEDS]),
                                            mean([R[(a, s)]["ep"][e][0] for s in SEEDS])))
        print("  %-6s %s" % (a, "  ".join(cells)))

    print("\nVIOLATIONS %d" % len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
