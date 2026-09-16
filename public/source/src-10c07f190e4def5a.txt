#!/usr/bin/env python3
"""cvt4_attack_indep.py <runsdir> <scorer.py> <score.log> -- independent re-derivation of `cvt4` (CORRECTIONS 237).

Imports NOTHING from analysis/cVT4_betahold_score.py or any other repo module and uses no regex.
Reads, by string splitting only: every raw cvt4-<arm>-s<seed>-<jobid>.out in <runsdir> (ARGS / ENV / PROBE_TENSOR /
VOTE_W / BETA_HOLD lines, `Epoch` lines, RUN_DONE, Traceback); <runsdir>/cvt4/PARTITION-MANIFEST.txt and
PROVENANCE.txt; each run's probe dir (probe_tensor.json, block_sizes.json, probe.jsonl).  The CSV is NOT read (the
scorer's corpus line is a disclosure that feeds no bar, sigma, branch or stamp; it is not re-derived here).
plateau5 = mean TEST over the run's own epochs 95..99; TRAIN5 the same on the Train column.
Bars, witnesses, the hold schedule, G-BITE's clauses and the branch order are RE-TYPED from the registration text
(237.2, 237.3, 237.4, 237.6); nothing is copied from the scorer's code.

AGREEMENT.  Every line the scorer prints that carries a level, contrast, gate, G-BITE count, sigma, branch, stamp or
descriptive readout is RE-BUILT here, in the scorer's printed format and precision, from this file's own numbers, and
then looked up VERBATIM in <score.log>.  A line that is not found is a VIOLATION.  The FINAL token list is compared
token by token and as one string.

Sections:
  [0] manifest and PROVENANCE (sha prefixes re-typed from 237.4 / 237.8 / 237.12)
  [1] completeness, ARGS / ENV / PROBE_TENSOR / VOTE_W / BETA_HOLD per run (witnesses BUILT from mode and P)
  [2] levels: per-run plateau5 + TRAIN5, per-arm mean / sd / range, TEST OLS tail slope 80-99
  [3] sigma (max(frozen prior, in-batch)), SE, every contrast, TEST beside TRAIN; per-seed paired co-primaries
  [4] G-BITE re-derived from the probe records (Lion on every unheld group; the schedule and its beta_pre half on the
      held group; the bh_* audit; no bh_* key on controls), plus the vote-out decomposition (z_agg == fsum z_tensor
      over each group) and a NON-VACUITY cross-read (every run's group 1 read as every held mode)
  [5] branch, states, stamps, every bar margin; FINAL compared with the scorer's
  [6] the scorer's descriptive readouts re-derived (beta peak / pin / r15-30 per run; arm-mean TEST at 9..99)
  [7] LINE-BY-LINE AGREEMENT with the scorer log
  DESCRIPTIVE, NOT REGISTERED (cannot move anything above):
  [D1] tensor 50's APPLIED beta and the complement's beta, epochs 0-40, median seed per arm
  [D2] TEST (TRAIN) after 15 / 30 / 50 / 100 epochs
  [D3] when each level is set: settle epoch, share of plateau5 reached, divergence from HEAD
  [D4] HOLDSHARED's and HOLDHIGH's live complement against MUTE's measured trajectory (cvt1 MUTE, cvt3 MUTE50;
       between batch) and against the replay schedule v_9428
"""
import hashlib
import json
import math
import os
import struct
import sys

PREFIX = "cvt4"
NET = "PlainNet18_c100"
SEEDS = [90, 91, 92]
ARMS = ["k01", "HEAD", "HOLDLOW", "HOLDSHARED", "HOLDHIGH", "HOLDHEAD"]
HELD = ["HOLDLOW", "HOLDSHARED", "HOLDHIGH", "HOLDHEAD"]
HEADSPEC = "sets:1-49,51-53/layer4.1.bn2.weight"
SPEC = dict((a, HEADSPEC) for a in ARMS)
SPEC["k01"] = "scalar"
IDX50 = 50
NAME50 = "layer4.1.bn2.weight"
# 237.2 table, re-typed
HOLD = {"HOLDLOW": ("floor", None), "HOLDSHARED": ("shared", None), "HOLDHIGH": ("tri", 9428), "HOLDHEAD": ("tri", 5041)}
ENV_WANT = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
            "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 "
            "EB_RHO=na EB_LOG=0")
FIXED = {"optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99", "weight-decay-base": "0.1",
         "alg-meta": "Lion", "momentum-param-meta": "0.99", "Lion-beta2-meta": "0.9",
         "weight-decay-meta": "0", "dataset": "CIFAR100", "batch-size": "100", "meta-stepsize": "1e-3",
         "alpha0": "1e-6", "num-epochs": "100", "NN-name": NET, "gamma": "1"}
# 237.6 frozen bars, re-typed
SIGMA_PRIOR = 0.694442846939599
GAP_MIN, K01_MAX, MATCH, NULL, RESCUE, DIVERGED = 20.0, 20.0, 5.0, 2.0, 10.0, 5.0
FLOOR_MIN, CEIL_MAX = 15.0, 90.0
BH_TOL, TIE_REL, N_REC = 1e-5, 1e-12, 500
MS, B2 = 1e-3, 0.9
LO, HI = -15.0, -2.3026
SPE = 500
# sha prefixes as recorded in CORRECTIONS 237.4 / 237.8 / 237.12
PREF = {"BUILD_NETWORK_SHA256": "e65e6773", "HF_SHA256": "84b345ad", "RUNNER_SHA256": "c801fc85"}
MANIFEST_PREFIX, MANIFEST_BYTES = "b2f22089", 4508


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


B0 = f32(math.log(1e-6))


def mean(v):
    return math.fsum(v) / len(v)


def sdev(v):
    m = mean(v)
    return math.sqrt(math.fsum((x - m) ** 2 for x in v) / (len(v) - 1))


def ols(xs, ys):
    xm, ym = mean(xs), mean(ys)
    return math.fsum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / math.fsum((x - xm) ** 2 for x in xs)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sgn(x):
    return (x > 0) - (x < 0)


def tri(n, P):
    """237.3(1): u(n) = min(n-1, P) - max(0, n-1-P); beta = min(hi, max(lo, b0 + ms u(n))), stored float32."""
    if n <= 0:
        u = 0
    else:
        u = min(n - 1, P) - max(0, n - 1 - P)
    return f32(min(HI, max(LO, B0 + MS * u)))


def lion(bp, m, z):
    """237.6(b): clamp(beta_pre - ms*sign(b2*mom_pre + (1-b2)*z_agg)); tie iff |L| <= TIE_REL of its parts."""
    p, q = B2 * m, (1.0 - B2) * z
    L = p + q
    return max(LO, min(HI, bp - MS * sgn(L))), abs(L) <= TIE_REL * (abs(p) + abs(q))


def witness(arm):
    if arm not in HOLD:
        return "BETA_HOLD: off"
    mode, P = HOLD[arm]
    w = "BETA_HOLD: on type=blockwise group=1 groupsize=1 name=%s mode=%s" % (NAME50, mode)
    if mode == "floor":
        return w + " value=" + repr(LO)
    if mode == "shared":
        return w + " source=0 sourcesize=52"
    return w + " P=%d b0=%s ms=%s lo=%s hi=%s peak=%s" % (P, repr(B0), repr(MS), repr(LO), repr(HI),
                                                          repr(min(HI, max(LO, B0 + MS * P))))


def parse_out(path):
    r = {"args": [], "env": [], "pt": [], "vw": [], "bh": [], "ep": {}, "done": False, "tb": 0, "dup": 0}
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
        elif s.startswith("BETA_HOLD"):
            r["bh"].append(s)
        elif s.startswith("Epoch "):
            p = s.split(",")
            e = int(p[0].split()[1])
            tr = float(p[1].split(":")[1].replace("%", "").strip())
            te = float(p[2].split(":")[1].replace("%", "").strip())
            r["dup"] += e in r["ep"]
            r["ep"][e] = (tr, te)
        elif s.strip() == "RUN_DONE":
            r["done"] = True
        elif "Traceback (most recent call last)" in s:
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


def load_recs(pdir):
    out = []
    for x in open(os.path.join(pdir, "probe.jsonl")):
        if not x.strip():
            continue
        try:
            v = json.loads(x)
        except ValueError:
            v = None
        out.append(v if isinstance(v, dict) else None)
    return out


def traj_summary(recs, k):
    """(peak, peak epoch, continuous-clamp pin epoch or None, mean r over epochs [15,30)) of group k."""
    seq = [(x["step"], float(x["beta"][k])) for x in recs]
    peak = seq[0]
    for s, b in seq:
        if b > peak[1]:
            peak = (s, b)
    pin = None
    for j in range(len(seq)):
        if all(b <= LO + 1e-4 for _s, b in seq[j:]):
            pin = seq[j][0] / float(SPE)
            break
    r = mean([math.exp(b + 15.0) for s, b in seq if 15 * SPE <= s < 30 * SPE])
    return peak[1], peak[0] / float(SPE), pin, r


def main():
    runsdir, scorer, slog = sys.argv[1], sys.argv[2], sys.argv[3]
    bad = []
    expect = []          # (section, line) that must appear verbatim in the scorer log

    def viol(msg):
        bad.append(msg)
        print("  VIOLATION " + msg)

    def exp(sec, line):
        expect.append((sec, line))

    base = os.path.join(runsdir, PREFIX)
    # ---------------------------------------------------------------- [0]
    names, numel, man = [], [], {}
    mpath = os.path.join(base, "PARTITION-MANIFEST.txt")
    mtext = open(mpath).read()
    for ln in mtext.split("\n"):
        p = ln.split()
        if not p:
            continue
        if p[0] == "TENSOR":
            names.append(p[2])
            numel.append(int(p[3]))
        else:
            man.setdefault(p[0], []).append(ln)
    msha = sha(mpath)
    print("[0] manifest: %d tensors, %d params; idx50 %s; %d bytes, sha256 %s…" % (
        len(names), sum(numel), names[IDX50 - 1], len(mtext.encode()), msha[:16]))
    ok_man = (len(names) == 53 and sum(numel) == 11046308 and names[IDX50 - 1] == NAME50
              and numel[IDX50 - 1] == 512 and msha.startswith(MANIFEST_PREFIX) and len(mtext.encode()) == MANIFEST_BYTES
              and man.get("REPLAY") == ["REPLAY B0 %s MS %s P_HIGH 9428 P_HEAD 5041" % (repr(B0), repr(MS))]
              and man.get("WITNESS") == ["WITNESS %s %s" % (a, witness(a)) for a in ARMS])
    print("    REPLAY line, six WITNESS lines == the witnesses built here, sha prefix %s, %d bytes: %s"
          % (MANIFEST_PREFIX, MANIFEST_BYTES, ok_man))
    if not ok_man:
        viol("manifest")
    prov = {}
    for ln in open(os.path.join(base, "PROVENANCE.txt")):
        p = ln.split(None, 1)
        if len(p) == 2:
            prov.setdefault(p[0], p[1].strip())
    s_sha = sha(scorer)
    ok_prov = (prov.get("MODE") == "submit" and all(prov.get(k, "").startswith(v) for k, v in PREF.items())
               and prov.get("SCORER_SHA256") == s_sha)
    print("    PROVENANCE MODE %s  BUILD %s…  HF %s…  RUNNER %s…  SCORER %s… == scorer file: %s" % (
        prov.get("MODE"), prov.get("BUILD_NETWORK_SHA256", "")[:8], prov.get("HF_SHA256", "")[:8],
        prov.get("RUNNER_SHA256", "")[:8], prov.get("SCORER_SHA256", "")[:8], prov.get("SCORER_SHA256") == s_sha))
    if not ok_prov:
        viol("provenance")
    exp("G-STRUCT", "  PASS G-STRUCT byte-identical to synthetic_manifest_text() (53 tensors, HEAD [52,1], the hold table)"
        "   %d vs %d bytes" % (MANIFEST_BYTES, MANIFEST_BYTES))
    exp("G-PROV", "  PASS G-PROV MODE submit   submit")
    exp("G-PROV", "  PASS G-PROV build_network.py sha256 == POST_SHA %s...   %s" % (
        prov["BUILD_NETWORK_SHA256"][:12], prov["BUILD_NETWORK_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV HF.py sha256 == HF_POST_SHA %s... (cvt4's PATCH_BETAHOLD tree)   %s" % (
        prov["HF_SHA256"][:12], prov["HF_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV RUNNER_SHA256 == cvt4's runner %s...   %s" % (prov["RUNNER_SHA256"][:12], prov["RUNNER_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV SCORER_SHA256 == this scorer's sha256 %s...   %s" % (s_sha[:12], s_sha[:16]))

    # ---------------------------------------------------------------- [1]
    print("\n[1] completeness and header lines (BETA_HOLD witnesses built from mode / P / b0 / ms)")
    files = {}
    for fn in sorted(os.listdir(runsdir)):
        if not (fn.startswith(PREFIX + "-") and fn.endswith(".out")):
            continue
        stem = fn[:-4].split("-")
        if len(stem) != 4 or stem[1] not in ARMS:
            continue
        files.setdefault((stem[1], int(stem[2][1:])), []).append((int(stem[3]), fn))
    R = {}
    envs = set()
    n_pt_ok = 0
    for s in SEEDS:
        for a in ARMS:
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
            for e in env:
                envs.add(e)
            ptw = "PROBE_TENSOR: on every=100 type=%s tensors=53 " % ("scalar" if a == "k01" else "blockwise")
            pt_ok = len(r["pt"]) == 1 and r["pt"][0].startswith(ptw)
            n_pt_ok += pt_ok
            complete = ok_ep and r["done"] and r["tb"] == 0
            args_ok = len(r["args"]) == 1 and not wrong and not rep
            vw_ok = r["vw"] == ["VOTE_W: off"]
            bh_ok = r["bh"] == [witness(a)]
            line_ok = complete and args_ok and env == [ENV_WANT] and pt_ok and vw_ok and bh_ok
            print("  %-4s %-10s s%d job %d  epochs %d  RUN_DONE %s  tb %d  ARGS wrong %s rep %s  ENV %s  PT %s  VOTE_W %s  BETA_HOLD %s"
                  % ("OK" if line_ok else "BAD", a, s, jid, len(r["ep"]), r["done"], r["tb"], wrong or "-", rep or "-",
                     env == [ENV_WANT], pt_ok, vw_ok, bh_ok))
            if not line_ok:
                viol("%s-s%d header/completeness" % (a, s))
            if args_ok:
                exp("G-ARGS", "  PASS G-ARGS %s-s%d" % (a, s))
            if vw_ok:
                exp("G-VOTEW", "  PASS G-VOTEW %s-s%d" % (a, s))
            if bh_ok:
                exp("G-HOLDW", "  PASS G-HOLDW %s-s%d" % (a, s))
    if envs == {ENV_WANT} and all(len(R[k]["env"]) == 1 for k in R):
        exp("G-ENV", "  PASS G-ENV one distinct ENV line, one per run   1 distinct")
        exp("G-ENV", "        " + ENV_WANT)
        exp("G-ENV", "  PASS G-ENV the ENV line is cvt1's / cpl2's, byte for byte (PROBE_DIR stripped)")
    if n_pt_ok == 18:
        exp("G-ENV", "  PASS G-ENV every run printed ONE `PROBE_TENSOR: on every=100 type=<its arm's> tensors=53`   18 runs")

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
            r = R[(a, s)]
            exp("COMPLETE", "  OK   %s-%s-s%d  %d epoch lines, RUN_DONE=%s, traceback=%s, plateau5 %.4f (train %.4f)"
                % (PREFIX, a, s, len(r["ep"]), r["done"], r["tb"] > 0, v[-1], t[-1]))
        P[a], T[a] = v, t
        print("  %-10s TEST %.4f (sd %.4f, range %.4f)  TRAIN %.4f  seeds %s  train %s  slope80-99 %s"
              % (a, mean(v), sdev(v), max(v) - min(v), mean(t), " / ".join("%.4f" % x for x in v),
                 " / ".join("%.4f" % x for x in t), " / ".join("%+.3f" % x for x in sl)))
        exp("LEVELS", "  %-10s TEST %.4f  sd %.4f  range %.4f   TRAIN %.4f   tail slope %s pp/ep"
            % (a, mean(v), sdev(v), max(v) - min(v), mean(t), "/".join("%.3f" % x for x in sl)))
        exp("LEVELS", "             seeds: %s" % ", ".join("s%d %.4f (train %.4f)" % (s, x, y) for s, x, y in zip(SEEDS, v, t)))
    M = dict((a, mean(P[a])) for a in ARMS)
    TM = dict((a, mean(T[a])) for a in ARMS)
    hi = max(M.values())
    exp("G-FLOOR", "  PASS G-FLOOR max arm mean >= 15.00 pp (chance 1.00)   max %.4f" % hi)
    exp("G-CEIL", "  PASS G-CEIL max arm mean <= 90.00 pp   max %.4f" % hi)

    # ---------------------------------------------------------------- [3]
    print("\n[3] sigma and contrasts (within batch)")
    ss = math.fsum((x - M[a]) ** 2 for a in ARMS for x in P[a])
    df = len(ARMS) * (len(SEEDS) - 1)
    sig_in = math.sqrt(ss / df)
    sig = max(SIGMA_PRIOR, sig_in)
    which = "SIGMA_PRIOR_frozen" if sig == SIGMA_PRIOR else "SIGMA_INBATCH"
    se = sig * math.sqrt(2.0 / 3.0)
    print("  SIGMA_INBATCH %.6f (df %d)  SIGMA_PRIOR %.6f  -> used %.6f (%s)  SE %.6f" % (sig_in, df, SIGMA_PRIOR, sig, which, se))
    exp("SIGMA", "  %-20s %.6f" % ("SIGMA_PRIOR_frozen", SIGMA_PRIOR))
    exp("SIGMA", "  %-20s %.6f" % ("SIGMA_INBATCH", sig_in))
    exp("SIGMA", "  SIGMA_USED           %.6f  (= %s)" % (sig, which))
    exp("SIGMA", "  SE_ARM_DIFF          %.6f  = SIGMA_USED * sqrt(2/3)  (df_inbatch %d)" % (se, df))
    C = [("P_HIGH", "HEAD", "HOLDHIGH", "PRIMARY"), ("P_COUP", "HOLDHIGH", "HOLDSHARED", "PRIMARY"),
         ("P_LOW", "HOLDLOW", "HEAD", "PRIMARY"), ("D_HEAD", "HEAD", "k01", "KEY"), ("P_CTL", "HEAD", "HOLDHEAD", "KEY")]
    C += [("D_" + a, a, "k01", "KEY") for a in HELD]
    CT = {}
    for lab, x, y, kind in C:
        d, dt = M[x] - M[y], TM[x] - TM[y]
        CT[lab] = (d, dt)
        print("  %-8s %-14s = %-10s - %-10s TEST %+9.4f pp = %+8.2f SE   TRAIN %+9.4f   per seed %s"
              % (kind, lab, x, y, d, d / se, dt, " / ".join("%+.4f" % (P[x][j] - P[y][j]) for j in range(3))))
    fmt3 = {"P_HIGH": "HEAD - HOLDHIGH      ", "P_COUP": "HOLDHIGH - HOLDSHARED", "P_LOW": "HOLDLOW - HEAD       "}
    for lab in ("P_HIGH", "P_COUP", "P_LOW"):
        d, dt = CT[lab]
        exp("CONTRASTS", "  PRIMARY  %-8s = %s = %+.4f pp = %+.2f SE   (TRAIN %+.4f)" % (lab, fmt3[lab], d, d / se, dt))
    d, dt = CT["D_HEAD"]
    exp("CONTRASTS", "  KEY      D_HEAD   = HEAD - k01            = %+.4f pp = %+.2f SE   (TRAIN %+.4f)  [positive control]" % (d, d / se, dt))
    d, dt = CT["P_CTL"]
    exp("CONTRASTS", "  KEY      P_CTL    = HEAD - HOLDHEAD       = %+.4f pp = %+.2f SE   (TRAIN %+.4f)  [hold control]" % (d, d / se, dt))
    for a in HELD:
        d, dt = CT["D_" + a]
        exp("CONTRASTS", "  KEY      D_%-10s = %-10s - k01  = %+.4f pp = %+.2f SE   (TRAIN %+.4f)   HEAD - %-10s = %+.4f"
            % (a, a, d, d / se, dt, a, M["HEAD"] - M[a]))
    exp("CONTRASTS", "  bars: GAP_MIN %.1f (%.2f SE) | MATCH %.1f pp (%.2f SE) | NULL %.1f pp (%.2f SE) | RESCUE %.1f pp (%.2f SE)"
        % (GAP_MIN, GAP_MIN / se, MATCH, MATCH / se, NULL, NULL / se, RESCUE, RESCUE / se))

    # ---------------------------------------------------------------- [4]
    print("\n[4] G-BITE re-derived from the raw probe records (237.6 (a)-(e)), the vote-out decomposition, non-vacuity")
    RECS = {}
    bite_ok_all = True
    nonfin = {}
    cross = {}
    for s in SEEDS:
        for a in ARMS:
            pdir = os.path.join(base, "probe_%s-%s-s%d" % (PREFIX, a, s))
            gs = groups_of(SPEC[a], names)
            ng = len(gs)
            recs = load_recs(pdir)
            RECS[(a, s)] = recs
            ptj = json.load(open(os.path.join(pdir, "probe_tensor.json")))
            bsz = json.load(open(os.path.join(pdir, "block_sizes.json")))
            meta_ok = (ptj.get("n_tensors") == 53 and ptj.get("param_numels") == numel
                       and bsz.get("n_b") == [sum(numel[i - 1] for i in g) for g in gs])
            held = a in HOLD
            c = dict(n=0, nonfin=0, badrec=0, badstep=0, lion=0, ties=0, sched=0, schedpre=0, bh=0, bhctl=0, act=0)
            wl = ws = 0.0
            last_act = None
            prev = -1
            worst_dec = 0.0
            for k, x in enumerate(recs):
                if (x is None or type(x.get("step")) is not int or not isinstance(x.get("beta"), list)
                        or len(x["beta"]) != ng):
                    c["badrec"] += 1
                    continue
                try:
                    beta = [float(v) for v in x["beta"]]
                    bp = [float(v) for v in x["beta_pre"][0]]
                    mo = [float(v) for v in x["mom_pre"][0]]
                    za = [float(v) for v in x["z_agg"][0]]
                except (KeyError, TypeError, ValueError, IndexError):
                    c["badrec"] += 1
                    continue
                if not (len(x["beta_pre"]) == 1 and len(bp) == ng and len(mo) == ng and len(za) == ng):
                    c["badrec"] += 1
                    continue
                if not all(math.isfinite(v) for v in beta + bp + mo + za):
                    c["nonfin"] += 1
                    continue
                st = x["step"]
                c["badstep"] += st != 100 * k
                c["n"] += 1
                for g in ((0,) if held else range(ng)):
                    nat, tie = lion(bp[g], mo[g], za[g])
                    if tie:
                        c["ties"] += 1
                        continue
                    e = abs(nat - beta[g])
                    wl = max(wl, e)
                    c["lion"] += e > BH_TOL
                # vote-out decomposition (not a G-BITE clause): z_agg[g] == fsum z_tensor over the group
                zt = x["z_tensor"]
                for g, grp in enumerate(gs):
                    zs = math.fsum(zt[i - 1] for i in grp)
                    den = math.fsum(abs(zt[i - 1]) for i in grp) + 1e-30
                    worst_dec = max(worst_dec, abs(zs - za[g]) / den)
                has_bh = any(key in x for key in ("bh_n", "bh_active", "bh_nat", "bh_held"))
                # cross-read: group 1 against every held mode (non-vacuity)
                if ng == 2:
                    for h in HELD:
                        mode, Pp = HOLD[h]
                        if mode == "shared":
                            e1, e2 = abs(beta[1] - beta[0]), abs(bp[1] - bp[0])
                        elif mode == "floor":
                            e1, e2 = abs(beta[1] - LO), abs(bp[1] - LO)
                        else:
                            e1, e2 = abs(beta[1] - tri(st + 2, Pp)), abs(bp[1] - tri(st + 1, Pp))
                        cross.setdefault((a, s, h), [0, 0.0])
                        cross[(a, s, h)][0] += (e1 > BH_TOL) or (e2 > BH_TOL)
                        cross[(a, s, h)][1] = max(cross[(a, s, h)][1], e1, e2)
                if not held:
                    c["bhctl"] += has_bh
                    continue
                n = st + 2
                mode, Pp = HOLD[a]
                if mode == "shared":
                    e1, e2 = abs(beta[1] - beta[0]), abs(bp[1] - bp[0])
                elif mode == "floor":
                    e1, e2 = abs(beta[1] - LO), abs(bp[1] - LO)
                else:
                    e1, e2 = abs(beta[1] - tri(n, Pp)), abs(bp[1] - tri(n - 1, Pp))
                ws = max(ws, e1, e2)
                c["sched"] += e1 > BH_TOL
                c["schedpre"] += e2 > BH_TOL
                try:
                    bn, ba, bnat, bheld = int(x["bh_n"]), int(x["bh_active"]), float(x["bh_nat"]), float(x["bh_held"])
                except (KeyError, TypeError, ValueError):
                    c["bh"] += 1
                    continue
                nat1, tie1 = lion(bp[1], mo[1], za[1])
                good = (bn == n and abs(bheld - beta[1]) <= BH_TOL and 0 <= ba <= bn and ba >= prev
                        and (tie1 or abs(bnat - nat1) <= BH_TOL))
                c["bh"] += not good
                c["act"] += abs(bnat - bheld) > BH_TOL
                prev = max(prev, ba)
                last_act = ba
            ok = (meta_ok and len(recs) == N_REC and c["badrec"] == 0 and c["n"] + c["nonfin"] == N_REC
                  and c["badstep"] == 0 and c["lion"] == 0)
            if held:
                ok = ok and c["sched"] == 0 and c["schedpre"] == 0 and c["bh"] == 0 and (last_act is None or last_act >= c["act"])
            else:
                ok = ok and c["bhctl"] == 0
            nonfin[(a, s)] = c["nonfin"]
            bite_ok_all = bite_ok_all and ok
            print("  %-4s %-10s s%d meta %s recs %d lion-mism %d ties %d worst %.1e | sched %d/%d worst %.1e | bh-audit %d"
                  " bh-on-control %d active %d bh_active %s | vote-out decomposition worst rel %.1e"
                  % ("PASS" if ok else "FAIL", a, s, meta_ok, len(recs), c["lion"], c["ties"], wl, c["sched"], c["schedpre"],
                     ws, c["bh"], c["bhctl"], c["act"], last_act, worst_dec))
            if not ok:
                viol("G-BITE %s-s%d" % (a, s))
            if worst_dec > 1e-4:
                viol("decomposition %s-s%d" % (a, s))
            exp("G-BITE", "  %-4s G-BITE %-10s s%d  records %s  bad %d  nonfinite %d  lion-mismatch %d (ties %d, worst %.1e)  "
                "sched %d/%d (worst %.1e)  bh-audit %d  bh-on-control %d  active records %d  bh_active %s"
                % ("PASS" if ok else "FAIL", a, s, len(recs), c["badrec"], c["nonfin"], c["lion"], c["ties"], wl,
                   c["sched"], c["schedpre"], ws, c["bh"], c["bhctl"], c["act"], last_act))
    print("  NON-VACUITY cross-read: records (of 500) on which group 1 FAILS each held mode's schedule (either half); max |error|")
    print("  %-10s %s" % ("run", "  ".join("%-22s" % ("as " + h) for h in HELD)))
    for a in ARMS[1:]:
        for s in SEEDS:
            print("  %-10s s%d %s" % (a, s, "  ".join("%3d  max %8.4f      " % tuple(cross[(a, s, h)]) for h in HELD)))
    nv_ok = all(cross[(a, s, h)][0] > 0 for a in ARMS[1:] for s in SEEDS for h in HELD if h != a)
    nv_own = all(cross[(a, s, a)][0] == 0 for a in HELD for s in SEEDS)
    print("  every run FAILS every held mode but its own on >= 1 record: %s; every held run passes its own on all: %s" % (nv_ok, nv_own))
    if not (nv_ok and nv_own):
        viol("non-vacuity cross-read")

    # ---------------------------------------------------------------- [5]
    print("\n[5] branch, states, stamps (237.6, re-typed; first match wins)")
    div = [a for a in ARMS if max(P[a]) - min(P[a]) > DIVERGED]
    gap = M["HEAD"] - M["k01"]

    def st_of(a):
        if M[a] >= M["HEAD"] - MATCH:
            return "AT-HEAD"
        if M[a] <= M["k01"] + NULL:
            return "AT-K01"
        return "BETWEEN"

    states = dict((a, st_of(a)) for a in HELD)
    if not bite_ok_all:
        branch = "PATCH-NOT-VERIFIED"
    elif not (FLOOR_MIN <= hi <= CEIL_MAX):
        branch = "HARNESS-UNSOUND"
    elif div:
        branch = "UNRESOLVED-DIVERGED"
    elif M["k01"] > K01_MAX:
        branch = "SCALAR-NOT-COLLAPSED"
    elif gap < GAP_MIN:
        branch = "POSITIVE-CONTROL-FAILED"
    elif states["HOLDHEAD"] == "AT-K01":
        branch = "HOLD-CONTROL-FAILED"
    elif states["HOLDHEAD"] == "BETWEEN":
        branch = "HOLD-CONTROL-PARTIAL"
    elif states["HOLDSHARED"] == "AT-HEAD":
        branch = "VOTE-REMOVAL"
    elif states["HOLDSHARED"] == "BETWEEN":
        branch = "SHARED-PARTIAL"
    elif states["HOLDLOW"] == "AT-K01":
        branch = "LOW-STALLS"
    elif states["HOLDLOW"] == "BETWEEN":
        branch = "LOW-PARTIAL"
    else:
        branch = {"AT-K01": "OWN-STEP-MAGNITUDE", "BETWEEN": "OWN-STEP-GRADED", "AT-HEAD": "COUPLING"}[states["HOLDHIGH"]]
    stamps = ["HARNESS-CLEAN", "PATCH-BITES", "HOLD-FROM-INIT", "REPLAY-MAX-RATE-TRIANGLE", "SHARED-IS-MUTE-ARITHMETIC",
              "ONE-NETWORK-PLAINNET", "HORIZON-100-ONLY",
              "SIGMA-PRIOR-FROZEN" if which == "SIGMA_PRIOR_frozen" else "SIGMA-INBATCH"]
    if div:
        stamps.append("DIVERGED-" + "-".join(div))
    stamps.append("POSITIVE-CONTROL-REPRODUCES" if gap >= GAP_MIN else "POSITIVE-CONTROL-FAILED")
    if gap >= GAP_MIN:
        stamps += ["%s-%s" % (a, states[a]) for a in HELD]
    for a in HELD:
        if M[a] - M["HEAD"] > MATCH:
            stamps.append(a + "-ABOVE-HEAD")
        if M[a] - M["k01"] < -NULL:
            stamps.append(a + "-BELOW-K01")
        if any(nonfin[(a, s)] for s in SEEDS):
            stamps.append("NONFINITE-" + a)
    if any(abs(M[a] - M["k01"]) <= NULL for a in HELD):
        stamps.append("FLOOR-READINGS-ARE-BOUNDS")
    agree = all((CT[l][1] > 0) == (CT[l][0] > 0) for l in ("P_HIGH", "P_COUP", "P_LOW") if abs(CT[l][0]) >= RESCUE)
    stamps.append("TRAIN-AGREES" if agree else "TRAIN-DISAGREES")
    final = "FINAL: %s | %s" % (branch, " | ".join(stamps))
    print("  states: %s" % "  ".join("%s %s" % (a, states[a]) for a in HELD))
    print("  " + final)
    exp("G-DIVERGE", "  %s every arm's seed range <= 5.0 pp" % ("PASS" if not div else "FAIL"))
    margins = [("D_HEAD over GAP_MIN", gap - GAP_MIN), ("k01 under K01_MAX", K01_MAX - M["k01"]),
               ("worst seed range under DIVERGED (%s)" % max(ARMS, key=lambda a: max(P[a]) - min(P[a])),
                DIVERGED - max(max(P[a]) - min(P[a]) for a in ARMS))]
    for a in HELD:
        margins.append(("%s: L - (HEAD - MATCH)   (>= 0 is AT-HEAD)" % a, M[a] - (M["HEAD"] - MATCH)))
        margins.append(("%s: (k01 + NULL) - L     (>= 0 is AT-K01)" % a, M["k01"] + NULL - M[a]))
    for k, v in margins:
        print("    margin %-48s %+9.4f pp = %+.2f SE" % (k, v, v / se))
    print("  per-seed readings against the in-batch k01 mean + NULL and HEAD mean - MATCH (descriptive; the branch reads arm means):")
    for a in HELD:
        print("    %-10s %s" % (a, " / ".join("s%d %.4f %s" % (s, P[a][j], "AT-HEAD" if P[a][j] >= M["HEAD"] - MATCH else
                                                                 ("AT-K01" if P[a][j] <= M["k01"] + NULL else "BETWEEN"))
                                           for j, s in enumerate(SEEDS))))
    logl = [ln.rstrip("\n") for ln in open(slog)]
    fl = [ln for ln in logl if ln.startswith("FINAL:")]
    print("  scorer log FINAL == this FINAL (whole string): %s" % (fl == [final]))
    if fl:
        toks = [t.strip() for t in fl[-1][len("FINAL:"):].split("|")]
        mine = [branch] + stamps
        print("  branch token equal: %s;  stamp lists equal as lists: %s (scorer %d tokens, here %d)"
              % (toks[0] == branch, toks == mine, len(toks), len(mine)))
        if toks != mine:
            viol("FINAL tokens differ: scorer %s / here %s" % (sorted(set(toks) - set(mine)), sorted(set(mine) - set(toks))))
    else:
        viol("no FINAL line in the scorer log")
    exp("BETWEEN", "  cvt4 {90,91,92}: %s" % "  ".join("%s %.4f" % (a, M[a]) for a in ARMS))

    # ---------------------------------------------------------------- [6]
    print("\n[6] the scorer's descriptive readouts, re-derived (beta peak / pin (<= -15+1e-4 for good) / mean r epochs 15-30)")
    SUMM = {}
    for a in ARMS:
        for s in SEEDS:
            recs = RECS[(a, s)]
            parts = []
            ng = 1 if a == "k01" else 2
            for k in range(ng):
                pk, pke, pin, r = traj_summary(recs, k)
                SUMM[(a, s, k)] = (pk, pke, pin, r)
                lab = "beta" if ng == 1 else ("cmpl" if k == 0 else "b50")
                parts.append("%s peak %.3f @%.1f pin %s r %.0f" % (lab, pk, pke, "never" if pin is None else "%.1f" % pin, r))
            acts = [x["bh_active"] for x in recs if "bh_active" in x]
            line = "  %-10s s%d  %s  %s" % (a, s, " | ".join(parts), "bh_active %s" % (acts[-1] if acts else "-"))
            print(line)
            exp("DESCRIPTIVE", line)
    for a in ARMS:
        cells = []
        for e in (9, 19, 29, 39, 49, 99):
            cells.append("%.1f (%.1f)" % (mean([R[(a, s)]["ep"][e][1] for s in SEEDS]), mean([R[(a, s)]["ep"][e][0] for s in SEEDS])))
        exp("DESCRIPTIVE", "  %-10s %s" % (a, " / ".join(cells)))

    # ---------------------------------------------------------------- [7]
    print("\n[7] LINE-BY-LINE AGREEMENT with the scorer log (each re-built line looked up verbatim)")
    logset = set(logl)
    by = {}
    for sec, line in expect:
        hit = line in logset
        by.setdefault(sec, [0, 0])
        by[sec][0 if hit else 1] += 1
        if not hit:
            viol("line not in scorer log [%s]: %r" % (sec, line))
    for sec in ("COMPLETE", "G-ARGS", "G-ENV", "G-VOTEW", "G-HOLDW", "G-STRUCT", "G-PROV", "LEVELS", "G-FLOOR", "G-CEIL",
                "G-BITE", "SIGMA", "CONTRASTS", "G-DIVERGE", "BETWEEN", "DESCRIPTIVE"):
        h, m = by.get(sec, [0, 0])
        print("  %-12s %3d lines found verbatim, %d not found" % (sec, h, m))
    n_fail = sum(1 for ln in logl if ln.lstrip().startswith("FAIL"))
    print("  total %d re-built lines, %d found; FAIL lines in the scorer log: %d" % (len(expect), sum(v[0] for v in by.values()), n_fail))

    # ================================================================ DESCRIPTIVE
    print("\n" + "=" * 100)
    print("DESCRIPTIVE, NOT REGISTERED -- nothing below can move a level, contrast, gate, branch or stamp above")
    print("=" * 100)
    med = {}
    for a in ARMS:
        order = sorted(range(3), key=lambda j: P[a][j])
        med[a] = SEEDS[order[1]]
    print("\n[D1] APPLIED beta after the update that ends each epoch (probe record at step 500e; the value the next update"
          " uses), MEDIAN seed per arm (by plateau5): %s" % "  ".join("%s s%d" % (a, med[a]) for a in ARMS))
    print("     c = complement (group 0), 50 = layer4.1.bn2.weight (group 1); k01 has one shared beta (50 rides it)")
    hdr = "  ep  " + " ".join("%-15s" % ("k01" if a == "k01" else a) for a in ARMS)
    print(hdr)
    print("      " + " ".join("%-15s" % ("shared" if a == "k01" else "c / 50") for a in ARMS))
    for e in range(0, 41, 2):
        cells = []
        for a in ARMS:
            x = RECS[(a, med[a])][5 * e]
            if x["step"] != SPE * e:
                viol("record index %d is not epoch %d" % (5 * e, e))
            if a == "k01":
                cells.append("%-15s" % ("%7.3f" % x["beta"][0]))
            else:
                cells.append("%-15s" % ("%7.3f/%7.3f" % (x["beta"][0], x["beta"][1])))
        print("  %2d  %s" % (e, " ".join(cells)))
    print("  mean r = exp(beta+15) over epochs [15,30), arm mean of the three seeds (complement / 50):")
    for a in ARMS:
        if a == "k01":
            print("    %-10s shared %9.1f" % (a, mean([SUMM[(a, s, 0)][3] for s in SEEDS])))
        else:
            print("    %-10s complement %9.1f   50 %9.1f" % (a, mean([SUMM[(a, s, 0)][3] for s in SEEDS]),
                                                           mean([SUMM[(a, s, 1)][3] for s in SEEDS])))
    print("  complement beta, arm mean at epochs 18 / 20 / 25 / 30 / 35 / 40 / 50 / 60 / 80 / 99.8:")
    for a in ARMS:
        vals = []
        for e in (18, 20, 25, 30, 35, 40, 50, 60, 80):
            vals.append("%.2f" % mean([RECS[(a, s)][5 * e]["beta"][0] for s in SEEDS]))
        vals.append("%.2f" % mean([RECS[(a, s)][-1]["beta"][0] for s in SEEDS]))
        print("    %-10s %s" % (a, " / ".join(vals)))

    print("\n[D2] TEST (TRAIN) after 15 / 30 / 50 / 100 completed epochs (Epoch lines 14 / 29 / 49 / 99): arm mean, then per seed")
    for a in ARMS:
        cells = ["%.2f (%.2f)" % (mean([R[(a, s)]["ep"][e][1] for s in SEEDS]), mean([R[(a, s)]["ep"][e][0] for s in SEEDS]))
                 for e in (14, 29, 49, 99)]
        per = ["s%d %s" % (s, "/".join("%.2f" % R[(a, s)]["ep"][e][1] for e in (14, 29, 49, 99))) for s in SEEDS]
        print("  %-10s %s   | %s" % (a, "  ".join(cells), "  ".join(per)))

    print("\n[D3] when the level is set.  settle = first epoch line after which TEST stays within 0.5 pp of plateau5;"
          " share = TEST / plateau5 at Epoch lines 14 / 19 / 24 / 29")
    for a in ARMS:
        am = [mean([R[(a, s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
        settle = next(e for e in range(100) if all(abs(v - M[a]) <= 0.5 for v in am[e:]))
        ps = []
        for j, s in enumerate(SEEDS):
            tv = [R[(a, s)]["ep"][e][1] for e in range(100)]
            ps.append(next(e for e in range(100) if all(abs(v - P[a][j]) <= 0.5 for v in tv[e:])))
        share = " / ".join("%.2f" % (am[e] / M[a]) for e in (14, 19, 24, 29))
        print("  %-10s arm-mean settle %2d  per seed %s  share %s" % (a, settle, "/".join(str(x) for x in ps), share))
    hm = [mean([R[("HEAD", s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
    for a in ("HOLDHIGH", "HOLDSHARED"):
        am = [mean([R[(a, s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
        gaps = [hm[e] - am[e] for e in range(100)]
        first2 = next(e for e in range(100) if all(g > 2.0 for g in gaps[e:]))
        print("  HEAD - %-10s arm-mean TEST gap by Epoch line 10..30: %s ; first line after which it stays > 2 pp: %d"
              % (a, " ".join("%d:%+.1f" % (e, gaps[e]) for e in range(10, 31, 2)), first2))
        dc = []
        for e in range(0, 41):
            hv = mean([RECS[("HEAD", s)][5 * e]["beta"][0] for s in SEEDS])
            av = mean([RECS[(a, s)][5 * e]["beta"][0] for s in SEEDS])
            dc.append(av - hv)
        firstc = next((e for e in range(41) if abs(dc[e]) > 0.1), None)
        print("  %-10s complement beta minus HEAD's (arm means) at epochs 10..40: %s ; first epoch |diff| > 0.1: %s"
              % (a, " ".join("%d:%+.2f" % (e, dc[e]) for e in range(10, 41, 2)), firstc))
    b50h = [mean([RECS[("HEAD", s)][5 * e]["beta"][1] for s in SEEDS]) for e in range(0, 41)]
    b50x = [mean([RECS[("HOLDHIGH", s)][5 * e]["beta"][1] for s in SEEDS]) for e in range(0, 41)]
    print("  tensor 50's beta, HOLDHIGH minus HEAD (arm means) at epochs 2..20: %s"
          % " ".join("%d:%+.2f" % (e, b50x[e] - b50h[e]) for e in range(2, 21, 2)))

    print("\n[D4] HOLDSHARED's live complement and HOLDHIGH's live complement against MUTE's measured shared trajectory"
          " (between batch: cvt1 MUTE s78-80, cvt3 MUTE50 s84-86; orientation only) and the replay v_9428")
    refs = [("cvt1", "MUTE", s) for s in (78, 79, 80)] + [("cvt3", "MUTE50", s) for s in (84, 85, 86)]
    have = all(os.path.isdir(os.path.join(runsdir, b, "probe_%s-%s-s%d" % (b, a, s))) for b, a, s in refs)
    rows = []
    if have:
        for b, a, s in refs:
            recs = load_recs(os.path.join(runsdir, b, "probe_%s-%s-s%d" % (b, a, s)))
            rows.append(("%s %s s%d" % (b, a, s), recs, 0))
    for a in ("HOLDSHARED", "HOLDHIGH"):
        for s in SEEDS:
            rows.append(("cvt4 %s s%d cmpl" % (a, s), RECS[(a, s)], 0))
    for s in SEEDS:
        rows.append(("cvt4 HEAD s%d cmpl" % s, RECS[("HEAD", s)], 0))
    for label, recs, k in rows:
        pk, pke, pin, r = traj_summary(recs, k)
        dev = max(abs(float(x["beta"][k]) - tri(x["step"] + 2, 9428)) for x in recs)
        dev40 = max(abs(float(x["beta"][k]) - tri(x["step"] + 2, 9428)) for x in recs if x["step"] <= 40 * SPE)
        print("  %-26s peak %.3f @%.1f  pin %-5s  r15-30 %8.1f   max|beta - v_9428| all %.3f, epochs 0-40 %.3f"
              % (label, pk, pke, "never" if pin is None else "%.1f" % pin, r, dev, dev40))
    if not have:
        print("  (MUTE reference probe dirs not under this runsdir; skipped)")

    print("\nVIOLATIONS %d" % len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
