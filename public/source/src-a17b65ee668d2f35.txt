#!/usr/bin/env python3
"""cvt5_attack_indep.py <runsdir> <scorer.py> <score.log> [<all_runs.csv> <CORPUS-EXCLUSIONS.tsv>]
-- independent re-derivation of `cvt5` (CORRECTIONS 238): PlainNet18_c100 at 300 epochs, arms k01 / HEAD / K13 / K33.

Imports NOTHING from analysis/cVT5_plainhorizon_score.py or any other repo module and uses no regex.  Reads, by
string splitting only: every raw cvt5-<arm>-s<seed>-<jobid>.out in <runsdir> (ARGS / ENV / PROBE_TENSOR / VOTE_W
lines, `Epoch` lines, RUN_DONE, Traceback), <runsdir>/cvt5/PARTITION-MANIFEST.txt and PROVENANCE.txt, each run's
probe dir (probe_tensor.json, block_sizes.json, probe.jsonl), and -- for the one corpus DISCLOSURE line only -- the
CSV and the exclusion TSV with the csv module.  plateau5 = mean TEST over the run's own epochs 95..99 (@100) and
295..299 (@300); TRAIN the same on the Train column.  Bars, the pin gate and the branch order are RE-TYPED from the
registration text (238.4 / 238.6); witnesses are BUILT from K and the manifest's names.

The scorer's printed lines are rebuilt here from these independent numbers, with the scorer's print formats, and
each must appear VERBATIM in <score.log> (agreement to the printed precision).  The FINAL line is rebuilt token by
token and must equal the log's.  No absolute path is printed, so the output is byte-comparable across hosts.

Sections:
  [0] manifest, provenance, scorer sha, corpus disclosure
  [1] completeness and the harness gates (G-ARGS, G-ENV, G-VOTEW, G-STRUCT, G-PROV), re-derived
  [2] levels @100 and @300, TEST beside TRAIN, tail slopes; G-FLOOR / G-CEIL
  [3] G-BITE re-derived (fsum), plus THE APPLIED STEP against the weighted and unweighted Lion directions
  [4] sigma, every contrast, per-seed pairs
  [5] G-DIVERGE, global gates, branches (a) (b) on TEST and TRAIN, level stamps
  [6] the pin gate (magnitude form), pin stamps, (c), floor stamps; FINAL rebuilt
  [7] the scorer's between-batch and descriptive readout, rebuilt
  [8] every bar margin
  [9] DESCRIPTIVE ONLY (cannot move anything): TEST / TRAIN at epochs 100..300, complement beta and r at the same
      epochs, pin epochs, K13's complement after its first r<=2, whether K13's TEST kept rising after 100
"""
import csv
import hashlib
import json
import math
import os
import sys

PREFIX = "cvt5"
NET = "PlainNet18_c100"
SEEDS = [93, 94, 95]
ARMS = ["k01", "HEAD", "K13", "K33"]
LADDER = ["K13", "K33"]
KOF = {"K13": 13, "K33": 33}
HEADSPEC = "sets:1-49,51-53/layer4.1.bn2.weight"
SPEC = {"k01": "scalar", "HEAD": HEADSPEC, "K13": HEADSPEC, "K33": HEADSPEC}
TWIN, HEAD_IDX = 47, 50
E, C = 300, 100
SPE, PROBE_EVERY = 500, 100
N_REC = E * SPE // PROBE_EVERY
ENV_WANT = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
            "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 "
            "EB_RHO=na EB_LOG=0")
FIXED = {"optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99", "weight-decay-base": "0.1",
         "alg-meta": "Lion", "momentum-param-meta": "0.99", "Lion-beta2-meta": "0.9", "weight-decay-meta": "0",
         "dataset": "CIFAR100", "NN-name": NET, "batch-size": "100", "max-time": "999:00:00", "gamma": "1",
         "num-epochs": "300", "meta-stepsize": "1e-3", "alpha0": "1e-6"}
# provenance prefixes, re-typed from CORRECTIONS 238.3 / 238.10
PROV_WANT = {"MODE": "submit", "BUILD_NETWORK_SHA256": "e65e6773", "HF_SHA256": "3f2b98e1",
             "RUNNER_SHA256": "d389e8a5", "TRAIN_SHA256": "3fea309e"}
MANIFEST_SHA_PREFIX = "d78c0322"      # 238.3: the live manifest, 3,634 bytes
# 238.4, re-typed
SIGMA_PRIOR = 0.8475179908088943
GAP_MIN = K01_MAX = 20.0
SHAPE_GAP, MATCH, NULL, DIVERGED = 15.0, 5.0, 2.0, 5.0
REFUTE, SUPPORT, SURVIVE, ATTEN_MILD = 5.0, 15.0, 0.80, 0.50
FLOOR_MIN, CEIL_MAX = 15.0, 90.0
PIN_R, FREE_R, TAILQ, POST_MIN = 2.0, 10.0, 0.25, 15.0
W_POST, PIN_BOUND = 64.2, 223.5
LO, HI, MS = -15.0, -2.3026, 1e-3
EXACT = 1e-3
BITE_TOL, INFORM, MIN_INF = 1e-4, 1e-2, 50
R_CVT2 = {"k01": 11.4767, "HEAD": 64.4187, "K13": 52.3493, "K33": 35.5933}


def mean(v):
    return math.fsum(v) / len(v)


def sdev(v):
    m = mean(v)
    return math.sqrt(math.fsum((x - m) ** 2 for x in v) / (len(v) - 1))


def rng(v):
    return max(v) - min(v)


def ols(xs, ys):
    xm, ym = mean(xs), mean(ys)
    return math.fsum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / math.fsum((x - xm) ** 2 for x in xs)


def sgn(x):
    return (x > 0) - (x < 0)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def f4(x):
    return "n/a" if x is None else "%.4f" % x


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
        elif "Traceback" in s:
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


def win(ep, end, k):
    return mean([ep[e][k] for e in range(end - 5, end)])


def main():
    runsdir, scorer, slog = sys.argv[1], sys.argv[2], sys.argv[3]
    csvp = sys.argv[4] if len(sys.argv) > 4 else None
    tsvp = sys.argv[5] if len(sys.argv) > 5 else None
    bad = []
    LOG = [ln.rstrip("\n") for ln in open(slog, errors="replace")]
    LOGSET = set(LOG)
    agree = [0, 0]

    def viol(msg):
        bad.append(msg)
        print("  VIOLATION " + msg)

    def expect(line):
        """a line the scorer must have printed verbatim"""
        if line in LOGSET:
            agree[0] += 1
        else:
            agree[1] += 1
            viol("scorer log lacks the re-derived line: %r" % line)

    base = os.path.join(runsdir, PREFIX)
    # ---------------------------------------------------------------- [0]
    names, numel = [], []
    mtxt = open(os.path.join(base, "PARTITION-MANIFEST.txt")).read()
    mkv = {}
    for ln in mtxt.splitlines():
        p = ln.split()
        if p and p[0] == "TENSOR":
            names.append(p[2])
            numel.append(int(p[3]))
        elif p:
            mkv.setdefault(p[0], []).append(" ".join(p[1:]))
    msha = hashlib.sha256(mtxt.encode()).hexdigest()
    print("[0] manifest: %d tensors, %d params; idx50 %s; idx47 %s; %d bytes sha %s"
          % (len(names), sum(numel), names[HEAD_IDX - 1], names[TWIN - 1], len(mtxt.encode()), msha[:16]))
    if (len(names) != 53 or sum(numel) != 11046308 or names[HEAD_IDX - 1] != "layer4.1.bn2.weight"
            or names[TWIN - 1] != "layer4.1.bn1.weight"):
        viol("manifest names / numels")
    prov = {}
    for ln in open(os.path.join(base, "PROVENANCE.txt")):
        p = ln.split(None, 1)
        if len(p) == 2:
            prov.setdefault(p[0], p[1].strip())
    s_sha = sha(scorer)
    print("    PROVENANCE MODE %s HF %s BUILD %s RUNNER %s TRAIN %s SCORER %s" % (
        prov.get("MODE"), prov.get("HF_SHA256", "")[:8], prov.get("BUILD_NETWORK_SHA256", "")[:8],
        prov.get("RUNNER_SHA256", "")[:8], prov.get("TRAIN_SHA256", "")[:8], prov.get("SCORER_SHA256", "")[:16]))
    print("    scorer file sha256 %s == PROVENANCE SCORER_SHA256: %s" % (s_sha, s_sha == prov.get("SCORER_SHA256")))
    if s_sha != prov.get("SCORER_SHA256"):
        viol("scorer sha != provenance")
    if csvp and tsvp:
        keys = set()
        for ln in open(tsvp):
            if ln.startswith("#") or ln.startswith("run\t") or not ln.strip():
                continue
            p = ln.rstrip("\n").split("\t")
            keys.add((p[0], p[1]))
        rows = list(csv.DictReader(open(csvp)))
        kept = [r for r in rows if (r["run"], r["job_id"]) not in keys and not r["run"].startswith(PREFIX + "-")]
        print("    corpus: csv sha %s rows %d, exclusion tsv sha %s keys %d -> %d rows (cvt5- dropped)"
              % (sha(csvp)[:16], len(rows), sha(tsvp)[:16], len(keys), len(kept)))
        expect("corpus: %d rows after corpus_exclusions.filter_rows, %s- excluded  (DISCLOSURE ONLY -- no bar, sigma, "
               "branch or stamp reads it)" % (len(kept), PREFIX))

    # ---------------------------------------------------------------- [1]
    print("\n[1] completeness and harness gates, re-derived (witnesses built from K and the manifest's names)")
    files = {}
    for fn in sorted(os.listdir(runsdir)):
        if not (fn.startswith(PREFIX + "-") and fn.endswith(".out")):
            continue
        stem = fn[:-4].split("-")
        if len(stem) != 4 or stem[1] not in ARMS:
            continue
        files.setdefault((stem[1], int(stem[2][1:])), []).append((int(stem[3]), fn))
    R = {}
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
            ok_ep = sorted(r["ep"]) == list(range(E)) and r["dup"] == 0
            f = flags_of(r["args"][0]) if len(r["args"]) == 1 else []
            ks = [k for k, _ in f]
            fd = dict(f)
            want = dict(FIXED)
            want.update({"stepsize-groups": SPEC[a], "seed": str(s), "run-name": "%s-%s-s%d" % (PREFIX, a, s)})
            wrong = sorted(k for k, v in want.items() if fd.get(k) != v)
            rep = sorted(set(k for k in ks if ks.count(k) > 1))
            env = [" ".join(t for t in e_.split() if not t.startswith("PROBE_DIR=")) for e_ in r["env"]]
            ptw = "PROBE_TENSOR: on every=100 type=%s tensors=53 " % ("scalar" if a == "k01" else "blockwise")
            pt_ok = len(r["pt"]) == 1 and r["pt"][0].startswith(ptw)
            vw_ok = r["vw"] == [witness(a, names)]
            ok = ok_ep and r["done"] and r["tb"] == 0 and len(r["args"]) == 1 and not wrong and not rep
            ok = ok and env == [ENV_WANT] and pt_ok and vw_ok
            print("  %-4s %-5s s%d job %d  epochs %d (0..%d, dup %d)  RUN_DONE %s  tb %d  ARGS %d flags, wrong %s rep %s"
                  "  ENV %s  PT %s  VOTE_W %s"
                  % ("OK" if ok else "BAD", a, s, jid, len(r["ep"]), max(r["ep"]), r["dup"], r["done"], r["tb"],
                     len(f), wrong or "-", rep or "-", env == [ENV_WANT], pt_ok, vw_ok))
            if not ok:
                viol("%s-s%d header/completeness" % (a, s))
            expect("  OK   %s-%s-s%d  %d epoch lines, RUN_DONE=True, traceback=False" % (PREFIX, a, s, len(r["ep"])))
            expect("  PASS G-ARGS %s-s%d" % (a, s))
            expect("  PASS G-VOTEW %s-s%d" % (a, s))
    expect("  PASS G-ENV one distinct ENV line, one per run   1 distinct")
    expect("  PASS G-ENV every run printed ONE `PROBE_TENSOR: on every=100 type=<its arm's> tensors=53`   12 runs")
    # G-STRUCT, rebuilt from the registration's arm table (not from the scorer's generator)
    arml = []
    for a in ARMS:
        g = groups_of(SPEC[a], names)
        if a == "k01":
            arml.append("ARMSPEC k01 SPEC scalar TYPE scalar M 1")
        else:
            arml.append("ARMSPEC %s SPEC %s TYPE blockwise M 2 SIZES %d,%d PARAMS %d,%d GROUP1 %s"
                        % (a, SPEC[a], len(g[0]), len(g[1]), sum(numel[i - 1] for i in g[0]),
                           sum(numel[i - 1] for i in g[1]), names[g[1][0] - 1]))
    struct_ok = (msha.startswith(MANIFEST_SHA_PREFIX) and mkv.get("NETWORK") == [NET] and mkv.get("EPOCHS") == ["300"]
                 and mkv.get("CTRL_EPOCHS") == ["100"] and mkv.get("META_STEPS") == ["150000"]
                 and mkv.get("NUM_PARAM_TENSORS") == ["53"]
                 and ["ARMSPEC " + x for x in mkv.get("ARMSPEC", [])] == arml
                 and mkv.get("WITNESS") == ["%s %s" % (a, witness(a, names)) for a in ARMS]
                 and mkv.get("VOTEW") == ["k01 off", "HEAD off", "K13 layer4.1.bn1.weight:13", "K33 layer4.1.bn1.weight:33"])
    print("  G-STRUCT manifest sha %s… == 238.3's %s…, EPOCHS/CTRL_EPOCHS/META_STEPS, ARMSPEC/VOTEW/WITNESS rebuilt: %s"
          % (msha[:8], MANIFEST_SHA_PREFIX, struct_ok))
    if not struct_ok:
        viol("G-STRUCT")
    nb = len(mtxt.encode())
    expect("  PASS G-STRUCT byte-identical to synthetic_manifest_text() (53 tensors, EPOCHS 300, CTRL_EPOCHS 100, "
           "the VOTE_W table)   %d vs %d bytes" % (nb, nb))
    prov_ok = all(prov.get(k, "").startswith(v) for k, v in PROV_WANT.items())
    print("  G-PROV MODE / build_network / HF / RUNNER / TRAIN prefixes from 238.10, SCORER == file: %s"
          % (prov_ok and s_sha == prov.get("SCORER_SHA256")))
    if not prov_ok:
        viol("G-PROV")
    expect("  PASS G-PROV MODE submit   submit")
    expect("  PASS G-PROV SCORER_SHA256 == this scorer's sha256 %s...   %s" % (s_sha[:12], s_sha[:16]))
    nfail = sum(1 for ln in LOG if ln.startswith("  FAIL"))
    npass = sum(1 for ln in LOG if ln.startswith("  PASS"))
    print("  scorer log: %d PASS lines, %d FAIL lines" % (npass, nfail))
    if nfail:
        viol("scorer log carries FAIL lines")

    # ---------------------------------------------------------------- [2]
    print("\n[2] levels: plateau5 TEST @100 (epochs 95..99) and @300 (295..299), TRAIN beside")
    V1, VE, T1, TE = {}, {}, {}, {}
    for a in ARMS:
        eps = [R[(a, s)]["ep"] for s in SEEDS]
        V1[a] = [win(ep, C, 1) for ep in eps]
        VE[a] = [win(ep, E, 1) for ep in eps]
        T1[a] = [win(ep, C, 0) for ep in eps]
        TE[a] = [win(ep, E, 0) for ep in eps]
        sl1 = [ols(list(range(80, 100)), [ep[e][1] for e in range(80, 100)]) for ep in eps]
        slE = [ols(list(range(280, 300)), [ep[e][1] for e in range(280, 300)]) for ep in eps]
        l1 = ("  %-5s TEST @100 %.4f (sd %.4f, range %.4f)  @300 %.4f (sd %.4f, range %.4f)   TRAIN @100 %.4f  @300 %.4f"
              % (a, mean(V1[a]), sdev(V1[a]), rng(V1[a]), mean(VE[a]), sdev(VE[a]), rng(VE[a]), mean(T1[a]), mean(TE[a])))
        l2 = "        seeds @100: %s" % ", ".join("s%d %.4f (train %.4f)" % (s, x, y) for s, x, y in zip(SEEDS, V1[a], T1[a]))
        l3 = "        seeds @300: %s" % ", ".join("s%d %.4f (train %.4f)" % (s, x, y) for s, x, y in zip(SEEDS, VE[a], TE[a]))
        l4 = ("        TEST OLS slope pp/ep, epochs 80-99: %s;  epochs 280-299: %s"
              % ("/".join("%.4f" % x for x in sl1), "/".join("%.4f" % x for x in slE)))
        for ln in (l1, l2, l3, l4):
            print(ln)
            expect(ln)
    M1 = dict((a, mean(V1[a])) for a in ARMS)
    ME = dict((a, mean(VE[a])) for a in ARMS)
    MT1 = dict((a, mean(T1[a])) for a in ARMS)
    MTE = dict((a, mean(TE[a])) for a in ARMS)
    hi = max(M1.values())
    print("  G-FLOOR max arm mean @100 %.4f >= %.1f: %s;  G-CEIL max over both horizons %.4f <= %.1f: %s"
          % (hi, FLOOR_MIN, hi >= FLOOR_MIN, max(list(M1.values()) + list(ME.values())), CEIL_MAX,
             max(list(M1.values()) + list(ME.values())) <= CEIL_MAX))
    expect("  PASS G-FLOOR max arm mean >= 15.00 pp (chance 1.00)   max %.4f" % hi)
    if not (hi >= FLOOR_MIN and max(list(M1.values()) + list(ME.values())) <= CEIL_MAX):
        viol("G-FLOOR/G-CEIL")

    # ---------------------------------------------------------------- [3]
    print("\n[3] G-BITE re-derived (math.fsum), and THE APPLIED STEP: on every interior record where a group's beta moved,"
          "\n    step sign == -sign(b2*mom_pre + (1-b2)*z_agg) [weighted]; where weighted and UNWEIGHTED directions disagree,"
          "\n    which one the step followed")
    PR = {}
    for s in SEEDS:
        for a in ARMS:
            gs = groups_of(SPEC[a], names)
            w = weights(a)
            pdir = os.path.join(base, "probe_%s-%s-s%d" % (PREFIX, a, s))
            ptj = json.load(open(os.path.join(pdir, "probe_tensor.json")))
            bsz = json.load(open(os.path.join(pdir, "block_sizes.json")))
            recs = [json.loads(x) for x in open(os.path.join(pdir, "probe.jsonl")) if x.strip()]
            PR[(a, s)] = recs
            nb_want = [sum(numel[i - 1] for i in g) for g in gs]
            steps_ok = [x["step"] for x in recs] == list(range(0, E * SPE, PROBE_EVERY))
            meta_ok = (ptj.get("n_tensors") == 53 and ptj.get("param_numels") == numel and bsz.get("n_b") == nb_want
                       and len(recs) == N_REC and steps_ok)
            wz = wm = 0.0
            n_bad = n_badz = n_badm = n_inf = n_um = n_mov = n_movw = n_dis = n_fw = n_fu = 0
            for x in recs:
                try:
                    zt = [float(v) for v in x["z_tensor"]]
                    mt = [float(v) for v in x["m_tensor"]]
                    za_all, ma_all = x["z_agg"][0], x["mom_pre"][0]
                    good = (len(zt) == 53 and len(mt) == 53 and len(za_all) == len(gs) and len(ma_all) == len(gs)
                            and len(x["beta"]) == len(gs)
                            and all(math.isfinite(v) for v in zt + mt + za_all + ma_all + x["beta"]))
                except (KeyError, TypeError, ValueError):
                    good = False
                if not good:
                    n_bad += 1
                    continue
                b2 = x["pt_b2"]
                ez = em = 0.0
                rinf, rum = False, True
                for k, g in enumerate(gs):
                    sz = math.fsum(abs(w.get(i, 1.0) * zt[i - 1]) for i in g) + 1e-30
                    sm = math.fsum(abs(w.get(i, 1.0) * mt[i - 1]) for i in g) + 1e-30
                    zw = math.fsum(w.get(i, 1.0) * zt[i - 1] for i in g)
                    mw = math.fsum(w.get(i, 1.0) * mt[i - 1] for i in g)
                    zu = math.fsum(zt[i - 1] for i in g)
                    mu = math.fsum(mt[i - 1] for i in g)
                    za, ma = za_all[k], ma_all[k]
                    ez = max(ez, abs(zw - za) / sz)
                    em = max(em, abs(mw - ma) / sm)
                    if w and abs(zw - zu) / sz > INFORM:
                        rinf = True
                        rum = rum and abs(zu - za) / sz <= BITE_TOL
                    cw = b2 * ma + (1 - b2) * za
                    cu = b2 * mu + (1 - b2) * zu
                    bp, bn = x["beta_pre"][0][k], x["beta"][k]
                    st = bn - bp
                    if LO + 2 * MS < bp < HI - 2 * MS and abs(st) > 0.5 * MS:
                        n_mov += 1
                        n_movw += sgn(st) == -sgn(cw)
                        if sgn(cw) != sgn(cu):
                            n_dis += 1
                            n_fw += sgn(st) == -sgn(cw)
                            n_fu += sgn(st) == -sgn(cu)
                wz, wm = max(wz, ez), max(wm, em)
                n_badz += ez > BITE_TOL
                n_badm += em > BITE_TOL
                n_inf += rinf
                n_um += rinf and rum
            ok = meta_ok and n_bad == 0 and n_badz == 0 and n_badm == 0
            if w:
                ok = ok and n_inf >= MIN_INF and n_um == 0
            else:
                ok = ok and n_inf == 0
            applied_ok = n_movw == n_mov and (n_fw == n_dis and n_fu == 0 if w else n_dis == 0)
            print("  %-4s %-5s s%d recs %d steps 0..149900 %s meta %s bad %d | worst z %.2e m %.2e | informative %d "
                  "unweighted-match %d | moved interior %d follow-weighted %d | DISAGREE %d: weighted %d unweighted %d"
                  % ("OK" if ok and applied_ok else "BAD", a, s, len(recs), steps_ok, meta_ok, n_bad, wz, wm, n_inf, n_um,
                     n_mov, n_movw, n_dis, n_fw, n_fu))
            if not ok:
                viol("%s-s%d G-BITE" % (a, s))
            if not applied_ok:
                viol("%s-s%d applied step" % (a, s))
            expect("  %-4s G-BITE %-5s s%d  records %d  bad %d  worst z %.1e m %.1e  informative %d  unweighted-match %d"
                   % ("PASS" if ok else "FAIL", a, s, len(recs), n_bad, wz, wm, n_inf, n_um))

    # ---------------------------------------------------------------- [4]
    print("\n[4] sigma and contrasts (within batch; @100 and @300 in run)")
    ss = math.fsum((x - mean(v)) ** 2 for a in ARMS for v in (V1[a], VE[a]) for x in v)
    dfi = len(ARMS) * 2 * (len(SEEDS) - 1)
    sig_in = math.sqrt(ss / dfi)
    sig = max(SIGMA_PRIOR, sig_in)
    se = sig * math.sqrt(2.0 / 3.0)
    rb = 2.0 * se
    L = ["  SIGMA_PRIOR_frozen   %.6f" % SIGMA_PRIOR, "  SIGMA_INBATCH        %.6f" % sig_in,
         "  SIGMA_USED           %.6f  (= %s)" % (sig, "SIGMA_PRIOR_frozen" if sig == SIGMA_PRIOR else "SIGMA_INBATCH"),
         "  SE_ARM_DIFF          %.6f  = SIGMA_USED * sqrt(2/3)  (df_inbatch %d)" % (se, dfi),
         "  READ_BAR             %.6f  = 2 SE (stamp-level comparisons only)" % rb]
    d1 = M1["HEAD"] - M1["k01"]
    dE = ME["HEAD"] - ME["k01"]
    g13 = ME["K13"] - ME["K33"]
    drop = ME["K13"] - M1["K13"]
    rho = dE / d1
    L.append("  KEY      D_HEAD@100 = %+.4f pp = %+.2f SE (TRAIN %+.4f)   D_HEAD@300 = %+.4f pp = %+.2f SE (TRAIN %+.4f)"
             % (d1, d1 / se, MT1["HEAD"] - MT1["k01"], dE, dE / se, MTE["HEAD"] - MTE["k01"]))
    L.append("  PREMISE  @100: HEAD - K13 %+.4f | K13 - K33 %+.4f | HEAD - K33 %+.4f  (cvt2: +12.0693 / +16.7560 / +28.8253)"
             % (M1["HEAD"] - M1["K13"], M1["K13"] - M1["K33"], M1["HEAD"] - M1["K33"]))
    L.append("  PRIMARY  (a) G13@300 = K13 - K33 = %+.4f pp = %+.2f SE (TRAIN %+.4f)   [@100 %+.4f]"
             % (g13, g13 / se, MTE["K13"] - MTE["K33"], M1["K13"] - M1["K33"]))
    L.append("  PRIMARY  (a) DROP13 = K13@300 - K13@100 = %+.4f pp = %+.2f SE (TRAIN %+.4f);  HEAD@300 - K13@300 %+.4f"
             % (drop, drop / se, MTE["K13"] - MT1["K13"], ME["HEAD"] - ME["K13"]))
    L.append("  PRIMARY  (b) RHO = D_HEAD@300 / D_HEAD@100 = %.4f   (SURVIVE needs RHO >= 0.80 AND D_HEAD@300 >= 15.0; "
             "COLLAPSE <= 5.0)" % rho)
    for i, s in enumerate(SEEDS):
        a1 = V1["HEAD"][i] - V1["k01"][i]
        a2 = VE["HEAD"][i] - VE["k01"][i]
        L.append("           seed %d paired (descriptive): D_HEAD %+.4f -> %+.4f rho %.4f | K13 %+.4f -> %+.4f (drop %+.4f) "
                 "| K13 - K33 @300 %+.4f" % (s, a1, a2, a2 / a1, V1["K13"][i], VE["K13"][i], VE["K13"][i] - V1["K13"][i],
                                             VE["K13"][i] - VE["K33"][i]))
    L.append("  KEY      K33@300 - K33@100 %+.4f | HEAD@300 - HEAD@100 %+.4f | k01@300 - k01@100 %+.4f"
             % (ME["K33"] - M1["K33"], ME["HEAD"] - M1["HEAD"], ME["k01"] - M1["k01"]))
    L.append("  bars: GAP_MIN 20.0 | SHAPE_GAP 15.0 | MATCH 5.0 pp (%.2f SE) | REFUTE 5.0 | SUPPORT 15.0 | SURVIVE 0.80 | "
             "NULL 2.0" % (MATCH / se))
    for ln in L:
        print(ln)
        expect(ln)
    print("  (registration header literal READ_BAR_PRIOR 1.383992 vs computed 2*SIGMA_PRIOR*sqrt(2/3) = %.9f: a 6th-decimal"
          " rounding of the header text only; the scorer computes it)" % (2 * SIGMA_PRIOR * math.sqrt(2.0 / 3.0)))

    # ---------------------------------------------------------------- [5]
    print("\n[5] divergence, global gates, branches (238.6, re-typed)")
    div1 = [a for a in ARMS if rng(V1[a]) > DIVERGED]
    divE = [a for a in ARMS if rng(VE[a]) > DIVERGED]
    expect("  %s every arm's seed range @100 <= 5.0 pp%s" % ("PASS" if not div1 else "FAIL",
                                                         "" if not div1 else "  diverged: %s" % div1))
    expect("  %s every arm's seed range @300 <= 5.0 pp%s" % ("PASS" if not divE else "NOTE",
                                                         "" if not divE else "  diverged: %s (per-branch, below)" % divE))
    glob = None
    if div1:
        glob = "UNRESOLVED-DIVERGED"
    elif M1["k01"] > K01_MAX:
        glob = "SCALAR-NOT-COLLAPSED"
    elif d1 < GAP_MIN:
        glob = "POSITIVE-CONTROL-FAILED"
    print("  diverged @100 %s  @300 %s  global gate %s" % (div1 or "none", divE or "none", glob or "none"))

    def dec_a(m1, me, dk13, dk33):
        if m1["HEAD"] - m1["K33"] < SHAPE_GAP or m1["HEAD"] - m1["K13"] <= MATCH or m1["K13"] - m1["K33"] <= MATCH:
            return "K13-PREMISE-NOT-REPLICATED"
        if dk33:
            return "UNRESOLVED-K33-DIVERGED"
        if dk13:
            return "UNRESOLVED-K13-SPLIT"
        gg = me["K13"] - me["K33"]
        dd = me["K13"] - m1["K13"]
        if gg < -MATCH:
            return "K13-BELOW-K33"
        if gg <= MATCH:
            return "GRADED-WAS-A-DELAY"
        if me["K13"] - me["HEAD"] > MATCH:
            return "K13-ABOVE-HEAD"
        if dd >= MATCH and me["HEAD"] - me["K13"] <= MATCH:
            return "K13-RECOVERS-TO-HEAD"
        if dd >= -MATCH:
            return "K-DEPENDENT-EQUILIBRIUM"
        return "GRADED-PARTLY-A-DELAY"

    def dec_b(m1, me, dh):
        x1, x2 = m1["HEAD"] - m1["k01"], me["HEAD"] - me["k01"]
        rr = x2 / x1
        if dh:
            return "UNRESOLVED-HEAD-DIVERGED", rr
        if x2 <= REFUTE:
            return "RESCUE-COLLAPSES", rr
        if rr >= SURVIVE and x2 >= SUPPORT:
            return "RESCUE-SURVIVES", rr
        return "RESCUE-ATTENUATES", rr

    def three(dv, lo_, mid_, hi_, bar):
        return lo_ if dv <= -bar else (hi_ if dv >= bar else mid_)

    st = ["HARNESS-CLEAN", "PATCH-BITES", "ONE-NETWORK-PLAINNET", "HORIZON-300", "NO-TIME-GATE", "ONE-CELL-ONLY",
          "SIGMA-PRIOR-FROZEN" if sig == SIGMA_PRIOR else "SIGMA-INBATCH"]
    A = dec_a(M1, ME, "K13" in divE, "K33" in divE)
    B, rho_b = dec_b(M1, ME, "HEAD" in divE or "k01" in divE)
    At = dec_a(MT1, MTE, False, False)
    Bt, rho_t = dec_b(MT1, MTE, False)
    print("  (a) %s   TRAIN %s" % (A, At))
    print("  (b) %s   RHO %.4f   TRAIN %s RHO %.4f" % (B, rho_b, Bt, rho_t))
    if glob is None:
        expect("  (a) %s   (TRAIN, same map: %s)" % (A, At))
        expect("  (b) %s   RHO %.4f   (TRAIN, same map: %s, RHO %.4f)" % (B, rho_b, Bt, rho_t))
    st.append("RHO:%.4f" % rho_b)
    if B == "RESCUE-ATTENUATES":
        st.append("ATTENUATE-MILD" if rho_b >= ATTEN_MILD else "ATTENUATE-SEVERE")
    st.append("TRAIN-AGREES-A" if At == A else "TRAIN-DISAGREES-A")
    st.append("TRAIN-AGREES-B" if Bt == B else "TRAIN-DISAGREES-B")
    st.append(three(drop, "K13-LEVEL-FALLS", "K13-LEVEL-HOLDS", "K13-LEVEL-RISES", rb))
    st.append(three(ME["HEAD"] - M1["HEAD"], "HEAD-LEVEL-FALLS", "HEAD-LEVEL-HOLDS", "HEAD-LEVEL-RISES", rb))
    st.append(three(ME["K33"] - M1["K33"], "K33-LEVEL-FALLS", "K33-LEVEL-HOLDS", "K33-LEVEL-RISES", rb))

    # ---------------------------------------------------------------- [6]
    print("\n[6] the pin gate: r = exp(beta[0]+15); median r over the last quarter of records; PINNED <= 2, FREE >= 10;"
          "\n    first r<=2; exact clamp = every later record within 1e-3 of -15; post-100 = records at epoch >= 100")
    lr2 = LO + math.log(PIN_R)
    GS = {}
    for a in ARMS:
        for s in SEEDS:
            seq = [(x["step"] / float(SPE), float(x["beta"][0])) for x in PR[(a, s)]]
            n = len(seq)
            q = seq[n - max(1, int(n * TAILQ)):]
            mr = med([math.exp(v + 15.0) for _e, v in q])
            f2 = None
            for e_, v in seq:
                if v <= lr2:
                    f2 = e_
                    break
            pf = None
            for j in range(n - 1, -1, -1):          # scan backwards: the earliest j with every later record at the clamp
                if seq[j][1] > LO + EXACT:
                    break
                pf = seq[j][0]
            post = [v for e_, v in seq if e_ >= C]
            g = {"med_r": mr, "f2": f2, "pf": pf, "maxr100": max(math.exp(v + 15.0) for v in post),
                 "off100": sum(1 for v in post if v > LO + EXACT), "term": seq[-1][1], "seq": seq,
                 "cls": "PINNED" if mr <= PIN_R else ("FREE" if mr >= FREE_R else "NEITHER")}
            GS[(a, s)] = g
            ln = ("  %-5s s%d group0 median r %.4f -> %-8s first r<=2 %s  exact-clamp from %s  max r after ep 100 %.4f  "
                  "off-exact records after ep 100 %d  terminal %.4f"
                  % (a, s, mr, g["cls"], "never" if f2 is None else "%.1f" % f2, "never" if pf is None else "%.1f" % pf,
                     g["maxr100"], g["off100"], g["term"]))
            print(ln)
            expect(ln)

    def arm_cls(cl):
        for k in ("PINNED", "FREE"):
            if sum(1 for c in cl if c == k) >= 2:
                return k
        return "NEITHER"

    def pin_set(tag, level_E):
        cl = [GS[(tag, s)]["cls"] for s in SEEDS]
        ac = arm_cls(cl)
        out = ["%s-COMPLEMENT-%s" % (tag, ac)]
        info = {}
        if ac == "PINNED":
            pins = [GS[(tag, s)]["f2"] for s, c in zip(SEEDS, cl) if c == "PINNED" and GS[(tag, s)]["f2"] is not None]
            latest = max(pins)
            w_ = E - latest
            out += ["%s-PIN-EPOCH:%.1f" % (tag, sum(pins) / len(pins)), "%s-POST-PIN-WINDOW:%.1f" % (tag, w_),
                    "%s-WINDOW-MEETS-W100" % tag if w_ >= W_POST else "%s-WINDOW-SHORT" % tag]
            if tag == "K13":
                out.append("K13-PIN-INSIDE-BOUND" if latest <= PIN_BOUND else "K13-PIN-AFTER-BOUND")
            if w_ >= POST_MIN:
                e0 = int(math.ceil(latest)) + 5
                postlev = mean([mean([R[(tag, s)]["ep"][e][1] for e in range(e0, e0 + 5)]) for s in SEEDS])
                out.append(three(level_E - postlev, "%s-DECAYS-AFTER-PIN" % tag, "%s-FROZEN-AT-PIN" % tag,
                                 "%s-IMPROVES-AFTER-PIN" % tag, rb))
                info = {"latest": latest, "e0": e0, "post": postlev, "d": level_E - postlev}
            else:
                out.append("%s-POST-PIN-UNMEASURED" % tag)
        else:
            if tag == "K13":
                out.append("UNRESOLVED-NOT-PINNED")
            out.append("%s-POST-PIN-UNMEASURED" % tag)
        return out, info

    k13st, k13i = pin_set("K13", ME["K13"])
    hst, hi_ = pin_set("HEAD", ME["HEAD"])
    for tag, info in (("K13", k13i), ("HEAD", hi_)):
        if info:
            print("  %s: latest pin %.1f -> post-pin level epochs %d..%d = %.4f; @300 - that = %+.4f vs READ_BAR %.6f"
                  % (tag, info["latest"], info["e0"], info["e0"] + 4, info["post"], info["d"], rb))
    st += k13st + hst
    k33c = [GS[("K33", s)]["cls"] for s in SEEDS]
    k33m = [GS[("K33", s)]["maxr100"] for s in SEEDS]
    k33o = [GS[("K33", s)]["off100"] for s in SEEDS]
    rise = ME["K33"] - M1["K33"]
    if "K33" in divE:
        Cc = "K33-SPLIT-AT-300"
    elif any(x >= FREE_R for x in k33m) or rise >= MATCH:
        Cc = "K33-REOPENS"
    elif arm_cls(k33c) == "PINNED" and all(x <= PIN_R for x in k33m) and -MATCH < rise < MATCH:
        Cc = "K33-HOLDS-PINNED"
    else:
        Cc = "K33-UNRESOLVED"
    st.append("K33-COMPLEMENT-%s" % arm_cls(k33c))
    st.append("K33-MAXR-POST100:%.4f" % max(k33m))
    st.append("K33-EXACT-CLAMP-EXCURSIONS" if sum(1 for x in k33o if x > 0) >= 2 else "K33-EXACT-CLAMP-CONTINUOUS")
    st.append("K01-%s" % arm_cls([GS[("k01", s)]["cls"] for s in SEEDS]))
    st.append("FLOOR-HOLDS" if abs(ME["k01"] - M1["k01"]) < rb else "FLOOR-DRIFTS")
    if any(ME[a] - ME["k01"] <= NULL for a in ["HEAD"] + LADDER):
        st.append("FLOOR-READINGS-ARE-BOUNDS")
    ln = ("  (c) %s   K33 classes %s, max r after epoch 100 %s, off-exact records after epoch 100 %s"
          % (Cc, k33c, ["%.4f" % x for x in k33m], k33o))
    print(ln)
    expect(ln)
    final = "FINAL: %s | %s | %s | %s" % (A, B, Cc, " | ".join(st))
    logfinal = [x for x in LOG if x.startswith("FINAL:")]
    print("\n  REBUILT %s" % final)
    same = bool(logfinal) and logfinal[-1] == final
    print("  == the scorer's FINAL line, byte for byte: %s (%d tokens)" % (same, len(final.split(" | "))))
    if not same:
        viol("FINAL differs: scorer %r" % (logfinal[-1] if logfinal else None))

    # ---------------------------------------------------------------- [7]
    print("\n[7] the scorer's between-batch line and descriptive readout, rebuilt")
    L = ["  cvt5 {93,94,95} @100: %s" % "  ".join("%s %.4f" % (a, M1[a]) for a in ARMS)]
    for a in ARMS:
        cells = []
        for e in (29, 49, 99, 149, 199, 249, 299):
            cells.append("%d:%.2f(%.2f)" % (e, mean([R[(a, s)]["ep"][e][1] for s in SEEDS]),
                                            mean([R[(a, s)]["ep"][e][0] for s in SEEDS])))
        L.append("  %-5s %s" % (a, "  ".join(cells)))

    def beta_at(a, s, k, e):
        v = [float(x["beta"][k]) for x in PR[(a, s)] if x["step"] == e * SPE]
        return v[0] if v else None

    def pinned_from(a, s, k):
        seq = [(x["step"] / float(SPE), float(x["beta"][k])) for x in PR[(a, s)]]
        pf = None
        for j in range(len(seq) - 1, -1, -1):
            if seq[j][1] > LO + EXACT:
                break
            pf = seq[j][0]
        return pf

    for a in ARMS:
        cells = ["%d:%.3f" % (e, mean([beta_at(a, s, 0, e) for s in SEEDS])) for e in (50, 100, 150, 200, 250, 299)]
        g1 = ""
        if a != "k01":
            g1 = "  g1 pinned-from %s" % "/".join("n/a" if pinned_from(a, s, 1) is None else "%.1f" % pinned_from(a, s, 1)
                                                 for s in SEEDS)
        L.append("  %-5s %s%s" % (a, "  ".join(cells), g1))
    for s in SEEDS:
        cells = []
        for lo in range(0, E, 50):
            sel = [x for x in PR[("K13", s)] if lo * SPE <= x["step"] < (lo + 50) * SPE]
            dn = sum(1 for x in sel if x["pt_b2"] * x["mom_pre"][0][0] + (1 - x["pt_b2"]) * x["z_agg"][0][0] > 0)
            cells.append("%d-%d:%.3f" % (lo, lo + 50, dn / float(len(sel))))
        L.append("  K13 s%d %s" % (s, "  ".join(cells)))
    for ln in L:
        print(ln)
        expect(ln)

    # ---------------------------------------------------------------- [8]
    print("\n[8] every bar margin (pp; SE at SE_USED %.6f)" % se)
    mg = [("k01@100 under K01_MAX 20", K01_MAX - M1["k01"]),
          ("D_HEAD@100 over GAP_MIN 20", d1 - GAP_MIN),
          ("PREMISE HEAD-K33@100 over SHAPE_GAP 15", M1["HEAD"] - M1["K33"] - SHAPE_GAP),
          ("PREMISE HEAD-K13@100 over MATCH 5", M1["HEAD"] - M1["K13"] - MATCH),
          ("PREMISE K13-K33@100 over MATCH 5", M1["K13"] - M1["K33"] - MATCH),
          ("G13@300 over MATCH 5 (GRADED-WAS-A-DELAY edge)", g13 - MATCH),
          ("K13@300 - HEAD@300 under MATCH 5 (K13-ABOVE-HEAD edge)", MATCH - (ME["K13"] - ME["HEAD"])),
          ("DROP13 under +5 (K13-RECOVERS edge)", MATCH - drop),
          ("DROP13 over -5 (GRADED-PARTLY-A-DELAY edge)", drop + MATCH),
          ("D_HEAD@300 over REFUTE 5", dE - REFUTE),
          ("D_HEAD@300 over SUPPORT 15", dE - SUPPORT),
          ("K33@300-K33@100 under +5 (K33-REOPENS by level)", MATCH - rise),
          ("K33@300-K33@100 over -5", rise + MATCH),
          ("worst seed range @100 under DIVERGED 5", DIVERGED - max(rng(V1[a]) for a in ARMS)),
          ("worst seed range @300 under DIVERGED 5", DIVERGED - max(rng(VE[a]) for a in ARMS)),
          ("DROP13 inside READ_BAR (K13-LEVEL-HOLDS)", rb - abs(drop)),
          ("HEAD@300-HEAD@100 inside READ_BAR", rb - abs(ME["HEAD"] - M1["HEAD"])),
          ("K33@300-K33@100 inside READ_BAR", rb - abs(rise)),
          ("k01@300-k01@100 inside READ_BAR (FLOOR-HOLDS)", rb - abs(ME["k01"] - M1["k01"]))]
    if k13i:
        mg.append(("K13 @300 - post-pin level inside READ_BAR (FROZEN-AT-PIN)", rb - abs(k13i["d"])))
    if hi_:
        mg.append(("HEAD @300 - post-pin level inside READ_BAR (FROZEN-AT-PIN)", rb - abs(hi_["d"])))
    for k, v in mg:
        print("  margin %-62s %+9.4f pp = %+7.2f SE" % (k, v, v / se))
    print("  RHO %.4f over SURVIVE 0.80 by %.4f; ATTEN_MILD 0.50 by %.4f" % (rho, rho - SURVIVE, rho - ATTEN_MILD))
    print("  K13 latest pin %.1f under PIN_BOUND 223.5 by %.1f; post-pin window %.1f over W_POST 64.2 by %.1f"
          % (k13i["latest"], PIN_BOUND - k13i["latest"], E - k13i["latest"], E - k13i["latest"] - W_POST)
          if k13i else "  K13 not pinned by the gate")
    print("  K33 max r after 100 = %.4f under PIN 2 by %.4f; median r per seed %s (PIN 2)"
          % (max(k33m), PIN_R - max(k33m), " / ".join("%.4f" % GS[("K33", s)]["med_r"] for s in SEEDS)))
    print("  K13 median r per seed %s under PIN 2 by %s"
          % (" / ".join("%.4f" % GS[("K13", s)]["med_r"] for s in SEEDS),
             " / ".join("%.4f" % (PIN_R - GS[("K13", s)]["med_r"]) for s in SEEDS)))

    # ---------------------------------------------------------------- [9]
    print("\n[9] DESCRIPTIVE ONLY -- cannot move any gate, branch or stamp")
    print("  [9a] arm-mean TEST / TRAIN at single epochs 99 / 149 / 199 / 249 / 299 and as 5-epoch windows ending there")
    for a in ARMS:
        c1, c2 = [], []
        for e in (100, 150, 200, 250, 300):
            c1.append("%d:%.2f/%.2f" % (e - 1, mean([R[(a, s)]["ep"][e - 1][1] for s in SEEDS]),
                                        mean([R[(a, s)]["ep"][e - 1][0] for s in SEEDS])))
            c2.append("%d-%d:%.4f/%.4f" % (e - 5, e - 1, mean([win(R[(a, s)]["ep"], e, 1) for s in SEEDS]),
                                           mean([win(R[(a, s)]["ep"], e, 0) for s in SEEDS])))
        print("   %-5s epoch  %s" % (a, "  ".join(c1)))
        print("   %-5s window %s" % (a, "  ".join(c2)))
    print("  [9b] per-seed TEST window (TRAIN) at 95-99 / 145-149 / 195-199 / 245-249 / 295-299")
    for a in ARMS:
        for s in SEEDS:
            ep = R[(a, s)]["ep"]
            print("   %-5s s%d %s" % (a, s, "  ".join("%.3f(%.3f)" % (win(ep, e, 1), win(ep, e, 0))
                                                     for e in (100, 150, 200, 250, 300))))
    print("  [9c] did the TEST level keep rising after epoch 100?  per-seed OLS slope pp/epoch over 100-149 / 150-199 /"
          " 200-249 / 250-299; arm-mean settle epoch = first epoch after which the arm-mean TEST stays within 0.5 pp of @300")
    for a in ARMS:
        sl = []
        for s in SEEDS:
            ep = R[(a, s)]["ep"]
            sl.append("/".join("%+.4f" % ols(list(range(lo, lo + 50)), [ep[e][1] for e in range(lo, lo + 50)])
                               for lo in (100, 150, 200, 250)))
        am = [mean([R[(a, s)]["ep"][e][1] for s in SEEDS]) for e in range(E)]
        settle = next(e for e in range(E) if all(abs(v - ME[a]) <= 0.5 for v in am[e:]))
        amt = [mean([R[(a, s)]["ep"][e][0] for s in SEEDS]) for e in range(E)]
        settle_t = next(e for e in range(E) if all(abs(v - MTE[a]) <= 0.5 for v in amt[e:]))
        print("   %-5s %s | TEST settle %d, TRAIN settle %d | TEST 100-window->300-window %+.4f, 150->300 %+.4f, 200->300 %+.4f"
              % (a, "  ".join(sl), settle, settle_t, ME[a] - M1[a],
                 ME[a] - mean([win(R[(a, s)]["ep"], 150, 1) for s in SEEDS]),
                 ME[a] - mean([win(R[(a, s)]["ep"], 200, 1) for s in SEEDS])))
    print("  [9d] complement (group 0) beta and r = exp(beta+15) at epochs 100 / 150 / 200 / 250 and the last record (299.8),"
          " per seed")
    for a in ARMS:
        for s in SEEDS:
            seq = GS[(a, s)]["seq"]
            cells = []
            for e in (100, 150, 200, 250):
                b = beta_at(a, s, 0, e)
                cells.append("%d:%.3f(r %.2f)" % (e, b, math.exp(b + 15.0)))
            cells.append("299.8:%.3f(r %.2f)" % (seq[-1][1], math.exp(seq[-1][1] + 15.0)))
            print("   %-5s s%d %s" % (a, s, "  ".join(cells)))
    print("  [9e] pin epochs by the magnitude gate (first r<=2), and the complement AFTER that epoch: records above r 2,"
          " max r, fraction of records at the exact clamp; r by 50-epoch window (mean r / fraction r > 2)")
    for a in ARMS:
        for s in SEEDS:
            g = GS[(a, s)]
            aft = [(e_, v) for e_, v in g["seq"] if g["f2"] is not None and e_ >= g["f2"]]
            above = sum(1 for _e, v in aft if v > lr2)
            mx = max((math.exp(v + 15.0), e_) for e_, v in aft) if aft else (None, None)
            atc = sum(1 for _e, v in aft if v <= LO + EXACT) / float(len(aft)) if aft else None
            wins = []
            for lo in range(0, E, 50):
                rs = [math.exp(v + 15.0) for e_, v in g["seq"] if lo <= e_ < lo + 50]
                wins.append("%d-%d:%.2f/%.2f" % (lo, lo + 50, mean(rs), sum(1 for x in rs if x > PIN_R) / float(len(rs))))
            print("   %-5s s%d first r<=2 %s | after: %d records, %d above r 2, max r %s at ep %s, at exact clamp %s | %s"
                  % (a, s, "never" if g["f2"] is None else "%.1f" % g["f2"], len(aft), above,
                     "n/a" if mx[0] is None else "%.3f" % mx[0], "n/a" if mx[1] is None else "%.1f" % mx[1],
                     "n/a" if atc is None else "%.3f" % atc, "  ".join(wins)))
    print("  [9f] isolated beta[1] (idx 50) on the blockwise arms: exact-clamp from, max r after epoch 100")
    for a in ARMS[1:]:
        cells = []
        for s in SEEDS:
            b1 = [float(x["beta"][1]) for x in PR[(a, s)] if x["step"] >= C * SPE]
            pf1 = pinned_from(a, s, 1)
            cells.append("s%d from %s max r %.4f" % (s, "never" if pf1 is None else "%.1f" % pf1,
                                                     max(math.exp(v + 15.0) for v in b1)))
        print("   %-5s %s" % (a, " | ".join(cells)))

    print("\nAGREEMENT with the scorer's printed lines: %d matched, %d missing" % (agree[0], agree[1]))
    print("VIOLATIONS %d" % len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
