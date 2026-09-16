#!/usr/bin/env python3
"""cgn2_cpl2_attack_indep.py <cgn2|cpl2> <runsdir> [<scorer.py>] -- independent re-derivation (CORRECTIONS 225/226).

Imports NOTHING from analysis/cGN2_gn_isolation_score.py, analysis/cPL2_plainnet_head_score.py, the
earlier attack parsers, or any other repo module, and uses no regex.  It reads, by string splitting:
  * every raw <prefix>-<arm>-s<seed>-<jobid>.out in <runsdir>: NODE / ARGS / ENV / PROBE_TENSOR
    header lines, every `Epoch` line, RUN_DONE, Traceback;
  * <runsdir>/<prefix>/PARTITION-MANIFEST.txt (TENSOR / NORM lines) and PROVENANCE.txt;
  * each run's probe dir: block_sizes.json, probe_tensor.json, probe.jsonl (arity, beta trajectory,
    per-tensor hypergradient terms; NO accuracy is read from them).
plateau5 = mean TEST over the run's own epochs 95..99; TRAIN5 the same on the Train column.
The CSV is not read.  Bars and branch order are RE-TYPED from the registration text (221.5 for cgn2,
222.5 for cpl2).  Section [7] is new (not in either scorer): the identity-vs-magnitude table -- every
tensor's mean |L_i| = |0.9 m_i + 0.1 z_i| on the k01 arm's collapse-phase records (cgn2: post-peak, the
217.6 deviation, because the GN scalar never pins; cpl2: pinned records, 194.5's set), its rank, numel
and the fraction of records on which L_i > 0 (votes beta DOWN, 217.6's convention), plus the fraction of
records on which each m=2 arm's COMPLEMENT group still votes DOWN after the isolation.
Optional third argument: the registered scorer file, whose sha256 is compared with PROVENANCE.txt.
"""
import hashlib
import json
import math
import os
import sys

FIXED_HF = {"optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99",
            "weight-decay-base": "0.1", "alg-meta": "Lion", "momentum-param-meta": "0.99",
            "Lion-beta2-meta": "0.9", "weight-decay-meta": "0", "dataset": "CIFAR100",
            "batch-size": "100", "max-time": "999:00:00", "gamma": "1",
            "meta-stepsize": "1e-3", "alpha0": "1e-6", "num-epochs": "100"}
ENV_WANT = {"AUGMENT": "1", "BETA_CLIP": "-15:-2.3026", "HIER": "none", "LAM": "na",
            "ETA_RATIO": "na", "COS_TOTAL": "default", "COS_WARMUP": "default", "SCHED": "none",
            "SCHED_TOTAL": "none", "SCHED_WARMUP": "none", "SCHED_MIN": "none", "PROBE": "100",
            "EB_RHO": "na", "EB_LOG": "0"}
CLAMP = -14.999

BATCHES = {
    "cgn2": {
        "net": "ResNet18_gn_c100", "seeds": [72, 73, 74], "arms": ["k01", "kL", "ISO", "CTL", "ONE"],
        "spec": {"k01": "scalar", "kL": "layerwise",
                 "ISO": "sets:1-49,51-52,54-58,60-62/layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight",
                 "CTL": "sets:1-46,49-55,57-62/layer4.0.bn1.weight,layer4.0.bn1.bias,layer4.1.bn1.weight",
                 "ONE": "sets:1-49,51-62/layer4.0.bn2.weight"},
        "iso_idx": {"ISO": [50, 53, 59], "CTL": [47, 48, 56], "ONE": [50]},
        "nt": 62, "tot": 11220132, "norm": "GroupNorm 20 BatchNorm2d 0",
        "nb": {"k01": [11220132], "kL": None, "ISO": [11218596, 1536], "CTL": [11218596, 1536],
               "ONE": [11219620, 512]},
        "sigma_prior": 0.725869593430298,
        "carriers": [50, 53, 59], "record_set": "postpeak",
        "build_sha_prefix": "9f6e4ec9", "scorer_sha_prefix": "cdf6643e",
    },
    "cpl2": {
        "net": "PlainNet18_c100", "seeds": [75, 76, 77], "arms": ["k01", "kL", "ISO", "HEAD", "TWIN"],
        "spec": {"k01": "scalar", "kL": "layerwise",
                 "ISO": "sets:1-43,45-49,51-53/layer4.0.bn2.weight,layer4.1.bn2.weight",
                 "HEAD": "sets:1-49,51-53/layer4.1.bn2.weight",
                 "TWIN": "sets:1-46,48-53/layer4.1.bn1.weight"},
        "iso_idx": {"ISO": [44, 50], "HEAD": [50], "TWIN": [47]},
        "nt": 53, "tot": 11046308, "norm": None,
        "nb": {"k01": [11046308], "kL": None, "ISO": [11045284, 1024], "HEAD": [11045796, 512],
               "TWIN": [11045796, 512]},
        "sigma_prior": 0.694442846939599,
        # [50] only: 44 holds 0.0000 of k01's pinned mass (rank 29 of 53) -- the first pass used [44, 50]
        # (cpl1's pair) and its "within x2 of the smallest carrier" list was keyed to 44's near-zero mass.
        "carriers": [50], "record_set": "pinned",
        "build_sha_prefix": "e65e6773", "scorer_sha_prefix": "66c3adab",
    },
}


def mean(v):
    return sum(v) / len(v)


def sdev(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def ols(xs, ys):
    xm, ym = mean(xs), mean(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)


def parse_out(path):
    rec = {"node": [], "args": [], "env": [], "pt": [], "ep": {}, "done": False, "tb": 0, "dup_ep": 0}
    for ln in open(path, errors="replace"):
        s = ln.rstrip("\n")
        if s.startswith("NODE="):
            rec["node"].append(s)
        elif s.startswith("ARGS:"):
            rec["args"].append(s)
        elif s.startswith("ENV:"):
            rec["env"].append(s)
        elif s.startswith("PROBE_TENSOR:"):
            rec["pt"].append(s)
        elif s.startswith("Epoch "):
            p = s.split(",")
            e = int(p[0].split()[1])
            tr = float(p[1].split(":")[1].replace("%", "").strip())
            te = float(p[2].split(":")[1].replace("%", "").strip())
            if e in rec["ep"]:
                rec["dup_ep"] += 1
            rec["ep"][e] = (tr, te)
        elif s.strip() == "RUN_DONE":
            rec["done"] = True
        elif s.startswith("Traceback"):
            rec["tb"] += 1
    return rec


def args_flags(line):
    flags = []
    for t in line[len("ARGS:"):].split():
        if t.startswith("--"):
            flags.append([t[2:], None])
        elif flags:
            flags[-1][1] = t if flags[-1][1] is None else flags[-1][1] + " " + t
    return flags


def load_recs(pd):
    return [json.loads(x) for x in open(os.path.join(pd, "probe.jsonl")) if x.strip()]


def epoch_of(recs):
    stride = recs[1]["step"] - recs[0]["step"] if len(recs) > 1 else 1
    mul = 1 if stride >= 100 else 100
    return [x["step"] * mul / 500.0 for x in recs]


def terms(x):
    b2 = x.get("pt_b2", 0.9)
    return [b2 * m + (1 - b2) * z for m, z in zip(x["m_tensor"], x["z_tensor"])]


def main():
    prefix, runsdir = sys.argv[1], sys.argv[2]
    scorer = sys.argv[3] if len(sys.argv) > 3 else None
    B = BATCHES[prefix]
    NT, TOT, S3 = B["nt"], B["tot"], B["seeds"]
    files = {}
    for fn in sorted(os.listdir(runsdir)):
        if fn.startswith(prefix + "-") and fn.endswith(".out"):
            stem, _, jid = fn[:-4].rpartition("-")
            files.setdefault(stem, []).append((int(jid), fn))
    viol, runs = [], {}
    print("=" * 78)
    print("%s INDEPENDENT ATTACK -- %s, %d arms x seeds %s" % (prefix, B["net"], len(B["arms"]), S3))
    print("=" * 78)

    print("\n[1] files and completeness")
    for a in B["arms"]:
        for s in S3:
            rn = "%s-%s-s%d" % (prefix, a, s)
            lst = files.get(rn, [])
            if len(lst) != 1:
                viol.append("%s has %d .out files" % (rn, len(lst)))
                print("  VIOLATION %s: %d files" % (rn, len(lst)))
                continue
            jid, fn = lst[0]
            r = parse_out(os.path.join(runsdir, fn))
            r["jid"], r["rn"] = jid, rn
            ok = sorted(r["ep"]) == list(range(100)) and r["done"] and r["tb"] == 0 and r["dup_ep"] == 0
            if not ok:
                viol.append("%s incomplete" % rn)
            r["p5"] = mean([r["ep"][e][1] for e in range(95, 100)]) if ok else None
            r["t5"] = mean([r["ep"][e][0] for e in range(95, 100)]) if ok else None
            runs[(a, s)] = r
            print("  %-4s %-17s job %d  epochs %3d  RUN_DONE %s  traceback %d  dup-epoch %d"
                  % ("OK" if ok else "BAD", rn, jid, len(r["ep"]), r["done"], r["tb"], r["dup_ep"]))
    extra = sorted(set(files) - set("%s-%s-s%d" % (prefix, a, s) for a in B["arms"] for s in S3))
    if extra:
        viol.append("unexpected run names %s" % extra)

    print("\n[2] header lines against the registered values re-typed here")
    env_distinct = {}
    for (a, s), r in sorted(runs.items()):
        errs = []
        if len(r["args"]) != 1:
            errs.append("%d ARGS lines" % len(r["args"]))
        else:
            fl = args_flags(r["args"][0])
            names = [k for k, _ in fl]
            rep = sorted(set(k for k in names if names.count(k) > 1))
            if rep:
                errs.append("repeated flags %s" % rep)
            d = dict(fl)
            want = dict(FIXED_HF)
            want.update({"NN-name": B["net"], "stepsize-groups": B["spec"][a], "seed": str(s), "run-name": r["rn"]})
            for k, v in want.items():
                if d.get(k) != v:
                    errs.append("--%s=%s want %s" % (k, d.get(k), v))
            if not d.get("save-directory", "").endswith("/runs/" + prefix):
                errs.append("--save-directory %s" % d.get("save-directory"))
            if len(fl) != 20:
                errs.append("%d flags" % len(fl))
        if len(r["env"]) != 1:
            errs.append("%d ENV lines" % len(r["env"]))
        else:
            kv = dict(t.split("=", 1) for t in r["env"][0][4:].split())
            pdir = kv.pop("PROBE_DIR", "")
            if not pdir.endswith("/runs/%s/probe_%s" % (prefix, r["rn"])):
                errs.append("PROBE_DIR %s" % pdir)
            if kv != ENV_WANT:
                errs.append("ENV differs %s" % sorted(set(kv.items()) ^ set(ENV_WANT.items())))
            key = " ".join("%s=%s" % x for x in sorted(kv.items()))
            env_distinct[key] = env_distinct.get(key, 0) + 1
        if len(r["node"]) != 1:
            errs.append("%d NODE lines" % len(r["node"]))
        else:
            parts = [p.strip() for p in r["node"][0].split("|")]
            job = parts[1].split() if len(parts) == 3 else []
            if len(parts) != 3 or job != ["JOB=" + r["rn"], str(r["jid"])] or parts[2] != "AUGMENT=1":
                errs.append("NODE header %s" % r["node"][0])
        want_pt = "scalar" if a == "k01" else ("layerwise" if a == "kL" else "blockwise")
        if len(r["pt"]) != 1:
            errs.append("%d PROBE_TENSOR lines" % len(r["pt"]))
        else:
            pk = dict(t.split("=", 1) for t in r["pt"][0].split() if "=" in t)
            if pk.get("type") != want_pt or pk.get("tensors") != str(NT) or pk.get("every") != "100":
                errs.append("PROBE_TENSOR %s" % r["pt"][0][:90])
        if errs:
            viol.append("%s: %s" % (r["rn"], "; ".join(errs)))
            print("  VIOLATION %s: %s" % (r["rn"], "; ".join(errs)))
    print("  %d runs: ARGS (20 flags, none repeated, every design value), ENV (15 keys + own PROBE_DIR),"
          " NODE (job name, id, AUGMENT=1), PROBE_TENSOR (type, tensors=%d, every=100)" % (len(runs), NT))
    print("  distinct ENV lines (PROBE_DIR stripped): %d x%s" % (len(env_distinct), list(env_distinct.values())))

    print("\n[3] manifest, provenance, probe arity and per-group beta timelines")
    tens, norm = [], None
    for ln in open(os.path.join(runsdir, prefix, "PARTITION-MANIFEST.txt")):
        p = ln.split()
        if p and p[0] == "TENSOR":
            tens.append((int(p[1]), p[2], int(p[3])))
        elif p and p[0] == "NORM":
            norm = " ".join(p[1:])
    names = {i: n for i, n, _ in tens}
    numel = {i: q for i, _, q in tens}
    print("  manifest: %d tensors, %d params, NORM %s, 'shortcut' names %d"
          % (len(tens), sum(numel.values()), norm, sum("shortcut" in n for n in names.values())))
    if len(tens) != NT or sum(numel.values()) != TOT:
        viol.append("manifest arity")
    if B["norm"] and norm != B["norm"]:
        viol.append("manifest NORM %s" % norm)
    if prefix == "cpl2" and any("shortcut" in n for n in names.values()):
        viol.append("PlainNet manifest carries a shortcut tensor")
    for a, idx in B["iso_idx"].items():
        print("  %-4s isolates %s  numel %s" % (a, ["%d %s" % (i, names[i]) for i in idx], [numel[i] for i in idx]))
    prov = {}
    for ln in open(os.path.join(runsdir, prefix, "PROVENANCE.txt")):
        p = ln.split(None, 1)
        if len(p) == 2:
            prov[p[0]] = p[1].strip()
    print("  PROVENANCE: MODE %s  BUILD_NETWORK_SHA256 %s  SCORER_SHA256 %s  REGISTERED_COMMIT %s"
          % (prov.get("MODE"), prov.get("BUILD_NETWORK_SHA256", "")[:16], prov.get("SCORER_SHA256", "")[:16],
             prov.get("REGISTERED_COMMIT", "")[:12]))
    if prov.get("MODE", "").lower() != "submit" or not prov.get("BUILD_NETWORK_SHA256", "").startswith(B["build_sha_prefix"]) \
            or not prov.get("SCORER_SHA256", "").startswith(B["scorer_sha_prefix"]):
        viol.append("PROVENANCE")
    if scorer:
        h = hashlib.sha256(open(scorer, "rb").read()).hexdigest()
        print("  scorer file sha256 %s == PROVENANCE: %s" % (h[:16], h == prov.get("SCORER_SHA256")))
        if h != prov.get("SCORER_SHA256"):
            viol.append("scorer sha != PROVENANCE")
    recs_of = {}
    for (a, s), r in sorted(runs.items()):
        pd = os.path.join(runsdir, prefix, "probe_" + r["rn"])
        bs = json.load(open(os.path.join(pd, "block_sizes.json")))
        pt = json.load(open(os.path.join(pd, "probe_tensor.json")))
        recs = load_recs(pd)
        recs_of[(a, s)] = recs
        nb = bs["n_b"]
        wnb = B["nb"][a]
        wtype = "scalar" if a == "k01" else ("layerwise" if a == "kL" else "blockwise")
        ok = (sum(nb) == TOT and (len(nb) == NT if wnb is None else nb == wnb) and bs["stepsize_type"] == wtype
              and pt["n_tensors"] == NT and all(len(x["z_tensor"]) == NT and len(x["beta"]) == len(nb) for x in recs))
        if not ok:
            viol.append("%s probe arity" % r["rn"])
        ep = epoch_of(recs)
        line = "  %-17s %-9s n_b %-24s records %d" % (r["rn"], bs["stepsize_type"],
                                                    nb if len(nb) <= 2 else "(%d groups)" % len(nb), len(recs))
        if len(nb) <= 2:
            for g in range(len(nb)):
                col = [x["beta"][g] <= CLAMP for x in recs]
                k = len(col)
                while k > 0 and col[k - 1]:
                    k -= 1
                stay = ep[k] if k < len(col) else None
                pk = max(range(len(recs)), key=lambda i: recs[i]["beta"][g])
                tx = [e for e in ep if e >= 80]
                ty = [x["beta"][g] for x, e in zip(recs, ep) if e >= 80]
                line += " | g%d last %.3f peak %.3f@ep%.1f clamp %d/%d from %s beta-slope80-99 %+.4f/ep" % (
                    g, recs[-1]["beta"][g], recs[pk]["beta"][g], ep[pk], sum(col), len(col),
                    "ep%.1f" % stay if stay is not None else "never", ols(tx, ty) if len(set(ty)) > 1 else 0.0)
        else:
            last = recs[-1]["beta"]
            line += " | last: %d/%d groups at clamp %s, max beta %.3f" % (
                sum(b <= CLAMP for b in last), len(last), [i + 1 for i, b in enumerate(last) if b <= CLAMP], max(last))
        print(line)

    if any(r["p5"] is None for r in runs.values()) or len(runs) != len(B["arms"]) * 3:
        print("\nINCOMPLETE -- no number derived.  VIOLATIONS %d: %s" % (len(viol), viol))
        sys.exit(2)

    print("\n[4] levels: plateau5 TEST beside TRAIN5 (in batch); tail OLS slope 80..99; trajectory")
    M, T, R = {}, {}, {}
    ss, df = 0.0, 0
    xs = list(range(80, 100))
    for a in B["arms"]:
        v = [runs[(a, s)]["p5"] for s in S3]
        w = [runs[(a, s)]["t5"] for s in S3]
        M[a], T[a], R[a] = mean(v), mean(w), max(v) - min(v)
        ss += sum((x - M[a]) ** 2 for x in v)
        df += 2
        sl = [ols(xs, [runs[(a, s)]["ep"][e][1] for e in xs]) for s in S3]
        slt = [ols(xs, [runs[(a, s)]["ep"][e][0] for e in xs]) for s in S3]
        print("  %-4s TEST %.4f sd %.4f range %.4f | TRAIN %.4f | seeds %s | slope TEST %s TRAIN %s"
              % (a, M[a], sdev(v), R[a], T[a], " ".join("s%d %.4f/%.4f" % (s, x, y) for s, x, y in zip(S3, v, w)),
                 "/".join("%+.3f" % q for q in sl), "/".join("%+.3f" % q for q in slt)))
    for a in B["arms"]:
        print("    %-4s TEST@0,10..90,99: %s" % (a, " ".join("%.1f" % mean([runs[(a, s)]["ep"][e][1] for s in S3])
                                                        for e in list(range(0, 100, 10)) + [99])))
    sin = math.sqrt(ss / df)
    sig = max(B["sigma_prior"], sin)
    se = sig * math.sqrt(2.0 / 3.0)
    print("\n[5] sigma: in-batch %.6f (df %d); prior %.15f; USED %.6f (%s); SE_ARM_DIFF %.6f"
          % (sin, df, B["sigma_prior"], sig, "in-batch" if sin > B["sigma_prior"] else "prior", se))

    print("\n[6] contrasts and branch (registration text re-typed)")
    D = {a: M[a] - M["k01"] for a in B["arms"]}
    Dt = {a: T[a] - T["k01"] for a in B["arms"]}
    harness = bool(viol) or max(M.values()) < 15.0 or max(M.values()) > 90.0
    diverged = any(R[a] > 5.0 for a in B["arms"])
    if prefix == "cgn2":
        DID, DIDt = M["ISO"] - M["CTL"], T["ISO"] - T["CTL"]
        for lab, x, xt in (("PRIMARY DELTA_ID = ISO-CTL", DID, DIDt), ("KEY D_ONE = ONE-k01", D["ONE"], Dt["ONE"]),
                           ("D_ISO", D["ISO"], Dt["ISO"]), ("D_CTL", D["CTL"], Dt["CTL"]),
                           ("D_HEAD3 = ISO-ONE", M["ISO"] - M["ONE"], T["ISO"] - T["ONE"]),
                           ("D_GAP = kL-k01", D["kL"], Dt["kL"])):
            print("  %-28s %+.4f pp = %+.2f SE   (TRAIN %+.4f)" % (lab, x, x / se, xt))
        print("  RECOVERY %.4f  RECOVERY_ONE %.4f (descriptive)" % (D["ISO"] / D["kL"], D["ONE"] / D["kL"]))
        resc = {a: D[a] >= 10.0 for a in ("ISO", "CTL", "ONE")}
        if harness:
            br = "HARNESS-UNSOUND"
        elif M["kL"] < 30.0:
            br = "GN-UNTRAINABLE"
        elif diverged:
            br = "UNRESOLVED-DIVERGED"
        elif D["kL"] < 20.0:
            br = "GAP-NOT-REPRODUCED-GN"
        elif not any(resc.values()):
            br = "NO-RESCUE-GN"
        elif not resc["ISO"]:
            br = "UNEXPECTED-PATTERN"
        elif DID <= -15.0:
            br = "CONTROL-DOMINATES-GN"
        elif DID >= 15.0:
            br = ("IDENTITY-TRANSFERS-GN" if D["ONE"] >= 10.0 else
                  "IDENTITY-TRANSFERS-ONE-NULL-GN" if D["ONE"] <= 2.0 else "IDENTITY-TRANSFERS-ONE-PARTIAL-GN")
        elif resc["CTL"] and abs(DID) <= 2.0:
            br = "CLASS-OPERATIVE-GN"
        else:
            br = "IDENTITY-ATTENUATED-GN"
        near = min(abs(DID - 15.0), abs(D["ONE"] - 10.0), abs(D["ISO"] - 10.0), abs(D["CTL"] - 10.0), abs(D["kL"] - 20.0))
    else:
        DH, DHt = M["HEAD"] - M["TWIN"], T["HEAD"] - T["TWIN"]
        DP = M["ISO"] - M["HEAD"]
        for lab, x, xt in (("PRIMARY DELTA_HEAD = HEAD-TWIN", DH, DHt), ("KEY D_HEAD = HEAD-k01", D["HEAD"], Dt["HEAD"]),
                           ("D_PAIR = ISO-HEAD", DP, T["ISO"] - T["HEAD"]), ("D_ISO (pos. control)", D["ISO"], Dt["ISO"]),
                           ("D_TWIN", D["TWIN"], Dt["TWIN"]), ("D_GAP = kL-k01", D["kL"], Dt["kL"])):
            print("  %-30s %+.4f pp = %+.2f SE   (TRAIN %+.4f)" % (lab, x, x / se, xt))
        print("  RECOVERY_HEAD = D_HEAD/D_ISO %.4f (descriptive)" % (D["HEAD"] / D["ISO"]))
        if harness:
            br = "HARNESS-UNSOUND"
        elif M["kL"] < 30.0:
            br = "PLAIN-UNTRAINABLE"
        elif diverged:
            br = "UNRESOLVED-DIVERGED"
        elif D["kL"] < 20.0:
            br = "GAP-NOT-REPRODUCED-PLAIN"
        elif D["ISO"] < 10.0:
            br = "POSITIVE-CONTROL-FAILED-PLAIN"
        elif D["HEAD"] < 10.0 and D["TWIN"] < 10.0:
            br = "PAIR-REQUIRED-PLAIN" if D["HEAD"] <= 2.0 else "HEAD-PARTIAL-PLAIN"
        elif D["TWIN"] >= 10.0 and D["HEAD"] < 10.0:
            br = "UNEXPECTED-PATTERN"
        elif DH <= -15.0:
            br = "CONTROL-DOMINATES-PLAIN"
        elif DH >= 15.0:
            br = "HEAD-CARRIES-ALONE-PLAIN" if DP <= 5.0 else "HEAD-CARRIES-MOST-PLAIN"
        elif D["TWIN"] >= 10.0 and abs(DH) <= 2.0:
            br = "CLASS-OPERATIVE-PLAIN"
        else:
            br = "IDENTITY-ATTENUATED-PLAIN"
        near = min(abs(DH - 15.0), abs(DP - 5.0), abs(D["HEAD"] - 10.0), abs(D["HEAD"] - 2.0),
                   abs(D["ISO"] - 10.0), abs(D["TWIN"] - 10.0), abs(D["kL"] - 20.0))
    print("  gates: max arm %.4f in [15, 90]; kL %.4f >= 30; seed ranges %s" %
          (max(M.values()), M["kL"], {a: round(R[a], 4) for a in B["arms"]}))
    print("  arms within 2.0 pp of k01 (floor readings = BOUNDS): %s"
          % [a for a in B["arms"] if a != "k01" and abs(D[a]) <= 2.0])
    print("  nearest bar margin over the branch-deciding quantities: %.4f pp" % near)
    print("  INDEPENDENT BRANCH: %s" % br)

    print("\n[7] identity vs MAGNITUDE (new; descriptive, non-gating)")
    C = B["carriers"]
    mass = [0.0] * NT
    pos = [0.0] * NT
    down_share, nrec_tot = [], 0
    for s in S3:
        recs = recs_of[("k01", s)]
        if B["record_set"] == "pinned":
            sel = [x for x in recs if x["beta"][0] <= CLAMP]
        else:
            pk = max(range(len(recs)), key=lambda i: recs[i]["beta"][0])
            sel = recs[pk + 1:]
        nrec_tot += len(sel)
        mm, pp, dn = [0.0] * NT, [0.0] * NT, 0
        for x in sel:
            L = terms(x)
            dn += sum(L) > 0
            for i in range(NT):
                mm[i] += abs(L[i])
                pp[i] += L[i] > 0
        for i in range(NT):
            mass[i] += mm[i] / len(sel) / 3.0
            pos[i] += pp[i] / len(sel) / 3.0
        down_share.append(dn / len(sel))
    tot = sum(mass)
    order = sorted(range(NT), key=lambda i: -mass[i])
    rank = {i + 1: order.index(i) + 1 for i in range(NT)}
    print("  k01 %s records (%d over 3 seeds); total votes DOWN on %s; median tensor mean|L| %.4e"
          % (B["record_set"], nrec_tot, "/".join("%.4f" % q for q in down_share), sorted(mass)[NT // 2]))
    print("  rank idx name                              numel      mean|L|     share   frac L>0")
    for i in order[:10]:
        print("  %4d %3d %-32s %9d  %.4e  %.4f  %.3f%s" % (rank[i + 1], i + 1, names[i + 1], numel[i + 1], mass[i],
                                                        mass[i] / tot, pos[i], "  CARRIER" if i + 1 in C else ""))
    cmin = min(mass[i - 1] for i in C)
    for a, idx in B["iso_idx"].items():
        print("  set %-4s %s: summed mean|L| %.4e  share %.4f  ranks %s" % (
            a, idx, sum(mass[i - 1] for i in idx), sum(mass[i - 1] for i in idx) / tot, [rank[i] for i in idx]))
    cand = [i + 1 for i in order if i + 1 not in C and mass[i] >= 0.5 * cmin]
    print("  NON-CARRIER tensors within x2 of the smallest carrier's mean|L| (%.4e): %s" % (
        cmin, ["%d %s numel %d share %.4f fracL>0 %.3f" % (i, names[i], numel[i], mass[i - 1] / tot, pos[i - 1])
               for i in cand] or "NONE"))
    same_sign = [i for i in cand if pos[i - 1] >= 0.5]
    print("  ...of which vote DOWN on a majority of records (sign-matched to the carriers): %s" % (same_sign or "NONE"))
    dn_nc = [i + 1 for i in order if i + 1 not in C and pos[i] >= 0.5]
    if dn_nc:
        j = dn_nc[0]
        print("  largest DOWN-voting (frac L>0 >= 0.5) non-carrier: %d %s numel %d rank %d share %.4f frac %.3f;"
              " smallest carrier / it = x%.4g" % (j, names[j], numel[j], rank[j], mass[j - 1] / tot, pos[j - 1],
                                                   cmin / mass[j - 1]))
        k3 = dn_nc[:len(C)]
        print("  best DOWN-voting non-carrier set of the carriers' size %s: summed mean|L| %.4e; carriers' sum / it = x%.1f"
              % (k3, sum(mass[i - 1] for i in k3), sum(mass[i - 1] for i in C) / sum(mass[i - 1] for i in k3)))
    w512 = [i for i in names if numel[i] == 512 and i not in C and names[i].endswith(".weight")]
    best = sorted(w512, key=lambda i: -mass[i - 1])[:3]
    print("  best numel-matched (512-wide scale) non-carriers: %s; their summed mean|L| / carriers' summed = x%.1f short"
          % (["%d r%d" % (i, rank[i]) for i in best],
             sum(mass[i - 1] for i in C) / max(1e-30, sum(mass[i - 1] for i in best[:len(C)]))))
    print("  after isolation: fraction of records on which the COMPLEMENT group's summed L votes DOWN")
    for a, idx in B["iso_idx"].items():
        cells = []
        for s in S3:
            recs = recs_of[(a, s)]
            fr = mean([1.0 if sum(L for j, L in enumerate(terms(x)) if j + 1 not in idx) > 0 else 0.0 for x in recs])
            late = [x for x, e in zip(recs, epoch_of(recs)) if e >= 50]
            frl = mean([1.0 if sum(L for j, L in enumerate(terms(x)) if j + 1 not in idx) > 0 else 0.0 for x in late])
            cells.append("s%d all %.3f / ep>=50 %.3f" % (s, fr, frl))
        print("    %-4s %s" % (a, " | ".join(cells)))

    print("\nVIOLATIONS %d%s" % (len(viol), (": " + "; ".join(viol)) if viol else ""))
    sys.exit(1 if viol else 0)


if __name__ == "__main__":
    main()
