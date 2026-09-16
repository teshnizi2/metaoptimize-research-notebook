#!/usr/bin/env python3
"""cgn1_cpl1_attack_indep.py <cgn1|cpl1> <runsdir> -- independent re-derivation (CORRECTIONS 217/218).

Imports NOTHING from analysis/cGN1_gn_gap_score.py, analysis/cPL1_plainnet_residual_score.py or
any other repo module, and uses no regex.  It reads, by plain string splitting:
  * every raw <prefix>-<arm>-s<seed>-<jobid>.out in <runsdir>: NODE / ARGS / ENV / PROBE_TENSOR
    header lines, every `Epoch` line, RUN_DONE, Traceback, the `minutes` line;
  * the batch's own <runsdir>/<prefix>/PARTITION-MANIFEST.txt (TENSOR lines only);
  * each run's probe dir <runsdir>/<prefix>/probe_<run>/: block_sizes.json, probe_tensor.json,
    probe.jsonl (arity and beta trajectory; no accuracy is read from them).
plateau5 = mean TEST over the run's own epochs 95..99; TRAIN5 the same on the Train column.
TRAIN is printed beside TEST everywhere.  The CSV is not read at all.

The bars and the branch order are RE-TYPED here from the registration text (212.5/212.6 for
cgn1; 215.5/215.6 for cpl1), not imported, so a scorer bug cannot propagate into this check.
"""
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

BATCHES = {
    "cgn1": {
        "net": "ResNet18_gn_c100", "seeds": [60, 61, 62], "arms": ["k01", "kL"],
        "spec": {"k01": "scalar", "kL": "layerwise"},
        "ptype": {"k01": "scalar", "kL": "layerwise"},
        "nt": 62, "tot": 11220132,
        "nb": {"k01": [11220132], "kL": None},        # None: 62 groups summing to tot
        "sigma_prior": 0.5862321062215915,
    },
    "cpl1": {
        "net": "PlainNet18_c100", "seeds": [69, 70, 71], "arms": ["k01", "kL", "ISO", "CTL", "ONE"],
        "spec": {"k01": "scalar", "kL": "layerwise",
                 "ISO": "sets:1-43,45-49,51-53/layer4.0.bn2.weight,layer4.1.bn2.weight",
                 "CTL": "sets:1-40,42-46,48-53/layer4.0.bn1.weight,layer4.1.bn1.weight",
                 "ONE": "sets:1-43,45-53/layer4.0.bn2.weight"},
        "ptype": {"k01": "scalar", "kL": "layerwise", "ISO": "blockwise", "CTL": "blockwise",
                  "ONE": "blockwise"},
        "nt": 53, "tot": 11046308,
        "nb": {"k01": [11046308], "kL": None, "ISO": [11045284, 1024], "CTL": [11045284, 1024],
               "ONE": [11045796, 512]},
        "sigma_prior": 0.694442846939599,
    },
}


def mean(v):
    return sum(v) / len(v)


def sdev(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def parse_out(path):
    rec = {"node": [], "args": [], "env": [], "pt": [], "ep": {}, "done": False, "tb": 0,
           "minutes": None, "dup_ep": 0}
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
        elif s.strip().endswith("minutes") and s.split()[0].replace(".", "").isdigit():
            rec["minutes"] = float(s.split()[0])
    return rec


def args_flags(line):
    flags = []
    for t in line[len("ARGS:"):].split():
        if t.startswith("--"):
            flags.append([t[2:], None])
        elif flags:
            flags[-1][1] = t if flags[-1][1] is None else flags[-1][1] + " " + t
    return flags


def main():
    prefix, runsdir = sys.argv[1], sys.argv[2]
    B = BATCHES[prefix]
    NT, TOT = B["nt"], B["tot"]
    files = {}
    for fn in sorted(os.listdir(runsdir)):
        if fn.startswith(prefix + "-") and fn.endswith(".out"):
            stem, _, jid = fn[:-4].rpartition("-")
            files.setdefault(stem, []).append((int(jid), fn))
    viol = []
    runs = {}
    print("=" * 78)
    print("%s INDEPENDENT ATTACK -- %s, %d arms x seeds %s" % (prefix, B["net"], len(B["arms"]), B["seeds"]))
    print("=" * 78)
    print("\n[1] files and completeness (one .out per (arm, seed) expected)")
    for a in B["arms"]:
        for s in B["seeds"]:
            rn = "%s-%s-s%d" % (prefix, a, s)
            lst = files.get(rn, [])
            if len(lst) != 1:
                viol.append("%s has %d .out files" % (rn, len(lst)))
                print("  VIOLATION %s: %d files" % (rn, len(lst)))
                continue
            jid, fn = lst[0]
            r = parse_out(os.path.join(runsdir, fn))
            r["jid"], r["fn"], r["rn"] = jid, fn, rn
            ok = sorted(r["ep"]) == list(range(100)) and r["done"] and r["tb"] == 0 and r["dup_ep"] == 0
            if not ok:
                viol.append("%s incomplete" % rn)
            r["p5"] = mean([r["ep"][e][1] for e in range(95, 100)]) if ok else None
            r["t5"] = mean([r["ep"][e][0] for e in range(95, 100)]) if ok else None
            runs[(a, s)] = r
            print("  %-4s %-16s job %d  epochs %3d  RUN_DONE %s  traceback %d  dup-epoch %d  minutes %s"
                  % ("OK" if ok else "BAD", rn, jid, len(r["ep"]), r["done"], r["tb"], r["dup_ep"], r["minutes"]))
    extra = sorted(set(files) - set("%s-%s-s%d" % (prefix, a, s) for a in B["arms"] for s in B["seeds"]))
    if extra:
        viol.append("unexpected files %s" % extra)
        print("  VIOLATION unexpected run names: %s" % extra)

    print("\n[2] header lines, checked against the registered values re-typed here")
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
            want.update({"NN-name": B["net"], "stepsize-groups": B["spec"][a], "seed": str(s),
                         "run-name": r["rn"]})
            for k, v in want.items():
                if d.get(k) != v:
                    errs.append("--%s=%s want %s" % (k, d.get(k), v))
            sd_ = d.get("save-directory", "")
            if not sd_.endswith("/runs/" + prefix):
                errs.append("--save-directory %s" % sd_)
            if len(fl) != 20:
                errs.append("%d flags, want 20" % len(fl))
        if len(r["env"]) != 1:
            errs.append("%d ENV lines" % len(r["env"]))
        else:
            kv = dict(t.split("=", 1) for t in r["env"][0][4:].split())
            pdir = kv.pop("PROBE_DIR", "")
            if not pdir.endswith("/runs/%s/probe_%s" % (prefix, r["rn"])):
                errs.append("PROBE_DIR %s" % pdir)
            if kv != ENV_WANT:
                errs.append("ENV differs: %s" % sorted(set(kv.items()) ^ set(ENV_WANT.items())))
            key = " ".join("%s=%s" % x for x in sorted(kv.items()))
            env_distinct[key] = env_distinct.get(key, 0) + 1
        if len(r["node"]) != 1:
            errs.append("%d NODE lines" % len(r["node"]))
        else:
            parts = [p.strip() for p in r["node"][0].split("|")]
            job = parts[1].split() if len(parts) == 3 else []
            if len(parts) != 3 or job != ["JOB=" + r["rn"], str(r["jid"])] or parts[2] != "AUGMENT=1":
                errs.append("NODE header %s" % r["node"][0])
        if len(r["pt"]) != 1:
            errs.append("%d PROBE_TENSOR lines" % len(r["pt"]))
        else:
            pk = dict(t.split("=", 1) for t in r["pt"][0].split() if "=" in t)
            if pk.get("type") != B["ptype"][a] or pk.get("tensors") != str(NT) or pk.get("every") != "100":
                errs.append("PROBE_TENSOR %s" % r["pt"][0][:90])
        if errs:
            viol.append("%s: %s" % (r["rn"], "; ".join(errs)))
            print("  VIOLATION %s: %s" % (r["rn"], "; ".join(errs)))
    print("  %d runs checked: ARGS (20 flags, none repeated, every design value), ENV (15 keys +"
          " own PROBE_DIR), NODE (job name, id, AUGMENT=1), PROBE_TENSOR (type, tensors=%d, every=100)"
          % (len(runs), NT))
    print("  distinct ENV lines (PROBE_DIR stripped): %d  %s" % (len(env_distinct), list(env_distinct.values())))

    print("\n[3] manifest and probe arity (the runs' own files)")
    man = os.path.join(runsdir, prefix, "PARTITION-MANIFEST.txt")
    tens = []
    norm = None
    for ln in open(man):
        p = ln.split()
        if p and p[0] == "TENSOR":
            tens.append((int(p[1]), p[2], int(p[3])))
        elif p and p[0] == "NORM":
            norm = " ".join(p[1:])
    names = {i: n for i, n, _ in tens}
    print("  manifest: %d TENSOR lines, %d params, NORM %s, names containing 'shortcut': %d"
          % (len(tens), sum(q for _, _, q in tens), norm, sum("shortcut" in n for _, n, _ in tens)))
    if len(tens) != NT or sum(q for _, _, q in tens) != TOT:
        viol.append("manifest arity")
    if prefix == "cgn1" and norm != "GroupNorm 20 BatchNorm2d 0":
        viol.append("manifest NORM %s" % norm)
    if prefix == "cpl1" and any("shortcut" in n for _, n, _ in tens):
        viol.append("cpl1 manifest carries a shortcut tensor")
    traj = {}
    for (a, s), r in sorted(runs.items()):
        pd = os.path.join(runsdir, prefix, "probe_" + r["rn"])
        bs = json.load(open(os.path.join(pd, "block_sizes.json")))
        pt = json.load(open(os.path.join(pd, "probe_tensor.json")))
        recs = [json.loads(x) for x in open(os.path.join(pd, "probe.jsonl")) if x.strip()]
        nb = bs["n_b"]
        want_nb = B["nb"][a]
        ok = (sum(nb) == TOT and (len(nb) == NT if want_nb is None else nb == want_nb)
              and bs["stepsize_type"] == B["ptype"][a] and pt["n_tensors"] == NT
              and all(len(x["z_tensor"]) == NT and len(x["beta"]) == len(nb) for x in recs))
        if not ok:
            viol.append("%s probe arity" % r["rn"])
        stride = recs[1]["step"] - recs[0]["step"] if len(recs) > 1 else 1
        mul = 1 if stride >= 100 else 100
        first_pin = next((x["step"] * mul / 500.0 for x in recs if min(x["beta"]) <= -14.999), None)
        npin_rec = sum(1 for x in recs if x["beta"][0] <= -14.999)
        last = recs[-1]["beta"]
        traj[(a, s)] = (first_pin, npin_rec, len(recs))
        # per-group pin timeline (m <= 2 arms): first epoch at the clamp, the epoch from which the
        # group stays at the clamp to the end, and the fraction of records pinned
        if len(nb) <= 2:
            for g in range(len(nb)):
                col = [x["beta"][g] <= -14.999 for x in recs]
                ep = [x["step"] * mul / 500.0 for x in recs]
                fp = next((ep[i] for i, c in enumerate(col) if c), None)
                k = len(col)
                while k > 0 and col[k - 1]:
                    k -= 1
                stay = ep[k] if k < len(col) else None
                peak = max(x["beta"][g] for x in recs)
                print("      group %d (n_b %d): first at clamp epoch %s, pinned CONTINUOUSLY from epoch %s,"
                      " pinned on %d/%d records, peak beta %.3f"
                      % (g, nb[g], "%.2f" % fp if fp is not None else "never",
                         "%.2f" % stay if stay is not None else "never", sum(col), len(col), peak))
        elif s == B["seeds"][0]:
            pinned_last = [i + 1 for i, b in enumerate(last) if b <= -14.999]
            print("      %s last record: pinned group indices (1-based, = tensor index) %s" % (a, pinned_last))
        print("  %-16s %-9s m=%-2d sum(n_b)=%d n_b=%s  records %d  len(z_tensor) %d  first record with any"
              " beta<=-15 at epoch %s; beta[0] pinned on %d/%d; last record: %d/%d groups pinned, max beta %.3f"
              % (r["rn"], bs["stepsize_type"], len(nb), sum(nb), nb if len(nb) <= 2 else "(%d groups)" % len(nb),
                 len(recs), len(recs[0]["z_tensor"]), "%.2f" % first_pin if first_pin is not None else "never",
                 npin_rec, len(recs), sum(1 for b in last if b <= -14.999), len(last), max(last)))

    if any(r["p5"] is None for r in runs.values()) or len(runs) != len(B["arms"]) * len(B["seeds"]):
        print("\nINCOMPLETE -- no number derived.")
        print("VIOLATIONS %d: %s" % (len(viol), viol))
        sys.exit(2)

    print("\n[4] levels: plateau5 TEST beside TRAIN5, per run and per arm (in batch)")
    M, T, R = {}, {}, {}
    ss, df = 0.0, 0
    for a in B["arms"]:
        v = [runs[(a, s)]["p5"] for s in B["seeds"]]
        w = [runs[(a, s)]["t5"] for s in B["seeds"]]
        M[a], T[a], R[a] = mean(v), mean(w), max(v) - min(v)
        ss += sum((x - M[a]) ** 2 for x in v)
        df += len(v) - 1
        print("  %-4s TEST %.4f  sd %.4f  range %.4f   TRAIN %.4f   seeds %s"
              % (a, M[a], sdev(v), R[a], T[a],
                 ", ".join("s%d %.4f/%.4f" % (s, x, y) for s, x, y in zip(B["seeds"], v, w))))
    print("  TEST trajectory, arm means every 10 epochs:")
    for a in B["arms"]:
        print("    %-4s %s" % (a, " ".join("%.1f" % mean([runs[(a, s)]["ep"][e][1] for s in B["seeds"]])
                                           for e in list(range(0, 100, 10)) + [99])))
    print("  TAIL: per-seed OLS slope over epochs 80..99 (pp/epoch), TEST and TRAIN, and the TEST")
    print("        epoch-to-epoch jitter (sd of first differences over 80..99) -- is plateau5 a plateau?")
    xs = list(range(80, 100))
    xm = mean(xs)
    for a in B["arms"]:
        cells = []
        for s in B["seeds"]:
            ep = runs[(a, s)]["ep"]
            te = [ep[e][1] for e in xs]
            tr = [ep[e][0] for e in xs]
            sl = sum((x - xm) * (y - mean(te)) for x, y in zip(xs, te)) / sum((x - xm) ** 2 for x in xs)
            slt = sum((x - xm) * (y - mean(tr)) for x, y in zip(xs, tr)) / sum((x - xm) ** 2 for x in xs)
            jit = sdev([te[i + 1] - te[i] for i in range(len(te) - 1)])
            cells.append("s%d TEST %+.4f TRAIN %+.4f jitter %.3f" % (s, sl, slt, jit))
        print("    %-4s %s" % (a, " | ".join(cells)))
    sin = math.sqrt(ss / df)
    sig = max(B["sigma_prior"], sin)
    se = sig * math.sqrt(2.0 / 3.0)
    print("\n[5] sigma: in-batch %.6f (df %d); prior %.10f; USED %.6f (%s); SE_ARM_DIFF %.6f"
          % (sin, df, B["sigma_prior"], sig, "in-batch" if sin > B["sigma_prior"] else "prior", se))

    print("\n[6] contrasts and branch, from the registration text re-typed here")
    if prefix == "cgn1":
        D = M["kL"] - M["k01"]
        Dtr = T["kL"] - T["k01"]
        gap_bar, rev_bar = max(10.0, 3 * se), max(2.0, 3 * se)
        print("  D(TEST) = %.4f - %.4f = %+.4f pp = %+.2f SE;  D(TRAIN) = %+.4f pp" % (M["kL"], M["k01"], D, D / se, Dtr))
        print("  D_rel = D/(100-k01) = %.4f;  D/46.548667 = %.4f;  D/30.828 = %.4f"
              % (D / (100 - M["k01"]), D / 46.548667, D / 30.828))
        print("  bars: gap %.4f, reversal %.4f, null 2.0 (ABSENT also needs SE<=1.0); FULL-SIZE needs D >= %.4f"
              % (gap_bar, rev_bar, 0.5 * 46.548667))
        harness = []
        if M["kL"] < 40.0:
            harness.append("H-TRAINS")
        if max(M.values()) > 90.0:
            harness.append("H-CEIL")
        if viol:
            harness.append("HEADER/STRUCT")
        div = [a for a in B["arms"] if R[a] > 5.0]
        if harness:
            br = "HARNESS-UNSOUND"
        elif div:
            br = "UNRESOLVED-DIVERGED"
        elif D >= gap_bar:
            br = "GAP-REPLICATES (%s)" % ("FULL-SIZE" if D >= 0.5 * 46.548667 else "ATTENUATED")
        elif D <= -rev_bar:
            br = "GAP-REVERSED"
        elif abs(D) <= 2.0 and se <= 1.0:
            br = "GAP-ABSENT"
        elif abs(D) <= 2.0:
            br = "UNRESOLVED-NOISY"
        elif D > 2.0:
            br = "GAP-PARTIAL"
        else:
            br = "UNRESOLVED-SMALL-NEGATIVE"
        band = "BELOW" if M["k01"] < 21.694 else ("IN" if M["k01"] <= 24.174 else "ABOVE")
        print("  H-TRAINS kL %.4f >= 40: %s;  H-CEIL max %.4f <= 90: %s;  seed ranges %s"
              % (M["kL"], M["kL"] >= 40, max(M.values()), max(M.values()) <= 90,
                 {a: round(R[a], 4) for a in B["arms"]}))
        print("  k01 vs the BN scalar floor band [21.694, 24.174]: %s;  TRAIN sign %s TEST sign"
              % (band, "agrees with" if (Dtr > 0) == (D > 0) else "DISAGREES with"))
        print("  INDEPENDENT BRANCH: %s" % br)
    else:
        D_GAP = M["kL"] - M["k01"]
        D_ISO = M["ISO"] - M["k01"]
        D_CTL = M["CTL"] - M["k01"]
        D_ONE = M["ONE"] - M["k01"]
        DID = M["ISO"] - M["CTL"]
        DID_TR = T["ISO"] - T["CTL"]
        D_HEAD = M["ISO"] - M["ONE"]
        for lab, x in (("PRIMARY DELTA_ID = ISO-CTL", DID), ("KEY D_ONE = ONE-k01", D_ONE),
                       ("D_ISO = ISO-k01", D_ISO), ("D_CTL = CTL-k01", D_CTL),
                       ("D_HEAD = ISO-ONE", D_HEAD), ("D_GAP = kL-k01", D_GAP)):
            print("  %-28s %+.4f pp = %+.2f SE" % (lab, x, x / se))
        print("  TRAIN: DELTA_ID %+.4f  D_ONE %+.4f  D_ISO %+.4f  D_CTL %+.4f  D_GAP %+.4f"
              % (DID_TR, T["ONE"] - T["k01"], T["ISO"] - T["k01"], T["CTL"] - T["k01"], T["kL"] - T["k01"]))
        print("  RECOVERY D_ISO/D_GAP %.4f   RECOVERY_ONE D_ONE/D_GAP %.4f   (descriptive)"
              % (D_ISO / D_GAP if D_GAP else float("nan"), D_ONE / D_GAP if D_GAP else float("nan")))
        floored = [a for a in ("ISO", "CTL", "ONE") if abs(M[a] - M["k01"]) <= 2.0]
        print("  arms within 2.0 pp of k01 (floor readings = BOUNDS): %s" % floored)
        print("  gates: max arm %.4f >= 15 %s; kL %.4f >= 30 %s; max <= 90 %s; ranges %s"
              % (max(M.values()), max(M.values()) >= 15, M["kL"], M["kL"] >= 30, max(M.values()) <= 90,
                 {a: round(R[a], 4) for a in B["arms"]}))
        if max(M.values()) < 15.0 or viol:
            br = "HARNESS-UNSOUND"
        elif M["kL"] < 30.0:
            br = "PLAIN-UNTRAINABLE"
        elif any(R[a] > 5.0 for a in B["arms"]):
            br = "UNRESOLVED-DIVERGED"
        elif D_GAP < 10.0:
            br = "GAP-ABSENT-PLAIN"
        elif D_GAP < 20.0:
            br = "GAP-ATTENUATED-PLAIN"
        elif max(D_ISO, D_CTL, D_ONE) < 10.0:
            br = "NO-RESCUE-PLAIN"
        elif D_ISO < 10.0:
            br = "UNEXPECTED-PATTERN"
        elif DID <= -15.0:
            br = "CONTROL-DOMINATES-PLAIN"
        elif DID >= 15.0:
            br = ("CARRIER-SURVIVES-PLAIN" if D_ONE >= 10.0 else
                  "HEAD-CARRIES-PLAIN" if D_ONE <= 2.0 else "ONE-PARTIAL-PLAIN")
        elif D_CTL >= 10.0 and abs(DID) <= 2.0:
            br = "CLASS-OPERATIVE-PLAIN"
        else:
            br = "IDENTITY-ATTENUATED-PLAIN"
        print("  H-BROKEN %s in batch (an m=2 arm with D >= 10: %s)"
              % ("EXCLUDED" if max(D_ISO, D_CTL, D_ONE) >= 10.0 else "NOT EXCLUDED",
                 max(D_ISO, D_CTL, D_ONE) >= 10.0))
        print("  INDEPENDENT BRANCH: %s" % br)
        # the k01 nomination readout, re-implemented (pinned = beta[0] at the -15 clamp)
        sh = [0.0] * NT
        fl = {"44": 0.0, "50": 0.0, "44_50": 0.0, "41_47": 0.0}
        sets = {"44": [44], "50": [50], "44_50": [44, 50], "41_47": [41, 47]}
        npin_tot = 0
        for s in B["seeds"]:
            recs = [json.loads(x) for x in open(os.path.join(runsdir, prefix, "probe_%s-k01-s%d" % (prefix, s),
                                                              "probe.jsonl")) if x.strip()]
            pin = [x for x in recs if x["beta"][0] <= -14.999]
            npin_tot += len(pin)
            mass = [0.0] * NT
            cnt = dict((k, 0) for k in sets)
            for x in pin:
                b2 = x.get("pt_b2", 0.9)
                L = [b2 * m + (1 - b2) * z for m, z in zip(x["m_tensor"], x["z_tensor"])]
                S = sum(L)
                for i in range(NT):
                    mass[i] += abs(L[i])
                for k, idx in sets.items():
                    rem = S - sum(L[i - 1] for i in idx)
                    if (rem > 0) != (S > 0):
                        cnt[k] += 1
            tm = sum(mass)
            for i in range(NT):
                sh[i] += mass[i] / tm / 3.0
            for k in sets:
                fl[k] += cnt[k] / len(pin) / 3.0 if pin else 0.0
        top = sorted(range(NT), key=lambda i: -sh[i])[:5]
        print("  k01 nomination (descriptive): %d pinned records over 3 seeds; top shares %s"
              % (npin_tot, ", ".join("%d %s %.4f" % (i + 1, names[i + 1], sh[i]) for i in top)))
        print("  remainder-sign flip fractions: %s" % ", ".join("{%s} %.4f" % (k, v) for k, v in fl.items()))

    print("\nVIOLATIONS %d%s" % (len(viol), (": " + "; ".join(viol)) if viol else ""))
    sys.exit(1 if viol else 0)


if __name__ == "__main__":
    main()
