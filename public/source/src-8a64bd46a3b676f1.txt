#!/usr/bin/env python3
"""cau1_attack_indep.py -- an INDEPENDENT re-derivation of `cau1` (Track U2),
sharing NO code with analysis/cAU1_unaug_denominator_score.py: nothing is
imported from it, no function is copied from it, and the epoch lines are read
by string splitting, not by the scorer's regex.  Every statistic is written
afresh from the registration text (CORRECTIONS 202.4-202.6), not from the
scorer's source.

Usage:
  python3 analysis/cau1_attack_indep.py <runsdir> [--witness DIR]
          [--ref-m6 ub9-b6-sX-J.out] [--ref-m6t i3b-3e4-sX-J.out]
  python3 analysis/cau1_attack_indep.py <runsdir> --calib ub9-b6
          (calibration: plateau5 of every <runsdir>/ub9-b6-s<d>-<job>.out;
           CORRECTIONS 202.2 publishes the in-cell mean 74.7967, n=3)

Reads ONLY raw .out files (and, optionally, the runtime-environ witness files
written by Track U1/U2's read-only poller).  Never reads results/all_runs.csv,
the CSV `plateau` column or best_test.  plateau5 = mean TEST accuracy over
epochs 95..99.  TRAIN is printed beside TEST everywhere.
"""
import math
import os
import sys

N_EP = 100
SEEDS = [50, 51, 52]
RUNGS = [("lr0005", "0.005"), ("lr001", "0.01"), ("lr002", "0.02"), ("lr005", "0.05"),
         ("lr01", "0.1"), ("lr02", "0.2"), ("lr04", "0.4")]
METAS = [("m6", "1e-6", "-60:6.0"), ("m6t", "3e-4", "-15:-2.3026")]
ALL_ARMS = [r for r, _ in RUNGS] + [m for m, _, _ in METAS]
# registration text, 202.6: prior = the larger of the two pooled SDs per statistic
PRIOR_P5 = max(0.446775, 0.120142)
PRIOR_AUC = max(0.385791, 0.162216)
UB9_B6_PUBLISHED = 74.7967
TRAINS_MIN = 40.0
DIVERGE_MAX = 15.0
PRED_FLOOR = 55.0


def avg(xs):
    return sum(xs) / float(len(xs))


def sdev(xs):
    m = avg(xs)
    return math.sqrt(sum((x - m) * (x - m) for x in xs) / (len(xs) - 1))


def read_out(path):
    """-> dict; epoch lines split on ',' and ':' (no regex)."""
    tr, te, dup_ep, bad = {}, {}, [], 0
    args, env, node = [], [], []
    done = tb = False
    minutes = None
    with open(path, errors="replace") as fh:
        for raw in fh:
            ln = raw.rstrip("\n")
            if ln.startswith("Epoch "):
                try:
                    p = ln.split(",")
                    e = int(p[0].split()[1])
                    a = float(p[1].split(":")[1].replace("%", "").strip())
                    b = float(p[2].split(":")[1].replace("%", "").strip())
                except (IndexError, ValueError):
                    bad += 1
                    continue
                if e in te:
                    dup_ep.append(e)
                tr[e], te[e] = a, b
            elif ln.startswith("ARGS:"):
                args.append(ln)
            elif ln.startswith("ENV:"):
                env.append(ln)
            elif ln.startswith("NODE="):
                node.append(ln)
            elif ln.strip() == "RUN_DONE":
                done = True
            elif ln.strip().endswith("minutes"):
                try:
                    minutes = float(ln.split()[0])
                except ValueError:
                    pass
            if "Traceback" in ln:
                tb = True
    return {"tr": tr, "te": te, "dup_ep": dup_ep, "bad": bad, "args": args, "env": env,
            "node": node, "done": done, "tb": tb, "minutes": minutes}


def full(r):
    return (sorted(r["te"]) == list(range(N_EP)) and not r["dup_ep"] and r["done"]
            and not r["tb"] and r["bad"] == 0
            and all(v == v for v in list(r["te"].values()) + list(r["tr"].values())))


def flags_of(line):
    """ARGS line -> (list of (flag, value)).  Values never start with '--'."""
    toks = line.split()[1:]
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            v = None
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                v = toks[i + 1]
                i += 1
            out.append((t, v))
        else:
            out.append(("?STRAY", t))
        i += 1
    return out


def env_of(line):
    d = {}
    for t in line.split()[1:]:
        k, _, v = t.partition("=")
        d[k] = v
    return d


def stats(r):
    te = [r["te"][e] for e in range(N_EP)]
    tr = [r["tr"][e] for e in range(N_EP)]
    p5 = avg(te[95:])
    e99 = next((e for e in range(N_EP) if tr[e] >= 99.0), None)
    e999 = next((e for e in range(N_EP) if tr[e] >= 99.9), None)
    pk = max(range(N_EP), key=lambda e: te[e])
    xs = list(range(80, 100))
    ys = te[80:]
    mx, my = avg(xs), avg(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return {"p5": p5, "t5": avg(tr[95:]), "auc": avg(te), "auct": avg(tr), "e99": e99,
            "e999": e999, "peak": te[pk], "peak_ep": pk, "te_at_e99": te[e99] if e99 is not None else None,
            "slope80": slope, "min_tr_last5": min(tr[95:]), "te_last": te[-1]}


def collect(runsdir, prefix):
    by = {}
    for fn in os.listdir(runsdir):
        if not (fn.startswith(prefix + "-") and fn.endswith(".out")):
            continue
        stem, _, jid = fn[:-4].rpartition("-")
        if not jid.isdigit():
            continue
        by.setdefault(stem, []).append((int(jid), fn))
    return by


def calib(runsdir, cellprefix):
    by = collect(runsdir, cellprefix)
    vals = []
    for stem in sorted(by):
        for jid, fn in sorted(by[stem]):
            r = read_out(os.path.join(runsdir, fn))
            if not full(r):
                print("  %s NOT COMPLETE" % fn)
                continue
            s = stats(r)
            vals.append(s["p5"])
            print("  %-28s plateau5 %.4f  train5 %.4f" % (fn, s["p5"], s["t5"]))
    print("CALIB %s: n=%d mean plateau5 %.4f (published %.4f, diff %+.4f)"
          % (cellprefix, len(vals), avg(vals), UB9_B6_PUBLISHED, avg(vals) - UB9_B6_PUBLISHED))
    return 0


def masked_args(line):
    return [(k, v) for k, v in flags_of(line) if k not in ("--seed", "--run-name", "--save-directory")]


def masked_env(line):
    d = env_of(line)
    d.pop("PROBE_DIR", None)
    return d


def main(argv):
    runsdir = argv[1]
    opt = {}
    i = 2
    while i < len(argv):
        opt[argv[i]] = argv[i + 1]
        i += 2
    if "--calib" in opt:
        return calib(runsdir, opt["--calib"])

    viol = []
    by = collect(runsdir, "cau1")
    print("A. FILES AND SOUNDNESS")
    recs = {}
    for s in SEEDS:
        for a in ALL_ARMS:
            stem = "cau1-%s-s%d" % (a, s)
            files = sorted(by.get(stem, []))
            if not files:
                viol.append("MISSING " + stem)
                continue
            if len(files) > 1:
                print("  NOTE %s has %d files: %s" % (stem, len(files), [f for _, f in files]))
            cands = [(full(read_out(os.path.join(runsdir, f))), j, f) for j, f in files]
            cands.sort()
            ok, jid, fn = cands[-1]
            r = read_out(os.path.join(runsdir, fn))
            r["file"], r["jid"] = fn, jid
            if not ok:
                viol.append("NOT COMPLETE %s (%d epochs, done %s, tb %s, dup %s, bad %d)"
                            % (fn, len(r["te"]), r["done"], r["tb"], r["dup_ep"], r["bad"]))
            recs[(a, s)] = r
    extra = sorted(k for k in by if k not in ["cau1-%s-s%d" % (a, s) for s in SEEDS for a in ALL_ARMS])
    if extra:
        print("  NOTE unexpected cau1 stems: %s" % extra)
    print("  %d/27 present, %d complete" % (len(recs), sum(1 for r in recs.values() if full(r))))

    print("\nB. ARGS / ENV, own spec (202.4 text + the reference .out lines)")
    ref = {}
    for k in ("--ref-m6", "--ref-m6t"):
        if k in opt:
            rr = read_out(opt[k])
            ref[k[6:]] = (masked_args(rr["args"][0]), masked_env(rr["env"][0]), opt[k])
    lr_env_want = {"AUGMENT": "0", "BETA_CLIP": "none", "HIER": "none", "LAM": "na", "ETA_RATIO": "na",
                   "COS_TOTAL": "50000", "COS_WARMUP": "1000", "SCHED": "none", "SCHED_TOTAL": "none",
                   "SCHED_WARMUP": "none", "SCHED_MIN": "none", "PROBE": "0", "PROBE_DIR": "none",
                   "EB_RHO": "na", "EB_LOG": "0"}
    fam_env = {}
    fam_args = {}
    for (a, s), r in sorted(recs.items()):
        tag = r["file"]
        if len(r["args"]) != 1 or len(r["env"]) != 1 or len(r["node"]) != 1:
            viol.append("%s ARGS/ENV/NODE line counts %d/%d/%d" % (tag, len(r["args"]), len(r["env"]), len(r["node"])))
            continue
        fl = flags_of(r["args"][0])
        names = [k for k, _ in fl]
        if len(names) != len(set(names)) or "?STRAY" in names:
            viol.append("%s repeated/stray flag %s" % (tag, names))
        d = dict(fl)
        if d.get("--seed") != str(s) or d.get("--run-name") != "cau1-%s-s%d" % (a, s) \
                or not (d.get("--save-directory") or "").endswith("/runs/cau1"):
            viol.append("%s seed/run-name/save-dir %s %s %s" % (tag, d.get("--seed"), d.get("--run-name"), d.get("--save-directory")))
        e = env_of(r["env"][0])
        if e.get("AUGMENT") != "0":
            viol.append("%s ENV AUGMENT=%s" % (tag, e.get("AUGMENT")))
        if "AUGMENT=0" not in r["node"][0].split("|")[-1]:
            viol.append("%s NODE header %s" % (tag, r["node"][0]))
        if any(k.startswith("SGD_") for k in e):
            viol.append("%s ENV carries SGD_*" % tag)
        if a.startswith("lr"):
            lr = dict(RUNGS)[a]
            want = {"--optimizer": "SGD", "--dataset": "CIFAR10", "--NN-name": "ResNet18", "--batch-size": "100",
                    "--max-time": "999:00:00", "--num-epochs": "100", "--alpha0": lr}
            got = {k: v for k, v in d.items() if k not in ("--seed", "--run-name", "--save-directory")}
            if got != want:
                viol.append("%s ARGS %s != %s" % (tag, got, want))
            if e != lr_env_want:
                viol.append("%s ENV %s" % (tag, sorted(set(e.items()) ^ set(lr_env_want.items()))))
            fam = "lr"
        else:
            alpha, box = [(x[1], x[2]) for x in METAS if x[0] == a][0]
            if d.get("--alpha0") != alpha or d.get("--optimizer") != "HF":
                viol.append("%s alpha0/optimizer %s %s" % (tag, d.get("--alpha0"), d.get("--optimizer")))
            if e.get("BETA_CLIP") != box or e.get("PROBE") != "5" or \
                    not e.get("PROBE_DIR", "").endswith("/cau1/probe_cau1-%s-s%d" % (a, s)):
                viol.append("%s box/probe %s %s %s" % (tag, e.get("BETA_CLIP"), e.get("PROBE"), e.get("PROBE_DIR")))
            if "m6" in ref:
                ra, re_, rf = ref["m6"]
                ma = masked_args(r["args"][0])
                if a == "m6t":
                    ra = [(k, ("3e-4" if k == "--alpha0" else v)) for k, v in ra]
                if sorted(ma, key=str) != sorted(ra, key=str):
                    viol.append("%s ARGS differ from %s (alpha0-adjusted for m6t): %s"
                                % (tag, os.path.basename(rf), sorted(set(ma) ^ set(ra), key=str)))
                elif ma != ra:
                    print("  NOTE %s ARGS equal %s as a flag set but in a different ORDER (argparse-inert)"
                          % (tag, os.path.basename(rf)))
                me = masked_env(r["env"][0])
                re2 = dict(re_)
                if a == "m6t":
                    re2["BETA_CLIP"] = "-15:-2.3026"
                for k in sorted(set(me) | set(re2)):
                    if k in ("EB_RHO", "EB_LOG") and k not in re2:
                        continue
                    if me.get(k) != re2.get(k):
                        viol.append("%s ENV %s=%s vs ref %s" % (tag, k, me.get(k), re2.get(k)))
            fam = a
        fam_env.setdefault(fam, set()).add(tuple(sorted(masked_env(r["env"][0]).items())))
        fam_args.setdefault(fam, set()).add(tuple((k, v) for k, v in masked_args(r["args"][0]) if k != "--alpha0"))
    for fam in sorted(fam_env):
        print("  family %-4s distinct ENV (PROBE_DIR masked) %d, distinct ARGS (seed/name/dir/alpha0 masked) %d"
              % (fam, len(fam_env[fam]), len(fam_args[fam])))
        for ev in fam_env[fam]:
            print("      ENV " + " ".join("%s=%s" % kv for kv in ev))
    if "m6t" in ref:
        ra, re_, rf = ref["m6t"]
        for (a, s), r in sorted(recs.items()):
            if a != "m6t" or not r["args"]:
                continue
            ma = masked_args(r["args"][0])
            dif = sorted(set(ma) ^ set(ra))
            me = masked_env(r["env"][0])
            de = sorted(k for k in set(me) | set(re_) if me.get(k) != re_.get(k))
            print("  m6t s%d vs %s: ARGS sym-diff %s | ENV keys differing %s"
                  % (s, os.path.basename(rf), dif, ["%s:%s->%s" % (k, re_.get(k), me.get(k)) for k in de]))

    print("\nC. RUNTIME ENVIRON WITNESS (SGD_WD / SGD_MOM are not on any ENV line)")
    wdir = opt.get("--witness")
    if wdir and os.path.isdir(wdir):
        cov, zero, pos = 0, 0, 0
        for (a, s), r in sorted(recs.items()):
            fn = os.path.join(wdir, "cau1-%s-s%d-%d.txt" % (a, s, r["jid"]))
            if not os.path.exists(fn):
                print("  no witness: %s" % os.path.basename(fn))
                continue
            kv = {}
            sgd_vars = []
            for ln in open(fn):
                ln = ln.strip()
                if ln.startswith("SGD_VAR "):
                    sgd_vars.append(ln[8:])
                elif "=" in ln and " " not in ln:
                    k, _, v = ln.partition("=")
                    kv[k] = v
            cov += 1
            if kv.get("SGD_count") == "0" and not sgd_vars:
                zero += 1
            else:
                viol.append("WITNESS %s SGD vars %s %s" % (fn, kv.get("SGD_count"), sgd_vars))
            ctl = kv.get("AUGMENT") == "0" and kv.get("SLURM_JOB_ID") == str(r["jid"]) and \
                kv.get("SLURM_JOB_NAME") == "cau1-%s-s%d" % (a, s) and \
                (kv.get("COS_TOTAL") == "50000" if a.startswith("lr") else kv.get("PROBE") == "5")
            pos += ctl
            if not ctl:
                print("  positive control FAILS for %s: %s" % (os.path.basename(fn), kv))
        print("  witness coverage %d/27; SGD_count=0 on %d; positive control present on %d" % (cov, zero, pos))
    else:
        print("  (no --witness directory)")

    if viol:
        print("\nVIOLATIONS (%d):" % len(viol))
        for v in viol:
            print("  " + v)
    if len(recs) < 27 or any(not full(r) for r in recs.values()) or viol:
        print("\nINDEP-FINAL: NOT READABLE (%d violations, %d/27 complete) -- no number printed"
              % (len(viol), sum(1 for r in recs.values() if full(r))))
        return 1

    print("\nD. PER RUN (TEST beside TRAIN)")
    st = {k: stats(r) for k, r in recs.items()}
    print("  %-17s %8s %8s %8s %8s %4s %4s %7s %4s %8s %8s" % ("run", "p5", "train5", "auc", "aucTR", "e99",
                                                              "e999", "peakTE", "@ep", "slope80", "min"))
    for s in SEEDS:
        for a in ALL_ARMS:
            x = st[(a, s)]
            print("  %-17s %8.4f %8.4f %8.4f %8.4f %4s %4s %7.2f %4d %+8.4f %8.2f" % (
                "cau1-%s-s%d" % (a, s), x["p5"], x["t5"], x["auc"], x["auct"], x["e99"], x["e999"],
                x["peak"], x["peak_ep"], x["slope80"], x["min_tr_last5"]))

    print("\nE. PER ARM (3 seeds)")
    arm = {}
    for a in ALL_ARMS:
        xs = [st[(a, s)] for s in SEEDS]
        arm[a] = {k: [x[k] for x in xs] for k in ("p5", "t5", "auc", "auct", "peak")}
        arm[a]["m"] = avg(arm[a]["p5"])
        arm[a]["sd"] = sdev(arm[a]["p5"])
        arm[a]["mauc"] = avg(arm[a]["auc"])
        arm[a]["trains"] = arm[a]["m"] > TRAINS_MIN and min(arm[a]["p5"]) > DIVERGE_MAX
        print("  %-6s p5 %8.4f sd %.4f [%s]  train5 %8.4f  auc %8.4f  aucTR %8.4f  peakTE-p5 %+.3f"
              % (a, arm[a]["m"], arm[a]["sd"], ", ".join("%.2f" % v for v in arm[a]["p5"]),
                 avg(arm[a]["t5"]), arm[a]["mauc"], avg(arm[a]["auct"]),
                 avg(arm[a]["peak"]) - arm[a]["m"]))

    print("\nF. GATES")
    lad = [a for a, _ in RUNGS]
    lm = [arm[a]["m"] for a in lad]
    ist = lm.index(max(lm))
    interior = 0 < ist < len(lad) - 1
    print("  ladder argmax %s (lr %s) %.4f -- %s" % (lad[ist], RUNGS[ist][1], lm[ist],
                                                     "INTERIOR" if interior else "EDGE"))
    lowest = min(min(arm[a]["p5"]) for a in ALL_ARMS)
    print("  floor (202.6 prediction >= %.0f pp): lowest ARM MEAN %.4f, lowest SEED %.4f -> %s"
          % (PRED_FLOOR, min(arm[a]["m"] for a in ALL_ARMS), lowest,
             "HOLDS" if min(arm[a]["m"] for a in ALL_ARMS) >= PRED_FLOOR else "VIOLATED"))
    print("  trains-at-all (mean > 40, no seed <= 15): %s" % {a: arm[a]["trains"] for a in ALL_ARMS})

    print("\nG. PRIMARY GAP_END")
    ss = sum(sum((v - arm[a]["m"]) ** 2 for v in arm[a]["p5"]) for a in ALL_ARMS if arm[a]["trains"])
    df = 2 * sum(1 for a in ALL_ARMS if arm[a]["trains"])
    sig_in = math.sqrt(ss / df)
    sig = max(PRIOR_P5, sig_in)
    se = sig * math.sqrt(2.0 / 3.0)
    ok_meta = [m for m, _, _ in METAS if arm[m]["trains"]]
    mst = max(ok_meta, key=lambda m: arm[m]["m"])
    gap = arm[lad[ist]]["m"] - arm[mst]["m"]
    z = gap / se
    zm6 = (arm[lad[ist]]["m"] - arm["m6"]["m"]) / se
    print("  sigma in-batch %.4f (df %d), prior %.4f -> used %.4f; SE %.4f; 2 SE %.4f"
          % (sig_in, df, PRIOR_P5, sig, se, 2 * se))
    print("  S* %s %.4f | M* %s %.4f | GAP_END %+.4f pp = %+.2f SE | vs m6 %+.2f SE"
          % (lad[ist], arm[lad[ist]]["m"], mst, arm[mst]["m"], gap, z, zm6))
    if not arm[lad[ist]]["trains"]:
        br = "GATED-BASELINE-AT-FLOOR"
    elif z >= 2:
        br = "DEFICIT-HOLDS" + ("" if interior else " | LOWER-BOUND")
    elif z > -2:
        br = "DEFICIT-CLOSES" if interior else "UNRESOLVED-TIE-LADDER-EDGE"
    elif not interior:
        br = "UNRESOLVED-REVERSAL-LADDER-EDGE"
    else:
        br = "REVERSES-AT-PAPER-CONFIG" if zm6 <= -2 else "REVERSES-AT-TUNED-ALPHA0-ONLY"
    sv, mv = arm[lad[ist]]["p5"], arm[mst]["p5"]
    welch = gap / math.sqrt(sdev(sv) ** 2 / 3 + sdev(mv) ** 2 / 3)
    print("  seed separation: min(S*) %.4f vs max(M*) %.4f -> %s; Welch t %.2f"
          % (min(sv), max(mv), "DISJOINT" if min(sv) > max(mv) or max(sv) < min(mv) else "OVERLAP", welch))
    print("  INDEP BRANCH: %s" % br)

    print("\nH. SECONDARY GAP_AUC")
    am = [arm[a]["mauc"] for a in lad]
    ja = am.index(max(am))
    ssa = sum(sum((v - arm[a]["mauc"]) ** 2 for v in arm[a]["auc"]) for a in ALL_ARMS if arm[a]["trains"])
    siga = max(PRIOR_AUC, math.sqrt(ssa / df))
    sea = siga * math.sqrt(2.0 / 3.0)
    mau = max(ok_meta, key=lambda m: arm[m]["mauc"])
    ga = arm[lad[ja]]["mauc"] - arm[mau]["mauc"]
    za = ga / sea
    inta = 0 < ja < len(lad) - 1
    if za >= 2:
        cb = "CURVE-DEFICIT" + ("" if inta else "-LOWER-BOUND")
    elif za > -2:
        cb = "CURVE-TIE" if inta else "CURVE-UNRESOLVED-TIE-EDGE"
    else:
        cb = "CURVE-REVERSES" if inta else "CURVE-UNRESOLVED-REVERSAL-EDGE"
    print("  S^auc %s %.4f [%s] | M^auc %s %.4f | GAP_AUC %+.4f = %+.2f SE_AUC (SE_AUC %.4f) | %s"
          % (lad[ja], am[ja], "INTERIOR" if inta else "EDGE", mau, arm[mau]["mauc"], ga, za, sea, cb))
    print("  GAP_AUC at S* instead: %+.4f" % (arm[lad[ist]]["mauc"] - arm[mau]["mauc"]))

    print("\nI. NON-GATING")
    print("  m6 vs ub9-b6 published %.4f: %.4f -> %+.4f pp (%+.2f SE)"
          % (UB9_B6_PUBLISHED, arm["m6"]["m"], arm["m6"]["m"] - UB9_B6_PUBLISHED,
             (arm["m6"]["m"] - UB9_B6_PUBLISHED) / se))
    print("\nJ. ROBUSTNESS (non-gating; none moves the branch)")
    ms = arm[mst]["m"]
    n_hi = 0
    for i, a in enumerate(lad):
        g = arm[a]["m"] - ms
        n_hi += g / se >= 2
        print("  rung %-6s (lr %-5s) p5 %.4f  vs M* %+.4f pp = %+6.2f SE | vs m6 %+.4f | AUC %.4f vs M^auc %+.4f = %+6.2f SE_AUC"
              % (a, RUNGS[i][1], arm[a]["m"], g, g / se, arm[a]["m"] - arm["m6"]["m"],
                 arm[a]["mauc"], arm[a]["mauc"] - arm[mau]["mauc"], (arm[a]["mauc"] - arm[mau]["mauc"]) / sea))
    print("  rungs beating M* by >= 2 SE on plateau5: %d of %d" % (n_hi, len(lad)))
    print("  rungs beating M^auc on AUC: %d of %d; by >= 2 SE_AUC: %d"
          % (sum(arm[a]["mauc"] > arm[mau]["mauc"] for a in lad), len(lad),
             sum((arm[a]["mauc"] - arm[mau]["mauc"]) / sea >= 2 for a in lad)))
    bias7 = 1.3522 * sig / math.sqrt(3.0)
    print("  worst-case max-of-7 selection on S* (%.4f pp) removed: GAP_END %+.4f = %+.2f SE"
          % (bias7, gap - bias7, (gap - bias7) / se))
    bias7a = 1.3522 * siga / math.sqrt(3.0)
    print("  worst-case max-of-7 selection on S^auc (%.4f pp) removed: GAP_AUC %+.4f = %+.2f SE_AUC"
          % (bias7a, ga - bias7a, (ga - bias7a) / sea))
    dm = avg(arm[mst]["peak"]) - ms
    print("  post-memorisation decay bound: credit M* its mean peak-minus-plateau (%+.4f) and S* nothing:"
          " GAP_END %+.4f = %+.2f SE" % (dm, gap - dm, (gap - dm) / se))
    print("  best-test (BANNED as a primary; a bound only): mean peak S* %.4f vs M* %.4f -> %+.4f"
          % (avg(arm[lad[ist]]["peak"]), avg(arm[mst]["peak"]), avg(arm[lad[ist]]["peak"]) - avg(arm[mst]["peak"])))
    loo = []
    for i in range(3):
        for j in range(3):
            sv2 = [v for k, v in enumerate(arm[lad[ist]]["p5"]) if k != i]
            mv2 = [v for k, v in enumerate(arm[mst]["p5"]) if k != j]
            loo.append(avg(sv2) - avg(mv2))
    print("  leave-one-seed-out (9 combos, S* and M* each drop one): GAP_END in [%+.4f, %+.4f]"
          % (min(loo), max(loo)))
    for a in ALL_ARMS:
        xs = [st[(a, s)] for s in SEEDS]
        print("  memorisation %-6s TRAIN>=99%% at ep %s, >=99.9%% at ep %s; epochs trained after interpolation ~%d;"
              " TEST slope ep80-99 %+.4f pp/ep"
              % (a, [x["e99"] for x in xs], [x["e999"] for x in xs], 100 - max(x["e999"] for x in xs),
                 avg([x["slope80"] for x in xs])))
    print("\nINDEP-FINAL: %s | %s | M*=%s | %s" % (br, "INTERIOR" if interior else "EDGE", mst, cb))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
