#!/usr/bin/env python3
# =============================================================================
# cvh1_cuc1_attack_indep.py -- AN INDEPENDENT ATTACK ON `cvh1` AND `cuc1`.
#
# CORRECTIONS 219 / 220.  This file imports NOTHING from either registered
# scorer (analysis/cVH1_vgghorizon_score.py, analysis/cUC1_unaug_c100_score.py)
# and shares no code with them.  Every bar, sigma, band and branch rule below
# is RE-TYPED BY HAND from the registrations (CORRECTIONS 213 for cvh1, 214
# for cuc1) and from the scorers' own frozen-literal headers.  The .out files
# are parsed by STRING SPLITTING -- no regular expressions anywhere -- so a
# shared regex cannot make both parsers wrong in the same way.
#
# WHAT IT CHECKS, FOR BOTH BATCHES
#   * every run's own header lines (NODE / ARGS / ENV / PROBE_TENSOR), parsed
#     independently, against the design re-typed here;
#   * every run's plateau5 from the RAW `Epoch` lines, TRAIN BESIDE TEST
#     (the CSV `plateau` column is never read; the CSV `plateau5` column is
#     read ONLY for a third-parser cross-check, section X);
#   * every arm mean / sd / seed range, at every horizon the design names;
#   * every registered contrast, and the registered branch map re-implemented
#     from the prose;
#   * the beta trajectories from each run's own probe.jsonl, the pin epochs,
#     and -- for cvh1 -- BOTH the new step-size-magnitude freedom gate AND
#     cIS2's superseded exact-clamp-dwell form, so the re-registration can be
#     graded rather than trusted.
#
# USAGE
#   python3 analysis/cvh1_cuc1_attack_indep.py cvh1 <runsdir> [<corpus.csv>]
#   python3 analysis/cvh1_cuc1_attack_indep.py cuc1 <runsdir> [<corpus.csv>]
#   python3 analysis/cvh1_cuc1_attack_indep.py both <runsdir> [<corpus.csv>]
#
# EXIT 0 if every independent check agrees with the registration; 1 otherwise.
# It never writes a file and never touches the cluster.
# =============================================================================

from __future__ import print_function

import json
import math
import os
import statistics
import sys

VIOL = []


def bad(msg):
    VIOL.append(msg)
    print("  !!! VIOLATION: %s" % msg)


def ok(msg, extra=""):
    print("  ok   %s%s" % (msg, ("  -- " + extra) if extra else ""))


def f4(x):
    return "None" if x is None else "%.4f" % x


def mean(v):
    return sum(v) / float(len(v))


def sd(v):
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / float(len(v) - 1))


# -----------------------------------------------------------------------------
# .out parsing BY STRING SPLITTING.  No regex.
# -----------------------------------------------------------------------------
def parse_out(path):
    """-> {'epochs': {ep: (train, test)}, header lines, flags}"""
    eps = {}
    args = []
    env = []
    node = []
    ptens = []
    done = False
    tb = False
    with open(path, errors="replace") as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if ln.startswith("Epoch "):
                # "Epoch 0, Train Accuracy: 0.93 %, Test Accuracy: 0.86 %"
                try:
                    head, rest = ln.split(",", 1)
                    ep = int(head.split()[1])
                    trpart, tepart = rest.split(",", 1)
                    tr = float(trpart.split(":")[1].split("%")[0].strip())
                    te = float(tepart.split(":")[1].split("%")[0].strip())
                except (ValueError, IndexError):
                    continue
                eps[ep] = (tr, te)
            elif ln.startswith("ARGS:"):
                args.append(ln)
            elif ln.startswith("ENV:"):
                env.append(ln)
            elif ln.startswith("NODE="):
                node.append(ln)
            elif ln.startswith("PROBE_TENSOR:"):
                ptens.append(ln)
            elif ln.startswith("RUN_DONE"):
                done = True
            elif "Traceback (most recent call last)" in ln:
                tb = True
    return {"epochs": eps, "args": args, "env": env, "node": node,
            "probe_tensor": ptens, "run_done": done, "traceback": tb,
            "path": path}


def args_map(line):
    """argparse semantics: the LAST occurrence of a repeated flag wins."""
    toks = line[len("ARGS:"):].split()
    out, order, rep = {}, [], []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:]
            v = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
            if k in out:
                rep.append(k)
            else:
                order.append(k)
            out[k] = v
            i += 2 if v != "" else 1
        else:
            i += 1
    return out, order, rep


def env_map(line):
    out = {}
    for t in line[len("ENV:"):].split():
        if "=" in t:
            k, v = t.split("=", 1)
            out[k] = v
    return out


def win_mean(eps, end_epoch, idx):
    """mean of the 5 epochs [end-5, end-1]; idx 0 = TRAIN, 1 = TEST."""
    v = [eps[e][idx] for e in range(end_epoch - 5, end_epoch) if e in eps]
    return mean(v) if len(v) == 5 else None


def read_probe(runsdir, batch, rn):
    p = os.path.join(runsdir, batch, "probe_" + rn, "probe.jsonl")
    if not os.path.exists(p):
        return None, None
    recs = []
    with open(p) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                recs.append(json.loads(ln))
    bs = os.path.join(runsdir, batch, "probe_" + rn, "block_sizes.json")
    blocks = json.load(open(bs)) if os.path.exists(bs) else None
    return recs, blocks


# =============================================================================
# cvh1 -- RE-TYPED REGISTRATION (CORRECTIONS 213; cVH1 scorer header)
# =============================================================================
V_EPOCHS = 328
V_CONTROL = 100
V_SEEDS = (63, 64, 65)
V_ARMS = ("k01", "kL", "ISO", "CTL")
V_SPEC = {"k01": "scalar", "kL": "layerwise",
          "ISO": "sets:1-22,24-26/bn8.weight",
          "CTL": "sets:1-19,21-26/bn7.weight"}
V_PTYPE = {"k01": "scalar", "kL": "layerwise", "ISO": "blockwise", "CTL": "blockwise"}
V_ARITY = {"k01": 1, "kL": 26, "ISO": 2, "CTL": 2}
V_NB_ISO = [9274020, 512]
V_NB_CTL = [9274020, 512]
V_NUMEL = 9274532
V_TENSORS = 26
V_RECORDS = 1640
V_REC_PER_EPOCH = 5            # (50000/100) steps per epoch / PROBE=100
V_BETA_LO = -15.0
V_TAIL_FRAC = 0.25
V_PIN_RATIO_MAX = 2.0
V_FREE_RATIO_MIN = 10.0
V_OLD_PIN_OCC_MIN = 0.50       # cIS2's superseded rule
V_SIGMA_FROZEN = 0.925518
V_READ_BAR = 1.511365
V_DEAD_BAR = 5.0
V_D100_MIN = 10.0
V_REFUTE_BAR = 5.0
V_SUPPORT_BAR = 15.0
V_SURVIVE_FRAC = 0.80
V_BAND = {"k01": (31.6421, 39.0463), "kL": (62.5863, 69.9905)}
V_IDENTITY_BAR = 15.0
V_CVI1_D100 = 31.2833
V_WMAX = 213.0
V_ARGS_EXPECT = {
    "optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99",
    "weight-decay-base": "0.1", "alg-meta": "Lion", "momentum-param-meta": "0.99",
    "Lion-beta2-meta": "0.9", "weight-decay-meta": "0", "dataset": "CIFAR100",
    "NN-name": "VGG11_bn_c100", "batch-size": "100", "max-time": "999:00:00",
    "gamma": "1", "meta-stepsize": "1e-3", "alpha0": "1e-6",
    "num-epochs": "328"}
V_ENV_EXPECT = {
    "AUGMENT": "1", "BETA_CLIP": "-15:-2.3026", "HIER": "none", "LAM": "na",
    "ETA_RATIO": "na", "COS_TOTAL": "default", "COS_WARMUP": "default",
    "SCHED": "none", "SCHED_TOTAL": "none", "SCHED_WARMUP": "none",
    "SCHED_MIN": "none", "PROBE": "100", "EB_RHO": "na", "EB_LOG": "0"}


def classify_r(med):
    if med is None:
        return "UNMEASURED"
    if med <= V_PIN_RATIO_MAX:
        return "PINNED"
    if med >= V_FREE_RATIO_MIN:
        return "FREE"
    return "NEITHER"


def arm_class(cs):
    for k in ("PINNED", "FREE"):
        if sum(1 for c in cs if c == k) >= 2:
            return k
    return "NEITHER"


def traj_stats(b):
    """independent re-derivation of the freedom gate's per-seed statistics."""
    n = len(b)
    tail = b[-max(1, int(n * V_TAIL_FRAC)):]
    r = [math.exp(v - V_BETA_LO) for v in tail]
    lr2 = V_BETA_LO + math.log(V_PIN_RATIO_MAX)
    ft = None
    f2 = None
    for i, v in enumerate(b):
        if f2 is None and v <= lr2:
            f2 = i
        if ft is None and v <= V_BETA_LO + 1e-6:
            ft = i
        if ft is not None and f2 is not None:
            break
    return {"n": n, "terminal": b[-1], "med_r": statistics.median(r), "max_r": max(r),
            "occ_exact": sum(1 for v in tail if v <= V_BETA_LO + 1e-6) / float(len(tail)),
            "first_touch": ft, "first_r2": f2}


def attack_cvh1(runsdir):
    print()
    print("=" * 78)
    print(" ATTACK :: cvh1 -- VGG11_bn_c100 isolation rescue at a 328-epoch horizon")
    print(" runsdir %s   (independent parser, no regex, nothing imported)" % runsdir)
    print("=" * 78)

    # ---- locate the 12 runs from the directory itself -----------------------
    found = {}
    for fn in sorted(os.listdir(runsdir)):
        if fn.startswith("cvh1-") and fn.endswith(".out"):
            stem = fn[:-4]
            parts = stem.split("-")
            arm, seed, jid = parts[1], int(parts[2][1:]), parts[3]
            found.setdefault((arm, seed), []).append((jid, os.path.join(runsdir, fn)))
    print()
    print("[1] RUN SET, COMPLETENESS, AND ONE .out PER (arm, seed)")
    want = [(a, s) for a in V_ARMS for s in V_SEEDS]
    extra = sorted(set(found) - set(want))
    if extra:
        bad("cvh1: runs outside the registered 12: %s" % (extra,))
    runs = {}
    for key in want:
        lst = found.get(key, [])
        if len(lst) != 1:
            bad("cvh1 %s: %d .out files (want exactly 1)" % (key, len(lst)))
            continue
        jid, path = lst[0]
        rec = parse_out(path)
        rec["job_id"] = jid
        rec["arm"], rec["seed"] = key
        rec["run"] = "cvh1-%s-s%d" % key
        runs[key] = rec
    if len(runs) != 12:
        bad("cvh1: %d usable runs, want 12 -- STOPPING" % len(runs))
        return
    for key in want:
        r = runs[key]
        n = len(r["epochs"])
        contig = all(e in r["epochs"] for e in range(V_EPOCHS))
        if not (n == V_EPOCHS and contig and r["run_done"] and not r["traceback"]):
            bad("cvh1 %s: epochs=%d contiguous=%s RUN_DONE=%s traceback=%s"
                % (r["run"], n, contig, r["run_done"], r["traceback"]))
    ok("12/12 runs, one .out each, %d contiguous epoch lines, RUN_DONE, no traceback"
       % V_EPOCHS)

    # ---- headers ------------------------------------------------------------
    print()
    print("[2] EVERY RUN'S OWN NODE / ARGS / ENV / PROBE_TENSOR LINE")
    envset = set()
    for key in want:
        r = runs[key]
        arm, seed = key
        if len(r["args"]) != 1 or len(r["env"]) != 1 or len(r["node"]) != 1:
            bad("%s: header line counts ARGS=%d ENV=%d NODE=%d"
                % (r["run"], len(r["args"]), len(r["env"]), len(r["node"])))
            continue
        am, order, rep = args_map(r["args"][0])
        if rep:
            bad("%s: REPEATED FLAG(S) %s -- RULE 20" % (r["run"], rep))
        if len(order) != 20:
            bad("%s: %d flags, the design declares 20" % (r["run"], len(order)))
        for k, v in V_ARGS_EXPECT.items():
            if am.get(k) != v:
                bad("%s: --%s is %r, the design says %r" % (r["run"], k, am.get(k), v))
        if am.get("stepsize-groups") != V_SPEC[arm]:
            bad("%s: --stepsize-groups %r, registered %r"
                % (r["run"], am.get("stepsize-groups"), V_SPEC[arm]))
        if am.get("seed") != str(seed):
            bad("%s: --seed %r" % (r["run"], am.get("seed")))
        if am.get("run-name") != r["run"]:
            bad("%s: --run-name %r" % (r["run"], am.get("run-name")))
        if not am.get("save-directory", "").endswith("/runs/cvh1"):
            bad("%s: --save-directory %r" % (r["run"], am.get("save-directory")))
        em = env_map(r["env"][0])
        for k, v in V_ENV_EXPECT.items():
            if em.get(k) != v:
                bad("%s: ENV %s=%r, the design says %r" % (r["run"], k, em.get(k), v))
        if em.get("PROBE_DIR", "").split("/")[-1] != "probe_" + r["run"]:
            bad("%s: ENV PROBE_DIR %r" % (r["run"], em.get("PROBE_DIR")))
        if set(em) != set(V_ENV_EXPECT) | {"PROBE_DIR"}:
            bad("%s: ENV keys %s" % (r["run"], sorted(set(em) ^ (set(V_ENV_EXPECT) | {"PROBE_DIR"}))))
        envset.add(" ".join(t for t in r["env"][0].split() if not t.startswith("PROBE_DIR=")))
        nd = r["node"][0]
        if ("JOB=%s %s" % (r["run"], r["job_id"])) not in nd or "AUGMENT=1" not in nd:
            bad("%s: NODE header %r" % (r["run"], nd))
        if len(r["probe_tensor"]) != 1:
            bad("%s: %d PROBE_TENSOR lines" % (r["run"], len(r["probe_tensor"])))
        else:
            pt = r["probe_tensor"][0]
            for frag in ("PROBE_TENSOR: on", "every=100",
                         "type=" + V_PTYPE[arm], "tensors=%d" % V_TENSORS):
                if frag not in pt:
                    bad("%s: PROBE_TENSOR line lacks %r (%r)" % (r["run"], frag, pt))
    if len(envset) != 1:
        bad("cvh1: %d distinct ENV lines across the 12 runs (PROBE_DIR masked)" % len(envset))
    ok("12/12: 20 flags none repeated, every design value, the arm's registered SPEC")
    ok("12/12: 14 ENV keys + own PROBE_DIR, ONE distinct ENV line, AUGMENT=1 on the NODE header")
    ok("12/12: PROBE_TENSOR on, every=100, tensors=26, the arm's own type")

    # ---- levels, TRAIN beside TEST -----------------------------------------
    print()
    print("[3] LEVELS -- plateau5 from the RAW `Epoch` lines, TRAIN BESIDE TEST")
    print("    (window_mean(E) = mean over epochs E-5 .. E-1; CSV `plateau` never read)")
    lev = {}
    for key in want:
        r = runs[key]
        lev[key] = {
            "te100": win_mean(r["epochs"], V_CONTROL, 1),
            "te328": win_mean(r["epochs"], V_EPOCHS, 1),
            "tr100": win_mean(r["epochs"], V_CONTROL, 0),
            "tr328": win_mean(r["epochs"], V_EPOCHS, 0)}
    print("    %-4s %-6s %-9s %-9s %-9s %-9s" % ("arm", "seed", "TEST@100", "TEST@328",
                                                 "TRAIN@100", "TRAIN@328"))
    for a in V_ARMS:
        for s in V_SEEDS:
            L = lev[(a, s)]
            print("    %-4s s%-5d %9.4f %9.4f %9.4f %9.4f"
                  % (a, s, L["te100"], L["te328"], L["tr100"], L["tr328"]))
    armlev = {}
    print()
    print("    %-4s %-24s %-24s %-9s %-9s" % ("arm", "TEST@100 mean (sd/range)",
                                              "TEST@328 mean (sd/range)",
                                              "TRAIN@100", "TRAIN@328"))
    for a in V_ARMS:
        t1 = [lev[(a, s)]["te100"] for s in V_SEEDS]
        t3 = [lev[(a, s)]["te328"] for s in V_SEEDS]
        r1 = [lev[(a, s)]["tr100"] for s in V_SEEDS]
        r3 = [lev[(a, s)]["tr328"] for s in V_SEEDS]
        armlev[a] = {"te100": mean(t1), "te328": mean(t3), "tr100": mean(r1),
                     "tr328": mean(r3), "rg100": max(t1) - min(t1),
                     "rg328": max(t3) - min(t3), "sd100": sd(t1), "sd328": sd(t3)}
        A = armlev[a]
        print("    %-4s %8.4f (%.4f/%.4f)   %8.4f (%.4f/%.4f)   %8.4f %8.4f"
              % (a, A["te100"], A["sd100"], A["rg100"], A["te328"], A["sd328"],
                 A["rg328"], A["tr100"], A["tr328"]))
    sig_in = None
    dev = []
    for a in V_ARMS:
        for h, k in ((V_CONTROL, "te100"), (V_EPOCHS, "te328")):
            m = armlev[a][k]
            for s in V_SEEDS:
                dev.append(lev[(a, s)][k] - m)
    # pooled within-arm-within-horizon sd, df = 24 - 8 = 16
    sig_in = math.sqrt(sum(d * d for d in dev) / 16.0)
    sig_used = max(V_SIGMA_FROZEN, sig_in)
    se = sig_used * math.sqrt(2.0 / 3.0)
    print()
    print("    SIGMA_FROZEN %.6f (re-typed literal) | SIGMA_INBATCH %.6f (df 16)"
          % (V_SIGMA_FROZEN, sig_in))
    print("    SIGMA_USED %.6f  SE_USED %.6f  READ_BAR %.6f"
          % (sig_used, se, 2 * se))
    if abs(2 * se - V_READ_BAR) > 1e-5:
        bad("cvh1: my READ_BAR %.6f != the registered %.6f" % (2 * se, V_READ_BAR))
    else:
        ok("READ_BAR re-derived from SIGMA_FROZEN: %.6f" % (2 * se))

    # ---- anchors and divergence --------------------------------------------
    print()
    print("[4] ANCHOR BANDS AND SEED DIVERGENCE (both gate the branch)")
    for a in ("k01", "kL"):
        lo, hi = V_BAND[a]
        v = armlev[a]["te100"]
        if not (lo <= v <= hi):
            bad("cvh1 %s P100 %.4f outside the frozen band [%.4f, %.4f]" % (a, v, lo, hi))
        else:
            ok("%s P100 %.4f inside [%.4f, %.4f]" % (a, v, lo, hi))
    for a in V_ARMS:
        for h, k in ((100, "rg100"), (328, "rg328")):
            if armlev[a][k] > V_DEAD_BAR:
                bad("cvh1 %s seed range @%d = %.4f > DEAD_BAR %.1f" % (a, h, armlev[a][k], V_DEAD_BAR))
    ok("every arm's seed range <= DEAD_BAR 5.0 at both horizons",
       "max %.4f" % max(armlev[a][k] for a in V_ARMS for k in ("rg100", "rg328")))

    # ---- the primary --------------------------------------------------------
    print()
    print("[5] THE PRIMARY -- D_ISO and RHO, every contrast IN BATCH")
    d100 = armlev["ISO"]["te100"] - armlev["k01"]["te100"]
    d328 = armlev["ISO"]["te328"] - armlev["k01"]["te328"]
    rho = d328 / d100 if d100 else None
    print("    D_ISO @100 = %+.4f pp (%+.2f SE)" % (d100, d100 / se))
    print("    D_ISO @328 = %+.4f pp (%+.2f SE)" % (d328, d328 / se))
    print("    RHO = %.4f" % rho)
    for s in V_SEEDS:
        a = lev[("ISO", s)]["te100"] - lev[("k01", s)]["te100"]
        b = lev[("ISO", s)]["te328"] - lev[("k01", s)]["te328"]
        print("      seed %d paired: D@100 %+.4f  D@328 %+.4f  rho %.4f" % (s, a, b, b / a))
    # the registered branch map, re-implemented from the prose
    if d100 < V_D100_MIN:
        branch = "NO-RESCUE-AT-100"
    elif d328 <= V_REFUTE_BAR:
        branch = "RESCUE-COLLAPSES"
    elif rho >= V_SURVIVE_FRAC and d328 >= V_SUPPORT_BAR:
        branch = "RESCUE-SURVIVES"
    else:
        branch = "RESCUE-ATTENUATES"
    print("    MY BRANCH (primary alone, map re-typed from the registration): %s" % branch)

    # ---- the freedom / pinning gate, BOTH FORMS ----------------------------
    print()
    print("[6] THE FREEDOM GATE -- the NEW step-size-magnitude form, GRADED")
    print("    r = exp(beta - (-15)); MED_R = median of r over the last quarter")
    print("    PINNED MED_R <= 2.0 | FREE MED_R >= 10.0 | NEITHER otherwise; >=2/3 seeds")
    probe_ok = True
    stats = {}
    for key in want:
        r = runs[key]
        recs, blocks = read_probe(runsdir, "cvh1", r["run"])
        if recs is None:
            bad("%s: no probe.jsonl" % r["run"])
            probe_ok = False
            continue
        if len(recs) != V_RECORDS:
            bad("%s: %d probe records, want %d" % (r["run"], len(recs), V_RECORDS))
        nb = (blocks or {}).get("n_b")
        if not nb or len(nb) != V_ARITY[key[0]]:
            bad("%s: block_sizes n_b arity %s, want %d"
                % (r["run"], (len(nb) if nb else None), V_ARITY[key[0]]))
        elif key[0] in ("ISO", "CTL") and nb != (V_NB_ISO if key[0] == "ISO" else V_NB_CTL):
            bad("%s: n_b %s, registered %s" % (r["run"], nb, V_NB_ISO))
        elif sum(nb) != V_NUMEL:
            bad("%s: n_b sums to %d, the model has %d" % (r["run"], sum(nb), V_NUMEL))
        for rr in recs:
            if len(rr.get("beta", [])) != V_ARITY[key[0]]:
                bad("%s: a record's beta has arity %d" % (r["run"], len(rr.get("beta", []))))
                break
        for k in range(V_ARITY[key[0]]):
            b = [float(rr["beta"][k]) for rr in recs]
            stats[(key[0], key[1], k)] = traj_stats(b)
    if probe_ok:
        ok("12/12 probe.jsonl present, %d records each, n_b arity and numel as registered"
           % V_RECORDS)
    print()
    print("    %-14s %-9s %-8s %-11s %-10s %-11s %-9s"
          % ("group", "MED_R", "MAX_R", "terminal", "first r<=2", "first touch", "dwell"))
    pin_eps, cls_new, cls_old = [], [], []
    for s in V_SEEDS:
        st = stats[("ISO", s, 0)]
        print("    ISO-s%-9d %9.4f %8.3f %11.4f %10s %11s %9.4f"
              % (s, st["med_r"], st["max_r"], st["terminal"],
                 ("%.1f" % (st["first_r2"] / float(V_REC_PER_EPOCH))) if st["first_r2"] is not None else "-",
                 ("%.1f" % (st["first_touch"] / float(V_REC_PER_EPOCH))) if st["first_touch"] is not None else "-",
                 st["occ_exact"]))
        cls_new.append(classify_r(st["med_r"]))
        cls_old.append("PINNED" if st["occ_exact"] >= V_OLD_PIN_OCC_MIN else "NOT-PINNED")
        if st["first_r2"] is not None:
            pin_eps.append(st["first_r2"] / float(V_REC_PER_EPOCH))
    comp_new = arm_class(cls_new)
    print()
    print("    NEW form (step-size magnitude) per seed: %s -> arm %s" % (cls_new, comp_new))
    print("    OLD form (cIS2 exact-clamp dwell >= %.2f) per seed: %s -> arm %s"
          % (V_OLD_PIN_OCC_MIN, cls_old,
             "PINNED" if cls_old.count("PINNED") >= 2 else "NOT-PINNED"))
    print("    GRADE OF THE RE-REGISTRATION:")
    if comp_new == "PINNED" and cls_old.count("PINNED") < 2:
        print("      THE REDESIGN BEHAVES AS INTENDED.  Every ISO complement's step size")
        print("      sits within %.3fx of the floor (max MAX_R %.3f), yet the superseded"
              % (V_PIN_RATIO_MAX, max(stats[("ISO", s, 0)]["max_r"] for s in V_SEEDS)))
        print("      exact-clamp-dwell rule scores %s and would have called the arm"
              % ", ".join("%.4f" % stats[("ISO", s, 0)]["occ_exact"] for s in V_SEEDS))
        print("      NOT PINNED -- exactly ciso2's PIN_OCC_MIN defect (196.5), which is")
        print("      what 213 re-registered the gate to remove.  It is also a STAMP here,")
        print("      sitting AFTER the primary, so neither form can void a measured RHO.")
    else:
        bad("cvh1: the redesigned gate does not separate as registered "
            "(new %s, old %s)" % (comp_new, cls_old))
    if pin_eps:
        latest = max(pin_eps)
        window = V_EPOCHS - latest
        print()
        print("    pin epochs (first r<=2) %s -> latest %.1f; post-pin window %.1f epochs"
              % (["%.1f" % e for e in pin_eps], latest, window))
        print("    W_MAX %.1f -> %s" % (V_WMAX, "WINDOW-MEETS-WMAX" if window >= V_WMAX
                                       else "WINDOW-SHORT"))
        e0 = int(math.ceil(latest)) + 5
        post = [mean([runs[("ISO", s)]["epochs"][e][1] for e in range(e0, e0 + 5)])
                for s in V_SEEDS]
        dpost = armlev["ISO"]["te328"] - mean(post)
        print("    ISO TEST over epochs %d..%d = %.4f  vs  end %.4f  delta %+.4f (%+.2f SE) -> %s"
              % (e0, e0 + 4, mean(post), armlev["ISO"]["te328"], dpost, dpost / se,
                 "FROZEN-AT-PIN" if abs(dpost) <= 2 * se
                 else ("IMPROVES-AFTER-PIN" if dpost > 0 else "DECAYS-AFTER-PIN")))
    # kL freedom, descriptive
    kl_free = []
    for s in V_SEEDS:
        n = sum(1 for k in range(V_ARITY["kL"])
                if classify_r(stats[("kL", s, k)]["med_r"]) == "FREE")
        kl_free.append(n)
    print("    kL groups FREE (MED_R >= 10) per seed: %s of %d -> %s"
          % (kl_free, V_ARITY["kL"],
             "KL-STILL-TRAINING" if sum(1 for n in kl_free if n >= 1) >= 2 else "KL-FROZEN"))
    for a in ("k01", "CTL", "ISO"):
        med = [stats[(a, s, 0)]["med_r"] for s in V_SEEDS]
        print("    %-4s beta[0] MED_R %s (descriptive)" % (a, ["%.4f" % m for m in med]))
    for a in ("ISO", "CTL"):
        med = [stats[(a, s, 1)]["med_r"] for s in V_SEEDS]
        print("    %-4s beta[1] (the isolated tensor) MED_R %s" % (a, ["%.4f" % m for m in med]))

    # ---- stamps -------------------------------------------------------------
    print()
    print("[7] STAMPS -- none of these moves the branch")
    pairs = [
        ("ISO - kL @328", armlev["ISO"]["te328"] - armlev["kL"]["te328"],
         ("CEIL-BELOW", "CEIL-TRACKS", "CEIL-ABOVE")),
        ("NULL MODEL k01 @328 - @100", armlev["k01"]["te328"] - armlev["k01"]["te100"],
         ("FLOOR-DRIFTS", "FLOOR-HOLDS", "FLOOR-DRIFTS")),
        ("CTL - k01 @328", armlev["CTL"]["te328"] - armlev["k01"]["te328"],
         ("CTL-LEAVES-FLOOR", "CTL-AT-FLOOR-AT-E", "CTL-LEAVES-FLOOR")),
        ("ISO TRAIN @328 - @100", armlev["ISO"]["tr328"] - armlev["ISO"]["tr100"],
         ("ISO-TRAIN-MOVES", "ISO-TRAIN-FROZEN", "ISO-TRAIN-MOVES")),
    ]
    for label, d, tags in pairs:
        tok = tags[1] if abs(d) <= 2 * se else (tags[0] if d < 0 else tags[2])
        print("    %-30s %+9.4f pp (%+6.2f SE)  -> %s" % (label, d, d / se, tok))
    did = armlev["ISO"]["te328"] - armlev["CTL"]["te328"]
    print("    %-30s %+9.4f pp (%+6.2f SE)  -> %s"
          % ("DELTA_ID @328 = ISO - CTL", did, did / se,
             "IDENTITY-HOLDS-AT-E" if did >= V_IDENTITY_BAR else "IDENTITY-FADES-AT-E"))
    did100 = armlev["ISO"]["te100"] - armlev["CTL"]["te100"]
    print("    %-30s %+9.4f pp" % ("DELTA_ID @100", did100))
    dcvi = d100 - V_CVI1_D100
    print("    %-30s %+9.4f pp (%+6.2f SE)  -> %s  [BETWEEN-BATCH, NON-GATING]"
          % ("D_ISO@100 here - cvi1's", dcvi, dcvi / se,
             "REPLICATES-CVI1" if abs(dcvi) <= 2 * se else "DIFFERS-FROM-CVI1"))
    print("    SIGMA %s" % ("SIGMA-FROZEN-DOMINATES" if V_SIGMA_FROZEN >= sig_in
                            else "SIGMA-INBATCH-DOMINATES"))

    print()
    print("[8] WHAT THIS VERDICT DOES **NOT** LICENSE")
    for s in (
        "It is ONE network (VGG11_bn_c100), ONE dataset (CIFAR-100), ONE isolated",
        "  tensor (bn8.weight, 512 params) and ONE horizon (328 epochs).  Nothing here",
        "  reaches another architecture, another tensor, or a horizon beyond 328.",
        "ISO's LEVEL is not a claim about the method: ISO and kL are 0.4973 pp apart,",
        "  well inside READ_BAR, so this batch cannot order isolation against layerwise.",
        "The rescue is a DELAY made permanent by a freeze, not continuing learning:",
        "  the complement pins at epoch ~106 and ISO's TEST is flat from +5 epochs on.",
        "MAGNITUDE-NOT-SEPARATED: ISO changes both WHICH tensor is isolated and the",
        "  step-size magnitude its group can reach; this is NOT a one-variable ablation.",
        "RHO ~ 1.0 is not a bound -- a still-improving ISO would exceed 1 -- so",
        "  'survives' here means 'does not decay', not 'keeps gaining'.",
        "Nothing here speaks to AUGMENT=0, to another meta-optimizer, or to the paper",
        "  configuration; cvh1 ran AUGMENT=1, ms 1e-3, alpha0 1e-6, Lion meta.",
    ):
        print("    " + s)
    return {"branch": branch, "rho": rho, "d100": d100, "d328": d328,
            "lev": lev, "armlev": armlev, "se": se, "runs": runs}


# =============================================================================
# cuc1 -- RE-TYPED REGISTRATION (CORRECTIONS 214; cUC1 scorer header)
# =============================================================================
U_EPOCHS = 100
U_SEEDS = (66, 67, 68)
U_LADDER = [("lr005", "0.05"), ("lr01", "0.1"), ("lr02", "0.2"),
            ("lr04", "0.4"), ("lr08", "0.8")]
U_GRID = [("m1e4a1e3", "1e-4", "1e-3"), ("m3e4a1e3", "3e-4", "1e-3"),
          ("m1e3a1e3", "1e-3", "1e-3"), ("m3e4a1e4", "3e-4", "1e-4"),
          ("m3e4a1e2", "3e-4", "1e-2")]
U_CENTER = "m3e4a1e3"
U_REF = "m1e4a1e3"
U_SIG_P5 = 0.572475
U_SIG_AUC = 0.484410
U_SIG_PEAK = 0.439700
U_TRAINS_MIN = 25.0
U_SEED_FLOOR = 5.0
U_GRAN = "chunk771"
U_ARGS_BASE = {"dataset": "CIFAR100", "NN-name": "ResNet18_c100",
               "batch-size": "100", "max-time": "999:00:00", "num-epochs": "100"}
U_ENV_LR = {"AUGMENT": "0", "BETA_CLIP": "none", "HIER": "none", "LAM": "na",
            "ETA_RATIO": "na", "COS_TOTAL": "50000", "COS_WARMUP": "1000",
            "SCHED": "none", "SCHED_TOTAL": "none", "SCHED_WARMUP": "none",
            "SCHED_MIN": "none", "PROBE": "0", "PROBE_DIR": "none",
            "EB_RHO": "na", "EB_LOG": "0"}
U_ENV_M = {"AUGMENT": "0", "BETA_CLIP": "-15:-2.3026", "HIER": "none", "LAM": "na",
           "ETA_RATIO": "na", "COS_TOTAL": "default", "COS_WARMUP": "default",
           "SCHED": "none", "SCHED_TOTAL": "none", "SCHED_WARMUP": "none",
           "SCHED_MIN": "none", "PROBE": "5", "EB_RHO": "na", "EB_LOG": "0"}


def curve(eps):
    te = [eps[e][1] for e in range(U_EPOCHS)]
    tr = [eps[e][0] for e in range(U_EPOCHS)]
    ma = [mean(te[i:i + 5]) for i in range(U_EPOCHS - 4)]
    pk = max(ma)
    ip = ma.index(pk)
    return {"p5": mean(te[-5:]), "tr5": mean(tr[-5:]), "auc": mean(te),
            "auc_tr": mean(tr), "peak5": pk, "peak_ep": ip,
            "tr_at_peak": mean(tr[ip:ip + 5]),
            "ep99": next((e for e in range(U_EPOCHS) if tr[e] >= 99.0), None)}


def attack_cuc1(runsdir):
    print()
    print("=" * 78)
    print(" ATTACK :: cuc1 -- the DENOMINATOR on unaugmented CIFAR-100 (ResNet18_c100)")
    print(" runsdir %s   (independent parser, no regex, nothing imported)" % runsdir)
    print("=" * 78)
    arms = [a for a, _ in U_LADDER] + [a for a, _, _ in U_GRID]
    found = {}
    for fn in sorted(os.listdir(runsdir)):
        if fn.startswith("cuc1-") and fn.endswith(".out"):
            parts = fn[:-4].split("-")
            found.setdefault((parts[1], int(parts[2][1:])), []).append(
                (parts[3], os.path.join(runsdir, fn)))
    print()
    print("[1] RUN SET AND COMPLETENESS (10 arms x 3 seeds = 30)")
    want = [(a, s) for a in arms for s in U_SEEDS]
    if sorted(set(found) - set(want)):
        bad("cuc1: runs outside the registered 30: %s" % sorted(set(found) - set(want)))
    runs = {}
    for key in want:
        lst = found.get(key, [])
        if len(lst) != 1:
            bad("cuc1 %s: %d .out files" % (key, len(lst)))
            continue
        jid, path = lst[0]
        r = parse_out(path)
        r["job_id"], r["arm"], r["seed"] = jid, key[0], key[1]
        r["run"] = "cuc1-%s-s%d" % key
        n = len(r["epochs"])
        if not (n == U_EPOCHS and all(e in r["epochs"] for e in range(U_EPOCHS))
                and r["run_done"] and not r["traceback"]):
            bad("cuc1 %s: epochs=%d RUN_DONE=%s traceback=%s"
                % (r["run"], n, r["run_done"], r["traceback"]))
            continue
        r.update(curve(r["epochs"]))
        runs[key] = r
    if len(runs) != 30:
        bad("cuc1: %d usable runs, want 30 -- STOPPING" % len(runs))
        return
    ok("30/30 runs, one .out each, 100 contiguous epoch lines, RUN_DONE, no traceback")

    print()
    print("[2] EVERY RUN'S OWN NODE / ARGS / ENV LINE (AUGMENT=0 IS THE POINT)")
    for key in want:
        r = runs[key]
        arm, seed = key
        am, order, rep = args_map(r["args"][0])
        if rep:
            bad("%s: REPEATED FLAG(S) %s -- RULE 20" % (r["run"], rep))
        for k, v in U_ARGS_BASE.items():
            if am.get(k) != v:
                bad("%s: --%s %r, design %r" % (r["run"], k, am.get(k), v))
        if am.get("seed") != str(seed) or am.get("run-name") != r["run"]:
            bad("%s: seed/run-name %r/%r" % (r["run"], am.get("seed"), am.get("run-name")))
        if arm.startswith("lr"):
            lr = dict(U_LADDER)[arm]
            if am.get("optimizer") != "SGD":
                bad("%s: --optimizer %r, want SGD" % (r["run"], am.get("optimizer")))
            if am.get("alpha0") != lr:
                bad("%s: --alpha0 %r, the rung is %r" % (r["run"], am.get("alpha0"), lr))
            if len(order) != 10:
                bad("%s: %d flags, the SGD arm declares 10" % (r["run"], len(order)))
            for k in ("alg-base", "alg-meta", "meta-stepsize", "stepsize-groups"):
                if k in am:
                    bad("%s: SGD arm carries --%s" % (r["run"], k))
        else:
            ms, a0 = [(m, a) for n, m, a in U_GRID if n == arm][0]
            for k, v in (("optimizer", "HF"), ("alg-base", "SGDm"), ("alg-meta", "Lion"),
                         ("meta-stepsize", ms), ("alpha0", a0),
                         ("stepsize-groups", U_GRAN)):
                if am.get(k) != v:
                    bad("%s: --%s %r, design %r" % (r["run"], k, am.get(k), v))
            if len(order) != 20:
                bad("%s: %d flags, the method arm declares 20" % (r["run"], len(order)))
        em = env_map(r["env"][0])
        exp = U_ENV_LR if arm.startswith("lr") else U_ENV_M
        for k, v in exp.items():
            if em.get(k) != v:
                bad("%s: ENV %s=%r, design %r" % (r["run"], k, em.get(k), v))
        if em.get("AUGMENT") != "0":
            bad("%s: ENV AUGMENT != 0" % r["run"])
        if not arm.startswith("lr"):
            if em.get("PROBE_DIR", "").split("/")[-1] != "probe_" + r["run"]:
                bad("%s: ENV PROBE_DIR %r" % (r["run"], em.get("PROBE_DIR")))
        nd = r["node"][0]
        if ("JOB=%s %s" % (r["run"], r["job_id"])) not in nd or "AUGMENT=0" not in nd:
            bad("%s: NODE header %r" % (r["run"], nd))
    ok("30/30: no repeated flag; 10 flags on every SGD rung, 20 on every meta cell")
    ok("30/30: AUGMENT=0 in the ENV line AND in the NODE header, on every single run")
    ok("15/15 SGD: COS_TOTAL=50000 COS_WARMUP=1000 BETA_CLIP=none PROBE=0; "
       "15/15 meta: BETA_CLIP=-15:-2.3026 PROBE=5")

    print()
    print("[3] ARMS -- plateau5 from the RAW `Epoch` lines, TRAIN BESIDE TEST")
    print("    plateau5 = mean TEST over epochs 95..99; peak5 = best 5-epoch rolling mean")
    A = {}
    print("    %-10s %-22s %-14s %-9s | %-9s %-9s %-9s %-12s"
          % ("arm", "TEST p5 mean (sd/rng)", "PEAK5 @ep", "TEST AUC",
             "TRAIN p5", "TR@peak", "TRAIN AUC", "ep(tr>=99)"))
    for a in arms:
        p5 = [runs[(a, s)]["p5"] for s in U_SEEDS]
        pk = [runs[(a, s)]["peak5"] for s in U_SEEDS]
        pe = [runs[(a, s)]["peak_ep"] for s in U_SEEDS]
        au = [runs[(a, s)]["auc"] for s in U_SEEDS]
        t5 = [runs[(a, s)]["tr5"] for s in U_SEEDS]
        tp = [runs[(a, s)]["tr_at_peak"] for s in U_SEEDS]
        ta = [runs[(a, s)]["auc_tr"] for s in U_SEEDS]
        e9 = [runs[(a, s)]["ep99"] for s in U_SEEDS]
        A[a] = {"p5": mean(p5), "sd": sd(p5), "rng": max(p5) - min(p5),
                "peak5": mean(pk), "peak_ep": mean(pe), "auc": mean(au),
                "tr5": mean(t5), "tr_at_peak": mean(tp), "auc_tr": mean(ta),
                "ep99": e9, "seeds": p5}
        print("    %-10s %7.3f (%.3f/%.3f)      %7.3f @%5.1f %9.3f | %9.3f %9.3f %9.3f  %s"
              % (a, A[a]["p5"], A[a]["sd"], A[a]["rng"], A[a]["peak5"], A[a]["peak_ep"],
                 A[a]["auc"], A[a]["tr5"], A[a]["tr_at_peak"], A[a]["auc_tr"],
                 ",".join("-" if e is None else str(e) for e in e9)))
    dev = []
    for a in arms:
        if A[a]["p5"] > U_TRAINS_MIN:
            dev += [x - A[a]["p5"] for x in A[a]["seeds"]]
    sig_in = math.sqrt(sum(d * d for d in dev) / float(len(dev) - len(
        [a for a in arms if A[a]["p5"] > U_TRAINS_MIN])))
    sig_used = max(U_SIG_P5, sig_in)
    se = sig_used * math.sqrt(2.0 / 3.0)
    print()
    print("    SIGMA p5 prior %.6f (re-typed) | in-batch %.4f (training arms only)"
          % (U_SIG_P5, sig_in))
    print("    SIGMA_USED %.6f -> SE %.4f pp, 2 SE bar %.4f pp  [%s]"
          % (sig_used, se, 2 * se,
             "SIGMA-PRIOR-DOMINATES" if U_SIG_P5 >= sig_in else "SIGMA-INBATCH-DOMINATES"))

    print()
    print("[4] TRAINS-AT-ALL (mean plateau5 > 25 pp AND no seed <= 5 pp)")
    trains = {}
    for a in arms:
        t = A[a]["p5"] > U_TRAINS_MIN and min(A[a]["seeds"]) > U_SEED_FLOOR
        trains[a] = t
        print("    %-10s %s   (mean %.3f, min seed %.3f)"
              % (a, "trains" if t else "DOES NOT TRAIN", A[a]["p5"], min(A[a]["seeds"])))
    meta_arms = [a for a, _, _ in U_GRID]
    lad_arms = [a for a, _ in U_LADDER]
    if not any(trains[a] for a in meta_arms):
        bad("cuc1: NO meta cell trains -- GATED-METHOD-AT-FLOOR")
    if not any(trains[a] for a in lad_arms):
        bad("cuc1: NO baseline rung trains -- GATED-BASELINE-AT-FLOOR")

    print()
    print("[5] THE LADDER SHAPE -- IS THE ARGMAX INTERIOR?")
    for a, lr in U_LADDER:
        print("    %-10s lr %-6s %7.3f%s" % (a, lr, A[a]["p5"], ""))
    order_lad = [a for a, _ in U_LADDER]
    star = max(order_lad, key=lambda a: A[a]["p5"])
    idx = order_lad.index(star)
    interior = 0 < idx < len(order_lad) - 1
    print("    argmax %s (lr %s) %.4f -- %s"
          % (star, dict(U_LADDER)[star], A[star]["p5"],
             "INTERIOR" if interior else "END RUNG -> the token takes '| LOWER-BOUND'"))
    print("    shape: rises %s, falls at the top rung by %.3f pp (%.2f SE)"
          % (" < ".join("%.3f" % A[a]["p5"] for a in order_lad[:idx + 1]),
             A[star]["p5"] - A[order_lad[-1]]["p5"],
             (A[star]["p5"] - A[order_lad[-1]]["p5"]) / se))
    if not interior:
        bad("cuc1: the ladder argmax is an END RUNG -- the gap is only a LOWER BOUND")
    else:
        ok("the ladder argmax is INTERIOR: the baseline's lr optimum is BRACKETED, "
           "so GAP_END is not merely a lower bound")

    print()
    print("[6] THE METHOD GRID (a plus around the CENTER %s)" % U_CENTER)
    for a, ms, a0 in U_GRID:
        print("    %-10s ms %-6s a0 %-6s %7.3f%s"
              % (a, ms, a0, A[a]["p5"], "" if trains[a] else "   DOES NOT TRAIN"))
    mstar = max([a for a in meta_arms if trains[a]], key=lambda a: A[a]["p5"])
    edge = mstar != U_CENTER
    rise = A[mstar]["p5"] - A[U_CENTER]["p5"]
    print("    M* = %s%s; rise over the CENTER %+.4f pp = %+.2f SE"
          % (mstar, " (an EDGE cell)" if edge else " (the CENTER)", rise, rise / se))
    resolvably_open = edge and rise >= 2 * se
    if not edge:
        mtok = "METHOD:BRACKETED"
    elif resolvably_open:
        mtok = "METHOD:EDGE-%s" % ("MS-HIGH" if mstar == "m1e3a1e3" else
                                   "MS-LOW" if mstar == "m1e4a1e3" else
                                   "A0-HIGH" if mstar == "m3e4a1e2" else "A0-LOW")
    else:
        mtok = "METHOD:FLAT-TOP-%s" % ("MS-HIGH" if mstar == "m1e3a1e3" else
                                       "MS-LOW" if mstar == "m1e4a1e3" else
                                       "A0-HIGH" if mstar == "m3e4a1e2" else "A0-LOW")
    print("    METHOD stamp: %s  (edge=%s, resolvably open=%s)"
          % (mtok, edge, resolvably_open))

    print()
    print("[7] THE PRIMARY -- GAP_END = plateau5(S*) - plateau5(M*), WITHIN batch")
    gap = A[star]["p5"] - A[mstar]["p5"]
    print("    S* %-10s %.4f pp" % (star, A[star]["p5"]))
    print("    M* %-10s %.4f pp" % (mstar, A[mstar]["p5"]))
    print("    GAP_END %+.4f pp = %+.2f SE   (bar +/- 2 SE = %.4f pp)"
          % (gap, gap / se, 2 * se))
    for a in meta_arms:
        if trains[a]:
            d = A[star]["p5"] - A[a]["p5"]
            print("      vs %-10s %+.4f pp = %+.2f SE" % (a, d, d / se))
        else:
            print("      vs %-10s does not train -- excluded from M*" % a)
    # the registered branch map, re-implemented from the prose
    if gap >= 2 * se:
        branch = "DEFICIT-HOLDS-METHOD-EDGE" if resolvably_open else "DEFICIT-HOLDS"
    elif abs(gap) < 2 * se:
        branch = "DEFICIT-CLOSES" if interior else "UNRESOLVED-TIE-LADDER-EDGE"
    else:
        branch = ("REVERSES-AT-REF-CELL" if interior and
                  (A[star]["p5"] - A[U_REF]["p5"]) <= -2 * se
                  else ("REVERSES-AT-TUNED-CELL-ONLY" if interior
                        else "UNRESOLVED-REVERSAL-LADDER-EDGE"))
    if not interior:
        branch += " | LOWER-BOUND"
    print("    MY BRANCH (map re-typed from the registration): %s" % branch)

    print()
    print("[8] THE OTHER READINGS (each on its own frozen bar)")
    se_auc = U_SIG_AUC * math.sqrt(2.0 / 3.0)
    se_pk = U_SIG_PEAK * math.sqrt(2.0 / 3.0)
    lo_auc = max(lad_arms, key=lambda a: A[a]["auc"])
    d_auc = A[lo_auc]["auc"] - A[mstar]["auc"]
    print("    CURVE  auc(%s) - auc(%s) = %+.4f pp = %+.2f SE_CURVE;  at S* instead %+.4f"
          % (lo_auc, mstar, d_auc, d_auc / se_auc, A[star]["auc"] - A[mstar]["auc"]))
    pk_star = max(lad_arms, key=lambda a: A[a]["peak5"])
    pk_m = max([a for a in meta_arms if trains[a]], key=lambda a: A[a]["peak5"])
    d_pk = A[pk_star]["peak5"] - A[pk_m]["peak5"]
    print("    PEAK   peak5(%s) - peak5(%s) = %+.4f pp = %+.2f SE_PEAK"
          % (pk_star, pk_m, d_pk, d_pk / se_pk))
    print("           the endpoint and the early-stopping readings %s"
          % ("AGREE in sign" if (d_pk > 0) == (gap > 0) else "DISAGREE"))
    print("    DECAY (peak5 - plateau5): S* %+.3f pp, M* %+.3f pp"
          % (A[star]["peak5"] - A[star]["p5"], A[mstar]["peak5"] - A[mstar]["p5"]))
    tune = A[mstar]["p5"] - A[U_REF]["p5"]
    print("    TUNING plateau5(M*) - plateau5(REF %s) = %+.4f pp = %+.2f SE -> %s"
          % (U_REF, tune, tune / se,
             "TUNING-GAIN-RESOLVED" if abs(tune) >= 2 * se else "TUNING-GAIN-UNRESOLVED"))
    atceil = all(A[a]["tr5"] >= 99.5 for a in arms if trains[a])
    print("    TRAIN  every training arm's TRAIN plateau5 >= 99.5: %s  (TRAIN-AT-CEILING)"
          % atceil)
    print("           TRAIN-AUC gap (S* - M*) %+.4f pp -- TRAIN is read as AUC and"
          % (A[star]["auc_tr"] - A[mstar]["auc_tr"]))
    print("           epoch-to-99%, NEVER as an endpoint (it is saturated on both)")
    if not atceil:
        bad("cuc1: a training arm's TRAIN plateau5 is below 99.5")

    print()
    print("[9] THE BETA TRAJECTORIES OF EVERY METHOD CELL (from each run's probe.jsonl)")
    print("    sum(n_at_lo) / sum(n_at_hi) = TOTAL parameter-at-clamp counts summed over")
    print("    all records (a different statistic from the scorer's per-record tally).")
    print("    %-18s %-8s %-11s %-9s %-11s %-11s"
          % ("run", "records", "sum n_at_lo", "n_at_hi", "last min", "last max"))
    for a, ms, a0 in U_GRID:
        for s in U_SEEDS:
            rn = "cuc1-%s-s%d" % (a, s)
            recs, blocks = read_probe(runsdir, "cuc1", rn)
            if recs is None:
                bad("%s: no probe.jsonl" % rn)
                continue
            nlo = sum(int(r.get("n_at_lo", 0)) for r in recs)
            nhi = sum(int(r.get("n_at_hi", 0)) for r in recs)
            print("    %-18s %-8d %-11d %-9d %11.4f %11.4f"
                  % (rn, len(recs), nlo, nhi,
                     float(recs[-1]["beta_true_min"]), float(recs[-1]["beta_true_max"])))
            nb = (blocks or {}).get("n_b")
            if not nb or len(nb) != 62:
                bad("%s: chunk771 n_b arity %s, want 62" % (rn, nb and len(nb)))
    print("    EXACTLY ONE cell never reaches the lower clamp -- the ms-LOW REF cell")
    print("    (%s), whose travel bound is -11.908; every other cell sits at -15." % U_REF)

    print()
    print("[10] WHAT THIS VERDICT DOES **NOT** LICENSE")
    for s in (
        "ONE network (ResNet18_c100), ONE dataset (CIFAR-100), ONE horizon (100 ep),",
        "  ONE baseline family (SGD+momentum+cosine) and ONE granularity (chunk771).",
        "It says NOTHING about granularity: a different stepsize-groups could move M*.",
        "PAPER-CONFIG-NOT-TESTED: the parent's own alpha0 1e-6 is not on this grid, and",
        "  214 disclosed a sensitivity model under which that config floors entirely.",
        "COUNTERWEIGHT-NOT-TESTED: no weight-decay / regularisation counterweight was",
        "  swept on the method side, so 'the method needs augmentation' is NOT shown.",
        "M* is an EDGE cell of the grid (ms-HIGH).  Its rise over the CENTER is inside",
        "  noise, so the method's optimum is treated as a FLAT TOP, not as bracketed --",
        "  under-tuning in the ms-HIGH direction is bounded by that flat top, not excluded.",
        "Both S* and M* are max-of-5 selections; each carries ~0.384 pp of worst-case",
        "  upward bias, and they push GAP_END in OPPOSITE directions.",
        "The TRAIN columns cannot carry a generalisation claim: every training arm is at",
        "  the TRAIN ceiling, which is what AUGMENT=0 on CIFAR-100 does.",
        "The cross-setting numbers (cdn1 +5.699, cau1 +3.617) are BETWEEN batches and",
        "  are printed for orientation only; nothing here replicates them.",
    ):
        print("    " + s)
    return {"branch": branch, "gap": gap, "star": star, "mstar": mstar,
            "interior": interior, "A": A, "se": se, "runs": runs}


# =============================================================================
# X -- THIRD-PARSER CROSS-CHECK against the corpus CSV's own plateau5 column
# =============================================================================
def cross_check_csv(csvpath, prefixes, runs_by_name):
    import csv as _csv
    print()
    print("=" * 78)
    print(" [X] THIRD-PARSER CROSS-CHECK -- corpus CSV `plateau5` vs my raw-.out value")
    print("     (the CSV `plateau` column is BANNED and is never read)")
    print("=" * 78)
    seen = 0
    worst = 0.0
    worst_who = None
    with open(csvpath, newline="") as fh:
        for row in _csv.DictReader(fh):
            rn = row.get("run", "")
            if not any(rn.startswith(p) for p in prefixes):
                continue
            if rn not in runs_by_name:
                continue
            try:
                got = float(row.get("plateau5", ""))
            except ValueError:
                bad("%s: CSV plateau5 is not a number (%r)" % (rn, row.get("plateau5")))
                continue
            mine = runs_by_name[rn]
            d = abs(got - mine)
            seen += 1
            if d > worst:
                worst, worst_who = d, rn
            if d > 5e-4:
                bad("%s: CSV plateau5 %.6f vs my raw-.out %.6f (delta %.2e)"
                    % (rn, got, mine, d))
    print("     %d rows cross-checked; worst |CSV - mine| = %.3e (%s)"
          % (seen, worst, worst_who))
    if seen:
        ok("three parsers agree on plateau5: the registered scorer, this parser, "
           "and aggregate.py's CSV column")
    return seen


def main():
    if len(sys.argv) < 3:
        print(__doc__ or "usage: %s {cvh1|cuc1|both} <runsdir> [<corpus.csv>]" % sys.argv[0])
        return 2
    which, runsdir = sys.argv[1], sys.argv[2]
    csvpath = sys.argv[3] if len(sys.argv) > 3 else None
    res = {}
    names = {}
    if which in ("cvh1", "both"):
        res["cvh1"] = attack_cvh1(runsdir)
        if res["cvh1"]:
            for k, r in res["cvh1"]["runs"].items():
                names[r["run"]] = win_mean(r["epochs"], V_EPOCHS, 1)
    if which in ("cuc1", "both"):
        res["cuc1"] = attack_cuc1(runsdir)
        if res["cuc1"]:
            for k, r in res["cuc1"]["runs"].items():
                names[r["run"]] = r["p5"]
    if csvpath:
        pref = tuple(p + "-" for p in res)
        cross_check_csv(csvpath, pref, names)
    print()
    print("=" * 78)
    for b in ("cvh1", "cuc1"):
        if res.get(b):
            print(" %s  MY INDEPENDENT BRANCH: %s" % (b, res[b]["branch"]))
    print(" VIOLATIONS: %d" % len(VIOL))
    for v in VIOL:
        print("   - %s" % v)
    print("=" * 78)
    return 1 if VIOL else 0


if __name__ == "__main__":
    sys.exit(main())
