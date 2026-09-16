#!/usr/bin/env python3
"""cgn3_attack_indep.py <runsdir> [<scorer.py>] [<score.log>] -- independent re-derivation of `cgn3` (CORRECTIONS 231).

Imports NOTHING from analysis/cGN3_gn_horizon_score.py or any other repo module and uses no regex.
Reads, by string splitting only:
  * every raw cgn3-<arm>-s<seed>-<jobid>.out in <runsdir>: ARGS / ENV / PROBE_TENSOR lines, every
    `Epoch` line, RUN_DONE, Traceback;
  * <runsdir>/cgn3/PARTITION-MANIFEST.txt (TENSOR lines give the 62 names) and PROVENANCE.txt;
  * each run's probe dir: probe_tensor.json, block_sizes.json, probe.jsonl (NO accuracy is read there).
plateau5@E = mean TEST over the run's own epochs E-5..E-1 (E = 100 and 430); TRAIN5 the same.
The CSV is NOT read.  Bars and branch order are RE-TYPED from the registration text (228.3-228.6).

Sections:
  [1] completeness, ARGS / ENV / PROBE_TENSOR per run, PROVENANCE
  [2] levels at 100 and 430, per seed and per arm, TEST beside TRAIN; sigma, SE, READ_BAR
  [3] the primary RHO, the branch, ONE's secondaries and every conditional stamp, re-typed
  [4] the complement beta trajectories: per arm / seed / group, median r over the last quarter,
      first r<=2 and first exact touch, and whether the step size STAYS pinned after first reaching r<=2
      (fraction of later records with r>2, max r after) -- plus beta at epochs 50..430
  [5] is accuracy flat after the pin?  arm-mean TEST/TRAIN at 5-epoch windows 100..430; per seed, the
      level 5 epochs after its OWN pin against the end, and the TEST OLS slope from its own pin+5 to 429;
      the same for ONE (its own complement pin) and k01; when kL passes ISO; [5b] when each arm's TEST
      settles (within 0.5 pp of its end, for good) and the complement r at that epoch
  [6] probe group-sum decomposition (z_agg == fsum z_tensor over the group; mom_pre likewise)
  [7] the scorer log's FINAL line carries the re-derived branch and every re-derived stamp
"""
import hashlib
import json
import math
import os
import sys

PREFIX = "cgn3"
NET = "ResNet18_gn_c100"
SEEDS = [81, 82, 83]
ARMS = ["k01", "kL", "ISO", "ONE"]
SPEC = {"k01": "scalar", "kL": "layerwise",
        "ISO": "sets:1-49,51-52,54-58,60-62/layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight",
        "ONE": "sets:1-49,51-62/layer4.0.bn2.weight"}
PT_TYPE = {"k01": "scalar", "kL": "layerwise", "ISO": "blockwise", "ONE": "blockwise"}
ENV_WANT = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
            "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 "
            "EB_RHO=na EB_LOG=0")
FIXED = {"optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99", "weight-decay-base": "0.1",
         "alg-meta": "Lion", "momentum-param-meta": "0.99", "Lion-beta2-meta": "0.9",
         "weight-decay-meta": "0", "dataset": "CIFAR100", "batch-size": "100", "meta-stepsize": "1e-3",
         "alpha0": "1e-6", "num-epochs": "430", "NN-name": NET, "max-time": "999:00:00", "gamma": "1"}
E, C = 430, 100
NREC = 2150
REC_PER_EPOCH = 5.0
# 228.4 / 228.6, re-typed
SIGMA_FROZEN = 0.953197
D100_MIN, REFUTE, SUPPORT, SURVIVE_FRAC, ATTEN_MILD, DEAD = 10.0, 5.0, 15.0, 0.80, 0.50, 5.0
ONE_SHARE_BAR = 0.10
ANCHOR = {"k01": 14.2640, "kL": 52.8610}
BAND = 3.812788
PIN_R, FREE_R, TAIL = 2.0, 10.0, 0.25
LO = -15.0
PIN_BOUND_R2 = 335.5036
W_POST = 64.2
CGN2_D_ISO_100 = 41.6207


def mean(v):
    return sum(v) / float(len(v))


def ols(xs, ys):
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def parse_out(path):
    d = {"args": [], "env": [], "pt": [], "test": {}, "train": {}, "done": False, "tb": False, "dup_ep": 0}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("ARGS:"):
                d["args"].append(line)
            elif line.startswith("ENV:"):
                d["env"].append(line)
            elif line.startswith("PROBE_TENSOR:"):
                d["pt"].append(line)
            elif line.startswith("Epoch ") and "Test Accuracy:" in line:
                head, rest = line.split(",", 1)
                ep = int(head.split()[1])
                tr = float(rest.split("Train Accuracy:")[1].split("%")[0])
                te = float(rest.split("Test Accuracy:")[1].split("%")[0])
                if ep in d["test"]:
                    d["dup_ep"] += 1
                d["test"][ep] = te
                d["train"][ep] = tr
            if "RUN_DONE" in line:
                d["done"] = True
            if "Traceback" in line:
                d["tb"] = True
    return d


def flags_of(line):
    toks = line[len("ARGS:"):].split()
    out, rep = {}, []
    i = 0
    while i < len(toks):
        if toks[i].startswith("--"):
            k = toks[i][2:]
            v = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else ""
            if k in out:
                rep.append(k)
            out[k] = v
            i += 2 if v != "" else 1
        else:
            i += 1
    return out, rep


def groups_of(spec, names):
    n = len(names)
    if spec == "scalar":
        return [list(range(n))]
    if spec == "layerwise":
        return [[i] for i in range(n)]
    body = spec[len("sets:"):]
    ranges, named = body.split("/")
    g0 = []
    for part in ranges.split(","):
        if "-" in part:
            a, b = part.split("-")
            g0 += list(range(int(a) - 1, int(b)))
        else:
            g0.append(int(part) - 1)
    g1 = [names.index(x) for x in named.split(",")]
    return [g0, g1]


def p5(acc, end):
    return mean([acc[e] for e in range(end - 5, end)])


def gstats(b):
    n = len(b)
    t = b[-int(n * TAIL):]
    r = [math.exp(v - LO) for v in t]
    f2 = next((i for i, v in enumerate(b) if v <= LO + math.log(PIN_R)), None)
    ft = next((i for i, v in enumerate(b) if v <= LO + 1e-4), None)
    after = b[f2:] if f2 is not None else []
    ra = [math.exp(v - LO) for v in after]
    return {"med_r": median(r), "max_r": max(r), "term": b[-1], "f2": f2, "ft": ft,
            "frac_gt2_after": (sum(1 for x in ra if x > PIN_R) / float(len(ra))) if ra else None,
            "max_r_after": max(ra) if ra else None}


def cls(m):
    return "PINNED" if m <= PIN_R else ("FREE" if m >= FREE_R else "NEITHER")


def arm_cls(cs):
    for k in ("PINNED", "FREE"):
        if sum(1 for c in cs if c == k) >= 2:
            return k
    return "NEITHER"


def three(d, lo, mid, hi, bar):
    return lo if d <= -bar else (hi if d >= bar else mid)


def branch_of(d100, de):
    rho = de / d100
    if d100 < D100_MIN:
        return "NO-RESCUE-AT-100", rho
    if de <= REFUTE:
        return "RESCUE-COLLAPSES", rho
    if rho >= SURVIVE_FRAC and de >= SUPPORT:
        return "RESCUE-SURVIVES", rho
    return "RESCUE-ATTENUATES", rho


def main():
    runsdir = sys.argv[1]
    scorer = sys.argv[2] if len(sys.argv) > 2 else None
    slog = sys.argv[3] if len(sys.argv) > 3 else None
    viol = []

    def V(ok, msg):
        if not ok:
            viol.append(msg)
            print("  !!! VIOLATION " + msg)

    names = []
    with open(os.path.join(runsdir, PREFIX, "PARTITION-MANIFEST.txt")) as f:
        for line in f:
            if line.startswith("TENSOR "):
                names.append(line.split()[2])
    V(len(names) == 62, "manifest has %d tensors" % len(names))
    prov = {}
    with open(os.path.join(runsdir, PREFIX, "PROVENANCE.txt")) as f:
        for line in f:
            p = line.rstrip("\n").split(" ", 1)
            if len(p) == 2:
                prov[p[0]] = p[1]

    print("[1] COMPLETENESS / ARGS / ENV / PROBE_TENSOR")
    files = sorted(x for x in os.listdir(runsdir) if x.startswith(PREFIX + "-") and x.endswith(".out"))
    runs = {}
    for fn in files:
        stem = fn[:-4]
        parts = stem.split("-")
        arm, seed = parts[1], int(parts[2][1:])
        V((arm, seed) not in runs, "duplicate .out for %s-s%d" % (arm, seed))
        d = parse_out(os.path.join(runsdir, fn))
        d["job"] = parts[3]
        runs[(arm, seed)] = d
    for arm in ARMS:
        for s in SEEDS:
            d = runs.get((arm, s))
            V(d is not None, "missing %s-s%d" % (arm, s))
            if d is None:
                continue
            V(sorted(d["test"]) == list(range(E)) and d["dup_ep"] == 0, "%s-s%d epochs" % (arm, s))
            V(d["done"] and not d["tb"], "%s-s%d done/tb" % (arm, s))
            V(len(d["args"]) == 1, "%s-s%d ARGS count" % (arm, s))
            fl, rep = flags_of(d["args"][0])
            V(not rep, "%s-s%d repeated %s" % (arm, s, rep))
            for k, v in FIXED.items():
                V(fl.get(k) == v, "%s-s%d flag %s=%s" % (arm, s, k, fl.get(k)))
            V(fl.get("stepsize-groups") == SPEC[arm], "%s-s%d spec" % (arm, s))
            V(fl.get("seed") == str(s), "%s-s%d seed" % (arm, s))
            V(fl.get("run-name") == "%s-%s-s%d" % (PREFIX, arm, s), "%s-s%d run-name" % (arm, s))
            env = [" ".join(t for t in e.split() if not t.startswith("PROBE_DIR=")) for e in d["env"]]
            V(env == [ENV_WANT], "%s-s%d ENV" % (arm, s))
            V(len(d["pt"]) == 1 and d["pt"][0].split()[3] == "type=" + PT_TYPE[arm] and d["pt"][0].split()[4] == "tensors=62",
              "%s-s%d PROBE_TENSOR" % (arm, s))
    V(len(runs) == 12, "run count %d" % len(runs))
    print("  %d runs, %d epoch lines each, RUN_DONE, 0 tracebacks, ARGS %d flags checked each, ENV == registered" %
          (len(runs), E, len(FIXED) + 3))
    print("  PROVENANCE MODE %s SCORER_SHA256 %s..." % (prov.get("MODE"), prov.get("SCORER_SHA256", "")[:16]))
    V(prov.get("MODE") == "submit", "PROVENANCE MODE")
    if scorer:
        V(prov.get("SCORER_SHA256") == sha(scorer), "PROVENANCE scorer sha != file")
        print("  scorer file sha256 %s == PROVENANCE: %s" % (sha(scorer)[:16], prov.get("SCORER_SHA256") == sha(scorer)))

    print("\n[2] LEVELS (plateau5 from the raw .out)")
    L = {}
    for arm in ARMS:
        for s in SEEDS:
            d = runs[(arm, s)]
            L[(arm, s)] = {"P100": p5(d["test"], C), "PE": p5(d["test"], E), "T100": p5(d["train"], C), "TE": p5(d["train"], E)}
    A = {}
    for arm in ARMS:
        A[arm] = {k: mean([L[(arm, s)][k] for s in SEEDS]) for k in ("P100", "PE", "T100", "TE")}
        A[arm]["R100"] = max(L[(arm, s)]["P100"] for s in SEEDS) - min(L[(arm, s)]["P100"] for s in SEEDS)
        A[arm]["RE"] = max(L[(arm, s)]["PE"] for s in SEEDS) - min(L[(arm, s)]["PE"] for s in SEEDS)
        print("  %-3s TEST @100 %.4f (%s)  @430 %.4f (%s) | TRAIN @100 %.4f  @430 %.4f | range %.3f / %.3f" %
              (arm, A[arm]["P100"], " ".join("%.3f" % L[(arm, s)]["P100"] for s in SEEDS), A[arm]["PE"],
               " ".join("%.3f" % L[(arm, s)]["PE"] for s in SEEDS), A[arm]["T100"], A[arm]["TE"], A[arm]["R100"], A[arm]["RE"]))
    ss, df = 0.0, 0
    for arm in ARMS:
        for k in ("P100", "PE"):
            v = [L[(arm, s)][k] for s in SEEDS]
            m = mean(v)
            ss += sum((x - m) ** 2 for x in v)
            df += 2
    sig_in = math.sqrt(ss / df)
    sig = max(SIGMA_FROZEN, sig_in)
    se = sig * math.sqrt(2.0 / 3.0)
    rb = 2 * se
    print("  sigma in-batch %.6f (df %d)  used %.6f  SE %.6f  READ_BAR %.6f" % (sig_in, df, sig, se, rb))

    print("\n[3] PRIMARY, BRANCH, SECONDARIES (re-typed from 228.6)")
    stamps = []
    for arm in ("k01", "kL"):
        V(ANCHOR[arm] - BAND <= A[arm]["P100"] <= ANCHOR[arm] + BAND, "anchor %s" % arm)
    for arm in ARMS:
        V(A[arm]["R100"] <= DEAD and A[arm]["RE"] <= DEAD, "diverged %s" % arm)
    d100 = A["ISO"]["P100"] - A["k01"]["P100"]
    de = A["ISO"]["PE"] - A["k01"]["PE"]
    br, rho = branch_of(d100, de)
    brt, rhot = branch_of(A["ISO"]["T100"] - A["k01"]["T100"], A["ISO"]["TE"] - A["k01"]["TE"])
    print("  D_ISO@100 %+.4f (%+.2f SE)  D_ISO@430 %+.4f (%+.2f SE)  RHO %.4f  -> %s" % (d100, d100 / se, de, de / se, rho, br))
    print("  TRAIN: D@100 %+.4f  D@430 %+.4f  RHO %.4f -> %s" % (A["ISO"]["T100"] - A["k01"]["T100"], A["ISO"]["TE"] - A["k01"]["TE"], rhot, brt))
    for s in SEEDS:
        a = L[("ISO", s)]["P100"] - L[("k01", s)]["P100"]
        b = L[("ISO", s)]["PE"] - L[("k01", s)]["PE"]
        o1 = L[("ONE", s)]["P100"] - L[("k01", s)]["P100"]
        o2 = L[("ONE", s)]["PE"] - L[("k01", s)]["PE"]
        print("    seed %d: RHO %.4f (%+.3f -> %+.3f)   RHO_ONE %.4f (%+.3f -> %+.3f)" % (s, b / a, a, b, o2 / o1, o1, o2))
    print("  margins: D_ISO@430 - SUPPORT %+.3f pp; RHO - 0.80 %+.4f; D_ISO@100 - D100_MIN %+.3f" % (de - SUPPORT, rho - SURVIVE_FRAC, d100 - D100_MIN))
    stamps.append("RHO:%.4f" % rho)
    stamps.append("TRAIN-AGREES" if brt == br else "TRAIN-DISAGREES")
    do100 = A["ONE"]["P100"] - A["k01"]["P100"]
    doe = A["ONE"]["PE"] - A["k01"]["PE"]
    bo, rho1 = branch_of(do100, doe)
    g100 = A["kL"]["P100"] - A["k01"]["P100"]
    ge = A["kL"]["PE"] - A["k01"]["PE"]
    print("  ONE: D@100 %+.4f D@430 %+.4f RHO_ONE %.4f (%s); share %.4f -> %.4f; ISO-ONE %+.4f -> %+.4f" %
          (do100, doe, rho1, bo, do100 / g100, doe / ge, A["ISO"]["P100"] - A["ONE"]["P100"], A["ISO"]["PE"] - A["ONE"]["PE"]))
    print("  ONE TRAIN: D@100 %+.4f D@430 %+.4f" % (A["ONE"]["T100"] - A["k01"]["T100"], A["ONE"]["TE"] - A["k01"]["TE"]))
    print("  ISO - kL: @100 %+.4f  @430 %+.4f;  ONE - kL @430 %+.4f;  k01 drift %+.4f;  ISO TRAIN drift %+.4f;  D100 - cgn2 %+.4f" %
          (A["ISO"]["P100"] - A["kL"]["P100"], A["ISO"]["PE"] - A["kL"]["PE"], A["ONE"]["PE"] - A["kL"]["PE"],
           A["k01"]["PE"] - A["k01"]["P100"], A["ISO"]["TE"] - A["ISO"]["T100"], d100 - CGN2_D_ISO_100))

    print("\n[4] BETA TRAJECTORIES (probe records; r = exp(beta + 15))")
    P, G = {}, {}
    for arm in ARMS:
        for s in SEEDS:
            pd = os.path.join(runsdir, PREFIX, "probe_%s-%s-s%d" % (PREFIX, arm, s))
            recs = [json.loads(x) for x in open(os.path.join(pd, "probe.jsonl")) if x.strip()]
            V(len(recs) == NREC, "%s-s%d records %d" % (arm, s, len(recs)))
            V(all(recs[i]["step"] == 100 * i for i in range(len(recs))), "%s-s%d record steps not 100*i" % (arm, s))
            pt = json.load(open(os.path.join(pd, "probe_tensor.json")))
            V(pt["stepsize_type"] == PT_TYPE[arm] and pt["n_tensors"] == 62, "%s-s%d probe_tensor.json" % (arm, s))
            P[(arm, s)] = recs
    for arm, k in (("ISO", 0), ("ISO", 1), ("ONE", 0), ("ONE", 1), ("k01", 0)):
        cs = []
        for s in SEEDS:
            b = [r["beta"][k] for r in P[(arm, s)]]
            g = gstats(b)
            G[(arm, k, s)] = g
            cs.append(cls(g["med_r"]))
            at = " ".join("%.2f" % b[int(ep * REC_PER_EPOCH) - 1] for ep in (50, 100, 150, 200, 250, 300, 350, 430))
            print("  %-3s b[%d] s%d med_r %.4f max_r %.3f term %.3f | first r<=2 ep %s  touch ep %s | after first r<=2: frac r>2 %s, max r %s -> %s" %
                  (arm, k, s, g["med_r"], g["max_r"], g["term"],
                   "%.1f" % (g["f2"] / REC_PER_EPOCH) if g["f2"] is not None else "never",
                   "%.1f" % (g["ft"] / REC_PER_EPOCH) if g["ft"] is not None else "never",
                   "%.4f" % g["frac_gt2_after"] if g["frac_gt2_after"] is not None else "-",
                   "%.3f" % g["max_r_after"] if g["max_r_after"] is not None else "-", cls(g["med_r"])))
            print("        beta @ ep 50/100/150/200/250/300/350/430: %s" % at)
        print("  %s b[%d] ARM CLASS %s" % (arm, k, arm_cls(cs)))
        if (arm, k) == ("ISO", 0):
            stamps.append("COMPLEMENT-" + arm_cls(cs))
            iso_cls = arm_cls(cs)
        if (arm, k) == ("ONE", 0):
            one_cls = arm_cls(cs)
        if (arm, k) == ("k01", 0):
            k01_cls = arm_cls(cs)
    klfree = []
    for s in SEEDS:
        nb = len(P[("kL", s)][0]["beta"])
        klfree.append(sum(1 for k in range(nb) if cls(gstats([r["beta"][k] for r in P[("kL", s)]])["med_r"]) == "FREE"))
    print("  kL groups FREE per seed: %s of 62" % klfree)

    print("\n[5] IS ACCURACY FLAT AFTER THE PIN?")
    ends = (100, 150, 200, 225, 250, 275, 300, 350, 400, 430)
    for arm in ARMS:
        print("  %-3s TEST  %s" % (arm, "  ".join("%d:%.2f" % (e, mean([p5(runs[(arm, s)]["test"], e) for s in SEEDS])) for e in ends)))
        print("  %-3s TRAIN %s" % (arm, "  ".join("%d:%.2f" % (e, mean([p5(runs[(arm, s)]["train"], e) for s in SEEDS])) for e in ends)))
    if iso_cls == "PINNED":
        pins = [G[("ISO", 0, s)]["f2"] / REC_PER_EPOCH for s in SEEDS]
        latest = max(pins)
        e0 = int(math.ceil(latest)) + 5
        post = mean([mean([runs[("ISO", s)]["test"][e] for e in range(e0, e0 + 5)]) for s in SEEDS])
        dpp = A["ISO"]["PE"] - post
        stamps += ["ISO-PIN-EPOCH:%.1f" % mean(pins), "POST-PIN-WINDOW:%.1f" % (E - latest),
                   "WINDOW-MEETS-W100" if E - latest >= W_POST else "WINDOW-SHORT",
                   "PIN-INSIDE-BOUND" if latest <= PIN_BOUND_R2 else "PIN-AFTER-BOUND",
                   three(dpp, "DECAYS-AFTER-PIN", "FROZEN-AT-PIN", "IMPROVES-AFTER-PIN", rb)]
        print("  scorer's rule: latest pin %.1f, window %.1f, TEST ep %d..%d %.4f vs end %.4f, delta %+.4f (%+.2f SE)" %
              (latest, E - latest, e0, e0 + 4, post, A["ISO"]["PE"], dpp, dpp / se))
    else:
        stamps += ["UNRESOLVED-NOT-PINNED", "POST-PIN-UNMEASURED"]
    for arm, k in (("ISO", 0), ("ONE", 0), ("k01", 0)):
        for s in SEEDS:
            g = G[(arm, k, s)]
            if g["f2"] is None:
                print("  %s s%d: never r<=2" % (arm, s))
                continue
            pe = g["f2"] / REC_PER_EPOCH
            a0 = int(math.ceil(pe)) + 5
            d = runs[(arm, s)]
            lvl_pin = mean([d["test"][e] for e in range(a0, a0 + 5)])
            lvl_pin_tr = mean([d["train"][e] for e in range(a0, a0 + 5)])
            xs = list(range(a0, E))
            slope = ols(xs, [d["test"][e] for e in xs]) * 100.0
            slope_tr = ols(xs, [d["train"][e] for e in xs]) * 100.0
            pre = p5(d["test"], C)
            print("  %-3s s%d: own pin %.1f | TEST @pin+5 %.3f -> end %.3f (%+.3f; %+.3f pp/100ep OLS over %d ep) | TRAIN %.3f -> %.3f (%+.3f pp/100ep) | TEST gain 100->pin %+.3f" %
                  (arm, s, pe, lvl_pin, L[(arm, s)]["PE"], L[(arm, s)]["PE"] - lvl_pin, slope, len(xs),
                   lvl_pin_tr, L[(arm, s)]["TE"], slope_tr, lvl_pin - pre))
    cross = None
    for e in range(C, E + 1):
        if all(mean([p5(runs[("kL", s)]["test"], x) for s in SEEDS]) > mean([p5(runs[("ISO", s)]["test"], x) for s in SEEDS])
               for x in range(e, E + 1)):
            cross = e
            break
    print("  kL's 5-epoch TEST mean stays above ISO's from window-end epoch %s on" % cross)
    # [5b] (added after the first full pass) when does each arm's TEST level stop moving, and where is its
    # complement step size then?  "settled" = the first window-end epoch from which the arm-mean 5-epoch
    # TEST stays within 0.5 pp of its epoch-430 value; r read at that epoch's last record, per seed.
    for arm in ("ISO", "ONE", "kL"):
        endv = mean([p5(runs[(arm, s)]["test"], E) for s in SEEDS])
        settle = None
        for e in range(C, E + 1):
            if all(abs(mean([p5(runs[(arm, s)]["test"], x) for s in SEEDS]) - endv) <= 0.5 for x in range(e, E + 1)):
                settle = e
                break
        if arm == "kL":
            print("  kL TEST settles (within 0.5 pp of end, for good) at window-end epoch %s" % settle)
            continue
        rs = [math.exp(P[(arm, s)][int(settle * REC_PER_EPOCH) - 1]["beta"][0] - LO) for s in SEEDS]
        pins = [G[(arm, 0, s)]["f2"] / REC_PER_EPOCH for s in SEEDS]
        print("  %s TEST settles (within 0.5 pp of end, for good) at window-end epoch %s; complement r there %s; complement first r<=2 at %s" %
              (arm, settle, " / ".join("%.2f" % x for x in rs), " / ".join("%.1f" % x for x in pins)))

    stamps.append("RHO-ONE:%.4f" % rho1)
    stamps.append({"NO-RESCUE-AT-100": "ONE-NO-RESCUE-AT-100", "RESCUE-COLLAPSES": "ONE-COLLAPSES",
                   "RESCUE-SURVIVES": "ONE-SURVIVES", "RESCUE-ATTENUATES": "ONE-ATTENUATES"}[bo])
    stamps.append(three(doe / ge - do100 / g100, "ONE-SHARE-FALLS", "ONE-SHARE-HOLDS", "ONE-SHARE-RISES", ONE_SHARE_BAR))
    stamps.append("ONE-COMPLEMENT-" + one_cls)
    stamps.append(three((A["ISO"]["PE"] - A["ONE"]["PE"]) - (A["ISO"]["P100"] - A["ONE"]["P100"]), "HEAD3-NARROWS", "HEAD3-HOLDS", "HEAD3-WIDENS", rb))
    stamps.append(three(A["ISO"]["PE"] - A["kL"]["PE"], "CEIL-BELOW", "CEIL-TRACKS", "CEIL-ABOVE", rb))
    stamps.append("KL-STILL-TRAINING" if sum(1 for n in klfree if n >= 1) >= 2 else "KL-FROZEN")
    stamps.append("K01-" + k01_cls)
    stamps.append("FLOOR-HOLDS" if abs(A["k01"]["PE"] - A["k01"]["P100"]) < rb else "FLOOR-DRIFTS")
    stamps.append("ISO-TRAIN-FROZEN" if abs(A["ISO"]["TE"] - A["ISO"]["T100"]) < rb else "ISO-TRAIN-MOVES")
    stamps.append("SIGMA-FROZEN-DOMINATES" if sig == SIGMA_FROZEN else "SIGMA-INBATCH-DOMINATES")
    stamps.append("REPLICATES-CGN2" if abs(d100 - CGN2_D_ISO_100) < rb else "DIFFERS-FROM-CGN2")

    print("\n[6] GROUP-SUM DECOMPOSITION")
    worst, bad, n = 0.0, 0, 0
    for arm in ARMS:
        for s in SEEDS:
            grp = groups_of(SPEC[arm], names)
            for r in P[(arm, s)]:
                for gi, idx in enumerate(grp):
                    for key, agg in (("z_tensor", "z_agg"), ("m_tensor", "mom_pre")):
                        a = r[agg]
                        want = (a[0] if isinstance(a[0], list) else a)[gi]
                        terms = [r[key][i] for i in idx]
                        got = math.fsum(terms)
                        scale = max(abs(want), math.fsum(abs(x) for x in terms))
                        err = abs(got - want) / scale if scale > 1e-30 else abs(got - want)
                        n += 1
                        worst = max(worst, err)
                        if err > 1e-4:
                            bad += 1
    print("  %d group-sums checked, %d with error > 1e-4 relative to max(|agg|, sum|terms|), worst %.3e" % (n, bad, worst))
    V(bad == 0, "decomposition failures %d" % bad)

    print("\n[7] BRANCH + STAMPS vs the scorer log")
    print("  re-derived: %s | %s" % (br, " | ".join(stamps)))
    if slog:
        fin = [x for x in open(slog) if x.startswith("FINAL:")]
        V(len(fin) == 1, "scorer log FINAL lines %d" % len(fin))
        toks = [t.strip() for t in fin[0][len("FINAL:"):].split("|")]
        V(toks[0] == br, "branch %s vs scorer %s" % (br, toks[0]))
        for t in stamps:
            V(t in toks, "stamp %s not in scorer FINAL" % t)
        print("  scorer FINAL branch %s; %d re-derived stamps, all present: %s" % (toks[0], len(stamps), all(t in toks for t in stamps)))

    print("\nATTACK VERDICT: %d violations" % len(viol))
    return 1 if viol else 0


if __name__ == "__main__":
    sys.exit(main())
