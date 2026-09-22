#!/usr/bin/env python3
# =============================================================================
# caw2_attack_indep.py -- AN INDEPENDENT RE-DERIVATION OF EVERY LINE `analysis/cAW2_confound_score.py` (bdeb1321...,
#   CORRECTIONS 290) PRINTS ON THE LANDED caw2 BATCH, PLUS AN ATTACK SECTION THE SCORER DOES NOT PRINT.
#
#   STDLIB ONLY.  NO REPO IMPORT: nothing from caw2_design.py, cwd_common.py, cwd_design.py, corpus_exclusions.py or
#   the scorer is imported.  Every arm, flag, witness, bar and registered sha is RE-TYPED below from CORRECTIONS 290 /
#   the registered files as committed; the raw .out and probe.jsonl files are parsed by this file's own code.  The
#   scorer file and the corpus CSV / exclusion TSV are only READ AS BYTES / TEXT (a sha, a row count), never executed.
#
#   python3 analysis/caw2_attack_indep.py <runsdir> [--against <scorer stdout>] [--repo <repo root>]
#
#   Output is PATH-FREE (the runsdir is printed as <runsdir>) so it is byte-identical on every host that holds the
#   same files.  With --against, the scorer's stdout (its runsdir echo normalised to <runsdir>) is compared LINE BY
#   LINE with this file's reconstruction, and every line is accounted for.
#   Exit 0 iff the reconstruction is identical (when --against is given) and no attack check FAILs.
# =============================================================================
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys

# ---- RE-TYPED DESIGN (CORRECTIONS 290.5; caw2_design.py a219661f... as committed) -----------------------------------
PREFIX = "caw2"
SEEDS = (160, 161, 162)
EPOCHS = 100
NTENS = 62
N_RECORDS = 500
PROBE = 100
TOTPAR = 11220132
ARMS = ("K01", "MS", "ML", "LS", "LL", "AS", "AL", "XS", "XL")
TAB = {  # arm: (base, meta, grain, wd token, wd float, cell, pair, role)
    "K01": ("SGDm", "Lion", "scalar", "0.1", 0.1, "K", "SL", "SGDm(0.99)+Lion at wd 0.1: the mechanism cell, POSITIVE CONTROL"),
    "MS": ("SGDm", "Adam", "scalar", "0.1", 0.1, "M", "SA", "SGDm(0.99)+Adam at wd 0.1 (META swap alone), scalar"),
    "ML": ("SGDm", "Adam", "layerwise", "0.1", 0.1, "M", "SA", "SGDm(0.99)+Adam at wd 0.1, layerwise -- M's in-batch reference"),
    "LS": ("AdamW", "Lion", "scalar", "0.1", 0.1, "L", "WL", "AdamW+Lion at wd 0.1 (BASE swap alone), scalar"),
    "LL": ("AdamW", "Lion", "layerwise", "0.1", 0.1, "L", "WL", "AdamW+Lion at wd 0.1, layerwise -- L's in-batch reference"),
    "AS": ("AdamW", "Adam", "scalar", "0.1", 0.1, "A", "AW", "AdamW+Adam at wd 0.1 (the STANDARD RECIPE), scalar"),
    "AL": ("AdamW", "Adam", "layerwise", "0.1", 0.1, "A", "AW", "AdamW+Adam at wd 0.1, layerwise -- A's in-batch reference"),
    "XS": ("AdamW", "Adam", "scalar", "1.0", 1.0, "X", "AW", "AdamW+Adam at wd 1.0 (the DOSE arm, 10x A's decay), scalar"),
    "XL": ("AdamW", "Adam", "layerwise", "1.0", 1.0, "X", "AW", "AdamW+Adam at wd 1.0, layerwise -- X's in-batch reference"),
}
CELLS = ("M", "L", "A", "X")
SCAL = {"M": "MS", "L": "LS", "A": "AS", "X": "XS"}
LAYR = {"M": "ML", "L": "LL", "A": "AL", "X": "XL"}
SCALAR_ARMS = ("K01", "MS", "LS", "AS", "XS")
CARRIERS0 = (49, 52, 58)
CARRIER_NAMES = ("layer4.0.bn2.weight", "layer4.0.shortcut.1.weight", "layer4.1.bn2.weight")
BASEFL = {"SGDm": [("alg-base", "SGDm"), ("momentum-param-base", "0.99"), ("weight-decay-base", None)],
          "AdamW": [("alg-base", "AdamW"), ("normalizer-param-base", "0.999"), ("momentum-param-base", "0.9"),
                    ("weight-decay-base", None)]}
METAFL = {"Lion": [("alg-meta", "Lion"), ("momentum-param-meta", "0.99"), ("Lion-beta2-meta", "0.9"),
                   ("weight-decay-meta", "0")],
          "Adam": [("alg-meta", "Adam"), ("normalizer-param-meta", "0.999"), ("momentum-param-meta", "0.9"),
                   ("weight-decay-meta", "0")]}
COMMONFL = [("dataset", "CIFAR100"), ("NN-name", "ResNet18_c100"), ("batch-size", "100"), ("max-time", "999:00:00"),
            ("gamma", "1"), ("meta-stepsize", "1e-3"), ("alpha0", "1e-6"), ("num-epochs", "100")]
PTTAIL = {"Adam": "meta_alg=Adam momentum_param=0.9 Lion_beta2=0.9", "Lion": "meta_alg=Lion momentum_param=0.99 Lion_beta2=0.9"}
ENV_EXPECTED = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default COS_WARMUP=default "
                "SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 EB_RHO=na EB_LOG=0")
OFF = ("VOTE_W: off", "BETA_HOLD: off", "GROUP_HOLD: off", "REST_HOLD: off", "DECAY_MASK: off")
PROV_WANT = [("MODE", "submit", "MODE submit"),
             ("BUILD_NETWORK_SHA256", "c79988833c4399a06cc84d15c0830f54d86f59feeec345f47a2287c1cf2ba77d", "build_network.py == cvt8's"),
             ("HF_SHA256", "94aedc33046afb12782ffb7f2eb729bafc7fcba5f42227fec373fa67b6f58ebf", "HF.py == cwd1's tree, UNCHANGED"),
             ("RUNNER_SHA256", "34a8c90ee3a0bee00aa2d0b4a3ad5d16d461fbf5a1f0cda5797e44e44c158371", "runner == run_cifar_cwd1.sh, UNCHANGED"),
             ("SCORER_SHA256", "bdeb132189a2e907dfe9d9908e6fa2a0cc4f4fb0b1b6d9d0d6ecd8965675bfab", "SCORER == this file"),
             ("DESIGN_SHA256", "a219661f1e3ef140e2fa47fdf4aeae0d1aff1962748cf2489f6efffe2348b378", "DESIGN == the caw2_design.py this scorer loaded"),
             ("CWD_DESIGN_SHA256", "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195", "the reused cwd_design.py this scorer loaded"),
             ("COMMON_SHA256", "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52", "COMMON == the cwd_common.py this scorer loaded")]
REG_FILES = {"analysis/cAW2_confound_score.py": "bdeb132189a2e907dfe9d9908e6fa2a0cc4f4fb0b1b6d9d0d6ecd8965675bfab",
             "analysis/caw2_design.py": "a219661f1e3ef140e2fa47fdf4aeae0d1aff1962748cf2489f6efffe2348b378",
             "analysis/cwd_design.py": "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195",
             "analysis/cwd_common.py": "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"}
# ---- RE-TYPED BARS (CORRECTIONS 290.7) -----------------------------------------------------------------------------
SIGMA_PRIOR = 0.6415731883291765
R50, GAP_BAR, REF_MIN, DIVERGED_BAR, K01_MAX, DOM_BAR, TOL = 0.50, 10.0, 55.0, 5.0, 30.0, 0.50, 1e-3
FLOOR_MIN, CEIL_MAX = 15.0, 90.0
DOSE_PRE_END, LATE_FROM, DOSE_LO, DOSE_HI = 8500, 40000, 0.5, 2.0
FIT_SEED, SCORE_SEEDS = 160, (161, 162)
BETA_LO, BETA_HI = -15.0, -2.3026
K01_LANDED = (("cmo1 k01", 22.7887), ("cwd3 k01", 22.9853), ("cwd4 k01WD", 22.8213), ("cwd5 k01W1", 23.2240))

EPOCH_LINE = re.compile(r"^Epoch (\d+), Train Accuracy: ([0-9.]+|nan) %, Test Accuracy: ([0-9.]+|nan) %$")
OUT_NAME = re.compile(r"^caw2-(K01|MS|ML|LS|LL|AS|AL|XS|XL)-s(\d+)-(\d+)\.out$")

OUT = []


def P(s=""):
    OUT.append(s)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def avg(v):
    return sum(v) / len(v)


def sdev(v):
    m = avg(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def sgn(x):
    return (x > 0) - (x < 0)


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def expected_args(arm, seed):
    b, m = TAB[arm][0], TAB[arm][1]
    out = [("optimizer", "HF")]
    out += [(f, TAB[arm][3] if f == "weight-decay-base" else v) for f, v in BASEFL[b]]
    out += METAFL[m] + COMMONFL
    out += [("stepsize-groups", TAB[arm][2]), ("seed", str(seed)), ("run-name", "caw2-%s-s%d" % (arm, seed))]
    return out


def pt_prefix(arm):
    return "PROBE_TENSOR: on every=100 type=%s tensors=62 %s" % (TAB[arm][2], PTTAIL[TAB[arm][1]])


# ---- RAW PARSERS (this file's own) ---------------------------------------------------------------------------------
def parse_out(path):
    lines = open(path, errors="replace").read().splitlines()
    eps, dup, loose = {}, 0, 0
    for ln in lines:
        mm = EPOCH_LINE.match(ln)
        if mm:
            e = int(mm.group(1))
            if e in eps:
                dup += 1
            eps[e] = (float(mm.group(2)), float(mm.group(3)))
        elif ln.startswith("Epoch"):
            loose += 1
    r = {"eps": eps, "dup": dup, "loose": loose, "lines": lines,
         "run_done": any(ln.strip() == "RUN_DONE" for ln in lines),
         "tb": any("Traceback (most recent call last)" in ln for ln in lines),
         "args": [ln for ln in lines if ln.startswith("ARGS:")],
         "env": [ln for ln in lines if ln.startswith("ENV:")],
         "pt": [ln for ln in lines if ln.startswith("PROBE_TENSOR:")]}
    tail = [eps.get(e) for e in range(EPOCHS - 5, EPOCHS)]
    ok = all(t is not None and math.isfinite(t[0]) and math.isfinite(t[1]) for t in tail)
    r["p5"] = avg([t[1] for t in tail]) if ok else None
    r["t5"] = avg([t[0] for t in tail]) if ok else None
    r["complete"] = sorted(eps) == list(range(EPOCHS)) and r["run_done"] and not r["tb"] and ok
    gpu = None
    for ln in lines:
        s = ln.strip()
        if s.endswith(" MiB") and ", " in s and not s.startswith(("ARGS:", "ENV:", "Epoch")):
            gpu = s.split(",")[0].strip()
            break
    r["gpu"] = gpu
    return r


def tokens_args(line):
    d = {}
    cur = None
    for t in line[len("ARGS:"):].split():
        if t.startswith("--"):
            cur = t[2:]
            d.setdefault(cur, []).append(None)
        elif cur is not None:
            d[cur][-1] = t if d[cur][-1] is None else d[cur][-1] + " " + t
    return d


def load_recs(runsdir, arm, seed):
    p = os.path.join(runsdir, PREFIX, "probe_caw2-%s-s%d" % (arm, seed), "probe.jsonl")
    out = []
    for ln in open(p, errors="replace"):
        ln = ln.strip()
        if ln:
            try:
                r = json.loads(ln)
            except ValueError:
                r = None
            out.append(r if isinstance(r, dict) else None)
    return out


def vote(r, meta):
    try:
        mp, b2 = float(r["pt_mp"]), float(r["pt_b2"])
        z, m = r["z_tensor"], r["m_tensor"]
        mom, za = float(r["mom_pre"][0][0]), float(r["z_agg"][0][0])
    except (KeyError, TypeError, IndexError, ValueError):
        return None
    if len(z) != NTENS or len(m) != NTENS or not all(fin(x) for x in z) or not all(fin(x) for x in m):
        return None
    if meta == "Adam":
        L = [mp / (1.0 - mp) * a + b for a, b in zip(m, z)]
        app = mp * mom + za
    else:
        L = [b2 * a + (1.0 - b2) * b for a, b in zip(m, z)]
        app = b2 * mom + (1.0 - b2) * za
    if not (fin(app) and all(fin(x) for x in L)):
        return None
    return L, app


def shrink(r, wd):
    b = r.get("beta") if isinstance(r, dict) else None
    if not isinstance(b, list) or not b or not all(fin(x) for x in b):
        return None
    return median([math.exp(x) for x in b]) * wd


def dose(recs, wd):
    a, pre, late = [], [], []
    for r in recs:
        s = shrink(r, wd)
        if s is None or not isinstance(r.get("step"), int):
            continue
        a.append(s)
        if r["step"] <= DOSE_PRE_END:
            pre.append(s)
        if r["step"] >= LATE_FROM:
            late.append(s)
    nan = float("nan")
    return (max(a) if a else nan, max(pre) if pre else nan, median(late) if late else nan)


def state(Lv, kv):
    L, k = avg(Lv), avg(kv)
    if L < REF_MIN or max(Lv) - min(Lv) > DIVERGED_BAR:
        return "UNREADABLE"
    n = sum(1 for x in kv if x <= R50 * L)
    if 1 <= n <= len(kv) - 1:
        return "SPLIT"
    if k <= R50 * L:
        return "COLLAPSE"
    if L - k < GAP_BAR:
        return "NOGAP"
    return "PARTIAL"


def branch_of(st, k01, rx):
    if k01 > K01_MAX:
        return "CONTROL-NOT-COLLAPSED"
    if st["A"] == "UNREADABLE":
        return "UNRESOLVED-REFERENCE-A"
    if st["A"] == "COLLAPSE":
        return "HAZARD-AT-STANDARD-RECIPE"
    if st["A"] in ("SPLIT", "PARTIAL"):
        return "PARTIAL-AT-STANDARD-RECIPE"
    if st["X"] in ("COLLAPSE", "SPLIT"):
        return "NO-COLLAPSE-AT-STANDARD-DOSE-ONLY"
    if st["X"] == "NOGAP" and fin(rx):
        return "NO-COLLAPSE-PAIRING-IMMUNE-AT-CONTROL-DOSE" if rx >= DOSE_LO else "NO-COLLAPSE-CONTROL-DOSE-NOT-REACHED"
    return "NO-COLLAPSE-DOSE-ARM-UNRESOLVED"


def pcu(s):
    return "P" if s == "NOGAP" else ("C" if s in ("COLLAPSE", "SPLIT") else "U")


def attr_of(st):
    return {("C", "P"): "ATTR-BASE-PROTECTS", ("P", "C"): "ATTR-META-PROTECTS", ("P", "P"): "ATTR-EITHER-PROTECTS",
            ("C", "C"): "ATTR-NEITHER-ALONE"}.get((pcu(st["M"]), pcu(st["L"])), "ATTR-UNRESOLVED")


# ---- the registered static text, re-typed (only what the reached branch prints) ------------------------------------
BOUND_NOTE = ("Every COLLAPSE and NOGAP state is a BOUND, never a point effect (164.6; FLOOR-READINGS-ARE-BOUNDS): a "
              "collapsed arm's gap is a LOCATION, and NOGAP means 'below the 10 pp bar', never 'the grains are equal'.")
NOT_ELSE = ("Not licensed: any other network, dataset, meta step size, alpha0, horizon, grouping, base optimiser "
            "(a Lion base was NOT run), any decay other than the two run (1e-2 was NOT run), or anything about "
            "ALPHA-INDEPENDENT decay; nothing about the parent paper's own cells; wd 1.0 is a DOSE arm, not a "
            "standard harness value.")
LIC = {"NO-COLLAPSE-CONTROL-DOSE-NOT-REACHED": [
    "\"With AdamW + Adam the grains do not differ by the 10 pp bar at wd 0.1 or 1.0, but even at wd 1.0 the scalar "
    "arm's realised peak shrink stayed below half the control's.\"  The standard recipe does not collapse here; "
    "whether the pairing is immune at the control's dose is NOT decided (the learned step size kept the dose low).  "
    "C1's gate does not fire.", BOUND_NOTE, NOT_ELSE]}
ATTR_NOTE = {
    "ATTR-BASE-PROTECTS": "M (SGDm base + Adam meta) collapses and L (AdamW base + Lion meta) does not: in this cell the "
                          "BASE swap, not the meta swap, is what removes the collapse -- read beside L's dose word.",
    "ATTR-META-PROTECTS": "L (AdamW base + Lion meta) collapses and M (SGDm base + Adam meta) does not: in this cell the "
                          "META swap is what removes the collapse -- read beside M's dose word.",
    "ATTR-EITHER-PROTECTS": "Neither single-swap cell collapses: either swap alone removes the collapse here.",
    "ATTR-NEITHER-ALONE": "BOTH single-swap cells collapse: neither swap alone removes it.",
    "ATTR-UNRESOLVED": "At least one single-swap cell is PARTIAL or UNREADABLE: no attribution is licensed.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runsdir")
    ap.add_argument("--against", default="")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    R = a.runsdir.rstrip("/")
    fails = []

    # ---------------- RECONSTRUCTION of the scorer's stdout ----------------
    P("=" * 78)
    P(" caw2 -- THE STANDARD-RECIPE GATE, WITH EACH CONFOUND IN ITS OWN CELL.  Does the shared-step-size collapse")
    P("         survive the META swap (M), the BASE swap (L), both (A: AdamW+Adam, the standard recipe) and a 10x")
    P("         dose (X), against the in-batch mechanism cell K01?")
    P(" ResNet18_c100 / CIFAR100   100 epochs   seeds 160,161,162   AUGMENT=1  BETA_CLIP=-15:-2.3026  PROBE=100  PROBE_TENSOR=1")
    for arm in ARMS:
        t = TAB[arm]
        P("   %-4s %-5s+%-4s %-9s wd %-4s  %s" % (arm, t[0], t[1], t[2], t[3], t[7]))
    P(" CO-PRIMARY: G_A = AL - AS.  Every state is WITHIN its cell.  plateau5 = mean TEST over epochs 95..99.")
    P(" BARS ARE FROZEN LITERALS (O2).  ALPHA-INDEPENDENT DECAY IS NOT TESTED.")
    P("=" * 78)
    csvp = os.path.join(a.repo, "results", "all_runs.csv")
    tsvp = os.path.join(a.repo, "results", "CORPUS-EXCLUSIONS.tsv")
    excl = set()
    tl = [ln.rstrip("\n") for ln in open(tsvp) if ln.strip() and not ln.startswith("#")]
    hd = tl[0].split("\t")
    for ln in tl[1:]:
        f = dict(zip(hd, ln.split("\t")))
        excl.add((f["run"], f["job_id"]))
    rows = [r for r in csv.DictReader(open(csvp)) if not r.get("run", "").startswith("caw2-")]
    nkeep = sum(1 for r in rows if (r.get("run"), r.get("job_id")) not in excl)
    P("corpus: %d rows after corpus_exclusions.filter_rows, caw2- excluded  (DISCLOSURE ONLY -- no bar, sigma, branch "
      "or stamp reads it)" % nkeep)

    cand = {}
    for fn in sorted(os.listdir(R)):
        mm = OUT_NAME.match(fn)
        if mm:
            cand.setdefault((mm.group(1), int(mm.group(2))), []).append((int(mm.group(3)), fn))
    runs = {}
    for k, lst in cand.items():
        jid, fn = max(lst)
        rr = parse_out(os.path.join(R, fn))
        rr["file"], rr["job"], rr["nfiles"] = fn, jid, len(lst)
        runs[k] = rr
    P("")
    P("COMPLETE  27 runs, epochs 0..99, RUN_DONE, no traceback, finite plateau5")
    for s in SEEDS:
        for arm in ARMS:
            r = runs[(arm, s)]
            P("  %-4s caw2-%s-s%d  %d epoch lines, RUN_DONE=%s, traceback=%s, plateau5 %.4f (train %.4f)"
              % ("OK" if r["complete"] else "MISS", arm, s, len(r["eps"]), r["run_done"], r["tb"], r["p5"], r["t5"]))
            if not r["complete"]:
                fails.append("incomplete %s-s%d" % (arm, s))

    def chk(cond, label, extra=""):
        P("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
        if not cond:
            fails.append(label)

    P("")
    P("G-ARGS    [HARNESS-NULL] every run's OWN ARGS line == its arm's registered flags (base, meta, decay, grain)")
    for s in SEEDS:
        for arm in ARMS:
            r = runs[(arm, s)]
            ok = len(r["args"]) == 1
            if ok:
                f = tokens_args(r["args"][0])
                want = dict(expected_args(arm, s))
                ok = (all(len(v) == 1 for v in f.values()) and all(f.get(k, [None])[-1] == v for k, v in want.items())
                      and sorted(set(f) - set(want)) == ["save-directory"]
                      and (f["save-directory"][-1] or "").rstrip("/").endswith("/caw2"))
            t = TAB[arm]
            chk(ok, "G-ARGS %s-s%d (%s+%s %s wd %s)" % (arm, s, t[0], t[1], t[2], t[3]))
    P("")
    P("G-ENV     the environment rode the ENV line on every run")
    envs = sorted(set(" ".join(x for x in ln.split() if not x.startswith("PROBE_DIR=")) for r in runs.values() for ln in r["env"]))
    chk(len(envs) == 1 and all(len(r["env"]) == 1 for r in runs.values()),
        "G-ENV one distinct ENV line (PROBE_DIR stripped), one per run", "%d distinct" % len(envs))
    chk(envs == [ENV_EXPECTED], "G-ENV the ENV line is cvt1 ... cwd5's, byte for byte (PROBE_DIR stripped)")
    P("")
    P("G-PT      [HARNESS-NULL] ONE PROBE_TENSOR line per run == its arm's witness (grain, 62 tensors, META alg)")
    for s in SEEDS:
        for arm in ARMS:
            pt = runs[(arm, s)]["pt"]
            chk(len(pt) == 1 and pt[0].startswith(pt_prefix(arm) + " dir="), "G-PT %s-s%d" % (arm, s))
    P("")
    P("G-WITNESS the five `off` lines once each; no COMP_HOLD / WINDOW_HOLD line")
    for s in SEEDS:
        for arm in ARMS:
            ls = runs[(arm, s)]["lines"]
            ok = all([x for x in ls if x.startswith(w.split(":")[0])] == [w] for w in OFF)
            ok = ok and not any(x.startswith(("COMP_HOLD", "WINDOW_HOLD")) for x in ls)
            chk(ok, "G-WITNESS %s-s%d" % (arm, s))
    P("")
    P("G-STRUCT  the batch's own PARTITION-MANIFEST.txt == caw2_design.manifest_text(), byte for byte")
    mpath = os.path.join(R, PREFIX, "PARTITION-MANIFEST.txt")
    P("  manifest: %s (<runsdir>/caw2/)" % os.path.join("<runsdir>", PREFIX, "PARTITION-MANIFEST.txt"))
    man = open(mpath).read()
    ml = man.split("\n")
    tens = [x.split() for x in ml if x.startswith("TENSOR ")]
    rest = [x for x in ml if not x.startswith("TENSOR ")]
    want = ["BATCH caw2", "NETWORK ResNet18_c100", "NUM_PARAM_TENSORS 62", "TOTAL_PARAMS 11220132", "CLIP_C -15:-2.3026",
            "EPOCHS 100", "META_STEPS 50000", "SEEDS 160,161,162",
            "CARRIERS 50:layer4.0.bn2.weight,53:layer4.0.shortcut.1.weight,59:layer4.1.bn2.weight"]
    for arm in ARMS:
        t = TAB[arm]
        want.append("ARM %s PAIR %s BASE %s META %s GRAIN %s WD %s CELL %s" % (arm, t[6], t[0], t[1], t[2], t[3], t[5]))
        ea = [(k, v) for k, v in expected_args(arm, 0)]
        ea.insert(len(ea) - 1, ("save-directory", "<save>"))
        want.append("ARGS %s %s" % (arm, " ".join("--%s %s" % kv for kv in ea)))
        want.append("PTWITNESS %s %s" % (arm, pt_prefix(arm)))
    want.append("")
    struct_ok = (rest == want and len(tens) == NTENS and [int(x[1]) for x in tens] == list(range(1, NTENS + 1))
                 and sum(int(x[3]) for x in tens) == TOTPAR and all(tens[i][2] == n for i, n in zip(CARRIERS0, CARRIER_NAMES))
                 and man.endswith("\n"))
    chk(struct_ok, "G-STRUCT byte-identical")
    P("")
    P("G-PROV    PROVENANCE.txt")
    P("  provenance: %s (<runsdir>/caw2/)" % os.path.join("<runsdir>", PREFIX, "PROVENANCE.txt"))
    prov = {}
    for ln in open(os.path.join(R, PREFIX, "PROVENANCE.txt")):
        p = ln.split()
        if len(p) >= 2:
            prov.setdefault(p[0], " ".join(p[1:]))
    for k, v, lab in PROV_WANT:
        chk(prov.get(k) == v, "G-PROV %s" % lab, "%s=%s" % (k, str(prov.get(k))[:16]))
    P("")
    P("G-PROBE   [HARNESS-NULL] 500 records per run; the step-size count is the GRAIN's (1 or 62); on the scalar arms "
      "the per-tensor vote sums to the harness's applied term (within 0.001 of the vote's mass sum|L_i|) with its "
      "sign, under the arm's OWN meta alg")
    recs = {}
    for s in SEEDS:
        for arm in ARMS:
            rc = load_recs(R, arm, s)
            recs[(arm, s)] = rc
            nb = 1 if TAB[arm][2] == "scalar" else NTENS
            bad = sum(1 for r in rc if not (isinstance(r, dict) and isinstance(r.get("step"), int)
                                            and isinstance(r.get("beta"), list) and len(r["beta"]) == nb
                                            and isinstance(r.get("z_tensor"), list) and len(r["z_tensor"]) == NTENS))
            line = "%d records, %d malformed / wrong step-size count" % (len(rc), bad)
            ok = len(rc) == N_RECORDS and bad == 0
            if TAB[arm][2] == "scalar":
                n = off = wsg = nf = 0
                for r in rc:
                    v = vote(r, TAB[arm][1]) if isinstance(r, dict) else None
                    if v is None:
                        nf += 1
                        continue
                    L, app = v
                    sm = sum(L)
                    sc = sum(abs(x) for x in L)
                    n += 1
                    off += not (abs(sm - app) <= TOL * sc or abs(sm - app) <= 1e-30)
                    wsg += sgn(sm) != sgn(app) and abs(app) > TOL * sc
                line += "; decomposition (%s): %d usable, %d off, %d wrong sign, %d non-finite (disclosed)" % (
                    TAB[arm][1], n, off, wsg, nf)
                ok = ok and off == 0 and wsg == 0 and n > 0
            chk(ok, "G-PROBE %s-s%d (%s, %d step size(s))" % (arm, s, TAB[arm][2], nb), line)
    V = dict((arm, [runs[(arm, s)]["p5"] for s in SEEDS]) for arm in ARMS)
    Tr = dict((arm, [runs[(arm, s)]["t5"] for s in SEEDS]) for arm in ARMS)
    M = dict((arm, avg(V[arm])) for arm in ARMS)
    T = dict((arm, avg(Tr[arm])) for arm in ARMS)
    P("")
    P("G-FLOOR / G-CEIL")
    hi = max(M.values())
    chk(hi >= FLOOR_MIN, "G-FLOOR max arm mean >= 15.00 pp", "max %.4f" % hi)
    chk(hi <= CEIL_MAX, "G-CEIL max arm mean <= 90.00 pp", "max %.4f" % hi)

    P("")
    P("LEVELS    plateau5 (TEST) and TRAIN, IN BATCH; tail slope = TEST OLS over epochs 80-99 (descriptive)")

    def slope(eps):
        xs = [e for e in range(80, 100) if e in eps]
        ys = [eps[e][1] for e in xs]
        mx, my = avg(xs), avg(ys)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    for arm in ARMS:
        t = TAB[arm]
        v = V[arm]
        P("  %-4s %-5s+%-4s %-9s wd %-4s TEST %.4f  sd %.4f  range %.4f   TRAIN %.4f   tail slope %s pp/ep"
          % (arm, t[0], t[1], t[2], t[3], M[arm], sdev(v), max(v) - min(v), T[arm],
             "/".join("%.3f" % slope(runs[(arm, s)]["eps"]) for s in SEEDS)))
        P("           seeds: %s" % ", ".join("s%d %.4f (train %.4f)" % (s, x, y) for s, x, y in zip(SEEDS, v, Tr[arm])))
    P("")
    P("G-HW      DISCLOSURE ONLY: the GPU each run's own nvidia-smi line names -- NO gate reads it")
    hw = sorted(set(str(runs[(arm, s)]["gpu"]) for s in SEEDS for arm in ARMS))
    P("  distinct devices: %s" % repr(hw))
    P("")
    P("SIGMA     O2: max(frozen floor, in-batch) -- nothing from the corpus")
    ss = sum(sum((x - M[arm]) ** 2 for x in V[arm]) for arm in ARMS)
    df = sum(len(V[arm]) - 1 for arm in ARMS)
    sig_in = math.sqrt(ss / df)
    which, sig = ("SIGMA_PRIOR_frozen", SIGMA_PRIOR) if SIGMA_PRIOR >= sig_in else ("SIGMA_INBATCH", sig_in)
    P("  %-20s %.6f" % ("SIGMA_PRIOR_frozen", SIGMA_PRIOR))
    P("  %-20s %.6f" % ("SIGMA_INBATCH", sig_in))
    se = sig * math.sqrt(2.0 / 3)
    P("  SIGMA_USED %.6f (= %s); SE_ARM_DIFF %.6f; 2 SE %.6f is the HALF-WIDTH of a +/-2 SE interval, NOT a bound on "
      "any effect (278.6 C1)" % (sig, which, se, 2 * se))
    P("")
    P("THE DOSE   the REALISED per-step shrink a*wd (median over the arm's step sizes, per record), from the arm's OWN "
      "probe records; arm value = median over its 3 seeds")
    DO = {}
    for arm in ARMS:
        per = [dose(recs[(arm, s)], TAB[arm][4]) for s in SEEDS]
        DO[arm] = {"peak": median([p[0] for p in per]), "pre": median([p[1] for p in per]),
                   "late": median([p[2] for p in per]), "seeds": per}
        t = TAB[arm]
        P("  %-4s %-5s+%-4s %-9s wd %-4s  DOSE_PEAK %.4e  DOSE_PRE(step<=8500) %.4e  late median %.4e   per seed peak %s"
          % (arm, t[0], t[1], t[2], t[3], DO[arm]["peak"], DO[arm]["pre"], DO[arm]["late"],
             " / ".join("%.4e" % p[0] for p in per)))
    RHO, DW = {}, {}
    for c in CELLS:
        k = SCAL[c]
        RHO[c] = DO[k]["peak"] / DO["K01"]["peak"]
        pre = DO[k]["pre"] / DO["K01"]["pre"]
        DW[c] = "BELOW" if RHO[c] < DOSE_LO else ("ABOVE" if RHO[c] > DOSE_HI else "MATCHED")
        P("  RHO_%s = DOSE_PEAK(%s) / DOSE_PEAK(K01) = %.4f -> DOSE-%s-%s   (pre-window ratio %.4f, descriptive)"
          % (c, k, RHO[c], c, DW[c], pre))
    P("")
    P("THE CELLS (every contrast WITHIN its cell)")
    ST, CP = {}, {}
    for c in CELLS:
        k, L = SCAL[c], LAYR[c]
        ST[c] = state(V[L], V[k])
        g = M[L] - M[k]
        CP[c] = (g, T[L] - T[k])
        nc = sum(1 for x in V[k] if x <= R50 * M[L])
        t = TAB[k]
        P("  %-2s %-5s+%-4s wd %-4s  scalar %7.4f  layerwise %7.4f   G_%s = %+.4f pp = %+.2f SE   R_%s = %.4f   -> %s"
          % (c, t[0], t[1], t[3], M[k], M[L], c, g, g / se, c, M[k] / M[L], ST[c]))
        P("           +/-2 SE [%+.4f, %+.4f]   TRAIN gap %+.4f   seeds %s   scalar seeds at/below the bar: %d/3"
          % (g - 2 * se, g + 2 * se, T[L] - T[k],
             " / ".join("%+.3f" % (runs[(L, s)]["p5"] - runs[(k, s)]["p5"]) for s in SEEDS), nc))
    P("           bars: COLLAPSE iff scalar <= 0.50 x layerwise; NOGAP iff G < 10.0 pp; reference healthy iff layerwise "
      ">= 55.0 and its seed range <= 5.0")
    kst = "COLLAPSED" if M["K01"] <= K01_MAX else "NOT-COLLAPSED"
    P("  CONTROL K01 (SGDm 0.99 + Lion, wd 0.1, scalar) %.4f  -> %s (bar <= 30.0)" % (M["K01"], kst))
    P("  CO-PRIMARY G_A = %+.4f pp = %+.2f SE" % (CP["A"][0], CP["A"][0] / se))
    P("")
    P("THE VOTE    DOM_C per scalar arm: C fitted on seed 160 (top-3 by signed contribution), scored on seeds 161/162;")
    P("            and the DECLARED carriers {50,53,59} on all seeds (KEY, descriptive)")

    def domc(rs, meta, C):
        n = d = 0
        for r in rs:
            v = vote(r, meta) if isinstance(r, dict) else None
            if v is None:
                continue
            L, app = v
            s = sgn(app)
            sc = sum(L[i] for i in C)
            rest_ = sum(abs(L[i]) for i in range(NTENS) if i not in C)
            n += 1
            d += (s != 0 and sgn(sc) == s and abs(sc) > rest_)
        return (d / n if n else float("nan")), n
    DOMS = {}
    for arm in SCALAR_ARMS:
        meta = TAB[arm][1]
        acc = [0.0] * NTENS
        for r in recs[(arm, FIT_SEED)]:
            v = vote(r, meta) if isinstance(r, dict) else None
            if v is None:
                continue
            L, app = v
            s = sgn(app)
            for i, x in enumerate(L):
                acc[i] += s * x
        C = tuple(sorted(sorted(range(NTENS), key=lambda i: (-acc[i], i))[:3]))
        dm, n = domc([r for s in SCORE_SEEDS for r in recs[(arm, s)]], meta, C)
        dd, dn = domc([r for s in SEEDS for r in recs[(arm, s)]], meta, CARRIERS0)
        DOMS[arm] = dm
        P("  %-4s meta %-4s  C_fit = {%s}  DOM_C %.4f (n %d)   declared {50,53,59} DOM %.4f (n %d)"
          % (arm, meta, ",".join("%d" % (i + 1) for i in C), dm, n, dd, dn))

    br = branch_of(ST, M["K01"], RHO["X"])
    word = "M-%s+L-%s+A-%s+X-%s+K-%s" % (ST["M"], ST["L"], ST["A"], ST["X"], kst)
    attr = attr_of(ST)
    if br == "PARTIAL-AT-STANDARD-RECIPE" or br.startswith("NO-COLLAPSE-"):
        gw = "GATE-DOES-NOT-FIRE"
    elif br != "HAZARD-AT-STANDARD-RECIPE":
        gw = "GATE-UNREADABLE"
    elif not DOMS["K01"] >= DOM_BAR:
        gw = "GATE-UNREADABLE-DOMC-CONTROL"
    else:
        gw = "GATE-FIRES-A" if (ST["A"] == "COLLAPSE" and DOMS["AS"] >= DOM_BAR) else "GATE-COLLAPSE-WITHOUT-DOMINANCE"
    stamps = [word, attr, gw] + ["DOSE-%s-%s" % (c, DW[c]) for c in CELLS] + [
        "HARNESS-CLEAN", "BOTH-GRAINS-IN-BATCH", "CONTROL-IN-BATCH", "ALPHA-SCALED-DECAY-ONLY",
        "ALPHA-INDEPENDENT-NOT-TESTED", "HARNESS-ADAMW-M-UNNORMALISED", "X-IS-A-DOSE-ARM-NOT-A-STANDARD-VALUE",
        "ONE-NETWORK-RESNET18", "ONE-CELL-OTHERWISE", "HORIZON-100-ONLY", "THREE-SEEDS", "WD-1E-2-NOT-RUN",
        "LION-BASE-NOT-RUN", "STEP1-HALF-OF-GATE-ONLY", "LEADLAG-NOT-READ", "FLOOR-READINGS-ARE-BOUNDS",
        "SIGMA-PRIOR-FROZEN" if which == "SIGMA_PRIOR_frozen" else "SIGMA-INBATCH"]
    div = [arm for arm in ARMS if max(V[arm]) - min(V[arm]) > DIVERGED_BAR]
    if div:
        stamps.append("DIVERGED-%s" % "-".join(div))
    if ST["A"] == "COLLAPSE" and ST["X"] == "NOGAP":
        stamps.append("NON-MONOTONE-DOSE")
    for c in CELLS:
        if CP[c][0] <= -2 * se and ST[c] == "NOGAP":
            stamps.append("SCALAR-AHEAD-%s-DESCRIPTIVE" % c)
    stamps.append(("HW-UNIFORM-%s" % hw[0].replace(" ", "_")) if len(hw) == 1 else "HW-MIXED")
    stamps.append("TRAIN-AGREES" if all((t > 0) == (p > 0) for p, t in CP.values() if abs(p) >= GAP_BAR) else "TRAIN-DISAGREES")
    final = "FINAL: %s | %s" % (br, " | ".join(stamps))
    P("")
    P("=" * 78)
    P(final)
    P("=" * 78)
    P("")
    P("BETWEEN-BATCH, NON-GATING (frozen literals): the control's landed levels %s; caw2 K01 %.4f"
      % (", ".join("%s %.4f" % kv for kv in K01_LANDED), M["K01"]))
    P("REGISTERED PREDICTION (UNSURE): A-NOGAP / ATTR-BASE-PROTECTS / X-UNSURE; this batch: A-%s, %s" % (ST["A"], attr))
    P("")
    P("WHAT THIS DECIDES -- as registered, before any run existed:")
    for ln in LIC.get(br, ["<licence text for %s NOT re-typed by this attack>" % br]):
        P("  " + ln)
    P("  ATTRIBUTION %s: %s  (DOSE-M-%s, DOSE-L-%s)" % (attr, ATTR_NOTE[attr], DW["M"], DW["L"]))
    P("")
    P("THE GATE WORD %s: the Step-1 HALF of the ICML plan's gate only (A COLLAPSE AND DOM_C(AS) >= 0.50, C fixed on seed "
      "160).  Step 0's lead-lag is NOT read here (V1: biased toward 'lag' by construction)." % gw)
    P("")
    P("OWED AT LANDING: CORPUS-EXCLUSIONS rows -- LS/LL/AS/AL ARGS_MOMENTUM_BASE (momentum-param-base 0.9); XS/XL "
      "deviate on TWO ARGS kinds (0.9 and wd 1.0), which corpus_exclusions.py cannot list today (as caw1's AS2/AL2 "
      "would have; reported, not fixed); K01 / MS / ML none (the meta alg is a CSV cell-key column).")
    recon = list(OUT)

    # ---------------- ATTACK SECTION (not printed by the scorer) ----------------
    P("")
    P("#" * 78)
    P("# ATTACK (caw2_attack_indep.py only; the scorer prints none of this).  Readings here are DESCRIPTIVE unless")
    P("# labelled a check; none of them is a registered gate, bar, state or licence.")
    P("#" * 78)

    def A(cond, label):
        P("  %-4s %s" % ("PASS" if cond else "FAIL", label))
        if not cond:
            fails.append("ATTACK " + label)
    P("A0  registered files at their registered bytes (read as bytes, never executed):")
    for rel, want_sha in sorted(REG_FILES.items()):
        p = os.path.join(a.repo, rel)
        A(os.path.exists(p) and sha(p) == want_sha, "%s sha256 %s" % (rel, want_sha[:16]))
    P("A1  raw-parser strictness over the 27 .out files (this file's own exact-line regex):")
    A(all(r["nfiles"] == 1 for r in runs.values()) and len(runs) == 27, "exactly one .out per (arm, seed); 27 runs")
    A(sum(r["dup"] for r in runs.values()) == 0, "no duplicated epoch line (the scorer's dict would keep the last one)")
    A(sum(r["loose"] for r in runs.values()) == 0, "no 'Epoch' line outside the exact format")
    A(all(sorted(r["eps"]) == list(range(100)) for r in runs.values()), "epochs 0..99 on every run")
    A(all(math.isfinite(x) for r in runs.values() for t in r["eps"].values() for x in t), "no nan accuracy on any epoch")
    P("A2  probe-record structure:")
    steps_ok = all([r["step"] for r in recs[k]] == list(range(0, 50000, 100)) for k in recs)
    A(steps_ok, "every probe.jsonl carries steps 0, 100, ..., 49,900 in order (500 records)")
    inb = all(BETA_LO - 1e-4 <= x <= BETA_HI + 1e-4 for k in recs for r in recs[k] for x in r["beta"])
    A(inb, "every beta lies inside BETA_CLIP [-15, -2.3026] (to 1e-4)")
    P("A3  the REALISED per-step shrink a*wd, per arm, beside its level (plateau5 TEST mean; DESCRIPTIVE):")
    P("      arm  base +meta  grain      wd    TEST     DOSE_PEAK   peak step (per seed)      DOSE_PRE    late median")
    for arm in ARMS:
        t = TAB[arm]
        pk_steps = []
        for s in SEEDS:
            best, bstep = -1.0, None
            for r in recs[(arm, s)]:
                v = shrink(r, t[4])
                if v is not None and v > best:
                    best, bstep = v, r["step"]
            pk_steps.append(bstep)
        P("      %-4s %-5s+%-4s %-9s  %-4s  %7.4f   %.4e   %-24s  %.4e  %.4e"
          % (arm, t[0], t[1], t[2], t[3], M[arm], DO[arm]["peak"], "/".join(str(x) for x in pk_steps),
             DO[arm]["pre"], DO[arm]["late"]))
    P("A4  clip-floor readings (beta <= -15 + 1e-6 on the arm's step size(s)); the late median of an arm pinned at the")
    P("    floor is exp(-15) * wd = %.4e at wd 0.1, i.e. a FLOOR reading (a BOUND, 164.6), not a measured shrink:" % (math.exp(-15) * 0.1))
    for arm in SCALAR_ARMS:
        n_all = [sum(1 for r in recs[(arm, s)] if r["beta"][0] <= BETA_LO + 1e-6) for s in SEEDS]
        n_late = [sum(1 for r in recs[(arm, s)] if r["step"] >= LATE_FROM and r["beta"][0] <= BETA_LO + 1e-6) for s in SEEDS]
        first = []
        for s in SEEDS:
            fs = [r["step"] for r in recs[(arm, s)] if r["beta"][0] <= BETA_LO + 1e-6]
            first.append(str(fs[0]) if fs else "-")
        P("      %-4s records at the floor %s of 500; in the late window (step >= 40,000) %s of 100; first floor step %s"
          % (arm, "/".join(str(x) for x in n_all), "/".join(str(x) for x in n_late), "/".join(first)))
    P("A5  K01's pre-window: the Lion meta update moves beta by at most meta-step 1e-3 per step, so beta(8,500) -")
    P("    beta(0) <= 8.5; the observed rise (per seed) and the implied mean per-step move:")
    for s in SEEDS:
        rr = dict((r["step"], r["beta"][0]) for r in recs[("K01", s)])
        P("      s%d  beta(0) %.6f  beta(8500) %.6f  rise %.6f  = %.6e per step" % (
            s, rr[0], rr[8500], rr[8500] - rr[0], (rr[8500] - rr[0]) / 8500))
    P("A6  how far the dose word sits from its bar (the branch reads RHO_X only):")
    xs = [p[0] for p in DO["XS"]["seeds"]]
    ks = [p[0] for p in DO["K01"]["seeds"]]
    P("      RHO_X (registered, median/median) %.4f; worst case max(XS)/min(K01) %.4f; mean/mean %.4f; bar 0.5 is %.1fx above"
      % (RHO["X"], max(xs) / min(ks), avg(xs) / avg(ks), DOSE_LO / (max(xs) / min(ks))))
    A(max(xs) / min(ks) < DOSE_LO, "RHO_X < 0.5 under the worst seed pairing too (the dose word does not hinge on the median)")
    xl = [max(max(math.exp(b) for b in r["beta"]) * 1.0 for r in recs[("XL", s)]) for s in SEEDS]
    P("      XL layerwise, MAX over its 62 step sizes (not the registered median): peak shrink per seed %s; / K01 median peak = %s"
      % (" / ".join("%.4e" % x for x in xl), " / ".join("%.4f" % (x / DO["K01"]["peak"]) for x in xl)))
    P("A7  leave-one-seed-out (2 seeds per arm; DESCRIPTIVE -- the registered decision is 3 seeds):")
    for drop in SEEDS:
        keep = [i for i, s in enumerate(SEEDS) if s != drop]
        V2 = dict((arm, [V[arm][i] for i in keep]) for arm in ARMS)
        st2 = dict((c, state(V2[LAYR[c]], V2[SCAL[c]])) for c in CELLS)
        dk = median([DO["K01"]["seeds"][i][0] for i in keep])
        dx = median([DO["XS"]["seeds"][i][0] for i in keep])
        P("      drop s%d: M-%s L-%s A-%s X-%s  K01 %.4f  RHO_X %.4f -> %s | %s"
          % (drop, st2["M"], st2["L"], st2["A"], st2["X"], avg(V2["K01"]), dx / dk,
             branch_of(st2, avg(V2["K01"]), dx / dk), attr_of(st2)))
    P("A8  exclusion rows owed (results/CORPUS-EXCLUSIONS.tsv), read from each run's OWN ARGS line against the ARGS")
    P("    kinds' standard values (momentum-param-base 0.99, weight-decay-base 0.1; CORRECTIONS 263):")
    n_owed = 0
    for arm in ARMS:
        for s in SEEDS:
            r = runs[(arm, s)]
            f = tokens_args(r["args"][0])
            kinds = []
            if float(f["momentum-param-base"][-1]) != 0.99:
                kinds.append("ARGS_MOMENTUM_BASE: momentum-param-base=%s" % f["momentum-param-base"][-1])
            if float(f["weight-decay-base"][-1]) != 0.1:
                kinds.append("ARGS_WD_BASE: weight-decay-base=%s" % f["weight-decay-base"][-1])
            n_owed += len(kinds) > 0
            P("      %-16s job %d  arm %-3s  %s" % ("caw2-%s-s%d" % (arm, s), r["job"], arm,
                                                  " + ".join(kinds) if kinds else "none"))
    P("      runs owing a row: %d (single-kind 12, two-kind 6 expected by CORRECTIONS 290.9)" % n_owed)

    # ---------------- COMPARISON ----------------
    P("")
    if a.against:
        sc = open(a.against).read().replace(R, "<runsdir>").split("\n")
        if sc and sc[-1] == "":
            sc = sc[:-1]
        same = sum(1 for x, y in zip(sc, recon) if x == y)
        diffs = [i for i in range(max(len(sc), len(recon)))
                 if i >= len(sc) or i >= len(recon) or sc[i] != recon[i]]
        P("RECONSTRUCTION vs the scorer's stdout (runsdir echo normalised to <runsdir>): scorer %d lines, re-derived %d "
          "lines, %d identical, %d differing" % (len(sc), len(recon), same, len(diffs)))
        for i in diffs[:10]:
            P("   line %d  scorer  : %r" % (i + 1, sc[i] if i < len(sc) else None))
            P("   line %d  attack  : %r" % (i + 1, recon[i] if i < len(recon) else None))
        if diffs:
            fails.append("reconstruction differs on %d lines" % len(diffs))
    P("ATTACK VERDICT: %s  (%d failure(s))" % ("PASS" if not fails else "FAIL", len(fails)))
    for f in fails:
        P("   FAILED  %s" % f)
    sys.stdout.write("\n".join(OUT) + "\n")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
