#!/usr/bin/env python3
# =============================================================================
# cG1_tl_score.py -- THE REGISTERED SCORER FOR `tl1`, THE TWO-LEVEL SHRINKAGE
#                    LADDER (gap G1: the proposal as literally written).
#
# STANDING RULE 21: this file is git-committed BEFORE any tl1 run exists.  Every
# bar, band and refusal below is frozen here.  selftest() asserts each constant
# against bin/cG1_twolevel_ladder.sh's OWN TEXT so the registration and the
# batch cannot drift apart, and RE-DERIVES the noise floor and both priors from
# results/all_runs.csv rather than quoting a number.
#
# STANDING RULE 16: run this file UNEDITED and quote its verdict.  Passing its
# documented arguments is not editing it.  Once a tl1 run exists it is frozen.
#
# -----------------------------------------------------------------------------
# THE QUESTION
# -----------------------------------------------------------------------------
# The proposal is: learn a scalar and layerwise step sizes JOINTLY and shrink the
# layerwise ones toward the scalar with a tunable strength.  What the tree
# contained before PATCH_TWOLEVEL was a per-step PROJECTION
#
#       shrink:   beta  <-  beta - lam * (beta - mean(beta))
#
# whose "strength" lam is a TIME CONSTANT: applied once per optimizer step, the
# surviving deviation after T steps is (1-lam)^T, i.e. a half-life of
# ln2 / -ln(1-lam) steps.  A 100-epoch CIFAR-10 run at batch_size 100 is 50,000
# steps, so lam = 1e-3 runs 72 half-lives and every rung of the only archived
# ladder measured FULL POOLING.  The ladder was flat by construction.
#
# PATCH_TWOLEVEL replaces the projection with the two-level parameterisation:
#
#       beta_g(t) = s(t) + d_g(t)                                          (1)
#       s     <-  META( s,   sum_g z_g,  eta       )                       (2)
#       d_g   <-  META( d_g, z_g,        rho * eta )                       (3)
#
# z_g = HtT_gradft[g] is the meta-gradient HF already computes; META is the run's
# own --alg-meta rule with one independent state per level.  rho is the dial
# (env ETA_RATIO); SHRINKAGE STRENGTH = 1 - rho.  rho is dimensionless and
# stationary in T -- it scales the deviations' meta-step-size, it does not decay
# them -- and rho = 0 pins d_g at exactly 0 for every t, so that arm IS the
# scalar arm with the shared level still learned at full rate.
#
# -----------------------------------------------------------------------------
# THE DESIGN.  42 jobs, ONE submission, so BATCH -- the unit of replication in
# this corpus (F(62,85)=5.47, p 6.9e-13; SEED is null, F 1.21, p 0.213) -- is
# held constant and every contrast below is WITHIN this batch.
#
#   arm   in {r0, r003, r01, r03, r1}   HIER=twolevel, layerwise, ETA_RATIO=rho
#         +  plain                      HIER unset,    layerwise
#         +  scal                       HIER unset,    scalar
#   alpha0 in {1e-6, 1e-3}              co-primary; NOT pooled
#   seed  in {0, 1, 2}
#   ResNet18 / CIFAR-10 / SGDm+Lion / meta-stepsize 1e-3 / 100 ep / batch 100 /
#   gamma 1 / BETA_CLIP -15:-2.3026 / AUGMENT=1 / PROBE=500.
#
# -----------------------------------------------------------------------------
# THE PRIMARY STATISTIC, FROZEN
# -----------------------------------------------------------------------------
#   plateau5 = mean of the LAST 5 test epochs.  The CSV `plateau` column
#   (mean-of-last-20) is BANNED as primary and is never read by this file;
#   `best_test` is not a plateau and is never read either.
#
#   M(arm)     = mean plateau5 over that arm's completed seeds, within alpha0
#   ENDPOINTS  = {r0, r1, plain}       the pure-scalar end, the free-deviation
#                                      end, and the pre-existing free-layerwise
#                                      reference
#   INTERIOR   = {r003, r01, r03}      genuinely partial pooling
#   PEAK       = max_INTERIOR M  -  max_ENDPOINTS M
#   SPAN       = max_rho M - min_rho M   over the five rho rungs
#
# PEAK is the quantity the proposal predicts is positive: partial shrinkage
# toward the scalar beats BOTH not shrinking at all and shrinking all the way.
#
# -----------------------------------------------------------------------------
# THE BARS.  DERIVED FROM THE MEASURED NOISE FLOOR, FROZEN HERE
# -----------------------------------------------------------------------------
# noise_floor_from_csv() re-derives the pooled WITHIN-CELL standard deviation of
# plateau5 over every populated cell of this exact configuration in the archived
# corpus (ResNet18 / CIFAR-10 / SGDm+Lion / meta-stepsize 1e-3 / box
# -15:-2.3026 / AUGMENT=1 / 100 ep / batch 100 / gamma 1 / not collapsed):
# the 44 archived additive-layerwise (batch x alpha0 x r) cells and the 25
# archived plain layerwise/scalar (batch x alpha0 x granularity) cells.
#
#     SIGMA_W = 0.1735 pp   (df = 153)
#     SE of a 3-seed arm mean         = SIGMA_W / sqrt(3)   = 0.1002 pp
#     SE of a difference of two       = SIGMA_W * sqrt(2/3) = 0.1417 pp
#
# SUPPORT_BAR = 0.50 pp.  Under the flat null "all six arms have one true mean",
#   PEAK is max-of-3 minus max-of-3 exchangeable 3-seed means; a 10^6-draw
#   simulation (reproduced by selftest() with a fixed seed) puts its 99th
#   percentile at 0.248 pp and P(PEAK >= 0.50) at 0.00000.  0.50 pp is also
#   exactly HALF the interior lift the archived additive r-curve already shows
#   at alpha0=1e-6 (+0.997 pp over the better endpoint), so the bar is both a
#   <1e-5 false-positive threshold and half the effect size being claimed.
#   Power against a true +0.997 pp interior lift is 0.999.
#
# NULL_BAR = 0.25 pp.  This is the flat null's own 99th percentile, so a genuinely
#   flat ladder returns NULL 99% of the time, while P(NULL | true lift 0.997 pp)
#   is 0.00000 in the same 10^6 draws.  PEAK <= 0.25 therefore licenses the
#   statement "no interior optimum at the archived effect size", and nothing
#   stronger.
#
# Anything strictly between the two bars is UNRESOLVED and is reported as
# UNRESOLVED.  At a true lift of 0.50 pp the design returns UNRESOLVED 66% of
# the time; that is a known and accepted limit of 3 seeds, registered here so it
# cannot be discovered afterwards and spun.
#
# TWO CO-PRIMARY TESTS.  alpha0 = 1e-6 and alpha0 = 1e-3 are reported
# SEPARATELY and are never pooled: the archived interior lift exists at 1e-6 and
# does not at 1e-3, and combining them would hide exactly the thing the batch is
# for.  Two co-primary tests at a <1e-5 bar each leave the family-wise
# false-positive rate under 1e-4.  There is no "significant at either alpha0"
# headline: each alpha0 gets its own verdict line and that is the report.
#
# -----------------------------------------------------------------------------
# THE GATES, IN ORDER
# -----------------------------------------------------------------------------
# G0  PROVENANCE, absolutely gating.  Every design fact is read from the run's
#     OWN ARGS/ENV line, never from its filename.  Any repeated flag voids that
#     run (STANDING RULE 20 -- `ml2` and `sm3` were both run and reported with a
#     duplicated --alg-meta before anyone read an ARGS line).
#
# G1  THE IDENTITY GATE.  rho = 0 must reproduce the IN-BATCH scalar control:
#     |M(r0) - M(scal)| <= 0.45 pp at each alpha0.  This is the in-batch version
#     of the offline proof in tests/tl_equivalence.py, and it is the check that
#     the operator is what the patch says it is.  0.45 pp is 3.18 SE of that
#     difference (two-sided p 0.0015 under the identity).  A miss means the
#     operator does not reduce to the scalar arm and the batch is VOID.
#
# G2  THE MECHANISM GATE -- the gate that makes this G1 and not another r-ladder.
#     From each run's own probe.jsonl, END_SPREAD = std over groups of beta at
#     the LAST probe record, i.e. at ~50,000 steps, the very horizon at which
#     the archived lambda ladder had long since collapsed.  Registered:
#        (i)   END_SPREAD(r0) == 0 exactly, in every seed
#        (ii)  END_SPREAD strictly increasing over rho in {0.03, 0.1, 0.3, 1}
#        (iii) END_SPREAD(0.03) > 0
#     A violation means rho did not behave as a strength dial and the batch is
#     VOID.  Absent probes make it UNVERIFIED, which is carried as a suffix on
#     the verdict rather than silently ignored.  The calibration ratio
#     END_SPREAD(rho) / END_SPREAD(plain) against rho is printed but is
#     DESCRIPTIVE ONLY: the composite beta is clipped to the box, so the box can
#     legitimately compress the ratio at large rho, and a descriptive number must
#     not be allowed to void a batch.
#
# G3  THE TREE GATE.  The in-batch plain layerwise and scalar controls must land
#     within 0.60 pp of their archived grand means (90.893 / 91.196 layerwise and
#     87.880 / 87.839 scalar, at alpha0 1e-6 / 1e-3, re-derived by
#     priors_from_csv()).  0.60 pp is ~3.5x the SD of a new batch mean implied by
#     the measured between-batch SD (0.107-0.148) and the within-cell SE (0.100),
#     so it passes a legitimate batch effect and fails a moved tree.
#
# G4  THE RAIL METER.  Box occupancy from each run's own probe.jsonl.
#     DESCRIPTIVE ONLY -- it cannot change G5.
#
# G5  THE PRIMARY.  PEAK against SUPPORT_BAR and NULL_BAR, per alpha0.
#
# -----------------------------------------------------------------------------
# MAY NOT CLAIM
# -----------------------------------------------------------------------------
# Anything about ResNet-50, ResNet-10/34, CIFAR-100, Tiny-ImageNet, a
# granularity other than `layerwise`, a meta-stepsize other than 1e-3, a base or
# meta optimizer other than SGDm+Lion, a beta box other than -15:-2.3026, an
# epoch budget other than 100, or the `shrink`/`additive`/`zpool`/`zmpool`
# operators.  Five rho rungs cannot locate an optimum; they can only say whether
# an interior arm beats the endpoints by the registered margin.  A NULL here is
# a statement about THIS cell and about the two-level operator, and it does not
# retire the proposal at other alpha0, other architectures, or other scales.
# =============================================================================
import os
import re
import sys
import json
import math
import argparse
import statistics
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPT = os.path.join(REPO, "bin", "cG1_twolevel_ladder.sh")
CSV = os.path.join(REPO, "results", "all_runs.csv")
PATCH = os.path.join(REPO, "patches", "patch_twolevel.py")

# --------------------------------------------------------------------------
# THE REGISTERED DESIGN.  Asserted against the batch script's own text below.
# --------------------------------------------------------------------------
PREFIX = "tl1"
NET = "ResNet18"
DSET = "CIFAR10"
EPOCHS = 100
BATCH = 100
GAMMA = 1
CLIP = "-15:-2.3026"
AUG = "1"
PROBE = "500"
MS = "1e-3"
M_LAYERWISE = 62
WALL = "02:30:00"

ALPHA0S = ["1e-6", "1e-3"]
RHOS = ["0", "0.03", "0.1", "0.3", "1"]
SEEDS = ["0", "1", "2"]
# arm tag -> (granularity, HIER env value, ETA_RATIO env value)
ARMS = collections.OrderedDict([
    ("r0",    ("layerwise", "twolevel", "0")),
    ("r003",  ("layerwise", "twolevel", "0.03")),
    ("r01",   ("layerwise", "twolevel", "0.1")),
    ("r03",   ("layerwise", "twolevel", "0.3")),
    ("r1",    ("layerwise", "twolevel", "1")),
    ("plain", ("layerwise", "none",     "na")),
    ("scal",  ("scalar",    "none",     "na")),
])
RHO_ARM = {"r0": "0", "r003": "0.03", "r01": "0.1", "r03": "0.3", "r1": "1"}
ENDPOINT_ARMS = ("r0", "r1", "plain")
INTERIOR_ARMS = ("r003", "r01", "r03")
NJOBS = len(ARMS) * len(ALPHA0S) * len(SEEDS)          # 42

# --------------------------------------------------------------------------
# THE REGISTERED BARS AND PRIORS.  Every one is RE-DERIVED from the CSV at run
# time by the functions below; the literals are the registration, and selftest()
# fails if the re-derivation drifts away from them.
# --------------------------------------------------------------------------
SIGMA_W = 0.1735            # pooled within-cell SD of plateau5, df 153
SIGMA_TOL = 0.005
SUPPORT_BAR = 0.50          # PEAK >= this  -> partial shrinkage HELPS
NULL_BAR = 0.25             # PEAK <= this  -> no interior optimum
G1_BAND = 0.45              # rho=0 vs the in-batch scalar control, pp
G3_BAND = 0.60              # in-batch controls vs their archived grand means
PLAIN_PRIOR = {"1e-6": 90.893, "1e-3": 91.196}
SCALAR_PRIOR = {"1e-6": 87.880, "1e-3": 87.839}
PRIOR_TOL = 0.01
MIN_SEEDS = 2               # per arm, below which that alpha0 is VOID

VOID = "VOID"
UNRES = "UNRESOLVED"
SUPPORTED = "SUPPORTED"
NULLV = "NULL"

EP_RE = re.compile(r"Epoch\s+(\d+).*?Test Accuracy:\s*([0-9.]+)", re.I)
ARGS_RE = re.compile(r"^\s*ARGS:\s*(.*)$")
ENV_RE = re.compile(r"^\s*ENV:\s*(.*)$")


# --------------------------------------------------------------------------
# re-derivation from the corpus.  NOTHING here is quoted from prose.
# --------------------------------------------------------------------------
def _csv_rows(path):
    import csv
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if (r.get("superseded") or "0") != "1"]


def _batch_of(run):
    return re.split(r"[-_]", str(run))[0]


def _base_cell(r):
    """The archived configuration this batch replicates, on every axis."""
    try:
        p5 = float(r["plateau5"])
    except (TypeError, ValueError):
        return None
    if not (r["network"] == NET and r["dataset"] == DSET and r["base"] == "SGDm"
            and r["meta"] == "Lion" and r["meta_stepsize"] == MS
            and r["beta_clip"] == CLIP and r["augment"] == AUG
            and r["epochs_done"] == str(EPOCHS) and r["collapsed"] == "0"
            and r["batch_size"] == str(BATCH) and r["gamma"] == str(GAMMA)):
        return None
    return p5


def noise_floor_from_csv(path=CSV):
    """Pooled WITHIN-CELL SD of plateau5.  Cells: the archived additive-layerwise
    (batch x alpha0 x eta_ratio) cells and the archived plain layerwise/scalar
    (batch x alpha0 x granularity) cells at this configuration."""
    cells = collections.defaultdict(list)
    for r in _csv_rows(path):
        p5 = _base_cell(r)
        if p5 is None:
            continue
        h = (r.get("hier") or "").strip()
        if h == "additive" and r["granularity"] == "layerwise":
            cells[("add", _batch_of(r["run"]), r["alpha0"], r["eta_ratio"])].append(p5)
        elif h == "" and r["granularity"] in ("layerwise", "scalar"):
            cells[("plain", _batch_of(r["run"]), r["alpha0"], r["granularity"])].append(p5)
    ss = 0.0
    df = 0
    n = 0
    for v in cells.values():
        if len(v) >= 2:
            m = statistics.mean(v)
            ss += sum((x - m) ** 2 for x in v)
            df += len(v) - 1
            n += 1
    return (math.sqrt(ss / df) if df else float("nan")), df, n


def priors_from_csv(path=CSV):
    """Archived grand means of the plain layerwise and scalar arms, per alpha0,
    with the number of distinct BATCHES each rests on."""
    out = {}
    acc = collections.defaultdict(list)
    for r in _csv_rows(path):
        p5 = _base_cell(r)
        if p5 is None or (r.get("hier") or "").strip():
            continue
        if r["granularity"] in ("layerwise", "scalar") and r["alpha0"] in ALPHA0S:
            acc[(r["granularity"], r["alpha0"])].append((p5, _batch_of(r["run"])))
    for k, v in acc.items():
        out[k] = (statistics.mean(x for x, _ in v), len(v), len(set(b for _, b in v)))
    return out


def premise_no_twolevel_runs(path=CSV):
    """The corpus must contain NO twolevel run before this batch: the operator
    did not exist.  Returns the count."""
    return sum(1 for r in _csv_rows(path) if (r.get("hier") or "").strip() == "twolevel")


def shrink_halflives(lam, steps=None):
    """The arithmetic the whole gap rests on: half-life of the per-step shrink,
    in optimizer steps, and how many of them a run contains.  50,000 CIFAR-10
    images / batch 100 = 500 steps per epoch."""
    if steps is None:
        steps = 500 * EPOCHS
    hl = math.log(2.0) / -math.log(1.0 - lam) if 0 < lam < 1 else 0.0
    return hl, (steps / hl if hl > 0 else float("inf"))


# --------------------------------------------------------------------------
# reading the runs.  The arm of a run is MEASURED from its own ARGS/ENV.
# --------------------------------------------------------------------------
def _tokens(line):
    import shlex
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def parse_args_line(line):
    """(effective_flags, repeated_flags).  argparse keeps the LAST occurrence."""
    toks = _tokens(line)
    occ = []
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        if t.startswith("--") and len(t) > 2:
            if "=" in t:
                f, v = t.split("=", 1)
                occ.append((f.lstrip("-"), v))
                i += 1
                continue
            vals = []
            j = i + 1
            while j < n and not (toks[j].startswith("--") and len(toks[j]) > 2):
                vals.append(toks[j])
                j += 1
            occ.append((t.lstrip("-"), " ".join(vals)))
            i = j
            continue
        i += 1
    eff = collections.OrderedDict()
    seen = collections.OrderedDict()
    for f, v in occ:
        eff[f] = v
        seen.setdefault(f, []).append(v)
    return eff, dict((f, v) for f, v in seen.items() if len(v) > 1)


def parse_env_line(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def read_out(path):
    args = env = None
    ser = {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                if args is None:
                    m = ARGS_RE.match(line)
                    if m:
                        args = m.group(1).strip()
                        continue
                if env is None:
                    m = ENV_RE.match(line)
                    if m:
                        env = m.group(1).strip()
                        continue
                m = EP_RE.search(line)
                if m:
                    ser[int(m.group(1))] = float(m.group(2))
    except IOError:
        return None
    if args is None:
        return None
    eff, dup = parse_args_line(args)
    return {"args": eff, "dup": dup, "argsline": args,
            "env": parse_env_line(env or ""), "series": ser, "path": path}


def plateau5(series, budget=EPOCHS, w=5):
    """PRIMARY.  Mean of the LAST 5 test epochs; epochs are 0-indexed."""
    v = [series[e] for e in range(budget - w, budget) if e in series]
    return sum(v) / len(v) if len(v) == w else None


def norm_num(s):
    try:
        return "%g" % float(s)
    except (TypeError, ValueError):
        return None


def gate0(recs):
    """PROVENANCE.  Returns (cells, dropped); cells maps (alpha0, arm, seed) to
    plateau5.  Every check is against the run's OWN ARGS/ENV line."""
    want_args = {"NN-name": NET, "dataset": DSET, "num-epochs": str(EPOCHS),
                 "batch-size": str(BATCH), "gamma": str(GAMMA),
                 "alg-base": "SGDm", "alg-meta": "Lion", "meta-stepsize": MS}
    want_env = {"AUGMENT": AUG, "BETA_CLIP": CLIP, "PROBE": PROBE, "SCHED": "none"}
    cells, dropped = {}, []
    for rec in recs:
        why = []
        if rec["dup"]:
            why.append("REPEATED FLAG %s (RULE 20)" % ",".join(sorted(rec["dup"])))
        for f, w in want_args.items():
            got = rec["args"].get(f)
            if got is None:
                why.append("--%s absent" % f)
            elif norm_num(w) is not None and norm_num(got) is not None:
                if norm_num(got) != norm_num(w):
                    why.append("--%s=%s want %s" % (f, got, w))
            elif got != w:
                why.append("--%s=%s want %s" % (f, got, w))
        for k, w in want_env.items():
            if rec["env"].get(k) != w:
                why.append("ENV %s=%s want %s" % (k, rec["env"].get(k), w))
        a0 = norm_num(rec["args"].get("alpha0"))
        gran = rec["args"].get("stepsize-groups")
        hier = rec["env"].get("HIER")
        er = rec["env"].get("ETA_RATIO")
        sd = rec["args"].get("seed")
        # the arm is IDENTIFIED by (granularity, HIER, ETA_RATIO), never by name
        arm = None
        for tag, (g, h, r) in ARMS.items():
            if gran == g and hier == h and (
                    r == "na" and er in ("na", None) or
                    r != "na" and norm_num(er) == norm_num(r)):
                arm = tag
                break
        if arm is None:
            why.append("(stepsize-groups=%s, HIER=%s, ETA_RATIO=%s) is not a "
                       "registered arm" % (gran, hier, er))
        if a0 not in [norm_num(x) for x in ALPHA0S]:
            why.append("--alpha0=%s not a registered rung" % rec["args"].get("alpha0"))
        if sd not in SEEDS:
            why.append("--seed=%s not a registered seed" % sd)
        p5 = plateau5(rec["series"])
        if p5 is None:
            why.append("fewer than %d completed epochs (max ep %s)"
                       % (EPOCHS, max(rec["series"]) if rec["series"] else "none"))
        rn = rec["args"].get("run-name", "")
        if rn and not os.path.basename(rec["path"]).startswith(rn + "-"):
            why.append("filename disagrees with --run-name %s" % rn)
        if why:
            dropped.append((os.path.basename(rec["path"]), why))
            continue
        cells[(a0, arm, sd)] = p5
    return cells, dropped


def arm_means(cells):
    by = collections.defaultdict(list)
    for (a0, arm, _s), v in cells.items():
        by[(a0, arm)].append(v)
    return dict((k, (statistics.mean(v), len(v), sorted(v))) for k, v in by.items())


# --------------------------------------------------------------------------
# the probe: G2 (mechanism, gating) and G4 (rails, descriptive)
# --------------------------------------------------------------------------
def _last_probe_record(probe_dir):
    p = os.path.join(probe_dir, "probe.jsonl")
    if not os.path.exists(p):
        return None
    last = None
    try:
        with open(p, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except ValueError:
                    continue
    except IOError:
        return None
    return last


def end_spread(probe_dir):
    """std over groups of beta at the LAST probe record, or None."""
    rec = _last_probe_record(probe_dir)
    if rec is None or not isinstance(rec.get("beta"), list) or len(rec["beta"]) < 2:
        return None
    b = rec["beta"]
    m = sum(b) / len(b)
    return math.sqrt(sum((x - m) ** 2 for x in b) / (len(b) - 1)), int(rec.get("step", -1))


def rails(probe_dir):
    """(frac at the low rail, frac at the high rail) at the last record."""
    rec = _last_probe_record(probe_dir)
    if rec is None:
        return None
    n = rec.get("n_beta")
    if not n:
        return None
    return (rec.get("n_at_lo", 0) / float(n), rec.get("n_at_hi", 0) / float(n))


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------
def verdict(peak):
    if peak >= SUPPORT_BAR:
        return SUPPORTED
    if peak <= NULL_BAR:
        return NULLV
    return UNRES


# --------------------------------------------------------------------------
def selftest():
    fails = []

    def ok(label, cond, extra=""):
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", label,
                               ("  " + extra) if extra else ""))
        if not cond:
            fails.append(label)

    print("--- registration: the design matches the batch script's own text ---")
    txt = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""
    ok("bin/cG1_twolevel_ladder.sh exists", bool(txt))
    if txt:
        for lit in ('PREFIX=%s' % PREFIX, 'NET=%s' % NET, 'DSET=%s' % DSET,
                    'ALPHA0S="%s"' % " ".join(ALPHA0S),
                    'RHOS="%s"' % " ".join(RHOS),
                    'SEEDS="%s"' % " ".join(SEEDS),
                    'EPOCHS=%d' % EPOCHS, 'BATCH=%d' % BATCH, 'GAMMA=%d' % GAMMA,
                    'CLIP="%s"' % CLIP, 'AUG=%s' % AUG, 'PROBE=%s' % PROBE,
                    'MS=%s' % MS, 'NJOBS=%d' % NJOBS, 'WALL=%s' % WALL,
                    'SIGMA_W=%.4f' % SIGMA_W, 'SUPPORT_BAR=%.2f' % SUPPORT_BAR,
                    'NULL_BAR=%.2f' % NULL_BAR, 'G1_BAND=%.2f' % G1_BAND,
                    'G3_BAND=%.2f' % G3_BAND, 'M_LAYERWISE=%d' % M_LAYERWISE):
            ok("script declares %s" % lit, lit in txt)

    print("--- the operator: PATCH_TWOLEVEL says what this scorer says it says ---")
    ptxt = open(PATCH).read() if os.path.exists(PATCH) else ""
    ok("patches/patch_twolevel.py exists", bool(ptxt))
    if ptxt:
        ok("the dial is ETA_RATIO", "self._hier_ratio * _eta" in ptxt)
        ok("the mode string is 'twolevel'", "self._hier == 'twolevel'" in ptxt)
        ok("the shared level takes sum_g z_g",
           "_zs = sum(_zi.sum() for _zi in z)" in ptxt)
        ok("the composite is s + d", "self.beta[_i] = self._tl_s + self._tl_d[_i]" in ptxt)
        ok("the disabled path keeps the pinned two lines",
           "self.meta_update(HtT_gradft)\n                if self._hier:  # PATCH_HIER"
           in ptxt)

    print("--- the arithmetic that makes lam a time constant and rho a strength ---")
    for lam, want_hl, want_n in ((1e-3, 692.8, 72.2), (1e-2, 69.0, 725.4), (1e-1, 6.6, 7602.0)):
        hl, nhl = shrink_halflives(lam)
        ok("lam=%g -> half-life %.1f steps, %.1f half-lives in a 100-epoch run"
           % (lam, hl, nhl),
           abs(hl - want_hl) < 0.6 and abs(nhl - want_n) / want_n < 0.02)
    ok("a 100-epoch CIFAR-10 run at batch 100 is 50,000 optimizer steps",
       500 * EPOCHS == 50000)

    print("--- the noise floor and the priors, RE-DERIVED from the corpus ---")
    if os.path.exists(CSV):
        s, df, ncell = noise_floor_from_csv()
        ok("SIGMA_W re-derives to %.4f (registered %.4f, df %d, %d cells)"
           % (s, SIGMA_W, df, ncell), abs(s - SIGMA_W) < SIGMA_TOL)
        pri = priors_from_csv()
        for a0 in ALPHA0S:
            m, n, nb = pri[("layerwise", a0)]
            ok("plain layerwise a0=%s re-derives to %.3f (registered %.3f, n=%d, %d batches)"
               % (a0, m, PLAIN_PRIOR[a0], n, nb), abs(m - PLAIN_PRIOR[a0]) < PRIOR_TOL)
            m, n, nb = pri[("scalar", a0)]
            ok("scalar a0=%s re-derives to %.3f (registered %.3f, n=%d, %d batches)"
               % (a0, m, SCALAR_PRIOR[a0], n, nb), abs(m - SCALAR_PRIOR[a0]) < PRIOR_TOL)
        nt = premise_no_twolevel_runs()
        ok("the corpus holds NO twolevel run yet (found %d) -- RULE 21" % nt, nt == 0)
    else:
        ok("results/all_runs.csv present", False)

    print("--- the bars, re-simulated from the registered SIGMA_W ---")
    try:
        import numpy as np
        rng = np.random.default_rng(20260903)
        se = SIGMA_W / math.sqrt(len(SEEDS))
        NS = 200000

        def peaks(mu):
            x = rng.normal(mu, se, size=(NS, 6))
            return x[:, 1:4].max(1) - x[:, [0, 4, 5]].max(1)

        p0 = peaks(np.zeros(6))
        fp = float((p0 >= SUPPORT_BAR).mean())
        p99 = float(np.percentile(p0, 99))
        ok("flat-null P(PEAK >= SUPPORT_BAR=%.2f) = %.5f (<= 0.001)" % (SUPPORT_BAR, fp),
           fp <= 0.001)
        ok("NULL_BAR=%.2f is the flat null's 99th percentile (%.3f)" % (NULL_BAR, p99),
           abs(p99 - NULL_BAR) < 0.03)
        mu = np.zeros(6)
        mu[2] = 0.997                      # the archived interior lift at alpha0=1e-6
        p1 = peaks(mu)
        pw = float((p1 >= SUPPORT_BAR).mean())
        fn = float((p1 <= NULL_BAR).mean())
        ok("power against the archived +0.997 pp lift = %.3f (>= 0.99)" % pw, pw >= 0.99)
        ok("P(NULL | that lift) = %.5f (<= 0.001)" % fn, fn <= 0.001)
    except ImportError:
        print("  [SKIP] numpy unavailable; the bar simulation was not re-run")

    print("--- the parsers ---")
    line = ("--optimizer HF --alg-meta RMSProp --alg-meta Lion --alpha0 1e-3 "
            "--stepsize-groups layerwise --seed 2")
    eff, dup = parse_args_line(line)
    ok("a repeated flag is detected", "alg-meta" in dup)
    ok("and argparse's effective value is reported", eff["alg-meta"] == "Lion")
    ok("ENV parses", parse_env_line("HIER=twolevel ETA_RATIO=0.1")["ETA_RATIO"] == "0.1")
    ok("plateau5 is the mean of the LAST 5 epochs",
       abs(plateau5(dict((e, float(e)) for e in range(EPOCHS))) - 97.0) < 1e-9)
    ok("plateau5 refuses a truncated run",
       plateau5(dict((e, 1.0) for e in range(EPOCHS - 3))) is None)

    print("--- the verdict function ---")
    ok("PEAK 0.60 -> SUPPORTED", verdict(0.60) == SUPPORTED)
    ok("PEAK 0.50 -> SUPPORTED (bar is inclusive)", verdict(0.50) == SUPPORTED)
    ok("PEAK 0.35 -> UNRESOLVED", verdict(0.35) == UNRES)
    ok("PEAK 0.25 -> NULL (bar is inclusive)", verdict(0.25) == NULLV)
    ok("PEAK -1.0 -> NULL", verdict(-1.0) == NULLV)
    ok("the two bars do not overlap", NULL_BAR < SUPPORT_BAR)

    print("\nSELFTEST: %s (%d checks failed)"
          % ("PASS" if not fails else "FAIL", len(fails)))
    return 0 if not fails else 1


# --------------------------------------------------------------------------
def score(runs_dir, probes_dir):
    import glob
    paths = sorted(glob.glob(os.path.join(runs_dir, "%s-*.out" % PREFIX)))
    print("=" * 78)
    print(" cG1_tl_score.py -- `%s`, the TWO-LEVEL shrinkage ladder (gap G1)" % PREFIX)
    print(" plateau5 (mean of the LAST 5 test epochs) is PRIMARY throughout.")
    print(" bars: SUPPORTED PEAK>=%.2f   NULL PEAK<=%.2f   else UNRESOLVED"
          % (SUPPORT_BAR, NULL_BAR))
    print("=" * 78)
    print("found %d %s-*.out under %s" % (len(paths), PREFIX, runs_dir))

    recs = [r for r in (read_out(p) for p in paths) if r]
    cells, dropped = gate0(recs)
    print("\n--- G0  PROVENANCE (read from each run's OWN ARGS/ENV line) ---")
    print("  %d/%d runs carry a design that matches the registration" % (len(cells), len(paths)))
    for name, why in dropped:
        print("  DROPPED %-34s %s" % (name, "; ".join(why)))
    if not cells:
        print("\nVERDICT: %s -- no run survives G0" % VOID)
        return 2

    means = arm_means(cells)
    print("\n--- the arms, plateau5, per alpha0 ---")
    for a0 in ALPHA0S:
        print("  alpha0 = %s" % a0)
        for arm in ARMS:
            m = means.get((norm_num(a0), arm))
            if m is None:
                print("    %-6s   MISSING" % arm)
            else:
                print("    %-6s   %7.3f   n=%d   %s"
                      % (arm, m[0], m[1], " ".join("%.3f" % v for v in m[2])))

    # ---- G1 the identity gate ------------------------------------------------
    print("\n--- G1  THE IDENTITY GATE: rho=0 must BE the in-batch scalar arm ---")
    g1_ok = True
    for a0 in ALPHA0S:
        a = means.get((norm_num(a0), "r0"))
        b = means.get((norm_num(a0), "scal"))
        if a is None or b is None or a[1] < MIN_SEEDS or b[1] < MIN_SEEDS:
            print("  alpha0=%s  UNVERIFIABLE (r0 or scal under %d seeds)" % (a0, MIN_SEEDS))
            g1_ok = False
            continue
        d = a[0] - b[0]
        good = abs(d) <= G1_BAND
        g1_ok = g1_ok and good
        print("  alpha0=%s  M(r0)=%7.3f  M(scal)=%7.3f  delta=%+6.3f  band +-%.2f  [%s]"
              % (a0, a[0], b[0], d, G1_BAND, "PASS" if good else "FAIL"))

    # ---- G2 the mechanism gate ----------------------------------------------
    print("\n--- G2  THE MECHANISM GATE: std(beta) at the END of the run ---")
    print("  registered: spread(r0)==0 exactly; spread strictly increasing over")
    print("  rho in {0.03,0.1,0.3,1}; spread(0.03)>0.  Calibration is DESCRIPTIVE.")
    g2_state = "UNVERIFIED"
    spreads = {}
    if probes_dir:
        for a0 in ALPHA0S:
            for arm in ARMS:
                vals = []
                for s in SEEDS:
                    d = os.path.join(probes_dir, "probe_%s_%s_%s_s%s"
                                     % (PREFIX, _a0tag(a0), arm, s))
                    e = end_spread(d)
                    if e is not None:
                        vals.append(e)
                if vals:
                    spreads[(a0, arm)] = (statistics.mean(v for v, _ in vals),
                                          len(vals), max(t for _, t in vals),
                                          [v for v, _ in vals])
    if spreads:
        g2_state = "PASS"
        for a0 in ALPHA0S:
            print("  alpha0 = %s" % a0)
            base = spreads.get((a0, "plain"))
            for arm in ARMS:
                v = spreads.get((a0, arm))
                if v is None:
                    print("    %-6s   no probe" % arm)
                    continue
                cal = ("  cal=%.4f (rho=%s)" % (v[0] / base[0], RHO_ARM[arm])
                       if base and base[0] > 0 and arm in RHO_ARM else "")
                print("    %-6s   std(beta)=%.6f  n=%d  last step %d%s"
                      % (arm, v[0], v[1], v[2], cal))
            checks = []
            r0 = spreads.get((a0, "r0"))
            if r0 is not None:
                checks.append(("spread(r0) == 0 exactly", all(x == 0.0 for x in r0[3])))
            seq = [spreads.get((a0, a)) for a in ("r003", "r01", "r03", "r1")]
            if all(x is not None for x in seq):
                checks.append(("spread increasing over rho",
                               all(seq[i][0] < seq[i + 1][0] for i in range(3))))
                checks.append(("spread(rho=0.03) > 0", seq[0][0] > 0.0))
            for lab, good in checks:
                print("    [%s] %s" % ("PASS" if good else "FAIL", lab))
                if not good:
                    g2_state = "FAIL"
            if not checks:
                g2_state = "UNVERIFIED"
    else:
        print("  no probe.jsonl found under %s -- G2 is UNVERIFIED" % probes_dir)

    # ---- G3 the tree gate ----------------------------------------------------
    print("\n--- G3  THE TREE GATE: the in-batch controls vs their archived means ---")
    g3_ok = True
    for a0 in ALPHA0S:
        for arm, prior in (("plain", PLAIN_PRIOR), ("scal", SCALAR_PRIOR)):
            m = means.get((norm_num(a0), arm))
            if m is None or m[1] < MIN_SEEDS:
                print("  alpha0=%s %-6s UNVERIFIABLE" % (a0, arm))
                g3_ok = False
                continue
            d = m[0] - prior[a0]
            good = abs(d) <= G3_BAND
            g3_ok = g3_ok and good
            print("  alpha0=%s %-6s  %7.3f  archived %7.3f  delta=%+6.3f  band +-%.2f  [%s]"
                  % (a0, arm, m[0], prior[a0], d, G3_BAND, "PASS" if good else "FAIL"))

    # ---- G4 rails, descriptive ----------------------------------------------
    print("\n--- G4  THE RAIL METER (descriptive only) ---")
    if probes_dir:
        for a0 in ALPHA0S:
            for arm in ARMS:
                fr = [rails(os.path.join(probes_dir, "probe_%s_%s_%s_s%s"
                                         % (PREFIX, _a0tag(a0), arm, s)))
                      for s in SEEDS]
                fr = [x for x in fr if x]
                if fr:
                    print("  alpha0=%s %-6s  lo %.3f  hi %.3f"
                          % (a0, arm, statistics.mean(x[0] for x in fr),
                             statistics.mean(x[1] for x in fr)))
    else:
        print("  no --probes given")

    # ---- G5 the primary ------------------------------------------------------
    print("\n--- G5  THE PRIMARY.  PEAK = max(interior) - max(endpoints) ---")
    verdicts = {}
    for a0 in ALPHA0S:
        got = dict((arm, means.get((norm_num(a0), arm))) for arm in ARMS)
        short = [a for a in ARMS if got[a] is None or got[a][1] < MIN_SEEDS]
        if short:
            print("  alpha0=%s  VOID -- arms under %d seeds: %s"
                  % (a0, MIN_SEEDS, ",".join(short)))
            verdicts[a0] = VOID
            continue
        mi = max((got[a][0], a) for a in INTERIOR_ARMS)
        me = max((got[a][0], a) for a in ENDPOINT_ARMS)
        peak = mi[0] - me[0]
        rho_vals = [got[a][0] for a in RHO_ARM]
        span = max(rho_vals) - min(rho_vals)
        v = verdict(peak)
        verdicts[a0] = v
        print("  alpha0=%s  best interior %s=%.3f   best endpoint %s=%.3f"
              % (a0, mi[1], mi[0], me[1], me[0]))
        print("            PEAK=%+.3f   SPAN(rho ladder)=%.3f   -> %s"
              % (peak, span, v))

    if not g1_ok:
        head = "%s-IDENTITY" % VOID
    elif not g3_ok:
        head = "%s-TREE" % VOID
    elif g2_state == "FAIL":
        head = "%s-MECHANISM" % VOID
    else:
        head = "  |  ".join("alpha0=%s: %s" % (a0, verdicts[a0]) for a0 in ALPHA0S)
        if g2_state == "UNVERIFIED":
            head += "   [MECHANISM UNVERIFIED]"

    print("\n" + "=" * 78)
    print("VERDICT: %s" % head)
    print("G1 identity %s | G2 mechanism %s | G3 tree %s"
          % ("PASS" if g1_ok else "FAIL", g2_state, "PASS" if g3_ok else "FAIL"))
    print("The two alpha0 are co-primary and are NOT pooled.  No claim is licensed")
    print("beyond ResNet18 / CIFAR-10 / layerwise / SGDm+Lion / ms 1e-3 / 100 ep /")
    print("box %s.  Five rho rungs cannot locate an optimum." % CLIP)
    print("=" * 78)
    return 0


def _a0tag(a0):
    return {"1e-6": "a6", "1e-3": "a3"}[a0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=".", help="directory holding tl1-*.out files")
    ap.add_argument("--probes", default=None, help="directory holding probe_tl1_* dirs")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    return score(a.runs, a.probes)


if __name__ == "__main__":
    sys.exit(main())
