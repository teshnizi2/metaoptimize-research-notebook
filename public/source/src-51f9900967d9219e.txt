#!/usr/bin/env python3
# =============================================================================
# cwd2_attack_indep.py -- THE INDEPENDENT PARSER OF `cwd2` (CORRECTIONS 261, landing entry).
#
# It re-derives, FROM THE RAW RECORDS ALONE, every line the REGISTERED scorer
# analysis/cWD2_carrierwd_score.py prints, and compares its rebuild with a committed scorer log line by line, in order.
#
#   python3 analysis/cwd2_attack_indep.py <runsdir> <scorer-log> [--csv P] [--tsv P] [--replay P]
#
# INDEPENDENCE, deliberately:
#   * it imports NOTHING from cWD2_carrierwd_score.py, cwd_design.py, cwd_common.py, cVT9_dosewindow_score.py,
#     corpus_exclusions.py, argsline_guard.py or any other repo module.  Stdlib only: hashlib, json, math, os, struct,
#     sys.  csv is NOT used either -- the corpus is split by hand.
#   * NO regular expressions anywhere.  The `.out` lines are split on commas / colons / whitespace, the TSV on tabs,
#     the file names on hyphens.
#   * every bar, sigma literal, witness string, tree sha, licence sentence and stamp name below is RE-TYPED from
#     CORRECTIONS 261 (and, for the hold strings it inherits, from 249 / 253), not read from any module.
#   * the PlainNet18_c100 tensor table is rebuilt from the architecture, not read from a manifest.
#   * float32 rounding is re-derived through struct, and the Lion natural step is recomputed from the records' own
#     beta_pre / mom_pre / z_agg, so G-BITE's every printed counter is independent of the scorer's implementation.
#
# HOST INDEPENDENCE.  Two kinds of line in the scorer's output are host-dependent and are matched, not rebuilt
# byte for byte: (1) the two path disclosures (`manifest:` / `provenance:`), matched by their host-independent
# suffix; (2) the corpus disclosure, whose row count depends on which corpus commit the host's tree carries -- this
# parser re-filters the corpus IT IS GIVEN and requires the log's count to equal its own, without printing either
# number, so that THIS parser's own stdout is byte-identical on both hosts.  Everything else is rebuilt verbatim.
# =============================================================================

import hashlib
import json
import math
import os
import struct
import sys

# --------------------------------------------------------------------------------------------------------------------
# RE-TYPED LITERALS (CORRECTIONS 261; the hold strings from 249 / 253, the cell from 230 / 242)
# --------------------------------------------------------------------------------------------------------------------
PREFIX = "cwd2"
NET = "PlainNet18_c100"
DSET = "CIFAR100"
EPOCHS = 100
BATCH = 100
CLIP = "-15:-2.3026"
MST = "1e-3"
A0 = "1e-6"
AUG = "1"
PROBE = 100
STEPS_PER_EPOCH = 50000 // BATCH
N_RECORDS = EPOCHS * STEPS_PER_EPOCH // PROBE            # 500
WD_BASE = 0.1
SEEDS = (132, 133, 134)
ARMS = ("k01", "k01WD0", "HIGHHEADPATH", "HIGHWD0", "LOWWD0")
CTL = "LOWWD0"
MASKED = ("k01WD0", "HIGHWD0", "LOWWD0")
CARRIER = "layer4.1.bn2.weight"
HEADSPEC = "sets:1-49,51-53/layer4.1.bn2.weight"
SPEC = {"k01": "scalar", "k01WD0": "scalar", "HIGHHEADPATH": HEADSPEC, "HIGHWD0": HEADSPEC, "LOWWD0": HEADSPEC}
PT_TYPE = {"k01": "scalar", "k01WD0": "scalar", "HIGHHEADPATH": "blockwise", "HIGHWD0": "blockwise",
           "LOWWD0": "blockwise"}
DMASK = {"k01": "", "k01WD0": CARRIER, "HIGHHEADPATH": "", "HIGHWD0": CARRIER, "LOWWD0": CARRIER}
AS_CVT9 = {"k01": "k01", "k01WD0": "k01", "HIGHHEADPATH": "HIGHHEADPATH", "HIGHWD0": "HIGHHEADPATH",
           "LOWWD0": "LOWHEADPATH"}
HELD_ARMS = ("HIGHHEADPATH", "HIGHWD0", "LOWWD0")

NTENS = 53
TOTPAR = 11046308
IDX_HEAD_1BASED = 50                                     # the DECAY_MASK / BETA_HOLD witnesses print 1-based indices
P_HIGH = 9428
MS = 1e-3
B2 = 0.9
LO, HI = -15.0, -2.3026
BH_TOL = 1e-5
TIE_REL = 1e-12
HEADPATH_ID = "cvt6_headpath"
HEADPATH_SHA = "74be71fa524ad0122d1408e01dd2b633b004b2e27228393fe6e593f494358a5d"
HEADPATH_KNOTS = 500
HEADPATH_N0, HEADPATH_N1 = 2, 49902
HEADPATH_VMAX = -4.852388381958008
HEADPATH_VLAST = -15.0

ENV_EXPECTED = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
                "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 "
                "EB_RHO=na EB_LOG=0")

# provenance literals (261.10)
BN_SHA = "e65e67738e45418045ccc7f737fb0ea5b2523a9294a4c3a288e543e95508a18b"
HF_POST_SHA = "133c12296b4b9e4428cfe8ba882413c85fc0ec7e5372f3043e6aaa99b623a4b4"
HF_PARENT_SHA = "816e335777f7c0bb52747949dff81aa57fe0695bab2d61d984ae6ce9daab3a9d"
RUNNER_SHA = "f3dfa97fb93e3d749aab156ca2562dd24e9b97e77664cb1537ad1c16a6df73ef"
SCORER_SHA = "83a2d00efe444ff5429907b6da84140072ade126b36146990961b5c18b2d7f8c"
DESIGN_SHA = "03d2d7614915757798699ff75ca971a11fca2056d383649fecbbf5adf06c0195"
COMMON_SHA = "d7ddb49e71013e3a21eaf5129861ab43dbc16e181bbd92af7c70cdc95e82bf52"

# frozen noise floor and bars (261.4 / 261.5)
SIGMA_PRIOR = 0.6811983409351905
GAP_MIN = 20.0
K01_MAX = 20.0
RESCUE_BAR = 10.0
NULL_BAR = 2.0
MATCH_BAR = 5.0
DIVERGED_BAR = 5.0
FLOOR_MIN = 15.0
CEIL_MAX = 90.0
CHANCE = 1.0
R_BETWEEN_ORDER = ("cvt9 (253)", "cvt6 (246)")
R_BETWEEN = {"cvt9 (253)": (("k01", 12.1347), ("LOWHEADPATH", 65.2080), ("HIGHHEADPATH", 11.2107)),
             "cvt6 (246)": (("k01", 12.0700), ("LOWHEADPATH", 65.0207), ("HIGHHEADPATH", 11.4933))}


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


B0 = f32(math.log(1e-6))


# --------------------------------------------------------------------------------------------------------------------
# the architecture, rebuilt (no manifest is read)
# --------------------------------------------------------------------------------------------------------------------
def plain18_tensors():
    """[(name, numel, owner, ndim)] in parameter order for PlainNet18_c100 (ResNet18 with the 9 shortcut tensors gone)."""
    t = [("conv1.weight", 64 * 3 * 9, "Conv2d"), ("bn1.weight", 64, "BatchNorm2d"), ("bn1.bias", 64, "BatchNorm2d")]
    inp = 64
    for li, w in enumerate((64, 128, 256, 512), 1):
        for b in (0, 1):
            cin = inp if b == 0 else w
            pre = "layer%d.%d." % (li, b)
            t += [(pre + "conv1.weight", w * cin * 9, "Conv2d"), (pre + "bn1.weight", w, "BatchNorm2d"),
                  (pre + "bn1.bias", w, "BatchNorm2d"), (pre + "conv2.weight", w * w * 9, "Conv2d"),
                  (pre + "bn2.weight", w, "BatchNorm2d"), (pre + "bn2.bias", w, "BatchNorm2d")]
        inp = w
    t += [("linear.weight", 512 * 100, "Linear"), ("linear.bias", 100, "Linear")]
    out = []
    for n, q, o in t:
        nd = 1 if (o in ("BatchNorm2d", "GroupNorm") or n.endswith(".bias")) else (4 if o == "Conv2d" else 2)
        out.append((n, q, o, nd))
    return out


TENSORS = plain18_tensors()
NAMES = [x[0] for x in TENSORS]


def dm_indices(spec):
    if not spec:
        return []
    if spec == "normscale":
        return sorted(i for i, t in enumerate(TENSORS) if t[0].endswith(".weight") and t[3] == 1)
    return sorted(NAMES.index(x) for x in spec.split("+"))


def dm_witness(spec):
    if not spec:
        return "DECAY_MASK: off"
    idx = dm_indices(spec)
    return ("DECAY_MASK: on base=SGDm wd=%r spec=%s masked=%d of=%d numel=%d idx=%s names=%s"
            % (WD_BASE, spec, len(idx), len(TENSORS), sum(TENSORS[i][1] for i in idx),
               ",".join("%d" % (i + 1) for i in idx), ",".join(TENSORS[i][0] for i in idx)))


def u_of(n, P):
    if n <= 0:
        return 0
    return min(n - 1, P) - max(0, n - 1 - P)


def v_of(n, P):
    return f32(min(HI, max(LO, B0 + MS * u_of(n, P))))


def witness_bh(arm9):
    if arm9 == "k01":
        return "BETA_HOLD: off"
    d = "BETA_HOLD: on type=blockwise group=1 groupsize=1 name=%s" % CARRIER
    if arm9 == "LOWHEADPATH":
        return d + " mode=floor value=%r" % (LO,)
    return d + " mode=tri P=%d b0=%r ms=%r lo=%r hi=%r peak=%r" % (P_HIGH, B0, MS, LO, HI,
                                                                   min(HI, max(LO, B0 + MS * P_HIGH)))


def witness_ch(arm9):
    if arm9 == "k01":
        return "COMP_HOLD: off"
    return ("COMP_HOLD: on type=blockwise group=0 groupsize=52 mode=rec id=%s sha256=%s knots=%d n0=%d n1=%d b0=%r "
            "lo=%r hi=%r vmax=%r vlast=%r" % (HEADPATH_ID, HEADPATH_SHA, HEADPATH_KNOTS, HEADPATH_N0, HEADPATH_N1,
                                              B0, LO, HI, HEADPATH_VMAX, HEADPATH_VLAST))


BETAHOLD_SHORT = {"k01": "", "LOWHEADPATH": "%s:floor" % CARRIER, "HIGHHEADPATH": "%s:tri:%d" % (CARRIER, P_HIGH)}
COMPHOLD_SHORT = {"k01": "", "LOWHEADPATH": "rec:%s" % HEADPATH_ID, "HIGHHEADPATH": "rec:%s" % HEADPATH_ID}
WITNESS_VW = "VOTE_W: off"
WITNESS_WH = "WINDOW_HOLD: off"
WITNESS_BH = dict((a, witness_bh(AS_CVT9[a])) for a in ARMS)
WITNESS_CH = dict((a, witness_ch(AS_CVT9[a])) for a in ARMS)
WITNESS_DM = dict((a, dm_witness(DMASK[a])) for a in ARMS)
BETAHOLD = dict((a, BETAHOLD_SHORT[AS_CVT9[a]]) for a in ARMS)
COMPHOLD = dict((a, COMPHOLD_SHORT[AS_CVT9[a]]) for a in ARMS)
K_MASKED = dict((a, len(dm_indices(DMASK[a]))) for a in ARMS)


def manifest_text():
    lines = ["NETWORK %s" % NET, "NUM_PARAM_TENSORS %d" % len(TENSORS),
             "TOTAL_PARAMS %d" % sum(x[1] for x in TENSORS), "CLIP_C %s" % CLIP, "EPOCHS %d" % EPOCHS,
             "META_STEPS %d" % (EPOCHS * STEPS_PER_EPOCH), "WD_BASE %r" % WD_BASE]
    for i, (n, q, o, nd) in enumerate(TENSORS, 1):
        lines.append("TENSOR %d %s %d %s ndim=%d" % (i, n, q, o, nd))
    lines.append("NORMSCALE %s" % ",".join("%d:%s" % (i + 1, TENSORS[i][0]) for i in dm_indices("normscale")))
    for a in ARMS:
        lines.append("ARMSPEC %s SPEC %s TYPE %s" % (a, SPEC[a], PT_TYPE[a]))
        lines.append("DECAYMASK %s %s" % (a, DMASK[a] or "off"))
        lines.append("DWITNESS %s %s" % (a, WITNESS_DM[a]))
        lines.append("BETAHOLD %s %s" % (a, BETAHOLD[a] or "off"))
        lines.append("COMPHOLD %s %s" % (a, COMPHOLD[a] or "off"))
    return "\n".join(lines) + "\n"


def arm_groups(arm):
    """1-based tensor indices per step-size group, rebuilt from the spec string by hand (no regex)."""
    if SPEC[arm] == "scalar":
        return [list(range(1, len(NAMES) + 1))]
    body = SPEC[arm][len("sets:"):]
    out = []
    for grp in body.split("/"):
        idx = []
        for tok in grp.split(","):
            if tok and all(c.isdigit() for c in tok):
                idx.append(int(tok))
            elif tok.count("-") == 1 and all(p and all(c.isdigit() for c in p) for p in tok.split("-")):
                a, b = (int(x) for x in tok.split("-"))
                idx += list(range(a, b + 1))
            else:
                idx.append(NAMES.index(tok) + 1)
        out.append(sorted(idx))
    return out


# --------------------------------------------------------------------------------------------------------------------
# small numerics
# --------------------------------------------------------------------------------------------------------------------
def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def fmt(x, nd=4):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --------------------------------------------------------------------------------------------------------------------
# raw `.out` reading -- no regex
# --------------------------------------------------------------------------------------------------------------------
def _num(tok):
    tok = tok.strip()
    if tok == "nan":
        return float("nan")
    return float(tok)


def parse_out(path):
    rec = {"eps": {}, "args": [], "env": [], "pt": [], "node": [], "device": None, "traceback": False,
           "run_done": False}
    for kind in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD", "COMP_HOLD", "WINDOW_HOLD", "DECAY_MASK"):
        rec[kind] = []
    prev = None
    for ln in open(path, errors="replace"):
        ln = ln.rstrip("\n")
        s = ln.strip()
        if s == "RUN_DONE":
            rec["run_done"] = True
        if "Traceback (most recent call last)" in ln:
            rec["traceback"] = True
        if ln.startswith("ARGS:"):
            rec["args"].append(ln)
        elif ln.startswith("ENV:"):
            rec["env"].append(ln)
        elif ln.startswith("PROBE_TENSOR:"):
            rec["pt"].append(ln)
        elif ln.startswith("NODE="):
            rec["node"].append(ln)
        else:
            for kind in ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD", "COMP_HOLD", "WINDOW_HOLD", "DECAY_MASK"):
                if ln.startswith(kind):
                    rec[kind].append(ln)
        if rec["device"] is None and prev is not None and prev.startswith("ARGS:") is False:
            pass
        if ln.startswith("Epoch "):
            parts = ln.split(",")
            if len(parts) >= 3 and "Train Accuracy" in parts[1] and "Test Accuracy" in parts[2]:
                head = parts[0].split()
                if len(head) == 2 and all(c.isdigit() for c in head[1]):
                    tr = _num(parts[1].split(":")[1].split("%")[0])
                    te = _num(parts[2].split(":")[1].split("%")[0])
                    rec["eps"][int(head[1])] = (tr, te)
        if rec["device"] is None and ln.endswith(" MiB") and "," in ln:
            rec["device"] = ln.split(",")[0].strip()
        prev = ln
    tail = [rec["eps"].get(e) for e in range(EPOCHS - 5, EPOCHS)]
    if all(x is not None for x in tail) and all(finite(x[0]) and finite(x[1]) for x in tail):
        rec["plateau5"] = mean([x[1] for x in tail])
        rec["train5"] = mean([x[0] for x in tail])
    else:
        rec["plateau5"] = rec["train5"] = None
    rec["n_epochs"] = len(rec["eps"])
    rec["complete"] = (sorted(rec["eps"]) == list(range(EPOCHS)) and rec["run_done"] and not rec["traceback"]
                       and rec["plateau5"] is not None)
    return rec


def parse_args_line(line):
    toks = line[len("ARGS:"):].split()
    flags, cur = {}, None
    for t in toks:
        if t.startswith("--"):
            cur = t[2:]
            flags.setdefault(cur, []).append(None)
        elif cur is not None:
            v = flags[cur][-1]
            flags[cur][-1] = t if v is None else v + " " + t
    return flags


def read_runs(runsdir):
    cand = {}
    for fn in sorted(os.listdir(runsdir)):
        if not fn.endswith(".out") or not fn.startswith(PREFIX + "-"):
            continue
        stem = fn[:-len(".out")]
        bits = stem.split("-")
        if len(bits) != 4 or bits[0] != PREFIX or bits[1] not in ARMS:
            continue
        if not bits[2].startswith("s") or not bits[2][1:].isdigit() or not bits[3].isdigit():
            continue
        arm, seed, jid = bits[1], int(bits[2][1:]), int(bits[3])
        r = parse_out(os.path.join(runsdir, fn))
        r.update({"file": fn, "job": jid, "arm": arm, "seed": seed})
        cand.setdefault((arm, seed), []).append(r)
    got = {}
    for k, lst in cand.items():
        comp = [r for r in lst if r["complete"]]
        got[k] = max(comp or lst, key=lambda r: r["job"])
    return got


def load_probe(runsdir, arm, seed):
    for base in (os.path.join(runsdir, PREFIX), runsdir):
        p = os.path.join(base, "probe_%s-%s-s%d" % (PREFIX, arm, seed), "probe.jsonl")
        if os.path.exists(p):
            out = []
            for ln in open(p, errors="replace"):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    out.append(None)
                    continue
                out.append(r if isinstance(r, dict) else None)
            return out
    return None


def read_kv(path):
    d = {}
    for ln in open(path):
        p = ln.split()
        if len(p) >= 2:
            d.setdefault(p[0], " ".join(p[1:]))
    return d


# --------------------------------------------------------------------------------------------------------------------
# the replay schedule (cvt6_headpath), re-derived
# --------------------------------------------------------------------------------------------------------------------
def load_knots(path):
    if not path or not os.path.exists(path):
        return None
    body = open(path, "rb").read()
    if hashlib.sha256(body).hexdigest() != HEADPATH_SHA:
        return None
    d = json.loads(body.decode("utf-8"))
    return [int(x) for x in d["n"]], [float(x) for x in d["v"]]


def rec_value(n, kn, kv):
    if n < kn[0]:
        x = B0
    elif n >= kn[-1]:
        x = kv[-1]
    else:
        lo_, hi_ = 0, len(kn) - 1
        while hi_ - lo_ > 1:
            mid = (lo_ + hi_) // 2
            if kn[mid] <= n:
                lo_ = mid
            else:
                hi_ = mid
        k = lo_
        x = kv[k] + (kv[k + 1] - kv[k]) * (n - kn[k]) / float(kn[k + 1] - kn[k])
    return f32(min(HI, max(LO, x)))


def hold_value(arm, n):
    """idx 50's applied step-size exponent under this arm's BETA_HOLD, as float32."""
    a9 = AS_CVT9[arm]
    if a9 == "LOWHEADPATH":
        return f32(LO)
    return v_of(n, P_HIGH)


def lion_natural(beta_pre, mom_pre, z):
    a, b = B2 * mom_pre, (1 - B2) * z
    L = a + b
    tie = abs(L) <= TIE_REL * (abs(a) + abs(b))
    s = (L > 0) - (L < 0)
    return max(LO, min(HI, beta_pre - MS * s)), tie


BH_KEYS = ("bh_n", "bh_active", "bh_nat", "bh_held")
CH_KEYS = ("ch_n", "ch_active", "ch_nat", "ch_held")
WH_KEYS = ("wh_n", "wh_active", "wh_nat", "wh_held")
DM_KEYS = ("dm_n", "dm_masked", "dm_skipped", "dm_wdterm", "dm_norm", "dm_absmin", "dm_small")


def _groupvec(r, key, ng):
    v = r.get(key)
    if not (isinstance(v, list) and len(v) == 1 and isinstance(v[0], list) and len(v[0]) == ng):
        return None
    try:
        return [float(x) for x in v[0]]
    except (TypeError, ValueError):
        return None


def _audit(r, pfx, n, g, beta, want_nat, prev, tie=False):
    try:
        cn, ca, cnat, cheld = (int(r[pfx + "_n"]), int(r[pfx + "_active"]), float(r[pfx + "_nat"]),
                               float(r[pfx + "_held"]))
    except (KeyError, TypeError, ValueError):
        return False, False, None
    ok = (cn == n and abs(cheld - beta[g]) <= BH_TOL and 0 <= ca <= cn and ca >= prev
          and (tie or abs(cnat - want_nat) <= BH_TOL))
    return ok, abs(cnat - cheld) > BH_TOL, ca


def bite(arm, recs, knots):
    """G-BITE for ONE run, re-implemented: the hold audit (cvt9's, as re-derived here) AND the DECAY_MASK audit."""
    ng = len(arm_groups(arm))
    held = arm in HELD_ARMS
    k = K_MASKED[arm]
    d = {"n": 0, "nonfinite": 0, "bad_rec": 0, "bad_step": 0, "bad_lion": 0, "ties": 0, "bad_sched": 0,
         "bad_sched_pre": 0, "bad_csched": 0, "bad_csched_pre": 0, "bad_bh": 0, "bad_ch": 0,
         "keys_on_control": 0, "active_records": 0, "cactive_records": 0,
         "bh_active_last": None, "ch_active_last": None}
    dd = {"n": 0, "bad_rec": 0, "bad_n": 0, "bad_masked": 0, "bad_skipped": 0, "bad_types": 0,
          "keys_on_unmasked": 0, "nonfinite": 0, "pos_wdterm": 0, "last": None}
    if recs is None:
        return False, d, dd, "no probe.jsonl"
    if held and knots is None:
        return False, d, dd, "replay unavailable"
    prev_b = prev_c = -1
    prev_dn = -1
    for j, r in enumerate(recs):
        # ---- the hold half
        if r is None or not isinstance(r.get("step"), int) or not isinstance(r.get("beta"), list) \
                or len(r["beta"]) != ng:
            d["bad_rec"] += 1
            dd["bad_rec"] += 1
            continue
        try:
            beta = [float(x) for x in r["beta"]]
        except (TypeError, ValueError):
            d["bad_rec"] += 1
            dd["bad_rec"] += 1
            continue
        bpre, mom, z = _groupvec(r, "beta_pre", ng), _groupvec(r, "mom_pre", ng), _groupvec(r, "z_agg", ng)
        if bpre is None or mom is None or z is None:
            d["bad_rec"] += 1
            dd["bad_rec"] += 1
            continue
        s = r["step"]
        n = s + 2
        if not all(finite(x) for x in beta + bpre + mom + z):
            d["nonfinite"] += 1
        else:
            if s != 100 * j:
                d["bad_step"] += 1
            d["n"] += 1
            has_hold_key = (any(key in r for key in BH_KEYS) or any(key in r for key in CH_KEYS)
                            or any(key in r for key in WH_KEYS))
            if not held:
                for g in range(ng):
                    nat, tie = lion_natural(bpre[g], mom[g], z[g])
                    if tie:
                        d["ties"] += 1
                        continue
                    d["bad_lion"] += abs(nat - beta[g]) > BH_TOL
                d["keys_on_control"] += has_hold_key
            else:
                d["keys_on_control"] += any(key in r for key in WH_KEYS)
                d["bad_sched"] += abs(beta[1] - hold_value(arm, n)) > BH_TOL
                d["bad_sched_pre"] += abs(bpre[1] - hold_value(arm, n - 1)) > BH_TOL
                nat1, tie1 = lion_natural(bpre[1], mom[1], z[1])
                ok, act, ca = _audit(r, "bh", n, 1, beta, nat1, prev_b, tie1)
                d["bad_bh"] += not ok
                if ca is not None:
                    d["active_records"] += act
                    prev_b = max(prev_b, ca)
                    d["bh_active_last"] = ca
                d["bad_csched"] += abs(beta[0] - rec_value(n, knots[0], knots[1])) > BH_TOL
                d["bad_csched_pre"] += abs(bpre[0] - rec_value(n - 1, knots[0], knots[1])) > BH_TOL
                nat0, tie0 = lion_natural(bpre[0], mom[0], z[0])
                ok, act, ca = _audit(r, "ch", n, 0, beta, nat0, prev_c, tie0)
                d["bad_ch"] += not ok
                if ca is not None:
                    d["cactive_records"] += act
                    prev_c = max(prev_c, ca)
                    d["ch_active_last"] = ca
        # ---- the DECAY_MASK half
        dd["n"] += 1
        has = [key for key in r if key.startswith("dm_")]
        if k == 0:
            dd["keys_on_unmasked"] += bool(has)
            continue
        dn = r.get("dm_n")
        if not (isinstance(dn, int) and dn == s + 2 and dn > prev_dn):
            dd["bad_n"] += 1
        prev_dn = dn if isinstance(dn, int) else prev_dn
        if r.get("dm_masked") != k:
            dd["bad_masked"] += 1
        if not (isinstance(dn, int) and r.get("dm_skipped") == dn * k):
            dd["bad_skipped"] += 1
        wt, nr, am, sm = r.get("dm_wdterm"), r.get("dm_norm"), r.get("dm_absmin"), r.get("dm_small")
        types_ok = (isinstance(wt, (int, float)) and not isinstance(wt, bool) and isinstance(nr, list) and len(nr) == k
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in nr)
                    and isinstance(am, (int, float)) and not isinstance(am, bool) and isinstance(sm, int) and sm >= 0
                    and sorted(has) == sorted(DM_KEYS))
        if not types_ok:
            dd["bad_types"] += 1
            continue
        if not (finite(wt) and all(finite(x) for x in nr) and finite(am)):
            dd["nonfinite"] += 1
            continue
        if wt < 0 or am < 0:
            dd["bad_types"] += 1
            continue
        dd["pos_wdterm"] += wt > 0
        dd["last"] = {"dm_n": dn, "dm_skipped": r.get("dm_skipped"), "dm_wdterm": wt, "dm_absmin": am,
                      "dm_small": sm}
    ok = (len(recs) == N_RECORDS and d["bad_rec"] == 0 and d["n"] + d["nonfinite"] == N_RECORDS
          and d["bad_step"] == 0 and d["bad_lion"] == 0 and d["keys_on_control"] == 0)
    if held:
        ok = (ok and d["bad_sched"] == 0 and d["bad_sched_pre"] == 0 and d["bad_bh"] == 0 and d["bad_csched"] == 0
              and d["bad_csched_pre"] == 0 and d["bad_ch"] == 0
              and (d["ch_active_last"] is None or d["ch_active_last"] >= d["cactive_records"])
              and (d["bh_active_last"] is None or d["bh_active_last"] >= d["active_records"]))
    okd = (dd["bad_rec"] == 0 and dd["bad_n"] == 0 and dd["bad_masked"] == 0 and dd["bad_skipped"] == 0
           and dd["bad_types"] == 0 and dd["keys_on_unmasked"] == 0 and (k == 0 or dd["pos_wdterm"] > 0))
    return (ok and okd), d, dd, None


def _dm_only(recs, k):
    """the DECAY_MASK half of G-BITE alone, on ONE run, read with k masked tensors.  Used for the cross-read."""
    if recs is None:
        return False
    bad = 0
    pos = 0
    prev_dn = -1
    for r in recs:
        if r is None or not isinstance(r.get("step"), int):
            return False
        s = r["step"]
        has = [key for key in r if key.startswith("dm_")]
        if k == 0:
            bad += bool(has)
            continue
        dn = r.get("dm_n")
        if not (isinstance(dn, int) and dn == s + 2 and dn > prev_dn):
            bad += 1
            continue
        prev_dn = dn
        if r.get("dm_masked") != k or r.get("dm_skipped") != dn * k:
            bad += 1
            continue
        wt, nr, am, sm = r.get("dm_wdterm"), r.get("dm_norm"), r.get("dm_absmin"), r.get("dm_small")
        if not (isinstance(wt, (int, float)) and not isinstance(wt, bool) and isinstance(nr, list) and len(nr) == k
                and isinstance(am, (int, float)) and not isinstance(am, bool) and isinstance(sm, int) and sm >= 0
                and sorted(has) == sorted(DM_KEYS) and finite(wt) and finite(am) and wt >= 0 and am >= 0):
            bad += 1
            continue
        pos += wt > 0
    return bad == 0 and (k == 0 or pos > 0)


def tail_slope(eps, lo=80, hi=99):
    xs = [e for e in range(lo, hi + 1) if e in eps]
    if len(xs) < 3:
        return None
    ys = [eps[e][1] for e in xs]
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


# --------------------------------------------------------------------------------------------------------------------
# the branch, the stamps and the licence text -- every word RE-TYPED from CORRECTIONS 261
# --------------------------------------------------------------------------------------------------------------------
def decide(M, diverged=False):
    if diverged:
        return "UNRESOLVED-DIVERGED"
    if M["k01"] > K01_MAX:
        return "SCALAR-NOT-COLLAPSED"
    if M[CTL] - M["k01"] < GAP_MIN:
        return "CONTROL-FAILED"
    if M["HIGHHEADPATH"] > M["k01"] + NULL_BAR:
        return "REPLICATE-FAILED"
    if M["HIGHWD0"] >= M[CTL] - MATCH_BAR:
        route = "WD-ROUTE"
    elif M["HIGHWD0"] <= M["HIGHHEADPATH"] + NULL_BAR:
        route = "STEP-ROUTE"
    else:
        route = "BOTH-ROUTES"
    if M["k01WD0"] >= M[CTL] - MATCH_BAR:
        scal = "SCALAR-NEEDS-CARRIER-WD"
    elif M["k01WD0"] <= M["k01"] + NULL_BAR:
        scal = "SCALAR-WITHOUT-CARRIER-WD"
    else:
        scal = "SCALAR-PARTIAL"
    return "%s | %s" % (route, scal)


READING = "240's OWN-STEP-MAGNITUDE, 246's HIGHHEADPATH-AT-K01 and 253's DOSE-GRADED"
LICENSE = {
    "WD-ROUTE": [
        "HIGHWD0 keeps the control's level: with coupled weight decay removed from layer4.1.bn2.weight ALONE (its weight",
        "update; its trace h feeds only a vote the holds overwrite), the held large step no longer stalls PlainNet at this",
        "cell.  Sentence licensed: 'at this cell the held large step's damage requires coupled WD on the carrier scale' --",
        "WD on that scale is NECESSARY for the damage and the large step is sufficient only together with it.  EFFECT ON",
        READING + ": REINTERPRETED -- 'own large step size' acts through the WD shrinkage it multiplies (a*wd per step);",
        "novelty narrows to a configuration (coupled WD on a norm scale, which common practice avoids).  Not licensed: the",
        "scalar k01 collapse (read the SCALAR token), ResNet, decoupled WD, the filter-collapse mechanism."],
    "STEP-ROUTE": [
        "HIGHWD0 sits within %.0f pp of HIGHHEADPATH: removing coupled WD from layer4.1.bn2.weight does NOT remove the held" % NULL_BAR,
        "large step's stall at this cell.  IDENTICAL IN LEVEL TO THE BROKEN-MASK NULL, which G-BITE excluded.  Sentence",
        "licensed: 'WD on the carrier scale is not necessary for the held-step damage at this cell; the damage goes through",
        "the step itself (gradient / momentum route)'.  EFFECT ON " + READING + ": STANDS, now as a step-size effect not",
        "reducible to WD on that scale.  Not licensed: WD on the other 52 tensors (unchanged, 0.1), ResNet, other doses."],
    "BOTH-ROUTES": [
        "HIGHWD0 sits between the stall and the control (state stamped): removing WD on 50 gives back part of the damage.",
        "EFFECT ON " + READING + ": PARTLY a WD route; read P_WD, P_LEFT and F_WD.  No single-route sentence."],
    "SCALAR-NEEDS-CARRIER-WD": [
        "k01WD0 keeps the control's level: removing coupled WD from the ONE carrier tensor (update AND trace) rescues the",
        "scalar collapse at this cell.  Sentence licensed: 'the PlainNet scalar collapse requires coupled WD on",
        "layer4.1.bn2.weight at this cell'.  Not licensed: which route (weight shrink vs meta trace; both change); ResNet."],
    "SCALAR-WITHOUT-CARRIER-WD": [
        "k01WD0 sits within %.0f pp of k01: the PlainNet scalar collapse does not need coupled WD on the carrier at this cell." % NULL_BAR,
        "Not licensed: WD on the other tensors (still on); ResNet (cwd1 masks all 20 norm scales there)."],
    "SCALAR-PARTIAL": [
        "k01WD0 sits between k01 and the control (state stamped): WD on the carrier carries part of the scalar collapse.",
        "Read P_SC; no single sentence."],
}
NOT_LICENSED = [
    "  * Anything about ResNet18_c100 (cwd1 is the ResNet batch), VGG, GroupNorm, or any tensor but idx 50.",
    "  * Decoupled weight decay, a weight-decay value other than 0.1 on the other 52 tensors, momentum other than 0.99.",
    "  * Which sub-route of k01WD0 acts (weight shrink vs meta trace).  Filter collapse (read dm_absmin / dm_small only",
    "    descriptively).  That the forced complement stands for a free one.  Anything beyond 100 epochs.",
]


# --------------------------------------------------------------------------------------------------------------------
# the corpus disclosure, re-derived by hand (no csv module, no corpus_exclusions import)
# --------------------------------------------------------------------------------------------------------------------
def corpus_filtered_count(csvpath, tsvpath):
    if not csvpath or not os.path.exists(csvpath) or not tsvpath or not os.path.exists(tsvpath):
        return None
    ex = set()
    lines = [ln.rstrip("\n") for ln in open(tsvpath, errors="replace") if ln.strip() and not ln.startswith("#")]
    head = lines[0].split("\t")
    ir, ij = head.index("run"), head.index("job_id")
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) == len(head):
            ex.add((f[ir], f[ij]))
    body = open(csvpath, errors="replace").read().splitlines()
    hdr = body[0].split(",")
    kr, kj = hdr.index("run"), hdr.index("job_id")
    n = 0
    for ln in body[1:]:
        if not ln:
            continue
        f = _split_csv(ln)
        run = f[kr] if kr < len(f) else ""
        job = f[kj] if kj < len(f) else ""
        if run.startswith(PREFIX + "-"):
            continue
        if (run, job) in ex:
            continue
        n += 1
    return n


def _split_csv(ln):
    out, cur, q = [], [], False
    for ch in ln:
        if ch == '"':
            q = not q
        elif ch == "," and not q:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


# --------------------------------------------------------------------------------------------------------------------
# REBUILD the scorer's whole stdout
# --------------------------------------------------------------------------------------------------------------------
def rebuild(runsdir, replay_path):
    L = []
    P = L.append
    bar = "=" * 78
    P(bar)
    P(" cwd2 -- PLAINNET TENSOR 50: DOES THE HELD LARGE STEP ACT THROUGH COUPLED WEIGHT DECAY ON THAT SCALE?  (PATCH_DECAYMASK)")
    P(" %s / %s   %d epochs   seeds %s   AUGMENT=%s  BETA_CLIP=%s  PROBE=%d  PROBE_TENSOR=1"
      % (NET, DSET, EPOCHS, ",".join(str(s) for s in SEEDS), AUG, CLIP, PROBE))
    for a in ARMS:
        P("   %-12s %-38s BETA_HOLD %-30s COMP_HOLD %-18s DECAY_MASK %s"
          % (a, SPEC[a], BETAHOLD[a] or "(unset)", COMPHOLD[a] or "(unset)", DMASK[a] or "(unset)"))
    P(" CO-PRIMARY: P_WD = HIGHWD0 - HIGHHEADPATH; P_LEFT = LOWWD0 - HIGHWD0; P_SC = k01WD0 - k01.")
    P(" plateau5 = mean TEST over epochs %d..%d of each run's own .out.  BARS ARE FROZEN LITERALS (O2)." % (EPOCHS - 5, EPOCHS - 1))
    P(bar)
    P("@CORPUS@")                                        # host-dependent; checked separately
    runs = read_runs(runsdir)
    P("")
    P("COMPLETE  %d runs, epochs 0..%d, RUN_DONE, no traceback, finite plateau5" % (len(ARMS) * len(SEEDS), EPOCHS - 1))
    for s in SEEDS:
        for a in ARMS:
            r = runs.get((a, s))
            ok = r is not None and r["complete"]
            P("  %-4s %s-%s-s%d  %s" % ("OK" if ok else "MISS", PREFIX, a, s,
              "absent" if r is None else "%d epoch lines, RUN_DONE=%s, traceback=%s, plateau5 %s (train %s)"
              % (r["n_epochs"], r["run_done"], r["traceback"], fmt(r["plateau5"]), fmt(r["train5"]))))
    P("")
    P("G-ARGS    every run's OWN ARGS line says what its file name registers")
    for s in SEEDS:
        for a in ARMS:
            r = runs[(a, s)]
            f = parse_args_line(r["args"][0])
            dup = [k for k, v in f.items() if len(v) > 1]
            want = {"optimizer": "HF", "alg-base": "SGDm", "alg-meta": "Lion", "dataset": DSET, "NN-name": NET,
                    "batch-size": str(BATCH), "num-epochs": str(EPOCHS), "meta-stepsize": MST, "alpha0": A0,
                    "stepsize-groups": SPEC[a], "seed": str(s), "run-name": "%s-%s-s%d" % (PREFIX, a, s),
                    "momentum-param-base": "0.99", "weight-decay-base": "0.1", "momentum-param-meta": "0.99",
                    "Lion-beta2-meta": "0.9", "weight-decay-meta": "0", "gamma": "1"}
            bad = [k for k, v in want.items() if f.get(k, [None])[-1] != v]
            P("  %-4s G-ARGS %s-s%d" % ("PASS" if (not dup and not bad) else "FAIL", a, s))
    P("")
    P("G-ENV     the environment rode the ENV line on every run")
    envs = sorted(set(" ".join(t for t in ln.split() if not t.startswith("PROBE_DIR=")) for r in runs.values()
                      for ln in r["env"]))
    ok1 = len(envs) == 1 and all(len(r["env"]) == 1 for r in runs.values())
    P("  %-4s G-ENV one distinct ENV line, one per run   %d distinct" % ("PASS" if ok1 else "FAIL", len(envs)))
    for ln in envs:
        P("        %s" % ln)
    P("  %-4s G-ENV the ENV line is cvt1's ... cvt9's, byte for byte (PROBE_DIR stripped)"
      % ("PASS" if envs == [ENV_EXPECTED] else "FAIL"))
    badpt = [(a, s) for (a, s), r in sorted(runs.items())
             if len(r["pt"]) != 1 or not r["pt"][0].startswith("PROBE_TENSOR: on every=%d type=%s tensors=%d "
                                                               % (PROBE, PT_TYPE[a], NTENS))]
    P("  %-4s G-ENV every run printed ONE `PROBE_TENSOR: on every=%d type=<its arm's> tensors=%d`   %s"
      % ("PASS" if not badpt else "FAIL", PROBE, NTENS, ("bad: %s" % badpt[:5]) if badpt else "%d runs" % len(runs)))
    P("")
    P("G-WITNESS one line of each kind per run == the arm's registered witness (VOTE_W / BETA_HOLD / COMP_HOLD / WINDOW_HOLD / DECAY_MASK)")
    for s in SEEDS:
        for a in ARMS:
            r = runs[(a, s)]
            for kind, want in (("VOTE_W", WITNESS_VW), ("BETA_HOLD", WITNESS_BH[a]), ("COMP_HOLD", WITNESS_CH[a]),
                               ("WINDOW_HOLD", WITNESS_WH), ("DECAY_MASK", WITNESS_DM[a])):
                P("  %-4s G-WITNESS %s %s-s%d" % ("PASS" if r[kind] == [want] else "FAIL", kind, a, s))
            others = [k for k in ("GROUP_HOLD", "REST_HOLD") if r[k]]
            P("  %-4s G-WITNESS %s-s%d prints no GROUP_HOLD / REST_HOLD line (cvt9 lineage)"
              % ("PASS" if not others else "FAIL", a, s))
    P("")
    P("G-STRUCT  the batch's own PARTITION-MANIFEST.txt == the frozen manifest, byte for byte")
    mp = os.path.join(runsdir, PREFIX, "PARTITION-MANIFEST.txt")
    P("@MANIFESTPATH@")
    txt = open(mp).read()
    mt = manifest_text()
    P("  %-4s G-STRUCT byte-identical to cwd_design.manifest_text(CWD2)   %d vs %d bytes"
      % ("PASS" if txt == mt else "FAIL", len(txt), len(mt)))
    P("")
    P("G-PROV    PROVENANCE.txt")
    pp = os.path.join(runsdir, PREFIX, "PROVENANCE.txt")
    prov = read_kv(pp)
    P("@PROVPATH@")
    for key, want, lab in (("MODE", "submit", "MODE submit"),
                           ("BUILD_NETWORK_SHA256", BN_SHA, "build_network.py == cvt9's"),
                           ("HF_SHA256", HF_POST_SHA, "HF.py == cwd2's PATCH_DECAYMASK tree"),
                           ("HF_PRE_DECAYMASK_SHA256", HF_PARENT_SHA, "HF.py.pre_decaymask == cvt9's HF.py"),
                           ("RUNNER_SHA256", RUNNER_SHA, "runner == run_cifar_cwd2.sh"),
                           ("REPLAY_SHA256", HEADPATH_SHA, "replay == cvt6_headpath"),
                           ("SCORER_SHA256", SCORER_SHA, "SCORER == this file"),
                           ("DESIGN_SHA256", DESIGN_SHA, "DESIGN == the cwd_design.py this scorer loaded"),
                           ("COMMON_SHA256", COMMON_SHA, "COMMON == the cwd_common.py this scorer loaded")):
        P("  %-4s G-PROV %s   %s=%s" % ("PASS" if prov.get(key) == want else "FAIL", lab, key,
                                        str(prov.get(key))[:16]))
    P("")
    P("G-REPLAY  the replay file cvt9's bite_check reads")
    knots = load_knots(replay_path)
    P("  %-4s G-REPLAY patches/%s.json sha256 == HEADPATH_SHA" % ("PASS" if knots is not None else "FAIL",
                                                                  HEADPATH_ID))
    P("")
    P("LEVELS    plateau5 (TEST) and TRAIN, IN BATCH; tail slope = TEST OLS over epochs 80-99 (descriptive)")
    arm_v = dict((a, [runs[(a, s)]["plateau5"] for s in SEEDS]) for a in ARMS)
    arm_t = dict((a, [runs[(a, s)]["train5"] for s in SEEDS]) for a in ARMS)
    for a in ARMS:
        v, t = arm_v[a], arm_t[a]
        sl = [tail_slope(runs[(a, s)]["eps"]) for s in SEEDS]
        P("  %-12s TEST %.4f  sd %s  range %.4f   TRAIN %.4f   tail slope %s pp/ep"
          % (a, mean(v), fmt(sd(v), 4), max(v) - min(v), mean(t), "/".join(fmt(x, 3) for x in sl)))
        P("               seeds: %s" % ", ".join("s%d %.4f (train %.4f)" % (s, x, y) for s, x, y in zip(SEEDS, v, t)))
    M = dict((a, mean(arm_v[a])) for a in ARMS)
    T = dict((a, mean(arm_t[a])) for a in ARMS)
    hi = max(M.values())
    P("")
    P("G-FLOOR / G-CEIL")
    P("  %-4s G-FLOOR max arm mean >= %.2f pp (chance %.2f)   max %.4f"
      % ("PASS" if hi >= FLOOR_MIN else "FAIL", FLOOR_MIN, CHANCE, hi))
    P("  %-4s G-CEIL max arm mean <= %.2f pp   max %.4f" % ("PASS" if hi <= CEIL_MAX else "FAIL", CEIL_MAX, hi))
    P("")
    P("G-BITE    cvt9's registered bite_check on each run read as its cvt9 twin, AND the DECAY_MASK record audit")
    nonfin = {}
    for s in SEEDS:
        for a in ARMS:
            recs = load_probe(runsdir, a, s)
            ok, d9, dd, why = bite(a, recs, knots)
            nonfin[(a, s)] = d9["nonfinite"] + dd["nonfinite"]
            P("  %-4s G-BITE %-12s s%d  as cvt9 %-12s records %s  lion-mismatch %d  sched50 %d/%d  schedC %d/%d  bh %d ch %d "
              "keys-on-control %d nonfinite %d | dm: k=%d bad_n %d bad_skipped %d bad_masked %d bad_types %d on-unmasked %d "
              "pos_wdterm %d nonfinite %d last %s%s"
              % ("PASS" if ok else "FAIL", a, s, AS_CVT9[a], "absent" if recs is None else len(recs), d9["bad_lion"],
                 d9["bad_sched"], d9["bad_sched_pre"], d9["bad_csched"], d9["bad_csched_pre"], d9["bad_bh"],
                 d9["bad_ch"], d9["keys_on_control"], d9["nonfinite"], K_MASKED[a], dd["bad_n"], dd["bad_skipped"],
                 dd["bad_masked"], dd["bad_types"], dd["keys_on_unmasked"], dd["pos_wdterm"], dd["nonfinite"],
                 dd["last"], ("  (%s)" % why) if why else ""))
    P("")
    P("SIGMA     O2: max(frozen floor, in-batch) -- nothing from the corpus")
    ss = sum(sum((x - M[a]) ** 2 for x in arm_v[a]) for a in ARMS)
    df = sum(len(arm_v[a]) - 1 for a in ARMS)
    sigma_in = math.sqrt(ss / df)
    which, sigma_used = ("SIGMA_PRIOR_frozen", SIGMA_PRIOR) if SIGMA_PRIOR >= sigma_in else ("SIGMA_INBATCH", sigma_in)
    P("  %-20s %.6f" % ("SIGMA_PRIOR_frozen", SIGMA_PRIOR))
    P("  %-20s %.6f" % ("SIGMA_INBATCH", sigma_in))
    se = sigma_used * math.sqrt(2.0 / len(SEEDS))
    P("  SIGMA_USED           %.6f  (= %s)" % (sigma_used, which))
    P("  SE_ARM_DIFF          %.6f  = SIGMA_USED * sqrt(2/%d)  (df_inbatch %d)" % (se, len(SEEDS), df))
    P("")
    P("CONTRASTS (every one WITHIN batch)")
    CP = {}
    for nm, (x, y) in (("P_WD", ("HIGHWD0", "HIGHHEADPATH")), ("P_LEFT", (CTL, "HIGHWD0")), ("P_SC", ("k01WD0", "k01"))):
        CP[nm] = (M[x] - M[y], T[x] - T[y])
        P("  PRIMARY  %-7s = %-12s - %-12s = %+.4f pp = %+.2f SE   (TRAIN %+.4f)   seeds %s"
          % (nm, x, y, CP[nm][0], CP[nm][0] / se, CP[nm][1],
             " / ".join("%+.3f" % (runs[(x, s)]["plateau5"] - runs[(y, s)]["plateau5"]) for s in SEEDS)))
    D_CTL = M[CTL] - M["k01"]
    R_HIGH = M[CTL] - M["HIGHHEADPATH"]
    P("  KEY      D_CTL   = LOWWD0 - k01          = %+.4f pp = %+.2f SE   (TRAIN %+.4f)  [positive control]"
      % (D_CTL, D_CTL / se, T[CTL] - T["k01"]))
    P("  KEY      R_HIGH  = LOWWD0 - HIGHHEADPATH = %+.4f pp = %+.2f SE   (TRAIN %+.4f)  [the damage to explain]"
      % (R_HIGH, R_HIGH / se, T[CTL] - T["HIGHHEADPATH"]))
    F_WD = CP["P_WD"][0] / R_HIGH if R_HIGH else float("nan")
    P("  DESCR    F_WD    = P_WD / R_HIGH         = %.3f  [fraction of the held-step damage removed by WD 0 on 50]" % F_WD)
    P("  bars: GAP_MIN %.1f (%.2f SE) | MATCH %.1f pp (%.2f SE) | NULL %.1f pp (%.2f SE) | RESCUE %.1f pp (%.2f SE)"
      % (GAP_MIN, GAP_MIN / se, MATCH_BAR, MATCH_BAR / se, NULL_BAR, NULL_BAR / se, RESCUE_BAR, RESCUE_BAR / se))
    P("  state bars: WD-ROUTE / NEEDS iff L >= %.4f ; STEP-ROUTE iff HIGHWD0 <= %.4f ; WITHOUT iff k01WD0 <= %.4f"
      % (M[CTL] - MATCH_BAR, M["HIGHHEADPATH"] + NULL_BAR, M["k01"] + NULL_BAR))
    P("")
    P("G-DIVERGE (a branch condition, not a harness gate)")
    div = [a for a in ARMS if max(arm_v[a]) - min(arm_v[a]) > DIVERGED_BAR]
    P("  %s every arm's seed range <= %.1f pp%s" % ("PASS" if not div else "FAIL", DIVERGED_BAR,
                                                    "" if not div else "  diverged: %s" % div))
    branch = decide(M, bool(div))
    stamps = ["HARNESS-CLEAN", "PATCH-BITES", "HOLDS-FROM-INIT", "COMPLEMENT-ON-HEADPATH", "HELD-ARMS-OPEN-LOOP",
              "MASK-ONE-TENSOR-IDX50", "K01WD0-BOTH-ROUTES-CHANGED", "ONE-NETWORK-PLAINNET", "ONE-CELL",
              "HORIZON-100-ONLY", "SIGMA-PRIOR-FROZEN" if which == "SIGMA_PRIOR_frozen" else "SIGMA-INBATCH"]
    if div:
        stamps.append("DIVERGED-%s" % "-".join(div))
    stamps.append("POSITIVE-CONTROL-REPRODUCES" if D_CTL >= GAP_MIN else "POSITIVE-CONTROL-FAILED")
    for a in MASKED + ("HIGHHEADPATH",):
        if M[a] - M[CTL] > MATCH_BAR:
            stamps.append("%s-ABOVE-CTL" % a)
        if M[a] - M["k01"] < -NULL_BAR:
            stamps.append("%s-BELOW-K01" % a)
        if any(nonfin[(a, s)] for s in SEEDS):
            stamps.append("NONFINITE-%s" % a)
    if M["HIGHWD0"] < M["HIGHHEADPATH"] - NULL_BAR:
        stamps.append("HIGHWD0-BELOW-HIGHHEADPATH")
    if abs(M[CTL] - 65.2080) > MATCH_BAR:
        stamps.append("CTL-DIFFERS-FROM-CVT9-LOWHEADPATH")
    if any(abs(M[a] - M["k01"]) <= NULL_BAR for a in ARMS if a != "k01"):
        stamps.append("FLOOR-READINGS-ARE-BOUNDS")
    agree = all((p > 0) == (t > 0) for p, t in CP.values() if abs(p) >= RESCUE_BAR)
    stamps.append("TRAIN-AGREES" if agree else "TRAIN-DISAGREES")
    final = "FINAL: %s | %s" % (branch, " | ".join(stamps))
    P("")
    P(bar)
    P(final)
    P(bar)
    P("")
    P("BETWEEN-BATCH, NON-GATING (frozen literals; NO other batch enters any cwd2 bar or contrast):")
    for b_ in R_BETWEEN_ORDER:
        P("  %s: %s" % (b_, "  ".join("%s %.4f" % kv for kv in R_BETWEEN[b_])))
    P("  cwd2 {%s}: %s" % (",".join(str(s) for s in SEEDS), "  ".join("%s %.4f" % (a, M[a]) for a in ARMS)))
    P("")
    P("DESCRIPTIVE (non-gating) -- PATCH_DECAYMASK's own readout on the masked arms: ||w_50|| (dm_norm), min |w_50|,")
    P("entries |w| < 1e-3, at records 0 / 50 / 95 / 200 / 499 (epochs 0 / 10 / 19 / 40 / 99.8); arm means over seeds")
    for a in MASKED:
        rows = [load_probe(runsdir, a, s) or [] for s in SEEDS]
        parts = []
        for k in (0, 50, 95, 200, 499):
            v = [r[k] for r in rows if len(r) > k and isinstance(r[k], dict) and isinstance(r[k].get("dm_norm"), list)]
            if not v:
                parts.append("rec %d n/a" % k)
                continue
            nr = mean([x["dm_norm"][0] for x in v if finite(x["dm_norm"][0])] or [float("nan")])
            am = mean([x["dm_absmin"] for x in v if finite(x["dm_absmin"])] or [float("nan")])
            smv = mean([x["dm_small"] for x in v])
            parts.append("rec %d: norm %.4g absmin %.3g small %.1f" % (k, nr, am, smv))
        P("  %-8s %s" % (a, " | ".join(parts)))
    P("")
    P("WHAT THIS DECIDES -- as registered, before any run existed:")
    for tok in branch.split(" | "):
        for ln in LICENSE[tok]:
            P("  " + ln)
    P("")
    P("WHAT IT DOES NOT LICENSE, UNCONDITIONALLY:")
    for ln in NOT_LICENSED:
        P(ln)
    extras = {"M": M, "T": T, "branch": branch, "final": final, "runs": runs, "se": se, "sigma_in": sigma_in,
              "CP": CP, "D_CTL": D_CTL, "R_HIGH": R_HIGH, "F_WD": F_WD, "arm_v": arm_v, "arm_t": arm_t,
              "manifest_sha": hashlib.sha256(mt.encode()).hexdigest(), "div": div}
    return L, extras


# --------------------------------------------------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: cwd2_attack_indep.py <runsdir> <scorer-log> [--csv P] [--tsv P] [--replay P]")
        return 2
    runsdir, logpath = args[0], args[1]
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    opt = {"--csv": os.path.join(repo, "results", "all_runs.csv"),
           "--tsv": os.path.join(repo, "results", "CORPUS-EXCLUSIONS.tsv"),
           "--replay": os.path.join(repo, "patches", HEADPATH_ID + ".json")}
    i = 2
    while i + 1 < len(args) + 1 and i < len(args):
        if args[i] in opt:
            opt[args[i]] = args[i + 1]
            i += 2
        else:
            i += 1

    rebuilt, X = rebuild(runsdir, opt["--replay"])
    got = [ln.rstrip("\n") for ln in open(logpath, errors="replace")]

    bar = "=" * 78
    print(bar)
    print(" cwd2_attack_indep -- AN INDEPENDENT RE-DERIVATION OF THE REGISTERED cwd2 SCORER'S OUTPUT")
    print(" CORRECTIONS 261.  No repo module imported, no regular expression, every literal re-typed.")
    print(bar)

    print("")
    print("[1] WHAT WAS REBUILT FROM SCRATCH, BEFORE THE LOG WAS OPENED")
    v = []

    def ck(cond, label, extra=""):
        print("  %-4s %s%s" % ("PASS" if cond else "FAIL", label, ("   " + extra) if extra else ""))
        if not cond:
            v.append(label)

    ck(len(TENSORS) == NTENS and sum(x[1] for x in TENSORS) == TOTPAR,
       "PlainNet18_c100 rebuilt from the architecture", "%d tensors, %d params" % (len(TENSORS), sum(x[1] for x in TENSORS)))
    t50 = TENSORS[IDX_HEAD_1BASED - 1]
    ck(t50[0] == CARRIER and t50[1] == 512 and t50[2] == "BatchNorm2d",
       "idx 50 (1-based) is the carrier", "%s %s numel %d" % (t50[0], t50[2], t50[1]))
    ck(dm_indices(CARRIER) == [IDX_HEAD_1BASED - 1], "DECAY_MASK resolves to exactly one tensor, 0-based index 49",
       "masked numel %d" % sum(TENSORS[i][1] for i in dm_indices(CARRIER)))
    g = arm_groups("HIGHHEADPATH")
    ck(len(g) == 2 and g[1] == [IDX_HEAD_1BASED] and len(g[0]) == 52,
       "HEAD's grouping rebuilt from the spec string", "group sizes %s" % [len(x) for x in g])
    ck(X["manifest_sha"] is not None, "the frozen manifest rebuilt", "sha256 %s" % X["manifest_sha"][:16])
    knots = load_knots(opt["--replay"])
    ck(knots is not None and len(knots[0]) == HEADPATH_KNOTS and knots[0][0] == HEADPATH_N0
       and knots[0][-1] == HEADPATH_N1, "the replay file verified by sha and re-read",
       "knots %d n0 %d n1 %d" % (len(knots[0]) if knots else -1, knots[0][0] if knots else -1,
                                 knots[0][-1] if knots else -1))
    ck(abs(f32(math.log(1e-6)) - (-13.815510749816895)) == 0.0, "B0 re-derived through float32", "%r" % B0)
    ck(v_of(P_HIGH + 1, P_HIGH) == f32(min(HI, max(LO, B0 + MS * P_HIGH))), "the tri schedule's peak re-derived",
       "peak %r" % min(HI, max(LO, B0 + MS * P_HIGH)))

    print("")
    print("[1b] COULD THE DECAY_MASK HALF OF G-BITE HAVE FAILED?  every run's records read BOTH ways")
    print("     (k = the arm's registered number of masked tensors, and k = the other value): exactly one may pass.")
    cross = []
    for s in SEEDS:
        for a in ARMS:
            recs = load_probe(runsdir, a, s)
            own = _dm_only(recs, K_MASKED[a])
            other = _dm_only(recs, 1 - K_MASKED[a])
            cross.append((a, s, own, other))
            print("      %-12s s%d  as-registered(k=%d) %s   cross-read(k=%d) %s"
                  % (a, s, K_MASKED[a], "PASS" if own else "FAIL", 1 - K_MASKED[a], "PASS" if other else "FAIL"))
    ck(all(o and not x for _a, _s, o, x in cross),
       "every run passes its own DECAY_MASK audit and FAILS the other arm-kind's",
       "%d runs, %d cross-reads refused" % (len(cross), sum(1 for c in cross if not c[3])))

    print("")
    print("[2] THE NUMBERS, RE-DERIVED FROM THE RAW RECORDS ONLY")
    for a in ARMS:
        print("  %-12s TEST %.4f  TRAIN %.4f  seeds %s" % (a, X["M"][a], X["T"][a],
              " / ".join("%.4f" % x for x in X["arm_v"][a])))
    print("  SIGMA_INBATCH %.6f   SE_ARM_DIFF %.6f" % (X["sigma_in"], X["se"]))
    for nm in ("P_WD", "P_LEFT", "P_SC"):
        print("  %-7s %+.4f pp = %+.2f SE   (TRAIN %+.4f)" % (nm, X["CP"][nm][0], X["CP"][nm][0] / X["se"],
                                                              X["CP"][nm][1]))
    print("  D_CTL   %+.4f pp   R_HIGH  %+.4f pp   F_WD %.3f" % (X["D_CTL"], X["R_HIGH"], X["F_WD"]))
    print("  branch  %s" % X["branch"])

    print("")
    print("[3] LINE-BY-LINE AGREEMENT WITH THE SCORER LOG")
    n_lines = min(len(rebuilt), len(got))
    ck(len(rebuilt) == len(got), "the rebuild has the scorer log's line count",
       "%d rebuilt vs %d in the log" % (len(rebuilt), len(got)))
    verbatim = host = mism = 0
    first = []
    for i in range(n_lines):
        mine, theirs = rebuilt[i], got[i]
        if mine == "@CORPUS@":
            want_n = corpus_filtered_count(opt["--csv"], opt["--tsv"])
            okc = want_n is not None and theirs == (
                "corpus: %d rows after corpus_exclusions.filter_rows, %s- excluded  "
                "(DISCLOSURE ONLY -- no bar, sigma, branch or stamp reads it)" % (want_n, PREFIX))
            if okc:
                host += 1
            else:
                mism += 1
                first.append(i + 1)
            continue
        if mine == "@MANIFESTPATH@":
            if theirs.startswith("  manifest: ") and theirs.endswith("/cwd2/PARTITION-MANIFEST.txt (<runsdir>/cwd2/)"):
                host += 1
            else:
                mism += 1
                first.append(i + 1)
            continue
        if mine == "@PROVPATH@":
            if theirs.startswith("  provenance: ") and theirs.endswith("/cwd2/PROVENANCE.txt (<runsdir>/cwd2/)"):
                host += 1
            else:
                mism += 1
                first.append(i + 1)
            continue
        if mine == theirs:
            verbatim += 1
        else:
            mism += 1
            first.append(i + 1)
    print("  lines in the scorer log            %d" % len(got))
    print("  rebuilt and matched VERBATIM       %d" % verbatim)
    print("  matched host-independently          %d   (2 path disclosures by suffix; 1 corpus count re-filtered here)" % host)
    print("  MISMATCHES                          %d%s" % (mism, ("   first at lines %s" % first[:10]) if first else ""))
    if mism:
        for i in first[:5]:
            print("    line %d" % i)
            print("      log      %r" % got[i - 1])
            print("      rebuilt  %r" % rebuilt[i - 1])
        v.append("line-by-line agreement")

    print("")
    print("[4] THE FINAL LINE")
    logfinal = [ln for ln in got if ln.startswith("FINAL:")]
    ck(len(logfinal) == 1 and logfinal[0] == X["final"], "the FINAL line is rebuilt token by token")
    print("  %s" % X["final"])

    print("")
    print("VIOLATIONS %d" % len(v))
    return 0 if not v else 1


if __name__ == "__main__":
    raise SystemExit(main())
