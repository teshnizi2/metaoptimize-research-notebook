#!/usr/bin/env python3
"""cwd_common.py -- run readers, the DECAY_MASK record audit, the Lion recompute and the synthetic-run writers shared by
the registered scorers of `cwd1` (analysis/cWD1_normwd_score.py, CORRECTIONS 260) and `cwd2`
(analysis/cWD2_carrierwd_score.py, CORRECTIONS 261).  Stdlib only.  Committed with the scorers, before any run (RULE 21);
each scorer's G-PROV records this file's sha256 and its selftest pins it.

Nothing here reads the corpus or decides a branch.
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

EPTR_RE = re.compile(r"Epoch\s+(\d+),\s*Train Accuracy:\s*([0-9.]+|nan)\s*%,\s*Test Accuracy:\s*([0-9.]+|nan)")
KINDS = ("VOTE_W", "BETA_HOLD", "GROUP_HOLD", "REST_HOLD", "COMP_HOLD", "WINDOW_HOLD", "DECAY_MASK")
DM_KEYS = ("dm_n", "dm_masked", "dm_skipped", "dm_wdterm", "dm_norm", "dm_absmin", "dm_small")
HOLD_PREFIXES = ("bh_", "gh_", "rh_", "ch_", "wh_")


def mean(v):
    return sum(v) / len(v) if v else None


def sd(v):
    if len(v) < 2:
        return None
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def fmt(x, nd=4):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def parse_run(path, epochs=100):
    txt = open(path, errors="replace").read()
    eps = {}
    for a, b, c in EPTR_RE.findall(txt):
        eps[int(a)] = (float(b), float(c))
    lines = txt.splitlines()
    rec = {"eps": eps, "n_epochs": len(eps),
           "traceback": "Traceback (most recent call last)" in txt,
           "run_done": any(ln.strip() == "RUN_DONE" for ln in lines),
           "args": [ln for ln in lines if ln.startswith("ARGS:")],
           "env": [ln for ln in lines if ln.startswith("ENV:")],
           "pt": [ln for ln in lines if ln.startswith("PROBE_TENSOR:")],
           "plateau5": None, "train5": None}
    for k in KINDS:
        rec[k] = [ln for ln in lines if ln.startswith(k)]
    tail = [eps.get(e) for e in range(epochs - 5, epochs)]
    if all(x is not None for x in tail) and all(math.isfinite(x[0]) and math.isfinite(x[1]) for x in tail):
        rec["plateau5"] = mean([x[1] for x in tail])
        rec["train5"] = mean([x[0] for x in tail])
    rec["complete"] = (sorted(eps) == list(range(epochs)) and rec["run_done"]
                       and not rec["traceback"] and rec["plateau5"] is not None)
    return rec


def read_runs(runsdir, prefix, arms, epochs=100):
    out_re = re.compile(r"^%s-(%s)-s(\d+)-(\d+)\.out$" % (re.escape(prefix), "|".join(re.escape(a) for a in arms)))
    cand, notes = {}, []
    if not runsdir or not os.path.isdir(runsdir):
        return {}, notes
    for fn in sorted(os.listdir(runsdir)):
        m = out_re.match(fn)
        if not m:
            continue
        arm, seed, jid = m.group(1), int(m.group(2)), int(m.group(3))
        rec = parse_run(os.path.join(runsdir, fn), epochs)
        rec.update({"file": fn, "job": jid, "arm": arm, "seed": seed})
        cand.setdefault((arm, seed), []).append(rec)
    got = {}
    for k, lst in cand.items():
        comp = [r for r in lst if r["complete"]]
        pick = max(comp or lst, key=lambda r: r["job"])
        if len(lst) > 1:
            notes.append("%s-s%d has %d files (%d complete); using %s" % (k[0], k[1], len(lst), len(comp), pick["file"]))
        got[k] = pick
    return got, notes


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


def load_probe(runsdir, prefix, arm, seed):
    for base in (os.path.join(runsdir, prefix), runsdir):
        p = os.path.join(base, "probe_%s-%s-s%d" % (prefix, arm, seed), "probe.jsonl")
        if os.path.exists(p):
            recs = []
            for ln in open(p, errors="replace"):
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    recs.append(None)
                    continue
                recs.append(r if isinstance(r, dict) else None)
            return recs
    return None


def tail_slope(eps, lo=80, hi=99):
    xs = [e for e in range(lo, hi + 1) if e in eps]
    if len(xs) < 3:
        return None
    ys = [eps[e][1] for e in xs]
    mx, my = mean(xs), mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def dm_audit(recs, k, n_records):
    """PATCH_DECAYMASK's own record on ONE run.  k = number of masked tensors (0 = unmasked arm).
    -> (ok, detail).  Pure function of the records.
    masked (k > 0): every record carries dm_n == step + 2, dm_masked == k, dm_skipped == dm_n * k, a dm_wdterm >= 0
      (non-finite counted, not failed), dm_norm a list of k numbers, dm_absmin a number, dm_small an int >= 0; dm_n
      strictly increasing; at least one record with a finite dm_wdterm > 0 (the skipped WD term was non-trivial).
    unmasked (k == 0): no record carries any dm_* key."""
    d = {"n": 0, "bad_rec": 0, "bad_n": 0, "bad_masked": 0, "bad_skipped": 0, "bad_types": 0, "keys_on_unmasked": 0,
         "nonfinite": 0, "pos_wdterm": 0, "last": None}
    if recs is None:
        d["why"] = "no probe.jsonl"
        return False, d
    prev = -1
    for kk, r in enumerate(recs):
        if r is None or not isinstance(r.get("step"), int):
            d["bad_rec"] += 1
            continue
        d["n"] += 1
        has = [key for key in r if key.startswith("dm_")]
        if k == 0:
            d["keys_on_unmasked"] += bool(has)
            continue
        s = r["step"]
        dn = r.get("dm_n")
        if not (isinstance(dn, int) and dn == s + 2 and dn > prev):
            d["bad_n"] += 1
        prev = dn if isinstance(dn, int) else prev
        if r.get("dm_masked") != k:
            d["bad_masked"] += 1
        if not (isinstance(dn, int) and r.get("dm_skipped") == dn * k):
            d["bad_skipped"] += 1
        wt, nr, am, sm = r.get("dm_wdterm"), r.get("dm_norm"), r.get("dm_absmin"), r.get("dm_small")
        types_ok = (isinstance(wt, (int, float)) and not isinstance(wt, bool) and isinstance(nr, list) and len(nr) == k
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in nr)
                    and isinstance(am, (int, float)) and not isinstance(am, bool) and isinstance(sm, int) and sm >= 0
                    and sorted(has) == sorted(DM_KEYS))
        if not types_ok:
            d["bad_types"] += 1
            continue
        if not (_finite(wt) and all(_finite(x) for x in nr) and _finite(am)):
            d["nonfinite"] += 1
            continue
        if wt < 0 or am < 0:
            d["bad_types"] += 1
            continue
        d["pos_wdterm"] += wt > 0
        d["last"] = {"dm_n": dn, "dm_skipped": r.get("dm_skipped"), "dm_wdterm": wt, "dm_absmin": am, "dm_small": sm}
    ok = (len(recs) == n_records and d["bad_rec"] == 0 and d["n"] == n_records)
    if k == 0:
        ok = ok and d["keys_on_unmasked"] == 0
    else:
        ok = (ok and d["bad_n"] == 0 and d["bad_masked"] == 0 and d["bad_skipped"] == 0 and d["bad_types"] == 0
              and d["pos_wdterm"] > 0)
    return ok, d


def hold_keys_present(recs):
    """number of records carrying any hold patch key (bh_/gh_/rh_/ch_/wh_)."""
    if recs is None:
        return None
    return sum(1 for r in recs if isinstance(r, dict) and any(key.startswith(HOLD_PREFIXES) for key in r))


def add_dm_keys(recs, k, wdterm=1e-6, forge=None):
    """synthetic: append PATCH_DECAYMASK keys to records (selftests only).  forge: one of 'skipped', 'n', 'masked',
    'zero-wdterm', 'types' to break the audit."""
    out = []
    for r in recs:
        r = dict(r)
        n = r["step"] + 2
        r.update({"dm_n": n + (1 if forge == "n" else 0), "dm_masked": k + (1 if forge == "masked" else 0),
                  "dm_skipped": n * k + (1 if forge == "skipped" else 0),
                  "dm_wdterm": 0.0 if forge == "zero-wdterm" else wdterm * (1 + 0.001 * r["step"] / 100.0),
                  "dm_norm": ([22.6] * k) if forge != "types" else "x", "dm_absmin": 0.01, "dm_small": 0})
        out.append(r)
    return out
