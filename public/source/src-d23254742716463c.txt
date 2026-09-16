#!/usr/bin/env python3
"""cvt5 REGISTRATION-TIME DERIVATIONS (CORRECTIONS 238).  Descriptive; reads ONLY the landed
records of cvt2 (k01, HEAD, K13, K33, K152; seeds 87/88/89) and cvt3 (k01, HEAD; seeds 84/85/86);
feeds no bar of cvt5's scorer (E, PIN_BOUND_R2 and the prices are frozen literals there, and the
scorer's --selftest re-runs H1-H3 below on the host's records and requires equality).

beta = the log step size of a group (probe record `beta`); group 0 is the complement (HEAD's [52,1]
grouping) or the shared step size (k01).  r = exp(beta - (-15)) = alpha / alpha_floor.  A group is
PINNED by cvh1's MAGNITUDE form when r <= 2, i.e. beta <= -15 + ln 2 = -14.3069.  5 records / epoch.

  H1  PER RUN (cvt2 k01 / HEAD / K13 / K33 / K152, cvt3 k01 / HEAD): group-0 beta at epoch 100,
      OLS slopes over the last 50 / 100 records, window slopes over epochs 20-40 / 40-60 / 60-80 /
      80-100, first record with r <= 2, first exact-clamp touch, the epoch from which beta stays on
      the exact clamp (<= -15 + 1e-3) to the end ("pinned-from", the 236.4(3) column), the median r
      over the last quarter (the gate), and after the first r <= 2: how many records return above
      r = 2, the max r, and how many leave the exact clamp.  Group 1 (idx 50) pinned-from.
  H2  K13's COMPLEMENT PIN EPOCH (r <= 2 and exact clamp) under five model families, per seed:
      tail-linear (last 50 / 100 records), long-window linear (epochs >= 50 / 40 / 30), beta = a +
      c sqrt(t) and beta = a + c ln(t) (epochs >= 30); and a HOLDOUT (fit epochs 30-80, predict the
      mean beta over epochs 95-100).  PIN_BOUND_R2 = the LATEST r <= 2 epoch over every family and
      seed; PIN_BOUND_EXACT_TAIL = the latest tail-linear exact-clamp epoch.
  H3  THE HORIZON.  E_MIN = max(ceil(PIN_BOUND_R2 + W_POST), ceil(PIN_BOUND_R2 / (1 - TAIL_FRAC))):
      the post-pin window cgn3 registered (W_POST 64.2) AND the gate's last-quarter window starting
      after the latest predicted pin (so a K13 complement that pins at the bound is classified by
      records that all lie after it).  E = E_MIN rounded UP to the next multiple of 10.
  H4  TEST / TRAIN trajectories of cvt2 K13 / K33 / HEAD / k01 from the raw .out (epochs 29..99), and
      the TEST OLS slope over 80-99 -- what "still rising at 100" means.

  python3 analysis/cvt5_registration_derivations.py $METAOPT_WS/runs
"""
import json
import math
import os
import re
import statistics
import sys

LO = -15.0
LR2 = LO + math.log(2.0)
EXACT_EPS = 1e-3
REC_PER_EPOCH = 5
TAIL_FRAC = 0.25
W_POST = 64.2
EPTR = re.compile(r"^Epoch (\d+), Train Accuracy: ([-\d.]+) %, Test Accuracy: ([-\d.]+) %")


def load(runs, batch, arm, s):
    p = os.path.join(runs, batch, "probe_%s-%s-s%d" % (batch, arm, s), "probe.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()]


def ols(y):
    n = len(y)
    mx = (n - 1) / 2.0
    my = math.fsum(y) / n
    den = math.fsum((i - mx) ** 2 for i in range(n))
    return math.fsum((i - mx) * (v - my) for i, v in enumerate(y)) / den


def fit(x, y):
    n = len(x)
    mx = math.fsum(x) / n
    my = math.fsum(y) / n
    c = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / math.fsum((a - mx) ** 2 for a in x)
    return my - c * mx, c


def run_stats(recs, k=0):
    b = [float(r["beta"][k]) for r in recs]
    n = len(b)
    ep = [r["step"] / 500.0 for r in recs]
    f2 = next((i for i, v in enumerate(b) if v <= LR2), None)
    ft = next((i for i, v in enumerate(b) if v <= LO + 1e-6), None)
    pf = None
    for i in range(n):
        if all(v <= LO + EXACT_EPS for v in b[i:]):
            pf = i
            break
    tail = b[-max(1, int(n * TAIL_FRAC)):]
    med_r = statistics.median([math.exp(v - LO) for v in tail])
    d = {"b": b, "ep": ep, "n": n, "terminal": b[-1], "first_r2": None if f2 is None else ep[f2],
         "first_touch": None if ft is None else ep[ft], "pinned_from": None if pf is None else ep[pf],
         "med_r": med_r, "s50": ols(b[-50:]), "s100": ols(b[-100:])}
    d["win"] = [ols(b[lo * REC_PER_EPOCH:hi * REC_PER_EPOCH]) * REC_PER_EPOCH
                for lo, hi in ((20, 40), (40, 60), (60, 80), (80, 100))]
    if f2 is not None:
        post = b[f2:]
        d["above_r2_after"] = sum(1 for v in post if v > LR2)
        d["max_r_after"] = max(math.exp(v - LO) for v in post)
    if ft is not None:
        d["off_exact_after_touch"] = sum(1 for v in b[ft:] if v > LO + EXACT_EPS)
    return d


def ext(d, level, window):
    sl = d["s50"] if window == 50 else d["s100"]
    if not sl < 0 or d["terminal"] <= level:
        return None
    return (d["n"] + (d["terminal"] - level) / (-sl)) / float(REC_PER_EPOCH)


def fam(d, lab, lo_ep, level):
    idx = [i for i in range(d["n"]) if d["ep"][i] >= lo_ep]
    f = {"lin": lambda e: e, "sqrt": math.sqrt, "ln": math.log}[lab]
    a, c = fit([f(d["ep"][i]) for i in idx], [d["b"][i] for i in idx])
    if not c < 0:
        return None
    v = (level - a) / c
    return v if lab == "lin" else (v * v if lab == "sqrt" else math.exp(v))


def holdout(d, lab):
    f = {"lin": lambda e: e, "sqrt": math.sqrt, "ln": math.log}[lab]
    idx = [i for i in range(d["n"]) if 30 <= d["ep"][i] < 80]
    a, c = fit([f(d["ep"][i]) for i in idx], [d["b"][i] for i in idx])
    tail = [i for i in range(d["n"]) if d["ep"][i] >= 95]
    return math.fsum(a + c * f(d["ep"][i]) for i in tail) / len(tail) - math.fsum(d["b"][i] for i in tail) / len(tail)


def s1(x):
    return "never" if x is None else "%.1f" % x


def out_curve(runs, batch, arm, s):
    fn = [f for f in os.listdir(runs) if f.startswith("%s-%s-s%d-" % (batch, arm, s)) and f.endswith(".out")]
    te, tr = {}, {}
    for ln in open(os.path.join(runs, sorted(fn)[-1])):
        m = EPTR.match(ln)
        if m:
            tr[int(m.group(1))] = float(m.group(2))
            te[int(m.group(1))] = float(m.group(3))
    return te, tr


def derive(runs, quiet=False):
    """-> dict of the frozen quantities; prints the whole derivation unless quiet."""
    P = (lambda *a: None) if quiet else print
    res = {"k13": {}, "head_first_r2": [], "k33": {}}
    P("H1  group-0 beta per run (r = exp(beta + 15); PINNED by magnitude iff r <= 2)")
    for batch, arms, seeds in (("cvt2", ("k01", "HEAD", "K13", "K33", "K152"), (87, 88, 89)),
                               ("cvt3", ("k01", "HEAD"), (84, 85, 86))):
        for a in arms:
            for s in seeds:
                recs = load(runs, batch, a, s)
                d = run_stats(recs, 0)
                P("  %s %-4s s%d  records %d  beta@100 %.4f  slope/rec last50 %+.6f last100 %+.6f  window/ep 20-40 %+.3f "
                  "40-60 %+.3f 60-80 %+.3f 80-100 %+.3f" % ((batch, a, s, d["n"], d["terminal"], d["s50"], d["s100"]) + tuple(d["win"])))
                P("                first r<=2 %s  first touch %s  pinned-from (exact) %s  median r last quarter %.4f  "
                  "after first r<=2: %s records above r 2, max r %s; after first touch: %s records off the exact clamp"
                  % (s1(d["first_r2"]), s1(d["first_touch"]), s1(d["pinned_from"]), d["med_r"],
                     d.get("above_r2_after", "-"), "%.4f" % d["max_r_after"] if "max_r_after" in d else "-",
                     d.get("off_exact_after_touch", "-")))
                if len(recs[0]["beta"]) > 1:
                    d1 = run_stats(recs, 1)
                    P("                group 1 (idx 50) pinned-from %s  first r<=2 %s" % (s1(d1["pinned_from"]), s1(d1["first_r2"])))
                if a == "HEAD":
                    res["head_first_r2"].append(d["first_r2"])
                if batch == "cvt2" and a == "K13":
                    res["k13"][s] = d
                if batch == "cvt2" and a == "K33":
                    res["k33"][s] = (d["first_r2"], d["pinned_from"], d["max_r_after"], d["above_r2_after"],
                                     d["off_exact_after_touch"])
    P("\nH2  K13's complement: the pin epoch under five model families (r <= 2 | exact clamp)")
    r2_all, exact_tail = [], []
    for s, d in sorted(res["k13"].items()):
        cells = []
        for lab, v2, vx in (("tail-linear last 50", ext(d, LR2, 50), ext(d, LO, 50)),
                            ("tail-linear last 100", ext(d, LR2, 100), ext(d, LO, 100)),
                            ("linear ep>=50", fam(d, "lin", 50, LR2), fam(d, "lin", 50, LO)),
                            ("linear ep>=40", fam(d, "lin", 40, LR2), fam(d, "lin", 40, LO)),
                            ("linear ep>=30", fam(d, "lin", 30, LR2), fam(d, "lin", 30, LO)),
                            ("sqrt ep>=30", fam(d, "sqrt", 30, LR2), fam(d, "sqrt", 30, LO)),
                            ("ln ep>=30", fam(d, "ln", 30, LR2), fam(d, "ln", 30, LO))):
            cells.append("%s %s | %s" % (lab, s1(v2), s1(vx)))
            if v2 is not None:
                r2_all.append((v2, s, lab))
            if lab.startswith("tail") and vx is not None:
                exact_tail.append((vx, s, lab))
        P("  K13 s%d: %s" % (s, "; ".join(cells)))
        P("          holdout (fit epochs 30-80, predict mean beta over 95-100; predicted - actual): linear %+.3f  sqrt %+.3f  ln %+.3f"
          % (holdout(d, "lin"), holdout(d, "sqrt"), holdout(d, "ln")))
    pb = max(r2_all)
    pbx = max(exact_tail)
    res["PIN_BOUND_R2"] = round(pb[0], 1)
    res["PIN_BOUND_R2_WHO"] = "s%d %s" % (pb[1], pb[2])
    res["PIN_BOUND_EXACT_TAIL"] = round(pbx[0], 1)
    res["R2_RANGE"] = (round(min(r2_all)[0], 1), round(pb[0], 1))
    P("  PIN_BOUND_R2 = %.1f (%s), the latest r <= 2 epoch over every family and seed; earliest %.1f"
      % (res["PIN_BOUND_R2"], res["PIN_BOUND_R2_WHO"], res["R2_RANGE"][0]))
    P("  PIN_BOUND_EXACT_TAIL = %.1f (s%d %s), the latest tail-linear exact-clamp epoch" % (res["PIN_BOUND_EXACT_TAIL"], pbx[1], pbx[2]))
    P("\nH3  THE HORIZON")
    e_post = int(math.ceil(res["PIN_BOUND_R2"] + W_POST))
    e_tail = int(math.ceil(res["PIN_BOUND_R2"] / (1.0 - TAIL_FRAC)))
    e_min = max(e_post, e_tail)
    e = int(math.ceil(e_min / 10.0) * 10)
    res.update({"E_POST": e_post, "E_TAIL": e_tail, "E_MIN": e_min, "E": e})
    P("  ceil(PIN_BOUND_R2 + W_POST %.1f) = %d;  ceil(PIN_BOUND_R2 / (1 - %.2f)) = %d;  E_MIN = %d;  E = %d (next multiple of 10)"
      % (W_POST, e_post, TAIL_FRAC, e_tail, e_min, e))
    hf = [x for x in res["head_first_r2"] if x is not None]
    res["HEAD_FIRST_R2_MAX"] = max(hf)
    P("  HEAD complement first r <= 2 (cvt2 + cvt3): %s -> latest %.1f; post-pin window at E %.1f; at the in-run 100-epoch control %.1f"
      % (" / ".join("%.1f" % x for x in res["head_first_r2"]), max(hf), e - max(hf), 100 - max(hf)))
    P("  K33 (first r<=2, pinned-from exact, max r after first r<=2, records above r 2, records off exact after touch): %s"
      % "; ".join("s%d %s" % (s, ("%.1f" % v[0], s1(v[1]), "%.4f" % v[2], v[3], v[4])) for s, v in sorted(res["k33"].items())))
    P("\nH4  cvt2 TEST (TRAIN), arm mean, from the raw .out; TEST OLS slope over epochs 80-99 per seed")
    for a in ("k01", "HEAD", "K13", "K33"):
        cur = [out_curve(runs, "cvt2", a, s) for s in (87, 88, 89)]
        cells = ["%d:%.2f(%.2f)" % (e_, statistics.mean(c[0][e_] for c in cur), statistics.mean(c[1][e_] for c in cur))
                 for e_ in (29, 39, 49, 59, 69, 79, 89, 99)]
        sl = []
        for te, _tr in cur:
            xs = list(range(80, 100))
            a_, c_ = fit([float(x) for x in xs], [te[x] for x in xs])
            sl.append(c_)
        P("  %-4s %s   slope %s pp/ep" % (a, " ".join(cells), " / ".join("%+.4f" % x for x in sl)))
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    r = derive(sys.argv[1])
    print("\nFROZEN: PIN_BOUND_R2 %.1f  PIN_BOUND_EXACT_TAIL %.1f  E_MIN %d  E %d  HEAD_FIRST_R2_MAX %.1f"
          % (r["PIN_BOUND_R2"], r["PIN_BOUND_EXACT_TAIL"], r["E_MIN"], r["E"], r["HEAD_FIRST_R2_MAX"]))
