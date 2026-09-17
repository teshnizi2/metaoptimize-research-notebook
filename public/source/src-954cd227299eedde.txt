#!/usr/bin/env python3
"""cvt8_attack_indep.py <runsdir> <scorer.py> <score.log> <cvt8_isopath.json> <all_runs.csv> <CORPUS-EXCLUSIONS.tsv>
-- independent re-derivation of `cvt8` (CORRECTIONS 248).

Imports NOTHING from analysis/cVT8_doseroute_score.py, analysis/corpus_exclusions.py or any other repo module, and
uses no regex.  Reads, by string splitting only: every raw cvt8-<arm>-s<seed>-<jobid>.out in <runsdir> (ARGS / ENV /
PROBE_TENSOR / VOTE_W / BETA_HOLD / COMP_HOLD / GROUP_HOLD / REST_HOLD lines, `Epoch` lines, RUN_DONE, Traceback);
<runsdir>/cvt8/PARTITION-MANIFEST.txt and PROVENANCE.txt; each run's probe dir (probe_tensor.json, block_sizes.json,
probe.jsonl); the replay file cvt8_isopath.json (its sha checked against 248.3(2)).
plateau5 = mean TEST over the run's own epochs 95..99; TRAIN5 the same on the Train column.
Bars, witnesses (built from mode / P / b0 / ms and the replay's own knots), the max-rate triangle, the rec: replay
arithmetic, Lion, G-BITE's clauses (a)-(e), the stamps and the branch order are RE-TYPED from the registration text
(248.2-248.7); nothing is imported from the scorer.  The scorer file is read only for its sha256.

CORPUS-DEPENDENT LINE, EXPLICIT.  Exactly ONE re-built line depends on the corpus: the scorer's DISCLOSURE line
`corpus: N rows after corpus_exclusions.filter_rows, cvt8- excluded ...`.  It is re-derived here from the two files
given on the command line (CSV rows whose `run` does not start with `cvt8-`, minus rows whose (run, job_id) is a TSV
key), and the shas of both files are printed.  The committed output was produced against the corpus at repo commit
45eac02 (all_runs.csv 355c0977..., 3,085 rows; CORPUS-EXCLUSIONS.tsv d0b955cb..., 75 keys -> 3,010), unchanged through
8af7ee2, which is also what both scorer logs were written against.  After the cvt8 / cvt9 ingest this line (and only
it) moves, and the parser reports 1 violation unless run against those two files at 45eac02
(`git show 45eac02:<path>`).  No bar, sigma, contrast, branch or stamp reads the corpus.

HOST-PATH LINES.  The scorer prints two lines carrying the runs path (`manifest: <runsdir>/cvt8/...`,
`provenance: <runsdir>/cvt8/...`).  They are re-built from the runsdir given and looked up verbatim, but never printed,
so the output is host-independent; the scorer log given must therefore be the one written for that runsdir.

AGREEMENT.  Every scorer line carrying a level, contrast, gate, G-BITE count, sigma, branch, stamp or descriptive
readout (and the design header) is RE-BUILT here in the scorer's printed format and looked up VERBATIM in
<score.log>; a line not found is a VIOLATION.  The FINAL line is compared as one string and token by token.  The
scorer-log lines NOT re-built are counted and classified.

Sections:
  [0] corpus disclosure; manifest (re-built byte for byte from its TENSOR lines + the registered design); PROVENANCE;
      the replay file
  [1] completeness, ARGS / ENV / PROBE_TENSOR / VOTE_W / BETA_HOLD / COMP_HOLD / GROUP_HOLD / REST_HOLD per run
  [2] levels: per-run plateau5 + TRAIN5, per-arm mean / sd / range, TEST OLS tail slope 80-99
  [3] sigma, SE, every contrast, TEST beside TRAIN; per-seed paired contrasts
  [4] G-BITE from the probe records; the vote-out decomposition (z_agg == fsum z_tensor per group);
      NON-VACUITY cross-read (every blockwise run's group 1 read as each carrier mode; group 0 read against ISOPATH)
  [5] branch, states, stamps, every bar margin, per-seed states, the 248.5 accounts; FINAL compared with the scorer's
  [6] the scorer's descriptive readouts re-derived
  [7] LINE-BY-LINE AGREEMENT
  DESCRIPTIVE, NOT REGISTERED (cannot move anything above):
  [D1] TEST (TRAIN) after 15 / 17 / 19 / 30 / 50 / 100 epochs (arm mean; TEST per seed)
  [D2] APPLIED beta of the carrier group and the complement at epochs 10 / 17 / 20 / 30 / 40 (median seed; arm mean)
  [D3] the complement's pins: magnitude gate (first r <= 2; for good), continuous exact clamp, per run
  [D4] HIGHISOPATH / BIGISOPATH (and HOLDHIGH / HOLDBIG) against LOWISOPATH (and ISO): the Epoch line from which TEST
       stays > 2 pp below, per seed and arm mean
  [D5] HOLDHIGH's free complement against cvt7's (early pin) and ISOPATH; HOLDBIG's free complement against v_9428
  [D6] timing: HOLDBIG / BIGISOPATH against k01; the free complements' departure from ISOPATH
"""
import csv
import hashlib
import json
import math
import os
import struct
import sys

PREFIX = "cvt8"
NET = "ResNet18_c100"
SEEDS = [102, 103, 104]
# 248.2, re-typed
ARMS = ["k01", "ISO", "HOLDHIGH", "HOLDBIG", "HIGHISOPATH", "BIGISOPATH", "LOWISOPATH"]
HELD = ["HOLDHIGH", "HOLDBIG", "HIGHISOPATH", "BIGISOPATH", "LOWISOPATH"]
FORCED = ["HIGHISOPATH", "BIGISOPATH", "LOWISOPATH"]
CARRIERS = ["layer4.0.bn2.weight", "layer4.0.shortcut.1.weight", "layer4.1.bn2.weight"]
CARRIER_IDX = [50, 53, 59]
ISOSPEC = "sets:1-49,51-52,54-58,60-62/" + ",".join(CARRIERS)
SPEC = dict((a, ISOSPEC) for a in ARMS)
SPEC["k01"] = "scalar"
GHOLD = {"HOLDHIGH": ("tri", 8609), "HOLDBIG": ("tri", 9428), "HIGHISOPATH": ("tri", 8609),
         "BIGISOPATH": ("tri", 9428), "LOWISOPATH": ("floor", None)}
REPLAY_ID = "cvt8_isopath"
REPLAY_SHA = "08ab25f3a329cb260bb39fb72f3299c021e7169bf612fa27d539166296e28e70"
ENV_WANT = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
            "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 "
            "EB_RHO=na EB_LOG=0")
FIXED = {"optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99", "weight-decay-base": "0.1",
         "alg-meta": "Lion", "momentum-param-meta": "0.99", "Lion-beta2-meta": "0.9",
         "weight-decay-meta": "0", "dataset": "CIFAR100", "batch-size": "100", "meta-stepsize": "1e-3",
         "alpha0": "1e-6", "num-epochs": "100", "NN-name": NET, "gamma": "1"}
# 248.6 / 248.7 frozen bars, re-typed
SIGMA_PRIOR = 0.6881530279923828
GAP_MIN, K01_MAX, MATCH, NULL, REP, RESCUE, DIVERGED = 20.0, 30.0, 5.0, 2.0, 10.0, 10.0, 5.0
FLOOR_MIN, CEIL_MAX = 15.0, 90.0
BH_TOL, TIE_REL, N_REC = 1e-5, 1e-12, 500
MS, B2 = 1e-3, 0.9
LO, HI = -15.0, -2.3026
SPE = 500
# between-batch anchors (247.3; non-gating)
BETWEEN = [("cvt7", [("k01", 22.9527), ("ISO", 70.0413), ("HOLDHIGH", 50.2273)]),
           ("ciso1", [("k01", 23.2807), ("ISO", 70.2113)]), ("cdep1", [("k01", 23.3520), ("ISO", 70.0440)])]
# sha prefixes as recorded in CORRECTIONS 248.4 / 248.8 / 248.10 / 248.12
PREF = {"BUILD_NETWORK_SHA256": "c7998883", "HF_SHA256": "5197dc2e", "RUNNER_SHA256": "1dbe4ff5",
        "REPLAY_SHA256": "08ab25f3", "SCORER_SHA256": "c6dc4a8e"}
MANIFEST_PREFIX, MANIFEST_BYTES = "a8b0a1f4", 7525
NTENS, TOTPAR = 62, 11220132
# 248.5 floor-table intervals, re-typed
K01_PL, FREE_PL, HIGH_PL, PART_PL, PV_PL = (18.0, 28.0), (62.0, 78.0), (44.0, 56.0), (30.0, 42.0), (57.0, 63.0)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


B0 = f32(math.log(1e-6))


def mean(v):
    return math.fsum(v) / len(v)


def sdev(v):
    m = mean(v)
    return math.sqrt(math.fsum((x - m) ** 2 for x in v) / (len(v) - 1))


def median(v):
    w = sorted(v)
    n = len(w)
    return w[n // 2] if n % 2 else 0.5 * (w[n // 2 - 1] + w[n // 2])


def ols(xs, ys):
    xm, ym = mean(xs), mean(ys)
    return math.fsum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / math.fsum((x - xm) ** 2 for x in xs)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sgn(x):
    return (x > 0) - (x < 0)


def tri(n, P):
    """248.2 / 243.3: u(n) = min(n-1, P) - max(0, n-1-P); beta = min(hi, max(lo, b0 + ms u(n))), float32."""
    u = 0 if n <= 0 else min(n - 1, P) - max(0, n - 1 - P)
    return f32(min(HI, max(LO, B0 + MS * u)))


def carrier_at(mode, P, n):
    return LO if mode == "floor" else tri(n, P)


class Replay(object):
    """242.3(2)'s rec: arithmetic (248.3(2)): b0 before the first knot, the last knot's value from it on, linear
    between knots, then clamp, float32.  Knot lookup by bisection written here."""

    def __init__(self, path):
        d = json.loads(open(path, "rb").read().decode("utf-8"))
        self.id = d["id"]
        self.n = [int(x) for x in d["n"]]
        self.v = [float(x) for x in d["v"]]
        self.source = list(d.get("source", []))

    def at(self, n):
        kn, kv = self.n, self.v
        if n < kn[0]:
            x = B0
        elif n >= kn[-1]:
            x = kv[-1]
        else:
            a, b = 0, len(kn) - 1
            while b - a > 1:
                c = (a + b) // 2
                if kn[c] <= n:
                    a = c
                else:
                    b = c
            x = kv[a] + (kv[a + 1] - kv[a]) * (n - kn[a]) / float(kn[a + 1] - kn[a])
        return f32(min(HI, max(LO, x)))


def lion(bp, m, z):
    """248.6 (b): clamp(beta_pre - ms*sign(b2*mom_pre + (1-b2)*z_agg)); tie iff |L| <= TIE_REL of its parts."""
    p, q = B2 * m, (1.0 - B2) * z
    L = p + q
    return max(LO, min(HI, bp - MS * sgn(L))), abs(L) <= TIE_REL * (abs(p) + abs(q))


def gh_witness(arm):
    """248.4: the GROUP_HOLD witness line, built from mode / P / b0 / ms."""
    if arm not in GHOLD:
        return "GROUP_HOLD: off"
    mode, P = GHOLD[arm]
    w = "GROUP_HOLD: on type=blockwise group=1 groupsize=3 names=%s mode=%s" % ("+".join(CARRIERS), mode)
    if mode == "floor":
        return w + " value=" + repr(LO)
    return w + " P=%d b0=%s ms=%s lo=%s hi=%s peak=%s" % (P, repr(B0), repr(MS), repr(LO), repr(HI),
                                                          repr(min(HI, max(LO, B0 + MS * P))))


def rh_witness(arm, rp):
    """248.4: the REST_HOLD witness line, built from the replay file's own sha / knots / first / last / max."""
    if arm not in FORCED:
        return "REST_HOLD: off"
    return ("REST_HOLD: on type=blockwise group=0 groupsize=59 mode=rec id=%s sha256=%s knots=%d n0=%d n1=%d b0=%s lo=%s "
            "hi=%s vmax=%s vlast=%s" % (rp.id, REPLAY_SHA, len(rp.n), rp.n[0], rp.n[-1], repr(B0), repr(LO), repr(HI),
                                        repr(max(rp.v)), repr(rp.v[-1])))


def gh_value(arm):
    mode, P = GHOLD[arm]
    return "+".join(CARRIERS) + ":" + (mode if mode == "floor" else "tri:%d" % P)


def parse_out(path):
    r = {"args": [], "env": [], "pt": [], "vw": [], "bh": [], "ch": [], "gh": [], "rh": [], "ep": {}, "done": False,
         "tb": 0, "dup": 0}
    for ln in open(path, errors="replace"):
        s = ln.rstrip("\n")
        for key, pfx in (("args", "ARGS:"), ("env", "ENV:"), ("pt", "PROBE_TENSOR:"), ("vw", "VOTE_W"), ("bh", "BETA_HOLD"),
                         ("ch", "COMP_HOLD"), ("gh", "GROUP_HOLD"), ("rh", "REST_HOLD")):
            if s.startswith(pfx):
                r[key].append(s)
                break
        if s.startswith("Epoch "):
            p = s.split(",")
            e = int(p[0].split()[1])
            tr = float(p[1].split(":")[1].replace("%", "").strip())
            te = float(p[2].split(":")[1].replace("%", "").strip())
            r["dup"] += e in r["ep"]
            r["ep"][e] = (tr, te)
        elif s.strip() == "RUN_DONE":
            r["done"] = True
        if "Traceback (most recent call last)" in s:
            r["tb"] += 1
    return r


def flags_of(line):
    out = []
    for t in line[len("ARGS:"):].split():
        if t.startswith("--"):
            out.append([t[2:], None])
        elif out:
            out[-1][1] = t if out[-1][1] is None else out[-1][1] + " " + t
    return out


def groups_of(spec, names):
    if spec == "scalar":
        return [list(range(1, len(names) + 1))]
    gs = []
    for g in spec[len("sets:"):].split("/"):
        idx = []
        for tok in g.split(","):
            if tok.isdigit():
                idx.append(int(tok))
            elif "-" in tok and all(x.isdigit() for x in tok.split("-")):
                a, b = tok.split("-")
                idx.extend(range(int(a), int(b) + 1))
            else:
                idx.append(names.index(tok) + 1)
        gs.append(sorted(idx))
    return gs


def load_recs(pdir):
    out = []
    for x in open(os.path.join(pdir, "probe.jsonl")):
        if not x.strip():
            continue
        try:
            v = json.loads(x)
        except ValueError:
            v = None
        out.append(v if isinstance(v, dict) else None)
    return out


def summary(recs, k):
    """scorer-style descriptive readout of group k, written here: (peak, peak epoch, pin = first record at/after the
    FIRST maximum from which r = exp(beta+15) <= 2 on every later record, mean r over [15,30), mean r over [30,60))."""
    seq = [(x["step"], float(x["beta"][k])) for x in recs]
    ipk = 0
    for i in range(len(seq)):
        if seq[i][1] > seq[ipk][1]:
            ipk = i
    last_hi = None
    for i in range(len(seq)):
        if math.exp(seq[i][1] + 15.0) > 2.0:
            last_hi = i
    if last_hi is None or last_hi < ipk:
        pin = seq[ipk][0] / float(SPE)
    elif last_hi + 1 < len(seq):
        pin = seq[last_hi + 1][0] / float(SPE)
    else:
        pin = None
    r1 = mean([math.exp(b + 15.0) for s, b in seq if 15 * SPE <= s < 30 * SPE])
    r2 = mean([math.exp(b + 15.0) for s, b in seq if 30 * SPE <= s < 60 * SPE])
    return seq[ipk][1], seq[ipk][0] / float(SPE), pin, r1, r2


def pins(recs, k):
    """[D3]: first record with r <= 2; records at/after it with r > 2; first record from which r <= 2 for good;
    continuous exact clamp (beta == -15.0 on that and every later record); median r over the last quarter of records."""
    seq = [(x["step"] / float(SPE), float(x["beta"][k])) for x in recs]
    f2 = None
    for e, b in seq:
        if math.exp(b + 15.0) <= 2.0:
            f2 = e
            break
    above = None if f2 is None else sum(1 for e, b in seq if e >= f2 and math.exp(b + 15.0) > 2.0)
    good = None
    for j in range(len(seq) - 1, -1, -1):
        if math.exp(seq[j][1] + 15.0) <= 2.0:
            good = seq[j][0]
        else:
            break
    exact = None
    for j in range(len(seq) - 1, -1, -1):
        if seq[j][1] == LO:
            exact = seq[j][0]
        else:
            break
    medr = median([math.exp(b + 15.0) for _e, b in seq[len(seq) - len(seq) // 4:]])
    return f2, above, good, exact, medr


def read_corpus_count(csvp, tsvp):
    lines = [ln.rstrip("\n") for ln in open(tsvp) if ln.strip() and not ln.startswith("#")]
    head = lines[0].split("\t")
    ir, ij = head.index("run"), head.index("job_id")
    keys = set()
    for ln in lines[1:]:
        f = ln.split("\t")
        keys.add((f[ir], f[ij]))
    n_all = n_keep = 0
    with open(csvp, newline="") as fh:
        for row in csv.DictReader(fh):
            n_all += 1
            if row.get("run", "").startswith(PREFIX + "-"):
                continue
            if (row.get("run"), row.get("job_id")) in keys:
                continue
            n_keep += 1
    return n_all, len(lines) - 1, len(keys), n_keep


def first_stays(vals, cond):
    """first index e from which cond(vals[f]) holds for every f >= e; None if never."""
    first = None
    for e in range(len(vals) - 1, -1, -1):
        if cond(vals[e]):
            first = e
        else:
            break
    return first


def main():
    runsdir, scorer, slog, rpath, csvp, tsvp = sys.argv[1:7]
    bad = []
    expect = []          # (section, line) that must appear verbatim in the scorer log

    def viol(msg):
        bad.append(msg)
        print("  VIOLATION " + msg)

    def exp(sec, line):
        expect.append((sec, line))

    base = os.path.join(runsdir, PREFIX)
    logl = [ln.rstrip("\n") for ln in open(slog)]
    rp = Replay(rpath)

    # ---------------------------------------------------------------- header (design lines, 248.2 / 248.6)
    rule = "=" * 78
    exp("HEADER", rule)
    exp("HEADER", " cvt8 -- ResNet18_c100: DOSE x ROUTE on the carrier group  (PATCH_GROUPHOLD + PATCH_RESTHOLD)")
    exp("HEADER", " %s / CIFAR100   100 epochs   seeds %s   AUGMENT=1  BETA_CLIP=-15:-2.3026  PROBE=100  PROBE_TENSOR=1"
        % (NET, ",".join(str(s) for s in SEEDS)))
    for a in ARMS:
        ghs = ("tri:%d" % GHOLD[a][1] if GHOLD[a][0] == "tri" else "floor") if a in GHOLD else "(unset)"
        exp("HEADER", "   %-12s %-7s GROUP_HOLD %-26s REST_HOLD %s" % (a, "scalar" if a == "k01" else "ISO's", ghs,
                                                                       "rec:" + REPLAY_ID if a in FORCED else "(unset)"))
    exp("HEADER", " CO-PRIMARY: P_DOSE = HOLDHIGH - HOLDBIG; P_ROUTE = HIGHISOPATH - HOLDHIGH; P_ROUTE_BIG = BIGISOPATH - HOLDBIG.")
    exp("HEADER", " plateau5 = mean TEST over epochs 95..99 of each run's own .out.  BARS ARE FROZEN LITERALS (O2).")

    # ---------------------------------------------------------------- [0]
    print("[0] corpus disclosure (the ONE corpus-dependent line; no bar reads it); manifest; provenance; replay file")
    n_all, n_tsv, n_keys, n_keep = read_corpus_count(csvp, tsvp)
    print("    all_runs.csv sha256 %s  rows %d;  CORPUS-EXCLUSIONS.tsv sha256 %s  rows %d keys %d  ->  %d rows (cvt8- dropped)"
          % (sha(csvp)[:16], n_all, sha(tsvp)[:16], n_tsv, n_keys, n_keep))
    exp("CORPUS", "corpus: %d rows after corpus_exclusions.filter_rows, %s- excluded  (DISCLOSURE ONLY -- no bar, sigma, "
        "branch or stamp reads it)" % (n_keep, PREFIX))

    # the replay file
    rsha = sha(rpath)
    ok_rep = (rsha == REPLAY_SHA and rp.id == REPLAY_ID and len(rp.n) == 500 and len(rp.v) == 500
              and rp.n == [100 * k + 2 for k in range(500)] and len(rp.source) == 12)
    print("    replay %s: sha256 %s... == 248.3(2): %s; knots %d, n = 100k+2 for k 0..499: %s; source runs %d; vmax %r, vlast %r"
          % (rp.id, rsha[:16], rsha == REPLAY_SHA, len(rp.n), rp.n == [100 * k + 2 for k in range(500)], len(rp.source),
             max(rp.v), rp.v[-1]))
    print("    replay schedule: value at n 0 / 1 / 2 / 102 / 25002 / 49902 / 50001 = %s" % " / ".join(
        repr(rp.at(n)) for n in (0, 1, 2, 102, 25002, 49902, 50001)))
    if not ok_rep:
        viol("replay file")
    exp("G-REPLAY", "  %s G-REPLAY patches/cvt8_isopath.json sha256 == ISOPATH_SHA, 500 knots" % ("PASS" if ok_rep else "FAIL"))

    names, numel, ops, head = [], [], [], []
    mpath = os.path.join(base, "PARTITION-MANIFEST.txt")
    mtext = open(mpath).read()
    for ln in mtext.split("\n"):
        p = ln.split()
        if p and p[0] == "TENSOR":
            names.append(p[2])
            numel.append(int(p[3]))
            ops.append(p[4])
    msha = sha(mpath)
    isog = groups_of(ISOSPEC, names)
    # the manifest re-built byte for byte: header from the design, TENSOR lines from the file, the rest from 248.2-248.4
    nbn = len(set(n.rsplit(".", 1)[0] for n, o in zip(names, ops) if o == "BatchNorm2d"))
    L = ["NETWORK %s" % NET, "NUM_PARAM_TENSORS %d" % len(names), "TOTAL_PARAMS %d" % sum(numel),
         "NORM BatchNorm2d %d GroupNorm 0" % nbn, "CLIP_C -15:-2.3026", "EPOCHS 100", "META_STEPS %d" % (100 * SPE)]
    L += ["TENSOR %d %s %d %s" % (i + 1, names[i], numel[i], ops[i]) for i in range(len(names))]
    L.append("CARRIERS " + ",".join("%d:%s" % (i, n) for i, n in zip(CARRIER_IDX, CARRIERS)))
    L.append("REPLAY B0 %s MS %s P_HIGH 8609 P_BIG 9428" % (repr(B0), repr(MS)))
    L.append("RESTREPLAY ISOPATH %s SHA256 %s KNOTS %d" % (REPLAY_ID, REPLAY_SHA, len(rp.n)))
    for a in ARMS:
        if a == "k01":
            L.append("ARMSPEC k01 SPEC scalar TYPE scalar M 1")
        else:
            L.append("ARMSPEC %s SPEC %s TYPE blockwise M %d SIZES %s PARAMS %s GROUP1 %s" % (
                a, ISOSPEC, len(isog), ",".join(str(len(g)) for g in isog),
                ",".join(str(sum(numel[i - 1] for i in g)) for g in isog), " ".join(names[i - 1] for i in isog[1])))
        L.append("GROUPHOLD %s %s" % (a, gh_value(a) if a in GHOLD else "off"))
        L.append("RESTHOLD %s %s" % (a, "rec:" + REPLAY_ID if a in FORCED else "off"))
        L.append("WITNESS %s %s" % (a, gh_witness(a)))
        L.append("RWITNESS %s %s" % (a, rh_witness(a, rp)))
    rebuilt_manifest = "\n".join(L) + "\n"
    ok_man = (rebuilt_manifest == mtext and len(names) == NTENS and sum(numel) == TOTPAR
              and [names[i - 1] for i in CARRIER_IDX] == CARRIERS and msha.startswith(MANIFEST_PREFIX)
              and len(mtext.encode()) == MANIFEST_BYTES and [len(g) for g in isog] == [59, 3] and isog[1] == CARRIER_IDX)
    print("    manifest: %d tensors, %d params, %d BatchNorm modules; carriers %s; ISO spec -> sizes %s; %d bytes, sha256 %s...;"
          " re-built byte for byte: %s" % (len(names), sum(numel), nbn, ",".join("%d:%s" % (i, names[i - 1]) for i in CARRIER_IDX),
                                          [len(g) for g in isog], len(mtext.encode()), msha[:16], rebuilt_manifest == mtext))
    if not ok_man:
        viol("manifest")
    prov = {}
    for ln in open(os.path.join(base, "PROVENANCE.txt")):
        p = ln.split(None, 1)
        if len(p) == 2:
            prov.setdefault(p[0], p[1].strip())
    s_sha = sha(scorer)
    ok_prov = (prov.get("MODE") == "submit" and all(prov.get(k, "").startswith(v) for k, v in PREF.items())
               and prov.get("SCORER_SHA256") == s_sha and prov.get("REPLAY_SHA256") == rsha)
    print("    PROVENANCE MODE %s  BUILD %s...  HF %s...  RUNNER %s...  REPLAY %s... == replay file: %s  SCORER %s... == scorer file: %s"
          % (prov.get("MODE"), prov.get("BUILD_NETWORK_SHA256", "")[:8], prov.get("HF_SHA256", "")[:8],
             prov.get("RUNNER_SHA256", "")[:8], prov.get("REPLAY_SHA256", "")[:8], prov.get("REPLAY_SHA256") == rsha,
             prov.get("SCORER_SHA256", "")[:8], prov.get("SCORER_SHA256") == s_sha))
    if not ok_prov:
        viol("provenance")
    exp("G-STRUCT", "  manifest: %s (<runsdir>/cvt8/)" % mpath)
    exp("G-STRUCT", "  PASS G-STRUCT byte-identical to synthetic_manifest_text() (62 tensors, ISO [59,3], both hold tables)"
        "   %d vs %d bytes" % (MANIFEST_BYTES, MANIFEST_BYTES))
    exp("G-PROV", "  provenance: %s (<runsdir>/cvt8/)" % os.path.join(base, "PROVENANCE.txt"))
    exp("G-PROV", "  PASS G-PROV MODE submit   submit")
    exp("G-PROV", "  PASS G-PROV build_network.py == the LIVE one %s...   %s" % (
        prov["BUILD_NETWORK_SHA256"][:12], prov["BUILD_NETWORK_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV HF.py == HF_POST_SHA %s... (cvt8's PATCH_RESTHOLD tree)   %s" % (
        prov["HF_SHA256"][:12], prov["HF_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV RUNNER_SHA256 == cvt8's runner %s...   %s" % (prov["RUNNER_SHA256"][:12], prov["RUNNER_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV REPLAY_SHA256 (the tree's %s.json) == ISOPATH_SHA %s...   %s" % (
        REPLAY_ID, rsha[:12], prov["REPLAY_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV SCORER_SHA256 == this scorer's sha256 %s...   %s" % (s_sha[:12], s_sha[:16]))

    # ---------------------------------------------------------------- [1]
    print("\n[1] completeness and header lines (witnesses built from mode / P / b0 / ms and the replay's knots)")
    files = {}
    for fn in sorted(os.listdir(runsdir)):
        if not (fn.startswith(PREFIX + "-") and fn.endswith(".out")):
            continue
        stem = fn[:-4].split("-")
        if len(stem) != 4 or stem[1] not in ARMS:
            continue
        files.setdefault((stem[1], int(stem[2][1:])), []).append((int(stem[3]), fn))
    R = {}
    envs = set()
    n_pt_ok = 0
    exp("COMPLETE", "COMPLETE  21 runs, epochs 0..99, RUN_DONE, no traceback, finite plateau5")
    for s in SEEDS:
        for a in ARMS:
            lst = files.get((a, s), [])
            if len(lst) != 1:
                viol("%s-s%d has %d .out files" % (a, s, len(lst)))
                continue
            jid, fn = lst[0]
            r = parse_out(os.path.join(runsdir, fn))
            r["jid"] = jid
            R[(a, s)] = r
            ok_ep = sorted(r["ep"]) == list(range(100)) and r["dup"] == 0
            f = flags_of(r["args"][0]) if len(r["args"]) == 1 else []
            keys = [k for k, _ in f]
            fd = dict(f)
            want = dict(FIXED)
            want.update({"stepsize-groups": SPEC[a], "seed": str(s), "run-name": "%s-%s-s%d" % (PREFIX, a, s)})
            wrong = [k for k, v in sorted(want.items()) if fd.get(k) != v]
            rep = sorted(set(k for k in keys if keys.count(k) > 1))
            env = [" ".join(t for t in e.split() if not t.startswith("PROBE_DIR=")) for e in r["env"]]
            for e in env:
                envs.add(e)
            ptw = "PROBE_TENSOR: on every=100 type=%s tensors=62 " % ("scalar" if a == "k01" else "blockwise")
            pt_ok = len(r["pt"]) == 1 and r["pt"][0].startswith(ptw)
            n_pt_ok += pt_ok
            complete = ok_ep and r["done"] and r["tb"] == 0
            args_ok = len(r["args"]) == 1 and not wrong and not rep
            vw_ok = r["vw"] == ["VOTE_W: off"]
            bh_ok = r["bh"] == ["BETA_HOLD: off"]
            ch_ok = r["ch"] == []
            gh_ok = r["gh"] == [gh_witness(a)]
            rh_ok = r["rh"] == [rh_witness(a, rp)]
            line_ok = complete and args_ok and env == [ENV_WANT] and pt_ok and vw_ok and bh_ok and ch_ok and gh_ok and rh_ok
            print("  %-4s %-11s s%-3d job %d  epochs %d  RUN_DONE %s  tb %d  ARGS wrong %s rep %s  ENV %s  PT %s  VOTE_W %s"
                  "  BETA_HOLD %s  no COMP_HOLD %s  GROUP_HOLD %s  REST_HOLD %s"
                  % ("OK" if line_ok else "BAD", a, s, jid, len(r["ep"]), r["done"], r["tb"], wrong or "-", rep or "-",
                     env == [ENV_WANT], pt_ok, vw_ok, bh_ok, ch_ok, gh_ok, rh_ok))
            if not line_ok:
                viol("%s-s%d header/completeness" % (a, s))
            if args_ok:
                exp("G-ARGS", "  PASS G-ARGS %s-s%d" % (a, s))
            for ok_, lab in ((vw_ok, "G-VOTEW"), (bh_ok, "G-BHOLD"), (ch_ok, "G-CHOLD"), (gh_ok, "G-GHOLD"), (rh_ok, "G-RHOLD")):
                if ok_:
                    exp(lab, "  PASS %s %s-s%d" % (lab, a, s))
    if len(R) != 21:
        print("\nVIOLATIONS %d (incomplete batch; stopping)" % len(bad))
        sys.exit(1)
    if envs == {ENV_WANT} and all(len(R[k]["env"]) == 1 for k in R):
        exp("G-ENV", "  PASS G-ENV one distinct ENV line, one per run   1 distinct")
        exp("G-ENV", "        " + ENV_WANT)
        exp("G-ENV", "  PASS G-ENV the ENV line is ciso1's / cdep1's / cvt7's, byte for byte (PROBE_DIR stripped)")
    if n_pt_ok == 21:
        exp("G-ENV", "  PASS G-ENV every run printed ONE `PROBE_TENSOR: on every=100 type=<its arm's> tensors=62`   21 runs")

    # ---------------------------------------------------------------- [2]
    print("\n[2] levels (plateau5 = mean TEST epochs 95..99; TRAIN5 beside)")
    P, T = {}, {}
    for a in ARMS:
        v, t, sl = [], [], []
        for s in SEEDS:
            ep = R[(a, s)]["ep"]
            v.append(mean([ep[e][1] for e in range(95, 100)]))
            t.append(mean([ep[e][0] for e in range(95, 100)]))
            xs = list(range(80, 100))
            sl.append(ols(xs, [ep[e][1] for e in xs]))
            r = R[(a, s)]
            exp("COMPLETE", "  OK   %s-%s-s%d  %d epoch lines, RUN_DONE=%s, traceback=%s, plateau5 %.4f (train %.4f)"
                % (PREFIX, a, s, len(r["ep"]), r["done"], r["tb"] > 0, v[-1], t[-1]))
        P[a], T[a] = v, t
        print("  %-11s TEST %.4f (sd %.4f, range %.4f)  TRAIN %.4f  seeds %s  train %s  slope80-99 %s"
              % (a, mean(v), sdev(v), max(v) - min(v), mean(t), " / ".join("%.4f" % x for x in v),
                 " / ".join("%.4f" % x for x in t), " / ".join("%+.3f" % x for x in sl)))
        exp("LEVELS", "  %-12s TEST %.4f  sd %.4f  range %.4f   TRAIN %.4f   tail slope %s pp/ep"
            % (a, mean(v), sdev(v), max(v) - min(v), mean(t), "/".join("%.3f" % x for x in sl)))
        exp("LEVELS", "               seeds: %s" % ", ".join("s%d %.4f (train %.4f)" % (s, x, y) for s, x, y in zip(SEEDS, v, t)))
    M = dict((a, mean(P[a])) for a in ARMS)
    TM = dict((a, mean(T[a])) for a in ARMS)
    hi = max(M.values())
    exp("G-FLOOR", "  PASS G-FLOOR max arm mean >= 15.00 pp (chance 1.00)   max %.4f" % hi)
    exp("G-CEIL", "  PASS G-CEIL max arm mean <= 90.00 pp   max %.4f" % hi)

    # ---------------------------------------------------------------- [3]
    print("\n[3] sigma and contrasts (within batch)")
    ss = math.fsum((x - M[a]) ** 2 for a in ARMS for x in P[a])
    df = len(ARMS) * (len(SEEDS) - 1)
    sig_in = math.sqrt(ss / df)
    sig = max(SIGMA_PRIOR, sig_in)
    which = "SIGMA_PRIOR_frozen" if sig == SIGMA_PRIOR else "SIGMA_INBATCH"
    se = sig * math.sqrt(2.0 / 3.0)
    print("  SIGMA_INBATCH %.6f (df %d)  SIGMA_PRIOR %.6f  -> used %.6f (%s)  SE %.6f" % (sig_in, df, SIGMA_PRIOR, sig, which, se))
    exp("SIGMA", "  %-20s %.6f" % ("SIGMA_PRIOR_frozen", SIGMA_PRIOR))
    exp("SIGMA", "  %-20s %.6f" % ("SIGMA_INBATCH", sig_in))
    exp("SIGMA", "  SIGMA_USED           %.6f  (= %s)" % (sig, which))
    exp("SIGMA", "  SE_ARM_DIFF          %.6f  = SIGMA_USED * sqrt(2/3)  (df_inbatch %d)" % (se, df))
    C = [("P_DOSE", "HOLDHIGH", "HOLDBIG", "PRIMARY"), ("P_ROUTE", "HIGHISOPATH", "HOLDHIGH", "PRIMARY"),
         ("P_ROUTE_BIG", "BIGISOPATH", "HOLDBIG", "PRIMARY"), ("D_ISO", "ISO", "k01", "KEY"),
         ("P_CTLC", "ISO", "LOWISOPATH", "KEY"), ("R_HIGH", "ISO", "HOLDHIGH", "KEY"),
         ("P_DOSE_PATH", "HIGHISOPATH", "BIGISOPATH", "KEY")]
    CT = {}
    for lab, x, y, kind in C:
        d, dt = M[x] - M[y], TM[x] - TM[y]
        CT[lab] = (d, dt)
        print("  %-8s %-12s = %-11s - %-11s TEST %+9.4f pp = %+8.2f SE   TRAIN %+9.4f   per seed %s"
              % (kind, lab, x, y, d, d / se, dt, " / ".join("%+.4f" % (P[x][j] - P[y][j]) for j in range(3))))
        exp("CONTRASTS", "  %-8s %-12s = %-11s - %-11s = %+.4f pp = %+.2f SE   (TRAIN %+.4f)" % (kind, lab, x, y, d, d / se, dt))
    for a in HELD:
        d, dt = M[a] - M["k01"], TM[a] - TM["k01"]
        CT["D_" + a] = (d, dt)
        print("  KEY      D_%-10s = %-11s - k01         TEST %+9.4f pp = %+8.2f SE   TRAIN %+9.4f   per seed %s"
              % (a, a, d, d / se, dt, " / ".join("%+.4f" % (P[a][j] - P["k01"][j]) for j in range(3))))
        exp("CONTRASTS", "  KEY      D_%-11s = %-11s - k01 = %+.4f pp = %+.2f SE   (TRAIN %+.4f)   ISO - %s = %+.4f"
            % (a, a, d, d / se, dt, a, M["ISO"] - M[a]))
    pint = CT["P_ROUTE_BIG"][0] - CT["P_ROUTE"][0]
    print("  DESCRIPTIVE P_INT = P_ROUTE_BIG - P_ROUTE = %+.4f pp (TRAIN %+.4f)" % (pint, CT["P_ROUTE_BIG"][1] - CT["P_ROUTE"][1]))
    exp("CONTRASTS", "  DESCRIPTIVE P_INT = P_ROUTE_BIG - P_ROUTE = %+.4f pp" % pint)
    exp("CONTRASTS", "  bars: GAP_MIN %.1f (%.2f SE) | MATCH %.1f pp (%.2f SE) | NULL %.1f pp (%.2f SE) | REP_MARGIN %.1f pp (%.2f SE) | RESCUE %.1f pp"
        % (GAP_MIN, GAP_MIN / se, MATCH, MATCH / se, NULL, NULL / se, REP, REP / se, RESCUE))

    # ---------------------------------------------------------------- [4]
    print("\n[4] G-BITE re-derived from the raw probe records (248.6 (a)-(e)), the vote-out decomposition, non-vacuity")
    RECS = {}
    bite_ok_all = True
    nonfin = {}
    cross = {}
    exp("G-BITE", "  (a) 500 records; (b) every UNHELD group == Lion recomputed from its own beta_pre/mom_pre/z_agg (tol 1e-05);")
    exp("G-BITE", "  (c) carrier group 1 of every held arm: beta == hold(step+2), beta_pre == hold(step+1); complement group 0 of every")
    exp("G-BITE", "      forced arm: the same against ISOPATH; (d) gh_* / rh_* records audit; (e) no hold key where none is registered,")
    exp("G-BITE", "      no bh_* / ch_* key anywhere.  Non-finite records are counted and stamped, not failed.")
    CMODES = [("floor", ("floor", None)), ("tri:8609", ("tri", 8609)), ("tri:9428", ("tri", 9428))]
    for s in SEEDS:
        for a in ARMS:
            pdir = os.path.join(base, "probe_%s-%s-s%d" % (PREFIX, a, s))
            gs = groups_of(SPEC[a], names)
            ng = len(gs)
            recs = load_recs(pdir)
            RECS[(a, s)] = recs
            ptj = json.load(open(os.path.join(pdir, "probe_tensor.json")))
            bsz = json.load(open(os.path.join(pdir, "block_sizes.json")))
            meta_ok = (ptj.get("n_tensors") == NTENS and ptj.get("param_numels") == numel
                       and bsz.get("n_b") == [sum(numel[i - 1] for i in g) for g in gs])
            held, forced = a in GHOLD, a in FORCED
            c = dict(n=0, nonfin=0, badrec=0, badstep=0, lion=0, ties=0, sched=0, schedpre=0, rsched=0, rschedpre=0,
                     gh=0, rh=0, keys=0, gact=0, ract=0)
            wl = ws = wr = 0.0
            last_g = last_r = None
            prev_g = prev_r = -1
            worst_dec = 0.0
            for k, x in enumerate(recs):
                if (x is None or type(x.get("step")) is not int or not isinstance(x.get("beta"), list)
                        or len(x["beta"]) != ng):
                    c["badrec"] += 1
                    continue
                try:
                    beta = [float(v) for v in x["beta"]]
                    bp = [float(v) for v in x["beta_pre"][0]]
                    mo = [float(v) for v in x["mom_pre"][0]]
                    za = [float(v) for v in x["z_agg"][0]]
                except (KeyError, TypeError, ValueError, IndexError):
                    c["badrec"] += 1
                    continue
                if not (len(x["beta_pre"]) == 1 and len(x["mom_pre"]) == 1 and len(x["z_agg"]) == 1
                        and len(bp) == ng and len(mo) == ng and len(za) == ng):
                    c["badrec"] += 1
                    continue
                if not all(math.isfinite(v) for v in beta + bp + mo + za):
                    c["nonfin"] += 1
                    continue
                st = x["step"]
                c["badstep"] += st != 100 * k
                c["n"] += 1
                n = st + 2
                has_b = any(key in x for key in ("bh_n", "bh_active", "bh_nat", "bh_held", "ch_n", "ch_active", "ch_nat", "ch_held"))
                has_g = any(key in x for key in ("gh_n", "gh_active", "gh_nat", "gh_held"))
                has_r = any(key in x for key in ("rh_n", "rh_active", "rh_nat", "rh_held"))
                c["keys"] += has_b
                if not held:
                    c["keys"] += has_g or has_r
                elif not forced:
                    c["keys"] += has_r
                free = list(range(ng)) if not held else ([] if forced else [0])
                for g in free:
                    nat, tie = lion(bp[g], mo[g], za[g])
                    if tie:
                        c["ties"] += 1
                        continue
                    e = abs(nat - beta[g])
                    wl = max(wl, e)
                    c["lion"] += e > BH_TOL
                zt = x["z_tensor"]
                for g, grp in enumerate(gs):
                    zs = math.fsum(zt[i - 1] for i in grp)
                    den = math.fsum(abs(zt[i - 1]) for i in grp) + 1e-30
                    worst_dec = max(worst_dec, abs(zs - za[g]) / den)
                if ng == 2:
                    for lab, (mode, Pp) in CMODES:
                        e1, e2 = abs(beta[1] - carrier_at(mode, Pp, n)), abs(bp[1] - carrier_at(mode, Pp, n - 1))
                        cc = cross.setdefault((a, s, lab), [0, 0.0])
                        cc[0] += (e1 > BH_TOL) or (e2 > BH_TOL)
                        cc[1] = max(cc[1], e1, e2)
                    e1, e2 = abs(beta[0] - rp.at(n)), abs(bp[0] - rp.at(n - 1))
                    cc = cross.setdefault((a, s, "ISOPATH"), [0, 0.0])
                    cc[0] += (e1 > BH_TOL) or (e2 > BH_TOL)
                    cc[1] = max(cc[1], e1, e2)
                if not held:
                    continue
                mode, Pp = GHOLD[a]
                e1, e2 = abs(beta[1] - carrier_at(mode, Pp, n)), abs(bp[1] - carrier_at(mode, Pp, n - 1))
                ws = max(ws, e1, e2)
                c["sched"] += e1 > BH_TOL
                c["schedpre"] += e2 > BH_TOL
                audits = [("gh", 1)] + ([("rh", 0)] if forced else [])
                if forced:
                    e1, e2 = abs(beta[0] - rp.at(n)), abs(bp[0] - rp.at(n - 1))
                    wr = max(wr, e1, e2)
                    c["rsched"] += e1 > BH_TOL
                    c["rschedpre"] += e2 > BH_TOL
                for pf, g in audits:
                    try:
                        an, aa, anat, aheld = int(x[pf + "_n"]), int(x[pf + "_active"]), float(x[pf + "_nat"]), float(x[pf + "_held"])
                    except (KeyError, TypeError, ValueError):
                        c[pf] += 1
                        continue
                    natg, tieg = lion(bp[g], mo[g], za[g])
                    prev = prev_g if pf == "gh" else prev_r
                    good = (an == n and abs(aheld - beta[g]) <= BH_TOL and 0 <= aa <= an and aa >= prev
                            and (tieg or abs(anat - natg) <= BH_TOL))
                    c[pf] += not good
                    if pf == "gh":
                        c["gact"] += abs(anat - aheld) > BH_TOL
                        prev_g, last_g = max(prev_g, aa), aa
                    else:
                        c["ract"] += abs(anat - aheld) > BH_TOL
                        prev_r, last_r = max(prev_r, aa), aa
            ok = (meta_ok and len(recs) == N_REC and c["badrec"] == 0 and c["n"] + c["nonfin"] == N_REC
                  and c["badstep"] == 0 and c["lion"] == 0 and c["keys"] == 0)
            if held:
                ok = ok and c["sched"] == 0 and c["schedpre"] == 0 and c["gh"] == 0 and (last_g is None or last_g >= c["gact"])
            if forced:
                ok = ok and c["rsched"] == 0 and c["rschedpre"] == 0 and c["rh"] == 0 and (last_r is None or last_r >= c["ract"])
            nonfin[(a, s)] = c["nonfin"]
            bite_ok_all = bite_ok_all and ok
            print("  %-4s %-11s s%-3d meta %s recs %d lion-mism %d ties %d worst %.1e | sched %d/%d worst %.1e | rsched %d/%d worst %.1e"
                  " | gh-audit %d rh-audit %d keys-wrong %d | active records g %d r %d | gh_active %s rh_active %s | decomposition worst rel %.1e"
                  % ("PASS" if ok else "FAIL", a, s, meta_ok, len(recs), c["lion"], c["ties"], wl, c["sched"], c["schedpre"], ws,
                     c["rsched"], c["rschedpre"], wr, c["gh"], c["rh"], c["keys"], c["gact"], c["ract"], last_g, last_r, worst_dec))
            if not ok:
                viol("G-BITE %s-s%d" % (a, s))
            if worst_dec > 1e-4:
                viol("decomposition %s-s%d" % (a, s))
            exp("G-BITE", "  %-4s G-BITE %-12s s%-3d records %s  bad %d  nonfinite %d  lion %d (ties %d, worst %.1e)  sched %d/%d (worst %.1e)"
                "  rsched %d/%d (worst %.1e)  gh-audit %d  rh-audit %d  keys-wrong %d  gh_active %s  rh_active %s"
                % ("PASS" if ok else "FAIL", a, s, len(recs), c["badrec"], c["nonfin"], c["lion"], c["ties"], wl, c["sched"],
                   c["schedpre"], ws, c["rsched"], c["rschedpre"], wr, c["gh"], c["rh"], c["keys"], last_g, last_r))
    print("  NON-VACUITY cross-read: records (of 500) on which a group FAILS a schedule (beta or beta_pre); max |error|")
    cols = [lab for lab, _m in CMODES] + ["ISOPATH"]
    print("  %-16s %s" % ("run", "  ".join("%-20s" % ("g1 as " + l if l != "ISOPATH" else "g0 as ISOPATH") for l in cols)))
    for a in ARMS[1:]:
        for s in SEEDS:
            print("  %-11s s%-3d %s" % (a, s, "  ".join("%3d  max %9.4f    " % tuple(cross[(a, s, l)]) for l in cols)))
    own = dict((a, ("floor" if GHOLD[a][0] == "floor" else "tri:%d" % GHOLD[a][1])) for a in GHOLD)
    nv_own = all(cross[(a, s, own[a])][0] == 0 for a in GHOLD for s in SEEDS) and all(
        cross[(a, s, "ISOPATH")][0] == 0 for a in FORCED for s in SEEDS)
    nv_other = min(cross[(a, s, l)][0] for a in ARMS[1:] for s in SEEDS for l, _m in CMODES if own.get(a) != l)
    nv_free = min(cross[(a, s, "ISOPATH")][0] for a in ("ISO", "HOLDHIGH", "HOLDBIG") for s in SEEDS)
    nv_free_max = dict((a, [cross[(a, s, "ISOPATH")][1] for s in SEEDS]) for a in ("ISO", "HOLDHIGH", "HOLDBIG"))
    print("  every held run passes its own carrier mode (and each forced run ISOPATH) on all 500: %s; fewest failing records"
          " for a carrier mode not its own: %d; fewest for a FREE complement (ISO / HOLDHIGH / HOLDBIG) read as ISOPATH: %d"
          % (nv_own, nv_other, nv_free))
    print("  the broken-REST_HOLD nulls (a forced arm's free twin read as ISOPATH), max |error| per seed: %s"
          % ";  ".join("%s %s" % (a, " / ".join("%.3f" % x for x in nv_free_max[a])) for a in ("ISO", "HOLDHIGH", "HOLDBIG")))
    if not (nv_own and nv_other > 0 and nv_free > 0):
        viol("non-vacuity cross-read")

    # ---------------------------------------------------------------- [5]
    print("\n[5] branch, states, stamps (248.6, re-typed; first match wins)")
    div = [a for a in ARMS if max(P[a]) - min(P[a]) > DIVERGED]
    gap = M["ISO"] - M["k01"]

    def st_of(L, m=M):
        if L >= m["ISO"] - MATCH:
            return "AT-ISO"
        if L <= m["k01"] + NULL:
            return "AT-K01"
        return "BETWEEN"

    def route(L, ref, pfx, m=M):
        if ref > m["ISO"] - REP:
            return pfx + "-UNREADABLE"
        if L >= m["ISO"] - MATCH:
            return pfx + "-VIA"
        if L <= ref + MATCH:
            return pfx + "-DIRECT"
        return pfx + "-PARTIAL"

    def dose(m=M):
        if m["HOLDBIG"] <= m["k01"] + NULL:
            return "DOSE-FULL"
        if abs(m["HOLDBIG"] - m["HOLDHIGH"]) <= MATCH:
            return "NO-DOSE"
        if m["HOLDBIG"] > m["HOLDHIGH"] + MATCH:
            return "DOSE-REVERSED"
        return "DOSE-PARTIAL"

    def branch_of(m, dv):
        if dv:
            return "UNRESOLVED-DIVERGED"
        if m["k01"] > K01_MAX:
            return "SCALAR-NOT-COLLAPSED"
        if m["ISO"] - m["k01"] < GAP_MIN:
            return "POSITIVE-CONTROL-FAILED"
        sl = st_of(m["LOWISOPATH"], m)
        if sl == "AT-K01":
            return "COMP-HOLD-CONTROL-FAILED"
        if sl == "BETWEEN":
            return "COMP-HOLD-CONTROL-PARTIAL"
        if not (m["k01"] + REP <= m["HOLDHIGH"] <= m["ISO"] - REP):
            return "REPLICATE-FAILED"
        return "%s+%s+%s" % (dose(m), route(m["HIGHISOPATH"], m["HOLDHIGH"], "ROUTE", m),
                             route(m["BIGISOPATH"], m["HOLDBIG"], "BIGROUTE", m))

    if not bite_ok_all:
        branch = "PATCH-NOT-VERIFIED"
    elif not (FLOOR_MIN <= hi <= CEIL_MAX):
        branch = "HARNESS-UNSOUND"
    else:
        branch = branch_of(M, bool(div))
    states = dict((a, st_of(M[a])) for a in HELD)
    stamps = ["HARNESS-CLEAN", "PATCH-BITES", "HOLD-FROM-INIT", "REPLAY-MAX-RATE-TRIANGLE", "BIG-IS-PLAINNET-DOSE",
              "ISOPATH-REPLAY-FILE", "FORCED-ARMS-OPEN-LOOP", "CARRIER-GROUP-OF-3", "ONE-NETWORK-RESNET", "HORIZON-100-ONLY",
              "SIGMA-PRIOR-FROZEN" if which == "SIGMA_PRIOR_frozen" else "SIGMA-INBATCH"]
    if div:
        stamps.append("DIVERGED-" + "-".join(div))
    stamps.append("POSITIVE-CONTROL-REPRODUCES" if gap >= GAP_MIN else "POSITIVE-CONTROL-FAILED")
    if gap >= GAP_MIN:
        stamps += ["%s-%s" % (a, states[a]) for a in HELD]
        if branch in ("COMP-HOLD-CONTROL-FAILED", "COMP-HOLD-CONTROL-PARTIAL", "REPLICATE-FAILED"):
            stamps.append("UNBRANCHED-" + dose())
    for a in HELD:
        if M[a] - M["ISO"] > MATCH:
            stamps.append(a + "-ABOVE-ISO")
        if M[a] - M["k01"] < -NULL:
            stamps.append(a + "-BELOW-K01")
        if any(nonfin[(a, s)] for s in SEEDS):
            stamps.append("NONFINITE-" + a)
    for x, y in (("HIGHISOPATH", "HOLDHIGH"), ("BIGISOPATH", "HOLDBIG")):
        if M[x] < M[y] - MATCH:
            stamps.append("%s-BELOW-%s" % (x, y))
    if any(abs(M[a] - M["k01"]) <= NULL for a in HELD):
        stamps.append("FLOOR-READINGS-ARE-BOUNDS")
    agree = all((CT[l][1] > 0) == (CT[l][0] > 0) for l in ("P_DOSE", "P_ROUTE", "P_ROUTE_BIG") if abs(CT[l][0]) >= RESCUE)
    stamps.append("TRAIN-AGREES" if agree else "TRAIN-DISAGREES")
    final = "FINAL: %s | %s" % (branch, " | ".join(stamps))
    print("  states: %s" % "  ".join("%s %s" % (a, states[a]) for a in HELD))
    print("  " + final)
    exp("G-DIVERGE", "  %s every arm's seed range <= 5.0 pp" % ("PASS" if not div else "FAIL"))
    exp("FINAL", final)
    print("  bars: AT-ISO (ISO - 5) = %.4f;  AT-K01 (k01 + 2) = %.4f;  replicate window [k01 + 10, ISO - 10] = [%.4f, %.4f];"
          "  ROUTE-DIRECT (HOLDHIGH + 5) = %.4f;  BIGROUTE-DIRECT (HOLDBIG + 5) = %.4f;  BIGROUTE-UNREADABLE iff HOLDBIG > %.4f"
          % (M["ISO"] - MATCH, M["k01"] + NULL, M["k01"] + REP, M["ISO"] - REP, M["HOLDHIGH"] + MATCH, M["HOLDBIG"] + MATCH,
             M["ISO"] - REP))
    wr_arm = max(ARMS, key=lambda a: max(P[a]) - min(P[a]))
    margins = [("D_ISO over GAP_MIN", gap - GAP_MIN), ("k01 under K01_MAX", K01_MAX - M["k01"]),
               ("worst seed range under DIVERGED (%s)" % wr_arm, DIVERGED - (max(P[wr_arm]) - min(P[wr_arm]))),
               ("LOWISOPATH over AT-ISO bar", M["LOWISOPATH"] - (M["ISO"] - MATCH)),
               ("HOLDHIGH over k01 + 10", M["HOLDHIGH"] - (M["k01"] + REP)),
               ("HOLDHIGH under ISO - 10", M["ISO"] - REP - M["HOLDHIGH"]),
               ("DOSE-FULL: (k01 + 2) - HOLDBIG", M["k01"] + NULL - M["HOLDBIG"]),
               ("  |HOLDBIG - HOLDHIGH| over NO-DOSE's 5", abs(M["HOLDBIG"] - M["HOLDHIGH"]) - MATCH),
               ("ROUTE: HIGHISOPATH - (HOLDHIGH + 5)  (>0 not DIRECT)", M["HIGHISOPATH"] - (M["HOLDHIGH"] + MATCH)),
               ("ROUTE: (ISO - 5) - HIGHISOPATH       (>0 not VIA)", M["ISO"] - MATCH - M["HIGHISOPATH"]),
               ("BIGROUTE: (HOLDBIG + 5) - BIGISOPATH (>=0 DIRECT)", M["HOLDBIG"] + MATCH - M["BIGISOPATH"]),
               ("BIGROUTE: (ISO - 10) - HOLDBIG       (>=0 readable)", M["ISO"] - REP - M["HOLDBIG"])]
    for k, v in margins:
        print("    margin %-54s %+9.4f pp = %+.2f SE" % (k, v, v / se))
    print("  per-seed readings (descriptive; the branch reads arm means):")
    for a in HELD:
        print("    %-11s %s" % (a, " / ".join("s%d %.4f %s" % (s, P[a][j], st_of(P[a][j])) for j, s in enumerate(SEEDS))))
    for lab, x, y, _k in C[:3]:
        print("    per-seed paired %-11s (%s - %s, same seed) %s" % (lab, x, y, " / ".join("%+.4f" % (P[x][j] - P[y][j]) for j in range(3))))
    print("  per-seed words, each seed's own seven runs read through the registered map (descriptive):")
    for j, s in enumerate(SEEDS):
        ms = dict((a, P[a][j]) for a in ARMS)
        print("    s%d  %s" % (s, branch_of(ms, False)))
    print("  registered accounts (248.5) -- misses of each arm's predicted band (0 = inside); hits of 7:")

    def acc(hb, hip, bip, low=FREE_PL, hh=HIGH_PL):
        return [("k01", K01_PL), ("ISO", FREE_PL), ("HOLDHIGH", hh), ("HOLDBIG", hb), ("HIGHISOPATH", hip),
                ("BIGISOPATH", bip), ("LOWISOPATH", low)]

    ACC = [("DOSE x DIRECT", acc(K01_PL, HIGH_PL, K01_PL)), ("DOSE x VIA", acc(K01_PL, FREE_PL, FREE_PL)),
           ("DOSE x PARTIAL-VIA", acc(K01_PL, PV_PL, PART_PL)), ("DOSE, DIRECT ONLY AT BIG", acc(K01_PL, FREE_PL, K01_PL)),
           ("NO-DOSE x DIRECT", acc(HIGH_PL, HIGH_PL, HIGH_PL)), ("NO-DOSE x VIA", acc(HIGH_PL, FREE_PL, FREE_PL)),
           ("NO-DOSE x PARTIAL-VIA", acc(HIGH_PL, PV_PL, PV_PL)), ("PARTIAL-DOSE x DIRECT", acc(PART_PL, HIGH_PL, PART_PL)),
           ("PARTIAL-DOSE x VIA", acc(PART_PL, FREE_PL, FREE_PL)), ("COMP-HOLD-BREAKS", acc(HIGH_PL, K01_PL, K01_PL, low=K01_PL)),
           ("HOLDHIGH-DOES-NOT-REPLICATE", acc(K01_PL, K01_PL, K01_PL, hh=K01_PL)),
           ("BROKEN-REST_HOLD (null)", acc(HIGH_PL, HIGH_PL, HIGH_PL))]
    for nm, bands in ACC:
        ms_ = [(a, max(0.0, b[0] - M[a], M[a] - b[1])) for a, b in bands]
        print("    %-28s hits %d  %s" % (nm, sum(1 for _a, v in ms_ if v == 0.0), "  ".join("%s %.2f" % kv for kv in ms_)))
    fl = [ln for ln in logl if ln.startswith("FINAL:")]
    print("  scorer log FINAL == this FINAL (whole string): %s" % (fl == [final]))
    if fl:
        toks = [t.strip() for t in fl[-1][len("FINAL:"):].split("|")]
        mine = [branch] + stamps
        print("  branch token equal: %s;  token lists equal: %s (scorer %d tokens, here %d)"
              % (toks[0] == branch, toks == mine, len(toks), len(mine)))
        if toks != mine:
            viol("FINAL tokens differ: scorer %s / here %s" % (sorted(set(toks) - set(mine)), sorted(set(mine) - set(toks))))
    else:
        viol("no FINAL line in the scorer log")
    exp("BETWEEN", "BETWEEN-BATCH, NON-GATING (frozen literals; NO other batch enters any cvt8 bar or contrast):")
    for b, kv in BETWEEN:
        exp("BETWEEN", "  %s: %s" % (b, "  ".join("%s %.4f" % x for x in kv)))
    exp("BETWEEN", "  cvt8 {102,103,104}: %s" % "  ".join("%s %.4f" % (a, M[a]) for a in ARMS))

    # ---------------------------------------------------------------- [6]
    print("\n[6] the scorer's descriptive readouts, re-derived")
    exp("DESCRIPTIVE", "DESCRIPTIVE, NON-GATING -- beta from the runs' own probe.jsonl; r = exp(beta + 15) over epochs 15-30 / 30-60;"
        " pin = first r <= 2 for good")
    SUMM = {}
    for a in ARMS:
        for s in SEEDS:
            recs = RECS[(a, s)]
            parts = []
            ng = 1 if a == "k01" else 2
            for k in range(ng):
                pk, pke, pin, r1, r2 = summary(recs, k)
                SUMM[(a, s, k)] = (pk, pke, pin, r1, r2)
                lab = "beta" if ng == 1 else ("cmpl" if k == 0 else "carr")
                parts.append("%s peak %.3f @%.1f pin %s r %.0f/%.0f" % (lab, pk, pke, "never" if pin is None else "%.1f" % pin, r1, r2))
            ga = [x["gh_active"] for x in recs if "gh_active" in x]
            ra = [x["rh_active"] for x in recs if "rh_active" in x]
            line = "  %-12s s%d  %s  gh_active %s rh_active %s" % (a, s, " | ".join(parts), ga[-1] if ga else "-", ra[-1] if ra else "-")
            print(line)
            exp("DESCRIPTIVE", line)
    EPL = (9, 14, 15, 16, 17, 18, 19, 24, 29, 39, 49, 99)
    exp("DESCRIPTIVE", "DESCRIPTIVE, NON-GATING -- arm-mean TEST (TRAIN) at epoch lines %s" % " / ".join(str(e) for e in EPL))
    AMT = dict((a, [mean([R[(a, s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]) for a in ARMS)
    AMR = dict((a, [mean([R[(a, s)]["ep"][e][0] for s in SEEDS]) for e in range(100)]) for a in ARMS)
    for a in ARMS:
        exp("DESCRIPTIVE", "  %-12s %s" % (a, " / ".join("%.1f (%.1f)" % (AMT[a][e], AMR[a][e]) for e in EPL)))
    for x, y in (("ISO", "HOLDHIGH"), ("HOLDHIGH", "HOLDBIG"), ("HIGHISOPATH", "HOLDHIGH"), ("BIGISOPATH", "HOLDBIG"),
                 ("ISO", "LOWISOPATH")):
        fe = first_stays([abs(AMT[x][e] - AMT[y][e]) for e in range(100)], lambda g: g > 2.0)
        line = "  first epoch line from which |arm-mean TEST %s - %s| stays > 2 pp: %s" % (x, y, "never" if fe is None else fe)
        print(line)
        exp("DESCRIPTIVE", line)

    # ---------------------------------------------------------------- [7]
    print("\n[7] LINE-BY-LINE AGREEMENT with the scorer log (each re-built line looked up verbatim)")
    logset = set(logl)
    by = {}
    rebuilt = set()
    for sec, line in expect:
        hit = line in logset
        by.setdefault(sec, [0, 0])
        by[sec][0 if hit else 1] += 1
        if hit:
            rebuilt.add(line)
        else:
            viol("line not in scorer log [%s]: %r" % (sec, line if "<runsdir>" not in line else "(host-path line)"))
    SECS = ("HEADER", "CORPUS", "COMPLETE", "G-ARGS", "G-ENV", "G-VOTEW", "G-BHOLD", "G-CHOLD", "G-GHOLD", "G-RHOLD", "G-STRUCT",
            "G-PROV", "G-REPLAY", "LEVELS", "G-FLOOR", "G-CEIL", "G-BITE", "SIGMA", "CONTRASTS", "G-DIVERGE", "FINAL", "BETWEEN",
            "DESCRIPTIVE")
    for sec in SECS:
        h, m = by.get(sec, [0, 0])
        print("  %-12s %3d lines found verbatim, %d not found" % (sec, h, m))
    n_fail = sum(1 for ln in logl if ln.lstrip().startswith("FAIL"))
    print("  total %d re-built lines (%d distinct), %d found; FAIL lines in the scorer log: %d"
          % (len(expect), len(set(l for _s, l in expect)), sum(v[0] for v in by.values()), n_fail))
    rest = [ln for ln in logl if ln not in rebuilt]
    blank = sum(1 for ln in rest if not ln.strip())
    rules = sum(1 for ln in rest if ln.strip() and set(ln.strip()) == {"="})
    heads = sum(1 for ln in rest if ln.strip() and not ln.startswith(" ") and set(ln.strip()) != {"="})
    prose = len(rest) - blank - rules - heads
    print("  scorer log: %d lines; %d are re-built lines; not re-built: %d blank, %d '=' rules, %d unindented section titles,"
          " %d indented prose lines (the G-VOTEW title's second line, the registered licence and not-licensed text)"
          % (len(logl), len(logl) - len(rest), blank, rules, heads, prose))
    for ln in rest:
        if ln.strip() and not ln.startswith(" ") and set(ln.strip()) != {"="}:
            print("    title not re-built: %r" % ln[:110])
    for ln in rest:
        if ln.strip() and ln.startswith(" ") and any(ch.isdigit() for ch in ln):
            print("    indented, not re-built, carries a digit: %r" % ln[:110])

    # ================================================================ DESCRIPTIVE
    print("\n" + "=" * 100)
    print("DESCRIPTIVE, NOT REGISTERED -- nothing below can move a level, contrast, gate, branch or stamp above")
    print("=" * 100)
    med = {}
    for a in ARMS:
        order = sorted(range(3), key=lambda j: P[a][j])
        med[a] = SEEDS[order[1]]

    print("\n[D1] TEST (TRAIN) after 15 / 17 / 19 / 30 / 50 / 100 completed epochs = Epoch lines 14 / 16 / 18 / 29 / 49 / 99"
          " (line n = after n+1 epochs): arm mean, then TEST per seed")
    DL = (14, 16, 18, 29, 49, 99)
    for a in ARMS:
        cells = ["%.2f (%.2f)" % (AMT[a][e], AMR[a][e]) for e in DL]
        per = ["s%d %s" % (s, "/".join("%.2f" % R[(a, s)]["ep"][e][1] for e in DL)) for s in SEEDS]
        print("  %-11s %s   | %s" % (a, "  ".join(cells), "  ".join(per)))

    EP = (10, 17, 20, 30, 40)
    print("\n[D2] APPLIED beta: the probe record with step = 500e (record index 5e; for a held group it equals the schedule at"
          " n = 500e + 2, 248.6 (c)); c = complement (group 0), g = carrier group (group 1); k01 one shared beta")
    print("  MEDIAN seed per arm (by plateau5): %s" % "  ".join("%s s%d" % (a, med[a]) for a in ARMS))
    print("  %-11s %s" % ("arm", "  ".join("%-19s" % ("epoch %d" % e) for e in EP)))
    for a in ARMS:
        cells = []
        for e in EP:
            x = RECS[(a, med[a])][5 * e]
            if x["step"] != SPE * e:
                viol("record index %d is not epoch %d" % (5 * e, e))
            cells.append("%-19s" % (("%.3f" % x["beta"][0]) if a == "k01" else ("c %.3f / g %.3f" % (x["beta"][0], x["beta"][1]))))
        print("  %-11s %s" % (a, "  ".join(cells)))
    print("  arm mean of the three seeds:")
    for a in ARMS:
        cells = []
        for e in EP:
            c0 = mean([RECS[(a, s)][5 * e]["beta"][0] for s in SEEDS])
            cells.append("%-19s" % (("%.3f" % c0) if a == "k01" else
                                   ("c %.3f / g %.3f" % (c0, mean([RECS[(a, s)][5 * e]["beta"][1] for s in SEEDS])))))
        print("  %-11s %s" % (a, "  ".join(cells)))
    print("  mean r = exp(beta+15) over records with 15 <= step/500 < 30 and 30 <= step/500 < 60, arm mean"
          " (complement / carrier group):")
    for a in ARMS:
        if a == "k01":
            print("    %-11s shared %9.1f / %7.1f" % (a, mean([SUMM[(a, s, 0)][3] for s in SEEDS]), mean([SUMM[(a, s, 0)][4] for s in SEEDS])))
        else:
            print("    %-11s complement %9.1f / %7.1f   carrier group %9.1f / %7.1f" % (
                a, mean([SUMM[(a, s, 0)][3] for s in SEEDS]), mean([SUMM[(a, s, 0)][4] for s in SEEDS]),
                mean([SUMM[(a, s, 1)][3] for s in SEEDS]), mean([SUMM[(a, s, 1)][4] for s in SEEDS])))
    same = all(RECS[("HIGHISOPATH", s)][k]["beta"][0] == RECS[("BIGISOPATH", s2)][k]["beta"][0] == RECS[("LOWISOPATH", s3)][k]["beta"][0]
               for s in SEEDS for s2 in SEEDS for s3 in SEEDS for k in range(N_REC))
    print("  the three forced arms' complement beta identical at every record across all 9 runs: %s" % same)

    print("\n[D3] complement pins per run (group 0; k01's shared beta): first r <= 2 (the MAGNITUDE gate, 247.7 [D4]'s form);"
          " records at/after it back above r 2; first record from which r <= 2 for good (the scorer's `pin`, which starts at"
          " the peak); continuous exact clamp = first record from which beta == -15.0 on every later record; median r over the"
          " last 125 records.  Epochs = step / 500.")
    PINS = {}
    for a in ARMS:
        for s in SEEDS:
            f2, above, good, exact, medr = pins(RECS[(a, s)], 0)
            PINS[(a, s)] = (f2, above, good, exact, medr)
            print("  %-11s s%-3d %-6s first r<=2 %-6s back above %-4s for good %-6s exact clamp %-6s median r last quarter %8.3f"
                  % (a, s, "shared" if a == "k01" else "cmpl", "never" if f2 is None else "%.1f" % f2,
                     "-" if above is None else str(above), "never" if good is None else "%.1f" % good,
                     "never" if exact is None else "%.1f" % exact, medr))
    print("  carrier group (group 1) continuous exact clamp, held arms (the schedule's; one per arm): %s" % "  ".join(
        "%s %s" % (a, "never" if pins(RECS[(a, SEEDS[0])], 1)[3] is None else "%.1f" % pins(RECS[(a, SEEDS[0])], 1)[3]) for a in HELD))

    print("\n[D4] the Epoch line FROM WHICH an arm's TEST stays MORE THAN 2 pp BELOW the reference's (line n = after n+1"
          " epochs); per seed = same seed number (different runs); arm mean = on the arm-mean curves")
    for x, y in (("HIGHISOPATH", "LOWISOPATH"), ("BIGISOPATH", "LOWISOPATH"), ("HOLDHIGH", "LOWISOPATH"), ("HOLDBIG", "LOWISOPATH"),
                 ("HOLDHIGH", "ISO"), ("HOLDBIG", "ISO"), ("HIGHISOPATH", "ISO"), ("BIGISOPATH", "ISO"),
                 ("BIGISOPATH", "HIGHISOPATH"), ("HOLDBIG", "HOLDHIGH"), ("HOLDHIGH", "HIGHISOPATH")):
        per = []
        for s in SEEDS:
            g = [R[(y, s)]["ep"][e][1] - R[(x, s)]["ep"][e][1] for e in range(100)]
            fe = first_stays(g, lambda v: v > 2.0)
            per.append("never" if fe is None else str(fe))
        g = [AMT[y][e] - AMT[x][e] for e in range(100)]
        fe = first_stays(g, lambda v: v > 2.0)
        print("  %-11s below %-11s by > 2 pp from line: arm mean %-5s  seeds %s   (arm-mean gap at lines 14/16/18/20/24/29/39/49/99: %s)"
              % (x, y, "never" if fe is None else fe, " / ".join(per), " ".join("%+.2f" % g[e] for e in (14, 16, 18, 20, 24, 29, 39, 49, 99))))

    print("\n[D5] the FREE complements: HOLDHIGH's against cvt7's HOLDHIGH (between batch, orientation) and ISOPATH; HOLDBIG's"
          " against the carriers' own v_9428")
    for a in ("ISO", "HOLDHIGH", "HOLDBIG"):
        for s in SEEDS:
            recs = RECS[(a, s)]
            d8609 = max(abs(float(x["beta"][0]) - tri(x["step"] + 2, 8609)) for x in recs)
            d9428 = max(abs(float(x["beta"][0]) - tri(x["step"] + 2, 9428)) for x in recs)
            d9428e = max(abs(float(x["beta"][0]) - tri(x["step"] + 2, 9428)) for x in recs if x["step"] <= 40 * SPE)
            diso = max(abs(float(x["beta"][0]) - rp.at(x["step"] + 2)) for x in recs)
            fdep = None
            for x in recs:
                if abs(float(x["beta"][0]) - rp.at(x["step"] + 2)) > 0.1:
                    fdep = x["step"] / float(SPE)
                    break
            pk, pke, pin, r1, r2 = SUMM[(a, s, 0)]
            print("  %-9s s%-3d cmpl peak %.3f @%.1f  r15-30 %7.0f  r30-60 %6.0f  first r<=2 %-5s  exact clamp %-5s  max|c - v_8609| %.3f"
                  "  max|c - v_9428| %.3f (ep 0-40 %.3f)  max|c - ISOPATH| %.3f, first > 0.1 at epoch %s"
                  % (a, s, pk, pke, r1, r2, "never" if PINS[(a, s)][0] is None else "%.1f" % PINS[(a, s)][0],
                     "never" if PINS[(a, s)][3] is None else "%.1f" % PINS[(a, s)][3], d8609, d9428, d9428e, diso,
                     "never" if fdep is None else "%.1f" % fdep))
    for s in SEEDS:
        recs = RECS[("k01", s)]
        print("  k01       s%-3d shared max|beta - v_8609| %.3f (ep 0-40 %.3f)" % (
            s, max(abs(float(x["beta"][0]) - tri(x["step"] + 2, 8609)) for x in recs),
            max(abs(float(x["beta"][0]) - tri(x["step"] + 2, 8609)) for x in recs if x["step"] <= 40 * SPE)))
    c7 = os.path.join(runsdir, "cvt7")
    have7 = all(os.path.isfile(os.path.join(c7, "probe_cvt7-%s-s%d" % (a, s), "probe.jsonl")) for a in ("HOLDHIGH", "ISO")
                for s in (99, 100, 101))
    if have7:
        for a in ("HOLDHIGH", "ISO"):
            for s in (99, 100, 101):
                recs = load_recs(os.path.join(c7, "probe_cvt7-%s-s%d" % (a, s)))
                f2, above, good, exact, medr = pins(recs, 0)
                pk, pke, _pin, r1, r2 = summary(recs, 0)
                print("  cvt7 %-9s s%-3d cmpl peak %.3f @%.1f  r15-30 %7.0f  first r<=2 %-5s  for good %-5s  exact clamp %-5s  (orientation only)"
                      % (a, s, pk, pke, r1, "never" if f2 is None else "%.1f" % f2, "never" if good is None else "%.1f" % good,
                         "never" if exact is None else "%.1f" % exact))
    else:
        print("  (cvt7 probe dirs not under this runsdir; the between-batch orientation is skipped)")
    print("  complement beta, arm mean, at epochs 16 / 17 / 18 / 19 / 20 / 22 / 25 / 30 / 35 / 40 / 45 / 50 / 55 / 60 / 70 / 80 / 99.8:")
    EPS = (16, 17, 18, 19, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80)

    def cm(a, idx, k=0):
        return mean([RECS[(a, s)][idx]["beta"][k] for s in SEEDS])

    for a in ARMS:
        print("    %-11s %s" % (a + (" (sh)" if a == "k01" else ""), " / ".join(["%.2f" % cm(a, 5 * e) for e in EPS] + ["%.2f" % cm(a, -1)])))

    print("\n[D6] timing against k01 and the free complements' departure")
    for a in ("HOLDBIG", "BIGISOPATH", "HOLDHIGH", "HIGHISOPATH"):
        g = [AMT[a][e] - AMT["k01"][e] for e in range(100)]
        below = first_stays(g, lambda v: v < -2.0)
        above = first_stays(g, lambda v: v > 2.0)
        print("  %-11s - k01 arm-mean TEST at lines 10..40 step 2: %s | stays < -2 pp from line %s; stays > +2 pp from line %s"
              % (a, " ".join("%d:%+.2f" % (e, g[e]) for e in range(10, 41, 2)), "never" if below is None else below,
                 "never" if above is None else above))
        print("  %-11s per seed (same seed number) TEST - k01 at lines 16 / 18 / 20 / 24 / 29 / 49 / 99: %s" % (
            a, "  ".join("s%d %s" % (s, "/".join("%+.2f" % (R[(a, s)]["ep"][e][1] - R[("k01", s)]["ep"][e][1]) for e in (16, 18, 20, 24, 29, 49, 99)))
                         for s in SEEDS)))
    for a in ("HOLDHIGH", "HOLDBIG"):
        dc = [cm(a, 5 * e) - cm("ISO", 5 * e) for e in range(100)]
        firstc = next((e for e in range(100) if abs(dc[e]) > 0.1), None)
        print("  %-8s complement beta minus ISO's (arm means) at epochs 10..40 step 2: %s ; first epoch |diff| > 0.1: %s"
              % (a, " ".join("%d:%+.2f" % (e, dc[e]) for e in range(10, 41, 2)), firstc))
        dg = [cm(a, 5 * e, 1) - cm("ISO", 5 * e, 1) for e in range(0, 41)]
        print("  %-8s carrier-group beta minus ISO's (arm means) at epochs 2..24 step 2: %s" % (
            a, " ".join("%d:%+.2f" % (e, dg[e]) for e in range(2, 25, 2))))

    print("\nVIOLATIONS %d" % len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
