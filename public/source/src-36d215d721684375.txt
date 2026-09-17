#!/usr/bin/env python3
"""cvt7_attack_indep.py <runsdir> <scorer.py> <score.log> <all_runs.csv> <CORPUS-EXCLUSIONS.tsv>
-- independent re-derivation of `cvt7` (CORRECTIONS 243).

Imports NOTHING from analysis/cVT7_grouphold_score.py, analysis/corpus_exclusions.py or any other repo module, and
uses no regex.  Reads, by string splitting only: every raw cvt7-<arm>-s<seed>-<jobid>.out in <runsdir> (ARGS / ENV /
PROBE_TENSOR / VOTE_W / BETA_HOLD / GROUP_HOLD lines, `Epoch` lines, RUN_DONE, Traceback); <runsdir>/cvt7/
PARTITION-MANIFEST.txt and PROVENANCE.txt; each run's probe dir (probe_tensor.json, block_sizes.json, probe.jsonl).
plateau5 = mean TEST over the run's own epochs 95..99; TRAIN5 the same on the Train column.
Bars, witnesses (built from mode / P / b0 / ms), the max-rate triangle, G-BITE's clauses (a)-(e) and the branch
order are RE-TYPED from the registration text (243.2-243.6); nothing is copied from the scorer's code.

CORPUS-DEPENDENT LINE, EXPLICIT.  Exactly ONE re-built line depends on the corpus: the scorer's DISCLOSURE line
`corpus: N rows after corpus_exclusions.filter_rows, cvt7- excluded ...`.  It is re-derived here from the two files
given on the command line (CSV rows whose `run` does not start with `cvt7-`, minus rows whose (run, job_id) is a
TSV key), and the shas of both files are printed.  The committed output was produced against the corpus at repo
commit 5309a96 (all_runs.csv c2d1164d..., 3,049 rows; CORPUS-EXCLUSIONS.tsv f8455571..., 51 keys -> 2,998), which is
also what both scorer logs were written against.  After a later ingest the line (and only it) moves, and the parser
reports 1 violation until run against those two files at 5309a96 (`git show 5309a96:<path>`).  No bar, sigma,
contrast, branch or stamp reads the corpus.

AGREEMENT.  Every scorer line carrying a level, contrast, gate, G-BITE count, sigma, branch, stamp or descriptive
readout (and the design header) is RE-BUILT here in the scorer's printed format and looked up VERBATIM in
<score.log>; a line not found is a VIOLATION.  The FINAL line is compared as one string and token by token.  The
scorer-log lines NOT re-built are counted and classified (registered licence prose, rules, blank lines, and the two
host-path lines, which are matched against the runsdir given but not printed so the output is host-independent).

Sections:
  [0] corpus disclosure; manifest; PROVENANCE
  [1] completeness, ARGS / ENV / PROBE_TENSOR / VOTE_W / BETA_HOLD / GROUP_HOLD per run
  [2] levels: per-run plateau5 + TRAIN5, per-arm mean / sd / range, TEST OLS tail slope 80-99
  [3] sigma, SE, every contrast, TEST beside TRAIN; per-seed paired contrasts
  [4] G-BITE from the probe records; the vote-out decomposition (z_agg == fsum z_tensor per group);
      NON-VACUITY cross-read (every run's group 1 read as every held mode)
  [5] branch, states, stamps, every bar margin, per-seed states; FINAL compared with the scorer's
  [6] the scorer's descriptive readouts re-derived
  [7] LINE-BY-LINE AGREEMENT
  DESCRIPTIVE, NOT REGISTERED (cannot move anything above):
  [D1] APPLIED beta of the complement and the carrier group at epochs 10 / 17 / 20 / 30 / 40 (median seed; arm mean)
  [D2] TEST (TRAIN) after 15 / 30 / 50 / 100 epochs
  [D3] settle epochs (TEST and TRAIN), share of plateau5 by epoch
  [D4] the complement's pin by the MAGNITUDE gate (cvh1's form: first r <= 2; records back above r 2; median r over
       the last quarter) beside the exact clamp, per run
  [D5] HOLDHIGH's free complement against ISO's and k01's shared beta (and the replay v_8609): trajectory, r(15-30),
       timing of the TEST gap against the complement difference and the carrier difference
  [D6] orientation only (between batch, PlainNet): cvt4's HOLDHIGH / HEAD complement pin by the same gate
"""
import csv
import hashlib
import json
import math
import os
import struct
import sys

PREFIX = "cvt7"
NET = "ResNet18_c100"
SEEDS = [99, 100, 101]
ARMS = ["k01", "ISO", "HOLDLOW", "HOLDHIGH", "HOLDISO"]
HELD = ["HOLDLOW", "HOLDHIGH", "HOLDISO"]
# 243.2, re-typed
CARRIERS = ["layer4.0.bn2.weight", "layer4.0.shortcut.1.weight", "layer4.1.bn2.weight"]
CARRIER_IDX = [50, 53, 59]
ISOSPEC = "sets:1-49,51-52,54-58,60-62/" + ",".join(CARRIERS)
SPEC = dict((a, ISOSPEC) for a in ARMS)
SPEC["k01"] = "scalar"
HOLD = {"HOLDLOW": ("floor", None), "HOLDHIGH": ("tri", 8609), "HOLDISO": ("tri", 5153)}
ENV_WANT = ("ENV: AUGMENT=1 BETA_CLIP=-15:-2.3026 HIER=none LAM=na ETA_RATIO=na COS_TOTAL=default "
            "COS_WARMUP=default SCHED=none SCHED_TOTAL=none SCHED_WARMUP=none SCHED_MIN=none PROBE=100 "
            "EB_RHO=na EB_LOG=0")
FIXED = {"optimizer": "HF", "alg-base": "SGDm", "momentum-param-base": "0.99", "weight-decay-base": "0.1",
         "alg-meta": "Lion", "momentum-param-meta": "0.99", "Lion-beta2-meta": "0.9",
         "weight-decay-meta": "0", "dataset": "CIFAR100", "batch-size": "100", "meta-stepsize": "1e-3",
         "alpha0": "1e-6", "num-epochs": "100", "NN-name": NET, "gamma": "1"}
# 243.6 frozen bars, re-typed
SIGMA_PRIOR = 0.694442846939599
GAP_MIN, K01_MAX, MATCH, NULL, RESCUE, DIVERGED = 20.0, 30.0, 5.0, 2.0, 10.0, 5.0
FLOOR_MIN, CEIL_MAX = 15.0, 90.0
BH_TOL, TIE_REL, N_REC = 1e-5, 1e-12, 500
MS, B2 = 1e-3, 0.9
LO, HI = -15.0, -2.3026
SPE = 500
# between-batch anchors, 243.2 (non-gating)
BETWEEN = [("ciso1", 23.2807, 70.2113), ("cdep1", 23.3520, 70.0440)]
# sha prefixes as recorded in CORRECTIONS 243.4 / 243.8 / 243.12
PREF = {"BUILD_NETWORK_SHA256": "c7998883", "HF_SHA256": "2396f2be", "RUNNER_SHA256": "805267e4"}
MANIFEST_PREFIX, MANIFEST_BYTES = "6734d0ff", 5085
NTENS, TOTPAR = 62, 11220132


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
    """243.3 / 237.3(1): u(n) = min(n-1, P) - max(0, n-1-P); beta = min(hi, max(lo, b0 + ms u(n))), float32."""
    u = 0 if n <= 0 else min(n - 1, P) - max(0, n - 1 - P)
    return f32(min(HI, max(LO, B0 + MS * u)))


def hold_at(mode, P, n):
    return LO if mode == "floor" else tri(n, P)


def lion(bp, m, z):
    """243.6 (b): clamp(beta_pre - ms*sign(b2*mom_pre + (1-b2)*z_agg)); tie iff |L| <= TIE_REL of its parts."""
    p, q = B2 * m, (1.0 - B2) * z
    L = p + q
    return max(LO, min(HI, bp - MS * sgn(L))), abs(L) <= TIE_REL * (abs(p) + abs(q))


def witness(arm):
    """243.4: the GROUP_HOLD witness line, built from mode / P / b0 / ms."""
    if arm not in HOLD:
        return "GROUP_HOLD: off"
    mode, P = HOLD[arm]
    w = "GROUP_HOLD: on type=blockwise group=1 groupsize=3 names=%s mode=%s" % ("+".join(CARRIERS), mode)
    if mode == "floor":
        return w + " value=" + repr(LO)
    return w + " P=%d b0=%s ms=%s lo=%s hi=%s peak=%s" % (P, repr(B0), repr(MS), repr(LO), repr(HI),
                                                          repr(min(HI, max(LO, B0 + MS * P))))


def grouphold_value(arm):
    mode, P = HOLD[arm]
    return "+".join(CARRIERS) + ":" + (mode if mode == "floor" else "tri:%d" % P)


def parse_out(path):
    r = {"args": [], "env": [], "pt": [], "vw": [], "bh": [], "gh": [], "ep": {}, "done": False, "tb": 0, "dup": 0}
    for ln in open(path, errors="replace"):
        s = ln.rstrip("\n")
        if s.startswith("ARGS:"):
            r["args"].append(s)
        elif s.startswith("ENV:"):
            r["env"].append(s)
        elif s.startswith("PROBE_TENSOR:"):
            r["pt"].append(s)
        elif s.startswith("VOTE_W"):
            r["vw"].append(s)
        elif s.startswith("BETA_HOLD"):
            r["bh"].append(s)
        elif s.startswith("GROUP_HOLD"):
            r["gh"].append(s)
        elif s.startswith("Epoch "):
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


def traj_summary(recs, k):
    """(peak, peak epoch, continuous-clamp pin epoch or None, mean r over epochs [15,30), last) of group k."""
    seq = [(x["step"], float(x["beta"][k])) for x in recs]
    peak = seq[0]
    for s, b in seq:
        if b > peak[1]:
            peak = (s, b)
    pin = None
    for j in range(len(seq)):
        if all(b <= LO + 1e-4 for _s, b in seq[j:]):
            pin = seq[j][0] / float(SPE)
            break
    r = mean([math.exp(b + 15.0) for s, b in seq if 15 * SPE <= s < 30 * SPE])
    return peak[1], peak[0] / float(SPE), pin, r, seq[-1][1]


def magnitude_pin(recs, k):
    """cvh1's MAGNITUDE form: first record with r = exp(beta+15) <= 2; records after it back above r 2; median r over
    the last quarter of records; the exact-clamp (<= -15 + 1e-4 for good) epoch; max r after the first r <= 2."""
    seq = [(x["step"] / float(SPE), float(x["beta"][k])) for x in recs]
    lr2 = LO + math.log(2.0)
    f2 = None
    for e, b in seq:
        if b <= lr2:
            f2 = e
            break
    above = None if f2 is None else sum(1 for e, b in seq if e >= f2 and b > lr2)
    tail = [math.exp(b - LO) for _e, b in seq[-max(1, int(len(seq) * 0.25)):]]
    exact = None
    for j in range(len(seq)):
        if all(b <= LO + 1e-4 for _e, b in seq[j:]):
            exact = seq[j][0]
            break
    maxr = None if f2 is None else max(math.exp(b - LO) for e, b in seq if e >= f2)
    return f2, above, median(tail), exact, maxr


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


def main():
    runsdir, scorer, slog, csvp, tsvp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    bad = []
    expect = []          # (section, line) that must appear verbatim in the scorer log

    def viol(msg):
        bad.append(msg)
        print("  VIOLATION " + msg)

    def exp(sec, line):
        expect.append((sec, line))

    base = os.path.join(runsdir, PREFIX)
    logl = [ln.rstrip("\n") for ln in open(slog)]

    # ---------------------------------------------------------------- header (design lines, 243.2 / 243.6)
    rule = "=" * 78
    exp("HEADER", rule)
    exp("HEADER", " cvt7 -- DOES 'THE CARRIER NEEDS A SMALL STEP SIZE' TRANSFER TO ResNet18_c100?  (PATCH_GROUPHOLD)")
    exp("HEADER", " %s / CIFAR100   100 epochs   seeds %s   AUGMENT=1  BETA_CLIP=-15:-2.3026  PROBE=100  PROBE_TENSOR=1"
        % (NET, ",".join(str(s) for s in SEEDS)))
    for a in ARMS:
        exp("HEADER", "   %-9s %-10s GROUP_HOLD %s" % (a, "scalar" if a == "k01" else "ISO's",
                                                      grouphold_value(a) if a in HOLD else "(unset)"))
    exp("HEADER", " CO-PRIMARY: P_HIGH = ISO - HOLDHIGH; P_LOW = HOLDLOW - ISO.")
    exp("HEADER", " plateau5 = mean TEST over epochs 95..99 of each run's own .out.  BARS ARE FROZEN LITERALS (O2).")

    # ---------------------------------------------------------------- [0]
    print("[0] corpus disclosure (the ONE corpus-dependent line; no bar reads it)")
    n_all, n_tsv, n_keys, n_keep = read_corpus_count(csvp, tsvp)
    print("    all_runs.csv sha256 %s  rows %d;  CORPUS-EXCLUSIONS.tsv sha256 %s  rows %d keys %d  ->  %d rows (cvt7- dropped)"
          % (sha(csvp)[:16], n_all, sha(tsvp)[:16], n_tsv, n_keys, n_keep))
    exp("CORPUS", "corpus: %d rows after corpus_exclusions.filter_rows, %s- excluded  (DISCLOSURE ONLY -- no bar, sigma, "
        "branch or stamp reads it)" % (n_keep, PREFIX))

    names, numel, man = [], [], {}
    mpath = os.path.join(base, "PARTITION-MANIFEST.txt")
    mtext = open(mpath).read()
    for ln in mtext.split("\n"):
        p = ln.split()
        if not p:
            continue
        if p[0] == "TENSOR":
            names.append(p[2])
            numel.append(int(p[3]))
        else:
            man.setdefault(p[0], []).append(ln)
    msha = sha(mpath)
    print("    manifest: %d tensors, %d params; carriers %s; %d bytes, sha256 %s..." % (
        len(names), sum(numel), ",".join("%d:%s" % (i, names[i - 1]) for i in CARRIER_IDX), len(mtext.encode()), msha[:16]))
    isog = groups_of(ISOSPEC, names)
    ok_man = (len(names) == NTENS and sum(numel) == TOTPAR and [names[i - 1] for i in CARRIER_IDX] == CARRIERS
              and all(numel[i - 1] == 512 for i in CARRIER_IDX) and msha.startswith(MANIFEST_PREFIX)
              and len(mtext.encode()) == MANIFEST_BYTES
              and [len(g) for g in isog] == [59, 3] and isog[1] == CARRIER_IDX
              and man.get("CARRIERS") == ["CARRIERS " + ",".join("%d:%s" % (i, n) for i, n in zip(CARRIER_IDX, CARRIERS))]
              and man.get("REPLAY") == ["REPLAY B0 %s MS %s P_HIGH 8609 P_ISO 5153" % (repr(B0), repr(MS))]
              and man.get("WITNESS") == ["WITNESS %s %s" % (a, witness(a)) for a in ARMS]
              and man.get("GROUPHOLD") == ["GROUPHOLD %s %s" % (a, grouphold_value(a) if a in HOLD else "off") for a in ARMS])
    print("    ISO spec -> sizes %s, group 1 = %s; CARRIERS / REPLAY / 5 GROUPHOLD / 5 WITNESS lines == built here; "
          "sha prefix %s, %d bytes: %s" % ([len(g) for g in isog], isog[1], MANIFEST_PREFIX, MANIFEST_BYTES, ok_man))
    if not ok_man:
        viol("manifest")
    prov = {}
    for ln in open(os.path.join(base, "PROVENANCE.txt")):
        p = ln.split(None, 1)
        if len(p) == 2:
            prov.setdefault(p[0], p[1].strip())
    s_sha = sha(scorer)
    ok_prov = (prov.get("MODE") == "submit" and all(prov.get(k, "").startswith(v) for k, v in PREF.items())
               and prov.get("SCORER_SHA256") == s_sha)
    print("    PROVENANCE MODE %s  BUILD %s...  HF %s...  RUNNER %s...  SCORER %s... == scorer file: %s" % (
        prov.get("MODE"), prov.get("BUILD_NETWORK_SHA256", "")[:8], prov.get("HF_SHA256", "")[:8],
        prov.get("RUNNER_SHA256", "")[:8], prov.get("SCORER_SHA256", "")[:8], prov.get("SCORER_SHA256") == s_sha))
    if not ok_prov:
        viol("provenance")
    exp("G-STRUCT", "  manifest: %s (<runsdir>/cvt7/)" % mpath)
    exp("G-STRUCT", "  PASS G-STRUCT byte-identical to synthetic_manifest_text() (62 tensors, ISO [59,3], the hold table)"
        "   %d vs %d bytes" % (MANIFEST_BYTES, MANIFEST_BYTES))
    exp("G-PROV", "  provenance: %s (<runsdir>/cvt7/)" % os.path.join(base, "PROVENANCE.txt"))
    exp("G-PROV", "  PASS G-PROV MODE submit   submit")
    exp("G-PROV", "  PASS G-PROV build_network.py sha256 == the LIVE one %s...   %s" % (
        prov["BUILD_NETWORK_SHA256"][:12], prov["BUILD_NETWORK_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV HF.py sha256 == HF_POST_SHA %s... (cvt7's PATCH_GROUPHOLD tree)   %s" % (
        prov["HF_SHA256"][:12], prov["HF_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV RUNNER_SHA256 == cvt7's runner %s...   %s" % (prov["RUNNER_SHA256"][:12], prov["RUNNER_SHA256"][:16]))
    exp("G-PROV", "  PASS G-PROV SCORER_SHA256 == this scorer's sha256 %s...   %s" % (s_sha[:12], s_sha[:16]))

    # ---------------------------------------------------------------- [1]
    print("\n[1] completeness and header lines (GROUP_HOLD witnesses built from mode / P / b0 / ms)")
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
    exp("COMPLETE", "COMPLETE  15 runs, epochs 0..99, RUN_DONE, no traceback, finite plateau5")
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
            wrong = [k for k, v in want.items() if fd.get(k) != v]
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
            gh_ok = r["gh"] == [witness(a)]
            line_ok = complete and args_ok and env == [ENV_WANT] and pt_ok and vw_ok and bh_ok and gh_ok
            print("  %-4s %-9s s%-3d job %d  epochs %d  RUN_DONE %s  tb %d  ARGS wrong %s rep %s  ENV %s  PT %s  VOTE_W %s"
                  "  BETA_HOLD %s  GROUP_HOLD %s"
                  % ("OK" if line_ok else "BAD", a, s, jid, len(r["ep"]), r["done"], r["tb"], wrong or "-", rep or "-",
                     env == [ENV_WANT], pt_ok, vw_ok, bh_ok, gh_ok))
            if not line_ok:
                viol("%s-s%d header/completeness" % (a, s))
            if args_ok:
                exp("G-ARGS", "  PASS G-ARGS %s-s%d" % (a, s))
            if vw_ok:
                exp("G-VOTEW", "  PASS G-VOTEW %s-s%d" % (a, s))
            if bh_ok:
                exp("G-BHOLD", "  PASS G-BHOLD %s-s%d" % (a, s))
            if gh_ok:
                exp("G-GHOLD", "  PASS G-GHOLD %s-s%d" % (a, s))
    if len(R) != 15:
        print("\nVIOLATIONS %d (incomplete batch; stopping)" % len(bad))
        sys.exit(1)
    if envs == {ENV_WANT} and all(len(R[k]["env"]) == 1 for k in R):
        exp("G-ENV", "  PASS G-ENV one distinct ENV line, one per run   1 distinct")
        exp("G-ENV", "        " + ENV_WANT)
        exp("G-ENV", "  PASS G-ENV the ENV line is ciso1's / cdep1's, byte for byte (PROBE_DIR stripped)")
    if n_pt_ok == 15:
        exp("G-ENV", "  PASS G-ENV every run printed ONE `PROBE_TENSOR: on every=100 type=<its arm's> tensors=62`   15 runs")

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
        print("  %-9s TEST %.4f (sd %.4f, range %.4f)  TRAIN %.4f  seeds %s  train %s  slope80-99 %s"
              % (a, mean(v), sdev(v), max(v) - min(v), mean(t), " / ".join("%.4f" % x for x in v),
                 " / ".join("%.4f" % x for x in t), " / ".join("%+.3f" % x for x in sl)))
        exp("LEVELS", "  %-9s TEST %.4f  sd %.4f  range %.4f   TRAIN %.4f   tail slope %s pp/ep"
            % (a, mean(v), sdev(v), max(v) - min(v), mean(t), "/".join("%.3f" % x for x in sl)))
        exp("LEVELS", "            seeds: %s" % ", ".join("s%d %.4f (train %.4f)" % (s, x, y) for s, x, y in zip(SEEDS, v, t)))
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
    C = [("P_HIGH", "ISO", "HOLDHIGH", "PRIMARY"), ("P_LOW", "HOLDLOW", "ISO", "PRIMARY"),
         ("D_ISO", "ISO", "k01", "KEY"), ("P_CTL", "ISO", "HOLDISO", "KEY")]
    C += [("D_" + a, a, "k01", "KEY") for a in HELD]
    CT = {}
    for lab, x, y, kind in C:
        d, dt = M[x] - M[y], TM[x] - TM[y]
        CT[lab] = (d, dt)
        print("  %-8s %-11s = %-9s - %-9s TEST %+9.4f pp = %+8.2f SE   TRAIN %+9.4f   per seed %s"
              % (kind, lab, x, y, d, d / se, dt, " / ".join("%+.4f" % (P[x][j] - P[y][j]) for j in range(3))))
    fmt3 = {"P_HIGH": "ISO - HOLDHIGH       ", "P_LOW": "HOLDLOW - ISO        "}
    for lab in ("P_HIGH", "P_LOW"):
        d, dt = CT[lab]
        exp("CONTRASTS", "  PRIMARY  %-8s = %s = %+.4f pp = %+.2f SE   (TRAIN %+.4f)" % (lab, fmt3[lab], d, d / se, dt))
    d, dt = CT["D_ISO"]
    exp("CONTRASTS", "  KEY      D_ISO    = ISO - k01             = %+.4f pp = %+.2f SE   (TRAIN %+.4f)  [positive control]" % (d, d / se, dt))
    d, dt = CT["P_CTL"]
    exp("CONTRASTS", "  KEY      P_CTL    = ISO - HOLDISO         = %+.4f pp = %+.2f SE   (TRAIN %+.4f)  [hold control]" % (d, d / se, dt))
    for a in HELD:
        d, dt = CT["D_" + a]
        exp("CONTRASTS", "  KEY      D_%-9s = %-9s - k01  = %+.4f pp = %+.2f SE   (TRAIN %+.4f)   ISO - %-9s = %+.4f"
            % (a, a, d, d / se, dt, a, M["ISO"] - M[a]))
    exp("CONTRASTS", "  bars: GAP_MIN %.1f (%.2f SE) | MATCH %.1f pp (%.2f SE) | NULL %.1f pp (%.2f SE) | RESCUE %.1f pp (%.2f SE)"
        % (GAP_MIN, GAP_MIN / se, MATCH, MATCH / se, NULL, NULL / se, RESCUE, RESCUE / se))

    # ---------------------------------------------------------------- [4]
    print("\n[4] G-BITE re-derived from the raw probe records (243.6 (a)-(e)), the vote-out decomposition, non-vacuity")
    RECS = {}
    bite_ok_all = True
    nonfin = {}
    cross = {}
    exp("G-BITE", "  (a) 500 records; (b) every UNHELD group's beta == Lion recomputed from its own beta_pre/mom_pre/z_agg (tol 1e-05);")
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
            held = a in HOLD
            c = dict(n=0, nonfin=0, badrec=0, badstep=0, lion=0, ties=0, sched=0, schedpre=0, gh=0, ghctl=0, bhk=0, act=0)
            wl = ws = 0.0
            last_act = None
            prev = -1
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
                c["bhk"] += any(key in x for key in ("bh_n", "bh_active", "bh_nat", "bh_held"))
                for g in ((0,) if held else range(ng)):
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
                has_gh = any(key in x for key in ("gh_n", "gh_active", "gh_nat", "gh_held"))
                if ng == 2:
                    for h in HELD:
                        mode, Pp = HOLD[h]
                        e1, e2 = abs(beta[1] - hold_at(mode, Pp, st + 2)), abs(bp[1] - hold_at(mode, Pp, st + 1))
                        cross.setdefault((a, s, h), [0, 0.0])
                        cross[(a, s, h)][0] += (e1 > BH_TOL) or (e2 > BH_TOL)
                        cross[(a, s, h)][1] = max(cross[(a, s, h)][1], e1, e2)
                if not held:
                    c["ghctl"] += has_gh
                    continue
                n = st + 2
                mode, Pp = HOLD[a]
                e1, e2 = abs(beta[1] - hold_at(mode, Pp, n)), abs(bp[1] - hold_at(mode, Pp, n - 1))
                ws = max(ws, e1, e2)
                c["sched"] += e1 > BH_TOL
                c["schedpre"] += e2 > BH_TOL
                try:
                    gn, ga, gnat, gheld = int(x["gh_n"]), int(x["gh_active"]), float(x["gh_nat"]), float(x["gh_held"])
                except (KeyError, TypeError, ValueError):
                    c["gh"] += 1
                    continue
                nat1, tie1 = lion(bp[1], mo[1], za[1])
                good = (gn == n and abs(gheld - beta[1]) <= BH_TOL and 0 <= ga <= gn and ga >= prev
                        and (tie1 or abs(gnat - nat1) <= BH_TOL))
                c["gh"] += not good
                c["act"] += abs(gnat - gheld) > BH_TOL
                prev = max(prev, ga)
                last_act = ga
            ok = (meta_ok and len(recs) == N_REC and c["badrec"] == 0 and c["n"] + c["nonfin"] == N_REC
                  and c["badstep"] == 0 and c["lion"] == 0 and c["bhk"] == 0)
            if held:
                ok = ok and c["sched"] == 0 and c["schedpre"] == 0 and c["gh"] == 0 and (last_act is None or last_act >= c["act"])
            else:
                ok = ok and c["ghctl"] == 0
            nonfin[(a, s)] = c["nonfin"]
            bite_ok_all = bite_ok_all and ok
            print("  %-4s %-9s s%-3d meta %s recs %d lion-mism %d ties %d worst %.1e | sched %d/%d worst %.1e | gh-audit %d"
                  " gh-on-control %d bh-keys %d active %d gh_active %s | vote-out decomposition worst rel %.1e"
                  % ("PASS" if ok else "FAIL", a, s, meta_ok, len(recs), c["lion"], c["ties"], wl, c["sched"], c["schedpre"],
                     ws, c["gh"], c["ghctl"], c["bhk"], c["act"], last_act, worst_dec))
            if not ok:
                viol("G-BITE %s-s%d" % (a, s))
            if worst_dec > 1e-4:
                viol("decomposition %s-s%d" % (a, s))
            exp("G-BITE", "  %-4s G-BITE %-9s s%-3d records %s  bad %d  nonfinite %d  lion-mismatch %d (ties %d, worst %.1e)  "
                "sched %d/%d (worst %.1e)  gh-audit %d  gh-on-control %d  bh-keys %d  active records %d  gh_active %s"
                % ("PASS" if ok else "FAIL", a, s, len(recs), c["badrec"], c["nonfin"], c["lion"], c["ties"], wl,
                   c["sched"], c["schedpre"], ws, c["gh"], c["ghctl"], c["bhk"], c["act"], last_act))
    print("  NON-VACUITY cross-read: records (of 500) on which group 1 FAILS each held mode's schedule (either half); max |error|")
    print("  %-13s %s" % ("run", "  ".join("%-22s" % ("as " + h) for h in HELD)))
    for a in ARMS[1:]:
        for s in SEEDS:
            print("  %-9s s%-3d %s" % (a, s, "  ".join("%3d  max %8.4f      " % tuple(cross[(a, s, h)]) for h in HELD)))
    nv_ok = all(cross[(a, s, h)][0] > 0 for a in ARMS[1:] for s in SEEDS for h in HELD if h != a)
    nv_own = all(cross[(a, s, a)][0] == 0 for a in HELD for s in SEEDS)
    nv_min = min(cross[(a, s, h)][0] for a in ARMS[1:] for s in SEEDS for h in HELD if h != a)
    print("  every run FAILS every held mode but its own on >= 1 record: %s (fewest failing records %d); every held run passes"
          " its own on all: %s" % (nv_ok, nv_min, nv_own))
    if not (nv_ok and nv_own):
        viol("non-vacuity cross-read")

    # ---------------------------------------------------------------- [5]
    print("\n[5] branch, states, stamps (243.6, re-typed; first match wins)")
    div = [a for a in ARMS if max(P[a]) - min(P[a]) > DIVERGED]
    gap = M["ISO"] - M["k01"]

    def st_of(L):
        if L >= M["ISO"] - MATCH:
            return "AT-ISO"
        if L <= M["k01"] + NULL:
            return "AT-K01"
        return "BETWEEN"

    states = dict((a, st_of(M[a])) for a in HELD)
    if not bite_ok_all:
        branch = "PATCH-NOT-VERIFIED"
    elif not (FLOOR_MIN <= hi <= CEIL_MAX):
        branch = "HARNESS-UNSOUND"
    elif div:
        branch = "UNRESOLVED-DIVERGED"
    elif M["k01"] > K01_MAX:
        branch = "SCALAR-NOT-COLLAPSED"
    elif gap < GAP_MIN:
        branch = "POSITIVE-CONTROL-FAILED"
    elif states["HOLDISO"] == "AT-K01":
        branch = "HOLD-CONTROL-FAILED"
    elif states["HOLDISO"] == "BETWEEN":
        branch = "HOLD-CONTROL-PARTIAL"
    elif states["HOLDLOW"] == "AT-K01":
        branch = "LOW-STALLS"
    elif states["HOLDLOW"] == "BETWEEN":
        branch = "LOW-PARTIAL"
    else:
        branch = {"AT-K01": "TRANSFERS", "BETWEEN": "TRANSFERS-GRADED", "AT-ISO": "DOES-NOT-TRANSFER"}[states["HOLDHIGH"]]
    stamps = ["HARNESS-CLEAN", "PATCH-BITES", "HOLD-FROM-INIT", "REPLAY-MAX-RATE-TRIANGLE", "HIGH-IS-K01-TRAJECTORY",
              "CARRIER-GROUP-OF-3", "ONE-NETWORK-RESNET", "HORIZON-100-ONLY",
              "SIGMA-PRIOR-FROZEN" if which == "SIGMA_PRIOR_frozen" else "SIGMA-INBATCH"]
    if div:
        stamps.append("DIVERGED-" + "-".join(div))
    stamps.append("POSITIVE-CONTROL-REPRODUCES" if gap >= GAP_MIN else "POSITIVE-CONTROL-FAILED")
    if gap >= GAP_MIN:
        stamps += ["%s-%s" % (a, states[a]) for a in HELD]
    for a in HELD:
        if M[a] - M["ISO"] > MATCH:
            stamps.append(a + "-ABOVE-ISO")
        if M[a] - M["k01"] < -NULL:
            stamps.append(a + "-BELOW-K01")
        if any(nonfin[(a, s)] for s in SEEDS):
            stamps.append("NONFINITE-" + a)
    if any(abs(M[a] - M["k01"]) <= NULL for a in HELD):
        stamps.append("FLOOR-READINGS-ARE-BOUNDS")
    agree = all((CT[l][1] > 0) == (CT[l][0] > 0) for l in ("P_HIGH", "P_LOW") if abs(CT[l][0]) >= RESCUE)
    stamps.append("TRAIN-AGREES" if agree else "TRAIN-DISAGREES")
    final = "FINAL: %s | %s" % (branch, " | ".join(stamps))
    print("  states: %s" % "  ".join("%s %s" % (a, states[a]) for a in HELD))
    print("  " + final)
    exp("G-DIVERGE", "  %s every arm's seed range <= 5.0 pp" % ("PASS" if not div else "FAIL"))
    exp("FINAL", final)
    margins = [("D_ISO over GAP_MIN", gap - GAP_MIN), ("k01 under K01_MAX", K01_MAX - M["k01"]),
               ("worst seed range under DIVERGED (%s)" % max(ARMS, key=lambda a: max(P[a]) - min(P[a])),
                DIVERGED - max(max(P[a]) - min(P[a]) for a in ARMS))]
    for a in HELD:
        margins.append(("%s: L - (ISO - MATCH)    (>= 0 is AT-ISO)" % a, M[a] - (M["ISO"] - MATCH)))
        margins.append(("%s: (k01 + NULL) - L     (>= 0 is AT-K01)" % a, M["k01"] + NULL - M[a]))
    print("  AT-ISO bar (ISO - 5) = %.4f;  AT-K01 bar (k01 + 2) = %.4f" % (M["ISO"] - MATCH, M["k01"] + NULL))
    for k, v in margins:
        print("    margin %-48s %+9.4f pp = %+.2f SE" % (k, v, v / se))
    print("  per-seed readings against the in-batch bars (descriptive; the branch reads arm means):")
    for a in HELD:
        print("    %-9s %s" % (a, " / ".join("s%d %.4f %s" % (s, P[a][j], st_of(P[a][j])) for j, s in enumerate(SEEDS))))
    print("  per-seed paired P_HIGH (ISO - HOLDHIGH, same seed) %s; per-seed P_LOW %s"
          % (" / ".join("%+.4f" % (P["ISO"][j] - P["HOLDHIGH"][j]) for j in range(3)),
             " / ".join("%+.4f" % (P["HOLDLOW"][j] - P["ISO"][j]) for j in range(3))))
    print("  registered accounts (243.5) -- misses of each arm's predicted band (0 = inside):")
    K01_PL, FREE_PL, PART_PL = (18.0, 28.0), (62.0, 78.0), (34.0, 56.0)
    acc = [("TRANSFERS", FREE_PL, K01_PL, FREE_PL, (34, 60)), ("TRANSFERS-GRADED", FREE_PL, PART_PL, FREE_PL, (6, 44)),
           ("DOES-NOT-TRANSFER", FREE_PL, FREE_PL, FREE_PL, (-16, 16)), ("LOW-STALLS", K01_PL, K01_PL, FREE_PL, (34, 60)),
           ("ANY-HOLD-BREAKS", K01_PL, K01_PL, K01_PL, (34, 60))]

    def miss(x, band):
        return max(0.0, band[0] - x, x - band[1])

    for nm, lo_b, hi_b, iso_b, ph in acc:
        ms_ = [("k01", miss(M["k01"], K01_PL)), ("ISO", miss(M["ISO"], FREE_PL)), ("HOLDLOW", miss(M["HOLDLOW"], lo_b)),
               ("HOLDHIGH", miss(M["HOLDHIGH"], hi_b)), ("HOLDISO", miss(M["HOLDISO"], iso_b)),
               ("P_HIGH", miss(CT["P_HIGH"][0], ph))]
        print("    %-18s %s" % (nm, "  ".join("%s %.2f" % kv for kv in ms_)))
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
    exp("BETWEEN", "BETWEEN-BATCH, NON-GATING (frozen literals; NO other batch enters any cvt7 bar or contrast):")
    for b, k0, i0 in BETWEEN:
        exp("BETWEEN", "  %s: k01 %.4f  ISO %.4f" % (b, k0, i0))
    exp("BETWEEN", "  cvt7 {99,100,101}: %s" % "  ".join("%s %.4f" % (a, M[a]) for a in ARMS))

    # ---------------------------------------------------------------- [6]
    print("\n[6] the scorer's descriptive readouts, re-derived (beta peak / pin (<= -15+1e-4 for good) / mean r epochs 15-30 / last)")
    SUMM = {}
    for a in ARMS:
        for s in SEEDS:
            recs = RECS[(a, s)]
            parts = []
            ng = 1 if a == "k01" else 2
            for k in range(ng):
                pk, pke, pin, r, last = traj_summary(recs, k)
                SUMM[(a, s, k)] = (pk, pke, pin, r, last)
                lab = "beta" if ng == 1 else ("cmpl" if k == 0 else "carr")
                parts.append("%s peak %.3f @%.1f pin %s r %.0f last %.3f" % (lab, pk, pke, "never" if pin is None else "%.1f" % pin, r, last))
            acts = [x["gh_active"] for x in recs if "gh_active" in x]
            line = "  %-9s s%-3d %s  %s" % (a, s, " | ".join(parts), "gh_active %s" % (acts[-1] if acts else "-"))
            print(line)
            exp("DESCRIPTIVE", line)
    for a in ARMS:
        cells = []
        for e in (9, 19, 29, 39, 49, 99):
            cells.append("%.1f (%.1f)" % (mean([R[(a, s)]["ep"][e][1] for s in SEEDS]), mean([R[(a, s)]["ep"][e][0] for s in SEEDS])))
        exp("DESCRIPTIVE", "  %-9s %s" % (a, " / ".join(cells)))

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
            viol("line not in scorer log [%s]: %r" % (sec, line[:160]))
    for sec in ("HEADER", "CORPUS", "COMPLETE", "G-ARGS", "G-ENV", "G-VOTEW", "G-BHOLD", "G-GHOLD", "G-STRUCT", "G-PROV",
                "LEVELS", "G-FLOOR", "G-CEIL", "G-BITE", "SIGMA", "CONTRASTS", "G-DIVERGE", "FINAL", "BETWEEN", "DESCRIPTIVE"):
        h, m = by.get(sec, [0, 0])
        print("  %-12s %3d lines found verbatim, %d not found" % (sec, h, m))
    n_fail = sum(1 for ln in logl if ln.lstrip().startswith("FAIL"))
    print("  total %d re-built lines, %d found; FAIL lines in the scorer log: %d" % (len(expect), sum(v[0] for v in by.values()), n_fail))
    rest = [ln for ln in logl if ln not in rebuilt]
    blank = sum(1 for ln in rest if not ln.strip())
    rules = sum(1 for ln in rest if ln.strip() and set(ln.strip()) == {"="})
    heads = sum(1 for ln in rest if ln.strip() and not ln.startswith(" ") and set(ln.strip()) != {"="})
    prose = len(rest) - blank - rules - heads
    print("  scorer log: %d lines; %d are re-built lines; not re-built: %d blank, %d '=' rules, %d unindented section titles,"
          " %d indented prose lines (the G-BITE clause legend, the registered licence and not-licensed text)"
          % (len(logl), len(logl) - len(rest), blank, rules, heads, prose))
    for ln in rest:
        if ln.strip() and set(ln.strip()) != {"="} and ln.startswith(" ") and any(ch.isdigit() for ch in ln) \
                and not ln.startswith("  * ") and "pp above k01" not in ln and "pp but keeps" not in ln:
            print("    not re-built, carries a digit: %r" % ln[:120])

    # ================================================================ DESCRIPTIVE
    print("\n" + "=" * 100)
    print("DESCRIPTIVE, NOT REGISTERED -- nothing below can move a level, contrast, gate, branch or stamp above")
    print("=" * 100)
    med = {}
    for a in ARMS:
        order = sorted(range(3), key=lambda j: P[a][j])
        med[a] = SEEDS[order[1]]
    EP = (10, 17, 20, 30, 40)
    print("\n[D1] APPLIED beta after the update that ends each epoch (probe record at step 500e; the value the next update"
          " uses); c = complement (group 0), g = carrier group (group 1); k01 has one shared beta")
    print("  MEDIAN seed per arm (by plateau5): %s" % "  ".join("%s s%d" % (a, med[a]) for a in ARMS))
    print("  %-9s %s" % ("arm", "  ".join("%-17s" % ("epoch %d" % e) for e in EP)))
    for a in ARMS:
        cells = []
        for e in EP:
            x = RECS[(a, med[a])][5 * e]
            if x["step"] != SPE * e:
                viol("record index %d is not epoch %d" % (5 * e, e))
            cells.append("%-17s" % (("%.2f" % x["beta"][0]) if a == "k01" else ("c %.2f / g %.2f" % (x["beta"][0], x["beta"][1]))))
        print("  %-9s %s" % (a, "  ".join(cells)))
    print("  arm mean of the three seeds:")
    for a in ARMS:
        cells = []
        for e in EP:
            c0 = mean([RECS[(a, s)][5 * e]["beta"][0] for s in SEEDS])
            cells.append("%-17s" % (("%.2f" % c0) if a == "k01" else
                                   ("c %.2f / g %.2f" % (c0, mean([RECS[(a, s)][5 * e]["beta"][1] for s in SEEDS])))))
        print("  %-9s %s" % (a, "  ".join(cells)))
    print("  mean r = exp(beta+15) over epochs [15,30), arm mean (complement / carrier group):")
    for a in ARMS:
        if a == "k01":
            print("    %-9s shared %9.1f" % (a, mean([SUMM[(a, s, 0)][3] for s in SEEDS])))
        else:
            print("    %-9s complement %9.1f   carrier group %9.1f" % (a, mean([SUMM[(a, s, 0)][3] for s in SEEDS]),
                                                                      mean([SUMM[(a, s, 1)][3] for s in SEEDS])))

    print("\n[D2] TEST (TRAIN) after 15 / 30 / 50 / 100 completed epochs (Epoch lines 14 / 29 / 49 / 99): arm mean, then TEST per seed")
    for a in ARMS:
        cells = ["%.2f (%.2f)" % (mean([R[(a, s)]["ep"][e][1] for s in SEEDS]), mean([R[(a, s)]["ep"][e][0] for s in SEEDS]))
                 for e in (14, 29, 49, 99)]
        per = ["s%d %s" % (s, "/".join("%.2f" % R[(a, s)]["ep"][e][1] for e in (14, 29, 49, 99))) for s in SEEDS]
        print("  %-9s %s   | %s" % (a, "  ".join(cells), "  ".join(per)))

    print("\n[D3] when the level is set.  settle = first Epoch line after which the value stays within 0.5 pp of its own"
          " plateau5 (TEST) / TRAIN5 (TRAIN); share = arm-mean TEST / plateau5 at Epoch lines 14 / 19 / 24 / 29 / 39")
    for a in ARMS:
        am = [mean([R[(a, s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
        tm = [mean([R[(a, s)]["ep"][e][0] for s in SEEDS]) for e in range(100)]
        settle = next(e for e in range(100) if all(abs(v - M[a]) <= 0.5 for v in am[e:]))
        tsettle = next(e for e in range(100) if all(abs(v - TM[a]) <= 0.5 for v in tm[e:]))
        ps = []
        for j, s in enumerate(SEEDS):
            tv = [R[(a, s)]["ep"][e][1] for e in range(100)]
            ps.append(next(e for e in range(100) if all(abs(v - P[a][j]) <= 0.5 for v in tv[e:])))
        share = " / ".join("%.2f" % (am[e] / M[a]) for e in (14, 19, 24, 29, 39))
        print("  %-9s TEST settle arm-mean %2d (per seed %s)  TRAIN settle arm-mean %2d  share %s"
              % (a, settle, "/".join(str(x) for x in ps), tsettle, share))

    print("\n[D4] the complement's pin by the MAGNITUDE gate (cvh1's form): first record with r = exp(beta+15) <= 2; records"
          " at or after it back above r 2; max r after it; median r over the last quarter of records; exact clamp from")
    for a in ARMS:
        for s in SEEDS:
            f2, above, medr, exact, maxr = magnitude_pin(RECS[(a, s)], 0)
            print("  %-9s s%-3d %-6s first r<=2 %-6s back above %-4s max r after %-7s median r last quarter %8.3f  exact clamp from %s"
                  % (a, s, "shared" if a == "k01" else "cmpl", "never" if f2 is None else "%.1f" % f2,
                     "-" if above is None else str(above), "-" if maxr is None else "%.3f" % maxr, medr,
                     "never" if exact is None else "%.1f" % exact))

    print("\n[D5] HOLDHIGH's free complement against ISO's complement, HOLDLOW's / HOLDISO's complement and k01's shared beta")
    print("  complement beta, arm mean, at epochs 10 / 14 / 16 / 17 / 18 / 19 / 20 / 22 / 25 / 30 / 40 / 50 / 60 / 70 / 80 / 90 / 99.8:")
    EPS = (10, 14, 16, 17, 18, 19, 20, 22, 25, 30, 40, 50, 60, 70, 80, 90)

    def cm(a, idx, k=0):
        return mean([RECS[(a, s)][idx]["beta"][k] for s in SEEDS])

    for a in ARMS:
        vals = ["%.2f" % cm(a, 5 * e) for e in EPS] + ["%.2f" % cm(a, -1)]
        print("    %-9s %s" % (a + (" (sh)" if a == "k01" else ""), " / ".join(vals)))
    for a in ("HOLDHIGH", "HOLDLOW", "HOLDISO"):
        pk = [SUMM[(a, s, 0)] for s in SEEDS]
        print("  %-9s complement peak %s @ %s; ISO's %s @ %s" % (
            a, " / ".join("%.3f" % x[0] for x in pk), " / ".join("%.1f" % x[1] for x in pk),
            " / ".join("%.3f" % SUMM[("ISO", s, 0)][0] for s in SEEDS), " / ".join("%.1f" % SUMM[("ISO", s, 0)][1] for s in SEEDS)))
    for a in ("HOLDHIGH", "k01", "ISO"):
        devs = []
        for s in SEEDS:
            recs = RECS[(a, s)]
            dall = max(abs(float(x["beta"][0]) - tri(x["step"] + 2, 8609)) for x in recs)
            d40 = max(abs(float(x["beta"][0]) - tri(x["step"] + 2, 8609)) for x in recs if x["step"] <= 40 * SPE)
            devs.append("s%d all %.3f / ep0-40 %.3f" % (s, dall, d40))
        print("  %-9s %s max|beta[0] - v_8609|: %s" % (a, "shared" if a == "k01" else "complement", "; ".join(devs)))
    im = [mean([R[("ISO", s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
    for a in ("HOLDHIGH",):
        am = [mean([R[(a, s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
        gaps = [im[e] - am[e] for e in range(100)]
        first2 = next(e for e in range(100) if all(g > 2.0 for g in gaps[e:]))
        print("  ISO - %s arm-mean TEST gap by Epoch line 10..40: %s" % (a, " ".join("%d:%+.2f" % (e, gaps[e]) for e in range(10, 41, 2))))
        print("    first Epoch line after which the gap stays > 2 pp: %d (line n = after n+1 epochs); per seed first line > 2 pp: %s"
              % (first2, " / ".join(str(next(e for e in range(100) if R[("ISO", s)]["ep"][e][1] - R[(a, s)]["ep"][e][1] > 2.0))
                                    for s in SEEDS)))
        dc = [cm(a, 5 * e) - cm("ISO", 5 * e) for e in range(0, 100)]
        firstc = next((e for e in range(100) if abs(dc[e]) > 0.1), None)
        print("  %s complement beta minus ISO's (arm means) at epochs 10..40: %s ; first epoch |diff| > 0.1: %s"
              % (a, " ".join("%d:%+.2f" % (e, dc[e]) for e in range(10, 41, 2)), firstc))
        per = []
        for s in SEEDS:
            fe = None
            for k in range(len(RECS[(a, s)])):
                if abs(RECS[(a, s)][k]["beta"][0] - RECS[("ISO", s)][k]["beta"][0]) > 0.1:
                    fe = RECS[(a, s)][k]["step"] / float(SPE)
                    break
            per.append("n/a" if fe is None else "%.1f" % fe)
        print("    per seed (same seed number, different runs) first record |cmpl diff| > 0.1 at epoch: %s" % " / ".join(per))
        dg = [cm(a, 5 * e, 1) - cm("ISO", 5 * e, 1) for e in range(0, 41)]
        print("  %s carrier-group beta minus ISO's (arm means) at epochs 2..24: %s" % (a, " ".join("%d:%+.2f" % (e, dg[e]) for e in range(2, 25, 2))))
    kt = [mean([R[("k01", s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
    ht = [mean([R[("HOLDHIGH", s)]["ep"][e][1] for s in SEEDS]) for e in range(100)]
    print("  HOLDHIGH - k01 arm-mean TEST by Epoch line 10..40: %s" % " ".join("%d:%+.2f" % (e, ht[e] - kt[e]) for e in range(10, 41, 2)))

    print("\n[D6] orientation only, between batch (PlainNet18_c100, cvt4 s90-92; not this batch's statistic): the same"
          " magnitude gate on cvt4 HEAD's and HOLDHIGH's complement")
    have = all(os.path.isdir(os.path.join(runsdir, "cvt4", "probe_cvt4-%s-s%d" % (a, s))) for a in ("HEAD", "HOLDHIGH") for s in (90, 91, 92))
    if have:
        for a in ("HEAD", "HOLDHIGH"):
            for s in (90, 91, 92):
                recs = load_recs(os.path.join(runsdir, "cvt4", "probe_cvt4-%s-s%d" % (a, s)))
                f2, above, medr, exact, maxr = magnitude_pin(recs, 0)
                pk, pke, pin, r, last = traj_summary(recs, 0)
                print("  cvt4 %-9s s%d cmpl peak %.3f @%.1f r15-30 %8.1f  first r<=2 %-6s back above %-4s exact clamp from %s"
                      % (a, s, pk, pke, r, "never" if f2 is None else "%.1f" % f2, "-" if above is None else str(above),
                         "never" if exact is None else "%.1f" % exact))
    else:
        print("  (cvt4 probe dirs not under this runsdir; skipped)")

    print("\nVIOLATIONS %d" % len(bad))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
