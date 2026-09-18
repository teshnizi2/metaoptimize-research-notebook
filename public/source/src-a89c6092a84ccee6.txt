#!/usr/bin/env python3
"""cmg1_registration_derivations.py -- every number CORRECTIONS 259 (`cmg1`) freezes, re-derived from the record.

`cmg1` (Track D, LIMITS-PREP 5.1 N3): on ResNet18_c100 at the mechanism cell, inside the singleton ("layerwise")
grouping run through the blockwise code path, does moving the three carrier BN scales {50,53,59} into ONE step-size group
with a fixed nine-tensor layer4 neighbour set N hurt, and does moving the count / numel / width / depth / norm-layer
matched non-carrier BN triple {47,48,56} into the SAME N hurt less?

Read-only.  Stdlib only.  Nothing is written.  Usage:

    python3 analysis/cmg1_registration_derivations.py [<runsdir>] [<all_runs.csv>]

  S1  freshness: no `cmg1-` row in the CSV, no row with seed 124-127 (and, with <runsdir>, no .out ARGS line with them)
  S2  the noise floor (227.6's SIGMA_R18ALL) through corpus_exclusions.filter_rows (OPERATIONS 36), and the naive trap,
      on the REGISTRATION corpus = the first REG_ROWS data rows of the CSV (the ingest appends; sha of those rows printed)
  S3  the in-corpus analogues (filtered, registration corpus): layerwise / scalar / ISO / cbl1 chunk ladder / cts1 cliff /
      resnet18_blocks at the mechanism cell -- the basis of every predicted level in 259.5
  S4  PREMISE L (the LAYERWISE trajectory): on the 9 landed layerwise PROBE_TENSOR runs (ciso1 / cdep1 / ciso2 k62) the
      Lion sign input L_i = b2*m_i + (1-b2)*z_i per tensor; per record the share on which adding C (resp. T) to N flips
      the sign of N's sum, the share on which the merged sum is positive (= votes beta DOWN), |sum| magnitudes, and the
      median late beta of every layer4 tensor
  S5  PREMISE S (the SCALAR trajectory): the same shares on the 9 landed ResNet18_c100 scalar PROBE_TENSOR runs
      (ciso1 / cdep1 / ciso2 k01), where every tensor shares one step size -- the state a merged group is in
  S6  G-PARTITION non-vacuity on REAL records: synthetic MCAR / MCTL / KLS records built from the real z_tensor /
      m_tensor of the landed k62 runs; each read as its own arm passes, read as either other arm fails
"""
import csv
import glob
import hashlib
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import corpus_exclusions as CE  # noqa: E402  (imported UNEDITED)

REG_ROWS = 3127
N = [46, 49, 51, 52, 54, 55, 57, 58, 60]
C = [50, 53, 59]
T = [47, 48, 56]
SEEDS = ("124", "125", "126", "127")
B2 = 0.9


def sgn(x):
    return (x > 0) - (x < 0)


def reg_rows(csvpath):
    raw = open(csvpath, newline="").read().splitlines(True)
    head, body = raw[0], raw[1:]
    if len(body) < REG_ROWS:
        raise SystemExit("CSV has %d data rows < REG_ROWS %d" % (len(body), REG_ROWS))
    blob = (head + "".join(body[:REG_ROWS])).encode()
    rows = list(csv.DictReader([head] + body[:REG_ROWS]))
    return rows, hashlib.sha256(blob).hexdigest()


def cell_ok(r):
    return (r.get("superseded") == "0" and r.get("collapsed") == "0" and r.get("complete") == "1"
            and (r.get("plateau5") or "").strip() and r.get("epochs_requested") == "100" and r.get("augment") == "1"
            and r.get("beta_clip") == "-15:-2.3026" and r.get("meta_stepsize") == "1e-3" and r.get("alpha0") == "1e-6"
            and r.get("batch_size") == "100" and r.get("network") == "ResNet18_c100" and r.get("dataset") == "CIFAR100"
            and r.get("base") == "SGDm" and r.get("meta") == "Lion" and r.get("gamma") == "1" and r.get("hier") == ""
            and r.get("lam") == "na" and r.get("eta_ratio") == "na")


def floor(rows):
    filt = CE.filter_rows(rows)
    return {"rows": len(rows), "filtered": len(filt), "R18ALL": CE._pooled(filt, "ResNet18_c100"),
            "R18ALL_naive": CE._pooled(rows, "ResNet18_c100")}


def analogues(rows):
    filt = CE.filter_rows(rows)
    want = {"layerwise": "layerwise", "scalar": "scalar", "ISO": CE_ISO, "cbl1 [2]x30+[1,1]": CHUNK2,
            "cbl1 [4]x14+[3,3]": CHUNK4, "cbl1 [8]x6+[7,7]": CHUNK8, "cbl1 [16,16,15,15]": "[16,16,15,15]",
            "cts1 [49,13]": "[49,13]", "cts1 [50,12]": "[50,12]", "resnet18_blocks": "resnet18_blocks"}
    out = {}
    for lab, g in want.items():
        v = [float(r["plateau5"]) for r in filt if cell_ok(r) and r["granularity"] == g]
        out[lab] = (len(v), statistics.mean(v) if v else float("nan"), statistics.stdev(v) if len(v) > 1 else float("nan"))
    return out


CE_ISO = "sets:1-49,51-52,54-58,60-62/layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight"
CHUNK2 = "[" + ",".join(["2"] * 30 + ["1", "1"]) + "]"
CHUNK4 = "[" + ",".join(["4"] * 14 + ["3", "3"]) + "]"
CHUNK8 = "[" + ",".join(["8"] * 6 + ["7", "7"]) + "]"


def premise(dirs, per_tensor_keys):
    """per_tensor_keys: ('mom_pre','z_agg') for layerwise records (index 0 list of 62), or ('m_tensor','z_tensor')."""
    agg = []
    for d in dirs:
        c = dict(n=0, flipC=0, flipT=0, posN=0, posGC=0, posGT=0, posC=0, posT=0)
        mag = dict(C=0.0, T=0.0, N=0.0)
        for line in open(os.path.join(d, "probe.jsonl")):
            r = json.loads(line)
            mk, zk = per_tensor_keys
            if mk not in r:
                continue
            m = r[mk][0] if mk == "mom_pre" else r[mk]
            z = r[zk][0] if zk == "z_agg" else r[zk]
            if len(m) != 62 or len(z) != 62:
                continue
            L = [B2 * m[i] + (1 - B2) * z[i] for i in range(62)]
            sN = sum(L[i - 1] for i in N)
            sC = sum(L[i - 1] for i in C)
            sT = sum(L[i - 1] for i in T)
            c["n"] += 1
            c["flipC"] += sgn(sN + sC) != sgn(sN)
            c["flipT"] += sgn(sN + sT) != sgn(sN)
            c["posN"] += sN > 0
            c["posGC"] += (sN + sC) > 0
            c["posGT"] += (sN + sT) > 0
            c["posC"] += sC > 0
            c["posT"] += sT > 0
            mag["C"] += abs(sC)
            mag["T"] += abs(sT)
            mag["N"] += abs(sN)
        agg.append((os.path.basename(d), c, mag))
    return agg


def late_beta(d):
    vals = {i: [] for i in range(46, 63)}
    for line in open(os.path.join(d, "probe.jsonl")):
        r = json.loads(line)
        if r["step"] >= 40000 and "beta_pre" in r:
            for i in vals:
                vals[i].append(r["beta_pre"][0][i - 1])
    return {i: statistics.median(v) for i, v in vals.items() if v}


# ---- S6: the G-PARTITION reader, re-typed here so the scorer's copy can be checked against it -----------------------
def groups_of(arm):
    if arm == "KLS":
        return [[i] for i in range(1, 63)]
    merged = sorted(N + (C if arm == "MCAR" else T))
    single = T if arm == "MCAR" else C
    return [[i] for i in range(1, 46)] + [merged] + [[i] for i in single] + [[61], [62]]


def synth_record(z, m, arm):
    g = groups_of(arm)
    return {"z_tensor": z, "m_tensor": m, "z_agg": [[sum(z[i - 1] for i in grp) for grp in g]],
            "mom_pre": [[sum(m[i - 1] for i in grp) for grp in g]]}


def partition_ok(rec, arm, tol_single=1e-6, tol_merged=1e-2):
    g = groups_of(arm)
    za = rec["z_agg"][0]
    if len(za) != len(g):
        return False
    z = rec["z_tensor"]
    for k, grp in enumerate(g):
        s = sum(z[i - 1] for i in grp)
        tol = tol_single if len(grp) == 1 else tol_merged
        if abs(s - za[k]) > tol * max(abs(za[k]), abs(s), 1e-30):
            return False
    return True


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else None
    csvpath = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "results", "all_runs.csv")
    rows, rsha = reg_rows(csvpath)
    print("cmg1 registration derivations (CORRECTIONS 259)")
    print("  csv %s ; registration corpus = first %d data rows, sha256 %s" % (csvpath, REG_ROWS, rsha))

    print("\nS1  FRESHNESS")
    allrows = list(csv.DictReader(open(csvpath)))
    print("  cmg1- rows in the CSV: %d" % sum(r["run"].startswith("cmg1-") for r in allrows))
    print("  rows carrying seed 124/125/126/127: %d ; max integer seed %d"
          % (sum(r["seed"] in SEEDS for r in allrows), max(int(r["seed"]) for r in allrows if r["seed"].isdigit())))
    if runs:
        hit = 0
        for f in glob.glob(os.path.join(runs, "*.out")):
            with open(f, errors="replace") as fh:
                for line in fh:
                    if line.startswith("ARGS:"):
                        tok = line.split()
                        if "--seed" in tok and tok[tok.index("--seed") + 1] in SEEDS:
                            hit += 1
                        break
        print("  .out ARGS lines under %s with --seed 124-127: %d ; cmg1-*.out: %d"
              % (runs, hit, len(glob.glob(os.path.join(runs, "cmg1-*.out")))))

    print("\nS2  THE NOISE FLOOR through corpus_exclusions.filter_rows (227.6's SIGMA_R18ALL), registration corpus")
    F = floor(rows)
    print("  %d rows, %d after filter_rows; CORPUS-EXCLUSIONS.tsv sha %s" % (F["rows"], F["filtered"],
          hashlib.sha256(open(CE.DEFAULT_TSV, "rb").read()).hexdigest()))
    print("  SIGMA_R18ALL  FILTERED %r (df %d, %d cells)" % F["R18ALL"])
    print("  SIGMA_R18ALL  NAIVE    %r (df %d, %d cells)  <- the trap, NOT used" % F["R18ALL_naive"])
    s = F["R18ALL"][0]
    print("  SE_ARM_DIFF (4 seeds per arm) = sigma*sqrt(2/4) = %.6f ; READ_BAR = 2 SE = %.6f" % (s * math.sqrt(0.5), 2 * s * math.sqrt(0.5)))

    print("\nS3  IN-CORPUS ANALOGUES at the mechanism cell (filtered, registration corpus): n, mean plateau5, sd")
    for lab, (n, mu, sd) in analogues(rows).items():
        print("  %-22s n %2d  mean %8.4f  sd %s" % (lab, n, mu, "%.4f" % sd if sd == sd else "-"))

    if not runs:
        print("\nS4-S6 SKIPPED (no <runsdir>)")
        return
    kl = sorted(glob.glob(os.path.join(runs, "c*", "probe_c*-k62-s*")))
    kl = [d for d in kl if os.path.basename(d).split("-")[0] in ("probe_ciso1", "probe_cdep1", "probe_ciso2")]
    k1 = sorted(glob.glob(os.path.join(runs, "c*", "probe_c*-k01-s*")))
    k1 = [d for d in k1 if os.path.basename(d).split("-")[0] in ("probe_ciso1", "probe_cdep1", "probe_ciso2")]

    for lab, dirs, keys in (("S4  PREMISE L -- the LAYERWISE trajectory", kl, ("mom_pre", "z_agg")),
                            ("S5  PREMISE S -- the SCALAR trajectory", k1, ("m_tensor", "z_tensor"))):
        print("\n%s (%d runs); L = 0.9*m + 0.1*z; positive sum = the Lion update moves beta DOWN" % (lab, len(dirs)))
        print("  %-24s %5s %6s %6s %6s %6s %6s %6s %6s  %9s %9s %9s" % ("run", "n", "flipC", "flipT", "posN", "posGC",
              "posGT", "posC", "posT", "|sumC|", "|sumT|", "|sumN|"))
        tot = dict(n=0, flipC=0, flipT=0, posN=0, posGC=0, posGT=0, posC=0, posT=0)
        for name, c, mag in premise(dirs, keys):
            n = c["n"]
            for k in tot:
                tot[k] += c[k]
            print("  %-24s %5d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f  %9.3e %9.3e %9.3e"
                  % (name, n, c["flipC"] / n, c["flipT"] / n, c["posN"] / n, c["posGC"] / n, c["posGT"] / n,
                     c["posC"] / n, c["posT"] / n, mag["C"] / n, mag["T"] / n, mag["N"] / n))
        n = tot["n"]
        print("  %-24s %5d %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f" % ("POOLED", n, tot["flipC"] / n, tot["flipT"] / n,
              tot["posN"] / n, tot["posGC"] / n, tot["posGT"] / n, tot["posC"] / n, tot["posT"] / n))
        if dirs is kl:
            print("  median beta_pre over steps >= 40000 (epochs 80-99), per layer4 / head tensor, per run:")
            for d in dirs:
                b = late_beta(d)
                print("    %-22s %s" % (os.path.basename(d), " ".join("%d:%.2f" % (i, b[i]) for i in sorted(b))))

    print("\nS6  G-PARTITION NON-VACUITY on REAL records (synthetic group sums from the landed k62 runs' per-tensor terms)")
    arms = ("KLS", "MCAR", "MCTL")
    npass = {a: 0 for a in arms}
    nfalse = {(a, b): 0 for a in arms for b in arms if a != b}
    nrec = 0
    for d in kl:
        for line in open(os.path.join(d, "probe.jsonl")):
            r = json.loads(line)
            z = r["z_agg"][0]
            m = r["mom_pre"][0]
            nrec += 1
            for a in arms:
                rec = synth_record(z, m, a)
                npass[a] += partition_ok(rec, a)
                for b in arms:
                    if b != a:
                        nfalse[(a, b)] += partition_ok(rec, b)
    print("  %d real records x 3 arms" % nrec)
    for a in arms:
        print("  %-5s read as itself passes on %d / %d" % (a, npass[a], nrec))
    for (a, b), v in sorted(nfalse.items()):
        print("  %-5s read as %-5s passes on %d / %d   (must be 0)" % (a, b, v, nrec))


if __name__ == "__main__":
    main()
