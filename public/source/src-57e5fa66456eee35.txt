#!/usr/bin/env python3
# =============================================================================
# cst1_cct1_registration_derivations.py -- every number CORRECTIONS 256 (`cst1`) and 257 (`cct1`) take from
# landed data, derived ONCE here before any `cst1` / `cct1` run exists.  Read-only; zero GPU; stdlib only.
#
#   S0  corpus identity (sha, rows, rows after corpus_exclusions.filter_rows)
#   S1  freshness: no `cst1-` / `cct1-` row; no row with seed 112..119
#   S2  the corpus cells the two batches sit in or next to (NON-GATING anchors for the accounts)
#   S3  the noise floors, through filter_rows: SIGMA_R18ALL (ResNet18_c100 / CIFAR-100, ms 1e-3, alpha0 1e-6,
#       cVT8's definition) and SIGMA_R18C10 (ResNet18 / CIFAR-10, same cell) -- both as FILTERED and NAIVE
#   S4  CALIBRATION of the carrier-vote statistics (analysis/cCV0_carrier_vote_core.py, the SAME code both scorers
#       run) on every landed probe run at the mechanism cell that ran the plain scalar or layerwise partition:
#       ctd1 sc/lay s18-20, ciso1 k01/k62 s21-23, cdep1 k01/k62 s24-26 (18 runs, 9,000 records)
#
# USAGE  python3 analysis/cst1_cct1_registration_derivations.py <runsdir>      (runsdir holds ctd1/, ciso1/, cdep1/)
# =============================================================================
import csv
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cCV0_carrier_vote_core as core  # noqa: E402

REPO = os.path.dirname(HERE)
CSV = os.path.join(REPO, "results", "all_runs.csv")
TSV = os.path.join(REPO, "results", "CORPUS-EXCLUSIONS.tsv")

CALIB = [("ctd1", "sc", "scalar", (18, 19, 20)), ("ciso1", "k01", "scalar", (21, 22, 23)),
         ("cdep1", "k01", "scalar", (24, 25, 26)),
         ("ctd1", "lay", "layerwise", (18, 19, 20)), ("ciso1", "k62", "layerwise", (21, 22, 23)),
         ("cdep1", "k62", "layerwise", (24, 25, 26))]


def main():
    runsdir = sys.argv[1] if len(sys.argv) > 1 else None
    print("core sha256 %s" % core.core_sha())
    print("\n[S0] CORPUS")
    raw = list(csv.DictReader(open(CSV)))
    filt = core.read_corpus(CSV)
    print("  all_runs.csv sha256 %s  rows %d  after filter_rows %d  (TSV sha256 %s)"
          % (core.file_sha(CSV), len(raw), len(filt), core.file_sha(TSV)))

    print("\n[S1] FRESHNESS")
    print("  rows named cst1-*: %d   cct1-*: %d" % (sum(r["run"].startswith("cst1-") for r in raw),
                                                  sum(r["run"].startswith("cct1-") for r in raw)))
    for s in range(112, 120):
        print("  rows with seed %d: %d" % (s, sum(r.get("seed") == str(s) for r in raw)))

    print("\n[S2] CELLS (filtered; usable rows; plateau5)")
    ISO = "sets:1-49,51-52,54-58,60-62/layer4.0.bn2.weight,layer4.0.shortcut.1.weight,layer4.1.bn2.weight"
    CTL = "sets:1-46,49-55,57-62/layer4.0.bn1.weight,layer4.0.bn1.bias,layer4.1.bn1.weight"
    for net, ds, ms, a0, g, tag in (
            ("ResNet18_c100", "CIFAR100", "3e-4", "1e-6", "scalar", "S2 k01 cell"),
            ("ResNet18_c100", "CIFAR100", "3e-4", "1e-6", "layerwise", "S2 layerwise (not run)"),
            ("ResNet18_c100", "CIFAR100", "3e-4", "1e-6", ISO, "S2 ISO cell (empty)"),
            ("ResNet18_c100", "CIFAR100", "3e-4", "1e-6", CTL, "S2 CTL cell (empty)"),
            ("ResNet18_c100", "CIFAR100", "1e-3", "1e-6", "scalar", "mechanism k01"),
            ("ResNet18_c100", "CIFAR100", "1e-3", "1e-6", "layerwise", "mechanism layerwise"),
            ("ResNet18_c100", "CIFAR100", "1e-3", "1e-6", ISO, "mechanism ISO"),
            ("ResNet18_c100", "CIFAR100", "1e-3", "1e-6", CTL, "mechanism CTL (cdep1 DEPTH)"),
            ("ResNet18", "CIFAR10", "1e-3", "1e-6", "scalar", "S3 k01 cell"),
            ("ResNet18", "CIFAR10", "1e-3", "1e-6", "layerwise", "S3 kL cell")):
        rows = core.cell_rows(filt, net, ds, ms, a0, g)
        v = [float(r["plateau5"]) for r in rows]
        print("  %-28s n=%-3d mean %s sd %s range %s  batches %s" % (
            tag, len(v), core.fmt(core.mean(v)), core.fmt(core.sd(v)),
            ("[%.3f, %.3f]" % (min(v), max(v))) if v else "-",
            ",".join(sorted(set(r["run"].split("-")[0].split("_")[0] for r in rows)))))

    print("\n[S3] NOISE FLOORS (pooled within-cell SD, any granularity, standard cell ms 1e-3 / alpha0 1e-6)")
    for net, ds in (("ResNet18_c100", "CIFAR100"), ("ResNet18", "CIFAR10")):
        s_f, df_f, nc_f = core.pooled_sigma(filt, net, ds)
        s_n, df_n, nc_n = core.pooled_sigma(raw, net, ds)
        print("  %-14s %-9s FILTERED %r (df %d, %d cells)   NAIVE %r (df %d, %d cells)"
              % (net, ds, s_f, df_f, nc_f, s_n, df_n, nc_n))
    pair = core.cell_rows(filt, "ResNet18", "CIFAR10", "1e-3", "1e-6", "scalar") + \
        core.cell_rows(filt, "ResNet18", "CIFAR10", "1e-3", "1e-6", "layerwise")
    sp, dfp, ncp = core.pooled_sigma(pair, "ResNet18", "CIFAR10")
    print("  %-14s %-9s the S3 PAIR only (hier '' scalar + layerwise cells) FILTERED %r (df %d, %d cells) -- the "
          "cct1 disclosure sigma" % ("ResNet18", "CIFAR10", sp, dfp, ncp))
    s3, df3, nc3 = core.pooled_sigma(filt, "ResNet18_c100", "CIFAR100", ms="3e-4", a0="1e-6")
    print("  %-14s %-9s ms 3e-4 FILTERED %r (df %d, %d cells) -- disclosure only" % ("ResNet18_c100", "CIFAR100",
                                                                                     s3, df3, nc3))

    print("\n[S4] CALIBRATION ON LANDED PROBE RECORDS (the scorers' own vote_stats / decomposition)")
    if not runsdir:
        print("  SKIP: no runsdir given")
        return
    for typ in ("scalar", "layerwise"):
        pooled = []
        for batch, arm, t, seeds in CALIB:
            if t != typ:
                continue
            for s in seeds:
                P = core.read_probe(runsdir, batch, arm, s)
                if P is None:
                    print("  MISSING %s-%s-s%d" % (batch, arm, s))
                    continue
                groups = core.parse_sets(typ)
                d = core.decomposition(P["recs"], groups, 1e-3)
                st = core.vote_stats(P["recs"])
                pooled += P["recs"]
                print("  %-18s recs %d decomp_ok %s  DOM_C %s TOP3_C %s DOM_TOP3 %s SHARE_C %s R_T %s CAR_EQ %s "
                      "PINNED %s DOM_C|pin %s" % ("%s-%s-s%d" % (batch, arm, s), len(P["recs"]), core.decomposition_ok(d),
                                                 core.fmt(st["DOM_C"]), core.fmt(st["TOP3_C"]), core.fmt(st["DOM_TOP3"]),
                                                 core.fmt(st["SHARE_C"]), core.fmt(st["R_T"]), core.fmt(st["CAR_EQ"]),
                                                 core.fmt(st["PINNED_FRAC"]), core.fmt(st["DOM_C_PINNED"])))
        print("  POOLED %s (%d records):" % (typ, len(pooled)))
        core.describe_stats(typ, core.vote_stats(pooled))


if __name__ == "__main__":
    main()
