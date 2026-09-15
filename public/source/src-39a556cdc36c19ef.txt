#!/usr/bin/env python3
# =============================================================================
# cms1_ms1e4_stratum_census.py
#
#   WHAT DOES THE CIFAR-100 meta_stepsize=1e-4 STRATUM CONTAIN *NOW*, BY ARM
#   AND BY alpha0, WITH n?  -- the census that replaces cU1's frozen premise
#   ("no primary-set arm exists at ms=1e-4 on CIFAR-100; every ms=1e-4 row
#   sits at alpha0=1e-3") after `cru1` (CORRECTIONS 174) falsified it BY
#   CONSTRUCTION.
#
# Cycle 144.  ZERO GPU.  This scorer has NO RUNS OF ITS OWN: it reads rows that
# already exist in results/all_runs.csv.  It exists so that the corrected prose
# in docs/ can state the CURRENT fact ("the stratum contains ...") rather than
# merely negate the old one ("... is no longer true").
#
# -----------------------------------------------------------------------------
# RULE 21 STATUS -- STATED FIRST
# -----------------------------------------------------------------------------
# THIS FILE DOES NOT CARRY THE RULE 21 LABEL.  Precedent: cQ1 (CORRECTIONS 149),
# cS1 (159), cZ1 (170).  RULE 21's proof is "the scorer's commit precedes the
# earliest sacct Submit of its own batch".  This scorer has no batch, so that
# proof DOES NOT EXIST and is not claimed.
#
# WHAT IS CLAIMED, AND ONLY THIS:
#   * COMMIT-BEFORE-FIRST-EXECUTION.  Every filter, regex, token rule and
#     prediction below is fixed in this one committed file, and the file is run
#     UNEDITED afterwards.  The margin (commit timestamp -> first execution
#     timestamp) is recorded in docs/CORRECTIONS.md.  The file was
#     syntax-checked with `ast.parse` before the commit; that parses and does
#     not execute.
#
# WHAT WAS VISIBLE TO THE AUTHOR WHILE DESIGNING -- DISCLOSED, NOT LIFTED:
#   * From the PROSE of CORRECTIONS 168 / 174 and docs/STATUS.md cycle 143 (not
#     from the CSV): cru1 = 60 rows, arms `scalar` and `layerwise`, seeds
#     {15,16,17}, ladder A alpha0=1e-3 at ms in {1e-5,3e-5,1e-4,3e-4,1e-3,3e-3},
#     ladder B alpha0=1e-6 at ms in {1e-4,3e-4,1e-3,3e-3}; the pre-cru1
#     ms=1e-4 stratum = 20 rows {chunk771 7, nodewise 7, chunk2293 3,
#     nodewise1d 3}, all at alpha0=1e-3; the corpus's best CIFAR-100
#     plateau5 = 72.408 (`gm2-ch-s1`, chunk771); layerwise at
#     ms=1e-4/alpha0=1e-3 observed 71.0827 (174.9).  The predictions in the
#     PREDICTIONS block are arithmetic on that prose (20 + 2 arms x 2 alpha0 x
#     3 seeds = 32) and were written BEFORE this file touched the CSV's
#     ms=1e-4 rows.
#   * From the CSV, BEFORE commit: the header line, and the DISTINCT LABEL SETS
#     of `granularity`, `meta_stepsize`, `alpha0`, `dataset`, `network`
#     (labels only -- no counts, no plateau values, no row read in full).
#     They were read so that the cut-position regex below is written against
#     the corpus's real spellings.
#   * cU1_alpha0_granularity_score.py had NOT been executed by this author at
#     the time this file was written; only its source was read.
#
# STANDING RULE 16: run this file UNEDITED and quote its output.  Passing its
# documented arguments (--csv, --selftest) is NOT editing it.
#
# -----------------------------------------------------------------------------
# FROZEN PREMISES -- STATED SO THAT THEY STAY TRUE UNDER FUTURE INGESTS
# -----------------------------------------------------------------------------
#   P1  EVERY row of the CSV is read.  The RAW census EXCLUDES NOTHING: every
#       CIFAR-100 row at ms=1e-4 is counted, whatever its flags, and the flags
#       are printed beside the count.  A second, cU1-STRATUM census applies
#       cU1's frozen filters VERBATIM (network ResNet18_c100, base SGDm, meta
#       Lion, gamma 1, augment 1, beta_clip -15:-2.3026, batch_size 100,
#       hier '', collapsed 0, complete 1, superseded 0, epochs_done 100) so
#       that the corrected sentence is stated on cU1's own terms.
#   P2  NO count is frozen as a constant.  Every count is corpus-conditional
#       and is printed with the row count it was computed from.  A future
#       ingest changes the numbers; it cannot make this file's rules wrong.
#   P3  The FINAL tokens are RULES over the live corpus (below), meaningful
#       under any corpus.  The PREDICTIONS block is the one place a design-time
#       expectation is frozen; it is a prediction about the corpus AS OF THE
#       ROW COUNT PRINTED, and its failure under a later ingest is a CENSUS
#       CHANGE to be recorded, never a defect of this file.
#   P4  `plateau5` is PRIMARY.  `final_train` is printed ALONGSIDE it as the
#       CSV's train column (the campaign's `train5` is not in the CSV).  The
#       CSV `plateau` column is not read.  `best_test` is not read.
#   P5  Batch = `run` up to the first '-', exactly as cU1's batch_of().
#   P6  "Cut-position arm" = granularity matching ^\[\d+,\d+\]$ (the two-group
#       prefix cuts, cU1's "[k,62-k] arms").  `sets:` and `tn:` name-list
#       partitions are reported separately as NAME-LIST and are NOT counted as
#       cut-position, nor as any primary arm.
#
# -----------------------------------------------------------------------------
# THE TOKENS -- RULES FIXED HERE
# -----------------------------------------------------------------------------
#   HEADLINE-ARMS-AT-MS1E4:PRESENT   iff  `scalar` >= 1 row AND `layerwise` >= 1
#                                         row in the cU1-STRATUM at ms=1e-4;
#                                    else ABSENT.
#   MS1E4-ALPHA0-LEVELS=k            k = number of distinct alpha0 among RAW
#                                    CIFAR-100 ms=1e-4 rows.
#   ALIAS:BROKEN                     iff some primary arm (scalar,
#                                    resnet18_blocks, layerwise) has
#                                    cU1-STRATUM rows at >= 2 distinct ms at ONE
#                                    fixed alpha0 -- i.e. an ms contrast at
#                                    fixed alpha0 exists for a primary arm,
#                                    which is exactly what 160 said did not;
#                                    else HOLDS.
#   BLK6-AT-MS1E4:PRESENT/ABSENT     `resnet18_blocks` >= 1 row, cU1-STRATUM,
#                                    ms=1e-4.
#   CUTPOS-AT-MS1E4:PRESENT/ABSENT   any RAW CIFAR-100 ms=1e-4 row whose
#                                    granularity matches P6.
#   PREDICTIONS:HOLD/FAIL            all of PRED-A..PRED-D hold.
#
# -----------------------------------------------------------------------------
# THE PREDICTIONS -- FROZEN AT DESIGN TIME FROM THE PROSE OF 168/174 ONLY
# -----------------------------------------------------------------------------
#   PRED-A  RAW CIFAR-100 ms=1e-4 rows == 20 + 12 == 32.
#   PRED-B  cU1-STRATUM ms=1e-4 rows: scalar n=3 at alpha0=1e-6, n=3 at
#           alpha0=1e-3; layerwise n=3 at each; all 12 named `cru1-*`.
#   PRED-C  cU1-STRATUM ms=1e-4 rows: resnet18_blocks n=0; RAW cut-position
#           n=0.
#   PRED-D  RAW CIFAR-100 ms=1e-4 rows NOT named `cru1-*` are exactly
#           {chunk771 7, nodewise 7, chunk2293 3, nodewise1d 3}, all at
#           alpha0=1e-3 (168.2's census, unchanged by this batch).
#
# USAGE
#   python3 analysis/cms1_ms1e4_stratum_census.py [--csv PATH] [--selftest]
# =============================================================================

import argparse
import collections
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "results", "all_runs.csv")

PRIMARY_GRANS = ("scalar", "resnet18_blocks", "layerwise")
CUTPOS_RE = re.compile(r"^\[\d+,\d+\]$")
NAMELIST_RE = re.compile(r"^(tn:)?sets:")
MS_TARGET = "1e-4"

PRED_A_ROWS = 32
PRED_B = {("scalar", "1e-6"): 3, ("scalar", "1e-3"): 3,
          ("layerwise", "1e-6"): 3, ("layerwise", "1e-3"): 3}
PRED_D = {("chunk771", "1e-3"): 7, ("nodewise", "1e-3"): 7,
          ("chunk2293", "1e-3"): 3, ("nodewise1d", "1e-3"): 3}


# ------------------------------------------------------------------ data ----
def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def batch_of(row):
    return row["run"].split("-")[0]


def is_c100(r):
    return r["dataset"] == "CIFAR100"


def in_cu1_stratum(r):
    """cU1's frozen filters, verbatim, MINUS its ms / granularity / alpha0
    restrictions (those are the axes being censused here)."""
    return (r["dataset"] == "CIFAR100"
            and r["network"] == "ResNet18_c100"
            and r["base"] == "SGDm"
            and r["meta"] == "Lion"
            and r["gamma"] == "1"
            and r["augment"] == "1"
            and r["beta_clip"] == "-15:-2.3026"
            and r["batch_size"] == "100"
            and r["hier"] == ""
            and r["collapsed"] == "0"
            and r["complete"] == "1"
            and r["superseded"] == "0"
            and r["epochs_done"] == "100")


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def mean_or_none(vals):
    v = [x for x in vals if x is not None]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


def fmt(x):
    return "     -   " if x is None else "%9.4f" % x


def arm_class(g):
    if g in PRIMARY_GRANS:
        return "PRIMARY"
    if CUTPOS_RE.match(g):
        return "CUT-POSITION"
    if NAMELIST_RE.match(g):
        return "NAME-LIST"
    return "OTHER"


# ---------------------------------------------------------------- census ----
def per_cell(rows):
    """(granularity, alpha0) -> dict(n, batches, seeds, p5 mean, tr mean, flags)."""
    cells = collections.OrderedDict()
    for r in sorted(rows, key=lambda r: (arm_class(r["granularity"]) != "PRIMARY",
                                         r["granularity"], r["alpha0"], r["run"])):
        k = (r["granularity"], r["alpha0"])
        c = cells.setdefault(k, {"n": 0, "batches": collections.Counter(),
                                 "seeds": [], "p5": [], "tr": [],
                                 "flags": collections.Counter()})
        c["n"] += 1
        c["batches"][batch_of(r)] += 1
        c["seeds"].append(r["seed"])
        c["p5"].append(fnum(r["plateau5"]))
        c["tr"].append(fnum(r["final_train"]))
        for f in ("collapsed", "complete", "superseded"):
            c["flags"]["%s=%s" % (f, r[f])] += 1
        c["flags"]["hier=%s" % (r["hier"] or "''")] += 1
        c["flags"]["epochs_done=%s" % r["epochs_done"]] += 1
    return cells


def print_cells(title, cells, n_rows):
    print("\n-- %s (%d rows) %s" % (title, n_rows, "-" * max(1, 60 - len(title))))
    if not cells:
        print("   (empty)")
        return
    print("   %-14s %-14s %5s  %9s %9s   %s" %
          ("granularity", "alpha0", "n", "mean p5", "mean tr", "batches / seeds / flags"))
    for (g, a), c in cells.items():
        p5, n5 = mean_or_none(c["p5"])
        tr, ntr = mean_or_none(c["tr"])
        b = ",".join("%s:%d" % kv for kv in sorted(c["batches"].items()))
        s = ",".join(sorted(c["seeds"], key=lambda x: (len(x), x)))
        fl = " ".join("%s:%d" % kv for kv in sorted(c["flags"].items()))
        print("   %-14s %-14s %5d  %s %s   [%s] %s" % (g, a, c["n"], fmt(p5), fmt(tr), arm_class(g), b))
        print("   %-14s %-14s %5s  seeds {%s}" % ("", "", "", s))
        print("   %-14s %-14s %5s  %s" % ("", "", "", fl))


def tokens(rows):
    raw4 = [r for r in rows if is_c100(r) and r["meta_stepsize"] == MS_TARGET]
    str4 = [r for r in raw4 if in_cu1_stratum(r)]
    n_sc = sum(1 for r in str4 if r["granularity"] == "scalar")
    n_lay = sum(1 for r in str4 if r["granularity"] == "layerwise")
    n_blk = sum(1 for r in str4 if r["granularity"] == "resnet18_blocks")
    headline = "PRESENT" if (n_sc >= 1 and n_lay >= 1) else "ABSENT"
    a0_levels = len(set(r["alpha0"] for r in raw4))
    # ALIAS rule: any primary arm with >= 2 distinct ms at one fixed alpha0 (cU1-STRATUM)
    ms_by = collections.defaultdict(set)
    for r in rows:
        if in_cu1_stratum(r) and r["granularity"] in PRIMARY_GRANS:
            ms_by[(r["granularity"], r["alpha0"])].add(r["meta_stepsize"])
    contrasts = sorted((g, a, sorted(v)) for (g, a), v in ms_by.items() if len(v) >= 2)
    alias = "BROKEN" if contrasts else "HOLDS"
    blk = "PRESENT" if n_blk >= 1 else "ABSENT"
    cut = "PRESENT" if any(CUTPOS_RE.match(r["granularity"]) for r in raw4) else "ABSENT"
    return dict(headline=headline, a0_levels=a0_levels, alias=alias, blk=blk, cut=cut,
                contrasts=contrasts, n_sc=n_sc, n_lay=n_lay, n_blk=n_blk,
                raw4=raw4, str4=str4)


def predictions(rows):
    raw4 = [r for r in rows if is_c100(r) and r["meta_stepsize"] == MS_TARGET]
    str4 = [r for r in raw4 if in_cu1_stratum(r)]
    out = []
    # A
    out.append(("PRED-A raw CIFAR-100 ms=1e-4 rows == %d" % PRED_A_ROWS,
                len(raw4) == PRED_A_ROWS, "got %d" % len(raw4)))
    # B
    got_b = collections.Counter((r["granularity"], r["alpha0"]) for r in str4
                                if r["granularity"] in ("scalar", "layerwise"))
    all_cru1 = all(r["run"].startswith("cru1-") for r in str4
                   if r["granularity"] in ("scalar", "layerwise"))
    out.append(("PRED-B cU1-stratum ms=1e-4 scalar/layerwise n=3 at each alpha0, all cru1-*",
                dict(got_b) == PRED_B and all_cru1,
                "got %s, all cru1-*: %s" % (dict(sorted(got_b.items())), all_cru1)))
    # C
    n_blk = sum(1 for r in str4 if r["granularity"] == "resnet18_blocks")
    n_cut = sum(1 for r in raw4 if CUTPOS_RE.match(r["granularity"]))
    out.append(("PRED-C resnet18_blocks n=0 (stratum) and cut-position n=0 (raw) at ms=1e-4",
                n_blk == 0 and n_cut == 0, "got blk6 %d, cut-position %d" % (n_blk, n_cut)))
    # D
    got_d = collections.Counter((r["granularity"], r["alpha0"]) for r in raw4
                                if not r["run"].startswith("cru1-"))
    out.append(("PRED-D non-cru1 raw ms=1e-4 rows == {chunk771 7, nodewise 7, chunk2293 3, nodewise1d 3} all alpha0=1e-3",
                dict(got_d) == PRED_D, "got %s" % dict(sorted(got_d.items()))))
    return out


# ----------------------------------------------------------------- score ----
def score(csv_path):
    rows = load(csv_path)
    n_rows = len(rows)
    print("cms1_ms1e4_stratum_census.py -- CIFAR-100 ms=1e-4 stratum census")
    print("csv: %s   rows: %d" % (os.path.relpath(csv_path), n_rows))
    print("(RULE 21 label NOT carried; commit-before-first-execution only; see header)")

    c100 = [r for r in rows if is_c100(r)]
    c100s = [r for r in c100 if in_cu1_stratum(r)]

    # -- ms levels on CIFAR-100 ------------------------------------------------
    print("\n-- CIFAR-100 meta_stepsize levels --------------------------------------")
    raw_ms = collections.Counter(r["meta_stepsize"] for r in c100)
    str_ms = collections.Counter(r["meta_stepsize"] for r in c100s)
    print("   %-10s %8s %12s" % ("ms", "RAW n", "cU1-STRATUM n"))
    for ms in sorted(set(raw_ms) | set(str_ms), key=lambda s: fnum(s) if fnum(s) is not None else -1):
        print("   %-10s %8d %12d" % (ms or "''", raw_ms[ms], str_ms[ms]))
    print("   CIFAR-100 RAW rows %d; cU1-STRATUM rows %d; distinct ms RAW %d, STRATUM %d"
          % (len(c100), len(c100s), len(raw_ms), len(str_ms)))

    # -- primary arms x ms x alpha0 (cU1-STRATUM) --------------------------------
    print("\n-- PRIMARY ARMS x ms x alpha0, cU1-STRATUM: n --------------------------")
    grid = collections.Counter((r["granularity"], r["alpha0"], r["meta_stepsize"]) for r in c100s
                               if r["granularity"] in PRIMARY_GRANS)
    ms_all = sorted(set(k[2] for k in grid), key=lambda s: fnum(s))
    a0_all = sorted(set(k[1] for k in grid), key=lambda s: fnum(s))
    print("   %-16s %-8s " % ("arm", "alpha0") + " ".join("%8s" % m for m in ms_all))
    for g in PRIMARY_GRANS:
        for a in a0_all:
            line = [grid[(g, a, m)] for m in ms_all]
            if sum(line) == 0:
                continue
            print("   %-16s %-8s " % (g, a) + " ".join("%8d" % x for x in line))

    t = tokens(rows)

    # -- the ms=1e-4 stratum, RAW then cU1-STRATUM --------------------------------
    print_cells("CIFAR-100 ms=1e-4 STRATUM, RAW (nothing excluded), by arm x alpha0",
                per_cell(t["raw4"]), len(t["raw4"]))
    print_cells("CIFAR-100 ms=1e-4 STRATUM, cU1-STRATUM filters, by arm x alpha0",
                per_cell(t["str4"]), len(t["str4"]))

    # -- the by-arm summary the prose needs -------------------------------------
    print("\n-- BY-ARM SUMMARY at ms=1e-4 (cU1-STRATUM n / RAW n) -------------------")
    raw_c = collections.Counter(r["granularity"] for r in t["raw4"])
    str_c = collections.Counter(r["granularity"] for r in t["str4"])
    for g in sorted(set(raw_c) | set(str_c), key=lambda g: (arm_class(g) != "PRIMARY", g)):
        a0s = sorted(set(r["alpha0"] for r in t["raw4"] if r["granularity"] == g), key=fnum)
        print("   %-14s [%-12s] stratum %3d / raw %3d   alpha0 levels %s"
              % (g, arm_class(g), str_c[g], raw_c[g], a0s))
    print("   alpha0 levels present at ms=1e-4 (RAW): %s"
          % sorted(set(r["alpha0"] for r in t["raw4"]), key=fnum))

    # -- ms contrasts at fixed alpha0 for primary arms ----------------------------
    print("\n-- ms CONTRASTS AT FIXED alpha0 FOR PRIMARY ARMS (cU1-STRATUM) --------")
    if t["contrasts"]:
        for g, a, v in t["contrasts"]:
            print("   %-16s alpha0=%-6s ms levels %s" % (g, a, v))
    else:
        print("   none -- 160's alias statement would still hold")

    # -- best rows ---------------------------------------------------------------
    print("\n-- BEST plateau5 (RAW; rows with a numeric plateau5) -------------------")
    def best(rs, label):
        cand = [(fnum(r["plateau5"]), r) for r in rs if fnum(r["plateau5"]) is not None]
        if not cand:
            print("   %s: (none)" % label)
            return
        v, r = max(cand, key=lambda x: x[0])
        print("   %-38s %9.4f  %s  (%s, alpha0 %s, ms %s, final_train %s)"
              % (label, v, r["run"], r["granularity"], r["alpha0"], r["meta_stepsize"], r["final_train"]))
    best(t["raw4"], "CIFAR-100 ms=1e-4 stratum")
    best([r for r in t["raw4"] if r["granularity"] in PRIMARY_GRANS], "  ... among PRIMARY arms")
    best(c100, "CIFAR-100 anywhere")
    best(c100s, "CIFAR-100 cU1-STRATUM anywhere")

    # -- predictions -------------------------------------------------------------
    print("\n-- PREDICTIONS frozen at design time (corpus-conditional, AS OF %d rows) --" % n_rows)
    preds = predictions(rows)
    for label, okv, got in preds:
        print("   [%s] %s -- %s" % ("HOLD" if okv else "FAIL", label, got))
    pred_tok = "HOLD" if all(p[1] for p in preds) else "FAIL"

    print("\n-- SCOPE ---------------------------------------------------------------")
    print("   A CENSUS, not a comparison.  It issues no accuracy claim, no gap, no")
    print("   SE.  cU1-STRATUM = cU1's frozen filters verbatim (header P1).  Says")
    print("   nothing about any dataset but CIFAR-100.  The token ALIAS is defined")
    print("   in this file's header and is NOT cY1's G2 token, which is a different")
    print("   rule over the cru1 batch alone.")
    print("")
    print("   FINAL: HEADLINE-ARMS-AT-MS1E4:%s | MS1E4-ALPHA0-LEVELS=%d | ALIAS:%s | BLK6-AT-MS1E4:%s | CUTPOS-AT-MS1E4:%s | PREDICTIONS:%s"
          % (t["headline"], t["a0_levels"], t["alias"], t["blk"], t["cut"], pred_tok))
    return 0


# -------------------------------------------------------------- selftest ----
def _row(**kw):
    base = {"run": "x-a-s0", "dataset": "CIFAR100", "network": "ResNet18_c100", "base": "SGDm",
            "meta": "Lion", "gamma": "1", "augment": "1", "beta_clip": "-15:-2.3026",
            "batch_size": "100", "hier": "", "collapsed": "0", "complete": "1",
            "superseded": "0", "epochs_done": "100", "meta_stepsize": "1e-3",
            "granularity": "scalar", "alpha0": "1e-3", "seed": "0", "plateau5": "50.0",
            "final_train": "60.0"}
    base.update(kw)
    return base


def selftest(csv_path):
    fails = []
    total = [0]

    def ok(label, cond):
        total[0] += 1
        print("   [%s] %s" % ("PASS" if cond else "FAIL", label))
        if not cond:
            fails.append(label)

    print("-- the token rules, on synthetic rows (corpus-independent) --------------")
    empty = tokens([])
    ok("empty corpus: ABSENT / 0 levels / HOLDS / ABSENT / ABSENT",
       (empty["headline"], empty["a0_levels"], empty["alias"], empty["blk"], empty["cut"])
       == ("ABSENT", 0, "HOLDS", "ABSENT", "ABSENT"))
    pre = [_row(run="gm2-ch-s%d" % i, granularity="chunk771", meta_stepsize="1e-4") for i in range(7)]
    tp = tokens(pre)
    ok("pre-cru1-shaped corpus (chunk771 only at 1e-4, alpha0 1e-3): ABSENT, 1 level, HOLDS",
       (tp["headline"], tp["a0_levels"], tp["alias"]) == ("ABSENT", 1, "HOLDS"))
    post = pre + [_row(run="cru1-sc-s15", meta_stepsize="1e-4", alpha0="1e-3"),
                  _row(run="cru1-lay-s15", granularity="layerwise", meta_stepsize="1e-4", alpha0="1e-6"),
                  _row(run="cru1-sc-s16", meta_stepsize="1e-3", alpha0="1e-3")]
    tq = tokens(post)
    ok("post-cru1-shaped corpus: PRESENT, 2 levels, BROKEN (scalar at 1e-4 and 1e-3 at alpha0=1e-3)",
       (tq["headline"], tq["a0_levels"], tq["alias"]) == ("PRESENT", 2, "BROKEN"))
    ok("ALIAS needs the contrast at ONE fixed alpha0: scalar 1e-4@1e-6 vs 1e-3@1e-3 is HOLDS",
       tokens(pre + [_row(run="a-s0", meta_stepsize="1e-4", alpha0="1e-6"),
                     _row(run="b-s0", meta_stepsize="1e-3", alpha0="1e-3")])["alias"] == "HOLDS")
    ok("a scalar-only 1e-4 row is ABSENT (needs BOTH headline arms)",
       tokens(pre + [_row(run="c-s0", meta_stepsize="1e-4")])["headline"] == "ABSENT")
    ok("a collapsed=1 scalar row at 1e-4 is in RAW, not in cU1-STRATUM",
       tokens(pre + [_row(run="c-s0", meta_stepsize="1e-4", collapsed="1"),
                     _row(run="d-s0", meta_stepsize="1e-4", granularity="layerwise", collapsed="1")])
       ["headline"] == "ABSENT")
    ok("hier=additive is EXCLUDED from cU1-STRATUM", not in_cu1_stratum(_row(hier="additive")))
    ok("superseded=1 is EXCLUDED from cU1-STRATUM", not in_cu1_stratum(_row(superseded="1")))
    ok("epochs_done=20 is EXCLUDED from cU1-STRATUM", not in_cu1_stratum(_row(epochs_done="20")))
    ok("a CIFAR10 row is never in the census", tokens([_row(dataset="CIFAR10", meta_stepsize="1e-4",
                                                             granularity="layerwise"),
                                                        _row(dataset="CIFAR10", meta_stepsize="1e-4")])["headline"] == "ABSENT")
    ok("resnet18_blocks at 1e-4 -> BLK6 PRESENT",
       tokens([_row(granularity="resnet18_blocks", meta_stepsize="1e-4")])["blk"] == "PRESENT")

    print("\n-- the cut-position / name-list classifier ------------------------------")
    ok("[49,13] is CUT-POSITION", arm_class("[49,13]") == "CUT-POSITION")
    ok("[2,60] is CUT-POSITION", arm_class("[2,60]") == "CUT-POSITION")
    ok("[8,8,8,8,8,8,7,7] is OTHER (not a two-group cut)", arm_class("[8,8,8,8,8,8,7,7]") == "OTHER")
    ok("sets:1-49/50-62 is NAME-LIST", arm_class("sets:1-49/50-62") == "NAME-LIST")
    ok("tn:sets:1-49/50-62 is NAME-LIST", arm_class("tn:sets:1-49/50-62") == "NAME-LIST")
    ok("tn:scalar is OTHER (not the primary `scalar`)", arm_class("tn:scalar") == "OTHER")
    ok("chunk771 is OTHER", arm_class("chunk771") == "OTHER")
    ok("scalar / resnet18_blocks / layerwise are PRIMARY",
       all(arm_class(g) == "PRIMARY" for g in PRIMARY_GRANS))
    ok("a cut-position row at 1e-4 -> CUTPOS PRESENT",
       tokens([_row(granularity="[49,13]", meta_stepsize="1e-4", alpha0="1e-6")])["cut"] == "PRESENT")
    ok("a name-list row at 1e-4 -> CUTPOS ABSENT",
       tokens([_row(granularity="sets:1-49/50-62", meta_stepsize="1e-4")])["cut"] == "ABSENT")

    print("\n-- structure of the live CSV (ingest-proof) -----------------------------")
    rows = load(csv_path)
    need = ("run", "dataset", "network", "base", "meta", "gamma", "augment", "beta_clip",
            "batch_size", "hier", "collapsed", "complete", "superseded", "epochs_done",
            "meta_stepsize", "granularity", "alpha0", "seed", "plateau5", "final_train")
    ok("every column this file reads exists", all(c in rows[0] for c in need) if rows else False)
    ok("the CSV `plateau` column is NOT read by this file",
       "r[\"plateau\"]" not in open(__file__).read().split("def selftest")[0])

    print("\n-- the design-time PREDICTIONS (corpus-conditional, AS OF %d rows) -------" % len(rows))
    print("   a FAIL here after a later ingest is a CENSUS CHANGE, not a defect (header P3)")
    for label, okv, got in predictions(rows):
        ok("%s -- %s" % (label, got), okv)

    print("\n%d FAIL / %d checks" % (len(fails), total[0]))
    return 1 if fails else 0


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    return selftest(a.csv) if a.selftest else score(a.csv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
