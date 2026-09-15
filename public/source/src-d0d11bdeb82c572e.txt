#!/usr/bin/env python3
# =============================================================================
# args_repair.py -- the CSV repair implied by docs/ARGS-AUDIT.md.
#
# WRITTEN, NOT APPLIED.  Default mode is a dry run that prints the exact diff.
# `--apply` is required to touch results/all_runs.csv, and it always writes a
# timestamped .bak first.
#
# WHAT THE AUDIT FOUND, AND WHAT THIS THEREFORE DOES *NOT* DO
# -----------------------------------------------------------
# The duplicate-flag bug (STANDING RULE 20) did NOT corrupt any CSV value.
# `analysis/aggregate.py`'s parse_args_line is `out[k] = v` in one forward
# pass -- last-wins, exactly like argparse -- so the aggregator recorded what
# actually ran.  All 24 `ml2` rows already read meta = Lion.
#
#   => There is NO value repair to make.  The void is in the run NAMES and in
#      the batches' declared designs, not in the columns.
#
# What IS missing is the duplicate bookkeeping.  After last-wins resolution 18
# pairs of differently-named runs are the SAME experiment -- same effective
# ARGS, same ENV, and the SAME --seed -- yet all 36 rows carry dup_group "".
# `ml2`'s 24 runs are 3 seeds run twice, not 6 seeds; reported as "6 v 6" they
# overstate independence (D: se 0.142 -> 0.195, t 3.20 -> 2.34).
#
# So this patch writes ONE column, `dup_group`, on 36 rows.
#   * No accuracy value changes.
#   * No row is deleted.
#   * No row is marked `superseded` -- both members of every pair are real
#     measurements; neither supersedes the other.  (Contrast the 3 existing
#     `a0` rows, which are genuine reruns where the later one wins.)
#
# THE RULE THE COLUMN CARRIES.  Any n, se or t computed over rows that share a
# dup_group MUST average within the group first and count n as the number of
# distinct groups.  That rule, applied to `ml2`, reproduces the honest column
# of ARGS-AUDIT.md §5 automatically.
#
# NOT DONE HERE.  Ingesting `sm3`'s 12 rows (2,113 -> 2,125).  Run
# `aggregate.py` unedited; its last-wins parser writes meta = Lion by itself.
#
# USAGE
#   python3 analysis/args_repair.py            # dry run: print the diff
#   python3 analysis/args_repair.py --report   # + re-verify the audit's claims
#   python3 analysis/args_repair.py --apply    # write (makes a .bak first)
# =============================================================================

from __future__ import print_function

import os
import csv
import sys
import shutil
import argparse
import collections
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, os.pardir, "results", "all_runs.csv")

# -----------------------------------------------------------------------------
# The 18 collapsed pairs.  Derived in docs/ARGS-AUDIT.md §5 from the runs' own
# ARGS + ENV lines on both clusters; listed literally here so this script needs
# no cluster access and the change set is auditable by eye.
#
#   (dup_group, [run names in results/all_runs.csv])
# -----------------------------------------------------------------------------
GROUPS = []

# ml2 -- 4 granularities x 3 seeds.  The two halves resolve to an identical
# command line INCLUDING --seed; they differ only in PROBE_DIR (an output path).
for _gran in ("node", "ch", "n1d", "c23"):
    for _s in (0, 1, 2):
        GROUPS.append((
            "ml2-%s-s%d" % (_gran, _s),
            ["ml2-adam-%s-s%d" % (_gran, _s), "ml2-rms-%s-s%d" % (_gran, _s)],
        ))

# h2 -- two naming schemes for one config (HIER=shrink LAM=0.1).  Inert:
# `M0 shrink` is withdrawn in paper/DRAFT-v2.md.
for _s in (0, 1, 2):
    GROUPS.append((
        "h2-blocks-shrink0.1-s%d" % _s,
        ["h2-blk6_s%d" % _s, "h2-b-L0p1_s%d" % _s],
    ))
GROUPS.append(("h2-layer-shrink0.1-s2", ["h2-lam01_s2", "h2-l-L0p1_s2"]))

# c100 smoke tests -- `layer` and `probe` are the same 3-epoch run.  Inert:
# plateau5 is empty and window_ok = 0, so STANDING RULE 21's filter drops them.
GROUPS.append(("c100smoke-layer", ["c100smoke_layer", "c100smoke_probe"]))
GROUPS.append(("c100pin-layer", ["c100pin_layer", "c100pin_probe"]))

TARGET_COL = "dup_group"


def load(path):
    with open(path, "r", newline="") as fh:
        rdr = csv.DictReader(fh)
        return rdr.fieldnames, list(rdr)


def plan(rows):
    """Return (changes, problems). changes = [(run, col, old, new)]."""
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    changes, problems = [], []
    for group, members in GROUPS:
        for name in members:
            hits = by_run.get(name, [])
            if len(hits) != 1:
                problems.append("%s: expected exactly 1 CSV row, found %d"
                                % (name, len(hits)))
                continue
            old = hits[0].get(TARGET_COL, "")
            if old == group:
                continue
            if old not in ("", group):
                problems.append("%s: dup_group already set to %r, refusing to "
                                "overwrite" % (name, old))
                continue
            changes.append((name, TARGET_COL, old, group))
    return changes, problems


def report(rows):
    """Re-verify the audit's arithmetic from the CSV, so the patch is never
    trusted further than the data it rests on."""
    import math
    try:
        import statistics as st
    except ImportError:
        print("  (statistics unavailable; skipping)")
        return

    v = {}
    for r in rows:
        if not r["run"].startswith("ml2-"):
            continue
        _, arm, gran, seed = r["run"].split("-")
        v[(arm, gran, int(seed[1:]))] = float(r["plateau5"])
    if len(v) != 24:
        print("  ml2: expected 24 rows, found %d -- skipping" % len(v))
        return

    def unpaired(hi, lo, arms):
        A = [v[(a, hi, s)] for a in arms for s in range(3)]
        B = [v[(a, lo, s)] for a in arms for s in range(3)]
        m = st.mean(A) - st.mean(B)
        se = math.sqrt(st.stdev(A) ** 2 / len(A) + st.stdev(B) ** 2 / len(B))
        return m, se

    def honest(hi, lo):
        A = [(v[("adam", hi, s)] + v[("rms", hi, s)]) / 2 for s in range(3)]
        B = [(v[("adam", lo, s)] + v[("rms", lo, s)]) / 2 for s in range(3)]
        m = st.mean(A) - st.mean(B)
        se = math.sqrt(st.stdev(A) ** 2 / 3 + st.stdev(B) ** 2 / 3)
        return m, se

    print("\n--- ARGS-AUDIT.md sec.5, re-derived from plateau5 ---")
    for label, hi, lo in (("D  (chunk777 - nodewise) ", "ch", "node"),
                          ("G  (chunk2325 - nodewise1d)", "c23", "n1d")):
        m, se = unpaired(hi, lo, ["adam", "rms"])
        hm, hse = honest(hi, lo)
        print("  %s" % label)
        print("     as reported  6 v 6 : %+.3f +- %.3f   t %.2f" % (m, se, m / se))
        print("     honest  3 seeds x2 : %+.3f +- %.3f   t %.2f" % (hm, hse, hm / hse))

    nd = [v[("adam", g, s)] - v[("rms", g, s)]
          for g in ("node", "ch", "n1d", "c23") for s in range(3)]
    bs = [st.stdev([v[(a, g, s)] for s in range(3)])
          for a in ("adam", "rms") for g in ("node", "ch", "n1d", "c23")]
    print("\n  nondeterminism (same config, same seed): "
          "mean |d| %.3f  sd %.3f  max %.3f" %
          (st.mean([abs(x) for x in nd]), st.pstdev(nd),
           max(abs(x) for x in nd)))
    print("  between-seed sd: mean %.3f   ratio to nondeterminism %.2fx"
          % (st.mean(bs), st.mean(bs) / st.pstdev(nd)))
    print("  -> seed is null here, which is why the correction is small.")


def main(argv):
    ap = argparse.ArgumentParser(
        prog="args_repair.py",
        description="Dup-group repair implied by docs/ARGS-AUDIT.md. "
                    "Dry run unless --apply.")
    ap.add_argument("--apply", action="store_true",
                    help="write results/all_runs.csv (a .bak is made first)")
    ap.add_argument("--report", action="store_true",
                    help="also re-derive the audit's ml2 arithmetic")
    ap.add_argument("--csv", default=CSV_PATH)
    a = ap.parse_args(argv)

    path = os.path.normpath(a.csv)
    fields, rows = load(path)
    if TARGET_COL not in fields:
        sys.stderr.write("args_repair: %s has no %r column\n" % (path, TARGET_COL))
        return 2

    changes, problems = plan(rows)

    print("=== args_repair: %s ===" % path)
    print("rows: %d   groups declared: %d   rows in a group: %d"
          % (len(rows), len(GROUPS), sum(len(m) for _, m in GROUPS)))
    print("\n--- CHANGES (%d rows, column %r) ---" % (len(changes), TARGET_COL))
    for run, col, old, new in changes:
        print("  %-24s %s: %-8r -> %r" % (run, col, old, new))
    if not changes:
        print("  (none -- already applied, or nothing to do)")

    print("\n--- NOT CHANGED ---")
    print("  accuracy / config values : 0 rows (the CSV already matches every")
    print("                             run's ARGS line -- see ARGS-AUDIT sec.4)")
    print("  superseded               : 0 rows (both members of each pair are")
    print("                             real measurements)")
    print("  untouched rows           : %d" % (len(rows) - len(changes)))

    if problems:
        print("\n!!! PROBLEMS (%d) -- refusing to apply:" % len(problems))
        for p in problems:
            print("    %s" % p)

    if a.report:
        report(rows)

    if not a.apply:
        print("\nDRY RUN. Nothing written. Re-run with --apply to write.")
        return 0
    if problems:
        print("\nNOT APPLIED: resolve the problems above first.")
        return 1

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = "%s.pre-argsrepair-%s.bak" % (path, stamp)
    shutil.copy2(path, bak)
    tgt = dict((run, new) for run, _, _, new in changes)
    for r in rows:
        if r["run"] in tgt:
            r[TARGET_COL] = tgt[r["run"]]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    print("\nAPPLIED: %d rows updated. Backup: %s" % (len(changes), bak))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
