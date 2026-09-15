#!/usr/bin/env python3
# =============================================================================
# dup_group_guard.py -- refuse a run table whose `dup_group` repair has been
# silently reverted by a bare `aggregate.py` rebuild.
#
# WHY THIS EXISTS (docs/CORRECTIONS.md 133).
# `analysis/aggregate.py` regenerates results/all_runs.csv from the `.out` tree
# and knows nothing about `dup_group`: it stamps the column only where a *run
# name* collides, which catches the three `a0` reruns and misses all eighteen
# pairs of DIFFERENTLY-NAMED same-experiment runs that
# `analysis/args_repair.py --apply` writes.  Run alone, therefore, aggregate.py
# wipes 36 stamps and silently reverts the A3 duplicate-pair repair --- and the
# only visible symptom is that `ml2` goes back to being reported as 6 v 6:
# se 0.195 -> 0.142, t 2.34 -> 3.20.  Nothing errors.  Nothing looks wrong.
#
# THE STANDING RULE THIS ENFORCES.
#   The ingest is `aggregate.py` THEN `args_repair.py --apply`.  Never one alone.
#
# USAGE
#   python3 analysis/dup_group_guard.py              # guard results/all_runs.csv
#   python3 analysis/dup_group_guard.py --csv PATH   # guard another table
#   python3 analysis/dup_group_guard.py --selftest   # check the guard itself
#
# EXIT STATUS.  0 iff every invariant holds.  1 otherwise, with the exact
# remediation command printed.  Wire it after every ingest and before any
# commit that touches results/all_runs.csv.
#
# The expected pairs are NOT restated here.  They are imported from
# args_repair.GROUPS, which is their single source of truth, so this guard
# cannot drift away from the repair it guards.
# =============================================================================
from __future__ import print_function

import os
import csv
import sys
import argparse
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import args_repair as AR                                    # noqa: E402

DEFAULT_CSV = os.path.normpath(os.path.join(HERE, os.pardir,
                                            "results", "all_runs.csv"))

# Groups aggregate.py CAN produce on its own, because their two members share a
# run name: the three `a0` reruns, where one member carries superseded = 1.
SELF_GROUPS = 3
SELF_GROUP_ROWS = 6
SUPERSEDED_ROWS = 3

FIX = ("python3 analysis/aggregate.py <runs_dir> ... > results/all_runs.csv && "
       "python3 analysis/args_repair.py --apply")


def check(path):
    """Return a list of failure strings; empty means the table is intact."""
    bad = []
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["%s: no rows" % path]
    if "dup_group" not in rows[0]:
        return ["%s: no dup_group column at all -- aggregate.py output was "
                "committed raw" % path]

    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)

    # 1. every pair args_repair.py stamps is stamped, and stamped correctly.
    missing = []
    for group, members in AR.GROUPS:
        for run in members:
            got = by_run.get(run)
            if not got:
                bad.append("run %r of dup_group %r is not in the table" %
                           (run, group))
                continue
            for r in got:
                have = (r.get("dup_group") or "").strip()
                if have == "":
                    missing.append(run)
                elif have != group:
                    bad.append("run %r carries dup_group %r, expected %r" %
                               (run, have, group))
    if missing:
        bad.append("THE dup_group REPAIR HAS BEEN REVERTED: %d of %d stamped "
                   "rows are blank (e.g. %s). This is what a bare aggregate.py "
                   "rebuild does. ml2 will read 3 v 3 -> 6 v 6, se 0.195 -> "
                   "0.142, t 2.34 -> 3.20." %
                   (len(missing), sum(len(m) for _, m in AR.GROUPS),
                    ", ".join(sorted(missing)[:3])))

    # 2. the totals, so a partial or a spurious stamping is caught too.
    seen = collections.Counter((r.get("dup_group") or "").strip()
                               for r in rows)
    del seen[""]
    exp_groups = len(AR.GROUPS) + SELF_GROUPS
    exp_rows = sum(len(m) for _, m in AR.GROUPS) + SELF_GROUP_ROWS
    if len(seen) != exp_groups:
        bad.append("distinct dup_groups = %d, expected %d (%d from "
                   "args_repair.GROUPS + %d run-name collisions)" %
                   (len(seen), exp_groups, len(AR.GROUPS), SELF_GROUPS))
    if sum(seen.values()) != exp_rows:
        bad.append("rows carrying a dup_group = %d, expected %d" %
                   (sum(seen.values()), exp_rows))
    for group, n in sorted(seen.items()):
        if n != 2:
            bad.append("dup_group %r has %d members, expected 2" % (group, n))

    # 3. supersession is the a0 reruns' business and nobody else's.
    sup = sum(1 for r in rows if (r.get("superseded") or "").strip() == "1")
    if sup != SUPERSEDED_ROWS:
        bad.append("superseded = 1 on %d rows, expected %d" %
                   (sup, SUPERSEDED_ROWS))
    return bad


def selftest():
    """The guard must FAIL on a reverted table and PASS on a repaired one."""
    import tempfile
    with open(DEFAULT_CSV) as fh:
        rows = list(csv.DictReader(fh))
        fields = rows[0].keys()
    n = 0
    if check(DEFAULT_CSV):
        print("selftest: FAIL -- the live table does not pass the guard")
        return 1
    n += 1
    stamped = {run for _, members in AR.GROUPS for run in members}
    for label, mutate in (
            ("bare aggregate.py rebuild (all 36 stamps wiped)",
             lambda r: dict(r, dup_group="")
             if r["run"] in stamped else r),
            ("one stamp lost",
             lambda r: dict(r, dup_group="")
             if r["run"] == sorted(stamped)[0] else r),
            ("a stamp rewritten to the wrong group",
             lambda r: dict(r, dup_group="not-a-group")
             if r["run"] == sorted(stamped)[0] else r),
            ("supersession invented",
             lambda r: dict(r, superseded="1")
             if r["run"] == sorted(stamped)[0] else r)):
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as out:
            w = csv.DictWriter(out, fieldnames=list(fields))
            w.writeheader()
            for r in rows:
                w.writerow(mutate(r))
        failed = check(tmp)
        os.unlink(tmp)
        if not failed:
            print("selftest: FAIL -- guard did not catch: %s" % label)
            return 1
        n += 1
    print("selftest: %d/%d PASS" % (n, n))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    bad = check(a.csv)
    if bad:
        print("dup_group_guard: %d FAILURE(S) on %s" % (len(bad), a.csv))
        for b in bad:
            print("  !! %s" % b)
        print("\n  THE INGEST IS aggregate.py THEN args_repair.py --apply:")
        print("    %s" % FIX)
        print("VERDICT: FAIL")
        return 1
    print("dup_group_guard: %s -- %d groups, %d rows stamped, %d superseded. "
          "VERDICT: PASS" %
          (os.path.basename(a.csv), len(AR.GROUPS) + SELF_GROUPS,
           sum(len(m) for _, m in AR.GROUPS) + SELF_GROUP_ROWS,
           SUPERSEDED_ROWS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
