#!/usr/bin/env python3
"""c73 -- MASTER-TABLE.md structural + arithmetic validator.

WHY THIS EXISTS.  STANDING RULE (15) makes `docs/MASTER-TABLE.md` the mandatory grep
target before any cross-granularity claim, which makes a WRONG HEADER in that file
more dangerous than a wrong header anywhere else.  Cycle 73 wrote its verdict tally
from estimate rather than by counting, and got four of six numbers wrong (claimed
CONFIRMED 29 / OPEN 10 / WITHDRAWN 2 / UNINTERPRETABLE 3 / 72 rows against an actual
30+2 / 9 / 3 / 2 / 74).  That is STANDING RULE 1's failure -- quoting a statistic
instead of re-deriving it -- committed inside the file the rule points people at.

It also catches the markdown defect that produced it: an UNESCAPED `|` inside a cell
silently splits that cell in two, so the verdict column shifts and every downstream
reader (human or grep) mis-reads the row.  Three such rows existed this cycle.

Checks, all mechanical, none of them a judgement about the science:
  1. every table row has exactly the 7 columns its section header declares
  2. no row contains an unescaped `|` inside a cell
  3. the run count and GPU-hours in the header match results/all_runs.csv
  4. the verdict tally in the header matches the verdicts actually in the table

Run:  python3 analysis/c73_mastertable_check.py            # report + exit code
      python3 analysis/c73_mastertable_check.py --selftest
"""

import collections
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TABLE = os.path.join(REPO, "docs", "MASTER-TABLE.md")
CSV = os.path.join(REPO, "results", "all_runs.csv")

NCOL = 7                     # tested | varied | scale | result | verdict | so what | ref
SPLIT = re.compile(r"(?<!\\)\|")


def rows(text):
    """Yield (lineno, cells) for every data row under a table header."""
    hdr = False
    for i, line in enumerate(text.split("\n"), 1):
        if line.startswith("## "):
            hdr = False
        elif line.startswith("|---"):
            hdr = True
        elif line.startswith("|") and hdr:
            parts = SPLIT.split(line)
            # a well-formed row is '' + NCOL cells + ''
            yield i, parts


def csv_stats(path):
    rd = list(csv.DictReader(open(path)))
    mins = 0.0
    for r in rd:
        if r.get("wallclock_min"):
            try:
                mins += float(r["wallclock_min"])
            except ValueError:
                pass
    return len(rd), mins / 60.0


def check(verbose=True):
    text = open(TABLE).read()
    bad = []

    # -- 1 + 2: column arity (an unescaped pipe shows up here as a surplus cell)
    tally = collections.Counter()
    n_rows = 0
    for ln, parts in rows(text):
        if len(parts) != NCOL + 2:
            bad.append(f"line {ln}: {len(parts) - 2} cells, expected {NCOL} "
                       f"-- likely an UNESCAPED '|' inside a cell")
            continue
        n_rows += 1
        tally[parts[5].strip().replace("*", "")] += 1

    # -- 3: the corpus numbers in the header
    runs, hours = csv_stats(CSV)
    m = re.search(r"Compiled\s+([\d,]+)\s+runs\s*/\s*([\d.]+)\s+GPU-hours", text)
    if not m:
        bad.append("header: no 'Compiled N runs / X GPU-hours' line found")
    else:
        h_runs = int(m.group(1).replace(",", ""))
        h_hours = float(m.group(2))
        if h_runs != runs:
            bad.append(f"header says {h_runs} runs; the CSV has {runs}")
        if abs(h_hours - hours) > 0.15:
            bad.append(f"header says {h_hours} GPU-hours; the CSV gives {hours:.1f}")

    # -- 4: the verdict tally
    m = re.search(r"Verdicts:\s*(.+?)\s*\((\d+)\s+rows total\)", text)
    if not m:
        bad.append("header: no 'Verdicts: ... (N rows total)' line found")
    else:
        claimed_total = int(m.group(2))
        if claimed_total != n_rows:
            bad.append(f"header claims {claimed_total} rows; the table has {n_rows}")
        # families: everything starting CONFIRMED counts as CONFIRMED, etc.
        fam = collections.Counter()
        for k, v in tally.items():
            root = k.split(",")[0].split("(")[0].strip()
            fam[root] += v
        for name, want in re.findall(r"([A-Z][A-Z ]+?)\s+(\d+)", m.group(1)):
            name = name.strip()
            if fam.get(name, 0) != int(want):
                bad.append(f"header claims {name} {want}; the table has {fam.get(name, 0)}")

    if verbose:
        print("MASTER-TABLE.md: %d data rows, %d columns each" % (n_rows, NCOL))
        print("CSV: %d runs / %.1f GPU-hours" % (runs, hours))
        print("verdicts actually in the table:")
        for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
            print("   %-26s %d" % (k, v))
        if bad:
            print("\nFAILURES:")
            for b in bad:
                print("  -", b)
        else:
            print("\nheader is consistent with both the table and the CSV  OK")
    return bad


def selftest():
    p = n = 0

    def chk(name, cond):
        nonlocal p, n
        n += 1
        if cond:
            p += 1
        else:
            print("  FAIL:", name)

    # the splitter must ignore an ESCAPED pipe and honour a real one
    chk("escaped pipe does not split", len(SPLIT.split(r"| a \| b | c |")) == 4)
    chk("real pipe splits", len(SPLIT.split("| a | b | c |")) == 5)
    # a row with an unescaped pipe must be REJECTED, not silently mis-columned
    good = "| " + " | ".join(str(i) for i in range(NCOL)) + " |"
    chk("well-formed row has NCOL+2 parts", len(SPLIT.split(good)) == NCOL + 2)
    broken = good.replace("| 4 |", "| 4|x |")
    chk("unescaped pipe is detected", len(SPLIT.split(broken)) != NCOL + 2)
    # the real file must parse
    chk("the live table has rows", sum(1 for _ in rows(open(TABLE).read())) > 50)
    chk("the CSV is readable", csv_stats(CSV)[0] > 1000)
    print("selftest: %d/%d passed" % (p, n))
    return p == n


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    sys.exit(1 if check() else 0)
