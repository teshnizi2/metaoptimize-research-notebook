#!/usr/bin/env python3
r"""c69_orphan_census.py -- THE CAMPAIGN-WIDE ORPHAN CENSUS, offline, zero compute.

WHY THIS EXISTS
---------------
The operator's standing per-tick instruction has never been discharged at campaign scale:

    "Also report ORPHANS: families present in results/all_runs.csv that no document in docs/
     cites.  As of 2026-08-22 there were roughly 290 such runs.  These are already-spent
     compute, so never re-run them; either write them up in one line each or mark them
     explicitly abandoned in docs/CORRECTIONS.md so the next tick stops rediscovering them."

Cycle 68 (CORRECTIONS 97.10) reported "ORPHANS: ZERO" -- but *scoped to the 380-run k=20
stratum only*, not campaign-wide.  The ~290-run figure has never been reproduced, localised,
or discharged.  This instrument does the campaign-wide census.

WHY IT IS WORTH A TICK.  Cycles 67 and 68 each found a load-bearing claim refuted by a
document this campaign had already written (96.6, 97.8).  An orphan run is the same failure
mode one level down: a result this campaign already PAID FOR and never read.  STANDING RULE
(15) was installed because unread evidence changed a headline.  This asks whether unread
*runs* can do the same.

THE CITATION RULE, FIXED BEFORE ANY FAMILY WAS CLASSIFIED
---------------------------------------------------------
FAMILY.  `family(run)` = the leading run-name token up to the first `-` or `_`.  This is how
`docs/` names families throughout (`bo7-*`, `c100f-*`, `mx/probe_sig_*`, `p5/probe_w_a3_s0`).

CORPUS.  Every `.md` under `docs/`, every `.md` at the repo root, every `.md` under `paper/`.
The corpus is listed in the report so the denominator is auditable.

TWO RULES, BOTH REPORTED.  Short family tokens (`ms`, `ac`, `dc`, `fx`, `bl`, `p5` ...) will
match unrelated prose, so a single rule cannot be trusted:

  LOOSE   `(?<![A-Za-z0-9])FAM(?![A-Za-z0-9])`     -- any whole-token occurrence.
          GENEROUS: over-counts citations, therefore UNDER-counts orphans.
  STRICT  `(?<![A-Za-z0-9])FAM(?=[-_/*.]|\b\s*(?:family|arm|runs))`
          -- the token in a family-reference context (`mx/`, `bo7-`, `p5/probe`, "`ms` family").
          CONSERVATIVE: under-counts citations, therefore OVER-counts orphans.

**ORPHAN is defined as NOT CITED UNDER THE LOOSE RULE.**  That is the rule that is hardest on
this instrument's own headline: it makes the orphan set as SMALL as the evidence allows, so
any orphan it reports survived the most generous possible search.  Families that are LOOSE-cited
but not STRICT-cited are reported separately as AMBIGUOUS and are NOT counted as orphans --
they are the set a human must eyeball, and the report prints their matching context lines.

WHAT IS *NOT* CLAIMED
---------------------
A citation is a TOKEN MATCH, not a reading.  A family can be "cited" by one passing mention in
a superseded section and still be scientifically unread.  This instrument measures the WEAKER
property (is the family named anywhere at all) and therefore its orphan set is a LOWER BOUND on
unread compute, never an estimate of it.  No document may write "only N runs were unread."

GATES
-----
G1  PARTITION.  Every one of the 1707 rows lands in exactly one family; family run-counts sum
    to the row count.
G2  MONOTONICITY.  STRICT-cited is a SUBSET of LOOSE-cited, for every family.
G3  BOUNDARY.  The token regexes do not match inside longer alphanumeric tokens
    (`p5` must not match `p50`, `ap5`, `p5x`).
G4  NEGATIVE CONTROL.  A synthetic family name absent from the corpus is uncited under both
    rules.
G5  POSITIVE CONTROL.  A family this project demonstrably discusses by name (`ml5`, quoted in
    FINDINGS 62.7's dose-response table) is STRICT-cited.
G6  CORPUS.  The corpus is non-empty and contains `docs/FINDINGS.md` and `docs/CORRECTIONS.md`.

USAGE
    python3 analysis/c69_orphan_census.py --selftest
    python3 analysis/c69_orphan_census.py --report   [--root .]
    python3 analysis/c69_orphan_census.py --detail FAM
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FAM_RE = re.compile(r'^([A-Za-z0-9]+?)(?=[-_])')


def family(run):
    """Leading run-name token up to the first '-' or '_'; the whole name if neither occurs."""
    m = FAM_RE.match(run)
    return m.group(1) if m else run


def loose_re(fam):
    return re.compile(r'(?<![A-Za-z0-9])' + re.escape(fam) + r'(?![A-Za-z0-9])')


def strict_re(fam):
    # family-reference context: fam followed by - _ / * . or by whitespace + family/arm/runs
    return re.compile(r'(?<![A-Za-z0-9])' + re.escape(fam) +
                      r'(?:[-_/*.]|\s+(?:family|families|arm|arms|runs))')


def corpus_files(root):
    files = []
    files += sorted(glob.glob(os.path.join(root, 'docs', '*.md')))
    files += sorted(glob.glob(os.path.join(root, '*.md')))
    files += sorted(glob.glob(os.path.join(root, 'paper', '*.md')))
    return files


def load_corpus(root):
    out = {}
    for f in corpus_files(root):
        try:
            out[os.path.relpath(f, root)] = open(f, errors='replace').read()
        except OSError:
            pass
    return out


def load_rows(root):
    p = os.path.join(root, 'results', 'all_runs.csv')
    with open(p) as fh:
        return list(csv.DictReader(fh))


def census(root):
    rows = load_rows(root)
    corpus = load_corpus(root)
    fams = defaultdict(list)
    for r in rows:
        fams[family(r['run'])].append(r)

    res = {}
    for fam in fams:
        lre, sre = loose_re(fam), strict_re(fam)
        lhits, shits = [], []
        for name, txt in corpus.items():
            if lre.search(txt):
                lhits.append(name)
            if sre.search(txt):
                shits.append(name)
        res[fam] = dict(n=len(fams[fam]), rows=fams[fam],
                        loose=lhits, strict=shits)
    return rows, corpus, res


# --------------------------------------------------------------------------------------
# selftests
# --------------------------------------------------------------------------------------
def selftest():
    t = 0

    def ck(cond, label):
        nonlocal t
        t += 1
        if not cond:
            print(f'  FAIL T{t}: {label}')
            sys.exit(1)
        print(f'  ok  T{t}: {label}')

    # family parsing
    ck(family('bo7-r18-s0') == 'bo7', "family('bo7-r18-s0')=='bo7'")
    ck(family('c100f-lay-s1') == 'c100f', "family('c100f-lay-s1')=='c100f'")
    ck(family('mx_sig_w') == 'mx', "family('mx_sig_w')=='mx' (underscore split)")
    ck(family('p7-c100-s0') == 'p7', "family('p7-c100-s0')=='p7'")
    ck(family('solo') == 'solo', 'family() of a token with no separator is itself')

    # G3 boundary
    lr = loose_re('p5')
    ck(lr.search('p5') is not None, "loose 'p5' matches 'p5'")
    ck(lr.search('p50') is None, "loose 'p5' does NOT match 'p50'")
    ck(lr.search('ap5') is None, "loose 'p5' does NOT match 'ap5'")
    ck(lr.search('p5x') is None, "loose 'p5' does NOT match 'p5x'")
    ck(lr.search('runs/p5/probe') is not None, "loose 'p5' matches 'runs/p5/probe'")
    sr = strict_re('p5')
    ck(sr.search('p5/probe_w_a3_s0') is not None, "strict 'p5' matches 'p5/probe...'")
    ck(sr.search('the p5 value') is None, "strict 'p5' does NOT match bare prose 'the p5 value'")
    ck(sr.search('the p5 family') is not None, "strict 'p5' matches 'p5 family'")
    ck(sr.search('bo7-r18') is None, "strict 'p5' does not match unrelated text")

    # G2 monotonicity on synthetic text
    txt = 'see mx/probe_sig_w and also mx alone'
    ck(loose_re('mx').search(txt) is not None and strict_re('mx').search(txt) is not None,
       'strict-hit implies loose-hit on synthetic text')
    txt2 = 'the mx value is 3'
    ck(loose_re('mx').search(txt2) is not None and strict_re('mx').search(txt2) is None,
       'loose can hit where strict does not (AMBIGUOUS class is reachable)')

    # live data gates
    rows, corpus, res = census(ROOT)
    ck(len(rows) > 0, f'CSV loaded ({len(rows)} rows)')
    # G1 partition
    ck(sum(v['n'] for v in res.values()) == len(rows),
       f'G1 PARTITION: family counts sum to {len(rows)} rows')
    # G6 corpus
    ck(len(corpus) > 0, f'G6 corpus non-empty ({len(corpus)} files)')
    ck('docs/FINDINGS.md' in corpus, 'G6 corpus contains docs/FINDINGS.md')
    ck('docs/CORRECTIONS.md' in corpus, 'G6 corpus contains docs/CORRECTIONS.md')
    # G2 on live data
    bad = [f for f, v in res.items() if v['strict'] and not v['loose']]
    ck(not bad, f'G2 MONOTONICITY: strict subset of loose on all {len(res)} families')
    # G4 negative control
    joined = '\n'.join(corpus.values())
    ck(loose_re('zzzznotafamily').search(joined) is None,
       'G4 NEGATIVE CONTROL: synthetic family uncited (loose)')
    # G5 positive control
    ck('ml5' in res and res['ml5']['strict'],
       'G5 POSITIVE CONTROL: ml5 is STRICT-cited')

    print(f'\n{t}/{t} selftests PASS')
    return 0


def report(root):
    rows, corpus, res = census(root)
    orph = {f: v for f, v in res.items() if not v['loose']}
    amb = {f: v for f, v in res.items() if v['loose'] and not v['strict']}
    cited = {f: v for f, v in res.items() if v['strict']}

    print(f'CORPUS: {len(corpus)} documents')
    for n in sorted(corpus):
        print(f'    {n}')
    print()
    print(f'RUNS {len(rows)}   FAMILIES {len(res)}')
    print(f'  STRICT-CITED   {len(cited):3d} families  {sum(v["n"] for v in cited.values()):5d} runs')
    print(f'  AMBIGUOUS      {len(amb):3d} families  {sum(v["n"] for v in amb.values()):5d} runs'
          '   (loose-cited only; needs eyeball)')
    print(f'  ORPHAN         {len(orph):3d} families  {sum(v["n"] for v in orph.values()):5d} runs'
          '   (uncited under the GENEROUS rule)')
    print()

    def block(title, d):
        print(f'=== {title} ===')
        if not d:
            print('  (none)')
            return
        for f, v in sorted(d.items(), key=lambda x: -x[1]['n']):
            print(f'  {f:12s} n={v["n"]:4d}  {describe(v["rows"])}')
        print()

    block('ORPHAN FAMILIES (uncited, generous rule)', orph)
    block('AMBIGUOUS FAMILIES (bare-token match only)', amb)
    return 0


def describe(rs):
    """Compact config + outcome summary for a family."""
    def uni(k):
        vals = sorted({(r.get(k) or '').strip() for r in rs} - {''})
        if not vals:
            return '?'
        return vals[0] if len(vals) == 1 else f'{{{",".join(vals[:4])}{"..." if len(vals) > 4 else ""}}}'

    pl = []
    for r in rs:
        try:
            pl.append(float(r['plateau5'] or r['plateau']))
        except (ValueError, KeyError, TypeError):
            pass
    ep = []
    for r in rs:
        try:
            ep.append(int(float(r['epochs_done'] or 0)))
        except (ValueError, TypeError):
            pass
    rng = f'{min(pl):.2f}-{max(pl):.2f}' if pl else 'n/a'
    return (f'{uni("network")}/{uni("dataset")} gran={uni("granularity")} base={uni("base")} '
            f'meta={uni("meta")} ms={uni("meta_stepsize")} a0={uni("alpha0")} aug={uni("augment")} '
            f'ep={min(ep) if ep else "?"}-{max(ep) if ep else "?"} plat5={rng}')


def detail(root, fam):
    _rows, corpus, res = census(root)
    if fam not in res:
        print(f'no such family: {fam}')
        return 1
    v = res[fam]
    print(f'FAMILY {fam}  n={v["n"]}')
    print(f'  loose-cited in : {v["loose"] or "(nowhere)"}')
    print(f'  strict-cited in: {v["strict"] or "(nowhere)"}')
    print()
    hdr = ['run', 'network', 'dataset', 'granularity', 'base', 'meta', 'meta_stepsize',
           'alpha0', 'augment', 'seed', 'epochs_done', 'epochs_requested',
           'plateau', 'plateau5', 'best_test', 'window_ok']
    print('  ' + ' | '.join(hdr))
    for r in sorted(v['rows'], key=lambda r: r['run']):
        print('  ' + ' | '.join(str(r.get(h, ''))[:18] for h in hdr))
    # show citation context
    for name in v['loose'][:3]:
        for i, line in enumerate(corpus[name].splitlines()):
            if loose_re(fam).search(line):
                print(f'  CTX {name}:{i+1}: {line.strip()[:160]}')
                break
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--detail')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.detail:
        return detail(a.root, a.detail)
    if a.report:
        return report(a.root)
    ap.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
