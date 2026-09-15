#!/usr/bin/env python3
r"""c70_composition_audit.py -- DOES CROSS-NETWORK POOLING REACH BEYOND 68.6's CELL?

THE REGISTERED QUESTION
-----------------------
CORRECTIONS 99.5, written at the close of cycle 69, states the open question verbatim:

    "The cross-network audit has ONLY been run on the one cell carrying the newest load-bearing
     claim.  Whether cross-network (or cross-family) pooling reaches into the other 100-epoch
     cells quoted in cycles 65-68 is OPEN, and is the obvious next offline job after
     MASTER-TABLE.  `analysis/c69_c100_armset.py` generalises to it directly."

This instrument is that generalisation.  It is REGISTERED AND COMMITTED BEFORE ANY CELL IS
SCORED, per the practice CORRECTIONS 76(1)/79 installed and cycle 69 followed.

THE DEFECT BEING HUNTED, STATED PRECISELY
-----------------------------------------
A published cell of the form "arm A = x +- s (n1), arm B = y +- t (n2), Delta = x-y" carries an
implicit claim: **the two arms differ ONLY in the axis the cell is comparing.**  That is STANDING
RULE (10), installed cycle 64: *"a series across any axis must hold the arm fixed."*

Cycle 69 (CORRECTIONS 99.1) found that claim FALSE for the campaign's newest headline: 68.6's
"fully matched" CIFAR-100 cell compared a single-network nodewise arm (ResNet18_c100, n=3) with a
THREE-network layerwise arm (R10+R18+R34, n=14).  +2.367 pp was withdrawn; +1.821 pp replaced it.

Note carefully what the defect IS and IS NOT.  Pooling several networks into a cell is not by
itself an error -- if every arm pools them in the SAME proportions, the contrast is still a
within-network contrast on average.  The error is **ASYMMETRIC composition**: arm A drawn from a
different mixture than arm B.  This instrument therefore measures composition DIVERGENCE between
arms, not the mere presence of pooling.

DIVERGENCE STATISTIC (fixed here, not chosen after looking)
-----------------------------------------------------------
For an axis c and two arms A, B, let p_A and p_B be the empirical distributions of c within each
arm.  Total variation distance

    TVD(A,B; c) = 0.5 * sum_v | p_A(v) - p_B(v) |          in [0, 1]

TVD = 0 means identical composition (the arms are matched on c).  TVD = 1 means disjoint
composition (the 68.6 nodewise-vs-layerwise case has TVD = 1 - 8/14 = 0.4286 on `network`).

    CONTAMINATION THRESHOLD, registered: TVD > 0.25 on any free axis, for the two arms forming
    the cell's HEADLINE contrast (its argmax and runner-up).

0.25 is chosen a priori as "a quarter of one arm's mass sits on values the other arm does not
match".  It is NOT tuned: G3 below is the registered check that it does not flag everything, and
the sensitivity of every headline verdict to the threshold is printed (--report prints the
per-cell TVD so any reader can re-threshold).

STATISTIC AND GATE ARE THE PUBLISHED ONES, NOT NEW
--------------------------------------------------
`plateau5` (mean of last 5 epochs, the documented window) and `plateau` (the CSV's k=20 column)
are read straight from `results/all_runs.csv`.  No re-windowing is done here.  The resolution gate
is 65.6/94.3's own rule, `margin > 2*sqrt(sem1^2 + sem2^2)`, exactly as c69 used it.
Each published cell is re-derived in THE COLUMN ITS OWN DOCUMENT USED (65.3 predates the
plateau5 correction and published `plateau`; 68.6 published `plateau5`), recorded per cell.

TWO PARTS
---------
PART A -- BASE RATE, mechanical and blind to which cells are published.  Every "relaxed cell" in
    the CSV (group on all config axes EXCEPT network, granularity and seed; keep cells with >=2
    granularity arms of n>=3) is scored for headline contamination, and re-scored network-matched.
    This measures how often the 68.6 defect is even POSSIBLE, and how often it MATTERS -- which
    is the prior for every cell the campaign has published but not audited.

PART B -- THE PUBLISHED CELLS.  The load-bearing cells of cycles 65-68 are hand-encoded with
    their document citation and their published numbers.  Each is RECONSTRUCTED from the CSV (a
    key search, generalising what 99.1 did by hand for one cell), then audited on every axis its
    final key leaves free, then re-scored composition-matched.

GATES
-----
G1  RECONSTRUCTION.  Every published cell must be reproduced to 3 decimals (n exact, mean and sem
    to <5e-3) by some key.  A cell that cannot be reconstructed is reported UNRECONSTRUCTABLE and
    NOTHING is concluded about it -- that is itself the 99.1 failure mode and is reported as such.
G2  POSITIVE CONTROL.  Run on 68.6's stated cell, the GENERALISED search must independently
    rediscover the network drop and reproduce 99.3's corrected numbers: relaxed Delta +2.367 /
    gate 0.452, matched Delta +1.821 / gate 0.331.  If it does not, the generalisation is broken
    and PART A is VOID.
G3  NEGATIVE CONTROL.  At least one published cell must come back CLEAN (headline TVD = 0 on
    every free axis).  If the audit flags every cell, the threshold is meaningless and no
    prevalence number may be quoted.
G4  PREVALENCE (PART A).  Fraction of scorable relaxed cells whose headline contrast is
    contaminated on `network`.  Registered bar: < 10% => the defect is LOCALISED (68.6 an
    outlier);  >= 10% => SYSTEMIC and the corpus needs a re-audit.
G5  MATERIALITY (PART A).  Among contaminated cells, the fraction in which network-matching FLIPS
    the argmax or FLIPS the resolution verdict.  Registered bar: < 20% => cosmetic;
    >= 20% => material.
G6  DIRECTION, registered AGAINST this campaign's interest.  Among corrected cells, how many move
    the headline margin DOWN (the campaign was over-claiming) vs UP.  A ~50/50 split means the
    defect is noise-like; a systematic DOWN skew means the campaign has an inflation bias and
    every unaudited margin should be read as an upper bound.  No bar -- reported either way.

USAGE
    python3 analysis/c70_composition_audit.py --selftest
    python3 analysis/c70_composition_audit.py --report
"""
import argparse
import collections
import csv
import itertools
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Every configuration axis a run can differ on.  `seed` is deliberately ABSENT: it is the
# replication axis and arms are SUPPOSED to differ on it.  `granularity` is absent: it is the
# axis being compared.  Outcome columns and bookkeeping (job_id, node, account, ...) are absent.
AXES = ['network', 'dataset', 'base', 'meta', 'meta_stepsize', 'alpha0', 'gamma',
        'augment', 'beta_clip', 'hier', 'lam', 'eta_ratio', 'epochs_done', 'window_ok']

TVD_THRESHOLD = 0.25      # registered above
MIN_ARM_N = 3             # an arm smaller than this is not scorable
GRAN_ORDER = ['weightwise', 'nodewise', 'resnet18_blocks', 'layerwise', 'scalar']


# --------------------------------------------------------------------------- primitives
def fnum(r, k):
    try:
        return float(r[k])
    except (ValueError, TypeError, KeyError):
        return None


def load(root=ROOT):
    with open(os.path.join(root, 'results', 'all_runs.csv')) as fh:
        return list(csv.DictReader(fh))


def stat(rs, col):
    v = [fnum(r, col) for r in rs]
    v = [x for x in v if x is not None]
    if not v:
        return None
    sem = st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return dict(n=len(v), mean=st.mean(v), sem=sem, min=min(v), max=max(v))


def gate_of(a, b):
    """65.6 / 94.3's resolution rule, unchanged."""
    return 2.0 * math.sqrt(a['sem'] ** 2 + b['sem'] ** 2)


def resolved(a, b):
    g = gate_of(a, b)
    return abs(a['mean'] - b['mean']) > g if g > 0 else False


def tvd(rows_a, rows_b, axis):
    """Total variation distance between two arms' composition on one axis."""
    ca, cb = collections.Counter(r[axis] for r in rows_a), collections.Counter(r[axis] for r in rows_b)
    na, nb = sum(ca.values()), sum(cb.values())
    if not na or not nb:
        return 0.0
    keys = set(ca) | set(cb)
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in keys)


def arms_of(rows, col):
    """Group rows by granularity, keeping only arms with a usable statistic."""
    g = collections.defaultdict(list)
    for r in rows:
        if fnum(r, col) is not None:
            g[r['granularity']].append(r)
    return {k: v for k, v in g.items() if k != '?'}


def headline(g, col, min_n=MIN_ARM_N):
    """The cell's argmax arm and its runner-up -- the pair a reader would quote."""
    scored = [(a, stat(v, col)) for a, v in g.items() if len(v) >= min_n]
    scored = [(a, s) for a, s in scored if s]
    if len(scored) < 2:
        return None
    scored.sort(key=lambda x: -x[1]['mean'])
    return scored[0][0], scored[1][0]


def contrast(g, a, b, col):
    sa, sb = stat(g[a], col), stat(g[b], col)
    gt = gate_of(sa, sb)
    return dict(arm_a=a, arm_b=b, a=sa, b=sb, delta=sa['mean'] - sb['mean'], gate=gt,
                mult=(sa['mean'] - sb['mean']) / gt if gt > 0 else float('inf'),
                resolved=resolved(sa, sb),
                overlap=not (sa['min'] > sb['max']))


# --------------------------------------------------------------------------- PART A
def relaxed_cells(rows, col='plateau5'):
    """Group on every axis EXCEPT network -- i.e. exactly the relaxation 68.6 performed."""
    key_axes = [a for a in AXES if a != 'network']
    cells = collections.defaultdict(list)
    for r in rows:
        if fnum(r, col) is None:
            continue
        cells[tuple(r[a] for a in key_axes)].append(r)
    return key_axes, cells


def pick_network(g, col, min_n=MIN_ARM_N):
    """Registered, deterministic: the network maximising the number of arms with n>=min_n,
    tie-broken by total run count, then alphabetically."""
    nets = sorted({r['network'] for v in g.values() for r in v})
    best = None
    for net in nets:
        sub = {a: [r for r in v if r['network'] == net] for a, v in g.items()}
        n_arms = sum(1 for v in sub.values() if len(v) >= min_n)
        total = sum(len(v) for v in sub.values())
        cand = (n_arms, total, net)
        if best is None or cand[:2] > best[:2]:
            best = cand
    return best[2] if best else None


def part_a(rows, col='plateau5'):
    key_axes, cells = relaxed_cells(rows, col)
    out = []
    for key, rs in cells.items():
        g = arms_of(rs, col)
        h = headline(g, col)
        if not h:
            continue
        a, b = h
        c_rel = contrast(g, a, b, col)
        t = tvd(g[a], g[b], 'network')
        nets = sorted({r['network'] for r in rs})
        rec = dict(key=dict(zip(key_axes, key)), n_rows=len(rs), arms={k: len(v) for k, v in g.items()},
                   networks=nets, headline=(a, b), tvd=t,
                   contaminated=t > TVD_THRESHOLD, rel=c_rel, matched=None,
                   flip_argmax=None, flip_resolution=None, net=None)
        if rec['contaminated']:
            net = pick_network(g, col)
            gm = {k: [r for r in v if r['network'] == net] for k, v in g.items()}
            gm = {k: v for k, v in gm.items() if v}
            hm = headline(gm, col)
            rec['net'] = net
            if hm:
                c_mat = contrast(gm, hm[0], hm[1], col)
                rec['matched'] = c_mat
                rec['flip_argmax'] = (hm[0] != a)
                rec['flip_resolution'] = (c_mat['resolved'] != c_rel['resolved'])
        out.append(rec)
    return out


# --------------------------------------------------------------------------- PART B
class Cell:
    """A published cell: what the document SAYS, and what it PRINTED."""

    def __init__(self, name, cite, col, stated, published, extra=None):
        self.name, self.cite, self.col = name, cite, col
        self.stated = stated              # axis -> value, as the document's own text names it
        self.published = published        # granularity -> (n, mean)  as printed
        self.extra = extra or {}          # non-axis predicates the document's prose states


def sel(rows, stated, extra):
    out = []
    for r in rows:
        if any(r.get(k) != v for k, v in stated.items()):
            continue
        if extra.get('free_beta') and r['meta'] == 'fixed':
            continue
        out.append(r)
    return out


def reconstruct(rows, cell, max_edit=2):
    """Search the MINIMAL edit to the stated key that reproduces every published arm.

    An edit is a DROP (an axis the document names but the data does not honour -- the 99.1
    defect) or an ADD (an axis the document's row label leaves unnamed but which the cell was
    in fact restricted to).  Minimal by (#drops + #adds), then by row count.  Deterministic.
    """
    free_axes = [a for a in AXES if a not in cell.stated]
    universe = sel(rows, {}, cell.extra)

    def matches(key):
        rs = sel(universe, key, {})
        g = arms_of(rs, cell.col)
        for gran, (n, mean) in cell.published.items():
            s = stat(g.get(gran, []), cell.col)
            if s is None or s['n'] != n or abs(s['mean'] - mean) >= 5e-3:
                return None
        return rs

    cands = []
    for n_drop in range(0, max_edit + 1):
        for drops in itertools.combinations(sorted(cell.stated), n_drop):
            base = {k: v for k, v in cell.stated.items() if k not in drops}
            for n_add in range(0, max_edit + 1 - n_drop):
                for adds in itertools.combinations(free_axes, n_add):
                    pools = []
                    rs0 = sel(universe, base, {})
                    for ax in adds:
                        pools.append(sorted({r[ax] for r in rs0}))
                    for combo in itertools.product(*pools) if pools else [()]:
                        key = dict(base)
                        key.update(dict(zip(adds, combo)))
                        rs = matches(key)
                        if rs is not None:
                            cands.append((n_drop + n_add, len(rs), tuple(sorted(drops)),
                                          tuple(sorted(adds)), key, rs))
            if cands:
                break
        if cands:
            break
    if not cands:
        return None
    cands.sort(key=lambda c: (c[0], c[1], c[2], c[3]))
    _, _, drops, adds, key, rs = cands[0]
    return dict(key=key, drops=list(drops), adds=list(adds), rows=rs)


def audit_cell(rows, cell):
    rec = reconstruct(rows, cell)
    if rec is None:
        return dict(cell=cell, ok=False)
    g = arms_of(rec['rows'], cell.col)
    free = [a for a in AXES if a not in rec['key']]
    h = headline(g, cell.col)
    res = dict(cell=cell, ok=True, key=rec['key'], drops=rec['drops'], adds=rec['adds'],
               n_rows=len(rec['rows']), arms={k: len(v) for k, v in g.items()},
               free=free, headline=h, tvds={}, contaminated=False,
               rel=None, matched=None, match_on=None)
    if not h:
        return res
    a, b = h
    res['rel'] = contrast(g, a, b, cell.col)
    for ax in free:
        vals = {r[ax] for r in rec['rows']}
        if len(vals) <= 1:
            continue
        res['tvds'][ax] = tvd(g[a], g[b], ax)
    bad = {ax: t for ax, t in res['tvds'].items() if t > TVD_THRESHOLD}
    res['contaminated'] = bool(bad)
    if bad:
        # match on the worst-offending axis, at its modal value in the headline pair
        ax = max(bad, key=lambda k: bad[k])
        cnt = collections.Counter(r[ax] for r in g[a] + g[b])
        val = cnt.most_common(1)[0][0]
        gm = {k: [r for r in v if r[ax] == val] for k, v in g.items()}
        gm = {k: v for k, v in gm.items() if v}
        res['match_on'] = (ax, val)
        hm = headline(gm, cell.col)
        if hm:
            res['matched'] = contrast(gm, hm[0], hm[1], cell.col)
            res['flip_argmax'] = (hm[0] != a)
            res['flip_resolution'] = (res['matched']['resolved'] != res['rel']['resolved'])
    return res


# The published cells of cycles 65-68, each with its citation and its printed numbers.
def cells():
    c653 = dict(dataset='CIFAR10', meta_stepsize='1e-3', alpha0='1e-3',
                epochs_done='20', hier='')
    return [
        Cell('68.6 c100 100ep (POSITIVE CONTROL)', 'FINDINGS 68.6 / CORRECTIONS 97.5, 99.1',
             'plateau5',
             dict(network='ResNet18_c100', dataset='CIFAR100', epochs_done='100', window_ok='1',
                  alpha0='1e-3', meta_stepsize='1e-3', base='SGDm', meta='Lion', augment='1',
                  hier='', beta_clip='-15:-2.3026'),
             {'layerwise': (14, 69.048), 'nodewise': (3, 71.415), 'scalar': (11, 22.208)}),
        Cell('65.3 r18 ms1e-3 a0=1e-3 20ep', 'FINDINGS 65.3 row 1 / 67.6 W2', 'plateau',
             dict(network='ResNet18', **c653),
             {'layerwise': (18, 74.524), 'nodewise': (16, 72.858), 'weightwise': (17, 68.496),
              'resnet18_blocks': (16, 73.120), 'scalar': (2, 71.960)},
             dict(free_beta=True)),
        Cell('65.3 c100 ms1e-3 a0=1e-3 20ep', 'FINDINGS 65.3 row 2', 'plateau',
             dict(dataset='CIFAR100', meta_stepsize='1e-3', alpha0='1e-3',
                  epochs_done='20', hier=''),
             {'layerwise': (13, 38.166), 'nodewise': (13, 37.424), 'weightwise': (13, 26.887),
              'resnet18_blocks': (10, 25.718)},
             dict(free_beta=True)),
        Cell('65.3 r10 ms1e-3 a0=1e-3 20ep', 'FINDINGS 65.3 row 3', 'plateau',
             dict(network='ResNet10', **c653),
             {'layerwise': (5, 70.900), 'nodewise': (5, 69.840), 'weightwise': (5, 61.568)},
             dict(free_beta=True)),
        Cell('65.3 r34 ms1e-3 a0=1e-3 20ep', 'FINDINGS 65.3 row 4', 'plateau',
             dict(network='ResNet34', **c653),
             {'layerwise': (5, 74.446), 'nodewise': (5, 71.346), 'weightwise': (5, 70.246)},
             dict(free_beta=True)),
    ]


# --------------------------------------------------------------------------- selftests
def selftest():
    t = 0

    def ck(cond, label):
        nonlocal t
        t += 1
        if not cond:
            print(f'  FAIL T{t}: {label}')
            sys.exit(1)
        print(f'  ok  T{t}: {label}')

    rows = load()
    ck(len(rows) == 1707, f'CSV has 1707 rows ({len(rows)})')
    ck('seed' not in AXES and 'granularity' not in AXES,
       'AXES excludes the replication axis and the compared axis')

    # --- TVD primitive
    A = [{'x': 'a'}] * 4
    B = [{'x': 'a'}] * 4
    ck(tvd(A, B, 'x') == 0.0, 'TVD of identical composition is 0')
    C = [{'x': 'b'}] * 4
    ck(tvd(A, C, 'x') == 1.0, 'TVD of disjoint composition is 1')
    D = [{'x': 'a'}] * 2 + [{'x': 'b'}] * 2
    ck(abs(tvd(A, D, 'x') - 0.5) < 1e-12, 'TVD of half-overlap is 0.5')
    ck(abs(tvd(A, A + C, 'x') - 0.5) < 1e-12, 'TVD is symmetric in construction (0.5)')

    # --- gate primitive reproduces c69/68.6's published gate
    a68 = dict(sem=0.049)
    b68 = dict(sem=0.220)
    ck(abs(gate_of(a68, b68) - 0.451) < 2e-3,
       f'gate rule reproduces 68.6\'s published 0.452 ({gate_of(a68, b68):.3f})')

    # --- G1 RECONSTRUCTION, every published cell
    audits = [audit_cell(rows, c) for c in cells()]
    for r in audits:
        ck(r['ok'], f"G1: {r['cell'].name} reconstructs from the CSV")

    pc = audits[0]
    # --- G2 POSITIVE CONTROL: rediscover 99.1/99.3 without being told
    ck(pc['drops'] == ['network'],
       f"G2: search independently drops the NETWORK axis on 68.6's cell (got {pc['drops']})")
    ck(pc['adds'] == [], f"G2: no compensating add needed (got {pc['adds']})")
    ck(pc['headline'] == ('nodewise', 'layerwise'),
       f"G2: headline pair is nodewise-vs-layerwise (got {pc['headline']})")
    ck(abs(pc['rel']['delta'] - 2.367) < 5e-3,
       f"G2: relaxed delta = 68.6's published +2.367 (got {pc['rel']['delta']:.3f})")
    ck(abs(pc['rel']['gate'] - 0.452) < 1e-3,
       f"G2: relaxed gate = 0.452 (got {pc['rel']['gate']:.3f})")
    ck(pc['contaminated'], 'G2: 68.6 cell is flagged CONTAMINATED')
    ck(abs(pc['tvds']['network'] - (1 - 8 / 14)) < 1e-9,
       f"G2: network TVD = 1-8/14 = 0.4286 (got {pc['tvds']['network']:.4f})")
    ck(pc['match_on'] == ('network', 'ResNet18_c100'),
       f"G2: matches on network=ResNet18_c100 (got {pc['match_on']})")
    ck(abs(pc['matched']['delta'] - 1.821) < 5e-3,
       f"G2: matched delta reproduces 99.3's +1.821 (got {pc['matched']['delta']:.3f})")
    ck(abs(pc['matched']['gate'] - 0.331) < 1e-3,
       f"G2: matched gate reproduces 99.3's 0.331 (got {pc['matched']['gate']:.3f})")
    ck(abs(pc['matched']['mult'] - 5.51) < 5e-3,
       f"G2: matched multiple reproduces 99.3's 5.51x (got {pc['matched']['mult']:.2f})")
    ck(pc['matched']['overlap'] is False, 'G2: no seed overlap under matching (99.3)')

    # --- G3 NEGATIVE CONTROL
    clean = [r for r in audits if r['ok'] and not r['contaminated']]
    ck(len(clean) >= 1,
       f'G3: at least one published cell is CLEAN ({len(clean)}/{len(audits)})')

    # --- PART A machinery
    pa = part_a(rows)
    ck(len(pa) > 0, f'PART A produces scorable cells ({len(pa)})')
    ck(all(r['tvd'] >= 0.0 and r['tvd'] <= 1.0 for r in pa), 'PART A TVDs are in [0,1]')
    ck(all(r['matched'] is None or r['net'] is not None for r in pa),
       'PART A: a matched re-score always names the network it matched on')
    single = [r for r in pa if len(r['networks']) == 1]
    ck(all(not r['contaminated'] for r in single),
       'PART A: a single-network cell can never be network-contaminated')
    ck(all(r['rel']['a']['mean'] >= r['rel']['b']['mean'] for r in pa),
       'PART A: headline pair is ordered argmax-first')

    # --- pick_network determinism
    g_demo = {'x': [{'network': 'N1'}] * 3 + [{'network': 'N2'}] * 5,
              'y': [{'network': 'N1'}] * 4 + [{'network': 'N2'}] * 2}
    ck(pick_network(g_demo, 'plateau5') == 'N1',
       'pick_network prefers the network with most arms at n>=3')

    print(f'\n{t}/{t} selftests PASS')
    return 0


# --------------------------------------------------------------------------- report
def report():
    rows = load()

    print('=' * 100)
    print('C70  COMPOSITION AUDIT -- does cross-network pooling reach beyond 68.6?   '
          '(CORRECTIONS 99.5)')
    print('=' * 100)

    print('\n' + '-' * 100)
    print('PART B -- THE PUBLISHED CELLS OF CYCLES 65-68')
    print('-' * 100)
    audits = [audit_cell(rows, c) for c in cells()]
    down = up = 0
    for r in audits:
        c = r['cell']
        print(f"\n### {c.name}\n    cite: {c.cite}   column: {c.col}")
        if not r['ok']:
            print('    *** UNRECONSTRUCTABLE from the CSV -- nothing concluded (see G1) ***')
            continue
        print(f"    key edits: DROP={r['drops'] or '-'}  ADD={r['adds'] or '-'}   "
              f"rows={r['n_rows']}")
        print(f"    arms: " + '  '.join(f'{a}={n}' for a, n in sorted(r['arms'].items())))
        if r['tvds']:
            print('    free-axis composition TVD (headline pair '
                  f"{r['headline'][0]} vs {r['headline'][1]}):")
            for ax, t in sorted(r['tvds'].items(), key=lambda x: -x[1]):
                flag = '  <== CONTAMINATED' if t > TVD_THRESHOLD else ''
                print(f'        {ax:16s} {t:.4f}{flag}')
        else:
            print('    free-axis composition: all free axes single-valued (fully matched)')
        rel = r['rel']
        print(f"    as published : {rel['arm_a']} - {rel['arm_b']} = {rel['delta']:+.3f} pp   "
              f"gate {rel['gate']:.3f}  {rel['mult']:.2f}x  "
              f"{'RESOLVED' if rel['resolved'] else 'unresolved'}")
        if r['matched']:
            m = r['matched']
            print(f"    matched on {r['match_on'][0]}={r['match_on'][1]}:")
            print(f"    corrected    : {m['arm_a']} - {m['arm_b']} = {m['delta']:+.3f} pp   "
                  f"gate {m['gate']:.3f}  {m['mult']:.2f}x  "
                  f"{'RESOLVED' if m['resolved'] else 'unresolved'}")
            print(f"    argmax flips: {r['flip_argmax']}   resolution flips: {r['flip_resolution']}")
            if abs(m['delta']) < abs(rel['delta']):
                down += 1
            elif abs(m['delta']) > abs(rel['delta']):
                up += 1
        elif not r['contaminated']:
            print('    verdict: CLEAN -- no composition correction applies')

    print(f"\nG6 DIRECTION (published cells): margin shrinks in {down}, grows in {up}")

    print('\n' + '-' * 100)
    print('PART A -- CAMPAIGN-WIDE BASE RATE (mechanical, blind to publication)')
    print('-' * 100)
    pa = part_a(rows)
    multi = [r for r in pa if len(r['networks']) > 1]
    cont = [r for r in pa if r['contaminated']]
    scored = [r for r in cont if r['matched']]
    fa = [r for r in scored if r['flip_argmax']]
    fr = [r for r in scored if r['flip_resolution']]
    print(f'  scorable relaxed cells (>=2 arms at n>={MIN_ARM_N}) : {len(pa)}')
    print(f'  of which span >1 network                          : {len(multi)} '
          f'({100*len(multi)/len(pa):.1f}%)')
    print(f'  of which headline contrast is CONTAMINATED        : {len(cont)} '
          f'({100*len(cont)/len(pa):.1f}%)   [G4 bar: 10%]')
    print(f"  G4 VERDICT: {'SYSTEMIC' if len(cont)/len(pa) >= 0.10 else 'LOCALISED'}")
    if scored:
        print(f'  contaminated cells re-scorable network-matched    : {len(scored)}')
        print(f'  argmax flips under matching                       : {len(fa)} '
              f'({100*len(fa)/len(scored):.1f}%)')
        print(f'  resolution verdict flips under matching           : {len(fr)} '
              f'({100*len(fr)/len(scored):.1f}%)')
        material = len({id(x) for x in fa} | {id(x) for x in fr})
        print(f'  EITHER flip (materiality)                         : {material} '
              f'({100*material/len(scored):.1f}%)   [G5 bar: 20%]')
        print(f"  G5 VERDICT: {'MATERIAL' if material/len(scored) >= 0.20 else 'COSMETIC'}")
        d = sum(1 for r in scored if abs(r['matched']['delta']) < abs(r['rel']['delta']))
        u = sum(1 for r in scored if abs(r['matched']['delta']) > abs(r['rel']['delta']))
        print(f'  G6 DIRECTION: margin shrinks in {d}, grows in {u} '
              f'({100*d/max(1,d+u):.0f}% shrink)')

        print('\n  the contaminated cells, worst TVD first:')
        print(f"    {'TVD':>6s} {'net':>16s} {'headline':>26s} {'as-pub':>9s} {'matched':>9s} "
              f"{'flipA':>5s} {'flipR':>5s}  key")
        for r in sorted(scored, key=lambda x: -x['tvd'])[:25]:
            k = r['key']
            lab = (f"{k['dataset']}/{k['base']}/{k['meta']}/ms{k['meta_stepsize']}/"
                   f"a0{k['alpha0']}/{k['epochs_done']}ep/hier{k['hier'] or '-'}")
            hl = f"{r['headline'][0][:11]}-{r['headline'][1][:11]}"
            print(f"    {r['tvd']:6.3f} {r['net']:>16s} {hl:>26s} "
                  f"{r['rel']['delta']:+9.3f} {r['matched']['delta']:+9.3f} "
                  f"{str(r['flip_argmax'])[0]:>5s} {str(r['flip_resolution'])[0]:>5s}  {lab}")
    return 0


# --------------------------------------------------------------------------- POST-HOC supplement
def posthoc():
    r"""POST-HOC, WRITTEN AFTER --report WAS READ, AND LABELLED SO EVERYWHERE.

    --report's G4/G5/G6 audit the HEADLINE pair only (a cell's argmax vs its runner-up), because
    that is the pair a reader quotes.  Reading the report exposed a scope limit: a document may
    quote a NON-headline pair.  The plain guarded CIFAR-10 ladder is the case that showed it --
    its headline pair (nodewise vs blocks) is TVD = 0.0000 and CLEAN, while the nodewise-vs-
    LAYERWISE contrast that carries the published "peak at nodewise, not layerwise" claim has
    TVD = 0.3704 and IS contaminated.

    So the registered G4 = 10.7% is a LOWER BOUND.  This mode measures the any-pair rate.
    It may not overturn a registered gate (CORRECTIONS 76(1)/79) and does not claim to; it
    bounds the registered gate's scope, which is a mechanical fact about coverage.
    """
    rows = load()
    print('=' * 100)
    print('C70 POST-HOC SUPPLEMENT -- any-pair contamination.  NOT a registered gate.')
    print('=' * 100)

    print('\n-- the case that motivated it: plain guarded ladder, CIFAR-10 a0=1e-6, 100 ep')
    sel = [r for r in rows if r['dataset'] == 'CIFAR10' and r['alpha0'] == '1e-6'
           and r['epochs_done'] == '100' and r['hier'] == '' and r['augment'] == '1'
           and r['beta_clip'] == '-15:-2.3026' and r['base'] == 'SGDm' and r['meta'] == 'Lion'
           and r['meta_stepsize'] == '1e-3' and r['window_ok'] == '1']
    g = arms_of(sel, 'plateau5')
    for col in ('plateau5', 'plateau'):
        c = contrast(g, 'nodewise', 'layerwise', col)
        gm = {k: [r for r in v if r['network'] == 'ResNet18'] for k, v in g.items()}
        cm = contrast(gm, 'nodewise', 'layerwise', col)
        print(f"   [{col:8s}] as-pooled  node-lay {c['delta']:+.3f}  gate {c['gate']:.3f}  "
              f"{c['mult']:.2f}x  n={c['a']['n']}/{c['b']['n']}  "
              f"TVDnet={tvd(g['nodewise'], g['layerwise'], 'network'):.4f}")
        print(f"   [{col:8s}] R18-match  node-lay {cm['delta']:+.3f}  gate {cm['gate']:.3f}  "
              f"{cm['mult']:.2f}x  n={cm['a']['n']}/{cm['b']['n']}")

    key_axes, cells_ = relaxed_cells(rows, 'plateau5')
    tot = head = anyp = anyp_flip = 0
    pairs = []
    for key, rs in cells_.items():
        g = arms_of(rs, 'plateau5')
        h = headline(g, 'plateau5')
        if not h:
            continue
        tot += 1
        if tvd(g[h[0]], g[h[1]], 'network') > TVD_THRESHOLD:
            head += 1
        scor = [a for a, v in g.items() if len(v) >= MIN_ARM_N]
        bad = [(a, b) for a, b in itertools.combinations(sorted(scor), 2)
               if tvd(g[a], g[b], 'network') > TVD_THRESHOLD]
        if not bad:
            continue
        anyp += 1
        net = pick_network(g, 'plateau5')
        flip = False
        for a, b in bad:
            ra = [r for r in g[a] if r['network'] == net]
            rb = [r for r in g[b] if r['network'] == net]
            if len(ra) < 2 or len(rb) < 2:
                continue
            c0 = contrast(g, a, b, 'plateau5')
            c1 = contrast({a: ra, b: rb}, a, b, 'plateau5')
            f = (c1['resolved'] != c0['resolved']) or ((c0['delta'] > 0) != (c1['delta'] > 0))
            flip = flip or f
            pairs.append((tvd(g[a], g[b], 'network'), a, b, c0, c1, dict(zip(key_axes, key))))
        if flip:
            anyp_flip += 1
    print(f'\n  scorable cells                     : {tot}')
    print(f'  HEADLINE-pair contaminated (G4)    : {head} ({100*head/tot:.1f}%)  <- registered')
    print(f'  ANY-pair contaminated              : {anyp} ({100*anyp/tot:.1f}%)  <- post-hoc')
    print(f'  ...with >=1 flipping pair          : {anyp_flip} ({100*anyp_flip/max(1,anyp):.1f}%)')
    d = sum(1 for p in pairs if abs(p[4]['delta']) < abs(p[3]['delta']))
    u = sum(1 for p in pairs if abs(p[4]['delta']) > abs(p[3]['delta']))
    print(f'  re-scorable contaminated PAIRS     : {len(pairs)}')
    print(f'  |margin| shrinks {d}, grows {u}  ({100*d/max(1,d+u):.0f}% shrink)  '
          f'<- G6 does NOT generalise; see FINDINGS 70.5')
    print('\n  worst 14 contaminated pairs:')
    for t, a, b, c0, c1, k in sorted(pairs, key=lambda x: -x[0])[:14]:
        print(f"   TVD {t:.3f} {a[:11]:>11s}-{b[:11]:<11s} {c0['delta']:+8.3f} -> {c1['delta']:+8.3f}"
              f"  res {str(c0['resolved'])[0]}->{str(c1['resolved'])[0]}  "
              f"{k['dataset']}/a0{k['alpha0']}/{k['epochs_done']}ep/"
              f"hier{k['hier'] or '-'}/eta{k['eta_ratio']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--posthoc', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.report:
        return report()
    if a.posthoc:
        return posthoc()
    ap.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
