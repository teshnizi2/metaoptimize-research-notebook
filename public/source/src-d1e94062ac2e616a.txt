#!/usr/bin/env python3
"""c57_surface_truncation.py -- the c40 response surface is contaminated by TRUNCATED runs,
and the contamination is GRANULARITY-ASYMMETRIC.

A run whose `epochs_done` is below its `epochs_requested` was killed early.  Its `plateau`
(mean of the last 5 epochs) is therefore a plateau read at epoch `epochs_done`, NOT at
`epochs_requested`.  Mixing such a cell into a 100-epoch response surface is the SAME unit
error CORRECTIONS 85 found in the FINDINGS 40.3 citation -- but here it is inside the surface
itself rather than in a citation of it.

Modes:
  --audit       corpus-wide: which families contain truncated runs, and how severe
  --surface     the rs-* surface, as-published vs full-100-only, per granularity x ms
  --envelope    per-arm tuned envelope (max over the ms grid) + selection-bias estimate
  --weightwise  the 100-epoch weightwise inventory (what rs-w would be compared against)
  --selftest    run the tests

Every number is re-derived from results/all_runs.csv at call time (STANDING RULE 1).
"""
import csv
import sys
import os
import math
import random
import collections
import statistics as st

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'all_runs.csv')

# The c40 surface's fixed operating point, read off bin/c40_surface.sh:48.
SURFACE_POINT = dict(network='ResNet18', dataset='CIFAR10', base='SGDm', meta='Lion',
                     alpha0='1e-3', augment='1', beta_clip='-15:-2.3026', hier='')
ARMS = ['scalar', 'resnet18_blocks', 'layerwise', 'nodewise', 'weightwise']


def load(path=CSV):
    with open(path) as f:
        return list(csv.DictReader(f))


def epochs(r):
    """(done, requested) as ints, or (None, None) if either is unparseable."""
    try:
        return int(r['epochs_done']), int(r['epochs_requested'])
    except (ValueError, KeyError):
        return None, None


def is_truncated(r):
    """True iff the run stopped before the budget it was launched with.

    A row whose epoch fields do not parse is NOT called truncated -- absence of evidence.
    """
    d, q = epochs(r)
    if d is None or q is None:
        return False
    return d < q


def severity(r):
    """done/requested in (0,1] for a truncated run, else 1.0."""
    d, q = epochs(r)
    if d is None or q is None or q == 0:
        return 1.0
    return d / q


def plateau(r):
    try:
        return float(r['plateau'])
    except (ValueError, KeyError):
        return None


def family(run):
    """Leading token of a run name, up to the first '-' or '_'."""
    for i, ch in enumerate(run):
        if ch in '-_':
            return run[:i]
    return run


def surface_rows(rows, prefix='rs-'):
    return [r for r in rows if r['run'].startswith(prefix)]


def cells(rows, key='meta_stepsize'):
    """{(granularity, float(key)): [plateau, ...]} skipping unusable rows."""
    out = collections.defaultdict(list)
    for r in rows:
        p = plateau(r)
        if p is None:
            continue
        try:
            k = float(r[key])
        except (ValueError, KeyError):
            continue
        out[(r['granularity'], k)].append(p)
    return out


def cell_means(c):
    return {k: (st.mean(v), len(v), (st.pstdev(v) if len(v) > 1 else None)) for k, v in c.items()}


def envelope(c, arm):
    """(best_mean, ms, n, n_cells) over the ms grid for one arm; None if the arm is absent."""
    sub = {ms: v for (g, ms), v in c.items() if g == arm}
    if not sub:
        return None
    ms = max(sub, key=lambda m: st.mean(sub[m]))
    return st.mean(sub[ms]), ms, len(sub[ms]), len(sub)


def pooled_sd(c, min_n=2):
    """Pooled within-cell sd across every cell with n>=min_n. None if no such cell."""
    num, den = 0.0, 0
    for v in c.values():
        if len(v) >= min_n:
            num += st.pstdev(v) ** 2 * (len(v) - 1)
            den += len(v) - 1
    return math.sqrt(num / den) if den else None


def envelope_bias(cell_ns, sd, trials=20000, rng=None):
    """E[max of k noisy cell means] - max of their true means, for cells all at the SAME
    true mean (the null in which tuning buys nothing).  cell_ns is the list of per-cell n.

    Returns the expected upward inflation, in the same units as sd.  This is the amount an
    envelope (a max over a grid) is inflated by selection alone.
    """
    if sd is None or not cell_ns:
        return None
    rng = rng or random.Random(0)
    ses = [sd / math.sqrt(n) for n in cell_ns]
    tot = 0.0
    for _ in range(trials):
        tot += max(rng.gauss(0.0, se) for se in ses)
    return tot / trials


# ---------------------------------------------------------------- reports

def report_audit(rows):
    tr = [r for r in rows if is_truncated(r)]
    print('CORPUS-WIDE TRUNCATION AUDIT  (epochs_done < epochs_requested)')
    print('rows=%d  truncated=%d (%.1f%%)\n' % (len(rows), len(tr), 100 * len(tr) / len(rows)))
    tot = collections.Counter(family(r['run']) for r in rows)
    byfam = collections.defaultdict(list)
    for r in tr:
        byfam[family(r['run'])].append(r)
    print('%-12s %6s %6s %6s  %s' % ('family', 'trunc', 'total', 'pct', 'severity done/req'))
    for f in sorted(byfam, key=lambda f: -len(byfam[f])):
        sev = sorted(severity(r) for r in byfam[f])
        print('%-12s %6d %6d %5.0f%%  min=%.2f med=%.2f'
              % (f, len(byfam[f]), tot[f], 100 * len(byfam[f]) / tot[f], sev[0], sev[len(sev) // 2]))
    print('\nGRANULARITY ASYMMETRY within each affected family')
    print('%-12s %-18s %6s %6s %6s' % ('family', 'granularity', 'trunc', 'total', 'pct'))
    for f in sorted(byfam, key=lambda f: -len(byfam[f])):
        fam_rows = [r for r in rows if family(r['run']) == f]
        for g in ARMS:
            sub = [r for r in fam_rows if r['granularity'] == g]
            if not sub:
                continue
            t = [r for r in sub if is_truncated(r)]
            print('%-12s %-18s %6d %6d %5.0f%%' % (f, g, len(t), len(sub), 100 * len(t) / len(sub)))


def report_surface(rows):
    rs = surface_rows(rows)
    full = [r for r in rs if not is_truncated(r)]
    print('THE c40 RESPONSE SURFACE  (rs-*, %d runs, %d full-budget)\n' % (len(rs), len(full)))
    for label, sel in (('AS PUBLISHED (all %d)' % len(rs), rs),
                       ('FULL-100 ONLY (%d)' % len(full), full)):
        c = cells(sel)
        cm = cell_means(c)
        print('=== %s ===' % label)
        for g in ARMS:
            row = sorted((ms, cm[(gg, ms)]) for (gg, ms) in cm if gg == g)
            if not row:
                continue
            s = '  '.join('%g:%.2f(n%d)' % (ms, m, n) for ms, (m, n, _) in row)
            print('  %-17s %s' % (g, s))
        print()


def report_envelope(rows):
    rs = surface_rows(rows)
    full = [r for r in rs if not is_truncated(r)]
    c = cells(full)
    sd = pooled_sd(c)
    print('TUNED ENVELOPE per arm  (max over the ms grid), FULL-100 runs only')
    print('pooled within-cell sd = %s pp  (over cells with n>=2)\n'
          % ('%.4f' % sd if sd else 'n/a'))
    print('%-18s %8s %8s %5s %6s  %s' % ('arm', 'envelope', '@ms', 'n', 'cells', 'sel-bias(null)'))
    envs = {}
    for g in ARMS:
        e = envelope(c, g)
        if not e:
            continue
        envs[g] = e
        m, ms, n, k = e
        ns = [len(v) for (gg, _), v in c.items() if gg == g]
        b = envelope_bias(ns, sd)
        print('%-18s %8.3f %8g %5d %6d  %+.3f pp' % (g, m, ms, n, k, b if b else 0.0))
    if envs:
        vals = [v[0] for v in envs.values()]
        print('\nSPREAD ACROSS TUNED ARMS: %.3f pp  (%.3f .. %.3f)'
              % (max(vals) - min(vals), min(vals), max(vals)))


def report_weightwise(rows):
    print('100-EPOCH WEIGHTWISE INVENTORY (what an rs-w cell would join)\n')
    ww = [r for r in rows if r['granularity'] == 'weightwise' and r['epochs_requested'] == '100'
          and not is_truncated(r)]
    g = collections.defaultdict(list)
    for r in ww:
        k = (r['base'], r['meta'], r['meta_stepsize'], r['alpha0'], r['augment'],
             r['beta_clip'], r['hier'], r['dataset'])
        g[k].append(r)
    print('%-58s %3s %9s %8s' % ('base|meta|ms|alpha0|aug|clip|hier|data', 'n', 'plateau', 'sd'))
    for k in sorted(g, key=lambda k: -st.mean([plateau(r) for r in g[k]])):
        v = [plateau(r) for r in g[k]]
        print('%-58s %3d %9.3f %8s'
              % ('|'.join(k), len(v), st.mean(v), '%.3f' % st.pstdev(v) if len(v) > 1 else '-'))
    print('\nMATCHED to the surface operating point (%s):' % SURFACE_POINT)
    m = [r for r in ww if all(r[k] == v for k, v in SURFACE_POINT.items())]
    for r in m:
        print('  %-22s plateau=%.3f  seed=%s' % (r['run'], plateau(r), r['seed']))
    print('  n = %d' % len(m))


# ---------------------------------------------------------------- selftest

def selftest():
    t = [0, 0]

    def ok(name, cond):
        t[1] += 1
        if cond:
            t[0] += 1
        else:
            print('  FAIL: %s' % name)

    # -- is_truncated / epochs
    ok('truncated when done<req', is_truncated({'epochs_done': '24', 'epochs_requested': '100'}))
    ok('not truncated when equal', not is_truncated({'epochs_done': '100', 'epochs_requested': '100'}))
    ok('not truncated when done>req', not is_truncated({'epochs_done': '101', 'epochs_requested': '100'}))
    ok('unparseable is not truncated', not is_truncated({'epochs_done': '?', 'epochs_requested': '100'}))
    ok('missing key is not truncated', not is_truncated({'epochs_done': '20'}))
    ok('severity ratio', abs(severity({'epochs_done': '24', 'epochs_requested': '100'}) - 0.24) < 1e-9)
    ok('severity of full run is 1', severity({'epochs_done': '100', 'epochs_requested': '100'}) == 1.0)
    ok('severity guards req=0', severity({'epochs_done': '0', 'epochs_requested': '0'}) == 1.0)

    # -- family
    ok('family dash', family('rs-blk6-3e4-s2') == 'rs')
    ok('family underscore', family('gate0c_blk6_m2_s1') == 'gate0c')
    ok('family no sep', family('abc') == 'abc')

    # -- plateau
    ok('plateau parses', plateau({'plateau': '90.913'}) == 90.913)
    ok('plateau of blank is None', plateau({'plateau': ''}) is None)

    # -- cells / means on a synthetic surface
    syn = [{'run': 'rs-a', 'granularity': 'scalar', 'meta_stepsize': '1e-3', 'plateau': '90.0',
            'epochs_done': '100', 'epochs_requested': '100'},
           {'run': 'rs-b', 'granularity': 'scalar', 'meta_stepsize': '1e-3', 'plateau': '92.0',
            'epochs_done': '100', 'epochs_requested': '100'},
           {'run': 'rs-c', 'granularity': 'scalar', 'meta_stepsize': '1e-4', 'plateau': '80.0',
            'epochs_done': '25', 'epochs_requested': '100'}]
    c_all = cells(syn)
    ok('cells groups by (gran,ms)', len(c_all) == 2)
    ok('cell mean', abs(cell_means(c_all)[('scalar', 1e-3)][0] - 91.0) < 1e-9)
    ok('cell n', cell_means(c_all)[('scalar', 1e-3)][1] == 2)
    ok('n=1 cell has sd None', cell_means(c_all)[('scalar', 1e-4)][2] is None)
    c_full = cells([r for r in syn if not is_truncated(r)])
    ok('dropping truncated drops the cell', len(c_full) == 1)
    ok('envelope picks best ms', envelope(c_all, 'scalar')[1] == 1e-3)
    ok('envelope on clean set', abs(envelope(c_full, 'scalar')[0] - 91.0) < 1e-9)
    ok('envelope counts cells', envelope(c_all, 'scalar')[3] == 2)
    ok('envelope of absent arm is None', envelope(c_all, 'nodewise') is None)

    # -- the truncated cell is the LOW one here, so cleaning must RAISE the arm minimum
    ok('cleaning raises the arm minimum',
       min(st.mean(v) for (g, _), v in c_full.items() if g == 'scalar')
       > min(st.mean(v) for (g, _), v in c_all.items() if g == 'scalar'))

    # -- pooled sd
    ok('pooled sd of one 2-run cell', abs(pooled_sd(c_all) - st.pstdev([90.0, 92.0])) < 1e-9)
    ok('pooled sd None when no n>=2 cell', pooled_sd({('a', 1.0): [1.0]}) is None)

    # -- envelope bias: positive, grows with #cells, zero-ish for a single cell
    b1 = envelope_bias([2], 1.0, trials=4000, rng=random.Random(1))
    b5 = envelope_bias([2] * 5, 1.0, trials=4000, rng=random.Random(1))
    ok('bias of 1 cell ~ 0', abs(b1) < 0.05)
    ok('bias positive for many cells', b5 > 0.3)
    ok('bias grows with cell count', b5 > b1)
    ok('bias scales with sd',
       abs(envelope_bias([2] * 5, 2.0, trials=4000, rng=random.Random(1)) - 2 * b5) < 0.15)
    ok('bias None without sd', envelope_bias([2], None) is None)
    ok('bias None without cells', envelope_bias([], 1.0) is None)

    # -- real corpus invariants (guard against a silently-empty selection)
    rows = load()
    ok('corpus non-empty', len(rows) > 1000)
    rs = surface_rows(rows)
    ok('rs surface non-empty', len(rs) > 0)
    ok('rs is all 100-epoch requested', {r['epochs_requested'] for r in rs} == {'100'})
    ok('rs is all alpha0=1e-3', {r['alpha0'] for r in rs} == {'1e-3'})
    ok('rs has truncated rows', any(is_truncated(r) for r in rs))
    ok('rs has full rows', any(not is_truncated(r) for r in rs))
    ok('rs truncation is asymmetric',
       all(not is_truncated(r) for r in rs if r['granularity'] in ('scalar', 'layerwise')))
    ok('rs has no weightwise arm', not any(r['granularity'] == 'weightwise' for r in rs))
    c = cells([r for r in rs if not is_truncated(r)])
    ok('clean surface has >=4 arms', len({g for g, _ in c}) >= 4)
    ok('clean pooled sd is small', 0 < pooled_sd(c) < 1.0)
    ww = [r for r in rows if r['granularity'] == 'weightwise']
    ok('weightwise runs exist', len(ww) > 100)
    ok('no truncated 100ep weightwise at the surface point',
       all(not is_truncated(r) for r in ww
           if r['epochs_requested'] == '100'
           and all(r[k] == v for k, v in SURFACE_POINT.items())))

    print('selftest: %d/%d' % (t[0], t[1]))
    return t[0] == t[1]


if __name__ == '__main__':
    a = sys.argv[1] if len(sys.argv) > 1 else '--audit'
    if a == '--selftest':
        sys.exit(0 if selftest() else 1)
    rows = load()
    {'--audit': report_audit, '--surface': report_surface,
     '--envelope': report_envelope, '--weightwise': report_weightwise}[a](rows)
