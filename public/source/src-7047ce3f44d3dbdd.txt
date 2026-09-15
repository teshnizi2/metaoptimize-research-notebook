#!/usr/bin/env python3
"""c74_wm9_score.py -- score the `wm9` weightwise meta-stepsize ladder.

REGISTERED BEFORE ANY wm9 NUMBER WAS READ.  Gates W0 / W0.2 / W1 / W1b / W2 are
transcribed from `bin/c73_wm9_weightwise_ms.sh` and the selftests assert them
against THAT SCRIPT'S OWN TEXT, so the registration and the batch cannot drift
(the failure CORRECTIONS 102.2 caught in c72: a gate that stopped reading the
quantity it names).

THE QUESTION.  FINDINGS 72.4's 13-axis-matched census found that WEIGHTWISE has
exactly ONE matched run in the 1761-run corpus, at ms=1e-3 -- an ms past the
optimum of every other swept rung.  So every "weightwise is worst" reading in
this campaign is confounded with a single untuned meta-stepsize, on the very rung
the 53.1%-sign-agreement result is about.  wm9 fills that rung: ms in
{1e-5,3e-5,1e-4,3e-4,1e-3} x seeds {0,1}, 100 ep, BETA_CLIP=-15:-2.3026.

THE GATES, IN SCORING ORDER (bin/c73_wm9_weightwise_ms.sh:57-85):

  W0    VALIDITY.  epochs_done == 100 == requested, collapsed false, PER JOB.

  W0.2  THE REPRODUCTION CONTROL, SCORED FIRST.  wm9 at ms=1e-3 vs the existing
        matched weightwise ms=1e-3 plateau5 = 91.308.  Bar |diff| <= 0.50 pp
        (STANDING RULE 9's level bar).  FAILS -> the ladder is VOID and no
        optimum is read off it.  **THIS GATE CAN ONLY VOID.**

  W1    THE OPTIMUM.  argmax over ms of mean plateau5, n=2 per cell.  DECIDED
        only if the gap to the runner-up exceeds 2 SE of the difference;
        otherwise UNDECIDED and the grid is refined.

  W1b   IS THE OPTIMUM INTERIOR?  An argmax at a grid EDGE (1e-5 or 1e-3) means
        the optimum is NOT bracketed, W2 is NOT scored, and the result is "the
        grid is wrong".  Registering this is what stops an edge maximum being
        read as a peak.

  W2    THE LADDER, ONLY IF W1b SAYS INTERIOR.  REGISTERED PREDICTION:
        weightwise's tuned plateau5 lands INSIDE [92.2, 92.9] and its optimal ms
        is <= 1e-4.
        CONFIRMS -> the granularity ladder is FLAT once every rung is ms-tuned,
             across a 180,000x range in group count (62 -> 11.17M).
        REFUTES (below 92.2 at its own best ms) -> granularity has a real cost at
             the finest partition that tuning cannot buy back.

PRIMARY STATISTIC is `plateau5` (mean of last 5 epochs).  `best_test` inflates
~0.37 pp, the size of this effect, and is never the primary (METRIC RULES).

USAGE
  python3 analysis/c74_wm9_score.py --selftest
  python3 analysis/c74_wm9_score.py --csv results/all_runs.csv
"""
import argparse
import collections
import csv
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPT = os.path.join(REPO, "bin", "c73_wm9_weightwise_ms.sh")

# ---------------------------------------------------------------------------
# THE REGISTRATION.  Every constant here is asserted against SCRIPT by selftest.
# ---------------------------------------------------------------------------
GRID = ("1e-5", "3e-5", "1e-4", "3e-4", "1e-3")
EDGES = ("1e-5", "1e-3")            # W1b: an argmax here means the grid is wrong
SEEDS = (0, 1)
EPOCHS = 100
CLIP = "-15:-2.3026"
W02_BAR = 0.50                      # pp, STANDING RULE 9 level bar
W02_REF = 91.308                    # matched weightwise ms=1e-3, from the census
W2_BAND = (92.2, 92.9)              # registered: tuned weightwise lands INSIDE
W2_MS_MAX = 1e-4                    # registered: best ms at or below 1e-4
W1_SE_MULT = 2.0                    # DECIDED only at gap > 2 SE of the difference

# The 13 axes of the matched census (bin/c73_wm9_weightwise_ms.sh:174-175).
AX = ('network', 'dataset', 'batch_size', 'base', 'meta', 'alpha0', 'gamma',
      'augment', 'beta_clip', 'hier', 'lam', 'eta_ratio', 'epochs_done')

FALSEY = ('False', 'false', '0', '')


# ---------------------------------------------------------------------------
def load(csvp):
    with open(csvp) as f:
        return list(csv.DictReader(f))


def wm9_rows(rows):
    return [r for r in rows if r['run'].startswith('wm9-')]


def ms_of(r):
    """The meta-stepsize of a wm9 run, read from the CSV column, cross-checked
    against the run NAME.  A mismatch is a submit-script bug and must abort --
    STANDING RULE (19): read the quantity you name from the source that carries
    it, and if two sources carry it, they must agree."""
    col = (r.get('meta_stepsize') or '').strip()
    m = re.match(r'wm9-w-(\d)e(\d)-s\d+$', r['run'])
    if not m:
        raise ValueError("wm9 run name does not parse: %r" % r['run'])
    named = "%se-%s" % (m.group(1), m.group(2))
    if col and _f(col) != _f(named):
        raise ValueError("wm9 %s: meta_stepsize column %r != name %r"
                         % (r['run'], col, named))
    return named


def _f(s):
    return float(s)


def seed_of(r):
    return int(r['run'].rsplit('-s', 1)[1])


# ---------------------------------------------------------------------------
# W0
def gate_W0(rs):
    out = []
    for r in sorted(rs, key=lambda r: r['run']):
        ed = r.get('epochs_done', '')
        ok_ep = ed.isdigit() and int(ed) == EPOCHS
        ok_col = r.get('collapsed', '') in FALSEY
        ok_pl = bool(r.get('plateau5', '').strip())
        out.append((r['run'], ok_ep and ok_col and ok_pl,
                    "epochs_done=%s collapsed=%s plateau5=%s"
                    % (ed, r.get('collapsed'), r.get('plateau5'))))
    return out


# W0.2
def gate_W02(rs, ref=W02_REF, bar=W02_BAR):
    vals = [float(r['plateau5']) for r in rs if ms_of(r) == '1e-3']
    if not vals:
        return None, None, False, "no wm9 ms=1e-3 arm -- the control cannot be scored"
    mean = statistics.mean(vals)
    diff = mean - ref
    return mean, diff, abs(diff) <= bar, "n=%d vs ref %.3f" % (len(vals), ref)


# W1
def cells_of(rs):
    c = collections.defaultdict(list)
    for r in rs:
        c[ms_of(r)].append(float(r['plateau5']))
    return c


def gate_W1(cells):
    """argmax over ms, DECIDED only if gap > 2 SE of the DIFFERENCE."""
    order = sorted(cells, key=lambda k: statistics.mean(cells[k]), reverse=True)
    best, run2 = order[0], order[1]
    mb, m2 = statistics.mean(cells[best]), statistics.mean(cells[run2])
    se_d = math.sqrt(_sem(cells[best]) ** 2 + _sem(cells[run2]) ** 2)
    gap = mb - m2
    decided = se_d > 0 and gap > W1_SE_MULT * se_d
    return dict(best=best, mean=mb, runner_up=run2, runner_mean=m2, gap=gap,
                se_diff=se_d, ratio=(gap / se_d if se_d > 0 else float('inf')),
                decided=decided, order=order)


def _sem(v):
    if len(v) < 2:
        return 0.0
    return statistics.stdev(v) / math.sqrt(len(v))


# W1b
def gate_W1b(best):
    return best not in EDGES


# W2
def gate_W2(tuned_mean, best_ms):
    lo, hi = W2_BAND
    in_band = lo <= tuned_mean <= hi
    ms_ok = float(best_ms) <= W2_MS_MAX
    return in_band, ms_ok


# ---------------------------------------------------------------------------
def matched_ladder(rows):
    """The four tuned rungs above weightwise, re-derived from the CSV on the FULL
    13-axis match (STANDING RULE 1, 15) -- never quoted from a comment."""
    ref = [r for r in rows if r['run'].startswith('rs-lay-1e4')]
    if not ref:
        return None, None
    sig = tuple(ref[0][a] for a in AX)
    out = {}
    for g in ('scalar', 'nodewise', 'blockwise6', 'layerwise'):
        cel = collections.defaultdict(list)
        for r in rows:
            if (tuple(r[a] for a in AX) == sig and r['granularity'] == g
                    and r['collapsed'] in FALSEY and r.get('plateau5', '').strip()):
                cel[r['meta_stepsize']].append(float(r['plateau5']))
        if cel:
            b = max(cel, key=lambda k: statistics.mean(cel[k]))
            out[g] = (b, statistics.mean(cel[b]), len(cel[b]))
    return sig, out


# ---------------------------------------------------------------------------
def selftest():
    """Assert the registration against the BATCH SCRIPT's own text."""
    n = p = 0
    src = open(SCRIPT).read() if os.path.exists(SCRIPT) else ""

    def ck(name, cond):
        nonlocal n, p
        n += 1
        if cond:
            p += 1
        else:
            print("  FAIL %s" % name)

    ck("script exists", bool(src))
    # --- the batch's own constants
    ck("script GRID matches registration",
       re.search(r'^GRID="([^"]+)"', src, re.M).group(1).split() == list(GRID))
    ck("script EPOCHS==100", re.search(r'^EPOCHS=(\d+)', src, re.M).group(1) == str(EPOCHS))
    ck("script CLIP matches", re.search(r'^CLIP=(\S+)', src, re.M).group(1) == CLIP)
    ck("script NJOBS == len(GRID)*len(SEEDS)",
       int(re.search(r'^NJOBS=(\d+)', src, re.M).group(1)) == len(GRID) * len(SEEDS))
    # --- the registered gate TEXT is present and says what this file says
    ck("W0.2 ref 91.308 in script", "91.308" in src)
    ck("W0.2 bar 0.50 in script", re.search(r'W0\.2[\s\S]{0,400}?0\.50', src))
    ck("W0.2 can only VOID", re.search(r'W0\.2[\s\S]{0,500}?ONLY VOID', src))
    ck("W1 decided at 2 SE", re.search(r'W1\b[\s\S]{0,400}?2 SE', src))
    ck("W1b edge rule in script", re.search(r'W1b[\s\S]{0,300}?grid EDGE', src))
    ck("W1b names both edges", "1e-5 or\n#         1e-3" in src or "(1e-5 or" in src)
    ck("W2 band [92.2, 92.9] in script", "[92.2, 92.9]" in src)
    ck("W2 band matches registration", W2_BAND == (92.2, 92.9))
    ck("W2 ms<=1e-4 in script", re.search(r'W2[\s\S]{0,900}?at or below 1e-4', src))
    ck("W2 gated on W1b", re.search(r'W2\s+THE LADDER, ONLY IF W1b', src))
    ck("plateau5 is primary in script", "plateau5 is PRIMARY" in src)
    ck("edges are the grid ends", EDGES == (GRID[0], GRID[-1]))
    ck("13 axes", len(AX) == 13)

    # --- ms_of: name/column agreement is enforced, not assumed
    ok = ms_of({'run': 'wm9-w-1e4-s0', 'meta_stepsize': '1e-4'}) == '1e-4'
    ck("ms_of reads the name", ok)
    ck("ms_of accepts equal-value column",
       ms_of({'run': 'wm9-w-3e5-s1', 'meta_stepsize': '3e-05'}) == '3e-5')
    try:
        ms_of({'run': 'wm9-w-1e4-s0', 'meta_stepsize': '1e-3'})
        ck("ms_of rejects a name/column mismatch", False)
    except ValueError:
        ck("ms_of rejects a name/column mismatch", True)
    try:
        ms_of({'run': 'bf9-w-s0', 'meta_stepsize': '1e-3'})
        ck("ms_of rejects a non-wm9 run", False)
    except ValueError:
        ck("ms_of rejects a non-wm9 run", True)

    # --- W0
    good = {'run': 'wm9-w-1e4-s0', 'epochs_done': '100', 'collapsed': 'False',
            'plateau5': '92.5'}
    ck("W0 passes a clean row", gate_W0([good])[0][1])
    ck("W0 fails a short run",
       not gate_W0([dict(good, epochs_done='80')])[0][1])
    ck("W0 fails a collapsed run",
       not gate_W0([dict(good, collapsed='True')])[0][1])
    ck("W0 fails an empty plateau5",
       not gate_W0([dict(good, plateau5='')])[0][1])

    # --- W0.2 can only VOID, and its bar is two-sided
    mk = lambda ms, v, s=0: {'run': 'wm9-w-%s-s%d' % (ms.replace('-', ''), s),
                             'meta_stepsize': ms, 'plateau5': str(v)}
    _m, d, ok2, _ = gate_W02([{'run': 'wm9-w-1e3-s0', 'meta_stepsize': '1e-3',
                               'plateau5': '91.308'}])
    ck("W0.2 exact reproduction passes", ok2 and abs(d) < 1e-9)
    _m, d, ok2, _ = gate_W02([{'run': 'wm9-w-1e3-s0', 'meta_stepsize': '1e-3',
                               'plateau5': '91.808'}])
    ck("W0.2 passes at exactly the bar", ok2)
    _m, d, ok2, _ = gate_W02([{'run': 'wm9-w-1e3-s0', 'meta_stepsize': '1e-3',
                               'plateau5': '91.9'}])
    ck("W0.2 fails above the bar", not ok2)
    _m, d, ok2, _ = gate_W02([{'run': 'wm9-w-1e3-s0', 'meta_stepsize': '1e-3',
                               'plateau5': '90.7'}])
    ck("W0.2 is two-sided (fails below too)", not ok2)
    _m, _d, ok2, why = gate_W02([{'run': 'wm9-w-1e4-s0', 'meta_stepsize': '1e-4',
                                  'plateau5': '92.5'}])
    ck("W0.2 with no control arm does not pass", not ok2 and 'cannot be scored' in why)

    # --- W1
    c = {'1e-5': [90.0, 90.1], '3e-5': [91.0, 91.1], '1e-4': [93.0, 93.1],
         '3e-4': [92.0, 92.1], '1e-3': [91.3, 91.3]}
    r = gate_W1(c)
    ck("W1 finds the argmax", r['best'] == '1e-4')
    ck("W1 names the runner-up", r['runner_up'] == '3e-4')
    ck("W1 decides a clean separation", r['decided'])
    tie = {'1e-4': [92.0, 93.0], '3e-4': [92.4, 92.5]}
    ck("W1 is UNDECIDED on an overlapping pair", not gate_W1(tie)['decided'])
    ck("W1 gap is positive", r['gap'] > 0)

    # --- W1b
    ck("W1b: 1e-4 is interior", gate_W1b('1e-4'))
    ck("W1b: 3e-5 is interior", gate_W1b('3e-5'))
    ck("W1b: 3e-4 is interior", gate_W1b('3e-4'))
    ck("W1b: 1e-5 is an EDGE", not gate_W1b('1e-5'))
    ck("W1b: 1e-3 is an EDGE", not gate_W1b('1e-3'))

    # --- W2
    ck("W2 confirms inside the band", gate_W2(92.5, '1e-4') == (True, True))
    ck("W2 refutes below the band", gate_W2(91.0, '1e-4') == (False, True))
    ck("W2 refutes above the band", gate_W2(93.5, '1e-4') == (False, True))
    ck("W2 band edges are inclusive", gate_W2(92.2, '1e-4')[0] and gate_W2(92.9, '1e-4')[0])
    ck("W2 flags ms above 1e-4", gate_W2(92.5, '3e-4')[1] is False)
    ck("W2 accepts ms below 1e-4", gate_W2(92.5, '3e-5')[1] is True)

    # --- _sem
    ck("_sem of one value is 0", _sem([1.0]) == 0.0)
    ck("_sem of two values", abs(_sem([1.0, 3.0]) - 1.0) < 1e-12)

    print("selftest: %d/%d %s" % (p, n, "PASS" if p == n else "FAIL"))
    return 0 if p == n else 1


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(REPO, "results", "all_runs.csv"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    rows = load(a.csv)
    rs = wm9_rows(rows)
    print("=" * 78)
    print("c74 -- wm9 WEIGHTWISE ms LADDER.  %d rows in %s" % (len(rs), a.csv))
    print("=" * 78)
    if not rs:
        print("NO wm9 ROWS.  Nothing to score.")
        return 1

    # ---- W0
    w0 = gate_W0(rs)
    npass = sum(1 for _, ok, _ in w0 if ok)
    print("\n--- W0  VALIDITY (epochs_done==%d, collapsed false, plateau5 present)" % EPOCHS)
    for run, ok, why in w0:
        print("    %-18s %-4s  %s" % (run, "PASS" if ok else "FAIL", why))
    print("    W0: %d/%d" % (npass, len(w0)))
    if npass != len(w0):
        print("    W0 FAILS -- the batch is not valid.  STOP.")
        return 1
    ok_rows = [r for r in rs if r['run'] in {x for x, o, _ in w0 if o}]

    # ---- W0.2  (SCORED FIRST, can only VOID)
    mean13, diff, ok02, why = gate_W02(ok_rows)
    print("\n--- W0.2  REPRODUCTION CONTROL (scored FIRST; can only VOID)")
    print("    wm9 ms=1e-3 plateau5 = %.3f   ref %.3f   diff %+.3f pp   bar +-%.2f  [%s]"
          % (mean13, W02_REF, diff, W02_BAR, why))
    print("    W0.2: %s" % ("PASS" if ok02 else "FAIL -> LADDER IS VOID"))
    if not ok02:
        print("    No optimum is read off a VOID ladder.  STOP.")
        return 1

    # ---- W1
    cells = cells_of(ok_rows)
    print("\n--- W1  THE OPTIMUM (argmax over ms of mean plateau5)")
    print("    %-8s %8s %8s %6s   %s" % ("ms", "mean", "sem", "n", "values"))
    for ms in GRID:
        v = cells.get(ms, [])
        if not v:
            print("    %-8s %8s -- MISSING" % (ms, ""))
            continue
        print("    %-8s %8.3f %8.3f %6d   %s"
              % (ms, statistics.mean(v), _sem(v), len(v),
                 " ".join("%.3f" % x for x in sorted(v, reverse=True))))
    r1 = gate_W1(cells)
    print("    argmax = ms=%s  %.3f   runner-up ms=%s %.3f   gap %+.3f  SE_diff %.3f  gap/SE %.2f"
          % (r1['best'], r1['mean'], r1['runner_up'], r1['runner_mean'],
             r1['gap'], r1['se_diff'], r1['ratio']))
    print("    W1: %s" % ("DECIDED (gap > 2 SE)" if r1['decided']
                          else "UNDECIDED (gap <= 2 SE) -- the grid is refined, not read"))

    # ---- W1b
    interior = gate_W1b(r1['best'])
    print("\n--- W1b  IS THE OPTIMUM INTERIOR?  (grid %s; edges %s)"
          % (" ".join(GRID), ", ".join(EDGES)))
    print("    argmax ms=%s is %s" % (r1['best'], "INTERIOR" if interior else "AT A GRID EDGE"))
    if not interior:
        print("    W1b FIRES: the optimum is NOT bracketed.  **W2 IS NOT SCORED.**")
        print("    The registered result is 'THE GRID IS WRONG'; the next batch")
        print("    extends the grid in the ms=%s direction." % r1['best'])

    # ---- the matched ladder, re-derived from the CSV
    sig, lad = matched_ladder(rows)
    print("\n--- THE 13-AXIS-MATCHED TUNED LADDER (re-derived from the CSV, not quoted)")
    if sig:
        print("    signature: %s" % dict(zip(AX, sig)))
    for g in ('scalar', 'nodewise', 'blockwise6', 'layerwise'):
        if g in lad:
            b, m, n_ = lad[g]
            print("    %-11s best ms=%-6s plateau5 %8.3f  (n=%d)" % (g, b, m, n_))
        else:
            print("    %-11s -- no matched cell" % g)
    if lad:
        ms_ = [m for _, m, _ in lad.values()]
        print("    four-rung span: %.3f pp  [%.3f .. %.3f]"
              % (max(ms_) - min(ms_), min(ms_), max(ms_)))

    # ---- W2
    print("\n--- W2  THE REGISTERED PREDICTION  (tuned weightwise INSIDE [%.1f, %.1f], best ms <= 1e-4)"
          % W2_BAND)
    if not interior:
        print("    NOT SCORED -- W1b says the grid is wrong.  This is the registered")
        print("    behaviour and it is not a result about the ladder's flatness.")
    else:
        in_band, ms_ok = gate_W2(r1['mean'], r1['best'])
        print("    tuned weightwise = %.3f  -> %s the band" % (r1['mean'],
              "INSIDE" if in_band else "OUTSIDE"))
        print("    best ms = %s  -> %s" % (r1['best'],
              "<= 1e-4 as registered" if ms_ok else "ABOVE 1e-4 (registration missed)"))
        if in_band and ms_ok:
            print("    W2 CONFIRMS: the granularity ladder is FLAT once every rung is")
            print("    ms-tuned, across a 180,000x range in group count.")
        elif not in_band and r1['mean'] < W2_BAND[0]:
            print("    W2 REFUTES: the finest partition carries a real cost that tuning")
            print("    does not buy back.  The ladder is NOT flat.")
        else:
            print("    W2 MIXED -- report the components separately, never as one verdict.")
        if lad:
            ms_ = [m for _, m, _ in lad.values()] + [r1['mean']]
            print("    FIVE-rung span including tuned weightwise: %.3f pp  [%.3f .. %.3f]"
                  % (max(ms_) - min(ms_), min(ms_), max(ms_)))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
