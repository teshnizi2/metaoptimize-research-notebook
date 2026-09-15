#!/usr/bin/env python3
# =============================================================================
# cI1_in489g1_score.py -- THE REGISTERED SCORER FOR `in489g1`, THE IMAGENET-489
#                         STEP-SIZE GRANULARITY LADDER.
#
# STANDING RULE 21: this file is git-committed BEFORE any in489g1 run exists.
# Every bar, band, scale choice and refusal below is frozen here.  selftest()
# RE-DERIVES every archived constant from results/all_runs.csv rather than
# quoting prose, and asserts that the corpus contains ZERO imagenet489 rows at
# registration time.
#
# STANDING RULE 16: run this file UNEDITED and quote its verdict.  Passing its
# documented arguments is not editing it.  Once an in489g1 run exists it is
# frozen.
#
# =============================================================================
# WHAT THIS IS, AND WHAT IT IS NOT -- SCOPE, WHICH IS NOT OPTIONAL BOILERPLATE
# =============================================================================
# The task is **ImageNet-489**.  It is NOT ImageNet, and it must never be called
# ImageNet without the qualifier.
#
#   * 489 of ImageNet-1k's 1000 train classes -- the 489 the PI's tree actually
#     holds.  627,329 train images.
#   * Validation is 24,450 images, EXACTLY 50 per class, 0 broken symlinks.
#   * The validation LABELS WERE RECOVERED FROM THE PER-IMAGE XML ANNOTATIONS
#     under ILSVRC/Annotations/CLS-LOC/val/ (each carries <name>nXXXXXXXX</name>),
#     NOT from a devkit and NOT from a ground-truth .mat.  All 50,000 XMLs
#     parsed; 1000 distinct val classes found; 489 overlap the trainable set.
#   * Chance top-1 is 1/489 = 0.2045%.  The head is built num_classes=489, so
#     chance is 1/489 and NOT 1/1000.
#   * NO NUMBER PRODUCED BY THIS SCORER IS COMPARABLE TO A PUBLISHED
#     IMAGENET-1K RESULT.  Fewer classes is an easier task at equal everything
#     else, and the train set is 49% of ImageNet-1k's.
#
# =============================================================================
# THE QUESTION, PRE-REGISTERED
# =============================================================================
# The parent paper (Sharifnassab, Salehkaleybar & Sutton, ICML 2025,
# arXiv:2402.02342) motivates hierarchical pooling with a SCALE anomaly: finer
# step-size granularity beats a scalar step size on small vision tasks, and
# stops beating it on ImageNet.
#
# TWO CORRECTIONS TO HOW THAT ANOMALY IS USUALLY STATED IN THIS PROJECT, BOTH
# RE-DERIVED, BOTH LOAD-BEARING FOR WHICH CONTRAST IS PRIMARY:
#
#   (i)  The parent's ImageNet comparison is **BLOCKWISE vs scalar**, not
#        layerwise vs scalar.  The paper reports NO layerwise ImageNet arm.
#        So a scalar/layerwise/nodewise ladder alone does NOT test the parent's
#        own contrast.  That is why `resnet18_blocks` (m=6) is an arm here and
#        why BLOCK, not GRAN, is PRIMARY-A.
#   (ii) The parent's claim is **"no improvement over scalar"**, not "loses to
#        scalar".  A null is what they report.  So VANISHES -- not REVERSES --
#        is the band that corroborates them, and the bands below are written so
#        that a null is a POSITIVE, publishable finding rather than a failure.
#
# See docs/CORRECTIONS.md 119.0 K1: the parent states no numeric table, no seed
# count and no error bars for its granularity claims.  This batch therefore has
# STRICTLY better replication than the claim it tests, which is the point.
#
# EITHER ANSWER IS PUBLISHABLE, AND THE BANDS ARE SET BEFORE ANY RUN EXISTS:
#   * granularity does NOT help at 224px/489 classes -> the parent's ImageNet
#     report is corroborated at full resolution for the first time in this
#     project, and the hierarchical-pooling idea finally has the motivating
#     phenomenon it was invented to explain.
#   * granularity DOES help -> the anomaly does not survive at 489 classes, and
#     the premise of the whole hierarchical programme is reframed: whatever
#     kills granularity on ImageNet-1k is not resolution and not class count in
#     the 10 -> 489 range.
#
# =============================================================================
# THE DESIGN.  12 jobs, ONE submission, so BATCH -- the unit of replication in
# this corpus (SEED is null, F 1.21, p 0.213) -- is held constant and every
# contrast below is WITHIN this batch.
#
#   arm  scal   --stepsize-groups scalar           m =     1
#        blk    --stepsize-groups resnet18_blocks  m =     6   <- PARENT-MATCHED
#        lay    --stepsize-groups layerwise        m =    62
#        node   --stepsize-groups nodewise         m = 15378
#   seed in {0, 1, 2}
#
#   Held identical across all 12: ImageNet-489 / torchvision resnet18
#   (num_classes=489) / HF / base SGDm (momentum 0.99, wd 0.1) / meta Lion
#   (momentum 0.99, beta2 0.9, wd 0) / gamma 1 / meta-stepsize 1e-3 /
#   alpha0 1e-6 / BETA_CLIP -15:-2.3026 / AUGMENT=1 / HIER unset /
#   batch 256 / 84 epochs requested / full epochs (steps_per_epoch=0).
#
# WHY meta-stepsize 1e-3 AND NOT THE SMOKE TEST'S 3e-2.  Re-derived from
# results/all_runs.csv, not from any briefing: the campaign's CIFAR-10 corpus at
# this alpha0 and this BETA_CLIP is at meta-stepsize **1e-3**, and 1e-3 is ALSO
# the parent's own ImageNet meta-stepsize.  There is no CIFAR-10 scalar-vs-
# layerwise ladder at 3e-2 anywhere in the corpus, so 3e-2 would have left this
# batch with NO matched control on either side.  1e-3 makes the batch
# simultaneously comparable to the parent AND to this campaign's own CIFAR-10
# ladder.  Phase 1's smoke used 3e-2; that was fine for proving the pipeline and
# wrong for the ladder.  The ARGS line of every in489g1 run differs from the
# archived `mx` CIFAR ARGS line ONLY in dataset, batch size, epoch count and
# num-classes.
#
# =============================================================================
# THE PRIMARY STATISTIC AND THE SCALE, BOTH FROZEN
# =============================================================================
#   W5(run) = mean test accuracy over the LAST 5 SCORED epochs, where "scored"
#   means epochs 0 .. ESTAR-1 and ESTAR is the COMMON HORIZON (below).
#   The CSV `plateau` column (mean-of-last-20) is BANNED as primary and is never
#   read by this file.  `best_test` is never read.  M(arm) = mean W5 over seeds.
#
#   SCALE, PRE-REGISTERED, BECAUSE SWITCHING SCALES AFTER SEEING THE DATA IS THE
#   FAILURE CORRECTIONS 117.11 IS ABOUT:  the PRIMARY scale is ABSOLUTE
#   PERCENTAGE POINTS (pp), the same scale as the archived CIFAR-10 bars.  An
#   error-rate ratio R(arm) = (100 - M(arm)) / (100 - M(scal)) is computed and
#   ALWAYS PRINTED as a SECONDARY, and it may NEVER override the pp verdict.
#   Registered in advance: ImageNet-489 will sit at a much lower absolute
#   accuracy than CIFAR-10's 88-92% regime, so a given pp gap is a LARGER
#   relative effect here.  That asymmetry favours detection, is stated now, and
#   is the reason pp is the conservative choice for a REPLICATES verdict.
#
#   CONTRASTS:
#     BLOCK = M(blk)  - M(scal)     PRIMARY-A  (the parent's own contrast)
#     GRAN  = M(lay)  - M(scal)     PRIMARY-B  (this campaign's own contrast)
#     NODE  = M(node) - M(lay)      SECONDARY  (finer-still direction)
#
#   t for every contrast is computed from THIS BATCH's own 3-seed spread, never
#   from the archived CIFAR-10 noise floor.  A wide in-batch spread therefore
#   correctly yields UNRESOLVED instead of a verdict.
#
# =============================================================================
# THE ARCHIVED CIFAR-10 REFERENCE.  Re-derived by selftest() from
# results/all_runs.csv over the cell that matches this batch's optimizer
# configuration EXACTLY -- ResNet18 / CIFAR10 / base SGDm / meta Lion /
# meta-stepsize 1e-3 / alpha0 1e-6 / BETA_CLIP -15:-2.3026 / AUGMENT=1 /
# gamma 1 / batch 100 / 100 epochs / HIER unset / collapsed 0 / window_ok 1 /
# complete 1 / not superseded:
#
#     scalar           87.879 pp   n=12  over 5 batches
#     resnet18_blocks  91.448 pp   n=12  over 5 batches
#     layerwise        90.893 pp   n=17  over 6 batches
#     nodewise         91.739 pp   n= 8  over 3 batches
#     weightwise       79.183 pp   n= 4  over 2 batches   (collapses; not an arm)
#
#     BLOCK_CIFAR = +3.5683 pp     GRAN_CIFAR = +3.0134 pp
#     NODE_CIFAR  = +0.8461 pp
#     pooled within-cell SD of W5 = 0.1988 pp   (df 32, 17 cells)
#
# THE REFERENCE IS CROSS-BATCH REPLICATED, WHICH IS WHY IT CAN CARRY A BAR.
# Per-batch D vs scalar inside that same cell:
#     layerwise        a0L +3.162  a0h +3.088  mx +3.164  sc +2.817
#                      -> mean +3.058, BETWEEN-BATCH sd 0.142, 4 batches
#     resnet18_blocks  a0L +3.687  a0h +3.808  g4 +3.451  mx +3.540  sc +3.555
#                      -> mean +3.608, BETWEEN-BATCH sd 0.125, 5 batches
# A between-batch sd of ~0.13 pp on a ~+3.1 to +3.6 pp effect is what makes a
# half-size bar meaningful rather than arbitrary.
#
# ONE CONTRADICTION IN THE CORPUS WAS CHASED DOWN RATHER THAN AVERAGED OVER, AND
# IT IS RECORDED HERE BECAUSE IT NEARLY SET THE BAR WRONG.  The `bo-lion` runs
# carry the same nominal alpha0/BETA_CLIP/meta-stepsize and report scalar 91.452
# vs layerwise 91.281, i.e. D = -0.171, the OPPOSITE sign to `mx`'s +3.307.  The
# resolution is in the CSV's `base` column and in FINDINGS 5596: `bo-lion` has
# **base = Lion, not SGDm**.  It is a different base optimizer and is NOT
# config-matched, so it is excluded by the `base == SGDm` predicate above.  Had
# it been pooled in, the reference lift would have been diluted and this scorer
# would have carried a bar roughly half its correct size.
#
# =============================================================================
# THE BARS, FROZEN.  Bar = HALF the archived CIFAR-10 lift, the same bar SHAPE
# already used by the registered cT1 scorer, so this is a campaign convention
# and not a number invented for this batch.
# =============================================================================
#   BLOCK  (PRIMARY-A, the parent's own contrast; CIFAR reference +3.5683)
#     REPLICATES   BLOCK >= +1.784  AND t >= +2
#                  -> blockwise still beats scalar at 224px/489 classes.  The
#                     parent's ImageNet null does NOT survive here.
#     VANISHES     |BLOCK| <= 0.50
#                  -> "no improvement over scalar", which is EXACTLY the
#                     parent's ImageNet claim, reproduced at full resolution.
#                     A POSITIVE FINDING, not a failure.
#     REVERSES     BLOCK <= -0.50 AND t <= -2
#                  -> scalar BEATS blockwise; stronger than the parent's claim.
#     otherwise    UNRESOLVED.
#
#   GRAN   (PRIMARY-B; CIFAR reference +3.0134)
#     REPLICATES   GRAN >= +1.507  AND t >= +2
#     VANISHES     |GRAN| <= 0.50
#     REVERSES     GRAN <= -0.50 AND t <= -2
#     otherwise    UNRESOLVED.
#
#   NODE   (SECONDARY; CIFAR reference +0.8461)
#     FINER_HELPS  NODE >= +0.423 AND t >= +2
#     FINER_HURTS  NODE <= -0.423 AND t <= -2
#     NODE_NULL    |NODE| <= 0.25
#     otherwise    UNRESOLVED.
#
#   BATCH VERDICT, composed from the two primaries and frozen here:
#     ANOMALY_ABSENT        BLOCK REPLICATES and GRAN REPLICATES
#     CORROBORATES_PARENT   BLOCK in {VANISHES, REVERSES} and
#                           GRAN  in {VANISHES, REVERSES}
#     SPLIT                 the two primaries land in different families; the
#                           report must name both and claim neither pole.
#     UNRESOLVED            either primary UNRESOLVED, or any gate refuses.
#
# HONESTY ABOUT POWER, REGISTERED IN ADVANCE SO IT CANNOT BE DISCOVERED
# AFTERWARDS.  The 0.1988 pp floor is a CIFAR-10 number at 88-92% accuracy.
# ImageNet-489's run-to-run SD has NEVER been measured by anyone here -- this is
# the campaign's first 224px batch -- and a 489-class task at lower absolute
# accuracy is plausibly noisier.  With 3 seeds per arm, SE(diff) = SD*sqrt(2/3):
#     SD = 1x CIFAR   SE 0.162   t(full BLOCK) 22.0   t(half) 11.0
#     SD = 2x CIFAR   SE 0.325   t(full BLOCK) 11.0   t(half)  5.5
#     SD = 4x CIFAR   SE 0.649   t(full BLOCK)  5.5   t(half)  2.7
#     SD = 6x CIFAR   SE 0.974   t(full BLOCK)  3.7   t(half)  1.8 -> UNRESOLVED
# So this batch is powered for an effect of roughly the archived size at any
# plausible noise level, and is NOT powered for a half-size effect if
# ImageNet-489 turns out ~6x noisier than CIFAR-10.  Registered here in advance
# so that a wide result cannot afterwards be reported as a null.
#
# =============================================================================
# THE COMMON HORIZON, AND WHY IT IS PART OF THE PRE-REGISTRATION
# =============================================================================
# ImageNet-489 runs are long (measured 15.9 min/epoch on an L4) and share a
# contended partition, so walltime truncation is a REAL risk, not a hypothetical.
# The harness writes NO checkpoints but DOES flush a per-epoch accuracy line, so
# a truncated run still yields a valid learning curve up to where it stopped.
#
#   ESTAR = min over admissible runs of (number of scored epochs).
#   Every arm is read at the SAME ESTAR, so unequal run lengths cannot bias any
#   contrast.  Frozen bands on ESTAR:
#     ESTAR >= 60   full-budget verdict; W5 may be called a plateau.
#     30 <= ESTAR < 60   verdict allowed but STAMPED short_budget, and W5 must
#                        be called "window5@ESTAR", never a plateau.
#     ESTAR < 30    UNRESOLVED_INSUFFICIENT_BUDGET.  No verdict at any bar.
#
# AND THE BUDGET-INVERSION WARNING IS REGISTERED, NOT DISCOVERED LATER.
# CORRECTIONS 1 shows this very ordering INVERTS with budget on CIFAR-10: at 100
# epochs plain layerwise beats scalar, at 300 epochs scalar overtakes it.  So a
# verdict at ESTAR is a verdict AT THAT BUDGET ONLY.  This file therefore always
# prints the FULL epoch-wise contrast curve, and the report must carry the curve
# and not only the endpoint.
#
# =============================================================================
# THE GATES
# =============================================================================
# G0  PROVENANCE, ABSOLUTELY GATING (STANDING RULE 20).  Every fact about a run
#     is read from that run's OWN ARGS / ENV / EFFECTIVE lines.  The arm is
#     identified by --stepsize-groups, NEVER by the run name.  Any repeated flag
#     VOIDS that run: argparse keeps the last occurrence, and two whole batches
#     of this campaign were silently voided that way.
# G1  CHANCE FLOOR.  ImageNet-489 chance is 100/489 = 0.2045%.  A run whose W5
#     is below 5.0 pp has not learned 489 classes and is VOID as NOT_LEARNED.
# G2  m MUST BE READ FROM THE LIVE TREE.  Each run prints a GROUPS line with m
#     from its live beta.  scalar must be 1, resnet18_blocks 6, layerwise 62,
#     nodewise 15378 -- the values Phase 1 measured on the live 489-class tree.
#     m must also be identical across the seeds of an arm.
# G3  OPTIMIZER PROVENANCE.  HF_SOURCE must resolve inside the campaign's
#     maintained cifar10 tree.  The pre-existing imagenet/Optimizers/HF.py
#     cannot construct any non-scalar arm at all (list.cuda() AttributeError),
#     so a run whose HF came from there is not a granularity run.
# G4  DESIGN CONSISTENCY.  Every non-axis flag identical across all 12 runs;
#     --stepsize-groups and --seed are the ONLY axes.  DATASET must be
#     imagenet489, classes 489, train_images 627329, val_images 24450.
# G5  FULL EPOCHS.  EPOCH_DEF must report short_epoch=False.  A short-epoch run
#     is a smoke test, not a ladder run, and is VOID here.
# =============================================================================

import argparse
import csv
import glob
import math
import os
import re
import statistics as st
import sys

# ---- FROZEN CONSTANTS (RULE 21) --------------------------------------------
CIFAR_SCALAR = 87.879
CIFAR_BLOCK = 91.448
CIFAR_LAY = 90.893
CIFAR_NODE = 91.739
BLOCK_CIFAR = 3.5683
GRAN_CIFAR = 3.0134
NODE_CIFAR = 0.8461
POOLED_SD = 0.1988

BAR_BLOCK = 1.784          # half of BLOCK_CIFAR
BAR_GRAN = 1.507           # half of GRAN_CIFAR
BAR_NODE = 0.423           # half of NODE_CIFAR
NULL_BAND = 0.50           # |D| <= this  -> VANISHES  (the parent's own claim)
NODE_NULL_BAND = 0.25
T_BAR = 2.0

CHANCE = 100.0 / 489.0     # 0.2045 %
FLOOR_W5 = 5.0             # G1
ESTAR_FULL = 60            # >= this: plateau language allowed
ESTAR_MIN = 30             # < this: no verdict at any bar

EXPECT_M = {'scalar': 1, 'resnet18_blocks': 6, 'layerwise': 62, 'nodewise': 15378}
ARMS = ['scalar', 'resnet18_blocks', 'layerwise', 'nodewise']
ARM_SHORT = {'scalar': 'scal', 'resnet18_blocks': 'blk',
             'layerwise': 'lay', 'nodewise': 'node'}

EXPECT_DATA = {'classes': 489, 'train_images': 627329, 'val_images': 24450}

# Non-axis flags that must be identical across the batch (G4).
AXIS_FLAGS = {'stepsize-groups', 'seed', 'run-name', 'save-directory'}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- parsing ----------------------------------------------------------------
def _tokenise(argstr):
    """--flag value pairs -> dict, plus the list of flags in order of appearance
    so a REPEAT can be detected (RULE 20)."""
    toks = argstr.split()
    eff, order = {}, []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith('--'):
            flag = t[2:]
            val = ''
            if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                val = toks[i + 1]
                i += 1
            order.append(flag)
            eff[flag] = val
        i += 1
    return eff, order


def parse_out(path):
    r = {'path': path, 'name': os.path.basename(path), 'void': [], 'epochs': []}
    try:
        txt = open(path, errors='replace').read()
    except OSError as e:
        r['void'].append('UNREADABLE %s' % e)
        return r

    m = re.search(r'^ARGS:(.*)$', txt, re.M)
    if not m:
        r['void'].append('NO_ARGS_LINE')
        return r
    eff, order = _tokenise(m.group(1))
    r['args'] = eff
    dups = sorted({f for f in order if order.count(f) > 1})
    if dups:
        r['void'].append('REPEATED_FLAG %s' % ','.join(dups))

    m = re.search(r'^ENV:(.*)$', txt, re.M)
    r['env'] = dict(re.findall(r'(\w+)=(\S+)', m.group(1))) if m else {}
    if not m:
        r['void'].append('NO_ENV_LINE')

    m = re.search(r'^GROUPS:.*?\bm=(\d+)', txt, re.M)
    r['m'] = int(m.group(1)) if m else None
    m = re.search(r'^HF_SOURCE:\s*(\S+)', txt, re.M)
    r['hf_source'] = m.group(1) if m else None
    m = re.search(r'^DATA:.*?classes=(\d+)\s+train_images=(\d+)\s+val_images=(\d+)', txt, re.M)
    r['data'] = ({'classes': int(m.group(1)), 'train_images': int(m.group(2)),
                  'val_images': int(m.group(3))} if m else None)
    m = re.search(r'^EPOCH_DEF:.*?short_epoch=(\w+)', txt, re.M)
    r['short_epoch'] = (m.group(1) == 'True') if m else None

    r['arm'] = eff.get('stepsize-groups')
    r['seed'] = eff.get('seed')
    for e, tr, te in re.findall(
            r'^Epoch (\d+), Train Accuracy: ([0-9.]+) %, Test Accuracy: ([0-9.]+) %',
            txt, re.M):
        r['epochs'].append((int(e), float(tr), float(te)))
    r['epochs'].sort()
    r['n_ep'] = len(r['epochs'])
    r['done'] = 'RUN_DONE' in txt
    r['stopped'] = bool(re.search(r'^STOP: wall budget reached', txt, re.M))
    return r


def w5(run, estar):
    """Mean test accuracy over the last 5 epochs at or before estar."""
    te = [t for (e, _, t) in run['epochs'] if e < estar]
    return st.mean(te[-5:]) if len(te) >= 5 else None


# ---- gates ------------------------------------------------------------------
def apply_gates(runs, quiet=False):
    out = []
    for r in runs:
        if r['arm'] not in ARMS:
            r['void'].append('UNKNOWN_ARM %r' % r['arm'])
        if r.get('short_epoch') is not False:
            r['void'].append('G5_SHORT_EPOCH short_epoch=%s' % r.get('short_epoch'))
        if r.get('m') is None:
            r['void'].append('G2_NO_GROUPS_LINE')
        elif r['arm'] in EXPECT_M and r['m'] != EXPECT_M[r['arm']]:
            r['void'].append('G2_M_MISMATCH arm=%s m=%s expected=%s'
                             % (r['arm'], r['m'], EXPECT_M[r['arm']]))
        hs = r.get('hf_source') or ''
        if 'cifar10' not in hs or 'Optimizers/HF.py' not in hs:
            r['void'].append('G3_HF_PROVENANCE %r' % hs)
        d = r.get('data')
        if d != EXPECT_DATA:
            r['void'].append('G4_DATA %r' % (d,))
        if r['env'].get('DATASET') != 'imagenet489':
            r['void'].append('G4_DATASET %r' % r['env'].get('DATASET'))
        out.append(r)

    live = [r for r in out if not r['void']]
    # G4 design consistency across the surviving runs.
    incon = []
    if live:
        allflags = set()
        for r in live:
            allflags |= set(r['args'])
        for f in sorted(allflags - AXIS_FLAGS):
            vals = {r['args'].get(f) for r in live}
            if len(vals) > 1:
                incon.append((f, sorted(str(v) for v in vals)))
        for k in ('BETA_CLIP', 'AUGMENT', 'HIER'):
            vals = {r['env'].get(k) for r in live}
            if len(vals) > 1:
                incon.append(('ENV:' + k, sorted(str(v) for v in vals)))
        for a in ARMS:
            ms = {r['m'] for r in live if r['arm'] == a}
            if len(ms) > 1:
                incon.append(('m[%s]' % a, sorted(str(v) for v in ms)))
    return out, incon


# ---- contrast ---------------------------------------------------------------
def contrast(M, S, a, b):
    """(diff, t) for arm a minus arm b, t from THIS batch's own seed spread."""
    if a not in M or b not in M:
        return None, None
    d = M[a] - M[b]
    na, nb = len(S[a]), len(S[b])
    if na < 2 and nb < 2:
        return d, None
    var = 0.0
    df = 0
    for arr in (S[a], S[b]):
        if len(arr) >= 2:
            mu = st.mean(arr)
            var += sum((x - mu) ** 2 for x in arr)
            df += len(arr) - 1
    if df == 0:
        return d, None
    sp2 = var / df
    se = math.sqrt(sp2 * (1.0 / na + 1.0 / nb))
    return d, (d / se if se > 0 else None)


def band(d, t, bar, nullband):
    if d is None:
        return 'UNRESOLVED'
    if d >= bar and t is not None and t >= T_BAR:
        return 'REPLICATES'
    if abs(d) <= nullband:
        return 'VANISHES'
    if d <= -nullband and t is not None and t <= -T_BAR:
        return 'REVERSES'
    return 'UNRESOLVED'


def node_band(d, t):
    if d is None:
        return 'UNRESOLVED'
    if d >= BAR_NODE and t is not None and t >= T_BAR:
        return 'FINER_HELPS'
    if d <= -BAR_NODE and t is not None and t <= -T_BAR:
        return 'FINER_HURTS'
    if abs(d) <= NODE_NULL_BAND:
        return 'NODE_NULL'
    return 'UNRESOLVED'


# ---- report -----------------------------------------------------------------
def score(paths, quiet=False):
    runs = [parse_out(p) for p in paths]
    runs, incon = apply_gates(runs, quiet)
    live = [r for r in runs if not r['void']]

    print('=' * 78)
    print('cI1_in489g1_score.py -- ImageNet-489 granularity ladder')
    print('  ImageNet-489: 489 of 1000 classes, 627,329 train, 24,450 val')
    print('  (50/class, labels RECOVERED FROM PER-IMAGE XML ANNOTATIONS).')
    print('  chance top-1 = 100/489 = %.4f%%.  NOT comparable to ImageNet-1k.' % CHANCE)
    print('=' * 78)

    print('\nG0 PROVENANCE -- %d file(s), %d live, %d void' % (len(runs), len(live), len(runs) - len(live)))
    for r in sorted(runs, key=lambda r: r['name']):
        tag = 'LIVE ' if not r['void'] else 'VOID '
        print('  %s%-42s arm=%-16s seed=%-3s m=%-6s ep=%-4s%s'
              % (tag, r['name'], r.get('arm'), r.get('seed'), r.get('m'),
                 r.get('n_ep'), '' if not r['void'] else '  <- ' + '; '.join(r['void'])))
    if incon:
        print('\nG4 REFUSES -- non-axis flags differ across the batch:')
        for f, v in incon:
            print('    %-28s %s' % (f, v))
        print('\nVERDICT: UNRESOLVED (G4 design inconsistency; RULE 20)')
        return 'UNRESOLVED'
    if live:
        print('\nG4 PASS -- every non-axis flag identical across the %d live runs.' % len(live))

    if not live:
        print('\nVERDICT: UNRESOLVED (no live run)')
        return 'UNRESOLVED'

    estar = min(r['n_ep'] for r in live)
    print('\nCOMMON HORIZON  ESTAR = %d  (per-run scored epochs: %s)'
          % (estar, ', '.join('%s=%d' % (ARM_SHORT.get(r['arm'], r['arm']) + r['seed'], r['n_ep'])
                              for r in sorted(live, key=lambda r: (r['arm'], r['seed'])))))
    if estar < ESTAR_MIN:
        print('  ESTAR < %d -> no verdict at any bar.' % ESTAR_MIN)
        print('\nVERDICT: UNRESOLVED_INSUFFICIENT_BUDGET')
        return 'UNRESOLVED_INSUFFICIENT_BUDGET'
    short = estar < ESTAR_FULL
    label = 'window5@%d' % estar if short else 'plateau5'
    if short:
        print('  %d <= ESTAR < %d -> verdict STAMPED short_budget; the primary'
              % (ESTAR_MIN, ESTAR_FULL))
        print('  statistic is "%s" and MUST NOT be called a plateau.' % label)

    # G1 floor
    S = {}
    for r in live:
        v = w5(r, estar)
        if v is None:
            r['void'].append('NO_W5')
            continue
        if v < FLOOR_W5:
            r['void'].append('G1_NOT_LEARNED w5=%.2f < %.1f' % (v, FLOOR_W5))
            continue
        S.setdefault(r['arm'], []).append(v)
    live = [r for r in live if not r['void']]
    if not live:
        print('\nVERDICT: UNRESOLVED (every run below the G1 chance floor)')
        return 'UNRESOLVED'

    M = {a: st.mean(v) for a, v in S.items()}
    print('\nARM MEANS (%s, pp -- PRIMARY SCALE)' % label)
    print('  %-16s %-4s %-8s %-8s %-9s %s' % ('arm', 'n', label, 'sd', 'CIFAR', 'delta_vs_CIFAR'))
    ref = {'scalar': CIFAR_SCALAR, 'resnet18_blocks': CIFAR_BLOCK,
           'layerwise': CIFAR_LAY, 'nodewise': CIFAR_NODE}
    for a in ARMS:
        if a not in M:
            print('  %-16s MISSING' % a)
            continue
        sd = st.pstdev(S[a]) if len(S[a]) > 1 else float('nan')
        print('  %-16s %-4d %-8.3f %-8.3f %-9.3f %+.3f'
              % (a, len(S[a]), M[a], sd, ref[a], M[a] - ref[a]))

    if 'scalar' in M:
        print('\nSECONDARY SCALE (error-rate ratio vs scalar; NEVER overrides pp)')
        for a in ARMS:
            if a in M and a != 'scalar':
                den = 100.0 - M['scalar']
                print('  R(%-16s) = %.4f' % (a, (100.0 - M[a]) / den if den else float('nan')))

    dB, tB = contrast(M, S, 'resnet18_blocks', 'scalar')
    dG, tG = contrast(M, S, 'layerwise', 'scalar')
    dN, tN = contrast(M, S, 'nodewise', 'layerwise')
    vB = band(dB, tB, BAR_BLOCK, NULL_BAND)
    vG = band(dG, tG, BAR_GRAN, NULL_BAND)
    vN = node_band(dN, tN)

    def fmt(d, t):
        return ('%+.3f pp' % d if d is not None else '   n/a') + \
               ('  t=%+.2f' % t if t is not None else '  t=n/a')

    print('\nCONTRASTS (pp; t from THIS batch\'s own seed spread)')
    print('  PRIMARY-A  BLOCK = blk  - scal  %s   bar %+.3f   CIFAR %+.4f  -> %s'
          % (fmt(dB, tB), BAR_BLOCK, BLOCK_CIFAR, vB))
    print('  PRIMARY-B  GRAN  = lay  - scal  %s   bar %+.3f   CIFAR %+.4f  -> %s'
          % (fmt(dG, tG), BAR_GRAN, GRAN_CIFAR, vG))
    print('  SECONDARY  NODE  = node - lay   %s   bar %+.3f   CIFAR %+.4f  -> %s'
          % (fmt(dN, tN), BAR_NODE, NODE_CIFAR, vN))

    # Registered budget-inversion guard: the whole curve, never just the endpoint.
    print('\nEPOCH-WISE CONTRAST CURVE (registered: CORRECTIONS 1 shows this')
    print('ordering INVERTS with budget on CIFAR-10, so the endpoint alone is')
    print('not reportable.)')
    print('  %-6s %-9s %-9s %-9s %-9s' % ('epoch', 'scal', 'blk-scal', 'lay-scal', 'node-lay'))
    marks = [e for e in (10, 20, 30, 40, 50, 60, 70, 84) if e <= estar]
    if estar not in marks:
        marks.append(estar)
    for e in marks:
        cur = {}
        for a in ARMS:
            vs = [w5(r, e) for r in live if r['arm'] == a]
            vs = [v for v in vs if v is not None]
            if vs:
                cur[a] = st.mean(vs)
        if 'scalar' not in cur:
            continue
        f = lambda x, y: ('%+9.3f' % (cur[x] - cur[y])) if x in cur and y in cur else '      n/a'
        print('  %-6d %-9.3f %s %s %s'
              % (e, cur['scalar'], f('resnet18_blocks', 'scalar'),
                 f('layerwise', 'scalar'), f('nodewise', 'layerwise')))

    fam = lambda v: ('R' if v == 'REPLICATES' else
                     'N' if v in ('VANISHES', 'REVERSES') else 'U')
    fB, fG = fam(vB), fam(vG)
    if fB == 'U' or fG == 'U':
        verdict = 'UNRESOLVED'
    elif fB == 'R' and fG == 'R':
        verdict = 'ANOMALY_ABSENT'
    elif fB == 'N' and fG == 'N':
        verdict = 'CORROBORATES_PARENT'
    else:
        verdict = 'SPLIT'
    if short and verdict != 'UNRESOLVED':
        verdict += '_SHORT_BUDGET'

    print('\n' + '=' * 78)
    print('VERDICT: %s   (BLOCK %s / GRAN %s / NODE %s)' % (verdict, vB, vG, vN))
    if verdict.startswith('CORROBORATES_PARENT'):
        print('  Finer step-size granularity does NOT beat a scalar step size on')
        print('  ImageNet-489.  This reproduces the parent paper\'s ImageNet report')
        print('  at full resolution for the first time in this project, and it is a')
        print('  POSITIVE finding: the hierarchical-pooling idea now has the')
        print('  motivating phenomenon it was invented to explain.')
    elif verdict.startswith('ANOMALY_ABSENT'):
        print('  Finer granularity STILL beats scalar at 224px and 489 classes.')
        print('  The parent\'s ImageNet anomaly does NOT survive here, so it is not')
        print('  explained by resolution or by class count in the 10 -> 489 range,')
        print('  and the premise of the hierarchical programme is reframed.')
    elif verdict.startswith('SPLIT'):
        print('  The two primaries disagree.  Report BOTH; claim NEITHER pole.')
    else:
        print('  No verdict.  Report the arm means and the curve; claim nothing.')
    print('  SCOPE, mandatory with any number above: ImageNet-489, NOT ImageNet.')
    print('  489/1000 classes; val 24,450 imgs @ 50/class, labels recovered from')
    print('  per-image XML annotations, not a devkit.  chance %.4f%%.' % CHANCE)
    print('=' * 78)
    return verdict


# ---- selftest ---------------------------------------------------------------
def selftest():
    """RE-DERIVE every frozen constant from results/all_runs.csv.  Never quote."""
    n_ok = n_tot = 0

    def chk(label, got, want, tol=0.0005):
        nonlocal n_ok, n_tot
        n_tot += 1
        ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
        n_ok += ok
        print('  %-4s %-52s got=%s want=%s' % ('PASS' if ok else 'FAIL', label, got, want))

    path = os.path.join(REPO, 'results', 'all_runs.csv')
    rows = list(csv.DictReader(open(path)))
    print('selftest: %s, %d rows' % (path, len(rows)))

    def matched(r):
        return (r['window_ok'] == '1' and r['complete'] == '1'
                and r['superseded'] not in ('1', 'True', 'true')
                and r['hier'] in ('', 'none', 'na')
                and r['dataset'] == 'CIFAR10' and r['network'] == 'ResNet18'
                and r['base'] == 'SGDm' and r['meta'] == 'Lion'
                and r['meta_stepsize'] == '1e-3' and r['alpha0'] == '1e-6'
                and r['beta_clip'] == '-15:-2.3026' and r['augment'] == '1'
                and r['gamma'] == '1' and r['batch_size'] == '100'
                and r['epochs_requested'] == '100' and r['collapsed'] == '0')

    sel = [r for r in rows if matched(r)]
    by = {}
    for r in sel:
        by.setdefault(r['granularity'], []).append(float(r['plateau5']))

    chk('matched-cell row count', len(sel), 53)
    chk('CIFAR scalar', round(st.mean(by['scalar']), 3), CIFAR_SCALAR)
    chk('CIFAR resnet18_blocks', round(st.mean(by['resnet18_blocks']), 3), CIFAR_BLOCK)
    chk('CIFAR layerwise', round(st.mean(by['layerwise']), 3), CIFAR_LAY)
    chk('CIFAR nodewise', round(st.mean(by['nodewise']), 3), CIFAR_NODE)
    chk('BLOCK_CIFAR = blk - scal',
        round(st.mean(by['resnet18_blocks']) - st.mean(by['scalar']), 4), BLOCK_CIFAR)
    chk('GRAN_CIFAR = lay - scal',
        round(st.mean(by['layerwise']) - st.mean(by['scalar']), 4), GRAN_CIFAR)
    chk('NODE_CIFAR = node - lay',
        round(st.mean(by['nodewise']) - st.mean(by['layerwise']), 4), NODE_CIFAR)

    # pooled within-cell SD over (batch, granularity) cells
    cells = {}
    for r in sel:
        b = re.split(r'[-_]', r['run'])[0]
        cells.setdefault((b, r['granularity']), []).append(float(r['plateau5']))
    ss = df = 0
    for v in cells.values():
        if len(v) >= 2:
            mu = st.mean(v)
            ss += sum((x - mu) ** 2 for x in v)
            df += len(v) - 1
    chk('pooled within-cell SD', round(math.sqrt(ss / df), 4), POOLED_SD)

    # bars are exactly half the archived lifts
    chk('BAR_BLOCK == BLOCK_CIFAR/2', round(BLOCK_CIFAR / 2, 3), BAR_BLOCK, 0.001)
    chk('BAR_GRAN  == GRAN_CIFAR/2', round(GRAN_CIFAR / 2, 3), BAR_GRAN, 0.001)
    chk('BAR_NODE  == NODE_CIFAR/2', round(NODE_CIFAR / 2, 3), BAR_NODE, 0.001)

    # `bo-lion` is excluded because base=Lion -- assert that, don't assume it.
    bol = [r for r in rows if r['run'].startswith('bo-lion')]
    chk('bo-lion rows exist', len(bol) > 0, True)
    chk('bo-lion base is Lion (why excluded)',
        sorted({r['base'] for r in bol}), ['Lion'])
    chk('bo-lion excluded from matched cell',
        sum(1 for r in sel if r['run'].startswith('bo-lion')), 0)

    # RULE 21: no imagenet489 row may exist at registration time.
    in489 = [r for r in rows if 'imagenet489' in r['dataset'].lower()
             or r['run'].startswith('in489g1')]
    chk('RULE 21: zero imagenet489 rows at registration', len(in489), 0)

    # chance floor arithmetic
    chk('chance = 100/489', round(CHANCE, 4), 0.2045)
    chk('G1 floor is >= 20x chance', FLOOR_W5 >= 20 * CHANCE, True)

    # band logic is total and behaves as registered
    chk('band: at bar with t=3 -> REPLICATES',
        band(BAR_BLOCK, 3.0, BAR_BLOCK, NULL_BAND), 'REPLICATES')
    chk('band: at bar with t=1 -> UNRESOLVED (underpowered)',
        band(BAR_BLOCK, 1.0, BAR_BLOCK, NULL_BAND), 'UNRESOLVED')
    chk('band: 0.0 -> VANISHES (the parent\'s own claim)',
        band(0.0, 0.0, BAR_BLOCK, NULL_BAND), 'VANISHES')
    chk('band: -2.0 t=-3 -> REVERSES',
        band(-2.0, -3.0, BAR_BLOCK, NULL_BAND), 'REVERSES')
    chk('band: +1.0 (between null band and bar) -> UNRESOLVED',
        band(1.0, 9.0, BAR_BLOCK, NULL_BAND), 'UNRESOLVED')
    chk('band: -2.0 t=-1 -> UNRESOLVED (sign but no power)',
        band(-2.0, -1.0, BAR_BLOCK, NULL_BAND), 'UNRESOLVED')
    chk('node_band: 0.0 -> NODE_NULL', node_band(0.0, 0.0), 'NODE_NULL')
    chk('node_band: +0.5 t=3 -> FINER_HELPS', node_band(0.5, 3.0), 'FINER_HELPS')
    chk('node_band: -0.5 t=-3 -> FINER_HURTS', node_band(-0.5, -3.0), 'FINER_HURTS')

    # the archived CIFAR lift must itself clear the bar it sets (sanity)
    chk('CIFAR BLOCK lift clears BAR_BLOCK', BLOCK_CIFAR >= BAR_BLOCK, True)
    chk('CIFAR GRAN lift clears BAR_GRAN', GRAN_CIFAR >= BAR_GRAN, True)

    # horizon bands are ordered and non-overlapping
    chk('ESTAR_MIN < ESTAR_FULL', ESTAR_MIN < ESTAR_FULL, True)

    # RULE 20 tokeniser really catches a repeated flag
    _, order = _tokenise('--seed 1 --stepsize-groups scalar --seed 2')
    chk('tokeniser detects a repeated flag',
        sorted({f for f in order if order.count(f) > 1}), ['seed'])

    # m expectations, as measured live in Phase 1 on the 489-class tree
    chk('EXPECT_M ordering is strictly increasing in granularity',
        [EXPECT_M[a] for a in ARMS] == sorted(EXPECT_M[a] for a in ARMS), True)

    print('\nselftest: %d/%d PASS' % (n_ok, n_tot))
    return 0 if n_ok == n_tot else 1


def main():
    ap = argparse.ArgumentParser(
        description='REGISTERED scorer for in489g1 (RULE 21). Run UNEDITED (RULE 16).')
    ap.add_argument('--runs', default=None,
                    help='directory or glob holding the batch .out files')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.runs:
        ap.error('--runs is required (this scorer cannot find the raw records on its own)')
    pats = a.runs if any(c in a.runs for c in '*?[') else os.path.join(a.runs, 'in489g1-*.out')
    paths = sorted(glob.glob(pats))
    if not paths:
        print('no .out files matched %r' % pats)
        return 2
    v = score(paths, a.quiet)
    return 0 if v and not v.startswith('UNRESOLVED') else 1


if __name__ == '__main__':
    sys.exit(main())
