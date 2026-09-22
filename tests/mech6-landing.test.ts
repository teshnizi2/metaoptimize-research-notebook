import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 237 (cwd4, campaign commit 26baf4a, CORRECTIONS 283) and 238 (cwd5, campaign commit 8554afa,
// CORRECTIONS 285), imported together as MT237 and MT238 through scripts/register_model.py MECH6_ROWS. Each step edited
// header line 3 and appended one row; neither amended any earlier row, and neither touched line 5.
//
// Both records are MIXED, and neither is Goal met by default. The MT218-MT236 precedent grants Goal met only when the
// fired account returns its registered branch AND its registered state word in the registered direction, no registered
// account misses a band, and no control fails. MT237 answers its question and refutes the count/dose rival, but its own
// fired account predicted the non-sufficing carrier at the anchor's floor and it landed 30 points above it, and the
// second branch token turns on a 0.40 SE margin the row itself calls descriptive -- MT231's two reasons for Mixed.
// MT238 hits every registered band on the ladder and no control fails, but the batch's second registered question, the
// carrier companion, is unanswerable because its rung did not collapse -- MT228's ISO-UNREADABLE one rung higher.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
const cwd4 = byId.get('MT237')!;
const cwd5 = byId.get('MT238')!;

const CWD4_FINAL = ['ONE-SUFFICES-PARTIAL', 'TWO-REC+CTL2-NULL+ONE50-REC+ONE53-PART+ONE59-REC', 'HARNESS-CLEAN', 'PATCH-BITES',
  'MASK-UPDATE-AND-TRACE', 'CONV-LINEAR-WD-KEPT', 'ALL-ARMS-SCALAR', 'RECOVERY-AGAINST-CARWD0', 'CTL2-CLASS-PURE-COUNT-MATCHED',
  'CTL-NOT-MAGNITUDE-MATCHED', 'MAGNITUDE-NOT-SEPARATED', 'POSITION-CLASS-NOT-SEPARATED', 'ONE-NETWORK-RESNET18', 'ONE-CELL',
  'HORIZON-100-ONLY', 'SIGMA-PRIOR-FROZEN', 'TWO-SPECIFIC', 'ONE-ORDER-ONE59-ONE50-ONE53', 'FLOOR-READINGS-ARE-BOUNDS',
  'HW-UNIFORM-NVIDIA_L4', 'TRAIN-AGREES (two branch tokens + 19 stamps)'];
const CWD5_FINAL = ['THRESHOLD-W1-W2', 'W1-COLLAPSE+W2-NOGAP+W3-NOGAP+W4-NOGAP+CAR-UNREADABLE', 'HARNESS-CLEAN', 'LADDER-BITES',
  'ALL-RUNGS-IN-BATCH', 'BOTH-GRAINS-AT-EVERY-RUNG', 'REFERENCE-IS-IN-BATCH-LAYERWISE', 'CARRIER-AGAINST-KLW2', 'COUPLED-DECAY-ONLY',
  'DECOUPLED-NOT-TESTED', 'LADDER-IS-FOUR-POINTS', 'ONE-NETWORK-RESNET18', 'ONE-CELL-OTHERWISE', 'HORIZON-100-ONLY', 'THREE-SEEDS',
  'WD-IS-AN-ARGS-DEVIATION', 'FLOOR-READINGS-ARE-BOUNDS', 'SIGMA-PRIOR-FROZEN', 'MASK-UPDATE-AND-TRACE', 'HW-UNIFORM-NVIDIA_L4',
  'TRAIN-AGREES (two branch tokens + 19 stamps)'];

test('MT237 and MT238 carry the outcome the documented rules give them, with every FINAL token quoted', () => {
  assert.equal(CWD4_FINAL.length, 21);
  assert.equal(CWD5_FINAL.length, 21);
  assert.deepEqual(data.experiments.slice(-8, -6).map(e => e.id), ['MT237', 'MT238'], 'appended in line order, before the later caw2, cgw1, crt1, csh1, cvl1 and g3b rows');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [181, 163, 18, 3501]);
  // Two research questions, both Mixed; neither corrects an earlier published claim, so neither carries a Corrected badge.
  assert.deepEqual([cwd4.section, cwd4.area, cwd4.kind, cwd4.outcome, cwd4.corrected, cwd4.batches],
    [9, 'Mechanism and isolation', 'research', 'mixed', false, ['cwd4']]);
  assert.deepEqual([cwd5.section, cwd5.area, cwd5.kind, cwd5.outcome, cwd5.corrected, cwd5.batches],
    [9, 'Mechanism and isolation', 'research', 'mixed', false, ['cwd5']]);
  // Each FINAL carries TWO branch tokens, so the reason repeats both after the rule.
  assert.ok(cwd4.reason.startsWith(`Verdict: ${CWD4_FINAL.join(' + ')} Mixed: ONE-SUFFICES-PARTIAL + TWO-REC+CTL2-NULL+ONE50-REC+ONE53-PART+ONE59-REC: `), cwd4.reason);
  assert.ok(cwd5.reason.startsWith(`Verdict: ${CWD5_FINAL.join(' + ')} Mixed: THRESHOLD-W1-W2 + W1-COLLAPSE+W2-NOGAP+W3-NOGAP+W4-NOGAP+CAR-UNREADABLE: `), cwd5.reason);
  assert.deepEqual([cwd4.figureIds, cwd4.eventIds], [['page-19'], ['phase-13']]);
  assert.deepEqual([cwd5.figureIds, cwd5.eventIds], [['page-19'], ['phase-14']]);
  assert.ok(cwd4.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 237), 'MT237 anchors MASTER-TABLE line 237');
  assert.ok(cwd5.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 238), 'MT238 anchors MASTER-TABLE line 238');
  // cwd4 owns patch-intervention rows, cwd5 owns ARGS-value rows, and neither amended a row anywhere.
  assert.deepEqual(cwd4.warningIds, ['warning-MT237-verdict', 'warning-MT237-intervention']);
  assert.deepEqual(cwd5.warningIds, ['warning-MT238-verdict', 'warning-MT238-args-deviation']);
  assert.ok(!data.warnings.some(w => w.id.endsWith('-amendment-283') || w.id.endsWith('-amendment-285')), 'neither landing amended a row');
  for (const record of [cwd4, cwd5]) {
    assert.ok(!data.warnings.some(w => w.detail.includes('Referenced source not available') && w.experimentId === record.id), `${record.id} links every source it cites`);
  }
});

test('MT237 is Mixed because its own account missed a band and a token turns on a near bar', () => {
  // The question IS answered at matched count two, and the count/dose rival is refuted.
  assert.match(cwd4.result, /CO-PRIMARY P_2SPEC = TWOWD0 - CTL2WD0 = 67\.0713 - 22\.8547 = \+44\.2167 pp = \+83\.94 SE/);
  assert.match(cwd4.reason, /278's COUNT \/ DOSE rival is REFUTED at this cell/);
  assert.match(cwd4.reason, /WRITEUP-mechanism's O-12 closes/);
  // The registered level band that was missed, named as the reason the row is not Goal met.
  assert.match(cwd4.reason, /BUT A REGISTERED EXPECTATION IS DEFIED: the fired account's own registered LEVEL BAND for ONE53 is K01 \(18-28\), and ONE53 landed at 58\.5780/);
  assert.match(cwd4.reason, /a BRANCH and STATE-WORD match whose level prediction for the non-sufficing carrier is missed by about 30 pp/);
  // The near bar the second branch token rests on, which the row itself calls descriptive.
  assert.match(cwd4.reason, /ONE50's REC clears the in-batch 65\.0100 bar by \+0\.2133 pp = \+0\.40 SE/);
  assert.match(cwd4.scope, /ONE50's REC IS A NEAR-BAR READING/);
  assert.match(cwd4.scope, /DESCRIPTIVE \/ UNSURE/);
  // The floor reading is a bound, never a measured zero, and the forbidden wording is named.
  assert.match(cwd4.result, /P_CTL2 = \+0\.0333 pp = \+0\.06 SE, \+\/-2 SE \[-1\.0202, \+1\.0868\] SPANNING ZERO -- NULL, a FLOOR LOCATION and a BOUND, never a measured zero/);
  assert.match(cwd4.scope, /the licensed form is "leaves the run at k01's floor"/);
  // The magnitude rival is not refuted and this batch strengthens it; the ladder is not cleanly monotone.
  assert.match(cwd4.scope, /THE MAGNITUDE RIVAL IS NOT REFUTED AND THE DATA NOW MAKE IT MORE ATTRACTIVE, NOT LESS/);
  assert.match(cwd4.scope, /140\.1x in batch/);
  assert.match(cwd4.scope, /THE DOSE LADDER IS NOT CLEANLY MONOTONE AT THE POINT ESTIMATES/);
  assert.match(cwd4.scope, /THE POSITION-CLASS CONFOUND NOW BITES AT COUNT ONE/);
  assert.match(cwd4.scope, /One RULE 16 defect REPORTED and NOT fixed/);
});

test('MT238 is Mixed because its carrier half is unreadable, and the ladder only brackets', () => {
  // The co-primary the batch existed to produce, and the adverse answer.
  assert.match(cwd5.result, /CO-PRIMARY G_W4 = kLW4 - k01W4 = 67\.8813 - 72\.4080 = -4\.5267 pp = -8\.59 SE/);
  assert.match(cwd5.reason, /PRESENT at the campaign's coupled weight decay 0\.1 and ABSENT at 1e-2, 1e-3 and 5e-4/);
  assert.match(cwd5.reason, /every registered LEVEL BAND is hit/);
  assert.match(cwd5.reason, /No control fails/);
  // The unanswerable half, and the precedent it repeats.
  assert.match(cwd5.reason, /BUT THE BATCH'S SECOND REGISTERED QUESTION IS UNANSWERABLE AT THIS CELL/);
  assert.match(cwd5.reason, /That is MT228's ISO-UNREADABLE at weight decay 0 repeating one rung higher/);
  assert.match(cwd5.result, /CARW2 71\.8327 \(a LEVEL ONLY -- see the bounds\)/);
  assert.match(cwd5.scope, /CARW2 IS UNREADABLE AND IS READ AS NOTHING/);
  assert.match(cwd5.scope, /a design-scope limit, not a patch failure/);
  // Four rungs bracket and never locate; no threshold value may be named anywhere.
  assert.match(cwd5.scope, /THE LADDER BRACKETS A TRANSITION; IT CANNOT LOCATE ONE/);
  assert.match(cwd5.scope, /no finer statement is licensed/);
  assert.match(cwd5.reason, /nothing inside that decade may be named/);
  // Every reading is a bound in one direction or the other, and W2 is not resolved.
  assert.match(cwd5.scope, /EVERY COLLAPSE AND NOGAP READING IS A BOUND, NEVER A POINT EFFECT/);
  assert.match(cwd5.scope, /W2 IS NOT RESOLVED AND MUST NOT BE POOLED WITH W3 AND W4/);
  assert.match(cwd5.scope, /IT IS NOT A STATEMENT ABOUT GROUPING ALONE OR DECAY ALONE/);
  // The consequence, conceded in the record itself, and what it does not do.
  assert.match(cwd5.reason, /this is the area chair's CORNER-CASE charge LANDING/);
  assert.match(cwd5.reason, /nothing is retracted and only the scope moves from assumed to measured/);
  assert.match(cwd5.scope, /DECOUPLED weight decay is NOT TESTED in either direction/);
});

test("the cwd4 intervention warning names the six masked arms and the single exclusion kind", () => {
  const warning = data.warnings.find(w => w.id === 'warning-MT237-intervention')!;
  assert.equal(warning.title, "cwd4's six masked arms run with the coupled weight decay switched off on a named set of BatchNorm scales");
  assert.deepEqual([warning.severity, warning.status], ['caution', 'documented']);
  assert.match(warning.detail, /CARWD0, TWOWD0, CTL2WD0, ONE50, ONE53 and ONE59 ran PATCH_DECAYMASK/);
  assert.match(warning.detail, /in the weight update AND in the meta trace/);
  assert.match(warning.detail, /The batch needed NO new harness code/);
  assert.match(warning.detail, /one kind only, no run of this batch carries a second/);
  assert.match(warning.detail, /The three k01 runs print DECAY_MASK: off/);
  assert.match(warning.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 8b9fbd2; CORRECTIONS 269, 280 and 283\.$/);
});

test('the cwd5 warning names the ARGS-value kind and the first two-axis rows in the corpus', () => {
  const warning = data.warnings.find(w => w.id === 'warning-MT238-args-deviation')!;
  assert.deepEqual([warning.severity, warning.status], ['caution', 'documented']);
  assert.match(warning.detail, /cwd5 varies ONE axis, --weight-decay-base, across four rungs/);
  assert.match(warning.detail, /There is no PATCH ON line to read for them, because no patch ran/);
  assert.match(warning.detail, /The three CARW2 runs are different and are the corpus's FIRST TWO-AXIS rows/);
  assert.match(warning.detail, /The six anchor runs at the standard 0\.1 deviate on nothing and own no row/);
  assert.match(warning.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 91fcd57; CORRECTIONS 263, 281, 284 and 285\.$/);
  // At this landing exactly two records carried the ARGS-value mark, cmo1's and cwd5's; caw2 and cgw1 later added theirs.
  assert.deepEqual(data.warnings.filter(w => w.id.endsWith('-args-deviation')).map(w => w.id), ['warning-MT228-args-deviation', 'warning-MT238-args-deviation', 'warning-MT239-args-deviation', 'warning-MT240-args-deviation', 'warning-MT241-args-deviation', 'warning-MT242-args-deviation', 'warning-MT243-args-deviation', 'warning-MT244-args-deviation']);
});

test('the 48 runs link to their records with sanitized logs, and exactly the listed rows are marked', () => {
  assert.equal(runs.length, 3501);
  for (const [eid, batch, seeds, marked] of [['MT237', 'cwd4', ['143', '144', '145'], 18], ['MT238', 'cwd5', ['146', '147', '148'], 21]] as const) {
    const linked = runs.filter(run => run.batch === batch);
    assert.equal(linked.length, batch === 'cwd4' ? 21 : 27);
    assert.deepEqual([...byId.get(eid)!.runIds].sort(), linked.map(run => run.id).sort());
    for (const run of linked) {
      assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs],
        [[eid], 'Account2', 'completed', 'ResNet18_c100', 'CIFAR100', 100], run.id);
      assert.ok(seeds.includes(run.seed as (typeof seeds)[number]), run.id);
      const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
      const lines = log.split('\n');
      assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
      assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
      assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
      assert.equal(lines.filter(line => /^Epoch \d+,/.test(line)).length, 100, `${run.id} keeps every epoch line`);
      if (batch === 'cwd4') {
        assert.equal(run.parameters.argsDeviation, undefined, 'no cwd4 run deviates on an ARGS value');
        if (run.parameters.intervention) {
          assert.ok(lines.includes(run.parameters.interventionWitness!), `${run.id} log carries its witness line`);
          assert.equal(run.parameters.interventionAdditionalWitness, undefined, 'every cwd4 run carries exactly ONE kind');
        } else {
          assert.match(run.parameters.runLabel, /^cwd4-k01-s14[345]$/, 'only the three k01 runs are unmarked');
        }
      } else {
        assert.equal(run.parameters.intervention, undefined, 'no cwd5 run is marked as a plain patch intervention');
        if (run.parameters.argsDeviation) {
          // The published witness quotes the checked flag alone; the sanitized log still carries the whole ARGS line.
          const args = lines.filter(line => line.startsWith('ARGS:'));
          assert.equal(args.length, 1, `${run.id} prints exactly one ARGS line`);
          const value = run.parameters.argsDeviationWitness!.split('=')[1];
          assert.ok(args[0].includes(`--weight-decay-base ${value}`), `${run.id} log carries the listed flag value`);
          assert.equal(run.parameters.argsDeviationArgsWitness, `ARGS: --weight-decay-base ${value}`, run.id);
          assert.match(value, /^(1e-2|1e-3|5e-4)$/, run.id);
        } else {
          assert.match(run.parameters.runLabel, /^cwd5-(k01|kL)W1-s14[678]$/, 'only the six anchor runs at the standard 0.1 are unmarked');
        }
      }
    }
    assert.equal(linked.filter(run => run.parameters.intervention || run.parameters.argsDeviation).length, marked);
  }
  assert.equal(runs.filter(run => run.batch !== 'cvl1' && run.parameters.intervention).length, 183, 'the 165 earlier patch interventions plus cwd3 and cwd4');
  assert.equal(runs.filter(run => !['caw2', 'cgw1', 'crt1', 'csh1', 'cvl1', 'g3b'].includes(run.batch) && run.parameters.argsDeviation).length, 39, "cmo1's 18 plus cwd5's 21");
  // caw2 and cgw1 later added 18 and 28 of their own (tests/mech7-landing.test.ts), crt1 and csh1 36 and 18 (tests/mech8-landing.test.ts).
  assert.equal(runs.filter(run => !['cvl1', 'g3b'].includes(run.batch) && run.parameters.argsDeviation).length, 139);
});

test("cwd5's CARW2 rows are the corpus's first two-axis rows and are witnessed on both axes", () => {
  // cvl1's 16 W4 runs (VAL_SPLIT, CORRECTIONS 312) are the only later two-axis rows (tests/mech9-landing.test.ts).
  const twoAxis = runs.filter(run => run.batch !== 'cvl1' && run.parameters.argsDeviationAxes);
  assert.deepEqual(twoAxis.map(run => run.parameters.runLabel), ['cwd5-CARW2-s146', 'cwd5-CARW2-s147', 'cwd5-CARW2-s148']);
  for (const run of twoAxis) {
    assert.equal(run.parameters.argsDeviation,
      'CARW2: --weight-decay-base 1e-2 + DECAY_MASK=layer4.0.bn2.weight+layer4.0.shortcut.1.weight+layer4.1.bn2.weight');
    assert.equal(run.parameters.argsDeviationAxes, 'DECAY_MASK=layer4.0.bn2.weight+layer4.0.shortcut.1.weight+layer4.1.bn2.weight');
    assert.equal(run.parameters.argsDeviationWitness, 'ARGS_WD_BASE: weight-decay-base=1e-2');
    assert.equal(run.parameters.argsDeviationArgsWitness, 'ARGS: --weight-decay-base 1e-2');
    assert.match(run.parameters.argsDeviationNote!, /TWO-AXIS: this run ALSO ran the coupled weight-decay mask intervention/);
    // The patch axis is read back from the run's OWN log, at the rung's decay rather than the standard one.
    const witness = run.parameters.interventionAdditionalWitness!;
    assert.match(witness, /^DECAY_MASK: on base=SGDm wd=0\.01 spec=layer4\.0\.bn2\.weight\+layer4\.0\.shortcut\.1\.weight\+layer4\.1\.bn2\.weight masked=3 of=62 numel=1536 /);
    const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
    assert.ok(log.split('\n').includes(witness), `${run.id} log carries its DECAY_MASK line`);
  }
});

test('the published arm means reproduce every level the two rows read', () => {
  const mean = (batch: string, arm: string) => {
    const armRuns = runs.filter(run => run.batch === batch && run.parameters.runLabel.split('-')[1] === arm);
    assert.equal(armRuns.length, 3, `${batch}-${arm}`);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['k01', 'CARWD0', 'TWOWD0', 'CTL2WD0', 'ONE50', 'ONE53', 'ONE59'].map(arm => mean('cwd4', arm)),
    ['22.8213', '70.0100', '67.0713', '22.8547', '65.2233', '58.5780', '67.9567']);
  assert.deepEqual(['k01W1', 'kLW1', 'k01W2', 'kLW2', 'k01W3', 'kLW3', 'k01W4', 'kLW4', 'CARW2'].map(arm => mean('cwd5', arm)),
    ['23.2240', '69.2940', '69.3513', '68.2620', '72.4980', '68.3373', '72.4080', '67.8813', '71.8327']);
  // cwd4's registered state bars, re-derived from the published levels: the matched pair is the only NULL.
  const level = (batch: string, arm: string) => Number(mean(batch, arm));
  const rec = level('cwd4', 'CARWD0') - 5, floor = level('cwd4', 'k01') + 2;
  assert.ok(level('cwd4', 'TWOWD0') >= rec && level('cwd4', 'ONE50') >= rec && level('cwd4', 'ONE59') >= rec, 'TWOWD0, ONE50 and ONE59 are REC');
  assert.ok(level('cwd4', 'ONE53') < rec && level('cwd4', 'ONE53') > floor, 'ONE53 is PART: neither recovered nor on the floor');
  assert.ok(level('cwd4', 'CTL2WD0') <= floor, 'the matched non-carrier pair stays at the k01 floor');
  // ONE50 clears its bar by less than half a standard error of an arm difference (0.526756).
  assert.ok(level('cwd4', 'ONE50') - rec < 0.5 * 0.526756, 'the ONE50-REC token is a near-bar reading');
  // cwd5's registered rung states, re-derived the same way: COLLAPSE at 0.1 and NOGAP at all three lower rungs.
  const rung = (n: number) => [level('cwd5', `k01W${n}`), level('cwd5', `kLW${n}`)];
  const [scalar1, layer1] = rung(1);
  assert.ok(scalar1 <= 0.5 * layer1, 'W1 collapses against its own layerwise arm');
  for (const n of [2, 3, 4]) {
    const [scalar, layerwise] = rung(n);
    assert.ok(layerwise >= 55, `W${n}'s layerwise reference is healthy`);
    assert.ok(layerwise - scalar < 10, `W${n} is NOGAP`);
    assert.ok(scalar > 0.5 * layerwise, `W${n} does not collapse`);
  }
});

test('phase-13 and phase-14 are their own documented phases and carry one record each', () => {
  const phase13 = data.activity.find(e => e.id === 'phase-13')!;
  const phase14 = data.activity.find(e => e.id === 'phase-14')!;
  assert.deepEqual([phase13.experimentIds, phase14.experimentIds], [['MT237'], ['MT238']]);
  assert.deepEqual([phase13.kind, phase14.kind], ['research-phase', 'research-phase']);
  assert.deepEqual([phase13.date, phase14.date], ['2026-09-20', '2026-09-20']);
  assert.match(phase13.title, /Which carriers, or how many/);
  assert.match(phase14.title, /Does the collapse exist at normal weight decays/);
  // phase-12 keeps its one record; the new phases take none of the earlier ones.
  assert.deepEqual(data.activity.find(e => e.id === 'phase-12')!.experimentIds, ['MT236']);
  // phase-15 and phase-16 were added later with caw2 and cgw1 (tests/mech7-landing.test.ts), phase-17 and phase-18 with crt1 and csh1.
  // phase-19 and phase-20 were added later with cvl1 and g3b (tests/mech9-landing.test.ts, tests/mech10-landing.test.ts).
  assert.equal(data.activity.filter(e => e.kind === 'research-phase').length, 20);
});

test('the two landings link the scorers, parsers, patch, runner and audit tools they cite', () => {
  const pathsOf = (id: string) => new Set(byId.get(id)!.codeIds.map(cid => data.sources.find(s => s.id === cid)!.path));
  const cwd4Paths = pathsOf('MT237'), cwd5Paths = pathsOf('MT238');
  for (const path of ['analysis/cWD4_countwd_score.py', 'analysis/cwd4_attack_indep.py', 'analysis/cwd4_design.py',
    'analysis/cwd4_rule20_envaudit.py', 'analysis/argsline_guard.py', 'analysis/corpus_exclusions.py',
    'patches/patch_decaymask.py', 'jobs/run_cifar_cwd1.sh']) {
    assert.ok(cwd4Paths.has(path), `MT237 links ${path}`);
  }
  for (const path of ['analysis/cWD5_wdladder_score.py', 'analysis/cwd5_attack_indep.py', 'analysis/cwd5_design.py',
    'analysis/cwd5_rule20_envaudit.py', 'analysis/argsline_guard.py', 'analysis/corpus_exclusions.py',
    'patches/patch_decaymask.py', 'jobs/run_cifar_cwd1.sh']) {
    assert.ok(cwd5Paths.has(path), `MT238 links ${path}`);
  }
  // Both batches ran cwd1's tree and cwd1's runner unchanged, so neither links a runner of its own.
  assert.ok(!cwd4Paths.has('jobs/run_cifar_cwd4.sh') && !cwd5Paths.has('jobs/run_cifar_cwd5.sh'), 'neither batch added a runner');
});
