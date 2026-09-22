import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 239 (caw2, CORRECTIONS 295) and 240 (cgw1, CORRECTIONS 296), appended together at campaign commit
// 8a99001 and imported as MT239 and MT240 through scripts/register_model.py MECH7_ROWS. That one step edited header line 3
// and appended the two rows; it amended no earlier row and did not touch line 5. The joint ingest 477a853 touched results/
// alone and appended all 46 exclusion rows of both batches.
//
// MT239 is MIXED, as MT238: its co-primary is answered in the registered direction (the standard AdamW + Adam recipe does
// not collapse at this cell) and every band of the fired account is hit, but its dose half returned NOT-REACHED -- the dose
// arm never reached the control's realised shrink -- so immunity at the control's dose is undecided.
// MT240 is OPEN, as MT183: its primary returned AUDIT-UNDECIDED, whose registered licence is "report the interval". The
// co-reported scalar reading is resolved and is carried in the reason.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
const caw2 = byId.get('MT239')!;
const cgw1 = byId.get('MT240')!;

const CAW2_FINAL = ['NO-COLLAPSE-CONTROL-DOSE-NOT-REACHED', 'M-COLLAPSE+L-NOGAP+A-NOGAP+X-NOGAP+K-COLLAPSED', 'ATTR-BASE-PROTECTS',
  'GATE-DOES-NOT-FIRE', 'DOSE-M-ABOVE', 'DOSE-L-BELOW', 'DOSE-A-BELOW', 'DOSE-X-BELOW', 'HARNESS-CLEAN', 'BOTH-GRAINS-IN-BATCH',
  'CONTROL-IN-BATCH', 'ALPHA-SCALED-DECAY-ONLY', 'ALPHA-INDEPENDENT-NOT-TESTED', 'HARNESS-ADAMW-M-UNNORMALISED',
  'X-IS-A-DOSE-ARM-NOT-A-STANDARD-VALUE', 'ONE-NETWORK-RESNET18', 'ONE-CELL-OTHERWISE', 'HORIZON-100-ONLY', 'THREE-SEEDS',
  'WD-1E-2-NOT-RUN', 'LION-BASE-NOT-RUN', 'STEP1-HALF-OF-GATE-ONLY', 'LEADLAG-NOT-READ', 'FLOOR-READINGS-ARE-BOUNDS',
  'SIGMA-PRIOR-FROZEN', 'SCALAR-AHEAD-L-DESCRIPTIVE', 'SCALAR-AHEAD-A-DESCRIPTIVE', 'SCALAR-AHEAD-X-DESCRIPTIVE',
  'HW-UNIFORM-NVIDIA_L4', 'TRAIN-AGREES (four branch words', '26 stamps)'];
const CGW1_FINAL = ['AUDIT-UNDECIDED', 'SCALAR-BEATS-BEST', 'W1-SURVIVES+W2-SURVIVES+W4-UNDECIDED', 'ONE-CELL', 'THREE-RUNGS',
  'ALPHA-INDEPENDENT-DECAY-NOT-TESTED', 'FLOOR-READINGS-ARE-BOUNDS', 'SIGMA-PRIOR-FROZEN', 'ANCHOR-MATCHES-POOL',
  'LAYERWISE-BELOW-SCALAR-W4 (three branch words', '7 stamps)'];

test('MT239 and MT240 carry the outcome the documented rules give them, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(-4, -2).map(e => e.id), ['MT239', 'MT240'], 'appended in line order, before the later crt1 and csh1 rows');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [179, 161, 18, 3437]);
  // caw2 is a mechanism-scope record, Mixed; cgw1 belongs to the count-matched partition audit, Open. Neither corrects an
  // earlier published claim, so neither carries a Corrected badge.
  assert.deepEqual([caw2.section, caw2.area, caw2.kind, caw2.outcome, caw2.corrected, caw2.batches],
    [9, 'Mechanism and isolation', 'research', 'mixed', false, ['caw2']]);
  assert.deepEqual([cgw1.section, cgw1.area, cgw1.kind, cgw1.outcome, cgw1.corrected, cgw1.batches],
    [10, 'Count-matched partition audit', 'research', 'unresolved', false, ['cgw1']]);
  assert.ok(caw2.reason.startsWith(`Verdict: ${CAW2_FINAL.join(' + ')} Mixed: NO-COLLAPSE-CONTROL-DOSE-NOT-REACHED: `), caw2.reason);
  assert.ok(cgw1.reason.startsWith(`Verdict: ${CGW1_FINAL.join(' + ')} Open: AUDIT-UNDECIDED: `), cgw1.reason);
  assert.deepEqual([caw2.figureIds, caw2.eventIds], [['page-19'], ['phase-15']]);
  assert.deepEqual([cgw1.figureIds, cgw1.eventIds], [['page-8'], ['phase-16']]);
  assert.ok(caw2.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 239), 'MT239 anchors MASTER-TABLE line 239');
  assert.ok(cgw1.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 240), 'MT240 anchors MASTER-TABLE line 240');
  // Both own ARGS-value rows only, and neither landing amended a row anywhere.
  assert.deepEqual(caw2.warningIds, ['warning-MT239-verdict', 'warning-MT239-args-deviation']);
  assert.deepEqual(cgw1.warningIds, ['warning-MT240-verdict', 'warning-MT240-args-deviation', 'warning-MT240-source-1']);
  assert.ok(!data.warnings.some(w => /-amendment-29[5-7]$/.test(w.id)), 'neither landing amended a row');
  // cgw1's own runner lives in the campaign workspace, not in the repository, and the record says so rather than hiding it.
  assert.deepEqual(data.warnings.find(w => w.id === 'warning-MT240-source-1')!.detail, 'Referenced source not available: jobs/run_cifar_cgw1.sh');
  assert.ok(!data.warnings.some(w => w.detail.includes('Referenced source not available') && w.experimentId === 'MT239'), 'MT239 links every source it cites');
});

test('MT239 is Mixed: the standard recipe does not collapse here, but the dose half is undecided', () => {
  assert.match(caw2.result, /CO-PRIMARY G_A = AL - AS = 67\.8440 - 72\.9493 = -5\.1053 pp = -9\.75 SE/);
  assert.match(caw2.reason, /the co-primary is answered in its registered direction/);
  assert.match(caw2.reason, /The registered prediction \(A-NOGAP with ATTR-BASE-PROTECTS\) is returned/);
  assert.match(caw2.reason, /swapping only the meta optimiser still collapses \(MS 13\.2440 against its own layerwise 68\.2060/);
  assert.match(caw2.reason, /Every registered level band of the fired account is hit/);
  // The undecided half, and the precedent it repeats.
  assert.match(caw2.reason, /BUT THE BATCH'S DOSE HALF IS UNDECIDED: the dose arm's realised peak shrink reached only RHO_X = 0\.0591/);
  assert.match(caw2.reason, /the branch is NOT-REACHED and not IMMUNE/);
  assert.match(caw2.reason, /'the base protects' is not separated from 'the base never reaches the dangerous dose'/);
  assert.match(caw2.reason, /That is MT238's shape/);
  // The bounds are scope and are led with; the harness AdamW is named as the harness's, not torch's.
  assert.match(caw2.scope, /THE DOSE WAS NOT REACHED, SO NOTHING IS DECIDED ABOUT IMMUNITY/);
  assert.match(caw2.scope, /Every NOGAP is a BOUND/);
  assert.match(caw2.scope, /"The standard recipe" is the HARNESS's AdamW \+ Adam/);
  assert.match(caw2.scope, /a CORNER CASE OF THE SGDm BASE at this cell/);
  assert.match(caw2.scope, /With AdamW \+ Adam the grains do not differ by the 10 pp bar at wd 0\.1 or 1\.0, but even at wd 1\.0 the scalar arm's realised peak shrink stayed below half the control's\./);
});

test('MT240 is Open: the primary lies between the bars, and the scalar reading is resolved and qualified', () => {
  assert.match(cgw1.result, /PRIMARY D_W4 = chW4 - ndW4 = 87\.9060 - 87\.6650 = \+0\.2410 pp = \+1\.90 SE, \+\/-2 SE \[-0\.0120, \+0\.4940\] -- UNDECIDED/);
  assert.match(cgw1.reason, /the primary is undecided/);
  assert.match(cgw1.reason, /the registered licence is 'report the interval; no survive \/ vanish sentence'/);
  assert.match(cgw1.reason, /the count-matched sign is never reversed/);
  assert.match(cgw1.reason, /plain scalar \(90\.4650\) is ABOVE both audited partitions, by \+2\.5590 pp = \+20\.23 SE and \+2\.8000 pp = \+22\.13 SE \(SCALAR-BEATS-BEST\)/);
  assert.match(cgw1.reason, /the audit's practical significance must be qualified whatever D4 says/);
  // The verifier's caveat binds unweakened, and "standard decay" is the value, not a standard recipe's realised decay.
  assert.match(cgw1.scope, /W4 IS FAR LESS DECAYED THAN A STANDARD SGD RECIPE, and the registration verifier's caveat binds unweakened/);
  assert.match(cgw1.scope, /no reading at W4 settles whether the audit effect is an artefact of non-standard decay/);
  assert.match(cgw1.scope, /where "standard decay" means the decay VALUE 5e-4 applied alpha-scaled/);
  assert.match(cgw1.scope, /NOT licensed: "survives" or "vanishes at standard decay"/);
});

test('the two warnings name their ARGS-value kinds and the corpus\'s first TWO-ARGS rows', () => {
  const w239 = data.warnings.find(w => w.id === 'warning-MT239-args-deviation')!;
  const w240 = data.warnings.find(w => w.id === 'warning-MT240-args-deviation')!;
  for (const warning of [w239, w240]) assert.deepEqual([warning.severity, warning.status], ['caution', 'documented']);
  assert.match(w239.detail, /--momentum-param-base 0\.9 -- AdamW's own beta1, flagged only because the registered ARGS standard \(0\.99\) is set per flag/);
  assert.match(w239.detail, /They are the corpus's FIRST TWO-ARGS rows \(CORRECTIONS 294\)/);
  assert.match(w239.detail, /The control K01 and the meta-swap arms MS and ML run the standard 0\.99 \/ 0\.1 and own no row/);
  assert.match(w239.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 477a853; CORRECTIONS 263, 290, 294 and 296\.$/);
  assert.match(w240.detail, /cgw1 varies ONE axis, --weight-decay-base, across three rungs of the audit's core cell/);
  assert.match(w240.detail, /The twelve anchor runs at the standard 0\.1 deviate on nothing and own no row/);
  assert.match(w240.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 477a853; CORRECTIONS 263, 291 and 296\.$/);
  assert.deepEqual(data.warnings.filter(w => w.id.endsWith('-args-deviation')).map(w => w.id),
    ['warning-MT228-args-deviation', 'warning-MT238-args-deviation', 'warning-MT239-args-deviation', 'warning-MT240-args-deviation', 'warning-MT241-args-deviation', 'warning-MT242-args-deviation']);
});

test('the 67 runs link to their records with sanitized logs, and exactly the 46 listed rows are marked', () => {
  assert.equal(runs.length, 3437);
  const cases = [['MT239', 'caw2', ['160', '161', '162'], 27, 18, 'ResNet18_c100', 'CIFAR100'],
    ['MT240', 'cgw1', ['152', '153', '154', '155'], 40, 28, 'ResNet18', 'CIFAR10']] as const;
  for (const [eid, batch, seeds, count, marked, architecture, dataset] of cases) {
    const linked = runs.filter(run => run.batch === batch);
    assert.equal(linked.length, count);
    assert.deepEqual([...byId.get(eid)!.runIds].sort(), linked.map(run => run.id).sort());
    for (const run of linked) {
      assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs],
        [[eid], 'Account2', 'completed', architecture, dataset, 100], run.id);
      assert.ok((seeds as readonly string[]).includes(run.seed), run.id);
      assert.equal(run.parameters.intervention, undefined, `${run.id}: no patch ran in this batch`);
      const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
      const lines = log.split('\n');
      assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
      assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
      assert.equal(lines.filter(line => /^Epoch \d+,/.test(line)).length, 100, `${run.id} keeps every epoch line`);
      const label = run.parameters.runLabel;
      if (run.parameters.argsDeviation) {
        const args = lines.filter(line => line.startsWith('ARGS:'));
        assert.equal(args.length, 1, `${run.id} prints exactly one ARGS line`);
        const [flag, value] = run.parameters.argsDeviationWitness!.split(': ')[1].split('=');
        assert.ok(args[0].includes(`--${flag} ${value}`), `${run.id} log carries the listed flag value`);
        assert.equal(run.parameters.argsDeviationArgsWitness, `ARGS: --${flag} ${value}`, run.id);
      } else if (batch === 'caw2') {
        assert.match(label, /^caw2-(K01|MS|ML)-s16[012]$/, 'only the control and the meta-swap arms are unmarked');
      } else {
        assert.match(label, /^cgw1-(ch|nd|k01|kL)W1-s15[234]$/, 'only the twelve anchor runs at the standard 0.1 are unmarked');
      }
    }
    assert.equal(linked.filter(run => run.parameters.argsDeviation).length, marked);
  }
  assert.equal(runs.filter(run => run.parameters.intervention).length, 183, 'no patch intervention was added');
  assert.equal(runs.filter(run => !['crt1', 'csh1'].includes(run.batch) && run.parameters.argsDeviation).length, 85, "cmo1's 18, cwd5's 21, caw2's 18 and cgw1's 28");
  // crt1 and csh1 later added 36 and 18 of their own (tests/mech8-landing.test.ts).
  assert.equal(runs.filter(run => run.parameters.argsDeviation).length, 139);
});

test("caw2's XS and XL rows are the corpus's first TWO-ARGS rows and are witnessed on both ARGS kinds", () => {
  const twoArgs = runs.filter(run => run.parameters.argsDeviationAdditionalArgs);
  assert.deepEqual(twoArgs.map(run => run.parameters.runLabel).sort(),
    ['caw2-XL-s160', 'caw2-XL-s161', 'caw2-XL-s162', 'caw2-XS-s160', 'caw2-XS-s161', 'caw2-XS-s162']);
  for (const run of twoArgs) {
    const arm = run.parameters.runLabel.split('-')[1];
    assert.equal(run.parameters.argsDeviation, `${arm}: --momentum-param-base 0.9 + --weight-decay-base 1.0`);
    assert.equal(run.parameters.argsDeviationKind, 'ARGS_WD_BASE');
    assert.equal(run.parameters.argsDeviationWitness, 'ARGS_WD_BASE: weight-decay-base=1.0');
    assert.equal(run.parameters.argsDeviationArgsWitness, 'ARGS: --weight-decay-base 1.0');
    assert.equal(run.parameters.argsDeviationAdditionalArgs, 'ARGS_MOMENTUM_BASE: momentum-param-base=0.9');
    assert.equal(run.parameters.argsDeviationAdditionalArgsWitness, 'ARGS: --momentum-param-base 0.9');
    assert.equal(run.parameters.argsDeviationAxes, undefined, 'a two-ARGS row carries no patch axis');
    assert.match(run.parameters.argsDeviationNote!, /TWO-ARGS: this run ALSO deviates on the base momentum flag --momentum-param-base, set to 0\.9 against the standard 0\.99/);
    const args = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8').split('\n').filter(line => line.startsWith('ARGS:'));
    assert.ok(args[0].includes('--momentum-param-base 0.9 ') && args[0].includes('--weight-decay-base 1.0 '), `${run.id} log carries both flags`);
  }
  // The single-kind AdamW rows and CORRECTIONS 284's two-AXIS rows keep their own shapes.
  assert.equal(runs.filter(run => run.batch === 'caw2' && run.parameters.argsDeviationKind === 'ARGS_MOMENTUM_BASE').length, 12);
  assert.deepEqual(runs.filter(run => run.parameters.argsDeviationAxes).map(run => run.parameters.runLabel), ['cwd5-CARW2-s146', 'cwd5-CARW2-s147', 'cwd5-CARW2-s148']);
});

test('the published arm means reproduce every level the two rows read', () => {
  const mean = (batch: string, arm: string, n: number) => {
    const armRuns = runs.filter(run => run.batch === batch && run.parameters.runLabel.split('-')[1] === arm);
    assert.equal(armRuns.length, n, `${batch}-${arm}`);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['K01', 'MS', 'ML', 'LS', 'LL', 'AS', 'AL', 'XS', 'XL'].map(arm => mean('caw2', arm, 3)),
    ['22.5240', '13.2440', '68.2060', '72.5673', '68.1093', '72.9493', '67.8440', '73.2507', '68.6840']);
  const rung = (w: string, n: number) => ['ch', 'nd', 'k01', 'kL'].map(grain => mean('cgw1', `${grain}${w}`, n));
  assert.deepEqual(rung('W1', 3), ['92.4287', '92.0593', '92.1907', '92.7933']);
  assert.deepEqual(rung('W2', 3), ['89.2047', '88.7513', '91.1793', '90.6300']);
  assert.deepEqual(rung('W4', 4), ['87.9060', '87.6650', '90.4650', '89.8110']);
  // caw2's registered cell states, re-derived from the published levels: only the meta swap and the control collapse.
  const level = (batch: string, arm: string, n = 3) => Number(mean(batch, arm, n));
  assert.ok(level('caw2', 'MS') <= 0.5 * level('caw2', 'ML'), 'M collapses against its own layerwise arm');
  for (const [s, l] of [['LS', 'LL'], ['AS', 'AL'], ['XS', 'XL']]) {
    assert.ok(level('caw2', l) >= 55 && level('caw2', l) - level('caw2', s) < 10, `${s} is NOGAP against a healthy ${l}`);
  }
  assert.ok(level('caw2', 'K01') <= 30, 'the control collapses');
  // cgw1's primary contrast at 5e-4 lies between the +0.15 and +0.30 bars, and scalar beats both partitions there.
  const d4 = level('cgw1', 'chW4', 4) - level('cgw1', 'ndW4', 4);
  assert.ok(d4 > 0.15 && d4 < 0.30, `D4 ${d4} lies between the bars`);
  assert.ok(level('cgw1', 'k01W4', 4) - level('cgw1', 'chW4', 4) >= 0.30 && level('cgw1', 'k01W4', 4) - level('cgw1', 'ndW4', 4) >= 0.30, 'scalar is above both partitions');
});

test('phase-15 and phase-16 are their own documented phases and carry one record each', () => {
  const phase15 = data.activity.find(e => e.id === 'phase-15')!;
  const phase16 = data.activity.find(e => e.id === 'phase-16')!;
  assert.deepEqual([phase15.experimentIds, phase16.experimentIds], [['MT239'], ['MT240']]);
  assert.deepEqual([phase15.kind, phase16.kind], ['research-phase', 'research-phase']);
  assert.deepEqual([phase15.date, phase16.date], ['2026-09-22', '2026-09-22']);
  assert.match(phase15.title, /Does the collapse reach the standard recipe/);
  assert.match(phase16.title, /Does the partition audit survive standard weight decay/);
  assert.deepEqual(data.activity.find(e => e.id === 'phase-14')!.experimentIds, ['MT238']);
  // phase-17 and phase-18 were added later with crt1 and csh1 (tests/mech8-landing.test.ts).
  assert.equal(data.activity.filter(e => e.kind === 'research-phase').length, 18);
});

test('the two landings link the scorers, parsers and audit tools they cite', () => {
  const pathsOf = (id: string) => new Set(byId.get(id)!.codeIds.map(cid => data.sources.find(s => s.id === cid)!.path));
  const caw2Paths = pathsOf('MT239'), cgw1Paths = pathsOf('MT240');
  for (const path of ['analysis/cAW2_confound_score.py', 'analysis/caw2_attack_indep.py', 'analysis/caw2_design.py',
    'analysis/caw2_rule20_envaudit.py', 'bin/cAW2_stdrecipe.sh', 'bin/cAW2_rule20.sh', 'tests/test_caw2_realrun.py',
    'analysis/argsline_guard.py', 'analysis/corpus_exclusions.py']) {
    assert.ok(caw2Paths.has(path), `MT239 links ${path}`);
  }
  for (const path of ['analysis/cGW1_auditwd_score.py', 'analysis/cgw1_attack_indep.py', 'analysis/cgw1_design.py',
    'analysis/cgw1_rule20_envaudit.py', 'analysis/cgw1_live_check.py', 'bin/cGW1_auditwd.sh', 'bin/cGW1_rule20.sh',
    'bin/cGW1_stage_harness.sh', 'analysis/argsline_guard.py']) {
    assert.ok(cgw1Paths.has(path), `MT240 links ${path}`);
  }
});
