import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 241 (crt1, CORRECTIONS 309) and 242 (csh1, CORRECTIONS 310), appended together at campaign commit
// a53bce1 and imported as MT241 and MT242 through scripts/register_model.py MECH8_ROWS. That one step edited header line 3
// and appended the two rows; it amended no earlier row and did not touch line 5. The joint ingest 5db62be touched results/
// alone and appended all 54 exclusion rows of both batches.
//
// MT241 is MIXED: it returns the registered branch WEAKENED-TO-TIE with its licence (cgw1's scalar-beats-both reading becomes
// 'ties (a bound) after re-tuning'), but its clauses split (every per-config reading still beats), six registered level bands
// were missed and the branch rests on the in-batch sigma. It bears on MT240, which was NOT amended and keeps its outcome.
// MT242 is MIXED, as MT237: it returns the registered prediction HORIZON-DOES-NOT-REPRODUCE, but the fired account's GPL band
// (60-76) was missed at 56.9707.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
const crt1 = byId.get('MT241')!;
const csh1 = byId.get('MT242')!;

const CRT1_FINAL = ['WEAKENED-TO-TIE', 'TUNED-SCALAR-TIES-BEST', 'ORACLE-SCALAR-TIES-BEST', 'M2-BEATS+A2-BEATS+M3-BEATS+M1-BEATS',
  'SEL-TRAIN:ch=A2,nd=A2,k01=A2', 'SEL-TEST:ch=A2,nd=A2,k01=A2', 'SIGMA-INBATCH', 'REPLICATE-MATCHES-CGW1', 'GRID-EDGE-ch-A2',
  'GRID-EDGE-nd-A2', 'GRID-EDGE-k01-A2', 'SEL-AGREE-ch', 'SEL-AGREE-nd', 'SEL-AGREE-k01', 'SELECTED-BOX-BOUND-ch', 'BOX-BOUND-chA2',
  'BOX-BOUND-chM3', 'BOX-BOUND-ndM3 (branch', 'two readings', 'every-config', 'selections', '12 stamps)'];
const CSH1_FINAL = ['HORIZON-DOES-NOT-REPRODUCE', 'A-NOGAP+M-NOGAP+P-NOGAP', 'DECAY-DOSE-A-BELOW', 'DECAY-DOSE-M-BELOW',
  'DECAY-DOSE-P-BELOW', 'TRACE-P-DOMINATES', 'HARNESS-CLEAN', 'BOTH-GRAINS-IN-BATCH', 'ANCHOR-IN-BATCH', 'SIGMA-PRIOR-FROZEN',
  'SCALAR-AHEAD-A-DESCRIPTIVE', 'SCALAR-AHEAD-M-DESCRIPTIVE', 'SCALAR-AHEAD-P-DESCRIPTIVE', 'SCALAR-COST-M-DESCRIPTIVE',
  'LAYERWISE-COST-M-DESCRIPTIVE', 'SCALAR-COST-P-DESCRIPTIVE', 'LAYERWISE-COST-P-DESCRIPTIVE', 'HW-UNIFORM-NVIDIA_L4',
  'TRAIN-AGREES (branch', 'cell states', '17 stamps; the 13 registered bounds in the next column)'];

test('MT241 and MT242 carry the outcome the documented rules give them, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(-4, -2).map(e => e.id), ['MT241', 'MT242'], 'appended in line order, before the later cvl1 and g3b rows');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [181, 163, 18, 3501]);
  // crt1 belongs to the count-matched partition audit, csh1 to the mechanism area; both Mixed, and neither corrects an
  // earlier published claim, so neither carries a Corrected badge.
  assert.deepEqual([crt1.section, crt1.area, crt1.kind, crt1.outcome, crt1.corrected, crt1.batches],
    [10, 'Count-matched partition audit', 'research', 'mixed', false, ['crt1']]);
  assert.deepEqual([csh1.section, csh1.area, csh1.kind, csh1.outcome, csh1.corrected, csh1.batches],
    [9, 'Mechanism and isolation', 'research', 'mixed', false, ['csh1']]);
  assert.ok(crt1.reason.startsWith(`Verdict: ${CRT1_FINAL.join(' + ')} Mixed: WEAKENED-TO-TIE: `), crt1.reason);
  assert.ok(csh1.reason.startsWith(`Verdict: ${CSH1_FINAL.join(' + ')} Mixed: HORIZON-DOES-NOT-REPRODUCE: `), csh1.reason);
  assert.deepEqual([crt1.figureIds, crt1.eventIds], [['page-8'], ['phase-17']]);
  assert.deepEqual([csh1.figureIds, csh1.eventIds], [['page-19'], ['phase-18']]);
  assert.ok(crt1.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 241), 'MT241 anchors MASTER-TABLE line 241');
  assert.ok(csh1.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 242), 'MT242 anchors MASTER-TABLE line 242');
  // Both own ARGS-value rows only, and neither landing amended a row anywhere.
  assert.deepEqual(crt1.warningIds, ['warning-MT241-verdict', 'warning-MT241-args-deviation', 'warning-MT241-source-1']);
  assert.deepEqual(csh1.warningIds, ['warning-MT242-verdict', 'warning-MT242-args-deviation']);
  // crt1's own runner lives in the campaign workspace on the cluster, not in the repository (as cgw1's does), and the record
  // says so rather than hiding it; csh1 ran cwd1's archived runner and links every source it cites.
  assert.deepEqual(data.warnings.find(w => w.id === 'warning-MT241-source-1')!.detail, 'Referenced source not available: jobs/run_cifar_crt1.sh');
  assert.ok(!data.warnings.some(w => w.detail.includes('Referenced source not available') && w.experimentId === 'MT242'), 'MT242 links every source it cites');
  assert.ok(!data.warnings.some(w => /-amendment-(29[89]|30\d|310)$/.test(w.id)), 'neither landing amended a row');
});

test('MT241 is Mixed: the scalar reading is weakened to a tie after re-tuning, and MT240 keeps its record', () => {
  assert.match(crt1.result, /TUNED \(= ORACLE, every grain selected A2 on TRAIN and on TEST\): `?T_ch`? = k01A2 - chA2 = 91\.7887 - 91\.3400 = \+0\.4487 pp/);
  assert.match(crt1.result, /chunk777 misses the 2 SE resolution clause by 0\.1078 pp; min T-B-2SE = -0\.1078 >= -0\.30, so TIES/);
  assert.match(crt1.reason, /the registered branch is returned with its licence/);
  assert.match(crt1.reason, /HEADLINE-REFUTED-BY-RETUNE is not reached/);
  assert.match(crt1.reason, /row 240's scalar-beats-both reading is rewritten from 'beats' to 'ties \(a bound\) after re-tuning'/);
  assert.match(crt1.reason, /BUT THE REGISTERED CLAUSES SPLIT AND EXPECTATIONS WERE DEFIED/);
  assert.match(crt1.reason, /six of the twelve registered level bands were missed \(chA2, ndA2, k01A2, k01M3, ndM1, k01M1\)/);
  assert.match(crt1.reason, /at the frozen floor 0\.1782 the same rule would read NOT-A-TUNING-ARTEFACT, the opposite licence \(descriptive\)/);
  // The bounds lead the scope, and the licence is quoted as registered.
  assert.match(crt1.scope, /The registered branch rests on sigma/);
  assert.match(crt1.scope, /every grain selected at the grid edge A2/);
  assert.match(crt1.scope, /after re-tuning, scalar is at most 0\.30 pp below the better audited partition at the lower edge/);
  assert.match(crt1.scope, /NOT licensed: "robust to re-tuning", "refuted", "the grains are equal", any other cell/);
  // cgw1's record is untouched: same outcome, no amendment.
  const cgw1 = byId.get('MT240')!;
  assert.equal(cgw1.outcome, 'unresolved');
  assert.ok(!cgw1.warningIds.some(id => id.includes('amendment')));
});

test('MT242 is Mixed: T-C is excluded as a sufficient account for a constant gamma, but a registered band was missed', () => {
  assert.match(csh1.result, /All three cells NOGAP, and in every cell SCALAR is ABOVE layerwise/);
  assert.match(csh1.result, /M `?GMS`? 66\.2980 vs `?GML`? 60\.6707, -5\.6273 \(-10\.83 SE\)/);
  assert.match(csh1.reason, /the registered prediction is returned/);
  assert.match(csh1.reason, /T-C is excluded as a sufficient account, as a BOUND and for a constant gamma/);
  assert.match(csh1.reason, /BUT A REGISTERED LEVEL BAND OF THE FIRED ACCOUNT IS MISSED: the account predicted every arm at 60-76, and GPL landed at 56\.9707/);
  assert.match(csh1.reason, /That is MT237's shape/);
  assert.match(csh1.scope, /gamma is CONSTANT, not the learned horizon/);
  assert.match(csh1.scope, /SUFFICIENCY, NOT NECESSITY/);
  assert.match(csh1.scope, /Cell P's reference barely passed its health check/);
  assert.match(csh1.scope, /the learned horizon waits for `?crd1`?/);
});

test('the two warnings name the ARGS-value kind, and every row of both batches is listed', () => {
  const w241 = data.warnings.find(w => w.id === 'warning-MT241-args-deviation')!;
  const w242 = data.warnings.find(w => w.id === 'warning-MT242-args-deviation')!;
  for (const warning of [w241, w242]) assert.deepEqual([warning.severity, warning.status], ['caution', 'documented']);
  assert.match(w241.detail, /every one of its 36 runs is the audit's core cell at --weight-decay-base 5e-4/);
  assert.match(w241.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 5db62be; CORRECTIONS 263, 300 and 309\.$/);
  assert.match(w242.detail, /every one of its 18 runs is at --weight-decay-base 5e-4 against the cell's standard 0\.1/);
  assert.match(w242.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 5db62be; CORRECTIONS 263, 301 and 310\.$/);
  assert.deepEqual(data.warnings.filter(w => w.id.endsWith('-args-deviation')).map(w => w.id),
    ['warning-MT228-args-deviation', 'warning-MT238-args-deviation', 'warning-MT239-args-deviation', 'warning-MT240-args-deviation',
      'warning-MT241-args-deviation', 'warning-MT242-args-deviation', 'warning-MT243-args-deviation', 'warning-MT244-args-deviation']);
});

test('the 54 runs link to their records with sanitized logs, and every one carries its ARGS-value mark', () => {
  assert.equal(runs.length, 3501);
  const cases = [['MT241', 'crt1', ['176', '177', '178'], 36, 'ResNet18', 'CIFAR10'],
    ['MT242', 'csh1', ['180', '181', '182'], 18, 'ResNet18_c100', 'CIFAR100']] as const;
  for (const [eid, batch, seeds, count, architecture, dataset] of cases) {
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
      const args = lines.filter(line => line.startsWith('ARGS:'));
      assert.equal(args.length, 1, `${run.id} prints exactly one ARGS line`);
      assert.ok(args[0].includes('--weight-decay-base 5e-4'), `${run.id} log carries the listed flag value`);
      assert.equal(run.parameters.argsDeviationKind, 'ARGS_WD_BASE', run.id);
      assert.equal(run.parameters.argsDeviationWitness, 'ARGS_WD_BASE: weight-decay-base=5e-4', run.id);
      assert.equal(run.parameters.argsDeviationArgsWitness, 'ARGS: --weight-decay-base 5e-4', run.id);
      assert.equal(run.parameters.argsDeviationAdditionalArgs, undefined, `${run.id} is a one-kind row`);
      assert.equal(run.parameters.argsDeviationAxes, undefined, `${run.id} carries no patch axis`);
    }
  }
  assert.equal(runs.filter(run => run.batch !== 'cvl1' && run.parameters.intervention).length, 183, 'no patch intervention was added');
  assert.equal(runs.filter(run => !['cvl1', 'g3b'].includes(run.batch) && run.parameters.argsDeviation).length, 139, "cmo1's 18, cwd5's 21, caw2's 18, cgw1's 28, crt1's 36 and csh1's 18");
});

test('the published arm means reproduce every level the two rows read', () => {
  const mean = (batch: string, arm: string, n = 3) => {
    const armRuns = runs.filter(run => run.batch === batch && run.parameters.runLabel.split('-')[1] === arm);
    assert.equal(armRuns.length, n, `${batch}-${arm}`);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  const config = (c: string) => ['ch', 'nd', 'k01'].map(grain => mean('crt1', `${grain}${c}`));
  assert.deepEqual(config('M2'), ['87.7513', '87.5967', '90.5107']);
  assert.deepEqual(config('A2'), ['91.3400', '91.1007', '91.7887']);
  assert.deepEqual(config('M3'), ['89.7180', '89.2327', '91.4147']);
  assert.deepEqual(config('M1'), ['86.7000', '86.6680', '87.4820']);
  assert.deepEqual(['G1S', 'G1L', 'GMS', 'GML', 'GPS', 'GPL'].map(arm => mean('csh1', arm)),
    ['72.4673', '68.4547', '66.2980', '60.6707', '60.6487', '56.9707']);
  // crt1's tuned contrasts at A2, and csh1's registered cell states, re-derived from the published levels.
  const level = (batch: string, arm: string) => Number(mean(batch, arm));
  assert.equal((level('crt1', 'k01A2') - level('crt1', 'chA2')).toFixed(4), '0.4487');
  assert.equal((level('crt1', 'k01A2') - level('crt1', 'ndA2')).toFixed(4), '0.6880');
  for (const [s, l] of [['G1S', 'G1L'], ['GMS', 'GML'], ['GPS', 'GPL']]) {
    assert.ok(level('csh1', l) >= 55 && level('csh1', l) - level('csh1', s) < 10, `${s} is NOGAP against a readable ${l}`);
    assert.ok(level('csh1', s) > 0.5 * level('csh1', l), `${s} does not collapse`);
  }
});

test('phase-17 and phase-18 are their own documented phases and carry one record each', () => {
  const phase17 = data.activity.find(e => e.id === 'phase-17')!;
  const phase18 = data.activity.find(e => e.id === 'phase-18')!;
  assert.deepEqual([phase17.experimentIds, phase18.experimentIds], [['MT241'], ['MT242']]);
  assert.deepEqual([phase17.kind, phase18.kind], ['research-phase', 'research-phase']);
  assert.deepEqual([phase17.date, phase18.date], ['2026-09-22', '2026-09-22']);
  assert.match(phase17.title, /Does the scalar reading survive re-tuning/);
  assert.match(phase18.title, /Does a short meta-horizon alone cause the collapse/);
  assert.deepEqual(data.activity.find(e => e.id === 'phase-16')!.experimentIds, ['MT240']);
  // phase-19 and phase-20 were added later with cvl1 and g3b (tests/mech9-landing.test.ts, tests/mech10-landing.test.ts).
  assert.equal(data.activity.filter(e => e.kind === 'research-phase').length, 20);
});

test('the two landings link the scorers, parsers and audit tools they cite', () => {
  const pathsOf = (id: string) => new Set(byId.get(id)!.codeIds.map(cid => data.sources.find(s => s.id === cid)!.path));
  const crt1Paths = pathsOf('MT241'), csh1Paths = pathsOf('MT242');
  for (const path of ['analysis/cRT1_retune_score.py', 'analysis/crt1_attack_indep.py', 'analysis/crt1_design.py', 'bin/cRT1_rule20.sh']) {
    assert.ok(crt1Paths.has(path), `MT241 links ${path}`);
  }
  for (const path of ['analysis/cSH1_horizon_score.py', 'analysis/csh1_attack_indep.py', 'analysis/csh1_design.py', 'bin/cSH1_rule20.sh']) {
    assert.ok(csh1Paths.has(path), `MT242 links ${path}`);
  }
});
