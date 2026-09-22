import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE line 236 (cwd3) at campaign commit 97eb049 (CORRECTIONS 278), imported as MT236 through
// scripts/register_model.py MECH5_ROWS. The landing edited header line 3 and NOTHING else: it amended no row of its own and
// none of anybody else's, so it is the first landing since the MUST tier to move no outcome at all. MT236 is Goal met under
// VERDICT_RULES -- it returns its registered branch AND its registered state word in the registered direction, the scorer
// exits 0 with every gate passing, the in-batch reference replicates cwd1, G-BITE excludes the batch's own broken-mask null
// on records, and no registered account misses a band and no control fails. Its 12 masked runs are listed under the single
// DECAY_MASK kind; the three k01 runs print DECAY_MASK: off and own no exclusion row.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;
const record = byId.get('MT236')!;

const CWD3_FINAL = ['CARRIER-DECAY-SUFFICES', 'CAR-REC+CTL-NULL+CTL2-NULL', 'HARNESS-CLEAN', 'PATCH-BITES', 'MASK-UPDATE-AND-TRACE',
  'CONV-LINEAR-WD-KEPT', 'ALL-ARMS-SCALAR', 'RECOVERY-AGAINST-NWD', 'CTL-HAS-ONE-SHIFT', 'CTL-DEPTH-MATCHED-NOT-MAGNITUDE',
  'ONE-NETWORK-RESNET18', 'ONE-CELL', 'HORIZON-100-ONLY', 'SIGMA-PRIOR-FROZEN', 'FLOOR-READINGS-ARE-BOUNDS',
  'HW-UNIFORM-NVIDIA_L4', 'TRAIN-AGREES (two branch tokens + 15 stamps)'];

test('MT236 carries the outcome the documented rules give it, with every FINAL token quoted', () => {
  assert.equal(CWD3_FINAL.length, 17);
  assert.deepEqual(data.experiments.slice(-7, -6).map(e => e.id), ['MT236'], 'appended before the later cwd4, cwd5, caw2, cgw1, crt1 and csh1 rows');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [179, 161, 18, 3437]);
  // A research question, Goal met; it corrects no earlier published claim, so it carries no Corrected badge.
  assert.deepEqual([record.section, record.area, record.kind, record.outcome, record.corrected, record.batches],
    [9, 'Mechanism and isolation', 'research', 'success', false, ['cwd3']]);
  // cwd3's FINAL carries TWO branch tokens, so the reason repeats both after the rule.
  assert.ok(record.reason.startsWith(`Verdict: ${CWD3_FINAL.join(' + ')} Goal met: CARRIER-DECAY-SUFFICES + CAR-REC+CTL-NULL+CTL2-NULL: `), record.reason);
  assert.deepEqual([record.figureIds, record.eventIds], [['page-19'], ['phase-12']]);
  assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 236), 'MT236 anchors MASTER-TABLE line 236');
  // It amended no row, so it owes exactly a verdict warning and an intervention warning and nothing else.
  assert.deepEqual(record.warningIds, ['warning-MT236-verdict', 'warning-MT236-intervention']);
  assert.ok(!data.warnings.some(w => w.detail.includes('Referenced source not available') && w.experimentId === 'MT236'), 'MT236 links every source it cites');
  assert.ok(!data.warnings.some(w => w.id.endsWith('-amendment-278')), 'the cwd3 landing amended no row anywhere');
});

test('MT236 leads with what bounds it, and claims neither a measured zero nor a share', () => {
  // The one measured positive, and the specificity contrast.
  assert.match(record.result, /PRIMARY P_CAR = CARWD0 - k01 = 70\.2640 - 22\.9853 = \+47\.2787 pp = \+89\.34 SE/);
  assert.match(record.result, /PRIMARY P_SPEC = CARWD0 - CTLWD0 = \+47\.2627 pp = \+89\.31 SE/);
  // Both controls are LOCATIONS at the floor, so the record must state them as bounds and never as measured zeros.
  assert.match(record.result, /P_CTL = CTLWD0 - k01 = \+0\.0160 pp = \+0\.03 SE and P_CTL2 = CTL2WD0 - k01 = -0\.0520 pp = -0\.10 SE -- both NULL, both FLOOR READINGS/);
  assert.match(record.scope, /the ONLY measured positives are CARWD0's/);
  assert.match(record.scope, /2 SE = 1\.058364 pp is the HALF-WIDTH, not the bound/);
  assert.match(record.scope, /never a measured zero/);
  // The residual is a bound in the other direction, and its sign is horizon-dependent.
  assert.match(record.result, /PRIMARY P_SET = NWD - CARWD0 = \+0\.4587 pp = \+0\.87 SE, NOT distinguishable from zero at the MATCH bar 5\.0 -- A BOUND, NOT A MEASUREMENT/);
  assert.match(record.scope, /crossing zero near epoch 80 and still moving at 99/);
  // F_CAR is descriptive and drifts through 1, so no share may be read off it.
  assert.match(record.result, /F_CAR = P_CAR \/ P_NWD = 0\.9904/);
  assert.match(record.scope, /"the carriers carry 99 % of the effect" may NOT be written/);
  // The two confounds, named separately, and the arm that would separate them.
  assert.match(record.scope, /IDENTITY vs POSITION CLASS is inseparable here/);
  assert.match(record.scope, /a COUNT \/ DOSE rival is NOT excluded and is a DIFFERENT confound from \(4\)/);
  assert.match(record.scope, /a two-carrier arm would separate them and was not run/);
  // The registered licence sentence, read strictly, and the RULE 16 defect reported and not fixed.
  assert.match(record.scope, /read "matched non-carrier last-block normalisation parameters" STRICTLY as the two sets actually run/);
  assert.match(record.scope, /One RULE 16 defect REPORTED and NOT fixed/);
  // The landing's own ordering slip is in the record, not hidden.
  assert.match(record.reason, /A landing-entry ordering slip is disclosed in the record rather than papered over/);
});

test('the intervention warning names the four arms, the patch and the single exclusion kind', () => {
  const warning = data.warnings.find(w => w.id === 'warning-MT236-intervention')!;
  assert.equal(warning.title, "cwd3's four masked arms run with the coupled weight decay switched off on a named set of normalisation tensors");
  assert.deepEqual([warning.severity, warning.status], ['caution', 'documented']);
  assert.match(warning.detail, /CARWD0, CTLWD0, CTL2WD0 and NWD ran PATCH_DECAYMASK/);
  assert.match(warning.detail, /in the weight update AND in the meta trace/);
  assert.match(warning.detail, /The batch needed NO new harness code/);
  assert.match(warning.detail, /one of them a BatchNorm SHIFT rather than a scale/);
  assert.match(warning.detail, /one kind only, no run of this batch carries a second/);
  assert.match(warning.detail, /The three k01 runs print DECAY_MASK: off/);
  assert.match(warning.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 97eb049; CORRECTIONS 269, 275 and 278\.$/);
  // No base-optimiser flag deviates in cwd3, so this landing added no ARGS-value mark; cwd5 later added its own.
  assert.deepEqual(data.warnings.filter(w => w.id.endsWith('-args-deviation')).map(w => w.id), ['warning-MT228-args-deviation', 'warning-MT238-args-deviation', 'warning-MT239-args-deviation', 'warning-MT240-args-deviation', 'warning-MT241-args-deviation', 'warning-MT242-args-deviation']);
});

test('the 15 runs link to MT236 with sanitized logs, and exactly the 12 masked runs are marked', () => {
  assert.equal(runs.length, 3437);
  const linked = runs.filter(run => run.batch === 'cwd3');
  assert.equal(linked.length, 15);
  assert.deepEqual([...record.runIds].sort(), linked.map(run => run.id).sort());
  for (const run of linked) {
    assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs],
      [['MT236'], 'Account2', 'completed', 'ResNet18_c100', 'CIFAR100', 100], run.id);
    assert.ok(['140', '141', '142'].includes(run.seed), run.id);
    assert.equal(run.parameters.argsDeviation, undefined, run.id);
    const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
    const lines = log.split('\n');
    if (run.parameters.intervention) {
      assert.ok(lines.includes(run.parameters.interventionWitness!), `${run.id} log carries its witness line`);
      // Every cwd3 run carries exactly ONE kind, so none owes a further witness.
      assert.equal(run.parameters.interventionAdditionalWitness, undefined, run.id);
      assert.match(run.parameters.interventionNote!, /^Not a plain k01 \(granularity scalar\) measurement/, run.id);
    } else {
      assert.equal(run.parameters.interventionWitness, undefined, run.id);
      assert.match(run.parameters.runLabel, /^cwd3-k01-s14[012]$/, 'only the three k01 runs are unmarked');
    }
    assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
    assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
    assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
    assert.equal(lines.filter(line => /^Epoch \d+,/.test(line)).length, 100, `${run.id} keeps every epoch line`);
  }
  assert.equal(linked.filter(run => run.parameters.intervention).length, 12);
  const armOf = (label: string) => runs.find(run => run.parameters.runLabel === label)!;
  assert.equal(armOf('cwd3-CARWD0-s140').parameters.intervention,
    'CARWD0: DECAY_MASK=layer4.0.bn2.weight+layer4.0.shortcut.1.weight+layer4.1.bn2.weight');
  assert.equal(armOf('cwd3-CTLWD0-s140').parameters.intervention,
    'CTLWD0: DECAY_MASK=layer4.0.bn1.weight+layer4.0.bn1.bias+layer4.1.bn1.weight');
  assert.equal(armOf('cwd3-CTL2WD0-s140').parameters.intervention,
    'CTL2WD0: DECAY_MASK=layer4.0.bn1.weight+layer4.1.bn1.weight');
  assert.equal(armOf('cwd3-NWD-s140').parameters.intervention, 'NWD: DECAY_MASK=normscale', 'NWD re-runs cwd1 mask in batch');
  assert.equal(runs.filter(run => run.parameters.intervention).length, 183, 'the 153 earlier patch interventions, these 12, and cwd4\'s 18');
  // The arm means of the published plateau5 values reproduce every level the row reads.
  const mean = (arm: string) => {
    const armRuns = runs.filter(run => run.batch === 'cwd3' && run.parameters.runLabel.split('-')[1] === arm);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['k01', 'CARWD0', 'CTLWD0', 'CTL2WD0', 'NWD'].map(mean),
    ['22.9853', '70.2640', '23.0013', '22.9333', '70.7227']);
  // The registered state bars, re-derived from the published levels: CARWD0 is REC, both controls are NULL.
  const level = (arm: string) => Number(mean(arm));
  assert.ok(level('CARWD0') >= level('NWD') - 5, 'CARWD0 is inside NWD recovery band');
  assert.ok(level('CTLWD0') <= level('k01') + 2 && level('CTL2WD0') <= level('k01') + 2, 'both controls are at the k01 floor');
});

test('the landing links the scorer, the parser, the patch, the runner and the audit tools it cites', () => {
  const paths = new Set(record.codeIds.map(cid => data.sources.find(s => s.id === cid)!.path));
  for (const path of ['analysis/cWD3_carrierwd_score.py', 'analysis/cwd3_attack_indep.py', 'analysis/cwd3_design.py',
    'analysis/cwd3_rule20_envaudit.py', 'analysis/argsline_guard.py', 'analysis/corpus_exclusions.py',
    'patches/patch_decaymask.py', 'jobs/run_cifar_cwd1.sh', 'bin/cWD3_rule20.sh']) {
    assert.ok(paths.has(path), `MT236 links ${path}`);
  }
  // The batch ran cwd1's tree and cwd1's runner unchanged, so it links no runner of its own.
  assert.ok(!paths.has('jobs/run_cifar_cwd3.sh'), 'cwd3 added no runner');
});

test('phase-12 is its own documented phase and carries only the new record', () => {
  const phase = data.activity.find(e => e.id === 'phase-12')!;
  assert.deepEqual(phase.experimentIds, ['MT236']);
  assert.equal(phase.kind, 'research-phase');
  assert.equal(phase.date, '2026-09-19');
  assert.match(phase.title, /The carriers' own decay/);
  // phase-11's four records keep it; the new phase takes none of them.
  assert.deepEqual(data.activity.find(e => e.id === 'phase-11')!.experimentIds, ['MT232', 'MT233', 'MT234', 'MT235']);
});
