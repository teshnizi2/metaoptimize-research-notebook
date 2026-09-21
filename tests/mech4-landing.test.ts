import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 232-235 (cvt10, cwd1, csv1, cwd2) at campaign commit 2972d48 (CORRECTIONS 270, 271, 272 and 273),
// imported as MT232-MT235 through scripts/register_model.py MECH4_ROWS. The landing edited header line 3 and ONE existing
// row, 229 (MT229), which it amended in place to carry the frozen successor registered at CORRECTIONS 268; that is the only
// outcome this import moves. All four new rows are Goal met under VERDICT_RULES: each returns its registered branch in the
// registered direction, each scorer exits 0 with every gate passing, each positive control reproduces, and NO registered
// account misses a band and NO control fails. Their exclusion rows use the existing GROUP_HOLD and VOTE_W kinds plus the two
// CORRECTIONS 269 added, DECAY_MASK and SHADOW_VOTE; six cwd2 rows carry three kinds at once.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const warningsOf = (id: string) => data.warnings.filter(w => w.experimentId === id);
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;

const CVT10_FINAL = ['ONE-SUFFICES+ONE50-STALLS+ONE59-STALLS+ONE53-STALLS+SPLIT-NO-EFFECT', 'HARNESS-CLEAN', 'PATCH-BITES', 'HOLD-FROM-INIT',
  'BIG-IS-PLAINNET-DOSE', 'SINGLETON-GROUPS', 'NO-FREE-SPLIT-CONTROL', 'ONE-DOSE-ONLY', 'ONE-NETWORK-RESNET', 'HORIZON-100-ONLY',
  'SIGMA-PRIOR-FROZEN', 'POSITIVE-CONTROL-REPRODUCES', 'HOLDBIG3-AT-K01', 'ONE50-BETWEEN', 'ONE50BIG-AT-K01', 'ONE59-AT-ISO',
  'ONE59BIG-AT-K01', 'ONE53-BETWEEN', 'ONE53BIG-AT-K01', 'ISOSPLIT-AT-ISO', 'HOLDBIG3-BELOW-K01', 'FLOOR-READINGS-ARE-BOUNDS',
  'TRAIN-AGREES (one branch token of five words + 22 stamps)'];
const CWD1_FINAL = ['COLLAPSE-VANISHES', 'HARNESS-CLEAN', 'PATCH-BITES', 'MASK-20-NORM-SCALES', 'MASK-UPDATE-AND-TRACE', 'CONV-LINEAR-WD-KEPT',
  'ONE-NETWORK-RESNET18', 'ONE-CELL', 'HORIZON-100-ONLY', 'NO-WD-ON-REFERENCE-ARM', 'SIGMA-PRIOR-FROZEN', 'NOT-R50-UNDER-MASK',
  'TRAIN-AGREES (one branch token + 12 stamps)'];
const CSV1_FINAL = ['BOTH-ROUTES', 'HARNESS-CLEAN', 'PATCH-BITES', 'SCALAR-GROUPING', 'SHADOW-IS-COUNTERFACTUAL', 'APPLIED-LOW-IS-FLOOR-FROM-INIT',
  'ONE-TENSOR-50', 'ONE-NETWORK-PLAINNET', 'HORIZON-100-ONLY', 'SIGMA-PRIOR-FROZEN', 'POSITIVE-CONTROL-REPRODUCES', 'NAIVELOW-AT-HEAD',
  'SHADOWLOW-BETWEEN', 'MUTE-AT-K01', 'INERT-GAP:+0.24', 'SHARE-K01:0.527', 'SHARE-SHADOWLOW:0.359', 'SHARE-NAIVELOW:0.000',
  'SHADOW-VOTE-DOMINANT', 'PHYS-RECORDS-MIN:48', 'FLOOR-READINGS-ARE-BOUNDS', 'TRAIN-AGREES (one branch token + 21 stamps)'];
const CWD2_FINAL = ['WD-ROUTE', 'SCALAR-NEEDS-CARRIER-WD', 'HARNESS-CLEAN', 'PATCH-BITES', 'HOLDS-FROM-INIT', 'COMPLEMENT-ON-HEADPATH',
  'HELD-ARMS-OPEN-LOOP', 'MASK-ONE-TENSOR-IDX50', 'K01WD0-BOTH-ROUTES-CHANGED', 'ONE-NETWORK-PLAINNET', 'ONE-CELL', 'HORIZON-100-ONLY',
  'SIGMA-PRIOR-FROZEN', 'POSITIVE-CONTROL-REPRODUCES', 'FLOOR-READINGS-ARE-BOUNDS', 'TRAIN-AGREES (two branch tokens + 14 stamps)'];

test('MT232-MT235 carry the outcome the documented rules give them, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(-7, -3).map(e => e.id), ['MT232', 'MT233', 'MT234', 'MT235'], 'appended in line order, before the later cwd3, cwd4 and cwd5 rows');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [175, 157, 18, 3316]);
  assert.deepEqual([CVT10_FINAL.length, CWD1_FINAL.length, CSV1_FINAL.length, CWD2_FINAL.length], [23, 13, 22, 16]);
  // cwd2 is the one row of the cycle whose FINAL carries two branch tokens, so its reason opens with both.
  const cases: [string, string, string[], number, string][] = [
    ['MT232', 'cvt10', CVT10_FINAL, 232, CVT10_FINAL[0]],
    ['MT233', 'cwd1', CWD1_FINAL, 233, CWD1_FINAL[0]],
    ['MT234', 'csv1', CSV1_FINAL, 234, CSV1_FINAL[0]],
    ['MT235', 'cwd2', CWD2_FINAL, 235, 'WD-ROUTE + SCALAR-NEEDS-CARRIER-WD'],
  ];
  for (const [id, batch, final, line, branch] of cases) {
    const record = byId.get(id)!;
    assert.ok(record, `${id} must exist`);
    // Research questions, all four Goal met; none corrects an earlier published claim, so none carries the Corrected badge.
    assert.deepEqual([record.section, record.area, record.kind, record.outcome, record.corrected, record.batches],
      [9, 'Mechanism and isolation', 'research', 'success', false, [batch]], id);
    assert.ok(record.reason.startsWith(`Verdict: ${final.join(' + ')} Goal met: ${branch}: `), record.reason);
    assert.deepEqual([record.figureIds, record.eventIds], [['page-19'], ['phase-11']], id);
    assert.ok(data.activity.find(e => e.id === 'phase-11')!.experimentIds.includes(id));
    assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === line), `${id} anchors MASTER-TABLE line ${line}`);
    // The landing amended none of its own rows and corrected no registration text; each owes exactly one intervention warning.
    assert.deepEqual(warningsOf(id).map(w => w.id), [`warning-${id}-verdict`, `warning-${id}-intervention`], id);
    assert.ok(!data.warnings.some(w => w.detail.includes('Referenced source not available') && w.experimentId === id), `${id} links every source it cites`);
  }
});

test('every one of the four leads with what bounds it, and none claims necessity it did not test', () => {
  const cvt10 = byId.get('MT232')!;
  assert.match(cvt10.result, /CO-PRIMARIES P_ONE50 = ONE50 - ONE50BIG = 64\.7967 - 21\.1340 = \+43\.6627 pp = \+78\.50 SE/);
  assert.match(cvt10.reason, /each of the three carriers isolated alone in its own step-size group and held on PlainNet's measured dose .* is SUFFICIENT to stall/);
  assert.match(cvt10.reason, /the one-tensor stall needs the remaining carriers voting in the shared step size/);
  assert.match(cvt10.reason, /every reading is SUFFICIENCY and never necessity/);
  assert.match(cvt10.scope, /ISOSPLIT has NO free \[59,1,2\] control/);

  const cwd1 = byId.get('MT233')!;
  assert.match(cwd1.result, /P_NWD = k01NWD - k01 = 70\.7760 - 22\.9513 = \+47\.8247 pp = \+85\.99 SE/);
  assert.match(cwd1.reason, /the mask is ONE tensor set and BOTH routes at once, so the batch answers which tensors and NOT which route/);
  assert.match(cwd1.scope, /the COLLAPSING arm k01 carries no readout at all/);

  const csv1 = byId.get('MT234')!;
  assert.match(csv1.result, /P_VOTE = NAIVELOW - SHADOWLOW = 65\.2420 - 49\.8790 = \+15\.3630 pp = \+31\.89 SE/);
  // The registered licence says "large"; only the clamp floor was tested, so the record must carry the corrected sentence.
  assert.match(csv1.reason, /an applied step ABOVE THE CLAMP FLOOR on idx 50 is necessary for the FULL stall, not that a LARGE one is/);
  assert.match(csv1.reason, /a RULE 16 defect reported against the registered licence string itself and NOT fixed/);
  assert.match(csv1.scope, /INERT is NOT bitwise k01 over the landed 100-epoch run/);

  const cwd2 = byId.get('MT235')!;
  assert.match(cwd2.result, /PRIMARY P_WD = HIGHWD0 - HIGHHEADPATH = 65\.6260 - 10\.7127 = \+54\.9133 pp = \+98\.73 SE/);
  assert.match(cwd2.reason, /the mechanism is NOT watched where it is claimed to act/);
  assert.match(cwd2.reason, /leaves Zhou et al\. arXiv:2001\.11216 neither confirmed nor excluded/);
  assert.match(cwd2.scope, /240's OWN-STEP-MAGNITUDE, 246's HIGHHEADPATH-AT-K01 and 253's DOSE-GRADED are therefore REINTERPRETED/);
});

test('the row-229 amendment moves MT229 from Open to Mixed and keeps the superseded verdict, with no Corrected badge', () => {
  const record = byId.get('MT229')!;
  assert.deepEqual([record.outcome, record.corrected], ['mixed', false]);
  assert.ok(record.reason.startsWith('Verdict: NOMINATION-PARTIAL+ISO-RESCUES+CTL-NULL + FLOOR-READINGS-ARE-BOUNDS + '), record.reason);
  assert.match(record.reason, /SIGMA-PRIOR-FROZEN Mixed: NOMINATION-PARTIAL\+ISO-RESCUES\+CTL-NULL: /);
  // The predecessor is still frozen and unedited, and its verdict is kept verbatim in the record.
  assert.match(record.reason, /The predecessor's verdict, kept verbatim: UNRESOLVED-DECOMPOSITION \+ G-DECOMP k01-s112 -- a GATE, not a branch\. The batch has NO verdict and NO licence\.$/);
  assert.match(record.reason, /registered and pushed as commit 9581897 BEFORE it was run on a single cst1 record/);
  // The clauses split: the rescue transfers, the nomination misses its bar. That is Mixed, not Goal met.
  assert.match(record.reason, /the isolation rescue TRANSFERS to meta step 3e-4 \(D_ISO \+41\.1567 pp = \+74\.00 SE\)/);
  assert.match(record.reason, /DOM_C 0\.4727 is 709 records against a 750-record bar, MISSING BY 41/);
  assert.match(record.result, /\[AMENDED at cycle 155, CORRECTIONS 273/);
  assert.match(record.scope, /THE ISOLATION RESCUE TRANSFERS TO A SECOND META STEP SIZE \(\+41\.1567 pp, 74 SE\), THE VOTE-DOMINANCE NOMINATION DOES NOT/);
  const warning = data.warnings.find(w => w.id === 'warning-MT229-amendment-273')!;
  assert.ok(warning && record.warningIds.includes(warning.id));
  assert.deepEqual([warning.title, warning.severity, warning.status], ['Outcome moved by a later result', 'limitation', 'documented']);
  assert.match(warning.detail, /Outcome before the amendment: Open\./);
  assert.match(warning.detail, /Amendment record: docs\/MASTER-TABLE\.md line 229 at 2972d48; CORRECTIONS 273\.$/);
  // MT229 is the only record this landing amended, and the only outcome it moved.
  assert.deepEqual(data.warnings.filter(w => w.id.endsWith('-amendment-273')).map(w => w.id), ['warning-MT229-amendment-273']);
  assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === 229), 'MT229 anchors the amended row');
});

test('the four intervention warnings name the arms, the patches and the exclusion list', () => {
  const expected: Record<string, [string, RegExp[]]> = {
    MT232: ["cvt10's five held arms are group step-size holds, not plain arms", [
      /HOLDBIG3, ONE50BIG, ONE59BIG, ONE53BIG and ISOSPLIT ran PATCH_GROUPHOLD/,
      /ISOSPLIT is the one arm in the corpus with NO plain twin at its cell key/,
      /this batch added no new kind/]],
    MT233: ["cwd1's two masked arms run with the coupled weight decay switched off on the 20 BatchNorm scales", [
      /k01NWD and kLNWD ran PATCH_DECAYMASK \(DECAY_MASK=normscale\)/,
      /in the weight update AND in the meta trace/,
      /the DECAY_MASK witness kind added at CORRECTIONS 269/]],
    MT234: ["csv1's four intervened arms separate one tensor's applied step size from the vote it casts", [
      /INERT, SHADOWLOW and NAIVELOW ran PATCH_SHADOWVOTE/,
      /a COUNTERFACTUAL vote, not MetaOptimize's own/,
      /9 under the SHADOW_VOTE witness kind added at CORRECTIONS 269 and 3 under the existing VOTE_W kind/,
      /Track F proposed the same 12 rows independently/]],
    MT235: ["cwd2's twelve intervened arms combine a step-size hold, a forced complement replay and a decay mask", [
      /HIGHWD0 and LOWWD0 add the mask to that pair, so those six runs carry THREE interventions at once/,
      /the complement is FORCED onto a replay, so nothing here says a FREE complement behaves the same way/,
      /verified by the three MULTI_KIND entries registered at CORRECTIONS 269/]],
  };
  for (const [id, [title, patterns]] of Object.entries(expected)) {
    const warning = data.warnings.find(w => w.id === `warning-${id}-intervention`)!;
    assert.equal(warning.title, title, id);
    assert.deepEqual([warning.severity, warning.status], ['caution', 'documented'], id);
    for (const pattern of patterns) assert.match(warning.detail, pattern, id);
    assert.match(warning.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 2972d48; CORRECTIONS \d+, 269 and 27\d\.$|results\/CORPUS-EXCLUSIONS\.tsv at 2972d48; CORRECTIONS 258 and 270\.$/, id);
  }
  // None of the four ran with a base-optimiser flag deviation, so the ARGS-value kind stays cmo1's alone.
  assert.deepEqual(data.warnings.filter(w => w.id.endsWith('-args-deviation')).map(w => w.id), ['warning-MT228-args-deviation', 'warning-MT238-args-deviation']);
});

test('the 72 runs link to MT232-MT235 with sanitized logs, and exactly the 45 intervened runs are marked', () => {
  assert.equal(runs.length, 3316);
  const batches: [string, string, number, string[], string, number][] = [
    ['cvt10', 'MT232', 30, ['120', '121', '122'], 'ResNet18_c100', 15],
    ['cwd1', 'MT233', 9, ['128', '129', '130'], 'ResNet18_c100', 6],
    ['csv1', 'MT234', 18, ['136', '137', '138', '139'], 'PlainNet18_c100', 12],
    ['cwd2', 'MT235', 15, ['132', '133', '134'], 'PlainNet18_c100', 12],
  ];
  for (const [batch, id, count, seeds, network, intervened] of batches) {
    const linked = runs.filter(run => run.batch === batch);
    assert.equal(linked.length, count, batch);
    assert.deepEqual([...byId.get(id)!.runIds].sort(), linked.map(run => run.id).sort(), id);
    for (const run of linked) {
      assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs],
        [[id], 'Account2', 'completed', network, 'CIFAR100', 100], run.id);
      assert.ok(seeds.includes(run.seed), run.id);
      // None of the four changes a base-optimiser flag, so no run may carry the ARGS-value mark.
      assert.equal(run.parameters.argsDeviation, undefined, run.id);
      const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
      const lines = log.split('\n');
      if (run.parameters.intervention) {
        // The listed witness is the run's FIRST hold; every further hold prints its own line in the same log.
        assert.ok(lines.includes(run.parameters.interventionWitness), `${run.id} log carries its witness line`);
        for (const witness of (run.parameters.interventionAdditionalWitness ?? '').split(' | ').filter(Boolean)) {
          assert.ok(lines.includes(witness), `${run.id} log carries ${witness.split(':')[0]}`);
        }
        assert.match(run.parameters.interventionNote, /^Not a plain /, run.id);
      } else {
        assert.equal(run.parameters.interventionWitness, undefined, run.id);
        assert.equal(run.parameters.interventionAdditionalWitness, undefined, run.id);
      }
      assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
      assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
      assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
      assert.equal(lines.filter(line => /^Epoch \d+,/.test(line)).length, 100, `${run.id} keeps every epoch line`);
    }
    assert.equal(linked.filter(run => run.parameters.intervention).length, intervened, batch);
  }
  // The two new kinds, and the first rows in the corpus to carry three kinds at once.
  const armOf = (label: string) => runs.find(run => run.parameters.runLabel === label)!;
  assert.equal(armOf('cwd1-k01NWD-s128').parameters.intervention, 'k01NWD: DECAY_MASK=normscale');
  assert.match(armOf('cwd1-k01NWD-s128').parameters.interventionNote!, /no column records the coupled weight-decay mask intervention/);
  assert.equal(armOf('csv1-SHADOWLOW-s136').parameters.intervention, 'SHADOWLOW: SHADOW_VOTE=shadow:floor:layer4.1.bn2.weight');
  assert.equal(armOf('csv1-INERT-s136').parameters.intervention, 'INERT: SHADOW_VOTE=shadow:shared:layer4.1.bn2.weight');
  assert.match(armOf('csv1-SHADOWLOW-s136').parameters.interventionNote!, /no column records the shadow-vote intervention/);
  assert.equal(armOf('csv1-MUTE-s136').parameters.intervention, 'MUTE: VOTE_W=layer4.1.bn2.weight:0', 'csv1 keeps the existing vote-weight kind');
  // cvt9's EARLY and LATE were the first three-kind runs (CORRECTIONS 251); these six are the first to end in a decay mask.
  const three = runs.filter(run => ['cvt10', 'cwd1', 'csv1', 'cwd2'].includes(run.batch)
    && (run.parameters.intervention ?? '').split(' ').filter(part => part.includes('=')).length === 3);
  assert.deepEqual(three.map(run => run.parameters.runLabel).sort(),
    ['cwd2-HIGHWD0-s132', 'cwd2-HIGHWD0-s133', 'cwd2-HIGHWD0-s134', 'cwd2-LOWWD0-s132', 'cwd2-LOWWD0-s133', 'cwd2-LOWWD0-s134']);
  for (const run of three) {
    assert.match(run.parameters.interventionNote!, /step-size hold, complement step-size hold and coupled weight-decay mask interventions/, run.id);
    assert.deepEqual((run.parameters.interventionAdditionalWitness ?? '').split(' | ').map(w => w.split(':')[0]), ['COMP_HOLD', 'DECAY_MASK'], run.id);
  }
  // ISOSPLIT is the one listed arm with no plain twin at its cell key, so its note cannot name one.
  assert.match(armOf('cvt10-ISOSPLIT-s120').parameters.interventionNote!, /^Not a plain measurement of its cell key \(granularity sets:1-49,51-52,54-58,60-62\//);
  assert.match(armOf('cvt10-ISOSPLIT-s120').parameters.interventionNote!, /which no free arm anywhere in the corpus shares/);
  assert.equal(runs.filter(run => run.parameters.intervention).length, 183, 'the 108 earlier patch interventions, these 45, cwd3\'s 12 and cwd4\'s 18');
  // The arm means of the published plateau5 values reproduce every level the four rows read.
  const mean = (batch: string, arm: string) => {
    const armRuns = runs.filter(run => run.batch === batch && run.parameters.runLabel.split('-')[1] === arm);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['k01', 'ISO', 'HOLDBIG3', 'ONE50', 'ONE50BIG', 'ONE59', 'ONE59BIG', 'ONE53', 'ONE53BIG', 'ISOSPLIT'].map(arm => mean('cvt10', arm)),
    ['22.7540', '70.1660', '18.7233', '64.7967', '21.1340', '67.7053', '21.0973', '57.5033', '22.2893', '67.0893']);
  assert.deepEqual(['k01', 'k01NWD', 'kLNWD'].map(arm => mean('cwd1', arm)), ['22.9513', '70.7760', '69.3420']);
  assert.deepEqual(['k01', 'INERT', 'SHADOWLOW', 'NAIVELOW', 'MUTE', 'HEAD'].map(arm => mean('csv1', arm)),
    ['12.1180', '12.7320', '49.8790', '65.2420', '11.3027', '64.4293']);
  assert.deepEqual(['k01', 'k01WD0', 'HIGHHEADPATH', 'HIGHWD0', 'LOWWD0'].map(arm => mean('cwd2', arm)),
    ['12.0673', '65.7500', '10.7127', '65.6260', '65.0007']);
});

test('the landing links every scorer, launcher, patch and runner it cites', () => {
  const pathsOf = (id: string) => new Set(byId.get(id)!.codeIds.map(cid => data.sources.find(s => s.id === cid)!.path));
  const expected: Record<string, string[]> = {
    MT232: ['analysis/cVT10_onevsthree_score.py', 'analysis/cvt10_attack_indep.py', 'analysis/corpus_exclusions.py'],
    MT233: ['analysis/cWD1_normwd_score.py', 'analysis/cwd1_attack_indep.py', 'analysis/cwd1_livemodel_normscale_check.py', 'patches/patch_decaymask.py', 'jobs/run_cifar_cwd1.sh'],
    MT234: ['analysis/cSV1_shadowvote_score.py', 'analysis/csv1_attack_indep.py', 'patches/patch_shadowvote.py', 'jobs/run_cifar_csv1.sh'],
    MT235: ['analysis/cWD2_carrierwd_score.py', 'analysis/cwd2_attack_indep.py', 'patches/patch_decaymask.py', 'jobs/run_cifar_cwd2.sh'],
  };
  for (const [id, paths] of Object.entries(expected)) {
    const linked = pathsOf(id);
    for (const path of paths) assert.ok(linked.has(path), `${id} links ${path}`);
  }
  // The frozen successor that CORRECTIONS 273 carried into row 229 is linked from MT229 itself.
  assert.ok(pathsOf('MT229').has('analysis/cST2_carriervote_score.py'), 'MT229 links the frozen successor');
  assert.ok(pathsOf('MT229').has('analysis/cST1_carrier_contrast_score.py'), 'MT229 still links the unedited predecessor');
});

test('phase-11 is its own documented phase and carries only the four new records', () => {
  const phase = data.activity.find(e => e.id === 'phase-11')!;
  assert.deepEqual(phase.experimentIds, ['MT232', 'MT233', 'MT234', 'MT235']);
  assert.equal(phase.kind, 'research-phase');
  assert.equal(phase.date, '2026-09-18');
  assert.match(phase.title, /Which tensors, and through which route/);
  // phase-10's four records keep it; the new phase takes none of them.
  assert.deepEqual(data.activity.find(e => e.id === 'phase-10')!.experimentIds, ['MT228', 'MT229', 'MT230', 'MT231']);
});
