import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData, Run } from '../src/types.ts';

// MASTER-TABLE lines 228-231 (cmo1, cst1, cct1, cmg1) at campaign commit 40d29cf (CORRECTIONS 264, 265 and 266), imported as
// MT228-MT231 through scripts/register_model.py MUST_ROWS. The landing amended no existing row: it edited header line 3 alone,
// because CORRECTIONS 266.12 deliberately left the bottom-line paragraph (line 5) unamended. No registration text was corrected
// in place either, so no record gains a registration or amendment warning.
// Only cmo1 owes exclusion rows, and they are of a NEW kind: its M9 and W0 arms deviate from the standard cell in a
// base-optimiser CLI flag alone, listed under the ARGS-value witness kinds of CORRECTIONS 263. cst1, cct1 and cmg1 own none.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const runs = JSON.parse(readFileSync(new URL('../public/data/runs.json', import.meta.url), 'utf8')) as Run[];
const byId = new Map(data.experiments.map(e => [e.id, e]));
const warningsOf = (id: string) => data.warnings.filter(w => w.experimentId === id);
const master = data.sources.find(s => s.path === 'docs/MASTER-TABLE.md')!;

const CMO1_FINAL = ['M9:COLLAPSE-PERSISTS/ISO-RESCUES+W0:COLLAPSE-IS-CONFIG/ISO-UNREADABLE', 'HARNESS-CLEAN', 'FLAGS-WITNESSED', 'LIVE-HARNESS',
  'ONE-VALUE-PER-FACTOR', 'ONE-FACTOR-AT-A-TIME', 'ISO-IS-CISO1-CARRIERS', 'NO-CTL-AT-NEW-CONFIGS', 'ONE-NETWORK-RESNET', 'HORIZON-100-ONLY',
  'ALPHA0-1E-6', 'SIGMA-PRIOR-FROZEN', 'ANCHOR-REPRODUCES', 'M9-K01-AT-ANCHOR', 'W0-K01-LIFTED', 'W0-ISO-AT-REF', 'FLOOR-READINGS-ARE-BOUNDS',
  'TRAIN-AGREES'];
const CST1_FINAL = ['UNRESOLVED-DECOMPOSITION', 'G-DECOMP k01-s112 -- a GATE, not a branch. The batch has NO verdict and NO licence'];
const CCT1_FINAL = ['NOT-COLLAPSED+CARRIERS-DO-NOT-DOMINATE', 'RATIO=0.9641', 'DOM_C=0.0000', 'TOP3_C=0.6940', 'KL-DOM_C=0.0000',
  'KL-CARRIERS-DO-NOT-DOMINATE', 'DOM_TOP3=0.0793', 'R_T=0.4770', 'DOWN=1.0000', 'PINNED-FRAC=0.0000', 'MODAL-TOP3={50,53,59}', 'ARGMAX=59',
  'GAP-IN-BATCH=3.2607', 'HARNESS-CLEAN', 'SPEC-APPLIED', 'DECOMPOSITION-OK', 'DATASET-AND-HEAD-CO-VARY', 'ONE-CELL-CIFAR10-MS1E-3',
  'HORIZON-100-ONLY', 'ONE-NETWORK-RESNET18', 'FLOOR-NOT-APPLICABLE', 'SIGMA-PRIOR-FROZEN'];
const CMG1_FINAL = ['NO-MERGE-HARMS', 'HARNESS-LIVE-UNPATCHED', 'PARTITION-VERIFIED', 'ONE-MERGE-SET-N9', 'CONTROL-TRIPLE-47-48-56',
  'REFERENCE-SINGLETON-SETS', 'ONE-NETWORK-RESNET18', 'ONE-CELL', 'HORIZON-100-ONLY', 'SIGMA-PRIOR-FROZEN', 'MAGNITUDE-NOT-SEPARATED',
  'MCAR-NEAR-KLS-BELOW', 'MCTL-NEAR-KLS-BELOW', 'KLS-IN-LAYERWISE-BAND', 'SEEDS-AGREE', 'MCAR-GSTAR-PINS', 'MCTL-GSTAR-PINS', 'TRAIN-AGREES',
  'ACCOUNT-A3-MERGE-HARMLESS'];

test('MT228-MT231 carry the outcome the documented rules give them, with every FINAL token quoted', () => {
  assert.deepEqual(data.experiments.slice(-4).map(e => e.id), ['MT228', 'MT229', 'MT230', 'MT231'], 'appended after every existing record, in line order');
  assert.deepEqual([data.meta.stats.experiments, data.meta.stats.researchQuestions, data.meta.stats.methodChecks, data.meta.stats.runs], [168, 150, 18, 3181]);
  assert.deepEqual([CMO1_FINAL.length, CST1_FINAL.length, CCT1_FINAL.length, CMG1_FINAL.length], [18, 2, 22, 19]);
  // Each reason opens with the row's returned branch words, as MT221, MT223, MT226 and MT227 did.
  const cases: [string, string, string[], number, string, string][] = [
    ['MT228', 'cmo1', CMO1_FINAL, 228, 'mixed', 'Mixed'],
    ['MT229', 'cst1', CST1_FINAL, 229, 'unresolved', 'Open'],
    ['MT230', 'cct1', CCT1_FINAL, 230, 'success', 'Goal met'],
    ['MT231', 'cmg1', CMG1_FINAL, 231, 'mixed', 'Mixed'],
  ];
  for (const [id, batch, final, line, outcome, label] of cases) {
    const record = byId.get(id)!;
    assert.ok(record, `${id} must exist`);
    // Research questions with a real outcome; none corrects an earlier published claim, so none carries the Corrected badge.
    assert.deepEqual([record.section, record.area, record.kind, record.outcome, record.corrected, record.batches],
      [9, 'Mechanism and isolation', 'research', outcome, false, [batch]], id);
    assert.ok(record.reason.startsWith(`Verdict: ${final.join(' + ')} ${label}: ${final[0]}: `), record.reason);
    assert.deepEqual([record.figureIds, record.eventIds], [['page-19'], ['phase-10']], id);
    assert.ok(data.activity.find(e => e.id === 'phase-10')!.experimentIds.includes(id));
    assert.ok(record.sourceRefs.some(ref => ref.sourceId === master.id && ref.line === line), `${id} anchors MASTER-TABLE line ${line}`);
    // The landing amended no row and corrected no registration text, so neither warning kind exists on any of the four.
    assert.ok(!warningsOf(id).some(w => /-amendment|-registration/.test(w.id)), id);
    assert.ok(!data.warnings.some(w => w.detail.includes(`Referenced source not available`) && w.experimentId === id), `${id} links every source it cites`);
  }
});

test('MT228 (cmo1) is Mixed because the weight-decay leg leaves half its question unreadable', () => {
  const record = byId.get('MT228')!;
  assert.match(record.title, /^Does the scalar collapse, and the carrier-isolation rescue, need the harness's base-optimiser configuration/);
  assert.match(record.result, /CO-PRIMARIES, M9 leg: G_M9 = M9kL - M9k01 = 69\.3913 - 24\.5833 = \+44\.8080 pp = \+80\.56 SE/);
  assert.match(record.result, /L_W0 = W0k01 - k01 = \+48\.7713 pp = \+87\.69 SE/);
  // The momentum leg answers; the weight-decay leg defies the registered expectation (VERDICT_RULES mixed).
  assert.match(record.reason, /the momentum leg answers both halves of the question in its registered direction/);
  assert.match(record.reason, /the scalar arm is the BEST arm in the batch \(W0k01 71\.5600/);
  assert.match(record.reason, /the registered ISO reading is stamped W0-ISO-AT-REF and NOT read/);
  assert.match(record.reason, /10 RTX 2080 Ti, 17 NVIDIA L4/);
  assert.match(record.scope, /Bounded, and led with \(264\.4\): \(1\) ONE value per factor and ONE factor at a time/);
  assert.match(record.scope, /NOT licensed: specificity at M9; which weight-decay route acts;/);
  // The refuted delivered-report claim travels with the record (264.6 W1).
  assert.match(record.scope, /no run prints its Epoch 99 line twice \(264\.6 W1\)/);
});

test('MT229 (cst1) is Open: a gate, not a branch, and nothing in it is a result', () => {
  const record = byId.get('MT229')!;
  assert.match(record.result, /NO NUMBER HERE IS A RESULT\. The registered scorer STOPPED at G-DECOMP on all nine runs and exited 1/);
  assert.match(record.result, /0\.5\*ulp\(beta\)\/ms = 1\.589e-03 at ms 3e-4/);
  assert.match(record.reason, /^Verdict: UNRESOLVED-DECOMPOSITION \+ G-DECOMP k01-s112/);
  assert.match(record.reason, /the batch reaches NO branch, owns NO licence sentence and NO number in its row is a result/);
  assert.match(record.reason, /a FROZEN SUCCESSOR must be REGISTERED BEFORE its output is read/);
  assert.match(record.scope, /NOT licensed: anything\./);
  // The scorer gate is a reported, unfixed defect (RULE 16), so the record must not read as a scientific refutation.
  assert.match(record.reason, /reported under RULE 16 and deliberately NOT fixed/);
  assert.equal(record.outcome, 'unresolved');
});

test('MT230 (cct1) is Goal met with both registered words returned and neither bar near', () => {
  const record = byId.get('MT230')!;
  assert.match(record.result, /PRIMARY RATIO = k01 \/ kL = 87\.5147 \/ 90\.7753 = 0\.9641 >= 0\.90 -> NOT-COLLAPSED/);
  assert.match(record.result, /CO-PRIMARY DOM_C = 0\.0000 <= 0\.10 -> CARRIERS-DO-NOT-DOMINATE/);
  assert.match(record.reason, /no registered band is missed and neither bar is near/);
  assert.match(record.scope, /ASSOCIATION, NOT CAUSATION/);
  assert.match(record.scope, /The registered licence, verbatim:/);
  assert.match(record.scope, /NOT licensed \(the scorer's printed line, verbatim\): causation \(dataset and head width co-vary\)/);
});

test('MT231 (cmg1) is Mixed on a near bar, with no account reproducing the levels', () => {
  const record = byId.get('MT231')!;
  assert.match(record.result, /PRIMARY DELTA_ID = MCTL - MCAR = 68\.4560 - 65\.4335 = \+3\.0225 pp = \+6\.27 SE/);
  assert.match(record.reason, /NO account's LEVELS are reproduced \(A3 predicts MCAR 68\.4166 against the observed 65\.4335/);
  assert.match(record.reason, /0\.8850 pp \(1\.84 SE\) short of the margin/);
  assert.match(record.scope, /"harmless" is the wrong plain-English gloss/);
  assert.match(record.scope, /keeping the three carriers out of a shared step-size group with N is NOT necessary for the layerwise level, within \+\/-5 pp/);
  // The refuted descriptive claim, re-checked from the raw probe records, travels with the record (266.6).
  assert.match(record.scope, /N's members do NOT all stay off the floor late/);
});

test('the ARGS-value deviation warning names cmo1 arms, the flags and the exclusion list', () => {
  const warning = data.warnings.find(w => w.id === 'warning-MT228-args-deviation')!;
  assert.equal(warning.title, "cmo1's M9 and W0 arms differ from the standard cell in a base-optimiser CLI flag, not in any CSV column");
  assert.deepEqual([warning.severity, warning.status], ['caution', 'documented']);
  assert.match(warning.detail, /M9k01, M9kL and M9ISO ran --momentum-param-base 0\.9 with weight decay held at the standard 0\.1/);
  assert.match(warning.detail, /W0k01, W0kL and W0ISO ran --weight-decay-base 0 with momentum held at the standard 0\.99/);
  assert.match(warning.detail, /the witness is the run's OWN ARGS: line/);
  assert.match(warning.detail, /ARGS_MOMENTUM_BASE or ARGS_WD_BASE added at CORRECTIONS 263/);
  assert.match(warning.detail, /results\/CORPUS-EXCLUSIONS\.tsv at 40d29cf; CORRECTIONS 255, 263 and 264\.$/);
  // Only cmo1 owes one; no other record of this landing gains an exclusion warning of either kind.
  assert.deepEqual(data.warnings.filter(w => w.id.endsWith('-args-deviation')).map(w => w.id), ['warning-MT228-args-deviation']);
  for (const id of ['MT229', 'MT230', 'MT231']) assert.deepEqual(warningsOf(id).map(w => w.id), [`warning-${id}-verdict`]);
});

test('the 54 runs link to MT228-MT231 with sanitized logs; only cmo1 carries 18 ARGS-value marks', () => {
  assert.equal(runs.length, 3181);
  const batches: [string, string, number, string[], string, string, number][] = [
    ['cmo1', 'MT228', 27, ['108', '109', '110'], 'ResNet18_c100', 'CIFAR100', 18],
    ['cst1', 'MT229', 9, ['112', '113', '114'], 'ResNet18_c100', 'CIFAR100', 0],
    ['cct1', 'MT230', 6, ['116', '117', '118'], 'ResNet18', 'CIFAR10', 0],
    ['cmg1', 'MT231', 12, ['124', '125', '126', '127'], 'ResNet18_c100', 'CIFAR100', 0],
  ];
  for (const [batch, id, count, seeds, network, dataset, deviating] of batches) {
    const linked = runs.filter(run => run.batch === batch);
    assert.equal(linked.length, count, batch);
    assert.deepEqual([...byId.get(id)!.runIds].sort(), linked.map(run => run.id).sort(), id);
    for (const run of linked) {
      assert.deepEqual([run.experimentIds, run.account, run.status, run.architecture, run.dataset, run.epochs], [[id], 'Account2', 'completed', network, dataset, 100], run.id);
      assert.ok(seeds.includes(run.seed), run.id);
      const log = readFileSync(new URL(`../public${run.logHref}`, import.meta.url), 'utf8');
      const lines = log.split('\n');
      // No patch ran in any of these four batches, so no run may carry a patch-intervention mark.
      assert.equal(run.parameters.intervention, undefined, run.id);
      assert.equal(run.parameters.interventionAdditionalWitness, undefined, run.id);
      if (run.parameters.argsDeviation) {
        const arm = run.parameters.runLabel.split('-')[1];
        const [kind, flag, value] = arm.startsWith('M9')
          ? ['ARGS_MOMENTUM_BASE', 'momentum-param-base', '0.9']
          : ['ARGS_WD_BASE', 'weight-decay-base', '0'];
        assert.equal(run.parameters.argsDeviation, `${arm}: --${flag} ${value}`, run.id);
        assert.equal(run.parameters.argsDeviationKind, kind, run.id);
        assert.equal(run.parameters.argsDeviationWitness, `${kind}: ${flag}=${value}`, run.id);
        assert.equal(run.parameters.argsDeviationArgsWitness, `ARGS: --${flag} ${value}`, run.id);
        assert.match(run.parameters.argsDeviationNote, /^Not a plain .* measurement at the standard cell: /, run.id);
        assert.ok(run.parameters.argsDeviationNote.includes(`no column records the ${arm.startsWith('M9') ? 'base momentum' : 'base weight decay'} flag --${flag}`), run.id);
        // The run's own sanitized log still carries the ARGS line the witness was read from.
        assert.ok(lines.some(line => line.startsWith('ARGS: ') && line.includes(` --${flag} ${value} `)), `${run.id} log carries its ARGS line`);
      } else {
        // The anchor arms ran both flags at the standard cell, so they are ordinary measurements.
        assert.ok(lines.some(line => line.startsWith('ARGS: ')), run.id);
        if (batch === 'cmo1') assert.ok(lines.some(line => line.startsWith('ARGS: ') && line.includes(' --momentum-param-base 0.99 --weight-decay-base 0.1 ')), run.id);
      }
      assert.equal(run.parameters.logAvailability, 'published: sanitized original raw log', run.id);
      assert.notEqual(run.parameters.originalLogSha256, run.parameters.publicLogSha256, `${run.id} log was redacted`);
      assert.doesNotMatch(log, /\/home\/|\/zfsstore\/|\bs50\d{5}\b|saleh\w*bars|\bnode\d{3}\b/, run.id);
      assert.equal(lines.filter(line => /^Epoch \d+,/.test(line)).length, 100, `${run.id} keeps every epoch line`);
    }
    assert.equal(linked.filter(run => run.parameters.argsDeviation).length, deviating, batch);
  }
  // The 108 patch interventions of the earlier landings are untouched by the new kind.
  assert.equal(runs.filter(run => run.parameters.intervention).length, 108);
  assert.equal(runs.filter(run => run.parameters.argsDeviation).length, 18);
  // The arm means of the published plateau5 values reproduce the rows' levels.
  const mean = (batch: string, arm: string) => {
    const armRuns = runs.filter(run => run.batch === batch && run.parameters.runLabel.split('-')[1] === arm);
    return (armRuns.reduce((sum, run) => sum + run.testAccuracy!, 0) / armRuns.length).toFixed(4);
  };
  assert.deepEqual(['k01', 'kL', 'ISO', 'M9k01', 'M9kL', 'M9ISO', 'W0k01', 'W0kL', 'W0ISO'].map(arm => mean('cmo1', arm)),
    ['22.7887', '69.0507', '70.4227', '24.5833', '69.3913', '70.9047', '71.5600', '68.1700', '71.4620']);
  assert.deepEqual(['k01', 'ISO', 'CTL'].map(arm => mean('cst1', arm)), ['28.5173', '69.6740', '28.6980']);
  assert.deepEqual(['k01', 'kL'].map(arm => mean('cct1', arm)), ['87.5147', '90.7753']);
  assert.deepEqual(['KLS', 'MCAR', 'MCTL'].map(arm => mean('cmg1', arm)), ['69.5485', '65.4335', '68.4560']);
});

test('the MUST-tier landing links every scorer, launcher and runner it cites', () => {
  const pathsOf = (id: string) => new Set(byId.get(id)!.codeIds.map(cid => data.sources.find(s => s.id === cid)!.path));
  const expected: Record<string, string[]> = {
    MT228: ['analysis/cMO1_momwd_score.py', 'analysis/cmo1_attack_indep.py', 'analysis/corpus_exclusions.py', 'bin/cMO1_momwd.sh', 'jobs/run_cifar_cmo1.sh', 'tests/test_cmo1_flags_bite.py'],
    MT229: ['analysis/cST1_carrier_contrast_score.py', 'analysis/cst1_attack_indep.py', 'analysis/cCV0_carrier_vote_core.py', 'bin/cST1_carrier_contrast.sh', 'jobs/run_cifar_live_a0d0a1b9.sh'],
    MT230: ['analysis/cCT1_c10_dominance_score.py', 'analysis/cct1_attack_indep.py', 'bin/cCT1_c10_dominance.sh'],
    MT231: ['analysis/cMG1_mergecarrier_score.py', 'analysis/cmg1_attack_indep.py', 'analysis/cmg1_rule20_audit.py', 'bin/cMG1_mergecarrier.sh', 'jobs/run_cifar_cmg1.sh'],
  };
  for (const [id, paths] of Object.entries(expected)) {
    const linked = pathsOf(id);
    for (const path of paths) assert.ok(linked.has(path), `${id} links ${path}`);
  }
  // The archived live runner carries cst1's and cct1's unpatched harness; every linked source is available.
  assert.ok(data.sources.some(s => s.path === 'jobs/run_cifar_live_a0d0a1b9.sh'));
});
