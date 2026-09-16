import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { ResearchData } from '../src/types.ts';

// Campaign commit 0ade9cc (CORRECTIONS 244) inserted bracketed amendments into the so-what cells of MASTER-TABLE rows 222
// (cvt4) and 223 (cvt5): the verifiers' wording fixes that the landing had not carried. scripts/register_model.py
// C244_AMENDMENTS applies them to MT222 and MT223. Wording only: outcomes, kinds and badges stay, and the earlier wording
// is still in the scope because the brackets were inserted, not substituted.
const data = JSON.parse(readFileSync(new URL('../public/data/research.json', import.meta.url), 'utf8')) as ResearchData;
const byId = new Map(data.experiments.map(e => [e.id, e]));
const warning = (id: string) => data.warnings.find(w => w.id === `warning-${id}-amendment-244`);

test('MT222 carries CORRECTIONS 244: sufficiency at this cell, and P_COUP a floor location, outcome unchanged', () => {
  const record = byId.get('MT222')!;
  assert.deepEqual([record.outcome, record.kind, record.corrected], ['success', 'research', false]);
  assert.match(record.scope, /sharpens to 'on a SMALL step size'\. \[AMENDED at cycle 152, CORRECTIONS 244: read as SUFFICIENCY at this cell -- the one large trajectory tested \(MUTE's, replayed\) is sufficient to stall/);
  assert.match(record.scope, /\(3\) the floor readings are locations \[AMENDED at cycle 152, CORRECTIONS 244: so P_COUP ~ 0 \(-0\.27 pp\) is a reading between two arms at the same floor location, not a measured absence of coupling/);
  const note = warning('MT222')!;
  assert.ok(note && record.warningIds.includes(note.id));
  assert.equal(note.title, 'Wording amended by a later result');
  assert.match(note.detail, /^Outcome unchanged \(Goal met\)\. Amended at CORRECTIONS 244/);
  assert.match(note.detail, /MASTER-TABLE\.md line 222 at 0ade9cc; CORRECTIONS 244\.$/);
});

test('MT223 carries CORRECTIONS 244: it leads with its bound and reads the equilibrium as the level\'s, outcome unchanged', () => {
  const record = byId.get('MT223')!;
  assert.deepEqual([record.outcome, record.kind, record.corrected], ['success', 'research', false]);
  assert.match(record.scope, /\. \[LED WITH THE BOUND, AMENDED at cycle 152, CORRECTIONS 244: no branch bar within 4\.5 pp \(closest DROP13, 4\.58 pp\)/);
  assert.match(record.scope, /K-DEPENDENT-EQUILIBRIUM is an equilibrium of the LEVEL only, with 231\.4\(1\)'s qualifier: K13's TEST settled from ~epoch 105 and its complement first reached r <= 2 ~55 epochs later \(159\.2-161\.4\)\.\] \(a\) cvt2's GRADED is NOT a delay/);
  assert.ok(!/vote equilibrium|step-size equilibrium/i.test(record.scope + record.reason + record.result));
  const note = warning('MT223')!;
  assert.ok(note && record.warningIds.includes(note.id));
  assert.match(note.detail, /MASTER-TABLE\.md line 223 at 0ade9cc; CORRECTIONS 244\.$/);
});
