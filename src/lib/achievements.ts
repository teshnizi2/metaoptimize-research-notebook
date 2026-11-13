/** The "What we achieved" page. Every number and status comes from content/idea-tree.json:
 *  a fact's `quote` must appear in the text of the test (or part) it names, and a card's colour comes from its tests.
 *  tests/achievements.test.ts checks all of it. Nothing here is a new claim. */
import { findNode, ideaTree, type IdeaTreeData, type TreeStatus } from './idea-tree';

export type Source =
  | { test: string; quote: string }
  | { part: string; quote: string }
  /** Counts the tests of a part that have one status, for "6 of 6". */
  | { partCount: { part: string; status: TreeStatus; n: number } };

export interface Fact {
  /** Shown on the card; the first fact of a card is its key number, written with what it means. Say "points", not "pp". */
  label: string;
  source: Source;
}

export type Group = 'stand' | 'negative' | 'checked' | 'built';

export interface Achievement {
  id: string;
  group: Group;
  /** One plain sentence. */
  claim: string;
  /** The first fact is the key number; the others sit under "Show the tests". */
  facts: Fact[];
  /** The condition the result holds under: one short line that follows "Holds when". */
  holds: string;
  /** The rest of the honest limit, shown in smaller text under the line. */
  caveat?: string;
  holdsSources: Source[];
  /** One line on why the card matters. */
  matters: string;
  /** One line for the "At a glance" list at the top; only some cards have one. */
  glance?: string;
  /** Tests the card rests on; they set its colour. */
  tests: string[];
  /** Tests that set the limit, shown apart. */
  limits?: string[];
  /** Whole parts of the tree the card rests on. */
  parts?: string[];
  link?: { to: string; label: string };
}

export const groups: { id: Group; title: string; intro: string }[] = [
  { id: 'stand', title: 'Findings that stand', intro: 'Green in the Idea tree: the test held up, under the condition on the card.' },
  { id: 'negative', title: 'Clean negative results', intro: 'Things that did not work, shown clearly enough to be useful knowledge.' },
  { id: 'checked', title: 'How the work was checked', intro: 'What the project did to avoid fooling itself.' },
  { id: 'built', title: 'What was built', intro: 'Counts come from the notebook’s own record.' },
];

export type SectionId = 'key' | Group;

/** The page's sections, in reading order. Key results for the paper come first, whatever their group. */
export const sections: { id: SectionId; title: string; intro: string }[] = [
  { id: 'key', title: 'Key results for the paper', intro: 'The three results the paper rests on. Two held up; one is a clear loss.' },
  { id: 'stand', title: 'Other findings that hold', intro: 'Green in the Idea tree: the test held up, under the condition on the card.' },
  { id: 'negative', title: 'Clean negatives: what we ruled out', intro: 'Things that did not work, shown clearly enough to be useful.' },
  { id: 'checked', title: 'How the work was checked', intro: 'What the project did to avoid fooling itself.' },
  { id: 'built', title: 'What we built', intro: 'Counts come from the notebook’s own record.' },
];

/** The section a card sits in: a key result goes under "Key results"; every other card under its own group. */
export function sectionOf(card: Achievement, data: IdeaTreeData = ideaTree): SectionId {
  return isKeyAchievement(card, data) ? 'key' : card.group;
}

export const achievements: Achievement[] = [
  {
    id: 'audit', group: 'stand',
    claim: 'At the same number of step-size groups, equal-sized chunks beat groups that follow the network’s structure.',
    facts: [
      { label: '+0.49 points of accuracy, about 3 times the run-to-run noise', source: { test: 'B5.1', quote: '+0.49 percentage points, about three times the noise' } },
      { label: 'Ahead in 20 of 20 settings (datasets, networks and optimisers)', source: { test: 'B5.2', quote: '20 of 20 positive' } },
      { label: 'Group sizes alone recover the whole gap', source: { test: 'B5.4', quote: 'Sizes alone recover the whole gap' } },
    ],
    holds: 'at weight decay 0.1 and 0.01.',
    caveat: 'All 20 settings ran at 0.1. At the standard 5e-4 the size is undecided (+0.24, range −0.01 to +0.49), and why it works is still open.',
    holdsSources: [
      { test: 'B5.2', quote: 'All 20 ran at weight decay 0.1' },
      { test: 'B9.1', quote: 'Holds at 0.1, 0.01; standard undecided' },
      { test: 'B9.1', quote: '+0.24 with a range of -0.01 to +0.49' },
      { part: 'B5', quote: 'no explanation survived' },
    ],
    matters: 'This is the audit, the project’s main result: it held in all 20 settings tested.',
    glance: 'Equal-sized chunks beat structure-following groups in 20 of 20 settings.',
    tests: ['B5.1', 'B5.2', 'B5.4'], limits: ['B9.1'],
  },
  {
    id: 'collapse', group: 'stand',
    claim: 'The CIFAR-100 collapse of one shared step size is carried by just three BatchNorm scales.',
    facts: [
      { label: 'Three scales, only 0.014 % of the weights, carry the whole collapse', source: { test: 'B7.5', quote: '3 BatchNorm scales, 0.014 % of weights' } },
      { label: 'Giving them their own step sizes recovers +46.9 points of accuracy, the whole gap', source: { test: 'B7.6', quote: 'Recovers +46.9 pp, the whole gap' } },
      { label: 'Switching weight decay off on just those three recovers +47.3 points', source: { test: 'B8.3', quote: '+47.3 pp' } },
      { label: 'A matched trio of look-alike scales recovers 0.36 % of the gap', source: { test: 'B7.7', quote: 'matched trio recovers 0.36 %' } },
      { label: 'Also seen on VGG, GroupNorm and PlainNet', source: { test: 'B7.8', quote: 'Yes: VGG, GroupNorm, PlainNet' } },
    ],
    holds: 'at a weight decay near 0.1.',
    caveat: 'At the standard 5e-4 the collapse does not appear and the order reverses (one step size ahead by 4.5 points): a corner case, not a general hazard.',
    holdsSources: [
      { test: 'B8.5', quote: 'No: one step size ahead by 4.5 pp' },
      { part: 'B8', quote: 'A corner case, not a general hazard' },
    ],
    matters: 'It is the project’s clearest single measurement: a large accuracy drop traced to a tiny set of weights.',
    glance: 'A collapse of about 47 points comes from just three BatchNorm scales.',
    tests: ['B7.5', 'B7.6', 'B7.7', 'B7.8', 'B8.3'], limits: ['B8.5'],
  },
  {
    id: 'idea3', group: 'stand',
    claim: 'MetaOptimize forgives a badly chosen starting step size, where a fixed rate collapses.',
    facts: [
      { label: 'Steady over a 100,000-fold range of starting step sizes', source: { test: 'B4.7', quote: 'Steady over 100,000x range' } },
      { label: 'About 1.1 points below its own best setting', source: { test: 'B4.7', quote: 'about 1.1 percentage points below its own best setting' } },
    ],
    holds: 'as forgiveness of a bad start only.',
    caveat: 'It does not win against a tuned optimiser.',
    holdsSources: [
      { test: 'B4.7', quote: 'it does not win' },
      { part: 'B4', quote: 'Idea 3 (steady across step sizes) survives' },
    ],
    matters: 'It is a safety net: a fixed rate collapses on a bad start, and this stays steady.',
    glance: 'It forgives a badly chosen start over a 100,000-fold range, but does not win.',
    tests: ['B4.7'], limits: ['B6.2'],
  },
  {
    id: 'structure', group: 'stand',
    claim: 'Letting step sizes adapt freely creates shared structure, so the learned signal works as a diagnostic.',
    facts: [
      { label: '0.16 to 0.55 independent information with free step sizes, against 0.03 to 0.08 when frozen', source: { test: 'B4.3', quote: '0.16 to 0.55 with free step sizes, against 0.03 to 0.08 when they were frozen' } },
    ],
    holds: 'as a reading of the signal.',
    caveat: 'It is a diagnostic, not a method that wins.',
    holdsSources: [{ part: 'B4', quote: 'the signal is a diagnostic, not a method' }],
    matters: 'It shows the free step sizes themselves create the structure, so the signal can help diagnose training.',
    tests: ['B4.3'],
  },
  {
    id: 'rival', group: 'negative',
    claim: 'MetaOptimize loses to a plain, well-tuned SGD, and longer training does not close the gap.',
    facts: [
      { label: 'Behind by 1.8 to 14.6 points of accuracy', source: { part: 'B6', quote: 'MetaOptimize loses by 1.8 to 14.6 pp' } },
      { label: 'Stays flat, about 1.9 points, from 100 to 600 epochs', source: { test: 'B6.3', quote: 'No: stays flat, about 1.9 pp' } },
      { label: 'Without augmentation: 3.6 points (CIFAR-10), 12.2 points (CIFAR-100)', source: { test: 'B6.5', quote: '3.6 pp (CIFAR-10), 12.2 pp (CIFAR-100)' } },
    ],
    holds: 'against tuned SGD with momentum and a cosine schedule, on CIFAR-10 and CIFAR-100.',
    caveat: 'At the standard decay 5e-4 the CIFAR-100 gap grows to 14.6 points.',
    holdsSources: [
      { part: 'B6', quote: 'tuned SGD with momentum and a cosine schedule' },
      { test: 'B6.6', quote: 'Gap widens to 14.6 pp' },
    ],
    matters: 'It sets the bar: a plain tuned optimiser is the honest comparison, and MetaOptimize does not beat it.',
    glance: 'A plain, well-tuned SGD beats MetaOptimize by 1.8 to 14.6 points.',
    tests: ['B6.2', 'B6.3', 'B6.4', 'B6.5', 'B6.6'],
  },
  {
    id: 'pull', group: 'negative',
    claim: 'Pulling each layer’s step size toward one shared value does not help.',
    facts: [
      { label: 'Ruled out by 6 of 6 tests', source: { partCount: { part: 'B3', status: 'r', n: 6 } } },
      { label: 'Full pooling lost 3.0 points', source: { test: 'B3.1', quote: 'true pooling -3.0 pp' } },
      { label: 'A statistics-based pull gained +0.04 points, against a bar of 1.16', source: { test: 'B3.6', quote: '+0.04 pp; the bar was 1.16' } },
    ],
    holds: 'for what was tested.',
    caveat: 'That covers other networks (ResNet-10, ResNet-34), CIFAR-100, and the proposal itself in 42 runs.',
    holdsSources: [
      { test: 'B3.4', quote: 'ResNet-10, ResNet-34' },
      { test: 'B3.5', quote: 'tested directly in 42 runs' },
    ],
    matters: 'It closes off the original proposal: plain per-layer step sizes beat every partial pull.',
    glance: 'Pulling step sizes toward a shared value failed in 6 of 6 tests.',
    tests: ['B3.1', 'B3.2', 'B3.3', 'B3.4', 'B3.5', 'B3.6'],
    link: { to: '/main-idea', label: 'The main idea: why it lost' },
  },
  {
    id: 'tuning', group: 'negative',
    claim: 'The early lead for finer groups was mostly tuning.',
    facts: [
      { label: '+3.3 points shrank to +0.6 once each side was tuned', source: { test: 'B1.2', quote: '+3.3 pp shrinks to +0.6 once tuned' } },
      { label: 'About 81–83 % of the apparent gain was tuning', source: { test: 'B1.2', quote: 'About 81-83 % of the apparent gain was really tuning' } },
    ],
    holds: 'for SGD with momentum.',
    caveat: 'That is where the lead appeared; with AdamW it did not help.',
    holdsSources: [{ part: 'B1', quote: 'Helps with SGD+momentum, not AdamW; much of the gain was tuning' }],
    matters: 'It shows why each side of a comparison needs its own tuning.',
    glance: 'The early lead for finer groups was mostly tuning (81–83 %).',
    tests: ['B1.2'],
  },
  {
    id: 'review', group: 'checked',
    claim: 'An adversarial review tested the early headline claims against the project’s own control runs.',
    facts: [
      { label: 'Most early headlines were refuted', source: { part: 'B2', quote: 'Most early headlines refuted' } },
      { label: 'Scoring by best epoch inflated results by about 0.36 points', source: { test: 'B2.3', quote: 'Inflates ~0.36 pp' } },
    ],
    holds: 'for the claims it checked, which Part 2 lists.',
    caveat: 'New metric rules followed, such as scoring by the average of the last five epochs.',
    holdsSources: [
      { part: 'B2', quote: 'New metric rules followed' },
      { test: 'B2.3', quote: 'use last-5-epoch average' },
    ],
    matters: 'Weak early claims were replaced by stricter scoring rules.',
    tests: ['B2.1', 'B2.2', 'B2.3', 'B2.4'], parts: ['B2'],
  },
  {
    id: 'bars', group: 'checked',
    claim: 'Some bars were written down before the runs, and the results were held to them.',
    facts: [
      { label: 'A pre-set bar of 1.16 against a gain of +0.04 points', source: { test: 'B3.6', quote: '+0.04 pp; the bar was 1.16' } },
      { label: 'The first candidate paper was closed by its own pre-set gate', source: { test: 'B10.1', quote: 'Closed by its own gate' } },
    ],
    holds: 'for the two tests the record marks as pre-set.',
    caveat: 'The chance put on the closed paper was a judgement, not a measurement.',
    holdsSources: [{ test: 'B10.1', quote: '(a judgement)' }],
    matters: 'A bar fixed before the run stops a result from being judged after the fact.',
    tests: ['B3.6', 'B10.1'],
  },
  {
    id: 'corrections', group: 'checked',
    claim: 'The project’s record corrects itself in the open.',
    facts: [
      { label: '28 self-corrections, 2 withdrawn headlines, 7 batches killed before any GPU time', source: { test: 'B2.5', quote: '28 self-corrections, 2 withdrawn headlines and 7 batches killed before using any GPU time' } },
    ],
    holds: 'as counted from the record.',
    caveat: 'One list names only 6 of the 7 killed batches, and whether the process merits a methods paper is not certain.',
    holdsSources: [
      { test: 'B2.5', quote: 'one list names only 6' },
      { test: 'B2.5', quote: '(not certain)' },
    ],
    matters: 'Mistakes were found and fixed in the open rather than left standing.',
    tests: ['B2.5'],
  },
  {
    id: 'fair', group: 'checked',
    claim: 'Comparisons gave each side its own tuning, and the rival got a ladder of learning rates.',
    facts: [
      { label: 'With one meta step size for both, the lead looked like 3.3 points; tuned per side it was +0.6', source: { test: 'B1.2', quote: '+3.3 pp shrinks to +0.6 once tuned' } },
      { label: 'Once the meta step size was tuned, plain per-layer beat pooling by 0.4 points', source: { test: 'B3.2', quote: 'Plain layerwise beats pooling by 0.4 pp' } },
      { label: 'The rival ran as a ladder of six learning rates', source: { test: 'B6.4', quote: 'a plain SGD ladder of six learning rates' } },
    ],
    holds: 'for the comparisons that tuned both sides.',
    caveat: 'The untuned ones are the ones that inflated gains.',
    holdsSources: [{ test: 'B1.2', quote: 'when both used the same meta step size' }],
    matters: 'It keeps a comparison from favouring the side that got more tuning.',
    tests: ['B1.2', 'B3.2', 'B6.4'],
  },
  {
    id: 'code', group: 'built',
    claim: 'The finer step-size groupings were written, because the parent code never implemented them.',
    facts: [
      { label: 'Per-layer, per-node and per-weight groupings written first', source: { test: 'B1.0', quote: 'never actually implemented the finer groupings' } },
      { label: 'Two defects fixed along the way', source: { test: 'B1.0', quote: 'It also fixed two defects' } },
    ],
    holds: 'as tested: the code passed its correctness tests.',
    holdsSources: [{ test: 'B1.0', quote: 'passed its correctness tests' }],
    matters: 'The finer-grouping tests in Part 1 rest on this code.',
    tests: ['B1.0'],
  },
];

/** Counts taken from the notebook's own record (public/data/research.json, meta.stats). */
export interface NotebookStats { runs: number; experiments: number; figures: number }

export const notebookCard = {
  id: 'notebook',
  claim: 'A notebook that keeps the whole record in one place.',
  holdsText: 'as of the notebook’s snapshot date; the counts are those the record holds, not a claim about anything newer.',
  links: [
    { to: '/runs', label: 'Run ledger' },
    { to: '/experiments', label: 'Experiment pages' },
    { to: '/figures', label: 'Figures & tables' },
  ],
};

export function sourceText(source: Source, data: IdeaTreeData = ideaTree): string {
  if ('partCount' in source) throw new Error('A count has no text');
  if ('test' in source) {
    const node = findNode(data, source.test);
    if (!node || node.kind !== 'leaf') throw new Error(`Unknown test ${source.test}`);
    return `${node.leaf.title} | ${node.leaf.result} | ${node.leaf.summary}`;
  }
  const node = findNode(data, source.part);
  if (!node || node.kind !== 'part') throw new Error(`Unknown part ${source.part}`);
  return `${node.part.title} | ${node.part.verdict} | ${node.part.summary}`;
}

/** True when the source really says what the card says. */
export function sourceHolds(source: Source, data: IdeaTreeData = ideaTree): boolean {
  if ('partCount' in source) {
    const node = findNode(data, source.partCount.part);
    if (!node || node.kind !== 'part') return false;
    return node.part.leaves.length === source.partCount.n && node.part.leaves.every(l => l.status === source.partCount.status);
  }
  return sourceText(source, data).includes(source.quote);
}

/** A finding or a negative result is a key result for the paper when a test it rests on is one. The "checked" and "built" cards are not results. */
export function isKeyAchievement(card: Achievement, data: IdeaTreeData = ideaTree): boolean {
  return (card.group === 'stand' || card.group === 'negative') && card.tests.some(id => { const n = findNode(data, id); return n?.kind === 'leaf' && !!n.leaf.key; });
}

/** One status when the tests agree, mixed (a) when they differ. */
export function achievementStatus(card: Achievement, data: IdeaTreeData = ideaTree): TreeStatus {
  const statuses = new Set<TreeStatus>();
  for (const id of card.tests) {
    const node = findNode(data, id);
    if (!node || node.kind !== 'leaf') throw new Error(`Unknown test ${id}`);
    statuses.add(node.leaf.status);
  }
  return statuses.size === 1 ? [...statuses][0] : 'a';
}
