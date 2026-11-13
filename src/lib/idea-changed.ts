import { findNode, type IdeaTreeData, type TreeStatus } from './idea-tree';

/** One value or comparison of a setting. `tests` are idea-tree test ids; a chip with none is only a setting, with no verdict. */
export interface ChangedChip {
  label: string;
  /** Two to four words on what came out, using numbers the tests record. */
  outcome?: string;
  tests: string[];
}

export interface ChangedRow {
  title: string;
  question: string;
  chips: ChangedChip[];
  /** Shows a link to the page about the main idea under the row. */
  mainIdea?: boolean;
}

/** The ten settings the tests varied. Chip colours are not written here: they come from the linked tests (see chipStatus). */
export const changedRows: ChangedRow[] = [
  {
    title: 'How many step sizes',
    question: 'Does giving each part of the network its own step size help?',
    chips: [
      { label: '1, shared by all', tests: [] },
      { label: '6 blocks', outcome: '+3.5 with momentum, tie on AdamW', tests: ['B1.1'] },
      { label: '62, one per layer', outcome: '+3.3, only +0.6 once tuned', tests: ['B1.2'] },
      { label: '~14,400, one per channel', outcome: 'lead fades to a tie', tests: ['B1.4'] },
    ],
  },
  {
    title: 'How to cut, at the same count',
    question: 'Does the way we cut matter at the same count?',
    chips: [
      { label: 'By channel', tests: [] },
      { label: 'Equal chunks (777 weights)', outcome: 'chunks win, 20 of 20', tests: ['B5.1', 'B5.2', 'B5.4'] },
      { label: 'Shuffled channels', outcome: 'no change, still open', tests: ['B5.3'] },
    ],
  },
  {
    title: 'Pull toward one step size',
    question: 'Does pulling each step size toward a shared one help?',
    mainIdea: true,
    chips: [
      { label: 'Dial r = 1, no pull', tests: [] },
      { label: 'Full pull (r = 0)', outcome: 'lost 3.0 points', tests: ['B3.1'] },
      { label: 'Partial pulls, 42 runs', outcome: 'trailed by 0.25 to 0.82', tests: ['B3.5'] },
      { label: 'Other nets, CIFAR-100', outcome: 'no sweet spot, 9 %', tests: ['B3.4'] },
      { label: 'Statistics-based pull', outcome: '+0.04 against a 1.16 bar', tests: ['B3.6'] },
    ],
  },
  {
    title: 'Optimiser',
    question: 'Does the optimiser underneath change the picture?',
    chips: [
      { label: 'SGD + momentum', outcome: 'six groups won by 3.5', tests: ['B1.1'] },
      { label: 'AdamW', outcome: 'no collapse, test stopped', tests: ['B8.6'] },
      { label: 'Rival: tuned plain SGD + cosine', outcome: 'rival wins every time', tests: ['B6.2', 'B6.3', 'B6.4', 'B6.5'] },
    ],
  },
  {
    title: 'Meta step size',
    question: 'Does tuning the meta step size change the verdict?',
    chips: [
      { label: 'One shared by both arms', outcome: '+3.3 shrinks to +0.6', tests: ['B1.2'] },
      { label: 'The pull dial also cuts it ~10×', outcome: 'not a pure pull', tests: ['B3.1'] },
      { label: 'Tuned per arm (best ≈ 1/10)', outcome: 'plain per-layer wins', tests: ['B3.2'] },
    ],
  },
  {
    title: 'Starting step size',
    question: 'Did the idea only help a learner that started badly?',
    chips: [
      { label: '1e-6, crippled start', tests: [] },
      { label: '1e-3, sane start', outcome: 'gain +1.0 down to +0.24', tests: ['B3.3'] },
      { label: '100,000× range', outcome: 'steady, but no win', tests: ['B4.7'] },
    ],
  },
  {
    title: 'Weight decay',
    question: 'Does the strength of weight decay change the result?',
    chips: [
      { label: '0.1 and 0.01', outcome: 'audit lead holds', tests: ['B9.1'] },
      { label: '5e-4, normal', outcome: 'collapse gone, order flips', tests: ['B8.5'] },
      { label: 'Off entirely', outcome: 'collapse goes away', tests: ['B8.1'] },
      { label: 'Off on 20 BatchNorm scales', outcome: 'collapse vanishes', tests: ['B8.2'] },
      { label: 'Off on 3 BatchNorm scales', outcome: 'three do as much', tests: ['B8.3'] },
    ],
  },
  {
    title: 'Network',
    question: 'Does it depend on the network?',
    chips: [
      { label: 'ResNet-10 / 18 / 34 / 50', outcome: 'gain shrinks with size', tests: ['B1.3'] },
      { label: 'VGG11, GroupNorm, PlainNet', outcome: 'gap appears on all', tests: ['B7.8'] },
      { label: 'Pull on other networks', outcome: 'no sweet spot', tests: ['B3.4'] },
    ],
  },
  {
    title: 'Data',
    question: 'Does it depend on the dataset or augmentation?',
    chips: [
      { label: 'CIFAR-10', outcome: 'rival wins by 1.8', tests: ['B6.2'] },
      { label: 'CIFAR-100', outcome: 'rival wins by 5.7', tests: ['B6.4'] },
      { label: 'No augmentation', outcome: 'rival wins by 3.6 and 12.2', tests: ['B6.5'] },
      { label: 'CIFAR-10, collapse check', outcome: 'story holds', tests: ['B7.11'] },
      { label: 'Audit on CIFAR-100', outcome: 'holds at 0.1, unread at 5e-4', tests: ['B9.6'] },
    ],
  },
  {
    title: 'Training length',
    question: 'Does it change with how long we train?',
    chips: [
      { label: '10 vs 40 epochs', outcome: 'flip was an artefact', tests: ['B1.4'] },
      { label: '100 vs 300 epochs', outcome: 'gains only shrink', tests: ['B2.4'] },
      { label: '100 · 300 · 600 epochs', outcome: 'gap stays near 1.9', tests: ['B6.3'] },
      { label: '772 epochs', outcome: 'cliff shrinks, survives', tests: ['B7.3'] },
    ],
  },
];

/** The colour of a chip comes only from its tests: one status when they agree, mixed when they differ, plain "setting" when it has none. */
export function chipStatus(chip: ChangedChip, data: IdeaTreeData): TreeStatus | 'setting' {
  if (chip.tests.length === 0) return 'setting';
  const statuses = new Set<TreeStatus>();
  for (const id of chip.tests) {
    const node = findNode(data, id);
    if (!node || node.kind !== 'leaf') throw new Error(`Unknown test ${id}`);
    statuses.add(node.leaf.status);
  }
  return statuses.size === 1 ? [...statuses][0] : 'a';
}
