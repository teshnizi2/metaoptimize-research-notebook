/** Numbers for the "main idea" page. Every figure is copied from a recorded source: a test's summary in
 *  content/idea-tree.json (`quote` must appear there) or a recorded table under public/assets/tables (`csv`).
 *  tests/main-idea.test.ts checks each one against its source. */

export interface BarRow {
  label: string;
  value: number;
  /** Text shown beside the bar. */
  display: string;
  tone: 'claimed' | 'lost' | 'bar';
  /** Test whose summary holds this number, and the exact text that must appear in it. */
  test: string;
  quote: string;
}

export interface BarChartData {
  id: string;
  question: string;
  unit: string;
  rows: BarRow[];
  /** Optional reference line, e.g. the bar a result had to clear. */
  reference?: { value: number; label: string; test: string; quote: string };
  tests: string[];
  caption: string;
}

export const barCharts: BarChartData[] = [
  {
    id: 'early-lead',
    question: 'Did giving every layer its own step size help?',
    unit: 'points ahead of one overall step size',
    rows: [
      { label: 'Both sides share one meta step size', value: 3.3, display: '+3.3', tone: 'claimed', test: 'B1.2', quote: '3.3 percentage points' },
      { label: 'Each side gets its own best meta step size', value: 0.6, display: '+0.6', tone: 'lost', test: 'B1.2', quote: 'shrank to 0.6' },
    ],
    tests: ['B1.2'],
    caption: 'The early lead was mostly tuning: about 81–83 % of the apparent gain disappeared once each side was tuned fairly.',
  },
  {
    id: 'behind-plain',
    question: 'Did pulling toward one step size beat plain per-layer step sizes?',
    unit: 'points behind plain per-layer step sizes',
    rows: [
      { label: 'True full pull', value: -3.0, display: '−3.0', tone: 'lost', test: 'B3.1', quote: 'lost 3.0 percentage points' },
      { label: 'Best pooled version, meta step size tuned (92.4 % vs 92.9 %)', value: -0.4, display: '−0.4', tone: 'lost', test: 'B3.2', quote: 'by 0.4 percentage points' },
    ],
    tests: ['B3.1', 'B3.2'],
    caption: 'Full pull lost outright. After tuning the meta step size, plain per-layer step sizes still came out ahead, and pooling added another setting to tune.',
  },
  {
    id: 'start-gain',
    question: 'Was the gain from pulling still there with a sane starting step size?',
    unit: 'points gained from partial pulling',
    rows: [
      { label: 'Crippled start (1e-6)', value: 1.0, display: '+1.0', tone: 'claimed', test: 'B3.3', quote: '+1.0' },
      { label: 'Sane start (1e-3)', value: 0.24, display: '+0.24', tone: 'lost', test: 'B3.3', quote: '+0.24' },
    ],
    reference: { value: 0.5, label: 'bar for mattering: 0.5', test: 'B3.3', quote: '0.5-point bar' },
    tests: ['B3.3'],
    caption: 'About 76 % of the apparent gain went away, and what was left fell below the project’s own 0.5-point bar.',
  },
  {
    id: 'statistics-pull',
    question: 'Could a statistics-based pull save the idea without a dial?',
    unit: 'points of lift',
    rows: [
      { label: 'Lift from the statistics-based pull', value: 0.04, display: '+0.04', tone: 'lost', test: 'B3.6', quote: '0.04 points' },
      { label: 'Bar set in advance', value: 1.16, display: '1.16', tone: 'bar', test: 'B3.6', quote: '1.16' },
    ],
    tests: ['B3.6'],
    caption: 'The pull chosen from the data itself added almost nothing; the success bar was set before the run.',
  },
];

/** Accuracy against the dial r, read from recorded tables. The x axis lists the r values one after another (it is not to scale). */
export interface LineChartData {
  id: string;
  question: string;
  yLabel: string;
  xLabel: string;
  x: number[];
  series: { label: string; points: number[]; tone: 'a' | 'b' }[];
  /** A flat reference line, e.g. plain per-layer step sizes. */
  references: { label: string; value: number; tone: 'a' | 'b' }[];
  tests: string[];
  caption: string;
  /** Where the numbers come from. */
  csv: string;
  csvRows: { key: Record<string, string>; column: string }[];
}

const TL = 'public/assets/tables/measurement_and_mechanism/mechanism__two_level_shrinkage__arm_summary.csv';
const C100 = 'public/assets/tables/portfolio/panels/16_additive_transfer_curves__cells_plateau5.csv';

export const lineCharts: LineChartData[] = [
  {
    id: 'dial',
    question: 'Did any setting of the pull dial beat plain per-layer step sizes?',
    yLabel: 'CIFAR-10 test accuracy (%)',
    xLabel: 'dial r: 0 = full pull … 1 = no pull',
    x: [0, 0.03, 0.1, 0.3, 1],
    series: [
      { label: 'Start 1e-3', points: [87.96, 89.77, 90.36, 89.61, 90.62], tone: 'a' },
      { label: 'Start 1e-6', points: [87.93, 90.19, 90.75, 90.62, 90.67], tone: 'b' },
    ],
    references: [
      { label: 'Plain per-layer, start 1e-3', value: 91.18, tone: 'a' },
      { label: 'Plain per-layer, start 1e-6', value: 91.00, tone: 'b' },
    ],
    tests: ['B3.5'],
    caption: 'All 42 runs of the proposal itself: every point sits below its dashed line. The best partial pull trailed by 0.25 points at one start and 0.82 at the other.',
    csv: TL,
    csvRows: [
      { key: { arm: 'a3-r0' }, column: 'test_percent_mean' }, { key: { arm: 'a3-r003' }, column: 'test_percent_mean' },
      { key: { arm: 'a3-r01' }, column: 'test_percent_mean' }, { key: { arm: 'a3-r03' }, column: 'test_percent_mean' },
      { key: { arm: 'a3-r1' }, column: 'test_percent_mean' },
      { key: { arm: 'a6-r0' }, column: 'test_percent_mean' }, { key: { arm: 'a6-r003' }, column: 'test_percent_mean' },
      { key: { arm: 'a6-r01' }, column: 'test_percent_mean' }, { key: { arm: 'a6-r03' }, column: 'test_percent_mean' },
      { key: { arm: 'a6-r1' }, column: 'test_percent_mean' },
      { key: { arm: 'a3-plain' }, column: 'test_percent_mean' }, { key: { arm: 'a6-plain' }, column: 'test_percent_mean' },
    ],
  },
  {
    id: 'cifar100',
    question: 'Is there a sweet spot on CIFAR-100?',
    yLabel: 'CIFAR-100 test accuracy (%)',
    xLabel: 'dial r: 0 = full pull … 1 = no pull',
    x: [0, 0.03, 0.05, 0.07, 0.1, 0.2, 1],
    series: [{ label: 'ResNet-18, start 1e-3', points: [8.92, 13.78, 17.37, 26.65, 42.79, 68.52, 70.11], tone: 'a' }],
    references: [],
    tests: ['B3.4'],
    caption: 'With a sane start the curve climbs straight to the end: no sweet spot, and full pull falls to 9 %.',
    csv: C100,
    csvRows: [0, 0.03, 0.05, 0.07, 0.1, 0.2, 1].map(r => ({ key: { dataset: 'CIFAR100', alpha0: '1e-3', eta_ratio: String(r) }, column: 'accuracy_mean' })),
  },
];

export const mainIdeaIntro = [
  'MetaOptimize learns a step size while a network trains.',
  'The proposal, hierarchical MetaOptimize, gave every layer its own step size and tied them all to one shared master step size with a rubber band, so the layers could borrow strength from each other.',
  'A dial r sets how tight the rubber band is: r = 1 means no pull, r = 0 means full pull.',
  'The dial had a hidden second job: turning it also cut the meta step size, the speed at which the step sizes themselves are learned, by about 10 times.',
];

export interface Reason {
  title: string;
  text: string;
  /** Test or part ids that fit this reason, with a short note on what each shows. */
  evidence: { id: string; note: string }[];
}

/** Interpretation, in Reza's words, not a tested result. */
export const reasons: Reason[] = [
  {
    title: 'There was little noise to remove.',
    text: 'Pulling toward a shared value, called shrinkage, helps when each layer’s own value is a noisy guess. But each layer’s step size is learned over thousands of training steps, so it is already fairly settled. The statistics-based pull, built to remove exactly that noise, added only +0.04 points. That fits the idea that there was almost nothing to remove.',
    evidence: [{ id: 'B3.6', note: 'the statistics-based pull added +0.04 against a bar of 1.16' }],
  },
  {
    title: 'The pull mostly helped a learner that started badly.',
    text: 'From a tiny starting step size, the per-layer step sizes are slow to grow, and tying them to the master lets them borrow its progress. With a sensible start they don’t need that help, which is why most of the gain disappeared.',
    evidence: [{ id: 'B3.3', note: 'the gain fell from +1.0 to +0.24 once the start was sane' }],
  },
  {
    title: 'Layers really do need different step sizes.',
    text: 'The CIFAR-100 collapse shows that forcing one shared step size can be very harmful. Pulling toward the master pushes layers back toward that failure: full pull fell to 9 % on CIFAR-100. So “somewhere in between” didn’t land on a sweet spot. It just slid toward the worse end.',
    evidence: [
      { id: 'B3.4', note: 'full pull collapsed to 9 % on CIFAR-100, with no sweet spot' },
      { id: 'B7', note: 'Part 7 locates the CIFAR-100 collapse of one shared step size' },
    ],
  },
  {
    title: 'The engine underneath was weak.',
    text: 'MetaOptimize itself loses to a tuned plain optimiser by 1.8 to 14.6 points. Any layer-and-master scheme built on top of it starts from behind.',
    evidence: [
      { id: 'B6.2', note: 'loses by 1.8 points on CIFAR-10' },
      { id: 'B6.4', note: 'loses by 5.7 points on CIFAR-100' },
      { id: 'B6', note: 'Part 6 gathers the five deficits, 1.8 to 14.6 points' },
    ],
  },
];

/** The tests shown as verdict cards. */
export const mainIdeaTests = ['B1.2', 'B3.1', 'B3.2', 'B3.3', 'B3.4', 'B3.5', 'B3.6'];
