/** Where the Learn page finds its media. This is the one place to change when the files move.
 *  /learn-files/* is rewritten (vercel.json) to api/learn-file.ts, which reads a private Blob store behind the Vercel sign-in.
 *  `available: false` shows a "coming soon" box instead of a player. */
export const LEARN_FILES = {
  base: '/learn-files',
  available: true,
};

export const videoUrl = (id: string) => `${LEARN_FILES.base}/video/${id}.mp4`;
export const subtitleUrl = (id: string) => `${LEARN_FILES.base}/subs/${id}.vtt`;
export const pdfUrl = (name: 'guide' | 'glossary', page?: number) => `${LEARN_FILES.base}/pdf/${name}.pdf${page ? `#page=${page}` : ''}`;

export type LearnSeries = 'beginner' | 'first';

export interface LearnVideo {
  id: string;
  series: LearnSeries;
  title: string;
  /** One line: what you will learn. */
  learn: string;
  seconds: number;
}

/** Videos shown on the site. Four others (the PaperFactory review paper and the status videos that depend on it) are left out by rule. */
export const learnVideos: LearnVideo[] = [
  { id: 'B01', series: 'beginner', title: 'Learning the step size while you train', seconds: 302, learn: 'How a computer can learn its own step size while it trains, told as a walker in a valley.' },
  { id: 'B02', series: 'beginner', title: 'The loop: step, diary, vote', seconds: 249, learn: 'The three-part loop: take a stride, note it in a diary, and let the diary and the slope vote on whether it was too long.' },
  { id: 'B03', series: 'beginner', title: 'The box: a guard rail for β', seconds: 255, learn: 'Why the log of the step size needs a guard rail, and what happens when nothing stops it falling.' },
  { id: 'B04', series: 'beginner', title: 'How many step sizes: from 1 to 11 million', seconds: 242, learn: 'How many step sizes a network can have, from one for everything to one for each weight.' },
  { id: 'B05', series: 'beginner', title: 'The proposal: rubber bands to the master', seconds: 235, learn: 'The proposal: a step size for each part, pulled back toward one shared master step size.' },
  { id: 'B06', series: 'beginner', title: 'How the idea died: fair tuning and a hidden dial', seconds: 306, learn: 'Why the first big win was not a fair race, and what a hidden dial did to the result.' },
  { id: 'B07', series: 'beginner', title: 'Same count, different cut', seconds: 274, learn: 'Two ways to cut the weights into the same number of groups, and the question of whether the cut matters.' },
  { id: 'B08', series: 'beginner', title: 'Twenty of twenty, and nobody knows why', seconds: 275, learn: 'Equal chunks beat structure-based groups in 20 of 20 tries, and why the cause is still unknown.' },
  { id: 'B09', series: 'beginner', title: 'Against a plain tuned rival', seconds: 319, learn: 'What happens when the method finally races a plain, well-tuned optimiser.' },
  { id: 'B10', series: 'beginner', title: 'A decay that grows with the step size', seconds: 253, learn: 'Why the pull toward zero (weight decay) gets stronger as the step size grows.' },
  { id: 'B11', series: 'beginner', title: 'The collapse: three tiny tensors', seconds: 337, learn: 'Which three small blocks of numbers carry the collapse at heavy weight decay.' },
  { id: 'B12', series: 'beginner', title: 'What ran is what counts', seconds: 283, learn: 'How the project checks that every run really used the setting its label says.' },
  { id: 'B13', series: 'beginner', title: 'Three result words, and why a half-width isn’t a limit', seconds: 250, learn: 'How to read a noisy result: noise, standard error, intervals, and the three words the project uses for outcomes.' },
  { id: 'B14', series: 'beginner', title: 'A diary that corrects itself', seconds: 210, learn: 'Why the project’s record is only ever corrected by adding lines, and whether you can trust it.' },
  { id: 'B15', series: 'beginner', title: 'What may be said at two decays', seconds: 262, learn: 'Which claims the record allows at the heavy decay (0.1) and at the usual one (5e-4).' },
  { id: 'B16', series: 'beginner', title: 'The scalar row, the grid edge, and where to publish', seconds: 296, learn: 'The claim that one shared step size is as good as many, the edge of the search grid, and what that means for publishing.' },
  { id: '01', series: 'first', title: 'What MetaOptimize does', seconds: 196, learn: 'How MetaOptimize learns step sizes during training, and why a floor on β is needed.' },
  { id: '02', series: 'first', title: 'The idea that started it', seconds: 190, learn: 'What Hierarchical MetaOptimize proposed, how the parent paper led to it, and why the project was left with no method.' },
  { id: '03', series: 'first', title: 'Does the partition matter', seconds: 215, learn: 'The count-matched audit: positive in 20 of 20 cells, small, and still unexplained.' },
  { id: '04', series: 'first', title: 'Against a plain tuned optimiser', seconds: 183, learn: 'Why a tuned SGD with a cosine schedule beats MetaOptimize by 1.8 to 14.6 points, and why that frames everything else.' },
  { id: '05', series: 'first', title: 'The collapse and weight decay', seconds: 203, learn: 'Why one shared step size collapses on CIFAR-100 at decay 0.1, and why it is gone at the standard decay.' },
  { id: '06', series: 'first', title: 'Keeping the work honest', seconds: 224, learn: 'The record’s honesty rules and the three result words: resolved, undecided and unreadable.' },
  { id: '07', series: 'first', title: 'What can and cannot be claimed', seconds: 213, learn: 'Which claims the record allows at each decay, and why the scalar row ended as an interval.' },
];

/** Not shown on the site, by rule: they cover the PaperFactory review paper and the status videos built on it. */
export const leftOutVideos = ['B17', 'B18', 'First 08', 'First 09'] as const;

export const seriesTitle: Record<LearnSeries, string> = {
  beginner: 'Beginner series',
  first: 'First series',
};

export const videosOf = (series: LearnSeries) => learnVideos.filter(v => v.series === series);
export const videoLabel = (v: LearnVideo) => v.series === 'beginner' ? v.id : `First ${v.id}`;

export function formatLength(seconds: number): string {
  const m = Math.floor(seconds / 60), s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

export type StudyStep =
  | { kind: 'guide'; title: string; page: number; minutes: number; youtube?: { url: string; label: string } }
  | { kind: 'video'; videoId: string; minutes: number };

const yt = (id: string, label: string) => ({ url: `https://www.youtube.com/watch?v=${id}`, label });

/** The study path, in the order of the study list: guide chapters and our videos, one row at a time. */
export const studyPath: StudyStep[] = [
  { kind: 'guide', title: 'Start here and Before you begin', page: 2, minutes: 10 },
  { kind: 'guide', title: 'A1 What a neural network is', page: 5, minutes: 25, youtube: yt('aircAruvnKk', '3Blue1Brown video, 19 min') },
  { kind: 'guide', title: 'A2 Loss, accuracy and the slope', page: 10, minutes: 25, youtube: yt('IHZwWFHWa-w', '3Blue1Brown video, 21 min') },
  { kind: 'guide', title: 'A3 One step, and the step size', page: 14, minutes: 35, youtube: yt('sDv4f4s2SB8', 'StatQuest video, 24 min') },
  { kind: 'video', videoId: 'B01', minutes: 5 },
  { kind: 'guide', title: 'A4 Optimisers: SGD, momentum, Adam and Lion', page: 20, minutes: 25, youtube: yt('k8fTYJPd3_I', 'Andrew Ng video, 9 min') },
  { kind: 'guide', title: 'A5 Weight decay: an elastic band to zero', page: 26, minutes: 25, youtube: yt('6g0t3Phly2M', 'Andrew Ng video, 10 min') },
  { kind: 'guide', title: 'A6 Schedules, tuning and a fair rival', page: 31, minutes: 20, youtube: yt('AXDByU3D1hA', 'Andrew Ng video, 7 min') },
  { kind: 'guide', title: 'A7 Inside ResNet-18: BatchNorm', page: 36, minutes: 20, youtube: yt('tNIpEZLv_eg', 'Andrew Ng video, 9 min') },
  { kind: 'guide', title: 'A8 Logs and e', page: 41, minutes: 15, youtube: yt('mQTWzLpCcW0', 'Khan Academy video, 10 min') },
  { kind: 'guide', title: 'A9 Is the gap real? Runs, seeds and noise', page: 46, minutes: 25, youtube: yt('A82brFpdr9g', 'StatQuest video, 3 min') },
  ...['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16']
    .map(videoId => ({ kind: 'video', videoId, minutes: Math.round(learnVideos.find(v => v.id === videoId)!.seconds / 6) / 10 } as StudyStep)),
];

/** Pages of the site the Learn page points to. */
export const mapLinks = [
  { to: '/idea-tree?view=radial', label: 'Idea tree, radial', detail: 'Every idea the project tested, with the main question in the centre.' },
  { to: '/idea-tree?view=tree', label: 'Idea tree, left to right', detail: 'The same tree laid out as branches, easier on a phone.' },
  { to: '/idea-tree?view=list', label: 'Idea tree, list', detail: 'The same ideas in the order they happened.' },
  { to: '/idea-tree?view=changed', label: 'What we changed', detail: 'Ten settings the tests varied, coloured by whether they worked.' },
  { to: '/main-idea', label: 'The main idea: why it lost', detail: 'What the proposal was, the charts, and our best reading of why it failed.' },
] as const;
