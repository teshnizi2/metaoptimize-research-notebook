/** Plain-word meanings for terms the "What we achieved" page uses. A term is explained once, at its first use on the page. */
export interface Term { id: string; pattern: RegExp; meaning: string }

// Longer terms first, so "meta step size" is matched before "step size".
export const terms: Term[] = [
  { id: 'meta-step-size', pattern: /meta step sizes?/i, meaning: 'The step size of the learner that adjusts the step sizes.' },
  { id: 'step-size', pattern: /step[- ]sizes?/i, meaning: 'How far the optimiser moves the weights at each update, also called the learning rate. MetaOptimize learns one per group of weights.' },
  { id: 'batchnorm', pattern: /BatchNorm/, meaning: 'A layer that rescales each channel’s values. Each channel has its own learnable scale number.' },
  { id: 'weight-decay', pattern: /weight decay|decay/i, meaning: 'A small pull that keeps the weights from growing during training.' },
  { id: 'sgd', pattern: /\bSGD\b/, meaning: 'Stochastic gradient descent: the plain way to train a network.' },
  { id: 'momentum', pattern: /\bmomentum\b/, meaning: 'Keeps part of the previous update, which smooths training.' },
  { id: 'adamw', pattern: /AdamW/, meaning: 'A common optimiser that adapts its update to each weight.' },
  { id: 'cosine', pattern: /cosine schedule/, meaning: 'A learning rate that shrinks smoothly along a cosine curve.' },
  { id: 'cifar', pattern: /CIFAR-100?/, meaning: 'Standard sets of small photos: CIFAR-10 has 10 classes, CIFAR-100 has 100.' },
  { id: 'epoch', pattern: /\bepochs?\b/, meaning: 'One epoch is one full pass over the training images.' },
  { id: 'pooling', pattern: /\bpooling\b/, meaning: 'Here: pulling the step sizes of different layers toward one shared value.' },
];

export interface Piece { text: string; term?: Term }

/** Splits a text into plain pieces and pieces that carry a meaning: only the terms in `allowed`, and only their first match in this text. */
export function gloss(text: string, allowed: ReadonlySet<string>): Piece[] {
  const hits: { at: number; end: number; term: Term }[] = [];
  for (const term of terms) {
    if (!allowed.has(term.id)) continue;
    const m = term.pattern.exec(text);
    if (m) hits.push({ at: m.index, end: m.index + m[0].length, term });
  }
  hits.sort((a, b) => a.at - b.at);
  const out: Piece[] = [];
  let at = 0;
  for (const h of hits) {
    if (h.at < at) continue;
    if (h.at > at) out.push({ text: text.slice(at, h.at) });
    out.push({ text: text.slice(h.at, h.end), term: h.term });
    at = h.end;
  }
  if (at < text.length) out.push({ text: text.slice(at) });
  return out;
}

/** For texts in reading order, which terms to explain in each: the first text that uses a term explains it, no later one does. */
export function firstUses(texts: { key: string; text: string }[]): Map<string, Set<string>> {
  const seen = new Set<string>();
  const plan = new Map<string, Set<string>>();
  for (const { key, text } of texts) {
    const here = new Set<string>();
    // Same overlap rule as `gloss`: take terms in table order, skip one that sits inside an earlier match.
    const taken: [number, number][] = [];
    for (const term of terms) {
      const m = term.pattern.exec(text);
      if (!m) continue;
      const span: [number, number] = [m.index, m.index + m[0].length];
      if (taken.some(([a, b]) => span[0] < b && span[1] > a)) continue;
      taken.push(span);
      if (!seen.has(term.id)) { seen.add(term.id); here.add(term.id); }
    }
    plan.set(key, here);
  }
  return plan;
}
