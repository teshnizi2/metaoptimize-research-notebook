import ideaTreeSource from '../../content/idea-tree.json';

/** g = worked / held up, r = did not work, a = mixed or partly, u = open or not certain. */
export type TreeStatus = 'g' | 'r' | 'a' | 'u';

export interface TreeLeaf {
  id: string;
  title: string;
  result: string;
  status: TreeStatus;
  summary: string;
  date: string;
  /** Where the date comes from: the written record, the recorded runs, or the whole part. */
  dateBasis: 'primer' | 'runs' | 'part';
  /** Experiment pages of this test (MT numbers); empty when the notebook has none. */
  experiments: string[];
  /** Id of the test this one followed up, in the same part; set only where the written record says so. */
  parent?: string;
  /** One short line on why this test followed its parent. */
  parentNote?: string;
}

export interface TreePart {
  id: string;
  number: number;
  title: string;
  dates: string;
  status: TreeStatus;
  verdict: string;
  summary: string;
  leaves: TreeLeaf[];
}

export interface IdeaTreeData {
  root: { title: string; summary: string };
  parts: TreePart[];
}

export const ideaTree = ideaTreeSource as IdeaTreeData;

export const statusInfo: Record<TreeStatus, { icon: string; name: string; short: string }> = {
  g: { icon: '✓', name: 'Worked / held up', short: 'Worked' },
  r: { icon: '✗', name: 'Did not work / refuted', short: 'Failed' },
  a: { icon: '~', name: 'Mixed or only partly', short: 'Mixed' },
  u: { icon: '?', name: 'Open, not tested or not certain', short: 'Open' },
};

export const statusOrder: TreeStatus[] = ['g', 'r', 'a', 'u'];

export type TreeNode =
  | { kind: 'root'; id: 'root'; title: string; summary: string }
  | { kind: 'part'; id: string; part: TreePart }
  | { kind: 'leaf'; id: string; part: TreePart; leaf: TreeLeaf };

export function findNode(data: IdeaTreeData, id: string | null): TreeNode | null {
  if (!id) return null;
  if (id === 'root') return { kind: 'root', id: 'root', title: data.root.title, summary: data.root.summary };
  for (const part of data.parts) {
    if (part.id === id) return { kind: 'part', id, part };
    const leaf = part.leaves.find(l => l.id === id);
    if (leaf) return { kind: 'leaf', id, part, leaf };
  }
  return null;
}

export function countStatuses(leaves: { status: TreeStatus }[]): Record<TreeStatus, number> {
  const counts: Record<TreeStatus, number> = { g: 0, r: 0, a: 0, u: 0 };
  for (const l of leaves) counts[l.status] += 1;
  return counts;
}

/** Number of sentences in a summary, counted on end punctuation followed by a space or the end. */
export function sentenceCount(text: string): number {
  return (text.trim().match(/[.!?](?=\s|$)/g) || []).length;
}
