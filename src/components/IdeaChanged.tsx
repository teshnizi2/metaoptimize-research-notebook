import { ArrowRight } from 'lucide-react';
import { findNode, ideaTree, statusInfo, statusOrder, type IdeaTreeData } from '../lib/idea-tree';
import { changedRows, chipStatus } from '../lib/idea-changed';
import { KeyLegendItem, KeyStar, MAIN_IDEA_PAGE, TreeIcon, type TestLink } from './IdeaTreeDetail';
import './IdeaTree.css';
import './IdeaChanged.css';

/** "What we changed": the settings the tests varied, one row each, with every value or comparison coloured by the tests behind it. */
export function IdeaChanged({ data = ideaTree, onOpen, testLink }: {
  data?: IdeaTreeData;
  /** Links the pull row to the page about the main idea. */
  testLink: TestLink;
  /** Called with a test id when a chip is chosen; the page shows that test in the tree. */
  onOpen: (id: string) => void;
}) {
  return <div className="ic">
    <p className="ic-top">The audit (row 2) is the main result; most of it ran at weight decay 0.1.</p>
    <ul className="ic-legend" aria-label="Colours">
      {statusOrder.map(s => <li key={s} className={`tree-${s}`}><TreeIcon status={s}/>{statusInfo[s].short}</li>)}
      <KeyLegendItem/>
      <li className="ic-legend-setting"><span className="ic-dash" aria-hidden="true"/>Setting only, no test of its own</li>
    </ul>
    <p className="ic-hint">Choose a coloured box to see that test in the tree.</p>
    <ol className="ic-rows">
      {changedRows.map((row, i) => <li key={row.title} className="ic-row">
        <div className="ic-head">
          <span className="ic-num" aria-hidden="true">{i + 1}</span>
          <div>
            <h2>{row.title}</h2>
            <p>{row.question}</p>
          </div>
        </div>
        <ul className="ic-chips">
          {row.chips.map(chip => {
            const status = chipStatus(chip, data);
            if (status === 'setting') {
              return <li key={chip.label}><span className="ic-chip ic-setting">
                <span className="ic-label">{chip.label}</span>
                <span className="ic-outcome">setting</span>
              </span></li>;
            }
            const isKey = chip.tests.some(id => { const n = findNode(data, id); return n?.kind === 'leaf' && n.leaf.key; });
            return <li key={chip.label}><button type="button" className={`ic-chip tree-${status}${isKey ? ' tree-key' : ''}`} data-status={status} onClick={() => onOpen(chip.tests[0])}
              aria-label={`${chip.label}: ${chip.outcome}. ${statusInfo[status].name}.${isKey ? ' Key result for the paper.' : ''} Open ${chip.tests.join(', ')} in the tree.`}>
              <TreeIcon status={status}/>
              <span className="ic-text">
                <span className="ic-label">{chip.label}{isKey && <> <KeyStar/></>}</span>
                <span className="ic-outcome">{chip.outcome}</span>
                <span className="ic-ids">{chip.tests.join(' · ')}</span>
              </span>
            </button></li>;
          })}
        </ul>
        {row.mainIdea && <p className="ic-more">{testLink(MAIN_IDEA_PAGE, 'text-link', <>The main idea: why it lost<ArrowRight size={16}/></>)}</p>}
      </li>)}
    </ol>
  </div>;
}
