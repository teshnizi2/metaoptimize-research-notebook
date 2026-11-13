import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { PageHeading } from '../components/common';
import { TreeDetail, testPath, type TestLink } from '../components/IdeaTreeDetail';
import { BarChart, LineChart, TestChip } from '../components/MainIdeaCharts';
import { findNode, ideaTree } from '../lib/idea-tree';
import { barCharts, lineCharts, mainIdeaIntro, mainIdeaTests, reasons } from '../lib/main-idea';
import { useResearch } from '../data';
import '../components/IdeaTree.css';
import '../components/MainIdea.css';

const siteLink: TestLink = (id, className, children) => <Link className={className} to={`/${testPath(id)}`}>{children}</Link>;

/** Evidence ids are tests (B3.6) or whole parts (B7); both open in the idea tree. */
const inTree = (id: string) => `/idea-tree?view=radial&node=${id}`;

export function MainIdeaPage() {
  const research = useResearch();
  const pageIds = useMemo(() => new Set(research.experiments.map(e => e.id)), [research]);
  const charts = [
    ...barCharts.slice(0, 1).map(c => ({ key: c.id, tests: c.tests, question: c.question, caption: c.caption, body: <BarChart chart={c}/> })),
    ...barCharts.slice(1, 3).map(c => ({ key: c.id, tests: c.tests, question: c.question, caption: c.caption, body: <BarChart chart={c}/> })),
    ...lineCharts.slice(0, 1).map(c => ({ key: c.id, tests: c.tests, question: c.question, caption: c.caption, body: <LineChart chart={c}/> })),
    ...lineCharts.slice(1).map(c => ({ key: c.id, tests: c.tests, question: c.question, caption: c.caption, body: <LineChart chart={c}/> })),
    ...barCharts.slice(3).map(c => ({ key: c.id, tests: c.tests, question: c.question, caption: c.caption, body: <BarChart chart={c}/> })),
  ];
  return <div className="mi">
    <PageHeading
      eyebrow="Research map"
      title="The main idea: why it lost"
      description="The proposal that pulls each layer’s step size toward one shared master step size, how it failed in the tests, and our best reading of why."
    />
    <section className="mi-section" aria-labelledby="mi-what">
      <h2 id="mi-what">What the idea was</h2>
      <p className="mi-lead">{mainIdeaIntro.join(' ')}</p>
      <p className="mi-small">Part 3 of the <Link to={inTree('B3')}>idea tree</Link> lists the four tests that ended it, and the <Link to="/idea-tree?view=changed">what-we-changed view</Link> shows where the pull sits among everything else we varied.</p>
    </section>

    <section className="mi-section" aria-labelledby="mi-how">
      <h2 id="mi-how">How it died, in charts</h2>
      <p className="mi-small">Every number comes from a recorded test; each chart names its test. Red marks the result that went against the idea.</p>
      <div className="mi-charts">
        {charts.map(c => <article className="mi-card" key={c.key}>
          <h3>{c.question}</h3>
          {c.body}
          <p className="mi-caption">{c.caption}</p>
          <div className="mi-chips">{c.tests.map(id => <TestChip key={id} id={id}/>)}</div>
        </article>)}
      </div>
    </section>

    <section className="mi-section" aria-labelledby="mi-tests">
      <h2 id="mi-tests">The tests behind it</h2>
      <div className="mi-tests">
        {mainIdeaTests.map(id => <TreeDetail key={id} node={findNode(ideaTree, id)} inline pageIds={pageIds} testLink={siteLink}/>)}
      </div>
    </section>

    <section className="mi-section mi-why" aria-labelledby="mi-why">
      <h2 id="mi-why">Why we think it lost</h2>
      <p className="mi-tag">Interpretation, not a tested result</p>
      <ol className="mi-reasons">
        {reasons.map(r => <li key={r.title}>
          <h3>{r.title}</h3>
          <p>{r.text}</p>
          <p className="mi-evidence"><strong>Evidence that fits:</strong> {r.evidence.map((e, i) => <span key={e.id}>{i > 0 && '; '}<Link to={inTree(e.id)}>{e.id}</Link>, {e.note}</span>)}.</p>
        </li>)}
      </ol>
    </section>

    <p className="mi-close">What survived was a measuring tool, and the project then moved on to the count-matched audit. <Link className="text-link" to={inTree('B5')}>See Part 5 in the idea tree<ArrowRight size={16}/></Link> <Link className="text-link" to="/achievements">What we achieved<ArrowRight size={16}/></Link></p>
  </div>;
}
