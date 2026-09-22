import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import type { Experiment } from '../types';
import { experimentQuestion } from '../lib/experiment-copy';
import { laterEvidence, laterEvidenceFor, laterEvidenceLabels } from '../lib/later-evidence';
import type { LaterEvidenceRelation } from '../lib/later-evidence';

function RelationItem({relation,side,other}:{relation:LaterEvidenceRelation;side:'earlier'|'later';other?:Experiment}){
 const id=side==='earlier'?relation.later:relation.earlier;
 return <li className="later-evidence-item">
  <div className="later-evidence-head"><span className={`later-evidence-relation relation-${relation.relation}`}>{laterEvidenceLabels[relation.relation][side]}</span><Link className="record-id" to={`/experiments/${id}`}>{id}<ArrowUpRight size={14}/></Link>{other&&<span className="later-evidence-question">{experimentQuestion(other)}</span>}</div>
  <p>{relation.reason}</p>
  <blockquote>“{relation.quote}”<cite>{relation.source}</cite></blockquote>
 </li>;
}

/**
 * Curated cross-links between records (content/later-evidence.json). Neither record's registered text changes: the note
 * tells a reader of an earlier record that a later one bears on it, and a reader of the later record what it bears on.
 */
export function LaterEvidence({experiment,experiments,relations=laterEvidence}:{experiment:Pick<Experiment,'id'>;experiments:Experiment[];relations?:LaterEvidenceRelation[]}){
 const {later,bearsOn}=laterEvidenceFor(experiment.id,relations);
 if(!later.length&&!bearsOn.length)return null;
 const byId=new Map(experiments.map(e=>[e.id,e]));
 return <>
  {later.length>0&&<section className="panel later-evidence" aria-labelledby="later-evidence-title">
   <div className="section-title"><h2 id="later-evidence-title">Later evidence</h2></div>
   <p className="small muted">Later records bear on this one. This record’s registered text is unchanged; read it together with them. Each link quotes the campaign’s own CORRECTIONS entry.</p>
   <ul>{later.map(r=><RelationItem key={`${r.later}-${r.relation}`} relation={r} side="earlier" other={byId.get(r.later)}/>)}</ul>
  </section>}
  {bearsOn.length>0&&<section className="panel later-evidence bears-on" aria-labelledby="bears-on-title">
   <div className="section-title"><h2 id="bears-on-title">Bears on</h2></div>
   <p className="small muted">This record bears on earlier ones, whose registered text is unchanged. Each link quotes the campaign’s own CORRECTIONS entry.</p>
   <ul>{bearsOn.map(r=><RelationItem key={`${r.earlier}-${r.relation}`} relation={r} side="later" other={byId.get(r.earlier)}/>)}</ul>
  </section>}
 </>;
}
