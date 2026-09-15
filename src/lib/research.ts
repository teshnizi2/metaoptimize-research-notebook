import type { Experiment, Figure, Outcome, ResearchData, Run, ResearchWarning } from '../types';
export const outcomeLabels:Record<Outcome,string>={success:'Goal met',fail:'Goal missed',mixed:'Mixed',unresolved:'Open',correction:'Corrected'};
export const outcomeOrder:Outcome[]=['success','fail','mixed','unresolved','correction'];
export function filterExperiments(items:Experiment[],filters:{query?:string;area?:string;outcome?:string}={}) {
  const terms=(filters.query||'').toLowerCase().trim().split(/\s+/).filter(Boolean);
  return items.filter(e=>(!filters.area||e.area===filters.area)&&(!filters.outcome||e.outcome===filters.outcome)&&terms.every(term=>[e.id,e.title,e.goal,e.comparison,e.why,e.result,e.reason,e.scope,e.area,...e.batches].join(' ').toLowerCase().includes(term)));
}
export function outcomeCounts(items:Experiment[]):Record<Outcome,number>{return items.reduce((a,e)=>{a[e.outcome]++;return a;},{success:0,fail:0,mixed:0,unresolved:0,correction:0});}
export function resolveExperiment(items:Experiment[],id:string){return items.find(e=>e.id.toLowerCase()===id.toLowerCase());}
export function relatedFigures(items:Figure[],id:string){return items.filter(f=>f.experimentIds.includes(id));}
export function compactDate(value:string){const d=new Date(value);return Number.isNaN(d.getTime())?value:new Intl.DateTimeFormat('en-GB',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'}).format(d);}
export function safeHref(href:string){return href.startsWith('/')&&!href.startsWith('//')&&!/[\\\u0000-\u0020\u007f]/.test(href)?href:'/';}
export function notebookWarnings(data:Pick<ResearchData,'warnings'|'activity'>):ResearchWarning[]{
 return [...data.warnings,...(data.activity||[]).filter(e=>e.kind==='warning').flatMap(e=>e.experimentIds.map(experimentId=>({id:`journal-${e.id}-${experimentId}`,experimentId,severity:'caution' as const,status:'open' as const,title:e.title,detail:e.detail})))];
}
export function searchNotebook(data:Pick<ResearchData,'experiments'|'sources'|'figures'|'warnings'> & Partial<Pick<ResearchData,'activity'>>,query:string,runs:Run[]=[]){
 const terms=query.toLowerCase().trim().split(/\s+/).filter(Boolean);
 const matches=(values:unknown[])=>terms.length>0&&terms.every(term=>values.join(' ').toLowerCase().includes(term));
 return {
  experiments:terms.length?filterExperiments(data.experiments,{query}).slice(0,6):[],
  sources:data.sources.filter(s=>matches([s.path,s.role,...(s.experimentIds||[])])).slice(0,4),
  figures:data.figures.filter(f=>matches([f.title,f.goal,f.comparison,...f.experimentIds])).slice(0,3),
  warnings:notebookWarnings({warnings:data.warnings,activity:data.activity||[]}).filter(w=>matches([w.id,w.experimentId,w.title,w.detail])).slice(0,4),
  runs:runs.filter(r=>matches([r.id,r.jobId,r.batch,r.architecture,r.dataset,r.seed,...r.experimentIds])).slice(0,4),
 };
}
