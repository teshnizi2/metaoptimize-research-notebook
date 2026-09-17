import type { Experiment } from '../types';

import editorialCopySource from '../../content/experiment-copy.json';

export interface EditorialCopy {
  question: string;
  result: string;
  explanation?: string;
  comparison?: string;
  why?: string;
}

/**
 * Public-facing prose is maintained separately from the exact register. The
 * exact evidence fields remain unchanged and are shown under “Exact registered
 * record” on each experiment page.
 */
const editorialCopy=editorialCopySource as Record<string,EditorialCopy>;

const markerByOutcome={success:'Goal met:',fail:'Goal missed:',mixed:'Mixed:',unresolved:'Open:'} as const;

function registeredRationale(experiment:Experiment){
  const marker=experiment.kind==='method-check'?'Method check:':experiment.outcome?markerByOutcome[experiment.outcome]:'';
  const markerIndex=marker?experiment.reason.indexOf(marker):-1;
  const prose=(markerIndex>=0?experiment.reason.slice(markerIndex+marker.length):experiment.reason).trim();
  return prose.replace(/^[A-Z0-9][A-Z0-9 _+./'"=><()-]{1,100}:\s*/,'').trim()||experiment.reason;
}

export function experimentQuestion(experiment:Experiment){return editorialCopy[experiment.id]?.question||experiment.goal;}
export function experimentResult(experiment:Experiment){return editorialCopy[experiment.id]?.result||registeredRationale(experiment)||experiment.result;}
export function experimentReason(experiment:Experiment){return editorialCopy[experiment.id]?.explanation||registeredRationale(experiment)||experiment.reason;}
export function experimentComparison(experiment:Experiment){return editorialCopy[experiment.id]?.comparison||experiment.comparison;}
export function experimentWhy(experiment:Experiment){return editorialCopy[experiment.id]?.why||experiment.why;}
export function hasEditorialCopy(experiment:Experiment){return Boolean(editorialCopy[experiment.id]);}

const copyLimits={question:160,result:430,explanation:430,comparison:180,why:240} as const;
const scorerNotation=/Verdict:|CO-PRIMAR(?:Y|IES)|PRIMARY\s*\(|\b[A-Z]{2,}_[A-Z0-9_]+\b|\b[A-Z]{2,}(?:-[A-Z][A-Z0-9_]*)+\b|\bCORRECTIONS?\s+\d+|\btri:\d+|\bRULE\s+\d+|\b[A-Z][A-Z0-9_]{2,}\s*=\s*[+-]?\d|\d+\.\d+\s*-\s*\d+\.\d+\s*=/;

export function validateExperimentCopy(experiments:Experiment[],copy:Record<string,Partial<EditorialCopy>>=editorialCopy){
 const issues:string[]=[],byId=new Map(experiments.map(experiment=>[experiment.id,experiment]));
 for(const experiment of experiments){
  const match=/^MT(\d+)$/.exec(experiment.id),number=match?Number(match[1]):0;
  const required=Boolean(match&&number>=175),record=copy[experiment.id];
  if(required&&!record){
   issues.push(`${experiment.id}: public copy is missing from content/experiment-copy.json.`);
   continue;
  }
  if(!record){
   const question=experiment.goal,result=registeredRationale(experiment)||experiment.result;
   if(question.length>copyLimits.question)issues.push(`${experiment.id}: public question exceeds ${copyLimits.question} characters.`);
   if(result.length>copyLimits.result)issues.push(`${experiment.id}: public result exceeds ${copyLimits.result} characters.`);
   if(scorerNotation.test(result))issues.push(`${experiment.id}: public result contains register/scorer notation.`);
   continue;
  }
  const requiredFields:(keyof EditorialCopy)[]=['question','result','explanation','comparison','why'];
  for(const field of requiredFields){
   const value=record[field];
   if(typeof value!=='string'||!value.trim()){
    issues.push(`${experiment.id}: ${field} is required for imported records.`);
   }
  }
  for(const field of Object.keys(copyLimits) as (keyof typeof copyLimits)[]){
   const value=record[field];
   if(typeof value!=='string')continue;
   if(value.length>copyLimits[field])issues.push(`${experiment.id}: ${field} exceeds ${copyLimits[field]} characters.`);
   if(scorerNotation.test(value))issues.push(`${experiment.id}: ${field} contains register/scorer notation.`);
  }
  if(record.question===experiment.goal)issues.push(`${experiment.id}: question duplicates the exact register instead of editorial copy.`);
  if(record.result===experiment.result)issues.push(`${experiment.id}: result duplicates the exact register instead of editorial copy.`);
 }
 for(const id of Object.keys(copy).sort()){
  if(!byId.has(id))issues.push(`${id}: public copy has no matching experiment in research.json.`);
 }
 return issues;
}
