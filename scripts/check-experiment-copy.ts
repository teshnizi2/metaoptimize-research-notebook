import { readFile } from 'node:fs/promises';
import { validateExperimentCopy } from '../src/lib/experiment-copy.ts';
import type { EditorialCopy } from '../src/lib/experiment-copy.ts';
import { validateLaterEvidence } from '../src/lib/later-evidence.ts';
import type { LaterEvidenceRelation } from '../src/lib/later-evidence.ts';
import type { ResearchData } from '../src/types.ts';

const root=new URL('../',import.meta.url);

async function main(){
 const research=JSON.parse(await readFile(new URL('public/data/research.json',root),'utf8')) as ResearchData;
 const copy=JSON.parse(await readFile(new URL('content/experiment-copy.json',root),'utf8')) as Record<string,Partial<EditorialCopy>>;
 const issues=validateExperimentCopy(research.experiments,copy);
 if(issues.length){
  throw new Error(`Experiment copy check failed:\n- ${issues.join('\n- ')}\nRevise content/experiment-copy.json; the exact register in public/data/research.json must remain unchanged.`);
 }
 const ids=Object.keys(copy),required=research.experiments.filter(experiment=>{
  const match=/^MT(\d+)$/.exec(experiment.id);
  return Boolean(match&&Number(match[1])>=175);
 });
 console.log(`Experiment copy check passed: ${ids.length} editorial records; all ${required.length} required MASTER-TABLE records (MT175–MT244) are covered.`);
 // The curated later-evidence cross-links (content/later-evidence.json) quote the published CORRECTIONS copy.
 const corrections=research.sources.find(source=>source.path==='docs/CORRECTIONS.md');
 if(!corrections)throw new Error('Later evidence check failed: docs/CORRECTIONS.md is not a published source.');
 const correctionsText=await readFile(new URL(`public${corrections.href}`,root),'utf8');
 const relations=JSON.parse(await readFile(new URL('content/later-evidence.json',root),'utf8')) as LaterEvidenceRelation[];
 const relationIssues=validateLaterEvidence(research.experiments,relations,correctionsText);
 if(relationIssues.length){
  throw new Error(`Later evidence check failed:\n- ${relationIssues.join('\n- ')}\nRevise content/later-evidence.json; every relation must quote its cited CORRECTIONS entry.`);
 }
 console.log(`Later evidence check passed: ${relations.length} curated relations, every quote found in its cited CORRECTIONS entry.`);
}

main().catch(error=>{
 console.error(error instanceof Error?error.message:String(error));
 process.exitCode=1;
});
