import test from 'node:test';
import assert from 'node:assert/strict';
import { filterExperiments, outcomeCounts, relatedFigures, resolveExperiment } from '../src/lib/research.ts';
import type { Experiment, Figure } from '../src/types.ts';
const make=(id:string,changes:Partial<Experiment>={}):Experiment=>({id,section:1,area:'Mechanism',title:'Test a cut',goal:'Find the useful boundary',comparison:'Scalar versus layerwise',why:'Explain accuracy',result:'Mixed accuracy',outcome:'unresolved',reason:'Scope is limited',scope:'100 epochs',batches:['cvk1'],figureIds:['page-22'],codeIds:['s1'],tableIds:[],warningIds:[],eventIds:[],runIds:[],sourceRefs:[],...changes});
const entries=[make('CVK2'),make('MT133',{area:'Hierarchy',section:2,outcome:'fail',batches:['tl1'],comparison:'Two-level shrinkage'})];
test('search finds experiment IDs, scientific terms and batch names',()=>{assert.equal(filterExperiments(entries,{query:'cvK2'}).length,1);assert.equal(filterExperiments(entries,{query:'tl1'})[0].id,'MT133');assert.equal(filterExperiments(entries,{query:'layerwise'})[0].id,'CVK2');});
test('filters combine rather than replace each other and handle empty results',()=>{assert.deepEqual(filterExperiments(entries,{query:'cut',outcome:'fail',area:'Hierarchy'}).map(x=>x.id),['MT133']);assert.deepEqual(filterExperiments(entries,{outcome:'success'}),[]);});
test('outcome counts preserve unresolved separately from failure',()=>{assert.deepEqual(outcomeCounts(entries),{success:0,fail:1,mixed:0,unresolved:1,correction:0});});
test('stable ID lookup cannot select another experiment',()=>{assert.equal(resolveExperiment(entries,'cvk2')?.id,'CVK2');assert.equal(resolveExperiment(entries,'missing'),undefined);});
test('figure relation uses explicit IDs, with no substring collisions',()=>{const figs=[{id:'page-2',experimentIds:['MT133']},{id:'page-22',experimentIds:['CVK2']} ] as Figure[];assert.deepEqual(relatedFigures(figs,'CVK2').map(x=>x.id),['page-22']);});

test('global search connects warnings and job IDs to their records',async()=>{
 const { searchNotebook }=await import('../src/lib/research.ts');
 const data={experiments:entries,sources:[{id:'s1',path:'analysis/cVK2_vggcut_score.py'}],figures:[],warnings:[{id:'w1',experimentId:'CVK2',title:'Unresolved carrier',detail:'Magnitude is not separated'}]} as any;
 const runs=[{id:'run-5004252',jobId:'5004252',batch:'cvk1',architecture:'VGG',dataset:'CIFAR100',seed:'55',experimentIds:['CVK2']}] as any;
 assert.equal(searchNotebook(data,'magnitude',runs).warnings[0].experimentId,'CVK2');
 assert.equal(searchNotebook(data,'5004252',runs).runs[0].id,'run-5004252');
 assert.equal(searchNotebook(data,'vggcut_score',runs).sources[0].id,'s1');
 assert.equal(searchNotebook(data,'',runs).experiments.length,0);
});

test('journal warnings remain connected to experiment details and global search',async()=>{
 const { notebookWarnings,searchNotebook }=await import('../src/lib/research.ts');
 const data={experiments:entries,sources:[],figures:[],warnings:[],activity:[{id:'note-one',date:'2026-09-15',kind:'warning',title:'Check endpoint',detail:'A later audit flagged a mismatch.',experimentIds:['CVK2','MT133']}]} as any;
 assert.deepEqual(notebookWarnings(data).map(w=>w.experimentId),['CVK2','MT133']);
 assert.equal(new Set(notebookWarnings(data).map(w=>w.id)).size,2);
 assert.equal(searchNotebook(data,'mismatch').warnings.length,2);
});

test('public asset links cannot escape the site with protocol-relative or backslash URLs',async()=>{
 const { safeHref }=await import('../src/lib/research.ts');
 for(const value of ['//example.invalid/file','/\\example.invalid/file','/\n/example.invalid/file','https://example.invalid/file','javascript:alert(1)'])assert.equal(safeHref(value),'/');
 assert.equal(safeHref('/assets/logs/cvk1-job.out'),'/assets/logs/cvk1-job.out');
});
