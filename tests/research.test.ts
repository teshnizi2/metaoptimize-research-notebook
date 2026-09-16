import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { correctedCount, experimentBadges, filterExperiments, isMethodCheck, methodCheckCount, outcomeCounts, outcomeLabels, outcomeOrder, relatedFigures, resolveExperiment } from '../src/lib/research.ts';
import type { Experiment, Figure, ResearchData } from '../src/types.ts';
const make=(id:string,changes:Partial<Experiment>={}):Experiment=>({id,section:1,area:'Mechanism',title:'Test a cut',goal:'Find the useful boundary',comparison:'Scalar versus layerwise',why:'Explain accuracy',result:'Mixed accuracy',kind:'research',outcome:'unresolved',corrected:false,correction:null,reason:'Scope is limited',scope:'100 epochs',batches:['cvk1'],figureIds:['page-22'],codeIds:['s1'],tableIds:[],warningIds:[],eventIds:[],runIds:[],sourceRefs:[],...changes});
const entries=[make('CVK2'),make('MT133',{area:'Hierarchy',section:2,outcome:'fail',batches:['tl1'],comparison:'Two-level shrinkage'})];
test('search finds experiment IDs, scientific terms and batch names',()=>{assert.equal(filterExperiments(entries,{query:'cvK2'}).length,1);assert.equal(filterExperiments(entries,{query:'tl1'})[0].id,'MT133');assert.equal(filterExperiments(entries,{query:'layerwise'})[0].id,'CVK2');});
test('filters combine rather than replace each other and handle empty results',()=>{assert.deepEqual(filterExperiments(entries,{query:'cut',outcome:'fail',area:'Hierarchy'}).map(x=>x.id),['MT133']);assert.deepEqual(filterExperiments(entries,{outcome:'success'}),[]);});
test('outcome counts preserve unresolved separately from failure',()=>{assert.deepEqual(outcomeCounts(entries),{success:0,fail:1,mixed:0,unresolved:1});});
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

// ---- Four-outcome model: "Corrected" is a badge, method checks are not outcomes ----
const check=make('MT113',{area:'Measurement checks',kind:'method-check',outcome:null,corrected:true,correction:{note:'Tensor averages hid coordinate binding.',source:'register'},result:'Guard occupancy hid binding',batches:['ml5']});
const corrected=make('MT014',{outcome:'success',corrected:true,correction:{note:'The 1e-3 peak was corrected.',source:'register'}});
const model=[...entries,check,corrected];

test('the research model has exactly four outcomes and no correction outcome',()=>{
 assert.deepEqual(outcomeOrder,['success','fail','mixed','unresolved']);
 assert.deepEqual(Object.values(outcomeLabels),['Goal met','Goal missed','Mixed','Open']);
 assert.ok(!('correction' in outcomeLabels));
 assert.deepEqual(Object.keys(outcomeCounts(model)),['success','fail','mixed','unresolved']);
});

test('method checks are excluded from outcome counts but stay searchable and filterable',()=>{
 assert.deepEqual(outcomeCounts(model),{success:1,fail:1,mixed:0,unresolved:1});
 assert.equal(methodCheckCount(model),1);
 assert.ok(isMethodCheck(check));
 assert.deepEqual(filterExperiments(model,{query:'guard occupancy'}).map(e=>e.id),['MT113']);
 assert.deepEqual(filterExperiments(model,{query:'method check'}).map(e=>e.id),['MT113']);
 assert.deepEqual(filterExperiments(model,{kind:'method-check'}).map(e=>e.id),['MT113']);
 for(const outcome of outcomeOrder)assert.ok(!filterExperiments(model,{outcome}).some(isMethodCheck),outcome);
});

test('the corrected badge sits beside the outcome and can be filtered, including legacy links',()=>{
 assert.deepEqual(experimentBadges(corrected),['Corrected']);
 assert.deepEqual(experimentBadges(entries[0]),[]);
 assert.equal(corrected.outcome,'success');
 assert.equal(correctedCount(model),2);
 assert.deepEqual(filterExperiments(model,{corrected:true}).map(e=>e.id),['MT113','MT014']);
 assert.deepEqual(filterExperiments(model,{outcome:'correction'}).map(e=>e.id),['MT113','MT014'],'?outcome=correction now means the badge');
 assert.deepEqual(filterExperiments(model,{outcome:'success',corrected:true}).map(e=>e.id),['MT014']);
});

const published=JSON.parse(readFileSync(new URL('../public/data/research.json',import.meta.url),'utf8')) as ResearchData;
const byId=new Map(published.experiments.map(e=>[e.id,e]));
// The operator-approved split of the 23 records that used to carry outcome "correction".
const approved:Record<string,[Experiment['kind'],Experiment['outcome']]>={MT014:['research','success'],MT060:['research','mixed'],MT051:['research','fail'],MT063:['research','fail'],MT071:['research','fail'],MT074:['research','fail'],MT086:['research','fail'],MT154:['research','fail'],
 ...Object.fromEntries(['MT108','MT109','MT110','MT111','MT112','MT113','MT115','MT116','MT117','MT118','MT119','MT122','MT123','MT124','MT158'].map(id=>[id,['method-check',null]]))};

test('published records apply the approved 23-record mapping with the corrected badge and its history',()=>{
 assert.equal(Object.keys(approved).length,23);
 for(const [id,[kind,outcome]] of Object.entries(approved)){
  const e=byId.get(id)!;
  assert.ok(e,id);
  assert.equal(e.kind,kind,id);
  assert.equal(e.outcome,outcome,id);
  assert.equal(e.corrected,true,id);
  assert.equal(e.correction?.previousLabel,'Corrected',id);
  assert.ok(e.correction?.note.length,id);
  assert.ok(published.warnings.some(w=>w.experimentId===id&&w.severity==='correction'),`${id} keeps a correction warning`);
 }
 assert.match(byId.get('MT074')!.correction!.note,/Closest call/,'MT074 records that Mixed was the closest alternative');
 assert.ok(published.experiments.every(e=>(e.outcome as string)!=='correction'));
});

test('published research outcomes and method checks are counted separately',()=>{
 const counts=outcomeCounts(published.experiments),research=Object.values(counts).reduce((a,b)=>a+b,0);
 assert.deepEqual(counts,{success:45,fail:41,mixed:26,unresolved:24});
 assert.equal(research,136);
 assert.equal(methodCheckCount(published.experiments),18);
 assert.equal(research+methodCheckCount(published.experiments),published.experiments.length);
 assert.equal(published.meta.stats.researchQuestions,136);
 assert.equal(published.meta.stats.methodChecks,18);
 for(const e of published.experiments){
  if(e.kind==='method-check')assert.equal(e.outcome,null,e.id);
  else assert.ok(outcomeOrder.includes(e.outcome!),e.id);
  assert.equal(e.corrected,Boolean(e.correction),e.id);
 }
});
