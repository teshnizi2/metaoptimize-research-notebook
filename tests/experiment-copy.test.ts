import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { experimentComparison, experimentQuestion, experimentReason, experimentResult, experimentWhy, hasEditorialCopy } from '../src/lib/experiment-copy.ts';
import type { ResearchData } from '../src/types.ts';

const published=JSON.parse(readFileSync(new URL('../public/data/research.json',import.meta.url),'utf8')) as ResearchData;
const byId=new Map(published.experiments.map(experiment=>[experiment.id,experiment]));
const auditedQuestionIds=['MT175','MT176','MT177','MT178','MT179','MT180','MT181','MT182','MT183','MT184','MT185','MT186','MT187','MT188','MT189','MT190','MT191','MT192','MT193','MT194','MT195','MT196','MT197','MT198','MT199','MT200','MT201','MT202','MT203','MT204','MT205','MT206','MT207','MT208','MT209','MT210','MT211','MT212','MT213','MT214','MT215','MT216','MT217','MT218','MT219','MT220','MT221','MT222','MT223','MT224','MT225','MT226','MT227','MT228','MT229','MT230','MT231','MT232','MT233','MT234','MT235','MT236'];
const editorialIds=['MT019',...auditedQuestionIds];
const importedRecordIds=auditedQuestionIds;

test('long internal questions have concise public wording while the source data stays intact',()=>{
 for(const id of auditedQuestionIds){
  const experiment=byId.get(id)!;
  const sourceQuestion=experiment.goal;
  const displayQuestion=experimentQuestion(experiment);
  assert.ok(hasEditorialCopy(experiment),id);
  assert.notEqual(displayQuestion,sourceQuestion,id);
  assert.ok(displayQuestion.length<=160,`${id}: ${displayQuestion.length} characters`);
  assert.doesNotMatch(displayQuestion,/\u2014/,id);
  assert.equal(experiment.goal,sourceQuestion,`${id} source question was mutated`);
 }
});

test('every imported record has readable copy for each reader-facing field',()=>{
 for(const id of importedRecordIds){
  const experiment=byId.get(id)!;
  const displayResult=experimentResult(experiment);
  const explanation=experimentReason(experiment);
  const comparison=experimentComparison(experiment);
  const why=experimentWhy(experiment);
  assert.notEqual(displayResult,experiment.result,id);
  assert.ok(displayResult.length<=430,`${id} result: ${displayResult.length} characters`);
  assert.ok(explanation.length>0,id);
  assert.ok(comparison.length<=180,`${id} comparison: ${comparison.length} characters`);
  assert.ok(why.length<=240,`${id} rationale: ${why.length} characters`);
  assert.notEqual(comparison,experiment.comparison,id);
  assert.notEqual(why,experiment.why,id);
  assert.doesNotMatch(displayResult,/CO-PRIMARY|PRIMARY \(|Verdict:|[A-Z_]{4,}\s*=|\u2014/,id);
  assert.doesNotMatch(explanation,/Verdict:|\u2014/,id);
  assert.doesNotMatch(comparison+why,/PATCH_|CORRECTIONS|tri:\d|\u2014/,id);
 }
});

test('the four reported rows preserve their decisive measurements in readable prose',()=>{
 const expected:Record<string,string[]>={
  MT224:['11.49%','41.56%'],
  MT225:['50.23%','22.95%','70.04%'],
  MT226:['19.38%','23.06%','49.63%','8.56'],
  MT227:['55.86%','21.84%','11.21%','18.37%','27.71%'],
  MT228:['48.77'],
  MT230:['81 percent'],
  MT231:['4.12','1.09','3.02'],
  MT232:['65 percent','21'],
  MT233:['70.8','4,800'],
  MT234:['49.9','65.2'],
  MT235:['10.7','65.6','65.8'],
 };
 for(const [id,values] of Object.entries(expected)){
  const result=experimentResult(byId.get(id)!);
  for(const value of values)assert.match(result,new RegExp(value.replace('.','\\.')),`${id} keeps ${value}`);
 }
});

test('records without a manual result use prose from the registered rationale instead of scorer output',()=>{
 const experiment=byId.get('MT164')!;
 const result=experimentResult(experiment);
 assert.notEqual(result,experiment.result);
 assert.doesNotMatch(result,/^Verdict:|[A-Z_]{4,}\s*=/);
 assert.equal(experimentReason(experiment),result);
});

test('the complete register avoids formula dumps and oversized public summaries',()=>{
 for(const experiment of published.experiments){
  const question=experimentQuestion(experiment);
  const result=experimentResult(experiment);
  assert.ok(question.length<=160,`${experiment.id} question is too long`);
  assert.ok(result.length<=430,`${experiment.id} result is too long`);
  assert.doesNotMatch(result,/Verdict:|CO-PRIMAR|PRIMARY \(|\b[A-Z][A-Z0-9_]{2,}\s*=\s*[+-]?\d|\d+\.\d+\s*-\s*\d+\.\d+\s*=/,experiment.id);
 }
});

test('a newly imported MASTER-TABLE row cannot be published without editorial copy',async()=>{
 const module=await import('../src/lib/experiment-copy.ts');
 assert.equal(typeof module.validateExperimentCopy,'function');
 const future:typeof published.experiments[number]={
  ...byId.get('MT227')!,
  id:'MT237',
  goal:'On PlainNet, does the registered carrier schedule produce the DOSE-GRADED branch under the held complementary path?',
  result:'CO-PRIMARIES P_DOSE = HOLDHIGH - HOLDBIG = 49.6327 - 19.3780 = +30.2547 pp',
 };
 const issues=module.validateExperimentCopy([...published.experiments,future]);
 assert.deepEqual(issues.filter(issue=>issue.startsWith('MT237:')),['MT237: public copy is missing from content/experiment-copy.json.']);
});

test('editorial wording is maintained as content and covers the current imported register',async()=>{
 const path=new URL('../content/experiment-copy.json',import.meta.url);
 assert.ok(existsSync(path),'content/experiment-copy.json is missing');
 const copy=JSON.parse(readFileSync(path,'utf8')) as Record<string,unknown>;
 assert.deepEqual((await import('../src/lib/experiment-copy.ts')).validateExperimentCopy(published.experiments),[]);
 assert.equal(Object.keys(copy).length,editorialIds.length);
 assert.deepEqual(Object.keys(copy).sort(),[...editorialIds].sort());
});

test('the publication gate rejects stale, incomplete, and scorer-style editorial content',async()=>{
 const module=await import('../src/lib/experiment-copy.ts');
 const copy=JSON.parse(readFileSync(new URL('../content/experiment-copy.json',import.meta.url),'utf8')) as Record<string,{question:string;result:string}>;
 const future:typeof published.experiments[number]={
  ...byId.get('MT227')!,
  id:'MT237',
  goal:'Does the next registered intervention separate the two causal routes?',
  result:'PRIMARY P_ROUTE = HELD - FREE = 49.6327 - 19.3780 = +30.2547 pp',
 };
 const issues=module.validateExperimentCopy([...published.experiments,future],{
  ...copy,
  MT237:{question:'Does the next intervention separate the two causal routes?',result:'CO-PRIMARIES P_ROUTE = 49.6327 - 19.3780 = +30.2547 pp'},
  MT999:{question:'Stale copy',result:'Stale result'},
 });
 assert.ok(issues.includes('MT237: result contains register/scorer notation.'));
 assert.ok(issues.includes('MT237: explanation is required for imported records.'));
 assert.ok(issues.includes('MT237: comparison is required for imported records.'));
 assert.ok(issues.includes('MT237: why is required for imported records.'));
 assert.ok(issues.includes('MT999: public copy has no matching experiment in research.json.'));
});

test('the publication gate scans every reader-facing field for internal notation',async()=>{
 const module=await import('../src/lib/experiment-copy.ts');
 const copy=JSON.parse(readFileSync(new URL('../content/experiment-copy.json',import.meta.url),'utf8')) as Record<string,{question:string;result:string;explanation?:string;comparison?:string;why?:string}>;
 const future:typeof published.experiments[number]={...byId.get('MT227')!,id:'MT237'};
 const issues=module.validateExperimentCopy([...published.experiments,future],{
  ...copy,
  MT237:{
   question:'Does PATCH_FALSIFIED identify the failure?',
   result:'PATCH_FALSIFIED under the registered primary rule.',
   explanation:'Verdict: the scorer account did not pass.',
   comparison:'PRIMARY (held versus free).',
   why:'To test the tri:7 scorer account.',
  },
 });
 for(const field of ['question','result','explanation','comparison','why']){
  assert.ok(issues.includes(`MT237: ${field} contains register/scorer notation.`),`${field}: ${issues.join(' | ')}`);
 }
});

test('the editorial-copy check is a required build step',()=>{
 const packageJson=JSON.parse(readFileSync(new URL('../package.json',import.meta.url),'utf8')) as {scripts:Record<string,string>};
 assert.equal(packageJson.scripts['copy:check'],'tsx scripts/check-experiment-copy.ts');
 assert.match(packageJson.scripts.build,/^npm run copy:check && /);
 const result=spawnSync(process.execPath,['--import','tsx','scripts/check-experiment-copy.ts'],{cwd:new URL('..',import.meta.url),encoding:'utf8'});
 assert.equal(result.status,0,result.stderr);
 assert.match(result.stdout,/Experiment copy check passed: 63 editorial records; all 62 required MASTER-TABLE records \(MT175–MT236\) are covered\./);
});
