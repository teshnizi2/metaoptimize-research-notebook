import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { experimentComparison, experimentQuestion, experimentReason, experimentResult, experimentWhy, hasEditorialCopy } from '../src/lib/experiment-copy.ts';
import type { ResearchData } from '../src/types.ts';

const published=JSON.parse(readFileSync(new URL('../public/data/research.json',import.meta.url),'utf8')) as ResearchData;
const byId=new Map(published.experiments.map(experiment=>[experiment.id,experiment]));
const auditedQuestionIds=['MT175','MT176','MT177','MT178','MT179','MT180','MT181','MT182','MT184','MT185','MT186','MT187','MT188','MT189','MT190','MT191','MT192','MT193','MT194','MT195','MT196','MT197','MT198','MT199','MT200','MT201','MT202','MT203','MT204','MT205','MT206','MT207','MT208','MT209','MT210','MT211','MT212','MT213','MT214','MT215','MT216','MT217','MT218','MT219','MT220','MT221','MT222','MT223','MT224','MT225','MT226','MT227'];
const recentResultIds=['MT212','MT213','MT214','MT215','MT216','MT217','MT218','MT219','MT220','MT221','MT222','MT223','MT224','MT225','MT226','MT227'];

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

test('recent scorer output is replaced by a readable result and explanation',()=>{
 for(const id of recentResultIds){
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
