import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import type { ResearchData, ActivityEvent, Run } from './types';
const Context=createContext<ResearchData|null>(null);
export function ResearchProvider({children}:{children:ReactNode}){
 const [data,setData]=useState<ResearchData|null>(null),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);
 useEffect(()=>{const controller=new AbortController();setError('');Promise.all([fetch('/data/research.json',{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error('Research snapshot could not be loaded.');return r.json()}),fetch('/data/journal.json',{signal:controller.signal}).then(r=>r.ok?r.json():[])]).then(([d,journal]:[ResearchData,ActivityEvent[]])=>{if(d.experiments?.length!==d.meta?.stats?.experiments)throw new Error('The research snapshot is incomplete.');const ids=new Set(d.activity.map(e=>e.id));setData({...d,activity:[...d.activity,...journal.filter(e=>!ids.has(e.id))]});}).catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>controller.abort();},[attempt]);
 if(error)return <main className="boot-state"><div className="brand-mark">M</div><h1>We couldn’t open the notebook.</h1><p>{error}</p><button className="button primary" onClick={()=>setAttempt(x=>x+1)}>Try again</button></main>;
 if(!data)return <main className="boot-state" aria-busy="true"><div className="brand-mark pulse">M</div><h1>Opening the notebook</h1><p>Loading the verified research record…</p></main>;
 return <Context.Provider value={data}>{children}</Context.Provider>;
}
export function useResearch(){const data=useContext(Context);if(!data)throw new Error('Research provider required');return data;}
let cachedRuns:Run[]|null=null;
export function useRuns(enabled=true){const [runs,setRuns]=useState<Run[]|null>(cachedRuns),[error,setError]=useState(''),[attempt,setAttempt]=useState(0);useEffect(()=>{if(!enabled)return;if(cachedRuns){setRuns(cachedRuns);return;}const c=new AbortController();fetch('/data/runs.json',{signal:c.signal}).then(r=>{if(!r.ok)throw new Error('Run ledger could not be loaded.');return r.json()}).then(d=>{cachedRuns=d;setRuns(d);setError('');}).catch(e=>{if(e.name!=='AbortError')setError(e.message)});return()=>c.abort()},[attempt,enabled]);return {runs,error,retry:()=>setAttempt(x=>x+1)};}
