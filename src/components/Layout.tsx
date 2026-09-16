import { useEffect, useRef, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { LayoutDashboard, FlaskConical, Images, Braces, Terminal, TriangleAlert, History, Search, ArrowUpRight, Menu, X, FileDown, ChevronRight, Command, GitBranch } from 'lucide-react';
import { useResearch, useRuns } from '../data';
import { searchNotebook } from '../lib/research';
import { sectionLabel } from '../lib/navigation';
import { SnapshotDate } from './ArtifactDates';
const navigation=[{to:'/',label:'Overview',icon:LayoutDashboard},{to:'/experiments',label:'Experiments',icon:FlaskConical},{to:'/figures',label:'Figures & tables',icon:Images},{to:'/code',label:'Source code',icon:Braces},{to:'/runs',label:'Run ledger',icon:Terminal},{to:'/warnings',label:'Warnings & limits',icon:TriangleAlert},{to:'/activity',label:'Research log',icon:History}];
function SearchDialog({open,onClose}:{open:boolean;onClose:()=>void}){const dialog=useRef<HTMLDialogElement>(null),input=useRef<HTMLInputElement>(null);const [query,setQuery]=useState('');const data=useResearch();const {runs,error:runError}=useRuns(open);useEffect(()=>{if(open){dialog.current?.showModal();setQuery('');input.current?.focus()}else dialog.current?.close()},[open]);const q=query.trim().toLowerCase();const {experiments,sources,figures,warnings,runs:runMatches}=searchNotebook(data,q,runs||[]);return <dialog ref={dialog} className="search-dialog" aria-label="Search notebook" onCancel={onClose} onClick={e=>{if(e.target===dialog.current)onClose()}}><div className="search-dialog-inner"><div className="search-dialog-input"><Search size={22}/><input ref={input} aria-label="Search all research" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search goals, warnings, code or job IDs…"/><button className="icon-button" onClick={onClose} aria-label="Close search"><X size={20}/></button></div><div className="search-results">{!q?<div className="search-hints"><span className="eyebrow">Try a starting point</span>{['CVK2','layerwise','cVK2_vggcut_score.py','pinning'].map(v=><button key={v} className="chip" onClick={()=>setQuery(v)}>{v}<ArrowUpRight size={14}/></button>)}</div>:<>{experiments.length>0&&<><div className="search-group">Experiments</div>{experiments.map(e=><Link key={e.id} to={`/experiments/${e.id}`} onClick={onClose}><span className="mono">{e.id}</span><strong>{e.goal}</strong><ChevronRight size={16}/></Link>)}</>}{sources.length>0&&<><div className="search-group">Source files</div>{sources.map(s=><Link key={s.id} to={`/code/${s.id}`} onClick={onClose}><Braces size={17}/><strong>{s.path}</strong><ChevronRight size={16}/></Link>)}</>}{figures.length>0&&<><div className="search-group">Figures</div>{figures.map(f=><Link key={f.id} to={`/figures/${f.id}`} onClick={onClose}><Images size={17}/><strong>{f.title}</strong><ChevronRight size={16}/></Link>)}</>}{warnings.length>0&&<><div className="search-group">Warnings & limits</div>{warnings.map(w=><Link key={w.id} to={`/experiments/${w.experimentId}?tab=warnings`} onClick={onClose}><TriangleAlert size={17}/><strong>{w.experimentId} · {w.title}</strong><ChevronRight size={16}/></Link>)}</>}{runMatches.length>0&&<><div className="search-group">Run records</div>{runMatches.map(r=><Link key={r.id} to={`/runs/${r.id}`} onClick={onClose}><Terminal size={17}/><strong>Job {r.jobId} · {r.batch} · seed {r.seed}</strong><ChevronRight size={16}/></Link>)}</>}{!runs&&!runError&&<p className="small muted">Loading run IDs…</p>}{runError&&<p className="small muted">Run search unavailable. Open the Run ledger to retry.</p>}{!experiments.length&&!sources.length&&!figures.length&&!warnings.length&&!runMatches.length&&(runs||runError)&&<div className="empty-state">No records match “{query}”.</div>}</>}</div><div className="search-footer">Goals, figures, code, warnings and recorded jobs.<kbd>esc</kbd> to close</div></div></dialog>}
export function Layout() {
  const data = useResearch(), location = useLocation();
  const [menu, setMenu] = useState(false), [search, setSearch] = useState(false);
  useEffect(() => {
    setMenu(false);
    window.scrollTo({top: 0});
    document.title = `${sectionLabel(location.pathname)} · MetaOptimize`;
  }, [location.pathname]);
  useEffect(() => {
    const fn = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setSearch(v => !v);
      }
    };
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, []);
  const active = sectionLabel(location.pathname);

  return <div className="app-shell">
    <a className="skip-link" href="#main-content">Skip to content</a>
    {menu && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setMenu(false)}/>}
    <aside className={`sidebar ${menu ? 'is-open' : ''}`}>
      <Link className="brand" to="/">
        <div className="brand-mark"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M10 29V11l10 12 10-12v18"/><circle cx="20" cy="23" r="2"/></svg></div>
        <div><strong>MetaOptimize</strong><span>Group workspace</span></div>
      </Link>
      <div className="nav-label">Research notebook</div>
      <nav aria-label="Main navigation">
        {navigation.map(n => <NavLink key={n.to} to={n.to} end={n.to === '/'}>
          <n.icon size={19}/><span>{n.label}</span>
          {n.to === '/experiments' && <span className="nav-count">{data.meta.stats.experiments}</span>}
        </NavLink>)}
      </nav>
      <div className="sidebar-materials">
        <a className="sidebar-report" href={data.meta.downloads.pdf} target="_blank" rel="noreferrer">
          <FileDown size={18}/><span>Report PDF</span><ArrowUpRight size={15}/>
        </a>
        <div className="sidebar-repositories" aria-label="Public GitHub repositories">
          <div className="nav-label">GitHub repositories</div>
          <a href="https://github.com/teshnizi2/metaoptimize-research-notebook" target="_blank" rel="noopener noreferrer">
            <GitBranch size={17}/><span><strong>Notebook repository</strong><small>Public · site & evidence</small></span><ArrowUpRight size={14}/>
          </a>
          <a href="https://github.com/teshnizi2/hierarchical-metaoptimize" target="_blank" rel="noopener noreferrer">
            <GitBranch size={17}/><span><strong>Research repository</strong><small>Public · experiments & code</small></span><ArrowUpRight size={14}/>
          </a>
        </div>
      </div>
      <div className="sidebar-footer">
        <SnapshotDate/>
        <Link to="/activity" className="icon-button" aria-label="View research history"><History size={18}/></Link>
      </div>
    </aside>
    <div className="workspace">
      <header className="topbar">
        <div className="breadcrumb">
          <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMenu(true)}><Menu size={22}/></button>
          <span>Research group</span><ChevronRight size={14}/><strong>{active}</strong>
        </div>
        <button className="global-search" aria-label="Search everything" onClick={() => setSearch(true)}>
          <Search size={16}/><span>Search everything</span><kbd><Command size={11}/>K</kbd>
        </button>
      </header>
      <main id="main-content" className="main-content"><Outlet/></main>
      <footer className="workspace-footer">
        <span>MetaOptimize · Research group</span>
        <span>Outcomes refer to the stated goal, not a method win rate.</span>
      </footer>
    </div>
    <SearchDialog open={search} onClose={() => setSearch(false)}/>
  </div>;
}
