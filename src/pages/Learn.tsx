import { Link } from 'react-router-dom';
import { ArrowRight, BookOpen, ExternalLink, FileText, Play } from 'lucide-react';
import { PageHeading } from '../components/common';
import {
  LEARN_FILES, formatLength, learnVideos, mapLinks, pdfUrl, seriesTitle, studyPath, subtitleUrl, videoLabel, videoUrl, videosOf,
  type LearnSeries, type LearnVideo,
} from '../lib/learn';
import '../components/Learn.css';

const byId = new Map(learnVideos.map(v => [v.id, v]));

function VideoCard({ video }: { video: LearnVideo }) {
  return <article className="learn-card" id={video.series === 'beginner' ? video.id : `first-${video.id}`}>
    {LEARN_FILES.available
      ? <video className="learn-player" controls preload="none" playsInline>
        <source src={videoUrl(video.id)} type="video/mp4"/>
        <track kind="subtitles" srcLang="en" label="English" src={subtitleUrl(video.id)} default/>
      </video>
      : <div className="learn-player learn-soon" role="img" aria-label="Video not added yet"><Play size={28}/><span>Video coming soon</span></div>}
    <div className="learn-card-text">
      <div className="learn-meta"><span className="learn-tag">{videoLabel(video)}</span><span>{formatLength(video.seconds)}</span></div>
      <h3>{video.title}</h3>
      <p>{video.learn}</p>
    </div>
  </article>;
}

function Series({ series, intro }: { series: LearnSeries; intro: string }) {
  const videos = videosOf(series);
  return <section className="learn-section" aria-labelledby={`learn-${series}`}>
    <h2 id={`learn-${series}`}>{seriesTitle[series]}</h2>
    <p className="learn-small">{intro}</p>
    <div className="learn-grid">{videos.map(v => <VideoCard key={`${v.series}${v.id}`} video={v}/>)}</div>
  </section>;
}

export function LearnPage() {
  const total = studyPath.reduce((sum, s) => sum + s.minutes, 0);
  return <div className="learn">
    <PageHeading
      eyebrow="Learn"
      title="Learn the project"
      description="Short videos with subtitles, a guide and glossary for newcomers, and a study path that puts them in a good order."
    />
    <p className="learn-small">Want to see what the project found? <Link className="text-link" to="/achievements">What we achieved<ArrowRight size={16}/></Link></p>

    <section className="learn-section" aria-labelledby="learn-start">
      <h2 id="learn-start">Start here</h2>
      <div className="learn-docs">
        <a className="learn-doc" href={pdfUrl('guide')} target="_blank" rel="noreferrer"><BookOpen size={22}/><span><strong>Beginner guide (PDF)</strong>What the project is about, one chapter at a time, written for someone new to training networks.</span><ExternalLink size={16}/></a>
        <a className="learn-doc" href={pdfUrl('glossary')} target="_blank" rel="noreferrer"><FileText size={22}/><span><strong>Glossary (PDF)</strong>Keep it open for any word you forget.</span><ExternalLink size={16}/></a>
      </div>
    </section>

    <section className="learn-section" aria-labelledby="learn-path">
      <h2 id="learn-path">Study path</h2>
      <p className="learn-small">Go in order, one row at a time. About {Math.round(total / 30) / 2} hours in all. Each guide chapter names the minutes to watch of its outside video; our own videos play further down this page.</p>
      <ol className="learn-path">
        {studyPath.map((step, i) => {
          if (step.kind === 'guide') {
            return <li key={i}>
              <span className="learn-n">{i + 1}</span>
              <div className="learn-step">
                <a href={pdfUrl('guide', step.page)} target="_blank" rel="noreferrer"><FileText size={15}/>Guide: {step.title}<span className="learn-page"> (page {step.page})</span></a>
                {step.youtube && <a className="learn-ext" href={step.youtube.url} target="_blank" rel="noreferrer"><ExternalLink size={14}/>{step.youtube.label}</a>}
              </div>
              <span className="learn-min">{step.minutes} min</span>
            </li>;
          }
          const video = byId.get(step.videoId)!;
          return <li key={i}>
            <span className="learn-n">{i + 1}</span>
            <div className="learn-step"><a href={`#${video.id}`}><Play size={15}/>Our video {video.id}: {video.title}</a></div>
            <span className="learn-min">{step.minutes} min</span>
          </li>;
        })}
      </ol>
    </section>

    <section className="learn-section" aria-labelledby="learn-map">
      <h2 id="learn-map">See the results</h2>
      <div className="learn-links">
        {mapLinks.map(l => <Link key={l.to} to={l.to} className="learn-link"><span><strong>{l.label}</strong>{l.detail}</span><ArrowRight size={18}/></Link>)}
      </div>
    </section>

    <Series series="beginner" intro="Sixteen short videos, in the order of the study path. Each has English subtitles."/>
    <Series series="first" intro="A second, shorter set that goes over the same ground. Each has English subtitles."/>

    <p className="learn-small learn-leftout">Four videos about a separate, unpublished paper, and the status videos built on it, are not shown here.</p>
  </div>;
}
