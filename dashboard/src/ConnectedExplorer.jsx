import React from 'react';

const base = `${import.meta.env.BASE_URL}data/preview/`;
async function getJson(path, signal) {
  const response = await fetch(base + path, {signal});
  if (!response.ok || !response.headers.get('content-type')?.includes('application/json')) throw new Error('Preview data could not be loaded. Regenerate the preview exports and refresh.');
  return response.json();
}

// Use the reference app's React runtime for hooks; it owns the mounted tree.
export default function ConnectedExplorer({runtime: R}) {
  const [catalog, setCatalog] = R.useState(null), [line, setLine] = R.useState('A549');
  const [dataset, setDataset] = R.useState(null), [tid, setTid] = R.useState('');
  const [query, setQuery] = R.useState(''), [error, setError] = R.useState('');
  const [minimum, setMinimum] = R.useState(0), [motif, setMotif] = R.useState('');
  const [selected, setSelected] = R.useState(null), [start, setStart] = R.useState(0), [span, setSpan] = R.useState(1000);
  const [collapsed, setCollapsed] = R.useState(false), [sort, setSort] = R.useState({key:'position', desc:false});
  const drag = R.useRef(null);
  R.useEffect(() => {
    const controller = new AbortController();
    getJson('index.json', controller.signal).then(setCatalog).catch(e => {if(e.name !== 'AbortError') setError(e.message);});
    return () => controller.abort();
  }, []);
  R.useEffect(() => {
    const controller = new AbortController();
    setDataset(null); setError(''); setSelected(null);
    getJson(`${line}.json`, controller.signal).then(d => {setDataset(d); setTid(d.transcripts[0]?.id || ''); setStart(0); setSpan(1000); setQuery(''); setMotif(''); setMinimum(0);}).catch(e => {if(e.name !== 'AbortError') setError(e.message);});
    return () => controller.abort();
  }, [line]);
  const tx = dataset?.transcripts.find(t => t.id === tid);
  const matches = dataset?.transcripts.filter(t => `${t.gene} ${t.gene_id} ${t.id}`.toLowerCase().includes(query.toLowerCase())) || [];
  const extent = Math.max(500, ...(tx?.sites.map(s => s.position + 100) || []));
  const width = Math.min(span, extent), left = Math.min(start, extent-width), end = left+width;
  const filtered = tx?.sites.filter(s => s.reads >= minimum && (!motif || s.kmer.slice(1,6)===motif)) || [];
  const sites = filtered.filter(s => s.position >= left && s.position <= end);
  const rows = [...sites].sort((a,b) => (a[sort.key]-b[sort.key])*(sort.desc?-1:1));
  const maxReads = Math.max(1,...sites.map(s=>s.reads));
  const x = position => 135 + (position-left)/width*820;
  function pan(value) {setStart(Math.max(0, Math.min(extent-width,value)));}
  function zoom(factor) {const next=Math.max(50,Math.min(extent,width*factor)); setSpan(next); setStart(Math.max(0,Math.min(extent-next,left+(width-next)/2)));}
  function choose(id) {setTid(id); setSelected(null); setStart(0); setSpan(1000); setMotif('');}
  function download() {
    const fields=['cell_line','transcript_id','position','kmer','reads','raw','rank'];
    const escape=v=>`"${String(v??'').replaceAll('"','""')}"`;
    const csv=[fields.join(','),...rows.map(s=>fields.map(k=>escape(({...s,cell_line:line,transcript_id:tid})[k])).join(','))].join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));const a=document.createElement('a');a.href=url;a.download=`${line}_${tid}_subset.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  return <div className="work-page">
    <div className="page-heading"><div><div className="kicker">EXPLORE / TRANSCRIPT</div><h1>Site Explorer <span className="source-badge real">REAL SG-NEx SUBSETS</span></h1><p>Saved predictions from seven cell lines · your model is unchanged.</p></div><button className="secondary-btn" disabled={!sites.length} onClick={download}>Export visible sites CSV</button></div>
    <div className={`explorer-layout ${collapsed?'controls-collapsed':''}`}>
      <aside className="filter-pane"><div className="filter-heading">Explore controls <button aria-expanded={!collapsed} onClick={()=>setCollapsed(!collapsed)}>{collapsed?'Expand':'Collapse'}</button></div>{!collapsed&&<>
        <label className="field-label" htmlFor="real-line">Cell line</label><select id="real-line" className="full-select" value={line} onChange={e=>setLine(e.target.value)}>{(catalog?.lines||[{name:'A549'}]).map(l=><option key={l.name}>{l.name}</option>)}</select>
        <label className="field-label" htmlFor="real-search">Gene or transcript</label><div className="sidebar-search"><input id="real-search" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search loaded gene names or ENST IDs" /></div>
        <label className="field-label" htmlFor="real-tx">Transcript ({matches.length} matches)</label><select id="real-tx" className="full-select" value={matches.some(t=>t.id===tid)?tid:''} onChange={e=>choose(e.target.value)}><option value="" disabled>Select transcript</option>{matches.map(t=><option key={t.id} value={t.id}>{t.gene} · {t.id}</option>)}</select>
        <p className="field-help">Search filters the list; select a result to change the displayed transcript.</p>
        <section className="sidebar-section"><h3 className="sidebar-section-title">Site filters</h3><label className="field-label" htmlFor="real-reads">Minimum supporting reads</label><input id="real-reads" className="full-select" type="number" min="0" step="1" value={minimum} onChange={e=>{setMinimum(Math.max(0,Math.floor(+e.target.value||0)));setSelected(null);}} /><label className="field-label" htmlFor="real-motif">DRACH motif</label><select id="real-motif" className="full-select" value={motif} onChange={e=>{setMotif(e.target.value);setSelected(null);}}><option value="">All motifs</option>{[...new Set(tx?.sites.map(s=>s.kmer.slice(1,6))||[])].sort().map(m=><option key={m}>{m}</option>)}</select><button className="secondary-btn" onClick={()=>{setMinimum(0);setMotif('');}}>Reset filters</button></section>
        <section className="sidebar-section"><h3 className="sidebar-section-title">Data source</h3><p className="field-help">First 1,000 sites per pooled cell line. These are preview subsets, not representative samples. Scores retain their original full-file ranking.</p></section>
      </>}</aside>
      <div className="browser-main">
        {error&&<p role="alert" className="error-strip">{error}</p>}
        {!dataset&&!error&&<p role="status">Loading {line} predictions…</p>}
        {tx&&<>
          <div className="stat-grid">{[['Cell line',line],['Sites in subset',dataset.transcripts.reduce((n,t)=>n+t.sites.length,0)],['Sites in view',sites.length],['Transcript sites',tx.sites.length]].map(([label,value])=><div className="metric" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
          <section className="v-card"><div className="v-card-head"><div><h3>{tx.gene} · {tid}</h3><p>Transcript positions · {Math.round(left)}–{Math.round(end)} nt</p></div><div className="range-buttons"><button onClick={()=>pan(left-width/2)} disabled={left===0}>←</button><button onClick={()=>pan(left+width/2)} disabled={end>=extent}>→</button><button onClick={()=>zoom(.5)} disabled={width<=50}>Zoom in</button><button onClick={()=>zoom(2)} disabled={width>=extent}>Zoom out</button><button onClick={()=>{setStart(0);setSpan(extent);}}>Fit sites</button></div></div>
          <div className="genome-position-control"><label htmlFor="real-position">Position along transcript<span>Drag the slider or track area to pan</span></label><input id="real-position" type="range" min="0" max={extent-width} value={left} onChange={e=>pan(+e.target.value)} /></div>
          <div className="genome-svg-wrap"><svg viewBox="0 0 1000 270" role="group" aria-label="Transcript site and coverage tracks" onPointerDown={e=>{if(e.target.closest('[data-site]'))return;drag.current={x:e.clientX,left};e.currentTarget.setPointerCapture(e.pointerId);}} onPointerMove={e=>{if(drag.current)pan(drag.current.left-(e.clientX-drag.current.x)/e.currentTarget.getBoundingClientRect().width*1000/820*width);}} onPointerUp={()=>drag.current=null} onPointerCancel={()=>drag.current=null}>
            {[0,.25,.5,.75,1].map(f=><g key={f}><line x1={135+820*f} x2={135+820*f} y1="30" y2="245" stroke="#edf0f7"/><text x={135+820*f} y="20" textAnchor="middle" fontSize="11" fill="#6d7890">{Math.round(left+width*f)}</text></g>)}
            <text x="10" y="65" fontSize="12">DRACH sites</text><text x="10" y="130" fontSize="12">Rank (0–1)</text><text x="10" y="220" fontSize="12">Reads (max {maxReads})</text>
            <line x1="135" x2="955" y1="65" y2="65" stroke="#dfe5f0"/><line x1="135" x2="955" y1="155" y2="155" stroke="#dfe5f0"/><line x1="135" x2="955" y1="245" y2="245" stroke="#dfe5f0"/>
            {sites.map(s=><g key={s.position}><circle data-site="true" cx={x(s.position)} cy="65" r={selected?.position===s.position?7:5} fill={selected?.position===s.position?'#d58a3a':'#6479d8'} stroke="#fff" tabIndex="0" role="button" aria-label={`Site ${s.position}, ${s.reads} reads`} onClick={()=>setSelected(s)} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setSelected(s);}}}><title>{s.kmer} · site {s.position}</title></circle><rect x={x(s.position)-2} y={155-s.rank*65} width="4" height={Math.max(1,s.rank*65)} fill="#8b6ed1"/><rect x={x(s.position)-2} y={245-s.reads/maxReads*65} width="4" height={s.reads/maxReads*65} fill="#2f9d88"/></g>)}
          </svg></div>
          {!sites.length&&<p className="notice">No sites in this interval pass the filters. Reset filters or fit sites.</p>}
          <div className="notice">Positions come directly from the prediction files. The range follows observed sites, not the full transcript. Genomic conversion and reference sequence are not loaded; missing sites do not mean unmodified.</div></section>
          <div className="two-col"><section className="v-card"><div className="v-card-head"><h3>Site inspector</h3><span className="source-badge real">SAVED PREDICTION</span></div>{selected?<><div className="inspector-grid">{[['Transcript position',selected.position],['Sequence',selected.kmer],['DRACH motif',selected.kmer.slice(1,6)],['Supporting reads',selected.reads],['Raw score',selected.raw.toFixed(4)],['Within-file rank',selected.rank.toFixed(4)]].map(([label,value])=><div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div><p className="notice">{selected.reads<20?'Low coverage: fewer than 20 reads. ':''}Scores are not calibrated probabilities or fractions of modified molecules.</p></>:<div className="placehold">Select a marker or table position to inspect its real prediction.</div>}</section>
          <section className="v-card"><div className="v-card-head"><h3>Reference annotation</h3></div><div className="info-list"><div><span>Gene</span><strong>{tx.gene} · {tx.gene_id||'unavailable'}</strong></div><div><span>Chromosome / strand</span><strong>{tx.chromosome||'unavailable'} / {tx.strand||'unavailable'}</strong></div><div><span>Annotated exons</span><strong>{tx.exons.length}</strong></div><div><span>Reference</span><strong>Ensembl 91 · GRCh38.p10</strong></div></div><p className="notice">Annotation identifies the transcript. Genome placement of individual sites awaits sequence-coordinate verification.</p></section></div>
          <section className="v-card"><div className="v-card-head"><h3>Visible-site table</h3><p>{rows.length} sites · click a position to inspect</p></div><div className="table-overflow"><table className="clean-table"><thead><tr>{[['position','Position'],['reads','Reads'],['raw','Raw score'],['rank','Rank score']].map(([key,label])=><th key={key} aria-sort={sort.key===key?(sort.desc?'descending':'ascending'):'none'}><button onClick={()=>setSort({key,desc:sort.key===key?!sort.desc:false})}>{label}{sort.key===key?(sort.desc?' ↓':' ↑'):''}</button></th>)}<th>Sequence</th></tr></thead><tbody>{rows.map(s=><tr key={s.position} className={selected?.position===s.position?'selected-row':''}><td><button className="secondary-btn" onClick={()=>setSelected(s)}>{s.position}</button></td><td>{s.reads}</td><td>{s.raw.toFixed(4)}</td><td>{s.rank.toFixed(4)}</td><td>{s.kmer}</td></tr>)}</tbody></table></div></section>
        </>}
      </div>
    </div>
  </div>;
}
