import { useEffect, useMemo, useRef, useState } from "react";

const root = `${import.meta.env.BASE_URL}data/browser/`;
export async function readHostedJson(response, missingMessage) {
  if (!response.ok || !response.headers.get("content-type")?.toLowerCase().includes("application/json")) {
    throw new Error(missingMessage);
  }
  try { return await response.json(); }
  catch { throw new Error("This prediction file is not valid JSON. Regenerate the transcript export and reload it."); }
}
const example = { cell_line: "Synthetic example", transcript_id: "DEMO_TRANSCRIPT", sites: Array.from({ length: 32 }, (_, i) => ({ transcript_position: 100 + i * 57 + (i % 3) * 9, kmer: ["AAGACCA", "GGGACTT", "TAGACTG"][i % 3], n_reads: 2 + (i * 13) % 97, score_rank: ((i * 23) % 100) / 100, score_raw: -3 + ((i * 17) % 50) / 10 })) };

export function validateTranscript(data) {
  if (!data || typeof data.transcript_id !== "string" || typeof data.cell_line !== "string" || !Array.isArray(data.sites) || !data.sites.length) throw new Error("Expected cell_line, transcript_id and a non-empty sites array.");
  const seen = new Set();
  const sites = data.sites.map((s) => {
    if (!Number.isInteger(s.transcript_position) || s.transcript_position < 0 || !Number.isInteger(s.n_reads) || s.n_reads < 1 || !/^[ACGTU]{7}$/.test(s.kmer || "")) throw new Error("Every site needs a non-negative integer position, positive read count, and seven-base kmer.");
    if (!/^[AGTU][AG]AC[ACTU]$/.test(s.kmer.slice(1, 6))) throw new Error("A site does not contain a central DRACH motif.");
    for (const key of ["score_rank", "score_raw"]) if (s[key] != null && !Number.isFinite(s[key])) throw new Error(`Invalid ${key}.`);
    if (s.score_rank != null && (s.score_rank < 0 || s.score_rank > 1)) throw new Error("Rank scores must be between 0 and 1.");
    if (seen.has(s.transcript_position)) throw new Error("Duplicate site position.");
    seen.add(s.transcript_position);
    return s;
  }).sort((a, b) => a.transcript_position - b.transcript_position);
  return { ...data, sites };
}

export default function TranscriptBrowser() {
  const [catalog, setCatalog] = useState([]), [line, setLine] = useState("");
  const [query, setQuery] = useState(""), [data, setData] = useState(null);
  const [record, setRecord] = useState(""), [status, setStatus] = useState(""), [error, setError] = useState("");
  const [isDemo, setIsDemo] = useState(false), [selected, setSelected] = useState(null);
  const [start, setStart] = useState(0), [span, setSpan] = useState(500);
  const [collapsed, setCollapsed] = useState(false), [minReads, setMinReads] = useState(0), [motif, setMotif] = useState("");
  const [sort, setSort] = useState({key: "transcript_position", descending: false});
  const drag = useRef(null), tracks = useRef(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${root}index.json`, { signal: controller.signal }).then(r => readHostedJson(r, "No SG-NEx site predictions loaded yet. Load a transcript export or try the synthetic example.")).then(d => {
      if (!Array.isArray(d.transcripts) || d.transcripts.some(t => !t || typeof t.cell_line !== "string" || typeof t.transcript_id !== "string" || typeof t.path !== "string")) throw new Error("The transcript catalogue is invalid. Regenerate the browser export.");
      setCatalog(d.transcripts); setLine(d.transcripts[0]?.cell_line || "");
    }).catch(e => { if (e.name !== "AbortError") setStatus(e.message); });
    return () => controller.abort();
  }, []);
  function show(d, demo = false) {
    const checked = validateTranscript(d);
    setData(checked); setIsDemo(demo); setSelected(null); setError("");
    setMinReads(0); setMotif("");
    setStart(Math.max(0, checked.sites[0].transcript_position - 50)); setSpan(500);
  }
  useEffect(() => {
    if (!record) return;
    const controller = new AbortController();
    setData(null); setError(""); setStatus("Loading transcript…");
    fetch(`${root}${record}`, { signal: controller.signal }).then(r => readHostedJson(r, "This transcript export is missing. Regenerate the browser export or load its JSON file directly.")).then(d => { show(d); setStatus(""); }).catch(e => { if (e.name !== "AbortError") { setError(e.message); setStatus(""); } });
    return () => controller.abort();
  }, [record]);
  const matches = useMemo(() => catalog.filter(t => t.cell_line === line && `${t.gene_name || ""} ${t.gene_id || ""} ${t.transcript_id}`.toLowerCase().includes(query.toLowerCase())), [catalog, line, query]);
  const extent = data ? Math.max(500, data.sites[data.sites.length - 1].transcript_position + 100) : 500;
  const width = Math.min(span, extent), left = Math.min(start, Math.max(0, extent - width)), end = left + width;
  const filtered = data?.sites.filter(s => s.n_reads >= minReads && (!motif || s.kmer.slice(1, 6) === motif)) || [];
  const visible = filtered.filter(s => s.transcript_position >= left && s.transcript_position <= end);
  const rows = [...visible].sort((a, b) => {
    if (a[sort.key] == null) return b[sort.key] == null ? 0 : 1;
    if (b[sort.key] == null) return -1;
    return (typeof a[sort.key] === "string" ? a[sort.key].localeCompare(b[sort.key]) : a[sort.key] - b[sort.key]) * (sort.descending ? -1 : 1);
  });
  const maxReads = Math.max(1, ...visible.map(s => s.n_reads));
  const move = n => setStart(Math.max(0, Math.min(extent - width, n)));
  function zoom(factor) { const next = Math.max(50, Math.min(extent, Math.round(width * factor))); setSpan(next); setStart(Math.max(0, Math.min(extent - next, Math.round(left + (width - next) / 2)))); }
  useEffect(() => {
    const element = tracks.current;
    if (!element) return;
    const wheel = event => { if (event.ctrlKey) { event.preventDefault(); zoom(event.deltaY > 0 ? 1.25 : .8); } };
    element.addEventListener("wheel", wheel, {passive:false});
    return () => element.removeEventListener("wheel", wheel);
  }, [data, width, left, extent]);
  function selectSite(site) { setSelected(site); move(site.transcript_position - width / 2); }
  function exportVisible() {
    const fields = ["cell_line", "transcript_id", "source", "transcript_position", "kmer", "n_reads", "score_rank", "score_raw"];
    const escape = value => `"${(typeof value === "number" ? String(value) : String(value ?? "").replace(/^[=+@-]/, "'$&")).replaceAll('"', '""')}"`;
    const csv = [fields.join(","), ...rows.map(s => fields.map(k => escape(({...s, cell_line:data.cell_line, transcript_id:data.transcript_id, source:isDemo ? "synthetic" : "loaded export"})[k])).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], {type:"text/csv;charset=utf-8"}));
    const a = document.createElement("a"); a.href = url; a.download = `${isDemo ? "synthetic-" : ""}visible-sites.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className="transcript-browser">
    <h2>Explore a transcript</h2>
    <p className="note">Browse candidate DRACH sites in transcript coordinates. Click a site to inspect its saved model prediction.</p>
    <button className="tab" aria-expanded={!collapsed} aria-controls="row-filters" onClick={() => setCollapsed(!collapsed)}>{collapsed ? "Show filters" : "Hide filters"}</button>
    <div className={`row-workspace ${collapsed ? "filters-collapsed" : ""}`}>
    <aside id="row-filters" className="row-filters" hidden={collapsed}>
    <h3>Find your sites</h3>
    <div className="browser-controls">
      <label>Cell line<select value={line} disabled={!catalog.length} onChange={e => { setLine(e.target.value); setRecord(""); setData(null); }}><option value="">Select cell line</option>{[...new Set(catalog.map(t => t.cell_line))].map(c => <option key={c}>{c}</option>)}</select></label>
      <label>Gene or transcript<input type="search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search loaded transcripts" /></label>
      <label>Transcript<select value={record} onChange={e => { setRecord(e.target.value); if (!e.target.value) setData(null); }}><option value="">Choose a transcript ({matches.length})</option>{matches.slice(0, 200).map(t => <option key={t.path} value={t.path}>{t.gene_name ? `${t.gene_name} · ` : ""}{t.transcript_id}</option>)}</select></label>
    </div>
    {matches.length > 200 && <p className="note">Showing the first 200 matches. Refine your search.</p>}
    <div className="browser-controls"><label>Load a transcript export (.json)<input type="file" accept=".json,application/json" onChange={async e => { const f = e.target.files?.[0]; if (!f) return; setRecord(""); setStatus(""); try { if (f.size > 10_000_000) throw new Error("Choose a transcript export smaller than 10 MB."); show(JSON.parse(await f.text())); } catch (err) { setError(err.message); } e.target.value = ""; }} /></label><button className="tab" onClick={() => { setRecord(""); setStatus(""); show(example, true); }}>Try synthetic example</button></div>
    <p className="note">Files stay in your browser. This viewer retrieves existing predictions; it does not run the model.</p>
    {data && <div className="browser-controls"><label>Minimum supporting reads<input type="number" min="0" step="1" value={minReads} onChange={e => { setMinReads(Math.max(0, Math.floor(Number(e.target.value) || 0))); setSelected(null); }} /></label><label>DRACH motif<select value={motif} onChange={e => { setMotif(e.target.value); setSelected(null); }}><option value="">All motifs</option>{[...new Set(data.sites.map(s => s.kmer.slice(1, 6)))].sort().map(m => <option key={m}>{m}</option>)}</select></label><button className="tab" onClick={() => { setMinReads(0); setMotif(""); }}>Reset site filters</button></div>}
    </aside><div className="row-main">
    {status && <p role="status">{status}</p>}{error && <p role="alert">{error}</p>}
    {!data && <div className="browser-empty">Choose a hosted transcript, load a transcript export, or try the synthetic example. Full SG-NEx per-site results have not been bundled with this branch.</div>}
    {data && <>
      <div className="browser-heading"><h3>{data.transcript_id}</h3><span>{data.cell_line} · {data.sites.length.toLocaleString()} candidate sites</span></div>
      <span className={`source-tag ${isDemo ? "synthetic" : ""}`}>{isDemo ? "SYNTHETIC" : "LOADED PREDICTIONS"}</span>
      <div className="row-metrics"><div><span>Sites after filters</span><strong>{filtered.length}</strong></div><div><span>Sites in view</span><strong>{visible.length}</strong></div><div><span>Coordinate system</span><strong>Transcript · nt</strong></div></div>
      {isDemo && <p className="demo-notice">SYNTHETIC EXAMPLE — invented positions and scores for testing the interface, not SG-NEx results.</p>}
      <div className="browser-controls"><button className="tab" onClick={() => move(left - width / 2)} disabled={left === 0}>← Earlier</button><button className="tab" onClick={() => move(left + width / 2)} disabled={end >= extent}>Later →</button><button className="tab" onClick={() => zoom(0.5)} disabled={width <= 50}>Zoom in</button><button className="tab" onClick={() => zoom(2)} disabled={width >= extent}>Zoom out</button><button className="tab" onClick={() => { setStart(0); setSpan(extent); }}>Fit sites</button><span aria-live="polite">{Math.round(left)}–{Math.round(end)} nt · {visible.length} sites</span></div>
      <label className="pan-control">Scroll along transcript<input aria-label="Transcript position" type="range" min="0" max={Math.max(0, extent - width)} value={left} onChange={e => move(Number(e.target.value))} /></label>
      <div className="row-detail-layout"><div className="row-track-panel">
      <p className="note">Drag empty track space to pan. Hold Ctrl and scroll to zoom. Arrow keys pan when tracks are focused.</p>
      <div className="browser-tracks" ref={tracks} tabIndex="0" aria-label="Transcript tracks" onKeyDown={e => { if (["ArrowLeft", "ArrowRight"].includes(e.key)) { e.preventDefault(); move(left + (e.key === "ArrowLeft" ? -1 : 1) * width / 10); } }} onPointerDown={e => { if (e.target.closest("button")) return; drag.current = {x:e.clientX, start:left, pixels:e.currentTarget.clientWidth, width}; e.currentTarget.setPointerCapture(e.pointerId); }} onPointerMove={e => { if (drag.current) move(drag.current.start - (e.clientX - drag.current.x) / drag.current.pixels * drag.current.width); }} onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
        <div className="position-axis">{[0, 0.25, 0.5, 0.75, 1].map(f => <span key={f}>{Math.round(left + f * width)}</span>)}</div>
        <p className="track-label">DRACH candidates · lighter → darker = lower → higher within-file rank; hollow = unavailable</p>
        <div className="site-track">{visible.map(s => <button key={s.transcript_position} className={`site-marker ${selected?.transcript_position === s.transcript_position ? "chosen" : ""}`} style={{ left: `${(s.transcript_position - left) / width * 100}%`, background: s.score_rank == null ? "var(--card)" : `color-mix(in srgb, var(--accent) ${25 + s.score_rank * 75}%, var(--card))` }} aria-label={`Site ${s.transcript_position}, ${s.n_reads} reads, rank ${s.score_rank ?? "unavailable"}`} aria-pressed={selected?.transcript_position === s.transcript_position} title={`Position ${s.transcript_position} · ${s.kmer} · ${s.n_reads} reads`} onClick={() => setSelected(s)} />)}</div>
        <p className="track-label">Read coverage · visible maximum {maxReads}</p>
        <div className="coverage-track">{visible.map(s => <span key={s.transcript_position} style={{ left: `${(s.transcript_position - left) / width * 100}%`, height: `${s.n_reads / maxReads * 100}%` }} />)}</div>
        <p className="track-label">Within-file rank · fixed scale 0–1 · missing scores omitted</p>
        <div className="coverage-track rank-track">{visible.filter(s => s.score_rank != null).map(s => <span key={s.transcript_position} style={{left:`${(s.transcript_position-left)/width*100}%`,height:`${Math.max(1,s.score_rank*100)}%`}} title={`Rank ${s.score_rank}`} />)}</div>
      </div>
      {!visible.length && <p>No candidate sites in this interval. Pan or zoom out.</p>}
      <p className="note">The range follows observed sites, not annotated transcript length. Missing markers do not imply absence of a motif or modification. Gene structure annotations are not loaded.</p>
      </div>
      <div className="site-inspector" aria-live="polite">{selected ? <><h3>Site {selected.transcript_position}</h3><p className="sequence">{selected.kmer.slice(0, 3)}<mark>{selected.kmer[3]}</mark>{selected.kmer.slice(4)}</p><dl><dt>Central DRACH motif</dt><dd>{selected.kmer.slice(1, 6)}</dd><dt>Supporting reads</dt><dd>{selected.n_reads}{selected.n_reads < 20 ? " · low coverage (below 20 reads)" : ""}</dd><dt>Within-file rank score</dt><dd>{selected.score_rank?.toFixed(4) ?? "Unavailable"}</dd><dt>Raw model score</dt><dd>{selected.score_raw?.toFixed(4) ?? "Unavailable"}</dd></dl><p className="note">Higher scores indicate stronger model evidence. These are not calibrated probabilities or a confirmed modified/unmodified call.</p></> : <p>Select a marker or a row below to inspect a prediction.</p>}</div>
      </div>
      <div className="browser-heading"><h3>Sites in this interval ({visible.length})</h3><button className="tab" disabled={!visible.length} onClick={exportVisible}>Export visible sites CSV</button></div>
      <p className="note">Sort by a column. Select a position to centre the browser on that site.</p>
      <div className="site-table"><table><thead><tr>{[["transcript_position","Position"],["kmer","Sequence"],["n_reads","Reads"],["score_rank","Rank score"],["score_raw","Raw score"]].map(([key,label]) => <th key={key} aria-sort={sort.key === key ? sort.descending ? "descending" : "ascending" : "none"}><button className="tab" onClick={() => setSort({key,descending:sort.key===key ? !sort.descending : false})}>{label}{sort.key===key ? sort.descending ? " ↓" : " ↑" : ""}</button></th>)}</tr></thead><tbody>{rows.map(s => <tr key={s.transcript_position} className={selected?.transcript_position===s.transcript_position ? "selected-site-row" : ""}><td><button className="tab" aria-pressed={selected?.transcript_position===s.transcript_position} onClick={() => selectSite(s)}>{s.transcript_position}</button></td><td>{s.kmer}</td><td>{s.n_reads}</td><td>{s.score_rank?.toFixed(4) ?? "—"}</td><td>{s.score_raw?.toFixed(4) ?? "—"}</td></tr>)}</tbody></table></div>
    </>}
    </div></div>
  </section>;
}
