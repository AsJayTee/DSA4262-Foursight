import { useEffect, useMemo, useRef, useState } from 'react';
import { GENES, LINES, MOTIFS, getDemoSites, fmt } from '../data/mockSites';
import { loadLocalGene, loadGeneIndex, validateRecord } from '../data/realData';
const MOTIF_GROUP_COLORS = {
    A: '#D58A3A',
    G: '#2F9D88',
    T: '#8B6ED1'
};
const motifColor = motif => MOTIF_GROUP_COLORS[motif?.[0]] || '#94a3b8';
const statusLabel = status => status === 'evaluated' ? 'Scored by model' : 'Not scored by model';
const COLORS = ['#6479d8', '#09a5a0', '#d18d48', '#bd6297', '#609c70', '#a076d3', '#64879e'];
const rng = n => {
    const v = Math.sin(n * 94.173 + 10.31) * 14971.13;
    return v - Math.floor(v);
};
export const geneDetails = Object.fromEntries(GENES.map((g, i) => [g.name, {
        ...g,
        isoforms: [{
                id: g.tx,
                exons: g.exons,
                utr: [g.exons[0], g.exons.at(-1)]
            }, {
                id: `ENST_DEMO_${g.name}_02`,
                exons: g.exons.filter((_, j) => j !== 2),
                utr: [g.exons[0]]
            }, {
                id: `ENST_DEMO_${g.name}_03`,
                exons: g.exons.filter((_, j) => j !== 1),
                utr: [g.exons.at(-1)]
            }],
        note: 'Synthetic transcript structure; not assembly-verified',
        assembly: 'DEMO (no reference assembly)',
        i
    }]));
export function Badge({ kind = 'demo', children }) {
    return <span className={'source-badge ' + kind}>{children || {
            demo: 'SYNTHETIC',
            real: 'REPOSITORY DATA',
            placeholder: 'PLACEHOLDER',
            mixed: 'MIXED SOURCES'
        }[kind]}</span>;
}
export function Card({ title, sub, kind, children, extra }) {
    return <section className="v-card">
    <div className="v-card-head">
    <div>
    <h3>{title}</h3>{sub && <p>{sub}</p>}</div>
    <div className="head-actions">{kind && <Badge kind={kind}/>} {extra}</div>
    </div>{children}</section>;
}
export function PlotBars({ labels, values, color = '#6577d7', onPick }) {
    const max = Math.max(1, ...values);
    return <div className="plot-bars">{labels.map((label, i) => <button type="button" className="plot-row" key={label} onClick={() => onPick?.(label)} disabled={!onPick}>
        <span className="plot-label">{label}</span>
        <div className="plot-track">
        <div style={{
                width: `${values[i] / max * 100}%`,
                background: color
            }}/>
        </div>
        <strong>{typeof values[i] === 'number' ? values[i].toFixed(values[i] > 100 ? 0 : 2) : values[i]}</strong>
        </button>)}</div>;
}
export function Heatmap({ rows, cols, value, onCell, format }) {
    return <div className="heat-scroll">
    <div className="heatmap" style={{
            gridTemplateColumns: `minmax(126px,1.3fr) repeat(${cols.length}, minmax(68px,1fr))`
        }}>
    <div className="heat-label"/>{cols.map(c => <div className="heat-col" key={c}>{c}</div>)}{rows.map((r, i) => <div className="heat-frag" key={r}>
        <div className="heat-label">{r}</div>{cols.map((c, j) => {
                const v = value(i, j);
                return <button key={c} type="button" className="heat-cell" title={`${r} × ${c}: ${format?.(v) ?? v.toFixed(2)}`} style={{
                        background: `rgba(85,102,206,${Math.max(.09, Math.min(.88, (v + 2) / 5))})`
                    }} onClick={() => onCell?.(r, c)}>{format?.(v) ?? v.toFixed(2)}</button>;
            })}</div>)}</div>
    </div>;
}
function siteFor(g, line) {
    return getDemoSites(g, line).map((s, i) => ({
        ...s,
        status: i % 9 === 0 ? 'unevaluated' : 'evaluated',
        raw: i % 9 === 0 ? null : s.raw,
        rank: i % 9 === 0 ? null : s.rank,
        transcriptPosition: Math.round((s.position - geneDetails[g].start) * .7),
        ownScore: i % 9 === 0 ? null : s.raw - .1 - .5 * rng(i + 4),
        neighborCorrection: .1 + .5 * rng(i + 4),
        sample: `${line}_DEMO_${i % 3 + 1}`
    }));
}
export function CompleteExplorer({ jump }) {
    const [gene, setGene] = useState(jump?.gene || 'PTBP1'), [query, setQuery] = useState(''), [lines, setLines] = useState(['Hct116', 'A549']), [motifs, setMotifs] = useState(MOTIFS), [motifSearch, setMotifSearch] = useState(''), [minReads, setMinReads] = useState(10), [cutoff, setCutoff] = useState(-10), [showUneval, setShowUneval] = useState(true), [selected, setSelected] = useState(null), [window, setWindow] = useState(null), [mode, setMode] = useState('demo'), [localDoc, setLocalDoc] = useState(null), [uploadedDoc, setUploadedDoc] = useState(null), [loadError, setLoadError] = useState(''), [sort, setSort] = useState('position'), [ascending, setAscending] = useState(true), [expanded, setExpanded] = useState(true), [motifColorMode, setMotifColorMode] = useState('group'), [highlightMotif, setHighlightMotif] = useState('GGACT'), [transcriptView, setTranscriptView] = useState('top'), [selectedTranscript, setSelectedTranscript] = useState('all');
    const dragRef = useRef(null);
    const navigatorDragRef = useRef(null);
    const suppressClickRef = useRef(false);
    const [localSource, setLocalSource] = useState('file');
    const [geneIndex, setGeneIndex] = useState(null);
    const [indexError, setIndexError] = useState('');
    useEffect(() => {
        if (mode !== 'local') return;
        let alive = true;
        loadGeneIndex().then(d => { if (alive) { setGeneIndex(d); setIndexError(''); } }).catch(e => { if (alive) setIndexError(e.message); });
        return () => { alive = false; };
    }, [mode]);
    const activeDoc = mode === 'local' ? (localSource === 'upload' ? uploadedDoc : localDoc) : null;
    const realTx = activeDoc?.gene === gene && Array.isArray(activeDoc.transcripts) ? activeDoc.transcripts : [];
    const realGene = realTx.length ? {
        name: gene,
        chr: realTx[0].chr,
        strand: realTx[0].strand,
        start: Math.min(...realTx.flatMap(t => t.exons.map(e => e[0]))),
        end: Math.max(...realTx.flatMap(t => t.exons.map(e => e[1]))),
        isoforms: realTx.map(t => ({ id: t.transcript_id, name: t.transcript_name || '', exons: t.exons, utr: [] })),
        assembly: 'GRCh38 / Ensembl 91',
        note: 'GTF-derived transcript structures; CDS/UTR not classified'
    } : null;
    const g = mode === 'local' && realGene ? realGene : (geneDetails[gene] || {name:gene, chr:'—',strand:'?',start:0,end:1,isoforms:[]});
    const realReady = mode === 'local' && !!realGene;
    const activeTranscript = selectedTranscript === 'all' || g?.isoforms?.some(t => t.id === selectedTranscript) ? selectedTranscript : 'all';
    const [a, b] = window || [g.start, g.end];
    const span = b - a;
    useEffect(() => {
        if (jump?.gene && geneDetails[jump.gene]) {
            setGene(jump.gene);
            setWindow(null);
            setSelected(null);
        }
    }, [jump]);
    useEffect(() => {
        if (mode !== 'local') {
            setLocalDoc(null);
            setLoadError('');
            return;
        }
        if (localSource === 'upload') {
            setLoadError('');
            return;
        }
        let active = true;
        setLocalDoc(null);
        loadLocalGene(gene).then(d => {
            if (active) {
                setLocalDoc(d);
                setLoadError('');
            }
        }).catch(e => {
            if (active) {
                setLocalDoc(null);
                setLoadError(e.message);
            }
        });
        return () => {
            active = false;
        };
    }, [mode, gene, localSource]);
    const rawSites = useMemo(() => mode === 'demo' ? lines.flatMap(l => siteFor(gene, l)) : (activeDoc?.gene === gene ? activeDoc.sites : []).filter(s => s.gene === gene && lines.includes(s.line)), [mode, lines, gene, activeDoc]);
    const visible = useMemo(() => rawSites.filter(s => (activeTranscript === 'all' || s.transcript === activeTranscript) && s.position >= a && s.position <= b && motifs.includes(s.motif) && s.reads >= minReads && (s.status === 'unevaluated' ? showUneval && cutoff <= -10 : s.raw >= cutoff)).sort((x, y) => x.position - y.position), [rawSites, a, b, motifs, minReads, cutoff, showUneval, activeTranscript]);
    const candidates = visible;
    const ordered = useMemo(() => [...visible].sort((x, y) => {
        const xv = x[sort] ?? -Infinity, yv = y[sort] ?? -Infinity;
        return (typeof xv === 'number' ? xv - yv : String(xv).localeCompare(String(yv))) * (ascending ? 1 : -1);
    }), [visible, sort, ascending]);
    const selectedSite = selected && rawSites.find(s => s.id === selected.id);
    async function importFile(e) {
        const file = e.target.files?.[0];
        if (!file)
            return;
        try {
            const d = JSON.parse(await file.text());
            if (d.schema_version !== 1 || !Array.isArray(d.sites) || !d.sites.every(validateRecord) || d.provenance?.kind !== 'processed-predictions')
                throw new Error('Invalid data schema or provenance; see README.md (Local prediction JSON schema)');
            if (!Array.isArray(d.transcripts) || !d.transcripts.length) throw new Error('Imported JSON lacks real GTF transcript annotations.');
            setUploadedDoc(d);
            setGene(d.gene);
            setWindow(null);
            setLocalSource('upload');
            setMode('local');
            setLoadError('');
        }
        catch (err) {
            setLoadError(err.message);
        }
        e.target.value = '';
    }
    function navigate(q) {
        const name = q.trim().toUpperCase();
        if (mode === 'local' && geneIndex) {
            const matches = geneIndex.genes.filter(x => x.symbol.toUpperCase() === name || x.gene_id.toUpperCase() === name || x.transcripts.some(t => t.toUpperCase() === name));
            if (matches.length === 1) {
                setGene(matches[0].symbol); setWindow(null); setSelected(null); setSelectedTranscript('all'); setQuery(''); setLoadError(''); return;
            }
            if (matches.length > 1) { setLoadError(`Ambiguous gene name: ${name}. Search using an ENSG gene ID.`); return; }
            setLoadError(`No indexed gene or transcript found for ${name}.`); return;
        }
        if (geneDetails[name]) {
            setGene(name);
            setWindow(null);
            setSelected(null);
            setQuery('');
            return;
        }
        const t = GENES.find(x => x.tx.toUpperCase() === name || `ENST_DEMO_${x.name}_02` === name || `ENST_DEMO_${x.name}_03` === name);
        if (t) {
            setGene(t.name);
            setWindow(null);
            setQuery('');
            return;
        }
        const m = q.match(/^(chr\w+):([\d,]+)-([\d,]+)$/i);
        if (m) {
            const s = Number(m[2].replaceAll(',', '')), e = Number(m[3].replaceAll(',', ''));
            const x = GENES.find(x => x.chr.toLowerCase() === m[1].toLowerCase() && s >= x.start && e <= x.end && e > s);
            if (x) {
                setGene(x.name);
                setWindow([s, e]);
                setQuery('');
                return;
            }
        }
        setLoadError('Search available only for four synthetic demo genes and their displayed coordinates. Real reference search requires validated annotation.');
    }
    function changeRange(s, e) {
        const width = Math.max(150, Math.min(g.end - g.start, e - s));
        const start = Math.min(g.end - width, Math.max(g.start, s));
        setWindow([Math.round(start), Math.round(start + width)]);
    }
    function zoom(f) {
        const mid = (a + b) / 2;
        changeRange(mid - span * f / 2, mid + span * f / 2);
    }
    function pan(d) {
        changeRange(a + span * d, b + span * d);
    }
    function onTrackPointerDown(e) {
        if (e.button !== 0)
            return;
        suppressClickRef.current = false;
        if (e.target.closest('[data-site-marker]'))
            return;
        dragRef.current = {
            x: e.clientX,
            start: a,
            width: span,
            moved: false
        };
        e.currentTarget.setPointerCapture(e.pointerId);
    }
    function onTrackPointerMove(e) {
        const drag = dragRef.current;
        if (!drag)
            return;
        const width = e.currentTarget.getBoundingClientRect().width;
        const dx = e.clientX - drag.x;
        if (Math.abs(dx) > 4)
            drag.moved = true;
        if (!drag.moved)
            return;
        changeRange(drag.start - dx / Math.max(width, 1) * 1190 / 1040 * drag.width, drag.start - dx / Math.max(width, 1) * 1190 / 1040 * drag.width + drag.width);
    }
    function onTrackPointerUp(e) {
        if (dragRef.current?.moved) {
            suppressClickRef.current = true;
        }
        dragRef.current = null;
        if (e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.releasePointerCapture(e.pointerId);
    }
    function onNavigatorPointerDown(e) {
        if (e.button !== 0)
            return;
        const rect = e.currentTarget.getBoundingClientRect();
        const total = g.end - g.start;
        const fraction = span / total;
        const startFraction = (a - g.start) / total;
        const pointerFraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
        const withinThumb = pointerFraction >= startFraction && pointerFraction <= startFraction + fraction;
        // Clicking the grey rail centers the viewport; grabbing the colored window preserves the grab offset.
        const grabOffset = withinThumb ? pointerFraction - startFraction : fraction / 2;
        const nextStart = g.start + Math.max(0, Math.min(1 - fraction, pointerFraction - grabOffset)) * total;
        if (!withinThumb)
            changeRange(nextStart, nextStart + span);
        navigatorDragRef.current = {
            grabOffset
        };
        e.currentTarget.setPointerCapture(e.pointerId);
        e.preventDefault();
    }
    function onNavigatorPointerMove(e) {
        const drag = navigatorDragRef.current;
        if (!drag)
            return;
        const rect = e.currentTarget.getBoundingClientRect();
        const total = g.end - g.start;
        const fraction = span / total;
        const pointerFraction = (e.clientX - rect.left) / Math.max(1, rect.width);
        const startFraction = Math.max(0, Math.min(1 - fraction, pointerFraction - drag.grabOffset));
        const nextStart = g.start + startFraction * total;
        changeRange(nextStart, nextStart + span);
    }
    function onNavigatorPointerUp(e) {
        navigatorDragRef.current = null;
        if (e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.releasePointerCapture(e.pointerId);
    }
    function onNavigatorKeyDown(e) {
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            e.preventDefault();
            pan(e.key === 'ArrowLeft' ? -0.1 : 0.1);
        }
        else if (e.key === 'Home') {
            e.preventDefault();
            changeRange(g.start, g.start + span);
        }
        else if (e.key === 'End') {
            e.preventDefault();
            changeRange(g.end - span, g.end);
        }
    }
    function inspectSite(site) {
        setSelected(site);
    }
    function exportCSV() {
        const cols = ['gene', 'chr', 'position', 'strand', 'transcript', 'motif', 'reads', 'raw', 'rank', 'line', 'status', 'source'];
        const esc = v => `"${String(v ?? '').replaceAll('"', '""')}"`;
        const csv = [cols.join(','), ...ordered.map(s => cols.map(k => esc(s[k])).join(','))].join('\n');
        const url = URL.createObjectURL(new Blob([csv], {
            type: 'text/csv'
        }));
        const link = document.createElement('a');
        link.href = url;
        link.download = `${mode === 'demo' ? 'SYNTHETIC_' : ''}${gene}_visible_sites.csv`;
        link.click();
        URL.revokeObjectURL(url);
    }
    const x = p => 128 + (p - a) / span * 1040;
    const tick = Array.from({
        length: 6
    }, (_, i) => a + span * i / 5);
    const tracks = lines.map((l, i) => ({
        line: l,
        sites: visible.filter(s => s.line === l),
        i
    }));
    const sortedIsoforms = useMemo(() => [...g.isoforms].sort((t1, t2) => {
        const n1 = rawSites.filter(s => s.transcript === t1.id).length;
        const n2 = rawSites.filter(s => s.transcript === t2.id).length;
        return n2 - n1 || t1.id.localeCompare(t2.id);
    }), [g, rawSites]);
    const shownIsoforms = transcriptView === 'all' ? sortedIsoforms : transcriptView === 'top' ? sortedIsoforms.slice(0, 5) : [];
    const exonUnion = useMemo(() => {
        const intervals = g.isoforms.flatMap(t => t.exons).map(([start,end]) => [start,end]).sort((a,b) => a[0]-b[0]);
        const result = [];
        for (const [start,end] of intervals) {
            if (result.length && start <= result[result.length-1][1]+1) result[result.length-1][1] = Math.max(end,result[result.length-1][1]);
            else result.push([start,end]);
        }
        return result;
    }, [g]);
    const densityBins = useMemo(() => {
        const bins = Array(60).fill(0);
        // Unique genomic loci within each bin; no summation of overlapping isoform read counts.
        const seen = new Set();
        for (const site of visible) {
            const key = `${site.chr}:${site.position}:${site.strand}`;
            if (seen.has(key)) continue;
            seen.add(key);
            const i = Math.min(59, Math.max(0, Math.floor((site.position - a) / Math.max(1, span) * 60)));
            bins[i] += 1;
        }
        return bins;
    }, [visible, a, span]);
    const maxDensity = Math.max(1, ...densityBins);
    let rows = [{
            label: 'Reference ruler',
            h: 36
        }, {
            label: 'Reference sequence',
            h: 32
        }, ...(transcriptView === 'compact' ? [{label: 'Gene exon union', h: 26, tx: {id: 'union', exons: exonUnion, utr: []}}] : shownIsoforms.map(t => ({
            label: t.id,
            h: 26,
            tx: t
        }))), {
            label: 'DRACH sites',
            h: 42
        }, ...tracks.map(t => ({
            label: `${t.line} predictions`,
            h: 47,
            track: t
        })), {
            label: 'Site density',
            h: 72
        }];
    let yy = 0;
    rows = rows.map(r => ({
        ...r,
        y: (yy += r.h) - r.h
    }));
    const H = yy + 16;
    return <div className="work-page">
    <div className="page-heading">
    <div>
    <div className="kicker">EXPLORE / GENOME</div>
    <h1>Genome Explorer <Badge kind={mode === 'demo' ? 'demo' : 'real'}/>
    </h1>
    <p>Explore candidate DRACH motifs, transcript structures, cell-line predictions and supporting evidence.</p>
    </div>
    <button className="secondary-btn" onClick={exportCSV}>Export visible sites CSV</button>
    </div>
    <div className={`explorer-layout ${expanded ? "" : "controls-collapsed"}`}>
    <aside className="filter-pane">
    <div className="filter-heading">Explore controls <button onClick={() => setExpanded(!expanded)}>{expanded ? 'Collapse' : 'Expand'}</button>
    </div>{expanded && <>
        <label className="field-label">Gene, transcript or locus</label>
        <form onSubmit={e => {
                e.preventDefault();
                navigate(query);
            }} className="sidebar-search">
        <input aria-label="Gene, transcript or locus" value={query} onChange={e => setQuery(e.target.value)} placeholder="PTBP1 or chr19:792000-806000"/>
        <button type="submit">Go <span aria-hidden="true">→</span>
        </button>
        </form>
        <select value={gene} onChange={e => {
                setGene(e.target.value);
                setWindow(null);
                setSelected(null);
            }} className="full-select">{(mode === 'local' && geneIndex ? geneIndex.genes.filter(x => GENES.some(g => g.name === x.symbol) || x.symbol === gene).slice(0, 20).map(x => ({name:x.symbol,id:x.gene_id})) : GENES.map(x => ({name:x.name,id:x.name}))).map(x => <option key={x.id} value={x.name}>{x.name}{x.id !== x.name ? ` (${x.id})` : ''}</option>)}</select>
        <label className="field-label">Data source</label>
        <select className="full-select" value={mode} onChange={e => {
                setMode(e.target.value);
                setWindow(null);
                setSelected(null);
            }}>
        <option value="demo">Synthetic demo data</option>
        <option value="local">Locally processed data</option>
        </select>
        <label className="field-label">Import processed local JSON</label>
        <input type="file" accept=".json,application/json" onChange={importFile} className="full-select"/>
            {mode === 'local' && <div className="field-help">Active local source: <select aria-label="Local data source" value={localSource} onChange={e => {
                    setLocalSource(e.target.value);
                    setSelected(null);
                }}><option value="file">public/data/processed/ gene file</option><option value="upload" disabled={!uploadedDoc}>Imported browser file</option></select></div>}
        {mode === 'local' && <p className="field-help">{geneIndex ? `${geneIndex.genes.length.toLocaleString()} indexed genes with predictions. Use the search box for any gene symbol, ENSG or ENST; dropdown shows pilot/current genes.` : `Gene index not loaded: ${indexError || 'loading…'}`}</p>}
        <p className="field-help">Local mode fetches Git-ignored public/data/processed/&lt;GENE&gt;.json. Files in public/ are copied into production builds: never deploy restricted data. Browser import remains available.</p>
        <section className="sidebar-section cell-line-section">
        <h3 className="sidebar-section-title">Cell-line tracks</h3>
        <div className="check-list">{LINES.map(l => <label key={l}>
            <input type="checkbox" checked={lines.includes(l)} onChange={e => setLines(old => e.target.checked ? [...old, l] : old.filter(x => x !== l))}/>{l}</label>)}</div>
        </section>
        <section className="sidebar-section motif-section">
        <div className="filter-title">
        <span>DRACH motifs</span>
        <button onClick={() => setMotifs(motifs.length === 18 ? [] : MOTIFS)}>{motifs.length === 18 ? 'Clear' : 'All'}</button>
        </div>
        <input className="full-select" placeholder="Filter motifs…" value={motifSearch} onChange={e => setMotifSearch(e.target.value.toUpperCase())}/>
        <div className="motif-scroll">{MOTIFS.filter(m => m.includes(motifSearch)).map(m => <label key={m}>
            <input type="checkbox" checked={motifs.includes(m)} onChange={e => setMotifs(old => e.target.checked ? [...old, m] : old.filter(x => x !== m))}/>
            <span className="motif-code">{m.slice(0, 2)}<b>{m[2]}</b>{m.slice(3)}</span>
            </label>)}</div>
        </section>
        <section className="sidebar-section threshold-section">
        <label className="field-label">Minimum read count: {minReads}</label>
        <input type="range" min="1" max="80" value={minReads} onChange={e => setMinReads(Number(e.target.value))}/>
        <label className="field-label">Minimum model score (raw): {cutoff.toFixed(1)}</label>
        <input type="range" min="-10" max="4" step=".5" value={cutoff} onChange={e => setCutoff(Number(e.target.value))}/>
        <label className="tiny-check">
        <input type="checkbox" checked={showUneval} onChange={e => setShowUneval(e.target.checked)}/> Show candidates without predictions (when score filter is off)</label>
        </section>
        </>}</aside>
    <div className="browser-main">{mode === 'local' && !realReady && <div className="error-strip">Real gene annotations not loaded. No synthetic geometry is shown for this gene.</div>}{loadError && <div className="error-strip">{loadError}</div>}<div className="stat-grid">{[['Candidate records', candidates.length], ['Sites with model predictions', visible.filter(s => s.status === 'evaluated').length], ['Not scored by model', visible.filter(s => s.status === 'unevaluated').length], ['Mean model score (raw)', visible.some(s => s.raw !== null) ? (visible.filter(s => s.raw !== null).reduce((n, s) => n + s.raw, 0) / visible.filter(s => s.raw !== null).length).toFixed(2) : '—']].map(([k, v]) => <div className="metric" key={k}>
        <span>{k}</span>
        <strong>{v}</strong>
        </div>)}</div>
    <Card
    title={`${gene} · ${g.chr}:${fmt(a)}–${fmt(b)}`}
    sub={`${g.strand} strand · ${g.isoforms.length} ${realReady ? 'GTF-derived' : 'synthetic'} isoforms · ${fmt(span)} bp window`}
    kind={realReady ? 'real' : mode === 'demo' ? 'demo' : 'placeholder'}
    extra={<div className="range-buttons">
        <button onClick={() => pan(-.25)}>←</button>
        <button onClick={() => zoom(1.5)}>−</button>
        <button onClick={() => zoom(.67)}>+</button>
        <button onClick={() => pan(.25)}>→</button>
        <button onClick={() => setWindow(null)}>Reset</button>
        </div>}>
    <div className="transcript-controls">
      <label>Transcript display <select aria-label="Transcript display mode" value={transcriptView} onChange={e => setTranscriptView(e.target.value)}>
        <option value="compact">Compact exon union</option><option value="top">Top 5 by prediction records</option><option value="all">All {g.isoforms.length} transcripts</option>
      </select></label>
      <label>Filter predictions <select aria-label="Filter predictions by transcript" value={activeTranscript} onChange={e => {setSelectedTranscript(e.target.value); setSelected(null);}}>
        <option value="all">All transcripts</option>{g.isoforms.map(t => <option key={t.id} value={t.id}>{t.id}</option>)}
      </select></label>
      {transcriptView === 'compact' && <span className="tiny-muted">Union of exons across isoforms; not a biological transcript</span>}
    </div>
    <div className="genome-svg-wrap">
    <svg viewBox={`0 0 1190 ${H}`} role="img" aria-label={mode === 'demo' ? 'Interactive genomic tracks; synthetic demonstration' : realReady ? 'Interactive genomic tracks; GTF-derived transcripts with model predictions' : 'Waiting for real transcript annotations'} onWheel={e => {
            if (e.deltaY)
                zoom(e.deltaY > 0 ? 1.18 : .85);
        }} onPointerDown={onTrackPointerDown} onPointerMove={onTrackPointerMove} onPointerUp={onTrackPointerUp} onPointerCancel={onTrackPointerUp} onLostPointerCapture={() => {
            dragRef.current = null;
        }} onClickCapture={e => {
            if (suppressClickRef.current) {
                e.stopPropagation();
                suppressClickRef.current = false;
            }
        }}>{rows.map((r, ri) => <g key={ri}>
        <rect x="0" y={r.y} width="1190" height={r.h} fill={ri % 2 ? '#fafbfe' : '#fff'}/>
        <text x="9" y={r.y + Math.min(22, r.h / 2 + 5)} fontSize={r.tx && r.tx.id !== 'union' ? '10' : '12'} fill="#596781"><title>{r.tx?.name ? `${r.label} (${r.tx.name})` : r.label}</title>{r.label}</text>{ri > 0 && <line x1="126" x2="1170" y1={r.y} y2={r.y} stroke="#eef1f7"/>}{r.label === 'Reference ruler' && tick.map((p, i) => <g key={i}>
            <line x1={x(p)} x2={x(p)} y1={r.y + 20} y2={H - 12} stroke="#edf0f5"/>
            <text x={x(p)} y={r.y + 15} fontSize="11" textAnchor={i === 0 ? 'start' : i === 5 ? 'end' : 'middle'} fill="#66758e">{fmt(p)}</text>
            </g>)}{r.label === 'Reference sequence' && <text x="150" y={r.y + 21} fontSize="12" fill="#8a93a6">Reference FASTA not loaded · sequence track placeholder (assembly must be verified)</text>}{r.tx && <>
            <line x1={x(Math.max(a, Math.min(...r.tx.exons.map(e => e[0]))))} x2={x(Math.min(b, Math.max(...r.tx.exons.map(e => e[1]))))} y1={r.y + 13} y2={r.y + 13} stroke="#7b879c" strokeWidth="2"/>{r.tx.exons.map(([s, e], j) => e < a || s > b ? null : <g key={j}>
                <rect x={Math.max(128, x(s))} y={r.y + 7} width={Math.max(1, Math.min(1168, x(e)) - Math.max(128, x(s)))} height="12" rx="2" fill="#6677c8"/>{r.tx.utr.some(u => u[0] === s) && <rect x={Math.max(128, x(s))} y={r.y + 10} width={Math.max(1, Math.min(1168, x(e)) - Math.max(128, x(s)))} height="6" fill="#b7c5e8"/>}</g>)}</>}{r.label === 'DRACH sites' && [...new Map(candidates.map(s => [`${s.chr}:${s.position}:${s.strand}:${s.motif}`, s])).values()].map(s => {
                const dimmed = motifColorMode === 'highlight' && s.motif !== highlightMotif;
                const color = dimmed ? '#cbd2df' : motifColor(s.motif);
                return <g data-site-marker="true" key={`${s.chr}:${s.position}:${s.strand}:${s.motif}`} onClick={() => inspectSite(s)} style={{
                        cursor: 'pointer'
                    }}>
                <line x1={x(s.position)} x2={x(s.position)} y1={r.y + 8} y2={r.y + 35} stroke={color} opacity={dimmed ? .35 : .85} strokeWidth="1.6"/>
                <circle cx={x(s.position)} cy={r.y + 22} r="3.4" fill={color} opacity={dimmed ? .5 : 1}/>
                <title>{`${s.motif} · ${s.chr}:${fmt(s.position)}`}</title>
                </g>;
            })}{r.track && r.track.sites.map(s => <g data-site-marker="true" key={s.id} onClick={() => inspectSite(s)} style={{
                    cursor: 'pointer'
                }}>
            <line x1={x(s.position)} x2={x(s.position)} y1={r.y + 8} y2={r.y + 38} stroke={s.status === 'evaluated' ? COLORS[r.track.i] : '#a9aebb'} opacity=".5"/>
            <circle cx={x(s.position)} cy={r.y + 23} r={selectedSite?.id === s.id ? 7 : 4} stroke="white" strokeWidth="1.2" fill={s.status === 'evaluated' ? COLORS[r.track.i] : '#c2c6cf'}/>
            <title>{`${s.line} · ${s.motif} · ${s.chr}:${fmt(s.position)}`}</title>
            </g>)}{r.label === 'Site density' && densityBins.map((count, i) => {
                const height = 50 * count / maxDensity;
                return <rect key={i} x={128 + i * 1040 / 60} y={r.y + 59 - height} width={1040 / 60 - 1} height={height} fill="#9bb6d8"><title>{`${count} distinct genomic candidate loci in bin ${i+1}`}</title></rect>;
            })}</g>)}</svg>
    </div>
    <div className="genome-viewport-navigation">
    <div className="genome-viewport-rail" role="slider" tabIndex={0} aria-label="Genomic viewport position" aria-valuemin={g.start} aria-valuemax={g.end - span} aria-valuenow={a} aria-valuetext={`${g.chr}:${fmt(a)}–${fmt(b)}`} title="Drag the colored window to pan; click the grey rail to reposition. Use arrow keys to pan." onPointerDown={onNavigatorPointerDown} onPointerMove={onNavigatorPointerMove} onPointerUp={onNavigatorPointerUp} onPointerCancel={onNavigatorPointerUp} onLostPointerCapture={() => {
            navigatorDragRef.current = null;
        }} onKeyDown={onNavigatorKeyDown}>
 <div className="genome-viewport-thumb" style={{
            left: `${(a - g.start) / (g.end - g.start) * 100}%`,
            width: `${span / (g.end - g.start) * 100}%`
        }}/>
    </div>
    <div className="genome-interaction-hint">Drag to pan <span>·</span> Scroll to zoom <span>·</span> Click a site to inspect</div>
    </div>
    <div className="genome-legend" aria-label="Genome browser legend">
    <h4 className="genome-legend-heading">Genome browser legend</h4>
    <div className="legend-section">
    <strong>DRACH motifs among filtered candidate records</strong>
    <div className="legend-items">{[['A-starting', '#D58A3A'], ['G-starting', '#2F9D88'], ['T-starting', '#8B6ED1']].map(([name, color]) => <span key={name} className="legend-item">
        <i className="legend-swatch" style={{
                background: color
            }}/>{name}</span>)}</div>
    <small>Sequence categories, not modification confidence. {motifColorMode === 'highlight' && `Highlighting ${highlightMotif}; other motifs dimmed.`}</small>
    <div className="legend-color-control">
    <label htmlFor="motif-color-mode">Color DRACH candidates by</label>
    <select id="motif-color-mode" value={motifColorMode} onChange={e => setMotifColorMode(e.target.value)}>
    <option value="group">Motif group (A / G / T starting)</option>
    <option value="highlight">Highlight selected motif</option>
    </select>
    {motifColorMode === 'highlight' && <select aria-label="Motif to highlight" value={highlightMotif} onChange={e => setHighlightMotif(e.target.value)}>{MOTIFS.map(m => <option key={m} value={m}>{m}</option>)}</select>}
    </div>
    </div>
    <div className="legend-section">
    <strong>Model prediction tracks</strong>
    <div className="legend-items">
    <span className="legend-item">
    <span className="legend-multicolor">{tracks.slice(0, 3).map(t => <i key={t.line} className="legend-swatch round" style={{
                background: COLORS[t.i]
            }}/>)}{!tracks.length && <i className="legend-swatch round" style={{
                background: '#6479d8'
            }}/>}</span>Scored by model (track color)</span>
    <span className="legend-item">
    <i className="legend-swatch round" style={{
            background: '#B8BECC'
        }}/>Not scored by model</span>
    <span className="legend-item">
    <i className="legend-swatch" style={{
            background: '#94B3D7'
        }}/>Candidate-site density</span>
    </div>
    <small>Cell-line tracks use distinct colors; see each track label. Unscored does not mean unmodified.</small>
    </div>
    </div>
    <div className="track-note">{realReady ? 'Exon geometry is GTF-derived. Density bars count distinct genomic loci among currently filtered prediction records; they are not sequencing coverage. Isoform-specific predictions at the same locus are counted once per bin.' : 'Synthetic demonstration: density bars count distinct displayed candidate loci, not sequencing coverage.'}</div>
    </Card>
    <div className="two-col">
    <Card title="Site inspector" sub="Click a marker in any cell-line track" kind={mode === 'demo' ? 'demo' : 'real'}>{selectedSite ? <div className="inspector-grid">{[['Cell line', selectedSite.line], ['Genomic locus', `${selectedSite.chr}:${fmt(selectedSite.position)}`], ['Transcript', selectedSite.transcript], ['Transcript offset', selectedSite.transcriptPosition ?? '—'], ['Strand', selectedSite.strand], ['DRACH motif', selectedSite.motif], ['Read count', selectedSite.reads], ['Status', selectedSite.status], ['Raw score', selectedSite.raw ?? 'Not scored'], ['Within-file rank', selectedSite.rank ?? 'Not scored'], ['Sample', selectedSite.sample ?? '—']].map(([k, v]) => <div key={k}>
            <span>{k}</span>
            <strong>{v}</strong>
            </div>)}</div> : <div className="placehold">Select a site to inspect its coordinates, transcript, motif, score and read support.</div>}</Card>
    <Card title="Sequence and annotation details" sub="Transcript structure and genomic context" kind={realReady ? "real" : "demo"}>
    <div className="info-list">{g.isoforms.map(t => <div key={t.id}>
        <b>{t.id}</b>
        <span>{t.exons.length} exons · {realReady ? 'Ensembl 91 exons · CDS/UTR not classified' : 'Synthetic UTR/CDS depiction'}</span>
        </div>)}</div>
    <div className="notice">{realReady ? 'Exon structures from Ensembl 91 GTF; transcript-relative 7-mers matched the reference FASTA in a 1,750-site sample. Genomic exon mapping and CDS/UTR classification require separate verification. Density track counts distinct filtered genomic candidate loci, not continuous sequencing coverage.' : 'Demo exon structures are illustrative. Switch to locally processed data to load real GTF-derived transcript geometry.'}</div>
    </Card>
    </div>
    <Card title="Visible-site table" sub="Filtered records in the current genomic window" kind={mode === 'demo' ? 'demo' : 'real'} extra={<span className="tiny-muted">{ordered.length} rows</span>}>
    <div className="table-overflow">
    <table className="clean-table">
    <thead>
    <tr>{[['line', 'Cell line'], ['position', 'Position'], ['motif', 'Motif'], ['reads', 'Reads'], ['raw', 'Model score (raw)'], ['rank', 'Rank'], ['status', 'Prediction status']].map(([k, label]) => <th key={k}>
        <button onClick={() => {
                setAscending(sort === k ? !ascending : true);
                setSort(k);
            }}>{label}{sort === k ? ascending ? ' ↑' : ' ↓' : ''}</button>
        </th>)}</tr>
    </thead>
    <tbody>{ordered.slice(0, 150).map(s => <tr key={s.id} onClick={() => setSelected(s)} className={selectedSite?.id === s.id ? 'selected-row' : ''}>
        <td>{s.line}</td>
        <td>{fmt(s.position)}</td>
        <td>{s.motif}</td>
        <td>{s.reads}</td>
        <td>{s.raw ?? '—'}</td>
        <td>{s.rank ?? '—'}</td>
        <td>{statusLabel(s.status)}</td>
        </tr>)}</tbody>
    </table>
    </div>{ordered.length > 150 && <p className="track-note">Showing first 150 of {ordered.length} rows; CSV export includes all filtered visible records.</p>}</Card>
    </div>
    </div>
    </div>;
}
export function ExtraComparison({ summary, onOpen }) {
    const [depth, setDepth] = useState('20+'), [selected, setSelected] = useState('Hct116');
    const genes = GENES.map(g => g.name);
    return <>
    <div className="two-col">
    <Card title="Gene × cell-line score heatmap" sub="Synthetic depth-matched raw-score illustration · not measured biological differences" kind="demo" extra={<select value={depth} onChange={e => setDepth(e.target.value)}>
        <option>20+</option>
        <option>10+</option>
        <option>All reads</option>
        </select>}>
    <Heatmap rows={genes} cols={LINES} value={(i, j) => -1.3 + 4 * rng(i * 23 + j * 11 + (depth === '20+' ? 1 : 3))} onCell={g => onOpen(g)}/>
    <p className="track-note">Click a gene row to open Genome Explorer. Values are synthetic.</p>
    </Card>
    <Card title="Site × cell-line comparison" sub="Shared candidate coordinates · synthetic example" kind="demo">
    <Heatmap rows={['Site 001', 'Site 002', 'Site 003', 'Site 004', 'Site 005']} cols={LINES} value={(i, j) => -1.4 + 4 * rng(i * 9 + j * 7)}/>
    <p className="track-note">Future real version requires transcript/strand-aware locus harmonization and explicit missing-data states.</p>
    </Card>
    </div>
    <div className="two-col">
    <Card title="Cell-line correlation matrix" sub="Illustrative correlations, not calculated from the SG-NEx prediction files" kind="demo">
    <Heatmap rows={LINES} cols={LINES} value={(i, j) => i === j ? 1 : .35 + .6 * rng(Math.min(i, j) * 9 + Math.max(i, j) * 3)} format={v => v.toFixed(2)}/>
    </Card>
    <Card title="Replicate consistency" sub="Sample-level agreement · synthetic demonstration" kind="demo">
    <label className="small-label">Cell line <select value={selected} onChange={e => setSelected(e.target.value)}>{LINES.map(l => <option key={l}>{l}</option>)}</select>
    </label>
    <PlotBars labels={['Replicate 1 vs 2', 'Replicate 1 vs 3', 'Replicate 2 vs 3']} values={[.55, .71, .64].map((x, i) => x + .12 * rng(i + LINES.indexOf(selected)))} color="#4f9c9a"/>
    <p className="track-note">Correlation coefficients shown here are invented. Replace using real per-sample outputs.</p>
    </Card>
    </div>
    <div className="two-col">
    <Card title="Top-ranked genes" sub="Synthetic ranking; click to inspect" kind="demo">
    <PlotBars labels={genes} values={genes.map((g, i) => 60 + 35 * rng(i + 4))} onPick={onOpen}/>
    </Card>
    <Card title="Depth-stratified model scores" sub="Illustration of why depth matching matters" kind="demo">
    <div className="select-inline">
    <select value={depth} onChange={e => setDepth(e.target.value)}>
    <option>20+</option>
    <option>10+</option>
    <option>All reads</option>
    </select>
    </div>
    <PlotBars labels={LINES} values={LINES.map((_, i) => .8 + 2.1 * rng(i + depth.length * 5))} color="#bd81aa"/>
    <p className="track-note">Synthetic scores on an uncalibrated raw-score scale.</p>
    </Card>
    </div>
    </>;
}
export function ExtraDiscovery({ summary, onOpen }) {
    const [line, setLine] = useState('Hct116');
    const genes = GENES.map(g => g.name);
    const rankedMotifs = [...MOTIFS].sort((a, b) => (b === 'GGACT') - (a === 'GGACT')).slice(0, 10);
    return <>
    <div className="two-col">
    <Card title="DRACH motif ranking" sub="Synthetic motif-level mean raw scores; GGACT is discussed in repository findings" kind="demo" extra={<select value={line} onChange={e => setLine(e.target.value)}>{LINES.map(l => <option key={l}>{l}</option>)}</select>}>
    <PlotBars labels={rankedMotifs} values={rankedMotifs.map((m, i) => m === 'GGACT' ? 2.8 : .5 + 2 * rng(i + LINES.indexOf(line) * 13))}/>
    <p className="track-note">Chart values are synthetic. The qualitative GGACT finding is documented in team analysis; exact per-motif statistics are not bundled.</p>
    </Card>
    <Card title="Exon-junction positional profile" sub="Illustrative positional trend; real aggregate series not available in dashboard summary" kind="demo">
    <ProfileChart />
    <p className="track-note">Synthetic distribution only. Replace with actual distance-to-junction bins when prepared.</p>
    </Card>
    </div>
    <div className="two-col">
    <Card title="Shared high-ranking genes" sub="Repository analysis reports 188 common top-ranked genes; this table is a demonstration" kind="mixed">
    <div className="info-list">{['POLDIP2', 'PTBP1', 'HSPA5'].map(g => <button key={g} className="link-row" onClick={() => onOpen(g)}>
        <strong>{g}</strong>
        <span>Open synthetic gene view →</span>
        </button>)}</div>
    <div className="notice">188 is a documented aggregate finding; the individual gene example views are synthetic and should not be interpreted as validated site calls.</div>
    </Card>
    <Card title="Gene case studies" sub="Explore hypotheses, not validated biological conclusions" kind="demo">{genes.map(g => <button key={g} className="case-row" onClick={() => onOpen(g)}>
        <span>
        <b>{g}</b>
        <small>Inspect synthetic transcript, motif and model-score tracks</small>
        </span>
        <span>Explore →</span>
        </button>)}</Card>
    </div>
    <Card title="Regional enrichment and metagene comparisons" sub="Dedicated layout for positional enrichment analysis" kind="placeholder">
    <div className="placeholder-chart">
    <span>Gene-body / 5′ UTR / CDS / 3′ UTR enrichment</span>
    <div className="placeholder-columns">
    <i />
    <i />
    <i />
    <i />
    <i />
    <i />
    <i />
    <i />
    </div>
    <p>Waiting for assembly-matched GTF, transcript-to-genome mapping and real candidate-site annotations. Synthetic stop-codon and exon-junction examples are shown above.</p>
    </div>
    </Card>
    </>;
}
function ProfileChart() {
    const pts = Array.from({
        length: 35
    }, (_, i) => {
        const x = i / 34 * 640 + 30;
        const v = .2 + .65 * Math.exp(-Math.pow((i - 17) / 7, 2)) + .08 * rng(i);
        return `${x},${180 - v * 150}`;
    }).join(' ');
    return <svg viewBox="0 0 710 220" className="mini-plot" role="img" aria-label="Synthetic exon-junction profile">
    <line x1="30" y1="185" x2="680" y2="185" stroke="#bdc7d7"/>
    <line x1="355" y1="15" x2="355" y2="185" stroke="#bbc4d7" strokeDasharray="4 5"/>
    <polyline points={pts} fill="none" stroke="#6377d3" strokeWidth="3"/>
    <text x="355" y="208" textAnchor="middle" fontSize="12" fill="#64748b">Distance from exon junction · demo</text>
    </svg>;
}
export function ExtraModel() {
    const [sample, setSample] = useState('Example A'), [depth, setDepth] = useState(20);
    return <>
    <div className="two-col">
    <Card title="Precision–recall evaluation" sub="Position reserved for real benchmark curves and PR AUC" kind="placeholder">
    <div className="placeholder-chart">
    <span>PR curve / PR AUC</span>
    <div className="plot-skeleton">
    <svg viewBox="0 0 400 160">
    <line x1="30" y1="10" x2="30" y2="140" stroke="#cbd5e1"/>
    <line x1="30" y1="140" x2="380" y2="140" stroke="#cbd5e1"/>
    <path d="M30 16 C 100 40, 130 78, 210 95 S 310 121, 380 132" stroke="#9da6c4" strokeWidth="2" fill="none" strokeDasharray="5 7"/>
    </svg>
    </div>
    <p>No verified evaluation curve bundled. This dashed curve is decorative, not a performance result. Never derive public curves from restricted labels without permission.</p>
    </div>
    </Card>
    <Card title="Model comparison" sub="Six ensemble members and alternative baselines" kind="placeholder">
    <div className="placeholder-chart">
    <span>Performance metrics table</span>
    <div className="skeleton-table">{['Model / method', 'PR AUC', 'Precision', 'Recall'].map(s => <div key={s}>{s}<em>—</em>
        </div>)}</div>
    <p>Awaiting authorized, reproducible evaluation outputs and dataset split definitions.</p>
    </div>
    </Card>
    </div>
    <div className="two-col">
    <Card title="Read-level evidence explorer" sub="Synthetic per-read signal features · not raw nanopore electrical current" kind="demo" extra={<select value={sample} onChange={e => setSample(e.target.value)}>
        <option>Example A</option>
        <option>Example B</option>
        <option>Example C</option>
        </select>}>
    <div className="read-matrix">{Array.from({
            length: Math.min(40, depth)
        }, (_, i) => <div className="read-row" key={i}>
        <span>read {i + 1}</span>{Array.from({
                length: 11
            }, (_, j) => <i key={j} style={{
                    background: `rgba(94,107,203,${.15 + .8 * rng(i * 7 + j * 3 + sample.length)})`
                }}/>)}</div>)}</div>
    <label className="small-label">Example reads: {depth} <input type="range" min="5" max="40" value={depth} onChange={e => setDepth(+e.target.value)}/>
    </label>
    <p className="track-note">Illustrative standardized read features. Actual signal data and processed read examples are not bundled.</p>
    </Card>
    <Card title="Own-read vs neighboring-context contribution" sub="Illustrative decomposition of an ensemble score" kind="demo">
    <PlotBars labels={['Own-read evidence', 'Neighbor contribution', 'Combined raw score']} values={[1.45, .42, 1.87].map((v, i) => v + (sample === 'Example B' ? .1 * i : sample === 'Example C' ? -.13 * i : 0))} color="#6c7bd2"/>
    <p className="track-note">Synthetic contributions, not attribution of real model predictions. Real outputs require the team's saved read-only and neighboring correction scores.</p>
    <div className="notice">Raw logits are not calibrated probabilities. Within-file rank scores cannot be directly compared across samples.</div>
    </Card>
    </div>
    </>;
}
