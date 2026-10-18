import { CompleteExplorer, ExtraComparison, ExtraDiscovery, ExtraModel, Badge } from './components/Advanced';
import { useEffect, useState } from 'react';
import { Activity, BarChart3, ChevronRight, Database, Dna, FlaskConical, Info, Layers, ShieldCheck, Sparkles } from 'lucide-react';
import { fmt, LINES } from './data/mockSites';
import { loadSummary } from './data/dataService';
const NAV = [{
        id: 'explorer',
        label: 'Genome Explorer',
        icon: Dna
    }, {
        id: 'compare',
        label: 'Comparative Analysis',
        icon: BarChart3
    }, {
        id: 'discover',
        label: 'Biological Discovery',
        icon: Sparkles
    }, {
        id: 'model',
        label: 'Model & Evidence',
        icon: Layers
    }];
const PALETTE = ['#6366f1', '#0891b2', '#e28d3b', '#d35b89', '#16a085', '#9a72c7', '#6b8e55'];
function Tag({ children, tone = 'default' }) {
    return <span className={`tag ${tone}`}>{children}</span>;
}
function Panel({ title, subtitle, children, action, className = '' }) {
    return <section className={`panel ${className}`}><div className="panel-head"><div><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section>;
}
function MiniStat({ label, value, sub, icon: Icon }) {
    return <div className="mini-stat"><div className="mini-stat-top"><span>{label}</span>{Icon && <Icon size={16}/>}</div><strong>{value}</strong>{sub && <small>{sub}</small>}</div>;
}
function Empty({ children }) {
    return <div className="empty">{children}</div>;
}
function ScoreChart({ summary, focus }) {
    if (!summary?.stop_codon_profile)
        return <Empty>Loading score profile…</Empty>;
    const { x, lines } = summary.stop_codon_profile;
    const selected = Object.entries(lines).filter(([k]) => !focus || k === focus);
    if (!selected.length)
        return <Empty>No profile for this cell line.</Empty>;
    const W = 850, H = 290, L = 54, R = 30, T = 16, B = 69, vals = selected.flatMap(([, v]) => v), lo = Math.floor(Math.min(...vals) * 2) / 2, hi = Math.ceil(Math.max(...vals) * 2) / 2 || 1;
    const sx = i => L + i / (x.length - 1) * (W - L - R), sy = v => H - B - (v - lo) / (hi - lo || 1) * (H - T - B);
    return <div className="chart-wrap"><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Real SG-NEx mean model score around stop codons"><rect x={L} y={T} width={W - L - R} height={H - T - B} fill="#fbfcff"/>{Array.from({
            length: 5
        }, (_, i) => {
            let v = lo + (hi - lo) * i / 4;
            return <g key={i}><line x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} stroke="#e9edf5"/><text x={L - 10} y={sy(v) + 4} textAnchor="end" className="chart-axis">{v.toFixed(1)}</text></g>;
        })}<line x1={sx(9.5)} x2={sx(9.5)} y1={T} y2={H - B} stroke="#a9b2c8" strokeDasharray="4 5"/>{selected.map(([name, ys]) => <polyline key={name} fill="none" stroke={PALETTE[LINES.indexOf(name)] || '#6366f1'} strokeWidth={focus ? 3 : 2.3} strokeOpacity={focus ? 1 : .78} strokeLinejoin="round" strokeLinecap="round" points={ys.map((v, i) => `${sx(i)},${sy(v)}`).join(' ')}><title>{name}</title></polyline>)}{[0, 4, 9, 14, 19].map(i => <text key={i} x={sx(i)} y={H - B + 22} textAnchor="middle" className="chart-axis">{x[i]}</text>)}<text x={(L + W - R) / 2} y={H - 14} textAnchor="middle" className="chart-axis chart-axis-title">Position relative to stop codon (nt)</text></svg><div className="legend">{selected.map(([name]) => <span key={name}><i style={{
                background: PALETTE[LINES.indexOf(name)]
            }}/>{name}</span>)}</div></div>;
}
function Comparison({ summary, loading, error }) {
    const [focus, setFocus] = useState('All');
    const rows = summary?.reads_per_line || [];
    const filtered = focus === 'All' ? rows : rows.filter(r => r.line === focus);
    const total = rows.reduce((n, r) => n + r.sites, 0);
    return <><div className="page-title"><div><div className="eyebrow">ANALYZE / CELL LINES</div><h1>Comparative Analysis <Tag tone="real">REPOSITORY DATA</Tag></h1><p>Explore aggregated SG-NEx candidate-site coverage and positional score patterns across seven cell lines.</p></div></div>{error && <div className="warning">{error}</div>}{loading ? <Empty>Loading real summary data…</Empty> : <><div className="stats-row wide"><MiniStat label="Cell lines" value={rows.length || '—'} sub="SG-NEx study" icon={Layers}/><MiniStat label="Candidate site records" value={total ? `${(total / 1e6).toFixed(2)}M` : '—'} sub="Sum across cell lines; not unique loci" icon={Dna}/><MiniStat label="Median read depth" value={rows.length ? `${Math.min(...rows.map(r => r.median_reads))}–${Math.max(...rows.map(r => r.median_reads))}` : '—'} sub="Range of cell-line medians" icon={Activity}/><MiniStat label="Read-depth caveat" value="1–2 reads" sub="Common among candidate sites" icon={Info}/></div><Panel title="Sequencing evidence by cell line" subtitle="Real pooled SG-NEx summary · proportions of candidate sites by read depth" action={<Tag tone="real">REAL AGGREGATES</Tag>}><div className="coverage-table"><div className="coverage-head"><span>Cell line</span><span>1–2 reads</span><span>20+ reads</span><span>Candidate sites</span></div>{rows.map((r, i) => <div className="coverage-row" key={r.line}><b><i style={{
                    background: PALETTE[i]
                }}/>{r.line}</b><div className="bar-cell"><div className="bar-bg"><div style={{
                    width: `${r.pct_1_2}%`,
                    background: '#f1a47e'
                }}/></div><span>{r.pct_1_2.toFixed(1)}%</span></div><div className="bar-cell"><div className="bar-bg"><div style={{
                    width: `${r.pct_20_plus}%`,
                    background: '#7a80d7'
                }}/></div><span>{r.pct_20_plus.toFixed(1)}%</span></div><strong>{fmt(r.sites)}</strong></div>)}</div><p className="subtle-foot">The two percentages are not complements: sites with 3–19 reads form the remaining category.</p></Panel><Panel title="Score profile around stop codons" subtitle="Mean model score relative to stop codon · repository summary, sites with 10+ reads" action={<select className="compact-select" value={focus} onChange={e => setFocus(e.target.value)}><option>All</option>{rows.map(r => <option key={r.line}>{r.line}</option>)}</select>}><ScoreChart summary={summary} focus={focus === 'All' ? null : focus}/><p className="subtle-foot">These are mean raw-model-score profiles, not modification probabilities or differential modification tests.</p></Panel><Panel title="Full summary table" subtitle="Source: dashboard/public/data/summary.json"><div className="table-scroll"><table className="data-table roomy"><thead><tr><th>Cell line</th><th>Candidate sites</th><th>Median reads</th><th>1–2 reads</th><th>20+ reads</th></tr></thead><tbody>{filtered.map(r => <tr key={r.line}><td><strong>{r.line}</strong></td><td>{fmt(r.sites)}</td><td>{r.median_reads}</td><td>{r.pct_1_2.toFixed(2)}%</td><td>{r.pct_20_plus.toFixed(2)}%</td></tr>)}</tbody></table></div></Panel></>}</>;
}
function Discovery({ summary }) {
    const rows = summary?.reads_per_line || [];
    return <><div className="page-title"><div><div className="eyebrow">DISCOVER / PATTERNS</div><h1>Biological Discovery <Tag tone="mixed">EVIDENCE + EXPLAINERS</Tag></h1><p>Understand biologically motivated patterns while separating measured summaries from illustrations.</p></div></div><div className="discovery-grid"><Panel title="What is a DRACH motif?" subtitle="The five-base sequence context used to define candidate m6A sites" action={<Tag>REFERENCE CONCEPT</Tag>}><div className="motif-hero"><span>D</span><span>R</span><span className="central-a">A</span><span>C</span><span>H</span></div><div className="motif-defs"><span><b>D</b> A / G / T</span><span><b>R</b> A / G</span><span><b>A</b> candidate site</span><span><b>C</b> C</span><span><b>H</b> A / C / T</span></div><p className="explain">The 18 DRACH combinations represent candidate sequence contexts. A DRACH motif alone does not prove that its central adenosine is modified.</p></Panel><Panel title="Coverage matters" subtitle="A critical constraint on interpreting model scores" action={<Tag tone="real">REAL AGGREGATES</Tag>}><div className="big-fact">{rows.length ? (rows.reduce((n, r) => n + r.pct_1_2, 0) / rows.length).toFixed(1) + '%' : '—'}<span>mean of cell-line percentages with just 1–2 reads</span></div><p className="explain">This is an unweighted mean across the seven cell-line summaries, not a pooled site-level percentage. Sparse coverage limits confidence in individual site predictions.</p></Panel><Panel title="Where might m6A signals accumulate?" subtitle="Actual positional model-score summary from the repository" action={<Tag tone="real">REAL AGGREGATES</Tag>}><ScoreChart summary={summary}/><p className="explain">The plotted values are model scores averaged by position relative to stop codons. They do not establish site-level modification or causal enrichment.</p></Panel><Panel title="Candidate discovery workflow" subtitle="A responsible interpretation pipeline" action={<Tag>GUIDANCE</Tag>}><div className="steps"><div><b>01</b><span><strong>Locate</strong><small>Identify a candidate DRACH site in a transcript.</small></span></div><div><b>02</b><span><strong>Check evidence</strong><small>Inspect coverage, raw score, transcript mapping, and sample support.</small></span></div><div><b>03</b><span><strong>Compare cautiously</strong><small>Use matched read-depth strata and independent replicates.</small></span></div><div><b>04</b><span><strong>Validate</strong><small>Use orthogonal evidence before claiming modification differences.</small></span></div></div></Panel></div></>;
}
function Model() {
    return <><div className="page-title"><div><div className="eyebrow">METHODS / INTERPRETATION</div><h1>Model & Evidence <Tag tone="mixed">DOCUMENTED + DEMO</Tag></h1><p>Learn what the ensemble scores represent and what the visualization cannot infer.</p></div></div><div className="model-flow"><div><Database size={22}/><strong>Nanopore RNA reads</strong><span>Processed per-site evidence</span></div><ChevronRight size={22}/><div><Layers size={22}/><strong>Six-network ensemble</strong><span>Site and neighboring context</span></div><ChevronRight size={22}/><div><Activity size={22}/><strong>Model outputs</strong><span>Raw and within-file rank scores</span></div><ChevronRight size={22}/><div><Dna size={22}/><strong>Candidate sites</strong><span>Interpret with read coverage</span></div></div><div className="discovery-grid"><Panel title="Raw score vs. rank score" subtitle="Two distinct outputs from the team's SG-NEx scoring pipeline"><div className="definition"><Tag>score_raw</Tag><p>Mean of the ensemble's raw outputs. On a fixed model scale, but not a calibrated modification probability. Depth effects must be considered when comparing samples.</p></div><div className="definition"><Tag>score_rank</Tag><p>Average within-file site rank. Useful for ordering candidates within the same scored file, but rank values are not directly comparable across files or cell lines.</p></div></Panel><Panel title="What is not available yet?" subtitle="The current demo does not have real site-level prediction files"><div className="checklist"><div><ShieldCheck size={18}/> Real aggregate coverage statistics are available.</div><div><ShieldCheck size={18}/> Stop-codon score profiles are available.</div><div><FlaskConical size={18}/> Site positions, read-depth tracks and scores in the browser are synthetic.</div><div><Info size={18}/> True site-level views require SG-NEx outputs and verified transcript-to-genome mapping.</div></div></Panel><Panel title="Avoid over-interpreting results" subtitle="Essential caveats"><ul className="caveat-list"><li>High score does not mean 100% of RNA molecules are modified.</li><li>Missing or unevaluated sites are not confirmed unmodified sites.</li><li>Very low read counts make predictions less stable.</li><li>Multiple transcripts can refer to the same genomic locus.</li><li>Different cell-line scores do not by themselves establish differential m6A.</li></ul></Panel><Panel title="Data provenance & privacy" subtitle="Local-first development policy"><div className="provenance"><div><Tag tone="real">REAL</Tag><span>Approved repository aggregate summaries</span></div><div><Tag tone="demo">SYNTHETIC</Tag><span>Demo gene tracks and candidate-site records</span></div><div><Tag tone="default">PRIVATE</Tag><span>Restricted data0/data1/data2 labels: never bundled or published</span></div></div><p className="explain">The website cannot securely hide files that it sends to public browsers. Keep restricted inputs and labels outside public assets and Git history.</p></Panel></div></>;
}
export default function App() {
    const [page, setPage] = useState('explorer');
    const [jump, setJump] = useState(null);
    const [summary, setSummary] = useState(null), [error, setError] = useState(''), [loading, setLoading] = useState(true);
    useEffect(() => {
        loadSummary().then(setSummary).catch(e => setError(e.message)).finally(() => setLoading(false));
    }, []);
    const openGene = gene => {
        setJump({
            gene,
            stamp: Date.now()
        });
        setPage('explorer');
        window.scrollTo?.({
            top: 0,
            behavior: 'smooth'
        });
    };
    return <div className="new-shell"><header className="new-header"><div className="header-inner"><div className="brand-inline"><div className="brand-mark"><Dna size={23}/></div><div><strong>m6A Explorer</strong><small>DSA4262 · Team Foursight</small></div></div><div className="header-right"><Badge kind="mixed">REAL SUMMARY + SYNTHETIC SITES</Badge><a href="https://github.com/AsJayTee/DSA4262-Foursight" target="_blank" rel="noreferrer">Repository ↗</a></div></div><nav className="horizontal-nav" aria-label="Dashboard pages">{NAV.map(n => {
            const Icon = n.icon;
            return <button key={n.id} className={page === n.id ? 'selected' : ''} aria-current={page === n.id ? 'page' : undefined} onClick={() => setPage(n.id)}><Icon size={17}/>{n.label}</button>;
        })}</nav></header><main className="new-main">{page === 'explorer' ? <CompleteExplorer jump={jump}/> : page === 'compare' ? <><Comparison summary={summary} loading={loading} error={error}/><ExtraComparison summary={summary} onOpen={openGene}/></> : page === 'discover' ? <><Discovery summary={summary}/><ExtraDiscovery summary={summary} onOpen={openGene}/></> : <><Model /><ExtraModel /></>}<footer className="new-footer"><span>m6A Explorer · DSA4262 Foursight</span><span>Source badges identify real repository summaries, synthetic demonstrations, and placeholders. Restricted labels are excluded.</span></footer></main></div>;
}
