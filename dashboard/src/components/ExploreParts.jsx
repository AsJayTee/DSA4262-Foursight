import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HOLDS_TOP, LOW_DEPTH, lineName } from "../lib/sites.js";
import { InfoTip, Tip } from "./Tip.jsx";

const clamp01 = (v) => Math.max(0, Math.min(1, v));
// Top 1% -> 1, top 50% or lower -> 0: visual weight of a rank.
export const strength = (rank) => clamp01(1 - Math.log10(Math.max((1 - rank) * 100, 1)) / Math.log10(50));
const heat = (t) => `color-mix(in srgb, var(--accent) ${Math.round(clamp01(t) * 100)}%, var(--node-low))`;
const comma = (n) => Math.round(n).toLocaleString("en-US");
// Percentile as "top x%" without the word: 0.13, 2.4, 15.
export const topPct = (rank) => {
  const t = (1 - rank) * 100;
  return t < 0.1 ? "<0.1" : t < 9.95 ? t.toFixed(t < 1 ? 2 : 1) : `${Math.round(t)}`;
};

/* ---------------- small shared pieces ---------------- */

// Strong figures (top 1%) are emphasised so the eye goes to them first.
const hl = (rank) => ((1 - rank) * 100 <= 1 ? "hl" : "");

export function Seq({ kmer }) {
  const s = kmer.replace(/T/g, "U");
  return <code className="seq">{s.slice(0, 3)}<b>{s[3]}</b>{s.slice(4)}</code>;
}

export function Depth({ reads }) {
  return reads <= LOW_DEPTH
    ? <span className="depth low" title="Depth ≤ 2: low-confidence rank">{reads}<span className="low-tag">low</span></span>
    : <span className="depth">{reads}</span>;
}

export function HoldsDots({ row }) {
  return (
    <span className="holds" aria-label={`Within the gene's top ${HOLDS_TOP} in ${row.holds.held} of ${row.holds.measured} replicates`}>
      {row.batches.map((b) => (
        <span key={b.name} className={b.rank == null ? "dot none" : b.position <= HOLDS_TOP ? "dot held" : "dot"}
          title={b.rank == null ? `${b.name}: not covered` : `${b.name}: rank ${b.position} in gene, depth ${b.reads}`} />
      ))}
    </span>
  );
}

const BAND = { full: "≥ 20 reads", "10 reads": "10–19 reads", "3 reads": "3–9 reads", "1 reads": "1–2 reads" };
export function Reliability({ trust }) {
  const m = trust.motif.replace(/T/g, "U");
  const text = trust.value == null
    ? <><strong>Not determined.</strong> Too few labelled positives for {m} to estimate accuracy.</>
    : <><strong>Pairwise accuracy {trust.value.toFixed(2)}</strong> for {m} sites at {BAND[trust.band]}: probability that a
      modified site ranks above an unmodified one (held-out genes, worse of two cell lines). High ≥ 0.84, moderate ≥ 0.75.</>;
  return <Tip text={text} className={`rel ${trust.label === "n.d." ? "nd" : trust.label}`} label={`Reliability ${trust.label}`}>{trust.label}</Tip>;
}

/* Column definitions shared by the shortlist and the full table. */
export const COLUMNS = (line) => [
  { key: "rank", head: "#", help: "Rank within the gene" },
  { key: "pos", head: "Position", help: "Genomic coordinate of the candidate A (GRCh38)" },
  { key: "seq", head: "7-mer", help: "Sequence context; the candidate A is highlighted" },
  { key: "region", head: "Region", help: "Transcript region (5′ UTR, CDS, 3′ UTR, non-coding)" },
  { key: "depth", head: "Depth", help: "Reads at the site, all samples of the line pooled. ≤ 2: low confidence", num: true },
  { key: "top", head: `Top % (${lineName(line)})`, help: "Rank percentile among all candidate sites in this cell line (smaller = higher)", num: true },
  { key: "reps", head: "Replicates", help: `One dot per biological replicate: filled if within the gene's top ${HOLDS_TOP} when that replicate is ranked alone; grey if not covered` },
  { key: "rel", head: "Reliability", help: "Held-out pairwise accuracy for this motif and depth: high ≥ 0.84, moderate ≥ 0.75, low below; n.d. if not estimable" },
];

export function SiteCells({ r, chrom }) {
  return (
    <>
      <td className="rk">{r.geneRank}</td>
      <td className="mono">{chrom}:{comma(r.pos)}</td>
      <td><Seq kmer={r.kmer} /></td>
      <td className="region">{r.region === "coding" ? "CDS" : r.region}</td>
      <td className="num"><Depth reads={r.reads} /></td>
      <td className="num">{topPct(r.rank)}</td>
      <td><HoldsDots row={r} /></td>
      <td><Reliability trust={r.trust} /></td>
    </>
  );
}

export function Head({ line, lead }) {
  return (
    <thead>
      <tr>
        {lead && <th className="pick"><span className="sr-only">Select</span></th>}
        {COLUMNS(line).map((c) => (
          <th key={c.key} className={c.num ? "num" : ""}><span className="th-in">{c.head}<InfoTip text={c.help} label={`About ${c.head}`} /></span></th>
        ))}
      </tr>
    </thead>
  );
}

/* ---------------- the gene viewer: zoom, pan, coordinates ---------------- */

const niceStep = (span) => {
  const raw = span / 5, p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
};
const parseCoord = (s) => {
  const n = Number(String(s).replace(/[,\s]/g, ""));
  return Number.isFinite(n) ? n : null;
};

export function GeneViewer({ structure, strand, chrom, rows, picked, onOpen, focus }) {
  const [g0, g1] = structure.span;
  const pad = Math.max(50, Math.round((g1 - g0) * 0.02));
  const lo = g0 - pad, hi = g1 + pad, MIN = 60;
  const [view, setView] = useState([lo, hi]);
  const [hover, setHover] = useState(null);
  const [draft, setDraft] = useState(null);           // edited start/end text, before applying
  const svgRef = useRef(null);
  const W = 1000, H = 150, L = 16, R = 16, axis = 112;
  const clampView = useCallback(([a, b]) => {
    let span = Math.min(Math.max(b - a, MIN), hi - lo);
    let s = Math.max(lo, Math.min(a, hi - span));
    return [s, s + span];
  }, [lo, hi]);
  useEffect(() => setView([lo, hi]), [lo, hi]);
  // Open a site from the table: bring it into view (same zoom), unless already visible.
  useEffect(() => {
    if (focus == null) return;
    setView((v) => (focus > v[0] && focus < v[1] ? v : clampView([focus - (v[1] - v[0]) / 2, focus + (v[1] - v[0]) / 2])));
  }, [focus, clampView]);

  const [a, b] = view;
  const flip = strand === "-";                          // drawn 5' to 3', left to right
  const x = (p) => L + ((flip ? b - p : p - a) / (b - a)) * (W - L - R);
  const toGenome = (px) => {
    const f = (px - L) / (W - L - R);
    return flip ? b - f * (b - a) : a + f * (b - a);
  };
  const zoom = (factor, centre = (a + b) / 2) => {
    const span = (b - a) * factor;
    const f = (centre - a) / (b - a);
    setView(clampView([centre - f * span, centre + (1 - f) * span]));
  };

  // Ctrl/⌘ + wheel zooms at the pointer; horizontal wheel or shift + wheel pans. A plain
  // vertical wheel scrolls the page as usual.
  useEffect(() => {
    const el = svgRef.current;
    const onWheel = (e) => {
      const rect = el.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * W;
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const c = toGenome(px);
        const factor = Math.exp(e.deltaY * 0.002);
        setView(([va, vb]) => {
          const span = (vb - va) * factor, f = (c - va) / (vb - va);
          return clampView([c - f * span, c + (1 - f) * span]);
        });
      } else if (Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey) {
        e.preventDefault();
        const d = (e.shiftKey ? e.deltaY : e.deltaX) / rect.width;
        setView(([va, vb]) => clampView([va + (flip ? -d : d) * (vb - va), vb + (flip ? -d : d) * (vb - va)]));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  });

  // Drag to pan.
  const drag = useRef(null);
  const onDown = (e) => {
    if (e.target.closest(".tick")) return;
    drag.current = { x: e.clientX, view };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = (e) => {
    if (!drag.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const d = ((e.clientX - drag.current.x) / rect.width) * ((W) / (W - L - R));
    const [va, vb] = drag.current.view, shift = (flip ? d : -d) * (vb - va);
    setView(clampView([va + shift, vb + shift]));
  };
  const onUp = () => { drag.current = null; };

  const span = b - a, step = niceStep(span);
  const ticks = [];
  for (let t = Math.ceil(a / step) * step; t <= b; t += step) ticks.push(t);
  const visible = rows.filter((r) => r.pos >= a && r.pos <= b);
  const labelSites = span < 400;
  const cds = structure.cds;
  const box = (s, e, y, h, cls, key) => {
    const x1 = x(Math.max(s, a)), x2 = x(Math.min(e, b));
    if (Math.min(e, b) < Math.max(s, a)) return null;
    return <rect key={key} className={cls} x={Math.min(x1, x2)} y={y} width={Math.max(Math.abs(x2 - x1), 1.5)} height={h} rx="2" />;
  };

  // Overview strip: the whole gene, with the current window.
  const OW = 1000, ox = (p) => L + ((flip ? hi - p : p - lo) / (hi - lo)) * (OW - L - R);
  // Grab the window to drag it (the grabbed point stays under the pointer); press anywhere
  // else to centre the window there, then keep dragging if you like.
  const oDrag = useRef(null);
  const overviewGenome = (clientX, rect) => {
    const f = ((clientX - rect.left) / rect.width) * (OW / (OW - L - R)) - L / (OW - L - R);
    return flip ? hi - f * (hi - lo) : lo + f * (hi - lo);
  };
  const onOverviewDown = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const g = overviewGenome(e.clientX, rect);
    if (g >= a && g <= b) oDrag.current = { offset: g - a };
    else {
      const next = clampView([g - span / 2, g + span / 2]);
      setView(next);
      oDrag.current = { offset: g - next[0] };
    }
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onOverviewMove = (e) => {
    if (!oDrag.current) return;
    const g = overviewGenome(e.clientX, e.currentTarget.getBoundingClientRect());
    setView(clampView([g - oDrag.current.offset, g - oDrag.current.offset + span]));
  };

  // Applied when focus leaves both fields (or on Enter), so start and end can be edited
  // in turn. An end at or before the start keeps the current width from the new start.
  const apply = () => {
    if (!draft) return;
    const s = parseCoord(draft.start), e = parseCoord(draft.end);
    if (s != null && e != null) setView(clampView(e > s ? [s, e] : [s, s + span]));
    setDraft(null);
  };
  const field = (k, v) => (
    <input className="coord" inputMode="numeric" aria-label={k === "start" ? "Start coordinate" : "End coordinate"}
      value={draft ? draft[k] : comma(v)}
      onFocus={() => setDraft((d) => d || { start: comma(a), end: comma(b) })}
      onChange={(e) => setDraft((d) => ({ ...(d || { start: comma(a), end: comma(b) }), [k]: e.target.value }))}
      onKeyDown={(e) => {
        if (e.key === "Enter") { apply(); e.currentTarget.blur(); }
        if (e.key === "Escape") { setDraft(null); e.currentTarget.blur(); }
      }}
      onBlur={(e) => { if (!e.relatedTarget?.classList?.contains("coord")) apply(); }} />
  );

  return (
    <section className="card viewer" aria-label="Gene viewer">
      <div className="viewer-bar">
        <div className="coords">
          <span className="mono muted">chr{chrom}:</span>{field("start", a)}<span className="muted">–</span>{field("end", b)}
          <span className="muted small">{span >= 1000 ? `${(span / 1000).toFixed(span < 1e4 ? 2 : 1)} kb` : `${Math.round(span)} nt`}</span>
        </div>
        <div className="zoom">
          <button onClick={() => zoom(0.5)} aria-label="Zoom in" title="Zoom in (or Ctrl/⌘ + scroll)">+</button>
          <button onClick={() => zoom(2)} aria-label="Zoom out" title="Zoom out">−</button>
          <button className="text" onClick={() => setView([lo, hi])}>Whole gene</button>
        </div>
      </div>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="viewer-main" role="img"
        aria-label={`${visible.length} sites between ${comma(a)} and ${comma(b)}`}
        onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        <defs><clipPath id="vclip"><rect x={L} y="0" width={W - L - R} height={H} /></clipPath></defs>
        {ticks.map((t) => {
          // Labels too close to either edge would be cut off: keep the grid line, drop the label.
          const X = x(t), room = X > L + 45 && X < W - R - 45;
          return (
            <g key={t}>
              <line className="vgrid" x1={X} x2={X} y1={18} y2={axis + 16} />
              {room && <text className="vtick" x={X} y={12} textAnchor="middle">{comma(t)}</text>}
            </g>
          );
        })}
        <g clipPath="url(#vclip)">
          <line className="intron" x1={L} x2={W - R} y1={axis + 6} y2={axis + 6} />
          {structure.exons.map(([s, e], i) => box(s, e, axis, 12, "exon", `e${i}`))}
          {cds && structure.exons.map(([s, e], i) => {
            const p = Math.max(s, cds[0]), q = Math.min(e, cds[1]);
            return p <= q ? box(p, q, axis - 3, 18, "exon cds", `c${i}`) : null;
          })}
          {visible.slice().reverse().map((r) => {
            const h = 12 + 62 * strength(r.rank), on = picked.has(r.site), X = x(r.pos);
            return (
              <g key={r.site} className={`tick${r.low ? " low" : ""}${focus === r.pos ? " focus" : ""}`} onClick={() => onOpen(r)}
                onMouseEnter={() => setHover(r)} onMouseLeave={() => setHover(null)}>
                <rect x={X - 6} y={axis - 92} width="12" height="92" fill="transparent" />
                <line x1={X} x2={X} y1={axis - 4} y2={axis - 4 - h} style={{ stroke: heat(strength(r.rank)) }} strokeWidth={on ? 3.5 : 2} />
                {on && <circle cx={X} cy={axis - 9 - h} r="3.5" className="tick-pick" />}
                {labelSites && <text className="tick-label" x={X} y={axis - 14 - h} textAnchor="middle">{r.geneRank}</text>}
              </g>
            );
          })}
        </g>
        <text className="vtick" x={L} y={H - 4}>5′</text>
        <text className="vtick" x={W - R} y={H - 4} textAnchor="end">3′</text>
      </svg>
      <svg viewBox={`0 0 ${OW} 26`} className="viewer-overview" aria-label="Whole gene; drag the window to move along it"
        onPointerDown={onOverviewDown} onPointerMove={onOverviewMove}
        onPointerUp={() => { oDrag.current = null; }} onPointerCancel={() => { oDrag.current = null; }}>
        <line className="intron" x1={L} x2={OW - R} y1="13" y2="13" />
        {structure.exons.map(([s, e], i) => (
          <rect key={i} className="exon" x={Math.min(ox(s), ox(e))} y="9" width={Math.max(Math.abs(ox(e) - ox(s)), 1)} height="8" />
        ))}
        {rows.map((r) => <line key={r.site} className="o-site" x1={ox(r.pos)} x2={ox(r.pos)} y1="3" y2="8" style={{ stroke: heat(strength(r.rank)) }} />)}
        <rect className="o-window" x={Math.min(ox(a), ox(b))} y="1" width={Math.max(Math.abs(ox(b) - ox(a)), 3)} height="24" rx="3" />
      </svg>
      <div className="viewer-legend" aria-label="Legend">
        <span><i className="lg-cds" />CDS exon</span>
        <span><i className="lg-utr" />UTR / non-coding exon</span>
        <span><i className="lg-intron" />intron</span>
        <span><i className="lg-site" />site: height and colour scale with rank</span>
        <span><i className="lg-low" />depth ≤ 2</span>
        <span><i className="lg-pick" />selected</span>
      </div>
      <p className="viewer-foot muted" aria-live="polite">
        {hover
          ? <>Site {hover.geneRank} · <Seq kmer={hover.kmer} /> · {hover.region === "coding" ? "CDS" : hover.region} · depth {hover.reads} · top {topPct(hover.rank)}%</>
          : `${visible.length} of ${rows.length} sites in view · drag to pan · click a site for details`}
      </p>
    </section>
  );
}

/* ---------------- the full table ---------------- */

export function SiteTable({ rows, picked, onToggle, onOpen, line, chrom, openSite }) {
  const [hideLow, setHideLow] = useState(false);
  const [region, setRegion] = useState("all");
  const [showAll, setShowAll] = useState(false);
  const [motif, setMotif] = useState("all");
  const regions = useMemo(() => ["all", ...new Set(rows.map((r) => r.region))], [rows]);
  // Motifs present in this gene, most frequent first, with their counts.
  const motifs = useMemo(() => {
    const n = {};
    rows.forEach((r) => { const m = r.kmer.slice(1, 6); n[m] = (n[m] || 0) + 1; });
    return Object.entries(n).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }, [rows]);
  useEffect(() => { setMotif("all"); }, [rows]);
  const shown = rows.filter((r) => (!hideLow || !r.low) && (region === "all" || r.region === region)
    && (motif === "all" || r.kmer.slice(1, 6) === motif));
  const lowCount = rows.filter((r) => r.low).length;
  const visible = showAll ? shown : shown.slice(0, 20);
  return (
    <section className="card site-table" aria-labelledby="all-sites">
      <div className="table-head">
        <h2 id="all-sites">All candidate sites <span className="muted">({shown.length === rows.length ? rows.length : `${shown.length} of ${rows.length}`})</span></h2>
        <div className="filters">
          {lowCount > 0 && (
            <label className="check">
              <input type="checkbox" checked={hideLow} onChange={(e) => setHideLow(e.target.checked)} />
              Exclude depth ≤ {LOW_DEPTH} ({lowCount})
            </label>
          )}
          <label className="select">
            Region
            <select value={region} onChange={(e) => setRegion(e.target.value)}>
              {regions.map((r) => <option key={r} value={r}>{r === "all" ? "All" : r === "coding" ? "CDS" : r}</option>)}
            </select>
          </label>
          <label className="select">
            Motif
            <select value={motif} onChange={(e) => setMotif(e.target.value)}>
              <option value="all">All</option>
              {motifs.map(([m, c]) => <option key={m} value={m}>{m.replace(/T/g, "U")} ({c})</option>)}
            </select>
          </label>
        </div>
      </div>
      <div className="table-scroll">
        <table>
          <Head line={line} lead />
          <tbody>
            {visible.map((r) => (
              <tr key={r.site} className={`${r.low ? "low" : ""} ${picked.has(r.site) ? "picked" : ""} ${openSite === r.site ? "open" : ""}`}
                onClick={() => onOpen(r)}>
                <td className="pick" onClick={(e) => e.stopPropagation()}>
                  <input type="checkbox" checked={picked.has(r.site)} onChange={() => onToggle(r.site)}
                    aria-label={`Select site ${r.geneRank}`} />
                </td>
                <SiteCells r={r} chrom={`chr${chrom}`} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {shown.length > 20 && (
        <button className="btn small show-all" onClick={() => setShowAll(!showAll)}>
          {showAll ? "Show top 20" : `Show all ${shown.length}`}
        </button>
      )}
    </section>
  );
}

/* ---------------- the site panel ---------------- */

// A percentile ruler on a log scale: 100% (left) to 0.1% (right). Markers sit at "top x%".
function Ruler({ marks, compact }) {
  const pos = (rank) => {
    const t = Math.min(100, Math.max(0.1, (1 - rank) * 100));
    return (1 - (Math.log10(t) + 1) / 3) * 100;     // 100% -> 0, 0.1% -> 100
  };
  return (
    <div className={compact ? "ruler compact" : "ruler"}>
      <div className="ruler-track">
        {!compact && [100, 10, 1, 0.1].map((t) => <span key={t} className="ruler-tick" style={{ left: `${pos(1 - t / 100)}%` }}><i>{t}%</i></span>)}
        {marks.filter((m) => m.rank != null).map((m) => (
          <span key={m.key} className={`ruler-mark ${m.kind}`} style={{ left: `${pos(m.rank)}%` }} title={`${m.label}: top ${topPct(m.rank)}%`} />
        ))}
      </div>
    </div>
  );
}

export function SitePanel({ row, rows, gene, strand, line, onClose, onMove }) {
  const ref = useRef(null);
  const i = rows.findIndex((r) => r.site === row.site);
  const prev = rows[i - 1], next = rows[i + 1];
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && prev) onMove(prev);
      if (e.key === "ArrowRight" && next) onMove(next);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onMove, prev, next]);
  useEffect(() => { ref.current?.focus(); setCopied(false); }, [row]);
  const coord = `chr${gene.chrom}:${comma(row.pos)}`;
  const copy = () => navigator.clipboard?.writeText(`${coord} (${strand})`).then(() => setCopied(true));
  const t = row.trust;
  const lines = [{ line, rank: row.rank, reads: row.reads, self: true }, ...row.elsewhere];

  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="site-panel" role="dialog" aria-modal="true" aria-labelledby="panel-title" tabIndex="-1" ref={ref}>
        <div className="sp-top">
          <span className="muted small">{gene.name} · {lineName(line)}</span>
          <div className="sp-nav">
            <button onClick={() => prev && onMove(prev)} disabled={!prev} aria-label="Previous site">‹</button>
            <span className="small">{row.geneRank} / {rows.length}</span>
            <button onClick={() => next && onMove(next)} disabled={!next} aria-label="Next site">›</button>
            <button onClick={onClose} aria-label="Close" className="sp-close">×</button>
          </div>
        </div>

        <header className="sp-head">
          <h2 id="panel-title"><Seq kmer={row.kmer} /></h2>
          <button className="sp-coord" onClick={copy} title="Copy coordinate">
            <span className="mono">{coord}</span> <span className="muted">({strand})</span>
            <span className="copy">{copied ? "copied" : "copy"}</span>
          </button>
          <div className="sp-badges">
            <span className="badge">{row.region === "coding" ? "CDS" : row.region}</span>
            <span className={row.low ? "badge warn" : "badge"}>depth {row.reads}</span>
            <Reliability trust={t} />
          </div>
        </header>

        <section className="sp-block">
          <div className="sp-label"><span>Rank</span><strong className="sp-big">top {topPct(row.rank)}%</strong></div>
          <Ruler marks={[
            { key: "e", kind: "ens", label: "Ensemble", rank: row.rank },
            { key: "o", kind: "own", label: "Read-level", rank: row.own },
          ]} />
          <dl className="sp-grid">
            <dt><span className="sw ens" />Ensemble (reported)</dt><dd className={hl(row.rank)}>{topPct(row.rank)}%</dd>
            {row.own != null && <><dt><span className="sw own" />Read-level only</dt><dd className={hl(row.own)}>{topPct(row.own)}%</dd></>}
            <dt>Rank by motif alone <InfoTip label="About rank by motif alone" text="Rank of this site within the gene if sites were ordered only by how often their DRACH motif is modified in the training labels. A large gap from the model's rank means the reads, not the sequence, drive the call." /></dt>
            <dd>{row.lettersRank} of {rows.length}</dd>
          </dl>
        </section>

        <section className="sp-block">
          <div className="sp-label"><span>Replicates</span><span className="muted small">rank in gene · depth</span></div>
          <table className="sp-reps">
            <tbody>
              {row.batches.map((b) => (
                <tr key={b.name}>
                  <td>{b.name.replace("replicate", "rep ")}</td>
                  <td className={`num${b.position <= 3 ? " hl" : ""}`}>{b.rank == null ? "—" : b.position}</td>
                  <td className="bar-td">{b.rank != null && <span className="mini" style={{ width: `${Math.min(100, (b.reads / Math.max(...row.batches.map((x) => x.reads), 1)) * 100)}%` }} />}</td>
                  <td className="num muted">{b.rank == null ? "not covered" : b.reads}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="sp-block">
          <div className="sp-label"><span>Across cell lines</span></div>
          <div className="sp-lines" role="table" aria-label="Rank of this site in each cell line">
            <div className="sl-row sl-head" role="row">
              <span role="columnheader">Line</span><span role="columnheader" /><span role="columnheader" className="num">Top %</span><span role="columnheader" className="num">Depth</span>
            </div>
            {lines.map((l) => (
              <div key={l.line} role="row" className={`sl-row${l.self ? " self" : ""}`}>
                <span className="ln" role="cell">{lineName(l.line)}</span>
                <span role="cell">{l.rank == null ? <span className="muted small">not covered</span> : <Ruler compact marks={[{ key: "r", kind: l.self ? "ens" : "other", label: lineName(l.line), rank: l.rank }]} />}</span>
                <span role="cell" className={`num ${l.rank == null ? "" : hl(l.rank)}`}>{l.rank == null ? "—" : topPct(l.rank)}</span>
                <span role="cell" className={`num${l.reads && l.reads <= LOW_DEPTH ? " warn" : ""}`}>{l.rank == null ? "—" : l.reads}</span>
              </div>
            ))}
          </div>
          <p className="sp-foot">Consistency only; between-line rank differences are within replicate variation.</p>
        </section>

        <details className="sp-block sp-tx">
          <summary><span className="sp-label"><span>Transcripts</span><span className="muted small">{row.transcripts.length}</span></span></summary>
          <ul>
            {row.transcripts.map((tx) => (
              <li key={tx}><a className="mono small" href={`https://www.ensembl.org/Homo_sapiens/Transcript/Summary?t=${tx}`} target="_blank" rel="noreferrer">{tx}</a></li>
            ))}
          </ul>
        </details>

        <p className="sp-foot">
          Reliability: held-out pairwise accuracy for {t.motif.replace(/T/g, "U")} at this depth
          {t.value == null ? " (not estimable)." : ` = ${t.value.toFixed(2)}.`}
        </p>
      </aside>
    </>
  );
}
