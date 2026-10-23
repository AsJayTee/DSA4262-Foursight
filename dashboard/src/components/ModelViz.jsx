import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { topPercent } from "../lib/data.js";
import { rna } from "./StoryCharts.jsx";
import { useNarrow } from "../lib/media.js";

// The three pictures at the heart of How it works, all drawn from real model data in
// public/data/model_story.json (analysis/sgnex/export_model_story.py).

const clamp01 = (v) => Math.max(0, Math.min(1, v));
// A colour between the "low" node grey and the accent, by strength t in [0, 1].
const heat = (t) => `color-mix(in srgb, var(--accent) ${Math.round(clamp01(t) * 100)}%, var(--node-low))`;
// Standardised inputs: positive towards the accent, negative towards the foreground.
const diverging = (v) => {
  const a = Math.round(clamp01(Math.abs(v) / 2.5) * 100);
  return v >= 0 ? `color-mix(in srgb, var(--accent) ${a}%, transparent)` : `color-mix(in srgb, var(--fg) ${a}%, transparent)`;
};
// How strongly a rank reads as "high priority": top 1% -> 1, top 50% or lower -> 0.
const strength = (rank) => clamp01(1 - Math.log10(Math.max((1 - rank) * 100, 1)) / Math.log10(50));

function useReducedMotion() {
  return useMemo(() => window.matchMedia("(prefers-reduced-motion: reduce)").matches, []);
}

/* 1. The RNA as text; each candidate site's reads as a table. */
export function RnaText({ text, sites }) {
  const narrow = useNarrow();                       // phones: the letters only, no tables or hover details
  const [active, setActive] = useState(null);       // index of the hovered site, 0-3
  const box = useRef(null);
  const panelRef = useRef(null);
  const tableRefs = useRef([]);
  const letterRefs = useRef([]);                    // the highlighted A of each site in the text
  const [panel, setPanel] = useState(null);         // where the details panel goes
  const start = text.start;
  const shownPos = text.tables.map((t) => t.position);
  // End the stretch before the next candidate site, so every marked site has a table.
  const nextSite = sites.map((s) => s.position).filter((p) => p > shownPos[shownPos.length - 1]).sort((a, b) => a - b)[0];
  const end = Math.min(text.letters.length, nextSite !== undefined ? nextSite - 3 - start - 2 : text.letters.length);
  const letters = text.letters.slice(0, end).split("");
  const number = new Map(shownPos.map((p, i) => [p, i]));
  const inWindow = (p) => shownPos.some((c) => Math.abs(p - c) <= 3);

  // Beside the hovered table (right if the window has room, else left), then nudged up or
  // down so the whole panel stays on screen below the nav bar. It must never cover the
  // letter being hovered: that would end the hover the moment the panel appeared.
  const place = () => {
    const t = tableRefs.current[active], b = box.current, p = panelRef.current;
    if (!t || !b) return;
    const bb = b.getBoundingClientRect();
    const width = Math.min(320, window.innerWidth - 32);
    const nav = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--nav-h")) || 57;
    const h = p ? p.offsetHeight : 0;
    const lo = nav + 12 - bb.top, hi = window.innerHeight - 12 - bb.top - h;
    const top = Math.max(lo, Math.min(t.offsetTop, hi));
    const a = letterRefs.current[active]?.getBoundingClientRect();
    const onScreen = (l) => bb.left + l >= 12 && bb.left + l + width <= window.innerWidth - 12;
    const covers = (l) => a && bb.left + l < a.right + 8 && bb.left + l + width > a.left - 8
      && bb.top + top < a.bottom + 8 && bb.top + top + h > a.top - 8;
    // In order of preference: beside the table (right, left); if neither is free, beside
    // the hovered letter (right, left), so it stays close to what the reader is pointing
    // at; only then against either screen edge.
    const options = [t.offsetLeft + t.offsetWidth + 20, t.offsetLeft - width - 20];
    if (a) options.push(a.right + 28 - bb.left, a.left - 16 - width - bb.left);   // +28 clears the number badge
    options.push(window.innerWidth - 12 - width - bb.left, 12 - bb.left);
    const left = options.find((l) => onScreen(l) && !covers(l)) ?? options.find(onScreen) ?? options[3];
    setPanel((old) => (old && old.left === left && old.top === top && old.width === width ? old : { left, top, width }));
  };
  useLayoutEffect(() => {
    if (active === null) { setPanel(null); return; }
    place();
    window.addEventListener("scroll", place, { passive: true });
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("scroll", place);
      window.removeEventListener("resize", place);
    };
  }, [active]);  // eslint-disable-line react-hooks/exhaustive-deps
  // Once the panel has rendered, its height is known: place it again.
  useLayoutEffect(() => { if (panel && active !== null) place(); }, [panel === null]);  // eslint-disable-line react-hooks/exhaustive-deps

  const table = active === null ? null : text.tables[active];
  const toggle = (i) => setActive((a) => (a === i ? null : i));

  return (
    <figure className="viz">
      <p className="rna-text" aria-label="A stretch of the MYC RNA with four candidate sites marked">
        <span className="rna-ellipsis">… </span>
        {letters.map((ch, i) => {
          const p = start + i;
          const idx = number.get(p);
          const letter = ch === "T" ? "U" : ch;
          if (idx === undefined) return <span key={i} className={inWindow(p) ? "nt motif" : "nt"}>{letter}</span>;
          return (
            <span key={i} className="nt centre" onMouseEnter={narrow ? undefined : () => setActive(idx)} onMouseLeave={() => setActive(null)}>
              <span ref={(el) => (letterRefs.current[idx] = el)} className={active === idx ? "nt-a on" : "nt-a"}>{letter}</span>
              <sup className="site-tag">{idx + 1}</sup>
            </span>
          );
        })}
        <span className="rna-ellipsis"> …</span>
      </p>
      {!narrow && <div className={active === null ? "tables" : "tables focus"} ref={box}>
        {text.tables.map((t, i) => (
          <div key={t.position} ref={(el) => (tableRefs.current[i] = el)}
            className={active === i ? "read-table on" : "read-table"}
            onMouseEnter={() => setActive(i)} onMouseLeave={() => setActive(null)} onClick={() => toggle(i)}>
            <ReadTable index={i + 1} table={t} />
          </div>
        ))}
        {table && (
          <aside ref={panelRef} className="table-panel" aria-live="polite"
            style={panel ? { left: panel.left, top: panel.top, width: panel.width } : { visibility: "hidden" }}>
            <h3>Site {active + 1}</h3>
            <p className="panel-kmer"><code>{rna(table.kmer)}</code>, position {table.position} of {text.length.toLocaleString()}</p>
            <dl>
              <dt>Rows</dt>
              <dd>{table.depth} rows, one per sequenced molecule that contains this site.</dd>
              <dt>Columns</dt>
              <dd><RowFormula /></dd>
            </dl>
            <div className="spectrum" aria-label="Colour scale: grey below the usual level, orange above">
              <span className="spectrum-bar" />
              <span className="spectrum-labels"><span>below usual</span><span>usual</span><span>above usual</span></span>
            </div>
          </aside>
        )}
      </div>}
      <figcaption>
        Part of the MYC RNA ({text.length.toLocaleString()} letters in all), with four candidate sites marked.
        {!narrow && " Below, each site's table. Hover over a site or a table for what it shows."}
      </figcaption>
    </figure>
  );
}

// One row as a vector, in native MathML (no maths library): time t, spread s and level l
// at the positions before (-1), at (0) and after (+1) the A.
function RowFormula() {
  const term = (v, p) => <msub key={v + p}><mi>{v}</mi><mrow><mn>{p}</mn></mrow></msub>;
  const parts = [];
  ["−1", "0", "+1"].forEach((p) => ["t", "s", "ℓ"].forEach((v) => parts.push(term(v, p))));
  const key = [
    [<mi key="t">t</mi>, "time in the sensor"],
    [<mi key="s">s</mi>, "spread of the signal"],
    [<mi key="l">ℓ</mi>, "level of the signal"],
    [<mi key="i">i</mi>, <>subscript: position from the A<span className="key-sub">−1 before, 0 at, +1 after</span></>],
  ];
  return (
    <>
      <math display="block" className="row-formula">
        <mrow>
          {/* Literal 𝐱: browsers' default maths fonts often ignore mathvariant. */}
          <mi>𝐱</mi><mo>=</mo><mo>(</mo>
          {parts.flatMap((t, i) => (i ? [<mo key={`c${i}`} separator="true">,</mo>, t] : [t]))}
          <mo>)</mo>
        </mrow>
      </math>
      <ul className="formula-key">
        {key.map(([sym, text], i) => (
          <li key={i}><math><mrow>{sym}</mrow></math><span>{text}</span></li>
        ))}
      </ul>
    </>
  );
}

function ReadTable({ index, table }) {
  const cw = 13, rh = 7, w = cw * 9 + 2;
  const gx = (c) => c * cw + (c >= 3) + (c >= 6);
  return (
    <>
      <div className="rt-head">
        <span className="site-tag">{index}</span> <code>{rna(table.kmer)}</code>
      </div>
      <svg width={w} height={table.rows.length * rh} viewBox={`0 0 ${w} ${table.rows.length * rh}`} role="img"
        aria-label={`Site ${index}: ${table.depth} molecules by 9 measurements`}>
        {table.rows.map((row, r) => row.map((v, c) => (
          <rect key={`${r}-${c}`} x={gx(c)} y={r * rh} width={cw - 1} height={rh - 1} fill={diverging(v)} />
        )))}
      </svg>
      <div className="rt-foot">{table.depth} molecules</div>
    </>
  );
}

/* 2. The read network: every molecule through the same layers, then averaged. */
export function ReadNetwork({ network, table, site }) {
  const n = network.inputs.length;
  const reduced = useReducedMotion();
  const [k, setK] = useState(1);
  const [playing, setPlaying] = useState(false);
  const host = useRef(null);
  const started = useRef(false);

  // Start once, the first time the diagram scrolls into view.
  useEffect(() => {
    if (reduced) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !started.current) {
        started.current = true;
        setPlaying(true);
      }
    }, { threshold: 0.4 });
    io.observe(host.current);
    return () => io.disconnect();
  }, [reduced]);
  useEffect(() => {
    if (!playing) return;
    if (k >= n) { setPlaying(false); return; }
    const t = setTimeout(() => setK((x) => x + 1), k < 6 ? 650 : 220);
    return () => clearTimeout(t);
  }, [playing, k, n]);

  const max = (rows) => Math.max(1e-6, ...rows.flat());
  const m1 = useMemo(() => max(network.h1), [network]);
  const m2 = useMemo(() => max(network.h2), [network]);
  const m3 = useMemo(() => max(network.site), [network]);
  const row = k - 1;
  const pooled = useMemo(() => network.h2[0].map((_, u) =>
    network.h2.slice(0, k).reduce((s, r) => s + r[u], 0) / k), [network, k]);
  const layers = [
    { key: "in", label: "Inputs", sub: "9 measurements", values: network.inputs[row], x: 190, color: (v) => diverging(v) },
    { key: "h1", label: "Layer 1", sub: `${network.units.h1} of ${network.units.of} units`, values: network.h1[row], x: 320, color: (v) => heat(v / m1) },
    { key: "h2", label: "Layer 2", sub: `${network.units.h2} of ${network.units.of} units`, values: network.h2[row], x: 450, color: (v) => heat(v / m2) },
    { key: "pool", label: "Average", sub: "over molecules so far", values: pooled, x: 580, color: (v) => heat(v / m2) },
    { key: "site", label: "Site network", sub: "8 of 64 units", values: network.site[row], x: 710, color: (v) => heat(v / m3) },
  ];
  const H = 400, top = 60, bottom = 340;
  const ys = (count) => Array.from({ length: count }, (_, i) => top + ((i + 0.5) * (bottom - top)) / count);
  const rank = network.own_rank[row];
  const narrow = useNarrow();   // phones: the network alone fits the width, the score chart goes below
  const outX = 840, outY = (top + bottom) / 2;
  const cw = 8, rh = Math.min(5, (bottom - top) / n), tableX = 24, tableY = top + ((bottom - top) - n * rh) / 2;
  // The score trace: x = molecules fed in, y = the site's rank percentile. Beside the
  // network on wide screens, its own drawing below it on phones.
  const [cx0, cx1, cy0, cy1, titleY] = narrow ? [70, 570, 76, 246, 26] : [960, 1220, top, bottom, top - 28];
  const px = (i) => cx0 + (i / (n - 1)) * (cx1 - cx0), py = (r) => cy1 - r * (cy1 - cy0);
  const trace = (m) => network.own_rank.slice(0, m).map((r, i) => `${i ? "L" : "M"}${px(i).toFixed(1)},${py(r).toFixed(1)}`).join("");
  const xTicks = [1, ...[10, 20, 50, 100, 200, 500].filter((v) => v < n * 0.85), n];
  const SHORT = { in: "Inputs", h1: "Layer 1", h2: "Layer 2", pool: "Average", site: "Site net" };
  const chart = (
    <g>
      <text className="layer-label" x={(cx0 + cx1) / 2} y={titleY} textAnchor="middle">Score as molecules are added</text>
      <text className="layer-sub" x={(cx0 + cx1) / 2} y={titleY + (narrow ? 24 : 16)} textAnchor="middle">rank percentile in the sample</text>
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line className="grid" x1={cx0} x2={cx1} y1={py(v)} y2={py(v)} />
          <text className="tick" x={cx0 - 8} y={py(v) + 4} textAnchor="end">{v.toFixed(2)}</text>
        </g>
      ))}
      {xTicks.map((v) => <text key={v} className="tick" x={px(v - 1)} y={cy1 + (narrow ? 24 : 18)} textAnchor="middle">{v}</text>)}
      <text className="layer-sub" x={(cx0 + cx1) / 2} y={cy1 + (narrow ? 50 : 40)} textAnchor="middle">molecules fed in</text>
      <path className="spark ghost" d={trace(n)} />
      <path className="spark" d={trace(k)} />
      <circle className="spark-dot" cx={px(row)} cy={py(rank)} r={narrow ? 7 : 4.5} />
    </g>
  );

  return (
    <figure className="viz" ref={host}>
      <div className="viz-controls">
        <button className="btn small" onClick={() => { if (k >= n) setK(1); setPlaying(!playing); }}>
          {playing ? "Pause" : k >= n ? "Replay" : "Play"}
        </button>
        <input type="range" min="1" max={n} value={k} aria-label="Molecules fed in"
          onChange={(e) => { setPlaying(false); setK(+e.target.value); }} />
        <span className="viz-readout">Molecule <strong>{k}</strong> of {n}</span>
      </div>
      <div className={narrow ? "viz-fit" : "viz-scroll"}>
        <svg viewBox={`0 0 ${narrow ? 912 : 1250} ${H + 10}`} className={narrow ? "network narrow" : "network"} role="img"
          aria-label={`A neural network reading molecule ${k} of ${n}; the site's score is ${topPercent(rank)}`}>
          {/* the site's table, current molecule outlined */}
          <text className="layer-label" x={tableX} y={top - 28}>{narrow ? "Table" : "Site's table"}</text>
          <text className="layer-sub" x={tableX} y={top - 12}>{n} molecules</text>
          {table.map((r, i) => r.map((v, c) => (
            <rect key={`${i}-${c}`} x={tableX + c * cw} y={tableY + i * rh} width={cw - 1} height={Math.max(rh - 0.6, 1)}
              fill={diverging(v)} opacity={i < k ? 1 : 0.25} />
          )))}
          <rect className="row-focus" x={tableX - 2} y={tableY + row * rh - 1.5} width={9 * cw + 3} height={rh + 2} />
          <line className="edge on" x1={tableX + 9 * cw + 4} y1={tableY + row * rh + rh / 2} x2={layers[0].x - 12} y2={outY} />

          {/* edges between consecutive layers, brighter where the source is active */}
          {layers.slice(0, -1).map((a, li) => {
            const b = layers[li + 1], ya = ys(a.values.length), yb = ys(b.values.length);
            const oneToOne = b.key === "pool";
            const lim = a.key === "in" ? 2.5 : a.key === "h1" ? m1 : m2;
            return a.values.map((va, i) => (oneToOne ? [i] : yb.map((_, j) => j)).map((j) => (
              <line key={`${a.key}${i}-${j}`} className="edge" x1={a.x} y1={ya[i]} x2={b.x} y2={yb[j]}
                strokeOpacity={0.05 + 0.45 * clamp01(Math.abs(va) / lim)} />
            )));
          })}
          {ys(layers[4].values.length).map((y, i) => (
            <line key={`out${i}`} className="edge" x1={layers[4].x} y1={y} x2={outX} y2={outY}
              strokeOpacity={0.05 + 0.45 * clamp01(layers[4].values[i] / m3)} />
          ))}

          {layers.map((L) => (
            <g key={L.key}>
              <text className="layer-label" x={L.x} y={top - 28} textAnchor="middle">{narrow ? SHORT[L.key] : L.label}</text>
              <text className="layer-sub" x={L.x} y={top - 12} textAnchor="middle">{L.sub}</text>
              {ys(L.values.length).map((y, i) => (
                <circle key={i} className="node" cx={L.x} cy={y} r={L.key === "in" ? 9 : 10} style={{ fill: L.color(L.values[i]) }} />
              ))}
            </g>
          ))}

          <text className="layer-label" x={outX} y={top - 28} textAnchor="middle">Score</text>
          <text className="layer-sub" x={outX} y={top - 12} textAnchor="middle">own molecules only</text>
          <circle className="node out" cx={outX} cy={outY} r="26" style={{ fill: heat(strength(rank)) }} />
          <text className="out-text" x={outX} y={outY + (narrow ? 72 : 48)} textAnchor="middle">{topPercent(rank)}</text>
          {!narrow && <line className="edge on" x1={outX + 28} y1={outY} x2={cx0 - 52} y2={outY} />}
          {!narrow && chart}
        </svg>
        {narrow && (
          <svg viewBox="0 0 600 310" className="network score-chart" role="img"
            aria-label={`The site's score after each molecule; now ${topPercent(rank)}`}>
            {chart}
          </svg>
        )}
      </div>
      <figcaption>
        Real activations of one of the six networks at site {site.position} of MYC (<code>{rna(site.kmer)}</code>). Each layer
        shows its 10 most active units of 64. "Average" is the running mean of layer 2 over the molecules fed in so far.
      </figcaption>
    </figure>
  );
}

/* 3. The site graph: sites pass their scores to neighbours within the radius. */
export function SiteGraph({ sites, highlight, radius, kernel }) {
  const reduced = useReducedMotion();
  const [selected, setSelected] = useState(highlight);
  const [preview, setPreview] = useState(null);      // hovered site: its arcs fade in, nothing changes
  const [updated, setUpdated] = useState(false);     // have the sites taken in their neighbours' evidence?
  const [pulse, setPulse] = useState(null);          // {site, progress} while messages travel
  const host = useRef(null);
  const ran = useRef(false);
  const stop = useRef(null);                         // cancels a running animation
  useEffect(() => () => stop.current?.(), []);
  // Phones show a stretch of 15 sites around the highlighted one, drawn large enough to tap.
  const narrow = useNarrow();
  const SPAN = 15;
  const lo = narrow ? Math.max(0, Math.min(highlight - Math.floor(SPAN / 2), sites.length - SPAN)) : 0;
  const hi = narrow ? Math.min(sites.length - 1, lo + SPAN - 1) : sites.length - 1;
  const shown = (i) => i >= lo && i <= hi;
  const W = narrow ? 400 : 960, L = narrow ? 20 : 30, R = narrow ? 20 : 30;
  const x = (i) => L + ((i - lo) * (W - L - R)) / (hi - lo);
  const neighbours = useMemo(() => sites.map((s, i) => sites
    .map((t, j) => ({ j, d: Math.abs(t.position - s.position) }))
    .filter(({ j, d }) => j !== i && d <= radius && shown(j))
    .map(({ j, d }) => ({ j, d, w: Math.exp(-d / kernel) }))), [sites, radius, kernel, lo, hi]);  // eslint-disable-line react-hooks/exhaustive-deps
  // As tall as the tallest arc any site can draw.
  const tallest = Math.max(20, ...neighbours.flatMap((ns, i) => (shown(i) ? ns.map(({ j }) => Math.min(Math.abs(x(j) - x(i)) / 2, 170)) : [])));
  const base = tallest + 24, H = base + 46;

  // Messages travel from a site's neighbours into it; then every site updates.
  const run = (site) => {
    stop.current?.();
    setSelected(site);
    setUpdated(false);
    if (reduced) { setUpdated(true); return; }
    let frame, t0;
    const tick = (now) => {
      t0 ??= now;
      const p = Math.min((now - t0) / 1300, 1);
      setPulse({ site, progress: p });
      if (p < 1) frame = requestAnimationFrame(tick);
      else { setPulse(null); setUpdated(true); }
    };
    frame = requestAnimationFrame(tick);
    stop.current = () => cancelAnimationFrame(frame);
  };
  // Once, a moment after the graph is well into view.
  useEffect(() => {
    let timer;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !ran.current) {
        ran.current = true;
        timer = setTimeout(() => run(highlight), 700);
      }
    }, { threshold: 0.8 });
    io.observe(host.current);
    return () => { io.disconnect(); clearTimeout(timer); };
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const score = (s) => (updated ? s.rank : s.alone);
  const sel = sites[selected];
  const arcPath = (i, j) => {
    const [a, b] = i < j ? [i, j] : [j, i];
    const rx = (x(b) - x(a)) / 2;
    return { d: `M${x(a)},${base} A${rx},${Math.min(rx, 170)} 0 0 1 ${x(b)},${base}`, rx, ry: Math.min(rx, 170), cx: (x(a) + x(b)) / 2 };
  };
  const arcs = (site, cls) => neighbours[site].map(({ j, w }) => (
    <path key={`${cls}${j}`} className={`arc ${cls}`} d={arcPath(site, j).d} strokeWidth={0.8 + 3.4 * w} />
  ));

  return (
    <figure className="viz" ref={host}>
      <div className="viz-controls">
        <span className="viz-readout">
          Site {selected + 1} (<code>{rna(sel.kmer)}</code>, {neighbours[selected].length} neighbours):{" "}
          {topPercent(sel.alone)} on its own
          {updated && <> → <strong>{topPercent(sel.rank)}</strong> with its neighbours</>}
        </span>
      </div>
      <div className={narrow ? "viz-fit" : "viz-scroll"}>
        <svg viewBox={`0 0 ${W} ${H}`} className={narrow ? "graph narrow" : "graph"} role="img"
          aria-label="Candidate sites of MYC as a graph; arcs join the selected site to sites within 150 letters">
          {preview !== null && preview !== selected && arcs(preview, "preview")}
          {arcs(selected, "on")}
          {pulse && neighbours[pulse.site].map(({ j }) => {
            // Along the arc from neighbour j into the receiving site.
            const { rx, ry, cx } = arcPath(pulse.site, j);
            const t = Math.PI * pulse.progress;
            return <circle key={j} className="pulse" r="3.2"
              cx={j < pulse.site ? cx - rx * Math.cos(t) : cx + rx * Math.cos(t)} cy={base - ry * Math.sin(t)} />;
          })}
          <line className="axis" x1={L - 10} x2={W - R + 10} y1={base} y2={base} />
          {sites.map((s, i) => shown(i) && (
            <circle key={s.position} cx={x(i)} cy={base} r={(narrow ? 1.35 : 1) * (i === selected ? 9 : i === preview ? 8 : 6.5)}
              tabIndex="0" role="button"
              className={i === selected ? "node sel" : i === preview ? "node preview" : "node"}
              style={{ fill: heat(strength(score(s))) }}
              aria-label={`Site ${i + 1}, ${topPercent(score(s))}. Select to pass its neighbours' messages.`}
              onMouseEnter={() => setPreview(i)} onMouseLeave={() => setPreview(null)}
              onFocus={() => setPreview(i)} onBlur={() => setPreview(null)}
              onClick={() => run(i)}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); run(i); } }} />
          ))}
          <text className="layer-sub" x={L - 10} y={base + 32}>{narrow ? `site ${lo + 1}` : "start of the RNA"}</text>
          <text className="layer-sub" x={W - R + 10} y={base + 32} textAnchor="end">{narrow ? `site ${hi + 1}` : "end"}</text>
        </svg>
      </div>
      <figcaption>
        {narrow ? `Sites ${lo + 1}–${hi + 1} of MYC's ${sites.length} candidate sites` : `MYC's ${sites.length} candidate sites`} in order along the RNA. Arcs join the selected site to every site within{" "}
        {radius} letters, thicker when closer. Colour: how high the site ranks (orange = top 1%).{" "}
        {narrow ? "Tap a site to pass its messages." : "Hover over a site to preview its connections; click it to pass its messages (click again to replay)."}
      </figcaption>
    </figure>
  );
}
