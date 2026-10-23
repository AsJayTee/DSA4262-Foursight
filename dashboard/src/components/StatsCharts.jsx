import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Popup, Tip } from "./Tip.jsx";

// SVG charts for the Model stats page (public/data/model_stats.json).

const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// Charts draw at the width they are given, in pixels, so every figure spans its column and
// text stays at its CSS size whatever the screen. Height grows with width, within bounds.
function useWidth(fallback = 640) {
  const ref = useRef(null);
  const [w, setW] = useState(fallback);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(Math.round(el.getBoundingClientRect().width) || fallback);
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width) || fallback));
    ro.observe(el);
    return () => ro.disconnect();
  }, [fallback]);
  return [ref, w];
}

const fmt = (v) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(v % 1000 ? 1 : 0)}k` : `${v}`);

/* Line chart: series = [{name, values, style: "accent"|"fg"|"mid"|"dash"|"thin"}] over shared x. */
export function Legend({ series, className = "" }) {
  return (
    <div className={`legend ${className}`}>
      {series.filter((s) => s.name).map((s) => <span key={s.name} className={`key ${s.style}`}>{s.name}</span>)}
    </div>
  );
}

// points: mark each measured value, and on hover show every series' value at that x
// (popTitle names the x, valueFormat formats the values). Lines between points are drawn
// for reading only; the points are what was measured.
export function LineChart({ x, series, yLabel, xLabel, yMax, yTicks, xTicks, refY, refX, height = 280, legend = true, xFormat = fmt,
  points = false, popTitle, valueFormat = (v) => v }) {
  const [ref, W] = useWidth();
  const svg = useRef(null);
  const [hover, setHover] = useState(null);           // {i, name, at}: the point under the pointer
  const H = Math.round(clamp((W * height) / 640, height * 0.85, height * 1.3)), L = 52, R = 16, T = 14, B = 42;
  const vals = series.flatMap((s) => s.values.filter((v) => v != null));
  const top = yMax ?? Math.max(...vals) * 1.08;
  const sx = scale(x[0], x[x.length - 1], L, W - R), sy = scale(0, top, H - B, T);
  const ticks = yTicks ?? [0, top / 4, top / 2, (3 * top) / 4].map((v) => +v.toFixed(2));
  const path = (vs) => {
    let d = "", pen = false;
    vs.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${sx(x[i]).toFixed(1)},${sy(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };
  return (
    <figure className="stat-chart" ref={ref}>
      {legend && <Legend series={series} />}
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={`${yLabel} by ${xLabel}`}>
        {ticks.map((v) => (
          <g key={v}>
            <line className="grid" x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} />
            <text className="tick" x={L - 8} y={sy(v) + 4} textAnchor="end">{v}</text>
          </g>
        ))}
        {(xTicks ?? x.filter((_, i) => i % Math.ceil(x.length / 6) === 0)).map((v) => (
          <text key={v} className="tick" x={sx(v)} y={H - B + 18} textAnchor="middle">{xFormat(v)}</text>
        ))}
        {refY != null && <line className="ref" x1={L} x2={W - R} y1={sy(refY)} y2={sy(refY)} />}
        {refX != null && <line className="ref" x1={sx(refX)} x2={sx(refX)} y1={T} y2={H - B} />}
        <text className="axis-label" x={(L + W - R) / 2} y={H - 4} textAnchor="middle">{xLabel}</text>
        <text className="axis-label" transform={`translate(13,${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
        {series.map((s, i) => <path key={i} className={`series-line ${s.style}`} d={path(s.values)} />)}
        {points && hover && <line className="hover-guide" x1={sx(x[hover.i])} x2={sx(x[hover.i])} y1={T} y2={H - B} />}
        {points && series.filter((s) => s.name).map((s) => s.values.map((v, i) => {
          if (v == null) return null;
          const on = hover?.i === i && hover.name === s.name;
          const enter = () => {
            const r = svg.current.getBoundingClientRect();
            setHover({ i, name: s.name, at: { x: r.left + sx(x[i]) + 4, y: r.top + sy(v) - 4 } });
          };
          // A visible dot, and a larger invisible circle around it that takes the pointer.
          return (
            <g key={`${s.name}${i}`} className={`series-pt-g${on ? " on" : ""}`}
              onPointerEnter={enter} onPointerLeave={() => setHover(null)}>
              <circle className={`series-pt ${s.style}`} cx={sx(x[i])} cy={sy(v)} r={on ? 7 : 5} />
              <circle className="series-hit" cx={sx(x[i])} cy={sy(v)} r="11" />
            </g>
          );
        }))}
      </svg>
      {points && hover && (
        <Popup at={hover.at}>
          <strong>{popTitle ? popTitle(x[hover.i]) : xFormat(x[hover.i])}</strong>
          {series.filter((s) => s.name && s.values[hover.i] != null).map((s) => (
            <span key={s.name} className={s.name === hover.name ? "pop-row on" : "pop-row"}>
              <i className={`sw ${s.style}`} />{s.name}<b>{valueFormat(s.values[hover.i])}</b>
            </span>
          ))}
        </Popup>
      )}
    </figure>
  );
}

/* Paired horizontal bars: one row per method, one bar per cell line. */
export function MetricBars({ rows, metric, lines, max = 1 }) {
  return (
    <table className="metric-bars">
      <thead><tr><th>Method</th>{lines.map((l) => <th key={l}>{l}</th>)}</tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.method} className={r.method === "This model" ? "ours" : ""}>
            <td>{r.method}{r.note && <sup title={r.note}>†</sup>}</td>
            {lines.map((l) => (
              <td key={l}>
                <span className="bar-cell">
                  <span className="bar" style={{ width: `${(r[metric][l] / max) * 100}%` }} />
                  <span className="num">{r[metric][l].toFixed(3)}</span>
                </span>
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* Motif x depth heatmap of pairwise accuracy. */
export function ReliabilityGrid({ data, depths, cutoffs }) {
  const motifs = Object.keys(data).sort((a, b) => (data[b].full ?? 0) - (data[a].full ?? 0));
  const label = { full: "≥ 20", "10 reads": "10–19", "3 reads": "3–9", "1 reads": "1–2" };
  const tone = (v) => (v == null ? "nd" : v >= cutoffs[0] ? "high" : v >= cutoffs[1] ? "mid" : "low");
  return (
    <div className="grid-wrap">
      <table className="reliability">
        <thead>
          <tr><th>Motif</th>{depths.map((d) => <th key={d}>{label[d]}</th>)}</tr>
        </thead>
        <tbody>
          {motifs.map((m) => (
            <tr key={m}>
              <td className="mono">{m.replace(/T/g, "U")}</td>
              {depths.map((d) => {
                const v = data[m][d];
                return <td key={d} className={`cell ${tone(v)}`} style={{ "--v": v == null ? 0 : (v - 0.5) / 0.5 }}>
                  {v == null ? "n.d." : v.toFixed(2)}
                </td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* A two-option toggle. */
export function Toggle({ options, value, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([v, text, tip]) => (tip
        ? <Tip key={v} as="button" text={tip} aria-pressed={value === v} onClick={() => onChange(v)}>{text}</Tip>
        : <button key={v} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>{text}</button>))}
    </div>
  );
}

export function useToggle(first) {
  return useState(first);
}

/* Cell-line selector shared by the charts. */
// A pill dropdown with a styled menu. With `search`, the menu opens with a box that filters
// the options as you type (`normalise` maps both the query and option text before matching).
export function LineSelect({ value, onChange, label = "Cell line", search = false, placeholder = "Search",
  normalise = (t) => t.toLowerCase(),
  options = [["both", "Both cell lines (mean)"], ["HCT116", "HCT116"], ["HEK293T", "HEK293T"]] }) {
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [query, setQuery] = useState("");
  const ref = useRef(null);
  const input = useRef(null);
  const current = options.find(([v]) => v === value) || options[0];
  const shown = search && query ? options.filter(([, t]) => normalise(String(t)).includes(normalise(query))) : options;
  useEffect(() => {
    if (!open) { setQuery(""); return; }
    setCursor(Math.max(0, options.findIndex(([v]) => v === value)));
    input.current?.focus();
    const away = (e) => !ref.current?.contains(e.target) && setOpen(false);
    document.addEventListener("pointerdown", away);
    return () => document.removeEventListener("pointerdown", away);
  }, [open]);  // eslint-disable-line react-hooks/exhaustive-deps
  const choose = (v) => { onChange(v); setOpen(false); };
  const onKey = (e) => {
    if (!open && ["ArrowDown", "Enter", " "].includes(e.key)) { e.preventDefault(); setOpen(true); return; }
    if (!open) return;
    if (e.key === "ArrowDown") { e.preventDefault(); setCursor((c) => Math.min(c + 1, shown.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
    else if (e.key === "Enter" || (e.key === " " && !search)) { e.preventDefault(); if (shown[cursor]) choose(shown[cursor][0]); }
    else if (e.key === "Escape") setOpen(false);
  };
  return (
    <div className={open ? "dropdown open" : "dropdown"} ref={ref}>
      <button type="button" className="dd-button" aria-haspopup="listbox" aria-expanded={open} aria-label={`${label}: ${current[1]}`}
        onClick={() => setOpen(!open)} onKeyDown={onKey}>
        <span className="dd-label">{label}</span>
        <span className="dd-value">{current[1]}</span>
        <svg className="dd-chev" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5 6 7.5 9 4.5" /></svg>
      </button>
      {open && (
        <div className={search ? "dd-menu searchable" : "dd-menu"}>
          {search && (
            <input ref={input} className="dd-search" type="text" value={query} placeholder={placeholder}
              aria-label={`Search ${label.toLowerCase()}`} spellCheck={false} autoComplete="off"
              onChange={(e) => { setQuery(e.target.value); setCursor(0); }} onKeyDown={onKey} />
          )}
          <ul role="listbox" aria-label={label}>
            {shown.map(([v, t, count], i) => (
              <li key={v} role="option" aria-selected={v === value} className={`${i === cursor ? "on" : ""}${v === value ? " sel" : ""}`}
                onPointerEnter={() => setCursor(i)} onClick={() => choose(v)}>
                <span>{t}{count != null && <span className="dd-count">{count}</span>}</span>
                {v === value && <svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 6.5 5 9l4.5-6" /></svg>}
              </li>
            ))}
            {!shown.length && <li className="dd-empty" aria-disabled="true">No match</li>}
          </ul>
        </div>
      )}
    </div>
  );
}

/* PR AUC (y) against ROC AUC (x), one circle per method. Labels sit to the right of their
   point, except LEFT ones (LightGBM sits beside m6Anet in several views). */
const SHORT = { "Logistic regression": "Logistic reg.", "Quantiles + LightGBM": "LightGBM", "m6Anet (pretrained)": "m6Anet",
  "DeepSet (reads only)": "DeepSet", "This model": "This model" };
const LEFT = new Set(["Quantiles + LightGBM"]);
// What each method is, for the details card under the scatter.
const METHOD_INFO = {
  "Logistic regression": "The required simple baseline. Each site's reads are summarised by the mean and standard deviation of the nine signal measurements; with the site's DRACH motif, these feed a logistic regression.",
  "Quantiles + LightGBM": "Each measurement is summarised by five quantiles across the site's reads (5th to 95th percentile), so a minority of modified reads still shows in the tails. Gradient-boosted trees (LightGBM) classify the site.",
  "m6Anet (pretrained)": "The published m6Anet model (Hendra et al., 2022), used with its released HCT116 (RNA002) weights and not retrained. Its training data overlap the HCT116 evaluation set, so its HCT116 values are not held out.",
  "DeepSet (reads only)": "This model's read encoder on its own: one network applied to every read, then pooled across the site's reads. It does not use neighbouring sites.",
  "This model": "Six site-graph networks (three designs, two random starts each). Each scores a site from its own reads, then adjusts the score using candidate sites within 150 nt on the same transcript. The reported score is the mean of the six within-sample ranks.",
};
const f3 = (v) => (v == null ? "–" : v.toFixed(3));
// m6Anet's HCT116 value is not held out; the two-line mean includes it.
const hollowIn = (lines) => lines.includes("HCT116") || lines.includes("both");
export function AucScatter({ rows, lines }) {
  const [ref, W] = useWidth();
  const [hover, setHover] = useState(null);           // {r, at}
  const [open, setOpen] = useState(null);             // method whose details card is shown
  const show = (r, e) => {
    const b = e.currentTarget.querySelector("circle").getBoundingClientRect();
    setHover({ r, at: { x: b.right, y: b.top + b.height / 2 } });
  };
  const toggle = (m) => setOpen((o) => (o === m ? null : m));
  const H = Math.round(clamp(W * 0.5, 320, 440)), L = 52, R = 16, T = 16, B = 46;
  const pts = rows.filter((r) => r.method !== "Random");
  const xs = pts.flatMap((r) => lines.map((l) => r.roc[l])), ys = pts.flatMap((r) => lines.map((l) => r.pr[l]));
  const x0 = Math.floor(Math.min(...xs) * 50) / 50 - 0.01, x1 = Math.ceil(Math.max(...xs) * 50) / 50 + 0.01;
  const y0 = Math.floor(Math.min(...ys) * 20) / 20 - 0.02, y1 = Math.ceil(Math.max(...ys) * 20) / 20 + 0.02;
  const sx = scale(x0, x1, L, W - R), sy = scale(y0, y1, H - B, T);
  const step = (lo, hi, n) => { const s = (hi - lo) / n, p = 10 ** Math.floor(Math.log10(s)); const st = [1, 2, 5].map((m) => m * p).find((v) => v >= s) || s; const out = []; for (let v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(3)); return out; };
  const anyHollow = pts.some((r) => r.note) && hollowIn(lines);
  return (
    <figure className="stat-chart scatter" ref={ref}>
      <div className="legend">
        <span className="mk ours">This model</span>
        <span className="mk">Other methods</span>
        {anyHollow && <span className="mk hollow">† not held out</span>}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="PR AUC against ROC AUC for each method">
        {step(y0, y1, 5).map((v) => <g key={`y${v}`}><line className="grid" x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} /><text className="tick" x={L - 8} y={sy(v) + 4} textAnchor="end">{v.toFixed(2)}</text></g>)}
        {step(x0, x1, 5).map((v) => <g key={`x${v}`}><line className="grid" x1={sx(v)} x2={sx(v)} y1={T} y2={H - B} /><text className="tick" x={sx(v)} y={H - B + 18} textAnchor="middle">{v.toFixed(2)}</text></g>)}
        <text className="axis-label" x={(L + W - R) / 2} y={H - 6} textAnchor="middle">ROC AUC</text>
        <text className="axis-label" transform={`translate(14,${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle">PR AUC</text>
        {pts.map((r) => {
          const ours = r.method === "This model";
          const l = lines[0], x = sx(r.roc[l]), y = sy(r.pr[l]);
          // A label that would run past the plot edge goes on the left of its point instead.
          const text = `${SHORT[r.method]}${r.note && hollowIn(lines) ? " †" : ""}`;
          const left = LEFT.has(r.method) || x + 11 + text.length * 7.5 > W - R;
          const hollow = r.note && hollowIn(lines);
          return (
            <g key={r.method} className={`pt${ours ? " ours" : ""}${open === r.method ? " sel" : ""}`}
              tabIndex="0" role="button" aria-expanded={open === r.method}
              aria-label={`${r.method}: PR AUC ${f3(r.pr[l])}, ROC AUC ${f3(r.roc[l])}. Show details.`}
              onPointerEnter={(e) => show(r, e)} onPointerLeave={() => setHover(null)}
              onFocus={(e) => show(r, e)} onBlur={() => setHover(null)}
              onClick={() => toggle(r.method)}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(r.method); } }}>
              <circle cx={x} cy={y} r="6.5" className={hollow ? "hollow" : ""} />
              <text className="pt-label" x={left ? x - 11 : x + 11} y={y + 4} textAnchor={left ? "end" : "start"}>
                {SHORT[r.method]}{hollow ? " †" : ""}
              </text>
            </g>
          );
        })}
      </svg>
      {hover && (
        <Popup at={hover.at}>
          <strong>{hover.r.method}</strong>
          <span className="pop-row">PR AUC<b>{f3(hover.r.pr[lines[0]])}</b></span>
          <span className="pop-row">ROC AUC<b>{f3(hover.r.roc[lines[0]])}</b></span>
          {hover.r.note && hollowIn(lines) && <span className="pop-note">† Not held out on HCT116</span>}
          <span className="pop-hint">{open === hover.r.method ? "Click to close details" : "Click for details"}</span>
        </Popup>
      )}
      {open && (
        <MethodPanel rows={pts} method={open} line={lines[0]} onMove={setOpen} onClose={() => setOpen(null)} />
      )}
    </figure>
  );
}

/* A method's details in a panel on the right, the same panel as a site's in Explore:
   previous / next through the methods, Escape or the backdrop to close. */
const LINE_NAME = { HCT116: "HCT116", HEK293T: "HEK293T", both: "Both cell lines (mean)" };
function MethodPanel({ rows, method, line, onMove, onClose }) {
  const ref = useRef(null);
  const i = rows.findIndex((r) => r.method === method);
  const r = rows[i], prev = rows[i - 1], next = rows[i + 1];
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && prev) onMove(prev.method);
      if (e.key === "ArrowRight" && next) onMove(next.method);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose, onMove, prev, next]);
  useEffect(() => { ref.current?.focus(); }, [method]);
  return (
    <>
      <div className="scrim" onClick={onClose} />
      <aside className="site-panel method-panel" role="dialog" aria-modal="true" aria-labelledby="method-title" tabIndex="-1" ref={ref}>
        <div className="sp-top">
          <span className="muted small">Discrimination · {LINE_NAME[line] || line}</span>
          <div className="sp-nav">
            <button onClick={() => prev && onMove(prev.method)} disabled={!prev} aria-label="Previous method">‹</button>
            <span className="small">{i + 1} / {rows.length}</span>
            <button onClick={() => next && onMove(next.method)} disabled={!next} aria-label="Next method">›</button>
            <button onClick={onClose} aria-label="Close" className="sp-close">×</button>
          </div>
        </div>
        <header className="sp-head">
          <h2 id="method-title">{r.method}</h2>
          <div className="sp-badges">
            {r.method === "This model" && <span className="badge ours">Shipped model</span>}
            {r.note && <span className="badge warn">HCT116 not held out</span>}
          </div>
        </header>
        <section className="sp-block">
          <div className="sp-label">Method</div>
          <p className="mp-text">{METHOD_INFO[r.method]}</p>
        </section>
        <section className="sp-block">
          <div className="sp-label">Held-out performance</div>
          <table className="mc-table">
            <thead><tr><th /><th className="num">HCT116{r.note ? " †" : ""}</th><th className="num">HEK293T</th><th className="num">Mean</th></tr></thead>
            <tbody>
              {[["PR AUC", "pr"], ["ROC AUC", "roc"]].map(([t, k]) => (
                <tr key={k}><th scope="row">{t}</th>
                  {["HCT116", "HEK293T", "both"].map((c) => <td key={c} className={c === line ? "num on" : "num"}>{f3(r[k][c])}</td>)}</tr>
              ))}
            </tbody>
          </table>
          <p className="mp-note">
            Five-fold cross-validation grouped by gene. A random ranking has PR AUC 0.045 (HCT116) and 0.072 (HEK293T).
            {r.note && " † m6Anet was trained on HCT116 data, so its HCT116 values are not held out."}
          </p>
        </section>
      </aside>
    </>
  );
}

/* Grouped bars: categories down, one bar per series, values as percentages. */
export function GroupedBars({ categories, series }) {
  return (
    <div className="grouped">
      <div className="legend">{series.map((s) => <span key={s.name} className={`key bar ${s.style}`}>{s.name}</span>)}</div>
      {categories.map((c, i) => (
        <div className="g-row" key={c}>
          <span className="g-cat">{c}</span>
          <span className="g-bars">
            {series.map((s) => (
              <span key={s.name} className="g-bar-line">
                <span className={`g-bar ${s.style}`} style={{ width: `${s.values[i] * 100}%` }} />
                <span className="g-val">{Math.round(s.values[i] * 100)}%</span>
              </span>
            ))}
          </span>
        </div>
      ))}
    </div>
  );
}

/* Reliability grid that fills the width: motifs as columns on wide screens, as rows on narrow. */
export function ReliabilityAuto({ data, depths, cutoffs }) {
  const ref = useRef(null);
  const [wide, setWide] = useState(true);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setWide(e.contentRect.width >= 760));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  const motifs = Object.keys(data).sort((a, b) => (data[b].full ?? 0) - (data[a].full ?? 0));
  const label = { full: "≥ 20 reads", "10 reads": "10–19", "3 reads": "3–9", "1 reads": "1–2" };
  const tone = (v) => (v == null ? "nd" : v >= cutoffs[0] ? "high" : v >= cutoffs[1] ? "mid" : "low");
  const cell = (m, d) => {
    const v = data[m][d];
    return <td key={`${m}${d}`} className={`cell ${tone(v)}`} style={{ "--v": v == null ? 0 : Math.max(0, (v - 0.5) / 0.5) }}>{v == null ? "n.d." : v.toFixed(2)}</td>;
  };
  return (
    <div ref={ref} className="grid-wrap">
      <table className={`reliability ${wide ? "transposed" : ""}`}>
        {wide ? (
          <>
            <thead><tr><th>Depth</th>{motifs.map((m) => <th key={m} className="mono">{m.replace(/T/g, "U")}</th>)}</tr></thead>
            <tbody>{depths.map((d) => <tr key={d}><th scope="row">{label[d]}</th>{motifs.map((m) => cell(m, d))}</tr>)}</tbody>
          </>
        ) : (
          <>
            <thead><tr><th>Motif</th>{depths.map((d) => <th key={d}>{label[d]}</th>)}</tr></thead>
            <tbody>{motifs.map((m) => <tr key={m}><th scope="row" className="mono">{m.replace(/T/g, "U")}</th>{depths.map((d) => cell(m, d))}</tr>)}</tbody>
          </>
        )}
      </table>
    </div>
  );
}
