import { useEffect, useState } from "react";
import TranscriptBrowser from "./TranscriptBrowser";

const VIEWS = ["Row", "Data"];

// import.meta.env.BASE_URL is the GitHub Pages sub-path (vite.config.js `base`).
const dataUrl = (name) => `${import.meta.env.BASE_URL}data/${name}`;

export default function App() {
  const [view, setView] = useState(VIEWS[0]);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);


  useEffect(() => {
    fetch(dataUrl("summary.json"))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setSummary)
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="page">
      <header>
        <h1>m6A Explorer</h1>
        <p className="lede">
          Predicted m6A sites across seven SG-NEx cell lines, from nanopore direct RNA sequencing.
        </p>
        <nav aria-label="Explore cell lines">
          {VIEWS.map((v) => (
            <button key={v} type="button" aria-pressed={v === view} className={v === view ? "tab active" : "tab"} onClick={() => setView(v)}>
              {v}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {error && <p className="note">Could not load data: {error}</p>}
        {view === "Row" && <TranscriptBrowser />}
        {view === "Data" && <>
          <h2>Compare cell lines</h2>
          <p className="note">View read coverage and score profiles across all seven cell lines.</p>
          <Overview summary={summary} />
        </>}
        <details className="limitations"><summary>What these results can tell you</summary><Limitations /></details>
      </main>
      <footer>
        DSA4262 team Foursight. Model and analysis:{" "}
        <a href="https://github.com/AsJayTee/DSA4262-Foursight">GitHub repository</a>.
      </footer>
    </div>
  );
}

function Overview({ summary, selectedLine }) {
  if (!summary) return <p className="note">Loading...</p>;
  const rows = summary.reads_per_line.filter((row) => !selectedLine || row.line === selectedLine);
  const series = { ...summary.stop_codon_profile,
    lines: Object.fromEntries(Object.entries(summary.stop_codon_profile.lines).filter(([line]) => !selectedLine || line === selectedLine)) };
  return (
    <>
      <section className="card">
        <h2>Reads per site</h2>
        <p className="note">Each read is one RNA molecule. Sites with few reads are less reliable.</p>
        <table>
          <thead>
            <tr>
              <th>Cell line</th>
              <th className="num">Sites</th>
              <th className="num">Median reads</th>
              <th className="num">1-2 reads</th>
              <th className="num">20+ reads</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.line}>
                <td>{r.line}</td>
                <td className="num">{r.sites.toLocaleString()}</td>
                <td className="num">{r.median_reads}</td>
                <td className="num">{r.pct_1_2.toFixed(1)}%</td>
                <td className="num">{r.pct_20_plus.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="card">
        <h2>Around the stop codon</h2>
        <p className="note">Mean model score by position relative to the stop codon (sites with 10+ reads).{selectedLine ? ` Cell line: ${selectedLine}.` : " All seven cell lines."}</p>
        <LineChart series={series} />
      </section>
    </>
  );
}

// Minimal SVG line chart: no chart library yet, so the template has two dependencies.
function LineChart({ series }) {
  const W = 640, H = 260, P = 36;
  const xs = series.x, lines = series.lines;
  const all = Object.values(lines).flat();
  const [lo, hi] = [Math.min(...all), Math.max(...all)];
  const sx = (x) => P + ((x - xs[0]) / (xs[xs.length - 1] - xs[0])) * (W - 2 * P);
  const sy = (y) => H - P - ((y - lo) / (hi - lo)) * (H - 2 * P);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Mean score around the stop codon, per cell line">
      <line x1={sx(0)} x2={sx(0)} y1={P} y2={H - P} className="ref" />
      {Object.entries(lines).map(([name, ys]) => (
        <polyline key={name} className="series" points={ys.map((y, i) => `${sx(xs[i])},${sy(y)}`).join(" ")}>
          <title>{name}</title>
        </polyline>
      ))}
      <text x={W / 2} y={H - 6} textAnchor="middle" className="axis">position relative to stop codon (nt)</text>
    </svg>
  );
}

function Limitations() {
  return (
    <section className="card">
      <ul>
        <li>Scores order sites by evidence of m6A; they are not calibrated probabilities.</li>
        <li>Most sites have only 1-3 reads; those scores are much less reliable.</li>
        <li>The model detects whether a site is modified, not what fraction of molecules is.</li>
        <li>Differences between cell lines at single sites are mostly not visible in the reads.</li>
      </ul>
    </section>
  );
}
