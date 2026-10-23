import { useMemo } from "react";
import { topPercent } from "../lib/data.js";

// Plain SVG charts for the How it works page. Every value comes from
// public/data/model_story.json (analysis/sgnex/export_model_story.py).

const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
// m6Anet writes sequences in DNA letters; on the page they are RNA (U, not T).
export const rna = (s) => s.replace(/T/g, "U");

/* 3. The ensemble: six networks' ranks for one site. */
export function EnsembleStrip({ networks, rank }) {
  const names = Object.keys(networks);
  const label = (n) => n.replace("h2gcn_twohead_aux", "Two-head H2GCN").replace("res_gate", "Residual + kernel")
    .replace("scalar_drop", "Scalar messages").replace(/_s(\d)/, " · seed $1");
  return (
    <figure className="chart">
      <table className="mini-table">
        <thead><tr><th>Network</th><th className="num">Rank in the sample</th></tr></thead>
        <tbody>
          {names.map((n) => <tr key={n}><td>{label(n)}</td><td className="num">{topPercent(networks[n])}</td></tr>)}
          <tr className="total"><td>Average of the six ranks (what we report)</td><td className="num">{topPercent(rank)}</td></tr>
        </tbody>
      </table>
    </figure>
  );
}

/* 4. Accuracy by read count. */
export function DepthCurve({ byReads, positiveRate }) {
  const order = ["1 reads", "3 reads", "10 reads", "full"];
  const xs = { "1 reads": 1, "3 reads": 3, "10 reads": 10, full: 30 };
  const W = 560, H = 300, L = 52, R = 110, T = 16, B = 44;
  const sx = scale(Math.log10(1), Math.log10(30), L, W - R), sy = scale(0, 0.7, H - B, T);
  const lines = Object.entries(byReads);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="PR AUC by number of reads per site, for two cell lines">
        {[0, 0.2, 0.4, 0.6].map((v) => (
          <g key={v}>
            <line className="grid" x1={L} x2={W - R} y1={sy(v)} y2={sy(v)} />
            <text className="tick" x={L - 8} y={sy(v) + 4} textAnchor="end">{v.toFixed(1)}</text>
          </g>
        ))}
        {order.map((o) => (
          <text key={o} className="tick" x={sx(Math.log10(xs[o]))} y={H - B + 18} textAnchor="middle">
            {o === "full" ? "20+" : o.split(" ")[0]}
          </text>
        ))}
        <text className="axis-label" x={(L + W - R) / 2} y={H - 6} textAnchor="middle">Molecules read at each site</text>
        <text className="axis-label" transform={`translate(14,${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle">PR AUC</text>
        {lines.map(([cl, v], i) => {
          const d = order.map((o, j) => `${j ? "L" : "M"}${sx(Math.log10(xs[o]))},${sy(v[o])}`).join("");
          const end = v.full;
          return (
            <g key={cl} className={i ? "series neutral" : "series accent"}>
              <line className="baseline" x1={L} x2={W - R} y1={sy(positiveRate[cl])} y2={sy(positiveRate[cl])} />
              <path d={d} />
              {order.map((o) => <circle key={o} cx={sx(Math.log10(xs[o]))} cy={sy(v[o])} r="4" />)}
              <text className="series-label" x={W - R + 8} y={sy(end) + 4}>{cl}</text>
            </g>
          );
        })}
      </svg>
      <figcaption>
        Held-out genes, both labelled cell lines. Dashed lines: a random ranking ({(positiveRate.HCT116 * 100).toFixed(1)}% and{" "}
        {(positiveRate.HEK293T * 100).toFixed(1)}% of sites are modified). Fewer reads, weaker rankings, but well above random even at one.
      </figcaption>
    </figure>
  );
}

/* 5. Against other methods. */
export function BaselineBars({ rows }) {
  const lines = ["HCT116", "HEK293T"];
  return (
    <figure className="chart">
      <table className="mini-table bars">
        <thead><tr><th>Model</th>{lines.map((l) => <th key={l}>{l}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.model} className={r.model === "Our model" ? "ours" : ""}>
              <td>{r.model}{r.note && <span className="footnote">*</span>}</td>
              {lines.map((l) => (
                <td key={l}>
                  <span className="bar-cell">
                    <span className="bar" style={{ width: `${(r[l] / 0.65) * 100}%` }} />
                    <span className="num">{r[l].toFixed(2)}</span>
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <figcaption>PR AUC on held-out genes. *{rows.find((r) => r.note)?.note}.</figcaption>
    </figure>
  );
}

/* 6. Per motif: how well the model orders sites of each letter pattern. */
export function MotifDots({ acc }) {
  const items = useMemo(() => Object.entries(acc).sort((a, b) => b[1] - a[1]), [acc]);
  const W = 560, rowH = 20, L = 70, R = 60, T = 22;
  const H = T + items.length * rowH + 30;
  const sx = scale(0.5, 1, L, W - R);
  const group = (v) => (v >= 0.84 ? "strong" : v >= 0.75 ? "middle" : "weak");
  return (
    <figure className="chart compact">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Pairwise ranking accuracy for each DRACH motif">
        {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map((v) => (
          <g key={v}>
            <line className="grid" x1={sx(v)} x2={sx(v)} y1={T - 8} y2={H - 26} />
            <text className="tick" x={sx(v)} y={H - 8} textAnchor="middle">{v.toFixed(1)}</text>
          </g>
        ))}
        <text className="tick" x={sx(0.5)} y={12} textAnchor="middle">coin flip</text>
        {items.map(([m, v], i) => (
          <g key={m} transform={`translate(0,${T + i * rowH + rowH / 2})`}>
            <text className="motif" x={L - 10} y="4" textAnchor="end">{rna(m)}</text>
            <line className="stem" x1={sx(0.5)} x2={sx(v)} />
            <circle className={`dot ${group(v) === "strong" ? "accent" : group(v) === "middle" ? "mid" : "neutral"}`} cx={sx(v)} r="5" />
            <text className="tick" x={sx(v) + 10} y="4">{v.toFixed(2)}</text>
          </g>
        ))}
      </svg>
      <figcaption>
        Pick one tagged and one untagged site with the same five letters: how often the model ranks the tagged one higher
        (worse of the two cell lines, full reads). Rarely tagged patterns are ranked least well; UAACC has too few examples to judge.
      </figcaption>
    </figure>
  );
}
