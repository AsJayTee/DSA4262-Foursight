import { useMemo, useState } from "react";
import { useJson } from "../lib/data.js";
import Doors from "../components/Doors.jsx";
import { lineName } from "../lib/sites.js";
import { AucScatter, GroupedBars, Legend, LineChart, LineSelect, ReliabilityAuto, Toggle } from "../components/StatsCharts.jsx";

const pct = (v) => (v == null ? null : +(v * 100).toFixed(1));
// "Both cell lines" is the unweighted mean of the HCT116 and HEK293T values: one series,
// drawn exactly like a single line. Pooling the sites instead would mix two positive
// rates (4.5% and 7.2%) inside one PR AUC.
const mean2 = (o) => (o.HCT116 + o.HEK293T) / 2;
function withMean(s) {
  const perf = s.performance.map((r) => ({ ...r, pr: { ...r.pr, both: mean2(r.pr) }, roc: { ...r.roc, both: mean2(r.roc) } }));
  const depth = { ...s.depth, both: Object.fromEntries(Object.keys(s.depth.HCT116).map((d) => [d, {
    pr_auc: mean2({ HCT116: s.depth.HCT116[d].pr_auc, HEK293T: s.depth.HEK293T[d].pr_auc }),
    pair_acc: mean2({ HCT116: s.depth.HCT116[d].pair_acc, HEK293T: s.depth.HEK293T[d].pair_acc }) }])) };
  const topk = Object.fromEntries(Object.entries(s.topk).map(([m, v]) => [m, { ...v, both: v.HCT116.map((x, i) => (x + v.HEK293T[i]) / 2) }]));
  const gene_best = Object.fromEntries(Object.entries(s.gene_best).map(([k, bands]) => [k, bands.map((b) => ({ ...b, both: mean2(b) }))]));
  let dvm = s.depth_vs_m6anet;
  if (dvm) {
    const both = {};
    for (const d of Object.keys(dvm.HCT116)) {
      both[d] = {};
      for (const m of ["model", "m6anet"]) {
        both[d][m] = Object.fromEntries(Object.keys(dvm.HCT116[d][m]).map((k) => [k, mean2({ HCT116: dvm.HCT116[d][m][k], HEK293T: dvm.HEK293T[d][m][k] })]));
      }
    }
    dvm = { ...dvm, both };
  }
  return { ...s, performance: perf, depth, topk, gene_best, depth_vs_m6anet: dvm };
}

function Section({ id, title, children, note, control }) {
  return (
    <section className="stat-section" id={id}>
      <div className="stat-head"><h2>{title}</h2>{control}</div>
      {children}
      {note && <p className="stat-note">{note}</p>}
    </section>
  );
}

export default function ModelStats() {
  const { data: raw, error } = useJson("model_stats.json");
  const s = useMemo(() => (raw ? withMean(raw) : null), [raw]);
  const { data: explore } = useJson("explore.json");
  const [aucLine, setAucLine] = useState("both");
  const [depthLine, setDepthLine] = useState("both");
  const [depthMetric, setDepthMetric] = useState("pr_auc");
  const [kLine, setKLine] = useState("both");
  const [calLine, setCalLine] = useState("both");
  const [source, setSource] = useState("HCT116");
  if (error) return <div className="page-narrow"><p role="alert">{error}</p></div>;
  if (!s || !explore) return <div className="page-narrow"><p role="status">Loading…</p></div>;

  const depthOrder = ["1 reads", "3 reads", "10 reads", "full"];
  const dvm = s.depth_vs_m6anet;
  const medians = s.sgnex_depth.map((r) => r.median);
  const sgMedian = `${Math.min(...medians)}–${Math.max(...medians)}`;
  const cut = explore.trust.map((t) => t.min);
  const L = s.landmarks;
  const lab = L.labelled[source];
  const sg = L.sgnex;
  const sgLines = Object.entries(sg.lines);
  const mean = (key, n) => Array.from({ length: n }, (_, i) => {
    const v = sgLines.map(([, d]) => d[key][i]).filter((x) => x != null);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  });
  const landmarkSeries = (key) => (source === "SG-NEx"
    ? [...sgLines.map(([, d]) => ({ name: null, values: d[key], style: "thin" })),
      { name: "Mean of 7 cell lines", values: mean(key, sg[`${key}_centre`].length), style: "accent" },
      { name: "Individual cell lines", values: [], style: "thin" }]
    : [{ name: "m6ACE-seq labels", values: lab[key].labels, style: "dash" },
      { name: "This model", values: lab[key].model, style: "accent" },
      { name: "m6Anet", values: lab[key].m6anet, style: "mid" }]);
  const sourceNote = source === "SG-NEx"
    ? `SG-NEx sites with ≥ ${sg.min_reads} reads; positive = top ${sg.top * 100}% of each cell line.`
    : `Held-out sites; model and m6Anet positive = top ${pct(lab.base_rate)}% of the file, the labelled positive rate.`;
  const topkChart = (line, legend = true) => (
    <LineChart key={line} x={[1, 2, 3, 4, 5]} xTicks={[1, 2, 3, 4, 5]} xLabel="Sites tested (top k)" yLabel="Genes recovered (%)"
      yMax={100} yTicks={[0, 25, 50, 75, 100]} legend={legend}
      series={[["This model", "model", "accent"], ["m6Anet", "m6Anet (pretrained)", "fg"], ["Motif rule", "motif rule", "dash"], ["Random", "random", "mid"]]
        .map(([name, key, style]) => ({ name, values: s.topk[key][line].map(pct), style }))} />
  );
  const BANDS = ["Top 1%", "1–5%", "5–10%", "10–25%", "Below 25%"];
  const calChart = (line) => (
    <GroupedBars key={line} categories={BANDS} series={[
      { name: "All reads (≥ 20 per site)", style: "accent", values: s.gene_best.full.map((b) => b[line]) },
      { name: "One read per site (subsampled)", style: "mid", values: s.gene_best.one_read.map((b) => b[line]) },
    ]} />
  );

  return (
    <div className="stats">
      <header className="stats-intro">
        <p className="eyebrow">Model stats</p>
        <h1>Model performance</h1>
        <p className="lede">
          Held-out evaluation of the site-graph ensemble in HCT116 (m6ACE-seq labels) and HEK293T, using five-fold
          cross-validation grouped by gene, and the positional distribution of its predictions in seven SG-NEx cell
          lines. Where both cell lines are selected, values are the unweighted mean of the two.
        </p>
      </header>

      <Section id="discrimination" title="Discrimination" control={<LineSelect value={aucLine} onChange={setAucLine} />}
        note="PR AUC against ROC AUC on held-out genes; a random ranking has PR AUC 0.045 (HCT116) and 0.072 (HEK293T). m6Anet (hollow, †) was trained on HCT116 and is not held out in that cell line.">
        <AucScatter rows={s.performance} lines={[aucLine]} />
      </Section>

      <Section id="depth" title="Read depth" control={<LineSelect value={depthLine} onChange={setDepthLine} />}
        note={dvm
          ? `Held-out sites subsampled to 1, 3 and 10 reads, with identical reads given to both methods; SG-NEx sites have a median of ${sgMedian} reads. m6Anet was run with its 20-read minimum lowered and is otherwise unchanged. It is not held out on HCT116, and the four HEK293T sites it does not score are excluded.`
          : `Held-out sites subsampled to 1, 3 and 10 reads; training sites all have ≥ 20 reads. SG-NEx sites have a median of ${sgMedian} reads.`}>
        {dvm && <Toggle label="Metric" value={depthMetric} onChange={setDepthMetric} options={[
          ["pr_auc", "PR AUC", <><strong>Area under the precision–recall curve.</strong> Primary metric; a random ranking scores the positive rate.</>],
          ["pair_acc", "ROC AUC", <><strong>Area under the ROC curve.</strong> Probability that a positive site outranks a negative one; 0.5 is random.</>],
          ["top3", "Top-3 recovery", <><strong>Top-3 recovery.</strong> Share of held-out genes whose three top-ranked sites include an m6ACE-seq-positive site.</>],
        ]} />}
        {dvm ? (
          <LineChart x={[0, 1, 2, 3]} xTicks={[0, 1, 2, 3]} xFormat={(v) => ["1", "3", "10", "≥ 20"][v]}
            xLabel="Reads per site" yLabel={{ pr_auc: "PR AUC", pair_acc: "ROC AUC", top3: "Genes recovered (%)" }[depthMetric]}
            yMax={{ pr_auc: 0.7, pair_acc: 1, top3: 100 }[depthMetric]}
            yTicks={{ pr_auc: [0, 0.2, 0.4, 0.6], pair_acc: [0, 0.25, 0.5, 0.75, 1], top3: [0, 25, 50, 75, 100] }[depthMetric]}
            series={[["This model", "model", "accent"], ["m6Anet", "m6anet", "fg-dash"]].map(([name, key, style]) => ({
              name, style,
              values: depthOrder.map((d) => { const v = dvm[depthLine][d][key][depthMetric]; return depthMetric === "top3" ? +(v * 100).toFixed(1) : v; }),
            }))} />
        ) : (
          <LineChart x={[0, 1, 2, 3]} xTicks={[0, 1, 2, 3]} xFormat={(v) => ["1", "3", "10", "≥ 20"][v]}
            xLabel="Reads per site" yLabel="PR AUC" yMax={0.7} yTicks={[0, 0.2, 0.4, 0.6]}
            series={[{ name: "This model", values: depthOrder.map((d) => s.depth[depthLine][d].pr_auc), style: "accent" }]} />
        )}
      </Section>

      <Section id="sgnex-depth" title="Read depth in SG-NEx"
        note="Reads per site with all samples of a cell line pooled and transcript copies of one position merged, as in Explore sites.">
        <DepthTable rows={s.sgnex_depth} />
      </Section>

      <Section id="shortlist" title="Shortlist recovery" control={<LineSelect value={kLine} onChange={setKLine} />}
        note="Share of held-out genes (≥ 5 sites, ≥ 1 positive) whose k top-ranked sites include an m6ACE-seq-positive site, at full depth. The motif rule orders sites by the positive rate of their DRACH motif in training genes.">
        {topkChart(kLine)}
      </Section>

      <Section id="calibration" title="Gene-level calibration" control={<LineSelect value={calLine} onChange={setCalLine} />}
        note="Genes are grouped by the percentile of their highest-ranked site; bars show the share carrying at least one m6ACE-seq-positive site. Grey bars repeat the analysis with every site subsampled to a single read.">
        <p className="stat-q">How often does a gene carry m6A, given the rank of its best site?</p>
        {calChart(calLine)}
      </Section>

      <Section id="reliability" title="Reliability by motif and depth"
        note={`Probability that a positive site outranks a negative site of the same motif and depth (ROC AUC within stratum, worse of the two cell lines). Explore labels sites high (≥ ${cut[0]}), moderate (≥ ${cut[1]}) or low; n.d., too few positives.`}>
        <ReliabilityAuto data={s.reliability} depths={s.depths} cutoffs={cut} />
      </Section>

      <Section id="position" title="Transcript position"
        control={<LineSelect label="Data" value={source} onChange={setSource}
          options={[["HCT116", "HCT116 (labelled)"], ["HEK293T", "HEK293T (labelled)"], ["SG-NEx", "SG-NEx, 7 cell lines"]]} />}
        note={`${sourceNote} Enrichment is the share of positive sites in a bin relative to the overall share. The model reproduces the enrichment downstream of the stop codon but depletes sites within ~75 nt of exon junctions more strongly than the labels.`}>
        <div className="two">
          <div>
            <h3>Distance to stop codon</h3>
            <LineChart x={source === "SG-NEx" ? sg.stop_centre : lab.stop.centre} xTicks={[-1000, -500, 0, 500, 1000]}
              xFormat={(v) => (v > 0 ? `+${v}` : `${v}`)} xLabel="nt (negative = upstream)" yLabel="Enrichment (×)"
              refY={1} refX={0} yMax={3.6} yTicks={[0, 1, 2, 3]} series={landmarkSeries("stop")} legend={false} />
          </div>
          <div>
            <h3>Distance to nearest exon junction</h3>
            <LineChart x={source === "SG-NEx" ? sg.junction_centre : lab.junction.centre} xTicks={[0, 250, 500, 750, 1000]}
              xLabel="nt" yLabel="Enrichment (×)" refY={1} yMax={4.2} yTicks={[0, 1, 2, 3, 4]} series={landmarkSeries("junction")} legend={false} />
          </div>
        </div>
        <Legend series={landmarkSeries("stop")} className="centred" />
      </Section>

      <Section id="limitations" title="Limitations">
        <ul className="plain-list">
          <li><strong>Ranks, not probabilities.</strong> Scores are within-sample percentiles: uncalibrated and not comparable across samples.</li>
          <li><strong>Presence, not stoichiometry.</strong> Trained on binary site labels; the score does not estimate the modified fraction.</li>
          <li><strong>Low depth is simulated.</strong> All labelled sites have ≥ 20 reads; performance at 1–3 reads is estimated by subsampling.</li>
          <li><strong>No differential calls.</strong> Rank agreement between replicates of one line matches that between lines.</li>
          <li><strong>Junction-proximal sites.</strong> Sites within ~75 nt of an exon junction are likely under-ranked.</li>
          <li><strong>Label noise.</strong> m6ACE-seq misses true sites, so reported accuracy is likely an underestimate.</li>
        </ul>
      </Section>
      <Doors to={["explore", "how"]} />
    </div>
  );
}

const BANDS = [["1-2", "1–2"], ["3-9", "3–9"], ["10-19", "10–19"], ["20+", "≥ 20"]];

/* Reads per site in each SG-NEx line: counts, then the share in each depth band as a bar. */
function DepthTable({ rows }) {
  return (
    <div className="table-scroll">
      <table className="depth-table">
        <thead>
          <tr>
            <th scope="col">Cell line</th><th scope="col" className="num sites-col">Sites</th><th scope="col" className="num">Median reads</th>
            {BANDS.map(([k, t]) => <th key={k} scope="col" className="num">{t} reads</th>)}
            <th scope="col" className="bar-col"><span className="sr-only">Share by band</span></th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.line}>
              <th scope="row">{lineName(r.line)}</th>
              <td className="num sites-col">{r.sites.toLocaleString()}</td>
              <td className="num">{r.median}</td>
              {BANDS.map(([k]) => <td key={k} className="num">{(r.bands[k] * 100).toFixed(0)}%</td>)}
              <td className="bar-col">
                <span className="band-bar">
                  {BANDS.map(([k], i) => <span key={k} className={`b${i}`} style={{ width: `${r.bands[k] * 100}%` }} />)}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
