import { useMemo, useState } from "react";
import { useJson } from "../lib/data.js";
import Doors from "../components/Doors.jsx";
import { lineName } from "../lib/sites.js";
import { AucScatter, GloriBars, GroupedBars, Legend, LineChart, LineSelect, ReliabilityAuto, Toggle } from "../components/StatsCharts.jsx";

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

// `question` is the section's point in plain words, read before the chart; `note` is the method.
function Section({ id, title, question, children, note, control }) {
  return (
    <section className="stat-section" id={id}>
      <div className="stat-head"><h2>{title}</h2>{control}</div>
      {question && <p className="stat-q">{question}</p>}
      {children}
      {note && <p className="stat-note">{note}</p>}
    </section>
  );
}

export default function ModelStats() {
  const { data: raw, error } = useJson("model_stats.json");
  const s = useMemo(() => (raw ? withMean(raw) : null), [raw]);
  const { data: explore } = useJson("explore.json");
  const { data: glori } = useJson("glori_stats.json");
  const [aucLine, setAucLine] = useState("both");
  const [depthLine, setDepthLine] = useState("both");
  const [depthMetric, setDepthMetric] = useState("pr_auc");
  const [kLine, setKLine] = useState("both");
  const [calLine, setCalLine] = useState("both");
  const [source, setSource] = useState("HCT116");
  if (error) return <div className="page-narrow"><p role="alert">{error}</p></div>;
  if (!s || !explore) return <div className="page-narrow"><p role="status">Loading…</p></div>;

  const depthOrder = ["1 reads", "3 reads", "10 reads", "full"];
  const randomRates = glori ? glori.lines.map((l) => glori.by_line[l].all.positive_rate) : [];
  const randomRange = `${Math.min(...randomRates).toFixed(2)}–${Math.max(...randomRates).toFixed(2)}`;
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
    ? `SG-NEx sites with ≥ ${sg.min_reads} reads; a ranking has no yes/no answer, so the top ${sg.top * 100}% of each cell line's sites count as positive.`
    : `Held-out sites. To compare a ranking with yes/no labels, the model's and m6Anet's top ${pct(lab.base_rate)}% of sites count as positive, the same share as the labels.`;
  const depthTitle = (v) => ["1 read per site", "3 reads per site", "10 reads per site", "All reads (≥ 20 per site)"][v];
  const topkChart = (line, legend = true) => (
    <LineChart key={line} x={[1, 2, 3, 4, 5]} xTicks={[1, 2, 3, 4, 5]} xLabel="Sites tested (top k)" yLabel="Genes recovered (%)"
      yMax={100} yTicks={[0, 25, 50, 75, 100]} legend={legend}
      points popTitle={(k) => `Top ${k} site${k > 1 ? "s" : ""} tested`} valueFormat={(v) => `${v.toFixed(1)}%`}
      series={[["This model", "model", "accent"], ["m6Anet", "m6Anet (pretrained)", "fg"], ["Motif rule", "motif rule", "dash"], ["Random", "random", "mid"]]
        .map(([name, key, style]) => ({ name, values: s.topk[key][line].map(pct), style }))} />
  );
  const BANDS = ["Top 1%", "1–5%", "5–10%", "10–25%", "Below 25%"];
  const calChart = (line) => (
    <GroupedBars key={line} categories={BANDS} series={[
      { name: "All reads (≥ 20 per site)", style: "accent", values: s.gene_best.full.map((b) => b[line]) },
      { name: "One read per site (simulated)", style: "mid", values: s.gene_best.one_read.map((b) => b[line]) },
    ]} />
  );

  return (
    <div className="stats">
      <header className="stats-intro">
        <p className="eyebrow">Model stats</p>
        <h1>Model performance</h1>
        <p className="lede">
          How far can you trust the model's ranking of sites? This page measures it on genes the model never saw
          while it was being trained, in the two human cell lines where lab experiments had already located the m6A
          sites: HCT116 (colon cancer) and HEK293T (embryonic kidney). It then checks that the model's predictions in
          the seven public SG-NEx cell lines fall where m6A is known to occur along a gene.
        </p>
      </header>

      <details className="glossary">
        <summary>Glossary</summary>
        <dl>
          <dt>Held out</dt><dd>Measured on genes the model was not trained on.</dd>
          <dt>Read depth</dt><dd>How many RNA molecules were measured at a site.</dd>
          <dt>PR AUC</dt><dd>How well a ranking puts modified sites at the top. A random ranking scores the share
            of sites that are modified.</dd>
          <dt>ROC AUC</dt><dd>The chance that a modified site ranks above an unmodified one. 0.5 is random.</dd>
          <dt>m6ACE-seq, GLORI</dt><dd>Two lab methods that locate m6A. m6ACE-seq gave the labels the model learned
            from; GLORI is an independent check.</dd>
          <dt>Both cell lines</dt><dd>The average of HCT116 and HEK293T.</dd>
        </dl>
      </details>

      <Section id="discrimination" title="Comparison with other methods" control={<LineSelect value={aucLine} onChange={setAucLine} />}
        question="How well does each method separate modified sites from unmodified ones?"
        note="Each point is one method, scored on held-out genes; higher and further right is better. Click a point for how the method works. m6Anet (hollow, †) was trained on HCT116, so its HCT116 values are not held out.">
        <AucScatter rows={s.performance} lines={[aucLine]} />
      </Section>

      <Section id="shortlist" title="Shortlist recovery" control={<LineSelect value={kLine} onChange={setKLine} />}
        question="If you test the top few sites the model picks in a gene, how often is at least one of them modified?"
        note="Share of held-out genes (≥ 5 sites, ≥ 1 positive) whose k top-ranked sites include an m6ACE-seq-positive site, at full depth. The motif rule orders sites by the positive rate of their DRACH motif in training genes.">
        {topkChart(kLine)}
      </Section>

      <Section id="calibration" title="Gene-level calibration" control={<LineSelect value={calLine} onChange={setCalLine} />}
        question="How often does a gene carry m6A, given the rank of its best site?"
        note="Genes are grouped by the percentile of their highest-ranked site; bars show the share carrying at least one m6ACE-seq-positive site. Grey bars repeat the analysis with reads randomly removed until every site has a single read.">
        {calChart(calLine)}
      </Section>

      {glori && (
        <Section id="glori" title="New cell lines, checked by an independent lab method"
          question="Does the ranking still hold up in cell lines the model never saw, judged by a different lab method?"
          note={`GLORI is a chemical method that maps m6A one letter at a time, independently of the m6ACE-seq labels the model learned from (A549 and H9: GEO GSE290408; HepG2: GSE301419). Each bar is the PR AUC with all of a cell line's candidate sites ranked together, deep and shallow, as in Explore sites; a random ranking would score ${randomRange}. Hover over a cell line for the split by reads per site. GLORI lists only the sites it detected, so every value is a lower bound, for both methods alike. HCT116 is left out because the model was trained on it.`}>
          <GloriBars data={glori} names={lineName} />
        </Section>
      )}

      <Section id="depth" title="Read depth" control={<LineSelect value={depthLine} onChange={setDepthLine} />}
        question="How much worse does the model get when fewer molecules are read at a site?"
        note={dvm
          ? `Every labelled site has at least 20 reads, so lower depths are simulated: reads are removed at random until 1, 3 or 10 remain, and both methods get the same reads. SG-NEx sites have a median of ${sgMedian} reads. m6Anet was run with its 20-read minimum lowered and is otherwise unchanged. It is not held out on HCT116, and the four HEK293T sites it does not score are excluded.`
          : `Every labelled site has at least 20 reads, so lower depths are simulated: reads are removed at random until 1, 3 or 10 remain. SG-NEx sites have a median of ${sgMedian} reads.`}>
        {dvm && <Toggle label="Metric" value={depthMetric} onChange={setDepthMetric} options={[
          ["pr_auc", "PR AUC", <><strong>Area under the precision–recall curve.</strong> Primary metric; a random ranking scores the positive rate.</>],
          ["pair_acc", "ROC AUC", <><strong>Area under the ROC curve.</strong> Probability that a positive site outranks a negative one; 0.5 is random.</>],
          ["top3", "Top-3 recovery", <><strong>Top-3 recovery.</strong> Share of held-out genes whose three top-ranked sites include an m6ACE-seq-positive site.</>],
        ]} />}
        {dvm ? (
          <LineChart x={[0, 1, 2, 3]} xTicks={[0, 1, 2, 3]} xFormat={(v) => ["1", "3", "10", "≥ 20"][v]}
            points popTitle={depthTitle} valueFormat={depthMetric === "top3" ? (v) => `${v}%` : (v) => v.toFixed(3)}
            xLabel="Reads per site" yLabel={{ pr_auc: "PR AUC", pair_acc: "ROC AUC", top3: "Genes recovered (%)" }[depthMetric]}
            yMax={{ pr_auc: 0.7, pair_acc: 1, top3: 100 }[depthMetric]}
            yTicks={{ pr_auc: [0, 0.2, 0.4, 0.6], pair_acc: [0, 0.25, 0.5, 0.75, 1], top3: [0, 25, 50, 75, 100] }[depthMetric]}
            series={[["This model", "model", "accent"], ["m6Anet", "m6anet", "fg-dash"]].map(([name, key, style]) => ({
              name, style,
              values: depthOrder.map((d) => { const v = dvm[depthLine][d][key][depthMetric]; return depthMetric === "top3" ? +(v * 100).toFixed(1) : v; }),
            }))} />
        ) : (
          <LineChart x={[0, 1, 2, 3]} xTicks={[0, 1, 2, 3]} xFormat={(v) => ["1", "3", "10", "≥ 20"][v]}
            points popTitle={depthTitle} valueFormat={(v) => v.toFixed(3)}
            xLabel="Reads per site" yLabel="PR AUC" yMax={0.7} yTicks={[0, 0.2, 0.4, 0.6]}
            series={[{ name: "This model", values: depthOrder.map((d) => s.depth[depthLine][d].pr_auc), style: "accent" }]} />
        )}
      </Section>

      <Section id="sgnex-depth" title="Read depth in SG-NEx"
        question="How many molecules does a typical site in the public SG-NEx data have?"
        note="All samples of a cell line are pooled. A position shared by several versions (transcripts) of a gene's RNA counts once, with its molecules added together, as in Explore sites.">
        <DepthTable rows={s.sgnex_depth} />
      </Section>

      <Section id="reliability" title="Reliability by motif and depth"
        question="For a site with a given DRACH motif and number of reads, how often does the model rank a modified site above an unmodified one?"
        note={`Each value is the ROC AUC within one motif-and-depth group, taking the worse of the two cell lines. Depths are simulated by removing reads until exactly 1, 3 or 10 remain; Explore applies them to sites with 1–2, 3–9 and 10–19 reads, which makes its labels slightly conservative. Explore labels sites high (≥ ${cut[0]}), moderate (≥ ${cut[1]}) or low; n.d. means too few modified sites to measure.`}>
        <ReliabilityAuto data={s.reliability} depths={s.depths} cutoffs={cut} />
      </Section>

      <Section id="position" title="Transcript position"
        control={<LineSelect label="Data" value={source} onChange={setSource}
          options={[["HCT116", "HCT116 (labelled)"], ["HEK293T", "HEK293T (labelled)"], ["SG-NEx", "SG-NEx, 7 cell lines"]]} />}
        question="Do the predicted sites fall where m6A is known to occur along a gene?"
        note={`The stop codon is where the protein-coding part of an RNA ends; exon junctions are where separately copied pieces of a gene were joined together. m6A is known to gather just after the stop codon and to be scarce close to junctions. ${sourceNote} Enrichment is the share of positive sites in a bin relative to the overall share. The model is never told where a site sits in its gene, yet it reproduces the rise after the stop codon. Near exon junctions it goes too far: within about 75 nt it ranks sites even lower than the labels show they should be.`}>
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
          <li><strong>Scores are ranks, not probabilities.</strong> A site in the top 1% is more likely to be modified than one in the top 10%, but the number is not the chance that it is modified. Ranks can only be compared within one cell line.</li>
          <li><strong>Most results at low read depth are simulated.</strong> Every lab-labelled site was measured on at least 20 molecules, so we tested fewer by throwing molecules away at random. The GLORI check above, which uses sites that really were measured on few molecules, gives a similar picture.</li>
          <li><strong>Modified sites near exon junctions may be missed.</strong> m6A is rare within about 75 letters of a junction, where two pieces of a gene were joined. The model has learned this too well: it ranks sites there even lower than the lab data suggests, so the few modified ones are easy to miss.</li>
          <li><strong>The lab labels miss some real sites.</strong> When the model ranks one of these highly, it counts as a mistake, so the true accuracy is probably higher than shown.</li>
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
