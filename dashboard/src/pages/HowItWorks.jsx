import { useJson, topPercent } from "../lib/data.js";
import { RnaText, ReadNetwork, SiteGraph } from "../components/ModelViz.jsx";
import Doors from "../components/Doors.jsx";

function Step({ n, title, summary, children, detail }) {
  return (
    <section className="story-step" id={`step-${n}`}>
      <div className="step-head">
        <span className="step-n">{n}</span>
        <div>
          <h2>{title}</h2>
          <p className="summary">{summary}</p>
        </div>
      </div>
      {children}
      {detail && (
        <details className="tech">
          <summary>Technical detail</summary>
          {detail}
        </details>
      )}
    </section>
  );
}

export default function HowItWorks() {
  const { data, error } = useJson("model_story.json");
  if (error) return <div className="page-narrow"><p role="alert">{error}</p></div>;
  if (!data) return <div className="page-narrow"><p role="status">Loading…</p></div>;
  const { slider, sites, network } = data;

  return (
    <div className="story">
      <header className="story-intro page-narrow">
        <p className="eyebrow">How it works</p>
        <h1>How the model ranks a site.</h1>
        <p className="lede">
          The model treats an RNA as text and each candidate site on it as a small table of measurements. It scores
          a site from its own table, then refines the score using the sites around it. Each step below runs on real
          data: the gene <strong>MYC</strong> in {data.cell_line} cells.
        </p>
      </header>

      <Step n="1" title="The input: text and tables"
        summary="An RNA is a string of four letters. Some positions are candidate sites, where the modification can occur. For each candidate site we get a table: one row per molecule measured there, and nine numbers per row describing the signal around that position. The number of rows differs from site to site, from one to several hundred, so the model must accept a table of any height."
        detail={
          <>
            <p>
              The nine numbers are, at three neighbouring positions, how long the molecule stayed in the sensor and the
              average and spread of its electrical signal. Times and spreads are log-transformed, and every column is
              standardised with the training set's mean and scale; the colours above show those standardised values.
              The site's seven surrounding letters are added to every row as 28 one-hot inputs.
            </p>
            <p>
              The label is per site, not per molecule: a lab measurement says whether the site is modified, not
              which molecules. Learning a site label from a variable-size set of rows is a multiple-instance problem.
            </p>
          </>
        }>
        <RnaText text={data.text} sites={sites} />
      </Step>

      <Step n="2" title="Reading one site: one network for every row"
        summary="Every row passes through the same small neural network. The outputs are averaged over the rows, so their order does not matter and any number of rows works. A second network turns that average into a score. Press play to feed the molecules in one at a time and watch the score settle as evidence accumulates."
        detail={
          <>
            <p>
              This is a Deep Sets encoder (Zaheer et al., 2017). The row network φ maps 37 inputs (9 measurements and 28
              letter inputs) through two ReLU layers of 64 units. The site is summarised by the mean and standard
              deviation of those 64-unit outputs and the log of the row count; a site network ρ (157 → 64 → 64) and a
              linear head give the score from the site's own molecules. Pooling makes the model invariant to row order
              and indifferent to table height.
            </p>
            <p>
              The diagram shows real activations of one of the six networks (residual design, seed 0) for the site at
              position {slider.position}, {network.inputs.length} molecules in a fixed random order. With few molecules
              the score swings; with more it settles. This is why we report the number of molecules beside every rank.
            </p>
          </>
        }>
        <ReadNetwork network={network} table={network.inputs} site={slider} />
      </Step>

      <Step n="3" title="Sites inform each other: a graph"
        summary="Modified sites tend to occur near each other. So each RNA becomes a graph: candidate sites are nodes, and sites within 150 letters are connected, more strongly when closer. Each site receives its neighbours' evidence and updates its score. Click a site in the graph to pass its neighbours' evidence to it and see the scores change."
        detail={
          <>
            <p>
              Three graph designs are used. One is a two-head H2GCN (Zhu et al., 2020) that keeps a site's own
              representation separate from its neighbours'. Another adds a correction to the site's own score from
              neighbours within 150 letters, weighted by e<sup>−d/λ</sup> with λ = {data.kernel_nt} letters fitted on
              training labels. In the third, each neighbour passes only its own score. The last two are trained with
              neighbours randomly removed, so no single neighbour can carry a site, which helps them transfer to new
              cell lines.
            </p>
            <p>
              The highlighted site is {topPercent(slider.alone)} of the sample on its own molecules and{" "}
              {topPercent(slider.rank)} after its neighbours are taken into account. A site with no neighbours within
              150 letters gets no help from this step.
            </p>
          </>
        }>
        <SiteGraph sites={sites} highlight={slider.index} radius={data.radius_nt} kernel={data.kernel_nt} />
      </Step>

      <Doors to={["explore", "stats"]} />
    </div>
  );
}
