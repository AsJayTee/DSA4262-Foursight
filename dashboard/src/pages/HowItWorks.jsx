import { useState } from "react";
import { useJson, topPercent } from "../lib/data.js";
import { RnaText, ReadNetwork, SiteGraph } from "../components/ModelViz.jsx";
import Doors from "../components/Doors.jsx";

// The method detail sits behind a quiet link at the end of each step's text, and opens
// in place under it, so it reads as a footnote to the step rather than a separate block.
function Step({ n, title, summary, children, detail }) {
  const [open, setOpen] = useState(false);
  return (
    <section className="story-step" id={`step-${n}`}>
      <div className="step-head">
        <span className="step-n">{n}</span>
        <div>
          <h2>{title}</h2>
          {[].concat(summary).map((para, i) => <p className="summary" key={i}>{para}</p>)}
          {detail && (
            <>
              <button type="button" className={open ? "method-link open" : "method-link"} aria-expanded={open}
                aria-controls={`method-${n}`} onClick={() => setOpen(!open)}>
                Method notes
                <svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5 6 7.5 9 4.5" /></svg>
              </button>
              <div className={open ? "method-notes open" : "method-notes"} id={`method-${n}`} hidden={!open}>
                <div>{detail}</div>
              </div>
            </>
          )}
        </div>
      </div>
      {children}
    </section>
  );
}

// What the worked example's cell line is, for readers without biology.
const LINE_DESC = { MCF7: "a breast cancer cell line" };

export default function HowItWorks() {
  const { data, error } = useJson("model_story.json");
  if (error) return <div className="page-narrow"><p role="alert">{error}</p></div>;
  if (!data) return <div className="page-narrow"><p role="status">Loading…</p></div>;
  const { slider, sites, network } = data;

  return (
    <div className="story">
      <header className="story-intro page-narrow">
        <p className="eyebrow">How it works</p>
        <h1>How the model ranks a site</h1>
        <p className="lede">
          The model treats an RNA as text and each candidate site on it as a small table of measurements. It learned
          from sites where lab experiments in two human cell lines had already established, site by site, whether
          m6A is present. It scores a site from its own table, then refines the score using the sites around it.
          Each step below runs on real data: the gene <strong>MYC</strong> in {data.cell_line} cells
          {LINE_DESC[data.cell_line] ? `, ${LINE_DESC[data.cell_line]}` : ""}.
        </p>
      </header>

      <Step n="1" title="Input: sequence and signal per molecule"
        summary={[
          "An RNA is a string of four letters (A, C, G and U). Only some A's can carry m6A: those in the middle of a short letter pattern called DRACH. Every such A is a candidate site, and candidate sites are the only positions the model scores.",
          "The measurements come from nanopore sequencing, which pulls single RNA molecules through a tiny pore and records the electrical current as each stretch of letters passes; a modified A shifts that current slightly. A sample contains many copies of the same RNA, and each copy is read separately. So for each candidate site we get a table: one row per molecule that contains the site, and nine numbers per row describing the current around that position. The number of rows differs from site to site, from one to several hundred, so the model must accept a table of any height.",
        ]}
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

      <Step n="2" title="Scoring a site from its molecules"
        summary="Every row passes through the same small neural network. The outputs are averaged over the rows, so their order does not matter and any number of rows works. A second network turns that average into a score. Press play to feed the molecules in one at a time and watch the score settle as evidence accumulates."
        detail={
          <>
            <p>
              This is a Deep Sets encoder (Zaheer et al., 2017). The row network φ maps 37 inputs (9 measurements and 28
              letter inputs) through two ReLU layers of 64 units. The site is summarised by the mean and standard
              deviation of those 64-unit outputs, the log of the row count and the site's 28 letter inputs, added again
              at this stage; a site network ρ (157 → 64 → 64) and a linear head give the score from the site's own
              molecules. Pooling makes the model invariant to row order and indifferent to table height.
            </p>
            <p>
              The diagram shows real activations of one of the model's six networks (the residual design described in
              step 3, first random start) for the site at position {slider.position}, {network.inputs.length} molecules
              in a fixed random order. With few molecules
              the score swings; with more it settles. This is why we report the number of molecules beside every rank.
            </p>
          </>
        }>
        <ReadNetwork network={network} table={network.inputs} site={slider} />
      </Step>

      <Step n="3" title="Adding evidence from nearby sites"
        summary="Modified sites tend to occur near each other, so a site's neighbours carry evidence about it. To use that evidence, each RNA becomes a graph, a network of connections: candidate sites are its nodes, and sites within 150 letters of each other are connected, more strongly when closer. Each site receives its neighbours' evidence and updates its score. Click a site in the graph to pass its neighbours' evidence to it and see the scores change."
        detail={
          <>
            <p>
              The clustering is measured in our training labels: two modified sites less than 25 letters apart occur
              about 2.2 times as often as expected from how heavily their transcripts are modified overall, and the
              excess fades to nothing by 200–400 letters.
            </p>
            <p>
              Three graph designs are used, and each is trained twice from different random starting points, giving
              the six networks combined in step 4. The first is a two-head H2GCN (Zhu et al., 2020) that keeps a site's
              own representation separate from its neighbours'. The second, the residual design, adds a correction to
              the site's own score from neighbours within 150 letters, weighted by e<sup>−d/λ</sup> with
              λ = {data.kernel_nt} letters fitted on training labels. In the third, each neighbour passes only its own
              score. The last two are trained with
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

      <Step n="4" title="From six networks to one ranking"
        summary="The model is six networks: the three graph designs from step 3, each trained twice. Each network scores every candidate site in a sample, and the scores are turned into ranks. The final result is the average of the six ranks, expressed as a percentile: a site in the top 2% outranks 98% of the candidate sites in that sample. This is the number shown beside every site in Explore sites. It is a ranking, not a probability: it tells a researcher which sites to test first, not the chance that a given site is modified."
        detail={
          <>
            <p>
              Averaging ranks rather than raw outputs stops any one network from dominating just because its outputs
              span a wider range. The networks' raw outputs are not calibrated probabilities, so a rank is the more
              honest summary.
            </p>
            <p>
              Ranks are computed within one file (one sample, or one cell line's samples pooled), so a rank in one
              cell line cannot be compared directly with a rank in another. Where scores must be compared across
              files, our analyses use the average of the six networks' raw outputs instead.
            </p>
          </>
        } />

      <Doors to={["explore", "stats"]} />
    </div>
  );
}
