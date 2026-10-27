import { lazy, Suspense } from "react";
import { useJson } from "../lib/data.js";
import { useNarrow } from "../lib/media.js";
import Doors from "../components/Doors.jsx";

// three.js loads only with this page, not with the rest of the site.
const RnaStrand = lazy(() => import("../components/RnaStrand.jsx"));

function Letters({ tagged }) {
  // A short RNA "word" with one tagged A: the idea of m6A in one line.
  const seq = "GGACUAGGACU".split("");
  return (
    <svg viewBox="0 0 330 70" className="letters" role="img" aria-label="A row of RNA letters; one A carries a small tag">
      {seq.map((ch, i) => {
        const on = tagged && i === 2; // the A of GGACU
        return (
          <g key={i} transform={`translate(${i * 30 + 15},44)`}>
            {on && <circle cy="-30" r="7" className="tag-dot" />}
            {on && <line y1="-23" y2="-14" className="tag-link" />}
            <text textAnchor="middle" className={on ? "letter on" : "letter"}>{ch}</text>
          </g>
        );
      })}
    </svg>
  );
}

// The tagged A from the card before, on its own.
function TaggedA() {
  return (
    <svg viewBox="0 0 30 70" className="tagged-a" role="img" aria-label="A tagged A">
      <g transform="translate(15,44)">
        <circle cy="-30" r="7" className="tag-dot" />
        <line y1="-23" y2="-14" className="tag-link" />
        <text textAnchor="middle" className="letter on">A</text>
      </g>
    </svg>
  );
}

function ResultBars({ hits }) {
  const rows = [
    { label: "Our model's top 3", key: "model" },
    { label: "Textbook rule (by letters)", key: "motif rule" },
    { label: "Random picks", key: "random" },
  ];
  return (
    <div className="result-bars">
      {rows.map((r) => {
        const o = hits[r.key];
        const v = (o.HCT116 + o.HEK293T) / 2;
        return (
          <div className="result-row" key={r.key}>
            <span className="result-label">{r.label}</span>
            <span className="result-track">
              <span className={r.key === "model" ? "result-fill strong" : "result-fill"} style={{ width: `${v * 100}%` }} />
            </span>
            <span className="result-value">{inTen(o)} in 10</span>
          </div>
        );
      })}
      <p className="note">Genes with at least one tag, in the two cell types with lab results.</p>
    </div>
  );
}

// "8–9" from the two labelled cell lines, rather than one averaged figure.
function inTen(o) {
  const [a, b] = [Math.round(o.HEK293T * 10), Math.round(o.HCT116 * 10)].sort((x, y) => x - y);
  return a === b ? `${a}` : `${a}–${b}`;
}

export default function Home() {
  const { data } = useJson("model_story.json");
  const hits = data?.performance.top3_hit;
  // On phones the strand is left out entirely, so three.js is never downloaded.
  const narrow = useNarrow();

  return (
    <div className="home">
      <section className={narrow ? "hero no-strand" : "hero"}>
        {/* Full-width layer behind the text, so the strand can run off the screen. */}
        {!narrow && (
          <Suspense fallback={<div className="hero-canvas" />}>
            <RnaStrand className="hero-canvas" />
          </Suspense>
        )}
        <div className="hero-inner">
          <div className="hero-copy">
            <p className="eyebrow">Predicting m6A modifications</p>
            <h1>Where does RNA carry m6A?</h1>
            <p className="lede">
              m6A is the most common chemical modification of messenger RNA. It affects how long an RNA lasts
              and how much protein is made from it, and abnormalities in its occurrence are linked to cancer.
              We built a model to predict where m6A modifications occur on RNA.
            </p>
          </div>
        </div>
      </section>

      <section className="band">
        <h2>What is m6A?</h2>
        <div className="three-up">
          <article>
            <Letters tagged={false} />
            <h3>RNA is a message</h3>
            <p>Your DNA is copied into RNA, a string of four letters (A, C, G, U) that tells the cell which protein to make.</p>
          </article>
          <article>
            <Letters tagged />
            <h3>Some A's get a tag</h3>
            <p>At particular sites, a small chemical group is added to an A. That tagged A is called m6A, and it is the most common tag found on our messages.</p>
          </article>
          <article>
            <div className="fate" aria-hidden="true">
              <TaggedA />
              <span className="arrow">→</span>
              <span className="chip">different behaviour</span>
            </div>
            <h3>A tag is an instruction</h3>
            <p>A tag changes how a cell builds a protein. When the tagging mechanism goes wrong, it can contribute to diseases such as leukaemia.</p>
          </article>
        </div>
      </section>

      <section className="band">
        <h2>Why are the tags hard to find?</h2>
        <p className="wide">
          RNA has thousands of places where a tag could sit, and confirming even one of them in the lab takes
          weeks. Scientists need to know where to look first.
        </p>
      </section>

      <section className="band">
        <h2>Our solution</h2>
        <p className="wide">
          We built a model from data where lab experiments had already located the m6A sites. We then applied
          it to public data from human cells, so that scientists can see which sites in their gene are most
          likely to carry m6A, and test those first.
        </p>
      </section>

      <section className="band result">
        <div>
          <h2>Does our model help?</h2>
          <p className="wide">
            We tested it on genes the model had never seen. If a scientist checks our top three picks in a gene, at
            least one carries the tag in {hits ? inTen(hits.model) : "8–9"} genes out of 10.
          </p>
        </div>
        {hits && <ResultBars hits={hits} />}
      </section>

      <Doors to={["explore", "how", "stats"]} />
    </div>
  );
}
