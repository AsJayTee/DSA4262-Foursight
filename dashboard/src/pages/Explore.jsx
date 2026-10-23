import { useEffect, useMemo, useState } from "react";
import { useJson } from "../lib/data.js";
import { LINES, LOW_DEPTH, lineName, loadGene, loadIndex, loadMeta, loadStructure, rowsFor, searchGenes, toCsv } from "../lib/sites.js";
import { GeneViewer, Head, SiteCells, SitePanel, SiteTable, topPct } from "../components/ExploreParts.jsx";
import { Link } from "../lib/route.jsx";

const EXAMPLES = ["MYC", "TP53", "KRAS", "EGFR", "MALAT1", "GAPDH"];

// The address keeps the choice (#/explore?gene=MYC&line=MCF7) without a page jump.
function writeUrl(gene, line) {
  const q = new URLSearchParams();
  if (gene) q.set("gene", gene);
  if (line) q.set("line", line);
  const s = q.toString();
  window.history.replaceState(null, "", `#/explore${s ? `?${s}` : ""}`);
}

function useAsync(fn, deps) {
  const [state, set] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
    let live = true;
    set((s) => ({ ...s, loading: true }));
    fn().then((data) => live && set({ data, error: null, loading: false }))
      .catch((error) => live && set({ data: null, error, loading: false }));
    return () => { live = false; };
  }, deps);  // eslint-disable-line react-hooks/exhaustive-deps
  return state;
}

function GeneSearch({ index, onPick, autoFocus }) {
  const [q, setQ] = useState("");
  const [cursor, setCursor] = useState(0);
  const [open, setOpen] = useState(false);
  const hits = useMemo(() => (index ? searchGenes(index, q) : []), [index, q]);
  const pick = (g) => { setQ(""); setOpen(false); onPick(g); };
  return (
    <div className="search">
      <input className="search-input" type="search" value={q} autoFocus={autoFocus} spellCheck="false"
        placeholder={index ? "Gene symbol or Ensembl ID" : "Loading gene index…"} disabled={!index}
        role="combobox" aria-expanded={open && hits.length > 0} aria-controls="gene-options" aria-label="Gene"
        aria-activedescendant={hits[cursor] ? `opt-${hits[cursor].id}` : undefined}
        onChange={(e) => { setQ(e.target.value); setCursor(0); setOpen(true); }}
        onFocus={() => setOpen(true)} onBlur={() => setTimeout(() => setOpen(false), 120)}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setCursor((c) => Math.min(c + 1, hits.length - 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)); }
          else if (e.key === "Enter" && hits[cursor]) { e.preventDefault(); pick(hits[cursor]); }
          else if (e.key === "Escape") setOpen(false);
        }} />
      {open && q && (
        <ul className="search-options" id="gene-options" role="listbox">
          {hits.length ? hits.map((g, i) => (
            <li key={g.id} id={`opt-${g.id}`} role="option" aria-selected={i === cursor}
              className={i === cursor ? "on" : ""} onMouseDown={(e) => { e.preventDefault(); pick(g); }}
              onMouseEnter={() => setCursor(i)}>
              <strong>{g.name}</strong><span>{g.id} · chr{g.chrom}</span>
            </li>
          )) : <li className="none">No match for “{q}”.</li>}
        </ul>
      )}
    </div>
  );
}

function LinePicker({ gene, line, onPick }) {
  return (
    <div className="lines" role="radiogroup" aria-label="Cell line">
      {Object.entries(LINES).map(([key, { name, tissue }]) => {
        const n = gene.n[key] || 0;
        return (
          <button key={key} role="radio" aria-checked={line === key} disabled={!n}
            className={line === key ? "line-chip on" : "line-chip"} onClick={() => onPick(key)}>
            <span className="line-name">{name}</span>
            <span className="line-tissue">{tissue}</span>
            <span className="line-n">{n ? `${n} site${n === 1 ? "" : "s"}` : "not covered"}</span>
          </button>
        );
      })}
    </div>
  );
}

function Shortlist({ rows, picked, k, setK, gene, line, onOpen, onRemove }) {
  const chosen = rows.filter((r) => picked.has(r.site));
  const download = () => {
    const url = URL.createObjectURL(new Blob([toCsv(chosen, gene, line)], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${gene.name}_${lineName(line)}_m6A_sites.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <section className="card shortlist" aria-labelledby="shortlist-title">
      <div className="shortlist-head">
        <div>
          <h2 id="shortlist-title">Selected sites</h2>
          <p className="muted small">
            Default: the {k} highest-ranked. Held-out recovery by k is reported in <Link to="/model-stats">Model stats</Link>.
          </p>
        </div>
        <div className="k-picker" role="group" aria-label="Number of top-ranked sites">
          <span>Top k</span>
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} aria-pressed={k === n} onClick={() => setK(n)} disabled={n > rows.length}>{n}</button>
          ))}
        </div>
      </div>
      <div className="table-scroll">
        <table className="sl-table">
          <Head line={line} />
          <tbody>
            {chosen.map((r) => (
              <tr key={r.site} className={r.low ? "low" : ""} onClick={() => onOpen(r)}>
                <SiteCells r={r} chrom={`chr${gene.chrom}`} />
                <td className="rm" onClick={(e) => { e.stopPropagation(); onRemove(r.site); }}>
                  <button aria-label={`Remove site ${r.geneRank}`} title="Remove">×</button>
                </td>
              </tr>
            ))}
            {!chosen.length && <tr><td colSpan="9" className="muted small">No sites selected.</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="shortlist-foot">
        <span className="muted small">{chosen.length} selected · add sites from the table below</span>
        <button className="btn primary small" onClick={download} disabled={!chosen.length}>Export CSV</button>
      </div>
    </section>
  );
}

export default function Explore({ route }) {
  const index = useAsync(loadIndex, []);
  const meta = useAsync(loadMeta, []);
  const { data: explore } = useJson("explore.json");
  const [geneId, setGeneId] = useState(route.params.get("gene") || "");
  const [line, setLine] = useState(route.params.get("line") || "");
  const [k, setK] = useState(3);
  const [picked, setPicked] = useState(new Set());
  const [open, setOpen] = useState(null);           // the site whose panel is open

  const gene = useMemo(() => {
    if (!index.data || !geneId) return null;
    const q = geneId.toUpperCase();
    return index.data.find((g) => g.id === q) || index.data.find((g) => g.name.toUpperCase() === q) || null;
  }, [index.data, geneId]);
  const full = useAsync(() => (gene ? Promise.all([loadGene(gene), loadStructure(gene)]) : Promise.resolve(null)), [gene?.id]);
  const lineOk = gene && line && gene.n[line];
  const rows = useMemo(() => {
    if (!full.data || !meta.data || !explore || !lineOk) return [];
    const [entry] = full.data;
    const [m, trust] = meta.data;
    return rowsFor(entry, line, m, trust, explore.trust);
  }, [full.data, meta.data, explore, line, lineOk]);

  // Follow the address when it changes under us (a shared link opened in this tab, an edit in
  // the address bar). Our own writes use replaceState, which does not fire this, so no loop.
  useEffect(() => {
    setGeneId(route.params.get("gene") || "");
    setLine(route.params.get("line") || "");
  }, [route]);
  // A new gene, line or k selects a fresh top k.
  useEffect(() => { setPicked(new Set(rows.slice(0, k).map((r) => r.site))); setOpen(null); }, [rows, k]);
  useEffect(() => { writeUrl(gene?.name || geneId, line); }, [gene, geneId, line]);

  const pickGene = (g) => {
    setGeneId(g.name);
    if (!(line && g.n[line])) setLine("");
  };
  const toggle = (site) => setPicked((p) => { const n = new Set(p); n.has(site) ? n.delete(site) : n.add(site); return n; });

  if (index.error || meta.error) {
    return (
      <div className="page-narrow">
        <h1>Explore sites</h1>
        <p role="alert" className="lede">
          Site data unavailable. In development it is read from <code>data/dashboard/</code>; generate it with{" "}
          <code>python analysis/sgnex/export_site_shortlist.py</code> and{" "}
          <code>python analysis/sgnex/export_gene_structure.py</code>.
        </p>
      </div>
    );
  }

  const entry = full.data?.[0];
  const structure = full.data?.[1];
  const best = rows[0];

  return (
    <div className="explore">
      <header className="explore-intro">
        <p className="eyebrow">Explore sites</p>
        <h1>Site-level m6A predictions</h1>
        <p className="lede">
          Candidate DRACH sites of a gene, ranked within one SG-NEx cell line, with read depth, replicate
          consistency and estimated reliability for each site.
        </p>
      </header>

      <div className="pickers">
        <section className="step-block" aria-label="Gene">
          <div className="step-label"><span className="step-n">1</span>Gene</div>
          <GeneSearch index={index.data} onPick={pickGene} autoFocus={!geneId} />
          {!gene && (
            <div className="examples">
              <span className="muted small">Examples</span>
              {EXAMPLES.filter((n) => index.data?.some((g) => g.name === n)).map((n) => (
                <button key={n} className="chip" onClick={() => pickGene(index.data.find((g) => g.name === n))}>{n}</button>
              ))}
            </div>
          )}
          {geneId && index.data && !gene && (
            <p className="notice">No candidate sites for “{geneId}”. Check the symbol or use the Ensembl gene ID.</p>
          )}
        </section>

        {gene && (
          <section className="step-block" aria-label="Cell line">
            <div className="step-label"><span className="step-n">2</span>Cell line <span className="muted">· {gene.name}</span></div>
            <LinePicker gene={gene} line={line} onPick={setLine} />
            {line && !gene.n[line] && (
              <p className="notice">{gene.name} has no covered candidate sites in {lineName(line)}. Absence of data does not indicate absence of m6A.</p>
            )}
          </section>
        )}
      </div>

      {gene && lineOk && (
        <section className="result" aria-label="Ranked sites">
          {(full.loading || !explore) && <p role="status" className="muted">Loading {gene.name}…</p>}
          {full.error && <p role="alert" className="notice">Could not load {gene.name}: {String(full.error.message)}</p>}
          {best && explore && entry && (
            <>
              <div className="gene-head">
                <div>
                  <h2>{gene.name} <span className="muted">· {lineName(line)}</span></h2>
                  <p className="muted small">
                    {gene.id} · chr{gene.chrom} ({entry.strand}) · {rows.length} candidate site{rows.length === 1 ? "" : "s"}
                    {" "}· median depth {rows.map((r) => r.reads).sort((x, y) => x - y)[Math.floor(rows.length / 2)]}
                  </p>
                </div>
                <div className="best">
                  <span className="muted small">Top-ranked site</span>
                  <strong>top {topPct(best.rank)}%</strong>
                  <span className="muted small">of {lineName(line)} sites</span>
                </div>
              </div>
              {line === "Hct116" && (
                <p className="notice">One HCT116 sample (replicate 3, run 1) was used for training; HCT116 ranks may be optimistic.</p>
              )}
              {rows.every((r) => r.low) && (
                <p className="notice">All sites have depth ≤ {LOW_DEPTH}; ranks are low-confidence.</p>
              )}
              <Shortlist rows={rows} picked={picked} k={k} setK={setK} gene={{ ...gene, strand: entry.strand }} line={line}
                onOpen={setOpen} onRemove={toggle} />
              {structure && <GeneViewer structure={structure} strand={entry.strand} chrom={gene.chrom} rows={rows}
                picked={picked} onOpen={setOpen} focus={open?.pos} />}
              <SiteTable rows={rows} picked={picked} onToggle={toggle} onOpen={setOpen} line={line} chrom={gene.chrom} openSite={open?.site} />
            </>
          )}
        </section>
      )}

      {open && entry && (
        <SitePanel row={open} rows={rows} gene={gene} strand={entry.strand} line={line}
          onClose={() => setOpen(null)} onMove={setOpen} />
      )}
    </div>
  );
}
