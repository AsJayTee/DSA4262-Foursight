// The full site export for the Explore page (analysis/sgnex/export_site_shortlist.py and
// export_gene_structure.py), served at data/sites/ - from data/dashboard/ in development
// (vite.config.js), from a GitHub Release when deployed. Files are gzipped JSON, unpacked
// here in the browser.

const base = `${import.meta.env.BASE_URL}data/sites/`;
const cache = new Map();

async function bytesToJson(buf) {
  const head = new Uint8Array(buf, 0, 2);
  // Some servers unpack .gz on the way; only gunzip what is still gzip (magic 1f 8b).
  if (head[0] === 0x1f && head[1] === 0x8b) {
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  return JSON.parse(new TextDecoder().decode(buf));
}

export function load(path) {
  if (!cache.has(path)) {
    cache.set(path, fetch(base + path).then(async (r) => {
      if (!r.ok) throw new Error(`data/sites/${path}: HTTP ${r.status}`);
      return bytesToJson(await r.arrayBuffer());
    }).catch((e) => {
      cache.delete(path);   // let a later attempt retry
      throw e;
    }));
  }
  return cache.get(path);
}

const shardName = (n) => String(n).padStart(3, "0");
export const loadIndex = () => load("genes.json.gz");
export const loadMeta = () => Promise.all([load("meta.json"), load("trust.json")]);
export const loadGene = async (g) => (await load(`shards/${shardName(g.shard)}.json.gz`))[g.id];
export const loadStructure = async (g) => (await load(`structure/${shardName(g.shard)}.json.gz`))[g.id];

// Display names and what each SG-NEx line is.
export const LINES = {
  A549: { name: "A549", tissue: "lung cancer" },
  H9: { name: "H9", tissue: "embryonic stem cells" },
  HEYA8: { name: "HEYA8", tissue: "ovarian cancer" },
  Hct116: { name: "HCT116", tissue: "colon cancer" },
  HepG2: { name: "HepG2", tissue: "liver cancer" },
  K562: { name: "K562", tissue: "leukaemia" },
  MCF7: { name: "MCF7", tissue: "breast cancer" },
};
export const lineName = (l) => LINES[l]?.name || l;

export const LOW_DEPTH = 2;         // molecules: at or below this, a site is flagged as low depth
export const HOLDS_TOP = 5;         // "holds up" in a batch = in the gene's top 5 there

const pct = (v) => (v == null || v < 0 ? null : v / 9999);

function trustOf(kmer, reads, trust, cutoffs) {
  const motif = kmer.slice(1, 6);
  const band = trust.read_bands.find(([lo, hi]) => reads >= lo && reads <= hi)?.[2] ?? "full";
  const value = trust.pair_accuracy[motif]?.[band] ?? null;
  const label = value == null ? "n.d." : cutoffs.find((c) => value >= c.min).label;
  return { motif, band, value, label };
}

// One row per site of `gene` measured in `line`, ranked within the gene.
export function rowsFor(gene, line, meta, trust, cutoffs) {
  const L = gene.lines[line];
  if (!L) return [];
  const regions = meta.regions;
  const batchNames = meta.batches[line] || [];
  const rows = L.i.map((s, j) => ({
    site: s,
    pos: gene.sites.pos[s],
    kmer: gene.sites.kmer[s],
    region: regions[gene.sites.region[s]],
    transcripts: gene.sites.tx[s].map((t) => gene.tx[t]),
    rank: pct(L.r[j]),
    own: pct(L.o[j]),
    reads: L.n[j],
    batches: batchNames.map((name, k) => ({ name, rank: pct(L.b[k][j]), reads: L.bn[k][j] })),
  }));
  rows.sort((a, b) => b.rank - a.rank);
  rows.forEach((r, i) => (r.geneRank = i + 1));
  // Position within the gene in each batch of cells, among the sites measured there.
  batchNames.forEach((_, k) => {
    rows.filter((r) => r.batches[k].rank != null)
      .sort((a, b) => b.batches[k].rank - a.batches[k].rank)
      .forEach((r, i) => (r.batches[k].position = i + 1));
  });
  // The textbook rule: order by how often the site's DNA pattern is modified.
  [...rows].sort((a, b) => (trust.motif_rate[b.kmer.slice(1, 6)] ?? 0) - (trust.motif_rate[a.kmer.slice(1, 6)] ?? 0))
    .forEach((r, i) => (r.lettersRank = i + 1));
  rows.forEach((r) => {
    r.trust = trustOf(r.kmer, r.reads, trust, cutoffs);
    r.low = r.reads <= LOW_DEPTH;
    const measured = r.batches.filter((b) => b.rank != null);
    r.holds = { held: measured.filter((b) => b.position <= HOLDS_TOP).length, measured: measured.length, of: r.batches.length };
    r.elsewhere = Object.entries(gene.lines).filter(([l]) => l !== line).map(([l, M]) => {
      const j = M.i.indexOf(r.site);
      return j < 0 ? { line: l, rank: null, reads: 0 } : { line: l, rank: pct(M.r[j]), reads: M.n[j] };
    });
  });
  return rows;
}

// Gene search: exact name, then name prefix, then ID prefix, then name contains.
export function searchGenes(index, query, limit = 8) {
  const q = query.trim().toUpperCase();
  if (!q) return [];
  const out = [], seen = new Set();
  const take = (pred) => {
    for (const g of index) {
      if (out.length >= limit) return;
      if (!seen.has(g.id) && pred(g)) { seen.add(g.id); out.push(g); }
    }
  };
  take((g) => g.name.toUpperCase() === q || g.id === q);
  take((g) => g.name.toUpperCase().startsWith(q));
  take((g) => g.id.startsWith(q));
  if (q.length >= 3) take((g) => g.name.toUpperCase().includes(q));
  return out;
}

export function toCsv(rows, gene, line) {
  const cols = ["gene", "gene_id", "cell_line", "chromosome", "position_grch38", "strand", "sequence", "region",
    "depth", "rank_in_gene", "percentile_in_cell_line", "top_percent", "replicates_in_gene_top5", "reliability", "transcripts"];
  const q = (v) => `"${String(v ?? "").replaceAll('"', '""')}"`;
  const lines = rows.map((r) => [gene.name, gene.id, lineName(line), gene.chrom, r.pos, gene.strand,
    r.kmer.replace(/T/g, "U"), r.region, r.reads, r.geneRank, r.rank.toFixed(4), ((1 - r.rank) * 100).toFixed(2),
    `${r.holds.held}/${r.holds.measured}`, r.trust.label, r.transcripts.join(" ")].map(q).join(","));
  return [cols.join(","), ...lines].join("\r\n");
}
