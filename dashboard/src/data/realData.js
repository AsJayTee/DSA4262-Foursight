// Local-first site adapter. Only explicitly selected, validated files are read.
// NEVER put restricted labels in public/ or commit real prediction files.
export const DATA_MODES = {
  demo: 'Synthetic demonstration',
  local: 'Local processed predictions'
};
const base = import.meta.env.BASE_URL;
const localUrl = gene => `${base}data/processed/${encodeURIComponent(gene)}.json`;
let cachedIndex = null;
export async function loadGeneIndex() {
  if (cachedIndex) return cachedIndex;
  const res = await fetch(`${base}data/processed/gene_index.json`);
  if (!res.ok) throw new Error(`gene_index.json not found (HTTP ${res.status}); run build_full.py`);
  const data = await res.json();
  if (data.schema_version !== 1 || !Array.isArray(data.genes)) throw new Error('Invalid gene index');
  cachedIndex = data;
  return data;
}
const REQUIRED = ['id', 'gene', 'chr', 'position', 'strand', 'transcript', 'motif', 'reads', 'raw', 'rank', 'line', 'status'];
export function validateRecord(s) {
  if (!s || REQUIRED.some(k => !(k in s))) return false;
  return typeof s.position === 'number' && Number.isFinite(s.position) && typeof s.reads === 'number' && Number.isFinite(s.reads) && s.reads >= 0 && /^[AGT][AG]AC[ACT]$/.test(s.motif) && ['+', '-'].includes(s.strand) && ['evaluated', 'unevaluated'].includes(s.status) && typeof s.id === 'string' && typeof s.gene === 'string' && typeof s.chr === 'string' && typeof s.transcript === 'string' && typeof s.line === 'string' && (s.status === 'evaluated' ? typeof s.raw === 'number' && Number.isFinite(s.raw) && typeof s.rank === 'number' && Number.isFinite(s.rank) : (s.raw == null || typeof s.raw === 'number' && Number.isFinite(s.raw)) && (s.rank == null || typeof s.rank === 'number' && Number.isFinite(s.rank)));
}
export async function loadLocalGene(gene) {
  // Generated files are explicitly opt-in, never fetched by default.
  const index = await loadGeneIndex().catch(() => null);
  const entry = index?.genes.find(x => x.symbol === gene || x.gene_id === gene);
  const url = entry?.file ? `${base}data/processed/${entry.file}` : localUrl(gene);
  const res = await fetch(url, {
    cache: 'no-store'
  });
  if (!res.ok) throw new Error(`No validated local dataset for ${gene} (HTTP ${res.status}). Generate the gene export in dashboard4/public/data/processed/ and use npm run dev.`);
  const doc = await res.json();
  if (doc.schema_version !== 1 || doc.gene !== gene || !Array.isArray(doc.transcripts) || !doc.transcripts.length || !doc.transcripts.every(t => t.transcript_id && Array.isArray(t.exons) && t.exons.length && t.exons.every(e => Array.isArray(e) && e.length === 2 && e[0] <= e[1])) || !Array.isArray(doc.sites) || !doc.sites.every(validateRecord)) throw new Error('Local gene data failed schema validation; see README.md (Local prediction JSON schema)');
  if (!doc.provenance || doc.provenance.kind !== 'processed-predictions') throw new Error('Missing processed-predictions provenance');
  return doc;
}
