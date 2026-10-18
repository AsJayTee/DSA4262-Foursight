import { getDemoSites } from './mockSites';
// Swap this adapter when local SG-NEx CSV outputs have been mapped to genomic coordinates.
// Never ship restricted labels or private raw files inside dashboard/public/.
export async function loadSummary() {
  const response = await fetch(`${import.meta.env.BASE_URL}data/summary.json`);
  if (!response.ok) throw new Error(`Could not load repository summary.json (HTTP ${response.status}). Copy the original file into dashboard/public/data/summary.json.`);
  return response.json();
}
export async function loadSites({
  gene,
  line
}) {
  return getDemoSites(gene, line);
}
;
