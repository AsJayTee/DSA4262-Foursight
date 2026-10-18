// SYNTHETIC DEMO ONLY. Coordinates, scores, read depths and transcripts below are invented.
// No model predictions or restricted labels are included.
export const LINES = ['A549', 'H9', 'HEYA8', 'Hct116', 'HepG2', 'K562', 'MCF7'];
export const GENES = [{
  name: 'PTBP1',
  chr: 'chr19',
  start: 792000,
  end: 806000,
  strand: '-',
  tx: 'ENST_DEMO_PTBP1',
  exons: [[792450, 793480], [795100, 796050], [797300, 798260], [800200, 801260], [803200, 805200]]
}, {
  name: 'HSPA5',
  chr: 'chr9',
  start: 127990000,
  end: 128004000,
  strand: '-',
  tx: 'ENST_DEMO_HSPA5',
  exons: [[127990500, 127991700], [127993100, 127994350], [127996100, 127997400], [128000000, 128001400], [128002400, 128003400]]
}, {
  name: 'POLDIP2',
  chr: 'chr17',
  start: 28350000,
  end: 28364000,
  strand: '-',
  tx: 'ENST_DEMO_POLDIP2',
  exons: [[28350300, 28351500], [28352700, 28353900], [28355300, 28356500], [28358100, 28359200], [28360800, 28363400]]
}, {
  name: 'METTL3',
  chr: 'chr14',
  start: 21800000,
  end: 21814000,
  strand: '+',
  tx: 'ENST_DEMO_METTL3',
  exons: [[21800300, 21801600], [21803300, 21804600], [21806000, 21807500], [21808900, 21810100], [21811900, 21813600]]
}];
export const MOTIFS = ['AAACA', 'AAACC', 'AAACT', 'AGACA', 'AGACC', 'AGACT', 'GAACA', 'GAACC', 'GAACT', 'GGACA', 'GGACC', 'GGACT', 'TAACA', 'TAACC', 'TAACT', 'TGACA', 'TGACC', 'TGACT'];
const seeded = n => {
  const x = Math.sin(n * 127.1 + 78.233) * 43758.5453;
  return x - Math.floor(x);
};
export function getDemoSites(geneName, line) {
  const gene = GENES.find(g => g.name === geneName) || GENES[0];
  const seed = LINES.indexOf(line) + 1;
  return Array.from({
    length: 110
  }, (_, i) => {
    const p = gene.start + 250 + Math.floor(seeded(i * 11 + seed * .2 + GENES.indexOf(gene) * 9) * (gene.end - gene.start - 500));
    const motif = MOTIFS[Math.floor(seeded(i * 19 + 7) * MOTIFS.length)];
    const reads = 1 + Math.floor(Math.pow(seeded(i * 5 + seed * 13), 2.1) * 110);
    const raw = Number((-1.9 + seeded(i * 3 + seed * 2 + 3) * 4.7 + Math.log1p(reads) * .17).toFixed(2));
    return {
      id: `${gene.name}-${line}-${i}`,
      gene: gene.name,
      chr: gene.chr,
      position: p,
      strand: gene.strand,
      transcript: gene.tx,
      motif,
      reads,
      raw,
      rank: Number(seeded(i * 17 + seed * 3).toFixed(3)),
      line,
      source: 'synthetic'
    };
  }).sort((a, b) => a.position - b.position);
}
export const fmt = n => Math.round(n).toLocaleString('en-US');
