"""Each gene's exon structure, for the Explore page's gene map.

    python analysis/sgnex/export_site_shortlist.py     # first: the genes and their sites
    python analysis/sgnex/export_gene_structure.py

From the Ensembl 91 annotation, for every gene in the site export:
  - its exons merged over all annotated transcripts (the gene map's boxes),
  - the span of its coding region (thicker on the map), and
  - the exons of each transcript that carries one of its sites (for distances to exon
    junctions, e.g. a "where along the gene" view on a model stats page).
Genome coordinates (GRCh38), 1-based inclusive, as in the GTF.

Written beside the site export, in the same 256 shards, so a gene's structure is in the
file with the same number as its sites: data/dashboard/structure/NNN.json.gz.
Annotation only: no scores, labels or reads.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GTF = ROOT / "data" / "annotation" / "gtf_91.gtf.gz"
EXPORT = ROOT / "data" / "dashboard"
OUT = EXPORT / "structure"


def merge(intervals: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main() -> None:
    index = json.load(gzip.open(EXPORT / "genes.json.gz"))
    shard_of = {g["id"]: g["shard"] for g in index}
    # Transcripts that carry a site, per gene (from the site export itself).
    carried: dict[str, set[str]] = {}
    for f in sorted((EXPORT / "shards").glob("*.json.gz")):
        for gid, entry in json.load(gzip.open(f)).items():
            carried[gid] = set(entry["tx"])

    exons: dict[str, dict[str, list]] = {}
    cds: dict[str, list[int]] = {}
    with gzip.open(GTF, "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            p = line.split("\t", 8)
            if p[2] not in ("exon", "CDS"):
                continue
            gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
            if gid not in shard_of:
                continue
            tid = re.search(r'transcript_id "([^"]+)"', p[8]).group(1)
            s, e = int(p[3]), int(p[4])
            if p[2] == "exon":
                exons.setdefault(gid, {}).setdefault(tid, []).append([s, e])
            else:
                c = cds.setdefault(gid, [s, e])
                c[0], c[1] = min(c[0], s), max(c[1], e)

    shards: dict[int, dict] = {}
    for gid, by_tx in exons.items():
        all_exons = [x for xs in by_tx.values() for x in xs]
        merged = merge(all_exons)
        shards.setdefault(shard_of[gid], {})[gid] = {
            "span": [merged[0][0], merged[-1][1]],
            "exons": merged,
            "cds": cds.get(gid),
            "tx": {t: sorted(xs) for t, xs in by_tx.items() if t in carried.get(gid, ())},
        }
    OUT.mkdir(exist_ok=True)
    for shard, content in shards.items():
        with gzip.open(OUT / f"{shard:03d}.json.gz", "wt", compresslevel=9) as fh:
            json.dump(content, fh, separators=(",", ":"))
    size = sum(p.stat().st_size for p in OUT.glob("*.json.gz"))
    print(f"{sum(len(c) for c in shards.values()):,} genes -> {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
