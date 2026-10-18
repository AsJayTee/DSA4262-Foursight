#!/usr/bin/env python3
"""Create small, local-only dashboard pilot exports from pooled SG-NEx predictions.

Uses standard library only. Outputs are NOT automatically safe to publish.
Genomic mapping assumes transcript_position is ZERO-based; treat coordinates
as provisional until checked against the matching transcriptome FASTA.
"""
import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import re

LINES = ("A549", "H9", "Hct116", "HepG2", "HEYA8", "K562", "MCF7")
ATTR = re.compile(r'(\w+)\s+"([^"]*)"')

def parse_gtf(path, gene_names):
    transcripts = {}
    genes = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9 or cols[2] not in ("transcript", "exon"):
                continue
            a = dict(ATTR.findall(cols[8]))
            name = a.get("gene_name", "")
            if name.upper() not in gene_names:
                continue
            tid = a.get("transcript_id", "").split(".")[0]
            if not tid:
                continue
            entry = transcripts.setdefault(tid, dict(
                transcript_id=tid, gene_id=a.get("gene_id", ""),
                gene_name=name, chr=cols[0] if cols[0].startswith("chr") else "chr"+cols[0],
                strand=cols[6], exons=[]))
            if cols[2] == "exon":
                entry["exons"].append([int(cols[3]), int(cols[4])])
            genes[name.upper()] = name
    for entry in transcripts.values():
        entry["exons"].sort(key=lambda e:e[0], reverse=entry["strand"]=="-")
    return transcripts, genes

def map_position(tx, pos, base):
    offset = pos - base
    if offset < 0:
        return None
    for start, end in tx["exons"]:
        length = end - start + 1
        if offset < length:
            return (start + offset) if tx["strand"] == "+" else (end - offset)
        offset -= length
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, type=Path,
                   help="Directory containing annotation/ and sgnex_scores/")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--genes", nargs="+", default=["PTBP1","HSPA5","POLDIP2","METTL3"])
    p.add_argument("--lines", nargs="+", choices=LINES, default=list(LINES))
    p.add_argument("--position-base", type=int, choices=(0,1), default=0)
    p.add_argument("--max-sites-per-gene", type=int, default=50000)
    args=p.parse_args()
    if args.max_sites_per_gene < 1:
        p.error("--max-sites-per-gene must be positive")
    wanted={g.upper() for g in args.genes}
    gtf=args.input_dir/"annotation"/"gtf_91.gtf.gz"
    if not gtf.exists():
        p.error(f"Missing {gtf}")
    transcripts, names = parse_gtf(gtf, wanted)
    if not transcripts:
        p.error("None of the requested genes were found in GTF")
    grouped=defaultdict(list)
    counts=Counter()
    skipped=Counter()
    for line in args.lines:
        path=args.input_dir/"sgnex_scores"/f"{line}_pooled.csv.gz"
        if not path.exists():
            p.error(f"Missing {path}")
        with gzip.open(path,"rt",newline="") as fh:
            for row in csv.DictReader(fh):
                tid=row["transcript_id"].split(".")[0]
                tx=transcripts.get(tid)
                if tx is None:
                    continue
                gene=tx["gene_name"]
                counts[gene]+=1
                if len(grouped[gene]) >= args.max_sites_per_gene:
                    skipped[gene]+=1
                    continue
                try:
                    tpos=int(row["transcript_position"])
                    pos=map_position(tx,tpos,args.position_base)
                    if pos is None:
                        skipped[gene]+=1
                        continue
                    kmer=row["kmer"].upper()
                    motif=kmer[1:6] if len(kmer)==7 else kmer
                    if len(motif)!=5:
                        skipped[gene]+=1
                        continue
                    grouped[gene].append(dict(
                        id=f"{line}:{tid}:{tpos}", gene=gene, chr=tx["chr"],
                        position=pos, strand=tx["strand"], transcript=tid,
                        transcriptPosition=tpos, kmer=kmer, motif=motif,
                        reads=int(row["n_reads"]), raw=float(row["score_raw"]),
                        rank=float(row["score_rank"]), line=line,
                        status="evaluated"))
                except (ValueError,KeyError):
                    skipped[gene]+=1
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={}
    for gene in sorted(names.values()):
        txs=[tx for tx in transcripts.values() if tx["gene_name"]==gene]
        doc=dict(schema_version=1,
                 provenance=dict(kind="processed-predictions",
                                 source="team SG-NEx pooled predictions",
                                 annotation="Ensembl 91",
                                 coordinate_mapping="PROVISIONAL: transcript positions interpreted as %s-based"%args.position_base),
                 gene=gene, transcripts=txs, sites=grouped[gene])
        dest=args.output_dir/f"{gene}.json"
        dest.write_text(json.dumps(doc,separators=(",",":")),encoding="utf-8")
        report[gene]=dict(transcripts=len(txs), sites_written=len(grouped[gene]),
                          sites_matching_gene=counts[gene], skipped=skipped[gene],
                          bytes=dest.stat().st_size)
    (args.output_dir/"pilot_report.json").write_text(json.dumps(
        dict(warning="Do not use mapped genomic positions as verified until FASTA validation",
             position_base=args.position_base, genes=report),indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    print("CAUTION: coordinates are provisional; see README before integrating into React.")

if __name__=="__main__":
    main()
