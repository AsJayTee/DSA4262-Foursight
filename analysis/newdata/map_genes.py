"""Give every data1 transcript a gene id, so data0 and data1 can share one
gene-grouped split without leakage.

    python analysis/newdata/map_genes.py

data1's data.info has no gene_id column. Leakage is a gene-level question: two
transcripts of one gene share sequence (AGENTS.md section 3), so a data1
"unseen" transcript whose gene is in the training data is not unseen.

Two sources, in order:

1. dataset0's own labels file (transcript -> gene), which covers most of data1.
2. Ensembl's public REST API (POST /lookup/id, 1,000 ids per request) for the
   rest. Only transcript ids are sent - public Ensembl identifiers, no signal
   and no labels.

Writes <data dir>/data1/transcript_genes.csv (transcript_id, gene_id, source)
next to the data, where .gitignore keeps it out of the public repo. Anything
Ensembl cannot resolve (e.g. an id retired in the current release) is written
with an empty gene_id and reported, never silently dropped.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a.data import resolve_data_dir  # noqa: E402

ENSEMBL = "https://rest.ensembl.org/lookup/id"
BATCH = 1000


def lookup(ids: list[str]) -> dict[str, str | None]:
    """transcript id -> gene id via Ensembl, None where it does not resolve."""
    out: dict[str, str | None] = {}
    for start in range(0, len(ids), BATCH):
        chunk = ids[start:start + BATCH]
        request = urllib.request.Request(
            ENSEMBL, data=json.dumps({"ids": chunk}).encode(),
            headers={"Content-Type": "application/json", "Accept": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.load(response)
                break
            except Exception as exc:  # noqa: BLE001 - network: retry, then fail loudly
                if attempt == 3:
                    raise SystemExit(f"Ensembl lookup failed after 4 attempts: {exc!r}")
                time.sleep(5 * (attempt + 1))
        for tid in chunk:
            record = body.get(tid)
            out[tid] = record.get("Parent") if record else None
        print(f"  looked up {min(start + BATCH, len(ids)):,} / {len(ids):,}", flush=True)
    return out


def main() -> None:
    data_dir = resolve_data_dir()
    labels = pd.read_csv(data_dir / "data.info.labelled")
    data1 = pd.read_csv(data_dir / "data1" / "data.info")

    per_tx = labels.groupby("transcript_id")["gene_id"].agg(lambda g: sorted(set(g)))
    if (per_tx.map(len) > 1).any():
        raise SystemExit("dataset0 maps some transcript to more than one gene - check the labels file.")
    known = per_tx.map(lambda g: g[0])

    transcripts = sorted(data1["transcript_id"].unique())
    rows = [{"transcript_id": t, "gene_id": known[t], "source": "dataset0"}
            for t in transcripts if t in known.index]
    missing = [t for t in transcripts if t not in known.index]
    print(f"data1: {len(transcripts):,} transcripts; {len(rows):,} mapped from dataset0, "
          f"{len(missing):,} to look up in Ensembl")

    resolved = lookup(missing) if missing else {}
    rows += [{"transcript_id": t, "gene_id": g or "", "source": "ensembl" if g else "unresolved"}
             for t, g in resolved.items()]

    # Cross-check a sample of the dataset0 mappings against Ensembl: if the
    # course's annotation and Ensembl's disagree, the two sources are not
    # interchangeable and that needs to be known before trusting either.
    sample = [r["transcript_id"] for r in rows if r["source"] == "dataset0"][:500]
    check = lookup(sample)
    agree = sum(check[t] == known[t] for t in sample if check[t])
    checked = sum(1 for t in sample if check[t])
    print(f"cross-check: Ensembl agrees with dataset0 on {agree:,} of {checked:,} sampled transcripts")

    table = pd.DataFrame(rows).sort_values("transcript_id")
    out = data_dir / "data1" / "transcript_genes.csv"
    table.to_csv(out, index=False)
    unresolved = (table["source"] == "unresolved").sum()
    print(f"wrote {out}: {len(table):,} transcripts, {table['gene_id'].replace('', pd.NA).nunique():,} "
          f"genes, {unresolved:,} unresolved")


if __name__ == "__main__":
    main()
