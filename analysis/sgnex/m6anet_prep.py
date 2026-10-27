"""Build m6Anet's input for one SG-NEx line: the sites glori_check.py scores, every sample pooled.

m6Anet is the published comparator (analysis/m6anet/). To compare it with our
model against GLORI it has to score the same sites from the same reads. This
writes one m6Anet input directory per line:

- **Same sites:** only the (transcript, position) keys in `keys_<line>.csv.gz`,
  which are glori_check.py's evaluated sites (GLORI-covered genes, one isoform
  copy per genomic A). Scoring the line's other ~1.1M sites would cost hours
  and could not be graded.
- **Same reads:** each site's reads from every sample of the line, pooled the
  way score_sgnex.py pooled them for our model.
- **At most 50 reads per site**, sampled without replacement (seed 4262).
  m6Anet's site probability is a noisy-OR over 20 reads sampled with
  replacement, so reads past 50 change it very little and cost time linearly.
  Our model saw every read; `n_reads_total` keeps the uncapped count so the
  depth bands are the same as glori_check.py's.

The output format is make_m6anet_index.py's: plain-text data.json with a
trailing read id per read, and data.info with byte offsets.

    python analysis/sgnex/m6anet_prep.py --line A549 --keys data/m6anet_glori/keys_A549.csv.gz \
        --out data/m6anet_glori/A549_input
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "sgnex"))

from m6a.data import iter_sites  # noqa: E402
from score_sgnex import download, log, samples  # noqa: E402

CAP = 50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--line", required=True)
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="first N sites of each sample (smoke)")
    args = ap.parse_args()

    keys = pd.read_csv(args.keys)
    wanted = set(zip(keys.transcript_id, keys.transcript_position.astype(int)))
    names = samples()[args.line]
    log(f"{args.line}: {len(wanted):,} sites wanted, {len(names)} samples: {names}")

    reads: dict[tuple[str, int], list[np.ndarray]] = {}
    kmer: dict[tuple[str, int], str] = {}
    for name in names:
        path = download(name)
        hit = 0
        for site in iter_sites(path, limit=args.limit):
            # Same normalisation as score_sgnex.read: spike-ins dropped, versions stripped.
            if not site.transcript_id.startswith("ENST"):
                continue
            k = (site.transcript_id.split(".")[0], site.position)
            if k in wanted:
                reads.setdefault(k, []).append(site.reads)
                kmer[k] = site.kmer
                hit += 1
        log(f"  {name}: {hit:,} wanted sites present")

    rng = np.random.default_rng(4262)
    args.out.mkdir(parents=True, exist_ok=True)
    total = written_reads = 0
    # newline="" keeps offsets byte-exact on any platform (make_m6anet_index.py).
    with open(args.out / "data.json", "w", encoding="utf-8", newline="") as dj, \
            open(args.out / "data.info", "w", encoding="utf-8", newline="") as di, \
            open(args.out / "n_reads_total.csv", "w", encoding="utf-8", newline="") as nt:
        di.write("transcript_id,transcript_position,n_reads,start,end\n")
        nt.write("transcript_id,transcript_position,n_reads_total\n")
        offset = 0
        for (t, p), parts in reads.items():
            r = np.concatenate(parts)
            n = len(r)
            if n > CAP:
                r = r[np.sort(rng.choice(n, CAP, replace=False))]
            rows = [[float(x) for x in row] + [i] for i, row in enumerate(r.tolist())]
            payload = json.dumps({t: {str(p): {kmer[(t, p)]: rows}}}, separators=(",", ":"))
            dj.write(payload + "\n")
            di.write(f"{t},{p},{len(rows)},{offset},{offset + len(payload)}\n")
            nt.write(f"{t},{p},{n}\n")
            offset += len(payload) + 1
            total += 1
            written_reads += len(rows)
    log(f"{args.line}: wrote {total:,} sites ({len(wanted) - total:,} wanted sites absent), "
        f"{written_reads:,} reads after the {CAP}-read cap -> {args.out}")


if __name__ == "__main__":
    main()
