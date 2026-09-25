"""Turn the course's `dataset0.json.gz` into the pair of files m6Anet wants.

m6Anet inference needs a directory holding **two** files, and the course gave us
one of them under a confusingly similar name (analysis/m6anet/README.md, Trap 1):

| m6Anet wants | we have |
|---|---|
| `data.json`, signal features | `dataset0.json.gz` - **almost** this, see below |
| `data.info`, a byte-offset index | nothing. `data.info.labelled` is the m6ACE **labels** |

## The two differences from our file, both found by reading m6anet's source

**1. It seeks, so the file cannot be gzipped.** `m6anet/utils/data_utils.py`
does `f.seek(start, 0); f.read(end - start)` and parses that slice as JSON, so
`data.json` has to be plain text and `data.info` has to carry byte offsets into
it. 625 MB uncompressed.

**2. Every read needs a trailing id column.** The same file does

    read_ids, features = features[:, -1], features[:, self.indices]

so it reads the **last** column of each read as an identifier and the rest as
signal. Our rows are nine floats with no id. This appends a per-site read index.

What does *not* need changing is the feature layout. m6Anet's dataprep builds
its triple as `['dwell_time', 'norm_std', 'norm_mean']` per position, laid out
position-major - which is exactly `m6a.data.READ_FEATURE_NAMES`
(`dwell_m1, sd_m1, mean_m1, dwell_0, ...`). The nine columns map one to one, in
order, with no rescaling. That is presumably why the course handed us this file.

## Offsets are bytes, and on Windows that needs care

The reader opens in **text** mode and seeks to a byte position, which only works
while byte offsets and character offsets agree. They do here - the JSON is
ASCII - but only if the file is written with `\\n` line endings. Python on
Windows would translate those to `\\r\\n` and silently shift every offset after
the first line, so the output is opened with `newline=""`.

Usage:

    python analysis/m6anet/make_m6anet_index.py --out data/m6anet/input
    python analysis/m6anet/make_m6anet_index.py --out /tmp/small --limit 200
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from m6a.data import resolve_data_dir  # noqa: E402

N_FEATURES = 9


def build(source: Path, out_dir: Path, limit: int | None = None) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "data.json"
    info_path = out_dir / "data.info"

    started = time.time()
    sites = reads_total = 0

    # newline="" so Python does not turn \n into \r\n and invalidate every
    # offset after the first line. See the module docstring.
    with gzip.open(source, "rt") as source_handle, \
            data_path.open("w", encoding="utf-8", newline="") as data_handle, \
            info_path.open("w", encoding="utf-8", newline="") as info_handle:

        info_handle.write("transcript_id,transcript_position,n_reads,start,end\n")
        offset = 0

        for line_number, line in enumerate(source_handle):
            if limit is not None and sites >= limit:
                break
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            transcript = next(iter(record))
            position = next(iter(record[transcript]))
            kmer = next(iter(record[transcript][position]))
            reads = record[transcript][position][kmer]

            if not reads or len(reads[0]) != N_FEATURES:
                raise ValueError(
                    f"{source} line {line_number + 1}: expected {N_FEATURES} "
                    f"features per read, got {len(reads[0]) if reads else 0}. "
                    "This script converts the course's dataset0.json.gz; point "
                    "it at that file."
                )

            # The trailing read id m6anet slices off as features[:, -1]. It is
            # only ever used as a label in the output, so a per-site index is
            # enough and keeps the file smaller than a uuid would.
            with_ids = [row + [index] for index, row in enumerate(reads)]
            payload = json.dumps(
                {transcript: {position: {kmer: with_ids}}}, separators=(",", ":")
            )
            written = data_handle.write(payload + "\n")
            assert written == len(payload) + 1  # ascii, so chars == bytes

            info_handle.write(
                f"{transcript},{position},{len(reads)},{offset},{offset + len(payload)}\n"
            )
            offset += len(payload) + 1
            sites += 1
            reads_total += len(reads)

            if sites % 20000 == 0:
                print(f"  {sites:,} sites, {reads_total:,} reads ...", flush=True)

    megabytes = data_path.stat().st_size / (1 << 20)
    print(
        f"{sites:,} sites and {reads_total:,} reads -> {out_dir} "
        f"({megabytes:,.0f} MB) in {time.time() - started:.0f}s"
    )
    return sites, reads_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None,
                        help="default: $M6A_DATA_DIR/dataset0.json.gz (data/raw if unset)")
    parser.add_argument("--out", required=True, help="directory to write data.json and data.info into")
    parser.add_argument("--limit", type=int, default=None, help="first N sites only, for a smoke test")
    args = parser.parse_args()
    source = Path(args.source) if args.source else resolve_data_dir() / "dataset0.json.gz"
    build(source, Path(args.out), args.limit)


if __name__ == "__main__":
    main()
