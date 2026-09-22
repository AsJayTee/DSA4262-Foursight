#!/usr/bin/env python
"""What is in SG-NEx, and how deep is it - straight from the public bucket.

    python analysis/sgnex/survey.py catalogue
    python analysis/sgnex/survey.py depth SGNex_A549_directRNA_replicate6_run1
    python analysis/sgnex/survey.py depth SGNex_Hct116_directRNA_replicate3_run4
           --labels data0/data.info.labelled

`s3://sg-nex-data` is public and needs no credentials - the client below is
explicitly unsigned, so it works on a machine that has never seen an AWS key.

**Why this exists.** Every SG-NEx number in GAPS.md used to come from an ad-hoc
crawl that was not in the repo, so nobody could re-derive or challenge it. This
script regenerates all of them. It is deliberately not part of the harness: it
talks to the network, it is read-only, and nothing imports it.

**It reads `data.readcount`, not `data.json`.** The readcount file is one line
per site - `transcript_id,transcript_position,n_reads` - and about 34 MB against
the 2 GB of signal data beside it. Read depth is the single most important
variable in this project (docs/data.md#read-depth) and this is the cheapest
honest way to measure it: well under a minute per sample rather than a 2 GB
download.

Needs boto3, which is in the `train` extra: pip install -e '.[train]'
"""

from __future__ import annotations

import argparse
import collections
import io
from pathlib import Path

import numpy as np
import pandas as pd

BUCKET = "sg-nex-data"
M6ANET_PREFIX = "data/processed_data/m6Anet/"
# fastq and fast5 hold the same sample set; fastq is listed because it is the
# one a re-basecalling route would start from.
RAW_PREFIX = "data/sequencing_data_ont/fastq/"

# Every training site has at least this many reads, because the course filtered
# it there - it is not a property of nanopore data. See docs/data.md#read-depth.
TRAINING_DEPTH_FLOOR = 20


def client():
    """An unsigned S3 client. The bucket is public; credentials would be refused."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        raise SystemExit("boto3 is not installed. Run:  pip install -e '.[train]'") from None

    return boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, connect_timeout=15, read_timeout=120),
    )


def sample_dirs(s3, prefix: str) -> list[str]:
    """The immediate subdirectory names under a prefix."""
    names: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix, Delimiter="/"
    ):
        names += [
            item["Prefix"].rstrip("/").split("/")[-1]
            for item in page.get("CommonPrefixes", [])
        ]
    return sorted(names)


def cell_line(sample: str) -> str:
    """`SGNex_A549_directRNA_replicate6_run1` -> `A549`."""
    parts = sample.split("_")
    return parts[1] if len(parts) > 1 else sample


def catalogue(args: argparse.Namespace) -> int:
    """What SG-NEx holds, in raw form and in the m6Anet-processed subset.

    The distinction matters and is easy to miss: the processed subset is seven
    cell lines, the raw data is fourteen. A cell line that exists only in the raw
    half needs nanopolish eventalign and m6Anet dataprep before anything in this
    repo can read it.
    """
    s3 = client()

    processed = sample_dirs(s3, M6ANET_PREFIX)
    raw = [s for s in sample_dirs(s3, RAW_PREFIX) if "directRNA" in s]

    processed_lines = collections.Counter(cell_line(s) for s in processed)
    raw_lines = collections.Counter(cell_line(s) for s in raw)

    print(f"m6Anet-processed:  {len(processed)} samples, {len(processed_lines)} cell lines")
    print(f"raw directRNA:     {len(raw)} samples, {len(raw_lines)} cell lines")
    print()
    print(f"{'cell line':<16}{'raw':>6}{'m6Anet':>9}   ready to predict on?")
    print("-" * 58)
    for line in sorted(raw_lines, key=lambda k: (-raw_lines[k], k)):
        ready = "yes" if processed_lines.get(line) else "no - needs eventalign"
        print(f"{line:<16}{raw_lines[line]:>6}{processed_lines.get(line, 0):>9}   {ready}")

    only_processed = set(processed_lines) - set(raw_lines)
    if only_processed:
        print(f"\nprocessed but not matched in the raw listing: {sorted(only_processed)}")
    print(
        "\n'ready to predict on' means scripts/predict.py runs on that sample's "
        "data.json\nunmodified. The rest would need nanopolish eventalign + m6Anet "
        "dataprep first."
    )
    if args.samples:
        print("\nm6Anet sample directories:")
        for sample in processed:
            print(f"  {sample}")
    return 0


def read_readcount(s3, sample: str) -> pd.DataFrame:
    """A sample's `data.readcount` as a frame, fetched whole.

    ~34 MB, so it is pulled in one GET rather than streamed in ranges: the
    parsing dominates either way and one request is far quicker.
    """
    key = f"{M6ANET_PREFIX}{sample}/data.readcount"
    try:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Could not read s3://{BUCKET}/{key} ({type(exc).__name__}: {exc}).\n"
            "Check the sample name with:\n"
            "  python analysis/sgnex/survey.py catalogue --samples"
        ) from None
    return pd.read_csv(io.BytesIO(body))


def describe(n_reads: np.ndarray, label: str) -> dict:
    """The depth summary GAPS.md quotes, for one set of sites."""
    n_reads = np.asarray(n_reads)
    if n_reads.size == 0:
        return {"what": label, "sites": 0}
    row = {
        "what": label,
        "sites": int(n_reads.size),
        "reads": int(n_reads.sum()),
        "min": int(n_reads.min()),
        "p25": float(np.percentile(n_reads, 25)),
        "median": float(np.median(n_reads)),
        "p75": float(np.percentile(n_reads, 75)),
        "max": int(n_reads.max()),
    }
    for floor in (2, 5, TRAINING_DEPTH_FLOOR):
        row[f">={floor}"] = f"{100 * float((n_reads >= floor).mean()):.1f}%"
    row[f"<{TRAINING_DEPTH_FLOOR}"] = (
        f"{100 * float((n_reads < TRAINING_DEPTH_FLOOR).mean()):.1f}%"
    )
    return row


def depth(args: argparse.Namespace) -> int:
    """The read-depth distribution of one sample, optionally against our sites.

    With `--labels` it also answers the question that decides whether another run
    of our own cell line could supply low-depth copies of our labelled sites: how
    many of them appear in this sample, and how deep are they there?
    """
    s3 = client()
    print(f"[{args.sample}] reading data.readcount from s3://{BUCKET}")
    frame = read_readcount(s3, args.sample)
    rows = [describe(frame["n_reads"].to_numpy(), "all sites")]

    if args.labels:
        labels_path = Path(args.labels)
        if not labels_path.exists():
            raise SystemExit(
                f"No labels file at {labels_path}. Point --labels at a "
                "data.info.labelled (see docs/setup.md), or leave it off."
            )
        labels = pd.read_csv(labels_path)
        ours = set(zip(labels["transcript_id"], labels["transcript_position"]))
        here = frame.set_index(["transcript_id", "transcript_position"])
        overlap = here.loc[here.index.isin(ours)]

        print(
            f"  {len(ours):,} labelled sites; {len(overlap):,} of them appear in "
            f"this sample ({100 * len(overlap) / len(ours):.1f}%)"
        )
        rows.append(describe(overlap["n_reads"].to_numpy(), "our labelled sites"))

    print()
    print(pd.DataFrame(rows).set_index("what").to_string(
        float_format=lambda v: f"{v:.1f}", na_rep="-"
    ))
    print(
        "\nDepth is the number of distinct RNA molecules measured at one site, not\n"
        "repeated readings of one molecule - docs/data.md#read-depth. Every site in\n"
        f"our training set has >= {TRAINING_DEPTH_FLOOR} reads, so the "
        f"'<{TRAINING_DEPTH_FLOOR}' column is the fraction\nof this sample the model "
        "has never seen anything like."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="mode", required=True)

    cat = sub.add_parser("catalogue", help="What cell lines and samples exist")
    cat.add_argument(
        "--samples", action="store_true", help="Also list every m6Anet sample directory"
    )
    cat.set_defaults(func=catalogue)

    dep = sub.add_parser("depth", help="Read-depth distribution of one m6Anet sample")
    dep.add_argument("sample", help="e.g. SGNex_A549_directRNA_replicate6_run1")
    dep.add_argument(
        "--labels",
        default=None,
        help="Our data.info.labelled, to intersect with our labelled sites",
    )
    dep.set_defaults(func=depth)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
