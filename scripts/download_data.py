#!/usr/bin/env python
"""Fetch data from the R2 bucket into data/raw/.

    python scripts/download_data.py                 # the course training set
    python scripts/download_data.py --set sgnex     # SG-NEx data for Task 2
    python scripts/download_data.py --list          # show what is in the bucket

Every object is checked against manifest.json, and anything already present with
a matching checksum is skipped. Re-running this is therefore free, and a
truncated download is caught here rather than surfacing an hour later as a
baffling JSON parse error.

Needs the train extra: pip install -e .[train]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a import r2
from m6a.data import resolve_data_dir
from m6a.env import load_env

PREFIXES = {
    "course": "course/",
    "sgnex": "sgnex/",
    "all": "",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--set", dest="which", default="course", choices=sorted(PREFIXES))
    ap.add_argument("--dest", default=None, help="Destination dir (default data/raw)")
    ap.add_argument("--list", action="store_true", help="List bucket contents and exit")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    load_env(ROOT / ".env")

    try:
        client = r2.client()
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n\nRun `python scripts/doctor.py` to see what is missing.")

    if args.list:
        for key in r2.list_keys(s3=client):
            print(key)
        return 0

    dest_root = Path(args.dest) if args.dest else resolve_data_dir()
    prefix = PREFIXES[args.which]

    try:
        manifest = r2.load_manifest(client)
    except Exception as exc:  # noqa: BLE001
        print(f"  no usable manifest.json ({type(exc).__name__}) - downloading without checksums")
        manifest = {}

    keys = [key for key in r2.list_keys(prefix, s3=client) if key != r2.MANIFEST_KEY]
    if not keys:
        raise SystemExit(f"Nothing under prefix {prefix!r} in bucket {r2.bucket()}.")

    print(f"{len(keys)} object(s) under {prefix!r} -> {dest_root}/")
    downloaded = skipped = 0
    for key in keys:
        # course/dataset0.json.gz lands at data/raw/dataset0.json.gz;
        # sgnex/A549/x.json.gz keeps its sub-path.
        relative = key[len(prefix):] if prefix else key
        dest = dest_root / relative
        outcome = r2.download(key, dest, expected=manifest.get(key), s3=client)
        size_mb = dest.stat().st_size / 1e6
        print(f"  [{outcome:>10}] {relative}  ({size_mb:.1f} MB)")
        if outcome == "downloaded":
            downloaded += 1
        else:
            skipped += 1

    print(f"\n{downloaded} downloaded, {skipped} already present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
