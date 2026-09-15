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

# The course training data sits at the root of the bucket; everything added
# later (SG-NEx, evaluation releases) goes under its own prefix. So "course"
# means "objects not in any folder" rather than a prefix of its own.
PREFIXES = {
    "course": None,  # root-level objects only
    "sgnex": "sgnex/",
    "all": "",
}


def select_keys(which: str, client) -> tuple[list[str], str]:
    """Keys to fetch, and the prefix to strip when building local paths."""
    prefix = PREFIXES[which]
    if prefix is None:
        keys = [k for k in r2.list_keys("", s3=client) if "/" not in k]
        return keys, ""
    keys = r2.list_keys(prefix, s3=client)
    return keys, prefix


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

    try:
        manifest = r2.load_manifest(client)
    except Exception:  # noqa: BLE001
        # No manifest yet: downloads still work, they just are not checksummed.
        print("  no manifest.json in the bucket - downloading without checksum verification")
        manifest = {}

    keys, prefix = select_keys(args.which, client)
    keys = [key for key in keys if key != r2.MANIFEST_KEY]
    if not keys:
        raise SystemExit(
            f"No objects found for --set {args.which} in bucket {r2.bucket()}.\n"
            "Run `python scripts/download_data.py --list` to see what is there."
        )

    print(f"{len(keys)} object(s) for --set {args.which} -> {dest_root}/")
    downloaded = skipped = 0
    for key in keys:
        # Root objects land directly in data/raw/; sgnex/A549/x.json.gz keeps
        # its sub-path so cell lines stay separated on disk.
        relative = key[len(prefix):] if prefix else key
        dest = dest_root / relative
        outcome = r2.download(key, dest, expected=manifest.get(key), s3=client)
        size_mb = dest.stat().st_size / 1e6
        print(f"  [{outcome:>10}] {relative}  ({size_mb:.1f} MB)")
        if outcome.startswith("downloaded"):
            downloaded += 1
        else:
            skipped += 1

    print(f"\n{downloaded} downloaded, {skipped} already present and verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
