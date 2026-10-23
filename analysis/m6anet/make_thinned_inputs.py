"""m6Anet inputs at reduced read depth: the same held-out sites and reads our model was scored on.

    python analysis/m6anet/make_thinned_inputs.py --out data/m6anet/thinned          # all
    python analysis/m6anet/make_thinned_inputs.py --out /tmp/probe --depths 1 --limit 2000

The Model stats page compares read depth (1, 3, 10 reads per site) for our model only,
because m6Anet's predictions existed only at full depth. This writes m6Anet's input
format (data.json + data.info; see make_m6anet_index.py for the two format rules) for
both labelled files at each depth, with every site thinned by the SAME keyed draw our
depth evaluation used (m6a.data.subsample_blocks, default seed, over the bundle built by
analysis/representation/xsrc_nets.load). So m6Anet and our model see identical reads.

m6Anet drops sites under 20 reads by default; scoring these inputs needs that minimum
lowered (see run_ronin.sh, step "thinned"). That is outside m6Anet's intended range, and
is exactly the SG-NEx situation (median 3-5 reads per site).

Writes <out>/<file>_d<depth>/data.json, data.info (gitignored under data/).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "representation"))

import numpy as np  # noqa: E402

import common  # noqa: E402
import xsrc_deepset  # noqa: E402
import xsrc_nets as X  # noqa: E402
from m6a import external, feature_cache  # noqa: E402
from m6a.config import Config  # noqa: E402
from m6a.data import subsample_blocks  # noqa: E402


def write(out_dir: Path, keys, kmers, blocks, rows) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    offset, n = 0, 0
    # newline="" keeps "\n": byte offsets in data.info must match (make_m6anet_index.py).
    with (out_dir / "data.json").open("w", encoding="utf-8", newline="") as dj, \
            (out_dir / "data.info").open("w", encoding="utf-8", newline="") as di:
        di.write("transcript_id,transcript_position,n_reads,start,end\n")
        for r in rows:
            t, p = keys[r]
            reads = blocks.values[blocks.offsets[r]:blocks.offsets[r + 1]]
            # m6Anet reads the last column as a read id (make_m6anet_index.py, rule 2).
            with_ids = [[float(v) for v in row] + [i] for i, row in enumerate(reads)]
            payload = json.dumps({t: {str(int(p)): {kmers[r]: with_ids}}}, separators=(",", ":"))
            dj.write(payload + "\n")
            di.write(f"{t},{int(p)},{len(reads)},{offset},{offset + len(payload)}\n")
            offset += len(payload) + 1
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depths", default="1,3,10")
    ap.add_argument("--limit", type=int, default=None, help="first N sites of each file (a probe)")
    args = ap.parse_args()
    started = time.time()
    sources, bundle = X.load(1.0)
    keys = list(sources["dataset0"].index) + list(sources["data1"].index)
    cfg = Config.load(xsrc_deepset.BASELINE)
    paths = {"dataset0": common.DATA, "data1": external.paths(common.DATA_DIR, "data1")[0]}
    kmers = np.concatenate([
        feature_cache.extract(paths[f], cfg.features, [None], log=lambda *a: None)[None].sites.loc[sources[f].index, "kmer"].to_numpy()
        for f in ("dataset0", "data1")])
    n0 = len(sources["dataset0"])
    print(f"loaded {len(keys):,} sites ({time.time() - started:.0f}s)", flush=True)
    for d in [int(x) for x in args.depths.split(",")]:
        thinned = subsample_blocks(bundle.reads, keys, d)     # the default draw, as depth_rescore.py
        for f, rows in (("dataset0", range(0, n0)), ("data1", range(n0, len(keys)))):
            rows = list(rows)[: args.limit] if args.limit else rows
            n = write(Path(args.out) / f"{f}_d{d}", keys, kmers, thinned, rows)
            print(f"{f} at {d} reads: {n:,} sites -> {Path(args.out) / f'{f}_d{d}'} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
