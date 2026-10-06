"""Do sites that RNA folding brings together co-modify more than sites the same
distance apart that it does not? (stage 1 of the folding question, 2026-10-06)

    python analysis/newdata/fold_comodification.py --jobs 60

The radius test showed that a wider neighbour radius does not transfer between
cell lines - but a radius averages ~13 neighbours at 400 nt, almost none
linked by a fold, so it cannot see whether the FEW fold-linked pairs co-modify.
This asks that directly, from labels alone, before any model is built:

1. Fold every transcript carrying a labelled site with RNAplfold (ViennaRNA;
   local folding, window -W 480, maximum base-pair span -L 400) on the
   Ensembl 91 cDNA, which matches every site's 7-mer (annotation_context.py).
2. Link two candidate sites on the same transcript if the ±5 nt windows around
   their A's are predicted to pair: max base-pair probability between the two
   windows >= tau (0.1 and 0.3).
3. For pairs in each distance band, compare co-modification of linked and
   unlinked pairs as in distance_bands.py: observed positive-positive pairs
   over the within-transcript expectation k(k-1)/(n(n-1)) - so positive-rich
   transcripts cannot fake a link effect. Intervals resample transcripts.
4. Repeat inside the methylation-permissive zone (both sites >= 100 nt from an
   exon junction), so the exon-junction effect cannot masquerade as structure.
5. Separately: is the modified A more often predicted UNPAIRED?

--predictor linearpartition repeats the test with LinearPartition (Zhang et al.,
Bioinformatics 2020; -V, the same Vienna energy model), which folds the WHOLE
transcript with no span limit - so the result cannot be an artefact of
RNAplfold's local windows, and pairs 400-800 nt apart can be tested too. Build:
git clone https://github.com/LinearFold/LinearPartition && make (path: --lp-bin).

Per cell line (dataset0, data1 are different cell lines). Predicted structure is
for the unmodified sequence in vitro; read every result with that caveat.
Analysis-side only: RNAplfold never enters predict.py.
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from m6a import external  # noqa: E402
from m6a.data import load_labels, resolve_data_dir  # noqa: E402

CDNA = ROOT / "data" / "annotation" / "cdna_91.fa.gz"
CONTEXT = ROOT / "analysis" / "newdata" / "annotation_context" / "site_context.csv.gz"
OUT = ROOT / "analysis" / "newdata" / "fold_comodification"
WINDOW, SPAN, FLANK = 480, 400, 5   # span 400: pairs up to 400 nt apart can link (200 left the 200-400 band empty)
TAUS = (0.1, 0.3)
BANDS = [(25, 50), (50, 100), (100, 200), (200, 400)]
PREDICTOR, LP_BIN = "rnaplfold", "/mnt/sdd/LinearPartition/linearpartition"   # set in main
RESAMPLES = 200
RNG = np.random.default_rng(4262)


def read_cdna(wanted: set[str]) -> dict[str, str]:
    seqs, name, chunks = {}, None, []
    with gzip.open(CDNA, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name in wanted:
                    seqs[name] = "".join(chunks)
                name, chunks = line[1:].split()[0].split(".")[0], []
            else:
                chunks.append(line.strip())
    if name in wanted:
        seqs[name] = "".join(chunks)
    return seqs


def init_worker(predictor: str, lp_bin: str) -> None:
    """Workers get the predictor explicitly. Python 3.14 starts pool workers
    fresh (forkserver) rather than forking, so a global set in main() is NOT
    inherited - which silently ran RNAplfold for a "linearpartition" run once."""
    global PREDICTOR, LP_BIN
    PREDICTOR, LP_BIN = predictor, lp_bin


def fold(item: tuple[str, str, list[int]]) -> tuple[str, str, dict, dict]:
    """RNAplfold one transcript. Returns, for its site positions (0-based A),
    the max pair probability between every two sites' windows, and each A's
    unpaired probability."""
    tid, seq, sites = item
    if PREDICTOR == "linearpartition":
        pairs, unpaired = fold_linearpartition(seq, sites)
    else:
        pairs, unpaired = fold_rnaplfold(seq)
    return tid, PREDICTOR, *site_links(pairs, unpaired, sites)


def fold_linearpartition(seq: str, sites: list[int]):
    """Global ensemble folding; pairs >= 0.001 kept so unpaired probabilities
    (1 - sum of a base's pair probabilities) are accurate."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "bpp.txt"
        subprocess.run([LP_BIN, "-V", "-r", str(out), "-c", "0.001"], input=seq + "\n", text=True,
                       check=True, capture_output=True)
        pairs, paired = [], np.zeros(len(seq))
        for line in open(out):
            f = line.split()
            if len(f) == 3:
                i, j, q = int(f[0]) - 1, int(f[1]) - 1, float(f[2])
                paired[i] += q
                paired[j] += q
                if q >= 0.01:
                    pairs.append((i, j, q))
    return pairs, {s: float(1 - paired[s]) for s in sites if 0 <= s < len(seq)}


def fold_rnaplfold(seq: str):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["RNAplfold", "-W", str(WINDOW), "-L", str(SPAN), "-u", "1", "--cutoff", "0.01"],
                       input=f">t\n{seq}\n", text=True, cwd=d, check=True, capture_output=True)
        pairs = []
        for line in open(Path(d) / "t_dp.ps"):
            f = line.split()
            if len(f) == 4 and f[3] == "ubox" and f[0].isdigit():
                pairs.append((int(f[0]) - 1, int(f[1]) - 1, float(f[2]) ** 2))
        unpaired = {}
        for line in open(Path(d) / "t_lunp"):
            if line[0].isdigit():
                i, u = line.split()[:2]
                unpaired[int(i) - 1] = float(u)
    return pairs, unpaired


def site_links(pairs, unpaired, sites):
    # nucleotide -> owning site(s), for windows of +-FLANK around each A
    owner: dict[int, list[int]] = {}
    for k, a in enumerate(sites):
        for x in range(a - FLANK, a + FLANK + 1):
            owner.setdefault(x, []).append(k)
    link: dict[tuple[int, int], float] = {}
    for i, j, p in pairs:
        for a in owner.get(i, ()):
            for b in owner.get(j, ()):
                if a != b:
                    key = (min(a, b), max(a, b))
                    link[key] = max(link.get(key, 0.0), p)
    return {(sites[a], sites[b]): p for (a, b), p in link.items()}, \
        {s: unpaired.get(s, np.nan) for s in sites}


def pair_table(frame: pd.DataFrame, links: dict, zone_ok: dict) -> pd.DataFrame:
    """One row per same-transcript site pair 25-400 nt apart: band, link prob,
    both positive, the within-transcript expectation, and whether both are in
    the permissive zone."""
    rows = []
    for t, g in frame.groupby("transcript_id"):
        if t not in links:
            continue
        p = g.transcript_position.to_numpy()
        y = g.label.to_numpy().astype(int)
        n, k = len(p), int(y.sum())
        if n < 2:
            continue
        frac = k * (k - 1) / (n * (n - 1))
        L = links[t]
        iu, ju = np.triu_indices(n, 1)
        d = np.abs(p[iu] - p[ju])
        keep = (d >= BANDS[0][0]) & (d < BANDS[-1][1])
        for a, b, dist in zip(iu[keep], ju[keep], d[keep]):
            pa, pb = int(p[a]), int(p[b])
            rows.append((t, dist, L.get((min(pa, pb), max(pa, pb)), 0.0), y[a] * y[b], frac,
                         zone_ok.get((t, pa), False) and zone_ok.get((t, pb), False)))
    return pd.DataFrame(rows, columns=["transcript_id", "dist", "link", "both", "expected", "zone"])


def ratios(pairs: pd.DataFrame, tau: float) -> dict:
    out = {}
    for lo, hi in BANDS:
        b = pairs[(pairs.dist >= lo) & (pairs.dist < hi)]
        for linked in (True, False):
            s = b[(b.link >= tau) == linked]
            out[(lo, hi, linked)] = (s.both.sum() / s.expected.sum() if s.expected.sum() else np.nan, len(s))
    return out


def compare(name: str, pairs: pd.DataFrame) -> list[dict]:
    rows = []
    tx = pairs.transcript_id.unique()
    groups = {t: g for t, g in pairs.groupby("transcript_id")}
    for tau in TAUS:
        point = ratios(pairs, tau)
        boots = []
        for _ in range(RESAMPLES):
            pick = RNG.choice(tx, len(tx))
            boots.append(ratios(pd.concat([groups[t] for t in pick]), tau))
        for lo, hi in BANDS:
            (rl, nl), (ru, nu) = point[(lo, hi, True)], point[(lo, hi, False)]
            diff = np.array([b[(lo, hi, True)][0] / b[(lo, hi, False)][0] for b in boots])
            rows.append({"data": name, "tau": tau, "band": f"{lo}-{hi} nt", "linked pairs": nl,
                         "unlinked pairs": nu, "linked local": rl, "unlinked local": ru,
                         "linked / unlinked": rl / ru if ru else np.nan,
                         "ci_lo": np.nanpercentile(diff, 2.5), "ci_hi": np.nanpercentile(diff, 97.5)})
    return rows


def main() -> None:
    global PREDICTOR, LP_BIN, BANDS, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=60)
    ap.add_argument("--limit", type=int, help="first N transcripts only (smoke)")
    ap.add_argument("--predictor", choices=("rnaplfold", "linearpartition"), default="rnaplfold")
    ap.add_argument("--lp-bin", default=LP_BIN)
    args = ap.parse_args()
    PREDICTOR, LP_BIN = args.predictor, args.lp_bin
    if PREDICTOR == "linearpartition":
        # Global folding has no span limit, so the next band out is testable too.
        BANDS = BANDS + [(400, 800)]
        OUT = OUT.with_name(OUT.name + "_linearpartition")
    if args.limit:
        OUT = OUT.with_name(OUT.name + "_smoke")
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_data_dir()
    keys = ["transcript_id", "transcript_position", "label"]
    files = {"dataset0 (cell line 1)": load_labels(data_dir / "data.info.labelled")[keys],
             "data1 (cell line 2)": external.load_info(data_dir, "data1").reset_index()[keys]}
    union = pd.concat(files.values())
    by_tx = union.groupby("transcript_id").transcript_position.apply(lambda s: sorted(set(s)))
    if args.limit:
        by_tx = by_tx.iloc[:args.limit]
    cdna = read_cdna(set(by_tx.index))
    items = [(t, cdna[t], list(map(int, s))) for t, s in by_tx.items() if t in cdna and len(s) >= 1]
    print(f"folding {len(items):,} transcripts with {PREDICTOR}, {args.jobs} processes ...", flush=True)
    links, unpaired = {}, {}
    with Pool(args.jobs, initializer=init_worker, initargs=(PREDICTOR, LP_BIN)) as pool:
        for i, (t, used, L, U) in enumerate(pool.imap_unordered(fold, items, chunksize=4)):
            if used != PREDICTOR:
                raise SystemExit(f"A worker folded {t} with {used}, not {PREDICTOR} - aborting.")
            links[t] = L
            unpaired.update({(t, s): u for s, u in U.items()})
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1:,} folded", flush=True)
    ctx = pd.read_csv(CONTEXT)
    zone_ok = {(t, int(p)): bool(d >= 100) for t, p, d in
               zip(ctx.transcript_id, ctx.transcript_position, ctx.dist_junction) if d == d}

    rows = []
    pd.set_option("display.width", 220)
    for name, frame in files.items():
        frame = frame[frame.transcript_id.isin(links)]
        pairs = pair_table(frame, links, zone_ok)
        for label, sub in ((name, pairs), (f"{name}, both sites >=100 nt from a junction", pairs[pairs.zone])):
            rows += compare(label, sub)
        u = frame.assign(unpaired=[unpaired.get((t, int(p)), np.nan)
                                   for t, p in zip(frame.transcript_id, frame.transcript_position)])
        u["unpaired bin"] = pd.qcut(u.unpaired, 5, duplicates="drop")
        acc = u.groupby("unpaired bin", observed=True).label.agg(["mean", "size"])
        acc["lift"] = acc["mean"] / u.label.mean()
        print(f"\n{name}: positive rate by predicted unpaired probability of the A (quintiles)")
        print(acc.round(4).to_string())
        acc.to_csv(OUT / f"unpaired_{name.split()[0]}.csv")
    table = pd.DataFrame(rows)
    print("\nCo-modification beyond the transcript expectation, fold-linked vs unlinked pairs:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    table.to_csv(OUT / "linked_vs_unlinked.csv", index=False)


if __name__ == "__main__":
    main()
