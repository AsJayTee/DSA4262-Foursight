"""R2: the decisive test, 50 paired observations, every training label used.

    python combine50.py --pairs 0:0 0:1 ...     # workers: one (rep, fold) per pair
    python combine50.py --aggregate             # paired tests -> results/r2.md

Per (repetition, fold), LightGBM with the `everything` parameters is trained
on ALL training rows. Two families of arm:

**Full-depth training**, scored at full depth, each paired against `hand`
(the 137 `quantiles_all_v1` columns):
  hand + kmer_norm, + each cross-fitted network score, + combinations.

**Depth-augmented training**, exactly as configs/everything.yaml does it
(training rows at 1, 3, 5, 10 reads and full; decision 0022), scored at full
depth AND at 1 and 3 reads, paired against `everything` (hand only). This is
the comparison against the real best model, and the first paired test at low
depth this project has had - the harness logs one pooled number per depth.

Tests: m6a.compare.paired_comparison, the harness's own Nadeau & Bengio
correction for 5-fold CV. Quote the corrected p and the win count.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import common
from crossfit50 import DEPTHS, job_path, repetition_folds
from probe import scores
from m6a import registry
from m6a.compare import paired_comparison

LGBM = dict(learning_rate=0.05, num_leaves=63, n_estimators=600, min_child_samples=40,
            feature_fraction=0.7, num_threads=4)
NETWORKS = ["mlp_hand", "deepset", "deepset_large", "read_ae_pred", "attn_mil"]
FULL_ARMS = {
    "hand": [],
    "hand + kmer_norm": ["kmer_norm"],
    **{f"hand + {m}": [m] for m in NETWORKS},
    "hand + kmer_norm + deepset": ["kmer_norm", "deepset"],
    "hand + kmer_norm + all networks": ["kmer_norm", *NETWORKS[1:]],
}
AUG_ARMS = {
    "everything": [],
    "everything + kmer_norm": ["kmer_norm"],
    "everything + deepset": ["deepset"],
    "everything + kmer_norm + deepset": ["kmer_norm", "deepset"],
    "everything + kmer_norm + all networks": ["kmer_norm", *NETWORKS[1:]],
}
TRAIN_DEPTHS = [1, 3, 5, 10, None]
TEST_DEPTHS = [None, 1, 3]
OUT = common.RESULTS / "r2"


def kmer_norm_all(bundle, train_rows: np.ndarray) -> dict:
    """kmer_norm features for EVERY site at every depth; per-7-mer statistics
    from the training rows' full-depth reads only (no labels involved)."""
    kmer_id = bundle.kmer_onehot.reshape(len(bundle.y), 7, 4).argmax(-1) @ (4 ** np.arange(7))
    blocks = bundle.reads
    owner = np.repeat(np.arange(len(bundle.y)), blocks.counts)
    is_train = np.zeros(len(bundle.y), bool)
    is_train[train_rows] = True
    values = common.transform_values(blocks.values)
    train_reads = is_train[owner]
    ids = kmer_id[owner]
    g_mean, g_sd = values[train_reads].mean(0), values[train_reads].std(0)
    mean, sd = np.tile(g_mean, (4 ** 7, 1)), np.tile(g_sd, (4 ** 7, 1))
    sel_ids = ids[train_reads]
    sel_vals = values[train_reads]
    order = np.argsort(sel_ids, kind="stable")
    sel_ids, sel_vals = sel_ids[order], sel_vals[order]
    bounds = np.flatnonzero(np.diff(sel_ids)) + 1
    for chunk_ids, chunk in zip(np.split(sel_ids, bounds), np.split(sel_vals, bounds)):
        if len(chunk) >= 200:
            mean[chunk_ids[0]] = chunk.mean(0)
            sd[chunk_ids[0]] = np.maximum(chunk.std(0), 1e-6)

    def features(block):
        vals = common.transform_values(block.values)
        own = np.repeat(np.arange(len(bundle.y)), block.counts)
        norm = (vals - mean[kmer_id[own]]) / sd[kmer_id[own]]
        z = common.pool_codes(norm, None, block.offsets, None, block.counts)
        return pd.DataFrame(z, index=bundle.X.index, columns=[f"kn{i}" for i in range(z.shape[1])])

    out = {None: features(blocks)}
    for d in DEPTHS:
        out[d] = features(bundle.depth_reads[d])
    return out


def run_pair(bundle, rep: int, fold: int, log) -> None:
    path = OUT / f"r{rep}f{fold}.json"
    if path.exists():
        log(f"skip r{rep}f{fold}")
        return
    started = time.time()
    folds = repetition_folds(bundle, rep)
    held = np.flatnonzero(folds == fold)
    train = np.flatnonzero(folds != fold)
    y = bundle.y
    kn = kmer_norm_all(bundle, train)
    nets = {}
    for m in NETWORKS:
        p = job_path(m, rep, fold)
        if p.exists():
            with np.load(p) as f:
                nets[m] = {None: f["full"], **{d: f[f"d{d}"] for d in DEPTHS}}

    def frame(depth, parts):
        X = bundle.X if depth is None else bundle.depth_X[depth]
        pieces = [X]
        for part in parts:
            if part == "kmer_norm":
                pieces.append(kn[depth])
            else:
                pieces.append(pd.DataFrame({f"net_{part}": nets[part][depth]}, index=bundle.X.index))
        return pd.concat(pieces, axis=1)

    def available(parts):
        return all(p == "kmer_norm" or p in nets for p in parts)

    record = {"repetition": rep, "fold": fold, "arms": {}}
    for arm, parts in FULL_ARMS.items():
        if not available(parts):
            continue
        X = frame(None, parts)
        model = registry.get("models", "lightgbm")(**LGBM)
        model.fit(X.iloc[train], y[train])
        record["arms"][arm] = {"full": scores(y[held], model.predict_proba(X.iloc[held]))}
    for arm, parts in AUG_ARMS.items():
        if not available(parts):
            continue
        frames = {d: frame(d, parts) for d in TRAIN_DEPTHS}
        model = registry.get("models", "lightgbm")(**LGBM)
        model.fit(pd.concat([frames[d].iloc[train] for d in TRAIN_DEPTHS]),
                  np.tile(y[train], len(TRAIN_DEPTHS)))
        record["arms"][arm] = {
            ("full" if d is None else f"d{d}"): scores(y[held], model.predict_proba(frames[d].iloc[held]))
            for d in TEST_DEPTHS}
    record["networks"] = sorted(nets)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1))
    log(f"done r{rep}f{fold} in {(time.time() - started) / 60:.1f} min: " + "  ".join(
        f"{a} {v['full']['pr_auc']:.4f}" for a, v in record["arms"].items()))


def aggregate() -> None:
    records = [json.loads(p.read_text()) for p in sorted(OUT.glob("r*f*.json"))]
    lines = [f"# R2: {len(records)} paired observations (repetition x fold)\n",
             "Every arm trains LightGBM (`everything` parameters) on all training labels. "
             "Network scores are cross-fitted (crossfit50.py). Corrected p is the harness's "
             "Nadeau & Bengio test for 5-fold CV. Quote it with the win count.\n"]
    summary = {}
    for family, baseline, arms, depths in (("Full-depth training", "hand", FULL_ARMS, ["full"]),
                                           ("Depth-augmented training (as everything.yaml)",
                                            "everything", AUG_ARMS, ["full", "d1", "d3"])):
        for depth in depths:
            lines += [f"\n## {family}, scored at {'full depth' if depth == 'full' else depth[1:] + ' read(s)'}\n",
                      "| arm | PR AUC | vs baseline | wins | corrected p | 95% CI (corrected) |",
                      "|---|---:|---:|---:|---:|---|"]
            for arm in arms:
                both = [r for r in records if arm in r["arms"] and baseline in r["arms"]]
                if not both:
                    continue
                obs = lambda name: [{"repetition": r["repetition"], "fold": r["fold"],  # noqa: E731
                                     "pr_auc": r["arms"][name][depth]["pr_auc"]} for r in both]
                mean = np.mean([o["pr_auc"] for o in obs(arm)])
                if arm == baseline:
                    lines.append(f"| {arm} | {mean:.4f} | - | - | - | n = {len(both)} |")
                    continue
                c = paired_comparison(obs(baseline), obs(arm), baseline, arm, n_folds=5)
                summary[f"{arm} @ {depth}"] = c
                lines.append(f"| {arm} | {mean:.4f} | {c['mean_difference']:+.4f} | "
                             f"{c['wins']}/{c['n_folds']} | {c['p_value_corrected']:.4f} | "
                             f"[{c['ci_low_corrected']:+.4f}, {c['ci_high_corrected']:+.4f}] |")
    (common.RESULTS / "r2.md").write_text("\n".join(lines) + "\n")
    (common.RESULTS / "r2.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=[], help="rep:fold")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
    bundle = common.load(args.limit, extra_depths=DEPTHS)
    for pair in args.pairs:
        rep, fold = map(int, pair.split(":"))
        try:
            run_pair(bundle, rep, fold, log)
        except Exception as exc:
            log(f"FAILED r{rep}f{fold}: {exc!r}")


if __name__ == "__main__":
    main()
