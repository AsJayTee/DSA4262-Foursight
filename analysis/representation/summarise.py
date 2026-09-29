"""Turn results/probe_*.json into the tables in results/summary.md.

    python analysis/representation/summarise.py

**The significance test.** Five paired folds, Nadeau & Bengio corrected. The
correction inflates the variance by (1/n + n_test/n_train). The harness uses
1/(k-1) = 0.25 for the train/test ratio of plain 5-fold CV; here probes train
on only half of four folds, so the ratio is ~0.5 and the correction is larger.
It is computed from the actual row counts rather than borrowed.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

import common

ORDER = ["hand", "hand_pca", "random", "mlp_hand", "deepset", "read_ae", "read_ae_pred",
         "pos_lstm_ae", "pos_attn_ae", "set_masked", "attn_mil", "contrastive",
         "deepset_nokmer", "kmer_norm"]
PROBES = ["logistic", "knn", "mlp", "lgbm", "lgbm_rotated"]


def load() -> dict[str, list[dict]]:
    out = {}
    for name in ORDER:
        path = common.RESULTS / f"probe_{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def paired(a, b, ratio: float) -> dict:
    d = np.asarray(b) - np.asarray(a)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return {"diff": float(d.mean()), "wins": int((d > 0).sum()), "p": float("nan"), "n": n}
    se = np.sqrt((1 / n + ratio) * d.var(ddof=1))
    t = d.mean() / se
    return {"diff": float(d.mean()), "wins": int((d > 0).sum()), "n": n,
            "p": float(2 * stats.t.sf(abs(t), n - 1)),
            "ci": [float(d.mean() - stats.t.ppf(0.975, n - 1) * se),
                   float(d.mean() + stats.t.ppf(0.975, n - 1) * se)]}


def pr(records, *path):
    values = []
    for r in records:
        node = r
        for key in path:
            node = node[key]
        values.append(node["pr_auc"])
    return values


def fmt_test(t: dict) -> str:
    return f"{t['diff']:+.4f} | {t['wins']}/{t['n']} | {t['p']:.3f}"


def main() -> None:
    results = load()
    bundle = common.load(with_reads=False)
    ratios = []
    for fold in sorted(np.unique(bundle.folds)):
        roles = common.split_roles(bundle, fold)
        ratios.append(roles["heldout"].sum() / roles["probe"].sum())
    ratio = float(np.mean(ratios))
    hand = results["hand"]
    lines = [f"# Representation screen: results\n",
             f"Canonical split, 5 folds, probes trained on half of each training fold "
             f"(test/train ratio {ratio:.2f}). PR AUC on the held-out fold, mean over folds. "
             f"Paired tests are Nadeau & Bengio corrected with that ratio. **A screen, not "
             f"a result to quote against the harness's 0.5408** - see PLAN.md section 3.\n"]

    lines += ["## E1. Each representation alone, through five probes\n",
              "| representation | dims | logistic | kNN | MLP | LightGBM | LightGBM, rotated |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, recs in results.items():
        row = [f"{np.mean(pr(recs, 'probes', p)):.4f}" for p in PROBES]
        lines.append(f"| `{name}` | {recs[0]['dims']} | " + " | ".join(row) + " |")

    lines += ["\n## E2. Does the tree need the axes? LightGBM on z vs a random rotation of z\n",
              "Negative = rotating hurt the tree, so the information sits along the "
              "coordinate axes. For `hand` this should be clearly negative (positive control).\n",
              "| representation | rotated minus plain | wins | p | best non-tree probe minus LightGBM |",
              "|---|---:|---:|---:|---:|"]
    for name, recs in results.items():
        t = paired(pr(recs, "probes", "lgbm"), pr(recs, "probes", "lgbm_rotated"), ratio)
        best = max(np.mean(pr(recs, "probes", p)) for p in ("logistic", "knn", "mlp"))
        lines.append(f"| `{name}` | {fmt_test(t)} | {best - np.mean(pr(recs, 'probes', 'lgbm')):+.4f} |")

    lines += ["\n## E3. What z adds to the hand features\n",
              "| representation | + hand, LightGBM: diff / wins / p | + hand, logistic: diff / wins / p |",
              "|---|---|---|"]
    for name, recs in results.items():
        if name == "hand":
            continue
        tl = paired(pr(hand, "probes", "lgbm"), pr(recs, "with_hand", "lgbm"), ratio)
        tg = paired(pr(hand, "probes", "logistic"), pr(recs, "with_hand", "logistic"), ratio)
        lines.append(f"| `{name}` | {fmt_test(tl)} | {fmt_test(tg)} |")
    lines.append(f"\nReference: `hand` alone scores LightGBM "
                 f"{np.mean(pr(hand, 'probes', 'lgbm')):.4f}, logistic "
                 f"{np.mean(pr(hand, 'probes', 'logistic')):.4f}.")

    lines += ["\n## E4. What is in the vector\n",
              "| representation | R^2 from hand | CKA with hand | R^2 of depth | motif accuracy | effective dims |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, recs in results.items():
        red = [r.get("redundancy", {}) for r in recs]
        r2 = f"{np.mean([x['r2_hand_to_z'] for x in red]):.3f}" if red[0] else "-"
        cka = f"{np.mean([x['cka_with_hand'] for x in red]):.3f}" if red[0] else "-"
        c = [r["content"] for r in recs]
        lines.append(f"| `{name}` | {r2} | {cka} | {np.mean([x['r2_depth'] for x in c]):.3f} | "
                     f"{np.mean([x['motif_accuracy'] for x in c]):.3f} | "
                     f"{np.mean([x['participation_ratio'] for x in c]):.1f} |")

    lines += ["\n## E5. Depth: probes trained at full depth, sites embedded from fewer reads\n",
              "| representation | 1 read (lgbm / logistic) | 3 reads | 10 reads | full |",
              "|---|---|---|---|---|"]
    for name, recs in results.items():
        cells = []
        for d in common.EVAL_DEPTHS:
            cells.append(f"{np.mean(pr(recs, 'depth', str(d), 'lgbm')):.4f} / "
                         f"{np.mean(pr(recs, 'depth', str(d), 'logistic')):.4f}")
        cells.append(f"{np.mean(pr(recs, 'probes', 'lgbm')):.4f} / "
                     f"{np.mean(pr(recs, 'probes', 'logistic')):.4f}")
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")

    lines += ["\n## Training record\n",
              "| representation | parameters | epochs (mean) | minutes / fold | best val AP | nonfinite |",
              "|---|---:|---:|---:|---:|---|"]
    for name, recs in results.items():
        enc = [r["encoder"] for r in recs]
        hist = [r["history"] for r in recs]
        epochs = np.mean([len(h) for h in hist]) if hist[0] else 0
        vals = [max((e.get("val_ap", np.nan) for e in h), default=np.nan) for h in hist]
        val = f"{np.nanmean(vals):.4f}" if np.isfinite(np.nanmean(vals) if len(vals) else np.nan) else "-"
        lines.append(f"| `{name}` | {enc[0].get('parameters', '-')} | {epochs:.0f} | "
                     f"{np.mean([e.get('minutes_total', 0) for e in enc]):.1f} | {val} | "
                     f"{any(r['nonfinite'] for r in recs)} |")

    (common.RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
