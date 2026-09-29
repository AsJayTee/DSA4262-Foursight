"""Round-3 combinations: does the gated winner beat deepset where it counts?

    python combine_r3.py --winner res_attn_mil --pairs 0:0 0:1 ...
    python combine_r3.py --winner res_attn_mil --aggregate

combine50.py's machinery, pointed at a separate output folder
(results_r3/) so round 2's recorded results are never overwritten, with the
arms that answer round 3's question. Besides each arm against `everything`, the
aggregate pairs the winner directly against deepset - the comparison that
decides whether to ship it instead.
"""

from __future__ import annotations

import argparse
import json
import time

import common
import combine50
from m6a.compare import paired_comparison


def configure(winner: str) -> None:
    common.RESULTS = common.RESULTS.parent / "results_r3"
    combine50.OUT = common.RESULTS / "r2"
    combine50.NETWORKS = ["deepset", winner]
    combine50.FULL_ARMS = {"hand": [], "hand + deepset": ["deepset"], f"hand + {winner}": [winner]}
    combine50.AUG_ARMS = {"everything": [], "everything + deepset": ["deepset"],
                          f"everything + {winner}": [winner],
                          f"everything + deepset + {winner}": ["deepset", winner]}


def head_to_head(winner: str) -> None:
    recs = [json.loads(p.read_text()) for p in sorted(combine50.OUT.glob("r*f*.json"))]
    lines = ["\n## Head to head: the winner against deepset (depth-augmented, as everything.yaml)\n",
             "| comparison | scored at | difference | wins | corrected p |", "|---|---|---:|---:|---:|"]
    for base, cand in (("everything + deepset", f"everything + {winner}"),
                       ("everything + deepset", f"everything + deepset + {winner}")):
        for depth in ("full", "d1", "d3"):
            both = [r for r in recs if base in r["arms"] and cand in r["arms"]]
            obs = lambda a: [{"repetition": r["repetition"], "fold": r["fold"],  # noqa: E731
                              "pr_auc": r["arms"][a][depth]["pr_auc"]} for r in both]
            c = paired_comparison(obs(base), obs(cand), base, cand, n_folds=5)
            lines.append(f"| {cand} vs {base} | {depth} | {c['mean_difference']:+.4f} | "
                         f"{c['wins']}/{c['n_folds']} | {c['p_value_corrected']:.4f} |")
    path = common.RESULTS / "r2.md"
    path.write_text(path.read_text() + "\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", required=True)
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    configure(args.winner)
    if args.aggregate:
        combine50.aggregate()
        head_to_head(args.winner)
        return
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)  # noqa: E731
    bundle = common.load(args.limit, extra_depths=combine50.DEPTHS)
    for pair in args.pairs:
        rep, fold = map(int, pair.split(":"))
        try:
            combine50.run_pair(bundle, rep, fold, log)
        except Exception as exc:
            log(f"FAILED r{rep}f{fold}: {exc!r}")


if __name__ == "__main__":
    main()
