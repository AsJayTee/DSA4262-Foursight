# 0033. Ship a site-graph network ensemble, scored in numpy

- **Date:** 2026-10-04
- **Status:** Accepted
- **Affects:** `models/final/` (the shipped model; the previous LightGBM moves to `models/final/lightgbm/`), `scripts/predict.py` (a branch for models that consume sites), `src/m6a/models/site_graph.py` (new), `analysis/representation/final_fit.py` and `export_final.py` (new), `tests/test_site_graph.py` (new). No new dependency.

## Context

Under cross-source evaluation ([0029](0029-models-are-selected-on-cross-source-gain.md),
ranked by the worse gain per [0032](0032-rank-on-the-worse-gain.md)), graph
networks over each transcript's sites beat everything else, including
pretrained m6Anet, under both labellings and on genes never seen in training
(analysis/representation/results/, W&B Decisions view). The best combination
is an ensemble of two designs trained on both files:

- `h2gcn_twohead_aux`: neighbours within 200 nt plus a transcript summary; one
  output per labelling (shipped as their mean); an auxiliary head that scores
  each site from its own reads alone during training;
- `h2gcn_aux`: neighbours within 50 nt only, with the auxiliary head.

Rank-averaged, the two scored data1 PR AUC 0.450 against 0.392 for the
quantiles + LightGBM model shipped until now, and kept a +0.10 lead at 3 reads
per site (Task 2's depth).

Two things stood between that and `predict.py`. It must run on other students'
machines with only the base dependencies (AGENTS.md section 4), and torch is
not one. And every model so far scored a per-site feature table; these need
each site's reads, 7-mer and position, to build transcript graphs.

## Decision

1. **Ship `h2gcn_twohead_aux` + `h2gcn_aux`, two seeds each**, trained on every
   labelled site of both files by `final_fit.py`, early-stopped on the same 10%
   of genes every evaluated network used. Combined as the mean of each
   network's within-file rank, which is the combination that was evaluated.
2. **Score them in numpy.** `m6a.models.site_graph` re-implements the forward
   pass (dense layers, ReLU, averages over reads and neighbours); weights are
   exported to `.npz` by `export_final.py`. `tests/test_site_graph.py` checks it
   against the torch model on the same sites (agreement to 1e-4), including a
   one-site transcript and a one-read site.
3. **`predict.py` hands sites, not a feature table, to a model that sets
   `CONSUMES_SITES`.** Every other model takes the unchanged path.
4. **The score is a rank in [0, 1], not a calibrated probability.** The
   networks train with positives up-weighted, so their raw outputs overstate
   probability anyway; the leaderboard is scored on ROC AUC and PR AUC, which
   depend only on the ordering. A count of modified sites must not be read off
   a threshold on this score.

## Why this and not the alternatives

**Add torch as a base dependency.** Simplest code, but a ~200 MB install that
can fail on someone else's machine, for a forward pass a few dozen lines of
numpy reproduce exactly.

**Export to ONNX.** Also removes torch, but adds onnxruntime as a dependency.

**Keep shipping LightGBM.** Loses about +0.06 PR AUC on data1 and +0.10 at
3 reads.

**A single network.** Two different designs, rank-averaged, gained +0.014 over
the better one alone; more seeds of one design gained little.

## Consequences

- `predict.py` reads the whole input before scoring, because a site's score
  depends on the other sites of its transcript. A test file must contain whole
  transcripts for the graph to see a site's neighbours; a site alone still gets
  a score, from its own reads.
- The scores are only comparable within one file (ranks).
- The numpy model supports only the variants it was written for (`h2gcn`,
  plain read encoder); exporting anything else fails loudly.
- Training needs the train extra plus torch, on the training side only.

## How to check it still holds

```bash
pytest tests/test_site_graph.py tests/test_smoke.py
python scripts/predict.py --model models/final --input data/sample/sample.json.gz --output /tmp/p.csv
```
