# 0034. Ship the constrained-corroboration designs beside the two-head network

- **Date:** 2026-10-10
- **Status:** Accepted
- **Affects:** `models/final/` (the shipped ensemble changes), `src/m6a/models/site_graph.py` (a second numpy forward pass), `tests/test_site_graph.py` (covers the new kinds), `analysis/representation/final_fit.py` (the ensemble list; the kernel's decay scale). Supersedes the ensemble choice in [0033](0033-ship-a-site-graph-ensemble-scored-in-numpy.md), not its numpy approach. No new dependency.

## Context

Under 0033 the shipped model was `h2gcn_twohead_aux` + `h2gcn_aux`. The 8-10 Oct
experiments (report/findings.md section 12) found that the graph network works
by corroboration: nearby sites that look modified raise a site's score. They
also found that constraining *how* neighbours are used transfers better to a
cell line the model never trained on. Two designs stood out:

- **`res_gate`:** the site's own-reads score, plus a neighbour correction from a
  kernel exp(-d / λ) over 150 nt, trained with neighbour dropout. The gate
  turned out not to matter; the dropout did.
- **`scalar_drop`:** neighbours pass only their own-reads score, plus read
  counts, into a small network that corrects the site's score. Also trained
  with dropout.

Every component was averaged over 3 seeds and compared against the shipped
ensemble on identical sites, with gene-resampled intervals
(`day_significance.py`). The best ensemble, `h2gcn_twohead_aux` + `res_gate` +
`scalar_drop`:

| | Cell line 1 | Cell line 2 | data1's new genes |
|---|---|---|---|
| Gain over the shipped ensemble | +0.001 [-0.005, +0.006] | **+0.006 [+0.002, +0.009]** | **+0.012 [+0.000, +0.023]** |

Under [0032](0032-rank-on-the-worse-gain.md), the worse gain is +0.001, a tie.
The tie breaks on data1's new genes (+0.012). On two further HCT116
sequencing runs (findings section 13), the closely related ensemble with
`scalar_msg` also beat the shipped one (+0.009 / +0.012).

## Decision

1. **Ship `h2gcn_twohead_aux` + `res_gate` + `scalar_drop`, two seeds each.**
   Every network gets equal weight in the mean of within-file ranks, as
   evaluated. The two-head networks are the 4 Oct fits, unchanged. The new
   designs are fitted by `final_fit.py` on every labelled site of both files.
2. **λ is fitted to cell line 1's training labels**, as in every evaluated
   network (`graph.comod_decay_scale`). It is stored in the exported weights
   (`kernel_nt`).
3. **numpy scores the new designs with a second forward pass**
   (`Network._logits_v2`). It is eval mode, so nothing is dropped. The test
   checks it against torch for all four kinds, with a non-trivial gate and λ.

## Why this and not the alternatives

**Keep the 0033 ensemble.** It is equal on cell line 1 and slightly worse on
the cell line it did not train on. The report argues that constrained,
transferable corroboration is the right approach, so the shipped model
should be the one it argues for. The gain is small, and this is that
argument made consistent.

**`res_gate` or `scalar_drop` alone.** Both lose to the shipped ensemble on
cell line 1 (-0.019 and -0.038). The two-head network carries what is
specific to HCT116, which helps if the test file is another HCT116 run.

**Add the graph transformer (`gps`).** It is better within each cell line but
not across them (findings 12E-12F). Its gain is the training cell line's own
labelling.

**`scalar_msg` instead of `scalar_drop`.** They are equal on cell line 2.
`scalar_drop` is +0.012 better on cell line 1.

## Cost

- **A second forward pass to maintain** in `site_graph.py`. The test pins it
  to torch.
- **Six networks instead of four:** about 1.5× slower prediction.
- **The gain is small** (+0.006 on cell line 2). It should be quoted with
  its interval, not as an improvement in general.
