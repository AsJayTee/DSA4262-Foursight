# Where the model ranks well, and whether a shortlist works

Task 2 analyses behind the site shortlist dashboard
([report/task2_plan.md](../../report/task2_plan.md), section 4). Everything is
measured on **held-out genes** of the two labelled cell lines: every site is
scored by a fold model that never saw its gene. The model is the shipped
ensemble's three designs (`h2gcn_twohead_aux`, `res_gate`, `scalar_drop`), three
seeds each, trained on both files; each network's score is ranked within its
file and the nine ranks are averaged, as `scripts/predict.py` does.

## Run

```bash
python analysis/site_types/build_table.py   # one row per labelled site -> .cache/site_types/
python analysis/site_types/by_type.py       # -> results/by_type.csv
python analysis/site_types/shortlist.py     # -> results/shortlist.csv, results/gene_best.csv
```

All three run on a laptop in under a minute. Their inputs are produced on a
larger machine and live in the gitignored `.cache/representation/xsrc_nets/`:

| Input | Made by |
|---|---|
| `site_keys.csv.gz` | `analysis/representation/export_site_keys.py` |
| held-out scores at full depth (`*_pooled_both_fold*.npy`) | `xsrc_nets.py` |
| held-out scores at 1, 3, 10 reads, 5 random draws (`depth_scores*/`) | `depth_rescore.py [--subsample-seed N]` |
| the same sites from two other HCT116 runs (`hct116_runs_sites/`) | `score_hct116_runs_sites.py` |

`.cache/site_types/` holds the course labels and never leaves the machine. The
`results/` tables are aggregates only.

## What the numbers mean

**Pairwise accuracy** (`by_type.csv`, ROC AUC): take one modified and one
unmodified site of the same type; how often the model ranks the modified one
higher. 0.5 is chance. It is the headline for comparing types because it does
not depend on how often a type is modified.

PR AUC is reported beside it, but **PR AUC / base rate ("lift") is not used to
compare types**: it cannot exceed 1 / base rate, so GGACT (23% modified) is
capped at 4.4x while a rarely modified motif can exceed 100x. That cap is why
GGACT's lift looked lowest in earlier per-motif tables.

**Hit rate** (`shortlist.csv`): among genes with at least one modified site and
at least 5 sites, the share where at least one of the model's top k is modified.
Compared with a motif rule (sites ordered by how often their motif is modified,
learned from other genes) and a random pick. Copies of one genome position on
several transcripts are merged first, as on the dashboard.

**Gene's best site** (`gene_best.csv`): genes grouped by where their best site
ranks in the whole file, and the share of each group with any modified site.

Intervals are 95%, from resampling whole genes. At 1, 3 and 10 reads, sites are
thinned at random; each number is averaged over five draws, and `draw_sd_*`
shows how much one draw moves it.

## Limits

- Labelled sites all have 20+ reads; low depth is simulated by thinning.
- Labels come from one assay (m6ACE-seq), which misses some real m6A, so hit
  rates are likely underestimates.
- Two cell lines only, and both were in the training data (other genes).
- Nearby-site counts are among labelled sites only, which are sparser than all
  candidate sites in an SG-NEx sample.
