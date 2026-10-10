# Report figures: what each one shows and how to read it

A guide for whoever writes or presents with these figures. Each entry covers:
- what the figure shows;
- how the numbers were computed;
- how to read it;
- what not to claim from it;
- where the numbers come from.

The full results and their context are in [../findings.md](../findings.md).

**Regenerating.** No Ronin, no torch, no data download needed: every figure
reads committed result tables.

```bash
python analysis/representation/report_models.py --plot-only   # models_* figures
python analysis/representation/report_figures.py              # fig_* figures
```

(`report_models.py` without `--plot-only` recomputes its tables from the cached
predictions in `.cache/representation/`, which live only on the machine that
copied them off Ronin.)

**Shared conventions.**
- **Cell line 1** = dataset0 (HCT116, the course's training file). **Cell
  line 2** = data1 (a different cell line, with its own labels).
- **"Held-out genes":** every score comes from a model that never saw that
  gene, in either file (gene-grouped 5-fold split, seed 4262).
- **Two training versions:**
  - **cl1** (*trained on cell line 1 only*): cell line 2 is a cell line the
    model never saw. The clean transfer test.
  - **both** (*trained on both cell lines*): the shipped setting. Cell line
    2's genes are unseen, the cell line is not.
- **Colours:** in the `fig_*` figures, blue = cell line 1 and orange = cell
  line 2. In the `models_*` figures, colour = model family:
  - grey: no neighbours (baselines, reads only);
  - orange: free use of neighbours;
  - blue: constrained corroboration;
  - aqua: ensembles.

  Hollow markers are controls.
- **PR AUC** is the headline metric (positives: 4.5% / 7.3%). ROC AUC
  flatters every model.
- **Scores are rankings, not calibrated probabilities.**

---

## Model comparison

### models_distribution_cl1.png, models_distribution_both.png

**Shows:** for each of 18 models, its PR AUC gain over quantiles + LightGBM
*on the same fold*. Left panel: scored on cell line 1. Right panel: scored on
cell line 2.

**Computed:**
- Each point is one fold (of 5) of one training seed (up to 3).
- Gain = model PR AUC - LightGBM PR AUC on the identical held-out sites of
  that fold, seed-matched where LightGBM has that seed.
- The violin is the spread of those points; the black bar is the median;
  `n` = folds × seeds.

**Why gains, not raw PR AUC:** folds differ in difficulty (positive rate
4.1-5.2%) more than most models differ from each other. Raw per-fold PR AUC
would hide every comparison; the paired gain cancels the fold's difficulty.

**Read:**
- Right of the dashed line = beats LightGBM.
- A narrow violin = consistent across folds and seeds.
- Compare the two panels: a model far right on cell line 1 but not on cell
  line 2 (H2GCN 400 nt, graph transformer) learned something specific to
  cell line 1.

**Don't claim:** that rows with n = 5 (one seed) are as precise as rows with
n = 15. Logistic regression and m6Anet are always 5: one is deterministic,
the other a fixed pretrained model.

**Source:** `analysis/representation/results/report_models_folds.csv`.

### models_scatter_pr_{cl1,both}.png, models_scatter_roc_{cl1,both}.png

**Shows:** each model's PR AUC (or ROC AUC) on cell line 1 (x) against cell
line 2 (y). Numbered points, with a key.

**Computed:** all held-out sites at once; each model's seeds averaged by
within-file rank.

**Read:**
- Top right = better on both.
- **Solid diagonal (y = x):** equal on both cell lines. Every model sits
  below it, because cell line 2 is harder.
- **Dashed horizontal line:** LightGBM's score on cell line 2. Above it =
  beats the baseline on the second cell line.
- Both axes use the same scale, so distances are comparable in both
  directions.
- The story in the `cl1` PR plot:
  - wide-context and high-capacity models (9, 10) sit far right but low;
  - the constrained designs (11, 13, 15) sit highest;
  - the random-neighbour control (12) falls back toward the reads-only
    models.

**Don't claim:** significance from point positions. Use the paired,
gene-resampled intervals in `results/day_significance.csv` (findings 12D).

**Source:** `analysis/representation/results/report_models_pooled.csv`.

---

## How the model works

### fig_corroboration_curve.png

**Shows:** what the *shipped* scalar-messages networks do in one concrete
situation. A site has one neighbour within 75 nt; both are read 30 times.
- **x-axis:** how modified the neighbour's own reads look.
- **y-axis:** the site's final score.
- **Three lines:** sites whose own reads say 5%, 20% or 50%.
- **Dotted line at each level:** "the neighbour changed nothing".

**Computed:** the two shipped `scalar_drop` networks
(`models/final/scalar_drop_s{0,1}.npz`) evaluated directly in numpy. No
data, no averaging over sites.

**Read:** the middle line (the site's own reads say 20%):
- a neighbour at 90% lifts the site to ~45%;
- a neighbour at 5% pulls it down to ~12%;
- the break-even is around 40%.

That is corroboration: neighbours confirm or contradict the site's own
evidence.

**Don't claim:** absolute probabilities. The networks were trained with
positives up-weighted, so read the direction and size of the shift.

**Source:** `models/final/`. Related: the partial-dependence version over
real held-out sites is `results/probe_scalar_curve.csv` (findings 12E).

### fig_neighbour_count.png

**Shows:** how much the graph adds over the same read encoder with no graph
(H2GCN + own-reads head minus DeepSet), split by how many other candidate
sites lie within 50 nt. Trained on cell line 1. Bars are 95% gene-resampled
intervals.

**Read:**
- On cell line 1 the gain rises from ~0 (no neighbours) to +0.06 (4+
  neighbours): more neighbours, more corroboration.
- On cell line 2 it is smaller and flat after one neighbour.
- Sites with no neighbours gain nothing, as expected: there is nothing to
  corroborate with.

**Don't claim:** that neighbour count alone causes the gain. Neighbour count
also tracks how well a transcript is covered. Transcript size (a coverage
proxy) does *not* show the same trend (findings 12, confound 1), which
argues for corroboration.

**Source:** `analysis/representation/results/gain_decomposition.csv`.

### fig_radius.png

**Shows:** the gain over LightGBM as the neighbour radius widens from 50 to
400 nt (H2GCN + own-reads head, only the radius changed, seed 0).
- Blue: cell line 1 (trained on both).
- Orange: cell line 2 (trained on cell line 1 only, the unseen line).

**Read:**
- Wider context keeps helping cell line 1 (0.081 → 0.112).
- For the unseen line it peaks at ~100 nt and falls back by 200-400 nt.
- So context beyond ~100 nt is specific to the training cell line.

**Don't claim:** a precise optimum. One seed per radius; seed spread is
~0.003-0.005.

**Source:** `analysis/representation/results/h2gcn_aux*_nn__xsrc.json`.

### fig_gps_attention.png

**Shows:** why the graph transformer (GPS) fails to transfer.
- **Left:** where its whole-transcript attention goes, as the share of
  attention by distance from the scored site. About half goes beyond
  150 nt.
- **Right:** the same trained network rescored with its attention cut to
  150 nt and to 50 nt, and the change in PR AUC on each cell line.

**Computed:** seed-0 fold models trained on cell line 1, every held-out
site, no retraining (`probe_day.py`).

**Read:** cutting the long-range attention costs the training line 2.6-3×
more than the unseen line. So that long-range context carries information
specific to cell line 1.

**Don't claim:** the size of the drops. The network never saw cut attention
in training, so both drops are inflated. The *asymmetry* is the evidence.

**Source:** `analysis/representation/results/probe_gps.csv`,
`probe_gps_attention.csv`.

---

## The data and its limits

### fig_comodification.png

**Shows:** how much more often two candidate sites on the same transcript
are *both* modified than chance predicts, by the distance between them.

**Computed:**
1. For every pair of candidate sites on the same transcript, record their
   distance and whether both are modified.
2. Compare with two expectations:
   - the **file-wide rate**: e.g. 4.5% of sites are modified, so a random
     pair is both modified 0.045² = 0.2% of the time;
   - the **transcript's own rate**: shuffle which sites are modified *within
     each transcript*. With 10 sites and 2 modified, a pair is both
     modified with probability 2/10 × 1/9 = 2.2%.
3. Total enrichment (observed ÷ file-wide expectation) = **transcript part ×
   local part**:
   - **Dotted (transcript part)** = transcript-rate expectation ÷ file-wide
     expectation. How concentrated modified sites are on certain
     transcripts. It ignores distance by construction, so it is roughly
     flat. It is **not** the base modification rate.
   - **Solid (local part)** = observed ÷ transcript-rate expectation. The
     clustering beyond "this transcript is m6A-rich"; this is the
     distance-dependent part. Shaded = 95% interval from resampling
     transcripts.

**Read:**
- **Solid:** modified sites cluster within ~100-200 nt (2.2× below
  25 nt), *identically in both cell lines*. That is the transferable
  signal the graph uses.
- **Dotted:** which transcripts are m6A-rich differs between the lines
  (3.3× vs 2.0×). That is the cell-line-specific signal the graph
  transformer picks up.
- Nothing forces the solid line down: random placement within transcripts
  would give a flat 1×. Its drop is the finding. It dips *below* 1× far
  away because if modified pairs are concentrated close together, fewer
  are left far apart.

**Source:** `analysis/newdata/distance_bands/bands.csv`
(`analysis/newdata/distance_bands.py`). Labels only, no model.

### fig_gene_structure.png

**Shows:** the modified-site rate, relative to each file's overall rate, by
position in the gene.
- **Left:** distance to the nearest exon junction.
- **Right:** position relative to the stop codon.

**Computed:** Ensembl 91 annotation joined to the labelled sites. Used for
analysis only; never a model input.

**Read:**
- Almost no m6A within 25-50 nt of a junction (0.10× / 0.28×).
- Enrichment 100-400 nt away (2.2-2.9×).
- A peak just after the stop codon (2.7× / 2.4×).
- The same shape in both cell lines.

This matches the published exon-junction-complex model (Uzonyi 2023; He
2023; Luo 2023): gene structure shared by all cells sets where m6A can sit,
on a ~100 nt scale.

**Don't claim:** mechanism. It is correlation that fits the published
model.

**Source:** `analysis/newdata/annotation_context/`.

### fig_ceiling.png

**Shows:** part of the cross-cell-line gap cannot be closed with these
labels.

**Left panel** (67,320 sites present in both files), each scored against
cell line 2's labels:
- cell line 1's true labels used as the prediction: PR AUC 0.326;
- the same, with ties broken by our model's score: 0.485;
- our model trained on cell line 1: 0.422.

Learning cell line 1's labelling *perfectly* would score below our
reads-based model.

**Right panel:** H2GCN (constrained) and the graph transformer, each trained
on cell line 1, cell line 2 or both, scored on cell line 2's held-out
genes.
- Trained on cell line 2 itself, the transformer is better (0.422 vs 0.412):
  within one cell line, capacity pays.
- Trained on cell line 1 it is not.
- Training on both is best for both models (0.436).

**What the left panel bounds (careful):**
- It is an exact cap only for predictions that depend on cell line 1's
  labels.
- It is **not** a mathematical cap on every model: a model could in
  principle find cell-line-2-specific signal in cell line 2's reads.
- Two more tests (findings 12D) found almost none: no model tells the two
  cell lines apart at the 3,872 sites where their labels disagree (AUC
  ~0.5), and a model-free check of the raw reads gives AUC 0.52-0.54.

Safe wording: "the gap cannot be closed by learning the training cell line
better, and the reads show almost no trace of the label difference".

**Right panel caveats:** seed 0 only. Cell line 2 alone has fewer training
sites (~73k vs ~122k).

**Source:** `analysis/representation/results/label_oracle.csv`,
`data1_arm.csv`, `discordant_signal.csv`.

### fig_depth.png

**Shows:** PR AUC against reads per site for the shipped ensemble, the
intermediate leaderboard ensemble and LightGBM. Trained on both cell lines,
held-out genes. The grey band is 1-3 reads, typical for SG-NEx.

**Computed:**
- Every training site has at least 20 reads, so lower depths are
  **simulated**.
- Each held-out site's reads, *and its neighbours' reads*, are randomly
  thinned to 1, 3 or 10, then scored by models trained at full depth.
- Each read is a separate RNA molecule, so thinning removes real evidence:
  a fair simulation of shallower sequencing.

**Read:**
- Every model loses much of its signal at 1-3 reads.
- The networks keep about twice LightGBM's PR AUC there (cell line 1: 0.333
  vs 0.140 at 1 read).

**Don't claim:** that real SG-NEx low-depth sites behave exactly like
thinned ones. Real low-depth sites come from lowly expressed genes, which
may differ in other ways.

**Source:** `analysis/representation/results/depth_ensembles.csv`
(`depth_ensembles.py`, from per-site scores saved by `depth_rescore.py`).

### fig_data2_fractions.png

**Shows:** the shipped model's scores on dataset2, the course's in-vitro
series. One synthetic RNA was sequenced as seven samples made with a *known*
share of modified molecules (0, 25, 50, 70, 75, 95, 100%), at the same 189
positions (one every 10 nt, all DRACH motifs). Each point is one position.
Both panels use the **same networks**: the shipped constrained designs
(res_gate and scalar_drop, two seeds each).
- **Left:** with their neighbour step.
- **Right:** their own-reads score only.

**Computed:** scored with `models/final` (numpy); never used for training.
Shown on a 0-100% scale (sigmoid of the raw output). The full shipped
ensemble, which also includes the two-head networks, has no own-reads score
to compare, so it is quoted in text only (Spearman 0.41).

**Read:**
- The scores **separate unmodified from modified RNA sharply**:
  - with neighbours, median 37% at 0% vs 90% at 25% and ~96% from 50% up;
  - telling 0% from any modified sample: ROC AUC 0.75 → 0.87 (res_gate)
    and 0.77 → 0.90 (scalar_drop) when neighbours are added.
- They **plateau from about 50%**: the model cannot tell a half-modified
  site from a fully modified one. The own-reads score even dips slightly at
  95-100%.
- **Why neighbours help in synthetic RNA.** Within one sample every position
  has the same modification level by construction, and each site has 14
  neighbours within 75 nt. So every neighbour is an independent extra
  measurement of the same state, and pooling them reduces noise. That is the
  mechanism learned on real RNA, in the setting where its assumption holds
  perfectly. Spearman with the fraction: 0.37 vs 0.24.
- **The neighbour step is outside its training range here.** dataset2 sites
  have 600-1,200 reads, far above training, and the scalar messages use read
  counts. That likely explains why neighbours also lift the 0% sample a
  little (median 26% → 37%).
- Within one sample, some positions score low regardless of level: sequence
  context matters.

**Don't claim:**
- that the score measures the modification *fraction* (stoichiometry). It
  detects presence, then saturates.
- that neighbours help this much in real transcripts. Here every neighbour
  shares the site's level by construction; in real RNA neighbours share it
  only partly (2.2× within 25 nt, fading by ~200 nt). This is a best-case
  demonstration of the mechanism, not evidence about biology.
- that the help comes only from separate sites. At 10 nt apart, neighbours'
  signals may partly overlap with the site's own (DeepRM: ±10 nt). In real
  data, neighbours within 20 nt added nothing (findings 12C).

**Source:** `analysis/representation/results/data2_score_summary.csv`
(`analysis/newdata/data2_scores.py`). Per-site scores are in
`.cache/representation/data2_scores.csv` (they carry course labels, so they
are not committed); the script regenerates them in about a minute.

---

## Outdated

`runs_distribution.png` and `runs_scatter.png` come from the September W&B
runs (LightGBM-era models only). Superseded by the `models_*` figures; kept
for history.
