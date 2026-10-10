# Literature: what we cite, why, and how far each claim was checked

> **Updated as of 10 October 2026.** Published work after this date is not
> covered. Prior-art statements ("we found no ...") hold only up to this date
> and only for the searches listed in [Search record](#search-record).

Every citation the report may use, grouped by the job it does in the
argument. Each entry has:
- what we rely it on for;
- its **verification status**:
  - **Verified**: the claim was checked against the paper's text, abstract,
    or code by us.
  - **Verified (abstract/mirror)**: checked only against an abstract or a
    secondary copy.
  - **Reported**: comes from a deep-research report and was not re-checked
    by us. Check before citing.
  - **Unverified**: we could not confirm it. Do not cite as fact.

Several citations reached us through AI deep-research reports, and two of
those reports contained errors (see [Errors found](#errors-found-in-ai-generated-literature)).
Nothing below is "Verified" on the strength of a report alone.

Related files:
- [findings.md](findings.md): the results each citation supports (sections
  12B and 13).
- [../docs/literature-review.md](../docs/literature-review.md): an earlier
  (23 Sep) deep-research review of nanopore m6A methods, kept for its own
  provenance notes.

---

## 1. Biology: where m6A sits, and why nearby sites go together

| Work | What we use it for | Status |
|---|---|---|
| Uzonyi et al., *Mol Cell* 2023, 83(2):237-251, doi:10.1016/j.molcel.2022.12.026 | The exon junction complex excludes m6A within ~100 nt of splice junctions. Explains our near-junction depletion (fig_gene_structure) and why ~100 nt of context transfers between cell lines | Verified |
| He et al., *Science* 2023, doi:10.1126/science.abj9090 | Exon junction complexes suppress m6A over average-length internal exons, not long internal or last exons. Fits our enrichment after the stop codon and in long exons | Verified |
| Luo et al., *Nat Commun* 2023, 14:4172, doi:10.1038/s41467-023-39897-1 | The exon-intron boundary represses 12-34% of m6A at adjacent exons, over ~100 nt | Verified |
| m6Aiso: *Single-molecule m6A detection empowered by endogenous labeling unveils complexities across RNA isoforms*, *Mol Cell* 2025, doi:10.1016/j.molcel.2025.01.014 (Jinkai Wang's group; bioRxiv 10.1101/2024.01.30.577990) | "Relatively weak but nonnegligible" single-molecule linkage between m6A sites below 200 bp, fading with distance, gone beyond 1 kb; 39.1% of sites have another within 50 bp. The biological basis for corroboration and for a distance-decaying neighbour weight | Verified (numbers). **First author not confirmed**: cite by title and DOI |
| DeepRM: Kang, Hwang & Baek, *Nat Commun* 2025, doi:10.1038/s41467-025-67417-w | (a) "4819 pairs of significantly co-occurring m6A sites", e.g. 40.0% of reads vs 21.8% expected: co-occurrence on the same molecule. (b) Signal changes "up to ±10 nts from m6A"; two m6As within 20 nt "interfere". Motivates our 0-10 / 10-20 nt band check | Verified (PMC full text) |
| Nanocompore: Leger et al., *Nat Commun* 2021, doi:10.1038/s41467-021-27393-3 | ~5 nucleotides sit in the R9 pore at once, so one modification shifts several overlapping k-mers. Why each site's input covers three overlapping 5-mers | Verified |

## 2. Nanopore m6A detection: the methods we position against

None of these shares evidence between candidate sites on a transcript: each
scores a site from its own reads, within a ≤9-21 nt window.

| Work | What it does / what we use | Status |
|---|---|---|
| m6Anet: Hendra et al., *Nat Methods* 2022, doi:10.1038/s41592-022-01666-1 | Multiple-instance learning over a site's reads. **Trained on SG-NEx HCT116 with m6ACE-seq labels** (dataset0's cell line and assay), DRACH only, 20 reads sampled per site. HEK293T test: ROC AUC 0.83, PR AUC 0.35. Claims to generalise across cell lines "without a loss in accuracy", tested with labels from a similar assay in both lines | Verified (PMC full text) |
| m6ATM, *Brief Bioinform* 2024, 25(6):bbae529 | WaveNet encoder + multiple-instance learning, per site; its authors note RNA structure is not used | Verified (abstract) |
| DeepRM (above) | Transformer over one molecule's signal and a 21-nt context; per site | Verified |
| Xron, *Genome Res* 2024 (bioRxiv 10.1101/2024.01.06.574484) | Methylation-aware basecaller; an HMM within a read, not across sites | Verified (abstract) |
| CHEUI, *Nat Commun* 2024, doi:10.1038/s41467-024-47953-7 | 9-mer per-site model; m6A-m5C co-occurrence analysed after prediction | Reported |
| TandemMod, *Nat Commun* 2024, doi:10.1038/s41467-024-48437-4 | Per-read 5-mer features, aggregated per site | Reported |
| SingleMod, *Nat Commun* 2025, doi:10.1038/s41467-025-60447-4 | Multiple-instance regression per site; describes m6A as "additive, position-independent" | Reported |
| ORCA, *Nat Commun* 2026, doi:10.1038/s41467-026-68419-y | 9-nt Bi-LSTM per site, with domain-adversarial training; neighbouring-site interplay analysed after prediction | Reported |
| MINES: Lorenz et al., *RNA* 2020, doi:10.1261/rna.072785.119 | Random forest per DRACH site | Reported |
| MultiNano (bioRxiv 10.1101/2025.08.04.668591), RNANO (bioRxiv 10.1101/2025.03.01.640267) | Per-site multiple-instance models | Reported |
| NanoFM (github.com/zhangjun640/NanoFM) | Nanopore signal + structRFM embeddings + cross-attention. The repo exists; it gives no paper, training data or results. A claimed DOI (10.1016/j.ijbiomac.2026.152629) resolves but could not be read | Unverified (repo only) |
| structRFM (bioRxiv 2025, 10.1101/2025.08.06.668731) | Structure-guided RNA foundation model | Verified (preprint) |
| m6A-IIN: Li et al., *Commun Biol* 2025, 8:1022, doi:10.1038/s42003-025-08265-8 | Sequence-only: a 41-nt fragment plus RNAfold structure; no nanopore signal; one fragment per prediction | Verified (PMC) |
| M6A-SAI (PeerJ), SMART-m6A (PLOS Comput Biol) | Sequence + structure m6A predictors | Reported (SMART-m6A's year conflicts with its DOI: check) |

## 3. Prior art for using neighbouring sites (limits our novelty claim)

| Work | Relation to our model | Status |
|---|---|---|
| **DeepMod**: Liu et al., *Nat Commun* 2019, 10, doi:10.1038/s41467-019-10168-2 | **Closest precedent.** Nanopore DNA 5mC, two stages. (1) A 3-layer bidirectional LSTM reads one read's signal around a cytosine and calls that read: the analogue of our read encoder. (2) An optional "cluster" network re-estimates a CpG's methylation fraction from 14 numbers: its own fraction, its partner's, the number of CpGs within ±25 bp, and an 11-bin histogram of their fractions. **The neighbour stage is a histogram aggregation, not an LSTM.** See [DeepMod vs our model](#deepmod-vs-our-model). **Must cite** | Verified: stage 1 (paper; DeepMod2 comparison); stage 2 input from `DeepMod_tools/hm_cluster_predict.py`. Stage 2's own architecture is not documented (loaded from a checkpoint). The claimed +1-3% AP / +3-5% AUC gain is **unverified** |
| DeepCpG, *Genome Biol* 2017, doi:10.1186/s13059-017-1189-z | Imputes single-cell CpG methylation from neighbouring CpG states | Reported |
| CpG Transformer, *Bioinformatics* 2022, 38(3):597 | Sliding-window attention over neighbouring CpGs (imputation) | Reported |
| GraphCpG, *Bioinformatics* 2023 (btad533) | Graph over neighbouring CpG loci (imputation) | Reported |
| ccsmeth, *Nat Commun* 2023, 14:4054, doi:10.1038/s41467-023-39784-9; hifimeth (bioRxiv 10.1101/2024.08.14.607879) | PacBio 5mC: neighbouring CpGs' read-level calls feed a site's call | Reported |
| HEPeak (BMC Genomics 2015), MeTPeak (*Bioinformatics* 2016), BaySeqPeak (2018) | MeRIP-seq peak callers with HMMs across neighbouring bins: coarse region smoothing | Reported |

**Agreed novelty wording:** *"To our knowledge, the first method to
propagate read-derived evidence between candidate RNA modification sites
along a transcript in nanopore direct RNA sequencing, and to characterise how
such propagation behaves when the labels change between cell lines. Analogous
neighbour-site models exist for DNA methylation (DeepMod; DeepCpG; CpG
Transformer)."*

### DeepMod vs our model

**Correction (10 Oct):** DeepMod's LSTM is its per-read *signal* model (stage
1, like our read encoder). Its *neighbour* model (stage 2) aggregates a
histogram. So "GNN vs LSTM" is **not** the difference.

Both fit one template:

$$z_i = F\big(\text{own}_i,\ \mathrm{AGG}_{j\in\mathcal{N}(i)}\ \psi(\text{evidence}_j, d_{ij})\big)$$

| Component | DeepMod cluster model (DNA 5mC) | Ours (RNA m6A) |
|---|---|---|
| Neighbourhood | CpGs within ±25 bp, both strands | Candidate sites on the transcript within 50-150 nt |
| What a neighbour sends | A final fraction $q_j$ of reads already called methylated | A learned vector $s_j$ from all its reads, or its raw own-reads score $a_j$ plus $\log n_j$ |
| Distance | Ignored inside the window | Weight $e^{-d_{ij}/\lambda}$, row-normalised, λ ≈ 100 nt fitted to training labels |
| Message $\psi$ | Fixed: a one-hot bin of $q_j$ | Learned |
| Aggregation | Normalised histogram | Weighted mean (+ max within 75 nt) |
| Evidence strength | Lost (1/1 and 50/50 reads both give $q = 1$) | Kept: $n_j$ in messages, $n_i$ in the encoder and the gate |
| Combination $F$ | A network on 14 numbers (architecture undocumented) | $z_i = a_i + g_i\,\Delta_i\,\mathbb{1}[\deg_i > 0]$, which reduces exactly to $a_i$ with no neighbours |
| Hops | 1 | 2 (H2GCN design) or 1 (scalar) |
| Training | Two-stage, on fixed stage-1 outputs | End to end: site $i$'s loss trains neighbour $j$'s read encoder |
| Missing neighbours | Unit dropout (keep 0.7) | Neighbour dropout (edges and whole sites), our active ingredient |
| Setting | DNA CpGs: dense, paired strands, strongly correlated | RNA m6A: sparse, single-stranded, moderately clustered (2.2× within 25 nt); label shift between cell lines |

**Agreed wording:** "DeepMod's optional second stage aggregates the
methylation fractions of CpGs within ±25 bp into a histogram. Our model
differs in passing learned, read-derived messages; weighting neighbours by
distance; carrying read-count confidence; combining them through a residual
correction that reduces to the site's own score; training end to end with
neighbour dropout; and targeting RNA m6A under label shift between cell
lines."

## 4. Labels and agreement between assays

| Work | What we use | Status |
|---|---|---|
| Koh et al., m6ACE-seq, 2019 | The assay behind dataset0's labels (docs/project-requirements.md) | Verified (course brief) |
| GLORI: Liu et al., *Nat Biotechnol* 2023 | Absolute single-base m6A quantification; reportedly among the most accurate assays (HEK293T) | Reported |
| NP-mFinder (claimed *Front Genet* 2026, doi:10.3389/fgene.2026.1770769) | Against GLORI v2.0: 28% site-level vs 85% gene-level concordance. Site-level agreement between assays is low | Verified (abstract, via a mirror); venue not confirmed |
| m6AConquer (bioRxiv 10.1101/2024.09.10.612173; NAR) | Cross-technique reproducibility; treats nanopore (m6Anet) and m6ACE-seq as non-independent | Reported |
| DeepRM (above), HEK293T site counts | miCLIP2 36,556; m6ACE-seq 33,163; m6A-SAC-seq 12,234; GLORI 176,642: a ~14× spread across assays in one cell line | Reported (from the DeepRM paper; re-check the numbers) |

No paper was found that argues cross-cell-line accuracy is bounded by label
disagreement. Our label oracle and discordant-site analyses (findings 12D)
appear to be new as stated.

## 5. Machine-learning methods

Standard methods, cited for the idea each of our designs borrows. Venues and
years are from the papers' own records; claims are limited to what each
paper's title and abstract state.

| Work | Idea | Used in | Status |
|---|---|---|---|
| H2GCN: Zhu et al., *Beyond Homophily in Graph Neural Networks*, NeurIPS 2020 | Keep a node's own embedding separate from its neighbours' | Base architecture | Standard reference |
| DeepSets: Zaheer et al., NeurIPS 2017 | Permutation-invariant set encoder | Read encoder | Standard reference |
| GraphGPS: Rampášek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*, NeurIPS 2022 | Local message passing + global attention | `gps` control | Standard reference |
| SchNet: Schütt et al., NeurIPS 2017 | Continuous filters as a function of distance | Distance-weighted messages | Standard reference |
| Correct & Smooth: Huang et al., ICLR 2021 (arXiv:2010.13993) | Base predictor + propagated residual correction | `res_gate`, scalar messages | Verified (arXiv) |
| DropEdge: Rong et al., ICLR 2020; GRAND (DropNode): Feng et al., NeurIPS 2020 | Randomly remove edges and nodes in training | Neighbour dropout | Standard reference |
| GPR-GNN: Chien et al., ICLR 2021; ACM-GCN: Luan et al., NeurIPS 2022 | Learned weighting of neighbour information | The depth gate | Standard reference |
| PNA: Corso et al., NeurIPS 2020 | Several aggregators instead of one mean | Per-band means (`fk_band`) | Standard reference |
| Gulrajani & Lopez-Paz, ICLR 2021; V-REx: Krueger et al., ICML 2021; group DRO: Sagawa et al., ICLR 2020 | Domain generalisation; select on the worst domain | Ranking on the worse gain (decision 0032) | Standard reference |
| GOOD benchmark: Gui et al., NeurIPS 2022 | Graph out-of-distribution benchmark | Context for the capacity result | Verified (proceedings listing) |
| Niv & Rabin, *Exploring Graph-Transformer Out-of-Distribution Generalization Abilities*, arXiv:2506.20575 (2025) | GPS generalised better than message passing on 4 of 6 GOOD benchmarks, but "vGIN leads by roughly 2% on the size shift and 5% on the scaffold shift ... locality in message passing can be better suited for certain distribution shifts". So "capacity hurts out of distribution" is **not** a general law; our result is about label shift in this domain | Verified (arXiv full text) |
| Single-cell DNA methylation imputation benchmark, *Brief Bioinform* 2026, 27(4):bbag434 | Neighbour-reliant models (CpG Transformer, MambaCpG) collapse when the split separates neighbouring sites: "random splitting inevitably places highly similar neighboring sites in both training and test sets". **Our gene-grouped split keeps a site and its neighbours together** (worth a sentence in Methods) | Verified (PMC) |

## 6. Data and tools we used

| Source | Use | Status |
|---|---|---|
| SG-NEx (Singapore Nanopore Expression project), public bucket `s3://sg-nex-data`; github.com/GoekeLab/sg-nex-data; *Nat Methods* 2025 | Source of dataset0 (HCT116 replicate 3 run 1, matched exactly: findings 13), the extra HCT116 runs, and the Task 2 cell lines | Verified (bucket listing; exact match) |
| ENA PRJEB44348 (SG-NEx HCT116), PRJEB40872 (HEK293T + METTL3 knock-out), PRJEB32782 (Arabidopsis), GEO GSE124309 (curlcake) | Accessions named in the m6Anet paper, for possible external tests | Verified (as given in m6Anet) |
| nanopolish eventalign; m6Anet dataprep | Produce the course's input format | Verified (course brief, m6Anet) |
| RNAplfold (ViennaRNA); LinearPartition | Folding test (findings 7) | Tools used directly |
| Ensembl 91 | Gene-structure analysis only, never a model input (fig_gene_structure) | Used directly |

---

## Errors found in AI-generated literature

Caught during verification, kept as a record (some are candidates for the
report's AI-use table: findings section 15):

- **A first summary (9-10 Oct):**
  - described "m6A-IIN" as a graph wavelet network: wrong; it is an
    invertible network on sequence + structure;
  - presented NanoFM as a published paper with results: not verifiable;
  - we initially failed to find m6A-IIN and called it unverified; it exists.
- **Deep research report 1 (10 Oct):**
  - EpiNano as "Science 2019" (Nat Commun 2019);
  - GraphGPS as "Dwivedi et al., ICLR 2022" (Rampášek et al., NeurIPS 2022);
  - SMART-m6A's year conflicts with its DOI.
- **Earlier notes:** m6Aiso cited as "Guo et al. 2025"; the first author is
  not confirmed.

## Search record

- **Citation checks for the 8 Oct design motivations (12B):** web search and
  full-text reads of PMC/publisher pages, 8 Oct.
- **Nanopore m6A tools and prior art:** web searches and reads, 9-10 Oct, plus
  two ChatGPT deep-research reports (10 Oct).
  - Report 2's search log has 26 queries over a general web index.
  - It flags as not searched: RECOMB/ISMB proceedings, NeurIPS/ICML/ICLR
    workshops, theses, Zenodo.
  - Re-run before submission: Google Scholar `"nanopore" "m6A" "graph neural
    network"` and ISMB/RECOMB 2024-2026 abstracts.
- **Not individually checked:** xPore, EpiNano, ELIGOS, Tombo, nanom6A,
  Dorado/Remora, modkit (believed per-read/per-site; unverified).
