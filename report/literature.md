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
- **Deep research report 1, found on re-reading its full text (16 Oct):**
  - CpG Transformer credited to "Clauwaert et al."; the first author is De
    Waele (De Waele, Clauwaert, Menschaert & Waegeman, *Bioinformatics* 2022);
  - EpiNano credited to "Pavanucci et al."; not the paper's authorship;
  - claims m6Anet "trained on HEK293T predicts HCT116 with no accuracy drop".
    Unverified, and the direction we reproduced is the reverse (trained on
    HCT116, scored on HEK293T: 0.356 / 0.819 vs the published 0.35 / 0.83);
  - attributes to Hendra et al. the point that "a model cannot exceed the
    overlap of labels"; a paraphrase we could not find in the paper.

## Search record

- **Citation checks for the 8 Oct design motivations (12B):** web search and
  full-text reads of PMC/publisher pages, 8 Oct.
- **Nanopore m6A tools and prior art:** web searches and reads, 9-10 Oct, plus
  two deep-research reports (10 Oct): report 1 by ChatGPT, report 2 by Claude.
  Full logs below.
  - Report 1's search log has 31 strings (Google/Bing, PubMed/PMC, arXiv,
    GitHub).
  - Report 2's search log has 26 queries over a general web index.
  - It flags as not searched: RECOMB/ISMB proceedings, NeurIPS/ICML/ICLR
    workshops, theses, Zenodo.
  - Re-run before submission: Google Scholar `"nanopore" "m6A" "graph neural
    network"` and ISMB/RECOMB 2024-2026 abstracts.
- **Not individually checked:** xPore, EpiNano, ELIGOS, Tombo, nanom6A,
  Dorado/Remora, modkit (believed per-read/per-site; unverified).

### Deep research report 1 (ChatGPT, 10 Oct): full search log and findings

Recorded 16 Oct from the report's text. Everything below is the **report's**
statement, status **Reported** unless a table above says otherwise; its
known errors are listed under [Errors found](#errors-found-in-ai-generated-literature).

**Sources searched:** Google/Bing, PubMed/PMC, arXiv, GitHub. Full text read
where accessible; abstracts otherwise (marked per row below).

**Search strings (31), by question:**

| Question | Strings (result where the report noted one) |
|---|---|
| Q1. Nanopore RNA callers that use other sites (14) | `"m6Anet nanopore RNA modifications"`; `"m6ATM nanopore direct RNA sequencing"`; `"DeepRM nanopore m6A single-molecule"`; `"CHEUI nanopore m6A m5C"` (no accessible full text); `"xPore differential modifications nanopore"`; `"Nanocompore direct RNA m6A"`; `"EpiNano nanopore m6A caller"`; `"ELIGOS nanopore RNA modifications"`; `"Tombo nanopore modification detection"`; `"MINES nanopore random forest"`; `"SingleMod nanopore m6A deep learning"`; `"TandemMod nanopore modifications"`; `"m6Aiso single-molecule nanopore"`; `"graph neural network RNA modifications nanopore"` (**no hits**) |
| Q2. Analogues in other modalities (7) | `"DeepCpG methylation RNN CpG neighbors"`; `"CpG Transformer single cell methylation imputation"`; `"nanopore 5mC methylation neighboring CpGs"` (no specific references); `"MeRIP-seq peak calling sliding window"`; `"SRAMP WHISTLE m6A prediction"`; `"SMART-m6A PLoS 2023"`; `"M6A-SAI structure graph predictor"` |
| Q3. Molecule-level co-occurrence (3) | `"DeepRM co-occurring m6A sites"`; `"m6Aiso distance-dependent linkages co-occurrence"` (press coverage only); `"m6A co-occurrence single-molecule nanopore"` (no hits beyond those two) |
| Q4. Cross-cell-line generalisation and concordance (5) | `"nanopore m6A cell line generalization"`; `"miCLIP vs m6ACE concordance sites"` (no direct hits); `"GLORI m6A-SAC eTAM overlap sites"` (no quick findings); `"inter-assay m6A concordance percentage"`; `"label noise m6A predictor transfer"` |
| Q5. ML methodology precedents (2) | `"graph neural network generalization distribution shift"` (no genomics-specific result); `"GNN neighbor dropout generalization"` (general robustness literature, e.g. DropEdge) |

**Methods the report examined, as it describes them:**

| Work | Category | Report's one-line relevance | Source the report read |
|---|---|---|---|
| m6Anet (Hendra et al., Nat Methods 2022) | per-site | MIL over a site's reads; 1-bp flanking features; no inter-site pooling | Full text (PMC) |
| m6ATM (Bi et al., Brief Bioinform 2024, bbae529) | per-site | WaveNet + MIL per site; no other sites | Full text |
| DeepRM (Kang et al., Nat Commun 2025) | per-site | Transformer over a 21-nt signal + sequence block per site; no cross-site modelling | Full text |
| MINES (Lorenz et al., RNA 2020) | per-site | Random forest on Tombo signal at each DRACH motif | Full text |
| TandemMod (Wu et al., Nat Commun 2024) | per-site | Multi-modification model per k-mer; no neighbour sites | Abstract only |
| DeepCpG (Angermueller et al., Genome Biol 2017) | DNA, neighbour CpGs | RNN over the 25 nearest CpGs' states + sequence CNN | Full text |
| CpG Transformer (Bioinformatics 2022; report misnames the first author) | DNA, neighbour CpGs | Self-attention over a CpG x cell matrix | Full text |
| SMART-m6A (PLoS Comput Biol; year conflicts with DOI) | sequence + structure | Sequence/structure graph, 71-nt context; no measured signal | Full text |
| Window averaging (DeepCpG's "WinAvg" baseline) | DNA, neighbour CpGs | Average of neighbours in a fixed window | Full text |

**The report's verdicts:**
- Q1: no nanopore RNA modification caller found that uses measurements at
  other candidate sites; CHEUI, xPore, Nanocompore, EpiNano, ELIGOS, Tombo,
  nanom6A, SingleMod, m6Aiso and modkit described as per-site (for most of
  these the report read no full text: see "Not individually checked" above).
- Q2: DNA methylation imputation (DeepCpG, CpG Transformer) uses neighbouring
  CpGs, but on bisulfite / single-cell data and to impute rates; MeRIP peak
  callers smooth over windows but not at single-site resolution;
  sequence-only m6A predictors use no measured signal.
- Q3: co-occurrence documented (DeepRM's 4,819 pairs; m6Aiso), but no method
  found that uses it as a prior.
- Q4: label concordance across assays is low; see the errors list for two
  claims here we could not verify.
- Q5: no precedent found for testing GNN expressivity under label shift in
  genomics, or for a residual own-score + neighbour correction with neighbour
  dropout for transfer.

### Deep research report 2 (Claude, 10 Oct): full search log and findings

Recorded 16 Oct from the report's text. Everything below is the **report's**
statement, status **Reported** unless a table above says otherwise. Unlike
report 1, it quotes the passages it relied on and lists what it could not
verify (see its caveats at the end).

**How it searched:** one general web-search tool indexing PubMed/PMC,
bioRxiv/medRxiv, arXiv, Nature, OUP, Cell, Springer and GitHub (not native
database queries). 18 searches by the main agent, 8 by a subagent; 3 more
planned but not run.

**Search strings, main agent (18):**
1. m6A-IIN Communications Biology 2025
2. DeepRM nanopore m6A co-occurring sites single molecule
3. m6Aiso nanopore m6A isoform linkage distance
4. graph neural network nanopore direct RNA sequencing modification neighbouring sites
5. m6ATM nanopore m6A transcriptome multiple instance learning
6. NanoFM zhangjun640 nanopore modification
7. CpG Transformer imputation single-cell methylation neighbouring CpG sites
8. DeepCpG neighbouring CpG sites methylation states cells prediction
9. nanopore 5mC calling neighboring CpG methylation states read-level correlation model
10. BSmooth local likelihood smoothing methylation neighbouring CpGs
11. MeRIP-seq peak calling hidden Markov model neighboring bins m6A
12. m6A sites overlap cell lines miCLIP m6ACE-seq GLORI concordance HEK293T HepG2
13. Correct and Smooth label propagation residual error correction graph neural network
14. graph transformer worse out-of-distribution generalization than message passing GNN benchmark
15. m6Anet generalises cell lines HEK293T K562 A549 m6ACE-seq AUC ROC PR
16. benchmark m6A detection methods low overlap miCLIP m6ACE-seq GLORI same cell line orthogonal
17. SingleMod m6A single molecule nanopore deep learning precise
18. "graph" m6A nanopore direct RNA site-level prediction GNN transcript neighboring m6A sites

**Subagent (8):** "NanoFM nanopore RNA modification bioRxiv"; "zhangjun640
NanoFM github"; "\"NanoFM\" nanopore modification"; "NanoFM foundation model
nanopore RNA modification github"; "graph neural network nanopore direct RNA
m6A detection"; "nanopore RNA modification neighboring sites hidden Markov
model transcript graph neural network m6A pseudouridine"; "TandemMod nanopore
multiple RNA modifications transfer learning"; "DeepMod cluster model
neighboring CpG nanopore Nature Communications".

**Planned, not run (budget):** "github NanoFM nanopore RNA modification
foundation model"; "M6A-SAI graph m6A prediction sequence similarity graph";
"GLORI m6A clustered sites distance neighboring m6A stoichiometry".

**Fetches:** NanoFM repository rejected by the tool; ORCA full text read
(first half); m6Anet PMC blocked by reCAPTCHA; DeepRM PMC rate-limited.

**Works examined** (category: B = per-site model over a local signal window;
DNA = DNA methylation; Q4/Q5 = concordance / ML methodology):

| Work | Year, venue | ID | Cat. | Report's relevance | What it read |
|---|---|---|---|---|---|
| m6Anet (Hendra et al.) | 2022 Nat Methods | 10.1038/s41592-022-01666-1 | B | Per-site MIL over reads; claims cross-cell-line generalisation | abstract + snippets |
| m6ATM | 2024 Brief Bioinform | bbae529 | B | WaveNet + dual-stream MIL, per site | abstract/snippets |
| DeepRM | 2025 Nat Commun | 10.1038/s41467-025-67417-w | B | Transformer per molecule; post-hoc 4,819 co-occurring pairs | snippets |
| CHEUI | 2024 Nat Commun | 10.1038/s41467-024-47953-7 | B | 9-mer per site; m6A-m5C co-occurrence; low orthogonal overlap | snippets |
| m6Aiso | 2025 Mol Cell | PMID 39922195 | B | Single-read model; linkage D' decays with distance | abstract + snippets |
| Nanocompore | 2021 Nat Commun | 10.1038/s41467-021-27393-3 | B | Combines p-values of neighbouring k-mers (one modification's footprint) | snippets |
| SingleMod | 2025 Nat Commun | 10.1038/s41467-025-60447-4 | B | Multiple-instance regression per site; "additive, position-independent" m6A | snippets, README |
| TandemMod | 2024 Nat Commun | 10.1038/s41467-024-48437-4 | B | 5-mer read-level features aggregated to site | full text (subagent) |
| Xron | 2024 Genome Res | bioRxiv 10.1101/2024.01.06.574484 | B | m6A-aware basecaller; HMM over k-mers within a read | snippets |
| MultiNano | 2025 bioRxiv | 10.1101/2025.08.04.668591 | B | Multi-view per-site MIL | snippets + README |
| NanoFM | - | github.com/zhangjun640/NanoFM | not verified | Repository could not be opened or found | nothing |
| RNANO | 2025 bioRxiv | 10.1101/2025.03.01.640267 | B | Attention-MIL, several modifications | snippets |
| ORCA | 2026 Nat Commun | 10.1038/s41467-026-68419-y | B | 9-nt Bi-LSTM; neighbouring-site interplay analysed post hoc; domain-adversarial training | full text (first half) |
| MINES | 2020 RNA | 10.1261/rna.072785.119 | B | Random forest per DRACH site | abstract |
| Yanocomp | 2021 bioRxiv | 10.1101/2021.06.15.448494 | B | Comparative GMM; per-read calls | snippets |
| Zhong et al. tool comparison | 2023 Nat Commun | 10.1038/s41467-023-37596-5 | Q4 | Low inter-tool overlap; motif-dependent recall | snippets |
| m6A-IIN (Li, Su, Hu et al.) | 2025 Commun Biol 8:1022 | 10.1038/s42003-025-08265-8 | sequence-only | Sequence + structure invertible network; not nanopore | abstract + snippets |
| DeepMod (Liu et al.) | 2019 Nat Commun | 10.1038/s41467-019-10168-2 | DNA (nanopore) | Second network on a CpG's and its neighbours' predicted methylation | full text (subagent) |
| ccsmeth (Ni et al.) | 2023 Nat Commun 14:4054 | 10.1038/s41467-023-39784-9 | DNA (PacBio) | Site call uses read-level probabilities of 10 adjacent CpGs | snippets |
| hifimeth | 2024 bioRxiv | 10.1101/2024.08.14.607879 | DNA (PacBio) | Uses <250 bp CpG state correlation for single-molecule calls | snippets |
| DeepCpG | 2017 Genome Biol | 10.1186/s13059-017-1189-z | DNA imputation | Neighbouring CpG states within/across cells | snippets |
| CpG Transformer | 2022 Bioinformatics 38(3):597 | PMC8756163 | DNA imputation | Sliding-window attention over neighbouring CpGs | snippets |
| GraphCpG | 2023 Bioinformatics | btad533 | DNA imputation | Graph over neighbouring loci | snippets |
| scMeFormer | 2025 Cell Genomics | PMC11960545 | DNA imputation | Transformer over neighbouring CpGs | snippets |
| scDNAm imputation benchmark | 2026 Brief Bioinform 27(4):bbag434 | PMC13464701 | DNA (Q5) | Chromosome split collapses neighbour-reliant models | snippets |
| BSmooth | 2012 Genome Biol 13:R83 | 10.1186/gb-2012-13-10-r83 | DNA smoothing | Local-likelihood smoothing across CpGs | snippets |
| Nanopore DNA methylation-caller survey | 2021 | PMC8524990 | DNA | Nanopolish/Tombo "borrow signals of neighboring CpG sites" | snippets |
| HEPeak | 2015 BMC Genomics 16(S4):S2 | 10.1186/1471-2164-16-S4-S2 | MeRIP | HMM across adjacent bins | snippets |
| MeTPeak | 2016 Bioinformatics 32(12):i378 | 10.1093/bioinformatics/btw281 | MeRIP | HMM for read dependency across a site | snippets |
| BaySeqPeak | 2018 Quant Biol | 10.1007/s40484-018-0149-2 | MeRIP | ZINB + HMM spatial dependency | snippets |
| FET-HMM | 2015 BioMed Res Int | 10.1155/2015/852070 | MeRIP | HMM over adjacent bins | snippets |
| TRES | 2021 Bioinformatics 37(18):2818 | (DOI unverified) | MeRIP | Describes HMM callers' local dependency | snippets |
| NP-mFinder | 2026 Front Genet | 10.3389/fgene.2026.1770769 | Q4 | 28% site / 85% gene concordance with GLORI v2.0 | snippets |
| m6AConquer | 2024 bioRxiv / NAR | 10.1101/2024.09.10.612173 | Q4 | Cross-technique reproducibility; m6Anet not orthogonal to m6ACE-seq | snippets |
| miCLIP2 / m6Aboost | 2021 NAR 49(16):e92 | - | Q4 | Cross-cell-line overlap depends on expression | snippets |
| m6A-isoSC-seq | 2025 Nat Commun | 10.1038/s41467-025-60869-0 | Q4 | Cell-line specificity of isoform methylation | snippets |
| m6Astorm | 2026 Brief Bioinform 27(2):bbag132 | PMC13043019 | Q4 | Claims generalisation to new cell lines (GLORI recall 93% in HEK293T) | snippets |
| ELIGOS2 / IVT basecalling | 2023 bioRxiv | 10.1101/2023.11.28.568965 | Q4 | 76% of IVT-model sites had orthogonal support | snippets |
| Correct & Smooth (Huang et al.) | ICLR 2021 | arXiv:2010.13993 | Q5 | Base prediction + propagated residual correction | snippets |
| NLCS | 2023 arXiv / ACM | arXiv:2310.05757 | Q5 | Nonlinear C&S; residual correction under heterophily | snippets |
| GT OOD generalisation (Niv & Rabin) | 2025 arXiv | arXiv:2506.20575 | Q5 | GPS more robust OOD in 4/6 GOOD tasks | snippets |
| Attending to Graph Transformers | 2023 arXiv | arXiv:2302.04181 | Q5 | "GTs may generalize poorly to larger graphs" | snippets |
| Graph Transformers survey | 2024 arXiv | arXiv:2407.09777 | Q5 | "often struggle to generalize to ... out-of-distribution graphs" | snippets |

**The report's verdicts:**
- **Q1 (nanopore RNA):** no detector found that uses measured evidence from
  other candidate sites on the transcript as an input to a site's
  prediction, by graph message passing, CRF/HMM over sites, smoothing or
  hierarchical sharing. All checked tools are per-site models over a <= 9-nt
  signal window; co-occurrence (DeepRM, m6Aiso, CHEUI, ORCA, Nanocompore) is
  analysed only after prediction. Nanocompore's pooling of one
  modification's footprint over neighbouring k-mers is the closest RNA
  "spatial smoothing" a reviewer may raise.
- **Q2 (other data):** strong precedent in DNA methylation. DeepMod's
  cluster network (nanopore 5mC) is the closest, and anticipates our scalar
  messages variant; ccsmeth and hifimeth (PacBio) use neighbouring CpGs;
  DeepCpG, CpG Transformer, GraphCpG and scMeFormer impute from neighbouring
  states; BSmooth and HMMs smooth. MeRIP peak callers (HEPeak, MeTPeak,
  BaySeqPeak, FET-HMM) chain adjacent bins. Suggested defensible claim: the
  first to propagate read-derived evidence between candidate RNA
  modification sites along a transcript in nanopore direct RNA sequencing,
  and to characterise such propagation under cross-cell-line label shift.
- **Q3 (co-occurrence):** documented (DeepRM 4,819 pairs; m6Aiso linkage
  below 200 bp, fading by 1 kb); used as a prior in RNA by no one found; only
  hifimeth does so, for DNA. SingleMod's "additive, position-independent"
  finding is a counterpoint.
- **Q4 (transfer and concordance):** m6Anet reports cross-cell-line
  generalisation "without a loss in accuracy", but with similar-assay labels
  in both lines. Cross-assay site concordance is low (NP-mFinder 28% site,
  85% gene vs GLORI v2.0; CHEUI's low CLIP overlap; assay site counts in
  HEK293(T) spanning 12,234 to 176,642). No paper found that bounds
  cross-cell-line accuracy by label disagreement.
- **Q5 (ML):** evidence that graph transformers generalise worse out of
  distribution is mixed (GPS more robust in 4/6 GOOD tasks). The scDNAm
  benchmark shows neighbour-reliant models collapse under a chromosome
  split and warns against site-level splits. Correct & Smooth is the
  conceptual ancestor of the residual design.

**Its own caveats (limits on what it supports, not errors):**
- about 26 searches; not a systematic review; RECOMB/ISMB, ML workshops,
  theses and Zenodo not searched;
- not verified: NanoFM; xPore, EpiNano, ELIGOS, Tombo, nanom6A,
  Dorado/Remora, modkit internals; SRAMP, WHISTLE, DeepM6ASeq, M6A-SAI,
  SMART-m6A; TRES's DOI;
- venues of m6Astorm and the scDNAm benchmark, and DeepRM's GLORI site count
  (176,642), came from an enrichment pass, not its own reading;
- most quotes come from search-result extracts, not complete readings;
- counter-evidence to state in the paper: graph transformers were more
  robust out of distribution in most GOOD tasks, and m6Anet reports no
  cross-cell-line loss;
- it recommends the phrasing "to our knowledge"; the report instead keeps a
  firmer claim, stated together with the search channels and their limits
  (decided 15 Oct).
