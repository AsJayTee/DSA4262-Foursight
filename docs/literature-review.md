<!-- Provenance banner: this file is not like the rest of docs/. -->

# Literature review: nanopore m6A methods and what they do with reads

**Where this came from.** ChatGPT deep research, 2026-09-23, run against a brief
describing our data, harness and results. **It is external output, not a
measured result from this repo**, and it is kept here because the alternative
was losing it in a chat window. Treat every claim as a lead, not a fact.

**What was independently verified against this repo** (all correct):

| claim | check |
|---|---|
| m6Anet's m6ACE prevalence 5,579 / (5,579 + 121,853) = 4.38% | arithmetic |
| our lifts 0.4844/0.0449 = 10.79x and 0.2593/0.0449 = 5.78x | arithmetic |
| the noisy-OR cardinality table, 1-(1-q)^N | arithmetic |
| the 7-mer is the union of three overlapping 5-mers at -1/0/+1 | agrees with [data.md](data.md), independently |
| class weight w = 21.25, logit(p_w) = logit(p) + log(w) | arithmetic; explains our calibration overcount |
| every one of our own numbers it quotes | matches W&B |

**What could NOT be verified:** the citations themselves. The ORCA (2026) and
SWARM references are the load-bearing ones for its depth-robustness argument and
for the co-modification claim behind the GNN discussion. Check them before
relying on either.

**A premise in Part 1 is wrong, and Part 2 of the exchange corrects it.** Part 1
was written believing low read depth was our primary target. It is not - the
graded evaluation is on data with >= 20 reads per site. That invalidates its
argument against set-structured architectures, which rested on "at N=1 there are
no read-read interactions". At our median depth of 47 there are plenty.

**Part 2 is not in this file.** The follow-up exchange - which re-ranks the
architectures for the >= 20 regime, proposes a contextual Deep Set, and explains
the calibration overcount - exists only in a Claude Code chat transcript. Its
conclusions are summarised in the experiment queue under Modelling in
[../GAPS.md](../GAPS.md). Paste the full reply in below as "Part 2" if you still
have it.

---

## Part 1: what the published methods actually do

## Bottom line

The most important conclusion from the literature is that **your deployment regime is not a regime the standard m6A callers have really solved**.

The closest published analogue to your formulation is **m6Anet**: it explicitly casts site-level m6A detection as multiple-instance learning with latent per-read modification states. But its practical training/evaluation protocol largely avoids the cardinality problem you face: it samples **20 reads per site during training**, evaluates only sites with at least 20 reads, and averages predictions across repeated 20-read samples at inference. citeturn8view0turn23view4 CHEUI's site-level stage likewise defaults to a **20-read minimum**, while DENA explicitly filters sites to **at least 50 reads**. citeturn22view9turn23view13 The much newer ORCA paper, published in 2026 and notable because it deliberately simulates both sequencing depth and modification stoichiometry during training, still says that reliable inference is difficult below **10 reads**. citeturn21view6turn22view6

That makes your A549/K562/Hct116 deployment distribution—median depth 2–3, roughly 90% of sites below 20—not a mild extrapolation from the literature. It is substantially outside the validated operating region of most site callers.

A second important conclusion is architectural: **the field has surprisingly little evidence that sophisticated set aggregation is the key missing ingredient**. m6Anet tested attention and gated-attention pooling and did not find a statistically significant improvement over its simpler noisy-OR pooling. citeturn8view1 Most subsequent methods do one of three simpler things: classify reads independently and summarize them, fit an exchangeable mixture model to the read distribution, or aggregate the reads into fixed site-level statistics before a neural network. CHEUI, DENA, xPore, RedNano and ORCA all fall broadly into those patterns rather than using a modern Deep Sets/Set Transformer/Perceiver-style learned set representation. citeturn22view0turn22view6turn22view8turn23view11

So your finding that **depth augmentation, rather than merely changing the model family, is doing most of the work** is quite consistent with what the literature now suggests. ORCA independently arrives at essentially the same high-level idea—simulate modification fractions *and sequencing depths* during training—although its representation and labels are very different from yours. citeturn22view6

There is also a striking negative result from the literature for your use case:

> I did not find a primary study among the major methods that convincingly demonstrates **site-level biological-ground-truth m6A classification at 1–3 reads** under a weak-label setting like yours.

CHEUI can emit individual-read calls; DENA and RedNano have read-level models; TandemMod uses single-molecule predictions. But “the software emits a prediction from one molecule” is not the same claim as “a one-read site prediction has been validated and calibrated against endogenous site labels.” Their training designs usually obtain much stronger read-level supervision from synthetic controls, knockout/deficient conditions, or fully modified material. citeturn23view11turn22view8turn7view4

Against that literature, your depth-augmented LightGBM result—**PR AUC 0.2593 at one read**, versus your 0.1537 motif-only floor—is actually an important baseline, not something I would casually expect a published neural architecture to beat.

## What the established methods actually do

There are really four generations of approaches here, and lumping them all together as “nanopore m6A models” obscures major differences.

| Method | What is actually modeled | How a variable number of reads is represented | Low-coverage story | How relevant to your formulation |
|---|---|---|---|---|
| **Nanopolish / eventalign** | Aligns raw nanopore events to reference k-mers and produces signal/event measurements | **It does not represent the site as a set at all.** It is preprocessing that emits per-read/per-position events | Not a site classifier, so no intrinsic solution | Very relevant as feature extraction; not an architectural baseline |
| **Tombo** | Resquiggles signal to reference and applies statistical tests/deviation scores against canonical signal | Per-read evidence is statistically aggregated rather than learned as a set embedding | Statistical evidence inevitably weakens with few observations; benchmarks show coverage affects nanopore modification callers substantially | Useful conceptual baseline, but not a learned MIL architecture |
| **EpiNano** | Uses basecalling-error/current-derived features and an SVM to classify sites | The reads are collapsed into **site-level frequencies/statistics before classification** | Original paper explicitly reports lower accuracy at low coverage | Very similar in spirit to your summary-statistic models |
| **xPore** | Generative model of the distribution of nanopore signal, with latent modified/unmodified components and condition-specific modification fractions | Reads enter as **exchangeable observations in a mixture model**; no arbitrary read order and no fixed-size neural set embedding | Variable \(N\) is mathematically natural, but inference needs enough observations to resolve mixture components; designed mainly for differential modification across samples | Extremely relevant conceptually for latent stoichiometry, less directly applicable to your single-site binary task |
| **DENA** | Bi-LSTM trained on per-read event features around candidate motifs | Classifies individual read windows; site modification rate is derived downstream from the read calls | Published analyses impose **≥50 reads** to reduce depth effects and false positives | Useful read encoder precedent, poor precedent for your deployment depth |
| **m6Anet** | Explicit MIL: latent read states, observed site label | Shared per-read MLP followed by **noisy-OR**; attention variants were also tested | Training subsamples **20 reads/site**; inference repeatedly samples reads; evaluation candidate sites have ≥20 reads | By far the closest conceptual match to your weak-label problem |
| **CHEUI** | Two-stage deep network: first individual-read modification probability, then site probability/stoichiometry | Model 2 consumes the collection of grouped model-1 read predictions | Default site analysis requires **20 reads** | Relevant two-stage design; again avoids your 1–3-read regime |
| **RedNano** | Deep residual models combining raw-current and basecalling-error information | Primarily learns richer **per-molecule representations**, then obtains site evidence from the read predictions | Strong reported results are not compelling evidence about ultra-low depth; some synthetic evaluations had enormous coverage | Relevant for multimodal per-read representation, not evidence for fancy set aggregation |
| **TandemMod** | Deep single-molecule modification model trained using experimentally controlled modified/unmodified material, with transfer learning | Independent molecule predictions, subsequently summarized at sites | Much stronger read supervision than you possess | Architecturally interesting, but its training problem is fundamentally easier |
| **ORCA** | Site-level modification detection and annotation from aggregate signal/base variability, with adversarial/domain-generalized learning | **Reads at a genomic position are aggregated into variability features**, then sequence-context features go through a dual-layer Bi-LSTM | Explicitly trains over simulated depths and stoichiometries, yet says **<10 reads remains challenging** | The most relevant recent precedent for deliberate depth robustness |
| **SWARM** | Current successor to CHEUI for m6A/m5C/Ψ and RNA002/RNA004 | Single-read and transcriptomic-site models | Current software documentation claims improved CHEUI accuracy, but it does not establish a 1–3-read site-level benchmark comparable to yours | Worth watching, but I would not treat it as evidence that your depth problem is solved |

The details behind those summaries matter.

### Nanopolish is mostly infrastructure here, not a competing classifier

This distinction is easy to lose because papers repeatedly say they “use Nanopolish.” m6Anet runs `nanopolish eventalign` to map signal to reference and derives event mean, standard deviation and dwell-time features. CHEUI likewise explicitly requires Nanopolish event alignment as preprocessing. citeturn8view0turn20view0

So I would **not** put “Nanopolish” in the same architectural comparison as m6Anet. For your purposes it is closer to the machinery that produced your nine measurements than to the classifier sitting above them.

That is important because some apparent literature differences between “Nanopolish-based methods” are actually differences in what is done *after* almost the same event-aligned signal representation.

### EpiNano is fundamentally an aggregate-before-classification approach

The original EpiNano work established that m6A perturbs basecalling-derived quantities such as mismatch/deletion behavior, base quality and current-related features, and trained SVMs on those site-level features. citeturn23view2turn23view14 Most importantly for you, the authors explicitly state that **low read coverage decreases accuracy**. citeturn23view15

Conceptually, this is close to your LightGBM pipeline: estimate population statistics from the molecules, then classify the resulting site vector.

That means your experiment already tests a substantially stronger modern version of the basic EpiNano representation idea. Your quantiles, IQR/tails, motif and nonlinear boosted trees provide much more expressive aggregate statistics than an old SVM. Given your +0.0164 gain from extensive summary-statistic engineering and the persistent low-depth collapse, I would not expect revisiting an EpiNano-style aggregate feature family to open another large performance frontier.

### xPore is statistically elegant because it treats the reads as exchangeable samples

xPore is quite different. It models the nanopore current distribution at a site with latent signal components corresponding to molecular states and estimates modification fractions across samples. It is principally a method for **differential RNA modification** rather than a binary single-condition site caller. citeturn22view0

From a machine-learning perspective, though, xPore is highly relevant to your architecture discussion because it makes the right symmetry assumption: the reads are **iid/exchangeable molecules**, not a sequence. There is no read ordering, adjacency or convolutional axis. Permuting molecules cannot alter the likelihood.

This is the clean probabilistic analogue of a permutation-invariant set network.

Its weakness for your deployment problem is statistical rather than architectural. A two-component distribution is hard to identify from one, two or three points. A model can use priors or shrinkage, but at \(N=1\) there is no empirical “mixture shape” to discover. That is an intrinsic information constraint, not something a Transformer solves.

That point will matter a lot when we discuss architectures: at depth one, **every set architecture degenerates to a single-read classifier plus site/sequence prior**. At depth two or three it has only a tiny amount of interaction information available.

### DENA learns much richer within-read structure, but then relies on substantial coverage

DENA moved from hand-crafted aggregate statistics toward learned per-read representations. Its training data contained per-read event matrices with mean, median, standard deviation, dwell time and base quality around candidate sites; it trains separate motif-specific **Bi-LSTM** models and then uses read-level predictions to quantify a site's modification rate. citeturn23view11

This is a sequence model because the recurrent axis is the **neighboring nucleotide/event positions within a molecule**, not the collection of molecules. That axis is physically defensible.

This is exactly the distinction you made about CNNs: DENA provides precedent for convolution/recurrent structure along **sequence**, but not across reads.

Its coverage policy is revealing. DENA says that, to reduce sequencing-depth effects and false positives, it retained sites supported by **at least 50 reads** for downstream calling. citeturn23view13 In other words, its strong read encoder did not make site-level inference depth agnostic.

That makes DENA poor evidence for “a Bi-LSTM will solve your low-depth problem,” even though it is good evidence that **joint multivariate structure within each molecule and across nearby nucleotide positions is worth learning**.

### m6Anet is almost your exact statistical problem—until the depth protocol

m6Anet is the most directly relevant paper by some margin. Its formulation explicitly distinguishes observed site label \(y_i\) from unobserved molecule labels \(y_{ij}\), exactly as in your description. Each read has a 15-dimensional vector: normalized signal mean, standard deviation and dwell time around three neighboring positions, plus learned sequence embeddings. citeturn11view0turn23view3

The read encoder is a small MLP. The original default aggregator is noisy-OR:

\[
P(Y_i=1\mid x_{i1},\ldots,x_{iN})
=
1-\prod_{j=1}^{N}\left(1-p_{ij}\right).
\]

The authors also tried attention and gated-attention pooling; these did not significantly outperform noisy-OR in their experiments. citeturn8view1

But there is a crucial qualification that I think is underappreciated when m6Anet is cited as a “variable-size MIL” model:

**the mathematical model accepts variable-size bags, but its experimental protocol intentionally normalizes bag size.**

During training, m6Anet samples **20 reads from each site**. During inference it performs repeated read sampling and averages predictions. The HEK293T benchmark also restricts candidate sites to those with at least 20 reads. citeturn8view0turn23view4

That is not a cosmetic implementation detail. Noisy-OR has very strong cardinality dependence. Suppose the encoder gives every molecule the same small false-positive probability \(q=0.01\):

\[
\begin{array}{c|c}
N & 1-(1-q)^N\\
\hline
1 & 0.010\\
3 & 0.0297\\
20 & 0.182\\
100 & 0.634
\end{array}
\]

Nothing about the molecules got more suspicious; only the bag got bigger.

With \(q=0.05\), noisy-OR rises from 0.05 at one molecule to 0.642 at 20 and 0.994 at 100.

So I interpret m6Anet's fixed-20 sampling as doing two jobs simultaneously:

1. making training computationally bounded; and
2. **removing cardinality as a nuisance variable from the pooling function**.

The second point is my inference from the published pooling equation and sampling protocol, rather than a claim the authors emphasize. citeturn11view1turn8view0

For your problem, where \(N=1\) versus \(N=100\) is precisely the deployment shift of interest, I would **not** transplant m6Anet's noisy-OR pooling unchanged.

Your existing depth augmentation is in one sense tackling a harder problem than the published m6Anet protocol.

### CHEUI turns single-read probabilities into a second site-level prediction

CHEUI's design is two-stage. Its first deep model predicts modification probability for individual molecules from signal around a 9-mer. Reads from the same transcriptomic site are then grouped and fed into a second model that produces site-level modification probability and a stoichiometry estimate. citeturn20view0turn22view8

That is appealing because it separates two questions:

\[
\text{What does this molecule look like?}
\quad\rightarrow\quad
\text{What does this population of molecules imply about the site?}
\]

But CHEUI again does not solve your low-depth deployment problem. Its released site-level interface defaults to `min_reads=20`. It estimates stoichiometry by treating sufficiently low model-1 scores as unmodified, sufficiently high scores as modified and ignoring an intermediate region. citeturn22view9

So CHEUI is strong evidence for **hierarchical read → site modeling**, but not for the proposition that such a hierarchy is robust at 1–3 molecules.

The CHEUI repository now points users to **SWARM**, saying it improves accuracy for RNA002 and provides m6A, m5C and pseudouridine models for both RNA002 and RNA004 chemistry at individual-read and transcriptomic-site resolution. citeturn22view8 I would treat SWARM as part of the current software frontier, but I would not infer from that documentation that its site calls have been demonstrated under your extreme low-coverage distribution.

### RedNano and TandemMod improve the read encoder more than the set encoder

The newer deep-learning papers increasingly invest modeling capacity in the **single molecule**.

RedNano combines raw-current information and basecalling-error information through residual neural networks and reports improvements over older methods on several datasets. But one of its very strong synthetic-data results comes from data with mean site coverage above 5,000 reads, which makes those headline scores nearly irrelevant to your 1–3-read question. citeturn7view3

TandemMod similarly focuses on learning high-quality single-molecule modification signatures, using experimentally generated modified/unmodified material and transfer learning. Its advantage is that the researchers can manufacture much cleaner per-molecule supervision than you have. citeturn7view4turn11view5

I therefore would not use either paper as evidence that “residual networks” or “deeper single-read models” are intrinsically superior under m6ACE weak labels. They answer a somewhat different question.

They *are* evidence for one hypothesis that is still open in your data: **a molecule's measurements should probably be considered jointly rather than as nine independent marginal distributions**.

That maps directly onto the limitation you identified in your current quantile representation.

### ORCA is the recent paper I think matters most for your depth question

ORCA, published in 2026, aggregates molecules aligned to a genomic position and extracts signal- and base-level features intended to capture the variability caused by mixtures of modified and unmodified transcripts. It then uses a dual-layer Bi-LSTM feature encoder and a domain-adversarial objective. citeturn22view6

Its especially relevant design decision is its training-set construction: synthetic modified and canonical transcripts are **randomly sampled and combined at varying modification stoichiometries and sequencing depths**, yielding more than seven million simulated sites. citeturn22view6

That is very close in spirit to what your winning model does:

\[
\text{same underlying site}
\rightarrow
\text{multiple artificial coverage realizations}.
\]

The difference is that ORCA can manipulate known modified/unmodified synthetic molecules and train over explicit stoichiometry, whereas your augmentation resamples a weakly labeled endogenous bag.

And ORCA gives us perhaps the most useful reality check in the recent literature: despite intentionally training across depth, its authors state that reliable detection remains difficult at **very low sequencing depth, specifically below ten reads**, and that single-read inference is challenging. citeturn21view6

Given that your deployed median is 2–3, I take that result seriously. It argues against assuming that enough architectural sophistication can simply “learn away” the information loss.

## What the literature says about low coverage

The pattern across papers is unusually consistent.

The 2023 systematic comparison explicitly investigated sequencing depth. Raising coverage from around 5 toward 20 reads improved performance for many tools, while gains generally diminished at higher coverage; error-profile and distribution-comparison methods were particularly susceptible to low coverage. It also found that all evaluated approaches struggled when the fraction of modified molecules was low. citeturn23view7turn8view7

The individual papers then impose floors:

- **m6Anet:** 20-read samples and ≥20-read evaluation sites. citeturn8view0
- **CHEUI site model:** default minimum 20 reads. citeturn22view9
- **DENA:** downstream site filtering at ≥50 reads specifically to reduce depth effects and false positives. citeturn23view13
- **EpiNano:** explicitly acknowledges decreased accuracy under low coverage. citeturn23view15
- **ORCA:** deliberately augments sequencing depth during training but still reports difficulty below 10 reads. citeturn21view6turn22view6

That gives me a fairly strong interpretation of your own depth sweep.

Your numbers are not showing an idiosyncratic failure of LightGBM:

| Your model | full, all training sites ≥20 | 3 reads | 1 read |
|---|---:|---:|---:|
| LGBM, quantiles | 0.4759 | 0.2458 | 0.1527 |
| MLP, quantiles | 0.4783 | 0.2019 | 0.1045 |
| **Depth-augmented LGBM** | **0.4844** | **0.3434** | **0.2593** |

The collapse without augmentation looks qualitatively like the general coverage dependence reported across the field. The interesting result is actually how much **augmentation recovers**.

At your 4.49% prevalence:

\[
\frac{0.4844}{0.0449}=10.79\times
\]

random-prevalence AP at full depth, while

\[
\frac{0.2593}{0.0449}=5.78\times
\]

at one molecule.

And the one-read model still substantially exceeds your motif-only AP of 0.1537. In other words, after augmentation, a single molecule still contains useful pore information in your harness.

That matters architecturally because it puts an upper-level constraint on what the next model should do. A successful set model **cannot depend on estimating a rich empirical distribution**, because you already know that useful deployment behavior must survive when the empirical distribution consists of one point.

This is one reason I am skeptical that a Set Transformer or Perceiver will win merely through read–read interactions. At \(N=1\), there are no interactions.

The highest-value learned object is therefore likely to be the **individual-molecule representation**, with the aggregation mechanism becoming increasingly useful as \(N\) rises.

## What the published m6Anet numbers actually mean

You specifically mentioned m6Anet's HEK293T ROC AUC of about 0.83 and PR AUC of about 0.35.

The comparison is less clean than it first appears.

m6Anet evaluated on HEK293T using **m6ACE-Seq and miCLIP jointly as ground truth**, and compared against EpiNano, MINES, Tombo and nanom6A. citeturn23view5 Across all 18 DRACH motifs, it reports approximately ROC AUC 0.83 and PR AUC 0.35; restricted motif subsets produce somewhat higher PR AUCs. citeturn8view2 The evaluation only includes DRACH candidate sites with at least 20 nanopore reads. citeturn8view0

The paper's HEK293T figure reports **5,579 m6ACE-positive and 121,853 m6ACE-negative** candidate positions, which by itself corresponds to:

\[
\pi_{\mathrm{m6ACE-only}}
=
\frac{5579}{5579+121853}
\approx 4.38\%.
\]

That happens to be extremely close to your **4.49%** positive rate. citeturn12view1

However, I would **not** divide 0.35 by 0.0438 and call the answer “m6Anet has 8× lift.” The PR-AUC benchmark's positive definition is the **union of m6ACE-Seq and miCLIP evidence**, on the eligible held-out sites/genes, whereas 4.38% above is the m6ACE-only count shown elsewhere in the figure. citeturn23view5turn12view1 The exact prevalence corresponding to the 0.35 AP calculation is therefore not established by that main-text count.

For intuition only, **if** the benchmark prevalence had been 4.38%, then 0.35 would correspond to about \(8.0\times\) prevalence. That should not be treated as an apples-to-apples published lift.

There are several other reasons not to rank your 0.4844 directly against its 0.35:

**Ground truth differs.** Yours is m6ACE-Seq; theirs uses m6ACE plus miCLIP. citeturn23view5

**Candidate universes can differ.** AP is prevalence-sensitive and can change dramatically depending on which negatives are admitted.

**Preprocessing differs.** m6Anet normalizes event-level quantities conditional on sequence context and includes learned sequence embeddings. citeturn8view0turn11view1

**The resampling protocol differs.** Their model predicts from controlled 20-read subsamples; your “full” score uses the naturally varying 20–991 molecules. citeturn8view0

**Your evaluation is unusually leakage-conscious.** m6Anet also ensured genes were disjoint between training, validation and testing in its cross-validation, which is reassuringly similar to your decision to group by gene. citeturn23view4 But your 10 repetitions and paired corrected comparison are another layer beyond simply quoting one held-out PR AUC.

So my reading is not “you beat m6Anet.” It is:

**Your 0.4844 is already in a performance regime where a new architecture needs to justify itself against a serious baseline; published headline PR AUCs are not evidence that swapping in m6Anet will automatically improve it.**

There is an additional subtle issue with ROC AUC. The field historically reported ROC heavily, which is why apparently impressive numbers from DENA, RedNano and others should not be mapped onto your AP scale. DENA, for example, reports per-read/motif ROC AUCs around 0.90–0.97 on its constructed read-classification test sets, but those sets and labels are completely different from a naturally imbalanced site-level m6ACE benchmark. citeturn23view11 That is exactly the kind of number I would avoid comparing with your 0.478–0.484 site-level AP.

## What is genuinely state of the art now

As of September 2026, I do **not** think there is a defensible single answer of the form “method X is the SOTA m6A detector.”

The newest systematic Nature Methods evaluation reflects how much the landscape has fragmented: it considers chemistry/generalization, retraining, multiple modification types and robustness rather than treating one fixed model's AUC as definitive; one of its main figure-level conclusions is explicitly that **retraining improves prediction performance and generalizability**. citeturn22view3 Its reference set includes m6Anet, xPore, DENA, CHEUI, newer semi-supervised m6A approaches, Dorado/basecaller models and recent single-molecule methods. citeturn22view5

For your purposes I would divide “SOTA” into different questions.

### Closest state of the art for weak-label MIL

**m6Anet remains the canonical reference.**

Not because its neural network is especially sophisticated, but because it correctly formalizes your exact latent-label structure:

\[
\text{site label known},\qquad
\text{molecule labels unknown}.
\]

Most newer models obtain stronger per-molecule supervision and therefore are not solving exactly the same learning problem. citeturn11view0turn23view3

The important lesson to borrow is the shared read encoder and weak-label training—not necessarily noisy-OR.

### Most relevant recent state of the art for depth robustness

**ORCA is the paper I would put at the top of your reading list.**

Not because I think you should reproduce its Bi-LSTM, but because it explicitly treats **sequencing depth and modification stoichiometry as nuisance variables to randomize during training**. citeturn22view6 That independently validates the experimental direction that produced your largest low-depth gain.

Its admission that <10 reads remains difficult is equally important. citeturn21view6 It prevents us from pretending there is already a published architecture with demonstrated robustness in the SG-NEx regime.

### State of the art for single-molecule callers

The modern direction is represented by methods such as **CHEUI and its SWARM successor, RedNano and TandemMod**, plus basecaller-integrated approaches. These increasingly learn molecule-level modification signatures from raw signal and/or basecalling information, often using controlled experimental labels. citeturn22view8turn7view3turn7view4

That direction is relevant if you can recover richer raw-signal/sequence information, but it should not be confused with evidence that their **site aggregator** is better than yours.

### State of the art for variable-size sets

This is the surprising one: **the nanopore m6A literature itself is not state of the art in generic set representation learning.**

The major methods mostly use:

\[
\text{read classifier}
\rightarrow
\text{simple aggregation}
\]

or

\[
\text{read distribution}
\rightarrow
\text{fixed aggregate statistics}
\]

or, for xPore,

\[
\{x_j\}_{j=1}^{N}
\rightarrow
\text{exchangeable probabilistic mixture}.
\]

m6Anet is the conspicuous MIL exception, and even there the tested attention mechanisms did not outperform its simple pooling. citeturn8view1

I found no evidence in these primary m6A papers that a Set Transformer, Perceiver-style latent bottleneck or graph neural network has established itself as the superior way to combine molecules at one transcriptomic site.

That is both an opportunity and a warning: **trying one would be a genuine architecture experiment, not an implementation of an established m6A best practice.**

## What I would carry from the literature into the architecture discussion

The literature narrows the architecture search quite a lot.

First, I would treat **permutation invariance as non-negotiable**. xPore's probabilistic formulation gets this exactly right, and nothing biological justifies adjacency between read 17 and read 18. A CNN over an arbitrarily ordered read matrix would therefore be difficult to defend. Convolution/recurrent processing across *sequence positions within one molecule*, as in DENA and newer read-level models, is a completely different matter and is physically meaningful. citeturn22view0turn23view11

Second, your most interesting untested signal is not “attention”; it is **joint within-read structure**. Your current representation knows things like

\[
Q_{0.95}(\mathrm{mean}_0)
\]

and

\[
Q_{0.95}(\mathrm{std}_{+1})
\]

but cannot know whether **the same molecule** was anomalous on both measurements. The literature's migration toward per-read neural encoders—m6Anet, DENA, CHEUI, RedNano—is strong precedent for preserving that coupling. citeturn11view1turn23view11turn22view8turn7view3

Third, I would demand that any architecture survive the algebraic fact that:

\[
N=1
\]

means there is no set structure to learn. Therefore the one-read score will mostly be determined by

\[
f(\text{one read},\ \text{sequence},\ \text{site priors}),
\]

not by Deep Sets versus attention versus Perceiver pooling.

That makes your **0.2593 one-read AP** a particularly useful diagnostic. When we compare architectures, I would inspect the one-read result almost as a direct assay of read-encoder quality, and the increment from 1→3→10→20 as an assay of aggregator quality.

Fourth, I would be extremely wary of pooling rules whose numerical scale changes mechanically with cardinality. m6Anet's noisy-OR is the clearest example. Mean pooling is cardinality-stable but loses mixture shape; sum pooling preserves evidence amount but confounds depth; max pooling throws away stoichiometry. That tradeoff is likely more important for your dataset than whether the encoder has two versus four residual blocks. m6Anet's own fixed-20 protocol is indirect evidence that cardinality control matters. citeturn8view0turn11view1

Fifth, the literature makes me **less**, not more, enthusiastic about a large Transformer. Your median training depth is only 47, your effective number of labeled units is 121,838 sites rather than 11 million independently labeled molecules, positives are only 5,475 sites, and the deployment regime often contains too few molecules for read–read interactions to exist. Those facts come from your data, and the published failure of attention to improve m6Anet gives no positive empirical reason to pay a large complexity penalty. citeturn8view1

Finally, I think your depth augmentation should be treated as **part of the problem definition from now on, not as a LightGBM trick**. The strongest recent parallel, ORCA, explicitly randomizes depth during training, and still identifies low coverage as its central limit. citeturn22view6turn21view6 Every neural architecture we discuss next should therefore be trained under your same 1/3/5/10/full depth exposure—or a principled refinement of it—otherwise the comparison will mostly repeat the experiment you have already resolved.

That leaves a fairly short architecture shortlist for the next step: a **small per-read encoder plus carefully chosen permutation-invariant pooling** is the serious baseline; **Deep Sets with explicit cardinality/uncertainty conditioning** is the first architecture I would test; a **mixture/latent-stoichiometry model** is scientifically attractive; and a **Set Transformer or Perceiver** has to earn its complexity rather than being presumed superior. I am considerably more skeptical of graph networks and read-axis CNNs. The published m6A literature gives no compelling evidence for either, whereas it gives substantial evidence for preserving joint per-molecule features, training explicitly across depth, and keeping sequence-position structure separate from molecule-set structure. citeturn8view1turn22view6turn23view11