# Theory: why the model is shaped the way it is

Short derivations that connect what we measured in the data to the model's
design and behaviour. The aim is that the report can **justify** the
architecture instead of only describing it. Each result says where its
evidence is.

**Status (11 Oct 2026).** These are *idealised* models.
- The derivations assume each site's reads depend only on that site's own
  modification state, and (where stated) that neighbours are independent
  of each other.
- The trained network does not literally compute these formulas.
- Read the theory as the **principle the architecture implements**, and
  the figures as evidence that the network learned something of that
  shape. Section 8 lists the assumptions.

Notation used throughout:

| Symbol | Meaning |
|---|---|
| $Y_i \in \{0,1\}$ | whether candidate site $i$ is modified |
| $\pi$ | base rate $P(Y_i=1)$: 4.49% in cell line 1, 7.24% in cell line 2 |
| $E_i$ | the reads at site $i$ ($n_i$ of them, one per RNA molecule) |
| $L_i = \dfrac{P(E_i \mid Y_i=1)}{P(E_i \mid Y_i=0)}$ | the likelihood ratio from site $i$'s own reads |
| $d_{ij}$ | distance in nucleotides between sites $i$ and $j$ on one transcript |
| $r(d)$ | co-modification ratio at distance $d$ (§1) |
| $\sigma(x) = 1/(1+e^{-x})$, $\operatorname{logit}(p) = \log\frac{p}{1-p}$ | the usual pair |

---

## 1. What we measured: local co-modification

$$r(d) = \frac{P(Y_i=1,\ Y_j=1 \mid d_{ij}=d)}{P(Y_i=1)\,P(Y_j=1)}\quad\text{(the "local part": beyond each transcript's own positive rate)}$$

| Distance | Cell line 1 | Cell line 2 |
|---|---|---|
| 1-25 nt | 2.20 [2.04, 2.38] | 2.13 [1.96, 2.29] |
| 25-50 nt | 2.10 | 2.00 |
| 50-100 nt | 1.88 | 1.73 |
| 100-200 nt | 1.46 | 1.34 |
| 200-400 nt | 1.11 | 1.02 |
| 400-800 nt | 0.80 | 0.74 |

Source: `analysis/newdata/distance_bands/bands.csv`; figure `fig_comodification`.

- **The same in both cell lines** within the intervals.
- **The excess $r(d)-1$ decays roughly exponentially:** a log-linear fit
  over 1-400 nt gives a decay scale $\lambda \approx 100$-$120$ nt. That
  matches the per-fold fits used by the model (101-122 nt; ~100 nt for
  both labellings on shared sites, findings 12A).
- **It dips below 1 beyond ~400 nt.** That follows from clustering
  itself: a transcript's modified pairs are concentrated close together,
  so fewer are left far apart.

---

## 2. The corroboration message

### 2.1 Derivation

Take site $i$, its reads $E_i$, and one neighbour $j$ with reads $E_j$.

**Assumption A1:** each site's reads depend only on its own state. So
$E_i \perp E_j \mid Y_i$, and $E_j \perp Y_i \mid Y_j$.

By Bayes' rule, in log-odds:

$$\operatorname{logit} P(Y_i=1 \mid E_i, E_j) = \operatorname{logit}\pi + \log L_i + \underbrace{\log\frac{P(E_j \mid Y_i=1)}{P(E_j \mid Y_i=0)}}_{m_{ij}}$$

Expand the neighbour term over the neighbour's own state:

$$P(E_j \mid Y_i=y) = \sum_{y_j} P(E_j \mid Y_j=y_j)\,P(Y_j=y_j \mid Y_i=y)$$

By the definition of $r$,

$$P(Y_j=1 \mid Y_i=1) = r\pi \qquad\text{and}\qquad P(Y_j=1 \mid Y_i=0) = \frac{\pi(1-r\pi)}{1-\pi} \equiv \pi'$$

Dividing through by $P(E_j \mid Y_j=0)$:

$$\boxed{\;m_{ij} = \log\frac{r(d_{ij})\,\pi\,L_j + 1 - r(d_{ij})\,\pi}{\pi' L_j + 1 - \pi'}\;}$$

**So the posterior splits into three separate, additive terms: prior +
own evidence + neighbour message.**

### 2.2 Properties, and the design choice each one justifies

1. **No co-modification, no information.**
   - If $r=1$, then $\pi'=\pi$, the numerator equals the denominator, and
     $m_{ij}=0$.
   - Neighbours help only where $r(d)$ differs from 1: within about
     100-200 nt (§1).
   - *Evidence:* the radius test. About 100 nt transfers between cell
     lines; wider context helps only the training line (`fig_radius`).
     Removing neighbours within 0-20 nt costs ≤ 0.001 PR AUC (findings
     12C).

2. **A neighbour without evidence says nothing.**
   - If $L_j=1$, the numerator and denominator both equal 1, so
     $m_{ij}=0$.
   - A neighbour with one read has $L_j$ near 1. A message must therefore
     carry the **strength** of the neighbour's evidence, not just its
     verdict.
   - *In our model:* scalar messages pass each neighbour's own score and
     its read count, and the gate sees the site's own read count.
   - *Contrast:* DeepMod's neighbour stage passes a histogram of
     neighbours' methylation *fractions*, where 1/1 and 50/50 reads are
     identical (`literature.md`, "DeepMod vs our model").

3. **Neighbours confirm or contradict, with a break-even point.**
   - $m_{ij}$ increases monotonically in $L_j$.
   - It is positive when $L_j > 1$ and negative when $L_j < 1$.
   - *Evidence:* `fig_corroboration_curve`. A modified-looking neighbour
     raises a site's score, an unmodified-looking one lowers it, and they
     cross over in between.

4. **Bounded influence.**
   - As $L_j\to\infty$: $m_{ij}\to\log r(d)$, at most $\log 2.2 \approx
     0.79$ logits per neighbour (below 25 nt).
   - As $L_j\to 0$: $m_{ij}\to\log\frac{1-r\pi}{1-\pi'}\approx-(r-1)\pi$,
     which is small when $\pi$ is small.
   - A neighbour can **adjust** the site's own evidence; it cannot
     override strong own evidence.
   - *In our model:* the residual design, $z_i = a_i + g_i\,\Delta_i$.
     The site's own score is the starting point and neighbours add a
     correction.
   - The predicted asymmetry (strong confirmation, weak contradiction)
     holds under the *true* prior. Our networks were trained with
     positives up-weighted (§5), which acts like a much larger $\pi$ and
     makes negative messages larger. That is consistent with the learned
     curve lowering sites noticeably.

5. **Own evidence and neighbour evidence enter as separate terms.** This
   is why a site's own vector must be kept apart from its neighbours'.
   - **H2GCN** concatenates them (self channel and neighbour channel).
   - **The residual design** adds a correction to the own score.
   - **A plain GCN fails.** It averages the site's vector with its
     neighbours',
     $h_i' = \phi\big(W\frac{h_i+\sum_j h_j}{\deg_i+1}\big)$, so with 4
     neighbours the site's own evidence gets 1/5 of the weight. In the
     posterior, $\log L_i$ keeps its full weight however many neighbours
     there are.
   - *Evidence:* GCN is the worst model, −0.10 / −0.14 PR AUC against
     LightGBM (`models_distribution_*`).

### 2.3 Distance weighting, derived rather than chosen

Linearise $m_{ij}$ in the excess $r-1$. Differentiating at $r=1$:

$$m_{ij} \;\approx\; \big(r(d_{ij})-1\big)\cdot\varphi(L_j),\qquad \varphi(L) = \frac{\pi\,(L-1)}{(1-\pi)\big(1+\pi(L-1)\big)}$$

**The message factorises: a distance factor times an evidence factor.**
- $\varphi$ is increasing, with $\varphi(1)=0$, and bounded:
  $\varphi \in \big(-\tfrac{\pi}{1-\pi},\ \tfrac{1}{1-\pi}\big)$.
- With $r(d)-1 \approx A\,e^{-d/\lambda}$ (§1):

$$m_{ij}\;\approx\;A\,e^{-d_{ij}/\lambda}\,\varphi(L_j)$$

That is the scalar-messages design: a distance kernel $e^{-d/\lambda}$, with
$\lambda$ fitted to the training labels, applied to a function of each
neighbour's own evidence.
- The constrained designs **build in** this structure.
- The free designs (GAT, the graph transformer) must learn it, and can also
  learn things it excludes. That is the cell-line-specific context that
  does not transfer (§7).

### 2.4 Several neighbours: sum or average?

If the neighbours were independent given $Y_i$, their messages would
**add**: $\sum_j m_{ij}$. They are not: neighbours co-modify with *each
other* as well. A plain sum treats correlated evidence as independent and
becomes overconfident on dense transcripts.

The exact posterior lies between "average" and "sum". Our models use a
distance-weighted **average** (row-normalised kernel), plus the
has-neighbour flag and the best neighbour, which is the conservative end.
- *Evidence:* the graph's gain rises with the number of neighbours, then
  levels off (`fig_neighbour_count`).
- *Evidence:* sites with no neighbours gain nothing ($m=0$ by
  definition).

---

## 3. Why neighbours matter most at low read depth

The reads at a site are different molecules, so given the site's state
their evidence adds:

$$\log L_i = \sum_{r=1}^{n_i}\log\ell_r,\qquad \mathbb{E}[\log L_i \mid Y_i] \propto n_i$$

The neighbour message is bounded by $\log r(d)$ (§2.2, property 4).
Therefore

$$\frac{|m_{ij}|}{|\log L_i|}\;\sim\;\frac{1}{n_i}$$

**The fewer a site's own reads, the larger the share of its evidence that
comes from neighbours.**
- *Evidence:* a 1-read site scores 0.367 with well-covered neighbours vs
  0.225 alone (findings 8).
- *Evidence:* the graph models keep about twice LightGBM's PR AUC at 1-3
  reads (`fig_depth`).

This is also the theoretical case for a **depth-dependent gate**,
$g_i = \sigma(\alpha\log(1+n_i)+\dots)$. In practice the gate barely
moved (findings 12E), because every training site has 20+ reads, so the
regime where it matters never appears in training.

---

## 4. Why scores plateau above ~50% modified (presence, not fraction)

The model is trained on **binary** labels ($Y_i \in \{0,1\}$) with
cross-entropy, a proper scoring rule. Its ideal output is therefore

$$s^*(x) = P(Y_i=1 \mid x)$$

That is the probability that the site is modified *at all*.
**Nothing in this objective rewards telling a 50%-modified site from a
100%-modified one:** both have $Y=1$.

Now consider the evidence. A site with modified fraction $f$ yields reads
from the mixture $(1-f)P_0 + fP_1$. The expected evidence for "modified at
all" grows with $n\cdot D(f)$, where $D(f)$ is the per-read divergence of
that mixture from $P_0$ (for small $f$, $D(f) \approx \tfrac{f^2}{2}\chi^2(P_1\|P_0)$).
With $n \approx 600$-$1{,}200$ reads, even $f=0.25$ gives overwhelming
evidence, $P(Y=1 \mid x)\to 1$, and the score saturates.

*Evidence:* `fig_data2_fractions`.
- Median scores: 21-37% at $f=0$, 88-90% at $f=0.25$, ~96% from $f=0.5$
  up.
- The model detects presence and cannot measure stoichiometry. To measure
  it, the target (and loss) would have to be the fraction, not a binary
  label.

---

## 5. The class weight inflates scores by an exact amount (the "RLHF" argument)

Positives are rare (4.49%), so training up-weights them by
$w = \frac{1-\pi}{\pi} \approx 21.25$. Here $\pi$ is the training set's
positive rate: for cell line 1 alone, $w = 21.25$; training on both cell
lines gives a somewhat smaller $w$.

For a site with true probability $q = P(Y=1 \mid x)$, the weighted loss is

$$\mathcal{L}(p) = -\,w\,q\log p - (1-q)\log(1-p)$$

Setting $\mathcal{L}'(p)=0$ gives $\frac{wq}{p} = \frac{1-q}{1-p}$, so

$$p_w = \frac{wq}{wq+1-q}\quad\Longleftrightarrow\quad \boxed{\operatorname{logit}p_w = \operatorname{logit}q + \log w}$$

**An objective chosen for one purpose** (learning from rare positives)
**systematically biases the output** (every score inflated by
$\log w = \log 21.25 \approx 3.06$ logits). A one-line calculation says by
how much, and how to undo it:

$$\operatorname{logit}q = \operatorname{logit}p_w - \log w$$

- It explains why our scores are not probabilities, and the earlier 1.8×
  overcount of modified sites (AGENTS.md, section 6).
- The same identity appears in `docs/literature-review.md`, checked there.
- **Moving to another cell line** with base rate $\pi_t$, a standard
  prior-shift correction applies (Saerens, Latinne & Decaestecker, *Neural
  Computation* 2002):

$$\operatorname{logit}q_t = \operatorname{logit}q_s - \operatorname{logit}\pi_s + \operatorname{logit}\pi_t$$

  This needs the target's base rate, which is unknown for SG-NEx. Hence the
  plan to calibrate empirically (`task2_briefing.md`) and quote ranges.
- Ranking metrics (PR AUC, ROC AUC) are unaffected: adding a constant to
  every logit does not change the order.

---

## 6. Why two output heads beat one model trained on both cell lines

**Setup.**
- Both versions train on the same sites from both files.
- **One head ("pooled"):** each site's target is its file's label. A site
  present in both files with *disagreeing* labels gets a soft target of
  0.5.
- **Two heads:** a shared network (trunk) and one output per cell line.
  Each head is trained on its own file's labels. A shared site's two
  labels go to the two heads, at weight 1/2 each, so a physical site still
  counts once.
- **At prediction,** the shipped two-head output is the **mean of the two
  heads' logits**.

**Model of the labels.** Suppose each labelling follows

$$\operatorname{logit}P(Y^{(c)}=1 \mid x) = w_c^\top h(x) + b_c$$

That is: shared features $h$, cell-line-specific weights $w_c$ and
offsets $b_c$.

- **Case 1: the labellings differ only in base rate** ($w_1=w_2$,
  $b_1\ne b_2$).
  - The pooled model's ideal output, $\tfrac12\big(\sigma(w^\top h+b_1)+\sigma(w^\top h+b_2)\big)$,
    is monotone in $w^\top h$.
  - So is the two-head average, $w^\top h + \tfrac{b_1+b_2}{2}$.
  - **Same ranking: no advantage from heads.**
- **Case 2: the labellings disagree about which features matter**
  ($w_1\ne w_2$).
  - The pooled model must fit **one** function to two partly conflicting
    targets.
  - At the discordant shared sites (64% of shared modified sites are
    modified in only one line, findings 12D), the soft target 0.5 looks
    like label noise to it. The gradient pulls those inputs toward the
    middle.
  - The shared features are shaped by a blurred average of the two
    labellings.
  - With two heads, the conflict is resolved **at the heads** (cheap,
    linear). The trunk learns features useful for *both* labellings,
    without being asked to reconcile them. This is standard multi-task
    learning: a shared representation with task-specific heads
    (Caruana, *Machine Learning* 1997).

**Our data are Case 2.** The two cell lines differ in *which transcripts*
are m6A-rich (3.3× vs 2.0× transcript-level enrichment, §1), not only in
base rate. So two heads should help, and should mostly help **on the
cell lines they were trained on**.

**Evidence:** the same architecture with one vs two heads, trained on both
cell lines, 3 seeds each (gain in PR AUC over LightGBM):

| | Cell line 2 | Cell line 1 | Seed spread (cell line 2) |
|---|---|---|---|
| One head (`h2gcn`) | +0.031 (+0.036, +0.027, +0.030) | +0.093 | sd 0.005 |
| Two heads (`h2gcn_twohead`) | **+0.039** (+0.039, +0.037, +0.041) | **+0.108** | **sd 0.002** |

Source: `analysis/representation/results/h2gcn{,_twohead}_nn__xsrc[__seed1,2].json`.

- Two heads gain about +0.008 on cell line 2 and +0.014 on cell line 1.
- They are **2.5× more stable across seeds.** Fewer conflicting targets
  means less gradient noise.

**Why averaging the heads is not "the same thing" as one pooled head:**
1. **The trunk is trained differently** (above). That is the main effect.
2. **Averaging logits is not averaging probabilities.** Logit-averaging
   multiplies the heads' odds (a geometric mean). The pooled model learns
   the arithmetic mean of probabilities. Take heads of 0.99 and 0.50: the
   pooled-style average is 0.745, the logit average is
   $\sigma\big(\tfrac{4.6+0}{2}\big)=0.909$. These are different functions
   of the two labellings, so the rankings differ where the heads disagree.

**For an unseen cell line,** the head average is a hedge between two
labellings, not a model of the third. That is part of why the shipped
ensemble pairs the two-head network (strong on the training lines) with
the constrained designs (strong on transfer) (decision 0034).

---

## 7. Why extra capacity does not transfer, and the label ceiling

### 7.1 Capacity learns the training line's labelling

Write each cell line's labelling as a shared, transferable part plus a
line-specific part:

$$\operatorname{logit}P(Y^{(c)}=1 \mid x) = \underbrace{\log L(x) + \sum_j m_{ij}}_{\text{reads and local co-modification: shared}} + \underbrace{u_c(x)}_{\text{e.g. which transcripts are m6A-rich}}$$

- A flexible model (attention over the whole transcript) can fit $u_1$ on
  cell line 1. That helps there and is noise on cell line 2.
- A constrained model (§2.3's form) can only represent the shared part.

*Evidence:*
- the graph transformer is better *within* either cell line, not across
  (findings 12F);
- half its attention lies beyond 150 nt, where $r(d)\approx1$ but
  transcript-level enrichment is cell-line-specific (`fig_gps_attention`).

### 7.2 A hard bound for "learn cell line 1 perfectly"

Any predictor that is a function of cell line 1's label $y^{(1)}$ alone can
only put the sites with $y^{(1)}=1$ above those with $y^{(1)}=0$ (tied
within each group). On the 67,320 shared sites:

$$P\big(y^{(2)}=1 \mid y^{(1)}=1\big)=\tfrac{2136}{3291}=0.649,\qquad \text{recall at that cut}=\tfrac{2136}{4853}=0.440$$

With a two-level score, average precision has a closed form:

$$\text{AP} = 0.440\times0.649 + (1-0.440)\times0.0721 = 0.326$$

This is an exact ceiling for that class of predictors. Our reads-based
model scores 0.422 on the same sites (`fig_ceiling`).
- It is **not** a bound on every model: a model could in principle find
  cell-line-2-specific signal in cell line 2's reads.
- Two tests found almost none: no model tells the cell lines apart at
  discordant sites (which-line AUC ≈ 0.5), and the raw reads give
  0.52-0.54 (findings 12D).

---

## 8. Assumptions and where they bend

| Assumption | Used in | How it bends in practice |
|---|---|---|
| A1: a site's reads depend only on its own state | §2, §3 | Nanopore signal from one m6A spreads ±10 nt (DeepRM). Neighbours within 20 nt could re-measure the site, but removing them changed nothing (findings 12C) |
| Neighbours independent given $Y_i$ | §2.4 (sum) | False: they co-modify. Hence an average, not a sum |
| $r(d)$ is the same for every transcript | §2.3 | The local part is near-identical across cell lines; per-transcript variation is not modelled |
| Reads are independent molecules | §3 | True by construction of direct RNA sequencing; PCR-free |
| Binary labels are correct | §4, §5, §7 | m6ACE-seq has its own noise; label disagreement between assays is large (`literature.md` §4) |
| The network reaches its ideal output | §5 | Approximately. Hence empirical calibration, not only the formula |

## 9. Suggested use in the report

- **Methods (≈ 8 lines):**
  - the boxed message $m_{ij}$ and the factorised form of §2.3;
  - one sentence each for properties 1, 2 and 5.

  This justifies "own evidence + distance-weighted, confidence-aware
  neighbour correction", and why GCN fails.
- **Results:** cite §6 when presenting the two-head model, and §7.2
  alongside `fig_ceiling`.
- **Discussion:** §3 (low depth), §4 (presence, not fraction), §5 (scores
  are not probabilities; how to calibrate).
- **Do not overclaim:** the theory gives the *shape*; the network's exact
  numbers differ (e.g. the class weight's $\log w$ shift, §5).
