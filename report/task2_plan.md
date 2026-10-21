# Task 2 plan: an m6A site shortlist for lab follow-up

**Draft for the team, 21 October 2026. The direction is tentative until the
team agrees; the analyses in section 4 and the dashboard data in section 5 are
done.**

The course coordinators asked that Task 2 serve a clear use case rather than
present a general analysis. This brief proposes one. It starts with what our
model can and cannot say, builds the use case from that, and then describes the
dashboard, the analyses behind it (with results), and the open decisions.

**Findings in brief** (section 4):

- **The shortlist works.** On genes the model never saw, its top 3 sites contain
  a truly modified site in 93% of HCT116 genes and 79% of HEK293T genes, against
  71% / 61% when picking by DNA letters and 31% / 32% at random. Two other real
  HCT116 sequencing runs give the same 92-93%.
- **Its lead shrinks with fewer reads**: at 1 read, 79% / 67% against the same
  71% / 61%. Still ahead, but the read count must sit next to every rank.
- **Testing 3 sites is the sensible default**: top 1 finds one in 72% of HCT116
  genes, top 3 in 93%, top 5 in 96%.
- **A gene's best rank in the cell line tells whether the gene is worth it**: if
  it is in the top 1%, 94-97% of genes have a modified site; below the top 25%,
  3-20%.
- **Trust depends on DNA-letter pattern and read count**; nearby sites and gene
  region matter little. Five rarely modified patterns are ranked poorly, but the
  model already keeps them off shortlists (32% of sites, 2% of top-3 sites).

---

## 1. What we have

### 1.1 The model ranks sites; it does not give probabilities

A **site** is one candidate position for m6A: an A inside a DRACH sequence
(five letters such as GGACT) on an RNA. For every site, the model outputs a
**rank**: how likely it is to be modified compared with the other sites in
the same file.

- A rank orders sites. "Site X ranks above site Y" is a valid statement.
- A rank is **not a probability**. A site in the top 1% is not "99% likely to
  be modified".
- A rank is **not an amount**. It says nothing about what share of the RNA
  molecules at that site carry m6A.

We keep it that way. The model was built and tuned to rank, so presenting its
output as a probability would claim something it was not designed to do.
Everything below uses ranks only. The words used on the dashboard are
"higher priority", "lower priority" and "not enough data", never "modified" or
"x% likely".

### 1.2 How the data is organised

```
cell line   (7: A549, H9, HEYA8, HCT116, HepG2, K562, MCF7)
 └─ gene          e.g. MYC
     └─ transcript    one version of the gene's RNA (a gene can have several)
         └─ site          one candidate position on that transcript
```

Three facts about this hierarchy shape the plan.

**Each cell line has its own ranks.** The model reads the RNA molecules
sequenced from that cell line, so MCF7's ranks come from MCF7's molecules.

**Each cell line was sequenced more than once.** SG-NEx repeated the
sequencing to measure reproducibility. A repeat is either a separately grown
**batch of cells** or the same RNA **re-sequenced**. We have 22 files:

| Cell line | Files | Separate batches of cells |
|---|---:|---:|
| A549 | 2 | 2 |
| H9 | 6 | 3 (each sequenced twice) |
| HEYA8 | 5 | 3 |
| HCT116 | 3 | 2 |
| HepG2 | 2 | 2 |
| K562 | 3 | 3 |
| MCF7 | 2 | 2 |

We scored each file on its own, and also each cell line's files **combined**
(every file's molecules at a site merged, then scored once). The combined
version has more molecules per site, so it ranks better.

**The same site can appear in several transcripts.** Where two versions of a
gene share a stretch of RNA, one physical site appears once in each. In
HCT116, the 1.87 million scored rows are only 0.96 million physical sites. The
sequencer assigns each molecule to one transcript, so the copies of a site are
each measured on a different part of the molecules, and their ranks differ:
by 17% of the file's ranking on average when a copy has 1-2 molecules, and 8%
when it has 20 or more. That difference is mostly sampling noise. It is **not**
evidence that one version of the RNA is modified and another is not; our data
is too thin to show that.

### 1.3 How things are aggregated in this plan

- **Sites are never averaged into a transcript score or a gene score.** The
  unit a researcher works with is always one site, because a site is what they
  test in the lab.
- **Copies of the same physical site are merged into one row.** Their ranks
  are averaged, weighted by how many molecules each copy has, so the copy with
  more evidence counts more. The transcripts that contain the site are kept as
  a detail.
- **A gene's list is its sites, sorted.** Each site keeps its rank against
  every site in the cell line (about 2 million). The dashboard sorts a gene's
  sites by that rank to get #1, #2, #3.
- **The ranking comes from the combined file of the chosen cell line.** The
  individual files are used only to check whether a rank holds up (section
  3.3).

### 1.4 Read depth

A **read** is one RNA molecule passing through the sequencer. A site's read
count is how many molecules the model saw there (see
[docs/data.md](../docs/data.md#read-depth)). The model was trained on sites
with 20 or more reads; in SG-NEx, a third of sites have only 1-2. Fewer reads
means less evidence and a less reliable rank, and the plan shows this
everywhere.

---

## 2. The use case

### 2.1 The researcher

A cancer biologist studies one gene in the cell line they work with: for
example **MYC in MCF7** breast cancer cells. They suspect m6A on MYC matters
and want to confirm it in the lab.

MYC has dozens of sites where m6A could be. Confirming one site takes weeks of
lab work (a targeted assay, or mutating the A and observing the effect), so the
researcher can afford to test perhaps three. Their question is:

> "In my gene, in my cell line, which sites should I test first, and how far
> can I trust that order?"

### 2.2 Why this fits our model

The question asks for an **order**, which is what the model produces. It needs
no probability and no amount of modification. Where the order is unreliable
(few reads, unstable between batches, a type of site the model handles badly),
the dashboard says so next to the site, so the researcher can skip it.

### 2.3 What the use case rules out

The dashboard does not claim:

- **that a site is modified**, only that it ranks high;
- **that a cell line has more or less m6A than another.** Our earlier analysis
  found that two batches of the same cell line agree no better than two
  different cell lines, so differences between lines cannot be separated from
  noise;
- **that a site is modified in one version of the gene's RNA but not another**
  (section 1.2).

---

## 3. The dashboard, step by step

### 3.1 Choose a gene and a cell line

The researcher types a gene name and picks their cell line. If the gene has
few or no reads in that line, the dashboard says **"not enough data"**. This is
different from "low rank": the gene may simply be barely active in that line.
For example, TP53 has 12 sites at a median of 1 read in K562 (which barely
expresses TP53) against about 570 sites in HCT116 and MCF7.

### 3.2 See the gene's sites as a ranked list

| Rank in gene | Site | Region | Reads | Rank in cell line | Holds up? | Trust note |
|---|---|---|---:|---|---|---|
| 1 | position A, GGACT | 3' end | 45 | top 1% | 2 of 2 batches | reliable for this type |
| 2 | position B, GAACT | coding | 30 | top 3% | 2 of 2 batches | reliable for this type |
| 3 | position C, AGACA | 3' end | 2 | top 4% | 1 of 2 batches | few reads |

*(Illustration, not real values.)*

- **Rank in gene** answers "which first?".
- **Region** says where the site sits: the start (5' UTR), the protein-coding
  middle, or the end (3' UTR). m6A is known to concentrate near the end of the
  coding region, so biologists use this to judge plausibility.
- **Reads** shows how much evidence the rank rests on.
- **Rank in cell line** shows whether the site is strong at all. If a gene's
  #1 is only in the top 40% of the cell line, nothing in that gene looks
  promising, and the researcher should know before spending weeks on it.
- **Holds up?** and **Trust note** are explained below.

Sites with too few reads stay in the list but are greyed out. The researcher
should still see that they exist.

### 3.3 Holds up? Does the rank survive a repeat?

For each batch of cells, we check whether the site is still near the top of
the gene's list when that batch is ranked on its own. A site that is near the
top in every batch is a safer choice than one that is #1 overall but drops out
in one batch.

This matters because ranks are not always stable. Between two HCT116 files,
the same site came out #1 in a transcript only **53%** of the time when sites
had 3 or more reads, and **71%** of the time with 20 or more.

Lines with only two batches give a weak check ("2 of 2"); H9, HEYA8 and K562,
with three, give a better one. Stability also largely follows read count,
since more molecules give a steadier rank, so this column is shown alongside
reads, not as a separate measure of truth.

### 3.4 Trust note: how good is the model on this type of site?

Each site gets a short note based on how well the model ranks sites with the
same DNA-letter pattern and a similar read count, measured on our labelled data
(analysis 4.2). For example: "the model ranks GGACA sites with 10+ reads well",
"weak: few reads", or "too few examples of this pattern to judge". Nearby sites
and gene region matter too little to change the note (4.2) and appear only in
the detail panel.

### 3.5 Click a site: why is it ranked here?

The detail panel shows:

- **what drives the rank**: the site's own molecules, a boost from nearby
  sites that look modified, or mostly its DNA letters (some letter patterns
  are modified far more often than others);
- **the site's rank by DNA letters alone**, against the model's rank, so the
  researcher can see where the model disagrees with the textbook rule;
- **which versions of the gene's RNA contain this site**, which matters when
  designing the lab test, since the test must target RNA that has the site;
- **the same site in the other six cell lines**, as a consistency note only
  ("also top-ranked in 5 of 6 other lines"), not as a comparison.

### 3.6 Warnings

- **HCT116 was used to train the model.** One of its files is our training
  data, and sites from training rank higher than they would otherwise. HCT116
  ranks carry a warning (or a per-site flag: see decision 6.2).
- **About 5% of sites cannot be placed in a gene**, because their transcript
  is not in the gene annotation we use (Ensembl 91). They are left out.

### 3.7 Export

The researcher can download their shortlist (sites, ranks, reads, notes) as a
CSV.

---

## 4. Analyses behind the dashboard

Each analysis answers a question the researcher would ask before trusting the
shortlist. 4.1 and 4.2 are **done** (21 October). They use predictions the model
made on genes it never saw in training, on our two labelled cell lines (HCT116
and HEK293T): the shipped model's three designs, three seeds each. Low read
counts are simulated by randomly removing reads, repeated five times and
averaged. Intervals are 95%, from resampling whole genes.

Code, tables and how to rerun: [analysis/site_types/](../analysis/site_types/README.md).
Every number below comes from `analysis/site_types/results/`.

### 4.1 Does the shortlist work? (top-3 hit rate)

**Question:** if a researcher tests the top 3 sites of a gene, how often is at
least one truly modified?

**Method:**
1. On the labelled cell lines, for each gene, sort its sites by the model's
   rank and take the top 3.
2. Check how many are truly modified.
3. Do the same for the two ways a researcher could choose without the
   dashboard:
   - **by DNA letters**: the most often modified patterns first (GGACT first);
   - **at random**.
4. Repeat for the top 1 to top 5, so the dashboard can say how many sites are
   worth testing.
5. Check the "rank in cell line" column: when a gene's best site is only
   mid-ranked, does the gene really tend to have no modified sites?

The DNA-letter rule is learned from other genes only, never the gene being
tested. Copies of one site on several transcripts are merged first, as on the
dashboard. Genes need at least 5 sites so that a top 3 is a real choice.

**Result: genes where at least 1 of the top 3 is truly modified**

| Reads per site | HCT116: model / DNA letters / random | HEK293T: model / DNA letters / random |
|---|---|---|
| Full (20+) | **93%** / 71% / 31% | **79%** / 61% / 32% |
| 10 | 90% / 71% / 31% | 77% / 61% / 32% |
| 3 | 85% / 71% / 31% | 72% / 61% / 32% |
| 1 | 79% / 71% / 31% | 67% / 61% / 32% |
| Two other real HCT116 runs | 92% and 93% / 71% / 31% | |

1,477 HCT116 genes and 1,706 HEK293T genes with at least one modified site. The
model's lead over the DNA-letter rule is +22 points (interval +20 to +24) on
HCT116 at full reads and +8 points (+6 to +11) at 1 read; every interval is
above zero.

**How many sites to test** (full reads):

| Top k | HCT116 model | HCT116 DNA letters | HEK293T model | HEK293T DNA letters |
|---|---:|---:|---:|---:|
| 1 | 72% | 42% | 54% | 30% |
| 2 | 89% | 60% | 72% | 50% |
| 3 | 93% | 71% | 79% | 61% |
| 4 | 95% | 78% | 84% | 69% |
| 5 | 96% | 82% | 87% | 74% |

Gains flatten after 3, so "test the top 3" is a sensible default for the
dashboard to suggest.

**Does a gene's best rank say whether the gene is worth it?** Genes grouped by
where their best site ranks among all sites of the cell line:

| Gene's best site | HCT116: genes with a modified site | HEK293T |
|---|---:|---:|
| Top 1% | 94% | 97% |
| 1-5% | 69% | 83% |
| 5-10% | 29% | 56% |
| 10-25% | 6% | 34% |
| Below top 25% | 3% | 20% |

Yes: the "rank in cell line" column (3.2) carries real information. At 1 read
the same pattern holds but is weaker (top 1%: 82% / 90%; below top 25%: 8% / 30%).

**What it means:** the dashboard's central promise holds. The model beats the
textbook rule at every read count, and by most where reads are plentiful.

**Limits:**
- Labelled sites are the well-covered ones (20+ reads) in about 3,600 HCT116 and
  2,900 HEK293T genes with 5+ sites; low read counts are simulated, not real.
- The lab method behind the labels misses some real m6A, so true hit rates are
  probably somewhat higher.
- Only two cell lines have labels, and both were in training (other genes).

### 4.2 How good is the model on each type of site?

**Question:** which sites on the shortlist can be trusted? This drives the
trust note (3.4).

**Method:** measure the model's performance separately for each type of site,
using the properties the researcher already sees:

| Site type | Why |
|---|---|
| DNA-letter pattern (18 DRACH patterns) | Some patterns may be ranked well, others poorly |
| Read count (1, 3, 10, 20+) | Already known to matter a lot; shown per site |
| Nearby sites or none | The model uses nearby sites as supporting evidence; a site with none gets no support. Our findings show the model is weaker there |
| Region (5' UTR, coding, 3' UTR) | Whether the model is equally good along the gene |

**The measure: pairwise accuracy.** Take one modified and one unmodified site
of the same type; how often does the model rank the modified one higher? 0.5 is
a coin flip, 1.0 is perfect. It is used because it does not depend on how often
a type is modified: patterns differ hugely there (22% of GGACT sites against
0.02% of TAACA in HCT116). The earlier per-motif measure in GAPS.md ("lift") is
capped by that rate, which is the only reason GGACT looked worst there.

**Overall, by read count:**

| | Full reads | 10 | 3 | 1 read |
|---|---:|---:|---:|---:|
| HCT116 | 0.96 | 0.94 | 0.91 | 0.88 |
| HEK293T | 0.87 | 0.84 | 0.82 | 0.79 |

The model is clearly better on HCT116. Other SG-NEx lines are more likely to
look like HEK293T, so the trust note uses **the worse of the two lines**.

**By DNA-letter pattern** (worse of the two lines, full reads):

| Group | Patterns | Accuracy | Share of sites modified (HEK293T) |
|---|---|---:|---:|
| Strong | AGACA, GGACC, GGACA, GAACC, AGACT, TGACT, GGACT | 0.84-0.86 | 3-22% |
| Middle | AAACT, GAACT, TAACT, AGACC, GAACA | 0.78-0.82 | 2.5-15% |
| Weak | TGACA, TGACC, AAACA, AAACC, TAACA | 0.61-0.74 | 1-4% |
| Too few to judge | TAACC | | 13 modified examples in all |

- **The weak patterns are the rarely modified ones**: few examples to learn
  from, and possibly a weaker signal in the reads.
- **GGACT, the best-known m6A pattern, is strong but not the best**: the model
  knows GGACT sites are often modified, but telling *which* ones are is harder.
- **Read the groups, not the exact order.** The order agrees only moderately
  between the two lines (rank correlation 0.61), and rare patterns have wide
  intervals (TAACA: 0.51-0.73).
- **Weak patterns rarely reach a shortlist.** In MCF7 they are 32% of all sites
  but 2% of transcripts' top-3 sites: the model already ranks them low.

**By read count, within patterns:** every pattern loses accuracy with fewer
reads. At 1 read on HEK293T the strong group falls to 0.70-0.74 and the weak
group to 0.58-0.63, close to a coin flip. The model is cautious with little
evidence: in MCF7, 1-2-read sites reach the cell line's top 5% a third as often
as 20+-read sites (2.8% vs 7.6%), so they do not crowd better-covered sites off a
shortlist. But **53% of all MCF7 top-3 sites have only 1-2 reads**, because many
transcripts are thinly covered everywhere. This is the strongest reason the read
count sits next to every rank.

**Nearby sites and region: small effects.**

| | HCT116 | HEK293T |
|---|---:|---:|
| No nearby site within 150 nt | 0.92 | 0.85 |
| 6+ nearby sites | 0.96 | 0.88 |
| 3' end (3' UTR) | 0.96 | 0.87 |
| Coding region | 0.96 | 0.84 |
| 5' end (5' UTR) | 0.90 | 0.84 |

Too small to drive the trust note; one line in the detail panel.

**What it means for the trust note:** pattern x read count, worse of the two
lines. The values are exported (5.1); the labels are decision 6.4.

### 4.3 Worked examples

Two or three genes walked through the dashboard, for the report and the demo:

- **MYC**: well covered in every line (5-6 transcripts, a median of 30-57 reads
  per site in the combined files) and discussed in the SG-NEx paper.
- **A gene with m6A sites known from published work** (for example MALAT1),
  so the reader can see whether the shortlist finds them. Not started.

---

## 5. Data and hosting

### 5.1 What we need, and whether we have it

**Built (21 October)** by `analysis/sgnex/export_site_shortlist.py` into
`data/dashboard/` (not in git; 5.3). Each batch of cells sequenced more than
once was scored from its runs combined (`score_sgnex.py --batch`).

| Needed | Status | Where |
|---|---|---|
| Site, DNA letters, region, transcripts containing it | Done | per gene, once |
| Rank in cell line, reads (each line's samples combined) | Done | per gene and line |
| Rank and reads in each batch of cells ("holds up?", 3.3) | Done | per gene and line |
| Rank from the site's own reads only (3.5) | Done | from 4 of the shipped model's 6 networks; the other 2 have no such split |
| Rank by DNA letters alone | Computed in the browser | from the pattern table in `trust.json` |
| Trust note values | Done | `trust.json`: pattern x read band, worse of the two lines |
| HCT116 training-site flag | Decision 6.2 | not included |

The dashboard shows the **shipped** model (2 seeds per design, what
`predict.py` ships). The analyses in section 4 use the held-out fold models (3
seeds), since only those never saw each gene.

### 5.2 Size

26,321 genes, 1.5 million physical sites, all 7 cell lines: **90 MB** in 256
gene files (240-550 KB each, around 100 genes per file) plus a 0.9 MB search
index. This fits on GitHub Pages (1 GB site, 100 MB per file) **without
dropping any sites**; a gene search loads one file. About 4-5% of copies per
line are dropped because their transcript is not in Ensembl 91.

### 5.3 Rules

- Only model outputs on public SG-NEx data go on the site. **Never the
  course's labels or read files** (AGENTS.md section 5). Analyses 4.1 and 4.2
  publish summary numbers only.
- If the data files are committed to git, each regeneration adds about 100 MB
  to the repository's history permanently. Publishing them through the deploy
  step, without committing them to `main`, avoids that.

---

## 6. Decisions for the team

1. **Adopt this use case?** Everything else follows from it.
2. **HCT116 training flag.** Publish which sites were in the training data
   (positions only, no labels) so they can be flagged individually, or show one
   general warning for HCT116?
3. **Merging copies of a site.** The plan uses a weighted average of existing
   ranks (1.3). The alternative is to re-score each physical site once with
   all its molecules together. That is cleaner but needs the 22 samples
   downloaded and scored again on Ronin.
4. **Trust-note labels.** Proposed, on pairwise accuracy (4.2): "strong"
   0.84 and above, "moderate" 0.75-0.84, "weak" below 0.75, and "too few
   examples" where a pattern cannot be judged in both lines.
5. **Worked examples (4.3):** who takes them, and which published gene.
6. **Dashboard build:** who builds the interface on the exported data. If the
   data format is adopted, it gets a `docs/decisions/` record (AGENTS.md
   section 1).

---

## 7. What happens to the earlier findings (A-E)

The earlier analyses
([analysis/sgnex/explore/summary.md](../analysis/sgnex/explore/summary.md))
are not the story any more, but three of them now justify design choices:

| Earlier finding | Role in this plan |
|---|---|
| A1: a third of sites have 1-2 reads | Why reads are shown and weak sites greyed out |
| A2: batches of one line agree no better than different lines | Why cross-line results are a consistency note, not a comparison |
| B: scores dip near exon junctions and peak after the stop codon, in every line | Evidence the ranks reflect known m6A biology in lines without labels; supports the region column |
| C: share of sites called modified per line | Dropped: needs a cut-off, which a rank does not support |
| D, E: genes and lines that look alike | Dropped as findings; D's shared genes could resurface as a consistency note |
