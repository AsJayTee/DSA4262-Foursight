# Checking our SG-NEx predictions against real lab data

*Notes for the team meeting, 26 Oct. Local only, not committed.*

## The problem this solves

For Task 2 we ran our model on the SG-NEx cell lines, which have no labels. Until
now, the only evidence that those predictions are any good was indirect. The
predictions *look like* known m6A biology: they peak after the stop codon,
they dip near exon junctions, and the strongest motif is GGACT. But looking
plausible is not the same as being right, and that is the obvious question an
examiner will ask of any Task 2 result.

(m6A is a chemical tag on single letters of RNA. Our model ranks candidate
sites by how likely each one is to carry it.)

## What we did

Other labs have mapped m6A in **four of the seven SG-NEx cell lines** using a
method called **GLORI**. GLORI is a chemical lab test that marks each tagged
site individually. It is completely independent of the antibody-based method
behind our course labels.

| SG-NEx line | GLORI data (public, from GEO) |
|---|---|
| A549 (lung cancer) | GSE290408 |
| H9 (embryonic stem cells) | GSE290408 |
| HepG2 (liver cancer) | GSE301419 |
| HCT116 (colon cancer, our training line) | GSE301419 |

We took the predictions **we had already made** on SG-NEx and checked them
against GLORI's answers. GLORI is not an input to the model. It is only used
afterwards to grade the predictions, the way you would check a weather
forecast against what actually happened.

The model was never trained on A549, H9 or HepG2, or on GLORI's method. For
HCT116 we removed every site the model was trained on before checking.

## The main result

The simplest way to read it: if you pick sites to test in the lab, how often
are they really m6A?

A549, sites with at least 20 reads:

| How you pick sites | Share confirmed by GLORI |
|---|---:|
| At random | 6% |
| Top 1% by sequence alone (the motif) | 22% |
| **Top 1% by our model** | **45%** |

So the model's top picks are confirmed about **7× more often than random**,
and about **twice as often** as a rule that only looks at the sequence. The
other lines show the same pattern:

| Line | Random | Sequence only, top 1% | Our model, top 1% |
|---|---:|---:|---:|
| A549 | 6.1% | 22.0% | **45.4%** |
| H9 | 5.1% | 18.4% | **39.5%** |
| HepG2 | 4.6% | 34.3% | **51.0%** |
| HCT116 (unseen sites) | 2.8% | 14.3% | **21.5%** |

*(Sites with 20+ reads. HepG2 and HCT116 have smaller GLORI datasets, so more
real sites are missing from their lists. See caveats.)*

## What it tells us

**1. The SG-NEx predictions are real signal, not noise.** This is the one
piece of Task 2 evidence that rests on a measurement rather than on
plausibility. A researcher using our dashboard to choose sites would do far
better than picking at random.

**2. Sites with very few reads are less reliable but still useful.** In
SG-NEx, about a third of sites have only 1-2 reads. A read is one RNA
molecule measured at the site
([what depth means](../docs/data.md#read-depth)). At 1-2 reads in A549, the
model's top picks are confirmed **19%** of the time, against 3% at random and
9% for sequence alone. Our earlier simulations suggested the model knows
almost nothing at that depth, so this is better than expected. It supports
the dashboard's choice to show low-read sites with a warning rather than hide
them.

**3. It works about half as well as on our own test data.** On dataset0 the
model is about 12× better than random; here it is about 6-8×. That is
expected: this is a new cell line *and* a different lab method. It matches
what we saw earlier on data1.

**4. Part of that drop is not the model's fault.** On HCT116, the one line
where we have both, **the two lab methods agree on fewer than half of the
sites**. The course labels recover 44% of GLORI's sites, and only 48% of the
course's positive sites are in GLORI's list. A model trained on one method's
answers cannot fully match the other's. Ours comes close to how well the two
methods agree with each other.

**5. The score also says how strongly a site is tagged.** GLORI measures what
fraction of molecules carry the tag at each site. The model was only ever
trained on yes/no labels. Even so, among confirmed sites, higher scores go
with a higher tagged fraction (correlation 0.4-0.6). That holds within a
single motif too, so it is not just the sequence.

## How do we compare with m6Anet?

m6Anet (Nature Methods, 2022) is the published tool the brief asks us to
benchmark against. Its released model was trained on HCT116; ours was
trained on HCT116 and HEK293T. Neither saw A549, H9 or HepG2, so those are
new cell lines for **both** models, and GLORI is an
outside judge for both. We ran m6Anet on exactly the same SG-NEx sites, using
the same reads, and graded both models the same way.

This is a fairer test than our earlier m6Anet benchmark on HCT116, where
m6Anet's model had very likely seen the test sites during its own training.

**1. As released, m6Anet can't score most of SG-NEx.** It skips any site with
fewer than 20 reads, and most SG-NEx sites have fewer. Ours scores every site.

| Line | Sites m6Anet scores | Real (GLORI) sites it can see |
|---|---:|---:|
| A549 | 24% | 32% |
| H9 | 39% | 62% |
| HepG2 | 61% | 86% |

**2. On the sites m6Anet does score (20+ reads), ours is as good or better.**
The score below is PR AUC: one number for how well a ranking puts real sites
near the top. Higher is better, and a random ranking scores about the share
of real sites (5-6% here).

| Line | Ours | m6Anet | Verdict |
|---|---:|---:|---|
| A549 | 0.37 | 0.36 | tie |
| H9 | 0.30 | 0.26 | **ours better** |
| HepG2 | 0.38 | 0.28 | **ours clearly better** |

"Better" means the gap held up when we re-ran the comparison on many random
resamples of the genes; "tie" means it didn't.

**3. When both models rank all of a gene's sites together, deep and shallow,
ours is well ahead** (A549 0.27 vs 0.20; H9 0.23 vs 0.15; HepG2 0.34 vs 0.17).
For this test we let m6Anet score low-read sites too. Our scores mean the same
thing whether a site has 3 reads or 300; m6Anet's apparently do not. This is
the situation a dashboard user is in, looking down one gene's list of sites.

**4. Where m6Anet does better.**
- **Estimating how many molecules carry the tag.** m6Anet's score follows
  GLORI's tagged fraction a little more closely. That is expected: m6Anet is
  built to estimate that fraction, while ours only ever saw yes/no labels.
- **Its very top picks.** In A549, m6Anet's top 1% of sites with 20+ reads
  are confirmed slightly more often than ours (49% vs 45%). In H9 and HepG2
  ours are confirmed more often (40% vs 32%, 51% vs 34%).
- At 10-19 reads in A549 and H9, m6Anet is slightly ahead (a gap of about
  0.01).

**In one sentence:** checked against independent lab data in three cell lines
neither model trained on, ours matches or beats m6Anet wherever m6Anet can
score a site, and it also covers the 39-76% of sites m6Anet skips.

## Caveats

- **GLORI's lists only contain the sites GLORI managed to detect.** Some of
  the model's "wrong" picks are probably real sites that GLORI missed. So
  these numbers are a floor; the true accuracy is likely somewhat higher.
  This matters most for HepG2 and HCT116, whose GLORI runs found only about
  7,000 sites each, against 35,000-47,000 for A549 and H9. It affects both
  models equally, so the m6Anet comparison still holds.
- To keep the m6Anet run affordable, it read at most 50 reads per site (it
  only samples 20 per site internally, so this changes its scores very
  little). Our model used every read.
- GLORI was run in different labs, on different batches of cells, from
  SG-NEx. The cell line is the same; the exact samples are not.
- Only four of the seven lines have GLORI data. K562, MCF7 and HEYA8 can only
  be judged on biology.

## A second finding: which kinds of genes carry more m6A

A trap first. Our raw scores rise with read depth, and read depth mostly
reflects how abundant a gene's RNA is. So on raw scores, any group of very
abundant genes looks "heavily tagged" whatever its biology. To avoid this,
each site was compared only with sites that have the same number of reads.
A group of genes counted as a finding only if:

- it went the same way in **all 7 cell lines**, and
- GLORI agreed.

| Group of genes | Result |
|---|---|
| Transcription factors (genes that switch other genes on and off) | **More** m6A |
| Fast-response inflammatory genes (NF-κB) | **More** m6A |
| Long non-coding RNAs | **More** m6A |
| Growth and housekeeping programmes (MYC, E2F, mTORC1, cell division) | **Less** m6A |
| Genes made inside mitochondria | **Almost none**, as expected: the enzyme that adds m6A works only in the nucleus |

This fits the published picture: m6A tends to mark short-lived, regulatory
RNAs for faster breakdown, and stable housekeeping RNAs carry less of it.

**The trap is real.** On raw scores, ribosomal-protein genes look among the
most heavily tagged groups. Once depth is accounted for, they come out
*below* average in every line. Any gene-level claim in the report needs depth
taken into account.

(54 groups were tested with no correction for testing many at once, so treat
this as descriptive.)

## If we only want SG-NEx-only claims

The brief doesn't restrict us to SG-NEx. It asks for "whatever insight can be
extracted", and the predictions here are still on SG-NEx. If the team prefers
to keep outside data out of the main story, these hold on SG-NEx alone:

- the predictions follow known m6A biology (stop codon, exon junctions, motif);
- mitochondrial genes score near zero (a built-in negative control);
- the gene-group differences above hold in all 7 lines;
- raw scores are distorted by read depth (the ribosomal-protein flip);
- the seven cell lines look very alike: differences between lines are about
  as small as differences between repeat runs of one line.

All of these show the predictions are *plausible*. Only the GLORI check shows
they are *correct*.

## Proposals for the meeting

These are suggestions, not decisions:

1. **Use GLORI as the validation for Task 2.** One paragraph and one table in
   the Task 2 section, with the details in an appendix.
2. **Use the GLORI comparison as our main m6Anet benchmark,** or alongside
   the existing one. It avoids the problem that m6Anet's model had likely
   seen the HCT116 test sites.
3. **Add a GLORI panel to the dashboard's Model stats page**, e.g. the
   "random vs sequence vs model" table above, so users can see how far to
   trust a rank.
4. **Lead the gene-level story with the depth trap.** It is a clear
   methodological point and protects every other claim.
5. **Possible follow-ups:** A549 also has GLORI data with the m6A enzyme
   knocked down, which would give very reliable "not m6A" sites. The gene
   groups could also be checked one motif at a time.

## Where it lives (to rerun)

```bash
python analysis/sgnex/glori_check.py    # model vs GLORI, ~2.5 min
python analysis/sgnex/gene_classes.py   # gene groups, ~3 min

# m6Anet: on Ronin (~1.5 h on 64 vCPUs), then copy data/m6anet_glori/*_out*/ back
bash analysis/sgnex/run_m6anet_glori.sh
python analysis/sgnex/m6anet_glori_compare.py   # ~2 min, locally
```

Results: `analysis/sgnex/glori/summary.md`,
`analysis/sgnex/gene_classes/summary.md` and
`analysis/sgnex/m6anet_glori/summary.md`. The write-up of the first two, with
every number, is `report/findings.md` section 16. The downloaded GLORI files
and m6Anet outputs sit under `data/`, which is never committed.
