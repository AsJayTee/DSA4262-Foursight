# 0032. Rank models on the worse of the two cross-source gains, not the mean

- **Date:** 2026-10-02
- **Status:** Accepted
- **Affects:** the selection rule of [0029](0029-models-are-selected-on-cross-source-gain.md) (amended), `scripts/build_dashboard.py` (Decisions view sorts on `xsrc/gain_worst`), the cross-source report text, AGENTS.md section 6. No metric changes; every key is still logged.

## Context

0029 chose models on the **mean** of two gains over the baseline: one under
dataset0's labels, one under data1's. Twelve cross-source runs later the mean
ranks `everything` first (+0.041) on a dataset0 gain (+0.065) that does not
transfer, ahead of `quantiles + deepset` (+0.026), which gains +0.022 on data1
and +0.014 on data1 genes absent from dataset0, against `everything`'s +0.016
and -0.003.

What we know about the leaderboard test set: it is a new sequencing run
(certain), probably of largely the same transcripts (data1 shared 74% of
dataset0's sites), and **labelled by a labelling we do not know** - dataset0's,
data1's, or a third. The two labellings disagree: only 65% of dataset0's
positives on shared sites are data1 positives.

## Decision

1. **Rank on `xsrc/gain_worst`**, the smaller of the two gains. The veto stays:
   reject below -0.005.
2. **Differences under ~0.005 are ties** (the detection floor of a single
   cross-source run); break them on `xsrc/data1_new_genes/gain` (generalising
   to unseen genes), then on simplicity.
3. The mean is still logged and shown; it is no longer the sort key.

## Why this and not the alternatives

**The mean (0029).** Rewards a model for excelling under one labelling we may
never be scored on - which is exactly how the gains that did not transfer won.

**The data1 gain alone.** Treats data1's labelling as the test's. Nothing says
it is; it is merely the newer release.

**The new-genes gain alone.** The cleanest generalisation test, but 12,061
sites on 623 genes - its intervals are about twice as wide - and the test set is
probably mostly *known* genes.

**Worst-case** is the choice that makes no assumption about the test
labelling: the model that does best in its weaker case.

## Consequences

- The ranking moved: `quantiles + deepset` and `everything_fulldepth` (+0.022
  each) now lead `everything` (+0.016).
- A model with a large gain on one labelling and none on the other ranks by the
  none - deliberately.
- The absolute leaderboard score still depends mostly on which labelling the
  test uses (dataset0's labels score only 0.326 on data1); this rule only picks
  the best model we control.

## How to check it still holds

The Decisions view's run table is sorted by `xsrc/gain_worst`, descending.
