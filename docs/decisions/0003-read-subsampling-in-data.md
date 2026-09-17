# 0003. Read subsampling belongs in `data.py`, and the draw is keyed not streamed

- **Date:** 2026-09-16
- **Status:** Accepted
- **Affects:** `src/m6a/data.py` (`subsample_reads`, `SUBSAMPLE_SEED`)

## Context

Every site in the training set has at least 20 reads. SG-NEx, the data Task 2
runs on, has a median of 3 and a 25th percentile of 1. To find out what that
does to a model, you drop reads from held-out data and score it again.

Two questions had to be answered: where the code lives, and how the draw is
made reproducible.

## Decision

`m6a.data.subsample_reads(site, depth, seed=SUBSAMPLE_SEED)` returns a `Site`
with `depth` of its reads, drawn without replacement. A site at or below `depth`
is returned unchanged.

The draw is keyed on a hash of **(seed, depth, transcript_id, position)** rather
than taken from one long RNG stream.

`SUBSAMPLE_SEED = 4262` — the same number as the split seed, but a **separate
knob**. Changing it does not invalidate any stored fold assignment; it only
redraws which reads are kept. It is recorded in every report.

## Why this and not the alternatives

**Put it in the evaluation harness.** Rejected. Subsampling is a property of the
data, not of how we score it. The depth sweep uses it to build low-depth *test*
sets; depth-augmented training — the obvious next experiment, and named in
GAPS.md as the missing controlled comparison — needs the same function to build
low-depth *training* rows. Writing it once means one implementation, one seed,
one meaning. Writing it twice means two, and they will drift.

`data.py` is on the frozen-import path (AGENTS.md section 4), so this is only
safe because the function needs nothing beyond numpy and `hashlib`.

**One shared RNG stream, advanced across sites** — what the original throwaway
script did. Rejected: it is reproducible only as long as nobody edits the depth
list and nothing changes the order sites are visited in. Add a depth to the
sweep and every previously computed depth silently changes.

Keyed, `subsample_reads(s, 3)` returns the same three reads whether you swept
`[1, 3]` or `[1, 3, 5, 10, 20]`, in forward or reverse order, on any machine.
That is worth a hash per site.

This is why the depth-sweep numbers differ from the previously recorded ones in
the third decimal (0.1527 vs 0.1543 at depth 1). Same method, different draw.

## Consequences

- Depth numbers move slightly if `SUBSAMPLE_SEED` changes, so it is reported
  alongside them. Varying it deliberately is the way to check a depth result is
  not an artefact of one draw — nobody has done that yet.
- Subsampling simulates **covariate** shift, not **label** shift. A well-covered
  site with reads removed at random is not the same as a site that was genuinely
  poorly covered: real low-depth sites are low-depth because the RNA was rare,
  and the m6ACE-Seq labelling assay is itself sequencing-based, so its
  sensitivity drops on exactly those transcripts. Real-world degradation is
  likely worse than the sweep shows, and this method structurally cannot see it.
  Recorded in GAPS.md under "Understanding the data".
- Reads are kept in file order (the draw is sorted). Every feature we compute is
  order-invariant, so this costs nothing and makes a subsample diffable.

## How to check it still holds

`tests/test_evaluation.py::test_subsampling_does_not_depend_on_iteration_order_or_the_depth_list`
is the one that matters — it is the property the keyed draw exists to provide.

Also `::test_subsampling_keeps_real_reads_and_never_invents_any`.
