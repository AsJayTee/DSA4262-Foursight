# Decision records

One file per architectural decision about shared infrastructure: **what** was
decided, **why**, and **what it costs**. So that anyone — teammate or agent —
can find out why the pipeline is shaped the way it is without reading the
source or asking the person who wrote it.

Adding an *experiment* does not need a record. A new feature set or model is
additive by design and explains itself (see [AGENTS.md](../../AGENTS.md)
section 1). Write a record when you change something that experiments sit on
top of:

- anything in `src/m6a/*.py` (not `features/` or `models/`)
- the scripts, the config schema, or a file format anything reads
- how the split, the metrics, or the comparison between runs works
- a new dependency, or a decision about where a dependency may be imported

## Why one file per decision

Four people work in parallel here, each through their own agent. A single
shared log file would produce a merge conflict every time two of them recorded
something in the same week. One file per decision never conflicts — the same
reason the registry discovers modules by directory scan instead of a shared
table.

The directory listing is the index. Do not add an index file; it would conflict.

## Format

`NNNN-short-kebab-title.md`, four digits, next unused number. Copy this:

```markdown
# NNNN. Title in plain words

- **Date:** YYYY-MM-DD
- **Status:** Proposed | Accepted | Amended by NNNN | Superseded by NNNN
- **Affects:** the files or behaviour a reader would notice this in

## Context
What was true before, and what problem it caused. Include the measurement that
made it a problem if there is one.

## Decision
What we do now. Be specific enough to disagree with.

## Why this and not the alternatives
The options that were on the table and why they lost. This is the section
people actually come back for.

## Consequences
What this costs, what it makes harder, and what it rules out later.

## How to check it still holds
A command, a test name, or a number someone can regenerate.
```

Keep it short. A record that takes ten minutes to write is a record that gets
written; the value is in it existing, not in it being thorough.

**Statuses.** `Proposed` means agreed but not built — say so plainly rather
than describing unbuilt things in the present tense. When it ships, change the
status in the same commit.

When a decision is replaced, set it to `Superseded by NNNN`. When only part of
it is reversed, `Amended by NNNN`. Either way **leave the original text alone**
and let the newer record carry the change: the reasoning that turned out to be
wrong is more useful than a clean history, and someone will otherwise re-derive
it from scratch.
