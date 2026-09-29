# 0027. Later course data releases get their own download set, and keep their folder on disk

- **Date:** 2026-09-29
- **Status:** Accepted
- **Affects:** `scripts/download_data.py` (`PREFIXES`, `KEEP_FOLDER`, `select_keys`), the R2 bucket layout, `data/raw/`

## Context

The course released two more datasets on Canvas (`Team_Project_export.zip`):
`data1/` (a second labelled run) and `data2/` (an in-vitro mixing series). See
[docs/data.md](../data.md#later-releases-data1-and-data2). They were uploaded to
the R2 bucket by hand as `data1/...` and `data2/...`.

`download_data.py` knew only `course` (root objects) and `sgnex/`, and for a
prefixed set it **strips the prefix** from local paths, which suits SG-NEx
(`sgnex/A549/...` becomes `data/raw/A549/...`). Both new folders contain a file
named `data.info`, so stripping would write both to `data/raw/data.info`, and
the second download would silently overwrite the first.

The bucket also now holds zero-byte `data1/` and `data2/` "folder" objects,
which a web console creates. `--set all` tried to download them as files and
would fail on a directory path.

## Decision

- Two new sets, `--set data1` and `--set data2`.
- Sets in `KEEP_FOLDER` keep their folder on disk: `data/raw/data1/data.info`,
  `data/raw/data2/data.info`. SG-NEx keeps its old behaviour.
- Keys ending in `/` are skipped for every prefixed set.

Future releases follow the same pattern: upload under a new top-level folder,
add it to `PREFIXES` and `KEEP_FOLDER`.

## Why this and not the alternatives

**Rename the files on upload** (`data1.info`, `data2.info`). Then the bucket no
longer mirrors what the course distributed, and m6Anet-format tools expect the
name `data.info` beside its `.json`. Keeping the folder costs nothing.

**One `--set releases` for both.** Possible later; two sets are clearer while
the two datasets serve such different purposes.

## Consequences

- Local paths for these datasets differ in shape from SG-NEx's (folder kept vs
  stripped). The table in `KEEP_FOLDER` is the one place that decides it.
- There is still no upload path to R2 (GAPS.md, Infrastructure); this changes
  only downloading.

## How to check it still holds

```bash
python scripts/download_data.py --set data1 --dest /tmp/r2check
python scripts/download_data.py --set data2 --dest /tmp/r2check
```

must produce `/tmp/r2check/data1/{data.info,dataset1.json.gz}` and
`/tmp/r2check/data2/{data.info,dataset2.json.gz}`, with neither `data.info`
overwritten. Checked 2026-09-29: byte-identical to the Canvas export.
