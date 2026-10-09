"""Which SG-NEx run does each course file come from?

    python analysis/newdata/sgnex_source_match.py <repo root>

For every m6Anet-processed SG-NEx run in the public bucket, the share of a
course file's sites present in the run and the share with the SAME read count.
An identical read count at every site means the file was cut from that run.
Writes analysis/newdata/sgnex_source_match.csv.
"""
import gzip, io, json, sys, urllib.request
import pandas as pd

ROOT = sys.argv[1]
BUCKET = "https://sg-nex-data.s3.amazonaws.com/"
PREFIX = "data/processed_data/m6Anet/"


def counts(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            for t, ps in json.loads(line).items():
                for p, km in ps.items():
                    for _, reads in km.items():
                        rows.append((t, int(p), len(reads)))
    return pd.DataFrame(rows, columns=["transcript_id", "transcript_position", "n_reads"])


course = {"dataset0": counts(f"{ROOT}/data0/dataset0.json.gz"),
          "data1": counts(f"{ROOT}/data0/data1/dataset1.json.gz")}
for k, v in course.items():
    print(k, len(v), "sites", flush=True)

xml = urllib.request.urlopen(f"{BUCKET}?list-type=2&prefix={PREFIX}&delimiter=/").read().decode()
samples = [s.split("/")[-2] for s in xml.split("<Prefix>")[2:] for s in [s.split("</Prefix>")[0]]]
out = []
for s in samples:
    raw = urllib.request.urlopen(f"{BUCKET}{PREFIX}{s}/data.readcount").read()
    rc = pd.read_csv(io.BytesIO(raw))
    rc.columns = ["transcript_id", "transcript_position", "n_reads"]
    rc["transcript_id"] = rc.transcript_id.str.split(".").str[0]
    for k, v in course.items():
        v2 = v.assign(transcript_id=v.transcript_id.str.split(".").str[0])
        m = v2.merge(rc, on=["transcript_id", "transcript_position"], how="left", suffixes=("", "_sg"))
        out.append({"course": k, "sample": s, "present %": 100 * m.n_reads_sg.notna().mean(),
                    "same n_reads %": 100 * (m.n_reads == m.n_reads_sg).mean(),
                    "sample sites": len(rc)})
        print(out[-1], flush=True)
pd.DataFrame(out).to_csv(f"{ROOT}/analysis/newdata/sgnex_source_match.csv", index=False)
