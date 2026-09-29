# Models trained on dataset0, scored on the later releases

## data1 (PR AUC / ROC AUC; lift = PR AUC over the slice's positive rate)

| model | all data1 (n=90,810, 7.3% pos) | new sites (n=23,490, 7.4% pos) | new transcripts (n=15,137, 8.1% pos) | shared sites (seen) (n=67,320, 7.2% pos) |
|---|---|---|---|---|
| quantiles (0.4759 in CV) | 0.3890 / 0.8227 (5.4x) | 0.3558 / 0.8253 (4.8x) | 0.3689 / 0.8293 (4.6x) | 0.4040 / 0.8217 (5.6x) |
| everything (0.5408 in CV) | 0.4109 / 0.8341 (5.7x) | 0.3636 / 0.8372 (4.9x) | 0.3647 / 0.8373 (4.5x) | 0.4332 / 0.8331 (6.0x) |
| everything minus cross-site | 0.4043 / 0.8357 (5.6x) | 0.3716 / 0.8422 (5.0x) | 0.3756 / 0.8436 (4.7x) | 0.4201 / 0.8337 (5.8x) |

## data2 (mean score per fraction of molecules modified)

| model | 0% | 25% | 50% | 70% | 75% | 95% | 100% | Spearman (site score vs fraction) | ROC AUC 100% vs 0% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| quantiles (0.4759 in CV) | 0.058 | 0.271 | 0.375 | 0.415 | 0.414 | 0.395 | 0.352 | 0.232 | 0.740 |
| everything (0.5408 in CV) | 0.080 | 0.276 | 0.366 | 0.407 | 0.396 | 0.337 | 0.270 | 0.143 | 0.666 |
| everything minus cross-site | 0.141 | 0.442 | 0.523 | 0.547 | 0.540 | 0.494 | 0.465 | 0.192 | 0.737 |
