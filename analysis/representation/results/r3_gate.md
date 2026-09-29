# Round-3 gate

Rule, fixed before the results: mean held-out gain over deepset@v1 >= +0.005 AND wins >= 4/5.

| candidate | mean gain | wins | passes |
|---|---:|---:|---|
| res_deepset@v2 | +0.0059 | 3/5 | no |
| res_attn_mil@v2 | -0.0068 | 3/5 | no |
| set_transformer@v2 | -0.0055 | 1/5 | no |
| deepset@v2 | +0.0083 | 3/5 | (recipe control) |
| attn_mil@v2 | -0.0025 | 2/5 | (recipe control) |

**Decision:** no candidate passes; stop here - deepset stays.
