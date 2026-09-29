# Representation screen: results

Canonical split, 5 folds, probes trained on half of each training fold (test/train ratio 0.49). PR AUC on the held-out fold, mean over folds. Paired tests are Nadeau & Bengio corrected with that ratio. **A screen, not a result to quote against the harness's 0.5408** - see PLAN.md section 3.

## E1. Each representation alone, through five probes

| representation | dims | logistic | kNN | MLP | LightGBM | LightGBM, rotated |
|---|---:|---:|---:|---:|---:|---:|
| `hand` | 137 | 0.4958 | 0.3718 | 0.5057 | 0.5351 | 0.4617 |
| `hand_pca` | 32 | 0.2516 | 0.3252 | 0.4329 | 0.4185 | 0.3627 |
| `random` | 32 | 0.1652 | 0.2017 | 0.1912 | 0.2635 | 0.2631 |
| `mlp_hand` | 32 | 0.4682 | 0.4536 | 0.4648 | 0.4519 | 0.4373 |
| `deepset` | 32 | 0.5088 | 0.4980 | 0.4758 | 0.4791 | 0.4617 |
| `read_ae` | 26 | 0.1938 | 0.2152 | 0.2113 | 0.2284 | 0.2237 |
| `read_ae_pred` | 26 | 0.4494 | 0.4689 | 0.4708 | 0.4616 | 0.4592 |
| `pos_lstm_ae` | 26 | 0.2591 | 0.2867 | 0.3008 | 0.2945 | 0.2839 |
| `pos_attn_ae` | 26 | 0.2265 | 0.2423 | 0.2728 | 0.2643 | 0.2499 |
| `set_masked` | 64 | 0.4509 | 0.3804 | 0.4778 | 0.4434 | 0.4453 |
| `attn_mil` | 32 | 0.5024 | 0.4929 | 0.4943 | 0.4701 | 0.4584 |
| `contrastive` | 32 | 0.3548 | 0.3271 | 0.4401 | 0.4190 | 0.4055 |
| `deepset_nokmer` | 32 | 0.5015 | 0.4858 | 0.4996 | 0.4632 | 0.4567 |
| `kmer_norm` | 46 | 0.2887 | 0.3252 | 0.3563 | 0.3731 | 0.3548 |

## E2. Does the tree need the axes? LightGBM on z vs a random rotation of z

Negative = rotating hurt the tree, so the information sits along the coordinate axes. For `hand` this should be clearly negative (positive control).

| representation | rotated minus plain | wins | p | best non-tree probe minus LightGBM |
|---|---:|---:|---:|---:|
| `hand` | -0.0734 | 0/5 | 0.001 | -0.0294 |
| `hand_pca` | -0.0558 | 0/5 | 0.002 | +0.0144 |
| `random` | -0.0004 | 4/5 | 0.978 | -0.0618 |
| `mlp_hand` | -0.0146 | 0/5 | 0.196 | +0.0163 |
| `deepset` | -0.0174 | 0/5 | 0.091 | +0.0298 |
| `read_ae` | -0.0047 | 2/5 | 0.678 | -0.0133 |
| `read_ae_pred` | -0.0024 | 2/5 | 0.839 | +0.0092 |
| `pos_lstm_ae` | -0.0105 | 0/5 | 0.024 | +0.0064 |
| `pos_attn_ae` | -0.0144 | 1/5 | 0.466 | +0.0085 |
| `set_masked` | +0.0019 | 4/5 | 0.840 | +0.0344 |
| `attn_mil` | -0.0118 | 0/5 | 0.101 | +0.0322 |
| `contrastive` | -0.0135 | 0/5 | 0.110 | +0.0211 |
| `deepset_nokmer` | -0.0065 | 2/5 | 0.494 | +0.0383 |
| `kmer_norm` | -0.0183 | 0/5 | 0.095 | -0.0168 |

## E3. What z adds to the hand features

| representation | + hand, LightGBM: diff / wins / p | + hand, logistic: diff / wins / p |
|---|---|---|
| `hand_pca` | -0.0035 | 2/5 | 0.656 | +0.0000 | 2/5 | 0.924 |
| `random` | -0.0050 | 0/5 | 0.084 | +0.0028 | 4/5 | 0.404 |
| `mlp_hand` | +0.0092 | 5/5 | 0.193 | +0.0330 | 5/5 | 0.003 |
| `deepset` | +0.0309 | 5/5 | 0.004 | +0.0584 | 5/5 | 0.000 |
| `read_ae` | +0.0012 | 2/5 | 0.832 | -0.0004 | 3/5 | 0.865 |
| `read_ae_pred` | +0.0276 | 5/5 | 0.015 | +0.0209 | 5/5 | 0.083 |
| `pos_lstm_ae` | +0.0002 | 3/5 | 0.965 | +0.0009 | 4/5 | 0.555 |
| `pos_attn_ae` | +0.0036 | 4/5 | 0.425 | +0.0004 | 2/5 | 0.864 |
| `set_masked` | +0.0049 | 4/5 | 0.234 | +0.0143 | 5/5 | 0.067 |
| `attn_mil` | +0.0244 | 5/5 | 0.004 | +0.0549 | 5/5 | 0.002 |
| `contrastive` | +0.0053 | 4/5 | 0.409 | +0.0098 | 5/5 | 0.167 |
| `deepset_nokmer` | +0.0236 | 5/5 | 0.005 | +0.0534 | 5/5 | 0.001 |
| `kmer_norm` | +0.0168 | 5/5 | 0.025 | +0.0067 | 5/5 | 0.083 |

Reference: `hand` alone scores LightGBM 0.5351, logistic 0.4958.

## E4. What is in the vector

| representation | R^2 from hand | CKA with hand | R^2 of depth | motif accuracy | effective dims |
|---|---:|---:|---:|---:|---:|
| `hand` | - | - | 1.000 | 1.000 | 18.0 |
| `hand_pca` | 1.000 | 0.614 | 0.891 | 1.000 | 31.9 |
| `random` | 0.661 | 0.392 | 0.212 | 1.000 | 12.1 |
| `mlp_hand` | 0.650 | 0.192 | 0.336 | 0.802 | 3.4 |
| `deepset` | 0.654 | 0.139 | 0.371 | 0.851 | 1.8 |
| `read_ae` | 0.730 | 0.465 | 1.000 | 0.505 | 5.4 |
| `read_ae_pred` | 0.743 | 0.311 | 1.000 | 0.614 | 4.4 |
| `pos_lstm_ae` | 0.817 | 0.601 | 1.000 | 0.704 | 4.9 |
| `pos_attn_ae` | 0.860 | 0.610 | 1.000 | 0.671 | 3.2 |
| `set_masked` | 0.946 | 0.832 | 0.807 | 1.000 | 6.5 |
| `attn_mil` | 0.690 | 0.161 | 0.480 | 0.905 | 1.8 |
| `contrastive` | 0.781 | 0.412 | 0.072 | 1.000 | 12.5 |
| `deepset_nokmer` | 0.659 | 0.171 | 0.345 | 0.940 | 2.3 |
| `kmer_norm` | 0.750 | 0.234 | 1.000 | 0.309 | 14.2 |

## E5. Depth: probes trained at full depth, sites embedded from fewer reads

| representation | 1 read (lgbm / logistic) | 3 reads | 10 reads | full |
|---|---|---|---|---|
| `hand` | 0.1787 / 0.1752 | 0.2655 / 0.2891 | 0.4022 / 0.3885 | 0.5351 / 0.4958 |
| `hand_pca` | 0.0665 / 0.0756 | 0.1143 / 0.1200 | 0.2502 / 0.1639 | 0.4185 / 0.2516 |
| `random` | 0.1177 / 0.1477 | 0.1438 / 0.1525 | 0.1889 / 0.1633 | 0.2635 / 0.1652 |
| `mlp_hand` | 0.0681 / 0.1150 | 0.1503 / 0.1920 | 0.3075 / 0.3302 | 0.4519 / 0.4682 |
| `deepset` | 0.1609 / 0.2022 | 0.2553 / 0.3133 | 0.3881 / 0.4262 | 0.4791 / 0.5088 |
| `read_ae` | 0.0626 / 0.0598 | 0.0849 / 0.0799 | 0.1344 / 0.1241 | 0.2284 / 0.1938 |
| `read_ae_pred` | 0.1430 / 0.1610 | 0.2491 / 0.2570 | 0.3527 / 0.3666 | 0.4616 / 0.4494 |
| `pos_lstm_ae` | 0.0814 / 0.0848 | 0.1222 / 0.1195 | 0.1882 / 0.1871 | 0.2945 / 0.2591 |
| `pos_attn_ae` | 0.0729 / 0.0788 | 0.1112 / 0.1071 | 0.1692 / 0.1644 | 0.2643 / 0.2265 |
| `set_masked` | 0.1568 / 0.1314 | 0.2594 / 0.2177 | 0.3739 / 0.3563 | 0.4434 / 0.4509 |
| `attn_mil` | 0.1925 / 0.2206 | 0.2735 / 0.3161 | 0.3704 / 0.4166 | 0.4701 / 0.5024 |
| `contrastive` | 0.2210 / 0.2091 | 0.2884 / 0.2611 | 0.3671 / 0.3207 | 0.4190 / 0.3548 |
| `deepset_nokmer` | 0.1819 / 0.1940 | 0.2772 / 0.2930 | 0.3812 / 0.4071 | 0.4632 / 0.5015 |
| `kmer_norm` | 0.0920 / 0.0888 | 0.1493 / 0.1306 | 0.2504 / 0.2113 | 0.3731 / 0.2887 |

## Training record

| representation | parameters | epochs (mean) | minutes / fold | best val AP | nonfinite |
|---|---:|---:|---:|---:|---|
| `hand` | - | 0 | 0.0 | - | False |
| `hand_pca` | - | 0 | 0.0 | - | False |
| `random` | 18817 | 0 | 0.1 | - | False |
| `mlp_hand` | 21825 | 120 | 0.9 | 0.5320 | False |
| `deepset` | 18817 | 120 | 6.1 | 0.5607 | False |
| `read_ae` | 13709 | 177 | 26.0 | - | False |
| `read_ae_pred` | 15022 | 61 | 8.7 | 0.5284 | False |
| `pos_lstm_ae` | 14951 | 70 | 26.6 | - | False |
| `pos_attn_ae` | 25671 | 25 | 28.2 | - | False |
| `set_masked` | 18953 | 55 | 28.2 | - | False |
| `attn_mil` | 18914 | 92 | 12.8 | 0.5507 | False |
| `contrastive` | 20929 | 232 | 25.2 | - | False |
| `deepset_nokmer` | 17025 | 114 | 8.2 | 0.5584 | False |
| `kmer_norm` | - | 0 | 0.7 | - | False |
