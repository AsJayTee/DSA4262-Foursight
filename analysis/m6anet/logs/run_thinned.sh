#!/bin/bash
cd /mnt/sdd/DSA4262
for f in dataset0 data1; do for d in 1 3 10; do
  ( analysis/m6anet/.venv/bin/python analysis/m6anet/infer_low_depth.py --input_dir data/m6anet/thinned/${f}_d${d} \
      --out_dir data/m6anet/thinned_out/${f}_d${d} --pretrained_model HCT116_RNA002 --n_processes 10 --min_reads 1 \
      > analysis/m6anet/logs/infer_${f}_d${d}.log 2>&1; echo "${f}_d${d} exit $?" >> analysis/m6anet/logs/infer_done.txt ) &
done; done
wait
echo ALL_DONE >> analysis/m6anet/logs/infer_done.txt
