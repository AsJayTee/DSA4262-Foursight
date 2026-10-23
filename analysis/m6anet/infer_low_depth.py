"""m6Anet's own inference with its 20-read minimum lowered, for the reduced-depth comparison.

    analysis/m6anet/.venv/bin/python analysis/m6anet/infer_low_depth.py \
        --input_dir data/m6anet/thinned/dataset0_d1 --out_dir data/m6anet/thinned_out/dataset0_d1 \
        --pretrained_model HCT116_RNA002 --n_processes 16 --min_reads 1

Runs in m6Anet's environment (run_ronin.sh setup). m6anet 2.1.0 drops every site with
fewer than DEFAULT_MIN_READS = 20 reads before scoring, and its command line does not
expose that number (scripts/inference.py builds NanopolishDS with the module constant).
Nothing else in the model needs 20 reads: each read is scored on its own, and the site
probability is a noisy-OR over 20 reads sampled WITH replacement
(utils/inference_utils._calculate_site_proba), which is defined for any read count.
So this sets the constant in the inference module, then calls m6Anet's main() unchanged.
"""

import argparse
import sys

import m6anet.scripts.inference as inference


def main() -> None:
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument("--min_reads", type=int, default=1)
    ours, rest = own.parse_known_args()
    inference.DEFAULT_MIN_READS = ours.min_reads
    args = inference.argparser().parse_args(rest)
    print(f"m6Anet inference, minimum reads per site = {ours.min_reads}", flush=True)
    inference.main(args)


if __name__ == "__main__":
    sys.exit(main())
