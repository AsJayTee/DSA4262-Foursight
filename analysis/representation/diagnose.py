"""Is attention-MIL limited by optimisation (underfitting) or by the labels?

    python analysis/representation/diagnose.py

Trains attn_mil and deepset on the canonical fold 0's half-A genes, exactly as
the screen does, logging average precision on 5,000 of the network's own
training sites beside validation every epoch. Then scores the held-out fold.

  train and validation both low and close  -> underfitting: depth/residuals may help
  train far above validation               -> label/information limited
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import common
import learn

bundle = common.load(extra_depths=())
roles = common.split_roles(bundle, 0)
held = np.flatnonzero(roles["heldout"])
out = {}
for name in ("attn_mil", "deepset"):
    torch.manual_seed(4262)
    started = time.time()
    std = common.Standardiser.fit(bundle.reads, roles["encoder"])
    values = std(bundle.reads.values)
    model = learn.build(name, bundle.X.shape[1])
    history = learn.train(model, name, "set", True, bundle, roles, values, 30, 400, 0,
                          lambda *a: print(name, *a, flush=True), track_train=5000)
    model.eval()
    s = learn.site_logits(model, values, bundle.reads.offsets, held, bundle.kmer_onehot,
                          bundle.window_onehot)
    out[name] = {"history": history, "heldout_ap": float(average_precision_score(bundle.y[held], s)),
                 "minutes": round((time.time() - started) / 60, 1)}
    best = max(history, key=lambda h: h["val_ap"])
    print(f"== {name}: best epoch {best['epoch']}  train AP {best['train_ap']:.4f}  "
          f"val AP {best['val_ap']:.4f}  held-out AP {out[name]['heldout_ap']:.4f}  "
          f"final-epoch train AP {history[-1]['train_ap']:.4f}", flush=True)
(common.RESULTS / "diagnose_fit.json").write_text(json.dumps(out, indent=1))
