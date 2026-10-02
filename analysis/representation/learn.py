"""Train one encoder on one fold's half-A genes and save its site vectors.

    python analysis/representation/learn.py --model deepset --fold 0
    python analysis/representation/learn.py --model deepset --fold 0 --limit 5000   # smoke

Writes .cache/representation/<model>/fold<k>.npz with:
  z       (n_sites, d)  site vectors; NaN on the encoder's own training rows,
                        which no probe may use (PLAN.md section 3)
  z_d1/3/10  (n_heldout, d)  the held-out sites embedded from 1 / 3 / 10 reads
  info    training record: epochs, minutes, loss per epoch, validation AP
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score

import common
import encoders as E

BATCH = 256

SPECS = {
    # name: (kind, supervised)
    "hand": ("static", False),
    "hand_pca": ("static", False),
    "random": ("set", False),
    "mlp_hand": ("hand", True),
    "deepset": ("set", True),
    "read_ae": ("read", False),
    "read_ae_pred": ("read", True),
    "pos_lstm_ae": ("read", False),
    "pos_attn_ae": ("read", False),
    "set_masked": ("set", False),
    "attn_mil": ("set", True),
    "contrastive": ("set", False),
    # Added after deepset's result, to find out why it works (see REPORT.md).
    "deepset_nokmer": ("set", True),
    "deepset_large": ("set", True),
    "kmer_norm": ("kmer_norm", False),
    # Added after the fit diagnostic showed underfitting (results/diagnose_fit.json).
    "res_deepset": ("set", True),
    "res_attn_mil": ("set", True),
    "set_transformer": ("set", True),
}
# Set encoders trained with a plain BCE loss on the site label.
SUPERVISED_SET = ("deepset", "attn_mil", "deepset_nokmer", "deepset_large",
                  "res_deepset", "res_attn_mil", "set_transformer")


def build(name: str, n_hand: int):
    return {
        "random": lambda: E.DeepSet(),
        "mlp_hand": lambda: E.HandMLP(n_hand),
        "deepset": lambda: E.DeepSet(),
        "read_ae": lambda: E.ReadAE(),
        "read_ae_pred": lambda: E.ReadAE(predict=True),
        "pos_lstm_ae": lambda: E.PositionLSTMAE(),
        "pos_attn_ae": lambda: E.PositionAttentionAE(),
        "set_masked": lambda: E.SetMasked(),
        "attn_mil": lambda: E.AttentionMIL(),
        "contrastive": lambda: E.Contrastive(),
        "deepset_nokmer": lambda: E.DeepSet(read_kmer=False),
        # "train larger": twice the width, twice the site vector.
        "deepset_large": lambda: E.DeepSet(width=128, z=64),
        "res_deepset": lambda: E.ResidualDeepSet(),
        "res_attn_mil": lambda: E.ResidualAttentionMIL(),
        "set_transformer": lambda: E.SetTransformer(),
    }[name]()


def t(x) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x))


# ------------------------------------------------------------------ embedding

@torch.no_grad()
def embed_sets(model, values, offsets, rows, kmer, windows, quadratic=False):
    out = {}
    for sites in common.site_batches(offsets, rows, quadratic=quadratic):
        reads, mask = common.pad_all(values, offsets, sites)
        z = model.embed(t(reads), t(mask), t(kmer[sites]), t(windows[sites]))
        out.update(zip(sites.tolist(), z.numpy()))
    return np.stack([out[i] for i in rows])


@torch.no_grad()
def site_logits(model, values, offsets, rows, kmer, windows):
    out = {}
    # Attention across reads costs reads^2 per site; batch accordingly.
    quadratic = getattr(model, "QUADRATIC", False)
    for sites in common.site_batches(offsets, rows, quadratic=quadratic):
        reads, mask = common.pad_all(values, offsets, sites)
        fn = model.site_logit if hasattr(model, "site_logit") else model
        out.update(zip(sites.tolist(), fn(t(reads), t(mask), t(kmer[sites]), t(windows[sites])).numpy()))
    return np.array([out[i] for i in rows])


@torch.no_grad()
def read_codes(model, values, offsets, rows, kmer, windows, chunk=131_072):
    """Codes and reconstruction errors for every read of `rows`, flat, plus the
    offsets of those rows within the flat result."""
    counts = np.diff(offsets)[rows]
    index = np.concatenate([np.arange(offsets[s], offsets[s + 1]) for s in rows])
    owner = np.repeat(rows, counts)
    codes, errors = [], []
    for start in range(0, len(index), chunk):
        part, who = index[start:start + chunk], owner[start:start + chunk]
        c, e = model.read_codes(t(values[part]), t(kmer[who]), t(windows[who]))
        codes.append(c.numpy())
        errors.append(e.numpy())
    return np.concatenate(codes), np.concatenate(errors), np.concatenate([[0], np.cumsum(counts)])


# ------------------------------------------------------------------ training

PATIENCE = 30   # supervised: stop after this many epochs without a better validation AP


def train(model, name, kind, supervised, bundle, roles, values, minutes, epochs, fold, log,
          track_train: int = 0, recipe: str = "v1"):
    """recipe v1: Adam 1e-3, constant rate, patience 30 - every recorded run.
    recipe v2: AdamW (weight decay 1e-4), linear warm-up over 3 epochs then
    cosine decay over `epochs`, patience 60. Added after the fit diagnostic
    showed networks still improving on their training data when v1 stopped."""
    offsets = bundle.reads.offsets
    kmer, windows, y = bundle.kmer_onehot, bundle.window_onehot, bundle.y
    rows = np.flatnonzero(roles["encoder"])
    val = np.flatnonzero(roles["encoder_val"])
    rng = np.random.default_rng(4262 + fold)
    batcher = common.SetBatcher(values, offsets, seed=4262 + fold)
    positives = y[rows].sum()
    pos_weight = torch.tensor([(len(rows) - positives) / max(positives, 1)])
    if recipe == "v2":
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        per_epoch = max(1, -(-len(rows) // BATCH))
        warm, total = 3 * per_epoch, max(epochs * per_epoch, 1)
        schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: min(
            (s + 1) / warm, 0.5 * (1 + np.cos(np.pi * min(s, total) / total))))
        patience = 60
    else:
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        schedule, patience = None, PATIENCE
    hand = getattr(bundle, "_hand_std", None)

    history, best, best_ap = [], None, -1.0
    started = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        order = rng.permutation(rows)
        for i in range(0, len(order), BATCH):
            sites = order[i:i + BATCH]
            k, target = t(kmer[sites]), t(y[sites])
            if kind == "hand":
                loss = F.binary_cross_entropy_with_logits(model(t(hand[sites])), target,
                                                          pos_weight=pos_weight)
            elif name == "contrastive":
                first, second = batcher.pair(sites)
                loss = model.pair_loss((t(first[0]), t(first[1])), (t(second[0]), t(second[1])), k)
            else:
                reads, mask = batcher.batch(sites)
                reads, mask, w = t(reads), t(mask), t(windows[sites])
                if name in SUPERVISED_SET:
                    loss = F.binary_cross_entropy_with_logits(model(reads, mask, k), target,
                                                              pos_weight=pos_weight)
                elif name == "read_ae_pred":
                    loss = model.loss(reads, mask, k, y=target, pos_weight=pos_weight)
                else:
                    loss = model.loss(reads, mask, k, w)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if schedule is not None:
                schedule.step()
            losses.append(loss.item())
            if time.time() - started > minutes * 60:
                break
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
                  "minutes": round((time.time() - started) / 60, 2)}
        if supervised and len(val):
            model.eval()
            if kind == "hand":
                with torch.no_grad():
                    scores = model(t(hand[val])).numpy()
            else:
                scores = site_logits(model, values, offsets, val, kmer, windows)
            # A soft target (xsrc_deepset.py's pooled_both: two labellings
            # averaged) cannot score AP; such a bundle carries 0/1 `y_val`.
            y_val = getattr(bundle, "y_val", y)
            record["val_ap"] = float(average_precision_score(y_val[val], scores))
            if track_train:
                # The underfit/overfit diagnostic: the same metric on a fixed
                # sample of the network's OWN training sites, all reads.
                seen = np.random.default_rng(0).choice(rows, min(track_train, len(rows)), replace=False)
                fit = site_logits(model, values, offsets, seen, kmer, windows)
                record["train_ap"] = float(average_precision_score(y[seen], fit))
            if record["val_ap"] > best_ap:
                best_ap, best = record["val_ap"], copy.deepcopy(model.state_dict())
        history.append(record)
        since_best = len(history) - 1 - max(
            (i for i, h in enumerate(history) if h.get("val_ap") == best_ap), default=len(history) - 1)
        if supervised and since_best >= patience:
            log(f"    stopping: no validation gain in {patience} epochs")
            break
        log(f"    epoch {record['epoch']}: " + ", ".join(f"{k} {v}" for k, v in record.items()
                                                       if k != "epoch"))
        if time.time() - started > minutes * 60:
            break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return history


def kmer_normalised(bundle, roles, held, rows):
    """Each read standardised against the average read for its 7-mer, then
    summarised per site. No network: the cheap version of 'read the current in
    the context of its sequence'. Per-7-mer statistics come from the encoder
    rows only; a 7-mer seen fewer than 200 times there falls back to global."""
    blocks = bundle.reads
    kmer_id = bundle.kmer_onehot.reshape(len(bundle.y), 7, 4).argmax(-1) @ (4 ** np.arange(7))
    values = common.transform_values(blocks.values)
    owner = np.repeat(np.arange(len(bundle.y)), blocks.counts)
    train_reads = roles["encoder"][owner]
    ids = kmer_id[owner]
    g_mean, g_sd = values[train_reads].mean(0), values[train_reads].std(0)
    mean = np.tile(g_mean, (4 ** 7, 1))
    sd = np.tile(g_sd, (4 ** 7, 1))
    counts = np.bincount(ids[train_reads], minlength=4 ** 7)
    for k in np.flatnonzero(counts >= 200):
        v = values[train_reads & (ids == k)]
        mean[k], sd[k] = v.mean(0), np.maximum(v.std(0), 1e-6)

    def pooled(block, site_rows):
        vals = common.transform_values(block.values)
        own = np.repeat(np.arange(len(bundle.y)), block.counts)
        norm = (vals - mean[kmer_id[own]]) / sd[kmer_id[own]]
        index = np.concatenate([np.arange(block.offsets[s], block.offsets[s + 1]) for s in site_rows])
        local = np.concatenate([[0], np.cumsum(block.counts[site_rows])])
        return common.pool_codes(norm[index], None, local, None, block.counts[site_rows])

    z = np.full((len(bundle.y), 9 * 5 + 1), np.nan, dtype=np.float32)
    z[rows] = pooled(blocks, rows)
    return z, {d: pooled(bundle.depth_reads[d], held) for d in common.EVAL_DEPTHS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SPECS))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=8.0, help="training time box per fold")
    ap.add_argument("--epochs", type=int, default=400)
    args = ap.parse_args()
    started = time.time()
    bundle = common.load(args.limit)
    print(f"[{args.model}] data loaded ({time.time() - started:.0f}s)", flush=True)
    for fold in args.folds:
        run_fold(args, fold, bundle)


def run_fold(args, fold: int, bundle) -> None:
    args.fold = fold
    torch.manual_seed(4262 + args.fold)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    started = time.time()
    roles = common.split_roles(bundle, args.fold)
    kind, supervised = SPECS[args.model]
    use = roles["probe"] | roles["heldout"]
    held = np.flatnonzero(roles["heldout"])
    log(f"[{args.model} fold {args.fold}] {len(bundle.y):,} sites: encoder "
        f"{roles['encoder'].sum():,} (+{roles['encoder_val'].sum():,} val), probe "
        f"{roles['probe'].sum():,}, held out {len(held):,}  ({time.time() - started:.0f}s)")

    hand_mean = bundle.X[roles["encoder"]].mean().to_numpy()
    hand_sd = bundle.X[roles["encoder"]].std().replace(0, 1).to_numpy()
    standardise_hand = lambda X: ((X.to_numpy() - hand_mean) / hand_sd).astype(np.float32)  # noqa: E731
    bundle._hand_std = standardise_hand(bundle.X)

    z = None
    info: dict = {"model": args.model, "fold": args.fold, "limit": args.limit}
    depth_z = {}
    if kind == "kmer_norm":
        z, depth_z = kmer_normalised(bundle, roles, held, np.flatnonzero(use))
    elif kind == "static":
        if args.model == "hand":
            project = lambda A: A  # noqa: E731
        else:
            pca = PCA(32, random_state=4262).fit(bundle._hand_std[roles["encoder"]])
            project = pca.transform
            info["explained_variance"] = float(pca.explained_variance_ratio_.sum())
        z = project(bundle._hand_std)
        depth_z = {d: project(standardise_hand(bundle.depth_X[d]))[held] for d in common.EVAL_DEPTHS}
    else:
        model = build(args.model, bundle.X.shape[1])
        info["parameters"] = int(sum(p.numel() for p in model.parameters()))
        std = common.Standardiser.fit(bundle.reads, roles["encoder"])
        values = std(bundle.reads.values)
        log(f"  {info['parameters']:,} parameters; reads standardised ({time.time() - started:.0f}s)")
        if args.model != "random":
            info["history"] = train(model, args.model, kind, supervised, bundle, roles, values,
                                    args.minutes, args.epochs, args.fold, log)
        model.eval()
        log(f"  trained ({time.time() - started:.0f}s); embedding")

        kmer, windows = bundle.kmer_onehot, bundle.window_onehot
        if kind == "hand":
            with torch.no_grad():
                z = model.embed(t(bundle._hand_std)).numpy()
                depth_z = {d: model.embed(t(standardise_hand(bundle.depth_X[d])[held])).numpy()
                           for d in common.EVAL_DEPTHS}
        elif kind == "set":
            quad = args.model == "set_masked"
            rows = np.flatnonzero(use)
            part = embed_sets(model, values, bundle.reads.offsets, rows, kmer, windows, quad)
            z = np.full((len(bundle.y), part.shape[1]), np.nan, dtype=np.float32)
            z[rows] = part
            for d in common.EVAL_DEPTHS:
                blocks = bundle.depth_reads[d]
                depth_z[d] = embed_sets(model, std(blocks.values), blocks.offsets, held,
                                        kmer, windows, quad)
        else:  # per-read codes, pooled per site
            sample = np.random.default_rng(4262).choice(
                np.flatnonzero(roles["encoder"]), size=min(3000, roles["encoder"].sum()),
                replace=False)
            _, errors, _ = read_codes(model, values, bundle.reads.offsets, sample, kmer, windows)
            threshold = float(np.quantile(errors, 0.95))
            info["error_threshold"] = threshold

            def pooled(vals, offsets, rows):
                codes, errs, local = read_codes(model, vals, offsets, rows, kmer, windows)
                return common.pool_codes(codes, errs, local, threshold, np.diff(offsets)[rows])

            rows = np.flatnonzero(use)
            part = pooled(values, bundle.reads.offsets, rows)
            z = np.full((len(bundle.y), part.shape[1]), np.nan, dtype=np.float32)
            z[rows] = part
            for d in common.EVAL_DEPTHS:
                blocks = bundle.depth_reads[d]
                depth_z[d] = pooled(std(blocks.values), blocks.offsets, held)

    z = np.asarray(z, dtype=np.float32)
    z[roles["encoder"] | roles["encoder_val"]] = np.nan
    info["dims"] = int(z.shape[1])
    info["minutes_total"] = round((time.time() - started) / 60, 2)
    common.save_embedding(args.model, args.fold, z, depth_z, info)
    log(f"  saved {z.shape[1]}-dim vectors -> {common.embedding_path(args.model, args.fold)} "
        f"({info['minutes_total']} min)")


if __name__ == "__main__":
    main()
