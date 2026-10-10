"""The numpy site-graph ensemble scores exactly what the torch networks score.

predict.py ships m6a.models.site_graph, a numpy re-implementation of the
networks trained in analysis/representation/graph.py (docs/decisions/0033). A
silent mismatch - a layer order, a missing log transform, a different variance
- would ship a model that scores plausibly and worse than the one evaluated.
This exports randomly initialised networks of both shipped kinds and compares
both implementations on the same synthetic transcripts.

Needs torch (the train extra); skipped where it is not installed, as on an
evaluator's machine - which is the point of the numpy version.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analysis" / "representation"))

import common  # noqa: E402
import export_final  # noqa: E402
import graph  # noqa: E402
import xsrc_nets as X  # noqa: E402
from m6a.data import ReadBlocks, Site  # noqa: E402
from m6a.models.site_graph import SiteGraphEnsemble, kmer_onehot  # noqa: E402

# Every kind ever shipped: the 0033 pair, and 0034's constrained designs.
SHIPPED = ("h2gcn_twohead_aux", "h2gcn_aux", "res_gate", "scalar_drop")


def synthetic_sites(seed: int = 0) -> list[Site]:
    rng = np.random.default_rng(seed)
    sites = []
    # Transcripts of 1, 4 and 9 sites; spacings straddle every window (50, 75, 150, 200 nt).
    for t, n in (("T1", 1), ("T2", 4), ("T3", 9)):
        positions = np.cumsum(rng.integers(5, 120, n))
        for p in positions:
            k = int(rng.choice([1, 2, 7, 30]))           # includes a 1-read site
            reads = np.abs(rng.normal(1.0, 0.5, (k, 9))).astype(np.float32) + 1e-3
            kmer = "".join(rng.choice(list("ACGT"), 7))
            sites.append(Site(t, int(p), kmer, reads))
    return sites


def torch_logits(model, sites, mean, scale) -> np.ndarray:
    counts = [len(s.reads) for s in sites]
    blocks = ReadBlocks(np.concatenate([s.reads for s in sites]),
                        np.concatenate([[0], np.cumsum(counts)]).astype(np.int64))
    std = common.Standardiser(mean, scale)
    ids = np.array([s.transcript_id for s in sites])
    _, gid = np.unique(ids, return_inverse=True)
    position = np.array([s.position for s in sites])
    graphs = graph.graphs_of(np.arange(len(sites)), gid, position)
    got = graph.score(model, graphs, std(blocks.values), blocks.offsets,
                      kmer_onehot([s.kmer for s in sites]), position)
    return np.array([got[i] for i in range(len(sites))])


@pytest.mark.parametrize("name", SHIPPED)
def test_numpy_matches_torch(tmp_path, name):
    torch.manual_seed(0)
    model = graph.GraphNet(**X.GRAPH_MODELS[name]).eval()
    if hasattr(model, "kernel_nt"):
        model.kernel_nt = torch.tensor(113.0)      # a fitted scale, not the 90 nt starting value
    if hasattr(model, "gate_params"):
        with torch.no_grad():                      # a gate that varies, not sigmoid(0) everywhere
            model.gate_params.copy_(torch.tensor([-0.12, -0.18, 0.12]))
    mean = np.random.default_rng(1).normal(0, 0.3, 9).astype(np.float32)
    scale = np.random.default_rng(2).uniform(0.5, 2.0, 9).astype(np.float32)
    export_final.export_network({"model": name, "seed": 0, "kwargs": X.GRAPH_MODELS[name],
                                 "state_dict": model.state_dict(), "read_mean": mean,
                                 "read_scale": scale}, tmp_path / "net.npz")
    (tmp_path / "meta.json").write_text(json.dumps({"name": "t", "model": "site_graph_ensemble",
                                                    "networks": ["net.npz"]}))
    sites = synthetic_sites()
    expected = torch_logits(model, sites, mean, scale)
    net = SiteGraphEnsemble.load(tmp_path).networks[0]
    counts = np.array([len(s.reads) for s in sites])
    offsets = np.concatenate([[0], np.cumsum(counts)])
    s = net.site_vectors(np.concatenate([s.reads for s in sites]), offsets,
                         kmer_onehot([x.kmer for x in sites]))
    got = np.empty(len(sites))
    for t in ("T1", "T2", "T3"):
        rows = np.array([i for i, x in enumerate(sites) if x.transcript_id == t])
        got[rows] = net.logits(s[rows], np.array([sites[i].position for i in rows], dtype=np.float32),
                               counts[rows])
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_ensemble_output_is_a_score_per_site(tmp_path):
    torch.manual_seed(0)
    names = []
    for i, name in enumerate(SHIPPED):
        model = graph.GraphNet(**X.GRAPH_MODELS[name]).eval()
        export_final.export_network({"model": name, "seed": 0, "kwargs": X.GRAPH_MODELS[name],
                                     "state_dict": model.state_dict(), "read_mean": np.zeros(9, np.float32),
                                     "read_scale": np.ones(9, np.float32)}, tmp_path / f"n{i}.npz")
        names.append(f"n{i}.npz")
    (tmp_path / "meta.json").write_text(json.dumps({"name": "t", "model": "site_graph_ensemble",
                                                    "networks": names}))
    sites = synthetic_sites(3)
    out = SiteGraphEnsemble.load(tmp_path).predict_sites(sites)
    assert list(out.columns) == ["transcript_id", "transcript_position", "score"]
    assert len(out) == len(sites) and out["score"].between(0, 1).all()
    assert [(r.transcript_id, r.transcript_position) for r in out.itertuples()] == \
           [(s.transcript_id, s.position) for s in sites]
