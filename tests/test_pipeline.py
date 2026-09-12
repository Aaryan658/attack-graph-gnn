"""
test_pipeline.py -- CPU-only, dataset-free tests for the shape-detection
loader (src/data_loader.py) and the GraphSAGE model (src/model.py).

No .pt sample files and no GPU are touched: every tensor here is synthetic and
small, built the same way src/data_loader.py's own docstring describes the
real schema (adjacency (V,V,E), features (V,p), target (V,V)).
"""

import pytest
import torch

from src.data_loader import (classify_fields, build_edge_structures,
                             MalformedSampleError, AttackGraphSample)
from src.model import GraphSAGEEdgeClassifier


# --------------------------------------------------------------------------- #
# Shape-based field classification
# --------------------------------------------------------------------------- #
def test_classify_fields_by_shape_not_name():
    V, E, p = 6, 3, 4
    tensors = {
        "totally_unrelated_key_name": torch.zeros(V, V, E),   # 3-D square face
        "another_odd_name": torch.zeros(V, V),                # 2-D square
        "features_or_whatever": torch.zeros(V, p),            # 2-D rectangular
    }
    roles = classify_fields(tensors)
    assert roles["totally_unrelated_key_name"] == "adjacency"
    assert roles["another_odd_name"] == "target"
    assert roles["features_or_whatever"] == "features"


def test_build_edge_structures_extracts_labels_from_target():
    V, E = 4, 2
    adj = torch.zeros(V, V, E)
    adj[0, 1, 0] = 1.0
    adj[1, 2, 1] = 1.0
    y = torch.zeros(V, V, dtype=torch.long)
    y[0, 1] = 1  # this physical edge is on the attack path
    # y[1, 2] left at 0 -- physical edge NOT on the attack path

    edge_index, edge_type_multihot, edge_label = build_edge_structures(adj, y)

    assert edge_index.shape == (2, 2)
    assert edge_type_multihot.shape == (2, E)
    pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert (0, 1) in pairs and (1, 2) in pairs
    label_of = dict(zip(pairs, edge_label.tolist()))
    assert label_of[(0, 1)] == 1.0
    assert label_of[(1, 2)] == 0.0


def test_build_edge_structures_rejects_empty_graph():
    adj = torch.zeros(3, 3, 2)          # no edges at all
    y = torch.zeros(3, 3, dtype=torch.long)
    with pytest.raises(MalformedSampleError):
        build_edge_structures(adj, y)


def test_sample_dense_binary_adjacency_and_edge_list():
    V, E = 3, 2
    adj = torch.zeros(V, V, E)
    adj[0, 1, 0] = 1.0
    adj[2, 0, 1] = 1.0
    y = torch.zeros(V, V, dtype=torch.long)
    edge_index, edge_type_multihot, edge_label = build_edge_structures(adj, y)

    sample = AttackGraphSample(
        adj_tensor=adj, x=torch.zeros(V, 5), y_matrix=y,
        edge_index=edge_index, edge_type_multihot=edge_type_multihot,
        edge_label=edge_label, num_nodes=V,
    )
    dense = sample.dense_binary_adjacency()
    assert dense[0, 1] == 1 and dense[2, 0] == 1 and dense[1, 2] == 0
    assert set(sample.edge_list()) == {(0, 1), (2, 0)}


# --------------------------------------------------------------------------- #
# Model -- forward/backward shape check, CPU only
# --------------------------------------------------------------------------- #
def test_model_forward_backward_shapes():
    torch.manual_seed(0)
    V, F_in, E, M = 20, 19, 16, 60

    x = torch.rand(V, F_in)
    edge_index = torch.randint(0, V, (2, M))
    edge_attr = (torch.rand(M, E) > 0.9).float()

    model = GraphSAGEEdgeClassifier(in_channels=F_in, edge_dim=E,
                                    hidden_channels=16, num_layers=2)
    logits = model(x, edge_index, edge_attr)
    assert logits.shape == (M,)

    probs = torch.sigmoid(logits)
    assert torch.all((probs >= 0) & (probs <= 1))

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.zeros(M)
    )
    loss.backward()
    grad_total = sum(p.grad.abs().sum() for p in model.parameters()
                     if p.grad is not None)
    assert grad_total > 0, "gradient must reach the parameters"


def test_model_config_roundtrip():
    from src.model import build_model_from_config
    model = GraphSAGEEdgeClassifier(in_channels=19, edge_dim=16,
                                    hidden_channels=32, num_layers=2)
    rebuilt = build_model_from_config(model.config())
    assert rebuilt.hidden_channels == 32
    assert rebuilt.num_layers == 2
    assert rebuilt.in_channels == 19


def test_model_rejects_single_layer():
    with pytest.raises(ValueError):
        GraphSAGEEdgeClassifier(num_layers=1)
