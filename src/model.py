"""
model.py -- PART 1 GraphSAGE model for edge-level "attack propensity"
prediction.

TASK
----
Binary classification of every *existing* directed edge ``u -> v`` in an
attack graph: is this edge part of the labelled attack path (1) or not (0)?
The model outputs a logit per edge; ``sigmoid(logit)`` in [0, 1] is the
"attack propensity" that PARTS 2-3 use as an edge weight.

ARCHITECTURE
------------
    node features x  (V, F=19)
          |
    [ SAGEConv -> ReLU -> Dropout ]  x  n_layers   (GraphSAGE, mean aggregator)
          |
    node embeddings h  (V, H)
          |
    for edge (u, v):  z = concat( h[u], h[v], edge_type_multihot[u,v] )   (2H + E)
          |
    [ Linear -> ReLU -> Dropout -> Linear ]   (edge MLP head)
          |
    logit  (1 per edge)      -- BCEWithLogitsLoss(pos_weight=...) during training

GraphSAGE (Hamilton et al., 2017) is a good fit here: it learns an inductive
aggregator over a node's neighbourhood, so a model trained on some AD graphs
generalises to unseen graphs of the same schema (all 361-node here, but the
mechanism is size-agnostic).

WHY CONCAT h[u] || h[v] FOR THE EDGE SCORE
-----------------------------------------
Attack-path membership is not symmetric (``u -> v`` may be on the path while
``v -> u`` is not), so an order-sensitive combiner is required; plain
concatenation keeps that asymmetry. The edge-type multi-hot is appended because
some AD relations (e.g. ``DCSync``, ``GenericAll``) are far more attack-relevant
than others.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEdgeClassifier(nn.Module):
    """
    GraphSAGE node encoder + MLP edge-scoring head.

    Parameters
    ----------
    in_channels
        Node feature dimension F (19 for this dataset).
    edge_dim
        Per-edge feature dimension E (16 edge-type multi-hot).
    hidden_channels
        Width H of every GraphSAGE layer and the head's hidden layer.
    num_layers
        Number of SAGEConv layers (2 or 3; brief asks for "2-3").
    dropout
        Dropout probability applied after each conv and inside the head.
    """

    def __init__(
        self,
        in_channels: int = 19,
        edge_dim: int = 16,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2 for a GraphSAGE stack.")
        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # --- GraphSAGE node encoder ------------------------------------- #
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="mean"))
        for _ in range(num_layers - 1):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr="mean")
            )

        # --- Edge-scoring head ---------------------------------------------- #
        # input = [h_u | h_v | edge_type_multihot]  =  2H + E
        head_in = 2 * hidden_channels + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(head_in, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.convs:
            conv.reset_parameters()
        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------------------- #
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run the GraphSAGE stack; return node embeddings ``h`` (V, H)."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            # No activation/dropout after the final layer -- keep the embedding
            # space linear for the head to work with.
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def score_edges(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a set of edges given node embeddings.

        ``edge_index`` here is the set of edges to *score* (normally the same as
        the message-passing connectivity, but kept separate so you could score
        candidate/negative edges too). Returns a 1-D logit tensor of length
        ``edge_index.shape[1]``.
        """
        src, dst = edge_index[0], edge_index[1]
        z = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.edge_mlp(z).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        score_edge_index: Optional[torch.Tensor] = None,
        score_edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full pass: encode nodes, then score edges.

        By default the scored edges are the message-passing edges themselves.
        Pass ``score_edge_index`` / ``score_edge_attr`` to score a different
        set.

        Returns
        -------
        logits : (M,) tensor -- apply ``torch.sigmoid`` for propensities.
        """
        h = self.encode(x, edge_index)
        if score_edge_index is None:
            score_edge_index = edge_index
            score_edge_attr = edge_attr
        if score_edge_attr is None:
            raise ValueError(
                "score_edge_attr must be provided when score_edge_index is."
            )
        return self.score_edges(h, score_edge_index, score_edge_attr)

    @torch.no_grad()
    def predict_propensity(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Convenience: sigmoid of the logits, in eval mode, no grad."""
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(x, edge_index, edge_attr)
            return torch.sigmoid(logits)
        finally:
            self.train(was_training)

    def config(self) -> dict:
        """Hyper-parameters, for stamping into checkpoints."""
        return {
            "in_channels": self.in_channels,
            "edge_dim": self.edge_dim,
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "arch": "GraphSAGEEdgeClassifier",
        }


def build_model_from_config(cfg: dict) -> "GraphSAGEEdgeClassifier":
    """Recreate a model from the dict stored by ``GraphSAGEEdgeClassifier.config``."""
    return GraphSAGEEdgeClassifier(
        in_channels=cfg["in_channels"],
        edge_dim=cfg["edge_dim"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.3),
    )


# --------------------------------------------------------------------------- #
# Standalone shape check -- runs on CPU, no dataset needed.
# --------------------------------------------------------------------------- #
def _demo() -> None:
    torch.manual_seed(0)
    V, F_in, E, M = 361, 19, 16, 2904

    x = torch.rand(V, F_in)
    edge_index = torch.randint(0, V, (2, M))
    edge_attr = (torch.rand(M, E) > 0.9).float()

    model = GraphSAGEEdgeClassifier(in_channels=F_in, edge_dim=E,
                                    hidden_channels=64, num_layers=3)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\ntrainable parameters: {n_params:,}")

    logits = model(x, edge_index, edge_attr)
    print(f"logits shape        : {tuple(logits.shape)}  (expected ({M},))")
    probs = torch.sigmoid(logits)
    print(f"propensity range    : [{probs.min():.3f}, {probs.max():.3f}]")

    assert logits.shape == (M,)
    # Gradient flows?
    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros(M))
    loss.backward()
    grad_norm = sum(
        p.grad.abs().sum() for p in model.parameters() if p.grad is not None
    )
    assert grad_norm > 0, "no gradient reached the parameters"
    print(f"backward OK, total |grad| = {grad_norm:.3f}")
    print("\nShape + gradient self-checks passed.")


if __name__ == "__main__":
    _demo()
