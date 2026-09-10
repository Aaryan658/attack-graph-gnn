"""
attack-graph-gnn -- source package.

Layout
------
env_check.py    -- hard CUDA / GPU precondition check (fail loudly, never
                   silently fall back to CPU).
diagnostic.py   -- one-shot dataset schema discovery tool.
data_loader.py  -- shape-auto-detecting loader for the .pt attack-graph samples
                   (PART 1).
model.py        -- GraphSAGE edge-classification model (PART 1).
train.py        -- training loop with checkpoint / resume / CSV logging
                   (PART 1).
integration.py  -- convert GNN per-edge propensities into the plain
                   adjacency / weight / edge-list structures the classical
                   algorithms consume (PART 3).
main.py         -- end-to-end pipeline (PART 3).
algorithms/     -- the five from-scratch classical graph algorithms (PART 2).
explain/        -- step-by-step execution-trace serialisation (PART 4).
"""
