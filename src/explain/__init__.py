"""
explain -- PART 4 explainability artifacts.

Each classical algorithm in :mod:`src.algorithms` optionally fills a ``trace``
list with one dict per step (Warshall's matrix after each k, Dijkstra's frontier
expansion, Kruskal's accept/reject decisions, ...). This sub-package serialises
those traces to:

    * a single JSON file per run  -- full fidelity, intended input for a future
      step-by-step animation;
    * one flat CSV per algorithm  -- human-scannable "here is what it did at
      every step", for showing the professor the algorithm is doing the right
      thing rather than just emitting a final answer.

See :func:`src.explain.logger.save_explainability_artifacts`.
"""

from .logger import ExplainRun, save_explainability_artifacts

__all__ = ["ExplainRun", "save_explainability_artifacts"]
