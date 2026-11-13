"""Independent heuristic and gradient-trained chess evaluation baselines."""

from owc.baselines.heuristic import heuristic_score
from owc.baselines.supervised import SupervisedSettings, train_supervised

__all__ = ["heuristic_score", "SupervisedSettings", "train_supervised"]
