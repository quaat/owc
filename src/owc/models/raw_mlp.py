"""Minimal zero-initialized network for weight-space programming experiments."""

import torch
from torch import Tensor, nn

from owc.board.encoding import INPUT_DIM

HIDDEN_1 = 32
HIDDEN_2 = 16
EXPECTED_PARAMETERS = 25_569


class RawMLP(nn.Module):
    """Static evaluator: positive output favors the side to move.

    Initial scores are exactly zero. Future milestones will program the
    parameters with agent-authored patches or train a separate gradient baseline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, 1),
            nn.Tanh(),
        )
        self.reset_to_zero()

    def reset_to_zero(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.zero_()

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != INPUT_DIM:
            raise ValueError(f"Expected last dimension {INPUT_DIM}, got {features.shape[-1]}")
        return self.layers(features).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
