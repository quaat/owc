"""Model shape, initialization and deterministic inference."""

import chess
import pytest
import torch

from owc.board.encoding import INPUT_DIM, encode_board, parse_legal_fen
from owc.models.raw_mlp import EXPECTED_PARAMETERS, RawMLP


def test_parameter_count_and_zero_weights() -> None:
    model = RawMLP()
    assert model.parameter_count == EXPECTED_PARAMETERS == 25_569
    assert all(torch.count_nonzero(p).item() == 0 for p in model.parameters())


@pytest.mark.parametrize("fen", [
    chess.STARTING_FEN,
    "8/8/8/8/8/8/8/K6k b - - 0 1",
    "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
])
def test_zero_network_scores_every_legal_position_zero(fen: str) -> None:
    model = RawMLP()
    features = encode_board(parse_legal_fen(fen))
    with torch.inference_mode():
        assert model(features).item() == 0.0
        assert model(features.unsqueeze(0)).shape == (1,)


def test_model_rejects_wrong_feature_shape() -> None:
    with pytest.raises(ValueError, match="Expected last dimension"):
        RawMLP()(torch.zeros(INPUT_DIM - 1))
