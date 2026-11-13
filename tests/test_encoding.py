"""The encoder's feature contract must never drift unnoticed."""

import chess
import pytest
import torch

from owc.board.encoding import ENCODING_VERSION, INPUT_DIM, encode_board, parse_legal_fen


def test_starting_position() -> None:
    board = parse_legal_fen(chess.STARTING_FEN)
    features = encode_board(board)
    assert ENCODING_VERSION == "owc.raw781.v1"
    assert features.shape == (INPUT_DIM,) == (781,)
    assert features.dtype == torch.float32
    assert int(features[:768].sum().item()) == 32
    assert features[0 * 64 + chess.E2].item() == 1.0  # white pawn
    assert features[5 * 64 + chess.E1].item() == 1.0  # white king
    assert features[11 * 64 + chess.E8].item() == 1.0  # black king
    assert features[768:773].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert features[773:].sum().item() == 0.0


def test_black_to_move_and_no_castling_rights() -> None:
    board = parse_legal_fen("8/8/8/8/8/8/8/K6k b - - 0 1")
    features = encode_board(board)
    assert features[768:773].sum().item() == 0
    assert int(features[:768].sum().item()) == 2


def test_legal_en_passant_encoded_by_file() -> None:
    board = parse_legal_fen("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1")
    assert board.has_legal_en_passant()
    features = encode_board(board)
    assert features[773 + chess.D1 % 8].item() == 1.0
    assert features[773:781].sum().item() == 1.0


def test_non_capturable_en_passant_is_not_encoded() -> None:
    board = parse_legal_fen("8/8/8/3p4/8/8/4P3/4K2k w - d6 0 1")
    assert not board.has_legal_en_passant()
    assert encode_board(board)[773:781].sum().item() == 0


@pytest.mark.parametrize("fen", [
    "not a FEN",
    "8/8/8/8/8/8/8/K7 w - - 0 1",  # missing black king
    "8/8/8/8/8/8/8/K6k w K - 0 1",  # impossible castling right
])
def test_invalid_fen_rejected(fen: str) -> None:
    with pytest.raises(ValueError):
        parse_legal_fen(fen)


def test_pinned_king_en_passant_stays_disabled() -> None:
    board = parse_legal_fen("4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    # En passant e5xd6 opens the e-file to the white king, thus illegal.
    assert board.ep_square == chess.D6
    assert not board.has_legal_en_passant()
    assert encode_board(board)[773:781].sum().item() == 0
