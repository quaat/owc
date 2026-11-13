"""Versioned raw-board feature encoding with strict position validation."""

import chess
import torch
from torch import Tensor

ENCODING_VERSION = "owc.raw781.v1"
PIECE_PLANES = 12
SQUARES_PER_PLANE = 64
INPUT_DIM = PIECE_PLANES * SQUARES_PER_PLANE + 1 + 4 + 8

PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def parse_legal_fen(fen: str) -> chess.Board:
    """Parse FEN, rejecting positions inconsistent with legal chess play."""
    try:
        board = chess.Board(fen)
    except ValueError as error:
        raise ValueError(f"Invalid FEN: {error}") from error

    if not board.is_valid():
        raise ValueError("FEN describes an invalid chess position")
    return board


def encode_board(board: chess.Board) -> Tensor:
    """Encode a legal board as 781 float32 features in absolute coordinates.

    The en-passant feature requires a *legal* capture, not just an EP target
    field supplied by a FEN producer. This avoids encoding impossible captures.
    """
    if not board.is_valid():
        raise ValueError("Cannot encode an invalid chess position")

    features = torch.zeros(INPUT_DIM, dtype=torch.float32)

    for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
        for piece_index, piece_type in enumerate(PIECE_TYPES):
            offset = (color_index * 6 + piece_index) * SQUARES_PER_PLANE
            for square in board.pieces(piece_type, color):
                features[offset + square] = 1.0

    features[768] = float(board.turn == chess.WHITE)
    features[769] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[770] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[771] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[772] = float(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None and board.has_legal_en_passant():
        features[773 + chess.square_file(board.ep_square)] = 1.0

    return features
