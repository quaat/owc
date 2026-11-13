"""A disclosed, non-neural material-plus-simple-placement baseline."""

import math

import chess

from owc.board.encoding import parse_legal_fen

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

HEURISTIC_VERSION = "owc.material-placement.v1"


def heuristic_cp(board: chess.Board) -> int:
    """Centipawn score from side-to-move perspective, no tree search."""
    if not board.is_valid():
        raise ValueError("Cannot evaluate invalid position")
    white_minus_black = 0
    for square, piece in board.piece_map().items():
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        relative_rank = rank if piece.color == chess.WHITE else 7 - rank
        center = 7 - abs(file * 2 - 7) - abs(rank * 2 - 7)
        center = max(0, center)
        bonus = {
            chess.PAWN: relative_rank * 6,
            chess.KNIGHT: center * 7,
            chess.BISHOP: center * 4,
            chess.ROOK: relative_rank * 2,
            chess.QUEEN: center * 2,
            chess.KING: 0,
        }[piece.piece_type]
        value = PIECE_VALUES[piece.piece_type] + bonus
        white_minus_black += value if piece.color == chess.WHITE else -value
    return white_minus_black if board.turn == chess.WHITE else -white_minus_black


def heuristic_score(fen: str, cp_scale: float = 400.0) -> float:
    if not math.isfinite(cp_scale) or cp_scale <= 0:
        raise ValueError("cp_scale must be positive and finite")
    return math.tanh(heuristic_cp(parse_legal_fen(fen)) / cp_scale)
