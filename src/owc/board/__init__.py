"""Chess positions and input encoding."""

from owc.board.encoding import (
    ENCODING_VERSION,
    INPUT_DIM,
    encode_board,
    parse_legal_fen,
)

__all__ = ["ENCODING_VERSION", "INPUT_DIM", "encode_board", "parse_legal_fen"]
