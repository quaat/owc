"""Check direct UCI score parsing without requiring Stockfish in CI."""

import chess

from owc.oracle.direct_uci import _score


def test_score_parses_side_to_move_and_node_provenance():
    lines = [
        "info depth 3 seldepth 4 score cp 100 nodes 120 pv e2e4 e7e5",
        "bestmove e2e4",
    ]
    result = _score(chess.STARTING_FEN, lines, 400.0)
    assert result["raw_cp_stm"] == result["raw_cp_white"] == 100
    assert result["searched_nodes"] == 120
    assert result["pv_uci"] == ["e2e4", "e7e5"]
    black = chess.Board()
    black.push_san("e4")
    result = _score(black.fen(), lines, 400.0)
    assert result["raw_cp_stm"] == 100
    assert result["raw_cp_white"] == -100
