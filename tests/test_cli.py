"""CLI smoke cases including terminal positions and invalid input."""

import json

import chess
import pytest

from owc.cli import main


def test_inspect_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["fen"] == chess.STARTING_FEN
    assert data["score"] == 0.0
    assert data["score_perspective"] == "side_to_move"
    assert data["features"] == 781
    assert data["parameters"] == 25_569
    assert data["legal_moves"] == 20
    assert data["outcome"] is None


def test_terminal_checkmate(capsys: pytest.CaptureFixture[str]) -> None:
    fen = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
    assert main(["inspect", "--fen", fen, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["legal_moves"] == 0
    assert data["outcome"] == "1-0"


def test_invalid_fen_exits_with_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["inspect", "--fen", "invalid"])
    assert error.value.code == 2
    assert "Invalid FEN" in capsys.readouterr().err
