"""Stockfish oracle contract tests using a strict fake UCI engine."""

import json
from pathlib import Path

import chess
import chess.engine
import pytest

from owc.cli import main
from owc.oracle.stockfish import OracleSettings, StockfishOracle, label_fen_file


class FakeEngine:
    options = {"Threads": object(), "Hash": object(), "UCI_ShowWDL": object()}
    id = {"name": "Stockfish Test", "author": "OWC"}

    def __init__(self):
        self.config: dict = {}
        self.closed = False

    def configure(self, options: dict) -> None:
        self.config = options

    def analyse(self, board: chess.Board, limit: chess.engine.Limit, **kwargs: object) -> dict:
        assert limit.nodes == 123
        assert kwargs["info"] == chess.engine.INFO_ALL
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(120), chess.WHITE),
            "nodes": 130,
            "depth": 8,
            "pv": [next(iter(board.legal_moves))],
        }

    def quit(self) -> None:
        self.closed = True


@pytest.fixture
def fake_stockfish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, FakeEngine]:
    binary = tmp_path / "stockfish-mock"
    binary.write_text("test engine")
    binary.chmod(0o755)  # shutil.which only resolves executable files
    mock = FakeEngine()
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *a, **kw: mock)
    return binary, mock


def test_target_sign_and_raw_score(
    fake_stockfish: tuple[Path, FakeEngine],
) -> None:
    binary, mock = fake_stockfish
    with StockfishOracle(OracleSettings(str(binary), nodes=123)) as oracle:
        white = oracle.score(chess.STARTING_FEN)
        black = oracle.score("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        assert white["raw_cp_stm"] == 120
        assert white["raw_cp_white"] == 120
        assert white["target_score"] > 0
        assert black["raw_cp_stm"] == -120
        assert black["target_score"] < 0
        assert oracle.metadata["uci_options"] == {
            "Threads": 1, "Hash": 16, "UCI_ShowWDL": True
        }
        assert len(oracle.metadata["binary_sha256"]) == 64
        assert isinstance(white["pv_uci"], list)
    assert mock.closed


def test_invalid_settings_and_terminal_rejected(
    fake_stockfish: tuple[Path, FakeEngine],
) -> None:
    binary, _ = fake_stockfish
    with pytest.raises(ValueError, match="nodes"):
        OracleSettings(nodes=0)
    with pytest.raises(ValueError, match="cp_scale"):
        OracleSettings(cp_scale=float("nan"))
    with StockfishOracle(OracleSettings(str(binary), nodes=123)) as oracle:
        with pytest.raises(ValueError, match="Terminal"):
            oracle.score("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")


def test_missing_wdl_fails_clearly(fake_stockfish: tuple[Path, FakeEngine]) -> None:
    binary, _ = fake_stockfish
    settings = OracleSettings(str(binary), nodes=123, target="wdl_expectation")
    with StockfishOracle(settings) as oracle:
        with pytest.raises(ValueError, match="WDL"):
            oracle.score(chess.STARTING_FEN)


def test_batch_is_immutable_and_cleans_after_error(
    fake_stockfish: tuple[Path, FakeEngine], tmp_path: Path
) -> None:
    binary, _ = fake_stockfish
    source = tmp_path / "positions.fen"
    target = tmp_path / "labels.jsonl"
    source.write_text(chess.STARTING_FEN + "\n# comment\n\n")
    with StockfishOracle(OracleSettings(str(binary), nodes=123)) as oracle:
        manifest = label_fen_file(oracle, source, target)
        assert manifest["records"] == 1
        record = json.loads(target.read_text().strip())
        assert record["source_line"] == 1
        assert target.with_suffix(".jsonl.manifest.json").exists()
        with pytest.raises(FileExistsError):
            label_fen_file(oracle, source, target)

        source.write_text(chess.STARTING_FEN + "\nnot a FEN\n")
        another = tmp_path / "partial.jsonl"
        with pytest.raises(ValueError, match=":2:"):
            label_fen_file(oracle, source, another)
        assert not another.exists()
        assert not another.with_suffix(".jsonl.manifest.json").exists()


def test_oracle_score_cli_json(
    fake_stockfish: tuple[Path, FakeEngine], capsys: pytest.CaptureFixture[str]
) -> None:
    binary, _ = fake_stockfish
    assert main([
        "oracle", "score", "--stockfish", str(binary), "--nodes", "123",
        "--fen", chess.STARTING_FEN, "--json",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["result"]["raw_cp_stm"] == 120
    assert output["oracle"]["engine_id"]["name"] == "Stockfish Test"
