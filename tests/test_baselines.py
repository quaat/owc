"""Supervised and handcrafted baselines are independent of agent weight edits."""

import json
import math
from pathlib import Path

import chess
import pytest
import torch

from owc.baselines.heuristic import heuristic_cp, heuristic_score
from owc.baselines.supervised import (
    SupervisedSettings,
    _make_model,
    evaluate_models,
    load_checkpoint,
    read_split,
    train_supervised,
)
from owc.dataset.positions import DatasetSettings, build_dataset
from owc.oracle.stockfish import LABEL_VERSION, sha256_file
from owc.weightspace.patches import model_hash
from owc.weightspace.session import WeightSession


def dataset(tmp_path: Path) -> tuple[Path, Path]:
    dev = tmp_path / "dev"
    build_dataset(
        dev_dir=dev,
        heldout_dir=tmp_path / "sealed",
        settings=DatasetSettings(seed=42, min_ply=4, stride=2, max_plies=8),
        synthetic_game_count=40,
    )
    paths = []
    for role in ("train", "dev"):
        fens = (dev / f"{role}.fen").read_text().splitlines()
        assert fens
        labels = dev / f"{role}.labels.jsonl"
        with labels.open("w") as output:
            for fen in fens:
                # Test labels are synthetic; NOT Stockfish chess-strength evidence.
                output.write(json.dumps({
                    "label_version": LABEL_VERSION,
                    "fen": fen, "score_perspective": "side_to_move",
                    "target_kind": "cp_tanh", "target_score": heuristic_score(fen),
                }) + "\n")
        labels.with_suffix(".jsonl.manifest.json").write_text(json.dumps({
            "schema_version": LABEL_VERSION,
            "source": f"{role}.fen",
            "source_sha256": sha256_file(dev / f"{role}.fen"),
            "labels_sha256": sha256_file(labels),
            "oracle": {"binary_sha256": "fixture" * 10},
            "settings": {"target": "cp_tanh"}, "records": len(fens),
        }))
        paths.append(labels)
    return paths[0], paths[1]


def test_heuristic_symmetry_and_material_perspective() -> None:
    board = chess.Board()
    assert heuristic_cp(board) == 0
    assert heuristic_score(chess.STARTING_FEN) == 0
    board.remove_piece_at(chess.D8)
    assert board.is_valid()
    # White moves: missing black queen is positive.
    board.turn = chess.WHITE
    assert heuristic_cp(board) > 800
    assert heuristic_score(board.fen()) > 0.8
    board.turn = chess.BLACK
    assert heuristic_cp(board) < -800
    assert heuristic_score(board.fen()) < -0.8


def test_grads_require_nonzero_random_initialization() -> None:
    model = _make_model(17)
    assert model.parameter_count == 25_569
    assert any(torch.count_nonzero(param).item() for param in model.parameters())
    assert model_hash(model) == model_hash(_make_model(17))
    assert model_hash(model) != model_hash(_make_model(18))


def test_train_checkpoint_and_compare(tmp_path: Path) -> None:
    train, dev = dataset(tmp_path)
    out = tmp_path / "trained"
    manifest = train_supervised(
        train_labels=train, dev_labels=dev, output_dir=out,
        settings=SupervisedSettings(seed=11, epochs=2, batch_size=8),
    )
    assert manifest["optimization"].startswith("Adam/")
    assert manifest["parameters"] == 25_569
    assert manifest["training_positions_exposed"] > 0
    assert len(manifest["epochs"]) == 2
    assert math.isfinite(manifest["epochs"][-1]["dev_mae"])
    model, checked = load_checkpoint(out)
    assert checked["model_hash"] == model_hash(model)
    direct = WeightSession.create(tmp_path / "direct")
    report = evaluate_models(dev, neural_model=model, direct_session=direct.directory)
    assert report["positions"] > 0
    assert "supervised_mae" in report and "direct_edit_mae" in report
    assert report["direct_edit_mae"] >= 0
    assert report["handcrafted_mae"] == pytest.approx(0.0)
    with pytest.raises(FileExistsError):
        train_supervised(
            train_labels=train, dev_labels=dev, output_dir=out,
            settings=SupervisedSettings(epochs=1),
        )


def test_training_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    train, dev = dataset(tmp_path)
    settings = SupervisedSettings(epochs=1, batch_size=8, seed=13)
    a = train_supervised(
        train_labels=train, dev_labels=dev, output_dir=tmp_path / "a",
        settings=settings,
    )
    b = train_supervised(
        train_labels=train, dev_labels=dev, output_dir=tmp_path / "b",
        settings=settings,
    )
    assert a["model_hash"] == b["model_hash"]
    assert a["epochs"] == b["epochs"]


def test_train_limit_and_validation(tmp_path: Path) -> None:
    train, dev = dataset(tmp_path)
    result = train_supervised(
        train_labels=train, dev_labels=dev, output_dir=tmp_path / "small",
        settings=SupervisedSettings(epochs=1, max_train_examples=4),
    )
    assert result["training_positions_exposed"] == 4
    with pytest.raises(ValueError, match="Only train and dev"):
        read_split(dev.with_name("test.labels.jsonl"), "test")
    with pytest.raises(ValueError, match="learning_rate"):
        SupervisedSettings(learning_rate=float("nan"))


def test_tampered_checkpoint_and_mislabeled_source(tmp_path: Path) -> None:
    train, dev = dataset(tmp_path)
    out = tmp_path / "trained"
    train_supervised(
        train_labels=train, dev_labels=dev, output_dir=out,
        settings=SupervisedSettings(epochs=1),
    )
    with (out / "weights.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Corrupt"):
        load_checkpoint(out)
    manifest_file = dev.with_suffix(".jsonl.manifest.json")
    manifest = json.loads(manifest_file.read_text())
    manifest["source"] = "test.fen"
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source role"):
        read_split(dev, "dev")
