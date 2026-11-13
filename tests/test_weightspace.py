"""Direct weight programming must be auditable, bounded and reversible."""

import json
import math
from pathlib import Path

import chess
import pytest
import torch

from owc.cli import main
from owc.models.raw_mlp import RawMLP
from owc.weightspace.patches import PATCH_SCHEMA, apply_patch, inspect_weights, model_hash
from owc.weightspace.session import WeightSession


def _patch(model_hash_value: str, ops: list[dict]) -> dict:
    return {
        "schema_version": PATCH_SCHEMA,
        "expected_model_hash": model_hash_value,
        "actor": "unit-test",
        "reason": "Deliberate test of a network circuit",
        "ops": ops,
    }


def test_sparse_patch_is_auditable_and_changes_inference() -> None:
    model = RawMLP()
    before = model_hash(model)
    report = apply_patch(
        model, _patch(before, [{"kind": "set", "tensor": "layers.4.bias",
                                "index": [0], "value": 0.25}])
    )
    assert report["before_hash"] == before
    assert report["after_hash"] == model_hash(model) != before
    assert report["changed_parameters"] == {"layers.4.bias": 1}
    assert report["edited_cells"] == 1
    with torch.inference_mode():
        assert model(torch.zeros(781)).item() == pytest.approx(math.tanh(0.25))


def test_rank_one_changes_only_selected_matrix_cells() -> None:
    model = RawMLP()
    patch = _patch(model_hash(model), [
        {"kind": "rank_one_add", "tensor": "layers.0.weight",
         "rows": [0, 2], "cols": [3, 5],
         "left": [1.0, -2.0], "right": [3.0, 4.0], "scale": 0.5},
    ])
    result = apply_patch(model, patch)
    assert result["edited_cells"] == 4
    weight = dict(model.named_parameters())["layers.0.weight"].detach()
    assert weight[0, 3].item() == 1.5
    assert weight[0, 5].item() == 2.0
    assert weight[2, 3].item() == -3.0
    assert weight[2, 5].item() == -4.0
    assert int(torch.count_nonzero(weight).item()) == 4


@pytest.mark.parametrize("operations", [
    [{"kind": "set", "tensor": "layers.4.bias", "index": [1], "value": 0.1}],
    [{"kind": "set", "tensor": "layers.4.bias", "index": [0], "value": float("nan")}],
    [{"kind": "set", "tensor": "layers.4.bias", "index": [0], "value": True}],
    [{"kind": "set", "tensor": "layers.4.bias", "index": [0], "value": 0}],
    [{"kind": "eval", "tensor": "layers.4.bias", "code": "import os"}],
    [{"kind": "rank_one_add", "tensor": "layers.4.bias",
      "rows": [0], "cols": [0], "left": [1], "right": [1], "scale": 1}],
    [{"kind": "rank_one_add", "tensor": "layers.0.weight",
      "rows": [0, 0], "cols": [0], "left": [1, 1], "right": [1], "scale": 1}],
])
def test_invalid_and_noop_patches_leave_model_untouched(operations: list[dict]) -> None:
    model = RawMLP()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    hash_before = model_hash(model)
    with pytest.raises(ValueError):
        apply_patch(model, _patch(hash_before, operations))
    assert model_hash(model) == hash_before
    assert all(torch.equal(original[name], p) for name, p in model.named_parameters())


def test_two_operations_rollback_atomically_on_error() -> None:
    model = RawMLP()
    initial = model_hash(model)
    patch = _patch(initial, [
        {"kind": "set", "tensor": "layers.4.bias", "index": [0], "value": 0.8},
        {"kind": "add", "tensor": "layers.4.bias", "index": [9], "value": 1},
    ])
    with pytest.raises(ValueError):
        apply_patch(model, patch)
    assert model_hash(model) == initial


def test_wrong_base_hash_and_unexpected_keys_fail_closed() -> None:
    model = RawMLP()
    patch = _patch("0" * 64, [
        {"kind": "set", "tensor": "layers.4.bias", "index": [0], "value": 0.1}
    ])
    with pytest.raises(ValueError, match="precondition"):
        apply_patch(model, patch)
    patch["expected_model_hash"] = model_hash(model)
    patch["python"] = "import os"
    with pytest.raises(ValueError, match="top-level"):
        apply_patch(model, patch)


def test_inspect_reports_bounded_values() -> None:
    model = RawMLP()
    result = inspect_weights(model, tensor="layers.4.bias", max_values=1)
    assert result["model_hash"] == model_hash(model)
    assert result["tensors"]["layers.4.bias"]["first_values"] == [0.0]
    assert result["tensors"]["layers.0.weight"]["shape"] == [32, 781]
    with pytest.raises(ValueError, match="Unknown tensor"):
        inspect_weights(model, tensor="not.real")
    with pytest.raises(ValueError, match="max_values"):
        inspect_weights(model, max_values=129)


def test_named_checkpoint_and_exact_rollback(tmp_path: Path) -> None:
    session = WeightSession.create(tmp_path / "experiment")
    baseline = session.inspect()
    checkpoint = session.checkpoint("baseline")
    assert checkpoint["model_hash"] == baseline["model_hash"]
    assert session.probe(chess.STARTING_FEN)["score"] == 0.0

    applied = session.apply(
        _patch(baseline["model_hash"], [
            {"kind": "set", "tensor": "layers.4.bias", "index": [0], "value": 0.5}
        ])
    )
    assert applied["revision"] != baseline["revision"]
    assert session.probe(chess.STARTING_FEN)["score"] == pytest.approx(math.tanh(0.5))
    with pytest.raises(FileExistsError):
        session.checkpoint("baseline")

    result = session.rollback("baseline")
    assert result["restored_from"] == baseline["revision"]
    assert result["after_hash"] == baseline["model_hash"]
    assert session.inspect()["model_hash"] == baseline["model_hash"]
    assert session.probe(chess.STARTING_FEN)["score"] == 0.0
    assert session.inspect()["revision"] not in (baseline["revision"], applied["revision"])
    assert session.inspect()["action"] == "rollback"


def test_tampering_and_concurrent_write_guard(tmp_path: Path) -> None:
    session = WeightSession.create(tmp_path / "experiment")
    baseline = session.inspect()
    (session.directory / ".write.lock").write_text("other writer")
    with pytest.raises(RuntimeError, match="locked"):
        session.apply(_patch(baseline["model_hash"], [
            {"kind": "set", "tensor": "layers.4.bias", "index": [0], "value": 0.5}
        ]))
    (session.directory / ".write.lock").unlink()
    with pytest.raises(ValueError, match="Checkpoint name"):
        session.checkpoint("../escape")
    state = session.directory / "revisions" / baseline["revision"] / "weights.npz"
    with state.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="byte hash"):
        session.inspect()


def test_cli_complete_edit_checkpoint_probe_rollback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "demo"
    assert main(["weights", "init", "--session", str(directory)]) == 0
    initial = json.loads(capsys.readouterr().out)
    assert initial["tensors"]["layers.4.bias"]["nonzero"] == 0
    assert main(["weights", "checkpoint", "--session", str(directory),
                 "--name", "baseline"]) == 0
    capsys.readouterr()

    patch_file = tmp_path / "edit.json"
    patch_file.write_text(json.dumps(_patch(initial["model_hash"], [
        {"kind": "set", "tensor": "layers.4.bias", "index": [0], "value": 0.25}
    ])))
    assert main(["weights", "apply", "--session", str(directory),
                 "--patch", str(patch_file)]) == 0
    changed = json.loads(capsys.readouterr().out)
    assert changed["after_hash"] != initial["model_hash"]

    assert main(["weights", "probe", "--session", str(directory),
                 "--fen", chess.STARTING_FEN]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["score"] == pytest.approx(math.tanh(0.25))

    assert main(["weights", "rollback", "--session", str(directory),
                 "--name", "baseline"]) == 0
    restored = json.loads(capsys.readouterr().out)
    assert restored["after_hash"] == initial["model_hash"]


def test_cli_rejects_invalid_patch_without_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "experiment"
    session = WeightSession.create(directory)
    initial_revision = session.head()
    bad_patch = tmp_path / "bad.json"
    bad_patch.write_text(json.dumps(_patch(model_hash(RawMLP()), [
        {"kind": "set", "tensor": "layers.4.bias", "index": [0], "value": 0}
    ])))
    with pytest.raises(SystemExit) as error:
        main(["weights", "apply", "--session", str(directory),
              "--patch", str(bad_patch)])
    assert error.value.code == 2
    assert session.head() == initial_revision
    assert "does not change" in capsys.readouterr().err


def test_direct_patch_can_build_a_position_sensitive_circuit() -> None:
    """Six explicit parameter edits detect a white pawn square vs a black one."""
    from owc.board.encoding import encode_board, parse_legal_fen

    model = RawMLP()
    white_pawn_e2 = chess.E2
    black_pawn_e7 = 6 * 64 + chess.E7
    operations = [
        {"kind": "set", "tensor": "layers.0.weight", "index": [0, white_pawn_e2],
         "value": 1.0},
        {"kind": "set", "tensor": "layers.0.weight", "index": [1, black_pawn_e7],
         "value": 1.0},
        {"kind": "set", "tensor": "layers.2.weight", "index": [0, 0], "value": 1.0},
        {"kind": "set", "tensor": "layers.2.weight", "index": [1, 1], "value": 1.0},
        {"kind": "set", "tensor": "layers.4.weight", "index": [0, 0], "value": 1.0},
        {"kind": "set", "tensor": "layers.4.weight", "index": [0, 1], "value": -1.0},
    ]
    report = apply_patch(model, _patch(model_hash(model), operations))
    assert report["edited_cells"] == 6

    starting = parse_legal_fen(chess.STARTING_FEN)
    after_e4 = starting.copy()
    after_e4.push_san("e4")
    with torch.inference_mode():
        start_score = float(model(encode_board(starting)).item())
        after_score = float(model(encode_board(after_e4)).item())
    assert start_score == 0.0
    assert after_score == pytest.approx(-math.tanh(1.0))
