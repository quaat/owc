"""Development-only experimental protocol invariants."""

import json
import math
import sys
from pathlib import Path

import chess
import pytest

from owc.experiments.runner import (
    CommandAgent,
    ExperimentSettings,
    FixtureAgent,
    load_dev_labels,
    run_experiment,
)
from owc.oracle.stockfish import LABEL_VERSION, sha256_file


def dev_fixture(tmp_path: Path) -> Path:
    folder = tmp_path / "dev"
    folder.mkdir()
    fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    ]
    (folder / "dev.fen").write_text("\n".join(fens) + "\n")
    sha = sha256_file(folder / "dev.fen")
    (folder / "manifest.json").write_text(json.dumps({
        "schema_version": "owc.positions.v1",
        "files": {"dev": {"fen_sha256": sha}},
    }))
    labels = folder / "dev.labels.jsonl"
    labels.write_text("".join(json.dumps({
        "label_version": LABEL_VERSION,
        "fen": fen,
        "score_perspective": "side_to_move",
        "target_score": (index * 2 - 1) / 4,
        "target_kind": "cp_tanh",
    }) + "\n" for index, fen in enumerate(fens)))
    (folder / "dev.labels.jsonl.manifest.json").write_text(json.dumps({
        "schema_version": LABEL_VERSION, "source": "dev.fen",
        "source_sha256": sha, "labels_sha256": sha256_file(labels),
        "oracle": {"binary_sha256": "ab" * 32, "engine_id": {"name": "fixture"}},
        "settings": {"target": "cp_tanh"}, "records": len(fens),
    }))
    return labels


class CaptureAgent(FixtureAgent):
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def propose(self, request: dict, timeout_seconds: float) -> dict:
        self.requests.append(json.loads(json.dumps(request)))
        return super().propose(request, timeout_seconds)


def test_e0_one_patch_and_no_teacher_disclosures(tmp_path: Path) -> None:
    labels = dev_fixture(tmp_path)
    agent = CaptureAgent()
    run = run_experiment(
        labels=labels, output_dir=tmp_path / "run",
        agent=agent, settings=ExperimentSettings(seed=9),
    )
    assert run["status"] == "completed"
    assert run["rounds_executed"] == 1
    assert run["feedback_disclosures"] == run["live_stockfish_queries"] == 0
    assert math.isfinite(run["dev_mae"])
    assert agent.requests[0]["teacher_feedback"] == []
    assert "teacher_target" not in json.dumps(agent.requests[0])
    assert run["events_sha256"] == sha256_file(tmp_path / "run" / "events.jsonl")


def test_e5_budget_and_feedback_visible_after_initial_patch(tmp_path: Path) -> None:
    labels = dev_fixture(tmp_path)
    agent = CaptureAgent()
    run = run_experiment(
        labels=labels, output_dir=tmp_path / "run", agent=agent,
        settings=ExperimentSettings(feedback_rounds=5, probe_count=2,
                                    max_label_disclosures=10),
    )
    assert run["rounds_executed"] == 6
    assert run["feedback_disclosures"] == 10
    assert len(agent.requests) == 6
    assert agent.requests[0]["teacher_feedback"] == []
    assert all(len(req["teacher_feedback"]) == 2 for req in agent.requests[1:])


def test_invalid_budget_refuses_to_create_run(tmp_path: Path) -> None:
    labels = dev_fixture(tmp_path)
    with pytest.raises(ValueError, match="budget"):
        run_experiment(
            labels=labels, output_dir=tmp_path / "run",
            agent=FixtureAgent(),
            settings=ExperimentSettings(feedback_rounds=5, probe_count=2,
                                        max_label_disclosures=9),
        )
    assert not (tmp_path / "run").exists()


def test_invalid_source_role_and_modified_manifest(tmp_path: Path) -> None:
    labels = dev_fixture(tmp_path)
    other = labels.with_name("test.labels.jsonl")
    other.write_text(labels.read_text())
    with pytest.raises(ValueError, match="Only dev.labels"):
        load_dev_labels(other)
    manifest = labels.with_suffix(".jsonl.manifest.json")
    meta = json.loads(manifest.read_text())
    meta["source"] = "test.fen"
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="dev.fen"):
        load_dev_labels(labels)


def test_label_tampering_refused(tmp_path: Path) -> None:
    labels = dev_fixture(tmp_path)
    labels.write_text(labels.read_text() + "tampering\n")
    with pytest.raises(ValueError, match="checksum"):
        load_dev_labels(labels)


def test_three_seed_runs_are_independent(tmp_path: Path) -> None:
    labels = dev_fixture(tmp_path)
    runs = [
        run_experiment(
            labels=labels, output_dir=tmp_path / f"seed-{seed}",
            agent=FixtureAgent(), settings=ExperimentSettings(seed=seed),
        )
        for seed in (11, 12, 13)
    ]
    assert [run["settings"]["seed"] for run in runs] == [11, 12, 13]
    assert all(run["status"] == "completed" for run in runs)


def test_external_command_adapter_contract(tmp_path: Path) -> None:
    labels = dev_fixture(tmp_path)
    agent = CommandAgent(
        f"{sys.executable} -m owc.experiments.fixture_command",
        "fixture:external-json",
    )
    run = run_experiment(
        labels=labels, output_dir=tmp_path / "cmd",
        agent=agent, settings=ExperimentSettings(),
    )
    assert run["status"] == "completed"
    assert run["agent_identity_declared"] == "fixture:external-json"


def test_wrong_actor_fails_closed_with_manifest(tmp_path: Path) -> None:
    labels = dev_fixture(tmp_path)

    class BadAgent(FixtureAgent):
        def propose(self, request: dict, timeout_seconds: float) -> dict:
            patch = super().propose(request, timeout_seconds)
            patch["actor"] = "impostor"
            return patch

    with pytest.raises(ValueError, match="actor"):
        run_experiment(
            labels=labels, output_dir=tmp_path / "bad",
            agent=BadAgent(), settings=ExperimentSettings(),
        )
    assert json.loads((tmp_path / "bad" / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("rounds", [-1, 1, 4, 6])
def test_only_declared_feedback_conditions(rounds: int) -> None:
    with pytest.raises(ValueError, match="feedback_rounds"):
        ExperimentSettings(feedback_rounds=rounds)
