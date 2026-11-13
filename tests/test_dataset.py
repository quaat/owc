"""Dataset grouping, deterministic generation, split integrity and isolation."""

import json
from pathlib import Path

import pytest

from owc.cli import main
from owc.dataset.positions import DatasetSettings, assign_split, build_dataset, signature


def _records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_group_assignment_is_stable() -> None:
    settings = DatasetSettings(seed=23)
    assert assign_split("source:game:1", settings) == assign_split("source:game:1", settings)
    assert assign_split("source:game:1", settings) in {"train", "dev", "test"}


def test_synthetic_split_and_dedup_are_reproducible(tmp_path: Path) -> None:
    settings = DatasetSettings(seed=7, min_ply=4, stride=2, max_plies=16)
    manifests = []
    for repeat in (1, 2):
        dev = tmp_path / f"dev-{repeat}"
        hidden = tmp_path / f"test-{repeat}"
        m = build_dataset(
            dev_dir=dev,
            heldout_dir=hidden,
            settings=settings,
            synthetic_game_count=100,
        )
        manifests.append(m)
        assert sum(m["positions"].values()) > 0
        assert m["groups"]["train"] + m["groups"]["dev"] + m["groups"]["test"] == 100

        group_sets = {}
        position_signatures: list[str] = []
        for split in ("train", "dev", "test"):
            root = hidden if split == "test" else dev
            records = _records(root / f"{split}.meta.jsonl")
            fens = (root / f"{split}.fen").read_text().splitlines()
            assert len(records) == len(fens) == m["positions"][split]
            assert [row["fen"] for row in records] == fens
            group_sets[split] = {row["group_id"] for row in records}
            position_signatures.extend(signature(fen) for fen in fens)
        assert not group_sets["train"] & group_sets["dev"]
        assert not group_sets["train"] & group_sets["test"]
        assert not group_sets["dev"] & group_sets["test"]
        assert len(position_signatures) == len(set(position_signatures))
        assert not (dev / "test.fen").exists()
        assert not (dev / "test.meta.jsonl").exists()

    assert manifests[0] == manifests[1]


def test_refuses_overwrite_and_invalid_placement(tmp_path: Path) -> None:
    settings = DatasetSettings(min_ply=4, max_plies=8)
    with pytest.raises(ValueError, match="separate"):
        build_dataset(
            dev_dir=tmp_path / "same",
            heldout_dir=tmp_path / "same",
            settings=settings,
            synthetic_game_count=3,
        )
    dev = tmp_path / "dev"
    hidden = tmp_path / "hidden"
    build_dataset(
        dev_dir=dev, heldout_dir=hidden, settings=settings, synthetic_game_count=5
    )
    with pytest.raises(FileExistsError):
        build_dataset(
            dev_dir=dev, heldout_dir=hidden, settings=settings, synthetic_game_count=5
        )


def test_pgn_requires_source_license_and_loads_games(tmp_path: Path) -> None:
    path = tmp_path / "examples.pgn"
    path.write_text(
        '[Event "Example"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 *\n\n'
        '[Event "Example 2"]\n\n1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 *\n'
    )
    settings = DatasetSettings(min_ply=2, stride=2, max_plies=8)
    with pytest.raises(ValueError, match="license"):
        build_dataset(
            dev_dir=tmp_path / "dev", heldout_dir=tmp_path / "test",
            settings=settings, pgn_path=path,
        )
    manifest = build_dataset(
        dev_dir=tmp_path / "dev", heldout_dir=tmp_path / "test",
        settings=settings, pgn_path=path, source_license="user-authorized",
    )
    assert sum(manifest["groups"].values()) == 2
    assert sum(manifest["positions"].values()) > 0
    assert manifest["source"]["name"] == path.name


def test_dataset_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dev, heldout = tmp_path / "dev", tmp_path / "heldout"
    assert main([
        "dataset", "build", "--synthetic-games", "5", "--min-ply", "4",
        "--stride", "2", "--max-plies", "8",
        "--dev-dir", str(dev), "--heldout-dir", str(heldout),
    ]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert sum(manifest["groups"].values()) == 5
    assert (dev / "manifest.json").exists()
    assert (heldout / "manifest.json").exists()
