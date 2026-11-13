"""Game-group partitioning and leakage-resistant chess position extraction.

A held-out dataset is written outside the working repository. File-path checks
reduce accidental exposure, but do not replace OS-level access isolation.
"""

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.pgn

from owc.oracle.stockfish import sha256_file

DATASET_VERSION = "owc.positions.v1"
SPLITS = ("train", "dev", "test")


@dataclass(frozen=True)
class DatasetSettings:
    seed: int = 42
    min_ply: int = 8
    stride: int = 4
    max_plies: int = 160
    train_fraction: float = 0.8
    dev_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.min_ply < 1 or self.stride < 1 or self.max_plies < self.min_ply:
            raise ValueError("Require 1 <= min_ply <= max_plies and stride >= 1")
        if not 0 < self.train_fraction < 1 or not 0 < self.dev_fraction < 1:
            raise ValueError("Train and dev fractions must lie strictly between 0 and 1")
        if self.train_fraction + self.dev_fraction >= 1:
            raise ValueError("Leave a positive fraction for held-out test games")


@dataclass(frozen=True)
class Position:
    fen: str
    group_id: str
    ply: int


@dataclass(frozen=True)
class GamePositions:
    group_id: str
    positions: tuple[Position, ...]


def assign_split(group_id: str, settings: DatasetSettings) -> str:
    """Stable per-game assignment; never make random splits by FEN or ply."""
    digest = hashlib.sha256(f"{settings.seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / (1 << 64)
    if value < settings.train_fraction:
        return "train"
    if value < settings.train_fraction + settings.dev_fraction:
        return "dev"
    return "test"


def signature(fen: str) -> str:
    """Ignore halfmove/fullmove counters, keep side/castling/legal EP."""
    return " ".join(fen.split()[:4])


def _positions_for_moves(
    group_id: str, moves: Iterator[chess.Move], settings: DatasetSettings
) -> GamePositions:
    board = chess.Board()
    collected: list[Position] = []
    for ply, move in enumerate(moves, start=1):
        if ply > settings.max_plies:
            break
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move in group {group_id} at ply {ply}: {move}")
        board.push(move)
        if board.is_game_over(claim_draw=False):
            break
        if ply >= settings.min_ply and ply % settings.stride == 0:
            collected.append(Position(board.fen(en_passant="legal"), group_id, ply))
    return GamePositions(group_id, tuple(collected))


def pgn_games(path: Path, settings: DatasetSettings, sha: str) -> Iterator[GamePositions]:
    """Consume user-supplied PGN games and abort if a game has parse errors."""
    with path.open(encoding="utf-8-sig") as stream:
        game_index = 0
        while (game := chess.pgn.read_game(stream)) is not None:
            game_index += 1
            if game.errors:
                raise ValueError(f"PGN game {game_index} contains parsing errors: {game.errors}")
            if game.board().fen() != chess.STARTING_FEN:
                raise ValueError("Non-standard PGN starts are not yet supported")
            group_id = f"pgn:{sha}:{game_index:06d}"
            yield _positions_for_moves(group_id, iter(game.mainline_moves()), settings)


def synthetic_games(
    count: int, settings: DatasetSettings
) -> Iterator[GamePositions]:
    """Generate deterministic random legal games as a *smoke dataset only*."""
    if count < 1:
        raise ValueError("synthetic_games must be positive")
    for index in range(1, count + 1):
        group_id = f"synthetic:{settings.seed}:{index:06d}"
        local_seed = int.from_bytes(hashlib.sha256(group_id.encode()).digest()[:8], "big")
        rng = random.Random(local_seed)
        board = chess.Board()
        moves: list[chess.Move] = []
        for _ in range(settings.max_plies):
            if board.is_game_over(claim_draw=False):
                break
            move = rng.choice(tuple(board.legal_moves))
            moves.append(move)
            board.push(move)
        yield _positions_for_moves(group_id, iter(moves), settings)


def _workspace_root() -> Path:
    cwd = Path.cwd().resolve()
    return next((root for root in (cwd, *cwd.parents) if (root / ".git").exists()), cwd)


def _validate_outputs(dev_dir: Path, heldout_dir: Path) -> tuple[Path, Path]:
    dev = dev_dir.resolve()
    test = heldout_dir.resolve()
    if dev == test or dev in test.parents or test in dev.parents:
        raise ValueError("Development and held-out output directories must be separate")
    if test.is_relative_to(_workspace_root()):
        raise ValueError("Held-out data must be outside the repository/working directory")
    if dev.exists() or test.exists():
        raise FileExistsError("Output directories must not already exist")
    return dev, test


def _write_records(
    groups: Iterator[GamePositions],
    dev_temp: Path,
    test_temp: Path,
    settings: DatasetSettings,
) -> tuple[dict[str, int], dict[str, int], int, int]:
    """Stream records without retaining FEN corpora in memory."""
    counts = {split: 0 for split in SPLITS}
    group_counts = {split: 0 for split in SPLITS}
    discarded = 0
    cross_split = 0
    seen: dict[str, str] = {}

    with ExitStack() as stack:
        fens = {}
        metadata = {}
        for split in SPLITS:
            directory = test_temp if split == "test" else dev_temp
            fens[split] = stack.enter_context(
                (directory / f"{split}.fen").open("w", encoding="utf-8")
            )
            metadata[split] = stack.enter_context(
                (directory / f"{split}.meta.jsonl").open("w", encoding="utf-8")
            )

        for game in groups:
            split = assign_split(game.group_id, settings)
            group_counts[split] += 1
            for position in game.positions:
                key = signature(position.fen)
                if key in seen:
                    discarded += 1
                    cross_split += int(seen[key] != split)
                    continue
                seen[key] = split
                counts[split] += 1
                fens[split].write(position.fen + "\n")
                metadata[split].write(
                    json.dumps(
                        asdict(position),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    return counts, group_counts, discarded, cross_split


def build_dataset(
    *,
    dev_dir: Path,
    heldout_dir: Path,
    settings: DatasetSettings,
    pgn_path: Path | None = None,
    synthetic_game_count: int | None = None,
    source_license: str | None = None,
) -> dict:
    """Create immutable development/held-out outputs with game-group splits."""
    if (pgn_path is None) == (synthetic_game_count is None):
        raise ValueError("Exactly one of pgn_path or synthetic_game_count is required")
    dev, test = _validate_outputs(dev_dir, heldout_dir)

    if pgn_path is not None:
        if not source_license or not source_license.strip():
            raise ValueError("PGN input requires a declared --source-license or usage rights")
        source = pgn_path.resolve(strict=True)
        source_sha = sha256_file(source)
        source_id = {"type": "pgn", "name": source.name, "sha256": source_sha}
        groups = pgn_games(source, settings, source_sha)
    else:
        if synthetic_game_count is None or synthetic_game_count < 1:
            raise ValueError("synthetic_game_count must be positive")
        source_id = {"type": "synthetic-random-legal", "games": synthetic_game_count}
        groups = synthetic_games(synthetic_game_count, settings)
        source_license = "generated-in-repository"

    dev.parent.mkdir(parents=True, exist_ok=True)
    test.parent.mkdir(parents=True, exist_ok=True)
    dev_tmp = Path(tempfile.mkdtemp(prefix=".owc-dev-", dir=dev.parent))
    test_tmp = Path(tempfile.mkdtemp(prefix=".owc-test-", dir=test.parent))
    published: list[Path] = []

    try:
        # Split decisions only depend on the group ID and the fixed seed.
        counts, group_counts, duplicates, crossing = _write_records(
            groups, dev_tmp, test_tmp, settings
        )
        if not sum(counts.values()):
            raise ValueError("No usable positions produced; lower --min-ply or provide more games")

        manifest = {
            "schema_version": DATASET_VERSION,
            "source": source_id,
            "source_license_declaration": source_license,
            "settings": asdict(settings),
            "positions": counts,
            "groups": group_counts,
            "deduplicated_positions": duplicates,
            "cross_split_duplicates_removed": crossing,
            "split_strategy": "SHA256(seed:source_game_id) then exact canonical-FEN dedup",
            "note": (
                "Game-level and exact-state isolation only; not a guarantee "
                "against all near-neighbor or opening-family leakage."
            ),
            "files": {
                split: {
                    "fen_sha256": sha256_file(
                        (test_tmp if split == "test" else dev_tmp) / f"{split}.fen"
                    ),
                    "metadata_sha256": sha256_file(
                        (test_tmp if split == "test" else dev_tmp) / f"{split}.meta.jsonl"
                    ),
                }
                for split in SPLITS
            },
        }

        # The development manifest has no held-out position content.
        with (dev_tmp / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        with (test_tmp / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")

        if dev.exists() or test.exists():
            raise FileExistsError("Output directory appeared during dataset construction")
        os.rename(test_tmp, test)
        published.append(test)
        os.rename(dev_tmp, dev)
        published.append(dev)
        return manifest
    except BaseException:
        for destination in published:
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(dev_tmp, ignore_errors=True)
        shutil.rmtree(test_tmp, ignore_errors=True)
