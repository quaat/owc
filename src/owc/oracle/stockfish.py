"""Deterministic-budget Stockfish position-labeling interface.

"Reproducible" here means *recorded* engine binary, options and requested
node budget; it does not promise identical results across different machines.
"""

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import chess
import chess.engine

from owc.board.encoding import parse_legal_fen

LABEL_VERSION = "owc.stockfish.v1"
TargetKind = Literal["cp_tanh", "wdl_expectation"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class OracleSettings:
    """Versioned analysis configuration; all scores are from side-to-move view."""

    stockfish_path: str = "stockfish"
    nodes: int = 10_000
    hash_mb: int = 16
    target: TargetKind = "cp_tanh"
    cp_scale: float = 400.0

    def __post_init__(self) -> None:
        if self.nodes < 1:
            raise ValueError("nodes must be positive")
        if self.hash_mb < 1:
            raise ValueError("hash_mb must be positive")
        if self.target not in ("cp_tanh", "wdl_expectation"):
            raise ValueError(f"Unsupported target: {self.target}")
        if not math.isfinite(self.cp_scale) or self.cp_scale <= 0:
            raise ValueError("cp_scale must be finite and positive")


class StockfishOracle:
    """Owns one UCI engine process; use as a context manager."""

    def __init__(self, settings: OracleSettings):
        self.settings = settings
        self.engine: chess.engine.SimpleEngine | None = None
        self.metadata: dict[str, object] = {}

    def __enter__(self) -> "StockfishOracle":
        resolved = shutil.which(self.settings.stockfish_path)
        if resolved is None:
            raise FileNotFoundError(
                f"Stockfish not found: {self.settings.stockfish_path!r}; "
                "install Stockfish or set STOCKFISH_PATH"
            )
        binary = Path(resolved).resolve()
        engine = chess.engine.SimpleEngine.popen_uci(str(binary), timeout=30.0)
        try:
            missing = {"Threads", "Hash"} - set(engine.options)
            if missing:
                raise RuntimeError(f"Expected Stockfish UCI options missing: {sorted(missing)}")

            wdl_available = "UCI_ShowWDL" in engine.options
            if self.settings.target == "wdl_expectation" and not wdl_available:
                raise RuntimeError("This Stockfish build does not support UCI_ShowWDL")

            options: dict[str, int | bool] = {"Threads": 1, "Hash": self.settings.hash_mb}
            if wdl_available:
                options["UCI_ShowWDL"] = True
            engine.configure(options)

            self.metadata = {
                "schema_version": LABEL_VERSION,
                "binary": str(binary),
                "binary_sha256": sha256_file(binary),
                "engine_id": dict(engine.id),
                "uci_options": options,
                "nodes_requested": self.settings.nodes,
                "target": self.settings.target,
                "cp_scale": self.settings.cp_scale,
                "wdl_enabled": wdl_available,
            }
        except BaseException:
            engine.quit()
            raise
        self.engine = engine
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            finally:
                self.engine = None

    def score(self, fen: str) -> dict[str, object]:
        """Label one non-terminal legal position, retaining raw evaluation data."""
        if self.engine is None:
            raise RuntimeError("StockfishOracle must be opened with a context manager")
        board = parse_legal_fen(fen)
        if board.is_game_over(claim_draw=False):
            raise ValueError("Terminal positions are resolved by chess rules, not Stockfish")

        info = self.engine.analyse(
            board,
            chess.engine.Limit(nodes=self.settings.nodes),
            info=chess.engine.INFO_ALL,
        )
        score = info.get("score")
        if not isinstance(score, chess.engine.PovScore):
            raise ValueError("Stockfish analysis did not return a position score")

        stm = score.pov(board.turn)
        white = score.pov(chess.WHITE)
        cp_stm = stm.score()
        mate_stm = stm.mate()

        wdl = info.get("wdl")
        wdl_stm: dict[str, int] | None = None
        expectation: float | None = None
        if isinstance(wdl, chess.engine.PovWdl):
            counts = wdl.pov(board.turn)
            wdl_stm = {
                "wins": counts.wins,
                "draws": counts.draws,
                "losses": counts.losses,
            }
            expectation = counts.expectation()

        if self.settings.target == "wdl_expectation":
            if expectation is None:
                raise ValueError("Stockfish did not supply WDL for this analysis")
            target_score = 2.0 * expectation - 1.0
        elif mate_stm is not None:
            target_score = 1.0 if mate_stm > 0 else -1.0
        elif cp_stm is not None:
            target_score = math.tanh(cp_stm / self.settings.cp_scale)
        else:
            raise ValueError("Stockfish did not supply a usable score")

        return {
            "label_version": LABEL_VERSION,
            "fen": board.fen(en_passant="legal"),
            "score_perspective": "side_to_move",
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "raw_cp_stm": cp_stm,
            "raw_cp_white": white.score(),
            "mate_ply_stm": mate_stm,
            "wdl_stm": wdl_stm,
            "target_kind": self.settings.target,
            "target_score": target_score,
            "searched_nodes": info.get("nodes"),
            "depth": info.get("depth"),
            "seldepth": info.get("seldepth"),
            "pv_uci": [move.uci() for move in info.get("pv", [])],
        }


def label_fen_file(oracle: StockfishOracle, input_path: Path, output_path: Path) -> dict:
    """Batch-score FEN lines with atomic outputs and engine/data provenance.

    An existing output or manifest is never overwritten. Invalid input does not
    publish partial labels. Full-file restart means rerunning after removing
    unpublished temporary files; resuming partial labeling is M1 dataset work.
    """
    if oracle.engine is None:
        raise RuntimeError("Oracle must be opened")
    source = input_path.resolve(strict=True)
    dest = output_path.resolve()
    manifest_path = dest.with_suffix(dest.suffix + ".manifest.json")
    if source in (dest, manifest_path):
        raise ValueError("Input and output paths must differ")
    if dest.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {dest} or {manifest_path}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    temp_labels: Path | None = None
    temp_manifest: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=dest.parent, delete=False
        ) as target:
            temp_labels = Path(target.name)
            count = 0
            with source.open(encoding="utf-8") as fh:
                for line_number, line in enumerate(fh, start=1):
                    fen = line.strip()
                    if not fen or fen.startswith("#"):
                        continue
                    try:
                        record = oracle.score(fen)
                    except (ValueError, chess.engine.EngineError) as error:
                        raise ValueError(f"{source}:{line_number}: {error}") from error
                    record["source_line"] = line_number
                    target.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    count += 1

        if count == 0:
            raise ValueError("Input file contains no positions")

        manifest = {
            "schema_version": LABEL_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source.name,
            "source_sha256": sha256_file(source),
            "records": count,
            "labels_file": dest.name,
            "labels_sha256": sha256_file(temp_labels),
            "oracle": oracle.metadata,
            "settings": asdict(oracle.settings),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=dest.parent, delete=False
        ) as mf:
            temp_manifest = Path(mf.name)
            json.dump(manifest, mf, indent=2, sort_keys=True, allow_nan=False)
            mf.write("\n")

        # No concurrent writer is supported; reserve destination names in a
        # dedicated run directory if multiple processes run simultaneously.
        os.replace(temp_labels, dest)
        temp_labels = None
        os.replace(temp_manifest, manifest_path)
        temp_manifest = None
        return manifest
    finally:
        for temp in (temp_labels, temp_manifest):
            if temp is not None:
                temp.unlink(missing_ok=True)
