"""Bounded direct-UCI labeler for local integration smoke data.

This avoids asyncio's cross-thread wakeup dependency in restricted sandboxes.
It emits the existing M1 cp_tanh record/manifest format. Run under `timeout`.
"""

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from owc.board.encoding import parse_legal_fen
from owc.oracle.stockfish import LABEL_VERSION, OracleSettings, sha256_file

SCORE = re.compile(r"\bscore (cp|mate) (-?\d+)\b")
NODES = re.compile(r"\bnodes (\d+)\b")
DEPTH = re.compile(r"\bdepth (\d+)\b")
SELDEPTH = re.compile(r"\bseldepth (\d+)\b")


def _until(process: subprocess.Popen, marker: str) -> list[str]:
    lines = []
    while True:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Stockfish ended before {marker}")
        line = line.strip()
        lines.append(line)
        if line.startswith(marker):
            return lines


def _send(process: subprocess.Popen, *commands: str) -> None:
    process.stdin.write("\n".join(commands) + "\n")
    process.stdin.flush()


def _score(fen: str, lines: list[str], cp_scale: float) -> dict:
    board = parse_legal_fen(fen)
    if board.is_game_over(claim_draw=False):
        raise ValueError("Terminal positions are not oracle targets")
    infos = [line for line in lines if line.startswith("info ") and SCORE.search(line)]
    if not infos:
        raise ValueError("Stockfish returned no score")
    info = infos[-1]
    score = SCORE.search(info)
    value = int(score.group(2))
    cp = value if score.group(1) == "cp" else None
    mate = value if score.group(1) == "mate" else None
    target = math.tanh(cp / cp_scale) if cp is not None else (1.0 if mate > 0 else -1.0)

    def number(pattern: re.Pattern) -> int | None:
        match = pattern.search(info)
        return int(match.group(1)) if match else None

    return {
        "label_version": LABEL_VERSION, "fen": board.fen(en_passant="legal"),
        "score_perspective": "side_to_move",
        "side_to_move": "white" if board.turn else "black",
        "raw_cp_stm": cp,
        "raw_cp_white": cp if board.turn else -cp if cp is not None else None,
        "mate_ply_stm": mate, "wdl_stm": None,
        "target_kind": "cp_tanh", "target_score": target,
        "searched_nodes": number(NODES), "depth": number(DEPTH),
        "seldepth": number(SELDEPTH),
        "pv_uci": info.split(" pv ", 1)[1].split() if " pv " in info else [],
    }


def label(source: Path, output: Path, settings: OracleSettings) -> dict:
    source = source.resolve(strict=True)
    output = output.resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite labels or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    binary = Path(settings.stockfish_path).resolve(strict=True)
    process = subprocess.Popen(
        [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    temporary = None
    try:
        _send(process, "uci")
        handshake = _until(process, "uciok")
        engine_id = {}
        for line in handshake:
            if line.startswith("id name "):
                engine_id["name"] = line.removeprefix("id name ")
            elif line.startswith("id author "):
                engine_id["author"] = line.removeprefix("id author ")
        _send(process, "setoption name Threads value 1",
              f"setoption name Hash value {settings.hash_mb}", "isready")
        _until(process, "readyok")
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            count = 0
            for line_number, raw in enumerate(source.read_text().splitlines(), start=1):
                fen = raw.strip()
                if not fen:
                    continue
                _send(process, f"position fen {fen}", f"go nodes {settings.nodes}")
                record = _score(fen, _until(process, "bestmove"), settings.cp_scale)
                record["source_line"] = line_number
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                count += 1
        if count == 0:
            raise ValueError("No positions to label")
        manifest = {
            "schema_version": LABEL_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source": source.name, "source_sha256": sha256_file(source),
            "records": count, "labels_file": output.name,
            "labels_sha256": sha256_file(temporary),
            "oracle": {
                "schema_version": LABEL_VERSION, "binary": str(binary),
                "binary_sha256": sha256_file(binary), "engine_id": engine_id,
                "uci_options": {"Threads": 1, "Hash": settings.hash_mb,
                                "UCI_ShowWDL": False},
                "nodes_requested": settings.nodes, "target": "cp_tanh",
                "cp_scale": settings.cp_scale, "wdl_enabled": False,
                "transport": "direct-uci-smoke-v1",
            },
            "settings": asdict(settings),
        }
        os.replace(temporary, output)
        temporary = None
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return manifest
    finally:
        if process.poll() is None:
            _send(process, "quit")
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stockfish", required=True)
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--hash-mb", type=int, default=16)
    args = parser.parse_args()
    settings = OracleSettings(stockfish_path=args.stockfish, nodes=args.nodes,
                              hash_mb=args.hash_mb)
    print(json.dumps(label(args.input, args.output, settings), sort_keys=True))


if __name__ == "__main__":
    main()
