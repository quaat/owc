"""CLI parser and runner for Stockfish oracle commands."""

import argparse
import json
import os
from pathlib import Path

from owc.oracle.stockfish import OracleSettings, StockfishOracle, label_fen_file


def add_oracle_parser(commands: argparse._SubParsersAction) -> None:
    oracle = commands.add_parser("oracle", help="score positions using Stockfish")
    sub = oracle.add_subparsers(dest="oracle_command", required=True)

    def add_settings(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--stockfish",
            default=os.environ.get("STOCKFISH_PATH", "stockfish"),
            help="Stockfish binary path or name (default: STOCKFISH_PATH or PATH)",
        )
        parser.add_argument("--nodes", type=int, default=10_000)
        parser.add_argument("--hash-mb", type=int, default=16)
        parser.add_argument(
            "--target", choices=("cp_tanh", "wdl_expectation"), default="cp_tanh"
        )
        parser.add_argument("--cp-scale", type=float, default=400.0)

    score = sub.add_parser("score", help="score one legal, non-terminal position")
    score.add_argument("--fen", required=True)
    score.add_argument("--json", action="store_true")
    add_settings(score)

    batch = sub.add_parser("batch", help="score one FEN per line into immutable JSONL")
    batch.add_argument("--input", type=Path, required=True, help="FEN text file")
    batch.add_argument("--output", type=Path, required=True, help="new JSONL path")
    add_settings(batch)


def run_oracle(args: argparse.Namespace) -> None:
    settings = OracleSettings(
        stockfish_path=args.stockfish,
        nodes=args.nodes,
        hash_mb=args.hash_mb,
        target=args.target,
        cp_scale=args.cp_scale,
    )
    with StockfishOracle(settings) as oracle:
        if args.oracle_command == "score":
            record = oracle.score(args.fen)
            if args.json:
                print(json.dumps({"oracle": oracle.metadata, "result": record}, sort_keys=True))
            else:
                print(
                    f"{record['fen']}\n"
                    f"Stockfish {oracle.metadata['engine_id']} at {args.nodes} nodes\n"
                    f"{record['target_kind']}: {record['target_score']:+.5f} "
                    "(side-to-move perspective)"
                )
        elif args.oracle_command == "batch":
            manifest = label_fen_file(oracle, args.input, args.output)
            print(json.dumps(manifest, sort_keys=True))
        else:
            raise ValueError(f"Unsupported oracle command: {args.oracle_command}")
