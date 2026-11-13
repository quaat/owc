"""Command-line entry point for the initial evaluation laboratory."""

import argparse
import json
from collections.abc import Sequence

import chess
import torch

from owc.board.encoding import ENCODING_VERSION, encode_board, parse_legal_fen
from owc.dataset.cli import add_dataset_parser, run_dataset
from owc.models.raw_mlp import RawMLP
from owc.oracle.cli import add_oracle_parser, run_oracle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="owc", description="Open Weight Chess laboratory")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="inspect a legal chess position")
    inspect.add_argument(
        "--fen",
        default=chess.STARTING_FEN,
        help="full legal six-field FEN (defaults to standard starting position)",
    )
    inspect.add_argument("--json", action="store_true", help="machine-readable output")
    add_oracle_parser(commands)
    add_dataset_parser(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "dataset":
        try:
            run_dataset(args)
        except (ValueError, FileNotFoundError, FileExistsError) as error:
            parser.error(str(error))
        return 0

    if args.command == "oracle":
        try:
            run_oracle(args)
        except (ValueError, FileNotFoundError, FileExistsError) as error:
            parser.error(str(error))
        return 0

    if args.command == "inspect":
        try:
            board = parse_legal_fen(args.fen)
        except ValueError as error:
            parser.error(str(error))

        features = encode_board(board)
        model = RawMLP().eval()
        with torch.inference_mode():
            score = float(model(features).item())

        outcome = board.outcome(claim_draw=True)
        result = {
            "fen": board.fen(en_passant="legal"),
            "encoding_version": ENCODING_VERSION,
            "features": int(features.numel()),
            "active_features": int(features.sum().item()),
            "parameters": model.parameter_count,
            "score": score,
            "score_perspective": "side_to_move",
            "side_to_move": "white" if board.turn == chess.WHITE else "black",
            "legal_moves": board.legal_moves.count(),
            "outcome": outcome.result() if outcome else None,
        }
        if args.json:
            print(json.dumps(result, sort_keys=True))
        else:
            print(board)
            for name, value in result.items():
                print(f"{name}: {value}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
