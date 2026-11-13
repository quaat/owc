"""CLI for preparing chess position corpora with sealed held-out output."""

import argparse
import json
from pathlib import Path

from owc.dataset.positions import DatasetSettings, build_dataset


def add_dataset_parser(commands: argparse._SubParsersAction) -> None:
    dataset = commands.add_parser("dataset", help="build chess position corpora")
    sub = dataset.add_subparsers(dest="dataset_command", required=True)
    build = sub.add_parser("build", help="group-split and deduplicate legal game positions")

    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument("--pgn", type=Path, help="user-supplied licensed PGN")
    source.add_argument(
        "--synthetic-games", type=int, help="random legal games (testing only)"
    )
    build.add_argument("--source-license", help="PGN rights/license declaration")
    build.add_argument("--dev-dir", type=Path, required=True)
    build.add_argument("--heldout-dir", type=Path, required=True)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--min-ply", type=int, default=8)
    build.add_argument("--stride", type=int, default=4)
    build.add_argument("--max-plies", type=int, default=160)


def run_dataset(args: argparse.Namespace) -> None:
    if args.dataset_command != "build":
        raise ValueError(f"Unsupported dataset command: {args.dataset_command}")
    settings = DatasetSettings(
        seed=args.seed,
        min_ply=args.min_ply,
        stride=args.stride,
        max_plies=args.max_plies,
    )
    manifest = build_dataset(
        dev_dir=args.dev_dir,
        heldout_dir=args.heldout_dir,
        settings=settings,
        pgn_path=args.pgn,
        synthetic_game_count=args.synthetic_games,
        source_license=args.source_license,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
