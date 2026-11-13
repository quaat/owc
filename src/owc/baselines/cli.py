"""Stand-alone CLI for independent gradient and handcrafted baselines."""

import argparse
import json
from pathlib import Path

from owc.baselines.heuristic import heuristic_score
from owc.baselines.supervised import (
    SupervisedSettings,
    evaluate_models,
    load_checkpoint,
    train_supervised,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m owc.baselines")
    sub = parser.add_subparsers(dest="command", required=True)

    handcrafted = sub.add_parser("handcrafted", help="evaluate material+placement")
    handcrafted.add_argument("--fen", required=True)

    train = sub.add_parser("train", help="train same-architecture Adam MLP")
    train.add_argument("--train-labels", type=Path, required=True)
    train.add_argument("--dev-labels", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, default=11)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--max-train-examples", type=int)

    compare = sub.add_parser("compare", help="compare static DEV MAE, not playing strength")
    compare.add_argument("--dev-labels", type=Path, required=True)
    compare.add_argument("--supervised-checkpoint", type=Path)
    compare.add_argument("--direct-session", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "handcrafted":
            result = {"fen": args.fen, "score": heuristic_score(args.fen)}
        elif args.command == "train":
            result = train_supervised(
                train_labels=args.train_labels,
                dev_labels=args.dev_labels,
                output_dir=args.output_dir,
                settings=SupervisedSettings(
                    seed=args.seed, epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    max_train_examples=args.max_train_examples,
                ),
            )
        elif args.command == "compare":
            model = None
            if args.supervised_checkpoint:
                model, _ = load_checkpoint(args.supervised_checkpoint)
            result = evaluate_models(
                args.dev_labels, neural_model=model,
                direct_session=args.direct_session,
            )
        else:
            parser.error("Unknown baseline command")
            return 2
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
        return 2
