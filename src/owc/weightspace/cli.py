"""Auditable neural network weight editing commands."""

import argparse
import json
from pathlib import Path

from owc.weightspace.session import WeightSession


def add_weights_parser(commands: argparse._SubParsersAction) -> None:
    weights = commands.add_parser("weights", help="inspect and edit parameter checkpoints")
    sub = weights.add_subparsers(dest="weights_command", required=True)

    def with_session(name: str, help_text: str) -> argparse.ArgumentParser:
        parser = sub.add_parser(name, help=help_text)
        parser.add_argument("--session", type=Path, required=True)
        return parser

    with_session("init", "create an immutable zero-weight initial revision")

    inspect = with_session("inspect", "list model tensors and bounded samples")
    inspect.add_argument("--tensor", help="optional tensor to preview")
    inspect.add_argument("--max-values", type=int, default=12)

    apply = with_session("apply", "apply a validated JSON patch to a new revision")
    apply.add_argument("--patch", type=Path, required=True)

    probe = with_session("probe", "score a legal chess position")
    probe.add_argument("--fen", required=True)

    checkpoint = with_session("checkpoint", "name the current immutable revision")
    checkpoint.add_argument("--name", required=True)

    rollback = with_session("rollback", "restore a named checkpoint in a new revision")
    rollback.add_argument("--name", required=True)


def run_weights(args: argparse.Namespace) -> None:
    if args.weights_command == "init":
        result = WeightSession.create(args.session).inspect()
    else:
        session = WeightSession(args.session)
        match args.weights_command:
            case "inspect":
                result = session.inspect(tensor=args.tensor, max_values=args.max_values)
            case "apply":
                if args.patch.stat().st_size > 512_000:
                    raise ValueError("Patch exceeds 512 KB JSON size budget")
                patch = json.loads(args.patch.read_text(encoding="utf-8"))
                result = session.apply(patch)
            case "probe":
                result = session.probe(args.fen)
            case "checkpoint":
                result = session.checkpoint(args.name)
            case "rollback":
                result = session.rollback(args.name)
            case _:
                raise ValueError(f"Unsupported weights command: {args.weights_command}")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
