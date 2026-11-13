"""CLI interface to explicit M3 weight-edit agent experiments."""

import argparse
import json
from pathlib import Path

from owc.experiments.runner import (
    CommandAgent,
    ExperimentSettings,
    FixtureAgent,
    run_experiment,
)


def add_experiment_parser(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser("experiment", help="run bounded agent weight-edit trials")
    subs = parser.add_subparsers(dest="experiment_command", required=True)
    run = subs.add_parser("run", help="run E0/E5/E20/E50 direct-edit trials")
    run.add_argument("--dev-labels", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--feedback-rounds", type=int, choices=(0, 5, 20, 50), default=0)
    run.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    run.add_argument("--probe-count", type=int, default=4)
    run.add_argument("--max-label-disclosures", type=int, default=200)
    run.add_argument("--max-seconds", type=float, default=120)
    run.add_argument("--agent-command", help="explicit operator-trusted JSON/stdin adapter")
    run.add_argument("--agent-id", help="declared actual agent/model identity")


def run_experiment_cli(args: argparse.Namespace) -> None:
    if args.experiment_command != "run":
        raise ValueError("Unsupported experiment command")
    if args.agent_command:
        if not args.agent_id:
            raise ValueError("--agent-id is required when --agent-command is used")
        agent = CommandAgent(args.agent_command, args.agent_id)
    else:
        if args.agent_id:
            raise ValueError("--agent-id requires --agent-command")
        agent = FixtureAgent()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seed values must be unique")
    root = args.output_dir.resolve()
    if root.exists():
        raise FileExistsError("Experiment output directory must be new")
    root.mkdir(parents=True)
    reports = []
    for seed in args.seeds:
        settings = ExperimentSettings(
            feedback_rounds=args.feedback_rounds,
            seed=seed,
            probe_count=args.probe_count,
            max_label_disclosures=args.max_label_disclosures,
            max_seconds=args.max_seconds,
        )
        results = run_experiment(
            labels=args.dev_labels,
            output_dir=root / f"E{args.feedback_rounds}-seed-{seed}",
            agent=agent,
            settings=settings,
        )
        reports.append({
            "run": f"E{args.feedback_rounds}-seed-{seed}",
            "status": results["status"],
            "dev_mae": results["dev_mae"],
            "feedback_disclosures": results["feedback_disclosures"],
            "agent_identity": agent.identity,
        })
    print(json.dumps({"runs": reports}, sort_keys=True, allow_nan=False))
