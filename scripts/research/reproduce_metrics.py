"""Recompute the November 2026 smoke-pilot archive from untouched local artifacts.

Writes compact public summaries only. Never reads generated test positions or
copies FENs, labels, checkpoint bytes, or provider diagnostics into reports.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from owc.baselines.supervised import evaluate_models, load_checkpoint, read_split
from owc.experiments.astra_adapter import SYSTEM_INSTRUCTIONS
from owc.experiments.runner import load_dev_labels
from owc.weightspace.session import WeightSession

EXPERIMENT_ID = "owc-m3-20261113-smoke-pilot"
E0_ID = EXPERIMENT_ID + "-e0-attempt1"
ADAM_ID = EXPERIMENT_ID + "-adam-seed11"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_index(root: Path, corpus: Path, e0_root: Path, adam_root: Path) -> dict:
    items = []
    for directory in (corpus, e0_root, adam_root):
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            sensitive = "diagnostics" in relative.parts
            items.append({
                "artifact_id": str(relative).replace("/", ":"),
                "purpose": "original provider adapter diagnostic" if sensitive else
                           "original local data or run artifact",
                "storage_location": str(relative), "format": path.suffix or "text",
                "size_bytes": path.stat().st_size, "sha256": _sha(path),
                "access": "local only; internal endpoint; do not publish raw" if sensitive else
                          "local ignored artifact; not committed",
                "retention": "workspace only; no durable external store configured",
                "experiment_ids": [E0_ID] if e0_root in path.parents else
                                  [ADAM_ID] if adam_root in path.parents else [EXPERIMENT_ID],
                "original": True,
            })
    dataset_manifest = _json(corpus / "manifest.json")
    for filename, key in (("test.fen", "fen_sha256"),
                          ("test.meta.jsonl", "metadata_sha256")):
        items.append({
            "artifact_id": "generated-test:" + filename,
            "purpose": "generated split; never labeled or evaluated in this session",
            "storage_location": "local /tmp/owc-astra-smoke-heldout-30/" + filename,
            "format": ".fen" if filename.endswith(".fen") else ".jsonl",
            "size_bytes": None,
            "sha256": dataset_manifest["files"]["test"][key],
            "sha256_status": "declared by dataset manifest; not independently rehashed",
            "access": "same OS owner; NOT a sealed held-out benchmark",
            "retention": "temporary local storage; no durable guarantee",
            "experiment_ids": [EXPERIMENT_ID], "original": True,
        })
    items.append({
        "artifact_id": "generated-test:manifest.json",
        "purpose": "generated test split manifest; not read in this archive",
        "storage_location": "local /tmp/owc-astra-smoke-heldout-30/manifest.json",
        "format": ".json", "size_bytes": None, "sha256": None,
        "sha256_status": "not inspected; missing independently verified checksum",
        "access": "same OS owner; NOT a sealed held-out benchmark",
        "retention": "temporary local storage; no durable guarantee",
        "experiment_ids": [EXPERIMENT_ID], "original": True,
    })
    for path, purpose, access in (
        (Path("/tmp/owc_astra_connection.py"),
         "original nonce connection script; embeds internal endpoint",
         "local only; do not publish raw"),
        (Path("/tmp/stockfish_17-1_amd64.deb"),
         "downloaded Debian Stockfish 17-1 package",
         "local system package; do not commit binary"),
        (Path("/tmp/owc-stockfish/usr/games/stockfish"),
         "extracted Stockfish 17.1 executable used for teacher labels",
         "local binary; do not commit"),
    ):
        items.append({
            "artifact_id": "external-local:" + path.name,
            "purpose": purpose, "storage_location": str(path),
            "format": path.suffix or "executable",
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": _sha(path) if path.exists() else None,
            "access": access,
            "retention": "temporary local storage; no durable guarantee",
            "experiment_ids": [EXPERIMENT_ID], "original": True,
        })
    return {"schema_version": "owc.research-artifact-index.v1",
            "experiment_id": EXPERIMENT_ID, "artifacts": items,
            "durable_raw_archive_available": False}


def build(root: Path) -> dict[str, str]:
    corpus = root / "datasets/astra-smoke-30"
    e0_root = root / "runs/astra-smoke-e0-attempt1"
    run = e0_root / "E0-seed-11"
    adam_root = root / "runs/astra-smoke-supervised-seed11"
    labels = corpus / "dev.labels.jsonl"
    _, train_source = read_split(corpus / "train.labels.jsonl", "train")
    dev, dev_source = read_split(labels, "dev")
    load_dev_labels(labels)
    model, adam = load_checkpoint(adam_root)
    comparison = evaluate_models(labels, neural_model=model)
    zero_mae = sum(abs(row["target"]) for row in dev) / len(dev)
    train_raw = [
        json.loads(line) for line in (corpus / "train.labels.jsonl").read_text().splitlines()
    ]
    dev_raw = [json.loads(line) for line in labels.read_text().splitlines()]
    e0 = _json(run / "manifest.json")
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    if len(events) != 1 or events[0]["status"] != "rejected":
        raise ValueError("Expected one rejected E0 invocation")
    event = events[0]
    session = WeightSession(run / "weights")
    current = session.inspect(max_values=0)
    diagnostic_files = list((e0_root / "diagnostics").glob("*.json"))
    if len(diagnostic_files) != 1:
        raise ValueError("Expected exactly one original adapter diagnostic")
    diagnostic = _json(diagnostic_files[0])
    if (current["model_hash"] != event["request"]["model_hash"]
            or event["model_hash_after"] != current["model_hash"]):
        raise ValueError("Failed E0 changed its checkpoint")
    if event["response_patch"] is not None or event["request"]["teacher_feedback"]:
        raise ValueError("E0 patch/feedback evidence differs from claimed failure")

    common = {
        "git_commit_at_run": e0["environment"]["commit"],
        "working_tree_dirty_at_run": e0["environment"]["working_tree_dirty"],
        "architecture": "781->32 ReLU->16 ReLU->1 tanh",
        "parameters": 25569, "encoding_version": "owc.raw781.v1",
        "perspective": "side_to_move",
        "dataset_source": "30 synthetic random legal games; integration smoke only",
        "dataset_license": "generated-in-repository",
        "dataset_manifest_sha256": _sha(corpus / "manifest.json"),
        "dev_source": dev_source, "train_source": train_source,
        "stockfish_node_budget_per_position": 1000,
        "stockfish_actual_nodes_train": sum(x["searched_nodes"] for x in train_raw),
        "stockfish_actual_nodes_dev": sum(x["searched_nodes"] for x in dev_raw),
        "evaluation_role": "development_only",
        "heldout_isolated": False,
    }
    e0_record = {
        "schema_version": "owc.research-run.v1", "experiment_id": E0_ID,
        "condition": "E0", "seed": 11, "status": "failed_before_model_inference",
        "started_at_utc": e0["started_at_utc"],
        "adapter_version": diagnostic["adapter_version"],
        "adapter_source_sha256_at_execution": None,
        "adapter_source_hash_missing_reason": (
            "Source changed after run; execution hash not recorded"
        ),
        "current_prompt_sha256": hashlib.sha256(SYSTEM_INSTRUCTIONS.encode()).hexdigest(),
        "prompt_hash_at_execution": None,
        "provider_requested": "Microsoft Foundry",
        "deployment_requested": diagnostic["deployment_requested"],
        "provider_model_response": None, "provider_response_id": None,
        "provider_call_attempts": 1, "completed_model_calls": 0,
        "model_invocation": diagnostic["parameters"],
        "input_tokens": None, "output_tokens": None, "cost_usd": None,
        "usage_missing_reason": "DNS failure before a provider response",
        "initial_model_hash": current["model_hash"],
        "final_model_hash": current["model_hash"],
        "initial_revision": current["revision"], "final_revision": current["revision"],
        "accepted_patches": 0, "rejected_patches": 0,
        "failed_invocations": 1, "changed_parameters": 0,
        "feedback_disclosures": e0["feedback_disclosures"],
        "dev_mae_initial_derived": zero_mae,
        "dev_mae_final_derived": zero_mae,
        "dev_mae_source": "recomputed from original dev labels and unchanged zero model",
        "elapsed_seconds": e0["elapsed_seconds"],
        "adapter_elapsed_seconds": diagnostic["elapsed_seconds"],
        "error_type": diagnostic["error_type"],
        "error_category": "sandbox_dns_resolution_failure",
        "error_detail_public": "Temporary failure in name resolution",
        "original_run_manifest_sha256": _sha(run / "manifest.json"),
        "original_events_sha256": _sha(run / "events.jsonl"),
        "original_diagnostic_sha256": _sha(diagnostic_files[0]),
        **common,
    }
    adam_record = {
        "schema_version": "owc.research-run.v1", "experiment_id": ADAM_ID,
        "condition": "Adam_supervised", "seed": adam["settings"]["seed"],
        "status": "completed", "model_provider": None,
        "initial_model_hash": None,
        "initial_hash_missing_reason": "Initialization hash not recorded by original trainer",
        "final_model_hash": adam["model_hash"],
        "checkpoint_sha256": adam["weights_sha256"],
        "training_positions_distinct": adam["training_positions_exposed"],
        "training_label_presentations": adam["training_label_presentations"],
        "optimizer_updates": adam["optimizer_updates"],
        "epochs": adam["settings"]["epochs"],
        "batch_size": adam["settings"]["batch_size"],
        "learning_rate": adam["settings"]["learning_rate"],
        "development_positions_evaluated_per_epoch": len(dev),
        "dev_mae_initial": None,
        "initial_mae_missing_reason": "Original trainer did not measure initial random model",
        "dev_mae_final": comparison["supervised_mae"],
        "elapsed_seconds": adam["elapsed_seconds"],
        "epoch_trajectory": adam["epochs"],
        "original_manifest_sha256": _sha(adam_root / "manifest.json"),
        **common,
    }
    result = {
        "schema_version": "owc.research-results.v1", "experiment_id": EXPERIMENT_ID,
        "scope": "synthetic development integration smoke only",
        "position_count": len(dev),
        "zero_model_dev_mae": zero_mae,
        "handcrafted_dev_mae": comparison["handcrafted_mae"],
        "supervised_dev_mae": comparison["supervised_mae"],
        "astra_e0_dev_mae": None,
        "astra_e0_missing_reason": "No model-generated patch; DNS failed before provider response",
        "astra_e5_dev_mae": None,
        "astra_e5_missing_reason": "Not run",
        "independent_test_mae": None,
        "independent_test_missing_reason": "No OS isolation or blind evaluator",
        "model_connection_observation": {
            "reported_model": "gpt-6-astra",
            "response_id": "resp_02fd1c4045f5ea08006af77b5754688195aaad69b1d470884b",
            "http_status": 200, "input_tokens": 18, "output_tokens": 11,
            "total_tokens": 29, "latency_seconds": 1.4584902620408684,
            "original_raw_response_archived": False,
            "source": "prior session tool output and contemporaneous note",
        },
    }
    trajectory = [
        {"experiment_id": E0_ID, "seed": 11, "condition": "E0", "iteration": 0,
         "model_hash": current["model_hash"], "feedback_disclosures": 0,
         "elapsed_seconds": None, "metric_name": "dev_mae",
         "metric_value": zero_mae, "model_calls_completed": 0,
         "input_tokens": None, "output_tokens": None,
         "provenance": "derived_zero_model_before_attempt"},
        {"experiment_id": E0_ID, "seed": 11, "condition": "E0", "iteration": 1,
         "model_hash": current["model_hash"], "feedback_disclosures": 0,
         "elapsed_seconds": event["elapsed_seconds"], "metric_name": "dev_mae",
         "metric_value": zero_mae, "model_calls_completed": 0,
         "input_tokens": None, "output_tokens": None,
         "provenance": "derived_unchanged_model_after_failed_invocation"},
    ]
    for epoch in adam["epochs"]:
        trajectory.append({
            "experiment_id": ADAM_ID, "seed": adam["settings"]["seed"],
            "condition": "Adam_supervised", "iteration": epoch["epoch"],
            "model_hash": (
                adam["model_hash"] if epoch["epoch"] == adam["settings"]["epochs"] else None
            ),
            "feedback_disclosures": 0, "elapsed_seconds": None,
            "metric_name": "dev_mae", "metric_value": epoch["dev_mae"],
            "model_calls_completed": 0, "input_tokens": None,
            "output_tokens": None, "provenance": "original_trainer_epoch_manifest",
        })
    return {
        "e0-attempt1.manifest.json": json.dumps(e0_record, indent=2, sort_keys=True) + "\n",
        "adam-seed11.manifest.json": json.dumps(adam_record, indent=2, sort_keys=True) + "\n",
        "results.json": json.dumps(result, indent=2, sort_keys=True) + "\n",
        "artifact-index.json": json.dumps(
            _artifact_index(root, corpus, e0_root, adam_root), indent=2, sort_keys=True
        ) + "\n",
        "trajectory.csv": _csv(trajectory),
        "metrics.csv": _csv([
            {"condition": "zero", "role": "development", "positions": len(dev),
             "metric": "mae", "value": zero_mae},
            {"condition": "handcrafted", "role": "development", "positions": len(dev),
             "metric": "mae", "value": comparison["handcrafted_mae"]},
            {"condition": "Adam_supervised", "role": "development", "positions": len(dev),
             "metric": "mae", "value": comparison["supervised_mae"]},
            {"condition": "Astra_E0", "role": "development", "positions": len(dev),
             "metric": "mae", "value": None},
        ]),
    }


def _csv(rows: list[dict]) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path,
                        default=Path("reports/experiments/2026-11-13-smoke-pilot"))
    parser.add_argument("--check", action="store_true",
                        help="compare generated summaries with committed files")
    args = parser.parse_args()
    products = build(args.root.resolve())
    target = args.output_dir.resolve()
    if args.check:
        for name, content in products.items():
            if (target / name).read_bytes().decode("utf-8") != content:
                raise ValueError(f"Derived report differs from original artifacts: {name}")
        print(f"Verified {len(products)} derived report files")
    else:
        target.mkdir(parents=True, exist_ok=True)
        for name, content in products.items():
            path = target / name
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
        print(f"Created {len(products)} derived report files")


if __name__ == "__main__":
    main()
