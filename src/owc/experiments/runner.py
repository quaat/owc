"""Development-only agent experiment runner with auditable feedback budgets.

This process never calls Stockfish. It consumes previously generated DEV labels,
not held-out TEST labels, and reveals teacher targets only in charged feedback
rounds. Actual isolation against an untrusted executable requires OS sandboxing.
"""

import hashlib
import json
import math
import os
import platform
import random
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

import torch

from owc.board.encoding import ENCODING_VERSION, encode_board, parse_legal_fen
from owc.dataset.positions import DATASET_VERSION
from owc.experiments.request_wire import (
    MAX_REQUEST_BYTES,
    RequestSizeError,
    bounded_request,
    canonical_bytes,
)
from owc.oracle.stockfish import LABEL_VERSION, sha256_file
from owc.weightspace.patches import PATCH_SCHEMA
from owc.weightspace.session import WeightSession

RUN_SCHEMA = "owc.agent-run.v1"
ALLOWED_FEEDBACK_ROUNDS = frozenset({0, 5, 20, 50})


@dataclass(frozen=True)
class ExperimentSettings:
    feedback_rounds: int = 0
    seed: int = 11
    probe_count: int = 4
    max_label_disclosures: int = 200
    max_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.feedback_rounds not in ALLOWED_FEEDBACK_ROUNDS:
            raise ValueError("feedback_rounds must be 0, 5, 20 or 50")
        if not 1 <= self.probe_count <= 32 or self.max_label_disclosures < 0:
            raise ValueError("probe_count must be in [1,32] and label budget nonnegative")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")


class Agent(Protocol):
    @property
    def identity(self) -> str: ...

    def propose(self, request: dict, timeout_seconds: float) -> dict: ...


class FixtureAgent:
    """Smoke-test adapter, NOT an AI agent or evidence of learning."""

    identity = "fixture:deterministic-output-bias"

    def propose(self, request: dict, timeout_seconds: float) -> dict:
        _ = timeout_seconds
        step = request["round"]
        return {
            "schema_version": PATCH_SCHEMA,
            "expected_model_hash": request["model_hash"],
            "actor": self.identity,
            "reason": "Deterministic protocol demonstration (not chess learning)",
            "ops": [{
                "kind": "set", "tensor": "layers.4.bias",
                "index": [0], "value": (step + 1) * 0.01,
            }],
        }


class CommandAgent:
    """Explicit operator-installed command. It reads one JSON request on stdin.

    Operator assumes responsibility for sandboxing the command and its network/
    filesystem permissions. We do not imply the identity string authenticates
    the underlying model or implementation.
    """

    def __init__(self, command: str, identity: str, env: dict[str, str] | None = None) -> None:
        self.command = shlex.split(command)
        if not self.command:
            raise ValueError("agent command must not be empty")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Actual agent identity must be declared")

        self.identity = identity
        self.env = dict(env) if env is not None else None

    def propose(self, request: dict, timeout_seconds: float) -> dict:
        wire = canonical_bytes(request)
        if len(wire) > MAX_REQUEST_BYTES:
            raise RequestSizeError(f"Agent request exceeds {MAX_REQUEST_BYTES} bytes")
        completed = subprocess.run(
            self.command,
            input=wire.decode("utf-8"),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=self.env,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Agent adapter exited {completed.returncode}; "
                "inspect agent-side logs without exposing credentials"
            )
        if len(completed.stdout.encode()) > 512_000:
            raise ValueError("Agent patch exceeds 512 KiB response limit")
        result = json.loads(completed.stdout)
        if not isinstance(result, dict):
            raise ValueError("Agent must return a JSON object")
        return result


def _manifest(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Invalid JSON manifest at {path}")
    return result


def load_dev_labels(path: Path) -> tuple[list[dict], dict]:
    """Require matching M1 dataset and labeling manifests for the DEV role."""
    source = path.resolve(strict=True)
    if source.name != "dev.labels.jsonl":
        raise ValueError("Only dev.labels.jsonl may be used in agent experiments")
    dev_fens = source.parent / "dev.fen"
    dataset = _manifest(source.parent / "manifest.json")
    label_manifest = _manifest(source.with_suffix(".jsonl.manifest.json"))
    if dataset.get("schema_version") != DATASET_VERSION:
        raise ValueError("Unrecognized M1 dataset manifest")
    if label_manifest.get("schema_version") != LABEL_VERSION:
        raise ValueError("Unrecognized Stockfish label schema")
    if label_manifest.get("source") != "dev.fen":
        raise ValueError("Expected Stockfish labels sourced from dev.fen")
    fen_hash = sha256_file(dev_fens)
    if label_manifest.get("source_sha256") != fen_hash:
        raise ValueError("The Stockfish label source hash has changed")
    if dataset.get("files", {}).get("dev", {}).get("fen_sha256") != fen_hash:
        raise ValueError("Dataset manifest does not match dev.fen")
    if label_manifest.get("labels_sha256") != sha256_file(source):
        raise ValueError("Labeled output checksum mismatch")
    if label_manifest.get("oracle", {}).get("binary_sha256") is None:
        raise ValueError("Stockfish provenance is missing")

    items = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("label_version") != LABEL_VERSION:
                raise ValueError("Unknown record label version")
            if record.get("score_perspective") != "side_to_move":
                raise ValueError("Labels must use side_to_move perspective")
            score = record.get("target_score")
            if isinstance(score, bool) or not isinstance(score, int | float):
                raise ValueError("A teacher target must be numeric")
            if not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError("Teacher target must be finite and bounded [-1,1]")
            fen = parse_legal_fen(record["fen"]).fen(en_passant="legal")
            if fen != record["fen"]:
                raise ValueError("Noncanonical FEN in teacher labels")
            items.append({"fen": fen, "target": float(score)})
    if len(items) != label_manifest.get("records") or not items:
        raise ValueError("Position count mismatch or empty development labels")
    if len({x["fen"] for x in items}) != len(items):
        raise ValueError("Duplicate labeled positions not allowed")
    # Compare actual FEN order, not merely hashes, to block unrelated labels.
    if [x["fen"] for x in items] != dev_fens.read_text(encoding="utf-8").splitlines():
        raise ValueError("Development position order does not match teacher labels")
    return items, {
        "dev_fen_sha256": fen_hash,
        "dev_label_sha256": sha256_file(source),
        "oracle": label_manifest["oracle"],
        "target": label_manifest.get("settings", {}).get("target"),
        "count": len(items),
    }


def _predict(session: WeightSession, positions: list[dict]) -> list[float]:
    model, _ = session.current()
    batch = torch.stack([
        encode_board(parse_legal_fen(row["fen"])) for row in positions
    ])
    with torch.inference_mode():
        return [float(value) for value in model(batch).tolist()]


def _nonzero_state(session: WeightSession, limit: int = 512) -> dict:
    """Give the agent a bounded view of previously programmed parameters."""
    model, _ = session.current()
    values = {}
    total = 0
    for name, tensor in model.named_parameters():
        cells = torch.nonzero(tensor.detach(), as_tuple=False)
        total += len(cells)
        for cell in cells[:max(0, limit - sum(map(len, values.values())))]:
            index = cell.tolist()
            values.setdefault(name, []).append({
                "index": index, "value": float(tensor[tuple(index)].item()),
            })
    return {"total_nonzero": total, "shown": values, "truncated": total > limit}


AGENT_CONTEXT = {
    "encoding_version": ENCODING_VERSION,
    "input_dimensions": 781,
    "feature_index": "64*plane+square; plane 0..5 white P,N,B,R,Q,K; "
                     "6..11 black P,N,B,R,Q,K; a1=0,h8=63; 768 white_to_move; "
                     "769..772 WK,WQ,BK,BQ castling; 773..780 legal EP file a..h",
    "network": "781 -> Linear(32) -> ReLU -> Linear(16) -> ReLU -> Linear(1) -> tanh",
    "tensor_index": "matrix [output_unit,input_unit], bias [output_unit]",
    "perspective": "positive is better for side to move",
    "initialization": "all zero; connect nonzero paths through all three Linear layers",
}


def _atomic_json(path: Path, payload: dict) -> None:
    # Only used inside exclusive, newly created run directory.
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _append_event(path: Path, event: dict) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")


def _environment() -> dict:
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, timeout=5, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True,
        text=True, timeout=5, check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "working_tree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "python": platform.python_version(), "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "dependencies": {name: version(name) for name in (
            "torch", "numpy", "python-chess", "chess",
        )},
    }


def run_experiment(
    *,
    labels: Path,
    output_dir: Path,
    agent: Agent,
    settings: ExperimentSettings,
) -> dict:
    """One zero-shot patch, then N supervised-feedback rounds (N=0/5/20/50).

    A round discloses probe_count *precomputed teacher labels*, charged even
    if the same record was shown in an earlier round. No new Stockfish calls.
    """
    records, provenance = load_dev_labels(labels)
    if settings.feedback_rounds * settings.probe_count > settings.max_label_disclosures:
        raise ValueError("Feedback disclosure budget would be exceeded")
    if not agent.identity or len(agent.identity) > 128:
        raise ValueError("Agent identity must be non-empty and <=128 characters")
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite run: {target}")
    target.mkdir(parents=True)
    log = target / "events.jsonl"
    session = WeightSession.create(target / "weights")
    rng = random.Random(settings.seed)
    disclosures = 0
    history: list[dict] = []
    started = time.monotonic()
    meta = {
        "schema_version": RUN_SCHEMA,
        "settings": asdict(settings),
        "agent_identity_declared": agent.identity,
        "teacher": provenance,
        "environment": _environment(),
        "feedback_definition": "disclosed precomputed dev teacher labels, repetitions count",
        "live_stockfish_queries": 0,
        "agent_request_max_bytes": MAX_REQUEST_BYTES,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(target / "manifest.json", meta)
    try:
        for step in range(settings.feedback_rounds + 1):
            elapsed = time.monotonic() - started
            if elapsed >= settings.max_seconds:
                raise TimeoutError("Total experiment wall-clock budget exceeded")
            state = session.inspect(max_values=0)
            probes = rng.sample(records, min(settings.probe_count, len(records)))
            baseline = _predict(session, probes)
            feedback = []
            if step > 0:
                feedback = [
                    {
                        "fen": p["fen"], "prediction": prediction,
                        "teacher_target": p["target"],
                        "error": prediction - p["target"],
                    }
                    for p, prediction in zip(probes, baseline, strict=True)
                ]
                if disclosures + len(feedback) > settings.max_label_disclosures:
                    raise ValueError("Feedback disclosure budget exceeded")
            request = {
                "protocol": RUN_SCHEMA,
                "round": step,
                "feedback_rounds_total": settings.feedback_rounds,
                "seed": settings.seed,
                "model_hash": state["model_hash"],
                "model_tensors": state["tensors"],
                "nonzero_state": _nonzero_state(session),
                "agent_context": AGENT_CONTEXT,
                "previous_rounds": [dict(item) for item in history[-5:]],
                "previous_rounds_omitted": max(0, len(history) - 5),
                "blind_probes": [
                    {"fen": p["fen"], "prediction": prediction}
                    for p, prediction in zip(probes, baseline, strict=True)
                ],
                "teacher_feedback": feedback,
                "constraints": {
                    "no_gradient_optimizer": True,
                    "patch_schema": PATCH_SCHEMA,
                    "label_disclosures_used": disclosures + len(feedback),
                    "label_disclosures_budget": settings.max_label_disclosures,
                },
            }
            remaining = settings.max_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Total experiment wall-clock budget exceeded")
            patch = None
            wire = None
            try:
                request, wire = bounded_request(request)
                disclosures += len(feedback)
                patch = agent.propose(request, remaining)
                if not isinstance(patch, dict) or patch.get("actor") != agent.identity:
                    raise ValueError("Patch actor must match the declared agent identity")
                outcome = session.apply(patch)
            except Exception as error:
                audit_request = request if not isinstance(error, RequestSizeError) else {
                    "protocol": RUN_SCHEMA, "round": step,
                    "model_hash": state["model_hash"],
                    "request_size_error": str(error),
                }
                _append_event(log, {
                    "round": step, "request": audit_request, "response_patch": patch,
                    "status": "rejected", "error_type": type(error).__name__,
                    "error": str(error)[:500],
                    "request_sha256": hashlib.sha256(wire).hexdigest() if wire else None,
                    "model_hash_after": session.inspect(max_values=0)["model_hash"],
                    "elapsed_seconds": time.monotonic() - started,
                })
                raise
            after_probes = _predict(session, probes)
            dev_mae = sum(abs(p["target"] - value) for p, value in zip(
                records, _predict(session, records), strict=True
            )) / len(records)
            event = {
                "round": step,
                "request": request,
                "request_bytes": len(wire),
                "request_sha256": hashlib.sha256(wire).hexdigest(),
                "response_patch": patch,
                "outcome": outcome,
                "post_patch_probes": [
                    {"fen": p["fen"], "prediction": prediction}
                    for p, prediction in zip(probes, after_probes, strict=True)
                ],
                "dev_mae_undisclosed": dev_mae,
                "elapsed_seconds": time.monotonic() - started,
            }
            _append_event(log, event)
            kinds: dict[str, int] = {}
            for op in patch["ops"]:
                kinds[op["kind"]] = kinds.get(op["kind"], 0) + 1
            history.append({
                "round": step,
                "model_hash_before": outcome["before_hash"],
                "model_hash_after": outcome["after_hash"],
                "patch_sha256": hashlib.sha256(canonical_bytes(patch)).hexdigest(),
                "patch_reason": patch["reason"],
                "operation_kinds": kinds,
                "operations": outcome["operations"],
                "edited_cells": outcome["edited_cells"],
                "changed_parameters": outcome["changed_parameters"],
                "teacher_feedback_disclosed": feedback,
            })
        predictions = _predict(session, records)
        mae = sum(
            abs(p["target"] - prediction)
            for p, prediction in zip(records, predictions, strict=True)
        ) / len(records)
        meta.update({
            "status": "completed",
            "rounds_executed": settings.feedback_rounds + 1,
            "feedback_disclosures": disclosures,
            "dev_mae": mae,
            "final_model_hash": session.inspect(max_values=0)["model_hash"],
            "elapsed_seconds": time.monotonic() - started,
            "events_sha256": sha256_file(log),
        })
    except BaseException as error:
        meta.update({
            "status": "failed", "exception_type": type(error).__name__,
            "error": str(error)[:500], "feedback_disclosures": disclosures,
            "elapsed_seconds": time.monotonic() - started,
        })
        _atomic_json(target / "manifest.json", meta)
        raise
    _atomic_json(target / "manifest.json", meta)
    return meta
