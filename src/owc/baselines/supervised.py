"""Same-architecture supervised training, isolated from direct-weight edits.

This module uses Adam on Stockfish-labeled TRAIN positions. There is *no*
gradient API in the direct-weight programming module.
"""

import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from owc.baselines.heuristic import HEURISTIC_VERSION, heuristic_score
from owc.board.encoding import ENCODING_VERSION, encode_board, parse_legal_fen
from owc.dataset.positions import DATASET_VERSION
from owc.models.raw_mlp import RawMLP
from owc.oracle.stockfish import LABEL_VERSION, sha256_file
from owc.weightspace.patches import model_hash
from owc.weightspace.session import WeightSession

BASELINE_VERSION = "owc.supervised-mlp.v1"


@dataclass(frozen=True)
class SupervisedSettings:
    seed: int = 11
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    max_train_examples: int | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.max_train_examples is not None and self.max_train_examples < 1:
            raise ValueError("max_train_examples must be positive")


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def read_split(path: Path, role: str) -> tuple[list[dict], dict]:
    """Read a manifest-bound train/dev set; intentionally refuse test role."""
    if role not in ("train", "dev"):
        raise ValueError("Only train and dev data are available to baselines")
    labels = path.resolve(strict=True)
    if labels.name != f"{role}.labels.jsonl":
        raise ValueError(f"Expected {role}.labels.jsonl")
    source = labels.parent / f"{role}.fen"
    dataset = _json(labels.parent / "manifest.json")
    manifest = _json(labels.with_suffix(".jsonl.manifest.json"))
    if dataset.get("schema_version") != DATASET_VERSION:
        raise ValueError("Unrecognized dataset schema")
    if manifest.get("schema_version") != LABEL_VERSION:
        raise ValueError("Unrecognized Stockfish label schema")
    if manifest.get("source") != source.name:
        raise ValueError("Incorrect source role for labeled positions")
    if not manifest.get("oracle", {}).get("binary_sha256"):
        raise ValueError("Missing Stockfish provenance")
    source_sha = sha256_file(source)
    if (dataset.get("files", {}).get(role, {}).get("fen_sha256") != source_sha or
            manifest.get("source_sha256") != source_sha):
        raise ValueError("FEN/label source checksum mismatch")
    if manifest.get("labels_sha256") != sha256_file(labels):
        raise ValueError("Label checksum mismatch")
    data = []
    with labels.open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("label_version") != LABEL_VERSION:
                raise ValueError("Unrecognized record version")
            if entry.get("score_perspective") != "side_to_move":
                raise ValueError("Expected side-to-move teacher perspective")
            y = entry.get("target_score")
            if isinstance(y, bool) or not isinstance(y, int | float):
                raise ValueError("Target must be numeric")
            if not math.isfinite(y) or not -1 <= y <= 1:
                raise ValueError("Target must be finite and bounded")
            fen = parse_legal_fen(entry["fen"]).fen(en_passant="legal")
            if fen != entry["fen"]:
                raise ValueError("Noncanonical FEN")
            data.append({
                "fen": fen, "target": float(y),
                "target_kind": entry.get("target_kind"),
            })
    if not data or len(data) != manifest.get("records"):
        raise ValueError("Empty or incorrect label count")
    if len({record["fen"] for record in data}) != len(data):
        raise ValueError("Duplicate positions found")
    if [x["fen"] for x in data] != source.read_text(encoding="utf-8").splitlines():
        raise ValueError("Label/FEN order mismatch")
    return data, {
        "role": role,
        "label_sha256": sha256_file(labels),
        "source_sha256": source_sha,
        "records": len(data),
        "target": manifest.get("settings", {}).get("target"),
        "oracle": manifest.get("oracle"),
    }


def _make_model(seed: int) -> RawMLP:
    """Break all-zero ReLU symmetry before using gradient descent."""
    torch.manual_seed(seed)
    model = RawMLP()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                fan_in = module.weight.shape[1]
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
    return model


def _tensor_data(items: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.stack([
        encode_board(parse_legal_fen(item["fen"])) for item in items
    ])
    target = torch.tensor([item["target"] for item in items], dtype=torch.float32)
    return features, target


def _mae(model: RawMLP, samples: tuple[torch.Tensor, torch.Tensor]) -> float:
    model.eval()
    with torch.inference_mode():
        x, y = samples
        return float(torch.abs(model(x) - y).mean().item())


def evaluate_models(
    labels: Path, *,
    neural_model: RawMLP | None = None,
    direct_session: Path | None = None,
) -> dict:
    """Development-only side-by-side MAE on one versioned teacher dataset."""
    items, provenance = read_split(labels, "dev")
    def mean(values: list[float]) -> float:
        return sum(values) / len(values)
    metrics = {
        "schema_version": BASELINE_VERSION,
        "dataset": provenance,
        "positions": len(items),
        "heuristic_version": HEURISTIC_VERSION,
        "handcrafted_mae": mean([
            abs(heuristic_score(item["fen"]) - item["target"])
            for item in items
        ]),
    }
    samples = _tensor_data(items)
    if neural_model is not None:
        metrics["supervised_mae"] = _mae(neural_model, samples)
        metrics["supervised_model_hash"] = model_hash(neural_model)
    if direct_session is not None:
        model, meta = WeightSession(direct_session).current()
        metrics["direct_edit_mae"] = _mae(model, samples)
        metrics["direct_model_hash"] = meta["model_hash"]
    return metrics


def train_supervised(
    *,
    train_labels: Path,
    dev_labels: Path,
    output_dir: Path,
    settings: SupervisedSettings,
) -> dict:
    """Train the same RawMLP architecture and publish a checked NumPy checkpoint."""
    train, source_train = read_split(train_labels, "train")
    dev, source_dev = read_split(dev_labels, "dev")
    if train_labels.resolve().parent != dev_labels.resolve().parent:
        raise ValueError("Training and development labels must share one M1 corpus")
    if {record["fen"] for record in train} & {record["fen"] for record in dev}:
        raise ValueError("Train/development positions overlap")
    kind = {record["target_kind"] for record in train + dev}
    if len(kind) != 1 or None in kind:
        raise ValueError("Training/development teacher target types must agree")
    if source_train["target"] not in kind or source_dev["target"] not in kind:
        raise ValueError("Teacher target manifest mismatch")

    # Precomputed teacher label budget: count unique sampled training records.
    rng = random.Random(settings.seed)
    if settings.max_train_examples is not None and len(train) > settings.max_train_examples:
        train = rng.sample(train, settings.max_train_examples)

    if output_dir.exists():
        raise FileExistsError("Refusing to replace an existing training checkpoint")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = _make_model(settings.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    x, y = _tensor_data(train)
    dev_samples = _tensor_data(dev)
    trajectory = []
    optimizer_updates = 0
    for epoch in range(settings.epochs):
        model.train()
        generator = torch.Generator().manual_seed(settings.seed + epoch)
        permutation = torch.randperm(len(train), generator=generator)
        for batch in permutation.split(settings.batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[batch])
            loss = torch.nn.functional.mse_loss(prediction, y[batch])
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
        trajectory.append({
            "epoch": epoch + 1,
            "train_mae": _mae(model, (x, y)),
            "dev_mae": _mae(model, dev_samples),
        })

    temporary = Path(tempfile.mkdtemp(prefix=".owc-trained-", dir=output_dir.parent))
    try:
        weights = temporary / "weights.npz"
        np.savez_compressed(weights, **{
            name: param.detach().cpu().numpy().copy()
            for name, param in model.named_parameters()
        })
        manifest = {
            "schema_version": BASELINE_VERSION,
            "encoding_version": ENCODING_VERSION,
            "parameters": model.parameter_count,
            "model_hash": model_hash(model),
            "weights_sha256": sha256_file(weights),
            "settings": asdict(settings),
            "training_positions_exposed": len(train),
            "training_label_presentations": len(train) * settings.epochs,
            "optimizer_updates": optimizer_updates,
            "development_positions_evaluated_per_epoch": len(dev),
            "epochs": trajectory,
            "train": source_train,
            "dev": source_dev,
            "elapsed_seconds": time.monotonic() - started,
            "optimization": "Adam/gradient-based (NOT direct weight programming)",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        os.rename(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def load_checkpoint(directory: Path) -> tuple[RawMLP, dict]:
    root = directory.resolve(strict=True)
    manifest = _json(root / "manifest.json")
    if manifest.get("schema_version") != BASELINE_VERSION:
        raise ValueError("Unknown supervised model manifest")
    if manifest.get("encoding_version") != ENCODING_VERSION:
        raise ValueError("Model encoding version mismatch")
    weights = root / "weights.npz"
    if sha256_file(weights) != manifest.get("weights_sha256"):
        raise ValueError("Corrupt supervised checkpoint")
    model = RawMLP().eval()
    parameters = dict(model.named_parameters())
    with np.load(weights, allow_pickle=False) as arrays:
        if set(arrays.files) != set(parameters):
            raise ValueError("Checkpoint parameter names mismatch")
        with torch.no_grad():
            for name, parameter in parameters.items():
                tensor = arrays[name]
                if tensor.shape != tuple(parameter.shape) or tensor.dtype != np.float32:
                    raise ValueError("Checkpoint parameter shape/dtype mismatch")
                if not np.isfinite(tensor).all():
                    raise ValueError("Non-finite checkpoint parameter")
                parameter.copy_(torch.from_numpy(tensor.copy()))
    if model_hash(model) != manifest.get("model_hash"):
        raise ValueError("Checkpoint model hash mismatch")
    return model, manifest
