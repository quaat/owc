"""Validated, explicit tensor edits. No arbitrary code or optimizer calls."""

import copy
import hashlib
import json
import math
from collections.abc import Mapping

import torch
from torch import nn

PATCH_SCHEMA = "owc.weightpatch.v1"
MAX_OPERATIONS = 128
MAX_SCALARS = 10_000
MAX_MAGNITUDE = 1_000_000.0


def model_hash(model: nn.Module) -> str:
    """Hash model state with stable tensor names, shapes, and float32 byte order."""
    digest = hashlib.sha256(b"OWC model parameters v1\0")
    for name, tensor in sorted(model.named_parameters()):
        array = tensor.detach().cpu().contiguous().numpy()
        if array.dtype.name != "float32":
            raise ValueError(f"Only float32 tensors are supported: {name}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Non-finite parameter: {name}")
        digest.update(
            json.dumps({"name": name, "shape": list(array.shape)}, sort_keys=True).encode()
        )
        digest.update(array.astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result) or abs(result) > MAX_MAGNITUDE:
        raise ValueError(f"{field} must be finite and <= {MAX_MAGNITUDE} in magnitude")
    return result


def _index(value: object, size: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
        raise ValueError(f"{field} must be an integer within [0, {size})")
    return value


def _indices(value: object, size: int, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    indices = [_index(item, size, field) for item in value]
    if len(indices) != len(set(indices)):
        raise ValueError(f"{field} contains duplicate indices")
    return indices


def _require_fields(op: dict, fields: set[str]) -> None:
    if set(op) != fields:
        raise ValueError(
            f"Operation fields must be {sorted(fields)}; got {sorted(op)}"
        )


def _apply_single(
    tensors: Mapping[str, nn.Parameter], op: dict
) -> int:
    if not isinstance(op, dict):
        raise ValueError("Every operation must be an object")
    kind = op.get("kind")
    name = op.get("tensor")
    if not isinstance(name, str) or name not in tensors:
        raise ValueError(f"Unknown tensor: {name!r}")
    tensor = tensors[name]
    if kind in ("set", "add"):
        _require_fields(op, {"kind", "tensor", "index", "value"})
        raw_index = op["index"]
        if not isinstance(raw_index, list) or len(raw_index) != tensor.ndim:
            raise ValueError("Scalar index must identify exactly one tensor element")
        index = tuple(
            _index(idx, size, f"index dimension {dimension}")
            for dimension, (idx, size) in enumerate(zip(raw_index, tensor.shape))
        )
        value = _number(op["value"], "value")
        if kind == "set":
            tensor[index] = value
        else:
            tensor[index] += value
        return 1
    if kind == "rank_one_add":
        _require_fields(
            op, {"kind", "tensor", "rows", "cols", "left", "right", "scale"}
        )
        if tensor.ndim != 2:
            raise ValueError("rank_one_add requires a matrix parameter")
        rows = _indices(op["rows"], tensor.shape[0], "rows")
        cols = _indices(op["cols"], tensor.shape[1], "cols")
        left, right = op["left"], op["right"]
        if (
            not isinstance(left, list)
            or not isinstance(right, list)
            or len(left) != len(rows)
            or len(right) != len(cols)
        ):
            raise ValueError("left/right must match rows/cols lengths")
        a = [_number(value, "left") for value in left]
        b = [_number(value, "right") for value in right]
        scale = _number(op["scale"], "scale")
        count = len(rows) * len(cols)
        if count > MAX_SCALARS:
            raise ValueError("rank_one_add exceeds scalar edit budget")
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                delta = scale * a[i] * b[j]
                if not math.isfinite(delta) or abs(delta) > MAX_MAGNITUDE:
                    raise ValueError("rank_one_add produces excessive/non-finite delta")
                tensor[row, col] += delta
        return count
    raise ValueError(f"Unknown patch operation: {kind!r}")


def apply_patch(model: nn.Module, patch: dict) -> dict:
    """Atomically stage all operations, then publish only if the result is valid.

    The caller may persist the model and audit record after this operation.
    An invalid patch leaves the original model bit-for-bit untouched.
    """
    if not isinstance(patch, dict) or set(patch) != {
        "schema_version", "expected_model_hash", "actor", "reason", "ops"
    }:
        raise ValueError("Patch must have exactly the documented top-level fields")
    if patch["schema_version"] != PATCH_SCHEMA:
        raise ValueError("Unsupported patch schema_version")
    for field in ("actor", "reason"):
        value = patch[field]
        if not isinstance(value, str) or not value.strip() or len(value) > 1000:
            raise ValueError(f"{field} must be non-empty and at most 1000 characters")
    ops = patch["ops"]
    if not isinstance(ops, list) or not 1 <= len(ops) <= MAX_OPERATIONS:
        raise ValueError(f"ops must have between 1 and {MAX_OPERATIONS} operations")
    before = model_hash(model)
    if patch["expected_model_hash"] != before:
        raise ValueError("Model hash precondition mismatch; refresh patch before editing")
    candidate = copy.deepcopy(model)
    edited = 0
    with torch.no_grad():
        tensors = dict(candidate.named_parameters())
        for op in ops:
            edited += _apply_single(tensors, op)
            if edited > MAX_SCALARS:
                raise ValueError("Patch exceeds total scalar edit budget")
        for name, tensor in tensors.items():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Patch produced non-finite values in {name}")
            if torch.any(torch.abs(tensor) > MAX_MAGNITUDE):
                raise ValueError(f"Patch exceeds parameter magnitude limit in {name}")

    after = model_hash(candidate)
    if after == before:
        raise ValueError("Patch does not change the network")
    changed = {
        name: int(torch.count_nonzero(old != dict(candidate.named_parameters())[name]).item())
        for name, old in model.named_parameters()
        if torch.any(old != dict(candidate.named_parameters())[name])
    }
    with torch.no_grad():
        model.load_state_dict(candidate.state_dict(), strict=True)
    return {
        "schema_version": PATCH_SCHEMA,
        "before_hash": before,
        "after_hash": after,
        "operations": len(ops),
        "edited_cells": edited,
        "changed_parameters": changed,
        "actor": patch["actor"],
        "reason": patch["reason"],
    }


def inspect_weights(
    model: nn.Module, *, tensor: str | None = None, max_values: int = 12
) -> dict:
    """Summarize parameters, optionally previewing a bounded tensor prefix."""
    if isinstance(max_values, bool) or not 0 <= max_values <= 128:
        raise ValueError("max_values must be between 0 and 128")
    named = dict(model.named_parameters())
    if tensor is not None and tensor not in named:
        raise ValueError(f"Unknown tensor: {tensor}")
    summary = {}
    for name, param in sorted(named.items()):
        array = param.detach().cpu().flatten()
        item = {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "nonzero": int(torch.count_nonzero(array).item()),
            "min": float(array.min().item()),
            "max": float(array.max().item()),
            "mean": float(array.mean().item()),
        }
        if name == tensor:
            item["first_values"] = array[:max_values].tolist()
        summary[name] = item
    return {"model_hash": model_hash(model), "tensors": summary}
