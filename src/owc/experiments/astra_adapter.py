"""Non-interactive Microsoft Foundry Responses adapter for direct weight edits.

Configuration is supplied through environment variables, never experiment JSON.
The external command still needs OS isolation from sealed test material.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

import torch

from owc.models.raw_mlp import RawMLP
from owc.weightspace.patches import (
    MAX_MAGNITUDE,
    MAX_OPERATIONS,
    MAX_SCALARS,
    PATCH_SCHEMA,
    _apply_single,
)

MAX_REQUEST_BYTES = 256_000
MAX_RESPONSE_BYTES = 1_000_000

SYSTEM_INSTRUCTIONS = """You are the declared direct weight-programming agent for OWC.
Return exactly one owc.weightpatch.v1 JSON object, with keys schema_version,
expected_model_hash, actor, reason, ops. Do not use gradient descent,
backpropagation, finite differences, random search, or an optimizer. Reason
about chess and explicitly program tensor values using only M2 set, add, and
rank_one_add operations. Each op must identify an existing tensor and legal
indices. Max 128 operations, 10000 addressed cells, finite values only.
The model starts at all zeros: a useful initial patch must create a connected
nonzero path through layers.0.weight -> layers.2.weight -> layers.4.weight.
layers.0 is 32x781, layers.2 is 16x32, layers.4 is 1x16. ReLU follows
layers.0 and layers.2; tanh follows layers.4. An output bias alone is useless.
Positive output means better for the side to move. Distinguish a neural
construction from a chess heuristic encoded into weights in your reason.
Feature planes are absolute board coordinates, not relative to the mover.
White pawn, knight, bishop, rook, queen, king use planes 0..5; black uses
6..11. Plane p square s has feature index 64*p+s, with a1=0 and h8=63.
Feature 768 is 1 for white to move and 0 for black. 769..772 are WK,WQ,BK,BQ
castling rights. 773..780 indicate legal en-passant target file a..h.
For rank_one_add supply rows, cols, left, right, scale; its contribution
at (rows[i],cols[j]) is scale*left[i]*right[j]. You may use it to program
many explicit piece/square weights in one operation. Explain the chess
hypothesis in reason. Do not request tools or filesystem access."""


def _read_request() -> dict:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("Agent request exceeds byte limit")
    request = json.loads(raw)
    if not isinstance(request, dict) or request.get("protocol") != "owc.agent-run.v1":
        raise ValueError("Expected owc.agent-run.v1 request")
    return request


def _endpoint() -> str:
    base = os.environ["OWC_ASTRA_ENDPOINT"].rstrip("/")
    parsed = urlparse(base)
    if (parsed.scheme != "https" or not parsed.netloc or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("OWC_ASTRA_ENDPOINT must be an HTTPS inference base URL")
    if not parsed.path.endswith("/openai/v1"):
        raise ValueError("OWC_ASTRA_ENDPOINT must end in /openai/v1")
    return base + "/responses"


def _extract_patch(response: dict, expected_hash: str, actor: str) -> dict:
    if response.get("status") != "completed":
        raise ValueError(f"Provider response status: {response.get('status')}")
    parts = [part.get("text") for item in response.get("output", [])
             for part in item.get("content", []) if part.get("type") == "output_text"]
    if len(parts) != 1 or not isinstance(parts[0], str):
        raise ValueError("Expected exactly one model output text")
    patch = json.loads(parts[0])
    if not isinstance(patch, dict) or set(patch) != {
        "schema_version", "expected_model_hash", "actor", "reason", "ops"
    }:
        raise ValueError("Model did not return exactly one M2 patch object")
    if patch["schema_version"] != PATCH_SCHEMA:
        raise ValueError("Wrong patch schema")
    if patch["expected_model_hash"] != expected_hash or patch["actor"] != actor:
        raise ValueError("Model patch hash or actor mismatch")
    if (not isinstance(patch["reason"], str) or not patch["reason"].strip()
            or len(patch["reason"]) > 1000):
        raise ValueError("Patch reason must be nonempty and bounded")
    if not isinstance(patch["ops"], list) or not 1 <= len(patch["ops"]) <= MAX_OPERATIONS:
        raise ValueError("Patch operation count exceeds M2 bounds")
    # Shape/value preflight uses M2's operation parser. The runner later applies
    # the patch against the actual revision and checks its hash atomically.
    scratch = RawMLP()
    edited = 0
    with torch.no_grad():
        tensors = dict(scratch.named_parameters())
        for operation in patch["ops"]:
            edited += _apply_single(tensors, operation)
            if edited > MAX_SCALARS:
                raise ValueError("Patch exceeds addressed-cell budget")
        for tensor in tensors.values():
            if not torch.isfinite(tensor).all() or torch.any(torch.abs(tensor) > MAX_MAGNITUDE):
                raise ValueError("Patch preflight produced invalid values")
    return patch


def _diagnostic(data: dict) -> None:
    directory = os.environ.get("OWC_ASTRA_DIAGNOSTICS_DIR")
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"{time.time_ns()}-{uuid.uuid4().hex}.json").open("x") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> int:
    started = time.monotonic()
    diagnostic: dict = {"adapter_version": "owc.foundry-responses.v1"}
    try:
        request = _read_request()
        endpoint = _endpoint()
        deployment = os.environ.get("OWC_ASTRA_DEPLOYMENT", "gpt-6-astra")
        actor = os.environ.get("OWC_ASTRA_ACTOR", f"foundry:{deployment}")
        max_tokens = int(os.environ.get("OWC_ASTRA_MAX_OUTPUT_TOKENS", "6000"))
        if not 256 <= max_tokens <= 12000:
            raise ValueError("Output token budget must be in [256,12000]")
        timeout = float(os.environ.get("OWC_ASTRA_TIMEOUT_SECONDS", "90"))
        if not 1 <= timeout <= 300:
            raise ValueError("Provider timeout must be in [1,300]")
        payload = {
            "model": deployment,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": json.dumps(request, sort_keys=True, allow_nan=False),
            "max_output_tokens": max_tokens,
            "text": {"format": {"type": "json_object"}},
        }
        diagnostic.update({
            "endpoint": endpoint, "deployment_requested": deployment,
            "actor": actor, "parameters": {"max_output_tokens": max_tokens,
                                      "text_format": "json_object"},
            "request_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
            "agent_request": request,
        })
        wire = json.dumps(payload, allow_nan=False).encode()
        http_request = urllib.request.Request(
            endpoint, data=wire, method="POST",
            headers={"Content-Type": "application/json",
                     "api-key": os.environ["AZURE_OPENAI_API_KEY"]},
        )
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Provider response exceeds byte limit")
            provider_response = json.loads(raw)
            diagnostic["http_status"] = response.status
            diagnostic["request_id_header"] = response.headers.get("x-request-id")
        diagnostic.update({
            "provider_response": provider_response,
            "response_id": provider_response.get("id"),
            "model_reported": provider_response.get("model"),
            "usage": provider_response.get("usage"),
        })
        if provider_response.get("model") != deployment:
            raise ValueError("Provider-reported model differs from requested deployment")
        patch = _extract_patch(provider_response, request["model_hash"], actor)
        diagnostic["patch"] = patch
        diagnostic["status"] = "completed"
        print(json.dumps(patch, sort_keys=True, allow_nan=False))
        return 0
    except urllib.error.HTTPError as error:
        diagnostic.update({"status": "failed", "http_status": error.code,
                           "error_type": "HTTPError"})
        print(f"Foundry HTTP {error.code}; see adapter diagnostics", file=sys.stderr)
        return 1
    except (KeyError, ValueError, OSError, TimeoutError) as error:
        diagnostic.update({"status": "failed", "error_type": type(error).__name__,
                           "error": str(error)[:500]})
        print(f"Astra adapter failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        diagnostic["elapsed_seconds"] = time.monotonic() - started
        _diagnostic(diagnostic)


if __name__ == "__main__":
    raise SystemExit(main())
