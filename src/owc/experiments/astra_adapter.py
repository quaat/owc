"""Non-interactive Microsoft Foundry Responses adapter for direct weight edits.

Configuration is supplied through environment variables, never experiment JSON.
The external command still needs OS isolation from sealed test material.
"""

import base64
import hashlib
import json
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

import torch

from owc.experiments.request_wire import MAX_REQUEST_BYTES, canonical_bytes
from owc.models.raw_mlp import RawMLP
from owc.weightspace.patches import (
    MAX_MAGNITUDE,
    MAX_OPERATIONS,
    MAX_SCALARS,
    PATCH_SCHEMA,
    _apply_single,
)

MAX_RESPONSE_BYTES = 1_000_000
PUBLIC_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

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


def _diagnostic_directory() -> Path:
    directory = os.environ.get("OWC_ASTRA_DIAGNOSTICS_DIR")
    if not directory:
        raise ValueError(
            "OWC_ASTRA_DIAGNOSTICS_DIR is required for restricted transcript retention"
        )
    path = Path(directory)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if (path.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) & 0o077):
        raise PermissionError("Diagnostics directory must be owned by adapter and mode 0700")
    return path


def _write_private(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _diagnostic(directory: Path, data: dict) -> None:
    def public_identifier(value: object) -> str | None:
        return value if isinstance(value, str) and PUBLIC_IDENTIFIER.fullmatch(value) else None

    usage = data.get("usage")
    public_usage = None
    if isinstance(usage, dict):
        public_usage = {
            key: usage.get(key) if isinstance(usage.get(key), int)
            and not isinstance(usage.get(key), bool) and usage[key] >= 0 else None
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
    stem = f"{time.time_ns()}-{uuid.uuid4().hex}"
    raw = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _write_private(directory / f"{stem}.restricted.json", raw)
    public = {
        "adapter_version": data["adapter_version"],
        "status": data.get("status"),
        "error_type": data.get("error_type"),
        "failure_reason_code": data.get("failure_reason_code"),
        "http_status": data.get("http_status"),
        "deployment_requested": public_identifier(data.get("deployment_requested")),
        "model_reported": public_identifier(data.get("model_reported")),
        "response_model_policy": data.get("response_model_policy"),
        "request_id_header": public_identifier(data.get("request_id_header")),
        "response_id": public_identifier(data.get("response_id")),
        "usage": public_usage,
        "elapsed_seconds": data.get("elapsed_seconds"),
        "agent_request_sha256": data.get("agent_request_sha256"),
        "provider_request_sha256": data.get("provider_request_sha256"),
        "provider_response_sha256": data.get("provider_response_sha256"),
        "instructions_sha256": data.get("instructions_sha256"),
        "adapter_source_sha256": data.get("adapter_source_sha256"),
        "preflight_sha256": data.get("preflight_sha256"),
        "restricted_transcript_sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write_private(
        directory / f"{stem}.public.json",
        (json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )


def _persist_preflight(directory: Path, diagnostic: dict) -> str:
    """Persist exact call inputs and hashes before any provider request."""
    preflight = {
        key: diagnostic[key] for key in (
            "adapter_version", "endpoint", "deployment_requested", "actor",
            "parameters", "agent_request_sha256", "instructions_sha256",
            "adapter_source_sha256", "provider_request_sha256",
            "provider_request_raw_b64",
        )
    }
    preflight["status"] = "prepared_before_provider_call"
    wire = (json.dumps(preflight, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    name = f"{time.time_ns()}-{uuid.uuid4().hex}.preflight.restricted.json"
    _write_private(directory / name, wire)
    return hashlib.sha256(wire).hexdigest()


def main() -> int:
    started = time.monotonic()
    diagnostic: dict = {"adapter_version": "owc.foundry-responses.v1"}
    directory = None
    patch = None
    error_message = None
    try:
        directory = _diagnostic_directory()
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
                                      "text_format": "json_object", "timeout_seconds": timeout},
            "agent_request_sha256": hashlib.sha256(canonical_bytes(request)).hexdigest(),
            "instructions_sha256": hashlib.sha256(SYSTEM_INSTRUCTIONS.encode()).hexdigest(),
            "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "agent_request": request,
        })
        wire = canonical_bytes(payload)
        diagnostic["provider_request_sha256"] = hashlib.sha256(wire).hexdigest()
        diagnostic["provider_request_raw_b64"] = base64.b64encode(wire).decode()
        diagnostic["preflight_sha256"] = _persist_preflight(directory, diagnostic)
        http_request = urllib.request.Request(
            endpoint, data=wire, method="POST",
            headers={"Content-Type": "application/json",
                     "api-key": os.environ["AZURE_OPENAI_API_KEY"]},
        )
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Provider response exceeds byte limit")
            diagnostic["provider_response_raw_b64"] = base64.b64encode(raw).decode()
            diagnostic["provider_response_sha256"] = hashlib.sha256(raw).hexdigest()
            diagnostic["http_status"] = response.status
            diagnostic["request_id_header"] = response.headers.get("x-request-id")
        provider_response = json.loads(raw)
        if not isinstance(provider_response, dict):
            raise ValueError("Provider response must be a JSON object")
        diagnostic.update({
            "provider_response": provider_response,
            "response_id": provider_response.get("id"),
            "model_reported": provider_response.get("model"),
            "usage": provider_response.get("usage"),
        })
        reported = provider_response.get("model")
        if reported == deployment:
            diagnostic["response_model_policy"] = "exact_match"
        elif (os.environ.get("OWC_ASTRA_ALLOWED_RESPONSE_MODEL")
              and reported == os.environ["OWC_ASTRA_ALLOWED_RESPONSE_MODEL"]):
            diagnostic["response_model_policy"] = "operator_allowed_mapping"
        else:
            raise ValueError("Provider-reported model has no explicit allowed mapping")
        patch = _extract_patch(provider_response, request["model_hash"], actor)
        diagnostic["patch"] = patch
        diagnostic["status"] = "completed"
    except urllib.error.HTTPError as error:
        try:
            error_body = error.read(MAX_RESPONSE_BYTES + 1)
            if len(error_body) <= MAX_RESPONSE_BYTES:
                diagnostic["provider_response_raw_b64"] = base64.b64encode(error_body).decode()
                diagnostic["provider_response_sha256"] = hashlib.sha256(error_body).hexdigest()
            diagnostic["request_id_header"] = error.headers.get("x-request-id")
        except OSError:
            pass
        diagnostic.update({"status": "failed", "http_status": error.code,
                           "error_type": "HTTPError", "failure_reason_code": "http_error"})
        error_message = f"Foundry HTTP {error.code}; see adapter diagnostics"
    except (KeyError, ValueError, OSError, TimeoutError, AttributeError, TypeError) as error:
        if isinstance(error, TimeoutError):
            reason_code = "provider_timeout"
        elif isinstance(error, urllib.error.URLError):
            reason_code = "network_error"
        elif isinstance(error, PermissionError):
            reason_code = "diagnostics_permission_error"
        elif isinstance(error, ValueError) and "model" in str(error).lower():
            reason_code = "model_identity_or_response_error"
        else:
            reason_code = "adapter_validation_or_io_error"
        diagnostic.update({"status": "failed", "error_type": type(error).__name__,
                           "error": str(error)[:500],
                           "failure_reason_code": reason_code})
        error_message = f"Astra adapter failed: {type(error).__name__}; see restricted diagnostics"
    diagnostic["elapsed_seconds"] = time.monotonic() - started
    if directory is not None:
        try:
            _diagnostic(directory, diagnostic)
        except OSError as error:
            print(
                f"Astra adapter diagnostic write failed: {type(error).__name__}",
                file=sys.stderr,
            )
            return 1
    if error_message is not None:
        print(error_message, file=sys.stderr)
        return 1
    print(json.dumps(patch, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
