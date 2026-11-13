"""Run one E0 adapter call in a network-permitted mount/PID namespace.

The runner stays outside the namespace to read development labels. The model
adapter sees only the read-only source/venv, an empty /tmp, an empty home,
masked datasets/runs, and its one private writable diagnostics directory.
Requires Linux bubblewrap. Never evaluates or reads the held-out benchmark.
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from owc.experiments.astra_adapter import SYSTEM_INSTRUCTIONS
from owc.experiments.runner import CommandAgent, ExperimentSettings, run_experiment

SANDBOX_DIAGNOSTICS = "/mnt/runs/diagnostics"
# The E0 cost preflight uses a smaller provider-wire cap than general M3
# feedback runs. This is a byte cap, not a live billing enforcement mechanism.
MAX_E0_PROVIDER_REQUEST_BYTES = 32_000
PROBE = """import json, os, urllib.error, urllib.request
from urllib.parse import urlparse
heldout = os.environ["OWC_HELDOUT_DENY_PATH"]
result = {
    "heldout_access_denied": not os.path.exists(heldout),
    "home_hidden": not os.path.exists("/home/thomas"),
    "datasets_hidden": not os.path.exists("/mnt/datasets/astra-smoke-30"),
    "credential_present": bool(os.environ.get("AZURE_OPENAI_API_KEY")),
}
base = os.environ["OWC_ASTRA_ENDPOINT"].rstrip("/")
if urlparse(base).scheme != "https" or not base.endswith("/openai/v1"):
    raise ValueError("Invalid inference base URL")
request = urllib.request.Request(base + "/models", headers={
    "api-key": os.environ["AZURE_OPENAI_API_KEY"]})
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        models = json.load(response)
        entries = models.get("data", []) if isinstance(models, dict) else []
        result["authenticated_models_http"] = response.status
        result["deployment_listed"] = any(
            isinstance(item, dict) and item.get("id") == os.environ["OWC_ASTRA_DEPLOYMENT"]
            for item in entries
        )
except urllib.error.HTTPError as error:
    result["authenticated_models_http"] = error.code
    result["deployment_listed"] = False
except Exception as error:
    result["network_error_type"] = type(error).__name__
    result["deployment_listed"] = False
print(json.dumps(result, sort_keys=True))
"""


def sandbox_prefix(root: Path, diagnostics: Path) -> list[str]:
    if shutil.which("bwrap") is None:
        raise FileNotFoundError("bubblewrap is required for isolated E0")
    root = root.resolve(strict=True)
    diagnostics = diagnostics.resolve(strict=True)
    if diagnostics.is_symlink() or diagnostics.stat().st_uid != os.getuid():
        raise PermissionError("Diagnostics must be owned by the current user")
    if diagnostics.stat().st_mode & 0o077:
        raise PermissionError("Diagnostics must have no group/world access")
    executable = Path(sys.executable).resolve(strict=True)
    extra = []
    if executable.is_relative_to("/tmp"):
        # The pinned Python distribution is installed under /tmp on this host.
        distribution = Path("/tmp") / executable.relative_to("/tmp").parts[0]
        extra = ["--ro-bind", str(distribution), str(distribution)]
    return [
        "bwrap", "--die-with-parent", "--unshare-user", "--unshare-pid",
        "--ro-bind", "/", "/", "--tmpfs", "/tmp", *extra,
        "--tmpfs", "/home", "--ro-bind", str(root), "/mnt",
        "--tmpfs", "/mnt/datasets", "--tmpfs", "/mnt/runs",
        "--bind", str(diagnostics), SANDBOX_DIAGNOSTICS,
        "--proc", "/proc", "--dev", "/dev",
        "--setenv", "PYTHONPATH", "/mnt/src", "--chdir", "/mnt", "--",
    ]


def restricted_environment(*, endpoint: str, diagnostics: Path,
                           heldout: Path, max_tokens: int, timeout: int) -> dict[str, str]:
    if not endpoint.startswith("https://") or not endpoint.rstrip("/").endswith("/openai/v1"):
        raise ValueError("OWC_ASTRA_ENDPOINT must be an HTTPS /openai/v1 base")
    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not key:
        raise ValueError("AZURE_OPENAI_API_KEY is required")
    if not 256 <= max_tokens <= 3000:
        raise ValueError("This E0 gate limits output to 256..4000 tokens")
    if not 1 <= timeout <= 120:
        raise ValueError("This E0 gate limits provider timeout to 1..120 seconds")
    env = {
        "PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONNOUSERSITE": "1",
        "AZURE_OPENAI_API_KEY": key,
        "OWC_ASTRA_ENDPOINT": endpoint.rstrip("/"),
        "OWC_ASTRA_DEPLOYMENT": os.environ.get("OWC_ASTRA_DEPLOYMENT", "gpt-6-astra"),
        "OWC_ASTRA_ACTOR": os.environ.get("OWC_ASTRA_ACTOR", "foundry:gpt-6-astra"),
        "OWC_ASTRA_DIAGNOSTICS_DIR": SANDBOX_DIAGNOSTICS,
        "OWC_ASTRA_MAX_OUTPUT_TOKENS": str(max_tokens),
        "OWC_ASTRA_MAX_PROVIDER_REQUEST_BYTES": str(MAX_E0_PROVIDER_REQUEST_BYTES),
        "OWC_ASTRA_TIMEOUT_SECONDS": str(timeout),
        "OWC_HELDOUT_DENY_PATH": str(heldout),
    }
    allowed = os.environ.get("OWC_ASTRA_ALLOWED_RESPONSE_MODEL")
    if allowed:
        env["OWC_ASTRA_ALLOWED_RESPONSE_MODEL"] = allowed
    return env


def _private_json(path: Path, payload: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(*, root: Path, diagnostics: Path, heldout: Path,
            endpoint: str, max_tokens: int, timeout: int) -> tuple[list[str], dict, dict]:
    if not heldout.is_absolute() or not heldout.is_file():
        raise FileNotFoundError("Held-out denial path must exist on the host")
    env = restricted_environment(endpoint=endpoint, diagnostics=diagnostics,
                                 heldout=heldout, max_tokens=max_tokens, timeout=timeout)
    diagnostics.mkdir(mode=0o700, parents=True, exist_ok=False)
    prefix = sandbox_prefix(root, diagnostics)
    command = prefix + ["/mnt/.venv/bin/python", "-c", PROBE]
    completed = subprocess.run(command, env=env, capture_output=True, text=True,
                               timeout=20, check=False)
    if completed.returncode:
        raise RuntimeError("Isolated network/access probe failed before E0")
    probe = json.loads(completed.stdout)
    required = ("heldout_access_denied", "home_hidden", "datasets_hidden",
                "credential_present", "deployment_listed")
    if not all(probe.get(key) is True for key in required):
        raise RuntimeError(f"Isolated E0 preflight failed: {probe}")
    if probe.get("authenticated_models_http") != 200:
        raise RuntimeError(f"Authenticated model listing failed: {probe}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    preparation = {
        "schema_version": "owc.isolated-e0-preparation.v1",
        "commit": commit, "working_tree_dirty": bool(status.strip()), "probe": probe,
        "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
        "instructions_sha256": hashlib.sha256(SYSTEM_INSTRUCTIONS.encode()).hexdigest(),
        "adapter_source_sha256": hashlib.sha256(
            (root / "src/owc/experiments/astra_adapter.py").read_bytes()
        ).hexdigest(),
        "launcher_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_output_tokens": max_tokens, "provider_timeout_seconds": timeout,
        "provider_request_max_bytes": MAX_E0_PROVIDER_REQUEST_BYTES,
        "request_max_bytes": 256_000, "request_headroom_bytes": 4_096,
        "dollar_cost_ceiling": None,
        "dollar_cost_note": (
            "Published Azure standard rates allow a conditional estimate, but the "
            "actual billing SKU must be verified before a paid E0 call. This byte "
            "limit is not a hard monetary spending control."
        ),
    }
    _private_json(diagnostics / "preparation.json", preparation)
    return prefix, env, preparation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--heldout-deny-path", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    parser.add_argument("--provider-timeout", type=int, default=90)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    endpoint = os.environ.get("OWC_ASTRA_ENDPOINT", "")
    prefix, env, preparation = prepare(
        root=root, diagnostics=args.diagnostics_dir.resolve(),
        heldout=args.heldout_deny_path, endpoint=endpoint,
        max_tokens=args.max_output_tokens, timeout=args.provider_timeout,
    )
    print(json.dumps(preparation, sort_keys=True))
    if args.probe_only:
        return
    command = prefix + ["/mnt/.venv/bin/python", "-m", "owc.experiments.astra_adapter"]
    agent = CommandAgent(shlex.join(command), env["OWC_ASTRA_ACTOR"], env=env)
    run = run_experiment(
        labels=args.dev_labels, output_dir=args.output_dir, agent=agent,
        settings=ExperimentSettings(feedback_rounds=0, seed=11, probe_count=4,
                                    max_label_disclosures=0, max_seconds=150),
    )
    print(json.dumps({"status": run["status"], "final_model_hash": run["final_model_hash"],
                      "feedback_disclosures": run["feedback_disclosures"]}, sort_keys=True))


if __name__ == "__main__":
    main()
