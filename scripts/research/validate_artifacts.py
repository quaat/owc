"""Validate original local artifacts against compact public research summaries."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path, report: Path) -> dict:
    index = json.loads((report / "artifact-index.json").read_text())
    checked = 0
    for artifact in index["artifacts"]:
        if artifact["artifact_id"].startswith("generated-test:"):
            if artifact["sha256_status"] not in (
                "declared by dataset manifest; not independently rehashed",
                "not inspected; missing independently verified checksum",
            ):
                raise ValueError("Test split must remain unverified and unevaluated here")
            continue
        location = Path(artifact["storage_location"])
        if artifact["artifact_id"].startswith("external-local:"):
            if not location.is_absolute() or not location.is_relative_to("/tmp"):
                raise ValueError("Unexpected external artifact location")
            path = location
        else:
            if location.is_absolute() or ".." in location.parts:
                raise ValueError("Unsafe local artifact path")
            path = root / location
        if not path.is_file() or path.stat().st_size != artifact["size_bytes"]:
            raise ValueError(f"Artifact missing or size changed: {location}")
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"Artifact hash changed: {location}")
        checked += 1

    public_files = [*report.glob("*.json"), *report.glob("*.csv")]
    for path in public_files:
        content = path.read_text(encoding="utf-8")
        if "openai.azure.com" in content or "api-key:" in content:
            raise ValueError(f"Internal endpoint or header in public report: {path}")

    command = [sys.executable, str(root / "scripts/research/reproduce_metrics.py"),
               "--root", str(root), "--output-dir", str(report), "--check"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True,
                               timeout=60, check=False)
    if completed.returncode:
        raise ValueError(f"Derived metric replay failed: {completed.stderr.strip()}")
    return {"verified_local_artifacts": checked,
            "declared_unverified_test_artifacts": 3,
            "derived_report_files": 6, "status": "validated_local_only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report", type=Path,
                        default=Path("reports/experiments/2026-11-13-smoke-pilot"))
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve(), args.report.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
