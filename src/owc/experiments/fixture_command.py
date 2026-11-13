"""Example external protocol adapter for CI (NOT a reasoning or Astra model).

Usage: python -m owc.experiments.fixture_command
One JSON request on stdin -> one patch object on stdout.
"""

import json
import sys

from owc.weightspace.patches import PATCH_SCHEMA


def main() -> None:
    request = json.load(sys.stdin)
    assert request["protocol"] == "owc.agent-run.v1"
    response = {
        "schema_version": PATCH_SCHEMA,
        "expected_model_hash": request["model_hash"],
        "actor": "fixture:external-json",
        "reason": "Deterministic subprocess protocol test, no learning",
        "ops": [{
            "kind": "set", "tensor": "layers.4.bias",
            "index": [0], "value": 0.01 * (request["round"] + 1),
        }],
    }
    print(json.dumps(response))


if __name__ == "__main__":
    main()
