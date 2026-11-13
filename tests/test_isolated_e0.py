"""E0 namespace and credential boundary contracts, without provider calls."""

import os
from pathlib import Path

import pytest

from owc.experiments import isolated_e0
from owc.experiments.isolated_e0 import restricted_environment, sandbox_prefix


def test_sandbox_masks_test_locations_and_only_binds_diagnostics(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(isolated_e0.shutil, "which", lambda _: "/usr/bin/bwrap")
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir(mode=0o700)
    diagnostics.chmod(0o700)
    command = sandbox_prefix(tmp_path, diagnostics)
    assert command[:4] == ["bwrap", "--die-with-parent", "--unshare-user", "--unshare-pid"]
    mounts = list(zip(command, command[1:], strict=False))
    assert ("--tmpfs", "/tmp") in mounts
    assert ("--tmpfs", "/home") in mounts
    assert ("--tmpfs", "/mnt/datasets") in mounts
    assert ("--tmpfs", "/mnt/runs") in mounts
    assert ("--bind", str(diagnostics)) in mounts
    assert command[-1] == "--"


def test_restricted_environment_forwards_only_required_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fixture-key")
    monkeypatch.setenv("GH_TOKEN", "must-not-reach-model-process")
    env = restricted_environment(
        endpoint="https://example.invalid/openai/v1",
        diagnostics=tmp_path, heldout=Path("/tmp/test.fen"),
        max_tokens=3000, timeout=90,
    )
    assert env["AZURE_OPENAI_API_KEY"] == "fixture-key"
    assert env["OWC_ASTRA_DIAGNOSTICS_DIR"] == "/mnt/runs/diagnostics"
    assert env["OWC_ASTRA_MAX_OUTPUT_TOKENS"] == "3000"
    assert "GH_TOKEN" not in env
    assert "OWC_ASTRA_ENDPOINT" in env
    assert "PYTHONPATH" not in env  # bubblewrap sets the mounted source path
    assert set(env).isdisjoint({"HOME_HOST", "CODEX_HOME", "GITHUB_TOKEN"})


def test_restricted_environment_rejects_unbounded_call(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fixture-key")
    with pytest.raises(ValueError, match="limits output"):
        restricted_environment(
            endpoint="https://example.invalid/openai/v1", diagnostics=tmp_path,
            heldout=Path("/tmp/test.fen"), max_tokens=12000, timeout=90,
        )
    assert os.environ["AZURE_OPENAI_API_KEY"] == "fixture-key"
