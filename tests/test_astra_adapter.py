"""Offline contract checks: CI never calls a billable model."""

import json

import pytest

from owc.experiments import astra_adapter


def test_foundry_adapter_emits_only_patch_and_saves_usage(monkeypatch, tmp_path, capsys):
    request = {"protocol": "owc.agent-run.v1", "model_hash": "a" * 64,
               "round": 0, "teacher_feedback": []}
    patch = {"schema_version": "owc.weightpatch.v1", "expected_model_hash": "a" * 64,
             "actor": "foundry:gpt-6-astra", "reason": "Connect the first layer",
             "ops": [{"kind": "set", "tensor": "layers.0.weight",
                      "index": [0, 0], "value": 0.1}]}
    provider_response = {
        "id": "resp_fixture", "model": "gpt-6-astra", "status": "completed",
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "output": [{"content": [{"type": "output_text", "text": json.dumps(patch)}]}],
    }

    class Response:
        status = 200
        headers = {"x-request-id": "fixture-request"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, limit):
            return json.dumps(provider_response).encode()

    def fake_open(http_request, timeout):
        payload = json.loads(http_request.data)
        assert payload["model"] == "gpt-6-astra"
        assert payload["text"]["format"]["type"] == "json_object"
        assert json.loads(payload["input"])["teacher_feedback"] == []
        assert timeout == 90
        return Response()

    monkeypatch.setattr(astra_adapter, "_read_request", lambda: request)
    monkeypatch.setattr(astra_adapter.urllib.request, "urlopen", fake_open)
    monkeypatch.setenv("OWC_ASTRA_ENDPOINT", "https://example.invalid/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fixture-secret")
    monkeypatch.setenv("OWC_ASTRA_DIAGNOSTICS_DIR", str(tmp_path))
    assert astra_adapter.main() == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == patch
    assert captured.err == ""
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["response_id"] == "resp_fixture"
    assert saved["usage"]["input_tokens"] == 100
    assert "fixture-secret" not in json.dumps(saved)


def test_patch_identity_mismatch_is_rejected():
    response = {"status": "completed", "output": [{"content": [{
        "type": "output_text", "text": json.dumps({
            "schema_version": "owc.weightpatch.v1", "expected_model_hash": "wrong",
            "actor": "foundry:gpt-6-astra", "reason": "bad", "ops": [],
        })
    }]}]}
    with pytest.raises(ValueError, match="hash or actor mismatch"):
        astra_adapter._extract_patch(response, "correct", "foundry:gpt-6-astra")


def test_nonfinite_edit_is_rejected_before_stdout():
    patch = {
        "schema_version": "owc.weightpatch.v1", "expected_model_hash": "a" * 64,
        "actor": "foundry:gpt-6-astra", "reason": "bad number",
        "ops": [{"kind": "set", "tensor": "layers.0.weight",
                 "index": [0, 0], "value": float("inf")}],
    }
    response = {"status": "completed", "output": [{"content": [{
        "type": "output_text", "text": json.dumps(patch),
    }]}]}
    with pytest.raises(ValueError, match="finite"):
        astra_adapter._extract_patch(response, "a" * 64, "foundry:gpt-6-astra")
