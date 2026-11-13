"""Offline contract checks: CI never calls a billable model."""

import hashlib
import json
import stat

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
        assert len(list(tmp_path.glob("*.preflight.restricted.json"))) >= 1
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
    restricted_path = next(path for path in tmp_path.glob("*.restricted.json")
                           if ".preflight." not in path.name)
    public_path = next(tmp_path.glob("*.public.json"))
    preflight_path = next(tmp_path.glob("*.preflight.restricted.json"))
    saved = json.loads(restricted_path.read_text())
    public = json.loads(public_path.read_text())
    assert saved["response_id"] == public["response_id"] == "resp_fixture"
    assert saved["usage"]["input_tokens"] == 100
    assert saved["provider_response_raw_b64"]
    assert saved["provider_request_raw_b64"]
    assert saved["instructions_sha256"] == hashlib.sha256(
        astra_adapter.SYSTEM_INSTRUCTIONS.encode()
    ).hexdigest()
    assert saved["adapter_source_sha256"] == hashlib.sha256(
        astra_adapter.Path(astra_adapter.__file__).read_bytes()
    ).hexdigest()
    assert public["provider_response_sha256"] == hashlib.sha256(
        json.dumps(provider_response).encode()
    ).hexdigest()
    assert public["restricted_transcript_sha256"] == hashlib.sha256(
        restricted_path.read_bytes()
    ).hexdigest()
    assert public["preflight_sha256"] == hashlib.sha256(preflight_path.read_bytes()).hexdigest()
    assert stat.S_IMODE(preflight_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(restricted_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(public_path.stat().st_mode) == 0o600
    assert "fixture-secret" not in json.dumps(saved)
    assert "example.invalid" not in json.dumps(public)
    assert "agent_request" not in public

    provider_response["model"] = "underlying-astra"
    assert astra_adapter.main() == 1
    assert "no explicit allowed mapping" in next(
        json.loads(path.read_text()).get("error", "") for path in
        tmp_path.glob("*.restricted.json") if "error" in json.loads(path.read_text())
    )
    monkeypatch.setenv("OWC_ASTRA_ALLOWED_RESPONSE_MODEL", "underlying-astra")
    assert astra_adapter.main() == 0
    assert any(json.loads(path.read_text()).get("response_model_policy") ==
               "operator_allowed_mapping" for path in tmp_path.glob("*.public.json"))
    provider_response["id"] = "https://internal.example/secret"
    assert astra_adapter.main() == 0
    assert any(json.loads(path.read_text()).get("response_id") is None
               for path in tmp_path.glob("*.public.json"))
    provider_response["model"] = None
    monkeypatch.delenv("OWC_ASTRA_ALLOWED_RESPONSE_MODEL")
    assert astra_adapter.main() == 1
    provider_response["model"] = "gpt-6-astra"
    provider_response["id"] = "resp_fixture"
    capsys.readouterr()
    monkeypatch.setattr(astra_adapter, "_diagnostic", lambda *_: (_ for _ in ()).throw(
        PermissionError("simulated storage failure")
    ))
    assert astra_adapter.main() == 1
    assert capsys.readouterr().out == ""


def test_diagnostics_directory_must_be_private(monkeypatch, tmp_path, capsys):
    public_dir = tmp_path / "public"
    public_dir.mkdir(mode=0o755)
    public_dir.chmod(0o755)
    monkeypatch.setenv("OWC_ASTRA_DIAGNOSTICS_DIR", str(public_dir))
    assert astra_adapter.main() == 1
    assert capsys.readouterr().out == ""
    assert list(public_dir.iterdir()) == []


def test_provider_timeout_is_recorded_without_patch(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(astra_adapter, "_read_request", lambda: {
        "protocol": "owc.agent-run.v1", "model_hash": "a" * 64,
        "round": 0, "teacher_feedback": [],
    })
    monkeypatch.setenv("OWC_ASTRA_ENDPOINT", "https://example.invalid/openai/v1")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fixture-secret")
    monkeypatch.setenv("OWC_ASTRA_DIAGNOSTICS_DIR", str(tmp_path))

    def timeout(*_args, **_kwargs):
        raise TimeoutError("fixture timeout")

    monkeypatch.setattr(astra_adapter.urllib.request, "urlopen", timeout)
    assert astra_adapter.main() == 1
    assert capsys.readouterr().out == ""
    public = json.loads(next(tmp_path.glob("*.public.json")).read_text())
    assert public["status"] == "failed"
    assert public["error_type"] == "TimeoutError"
    assert public["failure_reason_code"] == "provider_timeout"
    assert public["provider_response_sha256"] is None


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
