import importlib
import json
import sys

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    mode = tmp_path / "claude"
    mode.mkdir()
    (mode / "20260101-000000-abc.json").write_text(
        json.dumps(
            {
                "trace_id": "abc",
                "mode": "claude",
                "backend": "claude",
                "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": "2026-01-01T00:00:11+00:00",
                "query": "quiet sushi in Deep Ellum",
                "models": ["claude-sonnet-4-6"],
                "spans": [{"type": "llm"}, {"type": "tool"}],
                "totals": {"llm_calls": 1, "tool_calls": 1, "input_tokens": 100, "output_tokens": 20},
            }
        )
    )
    monkeypatch.setenv("AF_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("AF_API_BASE", "http://test-api:9999")
    sys.modules.pop("scripts.trace_ui", None)
    mod = importlib.import_module("scripts.trace_ui")
    return TestClient(mod.app)


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>Anything Finder" in r.text


def test_overview_aggregates_mode(client):
    m = client.get("/api/overview").json()["modes"]
    assert m[0]["name"] == "claude"
    assert m[0]["runs"] == 1
    assert m[0]["models"] == ["claude-sonnet-4-6"]
    assert m[0]["input_tokens"] == 100


def test_traces_and_trace(client):
    rows = client.get("/api/traces?mode=claude").json()
    assert rows[0]["query"] == "quiet sushi in Deep Ellum"
    assert rows[0]["mode"] == "claude"
    doc = client.get("/api/trace", params={"file": rows[0]["file"]}).json()
    assert doc["trace_id"] == "abc"


def test_trace_path_traversal_blocked(client):
    assert client.get("/api/trace", params={"file": "../../../etc/hosts"}).status_code == 404
    assert client.request("DELETE", "/api/trace", params={"file": "../../etc/hosts"}).status_code == 404


def test_delete_trace_removes_file_and_empty_mode_dir(client, tmp_path):
    rows = client.get("/api/traces?mode=claude").json()
    r = client.request("DELETE", "/api/trace", params={"file": rows[0]["file"]})
    assert r.status_code == 200 and r.json()["deleted"] == rows[0]["file"]
    assert not (tmp_path / rows[0]["file"]).exists()
    assert not (tmp_path / "claude").exists()  # emptied dir is pruned
    assert client.get("/api/traces?mode=claude").json() == []
    assert client.request("DELETE", "/api/trace", params={"file": rows[0]["file"]}).status_code == 404


def test_config_reports_api_base(client):
    cfg = client.get("/api/config").json()
    assert cfg["api_base"] == "http://test-api:9999"
    assert cfg["api"]["reachable"] is False  # nothing listening
