import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from bev.server import create_app


def test_decide_batch_presets_and_errors(fake):
    c = TestClient(create_app(fake))
    assert c.get("/health").json() == {"ok": True}
    assert "ticket_triage" in c.get("/v1/presets").json()["presets"]
    r = c.post("/v1/decide", json={"state": "x", "questions": {"u": {"type": "noul", "instructions": "Urgent?"}}})
    assert r.status_code == 200 and r.json()["answers"]["u"]["noul"] == 0.9
    assert c.post("/v1/batch", json={"requests": [{"state": "a", "preset": "ticket_triage"}]}).status_code == 200
    assert c.post("/v1/decide", json={"state": "x"}).status_code == 422
