"""Self-hosted HTTP server: run Beveridge on your own machine or server and call it from any language.

    pip install "bev-decisions[serve]"
    bev serve --port 8000 [--model ...] [--host 127.0.0.1]

    POST /v1/decide   {"state": "...", "questions": {...}}  ->  {"answers": {...}}
    POST /v1/batch    {"requests": [{"state": ..., "questions": ...}, ...]}  ->  {"results": [{"answers": ...}, ...]}
    GET  /v1/presets  ->  preset names;  GET /health
It binds to 127.0.0.1 by default and has no authentication: put it behind your own gateway before exposing it.
Requests are processed one batch at a time on the device.
"""


import threading  # no "from __future__ import annotations" here: FastAPI needs real types for the request models
from typing import Any


def create_app(model: Any):  # type: ignore[no-untyped-def]
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    from . import presets

    class Decide(BaseModel):
        state: Any
        questions: dict[str, dict] | None = None
        preset: str | None = None

    class Batch(BaseModel):
        requests: list[Decide]

    app = FastAPI(title="Beveridge", version="0.2")
    lock = threading.Lock()

    def qs(r: Decide) -> dict:
        if r.preset:
            return presets.get(r.preset)
        if not r.questions:
            raise HTTPException(422, "give questions or a preset")
        return r.questions

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/v1/presets")
    def preset_names() -> dict:
        return {"presets": presets.names()}

    @app.post("/v1/decide")
    def decide(r: Decide) -> dict:
        try:
            with lock:
                return {"answers": model.decide(r.state, qs(r))}
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/v1/batch")
    def batch(b: Batch) -> dict:
        try:
            with lock:
                return {"results": [{"answers": a} for a in model.decide_batch([(r.state, qs(r)) for r in b.requests])]}
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    return app


def serve(model: Any, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(model), host=host, port=port)
