"""MCP server: let AI assistants and agents call Bev as a tool (Model Context Protocol, stdio).

    pip install "bev-decisions[mcp]"
    bev mcp [--model ...]

Tools: decide(state, questions) -> answers;  decide_preset(state, preset) -> answers;  list_presets() -> names.
Example client configuration: {"command": "bev", "args": ["mcp"]}.
"""

from __future__ import annotations

from typing import Any


def run(model: Any) -> None:
    from mcp.server.fastmcp import FastMCP

    from . import presets

    mcp = FastMCP("bev")

    @mcp.tool()
    def decide(state: str, questions: dict[str, dict]) -> dict:
        """Answer typed questions about a document. questions: {name: {"type": "noul"|"choice"|"score",
        "instructions": "...", "criteria": {key: description} or [levels]}}. Returns answers with probabilities."""
        return model.decide(state, questions)

    @mcp.tool()
    def decide_preset(state: str, preset: str) -> dict:
        """Answer one of the ready-made question sets (see list_presets) about a document."""
        return model.decide(state, presets.get(preset))

    @mcp.tool()
    def list_presets() -> list[str]:
        """Names of the ready-made question sets."""
        return presets.names()

    mcp.run()
