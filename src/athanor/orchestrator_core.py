"""Compatibility wrapper around the canonical orchestrator module."""

from .solver.events import EventType, OrchestratorEvent, EventCallback
from .solver.orchestrator import (
    run_orchestration,
    get_tool_schemas,
    load_puzzle,
    load_system_prompt,
    format_puzzle_for_prompt,
    default_cli_callback,
)

__all__ = [
    "EventType",
    "OrchestratorEvent",
    "EventCallback",
    "run_orchestration",
    "get_tool_schemas",
    "load_puzzle",
    "load_system_prompt",
    "format_puzzle_for_prompt",
    "default_cli_callback",
]
