#!/usr/bin/env python3
"""FastAPI + Vue frontend for ARC solver orchestration.

This is an alternative to gradio_demo.py with:
- WebSocket streaming updates
- Solve / Stop controls
- Tabbed panels (Agent Conversation, prompts, code/tool panels, config)
- Auto-follow behavior in Agent Conversation
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import queue
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv, find_dotenv

try:
    from ..data import list_tasks, resolve_dataset_root, resolve_task_path
    from ..orchestrator_core import (
        run_orchestration,
        load_puzzle,
        load_system_prompt,
        format_puzzle_for_prompt,
        get_tool_schemas,
        EventType,
        OrchestratorEvent,
    )
    from ..solver.independent_reflector import _parse_reflector_response as parse_reflector_response
    from ..solver.phoenix_observability import initialize_phoenix
    from ..solver.prompts import build_candidate_expansion_guidance_prompt, CONTEXT_COMPRESSION_PROMPT
except ImportError:
    from athanor.data import list_tasks, resolve_dataset_root, resolve_task_path
    from athanor.orchestrator_core import (
        run_orchestration,
        load_puzzle,
        load_system_prompt,
        format_puzzle_for_prompt,
        get_tool_schemas,
        EventType,
        OrchestratorEvent,
    )
    from athanor.solver.independent_reflector import _parse_reflector_response as parse_reflector_response
    from athanor.solver.phoenix_observability import initialize_phoenix
    from athanor.solver.prompts import build_candidate_expansion_guidance_prompt, CONTEXT_COMPRESSION_PROMPT

load_dotenv(find_dotenv(usecwd=True), override=True)
logger = logging.getLogger(__name__)

MODEL_PRICING_USD_PER_MTOK = {
    # Anthropic base input/output list prices. Prompt caching is derived from
    # these with Anthropic's published 5m multipliers:
    #   cache write = 1.25x input
    #   cache read  = 0.10x input
    "claude-opus-4-6": {"provider": "anthropic", "input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"provider": "anthropic", "input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"provider": "anthropic", "input": 0.80, "output": 4.00},
    "gemini-3.1-pro-preview": {"provider": "gemini", "input": 2.00, "output": 12.00},
    "gemini-3-pro-preview": {"provider": "gemini", "input": 2.00, "output": 12.00},
    "gemini-2.5-pro-preview-06-05": {"provider": "gemini", "input": 1.25, "output": 10.00},
    # Fireworks Kimi pricing from https://fireworks.ai/pricing (checked 2026-03-09).
    "kimi-k2p5": {"provider": "fireworks", "input": 0.60, "output": 3.00, "cached_input": 0.10},
    "kimi-k2.5": {"provider": "fireworks", "input": 0.60, "output": 3.00, "cached_input": 0.10},
}

ANTHROPIC_CACHE_WRITE_MULTIPLIER = 1.25
ANTHROPIC_CACHE_READ_MULTIPLIER = 0.10
GEMINI_31_TIER_BREAKPOINT_TOKENS = 200_000
GEMINI_31_LOWER_TIER = {"input": 2.00, "output": 12.00, "cached_input": 0.20}
GEMINI_31_UPPER_TIER = {"input": 4.00, "output": 18.00, "cached_input": 0.40}

KIMI_RUN_CODE_GUIDANCE_TEXT = (
    "Access train/test grids via `train_samples[idx]['input']`, "
    "`train_samples[idx]['output']`, and `test_samples[idx]['input']` in exploratory code tools. "
    "Use `run_code` to start fresh and `run_code_in_previous_runtime` to reuse the current live exploratory runtime when available. "
    "Tool results report whether the runtime was fresh, reused, or a fresh fallback. "
    "To save tokens, do not copy/paste full grid lists into exploratory code payloads."
)

_DATA_URL_RE = re.compile(r"^data:(?P<media>[^;]+);base64,(?P<data>.+)$", re.DOTALL)


def _normalize_model_name(model_name: str) -> str:
    raw = str(model_name or "").strip().lower()
    if raw.startswith("accounts/fireworks/models/"):
        raw = raw.split("/")[-1]
    return raw


def _is_kimi_model(model_name: str) -> bool:
    normalized = _normalize_model_name(model_name)
    return normalized.startswith("kimi")


def _is_glm_model(model_name: str) -> bool:
    normalized = _normalize_model_name(model_name)
    return normalized.startswith("glm")


def _infer_provider(model_name: str, provider: str | None = None) -> str:
    explicit = str(provider or "").strip().lower()
    if explicit:
        return explicit
    normalized = _normalize_model_name(model_name)
    if normalized.startswith("claude-"):
        return "anthropic"
    if normalized.startswith("gemini-"):
        return "gemini"
    if normalized.startswith("kimi") or normalized.startswith("glm"):
        return "fireworks"
    return ""


def _pricing_profile_for_model(
    model_name: str,
    provider: str | None = None,
    *,
    request_input_tokens: int = 0,
) -> dict[str, float | str | bool]:
    normalized = _normalize_model_name(model_name)
    inferred_provider = _infer_provider(normalized, provider)

    if inferred_provider == "anthropic":
        if normalized.startswith("claude-opus-4"):
            return {"provider": "anthropic", "input": 5.00, "output": 25.00, "pricing_estimate_incomplete": False}
        if normalized.startswith("claude-sonnet-4"):
            return {"provider": "anthropic", "input": 3.00, "output": 15.00, "pricing_estimate_incomplete": False}
        if normalized.startswith("claude-haiku"):
            return {"provider": "anthropic", "input": 0.80, "output": 4.00, "pricing_estimate_incomplete": False}
        return {"provider": "anthropic", "input": 5.00, "output": 25.00, "pricing_estimate_incomplete": True}

    if inferred_provider == "gemini":
        if normalized.startswith("gemini-3.1-pro-preview") or normalized.startswith("gemini-3-pro-preview"):
            tier = GEMINI_31_LOWER_TIER if int(request_input_tokens or 0) <= GEMINI_31_TIER_BREAKPOINT_TOKENS else GEMINI_31_UPPER_TIER
            return {
                "provider": "gemini",
                "input": tier["input"],
                "output": tier["output"],
                "cached_input": tier["cached_input"],
                "pricing_estimate_incomplete": False,
            }
        if normalized.startswith("gemini-2.5-pro"):
            return {"provider": "gemini", "input": 1.25, "output": 10.00, "pricing_estimate_incomplete": False}
        return {"provider": "gemini", "input": 2.00, "output": 12.00, "pricing_estimate_incomplete": True}

    if normalized.startswith("kimi"):
        return {"provider": "fireworks", "input": 0.60, "output": 3.00, "cached_input": 0.10, "pricing_estimate_incomplete": False}

    rates = MODEL_PRICING_USD_PER_MTOK.get(normalized)
    if rates is not None:
        profile = dict(rates)
        profile["pricing_estimate_incomplete"] = False
        return profile

    return {"provider": inferred_provider or "unknown", "input": 5.00, "output": 25.00, "pricing_estimate_incomplete": True}


def _estimate_anthropic_request_cost_usd(
    *,
    input_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    input_rate: float,
    output_rate: float,
    output_tokens_include_reasoning: bool,
) -> float:
    generated_tokens = int(output_tokens or 0) if output_tokens_include_reasoning else int(output_tokens or 0) + int(reasoning_tokens or 0)
    cache_write_rate = input_rate * ANTHROPIC_CACHE_WRITE_MULTIPLIER
    cache_read_rate = input_rate * ANTHROPIC_CACHE_READ_MULTIPLIER
    return (
        int(input_tokens or 0) * input_rate
        + int(cache_creation_input_tokens or 0) * cache_write_rate
        + int(cache_read_input_tokens or 0) * cache_read_rate
        + generated_tokens * output_rate
    ) / 1_000_000.0


def _estimate_gemini_request_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    input_rate: float,
    output_rate: float,
    output_tokens_include_reasoning: bool,
) -> float:
    generated_tokens = int(output_tokens or 0) if output_tokens_include_reasoning else int(output_tokens or 0) + int(reasoning_tokens or 0)
    return (int(input_tokens or 0) * input_rate + generated_tokens * output_rate) / 1_000_000.0


def _estimate_fireworks_request_cost_usd(
    *,
    input_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    input_rate: float,
    output_rate: float,
    cached_input_rate: float,
    output_tokens_include_reasoning: bool,
) -> float:
    generated_tokens = int(output_tokens or 0) if output_tokens_include_reasoning else int(output_tokens or 0) + int(reasoning_tokens or 0)
    return (
        int(input_tokens or 0) * input_rate
        + int(cache_creation_input_tokens or 0) * input_rate
        + int(cache_read_input_tokens or 0) * cached_input_rate
        + generated_tokens * output_rate
    ) / 1_000_000.0


def _default_phoenix_project_for_model(model_name: str) -> str:
    normalized = _normalize_model_name(model_name)
    if "opus-4" in normalized:
        return "ARC_Opus_4_6"
    if "sonnet-4" in normalized:
        return "ARC_Sonnet_4_6"
    if "haiku-4" in normalized:
        return "ARC_Haiku_4_5"
    if normalized.startswith("kimi"):
        return "ARC_Kimi_K2p5"
    if normalized.startswith("glm"):
        return "ARC_GLM_5"
    return "ARC_Athanor"


def _default_efforts_for_model(model_name: str) -> dict[str, str]:
    normalized = _normalize_model_name(model_name)
    if "opus-4" in normalized or "sonnet-4" in normalized:
        return {
            "thinking_effort": "medium",
            "reflection_thinking_effort": "max",
            "compression_thinking_effort": "max",
        }
    return {
        "thinking_effort": "low",
        "reflection_thinking_effort": "medium",
        "compression_thinking_effort": "medium",
    }


def _append_kimi_run_code_guidance_if_needed(
    user_prompt_blocks: list[dict[str, Any]],
    model_name: str,
) -> list[dict[str, Any]]:
    if not _is_kimi_model(model_name) or _is_glm_model(model_name):
        return list(user_prompt_blocks or [])
    blocks = list(user_prompt_blocks or [])
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            if KIMI_RUN_CODE_GUIDANCE_TEXT in str(block.get("text", "") or ""):
                return blocks
    blocks.append({"type": "text", "text": KIMI_RUN_CODE_GUIDANCE_TEXT})
    return blocks


def _estimate_request_cost_usd(
    *,
    model_name: str,
    input_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    output_tokens: int,
    reasoning_tokens: int,
    output_tokens_include_reasoning: bool = False,
    provider: str | None = None,
) -> float:
    rates = _pricing_profile_for_model(model_name, provider, request_input_tokens=int(input_tokens or 0))
    provider_name = str(rates.get("provider", "") or _infer_provider(model_name, provider))
    input_rate = float(rates.get("input", 0.0))
    output_rate = float(rates.get("output", 0.0))

    if provider_name == "anthropic":
        return _estimate_anthropic_request_cost_usd(
            input_tokens=input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            input_rate=input_rate,
            output_rate=output_rate,
            output_tokens_include_reasoning=output_tokens_include_reasoning,
        )

    if provider_name == "gemini":
        return _estimate_gemini_request_cost_usd(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            input_rate=input_rate,
            output_rate=output_rate,
            output_tokens_include_reasoning=output_tokens_include_reasoning,
        )

    return _estimate_fireworks_request_cost_usd(
        input_tokens=input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        input_rate=input_rate,
        output_rate=output_rate,
        cached_input_rate=float(rates.get("cached_input", input_rate)),
        output_tokens_include_reasoning=output_tokens_include_reasoning,
    )


def _request_pricing_is_incomplete(
    *,
    model_name: str,
    provider: str | None,
    input_tokens: int,
) -> bool:
    profile = _pricing_profile_for_model(model_name, provider, request_input_tokens=int(input_tokens or 0))
    return bool(profile.get("pricing_estimate_incomplete", False))


def _build_request_ledger_entry(
    *,
    model_name: str,
    provider: str | None,
    input_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    output_tokens: int,
    reasoning_tokens: int,
    output_tokens_include_reasoning: bool = False,
) -> dict[str, Any]:
    billed_input_tokens = int(input_tokens or 0) + int(cache_creation_input_tokens or 0) + int(cache_read_input_tokens or 0)
    request_total_tokens = billed_input_tokens + (
        int(output_tokens or 0) if output_tokens_include_reasoning else int(output_tokens or 0) + int(reasoning_tokens or 0)
    )
    # Compute input vs output cost split
    rates = _pricing_profile_for_model(model_name, provider, request_input_tokens=int(input_tokens or 0))
    input_rate = float(rates.get("input", 0.0))
    output_rate = float(rates.get("output", 0.0))
    provider_name = str(rates.get("provider", "") or _infer_provider(model_name, provider))
    generated_tokens = int(output_tokens or 0) if output_tokens_include_reasoning else int(output_tokens or 0) + int(reasoning_tokens or 0)
    if provider_name == "anthropic":
        cache_write_rate = input_rate * ANTHROPIC_CACHE_WRITE_MULTIPLIER
        cache_read_rate = input_rate * ANTHROPIC_CACHE_READ_MULTIPLIER
        input_cost = (
            int(input_tokens or 0) * input_rate
            + int(cache_creation_input_tokens or 0) * cache_write_rate
            + int(cache_read_input_tokens or 0) * cache_read_rate
        ) / 1_000_000.0
    elif provider_name == "gemini":
        input_cost = int(input_tokens or 0) * input_rate / 1_000_000.0
    else:
        cached_input_rate = float(rates.get("cached_input", input_rate))
        input_cost = (
            int(input_tokens or 0) * input_rate
            + int(cache_creation_input_tokens or 0) * input_rate
            + int(cache_read_input_tokens or 0) * cached_input_rate
        ) / 1_000_000.0
    output_cost = generated_tokens * output_rate / 1_000_000.0
    return {
        "provider": _infer_provider(model_name, provider),
        "model": str(model_name or ""),
        "input_tokens_uncached": int(input_tokens or 0),
        "cache_write_tokens": int(cache_creation_input_tokens or 0),
        "cache_read_tokens": int(cache_read_input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "thinking_tokens": int(reasoning_tokens or 0),
        "output_includes_reasoning": bool(output_tokens_include_reasoning),
        "request_total_tokens": int(request_total_tokens),
        "estimated_cost_usd": float(input_cost + output_cost),
        "input_cost_usd": float(input_cost),
        "output_cost_usd": float(output_cost),
        "pricing_estimate_incomplete": _request_pricing_is_incomplete(
            model_name=model_name,
            provider=provider,
            input_tokens=int(input_tokens or 0),
        ),
    }


def _recalculate_usage_from_request_ledger(request_ledger: list[dict[str, Any]]) -> dict[str, Any]:
    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    reasoning_breakdown_incomplete = False
    pricing_estimate_incomplete = False

    for entry in request_ledger or []:
        if not isinstance(entry, dict):
            continue
        uncached_input_tokens = int(entry.get("input_tokens_uncached", 0) or 0)
        cache_write_tokens = int(entry.get("cache_write_tokens", 0) or 0)
        cache_read_tokens = int(entry.get("cache_read_tokens", 0) or 0)
        output_tokens = int(entry.get("output_tokens", 0) or 0)
        thinking_tokens = int(entry.get("thinking_tokens", 0) or 0)
        total_input_tokens += uncached_input_tokens + cache_write_tokens + cache_read_tokens
        total_output_tokens += output_tokens
        total_thinking_tokens += thinking_tokens
        total_tokens += int(
            entry.get(
                "request_total_tokens",
                uncached_input_tokens
                + cache_write_tokens
                + cache_read_tokens
                + (output_tokens if entry.get("output_includes_reasoning", False) else output_tokens + thinking_tokens),
            ) or 0
        )
        total_cost += float(entry.get("estimated_cost_usd", 0.0) or 0.0)
        if bool(entry.get("output_includes_reasoning", False)) and thinking_tokens == 0 and output_tokens > 0:
            reasoning_breakdown_incomplete = True
        if bool(entry.get("pricing_estimate_incomplete", False)):
            pricing_estimate_incomplete = True

    return {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "thinking_tokens": total_thinking_tokens,
        "total_tokens": total_tokens,
        "requests": len([e for e in request_ledger or [] if isinstance(e, dict)]),
        "total_cost": total_cost,
        "reasoning_breakdown_incomplete": reasoning_breakdown_incomplete,
        "pricing_estimate_incomplete": pricing_estimate_incomplete,
        "request_ledger": [copy.deepcopy(e) for e in request_ledger or [] if isinstance(e, dict)],
    }


def _recalculate_usage_from_history(history: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    requests = 0
    reasoning_breakdown_incomplete = False
    pricing_estimate_incomplete = False
    request_ledger: list[dict[str, Any]] = []

    main_model = str((config or {}).get("model") or "claude-opus-4-6")
    main_provider = _infer_provider(main_model)

    for msg in history or []:
        if not isinstance(msg, dict):
            continue

        billed_input_tokens = int(msg.get("usage_input_tokens", 0) or 0)
        output_tokens = int(msg.get("usage_output_tokens", 0) or 0)
        thinking_tokens = int(msg.get("usage_thinking_tokens", 0) or 0)
        cache_read_tokens = int(msg.get("usage_cache_read_tokens", 0) or 0)
        cache_write_tokens = int(msg.get("usage_cache_write_tokens", 0) or 0)
        output_includes_reasoning = bool(msg.get("usage_output_includes_reasoning", False))
        reasoning_reported = bool(msg.get("usage_reasoning_tokens_reported", thinking_tokens > 0))

        if billed_input_tokens <= 0 and output_tokens <= 0 and thinking_tokens <= 0:
            continue

        uncached_input_tokens = int(
            msg.get(
                "usage_uncached_input_tokens",
                max(0, billed_input_tokens - cache_write_tokens - cache_read_tokens),
            ) or 0
        )
        model_name = main_model
        provider = main_provider
        if str(msg.get("kind", "") or "") == "reflector_context":
            model_name = str(msg.get("reflector_model") or model_name)
            provider = _infer_provider(model_name, str(msg.get("reflector_provider") or provider))

        total_input_tokens += billed_input_tokens
        total_output_tokens += output_tokens
        total_thinking_tokens += thinking_tokens
        total_tokens += int(
            msg.get(
                "usage_total_tokens",
                billed_input_tokens + (output_tokens if output_includes_reasoning else output_tokens + thinking_tokens),
            ) or 0
        )
        ledger_entry = _build_request_ledger_entry(
            model_name=model_name,
            provider=provider,
            input_tokens=uncached_input_tokens,
            cache_creation_input_tokens=cache_write_tokens,
            cache_read_input_tokens=cache_read_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=thinking_tokens,
            output_tokens_include_reasoning=output_includes_reasoning,
        )
        total_cost += float(ledger_entry["estimated_cost_usd"])
        request_ledger.append(ledger_entry)
        requests += 1
        if output_includes_reasoning and not reasoning_reported and output_tokens > 0:
            reasoning_breakdown_incomplete = True
        if bool(ledger_entry.get("pricing_estimate_incomplete", False)):
            pricing_estimate_incomplete = True

    return {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "thinking_tokens": total_thinking_tokens,
        "total_tokens": total_tokens,
        "requests": requests,
        "total_cost": total_cost,
        "reasoning_breakdown_incomplete": reasoning_breakdown_incomplete,
        "pricing_estimate_incomplete": pricing_estimate_incomplete,
        "request_ledger": request_ledger,
    }


class WebEventHandler:
    """Event callback adapter + lightweight metrics/tool tracking."""

    def __init__(self):
        self.event_queue: queue.Queue[OrchestratorEvent] = queue.Queue()
        self.should_stop = False
        self.pause_requested = False
        self.terminate_requested = False
        self.streaming_message_active = False
        self.model_name = "claude-opus-4-6"
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.latest_code_submission: dict[str, Any] | None = None
        self.total_input_tokens = 0
        self.total_thinking_tokens = 0
        self.total_output_tokens = 0
        self.total_billed_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0
        self.total_input_cost = 0.0
        self.total_output_cost = 0.0
        self.reasoning_breakdown_incomplete = False
        self.pricing_estimate_incomplete = False
        self.request_ledger: list[dict[str, Any]] = []

    def _record_usage(
        self,
        *,
        model_name: str,
        provider: str | None,
        input_tokens: int,
        output_tokens: int,
        reasoning_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        output_tokens_include_reasoning: bool = False,
    ) -> None:
        ledger_entry = _build_request_ledger_entry(
            model_name=model_name,
            provider=provider,
            input_tokens=int(input_tokens or 0),
            cache_creation_input_tokens=int(cache_creation_input_tokens or 0),
            cache_read_input_tokens=int(cache_read_input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            reasoning_tokens=int(reasoning_tokens or 0),
            output_tokens_include_reasoning=output_tokens_include_reasoning,
        )
        billed_input_tokens = (
            int(ledger_entry["input_tokens_uncached"])
            + int(ledger_entry["cache_write_tokens"])
            + int(ledger_entry["cache_read_tokens"])
        )
        billed_total_tokens = int(ledger_entry["request_total_tokens"])

        self.total_input_tokens += billed_input_tokens
        self.total_output_tokens += int(output_tokens or 0)
        self.total_thinking_tokens += int(reasoning_tokens or 0)
        self.total_billed_tokens += billed_total_tokens
        self.total_requests += 1
        self.request_ledger.append(ledger_entry)
        if output_tokens_include_reasoning and reasoning_tokens == 0 and output_tokens > 0:
            self.reasoning_breakdown_incomplete = True
        if bool(ledger_entry.get("pricing_estimate_incomplete", False)):
            self.pricing_estimate_incomplete = True
        self.total_cost += float(ledger_entry["estimated_cost_usd"])
        self.total_input_cost += float(ledger_entry.get("input_cost_usd", 0.0))
        self.total_output_cost += float(ledger_entry.get("output_cost_usd", 0.0))

    def callback(self, event: OrchestratorEvent):
        self.event_queue.put(event)
        if event.type in (EventType.THINKING, EventType.TEXT):
            self.streaming_message_active = True
        elif event.type not in (EventType.SYSTEM,):
            self.streaming_message_active = False
        if event.type == EventType.SYSTEM and "_token_usage" in event.metadata:
            usage = event.metadata["_token_usage"]
            self._record_usage(
                model_name=self.model_name,
                provider=_infer_provider(self.model_name),
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
                cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
                reasoning_tokens=int(usage.get("reasoning_tokens", usage.get("thinking_tokens", 0)) or 0),
                output_tokens_include_reasoning=bool(usage.get("output_tokens_include_reasoning", False)),
            )

        if event.type == EventType.REFLECTOR_CONTEXT:
            meta = event.metadata if isinstance(event.metadata, dict) else {}
            if str(meta.get("phase", "")).lower() == "result":
                self._record_usage(
                    model_name=str(meta.get("model", "") or ""),
                    provider=str(meta.get("provider", "") or ""),
                    input_tokens=int(
                        meta.get(
                            "usage_uncached_input_tokens",
                            meta.get("usage_input_tokens", 0),
                        ) or 0
                    ),
                    cache_creation_input_tokens=int(meta.get("usage_cache_write_tokens", 0) or 0),
                    cache_read_input_tokens=int(meta.get("usage_cache_read_tokens", 0) or 0),
                    output_tokens=int(meta.get("usage_output_tokens", 0) or 0),
                    reasoning_tokens=int(meta.get("usage_thinking_tokens", 0) or 0),
                    output_tokens_include_reasoning=bool(meta.get("usage_output_includes_reasoning", False)),
                )

        if event.type == EventType.TOOL_CALL:
            self.tool_calls.append(
                {
                    "name": event.metadata.get("name", "unknown"),
                    "input": event.metadata.get("input", {}),
                    "timestamp": time.time(),
                }
            )
            if event.metadata.get("name") == "execute_python_solution":
                self.latest_code_submission = {
                    "input": event.metadata.get("input", {}),
                    "timestamp": time.time(),
                    "result": None,
                }

        if event.type == EventType.TOOL_RESULT:
            self.tool_results.append(
                {
                    "tool_name": event.metadata.get("tool_name", ""),
                    "output": event.metadata.get("output", event.content),
                    "result": event.metadata.get("result"),
                    "timestamp": time.time(),
                }
            )
            if self.latest_code_submission and event.metadata.get("tool_name") == "execute_python_solution":
                self.latest_code_submission["result"] = event.metadata.get("result", {})

    def check_stop(self) -> bool:
        if self.terminate_requested:
            return True
        if self.pause_requested and not self.streaming_message_active:
            return True
        return self.should_stop

    def stop(self):
        self.should_stop = True

    def request_pause(self):
        self.pause_requested = True

    def terminate(self):
        self.should_stop = True
        self.terminate_requested = True

    def stop_reason(self) -> str | None:
        if self.terminate_requested:
            return "terminate"
        if self.pause_requested:
            return "pause"
        if self.should_stop:
            return "stop"
        return None

    def reset(self):
        self.should_stop = False
        self.pause_requested = False
        self.terminate_requested = False
        self.streaming_message_active = False
        self.tool_calls = []
        self.tool_results = []
        self.latest_code_submission = None
        self.total_input_tokens = 0
        self.total_thinking_tokens = 0
        self.total_output_tokens = 0
        self.total_billed_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0
        self.total_input_cost = 0.0
        self.total_output_cost = 0.0
        self.reasoning_breakdown_incomplete = False
        self.pricing_estimate_incomplete = False
        self.request_ledger = []
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    def set_model_name(self, model_name: str) -> None:
        self.model_name = str(model_name or "claude-opus-4-6")


def _tool_schemas_markdown() -> str:
    tools = get_tool_schemas() or []
    if not tools:
        return "*No tool schemas available.*"
    parts: list[str] = []
    for t in tools:
        name = t.get("name", "unknown")
        desc = t.get("description", "")
        schema = t.get("input_schema", {})
        parts.append(f"### `{name}`")
        if desc:
            parts.append(desc)
        parts.append("**Input Schema**")
        parts.append("```json")
        parts.append(json.dumps(schema, indent=2, default=str))
        parts.append("```")
        parts.append("")
    return "\n".join(parts)


def _latest_tool_markdown(handler: WebEventHandler) -> str:
    if not handler.tool_calls:
        return "*No tool calls yet*"
    call = handler.tool_calls[-1]
    md: list[str] = []
    md.append("## Latest Tool Call")
    md.append("")
    md.append(f"**Name:** `{call.get('name', 'unknown')}`")
    md.append("")
    md.append(f"**Time:** {time.strftime('%H:%M:%S', time.localtime(call.get('timestamp', time.time())))}")
    md.append("")
    tool_input = call.get("input", {})
    if call.get("name") in {"run_code", "run_code_in_previous_runtime"} and isinstance(tool_input, dict) and "code" in tool_input:
        code = SolverAppState._clean_tool_code(tool_input.get("code", ""))
        md.append("### Code")
        md.append("```python")
        md.append(code)
        md.append("```")
        other = {k: v for k, v in tool_input.items() if k != "code"}
        if other:
            md.append("")
            md.append("### Other Parameters")
            md.append("```json")
            md.append(json.dumps(other, indent=2, default=str))
            md.append("```")
    else:
        md.append("### Input")
        md.append("```json")
        md.append(json.dumps(tool_input, indent=2, default=str))
        md.append("```")
    latest_result: dict[str, Any] | None = None
    call_name = str(call.get("name", ""))
    for tr in reversed(handler.tool_results):
        if str(tr.get("tool_name", "")) == call_name:
            latest_result = tr
            break
    if latest_result is None and handler.tool_results:
        latest_result = handler.tool_results[-1]

    if latest_result:
        md.append("")
        md.append("### Tool Result")
        result_obj = latest_result.get("result", None)
        output_obj = latest_result.get("output", "")
        if result_obj not in (None, "", [], {}):
            md.append("```json")
            md.append(json.dumps(result_obj, indent=2, default=str))
            md.append("```")
        else:
            text = str(output_obj or "").strip()
            if text:
                md.append("```text")
                md.append(text)
                md.append("```")
    return "\n".join(md)


def _latest_code_markdown(handler: WebEventHandler) -> str:
    submission = handler.latest_code_submission
    if not submission:
        return "*No code submitted yet*"
    md: list[str] = []
    md.append("## Latest Code Submission")
    md.append("")
    md.append(f"**Time:** {time.strftime('%H:%M:%S', time.localtime(submission.get('timestamp', time.time())))}")
    md.append("")
    submission_input = submission.get("input") or {}
    target_branch_id = str(submission_input.get("branch_id") or "").strip()
    target_branch_hypothesis = str(submission_input.get("branch_hypothesis") or "").strip()
    target_branch_code = str(submission_input.get("branch_code") or "").strip()
    branch_a_code = submission_input.get("branch_a_code", "")
    branch_b_code = submission_input.get("branch_b_code", "")
    updated_failed_code = submission_input.get("updated_failed_code", "")
    branch_a_hypothesis = str(submission_input.get("hypothesis") or "").strip()
    branch_b_hypothesis = ""
    ambiguity_rationale = ""
    code = SolverAppState._clean_tool_code(submission_input.get("code", "") or branch_a_code)
    branch_b_code_clean = SolverAppState._clean_tool_code(branch_b_code)
    updated_failed_code_clean = SolverAppState._clean_tool_code(updated_failed_code)

    result_obj = submission.get("result", None)
    result_branches = []
    if isinstance(result_obj, dict):
        result_branches = list(result_obj.get("branches", []) or [])
        if not branch_a_hypothesis and result_obj.get("hypothesis"):
            branch_a_hypothesis = str(result_obj.get("hypothesis") or "").strip()
        if result_branches and not branch_a_hypothesis:
            branch_a_hypothesis = str(result_branches[0].get("hypothesis", "") or "").strip()
        if len(result_branches) > 1:
            branch_b_hypothesis = str(result_branches[1].get("hypothesis", "") or "").strip()
        ambiguity_rationale = str(result_obj.get("ambiguity_rationale", "") or "").strip()

    if branch_a_hypothesis:
        md.append("### hypothesis")
        md.append(branch_a_hypothesis)
        md.append("")
    if branch_b_hypothesis:
        md.append("### branch_b_hypothesis")
        md.append(branch_b_hypothesis)
        md.append("")
    if ambiguity_rationale:
        md.append("### ambiguity_rationale")
        md.append(ambiguity_rationale)
        md.append("")
    if target_branch_id:
        md.append("### branch_id")
        md.append(target_branch_id)
        md.append("")
    if target_branch_hypothesis:
        md.append("### branch_hypothesis")
        md.append(target_branch_hypothesis)
        md.append("")
    if code:
        md.append("### code")
        md.append("```python")
        md.append(str(code))
        md.append("```")
    if target_branch_code:
        md.append("")
        md.append("### branch_code")
        md.append("```python")
        md.append(str(SolverAppState._clean_tool_code(target_branch_code)))
        md.append("```")
    if branch_b_code_clean:
        md.append("")
        md.append("### branch_b_code")
        md.append("```python")
        md.append(str(branch_b_code_clean))
        md.append("```")
    if updated_failed_code_clean:
        md.append("")
        md.append("### updated_failed_code")
        md.append("```python")
        md.append(str(updated_failed_code_clean))
        md.append("```")

    if result_obj not in (None, "", [], {}):
        md.append("")
        md.append("### tool_result")
        md.append("```json")
        md.append(json.dumps(result_obj, indent=2, default=str))
        md.append("```")
    return "\n".join(md)


class SolverAppState:
    def __init__(self):
        self.lock = threading.RLock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[WebSocket] = set()
        self.handler = WebEventHandler()

        self.running = False
        self.paused = False
        self.status = "idle"
        self.test_accuracy: float | None = None
        self.test_correct_count: int | None = None
        self.test_total: int | None = None
        self.test_solved_indices: list[int] | None = None
        self.worker_thread: threading.Thread | None = None

        self.history: list[dict[str, Any]] = []
        self.system_prompt_md = ""
        self.user_prompt_md = ""
        self.reflector_prompt_md = self._load_reflector_prompt_safe()
        self.latest_tool_md = "*No tool calls yet*"
        self.latest_code_md = "*No code submitted yet*"
        self.tools_md = _tool_schemas_markdown()

        self.current_thinking = ""
        self.current_text = ""
        self.current_thinking_idx: int | None = None
        self.current_text_idx: int | None = None
        self._turn_usage_target_idx: int | None = None
        self.partial_tool_call_indices: dict[str, int] = {}
        self.current_turn = 0
        self.current_iteration = 0
        self.reflector_context_idx: int | None = None
        self.current_canvas_id: str | None = None
        self.current_canvas_seq = 0
        self.current_canvas_label = ""
        self.current_canvas_owner = "main_agent"
        self.current_canvas_phase = "main"
        self.saved_runs_dir = Path(__file__).resolve().parent / "saved_runs"
        # Read-only release checkpoint set at repo root (release_runs/).
        # Listed alongside user's saved_runs/ but never written to.
        self.release_runs_dir = Path(__file__).resolve().parents[3] / "release_runs"
        self._task_ids_cache: dict[tuple[str, str], list[str]] = {}
        try:
            detected_dataset_root = str(resolve_dataset_root(None))
        except Exception:
            detected_dataset_root = os.getenv("ARC_DATA_ROOT", "")

        self.config = {
            "puzzle_path": "16b78196",
            "split": "public_eval",
            "dataset_root": detected_dataset_root,
            "unsafe_local_exec": True,
            "model": "claude-opus-4-6",
            **_default_efforts_for_model("claude-opus-4-6"),
            "thinking_budget": 16000,
            "compression_threshold": 170000,
            "compression_bypass_threshold": 120000,
            "max_turns": 200,
            "max_test_predictions": 2,
            "phoenix_project": _default_phoenix_project_for_model("claude-opus-4-6"),
            "enable_phoenix": True,
            "enable_independent_reflector": True,
            "reflector_provider": "gemini",
            "reflector_model": "gemini-3.1-pro-preview",
            "semi_cot_first_turn": False,
            "semi_cot_thinking_effort": "high",
        }
        self.task_ids: list[str] = self._list_task_ids_for_config(self.config)
        self._refresh_prompt_panels_from_config(emit=False)
        # Show system/user prompts in Agent Conversation immediately (before Solve).
        self.history = [
            {
                "role": "user",
                "kind": "prompt_bundle",
                "source": "orchestrator",
                "content": "",
                "system_md": self.system_prompt_md,
                "user_md": self.user_prompt_md,
            }
        ]

    def _usage_snapshot(self) -> dict[str, Any]:
        with self.lock:
            input_tokens = int(self.handler.total_input_tokens or 0)
            output_tokens = int(self.handler.total_output_tokens or 0)
            thinking_tokens = int(self.handler.total_thinking_tokens or 0)
            requests = int(self.handler.total_requests or 0)
            total_tokens = int(self.handler.total_billed_tokens or 0)
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "total_tokens": total_tokens,
                "requests": requests,
                "total_cost": float(self.handler.total_cost or 0.0),
                "input_cost": float(self.handler.total_input_cost or 0.0),
                "output_cost": float(self.handler.total_output_cost or 0.0),
                "reasoning_breakdown_incomplete": bool(self.handler.reasoning_breakdown_incomplete),
                "pricing_estimate_incomplete": bool(self.handler.pricing_estimate_incomplete),
                "request_ledger": copy.deepcopy(self.handler.request_ledger),
            }

    def _attach_usage_to_latest_agent_message(self, usage: dict[str, Any]) -> None:
        input_tokens = int(
            usage.get(
                "billed_input_tokens",
                int(usage.get("input_tokens", 0) or 0)
                + int(usage.get("cache_creation_input_tokens", 0) or 0)
                + int(usage.get("cache_read_input_tokens", 0) or 0),
            ) or 0
        )
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        thinking_tokens = int(usage.get("reasoning_tokens", usage.get("thinking_tokens", 0)) or 0)
        cache_read_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_write_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)
        output_includes_reasoning = bool(usage.get("output_tokens_include_reasoning", False))
        reasoning_tokens_reported = bool(usage.get("reasoning_tokens_reported", thinking_tokens > 0))
        billed_total_tokens = input_tokens + (
            output_tokens if output_includes_reasoning else output_tokens + thinking_tokens
        )

        with self.lock:
            # Use the dedicated turn usage target (set when first agent block
            # of the turn is created, survives _finalize_streams).
            target_idx = self._turn_usage_target_idx
            if target_idx is None:
                # Fallback: search backward for the latest agent block
                for i in range(len(self.history) - 1, -1, -1):
                    m = self.history[i]
                    kind = m.get("kind", "")
                    if kind == "turn_divider":
                        continue
                    if m.get("role") == "assistant" and m.get("source") == "agent":
                        target_idx = i
                        break
                    if kind in ("tool_result", "reflection_prompt", "prompt_bundle"):
                        break
            if target_idx is None:
                logger.warning("_attach_usage: no agent block found (total=%d, history_len=%d)",
                               input_tokens + output_tokens + thinking_tokens, len(self.history))
                return

            msg = dict(self.history[target_idx])
            # A single "turn" can include multiple provider requests (main pass +
            # compression/reflection follow-ups). Accumulate usage on the same
            # message instead of overwriting with the latest sub-request.
            msg["usage_input_tokens"] = int(msg.get("usage_input_tokens", 0) or 0) + input_tokens
            msg["usage_output_tokens"] = int(msg.get("usage_output_tokens", 0) or 0) + output_tokens
            msg["usage_thinking_tokens"] = int(msg.get("usage_thinking_tokens", 0) or 0) + max(0, thinking_tokens)
            msg["usage_cache_read_tokens"] = int(msg.get("usage_cache_read_tokens", 0) or 0) + cache_read_tokens
            msg["usage_cache_write_tokens"] = int(msg.get("usage_cache_write_tokens", 0) or 0) + cache_write_tokens
            msg["usage_output_includes_reasoning"] = bool(
                msg.get("usage_output_includes_reasoning", False) or output_includes_reasoning
            )
            msg["usage_reasoning_tokens_reported"] = bool(
                msg.get("usage_reasoning_tokens_reported", False) or reasoning_tokens_reported
            )
            msg["usage_total_tokens"] = int(msg.get("usage_total_tokens", 0) or 0) + billed_total_tokens
            # Accumulate cost from the latest ledger entry
            if self.handler.request_ledger:
                entry = self.handler.request_ledger[-1]
                msg["usage_input_cost_usd"] = float(msg.get("usage_input_cost_usd", 0.0) or 0.0) + float(entry.get("input_cost_usd", 0.0) or 0.0)
                msg["usage_output_cost_usd"] = float(msg.get("usage_output_cost_usd", 0.0) or 0.0) + float(entry.get("output_cost_usd", 0.0) or 0.0)
                msg["usage_cost_usd"] = float(msg.get("usage_cost_usd", 0.0) or 0.0) + float(entry.get("estimated_cost_usd", 0.0) or 0.0)
            self.history[target_idx] = msg

        self._emit({"type": "history_patch", "index": target_idx, "message": msg})

    async def add_client(self, ws: WebSocket):
        await ws.accept()
        with self.lock:
            self.clients.add(ws)
        await ws.send_json({"type": "snapshot", "state": self._snapshot()})

    def remove_client(self, ws: WebSocket):
        with self.lock:
            if ws in self.clients:
                self.clients.remove(ws)

    @staticmethod
    def _load_reflector_prompt_safe() -> str:
        try:
            from athanor.solver.independent_reflector import _load_reflector_prompt
            return _load_reflector_prompt()
        except Exception:
            return "*Reflector prompt not available.*"

    def _snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "paused": self.paused,
                "status": self.status,
                "history": self._normalize_history_list(self.history),
                "panels": {
                    "system_prompt": self.system_prompt_md or "*System prompt will be loaded when solving starts*",
                    "user_prompt": self.user_prompt_md or "*User prompt will be loaded when solving starts*",
                    "reflector_prompt": self.reflector_prompt_md,
                    "latest_tool": self.latest_tool_md,
                    "latest_code": self.latest_code_md,
                    "available_tools": _tool_schemas_markdown(),
                },
                "config": self.config,
                "task_ids": list(self.task_ids),
                "usage": self._usage_snapshot(),
                "saved_runs": self.list_saved_runs(),
            }

    def _list_task_ids_for_config(self, config_snapshot: dict[str, Any]) -> list[str]:
        split = str(config_snapshot.get("split") or "public_eval").strip() or "public_eval"
        dataset_root = str(config_snapshot.get("dataset_root") or "").strip()
        cache_key = (split.lower(), dataset_root)
        cached = self._task_ids_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        try:
            task_ids = list_tasks(split=split, dataset_root=dataset_root or None)
        except Exception:
            task_ids = []
        self._task_ids_cache[cache_key] = list(task_ids)
        return list(task_ids)

    @staticmethod
    def _safe_run_label(label: str) -> str:
        raw = str(label or "").strip().lower()
        raw = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-")
        return raw[:80] if raw else ""

    def list_saved_runs(self) -> list[dict[str, Any]]:
        self.saved_runs_dir.mkdir(parents=True, exist_ok=True)
        runs: list[dict[str, Any]] = []
        seen: set[str] = set()
        # User's saved_runs/ takes precedence over release_runs/ on filename collision.
        for src_dir in (self.saved_runs_dir, self.release_runs_dir):
            if not src_dir.exists():
                continue
            for p in sorted(src_dir.glob("*.json"), reverse=True):
                if p.name in seen:
                    continue
                seen.add(p.name)
                try:
                    doc = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                meta = doc.get("meta", {}) if isinstance(doc, dict) else {}
                runs.append(
                    {
                        "id": p.name,
                        "label": str(meta.get("label", "") or ""),
                        "saved_at": str(meta.get("saved_at", "") or ""),
                    }
                )
        return runs

    def save_current_run(self, label: str = "") -> tuple[bool, str]:
        self.saved_runs_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self._snapshot()

        safe = self._safe_run_label(label)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{safe}" if safe else ""
        filename_base = f"{ts}{suffix}"
        filename = f"{filename_base}.json"
        path = self.saved_runs_dir / filename
        counter = 2
        while path.exists():
            filename = f"{filename_base}_{counter}.json"
            path = self.saved_runs_dir / filename
            counter += 1

        # Saved runs should always be resumable when loaded back.
        snapshot["status"] = "loaded"
        payload = {
            "meta": {
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "label": str(label or "").strip(),
            },
            "state": snapshot,
        }
        try:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            return False, f"Failed to save run: {e}"
        return True, f"Saved run: {filename}"

    def load_saved_run(self, run_id: str) -> tuple[bool, str]:
        run_id_clean = str(run_id or "").strip()
        # Try user's saved_runs_dir first, fall back to release_runs_dir.
        target = self.saved_runs_dir / run_id_clean
        if not target.exists():
            target = self.release_runs_dir / run_id_clean
        if not target.exists() or target.suffix.lower() != ".json":
            return False, "Saved run not found."
        try:
            doc = json.loads(target.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Failed to read saved run: {e}"

        state = doc.get("state", {}) if isinstance(doc, dict) else {}
        if not isinstance(state, dict):
            return False, "Saved run is invalid."

        with self.lock:
            self.running = False
            self.paused = False
            self.status = str(state.get("status", "loaded"))
            loaded_history = list(state.get("history", [])) if isinstance(state.get("history", []), list) else []
            normalized_history = self._normalize_history_list(loaded_history)
            panels = state.get("panels", {}) if isinstance(state.get("panels", {}), dict) else {}
            main_system_prompt = str(panels.get("system_prompt", self.system_prompt_md))
            migrated_history, history_changed = self._migrate_history_canvases(normalized_history)
            backfilled_history, backfill_changed = self._backfill_loaded_history(
                migrated_history,
                main_system_md=main_system_prompt,
            )
            self.history = backfilled_history

            self.system_prompt_md = main_system_prompt
            self.user_prompt_md = str(panels.get("user_prompt", self.user_prompt_md))
            self.latest_tool_md = str(panels.get("latest_tool", self.latest_tool_md))
            self.latest_code_md = str(panels.get("latest_code", self.latest_code_md))
            # Tool definitions should always reflect the current backend code,
            # not the serialized snapshot from when a run was saved.
            self.tools_md = _tool_schemas_markdown()

            cfg = state.get("config", {}) if isinstance(state.get("config", {}), dict) else {}
            self.config.update(cfg)
            self.task_ids = self._list_task_ids_for_config(self.config)

            usage = state.get("usage", {}) if isinstance(state.get("usage", {}), dict) else {}
            request_ledger = usage.get("request_ledger", []) if isinstance(usage.get("request_ledger", []), list) else []
            if request_ledger:
                usage = _recalculate_usage_from_request_ledger(request_ledger)
            else:
                recomputed_usage = _recalculate_usage_from_history(self.history, self.config)
                if int(recomputed_usage.get("requests", 0) or 0) > 0:
                    usage = recomputed_usage

            self.handler.total_input_tokens = int(usage.get("input_tokens", 0) or 0)
            self.handler.total_output_tokens = int(usage.get("output_tokens", 0) or 0)
            self.handler.total_thinking_tokens = int(usage.get("thinking_tokens", 0) or 0)
            self.handler.total_billed_tokens = int(
                usage.get(
                    "total_tokens",
                    int(usage.get("input_tokens", 0) or 0)
                    + int(usage.get("output_tokens", 0) or 0)
                    + int(usage.get("thinking_tokens", 0) or 0),
                ) or 0
            )
            self.handler.total_requests = int(usage.get("requests", 0) or 0)
            self.handler.total_cost = float(usage.get("total_cost", 0.0) or 0.0)
            self.handler.total_input_cost = float(usage.get("input_cost", 0.0) or 0.0)
            self.handler.total_output_cost = float(usage.get("output_cost", 0.0) or 0.0)
            # Recompute input/output cost from ledger if not stored
            if self.handler.total_input_cost == 0 and self.handler.total_cost > 0:
                for entry in usage.get("request_ledger", []):
                    self.handler.total_input_cost += float(entry.get("input_cost_usd", 0.0) or 0.0)
                    self.handler.total_output_cost += float(entry.get("output_cost_usd", 0.0) or 0.0)
            self.handler.reasoning_breakdown_incomplete = bool(
                usage.get("reasoning_breakdown_incomplete", False)
            )
            self.handler.pricing_estimate_incomplete = bool(
                usage.get("pricing_estimate_incomplete", False)
            )
            self.handler.request_ledger = [
                copy.deepcopy(entry)
                for entry in usage.get("request_ledger", [])
                if isinstance(entry, dict)
            ]

            self._finalize_streams()
            self._turn_usage_target_idx = None
            self.partial_tool_call_indices.clear()
            self.reflector_context_idx = None
            self.current_turn = 0
            self.current_iteration = 0
            self._restore_runtime_state_from_history()

        if history_changed or backfill_changed:
            try:
                doc["state"]["history"] = self.history
                target.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to rewrite migrated saved run %s: %s", target, e)

        self._emit({"type": "snapshot", "state": self._snapshot()})
        return True, "Loaded saved run."

    async def _broadcast(self, payload: dict[str, Any]):
        stale: list[WebSocket] = []
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        if stale:
            with self.lock:
                for ws in stale:
                    self.clients.discard(ws)

    def _emit(self, payload: dict[str, Any]):
        if not self.loop or not self.loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)

    def _remove_history_entries(self, indices: list[int]):
        """Remove history entries at the given indices and tell clients to reload history."""
        indices_set = set(indices)
        with self.lock:
            self.history = [m for i, m in enumerate(self.history) if i not in indices_set]
            self._restore_runtime_state_from_history()
        # Full history resync is simplest since indices shift after removal
        self._emit({"type": "history_reset", "history": list(self.history)})

    def _remove_reflector_canvas(self) -> bool:
        """Remove the reflector canvas AND all subsequent canvases.

        When a reflector result is deleted, everything that followed it
        (post-reflector solver turns, later iterations, later reflectors)
        is invalid and must also be removed.
        """
        reflector_canvas_seq: int | None = None
        with self.lock:
            # Find the reflector canvas seq
            if self.reflector_context_idx is not None and 0 <= self.reflector_context_idx < len(self.history):
                reflector_canvas_seq = int(self.history[self.reflector_context_idx].get("canvas_seq", 0) or 0)
            if not reflector_canvas_seq:
                for msg in reversed(self.history):
                    if isinstance(msg, dict) and msg.get("canvas_owner") == "independent_reflector":
                        reflector_canvas_seq = int(msg.get("canvas_seq", 0) or 0)
                        if reflector_canvas_seq:
                            break

        if reflector_canvas_seq:
            with self.lock:
                # Remove all messages at or after the reflector canvas seq
                to_remove = [
                    idx for idx, msg in enumerate(self.history)
                    if isinstance(msg, dict)
                    and int(msg.get("canvas_seq", 0) or 0) >= reflector_canvas_seq
                ]
                # Also remove any turn_divider immediately before the first removed entry
                if to_remove:
                    first = min(to_remove)
                    if first > 0 and self.history[first - 1].get("kind") == "turn_divider":
                        to_remove.append(first - 1)
            if to_remove:
                self._remove_history_entries(to_remove)
                return True

        # Fallback: no canvas_seq — remove reflector_context message only
        to_remove = []
        with self.lock:
            rc_idx = self.reflector_context_idx
            if rc_idx is None:
                for i in range(len(self.history) - 1, -1, -1):
                    if self.history[i].get("kind") == "reflector_context":
                        rc_idx = i
                        break
            if rc_idx is not None:
                to_remove.append(rc_idx)
                if rc_idx > 0 and self.history[rc_idx - 1].get("kind") == "turn_divider":
                    to_remove.append(rc_idx - 1)
        if to_remove:
            self._remove_history_entries(to_remove)
            return True
        return False

    def _append_history(self, message: dict[str, Any]):
        message = self._stamp_with_current_canvas(self._normalize_history_message(message))
        with self.lock:
            self.history.append(message)
            idx = len(self.history) - 1
        self._emit({"type": "history_add", "message": message, "index": idx})
        return idx

    def _patch_history(self, idx: int, message: dict[str, Any], *, emit: bool = True):
        message = self._stamp_with_current_canvas(self._normalize_history_message(message))
        with self.lock:
            if idx < 0 or idx >= len(self.history):
                return
            self.history[idx] = message
        if emit:
            self._emit({"type": "history_patch", "index": idx, "message": message})

    def _append_delta_history(self, idx: int, delta: str):
        if idx < 0:
            return
        if not delta:
            return
        self._emit({"type": "history_append_delta", "index": idx, "delta": delta})

    def _update_panels(self, *, system: str | None = None, user: str | None = None, latest_tool: str | None = None, latest_code: str | None = None):
        updates: dict[str, Any] = {}
        with self.lock:
            if system is not None:
                self.system_prompt_md = system
                updates["system_prompt"] = system
            if user is not None:
                self.user_prompt_md = user
                updates["user_prompt"] = user
            if latest_tool is not None:
                self.latest_tool_md = latest_tool
                updates["latest_tool"] = latest_tool
            if latest_code is not None:
                self.latest_code_md = latest_code
                updates["latest_code"] = latest_code
        if updates:
            self._emit({"type": "panels_update", "updates": updates})

    def _sync_prompt_bundle_history(self):
        """Keep startup prompt bundle in Agent Conversation in sync with current panels."""
        bundle = {
            "role": "user",
            "kind": "prompt_bundle",
            "source": "orchestrator",
            "content": "",
            "system_md": self.system_prompt_md,
            "user_md": self.user_prompt_md,
        }

        with self.lock:
            existing_idx: int | None = None
            for i, msg in enumerate(self.history):
                if msg.get("kind") == "prompt_bundle":
                    existing_idx = i
                    break
            if existing_idx is None:
                # If no chat yet, seed with prompt bundle. Otherwise avoid duplicating during active runs.
                if not self.history:
                    self.history.append(bundle)
                    idx = len(self.history) - 1
                    add = True
                    patch = False
                else:
                    return
            else:
                self.history[existing_idx] = bundle
                idx = existing_idx
                add = False
                patch = True

        if add:
            self._emit({"type": "history_add", "message": bundle, "index": idx})
        elif patch:
            self._emit({"type": "history_patch", "index": idx, "message": bundle})

    def _restore_runtime_state_from_history(self) -> None:
        self._clear_current_canvas()
        max_canvas_seq = 0
        last_canvas_msg: dict[str, Any] | None = None
        self.reflector_context_idx = None
        self.current_iteration = 0
        self.current_turn = 0
        for idx, msg in enumerate(self.history):
            if not isinstance(msg, dict):
                continue
            if msg.get("kind") == "reflector_context":
                self.reflector_context_idx = idx
            turn_number = msg.get("turn_number")
            if isinstance(turn_number, (int, float)):
                self.current_turn = max(self.current_turn, int(turn_number))
            iteration = msg.get("iteration")
            if isinstance(iteration, (int, float)):
                self.current_iteration = max(self.current_iteration, int(iteration))
            canvas_seq = msg.get("canvas_seq")
            if isinstance(canvas_seq, (int, float)):
                max_canvas_seq = max(max_canvas_seq, int(canvas_seq))
            if msg.get("canvas_id"):
                last_canvas_msg = msg
        self.current_canvas_seq = max_canvas_seq
        if last_canvas_msg:
            self.current_canvas_id = str(last_canvas_msg.get("canvas_id") or "")
            self.current_canvas_label = str(last_canvas_msg.get("canvas_label") or "")
            self.current_canvas_owner = str(last_canvas_msg.get("canvas_owner") or "main_agent")
            self.current_canvas_phase = str(last_canvas_msg.get("canvas_phase") or "main")
            if isinstance(last_canvas_msg.get("iteration"), (int, float)):
                self.current_iteration = int(last_canvas_msg.get("iteration"))

    @staticmethod
    def _resolve_puzzle_path(puzzle_path: str, split: str = "public_eval", dataset_root: str | None = None) -> str:
        raw = str(puzzle_path or "").strip()
        if raw and not raw.endswith(".json") and "/" not in raw:
            try:
                return str(resolve_task_path(raw, split=split, dataset_root=dataset_root))
            except Exception:
                return raw
        return raw

    @staticmethod
    def _build_user_prompt_markdown(user_prompt_blocks: list[dict[str, Any]]) -> str:
        user_prompt_parts: list[str] = []
        img_idx = 0
        for block in user_prompt_blocks:
            btype = block.get("type")
            if btype == "text":
                user_prompt_parts.append(str(block.get("text", "")))
            elif btype == "image":
                img_data = str(block.get("source", {}).get("data", "") or "")
                if img_data:
                    img_idx += 1
                    user_prompt_parts.append(f"![User Prompt Image {img_idx}](data:image/png;base64,{img_data})")
        return "\n\n".join([p for p in user_prompt_parts if p is not None]).strip()

    def _refresh_prompt_panels_from_config(self, *, emit: bool):
        with self.lock:
            config_snapshot = dict(self.config)
        task_ids = self._list_task_ids_for_config(config_snapshot)
        with self.lock:
            self.task_ids = list(task_ids)

        system_prompt = ""
        user_prompt_md = ""
        puzzle_path = self._resolve_puzzle_path(
            str(config_snapshot.get("puzzle_path") or ""),
            split=str(config_snapshot.get("split") or "public_eval"),
            dataset_root=str(config_snapshot.get("dataset_root") or "") or None,
        )

        try:
            system_prompt = load_system_prompt()
        except Exception as e:
            system_prompt = f"⚠️ Failed to load system prompt:\n\n```text\n{e}\n```"

        if not puzzle_path:
            user_prompt_md = "*Provide a puzzle path or task ID in Configuration to load the user prompt.*"
        elif not Path(puzzle_path).exists():
            user_prompt_md = f"⚠️ Puzzle not found at:\n\n`{puzzle_path}`\n\nProvide a valid task ID or full path."
        else:
            try:
                puzzle_data = load_puzzle(puzzle_path)
                use_vision = not _is_glm_model(str(config_snapshot.get("model") or ""))
                user_prompt_blocks = format_puzzle_for_prompt(puzzle_data, use_vision=use_vision)
                user_prompt_blocks = _append_kimi_run_code_guidance_if_needed(
                    user_prompt_blocks,
                    str(config_snapshot.get("model") or "claude-opus-4-6"),
                )
                user_prompt_md = self._build_user_prompt_markdown(user_prompt_blocks)
            except Exception as e:
                user_prompt_md = f"⚠️ Failed to build user prompt:\n\n```text\n{e}\n```"

        self._update_panels(system=system_prompt, user=user_prompt_md, latest_tool=None, latest_code=None)
        self._sync_prompt_bundle_history()
        if emit:
            self._emit({"type": "config_update", "config": config_snapshot, "task_ids": task_ids})

    def _set_running(self, running: bool, status: str):
        with self.lock:
            self.running = running
            if running:
                self.paused = False
            self.status = status
        self._emit({"type": "run_state", "running": running, "paused": self.paused, "status": status})

    def _set_paused(self, paused: bool, status: str):
        with self.lock:
            self.running = False
            self.paused = paused
            self.status = status
        self._emit({"type": "run_state", "running": False, "paused": paused, "status": status})

    @staticmethod
    def _clean_content(content: str) -> str:
        return str(content or "")

    @staticmethod
    def _clean_tool_code(code: Any) -> str:
        raw = str(code or "")
        cleaned = raw.replace("\r\n", "\n").strip()

        # Drop a leading fenced header (``` or ```python etc)
        cleaned = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n?", "", cleaned, count=1)

        # Drop any trailing fence artifacts:
        # - ``` / ``` with whitespace/newline
        # - literal escaped newline before fence: \n```
        while True:
            new_cleaned = re.sub(r"(?:\\n|\n)?\s*```\s*$", "", cleaned, count=1)
            if new_cleaned == cleaned:
                break
            cleaned = new_cleaned.rstrip()

        # Last-line fence safety (when model emits stray closing fence line)
        lines = cleaned.split("\n")
        while lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).rstrip()
        while cleaned.endswith("\\n```"):
            cleaned = cleaned[: -len("\\n```")].rstrip()
        while cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
        return cleaned

    @staticmethod
    def _clean_tool_result_text(text: Any) -> str:
        raw = str(text or "")
        cleaned = raw.replace("\r\n", "\n").rstrip()
        while True:
            new_cleaned = re.sub(r"(?:\\n|\n)?\s*```\s*$", "", cleaned, count=1)
            if new_cleaned == cleaned:
                break
            cleaned = new_cleaned.rstrip()
        while cleaned.endswith("\\n```"):
            cleaned = cleaned[: -len("\\n```")].rstrip()
        while cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
        return cleaned

    @staticmethod
    def _extract_tool_result_content(content_blocks: Any) -> tuple[str, list[str], list[dict[str, Any]]]:
        """Extract content from tool result blocks.
        
        Returns:
            tuple of (combined_text, all_images, interleaved_blocks)
            - combined_text: all text joined (for backwards compat)
            - all_images: all images in order (for backwards compat)
            - interleaved_blocks: list of {"type": "text"/"image", "content": ...} in original order
        """
        text_parts: list[str] = []
        images: list[str] = []
        interleaved: list[dict[str, Any]] = []
        if not isinstance(content_blocks, list):
            return "", images, interleaved

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("text", "") or "")
                if text:
                    text_parts.append(text)
                    interleaved.append({"type": "text", "content": text})
                continue
            if btype == "image":
                source = block.get("source", {}) if isinstance(block.get("source"), dict) else {}
                data = str(source.get("data", "") or "")
                if not data:
                    continue
                media_type = str(source.get("media_type", "image/png") or "image/png")
                if data.startswith("data:image/"):
                    img_url = data
                else:
                    img_url = f"data:{media_type};base64,{data}"
                images.append(img_url)
                interleaved.append({"type": "image", "content": img_url})
        return "\n\n".join(text_parts).strip(), images, interleaved

    @staticmethod
    def _clone_jsonable(value: Any) -> Any:
        return copy.deepcopy(value)

    @staticmethod
    def _looks_like_legacy_snapshot_interleaved(blocks: Any) -> bool:
        if not isinstance(blocks, list):
            return False
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = str(block.get("content", "") or "").strip()
            if text in ("## User Message", "## Assistant Message"):
                return True
        return False

    @classmethod
    def _infer_iteration_context_mode(cls, message: dict[str, Any]) -> str:
        explicit = str(message.get("context_mode", "") or "").strip().lower()
        if explicit in {"handoff", "snapshot"}:
            return explicit
        if isinstance(message.get("snapshot_messages"), list) and message.get("snapshot_messages"):
            return "snapshot"
        if isinstance(message.get("prompt_blocks"), list) and message.get("prompt_blocks"):
            return "handoff"
        if cls._looks_like_legacy_snapshot_interleaved(message.get("interleaved_blocks")):
            return "snapshot"
        return "handoff"

    @staticmethod
    def _data_url_to_source(data_url: str) -> dict[str, Any] | None:
        match = _DATA_URL_RE.match(str(data_url or "").strip())
        if not match:
            return None
        return {
            "type": "base64",
            "media_type": match.group("media"),
            "data": match.group("data"),
        }

    @classmethod
    def _interleaved_blocks_to_api_content(cls, blocks: Any) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if not isinstance(blocks, list):
            return content
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = str(block.get("content", "") or "")
                if text.strip():
                    content.append({"type": "text", "text": text})
            elif btype == "image":
                source = cls._data_url_to_source(str(block.get("content", "") or ""))
                if source is not None:
                    content.append({"type": "image", "source": source})
        return content

    @classmethod
    def _normalize_history_message(cls, message: dict[str, Any]) -> dict[str, Any]:
        msg = dict(message or {})
        if msg.get("kind") != "iteration_context":
            return msg

        msg["context_mode"] = cls._infer_iteration_context_mode(msg)
        if isinstance(msg.get("resume_messages"), list):
            msg["resume_messages"] = cls._clone_jsonable(msg["resume_messages"])
        if isinstance(msg.get("snapshot_messages"), list):
            msg["snapshot_messages"] = cls._clone_jsonable(msg["snapshot_messages"])
        if isinstance(msg.get("prompt_blocks"), list):
            msg["prompt_blocks"] = cls._clone_jsonable(msg["prompt_blocks"])
        return msg

    @classmethod
    def _normalize_history_list(cls, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [cls._normalize_history_message(m) if isinstance(m, dict) else m for m in (history or [])]

    @staticmethod
    def _canvas_label_for(owner: str, phase: str) -> str:
        if owner == "independent_reflector":
            return "Independent Reflector"
        if phase == "post_reflector":
            return "Main Agent (post-reflector)"
        return "Main Agent"

    @staticmethod
    def _canvas_meta_from_label(context_label: str | None) -> tuple[str, str]:
        label = str(context_label or "").strip()
        if label == "Independent Reflector":
            return "independent_reflector", "reflector"
        if label == "Solver (post-reflector)":
            return "main_agent", "post_reflector"
        return "main_agent", "main"

    @staticmethod
    def _looks_like_post_reflector_title(message: dict[str, Any]) -> bool:
        title = str(message.get("ui_title", "") or "")
        return "Solver (post-reflector)" in title

    @staticmethod
    def _is_legacy_reflector_note_blocks(blocks: Any) -> bool:
        if not isinstance(blocks, list) or len(blocks) != 1:
            return False
        block = blocks[0]
        if not isinstance(block, dict) or block.get("type") != "text":
            return False
        text = str(block.get("content", "") or "").strip()
        return text.startswith("Legacy checkpoint note:")

    @classmethod
    def _extract_legacy_reflector_feedback(cls, blocks: Any) -> dict[str, Any] | None:
        if not isinstance(blocks, list):
            return None
        for block in blocks:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = str(block.get("content", "") or "")
            if "Independent Reflector Verdict:" not in text:
                continue
            marker = "An independent reviewer has carefully studied your solution and identified concerns."
            start_idx = text.find(marker)
            snippet = text[start_idx:] if start_idx >= 0 else text
            end_marker = "\n\nThe reviewer's concerns suggest your current approach may not generalize correctly to test inputs."
            end_idx = snippet.find(end_marker)
            if end_idx >= 0:
                snippet = snippet[:end_idx].rstrip()
            verdict_match = re.search(r"Independent Reflector Verdict:\s*([A-Z]+)", snippet)
            confidence_match = re.search(r"confidence:\s*([0-9]+)\s*/\s*5", snippet, re.IGNORECASE)
            verdict = verdict_match.group(1).upper() if verdict_match else ""
            confidence = int(confidence_match.group(1)) if confidence_match else None
            if not snippet.strip():
                return None
            return {
                "response": snippet.strip(),
                "verdict": verdict or "REJECT",
                "confidence": confidence,
            }
        return None

    @staticmethod
    def _sanitize_loaded_reflector_analysis(text: str) -> str:
        structured_headers = (
            "PHASE_1_SCORE:",
            "PHASE_2_SCORE:",
            "PHASE_3_SCORE:",
            "CONFIDENCE:",
            "VERDICT:",
            "FINDINGS:",
            "CONCERNS:",
            "REJECTION_REASON:",
            "ROOT_CAUSE:",
            "SUGGESTED_FOCUS:",
            "REPAIR_PLAN:",
            "VALIDATION_CHECKS:",
        )
        retained_lines: list[str] = []
        for line in str(text or "").splitlines():
            normalized = line.strip().replace("**", "").strip().upper()
            if normalized.startswith(structured_headers):
                break
            retained_lines.append(line)
        while retained_lines and retained_lines[-1].strip() in {"", "```", "---", "***"}:
            retained_lines.pop()
        return "\n".join(retained_lines).strip()

    @classmethod
    def _repair_loaded_reflector_handoff_text(cls, text: str) -> str:
        raw = str(text or "")
        marker = "### Reviewer's Full Analysis"
        trailer = "\n---\n\nTreat this reject as final"
        if "Independent Reflector Verdict: REJECT" not in raw or marker not in raw or trailer not in raw:
            return raw
        prefix, remainder = raw.split(marker, 1)
        analysis, suffix = remainder.split(trailer, 1)
        sanitized_analysis = cls._sanitize_loaded_reflector_analysis(analysis)
        repaired = prefix.rstrip()
        if sanitized_analysis:
            repaired = f"{repaired}\n\n{marker}\n{sanitized_analysis}"
        return f"{repaired}{trailer}{suffix}"

    @staticmethod
    def _looks_like_orchestrator_completion_message(message: dict[str, Any]) -> bool:
        if not isinstance(message, dict):
            return False
        if str(message.get("kind", "") or "") != "assistant_response":
            return False
        if str(message.get("source", "") or "") != "orchestrator":
            return False
        content = str(message.get("content", "") or "").strip()
        return content.startswith("🏁 **") or content.startswith("**Task fully solved") or content.startswith("**No solution") or content.startswith("**Solution found") or content.startswith("**Accepted candidate")

    @classmethod
    def _strip_loaded_midrun_completion_messages(
        cls,
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        if not history:
            return history, False

        kept: list[dict[str, Any]] = []
        changed = False
        total = len(history)
        for idx, msg in enumerate(history):
            if cls._looks_like_orchestrator_completion_message(msg):
                has_later_substantive = any(
                    isinstance(later, dict)
                    and str(later.get("kind", "") or "") not in {"turn_divider", "prompt_bundle"}
                    for later in history[idx + 1 : total]
                )
                if has_later_substantive:
                    changed = True
                    continue
            kept.append(msg)
        return kept, changed

    @classmethod
    def _strip_trailing_completion_messages_for_resume(
        cls,
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        trimmed = list(history)
        changed = False
        while trimmed and cls._looks_like_orchestrator_completion_message(trimmed[-1]):
            trimmed.pop()
            changed = True
            while trimmed and isinstance(trimmed[-1], dict) and str(trimmed[-1].get("kind", "") or "") == "turn_divider":
                trimmed.pop()

        # If a reflector verdict (EXPAND_CANDIDATES / REJECT) was not properly
        # acted on — i.e. the blocks after the reflector show a self-reflection
        # prompt or solver responses instead of expansion/rejection feedback —
        # strip everything after the reflector so the orchestrator can
        # re-process the verdict correctly on resume.
        # Find the last reflector_context
        for i in range(len(trimmed) - 1, -1, -1):
            block = trimmed[i]
            if block.get("kind") == "reflector_context":
                verdict = str(block.get("verdict", "") or "").upper()
                if verdict in ("EXPAND_CANDIDATES", "REJECT"):
                    # Check if the next substantive block is proper
                    # expansion/rejection feedback (iteration_context with
                    # post_reflector canvas) or something else (mishandled).
                    for j in range(i + 1, len(trimmed)):
                        jk = str(trimmed[j].get("kind", "") or "")
                        if jk == "turn_divider":
                            continue
                        if jk == "iteration_context":
                            # Proper post-reflector iteration — leave intact
                            break
                        # Anything else (reflection_prompt, thinking,
                        # assistant_response, etc.) means the verdict was
                        # not handled correctly → trim back to reflector
                        trimmed = trimmed[:i + 1]
                        changed = True
                        break
                break

        return trimmed, changed

    @staticmethod
    def _looks_like_train_failure_reflection_prompt(message: dict[str, Any]) -> bool:
        if not isinstance(message, dict):
            return False
        if str(message.get("kind", "") or "") != "reflection_prompt":
            return False
        content = str(message.get("content", "") or "").strip()
        return content.startswith("The solution failed on some training examples.")

    @classmethod
    def _strip_loaded_post_reflector_train_failure_prompt(
        cls,
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        if not history:
            return history, False

        repaired: list[dict[str, Any]] = []
        changed = False
        for msg in history:
            if (
                cls._looks_like_train_failure_reflection_prompt(msg)
                and repaired
                and isinstance(repaired[-1], dict)
                and str(repaired[-1].get("kind", "") or "") == "iteration_context"
                and (
                    str(repaired[-1].get("canvas_phase", "") or "") == "post_reflector"
                    or cls._looks_like_post_reflector_title(repaired[-1])
                )
            ):
                changed = True
                continue
            repaired.append(msg)
        return repaired, changed

    @classmethod
    def _backfill_loaded_history(
        cls,
        history: list[dict[str, Any]],
        *,
        main_system_md: str = "",
    ) -> tuple[list[dict[str, Any]], bool]:
        normalized = cls._normalize_history_list(history)
        if not normalized:
            return normalized, False

        changed = False
        backfilled: list[dict[str, Any]] = []
        for raw_msg in normalized:
            if not isinstance(raw_msg, dict):
                backfilled.append(raw_msg)
                continue
            msg = dict(raw_msg)
            kind = str(msg.get("kind", "") or "")
            if kind == "reflector_context" and cls._is_legacy_reflector_note_blocks(msg.get("interleaved_blocks")):
                msg.pop("interleaved_blocks", None)
                changed = True
            if kind == "reflector_context":
                for key, default in (
                    ("rejection_reason", ""),
                    ("root_cause", []),
                    ("suggested_focus", ""),
                    ("repair_plan", []),
                    ("validation_checks", []),
                ):
                    if key not in msg:
                        msg[key] = copy.deepcopy(default)
                        changed = True
                if str(msg.get("response", "") or "").strip():
                    parsed = parse_reflector_response(
                        str(msg.get("response", "") or ""),
                        str(msg.get("thinking", "") or ""),
                        None,
                    )
                    if str(msg.get("verdict", "") or "").upper() in {"", "UNKNOWN"} and parsed.get("verdict") not in {"", "UNKNOWN"}:
                        msg["verdict"] = parsed["verdict"]
                        changed = True
                    if not msg.get("confidence") and parsed.get("confidence"):
                        msg["confidence"] = parsed["confidence"]
                        changed = True
                    phase_scores = msg.get("phase_scores")
                    if (
                        not isinstance(phase_scores, dict)
                        or not any(int(phase_scores.get(key, 0) or 0) for key in ("phase_1", "phase_2", "phase_3"))
                    ) and any(int(parsed.get("phase_scores", {}).get(key, 0) or 0) for key in ("phase_1", "phase_2", "phase_3")):
                        msg["phase_scores"] = copy.deepcopy(parsed.get("phase_scores", {}))
                        changed = True
                    if not msg.get("findings") and parsed.get("findings"):
                        msg["findings"] = copy.deepcopy(parsed["findings"])
                        changed = True
                    if not msg.get("concerns") and parsed.get("concerns"):
                        msg["concerns"] = copy.deepcopy(parsed["concerns"])
                        changed = True
                    for key, default in (
                        ("rejection_reason", ""),
                        ("root_cause", []),
                        ("suggested_focus", ""),
                        ("repair_plan", []),
                        ("validation_checks", []),
                    ):
                        existing = msg.get(key)
                        if existing:
                            continue
                        parsed_value = parsed.get(key)
                        if parsed_value:
                            msg[key] = copy.deepcopy(parsed_value)
                            changed = True
                        elif key not in msg:
                            msg[key] = copy.deepcopy(default)
                            changed = True
            if (
                kind == "iteration_context"
                and str(msg.get("canvas_owner", "") or "") == "main_agent"
                and str(msg.get("canvas_phase", "") or "") == "post_reflector"
                and not str(msg.get("system_md", "") or "").strip()
                and main_system_md.strip()
            ):
                msg["system_md"] = main_system_md
                changed = True
            if isinstance(msg.get("interleaved_blocks"), list):
                repaired_blocks: list[dict[str, Any]] = []
                for block in msg["interleaved_blocks"]:
                    if not isinstance(block, dict):
                        repaired_blocks.append(block)
                        continue
                    repaired_block = dict(block)
                    for key in ("text", "content"):
                        if isinstance(repaired_block.get(key), str):
                            repaired_text = cls._repair_loaded_reflector_handoff_text(repaired_block[key])
                            if repaired_text != repaired_block[key]:
                                repaired_block[key] = repaired_text
                                changed = True
                    repaired_blocks.append(repaired_block)
                msg["interleaved_blocks"] = repaired_blocks
            for key in ("content", "text"):
                if isinstance(msg.get(key), str):
                    repaired_text = cls._repair_loaded_reflector_handoff_text(msg[key])
                    if repaired_text != msg[key]:
                        msg[key] = repaired_text
                        changed = True
            if isinstance(msg.get("message_content"), dict):
                message_content = dict(msg["message_content"])
                if isinstance(message_content.get("text"), str):
                    repaired_text = cls._repair_loaded_reflector_handoff_text(message_content["text"])
                    if repaired_text != message_content["text"]:
                        message_content["text"] = repaired_text
                        msg["message_content"] = message_content
                        changed = True
            backfilled.append(msg)
        stripped_history, strip_changed = cls._strip_loaded_post_reflector_train_failure_prompt(backfilled)
        stripped_history, completion_strip_changed = cls._strip_loaded_midrun_completion_messages(stripped_history)
        return stripped_history, (changed or strip_changed or completion_strip_changed)

    def _set_current_canvas(
        self,
        *,
        iteration: int,
        context_label: str | None = None,
        force_new: bool = False,
    ) -> dict[str, Any]:
        owner, phase = self._canvas_meta_from_label(context_label)
        if (
            not str(context_label or "").strip()
            and self.current_canvas_id is not None
            and int(iteration or 0) == int(self.current_iteration or 0)
            and self.current_canvas_owner == "main_agent"
            and self.current_canvas_phase == "post_reflector"
        ):
            owner = self.current_canvas_owner
            phase = self.current_canvas_phase
        label = self._canvas_label_for(owner, phase)
        needs_new = force_new or self.current_canvas_id is None
        if not needs_new:
            if iteration != self.current_iteration:
                needs_new = True
            elif owner != self.current_canvas_owner or phase != self.current_canvas_phase:
                needs_new = True
        existing_canvas = self._find_canvas_metadata(iteration=iteration, owner=owner, phase=phase)
        if existing_canvas and (
            needs_new
            or self.current_canvas_id != str(existing_canvas.get("canvas_id", "") or "")
        ):
            self.current_canvas_id = str(existing_canvas["canvas_id"])
            self.current_canvas_seq = max(
                int(self.current_canvas_seq or 0),
                int(existing_canvas.get("canvas_seq", 0) or 0),
            )
            self.current_canvas_label = str(existing_canvas.get("canvas_label", "") or label)
            self.current_canvas_owner = owner
            self.current_canvas_phase = phase
            return self._current_canvas_metadata()
        if needs_new:
            self.current_canvas_seq = max(
                int(self.current_canvas_seq or 0),
                self._max_canvas_seq_in_history(),
            ) + 1
            self.current_canvas_id = f"canvas-{self.current_canvas_seq:04d}-iter-{int(iteration)}-{owner}-{phase}"
        self.current_canvas_label = label
        self.current_canvas_owner = owner
        self.current_canvas_phase = phase
        return self._current_canvas_metadata()

    def _clear_current_canvas(self) -> None:
        self.current_canvas_id = None
        self.current_canvas_label = ""
        self.current_canvas_owner = "main_agent"
        self.current_canvas_phase = "main"

    def _current_canvas_metadata(self) -> dict[str, Any]:
        if not self.current_canvas_id:
            return {}
        return {
            "canvas_id": self.current_canvas_id,
            "canvas_seq": int(self.current_canvas_seq),
            "canvas_label": self.current_canvas_label,
            "canvas_owner": self.current_canvas_owner,
            "canvas_phase": self.current_canvas_phase,
        }

    def _max_canvas_seq_in_history(self) -> int:
        with self.lock:
            max_seq = 0
            for msg in self.history:
                if not isinstance(msg, dict):
                    continue
                max_seq = max(max_seq, int(msg.get("canvas_seq", 0) or 0))
        return max_seq

    def _find_canvas_metadata(
        self,
        *,
        iteration: int,
        owner: str,
        phase: str,
    ) -> dict[str, Any] | None:
        with self.lock:
            for msg in reversed(self.history):
                if not isinstance(msg, dict):
                    continue
                if int(msg.get("iteration", -1) or -1) != int(iteration):
                    continue
                if str(msg.get("canvas_owner", "") or "") != owner:
                    continue
                if str(msg.get("canvas_phase", "") or "") != phase:
                    continue
                canvas_id = str(msg.get("canvas_id", "") or "")
                if not canvas_id:
                    continue
                return {
                    "canvas_id": canvas_id,
                    "canvas_seq": int(msg.get("canvas_seq", 0) or 0),
                    "canvas_label": str(msg.get("canvas_label", "") or self._canvas_label_for(owner, phase)),
                    "canvas_owner": owner,
                    "canvas_phase": phase,
                }
        return None

    def _reserve_canvas_metadata(
        self,
        *,
        iteration: int,
        owner: str,
        phase: str,
    ) -> dict[str, Any]:
        existing = self._find_canvas_metadata(iteration=iteration, owner=owner, phase=phase)
        if existing:
            return existing
        seq = max(int(self.current_canvas_seq or 0), self._max_canvas_seq_in_history()) + 1
        return {
            "canvas_id": f"canvas-{seq:04d}-iter-{int(iteration)}-{owner}-{phase}",
            "canvas_seq": seq,
            "canvas_label": self._canvas_label_for(owner, phase),
            "canvas_owner": owner,
            "canvas_phase": phase,
        }

    def _stamp_with_current_canvas(self, message: dict[str, Any]) -> dict[str, Any]:
        msg = dict(message or {})
        if msg.get("kind") == "prompt_bundle":
            return msg
        canvas_meta = self._current_canvas_metadata()
        for key, value in canvas_meta.items():
            msg.setdefault(key, value)
        return msg

    @staticmethod
    def _is_empty_text(value: Any) -> bool:
        return not str(value or "").strip()

    @classmethod
    def _is_empty_iteration_context(cls, message: dict[str, Any]) -> bool:
        if str(message.get("kind", "")) != "iteration_context":
            return False
        if isinstance(message.get("snapshot_messages"), list) and message.get("snapshot_messages"):
            return False
        if isinstance(message.get("prompt_blocks"), list) and message.get("prompt_blocks"):
            return False
        if isinstance(message.get("interleaved_blocks"), list) and message.get("interleaved_blocks"):
            return False
        if isinstance(message.get("images"), list) and message.get("images"):
            return False
        return cls._is_empty_text(message.get("content"))

    @staticmethod
    def _looks_like_reflector_title(message: dict[str, Any]) -> bool:
        title = str(message.get("ui_title", "") or "")
        return "Independent Reflector" in title

    @classmethod
    def _migrate_history_canvases(
        cls,
        history: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        migrated = cls._normalize_history_list(history)
        if not migrated:
            return migrated, False

        changed = False
        canvas_seq = 0
        current_canvas: dict[str, Any] | None = None
        current_iteration = 0
        known_reflector_system_md = ""
        for msg in migrated:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("kind", "")) == "reflector_context":
                system_md = str(msg.get("system_md", "") or "")
                if system_md.strip():
                    known_reflector_system_md = system_md
                    break

        def start_canvas(iteration: int, owner: str, phase: str) -> dict[str, Any]:
            nonlocal canvas_seq, current_canvas, current_iteration
            canvas_seq += 1
            current_iteration = int(iteration or 0)
            current_canvas = {
                "canvas_id": f"canvas-{canvas_seq:04d}-iter-{current_iteration}-{owner}-{phase}",
                "canvas_seq": canvas_seq,
                "canvas_label": cls._canvas_label_for(owner, phase),
                "canvas_owner": owner,
                "canvas_phase": phase,
            }
            return dict(current_canvas)

        def ensure_canvas(iteration: int, owner: str, phase: str) -> dict[str, Any]:
            nonlocal current_canvas, current_iteration
            if (
                current_canvas is None
                or current_iteration != int(iteration or 0)
                or current_canvas.get("canvas_owner") != owner
                or current_canvas.get("canvas_phase") != phase
            ):
                return start_canvas(iteration, owner, phase)
            return dict(current_canvas)

        stamped: list[dict[str, Any]] = []
        for raw_msg in migrated:
            if not isinstance(raw_msg, dict):
                stamped.append(raw_msg)
                continue
            msg = dict(raw_msg)
            kind = str(msg.get("kind", "") or "")
            iteration = int(msg.get("iteration", current_iteration) or current_iteration or 0)
            context_label = str(msg.get("context_label", "") or "").strip()

            if kind == "prompt_bundle":
                stamped.append(msg)
                continue

            if kind == "turn_divider":
                if context_label == "Independent Reflector":
                    canvas = start_canvas(iteration, "independent_reflector", "reflector")
                elif context_label == "Solver (post-reflector)":
                    canvas = start_canvas(iteration, "main_agent", "post_reflector")
                elif current_canvas and current_canvas.get("canvas_phase") == "post_reflector" and current_iteration == iteration:
                    canvas = dict(current_canvas)
                else:
                    canvas = ensure_canvas(iteration, "main_agent", "main")
                msg.update(canvas)
                stamped.append(msg)
                continue

            if kind == "reflector_context":
                canvas = ensure_canvas(iteration, "independent_reflector", "reflector")
                msg.update(canvas)
                stamped.append(msg)
                continue

            if kind == "iteration_context":
                mode = cls._infer_iteration_context_mode(msg)
                if mode == "handoff":
                    canvas = ensure_canvas(iteration, "main_agent", "main")
                    msg.update(canvas)
                    stamped.append(msg)
                    continue
                if mode == "snapshot" and not cls._looks_like_post_reflector_title(msg):
                    msg.pop("system_md", None)
                    changed = True
                if cls._looks_like_post_reflector_title(msg):
                    feedback = cls._extract_legacy_reflector_feedback(msg.get("interleaved_blocks"))
                    has_prior_reflector = bool(current_canvas and current_canvas.get("canvas_owner") == "independent_reflector")
                    if feedback and not has_prior_reflector:
                        changed = True
                        reflector_canvas = start_canvas(iteration, "independent_reflector", "reflector")
                        stamped.append(
                            {
                                "role": "assistant",
                                "kind": "turn_divider",
                                "turn_number": None,
                                "iteration": iteration,
                                "content": "",
                                "context_label": "Independent Reflector",
                                **reflector_canvas,
                            }
                        )
                        stamped.append(
                            {
                                "role": "user",
                                "kind": "reflector_context",
                                "source": "orchestrator",
                                "content": "",
                                "iteration": iteration,
                                "reflector_provider": "",
                                "reflector_model": "",
                                "system_md": known_reflector_system_md,
                                "interleaved_blocks": [],
                                "verdict": feedback.get("verdict") or "REJECT",
                                "confidence": feedback.get("confidence"),
                                "phase_scores": {},
                                "thinking": "",
                                "response": feedback.get("response", ""),
                                "findings": [],
                                "concerns": [],
                                "rejection_reason": "",
                                "root_cause": [],
                                "suggested_focus": "",
                                "repair_plan": [],
                                "validation_checks": [],
                                "status": "done",
                                **reflector_canvas,
                            }
                        )
                    canvas = ensure_canvas(iteration, "main_agent", "post_reflector")
                    msg.update(canvas)
                    stamped.append(msg)
                    continue
                if cls._looks_like_reflector_title(msg) and cls._is_empty_iteration_context(msg):
                    canvas = ensure_canvas(iteration, "independent_reflector", "reflector")
                    msg.update(canvas)
                    stamped.append(msg)
                    continue
                if mode == "snapshot" and current_canvas and current_canvas.get("canvas_owner") == "independent_reflector":
                    canvas = start_canvas(iteration, "main_agent", "post_reflector")
                else:
                    owner = "main_agent"
                    phase = "main"
                    if (
                        current_canvas
                        and current_canvas.get("canvas_phase") == "post_reflector"
                        and current_iteration == iteration
                    ):
                        phase = "post_reflector"
                    canvas = ensure_canvas(iteration, owner, phase)
                msg.update(canvas)
                stamped.append(msg)
                continue

            if current_canvas is None:
                current_canvas = start_canvas(iteration, "main_agent", "main")
            msg.update(current_canvas)
            stamped.append(msg)

        canvas_entries: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for idx, msg in enumerate(stamped):
            if not isinstance(msg, dict):
                continue
            canvas_id = str(msg.get("canvas_id", "") or "")
            if not canvas_id:
                continue
            canvas_entries.setdefault(canvas_id, []).append((idx, msg))

        remove_indices: set[int] = set()
        for canvas_id, entries in canvas_entries.items():
            owner = str(entries[0][1].get("canvas_owner", "") or "")
            has_reflector_context = any(str(msg.get("kind", "")) == "reflector_context" for _, msg in entries)
            if owner == "independent_reflector" and not has_reflector_context:
                only_placeholder = all(
                    str(msg.get("kind", "")) == "turn_divider"
                    or (
                        str(msg.get("kind", "")) == "iteration_context"
                        and cls._looks_like_reflector_title(msg)
                        and cls._is_empty_iteration_context(msg)
                    )
                    for _, msg in entries
                )
                if only_placeholder:
                    changed = True
                    for idx, _ in entries:
                        remove_indices.add(idx)

        filtered: list[dict[str, Any]] = []
        for idx, msg in enumerate(stamped):
            if idx in remove_indices:
                continue
            if isinstance(msg, dict) and cls._looks_like_reflector_title(msg) and cls._is_empty_iteration_context(msg):
                changed = True
                continue
            filtered.append(msg)

        seq_by_canvas: dict[str, int] = {}
        next_seq = 0
        for msg in filtered:
            if not isinstance(msg, dict):
                continue
            canvas_id = str(msg.get("canvas_id", "") or "")
            if not canvas_id:
                continue
            if canvas_id not in seq_by_canvas:
                next_seq += 1
                seq_by_canvas[canvas_id] = next_seq
            if int(msg.get("canvas_seq", 0) or 0) != seq_by_canvas[canvas_id]:
                msg["canvas_seq"] = seq_by_canvas[canvas_id]
                changed = True

        return filtered, changed

    @staticmethod
    def _decode_jsonish_string(value: str, *, closed: bool) -> str:
        """Decode a JSON string fragment safely, including partial fragments."""
        text = value
        if closed:
            try:
                return str(json.loads(f'"{value}"'))
            except Exception:
                pass
        # Best-effort decode for partial chunks.
        text = text.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
        text = text.replace('\\"', '"').replace("\\\\", "\\")
        return text

    @classmethod
    def _extract_partial_json_string_field(cls, raw: str, field: str) -> str | None:
        needle = f'"{field}"'
        i = raw.find(needle)
        if i < 0:
            return None

        colon = raw.find(":", i + len(needle))
        if colon < 0:
            return None

        j = colon + 1
        raw_len = len(raw)
        while j < raw_len and raw[j].isspace():
            j += 1
        if j >= raw_len or raw[j] != '"':
            return None

        j += 1
        escaped = False
        buf: list[str] = []
        closed = False
        while j < raw_len:
            ch = raw[j]
            if escaped:
                buf.append(ch)
                escaped = False
                j += 1
                continue
            if ch == "\\":
                escaped = True
                buf.append(ch)
                j += 1
                continue
            if ch == '"':
                closed = True
                break
            buf.append(ch)
            j += 1
        return cls._decode_jsonish_string("".join(buf), closed=closed)

    @classmethod
    def _extract_partial_tool_fields(cls, tool_name: str, input_raw: str) -> dict[str, Any]:
        fields: dict[str, Any] = {}

        # If raw JSON is complete, prefer exact parsing.
        try:
            parsed = json.loads(input_raw)
            if isinstance(parsed, dict):
                fields.update(cls._extract_tool_display_fields(tool_name, parsed))
                return fields
        except Exception:
            pass

        # Otherwise extract JSON-string fields incrementally from partial chunks.
        if tool_name in {"run_code", "run_code_in_previous_runtime"}:
            code = cls._extract_partial_json_string_field(input_raw, "code")
            if code is not None:
                fields["tool_code"] = cls._clean_tool_code(code)
        elif tool_name == "submit_transform_hypothesis":
            hypothesis = cls._extract_partial_json_string_field(input_raw, "hypothesis")
            branch_id = cls._extract_partial_json_string_field(input_raw, "branch_id")
            branch_hypothesis = cls._extract_partial_json_string_field(input_raw, "branch_hypothesis")
            branch_a = cls._extract_partial_json_string_field(input_raw, "branch_a_hypothesis")
            branch_b = cls._extract_partial_json_string_field(input_raw, "branch_b_hypothesis")
            ambiguity = cls._extract_partial_json_string_field(input_raw, "ambiguity_rationale")
            updated_failed = cls._extract_partial_json_string_field(input_raw, "updated_failed_hypothesis")
            updated_ambiguity = cls._extract_partial_json_string_field(input_raw, "updated_ambiguity_rationale")
            withdraw_match = re.search(r'"withdraw_failed_branch"\s*:\s*(true|false)', input_raw, flags=re.IGNORECASE)
            if hypothesis is not None:
                fields["tool_transform"] = str(hypothesis)
            if branch_id is not None:
                fields["tool_target_branch_id"] = str(branch_id)
            if branch_hypothesis is not None:
                fields["tool_target_branch_transform"] = str(branch_hypothesis)
            if branch_a is not None:
                fields["tool_transform"] = str(branch_a)
            if branch_b is not None:
                fields["tool_secondary_transform"] = str(branch_b)
            if ambiguity is not None:
                fields["tool_ambiguity_rationale"] = str(ambiguity)
            if updated_failed is not None:
                fields["tool_updated_failed_transform"] = str(updated_failed)
            if updated_ambiguity is not None:
                fields["tool_updated_ambiguity_rationale"] = str(updated_ambiguity)
            if withdraw_match:
                fields["tool_withdraw_failed_branch"] = withdraw_match.group(1).lower() == "true"
        elif tool_name == "execute_python_solution":
            code = cls._extract_partial_json_string_field(input_raw, "code")
            branch_id = cls._extract_partial_json_string_field(input_raw, "branch_id")
            branch_code = cls._extract_partial_json_string_field(input_raw, "branch_code")
            branch_a_code = cls._extract_partial_json_string_field(input_raw, "branch_a_code")
            branch_b_code = cls._extract_partial_json_string_field(input_raw, "branch_b_code")
            updated_failed_code = cls._extract_partial_json_string_field(input_raw, "updated_failed_code")
            if code is not None:
                fields["tool_code"] = cls._clean_tool_code(code)
            if branch_id is not None:
                fields["tool_target_branch_id"] = str(branch_id)
            if branch_code is not None:
                fields["tool_target_branch_code"] = cls._clean_tool_code(branch_code)
            if branch_a_code is not None:
                fields["tool_code"] = cls._clean_tool_code(branch_a_code)
            if branch_b_code is not None:
                fields["tool_secondary_code"] = cls._clean_tool_code(branch_b_code)
            if updated_failed_code is not None:
                fields["tool_updated_failed_code"] = cls._clean_tool_code(updated_failed_code)
        return fields

    @classmethod
    def _extract_tool_display_fields(cls, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if tool_name in {"run_code", "run_code_in_previous_runtime"}:
            fields["tool_code"] = cls._clean_tool_code(tool_input.get("code", ""))
        elif tool_name == "submit_transform_hypothesis":
            hypothesis = tool_input.get("hypothesis", "")
            branch_id = tool_input.get("branch_id", "")
            branch_hypothesis = tool_input.get("branch_hypothesis", "")
            primary = tool_input.get("branch_a_hypothesis", "")
            secondary = tool_input.get("branch_b_hypothesis", "")
            ambiguity = tool_input.get("ambiguity_rationale", "")
            updated_failed = tool_input.get("updated_failed_hypothesis", "")
            updated_ambiguity = tool_input.get("updated_ambiguity_rationale", "")
            if hypothesis:
                fields["tool_transform"] = str(hypothesis)
            if branch_id:
                fields["tool_target_branch_id"] = str(branch_id)
            if branch_hypothesis:
                fields["tool_target_branch_transform"] = str(branch_hypothesis)
            fields["tool_transform"] = str(fields.get("tool_transform", "") or primary or "")
            if secondary:
                fields["tool_secondary_transform"] = str(secondary)
            if ambiguity:
                fields["tool_ambiguity_rationale"] = str(ambiguity)
            if updated_failed:
                fields["tool_updated_failed_transform"] = str(updated_failed)
            if updated_ambiguity:
                fields["tool_updated_ambiguity_rationale"] = str(updated_ambiguity)
            if tool_input.get("withdraw_failed_branch") is True:
                fields["tool_withdraw_failed_branch"] = True
        elif tool_name == "execute_python_solution":
            code = tool_input.get("code", "")
            target_branch_id = tool_input.get("branch_id", "")
            target_branch_code = tool_input.get("branch_code", "")
            primary_code = tool_input.get("branch_a_code", "")
            secondary_code = tool_input.get("branch_b_code", "")
            updated_failed_code = tool_input.get("updated_failed_code", "")
            if code:
                fields["tool_code"] = cls._clean_tool_code(code)
            if target_branch_id:
                fields["tool_target_branch_id"] = str(target_branch_id)
            if target_branch_code:
                fields["tool_target_branch_code"] = cls._clean_tool_code(target_branch_code)
            fields["tool_code"] = cls._clean_tool_code(fields.get("tool_code", "") or primary_code)
            if secondary_code:
                fields["tool_secondary_code"] = cls._clean_tool_code(secondary_code)
            if updated_failed_code:
                fields["tool_updated_failed_code"] = cls._clean_tool_code(updated_failed_code)
        return fields

    def _finalize_streams(self):
        self.current_thinking = ""
        self.current_text = ""
        self.current_thinking_idx = None
        self.current_text_idx = None

    def _consume_event(self, event: OrchestratorEvent):
        # Suppress verbose SYSTEM blocks in chat.
        if event.type == EventType.SYSTEM:
            metadata = event.metadata if isinstance(event.metadata, dict) else {}
            usage_meta = metadata.get("_token_usage", {})
            if usage_meta:
                self._attach_usage_to_latest_agent_message(dict(usage_meta))
                self._emit({"type": "usage_update", "usage": self._usage_snapshot()})
            # Capture thinking signature for checkpointing
            sig = metadata.get("_thinking_signature")
            if sig and self.current_thinking_idx is not None:
                with self.lock:
                    msg = dict(self.history[self.current_thinking_idx])
                msg["thinking_signature"] = sig
                self._patch_history(self.current_thinking_idx, msg, emit=False)
            # Remove reflector canvas (turn_divider + reflector_context) when reflector errors out
            if metadata.get("_remove_reflector_canvas"):
                self._remove_reflector_canvas()
                return
            # Visible system message — render as orchestrator message in chat
            if metadata.get("_visible_message"):
                self._append_history({
                    "role": "user",
                    "kind": "assistant_response",
                    "source": "orchestrator",
                    "content": str(event.content or ""),
                })
                return
            # Handle reflector prompt bundle — no canvas_id so it behaves
            # like the initial prompt_bundle (filtered by iteration match)
            if metadata.get("_reflector_prompt_bundle"):
                bundle = {
                    "role": "user",
                    "kind": "prompt_bundle",
                    "source": "orchestrator",
                    "content": "",
                    "system_md": metadata.get("system_md", ""),
                    "user_md": metadata.get("user_md", ""),
                    "iteration": metadata.get("iteration", 0),
                }
                self._append_history(bundle)
            return

        if event.type == EventType.TURN_START:
            self._finalize_streams()
            self._turn_usage_target_idx = None
            self.current_turn += 1
            next_iteration = int(event.metadata.get("iteration", 0) if event.metadata else 0)
            ctx_label = (event.metadata or {}).get("context_label")
            canvas_meta = self._set_current_canvas(iteration=next_iteration, context_label=ctx_label)
            self.current_iteration = next_iteration
            divider: dict[str, Any] = {
                "role": "assistant",
                "kind": "turn_divider",
                "turn_number": self.current_turn,
                "iteration": self.current_iteration,
                "content": "",
            }
            if ctx_label:
                divider["context_label"] = ctx_label
            divider.update(canvas_meta)
            self._append_history(divider)
            return

        if event.type == EventType.THINKING:
            metadata = event.metadata if isinstance(event.metadata, dict) else {}
            is_reflector = metadata.get("reflector", False)
            
            if is_reflector and self.reflector_context_idx is not None:
                # Update reflector_context message's thinking field
                incoming = self._clean_content(event.content)
                with self.lock:
                    msg = dict(self.history[self.reflector_context_idx])
                current_thinking = msg.get("thinking", "")
                msg["thinking"] = current_thinking + incoming
                self._patch_history(self.reflector_context_idx, msg, emit=True)
                return
            
            incoming = self._clean_content(event.content)
            if incoming and self.current_thinking and incoming.startswith(self.current_thinking):
                delta = incoming[len(self.current_thinking) :]
                self.current_thinking = incoming
            else:
                delta = incoming
                self.current_thinking += incoming
            self.current_text = ""
            self.current_text_idx = None
            if self.current_thinking_idx is None:
                msg = {"role": "assistant", "kind": "thinking", "content": self.current_thinking}
                msg["source"] = "agent"
                msg["iteration"] = self.current_iteration
                self.current_thinking_idx = self._append_history(msg)
                if self._turn_usage_target_idx is None:
                    self._turn_usage_target_idx = self.current_thinking_idx
            else:
                with self.lock:
                    msg = dict(self.history[self.current_thinking_idx])
                msg["content"] = self.current_thinking
                msg["source"] = "agent"
                self._patch_history(self.current_thinking_idx, msg, emit=False)
                self._append_delta_history(self.current_thinking_idx, delta)
            return

        if event.type == EventType.TEXT:
            metadata = event.metadata if isinstance(event.metadata, dict) else {}
            is_reflector = metadata.get("reflector", False)
            
            if is_reflector and self.reflector_context_idx is not None:
                # Update reflector_context message's response field
                incoming = self._clean_content(event.content)
                with self.lock:
                    msg = dict(self.history[self.reflector_context_idx])
                current_response = msg.get("response", "")
                msg["response"] = current_response + incoming
                self._patch_history(self.reflector_context_idx, msg, emit=True)
                return
            
            incoming = self._clean_content(event.content)
            if not incoming.strip():
                return
            if incoming and self.current_text and incoming.startswith(self.current_text):
                delta = incoming[len(self.current_text) :]
                self.current_text = incoming
            else:
                delta = incoming
                self.current_text += incoming
            self.current_thinking = ""
            self.current_thinking_idx = None
            if self.current_text_idx is None:
                msg = {"role": "assistant", "kind": "assistant_response", "content": self.current_text}
                msg["source"] = "agent"
                msg["iteration"] = self.current_iteration
                self.current_text_idx = self._append_history(msg)
                if self._turn_usage_target_idx is None:
                    self._turn_usage_target_idx = self.current_text_idx
            else:
                with self.lock:
                    msg = dict(self.history[self.current_text_idx])
                msg["content"] = self.current_text
                msg["source"] = "agent"
                self._patch_history(self.current_text_idx, msg, emit=False)
                self._append_delta_history(self.current_text_idx, delta)
            return

        # Any non-streaming event flushes current streaming handles.
        self._finalize_streams()

        if event.type == EventType.TOOL_CALL:
            if bool(event.metadata.get("_partial")):
                tool_id = str(event.metadata.get("id", ""))
                tool_name = str(event.metadata.get("name", "unknown"))
                input_raw = str(event.metadata.get("input_raw", "") or "")
                partial_fields = self._extract_partial_tool_fields(tool_name, input_raw)
                if not tool_id:
                    return
                existing_idx = self.partial_tool_call_indices.get(tool_id)
                if existing_idx is None:
                    msg = {
                        "role": "assistant",
                        "kind": "tool_call",
                        "source": "agent",
                        "content": "",
                        "tool_name": tool_name,
                        "tool_raw_json": input_raw,
                        "_partial": True,
                        "tool_id": tool_id,
                        "iteration": self.current_iteration,
                    }
                    msg.update(partial_fields)
                    idx = self._append_history(msg)
                    self.partial_tool_call_indices[tool_id] = idx
                else:
                    with self.lock:
                        existing = dict(self.history[existing_idx])
                    existing["tool_raw_json"] = input_raw
                    for k, v in partial_fields.items():
                        existing[k] = v
                    self._patch_history(existing_idx, existing)
                return

            name = str(event.metadata.get("name", "unknown"))
            tool_id = str(event.metadata.get("id", ""))
            tool_input = event.metadata.get("input", {}) or {}
            msg: dict[str, Any] = {
                "role": "assistant",
                "kind": "tool_call",
                "source": "agent",
                "content": "",
                "tool_name": name,
                "tool_id": tool_id,
                "iteration": self.current_iteration,
                "tool_input": self._clone_jsonable(tool_input),
            }
            display_fields = self._extract_tool_display_fields(name, tool_input) if isinstance(tool_input, dict) else {}
            if display_fields:
                msg.update(display_fields)
            elif name not in ("run_code", "run_code_in_previous_runtime", "submit_transform_hypothesis", "execute_python_solution"):
                msg["content"] = "```json\n" + json.dumps(tool_input, indent=2, default=str) + "\n```"
            else:
                msg["content"] = "```json\n" + json.dumps(tool_input, indent=2, default=str) + "\n```"
            existing_idx = self.partial_tool_call_indices.pop(tool_id, None) if tool_id else None
            if existing_idx is not None:
                self._patch_history(existing_idx, msg)
            else:
                self._append_history(msg)
            self._update_panels(latest_tool=_latest_tool_markdown(self.handler), latest_code=_latest_code_markdown(self.handler))
            return

        if event.type == EventType.TOOL_RESULT:
            output = event.metadata.get("output", event.content) or ""
            content_blocks = event.metadata.get("content_blocks")
            block_text, block_images, interleaved_blocks = self._extract_tool_result_content(content_blocks)
            if block_text:
                output = block_text

            images: list[str] = list(block_images)
            for img in event.images or []:
                img_data = str(img)
                if not img_data:
                    continue
                if img_data.startswith("data:image/"):
                    images.append(img_data)
                else:
                    images.append(f"data:image/png;base64,{img_data}")

            msg: dict[str, Any] = {
                "role": "assistant",
                "kind": "tool_result",
                "source": "orchestrator",
                "content": self._clean_tool_result_text(output),
                "tool_name": str(event.metadata.get("tool_name", "") or ""),
                "tool_use_id": str(
                    event.metadata.get("tool_use_id", event.metadata.get("id", "")) or ""
                ),
                "iteration": self.current_iteration,
            }
            # Store original API-level content_blocks for faithful checkpoint resume.
            # For execute_python_solution this is a list of text+image blocks;
            # for exploratory tools it is None (content is a plain string).
            if isinstance(content_blocks, list) and content_blocks:
                msg["api_content_blocks"] = self._clone_jsonable(content_blocks)
            # Preserve is_error flag for faithful reconstruction
            if event.metadata.get("is_error"):
                msg["is_error"] = True
            # Include interleaved blocks for proper text/image ordering in UI
            if interleaved_blocks:
                msg["interleaved_blocks"] = interleaved_blocks
            if images:
                deduped = list(dict.fromkeys(images))
                msg["images"] = deduped
            self._append_history(msg)
            self._update_panels(
                latest_tool=_latest_tool_markdown(self.handler),
                latest_code=_latest_code_markdown(self.handler),
            )
            return

        if event.type == EventType.REFLECTION:
            # Reflection prompt is a user message from orchestrator (not assistant)
            # Include iteration metadata so frontend knows this belongs to current iteration
            is_consolidated = bool(event.metadata and event.metadata.get("consolidated_prompt"))
            msg = {
                "role": "user",
                "kind": "iteration_context" if is_consolidated else "reflection_prompt",
                "source": "orchestrator",
                "content": str(event.content or ""),
                "iteration": event.metadata.get("iteration", 0) if event.metadata else 0,
            }
            if event.metadata and event.metadata.get("context_mode"):
                msg["context_mode"] = str(event.metadata.get("context_mode"))
            if event.metadata and isinstance(event.metadata.get("resume_messages"), list):
                msg["resume_messages"] = self._clone_jsonable(event.metadata["resume_messages"])
            if event.metadata and isinstance(event.metadata.get("snapshot_messages"), list):
                msg["snapshot_messages"] = self._clone_jsonable(event.metadata["snapshot_messages"])
            if event.metadata and isinstance(event.metadata.get("prompt_blocks"), list):
                msg["prompt_blocks"] = self._clone_jsonable(event.metadata["prompt_blocks"])
            # Preserve interleaved blocks structure from consolidated prompt (text/images properly interleaved)
            if event.metadata and event.metadata.get("interleaved_blocks"):
                msg["interleaved_blocks"] = event.metadata["interleaved_blocks"]
            # Fallback: preserve simple images array for backward compatibility
            elif event.metadata and event.metadata.get("images"):
                msg["images"] = event.metadata["images"]
            if event.metadata and event.metadata.get("ui_title"):
                msg["ui_title"] = event.metadata["ui_title"]
            if event.metadata and event.metadata.get("mode"):
                msg["mode"] = str(event.metadata.get("mode"))
            if event.metadata and isinstance(event.metadata.get("branch_expansion_state"), dict):
                msg["branch_expansion_state"] = self._clone_jsonable(event.metadata["branch_expansion_state"])
            context_mode = str(msg.get("context_mode", "") or self._infer_iteration_context_mode(msg))
            include_main_system_prompt = (
                bool(self.system_prompt_md)
                and (
                    (is_consolidated and self.current_canvas_owner == "main_agent"
                     and (context_mode == "handoff" or self.current_canvas_phase == "post_reflector"))
                    or context_mode == "snapshot"
                )
            )
            if (
                include_main_system_prompt
            ):
                msg["system_md"] = self.system_prompt_md
            target_iteration = int(msg.get("iteration", 0) or 0)
            if is_consolidated and context_mode == "handoff" and target_iteration > int(self.current_iteration or 0):
                msg.update(
                    self._reserve_canvas_metadata(
                        iteration=target_iteration,
                        owner="main_agent",
                        phase="main",
                    )
                )
            self._append_history(self._normalize_history_message(msg))
            return

        if event.type == EventType.REFLECTOR_CONTEXT:
            meta = event.metadata or {}
            phase = meta.get("phase", "start")
            if phase == "start":
                # Build interleaved_blocks matching the canonical Gemini API part order:
                # train examples (text+img+text+img), test inputs (text+img),
                # hypothesis, code, predicted outputs (text+img)
                train_inputs = meta.get("train_inputs", []) or []
                train_outputs = meta.get("train_outputs", []) or []
                test_inputs = meta.get("test_inputs", []) or []
                test_predictions = meta.get("test_predictions", []) or []
                train_input_images = meta.get("train_input_images", []) or []
                train_output_images = meta.get("train_output_images", []) or []
                test_input_images = meta.get("test_input_images", []) or []
                test_prediction_images = meta.get("test_prediction_images", []) or []
                hypothesis = meta.get("hypothesis", "")
                code = meta.get("code", "")
                ambiguity_rationale = meta.get("ambiguity_rationale", "")
                review_candidates = meta.get("review_candidates", []) or []
                review_branches = meta.get("review_branches", []) or []

                iblocks: list[dict] = []

                # 1. Training examples interleaved
                if train_inputs and train_outputs:
                    iblocks.append({"type": "text", "content": "## Training Input/Output Pairs\n\nStudy these examples first to form your own understanding of the transformation."})
                    for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
                        iblocks.append({"type": "text", "content": f"### Training Example {i}\n**Input Grid:** {inp}"})
                        if i < len(train_input_images) and train_input_images[i]:
                            iblocks.append({"type": "image", "content": f"data:image/png;base64,{train_input_images[i]}"})
                        iblocks.append({"type": "text", "content": f"**Expected Output Grid:** {out}"})
                        if i < len(train_output_images) and train_output_images[i]:
                            iblocks.append({"type": "image", "content": f"data:image/png;base64,{train_output_images[i]}"})

                # 2. Test inputs interleaved (only on turn 1 — omitted on followup)
                if test_inputs:
                    iblocks.append({"type": "text", "content": "## Test Inputs"})
                for i, inp in enumerate(test_inputs):
                    iblocks.append({"type": "text", "content": f"### Test Example {i}\n**Input Grid:** {inp}"})
                    if i < len(test_input_images) and test_input_images[i]:
                        iblocks.append({"type": "image", "content": f"data:image/png;base64,{test_input_images[i]}"})

                if ambiguity_rationale:
                    iblocks.append({"type": "text", "content": f"## Ambiguity Rationale\n\n{ambiguity_rationale}"})

                # 3. Solver submission + candidate outputs
                if hypothesis or code:
                    solver_submission = "## Solver Submission"
                    if hypothesis:
                        solver_submission += f"\n\n**Hypothesis:**\n{hypothesis}"
                    if code:
                        solver_submission += f"\n\n**solve() Code:**\n```python\n{code}\n```"
                    iblocks.append({"type": "text", "content": solver_submission})

                normalized_candidates = list(review_candidates)
                if not normalized_candidates and test_predictions:
                    normalized_candidates = [
                        {
                            "index": i,
                            "candidates": [pred] if pred is not None else [],
                            "candidate_images": [test_prediction_images[i]] if i < len(test_prediction_images) and test_prediction_images[i] else [],
                            "error": None,
                        }
                        for i, pred in enumerate(test_predictions)
                    ]

                _ord = lambda n: f"{n}{'th' if 11 <= n % 100 <= 13 else ('th','st','nd','rd')[min(n % 10, 4) if n % 10 < 4 else 0]}"
                if normalized_candidates:
                    iblocks.append({"type": "text", "content": "## Candidate Outputs By Test Example"})
                    for row in normalized_candidates:
                        test_index = row.get("index", "?")
                        iblocks.append({"type": "text", "content": f"### Test Example {test_index}"})
                        if row.get("error"):
                            iblocks.append({"type": "text", "content": f"Error: {row.get('error')}"})
                            continue
                        candidate_images = list(row.get("candidate_images", []) or [])
                        for candidate_idx, pred in enumerate(row.get("candidates", []) or [], start=1):
                            iblocks.append({"type": "text", "content": f"{_ord(candidate_idx)} candidate: {pred}"})
                            image_offset = candidate_idx - 1
                            if image_offset < len(candidate_images) and candidate_images[image_offset]:
                                iblocks.append({"type": "image", "content": f"data:image/png;base64,{candidate_images[image_offset]}"})
                else:
                    normalized_branches = list(review_branches)
                    if not normalized_branches and (hypothesis or code or test_predictions):
                        normalized_branches = [
                            {
                                "branch_id": "A",
                                "rank": 1,
                                "selection_status": "selected",
                                "hypothesis": hypothesis,
                                "code": code,
                                "test_predictions": test_predictions,
                                "test_prediction_images": test_prediction_images,
                            }
                        ]

                    if normalized_branches:
                        iblocks.append({"type": "text", "content": "## Solver Candidate Branches"})
                        for branch in normalized_branches:
                            branch_id = branch.get("branch_id", "?")
                            rank = branch.get("rank", "?")
                            selection_status = str(branch.get("selection_status", "") or "").strip()
                            header = f"### Branch {branch_id} (rank {rank})"
                            if selection_status:
                                header += f"\n**Status:** {selection_status}"
                            iblocks.append({"type": "text", "content": header})
                            branch_hypothesis = str(branch.get("hypothesis", "") or "").strip()
                            branch_code = str(branch.get("code", "") or "")
                            if branch_hypothesis:
                                iblocks.append({"type": "text", "content": f"**Hypothesis:**\n{branch_hypothesis}"})
                            if branch_code:
                                iblocks.append({"type": "text", "content": f"**solve() Code:**\n```python\n{branch_code}\n```"})
                            branch_predictions = list(branch.get("test_predictions", []) or [])
                            branch_prediction_images = list(branch.get("test_prediction_images", []) or [])
                            if branch_predictions:
                                iblocks.append({"type": "text", "content": "**Predicted Test Outputs:**"})
                                for i, pred in enumerate(branch_predictions):
                                    iblocks.append({"type": "text", "content": f"### Test Example {i}\n**Predicted Output Grid:** {pred}"})
                                    if i < len(branch_prediction_images) and branch_prediction_images[i]:
                                        iblocks.append({"type": "image", "content": f"data:image/png;base64,{branch_prediction_images[i]}"})

                iblocks.append({"type": "text", "content": "---\nPlease perform your independent review following the guidelines in your system prompt."})

                msg: dict[str, Any] = {
                    "role": "user",
                    "kind": "reflector_context",
                    "source": "orchestrator",
                    "content": "",
                    "iteration": meta.get("iteration", self.current_iteration),
                    "reflector_provider": meta.get("provider", ""),
                    "reflector_model": meta.get("model", ""),
                    "system_md": meta.get("system_md", ""),
                    "interleaved_blocks": iblocks,
                    "review_candidates": self._clone_jsonable(review_candidates),
                    "review_branches": self._clone_jsonable(review_branches),
                    "ambiguity_rationale": ambiguity_rationale,
                    "reflector_turn": meta.get("reflector_turn", 1),
                    "verdict": None,
                    "thinking": "",
                    "response": "",
                    "status": "running",
                }
                self.reflector_context_idx = self._append_history(msg)
            elif phase == "result" and self.reflector_context_idx is not None:
                with self.lock:
                    idx = self.reflector_context_idx
                    existing = dict(self.history[idx])
                existing["verdict"] = meta.get("verdict")
                existing["thinking"] = meta.get("thinking", "")
                existing["response"] = meta.get("response", "")
                existing["status"] = "done"
                # Store token usage if returned by the reflector.
                for _ukey in (
                    "usage_input_tokens",
                    "usage_uncached_input_tokens",
                    "usage_cache_read_tokens",
                    "usage_cache_write_tokens",
                    "usage_thinking_tokens",
                    "usage_output_tokens",
                    "usage_output_includes_reasoning",
                    "usage_reasoning_tokens_reported",
                    "usage_total_tokens",
                ):
                    if meta.get(_ukey) is not None:
                        existing[_ukey] = meta[_ukey]
                # Attach cost from the ledger entry created by handler._record_usage
                if self.handler.request_ledger:
                    _ledger = self.handler.request_ledger[-1]
                    existing["usage_input_cost_usd"] = float(_ledger.get("input_cost_usd", 0.0) or 0.0)
                    existing["usage_output_cost_usd"] = float(_ledger.get("output_cost_usd", 0.0) or 0.0)
                    existing["usage_cost_usd"] = float(_ledger.get("estimated_cost_usd", 0.0) or 0.0)
                self._patch_history(idx, existing)
                self.reflector_context_idx = None
            return

        if event.type == EventType.IMAGE:
            image_payloads: list[str] = []
            if event.metadata.get("image"):
                img_data = str(event.metadata["image"])
                if img_data.startswith("data:image/"):
                    image_payloads.append(img_data)
                else:
                    image_payloads.append(f"data:image/png;base64,{img_data}")
            for img in event.images or []:
                img_data = str(img)
                if img_data.startswith("data:image/"):
                    image_payloads.append(img_data)
                else:
                    image_payloads.append(f"data:image/png;base64,{img_data}")
            msg = {
                "role": "assistant",
                "kind": "image",
                "source": "orchestrator",
                "content": self._clean_content(event.content),
                "images": image_payloads,
            }
            self._append_history(msg)
            return

        if event.type == EventType.ERROR:
            self._append_history(
                {
                    "role": "assistant",
                    "kind": "assistant_response",
                    "source": "orchestrator",
                    "content": f"❌ **Error:** {event.content}",
                }
            )
            return

        if event.type == EventType.COMPLETE:
            meta = event.metadata if isinstance(event.metadata, dict) else {}
            with self.lock:
                self.test_accuracy = meta.get("test_accuracy")
                self.test_correct_count = meta.get("test_correct_count")
                self.test_total = meta.get("test_total")
                self.test_solved_indices = meta.get("test_solved_indices")
            self._append_history(
                {
                    "role": "assistant",
                    "kind": "assistant_response",
                    "source": "orchestrator",
                    "content": f"**{event.content}**",
                }
            )
            return

        self._append_history(
            {
                "role": "assistant",
                "kind": "assistant_response",
                "source": "orchestrator",
                "content": self._clean_content(event.content),
            }
        )

    def pause_run(self):
        self.handler.request_pause()
        self._emit({"type": "run_state", "running": True, "paused": False, "status": "pausing"})

    def terminate_run(self):
        self.handler.terminate()
        self._emit({"type": "run_state", "running": True, "paused": False, "status": "terminating"})

    def start_run(self, config: dict[str, Any], ui_history: list[dict] | None = None):
        with self.lock:
            was_paused = self.paused
            if self.running:
                # If a previous stop hasn't fully cleaned up, wait briefly for the
                # old worker thread to finish instead of immediately rejecting.
                # This prevents the "Solver is already running" error after stop+revert.
                old_thread = self.worker_thread
                if old_thread is not None and old_thread.is_alive():
                    self.handler.stop()  # ensure stop flag is set
                    self.lock.release()
                    try:
                        old_thread.join(timeout=5.0)
                    finally:
                        self.lock.acquire()
                    if old_thread.is_alive():
                        return False, "Previous run is still shutting down. Please try again in a few seconds."
                # Previous thread finished — safe to proceed
                self.running = True
                self.paused = False
            else:
                self.running = True
                self.paused = False
            
            # CRITICAL: Drain any leftover events from the old thread BEFORE starting new solver.
            # Otherwise, old events get consumed by the main loop AFTER the new solver starts,
            # creating duplicate/stale UI elements.
            while not self.handler.event_queue.empty():
                try:
                    self.handler.event_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.status = "starting"
            self.config.update(config or {})
            run_config = dict(self.config)
            # If UI sent a truncated history (checkpoint rollback), store it for resume
            checkpoint_history = ui_history if isinstance(ui_history, list) and len(ui_history) > 1 else None
            # Sync server-side history to match the truncated UI history.
            # Without this, new events append to the old full server history → duplicates,
            # and a browser refresh restores all "deleted" blocks from the stale snapshot.
            if ui_history is not None:
                import logging as _log
                _logger = _log.getLogger(__name__)
                _logger.info(f"Resume: received {len(ui_history)} UI history blocks")
                normalized_history = self._normalize_history_list(list(ui_history))
                migrated_history, _ = self._migrate_history_canvases(normalized_history)
                self.history = migrated_history
                self._restore_runtime_state_from_history()
                checkpoint_history = migrated_history if len(migrated_history) > 1 else None
            else:
                self._restore_runtime_state_from_history()

        self._emit({"type": "run_state", "running": True, "status": "starting"})
        self._refresh_prompt_panels_from_config(emit=False)
        self.worker_thread = threading.Thread(target=self._run_solver, args=(run_config, checkpoint_history), daemon=True)
        self.worker_thread.start()
        return True, None

    def rollback_history(self, truncated_history: list[dict]):
        """Immediately sync server-side history to the truncated UI history.

        Called as soon as the user deletes messages in the UI, so the server
        snapshot is correct even if the user refreshes before clicking Solve.
        Does nothing if the solver is currently running (live state takes precedence).
        """
        with self.lock:
            if self.running:
                return
            normalized_history = self._normalize_history_list(list(truncated_history))
            migrated_history, _ = self._migrate_history_canvases(normalized_history)
            self.history = migrated_history
            self._restore_runtime_state_from_history()

    def update_config(self, config: dict[str, Any]):
        with self.lock:
            if self.running:
                return
            old_model = str(self.config.get("model") or "")
            self.config.update(config or {})
            if "model" in (config or {}) and "phoenix_project" not in (config or {}):
                new_model = str(self.config.get("model") or "")
                if new_model != old_model:
                    self.config["phoenix_project"] = _default_phoenix_project_for_model(new_model)
        self._refresh_prompt_panels_from_config(emit=True)

    @classmethod
    def _reconstruct_api_messages(cls, ui_history: list[dict], user_prompt_content: list[dict]) -> list[dict] | None:
        """Convert UI history blocks back to Anthropic API messages.

        Anthropic API rules:
        - Messages alternate user / assistant roles.
        - A single assistant message can contain: thinking + text + tool_use blocks.
        - After assistant[tool_use], next must be user[tool_result].
        - Thinking blocks need signature to be passed back.

        Returns a list of API messages, or None if reconstruction fails.
        """
        if not ui_history:
            return None

        api_msgs: list[dict] = []
        # Start with the initial user prompt (always first)
        api_msgs.append({"role": "user", "content": user_prompt_content})

        # Walk through UI history, skipping prompt_bundle and turn_dividers
        pending_assistant_content: list[dict] = []
        pending_tool_results: list[dict] = []
        # Track tool_use IDs that have already been matched to tool_results
        matched_tool_ids: set[str] = set()

        def flush_assistant():
            nonlocal pending_assistant_content
            if pending_assistant_content:
                api_msgs.append({"role": "assistant", "content": list(pending_assistant_content)})
                pending_assistant_content = []

        def flush_tool_results():
            nonlocal pending_tool_results
            if pending_tool_results:
                api_msgs.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results = []

        for block in ui_history:
            kind = block.get("kind", "")
            role = block.get("role", "")

            # Skip non-API blocks
            if kind in ("prompt_bundle", "turn_divider", "image", "reflector_context"):
                continue

            if role == "assistant" and kind == "thinking":
                # Flush any pending tool results before new assistant content
                flush_tool_results()
                content_text = block.get("content", "")
                sig = block.get("thinking_signature", "")
                if content_text and sig:
                    # Reconstruct proper thinking block with valid cryptographic signature.
                    # The API requires signature for thinking blocks.
                    pending_assistant_content.append({
                        "type": "thinking",
                        "thinking": content_text,
                        "signature": sig,
                    })
                # If no signature available, skip this thinking block entirely
                # (can't be sent to API without a valid signature)

            elif role == "assistant" and kind == "tool_call":
                flush_tool_results()
                tool_id = block.get("tool_id", "")
                tool_name = block.get("tool_name", "")
                tool_input = block.get("tool_input")
                if not isinstance(tool_input, dict):
                    tool_input = {}
                    if tool_name in {"run_code", "run_code_in_previous_runtime"}:
                        tool_input = {"code": block.get("tool_code", "")}
                    elif tool_name == "submit_transform_hypothesis":
                        if block.get("tool_transform"):
                            tool_input["hypothesis"] = block.get("tool_transform", "")
                        if block.get("tool_target_branch_id"):
                            tool_input["branch_id"] = block.get("tool_target_branch_id", "")
                        if block.get("tool_target_branch_transform"):
                            tool_input["branch_hypothesis"] = block.get("tool_target_branch_transform", "")
                        if block.get("tool_transform") and "hypothesis" not in tool_input:
                            tool_input["branch_a_hypothesis"] = block.get("tool_transform", "")
                        if block.get("tool_secondary_transform"):
                            tool_input["branch_b_hypothesis"] = block.get("tool_secondary_transform", "")
                        if block.get("tool_ambiguity_rationale"):
                            tool_input["ambiguity_rationale"] = block.get("tool_ambiguity_rationale", "")
                        if block.get("tool_updated_failed_transform"):
                            tool_input["updated_failed_hypothesis"] = block.get("tool_updated_failed_transform", "")
                        if block.get("tool_updated_ambiguity_rationale"):
                            tool_input["updated_ambiguity_rationale"] = block.get("tool_updated_ambiguity_rationale", "")
                        if block.get("tool_withdraw_failed_branch"):
                            tool_input["withdraw_failed_branch"] = True
                    elif tool_name == "execute_python_solution":
                        if block.get("tool_code"):
                            tool_input["code"] = block.get("tool_code", "")
                        if block.get("tool_target_branch_id"):
                            tool_input["branch_id"] = block.get("tool_target_branch_id", "")
                        if block.get("tool_target_branch_code"):
                            tool_input["branch_code"] = block.get("tool_target_branch_code", "")
                        if block.get("tool_code") and "code" not in tool_input:
                            tool_input["branch_a_code"] = block.get("tool_code", "")
                        if block.get("tool_secondary_code"):
                            tool_input["branch_b_code"] = block.get("tool_secondary_code", "")
                        if block.get("tool_updated_failed_code"):
                            tool_input["updated_failed_code"] = block.get("tool_updated_failed_code", "")
                if tool_id and tool_name:
                    pending_assistant_content.append({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_input,
                    })

            elif kind == "tool_result":
                # Flush assistant content before adding tool result
                flush_assistant()
                # Use stored API-level content blocks if available (faithful resume);
                # fall back to plain text string for older checkpoints.
                api_content_blocks = block.get("api_content_blocks")
                if isinstance(api_content_blocks, list) and api_content_blocks:
                    tool_result_content: str | list = api_content_blocks
                else:
                    tool_result_content = block.get("content", "")
                # Find the matching tool_use_id from the last assistant message.
                # Walk tool_use blocks in order and pick the first unmatched one.
                tool_use_id = ""
                if api_msgs and api_msgs[-1].get("role") == "assistant":
                    for cb in api_msgs[-1].get("content", []):
                        if (isinstance(cb, dict)
                                and cb.get("type") == "tool_use"
                                and cb.get("id", "") not in matched_tool_ids):
                            tool_use_id = cb.get("id", "")
                            break
                if tool_use_id:
                    matched_tool_ids.add(tool_use_id)
                    tr: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_result_content,
                    }
                    if block.get("is_error"):
                        tr["is_error"] = True
                    pending_tool_results.append(tr)

            elif role == "assistant" and kind == "assistant_response":
                flush_tool_results()
                content_text = block.get("content", "")
                if content_text:
                    pending_assistant_content.append({"type": "text", "text": content_text})

            elif kind in ("reflection_prompt", "iteration_context"):
                # Reflection / iteration context is a user message from orchestrator
                flush_assistant()
                flush_tool_results()
                content_text = block.get("content", "")
                block_iter = block.get("iteration")
                if kind == "iteration_context":
                    resume_messages = block.get("resume_messages")
                    if isinstance(resume_messages, list) and resume_messages:
                        api_msgs = cls._clone_jsonable(resume_messages)
                        pending_assistant_content = []
                        pending_tool_results = []
                        matched_tool_ids = set()
                        continue

                    context_mode = cls._infer_iteration_context_mode(block)
                    if context_mode == "snapshot":
                        logger.warning(
                            "Checkpoint resume fell back to legacy snapshot heuristic; exact resume_messages were unavailable."
                        )
                        continue

                    prompt_blocks = block.get("prompt_blocks")
                    if isinstance(prompt_blocks, list) and prompt_blocks:
                        api_msgs = [{"role": "user", "content": cls._clone_jsonable(prompt_blocks)}]
                        pending_assistant_content = []
                        pending_tool_results = []
                        matched_tool_ids = set()
                        continue

                    legacy_content = cls._interleaved_blocks_to_api_content(block.get("interleaved_blocks"))
                    if legacy_content:
                        api_msgs = [{"role": "user", "content": legacy_content}]
                        pending_assistant_content = []
                        pending_tool_results = []
                        matched_tool_ids = set()
                        continue

                    if content_text:
                        api_msgs = [{"role": "user", "content": user_prompt_content + [{"type": "text", "text": content_text}]}]
                        pending_assistant_content = []
                        pending_tool_results = []
                        matched_tool_ids = set()
                        continue

                    logger.warning(
                        "Checkpoint resume encountered iteration_context without resume_messages or prompt content; using prior reconstructed state."
                    )
                elif isinstance(block_iter, (int, float)) and block_iter > 0:
                    api_msgs = [{"role": "user", "content": user_prompt_content + [{"type": "text", "text": content_text}]}]
                    pending_assistant_content = []
                    pending_tool_results = []
                    matched_tool_ids = set()
                elif content_text:
                    api_msgs.append({"role": "user", "content": [{"type": "text", "text": content_text}]})

        # Flush remaining
        flush_assistant()
        flush_tool_results()

        # Validate: must have at least the initial user message
        if len(api_msgs) < 1:
            return None
        # Ensure last message is user role for API to generate assistant response.
        # If checkpoint ends at an assistant message, append a continuation prompt.
        if api_msgs[-1].get("role") != "user":
            if api_msgs[-1].get("role") == "assistant":
                # Check if the last assistant message has tool_use blocks that need tool_results
                last_content = api_msgs[-1].get("content", [])
                tool_uses_in_last = [
                    b for b in (last_content if isinstance(last_content, list) else [])
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                ]
                if not tool_uses_in_last:
                    # Append a continuation user message (dangling tool_uses are left
                    # for the orchestrator to detect and re-execute on resume)
                    api_msgs.append({
                        "role": "user",
                        "content": [{"type": "text", "text": "Continue from where you left off. You may use run_code or run_code_in_previous_runtime for experiments, or execute_python_solution to test your hypothesis."}]
                    })
            else:
                return None

        # Keep all thinking blocks (with signatures) to faithfully reproduce
        # the conversation as if it were a fresh run.  The API accepts thinking
        # blocks from all turns — it only *requires* the latest one.

        return api_msgs

    def _run_solver(self, config: dict[str, Any], checkpoint_history: list[dict] | None = None):
        self.handler.reset()
        self.handler.set_model_name(str(config.get("model") or "claude-opus-4-6"))
        self._emit({"type": "usage_update", "usage": self._usage_snapshot()})
        self._finalize_streams()
        self._turn_usage_target_idx = None
        self.partial_tool_call_indices.clear()
        self.reflector_context_idx = None
        # Initialize current_turn from checkpoint if available, otherwise 0
        if checkpoint_history:
            self._restore_runtime_state_from_history()
        else:
            self._restore_runtime_state_from_history()
        phoenix = None
        try:
            puzzle_path = self._resolve_puzzle_path(
                str(config.get("puzzle_path") or ""),
                split=str(config.get("split") or "public_eval"),
                dataset_root=str(config.get("dataset_root") or "") or None,
            )
            model_name = str(config.get("model") or "claude-opus-4-6")
            effort_defaults = _default_efforts_for_model(model_name)
            thinking_effort = str(config.get("thinking_effort") or effort_defaults["thinking_effort"])
            reflection_thinking_effort = str(
                config.get("reflection_thinking_effort") or effort_defaults["reflection_thinking_effort"]
            )
            compression_thinking_effort = str(
                config.get("compression_thinking_effort") or effort_defaults["compression_thinking_effort"]
            )
            thinking_budget = int(config.get("thinking_budget") or 16000)
            compression_threshold = int(config.get("compression_threshold") or 170000)
            compression_bypass_threshold = int(config.get("compression_bypass_threshold") or 120000)
            max_turns = int(config.get("max_turns") or 200)
            max_test_predictions = int(config.get("max_test_predictions") or 2)
            phoenix_project = str(config.get("phoenix_project") or _default_phoenix_project_for_model(model_name))
            enable_phoenix = bool(config.get("enable_phoenix", True))
            unsafe_local_exec = bool(config.get("unsafe_local_exec", True))
            split = str(config.get("split") or "public_eval")
            dataset_root = str(config.get("dataset_root") or "").strip() or None
            enable_independent_reflector = bool(config.get("enable_independent_reflector", True))
            reflector_provider = str(config.get("reflector_provider") or "gemini")
            reflector_model = str(config.get("reflector_model") or "").strip() or None
            reflector_thinking_effort = str(config.get("reflector_thinking_effort") or "high")
            reflector_code_execution = bool(config.get("reflector_code_execution", False))
            semi_cot_first_turn = bool(config.get("semi_cot_first_turn", False))
            semi_cot_thinking_effort = str(config.get("semi_cot_thinking_effort") or "high")

            if not puzzle_path:
                self._append_history(
                    {
                        "role": "assistant",
                        "kind": "assistant_response",
                        "source": "orchestrator",
                        "content": "⚠️ Puzzle path is empty.",
                    }
                )
                self._set_running(False, "idle")
                return

            if not Path(puzzle_path).exists():
                self._append_history(
                    {
                        "role": "assistant",
                        "kind": "assistant_response",
                        "source": "orchestrator",
                        "content": f"⚠️ File not found: {puzzle_path}",
                    }
                )
                self._set_running(False, "idle")
                return

            # Load prompts for display bundle
            system_prompt = load_system_prompt()
            puzzle_data = load_puzzle(puzzle_path)
            use_vision = not _is_glm_model(model_name)
            user_prompt_blocks = format_puzzle_for_prompt(puzzle_data, use_vision=use_vision)
            user_prompt_blocks = _append_kimi_run_code_guidance_if_needed(
                user_prompt_blocks,
                model_name,
            )
            user_prompt_md = self._build_user_prompt_markdown(user_prompt_blocks)

            self._update_panels(system=system_prompt, user=user_prompt_md, latest_tool="*No tool calls yet*", latest_code="*No code submitted yet*")
            # Keep a single prompt bundle in chat (update existing startup bundle instead of appending duplicate).
            self._sync_prompt_bundle_history()

            if enable_phoenix:
                try:
                    os.environ["ENABLE_PHOENIX"] = "true"
                    os.environ["PHOENIX_PROJECT_NAME"] = phoenix_project
                    # Always instrument the main Anthropic agent.
                    # Instrument Google GenAI when reflector uses Gemini so
                    # reflector calls appear as MessageStream in Phoenix.
                    instrument_genai = (enable_independent_reflector and reflector_provider == "gemini")
                    phoenix = initialize_phoenix(
                        instrument_openai=False,
                        instrument_anthropic=True,
                        instrument_google_genai=instrument_genai,
                    )
                except Exception:
                    phoenix = None

            self._set_running(True, "running")

            # Reconstruct API messages from checkpoint history if available
            initial_messages = None
            checkpoint_iteration = 0
            checkpoint_turn = 0
            checkpoint_reflector_message_history = None
            checkpoint_reflector_message_history = None
            if checkpoint_history:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Checkpoint resume: {len(checkpoint_history)} UI history blocks")

                # Derive iteration, turn, and reflection mode from checkpoint history
                checkpoint_in_reflection = False
                for block in checkpoint_history:
                    if block.get("kind") == "turn_divider":
                        tn = block.get("turn_number", 0)
                        if isinstance(tn, (int, float)) and tn > checkpoint_turn:
                            checkpoint_turn = int(tn)
                    it = block.get("iteration")
                    if isinstance(it, (int, float)) and it > checkpoint_iteration:
                        checkpoint_iteration = int(it)
                # Detect if checkpoint ends at a reflection prompt (model hasn't responded yet),
                # OR if the reflection_prompt was deleted (rollback) and checkpoint ends at
                # a tool_result from execute_python_solution — in which case the orchestrator
                # must rebuild and inject the reflection prompt on resume.
                checkpoint_in_test_gen_reflection = False
                checkpoint_in_reflector_reject_compression = False
                checkpoint_expand_reflector_response = ""  # Set when EXPAND_CANDIDATES
                checkpoint_reflector_response = ""  # Set for both EXPAND and REJECT
                checkpoint_test_candidates = []  # Test prediction candidates from reflector
                checkpoint_branch_expansion_state = None
                last_exec_tool_name = None
                _saw_completion = False
                _saw_post_reflection_response = False
                for block in reversed(checkpoint_history):
                    kind = block.get("kind", "")
                    if self._looks_like_orchestrator_completion_message(block):
                        _saw_completion = True
                        continue
                    # If we already passed a completion message and see an
                    # assistant response, the solver already answered whatever
                    # reflection/prompt came before — the run completed normally.
                    if _saw_completion and kind == "assistant_response":
                        _saw_post_reflection_response = True
                        continue
                    if kind == "iteration_context":
                        if (
                            str(block.get("canvas_phase", "") or "") == "post_reflector"
                            or self._looks_like_post_reflector_title(block)
                        ):
                            checkpoint_in_reflection = True
                            checkpoint_in_reflector_reject_compression = True
                        break
                    if kind == "reflector_context":
                        verdict = str(block.get("verdict", "") or "").upper()
                        if verdict in ("EXPAND_CANDIDATES", "REJECT"):
                            checkpoint_in_reflection = True
                            checkpoint_in_reflector_reject_compression = True
                            checkpoint_reflector_response = str(block.get("response", "") or "")
                            if verdict == "EXPAND_CANDIDATES":
                                checkpoint_expand_reflector_response = checkpoint_reflector_response
                        break
                    if kind == "reflection_prompt":
                        if _last_substantive_is_response:
                            # The solver already answered this reflection — the
                            # run completed its current phase. Don't re-enter
                            # reflection mode on resume.
                            break
                        checkpoint_in_reflection = True
                        # Detect test generalization reflection by content marker
                        content = block.get("content", "")
                        if "Train accuracy achieved 100%" in content or "test-set generalization" in content.lower():
                            checkpoint_in_test_gen_reflection = True
                        if str(block.get("mode", "") or "") == "branch_expansion" and isinstance(block.get("branch_expansion_state"), dict):
                            checkpoint_branch_expansion_state = copy.deepcopy(block.get("branch_expansion_state"))
                        break
                    elif kind == "tool_result":
                        # Check if this tool_result is from execute_python_solution
                        tool_name = block.get("tool_name", "")
                        if tool_name == "execute_python_solution":
                            # Checkpoint ends right after execute_python_solution result.
                            # The reflection prompt was deleted (rollback). Orchestrator must rebuild it.
                            checkpoint_in_reflection = True
                            # Check the preceding tool_call to determine if it was a 100%-pass
                            # (test gen reflection) or a failure (regular reflection).
                            # Walk backwards past this tool_result to find the matching tool_call.
                            for prev_block in reversed(checkpoint_history[:checkpoint_history.index(block)]):
                                if prev_block.get("kind") == "tool_call" and prev_block.get("tool_name") == "execute_python_solution":
                                    # Check the tool_result content for 100% pass marker
                                    result_content = block.get("content", "")
                                    if "PASS" in result_content and "FAIL" not in result_content:
                                        checkpoint_in_test_gen_reflection = True
                                    break
                        break
                    elif kind in ("tool_call", "assistant_response", "thinking"):
                        break
                logger.info(
                    "Checkpoint state: iteration=%s, turn=%s, in_reflection=%s, test_gen=%s, reflector_reject=%s, branch_expansion=%s",
                    checkpoint_iteration,
                    checkpoint_turn,
                    checkpoint_in_reflection,
                    checkpoint_in_test_gen_reflection,
                    checkpoint_in_reflector_reject_compression,
                    bool(checkpoint_branch_expansion_state),
                )

                initial_messages = self._reconstruct_api_messages(
                    checkpoint_history, user_prompt_blocks
                )
                # Reconstruct reflector conversation history for stateful multi-turn
                checkpoint_reflector_message_history = []
                for block in checkpoint_history:
                    if (
                        block.get("kind") == "reflector_context"
                        and block.get("status") == "done"
                        and block.get("response")
                    ):
                        # Build text-only user message from interleaved_blocks
                        user_parts = []
                        for ib in (block.get("interleaved_blocks") or []):
                            if ib.get("type") == "text":
                                user_parts.append(ib.get("content", ""))
                        user_text = "\n".join(user_parts)
                        if user_text:
                            checkpoint_reflector_message_history.append(
                                {"role": "user", "content": user_text}
                            )
                            checkpoint_reflector_message_history.append(
                                {"role": "assistant", "content": block["response"]}
                            )
                if checkpoint_reflector_message_history:
                    logger.info(
                        f"Reconstructed reflector message history: "
                        f"{len(checkpoint_reflector_message_history) // 2} turn(s)"
                    )
                # Reconstruct reflector conversation history for stateful multi-turn
                checkpoint_reflector_message_history = []
                for block in checkpoint_history:
                    if (
                        block.get("kind") == "reflector_context"
                        and block.get("status") == "done"
                        and block.get("response")
                    ):
                        # Build text-only user message from interleaved_blocks
                        user_parts = []
                        for ib in (block.get("interleaved_blocks") or []):
                            if ib.get("type") == "text":
                                user_parts.append(ib.get("content", ""))
                        user_text = "\n".join(user_parts)
                        if user_text:
                            checkpoint_reflector_message_history.append(
                                {"role": "user", "content": user_text}
                            )
                            checkpoint_reflector_message_history.append(
                                {"role": "assistant", "content": block["response"]}
                            )
                if checkpoint_reflector_message_history:
                    logger.info(
                        f"Reconstructed reflector message history: "
                        f"{len(checkpoint_reflector_message_history) // 2} turn(s)"
                    )
                # If checkpoint ended at EXPAND_CANDIDATES, inject expansion
                # feedback into messages. The solver will process this during the
                # compression turn (creating a memory checkpoint), then start a
                # clean iteration with the expansion guidance incorporated.
                if initial_messages and checkpoint_expand_reflector_response:
                    expansion_text = build_candidate_expansion_guidance_prompt(
                        reflector_response=checkpoint_expand_reflector_response,
                        bypass_compression=False,
                    )
                    combined_text = expansion_text + CONTEXT_COMPRESSION_PROMPT
                    initial_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": combined_text}],
                    })
                    logger.info(
                        "Injected EXPAND_CANDIDATES expansion feedback + compression prompt "
                        f"({len(combined_text)} chars)"
                    )

                if initial_messages:
                    # Log reconstructed message stats for debugging
                    total_chars = sum(
                        len(str(m.get("content", ""))) for m in initial_messages
                    )
                    n_thinking = sum(
                        1 for m in initial_messages if m.get("role") == "assistant"
                        for b in (m.get("content", []) if isinstance(m.get("content"), list) else [])
                        if isinstance(b, dict) and b.get("type") == "thinking"
                    )
                    logger.info(
                        f"Reconstructed {len(initial_messages)} API messages "
                        f"(~{total_chars:,} chars, {n_thinking} thinking blocks with signatures)"
                    )
                    # Restore the UI history so the user sees the checkpoint messages
                    with self.lock:
                        self.history = list(checkpoint_history)
                    self._emit({"type": "snapshot", "state": self._snapshot()})
                else:
                    logger.warning("Checkpoint resume: _reconstruct_api_messages returned None, falling back to fresh start")
                    checkpoint_iteration = 0
                    checkpoint_turn = 0
                    checkpoint_in_reflection = False

            def _runner():
                run_orchestration(
                    puzzle_path=puzzle_path,
                    model_name=model_name,
                    use_streaming=True,
                    use_visual_mode=(not _is_glm_model(model_name)),
                    use_extended_thinking=True,
                    thinking_budget=thinking_budget,
                    thinking_effort=thinking_effort,
                    reflection_thinking_effort=reflection_thinking_effort,
                    compression_thinking_effort=compression_thinking_effort,
                    max_turns=max_turns,
                    compression_threshold=compression_threshold,
                    compression_bypass_threshold=compression_bypass_threshold,
                    max_test_predictions=max_test_predictions,
                    emit_tool_call_deltas=True,
                    event_callback=self.handler.callback,
                    phoenix=phoenix,
                    should_stop=self.handler.check_stop,
                    stop_reason=self.handler.stop_reason,
                    initial_messages=initial_messages,
                    initial_iteration=checkpoint_iteration,
                    initial_turn=checkpoint_turn,
                    initial_in_reflection_mode=checkpoint_in_reflection if checkpoint_history else False,
                    initial_in_test_generalization_reflection=checkpoint_in_test_gen_reflection if checkpoint_history else False,
                    initial_in_reflector_reject_compression=checkpoint_in_reflector_reject_compression if checkpoint_history else False,
                    initial_reflector_message_history=checkpoint_reflector_message_history if checkpoint_history else None,
                    initial_reflector_response=checkpoint_reflector_response if checkpoint_history else "",
                    initial_test_candidates=checkpoint_test_candidates if checkpoint_history else None,
                    enable_independent_reflector=enable_independent_reflector,
                    reflector_provider=reflector_provider,
                    reflector_model=reflector_model,
                    reflector_thinking_effort=reflector_thinking_effort,
                    reflector_code_execution=reflector_code_execution,
                    semi_cot_first_turn=semi_cot_first_turn,
                    semi_cot_thinking_effort=semi_cot_thinking_effort,
                    enable_phoenix=enable_phoenix,
                    unsafe_local_exec=unsafe_local_exec,
                    dataset_root=dataset_root,
                    dataset_split=split,
                )

            orchestration_thread = threading.Thread(target=_runner, daemon=True)
            orchestration_thread.start()

            while orchestration_thread.is_alive() or not self.handler.event_queue.empty():
                try:
                    event = self.handler.event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._consume_event(event)

            # Final panel refresh at run end.
            self._update_panels(latest_tool=_latest_tool_markdown(self.handler), latest_code=_latest_code_markdown(self.handler))
            if self.handler.terminate_requested:
                self._set_running(False, "terminated")
            elif self.handler.pause_requested:
                self._set_paused(True, "paused")
            else:
                self._set_running(False, "idle")

            # Auto-save if enabled
            if config.get("auto_save", True) and not self.handler.terminate_requested:
                puzzle_id = str(config.get("puzzle_path") or "").strip()
                ok, msg = self.save_current_run(label=puzzle_id)
                if ok:
                    self._emit({"type": "status", "level": "info", "message": f"Auto-saved: {msg}"})
                else:
                    self._emit({"type": "status", "level": "error", "message": f"Auto-save failed: {msg}"})
        except Exception as e:
            err = f"{e}\n{traceback.format_exc()}"
            self._append_history(
                {
                    "role": "assistant",
                    "kind": "assistant_response",
                    "source": "orchestrator",
                    "content": f"❌ **Server error:**\n\n```text\n{err}\n```",
                }
            )
            self._set_running(False, "idle")


state = SolverAppState()
app = FastAPI(title="ARC Solver Web Demo")

# Allow cross-origin requests from the batch dashboard
try:
    from starlette.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
except ImportError:
    pass  # starlette CORS not available; dashboard polling will still work if same-origin


@app.on_event("startup")
async def _startup():
    state.loop = asyncio.get_running_loop()


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status():
    """Lightweight JSON status for dashboard polling."""
    with state.lock:
        return {
            "running": state.running,
            "paused": state.paused,
            "status": state.status,
            "puzzle_id": state.config.get("puzzle_path", ""),
            "iteration": state.current_iteration,
            "turn": state.current_turn,
            "usage": {
                "input_tokens": state.handler.total_input_tokens,
                "output_tokens": state.handler.total_output_tokens,
                "thinking_tokens": state.handler.total_thinking_tokens,
                "total_cost": state.handler.total_cost,
                "input_cost": state.handler.total_input_cost,
                "output_cost": state.handler.total_output_cost,
                "total_requests": state.handler.total_requests,
            },
            "test_accuracy": state.test_accuracy,
            "test_correct_count": state.test_correct_count,
            "test_total": state.test_total,
            "test_solved_indices": state.test_solved_indices,
        }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await state.add_client(ws)
    try:
        while True:
            raw = await ws.receive_text()
            payload = json.loads(raw)
            ptype = payload.get("type")
            if ptype == "start":
                ok, err = state.start_run(payload.get("config", {}), payload.get("history"))
                if not ok:
                    await ws.send_json({"type": "toast", "level": "error", "message": err})
            elif ptype == "pause":
                state.pause_run()
            elif ptype == "terminate":
                state.terminate_run()
            elif ptype == "config_update":
                state.update_config(payload.get("config", {}))
            elif ptype == "save_run":
                try:
                    ok, msg = state.save_current_run(str(payload.get("label", "") or ""))
                except Exception as e:
                    ok, msg = False, f"Failed to save run: {e}"
                level = "info" if ok else "error"
                await ws.send_json({"type": "toast", "level": level, "message": msg})
                await ws.send_json({"type": "saved_runs", "runs": state.list_saved_runs()})
            elif ptype == "load_run":
                ok, msg = state.load_saved_run(str(payload.get("run_id", "") or ""))
                level = "info" if ok else "error"
                await ws.send_json({"type": "toast", "level": level, "message": msg})
                await ws.send_json({"type": "saved_runs", "runs": state.list_saved_runs()})
            elif ptype == "list_runs":
                await ws.send_json({"type": "saved_runs", "runs": state.list_saved_runs()})
            elif ptype == "history_rollback":
                state.rollback_history(payload.get("history") or [])
            elif ptype == "remove_reflector_canvas":
                if state.running:
                    await ws.send_json({
                        "type": "toast",
                        "level": "error",
                        "message": "Cannot remove reflector messages while a run is active.",
                    })
                else:
                    removed = state._remove_reflector_canvas()
                    await ws.send_json({
                        "type": "toast",
                        "level": "info" if removed else "error",
                        "message": (
                            "Removed reflector messages. Click Solve to retry from checkpoint."
                            if removed
                            else "No reflector messages were available to remove."
                        ),
                    })
            elif ptype == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        state.remove_client(ws)
    except Exception:
        print("WebSocket handler error:\n" + traceback.format_exc())
        state.remove_client(ws)


def main():
    import argparse
    
    try:
        import uvicorn
    except ImportError:
        print("Missing dependency: uvicorn")
        print("Install with: pip install fastapi uvicorn")
        raise
    
    parser = argparse.ArgumentParser(description="Run ARC web demo")
    parser.add_argument("--port", type=int, default=7861, help="Port to run server on (default: 7861)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()

    uvicorn.run(
        "athanor.web_demo.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
