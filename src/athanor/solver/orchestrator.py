#!/usr/bin/env python3
"""Solve ARC puzzles using Anthropic Claude models with native SDK + Phoenix logging.

This module is the CANONICAL orchestration implementation.
orchestrator_core.py delegates to this module.
"""

import base64
import json
import os
import sys
import io
import contextlib
import traceback
import argparse
import time
import copy
import threading
import ctypes
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from .prompts import (
        build_candidate_expansion_guidance_prompt,
        BEST_EFFORT_PROMPT,
        CONTEXT_COMPRESSION_PROMPT,
        TEST_GENERALIZATION_REFLECTION_PROMPT,
        TRAIN_FAILURE_REFLECTION_PROMPT,
    )
except ImportError:
    from prompts import (  # noqa: E402  (direct execution / sys.path mode)
        build_candidate_expansion_guidance_prompt,
        BEST_EFFORT_PROMPT,
        CONTEXT_COMPRESSION_PROMPT,
        TEST_GENERALIZATION_REFLECTION_PROMPT,
        TRAIN_FAILURE_REFLECTION_PROMPT,
    )

# OpenTelemetry context suppression — prevents SDK auto-instrumentation from
# creating duplicate spans when we already have a manual MessagesStream span.
try:
    from opentelemetry.context import attach, detach, set_value, _SUPPRESS_INSTRUMENTATION_KEY
    _otel_suppress_available = True
except ImportError:
    _otel_suppress_available = False

try:
    import httpx
    import httpcore
except ImportError:
    httpx = None
    httpcore = None

try:
    import anthropic
except ImportError:
    anthropic = None
import numpy as np
try:
    from .grid_visualizer import render_grid_to_base64
    from .phoenix_observability import initialize_phoenix
except ImportError:
    from grid_visualizer import render_grid_to_base64
    from phoenix_observability import initialize_phoenix


def _ordinal(n: int) -> str:
    """Return ordinal string for a positive integer (1st, 2nd, 3rd, …)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{('th','st','nd','rd')[min(n % 10, 4) if n % 10 < 4 else 0]}"


def numpy_to_python(obj):
    """Convert NumPy types to native Python types recursively for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj


def _empty_hypothesis_submission() -> dict[str, str]:
    return {
        "hypothesis": "",
        # Legacy compatibility fields for older saved runs / checkpoints.
        "branch_a_hypothesis": "",
        "branch_b_hypothesis": "",
        "ambiguity_rationale": "",
    }




def _normalize_hypothesis_submission(args: dict[str, Any] | None) -> dict[str, Any]:
    raw = args or {}
    hypothesis = str(raw.get("hypothesis") or raw.get("branch_a_hypothesis") or "").strip()
    branch_b = str(raw.get("branch_b_hypothesis") or "").strip()
    ambiguity = str(raw.get("ambiguity_rationale") or "").strip()
    reviewer_response = str(raw.get("reviewer_response") or "").strip()
    return {
        "hypothesis": hypothesis,
        "branch_a_hypothesis": hypothesis,
        "branch_b_hypothesis": branch_b,
        "ambiguity_rationale": ambiguity,
        "reviewer_response": reviewer_response,
    }


def _copy_active_hypothesis_submission(submission: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(submission, dict):
        return _empty_hypothesis_submission()
    hypothesis = str(
        submission.get("hypothesis")
        or submission.get("branch_a_hypothesis")
        or ""
    ).strip()
    return {
        "hypothesis": hypothesis,
        "branch_a_hypothesis": hypothesis,
        "branch_b_hypothesis": str(submission.get("branch_b_hypothesis") or "").strip(),
        "ambiguity_rationale": str(submission.get("ambiguity_rationale") or "").strip(),
    }


def _validate_hypothesis_submission(submission: dict[str, Any]) -> str | None:
    hypothesis = str(submission.get("hypothesis") or submission.get("branch_a_hypothesis") or "").strip()
    if not hypothesis:
        return "Error: Missing required hypothesis."
    return None




def _normalize_code_submission(args: dict[str, Any] | None) -> dict[str, str]:
    raw = args or {}
    code = raw.get("code", "")
    if not code:
        code = raw.get("branch_a_code", "")
    return {
        "code": str(code or ""),
        # Legacy compatibility fields for older code paths / saved runs.
        "branch_id": "",
        "branch_code": "",
        "branch_a_code": str(code or ""),
        "branch_b_code": "",
        "updated_failed_code": "",
    }


def _branch_a_hypothesis_text(submission: dict[str, Any] | None) -> str:
    if not isinstance(submission, dict):
        return ""
    return str(
        submission.get("hypothesis")
        or submission.get("branch_a_hypothesis")
        or ""
    ).strip()


def _copy_hypothesis_submission(submission: dict[str, str] | None) -> dict[str, str]:
    return _copy_active_hypothesis_submission(submission)


def _derive_execute_hypothesis_submission(
    last_hypothesis_submission: dict[str, str] | None,
) -> dict[str, str]:
    return _copy_hypothesis_submission(last_hypothesis_submission)


def _is_valid_grid_candidate(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) and row for row in value):
        return False
    width = len(value[0])
    if width <= 0:
        return False
    for row in value:
        if len(row) != width:
            return False
        for cell in row:
            if not isinstance(cell, int):
                return False
    return True


def _normalize_solve_candidates(
    raw_prediction: Any,
    *,
    max_candidates: int = 2,
) -> tuple[list[list[list[int]]] | None, str | None]:
    prediction = numpy_to_python(raw_prediction)

    if _is_valid_grid_candidate(prediction):
        return [prediction], None

    if isinstance(prediction, tuple):
        prediction = list(prediction)

    if not isinstance(prediction, list):
        return None, "solve(grid) must return either a grid or a list of one or two grids."

    if not prediction:
        return None, "solve(grid) returned an empty candidate list."

    if len(prediction) > max_candidates:
        return None, f"solve(grid) returned more than {max_candidates} candidate grid(s)."

    normalized: list[list[list[int]]] = []
    seen: set[str] = set()
    for candidate in prediction:
        candidate = numpy_to_python(candidate)
        if not _is_valid_grid_candidate(candidate):
            return None, "Every item in the returned candidate list must be a valid 2D integer grid."
        try:
            signature = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        except Exception:
            signature = str(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        normalized.append(candidate)

    if not normalized:
        return None, "solve(grid) returned only duplicate candidates and no valid unique grid remained."

    return normalized, None


def _serialize_candidate_predictions(test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for row in test_rows or []:
        if not isinstance(row, dict):
            continue
        serialized.append(
            {
                "index": row.get("index"),
                "candidates": copy.deepcopy(row.get("candidates") or []),
                "error": row.get("error"),
            }
        )
    return serialized


def _extract_prediction_signature(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    preds = []
    for row in result.get("test", []) or []:
        if isinstance(row, dict):
            preds.append({
                "index": row.get("index"),
                "predicted": row.get("predicted"),
                "error": row.get("error"),
            })
    try:
        return json.dumps(preds, sort_keys=True, default=str)
    except Exception:
        return str(preds)


def _serialize_prediction_branches(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for branch in branches:
        result = branch.get("result") if isinstance(branch, dict) else None
        test_results = []
        if isinstance(result, dict):
            for row in result.get("test", []) or []:
                if isinstance(row, dict):
                    test_results.append({
                        "index": row.get("index"),
                        "predicted": row.get("predicted"),
                        "error": row.get("error"),
                    })
        out.append({
            "branch_id": branch.get("branch_id"),
            "rank": branch.get("rank"),
            "hypothesis": branch.get("hypothesis"),
            "test_results": test_results,
        })
    return out


def _clone_branch_record(
    branch: dict[str, Any],
    *,
    include_runtime_refs: bool = False,
) -> dict[str, Any]:
    """Copy branch data without attempting to deepcopy live runtime objects."""
    cloned: dict[str, Any] = {}
    for key in (
        "branch_id",
        "rank",
        "hypothesis",
        "code",
        "status",
        "error",
        "selection_status",
        "drop_reason",
    ):
        if key in branch:
            cloned[key] = copy.deepcopy(branch[key])
    if "result" in branch:
        cloned["result"] = copy.deepcopy(branch.get("result"))
    if include_runtime_refs:
        cloned["solve_fn"] = branch.get("solve_fn")
        cloned["local_ctx"] = branch.get("local_ctx")
    return cloned


def _summarize_branch_outcomes(
    branch_outcomes: list[dict[str, Any]],
    ambiguity_rationale: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    normalized: list[dict[str, Any]] = []
    valid_branches: list[dict[str, Any]] = []
    seen_signatures: dict[str, str] = {}

    for branch in branch_outcomes:
        branch_copy = _clone_branch_record(branch)
        if branch_copy.get("status") != "ok":
            branch_copy["selection_status"] = "error"
            normalized.append(branch_copy)
            continue

        result = branch_copy.get("result") if isinstance(branch_copy.get("result"), dict) else {}
        if not result.get("correct", False):
            branch_copy["selection_status"] = "dropped"
            branch_copy["drop_reason"] = "failed_training"
            normalized.append(branch_copy)
            continue

        signature = _extract_prediction_signature(result)
        if signature in seen_signatures:
            branch_copy["selection_status"] = "dropped"
            branch_copy["drop_reason"] = f"duplicate_predictions_of_{seen_signatures[signature]}"
            normalized.append(branch_copy)
            continue

        branch_copy["selection_status"] = "selected" if not valid_branches else "secondary"
        branch_copy["drop_reason"] = ""
        normalized.append(branch_copy)
        valid_branches.append(branch_copy)
        seen_signatures[signature] = str(branch_copy.get("branch_id") or "")

    selected_branch = valid_branches[0] if valid_branches else None
    if selected_branch and isinstance(selected_branch.get("result"), dict):
        top_level = copy.deepcopy(selected_branch["result"])
    else:
        fallback = branch_outcomes[0] if branch_outcomes else {}
        fallback_result = fallback.get("result") if isinstance(fallback, dict) else {}
        if isinstance(fallback_result, dict):
            top_level = copy.deepcopy(fallback_result)
        else:
            top_level = {
                "train": [],
                "test": [],
                "correct": False,
                "test_correct": None,
                "train_pixel_accuracy": 0.0,
            }

    top_level["branches"] = normalized
    top_level["valid_branch_count"] = len(valid_branches)
    top_level["selected_branch_id"] = selected_branch.get("branch_id") if selected_branch else None
    top_level["selected_branch_rank"] = selected_branch.get("rank") if selected_branch else None
    top_level["ambiguity_rationale"] = ambiguity_rationale
    top_level["prediction_branches"] = _serialize_prediction_branches(valid_branches)
    if branch_outcomes and len(branch_outcomes) > 1 and len(valid_branches) == 1:
        top_level["collapsed_to_single_prediction"] = True
    if len(valid_branches) > 1:
        top_level["num_test_predictions"] = min(2, len(valid_branches))
    elif len(valid_branches) == 1:
        top_level["num_test_predictions"] = 1
    else:
        top_level["num_test_predictions"] = 0

    if valid_branches:
        summary_line = (
            f"Valid dual-attempt branches: {len(valid_branches)}"
            if len(branch_outcomes) > 1
            else "Valid branch count: 1"
        )
    else:
        summary_line = "No valid train-passing branches."

    return normalized, top_level, summary_line


from dotenv import load_dotenv, find_dotenv

# Import shared event types
try:
    from .events import EventType, OrchestratorEvent, EventCallback
except ImportError:
    from events import EventType, OrchestratorEvent, EventCallback

# Load environment variables
load_dotenv(find_dotenv(usecwd=True))

# Configuration
DEFAULT_PUZZLE_PATH = "16b78196"
PHOENIX_PROJECT = "ARC_Athanor"

# Get API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

# Fireworks Anthropic-compatible endpoint base URL
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference"
FIREWORKS_REQUEST_TIMEOUT_SEC = 45.0
FIREWORKS_STREAM_MAX_RETRIES = 6
DEFAULT_STREAM_MAX_RETRIES = 3
MAX_CONSECUTIVE_TRANSIENT_TURN_FAILURES = 4
# Keep compression/reflection thinking modest for Kimi to reduce first-token latency.
KIMI_COMPRESSION_THINKING_BUDGET_CAP = 8192
ANTHROPIC_PROMPT_CACHING_BETA = "prompt-caching-2024-07-31"

# Model name mapping for Fireworks-hosted models
_FIREWORKS_MODEL_MAP = {
    "kimi-k2p5": "accounts/fireworks/models/kimi-k2p5",
    "kimi-k2.5": "accounts/fireworks/models/kimi-k2p5",
    "glm-5": "accounts/fireworks/models/glm-5",
}

KIMI_RUN_CODE_GUIDANCE_TEXT = (
    "Access train/test grids via `train_samples[idx]['input']`, "
    "`train_samples[idx]['output']`, and `test_samples[idx]['input']` in exploratory code tools. "
    "Use `run_code` to start fresh and `run_code_in_previous_runtime` to reuse the current live exploratory runtime when available. "
    "Tool results report whether the runtime was fresh, reused, or a fresh fallback. "
    "To save tokens, do not copy/paste full grid lists into exploratory code payloads."
)

EXPLORATORY_CODE_TOOL_NAMES = {"run_code", "run_code_in_previous_runtime"}

# Code execution timeout defaults and limits (seconds)
CODE_EXEC_DEFAULT_TIMEOUT = 20
CODE_EXEC_SOLUTION_DEFAULT_TIMEOUT = 60
CODE_EXEC_MAX_TIMEOUT = 120


def _kill_thread(thread: threading.Thread) -> None:
    """Best-effort attempt to raise TimeoutError in a running thread."""
    if thread.ident is None:
        return
    try:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread.ident),
            ctypes.py_object(TimeoutError),
        )
    except Exception:
        pass


def _run_with_timeout(func, timeout_seconds: float):
    """Run func() in a daemon thread with timeout.

    Returns (result, timed_out).  On timeout *result* is None.
    """
    result_box: list = [None]
    exc_box: list = [None]

    def _target():
        try:
            result_box[0] = func()
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout_seconds)

    if t.is_alive():
        _kill_thread(t)
        return None, True

    if exc_box[0] is not None:
        raise exc_box[0]
    return result_box[0], False


def _clamp_timeout(value, default: float, maximum: float = CODE_EXEC_MAX_TIMEOUT) -> float:
    """Parse and clamp an agent-supplied timeout value."""
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(1.0, min(v, maximum))


def _should_use_anthropic_prompt_caching(use_fireworks: bool) -> bool:
    """Only enable prompt caching on native Anthropic requests."""
    return not use_fireworks


def _cache_control(ttl: str) -> dict[str, str]:
    """Build Anthropic cache control metadata."""
    return {"type": "ephemeral", "ttl": ttl}


def _with_cache_control(block: dict, ttl: str) -> dict:
    """Clone a content/tool block and attach cache control metadata."""
    updated = dict(block)
    updated["cache_control"] = _cache_control(ttl)
    return updated


def _cached_system_prompt_payload(system_prompt: str, use_fireworks: bool):
    """Cache the stable system prompt with the default 5-minute TTL on native Anthropic."""
    if not _should_use_anthropic_prompt_caching(use_fireworks):
        return system_prompt
    return [{
        "type": "text",
        "text": str(system_prompt or ""),
        "cache_control": _cache_control("5m"),
    }]


def _cached_tools_payload(tools: list | None, use_fireworks: bool) -> list | None:
    """Cache the entire tool schema prefix by marking the last tool definition."""
    if not tools or not _should_use_anthropic_prompt_caching(use_fireworks):
        return tools
    cached_tools = copy.deepcopy(list(tools))
    cached_tools[-1] = _with_cache_control(cached_tools[-1], "5m")
    return cached_tools


def _cached_messages_payload(messages: list, use_fireworks: bool, ttl: str = "5m") -> list:
    """Advance the Anthropic cache breakpoint to the end of the current prompt prefix."""
    if not _should_use_anthropic_prompt_caching(use_fireworks):
        return messages

    cached_messages = copy.deepcopy(list(messages or []))
    cacheable_block_types = {"text", "image", "document", "tool_use", "tool_result"}

    for msg in range(len(cached_messages) - 1, -1, -1):
        content = cached_messages[msg].get("content")
        if isinstance(content, str):
            text = str(content or "")
            if text.strip():
                cached_messages[msg]["content"] = [{
                    "type": "text",
                    "text": text,
                    "cache_control": _cache_control(ttl),
                }]
                return cached_messages
            continue
        if not isinstance(content, list):
            continue
        for idx in range(len(content) - 1, -1, -1):
            block = content[idx]
            if not isinstance(block, dict):
                continue
            if str(block.get("type", "")) not in cacheable_block_types:
                continue
            content[idx] = _with_cache_control(block, ttl)
            return cached_messages
    return cached_messages


def _anthropic_prompt_cache_headers(use_fireworks: bool) -> dict | None:
    """Headers required for Anthropic prompt caching."""
    if not _should_use_anthropic_prompt_caching(use_fireworks):
        return None
    return {
        "anthropic-beta": ANTHROPIC_PROMPT_CACHING_BETA
    }


def _apply_anthropic_prompt_caching(
    api_params: dict,
    *,
    use_fireworks: bool,
    include_tools: bool = False,
) -> dict:
    """Attach Anthropic prompt-caching metadata to a request payload."""
    if not _should_use_anthropic_prompt_caching(use_fireworks):
        return api_params

    cached_params = dict(api_params)
    cached_params["system"] = _cached_system_prompt_payload(
        str(api_params.get("system", "") or ""),
        use_fireworks=use_fireworks,
    )
    cached_params["messages"] = _cached_messages_payload(
        list(api_params.get("messages") or []),
        use_fireworks=use_fireworks,
        ttl="5m",
    )
    if include_tools and api_params.get("tools"):
        cached_params["tools"] = _cached_tools_payload(
            list(api_params.get("tools") or []),
            use_fireworks=use_fireworks,
        )
    headers = _anthropic_prompt_cache_headers(use_fireworks)
    if headers:
        merged_headers = dict(api_params.get("extra_headers") or {})
        merged_headers.update(headers)
        cached_params["extra_headers"] = merged_headers
    return cached_params


def _is_fireworks_model(model_name: str) -> bool:
    """Check if the model should be routed through Fireworks."""
    lowered = str(model_name or "").lower()
    return (
        lowered.startswith("kimi")
        or "/kimi" in lowered
        or lowered.startswith("glm")
        or "/glm" in lowered
    )


def _is_glm_model(model_name: str) -> bool:
    """Check if model is GLM family (currently treated as text-only)."""
    lowered = str(model_name or "").lower()
    return lowered.startswith("glm") or "/glm" in lowered

def _resolve_fireworks_model(model_name: str) -> str:
    """Map a short model name to the Fireworks model identifier."""
    return _FIREWORKS_MODEL_MAP.get(model_name.lower(), f"accounts/fireworks/models/{model_name}")


def _fireworks_timeout():
    """Conservative timeout for Fireworks requests so stop/cancel remains responsive."""
    if httpx is not None:
        return httpx.Timeout(
            connect=10.0,
            read=FIREWORKS_REQUEST_TIMEOUT_SEC,
            write=30.0,
            pool=30.0,
        )
    return FIREWORKS_REQUEST_TIMEOUT_SEC


def _max_retries_for_model(is_fireworks: bool) -> int:
    """Use a higher retry budget for Fireworks transport instability."""
    return FIREWORKS_STREAM_MAX_RETRIES if is_fireworks else DEFAULT_STREAM_MAX_RETRIES


def _error_payload_text(value) -> str:
    """Best-effort stringification for provider error payload inspection."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _is_transient_api_error(err: Exception) -> bool:
    """Classify retryable transport/provider errors."""
    transport_types = tuple(
        t
        for t in [
            getattr(httpx, "RemoteProtocolError", None),
            getattr(httpcore, "RemoteProtocolError", None),
            getattr(httpx, "ReadError", None),
            getattr(httpx, "ConnectError", None),
            getattr(httpx, "ReadTimeout", None),
            getattr(httpx, "TimeoutException", None),
        ]
        if t is not None
    )
    if transport_types and isinstance(err, transport_types):
        return True

    # Anthropic SDK wraps httpx errors into API-level exceptions.
    api_timeout_cls = getattr(anthropic, "APITimeoutError", None) if anthropic is not None else None
    api_conn_cls = getattr(anthropic, "APIConnectionError", None) if anthropic is not None else None
    api_status_cls = getattr(anthropic, "APIStatusError", None) if anthropic is not None else None

    if api_timeout_cls is not None and isinstance(err, api_timeout_cls):
        return True
    if api_conn_cls is not None and isinstance(err, api_conn_cls):
        return True
    if api_status_cls is not None and isinstance(err, api_status_cls):
        code = getattr(err, "status_code", None)
        try:
            status = int(code)
        except Exception:
            status = None
        if status in {408, 409, 429} or (status is not None and status >= 500):
            return True
        payload_text = " ".join(
            part
            for part in [
                _error_payload_text(getattr(err, "body", None)),
                _error_payload_text(getattr(err, "response", None)),
                _error_payload_text(getattr(err, "message", None)),
                _error_payload_text(getattr(err, "args", None)),
                str(err),
            ]
            if part
        ).lower()
        transient_markers = (
            "overloaded_error",
            "overloaded",
            "rate_limit_error",
            "rate limit",
            "temporarily unavailable",
            "service unavailable",
            "unavailable_error",
            "server overloaded",
        )
        if any(marker in payload_text for marker in transient_markers):
            return True

    return False


def _append_kimi_run_code_guidance_if_needed(content_blocks: list, model_name: str) -> list:
    """Append one Kimi-specific run_code efficiency reminder per context window."""
    if not _is_fireworks_model(str(model_name or "")) or _is_glm_model(str(model_name or "")):
        return content_blocks
    blocks = list(content_blocks or [])
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = str(block.get("text", "") or "")
            if KIMI_RUN_CODE_GUIDANCE_TEXT in text:
                return blocks
    blocks.append({"type": "text", "text": KIMI_RUN_CODE_GUIDANCE_TEXT})
    return blocks


def _resolve_task_path_if_needed(
    puzzle_path: str,
    dataset_root: str | None,
    dataset_split: str,
) -> str:
    """Resolve a task id to absolute JSON path when needed."""
    raw = str(puzzle_path or "").strip()
    if raw.endswith(".json") or "/" in raw:
        return raw

    try:
        from ..data import resolve_task_path
    except Exception:
        resolve_task_path = None

    if resolve_task_path is None:
        return raw

    resolved = resolve_task_path(
        task=raw,
        split=dataset_split,
        dataset_root=dataset_root,
    )
    return str(resolved)


def _strip_test_outputs(samples: list[dict]) -> list[dict]:
    """Expose test inputs to executed code without leaking test outputs."""
    sanitized = []
    for sample in samples or []:
        if not isinstance(sample, dict):
            continue
        if "input" in sample:
            sanitized.append({"input": sample.get("input")})
    return sanitized


def load_puzzle(puzzle_path: str) -> dict:
    """Load puzzle from JSON file."""
    with open(puzzle_path, 'r') as f:
        return json.load(f)


def format_puzzle_for_prompt(puzzle_data: dict, use_vision: bool = True) -> list:
    """Format puzzle data into multimodal prompt parts for Anthropic.

    Args:
        puzzle_data: Dictionary with 'train' and 'test' keys
        use_vision: Whether to include image rendering (for multimodal models)

    Returns:
        List of content blocks for Anthropic message
    """
    content_blocks = []
    content_blocks.append(
        {"type": "text", "text": "Analyze this ARC puzzle:\n\n"})

    # Training examples
    for i, example in enumerate(puzzle_data.get("train", [])):
        input_grid = example["input"]
        output_grid = example["output"]

        content_blocks.append(
            {"type": "text", "text": f"**Training Example {i}:**\nInput: {input_grid}\n\n"})

        if use_vision:
            img_base64 = render_grid_to_base64(input_grid)
            if img_base64:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64
                    }
                })

        content_blocks.append(
            {"type": "text", "text": f"\nOutput: {output_grid}\n\n"})

        if use_vision:
            img_base64 = render_grid_to_base64(output_grid)
            if img_base64:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64
                    }
                })

        content_blocks.append({"type": "text", "text": "\n"})

    # Test examples
    for i, example in enumerate(puzzle_data.get("test", [])):
        input_grid = example["input"]
        content_blocks.append(
            {"type": "text", "text": f"**Test Example {i}:**\n"})
        content_blocks.append(
            {"type": "text", "text": f"Input: {input_grid}\n\n"})

        if use_vision:
            img_base64 = render_grid_to_base64(input_grid)
            if img_base64:
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64
                    }
                })

        if i < len(puzzle_data.get("test", [])) - 1:
            content_blocks.append({"type": "text", "text": "\n"})

    return content_blocks


def _oi_sanitize_content_blocks(blocks: list) -> list:
    out = []
    for b in blocks or []:
        if isinstance(b, dict) and b.get("type") == "image":
            src = b.get("source") or {}
            data = src.get("data")
            if isinstance(data, str) and data:
                if len(data) <= 150000:
                    safe_data = f"data:image/png;base64,{data}"
                else:
                    safe_data = f"<base64:len={len(data)}>"
            else:
                safe_data = ""
            out.append({
                "type": "image",
                "source": {
                    "type": src.get("type"),
                    "media_type": src.get("media_type"),
                    "data": safe_data,
                },
            })
        else:
            out.append(b)
    return out


def _oi_set_input_messages(phoenix, system_prompt: str, messages_to_use: list):
    i = 0
    phoenix.set_attribute(f"llm.input_messages.{i}.message.role", "system")
    phoenix.set_attribute(
        f"llm.input_messages.{i}.message.content", system_prompt)
    i += 1

    for msg in messages_to_use or []:
        role = msg.get("role", "")
        content = msg.get("content")
        phoenix.set_attribute(f"llm.input_messages.{i}.message.role", role)
        if isinstance(content, str):
            phoenix.set_attribute(
                f"llm.input_messages.{i}.message.content", content)
        elif isinstance(content, list):
            blocks = _oi_sanitize_content_blocks(content)
            for j, b in enumerate(blocks):
                if not isinstance(b, dict):
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.type", "text")
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.text", str(b))
                    continue
                btype = b.get("type")
                if btype == "text":
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.type", "text")
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.text", b.get("text", ""))
                elif btype == "image":
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.type", "image")
                    url = (b.get("source") or {}).get("data", "")
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.image.image.url", url)
                elif btype == "tool_result":
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.type", "text")
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.text", json.dumps(b, default=str))
                else:
                    # For thinking, tool_use, and other blocks: pass through as JSON
                    # so Phoenix can render them with its native UI
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.type", "text")
                    phoenix.set_attribute(
                        f"llm.input_messages.{i}.message.contents.{j}.message_content.text", json.dumps(b, default=str))
        else:
            phoenix.set_attribute(
                f"llm.input_messages.{i}.message.content", str(content))
        i += 1


def _oi_set_output_messages(phoenix, aggregated_text: str, tool_uses: list):
    phoenix.set_attribute("llm.output_messages.0.message.role", "assistant")
    phoenix.set_attribute(
        "llm.output_messages.0.message.content", aggregated_text or "")

    for k, tu in enumerate(tool_uses or []):
        tid = tu.get("id", "")
        name = tu.get("name", "")
        args = tu.get("input", {})
        try:
            arg_str = json.dumps(args, default=str)
        except Exception:
            arg_str = str(args)
        phoenix.set_attribute(
            f"llm.output_messages.0.message.tool_calls.{k}.tool_call.id", tid)
        phoenix.set_attribute(
            f"llm.output_messages.0.message.tool_calls.{k}.tool_call.function.name", name)
        phoenix.set_attribute(
            f"llm.output_messages.0.message.tool_calls.{k}.tool_call.function.arguments", arg_str)



def get_tool_schemas() -> list:
    """Return the JSON schemas for tools exposed to the model.

    This is the CANONICAL tool schema definition.
    orchestrator_core.get_tool_schemas() delegates to this.
    """
    return [
        {
            "name": "submit_transform_hypothesis",
            "description": "REQUIRED before calling execute_python_solution. Submit a detailed natural language specification of the single candidate-generating transformation rule you believe applies. The hypothesis should explain the shared rule, any genuine unresolved ambiguity, how candidate behavior 1 differs from candidate behavior 2 if present, and the expected consequences for the test inputs. This will be recorded and sent to an independent reviewer once the candidate generator is ready. Write it as if explaining to a programmer who must reimplement solve() from scratch using only your description.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "hypothesis": {
                        "type": "string",
                        "description": "Exhaustive natural language specification of the candidate generator. Include: (1) high-level summary, (2) step-by-step algorithm, (3) all edge cases and conditional logic, (4) how the rule generalizes beyond training examples, and (5) if ambiguity remains, how candidate output 1 differs from candidate output 2. A competent programmer should be able to fully reconstruct solve() from this description alone."
                    },
                    "reviewer_response": {
                        "type": "string",
                        "description": "Optional response to prior reviewer feedback. After a rejection or expansion request, use this to acknowledge valid concerns and explain how you addressed them, and to challenge any specific claims you believe are incorrect, citing evidence from your analysis of the training examples."
                    },
                },
                "required": ["hypothesis"],
            }
        },
        {
            "name": "execute_python_solution",
            "description": "Run a COMPLETE self-contained Python implementation that defines solve(grid). The same solve(grid) must be used for all training and test inputs. It may return either one output grid or a list of up to two candidate output grids for a given input. You MUST call submit_transform_hypothesis first in the same turn and the code must be self-contained.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete self-contained Python code with a solve(grid) function. solve(grid) may return either a single grid or a list of one or two candidate grids derived from the input."
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum execution time in seconds (default 60, max 120). Covers both code definition and running solve() on all samples."
                    },
                },
                "required": ["code"],
            }
        },
        {
            "name": "run_code",
            "description": "Run an isolated Python snippet for quick testing/debugging. Each call starts a fresh exploratory runtime and discards any previously live exploratory runtime. Available globals: train_samples = list of {'input': grid, 'output': grid}; test_samples = list of {'input': grid}. Tool results report whether the runtime was fresh, reused, or a fresh fallback, along with runtime id and step metadata. IMPORTANT: Keep stdout token-efficient — convert NumPy types before printing. Use .tolist() for arrays/grids and int()/float() for scalars.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum execution time in seconds (default 20, max 120)."
                    }
                },
                "required": ["code"]
            }
        },
        {
            "name": "run_code_in_previous_runtime",
            "description": "Run Python code inside the current live exploratory runtime created by the most recent run_code or run_code_in_previous_runtime call. This inherits previously defined helpers and variables from that live runtime. If no live exploratory runtime exists, the tool falls back to starting a fresh exploratory runtime and reports clearly that reuse did not happen. Tool results report whether the runtime was fresh, reused, or a fresh fallback, along with runtime id and step metadata. The live exploratory runtime does not survive context compression, saved-run reload, backend restart, or a later fresh run_code call. IMPORTANT: Keep stdout token-efficient — convert NumPy types before printing. Use .tolist() for arrays/grids and int()/float() for scalars.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum execution time in seconds (default 20, max 120)."
                    }
                },
                "required": ["code"]
            }
        }
    ]


def load_system_prompt() -> str:
    """Load the solver system prompt."""
    prompt_path = Path(__file__).parent / "SOLVER_SYSTEM_PROMPT.md"
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return "You are an expert ARC puzzle solver."


def default_cli_callback(event: OrchestratorEvent):
    """Default callback that prints events to console (CLI mode)."""
    if event.type == EventType.SYSTEM:
        print(event.content)
    elif event.type == EventType.ERROR:
        print(f"ERROR: {event.content}")
    elif event.type == EventType.THINKING:
        sys.stdout.write(f"\033[90m{event.content}\033[0m")
        sys.stdout.flush()
    elif event.type == EventType.TEXT:
        sys.stdout.write(event.content)
        sys.stdout.flush()
    elif event.type == EventType.TOOL_CALL:
        print(f"\n🔧 TOOL CALL: {event.metadata.get('name', 'unknown')}")
    elif event.type == EventType.TOOL_RESULT:
        output = event.metadata.get('output', '')
        print(f"   -> Output: {len(output)} chars")
    elif event.type == EventType.TURN_START:
        turn = event.metadata.get('turn', 0)
        iteration = event.metadata.get('iteration', 0)
        max_iterations = event.metadata.get('max_iterations', 10)
        print(f"\n{'='*80}")
        print(f"🔄 Turn {turn} (Iteration {iteration}/{max_iterations})")
        print(f"{'='*80}")
    elif event.type == EventType.REFLECTION:
        print(f"   🔒 {event.content}")


def _truncate_tool_output(text: str, max_chars: int = 5000) -> str:
    """Truncate tool output to prevent context overflow.

    Args:
        text: Tool output text
        max_chars: Maximum characters to keep (default: 5000, ~1250 tokens)

    Returns:
        Truncated text with notification if truncated
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    remaining = len(text) - max_chars
    return f"{truncated}\n\n[... OUTPUT TRUNCATED: {remaining:,} additional characters omitted to prevent context overflow. The complete output was too large to include in the conversation.]"


# Context compression constants
CONTEXT_COMPRESSION_THRESHOLD = 170000
CHARS_PER_TOKEN_ESTIMATE = 3.3  # Mixed code/prose; 4.0 was too generous and caused under-counting
IMAGE_TOKENS_ESTIMATE = 500    # ARC grids at 24px/cell: 192-480px → fits in 1 Anthropic tile (~255 tokens); use 500 as conservative ceiling
TOOL_SCHEMA_TOKENS_OVERHEAD = 3000  # 3 tool definitions with descriptions (~3K tokens)


def _estimate_message_tokens(messages: list, system_prompt: str = "",
                             include_tool_overhead: bool = True) -> int:
    """Estimate token count for a full API request.

    Counts system prompt + all message content (text, images, tool_use,
    tool_result, thinking blocks) + tool-definition overhead.
    """
    total_chars = len(system_prompt)
    image_count = 0

    def _count_blocks(blocks: list):
        nonlocal total_chars, image_count
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                total_chars += len(block.get("text", ""))
            elif btype == "thinking":
                total_chars += len(block.get("thinking", ""))
            elif btype == "image":
                image_count += 1
            elif btype == "tool_use":
                total_chars += len(json.dumps(block.get("input", {})))
                total_chars += len(block.get("name", "")) + 20  # name + framing
            elif btype == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, str):
                    total_chars += len(rc)
                elif isinstance(rc, list):
                    _count_blocks(rc)

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            _count_blocks(content)

    text_tokens = int(total_chars / CHARS_PER_TOKEN_ESTIMATE)
    image_tokens = image_count * IMAGE_TOKENS_ESTIMATE
    tool_tokens = TOOL_SCHEMA_TOKENS_OVERHEAD if include_tool_overhead else 0

    return text_tokens + image_tokens + tool_tokens


def _extract_usage_meta(usage) -> dict:
    """Extract token usage metadata from an Anthropic API usage object.

    Returns dict with input_tokens, output_tokens, and optionally reasoning_tokens.
    """
    def _read_int(paths: list[str]) -> int:
        for path in paths:
            try:
                cursor = usage
                for part in path.split("."):
                    cursor = getattr(cursor, part, None)
                value = int(cursor or 0)
                if value > 0:
                    return value
            except Exception:
                continue
        return 0

    meta = {
        # Anthropic-native + provider-compat fallbacks
        "input_tokens": _read_int(
            ["input_tokens", "prompt_tokens", "prompt_token_count"]
        ),
        "output_tokens": _read_int(
            ["output_tokens", "completion_tokens", "completion_token_count", "generated_tokens"]
        ),
        "cache_creation_input_tokens": _read_int(
            ["cache_creation_input_tokens"]
        ),
        "cache_read_input_tokens": _read_int(
            ["cache_read_input_tokens"]
        ),
        # Anthropic bills thinking/tool-use generation inside output_tokens.
        "output_tokens_include_reasoning": True,
    }
    meta["billed_input_tokens"] = (
        int(meta["input_tokens"])
        + int(meta["cache_creation_input_tokens"])
        + int(meta["cache_read_input_tokens"])
    )

    # Try multiple extraction paths for reasoning/thinking tokens
    extraction_paths = [
        ("output_tokens_details.reasoning_tokens", lambda u: getattr(
            getattr(u, "output_tokens_details", None), "reasoning_tokens", 0)),
        ("output_tokens_details.thinking_tokens", lambda u: getattr(
            getattr(u, "output_tokens_details", None), "thinking_tokens", 0)),
        ("reasoning_tokens", lambda u: getattr(u, "reasoning_tokens", 0)),
        ("thinking_tokens", lambda u: getattr(u, "thinking_tokens", 0)),
    ]

    for _path_name, path_fn in extraction_paths:
        try:
            tokens = int(path_fn(usage) or 0)
            if tokens > 0:
                meta["reasoning_tokens"] = tokens
                meta["reasoning_tokens_reported"] = True
                break
        except Exception:
            continue

    meta.setdefault("reasoning_tokens_reported", False)

    return meta




@contextlib.contextmanager
def _suppress_auto_instrumentation():
    """Suppress OpenTelemetry SDK auto-instrumentation within this block.

    Used when we already have a manual MessagesStream span with correct
    attributes — prevents the Anthropic SDK from creating a duplicate
    inner MessagesStream span with less accurate data.
    """
    if _otel_suppress_available:
        token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
        try:
            yield
        finally:
            detach(token)
    else:
        yield


def _sanitize_messages(
    messages: list,
    image_source_mode: str = "base64",
    force_png_reencode: bool = False,
) -> list:
    """Remove empty text content blocks from messages before sending to API.

    Anthropic API requires all text content blocks to be non-empty.
    This defensively filters out any empty text blocks from all messages.

    Args:
        messages: Conversation messages payload
        image_source_mode: "base64" (default) or "url_data_uri" for providers
            that prefer URL image sources with data URIs.
    """
    def _normalize_base64_data(raw: str) -> str | None:
        data = str(raw or "").strip()
        if not data:
            return None
        if data.startswith("data:") and "," in data:
            data = data.split(",", 1)[1]
        # Remove accidental whitespace/newlines introduced during serialization.
        data = "".join(data.split())
        # Fix missing padding when possible.
        padding = len(data) % 4
        if padding:
            data += "=" * (4 - padding)
        try:
            base64.b64decode(data, validate=True)
        except Exception:
            return None
        return data

    def _reencode_png_for_compat(normalized_base64: str, media_type: str) -> tuple[str, str]:
        """Re-encode image to a conservative PNG form for provider compatibility.

        Some providers (observed with Fireworks Kimi) intermittently reject valid PNG
        streams with `Incorrect padding`. Re-encoding to uncompressed PNG
        (`compress_level=0`) avoids those decode failures in practice.
        """
        if not force_png_reencode:
            return normalized_base64, media_type
        try:
            from PIL import Image
        except Exception:
            return normalized_base64, media_type
        try:
            raw = base64.b64decode(normalized_base64, validate=True)
            with Image.open(io.BytesIO(raw)) as img:
                rgb = img.convert("RGB")
                out = io.BytesIO()
                rgb.save(out, format="PNG", compress_level=0)
            return base64.b64encode(out.getvalue()).decode("ascii"), "image/png"
        except Exception:
            return normalized_base64, media_type

    def _sanitize_blocks(blocks: list) -> list:
        filtered_content: list = []
        for block in blocks:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if not text or not str(text).strip():
                        continue
                    filtered_content.append(block)
                    continue
                if btype == "image":
                    source = block.get("source")
                    if not isinstance(source, dict):
                        continue
                    normalized = _normalize_base64_data(source.get("data", ""))
                    if not normalized:
                        continue
                    safe_block = dict(block)
                    safe_source = dict(source)
                    media_type = str(safe_source.get("media_type", "image/png"))
                    normalized, media_type = _reencode_png_for_compat(
                        normalized_base64=normalized,
                        media_type=media_type,
                    )
                    if image_source_mode == "url_data_uri":
                        safe_source = {
                            "type": "url",
                            "url": f"data:{media_type};base64,{normalized}",
                        }
                    else:
                        safe_source["type"] = "base64"
                        safe_source["media_type"] = media_type
                        safe_source["data"] = normalized
                    safe_block["source"] = safe_source
                    filtered_content.append(safe_block)
                    continue
                if btype == "tool_result":
                    content = block.get("content")
                    if isinstance(content, list):
                        nested = _sanitize_blocks(content)
                        safe_block = dict(block)
                        safe_block["content"] = nested
                        filtered_content.append(safe_block)
                        continue
            filtered_content.append(block)
        return filtered_content

    sanitized = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            filtered_content = _sanitize_blocks(content)
            # If all blocks were filtered out, add a placeholder
            if not filtered_content:
                filtered_content = [{"type": "text", "text": "(continued)"}]
            sanitized.append({**msg, "content": filtered_content})
        elif isinstance(content, str) and not content.strip():
            sanitized.append({**msg, "content": "(continued)"})
        else:
            sanitized.append(msg)
    return sanitized


def _extract_reflection_summary(messages: list, max_chars: int = 50000) -> str:
    """Extract agent's reflection text from the last assistant response.

    This reflection is the model's ONLY context for the next iteration,
    so we preserve it with minimal truncation.

    Args:
        messages: Full message history
        max_chars: Maximum characters to keep (default 50000 — generous to avoid losing insights)

    Returns:
        Extracted reflection prose (code blocks filtered out)
    """
    import re

    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", [])
            if isinstance(content, str):
                text_parts = [content]
            elif isinstance(content, list):
                text_parts = [b.get("text", "") for b in content if isinstance(
                    b, dict) and b.get("type") == "text"]
            else:
                continue

            # Filter out code blocks, keep prose
            prose_parts = []
            for text in text_parts:
                # Remove fenced code blocks
                cleaned = re.sub(r'```[\s\S]*?```', '', text)
                # Remove inline code
                cleaned = re.sub(r'`[^`]+`', '', cleaned)
                cleaned = cleaned.strip()
                if cleaned:
                    prose_parts.append(cleaned)

            if prose_parts:
                combined = "\n".join(prose_parts)
                if len(combined) > max_chars:
                    return combined[:max_chars] + "\n[... reflection truncated]"
                return combined

    return ""



def _clone_api_messages(messages: list[dict]) -> list[dict]:
    """Deep-copy Anthropic-style messages into JSON-safe structures."""
    return numpy_to_python(copy.deepcopy(messages or []))


def _clone_prompt_blocks(blocks: list[dict]) -> list[dict]:
    """Deep-copy user prompt blocks into JSON-safe structures."""
    return numpy_to_python(copy.deepcopy(blocks or []))


def _iteration_context_metadata(
    *,
    iteration: int,
    ui_title: str,
    context_mode: str,
    prompt_blocks: list[dict] | None = None,
    snapshot_messages: list[dict] | None = None,
    resume_messages: list[dict] | None = None,
) -> dict:
    """Build metadata for iteration-context canvases in the web UI."""
    meta = {
        "iteration": iteration,
        "consolidated_prompt": True,
        "ui_title": ui_title,
        "context_mode": context_mode,
    }
    if prompt_blocks is not None:
        meta["prompt_blocks"] = _clone_prompt_blocks(prompt_blocks)
    if snapshot_messages is not None:
        meta["snapshot_messages"] = _clone_api_messages(snapshot_messages)
    if resume_messages is not None:
        meta["resume_messages"] = _clone_api_messages(resume_messages)
    return meta


def _set_reflection_submode(
    *,
    test_generalization: bool,
    reflector_reject_compression: bool,
) -> tuple[bool, bool]:
    """Keep reflection submodes mutually exclusive."""
    if test_generalization and reflector_reject_compression:
        raise ValueError("Reflection submodes must be mutually exclusive")
    return test_generalization, reflector_reject_compression


def _normalize_reflector_text(value) -> str:
    return str(value or "").strip()


def _normalize_reflector_list(values) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        text = _normalize_reflector_text(value)
        if text:
            normalized.append(text)
    return normalized


_REFLECTOR_STRUCTURED_HEADERS = (
    "PHASE_1_SCORE:",
    "PHASE_2_SCORE:",
    "PHASE_3_SCORE:",
    "CONFIDENCE:",
    "VERDICT:",
    "FINDINGS:",
    "CONCERNS:",
    "APPROVED_PRESENTED_BRANCHES:",
    "REJECTED_PRESENTED_BRANCHES:",
    "SUGGESTED_TARGET_BRANCH_ID:",
    "AMBIGUITY_REASON:",
    "PROPOSED_BRANCH_HYPOTHESIS:",
    "PROPOSED_BRANCH_PSEUDOCODE:",
    "BRANCH_DIFFERENTIATOR:",
    "EXPANSION_VALIDATION_CHECKS:",
    "REJECTION_REASON:",
    "ROOT_CAUSE:",
    "SUGGESTED_FOCUS:",
    "REPAIR_PLAN:",
    "VALIDATION_CHECKS:",
)



def _build_reflector_reject_feedback(reflector_response: str, *, bypass_compression: bool = False) -> str:
    """Quote the reflector's full rejection verbatim with a short directive."""
    if bypass_compression:
        closing = (
            "Critically evaluate the reviewer's analysis. "
            "The reviewer works from artifacts only and can make mistakes. "
            "Note which concerns seem valid and which you plan to challenge, "
            "then continue solving with your revised approach."
        )
    else:
        closing = (
            "Critically evaluate the reviewer's analysis as you write your checkpoint. "
            "The reviewer works from artifacts only and can make mistakes. "
            "Note which concerns seem valid and which you plan to challenge "
            "— you will have full tool access in the next iteration to verify empirically."
        )
    return f"""\
An independent reviewer has carefully studied your solution and rejected it. Here is the reviewer's full analysis:

---

{reflector_response}

---

{closing}
"""


def _build_consolidated_summary(
    iteration_history: list,
    last_reflection: str,
    last_failed_results: dict,
    last_visual_parts: list,
    verified_rules: list = None,
    use_visual_mode: bool = True,
    retry_after_pass: bool = False,
    last_test_visual_parts: list = None,
    reflector_rejection: str = "",
) -> list:
    """Build consolidated summary content for new iteration.

    Instead of full chat history, provides structured summary of:
    - Last submitted code and hypothesis (NO TRUNCATION)
    - Failure details with visuals
    - Agent's reflection (NO TRUNCATION)
    - Verified rules (invariant observations)
    - Simplified graveyard (1-line per attempt)

    Args:
        iteration_history: List of iteration records
        last_reflection: Extracted reflection text from agent
        last_failed_results: Results dict from last execute_python_solution
        last_visual_parts: Visual feedback parts (images) from last failure
        verified_rules: List of verified invariant observations
        use_visual_mode: Whether to include images

    Returns:
        List of content blocks for consolidated summary message
    """
    if not iteration_history:
        return []

    current = iteration_history[-1]
    content_blocks = []

    # Guiding prompt
    if retry_after_pass:
        guiding_text = (
            "Your last submitted code passed all training examples, but was rejected after review. "
            "Below is your last hypothesis, code, test predictions, the independent reviewer's analysis, "
            "and your memory checkpoint from the previous attempt. "
            "Pick up from where you left off."
        )
    else:
        guiding_text = "Your last submitted code failed some training examples."

    if guiding_text.strip():  # Ensure non-empty text
        content_blocks.append({
            "type": "text",
            "text": guiding_text + "\n"
        })

    # Previous code (NO TRUNCATION)
    if current.get('code') and current['code'].strip():
        content_blocks.append({
            "type": "text",
            "text": f"---\n### Last Submitted Code\n```python\n{current['code']}\n```\n"
        })

    # Last submitted hypothesis (always show — agent's memory checkpoint may annotate but not restate it)
    if current.get('transform_hypothesis') and current['transform_hypothesis'].strip():
        content_blocks.append({
            "type": "text",
            "text": f"---\n### Last Transform Hypothesis\n{current['transform_hypothesis']}\n"
        })

    # Results summary (only show error if present)
    train_results = current.get('train_results', [])
    failed_indices = [r['index']
                      for r in train_results if not r.get('correct')]
    pixel_acc = current.get('pixel_accuracy', 0.0)

    if current.get('error') and current['error'].strip():
        content_blocks.append({
            "type": "text",
            "text": f"---\n### Error\n{current['error']}\n"
        })

    # Failure details with visuals
    if last_visual_parts and use_visual_mode:
        failure_details_text = "---\n### Failure Details\nBelow are the expected vs predicted outputs for failed examples:\n"
        if failure_details_text.strip():  # Ensure non-empty text
            content_blocks.append({
                "type": "text",
                "text": failure_details_text
            })
        content_blocks.extend(last_visual_parts)

    # Test predictions (when retrying after 100% train accuracy)
    if retry_after_pass and last_test_visual_parts:
        test_predictions_text = "---\n### Test Predictions\nBelow are your predicted outputs for the test inputs that triggered the retry decision:\n"
        if test_predictions_text.strip():  # Ensure non-empty text
            content_blocks.append({
                "type": "text",
                "text": test_predictions_text
            })
        content_blocks.extend(last_test_visual_parts)

    # Reflector feedback (rejection or expansion guidance from previous iteration)
    if reflector_rejection and reflector_rejection.strip():
        content_blocks.append({
            "type": "text",
            "text": (
                "---\n### Independent Reviewer Analysis\n"
                "An independent reviewer studied your submission and provided feedback. "
                "Critically evaluate this analysis — the reviewer works from artifacts only and can make mistakes.\n\n"
                f"{reflector_rejection}\n"
            )
        })

    # Agent's reflection / memory checkpoint (NO TRUNCATION)
    if last_reflection and last_reflection.strip():
        reflection_label = "Memory Checkpoint" if retry_after_pass else "Your Reflection"
        content_blocks.append({
            "type": "text",
            "text": f"---\n### {reflection_label}\n{last_reflection}\n"
        })

    # Verified rules (invariant observations that persist across iterations)
    if verified_rules:
        rules_text = "---\n### Verified Rules (Invariant Observations)\n"
        rules_text += "The following observations have been verified across examples:\n"
        for rule in verified_rules:
            rules_text += f"- {rule}\n"
        if rules_text.strip():  # Ensure non-empty text
            content_blocks.append({"type": "text", "text": rules_text})

    # Closing instruction
    if reflector_rejection:
        closing_text = (
            "---\n"
            "Verify the reviewer's specific claims using `run_code`, then submit your updated solution. "
            "In your next `submit_transform_hypothesis`, use the `reviewer_response` parameter to briefly "
            "state which concerns you addressed and which you dispute with evidence."
        )
    else:
        closing_text = "---\nCheck your reflection and proceed with your refined approach. Use `run_code` or `run_code_in_previous_runtime` for experiments, or `execute_python_solution` when you feel confident about the transform rule."
    if closing_text.strip():  # Ensure non-empty text
        content_blocks.append({
            "type": "text",
            "text": closing_text
        })

    return content_blocks


def run_orchestration(
    puzzle_path: str,
    model_name: str = "claude-opus-4-6",
    use_streaming: bool = True,
    use_visual_mode: bool = True,
    use_extended_thinking: bool = True,
    thinking_budget: int = 16000,
    thinking_effort: str = "medium",
    reflection_thinking_effort: str = "max",
    compression_thinking_effort: str = "max",
    max_iterations: int = 10,
    max_turns: int = 200,
    compression_threshold: int = 170000,
    compression_bypass_threshold: int = 120000,
    max_test_predictions: int = 2,
    emit_tool_call_deltas: bool = False,
    event_callback: Optional[EventCallback] = None,
    phoenix=None,
    should_stop: Optional[Callable[[], bool]] = None,
    stop_reason: Optional[Callable[[], str | None]] = None,
    initial_messages: Optional[list] = None,
    initial_iteration: int = 0,
    initial_turn: int = 0,
    initial_in_reflection_mode: bool = False,
    initial_in_test_generalization_reflection: bool = False,
    initial_in_reflector_reject_compression: bool = False,
    enable_independent_reflector: bool = True,
    reflector_provider: str = "gemini",
    reflector_model: str | None = None,
    reflector_thinking_effort: str = "high",
    reflector_code_execution: bool = False,
    semi_cot_first_turn: bool = False,
    semi_cot_thinking_effort: str = "high",
    enable_phoenix: bool = True,
    unsafe_local_exec: bool = False,
    dataset_root: str | None = None,
    dataset_split: str = "public_eval",
    initial_reflector_message_history: list[dict] | None = None,
    initial_reflector_response: str = "",
    initial_test_candidates: list[dict] | None = None,
) -> dict:
    """
    Run the full ARC solver orchestration loop.

    This is the CANONICAL orchestration implementation.
    orchestrator_core.run_orchestration() delegates to this.

    Args:
        puzzle_path: Path to the puzzle JSON file
        model_name: Anthropic model to use
        use_streaming: Whether to use streaming API
        use_visual_mode: Whether to include images
        use_extended_thinking: Whether to enable extended thinking
        thinking_budget: Token budget for thinking (default 16000, max for Haiku 4.5)
        thinking_effort: Thinking effort level for main turns ("low", "medium", "high", "max")
        reflection_thinking_effort: Thinking effort for reflection turns (default "max")
        compression_thinking_effort: Thinking effort for compression turns (default "max")
        max_iterations: Maximum solve iterations
        max_turns: Maximum API turns
        compression_threshold: Token threshold for automatic context compression (default 170000)
        max_test_predictions: Maximum distinct test-output candidates to keep (1 or 2)
        emit_tool_call_deltas: Emit partial TOOL_CALL updates while tool JSON is still streaming
        event_callback: Callback function for events (default: CLI print)
        phoenix: Phoenix observability instance (optional, will auto-init if None)
        should_stop: Callback to check if orchestration should stop
        stop_reason: Optional callback that returns the reason for a stop request
        enable_phoenix: Enable Phoenix tracing instrumentation
        unsafe_local_exec: Whether model-generated Python execution is allowed
        dataset_root: Dataset root used to resolve task IDs
        dataset_split: Dataset split used when puzzle_path is a task ID
        initial_messages: Pre-built API messages for checkpoint resume (optional).
            If provided, these replace the default initial [user_prompt] messages.
            Must follow Anthropic's alternating user/assistant format.
        initial_iteration: Starting iteration number for checkpoint resume (default 0).
        initial_turn: Starting turn number for checkpoint resume (default 0).
        initial_in_reflection_mode: Whether the checkpoint ended at a reflection prompt (default False).
        initial_in_test_generalization_reflection: Whether the checkpoint ended at test-generalization reflection.
        initial_in_reflector_reject_compression: Whether the checkpoint ended at reflector-reject compression.

    Returns:
        dict with 'solved', 'iterations', 'final_code', etc.
    """
    # Use default CLI callback if none provided
    if event_callback is None:
        event_callback = default_cli_callback

    try:
        max_test_predictions = max(1, min(2, int(max_test_predictions or 2)))
    except Exception:
        max_test_predictions = 2

    if should_stop is None:
        def should_stop(): return False

    def emit(event_type: EventType, content: str = "", metadata: dict = None, images: list = None):
        event_callback(OrchestratorEvent(
            type=event_type,
            content=content,
            metadata=metadata or {},
            images=images or []
        ))

    if anthropic is None:
        emit(EventType.ERROR,
             "anthropic package is not installed. Install with: pip install anthropic")
        emit(EventType.COMPLETE, "No solution", {"solved": False})
        return {"solved": False, "error": "missing_anthropic_package"}

    # Resolve task ID to path when needed
    puzzle_path = _resolve_task_path_if_needed(
        puzzle_path=puzzle_path,
        dataset_root=dataset_root,
        dataset_split=dataset_split,
    )
    emit(EventType.SYSTEM, f"📋 Task path resolved: {puzzle_path}")

    emit(EventType.SYSTEM, f"📂 Loading puzzle: {puzzle_path}")

    # Load puzzle
    puzzle_data = load_puzzle(puzzle_path)
    puzzle_id = Path(puzzle_path).stem

    emit(EventType.SYSTEM, f"🧩 Puzzle ID: {puzzle_id}")
    emit(EventType.SYSTEM,
         f"   Train examples: {len(puzzle_data.get('train', []))}")
    emit(EventType.SYSTEM,
         f"   Test examples: {len(puzzle_data.get('test', []))}")

    # Initialize Phoenix if not provided
    phoenix_auto_initialized = False
    if phoenix is None:
        emit(EventType.SYSTEM, "\n🔍 Initializing Phoenix observability...")
        os.environ["ENABLE_PHOENIX"] = "true" if enable_phoenix else "false"
        os.environ["PHOENIX_PROJECT_NAME"] = PHOENIX_PROJECT
        phoenix = initialize_phoenix(
            instrument_openai=False, instrument_anthropic=False)
        phoenix_auto_initialized = True

        if phoenix.enabled:
            emit(EventType.SYSTEM,
                 f"✅ Phoenix initialized - view traces at http://127.0.0.1:6006")
            emit(EventType.SYSTEM, f"   Project: {PHOENIX_PROJECT}")
        else:
            emit(EventType.SYSTEM, f"⚠️  Phoenix not available")

    root_span_ctx = None
    if phoenix.enabled:
        _run_ts = time.strftime("%Y%m%d_%H%M%S")
        root_span_ctx = phoenix.span(
            f"{puzzle_id}_{_run_ts}",
            {"puzzle_id": puzzle_id, "model": model_name,
                "openinference.span.kind": "CHAIN"},
            force_flush=True,
        )
        root_span_ctx.__enter__()

    # Load system prompt
    system_prompt = load_system_prompt()
    emit(EventType.SYSTEM, f"📝 System prompt: {len(system_prompt)} chars")

    with phoenix.span("📋 System Prompt", {"message_type": "system", "puzzle_id": puzzle_id}, force_flush=True):
        phoenix.set_large_attribute("content", system_prompt)
        phoenix.set_attribute("content_length", len(system_prompt))
        phoenix.set_attribute("role", "system")

    # Initialize API client (Anthropic or Fireworks Anthropic-compatible)
    _use_fireworks = _is_fireworks_model(model_name)
    image_source_mode = "base64"
    if _use_fireworks:
        emit(EventType.SYSTEM, "\n🔐 Initializing Fireworks client (Anthropic-compatible)...")
        if not FIREWORKS_API_KEY:
            emit(EventType.ERROR, "FIREWORKS_API_KEY environment variable is required for Kimi models")
            if root_span_ctx:
                root_span_ctx.__exit__(None, None, None)
            return {'solved': False, 'error': 'No Fireworks API key', 'iterations': 0}
        client = anthropic.Anthropic(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL)
        model_name = _resolve_fireworks_model(model_name)
        emit(EventType.SYSTEM, f"   Fireworks model: {model_name}")
    else:
        emit(EventType.SYSTEM, "\n🔐 Initializing Anthropic client...")
        if not ANTHROPIC_API_KEY:
            emit(EventType.ERROR, "ANTHROPIC_API_KEY environment variable is required")
            if root_span_ctx:
                root_span_ctx.__exit__(None, None, None)
            return {'solved': False, 'error': 'No API key', 'iterations': 0}
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # GLM-5 path is text-only in this release.
    if _is_glm_model(model_name) and use_visual_mode:
        emit(EventType.SYSTEM, "   ℹ️ GLM-5 is text-only. Disabling visual grid images for this run.")
        use_visual_mode = False

    # Format puzzle for prompt
    emit(EventType.SYSTEM, "\n📝 Formatting puzzle...")
    user_prompt_content = format_puzzle_for_prompt(
        puzzle_data, use_vision=use_visual_mode)
    emit(EventType.SYSTEM,
         f"   User prompt blocks: {len(user_prompt_content)}")

    emit(EventType.SYSTEM, f"\n🚀 Starting orchestration with {model_name}...")

    # Get tool schemas
    tools = get_tool_schemas()

    # Initial messages - use checkpoint resume if provided
    if initial_messages and len(initial_messages) > 0:
        messages = initial_messages
        emit(EventType.SYSTEM,
             f"📌 Resuming from checkpoint with {len(messages)} API messages")
        if initial_iteration > 0 or initial_turn > 0:
            emit(EventType.SYSTEM,
                 f"   Restored state: iteration={initial_iteration}, turn={initial_turn}")
    else:
        initial_user_content = _append_kimi_run_code_guidance_if_needed(
            user_prompt_content, model_name
        )
        messages = [{"role": "user", "content": initial_user_content}]

    # Log initial user prompt to Phoenix
    text_content_parts = [block["text"]
                          for block in user_prompt_content if block.get("type") == "text"]
    initial_prompt_text = "\n".join(text_content_parts)
    try:
        safe_blocks = []
        for b in user_prompt_content:
            if isinstance(b, dict) and b.get("type") == "image":
                src = b.get("source") or {}
                data = src.get("data")
                safe_blocks.append({
                    "type": "image",
                    "source": {
                        "type": src.get("type"),
                        "media_type": src.get("media_type"),
                        "data": f"<base64:len={len(data) if isinstance(data, str) else 0}>"
                    }
                })
            else:
                safe_blocks.append(b)
    except Exception:
        safe_blocks = []

    with phoenix.span("📝 Initial User Prompt", {"message_type": "user", "puzzle_id": puzzle_id}, force_flush=True):
        phoenix.set_attribute("content_head", initial_prompt_text[:50000])
        phoenix.set_attribute("content_tail", initial_prompt_text[-50000:] if len(
            initial_prompt_text) > 50000 else initial_prompt_text)
        phoenix.set_large_attribute("content_full", initial_prompt_text)
        phoenix.set_large_attribute("content_blocks", safe_blocks)
        phoenix.set_attribute("content_length", len(initial_prompt_text))
        phoenix.set_attribute("num_images", len(
            [b for b in user_prompt_content if isinstance(b, dict) and b.get("type") == "image"]))
        phoenix.set_attribute("role", "user")

    # Orchestration state
    iteration = initial_iteration
    turn = initial_turn
    solved = False
    final_code = None
    last_fully_passing_branches = None  # snapshot of active_solution_branches when training = 100%
    _best_effort_injected = False  # True once best-effort prompt has been injected

    emit(EventType.SYSTEM,
         f"\n🔬 Starting orchestration loop (max {max_iterations} iterations)...")
    emit(EventType.SYSTEM,
         f"   Note: Only execute_python_solution calls count as iterations")
    emit(EventType.SYSTEM, f"   Exploratory code calls are unlimited for debugging")

    # Track current iteration span for hierarchical tracing
    current_iteration_span = None
    current_iteration_num = -1
    last_hypothesis_submission = _empty_hypothesis_submission()
    pending_execute_after_hypothesis_submission = False
    pending_guided_followup = False

    # Iteration history for hypothesis graveyard
    iteration_history = []

    # Seed iteration_history from checkpoint if resuming at a reflection prompt.
    # This ensures _build_consolidated_summary has data when the model RETRYs.
    if initial_messages and initial_in_reflection_mode:
        # Walk initial_messages to find the last execute_python_solution tool_use + result
        _last_code = ""
        _last_hypothesis = ""
        _last_passed = False
        _resume_hypothesis_submission = _empty_hypothesis_submission()
        for msg in initial_messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                if block.get("name") == "submit_transform_hypothesis":
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        submission = _normalize_hypothesis_submission(inp)
                        if _branch_a_hypothesis_text(submission):
                            _resume_hypothesis_submission = submission
                elif block.get("name") == "execute_python_solution":
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        normalized_code_submission = _normalize_code_submission(inp)
                        _last_code = normalized_code_submission.get("branch_a_code", "")
                        _last_hypothesis = _branch_a_hypothesis_text(
                            _derive_execute_hypothesis_submission(
                                _resume_hypothesis_submission,
                            )
                        )
        if _last_code:
            # Test-generalization reflection and reflector-reject compression both occur
            # after a train-passing execute_python_solution result.
            _last_passed = (
                initial_in_test_generalization_reflection
                or initial_in_reflector_reject_compression
            )
            iteration_history.append({
                'iteration': iteration,
                'transform_hypothesis': _last_hypothesis,
                'passed': _last_passed,
                'train_results': [],
                'pixel_accuracy': 1.0 if _last_passed else 0.0,
                'code': _last_code,
            })

    # Store last failed results for visual feedback
    last_failed_results = None
    last_visual_parts = None
    last_test_visual_parts = None
    last_transform_hypothesis = None
    last_reviewer_response = ""
    last_reflector_rejection = ""
    _reflector_message_history: list[dict] = list(initial_reflector_message_history or [])
    # Restore last_transform_hypothesis from checkpoint messages on resume.
    # Without this, the independent reflector is silently skipped because its
    # gate checks `enable_independent_reflector and last_transform_hypothesis`.
    if initial_messages:
        for msg in initial_messages:
            role = msg.get("role")
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            if role == "assistant":
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    inp = block.get("input", {})
                    if not isinstance(inp, dict):
                        continue
                    if block.get("name") == "submit_transform_hypothesis":
                        submission = _normalize_hypothesis_submission(inp)
                        h = _branch_a_hypothesis_text(submission)
                        if h:
                            last_hypothesis_submission = _copy_active_hypothesis_submission(submission)
                            last_transform_hypothesis = h
                    elif block.get("name") == "execute_python_solution":
                        submission = _derive_execute_hypothesis_submission(
                            last_hypothesis_submission,
                        )
                        h = _branch_a_hypothesis_text(submission)
                        if h:
                            last_hypothesis_submission = submission
                            last_transform_hypothesis = h
        if last_transform_hypothesis:
            emit(EventType.SYSTEM,
                 f"   Restored last_transform_hypothesis from checkpoint ({len(last_transform_hypothesis)} chars)")
    last_test_accuracy = None
    last_test_correct_count = None
    last_test_total = None
    last_task_fully_solved = None
    last_test_solved_indices = None
    last_reflection = ""  # Track the most recent reflection across iterations

    # Verified rules: invariant observations that persist across iterations
    # Agent can populate this via reflection (e.g., "output is always 20x20")
    verified_rules = []

    # Track reflection mode
    in_reflection_mode = initial_in_reflection_mode
    in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
        test_generalization=initial_in_test_generalization_reflection,
        reflector_reject_compression=initial_in_reflector_reject_compression,
    )

    # Track where current iteration started (for filtering old exploratory code calls)
    current_iteration_start_idx = 0

    # Track context compression count (limit to prevent infinite loops)
    compression_count = 0

    # Track actual token counts from the last API response for accurate compression check.
    # last_input + last_output ≈ next turn's input (the output becomes part of the next input).
    last_actual_input_tokens = 0
    last_actual_output_tokens = 0
    consecutive_transient_turn_failures = 0

    # Helper functions for code execution
    _USER_CODE_FILENAME = "<agent_code>"

    def _cache_user_code_for_tracebacks(code: str, filename: str = _USER_CODE_FILENAME) -> None:
        try:
            import linecache

            lines = code.splitlines(True)
            if code and not code.endswith(("\n", "\r")):
                lines.append("\n")
            linecache.cache[filename] = (len(code), None, lines, filename)
        except Exception:
            pass

    def _format_user_code_exception(exc: Exception, filename: str = _USER_CODE_FILENAME) -> str:
        lines: list[str] = ["Traceback (most recent call last):"]

        if isinstance(exc, SyntaxError):
            err_filename = str(getattr(exc, "filename", "") or filename)
            err_lineno = int(getattr(exc, "lineno", 1) or 1)
            err_text = str(getattr(exc, "text", "") or "").rstrip("\n")
            err_offset = getattr(exc, "offset", None)

            lines.append(f'  File "{err_filename}", line {err_lineno}')
            if err_text:
                lines.append(f"    {err_text}")
                if isinstance(err_offset, int) and err_offset > 0:
                    lines.append(f"    {' ' * (err_offset - 1)}^")
            syntax_msg = str(getattr(exc, "msg", "") or "invalid syntax")
            lines.append(f"{type(exc).__name__}: {syntax_msg}")
            return "\n".join(lines) + "\n"

        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if str(frame.filename or "") == filename]
        if not user_frames:
            user_frames = [
                frame for frame in extracted
                if os.path.basename(str(frame.filename or "")) != "orchestrator.py"
            ]
        if not user_frames and extracted:
            user_frames = [extracted[-1]]

        for frame in user_frames:
            frame_name = frame.name or "<module>"
            lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame_name}')
            if frame.line:
                lines.append(f"    {frame.line.strip()}")

        lines.append("".join(traceback.format_exception_only(type(exc), exc)).strip())
        return "\n".join(lines) + "\n"

    def execute_code_safe(code, context, timeout_seconds=None):
        def _exec():
            out = io.StringIO()
            err = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                try:
                    _cache_user_code_for_tracebacks(code)
                    exec(compile(code, _USER_CODE_FILENAME, "exec"), context)
                except Exception as exc:
                    err.write(_format_user_code_exception(exc))
            return out.getvalue(), err.getvalue()

        if timeout_seconds is not None:
            result, timed_out = _run_with_timeout(_exec, timeout_seconds)
            if timed_out:
                return "", f"Code execution timed out after {timeout_seconds:.0f} seconds.\n"
            return result
        return _exec()

    def calculate_pixel_accuracy(predicted, expected):
        if predicted is None or expected is None:
            return 0.0
        try:
            pred_rows = len(predicted)
            pred_cols = len(predicted[0]) if pred_rows > 0 else 0
            exp_rows = len(expected)
            exp_cols = len(expected[0]) if exp_rows > 0 else 0

            if pred_rows != exp_rows or pred_cols != exp_cols:
                total_pixels = exp_rows * exp_cols
                if total_pixels == 0:
                    return 0.0
                matching = 0
                for r in range(min(pred_rows, exp_rows)):
                    for c in range(min(pred_cols, exp_cols)):
                        if predicted[r][c] == expected[r][c]:
                            matching += 1
                return matching / total_pixels
            else:
                total_pixels = exp_rows * exp_cols
                if total_pixels == 0:
                    return 1.0
                matching = sum(1 for r in range(exp_rows) for c in range(exp_cols)
                               if predicted[r][c] == expected[r][c])
                return matching / total_pixels
        except Exception:
            return 0.0

    def check_solution(solve_func, train_samples, test_samples, check_test_accuracy=False):
        import copy

        results = {
            "train": [],
            "test": [],
            "correct": True,
            "test_correct": None,
            "train_pixel_accuracy": 0.0,
            "num_test_predictions": 0,
        }
        total_pixel_acc = 0.0

        for idx, sample in enumerate(train_samples):
            inp = sample["input"]
            expected = sample["output"]
            try:
                inp_copy = copy.deepcopy(inp)
                raw_pred = solve_func(inp_copy)
                # Allow up to max_test_predictions for normalization so we
                # can detect multi-candidate returns and warn the solver.
                candidates, normalization_error = _normalize_solve_candidates(
                    raw_pred,
                    max_candidates=max_test_predictions,
                )
                if normalization_error:
                    raise ValueError(normalization_error)

                # Training: only the first candidate counts.
                multi_candidate_warning = len(candidates) > 1
                train_candidate = candidates[0:1]  # single-element list

                candidate_scores = [calculate_pixel_accuracy(c, expected) for c in train_candidate]
                best_candidate = train_candidate[0]
                best_pixel_acc = candidate_scores[0]
                is_correct = best_candidate == expected
                total_pixel_acc += best_pixel_acc

                row = {
                    "index": idx,
                    "correct": is_correct,
                    "input": inp,
                    "expected": expected,
                    "candidates": train_candidate,
                    "predicted": best_candidate,
                    "pixel_accuracy": best_pixel_acc,
                    "matched_candidate_index": 1 if is_correct else None,
                    "multi_candidate_warning": multi_candidate_warning,
                }
                results["train"].append(row)
                if not is_correct:
                    results["correct"] = False
            except Exception as e:
                results["correct"] = False
                results["train"].append(
                    {
                        "index": idx,
                        "error": str(e),
                        "input": inp,
                        "expected": expected,
                        "candidates": [],
                        "pixel_accuracy": 0.0,
                    }
                )

        if train_samples:
            results["train_pixel_accuracy"] = total_pixel_acc / len(train_samples)

        if check_test_accuracy:
            results["test_correct"] = True

        max_test_candidates = 0
        for idx, sample in enumerate(test_samples):
            inp = sample["input"]
            expected = sample.get("output")
            try:
                inp_copy = copy.deepcopy(inp)
                raw_pred = solve_func(inp_copy)
                candidates, normalization_error = _normalize_solve_candidates(
                    raw_pred,
                    max_candidates=max_test_predictions,
                )
                if normalization_error:
                    raise ValueError(normalization_error)

                max_test_candidates = max(max_test_candidates, len(candidates))
                is_correct = (
                    any(candidate == expected for candidate in candidates)
                    if expected is not None
                    else None
                )
                results["test"].append(
                    {
                        "index": idx,
                        "input": inp,
                        "expected": expected,
                        "correct": is_correct,
                        "candidates": candidates,
                        "predicted": candidates[0],
                        "matched_candidate_index": next(
                            (candidate_idx for candidate_idx, candidate in enumerate(candidates, start=1) if expected is not None and candidate == expected),
                            None,
                        ),
                    }
                )
                if expected is not None and not is_correct:
                    results["test_correct"] = False
            except Exception as e:
                if check_test_accuracy:
                    results["test_correct"] = False
                results["test"].append(
                    {
                        "index": idx,
                        "error": str(e),
                        "input": inp,
                        "expected": expected,
                        "candidates": [],
                    }
                )

        results["num_test_predictions"] = max_test_candidates
        return results

    def evaluate_solution_branches(
        code_submission: dict[str, str],
        effective_hypothesis_submission: dict[str, str],
    ) -> dict[str, Any]:
        transform_hypothesis = _branch_a_hypothesis_text(effective_hypothesis_submission)
        code = str(code_submission.get("code") or code_submission.get("branch_a_code") or "")

        solution_record: dict[str, Any] = {
            "hypothesis": transform_hypothesis,
            "code": code,
            "status": "error",
            "error": "",
            "result": {
                "train": [],
                "test": [],
                "correct": False,
                "test_correct": None,
                "train_pixel_accuracy": 0.0,
                "num_test_predictions": 0,
            },
        }

        solution_local_ctx = execution_context.copy()
        _, solution_stderr = execute_code_safe(code, solution_local_ctx)
        if solution_stderr:
            solution_record["error"] = solution_stderr
            solution_record["result"]["error"] = solution_stderr
            return {
                "branch_outcomes": [solution_record],
                "normalized_branches": [solution_record],
                "results": solution_record["result"],
                "branch_summary_line": "Execution error while running solve(grid).",
                "active_solution_branches": [],
                "selected_solution_branch": None,
            }

        if "solve" not in solution_local_ctx:
            solution_record["error"] = "No solve() function"
            solution_record["result"]["error"] = "No solve() function"
            return {
                "branch_outcomes": [solution_record],
                "normalized_branches": [solution_record],
                "results": solution_record["result"],
                "branch_summary_line": "Execution error: no solve() function defined.",
                "active_solution_branches": [],
                "selected_solution_branch": None,
            }

        solution_solve_fn = solution_local_ctx["solve"]
        solution_results = check_solution(
            solution_solve_fn,
            solution_local_ctx["train_samples"],
            solution_local_ctx["test_samples"],
            check_test_accuracy=False,
        )
        solution_results["hypothesis"] = transform_hypothesis
        solution_results["candidate_predictions"] = _serialize_candidate_predictions(solution_results.get("test", []))

        solution_record.update(
            {
                "status": "ok",
                "result": solution_results,
                "solve_fn": solution_solve_fn,
                "local_ctx": solution_local_ctx,
            }
        )

        covered_examples = sum(1 for row in solution_results.get("train", []) if row.get("correct"))
        total_examples = len(solution_results.get("train", []))
        if solution_results.get("correct", False):
            branch_summary_line = "All training examples are covered by the candidate generator."
            active_solution_records = [solution_record]
            selected_solution_record = solution_record
        else:
            branch_summary_line = (
                f"Training coverage: {covered_examples}/{total_examples} examples covered by at least one candidate."
            )
            active_solution_records = []
            selected_solution_record = None

        return {
            "branch_outcomes": [solution_record],
            "normalized_branches": [solution_record],
            "results": solution_results,
            "branch_summary_line": branch_summary_line,
            "active_solution_branches": active_solution_records,
            "selected_solution_branch": selected_solution_record,
        }

    def build_reflector_candidate_payloads(
        solution_record: dict[str, Any] | None,
        test_input_grids: list[Any],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        test_input_images: list[str] = []
        if use_visual_mode:
            for grid in test_input_grids:
                img_in = render_grid_to_base64(grid)
                test_input_images.append(img_in or "")

        payloads: list[dict[str, Any]] = []
        result = solution_record.get("result", {}) if isinstance(solution_record, dict) else {}
        for row in result.get("test", []) or []:
            if not isinstance(row, dict):
                continue
            candidate_images: list[str] = []
            if use_visual_mode:
                for candidate in row.get("candidates", []) or []:
                    img_pred = render_grid_to_base64(candidate) if candidate is not None else None
                    candidate_images.append(img_pred or "")
            payloads.append(
                {
                    "index": row.get("index"),
                    "candidates": copy.deepcopy(row.get("candidates") or []),
                    "candidate_images": candidate_images,
                    "error": row.get("error"),
                }
            )

        return test_input_images, payloads

    def update_internal_test_score(solve_func=None, solution_branches: list[dict[str, Any]] | None = None):
        nonlocal last_test_accuracy, last_test_correct_count, last_test_total, last_task_fully_solved, last_test_solved_indices

        branch_eval_results: list[dict[str, Any]] = []
        if solution_branches:
            for branch in solution_branches:
                branch_solve_fn = branch.get("solve_fn")
                branch_local_ctx = branch.get("local_ctx") or execution_context
                if branch_solve_fn is None:
                    continue
                branch_eval_results.append(
                    check_solution(
                        branch_solve_fn,
                        branch_local_ctx["train_samples"],
                        full_test_samples,
                        check_test_accuracy=True,
                    )
                )
        elif solve_func is not None:
            branch_eval_results.append(
                check_solution(
                    solve_func,
                    local_ctx['train_samples'],
                    full_test_samples,
                    check_test_accuracy=True,
                )
            )

        if not branch_eval_results:
            last_test_accuracy = None
            last_test_correct_count = None
            last_test_total = None
            last_task_fully_solved = None
            last_test_solved_indices = None
            emit(
                EventType.SYSTEM,
                "   [Internal] ARC task score unavailable (no executable candidate branches) - NOT shown to model",
            )
            return

        combined_by_index: dict[int, bool] = {}
        for branch_result in branch_eval_results:
            for res in branch_result.get("test", []):
                if res.get("correct") is None:
                    continue
                idx = int(res.get("index", -1))
                if idx < 0:
                    continue
                combined_by_index[idx] = combined_by_index.get(idx, False) or bool(res.get("correct", False))

        test_total = len(combined_by_index)
        if test_total == 0:
            last_test_accuracy = None
            last_test_correct_count = None
            last_test_total = None
            last_task_fully_solved = None
            last_test_solved_indices = None
            emit(
                EventType.SYSTEM,
                "   [Internal] ARC task score unavailable (no labeled test outputs) - NOT shown to model",
            )
            return

        solved_indices = [idx for idx, solved_flag in sorted(combined_by_index.items()) if solved_flag]
        test_correct_count = sum(1 for solved_flag in combined_by_index.values() if solved_flag)
        task_score = test_correct_count / test_total
        task_fully_solved = (test_correct_count == test_total)
        solve_status = "task fully solved" if task_fully_solved else "task partially solved"

        emit(
            EventType.SYSTEM,
            f"   [Internal] ARC task score: {task_score:.1%} "
            f"({test_correct_count}/{test_total} test outputs correct; {solve_status}) - NOT shown to model",
        )
        last_test_accuracy = task_score
        last_test_correct_count = test_correct_count
        last_test_total = test_total
        last_task_fully_solved = task_fully_solved
        last_test_solved_indices = solved_indices

    full_test_samples = puzzle_data.get("test", [])
    execution_test_samples = _strip_test_outputs(full_test_samples)

    # Execution context for tools
    execution_context = {
        'train_samples': puzzle_data.get('train', []),
        'test_samples': execution_test_samples,
        'np': np,
        'json': json,
    }
    exploratory_runtime_context: dict[str, Any] | None = None
    exploratory_runtime_id = 0
    exploratory_runtime_step = 0

    def _new_exploratory_runtime_context() -> dict[str, Any]:
        return execution_context.copy()

    def _invalidate_exploratory_runtime() -> None:
        nonlocal exploratory_runtime_context, exploratory_runtime_step
        exploratory_runtime_context = None
        exploratory_runtime_step = 0

    def _begin_fresh_exploratory_runtime() -> tuple[dict[str, Any], int, int]:
        nonlocal exploratory_runtime_context, exploratory_runtime_id, exploratory_runtime_step
        exploratory_runtime_id += 1
        exploratory_runtime_context = _new_exploratory_runtime_context()
        exploratory_runtime_step = 1
        return exploratory_runtime_context, exploratory_runtime_id, exploratory_runtime_step

    def _reuse_exploratory_runtime() -> tuple[dict[str, Any] | None, int | None, int | None]:
        nonlocal exploratory_runtime_step
        if exploratory_runtime_context is None:
            return None, None, None
        exploratory_runtime_step += 1
        return exploratory_runtime_context, exploratory_runtime_id, exploratory_runtime_step

    def _find_dangling_tool_use_ids(msgs: list[dict]) -> set[str]:
        """Return IDs of tool_use blocks in the last assistant message that have no tool_result."""
        if not msgs:
            return set()
        last = msgs[-1]
        if last.get("role") != "assistant":
            return set()
        content = last.get("content", [])
        if not isinstance(content, list):
            return set()
        return {
            b.get("id", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
        }

    def _restore_exploratory_runtime_from_initial_messages() -> None:
        nonlocal exploratory_runtime_id
        if not initial_messages or not unsafe_local_exec:
            if initial_messages and not unsafe_local_exec:
                emit(
                    EventType.SYSTEM,
                    "   ⚠️ Skipping exploratory runtime restoration from checkpoint because unsafe local Python execution is disabled.",
                )
            return

        # Determine the runtime ID offset from tool results in initial_messages.
        # Context compression may have removed earlier runtimes (e.g. R1) from
        # the messages, so the first run_code here might actually be R2 or later.
        # Parse the tool result text to find the correct starting ID.
        import re as _re_mod
        _first_fresh_runtime_id = None
        for _msg in initial_messages:
            if _msg.get("role") != "user":
                continue
            _mc = _msg.get("content", [])
            if not isinstance(_mc, list):
                continue
            for _mb in _mc:
                if not isinstance(_mb, dict) or _mb.get("type") != "tool_result":
                    continue
                _text = ""
                _mbc = _mb.get("content", "")
                if isinstance(_mbc, str):
                    _text = _mbc
                elif isinstance(_mbc, list):
                    _text = " ".join(
                        str(x.get("text", "")) for x in _mbc if isinstance(x, dict)
                    )
                _match = _re_mod.search(r"Runtime:\s*fresh\s*\(id=R(\d+)", _text)
                if _match:
                    _first_fresh_runtime_id = int(_match.group(1))
                    break
            if _first_fresh_runtime_id is not None:
                break
        if _first_fresh_runtime_id is not None and _first_fresh_runtime_id > 1:
            # Set the counter so _begin_fresh_exploratory_runtime() produces
            # the correct ID on its first call (it does id += 1).
            exploratory_runtime_id = _first_fresh_runtime_id - 1

        # Skip dangling tool_uses — they'll be re-executed separately with proper tool_results.
        dangling_ids = _find_dangling_tool_use_ids(initial_messages)

        replayed_calls = 0
        replayed_errors = 0
        fallback_replays = 0

        for msg in initial_messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                if block.get("id", "") in dangling_ids:
                    continue
                tool_name = str(block.get("name") or "")
                if tool_name not in EXPLORATORY_CODE_TOOL_NAMES:
                    continue
                tool_input = block.get("input", {})
                if not isinstance(tool_input, dict):
                    continue
                code = str(tool_input.get("code") or "")
                if not code:
                    continue

                if tool_name == "run_code":
                    runtime_ctx, _, _ = _begin_fresh_exploratory_runtime()
                else:
                    runtime_ctx, _, _ = _reuse_exploratory_runtime()
                    if runtime_ctx is None:
                        runtime_ctx, _, _ = _begin_fresh_exploratory_runtime()
                        fallback_replays += 1

                # Use the original timeout from the tool call to faithfully
                # reproduce the execution (the agent may have requested up
                # to CODE_EXEC_MAX_TIMEOUT for heavy computations).
                replay_timeout = _clamp_timeout(
                    tool_input.get("timeout_seconds"),
                    CODE_EXEC_DEFAULT_TIMEOUT,
                )
                _stdout, stderr = execute_code_safe(code, runtime_ctx, timeout_seconds=replay_timeout)
                replayed_calls += 1
                if stderr:
                    replayed_errors += 1

        if replayed_calls:
            summary = (
                f"   Restored live exploratory runtime from checkpoint "
                f"({replayed_calls} replayed exploratory call"
            )
            if replayed_calls != 1:
                summary += "s"
            summary += f"; current runtime=R{exploratory_runtime_id}, step={exploratory_runtime_step}"
            if fallback_replays:
                summary += f"; {fallback_replays} replay step"
                if fallback_replays != 1:
                    summary += "s"
                summary += " used fresh fallback"
            if replayed_errors:
                summary += f"; {replayed_errors} replayed call"
                if replayed_errors != 1:
                    summary += "s"
                summary += " reproduced stderr"
            summary += ")."
            emit(EventType.SYSTEM, summary)

    def _execute_dangling_tool_calls(msgs: list[dict]) -> None:
        """Detect and re-execute dangling tool_use blocks from a checkpoint resume.

        If the last assistant message has tool_use blocks with no matching
        tool_results (i.e. the previous session ended mid-execution), run them
        now with timeout and append the tool_results to *msgs*.
        """
        dangling_ids = _find_dangling_tool_use_ids(msgs)
        if not dangling_ids:
            return
        last = msgs[-1]
        content = last.get("content", [])
        if not isinstance(content, list):
            return

        dangling_tool_uses = [
            b for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") in dangling_ids
        ]
        emit(EventType.SYSTEM,
             f"🔄 Re-executing {len(dangling_tool_uses)} pending tool call(s) from checkpoint...")

        tool_results: list[dict[str, Any]] = []
        for tu in dangling_tool_uses:
            tu_name = str(tu.get("name") or "")
            tu_args = tu.get("input") or {}
            if not isinstance(tu_args, dict):
                tu_args = {}
            tu_id = str(tu.get("id") or "")
            emit(EventType.SYSTEM, f"   Running {tu_name} (id={tu_id})...")

            if tu_name in EXPLORATORY_CODE_TOOL_NAMES:
                if not unsafe_local_exec:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu_id,
                        "content": "Local Python execution is disabled.",
                        "is_error": True,
                    })
                    continue
                code = str(tu_args.get("code") or "")
                timeout = _clamp_timeout(tu_args.get("timeout_seconds"), CODE_EXEC_DEFAULT_TIMEOUT)
                # Respect the original tool name: reuse the runtime restored
                # by _restore_exploratory_runtime_from_initial_messages when
                # the dangling call was run_code_in_previous_runtime.
                runtime_status = "fresh"
                if tu_name == "run_code":
                    runtime_ctx, runtime_id, runtime_step = _begin_fresh_exploratory_runtime()
                else:
                    runtime_status = "reused"
                    runtime_ctx, runtime_id, runtime_step = _reuse_exploratory_runtime()
                    if runtime_ctx is None:
                        runtime_status = "fresh_fallback"
                        runtime_ctx, runtime_id, runtime_step = _begin_fresh_exploratory_runtime()
                stdout, stderr = execute_code_safe(code, runtime_ctx, timeout_seconds=timeout)
                result_text = f"Runtime: {runtime_status} (id=R{runtime_id}, step={runtime_step})\n"
                result_text += f"STDOUT:\n{stdout}\n"
                if stderr:
                    result_text += f"STDERR:\n{stderr}"
                result_text = _truncate_tool_output(result_text, max_chars=5000)
                emit(EventType.TOOL_RESULT, f"{tu_name} result", {
                    "id": tu_id, "tool_use_id": tu_id, "tool_name": tu_name,
                    "output": result_text,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": result_text,
                })

            elif tu_name == "execute_python_solution":
                if not unsafe_local_exec:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu_id,
                        "content": "Local Python execution is disabled.",
                        "is_error": True,
                    })
                    continue
                code_submission = _normalize_code_submission(tu_args)
                hypothesis_submission = _derive_execute_hypothesis_submission(
                    last_hypothesis_submission)
                solution_timeout = _clamp_timeout(
                    tu_args.get("timeout_seconds"), CODE_EXEC_SOLUTION_DEFAULT_TIMEOUT)
                branch_eval, eval_timed_out = _run_with_timeout(
                    lambda: evaluate_solution_branches(code_submission, hypothesis_submission),
                    solution_timeout,
                )
                if eval_timed_out:
                    timeout_msg = (
                        f"Code execution timed out after {solution_timeout:.0f} seconds. "
                        f"Your solve() function likely has an infinite loop or is too slow. "
                        f"Fix the bug and retry."
                    )
                    emit(EventType.SYSTEM, f"   -> Timed out after {solution_timeout:.0f}s")
                    emit(EventType.TOOL_RESULT, "execute_python_solution result", {
                        "id": tu_id, "tool_use_id": tu_id,
                        "tool_name": "execute_python_solution",
                        "output": timeout_msg,
                        "is_error": True,
                        "result": {"correct": False, "error": timeout_msg},
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu_id,
                        "content": timeout_msg,
                        "is_error": True,
                    })
                else:
                    results = branch_eval["results"]
                    report_lines = [branch_eval["branch_summary_line"]]
                    for res in results.get("train", []) or []:
                        if res.get("error"):
                            report_lines.append(f"Train {res['index']} Error: {res['error']}")
                        elif res.get("correct"):
                            report_lines.append(f"Train {res['index']} PASS")
                        else:
                            report_lines.append(
                                f"Train {res['index']} pixel acc: {res.get('pixel_accuracy', 0):.1%}")
                    report_text = "\n".join(report_lines)
                    emit(EventType.TOOL_RESULT, "execute_python_solution result", {
                        "id": tu_id, "tool_use_id": tu_id,
                        "tool_name": "execute_python_solution",
                        "output": report_text,
                        "result": results,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu_id,
                        "content": report_text,
                    })

            elif tu_name == "submit_transform_hypothesis":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": "Hypothesis recorded (checkpoint resume).",
                })

            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": f"Unknown tool '{tu_name}' — skipped on checkpoint resume.",
                    "is_error": True,
                })

        if tool_results:
            msgs.append({"role": "user", "content": tool_results})
            emit(EventType.SYSTEM,
                 f"   ✅ Appended {len(tool_results)} tool result(s) from checkpoint re-execution")

    _restore_exploratory_runtime_from_initial_messages()

    # Re-execute any dangling tool calls left over from a previous session that
    # ended mid-execution (e.g. infinite loop, crash, manual terminate).
    if initial_messages:
        _execute_dangling_tool_calls(messages)

    # Pre-loop best-effort check: if resuming an already-exhausted run (turn >= max_turns - 8)
    # with no 100%-training solution, inject best-effort prompt and lift the turn cap so the
    # while-loop condition is satisfied and the solver gets another chance.
    if (
        not _best_effort_injected
        and not last_fully_passing_branches
        and not isinstance(max_turns, float)
        and turn >= max_turns - 8
        and not solved
        and not should_stop()
    ):
        _best_effort_injected = True
        max_turns = float('inf')
        in_reflection_mode = False
        emit(EventType.SYSTEM,
             "   ⚡ Training-inconsistent puzzle detected (pre-loop) — injecting best-effort prompt (tools remain available)")
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": BEST_EFFORT_PROMPT}],
        })

    # Main orchestration loop
    while iteration < max_iterations and turn < max_turns and not solved and not should_stop():
        turn += 1
        emit(EventType.TURN_START, "", {
             "turn": turn, "iteration": iteration, "max_iterations": max_iterations})
        emit(EventType.SYSTEM,
             f"⏳ Waiting for model response (Streaming: {use_streaming})...")

        # Start new iteration span if iteration changed
        if iteration != current_iteration_num:
            if current_iteration_span:
                current_iteration_span.__exit__(None, None, None)
            current_iteration_num = iteration
            current_iteration_span = phoenix.span(
                f"iteration_{iteration}",
                {"iteration": iteration, "puzzle_id": puzzle_id, "model": model_name}
            )
            current_iteration_span.__enter__()

        # If resuming from checkpoint in reflection mode (reflection_prompt was deleted via rollback),
        # build and inject a fresh reflection prompt into messages before the first API call.
        # This runs only on the very first turn of a checkpoint resume.
        if turn == initial_turn + 1 and initial_in_reflection_mode and initial_messages:
            if initial_in_reflector_reject_compression:
                emit(
                    EventType.SYSTEM,
                    "   ↪️ Resuming from reflector-reject compression — using the reconstructed post-reflector prompt as-is.",
                )
                # Carry the reflector's response so the consolidated summary
                # includes it under "Independent Reviewer Analysis"
                if initial_reflector_response:
                    last_reflector_rejection = initial_reflector_response
                # Build test prediction visual parts from checkpoint candidates
                # so the consolidated summary includes them
                if initial_test_candidates:
                    test_prediction_only_parts = []
                    test_input_grids = [ex["input"] for ex in puzzle_data.get("test", [])]
                    for tc in initial_test_candidates:
                        test_index = int(tc.get("index", 0))
                        candidates = tc.get("candidates") or []
                        candidate_images = tc.get("candidate_images") or []
                        test_prediction_only_parts.append({"type": "text", "text": f"\n**Test Example {test_index}**"})
                        for ci, cand in enumerate(candidates):
                            label = f"{_ordinal(ci + 1)} candidate: {cand}"
                            test_prediction_only_parts.append({"type": "text", "text": label})
                            if use_visual_mode and ci < len(candidate_images) and candidate_images[ci]:
                                test_prediction_only_parts.append({
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/png", "data": candidate_images[ci]},
                                })
                    last_test_visual_parts = test_prediction_only_parts
                # Re-emit TURN_START with correct label so UI shows
                # "Solver (post-reflector)" instead of a generic iteration name
                emit(EventType.TURN_START, "", {
                     "turn": turn, "iteration": iteration, "max_iterations": max_iterations,
                     "context_label": "Solver (post-reflector)"})
                # Emit iteration context snapshot so the UI shows the full
                # conversation history (including system/user prompt) in the
                # post-reflector canvas — no separate prompt_bundle needed.
                snapshot_messages = messages[current_iteration_start_idx:]
                emit(
                    EventType.REFLECTION,
                    "",
                    _iteration_context_metadata(
                        iteration=iteration,
                        ui_title="📋 **Solver (post-reflector)**",
                        context_mode="snapshot",
                        snapshot_messages=snapshot_messages,
                        resume_messages=messages,
                    ),
                )
            elif initial_in_test_generalization_reflection:
                emit(EventType.SYSTEM, "   🔄 Checkpoint resume in reflection mode — rebuilding reflection prompt with latest logic...")
                # Rebuild test generalization reflection prompt + test visual parts
                # Re-run the solution to get test predictions
                solve_fn = None
                local_ctx = None
                _resume_code_submission: dict[str, str] = {}
                _resume_hypothesis_submission = _copy_hypothesis_submission(last_hypothesis_submission)
                for _msg in initial_messages:
                    if _msg.get("role") != "assistant":
                        continue
                    for _blk in (_msg.get("content", []) if isinstance(_msg.get("content"), list) else []):
                        if not isinstance(_blk, dict) or _blk.get("type") != "tool_use":
                            continue
                        if _blk.get("name") == "submit_transform_hypothesis":
                            _blk_input = _blk.get("input", {})
                            if isinstance(_blk_input, dict):
                                _resume_hypothesis_submission = _normalize_hypothesis_submission(_blk_input)
                        elif _blk.get("name") == "execute_python_solution":
                            _blk_input = _blk.get("input", {})
                            if isinstance(_blk_input, dict):
                                _resume_code_submission = _normalize_code_submission(_blk_input)

                if _resume_code_submission:
                    effective_hypothesis_submission = _derive_execute_hypothesis_submission(
                        _resume_hypothesis_submission,
                    )
                    last_transform_hypothesis = _branch_a_hypothesis_text(effective_hypothesis_submission) or last_transform_hypothesis
                    last_reviewer_response = str(effective_hypothesis_submission.get("reviewer_response") or "").strip()
                    last_execute_hypothesis_submission = _copy_hypothesis_submission(
                        effective_hypothesis_submission
                    )
                    last_execute_code_submission = dict(_resume_code_submission)
                    branch_eval = evaluate_solution_branches(
                        _resume_code_submission,
                        effective_hypothesis_submission,
                    )
                    active_solution_branches = branch_eval["active_solution_branches"]
                    selected_solution_branch = branch_eval["selected_solution_branch"]
                    if selected_solution_branch:
                        solve_fn = selected_solution_branch.get("solve_fn")
                        local_ctx = selected_solution_branch.get("local_ctx")
                        final_code = str(selected_solution_branch.get("code") or final_code or "")
                    else:
                        active_solution_branches = []
                else:
                    active_solution_branches = []

                if active_solution_branches:
                    # Full parts for reflection prompt (includes test inputs)
                    test_visual_parts = []
                    test_prediction_only_parts = []  # Prediction-only for consolidated summary
                    test_input_grids = [example["input"]
                                        for example in puzzle_data.get("test", [])]
                    _reflector_test_inputs = list(test_input_grids)
                    _reflector_test_input_images, _reflector_candidate_payloads = build_reflector_candidate_payloads(
                        selected_solution_branch,
                        test_input_grids,
                    )
                    _selected_test_predictions = []
                    selected_test_results = (
                        selected_solution_branch.get("result", {}).get("test", [])
                        if isinstance(selected_solution_branch, dict)
                        else []
                    ) or []
                    for res in selected_test_results:
                        if not isinstance(res, dict):
                            continue
                        test_index = int(res.get("index", 0))
                        candidates = copy.deepcopy(res.get("candidates") or [])
                        candidate_images: list[str] = []

                        test_visual_parts.append({"type": "text", "text": f"\n**Test Example {test_index}**"})
                        test_prediction_only_parts.append({"type": "text", "text": f"\n**Test Example {test_index}**"})
                        test_visual_parts.append({"type": "text", "text": f"Input: {test_input_grids[test_index]}"})
                        if use_visual_mode and test_index < len(_reflector_test_input_images) and _reflector_test_input_images[test_index]:
                            test_visual_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": _reflector_test_input_images[test_index],
                                    },
                                }
                            )

                        for candidate_idx, candidate in enumerate(candidates, start=1):
                            if candidate_idx == 1:
                                _selected_test_predictions.append(candidate)
                            label = f"{_ordinal(candidate_idx)} candidate: {candidate}"
                            test_visual_parts.append({"type": "text", "text": label})
                            test_prediction_only_parts.append({"type": "text", "text": label})
                            if use_visual_mode:
                                img_pred = render_grid_to_base64(candidate) or ""
                                candidate_images.append(img_pred)
                                if img_pred:
                                    image_block = {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": img_pred,
                                        },
                                    }
                                    test_visual_parts.append(image_block)
                                    test_prediction_only_parts.append(image_block)
                            else:
                                candidate_images.append("")

                        if res.get("error"):
                            error_line = f"Error: {res['error']}"
                            test_visual_parts.append({"type": "text", "text": error_line})
                            test_prediction_only_parts.append({"type": "text", "text": error_line})
                    last_test_visual_parts = test_prediction_only_parts

                    # ── Independent Reflector (checkpoint resume) ─────────────
                    _reflector_verdict = None
                    _reflector_result = {}
                    if enable_independent_reflector and last_transform_hypothesis:
                        iteration += 1
                        max_iterations += 2  # compensate for reflector + resume
                        emit(EventType.TURN_START, "", {
                             "turn": turn, "iteration": iteration, "max_iterations": max_iterations,
                             "context_label": "Independent Reflector"})
                        emit(EventType.SYSTEM,
                             f"\n🔍 Running independent reflector on checkpoint resume ({reflector_provider}: {reflector_model or 'default'})...")
                        try:
                            from .independent_reflector import run_independent_reflection
                        except ImportError:
                            from independent_reflector import run_independent_reflection

                        _reflector_test_preds = list(_selected_test_predictions)
                        _reflector_train_inputs = [ex["input"] for ex in puzzle_data.get("train", [])]
                        _reflector_train_outputs = [ex["output"] for ex in puzzle_data.get("train", [])]
                        _reflector_train_input_images = []
                        _reflector_train_output_images = []
                        if use_visual_mode:
                            for ex in puzzle_data.get("train", []):
                                img_in = render_grid_to_base64(ex["input"])
                                _reflector_train_input_images.append(img_in or "")
                                img_out = render_grid_to_base64(ex["output"])
                                _reflector_train_output_images.append(img_out or "")

                        _refl_model_label = reflector_model or ("claude-opus-4-6" if reflector_provider == "claude" else "gemini-3.1-pro-preview")

                        # Build per-candidate training accuracy breakdown
                        _ckpt_solution_result = selected_solution_branch.get("result", {}) if isinstance(selected_solution_branch.get("result"), dict) else {}
                        _train_match_lines = []
                        for _tres in _ckpt_solution_result.get("train", []) or []:
                            if not isinstance(_tres, dict):
                                continue
                            _tidx = _tres.get("index", "?")
                            _mci = _tres.get("matched_candidate_index")
                            if _mci:
                                _train_match_lines.append(f"Train {_tidx}: matched by {_ordinal(_mci)} candidate")
                            elif _tres.get("correct", False):
                                _train_match_lines.append(f"Train {_tidx}: PASS")
                            else:
                                _train_match_lines.append(f"Train {_tidx}: FAIL")
                        _reflector_training_accuracy = "100% on training set"
                        if _train_match_lines:
                            _reflector_training_accuracy += "\n" + "\n".join(_train_match_lines)

                        # Load reflector system prompt and build user message
                        try:
                            from .independent_reflector import _load_reflector_prompt, _build_reflector_user_message, _build_reflector_followup_message
                        except ImportError:
                            from independent_reflector import _load_reflector_prompt, _build_reflector_user_message, _build_reflector_followup_message

                        _reflector_system_prompt = _load_reflector_prompt()
                        _reflector_is_followup = bool(_reflector_message_history)
                        if _reflector_is_followup:
                            _reflector_user_msg = _build_reflector_followup_message(
                                transform_hypothesis=last_transform_hypothesis,
                                code=str(selected_solution_branch.get("code") or ""),
                                test_predictions=_reflector_test_preds,
                                training_accuracy=_reflector_training_accuracy,
                                ambiguity_rationale="",
                                candidate_predictions=_reflector_candidate_payloads,
                            )
                        else:
                            _reflector_user_msg = _build_reflector_user_message(
                                transform_hypothesis=last_transform_hypothesis,
                                code=str(selected_solution_branch.get("code") or ""),
                                train_inputs=_reflector_train_inputs,
                                train_outputs=_reflector_train_outputs,
                                test_inputs=_reflector_test_inputs,
                                test_predictions=_reflector_test_preds,
                                training_accuracy=_reflector_training_accuracy,
                                ambiguity_rationale="",
                                candidate_predictions=_reflector_candidate_payloads,
                                test_input_images=_reflector_test_input_images if use_visual_mode else None,
                                test_prediction_images=[],
                            )

                        _reflector_emit_meta = {
                            "turn": turn,
                            "iteration": iteration,
                            "provider": reflector_provider,
                            "model": _refl_model_label,
                            "system_md": _reflector_system_prompt,
                            "user_md": _reflector_user_msg,
                            "hypothesis": last_transform_hypothesis,
                            "code": str(selected_solution_branch.get("code") or ""),
                            "review_candidates": _reflector_candidate_payloads,
                            "phase": "start",
                            "reflector_turn": len(_reflector_message_history) // 2 + 1,
                            "prior_turns": list(_reflector_message_history) if _reflector_message_history else [],
                        }
                        if not _reflector_is_followup:
                            _reflector_emit_meta.update({
                                "train_inputs": _reflector_train_inputs,
                                "train_outputs": _reflector_train_outputs,
                                "test_inputs": _reflector_test_inputs,
                                "test_predictions": _reflector_test_preds,
                                "train_input_images": _reflector_train_input_images if use_visual_mode else [],
                                "train_output_images": _reflector_train_output_images if use_visual_mode else [],
                                "test_input_images": _reflector_test_input_images if use_visual_mode else [],
                            })
                        emit(EventType.REFLECTOR_CONTEXT, "", _reflector_emit_meta)

                        try:
                            _reflector_result = run_independent_reflection(
                                transform_hypothesis=last_transform_hypothesis,
                                code=str(selected_solution_branch.get("code") or ""),
                                test_inputs=_reflector_test_inputs,
                                test_predictions=_reflector_test_preds,
                                training_accuracy=_reflector_training_accuracy,
                                ambiguity_rationale="",
                                candidate_predictions=_reflector_candidate_payloads,
                                train_inputs=_reflector_train_inputs,
                                train_outputs=_reflector_train_outputs,
                                train_input_images=_reflector_train_input_images if use_visual_mode else None,
                                train_output_images=_reflector_train_output_images if use_visual_mode else None,
                                test_input_images=_reflector_test_input_images if use_visual_mode else None,
                                test_prediction_images=[],
                                reflector_provider=reflector_provider,
                                reflector_model=reflector_model,
                                reflector_thinking_effort=reflector_thinking_effort,
                                reflector_code_execution=reflector_code_execution,
                                emit=lambda msg: emit(EventType.SYSTEM, msg),
                                stream_emit=emit,
                                should_stop=should_stop,
                                message_history=_reflector_message_history,
                                reviewer_response=last_reviewer_response,
                            )
                            _reflector_message_history = _reflector_result.get("message_history", _reflector_message_history)
                            _reflector_verdict = _reflector_result.get("verdict", "UNKNOWN")
                            _reflector_response = _reflector_result.get("response", "")
                            _reflector_thinking = _reflector_result.get("thinking", "")

                            emit(EventType.SYSTEM,
                                 f"   Reflector verdict: {_reflector_verdict}")

                            _refl_emit_meta = {
                                "turn": turn,
                                "iteration": iteration,
                                "provider": reflector_provider,
                                "model": _refl_model_label,
                                "phase": "result",
                                "verdict": _reflector_verdict,
                                "thinking": _reflector_thinking,
                                "response": _reflector_response,
                            }
                            for _ukey in (
                                "usage_input_tokens",
                                "usage_uncached_input_tokens",
                                "usage_cache_write_tokens",
                                "usage_cache_read_tokens",
                                "usage_thinking_tokens",
                                "usage_output_tokens",
                                "usage_output_includes_reasoning",
                                "usage_reasoning_tokens_reported",
                                "usage_total_tokens",
                            ):
                                if _ukey in _reflector_result:
                                    _refl_emit_meta[_ukey] = _reflector_result[_ukey]
                            emit(EventType.REFLECTOR_CONTEXT, "", _refl_emit_meta)

                            with phoenix.span("Independent Reflector (Checkpoint Resume)", {"turn": turn, "iteration": iteration}, force_flush=True):
                                phoenix.set_attribute("openinference.span.kind", "LLM")
                                phoenix.set_attribute("llm.provider", reflector_provider)
                                phoenix.set_attribute("llm.model_name", _refl_model_label)
                                phoenix.set_large_attribute("input.value", _reflector_user_msg)
                                phoenix.set_large_attribute("output.value", _reflector_response)
                                phoenix.set_attribute("verdict", _reflector_verdict)

                        except Exception as _refl_err:
                            emit(EventType.SYSTEM,
                                 f"   ⚠️ Reflector error (non-fatal): {_refl_err}")
                            emit(EventType.REFLECTOR_CONTEXT, "", {
                                "turn": turn,
                                "iteration": iteration,
                                "phase": "result",
                                "verdict": "ERROR",
                                "thinking": "",
                                "response": f"Reflector error: {_refl_err}",
                            })
                            _reflector_verdict = None

                        # ── Handle reflector verdict (checkpoint resume) ──
                        if _reflector_verdict == "APPROVE":
                            # Reflector approved → accept solution directly
                            emit(EventType.SYSTEM,
                                 "   ✅ Independent reflector APPROVED — accepting solution!")
                            solved = True
                            last_test_visual_parts = test_prediction_only_parts

                            update_internal_test_score(solution_branches=active_solution_branches)
                            break  # Exit while loop

                        elif _reflector_verdict == "REJECT":
                            last_reflector_rejection = _reflector_response

                            current_context_tokens = last_actual_input_tokens + last_actual_output_tokens
                            _bypass_compression = (
                                current_context_tokens == 0
                                or current_context_tokens < compression_bypass_threshold
                            )

                            if _bypass_compression:
                                emit(EventType.SYSTEM,
                                     f"   ℹ️ Context below bypass threshold ({current_context_tokens:,} < {compression_bypass_threshold:,}) — continuing without context reset")
                                feedback_text = _build_reflector_reject_feedback(_reflector_response, bypass_compression=True)
                                feedback_content = [{"type": "text", "text": feedback_text}]
                                messages.append({"role": "user", "content": feedback_content})

                                iteration += 1
                                emit(EventType.SYSTEM, f"   📊 Starting Iteration {iteration}/{max_iterations}")
                                emit(
                                    EventType.REFLECTION,
                                    "",
                                    _iteration_context_metadata(
                                        iteration=iteration,
                                        ui_title=f"📋 **Iteration {iteration} — Full Context (bypass)**",
                                        context_mode="snapshot",
                                        snapshot_messages=messages,
                                        resume_messages=messages,
                                    ),
                                )
                                last_test_visual_parts = test_prediction_only_parts
                            else:
                                # Reflector rejected → feed back to main agent + compress context in one prompt
                                emit(EventType.SYSTEM,
                                     "   🔄 Independent reflector REJECTED — feeding back to solver with context compression...")

                                combined_prompt = _build_reflector_reject_feedback(_reflector_response) + CONTEXT_COMPRESSION_PROMPT

                                combined_content = [{"type": "text", "text": combined_prompt}]
                                snapshot_messages = messages[current_iteration_start_idx:] + [
                                    {"role": "user", "content": combined_content}
                                ]
                                messages.append({"role": "user", "content": combined_content})

                                emit(EventType.TURN_START, "", {
                                     "turn": turn, "iteration": iteration, "max_iterations": max_iterations,
                                     "context_label": "Solver (post-reflector)"})
                                emit(
                                    EventType.REFLECTION,
                                    "",
                                    _iteration_context_metadata(
                                        iteration=iteration,
                                        ui_title="📋 **Solver (post-reflector)**",
                                        context_mode="snapshot",
                                        snapshot_messages=snapshot_messages,
                                        resume_messages=messages,
                                    ),
                                )
                                in_reflection_mode = True
                                in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                                    test_generalization=False,
                                    reflector_reject_compression=True,
                                )
                                last_test_visual_parts = test_prediction_only_parts

                                with phoenix.span("🪞 Reflector Reject + Context Compression (Checkpoint)", {"turn": turn, "iteration": iteration}, force_flush=True):
                                    phoenix.set_large_attribute("prompt", combined_prompt)

                        elif _reflector_verdict == "EXPAND_CANDIDATES":

                            current_context_tokens = last_actual_input_tokens + last_actual_output_tokens
                            _bypass_compression = (
                                current_context_tokens == 0
                                or current_context_tokens < compression_bypass_threshold
                            )

                            if _bypass_compression:
                                emit(EventType.SYSTEM,
                                     f"   ℹ️ Context below bypass threshold ({current_context_tokens:,} < {compression_bypass_threshold:,}) — continuing without context reset")
                                expansion_feedback = build_candidate_expansion_guidance_prompt(
                                    reflector_response=_reflector_response,
                                )
                                messages.append({"role": "user", "content": [{"type": "text", "text": expansion_feedback}]})

                                iteration += 1
                                emit(EventType.SYSTEM, f"   📊 Starting Iteration {iteration}/{max_iterations}")
                                emit(EventType.SYSTEM, "", {
                                    "_reflector_prompt_bundle": True,
                                    "system_md": system_prompt,
                                    "user_md": "",
                                    "iteration": iteration,
                                })
                                last_test_visual_parts = test_prediction_only_parts
                                current_iteration_start_idx = len(messages)
                                # Clear reflection flags so the main loop doesn't
                                # trigger a compression turn after the solver responds
                                in_reflection_mode = False
                                in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                                    test_generalization=False,
                                    reflector_reject_compression=False,
                                )

                            else:
                                emit(EventType.SYSTEM,
                                     "   🌿 Independent reflector requested candidate expansion — feeding back to solver with context compression...")

                                expansion_feedback = build_candidate_expansion_guidance_prompt(
                                    reflector_response=_reflector_response,
                                )
                                combined_prompt = expansion_feedback + CONTEXT_COMPRESSION_PROMPT

                                combined_content = [{"type": "text", "text": combined_prompt}]
                                snapshot_messages = messages[current_iteration_start_idx:] + [
                                    {"role": "user", "content": combined_content}
                                ]
                                messages.append({"role": "user", "content": combined_content})

                                emit(EventType.TURN_START, "", {
                                     "turn": turn, "iteration": iteration, "max_iterations": max_iterations,
                                     "context_label": "Solver (post-reflector)"})
                                # Note: _reflector_prompt_bundle is intentionally NOT emitted here;
                                # the REFLECTION event below carries system_md and snapshot_messages,
                                # so emitting both would render the system prompt twice in the UI.
                                emit(
                                    EventType.REFLECTION,
                                    "",
                                    _iteration_context_metadata(
                                        iteration=iteration,
                                        ui_title="📋 **Solver (post-reflector — expansion)**",
                                        context_mode="snapshot",
                                        snapshot_messages=snapshot_messages,
                                        resume_messages=messages,
                                    ),
                                )
                                in_reflection_mode = True
                                in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                                    test_generalization=False,
                                    reflector_reject_compression=True,
                                )
                                last_test_visual_parts = test_prediction_only_parts

                                with phoenix.span("🌿 Reflector Expansion + Context Compression (Checkpoint)", {"turn": turn, "iteration": iteration}, force_flush=True):
                                    phoenix.set_large_attribute("prompt", combined_prompt)

                        else:
                            # ERROR or unknown → fall back to main agent self-reflection
                            iteration += 1
                            emit(EventType.TURN_START, "", {
                                 "turn": turn, "iteration": iteration, "max_iterations": max_iterations,
                                 "context_label": "Solver (post-reflector)"})
                            emit(EventType.SYSTEM, "", {
                                "_reflector_prompt_bundle": True,
                                "system_md": system_prompt,
                                "user_md": "",
                                "iteration": iteration,
                            })

                    elif enable_independent_reflector and not last_transform_hypothesis:
                        emit(EventType.SYSTEM,
                             f"   ⚠️ No transform hypothesis recorded — skipping independent reflector")

                    # Self-reflection: only when reflector didn't give a clear verdict
                    if not solved and not in_reflector_reject_compression and _reflector_verdict not in ("APPROVE", "REJECT", "EXPAND_CANDIDATES"):
                        generalization_reflection = TEST_GENERALIZATION_REFLECTION_PROMPT
                        in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                            test_generalization=True,
                            reflector_reject_compression=False,
                        )

                        generalization_content = [
                            {"type": "text", "text": generalization_reflection}] + test_visual_parts
                        messages.append(
                            {"role": "user", "content": generalization_content})
                        emit(EventType.REFLECTION, generalization_reflection,
                             {"iteration": iteration})
                        emit(
                            EventType.SYSTEM, f"   ✅ Rebuilt test generalization reflection prompt with {len(test_visual_parts)} visual parts")
                        with phoenix.span("🪞 Test Generalization Reflection Prompt (Rebuilt)", {"turn": turn, "iteration": iteration}, force_flush=True):
                            phoenix.set_attribute(
                                "num_visual_parts", len(test_visual_parts))
                            phoenix.set_large_attribute(
                                "prompt", generalization_reflection)
                else:
                    # Fallback: can't re-run code, use plain reflection
                    in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                        test_generalization=False,
                        reflector_reject_compression=False,
                    )
                    emit(
                        EventType.SYSTEM, "   ⚠️ Could not re-run solution for test predictions, falling back to plain reflection")

            if not in_test_generalization_reflection and not in_reflector_reject_compression and _reflector_verdict not in ("EXPAND_CANDIDATES",):
                emit(EventType.SYSTEM, "   🔄 Checkpoint resume in reflection mode — rebuilding reflection prompt with latest logic...")
                # Rebuild regular (train-failure) reflection prompt
                reflection_text = TRAIN_FAILURE_REFLECTION_PROMPT
                reflection_content = [
                    {"type": "text", "text": reflection_text}]
                messages.append(
                    {"role": "user", "content": reflection_content})
                emit(EventType.REFLECTION, reflection_text,
                     {"iteration": iteration})
                emit(EventType.SYSTEM, "   ✅ Rebuilt train-failure reflection prompt")
                with phoenix.span("🪞 Train Failure Reflection Prompt (Rebuilt)", {"turn": turn, "iteration": iteration}, force_flush=True):
                    phoenix.set_large_attribute("prompt", reflection_text)

        # After rebuilding reflection prompt during checkpoint resume, update the iteration start index
        # so the newly appended reflection prompt doesn't get filtered out
        if turn == initial_turn + 1 and in_reflection_mode and initial_messages:
            # The reflection prompt was just appended - mark everything from here as current iteration
            current_iteration_start_idx = len(messages) - 1
            emit(EventType.SYSTEM, f"   🔄 Updated iteration start index to {current_iteration_start_idx} (reflection prompt position)")

        # Reflection prompt will be emitted after tool execution if needed
        # (Don't emit a generic status message here)

        # Filter messages (remove old exploratory code calls from previous iterations)
        filtered_messages = []
        i = 0
        while i < len(messages):
            if i >= current_iteration_start_idx:
                filtered_messages.append(messages[i])
                i += 1
                continue

            msg = messages[i]
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    tool_uses = [b for b in content if isinstance(
                        b, dict) and b.get("type") == "tool_use"]
                    if tool_uses and all(tu.get("name") in EXPLORATORY_CODE_TOOL_NAMES for tu in tool_uses):
                        tool_use_ids = {tu.get("id") for tu in tool_uses}
                        i += 1
                        while i < len(messages) and messages[i].get("role") == "user":
                            user_content = messages[i].get("content", [])
                            if isinstance(user_content, list):
                                tool_results = [b for b in user_content if isinstance(
                                    b, dict) and b.get("type") == "tool_result"]
                                if tool_results and all(tr.get("tool_use_id") in tool_use_ids for tr in tool_results):
                                    i += 1
                                    continue
                            break
                        continue
            filtered_messages.append(msg)
            i += 1

        messages_to_use = filtered_messages
        emit(EventType.SYSTEM,
             f"   📝 Filtered history: keeping {len(filtered_messages)}/{len(messages)} items (current iteration from index {current_iteration_start_idx})")

        # ========== CONTEXT COMPRESSION CHECK ==========
        # Two independent estimates:
        #   1) Carryover: last API's billed input tokens + output tokens
        #      (most accurate when available, including Anthropic prompt-cache
        #      reads/writes that can dwarf the uncached input_tokens field)
        #   2) Fresh char-based estimate on current messages_to_use (used only before first turn)
        # Prefer carryover when available — it reflects real API measurements, not estimates.
        # Fresh estimate is only used on the very first turn (no prior API data).
        fresh_estimate = _estimate_message_tokens(messages_to_use, system_prompt)
        carryover_estimate = last_actual_input_tokens + last_actual_output_tokens
        estimated_tokens = carryover_estimate if carryover_estimate > 0 else fresh_estimate

        if carryover_estimate > 0:
            emit(EventType.SYSTEM,
                 f"   📊 Tokens: {estimated_tokens:,} (fresh={fresh_estimate:,}, carryover={carryover_estimate:,}) / {compression_threshold:,} threshold")
        else:
            emit(EventType.SYSTEM,
                 f"   📊 Estimated tokens: {estimated_tokens:,} / {compression_threshold:,} threshold")

        # Kimi+Fireworks: avoid pre-first-turn compression stall on large multimodal prompts.
        # This reduces "blank first response" latency and makes stop/cancel responsive.
        _skip_first_turn_compression_for_kimi = (
            _use_fireworks
            and turn == 1
            and carryover_estimate == 0
            and len(messages_to_use) <= 1
        )
        if _skip_first_turn_compression_for_kimi and estimated_tokens >= compression_threshold:
            emit(EventType.SYSTEM, "   ℹ️ Skipping first-turn auto-compression for Kimi to reduce initial latency.")

        _skip_auto_compression_after_hypothesis = (
            pending_execute_after_hypothesis_submission
            and estimated_tokens >= compression_threshold
            and compression_count < 3
            and not in_reflection_mode
            and not _skip_first_turn_compression_for_kimi
        )
        if _skip_auto_compression_after_hypothesis:
            pending_execute_after_hypothesis_submission = False
            emit(
                EventType.SYSTEM,
                "   ℹ️ Skipping auto-compression once because execute_python_solution is expected immediately after submit_transform_hypothesis.",
            )

        _skip_auto_compression_after_guidance = (
            pending_guided_followup
            and estimated_tokens >= compression_threshold
            and compression_count < 3
            and not in_reflection_mode
            and not _skip_first_turn_compression_for_kimi
            and not _skip_auto_compression_after_hypothesis
        )
        if _skip_auto_compression_after_guidance:
            pending_guided_followup = False
            emit(
                EventType.SYSTEM,
                "   ℹ️ Skipping auto-compression once because a guided branch follow-up is expected next turn.",
            )

        if (
            estimated_tokens >= compression_threshold
            and compression_count < 3
            and not in_reflection_mode
            and not _skip_first_turn_compression_for_kimi
            and not _skip_auto_compression_after_hypothesis
            and not _skip_auto_compression_after_guidance
        ):
            # Always increment to avoid infinite re-check on same turn
            compression_count += 1

            # Guard: if there's only 1 message (the initial user prompt), there's nothing to compress.
            # This happens when system_prompt + user_prompt alone exceed the threshold (e.g., many large images).
            if len(messages_to_use) <= 1:
                emit(EventType.SYSTEM,
                     f"\n⚠️  Context already at {estimated_tokens:,} tokens with just the initial prompt — nothing to compress. "
                     f"Consider reducing images or puzzle complexity.")
            else:
                emit(EventType.SYSTEM,
                     f"\n⚠️  Context approaching limit ({estimated_tokens:,} tokens). Triggering automatic compression...")
                _invalidate_exploratory_runtime()

                # Show the compression prompt in the UI so the user sees what was asked
                emit(EventType.REFLECTION, CONTEXT_COMPRESSION_PROMPT,
                     {"iteration": iteration, "compression_prompt": True})

                compression_messages = messages_to_use + [
                    {"role": "user", "content": [
                        {"type": "text", "text": CONTEXT_COMPRESSION_PROMPT}]}
                ]

                try:
                    with phoenix.span("🗜️ Context Compression", {"turn": turn, "iteration": iteration, "compression_count": compression_count}, force_flush=True):
                        phoenix.set_attribute(
                            "estimated_tokens_before", estimated_tokens)
                        phoenix.set_attribute(
                            "message_count_before", len(messages_to_use))

                        # Make compression API call (streaming, no tools)
                        compression_messages = _sanitize_messages(
                            compression_messages,
                            image_source_mode=image_source_mode,
                            force_png_reencode=_use_fireworks,
                        )
                        _comp_model_lc = str(model_name or "").lower()
                        if "haiku-4-5" in _comp_model_lc:
                            _comp_max_tokens = 64000
                        elif "kimi" in _comp_model_lc:
                            _comp_max_tokens = 128000
                        else:
                            _comp_max_tokens = 128000
                        compression_params = {
                            "model": model_name,
                            "max_tokens": _comp_max_tokens,
                            "system": system_prompt,
                            "messages": compression_messages,
                        }
                        if _use_fireworks:
                            compression_params["timeout"] = _fireworks_timeout()

                        # Apply compression thinking effort
                        if use_extended_thinking:
                            model_lc = str(model_name or "").lower()
                            if "haiku-4-5" in model_lc or "kimi" in model_lc:
                                if "kimi" in model_lc:
                                    _budget_cap = min(
                                        _comp_max_tokens - 1, KIMI_COMPRESSION_THINKING_BUDGET_CAP
                                    )
                                else:
                                    _budget_cap = _comp_max_tokens - 1
                                budget_tokens = max(
                                    1024, min(int(thinking_budget or 1024), _budget_cap))
                                compression_params["thinking"] = {
                                    "type": "enabled", "budget_tokens": budget_tokens}
                            elif "glm" in model_lc:
                                # GLM path: use explicit thinking budget only.
                                budget_tokens = int(thinking_budget or 1024)
                                budget_tokens = max(
                                    1024, min(budget_tokens, _comp_max_tokens - 1))
                                compression_params["thinking"] = {
                                    "type": "enabled", "budget_tokens": budget_tokens}
                            else:
                                allowed_efforts = ["low", "medium", "high", "max"]
                                normalized_effort = str(
                                    compression_thinking_effort or "medium").strip().lower()
                                if normalized_effort not in allowed_efforts:
                                    normalized_effort = "medium"
                                compression_params["thinking"] = {
                                    "type": "adaptive"}
                                compression_params["output_config"] = {
                                    "effort": normalized_effort}

                        compression_params = _apply_anthropic_prompt_caching(
                            compression_params,
                            use_fireworks=_use_fireworks,
                            include_tools=False,
                        )

                        emit(EventType.SYSTEM,
                             f"   🤖 Requesting context summary from model...")

                        # Log invocation params for Phoenix
                        try:
                            _comp_raw_req = json.dumps(compression_params, default=str)
                        except Exception:
                            _comp_raw_req = str(compression_params)

                        # Stream response with MessagesStream span (same pattern as regular agent calls)
                        summary_text = ""
                        summary_thinking = ""
                        final_message_obj = None

                        with phoenix.span(
                            "MessagesStream",
                            {
                                "openinference.span.kind": "LLM",
                                "llm.system": "anthropic",
                                "llm.model_name": model_name,
                                "turn": turn,
                                "compression": True,
                            },
                            force_flush=True,
                        ):
                            _comp_inv_params = {
                                k: v for k, v in compression_params.items() if k not in ("messages", "system")}
                            try:
                                _comp_inv_str = json.dumps(_comp_inv_params, default=str)
                            except Exception:
                                _comp_inv_str = str(_comp_inv_params)
                            phoenix.set_attribute(
                                "llm.invocation_parameters", _comp_inv_str)
                            phoenix.set_attribute(
                                "input.mime_type", "application/json")
                            phoenix.set_attribute("input.value", _comp_raw_req)
                            _oi_set_input_messages(
                                phoenix, system_prompt, compression_messages)

                            _comp_stopped_mid_stream = False
                            _comp_attempt = 0
                            _comp_max_retries = _max_retries_for_model(_use_fireworks)
                            while True:
                                _comp_attempt += 1
                                try:
                                    with _suppress_auto_instrumentation(), client.messages.stream(**compression_params) as stream:
                                        for event in stream:
                                            if should_stop():
                                                _comp_stopped_mid_stream = True
                                                break
                                            if event.type == "content_block_start":
                                                if hasattr(event, 'content_block'):
                                                    block = event.content_block
                                                    if hasattr(block, 'type'):
                                                        if block.type == "thinking":
                                                            pass  # thinking block started
                                                        elif block.type == "text":
                                                            pass  # text block started

                                            elif event.type == "content_block_delta":
                                                if hasattr(event, 'delta'):
                                                    delta = event.delta
                                                    if hasattr(delta, 'type'):
                                                        if delta.type == "text_delta":
                                                            chunk = delta.text
                                                            summary_text += chunk
                                                            # Stream to UI
                                                            emit(EventType.TEXT, summary_text)
                                                        elif delta.type == "thinking_delta":
                                                            chunk = delta.thinking
                                                            summary_thinking += chunk
                                                            emit(EventType.THINKING, summary_thinking)

                                        # Get final message for usage stats only when not stopped.
                                        if not _comp_stopped_mid_stream:
                                            try:
                                                final_message_obj = stream.get_final_message()
                                            except Exception:
                                                final_message_obj = None
                                    break
                                except Exception as _comp_err:
                                    if should_stop():
                                        _comp_stopped_mid_stream = True
                                        break
                                    if _is_transient_api_error(_comp_err) and _comp_attempt <= _comp_max_retries:
                                        _backoff = min(12, 2 ** _comp_attempt)
                                        emit(EventType.SYSTEM, f"   ⚠️ Compression stream retry {_comp_attempt}/{_comp_max_retries}: {_comp_err}")
                                        for _ in range(int(_backoff * 10)):
                                            if should_stop():
                                                _comp_stopped_mid_stream = True
                                                break
                                            time.sleep(0.1)
                                        if _comp_stopped_mid_stream:
                                            break
                                        continue
                                    raise

                            # Log output to Phoenix
                            _oi_set_output_messages(
                                phoenix, summary_text, tool_uses=[])
                            phoenix.set_attribute(
                                "output.mime_type", "application/json")
                            if final_message_obj is not None:
                                try:
                                    _comp_raw_final = final_message_obj.model_dump()
                                    _comp_out_str = json.dumps(_comp_raw_final, default=str)
                                except Exception:
                                    _comp_out_str = str(final_message_obj)
                                if len(_comp_out_str) > 200000:
                                    _comp_out_str = _comp_out_str[:200000]
                                phoenix.set_attribute("output.value", _comp_out_str)

                                # Extract usage stats
                                if hasattr(final_message_obj, 'usage'):
                                    usage_meta = _extract_usage_meta(final_message_obj.usage)
                                    emit(EventType.SYSTEM, "", {
                                        "_token_usage": usage_meta})

                        if summary_text:
                            emit(
                                EventType.SYSTEM, f"   ✅ Received summary ({len(summary_text):,} chars)")
                            phoenix.set_large_attribute(
                                "compression_summary", summary_text)

                            # Build single user message: grids + guiding1 + summary + guiding2
                            guiding_prompt_1 = "\n\nYou previously worked on this puzzle and left a handoff note describing what you've observed, tried, and learned:\n\n"
                            guiding_prompt_2 = "\n\n---\n\nYour handoff note ends here. Please continue solving the puzzle from where you left off."

                            # Agent produced the summary — render as AGENT message (no prefix)
                            emit(EventType.TEXT, summary_text)

                            # Single user message matching Phoenix structure exactly
                            rebuilt_content = user_prompt_content + [
                                {"type": "text", "text": guiding_prompt_1 + summary_text + guiding_prompt_2}
                            ]
                            rebuilt_content = _append_kimi_run_code_guidance_if_needed(
                                rebuilt_content, model_name
                            )
                            messages = [
                                {"role": "user", "content": rebuilt_content}
                            ]
                            messages_to_use = messages
                            current_iteration_start_idx = 0

                            # Reset carryover since context was rebuilt
                            last_actual_input_tokens = 0
                            last_actual_output_tokens = 0

                            # Increment iteration: post-compression is a new context window → new canvas
                            iteration += 1
                            emit(
                                EventType.SYSTEM, f"   📊 Starting Iteration {iteration}/{max_iterations} (post-compression)")

                            # New canvas: emit TURN_START so the UI iteration selector adds a new entry
                            emit(EventType.TURN_START, "", {
                                "turn": turn, "iteration": iteration,
                                "max_iterations": max_iterations,
                                "context_label": f"Compressed Context #{compression_count}",
                            })

                            # Build interleaved blocks matching the API message structure exactly
                            _ctx_blocks: list[dict] = []
                            for _blk in user_prompt_content:
                                if not isinstance(_blk, dict):
                                    continue
                                if _blk.get("type") == "text":
                                    _t = str(_blk.get("text", "")).strip()
                                    if _t:
                                        _ctx_blocks.append({"type": "text", "content": _t})
                                elif _blk.get("type") == "image":
                                    _src = _blk.get("source", {})
                                    if _src.get("data"):
                                        _ctx_blocks.append({"type": "image", "content":
                                            f"data:{_src.get('media_type', 'image/png')};base64,{_src['data']}"})
                            # Add the complete continuation text block (guiding1 + summary + guiding2)
                            _ctx_blocks.append({"type": "text", "content":
                                guiding_prompt_1 + summary_text + guiding_prompt_2})

                            emit(
                                EventType.REFLECTION,
                                "",
                                _iteration_context_metadata(
                                    iteration=iteration,
                                    ui_title=f"📋 **Rebuilt Context Window (Compression #{compression_count})**",
                                    context_mode="handoff",
                                    prompt_blocks=rebuilt_content,
                                    resume_messages=messages_to_use,
                                ),
                            )

                            new_token_estimate = _estimate_message_tokens(
                                messages_to_use, system_prompt)
                            phoenix.set_attribute(
                                "estimated_tokens_after", new_token_estimate)
                            phoenix.set_attribute(
                                "message_count_after", len(messages_to_use))
                            emit(
                                EventType.SYSTEM, f"   📉 New token estimate: {new_token_estimate:,} (reduced from {estimated_tokens:,})")
                        else:
                            emit(EventType.SYSTEM,
                                 f"   ⚠️ Compression failed - empty summary received")

                except Exception as comp_err:
                    emit(EventType.SYSTEM, f"   ❌ Compression failed: {comp_err}")
                    # Continue anyway - may still work or fail with original error

        try:
            # Build API params
            actual_thinking_budget = min(
                thinking_budget, 16000) if use_extended_thinking else 0
            model_lc = str(model_name or "").lower()
            if "haiku-4-5" in model_lc:
                model_max_tokens = 64000
            elif "kimi" in model_lc:
                model_max_tokens = 128000
            else:
                model_max_tokens = 128000
            # Sanitize messages to remove empty text content blocks (Anthropic API requirement)
            # Log any empty text blocks found for debugging
            for mi, msg in enumerate(messages_to_use):
                content = msg.get("content")
                if isinstance(content, list):
                    for bi, block in enumerate(content):
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if not text or not text.strip():
                                emit(
                                    EventType.SYSTEM, f"   ⚠️ SANITIZE: Found empty text block in msg[{mi}] (role={msg.get('role')}) block[{bi}]")
                elif isinstance(content, str) and not content.strip():
                    emit(
                        EventType.SYSTEM, f"   ⚠️ SANITIZE: Found empty string content in msg[{mi}] (role={msg.get('role')})")
            messages_to_use = _sanitize_messages(
                messages_to_use,
                image_source_mode=image_source_mode,
                force_png_reencode=_use_fireworks,
            )

            api_params = {
                "model": model_name,
                "max_tokens": model_max_tokens,
                "system": system_prompt,
                "messages": messages_to_use,
            }
            if _use_fireworks:
                api_params["timeout"] = _fireworks_timeout()

            # Semi-CoT first turn: remove exploratory code tools but keep
            # submission tools (submit_transform_hypothesis, execute_python_solution)
            # so the model knows its goal, while being forced to reason visually.
            _is_first_fresh_turn = semi_cot_first_turn and turn == 1 and initial_turn == 0
            if not in_reflection_mode and not _is_first_fresh_turn:
                api_params["tools"] = tools
            elif _is_first_fresh_turn:
                _submission_tools = [
                    t for t in tools
                    if t.get("name") in ("submit_transform_hypothesis", "execute_python_solution")
                ]
                api_params["tools"] = _submission_tools
                _semi_cot_guidance = "For this first response, reason through the puzzle visually without running exploratory code. You will have access to code execution tools (run_code, run_code_in_previous_runtime) in subsequent turns."
                api_params["messages"] = list(api_params["messages"]) + [
                    {"role": "user", "content": _semi_cot_guidance}
                ]
                emit(EventType.SYSTEM, _semi_cot_guidance, {"_visible_message": True})

            if use_extended_thinking:
                if "haiku-4-5" in model_lc or "kimi" in model_lc:
                    # Haiku 4.5 and Fireworks models: use enabled thinking with explicit budget tokens.
                    # Fireworks does not support adaptive thinking type.
                    budget_tokens = int(thinking_budget or 1024)
                    max_tokens = int(api_params.get(
                        "max_tokens", 128000) or 128000)
                    # budget_tokens must be less than max_tokens.
                    budget_tokens = max(
                        1024, min(budget_tokens, max_tokens - 1))
                    api_params["thinking"] = {
                        "type": "enabled", "budget_tokens": budget_tokens}
                elif "glm" in model_lc:
                    # GLM path: use explicit thinking budget only.
                    budget_tokens = int(thinking_budget or 1024)
                    max_tokens = int(api_params.get(
                        "max_tokens", 128000) or 128000)
                    budget_tokens = max(
                        1024, min(budget_tokens, max_tokens - 1))
                    api_params["thinking"] = {
                        "type": "enabled", "budget_tokens": budget_tokens}
                else:
                    # Opus 4.6 and Sonnet 4.6 share adaptive + effort API surface.
                    allowed_efforts = ["low", "medium", "high", "max"]
                    _effective_effort = semi_cot_thinking_effort if _is_first_fresh_turn else thinking_effort
                    normalized_effort = str(
                        _effective_effort or "low").strip().lower()
                    if normalized_effort not in allowed_efforts:
                        normalized_effort = allowed_efforts[0]
                        emit(
                            EventType.SYSTEM, f"   ℹ️ Adjusted thinking_effort to '{normalized_effort}' for model compatibility.")
                    api_params["thinking"] = {"type": "adaptive"}
                    api_params["output_config"] = {"effort": normalized_effort}

            api_params = _apply_anthropic_prompt_caching(
                api_params,
                use_fireworks=_use_fireworks,
                include_tools=not in_reflection_mode,
            )

            # Log raw request to Phoenix
            try:
                raw_request = json.dumps(api_params, default=str)
            except Exception:
                raw_request = str(api_params)

            with phoenix.span("📤 Raw Anthropic Request", {"turn": turn, "iteration": iteration}, force_flush=True):
                phoenix.set_attribute("raw_request_head", raw_request[:50000])
                phoenix.set_attribute(
                    "raw_request_tail", raw_request[-50000:] if len(raw_request) > 50000 else raw_request)
                phoenix.set_large_attribute("raw_request_full", raw_request)
                phoenix.set_attribute("raw_request_size", len(raw_request))

            # Make API call
            _stopped_mid_stream = False
            if use_streaming:
                aggregated_text = ""
                aggregated_thinking = ""
                thinking_signature = ""
                tool_uses = []
                current_block_index = -1
                current_block_type = None

                final_message_obj = None
                with phoenix.span(f"turn_{turn}", {"turn": turn, "iteration": iteration}, force_flush=True):
                    invocation_params = {
                        k: v for k, v in api_params.items() if k not in ("messages", "system")}
                    try:
                        inv_str = json.dumps(invocation_params, default=str)
                    except Exception:
                        inv_str = str(invocation_params)
                    if len(inv_str) > 200000:
                        inv_str = inv_str[:200000]

                    with phoenix.span(
                        "MessagesStream",
                        {
                            "openinference.span.kind": "LLM",
                            "llm.system": "anthropic",
                            "llm.model_name": model_name,
                            "turn": turn,
                        },
                        force_flush=True,
                    ):
                        phoenix.set_attribute(
                            "llm.invocation_parameters", inv_str)
                        phoenix.set_attribute(
                            "input.mime_type", "application/json")
                        phoenix.set_attribute("input.value", raw_request)
                        _oi_set_input_messages(
                            phoenix, system_prompt, messages_to_use)

                        # Retry loop for transient network errors (e.g. mid-stream TCP drop).
                        # On each retry the stream restarts from scratch, so all accumulated
                        # state must be cleared.  Deterministic API errors (4xx) are re-raised
                        # immediately — only RemoteProtocolError / connection errors are retried.
                        _max_stream_retries = _max_retries_for_model(_use_fireworks)
                        _stream_attempt = 0
                        _stopped_mid_stream = False
                        _transient_exhausted = False
                        while True:
                            _stream_attempt += 1
                            # Reset all accumulated state before (re-)starting the stream
                            aggregated_text = ""
                            aggregated_thinking = ""
                            thinking_signature = ""
                            tool_uses = []
                            current_block_index = -1
                            current_block_type = None
                            final_message_obj = None
                            _stopped_mid_stream = False
                            try:
                                with _suppress_auto_instrumentation(), client.messages.stream(**api_params) as stream:
                                    _past_thinking = False
                                    for event in stream:
                                        if should_stop():
                                            _stopped_mid_stream = True
                                            break

                                        if event.type == "content_block_start":
                                            if hasattr(event, 'content_block'):
                                                block = event.content_block
                                                current_block_index = event.index
                                                current_block_type = block.type
                                                if block.type in ("text", "tool_use"):
                                                    _past_thinking = True
                                                if block.type == "tool_use":
                                                    if not in_reflection_mode:
                                                        tool_uses.append({
                                                            "id": block.id,
                                                            "name": block.name,
                                                            "input": ""
                                                        })

                                        elif event.type == "content_block_delta":
                                            if hasattr(event, 'delta'):
                                                delta = event.delta
                                                if delta.type == "text_delta":
                                                    aggregated_text += delta.text
                                                    emit(EventType.TEXT, delta.text)
                                                elif delta.type == "thinking_delta":
                                                    aggregated_thinking += delta.thinking
                                                    emit(EventType.THINKING,
                                                         delta.thinking)
                                                elif delta.type == "signature_delta":
                                                    thinking_signature = delta.signature
                                                    emit(EventType.SYSTEM, "", {
                                                         "_thinking_signature": thinking_signature})
                                                elif delta.type == "input_json_delta":
                                                    if tool_uses and not in_reflection_mode:
                                                        tool_uses[-1]["input"] += delta.partial_json
                                                        if emit_tool_call_deltas:
                                                            emit(
                                                                EventType.TOOL_CALL,
                                                                "",
                                                                {
                                                                    "name": tool_uses[-1]["name"],
                                                                    "id": tool_uses[-1]["id"],
                                                                    "_partial": True,
                                                                    "input_raw": tool_uses[-1]["input"],
                                                                },
                                                            )

                                    # Only try to get final message if we didn't stop mid-stream.
                                    # get_final_message() consumes the rest of the stream and can
                                    # hang indefinitely if the response is still being generated.
                                    if not _stopped_mid_stream:
                                        try:
                                            final_message_obj = stream.get_final_message()
                                        except Exception:
                                            final_message_obj = None

                                # Stream completed successfully — exit retry loop
                                break

                            except Exception as _stream_err:
                                _is_transient = _is_transient_api_error(_stream_err)
                                if should_stop():
                                    emit(EventType.SYSTEM, "\n⏹️ Stop requested — aborting request.")
                                    _stopped_mid_stream = True
                                    break
                                if _is_transient and _stream_attempt <= _max_stream_retries:
                                    _backoff = min(12, 2 ** _stream_attempt)
                                    emit(EventType.SYSTEM,
                                         f"   ⚠️ Stream interrupted (attempt {_stream_attempt}/{_max_stream_retries}): {_stream_err}. Retrying in {_backoff}s...")
                                    for _ in range(int(_backoff * 10)):
                                        if should_stop():
                                            _stopped_mid_stream = True
                                            break
                                        time.sleep(0.1)
                                    if _stopped_mid_stream:
                                        break
                                    # continue while loop → retry
                                else:
                                    if _is_transient:
                                        emit(EventType.SYSTEM,
                                             f"   ❌ Stream failed after {_max_stream_retries} retries: {_stream_err}")
                                        _transient_exhausted = True
                                        break
                                    raise  # re-raise non-transient errors

                        if _transient_exhausted:
                            consecutive_transient_turn_failures += 1
                            emit(
                                EventType.SYSTEM,
                                (
                                    "   ⚠️ Request failed due to transient network/provider instability. "
                                    f"Consecutive transient turn failures: {consecutive_transient_turn_failures}/"
                                    f"{MAX_CONSECUTIVE_TRANSIENT_TURN_FAILURES}. Retrying on next turn."
                                ),
                            )
                            if consecutive_transient_turn_failures >= MAX_CONSECUTIVE_TRANSIENT_TURN_FAILURES:
                                emit(
                                    EventType.ERROR,
                                    (
                                        "Stopping run after repeated transient request failures "
                                        f"({consecutive_transient_turn_failures} consecutive turns)."
                                    ),
                                )
                                break
                            time.sleep(min(8, 2 ** min(consecutive_transient_turn_failures, 3)))
                            continue
                        else:
                            consecutive_transient_turn_failures = 0

                        raw_final = None
                        if final_message_obj is not None:
                            try:
                                raw_final = final_message_obj.model_dump()
                            except Exception:
                                raw_final = {"type": str(
                                    type(final_message_obj)), "repr": repr(final_message_obj)}
                            phoenix.set_large_attribute(
                                "raw_final_message", raw_final)

                            # Extract and emit token usage
                            if hasattr(final_message_obj, 'usage'):
                                usage_meta = _extract_usage_meta(final_message_obj.usage)
                                last_actual_input_tokens = usage_meta["billed_input_tokens"]
                                last_actual_output_tokens = usage_meta["output_tokens"]
                                emit(EventType.SYSTEM, "", {
                                     "_token_usage": usage_meta})

                        _oi_set_output_messages(
                            phoenix, aggregated_text, tool_uses)
                        phoenix.set_attribute(
                            "output.mime_type", "application/json")
                        if raw_final is not None:
                            try:
                                out_str = json.dumps(raw_final, default=str)
                            except Exception:
                                out_str = str(raw_final)
                            if len(out_str) > 200000:
                                out_str = out_str[:200000]
                            phoenix.set_attribute("output.value", out_str)

                    # If stopped mid-stream, skip all post-processing — data is incomplete.
                    # The main while loop will check should_stop() and exit cleanly.
                    if _stopped_mid_stream:
                        emit(EventType.SYSTEM, "\n⏹️ Stopped mid-stream — skipping incomplete turn.")
                        continue

                    # Emit TOOL_CALL events now that we have complete inputs
                    for tu in tool_uses:
                        # Parse the accumulated JSON string into a dict
                        try:
                            parsed_input = json.loads(tu["input"]) if isinstance(
                                tu["input"], str) else tu["input"]
                        except Exception:
                            parsed_input = {"raw": tu["input"]}
                        emit(EventType.TOOL_CALL, "", {
                             "name": tu["name"], "id": tu["id"], "input": parsed_input})

                # Log thinking to Phoenix
                if aggregated_thinking and thinking_signature:
                    with phoenix.span("💭 Model Thinking", {"message_type": "thought", "turn": turn}):
                        phoenix.set_attribute("content", aggregated_thinking)
                        phoenix.set_attribute(
                            "content_length", len(aggregated_thinking))
                        phoenix.set_attribute("role", "assistant_thought")

                # Log model text response to Phoenix
                if aggregated_text:
                    with phoenix.span("🗣️ Model Response", {"message_type": "text", "turn": turn}):
                        phoenix.set_attribute("content", aggregated_text)
                        phoenix.set_attribute(
                            "content_length", len(aggregated_text))
                        phoenix.set_attribute("role", "assistant")

                # Parse tool inputs
                for tu in tool_uses:
                    try:
                        tu["input"] = json.loads(
                            tu["input"]) if tu["input"] else {}
                    except json.JSONDecodeError:
                        tu["input"] = {}

                # Build assistant message content
                assistant_content = []
                if aggregated_thinking and thinking_signature:
                    assistant_content.append({
                        "type": "thinking",
                        "thinking": aggregated_thinking,
                        "signature": thinking_signature
                    })
                if aggregated_text:
                    assistant_content.append(
                        {"type": "text", "text": aggregated_text})
                for tu in tool_uses:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tu["id"],
                        "name": tu["name"],
                        "input": tu["input"]
                    })

                function_calls = [
                    {"id": tu["id"], "name": tu["name"], "input": tu["input"]} for tu in tool_uses]

            else:
                # Blocking mode
                with phoenix.span(f"turn_{turn}", {"turn": turn, "iteration": iteration}, force_flush=True):
                    invocation_params = {
                        k: v for k, v in api_params.items() if k not in ("messages", "system")}
                    try:
                        inv_str = json.dumps(invocation_params, default=str)
                    except Exception:
                        inv_str = str(invocation_params)
                    if len(inv_str) > 200000:
                        inv_str = inv_str[:200000]

                    with phoenix.span(
                        "MessagesStream",
                        {
                            "openinference.span.kind": "LLM",
                            "llm.system": "anthropic",
                            "llm.model_name": model_name,
                            "turn": turn,
                        },
                        force_flush=True,
                    ):
                        phoenix.set_attribute(
                            "llm.invocation_parameters", inv_str)
                        phoenix.set_attribute(
                            "input.mime_type", "application/json")
                        phoenix.set_attribute("input.value", raw_request)
                        _oi_set_input_messages(
                            phoenix, system_prompt, messages_to_use)

                        with _suppress_auto_instrumentation():
                            _create_attempt = 0
                            _create_max_retries = _max_retries_for_model(_use_fireworks)
                            while True:
                                _create_attempt += 1
                                try:
                                    response = client.messages.create(**api_params)
                                    break
                                except Exception as _create_err:
                                    if should_stop():
                                        raise
                                    if _is_transient_api_error(_create_err) and _create_attempt <= _create_max_retries:
                                        _backoff = min(12, 2 ** _create_attempt)
                                        emit(EventType.SYSTEM, f"   ⚠️ Create retry {_create_attempt}/{_create_max_retries}: {_create_err}")
                                        for _ in range(int(_backoff * 10)):
                                            if should_stop():
                                                break
                                            time.sleep(0.1)
                                        if should_stop():
                                            raise
                                        continue
                                    raise

                        try:
                            raw_resp = response.model_dump()
                        except Exception:
                            raw_resp = {"type": str(
                                type(response)), "repr": repr(response)}
                        phoenix.set_large_attribute("raw_response", raw_resp)
                        _oi_set_output_messages(phoenix, "", tool_uses=[])
                        phoenix.set_attribute(
                            "output.mime_type", "application/json")
                        try:
                            out_str = json.dumps(raw_resp, default=str)
                        except Exception:
                            out_str = str(raw_resp)
                        if len(out_str) > 200000:
                            out_str = out_str[:200000]
                        phoenix.set_attribute("output.value", out_str)

                    if hasattr(response, "usage"):
                        usage_meta = _extract_usage_meta(response.usage)
                        last_actual_input_tokens = usage_meta["billed_input_tokens"]
                        last_actual_output_tokens = usage_meta["output_tokens"]
                        emit(EventType.SYSTEM, "", {
                             "_token_usage": usage_meta})

                    aggregated_thinking = ""
                    thinking_signature = ""
                    aggregated_text = ""
                    tool_uses = []

                    for block in response.content:
                        if block.type == "thinking":
                            aggregated_thinking += block.thinking
                            emit(EventType.THINKING, block.thinking)
                            if hasattr(block, 'signature') and block.signature:
                                thinking_signature = block.signature
                                # Emit signature to UI for checkpointing
                                emit(EventType.SYSTEM, "", {
                                     "_thinking_signature": thinking_signature})
                        elif block.type == "text":
                            aggregated_text += block.text
                            emit(EventType.TEXT, block.text)
                        elif block.type == "tool_use":
                            if not in_reflection_mode:
                                tool_uses.append({
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input
                                })
                                emit(EventType.TOOL_CALL, "", {
                                     "name": block.name, "id": block.id, "input": block.input})

                    # Log thinking to Phoenix
                    if aggregated_thinking and thinking_signature:
                        with phoenix.span("💭 Model Thinking", {"message_type": "thought", "turn": turn}):
                            phoenix.set_attribute("content", aggregated_thinking)
                            phoenix.set_attribute(
                                "content_length", len(aggregated_thinking))
                            phoenix.set_attribute("role", "assistant_thought")

                    # Log model text response to Phoenix
                    if aggregated_text:
                        with phoenix.span("🗣️ Model Response", {"message_type": "text", "turn": turn}):
                            phoenix.set_attribute("content", aggregated_text)
                            phoenix.set_attribute(
                                "content_length", len(aggregated_text))
                            phoenix.set_attribute("role", "assistant")

                    # Build assistant message content
                    assistant_content = []
                    if aggregated_thinking and thinking_signature:
                        assistant_content.append({
                            "type": "thinking",
                            "thinking": aggregated_thinking,
                            "signature": thinking_signature
                        })
                    if aggregated_text:
                        assistant_content.append(
                            {"type": "text", "text": aggregated_text})
                    for tu in tool_uses:
                        assistant_content.append({
                            "type": "tool_use",
                            "id": tu["id"],
                            "name": tu["name"],
                            "input": tu["input"]
                        })

                    function_calls = tool_uses

            # Append assistant message to history
            if assistant_content:
                messages.append(
                    {"role": "assistant", "content": assistant_content})

            # Handle no tool calls
            if not function_calls:
                emit(EventType.SYSTEM, "\n⏹️ No tool calls.")

                # Semi-CoT first turn completed — inject tool-available message
                # and continue the loop so the agent can now use exploratory code tools.
                if _is_first_fresh_turn:
                    _tool_available_msg = "You now have access to run_code and run_code_in_previous_runtime for exploratory code execution. Use them to implement and verify the hypothesis you just developed."
                    messages.append({
                        "role": "user",
                        "content": _tool_available_msg,
                    })
                    emit(EventType.SYSTEM, _tool_available_msg, {"_visible_message": True})
                    continue

                if in_reflection_mode:
                    if in_reflector_reject_compression:
                        # Agent responded to reflector feedback + context compression in one prompt.
                        # The response IS the compressed context — no separate phase2 call needed.
                        emit(EventType.SYSTEM,
                             f"   ✅ Agent processed reflector feedback and wrote context compression ({len(aggregated_text):,} chars)")
                        in_reflection_mode = False
                        in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                            test_generalization=False,
                            reflector_reject_compression=False,
                        )

                        # Increment iteration
                        iteration += 1
                        emit(EventType.SYSTEM,
                             f"   📊 Starting Iteration {iteration}/{max_iterations}")

                        # Update reflection from agent's response
                        current_reflection = aggregated_text if aggregated_text else _extract_reflection_summary(messages)
                        if current_reflection:
                            last_reflection = current_reflection
                            emit(EventType.SYSTEM,
                                 f"   📝 Updated reflection from reflector-reject compression ({len(current_reflection)} chars)")

                        # Build consolidated summary for fresh context
                        # retry_after_pass=True because train passed 100% but reflector rejected
                        consolidated_content = _build_consolidated_summary(
                            iteration_history=iteration_history,
                            last_reflection=last_reflection,
                            last_failed_results=last_failed_results,
                            last_visual_parts=last_visual_parts,
                            verified_rules=verified_rules,
                            use_visual_mode=use_visual_mode,
                            retry_after_pass=True,
                            last_test_visual_parts=last_test_visual_parts,
                            reflector_rejection=last_reflector_rejection,
                        )
                        last_reflector_rejection = ""

                        if consolidated_content:
                            emit(EventType.SYSTEM,
                                 "   📋 Resetting context with consolidated iteration summary...")
                            combined_content = user_prompt_content + consolidated_content
                            combined_content = _append_kimi_run_code_guidance_if_needed(
                                combined_content, model_name
                            )

                            # Convert combined content to web UI format
                            interleaved_blocks = []
                            total_text_chars = 0
                            total_images = 0

                            for block in combined_content:
                                if isinstance(block, dict):
                                    if block.get("type") == "text":
                                        text_content = block.get("text", "")
                                        if text_content.strip():
                                            interleaved_blocks.append({
                                                "type": "text",
                                                "content": text_content
                                            })
                                            total_text_chars += len(text_content)
                                    elif block.get("type") == "image":
                                        source = block.get("source", {})
                                        if source.get("data"):
                                            img_data_url = f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
                                            interleaved_blocks.append({
                                                "type": "image",
                                                "content": img_data_url
                                            })
                                            total_images += 1

                            if interleaved_blocks:
                                emit(
                                    EventType.REFLECTION,
                                    "",
                                    _iteration_context_metadata(
                                        iteration=iteration,
                                        ui_title=f"📋 **Iteration {iteration} — Consolidated Context**",
                                        context_mode="handoff",
                                        prompt_blocks=combined_content,
                                        resume_messages=[{"role": "user", "content": combined_content}],
                                    ),
                                )
                                emit(EventType.SYSTEM,
                                     f"   📋 Consolidated prompt: {total_text_chars:,} chars, {total_images} images, {len(interleaved_blocks)} blocks")
                            messages = [
                                {"role": "user", "content": combined_content}
                            ]
                        else:
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "text",
                                    "text": (
                                        "Understood. Please proceed with your refined approach. "
                                        "Address the independent reviewer's concerns. "
                                        "You may use run_code or run_code_in_previous_runtime for experiments, or execute_python_solution to test your updated hypothesis."
                                    )
                                }]
                            })

                        current_iteration_start_idx = len(messages)
                        continue

                    elif in_test_generalization_reflection:
                        response_text = aggregated_text.upper() if aggregated_text else ""
                        if "ACCEPT" in response_text and "RETRY" not in response_text:
                            emit(
                                EventType.SYSTEM, "   ✅ Model ACCEPTED the solution after test generalization reflection!")
                            solved = True
                            in_reflection_mode = False
                            in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                                test_generalization=False,
                                reflector_reject_compression=False,
                            )
                            break
                        else:
                            in_reflection_mode = False
                            in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                                test_generalization=False,
                                reflector_reject_compression=False,
                            )

                            current_context_tokens = last_actual_input_tokens + last_actual_output_tokens
                            _bypass_compression = (
                                current_context_tokens == 0
                                or current_context_tokens < compression_bypass_threshold
                            )

                            if _bypass_compression:
                                emit(EventType.SYSTEM,
                                     f"   ℹ️ Context below bypass threshold ({current_context_tokens:,} < {compression_bypass_threshold:,}) — skipping Phase 2 compression")
                                iteration += 1
                                emit(EventType.SYSTEM, f"   📊 Starting Iteration {iteration}/{max_iterations}")
                                messages.append({"role": "user", "content": [{"type": "text", "text":
                                    "Continue with your revised approach. Address the issues you identified in your reflection. "
                                    "You may use run_code or run_code_in_previous_runtime for experiments, "
                                    "or execute_python_solution to test your updated hypothesis."
                                }]})
                                emit(
                                    EventType.REFLECTION,
                                    "",
                                    _iteration_context_metadata(
                                        iteration=iteration,
                                        ui_title=f"📋 **Iteration {iteration} — Full Context (bypass)**",
                                        context_mode="snapshot",
                                        snapshot_messages=messages,
                                        resume_messages=messages,
                                    ),
                                )
                                continue

                            emit(
                                EventType.SYSTEM, "   🔄 Model chose to RETRY - running Phase 2 context compression...")

                            # ── Context Compression (test-retry path) ────────────────────
                            # Phase 1 only produced a brief ACCEPT/RETRY verdict.
                            # Now compress the full session knowledge into a portable summary.
                            _invalidate_exploratory_runtime()
                            phase2_prompt = CONTEXT_COMPRESSION_PROMPT

                            phase2_messages = messages + [
                                {"role": "user", "content": [
                                    {"type": "text", "text": phase2_prompt}]}
                            ]

                            emit(EventType.REFLECTION, phase2_prompt,
                                 {"iteration": iteration})

                            phase2_summary = ""
                            phase2_thinking = ""
                            phase2_thinking_signature = ""
                            try:
                                with phoenix.span(f"turn_{turn}_phase2", {"turn": turn, "iteration": iteration}, force_flush=True):
                                    phase2_messages = _sanitize_messages(
                                        phase2_messages,
                                        image_source_mode=image_source_mode,
                                        force_png_reencode=_use_fireworks,
                                    )
                                    _p2_model_lc = str(model_name or "").lower()
                                    if "haiku-4-5" in _p2_model_lc:
                                        _p2_max_tokens = 64000
                                    elif "kimi" in _p2_model_lc:
                                        _p2_max_tokens = 128000
                                    else:
                                        _p2_max_tokens = 128000
                                    phase2_params = {
                                        "model": model_name,
                                        "max_tokens": _p2_max_tokens,
                                        "system": system_prompt,
                                        "messages": phase2_messages,
                                    }
                                    if _use_fireworks:
                                        phase2_params["timeout"] = _fireworks_timeout()
                                    if use_extended_thinking:
                                        model_lc = str(
                                            model_name or "").lower()
                                        if "haiku-4-5" in model_lc or "kimi" in model_lc:
                                            # Haiku 4.5 / Fireworks: use budget_tokens
                                            if "kimi" in model_lc:
                                                _budget_cap = min(
                                                    _p2_max_tokens - 1, KIMI_COMPRESSION_THINKING_BUDGET_CAP
                                                )
                                            else:
                                                _budget_cap = _p2_max_tokens - 1
                                            budget_tokens = max(
                                                1024, min(int(thinking_budget or 1024), _budget_cap))
                                            phase2_params["thinking"] = {
                                                "type": "enabled", "budget_tokens": budget_tokens}
                                        elif "glm" in model_lc:
                                            # GLM path: use explicit thinking budget only.
                                            budget_tokens = max(
                                                1024, min(int(thinking_budget or 1024), _p2_max_tokens - 1))
                                            phase2_params["thinking"] = {
                                                "type": "enabled", "budget_tokens": budget_tokens}
                                        else:
                                            # Opus/Sonnet: use categorical approach with reflection effort
                                            allowed_efforts = [
                                                "low", "medium", "high", "max"]
                                            normalized_effort = str(
                                                reflection_thinking_effort or "medium").strip().lower()
                                            if normalized_effort not in allowed_efforts:
                                                normalized_effort = "medium"
                                            phase2_params["thinking"] = {
                                                "type": "adaptive"}
                                            phase2_params["output_config"] = {
                                                "effort": normalized_effort}
                                        phase2_params["temperature"] = 1

                                    phase2_params = _apply_anthropic_prompt_caching(
                                        phase2_params,
                                        use_fireworks=_use_fireworks,
                                        include_tools=False,
                                    )

                                    # Log raw request to Phoenix
                                    try:
                                        phase2_raw_request = json.dumps(
                                            phase2_params, default=str)
                                    except Exception:
                                        phase2_raw_request = str(phase2_params)

                                    invocation_params_p2 = {
                                        k: v for k, v in phase2_params.items() if k not in ("messages", "system")}
                                    try:
                                        inv_str_p2 = json.dumps(
                                            invocation_params_p2, default=str)
                                    except Exception:
                                        inv_str_p2 = str(invocation_params_p2)
                                    if len(inv_str_p2) > 200000:
                                        inv_str_p2 = inv_str_p2[:200000]

                                    with phoenix.span(
                                        "MessagesStream",
                                        {
                                            "openinference.span.kind": "LLM",
                                            "llm.system": "anthropic",
                                            "llm.model_name": model_name,
                                            "turn": f"{turn}_phase2",
                                            "phase": "context_compression",
                                        },
                                        force_flush=True,
                                    ):
                                        phoenix.set_attribute(
                                            "llm.invocation_parameters", inv_str_p2)
                                        phoenix.set_attribute(
                                            "input.mime_type", "application/json")
                                        phoenix.set_attribute(
                                            "input.value", phase2_raw_request)
                                        _oi_set_input_messages(
                                            phoenix, system_prompt, phase2_messages)

                                        _phase2_stopped_mid_stream = False
                                        _phase2_attempt = 0
                                        _phase2_max_retries = _max_retries_for_model(_use_fireworks)
                                        while True:
                                            _phase2_attempt += 1
                                            try:
                                                with _suppress_auto_instrumentation(), client.messages.stream(**phase2_params) as stream:
                                                    for event in stream:
                                                        if should_stop():
                                                            _phase2_stopped_mid_stream = True
                                                            break

                                                        if event.type == "content_block_delta":
                                                            if hasattr(event, 'delta'):
                                                                delta = event.delta
                                                                if delta.type == "text_delta":
                                                                    phase2_summary += delta.text
                                                                    emit(
                                                                        EventType.TEXT, delta.text)
                                                                elif delta.type == "thinking_delta":
                                                                    phase2_thinking += delta.thinking
                                                                    emit(
                                                                        EventType.THINKING, delta.thinking)
                                                                elif delta.type == "signature_delta":
                                                                    phase2_thinking_signature = delta.signature
                                                                    emit(EventType.SYSTEM, "", {
                                                                         "_thinking_signature": phase2_thinking_signature})
                                                break
                                            except Exception as _phase2_err:
                                                if should_stop():
                                                    _phase2_stopped_mid_stream = True
                                                    break
                                                if _is_transient_api_error(_phase2_err) and _phase2_attempt <= _phase2_max_retries:
                                                    _backoff = min(12, 2 ** _phase2_attempt)
                                                    emit(EventType.SYSTEM, f"   ⚠️ Phase2 stream retry {_phase2_attempt}/{_phase2_max_retries}: {_phase2_err}")
                                                    for _ in range(int(_backoff * 10)):
                                                        if should_stop():
                                                            _phase2_stopped_mid_stream = True
                                                            break
                                                        time.sleep(0.1)
                                                    if _phase2_stopped_mid_stream:
                                                        break
                                                    continue
                                                raise

                                        try:
                                            phase2_final_message = None
                                            if not _phase2_stopped_mid_stream:
                                                phase2_final_message = stream.get_final_message()
                                            if phase2_final_message:
                                                phoenix.set_large_attribute(
                                                    "raw_final_message", phase2_final_message.model_dump())
                                                # If streaming produced no text (model only output thinking),
                                                # extract text from the final message content blocks and emit.
                                                if not phase2_summary and hasattr(phase2_final_message, 'content'):
                                                    for blk in (phase2_final_message.content or []):
                                                        if hasattr(blk, 'text') and blk.text:
                                                            phase2_summary += blk.text
                                                    if phase2_summary:
                                                        emit(EventType.TEXT,
                                                             phase2_summary)
                                                        emit(
                                                            EventType.SYSTEM, f"   ℹ️ Phase 2: Extracted text from final message (was missing from stream)")
                                                # Log usage to Phoenix and emit to Web UI
                                                if hasattr(phase2_final_message, 'usage') and phase2_final_message.usage:
                                                    _oi_set_output_messages(
                                                        phoenix, phase2_summary, tool_uses=[])
                                                    usage_meta = _extract_usage_meta(phase2_final_message.usage)
                                                    emit(EventType.SYSTEM, "", {
                                                         "_token_usage": usage_meta})
                                        except Exception:
                                            pass

                                    if phase2_summary:
                                        emit(
                                            EventType.SYSTEM, f"   ✅ Phase 2 context summary received ({len(phase2_summary):,} chars)")
                                        phoenix.set_large_attribute(
                                            "phase2_summary", phase2_summary)
                            except Exception as e:
                                emit(
                                    EventType.SYSTEM, f"   ⚠️ Phase 2 compression failed: {e} — falling back to Phase 1 response")

                            # Increment iteration AFTER both phases complete
                            iteration += 1
                            emit(
                                EventType.SYSTEM, f"   📊 Starting Iteration {iteration}/{max_iterations}")

                            # Update reflection from current iteration (before messages get reset)
                            # Use Phase 2 summary if available, else extract from current messages
                            current_reflection = phase2_summary if phase2_summary else _extract_reflection_summary(
                                messages)
                            if current_reflection:
                                last_reflection = current_reflection
                                emit(
                                    EventType.SYSTEM, f"   📝 Updated reflection from current iteration ({len(current_reflection)} chars)")

                            # Build consolidated summary for fresh context
                            # retry_after_pass=True because train passed 100% but model chose RETRY
                            consolidated_content = _build_consolidated_summary(
                                iteration_history=iteration_history,
                                last_reflection=last_reflection,
                                last_failed_results=last_failed_results,
                                last_visual_parts=last_visual_parts,
                                verified_rules=verified_rules,
                                use_visual_mode=use_visual_mode,
                                retry_after_pass=True,
                                last_test_visual_parts=last_test_visual_parts,
                            )

                            if consolidated_content:
                                # Reset messages: single consolidated user prompt (no fake assistant message)
                                emit(
                                    EventType.SYSTEM, "   📋 Resetting context with consolidated iteration summary...")
                                # Emit FULL consolidated prompt to UI (matching what Phoenix sees)
                                # This includes puzzle grids + iteration summary with dual modality content
                                combined_content = user_prompt_content + consolidated_content
                                combined_content = _append_kimi_run_code_guidance_if_needed(
                                    combined_content, model_name
                                )

                                # Convert combined content to web UI format preserving interleaved structure
                                interleaved_blocks = []
                                total_text_chars = 0
                                total_images = 0

                                for block in combined_content:
                                    if isinstance(block, dict):
                                        if block.get("type") == "text":
                                            text_content = block.get(
                                                "text", "")
                                            if text_content.strip():  # Only add non-empty text blocks
                                                interleaved_blocks.append({
                                                    "type": "text",
                                                    "content": text_content
                                                })
                                                total_text_chars += len(
                                                    text_content)
                                        elif block.get("type") == "image":
                                            # Extract image for Web UI display
                                            source = block.get("source", {})
                                            if source.get("data"):
                                                img_data_url = f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
                                                interleaved_blocks.append({
                                                    "type": "image",
                                                    "content": img_data_url
                                                })
                                                total_images += 1
                                            else:
                                                emit(
                                                    EventType.SYSTEM, f"   ⚠️ Image block missing data: {block}")

                                if interleaved_blocks:
                                    # Emit with interleaved blocks structure for proper text/image interleaving
                                    emit(
                                        EventType.REFLECTION,
                                        "",
                                        _iteration_context_metadata(
                                            iteration=iteration,
                                            ui_title=f"📋 **Iteration {iteration} — Consolidated Context**",
                                            context_mode="handoff",
                                            prompt_blocks=combined_content,
                                            resume_messages=[{"role": "user", "content": combined_content}],
                                        ),
                                    )
                                    emit(
                                        EventType.SYSTEM, f"   📋 Consolidated prompt: {total_text_chars:,} chars, {total_images} images, {len(interleaved_blocks)} blocks")
                                messages = [
                                    {"role": "user", "content": combined_content}
                                ]
                            else:
                                # Fallback: append simple continuation message
                                messages.append({
                                    "role": "user",
                                    "content": [{
                                        "type": "text",
                                        "text": (
                                            "Understood. Please proceed with your refined approach. "
                                            "You may use run_code or run_code_in_previous_runtime for experiments, or execute_python_solution to test your updated hypothesis."
                                        )
                                    }]
                                })

                            current_iteration_start_idx = len(messages)
                            continue
                    else:
                        in_reflection_mode = False

                        current_context_tokens = last_actual_input_tokens + last_actual_output_tokens
                        _bypass_compression = (
                            current_context_tokens == 0
                            or current_context_tokens < compression_bypass_threshold
                        )

                        if _bypass_compression:
                            emit(EventType.SYSTEM,
                                 f"   ℹ️ Context below bypass threshold ({current_context_tokens:,} < {compression_bypass_threshold:,}) — skipping Phase 2 compression")
                            iteration += 1
                            emit(EventType.SYSTEM, f"   📊 Starting Iteration {iteration}/{max_iterations}")
                            messages.append({"role": "user", "content": [{"type": "text", "text":
                                "Thank you for your reflection. Now please proceed with your next steps to refine the solve() function. "
                                "You may use run_code or run_code_in_previous_runtime for experiments, "
                                "or execute_python_solution to test your updated hypothesis."
                            }]})
                            emit(
                                EventType.REFLECTION,
                                "",
                                _iteration_context_metadata(
                                    iteration=iteration,
                                    ui_title=f"📋 **Iteration {iteration} — Full Context (bypass)**",
                                    context_mode="snapshot",
                                    snapshot_messages=messages,
                                    resume_messages=messages,
                                ),
                            )
                            continue

                        emit(
                            EventType.SYSTEM, "   ✅ Reflection complete. Running Phase 2 context compression...")

                        # ── Context Compression (train-failure path) ────────────────────
                        # Phase 1 produced a detailed reflection embedded in a long conversation.
                        # Now compress it into a portable summary for the next iteration.
                        _invalidate_exploratory_runtime()
                        phase2_prompt_train = CONTEXT_COMPRESSION_PROMPT

                        phase2_messages_train = messages + [
                            {"role": "user", "content": [
                                {"type": "text", "text": phase2_prompt_train}]}
                        ]

                        emit(EventType.REFLECTION, phase2_prompt_train,
                             {"iteration": iteration})

                        phase2_summary_train = ""
                        phase2_thinking_train = ""
                        phase2_thinking_signature_train = ""
                        try:
                            with phoenix.span(f"turn_{turn}_phase2_train", {"turn": turn, "iteration": iteration}, force_flush=True):
                                phase2_messages_train = _sanitize_messages(
                                    phase2_messages_train,
                                    image_source_mode=image_source_mode,
                                    force_png_reencode=_use_fireworks,
                                )
                                _p2t_model_lc = str(model_name or "").lower()
                                if "haiku-4-5" in _p2t_model_lc:
                                    _p2t_max_tokens = 64000
                                elif "kimi" in _p2t_model_lc:
                                    _p2t_max_tokens = 128000
                                else:
                                    _p2t_max_tokens = 128000
                                phase2_params_train = {
                                    "model": model_name,
                                    "max_tokens": _p2t_max_tokens,
                                    "system": system_prompt,
                                    "messages": phase2_messages_train,
                                }
                                if _use_fireworks:
                                    phase2_params_train["timeout"] = _fireworks_timeout()
                                if use_extended_thinking:
                                    model_lc = str(model_name or "").lower()
                                    if "haiku-4-5" in model_lc or "kimi" in model_lc:
                                        # Haiku 4.5 / Fireworks: use budget_tokens
                                        if "kimi" in model_lc:
                                            _budget_cap = min(
                                                _p2t_max_tokens - 1, KIMI_COMPRESSION_THINKING_BUDGET_CAP
                                            )
                                        else:
                                            _budget_cap = _p2t_max_tokens - 1
                                        budget_tokens = max(
                                            1024, min(int(thinking_budget or 1024), _budget_cap))
                                        phase2_params_train["thinking"] = {
                                            "type": "enabled", "budget_tokens": budget_tokens}
                                    elif "glm" in model_lc:
                                        # GLM path: use explicit thinking budget only.
                                        budget_tokens = max(
                                            1024, min(int(thinking_budget or 1024), _p2t_max_tokens - 1))
                                        phase2_params_train["thinking"] = {
                                            "type": "enabled", "budget_tokens": budget_tokens}
                                    else:
                                        # Opus/Sonnet: use categorical approach with reflection effort
                                        allowed_efforts = [
                                            "low", "medium", "high", "max"]
                                        normalized_effort = str(
                                            reflection_thinking_effort or "medium").strip().lower()
                                        if normalized_effort not in allowed_efforts:
                                            normalized_effort = "medium"
                                        phase2_params_train["thinking"] = {
                                            "type": "adaptive"}
                                        phase2_params_train["output_config"] = {
                                            "effort": normalized_effort}
                                    phase2_params_train["temperature"] = 1

                                phase2_params_train = _apply_anthropic_prompt_caching(
                                    phase2_params_train,
                                    use_fireworks=_use_fireworks,
                                    include_tools=False,
                                )

                                # Log raw request to Phoenix
                                try:
                                    phase2_raw_request_train = json.dumps(
                                        phase2_params_train, default=str)
                                except Exception:
                                    phase2_raw_request_train = str(
                                        phase2_params_train)

                                invocation_params_p2_train = {
                                    k: v for k, v in phase2_params_train.items() if k not in ("messages", "system")}
                                try:
                                    inv_str_p2_train = json.dumps(
                                        invocation_params_p2_train, default=str)
                                except Exception:
                                    inv_str_p2_train = str(
                                        invocation_params_p2_train)
                                if len(inv_str_p2_train) > 200000:
                                    inv_str_p2_train = inv_str_p2_train[:200000]

                                with phoenix.span(
                                    "MessagesStream",
                                    {
                                        "openinference.span.kind": "LLM",
                                        "llm.system": "anthropic",
                                        "llm.model_name": model_name,
                                        "turn": f"{turn}_phase2_train",
                                        "phase": "context_compression_train",
                                    },
                                    force_flush=True,
                                ):
                                    phoenix.set_attribute(
                                        "llm.invocation_parameters", inv_str_p2_train)
                                    phoenix.set_attribute(
                                        "input.mime_type", "application/json")
                                    phoenix.set_attribute(
                                        "input.value", phase2_raw_request_train)
                                    _oi_set_input_messages(
                                        phoenix, system_prompt, phase2_messages_train)

                                    _phase2_train_stopped_mid_stream = False
                                    _phase2_train_attempt = 0
                                    _phase2_train_max_retries = _max_retries_for_model(_use_fireworks)
                                    while True:
                                        _phase2_train_attempt += 1
                                        try:
                                            with _suppress_auto_instrumentation(), client.messages.stream(**phase2_params_train) as stream:
                                                for event in stream:
                                                    if should_stop():
                                                        _phase2_train_stopped_mid_stream = True
                                                        break

                                                    if event.type == "content_block_delta":
                                                        if hasattr(event, 'delta'):
                                                            delta = event.delta
                                                            if delta.type == "text_delta":
                                                                phase2_summary_train += delta.text
                                                                emit(
                                                                    EventType.TEXT, delta.text)
                                                            elif delta.type == "thinking_delta":
                                                                phase2_thinking_train += delta.thinking
                                                                emit(
                                                                    EventType.THINKING, delta.thinking)
                                                            elif delta.type == "signature_delta":
                                                                phase2_thinking_signature_train = delta.signature
                                                                emit(EventType.SYSTEM, "", {
                                                                     "_thinking_signature": phase2_thinking_signature_train})
                                            break
                                        except Exception as _phase2t_err:
                                            if should_stop():
                                                _phase2_train_stopped_mid_stream = True
                                                break
                                            if _is_transient_api_error(_phase2t_err) and _phase2_train_attempt <= _phase2_train_max_retries:
                                                _backoff = min(12, 2 ** _phase2_train_attempt)
                                                emit(EventType.SYSTEM, f"   ⚠️ Phase2-train stream retry {_phase2_train_attempt}/{_phase2_train_max_retries}: {_phase2t_err}")
                                                for _ in range(int(_backoff * 10)):
                                                    if should_stop():
                                                        _phase2_train_stopped_mid_stream = True
                                                        break
                                                    time.sleep(0.1)
                                                if _phase2_train_stopped_mid_stream:
                                                    break
                                                continue
                                            raise

                                        try:
                                            phase2_final_message_train = None
                                            if not _phase2_train_stopped_mid_stream:
                                                phase2_final_message_train = stream.get_final_message()
                                            if phase2_final_message_train:
                                                phoenix.set_large_attribute(
                                                    "raw_final_message", phase2_final_message_train.model_dump())
                                                # If streaming produced no text (model only output thinking),
                                                # extract text from the final message content blocks and emit.
                                                if not phase2_summary_train and hasattr(phase2_final_message_train, 'content'):
                                                    for blk in (phase2_final_message_train.content or []):
                                                        if hasattr(blk, 'text') and blk.text:
                                                            phase2_summary_train += blk.text
                                                    if phase2_summary_train:
                                                        emit(
                                                            EventType.TEXT, phase2_summary_train)
                                                        emit(
                                                            EventType.SYSTEM, f"   ℹ️ Phase 2 train: Extracted text from final message (was missing from stream)")
                                                # Log usage to Phoenix and emit to Web UI
                                                if hasattr(phase2_final_message_train, 'usage') and phase2_final_message_train.usage:
                                                    _oi_set_output_messages(
                                                        phoenix, phase2_summary_train, tool_uses=[])
                                                    usage_meta = _extract_usage_meta(phase2_final_message_train.usage)
                                                    emit(EventType.SYSTEM, "", {
                                                         "_token_usage": usage_meta})
                                        except Exception:
                                            pass

                                if phase2_summary_train:
                                    emit(
                                        EventType.SYSTEM, f"   ✅ Phase 2 context summary received ({len(phase2_summary_train):,} chars)")
                                    phoenix.set_large_attribute(
                                        "phase2_summary", phase2_summary_train)
                        except Exception as e:
                            emit(
                                EventType.SYSTEM, f"   ⚠️ Phase 2 compression failed: {e} — falling back to Phase 1 response")

                        # Increment iteration AFTER both phases complete
                        iteration += 1
                        emit(
                            EventType.SYSTEM, f"   📊 Starting Iteration {iteration}/{max_iterations}")

                        # Update reflection from current iteration (before messages get reset)
                        # Use Phase 2 summary if available, else extract from current messages
                        current_reflection = phase2_summary_train if phase2_summary_train else _extract_reflection_summary(
                            messages)
                        if current_reflection:
                            last_reflection = current_reflection
                            emit(
                                EventType.SYSTEM, f"   📝 Updated reflection from current iteration ({len(current_reflection)} chars)")

                        # Build consolidated summary for fresh context
                        consolidated_content = _build_consolidated_summary(
                            iteration_history=iteration_history,
                            last_reflection=last_reflection,
                            last_failed_results=last_failed_results,
                            last_visual_parts=last_visual_parts,
                            verified_rules=verified_rules,
                            use_visual_mode=use_visual_mode
                        )

                        if consolidated_content:
                            # Reset messages: single consolidated user prompt (no fake assistant message)
                            emit(
                                EventType.SYSTEM, "   📋 Resetting context with consolidated iteration summary...")
                            # Emit FULL consolidated prompt to UI (matching what Phoenix sees)
                            # This includes puzzle grids + iteration summary with dual modality content
                            combined_content = user_prompt_content + consolidated_content
                            combined_content = _append_kimi_run_code_guidance_if_needed(
                                combined_content, model_name
                            )

                            # Convert combined content to web UI format preserving interleaved structure
                            interleaved_blocks = []
                            total_text_chars = 0
                            total_images = 0

                            for block in combined_content:
                                if isinstance(block, dict):
                                    if block.get("type") == "text":
                                        text_content = block.get("text", "")
                                        if text_content.strip():  # Only add non-empty text blocks
                                            interleaved_blocks.append({
                                                "type": "text",
                                                "content": text_content
                                            })
                                            total_text_chars += len(text_content)
                                    elif block.get("type") == "image":
                                        # Extract image for Web UI display
                                        source = block.get("source", {})
                                        if source.get("data"):
                                            img_data_url = f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
                                            interleaved_blocks.append({
                                                "type": "image",
                                                "content": img_data_url
                                            })
                                            total_images += 1
                                        else:
                                            emit(
                                                EventType.SYSTEM, f"   ⚠️ Image block missing data: {block}")

                            if interleaved_blocks:
                                # Emit with interleaved blocks structure for proper text/image interleaving
                                emit(
                                    EventType.REFLECTION,
                                    "",
                                    _iteration_context_metadata(
                                        iteration=iteration,
                                        ui_title=f"📋 **Iteration {iteration} — Consolidated Context**",
                                        context_mode="handoff",
                                        prompt_blocks=combined_content,
                                        resume_messages=[{"role": "user", "content": combined_content}],
                                    ),
                                )
                                emit(
                                    EventType.SYSTEM, f"   📋 Consolidated prompt: {total_text_chars:,} chars, {total_images} images, {len(interleaved_blocks)} blocks")
                            messages = [
                                {"role": "user", "content": combined_content}
                            ]
                        else:
                            # Fallback: append simple continuation message
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "text",
                                    "text": (
                                        "Thank you for your reflection. Now please proceed with your next steps to refine the solve() function. "
                                        "You may use run_code or run_code_in_previous_runtime for experiments, or execute_python_solution to test your updated hypothesis."
                                    )
                                }]
                            })

                        current_iteration_start_idx = len(messages)
                        continue
                else:
                    emit(EventType.SYSTEM, "   Ending turn.")
                    break

            # Log tool calls to Phoenix
            if function_calls and pending_guided_followup:
                pending_guided_followup = False
            for fc in function_calls:
                with phoenix.span(f"🔧 Tool Call: {fc['name']}", {"message_type": "tool_call", "turn": turn}):
                    phoenix.set_attribute("tool_name", fc["name"])
                    phoenix.set_attribute("tool_id", fc["id"])
                    input_str = json.dumps(fc["input"], default=str)[:10000]
                    phoenix.set_attribute("tool_input", input_str)

            # Execute tools
            emit(EventType.SYSTEM,
                 f"⚙️ Executing {len(function_calls)} tool call(s)...")

            tool_results = []
            trigger_test_generalization = False
            local_ctx = None
            solve_fn = None
            results = None
            visual_parts = []
            fname = None
            active_solution_branches: list[dict[str, Any]] = []
            last_execute_hypothesis_submission = _empty_hypothesis_submission()
            last_execute_code_submission: dict[str, str] = {}

            for call in function_calls:
                fname = call["name"]
                args = call["input"]
                call_id = call["id"]
                emit(EventType.SYSTEM, f"   Running {fname}...")

                # Don't increment iteration yet - it will be incremented after reflection (if needed)
                # This ensures reflection belongs to the same iteration as execute_python_solution

                if fname in EXPLORATORY_CODE_TOOL_NAMES:
                    if not unsafe_local_exec:
                        blocked_msg = (
                            "Local Python execution is disabled.\n"
                            "Enable unsafe local execution via --unsafe-local-exec (or runtime.unsafe_local_exec=true) "
                            f"to allow {fname}."
                        )
                        emit(
                            EventType.TOOL_RESULT,
                            f"{fname} result",
                            {
                                "id": call_id,
                                "tool_use_id": call_id,
                                "tool_name": fname,
                                "output": blocked_msg,
                                "is_error": True,
                                "result": {"blocked": True, "reason": "unsafe_local_exec_disabled"},
                            },
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": blocked_msg,
                            "is_error": True,
                        })
                        continue

                    code = args.get('code', '')
                    timeout = _clamp_timeout(
                        args.get('timeout_seconds'), CODE_EXEC_DEFAULT_TIMEOUT)
                    runtime_status = "fresh"
                    runtime_ctx: dict[str, Any] | None
                    runtime_id: int | None
                    runtime_step: int | None
                    if fname == "run_code":
                        runtime_ctx, runtime_id, runtime_step = _begin_fresh_exploratory_runtime()
                    else:
                        runtime_status = "reused"
                        runtime_ctx, runtime_id, runtime_step = _reuse_exploratory_runtime()
                        if runtime_ctx is None:
                            runtime_status = "fresh_fallback"
                            runtime_ctx, runtime_id, runtime_step = _begin_fresh_exploratory_runtime()

                    stdout, stderr = execute_code_safe(code, runtime_ctx, timeout_seconds=timeout)
                    if runtime_status == "fresh_fallback":
                        runtime_prefix = (
                            f"Runtime: fresh fallback (id=R{runtime_id}, step={runtime_step})\n"
                            "Requested reuse did not happen because no live previous exploratory runtime was available.\n"
                        )
                    else:
                        runtime_prefix = (
                            f"Runtime: {runtime_status} "
                            f"(id=R{runtime_id}, step={runtime_step})\n"
                        )
                    result_text = runtime_prefix + f"STDOUT:\n{stdout}\n"
                    if stderr:
                        result_text += f"STDERR:\n{stderr}"

                    # Truncate large outputs to prevent context overflow
                    result_text_truncated = _truncate_tool_output(
                        result_text, max_chars=5000)

                    with phoenix.span(f"🔧 Tool Result: {fname}", {"turn": turn, "tool_use_id": call_id}, force_flush=True):
                        phoenix.set_attribute("tool_name", fname)
                        phoenix.set_attribute("is_error", bool(stderr))
                        phoenix.set_attribute(
                            "output_length", len(result_text))
                        phoenix.set_attribute(
                            "output_truncated", len(result_text) > 5000)
                        phoenix.set_attribute("runtime_status", runtime_status)
                        phoenix.set_attribute("runtime_id", f"R{runtime_id}")
                        phoenix.set_attribute("runtime_step", runtime_step)
                        # Store full output in Phoenix
                        phoenix.set_large_attribute("output", result_text)

                    emit(EventType.TOOL_RESULT, f"{fname} result",
                             {
                                 "id": call_id,
                                 "tool_use_id": call_id,
                                 "output": result_text_truncated,
                                 "tool_name": fname,
                                 "runtime_status": runtime_status,
                                 "fallback_from_previous_runtime": runtime_status == "fresh_fallback",
                                 "runtime_id": f"R{runtime_id}",
                                 "runtime_step": runtime_step,
                             })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": result_text_truncated  # Send truncated version to model
                    })

                elif fname == "submit_transform_hypothesis":
                    hypothesis_submission = _normalize_hypothesis_submission(args)
                    hypothesis_error = _validate_hypothesis_submission(hypothesis_submission)
                    if hypothesis_error:
                        emit(
                            EventType.TOOL_RESULT,
                            "submit_transform_hypothesis result",
                            {
                                "tool_name": "submit_transform_hypothesis",
                                "output": hypothesis_error,
                                "is_error": True,
                            },
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": hypothesis_error,
                            "is_error": True,
                        })
                        continue

                    last_hypothesis_submission = _copy_active_hypothesis_submission(hypothesis_submission)
                    last_transform_hypothesis = _branch_a_hypothesis_text(last_hypothesis_submission)
                    branch_repair_state = None

                    recorded_lines = [f"Hypothesis recorded ({len(last_transform_hypothesis)} chars)."]
                    emit(EventType.SYSTEM, f"   📝 {recorded_lines[0]}")
                    recorded_lines.append(
                        "You may now call execute_python_solution with your code."
                    )
                    hypothesis_recorded_msg = " ".join(recorded_lines)
                    emit(
                        EventType.TOOL_RESULT,
                        "submit_transform_hypothesis result",
                        {
                            "tool_name": "submit_transform_hypothesis",
                            "output": hypothesis_recorded_msg,
                        },
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": hypothesis_recorded_msg,
                    })
                    pending_execute_after_hypothesis_submission = True

                    with phoenix.span("📝 Transform Hypothesis", {"turn": turn, "tool_use_id": call_id}, force_flush=True):
                        phoenix.set_attribute("tool_name", "submit_transform_hypothesis")
                        phoenix.set_attribute("mode", "normal")
                        phoenix.set_large_attribute("hypothesis", last_hypothesis_submission["hypothesis"])

                elif fname == "execute_python_solution":
                    pending_execute_after_hypothesis_submission = False
                    if not unsafe_local_exec:
                        blocked_msg = (
                            "Local Python execution is disabled.\n"
                            "Enable unsafe local execution via --unsafe-local-exec (or runtime.unsafe_local_exec=true) "
                            "to allow execute_python_solution."
                        )
                        emit(
                            EventType.TOOL_RESULT,
                            "execute_python_solution result",
                            {
                                "id": call_id,
                                "tool_use_id": call_id,
                                "tool_name": "execute_python_solution",
                                "output": blocked_msg,
                                "result": {"correct": False, "blocked": True, "reason": "unsafe_local_exec_disabled"},
                            },
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": blocked_msg,
                            "is_error": True,
                        })
                        # This is a runtime policy block, not a model iteration attempt.
                        iteration -= 1
                        continue

                    code_submission = _normalize_code_submission(args)
                    effective_hypothesis_submission = _derive_execute_hypothesis_submission(
                        last_hypothesis_submission,
                    )
                    last_execute_hypothesis_submission = _copy_hypothesis_submission(
                        effective_hypothesis_submission
                    )
                    last_execute_code_submission = dict(code_submission)
                    transform_hypothesis = _branch_a_hypothesis_text(effective_hypothesis_submission)
                    last_reviewer_response = str(effective_hypothesis_submission.get("reviewer_response") or "").strip()
                    code = str(code_submission.get("code") or "")

                    if transform_hypothesis:
                        emit(EventType.SYSTEM, f"   📝 Hypothesis: {transform_hypothesis}")

                    invalid_submission_msg = None
                    if not code or not code.strip() or "def solve" not in code:
                        invalid_submission_msg = (
                            "Error: Missing or invalid code argument.\n\n"
                            "Your tool call must include a valid code field with a complete Python solve(grid) function.\n"
                            "Correct format example:\n"
                            "```json\n"
                            "{\n"
                            '  "code": "def solve(grid):\\n    return grid"\n'
                            "}\n"
                            "```"
                        )

                    if invalid_submission_msg:
                        emit(EventType.SYSTEM, "   -> Invalid execute_python_solution payload")
                        emit(
                            EventType.TOOL_RESULT,
                            "execute_python_solution result",
                            {
                                "id": call_id,
                                "tool_use_id": call_id,
                                "tool_name": "execute_python_solution",
                                "output": invalid_submission_msg,
                                "result": {"correct": False, "error": invalid_submission_msg},
                            },
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": invalid_submission_msg,
                            "is_error": True,
                        })
                        iteration -= 1
                        continue

                    solution_timeout = _clamp_timeout(
                        args.get('timeout_seconds'), CODE_EXEC_SOLUTION_DEFAULT_TIMEOUT)
                    branch_eval, eval_timed_out = _run_with_timeout(
                        lambda: evaluate_solution_branches(
                            code_submission,
                            effective_hypothesis_submission,
                        ),
                        solution_timeout,
                    )
                    if eval_timed_out:
                        timeout_msg = f"Code execution timed out after {solution_timeout:.0f} seconds. Your solve() function likely has an infinite loop or is too slow. Fix the bug and retry."
                        emit(EventType.SYSTEM, f"   -> Timed out after {solution_timeout:.0f}s")
                        emit(
                            EventType.TOOL_RESULT,
                            "execute_python_solution result",
                            {
                                "id": call_id,
                                "tool_use_id": call_id,
                                "tool_name": "execute_python_solution",
                                "output": timeout_msg,
                                "result": {"correct": False, "error": timeout_msg},
                            },
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": timeout_msg,
                            "is_error": True,
                        })
                        continue
                    results = branch_eval["results"]
                    branch_summary_line = branch_eval["branch_summary_line"]
                    active_solution_branches = branch_eval["active_solution_branches"]
                    selected_solution_branch = branch_eval["selected_solution_branch"]
                    if selected_solution_branch:
                        solve_fn = selected_solution_branch.get("solve_fn")
                        local_ctx = selected_solution_branch.get("local_ctx")
                    else:
                        solve_fn = None
                        local_ctx = None

                    last_transform_hypothesis = transform_hypothesis or last_transform_hypothesis

                    tool_result_content: list[dict[str, Any]] = []
                    report_lines = [branch_summary_line]
                    selected_visual_parts: list[dict[str, Any]] = []

                    if selected_solution_branch and selected_solution_branch.get("status") == "error":
                        error_text = str(selected_solution_branch.get("error") or "Unknown execution error")
                        tool_result_content.append({"type": "text", "text": f"Execution Error: {error_text}"})
                        report_lines.append(f"Execution Error: {error_text}")

                    for res in results.get("train", []) or []:
                        pixel_acc_str = f"pixel acc: {res.get('pixel_accuracy', 0):.1%}"
                        if res.get("error"):
                            tool_result_content.append(
                                {"type": "text", "text": f"Train {res['index']} Error: {res['error']}"}
                            )
                            report_lines.append(f"Train {res['index']} Error: {res['error']}")
                            continue
                        if res.get("correct", False):
                            pass_text = f"Train {res['index']} PASS"
                            tool_result_content.append({"type": "text", "text": pass_text})
                            report_lines.append(pass_text)
                            continue

                        tool_result_content.append({"type": "text", "text": f"Train {res['index']} {pixel_acc_str}"})
                        tool_result_content.append({"type": "text", "text": f"Expected Output: {res['expected']}"})
                        selected_visual_parts.append({"type": "text", "text": f"\nTrain {res['index']} {pixel_acc_str}"})
                        selected_visual_parts.append({"type": "text", "text": f"Expected Output: {res['expected']}"})
                        if use_visual_mode and (img_exp := render_grid_to_base64(res["expected"])):
                            image_block = {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": img_exp},
                            }
                            tool_result_content.append(image_block)
                            selected_visual_parts.append(image_block)
                        report_lines.append(f"Train {res['index']} {pixel_acc_str}")

                        for candidate_idx, candidate in enumerate(res.get("candidates", []) or [], start=1):
                            candidate_text = f"{_ordinal(candidate_idx)} candidate: {candidate}"
                            tool_result_content.append({"type": "text", "text": candidate_text})
                            selected_visual_parts.append({"type": "text", "text": candidate_text})
                            if use_visual_mode and (img_pred := render_grid_to_base64(candidate)):
                                image_block = {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/png", "data": img_pred},
                                }
                                tool_result_content.append(image_block)
                                selected_visual_parts.append(image_block)

                    for res in results.get("test", []) or []:
                        if res.get("error"):
                            tool_result_content.append(
                                {"type": "text", "text": f"Test {res['index']} Error: {res['error']}"}
                            )
                            report_lines.append(f"Test {res['index']} Error: {res['error']}")
                            continue

                        tool_result_content.append({"type": "text", "text": f"**Test Example {res['index']} Candidate Outputs**"})
                        report_lines.append(f"Test {res['index']} Generated {len(res.get('candidates', []) or [])} candidate output(s)")
                        for candidate_idx, candidate in enumerate(res.get("candidates", []) or [], start=1):
                            tool_result_content.append({"type": "text", "text": f"{_ordinal(candidate_idx)} candidate: {candidate}"})
                            if use_visual_mode and (img_pred := render_grid_to_base64(candidate)):
                                tool_result_content.append(
                                    {
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": "image/png", "data": img_pred},
                                    }
                                )

                    emit(
                        EventType.SYSTEM,
                        f"   -> Results: Correct={results['correct']}, Pixel Acc={results.get('train_pixel_accuracy', 0.0):.1%}",
                    )

                    branch_repair_state = None

                    train_summary = []
                    for res in results.get("train", []) or []:
                        train_summary.append(
                            {
                                "index": res["index"],
                                "correct": res.get("correct", False),
                                "pixel_accuracy": res.get("pixel_accuracy", 0.0),
                                "error": res.get("error"),
                            }
                        )

                    iteration_history.append({
                        "iteration": iteration,
                        "transform_hypothesis": transform_hypothesis,
                        "passed": bool(results.get("correct", False)),
                        "train_results": train_summary,
                        "pixel_accuracy": results.get("train_pixel_accuracy", 0.0),
                        "code": code,
                    })

                    if not results.get("correct", False):
                        last_failed_results = results
                        last_visual_parts = selected_visual_parts

                    num_images = sum(1 for p in tool_result_content if p.get("type") == "image")
                    if num_images > 0:
                        emit(EventType.SYSTEM, f"   -> Adding {num_images} visual feedback images (interleaved)")
                    tool_result_images = [
                        p.get("source", {}).get("data", "")
                        for p in tool_result_content
                        if isinstance(p, dict)
                        and p.get("type") == "image"
                        and isinstance(p.get("source"), dict)
                        and p.get("source", {}).get("data")
                    ]

                    emit(
                        EventType.TOOL_RESULT,
                        "execute_python_solution result",
                        {
                            "id": call_id,
                            "tool_use_id": call_id,
                            "tool_name": "execute_python_solution",
                            "output": "\n".join(report_lines),
                            "result": results,
                            "content_blocks": tool_result_content,
                        },
                        images=tool_result_images,
                    )

                    with phoenix.span("🔧 Tool Result: execute_python_solution", {"turn": turn, "tool_use_id": call_id}, force_flush=True):
                        phoenix.set_attribute("tool_name", "execute_python_solution")
                        phoenix.set_attribute("train_pass", results.get("correct", False))
                        phoenix.set_attribute("train_pixel_accuracy", results.get("train_pixel_accuracy", 0.0))
                        phoenix.set_attribute("num_test_predictions", results.get("num_test_predictions", 0))
                        phoenix.set_large_attribute("report", "\n".join(report_lines))

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": tool_result_content,
                    })

                    trigger_test_generalization = bool(results.get("correct", False))

                    # Case 2: best-effort mode — force submission path even with partial training
                    if _best_effort_injected and not trigger_test_generalization:
                        trigger_test_generalization = True
                        selected_solution_branch = branch_eval["branch_outcomes"][0]
                        active_solution_branches = [selected_solution_branch]
                        final_code = str(selected_solution_branch.get("code") or "")

                    if trigger_test_generalization and selected_solution_branch:
                        # Snapshot branches every time training hits 100% (Case 1 auto-submit)
                        if results.get("correct", False):
                            last_fully_passing_branches = list(active_solution_branches)
                        final_code = str(selected_solution_branch.get("code") or "")

            # Append tool results as user message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Inject test generalization reflection if train accuracy = 100%
            if trigger_test_generalization and selected_solution_branch:
                emit(EventType.SYSTEM,
                     f"\n✅ Train accuracy: 100% - Generating test prediction candidate set...")
                test_visual_parts = []
                test_prediction_only_parts = []
                test_input_grids = [example["input"] for example in puzzle_data.get("test", [])]
                _reflector_test_input_images = []
                if use_visual_mode:
                    for grid in test_input_grids:
                        _reflector_test_input_images.append(render_grid_to_base64(grid) or "")

                _reflector_test_inputs = list(test_input_grids)
                _reflector_test_preds = []
                _reflector_candidate_payloads = []

                solution_result = selected_solution_branch.get("result", {}) if isinstance(selected_solution_branch.get("result"), dict) else {}

                # Build per-candidate training accuracy breakdown for the reflector
                _train_match_lines = []
                for _tres in solution_result.get("train", []) or []:
                    if not isinstance(_tres, dict):
                        continue
                    _tidx = _tres.get("index", "?")
                    _mci = _tres.get("matched_candidate_index")
                    if _mci:
                        _train_match_lines.append(f"Train {_tidx}: matched by {_ordinal(_mci)} candidate")
                    elif _tres.get("correct", False):
                        _train_match_lines.append(f"Train {_tidx}: PASS")
                    else:
                        _train_match_lines.append(f"Train {_tidx}: FAIL")
                _reflector_training_accuracy = "100% on training set"
                if _train_match_lines:
                    _reflector_training_accuracy += "\n" + "\n".join(_train_match_lines)
                for res in solution_result.get("test", []) or []:
                    if not isinstance(res, dict):
                        continue
                    test_index = int(res.get("index", -1))
                    if test_index < 0 or test_index >= len(test_input_grids):
                        continue

                    candidates = list(res.get("candidates", []) or [])
                    candidate_images: list[str] = []

                    test_visual_parts.append({"type": "text", "text": f"\n**Test Example {test_index}**"})
                    test_prediction_only_parts.append({"type": "text", "text": f"\n**Test Example {test_index}**"})
                    test_visual_parts.append({"type": "text", "text": f"Input: {test_input_grids[test_index]}"})
                    if use_visual_mode and test_index < len(_reflector_test_input_images) and _reflector_test_input_images[test_index]:
                        test_visual_parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _reflector_test_input_images[test_index],
                                },
                            }
                        )

                    for candidate_idx, candidate in enumerate(candidates, start=1):
                        if candidate_idx == 1:
                            _reflector_test_preds.append(candidate)
                        label = f"{_ordinal(candidate_idx)} candidate: {candidate}"
                        test_visual_parts.append({"type": "text", "text": label})
                        test_prediction_only_parts.append({"type": "text", "text": label})
                        if use_visual_mode:
                            img_pred = render_grid_to_base64(candidate) or ""
                            candidate_images.append(img_pred)
                            if img_pred:
                                image_block = {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": img_pred,
                                    },
                                }
                                test_visual_parts.append(image_block)
                                test_prediction_only_parts.append(image_block)
                        else:
                            candidate_images.append("")

                    _reflector_candidate_payloads.append(
                        {
                            "index": test_index,
                            "candidates": candidates,
                            "candidate_images": candidate_images,
                            "error": res.get("error"),
                        }
                    )

                # ── Independent Reflector ──────────────────────────────
                _reflector_verdict = None
                _reflector_gate_debug = {
                    "enable": enable_independent_reflector,
                    "hypothesis_len": len(last_transform_hypothesis) if last_transform_hypothesis else 0,
                    "hypothesis_truthy": bool(last_transform_hypothesis),
                    "best_effort": _best_effort_injected,
                }
                emit(EventType.SYSTEM,
                     f"   🔍 Reflector gate: {_reflector_gate_debug}")
                if enable_independent_reflector and last_transform_hypothesis and not _best_effort_injected:
                    # New canvas: reflector has its own context window (separate system prompt,
                    # no solver chat history). Bump iteration so the UI shows a new canvas.
                    iteration += 1
                    max_iterations += 2  # compensate for 2 extra iteration bumps (reflector + resume)
                    emit(EventType.TURN_START, "", {
                         "turn": turn, "iteration": iteration, "max_iterations": max_iterations,
                         "context_label": "Independent Reflector"})
                    emit(EventType.SYSTEM,
                         f"\n🔍 Running independent reflector ({reflector_provider}: {reflector_model or 'default'})...")
                    try:
                        from .independent_reflector import run_independent_reflection
                    except ImportError:
                        from independent_reflector import run_independent_reflection

                    # Collect training data for Phase 1 validation
                    _reflector_train_inputs = [ex["input"] for ex in puzzle_data.get("train", [])]
                    _reflector_train_outputs = [ex["output"] for ex in puzzle_data.get("train", [])]
                    _reflector_train_input_images = []
                    _reflector_train_output_images = []
                    if use_visual_mode:
                        for ex in puzzle_data.get("train", []):
                            img_in = render_grid_to_base64(ex["input"])
                            _reflector_train_input_images.append(img_in or "")
                            img_out = render_grid_to_base64(ex["output"])
                            _reflector_train_output_images.append(img_out or "")

                    # ── Create new canvas for independent reflector ────────
                    # The reflector has a completely fresh context (different system prompt,
                    # no solver chat history). Emit a canvas boundary to visually separate it.
                    _refl_model_label = reflector_model or ("claude-opus-4-6" if reflector_provider == "claude" else "gemini-3.1-pro-preview")
                    
                    # Load reflector system prompt and build user message
                    try:
                        from .independent_reflector import _load_reflector_prompt, _build_reflector_user_message, _build_reflector_followup_message
                    except ImportError:
                        from independent_reflector import _load_reflector_prompt, _build_reflector_user_message, _build_reflector_followup_message

                    _reflector_system_prompt = _load_reflector_prompt()
                    _reflector_is_followup = bool(_reflector_message_history)
                    if _reflector_is_followup:
                        _reflector_user_msg = _build_reflector_followup_message(
                            transform_hypothesis=last_transform_hypothesis,
                            code=str(selected_solution_branch.get("code") or ""),
                            test_predictions=_reflector_test_preds,
                            training_accuracy=_reflector_training_accuracy,
                            ambiguity_rationale="",
                            candidate_predictions=_reflector_candidate_payloads,
                        )
                    else:
                        _reflector_user_msg = _build_reflector_user_message(
                            transform_hypothesis=last_transform_hypothesis,
                            code=str(selected_solution_branch.get("code") or ""),
                            train_inputs=_reflector_train_inputs,
                            train_outputs=_reflector_train_outputs,
                            test_inputs=_reflector_test_inputs,
                            test_predictions=_reflector_test_preds,
                            training_accuracy=_reflector_training_accuracy,
                            ambiguity_rationale="",
                            candidate_predictions=_reflector_candidate_payloads,
                            test_input_images=_reflector_test_input_images if use_visual_mode else None,
                            test_prediction_images=[],
                        )

                    _reflector_emit_meta = {
                        "turn": turn,
                        "iteration": iteration,
                        "provider": reflector_provider,
                        "model": _refl_model_label,
                        "system_md": _reflector_system_prompt,
                        "user_md": _reflector_user_msg,
                        "hypothesis": last_transform_hypothesis,
                        "code": str(selected_solution_branch.get("code") or ""),
                        "review_candidates": _reflector_candidate_payloads,
                        "phase": "start",
                        "reflector_turn": len(_reflector_message_history) // 2 + 1,
                        "prior_turns": list(_reflector_message_history) if _reflector_message_history else [],
                    }
                    if not _reflector_is_followup:
                        _reflector_emit_meta.update({
                            "train_inputs": _reflector_train_inputs,
                            "train_outputs": _reflector_train_outputs,
                            "test_inputs": _reflector_test_inputs,
                            "test_predictions": _reflector_test_preds,
                            "train_input_images": _reflector_train_input_images if use_visual_mode else [],
                            "train_output_images": _reflector_train_output_images if use_visual_mode else [],
                            "test_input_images": _reflector_test_input_images if use_visual_mode else [],
                        })
                    emit(EventType.REFLECTOR_CONTEXT, "", _reflector_emit_meta)

                    try:
                        _reflector_result = run_independent_reflection(
                            transform_hypothesis=last_transform_hypothesis,
                            code=str(selected_solution_branch.get("code") or ""),
                            test_inputs=_reflector_test_inputs,
                            test_predictions=_reflector_test_preds,
                            training_accuracy=_reflector_training_accuracy,
                            ambiguity_rationale="",
                            candidate_predictions=_reflector_candidate_payloads,
                            train_inputs=_reflector_train_inputs,
                            train_outputs=_reflector_train_outputs,
                            train_input_images=_reflector_train_input_images if use_visual_mode else None,
                            train_output_images=_reflector_train_output_images if use_visual_mode else None,
                            test_input_images=_reflector_test_input_images if use_visual_mode else None,
                            test_prediction_images=[],
                            reflector_provider=reflector_provider,
                            reflector_model=reflector_model,
                            reflector_thinking_effort=reflector_thinking_effort,
                            reflector_code_execution=reflector_code_execution,
                            emit=lambda msg: emit(EventType.SYSTEM, msg),
                            stream_emit=emit,
                            should_stop=should_stop,
                            message_history=_reflector_message_history,
                            reviewer_response=last_reviewer_response,
                        )
                        _reflector_message_history = _reflector_result.get("message_history", _reflector_message_history)
                        _reflector_verdict = _reflector_result.get("verdict", "UNKNOWN")
                        _reflector_response = _reflector_result.get("response", "")
                        _reflector_thinking = _reflector_result.get("thinking", "")

                        emit(EventType.SYSTEM,
                             f"   Reflector verdict: {_reflector_verdict}")

                        _refl_emit_meta = {
                            "turn": turn,
                            "iteration": iteration,
                            "provider": reflector_provider,
                            "model": _refl_model_label,
                            "phase": "result",
                            "verdict": _reflector_verdict,
                            "thinking": _reflector_thinking,
                            "response": _reflector_response,
                        }
                        for _ukey in (
                            "usage_input_tokens",
                            "usage_uncached_input_tokens",
                            "usage_cache_write_tokens",
                            "usage_cache_read_tokens",
                            "usage_thinking_tokens",
                            "usage_output_tokens",
                            "usage_output_includes_reasoning",
                            "usage_reasoning_tokens_reported",
                            "usage_total_tokens",
                        ):
                            if _ukey in _reflector_result:
                                _refl_emit_meta[_ukey] = _reflector_result[_ukey]
                        emit(EventType.REFLECTOR_CONTEXT, "", _refl_emit_meta)

                        with phoenix.span("Independent Reflector", {"turn": turn, "iteration": iteration}, force_flush=True):
                            phoenix.set_attribute("openinference.span.kind", "LLM")
                            phoenix.set_attribute("llm.provider", reflector_provider)
                            phoenix.set_attribute("llm.model_name", _refl_model_label)
                            phoenix.set_large_attribute("input.value", _reflector_user_msg)
                            phoenix.set_large_attribute("output.value", _reflector_response)
                            phoenix.set_attribute("verdict", _reflector_verdict)
                            if _reflector_thinking:
                                phoenix.set_large_attribute("llm.thinking", _reflector_thinking)

                    except Exception as _refl_err:
                        emit(EventType.SYSTEM,
                             f"   ⚠️ Reflector error (non-fatal): {_refl_err}")
                        # Remove the reflector canvas from history — fall back to self-reflection
                        emit(EventType.SYSTEM, "", {"_remove_reflector_canvas": True})
                        # Undo iteration/max_iterations bump so the UI doesn't show a phantom canvas
                        iteration -= 1
                        max_iterations -= 2
                        _reflector_verdict = None

                    # ── Handle reflector verdict ──────────────────────────
                    if _reflector_verdict == "APPROVE":
                        # Reflector approved → accept solution directly (skip main agent self-reflection)
                        emit(EventType.SYSTEM,
                             "   ✅ Independent reflector APPROVED — accepting solution!")
                        solved = True

                        update_internal_test_score(solution_branches=active_solution_branches)
                        last_test_visual_parts = test_prediction_only_parts
                        break  # Exit while loop

                    elif _reflector_verdict in ("REJECT", "EXPAND_CANDIDATES"):
                        _is_expand = _reflector_verdict == "EXPAND_CANDIDATES"
                        _invalidate_exploratory_runtime()
                        last_reflector_rejection = _reflector_response

                        current_context_tokens = last_actual_input_tokens + last_actual_output_tokens
                        _bypass_compression = (
                            current_context_tokens == 0
                            or current_context_tokens < compression_bypass_threshold
                        )
                        emit(EventType.SYSTEM,
                             f"   🔍 EXPAND bypass={_bypass_compression}, tokens={current_context_tokens}, refl_mode={in_reflection_mode}, refl_reject={in_reflector_reject_compression}")

                        if _is_expand:
                            feedback_text = build_candidate_expansion_guidance_prompt(
                                reflector_response=_reflector_response,
                                bypass_compression=_bypass_compression,
                            )
                        else:
                            feedback_text = _build_reflector_reject_feedback(
                                _reflector_response,
                                bypass_compression=_bypass_compression,
                            )

                        if _bypass_compression:
                            emit(EventType.SYSTEM,
                                 f"   ℹ️ Context below bypass threshold ({current_context_tokens:,} < {compression_bypass_threshold:,}) — continuing without context reset")
                            messages.append({"role": "user", "content": [{"type": "text", "text": feedback_text}]})

                            iteration += 1
                            emit(EventType.SYSTEM, f"   📊 Starting Iteration {iteration}/{max_iterations}")
                            emit(EventType.TURN_START, "", {
                                 "turn": turn, "iteration": iteration, "max_iterations": max_iterations,
                                 "context_label": "Solver (post-reflector)"})
                            # Note: _reflector_prompt_bundle is intentionally NOT emitted here;
                            # the REFLECTION event below carries system_md and snapshot_messages,
                            # so emitting both would render the system prompt twice in the UI.
                            emit(
                                EventType.REFLECTION,
                                "",
                                _iteration_context_metadata(
                                    iteration=iteration,
                                    ui_title=f"📋 **Iteration {iteration} — Full Context (bypass)**",
                                    context_mode="snapshot",
                                    snapshot_messages=messages,
                                    resume_messages=messages,
                                ),
                            )
                            last_test_visual_parts = test_prediction_only_parts
                            current_iteration_start_idx = len(messages)
                            # Clear reflection flags — solver continues normally with tools
                            in_reflection_mode = False
                            in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                                test_generalization=False,
                                reflector_reject_compression=False,
                            )
                        else:
                            _label = "expansion" if _is_expand else "rejection"
                            emit(EventType.SYSTEM,
                                 f"   🔄 Independent reflector {_reflector_verdict} — feeding back to solver with context compression...")
                            combined_prompt = feedback_text + CONTEXT_COMPRESSION_PROMPT
                            combined_content = [{"type": "text", "text": combined_prompt}]

                            snapshot_messages = messages[current_iteration_start_idx:] + [
                                {"role": "user", "content": combined_content}
                            ]
                            messages.append({"role": "user", "content": combined_content})
                            emit(EventType.TURN_START, "", {
                                 "turn": turn, "iteration": iteration, "max_iterations": max_iterations,
                                 "context_label": "Solver (post-reflector)"})
                            # Note: _reflector_prompt_bundle is intentionally NOT emitted here;
                            # the REFLECTION event below carries system_md and snapshot_messages,
                            # so emitting both would render the system prompt twice in the UI.
                            emit(
                                EventType.REFLECTION,
                                "",
                                _iteration_context_metadata(
                                    iteration=iteration,
                                    ui_title=f"📋 **Solver (post-reflector — {_label})**",
                                    context_mode="snapshot",
                                    snapshot_messages=snapshot_messages,
                                    resume_messages=messages,
                                ),
                            )
                            in_reflection_mode = True
                            in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                                test_generalization=False,
                                reflector_reject_compression=True,
                            )
                            last_test_visual_parts = test_prediction_only_parts

                            with phoenix.span(f"🪞 Reflector {_reflector_verdict} + Context Compression", {"turn": turn, "iteration": iteration}, force_flush=True):
                                phoenix.set_large_attribute("prompt", combined_prompt)

                        update_internal_test_score(solution_branches=active_solution_branches)

                    else:
                        # ERROR or unknown verdict → fall back to main agent self-reflection
                        # Stay on current iteration/canvas; reflector canvas already removed above
                        pass

                elif enable_independent_reflector and not last_transform_hypothesis:
                    emit(EventType.SYSTEM,
                         f"   ⚠️ No transform hypothesis recorded — skipping independent reflector")

                # ── Self-reflection (only when reflector didn't give a clear verdict) ──
                if not solved and not in_reflector_reject_compression and _reflector_verdict not in ("APPROVE", "REJECT", "EXPAND_CANDIDATES"):
                    emit(EventType.SYSTEM,
                         f"   -> Injecting test generalization reflection (model will self-verify)...")
                    in_reflection_mode = True
                    in_test_generalization_reflection, in_reflector_reject_compression = _set_reflection_submode(
                        test_generalization=True,
                        reflector_reject_compression=False,
                    )

                    generalization_reflection = TEST_GENERALIZATION_REFLECTION_PROMPT

                    # Emit the reflection prompt to UI
                    emit(EventType.REFLECTION, generalization_reflection,
                         {"iteration": iteration})

                    generalization_content = [
                        {"type": "text", "text": generalization_reflection}]
                    generalization_content.extend(test_visual_parts)

                    messages.append(
                        {"role": "user", "content": generalization_content})
                    last_test_visual_parts = test_prediction_only_parts
                    emit(EventType.SYSTEM,
                         f"   -> Added test generalization reflection with {len(test_visual_parts)} visual parts")

                    with phoenix.span("🪞 Test Generalization Reflection Prompt", {"turn": turn, "iteration": iteration}, force_flush=True):
                        phoenix.set_attribute(
                            "num_visual_parts", len(test_visual_parts))
                        phoenix.set_large_attribute(
                            "prompt", generalization_reflection)

                    update_internal_test_score(solution_branches=active_solution_branches)

                    # Case 2: best-effort mode — auto-accept without reflector
                    if _best_effort_injected:
                        emit(EventType.SYSTEM,
                             "   ⚡ Best-effort mode — auto-accepting submission")
                        solved = True
                        break

            # After execute_python_solution failures, inject reflection prompt
            if fname == "execute_python_solution" and results is not None and not results.get('correct', False):
                emit(EventType.SYSTEM,
                     f"   -> Injecting reflection prompt to trigger thinking...")

                in_reflection_mode = True

                # NOTE: We don't repeat hypothesis/code/accuracy here - it's already in the tool result above.
                # The reflection prompt just asks the agent to reflect on what they already see.

                reflection_text = TRAIN_FAILURE_REFLECTION_PROMPT

                reflection_content = [
                    {"type": "text", "text": reflection_text}]

                if visual_parts:
                    emit(
                        EventType.SYSTEM, f"   -> Adding {len(visual_parts)} visual feedback parts to reflection prompt")
                    reflection_content.append(
                        {"type": "text", "text": "\n\nHere is the visual feedback for failed examples:"})
                    reflection_content.extend(visual_parts)

                messages.append(
                    {"role": "user", "content": reflection_content})

                # Emit the actual reflection prompt to UI (belongs to current iteration, not next)
                emit(EventType.REFLECTION, reflection_text,
                     {"iteration": iteration})

                with phoenix.span("🪞 Train Failure Reflection Prompt", {"turn": turn, "iteration": iteration}, force_flush=True):
                    phoenix.set_attribute("num_visual_parts", len(
                        visual_parts) if visual_parts else 0)
                    phoenix.set_large_attribute("prompt", reflection_text)


            # Case 2: approaching turn exhaustion with no 100%-training solution ever achieved.
            # Checked on EVERY turn (not just after execute_python_solution failures) because
            # the solver may spend its final turns on exploratory code without submitting.
            _approaching_exhaustion = (turn >= max_turns - 8) if not isinstance(max_turns, float) else False
            if _approaching_exhaustion and not last_fully_passing_branches and not _best_effort_injected:
                _best_effort_injected = True
                max_turns = float('inf')  # remove turn cap — solver can explore freely
                in_reflection_mode = False  # keep all tools available
                emit(EventType.SYSTEM,
                     "   ⚡ Training-inconsistent puzzle detected — injecting best-effort prompt (tools remain available)")
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": BEST_EFFORT_PROMPT}],
                })

        except Exception as e:
            if _is_transient_api_error(e):
                consecutive_transient_turn_failures += 1
                emit(EventType.SYSTEM,
                     f"   ⚠️ Transient API error (will retry): {e}")
                if consecutive_transient_turn_failures >= MAX_CONSECUTIVE_TRANSIENT_TURN_FAILURES:
                    emit(EventType.ERROR,
                         f"Stopping run after {consecutive_transient_turn_failures} consecutive transient failures.")
                    break
                time.sleep(min(8, 2 ** min(consecutive_transient_turn_failures, 3)))
                continue
            emit(EventType.ERROR,
                 f"Error in orchestration loop: {e}\n{traceback.format_exc()}")
            break

    # Case 1: auto-submit last 100%-training solution after reflector exhaustion
    if not solved and last_fully_passing_branches and not should_stop():
        emit(EventType.SYSTEM,
             "⚡ Auto-submitting last 100%-training solution after exhaustion")
        solved = True
        update_internal_test_score(solution_branches=last_fully_passing_branches)

    # Close iteration span if open
    if current_iteration_span:
        current_iteration_span.__exit__(None, None, None)

    if root_span_ctx:
        # Log result summary to root span before closing
        if solved and last_task_fully_solved:
            phoenix.set_attribute("result", "solved")
        elif solved and last_test_solved_indices is not None and last_test_total is not None:
            solved_parts = []
            for ti in range(last_test_total):
                if ti in (last_test_solved_indices or []):
                    solved_parts.append(f"test_{ti}:solved")
                else:
                    solved_parts.append(f"test_{ti}:unsolved")
            phoenix.set_attribute("result", " | ".join(solved_parts))
        elif solved:
            phoenix.set_attribute("result", "solved")
        else:
            phoenix.set_attribute("result", "unsolved")
        root_span_ctx.__exit__(None, None, None)

    if solved:
        solved_indices_msg = ""
        if (
            last_test_total is not None
            and last_test_total > 1
            and last_test_solved_indices is not None
        ):
            if last_test_solved_indices:
                if len(last_test_solved_indices) == 1:
                    solved_indices_msg = f"; solved test example {last_test_solved_indices[0]}"
                else:
                    solved_examples = ", ".join(
                        f"test example {idx}" for idx in last_test_solved_indices
                    )
                    solved_indices_msg = f"; solved {solved_examples}"
            else:
                solved_indices_msg = "; solved test outputs: none"
        completion_metadata = {
            "solved": True,
            "test_accuracy": last_test_accuracy,
            "task_fully_solved": last_task_fully_solved,
            "test_correct_count": last_test_correct_count,
            "test_total": last_test_total,
            "test_solved_indices": last_test_solved_indices,
        }
        if last_test_accuracy is not None and last_test_correct_count is not None and last_test_total is not None:
            score_msg = (
                f"ARC task score: {last_test_accuracy:.1%} "
                f"({last_test_correct_count}/{last_test_total} test outputs correct{solved_indices_msg})"
            )
            if last_task_fully_solved:
                system_msg = f"\n🎉 TASK FULLY SOLVED AND VERIFIED! ({score_msg})"
                complete_msg = f"Task fully solved ({score_msg})"
            else:
                system_msg = (
                    "\n✅ Accepted candidate found and verified on training set. "
                    f"({score_msg}; task partially solved)"
                )
                complete_msg = f"Accepted candidate ({score_msg}; task partially solved)"
        else:
            system_msg = "\n🎉 SOLUTION FOUND AND VERIFIED!"
            complete_msg = "Solution found"

        emit(EventType.SYSTEM, system_msg)
        emit(EventType.COMPLETE, complete_msg, completion_metadata)
    else:
        final_stop_reason = str(stop_reason() or "").strip().lower() if stop_reason else ""
        if final_stop_reason == "pause":
            emit(EventType.SYSTEM, "\n⏸️ Pause requested — checkpoint preserved.")
        elif final_stop_reason == "terminate":
            emit(EventType.SYSTEM, "\n⏹️ Run terminated.")
        if final_stop_reason not in ("pause", "terminate"):
            emit(EventType.SYSTEM, "\n⛔ Max turns reached or no solution found.")
            emit(EventType.COMPLETE, "No solution", {"solved": False})

    # Shutdown Phoenix if we auto-initialized it
    if phoenix_auto_initialized and phoenix.enabled:
        phoenix.shutdown()

    return {
        'solved': solved,
        'best_effort': bool(_best_effort_injected and solved),
        'stopped_reason': (str(stop_reason() or "").strip().lower() if stop_reason else ""),
        'iterations': iteration,
        'final_code': final_code,
        'iteration_history': iteration_history,
        'test_accuracy': last_test_accuracy if solved else None,
        'task_fully_solved': last_task_fully_solved if solved else None,
        'test_correct_count': last_test_correct_count if solved else None,
        'test_total': last_test_total if solved else None,
        'test_solved_indices': last_test_solved_indices if solved else None,
    }


def main():
    """CLI entry point - parses args and calls run_orchestration()."""
    parser = argparse.ArgumentParser(
        description='Solve ARC puzzles using Anthropic Claude')
    parser.add_argument('--puzzle_path', type=str,
                        default=DEFAULT_PUZZLE_PATH, help='Path to puzzle JSON, or task id when dataset root is configured')
    parser.add_argument('--dataset_root', type=str, default=os.getenv("ARC_DATA_ROOT"),
                        help='Dataset root for task-id resolution (or set ARC_DATA_ROOT)')
    parser.add_argument('--dataset_split', type=str, default='public_eval',
                        help='Dataset split for task-id resolution (default: public_eval)')
    parser.add_argument('--streaming', type=lambda x: x.lower()
                        in ['true', '1', 'yes'], default=True, help='Use streaming API')
    parser.add_argument('--model', type=str, default='claude-opus-4-6',
                        help='Model name (default: claude-opus-4-6)')
    parser.add_argument('--visual_mode', type=lambda x: x.lower() in [
                        'true', '1', 'yes'], default=True, help='Use visual mode (images) in prompts and tool responses')
    parser.add_argument('--extended_thinking', type=lambda x: x.lower() in [
                        'true', '1', 'yes'], default=True, help='Enable extended thinking (Claude 3.5+)')
    parser.add_argument('--thinking_budget', type=int, default=64000,
                        help='Max tokens for extended thinking (default: 64000, max for Opus 4.6)')
    parser.add_argument('--thinking_effort', type=str, default='medium',
                        help='Thinking effort level (low, medium, high, max)')
    parser.add_argument('--max_iterations', type=int,
                        default=10, help='Maximum solve iterations')
    parser.add_argument('--unsafe_local_exec', type=lambda x: x.lower() in [
                        'true', '1', 'yes'], default=True, help='Allow model-generated local Python execution')
    parser.add_argument('--enable_phoenix', type=lambda x: x.lower() in [
                        'true', '1', 'yes'], default=True, help='Enable Phoenix tracing')
    args = parser.parse_args()

    # Call the canonical run_orchestration function
    result = run_orchestration(
        puzzle_path=args.puzzle_path,
        model_name=args.model,
        use_streaming=args.streaming,
        use_visual_mode=args.visual_mode,
        use_extended_thinking=args.extended_thinking,
        thinking_budget=args.thinking_budget,
        thinking_effort=args.thinking_effort,
        max_iterations=args.max_iterations,
        unsafe_local_exec=args.unsafe_local_exec,
        enable_phoenix=args.enable_phoenix,
        dataset_root=args.dataset_root,
        dataset_split=args.dataset_split,
        event_callback=None,  # Use default CLI callback
        phoenix=None,  # Auto-initialize
        should_stop=None,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(
        f"Summary: solved={result['solved']}, iterations={result['iterations']}")
    if result.get('test_accuracy') is not None:
        correct = result.get('test_correct_count')
        total = result.get('test_total')
        if correct is not None and total is not None:
            print(f"ARC task score: {result['test_accuracy']:.1%} ({correct}/{total} test outputs correct)")
            print(f"Task fully solved: {bool(result.get('task_fully_solved'))}")
        else:
            print(f"ARC task score: {result['test_accuracy']:.1%}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
