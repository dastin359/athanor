#!/usr/bin/env python3
"""Independent reflector for ARC puzzle solutions.

This module implements an independent review step that evaluates a solver's
proposed transformation rule and code using a fresh model context (no access
to the solver's reasoning chain).

Supports two backends:
  - Anthropic Claude (via anthropic SDK)
  - Google Gemini (via google-genai SDK)
"""

import base64
import os
import time
from pathlib import Path
from typing import Optional, Callable

from dotenv import load_dotenv, find_dotenv

ANTHROPIC_PROMPT_CACHING_BETA = "prompt-caching-2024-07-31"

def _ordinal(n: int) -> str:
    """Return ordinal string for a positive integer (1st, 2nd, 3rd, …)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{('th','st','nd','rd')[min(n % 10, 4) if n % 10 < 4 else 0]}"

load_dotenv(find_dotenv(usecwd=True))

# ---------------------------------------------------------------------------
# Lazy imports — these are only needed when the respective backend is used
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    """Create an Anthropic client."""
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is required for Claude reflector")
    return anthropic.Anthropic(api_key=api_key)


def _get_gemini_client():
    """Create a Google GenAI client (Vertex AI or AI Studio)."""
    from google import genai

    backend = (os.getenv("GENAI_BACKEND") or "vertex").strip().lower()
    if backend in {"aistudio", "ai_studio", "studio", "gemini"}:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GENAI_BACKEND=aistudio requires GOOGLE_API_KEY or GEMINI_API_KEY"
            )
        return genai.Client(api_key=api_key)

    # Default: Vertex AI
    project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError(
            "Vertex reflector requires VERTEX_PROJECT_ID or GOOGLE_CLOUD_PROJECT."
        )
    location = os.getenv("VERTEX_LOCATION") or "global"
    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )


def _anthropic_cache_control(ttl: str) -> dict[str, str]:
    return {"type": "ephemeral", "ttl": ttl}


def _cached_reflector_system(system_prompt: str) -> list[dict[str, object]]:
    return [{
        "type": "text",
        "text": str(system_prompt or ""),
        "cache_control": _anthropic_cache_control("5m"),
    }]


def _with_reflector_cache_headers(api_params: dict) -> dict:
    updated = dict(api_params)
    headers = dict(updated.get("extra_headers") or {})
    headers["anthropic-beta"] = ANTHROPIC_PROMPT_CACHING_BETA
    updated["extra_headers"] = headers
    return updated


def _extract_claude_usage(raw_response) -> dict[str, int | bool]:
    usage = getattr(raw_response, "usage", None)
    if usage is None:
        return {}

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

    uncached_input_tokens = _read_int(["input_tokens", "prompt_tokens", "prompt_token_count"])
    cache_write_tokens = _read_int(["cache_creation_input_tokens"])
    cache_read_tokens = _read_int(["cache_read_input_tokens"])
    output_tokens = _read_int(["output_tokens", "completion_tokens", "completion_token_count", "generated_tokens"])

    meta: dict[str, int | bool] = {
        "usage_input_tokens": uncached_input_tokens + cache_write_tokens + cache_read_tokens,
        "usage_uncached_input_tokens": uncached_input_tokens,
        "usage_cache_write_tokens": cache_write_tokens,
        "usage_cache_read_tokens": cache_read_tokens,
        "usage_output_tokens": output_tokens,
        "usage_output_includes_reasoning": True,
        "usage_reasoning_tokens_reported": False,
    }

    for path_name in (
        "output_tokens_details.reasoning_tokens",
        "output_tokens_details.thinking_tokens",
        "reasoning_tokens",
        "thinking_tokens",
    ):
        try:
            cursor = usage
            for part in path_name.split("."):
                cursor = getattr(cursor, part, None)
            tokens = int(cursor or 0)
            if tokens > 0:
                meta["usage_thinking_tokens"] = tokens
                meta["usage_reasoning_tokens_reported"] = True
                break
        except Exception:
            continue

    meta.setdefault("usage_thinking_tokens", 0)
    meta["usage_total_tokens"] = int(meta["usage_input_tokens"]) + int(meta["usage_output_tokens"])
    return meta


_REFLECTOR_CODE_EXECUTION_SECTION = """
---

## CODE EXECUTION

You have access to a Python sandbox. Use it to verify the solver's outputs
and test alternative hypotheses — don't just mentally trace, run the code.
In particular, when checking whether an alternative rule produces different
test outputs, modify and re-run the solver's submitted code rather than
reasoning about it abstractly.
"""


def _load_reflector_prompt(code_execution: bool = False) -> str:
    """Load the reflector system prompt, optionally appending code execution instructions."""
    prompt_path = Path(__file__).parent / "REFLECTOR_SYSTEM_PROMPT.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Reflector system prompt not found at {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read()
    if code_execution:
        prompt += _REFLECTOR_CODE_EXECUTION_SECTION
    return prompt


def _build_reflector_user_message(
    transform_hypothesis: str,
    code: str,
    train_inputs: list | None,
    train_outputs: list | None,
    test_inputs: list,
    test_predictions: list,
    training_accuracy: str = "",
    ambiguity_rationale: str = "",
    candidate_predictions: list[dict] | None = None,
    test_input_images: list[str] | None = None,
    test_prediction_images: list[str] | None = None,
    reviewer_response: str = "",
) -> str:
    """Build the user message text for the reflector (text-only portion).

    Order: puzzle data → hypothesis → code → predictions.
    Images are handled separately per backend.
    """
    parts = []

    # 1. Puzzle data: training examples
    if train_inputs and train_outputs:
        parts.append("## Training Input/Output Pairs\n")
        parts.append("Study these examples first to form your own understanding of the transformation.\n")
        for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
            parts.append(f"### Training Example {i}")
            parts.append(f"**Input Grid:** {inp}")
            parts.append(f"**Expected Output Grid:** {out}")
            parts.append("")

    # 2. Puzzle data: test inputs (without predictions yet)
    parts.append("## Test Inputs\n")
    for i, inp in enumerate(test_inputs):
        parts.append(f"### Test Example {i}")
        parts.append(f"**Input Grid:** {inp}")
        parts.append("")

    normalized_candidates = list(candidate_predictions or [])
    if not normalized_candidates:
        normalized_candidates = [
            {
                "index": i,
                "candidates": [pred] if pred is not None else [],
            }
            for i, pred in enumerate(test_predictions or [])
        ]

    parts.append("## Solver Submission\n")
    if training_accuracy:
        parts.append(f"**Training Accuracy:** {training_accuracy}")
        parts.append("")
    parts.append("**Hypothesis:**")
    parts.append(str(transform_hypothesis or "(No hypothesis provided)"))
    parts.append("")
    parts.append("**solve() Code:**")
    parts.append("```python")
    parts.append(str(code or "(No code provided)"))
    parts.append("```")
    parts.append("")

    if ambiguity_rationale:
        parts.append("**Ambiguity Description:**")
        parts.append(str(ambiguity_rationale))
        parts.append("")

    if reviewer_response:
        parts.append("**Response to Reviewer:**")
        parts.append(str(reviewer_response))
        parts.append("")

    parts.append("**Candidate Outputs By Test Example:**")
    for row in normalized_candidates:
        test_index = row.get("index", "?")
        parts.append(f"### Test Example {test_index}")
        if row.get("error"):
            parts.append(f"Error: {row.get('error')}")
        for candidate_idx, pred in enumerate(row.get("candidates", []) or [], start=1):
            parts.append(f"- {_ordinal(candidate_idx)} candidate: {pred}")
        parts.append("")

    parts.append("---\n")
    parts.append("Please perform your independent review following the guidelines in your system prompt.")

    return "\n".join(parts)


_FOLLOWUP_PREAMBLE = (
    "The solver has revised their submission based on your previous feedback. Evaluate this new submission on its own merits — verify that prior concerns are actually resolved, don't assume they are.\n"
)


def _build_reflector_followup_message(
    transform_hypothesis: str,
    code: str,
    test_predictions: list,
    training_accuracy: str = "",
    ambiguity_rationale: str = "",
    candidate_predictions: list[dict] | None = None,
    reviewer_response: str = "",
) -> str:
    """Build a submission-only follow-up message for turn >= 2.

    Omits puzzle data (training examples, test inputs) — those are already
    in the conversation history from turn 1.  Prefixes with an anti-anchoring
    preamble so the reflector evaluates the new submission on its own merits.
    """
    normalized_candidates = list(candidate_predictions or [])
    if not normalized_candidates:
        normalized_candidates = [
            {
                "index": i,
                "candidates": [pred] if pred is not None else [],
            }
            for i, pred in enumerate(test_predictions or [])
        ]

    parts = [_FOLLOWUP_PREAMBLE]

    parts.append("## Solver Submission\n")
    if training_accuracy:
        parts.append(f"**Training Accuracy:** {training_accuracy}")
        parts.append("")
    parts.append("**Hypothesis:**")
    parts.append(str(transform_hypothesis or "(No hypothesis provided)"))
    parts.append("")
    parts.append("**solve() Code:**")
    parts.append("```python")
    parts.append(str(code or "(No code provided)"))
    parts.append("```")
    parts.append("")

    if ambiguity_rationale:
        parts.append("**Ambiguity Description:**")
        parts.append(str(ambiguity_rationale))
        parts.append("")

    if reviewer_response:
        parts.append("**Response to Reviewer:**")
        parts.append(str(reviewer_response))
        parts.append("")

    parts.append("**Candidate Outputs By Test Example:**")
    for row in normalized_candidates:
        test_index = row.get("index", "?")
        parts.append(f"### Test Example {test_index}")
        if row.get("error"):
            parts.append(f"Error: {row.get('error')}")
        for candidate_idx, pred in enumerate(row.get("candidates", []) or [], start=1):
            parts.append(f"- {_ordinal(candidate_idx)} candidate: {pred}")
        parts.append("")

    parts.append("---\n")
    parts.append("Please perform your independent review following the guidelines in your system prompt.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Claude reflector
# ---------------------------------------------------------------------------

def reflect_with_claude(
    transform_hypothesis: str,
    code: str,
    test_inputs: list,
    test_predictions: list,
    training_accuracy: str = "100% on training set",
    ambiguity_rationale: str = "",
    candidate_predictions: list[dict] | None = None,
    train_inputs: list | None = None,
    train_outputs: list | None = None,
    train_input_images: list[str] | None = None,
    train_output_images: list[str] | None = None,
    test_input_images: list[str] | None = None,
    test_prediction_images: list[str] | None = None,
    model_name: str = "claude-sonnet-4-20250514",
    thinking_effort: str = "high",
    emit: Optional[Callable] = None,
    stream_emit: Optional[Callable] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    message_history: list[dict] | None = None,
    reviewer_response: str = "",
) -> dict:
    """Run independent reflection using Claude.

    Args:
        transform_hypothesis: NLP description of the transform rule
        code: solve() function source code
        test_inputs: List of test input grids (as lists)
        test_predictions: List of predicted test output grids (as lists)
        training_accuracy: Summary string of training accuracy
        train_inputs: List of training input grids (as lists)
        train_outputs: List of training output grids (as lists)
        train_input_images: Base64-encoded PNG images of training inputs
        train_output_images: Base64-encoded PNG images of training outputs
        test_input_images: Base64-encoded PNG images of test inputs
        test_prediction_images: Base64-encoded PNG images of test predictions
        model_name: Claude model to use
        thinking_effort: Thinking effort level ("low", "medium", "high", "max")
        emit: Optional callback for status messages

    Returns:
        dict with keys: verdict, confidence, findings, concerns, thinking, response, raw
    """
    if emit is None:
        def emit(msg): pass
    if should_stop is None:
        def should_stop() -> bool:
            return False

    client = _get_anthropic_client()
    system_prompt = _load_reflector_prompt()

    is_followup = bool(message_history)

    # -- Build content_blocks for the CURRENT turn's user message ----------
    content_blocks = []
    normalized_candidates = list(candidate_predictions or [])
    if not normalized_candidates:
        normalized_candidates = [
            {
                "index": i,
                "candidates": [pred] if pred is not None else [],
                "candidate_images": [],
            }
            for i, pred in enumerate(test_predictions or [])
        ]

    if not is_followup:
        # ── Turn 1: full puzzle data + submission ─────────────────────
        # Training examples with images (Phase 1)
        train_input_images = train_input_images or []
        train_output_images = train_output_images or []
        if train_inputs and train_outputs:
            content_blocks.append({"type": "text", "text": "## Training Input/Output Pairs\n"})
            for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
                content_blocks.append({"type": "text", "text": f"### Training Example {i}\n**Input Grid:** {inp}"})
                if i < len(train_input_images) and train_input_images[i]:
                    content_blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": train_input_images[i]}
                    })
                content_blocks.append({"type": "text", "text": f"**Expected Output Grid:** {out}"})
                if i < len(train_output_images) and train_output_images[i]:
                    content_blocks.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": train_output_images[i]}
                    })

        # Test inputs with images
        content_blocks.append({"type": "text", "text": "## Test Inputs\n"})
        test_input_images = test_input_images or []
        for i, inp in enumerate(test_inputs):
            content_blocks.append({"type": "text", "text": f"### Test Example {i}\n**Input Grid:** {inp}"})
            if i < len(test_input_images) and test_input_images[i]:
                content_blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": test_input_images[i]}
                })

        submission_parts = ["## Solver Submission\n"]
        if training_accuracy:
            submission_parts.append(f"**Training Accuracy:** {training_accuracy}\n")
        submission_parts.append(
            f"**Hypothesis:** {transform_hypothesis or '(No hypothesis provided)'}\n\n"
            f"**solve() Code:**\n```python\n{code or '(No code provided)'}\n```"
        )
        if ambiguity_rationale:
            submission_parts.append(f"\n\n**Ambiguity Description:** {ambiguity_rationale}")
        if reviewer_response:
            submission_parts.append(f"\n\n**Response to Reviewer:** {reviewer_response}")
        submission_text = "".join(submission_parts)
        content_blocks.append({"type": "text", "text": submission_text})
    else:
        # ── Turn >= 2: submission-only with anti-anchoring preamble ───
        content_blocks.append({"type": "text", "text": _FOLLOWUP_PREAMBLE})

        submission_parts = ["## Solver Submission\n"]
        if training_accuracy:
            submission_parts.append(f"**Training Accuracy:** {training_accuracy}\n")
        submission_parts.append(
            f"**Hypothesis:** {transform_hypothesis or '(No hypothesis provided)'}\n\n"
            f"**solve() Code:**\n```python\n{code or '(No code provided)'}\n```"
        )
        if ambiguity_rationale:
            submission_parts.append(f"\n\n**Ambiguity Description:** {ambiguity_rationale}")
        if reviewer_response:
            submission_parts.append(f"\n\n**Response to Reviewer:** {reviewer_response}")
        submission_text = "".join(submission_parts)
        content_blocks.append({"type": "text", "text": submission_text})

    # Candidate outputs (always included — these change each turn)
    content_blocks.append({"type": "text", "text": "## Candidate Outputs By Test Example\n"})
    for row in normalized_candidates:
        test_index = row.get("index", "?")
        content_blocks.append(
            {
                "type": "text",
                "text": f"### Test Example {test_index}",
            }
        )
        if row.get("error"):
            content_blocks.append({"type": "text", "text": f"Error: {row.get('error')}"})
            continue
        candidate_images = row.get("candidate_images", []) or []
        for candidate_idx, pred in enumerate(row.get("candidates", []) or [], start=1):
            content_blocks.append(
                {"type": "text", "text": f"{_ordinal(candidate_idx)} candidate: {pred}"}
            )
            image_offset = candidate_idx - 1
            if image_offset < len(candidate_images) and candidate_images[image_offset]:
                content_blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": candidate_images[image_offset]}
                })

    content_blocks.append({
        "type": "text",
        "text": "---\nPlease perform your independent review following the guidelines in your system prompt."
    })

    # -- Build messages list with conversation history ---------------------
    if is_followup:
        messages = []
        # Find last user message index for prefix cache breakpoint
        last_user_idx = max(
            i for i, t in enumerate(message_history) if t["role"] == "user"
        )
        for i, turn in enumerate(message_history):
            if turn["role"] == "user" and i == last_user_idx:
                # Cache breakpoint on last history user message
                messages.append({"role": "user", "content": [{
                    "type": "text",
                    "text": turn["content"],
                    "cache_control": _anthropic_cache_control("5m"),
                }]})
            elif turn["role"] == "assistant":
                assistant_content = []
                # Include signed thinking block only from Claude turns
                # (Gemini signatures are incompatible with Claude's format)
                if turn.get("provider") == "claude" and turn.get("thinking") and turn.get("thinking_signature"):
                    assistant_content.append({
                        "type": "thinking",
                        "thinking": turn["thinking"],
                        "signature": turn["thinking_signature"],
                    })
                assistant_content.append({"type": "text", "text": turn["content"]})
                messages.append({"role": "assistant", "content": assistant_content})
            else:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": content_blocks})
    else:
        messages = [{"role": "user", "content": content_blocks}]

    # -- Text-only snapshot of current user content for history storage ----
    if is_followup:
        user_text_for_history = _build_reflector_followup_message(
            transform_hypothesis=transform_hypothesis,
            code=code,
            test_predictions=test_predictions,
            training_accuracy=training_accuracy,
            ambiguity_rationale=ambiguity_rationale,
            candidate_predictions=candidate_predictions,
            reviewer_response=reviewer_response,
        )
    else:
        user_text_for_history = _build_reflector_user_message(
            transform_hypothesis=transform_hypothesis,
            code=code,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_inputs=test_inputs,
            test_predictions=test_predictions,
            training_accuracy=training_accuracy,
            ambiguity_rationale=ambiguity_rationale,
            candidate_predictions=candidate_predictions,
            reviewer_response=reviewer_response,
        )

    # Build API params
    api_params = {
        "model": model_name,
        "max_tokens": 128000,
        "system": _cached_reflector_system(system_prompt),
        "messages": messages,
    }

    api_params["thinking"] = {"type": "adaptive"}
    api_params["output_config"] = {"effort": thinking_effort}

    api_params = _with_reflector_cache_headers(api_params)

    try:
        from .events import EventType
    except ImportError:
        from events import EventType

    _MAX_RETRIES = 2
    _RETRY_DELAY = 30  # seconds

    for _attempt in range(_MAX_RETRIES):
        emit(f"   🤖 Calling Claude reflector (streaming){'...' if _attempt == 0 else f' (retry {_attempt})...'}")

        thinking_text = ""
        response_text = ""
        final_message = None
        stopped_mid_stream = False

        # Capture usage & signature from stream events (fallback when
        # get_final_message() fails).
        _stream_input_tokens = 0
        _stream_output_tokens = 0
        _stream_cache_creation = 0
        _stream_cache_read = 0
        _stream_thinking_signature = ""

        try:
            with client.messages.stream(**api_params) as stream:
                for event in stream:
                    if should_stop():
                        stopped_mid_stream = True
                        break
                    if event.type == "content_block_delta" and hasattr(event, "delta"):
                        delta = event.delta
                        if getattr(delta, "type", None) == "thinking_delta":
                            chunk = getattr(delta, "thinking", "") or ""
                            thinking_text += chunk
                            if stream_emit and chunk:
                                try:
                                    stream_emit(EventType.THINKING, chunk, {"reflector": True})
                                except Exception:
                                    pass
                        elif getattr(delta, "type", None) == "text_delta":
                            chunk = getattr(delta, "text", "") or ""
                            response_text += chunk
                            if stream_emit and chunk:
                                try:
                                    stream_emit(EventType.TEXT, chunk, {"reflector": True})
                                except Exception:
                                    pass
                        elif getattr(delta, "type", None) == "signature_delta":
                            _stream_thinking_signature = getattr(delta, "signature", "") or ""
                    elif event.type == "message_start":
                        msg = getattr(event, "message", None)
                        if msg:
                            u = getattr(msg, "usage", None)
                            if u:
                                _stream_input_tokens = int(getattr(u, "input_tokens", 0) or 0)
                                _stream_cache_creation = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
                                _stream_cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
                    elif event.type == "message_delta":
                        u = getattr(event, "usage", None)
                        if u:
                            _stream_output_tokens = int(getattr(u, "output_tokens", 0) or 0)
                if not stopped_mid_stream:
                    try:
                        final_message = stream.get_final_message()
                    except Exception as _gfm_err:
                        emit(f"   ⚠️ get_final_message() failed: {_gfm_err}")
                        final_message = None
            break  # success — exit retry loop
        except Exception as e:
            err_str = str(e)
            is_server_error = "500" in err_str or "server error" in err_str.lower() or "overloaded" in err_str.lower()
            if is_server_error and _attempt < _MAX_RETRIES - 1:
                emit(f"   ⚠️ Claude API server error (attempt {_attempt + 1}/{_MAX_RETRIES}). Retrying in {_RETRY_DELAY}s...")
                import time
                time.sleep(_RETRY_DELAY)
                continue
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "findings": [],
                "concerns": [f"API error: {e}"],
                "thinking": "",
                "response": f"Error calling Claude: {e}",
                "raw": None,
            }

    if stopped_mid_stream:
        _stopped = {
            "verdict": "ERROR",
            "confidence": 0,
            "findings": [],
            "concerns": ["Reflection interrupted by stop request"],
            "thinking": thinking_text,
            "response": response_text,
            "raw": None,
        }
        if _stream_output_tokens > 0:
            _stopped["usage_input_tokens"] = _stream_input_tokens + _stream_cache_creation + _stream_cache_read
            _stopped["usage_uncached_input_tokens"] = _stream_input_tokens
            _stopped["usage_cache_write_tokens"] = _stream_cache_creation
            _stopped["usage_cache_read_tokens"] = _stream_cache_read
            _stopped["usage_output_tokens"] = _stream_output_tokens
            _stopped["usage_output_includes_reasoning"] = True
            _stopped["usage_reasoning_tokens_reported"] = False
            _stopped["usage_thinking_tokens"] = 0
            _stopped["usage_total_tokens"] = _stopped["usage_input_tokens"] + _stream_output_tokens
        return _stopped

    result = _parse_reflector_response(response_text, thinking_text, final_message)
    if final_message is not None:
        result.update(_extract_claude_usage(final_message))

    # Fallback: use stream-event usage if get_final_message failed or had no usage
    if "usage_output_tokens" not in result and _stream_output_tokens > 0:
        result["usage_input_tokens"] = _stream_input_tokens + _stream_cache_creation + _stream_cache_read
        result["usage_uncached_input_tokens"] = _stream_input_tokens
        result["usage_cache_write_tokens"] = _stream_cache_creation
        result["usage_cache_read_tokens"] = _stream_cache_read
        result["usage_output_tokens"] = _stream_output_tokens
        result["usage_output_includes_reasoning"] = True
        result["usage_reasoning_tokens_reported"] = False
        result["usage_thinking_tokens"] = 0
        result["usage_total_tokens"] = result["usage_input_tokens"] + _stream_output_tokens

    # Store updated conversation history (include thinking + signature for
    # proper multi-turn replay — Claude requires signed thinking blocks).
    updated_history = list(message_history or [])
    updated_history.append({"role": "user", "content": user_text_for_history})
    assistant_entry: dict[str, str] = {"role": "assistant", "content": response_text}
    if thinking_text:
        assistant_entry["thinking"] = thinking_text
    assistant_entry["provider"] = "claude"
    # Extract thinking signature from final_message content blocks
    if final_message is not None:
        for block in getattr(final_message, "content", []):
            if getattr(block, "type", None) == "thinking":
                sig = getattr(block, "signature", None)
                if sig:
                    assistant_entry["thinking_signature"] = sig
                break
    # Fallback: use stream-captured signature
    if "thinking_signature" not in assistant_entry and _stream_thinking_signature:
        assistant_entry["thinking_signature"] = _stream_thinking_signature
    updated_history.append(assistant_entry)
    result["message_history"] = updated_history

    return result


# ---------------------------------------------------------------------------
# Gemini reflector
# ---------------------------------------------------------------------------

def reflect_with_gemini(
    transform_hypothesis: str,
    code: str,
    test_inputs: list,
    test_predictions: list,
    training_accuracy: str = "100% on training set",
    ambiguity_rationale: str = "",
    candidate_predictions: list[dict] | None = None,
    train_inputs: list | None = None,
    train_outputs: list | None = None,
    train_input_images: list[str] | None = None,
    train_output_images: list[str] | None = None,
    test_input_images: list[str] | None = None,
    test_prediction_images: list[str] | None = None,
    model_name: str = "gemini-2.5-pro-preview-06-05",
    thinking_effort: str = "high",
    code_execution: bool = False,
    emit: Optional[Callable] = None,
    stream_emit: Optional[Callable] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    message_history: list[dict] | None = None,
    reviewer_response: str = "",
) -> dict:
    """Run independent reflection using Gemini.

    Args:
        Same as reflect_with_claude (minus Claude-specific params).

    Returns:
        dict with keys: verdict, confidence, findings, concerns, thinking, response, raw
    """
    if emit is None:
        def emit(msg): pass
    if should_stop is None:
        def should_stop() -> bool:
            return False

    from google.genai import types

    client = _get_gemini_client()
    system_prompt = _load_reflector_prompt(code_execution=code_execution)

    is_followup = bool(message_history)

    # -- Build multimodal parts for the CURRENT turn's user message --------
    parts = []

    normalized_candidates = list(candidate_predictions or [])
    if not normalized_candidates:
        normalized_candidates = [
            {
                "index": i,
                "candidates": [pred] if pred is not None else [],
                "candidate_images": [],
            }
            for i, pred in enumerate(test_predictions or [])
        ]

    if not is_followup:
        # ── Turn 1: full puzzle data + submission ─────────────────────
        train_input_images = train_input_images or []
        train_output_images = train_output_images or []
        test_input_images = test_input_images or []
        # 1. Training examples with images
        if train_inputs and train_outputs:
            parts.append(types.Part.from_text(
                text="## Training Input/Output Pairs\n\nStudy these examples first to form your own understanding of the transformation.\n"))
            for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
                parts.append(types.Part.from_text(text=f"### Training Example {i}\n**Input Grid:** {inp}"))
                if i < len(train_input_images) and train_input_images[i]:
                    img_bytes = base64.b64decode(train_input_images[i])
                    parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
                parts.append(types.Part.from_text(text=f"**Expected Output Grid:** {out}"))
                if i < len(train_output_images) and train_output_images[i]:
                    img_bytes = base64.b64decode(train_output_images[i])
                    parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        # 2. Test inputs with images (no predictions yet — let reflector form own view)
        parts.append(types.Part.from_text(text="## Test Inputs\n"))
        for i, inp in enumerate(test_inputs):
            parts.append(types.Part.from_text(text=f"### Test Example {i}\n**Input Grid:** {inp}"))
            if i < len(test_input_images) and test_input_images[i]:
                img_bytes = base64.b64decode(test_input_images[i])
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        submission_parts_list = ["## Solver Submission\n\n"]
        if training_accuracy:
            submission_parts_list.append(f"**Training Accuracy:** {training_accuracy}\n\n")
        submission_parts_list.append(
            f"**Hypothesis:** {transform_hypothesis or '(No hypothesis provided)'}\n\n"
            f"**solve() Code:**\n```python\n{code or '(No code provided)'}\n```"
        )
        if ambiguity_rationale:
            submission_parts_list.append(f"\n\n**Ambiguity Description:** {ambiguity_rationale}")
        if reviewer_response:
            submission_parts_list.append(f"\n\n**Response to Reviewer:** {reviewer_response}")
        submission_text = "".join(submission_parts_list)
        parts.append(types.Part.from_text(text=submission_text))
    else:
        # ── Turn >= 2: submission-only with anti-anchoring preamble ───
        parts.append(types.Part.from_text(text=_FOLLOWUP_PREAMBLE))

        submission_parts_list = ["## Solver Submission\n\n"]
        if training_accuracy:
            submission_parts_list.append(f"**Training Accuracy:** {training_accuracy}\n\n")
        submission_parts_list.append(
            f"**Hypothesis:** {transform_hypothesis or '(No hypothesis provided)'}\n\n"
            f"**solve() Code:**\n```python\n{code or '(No code provided)'}\n```"
        )
        if ambiguity_rationale:
            submission_parts_list.append(f"\n\n**Ambiguity Description:** {ambiguity_rationale}")
        if reviewer_response:
            submission_parts_list.append(f"\n\n**Response to Reviewer:** {reviewer_response}")
        submission_text = "".join(submission_parts_list)
        parts.append(types.Part.from_text(text=submission_text))

    # Candidate outputs (always included — these change each turn)
    parts.append(types.Part.from_text(text="## Candidate Outputs By Test Example\n"))
    for row in normalized_candidates:
        test_index = row.get("index", "?")
        parts.append(types.Part.from_text(text=f"### Test Example {test_index}"))
        if row.get("error"):
            parts.append(types.Part.from_text(text=f"Error: {row.get('error')}"))
            continue
        candidate_images = row.get("candidate_images", []) or []
        for candidate_idx, pred in enumerate(row.get("candidates", []) or [], start=1):
            parts.append(types.Part.from_text(text=f"{_ordinal(candidate_idx)} candidate: {pred}"))
            image_offset = candidate_idx - 1
            if image_offset < len(candidate_images) and candidate_images[image_offset]:
                img_bytes = base64.b64decode(candidate_images[image_offset])
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    parts.append(types.Part.from_text(
        text="---\nPlease perform your independent review following the guidelines in your system prompt."
    ))

    # -- Build contents list with conversation history ---------------------
    # Gemini implicit caching automatically caches matching prefixes (75%
    # discount), so a stable system_instruction + history prefix gets cached
    # without any explicit cache management code.
    if is_followup:
        contents = []
        for turn in message_history:
            if turn["role"] == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=turn["content"])],
                ))
            else:
                # Model turn: include thinking part if available
                model_parts = []
                if turn.get("thinking"):
                    model_parts.append(types.Part(
                        text=turn["thinking"],
                        thought=True,
                        thought_signature=turn.get("thinking_signature"),
                    ))
                model_parts.append(types.Part.from_text(text=turn["content"]))
                contents.append(types.Content(role="model", parts=model_parts))
        contents.append(types.Content(role="user", parts=parts))
    else:
        contents = [types.Content(role="user", parts=parts)]

    # -- Text-only snapshot of current user content for history storage ----
    if is_followup:
        user_text_for_history = _build_reflector_followup_message(
            transform_hypothesis=transform_hypothesis,
            code=code,
            test_predictions=test_predictions,
            training_accuracy=training_accuracy,
            ambiguity_rationale=ambiguity_rationale,
            candidate_predictions=candidate_predictions,
            reviewer_response=reviewer_response,
        )
    else:
        user_text_for_history = _build_reflector_user_message(
            transform_hypothesis=transform_hypothesis,
            code=code,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            test_inputs=test_inputs,
            test_predictions=test_predictions,
            training_accuracy=training_accuracy,
            ambiguity_rationale=ambiguity_rationale,
            candidate_predictions=candidate_predictions,
            reviewer_response=reviewer_response,
        )

    # Build thinking config
    _thinking_config = types.ThinkingConfig(
        thinking_level={
            "low": "low", "medium": "medium", "high": "high", "max": "high",
        }.get(thinking_effort, "high"),
        include_thoughts=True,
    )
    _tools = [types.Tool(code_execution=types.ToolCodeExecution())] if code_execution else []
    config = types.GenerateContentConfig(
        thinking_config=_thinking_config,
        system_instruction=[types.Part.from_text(text=system_prompt)],
        tools=_tools,
    )

    emit("   🤖 Calling Gemini reflector (streaming)...")

    # Import EventType for streaming emissions
    try:
        from .events import EventType, OrchestratorEvent
    except ImportError:
        from events import EventType, OrchestratorEvent

    _MAX_RETRIES = 2
    _RETRY_DELAYS = [30, 0]  # seconds between retries (only 1 retry)

    thinking_text = ""
    response_text = ""
    final_response = None

    for _attempt in range(_MAX_RETRIES):
        try:
            stream = client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config,
            )

            thinking_text = ""
            thinking_signature = None  # Gemini thought signature for multi-turn replay
            response_text = ""
            final_response = None

            for chunk in stream:
                if should_stop():
                    return {
                        "verdict": "ERROR",
                        "confidence": 0,
                        "findings": [],
                        "concerns": ["Reflection interrupted by stop request"],
                        "thinking": thinking_text,
                        "response": response_text,
                        "raw": final_response,
                    }
                final_response = chunk
                if chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        # Capture thought signature even on empty-text parts
                        if getattr(part, "thought_signature", None):
                            thinking_signature = part.thought_signature
                        if part.thought and part.text:
                            delta = part.text
                            thinking_text += delta
                            if stream_emit:
                                try:
                                    stream_emit(EventType.THINKING, delta, {"reflector": True})
                                except Exception:
                                    pass
                        elif part.text:
                            delta = part.text
                            response_text += delta
                            if stream_emit:
                                try:
                                    stream_emit(EventType.TEXT, delta, {"reflector": True})
                                except Exception:
                                    pass
            break  # success — exit retry loop

        except Exception as e:
            err_str = str(e)
            is_503 = "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str
            if is_503 and _attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[_attempt]
                emit(f"   ⚠️ Gemini 503 UNAVAILABLE (attempt {_attempt + 1}/{_MAX_RETRIES}). Retrying in {delay}s...")
                time.sleep(delay)
                continue
            return {
                "verdict": "ERROR",
                "confidence": 0,
                "findings": [],
                "concerns": [f"API error: {e}"],
                "thinking": "",
                "response": f"Error calling Gemini: {e}",
                "raw": None,
            }

    result = _parse_reflector_response(response_text, thinking_text, final_response)
    result["thinking_signature"] = thinking_signature

    # Extract token usage from Gemini usage_metadata
    if final_response is not None:
        try:
            um = final_response.usage_metadata
            if um is not None:
                result["usage_input_tokens"] = int(getattr(um, "prompt_token_count", 0) or 0)
                result["usage_thinking_tokens"] = int(getattr(um, "thoughts_token_count", 0) or 0)
                result["usage_output_tokens"] = int(getattr(um, "candidates_token_count", 0) or 0)
                result["usage_total_tokens"] = (
                    result["usage_input_tokens"]
                    + result["usage_thinking_tokens"]
                    + result["usage_output_tokens"]
                )
        except Exception:
            pass

    # Store updated conversation history (include thinking for proper
    # multi-turn replay — Gemini uses thought=True parts in model turns).
    updated_history = list(message_history or [])
    updated_history.append({"role": "user", "content": user_text_for_history})
    assistant_entry: dict[str, str] = {"role": "assistant", "content": response_text}
    assistant_entry["provider"] = "gemini"
    if thinking_text:
        assistant_entry["thinking"] = thinking_text
    if thinking_signature:
        assistant_entry["thinking_signature"] = thinking_signature
    updated_history.append(assistant_entry)
    result["message_history"] = updated_history

    return result


# ---------------------------------------------------------------------------
# Response parser (shared)
# ---------------------------------------------------------------------------

def _parse_reflector_response(response_text: str, thinking_text: str, raw_response) -> dict:
    """Extract the VERDICT from the reflector's response.

    Everything else (findings, concerns, ambiguity analysis, repair guidance)
    is left as freeform markdown in the response text and quoted verbatim
    when forwarded to the solver agent.
    """
    verdict = "UNKNOWN"

    lines = response_text.split("\n")
    for i, line in enumerate(lines):
        header_text = line.strip().lstrip("#").replace("**", "").strip()
        upper = header_text.upper()
        if upper.startswith("VERDICT:"):
            val = header_text.split(":", 1)[1].strip().upper()
            # If the value after "VERDICT:" is empty (e.g. "**Verdict:**" header),
            # check the next non-empty line for the actual verdict.
            if not val:
                for j in range(i + 1, min(i + 4, len(lines))):
                    next_text = lines[j].strip().replace("**", "").strip().upper()
                    if next_text:
                        val = next_text
                        break
            if "EXPAND_CANDIDATES" in val or "EXPAND_BRANCHES" in val:
                verdict = "EXPAND_CANDIDATES"
                break
            elif "APPROVE" in val:
                verdict = "APPROVE"
                break
            elif "REJECT" in val:
                verdict = "REJECT"
                break
            # Value doesn't contain a known verdict (e.g. "Verdict: The submission is correct.")
            # Don't break — keep scanning for the actual VERDICT: APPROVE/REJECT line.

    return {
        "verdict": verdict,
        "thinking": thinking_text,
        "response": response_text,
        "raw": raw_response,
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def run_independent_reflection(
    transform_hypothesis: str,
    code: str,
    test_inputs: list,
    test_predictions: list,
    training_accuracy: str = "100% on training set",
    ambiguity_rationale: str = "",
    candidate_predictions: list[dict] | None = None,
    train_inputs: list | None = None,
    train_outputs: list | None = None,
    train_input_images: list[str] | None = None,
    train_output_images: list[str] | None = None,
    test_input_images: list[str] | None = None,
    test_prediction_images: list[str] | None = None,
    reflector_provider: str = "claude",
    reflector_model: str | None = None,
    reflector_thinking_effort: str = "high",
    reflector_code_execution: bool = False,
    emit: Optional[Callable] = None,
    stream_emit: Optional[Callable] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    message_history: list[dict] | None = None,
    reviewer_response: str = "",
) -> dict:
    """Run independent reflection using the configured provider.

    Args:
        reflector_provider: "claude" or "gemini"
        reflector_model: Model name override. Defaults per provider.
        reflector_thinking_effort: Thinking effort level ("low", "medium", "high", "max").
        Other args: passed through to backend-specific function.

    Returns:
        dict with verdict, confidence, findings, concerns, thinking, response, raw
    """
    if reflector_provider == "gemini":
        model = reflector_model or "gemini-3.1-pro-preview"
        return reflect_with_gemini(
            transform_hypothesis=transform_hypothesis,
            code=code,
            test_inputs=test_inputs,
            test_predictions=test_predictions,
            training_accuracy=training_accuracy,
            ambiguity_rationale=ambiguity_rationale,
            candidate_predictions=candidate_predictions,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            train_input_images=train_input_images,
            train_output_images=train_output_images,
            test_input_images=test_input_images,
            test_prediction_images=test_prediction_images,
            model_name=model,
            thinking_effort=reflector_thinking_effort,
            code_execution=reflector_code_execution,
            emit=emit,
            stream_emit=stream_emit,
            should_stop=should_stop,
            message_history=message_history,
            reviewer_response=reviewer_response,
        )
    else:
        model = reflector_model or "claude-opus-4-6"
        return reflect_with_claude(
            transform_hypothesis=transform_hypothesis,
            code=code,
            test_inputs=test_inputs,
            test_predictions=test_predictions,
            training_accuracy=training_accuracy,
            ambiguity_rationale=ambiguity_rationale,
            candidate_predictions=candidate_predictions,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
            train_input_images=train_input_images,
            train_output_images=train_output_images,
            test_input_images=test_input_images,
            test_prediction_images=test_prediction_images,
            model_name=model,
            thinking_effort=reflector_thinking_effort,
            emit=emit,
            stream_emit=stream_emit,
            should_stop=should_stop,
            message_history=message_history,
            reviewer_response=reviewer_response,
        )
