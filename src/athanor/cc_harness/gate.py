"""The verification gate — the harness's only enforcement point.

Claude Code owns the agent loop in this variant, so everything Athanor's
orchestrator enforced by construction has to be enforced here instead:

============================  ==================================================
Athanor orchestrator          CC harness gate
============================  ==================================================
tool ordering forces          ``submit`` refuses code that changed without a
``submit_transform_            correspondingly updated hypothesis
hypothesis`` first
``execute_python_solution``   ``submit`` consumes one budgeted iteration and
counts as one iteration       appends an immutable record under ``.athanor/``
reflection prompt injected    the failure report ends with the reflection
into the conversation         directive (see :mod:`.reporting`)
test-generalization           ``accept`` requires ``solution/audit.md`` with an
self-audit turn               explicit CONFIDENCE / DECISION
best-effort prompt near       the gate lifts the train-100% requirement over the
budget exhaustion             trailing iterations
============================  ==================================================

Exploration is untouched by all of this: running ``python explore/whatever.py``
costs nothing and is the intended way to do almost all of the work.

Usage (from inside a run workspace)::

    python gate.py status
    python gate.py submit
    python gate.py accept
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate import detect_hardcoding, run_solution_isolated
from .reporting import format_status, format_submission_report

STATE_DIR = ".athanor"
STATE_FILE = "state.json"
INVARIANTS_FILE = "invariants.jsonl"
EVENTS_FILE = "events.jsonl"
FINAL_FILE = "final.json"

HYPOTHESIS_PATH = "solution/hypothesis.md"
CODE_PATH = "solution/solve.py"
AUDIT_PATH = "solution/audit.md"
NOTES_PATH = "NOTES.md"
TASK_PATH = "task/task.json"

NOTES_TAIL_CHARS = 3000


class GateError(Exception):
    """A refusal: the precondition failed, so no iteration was consumed."""


# ── workspace plumbing ───────────────────────────────────────────────────────

def find_workspace(start: Path | None = None) -> Path:
    """Walk up from ``start`` to the directory holding ``.athanor/state.json``."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / STATE_DIR / STATE_FILE).is_file():
            return candidate
    raise GateError(
        "Not inside an Athanor CC run workspace (no .athanor/state.json found). "
        "Run this from the workspace root."
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def load_state(workspace: Path) -> dict[str, Any]:
    return json.loads((workspace / STATE_DIR / STATE_FILE).read_text(encoding="utf-8"))


def save_state(workspace: Path, state: dict[str, Any]) -> None:
    path = workspace / STATE_DIR / STATE_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_task(workspace: Path) -> dict[str, Any]:
    return json.loads((workspace / TASK_PATH).read_text(encoding="utf-8"))


def load_invariants(workspace: Path) -> list[dict[str, Any]]:
    path = workspace / STATE_DIR / INVARIANTS_FILE
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Later verifications of the same claim supersede earlier ones.
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        deduped[str(entry.get("claim"))] = entry
    return list(deduped.values())


def append_event(workspace: Path, event: dict[str, Any]) -> None:
    path = workspace / STATE_DIR / EVENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), **event}) + "\n")


def _read_text(workspace: Path, relative: str) -> str:
    path = workspace / relative
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def best_effort_active(state: dict[str, Any], *, used: int | None = None) -> bool:
    """True once the trailing best-effort window has been entered."""
    used = len(state.get("iterations") or []) if used is None else used
    remaining = int(state.get("max_iterations") or 0) - used
    return remaining <= int(state.get("best_effort_iterations") or 0)


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_status(workspace: Path) -> tuple[str, int]:
    state = load_state(workspace)
    iterations = state.get("iterations") or []
    hypothesis = ""
    if iterations:
        last = iterations[-1]
        stored = workspace / STATE_DIR / "iterations" / str(last.get("iteration")) / "hypothesis.md"
        if stored.is_file():
            hypothesis = stored.read_text(encoding="utf-8")
    if not hypothesis:
        hypothesis = _read_text(workspace, HYPOTHESIS_PATH)

    notes = _read_text(workspace, NOTES_PATH)
    if len(notes) > NOTES_TAIL_CHARS:
        notes = "… (earlier entries omitted) …\n" + notes[-NOTES_TAIL_CHARS:]

    report = format_status(
        state=state,
        invariants=load_invariants(workspace),
        hypothesis=hypothesis,
        notes_excerpt=notes,
    )
    append_event(workspace, {"command": "status"})
    return report, 0


def cmd_submit(workspace: Path) -> tuple[str, int]:
    state = load_state(workspace)

    if state.get("accepted"):
        raise GateError("This run has already been accepted. No further submissions are recorded.")

    used = len(state.get("iterations") or [])
    max_iterations = int(state.get("max_iterations") or 0)
    if used >= max_iterations:
        raise GateError(
            f"Iteration budget exhausted ({used}/{max_iterations}). "
            "Write solution/audit.md and run `python gate.py accept` on your best submission."
        )

    hypothesis = _read_text(workspace, HYPOTHESIS_PATH).strip()
    code = _read_text(workspace, CODE_PATH)

    if not hypothesis:
        raise GateError(
            f"{HYPOTHESIS_PATH} is missing or empty. The hypothesis is a first-class artifact: "
            "write the rule in prose before you submit the code that implements it."
        )
    minimum = int(state.get("min_hypothesis_chars") or 0)
    if len(hypothesis) < minimum:
        raise GateError(
            f"{HYPOTHESIS_PATH} is {len(hypothesis)} characters; at least {minimum} are required. "
            "It has to be complete enough that a programmer who has never seen this puzzle could "
            "reimplement solve() from it alone: high-level summary, step-by-step algorithm, every "
            "edge case and conditional, why the rule generalizes, and — if a genuine ambiguity "
            "survives — how candidate 1 differs from candidate 2."
        )
    if not code.strip():
        raise GateError(f"{CODE_PATH} is missing or empty.")
    if "def solve" not in code:
        raise GateError(f"{CODE_PATH} does not define solve(grid).")

    hypothesis_sha = _sha(hypothesis)
    code_sha = _sha(code)
    last_hypothesis_sha = state.get("last_hypothesis_sha") or ""
    last_code_sha = state.get("last_code_sha") or ""

    if used and code_sha == last_code_sha and hypothesis_sha == last_hypothesis_sha:
        raise GateError(
            f"Neither {HYPOTHESIS_PATH} nor {CODE_PATH} changed since iteration {used}. "
            "Resubmitting the same artifacts cannot produce a different result."
        )
    if used and code_sha != last_code_sha and hypothesis_sha == last_hypothesis_sha:
        raise GateError(
            f"{CODE_PATH} changed but {HYPOTHESIS_PATH} did not. Every code change is a change of "
            "rule, of implementation, or of edge-case handling — update the hypothesis to say which, "
            "so the artifact still describes the code it ships with. (No iteration was consumed.)"
        )

    task = load_task(workspace)
    train_samples = task.get("train") or []
    test_samples = task.get("test") or []

    started = time.time()
    evaluation = run_solution_isolated(
        code,
        train_samples,
        test_samples,
        max_candidates=int(state.get("max_test_predictions") or 2),
        timeout_seconds=float(state.get("solve_timeout_s") or 60.0),
    )
    elapsed = time.time() - started

    findings = detect_hardcoding(code, train_samples)
    iteration = used + 1

    record = {
        "iteration": iteration,
        "at": _now(),
        "hypothesis_sha": hypothesis_sha,
        "hypothesis_chars": len(hypothesis),
        "code_sha": code_sha,
        "code_chars": len(code),
        "status": evaluation.get("status"),
        "error": evaluation.get("error") or "",
        "timed_out": bool(evaluation.get("timed_out")),
        "train_correct": int(evaluation.get("train_correct") or 0),
        "train_total": int(evaluation.get("train_total") or len(train_samples)),
        "train_pixel_accuracy": float(evaluation.get("train_pixel_accuracy") or 0.0),
        "all_train_correct": bool(evaluation.get("all_train_correct")),
        "multi_candidate_on_train": bool(evaluation.get("multi_candidate_on_train")),
        "num_test_candidates": int(evaluation.get("num_test_candidates") or 0),
        "hardcoding_findings": findings,
        "elapsed_s": round(elapsed, 3),
    }

    iteration_dir = workspace / STATE_DIR / "iterations" / str(iteration)
    iteration_dir.mkdir(parents=True, exist_ok=True)
    (iteration_dir / "hypothesis.md").write_text(hypothesis, encoding="utf-8")
    (iteration_dir / "solve.py").write_text(code, encoding="utf-8")
    (iteration_dir / "predictions.json").write_text(
        json.dumps(
            {"test": [{"index": r.get("index"), "candidates": r.get("candidates") or [],
                       "error": r.get("error")} for r in evaluation.get("test", [])]},
            indent=2,
        ),
        encoding="utf-8",
    )

    state.setdefault("iterations", []).append(record)
    state["last_hypothesis_sha"] = hypothesis_sha
    state["last_code_sha"] = code_sha
    save_state(workspace, state)

    report = format_submission_report(
        iteration=iteration,
        max_iterations=max_iterations,
        evaluation=evaluation,
        train_samples=train_samples,
        hardcoding_findings=findings,
        best_effort_active=best_effort_active(state, used=iteration),
    )
    (iteration_dir / "report.txt").write_text(report, encoding="utf-8")

    append_event(
        workspace,
        {
            "command": "submit",
            "iteration": iteration,
            "all_train_correct": record["all_train_correct"],
            "train_correct": record["train_correct"],
        },
    )
    return report, 0


_DECISION_RE = re.compile(r"^\s*DECISION\s*:\s*(ACCEPT|RETRY)\b", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"^\s*CONFIDENCE\s*:\s*([1-5])\b", re.IGNORECASE | re.MULTILINE)


def cmd_accept(workspace: Path) -> tuple[str, int]:
    state = load_state(workspace)
    if state.get("accepted"):
        return "This run was already accepted. Nothing more to do — stop here.", 0

    iterations = state.get("iterations") or []
    if not iterations:
        raise GateError("Nothing to accept: no submission has been made yet.")

    last = iterations[-1]
    if last.get("status") != "ok":
        raise GateError(
            f"Iteration {last.get('iteration')} failed to run, so there is nothing to accept. "
            "Fix solve.py and submit again."
        )

    lenient = best_effort_active(state)
    if not last.get("all_train_correct") and not lenient:
        raise GateError(
            f"Iteration {last.get('iteration')} covers "
            f"{last.get('train_correct')}/{last.get('train_total')} training examples. "
            "A rule that cannot reproduce the examples you can check is not a rule you can trust "
            "on the ones you cannot. Keep iterating — the train-100% requirement is lifted only "
            "over the last "
            f"{state.get('best_effort_iterations')} iteration(s) of the budget."
        )

    audit = _read_text(workspace, AUDIT_PATH).strip()
    if not audit:
        raise GateError(
            f"{AUDIT_PATH} is missing. Passing training is not evidence of generalization — "
            "write the audit (CONFIDENCE / DECISION / REASONS) before accepting."
        )
    decision_match = _DECISION_RE.search(audit)
    if not decision_match:
        raise GateError(
            f"{AUDIT_PATH} has no `DECISION: ACCEPT` or `DECISION: RETRY` line. "
            "The audit needs an explicit verdict."
        )
    if decision_match.group(1).upper() == "RETRY":
        raise GateError(
            "Your own audit says DECISION: RETRY. Act on it — the concerns you just wrote down "
            "are the most informative signal you have. Revise and submit again."
        )

    confidence_match = _CONFIDENCE_RE.search(audit)
    iteration_dir = workspace / STATE_DIR / "iterations" / str(last.get("iteration"))
    predictions = json.loads((iteration_dir / "predictions.json").read_text(encoding="utf-8"))

    final = {
        "task_id": state.get("task_id"),
        "accepted_at": _now(),
        "iteration": last.get("iteration"),
        "iterations_used": len(iterations),
        "max_iterations": state.get("max_iterations"),
        "all_train_correct": last.get("all_train_correct"),
        "train_correct": last.get("train_correct"),
        "train_total": last.get("train_total"),
        "train_pixel_accuracy": last.get("train_pixel_accuracy"),
        "best_effort": bool(lenient and not last.get("all_train_correct")),
        "confidence": int(confidence_match.group(1)) if confidence_match else None,
        "hypothesis": (iteration_dir / "hypothesis.md").read_text(encoding="utf-8"),
        "code": (iteration_dir / "solve.py").read_text(encoding="utf-8"),
        "audit": audit,
        "test": predictions.get("test") or [],
        "hardcoding_findings": last.get("hardcoding_findings") or [],
        "verified_invariants": load_invariants(workspace),
    }
    (workspace / STATE_DIR / FINAL_FILE).write_text(json.dumps(final, indent=2), encoding="utf-8")

    state["accepted"] = {
        "iteration": last.get("iteration"),
        "at": final["accepted_at"],
        "best_effort": final["best_effort"],
        "confidence": final["confidence"],
    }
    save_state(workspace, state)
    append_event(workspace, {"command": "accept", "iteration": last.get("iteration")})

    lines = [
        "RUN COMPLETE — solution accepted.",
        f"  iteration      : {final['iteration']} of {final['max_iterations']}",
        f"  training       : {final['train_correct']}/{final['train_total']}"
        + ("  (best-effort acceptance)" if final["best_effort"] else ""),
        f"  confidence     : {final['confidence'] if final['confidence'] is not None else 'unstated'}",
        f"  test candidates: "
        + ", ".join(
            f"test {row.get('index')}: {len(row.get('candidates') or [])}" for row in final["test"]
        ),
        "",
        "Your predictions are recorded. Stop now — do not submit again, and do not "
        "keep working on this puzzle.",
    ]
    return "\n".join(lines), 0


# ── entry point ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gate.py",
        description="Athanor CC harness verification gate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="Print the distilled research state (use this after a compaction).")
    sub.add_parser("submit", help="Record a formal iteration: hypothesis + solve() vs. the training pairs.")
    sub.add_parser("accept", help="Finalize the run using the last submission and solution/audit.md.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        workspace = find_workspace()
        handler = {"status": cmd_status, "submit": cmd_submit, "accept": cmd_accept}[args.command]
        report, code = handler(workspace)
    except GateError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface harness bugs to the agent, don't hide them
        print(f"GATE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
