"""Run configuration for the Claude Code harness variant."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

#: Built-in tools the solver agent is allowed to use.
#:
#: Deliberately excludes WebSearch/WebFetch (ARC requires no external knowledge,
#: and network answers would contaminate the benchmark) and the Agent/Task tools
#: (this variant is single-agent by design — no reviewer, no sub-solvers).
DEFAULT_TOOLS: tuple[str, ...] = ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "TodoWrite")

#: Tools denied even if a settings file or future default would grant them.
DEFAULT_DISALLOWED_TOOLS: tuple[str, ...] = ("WebSearch", "WebFetch", "Task", "Agent")


@dataclass
class CCRunConfig:
    """Everything that shapes one CC-harness solve attempt.

    Persisted verbatim into each run's ``result.json`` so a run can be
    reproduced from its own record, mirroring how Athanor checkpoints embed
    ``state.config``.
    """

    # ── model / agent loop ───────────────────────────────────────────────
    model: str = "opus"
    effort: str = "high"
    """Claude Code reasoning effort: low | medium | high | xhigh | max."""

    # ── budgets ──────────────────────────────────────────────────────────
    max_iterations: int = 12
    """Formal `gate.py submit` calls allowed. Exploration is unbudgeted."""
    best_effort_iterations: int = 2
    """Trailing iterations during which the train-100% requirement is lifted."""
    max_budget_usd: float | None = None
    """Passed to `claude --max-budget-usd`; None leaves spend uncapped."""
    wall_clock_timeout_s: float = 3600.0
    solve_timeout_s: float = 60.0
    """Per-submission ceiling on running solve() across all inputs."""

    # ── task presentation ────────────────────────────────────────────────
    max_test_predictions: int = 2
    visual: bool = True
    """Render PNGs of every grid so the agent can Read them as images."""
    inline_grids: bool = True
    """Inline the text grids in the opening prompt (vs. file reference only)."""
    min_hypothesis_chars: int = 300
    """Floor enforced by the gate, standing in for 'be exhaustive'."""

    # ── harness isolation ────────────────────────────────────────────────
    permission_mode: str = "acceptEdits"
    """Claude Code permission mode.

    `acceptEdits` auto-approves file writes and common filesystem commands;
    everything else the solver needs is covered by `tools` via `--allowedTools`.

    Not `bypassPermissions`: Claude Code maps it to
    `--dangerously-skip-permissions`, which the CLI refuses outright when the
    process is running as root — the normal case for a containerised research
    harness. `dontAsk` is the stricter alternative for locked-down runs: it
    denies anything not explicitly allowed rather than prompting.
    """
    tools: tuple[str, ...] = DEFAULT_TOOLS
    disallowed_tools: tuple[str, ...] = DEFAULT_DISALLOWED_TOOLS
    stable_system_prompt: bool = True
    """Pass `--exclude-dynamic-system-prompt-sections`.

    Every task runs in its own workspace, so cwd differs per task — and cwd sits
    in Claude Code's default system prompt. That changes the cached prefix on
    every task and defeats cross-task prompt-cache reuse entirely: measured
    directly, a second workspace wrote byte-identical cache to the first rather
    than reading it.

    Keep the size honest. What this recovers is the system-prompt prefix across
    tasks, a few thousand tokens; the bulk of a run's cache traffic is
    within-task, as each turn writes the growing conversation. The flagship's
    own figure puts batch amortisation at $5.80 over 119 tasks — about $0.05 a
    task. This is a correctness fix to a mechanism that was fully broken, not a
    material cost lever.

    The flag moves cwd, env info, memory paths and git status into the first
    user message instead. It applies only alongside the default system prompt,
    which is what this harness uses (`--append-system-prompt`, not
    `--system-prompt`).
    """

    setting_sources: str = "project"
    """Which Claude Code settings sources to load. 'project' keeps the
    workspace's own .claude/settings.json (the compaction hook) while ignoring
    whatever the host user has configured."""
    bare: bool = False
    """Run `claude --bare` for maximum reproducibility. Requires
    ANTHROPIC_API_KEY — bare mode skips OAuth and keychain reads."""
    extra_cli_args: tuple[str, ...] = ()

    # ── bookkeeping ──────────────────────────────────────────────────────
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_iterations = max(1, int(self.max_iterations))
        self.best_effort_iterations = max(0, min(int(self.best_effort_iterations), self.max_iterations))
        self.max_test_predictions = max(1, min(2, int(self.max_test_predictions)))
        self.tools = tuple(self.tools)
        self.disallowed_tools = tuple(self.disallowed_tools)
        self.extra_cli_args = tuple(self.extra_cli_args)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CCRunConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})
