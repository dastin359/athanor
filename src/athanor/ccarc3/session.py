"""Workspace construction and session launch for CCARC3.

The division of labour mirrors CCARC (ARC-AGI-2): **Claude Code owns the agent
loop**, and this harness supplies a workspace, a toolkit, and a gate. There is
no reviewer and no orchestration — the solver decides what to do next, and the
harness only refuses the moves that are known to destroy a run.

What a workspace contains:

===================  ======================================================
``CLAUDE.md``        how to drive the game and what is available
``DOCTRINE.md``      the measured findings, several counterintuitive
``session.py``       a client and gate pre-wired to this game
``trace.jsonl``      every action and its frames, written unprompted
``rules.json``       the rule book the gate requires at level boundaries
``notes/``           the solver's own scratch space
===================  ======================================================
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cc_harness.config import (
    DEFAULT_ALLOWED_TOOLS,
    DEFAULT_DISALLOWED_TOOLS,
)
from ..cc_harness.runner import resolve_permission_mode
from .client import GameInfo, list_games

__all__ = [
    "Ccarc3Config",
    "Workspace",
    "build_workspace",
    "build_cli_args",
    "run_game",
    "collect_outcome",
    "snapshot_scorecard",
]

ASSETS = Path(__file__).parent / "assets"

# The operator's standing constraint for this project: Opus 5 at effort high,
# and no other model or effort, so runs stay comparable to one another.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"


@dataclass
class Ccarc3Config:
    """One solver run against one game."""

    game_id: str
    out_dir: Path = Path("runs/ccarc3")
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    level_budget_multiple: float = 5.0
    """Per-level cap as a multiple of that level's baseline — **the official rule**.

    ARC terminates an agent after 5n actions on a level with baseline n. Kept as
    the default so runs here are comparable to the published evaluation rather
    than to an arbitrary local choice.
    """

    budget_multiple: float = 4.0
    """Action cap as a multiple of the game's published baseline (§2.6).

    A flat cap cannot work: real games span 171 to 1843 baseline actions, so any
    single number either truncates the long games or wastes the short ones.
    """

    wall_clock_timeout_s: float = 7200.0
    permission_mode: str = "bypassPermissions"
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS
    """Pre-approved so a headless run never stalls on a permission prompt.

    ``--allowedTools`` grants permission as well as restricting the surface.
    Without it, ``acceptEdits`` approves file writes but *not* Bash, so every
    ``python -c ...`` the solver runs is denied and the run produces nothing.
    CCARC learned this the expensive way; these are its lists, imported rather
    than re-derived, because deriving them again is how this bug came back.
    """

    disallowed_tools: tuple[str, ...] = DEFAULT_DISALLOWED_TOOLS
    """Denied outright: research, network-by-another-door, escaping the run.

    ARC needs no external knowledge and a network answer would contaminate
    the benchmark. Everything else Claude Code ships stays available.
    """
    api_key: str | None = None
    fresh: bool = False
    """Discard any existing trace and start the game over.

    The default is to *resume*. A container can be recycled mid-run, and the
    client already persists its session, so relaunching against the same
    out_dir continues the same game rather than paying for the first N actions
    twice. Making that the default is deliberate; making it silent would not
    be, hence the flag and the log line.
    """

    extra_cli_args: tuple[str, ...] = ()


@dataclass
class Workspace:
    root: Path
    config: Ccarc3Config
    info: GameInfo
    initial_prompt: str = ""
    env: dict[str, str] = field(default_factory=dict)
    resumed: bool = False

    @property
    def trace_path(self) -> Path:
        return self.root / "trace.jsonl"

    @property
    def rules_path(self) -> Path:
        return self.root / "rules.json"


SESSION_TEMPLATE = '''\
"""Pre-wired client and gate for {game_id}. Import this; do not rebuild it.

    from session import client, gate, arc

    client.reset()
    client.act(1)
    print(client.status())

Every action is recorded to trace.jsonl automatically. The gate holds the first
action of a new level until you call ``gate.acknowledge(...)``.
"""
from pathlib import Path

from athanor import ccarc3 as arc
from athanor.ccarc3 import ArcClient, GameInfo, LevelGate

HERE = Path(__file__).parent

INFO = GameInfo(
    game_id={game_id!r},
    title={title!r},
    tags={tags!r},
    baseline_actions={baseline!r},
)

gate = LevelGate(HERE / "rules.json")
client = ArcClient(
    {game_id!r},
    trace_path=HERE / "trace.jsonl",
    info=INFO,
    gate=gate,
    max_actions={budget},
    level_budget_multiple={level_budget_multiple!r},
)
client.open()

ACTION_BUDGET = {budget}
'''


def _claude_binary() -> str:
    return os.environ.get("CLAUDE_BINARY") or shutil.which("claude") or "claude"


def _supports_flag(flag: str) -> bool:
    try:
        out = subprocess.run(
            [_claude_binary(), "--help"], capture_output=True, text=True, timeout=30
        )
        return flag in (out.stdout + out.stderr)
    except (OSError, subprocess.SubprocessError):
        return False


def build_workspace(config: Ccarc3Config, info: GameInfo | None = None) -> Workspace:
    """Create the workspace for one game."""
    if info is None:
        matches = [g for g in list_games(config.api_key) if g.game_id == config.game_id]
        if not matches:
            raise ValueError(f"no such game: {config.game_id}")
        info = matches[0]

    root = Path(config.out_dir) / config.game_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes").mkdir(exist_ok=True)

    # result.json describes a *finished* run. Leaving the previous one in place
    # while a new one is in flight means `report` presents a stale outcome as
    # final, and anything watching the directory sees a run that has not started
    # as already complete. It is rewritten by collect_outcome at the end.
    (root / "result.json").unlink(missing_ok=True)

    trace = root / "trace.jsonl"
    if config.fresh:
        for stale in (trace, trace.with_suffix(".state.json"), root / "rules.json"):
            stale.unlink(missing_ok=True)
    resumed = trace.exists()

    budget = info.suggested_budget(config.budget_multiple)

    (root / "session.py").write_text(
        SESSION_TEMPLATE.format(
            game_id=info.game_id,
            title=info.title,
            tags=tuple(info.tags),
            baseline=tuple(info.baseline_actions),
            budget=budget,
            level_budget_multiple=config.level_budget_multiple,
        ),
        encoding="utf-8",
    )
    shutil.copy(ASSETS / "CCARC3_DOCTRINE.md", root / "DOCTRINE.md")
    (root / "CLAUDE.md").write_text(_workspace_claude_md(info, budget), encoding="utf-8")
    (root / "meta.json").write_text(
        json.dumps(
            {
                "game_id": info.game_id,
                "title": info.title,
                "tags": list(info.tags),
                "baseline_actions": list(info.baseline_actions),
                "levels": info.levels,
                "action_budget": budget,
                "model": config.model,
                "effort": config.effort,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    key = config.api_key or os.environ.get("ARC_API_KEY", "")
    if key:
        env["ARC_API_KEY"] = key
    src = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = f"{src}:{env.get('PYTHONPATH', '')}".rstrip(":")

    # Put the interpreter that actually has numpy first on PATH. The first live
    # run wasted turns on `ModuleNotFoundError: No module named 'numpy'` from
    # bare `python3`, then had to discover the venv path by trial. Making
    # `python3` resolve to the right thing removes the whole class of error
    # rather than documenting a way around it.
    venv_bin = src.parent / ".venv" / "bin"
    if (venv_bin / "python3").exists() or (venv_bin / "python").exists():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}".rstrip(":")

    return Workspace(
        root=root,
        config=config,
        info=info,
        initial_prompt=_initial_prompt(info, budget, resumed=resumed),
        env=env,
        resumed=resumed,
    )


def _workspace_claude_md(info: GameInfo, budget: int) -> str:
    return f"""\
# {info.game_id} — ARC-AGI-3

You are playing an unfamiliar video game through an API. Nobody has told you the
rules. Working them out **is** the task.

`DOCTRINE.md` is not background reading. Several of its points are the opposite
of what instinct suggests, and each one was learned by losing progress. Read it
before your first action.

## The game

| | |
|---|---|
| id | `{info.game_id}` |
| levels | {info.levels} |
| action types | {', '.join(info.tags) or 'not published — read `available_actions`'} |
| baseline actions per level | {list(info.baseline_actions)} |
| your action budget | **{budget}** |

The baseline is what a playthrough costs when the rules are *already known*. You
must also discover them, hence the larger budget. But if you are several times
over the baseline for the level you are on, that is evidence your hypothesis is
wrong — go re-explore rather than grind.

## Driving it

```python
from session import client, gate, arc

client.reset()                  # start
client.act(1)                   # ACTION1..5,7 take no arguments
client.act(6, x=10, y=20)       # ACTION6 is a click; x,y in [0,63]
client.status()                 # level, state, and pace against this level's baseline
client.pace()                   # {{level: (spent, baseline, ratio)}} for every level
client.transitions()            # everything recorded so far
```

**The game persists across commands.** Each `python -c ...` is a new process,
and the client resumes the same game, scorecard and action count from disk. You
do not need to hold one long-running script open, and you must not try to start
over — importing `session` again continues where you left off.

`client` refuses moves that are known to destroy runs — acting while dead, a
RESET immediately after a level advance, actions this game does not accept. A
refusal costs no budget and is telling you something.

## Analysing

Every action and its frames land in `trace.jsonl` without you asking. Ask
questions of it in code rather than reading frames by eye:

```python
ts = client.transitions()
arc.effective_actions(ts, level=client.level)  # {{'ACTION6': (0, 31)}} -> stop clicking
arc.diff(ts[-1].before, ts[-1].after)          # what just changed
arc.objects(ts[-1].after)                      # connected components
print(arc.render(ts[-1].after))                # one char per cell
```

**Look at the board when the question is about shape.** `arc.png()` writes the
frame as an image in the real palette, and you can open it with the Read tool
and see it:

```python
ts = client.transitions()
arc.png(ts[-1].after, "notes/now.png", scale=8)   # then Read notes/now.png
```

Corridors, enclosures, symmetry and "which thing moved" read instantly as a
picture and slowly as 64 lines of text. Use the image for layout and the text
for exact values and diffs — they are complementary, not alternatives.

Rules are checked three ways, never two:

```python
ts = client.transitions()
r = arc.Rule(name="...", applies=lambda t: ..., holds=lambda t: ..., scope="game")
arc.verify(r, ts)     # THIS level only. Can refute.
arc.survey(r, ts)     # every level. Reports where it holds. Cannot refute.
```

Stronger than predicates: write a forward model and check it against everything
recorded.

```python
arc.predict(step, client.transitions())   # step(before, action, params) -> board
```

A model that reproduces every recorded board exactly means you understand the
mechanics. Chase `perfect`, not accuracy — one wrong prediction is worth more
than fifty right ones, and `failures` says which transition to look at.

Routing costs more than verifying here — testing a belief is one action, but
walking a bad route is many:

```python
arc.shortest_path(step, start, goal)   # fewest actions, over your step function
arc.reachable(step, start)             # reachable at all, or did I misread the board?
```

`applies` and `holds` are separate on purpose. A rule that did not apply has not
failed, and treating it as failed is how correct knowledge gets thrown away.

## Level boundaries

When you complete a level you will be **refused** the next action until you
call:

```python
gate.acknowledge(
    "what level N established",
    mechanics=["game-scoped beliefs worth carrying forward"],
    refuted=["tested and found false"],
    untested=["never exercised — nothing known either way"],
)
```

Keep `refuted` and `untested` apart. Something you never tried is not something
you disproved, and filing it as a refutation makes you stop asking. It is the
same distinction the three-valued outcome makes: `NOT_APPLICABLE` is not
`VIOLATED`.

Carry mechanics, not rules. The next level shares this game's logic but not its
arrangement, so a concrete rule will often be false there while the idea behind
it still holds. Treat carried mechanics as priors about what to test first.

## Constraints

- No network beyond the game API. Do not search for solutions or read anything
  outside this workspace.
- Do not modify `trace.jsonl` — it is the record.
- `notes/` is yours.
"""


def _initial_prompt(info: GameInfo, budget: int, *, resumed: bool = False) -> str:
    if resumed:
        return (
            f"You are resuming an interrupted run of `{info.game_id}`. The game is "
            f"still open and your previous actions are recorded.\n\n"
            "Read DOCTRINE.md, then `from session import client, gate, arc`. Start "
            "with `client.status()` and `client.transitions()` to see where you are, "
            "and read `rules.json` for what the earlier session established — those "
            "mechanics were paid for and re-deriving them wastes budget you have "
            "already spent. Do not RESET to 'start clean'; that discards real "
            "progress.\n\n"
            f"Continue toward winning as many of the {info.levels} levels as you can, "
            f"within the {budget}-action budget you are already partway through."
        )
    return (
        f"Play `{info.game_id}` and win as many of its {info.levels} levels as you can, "
        f"within {budget} actions.\n\n"
        "Read DOCTRINE.md first — several of its points contradict what seems obvious, "
        "and each was learned the hard way.\n\n"
        "Then `from session import client, gate, arc`, RESET, and work out the rules. "
        "Write code to interrogate the trace rather than reading frames by eye. "
        "Dying is cheap and is a legitimate experiment; being confused is expensive."
    )


def build_cli_args(workspace: Workspace, *, system_prompt_file: Path | None = None) -> list[str]:
    config = workspace.config
    args = [
        _claude_binary(),
        "-p",
        workspace.initial_prompt,
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    if config.model:
        args += ["--model", config.model]
    if config.effort and _supports_flag("--effort"):
        args += ["--effort", config.effort]
    if config.permission_mode:
        # bypassPermissions maps to --dangerously-skip-permissions, which the
        # CLI refuses under root -- and a containerised harness is usually
        # root. The refusal arrives as a one-line stderr and an empty run,
        # which is exactly how the first launch here failed.
        args += ["--permission-mode", resolve_permission_mode(config.permission_mode)]
    if config.allowed_tools:
        args += ["--allowedTools", ",".join(config.allowed_tools)]
    if config.disallowed_tools:
        args += ["--disallowed-tools", ",".join(config.disallowed_tools)]
    if system_prompt_file and _supports_flag("--append-system-prompt-file"):
        args += ["--append-system-prompt-file", str(system_prompt_file)]
    args += list(config.extra_cli_args)
    return args


def run_game(config: Ccarc3Config, info: GameInfo | None = None) -> dict[str, Any]:
    """Build a workspace and run one solver session against one game."""
    ws = build_workspace(config, info)
    _record_resume_state(ws)
    args = build_cli_args(ws)
    stream = ws.root / "stream.jsonl"
    if stream.exists():
        # Opening with "w" destroyed run 1's stream when the resume started,
        # which removed the only record of how the handoff actually went -- and
        # the handoff is exactly what needed diagnosing. Keep each attempt.
        #
        # **Not gated on ``ws.resumed``.** A run killed before it wrote a single
        # action leaves a stream but no trace, so the next launch is not a resume
        # (``resumed = trace.exists()``) and the old gate let it overwrite the
        # only record of what that attempt spent. Found on `vc33`, killed two
        # seconds after launch by a quota stop: 21 KB of stream, no trace, and
        # relaunching would have billed its tokens to nobody. Any existing stream
        # is a previous attempt whatever the trace says, so rotate on existence.
        n = len(list(ws.root.glob("stream.*.jsonl"))) + 1
        stream.rename(ws.root / f"stream.{n}.jsonl")

    with stream.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            args,
            cwd=ws.root,
            env=ws.env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            code = proc.wait(timeout=config.wall_clock_timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            code, timed_out = -1, True

    return collect_outcome(ws, exit_code=code, timed_out=timed_out)


def _record_resume_state(ws: Workspace) -> None:
    """Snapshot what a resume inherited, before the solver can change it.

    A resume once preserved the ledger but not the game -- trace indices
    continued from 370 while the server replayed levels 0-5 on a fresh
    scorecard, costing 370 re-spent actions. The mechanism was not
    reconstructable afterwards because nothing recorded what the client
    restored at construction time. This does.
    """
    state = Path(ws.trace_path).with_suffix(".state.json")
    snapshot: dict[str, Any] = {
        "resumed": ws.resumed,
        "trace_lines": sum(1 for _ in ws.trace_path.open()) if ws.trace_path.exists() else 0,
        "state_file_present": state.exists(),
    }
    if state.exists():
        try:
            saved = json.loads(state.read_text(encoding="utf-8"))
            snapshot |= {
                "card_id": saved.get("card_id", ""),
                "level": saved.get("level"),
                "actions_used": saved.get("actions_used"),
                "cookies": len(saved.get("cookies") or []),
            }
        except (OSError, json.JSONDecodeError) as exc:
            snapshot["state_file_error"] = f"{type(exc).__name__}: {exc}"
    (ws.root / "resume_state.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )


def ledger_facts(trace_path: Path | str) -> dict[str, Any]:
    """Everything about a run that can be re-derived from its ledger.

    Split out from :func:`collect_outcome` so a *finished* run can be re-read
    later. `result.json` is a derived artefact, and a harness fix can make it
    wrong after the fact: when full resets became detectable, every stored
    `ls20` figure was still the pre-fix one — 860 actions, zero full resets, no
    playthrough split. Deriving from the trace on demand means a fix reaches
    history too.
    """
    from .ledger import load

    path = Path(trace_path)
    transitions = load(path) if path.exists() else []
    states = [t.state for t in transitions]

    # A full reset restarts the game inside the same trace, so the ledger holds
    # more than one playthrough and the two honest numbers diverge. `ls20`
    # reported 860 actions for a game that was won in 490; the other 370 were a
    # replay, and the correction had to be made by hand in the write-up. Report
    # both: the total is what the budget paid, the last playthrough is what the
    # result cost.
    #
    # The restarting action is counted *in* the final playthrough, because it
    # was billed. That is why this reads 490 where a per-level table of the same
    # run sums to 489 -- the reset belongs to no level.
    last_restart = max(
        (i for i, t in enumerate(transitions) if t.full_reset),
        default=0,
    )
    final = transitions[last_restart:]
    resets = sum(1 for t in transitions if t.full_reset)

    return {
        "levels_reached": max((t.level for t in transitions), default=0),
        "won": "WIN" in states,
        "actions_used": len(transitions),
        "deaths": sum(
            1
            for prev, cur in zip(["NOT_PLAYED", *states], states)
            if cur == "GAME_OVER" and prev != "GAME_OVER"
        ),
        "wasted_actions": sum(t.wasted for t in transitions),
        "full_resets": resets,
        "playthroughs": resets + 1,
        "actions_final_playthrough": len(final),
        "levels_reached_final_playthrough": max((t.level for t in final), default=0),
    }


def run_cost(stream_path: Path | str) -> dict[str, Any]:
    """Turns, cost and wall time, from the stream's final ``result`` event.

    The ledger says what a run *did*; this says what it cost to do it, and the
    two came apart in a way worth being able to see. `sb26` won eight levels in
    **35 tool calls** where `cd82` needed 83 for six, and `ls20` cost $13.15
    against `sb26`'s $3.04. Nothing in `result.json` reported any of that, so
    the comparison had to be made by hand from the stream twice.

    **Sums every attempt, not just the last.** ``run_game`` archives a previous
    stream to ``stream.N.jsonl`` when it resumes, while the trace — and so
    ``actions_used`` — carries across attempts. Reading only ``stream.jsonl``
    therefore reported a resumed run's actions in full against the *final*
    attempt's turns and cost, understating the bill by whatever the earlier
    attempts spent. `ls20` was resumed once.

    Returns an empty dict when no stream has a result event — a killed run has
    none, and that is not an error worth raising over.
    """
    path = Path(stream_path)
    attempts = sorted(path.parent.glob("stream.*.jsonl")) + [path]
    turns = cost = duration = 0
    found = False
    for attempt in attempts:
        if not attempt.exists():
            continue
        final: dict[str, Any] = {}
        for line in attempt.open(encoding="utf-8", errors="ignore"):
            if '"type":"result"' not in line and '"type": "result"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # `run_game` merges the solver's stderr into this file, so a line can
            # be valid JSON without being an object. `.get` on a str is an
            # AttributeError out of a function whose whole contract is "never
            # raise over a killed run".
            if isinstance(record, dict) and record.get("type") == "result":
                final = record
        if not final:
            continue
        found = True
        turns += final.get("num_turns") or 0
        cost += final.get("total_cost_usd") or 0.0
        duration += final.get("duration_ms") or 0
    if not found:
        return {}
    return {
        "turns": turns,
        "cost_usd": cost,
        "duration_s": round(duration / 1000) if duration else None,
        "attempts": sum(1 for a in attempts if a.exists()),
    }


def collect_outcome(ws: Workspace, *, exit_code: int, timed_out: bool) -> dict[str, Any]:
    """Read the run's result off disk — never from what the solver claims."""
    outcome = {
        "game_id": ws.info.game_id,
        "levels_total": ws.info.levels,
        "baseline_total": ws.info.baseline_total,
        **ledger_facts(ws.trace_path),
        **run_cost(ws.root / "stream.jsonl"),
        "exit_code": exit_code,
        "timed_out": timed_out,
    }
    if ws.rules_path.exists():
        book = json.loads(ws.rules_path.read_text(encoding="utf-8"))
        outcome["mechanics_recorded"] = len(book.get("verified", []))
        outcome["refutations_recorded"] = len(book.get("refuted", []))

    snapshot_scorecard(ws)
    (ws.root / "result.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    return outcome


def snapshot_scorecard(ws: Workspace) -> dict[str, Any]:
    """Save the server's own scorecard next to the trace. Never raises.

    **This has to live here, in the parent, and that was nearly missed.**
    ``ArcClient.close()`` snapshots too, but nothing calls it: ``close`` runs
    only from ``__exit__``, the generated ``session.py`` calls ``client.open()``
    and never closes, and a solver does ``from session import client``. So the
    client-side snapshot would never have fired on a real run — the same shape
    as the finding in ``_ineffective``, that a signal in a function nobody calls
    is not a signal. The harness runs after the solver exits and always runs.

    **Deliberately does not close the card.** ``close()`` deletes the state file,
    which is what a resume reads to continue the same game; a run that timed out
    and will be resumed would be broken by it. Cards appear to be reaped
    server-side anyway — two from finished runs returned 404 the same day.

    Why bother: ``actions_by_level`` is the server's per-level action count,
    which is exactly what RHAE scores and which :mod:`athanor.ccarc3.scoring`
    currently re-derives from the trace, never yet compared against the real
    thing. And it holds one row per play, the only evidence that can settle
    which play the scorer uses.
    """
    state = ws.root / "trace.state.json"
    if not state.exists():
        return {}
    try:
        card_id = json.loads(state.read_text(encoding="utf-8")).get("card_id")
    except (json.JSONDecodeError, OSError):
        return {}
    if not card_id or card_id.startswith("card-"):   # a test stub, not a real card
        return {}
    try:
        from .client import ROOT_URL, _get

        card = _get(
            f"{ROOT_URL}/api/scorecard/{card_id}/{ws.info.game_id}",
            os.environ.get("ARC_API_KEY", ""),
        )
        (ws.root / "scorecard.json").write_text(
            json.dumps(card, indent=2) + "\n", encoding="utf-8"
        )
        return card
    except Exception:  # noqa: BLE001 -- bookkeeping must never replace a result
        return {}
