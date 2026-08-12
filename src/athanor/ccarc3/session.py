"""Workspace construction and Codex session launch for CCARC3.

The division of labour is deliberately small: **Codex owns the agent loop**, and this harness supplies a workspace, a toolkit, and a gate. There is
no reviewer and no orchestration — the solver decides what to do next, and the
harness only refuses the moves that are known to destroy a run.

What a workspace contains:

===================  ======================================================
``AGENTS.md``        how to drive the game and what is available
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
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import GameInfo, list_games

__all__ = [
    "Ccarc3Config",
    "Workspace",
    "build_workspace",
    "build_cli_args",
    "run_game",
    "collect_outcome",
    "snapshot_scorecard",
    "redact_self_reference",
    "ledger_facts",
]

ASSETS = Path(__file__).parent / "assets"

# Pin the Codex model and reasoning effort so runs stay comparable.
DEFAULT_MODEL = "gpt-5.3-codex"
DEFAULT_EFFORT = "high"


@dataclass
class Ccarc3Config:
    """One solver run against one game."""

    game_id: str
    out_dir: Path = Path("runs/ccarc3")
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    level_budget_multiple: float = 0.0
    """Per-level action cap as a multiple of that level's baseline. 0 disables it.

    **Disabled on operator instruction, 2026-08-04, and the reasoning is sound.**
    ARC's 5n rule is not a property of the game: the live API does not enforce it
    — three level-attempts in this project ran past it, `su15` L7 to 23.6×, every
    action accepted — and the technical report puts it in §4.3 *Leaderboards*,
    justified by *"the computational cost of evaluating high-reasoning frontier
    models ... tens of thousands of dollars in API costs"*. It is the organisers
    capping their own spend, not a rule the environment imposes on an agent.

    Enforcing it here only ever cost score. Worse, the harness made it invisible
    *and* binding: with the medians withheld the solver got no ratio, no warning
    and no counter, then had an action refused with the environment over. ARC's
    own agents can read `baseline_actions` off the public API and compute 5n for
    themselves, so that combination was harsher than anything ARC does.
    """
    budget_multiple: float = 5.0
    """Action cap as a multiple of the game's published baseline (§2.6).

    A flat cap cannot work: real games differ in length by an order of magnitude, so any
    single number either truncates the long games or wastes the short ones.

    **5.0 because that is ARC's own ceiling.** ARC imposes no game-wide pool at
    all — its limit is 5n *per level*, so an agent that maxed every level would
    spend `5 x baseline_total`. Anything lower is a constraint ARC does not have.

    The previous 2.0 was measurably distorting. Across 19 runs it bound in two —
    `tn36`'s control at 99.5% of its cap and `su15` at 97.1% — and both were
    stopped by the ceiling rather than by the game; `su15` ran out on its ninth
    level and lost roughly 0.2 to the cap alone. **Percentages, not the pair:**
    a cap is the baseline total times `budget_multiple`, and that multiplier is
    a default in this same class, so `used/cap` written out divides straight
    back to the median. It would not have moved any of the other seventeen by a
    single action, whose median usage was 29%. So raising it costs money only on
    the runaway games that were being truncated, which is exactly where
    truncating was destroying score.
    """

    wall_clock_timeout_s: float = 7200.0
    sandbox: str = "workspace-write"
    """Codex sandbox; the ARC proxy is the player's only network path."""
    network_access: bool = True
    """Allow loopback access to the ARC proxy from Codex's workspace sandbox.

    Codex exposes network access at sandbox scope rather than by destination.
    The proxy still withholds the ARC credential and rejects non-game routes;
    ``AGENTS.md`` carries the benchmark-integrity prohibition on other network.
    """
    api_key: str | None = None
    fresh: bool = False
    """Discard any existing trace and start the game over.

    The default is to *resume*. A container can be recycled mid-run, and the
    client already persists its session, so relaunching against the same
    out_dir continues the same game rather than paying for the first N actions
    twice. Making that the default is deliberate; making it silent would not
    be, hence the flag and the log line.

    Resuming restores the local side completely — ``card_id``, ``level``,
    ``actions_used`` and cookies all come back off ``trace.state.json``, and
    ``resume_state.json`` records exactly what was inherited. It cannot restore
    the *server* side: ARC reaps an idle game, and past the reap window the card
    is gone and the resume becomes a replay from level 0 on the old action count.
    Under the window the card is still there and the resume continues in place —
    ``sk48`` did this across a container replacement. See
    :func:`snapshot_scorecard` for the bracket. Resume promptly either way.
    """

    card_id: str = ""
    """Play onto a scorecard the driver already opened, instead of a new one.

    Empty keeps the historical behaviour: one card per game. Set, and the sweep
    lands on one card -- which is what a leaderboard submission takes. The id
    alone does not reach the card; the shim must also carry its stickiness
    cookies (:meth:`arc_proxy.ProxyState.adopt_session`), and the two are set
    together by the driver.
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
    #: Codex thread id, discovered from its ``thread.started`` JSON event.
    session_id: str = ""

    @property
    def trace_path(self) -> Path:
        return self.root / "trace.jsonl"

    @property
    def rules_path(self) -> Path:
        return self.root / "rules.json"


# **The solver reads this template, so the reasoning behind it lives here and not
# inside it.** The rendered `session.py` sits in the workspace, the doctrine tells
# the solver to import it, and `cat session.py` is one of the first things a
# careful one does. Until 2026-08-07 the `ArcClient(...)` call carried thirty
# lines of operator commentary explaining, to the solver, that a hidden action cap
# exists and is enforced in `arc_proxy`; that the human medians are withheld on
# purpose and are "resolved at import and never written to a workspace file"; and
# what a *different* environment had scored. Each of those is a map to something
# the harness spends real machinery hiding, handed over in the one file the solver
# is instructed to open. It also named `su15` outright, which
# `redact_self_reference` never sees because that function only rewrites
# `DOCTRINE.md` -- so a run of `su15` would have read about its own past failure.
#
# The four arguments below, and why they are what they are:
#
# `max_actions=0` -- no client-side cap, deliberately. ARC's FrameResponse carries
# no budget field and the technical report designed away from a per-environment
# allowance: "we won't ... encourage AI to waste actions on levels because they're
# still 'under budget' for a given environment". A solver that knows its allowance
# paces itself against it, which is the wrong objective -- the score is completion
# first and efficiency only as a tiebreak, and across 32 scored runs every point
# lost was lost by not finishing. A hard stop exists far out **on the
# baseline-free path**, enforced in `arc_proxy` where the solver does not run,
# so a runaway loop cannot spend without limit there. Scoped, because
# `Proxy.set_budget` has exactly one caller —
# `ablate_baselines.build_without_baselines` — and a plain `run_game` with no
# strip installed therefore arms no ceiling at any layer. That is fine for the
# local bench and is not what this sentence used to claim. This line used to
# read the cap from the environment; the environment no longer carries it, so it
# evaluated to 0 anyway.
#
# `quiet_pace=True` -- withhold the human medians from every solver-facing
# surface: the pace ratio, `pace()`, and the `raw`/ceiling half of the score
# block. (This used to also switch ARC's 5n rule on; that cap was removed on
# 2026-08-04 -- see `level_budget_multiple`.)
#
# `show_score=True` -- report the running score in `status()`. With baselines
# available that is `raw`, its ceiling and the completion cap; without them the
# cap alone, which needs no baselines because it is which levels fell rather than
# how fast. A solver could not previously see its own score at all: one run blew
# two levels to 8.65x and 22.75x with nothing able to tell it that its ceiling had
# already dropped to 0.82.
SESSION_TEMPLATE = '''\
"""Pre-wired client and gate for {game_id}. Import this; do not rebuild it.

    from session import client, gate, arc

    client.reset()
    client.act(1)
    print(client.status())

Every action is recorded to trace.jsonl automatically. The gate holds the first
action of a new level until you call ``gate.acknowledge(...)``.
"""
import os
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
    max_actions=0,
{level_budget_line}{card_line}    quiet_pace=True,
    show_score=True,
)
client.open()

'''


def _codex_binary() -> str:
    """Resolve the Codex CLI without requiring it in offline unit tests."""
    return os.environ.get("CODEX_BINARY") or shutil.which("codex") or "codex"


def _supports_flag(flag: str) -> bool:
    try:
        out = subprocess.run(
            [_codex_binary(), "exec", "--help"], capture_output=True, text=True, timeout=30
        )
        return flag in (out.stdout + out.stderr)
    except (OSError, subprocess.SubprocessError):
        return False


def build_workspace(config: Ccarc3Config, info: GameInfo | None = None,
                    *, arc_root: str | None = None) -> Workspace:
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
            # **A disabled knob still reads as a knob.** The per-level cap has
            # been off (0.0) since 2026-08-04, but the keyword stayed in the
            # rendered file, where the only word the solver sees is `budget` --
            # in a harness whose whole point is that it is not running one. Emit
            # it only when it is actually doing something.
            level_budget_line=(
                f"    level_budget_multiple={config.level_budget_multiple!r},\n"
                if config.level_budget_multiple else ""
            ),
            # Emitted only when a card is being shared, for the same reason as
            # the line above: a keyword the solver can see is a question the
            # solver can ask, and on the usual one-card-per-game run there is
            # nothing here to explain.
            card_line=(
                f"    card_id={config.card_id!r},\n" if config.card_id else ""
            ),
        ),
        encoding="utf-8",
    )
    shutil.copy(ASSETS / "CCARC3_DOCTRINE.md", root / "DOCTRINE.md")
    (root / "AGENTS.md").write_text(_workspace_agents_md(info, budget), encoding="utf-8")
    redact_self_reference(root, config.game_id)
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
    # The guardrail still binds; it just is not announced. Passing it through the
    # environment keeps the number out of every file the solver reads.
    env["CCARC3_MAX_ACTIONS"] = str(budget)

    # Put the interpreter that actually has numpy first on PATH. The first live
    # run wasted turns on `ModuleNotFoundError: No module named 'numpy'` from
    # bare `python3`, then had to discover the venv path by trial. Making
    # `python3` resolve to the right thing removes the whole class of error
    # rather than documenting a way around it.
    venv_bin = src.parent / ".venv" / "bin"
    if (venv_bin / "python3").exists() or (venv_bin / "python").exists():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}".rstrip(":")

    # **Take the ARC key away from the solver.**
    #
    # `CCARC3_HIDE_BASELINES` closes the in-process route to the human medians and
    # cannot close the deliberate one: with `ARC_API_KEY` in its environment a
    # solver can `urllib` its way to `/api/games`, which returns `baseline_actions`
    # for all 25 environments. `tu93` printed its own array verbatim; `bp35` did
    # the same call six minutes in. Both were caught by the watcher and both had
    # to be thrown away, which is detection paying for prevention's absence.
    #
    # With `CCARC3_PROXY_URL` set, the client is pointed at `arc_proxy` -- which
    # holds the real key, forwards the four endpoints a solver needs, and refuses
    # `/api/games` -- and the key is removed from the solver's environment
    # entirely. A hand-rolled request then has nothing to authenticate with.
    #
    # Unset, nothing changes: the key stays and the client talks to ARC directly.
    # That is deliberate, because this module is read fresh by every `python -c`
    # the solver runs, so a hard switch would break games already in flight.
    # **Passed in, not read from the environment, when the caller knows it.**
    # With several games in flight each has its own shim on its own port, and
    # a single process-wide `CCARC3_PROXY_URL` cannot name more than one of
    # them -- two builders racing on `os.environ` would hand a solver the
    # other game's proxy, and with it the other game's action budget and ARC
    # session. The env var remains the fallback for single-game callers.
    proxy = arc_root or os.environ.get("CCARC3_PROXY_URL")
    if proxy:
        env["CCARC3_ARC_ROOT"] = proxy
        env.pop("ARC_API_KEY", None)
        env.pop("ARCPRIZE_API_KEY", None)
    # **The parent's own bookkeeping never goes to the child.** `install()` keeps
    # `CCARC3_PROXY_URL` in this process's environment, and `env = dict(os.environ)`
    # copied it straight through -- so a solver held the address of a shim that was
    # not its own. With games running concurrently that is another game's budget;
    # with the startup probe it is no budget at all. The child needs exactly one
    # root, `CCARC3_ARC_ROOT`, and it is set above.
    env.pop("CCARC3_PROXY_URL", None)

    return Workspace(
        root=root,
        config=config,
        info=info,
        initial_prompt=_initial_prompt(info, budget, resumed=resumed),
        session_id="",
        env=env,
        resumed=resumed,
    )


def _workspace_agents_md(info: GameInfo, budget: int) -> str:
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

The baseline is what a playthrough costs someone who already knows the rules, so
running some way over it is normal — you are still working them out. Running
*several times* over it on one level is not: that is evidence your hypothesis is
wrong, and the answer is to go re-explore rather than grind.

Your job is to clear as many levels as you can. Nothing here is rationed and
there is nothing to save for later; a level you never reach scores zero however
carefully you played.

## Driving it

```python
from session import client, gate, arc

client.reset()                  # start
client.act(1)                   # ACTION1..5,7 take no arguments
client.act(6, x=10, y=20)       # ACTION6 is a click; x,y in [0,63]
client.status()                 # level, state, actions, and your completion cap
client.pace()                   # {{level: (spent, baseline, ratio)}} for every level
client.transitions()            # everything recorded so far
```

**The game persists across commands.** Each `python -c ...` is a new process,
and the client resumes the same game, scorecard and action count from disk. You
do not need to hold one long-running script open, and you must not try to start
over — importing `session` again continues where you left off.

`client` refuses moves that are known to destroy runs — acting while dead, a
RESET immediately after a level advance, actions this game does not accept. A
refusal costs you nothing and is telling you something.

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
acts = [...]                           # from `available_actions` on the frame
arc.shortest_path(step, start, goal, acts)  # fewest actions, over your step function
arc.reachable(step, start, acts)            # reachable at all, or did I misread?
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
            "with `client.status()` and `client.transitions()` to see where you are. "
            "If `rules.json` exists, read it: it holds what the earlier session "
            "established, what it refuted, and what it never tested — and those are "
            "three different things, not two. Carry the mechanics forward as priors "
            "about what to test first, not as settled fact.\n\n"
            "One RESET is genuinely dangerous: the one taken immediately after a "
            "level advance takes the full-reset branch and discards the whole game. "
            "`client.reset()` refuses that for you. Every other reset is a normal "
            "level reset and a legitimate move.\n\n"
            f"Then keep going: clear as many of the {info.levels} levels as you can, "
            "from wherever the trace leaves off."
        )
    return (
        f"Play `{info.game_id}` and win as many of its {info.levels} levels as you can.\n\n"
        "Read DOCTRINE.md first — several of its points contradict what seems obvious, "
        "and each was learned the hard way.\n\n"
        "Then `from session import client, gate, arc`, RESET, and work out the rules. "
        "Write code to interrogate the trace rather than reading frames by eye. "
        "Dying is cheap and is a legitimate experiment; being confused is expensive."
    )




def redact_self_reference(root: Path, game_id: str) -> int:
    """Remove the doctrine's worked examples that name *this* game.

    **A solver must not read about its own previous attempts.** `DOCTRINE.md`
    earns its keep with concrete measured examples, and those examples name the
    environments they came from -- nine of the twenty-five, at last count. That
    is fine for a game you are not playing and contamination for one you are.

    It stopped being hypothetical on 2026-08-06: §0b tabulates the three arm
    losses with their exact results, and a clean rollout of one of them was
    caught quoting that table back -- naming itself, its level count and its
    action total -- while working out what to do next. Every re-run and rollout
    of those three read a summary of its own prior failure.

    Table rows naming the game are dropped whole -- a row is self-contained, and
    blanking the id would leave its level count and action total, which identify
    it just as well. Prose mentions have the id replaced by a neutral phrase,
    because deleting a sentence mid-paragraph mangles the argument around it.

    **This does not make the doctrine unidentifiable, and should not be sold as
    though it did.** A prose example that says "one run finished at raw 0.449
    having cleared six of seven levels" still tells a solver on a seven-level
    game something about a seven-level game. What it removes is the direct,
    named, self-referential leak. Returns the number of lines changed.
    """
    doc = root / "DOCTRINE.md"
    if not doc.exists():
        return 0
    short = game_id.split("-")[0]
    if not short:
        return 0
    out, changed = [], 0
    for line in doc.read_text(encoding="utf-8").splitlines():
        if short not in line:
            out.append(line)
            continue
        changed += 1
        if line.lstrip().startswith("|"):
            continue                      # drop the whole row
        out.append(re.sub(rf"`?{re.escape(short)}[a-z0-9-]*`?",
                          "another environment", line))
    if changed:
        doc.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def build_cli_args(workspace: Workspace) -> list[str]:
    """Build a non-interactive Codex invocation with a machine-readable stream."""
    config = workspace.config
    args = [_codex_binary(), "exec", "--json", "--sandbox", config.sandbox]
    args += ["-c", f"sandbox_workspace_write.network_access={str(config.network_access).lower()}"]
    if config.model:
        args += ["--model", config.model]
    if config.effort:
        args += ["-c", f'model_reasoning_effort="{config.effort}"']
    args += list(config.extra_cli_args)
    args.append(workspace.initial_prompt)
    return args


def _signal_group(proc: "subprocess.Popen", sig: int) -> None:
    """Signal the solver's whole process group, falling back to the child.

    `start_new_session=True` makes the child a group leader, so one `killpg`
    reaches every descendant it spawned. The fallback matters on the path where
    the child has already exited: `getpgid` then raises ProcessLookupError, and
    a timeout arm that died there would skip the wait below it.
    """
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


#: Seconds a solver gets to shut down cleanly after SIGTERM before SIGKILL.
TERM_GRACE_S = 30


NUDGE_PROMPT = """You stopped, and the environment is not finished: you have not
cleared every level, nothing interrupted you, and you still have actions left to
spend.

Being stuck with actions in hand is a reason to change technique, not to stop —
that is DOCTRINE.md §0b, and it applies now. Re-read your own rules.json and
notes: what you have already worked out about this game is still on disk and
still true.

Do not restart from scratch and do not re-derive what you already know. Pick up
where you are, try an approach you have not tried, and keep playing."""
"""What a solver is told when it quits while it can still act.

**No figure appears in it, and none may be added.** "You still have actions
left" tells a solver only what it could already infer from not having been
refused. A count would say considerably more than that, and this module is on the
solver's own `PYTHONPATH` and reachable by `inspect.getsource` — so the rule
covers this docstring as well as the prompt, and the same reticence the rest of
this file keeps applies here. See `tests/test_the_nudge_says_nothing_it_should_not.py`.
"""


def _max_nudges() -> int:
    """How many times a solver that quits early is told to carry on.

    Distinct from `GIVE_UP_ATTEMPTS`, which counts whole re-runs from scratch in
    fresh workspaces. A nudge continues the *same conversation*: the solver keeps
    its context, its rules.json and its place in the game, and the cost is one
    more turn rather than one more game.

    That is also why this is bounded and why the bound is small. A solver that
    has been told twice to keep going and has stopped twice anyway is reporting
    something about the game, and each nudge still bills a full reasoning turn
    against a run that may not finish.

    **Off unless a caller asks for it, and that default is deliberate.** Nudging
    changes what `run_game` does for *every* caller: a give-up that used to end
    one launch now ends three. Defaulting it on rewrote the behaviour of four
    existing tests without anyone choosing that -- two of which only failed
    because a glob happened to return the new empty stream first, which is the
    kind of silent contract change this project keeps paying for.

    `tools/clean_rollouts.py` turns it on for the sweeps that want it, so the
    driver an operator actually runs gets the behaviour while the library stays
    predictable. `0` is the previous behaviour exactly: the give-up is marked,
    the driver discards, and the game is re-run from scratch.
    """
    raw = os.environ.get("CCARC3_MAX_NUDGES", "")
    try:
        n = int(raw)
    except ValueError:
        if raw:
            print(f"CCARC3_MAX_NUDGES={raw!r} is not a number; nudging disabled",
                  flush=True)
        return 0
    return max(0, n)


def _rotate_stream(ws: Workspace) -> None:
    """Move `stream.jsonl` aside so the next launch cannot overwrite it.

    Every launch is a separate record — the first attempt, and each nudge after
    it. Losing one loses the only evidence of how that handoff actually went,
    which is exactly what needs reading when a nudge fails to land.
    """
    stream = ws.root / "stream.jsonl"
    if stream.exists():
        n = len(list(ws.root.glob("stream.*.jsonl"))) + 1
        stream.rename(ws.root / f"stream.{n}.jsonl")


def _nudge_args(ws: Workspace) -> list[str] | None:
    """CLI args that resume this solver's own session and tell it to continue.

    `None` when the CLI cannot do it, which is not a failure: the caller then
    leaves the give-up marked and the driver re-runs the game as before.

    Codex assigns the thread id and reports it in ``thread.started``.  That event
    is the authoritative resume handle; guessing it from local state can attach
    a nudge to an unrelated conversation.
    """
    if not ws.session_id:
        return None
    args = [
        _codex_binary(), "exec", "resume", "--json",
    ]
    config = ws.config
    if config.model:
        args += ["--model", config.model]
    if config.effort:
        args += ["-c", f'model_reasoning_effort="{config.effort}"']
    args += [ws.session_id, NUDGE_PROMPT]
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

    # **One deadline for the whole run, not one per launch.** A nudged run makes
    # several Codex invocations, and giving each a fresh
    # `wall_clock_timeout_s` would let a game run for two or three times the
    # limit its caller set -- `clean_rollouts` picks that limit so a run fits
    # inside a container window, so multiplying it silently defeats the choice.
    deadline = (time.monotonic() + config.wall_clock_timeout_s
                if config.wall_clock_timeout_s else None)
    nudges_left = _max_nudges()
    nudged = 0

    while True:
        code, timed_out = _launch(ws, args, deadline)
        outcome = collect_outcome(ws, exit_code=code, timed_out=timed_out)
        outcome["nudges"] = nudged
        outcome["session_id"] = ws.session_id

        if not outcome.get("gave_up") or nudges_left <= 0:
            return outcome
        if deadline is not None and time.monotonic() >= deadline:
            return outcome
        nudge = _nudge_args(ws)
        if nudge is None:
            # The CLI cannot resume; leave the give-up marked and let the driver
            # re-run the game from scratch, which is what happened before nudging
            # existed.
            return outcome

        nudges_left -= 1
        nudged += 1
        print(f"    {ws.info.game_id}: solver quit with actions in hand — "
              f"nudge {nudged} of {nudged + nudges_left}, resuming its session",
              flush=True)
        _rotate_stream(ws)
        args = nudge


def _launch(ws: Workspace, args: list[str], deadline: float | None) -> tuple[int, bool]:
    """Run one Codex invocation to completion. Returns (exit code, timed out)."""
    config = ws.config
    stream = ws.root / "stream.jsonl"
    remaining = None if deadline is None else max(1.0, deadline - time.monotonic())
    with stream.open("w", encoding="utf-8") as fh:
        # **Its own session, so the timeout can reach the whole tree.** Without
        # this the child sits in the driver's process group and a signal to
        # `proc` reaches only the `claude` process. The solver drives the game
        # from `python -c ...` grandchildren -- the workspace CLAUDE.md tells it
        # to -- and each of those POSTs actions of its own. Killing the parent
        # left them running: still spending the budget, still writing the ledger,
        # against a run the driver had already declared timed out and moved past.
        proc = subprocess.Popen(
            args,
            cwd=ws.root,
            env=ws.env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=remaining)
            timed_out = False
        except subprocess.TimeoutExpired:
            # **SIGTERM first, and keep whatever exit status comes back.**
            # SIGKILL gave the solver no chance to flush: its stdout is a FILE,
            # not a tty, so the runtime block-buffers it and the tail of
            # `stream.jsonl` -- the only source for `run_cost`'s turn and dollar
            # figures -- died in userspace. The scored ledger is unaffected,
            # since `trace.jsonl` is written per action by the client, so this
            # buys accounting accuracy rather than score.
            #
            # The grace period also makes a clean exit possible for the first
            # time, and that status is worth recording: a solver interrupted one
            # write short of finishing can now exit 0 during the grace, and
            # hard-coding -1 would file that as a kill. `collect_outcome` acts on
            # `exit_code` only when `timed_out` is false, so carrying the real
            # value through is informative and changes no decision.
            _signal_group(proc, signal.SIGTERM)
            try:
                code = proc.wait(timeout=TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
                code = proc.wait()
            timed_out = True

    # Codex owns thread identifiers. Capture the exact id it emitted so a
    # give-up nudge can resume this conversation rather than starting over.
    if not ws.session_id:
        ws.session_id = _thread_id(stream)

    return code, timed_out


def _thread_id(stream: Path) -> str:
    """Return the first valid Codex ``thread.started`` id from a JSONL stream."""
    if not stream.exists():
        return ""
    for line in stream.open(encoding="utf-8", errors="ignore"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "thread.started":
            value = event.get("thread_id")
            if isinstance(value, str):
                return value
    return ""


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

    # **Count what ARC counts.** The server does not bill the opening RESET of a
    # play; the ledger records it, because it is a real event that returned a
    # frame. On `cd82` that read 179 against the scorecard's 177, and per play
    # [108, 71] against [107, 70] -- one apiece, every time.
    #
    # Two numbers both called "actions" is the same shape of defect as the
    # `hypothesis`/`reasoning` mismatch: anyone comparing a result to a scorecard
    # has to rediscover the difference and re-explain it. So `actions_used` is now
    # the billed figure and `trace_rows` keeps the raw count.
    #
    # Derived from the rule, not from arithmetic that happens to match today.
    # `len(transitions) - playthroughs` gives the right answer on every run so far
    # and is still wrong: a mid-play RESET after a death *is* billed, so only a
    # play's *first* transition is exempt, and only when it is actually a RESET.
    starts = {0, *(i for i, t in enumerate(transitions) if t.full_reset)}
    unbilled = sum(1 for i in starts
                   if i < len(transitions) and transitions[i].action == "RESET")

    return {
        "levels_reached": max((t.level for t in transitions), default=0),
        "won": "WIN" in states,
        "actions_used": len(transitions) - unbilled,
        "trace_rows": len(transitions),
        "deaths": sum(
            1
            for prev, cur in zip(["NOT_PLAYED", *states], states)
            if cur == "GAME_OVER" and prev != "GAME_OVER"
        ),
        "wasted_actions": sum(t.wasted for t in transitions),
        "full_resets": resets,
        "playthroughs": resets + 1,
        "actions_final_playthrough": len(final) - (
            1 if final and final[0].action == "RESET" else 0),
        "levels_reached_final_playthrough": max((t.level for t in final), default=0),
    }


def run_cost(stream_path: Path | str) -> dict[str, Any]:
    """Turns and token use from Codex's ``turn.completed`` events.

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

    Codex does not report a dollar amount or wall duration in its JSONL protocol,
    so this port records the auditable token counters instead of manufacturing
    either value. Returns an empty dict when no completed turn exists.
    """
    path = Path(stream_path)
    attempts = sorted(path.parent.glob("stream.*.jsonl")) + [path]
    turns = input_tokens = cached_input_tokens = output_tokens = 0
    found = False
    for attempt in attempts:
        if not attempt.exists():
            continue
        for line in attempt.open(encoding="utf-8", errors="ignore"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("type") != "turn.completed":
                continue
            usage = record.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            found = True
            turns += 1
            input_tokens += usage.get("input_tokens") or 0
            cached_input_tokens += usage.get("cached_input_tokens") or 0
            output_tokens += usage.get("output_tokens") or 0
    if not found:
        return {}
    return {
        "turns": turns,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "attempts": sum(1 for a in attempts if a.exists()),
    }


def _killing_signal(exit_code: int) -> int:
    """The signal that killed the solver, or 0 if it exited on its own.

    Two encodings, because two things report it. ``Popen.wait`` returns a
    *negative* number when the child dies on a signal; but the solver is a node
    CLI that installs its own handler and exits normally with ``128 + signum``,
    so a SIGTERM arrives as **143**, an ordinary-looking exit code. Only the
    second form has ever been observed here, and only the first is documented.
    """
    if exit_code < 0:
        return -exit_code
    if 128 < exit_code < 193:
        return exit_code - 128
    return 0


def _action_budget(ws: Workspace) -> int:
    """The run's action cap, as written into the workspace at setup.

    Read from `meta.json` first, so it reflects the multiple the run actually got.

    **The `meta.json` read has been dead in every baseline-free run since the
    strip landed, and nothing said so.** `strip_baselines` removes
    `action_budget` from `meta.json` on purpose -- the cap is the baseline total
    times `budget_multiple`, so a solver that can read its cap can recover the
    medians. Correct, and it silently zeroed this function for exactly the runs
    it exists to serve: `collect_outcome`'s "a timeout with most of the budget
    unspent is not a result" guard reads it, got 0 for all 25 clean rollouts, and
    never fired once. A guard reading a field that a different subsystem removed
    is a guard that passes by not running -- the same shape as the brake that
    read as engaged and the watcher blind to its own run.

    So fall back to the parent's own arithmetic. `ws.info` carries the real
    baselines in memory: the strip rewrites files in the workspace, not the
    `GameInfo` the driver holds, and `collect_outcome` already publishes
    `baseline_total` from it. This is parent-side only and never reaches the
    solver.

    Returns 0 only when neither source has anything, which leaves every caller
    falling through to its previous behaviour rather than guessing.
    """
    try:
        meta = json.loads((ws.root / "meta.json").read_text(encoding="utf-8"))
        budget = meta.get("action_budget")
        if isinstance(budget, int) and budget > 0:
            return budget
    except (OSError, ValueError):
        pass
    if ws.info and ws.info.baseline_total:
        return ws.info.suggested_budget(ws.config.budget_multiple)
    return 0


def _prior_give_ups(ws: Workspace) -> int:
    """How many earlier attempts at this game already quit with budget in hand.

    **`outcome["attempts"]` looked like this number and is not.** It counts
    ``stream.*.jsonl`` files inside ONE workspace -- solver relaunches into the
    same directory. The driver gives every retry a fresh ``attempt_N/``
    workspace, so it reads 1 on every attempt and a bound written against it
    never bites: the give-up guard would have re-run a game for all twelve of the
    driver's passes, which is the several-hundred-dollar outcome the bound was
    added to prevent.

    Written the same evening as the guard, in a comment describing this exact
    shape -- a proxy that resembles the quantity closely enough to pass reading.

    Counts the thing itself: sibling attempts whose own ``result.json`` records a
    give-up. Returns 0 for any layout without them, which restores the previous
    unbounded behaviour rather than blocking -- and outside the driver nothing
    retries, so there is nothing to bound.
    """
    root = Path(ws.root)
    # <out_dir>/<game>/attempt_N/<game>  ->  the attempts sit two levels up.
    for base in (root.parent.parent, root.parent):
        try:
            siblings = sorted(base.glob("attempt_*/*/result.json"))
        except OSError:
            continue
        if not siblings:
            continue
        n = 0
        for r in siblings:
            if r.parent == root:
                continue                      # this run has not been written yet
            try:
                if "gave up" in (json.loads(r.read_text(encoding="utf-8")).get("error") or ""):
                    n += 1
            except (OSError, ValueError):
                continue
        return n
    return 0


# Names only, never values -- a value in `result.json` is a leak with a longer
# half-life than the run. `False` is the answer the strip path is supposed to
# give for the first two.
# `ARCPRIZE_API_KEY` is here because `ablate_baselines` already guards both
# spellings and a record that watched only one would be a check with a hole in
# exactly the shape of its subject. `CCARC3_PROXY_URL` is deliberately absent: it
# is read on the runner and never reaches the child, so recording it would add a
# permanent `False` that reads like a verified absence.
RECORDED_ENV = ("ARC_API_KEY", "ARCPRIZE_API_KEY", "CCARC3_MAX_ACTIONS",
                "CCARC3_HIDE_BASELINES", "CCARC3_ARC_ROOT")


def _env_facts(ws: Workspace) -> dict[str, Any]:
    """Which of the env vars that matter reached the child, as presence flags.

    **The child's environment was recorded nowhere and was checked nowhere.**
    `build_trace_audit.surface_digest` excludes it from the digest -- correctly,
    since it cannot be recovered from a finished run and would make every digest
    mismatch by construction -- and justified that by saying `proofread_trace.py`
    "reads it from the live process rather than inferring it". It does not, and
    never did: `proofread_trace` takes a workspace path and reads `stream.jsonl`,
    and in `--gz` mode there is no process to read. Its only environment-related
    rule is `PROBES`, which flags that the solver *looked* -- inference from the
    transcript, and only when the solver happened to look.

    So the one solver-visible surface that matters most -- whether `ARC_API_KEY`
    and `CCARC3_MAX_ACTIONS` reached the child -- was covered by a sentence
    rather than by a check. Recording it at the moment the workspace is built is
    the only time it is knowable, and it makes the claim true.
    """
    return {"child_env": {k: k in ws.env for k in RECORDED_ENV}}


def _card_facts(ws: Workspace) -> dict[str, Any]:
    """Which scorecard this run scored on.

    **A sweep's central claim should be checkable, not assumed.** "All 25 games
    are on one card" is exactly the kind of statement that is true until one game
    quietly is not -- a resume onto its old card, a driver restart that minted a
    new one -- and a submission built on the assumption is wrong in a way nothing
    else in the results would show. Recording the id per run turns it into
    something a single pass over the outcomes can verify.
    """
    try:
        saved = json.loads(
            Path(ws.trace_path).with_suffix(".state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    facts = {"card_id": saved.get("card_id", ""),
             "card_plays_at_open": int(saved.get("card_plays_at_open", -1))}
    if saved.get("foreign_card"):
        # This game is NOT on the shared card, whatever the driver intended.
        facts["foreign_card"] = saved["foreign_card"]
    return facts


GIVE_UP_FRACTION_ENV = "CCARC3_GIVE_UP_FRACTION"


def _give_up_fraction() -> float:
    """How much of the allowance may be spent and the stop still count as quitting.

    **The default is 1.0: any allowance left over at all.** A solver that stops
    while it can still act has stopped by choice, and the harness treats that as
    something to continue rather than as a measurement of the environment.

    It was `0.5` until an operator decision on 2026-08-10. The old value drew the
    line in the middle on the reasoning that a solver past halfway had at least
    contested the game, and the cost of that reading is now visible: a run that
    stopped with a large minority of its allowance intact was banked as a real
    loss, and the nudge -- which exists precisely to say "you are not finished,
    carry on" -- never fired on it. Under a fraction of 1.0 the same run is
    continued instead.

    **A run that exhausts the allowance is not exempted because it counts as a
    result. It is exempted because there is nothing left to continue on.** Any
    run that does not score `E = 1` is a loss — grinding to the ceiling and
    losing is a loss too, and calling it "a measurement of the game" was a
    distinction this file used to draw and no longer does. The only question a
    nudge answers is whether the solver still has actions to spend; at
    `used >= budget` it does not, so there is nothing to say to it.

    The retry cost is bounded by `GIVE_UP_ATTEMPTS` regardless of this value, so
    loosening the fraction widens *which* runs get another go, never how many
    goes any one game gets.

    Set the environment variable to tighten it again -- `0.5` restores the old
    behaviour exactly.
    """
    raw = os.environ.get(GIVE_UP_FRACTION_ENV, "")
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        print(f"{GIVE_UP_FRACTION_ENV}={raw!r} is not a number; using 1.0", flush=True)
        return 1.0
    if not 0.0 < value <= 1.0:
        print(f"{GIVE_UP_FRACTION_ENV}={raw!r} is outside (0, 1]; using 1.0", flush=True)
        return 1.0
    return value


GIVE_UP_ATTEMPTS = 3
"""How many times a game that quit early is re-run before its loss is accepted.

Not unbounded, unlike an interruption. A run cut short by a signal or a crash
says nothing about the environment, so retrying it is free information. A run
that *stopped on its own* might be reporting something real about the game, and
at roughly $60 a run the difference between 3 tries and 12 is about $540 spent
to hear the same answer.
"""


def collect_outcome(ws: Workspace, *, exit_code: int, timed_out: bool) -> dict[str, Any]:
    """Read the run's result off disk — never from what the solver claims."""
    outcome = {
        "game_id": ws.info.game_id,
        **_card_facts(ws),
        **_env_facts(ws),
        "levels_total": ws.info.levels,
        "baseline_total": ws.info.baseline_total,
        **ledger_facts(ws.trace_path),
        **run_cost(ws.root / "stream.jsonl"),
        "exit_code": exit_code,
        "timed_out": timed_out,
    }

    # **A solver that was killed did not lose. It was interrupted.**
    #
    # `ft09` is why this exists. A container restart sent SIGTERM to the solver
    # mid-game; the parent survived, collected the trace as it stood -- 4 of 6
    # levels, about 5% of its budget spent -- and wrote a `result.json` with
    # `won: false`, `timed_out: false` and no error. Nothing in it says the run
    # was cut short, and every consumer reads it as an environment that beat us.
    # Worse, the arm's resume rule is `if prior and not prior.get("error"): skip`,
    # so the false loss was permanent: `ft09` would never have been re-run.
    #
    # Marking it as an error is what makes it retryable, and the same applies to
    # the supervisor's own quota stop, which kills solvers by design. A *timeout*
    # is excluded deliberately: that is a real outcome under a rule we chose, and
    # `bp35` is recorded that way on purpose.
    # **A win is exempt here too, and for four months it was not.** Guards 2, 3
    # and 4 below all carry `not won`; this one never did, because it landed
    # first and the exemption arrived with the crash guard afterwards. Nothing
    # justifies the asymmetry, and SIGTERM is not a rare event -- it is how the
    # supervisor stops solvers at the quota ceiling by design, so this fires
    # across a sweep rather than once. `cd82` attempt_2 won 6 of 6 on 655
    # actions, was recorded `interrupted, not a result`, and was discarded and
    # re-played; its scorecard shows both plays winning and the replacement's
    # final playthrough byte-identical to the one thrown away.
    #
    # **The exemption is "won *and* the last play went the distance", not bare
    # "won".** A solver SIGTERM'd early in its replay has already won, but its
    # final playthrough is a stub, and banking that pairs a tiny action count
    # with a level count from a different play -- the flattering half of each.
    # Guard 2 carries that exposure already; this one would carry it far more
    # often.
    #
    # The record of the interruption is kept either way. Folding it into the
    # same branch as the error would delete `killed_by_signal` from exactly the
    # runs this is about, which is how the naive form of this fix loses the
    # evidence it was written to preserve.
    sig = _killing_signal(exit_code)
    if sig and not timed_out:
        outcome["killed_by_signal"] = sig
    finished_cleanly = (
        outcome.get("won")
        and outcome.get("levels_reached_final_playthrough") == outcome.get("levels_total")
    )
    if sig and not timed_out and not finished_cleanly:
        outcome["error"] = (
            f"solver killed by signal {sig} after {outcome.get('actions_used', 0)} "
            f"actions — interrupted, not a result; re-run this game"
        )
    # **A crash is an interruption too, and only the signal case was covered.**
    # The guard above catches SIGTERM and friends. It does not catch a plain
    # non-zero exit, and that is the same false loss by another route: when a
    # container restart moved the agent proxy to a new port, three solvers died
    # with `exit 1` on `Connection refused` mid-game. `wa30` was banked at 5 of 9
    # levels and `lf52` at 6 of 10, both with no error field, both therefore
    # permanently un-retryable under `if prior and not prior.get("error"): skip`.
    #
    # A win is exempt. `re86` also exited 1 — it had already won 8 of 8 and was
    # partway through a replay when the network went — and that result is real.
    # A timeout stays exempt for the reason given above.
    elif exit_code and not timed_out and not outcome.get("won"):
        outcome["error"] = (
            f"solver exited {exit_code} after {outcome.get('actions_used', 0)} "
            f"actions without winning — crashed, not a result; re-run this game"
        )
    # **A wall-clock timeout is only a result if the budget is what ran out.**
    #
    # The two guards above exempt timeouts, on the reasoning that a timeout is "a
    # real outcome under a rule we chose". That holds for the rule the *score*
    # runs on -- the action budget. It does not hold for a wall clock, which is
    # an infrastructure limit picked to fit a container's lifetime and has
    # nothing to do with the game.
    #
    # `sk48` is why this exists. `tools/rerun_losses.py` caps a pass at one hour
    # so the driver survives container replacement. That cap fired while the
    # solver was mid-climb -- it had just reached level 4 of 8 and was averaging
    # 37 actions a level -- and banked `levels_reached: 4` with **4% of the
    # budget used**, `timed_out: true` and no error. Under the
    # resume rule `if prior and not prior.get("error"): skip`, that is the same
    # permanent false loss as `ft09` and `wa30`, and it would have scored the
    # re-run *below* the 5-of-8 it was sent to beat.
    #
    # So: budget genuinely spent -> a real, bad result, left alone. Budget mostly
    # unspent -> the clock stopped the run, not the game. Half is the line
    # because a solver that has used less than half its actions provably still
    # had the replay in reserve that doctrine 0a tells it to keep.
    elif timed_out and not outcome.get("won"):
        budget = _action_budget(ws)
        used = outcome.get("actions_used", 0) or 0
        if budget and used < budget / 2:
            outcome["error"] = (
                f"solver hit the wall clock after {used} of {budget} actions "
                f"({used / budget:.0%} of budget) — interrupted by the clock, "
                f"not a result; re-run this game"
            )
    # **A solver that stopped while it could still act did not lose either.
    # It gave up.**
    #
    # The threshold was half the allowance until 2026-08-10 and is now the whole
    # of it (`_give_up_fraction`, default 1.0). Half drew the line on the
    # reasoning that a solver past the midpoint had at least contested the game;
    # the cost of that reading was that a run stopping with a large minority of
    # its allowance intact was banked as a real loss and never nudged.
    #
    # `lf52` is why this exists, and it is the largest recoverable loss in the
    # set: `exit_code 0`, `error: null`, `timed_out: false`, stopped **three
    # levels short of the end** with most of its allowance unspent and the
    # ceiling nowhere in sight. Nothing stopped it; it stopped. Banked with no
    # error it keeps that score forever, and
    # `if prior and not prior.get("error"): skip` makes it permanent. An earlier
    # `bp35` run did the same, one level short, with almost all of its ceiling
    # untouched.
    #
    # (Deliberately no numbers: this file is on the solver's PYTHONPATH and
    # reachable by `inspect.getsource`, and an action count written beside a
    # baseline total is a human median in two subtractions. The first draft of
    # this comment carried both and the leak test rejected it.)
    #
    # This is the fourth face of one failure: interrupted, crashed, clocked out,
    # and now quit — all four look like a real loss in `result.json`, and all
    # four are the harness failing to record that the environment was never
    # actually contested.
    #
    # **Bounded, unlike the other three.** They mark a run retryable without
    # limit because an interruption says nothing about the game. Quitting might:
    # a solver that stops at 7 of 10 three times running may be telling us the
    # game is hard, and 12 passes at roughly $60 a run is $720 to learn it. After
    # `GIVE_UP_ATTEMPTS` tries the result stands as real.
    elif not exit_code and not timed_out and not outcome.get("won"):
        budget = _action_budget(ws)
        used = outcome.get("actions_used", 0) or 0
        if budget and used < budget * _give_up_fraction() and _prior_give_ups(ws) < GIVE_UP_ATTEMPTS:
            # **A flag, not a substring.** The nudge loop in `run_game` has to
            # recognise this exact condition, and the obvious way -- grepping the
            # message for "gave up" -- makes the wording load-bearing, so
            # rephrasing an error string would silently switch the loop off. This
            # file is full of that shape; it does not need another.
            outcome["gave_up"] = True
            outcome["error"] = (
                f"solver stopped at {outcome.get('levels_reached')} of "
                f"{outcome.get('levels_total')} levels after {used} of {budget} "
                f"actions ({used / budget:.0%} of budget) with no error and no "
                f"timeout — it gave up while it could still act, which is not "
                f"a result; re-run this game"
            )
    # **Bookkeeping must never cost the result.** This parsed rules.json
    # unguarded, three lines before result.json is written, so a file that is not
    # valid JSON took the whole outcome down with it -- a WON run recorded as no
    # result at all. Zero bytes is the realistic case and the sweep manufactures
    # it: `RuleBook.save` uses `write_text`, which truncates before it writes,
    # and the supervisor stops solvers with a signal at the quota ceiling by
    # design. The solver also has Write on its own workspace.
    #
    # Every other reader in this file already guards its parse. The two fields
    # derived here are counts for the log; losing them is a line of missing
    # bookkeeping, not a lost game.
    book = {}
    if ws.rules_path.exists():
        try:
            book = json.loads(ws.rules_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            outcome["rules_error"] = f"{type(exc).__name__}: {exc}"
            book = {}
    # **Absent cost is not zero cost.** `run_cost` returns {} when the stream
    # carries no final `result` event, and a wall-clock timeout guarantees that:
    # the CLI is signalled, so it never writes one. The outcome then has no
    # `cost_usd`, `turns` or `duration_s` at all, and any total that sums across
    # runs silently treats the missing one as free -- a $60 game costing $0 in
    # the arm's own accounting. Recording the absence is what lets a total say
    # "over 24 of 25 runs" instead of quietly meaning it.
    if "cost_usd" not in outcome:
        outcome["cost_unavailable"] = True

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
    and will be resumed would be broken by it.

    **The card is reaped server-side once the game sits idle.** Eight interrupted
    runs have been resumed and they split cleanly on the gap between the kill and
    the relaunch, with the boundary bracketed to **(13.6, 18.2] minutes**:

    ===========  =========  ===============================================
    ``ft09``       9.1 min  card live, resumed at level 4, went on to finish
    ``sb26``      11.9 min  card live, resumed at level 5, finished
    ``lf52``      12.2 min  card live, resumed at level 6, **won 10 of 10**
    ``sk48``     ≥13.6 min  card live across a *container replacement*
    ``bp35``      18.2 min  404 — fresh card, replayed from level 0
    ``bp35``      43.8 min  404 — fresh card, replayed from level 0
    ``ka59``      59.9 min  404 — fresh card, replayed from level 0
    ``tu93``     191.1 min  404 — discarded a restored level 7
    ===========  =========  ===============================================

    The 18.2-minute row is the 2026-08-08 resume, and it cost 344 actions: the
    guard refused, the solver correctly diagnosed a dead card, reconstructed the
    route for levels 0-8 out of the inherited trace and replayed it, arriving
    back at level 8 with the ledger reading 739. That card was re-read from
    **fifteen** independent jars afterwards and answered 404 every time, so it
    was genuinely reaped rather than merely unreachable -- worth stating because
    a wrong-backend 404 looks identical from one attempt (see
    :meth:`ArcClient._snapshot_scorecard`), and the replacement card, read the
    same way, came back 3 times in 15.

    Over the boundary the scorecard 404s and every ``/api/cmd`` answers
    ``game not found``, so the solver must open a fresh card and start again.

    Two earlier versions of this note were wrong and the corrections are the
    point. The first put the deadline at "about 12 minutes"; ``lf52`` resumed at
    12.2 and kept its card — one playthrough, 941 actions, ARC's own card
    recording ``plays=1, states=['WIN']`` and not one ``game not found``. The
    second assumed a *container replacement* always exceeded the window, so a
    restore could carry the notes but never the game. ``sk48`` (2026-08-06)
    falsifies that: it crossed the 02:48:15Z replacement on card
    ``57690598-daed-4fac-8b18-e8bb34734288`` and resumed on the same card, with
    zero ``full_reset`` markers and one continuous play, level 0 -> 1 -> 2.

    The ``sk48`` gap is a **lower bound**, not a measurement: evidence is
    preserved on a 5-minute poll, and its ledger was frozen at 87 actions from
    02:41:25Z through 02:54:59Z, growing again by 03:00:07Z. So the card sat idle
    at least 13.6 minutes and possibly ~19. That is why it tightens the lower
    edge of the bracket and nothing else. The true deadline is above 13.6 minutes
    and at or below **18.2**, per the row added on 2026-08-08; it is not a
    documented ARC policy, so treat the bracket as the claim and resume promptly
    regardless. (This sentence said 43.8 until the same day, because the table
    and the headline above were updated and it was not -- a stale upper bound
    thirty lines below the corrected one, in a paragraph about the bracket.)

    That is worse than losing the progress, because the ledger does not reset
    with the game. ``actions_used`` carries across attempts by design, so the
    re-played actions land on the denominator RHAE divides by while the numerator
    starts again: ``bp35`` spent 175 actions reaching level 4, then 84 more to
    get back to level 2, and is scored on 259.

    So a resume is only cheap if it is *prompt*. Anything that relaunches an
    interrupted run should do so immediately and hold its claim while it does,
    rather than releasing the game to whatever picks it up next.

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
