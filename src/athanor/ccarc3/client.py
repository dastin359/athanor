"""Live ARC-AGI-3 client — the surface a solver actually drives.

Written directly against the HTTP API rather than through ``arc_agi_3``. That is
not a preference: the installed SDK declares ``FrameData.score`` while the server
sends ``levels_completed``, so pydantic fills the default and **every score reads
0 forever, silently** (design note §2.5). Speaking HTTP is the correct call here,
and it also keeps this package free of any dependency beyond numpy.

What the client owns, so the solver does not have to:

- session plumbing (``card_id``, ``guid``, retries)
- appending every frame to the transition ledger, unprompted
- level tracking from ``levels_completed``
- the two moves that silently destroy progress, refused before they are sent

What the solver owns: which action to take, and why.
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gate import GateRefusal, LevelGate
from .ledger import TraceWriter, action_name
from .scoring import LEVEL_SCORE_CAP

__all__ = [
    "ROOT_URL",
    "ArcClient",
    "GameInfo",
    "ActionRefused",
    "GateRefusal",
    "list_games",
    "baselines_for",
]

# **Redirectable, so the key can live somewhere the solver cannot read it.**
# Unset (every run to date, and every solver currently in flight) this is the live
# API and nothing changes. Set, it points at `arc_proxy`, which holds the real key
# and forwards only the four endpoints a solver needs -- refusing `/api/games`,
# the one that carries `baseline_actions`.
#
# **This module is re-imported on every action.** The workspace tells the solver
# "each `python -c ...` is a new process", so a change here is live for games
# already running, not just future ones. Both branches below therefore have to be
# safe for a solver mid-game: with `ARC_API_KEY` set and this env var unset, the
# behaviour is byte-identical to before.
ROOT_URL = os.environ.get("CCARC3_ARC_ROOT") or "https://three.arcprize.org"

_TERMINAL = ("WIN", "GAME_OVER")


class ActionRefused(RuntimeError):
    """The harness declined an action the solver asked for.

    Distinct from a transport error: the request was never sent and the game
    did not step. Carries the reason so the solver can act on it.
    """


def new_session() -> urllib.request.OpenerDirector:
    """An opener with its own cookie jar.

    **The API binds a scorecard to the HTTP session, not to the API key.** Open a
    scorecard on one connection and send RESET on another and the server reports
    ``game <id> not found`` -- a message that points at the game and means the
    session. Closing then fails with ``scorecard <id> not found`` for the same
    reason. Both are cured by carrying cookies, which is why the stock SDK
    threads a ``RequestsCookieJar`` from ``Swarm`` into every ``Agent``.

    Bare ``urllib.request.urlopen`` uses no cookie jar, so every call would be a
    new session. Each client keeps one of these for its whole lifetime.
    """
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def _post(
    url: str,
    payload: dict[str, Any],
    key: str,
    *,
    retries: int = 3,
    opener: urllib.request.OpenerDirector | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"X-API-Key": key, "Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    return _send(req, retries, opener)


def _get(
    url: str,
    key: str,
    *,
    retries: int = 3,
    opener: urllib.request.OpenerDirector | None = None,
) -> Any:
    req = urllib.request.Request(
        url, headers={"X-API-Key": key, "Accept": "application/json"}, method="GET"
    )
    return _send(req, retries, opener)


def _send(
    req: urllib.request.Request,
    retries: int,
    opener: urllib.request.OpenerDirector | None = None,
) -> Any:
    do_open = (opener or urllib.request).open if opener else urllib.request.urlopen
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with do_open(req, timeout=30) as fh:
                return json.loads(fh.read().decode())
        except urllib.error.HTTPError as exc:
            # A bare "HTTP Error 400: Bad Request" is undebuggable, and this API
            # puts the actual reason in the response body. Read it before the
            # handle closes -- it is the only place the reason exists.
            try:
                detail = exc.read().decode()[:500]
            except Exception:  # noqa: BLE001 -- body may already be consumed
                detail = "<no body>"
            if 400 <= exc.code < 500:
                raise RuntimeError(
                    f"{req.get_method()} {req.full_url} -> {exc.code}: {detail}"
                ) from exc
            last = RuntimeError(f"{exc.code}: {detail}")
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


def _api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("ARC_API_KEY", "")
    if key:
        return key
    # Behind the proxy the solver has no key by design -- that is the whole point,
    # since a key in its environment is what let two runs fetch `/api/games` by
    # hand. The proxy injects the real one. Send a placeholder so the request
    # still carries the header the API expects.
    if os.environ.get("CCARC3_ARC_ROOT"):
        return "proxied"
    raise RuntimeError("no ARC_API_KEY; the live API returns 401 without one")


@dataclass(frozen=True)
class GameInfo:
    """A row from ``GET /api/games``."""

    game_id: str
    title: str = ""
    tags: tuple[str, ...] = ()
    baseline_actions: tuple[int, ...] = ()

    @property
    def levels(self) -> int:
        return len(self.baseline_actions)

    @property
    def baseline_total(self) -> int:
        return sum(self.baseline_actions)

    def baseline_for(self, level: int) -> int | None:
        """Baseline action count for a level, or ``None`` past the published list."""
        if 0 <= level < len(self.baseline_actions):
            return self.baseline_actions[level]
        return None

    def suggested_budget(self, multiple: float = 4.0) -> int:
        # A flat ceiling cannot work: real games differ by an order of magnitude
        # in length. See docs/ccarc3_withholding.md.
        """A ceiling derived from the game rather than guessed. Harness-side only."""
        return max(200, int(self.baseline_total * multiple))


HIDE_BASELINES_ENV = "CCARC3_HIDE_BASELINES"
"""Set to ``1`` in a solver's environment to make :func:`list_games` withhold
the per-level human medians.

**The leak that sanitising workspace files cannot reach.** A baseline-free arm
can strip `session.py`, `CLAUDE.md`, `DOCTRINE.md` and `meta.json` and still hand
the solver an API key and this package — at which point `arc.list_games()`
returns `baseline_actions` for all 25 environments on request. Every path in the
package that produces a `GameInfo` from the API comes through here, so this is
the one place the field can be withheld once.

Scoped to the environment rather than a call argument because the solver's
process is where it must apply: the *runner* needs the real numbers to size a
game's budget and score the result. `session.build_cli_args` sets it for the
child only, and nothing sets it on the runner.

**It is the whole rule, with no bypass.** `list_games` used to take an
`_unfiltered=True` escape hatch and `baselines_for` used to ignore the flag
outright, both so harness-side callers could still get the numbers. That was
backwards — those callers live in a process where the flag is unset, so they
never needed an exemption, and the exemptions were reachable from the solver's
namespace. Both are gone: under the flag, no path in this package returns a
baseline.

**Still not airtight, and should not be described as such.** A solver holding
the key can issue its own HTTP request to `/api/games`. What the flag closes is
*incidental* exposure, and the distinction is not academic — `cd82` ran
`dir(arc)` while orienting and read `baselines_for` straight out of the listing,
having gone looking for nothing at all. Check the traces, do not assume.
"""


# Harness-side only; raises under `HIDE_BASELINES_ENV`, which only a solver's
# environment carries. Not re-exported from `athanor.ccarc3`, so it is absent from
# `dir(arc)` in the namespace a solver holds.
# See docs/ccarc3_withholding.md.
def baselines_for(
    game_id: str, api_key: str | None = None, root: str = ROOT_URL
) -> tuple[int, ...]:
    """One game's per-level reference counts. Harness-side callers only."""
    if os.environ.get(HIDE_BASELINES_ENV) == "1":
        raise PermissionError(
            f"baselines_for() is unavailable while {HIDE_BASELINES_ENV}=1. This "
            "process is a solver; per-level human medians are withheld from it by "
            "design."
        )
    for game in list_games(api_key=api_key, root=root):
        if game.game_id == game_id:
            return tuple(game.baseline_actions)
    raise KeyError(f"{game_id} is not in the public set")


# Baselines come back empty under `HIDE_BASELINES_ENV`. There is deliberately no
# bypass argument: there was one, `_unfiltered`, for harness-side callers, and it
# made the flag advisory, because a keyword any caller can pass is not a boundary.
# The environment variable is now the whole rule, honoured in exactly one place.
# Harness callers are unaffected -- the runner never sets the flag on itself, only
# on the child it spawns.
# Per-level reference counts come back empty under `HIDE_BASELINES_ENV`. There is
# deliberately no bypass argument. See docs/ccarc3_withholding.md.
def list_games(api_key: str | None = None, root: str = ROOT_URL) -> list[GameInfo]:
    """Every public game, with its title and action-type tags."""
    raw = _get(f"{root}/api/games", _api_key(api_key))
    hide = os.environ.get(HIDE_BASELINES_ENV) == "1"
    return [
        GameInfo(
            game_id=g["game_id"],
            title=g.get("title", ""),
            tags=tuple(g.get("tags") or ()),
            baseline_actions=() if hide else tuple(g.get("baseline_actions") or ()),
        )
        for g in raw
    ]


@dataclass
class ArcClient:
    """One game, one scorecard, one ledger.

    Usage::

        with ArcClient(game_id, trace_path="trace.jsonl") as c:
            c.reset()
            while not c.done:
                c.act(1)
    """

    game_id: str
    trace_path: str | Path = "trace.jsonl"
    api_key: str | None = None
    root: str = ROOT_URL
    tags: tuple[str, ...] = ("ccarc3",)
    info: GameInfo | None = None
    gate: LevelGate | None = None
    level_budget_multiple: float = 0.0
    """Per-level action cap as a multiple of that level's baseline. 0 disables.

    **This is the official rule, and matching it matters more than it looks.**
    The ARC-AGI-3 technical report: *"we impose an action budget of five times
    the human-baseline median action count per level. That is, for a level with
    a human median of n actions to completion, the agent is terminated after 5n
    actions."*

    Per **level**, not per game. This harness originally capped only the game
    total, at 2.0x the summed baseline — which is roughly **40% of the official
    allowance** and binds in the wrong place. One run was stopped with 6 of its 7 levels
    cleared and roughly 40% of the official allowance still unspent, while its one
    pathological level was allowed to run to 5.6x -- the cap bit where it should
    not have and failed to bite where it should.

    A per-level cap is also the better instrument: it ends the level that is
    going nowhere instead of letting it consume the budget the *next* level
    needed.
    """

    max_actions: int = 0
    """Hard action cap across the whole game. 0 means uncapped.

    Enforced here rather than left to the solver's discipline. A cap the solver
    is merely told about is not a cap, and the failure mode is a run that spends
    its entire budget executing a plan it should have abandoned.
    """

    # Silences the pace ratio and the raw/ceiling half of the score block while
    # the harness keeps using the underlying values. See docs/ccarc3_withholding.md.
    quiet_pace: bool = False
    """Stop reporting pace against the per-level reference count."""

    show_score: bool = False
    """Report the running RHAE score and its ceiling in :meth:`status`.

    The workspace template sets this. The ``False`` default keeps a bare
    ``ArcClient`` silent, which is what the tests and the local bench want.
    """

    card_id: str = ""
    """The scorecard this game is scored on.

    **Set it to share one card across many games; leave it empty for the usual
    one-card-per-game.** A leaderboard submission takes exactly one
    `scorecard_url`, so a 25-game sweep that mints 25 cards has nothing to
    submit. When this is set at construction the client skips
    `/api/scorecard/open` and plays onto the card it was handed.

    Passing the id is necessary and **not sufficient**: the card is reachable
    only from a session carrying its `AWSALBAPP-*` stickiness cookies, and in a
    proxied run those live in the shim, not here. See
    :mod:`athanor.ccarc3.shared_card`.
    """

    guid: str = ""
    actions_used: int = 0
    level: int = 0
    win_levels: int = 0
    state: str = "NOT_PLAYED"
    available_actions: tuple[str, ...] = ()
    full_resets: int = 0
    wasted_actions: int = 0
    close_error: str = ""
    scorecard_error: str = ""
    """Why the scorecard could not be snapshotted before the card was closed.

    Recorded rather than raised: see :meth:`_snapshot_scorecard`.
    """
    level_actions: int = 0
    """Actions spent on the current level, reset when it advances.

    Exists so :meth:`status` can report the ratio against this level's
    published baseline. A run that never left level 0 burned 6.1x that
    level's baseline while the doctrine's own control law -- re-explore
    rather than grind -- sat unmechanised and unread.
    """
    level_costs: tuple[int, ...] = ()
    """Actions spent on each level **this play** has cleared, in order.

    The one piece of state the solver's own score depends on that the harness
    did not keep. RHAE is a weighted mean over per-level action counts, so
    without this list a running score cannot be computed at all -- and for six
    months of this project nothing could tell a solver what it was scoring
    while it still had budget to react.

    Maintained incrementally for the same reason as :attr:`level_repeats`:
    deriving it means parsing the whole trace, which ``status()`` cannot afford.

    Cleared on a full reset, because a new play's score is computed from that
    play alone -- the server records ``actions_by_level`` per play and scores
    the best one.
    """

    level_tried: int = 0
    level_dead: int = 0
    level_repeats: int = 0
    """Actions on this level that were tried, changed nothing, and were repeats.

    Maintained per action rather than derived, because deriving meant parsing
    the whole trace inside ``status()`` -- 654 ms on a real 860-action game,
    called 82 times in one run. See :meth:`_account_effect`.
    """

    level_revisits: int = 0
    """Actions on this level that landed on a board seen earlier on this level.

    Distinct from :attr:`level_repeats`, which counts repeating an action that
    changed *nothing*. This counts cycling: every action changes the board, and
    the board keeps coming back to where it has already been.

    **The distinction is the whole point, because the no-op signal missed the
    only run this project has lost.** `tn36` ground through a level at over five
    times what it was worth, with almost no no-ops in the whole level --
    invisible to ``level_dead``. Nearly a third of those actions landed on a
    board it had already stood on.

    Replaying all 22 level-attempts on record through this exact accounting,
    the two levels that cost `tn36` the game rank first and second:

    ====================  =====  ==========  ========
    level                 ratio  revisited   cleared?
    ====================  =====  ==========  ========
    tn36 L5               5.62x        31%   yes, barely
    tn36 L1               2.57x        16%   yes
    su15 L7               0.93x        11%   yes
    tn36 L3               1.52x         8%   yes
    su15 L5               1.52x         2%   yes
    every lp85 level      <0.9x         0%   yes
    ====================  =====  ==========  ========

    (Fractions, not counts. The ratio column is actions over the level's median,
    so an action count printed beside it divides straight back to the median --
    in a docstring inside ``inspect.getsource(ArcClient)``, which the withholding
    note records a solver actually reading. The percentages carry the entire
    argument; the numerators were the leak.

    The table was cleaned on 2026-08-07 and the prose above it was not, so one
    numerator survived ten lines away and this note reported the job done. That
    is the real lesson: a fix applied to the instance that was noticed, and a
    check that then declared the class closed.)

    **The control is the 1.52x pair.** `su15` L5 and `tn36` L3 ran at exactly
    the same multiple of baseline and both cleared, at 2% and 8%. `tn36` L5 ran
    at 5.62x and 31% and never fell. Both `tn36` L5 and `su15` L5 tripped the
    same 1.0x pace warning, so the ratio alone does not separate them -- over
    pace while reaching *new* states is exploration, over pace while cycling is
    being stuck. `lp85`, the most efficient run on record, revisited nothing at
    all in 84 accounted actions.

    **Not a threshold, and length is a live confound**: longer levels have more
    chances to collide, and the two high-revisit levels are also the two longest
    on record. Four thresholds have dissolved in this project already. This is
    reported as a number so the next batch generates the data to test it.
    """

    _dead_keys: list[str] = field(default_factory=list, repr=False)
    _seen_keys: set[str] = field(default_factory=set, repr=False)
    _last_frame_key: str = field(default="", repr=False)
    _writer: TraceWriter | None = field(default=None, repr=False)
    _opener: Any = field(default=None, repr=False)
    _key: str = field(default="", repr=False)
    _last_advanced: bool = field(default=False, repr=False)
    """The last action completed a level, so the next RESET is a *full* reset.

    Persisted, and that is the whole point. It defaulted to ``False`` in each new
    process, and a CC solver takes every action in a new process. A run that
    ended one action after clearing level 5 resumed with the flag lost, opened
    with RESET exactly as the workspace guide tells it to, and threw away 370
    actions of progress — while the refusal built to prevent that stood unarmed.
    """
    _resumed: bool = field(default=False, repr=False)
    _owns_card: bool = field(default=False, repr=False)
    """True only when this client opened the card itself.

    A shared card outlives the game that happens to finish first, so closing one
    we were merely lent would end the sweep for every game still playing.
    """

    foreign_card: str = field(default="", repr=False)
    """Set when a resumed game is on a different card than the one injected.

    A trace is bound to the card it was played on; it cannot be moved to another
    one after the fact. Resuming wins -- the alternative is a resume that replays
    from level 0 -- and this records that the game is NOT on the shared card, so
    the driver can say so instead of counting it in a submission it is missing
    from.
    """

    @property
    def state_path(self) -> Path:
        return Path(self.trace_path).with_suffix(".state.json")

    def _write_state_atomically(self, payload: str, encoding: str = "utf-8") -> None:
        """Write the state file so a reader never sees a half-written one."""
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding=encoding)
        os.replace(tmp, self.state_path)      # atomic on POSIX

    def _save_state(self) -> None:
        # Written via a temporary file and os.replace, which is atomic on POSIX.
        # `write_text` was measured to issue a single write() syscall, so a torn
        # file is not something this code produces on its own -- but a reader
        # can still observe a partial file, and the cost of an unreadable one is
        # a refused start. Two lines to remove the possibility entirely.
        self._write_state_atomically(
            json.dumps(
                {
                    "game_id": self.game_id,
                    "card_id": self.card_id,
                    "owns_card": self._owns_card,
                    "foreign_card": self.foreign_card,
                    "guid": self.guid,
                    "actions_used": self.actions_used,
                    "level": self.level,
                    "win_levels": self.win_levels,
                    "state": self.state,
                    "available_actions": list(self.available_actions),
                    "full_resets": self.full_resets,
                    "wasted_actions": self.wasted_actions,
                    "level_actions": self.level_actions,
                    "level_costs": list(self.level_costs),
                    "last_advanced": self._last_advanced,
                    "level_tried": self.level_tried,
                    "level_dead": self.level_dead,
                    "level_repeats": self.level_repeats,
                    "level_revisits": self.level_revisits,
                    "dead_keys": self._dead_keys,
                    "seen_keys": sorted(self._seen_keys),
                    "last_frame_key": self._last_frame_key,
                    "cookies": [
                        {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path}
                        for c in self._cookiejar()
                    ],
                    "gate_last_level": self.gate.last_level if self.gate else 0,
                    "gate_pending": self.gate.pending_level if self.gate else None,
                    "gate_acknowledged": (
                        {str(k): v for k, v in self.gate.acknowledged.items()}
                        if self.gate else {}
                    ),
                }
            ),
            encoding="utf-8",
        )

    def _cookiejar(self):
        for h in self._opener.handlers:
            if isinstance(h, urllib.request.HTTPCookieProcessor):
                return h.cookiejar
        return []

    def _restore_state(self) -> bool:
        """Resume a game left by an earlier process. Returns True if resumed.

        **This is what makes the client usable by a Claude Code solver at all.**
        An agent drives it with one-shot ``python -c`` commands, so a new process
        starts for every single action. Without resumption each command built a
        fresh client, which deleted the trace and opened a brand new scorecard --
        the game restarted every time and no run could ever get past action one.
        """
        if not self.state_path.exists():
            return False
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # **Refuse, do not fall through.** Returning False here means "no
            # previous run", and the caller then deletes trace.jsonl and opens a
            # fresh scorecard -- discarding a game because a small sidecar file
            # became unreadable. The file being present says a run exists; only
            # its details are lost, and those are cheaper to lose than the game.
            raise RuntimeError(
                f"{self.state_path} exists but cannot be read ({exc}). A run is "
                f"in progress and its trace has NOT been touched. Move the state "
                f"file aside to start over, or repair it to resume."
            ) from exc
        if saved.get("game_id") != self.game_id:
            return False

        # A card injected at construction must not silently displace the one
        # this trace was actually played on -- and vice versa. Resume wins,
        # because the alternative is replaying from level 0, but the divergence
        # is recorded rather than swallowed.
        injected, saved_card = self.card_id, saved.get("card_id", "")
        if injected and saved_card and injected != saved_card:
            self.foreign_card = injected
        self.card_id = saved_card or injected
        # **A state file written before this field existed always owned its
        # card** -- injection did not exist yet, so every card in an old file was
        # opened by the client that wrote it. Defaulting to False instead would
        # quietly stop resumed pre-existing runs from closing their own cards.
        self._owns_card = bool(saved.get("owns_card", not injected))
        self.foreign_card = self.foreign_card or saved.get("foreign_card", "")
        self.guid = saved.get("guid", "")
        self.actions_used = int(saved.get("actions_used", 0))
        self.level = int(saved.get("level", 0))
        self.win_levels = int(saved.get("win_levels", 0))
        self.state = saved.get("state", "NOT_PLAYED")
        self.available_actions = tuple(saved.get("available_actions") or ())
        self.full_resets = int(saved.get("full_resets", 0))
        self.wasted_actions = int(saved.get("wasted_actions", 0))
        # A state file written before these counters existed has none of them.
        # Defaulting `level_actions` to 0 makes `status()` report 0.0x pace and
        # suppress OVER BASELINE on precisely the resumed runs the warning is
        # for -- a silently wrong number is worse than an absent one, so derive
        # it from the trace instead.
        if "level_actions" in saved:
            self.level_actions = int(saved["level_actions"])
        else:
            self.level_actions = self._level_actions_from_trace(int(saved.get("level", 0)))
        # A state file written before this counter existed has no level_costs.
        # Rebuilding it from the trace is the same choice made for
        # `level_actions` just above, and for the same reason: a resumed run
        # would otherwise report a score computed from an empty history, which
        # reads as "you have cleared nothing" on a run that has cleared six.
        if "level_costs" in saved:
            self.level_costs = tuple(int(c) for c in saved["level_costs"])
        else:
            self.level_costs = self._level_costs_from_trace()
        self._last_advanced = bool(saved.get("last_advanced", False))
        self.level_tried = int(saved.get("level_tried", 0))
        self.level_dead = int(saved.get("level_dead", 0))
        self.level_repeats = int(saved.get("level_repeats", 0))
        self.level_revisits = int(saved.get("level_revisits", 0))
        self._dead_keys = list(saved.get("dead_keys") or [])
        self._seen_keys = set(saved.get("seen_keys") or ())
        self._last_frame_key = saved.get("last_frame_key", "")

        jar = self._cookiejar()
        for c in saved.get("cookies", []):
            jar.set_cookie(
                http.cookiejar.Cookie(
                    0, c["name"], c["value"], None, False,
                    c["domain"], True, c["domain"].startswith("."),
                    c["path"], True, False, None, True, None, None, {},
                )
            )
        if self.gate is not None:
            self.gate.last_level = int(saved.get("gate_last_level", 0))
            self.gate.pending_level = saved.get("gate_pending")
            self.gate.acknowledged = {
                int(k): v for k, v in (saved.get("gate_acknowledged") or {}).items()
            }
        return True

    def __post_init__(self) -> None:
        self._key = _api_key(self.api_key)
        self._opener = new_session()
        if self.gate is not None:
            # An acknowledgement must survive the process that made it.
            self.gate.on_change = self._save_state
        self._writer = TraceWriter(self.trace_path)
        if self._restore_state():
            self._resumed = True
            self._writer._index = self.actions_used
            return

        path = Path(self.trace_path)
        if path.exists():
            # An append-only ledger silently welds runs together and makes
            # load() chain one run's `before` from another's `after`. Never
            # inherit a previous run's trace by accident.
            path.unlink()
        self._writer = TraceWriter(path)

    # -- lifecycle ------------------------------------------------------- #

    def open(self) -> "ArcClient":
        """Open a scorecard, or keep the one a previous process opened.

        Re-opening on resume would abandon the in-flight game and start scoring
        from zero, which is precisely the bug this class exists to avoid.
        """
        if self._resumed and self.card_id:
            self._assert_server_agrees()
            return self
        if self.card_id:
            # A card the driver opened and lent us. Opening another one here is
            # exactly the bug that leaves a sweep with one card per game and
            # nothing to submit.
            self._save_state()
            return self
        card = _post(f"{self.root}/api/scorecard/open", {"tags": list(self.tags)},
                     self._key, opener=self._opener)
        self.card_id = card["card_id"]
        self._owns_card = True
        self._save_state()
        return self

    def _assert_server_agrees(self) -> None:
        """Refuse a resume where the server is not where the ledger thinks it is.

        **The 370-action loss, made impossible instead of merely diagnosable.**
        A resume once preserved the ledger but not the game: trace indices
        continued from 370 while the server replayed levels 0-5 on a fresh
        scorecard, and the run re-spent every one of those actions before anyone
        noticed. `_record_resume_state` was added afterwards and only snapshots
        what was inherited -- it makes the failure reconstructable, not
        preventable, which is the same detect-instead-of-enforce gap that has
        cost this project a run at every level of the stack.

        The mismatch is observable before a single action is spent, and for
        free: reading a scorecard is a `GET` the proxy allows and the server does
        not bill. If the card says this play has cleared fewer levels than the
        ledger claims, the two are describing different games and continuing
        would rewrite one with the other.

        Raising rather than warning is deliberate. The cost of stopping is one
        relaunch; the cost of continuing is the whole attempt, spent invisibly.
        """
        # Nothing spent, nothing to protect. A resume at the very start of a game
        # has no ledger for the server to disagree with, and demanding a card
        # read there would make an offline or not-yet-opened game unresumable for
        # no gain. The 370-action loss required 370 actions to already exist.
        if self.level <= 0 and self.actions_used <= 0:
            return

        try:
            card = self.scorecard()
        except Exception as exc:                  # noqa: BLE001
            # A resume that cannot be verified is one that should not proceed:
            # this call is the only thing standing between a mismatched card and
            # a silently re-spent run.
            raise RuntimeError(
                f"resume: could not read the scorecard to confirm the server is "
                f"at level {self.level} ({exc.__class__.__name__}: {exc}). "
                f"Refusing to continue on an unverified card."
            ) from exc

        entry = (card.get("cards") or {}).get(self.game_id) or card
        done = entry.get("levels_completed")
        if isinstance(done, list):
            done = done[-1] if done else None      # the play now in flight
        if done is None:
            return                                 # nothing to compare against
        if int(done) < self.level:
            raise RuntimeError(
                f"resume: the ledger says level {self.level} but the server's "
                f"card says {done} for this play. The game was replayed or the "
                f"card is not the one this trace was written against; "
                f"continuing would re-spend {self.actions_used} actions. "
                f"Start fresh instead of resuming."
            )

    def close(self) -> dict[str, Any]:
        """Close the scorecard. Never raises.

        A failed close is bookkeeping, not a result: the actions are already
        recorded server-side and the trace is already on disk. Raising here
        would replace whatever the solver was actually doing with a cleanup
        error -- which is exactly what happened during development, where a 404
        from this call buried the real exception from the run. The stock SDK
        takes the same posture and logs a warning.

        The reason, if any, is left on ``close_error`` rather than discarded.
        """
        if not self.card_id:
            return {}
        self._snapshot_scorecard()
        card_id, self.card_id = self.card_id, ""
        self.state_path.unlink(missing_ok=True)
        if not self._owns_card:
            # Someone else's card, still carrying games that have not finished.
            # Snapshotting it was the useful half; closing it would end the
            # sweep on whichever game happened to return first.
            return {}
        try:
            return _post(f"{self.root}/api/scorecard/close", {"card_id": card_id},
                         self._key, opener=self._opener)
        except Exception as exc:  # noqa: BLE001 -- deliberate: see docstring
            self.close_error = f"{type(exc).__name__}: {exc}"
            return {}

    SNAPSHOT_EVERY = 50
    """Actions between scorecard snapshots, on top of the event-driven ones."""

    def _snapshot_scorecard_if_due(self, *, advanced: bool) -> None:
        """Keep a live copy of the scorecard while the card still exists.

        **Snapshotting after the run is too late.** Both the client-side hook in
        ``close()`` and the harness-side one in ``collect_outcome`` were written
        on the assumption that the card outlives the solver. It does not:
        `vc33`'s card returned 404 roughly four minutes after its run ended, and
        cards from older runs 404 as well. The only reliable moment is *during*
        the game, which is here.

        Fires on a level advance, on reaching a terminal state, and every
        :attr:`SNAPSHOT_EVERY` actions as a backstop for a run that dies
        mid-level. A game of nine levels costs about a dozen GETs, against
        hundreds of action POSTs, so the overhead is noise.

        The file is overwritten each time: the newest snapshot strictly
        dominates, since the scorecard only accumulates.
        """
        due = (
            advanced
            or self.state in ("WIN", "GAME_OVER")
            or (self.SNAPSHOT_EVERY and self.actions_used % self.SNAPSHOT_EVERY == 0)
        )
        if due:
            self._snapshot_scorecard()

    def _snapshot_scorecard(self) -> None:
        """Persist the server's own scorecard beside the trace, before closing.

        **RETRACTED 2026-08-08: "the card does not survive the run."** The
        evidence was two `card_id`s from finished runs both answering ``404
        card_id not found`` while the same key still listed all 25 games. The
        reading was wrong. A card is state on **one backend instance**, reachable
        only from a session carrying its `AWSALBAPP-*` stickiness cookies, and
        those reads were made from fresh processes with fresh jars -- which land
        on a backend at random. Re-read on 2026-08-08 from twelve independent
        jars each, two of three finished cards came back with their full
        contents (3/12 and 2/12 hits; the third was 0/12). A 404 here means "you
        asked the wrong instance", not "it is gone", and one try cannot tell
        those apart. Same mechanism the shim was built for: `1 of 8` reads
        survive an unpinned hop, `8 of 8` survive a pinned one.

        Snapshotting is still right -- a one-in-five read is not a way to score a
        run, and the local copy costs nothing -- but the reason is reliability,
        not permanence.

        What is thrown away is not incidental. ``actions_by_level`` is the
        *server's* per-level action count, which is exactly the quantity RHAE
        scores; :mod:`athanor.ccarc3.scoring` currently re-derives it from the
        trace and the two have never been compared on real data. And the
        scorecard carries one row per play, which is the only evidence that
        could settle whether the scorer uses the first winning play or the best
        one -- the largest open question in ``docs/ccarc3_design.md``, and one
        that no finished run can answer any more.

        Never raises, for the same reason :meth:`close` does not: a failed
        snapshot is bookkeeping, and burying a real solver exception under a
        bookkeeping error is a mistake this file has already made once.
        """
        # Stub ids belong to the test suite. Without this the action path fires a
        # real GET on every level advance in every test that plays a game --
        # swallowed by the handler below, so it shows up only as a slower suite
        # (31s -> 40s when this was missing) and a test run that quietly needs
        # the network.
        if not self.card_id or self.card_id.startswith("card-"):
            return
        try:
            card = self.scorecard()
        except Exception as exc:  # noqa: BLE001 -- deliberate: see docstring
            self.scorecard_error = f"{type(exc).__name__}: {exc}"
            return
        try:
            path = self.trace_path.parent / "scorecard.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(card, indent=2), encoding="utf-8")
        except OSError as exc:
            self.scorecard_error = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> "ArcClient":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- state ------------------------------------------------------------ #

    @property
    def done(self) -> bool:
        return self.state == "WIN"

    @property
    def dead(self) -> bool:
        return self.state == "GAME_OVER"

    # Every solver-facing surface reads this one property, so returning None
    # here silences all of them at once. See docs/ccarc3_withholding.md.
    @property
    def baseline_here(self) -> int | None:
        """Reference action count for this level, or ``None`` when unavailable."""
        return None if self.quiet_pace else self._baseline_here_enforced

    @property
    def _baseline_here_enforced(self) -> int | None:
        """This level's human median, whatever the solver is allowed to see."""
        return self.info.baseline_for(self.level) if self.info else None

    # Reads the enforced value rather than the visible one, so hiding a number
    # does not silently switch a limit off. See docs/ccarc3_withholding.md.
    @property
    def level_budget(self) -> int:
        """Per-level ceiling when one is configured; ``0`` means none is."""
        base = self._baseline_here_enforced
        if not base or not self.level_budget_multiple:
            return 0
        return int(base * self.level_budget_multiple)

    # -- score ------------------------------------------------------------ #
    #
    # The solver could not see its own score. It could see this level's pace
    # ratio and nothing else -- no running total, no ceiling, no answer to the
    # only question that decides what to do next: *is what I am doing still
    # worth anything?*
    #
    # `su15` is the case that made this concrete. Its solver blew two levels at
    # many times what each was worth, then cleared the one after and correctly
    # chose to replay. Nothing in the harness could have told it that those two
    # levels had already fixed its ceiling at 0.82 -- which is the fact that made
    # the replay right, and the fact it had to guess.
    #
    # (No action counts here. `session.py` states the same two levels as ratios,
    # and a count beside a ratio is a median in one division -- the halves were
    # harmless apart and not together, which is why a per-file check missed it.)

    @property
    def completion_cap(self) -> float:
        """`C` if this play stopped here: the weighted fraction of levels cleared.

        **Computable without baselines**, unlike everything else in this block,
        because it is pure structure: which levels fell, not how fast. A solver
        that can see nothing else can still see this one.
        """
        n = self.win_levels
        if not n:
            return 0.0
        k = min(len(self.level_costs), n)
        return sum(range(1, k + 1)) / sum(range(1, n + 1))

    def _play_score(self, *, optimistic: bool = False) -> float | None:
        """RHAE for the current play, or None when baselines are unknown.

        ``optimistic=True`` scores every level not yet cleared at the 1.15 per-
        level cap, giving the **ceiling**: the best this play can still finish
        at, however perfectly it plays from here.
        """
        # Solver-facing: withheld baselines make this unanswerable, and the
        # honest answer is None rather than a score computed from numbers the
        # solver is not being shown.
        if self.quiet_pace:
            return None
        baselines = list(self.info.baseline_actions) if self.info else []
        n = self.win_levels or len(baselines)
        if not baselines or len(baselines) < n or not n:
            return None
        weighted = 0.0
        for index in range(1, n + 1):
            level = index - 1
            if level < len(self.level_costs):
                cost = self.level_costs[level]
                # A level credited without any action of its own -- see the
                # multi-level advance note in `_send` -- beat the human by
                # definition, so it takes the cap rather than a division by zero.
                score = (
                    LEVEL_SCORE_CAP if cost <= 0
                    else min(LEVEL_SCORE_CAP, (baselines[level] / cost) ** 2)
                )
            elif optimistic:
                score = LEVEL_SCORE_CAP
            else:
                score = 0.0
            weighted += index * score
        raw = weighted / sum(range(1, n + 1))
        cap = 1.0 if optimistic else self.completion_cap
        return min(raw, cap)

    @property
    def score_now(self) -> float | None:
        """What this play scores if it stops on this action. None without baselines."""
        return self._play_score()

    @property
    def score_ceiling(self) -> float | None:
        """The most this play can still score, playing perfectly from here.

        **The number a replay decision turns on.** RHAE is a weighted mean over
        levels already finished, so a level finished badly is finished badly for
        good -- no later brilliance repairs it. When this drops below what a
        fresh play could reach, this play is worth less than a clean one, and
        :meth:`restart_for_replay` is the instrument.
        """
        return self._play_score(optimistic=True)

    def scorecard(self) -> dict[str, Any]:
        return _get(f"{self.root}/api/scorecard/{self.card_id}/{self.game_id}",
                    self._key, opener=self._opener)

    # -- actions ---------------------------------------------------------- #

    def reset(self, *, force_full: bool = False) -> dict[str, Any]:
        """RESET. Refuses the one call that silently discards the whole game.

        Immediately after a level advance the server's action counter is zero,
        so RESET takes the full-reset branch: score to zero, back to level 0.
        It is invisible from outside and irreversible. Pass
        ``force_full=True`` if that really is what you want.
        """
        if self._last_advanced and not force_full:
            raise ActionRefused(
                "RESET as the first action after a level advance performs a FULL "
                "GAME RESET, discarding every level completed so far. Take any "
                "other action first, or pass force_full=True."
            )
        return self._send(0)

    def restart_for_replay(self) -> dict[str, Any]:
        """Start a **new play** of the same game, keeping everything you learned.

        This is the one legitimate use of the RESET that :meth:`reset` refuses.
        Measured behaviour of the API: a RESET issued while the server's action
        counter is zero — the state immediately after a level advance — begins a
        new play with its own ``guid``, its own ``actions`` row and its own
        ``actions_by_level`` row. Per-level action counts are recorded per play
        and never summed, and the benchmark scores the *best* play.

        So exploration and execution can be separated. Spend whatever it takes
        to work the game out, then restart and walk the route you now know. Only
        the second play's per-level counts are scored; the first play adds to
        ``total_actions`` and affects nothing else.

        **This is not a way to replay a recorded file.** The trace of a fumbling
        run replayed verbatim reproduces the fumbling. What earns the score is
        executing the route your *understanding* implies, which is work you can
        only do once you actually understand the game.

        Refuses unless the counter is at zero, because a RESET anywhere else is
        a level reset and would silently leave you in the same play.
        """
        if not self._last_advanced:
            raise ActionRefused(
                "restart_for_replay() only starts a new play when the server's "
                "action counter is zero, which is the state immediately after a "
                "level advance. Right now a RESET would be a level reset and you "
                "would stay in the same play, so the replay would not be scored "
                "separately. Finish the level you are on first."
            )
        return self.reset(force_full=True)

    def act(self, action: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
        """Take a game action. ``ACTION6`` needs ``x``/``y`` in ``[0, 63]``."""
        if action == 0:
            return self.reset()
        if self.state in _TERMINAL:
            raise ActionRefused(
                f"state is {self.state}; every non-RESET action is discarded "
                f"without stepping the game, so it tells you nothing and moves "
                f"nothing. RESET first."
            )
        if action == 6 and (x is None or y is None):
            raise ActionRefused("ACTION6 requires x and y")
        if action == 6 and not (0 <= x <= 63 and 0 <= y <= 63):
            raise ActionRefused(f"ACTION6 coordinates must lie in [0, 63]; got ({x}, {y})")
        if self.available_actions and action_name(action) not in self.available_actions:
            raise ActionRefused(
                f"{action_name(action)} is not in this game's available_actions "
                f"{list(self.available_actions)}; the frame says this game does "
                f"not accept it, so it would return an unchanged board and teach "
                f"you nothing."
            )
        return self._send(action, x=x, y=y)

    def _send(self, action: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
        if self.max_actions and self.actions_used >= self.max_actions:
            raise ActionRefused(
                f"harness ceiling reached at {self.actions_used} actions. "
                f"Reached level {self.level} of {self.win_levels or '?'}."
            )
        level_cap = self.level_budget
        if level_cap and self.level_actions >= level_cap:
            # `level_cap` is the baseline times the multiple, so naming it at all
            # reveals the baseline to arithmetic. That is acceptable *here* and
            # nowhere else: this refusal ends the environment, so there is no
            # subsequent decision the number could inform.
            base = self._baseline_here_enforced
            whose = "withheld" if self.quiet_pace else str(base)
            raise ActionRefused(
                f"per-level ceiling reached: {self.level_actions}/{level_cap} "
                f"on level {self.level} (baseline {whose}, "
                f"{self.level_budget_multiple:g}x). This is the official ARC-AGI-3 "
                f"rule -- an agent is terminated after {self.level_budget_multiple:g}n "
                f"actions on a level. Nothing further can be scored on this level, so "
                f"the environment is over."
            )
        if self.gate is not None:
            self.gate.check()
        name = action_name(action)
        payload: dict[str, Any] = {"game_id": self.game_id}
        if action == 0:
            payload["card_id"] = self.card_id
        if self.guid:
            payload["guid"] = self.guid
        if action == 6:
            payload["x"], payload["y"] = x, y

        frame = _post(f"{self.root}/api/cmd/{name}", payload, self._key, opener=self._opener)
        if "error" in frame:
            raise RuntimeError(f"{name} refused by server: {frame['error']}")

        previous_level = self.level
        self.guid = frame.get("guid") or self.guid
        self.state = str(frame.get("state", self.state))
        self.win_levels = int(frame.get("win_levels", self.win_levels) or 0)
        self.level = int(frame.get("levels_completed", frame.get("score", 0)) or 0)
        self.available_actions = tuple(
            action_name(a) for a in (frame.get("available_actions") or ())
        )
        self.actions_used += 1
        if self.level > previous_level:
            # This action is the last one taken *from* the level just cleared,
            # which is the attribution :func:`scoring.actions_per_level` uses, so
            # its cost is the running tally plus this one.
            #
            # A jump of more than one level has never been observed; if the
            # server ever credits two at once, the extra levels really did cost
            # zero actions of their own and are recorded as such. Zero is read
            # as "cleared for free" by :meth:`_play_score`, not as missing data.
            gained = self.level - previous_level
            self.level_costs = self.level_costs + (self.level_actions + 1,) + (0,) * (gained - 1)
        self.level_actions = 0 if self.level > previous_level else self.level_actions + 1
        # **The play-opening RESET is not an action, and this tally counted it.**
        # `scoring.actions_per_level` drops it explicitly and the server agrees:
        # seven ACTION6 calls preceded by the opening RESET come back as
        # `actions: [7]`, not 8. Measured across every preserved run carrying
        # `level_costs`, this was +1 against the scorer on element 0 of any run
        # with no full reset, and exact everywhere else -- a run whose last play
        # began with a full reset already zeroes here, twelve lines down.
        #
        # It matters now in a way it did not before: the workspace template sets
        # `show_score=True`, so every game in a sweep prints a running score
        # derived from this tally while the banked number comes from the offline
        # scorer. Two figures for one quantity, differing by one action on the
        # first level, is the kind of discrepancy that costs an afternoon to
        # rediscover.
        #
        # `actions_used` carries the same +1 and is deliberately left alone: it
        # feeds the budget ceiling, where counting the opening RESET is the
        # conservative direction, and changing it is a separate decision.
        if action == 0 and self.actions_used == 1:
            self.level_actions = 0
        # The server's own flag is not reliable. On the one full reset this
        # project has recorded, the level went 6 -> 0 and ``full_reset`` came
        # back **False**, so the counter read zero and the run reported "zero
        # full resets" while replaying the entire game. A level that goes *down*
        # is the fact; the flag is a hint.
        if frame.get("full_reset") or self.level < previous_level:
            self.full_resets += 1
            self.level_actions = 0
            self.level_costs = ()
        if not frame.get("frame"):
            self.wasted_actions += 1
        # `or frame.get("full_reset")` is defensive, not decorative: the server
        # has already been observed lying in the other direction (false on a
        # 6 -> 0 transition), so trust neither signal alone. Either one means
        # the board this level's tally describes is gone.
        self._account_effect(
            frame, name, payload,
            board_replaced=self.level != previous_level or bool(frame.get("full_reset")),
        )
        # Set *after* reading, so the flag describes the state the next call
        # will act in -- which is exactly when the RESET trap fires.
        self._last_advanced = self.level > previous_level

        if self.gate is not None:
            self.gate.observe(self.level)

        assert self._writer is not None
        self._writer.append(frame, level=self.level)
        # Persist after every action: the next action usually arrives in a
        # different process, and anything not on disk is gone.
        self._save_state()
        self._snapshot_scorecard_if_due(advanced=self.level > previous_level)
        return frame

    def transitions(self):
        from .ledger import load

        return load(self.trace_path)

    def pace(self) -> dict[int, tuple[int, int, float]]:
        """Per level: ``{level: (spent, baseline, ratio)}``. Costs no actions.

        A method rather than a bare function the solver has to feed baselines
        into, because the baselines live on ``info`` and an example that says
        ``level_pace(ts, baselines)`` names something that does not exist in
        the solver's namespace.
        """
        from .rules import level_pace

        if self.quiet_pace:
            return {}
        return level_pace(self.transitions(), self.info.baseline_actions if self.info else ())

    def status(self) -> str:
        """A one-line, honest progress report.

        Reports level, state, action count, and whichever of the score terms
        this run has the inputs to compute.

        **Facts on the first line, warnings on their own lines after it.**
        Interleaving them produced ``<- OVER BASELINE: re-explore rather than
        grind, wasted=3, FULL RESETS=1``, where two unrelated counters read as
        the tail of a sentence telling the solver what to do.
        """
        facts = [
            f"{self.game_id}: level {self.level}/{self.win_levels or '?'}",
            f"state={self.state}",
            f"actions={self.actions_used}",
        ]
        warnings: list[str] = []

        base = self.baseline_here
        if base:
            ratio = self.level_actions / base
            cap_note = f" of {self.level_budget} allowed" if self.level_budget else ""
            facts.append(
                f"[{self.level_actions}/{base} on this level = {ratio:.1f}x{cap_note}]"
            )
            # 1.0, not the 2.0 first shipped here. Over 26 level-attempts, 24 of
            # 25 cleared levels finished at or under 0.92x and the median was
            # 0.52x, so crossing 1.0 is already the unusual case. No cutpoint in
            # 1.0-3.4 fits the data better than any other -- the sample is empty
            # in between -- and warning at the bottom of that gap costs a re-read
            # while warning at the top costs the hundreds of actions in between.
            if ratio >= 1.0:
                warnings.append("OVER BASELINE: re-explore rather than grind")
        if self.show_score:
            ceiling = self.score_ceiling
            if ceiling is None:
                # **Label it a ceiling, because a bare "cap 1.000" reads as a
                # perfect score and cost a run 0.2748.** `E = min(cap, raw)`.
                # With the medians withheld only `cap` is computable, so this
                # used to print `[cap 1.000 = 9/9 levels]` and stop -- half the
                # formula, with no sign that the other half existed. On
                # 2026-08-07 `bp35` cleared 9 of 9 in 990 actions -- around half
                # again what the game was worth -- read that line, and concluded: "All nine
                # levels are cleared with a perfect score, so the game is won. I
                # should wrap this up." Its `raw` was 0.7252. It never replayed,
                # and the replay was available at that exact frame and worth up
                # to +0.2748.
                #
                # The fix is to say what the number is not. A solver that knows
                # `cap` is an upper bound and `raw` is unmeasurable has the
                # information §0a's blind-replay rule needs; one shown `cap
                # 1.000` has a number that contradicts the rule.
                facts.append(f"[cap {self.completion_cap:.3f} = {len(self.level_costs)}"
                             f"/{self.win_levels or '?'} levels — a CEILING, not "
                             f"your score: efficiency is unmeasurable here and can "
                             f"only lower it]")
                # At the winning frame the replay instrument is legal and one
                # action from being gone forever. Saying so once, at exactly the
                # moment it applies, is worth more than the doctrine paragraph
                # that says the same thing an hour earlier.
                if self.state == "WIN":
                    warnings.append(
                        f"WON — and `restart_for_replay()` is legal RIGHT NOW and "
                        f"illegal after any further action. Your score is "
                        f"min(cap, raw); cap is {self.completion_cap:.3f} and raw "
                        f"is unknown to you, so a clean replay can only raise it. "
                        f"You cannot compute raw on this run — replay anyway."
                    )
            else:
                facts.append(
                    f"[score {self.score_now:.3f}, ceiling {ceiling:.3f}, "
                    f"cap {self.completion_cap:.3f}]"
                )
                # A ceiling under 1.0 means levels already finished have put the
                # rest of this play out of reach of a perfect score. Grinding on
                # cannot recover it; only a new play can. The threshold is the
                # score a fresh play could still reach, which is 1.0 by
                # construction, so any shortfall at all is the signal.
                if ceiling < 0.999:
                    warnings.append(
                        f"CEILING {ceiling:.3f}: levels already finished cap this "
                        f"play below 1.000 however well you play from here. A new "
                        f"play would be worth up to {1.0 - ceiling:+.3f} if the "
                        f"remaining budget covers redoing what you have cleared "
                        f"-- see restart_for_replay()."
                    )
        if self.wasted_actions:
            facts.append(f"wasted={self.wasted_actions}")
        if self.full_resets:
            facts.append(f"FULL RESETS={self.full_resets}")
        waste = self._ineffective()
        if waste:
            warnings.append(waste)

        return " ".join(facts) + "".join(f"\n  <- {w}" for w in warnings)

    def _level_costs_from_trace(self) -> tuple[int, ...]:
        """Per-level costs for the **current play**, rebuilt from the ledger.

        Only used to repair a state file that predates the counter.
        :func:`athanor.ccarc3.scoring.actions_per_level` already cuts to the
        last play and attributes by the level an action was taken *from*, which
        is exactly this list; the trailing ``None`` entries are levels not yet
        cleared and are dropped.
        """
        from .scoring import actions_per_level

        try:
            transitions = self.transitions()
        except Exception:  # noqa: BLE001 -- a resume must not die on a bad trace
            return ()
        n = self.win_levels or (len(self.info.baseline_actions) if self.info else 0)
        if not n:
            return ()
        counts = actions_per_level(transitions, n)
        return tuple(c for c in counts if c is not None)

    def _level_actions_from_trace(self, level: int) -> int:
        """How many actions the ledger says were spent on ``level``.

        Only used to repair a state file that predates the counter. Attribution
        matches :func:`athanor.ccarc3.scoring.actions_per_level`: an action
        belongs to the level it was taken *from*.
        """
        try:
            transitions = self.transitions()
        except Exception:  # noqa: BLE001 -- a resume must not die on a bad trace
            return 0
        count = 0
        previous: int | None = None
        for t in transitions:
            if (previous if previous is not None else 0) == level:
                count += 1
            previous = t.level
        return count

    def _account_effect(
        self, frame: dict[str, Any], name: str, payload: dict[str, Any], *,
        board_replaced: bool,
    ) -> None:
        """Tally, in O(1), whether this action changed anything on this level.

        Counted here rather than derived in :meth:`status` because deriving it
        meant parsing the whole trace, and on a real 860-action game that is
        **654 ms** -- against 10 ms to merely read the file, so the cost is
        decoding 14 MB of frames. ``status()`` is the most-called method in the
        harness (82 times in one run), which would have added ~54 s of pure
        parsing to a run for a two-number summary.

        Memoising would not help: every action arrives in a new process (§9.1),
        so a per-process cache never gets a second hit. Persisted counters are
        the same shape ``level_actions`` and ``full_resets`` already use.

        Cross-checked against the version it replaced by replaying four real
        traces through both: identical on all of them, including the run that
        never cleared a level, where both give ``(336 tried, 67 dead, 17
        repeats)``. Two counters that are supposed to mean the same thing and
        are computed two different ways should be made to say so.
        """
        grids = frame.get("frame") or []
        if not grids:
            # A wasted action never reached the game, so it is not evidence
            # about the action -- but if the *level* changed anyway, the tally
            # describes a board that no longer exists and must not carry over.
            if board_replaced:
                self.level_tried = self.level_dead = self.level_repeats = 0
                self.level_revisits = 0
                self._dead_keys = []
                self._seen_keys = set()
                self._last_frame_key = ""
            return
        key = hashlib.blake2b(
            json.dumps(grids[-1], separators=(",", ":")).encode(), digest_size=16
        ).hexdigest()
        previous, self._last_frame_key = self._last_frame_key, key
        if board_replaced or not previous:
            self.level_tried = self.level_dead = self.level_repeats = 0
            self.level_revisits = 0
            self._dead_keys = []
            self._seen_keys = {key}
            return
        # Membership before insertion, and only for boards that actually moved:
        # a no-op leaves the board on a key already in the set, so counting it
        # here would double-report what ``level_dead`` already covers.
        if key != previous:
            if key in self._seen_keys:
                self.level_revisits += 1
            else:
                self._seen_keys.add(key)
        self.level_tried += 1
        if key != previous:
            return
        self.level_dead += 1
        what = f"{name}:{payload.get('x')},{payload.get('y')}" if name == "ACTION6" else name
        if what in self._dead_keys:
            self.level_repeats += 1
        else:
            self._dead_keys.append(what)

    def _ineffective(self) -> str:
        """Report actions that changed nothing on this level. Silent when none.

        Lives in ``status()`` rather than in a function of its own on deliberate
        evidence: across seven runs and 539 solver commands, every analytical
        helper this package exports was called **zero** times while ``status()``
        was called 82. A signal in a function nobody calls is not a signal.

        **Why a number and not a verdict.** Waste is rank-ordered with outcome
        across the six runs on record, and nothing sharper survives contact:

        =========================  =========  ===================
        run                        effective  repeats per 100
        =========================  =========  ===================
        never cleared a level      80%        5.1
        won, but the hard way      93%        1.8
        four clean wins            100%       0.0
        =========================  =========  ===================

        The first draft of this said "the threshold is zero, and that is
        measured". Four winning runs did change the board on literally every
        action -- 845/845, 351/351, 76/76, 69/69 -- so it looked settled. The
        fifth win wasted 8 of 114 and cleared six levels comfortably inside what
        they were worth. **Waste is compatible with winning**, so this reports
        the count and lets the solver judge.

        An earlier version looked for an action whose every attempt was dead.
        Replayed against the failing run it never fired once in 336 actions:
        ``ACTION6`` was not dead, it was 111/157. The waste hid inside a working
        action, which is why the fraction is reported rather than a
        stop-using-this verdict.

        A repeat is not *always* waste either -- a game with hidden state can
        make a previously-inert action live.
        """
        parts: list[str] = []
        if self.level_dead:
            dead, tried = self.level_dead, self.level_tried
            note = f"{dead}/{tried} actions on this level changed nothing"
            if self.level_repeats:
                note += (
                    f" ({self.level_repeats} repeated one you had already seen "
                    f"do nothing)"
                )
            parts.append(note)
        if self.level_revisits:
            # Reported separately because it is a different failure. On the
            # level `tn36` ground through, the no-op count above saw under 3% of
            # the actions; this one saw nearly a third. Every action was changing
            # the board -- and putting it back somewhere it had already been.
            parts.append(
                f"{self.level_revisits}/{self.level_tried} actions returned the "
                f"board to a state already seen on this level"
            )
        return "; ".join(parts)
