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
- the two budget traps that cost real progress (§2.3, §2.4)

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

__all__ = [
    "ROOT_URL",
    "ArcClient",
    "GameInfo",
    "ActionRefused",
    "GateRefusal",
    "list_games",
]

ROOT_URL = "https://three.arcprize.org"

_TERMINAL = ("WIN", "GAME_OVER")


class ActionRefused(RuntimeError):
    """The harness declined an action the solver asked for.

    Distinct from a transport error: the request was never sent, and no budget
    was spent. Carries the reason so the solver can act on it.
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
    if not key:
        raise RuntimeError("no ARC_API_KEY; the live API returns 401 without one")
    return key


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
        """An action cap derived from the game, not guessed.

        The published baseline is what a playthrough costs when the rules are
        already known; a solver must also *discover* them, hence the multiple.
        Design note §2.6 -- a flat cap cannot work when real games span 171 to
        1843 baseline actions.
        """
        return max(200, int(self.baseline_total * multiple))


def list_games(api_key: str | None = None, root: str = ROOT_URL) -> list[GameInfo]:
    """Every public game, with its per-level baselines and action-type tags."""
    raw = _get(f"{root}/api/games", _api_key(api_key))
    return [
        GameInfo(
            game_id=g["game_id"],
            title=g.get("title", ""),
            tags=tuple(g.get("tags") or ()),
            baseline_actions=tuple(g.get("baseline_actions") or ()),
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
    allowance** and binds in the wrong place. `tn36` was stopped at 634 actions
    having cleared 6 of 7 levels; the official rule would have allowed up to
    1585, while separately cutting its one pathological level at 5x rather than
    letting it run to 5.6x.

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

    card_id: str = ""
    guid: str = ""
    actions_used: int = 0
    level: int = 0
    win_levels: int = 0
    state: str = "NOT_PLAYED"
    available_actions: tuple[str, ...] = ()
    full_resets: int = 0
    wasted_actions: int = 0
    close_error: str = ""
    level_actions: int = 0
    """Actions spent on the current level, reset when it advances.

    Exists so :meth:`status` can report the ratio against this level's
    published baseline. A run that never left level 0 burned 6.1x that
    level's baseline while the doctrine's own control law -- re-explore
    rather than grind -- sat unmechanised and unread.
    """
    level_tried: int = 0
    level_dead: int = 0
    level_repeats: int = 0
    """Actions on this level that were tried, changed nothing, and were repeats.

    Maintained per action rather than derived, because deriving meant parsing
    the whole trace inside ``status()`` -- 654 ms on a real 860-action game,
    called 82 times in one run. See :meth:`_account_effect`.
    """

    _dead_keys: list[str] = field(default_factory=list, repr=False)
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
                    "guid": self.guid,
                    "actions_used": self.actions_used,
                    "level": self.level,
                    "win_levels": self.win_levels,
                    "state": self.state,
                    "available_actions": list(self.available_actions),
                    "full_resets": self.full_resets,
                    "wasted_actions": self.wasted_actions,
                    "level_actions": self.level_actions,
                    "last_advanced": self._last_advanced,
                    "level_tried": self.level_tried,
                    "level_dead": self.level_dead,
                    "level_repeats": self.level_repeats,
                    "dead_keys": self._dead_keys,
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

        self.card_id = saved.get("card_id", "")
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
        self._last_advanced = bool(saved.get("last_advanced", False))
        self.level_tried = int(saved.get("level_tried", 0))
        self.level_dead = int(saved.get("level_dead", 0))
        self.level_repeats = int(saved.get("level_repeats", 0))
        self._dead_keys = list(saved.get("dead_keys") or [])
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
            return self
        card = _post(f"{self.root}/api/scorecard/open", {"tags": list(self.tags)},
                     self._key, opener=self._opener)
        self.card_id = card["card_id"]
        self._save_state()
        return self

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
        card_id, self.card_id = self.card_id, ""
        self.state_path.unlink(missing_ok=True)
        try:
            return _post(f"{self.root}/api/scorecard/close", {"card_id": card_id},
                         self._key, opener=self._opener)
        except Exception as exc:  # noqa: BLE001 -- deliberate: see docstring
            self.close_error = f"{type(exc).__name__}: {exc}"
            return {}

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

    @property
    def baseline_here(self) -> int | None:
        return self.info.baseline_for(self.level) if self.info else None

    @property
    def level_budget(self) -> int:
        """Actions allowed on the current level, or 0 when uncapped."""
        base = self.baseline_here
        if not base or not self.level_budget_multiple:
            return 0
        return int(base * self.level_budget_multiple)

    def scorecard(self) -> dict[str, Any]:
        return _get(f"{self.root}/api/scorecard/{self.card_id}/{self.game_id}",
                    self._key, opener=self._opener)

    # -- actions ---------------------------------------------------------- #

    def reset(self, *, force_full: bool = False) -> dict[str, Any]:
        """RESET. Refuses the one call that silently discards the whole game.

        Immediately after a level advance the server's action counter is zero,
        so RESET takes the full-reset branch: score to zero, back to level 0
        (§2.3). It is invisible from outside and irreversible. Pass
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
        the second play's per-level counts are scored; the first play costs you
        ``total_actions`` — budget — and nothing else.

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
                f"without stepping the game and still costs budget (§2.4). "
                f"RESET first."
            )
        if action == 6 and (x is None or y is None):
            raise ActionRefused("ACTION6 requires x and y")
        if action == 6 and not (0 <= x <= 63 and 0 <= y <= 63):
            raise ActionRefused(f"ACTION6 coordinates must lie in [0, 63]; got ({x}, {y})")
        if self.available_actions and action_name(action) not in self.available_actions:
            raise ActionRefused(
                f"{action_name(action)} is not in this game's available_actions "
                f"{list(self.available_actions)}; it would cost budget and do nothing."
            )
        return self._send(action, x=x, y=y)

    def _send(self, action: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
        if self.max_actions and self.actions_used >= self.max_actions:
            raise ActionRefused(
                f"action budget exhausted: {self.actions_used}/{self.max_actions}. "
                f"Reached level {self.level} of {self.win_levels or '?'}."
            )
        level_cap = self.level_budget
        if level_cap and self.level_actions >= level_cap:
            raise ActionRefused(
                f"per-level action budget exhausted: {self.level_actions}/{level_cap} "
                f"on level {self.level} (baseline {self.baseline_here}, "
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
        self.level_actions = 0 if self.level > previous_level else self.level_actions + 1
        # The server's own flag is not reliable. On the one full reset this
        # project has recorded, the level went 6 -> 0 and ``full_reset`` came
        # back **False**, so the counter read zero and the run reported "zero
        # full resets" while replaying the entire game. A level that goes *down*
        # is the fact; the flag is a hint.
        if frame.get("full_reset") or self.level < previous_level:
            self.full_resets += 1
            self.level_actions = 0
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

        return level_pace(self.transitions(), self.info.baseline_actions if self.info else ())

    def status(self) -> str:
        """A one-line, honest progress report.

        Leads with the ratio of actions spent on this level against its published
        baseline, because that number is the doctrine's control law and it was
        previously computable but never shown. A failed run sat at 6.1x on one
        level without anything saying so.

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
        if self.wasted_actions:
            facts.append(f"wasted={self.wasted_actions}")
        if self.full_resets:
            facts.append(f"FULL RESETS={self.full_resets}")
        waste = self._ineffective()
        if waste:
            warnings.append(waste)

        return " ".join(facts) + "".join(f"\n  <- {w}" for w in warnings)

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
                self._dead_keys = []
                self._last_frame_key = ""
            return
        key = hashlib.blake2b(
            json.dumps(grids[-1], separators=(",", ":")).encode(), digest_size=16
        ).hexdigest()
        previous, self._last_frame_key = self._last_frame_key, key
        if board_replaced or not previous:
            self.level_tried = self.level_dead = self.level_repeats = 0
            self._dead_keys = []
            return
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
        fifth win wasted 8 of 114 and cleared six levels in 121 actions against
        a 171 baseline. **Waste is compatible with winning**, so this reports
        the count and lets the solver judge.

        An earlier version looked for an action whose every attempt was dead.
        Replayed against the failing run it never fired once in 336 actions:
        ``ACTION6`` was not dead, it was 111/157. The waste hid inside a working
        action, which is why the fraction is reported rather than a
        stop-using-this verdict.

        A repeat is not *always* waste either -- a game with hidden state can
        make a previously-inert action live.
        """
        if not self.level_dead:
            return ""
        dead, tried, repeats = self.level_dead, self.level_tried, self.level_repeats
        note = f"{dead}/{tried} actions on this level changed nothing"
        if repeats:
            note += f" ({repeats} repeated one you had already seen do nothing)"
        return note
