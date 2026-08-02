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
    max_actions: int = 0
    """Hard action cap. 0 means uncapped.

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

    def _save_state(self) -> None:
        self.state_path.write_text(
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
        except (OSError, json.JSONDecodeError):
            return False
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
        self.level_actions = int(saved.get("level_actions", 0))
        self._last_advanced = bool(saved.get("last_advanced", False))

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
        """
        base = self.baseline_here
        if base:
            ratio = self.level_actions / base
            pace = f" [{self.level_actions}/{base} on this level = {ratio:.1f}x]"
            # 1.0, not the 2.0 first shipped here. Over 26 level-attempts, 24 of
            # 25 cleared levels finished at or under 0.92x and the median was
            # 0.52x, so crossing 1.0 is already the unusual case. No cutpoint in
            # 1.0-3.4 fits the data better than any other -- the sample is empty
            # in between -- and warning at the bottom of that gap costs a re-read
            # while warning at the top costs the hundreds of actions in between.
            if ratio >= 1.0:
                pace += "  <- OVER BASELINE: re-explore rather than grind"
        else:
            pace = ""
        return (
            f"{self.game_id}: level {self.level}/{self.win_levels or '?'} "
            f"state={self.state} actions={self.actions_used}{pace}"
            f"{f', wasted={self.wasted_actions}' if self.wasted_actions else ''}"
            f"{f', FULL RESETS={self.full_resets}' if self.full_resets else ''}"
        )
