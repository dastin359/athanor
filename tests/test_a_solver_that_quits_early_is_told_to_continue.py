"""A solver that stops with actions in hand is resumed, not thrown away.

Before this, a give-up marked `error`, the driver discarded the attempt and
re-ran the game from scratch in a fresh workspace — paying again to re-derive
every level the solver had already worked out. Measured on `bp35` (2026-08-09):
attempt 1 stopped at 6 of 9 levels with 80% of its allowance unspent, cost
$23.74, and was discarded; the replacement attempt spent another $26.55 to reach
level 7 and was then killed. Nothing was banked.

A nudge continues the *same conversation*: the solver keeps its context, its
`rules.json` and its place in the game, and the cost is one more turn rather than
one more game.

**The resume targets an id the harness assigned.** Every event a solver emits
carries the *driver's* `session_id` — measured on `bp35`, all 5,233 of them —
because the child inherits `CLAUDE_CODE_SESSION_ID` and never mints its own; its
transcript on disk is literally named after the driver's session. So an id read
from the stream would resume the driver's own conversation. `--session-id` makes
each solver's id known instead of discoverable.
"""

from __future__ import annotations

import os
import pathlib
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from athanor.ccarc3 import session as sess  # noqa: E402


@pytest.fixture(autouse=True)
def _own_the_knob():
    """Restore `CCARC3_MAX_NUDGES` around every test in this file.

    `monkeypatch.delenv(name, raising=False)` records nothing when the variable
    is already absent, so a test that then calls `enable_nudging()` — which sets
    it outside monkeypatch's knowledge — leaves it set for the rest of the
    session. That is what happened: two stream-rotation tests in
    `test_ccarc3_session.py` began making three launches instead of one and
    failed on a glob returning the new empty stream first.

    Owning the variable here rather than trusting each test to is the only form
    that cannot be got wrong by the next test added to this file.
    """
    saved = os.environ.get("CCARC3_MAX_NUDGES")
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("CCARC3_MAX_NUDGES", None)
        else:
            os.environ["CCARC3_MAX_NUDGES"] = saved


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A `run_game` whose launches are recorded instead of spawned."""
    launches: list[list[str]] = []
    outcomes: list[dict] = []

    ws = types.SimpleNamespace(
        root=tmp_path,
        config=types.SimpleNamespace(
            wall_clock_timeout_s=0, model=None, effort=None, permission_mode=None,
            allowed_tools=None, disallowed_tools=None, extra_cli_args=()),
        info=types.SimpleNamespace(game_id="zz99-deadbeef"),
        session_id="11111111-2222-3333-4444-555555555555",
        resumed=False,
    )
    monkeypatch.setattr(sess, "build_workspace", lambda c, i=None: ws)
    monkeypatch.setattr(sess, "_record_resume_state", lambda w: None)
    monkeypatch.setattr(sess, "build_cli_args", lambda w: ["claude", "-p", "first"])
    monkeypatch.setattr(sess, "_supports_flag", lambda flag: True)
    monkeypatch.setattr(sess, "_claude_binary", lambda: "claude")
    monkeypatch.setattr(sess, "resolve_permission_mode", lambda m: m)

    def _launch(w, args, deadline):
        launches.append(list(args))
        return 0, False

    monkeypatch.setattr(sess, "_launch", _launch)
    monkeypatch.setattr(sess, "collect_outcome",
                        lambda w, *, exit_code, timed_out: dict(outcomes.pop(0)))
    monkeypatch.delenv("CCARC3_MAX_NUDGES", raising=False)
    return ws, launches, outcomes


def _cfg(wall_clock_timeout_s: int = 0):
    """The config `run_game` is called with.

    Deliberately a separate object from `ws.config`: in production
    `build_workspace` puts the same instance on both, and a test that conflated
    them would not notice `run_game` reading the wrong one.
    """
    return types.SimpleNamespace(wall_clock_timeout_s=wall_clock_timeout_s)


def test_a_give_up_is_nudged_rather_than_returned(harness, monkeypatch):
    ws, launches, outcomes = harness
    monkeypatch.setenv("CCARC3_MAX_NUDGES", "2")
    outcomes.extend([{"gave_up": True, "error": "quit"}, {"won": True}])

    out = sess.run_game(_cfg())

    assert len(launches) == 2, "the solver was not resumed after quitting"
    assert out.get("won") is True
    assert out["nudges"] == 1, f"the nudge count was not recorded: {out}"


def test_the_nudge_resumes_the_assigned_id_not_one_from_the_stream(harness, monkeypatch):
    """Resuming a stream-derived id would reach the driver's own conversation."""
    ws, launches, outcomes = harness
    monkeypatch.setenv("CCARC3_MAX_NUDGES", "1")
    outcomes.extend([{"gave_up": True}, {"gave_up": False}])

    sess.run_game(_cfg())

    second = launches[1]
    assert "--resume" in second, f"the second launch did not resume: {second}"
    assert second[second.index("--resume") + 1] == ws.session_id
    assert sess.NUDGE_PROMPT in second, "the nudge prompt was not sent"


def test_the_first_launch_claims_the_session_id(monkeypatch, tmp_path):
    """`--session-id` is what makes the resume above exact."""
    monkeypatch.setattr(sess, "_supports_flag", lambda flag: True)
    monkeypatch.setattr(sess, "_claude_binary", lambda: "claude")
    ws = types.SimpleNamespace(
        root=tmp_path, session_id="abc-123", initial_prompt="go",
        config=types.SimpleNamespace(
            model=None, effort=None, permission_mode=None, allowed_tools=None,
            disallowed_tools=None, extra_cli_args=()))
    args = sess.build_cli_args(ws)
    assert "--session-id" in args and args[args.index("--session-id") + 1] == "abc-123"


def test_a_run_that_did_not_give_up_is_never_nudged(harness, monkeypatch):
    ws, launches, outcomes = harness
    monkeypatch.setenv("CCARC3_MAX_NUDGES", "3")
    outcomes.append({"won": True})

    out = sess.run_game(_cfg())

    assert len(launches) == 1 and out["nudges"] == 0


def test_an_interrupted_run_is_never_nudged(harness, monkeypatch):
    """A killed solver says nothing about the game; the driver re-runs it."""
    ws, launches, outcomes = harness
    monkeypatch.setenv("CCARC3_MAX_NUDGES", "3")
    outcomes.append({"error": "solver killed by signal 15", "gave_up": False})

    sess.run_game(_cfg())

    assert len(launches) == 1, "an interrupted run was resumed as though it quit"


def test_the_bound_is_honoured_and_the_give_up_survives_it(harness, monkeypatch):
    """After the last nudge the error stands, and the driver discards as before."""
    ws, launches, outcomes = harness
    monkeypatch.setenv("CCARC3_MAX_NUDGES", "2")
    outcomes.extend([{"gave_up": True, "error": "quit"}] * 3)

    out = sess.run_game(_cfg())

    assert len(launches) == 3, f"expected 1 launch + 2 nudges, got {len(launches)}"
    assert out["nudges"] == 2
    assert out["error"], "the give-up was cleared rather than left for the driver"


def test_zero_nudges_restores_the_previous_behaviour_exactly(harness, monkeypatch):
    ws, launches, outcomes = harness
    monkeypatch.setenv("CCARC3_MAX_NUDGES", "0")
    outcomes.append({"gave_up": True, "error": "quit"})

    out = sess.run_game(_cfg())

    assert len(launches) == 1 and out["nudges"] == 0 and out["error"]


def test_a_cli_that_cannot_resume_falls_back_to_the_old_path(harness, monkeypatch):
    ws, launches, outcomes = harness
    monkeypatch.setenv("CCARC3_MAX_NUDGES", "3")
    monkeypatch.setattr(sess, "_supports_flag", lambda flag: flag != "--resume")
    outcomes.append({"gave_up": True, "error": "quit"})

    out = sess.run_game(_cfg())

    assert len(launches) == 1, "it resumed on a CLI with no --resume"
    assert out["error"], "the give-up was lost instead of left for the driver"


def test_the_wall_clock_spans_every_launch(harness, monkeypatch):
    """Each nudge must not get a fresh timeout, or a run outlives its limit.

    `clean_rollouts` picks `wall_clock_timeout_s` so a run fits inside a
    container window; giving three launches the full limit each defeats that
    choice silently.
    """
    ws, launches, outcomes = harness
    monkeypatch.setenv("CCARC3_MAX_NUDGES", "3")
    outcomes.extend([{"gave_up": True, "error": "quit"}] * 4)

    clock = {"t": 0.0}
    monkeypatch.setattr(sess.time, "monotonic", lambda: clock["t"])
    seen: list[float | None] = []

    def _launch(w, args, deadline):
        seen.append(deadline)
        clock["t"] += 60.0          # each launch burns more than half the limit
        return 0, False

    monkeypatch.setattr(sess, "_launch", _launch)

    sess.run_game(_cfg(wall_clock_timeout_s=100))

    assert len(set(seen)) == 1, f"the deadline moved between launches: {seen}"
    assert seen[0] == 100.0, f"the deadline was not the caller's limit: {seen[0]}"
    assert len(seen) == 2, (
        f"each launch burned 60s of a 100s limit, so the run must stop after two; "
        f"it made {len(seen)} — the deadline is being reset per launch"
    )


# --- the driver's opt-in must not be an import side effect ------------------- #

def test_importing_the_driver_does_not_turn_nudging_on(monkeypatch):
    """Process-wide state set at import reaches everything that imports it.

    The first version of this feature ran `os.environ.setdefault` at module
    scope in `clean_rollouts.py`. Any test that imported the driver then turned
    nudging on for the whole pytest process, and two stream-rotation tests with
    nothing to do with nudging began making three launches instead of one — they
    failed on a glob returning the new empty stream first, which took a while to
    read as env leakage rather than a rotation bug.

    `install_strip` is a function you call for exactly this reason, and
    `ablate_baselines._install_patch` says it outright: import side effects that
    rewrite another module's globals are invisible until they produce a wrong
    number.
    """
    monkeypatch.delenv("CCARC3_MAX_NUDGES", raising=False)
    sys.path.insert(0, str(REPO / "tools"))
    import importlib

    import clean_rollouts as cr
    importlib.reload(cr)

    assert os.environ.get("CCARC3_MAX_NUDGES") is None, (
        "importing the driver set process-wide state"
    )
    assert sess._max_nudges() == 0, "nudging is on for every caller after an import"


def test_the_driver_opts_in_when_it_actually_runs(monkeypatch):
    sys.path.insert(0, str(REPO / "tools"))
    import clean_rollouts as cr

    monkeypatch.delenv("CCARC3_MAX_NUDGES", raising=False)
    cr.enable_nudging()
    assert sess._max_nudges() == 2, "the driver did not enable nudging"


def test_an_operator_value_survives_the_driver(monkeypatch):
    """Including 0 — turning it off has to be possible from outside."""
    sys.path.insert(0, str(REPO / "tools"))
    import clean_rollouts as cr

    monkeypatch.setenv("CCARC3_MAX_NUDGES", "0")
    cr.enable_nudging()
    assert sess._max_nudges() == 0, "the driver overrode an explicit 0"
