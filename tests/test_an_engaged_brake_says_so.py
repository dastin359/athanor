"""A concurrency of 0 means "start nothing", and it has to announce itself.

`_take_slot` reports the live limit only when it *changes* from `_last_limit`,
and `_last_limit` was initialised to `0` — the same value that means "hold". So a
driver launched against an engaged brake parked in the gate and printed nothing
at all. The one line that says why nothing is happening is suppressed in exactly
the case where it is the answer.

Measured 2026-08-09. `scratchpad/concurrency` held `0` from a launch freeze on
Aug 7; the container was replaced at 02:44, the snapshot restored the file, and a
`bp35` validation run sat silent for fifteen minutes looking like a slow start —
driver alive, ppid 1, "pass 1/2", no workspace, no solver, no error. The brake
lives in the store that reverts, so a freeze set days ago comes back from the
dead and silently holds every launch after it.

The sentinel is `None` now, which no limit can be, and 0 says what it means and
where to release it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import clean_rollouts as cr  # noqa: E402


@pytest.fixture
def gate(monkeypatch, tmp_path):
    """A fresh gate, so `_last_limit` is at its start-of-process value."""
    monkeypatch.setattr(cr, "_last_limit", None)
    monkeypatch.setattr(cr, "_running", 0)
    monkeypatch.setattr(cr, "_waiting", set())
    monkeypatch.setattr(cr, "CONCURRENCY_FILE", tmp_path / "concurrency")
    cr._aborted.clear()
    return tmp_path / "concurrency"


def _take_with_timeout(gid: str, rank: int, seconds: float) -> bool:
    """True if a slot was granted within the window."""
    import threading
    got = []

    def _run():
        try:
            cr._take_slot(gid, rank)
            got.append(True)
        except BaseException:                          # noqa: BLE001
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(seconds)
    return bool(got)


def test_a_brake_engaged_from_the_start_is_announced(gate, capsys):
    """The failure this file is named for: silent hold."""
    gate.write_text("0", encoding="utf-8")

    granted = _take_with_timeout("bp35-0a0ad940", 0, 1.5)

    assert not granted, "a limit of 0 granted a slot; the brake is not a brake"
    out = capsys.readouterr().out
    assert "concurrency = 0" in out, (
        f"the gate held without saying why. stdout was {out!r}"
    )


def test_the_hold_says_where_to_release_it(gate, capsys):
    """`concurrency = 0` alone requires knowing the file exists."""
    gate.write_text("0", encoding="utf-8")
    _take_with_timeout("bp35-0a0ad940", 0, 1.5)

    out = capsys.readouterr().out
    assert str(gate) in out, f"the refusal did not name the control file: {out!r}"
    assert "holding" in out


def test_a_normal_limit_is_announced_once_without_a_was_clause(gate, capsys):
    """First reading has nothing to have changed from."""
    gate.write_text("2", encoding="utf-8")

    assert _take_with_timeout("bp35-0a0ad940", 0, 2.0), "a limit of 2 held anyway"

    out = capsys.readouterr().out
    assert "concurrency = 2" in out
    assert "was" not in out, f"the first reading claimed a previous value: {out!r}"


def test_a_change_reports_what_it_changed_from(gate, capsys, monkeypatch):
    gate.write_text("0", encoding="utf-8")
    _take_with_timeout("bp35-0a0ad940", 0, 1.5)
    capsys.readouterr()

    gate.write_text("2", encoding="utf-8")
    monkeypatch.setattr(cr, "_waiting", set())
    assert _take_with_timeout("bp35-0a0ad940", 0, 2.0)

    out = capsys.readouterr().out
    assert "concurrency = 2 (was 0)" in out, (
        f"the release did not report what it was released from: {out!r}"
    )


def test_a_running_limit_is_not_reannounced_every_poll(gate, capsys):
    """An always-on report is one people stop reading."""
    gate.write_text("1", encoding="utf-8")
    assert _take_with_timeout("aa11-11111111", 0, 2.0)
    capsys.readouterr()

    # A second waiter at the same limit: the limit has not changed.
    cr._waiting.add(1)
    _take_with_timeout("bb22-22222222", 1, 1.2)

    assert "concurrency" not in capsys.readouterr().out, (
        "the unchanged limit was announced again"
    )


@pytest.mark.parametrize("limit", ["0", "1", "2", "8"])
def test_the_modules_own_default_is_a_sentinel_no_limit_can_be(tmp_path, limit):
    """The cases above set `_last_limit` themselves, so they cannot see the default.

    That is not pedantry: the first version of this file forced the fixture to
    `None`, and the mutant restoring the module default to `0` -- which *is* the
    original bug -- passed all five tests. A fixture that guarantees the property
    under test is the exact defect this file documents, one level up.

    So this one imports the module fresh in a subprocess and never touches the
    variable: whatever the module ships as its default is what runs.

    Parametrised over the whole range, because the invariant is "the first
    reading is always announced" rather than "0 is announced". Any sentinel a
    real limit can equal silences exactly that one value, and picking which
    value to silence is not a decision anyone should be making by accident --
    the one it silenced in production was the brake.
    """
    control = tmp_path / "concurrency"
    control.write_text(limit, encoding="utf-8")
    script = f"""
import sys, threading
sys.path.insert(0, {str(REPO / "tools")!r})
sys.path.insert(0, {str(REPO / "src")!r})
import clean_rollouts as cr
cr.CONCURRENCY_FILE = __import__("pathlib").Path({str(control)!r})
t = threading.Thread(target=lambda: cr._take_slot("bp35-0a0ad940", 0), daemon=True)
t.start()
t.join(1.5)
"""
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, timeout=120,
                          env={**os.environ, "ARC_API_KEY": "x",
                               "CCARC3_SCRATCH": str(tmp_path)})
    assert f"concurrency = {limit}" in proc.stdout, (
        f"with the module's own default, a starting limit of {limit} was never "
        f"announced. Any sentinel a real limit can equal silences that one "
        f"value, and the value it silenced in production was 0 -- the brake.\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr[-800:]!r}"
    )
