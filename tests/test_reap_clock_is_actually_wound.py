"""The reap clock must be WOUND by a server answer, not merely read.

`test_the_stamp_advances_only_when_the_server_answered` covers one direction:
`_save_state` must not renew `last_touched`, because it runs on paths with no
server contact. Nothing covered the other direction -- that `_send` sets it at
all. Found by mutation: deleting `self.last_touched = time.time()` from `_send`
left all 1076 tests green.

That is not a cosmetic gap. With the stamp never written, `last_touched` stays
0.0, and the guard reads

    if self.last_touched:          # falsy
        idle = ...

so the reap check is skipped entirely, on every run. A guard that cannot fire is
the failure this project keeps finding, and here it protects the expensive case:
an idle game is reaped in (13.6, 18.2] minutes, and on a shared-card sweep a
resume onto a reaped game replays from level 0 with actions already banked.

Both halves are asserted here so neither can drift: the stamp advances on a
server answer, and the guard it arms actually fires once the gap is past the
deadline.
"""

from __future__ import annotations

import json
import time

import pytest

from athanor.ccarc3 import client as client_mod
from athanor.ccarc3.client import ArcClient

from test_ccarc3_client import _frame          # the module's own frame builder


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "k")
    monkeypatch.setattr(
        client_mod, "_post",
        lambda *a, **k: _frame(frame=[[[2]]], levels_completed=0, available_actions=[1]),
    )
    monkeypatch.setattr(client_mod, "_get",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("no network in this test")))
    c = ArcClient("g", trace_path=tmp_path / "run" / "t.jsonl")
    c.card_id = "card-1"
    return c


def test_a_server_answer_winds_the_clock(tmp_path, monkeypatch) -> None:
    c = _client(tmp_path, monkeypatch)
    assert c.last_touched == 0.0, "fixture already wound; the test would be vacuous"

    before = time.time()
    c.act(1)

    assert c.last_touched >= before, (
        "a frame came back from the server and the reap clock was not wound. "
        "`last_touched` stays 0.0, so the reap guard's `if self.last_touched:` "
        "is falsy and the check never runs on any resume."
    )
    assert c.last_touched <= time.time()


def test_a_stale_clock_is_what_makes_the_guard_fire(tmp_path, monkeypatch) -> None:
    """The consequence, so the two halves cannot drift apart.

    Winding the clock is only worth asserting because something reads it. This
    pins that the value written is the value the deadline is measured against.
    """
    c = _client(tmp_path, monkeypatch)
    c.act(1)
    wound = c.last_touched
    assert wound > 0

    c.last_touched = time.time() - (ArcClient.REAP_DEADLINE_S + 60)
    c._resumed = True
    c.scorecard = lambda: {"cards": {"g": {"levels_completed": []}}}  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="reap deadline"):
        c.open()


def test_the_clock_survives_a_save_and_reload(tmp_path, monkeypatch) -> None:
    """A resumed run inherits the gap; it must not restart the clock at zero.

    If the stamp did not persist, every resume would look freshly touched and
    the guard would pass on a game that had been idle for hours.
    """
    c = _client(tmp_path, monkeypatch)
    c.act(1)
    wound = c.last_touched
    c._save_state()

    saved = json.loads((tmp_path / "run" / "t.state.json").read_text())
    assert saved["last_touched"] == pytest.approx(wound, abs=1.0), (
        "the wound clock did not reach the state file, so a resumed run would "
        "measure its idle gap from zero"
    )
