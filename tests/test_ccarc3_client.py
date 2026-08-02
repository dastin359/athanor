"""Tests for the live ARC-AGI-3 client.

The refusal paths carry the weight here. Both guard against traps that cost real
progress during development and are invisible from the outside: a RESET issued
one action too early silently discards a whole game, and any action taken while
dead is billed but never executed.
"""

from __future__ import annotations

import json

import pytest

from athanor.ccarc3 import ActionRefused, ArcClient, GameInfo
from athanor.ccarc3 import client as client_mod


@pytest.fixture
def stub(monkeypatch, tmp_path):
    """An ArcClient wired to a scriptable fake server."""
    sent: list[tuple[str, dict]] = []
    replies: list[dict] = []

    def fake_post(url, payload, key, **kw):
        sent.append((url, payload))
        if url.endswith("/scorecard/open"):
            return {"card_id": "card-1"}
        if url.endswith("/scorecard/close"):
            return {"closed": True}
        reply = replies.pop(0) if replies else _frame()
        # The real server echoes the action it executed back in action_input.
        # A fake that hardcodes it would hide any bug in what we actually sent.
        name = url.rsplit("/", 1)[-1]
        reply = dict(reply)
        reply["action_input"] = {
            **reply.get("action_input", {}),
            "id": 0 if name == "RESET" else int(name.removeprefix("ACTION")),
        }
        return reply

    monkeypatch.setattr(client_mod, "_post", fake_post)
    monkeypatch.setenv("ARC_API_KEY", "test-key")
    c = ArcClient("g1", trace_path=tmp_path / "t.jsonl").open()
    return c, sent, replies


def _frame(**kw):
    base = {
        "game_id": "g1",
        "frame": [[[1, 1], [1, 1]]],
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "win_levels": 3,
        "action_input": {"id": 1, "data": {}, "reasoning": None},
        "guid": "guid-1",
        "full_reset": False,
        "available_actions": [1, 2, 3, 4],
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# the refusals
# --------------------------------------------------------------------------- #


def test_reset_right_after_a_level_advance_is_refused(stub):
    """The trap: that one RESET does a full game reset and discards everything."""
    c, sent, replies = stub
    replies.append(_frame(levels_completed=1))  # this action advanced a level
    c.act(1)
    assert c.level == 1

    before = len(sent)
    with pytest.raises(ActionRefused, match="FULL GAME RESET"):
        c.reset()
    assert len(sent) == before, "a refused action must not reach the server"


def test_the_same_reset_is_allowed_after_any_other_action(stub):
    c, sent, replies = stub
    replies.extend([_frame(levels_completed=1), _frame(levels_completed=1), _frame()])
    c.act(1)
    c.act(2)  # anything at all clears the condition
    c.reset()
    assert sent[-1][0].endswith("/cmd/RESET")


def test_force_full_overrides_the_refusal(stub):
    c, _, replies = stub
    replies.extend([_frame(levels_completed=1), _frame(full_reset=True)])
    c.act(1)
    c.reset(force_full=True)
    assert c.full_resets == 1


def test_acting_while_dead_is_refused(stub):
    c, sent, replies = stub
    replies.append(_frame(state="GAME_OVER"))
    c.act(1)
    assert c.dead

    before = len(sent)
    with pytest.raises(ActionRefused, match="costs budget"):
        c.act(2)
    assert len(sent) == before


def test_reset_still_works_while_dead(stub):
    c, _, replies = stub
    replies.extend([_frame(state="GAME_OVER"), _frame(state="NOT_FINISHED")])
    c.act(1)
    c.reset()
    assert not c.dead


def test_actions_outside_available_actions_are_refused(stub):
    c, _, replies = stub
    replies.append(_frame(available_actions=[1, 2]))
    c.act(1)
    with pytest.raises(ActionRefused, match="not in this game's available_actions"):
        c.act(4)


def test_action6_requires_in_range_coordinates(stub):
    c, _, replies = stub
    replies.append(_frame(available_actions=[6]))
    c.act(6, x=1, y=1)
    with pytest.raises(ActionRefused, match="requires x and y"):
        c.act(6)
    with pytest.raises(ActionRefused, match=r"\[0, 63\]"):
        c.act(6, x=64, y=0)


# --------------------------------------------------------------------------- #
# state tracking
# --------------------------------------------------------------------------- #


def test_level_is_read_from_levels_completed_not_score(stub):
    """The live server sends levels_completed; the stock SDK reads score."""
    c, _, replies = stub
    replies.append(_frame(levels_completed=2))
    c.act(1)
    assert c.level == 2
    assert c.win_levels == 3


def test_empty_frames_are_counted_as_wasted(stub):
    c, _, replies = stub
    replies.append(_frame(frame=[], state="GAME_OVER"))
    c.act(1)
    assert c.wasted_actions == 1


def test_every_action_lands_in_the_ledger(stub):
    c, _, replies = stub
    replies.extend([_frame(), _frame()])
    c.act(1)
    c.act(2)
    assert [t.action for t in c.transitions()] == ["ACTION1", "ACTION2"]


def test_status_surfaces_full_resets_because_they_are_silent_otherwise(stub):
    c, _, replies = stub
    replies.extend([_frame(levels_completed=1), _frame(full_reset=True)])
    c.act(1)
    c.reset(force_full=True)
    assert "FULL RESETS=1" in c.status()


# --------------------------------------------------------------------------- #
# pace against the published baseline — the control law, mechanised
# --------------------------------------------------------------------------- #


@pytest.fixture
def paced(monkeypatch, tmp_path):
    """A client that knows its own per-level baselines, as a live one does."""
    def fake_post(url, payload, key, **kw):
        if url.endswith("/scorecard/open"):
            return {"card_id": "card-1"}
        reply = _frame(**getattr(fake_post, "next", {}))
        name = url.rsplit("/", 1)[-1]
        reply["action_input"] = {"id": 0 if name == "RESET" else int(name.removeprefix("ACTION"))}
        return reply

    monkeypatch.setattr(client_mod, "_post", fake_post)
    monkeypatch.setenv("ARC_API_KEY", "k")
    info = GameInfo("g1", baseline_actions=(10, 20))
    return ArcClient("g1", trace_path=tmp_path / "t.jsonl", info=info).open(), fake_post


def test_status_reports_this_level_against_its_own_baseline(paced):
    c, _ = paced
    for _ in range(5):
        c.act(1)
    assert "5/10 on this level = 0.5x" in c.status()


def test_going_well_over_baseline_says_so_in_words(paced):
    """The doctrine said re-explore rather than grind; nothing ever said when.

    A run that never left level 0 spent 6.1x that level's baseline, and no
    surface in the harness reported the number while it was happening."""
    c, _ = paced
    for _ in range(20):
        c.act(1)
    s = c.status()
    assert "= 2.0x" in s and "WELL OVER BASELINE" in s


def test_pace_is_quiet_while_the_level_is_going_well(paced):
    c, _ = paced
    for _ in range(3):
        c.act(1)
    assert "WELL OVER BASELINE" not in c.status()


def test_the_per_level_count_restarts_when_the_level_does(paced):
    """Cumulative actions cannot be compared to a per-level baseline."""
    c, post = paced
    for _ in range(4):
        c.act(1)
    post.next = {"levels_completed": 1}
    c.act(1)
    assert c.level_actions == 0 and c.actions_used == 5
    post.next = {"levels_completed": 1}
    c.act(1)
    assert c.level_actions == 1
    assert "1/20 on this level" in c.status(), "the baseline is level 1's now"


def test_a_level_past_the_published_baselines_reports_no_pace(paced):
    c, post = paced
    post.next = {"levels_completed": 9}
    c.act(1)
    assert "on this level" not in c.status(), "invented baselines are worse than none"


def test_the_reset_trap_stays_armed_across_a_process_boundary(monkeypatch, tmp_path):
    """The bug that cost a finished run, reconstructed from its trace.

    `ls20` trace index 369 advanced level 5 -> 6; index 370 was a RESET and the
    level read 0. The refusal exists precisely for that, and could not fire: it
    is armed by `_last_advanced`, which lived only in memory, and a CC solver
    takes every action in a new process. 370 actions of progress were replayed.
    """
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    a = ArcClient("g", trace_path=path)
    a.card_id, a.level, a.actions_used = "card-1", 6, 370
    a._last_advanced = True                       # it just cleared a level
    a._save_state()

    b = ArcClient("g", trace_path=path)
    assert b._last_advanced, "the flag must survive, or the refusal is decorative"
    with pytest.raises(ActionRefused, match="FULL GAME RESET"):
        b.reset()


def test_a_level_going_down_is_a_full_reset_whatever_the_server_says(stub):
    """The server reported `full_reset: False` on a 6 -> 0 transition.

    So a counter that trusts the flag reports "zero full resets" for a run that
    replayed the entire game — which is exactly what the losing run reported.
    """
    c, _, replies = stub
    replies.extend([_frame(levels_completed=6), _frame(levels_completed=0, full_reset=False)])
    c.act(1)
    c.reset(force_full=True)
    assert c.full_resets == 1
    assert "FULL RESETS=1" in c.status()


def test_a_full_reset_restarts_the_per_level_count(stub):
    """Level 0 begins again; carrying the old level's tally makes the pace lie."""
    c, _, replies = stub
    replies.extend([_frame(), _frame(), _frame(levels_completed=0, full_reset=True)])
    c.act(1)
    c.act(1)
    assert c.level_actions == 2
    c.reset(force_full=True)
    assert c.level_actions == 0


def test_the_per_level_count_survives_a_new_process(monkeypatch, tmp_path):
    """Every action is a new process, so a counter that lives only in memory
    resets to zero on each one and the ratio reads 1/10 forever."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    a = ArcClient("g", trace_path=path)
    a.card_id, a.actions_used, a.level_actions = "card-1", 30, 17
    a._save_state()

    b = ArcClient("g", trace_path=path)
    assert b.level_actions == 17


def test_a_stale_trace_is_never_inherited(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    path.write_text('{"i":0,"action":"RESET","frames":[[[1]]],"score":0}\n')
    ArcClient("g", trace_path=path)
    assert not path.exists() or path.read_text() == ""


def test_missing_api_key_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("ARC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="401"):
        ArcClient("g", trace_path=tmp_path / "t.jsonl")


# --------------------------------------------------------------------------- #
# transport — both of these guard bugs that only showed up against the live API
# --------------------------------------------------------------------------- #


def test_each_client_carries_its_own_cookie_jar(monkeypatch, tmp_path):
    """The API binds a scorecard to the HTTP session, not to the API key.

    Without a cookie jar the server answers 'game <id> not found' for a game it
    just listed, because the RESET arrives on a different session than the
    scorecard was opened on.
    """
    import urllib.request

    monkeypatch.setenv("ARC_API_KEY", "k")
    a = ArcClient("g", trace_path=tmp_path / "a.jsonl")
    b = ArcClient("g", trace_path=tmp_path / "b.jsonl")

    def jar_of(c):
        (proc,) = [
            h for h in c._opener.handlers
            if isinstance(h, urllib.request.HTTPCookieProcessor)
        ]
        return proc.cookiejar

    assert jar_of(a) is not jar_of(b), "clients must not share a session"


def test_a_4xx_surfaces_the_server_message_not_just_the_code(monkeypatch):
    """'HTTP Error 400: Bad Request' alone is undebuggable; the reason is in the body."""
    import io
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {},
            io.BytesIO(b'{"error":"SERVER_ERROR","message":"game xyz not found"}'),
        )

    monkeypatch.setattr(client_mod.urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="game xyz not found"):
        client_mod._get("https://example.test/api/games", "k", retries=1)


def test_a_failed_close_never_masks_the_real_error(monkeypatch, tmp_path):
    """A cleanup 404 once buried the actual exception from a run."""
    monkeypatch.setenv("ARC_API_KEY", "k")

    def fake_post(url, payload, key, **kw):
        if url.endswith("/scorecard/open"):
            return {"card_id": "c"}
        raise RuntimeError("404: scorecard not found")

    monkeypatch.setattr(client_mod, "_post", fake_post)
    c = ArcClient("g", trace_path=tmp_path / "t.jsonl")
    with pytest.raises(ValueError, match="the real problem"):
        with c:
            raise ValueError("the real problem")
    assert "scorecard not found" in c.close_error


# --------------------------------------------------------------------------- #
# GameInfo
# --------------------------------------------------------------------------- #


def test_gameinfo_derives_a_budget_from_the_published_baseline():
    g = GameInfo("x", baseline_actions=(22, 123, 73))
    assert g.levels == 3
    assert g.baseline_total == 218
    assert g.baseline_for(1) == 123
    assert g.baseline_for(9) is None
    assert g.suggested_budget(4.0) == 872


def test_suggested_budget_has_a_floor_for_very_short_games():
    assert GameInfo("x", baseline_actions=(5,)).suggested_budget() == 200


def test_a_flat_cap_of_80_would_not_finish_any_real_game():
    """The measured range across the 25 public games is 171..1843."""
    shortest = GameInfo("cd82", baseline_actions=(55, 30, 30, 20, 20, 16))
    assert shortest.baseline_total > 80


def test_the_action_budget_is_enforced_not_merely_advertised(stub):
    """A cap the solver is only told about is not a cap."""
    c, sent, replies = stub
    c.max_actions = 2
    replies.extend([_frame(), _frame()])
    c.act(1)
    c.act(2)
    before = len(sent)
    with pytest.raises(ActionRefused, match="budget exhausted"):
        c.act(1)
    assert len(sent) == before


def test_an_uncapped_client_is_the_default(stub):
    c, _, replies = stub
    assert c.max_actions == 0
    replies.extend([_frame() for _ in range(5)])
    for _ in range(5):
        c.act(1)
    assert c.actions_used == 5


# --------------------------------------------------------------------------- #
# cross-process resumption — what makes the client usable by a CC solver at all
# --------------------------------------------------------------------------- #


def test_a_new_process_resumes_the_same_game(monkeypatch, tmp_path):
    """A CC solver drives this with one-shot `python -c`, so every action is a
    new process. Without resumption each one opened a fresh scorecard and wiped
    the trace, and no run could get past its first action."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"

    a = ArcClient("g", trace_path=path)
    a.card_id, a.guid, a.actions_used, a.level = "card-1", "guid-9", 12, 3
    a.available_actions = ("ACTION1",)
    a._save_state()

    b = ArcClient("g", trace_path=path)
    assert b._resumed
    assert (b.card_id, b.guid, b.actions_used, b.level) == ("card-1", "guid-9", 12, 3)
    assert b.available_actions == ("ACTION1",)


def test_resuming_does_not_open_a_second_scorecard(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "k")
    opened = []

    def fake_post(url, payload, key, **kw):
        opened.append(url)
        return {"card_id": "new-card"}

    monkeypatch.setattr(client_mod, "_post", fake_post)
    path = tmp_path / "t.jsonl"
    a = ArcClient("g", trace_path=path)
    a.card_id = "card-1"
    a._save_state()

    b = ArcClient("g", trace_path=path).open()
    assert b.card_id == "card-1", "resume must keep the in-flight game"
    assert not opened, "opening a second scorecard abandons the run"


def test_a_resumed_client_does_not_wipe_the_trace(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    a = ArcClient("g", trace_path=path)
    a.actions_used = 1
    a._save_state()
    path.write_text('{"i":0,"action":"RESET","frames":[[[1]]],"score":0,"level":0}\n')

    ArcClient("g", trace_path=path)
    assert path.read_text().strip(), "the trace is the record; resuming must keep it"


def test_state_from_a_different_game_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    a = ArcClient("game-a", trace_path=path)
    a.card_id = "card-1"
    a._save_state()
    b = ArcClient("game-b", trace_path=path)
    assert not b._resumed and b.card_id == ""


def test_corrupt_state_falls_back_to_a_fresh_client(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    path.with_suffix(".state.json").write_text("{not json")
    c = ArcClient("g", trace_path=path)
    assert not c._resumed


def test_closing_clears_the_state_so_the_next_run_starts_clean(stub, tmp_path):
    c, _, _ = stub
    c._save_state()
    assert c.state_path.exists()
    c.close()
    assert not c.state_path.exists()


def test_state_is_written_after_every_action(stub):
    c, _, replies = stub
    replies.append(_frame(levels_completed=1))
    c.act(1)
    saved = json.loads(c.state_path.read_text())
    assert saved["actions_used"] == 1 and saved["level"] == 1
