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
    with pytest.raises(ActionRefused, match="discarded without stepping the game"):
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


def test_going_over_baseline_says_so_in_words(paced):
    """The doctrine said re-explore rather than grind; nothing ever said when.

    A run that never left level 0 spent 6.1x that level's baseline, and no
    surface in the harness reported the number while it was happening."""
    c, _ = paced
    for _ in range(10):
        c.act(1)
    s = c.status()
    assert "= 1.0x" in s and "OVER BASELINE" in s


def test_pace_is_quiet_while_the_level_is_going_well(paced):
    """Quiet is the common case: 24 of 25 cleared levels finished under 1.0x."""
    c, _ = paced
    for _ in range(9):
        c.act(1)
    assert "OVER BASELINE" not in c.status()


def test_status_reports_actions_that_changed_nothing(stub):
    """In `status()` because that is the only surface solvers demonstrably read.

    Across five runs and 332 commands, every analytical helper this package
    exports was called zero times; `status()` was called 82.
    """
    av = [1, 6]
    c, _, replies = stub
    replies.append(_frame(frame=[[[9, 9], [9, 9]]], available_actions=av))
    replies.extend(_frame(available_actions=av) for _ in range(7))
    c.act(1)                      # first transition: no predecessor, not evidence
    c.act(1)                      # 9 -> 1: changes the board
    for _ in range(6):
        c.act(6, x=0, y=0)        # identical frame every time: no effect
    s = c.status()
    assert "6/7 actions on this level changed nothing" in s
    assert "5 repeated one you had already seen do nothing" in s


def test_a_run_where_everything_works_says_nothing_about_it(stub):
    """Silence is the common case: four wins ran at 845/845, 351/351, 76/76 and
    69/69. A fifth won while wasting 8 of 114, so waste is not disqualifying --
    but a clean run must not be nagged."""
    c, _, replies = stub
    replies.extend(_frame(frame=[[[i, i], [i, i]]]) for i in range(5))
    for _ in range(5):
        c.act(1)
    assert "changed nothing" not in c.status()


def test_waste_spread_across_a_working_action_is_still_reported(stub):
    """The failure this exists for, in miniature.

    A first version looked for an action whose *every* attempt was dead.
    Replayed against the run that never cleared a level it never fired once in
    336 actions: ACTION6 was not dead, it was 111/157. The waste hid inside a
    working action, so the fraction is what gets reported.
    """
    c, _, replies = stub
    av = [6]
    replies.extend([
        _frame(frame=[[[1, 1]]], available_actions=av),
        _frame(frame=[[[2, 2]]], available_actions=av),   # works
        _frame(frame=[[[2, 2]]], available_actions=av),   # (1,0) does nothing
        _frame(frame=[[[2, 2]]], available_actions=av),   # (2,0) does nothing too
        _frame(frame=[[[3, 3]]], available_actions=av),   # works again
    ])
    for x in range(5):
        c.act(6, x=x, y=0)
    s = c.status()
    assert "2/4 actions on this level changed nothing" in s
    # Two dead clicks at *different* coordinates. Keying ACTION6 by name alone
    # would call the second a repeat; only (x, y) keying gets this right, and
    # the previous version of this test used one dead click, so `repeats` was 0
    # under either keying and the assertion could not fail.
    assert "repeated" not in s, "distinct coordinates are not a repeat"


def test_the_same_dead_click_twice_is_a_repeat(stub):
    """The other half: identical coordinates must be recognised."""
    c, _, replies = stub
    av = [6]
    replies.extend([_frame(frame=[[[1]]], available_actions=av)] * 4)
    for _ in range(4):
        c.act(6, x=7, y=7)
    assert "2 repeated one you had already seen do nothing" in c.status()


def test_the_scorecard_is_captured_during_the_game_not_after(monkeypatch, tmp_path):
    """Snapshotting at the end is too late — the card is already gone.

    `vc33`'s card returned 404 about four minutes after its run finished, so
    both earlier hooks (`close()` and `collect_outcome`) could miss. A level
    advance is a moment the card certainly still exists.
    """
    monkeypatch.setenv("ARC_API_KEY", "k")
    replies = []
    monkeypatch.setattr(client_mod, "_post",
                        lambda *a, **k: replies.pop(0) if replies else _frame())
    card = {"environments": [{"runs": [{"level_actions": [3, 0]}]}]}
    gets = []
    monkeypatch.setattr(client_mod, "_get",
                        lambda *a, **k: gets.append(a) or card)

    c = ArcClient("g", trace_path=tmp_path / "run" / "t.jsonl")
    c.card_id = "real-card-xyz"          # not a stub, so the snapshot is live
    av = [1]
    replies.extend([_frame(frame=[[[1]]], available_actions=av),
                    _frame(frame=[[[2]]], levels_completed=1, available_actions=av)])
    c.act(1)
    assert not gets, "no snapshot mid-level; that would be one GET per action"
    c.act(1)                              # this one advances the level
    assert gets, "a level advance must capture the card while it still exists"
    assert json.loads((tmp_path / "run" / "scorecard.json").read_text()) == card


def test_a_terminal_state_captures_the_scorecard(monkeypatch, tmp_path):
    """WIN is the single most valuable moment to hold the card, and the run may
    end immediately after it."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    monkeypatch.setattr(client_mod, "_post",
                        lambda *a, **k: _frame(frame=[[[9]]], state="WIN",
                                               available_actions=[1]))
    gets = []
    monkeypatch.setattr(client_mod, "_get", lambda *a, **k: gets.append(a) or {"ok": 1})

    c = ArcClient("g", trace_path=tmp_path / "run" / "t.jsonl")
    c.card_id = "real-card-xyz"
    c.act(1)
    assert gets, "reaching WIN must capture the card"


def test_a_stub_card_never_reaches_the_network(monkeypatch, tmp_path):
    """Without this guard every test that plays a game fires a real GET on each
    level advance -- swallowed by the error handler, so it surfaces only as a
    slower suite and a test run that quietly depends on the network."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    monkeypatch.setattr(client_mod, "_post",
                        lambda *a, **k: _frame(frame=[[[2]]], levels_completed=1,
                                               available_actions=[1]))
    def boom(*a, **k):
        raise AssertionError("a stub card must never issue a request")
    monkeypatch.setattr(client_mod, "_get", boom)

    c = ArcClient("g", trace_path=tmp_path / "run" / "t.jsonl")
    c.card_id = "card-1"
    assert c.level == 0
    c.act(1)

    # **The guard is only exercised on a level advance**, which is what fires the
    # GET this test forbids. Without asserting the advance happened, an `act`
    # that short-circuited would take the same path as a passing run: no request
    # is issued either way, and `boom` never gets the chance to fire.
    assert c.level == 1, (
        f"the stub reported levels_completed=1 but the client sits at level "
        f"{c.level} -- the level-advance path never ran, so nothing here tested "
        f"the no-network guard"
    )


def test_closing_keeps_the_server_scorecard(monkeypatch, tmp_path):
    """The card is gone the instant it closes -- two card_ids from finished runs
    both 404'd while the same key still listed all 25 games. Close is the last
    moment the server's own actions_by_level can be read, and every run this
    project completed before this test threw it away."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    card = {"card_id": "c1", "score": 6,
            "actions_by_level": [[10, 20, 30], [8, 15, 25]]}
    monkeypatch.setattr(client_mod, "_get", lambda *a, **k: card)
    monkeypatch.setattr(client_mod, "_post", lambda *a, **k: {})

    c = ArcClient("g", trace_path=tmp_path / "run" / "t.jsonl")
    c.card_id = "c1"
    c.close()

    saved = json.loads((tmp_path / "run" / "scorecard.json").read_text())
    assert saved == card
    assert saved["actions_by_level"] == [[10, 20, 30], [8, 15, 25]], (
        "one row per play is the only evidence that can settle which play scores"
    )
    assert not c.card_id, "the card must still be closed"


def test_a_scorecard_that_cannot_be_read_does_not_block_the_close(monkeypatch, tmp_path):
    """Bookkeeping must never replace the result. A 404 from cleanup once buried
    the real exception from a run; the reason is recorded, not raised."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    def boom(*a, **k):
        raise RuntimeError("404: card_id not found")
    monkeypatch.setattr(client_mod, "_get", boom)
    closed = {}
    monkeypatch.setattr(client_mod, "_post",
                        lambda url, payload, *a, **k: closed.setdefault("hit", url) or {})

    c = ArcClient("g", trace_path=tmp_path / "run" / "t.jsonl")
    c.card_id = "c1"
    # Assigning `card_id` is how this suite says "has a card"; since shared cards
    # landed it also has to say WHOSE. A card the client opened is one it closes.
    c._owns_card = True
    c.close()

    assert "404" in c.scorecard_error
    assert "scorecard/close" in closed["hit"], "the close must still happen"
    assert not (tmp_path / "run" / "scorecard.json").exists()


def test_cycling_between_boards_is_counted_even_though_nothing_is_a_no_op(stub):
    """The failure that lost `tn36`, which the no-op tally could not see.

    309 actions on a 55-baseline level, **8** of them no-ops -- so ``level_dead``
    stayed near zero while a third of the actions put the board back somewhere it
    had already been. Every action "worked"; the level still never fell.
    """
    c, _, replies = stub
    av = [1]
    # A -> B -> A -> B: four real changes, zero no-ops, two returns to a
    # board already seen.
    replies.extend([_frame(frame=[[[g]]], available_actions=av)
                    for g in (1, 2, 1, 2)])
    for _ in range(4):
        c.act(1)
    assert c.level_dead == 0, "every action changed the board"
    assert c.level_revisits == 2
    assert "returned the board to a state already seen" in c.status()
    assert "changed nothing" not in c.status()


def test_reaching_new_boards_is_not_counted_however_long_it_takes(stub):
    """The control that makes the signal worth having.

    `su15` cleared its level 5 at 1.52x baseline with zero revisits, while
    `tn36` failed its at 5.62x with 33%. Both tripped the same 1.0x pace
    warning, so the ratio alone does not separate exploring from being stuck --
    and a signal that fires on every long level would separate nothing either.
    """
    c, _, replies = stub
    av = [1]
    replies.extend([_frame(frame=[[[g]]], available_actions=av) for g in range(20)])
    for _ in range(20):
        c.act(1)
    assert c.level_revisits == 0
    assert "already seen" not in c.status()


def test_the_revisit_tally_survives_a_new_process(monkeypatch, tmp_path):
    """Every solver action is a new process, and an unpersisted counter reads
    zero in each one -- the exact shape of the bug that replayed a won game."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    a = ArcClient("g", trace_path=path)
    a.card_id = "card-1"
    a.level_tried, a.level_revisits = 90, 30
    a._seen_keys, a._last_frame_key = {"aa", "bb"}, "aa"
    a._save_state()

    b = ArcClient("g", trace_path=path)
    assert (b.level_tried, b.level_revisits) == (90, 30)
    assert b._seen_keys == {"aa", "bb"}
    assert "30/90 actions returned the board to a state already seen" in b.status()


def test_the_revisit_tally_restarts_with_the_level(stub):
    """Boards from a cleared level are not states of the new one; carrying them
    over would report revisits against a board that no longer exists."""
    c, _, replies = stub
    av = [1]
    replies.extend([_frame(frame=[[[1]]], available_actions=av),
                    _frame(frame=[[[2]]], available_actions=av),
                    _frame(frame=[[[1]]], available_actions=av),
                    _frame(frame=[[[9]]], levels_completed=1, available_actions=av),
                    _frame(frame=[[[1]]], levels_completed=1, available_actions=av)])
    c.act(1); c.act(1); c.act(1)
    assert c.level_revisits == 1
    c.act(1)                                  # clears the level
    assert c.level_revisits == 0
    c.act(1)                                  # board [[1]] again -- but a new level
    assert c.level_revisits == 0, "a board from the previous level is not a revisit"


def test_the_waste_tally_restarts_with_the_level(stub):
    """It is a per-level figure; carrying it across makes the next level lie."""
    c, _, replies = stub
    av = [1]
    replies.extend([_frame(frame=[[[1]]], available_actions=av),
                    _frame(frame=[[[1]]], available_actions=av),
                    _frame(frame=[[[2]]], levels_completed=1, available_actions=av),
                    _frame(frame=[[[3]]], levels_completed=1, available_actions=av)])
    c.act(1)
    c.act(1)                                  # no effect
    assert c.level_dead == 1
    c.act(1)                                  # clears the level
    assert (c.level_dead, c.level_tried) == (0, 0)
    c.act(1)
    assert "changed nothing" not in c.status()


def test_the_waste_tally_survives_a_new_process(monkeypatch, tmp_path):
    """Counters live only in memory otherwise, and every action is a new process."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    a = ArcClient("g", trace_path=path)
    a.card_id = "card-1"
    a.level_tried, a.level_dead, a.level_repeats = 41, 4, 2
    a._dead_keys, a._last_frame_key = ["ACTION6:1,2"], "abc"
    a._save_state()

    b = ArcClient("g", trace_path=path)
    assert (b.level_tried, b.level_dead, b.level_repeats) == (41, 4, 2)
    assert b._dead_keys == ["ACTION6:1,2"] and b._last_frame_key == "abc"
    assert "4/41 actions on this level changed nothing" in b.status()


def test_facts_and_warnings_do_not_run_into_each_other(monkeypatch, tmp_path):
    """Interleaved, they read as one sentence: `<- OVER BASELINE: re-explore
    rather than grind, wasted=3, FULL RESETS=1`, where two unrelated counters
    look like the tail of an instruction."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    c = ArcClient("g", trace_path=tmp_path / "t.jsonl",
                  info=GameInfo("g", baseline_actions=(10,)))
    c.level_actions, c.wasted_actions, c.full_resets = 20, 3, 1
    c.level_tried, c.level_dead = 41, 4
    head, *warns = c.status().splitlines()
    assert "wasted=3" in head and "FULL RESETS=1" in head
    assert all(w.strip().startswith("<-") for w in warns)
    assert len(warns) == 2, "one line per warning, not one run-on line"


def test_status_works_on_a_client_that_has_done_nothing(monkeypatch, tmp_path):
    """status() is the solver's orientation call, and the first thing it does.

    Its extras must degrade rather than raise: no game info, no baselines, no
    actions taken, no trace on disk yet.
    """
    monkeypatch.setenv("ARC_API_KEY", "k")
    s = ArcClient("g", trace_path=tmp_path / "t.jsonl").status()
    assert "g: level 0" in s
    assert "on this level" not in s and "changed nothing" not in s


def test_pace_reads_the_baselines_off_the_client(paced):
    """A bare `level_pace(ts, baselines)` names something the solver has not got."""
    c, post = paced
    for _ in range(4):
        c.act(1)
    post.next = {"levels_completed": 1}
    c.act(1)
    post.next = {"levels_completed": 1}
    c.act(1)
    assert c.pace() == {0: (4, 10, 0.4), 1: (2, 20, 0.1)}


def test_pace_without_a_game_info_is_empty_rather_than_wrong(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "k")
    assert ArcClient("g", trace_path=tmp_path / "t.jsonl").pace() == {}


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
    with pytest.raises(ActionRefused, match="ceiling reached"):
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


def test_an_unreadable_state_file_refuses_rather_than_discarding_the_game(monkeypatch, tmp_path):
    """This test previously asserted the opposite, and the opposite is dangerous.

    Falling back to "no previous run" makes the caller delete `trace.jsonl` and
    open a fresh scorecard — throwing away a game because a small sidecar file
    became unreadable. The state file's *presence* says a run exists; only its
    details are lost, and those are far cheaper than the run.
    """
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    path.write_text('{"i":0,"action":"RESET","frames":[[[1]]],"score":0,"level":0}\n')
    path.with_suffix(".state.json").write_text("{not json")

    with pytest.raises(RuntimeError, match="cannot be read"):
        ArcClient("g", trace_path=path)
    assert path.read_text().strip(), "the trace must survive an unreadable state file"


def test_state_is_written_atomically(monkeypatch, tmp_path):
    """A reader must never observe a half-written state file."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    c = ArcClient("g", trace_path=path)
    c.card_id = "card-1"
    c._save_state()
    assert json.loads(c.state_path.read_text())["card_id"] == "card-1"
    assert not c.state_path.with_suffix(".json.tmp").exists(), "the temp file is renamed away"


def test_every_refusal_still_refuses_in_a_fresh_process(monkeypatch, tmp_path):
    """The check design note §9.9 asks for, and the one that was missing.

    Tests that a refusal *fires* all passed while the RESET guard was silently
    unarmed on resume, because they built the state in the same process that
    checked it. This crosses the boundary first. Anything added to the client
    that arms a refusal and is not persisted fails here rather than in a run.
    """
    from athanor.ccarc3 import GateRefusal, LevelGate

    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"

    a = ArcClient("g", trace_path=path, max_actions=50, gate=LevelGate(rulebook_path=tmp_path / "r.json"))
    a.card_id = "card-1"
    a.available_actions = ("ACTION1",)
    a.actions_used, a.level = 50, 4      # budget spent
    a._last_advanced = True              # RESET here is a full game reset
    a.gate.last_level, a.gate.pending_level = 4, 4   # unacknowledged advance
    a._save_state()

    b = ArcClient("g", trace_path=path, max_actions=50, gate=LevelGate(rulebook_path=tmp_path / "r.json"))
    with pytest.raises(ActionRefused, match="FULL GAME RESET"):
        b.reset()
    with pytest.raises(ActionRefused, match="ceiling reached"):
        b.act(1)
    with pytest.raises(GateRefusal):
        b.gate.check()

    b.actions_used = 0                   # clear the budget refusal to reach the rest
    with pytest.raises(ActionRefused, match="not in this game's available_actions"):
        b.act(3)
    b.state = "GAME_OVER"
    b._save_state()
    c = ArcClient("g", trace_path=path, max_actions=0)
    with pytest.raises(ActionRefused, match="every non-RESET action is discarded"):
        c.act(1)


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


def test_a_state_file_predating_the_counters_does_not_report_a_false_pace(monkeypatch, tmp_path):
    """Defaulting to 0 made `status()` claim 0.0x and hide OVER BASELINE on
    exactly the resumed runs the warning exists for."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    path = tmp_path / "t.jsonl"
    with path.open("w") as fh:
        for i in range(12):                       # 12 actions, all on level 0
            fh.write(json.dumps({
                "i": i, "level": 0, "action": "ACTION1", "params": {},
                "frames": [[[i]]], "score": 0, "state": "NOT_FINISHED",
                "full_reset": False, "available_actions": ["ACTION1"]}) + "\n")
    # an old-format state file: game and level, but none of today's counters
    path.with_suffix(".state.json").write_text(json.dumps(
        {"game_id": "g", "card_id": "card-1", "level": 0, "actions_used": 12}))

    c = ArcClient("g", trace_path=path, info=GameInfo("g", baseline_actions=(10,)))
    assert c.level_actions == 12, "recovered from the trace, not defaulted to 0"
    assert "OVER BASELINE" in c.status()


def test_restart_for_replay_starts_a_new_play(stub):
    """The one legitimate use of the RESET that reset() refuses.

    Measured against the live API: a RESET with the action counter at zero —
    the state right after a level advance — begins a new play with its own
    `actions_by_level` row, and the benchmark scores the best play.
    """
    c, sent, replies = stub
    replies.extend([_frame(levels_completed=1), _frame(levels_completed=0, full_reset=True)])
    c.act(1)                                   # advances a level: counter now zero
    before = len(sent)
    c.restart_for_replay()
    assert len(sent) == before + 1, "the reset must actually be sent"
    assert c.full_resets == 1


def test_restart_for_replay_refuses_when_it_would_only_reset_the_level(stub):
    """Anywhere but immediately after an advance, this stays in the same play —
    so the replay would not be scored separately and the actions would be
    wasted."""
    c, sent, replies = stub
    replies.extend([_frame(), _frame()])
    c.act(1)
    c.act(1)                                   # counter is not zero now
    before = len(sent)
    with pytest.raises(ActionRefused, match="counter is zero"):
        c.restart_for_replay()
    assert len(sent) == before, "a refused action must not reach the server"


def test_the_per_level_budget_matches_the_official_rule(monkeypatch, tmp_path):
    """ARC: "for a level with a human median of n actions to completion, the
    agent is terminated after 5n actions." Per LEVEL, not per game."""
    monkeypatch.setenv("ARC_API_KEY", "k")

    def fake_post(url, payload, key, **kw):
        if url.endswith("/scorecard/open"):
            return {"card_id": "c"}
        r = _frame()
        r["action_input"] = {"id": 1}
        return r

    monkeypatch.setattr(client_mod, "_post", fake_post)
    c = ArcClient("g", trace_path=tmp_path / "t.jsonl",
                  info=GameInfo("g", baseline_actions=(10, 40)),
                  level_budget_multiple=5.0).open()
    assert c.level_budget == 50, "5 x the baseline for level 0"
    for _ in range(50):
        c.act(1)
    with pytest.raises(ActionRefused, match="per-level ceiling reached"):
        c.act(1)


def test_the_per_level_budget_follows_the_level(monkeypatch, tmp_path):
    """Each level gets its own allowance — that is the point of a per-level cap.

    A per-*game* cap lets one pathological level consume what the next level
    needed. `tn36` spent 309 actions on a level with a 55 baseline (5.6x) and
    then hit the game cap two levels later.
    """
    monkeypatch.setenv("ARC_API_KEY", "k")
    state = {"level": 0}

    def fake_post(url, payload, key, **kw):
        if url.endswith("/scorecard/open"):
            return {"card_id": "c"}
        r = _frame(levels_completed=state["level"])
        r["action_input"] = {"id": 1}
        return r

    monkeypatch.setattr(client_mod, "_post", fake_post)
    c = ArcClient("g", trace_path=tmp_path / "t.jsonl",
                  info=GameInfo("g", baseline_actions=(10, 40)),
                  level_budget_multiple=5.0).open()
    for _ in range(30):
        c.act(1)
    state["level"] = 1                     # advance
    c.act(1)
    assert c.level_actions == 0, "the tally restarts with the level"
    assert c.level_budget == 200, "and so does the allowance: 5 x 40"


def test_no_per_level_budget_when_the_multiple_is_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "k")
    c = ArcClient("g", trace_path=tmp_path / "t.jsonl",
                  info=GameInfo("g", baseline_actions=(10,)))
    assert c.level_budget == 0


# --------------------------------------------------------------------------- #
# the running score
#
# `su15` is the case behind this block. Its solver spent 268 actions on a
# 31-baseline level and 182 on an 8-baseline one, cleared level 7, and then
# correctly chose to replay -- with nothing in the harness able to tell it that
# those two levels had already fixed its ceiling at 0.82.
# --------------------------------------------------------------------------- #


def _levelled(monkeypatch, tmp_path, baselines, win_levels, **kw):
    """A client whose level advances are driven by `post.next`."""
    def fake_post(url, payload, key, **kwargs):
        if url.endswith("/scorecard/open"):
            return {"card_id": "card-1"}
        reply = _frame(win_levels=win_levels, **getattr(fake_post, "next", {}))
        name = url.rsplit("/", 1)[-1]
        reply["action_input"] = {"id": 0 if name == "RESET" else int(name.removeprefix("ACTION"))}
        return reply

    monkeypatch.setattr(client_mod, "_post", fake_post)
    monkeypatch.setenv("ARC_API_KEY", "k")
    info = GameInfo("g1", baseline_actions=tuple(baselines))
    c = ArcClient("g1", trace_path=tmp_path / "t.jsonl", info=info, **kw).open()
    return c, fake_post


def _clear(c, post, level, actions):
    """Spend `actions` on the current level, the last of which clears it."""
    for _ in range(actions - 1):
        c.act(1)
    post.next = {"levels_completed": level}
    c.act(1)
    post.next = {"levels_completed": level}


def test_level_costs_charge_the_clearing_action_to_the_level_it_left(
    monkeypatch, tmp_path
):
    """The same attribution `scoring.actions_per_level` uses, and it must match.

    The action that finishes a level is recorded at the *next* level, so a
    counter that reads the recorded level shortens every level by one. Checked
    against the scorer on the very trace this client wrote rather than asserted
    twice from the same assumption.
    """
    from athanor.ccarc3.ledger import load
    from athanor.ccarc3.scoring import actions_per_level

    c, post = _levelled(monkeypatch, tmp_path, (10, 20, 30), 3)
    _clear(c, post, 1, 7)
    _clear(c, post, 2, 4)
    assert c.level_costs == (7, 4)
    assert actions_per_level(load(c.trace_path), 3)[:2] == [7, 4]


def test_a_full_reset_starts_the_score_over_because_plays_score_separately(
    monkeypatch, tmp_path
):
    """The server keeps `actions_by_level` per play and scores the best one.

    Carrying the abandoned play's costs into the replay is exactly the error
    that would read `su15`'s 1.000-pace replay as its 8.65x exploration.
    """
    c, post = _levelled(monkeypatch, tmp_path, (10, 20, 30), 3)
    _clear(c, post, 1, 40)
    assert c.level_costs == (40,)

    post.next = {"levels_completed": 0}
    c.restart_for_replay()
    assert c.level_costs == (), "a new play scores from nothing"
    assert c.full_resets == 1

    _clear(c, post, 1, 5)
    assert c.level_costs == (5,)


def test_the_completion_cap_needs_no_baselines(monkeypatch, tmp_path):
    """The half of the score that survives the baseline-free setup.

    C is pure structure -- which levels fell, not how fast -- so a solver that
    can see nothing else can still see this.
    """
    c, post = _levelled(monkeypatch, tmp_path, (), 9)
    for level in (1, 2):
        _clear(c, post, level, 3)
    assert c.score_now is None and c.score_ceiling is None
    assert c.completion_cap == pytest.approx(3 / 45)


def test_the_ceiling_reproduces_the_su15_replay_decision(monkeypatch, tmp_path):
    """The real numbers from the run that motivated this, checked end to end.

    su15's exploration play cleared seven of nine levels at
    [16, 22, 16, 18, 5, 268, 182] against baselines
    [22, 42, 26, 115, 36, 31, 8, 40, 41]. Two blown levels put a perfect score
    permanently out of reach: 0.8199, whatever it did on levels 8 and 9. That
    number is why replaying was right, and the solver had no way to see it.
    """
    from athanor.ccarc3.scoring import score_environment

    baselines = (22, 42, 26, 115, 36, 31, 8, 40, 41)
    c, post = _levelled(monkeypatch, tmp_path, baselines, 9)
    for level, cost in enumerate([16, 22, 16, 18, 5, 268, 182], start=1):
        _clear(c, post, level, cost)

    assert c.level_costs == (16, 22, 16, 18, 5, 268, 182)
    assert c.score_ceiling == pytest.approx(0.8199, abs=5e-5)
    # score_now must agree with the scorer on the same play, or the number in
    # status() is a second implementation free to drift from the real one.
    stopped = list(c.level_costs) + [None, None]
    assert c.score_now == pytest.approx(score_environment(list(baselines), stopped).score)


def test_a_clean_play_keeps_its_ceiling_at_one(monkeypatch, tmp_path):
    """The contrast case: under baseline everywhere, nothing is out of reach."""
    baselines = (22, 42, 26, 115, 36, 31, 8, 40, 41)
    c, post = _levelled(monkeypatch, tmp_path, baselines, 9)
    for level, cost in enumerate([14, 22, 16, 9, 5, 11, 7], start=1):
        _clear(c, post, level, cost)
    assert c.score_ceiling == pytest.approx(1.0)
    assert "CEILING" not in c.status()


def test_status_says_nothing_about_score_unless_asked(monkeypatch, tmp_path):
    """The flag is what keeps the baseline-free arm one-variable.

    Ten of its twenty-five games ran before this existed. A signal that turned
    itself on halfway would make the two halves different experiments.
    """
    baselines = (22, 42, 26, 115, 36, 31, 8, 40, 41)
    c, post = _levelled(monkeypatch, tmp_path, baselines, 9)
    for level, cost in enumerate([16, 22, 16, 18, 5, 268, 182], start=1):
        _clear(c, post, level, cost)
    assert c.score_ceiling < 0.9, "the client still knows"
    s = c.status()
    assert "CEILING" not in s and "score " not in s, "it just does not say"


def test_a_lost_ceiling_names_the_instrument_that_recovers_it(monkeypatch, tmp_path):
    """A warning that reports a number without an action is a number.

    The solver's move here is a new play, and the message has to say so --
    §0a of the doctrine spent a whole revision establishing that replaying is
    safe, and it is still the least-used call in the harness.
    """
    baselines = (22, 42, 26, 115, 36, 31, 8, 40, 41)
    c, post = _levelled(monkeypatch, tmp_path, baselines, 9, show_score=True)
    for level, cost in enumerate([16, 22, 16, 18, 5, 268, 182], start=1):
        _clear(c, post, level, cost)
    s = c.status()
    assert "score 0.385" in s and "ceiling 0.820" in s
    assert "CEILING 0.820" in s and "+0.180" in s and "restart_for_replay()" in s


def test_without_baselines_status_still_shows_the_cap(monkeypatch, tmp_path):
    """The cap is shown, and labelled as a bound rather than a score.

    It used to print `[cap 0.067 = 2/9 levels]` and stop. That is half of
    `E = min(cap, raw)` with nothing to say the other half exists, and at 9 of 9
    it renders as `cap 1.000`, which reads as a perfect score. `bp35` read
    exactly that, wrote "all nine levels are cleared with a perfect score, so the
    game is won", and stopped. Its `raw` was 0.7252 and the replay it skipped was
    worth up to +0.2748.

    The `score_ceiling` *warning* still must not appear -- that one is computed
    from the medians and is genuinely unavailable here, which is what this test
    originally guarded.
    """
    c, post = _levelled(monkeypatch, tmp_path, (), 9, show_score=True)
    for level in (1, 2):
        _clear(c, post, level, 3)
    s = c.status()
    assert "cap 0.067 = 2/9 levels" in s
    assert "not your score" in s, "a bare cap reads as a score"
    assert "CEILING 0." not in s, "no baselines, no score_ceiling to claim"


def test_winning_without_baselines_points_at_the_replay_instrument(monkeypatch, tmp_path):
    """The one moment the advice is actionable is the one moment it was missing.

    `restart_for_replay()` is legal only while the server's action counter is
    zero -- the frame immediately after the final level advance -- and any
    further action makes it illegal for good. The doctrine says so in §0a, an
    hour of context earlier. `bp35` reached that frame, saw `cap 1.000`, and
    wrapped up.
    """
    c, post = _levelled(monkeypatch, tmp_path, (), 2, show_score=True)
    _clear(c, post, 1, 3)
    for _ in range(2):
        c.act(1)
    post.next = {"levels_completed": 2, "state": "WIN"}
    c.act(1)
    s = c.status()
    assert "`restart_for_replay()` is legal RIGHT NOW" in s
    assert "replay anyway" in s


def test_a_resumed_run_rebuilds_its_level_costs_from_the_trace(monkeypatch, tmp_path):
    """A state file predating this counter must not report a score of zero.

    Same repair, and same reason, as `level_actions`: a silently wrong number is
    worse than an absent one, and here the wrong number reads as "you have
    cleared nothing" on a run that has cleared two.
    """
    c, post = _levelled(monkeypatch, tmp_path, (10, 20, 30), 3)
    _clear(c, post, 1, 7)
    _clear(c, post, 2, 4)

    state = json.loads(c.state_path.read_text())
    del state["level_costs"]
    c.state_path.write_text(json.dumps(state))

    resumed = ArcClient("g1", trace_path=tmp_path / "t.jsonl",
                        info=GameInfo("g1", baseline_actions=(10, 20, 30)))
    # `open()` now verifies the server is where the ledger says before resuming a
    # run with work in it (see tests/test_resume_agreement.py). This one has two
    # levels cleared, so it must present an agreeing card; the fake server here
    # serves actions, not scorecards.
    resumed.scorecard = lambda: {"cards": {"g1": {"levels_completed": [2]}}}
    resumed.open()
    assert resumed.level_costs == (7, 4)


# --------------------------------------------------------------------------- #
# withholding baselines from the solver without withholding them from the rules
# --------------------------------------------------------------------------- #


def test_hiding_baselines_does_not_lift_the_official_per_level_cap(
    monkeypatch, tmp_path
):
    """The two-variable bug that made the baseline-free arm not an ablation.

    That arm hid the medians by blanking `GameInfo.baseline_actions` -- the same
    array `level_budget` derives ARC's 5n termination rule from. So hiding the
    information removed the rule. `su15` then ran a 31-baseline level to 268
    actions where ARC would have stopped it at 155.

    ARC withholding a number from an agent does not stop ARC applying it.
    """
    c, _ = _levelled(monkeypatch, tmp_path, (31, 40), 2,
                     level_budget_multiple=5.0, quiet_pace=True)
    assert c.baseline_here is None, "the solver cannot see it"
    assert c.level_budget == 155, "the rule applies anyway: 5 x 31"

    blanked, _ = _levelled(monkeypatch, tmp_path, (), 2, level_budget_multiple=5.0)
    assert blanked.level_budget == 0, "how the arm actually ran"


def test_every_solver_facing_baseline_surface_goes_quiet_together(
    monkeypatch, tmp_path
):
    """One property gates them all, so none can be forgotten and leak."""
    c, post = _levelled(monkeypatch, tmp_path, (31, 40), 2,
                        level_budget_multiple=5.0, quiet_pace=True,
                        show_score=True)
    for _ in range(9):
        c.act(1)
    s = c.status()
    assert "on this level = " not in s and "OVER BASELINE" not in s
    assert "31" not in s and "155" not in s, "not even by arithmetic"
    assert c.pace() == {}
    assert c.score_now is None and c.score_ceiling is None
    assert "cap 0.000 = 0/2 levels" in s, "the structural half still shows"


def test_the_cap_refusal_does_not_name_a_withheld_baseline(monkeypatch, tmp_path):
    """It does name the cap, which is 5n -- but only as the environment ends."""
    c, post = _levelled(monkeypatch, tmp_path, (2, 40), 2,
                        level_budget_multiple=5.0, quiet_pace=True)
    for _ in range(10):
        c.act(1)
    with pytest.raises(ActionRefused, match="baseline withheld"):
        c.act(1)


def test_a_visible_baseline_is_still_named_in_the_refusal(monkeypatch, tmp_path):
    c, post = _levelled(monkeypatch, tmp_path, (2, 40), 2, level_budget_multiple=5.0)
    for _ in range(10):
        c.act(1)
    with pytest.raises(ActionRefused, match=r"baseline 2, 5x"):
        c.act(1)


def test_a_replay_is_still_possible_after_clearing_every_level(monkeypatch, tmp_path):
    """The case the doctrine does not currently cover, and it is not blocked.

    `E = min(cap, raw)`. Clear everything and `cap` is 1.0, so `E` *is* `raw` --
    and a run that finished every level at 1.05x is sitting on a score it could
    still raise. `sp80` did exactly that: 6/6 cleared, `raw` 0.9785, **71% of its
    action budget unspent**, and it stopped. That 0.0215 is the whole of its
    deficit against its control.

    `act()` refuses in a terminal state -- and its refusal names the way out --
    while `restart_for_replay()` routes through `reset(force_full=True)`, which
    is permitted because `_last_advanced` is true immediately after the winning
    advance. That is the real server's behaviour: `levels_completed` reaching
    `win_levels` and `state` becoming WIN arrive in the same frame.
    """
    c, post = _levelled(monkeypatch, tmp_path, (10, 20), 2)
    _clear(c, post, 1, 3)
    # The winning advance and the WIN state are one frame, as the server sends them.
    post.next = {"levels_completed": 2, "state": "WIN"}
    c.act(1)
    assert c.done and c.completion_cap == 1.0 and c.level_costs == (3, 1)

    with pytest.raises(ActionRefused, match="RESET first"):
        c.act(1)

    post.next = {"levels_completed": 0, "state": "NOT_FINISHED"}
    c.restart_for_replay()
    assert c.level == 0 and not c.done
    assert c.level_costs == (), "the replay scores from nothing, as any new play does"


def test_an_action_after_the_winning_one_closes_the_replay_door_for_good(
    monkeypatch, tmp_path
):
    """A dead end worth knowing about, found by getting the test above wrong.

    `restart_for_replay()` requires the server's action counter to be zero, which
    is true only immediately after a level advance. If WIN arrives on a *later*
    action than the final advance, that action is not an advance, the flag
    clears, and the replay is refused -- while `act()` refuses everything too
    because the state is terminal. The run is then stuck with the score it has.

    Not reachable against the live server, which sends the final advance and WIN
    in one frame. Recorded because the harness must not start relying on that.
    """
    c, post = _levelled(monkeypatch, tmp_path, (10, 20), 2)
    _clear(c, post, 1, 3)
    _clear(c, post, 2, 2)                      # advance to the last level
    post.next = {"levels_completed": 2, "state": "WIN"}
    c.act(1)                                   # WIN, but not an advance
    assert c.done

    with pytest.raises(ActionRefused, match="RESET first"):
        c.act(1)
    with pytest.raises(ActionRefused, match="only starts a new play"):
        c.restart_for_replay()


def test_list_games_can_withhold_baselines_from_the_solver(monkeypatch):
    """The leak channel that sanitising workspace files cannot reach.

    A baseline-free arm can strip `session.py`, `CLAUDE.md`, `DOCTRINE.md` and
    `meta.json` and still hand the solver an API key and this package — and
    `arc.list_games()` then returns `baseline_actions` for all 25 environments
    on request. The first twelve runs of that arm leaked through `meta.json`
    instead and nobody checked this door was also open.

    Every `GameInfo` the package builds from the API comes through here, so this
    is the single place it can be shut. Scoped to the environment because the
    *runner* still needs the numbers — to build its queue, to enforce ARC's 5n
    per-level rule, and to score — while the solver's child process must not.
    """
    raw = [{"game_id": "g1", "title": "G1", "tags": ["click"],
            "baseline_actions": [10, 20, 30]}]
    monkeypatch.setattr(client_mod, "_get", lambda *a, **k: raw)
    monkeypatch.setenv("ARC_API_KEY", "k")

    monkeypatch.delenv(client_mod.HIDE_BASELINES_ENV, raising=False)
    assert client_mod.list_games()[0].baseline_actions == (10, 20, 30)

    monkeypatch.setenv(client_mod.HIDE_BASELINES_ENV, "1")
    hidden = client_mod.list_games()
    assert hidden[0].baseline_actions == ()
    assert hidden[0].game_id == "g1" and hidden[0].tags == ("click",), (
        "only the medians are withheld; the game list itself still works"
    )

    # Anything other than exactly "1" leaves it on, so a stray empty string or a
    # "0" cannot silently blind a scoring run.
    monkeypatch.setenv(client_mod.HIDE_BASELINES_ENV, "0")
    assert client_mod.list_games()[0].baseline_actions == (10, 20, 30)


def test_the_hide_flag_has_no_bypass(monkeypatch):
    """Under the flag, *no* path in the package returns a baseline.

    Both bypasses were real and both were reachable from the solver's own
    namespace. `list_games(_unfiltered=True)` was a keyword any caller could
    pass; `baselines_for()` ignored the flag outright and was re-exported from
    `athanor.ccarc3`, so `dir(arc)` advertised it. `cd82` found it exactly that
    way on its first orientation turn, without going looking.

    They existed for harness-side callers, which never needed them: the runner
    does not set the flag on itself, only on the child it spawns. So the
    exemption bought nothing and cost the boundary.
    """
    raw = [{"game_id": "g1", "title": "G1", "tags": ["click"],
            "baseline_actions": [10, 20, 30]}]
    monkeypatch.setattr(client_mod, "_get", lambda *a, **k: raw)
    monkeypatch.setenv("ARC_API_KEY", "k")

    monkeypatch.delenv(client_mod.HIDE_BASELINES_ENV, raising=False)
    assert client_mod.baselines_for("g1") == (10, 20, 30), (
        "the runner, which never sets the flag, is unaffected"
    )

    monkeypatch.setenv(client_mod.HIDE_BASELINES_ENV, "1")
    with pytest.raises(PermissionError, match="withheld"):
        client_mod.baselines_for("g1")

    with pytest.raises(TypeError):
        client_mod.list_games(_unfiltered=True)  # type: ignore[call-arg]


def test_baselines_for_is_absent_from_the_solver_namespace():
    """`dir(arc)` must not name it. That listing is how the leak was found.

    The solver imports `athanor.ccarc3`, not `athanor.ccarc3.client`, and a
    first-turn `dir()` on an unfamiliar package is ordinary orientation, not
    probing. Keeping the name out of that listing is the difference between a
    solver having to decide to go looking and being handed the answer.
    """
    import athanor.ccarc3 as pkg

    assert "baselines_for" not in dir(pkg)
    assert "baselines_for" not in pkg.__all__
    assert callable(client_mod.baselines_for), "harness-side import still works"


def test_level_costs_match_the_scorer_when_the_play_opens_with_a_reset(
    monkeypatch, tmp_path
):
    """**The invariant above was already asserted and could not fail.**

    `_levelled` starts the game with an `ACTION1`, so transition 0 is never a
    RESET and `actions_per_level`'s "a RESET that opens a play is not an action"
    exclusion never fires. Both sides then agreed on a trace that avoided the
    only case where they differed: measured across every preserved run carrying
    `level_costs`, the client read +1 against the scorer on element 0 of any run
    with no full reset, and exact everywhere else.

    A real game always opens with RESET, so this is the shape that runs.
    """
    from athanor.ccarc3.ledger import load
    from athanor.ccarc3.scoring import actions_per_level

    c, post = _levelled(monkeypatch, tmp_path, (10, 20, 30), 3)
    c.reset()                                  # the opening RESET a real play has
    _clear(c, post, 1, 7)
    _clear(c, post, 2, 4)

    from_trace = actions_per_level(load(c.trace_path), 3)[:2]
    assert c.level_costs == tuple(from_trace), (
        f"client says {c.level_costs}, scorer says {tuple(from_trace)} — "
        f"the workspace prints one and the banked result uses the other"
    )


def test_a_frame_with_no_level_field_is_refused_not_read_as_zero(monkeypatch, tmp_path):
    """**Reading 0 here is scored as a full reset eleven lines later.**

    The fallback chain ended in `0`, so a frame missing both `levels_completed`
    and `score` would set `self.level = 0`, trip `self.level < previous_level`,
    increment `full_resets` and wipe `level_costs` — recording a replay that
    never happened, on a run that was fine.

    The fallback is already unreachable behind the shim: `score` is in
    `arc_proxy.HIDDEN_FIELDS` and `_strip` runs on every forwarded body.
    """
    c, post = _levelled(monkeypatch, tmp_path, (10, 20, 30), 3)
    c.reset()
    for _ in range(3):
        c.act(1)

    # `_frame` always supplies the key, which is the point -- the failure being
    # guarded is the server dropping it, so strip it from the reply itself.
    inner = client_mod._post
    def _no_level_field(url, payload, key, **kw):          # noqa: N802 -- see below
        reply = inner(url, payload, key, **kw)
        reply.pop("levels_completed", None)
        reply.pop("score", None)
        return reply
    monkeypatch.setattr(client_mod, "_post", _no_level_field)

    with pytest.raises(RuntimeError, match="neither"):
        c.act(1)
