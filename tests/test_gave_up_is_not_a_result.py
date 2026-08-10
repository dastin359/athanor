"""A solver that stopped with its allowance untouched did not lose — it quit.

Fourth face of one failure. Interrupted, crashed, clocked out and now quit all
look identical in `result.json` — `won: false`, no error — and all four are the
harness failing to record that the environment was never actually contested.
`if prior and not prior.get("error"): skip` then makes the false loss permanent.
"""
import json

import pytest

from athanor.ccarc3 import session as S
from athanor.ccarc3.client import GameInfo
from athanor.ccarc3.session import Ccarc3Config, Workspace


def _ws(tmp_path, *, levels=10, baselines=(32, 81, 60, 71, 205, 148, 244, 109, 164, 225),
        meta_budget=None):
    root = tmp_path / "ws"
    root.mkdir()
    meta = {"game_id": "lf52-x", "levels": levels}
    if meta_budget is not None:
        meta["action_budget"] = meta_budget
    (root / "meta.json").write_text(json.dumps(meta))
    info = GameInfo(game_id="lf52-x", title="LF52", tags=(), baseline_actions=tuple(baselines))
    return Workspace(root=root, config=Ccarc3Config("lf52-x", budget_multiple=5.0),
                     info=info, initial_prompt="")


def test_the_budget_survives_the_baseline_strip(tmp_path):
    """The strip removes `action_budget` from meta.json on purpose, which zeroed
    this for all 25 clean rollouts and silently disabled the guard that reads it."""
    ws = _ws(tmp_path)                       # stripped: no action_budget in meta

    assert S._action_budget(ws) == 1339 * 5, "must fall back to the parent's own arithmetic"


def test_meta_still_wins_when_it_has_one(tmp_path):
    """An unstripped run keeps reporting the multiple it actually got."""
    assert S._action_budget(_ws(tmp_path, meta_budget=4242)) == 4242


def test_quitting_with_the_allowance_untouched_is_not_a_result(tmp_path, monkeypatch):
    """`lf52`: exit 0, no timeout, 7 of 10 levels, 865 of 6695 actions."""
    ws = _ws(tmp_path)
    monkeypatch.setattr(S, "ledger_facts", lambda p: {"actions_used": 865, "levels_reached": 7})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    out = S.collect_outcome(ws, exit_code=0, timed_out=False)

    assert "gave up" in out["error"] and "7 of 10" in out["error"]


def test_a_solver_with_no_actions_left_is_not_asked_to_continue(tmp_path, monkeypatch):
    """Not because exhausting the budget earns the loss a pass — any run that
    does not score `E = 1` is a loss — but because a nudge has nothing to spend.

    The threshold used to be halfway, and 5,000 of 6,695 actions was banked
    without a retry on the reasoning that the solver had "contested the game".
    An operator removed that on 2026-08-10. What is left is the only exemption
    that does not rest on a judgement call: there are no actions remaining, so
    telling the solver to carry on would be telling it to do nothing.
    """
    ws = _ws(tmp_path)
    budget = S._action_budget(ws)
    monkeypatch.setattr(S, "ledger_facts", lambda p: {"actions_used": budget, "levels_reached": 7})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    assert not S.collect_outcome(ws, exit_code=0, timed_out=False).get("error")


def test_stopping_with_any_allowance_left_is_not_a_result(tmp_path, monkeypatch):
    """The restriction removed on 2026-08-10, pinned as a behaviour.

    5,000 of 6,695 actions is well past halfway and was banked as a real loss
    under the old rule. One action short of the budget is the tightest case the
    new rule must still catch.
    """
    ws = _ws(tmp_path)
    budget = S._action_budget(ws)
    for used in (5000, budget - 1):
        monkeypatch.setattr(S, "ledger_facts",
                            lambda p, u=used: {"actions_used": u, "levels_reached": 7})
        monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
        monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})
        out = S.collect_outcome(ws, exit_code=0, timed_out=False)
        assert out.get("gave_up"), f"{used} of {budget} was banked as a real loss"


def test_the_fraction_is_settable_for_an_operator_who_wants_the_old_rule(tmp_path, monkeypatch):
    """`0.5` restores the previous behaviour exactly, so the change is a default
    rather than a removal."""
    ws = _ws(tmp_path)
    budget = S._action_budget(ws)
    monkeypatch.setattr(S, "ledger_facts", lambda p: {"actions_used": 5000, "levels_reached": 7})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    monkeypatch.setenv(S.GIVE_UP_FRACTION_ENV, "0.5")
    assert not S.collect_outcome(ws, exit_code=0, timed_out=False).get("gave_up")

    monkeypatch.delenv(S.GIVE_UP_FRACTION_ENV)
    assert S.collect_outcome(ws, exit_code=0, timed_out=False).get("gave_up")


@pytest.mark.parametrize("bad", ["banana", "0", "-1", "2", ""])
def test_a_nonsense_fraction_falls_back_to_no_restriction(monkeypatch, bad):
    """Never to a *tighter* rule: a typo must not silently start banking runs
    the operator asked to have continued."""
    monkeypatch.setenv(S.GIVE_UP_FRACTION_ENV, bad)
    assert S._give_up_fraction() == 1.0


def test_the_retry_is_bounded(tmp_path, monkeypatch):
    """Unlike an interruption, quitting may be telling us the game is hard. At
    ~$60 a run, 12 passes is $720 to hear the same answer three times.

    **Built on the real driver layout, because the first version of this test
    could not fail.** It drove the bound through a monkeypatched
    `outcome["attempts"]`, which counts `stream.*.jsonl` files inside ONE
    workspace. The driver gives every retry a fresh `attempt_N/`, so that number
    is 1 on every attempt and the bound never bit — the test asserted a
    quantity nobody computed.
    """
    game = "lf52-x"
    monkeypatch.setattr(S, "ledger_facts", lambda p: {"actions_used": 865, "levels_reached": 7})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})     # always 1 in practice
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    def attempt(n, *, gave_up=True):
        """One prior attempt_N/<game>/result.json, as the driver writes it."""
        d = tmp_path / game / ("attempt_%d" % n) / game
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(
            {"error": "…it gave up with the allowance untouched…"} if gave_up else {}))
        return d

    for n in range(1, S.GIVE_UP_ATTEMPTS):
        attempt(n)
    live = tmp_path / game / ("attempt_%d" % S.GIVE_UP_ATTEMPTS) / game
    live.mkdir(parents=True)
    ws = _ws(tmp_path)
    object.__setattr__(ws, "root", live) if False else None
    ws.root = live
    (live / "meta.json").write_text(json.dumps({"game_id": game, "levels": 10}))

    assert S._prior_give_ups(ws) == S.GIVE_UP_ATTEMPTS - 1
    assert S.collect_outcome(ws, exit_code=0, timed_out=False).get("error"), (
        "one below the bound still retries"
    )

    attempt(S.GIVE_UP_ATTEMPTS + 1)
    assert S._prior_give_ups(ws) == S.GIVE_UP_ATTEMPTS
    assert not S.collect_outcome(ws, exit_code=0, timed_out=False).get("error"), (
        "at the bound the loss stands as real"
    )


def test_a_win_is_never_second_guessed(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    monkeypatch.setattr(S, "ledger_facts",
                        lambda p: {"actions_used": 100, "levels_reached": 10, "won": True})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    assert not S.collect_outcome(ws, exit_code=0, timed_out=False).get("error")


def test_a_completed_win_killed_by_signal_is_not_re_run(tmp_path, monkeypatch):
    """**Guards 2-4 exempt a win; guard 1 never did.** SIGTERM is how the
    supervisor stops solvers at the quota ceiling, so this fires across a sweep
    rather than once. `cd82` attempt_2 won 6 of 6, was recorded `interrupted, not
    a result`, and was discarded and re-played -- its scorecard shows both plays
    winning and the replacement's final playthrough identical to the one thrown
    away.
    """
    ws = _ws(tmp_path, levels=3, baselines=(30, 40, 50))
    monkeypatch.setattr(S, "ledger_facts", lambda p: {
        "actions_used": 90, "levels_reached": 3, "won": True,
        "levels_reached_final_playthrough": 3})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    out = S.collect_outcome(ws, exit_code=143, timed_out=False)

    assert not out.get("error"), "a finished win is a result however the run ended"
    assert out["killed_by_signal"] == 15, (
        "the interruption is still recorded -- folding it into the error branch "
        "would delete the evidence from exactly the runs this is about"
    )


def test_a_win_killed_mid_replay_is_still_re_run(tmp_path, monkeypatch):
    """The exemption is "won *and* the last play went the distance". A stub final
    playthrough banks a tiny action count beside a level count from a different
    play -- the flattering half of each."""
    ws = _ws(tmp_path, levels=3, baselines=(30, 40, 50))
    monkeypatch.setattr(S, "ledger_facts", lambda p: {
        "actions_used": 95, "levels_reached": 3, "won": True,
        "levels_reached_final_playthrough": 1})      # SIGTERM'd early in the replay
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})

    out = S.collect_outcome(ws, exit_code=143, timed_out=False)

    assert "re-run this game" in out.get("error", "")


def test_a_corrupt_rules_file_does_not_cost_the_result(tmp_path, monkeypatch):
    """**Bookkeeping must never take the outcome down with it.**

    `collect_outcome` parsed rules.json unguarded, three lines before it writes
    result.json — so a file that is not valid JSON raised and the run was
    recorded as nothing at all. A WON game, banked as no result.

    Zero bytes is the realistic case and the sweep manufactures it: `RuleBook.save`
    uses `write_text`, which truncates before it writes, and the supervisor stops
    solvers with a signal at the quota ceiling by design. The solver also has
    Write on its own workspace.
    """
    ws = _ws(tmp_path, levels=3, baselines=(30, 40, 50))
    monkeypatch.setattr(S, "ledger_facts", lambda p: {
        "actions_used": 90, "levels_reached": 3, "won": True,
        "levels_reached_final_playthrough": 3})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})
    ws.rules_path.write_text("", encoding="utf-8")          # the 0-byte case

    out = S.collect_outcome(ws, exit_code=0, timed_out=False)

    assert out["won"] is True, "a won run was lost to a bookkeeping file"
    assert (ws.root / "result.json").exists(), "result.json was never written"
    assert "rules_error" in out, (
        "the corruption is swallowed silently; it must be recorded"
    )
    assert out["mechanics_recorded"] == 0 and out["refutations_recorded"] == 0


def test_a_truncated_rules_file_is_also_survivable(tmp_path, monkeypatch):
    """Not just empty: a half-written file is what a signal mid-`write_text`
    actually leaves behind."""
    ws = _ws(tmp_path, levels=3, baselines=(30, 40, 50))
    monkeypatch.setattr(S, "ledger_facts", lambda p: {
        "actions_used": 90, "levels_reached": 3, "won": True,
        "levels_reached_final_playthrough": 3})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})
    ws.rules_path.write_text('{"verified": [{"rule": "a', encoding="utf-8")

    out = S.collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["won"] is True and (ws.root / "result.json").exists()
    assert "rules_error" in out


def test_a_valid_rules_file_is_still_counted(tmp_path, monkeypatch):
    """The guard must not become a shrug that ignores the file entirely."""
    ws = _ws(tmp_path, levels=3, baselines=(30, 40, 50))
    monkeypatch.setattr(S, "ledger_facts", lambda p: {
        "actions_used": 90, "levels_reached": 3, "won": True,
        "levels_reached_final_playthrough": 3})
    monkeypatch.setattr(S, "run_cost", lambda p: {"attempts": 1})
    monkeypatch.setattr(S, "snapshot_scorecard", lambda ws: {})
    ws.rules_path.write_text(
        '{"verified": [1, 2, 3], "refuted": [4, 5]}', encoding="utf-8")

    out = S.collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["mechanics_recorded"] == 3 and out["refutations_recorded"] == 2
    assert "rules_error" not in out
