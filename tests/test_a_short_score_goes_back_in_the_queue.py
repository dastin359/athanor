"""An environment that scores short is replayed by the harness, not by a person.

ARC scores an environment by its **best** play and renders every play on the
public card, so replaying is the documented mechanic rather than a loophole.
What separates an honest number from a cherry-picked one is therefore not
whether a replay happened -- it is whether the rule was fixed in advance,
applied to every environment alike, and stated alongside the result.

The driver did not have that rule. `_outstanding()` was "games with no
clean_result.json", so a game that **won with a poor score banked and was never
revisited**; the nudge and `GIVE_UP_ATTEMPTS` both fire on stopping early, not
on scoring low. Improving such a game meant an operator clearing its bank by
hand -- which is selection, and reads like it.

Now the harness decides: below `RETRY_BELOW`, and under `RETRY_ATTEMPTS` tries,
the environment returns to the queue -- **behind** every environment not yet
played, because a complete card is worth more than a polished one.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))


@pytest.fixture
def driver(monkeypatch, tmp_path):
    import clean_rollouts as cr
    monkeypatch.setattr(cr, "OUT", tmp_path)
    monkeypatch.setattr(cr, "RETRY_BELOW", 1.0)
    monkeypatch.setattr(cr, "RETRY_ATTEMPTS", 3)
    return cr


def _bank(cr, gid: str, e, attempts: int = 1) -> None:
    d = cr.OUT / gid
    d.mkdir(parents=True, exist_ok=True)
    (d / "clean_result.json").write_text(json.dumps({"E": e, "won": True}))
    for i in range(1, attempts + 1):
        ws = d / f"attempt_{i}" / gid
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "result.json").write_text("{}")          # a verdict was reached


def test_a_perfect_score_is_never_replayed(driver):
    gid = driver.GAMES[0]
    _bank(driver, gid, 1.0)
    assert not driver._wants_replay(gid)
    assert gid not in driver._outstanding()


def test_a_short_score_is_replayed(driver):
    gid = driver.GAMES[0]
    _bank(driver, gid, 0.7252)
    assert driver._wants_replay(gid)
    assert gid in driver._outstanding()


def test_an_unscored_run_is_left_alone(driver):
    """`None` means scoring failed, not that the run was bad.

    Treating unknown as zero would send a finished game back for another
    multi-hour attempt on the strength of a missing baseline.
    """
    gid = driver.GAMES[0]
    _bank(driver, gid, None)
    assert not driver._wants_replay(gid)
    assert gid not in driver._outstanding()


def test_replays_queue_behind_everything_unplayed(driver):
    """A complete card first. A banked game is already on the card; a fresh one is not."""
    short, fresh = driver.GAMES[0], driver.GAMES[1]
    _bank(driver, short, 0.5)
    q = driver._outstanding()
    assert q.index(fresh) < q.index(short), (
        "a replay was scheduled ahead of an environment that has never been "
        "played; that spends the budget on polish before the card is complete"
    )
    assert q[-1] == short or q.index(short) > q.index(fresh)


def test_the_attempt_bound_is_enforced(driver):
    """Unbounded retries hand one stubborn environment the whole budget."""
    spent, inside = driver.GAMES[0], driver.GAMES[1]
    _bank(driver, spent, 0.5, attempts=3)
    assert not driver._wants_replay(spent), "a 4th attempt was scheduled past the bound"
    # A separate game, because attempt directories accumulate: re-banking the
    # same id leaves the earlier attempts on disk and the count keeps climbing.
    _bank(driver, inside, 0.5, attempts=2)
    assert driver._wants_replay(inside), "a 3rd attempt was refused inside the bound"


def test_the_threshold_is_a_policy_not_a_constant(driver, monkeypatch):
    gid = driver.GAMES[0]
    _bank(driver, gid, 0.995)
    assert driver._wants_replay(gid)
    monkeypatch.setattr(driver, "RETRY_BELOW", 0.99)
    assert not driver._wants_replay(gid), "the threshold did not move with the policy"
