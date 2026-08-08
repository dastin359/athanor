"""One scorecard across many games -- the thing a leaderboard submission takes.

Every assertion here guards a failure that is **silent**: the sweep runs to
completion, every game scores, and the artifact at the end is unsubmittable or
incomplete. Nothing goes red at the time, which is why these are pinned.
"""
import json
import pathlib
import sys

import pytest

from athanor.ccarc3 import arc_proxy, shared_card as sc
from athanor.ccarc3 import client as client_mod
from athanor.ccarc3.client import ArcClient
from athanor.ccarc3.session import SESSION_TEMPLATE

COOKIES = (
    {"name": "GAMESESSION", "value": "gs", "domain": "three.arcprize.org", "path": "/"},
    {"name": "AWSALBAPP-0", "value": "pin", "domain": "three.arcprize.org", "path": "/"},
)


def _render(**kw):
    args = dict(game_id="ab12-x", title="t", tags=(), baseline=(), budget=1,
                level_budget_line="", card_line="")
    args.update(kw)
    return SESSION_TEMPLATE.format(**args)


def test_an_injected_card_is_never_opened_again(monkeypatch, tmp_path):
    """The whole bug in one line: a client that opens its own card gives the
    sweep one card per game, and 25 cards is not a submission."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    posts = []
    monkeypatch.setattr(client_mod, "_post",
                        lambda url, *a, **k: posts.append(url) or {"card_id": "MINTED"})

    c = ArcClient("g", trace_path=tmp_path / "t.jsonl", card_id="LENT").open()

    assert c.card_id == "LENT"
    assert not any("scorecard/open" in u for u in posts), posts
    assert not c._owns_card


def test_a_client_with_no_card_still_opens_one(monkeypatch, tmp_path):
    """The default must not change: a lone game still gets its own card."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    monkeypatch.setattr(client_mod, "_post", lambda *a, **k: {"card_id": "MINTED"})

    c = ArcClient("g", trace_path=tmp_path / "t.jsonl").open()

    assert (c.card_id, c._owns_card) == ("MINTED", True)


def test_a_lent_card_is_not_closed_by_the_first_game_to_finish(monkeypatch, tmp_path):
    """Games run concurrently. Closing a shared card when one of them ends would
    take the card out from under every game still playing."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    posts = []
    monkeypatch.setattr(client_mod, "_get", lambda *a, **k: {"ok": 1})
    monkeypatch.setattr(client_mod, "_post", lambda url, *a, **k: posts.append(url) or {})

    c = ArcClient("g", trace_path=tmp_path / "run" / "t.jsonl", card_id="LENT").open()
    c.close()

    assert not any("scorecard/close" in u for u in posts), posts
    assert (tmp_path / "run" / "scorecard.json").exists(), (
        "the snapshot is the useful half and must still happen"
    )


def test_a_card_the_client_opened_is_still_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_API_KEY", "k")
    posts = []
    monkeypatch.setattr(client_mod, "_get", lambda *a, **k: {"ok": 1})
    monkeypatch.setattr(client_mod, "_post",
                        lambda url, *a, **k: posts.append(url) or {"card_id": "MINTED"})

    c = ArcClient("g", trace_path=tmp_path / "run" / "t.jsonl").open()
    c.close()

    assert any("scorecard/close" in u for u in posts), posts


def test_a_state_file_written_before_ownership_existed_still_owns_its_card(
    monkeypatch, tmp_path
):
    """Every card in a pre-2026-08-08 state file was self-opened -- injection did
    not exist. Defaulting the missing field to False would stop resumed runs
    closing cards they do own."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    trace.with_suffix(".state.json").write_text(
        json.dumps({"game_id": "g", "card_id": "OLD", "actions_used": 5, "level": 0})
    )

    c = ArcClient("g", trace_path=trace)

    assert (c.card_id, c._owns_card) == ("OLD", True)


def test_resuming_onto_a_different_card_keeps_the_one_it_played_on(monkeypatch, tmp_path):
    """A trace is bound to the card it was played on and cannot be moved. Resume
    wins -- the alternative replays from level 0 -- but the game is then NOT on
    the shared card, and the driver has to be able to say so."""
    monkeypatch.setenv("ARC_API_KEY", "k")
    trace = tmp_path / "t.jsonl"
    trace.write_text("")
    trace.with_suffix(".state.json").write_text(
        json.dumps({"game_id": "g", "card_id": "PLAYED_ON", "actions_used": 5, "level": 0})
    )

    c = ArcClient("g", trace_path=trace, card_id="SHARED")

    assert c.card_id == "PLAYED_ON"
    assert c.foreign_card == "SHARED", "the divergence must be recorded, not swallowed"


def test_a_shim_keeps_its_adopted_session_across_set_budget():
    """`set_budget` drops the upstream session on purpose, so a new game cannot
    inherit the last one's pinning. Adopted cookies must outlive that, or the
    order of two unrelated calls silently decides whether the sweep works."""
    st = arc_proxy.ProxyState()
    st.adopt_session(COOKIES)
    st.set_budget(100)

    jar = [h for h in st.upstream().handlers if hasattr(h, "cookiejar")][0].cookiejar
    assert {c.name for c in jar} == {"GAMESESSION", "AWSALBAPP-0"}


def test_a_shim_without_a_shared_card_keeps_its_own_session():
    """The per-game default is the safe one and must not become opt-out."""
    jar = [h for h in arc_proxy.ProxyState().upstream().handlers
           if hasattr(h, "cookiejar")][0].cookiejar
    assert list(jar) == []


@pytest.mark.parametrize("cookies", [
    (),                                                          # nothing at all
    ({"name": "GAMESESSION", "value": "g", "domain": "d", "path": "/"},),
    ({"name": "GAMESESSION", "value": "g", "domain": "d", "path": "/"},
     {"name": "csrftoken", "value": "c", "domain": "d", "path": "/"}),
])
def test_a_card_that_came_back_unpinned_is_refused(monkeypatch, cookies):
    """Without a stickiness cookie the card is unreachable from anywhere, and
    every game would fail at its first RESET with a message naming the game.

    **The empty case alone could not catch a broken check.** This passed only
    `()`, and `any(...)` over an empty sequence is False whatever
    `PINNING_PREFIX` holds — so setting that prefix to `""`, which makes
    `pinned` true for *every* cookie and disables the guard entirely, left all
    17 tests in this file green. Found when an agent made exactly that edit.

    The middle case is the one that occurs: a real card always comes back with
    `GAMESESSION`, and the question is only whether a stickiness cookie came
    with it.
    """
    monkeypatch.setenv("ARC_API_KEY", "k")
    monkeypatch.setattr(sc, "_post", lambda *a, **k: {"card_id": "C"})
    monkeypatch.setattr(sc, "export_cookies", lambda opener: cookies)

    with pytest.raises(RuntimeError, match="stickiness"):
        sc.open_card()


def test_the_workspace_names_a_card_only_when_one_is_shared():
    """A keyword the solver can see is a question the solver can ask."""
    assert "card_id='CARD'" in _render(card_line="    card_id='CARD',\n")
    assert "card_id" not in _render()


def test_a_saved_card_round_trips(tmp_path):
    """The driver is restarted by a supervisor; a card that cannot survive that
    gives the sweep one card per restart."""
    card = sc.SharedCard("C1", COOKIES)
    path = sc.save(card, tmp_path / "shared_card.json")

    assert sc.load(path) == card
    assert path.stat().st_mode & 0o077 == 0, "cookies are the credential half"


def test_the_outcome_records_which_card_the_run_scored_on(tmp_path):
    """"All 25 games are on one card" is the claim a submission rests on. It has
    to be checkable from the results, not assumed from the driver's intent."""
    from types import SimpleNamespace

    from athanor.ccarc3.session import _card_facts

    trace = tmp_path / "t.jsonl"
    trace.with_suffix(".state.json").write_text(json.dumps({"card_id": "SHARED"}))

    assert _card_facts(SimpleNamespace(trace_path=trace)) == {"card_id": "SHARED"}


def test_a_game_not_on_the_shared_card_says_so_in_its_outcome(tmp_path):
    """The silent version of this failure is a submission that is missing a game
    nobody noticed was missing."""
    from types import SimpleNamespace

    from athanor.ccarc3.session import _card_facts

    trace = tmp_path / "t.jsonl"
    trace.with_suffix(".state.json").write_text(
        json.dumps({"card_id": "PLAYED_ON", "foreign_card": "SHARED"})
    )

    assert _card_facts(SimpleNamespace(trace_path=trace)) == {
        "card_id": "PLAYED_ON", "foreign_card": "SHARED",
    }


def test_a_run_with_no_state_file_does_not_break_the_outcome(tmp_path):
    """Bookkeeping must never replace the result -- the rule this file follows
    everywhere else."""
    from types import SimpleNamespace

    from athanor.ccarc3.session import _card_facts

    assert _card_facts(SimpleNamespace(trace_path=tmp_path / "missing.jsonl")) == {}


# --- the driver's card lifecycle -------------------------------------------
# Imported the way the rest of the suite reaches `tools/`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))


def test_a_restarted_driver_reuses_its_card(monkeypatch, tmp_path):
    """The supervisor restarts this driver on every container replacement. A card
    minted per process puts the 25 games on as many cards as the sweep had
    restarts -- the bug being fixed, arriving by a different door."""
    import clean_rollouts as cr

    monkeypatch.setattr(cr, "SHARED_CARD_FILE", tmp_path / "card.json")
    monkeypatch.setattr(cr, "SHARED_CARD_HISTORY", tmp_path / "history.jsonl")
    opens = []
    monkeypatch.setattr(cr.sc, "open_card",
                        lambda **k: opens.append(1) or sc.SharedCard("C1", COOKIES))
    monkeypatch.setattr(cr.sc, "read_card", lambda *a, **k: {"cards": {}})

    first, second = cr.sweep_card(), cr.sweep_card()

    assert (first.card_id, second.card_id) == ("C1", "C1")
    assert len(opens) == 1, "a restart must not mint a second card"


def test_an_unreachable_card_is_replaced_loudly_and_not_silently(monkeypatch, tmp_path):
    """Continuing onto a new card means the submission covers only what came
    after. Minting quietly would make a partial artifact look whole."""
    import clean_rollouts as cr

    monkeypatch.setattr(cr, "SHARED_CARD_FILE", tmp_path / "card.json")
    monkeypatch.setattr(cr, "SHARED_CARD_HISTORY", tmp_path / "history.jsonl")
    sc.save(sc.SharedCard("DEAD", COOKIES), tmp_path / "card.json")
    monkeypatch.setattr(cr.sc, "open_card", lambda **k: sc.SharedCard("FRESH", COOKIES))
    def gone(*a, **k):
        raise RuntimeError("404: card_id not found")
    monkeypatch.setattr(cr.sc, "read_card", gone)

    card = cr.sweep_card()

    assert card.card_id == "FRESH"
    history = (tmp_path / "history.jsonl").read_text()
    assert "DEAD" in history and "FRESH" in history, "succession must be recorded"
    assert list(tmp_path.glob("card.DEAD.dead.json")), "the retired card is kept"


def test_every_shim_gets_the_shared_session_without_the_caller_asking(monkeypatch):
    """Applied at the chokepoint, because one game that slips past it silently
    ruins the artifact for all 25."""
    import ablate_baselines as ab

    monkeypatch.setenv("ARC_API_KEY", "k")
    monkeypatch.setattr(ab, "_proxies", {})
    ab.use_shared_card(sc.SharedCard("C1", COOKIES))
    try:
        shim = ab.proxy_for("zz99-deadbeef")
        jar = [h for h in shim.state.upstream().handlers
               if hasattr(h, "cookiejar")][0].cookiejar
        assert {c.name for c in jar} == {"GAMESESSION", "AWSALBAPP-0"}
    finally:
        ab.use_shared_card(None)
        ab.release_proxy("zz99-deadbeef")
