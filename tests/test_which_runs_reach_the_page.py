"""The three functions that decide what the published page says, none covered.

`mark_contaminated` voids a run, `mark_generation` tiers it, and
`current_generation_only` decides whether it is rendered at all. Coverage over
the whole suite on 2026-08-09 shows every one of them dark. They are the last
decision logic on the reporting side, and between them they decide which numbers
a reader sees and which are silently absent — the failure mode being a page that
looks complete.

Each of these has a written rationale that nothing was holding it to:

* the generation filter keys on the **epoch**, not on `tier`. Keying it on
  `tier == "current"` renders empty every time the harness is improved, which it
  did twice in one morning.
* `void` and `excluded` outrank generation, because a contaminated run's
  generation is not the interesting fact about it.
* nothing is deleted — `--all-generations` still renders everything, because
  dropping evidence to tidy a page would be the opposite of what the page is for.
* the median-in-doctrine window is a **provenance** test, not a content one: it
  voids by *when the run read its doctrine*, because the worked example
  `65.533 = (17/21)^2 x 100` states a real per-level median in prose and no
  array-shaped scan can see it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_trace_audit as bta  # noqa: E402

BEFORE = "2026-08-06T12:00:00Z"          # older than the generation epoch
AFTER = "2026-08-08T12:00:00Z"           # newer than it
IN_WINDOW = "2026-08-05T00:00:00Z"       # inside the median-in-doctrine window


def _row(rid: str, started: str | None = AFTER, **kw) -> dict:
    row = {"id": rid, "tier": "clean", "started": started}
    row.update(kw)
    return row


# --- current_generation_only ---------------------------------------------- #

def test_only_runs_at_or_after_the_epoch_are_rendered():
    runs = [_row("new", AFTER), _row("old", BEFORE)]
    data = {"new": {"attempts": []}, "old": {"attempts": []}}

    kept, kept_data, dropped = bta.current_generation_only(runs, data, keep_all=False)

    assert [r["id"] for r in kept] == ["new"]
    assert set(kept_data) == {"new"}
    assert dropped == 1


def test_the_filter_is_the_epoch_and_not_the_tier():
    """Keying on `tier == "current"` renders the page empty after any fix.

    It did, twice in one morning: once for adding `client.py` to the digest,
    once for rewriting the doctrine's `raw` conditionals. Improving the harness
    after reading a batch is the working method here, so `superseded` must still
    reach the page.
    """
    runs = [_row("superseded_but_recent", AFTER, tier="superseded")]
    kept, _, dropped = bta.current_generation_only(runs, {}, keep_all=False)
    assert [r["id"] for r in kept] == ["superseded_but_recent"], (
        "a superseded run inside the generation was dropped; the page renders "
        "empty within minutes of any batch finishing"
    )
    assert dropped == 0


@pytest.mark.parametrize("tier", ["void", "excluded"])
def test_a_void_or_excluded_run_never_reaches_the_page(tier):
    runs = [_row("bad", AFTER, tier=tier)]
    kept, _, dropped = bta.current_generation_only(runs, {}, keep_all=False)
    assert kept == [] and dropped == 1


def test_a_run_with_no_start_time_cannot_be_placed_and_is_not_rendered():
    runs = [_row("unknown", None)]
    kept, _, _ = bta.current_generation_only(runs, {}, keep_all=False)
    assert kept == [], "a run that cannot be dated was rendered as current"


def test_all_generations_deletes_nothing():
    """The store keeps every span of every generation, by design."""
    runs = [_row("new", AFTER), _row("old", BEFORE), _row("bad", AFTER, tier="void")]
    data = {r["id"]: {"attempts": []} for r in runs}

    kept, kept_data, dropped = bta.current_generation_only(runs, data, keep_all=True)

    assert len(kept) == 3 and set(kept_data) == {"new", "old", "bad"}
    assert dropped == 0


# --- mark_generation ------------------------------------------------------- #

def test_a_recorded_surface_beats_the_commit_count(monkeypatch):
    """Counting commits counts diffs, not differences.

    `sb26` was superseded by a 289-line refactor that left every byte the solver
    reads identical. Where a run recorded its surface, that digest decides.
    """
    monkeypatch.setattr(bta, "harness_commits",
                        lambda: ["2026-08-09T00:00:00+00:00"] * 5)
    monkeypatch.setattr(bta, "reference_digest", lambda gid: "abc123")

    same = _row("zz99-deadbeef", AFTER, surface="abc123")
    moved = _row("yy88-cafebabe", AFTER, surface="different")
    bta.mark_generation([same, moved])

    assert (same["tier"], same["estimated"]) == ("current", False), (
        "five commits landed after this run and its surface is unchanged; "
        "it is current"
    )
    assert (moved["tier"], moved["estimated"]) == ("superseded", False)


def test_a_run_without_a_surface_is_marked_estimated(monkeypatch):
    """A guess must not read as a measurement."""
    monkeypatch.setattr(bta, "harness_commits", lambda: [])
    monkeypatch.setattr(bta, "reference_digest", lambda gid: "abc123")

    row = _row("zz99-deadbeef", AFTER)
    bta.mark_generation([row])

    assert row["tier"] == "current" and row["estimated"] is True
    assert row["behind"] == 0


@pytest.mark.parametrize("tier", ["void", "excluded"])
def test_contamination_outranks_generation(monkeypatch, tier):
    monkeypatch.setattr(bta, "harness_commits", lambda: [])
    monkeypatch.setattr(bta, "reference_digest", lambda gid: "abc123")

    row = _row("zz99-deadbeef", AFTER, tier=tier, surface="abc123")
    bta.mark_generation([row])

    assert row["tier"] == tier, (
        "a contaminated run was retiered by generation; its generation is not "
        "the interesting fact about it"
    )


# --- mark_contaminated, the provenance half -------------------------------- #

def test_a_run_that_read_a_doctrine_stating_a_median_is_voided(monkeypatch):
    """No content scan can see `65.533 = (17/21)^2 x 100`; the clock can."""
    monkeypatch.setattr(bta, "harness_commits", lambda: [])
    monkeypatch.setattr(bta, "reference_digest", lambda gid: "")

    inside = _row("zz99-deadbeef", IN_WINDOW)
    outside = _row("yy88-cafebabe", AFTER)
    voided = bta.mark_contaminated({}, [inside, outside])

    assert "zz99-deadbeef" in voided, (
        "a run started inside the median-in-doctrine window was left clean"
    )
    assert inside["tier"] == "void"
    assert "yy88-cafebabe" not in voided and outside["tier"] != "void"


def test_the_window_is_half_open_at_the_fix(monkeypatch):
    """`f66cc91` made the example symbolic, so a run at that instant is clean."""
    monkeypatch.setattr(bta, "harness_commits", lambda: [])
    monkeypatch.setattr(bta, "reference_digest", lambda gid: "")

    at_fix = _row("zz99-deadbeef",
                  bta.MEDIAN_IN_DOCTRINE_UNTIL.isoformat().replace("+00:00", "Z"))
    at_start = _row("yy88-cafebabe",
                    bta.MEDIAN_IN_DOCTRINE_FROM.isoformat().replace("+00:00", "Z"))
    voided = bta.mark_contaminated({}, [at_fix, at_start])

    assert "zz99-deadbeef" not in voided, "the run at the fix instant was voided"
    assert "yy88-cafebabe" in voided, "the run at the window's start was not voided"


def test_an_unreachable_api_does_not_disable_the_checks_that_do_not_need_it(monkeypatch):
    """The value scan needs the network. Nothing else in this function does.

    `mark_contaminated` used to `return []` when `list_games()` raised, which
    took the provenance void, `mark_generation` and `mark_clean` down with it --
    so a build made without a key, or during an ARC outage, published
    contaminated runs as clean, left every tier stale, and never recomputed the
    `clean` flag the page's filter reads. Its message said "tiers left as they
    were", which is true and describes about a quarter of what it did.
    """
    monkeypatch.setattr(bta, "harness_commits", lambda: [])
    monkeypatch.setattr(bta, "reference_digest", lambda gid: "")

    def _no_api():
        raise RuntimeError("no ARC_API_KEY in this environment")

    import athanor.ccarc3 as pkg
    monkeypatch.setattr(pkg, "list_games", _no_api, raising=False)

    inside = _row("zz99-deadbeef", IN_WINDOW)
    outside = _row("yy88-cafebabe", AFTER)
    data = {"zz99-deadbeef": {"result": {}, "attempts": []},
            "yy88-cafebabe": {"result": {}, "attempts": []}}

    voided = bta.mark_contaminated(data, [inside, outside])

    assert "zz99-deadbeef" in voided, (
        "with the API unreachable, the provenance void -- arithmetic on two "
        "hard-coded timestamps -- stopped running too"
    )
    assert inside["tier"] == "void"
    assert outside["tier"] in {"current", "superseded"}, (
        f"mark_generation did not run either: tier is {outside['tier']!r}"
    )
    assert "clean" in outside, "mark_clean did not run either"
