"""The baseline strip, tested as a module for the first time.

`tools/ablate_baselines.py` decides whether the baseline-free arm measures what
it claims to, and nothing imported it under test. That is how three separate
leaks shipped: the strip was never installed by the drivers, `meta.json` kept
`action_budget` (the baseline total times five), and `ARC_API_KEY` stayed in the
solver's environment with `GET /api/games` one call away.

Every test here is a leak that actually happened, pinned so it cannot come back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from athanor.ccarc3 import Ccarc3Config, GameInfo, arc_proxy
from athanor.ccarc3 import session as sess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import ablate_baselines as ab  # noqa: E402  -- needs the sys.path line above

INFO = GameInfo("zz99-deadbeef", "Test", ("click",), (17, 38, 31))


@pytest.fixture(autouse=True)
def _restore():
    """`install()` rewrites two modules' globals and one env var. Put them back."""
    import athanor.ccarc3 as pkg

    saved = (sess.build_workspace, pkg.build_workspace)
    yield
    sess.build_workspace, pkg.build_workspace = saved
    for gid in list(ab._proxies):
        ab.release_proxy(gid)
    arc_proxy.set_budget(0)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "test-key-not-real")
    monkeypatch.delenv("CCARC3_PROXY_URL", raising=False)
    return sess.build_workspace(
        Ccarc3Config(INFO.game_id, out_dir=tmp_path), INFO
    )


def test_the_strip_removes_every_baseline_a_solver_can_read(workspace):
    ab.strip_baselines(workspace.root)
    root = workspace.root

    assert "baseline_actions=()" in (root / "session.py").read_text()
    assert "17" not in (root / "AGENTS.md").read_text()
    assert "## 6. " not in (root / "DOCTRINE.md").read_text()

    meta = json.loads((root / "meta.json").read_text())
    assert "baseline_actions" not in meta
    # **And the budget, which is the total times `budget_multiple`.** The solver
    # caught doing this said so outright: "they total 518 across six levels, and
    # I have a budget of 2590, which is 5 times the baseline". Leaving a
    # trivially invertible function of the secret is not a strip.
    assert "action_budget" not in meta


def test_the_strip_scans_the_whole_workspace_not_just_what_it_edited(workspace):
    """The old check looked at the three files it had just rewritten.

    A leak check scoped to your own edits can only ever confirm your own edits.
    Any file the solver can open is in scope.
    """
    (workspace.root / "notes" / "orientation.md").write_text(
        'read this: "baseline_actions": [17, 38, 31]\n'
    )
    with pytest.raises(RuntimeError, match="still reachable"):
        ab.strip_baselines(workspace.root)


def test_the_strip_does_not_trip_over_its_own_blanking(workspace):
    """`baseline_actions=()` is what the strip produces; it must not self-flag.

    **This asserted nothing.** It called the strip twice and relied on "no
    exception", which passes if the strip becomes a no-op, if its guard is
    deleted, or if the second call quietly corrupts the file. The property has
    three parts and each is now stated: the first call strips, the second
    changes nothing, and the guard it must not trip is still armed.
    """
    sp = workspace.root / "session.py"
    assert "baseline_actions=()" not in sp.read_text(), (
        "the fixture is already stripped, so this test would pass vacuously"
    )

    ab.strip_baselines(workspace.root)
    after_first = sp.read_text()
    assert "baseline_actions=()" in after_first, "the first call did not strip"

    ab.strip_baselines(workspace.root)
    assert sp.read_text() == after_first, (
        "the second call changed the file -- the strip is not idempotent"
    )


def test_the_strip_still_raises_when_it_finds_nothing_to_strip(workspace):
    """The guard the idempotence above must not trip -- proven still armed.

    `strip_baselines` raises when it finds neither a populated
    `baseline_actions=(...)` nor the blanked `baseline_actions=()`, because that
    means the template moved and the strip silently did nothing. Without this
    case, "the second call does not raise" is satisfied just as well by a guard
    that never raises at all.
    """
    sp = workspace.root / "session.py"
    sp.write_text(sp.read_text().replace("baseline_actions=", "baselines_removed="),
                  encoding="utf-8")
    with pytest.raises(RuntimeError, match="baseline_actions not found"):
        ab.strip_baselines(workspace.root)


def test_the_strip_removes_prose_by_the_paragraph_not_by_the_line(workspace):
    """Line-wise removal on hard-wrapped prose is a shredder, not a redaction.

    `AGENTS.md`'s baseline paragraph wraps across four lines, two of which
    contain the word. Deleting those two left the other two standing as a pair of
    orphaned half-sentences — *"must also discover them, hence the larger
    budget. But if you are several times / wrong — go re-explore rather than
    grind."* — which is what all eight clean rollouts read during orientation.
    Incoherent, and still asserting an allowance the harness had spent four
    commits removing.

    A sentence is the smallest unit that means anything, and a hard-wrapped
    paragraph is the smallest unit that reliably contains whole sentences.
    """
    ab.strip_baselines(workspace.root)
    body = (workspace.root / "AGENTS.md").read_text()

    for orphan in ("must also discover them", "But if you are several times",
                   "wrong — go re-explore"):
        assert orphan not in body, f"line-wise shredding is back: {orphan!r}"
    # Nothing left dangling: every prose line outside a fence still ends in a
    # sentence or continues into one.
    assert "\n\n\n" not in body
    assert "baseline" not in body.lower()


def test_the_strip_keeps_the_code_examples_it_only_needs_to_edit(workspace):
    """Paragraph-dropping must not swallow a fenced block over one comment.

    The driving example is one paragraph by the prose rule, so treating it as
    prose would take `client.reset()` and `client.act()` out along with the
    `client.pace()` line that actually mentions a baseline — leaving a
    baseline-free solver with no documented way to drive the game at all. Code
    has no wrapped sentences, so inside a fence the line is the unit.
    """
    ab.strip_baselines(workspace.root)
    body = (workspace.root / "AGENTS.md").read_text()

    assert "client.reset()" in body
    assert "client.act(6, x=10, y=20)" in body
    assert "arc.effective_actions" in body
    # `status()` survives and `pace()` does not: under withheld baselines the
    # first still reports level, state, actions and the completion cap, and the
    # second can only report ratios against numbers that are gone.
    assert "client.status()" in body
    assert "client.pace()" not in body


def test_assert_installed_fails_before_install_is_called():
    """The failure mode is a *successful* run against a contaminated workspace.

    Nothing in a log or a result file shows it. Eleven runs went that way before
    a solver quoted the numbers back and gave it away.
    """
    with pytest.raises(RuntimeError, match="NOT installed"):
        ab.assert_installed()


def test_install_withholds_the_key_and_the_cap_from_the_solver(tmp_path, monkeypatch):
    """The two channels no file edit can close.

    `ARC_API_KEY` in the child's environment makes `/api/games` -- which returns
    `baseline_actions` for all 25 environments -- a nine-line urllib call, and two
    runs did exactly that. `CCARC3_MAX_ACTIONS` is the baseline total times
    `budget_multiple`, whose value is a default in this package's own source on
    the solver's PYTHONPATH.
    """
    monkeypatch.setenv("ARC_API_KEY", "test-key-not-real")
    # **Set the second name too, or the assertion below cannot fail.** It was
    # absent from the environment, so `"ARCPRIZE_API_KEY" not in ws.env` was
    # vacuously true whether or not the strip ran -- and deleting
    # `env.pop("ARCPRIZE_API_KEY", None)` from build_workspace survived the whole
    # 1079-test suite. The harness strips two credential names and only one of
    # them was verified; `ablate_baselines.assert_installed` and
    # `proofread_trace` both guard the pair, so the project treats it as live.
    monkeypatch.setenv("ARCPRIZE_API_KEY", "test-key-not-real-either")
    monkeypatch.delenv("CCARC3_PROXY_URL", raising=False)

    ab.install()
    ab.assert_installed()

    ws = sess.build_workspace(Ccarc3Config(INFO.game_id, out_dir=tmp_path), INFO)
    assert "ARC_API_KEY" not in ws.env
    assert "ARCPRIZE_API_KEY" not in ws.env
    assert "CCARC3_MAX_ACTIONS" not in ws.env
    assert ws.env["CCARC3_ARC_ROOT"].startswith("http://127.0.0.1:")

    # Withheld, not abolished: the cap still binds, in a process the solver does
    # not run and cannot edit -- and on *this game's* shim, not a shared one.
    assert ab.proxy_for(INFO.game_id).max_actions == INFO.suggested_budget(5.0)
    assert ws.env["CCARC3_ARC_ROOT"] == ab.proxy_for(INFO.game_id).url


def test_a_resumed_game_seeds_the_cap_from_what_it_already_spent(tmp_path, monkeypatch):
    """Otherwise the budget grows by its full size on every container replacement."""
    monkeypatch.setenv("ARC_API_KEY", "test-key-not-real")
    monkeypatch.delenv("CCARC3_PROXY_URL", raising=False)
    root = tmp_path / INFO.game_id
    root.mkdir(parents=True)
    (root / "trace.jsonl").write_text("")
    (root / "trace.state.json").write_text(json.dumps({"actions_used": 291}))

    ab.install()
    sess.build_workspace(
        Ccarc3Config(INFO.game_id, out_dir=tmp_path, fresh=False), INFO
    )
    assert ab.proxy_for(INFO.game_id).actions_used == 291


def test_the_strip_turns_the_raw_conditionals_into_statements(workspace):
    """A branch whose true arm is unreachable should not be written as a branch.

    The doctrine serves runs that *can* see the baselines too, so it phrases the
    replay rule conditionally: "if you cannot compute `raw`, replay anyway",
    offers `arc.score_run(transitions, baselines)` for the case where you can,
    and says "if you do not know the baselines, you still know the cap". In a
    stripped workspace every antecedent holds unconditionally — `baseline_actions`
    is `()` and there is no path to a median.

    Leaving them conditional is not neutral. It tells the solver that computing
    `raw` might be possible, which invites reaching for whatever looks closest:
    `tu93` estimated it from an in-game drain bar and concluded a replay would not
    help (worth +0.1202), and `bp35` read the harness's own `cap 1.000` as its
    score and stopped (worth +0.2748).
    """
    ab.strip_baselines(workspace.root)
    doc = (workspace.root / "DOCTRINE.md").read_text()

    for conditional in ("If you cannot compute `raw`",
                        "If you *do* have the per-level baselines",
                        "If you do not know the baselines, you still know the cap"):
        assert conditional not in doc, f"conditional survived the strip: {conditional!r}"

    assert "You cannot compute `raw` on this run" in doc
    assert "There is no call that will give you `raw`" in doc
    # The two proxies solvers actually reached for, named so they are not reached
    # for again.
    assert "an in-game meter is not the human median, and" in doc
    assert "neither is `cap`" in doc


def test_the_strip_drops_the_fenced_baseline_only_passages(workspace):
    """Sentence surgery only reaches the sentences you thought of.

    The previous fix rewrote three `raw` conditionals into statements and left
    two whole passages standing that a baseline-free solver cannot use: the
    "judge it by arithmetic" bullets, whose first branch is *a replay gains you
    exactly nothing... do not spend actions on it*, and the binding-term table,
    whose first row is *a replay is worth zero*. Both require knowing `raw` to
    evaluate, and a reader who cannot evaluate the antecedent takes the
    conclusion — which is the `bp35` failure the rewrite existed to prevent,
    still fully available two edits later.

    So the doctrine fences those passages and the strip drops whatever is
    between the markers. Rewording a fenced passage cannot reopen the hole.
    """
    ab.strip_baselines(workspace.root)
    doc = (workspace.root / "DOCTRINE.md").read_text()

    assert "BASELINE-ONLY" not in doc, "the markers themselves must not ship"
    for gone in ("A replay gains you exactly nothing",
                 "Do not spend actions on it",
                 "| binding term |",
                 "A replay is worth zero"):
        assert gone not in doc, f"fenced passage survived the strip: {gone!r}"

    # What the fences guard is arithmetic, not the conclusion. The unconditional
    # rule and the evidence for it both stay.
    assert "You cannot compute `raw` on this run" in doc
    assert "the worked case" in doc
    assert "\n\n\n" not in doc, "a removed fence left a visible hole"


def test_an_unfenced_doctrine_is_refused_rather_than_shipped(workspace, monkeypatch):
    """A fence lost to an edit puts the harmful branch back. Fail loudly."""
    doc = workspace.root / "DOCTRINE.md"
    doc.write_text(doc.read_text().replace("<!-- BASELINE-ONLY -->", ""))
    with pytest.raises(RuntimeError, match="BASELINE-ONLY"):
        ab.strip_baselines(workspace.root)


def test_a_renumbered_doctrine_cannot_silently_ship_section_six(workspace):
    """The one strip edit that asserted nothing, and it guards real medians.

    §6's worked examples carry per-level human medians, written as prose in a
    table rather than as a `baseline_actions` array — so the post-strip scan
    cannot see them either. A bare literal match on the heading meant an
    ordinary renumbering turned the removal into a no-op with every test still
    green, and the stripped doctrine already jumps 5b to 7, which makes
    renumbering the natural tidy-up.
    """
    doc = workspace.root / "DOCTRINE.md"
    doc.write_text(doc.read_text().replace("## 6. ", "## 5c. ", 1))
    with pytest.raises(RuntimeError, match="no '## 6\\. ' section"):
        ab.strip_baselines(workspace.root)


@pytest.mark.parametrize("name", ["notes/leak.csv", "rules", "probe.sh", "notes/.env"])
def test_the_post_strip_scan_reads_every_file_not_six_suffixes(workspace, name):
    """The comment said "any file the solver can open is in scope"; the code
    listed `.py .md .json .txt .yaml .yml`.

    So a workspace could carry a median in `notes/x.csv` or an extensionless
    `rules` file and strip cleanly — the same "scoped to what the author thought
    of" failure the whole-workspace scan was written to fix, one level down.
    """
    target = workspace.root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('"baseline_actions": [22, 123, 73]\n')
    with pytest.raises(RuntimeError, match="still reachable"):
        ab.strip_baselines(workspace.root)
