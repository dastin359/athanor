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
    assert "17" not in (root / "CLAUDE.md").read_text()
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
    """`baseline_actions=()` is what the strip produces; it must not self-flag."""
    ab.strip_baselines(workspace.root)
    ab.strip_baselines(workspace.root)      # idempotent, and still clean


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
    monkeypatch.delenv("CCARC3_PROXY_URL", raising=False)

    ab.install()
    ab.assert_installed()

    ws = sess.build_workspace(Ccarc3Config(INFO.game_id, out_dir=tmp_path), INFO)
    assert "ARC_API_KEY" not in ws.env
    assert "ARCPRIZE_API_KEY" not in ws.env
    assert "CCARC3_MAX_ACTIONS" not in ws.env
    assert ws.env["CCARC3_ARC_ROOT"].startswith("http://127.0.0.1:")

    # Withheld, not abolished: the cap still binds, in a process the solver does
    # not run and cannot edit.
    assert arc_proxy.MAX_ACTIONS == INFO.suggested_budget(5.0)


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
    assert arc_proxy.ACTIONS_USED == 291
