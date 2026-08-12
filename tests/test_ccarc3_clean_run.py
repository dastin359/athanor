"""The first-class clean single-game command fails closed at every launch."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from athanor.ccarc3 import Ccarc3Config, GameInfo
from athanor.ccarc3 import clean
from athanor.ccarc3 import cli
from athanor.ccarc3 import session as sess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import ablate_baselines as ab  # noqa: E402


GID = "zz99-deadbeef"
INFO = GameInfo(GID, "Test", ("click",), (17, 38, 31))


def _args(*, resume: bool = False) -> list[str]:
    command = ["codex", "exec"]
    if resume:
        command.append("resume")
    command += [
        "--json", "--ignore-user-config", "--ignore-rules",
        "--disable", "memories", "--skip-git-repo-check",
    ]
    return command


def _workspace(tmp_path: Path) -> sess.Workspace:
    root = tmp_path / GID
    root.mkdir()
    (root / "meta.json").write_text(json.dumps({"game_id": GID}))
    (root / "session.py").write_text("baseline_actions=(),\n")
    return sess.Workspace(
        root=root,
        config=Ccarc3Config(GID, out_dir=tmp_path),
        info=INFO,
        env={
            "CCARC3_HIDE_BASELINES": "1",
            "CCARC3_ARC_ROOT": "http://127.0.0.1:43210",
        },
    )


def test_clean_run_is_a_distinct_command_with_an_external_default():
    plain = cli.build_parser().parse_args(["run", "--game", GID])
    clean_args = cli.build_parser().parse_args(["clean-run", "--game", GID])

    assert plain.out_dir == "runs/ccarc3", "the historical plain command changed"
    assert Path(clean_args.out_dir) == clean.DEFAULT_OUT_DIR
    assert clean_args.max_nudges == 3
    assert clean._containing_repository(Path(clean_args.out_dir) / GID) is None


def test_both_initial_and_resume_launches_pass_the_same_clean_assertion(tmp_path):
    ws = _workspace(tmp_path)
    clean.assert_clean_launch(ws, _args())
    clean.assert_clean_launch(ws, _args(resume=True))


@pytest.mark.parametrize(
    ("break_it", "message"),
    [
        (lambda ws, args: ws.env.__setitem__("ARC_API_KEY", "not-a-real-key"),
         "child environment contains ARC_API_KEY"),
        (lambda ws, args: ws.env.pop("CCARC3_HIDE_BASELINES"),
         "does not set CCARC3_HIDE_BASELINES=true"),
        (lambda ws, args: args.remove("--ignore-rules"),
         "missing --ignore-rules"),
        (lambda ws, args: setattr(ws, "root", Path(__file__).resolve().parents[1]),
         "workspace is inside Git repository"),
    ],
)
def test_a_false_isolation_invariant_refuses_the_launch(tmp_path, break_it, message):
    ws, args = _workspace(tmp_path), _args()
    break_it(ws, args)
    with pytest.raises(RuntimeError, match=message):
        clean.assert_clean_launch(ws, args)


def test_a_parent_credential_in_any_workspace_file_refuses_resume(
    tmp_path, monkeypatch,
):
    ws = _workspace(tmp_path)
    monkeypatch.setenv("ARC_API_KEY", "test-secret-that-must-not-ship")
    (ws.root / "notes").mkdir()
    (ws.root / "notes" / "old.txt").write_text("test-secret-that-must-not-ship")

    with pytest.raises(RuntimeError, match="credential appears in workspace files"):
        clean.assert_clean_launch(ws, _args(resume=True))


def test_a_workspace_symlink_refuses_without_reading_its_target(tmp_path):
    ws = _workspace(tmp_path)
    (ws.root / "notes").mkdir()
    (ws.root / "notes" / "host").symlink_to("/definitely/not/a/workspace/file")

    with pytest.raises(RuntimeError, match="workspace contains symlinks"):
        clean.assert_clean_launch(ws, _args(resume=True))


def test_a_different_loopback_proxy_is_not_accepted_for_this_game(tmp_path):
    ws = _workspace(tmp_path)

    class WrongProxyBoundary:
        @staticmethod
        def assert_installed():
            return None

        @staticmethod
        def proxy_for(game_id):
            return type("Proxy", (), {"url": "http://127.0.0.1:54321",
                                      "max_actions": INFO.suggested_budget(5.0)})()

    with pytest.raises(RuntimeError, match="not this game's proxy"):
        clean.assert_clean_launch(ws, _args(), ablation=WrongProxyBoundary)


def test_real_strip_is_installed_checked_and_restored_around_one_run(
    tmp_path, monkeypatch,
):
    """No ARC request is made; the fake runner inspects both Codex launch paths."""
    import athanor.ccarc3 as package

    monkeypatch.setenv("ARC_API_KEY", "test-key-not-used-offline")
    monkeypatch.setenv("ARCPRIZE_API_KEY", "second-test-key-not-used-offline")
    monkeypatch.setenv("CCARC3_PROXY_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("CCARC3_MAX_NUDGES", "19")
    original_session_builder = sess.build_workspace
    original_package_builder = package.build_workspace
    seen: list[tuple[sess.Workspace, list[str]]] = []

    def fake_run(config):
        assert sess.build_workspace is ab.build_without_baselines
        ab.assert_installed()
        assert os.environ["CCARC3_MAX_NUDGES"] == "2"
        ws = sess.build_workspace(config, INFO)
        initial = sess.build_cli_args(ws)
        config.launch_validator(ws, initial)
        assert ab.proxy_for(GID).max_actions == INFO.suggested_budget(
            config.budget_multiple
        )
        seen.append((ws, initial))
        ws.session_id = "11111111-2222-3333-4444-555555555555"
        resumed = sess._nudge_args(ws)
        assert resumed is not None
        config.launch_validator(ws, resumed)
        seen.append((ws, resumed))
        return {"game_id": GID, "levels_reached": 1}

    monkeypatch.setattr(sess, "run_game", fake_run)
    config = Ccarc3Config(GID, out_dir=tmp_path / "external")
    out = clean.run_clean_game(config, max_nudges=2)

    assert out["levels_reached"] == 1
    assert len(seen) == 2 and seen[1][1][1:3] == ["exec", "resume"]
    ws = seen[0][0]
    assert sess._env_facts(ws)["child_env"] == {
        "ARC_API_KEY": False,
        "ARCPRIZE_API_KEY": False,
        "CCARC3_MAX_ACTIONS": False,
        "CCARC3_HIDE_BASELINES": True,
        "CCARC3_ARC_ROOT": True,
    }
    assert "baseline_actions=()," in (ws.root / "session.py").read_text()
    assert "action_budget" not in json.loads((ws.root / "meta.json").read_text())
    assert sess.build_workspace is original_session_builder
    assert package.build_workspace is original_package_builder
    assert os.environ["CCARC3_PROXY_URL"] == "http://127.0.0.1:1"
    assert os.environ["CCARC3_MAX_NUDGES"] == "19"
    assert config.launch_validator is None
    assert config.out_dir == tmp_path / "external"


def test_clean_command_routes_to_the_clean_runner(tmp_path, monkeypatch, capsys):
    called = {}

    def fake(config, *, max_nudges):
        called.update(config=config, max_nudges=max_nudges)
        return {"levels_reached": 2}

    monkeypatch.setattr(clean, "run_clean_game", fake)
    assert cli.main([
        "clean-run", "--game", GID, "--out-dir", str(tmp_path), "--max-nudges", "1",
    ]) == 0
    assert called["config"].game_id == GID
    assert called["config"].out_dir == tmp_path
    assert called["max_nudges"] == 1
    assert '"levels_reached": 2' in capsys.readouterr().out
