"""A shared-card sweep must not be able to mint its way out of a lost binding.

On 2026-08-12 a workspace was rebuilt at 00:00:36 PDT while the driver's
`_shared_card` module global was unset. The launcher was generated with
`card_id=''`, `ArcClient.open()` could not tell that from a standalone run, and
the solver quietly opened its own card and kept scoring onto it. The 877-action
play on the sweep's card became unreadable, unplayable and unclosable.

Nothing failed. The driver's entire card-succession apparatus — liveness probe,
stranding refusal, committed history — sits on a path that was never taken. On a
25-game sweep the same fault yields up to 25 cards and a submission worth
nothing, with every game reporting success.

Three layers here, because the first two can each be lost:
  1. the rebuild recovers the card from the workspace's own checkpoint;
  2. the launcher carries `require_card` alongside the id;
  3. the client refuses to mint when `require_card` is set.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from athanor.ccarc3.client import ArcClient
from athanor.ccarc3.session import Ccarc3Config, GameInfo, build_workspace

INFO = GameInfo(game_id="gm01-abcd", title="X", tags=("a",), baseline_actions=())
CARD = "50e3871e-c684-404f-a29a-f87023788f8f"


# --------------------------------------------------------------------------- #
# 3. the client refuses
# --------------------------------------------------------------------------- #


def test_a_shared_card_run_refuses_to_mint_its_own(tmp_path):
    """The last line of defence, and the one that makes the failure loud."""
    c = ArcClient("gm01-abcd", trace_path=str(tmp_path / "trace.jsonl"),
                  api_key="k", root="http://127.0.0.1:1", require_card=True)
    with pytest.raises(RuntimeError) as exc:
        c.open()
    msg = str(exc.value)
    assert "shared-card sweep" in msg and "Refusing" in msg, msg
    assert "does not know about" in msg, "the message must say what the harm is"


def test_a_standalone_run_still_mints(tmp_path, monkeypatch):
    """The refusal must not break the ordinary one-card-per-game run, which is
    every ablation and every single-game debug session."""
    c = ArcClient("gm01-abcd", trace_path=str(tmp_path / "trace.jsonl"),
                  api_key="k", root="http://127.0.0.1:1")
    assert c.require_card is False
    seen = {}

    def fake_post(url, payload, key, opener=None):
        seen["url"] = url
        return {"card_id": "fresh-card"}

    monkeypatch.setattr("athanor.ccarc3.client._post", fake_post)
    c.open()
    assert c.card_id == "fresh-card" and c._owns_card is True
    assert seen["url"].endswith("/api/scorecard/open")


# --------------------------------------------------------------------------- #
# 2. the launcher carries the flag
# --------------------------------------------------------------------------- #


def test_a_shared_card_workspace_is_born_refusing(tmp_path):
    out = build_workspace(
        Ccarc3Config(game_id="gm01-abcd", out_dir=str(tmp_path), card_id=CARD), INFO)
    launcher = (pathlib.Path(out.root) / "session.py").read_text()
    assert f"card_id={CARD!r}" in launcher
    assert "require_card=True" in launcher, (
        "a workspace built for a shared card must carry the refusal with it, so "
        "no later rebuild or resume can decide minting is an acceptable default")


def test_a_standalone_workspace_carries_neither(tmp_path):
    out = build_workspace(
        Ccarc3Config(game_id="gm01-abcd", out_dir=str(tmp_path)), INFO)
    launcher = (pathlib.Path(out.root) / "session.py").read_text()
    assert "card_id" not in launcher and "require_card" not in launcher


# --------------------------------------------------------------------------- #
# 1. the rebuild recovers the card
# --------------------------------------------------------------------------- #


def test_a_rebuild_recovers_the_card_from_the_workspace_checkpoint(tmp_path, capsys):
    """The exact 00:00:36 failure: the global is gone, the checkpoint is not."""
    import ablate_baselines as ab

    root = tmp_path / "gm01-abcd"
    root.mkdir(parents=True)
    (root / "trace.state.json").write_text(json.dumps(
        {"game_id": "gm01-abcd", "card_id": CARD, "actions_used": 877}))

    cfg = Ccarc3Config(game_id="gm01-abcd", out_dir=str(tmp_path))
    assert not cfg.card_id
    assert ab._card_from_existing_workspace(cfg) == CARD

    ab._shared_card = None                       # the lost global
    recovered = ab._card_from_existing_workspace(cfg)
    assert recovered == CARD, (
        "a rebuild with no global and a perfectly good checkpoint beside it must "
        "not produce an empty card_id")


def test_recovery_is_silent_about_a_workspace_that_never_had_a_card(tmp_path):
    """A first build has no checkpoint; that is not a fault and must not warn."""
    import ablate_baselines as ab
    cfg = Ccarc3Config(game_id="gm01-abcd", out_dir=str(tmp_path))
    assert ab._card_from_existing_workspace(cfg) == ""
    (tmp_path / "gm01-abcd").mkdir(parents=True)
    (tmp_path / "gm01-abcd" / "trace.state.json").write_text("{ truncated")
    assert ab._card_from_existing_workspace(cfg) == "", "unreadable is not a card"


def test_the_rebuild_actually_consults_the_checkpoint(tmp_path, monkeypatch):
    """The wiring, not just the helper.

    A first version of this file tested `_card_from_existing_workspace` on its
    own and never checked that `build_without_baselines` calls it — so replacing
    the call with `recovered = ""` passed the whole suite. A recovery routine
    nobody invokes is the same defect as a guard nobody calls, which is what this
    entire file is about.
    """
    import ablate_baselines as ab

    root = tmp_path / "gm01-abcd"
    root.mkdir(parents=True)
    (root / "trace.state.json").write_text(json.dumps(
        {"game_id": "gm01-abcd", "card_id": CARD, "actions_used": 877}))

    class _Proxy:
        url = "http://127.0.0.1:1"
        def set_budget(self, *a, **k): pass

    seen = {}

    class _Stop(Exception):
        pass

    def _capture(config, info=None, *, arc_root=None):
        seen["card_id"] = config.card_id
        raise _Stop                                  # stop before the heavy tail

    monkeypatch.setattr(ab, "proxy_for", lambda gid: _Proxy())
    monkeypatch.setattr(ab, "_original_build", _capture)
    monkeypatch.setattr(ab, "_shared_card", None)    # the global that was lost

    cfg = Ccarc3Config(game_id="gm01-abcd", out_dir=str(tmp_path))
    with pytest.raises(_Stop):
        ab.build_without_baselines(cfg)

    assert seen["card_id"] == CARD, (
        "the rebuild handed an empty card_id to build_workspace even though the "
        "workspace's own checkpoint named the sweep's card — this is the "
        "00:00:36 failure exactly")
