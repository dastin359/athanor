"""`CCARC3_ONLY` scopes the sweep, and unset it must change nothing.

`GAMES` is the full 25-environment benchmark with no way to run a subset, so a
sweep interrupted at game 14 could only be resumed by re-running the whole list
and leaning on the skip-if-banked guard, and a single-game validation of the
harness could not go through this driver at all.

The load-bearing property is the NO-OP one. A filter that silently narrowed the
submission sweep would be far worse than no filter: the run would finish, print
"all N have a clean run", and produce a scorecard missing most of the benchmark.
So the unset case is asserted against the literal list in the source, not
against itself.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PY_BIN = REPO / ".venv" / "bin" / "python"
DRIVER = REPO / "tools" / "clean_rollouts.py"


def _games(only: str | None) -> tuple[int, list[str], str]:
    env = {k: v for k, v in os.environ.items() if k != "CCARC3_ONLY"}
    env.update(CCARC3_SCRATCH="/tmp/does-not-exist", ARC_API_KEY="dummy")
    if only is not None:
        env["CCARC3_ONLY"] = only
    r = subprocess.run(
        [str(PY_BIN), "-c",
         "import sys; sys.path[:0]=['tools','src']\n"
         "import clean_rollouts as cr\n"
         "print('GAMES=' + ','.join(cr.GAMES))"],
        capture_output=True, text=True, cwd=str(REPO), env=env, timeout=180,
    )
    line = next((l for l in r.stdout.splitlines() if l.startswith("GAMES=")), "")
    ids = [g for g in line.removeprefix("GAMES=").split(",") if g]
    return r.returncode, ids, r.stdout + r.stderr


def _literal_games() -> list[str]:
    """The list as written in the source, independent of any import."""
    src = DRIVER.read_text(encoding="utf-8")
    body = src[src.index("GAMES = ["):src.index("\n]\n", src.index("GAMES = ["))]
    return re.findall(r'"([a-z0-9]{4}-[0-9a-f]{8})"', body)


def test_unset_is_a_no_op() -> None:
    """Byte-identical to the literal list. This is the property that matters."""
    rc, ids, out = _games(None)
    assert rc == 0, out[-1500:]
    assert ids == _literal_games(), (
        "the sweep's game list changed with CCARC3_ONLY unset -- a submission "
        "sweep would silently cover a different set than the source says"
    )
    assert len(ids) == 25


def test_an_empty_value_is_also_a_no_op() -> None:
    """`CCARC3_ONLY=` in a stale env file must not narrow the sweep."""
    rc, ids, out = _games("")
    assert rc == 0, out[-1500:]
    assert ids == _literal_games()


def test_a_bare_prefix_selects_one_game() -> None:
    rc, ids, out = _games("bp35")
    assert rc == 0, out[-1500:]
    assert len(ids) == 1 and ids[0].startswith("bp35-")


def test_a_full_id_and_a_prefix_can_be_mixed() -> None:
    rc, ids, out = _games("bp35-0a0ad940,sb26")
    assert rc == 0, out[-1500:]
    assert len(ids) == 2
    assert {i.split("-")[0] for i in ids} == {"bp35", "sb26"}


def test_order_follows_the_source_not_the_argument() -> None:
    """Queue order is deliberate; a filter must not reorder it."""
    rc, ids, _ = _games("re86,sb26")
    assert rc == 0
    literal = _literal_games()
    assert ids == [g for g in literal if g.split("-")[0] in {"re86", "sb26"}]


def test_a_typo_refuses_rather_than_running_nothing() -> None:
    """Zero games would finish instantly and report every game complete.

    That is the silent-success shape this project keeps finding, and a mistyped
    prefix is the likely cause.
    """
    rc, ids, out = _games("bp53")          # transposed digits
    assert rc != 0, f"an unmatched filter ran anyway: {out[-800:]}"
    assert "matched none" in out
    assert not ids


def test_whitespace_and_stray_commas_are_tolerated() -> None:
    rc, ids, out = _games(" bp35 , , sb26 ")
    assert rc == 0, out[-1500:]
    assert len(ids) == 2
