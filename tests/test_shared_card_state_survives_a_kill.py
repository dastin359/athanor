"""The shared card is the submission's identity; its file must never half-exist.

`save` used a plain `write_text`, leaving a window in which the file exists and
is truncated -- and this container is reaped every 10-50 minutes. The cost of
landing in that window is not one lost write:

    sweep_card() sees the file exists
    -> sc.load() raises on the partial JSON
    -> the driver dies
    -> the supervisor relaunches it ten minutes later into the same crash

forever, with the card carrying the banked games stranded. From the outside a
crash-looping driver and a busy one look identical.

Two fixes, tested here: the write is atomic, and an unreadable file is refused
loudly rather than silently reminted.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

from athanor.ccarc3 import shared_card as sc

REPO = pathlib.Path(__file__).resolve().parents[1]


def _card() -> sc.SharedCard:
    return sc.SharedCard("card-fixture-1234", ({"name": "AWSALBAPP-0",
                                                "value": "cookie-value"},))


def test_save_leaves_no_temporary_behind(tmp_path: pathlib.Path) -> None:
    path = sc.save(_card(), tmp_path / "shared_card.json")
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp")), (
        f"a temporary survived the save: {list(tmp_path.iterdir())!r}"
    )


def test_save_is_atomic_not_truncate_then_write(tmp_path: pathlib.Path) -> None:
    """The load-bearing property: the destination is never observed truncated.

    Asserting "the file is valid after save returns" would pass on the broken
    form too -- the window is DURING the write. So this watches the inode: an
    atomic write replaces the file (new inode), a truncating write reuses it.
    A reader holding the old inode still sees whole content; a reader opening
    the path mid-truncate does not.
    """
    path = tmp_path / "shared_card.json"
    sc.save(_card(), path)
    first_inode = path.stat().st_ino

    sc.save(sc.SharedCard("card-fixture-5678", ()), path)
    second_inode = path.stat().st_ino

    assert first_inode != second_inode, (
        "save reused the destination inode -- it truncated the live file in "
        "place instead of renaming a complete one over it, so a reader can "
        "observe a partial card"
    )
    assert json.loads(path.read_text())["card_id"] == "card-fixture-5678"


def test_saved_card_is_never_world_readable(tmp_path: pathlib.Path) -> None:
    """The cookies ARE the credential half of the card.

    The mode must be set on the temporary file before the rename, or the card
    sits briefly readable at its final path.
    """
    path = sc.save(_card(), tmp_path / "shared_card.json")
    assert path.stat().st_mode & 0o077 == 0, oct(path.stat().st_mode)


def test_roundtrip_preserves_card_and_cookies(tmp_path: pathlib.Path) -> None:
    path = sc.save(_card(), tmp_path / "shared_card.json")
    got = sc.load(path)
    assert got.card_id == "card-fixture-1234"
    assert got.cookies and got.cookies[0]["name"] == "AWSALBAPP-0"


def _sweep_card(scratch: pathlib.Path) -> subprocess.CompletedProcess:
    """Run the real `sweep_card` against a fixture scratchpad."""
    code = (
        "import sys; "
        f"sys.path[:0] = [{str(REPO / 'tools')!r}, {str(REPO / 'src')!r}]; "
        "import clean_rollouts as cr; cr.sweep_card()"
    )
    return subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True, text=True,
        env={**os.environ, "CCARC3_SCRATCH": str(scratch), "ARC_API_KEY": "dummy"},
        timeout=120,
    )


def test_a_truncated_card_file_refuses_instead_of_crash_looping(
    tmp_path: pathlib.Path
) -> None:
    """It must exit with a message, not an unhandled JSONDecodeError.

    Both die, so "the driver stopped" does not distinguish them. What does is
    whether the operator is told which file to fix -- a bare traceback in a
    detached log, relaunched every ten minutes, is what the old form produced.
    """
    (tmp_path / "shared_card.json").write_text('{"card_id": "card-abc", "coo',
                                               encoding="utf-8")
    (tmp_path / "shared_card_history.jsonl").write_text(
        json.dumps({"card_id": "card-abc", "opened": 1}) + "\n", encoding="utf-8"
    )
    out = _sweep_card(tmp_path)
    combined = out.stdout + out.stderr
    assert "cannot be read" in combined, combined[-1500:]
    assert "Traceback" not in combined, (
        "an unhandled exception reached the log instead of a refusal:\n"
        + combined[-1500:]
    )
    assert "card-abc" in combined, (
        "the refusal does not name the last card from the history file, so the "
        "operator cannot recover it:\n" + combined[-1500:]
    )


def test_a_truncated_card_file_is_not_silently_reminted(
    tmp_path: pathlib.Path
) -> None:
    """The file must still be there afterwards, unchanged.

    Reminting is exactly the operator decision the `played` branch refuses to
    make -- games already scored onto the old card cannot be moved, so a fresh
    card silently builds a submission that omits them.
    """
    corrupt = '{"card_id": "card-abc", "coo'
    (tmp_path / "shared_card.json").write_text(corrupt, encoding="utf-8")
    _sweep_card(tmp_path)
    assert (tmp_path / "shared_card.json").read_text() == corrupt, (
        "the driver rewrote or retired the unreadable card file on its own"
    )
