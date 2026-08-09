"""Backfilling a start time must read the run's *first* attempt, numerically.

`backfill_start_times` recovers "start time unknown" rows from the preserved
streams under `evidence/ccarc3/`. It sorted the candidates with

    key=lambda p: (0 if ".jsonl" == p.name[-11:-3] else 1, p.name)

which compares a six-character string against an eight-character slice. It is
`False` for every filename that can exist -- checked against the whole evidence
tree on 2026-08-09: 12 distinct stream names, zero matches -- so the first
element of the tuple was the constant `1` and the sort was plain lexicographic on
the filename. That is the ordering the comment above it says it is avoiding:
`stream.11` sorts before `stream.2` as a string, and `sp80` has streams up to
`stream.11`.

It escaped notice because the loop takes the first candidate that yields a
timestamp, and `stream.1` leads under both rules. The rules diverge exactly when
`stream.1` is missing or carries no timestamp -- a truncated first attempt, which
is the common reason a row needs backfilling at all -- and then the page reports
a *later* attempt's start as the run's start.

The second element mattered too: keyed on `p.name` alone, a game preserved under
two batches interleaved by `iterdir()` order, so which one won was filesystem
luck.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_trace_audit as bta  # noqa: E402


def _stream(path: pathlib.Path, stamp: str | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "system", "subtype": "init"}]
    if stamp:
        rows.append({"type": "assistant", "timestamp": stamp,
                     "message": {"id": "m1", "content": []}})
    with gzip.open(path, "wt") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_attempt_two_beats_attempt_eleven(tmp_path, monkeypatch):
    """With attempt 1 unstamped, the next attempt is 2 -- not 11."""
    gid = "sp80-589a99af"
    game = tmp_path / "evidence" / "ccarc3" / "rerun_losses" / gid
    _stream(game / "stream.1.jsonl.gz", None)            # truncated first attempt
    _stream(game / "stream.11.jsonl.gz", "2026-08-06T20:00:00Z")
    _stream(game / "stream.2.jsonl.gz", "2026-08-06T10:00:00Z")
    _stream(game / "stream.jsonl.gz", "2026-08-06T23:00:00Z")
    monkeypatch.setattr(bta, "REPO", tmp_path)

    runs = [{"id": gid}]
    filled = bta.backfill_start_times(runs)

    assert filled, "nothing was backfilled at all"
    assert runs[0]["started"] == "2026-08-06T10:00:00Z", (
        f"picked {runs[0]['started']}; attempt 2 started at 10:00 and attempt 11 "
        f"at 20:00, so a lexicographic sort takes the later one"
    )


def test_the_unnumbered_stream_is_the_last_attempt_not_the_first(tmp_path, monkeypatch):
    """`stream.jsonl` is the latest attempt, so it must never lead the order."""
    gid = "tn36-ef4dde99"
    game = tmp_path / "evidence" / "ccarc3" / "rerun_losses" / gid
    _stream(game / "stream.1.jsonl.gz", "2026-08-06T08:00:00Z")
    _stream(game / "stream.jsonl.gz", "2026-08-06T22:00:00Z")
    monkeypatch.setattr(bta, "REPO", tmp_path)

    runs = [{"id": gid}]
    bta.backfill_start_times(runs)

    assert runs[0]["started"] == "2026-08-06T08:00:00Z", (
        f"picked {runs[0]['started']}; the run started with attempt 1 at 08:00"
    )


def test_a_row_that_already_has_a_start_is_left_alone(tmp_path, monkeypatch):
    """Backfill fills gaps; it must not overwrite what ingest recorded."""
    gid = "sk48-d8078629"
    game = tmp_path / "evidence" / "ccarc3" / "rerun_losses" / gid
    _stream(game / "stream.1.jsonl.gz", "2026-08-06T08:00:00Z")
    monkeypatch.setattr(bta, "REPO", tmp_path)

    runs = [{"id": gid, "started": "2026-08-06T07:00:00Z", "started_local": "Aug 6 00:00"}]
    filled = bta.backfill_start_times(runs)

    assert filled == [], "backfill touched a row that already had a start time"
    assert runs[0]["started"] == "2026-08-06T07:00:00Z"


def test_the_batch_a_stream_came_from_breaks_ties(tmp_path, monkeypatch):
    """Two batches holding the same game must order deterministically.

    Keyed on the filename alone, `stream.1.jsonl.gz` from one batch and from
    another compare equal, and `sorted` is stable -- so the winner is whatever
    `iterdir()` happened to yield first. Directory order is hash order on this
    filesystem, which means the page's start time for such a game is decided by
    filesystem luck and can change between two builds over identical evidence.

    Asserting determinism means controlling the thing that is allowed to vary, so
    the directory order is forced both ways rather than hoped to be stable.
    """
    gid = "cd82-fb555c5d"
    root = tmp_path / "evidence" / "ccarc3"
    _stream(root / "aa_first_batch" / gid / "stream.1.jsonl.gz", "2026-08-06T09:00:00Z")
    _stream(root / "zz_later_batch" / gid / "stream.1.jsonl.gz", "2026-08-06T21:00:00Z")
    monkeypatch.setattr(bta, "REPO", tmp_path)

    real_iterdir = pathlib.Path.iterdir
    answers = {}
    for label, backwards in (("forward", False), ("reversed", True)):
        monkeypatch.setattr(
            pathlib.Path, "iterdir",
            lambda self, _b=backwards: iter(sorted(real_iterdir(self),
                                                   key=str, reverse=_b)),
        )
        runs = [{"id": gid}]
        bta.backfill_start_times(runs)
        answers[label] = runs[0].get("started")
    monkeypatch.setattr(pathlib.Path, "iterdir", real_iterdir)

    assert answers["forward"] == answers["reversed"], (
        f"the answer depends on directory order: {answers}"
    )
    assert answers["forward"] == "2026-08-06T09:00:00Z", (
        f"picked {answers['forward']}; the earliest preserved attempt is 09:00"
    )
