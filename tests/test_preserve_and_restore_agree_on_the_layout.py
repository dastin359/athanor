"""The two halves of the durability story, run against each other for once.

`preserve_evidence.sh` writes the evidence tree; `restore_clean_rollouts.py`
reads it back after a container replacement. Both are well tested and neither had
ever been run against the other: every restore test builds the evidence tree by
hand, so the layout one produces and the layout the other expects have only been
checked against one person's idea of both.

That is not a hypothetical failure mode here. `preserve_dir`'s own comment records
that it used to derive the destination from a single directory level, which worked
for the flat `<batch>/<game>` layout and, once `clean_rollouts` began nesting
`<batch>/<game>/attempt_N/<game>`, produced `attempt_2/<game>` — the batch name
dropped and every batch's `attempt_1` colliding in one directory. Hand-written
restore fixtures would have kept passing throughout.

Tonight's container replacement is the argument for testing the pair: the
scratchpad reverted, and what the box could recover afterwards was decided
entirely by whether these two agree.

Run through `--once`, which does a single preserve/commit/push cycle and exits,
rather than racing a timeout against a `while true`.
"""

from __future__ import annotations

import gzip
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PRESERVE = REPO / "tools" / "preserve_evidence.sh"
RESTORE = REPO / "tools" / "restore_clean_rollouts.py"

GAME = "zz99-deadbeef"
RESULT = {"game_id": GAME, "won": True, "levels_reached": 9, "levels_total": 9,
          "actions_used": 991, "duration_s": 4200.0, "cost_usd": 57.3,
          "attempts": 1, "playthroughs": 1, "trace_rows": 992}
# Shaped from a real preserved card (`bp35-0a0ad940`), because the restore
# corroborates the result against it and a fixture that cannot express agreement
# tests only the refusal. `actions_by_level` is one entry per play, each a list
# of `[level, cumulative_actions]`.
CARD = {"cards": {GAME: {
            "game_id": GAME,
            "actions_by_level": [[[1, 31], [2, 106], [3, 148], [4, 181],
                                  [5, 217], [6, 526], [7, 616], [8, 716],
                                  [9, 991]]],
            "actions": [991], "guids": ["83ae1163-32b1-4713-8894-c29a7d428161"],
            "levels_completed": [9], "resets": [22], "states": ["WIN"],
            "total_actions": 991, "total_plays": 1}},
        "played": 1, "won": 1, "total_actions": 991, "levels_completed": 9}


def _git(repo: pathlib.Path, *args: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, env=env,
                          capture_output=True, text=True, timeout=60)


@pytest.fixture
def box(tmp_path):
    """A scratchpad holding one finished rollout, and a repo to preserve into."""
    scratch = tmp_path / "scratchpad"
    ws = scratch / "clean_rollouts" / GAME / "attempt_1" / GAME
    ws.mkdir(parents=True)
    (ws / "result.json").write_text(json.dumps(RESULT), encoding="utf-8")
    (ws / "scorecard.json").write_text(json.dumps(CARD), encoding="utf-8")
    (ws / "meta.json").write_text(json.dumps({"game_id": GAME, "levels": 9}),
                                  encoding="utf-8")
    (ws / "trace.jsonl").write_text('{"level": 9}\n', encoding="utf-8")
    (ws / "stream.jsonl").write_text('{"type": "result"}\n', encoding="utf-8")
    # The banked marker the driver actually reads, at the game level.
    (scratch / "clean_rollouts" / GAME / "clean_result.json").write_text(
        json.dumps(RESULT), encoding="utf-8")

    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    # The runtime surface the preserver copies in from the package. Without these
    # the copy loop's `[ -f ... ] || continue` skips everything, and any test
    # asserting they land in the right place asserts an absence the fixture
    # guarantees — which is what the first version of this file did, and a mutant
    # removing the game-level guard survived it.
    athanor_pkg = repo / "src" / "athanor"
    pkg = athanor_pkg / "ccarc3"
    pkg.mkdir(parents=True)
    (athanor_pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "client.py").write_text("# the version this run talked to\n", encoding="utf-8")
    (pkg / "gate.py").write_text("# and its gate\n", encoding="utf-8")
    for module in ("grids.py", "ledger.py", "scoring.py"):
        (pkg / module).write_bytes((REPO / "src" / "athanor" / "ccarc3" / module).read_bytes())
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home),
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x",
           "GIT_CONFIG_NOSYSTEM": "1"}
    _git(repo, "init", "-q", "-b", "codexarc3", env=env)
    (repo / "README").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "-A", env=env)
    _git(repo, "commit", "-q", "-m", "base", env=env)
    # A real local remote, so the push succeeds immediately. Without one the
    # script's two retry ladders sleep 2+4+8+16 twice per cycle, and the test
    # spends a minute proving nothing about the layout it exists to check.
    bare = tmp_path / "origin.git"
    _git(repo, "init", "-q", "--bare", str(bare), env=env)
    _git(repo, "remote", "add", "origin", str(bare), env=env)
    _git(repo, "push", "-q", "-u", "origin",
         "codexarc3", env=env)
    # The real script, reached through the fixture repo so REPO derives to it.
    # Both real scripts, reached through the fixture repo so each derives its
    # own `REPO` to it. Copying rather than importing is deliberate: the paths
    # under test are the ones each script computes from its own location, which
    # is exactly what a fresh clone on another account will do.
    for tool in (PRESERVE, RESTORE):
        (repo / "tools" / tool.name).write_bytes(tool.read_bytes())
        (repo / "tools" / tool.name).chmod(0o755)
    return scratch, repo, env


def _preserve(scratch: pathlib.Path, repo: pathlib.Path, env: dict) -> str:
    proc = subprocess.run(
        ["bash", str(repo / "tools" / "preserve_evidence.sh"), "--once"],
        capture_output=True, text=True, timeout=180,
        env={**env, "CCARC3_SCRATCH": str(scratch),
             "CCARC3_BRANCH": "codexarc3",
             "ARC_API_KEY": "not-a-real-key-for-the-guard"},
    )
    # The push has no remote in this sandbox and is expected to fail loudly;
    # what this test is about is the tree on disk.
    assert "REFUSING" not in proc.stdout, proc.stdout
    return proc.stdout


def test_what_the_preserver_writes_is_what_the_restore_reads(box, tmp_path):
    """The round trip, with the scratchpad wiped in between — a replacement."""
    scratch, repo, env = box
    _preserve(scratch, repo, env)

    preserved = repo / "evidence" / "ccarc3" / "clean_rollouts" / GAME / "attempt_1" / GAME
    assert preserved.is_dir(), (
        f"the preserver did not mirror the nested layout; tree was:\n"
        + "\n".join(sorted(str(p.relative_to(repo)) for p in
                           (repo / "evidence").rglob("*") if p.is_file()))
    )
    assert (preserved / "result.json.gz").exists()

    # The replacement: everything on disk goes, the repo survives.
    import shutil
    shutil.rmtree(scratch)
    scratch.mkdir()

    proc = subprocess.run(
        [sys.executable, str(repo / "tools" / RESTORE.name),
         "--out", str(scratch / "clean_rollouts")],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "CCARC3_SCRATCH": str(scratch)},
        cwd=repo,
    )
    banked = scratch / "clean_rollouts" / GAME / "clean_result.json"
    assert banked.exists(), (
        f"the restore could not read the tree the preserver wrote.\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr[-1500:]}"
    )
    assert json.loads(banked.read_text())["levels_reached"] == 9


def test_the_batch_name_survives_the_round_trip(box):
    """The regression the one-level derivation caused: batch dropped, attempts collide."""
    scratch, repo, env = box
    # A second batch holding the same game and the same attempt number.
    other = scratch / "clean_rollouts_validate" / GAME / "attempt_1" / GAME
    other.mkdir(parents=True)
    (other / "result.json").write_text(
        json.dumps({**RESULT, "levels_reached": 3}), encoding="utf-8")
    (other / "trace.jsonl").write_text('{"level": 3}\n', encoding="utf-8")

    _preserve(scratch, repo, env)

    root = repo / "evidence" / "ccarc3"
    first = root / "clean_rollouts" / GAME / "attempt_1" / GAME / "result.json.gz"
    second = root / "clean_rollouts_validate" / GAME / "attempt_1" / GAME / "result.json.gz"
    assert first.exists() and second.exists(), (
        "two batches' attempt_1 did not both survive; the batch name was dropped"
    )
    with gzip.open(first, "rt") as fh:
        assert json.load(fh)["levels_reached"] == 9
    with gzip.open(second, "rt") as fh:
        assert json.load(fh)["levels_reached"] == 3, "one batch overwrote the other"


def test_a_game_level_directory_gets_no_runtime_copy(box):
    """`client.py`/`gate.py` are copied from the repo, so only beside a real run.

    Without the guard this produced 25 directories whose entire contents were a
    copy of HEAD, recording nothing about any run.
    """
    scratch, repo, env = box
    _preserve(scratch, repo, env)

    root = repo / "evidence" / "ccarc3" / "clean_rollouts" / GAME
    run_level = root / "attempt_1" / GAME

    # The positive control first: without it this test passes on a preserver that
    # copies nothing anywhere.
    assert (run_level / "client.py.gz").exists(), (
        "the runtime surface was not preserved beside the run at all, so the "
        "assertion below discriminates nothing"
    )
    assert (run_level / "gate.py.gz").exists()

    assert not (root / "client.py.gz").exists(), (
        "the game-level directory got a copy of HEAD's client.py — this produced "
        "25 directories whose entire contents recorded nothing about any run"
    )
    assert not (root / "gate.py.gz").exists()


def test_the_stream_is_preserved_beside_the_run(box):
    """The one class of solver log a replacement can still lose."""
    scratch, repo, env = box
    _preserve(scratch, repo, env)
    preserved = repo / "evidence" / "ccarc3" / "clean_rollouts" / GAME / "attempt_1" / GAME
    assert (preserved / "stream.jsonl.gz").exists(), "the solver stream was not preserved"


def test_a_preserved_runtime_copy_is_never_restamped(box):
    """Write once. A record that tracks HEAD records nothing.

    The preserve loop runs every five minutes over every run it has ever
    preserved, and `client.py`/`gate.py` are copied from the *repo* rather than
    from the run — so without the write-once guard each pass restamps a finished
    run with whatever the package looks like now. It did: on 2026-08-07 a
    `client.py` edit made 22 historical runs' preserved copies byte-identical to
    source written hours after they ended.

    That cost something specific. `bp35` read a `status()` line printing
    `[cap 1.000 = 9/9 levels]`, mistook a completion ceiling for a perfect score,
    and lost 0.2748; `re86` ran after that line was changed. Their preserved
    copies were identical, so the one artifact that could have shown which
    version each run was talking to showed nothing.
    """
    scratch, repo, env = box
    _preserve(scratch, repo, env)

    preserved = (repo / "evidence" / "ccarc3" / "clean_rollouts" / GAME
                 / "attempt_1" / GAME / "client.py.gz")
    with gzip.open(preserved, "rt") as fh:
        first = fh.read()
    assert "the version this run talked to" in first

    # The package moves on, as it does between every pair of runs.
    (repo / "src" / "athanor" / "ccarc3" / "client.py").write_text(
        "# rewritten hours after the run ended\n", encoding="utf-8")
    # Something new to preserve, so the cycle has work and reaches the copy loop.
    (scratch / "clean_rollouts" / GAME / "attempt_1" / GAME / "trace.jsonl").write_text(
        '{"level": 9}\n{"level": 9}\n', encoding="utf-8")
    _preserve(scratch, repo, env)

    with gzip.open(preserved, "rt") as fh:
        again = fh.read()
    assert again == first, (
        f"the preserved copy was restamped with source written after the run: "
        f"{again!r}"
    )


# --- gz_atomic, extracted and run for real ---------------------------------- #
#
# Two properties nothing held. Both survived the round-trip tests above, which is
# fair: a single preserve cycle over a healthy tree cannot show either.

def _slice(text: str, marker: str) -> str:
    start = text.index(marker)
    return text[start:text.index("\n}\n", start) + 3]


def _gz_atomic_fn() -> str:
    """The real function, lifted out of the real file — with what it depends on.

    **A slice that takes only `gz_atomic` runs it with its guards unbound**, and
    the failure is not a clean one: `grep -E -e ""` matches every line, so every
    file is treated as carrying a cookie, `redact_cookies` is undefined, the
    error branch fires, and the function returns having written nothing. Both
    round-trip tests below then fail with `FileNotFoundError` on the output --
    which is at least loud. The quiet version of the same mistake is what this
    file's own header is about: an extraction that succeeds at pulling the wrong
    thing tests something that does not ship.
    """
    text = PRESERVE.read_text(encoding="utf-8")
    patterns = [ln for ln in text.splitlines() if re.match(r"^_COOKIE_[A-Z]+=", ln)]
    assert len(patterns) >= 3, f"expected the cookie patterns, found {len(patterns)}"
    return "\n".join([
        'log() { echo "$*"; }',
        *patterns,
        _slice(text, "redact_cookies() {"),
        _slice(text, "strip_cookies() {"),
        _slice(text, "gz_atomic() {"),
    ]) + "\n"


def _run_gz(body: str, tmp_path: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", _gz_atomic_fn() + "\n" + body],
                          cwd=tmp_path, capture_output=True, text=True, timeout=60)


def test_a_failed_compression_leaves_the_previous_copy_intact(tmp_path):
    """Written to a temp file and renamed, so a failure cannot truncate evidence.

    The destination here is the only copy of a finished run that survives a
    container replacement. Compressing straight into it means any failure --
    a disappearing source, a full disk -- replaces good evidence with a partial
    file, and the log line for that is a warning nobody reads against a file
    nobody re-checks.
    """
    good = tmp_path / "kept.gz"
    subprocess.run(["bash", "-c", f'printf "old but good\\n" | gzip -9 -n -c > "{good}"'],
                   check=True, timeout=60)
    before = good.read_bytes()

    proc = _run_gz(f'gz_atomic "{tmp_path}/does-not-exist" "{good}"', tmp_path)

    assert good.read_bytes() == before, (
        "a failed compression clobbered the destination; the only durable copy "
        "of a finished run is now partial"
    )
    assert "WARNING" in proc.stdout, "the failure was silent"
    assert not list(tmp_path.glob("*.tmp.*")), "the temp file was left behind"


def test_unchanged_content_compresses_to_the_same_bytes(tmp_path):
    """`-n`, so git sees no diff when a run rewrites a file it did not change.

    Measured: with `-n` the output is byte-identical for identical content, and
    without it a rewrite with a new mtime produces different bytes. Solvers
    rewrite `trace.state.json` and `rules.json` on every action, so without this
    the preserver commits and pushes files whose content never changed — every
    five minutes, forever.
    """
    src = tmp_path / "rules.json"
    src.write_text('{"mechanics": []}', encoding="utf-8")
    _run_gz(f'gz_atomic "{src}" "{tmp_path}/a.gz"', tmp_path)
    first = (tmp_path / "a.gz").read_bytes()

    import time
    time.sleep(1.1)                       # gzip's stored mtime has 1s granularity
    src.write_text('{"mechanics": []}', encoding="utf-8")   # same content, new mtime
    _run_gz(f'gz_atomic "{src}" "{tmp_path}/b.gz"', tmp_path)

    assert (tmp_path / "b.gz").read_bytes() == first, (
        "a rewrite with unchanged content produced different bytes; every such "
        "rewrite becomes a commit and a push"
    )


def test_changed_content_does_compress_differently(tmp_path):
    """The positive control: `-n` must not make the preserver blind to real change."""
    src = tmp_path / "rules.json"
    src.write_text('{"mechanics": []}', encoding="utf-8")
    _run_gz(f'gz_atomic "{src}" "{tmp_path}/a.gz"', tmp_path)
    first = (tmp_path / "a.gz").read_bytes()

    src.write_text('{"mechanics": ["the avatar climbs"]}', encoding="utf-8")
    _run_gz(f'gz_atomic "{src}" "{tmp_path}/b.gz"', tmp_path)

    assert (tmp_path / "b.gz").read_bytes() != first
