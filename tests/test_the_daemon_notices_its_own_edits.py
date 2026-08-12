"""A running bash daemon keeps the code it was started with.

Editing `preserve_evidence.sh` changes nothing until the process restarts —
bash holds the function definitions in memory. That is easy to state and easy to
forget: on 2026-08-11 the card-vault call was fixed at 22:05 and the daemon kept
running the broken version for 13 consecutive cycles, logging
`card_vault: save failed` every five minutes, because the person who had written
"a fix is not live until the daemon restarts" at 21:50 did not restart it at
22:05. It cost nothing only because the session cookies happened not to rotate
in that window.

CLAUDE.md already makes the general argument, about timestamps: a convention
that has to be applied by hand at the right moment gets skipped. The fix is the
same shape — have the machine notice.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRESERVER = REPO / "tools" / "preserve_evidence.sh"
DEFAULT_BRANCH = "codexarc3"


def _reexec_guard() -> str:
    """The guard, lifted out of the real loop."""
    text = PRESERVER.read_text(encoding="utf-8")
    m = re.search(r'    if \[ "\$\{ONCE:-0\}" != 1 \]; then\n(?:.*?\n)*?    fi\n',
                  text)
    assert m, "the re-exec guard is not in the daemon any more"
    body = m.group(0)
    assert "exec bash" in body and "SELF_HASH" in body, f"extraction looks wrong:\n{body}"
    return body


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)


def _harness(tmp_path: Path, *, stale: bool, once: bool = False) -> str:
    """`exec` shadowed by a function, so we observe the decision without taking it."""
    self_copy = tmp_path / "preserve_evidence.sh"
    shutil.copy(PRESERVER, self_copy)
    real = subprocess.run(["sha256sum", str(self_copy)], capture_output=True, text=True)
    live = real.stdout.split()[0]
    return "\n".join([
        'log() { echo "$*"; }',
        'exec() { echo "WOULD-REEXEC $*"; }',      # a function beats the builtin
        f'SELF="{self_copy}"',
        f'SELF_HASH="{"0" * 64 if stale else live}"',
        f'ONCE={1 if once else 0}',
        _reexec_guard(),
        'echo DONE',
    ])


def test_an_unchanged_script_keeps_running(tmp_path):
    out = _run(_harness(tmp_path, stale=False))
    assert "WOULD-REEXEC" not in out.stdout, (
        "a daemon that re-execs every cycle never does any work:\n" + out.stdout)
    assert "DONE" in out.stdout, out.stdout + out.stderr


def test_an_edited_script_is_picked_up(tmp_path):
    """The case that cost 13 cycles."""
    out = _run(_harness(tmp_path, stale=True))
    assert "WOULD-REEXEC" in out.stdout, (
        "the daemon ignored an edit to its own file:\n" + out.stdout + out.stderr)
    assert "changed on disk" in out.stdout, out.stdout


def test_once_mode_never_reexecs(tmp_path):
    """`--once` must run exactly one cycle and exit — several tests extract this
    loop, and a re-exec there would turn a test into a daemon."""
    out = _run(_harness(tmp_path, stale=True, once=True))
    assert "WOULD-REEXEC" not in out.stdout, out.stdout
    assert "DONE" in out.stdout, out.stdout + out.stderr


def test_a_missing_hash_does_not_trigger_a_loop(tmp_path):
    """If `sha256sum` cannot read the file, the empty result must NOT read as
    'changed' — that would re-exec every cycle, forever, preserving nothing."""
    script = "\n".join([
        'log() { echo "$*"; }',
        'exec() { echo "WOULD-REEXEC $*"; }',
        f'SELF="{tmp_path / "gone.sh"}"',
        'SELF_HASH="somethingelse"',
        'ONCE=0',
        _reexec_guard(),
        'echo DONE',
    ])
    out = _run(script)
    assert "WOULD-REEXEC" not in out.stdout, (
        "an unreadable script file must not be treated as an edit:\n" + out.stdout)
    assert "DONE" in out.stdout, out.stdout + out.stderr


def test_the_tick_is_overridable():
    """Not a convenience. The re-exec guard can only be proved in a live process,
    and at the default 300s nobody will watch one — so the loop would be held
    honest by fixtures alone, which is how `shortest_path` ended up with a
    careful design and zero calls."""
    text = PRESERVER.read_text(encoding="utf-8")
    assert 'TICK="${CCARC3_PRESERVE_TICK:-300}"' in text, (
        "TICK is hardcoded again; the live path becomes untestable")


# --------------------------------------------------------------------------- #
# one preserver per repository
# --------------------------------------------------------------------------- #


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "evidence").mkdir()
    shutil.copy(PRESERVER, repo / "tools" / "preserve_evidence.sh")
    for cmd in (["git", "init", "-q", "."],
                ["git", "config", "user.email", "t@t"],
                ["git", "config", "user.name", "t"],
                ["git", "checkout", "-q", "-b", DEFAULT_BRANCH],
                ["git", "commit", "-q", "--allow-empty", "-m", "init"]):
        subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    return repo


def test_a_second_preserver_refuses_rather_than_racing_the_index(tmp_path):
    """Two preservers commit to the same git index and one of them loses.

    `daemon_watchdog.sh` relaunches this daemon whenever it sees none running —
    correct, and on 2026-08-11 it fired inside the two-second window of a manual
    restart and produced two, three seconds apart. Measured consequence, from the
    live log: `COMMIT FAILED — 1 files staged, nothing preserved this cycle`,
    plus a stale `.git/index.lock` that blocks every later commit until something
    removes it. The driver has held a lock for exactly this reason since the
    duplicate-driver incident; this daemon assumed uniqueness instead.
    """
    repo = _fake_repo(tmp_path)
    script = repo / "tools" / "preserve_evidence.sh"
    env = dict(os.environ, CCARC3_PRESERVE_TICK="60",
               CCARC3_SCRATCH=str(tmp_path / "absent"), ARC_API_KEY="fixture-key")

    # Output to a file, not a pipe. A pipe cannot be read without blocking while
    # the process is alive, so a failure here reported "it died" and nothing
    # about why — which is the same complaint this repo keeps making about its
    # own guards.
    log = tmp_path / "first.log"
    first = subprocess.Popen(["bash", str(script)], env=env,
                             stdout=log.open("w"), stderr=subprocess.STDOUT)
    try:
        # **Wait on a line this process wrote, not on a file anyone could have
        # written.** The first version waited until a `/tmp/ccarc3-preserve-*.lock`
        # appeared — and the live daemon on this box already holds one, so the
        # glob matched an unrelated file and the wait ended before the fixture
        # daemon had started. A readiness check satisfied by someone else's
        # artifact is the same defect this whole file is about.
        deadline = time.time() + 30
        while time.time() < deadline:
            if "preserving evidence every" in log.read_text(errors="ignore"):
                break
            if first.poll() is not None:
                break
            time.sleep(0.2)
        assert first.poll() is None, (
            "the first preserver died before the test could run:\n"
            + log.read_text(errors="ignore"))

        second = subprocess.run(["bash", str(script), "--once"], env=env,
                                capture_output=True, text=True, timeout=120)
        assert "already holds" in second.stdout, (
            "a second preserver did not refuse:\n" + second.stdout + second.stderr)
        assert second.returncode == 0, "refusing is not an error — the job is being done"
    finally:
        first.terminate()
        first.wait(timeout=30)


def test_the_lock_is_keyed_on_the_repository(tmp_path):
    """A test running `--once` against a temporary clone must not be blocked by
    the live daemon — `test_no_tool_is_pinned_to_this_container.py` does exactly
    that. Two different repos, two different locks."""
    text = PRESERVER.read_text(encoding="utf-8")
    assert 'LOCK="/tmp/ccarc3-preserve-$(printf' in text and '"$REPO"' in text, (
        "the lock is no longer derived from $REPO; a clone would now collide "
        "with the live daemon")


def test_the_lock_fails_open(tmp_path):
    """If the lock cannot be taken for any reason other than contention, run
    anyway. A duplicate costs a noisy cycle; refusing to start costs evidence on
    a disk that has been rolled back six times."""
    text = PRESERVER.read_text(encoding="utf-8")
    i = text.index('LOCK="/tmp/ccarc3-preserve-')
    block = text[i:i + 1800]
    assert "running anyway" in block, "the lock no longer fails open"
    assert "exit 0" in block, "contention must exit cleanly, not error"
