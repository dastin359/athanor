"""Nothing relaunched the watchdog, so its death was permanent.

The daemon cycle closed for DETECTION and not for repair. `daemon_watchdog.sh`
relaunches supervisor/heartbeat/preserver. `heartbeat.sh` notices when the
watchdog itself dies -- and only *prints* `DAEMON DOWN`, into a Monitor that
expires roughly every 30 minutes and does not exist at all after a container
replacement.

Measured 2026-08-12: the watchdog died with the 08:18 PDT replacement and was
still down at 10:25, two hours later, with the heartbeat running the entire time
and reporting it to nobody. The container-recovery block in the session keepalive
had also omitted it, so the one agent that could have noticed was not looking.

`supervisor.sh` now revives it at the top of each loop pass. A redundant
relaunch is safe: a second watchdog over the same daemon set exits via
`already_watching()`.

The function is EXTRACTED FROM THE SCRIPT rather than restated here. Two test
harnesses in this project lifted a shell function without its module-scope
variables, got `grep -E -e ""` -- which matches every line -- and asserted
happily against a guard that had been inverted. So the slice is taken by
markers and the test fails loudly if the extraction misses.

**And the decision is tested apart from the scan, because the first draft of
this file tested neither.** `pids_of` walks the real `/proc`, and this box runs
a production watchdog -- so "no watchdog is running" was never true inside the
fixture, and BOTH the positive and the negative test were decided by the live
box rather than by anything they set up. One of them failed for that reason and
the other PASSED for it, which is the more dangerous half.

So: `pids_of` is stubbed where the question is "what does revive_watchdog
decide", and exercised for real where the question is "does the matcher find a
process" -- the latter against a uniquely named script that cannot collide with
whatever this machine happens to be running.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import textwrap
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SUPERVISOR = REPO / "tools" / "supervisor.sh"


def _slice(path: pathlib.Path, start: str, end: str) -> str:
    src = path.read_text(encoding="utf-8")
    i = src.index(start)
    j = src.index(end, i)
    return src[i:j]


@pytest.fixture
def revive() -> str:
    """`revive_watchdog` alone, from the real file.

    **Without `pids_of`, deliberately.** The first version returned both, so the
    real scan was appended AFTER the harness's stub and silently overrode it --
    every decision test then read the live `/proc` again, which is the exact
    failure this file's docstring describes, reintroduced by the fix for it.
    The scan has its own test below, against a real process.
    """
    fn = _slice(SUPERVISOR, "revive_watchdog() {", "\nceiling_stopped=0")
    assert "daemon_watchdog" in fn, "extraction missed revive_watchdog"
    assert "pids_of" in fn, "extraction missed the liveness check"
    assert "pids_of() {" not in fn, (
        "the slice swallowed pids_of; the stub in _harness would be overridden "
        "and the tests would read the live process table")
    return fn


def _run(script: str, env: dict | None = None):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                          timeout=120, env=env)


def _harness(revive: str, repo: pathlib.Path, sp: pathlib.Path,
             running: bool) -> str:
    """`revive_watchdog` with a STUBBED `pids_of`, so the box cannot decide.

    `running` is what the stub claims about a live watchdog.
    """
    stub = ('pids_of() { echo 4242; }' if running else 'pids_of() { :; }')
    return "\n".join([f'REPO="{repo}"', f'SP="{sp}"', stub, revive,
                       "revive_watchdog", "wait 2>/dev/null || true"])


@pytest.fixture
def fake_repo(tmp_path):
    repo, sp = tmp_path / "repo", tmp_path / "sp"
    (repo / "tools").mkdir(parents=True)
    sp.mkdir()
    (repo / "tools" / "daemon_watchdog.sh").write_text(
        "#!/usr/bin/env bash\nsleep 30\n", encoding="utf-8")
    return repo, sp


# ── the decision ─────────────────────────────────────────────────────────────

def test_it_relaunches_a_watchdog_that_is_gone(revive, fake_repo):
    repo, sp = fake_repo
    out = _run(_harness(revive, repo, sp, running=False))
    assert out.returncode == 0, out.stderr
    assert "relaunching" in out.stdout, (
        "no watchdog is running and the supervisor did not start one:\n"
        + out.stdout + out.stderr)


def test_it_does_not_start_a_second_one(revive, fake_repo):
    """The failure this would cause is duplicates, which happened for real.

    Two supervisors were produced at 08:22 PDT by the watchdog racing a manual
    relaunch. A revive blind to an existing watchdog would do that every ten
    minutes, forever.
    """
    repo, sp = fake_repo
    out = _run(_harness(revive, repo, sp, running=True))
    assert out.returncode == 0, out.stderr
    assert "relaunching" not in out.stdout, (
        "a watchdog is already running and it started another:\n" + out.stdout)


def test_the_switch_turns_it_off(revive, fake_repo):
    repo, sp = fake_repo
    env = dict(os.environ, CCARC3_SUPERVISOR_WATCHES_WATCHDOG="0")
    out = _run(_harness(revive, repo, sp, running=False), env=env)
    assert out.returncode == 0, out.stderr
    assert "relaunching" not in out.stdout, out.stdout


def test_a_missing_watchdog_script_is_not_an_error(revive, fake_repo):
    """A checkout without the file must not make the supervisor loop noisy."""
    repo, sp = fake_repo
    (repo / "tools" / "daemon_watchdog.sh").unlink()
    out = _run(_harness(revive, repo, sp, running=False))
    assert out.returncode == 0, out.stderr
    assert "relaunching" not in out.stdout, out.stdout


def test_it_starts_a_real_process_not_just_a_message(revive, tmp_path):
    """`relaunching` must be a report of something, not the something.

    Uses a uniquely named script so the assertion is about THIS launch. The
    stub still supplies the liveness answer; what is real here is the spawn.
    """
    repo, sp = tmp_path / "repo", tmp_path / "sp"
    (repo / "tools").mkdir(parents=True)
    sp.mkdir()
    marker = sp / "it-ran"
    (repo / "tools" / "daemon_watchdog.sh").write_text(
        f'#!/usr/bin/env bash\ntouch "{marker}"\n', encoding="utf-8")
    out = _run(_harness(revive, repo, sp, running=False))
    assert out.returncode == 0, out.stderr
    for _ in range(50):
        if marker.exists():
            break
        time.sleep(0.1)
    assert marker.exists(), (
        "it printed 'relaunching' but nothing ran:\n" + out.stdout + out.stderr)


# ── the scan, exercised for real ─────────────────────────────────────────────

def test_the_matcher_finds_a_running_process_and_needs_a_path(tmp_path):
    """`pids_of` against a real process, under a name nothing else can hold.

    Also pins the shape of the pattern: `.*/name` requires a SLASH, so a daemon
    launched as `bash daemon_watchdog.sh` from inside `tools/` would read as
    down while alive -- and the supervisor would relaunch it every ten minutes.
    Documenting it rather than widening it: every launch path in this repo and
    in the recovery block passes `tools/<name>`, and a pattern without the
    anchor would match a substring of some unrelated command.
    """
    pids = _slice(SUPERVISOR, "pids_of() {", "\n}\n") + "\n}\n"
    script = tmp_path / "zzz_unique_daemon.sh"
    script.write_text("#!/usr/bin/env bash\nsleep 30\n", encoding="utf-8")

    proc = subprocess.Popen(["bash", str(script)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.4)
        with_path = _run(pids + '\npids_of ".*/zzz_unique_daemon\\.sh"')
        assert str(proc.pid) in with_path.stdout.split(), (
            f"pids_of did not find pid {proc.pid}: {with_path.stdout!r}")

        bare = _run(pids + '\npids_of "zzz_unique_daemon\\.sh"')
        assert str(proc.pid) not in bare.stdout.split(), (
            "the pattern matched without a path component; this test records "
            "the anchor's behaviour and it has changed")
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_the_loop_actually_calls_it(revive):
    """The function existing is not the fix; being called is.

    A helper defined and never invoked is this project's most repeated shape --
    the key shim that was finished, tested and never switched on, and the branch
    in `build_workspace` that never once ran.
    """
    src = SUPERVISOR.read_text(encoding="utf-8")
    loop = src[src.index("while true; do"):]
    assert "revive_watchdog" in loop, (
        "revive_watchdog is defined but the main loop never calls it")
