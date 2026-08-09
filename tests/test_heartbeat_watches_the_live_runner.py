"""Regression tests for tools/heartbeat.sh.

Every defect covered here had the same shape: a guard that *reported by not
running*.  The heartbeat watched a retired process name, so `runner_alive` was
permanently false; it treated one false poll as "the arm ended" and broke out on
its first iteration; `work_pending` rebuilt a path that did not exist, so work
looked pending forever; the failure grep was unanchored, so `tail -1` printed a
healthy proxy-port line every poll and buried the real tracebacks.  All of them
looked identical to a healthy heartbeat from the outside.

So the rule for this file: **never assert only on an exit code, and never assert
only on silence.**  Every test either asserts the specific line the check emits,
or -- where the correct behaviour *is* silence -- proves the loop completed the
polls it was supposed to and was still running when the test stopped it.

Two things are deliberate:

* The script's paths are absolute (`SP=...`, `/home/user/athanor/tools`), so the
  loop tests run a *patched copy* in a tmp sandbox.  Every substitution asserts
  its own hit count, so the test breaks loudly if the real file drifts rather
  than quietly testing something that is no longer there.
* The source is read once, at import, through an absolute path.  Nothing here
  cd's into a sandbox before reading the script under test.

The environment handed to every subprocess is built from scratch (`_ENV`): no
ARC_API_KEY, no network, nothing inherited from the launching shell that could
decide a result.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

#: tools/ lives beside the test suite in the checkout; the shared checkout is the
#: fallback for worktrees that do not carry the (untracked) shell tools.
_ROOTS = [Path(__file__).resolve().parents[1], Path("/home/user/athanor")]


def _tool(name: str) -> Path:
    for root in _ROOTS:
        candidate = root / "tools" / name
        if candidate.exists():
            return candidate
    raise AssertionError(f"cannot find tools/{name} under any of {_ROOTS}")


HEARTBEAT = _tool("heartbeat.sh")
SUPERVISOR = _tool("supervisor.sh")
TOOLS = HEARTBEAT.parent
REPO_ROOT = TOOLS.parent

#: Read once, absolutely, before any test touches a sandbox directory.
SRC = HEARTBEAT.read_text(encoding="utf-8")

#: Explicit environment.  ARC_API_KEY is absent on purpose: the heartbeat must
#: not need it, and a test that reads it from the launching shell passes or
#: fails for reasons unrelated to the code.
_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": os.environ.get("HOME", "/root"),
    "LC_ALL": "C",
    "CCARC3_SWEEP_DIR": "clean_rollouts",
}


# --------------------------------------------------------------------------
# extraction helpers -- pull the real text out of the real file
# --------------------------------------------------------------------------
def _extract(pattern: str, what: str) -> str:
    m = re.search(pattern, SRC, re.S | re.M)
    assert m, f"could not find {what} in {HEARTBEAT}; this test needs updating"
    return m.group(0)


def _fail_pattern() -> str:
    """The real grep pattern, taken from the file, so the test tracks it."""
    m = re.search(r"^\s*fail=\$\(grep -E '([^']+)'", SRC, re.M)
    assert m, "could not extract the fail= grep pattern from heartbeat.sh"
    return m.group(1)


# Extracted lazily, per test: a fix that is reverted should fail the test that
# names it, not error out collection for the whole file.
def _runner_name_block() -> str:
    return _extract(
        r"^RUNNER_NAME=\$\(grep.*?^RUNNER_NAME=\"\$\{RUNNER_NAME:-[^\n]*$",
        "the RUNNER_NAME derivation",
    )


def _daemon_check_fn() -> str:
    return _extract(r"^daemon_check\(\) \{.*?^\}$", "daemon_check()")


def _work_pending_fn() -> str:
    return _extract(r"^work_pending\(\) \{.*?^\}$", "work_pending()")

DAEMON_LIST = "for name in supervisor.sh preserve_evidence.sh context_watch.py"


def _sub(text: str, needle: str, repl: str, expect: int) -> str:
    """Replace, and *prove* the replacement landed."""
    hits = text.count(needle)
    assert hits == expect, (
        f"expected {expect} occurrence(s) of {needle!r} in heartbeat.sh, found "
        f"{hits}: the script changed and this test is patching a stale shape"
    )
    return text.replace(needle, repl)


# --------------------------------------------------------------------------
# process helpers
# --------------------------------------------------------------------------
def _argv_elements(pid: str) -> list[str]:
    try:
        raw = Path("/proc", pid, "cmdline").read_bytes()
    except OSError:
        return []
    return [a.decode("utf-8", "replace") for a in raw.split(b"\0") if a]


def _running_by_argv(name: str) -> bool:
    """Ground truth for the rule the script uses: a whole argv element that is a
    path ending in `name`.  Deliberately not a substring search -- substring
    matching is the bug the script's own comments are about, and it would find
    this test process."""
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        if any(a.endswith("/" + name) for a in _argv_elements(d.name)):
            return True
    return False


@pytest.fixture
def procs():
    """Spawn throwaway processes whose argv carries an absolute path ending in a
    chosen filename, and guarantee they die with the test."""
    started: list[subprocess.Popen] = []

    def spawn(path: Path) -> subprocess.Popen:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nsleep 300\n", encoding="utf-8")
        path.chmod(0o755)
        p = subprocess.Popen(
            [str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_ENV,
            start_new_session=True,
        )
        started.append(p)
        deadline = time.time() + 10
        while time.time() < deadline:
            if _running_by_argv(path.name):
                return p
            time.sleep(0.02)
        raise AssertionError(f"fake process {path} never appeared in /proc")

    yield spawn

    for p in started:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


# --------------------------------------------------------------------------
# sandbox: a patched copy of the real script over fixture roots
# --------------------------------------------------------------------------
class Sandbox:
    def __init__(self, root: Path, script: Path, sp: Path, tools: Path, out: Path):
        self.root = root
        self.script = script
        self.sp = sp
        self.tools = tools
        self.out = out

    @property
    def polls(self) -> int:
        f = self.sp / "polls.txt"
        return len(f.read_text().splitlines()) if f.exists() else 0


def _sandbox(
    tmp_path: Path,
    *,
    runner_name: str = "zz_fake_runner.py",
    games: int = 3,
    banked: int = 1,
    stop_after_polls: int = 3,
) -> Sandbox:
    """Build a runnable copy of heartbeat.sh over fixture roots.

    `stop_after_polls` freezes the loop deterministically: the stub the script
    runs at the top of every iteration records the poll and then blocks forever
    once the target is reached.  So "the counter reads N" means exactly
    "iterations 1..N-1 completed and iteration N did nothing" -- no
    sleep-and-hope, and no test that silently observes fewer polls than it
    claims.
    """
    root = tmp_path
    tools = root / "tools"
    sp = root / "sp"
    # OUT lives where a "$REPO/$CCARC3_SWEEP_DIR" reconstruction would never
    # land, and both directories such a reconstruction would build are created
    # empty: a rebuilt path reads 0 banked and calls work pending forever.
    out = root / "elsewhere" / "sweep_out"
    for decoy in (root / "clean_rollouts", sp / "clean_rollouts"):
        decoy.mkdir(parents=True, exist_ok=True)
    tools.mkdir(parents=True, exist_ok=True)
    sp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    game_ids = [f"g{i:02d}-fake" for i in range(games)]
    for gid in game_ids[:banked]:
        (out / gid).mkdir(parents=True, exist_ok=True)
        (out / gid / "clean_result.json").write_text("{}", encoding="utf-8")
    # One trace, so the progress line's act count and age are real values.
    trace = out / game_ids[0] / "attempt_1" / "ws" / "trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")

    # The supervisor the heartbeat must take its runner name from.
    (tools / "supervisor.sh").write_text(
        '#!/bin/bash\nREPO=/nowhere\nRUNNER="${1:-$REPO/tools/%s}"\n' % runner_name,
        encoding="utf-8",
    )
    # The driver work_pending is supposed to ask for OUT.
    (tools / "clean_rollouts.py").write_text(
        "import pathlib\n"
        f"OUT = pathlib.Path({str(out)!r})\n"
        f"GAMES = {game_ids!r}\n",
        encoding="utf-8",
    )

    (sp / "snapshot_results.py").write_text(
        "import time, pathlib\n"
        f"p = pathlib.Path({str(sp / 'polls.txt')!r})\n"
        "with p.open('a') as fh:\n"
        "    fh.write('poll\\n')\n"
        f"if len(p.read_text().splitlines()) >= {stop_after_polls}:\n"
        "    time.sleep(3600)\n",
        encoding="utf-8",
    )
    (sp / "quota.sh").write_text(
        "#!/bin/bash\necho 'five_hour   12%'\necho 'seven_day   30%'\n",
        encoding="utf-8",
    )

    text = SRC
    # The tools directory is NOT patched in either. The script derives
    # `REPO` from `${BASH_SOURCE[0]}`, so writing it into the sandbox's own
    # `tools/` makes that derivation resolve to the sandbox for real --
    # testing the mechanism instead of a rewritten copy of its output.
    # The scratchpad is NOT patched into the script any more -- it is handed in
    # through `CCARC3_SCRATCH` by `_run`, so this exercises the real override
    # rather than a rewritten copy of the line. Proven below, because a broken
    # override would silently point the sandbox at the LIVE scratchpad, and a
    # test that runs against the real tree is the exact defect this file warns
    # about in its own docstring.
    text = _sub(text, "sleep 420", "sleep 0.2", expect=3)
    script = tools / "heartbeat_under_test.sh"
    script.write_text(text, encoding="utf-8")

    # `PY_BIN` is "$REPO/.venv/bin/python", so the sandbox repo needs one -- and
    # it must be DISTINGUISHABLE from the live interpreter, not a symlink to it.
    # A symlink made a hardcoded `PY_BIN=/home/user/athanor/.venv/bin/python`
    # behave identically to the derived one, so that mutant survived the whole
    # suite. It is not a cosmetic mutant: on a fresh container that path does not
    # exist, and every python call in the loop is swallowed by `2>/dev/null` or
    # `|| true`, leaving the heartbeat printing tidy status lines having computed
    # nothing. The wrapper records each use so a test can prove whose python ran.
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    shim = venv_bin / "python"
    shim.write_text(
        "#!/bin/bash\n"
        f'echo used >> {str(root / "pybin_used.txt")!r}\n'
        f'exec {str(REPO_ROOT / ".venv" / "bin" / "python")!r} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)

    # Prove the sandbox owns both roots before anything runs in it.
    #
    # The probe is a FILE in the same directory as the script under test, not a
    # `bash -c` string: `REPO` derives from `${BASH_SOURCE[0]}`, which is empty
    # under `-c`, so evaluating the line that way resolves it against the
    # pytest process's cwd and "proves" something about /home/user. Reading a
    # proxy for the thing is the defect this whole file is about, and the first
    # draft of this guard committed it.
    probe = tools / "_probe_roots.sh"
    probe.write_text(
        "\n".join(
            [line for line in text.splitlines()
             if line.startswith("REPO=") or line.startswith("SP=")
             or line.startswith("PY_BIN=")]
        )
        + '\nprintf "%s\\n%s" "$REPO" "$SP"\n',
        encoding="utf-8",
    )
    seen = subprocess.run(
        ["/bin/bash", str(probe)],
        capture_output=True, text=True, check=True,
        env={**_ENV, "CCARC3_SCRATCH": str(sp)},
    ).stdout.splitlines()
    probe.unlink()
    assert seen == [str(root), str(sp)], (
        f"sandboxed heartbeat resolved (REPO, SP) to {seen!r}, not "
        f"{[str(root), str(sp)]!r} -- it would have read the LIVE supervisor, "
        f"driver and scratchpad while claiming to describe a fixture"
    )

    return Sandbox(root, script, sp, tools, out)


def _run(box: Sandbox, *, until_polls: int, timeout: float = 120.0):
    """Run the sandboxed heartbeat until it has completed `until_polls - 1`
    iterations and is frozen at the top of iteration `until_polls`.

    Returns (stdout, still_running, polls)."""
    proc = subprocess.Popen(
        ["/bin/bash", str(box.script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**_ENV, "CCARC3_SCRATCH": str(box.sp)},
        cwd=str(box.root),
        start_new_session=True,
    )
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            if box.polls >= until_polls:
                time.sleep(0.3)  # let the frozen iteration settle
                break
            time.sleep(0.05)
        still_running = proc.poll() is None
        if still_running:
            os.killpg(proc.pid, signal.SIGKILL)
        out = proc.communicate(timeout=30)[0]
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return out, still_running, box.polls


def _context(repo) -> str:
    """The two definitions the full script provides to any block extracted from it.

    An extracted function is a fragment: `REPO` and `PY_BIN` are set near the top
    of heartbeat.sh, outside every function body. Supplying them here is how the
    fragment gets its real environment -- and passing `repo` is how a test aims
    the SAME derivation at a fixture, instead of rewriting a path into the text
    and testing the rewrite.
    """
    return f'REPO={repo}\nPY_BIN={REPO_ROOT / ".venv" / "bin" / "python"}\n'


def _bash(script_text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", "-c", script_text],
        capture_output=True,
        text=True,
        env=_ENV,
        timeout=90,
    )


# ==========================================================================
# (a) RUNNER_NAME is derived from supervisor.sh, not hardcoded
# ==========================================================================
def test_runner_name_is_read_out_of_supervisor_sh(tmp_path):
    """The name comes from supervisor.sh and *moves* when supervisor.sh moves.
    Asserting only "it equals clean_rollouts.py" would pass on a hardcoded
    version too, so the fixture half below is the load-bearing half."""
    m = re.search(
        r'^RUNNER="\$\{1:-\$REPO/tools/([a-z_]+\.py)\}"',
        SUPERVISOR.read_text(encoding="utf-8"),
        re.M,
    )
    assert m, f"{SUPERVISOR} no longer declares RUNNER in the expected form"
    real_runner = m.group(1)

    block = _runner_name_block()
    live = _bash(_context(REPO_ROOT) + block + '\nprintf "%s" "$RUNNER_NAME"\n')
    assert live.stdout == real_runner, (
        f"heartbeat derived {live.stdout!r} but supervisor.sh launches "
        f"{real_runner!r}"
    )
    assert live.stdout != "ablate_baselines.py", "still watching the retired arm"

    # Point the same derivation at a different supervisor: the name must follow.
    fake_tools = tmp_path / "tools"
    fake_tools.mkdir()
    (fake_tools / "supervisor.sh").write_text(
        '#!/bin/bash\nRUNNER="${1:-$REPO/tools/zz_other_runner.py}"\n',
        encoding="utf-8",
    )
    # Same block, same derivation -- only REPO moves.
    moved = _bash(_context(tmp_path) + block + '\nprintf "%s" "$RUNNER_NAME"\n')
    assert moved.stdout == "zz_other_runner.py", (
        "RUNNER_NAME did not follow supervisor.sh -- it is hardcoded, or the "
        f"fallback is answering (got {moved.stdout!r})"
    )
    assert moved.stdout != real_runner, "fixture is degenerate; pick another name"


def test_live_runner_is_seen_and_its_own_log_is_the_one_read(tmp_path, procs):
    """End to end: with the supervisor's runner actually running, the heartbeat
    emits a progress line built from *that* runner's log -- not the retired
    arm's log sitting beside it, and not the idle path."""
    box = _sandbox(tmp_path, banked=1, games=3, stop_after_polls=3)
    procs(box.tools / "zz_fake_runner.py")

    (box.sp / "zz_fake_runner.log").write_text(
        "=== live-runner game g00-fake ===\nsome ordinary chatter\n",
        encoding="utf-8",
    )
    # The retired arm's log, still on disk. Nothing may come from it.
    (box.sp / "ablate.log").write_text(
        "=== RETIRED-ARM wa30: already finished, skipping ===\n", encoding="utf-8"
    )

    out, still_running, polls = _run(box, until_polls=3)

    assert polls == 3, f"loop did not reach three polls (got {polls}): {out!r}"
    assert still_running, "heartbeat exited while its runner was alive"
    assert "NO RUNNER" not in out, f"the live runner was not seen: {out!r}"
    assert "=== live-runner game g00-fake ===" in out, (
        f"progress line did not come from the live runner's log: {out!r}"
    )
    assert "RETIRED-ARM" not in out, f"read the retired arm's log: {out!r}"
    assert "1/3 done" in out, f"progress counts missing: {out!r}"
    assert "2 acts" in out, f"act count missing: {out!r}"
    assert "five_hour" in out, f"quota block missing: {out!r}"


# ==========================================================================
# (b) "no runner" is not "the arm ended"
# ==========================================================================
def test_a_missing_runner_is_not_reported_before_the_idle_limit(tmp_path):
    """Two idle polls with work pending must be silent *and* must not end the
    heartbeat.  The old loop broke on the first one."""
    box = _sandbox(tmp_path, banked=1, games=3, stop_after_polls=3)
    out, still_running, polls = _run(box, until_polls=3)

    # Proof the idle branch really ran twice, rather than the script exiting
    # early and the silence assertions passing on an empty string.
    assert polls == 3, f"loop did not complete two polls (got {polls}): {out!r}"
    assert still_running, "heartbeat exited on a missing runner -- that is the bug"
    assert "NO RUNNER" not in out, f"reported after only two idle polls: {out!r}"
    assert "SWEEP COMPLETE" not in out, f"work was pending: {out!r}"


def test_sustained_absence_with_work_pending_is_reported(tmp_path):
    """IDLE_LIMIT consecutive idle polls with work pending -> exactly one
    report, naming the window."""
    box = _sandbox(tmp_path, banked=1, games=3, stop_after_polls=4)
    out, still_running, polls = _run(box, until_polls=4)

    assert polls == 4, f"loop did not complete three polls (got {polls}): {out!r}"
    assert still_running, "heartbeat exited instead of continuing to watch"
    assert out.count("NO RUNNER") == 1, (
        f"expected exactly one NO RUNNER report after three idle polls: {out!r}"
    )
    assert "for 21 min with work pending" in out, (
        f"the report did not name the sustained window: {out!r}"
    )


def test_sweep_complete_is_announced_once_and_then_held(tmp_path):
    """All games banked: announce once, keep polling, do not exit.  Reaching
    this branch at all is also the (c) regression -- work_pending can only be
    false if it asked clean_rollouts for OUT."""
    box = _sandbox(tmp_path, banked=3, games=3, stop_after_polls=5)
    out, still_running, polls = _run(box, until_polls=5)

    assert polls == 5, f"loop did not complete four polls (got {polls}): {out!r}"
    assert still_running, "heartbeat exited on a complete sweep instead of holding"
    assert out.count("SWEEP COMPLETE") == 1, (
        f"expected exactly one SWEEP COMPLETE across four polls: {out!r}"
    )
    assert "NO RUNNER" not in out, (
        f"a complete sweep is not a missing-runner report: {out!r}"
    )


# ==========================================================================
# (c) work_pending asks clean_rollouts for OUT
# ==========================================================================
def test_work_pending_asks_the_driver_for_out(tmp_path):
    """Both cases share an empty `$SP/clean_rollouts` decoy -- the directory a
    rebuilt path points at.  A rebuild reads 0 banked in both and calls both
    "work pending"; only asking `cr.OUT` tells them apart."""
    results = {}
    for label, banked in (("pending", 1), ("complete", 3)):
        box = _sandbox(tmp_path / label, banked=banked, games=3)
        assert not list((box.sp / "clean_rollouts").iterdir()), "decoy must be empty"
        fn = _work_pending_fn()
        cp = _bash(_context(box.root) + fn + '\nwork_pending; echo "rc=$?"\n')
        results[label] = cp.stdout.strip()

    assert results["pending"] == "rc=0", (
        f"1 of 3 banked should be work pending, got {results['pending']!r}"
    )
    assert results["complete"] == "rc=1", (
        "3 of 3 banked should be no work pending -- work_pending is reading a "
        f"rebuilt path instead of cr.OUT (got {results['complete']!r})"
    )


# ==========================================================================
# (d) the fail= grep is anchored -- tested in both directions
# ==========================================================================
MUST_MATCH = [
    "HTTP Error 429: Too Many Requests",
    "429 Too Many Requests",
    "status=429",
    "Traceback (most recent call last):",
    "PROOFREAD DID NOT RUN",
    "aborting: scorecard closed",
    "openai.RateLimitError: rate_limit exceeded",
]

MUST_NOT_MATCH = [
    "arc_proxy: http://127.0.0.1:44297 (key withheld from the child)",
    "scorecard 3fa85f64-5717-4562-b3fc-429c66afa429 opened",
    "CLEAN in 20 min - 9/9, 429 actions",
]


def _greps(pattern: str, line: str) -> bool:
    cp = subprocess.run(
        ["grep", "-E", pattern],
        input=line + "\n",
        capture_output=True,
        text=True,
        env=_ENV,
        timeout=30,
    )
    assert cp.returncode in (0, 1), cp.stderr
    return cp.returncode == 0


@pytest.mark.parametrize("line", MUST_MATCH)
def test_failure_pattern_catches_real_failures(line):
    assert _greps(_fail_pattern(), line), f"real failure went unreported: {line!r}"


@pytest.mark.parametrize("line", MUST_NOT_MATCH)
def test_failure_pattern_ignores_lookalikes(line):
    pattern = _fail_pattern()
    # The lookalikes must be genuine near-misses: an unanchored `429` fires on
    # every one of them.  Without this, the negative test could pass on lines
    # that were never a hazard in the first place.
    assert _greps("429", line), f"fixture line is not a 429 lookalike: {line!r}"
    assert not _greps(pattern, line), f"false alarm on a healthy line: {line!r}"


def test_the_newest_lookalike_does_not_bury_the_real_failure(tmp_path, procs):
    """`tail -1` takes the newest match.  With a healthy proxy line written
    *after* a real 429, an unanchored pattern reports the proxy line and hides
    the failure -- so this drives the whole script, not just the regex."""
    box = _sandbox(tmp_path, banked=1, games=3, stop_after_polls=3)
    procs(box.tools / "zz_fake_runner.py")
    (box.sp / "zz_fake_runner.log").write_text(
        "=== live-runner game g00-fake ===\n"
        "urllib.error.HTTPError: HTTP Error 429: Too Many Requests\n"
        "arc_proxy: http://127.0.0.1:44297 (key withheld from the child)\n",
        encoding="utf-8",
    )

    out, still_running, polls = _run(box, until_polls=3)

    assert polls == 3 and still_running, f"the loop did not run: {out!r}"
    assert "LAST FAILURE: " in out, f"the real 429 was not reported at all: {out!r}"
    assert "HTTP Error 429: Too Many Requests" in out, (
        f"reported something other than the 429: {out!r}"
    )
    assert "44297" not in out, f"the healthy proxy line buried the failure: {out!r}"


# ==========================================================================
# (e) daemon_check -- positive and negative control
# ==========================================================================
def test_daemon_check_names_the_absent_and_stays_quiet_for_the_present(
    tmp_path, procs
):
    """Controlled both-directions test: two daemon names, one running, one not,
    so the result cannot depend on what happens to be running on the box."""
    daemon_fn = _daemon_check_fn()
    hits = daemon_fn.count(DAEMON_LIST)
    assert hits == 2, f"daemon_check's name list changed shape ({hits} lists found)"
    fn = daemon_fn.replace(
        DAEMON_LIST, "for name in zz_present_daemon.sh zz_absent_daemon.sh"
    )

    procs(tmp_path / "zz_present_daemon.sh")
    assert not _running_by_argv("zz_absent_daemon.sh"), "the absent fixture is running"

    out = _bash(fn + "\ndaemon_check\n").stdout

    assert "DAEMON DOWN: zz_absent_daemon.sh" in out, (
        f"daemon_check did not name the absent daemon: {out!r}"
    )
    assert "zz_present_daemon.sh" not in out, (
        f"daemon_check reported a daemon that is running: {out!r}"
    )
    assert out.count("DAEMON DOWN") == 1, f"unexpected extra reports: {out!r}"


def test_daemon_check_agrees_with_the_real_process_table():
    """The shipped name list, checked against ground truth computed here rather
    than against whatever the box happens to be doing -- so this neither passes
    by luck nor fails when a daemon legitimately restarts.  That daemon_check
    can speak at all is established by the controlled test above."""
    names = ["supervisor.sh", "preserve_evidence.sh", "context_watch.py"]
    assert SRC.count(DAEMON_LIST) == 2, (
        "daemon_check no longer watches exactly those three daemons"
    )

    out = _bash(_daemon_check_fn() + "\ndaemon_check\n").stdout
    reported = {n for n in names if f"DAEMON DOWN: {n}" in out}
    truly_absent = {n for n in names if not _running_by_argv(n)}

    assert reported == truly_absent, (
        f"daemon_check reported {sorted(reported)} but {sorted(truly_absent)} are "
        f"actually absent; output was {out!r}"
    )


# ==========================================================================
# Gaps found by mutation-testing THIS file, closed.
#
# The mutation pass that proved the tests above also found eight mutants they
# could not see. Three of those are behaviours whose loss would cost something
# real, and they are pinned here. A test that names a behaviour and stays green
# when it is reverted is the exact defect this file exists to prevent, so the
# survivors are closed rather than recorded and left.
# ==========================================================================

def _line(assign: str) -> str:
    """One `name=$(...)` assignment, verbatim, from heartbeat.sh."""
    return _extract(rf"^\s*{assign}=\$\(grep.*?\)$", f"{assign}= assignment")


def test_work_pending_assumes_work_when_it_cannot_tell(tmp_path):
    """**The fail-safe that decides whether a blind heartbeat shouts or shrugs.**

    `work_pending` shells out to import `clean_rollouts` and read its OUT. When
    that import fails there is no way to know whether the sweep is finished, and
    the script answers "assume work, stay watching" — `sys.exit(0)`. Invert it to
    `sys.exit(1)` and an unimportable driver makes the heartbeat announce SWEEP
    COMPLETE and hold: reporting a finished sweep *because it could not tell*,
    which is the failure mode the whole file was written against.

    Every other test here builds a sandbox where the import succeeds, so this
    branch was never executed and the mutation survived.
    """
    fn = _work_pending_fn()
    broken = tmp_path / "no_driver"
    broken.mkdir()
    # A repo with no tools/ and no src/: the sys.path entries the function
    # derives from REPO exist nowhere, so `import clean_rollouts` fails. Aiming
    # REPO at it beats rewriting the paths into the text -- the function is left
    # exactly as shipped.
    script = _context(broken) + fn
    r = subprocess.run(["bash", "-c", f"{script}\nwork_pending && echo PENDING || echo COMPLETE"],
                       capture_output=True, text=True, timeout=60,
                       env={"PATH": os.environ["PATH"]})
    assert "PENDING" in r.stdout, (
        "an unimportable driver must read as 'work pending, keep watching'. "
        f"Got: {r.stdout!r} {r.stderr[-200:]!r}"
    )


def test_the_failure_reported_is_the_newest_one(tmp_path):
    """`tail -1`, not `head -1`.

    The fixtures above each contain exactly one line the anchored pattern
    matches, so the two are indistinguishable and `head -1` passed. With `head`
    the heartbeat pins the first traceback of a run and never shows the current
    failure — a status line frozen on old news, which is the same shape as the
    stale-log bug this file was written for.
    """
    log = tmp_path / "runner.log"
    log.write_text(
        "Traceback (most recent call last):\n"
        "  File \"old.py\", line 1\n"
        "ok, carrying on\n"
        "arc_proxy: http://127.0.0.1:44297 (key withheld)\n"
        "aborting: the ARC endpoint is unreachable\n",
        encoding="utf-8",
    )
    got = subprocess.run(
        ["bash", "-c", f'LOG={log}\n{_line("fail")}\necho "$fail"'],
        capture_output=True, text=True, timeout=60).stdout.strip()
    assert got.startswith("aborting:"), (
        f"expected the NEWEST failure, got {got!r} — `head -1` would report the "
        "traceback and never advance"
    )


def test_the_progress_line_reported_is_the_newest_one(tmp_path):
    """Same trap on `arm=`. A progress field pinned to the first game of the
    sweep is precisely the symptom the retired-log bug produced."""
    log = tmp_path / "runner.log"
    log.write_text(
        "=== aa11-first — starting\n"
        "--- pass 1/12: 24 outstanding\n"
        "=== zz99-latest — starting\n",
        encoding="utf-8",
    )
    got = subprocess.run(
        ["bash", "-c", f'LOG={log}\n{_line("arm")}\necho "$arm"'],
        capture_output=True, text=True, timeout=60).stdout.strip()
    assert "zz99-latest" in got, f"expected the newest progress line, got {got!r}"


# ==========================================================================
# Loop-state resets. Three mutations survived the first pass because nothing
# ever drove the loop through a TRANSITION -- idle then live, or complete then
# work again. Each reset below is the line that makes a report happen once
# rather than every poll, or lets it happen again when the state returns.
#
# `runner_alive` is replaced with a file check so the test can flip liveness
# between polls. That is a seam, not a dodge: what is under test here is the
# counter arithmetic around the call, and `runner_alive` itself is pinned by
# test_runner_name_is_read_out_of_supervisor_sh and by the argv-exactness test.
# ==========================================================================

def _scriptable(box, test_expr: str) -> None:
    """Rewrite the sandbox script so runner_alive is `test_expr`.

    Liveness is made a function of the poll counter rather than of a file a
    thread races to touch: `snapshot_results.py` appends one line per iteration
    at the top of the loop, so during iteration N the counter reads N and the
    schedule below is exact. The first version of this used a thread and a flag
    file, and it could not distinguish the reset from its absence -- it never
    accumulated enough idle polls after the runner returned.
    """
    text = box.script.read_text(encoding="utf-8")
    start = text.index("runner_alive() {")
    end = text.index("\n}", start) + 2
    body = "runner_alive() {\n" + f"    {test_expr}\n" + "}\n"
    box.script.write_text(text[:start] + body + text[end:], encoding="utf-8")


def _polls_expr(box) -> str:
    return f'[ "$(wc -l < {box.sp / "polls.txt"} 2>/dev/null || echo 0)" -eq 3 ]'


def test_the_idle_counter_clears_when_the_runner_comes_back(tmp_path):
    """`IDLE_POLLS=0` on the live path.

    Without it a runner that flaps accumulates idle polls across the gaps and
    reports an absence that was never sustained. The schedule is exact: the
    runner is alive in poll 3 only.

      with the reset: 1,2 idle (2) -> poll 3 alive, cleared -> 4,5 idle (2). Silent.
      without it:     1,2 idle (2) -> poll 3 alive, not cleared -> poll 4 hits
                      IDLE_LIMIT=3 and reports.

    Run to poll 5, assert silence. Anything longer would reach the limit legitimately
    and the test would pass for the wrong reason.
    """
    # The stub freezes AT poll 5, so polls 6+ cannot slip through before the
    # kill and reach IDLE_LIMIT legitimately -- which is what happened when this
    # was written with stop_after_polls=7, and would have made the test pass for
    # the wrong reason in the other direction.
    box = _sandbox(tmp_path, games=3, banked=1, stop_after_polls=5)
    _scriptable(box, _polls_expr(box))

    out, _running, _polls = _run(box, until_polls=5, timeout=90)
    assert "NO RUNNER" not in out, (
        "the idle counter did not clear when the runner returned, so a flapping "
        f"runner reported an absence that was never sustained:\n{out}"
    )


def test_sweep_completion_can_be_announced_again_after_new_work(tmp_path):
    """`ANNOUNCED_COMPLETE=0` on the work-pending path.

    The announcement itself promises "holding, will report ... new work". Deleting
    the re-arm keeps that promise for the first cycle only: once a sweep completes
    the flag stays set, so a later completion is silent. Nothing drove
    complete -> new work -> complete.
    """
    box = _sandbox(tmp_path, games=2, banked=2, stop_after_polls=12)
    _scriptable(box, "false")                    # always idle

    import threading
    def churn():
        sp_polls = box.sp / "polls.txt"
        def n():
            return len(sp_polls.read_text().splitlines()) if sp_polls.exists() else 0
        while n() < 2: time.sleep(0.02)
        # New work appears: unbank one game, so work_pending flips to true.
        banked = sorted(box.out.glob("*/clean_result.json"))
        moved = banked[0].with_suffix(".json.away")
        banked[0].rename(moved)
        while n() < 6: time.sleep(0.02)
        moved.rename(banked[0])                  # and the sweep completes again
    t = threading.Thread(target=churn, daemon=True); t.start()

    out, _running, _polls = _run(box, until_polls=11, timeout=120)
    assert out.count("SWEEP COMPLETE") >= 2, (
        "completion was announced once and never again after new work arrived, "
        f"so the announcement's own promise is not kept:\n{out}"
    )


def test_py_bin_follows_repo(tmp_path):
    """The interpreter must come from the sandbox repo, not the live one.

    `PY_BIN` is `$REPO/.venv/bin/python`. Hardcode it and every python call in
    the loop points at this container's clone -- which on a fresh container does
    not exist, and whose absence is silent: the calls are wrapped in
    `2>/dev/null` and `|| true`, so `work_pending`, the progress counts and the
    snapshot all fail without a word while the status line keeps printing.

    The sandbox's python is a wrapper that records every invocation, so this
    asserts whose interpreter actually ran rather than that the script mentions
    a variable.
    """
    box = _sandbox(tmp_path, banked=1, games=3)
    _run(box, until_polls=2)
    marker = box.root / "pybin_used.txt"
    assert marker.exists() and marker.read_text().strip(), (
        "the sandbox interpreter was never invoked -- PY_BIN is not derived "
        "from REPO, so the heartbeat ran the live tree's python (or none at all)"
    )
