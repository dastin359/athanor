"""A wall-clock timeout must not SIGKILL away the run's own accounting.

The solver's stdout is a FILE, not a tty, so its runtime block-buffers it.
`proc.kill()` on timeout therefore discarded whatever sat in that buffer — and
`stream.jsonl` is the only source for `run_cost`'s turn count and dollar figure.
The scored ledger is unaffected (`trace.jsonl` is written per action by the
client), so this is accounting accuracy, not score.

The grace period also makes a clean exit possible for the first time, which is
worth recording rather than filing as a kill.

**Provenance, stated because it matters:** the SIGTERM idea arrived as an
unrequested edit from an agent in a read-only audit, outside the finding channel
and with no verification. It was reverted and re-derived here — the agent's
version hard-coded `code = -1` after the graceful wait, throwing away the one new
fact the change creates.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from athanor.ccarc3 import session as S  # noqa: E402


def _child(body: str) -> list[str]:
    return [sys.executable, "-c", textwrap.dedent(body)]


def _run_under_timeout(tmp_path: Path, argv: list[str], timeout: float):
    """The exact shutdown sequence from run_game, against a real child."""
    out = tmp_path / "stream.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(argv, cwd=str(tmp_path), stdout=fh,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            S._signal_group(proc, signal.SIGTERM)
            try:
                code = proc.wait(timeout=S.TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                S._signal_group(proc, signal.SIGKILL)
                code = proc.wait()
            timed_out = True
    return code, timed_out, out.read_text(encoding="utf-8")


def test_a_buffered_tail_survives_the_shutdown(tmp_path):
    """The point of the change. The child writes without flushing, so under
    SIGKILL the tail is lost; a SIGTERM handler that flushes keeps it."""
    argv = _child("""
        import signal, sys, time
        def bye(sig, frm):
            sys.stdout.flush()
            sys.exit(0)
        signal.signal(signal.SIGTERM, bye)
        for i in range(200):
            sys.stdout.write('{"event": %d}\\n' % i)   # no flush: block-buffered
        time.sleep(60)
    """)
    code, timed_out, text = _run_under_timeout(tmp_path, argv, timeout=1.0)
    assert timed_out is True
    assert '{"event": 199}' in text, (
        "the buffered tail was lost — this is the accounting data run_cost reads"
    )
    assert code == 0, f"a clean shutdown must keep its exit status, got {code!r}"


def test_a_solver_that_ignores_sigterm_is_still_killed(tmp_path):
    """The grace period must not become a hang. A child that traps SIGTERM and
    keeps running is SIGKILLed after TERM_GRACE_S."""
    argv = _child("""
        import signal, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        sys.stdout.write("started\\n"); sys.stdout.flush()
        time.sleep(600)
    """)
    monkey = S.TERM_GRACE_S
    try:
        S.TERM_GRACE_S = 2          # keep the test fast; the path is identical
        began = time.monotonic()
        code, timed_out, text = _run_under_timeout(tmp_path, argv, timeout=1.0)
        elapsed = time.monotonic() - began
    finally:
        S.TERM_GRACE_S = monkey
    assert timed_out is True
    assert "started" in text
    assert code == -9, f"an unresponsive solver must be SIGKILLed, got {code!r}"
    assert elapsed < 30, f"the grace period became a hang ({elapsed:.1f}s)"


def _run_path_source() -> str:
    """`run_game` plus the helper it hands the spawn to.

    The spawn and the timeout arm moved out of `run_game` into `_launch` when
    nudging landed, because a nudged run makes several launches. Both tests below
    read the source of the run path, so they have to follow it -- otherwise they
    assert about a function that no longer contains the thing they name, which is
    how the first version of this failed after the refactor.

    The delegation is asserted rather than assumed: reading `_launch` would
    "pass" just as well if `run_game` had stopped calling it.
    """
    import inspect
    run = inspect.getsource(S.run_game)
    assert "_launch(" in run, (
        "run_game no longer delegates to _launch; this helper is reading a "
        "function that may not be on the run path at all"
    )
    return run + "\n" + inspect.getsource(S._launch)


def test_the_run_path_terminates_first_and_keeps_the_status():
    """Pinned against the real function, because the behavioural tests above run
    a COPY of the sequence and cannot see a change to `run_game` itself.

    Asserted as a property rather than as the absence of one wrong spelling: the
    first version checked `"code, timed_out = -1, True" not in src`, and a mutant
    that wrote `proc.wait(...); code = -1` sailed past it. What matters is that
    every `code` assignment inside the timeout handler comes FROM a wait call,
    never from a constant.
    """
    import ast
    import inspect

    src = textwrap.dedent(_run_path_source())
    tree = ast.parse(src)
    handlers = [h for n in ast.walk(tree) if isinstance(n, ast.Try) for h in n.handlers
                if "TimeoutExpired" in ast.dump(h)]
    assert handlers, "the run path no longer handles TimeoutExpired"

    body = "\n".join(ast.unparse(h) for h in handlers)
    assert body.index("SIGTERM") < body.index("SIGKILL"), (
        "run_game must SIGTERM before it SIGKILLs"
    )
    assert "TERM_GRACE_S" in body, "the grace period is not the shared constant"

    assigns = [n for h in handlers for n in ast.walk(h)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "code" for t in n.targets)]
    assert assigns, "nothing assigns `code` in the timeout handler"
    for a in assigns:
        rendered = ast.unparse(a.value)
        assert "proc.wait" in rendered, (
            f"`code` is assigned from {rendered!r}, not from the wait: a clean "
            "exit during the grace period would be filed as a kill"
        )


def test_the_timeout_reaches_grandchildren(tmp_path):
    """**A solver's game-driving children outlived the timeout.**

    The workspace CLAUDE.md tells the solver to drive the game from `python -c`
    subprocesses, and each POSTs actions of its own. Without a new session the
    child sat in the driver's process group, so a signal to `proc` reached only
    the `claude` process and left those spending the budget against a run the
    driver had already declared timed out.
    """
    marker = tmp_path / "grandchild_alive"
    argv = _child(f"""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c",
            "import time\\nwhile True:\\n    open({str(marker)!r}, 'a').write('x')\\n    time.sleep(0.05)"])
        sys.stdout.write("spawned\\n"); sys.stdout.flush()
        time.sleep(600)
    """)
    grace = S.TERM_GRACE_S
    try:
        S.TERM_GRACE_S = 2
        _run_under_timeout(tmp_path, argv, timeout=1.5)
    finally:
        S.TERM_GRACE_S = grace

    assert marker.exists(), "the grandchild never ran; the fixture proves nothing"
    size = marker.stat().st_size
    time.sleep(1.0)
    assert marker.stat().st_size == size, (
        "the grandchild is still writing after the timeout — it survived the "
        "kill and is still spending the budget"
    )


def test_the_run_path_starts_its_own_session_and_signals_the_group():
    """Pinned against the real function, since the behavioural test drives a copy."""
    src = _run_path_source()
    assert "start_new_session=True" in src, (
        "the solver is not a process-group leader, so a kill cannot reach its "
        "descendants"
    )
    assert "_signal_group(" in src, "the timeout arm signals only the direct child"
    assert "proc.terminate()" not in src and "proc.kill()" not in src, (
        "a direct-child signal survives beside the group signal, so the tree can "
        "still be left running"
    )
