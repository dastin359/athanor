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
                                stderr=subprocess.STDOUT, text=True)
        try:
            code = proc.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                code = proc.wait(timeout=S.TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
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

    src = textwrap.dedent(inspect.getsource(S.run_game))
    fn = ast.parse(src).body[0]
    handlers = [h for n in ast.walk(fn) if isinstance(n, ast.Try) for h in n.handlers
                if "TimeoutExpired" in ast.dump(h)]
    assert handlers, "run_game no longer handles TimeoutExpired"

    body = "\n".join(ast.unparse(h) for h in handlers)
    assert body.index("proc.terminate()") < body.index("proc.kill()"), (
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
