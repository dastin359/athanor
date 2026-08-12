"""Apply a mutant, run the tests, restore. Report which mutants survive.

**The counter-practice this project keeps arriving at.** Nearly every defect
found here is a check, a test or a claim that names a thing and reads a proxy for
it, and the signature is that it *passes by not running*. Reading a test does not
reveal that; mutating the code it claims to protect does. `docs/ccarc3_open_findings.md`
records the tally -- six of twelve findings originally found this way, and all
six fixes verified this way.

It lived in the scratchpad until 2026-08-09, which is the one store that reverts
to an image snapshot when the container is replaced. It reverted, taking the
harness with it, on the same wake-up that found `rehydrate_box.sh` had left a
pre-fix baseline strip in that snapshot. A tool kept only there is a tool that
expires.

Two hazards it has to survive, both measured rather than imagined:

**A stale `.pyc` can serve a mutant to the next run.** CPython validates a cached
bytecode file on `(source_mtime_seconds, source_size)`. `shutil.copyfile` sets
the destination mtime to *now* and leaves the size alone -- so when a mutant and
its restore land in the same wall-clock second AND the mutation is
length-preserving (`str(p)` -> `p.name`: both six characters), the `.pyc`
compiled from the MUTANT stays "valid" and is served afterwards. That made a
correct test fail four times in a row, and adding any statement to the function
"fixed" it, by changing the file's size. Both verdict directions are corrupted by
this: a real survivor reads CAUGHT and a killed mutant reads UNCAUGHT. So:
bytecode writing off in the child, caches purged between mutants.

**It edits the live source file, so prefer `tools/audit_in_clone.sh`.** The
restore-on-signal below is sound, but while a battery runs the working tree is
dirty and "is the tree clean?" stops being answerable -- and the one resolution
that must never be taken, committing it, publishes deliberately broken source.
The wrapper runs the whole battery in a throwaway git worktree at HEAD instead.

**A green baseline is a precondition, not a nicety.** See the refusal in
:func:`run`: a suite that is already failing scores every mutant CAUGHT, because
the old verdict asked whether the suite failed rather than whether this mutant
broke it. That produced a fraudulent "31 caught, 0 survivors" on 2026-08-10.

**A mutant can kill the runner.** Removing the `ppid == 1` guard from
`clean_rollouts.kill_orphan_solvers` made the tested function SIGTERM the process
group of the pytest run testing it -- which is the test working, since a live
solver must survive -- and took this process with it. The `finally` never ran and
the mutated source stayed in the working tree, where the next command read it as
the real thing. A harness that can leave a mutant in the tree is worse than no
harness, so the pristine text is written to disk before anything is mutated and a
signal handler restores it.

Usage:

    import mutation_check as mc
    mc.run("tools/clean_rollouts.py", ["tests/test_x.py"], [
        ("name", "what this mutant does", "old text", "new text"),
    ])

`old` must occur exactly once, or the mutant is reported SKIP rather than applied
somewhere unintended.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import signal
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PYTEST = str(REPO / ".venv" / "bin" / "pytest")


def _purge() -> None:
    for cache in REPO.rglob("__pycache__"):
        if ".venv" not in cache.parts:
            shutil.rmtree(cache, ignore_errors=True)


def failed_nodes(stdout: str) -> set[str]:
    """The test ids pytest's short summary reported as FAILED or ERROR.

    Node ids, not a count. A mutant that breaks one test while coincidentally
    fixing another leaves the count unchanged, and counting would call that
    UNCAUGHT.
    """
    out = set()
    for line in stdout.splitlines():
        for tag in ("FAILED ", "ERROR "):
            if line.startswith(tag):
                out.add(line[len(tag):].split(" ")[0])
    return out


def run(source: str, tests: list[str], muts: list[tuple[str, str, str, str]],
        *, timeout: int = 1800) -> int:
    """Return the number of survivors. Zero is the only good answer."""
    src = REPO / source
    pristine = src.read_text(encoding="utf-8")
    backup = pathlib.Path("/tmp") / (src.name + ".PRISTINE")
    backup.write_text(pristine, encoding="utf-8")
    print(f"pristine copy: {backup}")

    def _restore_and_die(signum, _frame):
        src.write_text(pristine, encoding="utf-8")
        _purge()
        print(f"\nsignal {signum}: source restored, exiting")
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _restore_and_die)
        except (ValueError, OSError):
            pass

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    def _pytest():
        return subprocess.run([PYTEST, *tests, "-q", "-p", "no:cacheprovider"],
                              capture_output=True, text=True, timeout=timeout,
                              cwd=REPO, env=env)

    # **Baseline first, and refuse to proceed if it is red.**
    #
    # The verdict used to be `"CAUGHT" if proc.returncode else "UNCAUGHT"`, which
    # asks "did the suite fail?" and not "did this mutant break something that
    # was working". Those differ the moment any test in `tests` is already
    # failing: every run then exits non-zero, every mutant reads CAUGHT, and the
    # battery reports a clean sweep without having detected anything.
    #
    # Not hypothetical. On 2026-08-10 an `arc_proxy` refactor removed
    # `ProxyState.charge()` while two tests still called it. The proxy battery
    # ran 31 mutants against that suite and reported "31 caught, 0 survivors" --
    # a perfect score produced entirely by two AttributeErrors. The audit passed
    # by not running, which is the exact defect this whole tool exists to find,
    # sitting in the tool.
    #
    # So: measure the failures the mutant is responsible for, by set difference
    # against a pristine run, and refuse outright when the pristine run is not
    # green -- a red baseline makes every verdict here meaningless, and a
    # meaningless verdict that reads CAUGHT is worse than no verdict at all.
    _purge()
    base = _pytest()
    base_failed = failed_nodes(base.stdout)
    if base.returncode:
        src.write_text(pristine, encoding="utf-8")
        _purge()
        listing = "\n".join(f"    {n}" for n in sorted(base_failed)) or "    (see output)"
        raise SystemExit(
            "mutation_check: the suite is ALREADY RED on unmutated source, so "
            "every mutant would be scored CAUGHT by a failure it did not cause. "
            f"Refusing to run.\n  failing before any mutation:\n{listing}\n"
            "  Fix the suite, then re-run the battery."
        )

    survivors = 0
    try:
        for name, desc, old, new in muts:
            src.write_text(pristine, encoding="utf-8")
            _purge()
            body = src.read_text(encoding="utf-8")
            if body.count(old) != 1:
                print(f"SKIP      {name} (matches {body.count(old)}x) -- {desc}")
                continue
            src.write_text(body.replace(old, new), encoding="utf-8")
            _purge()
            proc = _pytest()
            tail = [ln for ln in proc.stdout.strip().splitlines()
                    if "passed" in ln or "failed" in ln or "error" in ln]
            # Caught only by a failure this mutant introduced. `base_failed` is
            # empty here -- the refusal above guarantees it -- so this is
            # equivalent to the old returncode test today, and stays correct if
            # that ever stops being true.
            caught = bool(failed_nodes(proc.stdout) - base_failed)
            verdict = "CAUGHT" if caught else "UNCAUGHT"
            survivors += not caught
            print(f"{verdict:9} {name} -- {desc}\n          "
                  f"{tail[-1] if tail else '(no summary)'}")
    finally:
        src.write_text(pristine, encoding="utf-8")
        _purge()

    print(f"\n{survivors} survivor(s) of {len(muts)}")
    return survivors


if __name__ == "__main__":
    sys.exit("import me; see the module docstring")
