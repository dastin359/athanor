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
            proc = subprocess.run([PYTEST, *tests, "-q", "-p", "no:cacheprovider"],
                                  capture_output=True, text=True, timeout=timeout,
                                  cwd=REPO, env=env)
            tail = [ln for ln in proc.stdout.strip().splitlines()
                    if "passed" in ln or "failed" in ln or "error" in ln]
            verdict = "CAUGHT" if proc.returncode else "UNCAUGHT"
            survivors += proc.returncode == 0
            print(f"{verdict:9} {name} -- {desc}\n          "
                  f"{tail[-1] if tail else '(no summary)'}")
    finally:
        src.write_text(pristine, encoding="utf-8")
        _purge()

    print(f"\n{survivors} survivor(s) of {len(muts)}")
    return survivors


if __name__ == "__main__":
    sys.exit("import me; see the module docstring")
