"""The reach pass, which decides whether a run's evidence is believed.

Two defects, found by the 2026-08-07 audit *after* the pass had already been
rewritten once for the same directory:

1. It looked only at absolute paths, so `cat ../../../../best_or_last/card.json`
   — which reaches a real scorecard holding two complete `level_baseline_actions`
   arrays — was invisible. Every other check missed the same read: the
   own-per-level-array test needs literal brackets and commas, and the
   foreign-median test needs the literal string `baseline_actions`, both of which
   parsing rather than printing avoids.
2. In `--gz` mode the root came from wherever the evidence was filed, not from
   where the solver stood, so every self-reference read as an escape and the
   documented way to re-audit the durable record declared all 30 banked runs
   void. A check that fails on the whole corpus is not a check.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import pytest

import proofread_trace as pt

WS = pathlib.Path("/tmp/scratchpad/clean_rollouts/zz99-deadbeef/attempt_1/zz99-deadbeef")


@pytest.mark.parametrize("command", [
    "cat ../../../../best_or_last/card.json",
    "cat ../../../../arc3/.env",
    "python3 -c \"print(open('../../../../../evidence/x').read())\"",
])
def test_a_relative_path_that_escapes_the_workspace_is_flagged(command):
    assert pt.strayed(command, WS), f"relative escape went unseen: {command}"


@pytest.mark.parametrize("command", [
    "sys.path.insert(0,'..')",          # from notes/, lands back in the workspace
    "cat ../notes/plan.md",
    "cat notes/plan.md",
    "ls /usr/bin/python3",
    f"cd {WS} && cat session.py",
])
def test_ordinary_work_inside_the_workspace_is_not_flagged(command):
    """A `..` is not an escape. Resolution is from the deepest plausible cwd, so
    this can only under-report — never cry wolf on a run that stayed put."""
    assert not pt.strayed(command, WS), f"false positive on: {command}"


def test_the_root_is_recovered_from_the_run_not_from_where_evidence_sits():
    """`--gz` hands in the evidence directory; the run ran somewhere else."""
    live = "/tmp/scratchpad/clean_rollouts/zz99-deadbeef/attempt_1/zz99-deadbeef"
    cmds = [f"cd {live} && cat session.py",
            f"python3 {live}/notes/probe.py",
            f"cat {live}/DOCTRINE.md"]
    assert pt.recover_root(cmds, "zz99-deadbeef") == pathlib.Path(live)


def test_root_recovery_declines_rather_than_guessing():
    """No absolute self-reference means keep what the caller passed in."""
    assert pt.recover_root(["cat notes/plan.md", "ls"], "zz99-deadbeef") is None
