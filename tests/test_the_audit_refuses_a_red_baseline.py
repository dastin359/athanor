"""The mutation harness scored every mutant CAUGHT when the suite was already red.

`mutation_check.run` decided the verdict with

    verdict = "CAUGHT" if proc.returncode else "UNCAUGHT"

which asks *did the suite fail?* rather than *did this mutant break something
that was working*. Those are the same question only while the suite is green.
The moment any test in the list fails for its own reasons, every mutated run
exits non-zero and every mutant is scored CAUGHT — a perfect audit, detecting
nothing.

**Measured, not imagined.** On 2026-08-10 an `arc_proxy` refactor removed
`ProxyState.charge()` while two tests in the battery's own list still called it.
The proxy battery ran 31 mutants against that suite and reported *31 caught, 0
survivors*. The score was produced entirely by two AttributeErrors that had
nothing to do with any mutant. The tool built to find checks that pass by not
running had become one.

Two things now stop it: verdicts are the set difference of failing node ids
against a pristine baseline, and a red baseline is refused outright rather than
audited against.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, "tools")
import mutation_check as mc


class _Proc:
    def __init__(self, stdout, returncode=1):
        self.stdout, self.returncode = stdout, returncode


SUMMARY = textwrap.dedent("""\
    =========================== short test summary info ============================
    FAILED tests/test_a.py::test_one - AssertionError: nope
    FAILED tests/test_b.py::test_two[3-4] - AttributeError: no attribute 'charge'
    ERROR tests/test_c.py::test_three
    3 failed, 470 passed in 26.10s
""")


def test_failed_nodes_reads_ids_and_not_a_count():
    assert mc.failed_nodes(SUMMARY) == {
        "tests/test_a.py::test_one",
        "tests/test_b.py::test_two[3-4]",
        "tests/test_c.py::test_three",
    }


def test_a_green_run_names_no_failures():
    assert mc.failed_nodes("470 passed in 26.10s\n") == set()


def test_a_mutant_is_caught_only_by_a_failure_it_introduced():
    """The set difference is the whole point: pre-existing red must not count."""
    baseline = mc.failed_nodes(SUMMARY)
    unchanged = mc.failed_nodes(SUMMARY)
    assert not (unchanged - baseline), (
        "a mutant that changed nothing was scored CAUGHT by the baseline's own "
        "failures — this is the 31-of-31 result that was not a result"
    )

    broke_something = mc.failed_nodes(
        SUMMARY + "FAILED tests/test_d.py::test_four - the mutant did this\n")
    assert broke_something - baseline == {"tests/test_d.py::test_four"}


def test_a_mutant_that_repairs_a_failing_test_is_still_not_caught():
    """Counting failures would call this CAUGHT; comparing ids does not."""
    baseline = mc.failed_nodes(SUMMARY)
    swapped = mc.failed_nodes(
        "FAILED tests/test_a.py::test_one - AssertionError: nope\n"
        "FAILED tests/test_b.py::test_two[3-4] - AttributeError\n"
        "FAILED tests/test_e.py::test_five - new\n")
    assert len(swapped) == len(baseline)          # a count sees no change
    assert swapped - baseline == {"tests/test_e.py::test_five"}


def test_run_refuses_outright_when_the_baseline_is_red(tmp_path, monkeypatch):
    """End to end: a red baseline must stop the battery, not be audited against.

    The refusal is the half that matters. Set-difference verdicts alone would
    quietly keep working against a red suite, and every survivor found that way
    is reported beside failures nobody has explained.
    """
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _Proc("FAILED tests/test_x.py::test_broken - boom\n"
                     "1 failed, 2 passed in 0.10s\n", returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    target = mc.REPO / "tools" / "mutation_check.py"      # any real file; never mutated
    with pytest.raises(SystemExit) as excinfo:
        mc.run("tools/mutation_check.py", ["tests/test_x.py"],
               [("noop", "would never be applied", "def run(", "def run(")])

    message = str(excinfo.value)
    assert "ALREADY RED" in message
    assert "tests/test_x.py::test_broken" in message
    assert len(calls) == 1, "it kept going after the baseline came back red"
    assert "def run(" in target.read_text(), "the source was left damaged"
