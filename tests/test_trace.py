"""Trace reconstruction — the instrument the improvement loop runs on."""

from __future__ import annotations

import json

from conftest import IDENTITY_SOLVE, LONG_HYPOTHESIS, write_solution

from athanor.cc_harness import gate
from athanor.cc_harness.trace import collect_trace, format_trace

AUDIT = "CONFIDENCE: 4\nDECISION: ACCEPT\nREASONS:\nReflection about the vertical axis.\n"


def _verify(workspace, claim: str, holds: bool, at: str, source: str = "explore/probe.py") -> None:
    ledger = workspace.root / ".athanor" / "invariants.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": at, "claim": claim, "holds": holds, "source": source}) + "\n")


class TestCollect:
    def test_empty_workspace_traces_cleanly(self, workspace):
        trace = collect_trace(workspace.root)
        assert trace["task_id"] == "mirror01"
        assert trace["iterations"] == []
        assert trace["invariants"] == []
        assert trace["accepted"] is None

    def test_accepts_a_run_directory(self, workspace):
        # The run dir holds workspace/ — passing either should work.
        trace = collect_trace(workspace.root.parent)
        assert trace["task_id"] == "mirror01"

    def test_collects_submissions_and_acceptance(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text(AUDIT, encoding="utf-8")
        gate.cmd_accept(workspace.root)

        trace = collect_trace(workspace.root)
        assert len(trace["iterations"]) == 1
        assert trace["accepted"]["iteration"] == 1
        assert trace["final"]["confidence"] == 4
        assert trace["audit"].startswith("CONFIDENCE: 4")
        assert trace["hypothesis_history"][0]["text"] == LONG_HYPOTHESIS.strip()

    def test_counts_exploration_scripts(self, workspace):
        (workspace.root / "explore" / "shapes.py").write_text("print(1)\nprint(2)\n", encoding="utf-8")
        (workspace.root / "explore" / "colours.py").write_text("print(1)\n", encoding="utf-8")
        trace = collect_trace(workspace.root)
        assert [s["name"] for s in trace["explore_scripts"]] == ["colours.py", "shapes.py"]
        assert trace["explore_scripts"][1]["lines"] == 2

    def test_excludes_the_harness_owned_toolkit_mirror(self, workspace):
        """explore/arc.py ships with the workspace; counting it would inflate
        verification density in a run that explored nothing."""
        assert (workspace.root / "explore" / "arc.py").exists()
        trace = collect_trace(workspace.root)
        assert trace["explore_scripts"] == []
        assert trace["density"]["scripts_per_iteration"] == 0.0


class TestVerificationDensity:
    def test_splits_held_from_refuted(self, workspace):
        _verify(workspace, "outputs keep the shape", True, "2026-01-01T00:00:00+00:00")
        _verify(workspace, "outputs are 9x9", False, "2026-01-01T00:00:01+00:00")
        trace = collect_trace(workspace.root)
        assert trace["invariants_held"] == 1
        assert trace["invariants_refuted"] == 1

    def test_counts_facts_established_before_the_first_submission(self, workspace):
        _verify(workspace, "verified early", True, "2020-01-01T00:00:00+00:00")
        write_solution(workspace, code=IDENTITY_SOLVE)
        gate.cmd_submit(workspace.root)
        _verify(workspace, "verified late", True, "2099-01-01T00:00:00+00:00")

        trace = collect_trace(workspace.root)
        assert trace["density"]["verified_before_first_submission"] == 1
        assert len(trace["invariants"]) == 2

    def test_guess_first_workflow_shows_up_as_zero(self, workspace):
        """The failure mode the doctrine exists to prevent, made measurable."""
        write_solution(workspace, code=IDENTITY_SOLVE)
        gate.cmd_submit(workspace.root)
        trace = collect_trace(workspace.root)
        assert trace["density"]["verified_before_first_submission"] == 0
        assert trace["density"]["invariants_per_iteration"] == 0.0

    def test_density_is_per_iteration(self, workspace):
        for name in ("a.py", "b.py", "c.py", "d.py"):
            (workspace.root / "explore" / name).write_text("pass\n", encoding="utf-8")
        write_solution(workspace, code=IDENTITY_SOLVE)
        gate.cmd_submit(workspace.root)
        trace = collect_trace(workspace.root)
        assert trace["density"]["scripts_per_iteration"] == 4.0


class TestFormat:
    def test_reports_the_headline_numbers(self, workspace):
        _verify(workspace, "outputs keep the shape", True, "2026-01-01T00:00:00+00:00")
        write_solution(workspace, code=IDENTITY_SOLVE)
        gate.cmd_submit(workspace.root)

        text = format_trace(collect_trace(workspace.root))
        assert "VERIFICATION DENSITY" in text
        assert "outputs keep the shape" in text
        assert "#1  fail  train 0/3" in text
        assert "GATE REFUSALS: none recorded" in text

    def test_surfaces_overfit_findings_on_the_submission_line(self, workspace):
        write_solution(workspace, code="def solve(grid):\n    return [[9, 8, 7], [2, 1, 1]]\n")
        gate.cmd_submit(workspace.root)
        text = format_trace(collect_trace(workspace.root))
        assert "overfit:" in text

    def test_verbose_includes_hypothesis_and_audit(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text(AUDIT, encoding="utf-8")
        gate.cmd_accept(workspace.root)
        text = format_trace(collect_trace(workspace.root), verbose=True)
        assert "HYPOTHESIS EVOLUTION" in text
        assert "AUDIT" in text
        assert "DECISION: ACCEPT" in text
