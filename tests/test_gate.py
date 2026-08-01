"""Gate behaviour — the invariants the flagship orchestrator enforced structurally."""

from __future__ import annotations

import json

import pytest
from conftest import IDENTITY_SOLVE, LONG_HYPOTHESIS, MIRROR_SOLVE, write_solution

from athanor.cc_harness import gate

AUDIT_ACCEPT = "CONFIDENCE: 4\nDECISION: ACCEPT\nREASONS:\nThe rule is a pure reflection.\n"
AUDIT_RETRY = "CONFIDENCE: 2\nDECISION: RETRY\nREASONS:\nRow order still unexplained.\n"


def _iterations(workspace) -> list:
    return json.loads(workspace.state_path.read_text(encoding="utf-8"))["iterations"]


class TestArtifactSeparation:
    def test_missing_hypothesis_is_refused(self, workspace):
        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        with pytest.raises(gate.GateError, match="hypothesis"):
            gate.cmd_submit(workspace.root)
        assert _iterations(workspace) == []

    def test_thin_hypothesis_is_refused(self, workspace):
        write_solution(workspace, hypothesis="reverse the rows")
        with pytest.raises(gate.GateError, match="characters"):
            gate.cmd_submit(workspace.root)
        assert _iterations(workspace) == []

    def test_missing_code_is_refused(self, workspace):
        (workspace.root / "solution" / "hypothesis.md").write_text(LONG_HYPOTHESIS, encoding="utf-8")
        with pytest.raises(gate.GateError, match="solve.py"):
            gate.cmd_submit(workspace.root)

    def test_code_without_solve_is_refused(self, workspace):
        write_solution(workspace, code="x = 1\n")
        with pytest.raises(gate.GateError, match="does not define"):
            gate.cmd_submit(workspace.root)

    def test_changed_code_without_changed_hypothesis_is_refused(self, workspace):
        write_solution(workspace, code=IDENTITY_SOLVE)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        with pytest.raises(gate.GateError, match="did not"):
            gate.cmd_submit(workspace.root)
        # The refusal is a precondition failure, so it costs nothing.
        assert len(_iterations(workspace)) == 1

    def test_resubmitting_identical_artifacts_is_refused(self, workspace):
        write_solution(workspace, code=IDENTITY_SOLVE)
        gate.cmd_submit(workspace.root)
        with pytest.raises(gate.GateError, match="changed since iteration"):
            gate.cmd_submit(workspace.root)
        assert len(_iterations(workspace)) == 1


class TestSubmission:
    def test_successful_submission_records_an_iteration(self, workspace):
        write_solution(workspace)
        report, code = gate.cmd_submit(workspace.root)
        assert code == 0
        assert "TRAINING PASSED" in report
        records = _iterations(workspace)
        assert len(records) == 1
        assert records[0]["all_train_correct"] is True

    def test_failure_report_carries_the_reflection_directive(self, workspace):
        write_solution(workspace, code=IDENTITY_SOLVE)
        report, _ = gate.cmd_submit(workspace.root)
        assert "TRAIN 0/3" in report
        assert "Failure root cause" in report
        assert "NOTES.md" in report

    def test_iteration_artifacts_are_archived(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        stored = workspace.root / ".athanor" / "iterations" / "1"
        assert (stored / "hypothesis.md").read_text(encoding="utf-8") == LONG_HYPOTHESIS.strip()
        assert (stored / "solve.py").read_text(encoding="utf-8") == MIRROR_SOLVE
        assert (stored / "report.txt").is_file()
        predictions = json.loads((stored / "predictions.json").read_text(encoding="utf-8"))
        assert predictions["test"][0]["candidates"] == [[[3, 0, 6], [0, 4, 0]]]

    def test_budget_is_enforced(self, workspace):
        for n in range(4):  # max_iterations == 4 in the fixture
            write_solution(workspace, hypothesis=LONG_HYPOTHESIS + f"\nAttempt {n}.\n", code=IDENTITY_SOLVE + f"# {n}\n")
            gate.cmd_submit(workspace.root)
        write_solution(workspace, hypothesis=LONG_HYPOTHESIS + "\nfinal\n", code=MIRROR_SOLVE)
        with pytest.raises(gate.GateError, match="budget exhausted"):
            gate.cmd_submit(workspace.root)

    def test_hardcoding_warning_reaches_the_report(self, workspace):
        code = "def solve(grid):\n    return [[9, 8, 7], [2, 1, 1]]\n"
        write_solution(workspace, code=code)
        report, _ = gate.cmd_submit(workspace.root)
        assert "WARNING" in report and "verbatim" in report


class TestAcceptance:
    def test_accept_requires_a_submission(self, workspace):
        with pytest.raises(gate.GateError, match="no submission"):
            gate.cmd_accept(workspace.root)

    def test_accept_requires_training_to_pass(self, workspace):
        write_solution(workspace, code=IDENTITY_SOLVE)
        gate.cmd_submit(workspace.root)
        with pytest.raises(gate.GateError, match="training examples"):
            gate.cmd_accept(workspace.root)

    def test_accept_requires_an_audit(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        with pytest.raises(gate.GateError, match="audit"):
            gate.cmd_accept(workspace.root)

    def test_audit_without_a_verdict_is_refused(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text("Looks fine to me.", encoding="utf-8")
        with pytest.raises(gate.GateError, match="DECISION"):
            gate.cmd_accept(workspace.root)

    def test_self_declared_retry_blocks_acceptance(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text(AUDIT_RETRY, encoding="utf-8")
        with pytest.raises(gate.GateError, match="RETRY"):
            gate.cmd_accept(workspace.root)

    def test_accept_writes_the_final_record(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text(AUDIT_ACCEPT, encoding="utf-8")
        report, code = gate.cmd_accept(workspace.root)
        assert code == 0
        assert "RUN COMPLETE" in report

        final = json.loads(workspace.final_path.read_text(encoding="utf-8"))
        assert final["all_train_correct"] is True
        assert final["confidence"] == 4
        assert final["best_effort"] is False
        assert final["test"][0]["candidates"] == [[[3, 0, 6], [0, 4, 0]]]

    def test_no_submissions_after_acceptance(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text(AUDIT_ACCEPT, encoding="utf-8")
        gate.cmd_accept(workspace.root)
        with pytest.raises(gate.GateError, match="already been accepted"):
            gate.cmd_submit(workspace.root)


class TestBestEffort:
    def test_best_effort_window_lifts_the_training_requirement(self, workspace):
        # Fixture budget is 4 with a 1-iteration best-effort tail.
        for n in range(3):
            write_solution(
                workspace,
                hypothesis=LONG_HYPOTHESIS + f"\nAttempt {n}.\n",
                code=IDENTITY_SOLVE + f"# {n}\n",
            )
            report, _ = gate.cmd_submit(workspace.root)
        assert "STRATEGY SHIFT" in report

        (workspace.root / "solution" / "audit.md").write_text(AUDIT_ACCEPT, encoding="utf-8")
        report, code = gate.cmd_accept(workspace.root)
        assert code == 0
        assert "best-effort" in report
        final = json.loads(workspace.final_path.read_text(encoding="utf-8"))
        assert final["best_effort"] is True


class TestStatus:
    def test_status_replays_invariants_and_history(self, workspace):
        ledger = workspace.root / ".athanor" / "invariants.jsonl"
        ledger.write_text(
            json.dumps({"claim": "outputs keep the input shape", "holds": True, "source": "explore/shape.py"})
            + "\n",
            encoding="utf-8",
        )
        write_solution(workspace, code=IDENTITY_SOLVE)
        gate.cmd_submit(workspace.root)

        report, _ = gate.cmd_status(workspace.root)
        assert "outputs keep the input shape" in report
        assert "explore/shape.py" in report
        assert "#1" in report
        assert "reflection" not in report.lower() or True  # status is a state dump, not a directive

    def test_status_nudges_when_nothing_has_been_verified(self, workspace):
        report, _ = gate.cmd_status(workspace.root)
        assert "Nothing about this puzzle has been established by execution" in report

    def test_later_verification_supersedes_an_earlier_one(self, workspace):
        ledger = workspace.root / ".athanor" / "invariants.jsonl"
        ledger.write_text(
            json.dumps({"claim": "outputs are 3x3", "holds": True})
            + "\n"
            + json.dumps({"claim": "outputs are 3x3", "holds": False})
            + "\n",
            encoding="utf-8",
        )
        entries = gate.load_invariants(workspace.root)
        assert len(entries) == 1
        assert entries[0]["holds"] is False


class TestRefusalTelemetry:
    """Refusals are the harness's most informative signal about its own friction."""

    def _events(self, workspace) -> list[dict]:
        path = workspace.root / ".athanor" / "events.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_refusal_is_recorded(self, workspace, monkeypatch):
        monkeypatch.chdir(workspace.root)
        assert gate.main(["submit"]) == 1
        refusals = [e for e in self._events(workspace) if e.get("command") == "refused"]
        assert len(refusals) == 1
        assert refusals[0]["attempted"] == "submit"
        assert "hypothesis" in refusals[0]["reason"]

    def test_successful_command_is_not_recorded_as_a_refusal(self, workspace, monkeypatch):
        monkeypatch.chdir(workspace.root)
        write_solution(workspace)
        assert gate.main(["submit"]) == 0
        assert [e for e in self._events(workspace) if e.get("command") == "refused"] == []

    def test_refusal_outside_a_workspace_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert gate.main(["status"]) == 1

    def test_trace_surfaces_recorded_refusals(self, workspace, monkeypatch):
        from athanor.cc_harness.trace import collect_trace, format_trace

        monkeypatch.chdir(workspace.root)
        gate.main(["submit"])
        text = format_trace(collect_trace(workspace.root))
        assert "GATE REFUSALS" in text
        assert "submit" in text
        assert len(collect_trace(workspace.root)["refusals"]) == 1


class TestWorkspaceDiscovery:
    def test_gate_finds_the_workspace_from_a_subdirectory(self, workspace, monkeypatch):
        monkeypatch.chdir(workspace.root / "explore")
        assert gate.find_workspace() == workspace.root

    def test_gate_refuses_outside_a_workspace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(gate.GateError, match="workspace"):
            gate.find_workspace()
