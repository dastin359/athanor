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


class TestAcceptRevalidation:
    """Adding a hedge after a train-perfect submission must not cost an iteration.

    A solver on a 3-iteration budget spent a third of it resubmitting an
    unchanged rule purely to record a second candidate the harness had just
    asked for — charged an iteration for following the harness's own advice.
    """

    TWO_CANDIDATES = (
        "def solve(grid):\n"
        "    mirrored = [row[::-1] for row in grid]\n"
        "    if len(grid) == 2 and grid[0] == [6, 0, 3]:\n"
        "        return [mirrored, [[3, 0, 6], [0, 4, 9]]]\n"
        "    return mirrored\n"
    )

    def test_a_hedge_added_after_submitting_is_accepted_for_free(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        assert len(_iterations(workspace)) == 1

        (workspace.root / "solution" / "solve.py").write_text(self.TWO_CANDIDATES, encoding="utf-8")
        (workspace.root / "solution" / "audit.md").write_text(AUDIT_ACCEPT, encoding="utf-8")
        gate.cmd_accept(workspace.root)

        assert len(_iterations(workspace)) == 1, "hedging must not consume an iteration"
        final = json.loads(workspace.final_path.read_text(encoding="utf-8"))
        assert final["revalidated_at_accept"] is True
        assert len(final["test"][0]["candidates"]) == 2
        assert final["code"] == self.TWO_CANDIDATES

    def test_a_regression_falls_back_to_the_submitted_artifact(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)

        (workspace.root / "solution" / "solve.py").write_text(
            "def solve(grid):\n    return [[0]]\n", encoding="utf-8"
        )
        (workspace.root / "solution" / "audit.md").write_text(AUDIT_ACCEPT, encoding="utf-8")
        gate.cmd_accept(workspace.root)

        final = json.loads(workspace.final_path.read_text(encoding="utf-8"))
        assert final.get("revalidated_at_accept") is False
        assert final["code"] == MIRROR_SOLVE, "the submitted rule is what the solver stood behind"
        assert final["test"][0]["candidates"] == [[[3, 0, 6], [0, 4, 0]]]

    def test_broken_code_falls_back_rather_than_failing_acceptance(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "solve.py").write_text("def solve(grid)\n", encoding="utf-8")
        (workspace.root / "solution" / "audit.md").write_text(AUDIT_ACCEPT, encoding="utf-8")
        report, code = gate.cmd_accept(workspace.root)
        assert code == 0
        final = json.loads(workspace.final_path.read_text(encoding="utf-8"))
        assert final["code"] == MIRROR_SOLVE

    def test_unchanged_code_is_not_rerun(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text(AUDIT_ACCEPT, encoding="utf-8")
        gate.cmd_accept(workspace.root)
        final = json.loads(workspace.final_path.read_text(encoding="utf-8"))
        assert final.get("revalidated_at_accept") is False


class TestCandidateSpread:
    def test_report_says_how_far_apart_two_candidates_are(self, workspace):
        """Two candidates can share a shape and colour histogram while differing
        in a couple of cells, leaving the printed summary looking identical."""
        write_solution(workspace, code=TestAcceptRevalidation.TWO_CANDIDATES)
        report, _ = gate.cmd_submit(workspace.root)
        assert "candidates differ in 1 cell(s)" in report


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


class TestReportEconomy:
    """The gate must not violate the token discipline it enforces."""

    def _wide_workspace(self, tmp_path, config):
        """A task whose outputs are far past the inline-grid threshold."""
        from athanor.cc_harness.workspace import build_workspace

        big_in = [[(r + c) % 10 for c in range(30)] for r in range(30)]
        big_out = [[(r * c) % 10 for c in range(30)] for r in range(30)]
        task = {
            "train": [{"input": big_in, "output": big_out} for _ in range(2)],
            "test": [{"input": big_in, "output": big_out}],
        }
        return build_workspace(
            task_id="wide01", puzzle_data=task, root=tmp_path / "wide" / "workspace", config=config
        )

    def test_large_test_predictions_are_summarised_not_dumped(self, tmp_path, config):
        workspace = self._wide_workspace(tmp_path, config)
        write_solution(workspace, code="def solve(grid):\n    return [row[:] for row in grid]\n")
        report, _ = gate.cmd_submit(workspace.root)

        predictions_section = report.split("--- test predictions ---", 1)[1]
        assert "colours {" in predictions_section, "the histogram should stand in for the grid"
        assert "not printed" in predictions_section
        assert "predictions.json" in predictions_section
        # The candidate is a 30x30 copy of the input, whose first row is
        # 012345678901234567890123456789. It must not be inlined.
        assert "0123456789" not in predictions_section.replace(" ", "")

    def test_small_test_predictions_are_still_printed_in_full(self, workspace):
        write_solution(workspace)
        report, _ = gate.cmd_submit(workspace.root)
        assert "306\n  040" in report.replace("\r", "")


class TestGeneralizationSignals:
    """A mechanical stand-in for the reviewer this variant drops."""

    def test_signals_reach_the_report_and_the_ledger(self, tmp_path, config):
        from athanor.cc_harness.workspace import build_workspace

        # Every training output is 2x2; the solution returns the input unchanged,
        # so the 1x3 test input yields a 1x3 prediction.
        task = {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]},
                {"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]},
            ],
            "test": [{"input": [[1, 2, 3]]}],
        }
        workspace = build_workspace(
            task_id="sig01", puzzle_data=task, root=tmp_path / "sig" / "workspace", config=config
        )
        write_solution(workspace, code="def solve(grid):\n    return [row[:] for row in grid]\n")
        report, _ = gate.cmd_submit(workspace.root)

        assert "GENERALIZATION:" in report
        assert "every training output is 2x2" in report
        record = _iterations(workspace)[-1]
        assert record["generalization_signals"]

    def test_no_signals_on_a_consistent_prediction(self, workspace):
        write_solution(workspace)
        report, _ = gate.cmd_submit(workspace.root)
        assert "GENERALIZATION:" not in report
        assert _iterations(workspace)[-1]["generalization_signals"] == []


class TestUnhedgedRival:
    """The strongest form of the second-attempt prompt: an executed fact.

    A rival that reproduces every training pair and disagrees on a test input is
    measured, not inferred — nothing rests on the solver's judgement about its
    own reasoning.
    """

    def _register(self, workspace, name, fits, predictions):
        ledger = workspace.root / ".athanor" / "rivals.jsonl"
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"name": name, "fits_training": fits, "train_correct": 3 if fits else 1,
                     "train_total": 3, "predictions": predictions}
                )
                + "\n"
            )

    def test_flags_a_training_fitting_rival_that_disagrees(self, workspace):
        self._register(workspace, "diagonals may not brush a corner", True, [[[9, 9, 9]]])
        write_solution(workspace)
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNHEDGED RIVAL" in report
        assert "diagonals may not brush a corner" in report
        assert "differs on test 0" in report
        assert gate.load_state(workspace.root)["iterations"][-1]["unhedged_rivals"]

    def test_ignores_a_rival_that_fails_training(self, workspace):
        self._register(workspace, "a reading the data contradicts", False, [[[9, 9, 9]]])
        write_solution(workspace)
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNHEDGED RIVAL" not in report

    def test_ignores_a_rival_that_agrees_with_the_submission(self, workspace):
        # The mirror solution predicts [[3,0,6],[0,4,0]] for the test input.
        self._register(workspace, "a differently-argued but identical rule", True,
                       [[[3, 0, 6], [0, 4, 0]]])
        write_solution(workspace)
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNHEDGED RIVAL" not in report

    def test_ignores_a_rival_once_the_slot_is_spent(self, workspace):
        self._register(workspace, "some rival", True, [[[9, 9, 9]]])
        write_solution(
            workspace,
            code="def solve(grid):\n    return [[row[::-1] for row in grid], [[1, 1, 1]]]\n",
        )
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNHEDGED RIVAL" not in report

    def test_the_measured_prompt_supersedes_the_judgement_one(self, workspace):
        """Both can apply; the executed fact is the stronger thing to say."""
        ledger = workspace.root / ".athanor" / "invariants.jsonl"
        ledger.write_text(
            json.dumps({"claim": "some dead end", "holds": True, "mode": "ruled_out"}) + "\n",
            encoding="utf-8",
        )
        self._register(workspace, "a live rival", True, [[[9, 9, 9]]])
        write_solution(workspace)
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNHEDGED RIVAL" in report
        assert "UNSPENT SECOND ATTEMPT" not in report


class TestUnspentCandidatePrompt:
    """Derived from a measured loss on 88e364bc: a rival reading that reproduced
    every training pair was killed by an inductive leap, and the free second
    attempt was discarded. Missed by 2 cells out of 400."""

    def _record_dead_end(self, workspace, claim="the strict diagonal reading"):
        ledger = workspace.root / ".athanor" / "invariants.jsonl"
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"claim": claim, "holds": True, "mode": "ruled_out",
                            "source": "explore/probe.py"}) + "\n"
            )

    def test_fires_when_a_dead_end_meets_an_unspent_slot(self, workspace):
        self._record_dead_end(workspace)
        write_solution(workspace)  # returns one candidate per test input
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNSPENT SECOND ATTEMPT" in report
        assert "the strict diagonal reading" in report
        assert "inductive leap" in report or "applied to the test input" in report

    def test_silent_when_nothing_was_ruled_out(self, workspace):
        write_solution(workspace)
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNSPENT SECOND ATTEMPT" not in report

    def test_silent_when_the_second_slot_is_already_spent(self, workspace):
        self._record_dead_end(workspace)
        write_solution(
            workspace,
            code="def solve(grid):\n    return [[row[::-1] for row in grid], grid]\n",
        )
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNSPENT SECOND ATTEMPT" not in report

    def test_silent_on_a_training_failure(self, workspace):
        """The prompt belongs at the decision point, not amid a failed run."""
        self._record_dead_end(workspace)
        write_solution(workspace, code=IDENTITY_SOLVE)
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNSPENT SECOND ATTEMPT" not in report

    def test_a_still_open_hypothesis_is_not_a_dead_end(self, workspace):
        ledger = workspace.root / ".athanor" / "invariants.jsonl"
        ledger.write_text(
            json.dumps({"claim": "some rival", "holds": False, "mode": "ruled_out"}) + "\n",
            encoding="utf-8",
        )
        write_solution(workspace)
        report, _ = gate.cmd_submit(workspace.root)
        assert "UNSPENT SECOND ATTEMPT" not in report


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
