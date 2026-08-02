"""Launcher argv construction and stream-json translation."""

from __future__ import annotations

import pytest
import json

from conftest import LONG_HYPOTHESIS, MIRROR_SOLVE, MIRROR_TASK, write_solution

from athanor.cc_harness import gate, runner
from athanor.cc_harness.config import CCRunConfig
from athanor.solver.events import EventType

AUDIT = "CONFIDENCE: 5\nDECISION: ACCEPT\nREASONS:\nPure reflection, no incidental constants.\n"

FULL_HELP = (
    "--bare --effort --tools --append-system-prompt-file --setting-sources "
    "--strict-mcp-config --max-budget-usd --permission-mode --disallowed-tools"
)


@pytest.fixture
def full_featured_cli(monkeypatch):
    monkeypatch.setattr(runner, "claude_binary", lambda: "/usr/bin/claude")
    monkeypatch.setattr(runner, "_help_text", lambda: FULL_HELP)
    monkeypatch.setattr(runner, "supports_flag", lambda flag: flag in FULL_HELP)


class TestCliArgs:
    def test_core_flags(self, workspace, tmp_path, full_featured_cli):
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert args[0] == "/usr/bin/claude"
        assert args[1] == "-p"
        assert "--output-format" in args and args[args.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in args

    def test_the_solver_gets_claude_codes_full_tool_surface(self, workspace, tmp_path,
                                                            full_featured_cli):
        """"Claude Code as harness" is only honest if the agent gets Claude Code.

        The original allowlist withheld sub-agents, workflows, skills and
        background tasks — capability the product ships and a real user would
        have. Passing no --tools/--allowedTools leaves the default surface
        intact and lets the deny list do the work.
        """
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert "--tools" not in args, "surface must stay unrestricted"

    def test_bash_is_pre_approved_so_a_headless_run_never_stalls(self, workspace, tmp_path,
                                                                 full_featured_cli):
        """--allowedTools does two jobs: restrict the surface AND grant permission.

        Dropping it to open the surface also removed the grant. acceptEdits
        auto-approves file writes but not arbitrary Bash, so every
        `python explore/foo.py` was denied and two runs burned ~$2 each
        producing nothing at all.
        """
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        allowed = args[args.index("--allowedTools") + 1]
        for tool in ("Bash", "Read", "Write", "Task", "Workflow"):
            assert tool in allowed, tool

    def test_routes_out_of_the_run_are_denied(self, workspace, tmp_path, full_featured_cli):
        """Three ways out, all closed: research, network-by-another-door, escape.

        A benchmark that claims no network access has to mean it, and a solver
        must not be able to emit anything except into its workspace.
        """
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        denied = args[args.index("--disallowed-tools") + 1]
        for tool in ("WebSearch", "WebFetch",                      # research
                     "SearchMcpRegistry", "ListConnectors",         # network, other door
                     "Artifact", "SendUserFile", "PushNotification",  # escape
                     "CronCreate", "ScheduleWakeup"):
            assert tool in denied, tool

    def test_delegation_and_orchestration_are_available(self, workspace, tmp_path,
                                                        full_featured_cli):
        """Task and Workflow are deliberately NOT denied.

        This changes what the variant measures: a solver can now reconstruct
        the artifact-only reviewer this variant dropped. That is the point —
        it should be the agent's choice, not the harness's omission.
        """
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        denied = args[args.index("--disallowed-tools") + 1]
        for tool in ("Task", "Workflow", "Skill"):
            assert tool not in denied, tool

    def test_system_prompt_goes_by_file_when_supported(self, workspace, tmp_path, full_featured_cli):
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert args[args.index("--append-system-prompt-file") + 1] == str(tmp_path / "sp.md")
        assert "--append-system-prompt" not in args

    def test_falls_back_to_inline_system_prompt_on_older_cli(self, workspace, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "claude_binary", lambda: "/usr/bin/claude")
        monkeypatch.setattr(runner, "supports_flag", lambda flag: False)
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert "--append-system-prompt-file" not in args
        assert args[args.index("--append-system-prompt") + 1] == workspace.system_prompt

    def test_unsupported_flags_are_dropped(self, workspace, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "claude_binary", lambda: "/usr/bin/claude")
        monkeypatch.setattr(runner, "supports_flag", lambda flag: False)
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        for flag in ("--effort", "--tools", "--setting-sources", "--strict-mcp-config"):
            assert flag not in args

    def test_bypass_permissions_is_swapped_out_when_running_as_root(
        self, workspace, tmp_path, full_featured_cli, monkeypatch
    ):
        """`bypassPermissions` maps to --dangerously-skip-permissions, which the
        CLI refuses as root — the normal case for a containerised harness. The
        refusal arrives as an empty stream and one line of stderr."""
        monkeypatch.setattr(runner, "running_as_root", lambda: True)
        workspace.config = CCRunConfig(permission_mode="bypassPermissions", visual=False)
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert args[args.index("--permission-mode") + 1] == "acceptEdits"

    def test_bypass_permissions_is_kept_for_a_non_root_user(
        self, workspace, tmp_path, full_featured_cli, monkeypatch
    ):
        monkeypatch.setattr(runner, "running_as_root", lambda: False)
        workspace.config = CCRunConfig(permission_mode="bypassPermissions", visual=False)
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert args[args.index("--permission-mode") + 1] == "bypassPermissions"

    def test_other_modes_are_never_rewritten(self, workspace, tmp_path, full_featured_cli, monkeypatch):
        monkeypatch.setattr(runner, "running_as_root", lambda: True)
        for mode in ("acceptEdits", "dontAsk", "plan"):
            workspace.config = CCRunConfig(permission_mode=mode, visual=False)
            args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
            assert args[args.index("--permission-mode") + 1] == mode

    def test_dynamic_system_prompt_sections_are_excluded_by_default(
        self, workspace, tmp_path, full_featured_cli, monkeypatch
    ):
        """Per-task workspaces mean per-task cwd, which sits in the default
        system prompt and breaks cross-task prompt-cache reuse. Measured: the
        second task of a sequential batch wrote more cache than the first."""
        monkeypatch.setattr(
            runner, "supports_flag",
            lambda flag: flag in FULL_HELP or flag == "--exclude-dynamic-system-prompt-sections",
        )
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert "--exclude-dynamic-system-prompt-sections" in args

    def test_stable_system_prompt_can_be_turned_off(self, workspace, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "claude_binary", lambda: "/usr/bin/claude")
        monkeypatch.setattr(runner, "supports_flag", lambda flag: True)
        workspace.config = CCRunConfig(stable_system_prompt=False, visual=False)
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert "--exclude-dynamic-system-prompt-sections" not in args

    def test_the_flag_is_dropped_on_a_cli_that_lacks_it(self, workspace, tmp_path, monkeypatch):
        monkeypatch.setattr(runner, "claude_binary", lambda: "/usr/bin/claude")
        monkeypatch.setattr(runner, "supports_flag", lambda flag: False)
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert "--exclude-dynamic-system-prompt-sections" not in args

    def test_default_permission_mode_works_as_root(self):
        assert CCRunConfig().permission_mode == "acceptEdits"

    def test_budget_and_bare_are_opt_in(self, workspace, tmp_path, full_featured_cli):
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert "--max-budget-usd" not in args
        assert "--bare" not in args

        workspace.config = CCRunConfig(max_budget_usd=5.0, bare=True, visual=False)
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        assert args[args.index("--max-budget-usd") + 1] == "5.0"
        assert "--bare" in args


class TestStreamTranslation:
    def test_init_becomes_a_system_event(self):
        events = list(
            runner.stream_message_to_events(
                {"type": "system", "subtype": "init", "session_id": "s1", "model": "opus"}
            )
        )
        assert events[0].type is EventType.SYSTEM
        assert "s1" in events[0].content

    def test_assistant_blocks_split_by_kind(self):
        message = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "hello"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                ]
            },
        }
        events = list(runner.stream_message_to_events(message))
        assert [e.type for e in events] == [EventType.THINKING, EventType.TEXT, EventType.TOOL_CALL]
        assert events[2].metadata["tool_input"]["command"] == "ls"

    def test_tool_results_become_tool_result_events(self):
        message = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "ok"}]}
                ]
            },
        }
        events = list(runner.stream_message_to_events(message))
        assert events[0].type is EventType.TOOL_RESULT
        assert events[0].content == "ok"

    def test_result_carries_cost_and_turns(self):
        message = {
            "type": "result",
            "subtype": "success",
            "result": "done",
            "total_cost_usd": 1.25,
            "num_turns": 40,
            "session_id": "s1",
        }
        events = list(runner.stream_message_to_events(message))
        assert events[0].type is EventType.COMPLETE
        assert events[0].metadata["cost_usd"] == 1.25
        assert events[0].metadata["num_turns"] == 40

    def test_api_retry_is_surfaced(self):
        events = list(
            runner.stream_message_to_events(
                {"type": "system", "subtype": "api_retry", "attempt": 2, "max_retries": 5, "error": "overloaded"}
            )
        )
        assert events[0].type is EventType.SYSTEM
        assert "overloaded" in events[0].content

    def test_unknown_message_types_are_ignored(self):
        assert list(runner.stream_message_to_events({"type": "stream_event"})) == []


class TestRescoringFromAWorkspace:
    """A workspace driven by a sub-agent has no result.json — score it anyway."""

    def _accept(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text(AUDIT, encoding="utf-8")
        gate.cmd_accept(workspace.root)

    def test_scores_a_run_with_no_result_json(self, workspace, task_json):
        self._accept(workspace)
        run_dir = workspace.root.parent
        assert not (run_dir / "result.json").is_file()

        record = runner.rescore_run(run_dir, dataset_root=task_json.parent)
        assert record["task_id"] == "mirror01"
        assert record["accepted"] is True
        assert record["score"]["solved"] is True
        assert (run_dir / "result.json").is_file()

    def test_rescoring_supersedes_a_stale_record(self, workspace, task_json):
        run_dir = workspace.root.parent
        # A record written before the agent did any work — exactly what
        # `cc workspace` used to leave behind.
        (run_dir / "result.json").write_text(
            json.dumps({"task_id": "mirror01", "accepted": False, "iterations_used": 0}), encoding="utf-8"
        )
        self._accept(workspace)

        record = runner.rescore_run(run_dir, dataset_root=task_json.parent)
        assert record["accepted"] is True
        assert record["iterations_used"] == 1

    def test_records_an_integrity_verdict(self, workspace, task_json):
        self._accept(workspace)
        record = runner.rescore_run(workspace.root.parent, dataset_root=task_json.parent)
        assert record["integrity"]["suspected"] is False

    def test_discover_runs_finds_workspace_only_directories(self, workspace):
        out_dir = workspace.root.parent.parent
        found = runner.discover_runs(out_dir)
        assert workspace.root.parent in found

    def test_discover_runs_on_a_missing_directory(self, tmp_path):
        assert runner.discover_runs(tmp_path / "nope") == []


class TestBatch:
    """A batch is sequential so the provider cache warms across tasks; one bad
    task must not cost the rest of the sweep."""

    def _fake_result(self, task):
        return {
            "task_id": task, "accepted": True, "cost_usd": 1.5, "iterations_used": 2,
            "score": {"scored": True, "solved": True, "score": 1.0, "num_solved": 1, "num_test": 1},
        }

    def test_a_failing_task_does_not_stop_the_sweep(self, tmp_path, monkeypatch):
        attempted = []

        def fake_run_task(task, **kwargs):
            attempted.append(task)
            if task == "boom":
                raise RuntimeError("resolver exploded")
            return self._fake_result(task)

        monkeypatch.setattr(runner, "run_task", fake_run_task)
        results = runner.run_batch(["a", "boom", "c"], out_dir=tmp_path, config=CCRunConfig())

        assert attempted == ["a", "boom", "c"]
        assert results[1]["error"].startswith("RuntimeError")
        assert [r.get("task_id") for r in results] == ["a", "boom", "c"]

    def test_aggregate_separates_errors_from_unsolved(self, tmp_path, monkeypatch):
        from athanor.cc_harness.scoring import aggregate

        def fake_run_task(task, **kwargs):
            if task == "boom":
                raise RuntimeError("nope")
            return self._fake_result(task)

        monkeypatch.setattr(runner, "run_task", fake_run_task)
        results = runner.run_batch(["a", "boom", "c"], out_dir=tmp_path, config=CCRunConfig())
        summary = aggregate(results)

        assert summary["solved"] == 2
        assert summary["scored"] == 2, "an errored task must not count as scored"
        assert summary["errors"] == ["boom"]
        assert summary["mean_cost_usd"] == 1.5

    def test_on_task_done_fires_per_task(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(runner, "run_task", lambda task, **kw: self._fake_result(task))
        runner.run_batch(
            ["a", "b"], out_dir=tmp_path, config=CCRunConfig(),
            on_task_done=lambda record: seen.append(record["task_id"]),
        )
        assert seen == ["a", "b"]


class TestResume:
    """Crash recovery works because the workspace is the state."""

    def test_resume_prompt_carries_the_surviving_state(self, workspace):
        from athanor.cc_harness import prompt as prompt_mod

        write_solution(workspace, code="def solve(grid):\n    return grid\n")
        gate.cmd_submit(workspace.root)

        text = prompt_mod.build_resume_prompt(
            task_id="mirror01",
            state=workspace.read_state(),
            interpreter={"command": "python"},
        )
        assert "gate.py status" in text
        assert "1 of 4 submissions" in text
        assert "3 left" in text
        assert "scored 0/3 on training" in text
        assert ".athanor/iterations/1/report.txt" in text
        # It must be explicit about what did not survive, or the new agent will
        # assume the previous one's unrecorded conclusions still hold.
        assert "did NOT survive" in text
        assert "re-derived" in text

    def test_resume_is_skipped_for_an_accepted_run(self, workspace, task_json, monkeypatch):
        from athanor.cc_harness import runner as runner_mod

        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text(AUDIT, encoding="utf-8")
        gate.cmd_accept(workspace.root)

        monkeypatch.setattr(
            runner_mod, "_launch", lambda *a, **k: pytest.fail("must not relaunch an accepted run")
        )
        record = runner.resume_task(workspace.root.parent, dataset_root=task_json.parent)
        assert record["resumed"] is False
        assert record["resume_skipped"] == "already accepted"
        assert record["score"]["solved"] is True

    def test_resume_is_skipped_when_the_budget_is_gone(self, workspace, task_json, monkeypatch):
        from athanor.cc_harness import runner as runner_mod

        for n in range(4):  # fixture budget is 4
            write_solution(
                workspace,
                hypothesis=LONG_HYPOTHESIS + f"\nAttempt {n}.\n",
                code=f"def solve(grid):\n    return grid  # {n}\n",
            )
            gate.cmd_submit(workspace.root)

        monkeypatch.setattr(
            runner_mod, "_launch", lambda *a, **k: pytest.fail("must not relaunch with no budget")
        )
        record = runner.resume_task(workspace.root.parent, dataset_root=task_json.parent)
        assert record["resumed"] is False
        assert record["resume_skipped"] == "budget exhausted"

    def test_resume_relaunches_and_rescores(self, workspace, task_json, monkeypatch):
        from athanor.cc_harness import runner as runner_mod

        write_solution(workspace, code="def solve(grid):\n    return grid\n")
        gate.cmd_submit(workspace.root)

        def fake_launch(ws, **kwargs):
            # Stand in for the agent finishing the job.
            write_solution(ws, hypothesis=LONG_HYPOTHESIS + "\nCorrected.\n")
            gate.cmd_submit(ws.root)
            (ws.root / "solution" / "audit.md").write_text(AUDIT, encoding="utf-8")
            gate.cmd_accept(ws.root)
            return {"result_message": {"total_cost_usd": 0.5}, "returncode": 0}

        monkeypatch.setattr(runner_mod, "_launch", fake_launch)
        record = runner.resume_task(workspace.root.parent, dataset_root=task_json.parent)
        assert record["resumed"] is True
        assert record["accepted"] is True
        assert record["iterations_used"] == 2
        assert record["score"]["solved"] is True


class TestOutcomeCollection:
    def test_accepted_run_is_collected_and_scored(self, workspace):
        write_solution(workspace)
        gate.cmd_submit(workspace.root)
        (workspace.root / "solution" / "audit.md").write_text(AUDIT, encoding="utf-8")
        gate.cmd_accept(workspace.root)

        outcome = runner.collect_outcome(workspace, MIRROR_TASK)
        assert outcome["accepted"] is True
        assert outcome["iterations_used"] == 1
        assert outcome["train_perfect"] is True
        assert outcome["confidence"] == 5
        assert outcome["score"]["solved"] is True
        assert outcome["code"] == MIRROR_SOLVE
        assert outcome["hypothesis"] == LONG_HYPOTHESIS.strip()

    def test_unfinished_run_collects_without_a_score(self, workspace):
        write_solution(workspace, code="def solve(grid):\n    return grid\n")
        gate.cmd_submit(workspace.root)
        outcome = runner.collect_outcome(workspace, MIRROR_TASK)
        assert outcome["accepted"] is False
        assert outcome["iterations_used"] == 1
        assert outcome["score"]["scored"] is False
