"""Launcher argv construction and stream-json translation."""

from __future__ import annotations

import pytest
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

    def test_network_and_delegation_tools_are_denied(self, workspace, tmp_path, full_featured_cli):
        args = runner.build_cli_args(workspace, system_prompt_file=tmp_path / "sp.md")
        denied = args[args.index("--disallowed-tools") + 1]
        for tool in ("WebSearch", "WebFetch", "Task", "Agent"):
            assert tool in denied
        allowed = args[args.index("--allowedTools") + 1]
        assert "WebSearch" not in allowed
        assert "Task" not in allowed

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
