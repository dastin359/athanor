"""Prompt composition, including the shared-with-flagship domain knowledge."""

from __future__ import annotations

import pytest
from conftest import MIRROR_TASK

from athanor.cc_harness import prompt
from athanor.cc_harness.config import CCRunConfig


class TestSharedSections:
    def test_flagship_prompt_still_has_the_shared_headings(self):
        """The two harnesses must agree on ARC domain knowledge.

        If this fails, SOLVER_SYSTEM_PROMPT.md was restructured and the CC
        variant is silently diverging on priors — which would make any
        harness-vs-harness comparison meaningless.
        """
        sections = prompt.split_sections(prompt.solver_prompt_path().read_text(encoding="utf-8"))
        for title in prompt.SHARED_SECTIONS:
            assert title in sections, f"missing section {title!r} in the flagship solver prompt"

    def test_shared_sections_carry_the_domain_knowledge(self):
        text = prompt.shared_arc_sections()
        assert "ARC Cognitive Priors" in text
        assert "Occam's Razor" in text
        assert "Geometric Transformations" in text

    def test_shared_sections_exclude_flagship_tool_names(self):
        # Sections 3+ are harness-specific; the shared slice must not mention
        # tools that do not exist in this variant.
        text = prompt.shared_arc_sections()
        assert "execute_python_solution" not in text
        assert "run_code_in_previous_runtime" not in text

    def test_missing_section_raises_a_clear_error(self, monkeypatch):
        monkeypatch.setattr(prompt, "SHARED_SECTIONS", ("99. NOT A SECTION",))
        with pytest.raises(RuntimeError, match="no longer contains"):
            prompt.shared_arc_sections()

    def test_split_sections_drops_trailing_rules(self):
        sections = prompt.split_sections("## A\n\nbody\n\n---\n\n## B\n\nother\n")
        assert sections["A"] == "body"
        assert sections["B"] == "other"


class TestSystemPrompt:
    def test_contains_shared_knowledge_and_doctrine(self):
        text = prompt.build_system_prompt()
        assert "ARC Cognitive Priors" in text
        assert "CODE AS VERIFICATION" in text
        assert "You are your own reviewer" in text

    def test_is_task_independent(self):
        # Task-independence is what lets a batch share the provider prompt cache.
        text = prompt.build_system_prompt()
        assert "mirror01" not in text
        assert "task/task.json" not in text


class TestTaskPresentation:
    def test_grid_text_is_one_row_per_line(self):
        assert prompt.render_grid_text([[1, 2, 0], [0, 3, 0]]) == "120\n030"

    def test_task_markdown_covers_every_pair(self):
        text = prompt.render_task_markdown("mirror01", MIRROR_TASK)
        assert text.count("## Training pair") == 3
        assert text.count("## Test input") == 1
        assert "120\n030" in text

    def test_initial_prompt_points_at_the_contract(self):
        text = prompt.build_initial_prompt(
            task_id="mirror01",
            puzzle_data=MIRROR_TASK,
            config=CCRunConfig(max_iterations=7),
            image_files=["task/images/train_0_input.png"],
        )
        assert "CLAUDE.md" in text
        assert "arc.py" in text
        assert "7 formal submissions" in text
        assert "task/images/train_0_input.png" in text
        assert "120\n030" in text

    def test_initial_prompt_can_omit_inline_grids(self):
        text = prompt.build_initial_prompt(
            task_id="mirror01",
            puzzle_data=MIRROR_TASK,
            config=CCRunConfig(inline_grids=False),
            image_files=[],
        )
        assert "120\n030" not in text
        assert "task/grids.md" in text


class TestWorkspaceContract:
    def test_claude_md_is_fully_substituted(self):
        text = prompt.build_workspace_claude_md(
            task_id="mirror01",
            puzzle_data=MIRROR_TASK,
            config=CCRunConfig(max_iterations=9, best_effort_iterations=3, min_hypothesis_chars=250),
            interpreter={"command": "python", "version": "3.11.0", "modules": ["numpy"], "probed": True},
        )
        import re

        leftover = re.findall(r"__[A-Z_]+__", text)
        assert leftover == [], f"unsubstituted placeholders survived: {leftover}"
        assert "task `mirror01`" in text
        assert "**9 submissions.**" in text
        assert "last 3 of" in text
        assert "250" in text

    def test_claude_md_substitutes_the_interpreter_command(self):
        text = prompt.build_workspace_claude_md(
            task_id="mirror01",
            puzzle_data=MIRROR_TASK,
            config=CCRunConfig(),
            interpreter={"command": "/opt/py/bin/python", "version": "3.12.1", "modules": [], "probed": True},
        )
        assert "/opt/py/bin/python gate.py submit" in text
        assert "__PYTHON__" not in text


class TestEnvironmentDescription:
    """A contract that promises NumPy on a runtime without it is worse than silence."""

    def test_lists_what_is_present(self):
        text = prompt.describe_environment(
            {"command": "python", "version": "3.11.0", "modules": ["numpy", "PIL"], "probed": True}
        )
        assert "Python 3.11.0" in text
        assert "`numpy`" in text and "`PIL`" in text

    def test_names_what_is_missing(self):
        text = prompt.describe_environment(
            {"command": "python", "version": "3.11.0", "modules": [], "probed": True}
        )
        assert "NOT installed" in text
        assert "`numpy`" in text
        assert "do not spend an experiment discovering this" in text

    def test_unprobed_runtime_says_so_rather_than_guessing(self):
        text = prompt.describe_environment({"command": "python", "probed": False})
        assert "could not probe" in text

    def test_missing_interpreter_info_is_handled(self):
        assert "python" in prompt.describe_environment(None)

    def test_settings_json_wires_the_compaction_hook(self, tmp_path):
        import json

        settings = json.loads(prompt.settings_json(hook_script=tmp_path / "on_compact.sh"))
        entry = settings["hooks"]["SessionStart"][0]
        assert entry["matcher"] == "compact"
        assert entry["hooks"][0]["args"] == [str(tmp_path / "on_compact.sh")]


class TestAblation:
    """An ablation that is not verified to have removed anything is not an ablation.

    These exist because a doctrine ablation ran for weeks against an edited
    workspace file while every arm received the doctrine verbatim in the appended
    system prompt. Nothing in the harness could compose a prompt without it, and
    nothing in the run record said which arm was which.
    """

    def test_the_default_prompt_contains_the_doctrine(self):
        """Regression guard for the mislabelled ablation: assert the baseline."""
        assert "## 3. GOAL" in prompt.build_system_prompt()

    def test_ablating_the_doctrine_actually_removes_it(self):
        full = prompt.build_system_prompt()
        ablated = prompt.build_system_prompt(("doctrine",))
        assert "## 3. GOAL" in full
        assert "## 3. GOAL" not in ablated
        assert len(ablated) < len(full)
        # the shared domain knowledge must survive, or the arms differ on priors too
        assert "1. ROLE & IDENTITY" in ablated

    def test_an_unknown_target_raises_rather_than_doing_nothing(self):
        with pytest.raises(ValueError, match="unknown ablation target"):
            prompt.build_system_prompt(("Doctrine",))

    def test_a_workspace_target_is_refused_at_the_system_prompt_layer(self):
        with pytest.raises(ValueError, match="workspace CLAUDE.md"):
            prompt.build_system_prompt(("workspace:Rival readings",))

    def test_stripping_a_section_leaves_its_neighbours_intact(self):
        text = "# T\n\nintro\n\n## Keep me\n\nA\n\n## Drop me\n\nB\n\n## Also keep\n\nC\n"
        out = prompt.strip_markdown_sections(text, ["Drop me"])
        assert "## Keep me" in out and "## Also keep" in out
        assert "## Drop me" not in out and "\nB\n" not in out

    def test_stripping_a_missing_section_raises(self):
        with pytest.raises(ValueError, match="no '## Nope' section"):
            prompt.strip_markdown_sections("## Real\n\nx\n", ["Nope"])

    def test_workspace_claude_md_honours_the_ablation(self):
        config = CCRunConfig(ablate=("workspace:Rival readings",))
        baseline = prompt.build_workspace_claude_md(
            task_id="t", puzzle_data=MIRROR_TASK, config=CCRunConfig()
        )
        ablated = prompt.build_workspace_claude_md(
            task_id="t", puzzle_data=MIRROR_TASK, config=config
        )
        assert "## Rival readings" in baseline
        assert "## Rival readings" not in ablated
        assert "## The invariant ledger" in ablated  # only the named section goes

    def test_the_run_record_states_its_own_manipulation(self):
        """The original ablation was invisible in result.json — same config both arms."""
        assert CCRunConfig().to_dict()["ablate"] == ()
        assert CCRunConfig(ablate=("doctrine",)).to_dict()["ablate"] == ("doctrine",)
        assert CCRunConfig(ablate=("doctrine",)).to_dict() != CCRunConfig().to_dict()
