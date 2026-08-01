"""Workspace construction, including the containment guarantees."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import MIRROR_TASK

from athanor.cc_harness.config import CCRunConfig
from athanor.cc_harness.workspace import (
    build_workspace,
    ground_truth,
    load_workspace,
    strip_test_outputs,
    workspace_env,
)


class TestContainment:
    def test_test_outputs_never_reach_the_workspace(self, workspace):
        raw = (workspace.root / "task" / "task.json").read_text(encoding="utf-8")
        task = json.loads(raw)
        assert all("output" not in entry for entry in task["test"])
        # The expected test grid must not appear anywhere in the workspace.
        expected = "".join(str(c) for row in MIRROR_TASK["test"][0]["output"] for c in row)
        for path in workspace.root.rglob("*"):
            if path.is_file() and path.suffix in {".md", ".json", ".py", ".jsonl"}:
                assert expected not in path.read_text(encoding="utf-8", errors="ignore").replace("\n", "")

    def test_strip_test_outputs_keeps_training_outputs(self):
        stripped = strip_test_outputs(MIRROR_TASK)
        assert all("output" in pair for pair in stripped["train"])
        assert all("output" not in pair for pair in stripped["test"])

    def test_ground_truth_is_extracted_for_scoring(self):
        assert ground_truth(MIRROR_TASK) == [[[3, 0, 6], [0, 4, 0]]]

    def test_solver_env_drops_the_dataset_root(self, monkeypatch):
        monkeypatch.setenv("ARC_DATA_ROOT", "/some/where/ARC-AGI-2")
        assert "ARC_DATA_ROOT" not in workspace_env()


class TestLayout:
    def test_expected_files_exist(self, workspace):
        for relative in (
            "CLAUDE.md",
            "NOTES.md",
            "arc.py",
            "gate.py",
            "task/task.json",
            "task/grids.md",
            "explore/README.md",
            ".athanor/state.json",
            ".athanor/system_prompt.md",
            ".athanor/initial_prompt.md",
            ".claude/settings.json",
            ".claude/hooks/on_compact.sh",
        ):
            assert (workspace.root / relative).is_file(), relative

    def test_gate_shim_points_at_this_checkout(self, workspace):
        source = (workspace.root / "gate.py").read_text(encoding="utf-8")
        assert "__ATHANOR_SRC__" not in source
        assert str(Path(__file__).resolve().parents[1] / "src") in source

    def test_compaction_hook_is_substituted(self, workspace):
        source = (workspace.root / ".claude" / "hooks" / "on_compact.sh").read_text(encoding="utf-8")
        assert "__WORKSPACE__" not in source and "__PYTHON__" not in source
        assert str(workspace.root) in source

    def test_state_embeds_the_config(self, workspace):
        state = json.loads(workspace.state_path.read_text(encoding="utf-8"))
        assert state["max_iterations"] == 4
        assert state["config"]["model"]

    def test_refuses_to_clobber_without_overwrite(self, tmp_path):
        root = tmp_path / "ws"
        build_workspace(task_id="a", puzzle_data=MIRROR_TASK, root=root, config=CCRunConfig(visual=False))
        with pytest.raises(FileExistsError):
            build_workspace(task_id="a", puzzle_data=MIRROR_TASK, root=root, config=CCRunConfig(visual=False))

    def test_overwrite_rebuilds_cleanly(self, tmp_path):
        root = tmp_path / "ws"
        build_workspace(task_id="a", puzzle_data=MIRROR_TASK, root=root, config=CCRunConfig(visual=False))
        (root / "explore" / "stale.py").write_text("# stale", encoding="utf-8")
        build_workspace(
            task_id="a", puzzle_data=MIRROR_TASK, root=root, config=CCRunConfig(visual=False), overwrite=True
        )
        assert not (root / "explore" / "stale.py").exists()

    def test_reopening_a_workspace_recovers_config_and_prompts(self, workspace):
        reopened = load_workspace(workspace.root)
        assert reopened.task_id == "mirror01"
        assert reopened.config.max_iterations == 4
        assert "CODE AS VERIFICATION" in reopened.system_prompt


class TestArcToolkit:
    """The toolkit is what makes 'code as verification' the literal API."""

    def _run(self, workspace, script: str) -> subprocess.CompletedProcess:
        path = workspace.root / "explore" / "probe.py"
        path.write_text(script, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(path)],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_loads_puzzle_data(self, workspace):
        result = self._run(
            workspace,
            "import sys; sys.path.insert(0, '.')\n"
            "from arc import train_samples, test_samples\n"
            "print(len(train_samples), len(test_samples))\n"
            "print('output' in test_samples[0])\n",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[0] == "3 1"
        assert result.stdout.splitlines()[1] == "False"

    def test_verify_records_to_the_ledger(self, workspace):
        result = self._run(
            workspace,
            "import sys; sys.path.insert(0, '.')\n"
            "from arc import verify, train_samples\n"
            "verify('every output keeps the input shape',\n"
            "       all(len(s['input']) == len(s['output']) for s in train_samples))\n"
            "verify('every output is 9x9', lambda: len(train_samples[0]['output']) == 9)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "[VERIFIED] every output keeps the input shape" in result.stdout
        assert "[REFUTED ] every output is 9x9" in result.stdout

        entries = [
            json.loads(line)
            for line in (workspace.root / ".athanor" / "invariants.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert [e["holds"] for e in entries] == [True, False]
        assert entries[0]["source"] == "explore/probe.py"

    def test_verify_records_a_raising_check_as_refuted(self, workspace):
        result = self._run(
            workspace,
            "import sys; sys.path.insert(0, '.')\n"
            "from arc import verify\n"
            "verify('this blows up', lambda: 1 / 0)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "[REFUTED ]" in result.stdout
        assert "ZeroDivisionError" in result.stdout

    def test_check_dry_runs_a_candidate_for_free(self, workspace):
        result = self._run(
            workspace,
            "import sys; sys.path.insert(0, '.')\n"
            "from arc import check\n"
            "check(lambda g: [row[::-1] for row in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "3/3 training examples reproduced" in result.stdout
        assert "gate.py submit" in result.stdout
        # A dry run is not an iteration.
        state = json.loads(workspace.state_path.read_text(encoding="utf-8"))
        assert state["iterations"] == []

    def test_check_reports_failures_without_raising(self, workspace):
        result = self._run(
            workspace,
            "import sys; sys.path.insert(0, '.')\n"
            "from arc import check\n"
            "check(lambda g: [[0]])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "0/3 training examples reproduced" in result.stdout

    def test_toolkit_exposes_no_transformation_primitives(self, workspace):
        """Handing over rotate/flood-fill/objects would change what is measured."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("arc_probe", workspace.root / "arc.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for banned in ("rotate", "flip", "mirror", "objects", "components", "flood_fill", "crop", "tile"):
            assert not hasattr(module, banned), f"arc.py exposes a transformation primitive: {banned}"
