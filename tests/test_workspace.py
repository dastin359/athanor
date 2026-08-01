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
            "dryrun.py",
            "explore/arc.py",
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

    def test_import_works_from_the_documented_invocation(self, workspace):
        """`python explore/foo.py` puts explore/ on sys.path, not the workspace.

        Regression test: the contract tells the agent to run experiments exactly
        this way, and without the mirrored module `from arc import ...` raises
        ModuleNotFoundError — friction on the single most common action in the
        loop.
        """
        result = self._run(workspace, "from arc import train_samples\nprint(len(train_samples))\n")
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "3"

    def test_import_works_from_a_root_one_liner(self, workspace):
        import subprocess
        import sys as _sys

        result = subprocess.run(
            [_sys.executable, "-c", "from arc import train_samples; print(len(train_samples))"],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "3"

    def test_dryrun_scores_a_correct_solution(self, workspace):
        import subprocess
        import sys as _sys

        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        result = subprocess.run(
            [_sys.executable, "dryrun.py"],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "3/3 training examples reproduced" in result.stdout

    def test_rival_says_when_the_slot_would_change_nothing(self, workspace):
        """The payload is whether spending the slot changes anything.

        A solver reported rival() advising "consider it for your second
        candidate" about a reading that predicted identically — advice the
        function had the data to know was wrong.
        """
        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import rival\n"
            "rival('a differently-argued but identical rule', lambda g: [r[::-1] for r in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "reproduces all 3 training pairs" in result.stdout
        assert "would change nothing" in result.stdout

    def test_rival_names_where_it_diverges(self, workspace):
        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import rival\n"
            "def alt(g):\n"
            "    if g[0][0] == 6:\n"           # the test input, not any training input
            "        return [[9, 9, 9], [9, 9, 9]]\n"
            "    return [r[::-1] for r in g]\n"
            "rival('a genuinely divergent reading', alt)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "predicts differently on test 0" in result.stdout
        assert "your second candidate" in result.stdout

    def test_rival_recognises_itself_as_the_existing_second_candidate(self, workspace):
        """Reported live, and the advice was inverted where it mattered most.

        A solver hedged correctly — its 3/3-fitting rival was already candidate
        2 — and rival() told it the reading "is not a divergent reading and
        needs no slot", because the rival's prediction was found inside the
        candidate list it had just been added to. Acting on that line deletes a
        correct hedge. "Already your second candidate" and "redundant with your
        first" are opposite situations.
        """
        (workspace.root / "solution" / "solve.py").write_text(
            "def solve(grid):\n"
            "    mine = [row[::-1] for row in grid]\n"
            "    if grid[0][0] == 6:\n"        # the test input, not any training input
            "        return [mine, [[9, 9, 9], [9, 9, 9]]]\n"
            "    return mine\n",
            encoding="utf-8",
        )
        result = self._run(
            workspace,
            "from arc import rival\n"
            "def alt(g):\n"
            "    if g[0][0] == 6:\n"
            "        return [[9, 9, 9], [9, 9, 9]]\n"
            "    return [r[::-1] for r in g]\n"
            "rival('the reading already hedged into slot 2', alt)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "already your second candidate" in result.stdout
        assert "Leave it in place" in result.stdout
        assert "needs no slot" not in result.stdout
        assert "would change nothing" not in result.stdout

    def test_rival_says_when_both_slots_are_already_spent_elsewhere(self, workspace):
        """Reported by both round-5 solvers.

        With two candidates already shipped on a test example, "That is your
        second candidate" describes an addition that is not available — the
        second candidate exists and holds a different reading. The real choice
        is which two of three survive.
        """
        (workspace.root / "solution" / "solve.py").write_text(
            "def solve(grid):\n"
            "    mine = [row[::-1] for row in grid]\n"
            "    if grid[0][0] == 6:\n"
            "        return [mine, [[7, 7, 7], [7, 7, 7]]]\n"
            "    return mine\n",
            encoding="utf-8",
        )
        result = self._run(
            workspace,
            "from arc import rival\n"
            "def alt(g):\n"
            "    if g[0][0] == 6:\n"
            "        return [[9, 9, 9], [9, 9, 9]]\n"   # neither shipped candidate
            "    return [r[::-1] for r in g]\n"
            "rival('a third reading', alt)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "both slots are already spent" in result.stdout
        assert "which two of the three" in result.stdout
        assert "That is your second candidate" not in result.stdout

    def test_rival_defers_when_there_is_no_solution_yet(self, workspace):
        result = self._run(
            workspace,
            "from arc import rival\nrival('early rival', lambda g: [r[::-1] for r in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "No solution/solve.py to compare against yet" in result.stdout

    def test_refute_accepts_retract(self, workspace):
        """Reported as a TypeError: retraction only worked through verify()."""
        result = self._run(
            workspace,
            "from arc import refute, invariants\n"
            "refute('a mistaken dead end', True)\n"
            "refute('a mistaken dead end', retract=True)\n"
            "print('LIVE', len(invariants()))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "[RETRACTED]" in result.stdout
        assert "LIVE 0" in result.stdout

    def test_hedging_advice_fires_on_an_empty_rival_ledger(self, workspace):
        """The first dry run is the one moment before submitting when acting is free.

        Requiring a pre-existing rival meant silence exactly there: a solver
        registered its rivals afterwards and only saw the question from the
        gate, after the iteration was spent.
        """
        result = self._run(
            workspace,
            "from arc import check\ncheck(lambda g: [row[::-1] for row in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "registered no rival readings" in result.stdout
        assert "arc.rival(name, fn)" in result.stdout

    def test_verify_flags_a_claim_recorded_without_a_readable_call_site(self, workspace):
        """Reported live: a heredoc refutation with a literal True, and no warning.

        The constant-condition check is derived from the caller's source, so
        when the source cannot be read the check cannot fire — and the entry
        looked exactly like a well-evidenced one.
        """
        import subprocess
        import sys as _sys

        result = subprocess.run(
            [_sys.executable, "-c", "from arc import verify\nverify('a claim from a one-liner', True)\n"],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "NO EVIDENCE CAPTURED" in result.stdout
        assert "explore/" in result.stdout

        import json as _json

        ledger = workspace.root / ".athanor" / "invariants.jsonl"
        entries = [_json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        assert entries[-1]["unsourced"] is True

    def test_verify_from_a_file_is_not_flagged_as_unsourced(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "verify('there are three training pairs', len(train_samples) == 3)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "NO EVIDENCE CAPTURED" not in result.stdout

    def test_verify_shows_the_claim_a_rewording_displaced(self, workspace):
        """Observed live: a live invariant whose own first clause was false.

        With no way to say "that reading died, here is the replacement", a
        solver wrote the correction into the claim text — "the jog has a fixed
        width of 1 (m = a-1); it is instead always a-2 wide" — and the harness
        stamped [VERIFIED] on it. After a compaction that sentence is what gets
        read back.
        """
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "n = len(train_samples)\n"
            "verify('there are four training pairs', n == 4, key='pair-count')\n"
            "verify('there are three training pairs', n == 3, key='pair-count')\n",
        )
        assert result.returncode == 0, result.stderr
        assert "supersedes under key `pair-count`" in result.stdout
        assert "there are four training pairs" in result.stdout
        assert "[REFUTED]" in result.stdout
        assert "State only what is true" in result.stdout

    def test_verify_announces_a_changed_verdict(self, workspace):
        """A claim that flips is the most informative event in the ledger.

        It used to print twice, identically, with nothing marking that the
        second run contradicted the first.
        """
        result = self._run(
            workspace,
            "from arc import verify\n"
            "import os\n"
            "verify('the flag file exists', os.path.exists('flag.tmp'))\n"
            "open('flag.tmp', 'w').close()\n"
            "verify('the flag file exists', os.path.exists('flag.tmp'))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "CHANGED VERDICT" in result.stdout
        assert "[REFUTED]" in result.stdout
        assert "now suspect" in result.stdout

    def test_verify_is_quiet_when_a_claim_is_merely_reconfirmed(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "verify('there are three training pairs', len(train_samples) == 3)\n"
            "verify('there are three training pairs', len(train_samples) == 3)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "CHANGED VERDICT" not in result.stdout
        assert "supersedes" not in result.stdout

    def test_sweep_records_a_whole_tie_break_as_one_entry(self, workspace):
        """Three solvers ran 8-, 14- and 18-way sweeps and recorded 2-3 lines.

        One reported it directly: "the ledger under-represents what was
        actually ruled out." A sweep is a single finding.
        """
        result = self._run(
            workspace,
            "from arc import sweep, invariants\n"
            "n = 3\n"
            "alive = sweep('what sets the centre colour?', {\n"
            "    'largest blob': n == 4,\n"
            "    'longest branch': n == 5,\n"
            "    'most branch cells': n == 3,\n"
            "})\n"
            "print('ALIVE', alive)\n"
            "print('ENTRIES', len(invariants()))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "3 readings tested, 1 survive" in result.stdout
        assert "alive: most branch cells" in result.stdout
        assert "dead : largest blob" in result.stdout
        assert "ALIVE ['most branch cells']" in result.stdout
        assert "ENTRIES 1" in result.stdout

    def test_sweep_calls_multiple_survivors_a_hedging_obligation(self, workspace):
        result = self._run(
            workspace,
            "from arc import sweep\n"
            "n = 3\n"
            "sweep('what sets the centre colour?', {\n"
            "    'blob majority': n == 3,\n"
            "    'branch count': n == 3,\n"
            "    'longest branch': n == 9,\n"
            "})\n",
        )
        assert result.returncode == 0, result.stderr
        assert "2 survive" in result.stdout
        assert "hedging obligation" in result.stdout
        assert "arc.rival(name, fn)" in result.stdout

    def test_sweep_with_no_survivors_says_not_to_pick_the_least_bad(self, workspace):
        result = self._run(
            workspace,
            "from arc import sweep\n"
            "n = 3\n"
            "sweep('what sets the centre colour?', {'a': n == 1, 'b': n == 2})\n",
        )
        assert result.returncode == 0, result.stderr
        assert "nothing survived" in result.stdout
        assert "least-bad" in result.stdout

    def test_hedging_advice_names_an_undecided_sweep(self, workspace):
        """A multi-survivor sweep is the strongest hedging signal in the workspace.

        The solver has already established by execution that training cannot
        separate the survivors — asking it to go find rivals ignores work it has
        already done.
        """
        result = self._run(
            workspace,
            "from arc import sweep, check\n"
            "n = 3\n"
            "sweep('what sets the centre colour?', {\n"
            "    'blob majority': n == 3,\n"
            "    'branch count': n == 3,\n"
            "    'longest branch': n == 9,\n"
            "})\n"
            "check(lambda g: [row[::-1] for row in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "left 2 readings alive" in result.stdout
        assert "blob majority, branch count" in result.stdout
        assert "you already did the hard part" in result.stdout
        assert "registered no rival readings" not in result.stdout

    def test_hedging_advice_ignores_a_decided_sweep(self, workspace):
        result = self._run(
            workspace,
            "from arc import sweep, check\n"
            "n = 3\n"
            "sweep('what sets the centre colour?', {'a': n == 3, 'b': n == 9})\n"
            "check(lambda g: [row[::-1] for row in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "readings alive" not in result.stdout
        assert "registered no rival readings" in result.stdout

    def test_situation_types_prompt_reaches_the_free_path(self, workspace):
        """A solver called this "the single highest-value string the harness printed".

        It produced the script that found the one unwitnessed case on its test
        input and changed what it shipped — and it had only ever appeared after
        an iteration was spent. Third time a convenience has been found on the
        budgeted path alone.
        """
        result = self._run(
            workspace,
            "from arc import check\ncheck(lambda g: [row[::-1] for row in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "enumerate the situation types your rule has to handle" in result.stdout
        assert "witnessed in a training pair" in result.stdout

    def test_situation_types_prompt_matches_the_gate_wording(self):
        """arc.py is standalone in the workspace, so the text is duplicated.

        Duplicated guidance drifts silently; this is what keeps it honest.
        """
        from pathlib import Path

        from athanor.cc_harness import reporting
        from athanor.cc_harness.workspace import ASSETS

        toolkit = (Path(ASSETS) / "arc_toolkit.py").read_text(encoding="utf-8")
        for phrase in (
            "enumerate the situation types your rule has to handle",
            "witnessed in a training pair",
        ):
            assert phrase in reporting._unspent_candidate_prompt([0], ["a dead end"]), phrase
            assert phrase in toolkit, phrase

    def test_rival_announces_that_it_replaced_an_earlier_registration(self, workspace):
        """Reported live: re-registering a name silently changed what it meant.

        A solver fixed a buggy rival implementation by re-registering the same
        name. verify() announces supersession; rival() said nothing, so the
        earlier entry — which had claimed divergence on both tests — vanished
        without trace.
        """
        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import rival\n"
            "rival('the diagonal reading', lambda g: [r[::-1] for r in g])\n"
            "def fixed(g):\n"
            "    if g[0][0] == 6:\n"
            "        return [[9, 9, 9], [9, 9, 9]]\n"
            "    return [r[::-1] for r in g]\n"
            "rival('the diagonal reading', fixed)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "[REPLACED]" in result.stdout
        assert "give them two names" in result.stdout

    def test_rival_is_quiet_when_a_re_registration_changes_nothing(self, workspace):
        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import rival\n"
            "rival('the same reading', lambda g: [r[::-1] for r in g])\n"
            "rival('the same reading', lambda g: [r[::-1] for r in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "[REPLACED]" not in result.stdout

    def test_opaque_name_warning_is_suppressed_when_a_note_carries_the_values(self, workspace):
        """A solver passed both a variable and a full per-candidate note.

        It was still told to "pass note= with the measured value". It had.
        """
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "ok = len(train_samples) == 3\n"
            "verify('there are three training pairs', ok, note='measured 3 pairs')\n",
        )
        assert result.returncode == 0, result.stderr
        assert "recorded evidence is just the name" not in result.stdout

    def test_opaque_name_warning_still_fires_without_a_note(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "ok = len(train_samples) == 3\n"
            "verify('there are three training pairs', ok)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "recorded evidence is just the name" in result.stdout

    def test_evidence_records_what_a_multiline_check_measured(self, workspace):
        """The AST capture only reads a single expression at the call site.

        A solver whose every substantive check was a multi-line function had two
        options: a bare name (correctly flagged as opaque) or a one-expression
        comprehension so unreadable it produced "a genuinely bad ledger entry I
        had to retract". Neither is good.
        """
        result = self._run(
            workspace,
            "from arc import verify, train_samples, invariants\n"
            "shapes = [(len(s['output']), len(s['output'][0])) for s in train_samples]\n"
            "ok = all(h == 2 for h, _ in shapes)\n"
            "verify('every training output has two rows', ok, evidence=shapes)\n"
            "print('MEASURED', invariants()[-1].get('measured'))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "recorded evidence is just the name" not in result.stdout
        assert "MEASURED [(2, 3), (2, 3), (2, 3)]" in result.stdout

    def test_evidence_answers_the_unreadable_call_site_warning(self, workspace):
        import subprocess
        import sys as _sys

        result = subprocess.run(
            [_sys.executable, "-c",
             "from arc import verify\nverify('a claim', True, evidence={'n': 3})\n"],
            cwd=str(workspace.root), capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "NO EVIDENCE CAPTURED" not in result.stdout

    def test_evidence_is_truncated_not_dumped(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify, invariants\n"
            "verify('a claim', True, evidence=list(range(500)))\n"
            "print('LEN', len(invariants()[-1]['measured']))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "LEN 400" in result.stdout

    def test_sweep_treats_a_raising_candidate_as_dead(self, workspace):
        result = self._run(
            workspace,
            "from arc import sweep\n"
            "def boom():\n"
            "    raise ValueError('nope')\n"
            "sweep('which reading holds?', {'explodes': boom, 'fine': lambda: True})\n",
        )
        assert result.returncode == 0, result.stderr
        assert "1 survive" in result.stdout
        assert "dead : explodes" in result.stdout

    def test_durability_note_fires_without_show(self, workspace):
        """Hung off show() alone it reached almost nobody.

        Measured across a round: agents called show() in one script out of five
        to eight, usually an early one — so the reminder was gated on a call
        they had mostly stopped making by the time the condition became true.
        """
        explore = workspace.root / "explore"
        for index in range(5):
            (explore / f"probe_{index}.py").write_text("pass\n", encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import train_samples\nprint('rows', len(train_samples))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "exploration scripts, nothing recorded" in result.stdout

    def test_durability_note_is_silent_once_something_is_recorded(self, workspace):
        explore = workspace.root / "explore"
        for index in range(5):
            (explore / f"probe_{index}.py").write_text("pass\n", encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "verify('there are three training pairs', len(train_samples) == 3)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "nothing recorded" not in result.stdout

    def test_durability_note_is_silent_below_the_threshold(self, workspace):
        result = self._run(
            workspace,
            "from arc import train_samples\nprint('rows', len(train_samples))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "nothing recorded" not in result.stdout

    def test_verify_flags_a_bare_name_as_opaque_evidence(self, workspace):
        """Observed live: a ledger entry whose recorded evidence was `allok`.

        The ledger's whole value is that it says what ran; a variable name says
        nothing to the context that reads it after a compaction.
        """
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "allok = all(len(s['input']) == len(s['output']) for s in train_samples)\n"
            "verify('shapes are preserved', allok)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "just the name `allok`" in result.stdout

        entry = json.loads(
            (workspace.root / ".athanor" / "invariants.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert entry["opaque"] is True

    def test_an_inline_expression_is_not_opaque(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "verify('three pairs', len(train_samples) == 3)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "just the name" not in result.stdout

    def test_exploration_reminder_fires_before_any_dry_run(self, workspace):
        """The dry-run nudges only reach a solver that already has a solution.

        Two agents were observed forty minutes into a hard task with six
        exploration scripts, no solve.py, and an empty ledger — the window where
        a compaction costs most, and nothing was saying so.
        """
        for name in ("a", "b", "c", "d", "e"):
            (workspace.root / "explore" / f"{name}.py").write_text("pass\n", encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import show, train_samples\nshow(train_samples[0]['input'])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "nothing recorded with arc.verify()" in result.stdout

    def test_exploration_reminder_fires_at_most_once(self, workspace):
        for name in ("a", "b", "c", "d", "e"):
            (workspace.root / "explore" / f"{name}.py").write_text("pass\n", encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import show, train_samples\n"
            "show(train_samples[0]['input'])\n"
            "show(train_samples[1]['input'])\n"
            "show(train_samples[2]['input'])\n",
        )
        assert result.stdout.count("nothing recorded with arc.verify()") == 1

    def test_exploration_reminder_is_quiet_once_something_is_recorded(self, workspace):
        for name in ("a", "b", "c", "d", "e"):
            (workspace.root / "explore" / f"{name}.py").write_text("pass\n", encoding="utf-8")
        (workspace.root / ".athanor" / "invariants.jsonl").write_text(
            json.dumps({"claim": "a recorded fact", "holds": True}) + "\n", encoding="utf-8"
        )
        result = self._run(
            workspace,
            "from arc import show, train_samples\nshow(train_samples[0]['input'])\n",
        )
        assert "nothing recorded" not in result.stdout

    def test_exploration_reminder_is_quiet_early(self, workspace):
        """Two scripts in is not the moment to lecture about durability."""
        (workspace.root / "explore" / "a.py").write_text("pass\n", encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import show, train_samples\nshow(train_samples[0]['input'])\n",
        )
        assert "nothing recorded" not in result.stdout

    def test_notes_nudge_fires_when_facts_outpace_direction(self, workspace):
        """Observed live: seven scripts and six invariants, NOTES.md pristine."""
        import subprocess
        import sys as _sys

        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        ledger = workspace.root / ".athanor" / "invariants.jsonl"
        ledger.write_text(
            "\n".join(
                json.dumps({"claim": f"fact {n}", "holds": True, "key": f"k{n}"}) for n in range(3)
            )
            + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [_sys.executable, "dryrun.py"], cwd=str(workspace.root),
            capture_output=True, text=True, timeout=60,
        )
        assert "NOTES.md is still the template" in result.stdout

    def test_notes_nudge_is_quiet_once_notes_are_written(self, workspace):
        import subprocess
        import sys as _sys

        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        (workspace.root / ".athanor" / "invariants.jsonl").write_text(
            "\n".join(
                json.dumps({"claim": f"fact {n}", "holds": True, "key": f"k{n}"}) for n in range(3)
            )
            + "\n",
            encoding="utf-8",
        )
        (workspace.root / "NOTES.md").write_text(
            "# NOTES\n\n## Current hypothesis\n\nRows are reversed.\n", encoding="utf-8"
        )
        result = subprocess.run(
            [_sys.executable, "dryrun.py"], cwd=str(workspace.root),
            capture_output=True, text=True, timeout=60,
        )
        assert "still the template" not in result.stdout

    def test_notes_nudge_is_quiet_early(self, workspace):
        """One fact in is not the moment to lecture about durability."""
        import subprocess
        import sys as _sys

        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        (workspace.root / ".athanor" / "invariants.jsonl").write_text(
            json.dumps({"claim": "one fact", "holds": True}) + "\n", encoding="utf-8"
        )
        result = subprocess.run(
            [_sys.executable, "dryrun.py"], cwd=str(workspace.root),
            capture_output=True, text=True, timeout=60,
        )
        assert "still the template" not in result.stdout

    def test_hedging_advice_arrives_where_acting_on_it_is_free(self, workspace):
        """A candidate comes out of solve(), so acting on the gate's version of
        this costs a whole iteration to resubmit. One solver on a 3-iteration
        budget spent a third of it doing exactly that."""
        (workspace.root / ".athanor" / "invariants.jsonl").write_text(
            json.dumps({"claim": "the strict reading", "holds": True, "mode": "ruled_out"}) + "\n",
            encoding="utf-8",
        )
        result = self._run(
            workspace,
            "from arc import check\ncheck(lambda g: [row[::-1] for row in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "carries one candidate" in result.stdout
        assert "costs an iteration" in result.stdout

    def test_hedging_advice_prefers_a_measured_rival(self, workspace):
        (workspace.root / ".athanor" / "rivals.jsonl").write_text(
            json.dumps({"name": "the strict reading", "fits_training": True,
                        "predictions": [[[9]]]}) + "\n",
            encoding="utf-8",
        )
        result = self._run(
            workspace,
            "from arc import check\ncheck(lambda g: [row[::-1] for row in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "differing on test 0: the strict reading" in result.stdout
        assert "Training cannot separate it" in result.stdout

    def test_hedging_advice_skips_a_rival_that_diverges_elsewhere(self, workspace):
        """Naming a rival that differs on another test example trains the solver
        to skim the nudge. Reported from a two-test frontier run."""
        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        (workspace.root / ".athanor" / "rivals.jsonl").write_text(
            json.dumps(
                {
                    "name": "agrees here, differs elsewhere",
                    "fits_training": True,
                    # identical to the mirror solution's prediction for test 0
                    "predictions": [[[3, 0, 6], [0, 4, 0]]],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = self._run(
            workspace,
            "from arc import check\ncheck(lambda g: [row[::-1] for row in g])\n",
        )
        assert result.returncode == 0, result.stderr
        assert "agrees here, differs elsewhere" not in result.stdout
        # It should fall through to the generic prompt, not stay silent.
        assert "carries one candidate" in result.stdout

    def test_check_reports_how_far_apart_two_candidates_are(self, workspace):
        """Otherwise the only way to see this is the gate's report, which costs
        an iteration — a solver spent one using the gate as a viewer."""
        result = self._run(
            workspace,
            "from arc import check\n"
            "def alt(g):\n"
            "    if g[0][0] == 6:\n"
            "        return [[r[::-1] for r in g], [[3, 0, 6], [0, 4, 9]]]\n"
            "    return [r[::-1] for r in g]\n"
            "check(alt)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "candidates differ in 1 cell(s)" in result.stdout

    def test_hedging_advice_is_raised_even_with_nothing_recorded(self, workspace):
        """An empty rival ledger is not evidence that there are no rivals.

        Superseded an earlier design where silence was the response; that
        skipped the prompt on the first dry run, the one place it is free to
        act on.
        """
        result = self._run(
            workspace,
            "from arc import check\ncheck(lambda g: [row[::-1] for row in g])\n",
        )
        assert "carries one candidate" in result.stdout
        assert "not the same as there being" in result.stdout

    def test_no_hedging_advice_once_the_slot_is_spent(self, workspace):
        (workspace.root / ".athanor" / "invariants.jsonl").write_text(
            json.dumps({"claim": "the strict reading", "holds": True, "mode": "ruled_out"}) + "\n",
            encoding="utf-8",
        )
        result = self._run(
            workspace,
            "from arc import check\n"
            "check(lambda g: [[row[::-1] for row in g], [[1, 1, 1]]])\n",
        )
        assert "carries one candidate" not in result.stdout

    def test_no_hedging_advice_while_training_still_fails(self, workspace):
        """Hedging is a question for a rule that works, not one being debugged."""
        (workspace.root / ".athanor" / "invariants.jsonl").write_text(
            json.dumps({"claim": "the strict reading", "holds": True, "mode": "ruled_out"}) + "\n",
            encoding="utf-8",
        )
        result = self._run(
            workspace,
            "from arc import check\ncheck(lambda g: [row[:] for row in g])\n",
        )
        assert "carries one candidate" not in result.stdout

    def test_dryrun_nudges_when_the_ledger_is_empty(self, workspace):
        """Running experiments and recording them are different acts.

        On the hardest observed tasks agents wrote several exploration scripts
        and recorded nothing — so a compaction would have discarded everything
        they had worked out.
        """
        import subprocess
        import sys as _sys

        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        result = subprocess.run(
            [_sys.executable, "dryrun.py"],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "nothing recorded in the invariant ledger yet" in result.stdout

    def test_dryrun_is_quiet_once_something_is_recorded(self, workspace):
        import subprocess
        import sys as _sys

        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        (workspace.root / ".athanor" / "invariants.jsonl").write_text(
            json.dumps({"claim": "outputs keep the input shape", "holds": True}) + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [_sys.executable, "dryrun.py"],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "nothing recorded" not in result.stdout

    def test_dryrun_reports_a_missing_solution_without_a_traceback(self, workspace):
        import subprocess
        import sys as _sys

        result = subprocess.run(
            [_sys.executable, "dryrun.py"],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 1
        assert "does not exist yet" in result.stdout
        assert "Traceback" not in result.stderr

    def test_dryrun_reports_a_broken_solution_without_a_traceback(self, workspace):
        import subprocess
        import sys as _sys

        (workspace.root / "solution" / "solve.py").write_text("def solve(grid)\n", encoding="utf-8")
        result = subprocess.run(
            [_sys.executable, "dryrun.py"],
            cwd=str(workspace.root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 1
        assert "failed to load" in result.stdout

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
        assert "[VERIFIED ] every output keeps the input shape" in result.stdout
        assert "[REFUTED  ] every output is 9x9" in result.stdout

        entries = [
            json.loads(line)
            for line in (workspace.root / ".athanor" / "invariants.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert [e["holds"] for e in entries] == [True, False]
        assert entries[0]["source"] == "explore/probe.py"

    def test_verify_records_the_expression_as_evidence(self, workspace):
        """The ledger records what was executed, not only what was claimed."""
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "verify('all inputs are 2 rows tall', all(len(s['input']) == 2 for s in train_samples))\n",
        )
        assert result.returncode == 0, result.stderr
        entry = json.loads(
            (workspace.root / ".athanor" / "invariants.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert "len(s['input']) == 2 for s in train_samples" in entry["expression"]
        assert "literal" not in entry

    def test_verify_flags_a_constant_condition(self, workspace):
        """A tautology is an assertion wearing a verification's clothes.

        An agent shipped exactly this — `verify(claim, True if ... else True)` —
        into the ledger that survives compaction and is replayed as established
        fact. The doctrine's own failure mode, occurring inside the tool built
        to prevent it.
        """
        result = self._run(
            workspace,
            "from arc import verify\nverify('the noise colour is well separated', True)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "WARNING" in result.stdout
        assert "compile-time constant" in result.stdout

        entry = json.loads(
            (workspace.root / ".athanor" / "invariants.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert entry["literal"] is True
        assert entry["holds"] is True  # it did evaluate true — that is the trap

    def test_verify_flags_a_constant_lambda_body(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify\nverify('always fine', lambda: True)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "WARNING" in result.stdout

    def test_verify_does_not_flag_a_real_lambda(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "verify('three pairs', lambda: len(train_samples) == 3)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "WARNING" not in result.stdout

    def test_refute_records_a_dead_end_as_a_finding(self, workspace):
        """A false verify() reads as a defect; ruling something out is a result."""
        result = self._run(
            workspace,
            "from arc import refute, train_samples\n"
            "refute('every output equals its input',\n"
            "       any(s['input'] != s['output'] for s in train_samples))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "[RULED OUT]" in result.stdout

        entry = json.loads(
            (workspace.root / ".athanor" / "invariants.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        assert entry["mode"] == "ruled_out"
        assert "s['input'] != s['output']" in entry["expression"]

    def test_refute_that_fails_to_rule_out_says_still_open(self, workspace):
        result = self._run(
            workspace,
            "from arc import refute\nrefute('something', False)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "[STILL OPEN]" in result.stdout

    def test_key_supersedes_across_a_reworded_claim(self, workspace):
        """Supersession keyed on the claim string fails the moment you reword."""
        result = self._run(
            workspace,
            "from arc import verify, invariants, train_samples\n"
            "verify('outputs are 3 rows tall', len(train_samples) == 99, key='height')\n"
            "verify('outputs are exactly 2 rows tall',\n"
            "       all(len(s['output']) == 2 for s in train_samples), key='height')\n"
            "print('LIVE', len(invariants()))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "LIVE 1" in result.stdout

    def test_without_a_key_a_reworded_claim_duplicates(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify, invariants, train_samples\n"
            "verify('outputs are 2 rows tall', len(train_samples) == 3)\n"
            "verify('outputs are exactly 2 rows tall', len(train_samples) == 3)\n"
            "print('LIVE', len(invariants()))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "LIVE 2" in result.stdout

    def test_retraction_removes_a_claim(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify, invariants\n"
            "verify('outputs are square', True)\n"
            "verify('outputs are square', retract=True)\n"
            "print('LIVE', len(invariants()))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "[RETRACTED]" in result.stdout
        assert "LIVE 0" in result.stdout

    def test_verify_records_a_raising_check_as_refuted(self, workspace):
        result = self._run(
            workspace,
            "import sys; sys.path.insert(0, '.')\n"
            "from arc import verify\n"
            "verify('this blows up', lambda: 1 / 0)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "[REFUTED  ]" in result.stdout
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

    def test_load_solution_returns_the_shipped_solve(self, workspace):
        """The doctrine's prediction-sanity step needs the real solution.

        Before this existed the only route was `sys.path.insert(0, 'solution')`
        — a relative path that broke arc.py's "import it the same way from
        anywhere" contract, in the one place the doctrine sends you.
        """
        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        result = self._run(
            workspace,
            "from arc import load_solution, test_samples\n"
            "solve = load_solution()\n"
            "print(solve(test_samples[0]['input']))\n",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "[[3, 0, 6], [0, 4, 0]]"

    def test_load_solution_works_from_another_directory(self, workspace):
        import subprocess
        import sys as _sys

        from conftest import MIRROR_SOLVE

        (workspace.root / "solution" / "solve.py").write_text(MIRROR_SOLVE, encoding="utf-8")
        script = workspace.root / "explore" / "fromelsewhere.py"
        script.write_text(
            "from arc import load_solution\nprint(callable(load_solution()))\n", encoding="utf-8"
        )
        result = subprocess.run(
            [_sys.executable, str(script)],
            cwd=str(workspace.root.parent),  # deliberately not the workspace
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"

    def test_solution_module_exposes_the_shipped_helpers(self, workspace):
        """A rival that shares the shipped parse needs more than `solve`.

        A solver building exactly that had to hand-roll importlib to reach the
        module's helpers — the boilerplate this toolkit promises you never need.
        """
        (workspace.root / "solution" / "solve.py").write_text(
            "def _parse(grid):\n    return len(grid)\n"
            "def solve(grid):\n    return [row[::-1] for row in grid]\n",
            encoding="utf-8",
        )
        result = self._run(
            workspace,
            "from arc import solution_module\n"
            "m = solution_module()\n"
            "print(m._parse([[1], [2]]), callable(m.solve))\n",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "2 True"

    def test_verify_flags_a_boolean_literal_comparison(self, workspace):
        """`True is (...)` slips past the constant check while being the same
        anti-pattern. A solver shipped one and caught it unaided."""
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "verify('a dressed-up assertion', True is (len(train_samples) > 0))\n",
        )
        assert result.returncode == 0, result.stderr
        assert "compile-time constant" in result.stdout

    def test_verify_does_not_flag_an_ordinary_comparison(self, workspace):
        result = self._run(
            workspace,
            "from arc import verify, train_samples\n"
            "verify('three pairs', len(train_samples) == 3)\n",
        )
        assert result.returncode == 0, result.stderr
        assert "compile-time constant" not in result.stdout

    def test_load_solution_reports_a_missing_file_clearly(self, workspace):
        result = self._run(workspace, "from arc import load_solution\nload_solution()\n")
        assert result.returncode != 0
        assert "FileNotFoundError" in result.stderr

    def test_load_solution_reports_a_missing_solve_clearly(self, workspace):
        (workspace.root / "solution" / "solve.py").write_text("x = 1\n", encoding="utf-8")
        result = self._run(workspace, "from arc import load_solution\nload_solution()\n")
        assert result.returncode != 0
        assert "does not define a callable solve" in result.stderr

    def test_toolkit_exposes_no_transformation_primitives(self, workspace):
        """Handing over rotate/flood-fill/objects would change what is measured."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("arc_probe", workspace.root / "arc.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for banned in ("rotate", "flip", "mirror", "objects", "components", "flood_fill", "crop", "tile"):
            assert not hasattr(module, banned), f"arc.py exposes a transformation primitive: {banned}"
