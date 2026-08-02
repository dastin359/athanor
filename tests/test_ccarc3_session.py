"""Tests for CCARC3 workspace construction and outcome collection.

All offline. The one thing worth stating outright: ``collect_outcome`` reads the
ledger, never the solver's own account of how it did. On ARC-AGI-2 the gate was
what made a claimed solve and a real one the same thing; here the trace plays
that role, so it must be the only thing consulted.
"""

from __future__ import annotations

import json

import pytest

from athanor.ccarc3 import Ccarc3Config, GameInfo, build_workspace
from athanor.ccarc3.session import DEFAULT_EFFORT, DEFAULT_MODEL, build_cli_args, collect_outcome

INFO = GameInfo("ls20-test", "LS20", ("keyboard",), (22, 123, 73))


@pytest.fixture
def ws(tmp_path):
    return build_workspace(Ccarc3Config("ls20-test", out_dir=tmp_path), INFO)


def test_workspace_has_everything_the_solver_needs(ws):
    names = {p.name for p in ws.root.iterdir()}
    assert {"CLAUDE.md", "DOCTRINE.md", "session.py", "meta.json", "notes"} <= names


def test_the_action_budget_is_derived_from_the_game_not_guessed(ws):
    meta = json.loads((ws.root / "meta.json").read_text())
    assert meta["action_budget"] == int(218 * 4.0)
    assert str(meta["action_budget"]) in (ws.root / "CLAUDE.md").read_text()


def test_a_short_game_still_gets_a_workable_floor(tmp_path):
    tiny = GameInfo("t", baseline_actions=(3, 4))
    w = build_workspace(Ccarc3Config("t", out_dir=tmp_path), tiny)
    assert json.loads((w.root / "meta.json").read_text())["action_budget"] == 200


def test_the_generated_session_is_prewired_to_this_game(ws):
    src = (ws.root / "session.py").read_text()
    assert "'ls20-test'" in src or '"ls20-test"' in src
    assert "LevelGate" in src and "ArcClient" in src
    assert "(22, 123, 73)" in src, "baselines must reach the solver"


def test_the_generated_files_are_valid_after_template_substitution(ws):
    """Both files come out of `.format()`, so a literal brace must be doubled.

    Getting that wrong does not fail loudly -- `session.py` stops importing, or
    `CLAUDE.md` ships `{{'ACTION6': (0, 31)}}` as the solver's worked example.
    Every run built from the template is affected and no unit test of the
    modules themselves would notice.
    """
    compile((ws.root / "session.py").read_text(), "session.py", "exec")
    md = (ws.root / "CLAUDE.md").read_text()
    assert "{{" not in md and "}}" not in md
    assert "{'ACTION6': (0, 31)}" in md, "the dict example must survive as a dict"


def test_the_worked_examples_name_things_that_exist(ws):
    """A CLAUDE.md example is API documentation the solver will run verbatim."""
    import re

    from athanor.ccarc3 import ArcClient
    from athanor.ccarc3 import __all__ as exported

    text = (ws.root / "CLAUDE.md").read_text() + (ws.root / "DOCTRINE.md").read_text()
    for attr in set(re.findall(r"\bclient\.(\w+)\(", text)):
        assert hasattr(ArcClient, attr), f"CLAUDE.md calls client.{attr}(), which does not exist"
    for name in set(re.findall(r"\barc\.(\w+)\(", text)):
        assert name in exported, f"CLAUDE.md calls arc.{name}(), which is not exported"


def test_the_worked_examples_do_not_reference_undefined_variables(ws):
    """The bug the test above was written for, which it could not actually catch.

    It validated the *callee* names in `client.X(` / `arc.X(` and nothing else.
    The shipped defect was `arc.level_pace(client.transitions(), baselines)` --
    a real function, called with an **argument** that exists nowhere in a
    solver's namespace. Checking names alone passes that happily.

    This parses every python block and resolves each free variable against what
    a solver actually has: the three names `session` provides, plus builtins,
    plus anything the block defines itself.
    """
    import ast
    import builtins
    import re

    provided = {"client", "gate", "arc", "INFO", "ACTION_BUDGET", "HERE"}
    # Names the reader is explicitly asked to supply -- a forward model and the
    # endpoints to route between. Every other free name is a defect, and this
    # set is closed on purpose: a new placeholder has to be justified here.
    placeholders = {"step", "start", "goal"}
    text = (ws.root / "CLAUDE.md").read_text() + "\n" + (ws.root / "DOCTRINE.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert blocks, "the guide is supposed to contain worked examples"

    unknown = []
    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue                      # a fragment, not a runnable example
        bound = set(provided) | set(dir(builtins))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, (ast.comprehension,)):
                for name in ast.walk(node.target):
                    if isinstance(name, ast.Name):
                        bound.add(name.id)
            elif isinstance(node, (ast.FunctionDef, ast.Lambda)):
                bound |= {a.arg for a in node.args.args}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in bound and node.id not in placeholders:
                    unknown.append(node.id)
    assert not unknown, (
        f"worked examples reference names a solver does not have: {sorted(set(unknown))}"
    )


def test_the_workspace_puts_athanor_on_the_path(ws):
    assert "athanor" in ws.env["PYTHONPATH"] or ws.env["PYTHONPATH"].endswith("src")


def test_doctrine_carries_the_findings_that_contradict_instinct(ws):
    doctrine = (ws.root / "DOCTRINE.md").read_text().lower()
    assert "dying is cheap" in doctrine
    assert "never make reset your first action after completing a level" in doctrine
    assert "not refutation unless the belief was applicable" in doctrine
    # The three-valued distinction must reach the surface solvers actually
    # touch: no run has ever constructed a Rule, but gate.acknowledge() was
    # called 21 times across five runs.
    assert "refuted=" in doctrine and "untested=" in doctrine


def test_the_run_defaults_are_the_ones_the_project_requires():
    """Opus 5 at high effort, so runs stay comparable across the project."""
    assert DEFAULT_MODEL == "claude-opus-5"
    assert DEFAULT_EFFORT == "high"
    cfg = Ccarc3Config("g")
    assert cfg.model == DEFAULT_MODEL and cfg.effort == DEFAULT_EFFORT


def test_cli_args_carry_model_and_prompt(ws):
    args = build_cli_args(ws)
    assert "--model" in args and DEFAULT_MODEL in args
    assert "-p" in args
    assert "--output-format" in args and "stream-json" in args


# --------------------------------------------------------------------------- #
# outcome collection
# --------------------------------------------------------------------------- #


def _trace(ws, rows):
    with ws.trace_path.open("w", encoding="utf-8") as fh:
        for i, (action, level, state, frames) in enumerate(rows):
            fh.write(json.dumps({
                "i": i, "level": level, "action": action, "params": {},
                "frames": frames, "score": level, "state": state,
                "full_reset": False, "available_actions": ["ACTION1"],
            }) + "\n")


def test_outcome_is_read_from_the_ledger_not_from_any_claim(ws):
    _trace(ws, [
        ("RESET", 0, "NOT_FINISHED", [[[1]]]),
        ("ACTION1", 0, "NOT_FINISHED", [[[2]]]),
        ("ACTION1", 1, "NOT_FINISHED", [[[3]]]),
    ])
    out = collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["levels_reached"] == 1
    assert out["levels_total"] == 3
    assert out["actions_used"] == 3
    assert out["won"] is False
    assert json.loads((ws.root / "result.json").read_text())["levels_reached"] == 1


def test_a_win_is_detected_from_the_state(ws):
    _trace(ws, [("RESET", 0, "NOT_FINISHED", [[[1]]]), ("ACTION1", 3, "WIN", [[[2]]])])
    assert collect_outcome(ws, exit_code=0, timed_out=False)["won"] is True


def test_deaths_count_episodes_not_frames(ws):
    """Three frames while dead is one death plus two wasted actions."""
    _trace(ws, [
        ("RESET", 0, "NOT_FINISHED", [[[1]]]),
        ("ACTION1", 0, "GAME_OVER", [[[2]]]),
        ("ACTION1", 0, "GAME_OVER", []),
        ("ACTION1", 0, "GAME_OVER", []),
    ])
    out = collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["deaths"] == 1
    assert out["wasted_actions"] == 2


def test_a_full_reset_splits_the_trace_into_playthroughs(ws):
    """`ls20` reported 860 actions for a game won in 490 of them.

    The other 370 bought progress a full reset then discarded, so charging them
    to the result overstates its cost by whatever was replayed. That correction
    was made by hand in the write-up; here it comes off the ledger. Both figures
    are reported — the total is what the budget paid.
    """
    _trace(ws, [
        ("RESET", 0, "NOT_FINISHED", [[[1]]]),
        ("ACTION1", 1, "NOT_FINISHED", [[[2]]]),
        ("ACTION1", 2, "NOT_FINISHED", [[[6]]]),    # the discarded run got to 2
        ("RESET", 0, "NOT_FINISHED", [[[3]]]),      # level went down: full reset
        ("ACTION1", 1, "NOT_FINISHED", [[[4]]]),
    ])
    out = collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["actions_used"] == 5, "the budget paid for all of them"
    assert out["playthroughs"] == 2
    assert out["actions_final_playthrough"] == 2, "the restart itself was billed"
    # The surviving playthrough peaks *below* the discarded one on purpose. With
    # both peaking at the same level this assertion passed under a whole-trace
    # max too, and could not detect the scoping it exists to check.
    assert out["levels_reached"] == 2, "the best ever reached"
    assert out["levels_reached_final_playthrough"] == 1, "what actually survived"


def test_a_run_with_no_full_reset_reports_one_playthrough(ws):
    """The common case must not acquire a second, confusing number."""
    _trace(ws, [
        ("RESET", 0, "NOT_FINISHED", [[[1]]]),
        ("ACTION1", 1, "NOT_FINISHED", [[[2]]]),
    ])
    out = collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["playthroughs"] == 1
    assert out["actions_final_playthrough"] == out["actions_used"] == 2


def test_turns_and_cost_are_read_from_the_stream(ws):
    """The ledger says what a run did; only the stream says what it cost."""
    from athanor.ccarc3.session import run_cost

    (ws.root / "stream.jsonl").write_text(
        json.dumps({"type": "assistant"}) + "\n"
        + json.dumps({"type": "result", "num_turns": 36,
                      "total_cost_usd": 3.0381, "duration_ms": 732994}) + "\n"
    )
    assert run_cost(ws.root / "stream.jsonl") == {
        "turns": 36, "cost_usd": 3.0381, "duration_s": 733, "attempts": 1,
    }


def test_cost_sums_every_attempt_of_a_resumed_run(ws):
    """`run_game` archives the previous stream on resume while the trace carries
    across, so reading only stream.jsonl charged a resumed run's full action
    count against the last attempt's bill alone."""
    from athanor.ccarc3.session import run_cost

    def stream(path, turns, cost, ms):
        path.write_text(json.dumps({
            "type": "result", "num_turns": turns,
            "total_cost_usd": cost, "duration_ms": ms}) + "\n")

    stream(ws.root / "stream.1.jsonl", 40, 5.00, 600_000)   # the archived attempt
    stream(ws.root / "stream.jsonl", 36, 3.00, 400_000)     # the one that finished
    assert run_cost(ws.root / "stream.jsonl") == {
        "turns": 76, "cost_usd": 8.00, "duration_s": 1000, "attempts": 2,
    }


def test_a_non_object_line_in_the_stream_is_not_a_crash(ws):
    """`run_game` merges the solver's stderr into stream.jsonl, so a line can be
    valid JSON without being an object."""
    from athanor.ccarc3.session import run_cost

    (ws.root / "stream.jsonl").write_text(
        '"a bare string mentioning \\"type\\":\\"result\\""\n'
        + json.dumps({"type": "result", "num_turns": 5,
                      "total_cost_usd": 1.0, "duration_ms": 1000}) + "\n"
    )
    assert run_cost(ws.root / "stream.jsonl")["turns"] == 5


def test_a_run_with_no_stream_reports_no_cost_rather_than_failing(ws):
    """A killed run has no result event, and that is not an error."""
    from athanor.ccarc3.session import run_cost

    assert run_cost(ws.root / "stream.jsonl") == {}
    (ws.root / "stream.jsonl").write_text(json.dumps({"type": "assistant"}) + "\n")
    assert run_cost(ws.root / "stream.jsonl") == {}


def test_an_empty_run_collects_without_crashing(ws):
    out = collect_outcome(ws, exit_code=1, timed_out=True)
    assert out["actions_used"] == 0 and out["timed_out"] is True


def test_the_rule_book_is_summarised_when_present(ws):
    _trace(ws, [("RESET", 0, "NOT_FINISHED", [[[1]]])])
    ws.rules_path.write_text(json.dumps({
        "verified": [{"rule": "a"}, {"rule": "b"}],
        "refuted": [{"rule": "c"}],
        "open_questions": [],
    }))
    out = collect_outcome(ws, exit_code=0, timed_out=False)
    assert out["mechanics_recorded"] == 2
    assert out["refutations_recorded"] == 1


def test_bypass_permissions_is_downgraded_under_root(ws, monkeypatch):
    """--dangerously-skip-permissions is refused as root; the run dies empty."""
    import athanor.ccarc3.session as sess

    monkeypatch.setattr(sess, "resolve_permission_mode", lambda m: "acceptEdits")
    args = sess.build_cli_args(ws)
    assert "acceptEdits" in args
    assert "bypassPermissions" not in args


def test_bash_is_pre_approved_or_the_run_produces_nothing(ws):
    """acceptEdits grants writes but not Bash; CCARC burned two runs on this."""
    args = build_cli_args(ws)
    assert "--allowedTools" in args
    allowed = args[args.index("--allowedTools") + 1].split(",")
    assert "Bash" in allowed and "Read" in allowed and "Write" in allowed


def test_network_tools_are_denied_for_benchmark_integrity(ws):
    args = build_cli_args(ws)
    assert "--disallowed-tools" in args
    denied = args[args.index("--disallowed-tools") + 1]
    assert "WebSearch" in denied or "WebFetch" in denied


def test_the_doctrine_asset_is_declared_as_package_data():
    """Without this, build_workspace raises FileNotFoundError on an installed
    (non-editable) copy -- and only there, so tests would never catch it."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    patterns = data["tool"]["setuptools"]["package-data"]["athanor.ccarc3"]
    assert any(p.endswith(".md") for p in patterns)


def test_the_doctrine_asset_actually_exists_where_the_code_looks():
    from athanor.ccarc3.session import ASSETS

    assert (ASSETS / "CCARC3_DOCTRINE.md").is_file()


# --------------------------------------------------------------------------- #
# resume vs fresh — containers get recycled mid-run
# --------------------------------------------------------------------------- #


def _seed(root):
    (root / "trace.jsonl").write_text(json.dumps({
        "i": 0, "level": 0, "action": "RESET", "params": {}, "frames": [[[1]]],
        "score": 0, "state": "NOT_FINISHED", "full_reset": False,
        "available_actions": [],
    }) + "\n")
    (root / "trace.state.json").write_text('{"game_id":"ls20-test","actions_used":1}')
    (root / "rules.json").write_text('{"verified":[{"rule":"a"}],"refuted":[],"open_questions":[]}')


def test_a_rebuilt_workspace_resumes_by_default(tmp_path):
    cfg = Ccarc3Config("ls20-test", out_dir=tmp_path)
    ws = build_workspace(cfg, INFO)
    _seed(ws.root)

    again = build_workspace(cfg, INFO)
    assert again.resumed
    assert again.trace_path.read_text().strip(), "the trace must survive"
    assert again.rules_path.exists(), "so must what the earlier session learned"


def test_the_resumed_prompt_tells_the_solver_not_to_start_over(tmp_path):
    cfg = Ccarc3Config("ls20-test", out_dir=tmp_path)
    _seed(build_workspace(cfg, INFO).root)
    prompt = build_workspace(cfg, INFO).initial_prompt
    assert "resuming" in prompt.lower()
    assert "rules.json" in prompt
    assert "reset" in prompt.lower()


def test_fresh_discards_the_previous_run(tmp_path):
    cfg = Ccarc3Config("ls20-test", out_dir=tmp_path)
    _seed(build_workspace(cfg, INFO).root)

    ws = build_workspace(Ccarc3Config("ls20-test", out_dir=tmp_path, fresh=True), INFO)
    assert not ws.resumed
    assert not ws.trace_path.exists()
    assert not ws.rules_path.exists()
    assert "resuming" not in ws.initial_prompt.lower()


def test_a_first_run_is_not_reported_as_resumed(tmp_path):
    ws = build_workspace(Ccarc3Config("ls20-test", out_dir=tmp_path), INFO)
    assert not ws.resumed


def test_resuming_with_a_bigger_budget_keeps_the_game(tmp_path):
    """The natural move when a first run hits its cap partway through.

    Rebuilding with a larger multiple must raise the cap in the generated
    session.py while leaving the trace, state and rule book alone -- otherwise
    the only way to buy more actions is to throw away the ones already paid for.
    """
    small = Ccarc3Config("ls20-test", out_dir=tmp_path, budget_multiple=0.5)
    ws = build_workspace(small, INFO)
    _seed(ws.root)
    first_cap = json.loads((ws.root / "meta.json").read_text())["action_budget"]

    big = Ccarc3Config("ls20-test", out_dir=tmp_path, budget_multiple=4.0)
    ws2 = build_workspace(big, INFO)
    second_cap = json.loads((ws2.root / "meta.json").read_text())["action_budget"]

    assert second_cap > first_cap
    assert f"max_actions={second_cap}" in (ws2.root / "session.py").read_text()
    assert ws2.resumed
    assert ws2.trace_path.read_text().strip(), "the actions already paid for survive"
    assert ws2.rules_path.exists(), "so does what was learned"


def test_the_workspace_puts_an_interpreter_with_numpy_first_on_path(ws):
    """The first live run burned turns on ModuleNotFoundError from bare python3."""
    import shutil

    first = ws.env["PATH"].split(":")[0]
    python = shutil.which("python3", path=first)
    assert python, f"no python3 in the first PATH entry: {first}"
    import subprocess
    r = subprocess.run([python, "-c", "import numpy"], capture_output=True)
    assert r.returncode == 0, "the default interpreter must have numpy"


def test_a_stale_result_is_cleared_when_a_run_starts(tmp_path):
    """result.json describes a finished run. A resume that leaves the previous
    one in place makes `report` present a stale outcome as final, and makes a
    watcher see a not-yet-started run as already complete."""
    cfg = Ccarc3Config("ls20-test", out_dir=tmp_path)
    ws = build_workspace(cfg, INFO)
    (ws.root / "result.json").write_text(json.dumps({"levels_reached": 99}))
    _seed(ws.root)

    again = build_workspace(cfg, INFO)
    assert not (again.root / "result.json").exists()
    assert again.resumed, "clearing the result must not discard the run itself"
    assert again.trace_path.read_text().strip()


def test_a_resume_does_not_destroy_the_previous_stream(tmp_path, monkeypatch):
    """run_game opened stream.jsonl with "w". When the ls20 resume started it
    truncated run 1's stream -- the only record of how the handoff went, which
    was exactly what needed diagnosing when the resume replayed the game."""
    import athanor.ccarc3.session as sess

    cfg = Ccarc3Config("ls20-test", out_dir=tmp_path)
    ws = build_workspace(cfg, INFO)
    _seed(ws.root)
    (ws.root / "stream.jsonl").write_text('{"run":1}\n')

    monkeypatch.setattr(sess, "build_cli_args", lambda w, **k: ["true"])
    sess.run_game(Ccarc3Config("ls20-test", out_dir=tmp_path), INFO)

    archived = list(ws.root.glob("stream.*.jsonl"))
    assert archived, "the previous stream must be kept, not overwritten"
    assert '{"run":1}' in archived[0].read_text()


def test_a_resume_records_what_it_inherited(tmp_path, monkeypatch):
    """Nothing captured what the client restored, so when a resume preserved the
    ledger but not the game it could not be reconstructed afterwards."""
    import athanor.ccarc3.session as sess

    cfg = Ccarc3Config("ls20-test", out_dir=tmp_path)
    ws = build_workspace(cfg, INFO)
    _seed(ws.root)

    monkeypatch.setattr(sess, "build_cli_args", lambda w, **k: ["true"])
    sess.run_game(Ccarc3Config("ls20-test", out_dir=tmp_path), INFO)

    snap = json.loads((ws.root / "resume_state.json").read_text())
    assert snap["resumed"] is True
    assert snap["state_file_present"] is True
    assert snap["trace_lines"] == 1


def test_the_doctrine_states_the_objective_function(ws):
    """Every run before this optimised blind.

    The doctrine described budgets and baselines but never said how a run is
    scored, so a solver had to infer "fewer actions is better" — which is true
    only up to the 1.15 cap, and misses that later levels weigh more and that an
    unfinished level scores nothing at all.
    """
    doctrine = (ws.root / "DOCTRINE.md").read_text()
    assert "min(1.15, (h / a) ** 2)" in doctrine, "the actual formula"
    assert "squared" in doctrine.lower()
    assert "Later levels are worth more" in doctrine
    assert "unfinished level scores 0" in doctrine or "scores 0" in doctrine
