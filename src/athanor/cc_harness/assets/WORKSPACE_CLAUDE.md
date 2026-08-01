# Puzzle workspace — task `__TASK_ID__`

Solve one ARC-AGI-2 task in this directory. Everything you need is here; nothing
outside this directory is relevant, and the test outputs are not in it.

## Environment

__ENVIRONMENT__

`from arc import ...` works from anywhere in the workspace — from a script under
`explore/`, from a `__PYTHON__ -c` one-liner at the root, or via
`__PYTHON__ -m explore.foo`. You never need `sys.path` boilerplate.

## Layout

```
task/task.json        __N_TRAIN__ training pairs + __N_TEST__ test input(s). No test outputs.
task/grids.md         the same grids as text, one row per line
task/images/          every grid rendered as a PNG — open them with the Read tool
arc.py                observation + verification helpers (read it; it is short)
dryrun.py             free scoring of solution/solve.py against the training pairs
gate.py               the verification gate
explore/              your scratch scripts. Unlimited, free, not recorded.
solution/hypothesis.md   the rule, in prose        (you create this)
solution/solve.py        the rule, as code         (you create this)
solution/audit.md        generalization audit      (you create this, at the end)
NOTES.md              your durable research state — keep it current
.athanor/             run ledger: iterations, verified invariants, reports
```

## The loop

1. **Look.** `Read task/images/train_0.png` … and the test input(s). Then dump
   structure with a script under `explore/`. ARC is a vision task; the pictures
   and the numbers tell you different things.
2. **Verify.** Establish facts by executing them:
   `arc.verify("every output is 20x20", ...)`. Every recorded invariant narrows
   the space of rules that can be correct.
3. **Hypothesize.** Write the rule into `solution/hypothesis.md` — complete
   enough that a programmer who has never seen this puzzle could reimplement
   `solve()` from it alone.
4. **Implement and dry-run.** Write `solution/solve.py`, then run
   `__PYTHON__ dryrun.py`. It scores your `solve()` against every training pair
   and costs nothing — no iteration, no record. Iterate here until it passes.
   Every bug you catch with `dryrun.py` is a budgeted submission you keep.
5. **Submit.** `__PYTHON__ gate.py submit`. Budgeted and permanent.
6. **Reflect.** The gate's output ends with what to do next. Follow it, and
   append the reflection to `NOTES.md` — `Read` that file before your first
   `Write`, since it already exists with template content.
7. **Accept.** Once training passes, write `solution/audit.md`, then
   `__PYTHON__ gate.py accept`. That ends the run — stop working after it
   succeeds.

## Gate commands

| Command | Cost | What it does |
|---|---|---|
| `__PYTHON__ dryrun.py` | free | Scores `solution/solve.py` against the training pairs. Use it before every submission. |
| `__PYTHON__ gate.py status` | free | Distilled research state: iterations, executed ledger, last hypothesis, notes tail. **Run this first after any context compaction.** |
| `__PYTHON__ gate.py status --brief` | free | Same, minus the hypothesis dump and notes tail — iteration history and ledger only. Use this for a mid-run check. |
| `__PYTHON__ gate.py submit` | 1 iteration | Runs `solution/solve.py` against every training pair and test input, records the result, reports failures and what to do next. |
| `__PYTHON__ gate.py accept` | free | Finalizes the run, using `solution/audit.md`. It re-runs `solution/solve.py` as it stands, so adding a second candidate after a train-perfect submission costs you nothing — if the rule regresses, it falls back to what you submitted. |

## The invariant ledger

`arc.verify(claim, condition)` appends to `.athanor/invariants.jsonl`, and the
gate replays it. What is worth knowing about it:

- **The expression is recorded, not just the claim.** `verify()` reads the
  condition's source from your script, so the ledger says what was executed. A
  condition that is a compile-time constant is flagged: it measured nothing.
- **Ruling something out is a result — use `arc.refute()` for it.**
  `refute("8-connectivity explains the selection", ...)` records a dead end as a
  finding rather than leaving a `[REFUTED]` line that reads like a defect in
  your own work. Re-deriving a hypothesis you already killed is the most common
  way to burn an iteration budget.
- **Most recent entry per key wins**, and the key defaults to the claim string —
  so rewording a claim while correcting it silently creates a duplicate instead
  of superseding. Pass `key="height-relation"` when you expect to revise.
- **State only what is currently true.** A replacement prints the claim it
  displaced, so you never need to write a correction into the claim text. "The
  jog is 1 wide; it is instead always a-2 wide" leaves a live *verified*
  invariant whose own first clause is false — and after a compaction that
  sentence is what gets read back. Say `verify("the jog is always a-2 wide",
  ..., key="jog-width")` and let the ledger carry the history.
- **A changed verdict is loud, and worth stopping for.** If a claim held
  earlier and fails now, `CHANGED VERDICT` prints: whatever you built while it
  held is now suspect.
- **`gate.py status` marks dead ends `[KILLED]`, not `[OK]`.** A `refute()`
  entry that "holds" means the hypothesis is dead, so the claim beside
  `[KILLED]` is something you ruled *out*. Read the marker, not just the
  sentence.
- **A multi-line check? Pass `evidence=`.** The expression capture reads one
  expression at the call site, so a check that is really a function leaves you
  choosing between a bare name and an unreadable comprehension. Neither records
  anything useful. Instead:

  ```python
  depths = {name: depth(name) for name in rings}
  arc.verify("every surviving ring sits at even depth",
             all(d % 2 == 0 for d in survivor_depths),
             evidence=depths)
  ```

  `note=` is prose about the finding; `evidence=` is the value you measured.
- **Re-run an invariant against your own predictions with `over=`.** Write the
  predicate once and point it at different grids, rather than copying the check
  into an audit script where it can drift from what the ledger says:

  ```python
  def width_is_20(grid): return len(grid[0]) == 20

  arc.verify("every training output is 20 wide", width_is_20,
             over=[s["output"] for s in train_samples])
  ...
  arc.verify("my test prediction is 20 wide", width_is_20, over=[prediction])
  ```
- **Record from a script, not from `python -c`.** `verify()` recovers evidence
  by reading your script's source; from a `-c` one-liner or a heredoc there is
  no source to read, so the entry carries the claim and nothing about what ran,
  and the constant-condition check cannot fire at all. Such entries are marked
  `NO EVIDENCE`.
- **Sweeping many readings at once? Record it as one entry with `arc.sweep()`.**
  When you score a dozen candidate rules against training in a single pass, that
  is one finding, not a dozen — and it is usually the highest-yield thing in the
  run.

  ```python
  survivors = arc.sweep("what sets the hub centre colour?", {
      "largest blob":      fits(largest_blob),
      "longest branch":    fits(longest_branch),
      "most branch cells": fits(most_branch_cells),
  })
  ```

  Pass `(survives, detail)` instead of a bare bool to keep whatever makes the
  entry self-explaining after a compaction — the score that produced the verdict
  (`{"outer slot": (True, "3/3")}`), or, when several readings survive, what
  each one predicts out of sample (`(True, "12 vs 31 -> test top = 6")`). The
  second is usually the more useful, because a sweep with more than one survivor
  is by definition a hedging obligation and the next question is always whether
  they diverge.

  If more than one reading survives, that is a **hedging obligation** you have
  found before spending any budget: run the survivors through `arc.rival()` and
  see which diverge on the test input.
- **Retraction.** If a check was wrong — a tautology, say — withdraw it with
  `arc.verify("the claim", retract=True)`, or `arc.refute("the claim",
  retract=True)` for a dead end. It then disappears from `gate.py status`. Do this rather than leaving it: the ledger is only worth reading
  because everything in it has been executed, and one claim that merely looks
  verified devalues all of them.

## Rival readings

When you implement an alternative interpretation in order to compare it —
usually right before discarding it — register it:

```python
from arc import rival
def strict(grid): ...          # the reading you suspect is wrong
rival("diagonals may not brush a wall corner", strict)
```

`rival()` scores it against every training pair, and if `solution/solve.py`
already exists it compares predictions and tells you immediately whether the
rival diverges — that is, whether spending the second slot on it would change
anything. Register rivals **once `solve.py` exists but before you submit**:
that comparison is the useful half of the output, and without a solution to
compare against all `rival()` can say is that there is nothing to compare yet.
Everything before submission is free, so there is no cost to waiting until you
have something to hold the rival up against.

This exists because of a measured loss. A solver ruled out exactly such a rival
by taking a regularity that held across the training *outputs* and applying it
to the test *input*, concluded the ambiguity was resolved, left the second slot
empty, and missed by two cells out of four hundred. It had the rival
implemented at the time. Killing an alternative with a training pair it fails is
a proof; killing it with an out-of-sample extrapolation is not.

## Rules the gate enforces

- `solution/hypothesis.md` must exist and be at least __MIN_HYPOTHESIS_CHARS__
  characters before any submission.
- If `solution/solve.py` changed, `solution/hypothesis.md` must have changed
  too. A code change is a change of rule, implementation, or edge case — say
  which. (A refusal here does not cost an iteration.)
- `solve(grid)` receives **only** its argument. It cannot reach the puzzle data,
  and embedding a training output as a literal will be reported.
- `solve(grid)` should return one grid for training inputs; up to
  __MAX_CANDIDATES__ candidate grids are allowed for test inputs, and only for a
  real ambiguity. Returning extras on a training input is **not refused** —
  training validation simply scores the first and discards the rest — but an
  ambiguity on an example whose correct output you can *see* means the rule is
  unfinished. Resolve it rather than hedging. You do not need to restructure
  `solve()` to guarantee a single training candidate; you need the first one to
  be right.
- Budget: **__MAX_ITERATIONS__ submissions.** Over the last __BEST_EFFORT__ of
  them the train-100% requirement is lifted so you can commit to your best
  reading of the puzzle.
- `accept` requires `solution/audit.md` with a `CONFIDENCE: <1-5>` line and a
  `DECISION: ACCEPT` line.

## Working habits

- One question per exploration script, named for the question it answers.
- Shared helpers go in `explore/lib.py`; another script imports them with
  `from lib import ...` (scripts under `explore/` can import each other by bare
  module name — except when the name starts with a digit, which is not a legal
  identifier. `arc.explore_module("04_yellow_flip")` reaches those, so you never
  need `importlib` boilerplate). Guard any report in an imported script behind
  `if __name__ == "__main__":` — otherwise its output re-prints into every
  downstream run and pollutes your context.
- **`solution/solve.py` must stand alone**, so helpers you developed in
  `explore/lib.py` have to be inlined into it. That duplication can drift: an
  experiment can pass against `lib.py` while the code that actually ships says
  something subtly different. After inlining, re-check the shipped version —
  `__PYTHON__ dryrun.py` runs `solution/solve.py` itself, and
  `arc.load_solution()` hands you that same function for use in an experiment.
- To audit your own predictions, load the real solution rather than a copy:

  ```python
  from arc import load_solution, test_samples, verify
  solve = load_solution()
  prediction = solve(test_samples[0]["input"])
  ```

  `arc.solution_module()` returns the whole module when you need its helpers —
  building a rival that reuses the shipped parse rather than re-deriving it.
  **`solution/` is not a package**: `from solution.solve import _helper` raises
  `ModuleNotFoundError`, which is the natural next thought once you have inlined
  helpers into `solve.py`. Reach into it with `arc.solution_module()` instead.

- Print summaries, not grids you have already seen. A boolean, a count, or a
  set of shapes usually carries the finding.
- `arc.png()` writes relative to the current directory, so `cd` into the
  workspace first if you invoke it from elsewhere.

## Out of bounds

- No network access, and no need for it: ARC requires no external knowledge.
- Do not read anything outside this workspace. The test outputs live in the
  benchmark dataset; reaching for them invalidates the run, and the harness
  checks.
- Single agent. Do not delegate to sub-agents.

## When context gets compacted

Exploratory interpreter state and your transcript do not survive it.
`.athanor/invariants.jsonl` and `NOTES.md` do. Keep `NOTES.md` written for a
fresh version of yourself: current hypothesis, confirmed facts, refuted
hypotheses *and why*, and the next experiment you meant to run.
