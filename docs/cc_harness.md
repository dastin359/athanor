# Claude Code as Harness

A variant of the Athanor solver in which **Claude Code owns the agent loop** and
Athanor supplies only the workspace, the doctrine, and the verification gate.

Status: exploratory. Single solver agent, no reviewer.

## Contents

1. [Why](#why)
2. [What is kept, what is dropped](#what-is-kept-what-is-dropped)
3. [Mechanism mapping](#mechanism-mapping)
4. [Workspace](#workspace)
5. [The gate](#the-gate)
6. [The toolkit](#the-toolkit)
7. [Prompt composition](#prompt-composition)
8. [Running it](#running-it)
9. [Integrity and scoring](#integrity-and-scoring)
10. [Deliberate deviations](#deliberate-deviations)
11. [Open questions](#open-questions)

---

## Why

The flagship harness (`solver/orchestrator.py`, ~7k lines) implements its own
agent loop: it drives turns, dispatches tools, injects reflection prompts,
measures context pressure and compresses, and brokers an independent reflector.
Athanor's *thesis*, though, is not the loop. It is three mechanisms — code as
verification, artifact-only review, and inter-context artifact exchange.

That raises a question worth answering empirically: **how much of the result is
the thesis, and how much is the bespoke loop?** This variant is the instrument
for asking it. It keeps the thesis and throws the loop away, replacing it with a
general-purpose coding agent that already has a shell, a filesystem, and its own
context management.

The interesting constraint is that the harness loses the ability to *speak in
the conversation*. Athanor's orchestrator injects a reflection prompt after
every failed iteration; Claude Code has no such seam. So the harness speaks
where it still can — through **tool output** and through **the filesystem** — and
the design question becomes how much of an organizing principle survives that
translation.

## What is kept, what is dropped

**Kept.**

- *Code as verification.* Promoted from a habit to an API: `arc.verify(claim,
  condition)` is the only sanctioned way to establish a fact about the puzzle,
  and every call is recorded.
- *Artifact separation.* `solution/hypothesis.md` and `solution/solve.py` are
  distinct files, and the gate refuses code that changed without a
  correspondingly updated hypothesis.
- *Budgeted formal iteration.* Exploration is free and unlimited; `gate.py
  submit` is counted and permanently recorded.
- *Structured reflection.* Athanor's `TRAIN_FAILURE_REFLECTION_PROMPT` and
  `TEST_GENERALIZATION_REFLECTION_PROMPT` survive as directives appended to gate
  output, retargeted from "verbalize this" to "write this into `NOTES.md`".
- *Best-effort switch.* The train-100% requirement lifts over the trailing
  iterations, as in `BEST_EFFORT_PROMPT`.
- *ARC domain knowledge.* Shared verbatim with the flagship prompt — see
  [Prompt composition](#prompt-composition).

**Dropped.**

- *The independent reflector.* No second context reviews the artifacts. This is
  the variant's biggest known weakness: per `docs/design.md`, artifact-only
  review is the only mechanism among top ARC systems that can reject a solution
  passing training 100% on generalization grounds. Here that job falls back on
  the solver's own audit, which is exactly the false-confidence anchoring the
  reflector existed to break.
- *Agent-driven context compression.* Claude Code compacts on its own schedule
  and by its own summarizer. ICAE's *artifact* half survives (state lives on
  disk); its *active, agent-authored checkpoint* half does not.

Two compensations were added for the missing reviewer, both cheap and
mechanical rather than model-based: `solve()` runs with **no puzzle globals in
scope**, and submitted code is screened for verbatim training outputs. Neither
substitutes for review; they only close the crudest overfitting routes.

## Mechanism mapping

| Athanor orchestrator | CC harness |
|---|---|
| `run_code` / `run_code_in_previous_runtime` | `python explore/<name>.py` — a persistent filesystem instead of a persistent runtime |
| `submit_transform_hypothesis` | `solution/hypothesis.md`, enforced by the gate as a submission precondition |
| `execute_python_solution` | `python gate.py submit` |
| tool ordering forces hypothesis-before-code | gate refuses changed code with an unchanged hypothesis (costs no iteration) |
| iteration counter in orchestrator memory | `.athanor/state.json`, append-only |
| reflection prompt injected into the conversation | reflection directive appended to the gate's stdout |
| test-generalization self-audit turn | `solution/audit.md` with `CONFIDENCE` / `DECISION`, required by `gate.py accept` |
| independent reflector verdict | *(dropped)* |
| ICAE memory checkpoint on context pressure | `NOTES.md` + `.athanor/invariants.jsonl`, replayed by `gate.py status` |
| ICAE resume into a fresh context | `SessionStart(compact)` hook injects `gate.py status` output automatically |
| verified invariants inside the checkpoint schema | invariant ledger, written by `arc.verify()` at the moment of verification |
| checkpoint JSON with full history | run directory: `stream.jsonl`, `result.json`, per-iteration artifacts |

The two rows worth dwelling on:

**The harness speaks through tool output.** Every place the orchestrator would
have pushed text into the conversation, the gate appends it to the result of the
command the agent just ran. This works better than expected, because the agent
reads tool output attentively and at exactly the moment the guidance is
relevant. It is also strictly weaker in one respect: the harness can only speak
when spoken to. It cannot interrupt an agent that is drifting; it can only
respond when the agent next submits.

**The filesystem is the artifact-exchange medium.** With one agent there is no
inter-*agent* exchange left, so ICAE degenerates into inter-*context* exchange
across a compaction boundary. The `SessionStart(compact)` hook makes that
automatic: when Claude Code compacts, the hook runs `gate.py status` and its
stdout is injected as context, so the fresh window opens with the iteration
history, the verified invariants, the last hypothesis, and the tail of
`NOTES.md` already in it.

## Workspace

Each run gets a self-contained directory. The workspace *is* the harness state.

```
<out-dir>/<task_id>/
  workspace/                  the agent's cwd — contains no ground truth
    CLAUDE.md                 workspace contract: layout, loop, gate rules
    NOTES.md                  durable research state (agent-maintained)
    arc.py                    observation + verification toolkit
    gate.py                   verification gate (shim onto athanor.cc_harness.gate)
    task/task.json            train pairs + test inputs, test outputs stripped
    task/grids.md             the same grids as text
    task/images/*.png         every grid rendered — the agent Reads them as images
    explore/                  scratch scripts; unlimited, free, unrecorded
    solution/hypothesis.md    the rule, in prose
    solution/solve.py         the rule, as code
    solution/audit.md         generalization audit
    .claude/settings.json     SessionStart(compact) hook
    .athanor/state.json       iteration ledger + embedded config
    .athanor/invariants.jsonl  verified invariants, appended by arc.verify()
    .athanor/iterations/<n>/  archived hypothesis, code, report, predictions
    .athanor/final.json       accepted solution
  system_prompt.md            what the agent is given as doctrine
  initial_prompt.md           the opening message
  stream.jsonl                full Claude Code stream-json transcript
  run.log                     stderr
  result.json                 harness record: config, ledger, cost, score, integrity
```

Grids are rendered to PNG so the agent can open them with the `Read` tool. This
is the CC-native rendering of the flagship's `use_visual_mode`, and it is
arguably better: images arrive on demand rather than occupying context from
turn one.

## The gate

`python gate.py {status,submit,accept}` is the only enforcement point.

**`submit`** — one budgeted iteration. Preconditions, each of which refuses
*without* consuming an iteration:

- `solution/hypothesis.md` exists and clears a character floor (a crude but
  effective stand-in for Athanor's "be exhaustive" instruction).
- `solution/solve.py` exists and defines `solve`.
- If the code changed, the hypothesis changed too.
- Resubmitting byte-identical artifacts is refused.
- The iteration budget is not exhausted.

Then `solve.py` is executed **in a fresh interpreter** (`evaluate.run_solution_
isolated`) against every training pair and test input, bounded by a timeout. The
gate owns the ledger; model-authored code that hangs, calls `sys.exit`, or
corrupts interpreter state must not be able to take that ledger with it.

The report that comes back carries the failure diff and then the reflection
directive. On a training pass, it carries the generalization-audit directive
instead.

**`accept`** — requires the last submission to be train-perfect (or the run to
be inside the best-effort window), plus `solution/audit.md` with an explicit
`CONFIDENCE: <1-5>` and `DECISION: ACCEPT`. A self-declared `DECISION: RETRY`
is refused, deliberately: the agent's own stated concerns are the most
informative signal available in a configuration with no reviewer.

**`status`** — the distilled research state. The ICAE resume artifact.

## Where the harness actually speaks

A finding from running this, not from designing it.

The flagship's reflection prompt fires on a failed `execute_python_solution`,
and it fires often, because in that harness the tool call **is** the only way to
score a candidate against training. Reflection-on-failure is therefore the main
channel through which the orchestrator shapes the search.

This variant added `dryrun.py` so the doctrine's "never use the gate as a
debugger" had a cheap alternative. That turned out to relocate the harness's
leverage entirely. `dryrun.py` loads `solution/solve.py` in a fresh module and
the gate runs it in a fresh interpreter, so the two agree on everything —
including the one case worth worrying about, where the code depends on a name
that only existed in an exploration session. Both report the same
`NameError`.

The consequence: **a solver that follows the contract essentially never submits
a failing solution.** Across four rounds and thirteen test examples, every
accepted run passed training on iteration 1 or 2, no submission has ever failed,
and no gate refusal has ever fired. The elaborate failure report — diff
listings, bounding boxes, expected-vs-predicted grids, the reflection directive
— is well-formed and almost entirely unreached.

That is the doctrine working, not a defect. But it means the design instinct
carried over from the flagship was wrong about *where* the words matter. Every
measured effect in this variant has come from the **acceptance** path:

- the generalization audit is what prompted the ablation batteries and
  equivariance tests that agents repeatedly credited with changing their answer;
- the candidate-budget paragraph changed a solver's behaviour on the very next
  run, by its own account;
- the rival check and the mechanical signals both live there.

So the harness's real surface is not "what to say when the solver is wrong" but
"what to say when the solver believes it is right". Effort spent on the failure
path is effort spent on a branch the doctrine is designed to avoid.

## The toolkit

`arc.py` deliberately contains **no transformation primitives** — no rotate, no
flood-fill, no connected components. Building those is the reasoning work, and
shipping them would change what is being measured. There is a test that fails if
one appears.

What it does contain:

- `verify(claim, condition)` — the core. Prints `[VERIFIED]`/`[REFUTED]`,
  appends to the invariant ledger, returns the boolean. A `condition` that
  raises is recorded as a refutation rather than crashing the script.
- `check(solve_fn)` — free dry-run against every training pair. This exists so
  an iteration is never spent on a bug that a dry run would have caught. The
  gate is not a debugger.
- `show`, `diff`, `shape`, `colors`, `histogram` — observation.
- `png(grid, path)` — render a prediction so it can be looked at as a picture.

## Prompt composition

The system prompt is assembled from two sources:

- **Sections 1–2 of `solver/SOLVER_SYSTEM_PROMPT.md`, verbatim** — role, ARC
  priors, and the transformation-primitive taxonomy. Shared with the flagship on
  purpose: comparing two harnesses is only meaningful if the domain knowledge is
  held constant, so that a measured difference is attributable to the harness
  rather than to the priors. `tests/test_prompt.py` fails loudly if those
  sections are renamed or removed, rather than letting the two drift apart
  silently.
- **`cc_harness/assets/CC_SOLVER_DOCTRINE.md`** — the goal, and the
  code-as-verification methodology retargeted at a shell and a filesystem.

The result is task-independent, so a batch shares the provider-side prompt
cache the way the flagship's cached system prompt does. Per-run specifics live
in the workspace `CLAUDE.md` instead.

## Running it

```bash
export ARC_DATA_ROOT=/path/to/ARC-AGI-2

# one task
athanor cc run 28a6681f --max-iterations 8

# a batch, then the summary
athanor cc batch --tasks 28a6681f 0934a4d8 --out-dir cc_runs
athanor cc score cc_runs

# prepare a workspace without launching anything
athanor cc workspace 28a6681f --out-dir cc_runs

# reconstruct what an agent actually did
athanor cc trace cc_runs/28a6681f

# pick up a run that died before accepting
athanor cc resume cc_runs/28a6681f
```

`resume` exists because over a large batch individual runs die — a timeout, a
rate limit, a killed process — and the workspace already holds everything needed
to continue. The resumed agent inherits the iteration ledger, the verified
invariants and `NOTES.md`, spends only the budget that is left, and is told
explicitly what did *not* survive so it does not assume an unrecorded conclusion
still holds.

`athanor cc run` launches `claude -p` with `--output-format stream-json` inside
the workspace, restricted to `Bash,Read,Write,Edit,Glob,Grep,TodoWrite`.
`WebSearch`, `WebFetch`, `Task`, and `Agent` are explicitly denied: ARC requires
no external knowledge, and this variant is single-agent by definition.

Flags that move between Claude Code releases (`--effort`, `--tools`,
`--append-system-prompt-file`, `--setting-sources`, `--strict-mcp-config`,
`--max-budget-usd`, `--bare`) are probed against `claude --help` and dropped
when unavailable, so the harness degrades instead of failing on an unknown
option.

**Two launch modes.** `athanor cc run` starts a subprocess. `athanor cc
workspace` prepares everything and starts nothing, so an already-running Claude
Code session can drive the same workspace through a sub-agent. The workspace
contract is identical either way; only the delivery of the doctrine differs
(system prompt vs. a file the agent is told to read first).

Note that `--max-budget-usd` is a real ceiling only under API-key billing. On a
subscription it has nothing to meter, so it defaults to `None`.

### Operational notes

Validated against Claude Code 2.1.220:

- **Permission mode.** The default is `acceptEdits`. Not `bypassPermissions`:
  Claude Code maps that to `--dangerously-skip-permissions`, which the CLI
  refuses outright when running as root — the normal case for a containerised
  harness — and the refusal arrives as an empty stream with one line of stderr.
  A `bypassPermissions` config is swapped for `acceptEdits` automatically when
  the harness detects root. `dontAsk` also works and is stricter: it denies
  anything not allow-listed rather than prompting.
- **Runtime libraries.** The workspace probes its interpreter at build time and
  writes what it found into `CLAUDE.md`. If the solver's `python` is not the one
  that installed Athanor's dependencies, the contract says so rather than
  letting the agent discover it mid-experiment.
- **Compaction hook.** `.claude/settings.json` registers a `SessionStart`
  matcher on `compact`, whose stdout Claude Code injects as context. Settings
  that fail validation are silently ignored in `-p` mode, so the hook is a
  best-effort accelerator: `CLAUDE.md` independently instructs the agent to run
  `gate.py status` after any compaction, and the run is correct either way.
- **No result message.** If the CLI exits without one, the runner records the
  stderr tail in `result.json` rather than reporting an empty run.

## Integrity and scoring

Ground truth never enters the workspace: `task.json` is written with test
outputs stripped, and the harness holds the expected outputs for post-run
scoring only. `ARC_DATA_ROOT` is removed from the solver's environment.

`scoring.contamination_scan` then checks the obvious routes — a transcript
referencing the dataset root or an out-of-workspace task file, a `WebFetch`/
`WebSearch` call, test outputs appearing in `task.json` — and records the
findings in `result.json`. It is advisory, not authoritative; it exists so the
harness can say what it checked.

Scoring follows ARC-AGI-2: each test example counts as solved when any submitted
candidate matches exactly, and the task score is the fraction solved.

## Deliberate deviations

Three places where this variant does not mirror the flagship, each on purpose:

1. **`solve()` gets no puzzle globals.** The flagship runs final solution code in
   the exploratory context, leaving `train_samples`/`test_samples` reachable
   from `solve()`; its reviewer catches the resulting lookup tables. With no
   reviewer, the contract is tightened instead: `solve(grid)` sees only its
   argument.
2. **Grids are presented as digit rows, not Python list literals.** More legible
   spatially and cheaper in tokens. It does change perception, and it is a
   harness difference, which is the point of the variant — but it is a
   confound to keep in mind when comparing scores.
3. **`arc.check()` exists.** The flagship has no free dry-run; agents improvise
   one inside `run_code`. Making it a named helper reduces wasted iterations and
   pushes further in the direction of the thesis.

## Open questions

The variant exists to answer these. See
[cc_harness_results.md](cc_harness_results.md) for the running experiment log;
partial answers so far are noted inline below.

- **Does the doctrine survive without the loop?** The flagship *forces* the
  hypothesis-then-code ordering through tool sequencing. Here it is a gate
  refusal after the fact. Does the agent internalise the discipline, or does it
  learn to satisfy the check?

  *Early evidence: internalised.* Across six tasks, agents established 7–29
  invariants before their first submission and never once tripped the ordering
  refusal. Each independently named an experiment that changed its answer and
  that it would not have run unprompted. But the sample is small, the tasks were
  not randomly chosen, and every run was driven as a sub-agent rather than
  through the subprocess launcher.

- **What does losing the reviewer cost?** `docs/design.md` claims artifact-only
  review is the only mechanism that can reject a train-perfect but overfit
  solution. The self-audit is a weaker substitute by construction. The gap
  should be measurable on the hard-pair frontier in `RESULTS.md`.

  *Partly answered, and the answer was not what the question assumed.* The cost
  showed up on `88e364bc`: train-perfect, wrong by 2 cells out of 400, accepted
  at confidence 4. But the failure was not overfitting in the sense the question
  anticipated. The solver had the correct rival reading implemented, and killed
  it by taking a regularity that held across the training *outputs* and applying
  it to the test *input*. That is an inductive leap wearing a proof's clothes,
  and it is *more* dangerous under this doctrine than under prose reasoning,
  because it arrives with a `[REFUTED]` line in the ledger and reads as settled.
  Executed verification compresses the cost of checking; it does not turn
  induction into deduction.

  The mechanical signals in `signals.py` cannot reach that class of error —
  shape, palette and structure were all consistent, and they correctly fired
  nothing. What *did* recover it is cheaper than a reviewer and narrower:
  `arc.rival()` plus the gate reporting a registered rival that fits every
  training pair and disagrees with the submission. On `abc82100` — a zero-solve
  frontier pair — the registered rival was the correct answer and the task
  scored on candidate 2. Second-attempt usage across the experiment moved from
  2/11 test examples to 5/8.

  So the reviewer-shaped hole is real but partly fillable without a reviewer:
  the recoverable part is *ambiguity the solver already noticed and then argued
  away*, and the fix is to make the arguing-away visible rather than to add a
  second opinion. What remains unfilled is genuine misreading that the solver
  never entertained an alternative to.
- **Is compaction a real regression?** ICAE compresses on measured context
  pressure with a purpose-built schema; Claude Code compacts on its own schedule
  with a general summarizer. The disk-backed state is meant to cover the
  difference. Whether it does is an empirical question.

  *Untested directly — no run has been compacted — but the state that would
  survive one has been inspected, and it degrades in ways worth knowing.* Three
  separate failures of the same kind turned up: a tautology recorded as a
  verified invariant; an invariant whose recorded evidence was the bare name
  `allok`; and `NOTES.md` still holding its seeded template after seven
  exploration scripts and six invariants. Each looks recorded and carries no
  information, which is worse than an empty ledger because it reads as
  established fact to the context that inherits it. All three now warn at the
  point of the mistake. The pattern is consistent enough to state as a design
  rule: **durability machinery degrades silently, so the checks that guard it
  earn their cost.**
- **Where does the cost land?** The flagship's headline is $3.12/task. Claude
  Code carries a larger default system prompt and a general-purpose tool
  surface, but avoids re-sending puzzle data on every turn and does not pay for
  a second reviewer context. The sign of the net effect is not obvious.
