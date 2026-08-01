# CC Harness — experiment log

Running record of the Claude Code harness variant on ARC-AGI-2 public eval.

Each round runs several solver agents on different tasks, then reads the traces
(`athanor cc trace`) and the agents' own friction reports to find gaps in the
*infrastructure*, not the puzzles. Fixes ship before the next round.

Method note: every agent is asked to critique the harness as explicitly as it
reports its solution. That has been the highest-yield part of the loop — a
solver that has just spent twenty minutes inside the workspace knows things
about it that no amount of reading the code reveals.

---

## Round 1 — baseline

Three agents (Opus 5), 8-iteration budget, tasks chosen for size and shape
variety rather than difficulty.

| task | solved | test examples | iterations | confidence | explore scripts | invariants | verified before 1st submit | elapsed |
|---|---|---|---|---|---|---|---|---|
| `28a6681f` | yes | 1/1 | 1/8 | 4 | 12 | 29 | 23 | 737s |
| `b0039139` | yes | 2/2 | 2/8 | 5 | 5 | 17 | 10 | 454s |
| `89565ca0` | yes | 1/1 | 1/8 | 4 | 9 | 14 | 12 | 739s |

**3/3 solved.** All integrity-clean. No gate refusals. Budget never approached.

Token cost, measured as sub-agent tokens per task: 81k, 100k, 107k — mean ~96k.
Treat that as an order-of-magnitude figure, not a comparable one: these runs were
driven as sub-agents inside an existing session rather than through
`athanor cc run`, so they carry that session's prompt-cache behaviour and none
of the batch amortisation the flagship's $3.12/task figure assumes. The
comparable number has to come from the subprocess launcher over a real batch.

### What the traces showed

The doctrine was followed rather than merely satisfied. Every agent front-loaded
verification — 10 to 23 invariants established *before* the first submission —
and two of the three passed training on their first formal attempt. That is the
intended shape: exploration is free, submission is scarce, so the work moves
into exploration.

More telling, all three independently reported that the discipline changed what
they actually did, and each named a specific place where it mattered:

- On `28a6681f`, the agent wrote an independent time-stepped simulation to
  cross-check its own rule. It *refuted* — which isolated the exact semantic
  commitment the rule needed (grains settle serially, not concurrently). Its
  own summary: "I would never have articulated that from staring at grids."
- On `b0039139`, a colour-permutation equivariance test found a hard-coded
  separator colour that all four training pairs happened to agree on. The
  predictions did not change, but an incidental constant became a role-based
  one.
- On `89565ca0`, an occlusion-reconstruction check found a lattice bridged
  across another shape's line — a case the agent said hand-counting from ASCII
  "would very likely have missed".

Each of those is the same shape: a check that was cheap to run, that the agent
would not have run unprompted, and that converted a visual impression into an
executed fact.

### What broke

| gap | severity | found by |
|---|---|---|
| `from arc import ...` failed in the invocation the contract documents | high | all three agents, independently |
| `arc.png()` was dead code — no Pillow in the solver runtime | medium | all three agents |
| Nothing stated what the runtime actually provided (no numpy either) | medium | inferred from the above |
| A tautology was recorded as a verified invariant | high | `89565ca0` |
| No way to retract a bad invariant | medium | `28a6681f` |
| Gate refusals were never written to the event ledger | medium | building `cc trace` |
| `cc trace` counted a harness-shipped file as agent exploration | medium | the test suite |

The first one is the instructive failure. `python explore/foo.py` puts the
*script's* directory on `sys.path`, not the working directory — so the toolkit
at the workspace root was invisible to exactly the command CLAUDE.md told the
agent to run. Every agent hit `ModuleNotFoundError` on its first experiment and
carried `sys.path` boilerplate for the rest of the run. The harness put friction
on the single most common action in the entire loop, and it took real agents to
notice, because reading the docs and reading the code both look fine in
isolation.

The tautology is the most interesting. An agent wrote
`verify("noise colour is well separated", True if ... else True)` and the ledger
recorded it as `[VERIFIED]`. It caught the mistake itself and re-ran the claim
properly, but nothing in the tooling distinguished a claim backed by a
measurement from one backed by nothing — in the artifact that survives
compaction and is replayed as established fact. That is the doctrine's own
failure mode occurring inside the tool built to prevent it.

### Shipped after round 1

- Mirror `arc.py` into `explore/` so the documented invocation works.
- `dryrun.py`: one-word free scoring of `solution/solve.py`. The doctrine says
  never to use the gate as a debugger; the free alternative had been a
  90-character one-liner with a manual `sys.path` insert.
- Probe the solver's interpreter at build time and state in `CLAUDE.md` exactly
  what is installed and what is not. Advice about a missing library is worse
  than no advice.
- `arc.verify()` now recovers the condition's **source expression** from the
  caller's AST and stores it with the claim, so the ledger records what was
  executed. Compile-time constants are flagged at the call site and surfaced
  separately by `gate.py status` and `cc trace`.
- `arc.verify(claim, retract=True)` withdraws a bad invariant.
- `arc.png()` falls back to the interpreter that built the workspace.
- Gate refusals are written to the event ledger and reported by `cc trace`.
- numpy installed in the solver runtime, matching the flagship's environment.

### What round 1 could not test

Nothing exercised the harness under stress: no run hit a gate refusal, exhausted
its budget, entered the best-effort window, or went through more than two
reflection cycles. The failure-report path — the place where the harness does
most of its talking — is almost entirely unexercised. Round 2 targets that.

---

## Round 2 — harder ground

Round 2 agents were deliberately **not** told the `sys.path` workaround, so the
import fix was under test. All three confirmed it worked with zero boilerplate.

| task | solved | test examples | iterations | confidence | explore scripts | invariants | elapsed |
|---|---|---|---|---|---|---|---|
| `13e47133` | yes | 2/2 | 1/8 | 5 | 7 | 7 | 480s |
| `dd6b8c4b` | yes | 2/2 | 1/8 | 4 | 6 | 17 | 855s |
| `67e490f4` | yes | 1/1 | 1/8 | 4 | 13 | 21 | 834s |

**3/3 solved.** Running total: **6/6 tasks, 9/9 test examples**, mean 1.17
iterations, all integrity-clean, zero gate refusals.

`13e47133` is the notable one: it sits on the zero-solve frontier — in the
frozen public corpus of 117 logged attempts, no submission had ever solved it,
and `RESULTS.md` records it as solved exclusively by the flagship system. The CC
variant solved both its test outputs on the first formal submission, with
confidence 5.

### What the traces showed

The same pattern as round 1, sharper. Every agent named a specific experiment
that changed its answer, and in each case it was an experiment whose *outcome it
could not predict*:

- `13e47133`: a two-line sweep over `(connectivity, metric)` settled Chebyshev
  vs Manhattan erosion. The agent's own note: unprompted it would have written
  the obvious bounding-box formula, "been right for the wrong reason" on convex
  rooms, and lost the cells beside a concave corner.
- `67e490f4`: a script asking "does any *other* simple selector also fit?"
  returned 2/7, 4/7, 5/7 for the alternatives — converting a hypothesis that
  merely fit into the only one of four that fits.
- `dd6b8c4b`: BFS distances printed as a sorted list showed
  `[2,3,7,9,11,11,12,15,19 | 20,20,21,22,22,23]` — a clean gap at the cut-off.
  The agent had been hand-computing Manhattan distances and, in its own words,
  "my prose had in fact been converging on the wrong answer".

Two agents also reported the audit requirement doing real work: it is what
prompted the ablation batteries and equivariance tests that turned arbitrary
implementation choices into *verified non-choices*.

### What broke

| gap | severity | found by |
|---|---|---|
| `solution/` not importable — the doctrine's own step 5 needs it | high | all three agents |
| Three documents said bare `python` while CLAUDE.md named the venv | medium | `dd6b8c4b` |
| Gate dumped full test predictions unconditionally | medium | `b0039139`, `13e47133` |
| `verify()` conflates "established X" with "X is false" | medium | `dd6b8c4b` |
| Supersession keyed on the claim string breaks on rewording | medium | `67e490f4` |
| Numeric script names (`05_x.py`) are not importable | low | `67e490f4` |
| `NOTES.md` needs a Read before the first Write | low | `dd6b8c4b` |

The first is the round's instructive failure, and it rhymes with round 1's. The
doctrine tells the solver to run its verified invariants against its own test
predictions before accepting — the last line of defence in a configuration with
no reviewer. Doing that means importing `solution/solve.py`, and the toolkit
offered no way. All three agents wrote `sys.path.insert(0, 'solution')`, a
*relative* path that breaks if the script runs from anywhere but the workspace
root — the exact boilerplate `arc.py` had gone to the trouble of eliminating
everywhere else. **The harness put friction in the one place its own doctrine
sends you**, twice in two rounds.

The interpreter inconsistency is the cheapest lesson: bare `python` worked right
up until `import numpy`, at which point it failed in a way that looks like a
workspace bug rather than a documentation bug.

### Shipped after round 2

- `arc.load_solution()` returns the shipped `solve`, resolved against the
  workspace.
- All documents name the same interpreter.
- Test predictions over 400 cells are replaced by a colour histogram and a
  pointer to the archived `predictions.json`.
- `arc.refute(claim, condition)` records a dead end as a finding rather than as
  an apparent defect.
- `verify(..., key=)` for explicit supersession across rewording.
- Mechanical generalization signals wired into the gate (see below).
- Guidance on importable script names and on `NOTES.md` needing a Read first.

### The reviewer-shaped hole

With no independent reflector, nothing rejects a train-perfect but wrong rule.
`signals.py` recovers the cheap, model-free half of that job: regularities every
training output obeys that a test prediction breaks — fixed output shape, shape
relation to the input, squareness, a fixed output palette, colours absent from
the prediction's own input, and duplicate candidates that waste ARC-AGI-2's
second attempt.

It only fires when the training set is unanimous and the prediction dissents. On
round 1's three correct solutions it reported nothing; against deliberately
corrupted predictions for the same three tasks it fired on all three.

### Still untested after two rounds

**No submission has ever failed.** Nine of nine tasks passed training on
iteration 1 or 2, so the failure report, the reflection directive, the
best-effort switch, and every gate refusal remain unexercised by a real agent.
The harness does most of its talking on the failure path, and that path has been
validated only by unit tests.

---

## Round 3 — in progress

Three agents on tasks the **flagship system itself fails**, chosen specifically
to produce failures:

- `faa9f03d` — Athanor scores 0/1 even in an extended 120-turn attempt; never
  solved by CoT-only Opus 4.6 at any thinking level.
- `2b83f449` — Athanor scores 0/1; never solved by CoT-only Opus 4.6 in 8
  attempts.
- `88e364bc` — Athanor solves 1 of 2 test outputs; test 0 never solved by
  CoT-only Opus 4.6.

Agents were told the difficulty honestly and asked to prioritise evaluating the
failure reports over solving.

---

## The finding that matters most

`88e364bc` produced the first train-perfect-but-wrong result, and it is worth
more than the six clean solves.

The task's rule is "tokens slide until blocked". One question was genuinely
ambiguous: may a diagonal slide pass the tip of a wall? Both readings reproduce
all three training pairs. The agent killed the strict reading by execution — it
strands a token in open space, "whereas all ten training tokens rest with a wall
directly ahead" — and recorded that as a refutation. It then wrote, in its own
report:

> the ambiguity is gone and the second candidate slot went unused, which is the
> right outcome.

It was wrong. The prediction missed by **2 cells out of 400** — right shape,
right palette, 99.5% pixel accuracy — and the alternative reading it had in hand
was very likely the correct one. The result matched the flagship system exactly
(1 of 2 test examples, the same one missed).

Two things follow, and both are uncomfortable for the thesis.

**Executed verification can produce confident wrong answers.** The invariant was
real: all ten training tokens do rest against a wall. The error was extending it
to the test input, which nothing had established. That is an inductive leap
wearing a proof's clothes, and it is *more* dangerous than prose reasoning
precisely because it comes with a `[REFUTED]` line in the ledger and reads as
settled. Code-as-verification compresses the cost of checking; it does not
convert induction into deduction, and the harness had been silent about the
difference.

**Mechanical signals cannot cover this.** `signals.py` fired nothing, correctly:
shape, palette and structure were all consistent. A 2-cell semantic near-miss is
exactly the class of failure that needs a *reasoning* reviewer, which is what
Athanor's independent reflector is and what this variant drops. The mechanical
substitute bounds the reviewer-shaped hole; it does not close it.

There is also a cheaper lesson. Across every accepted run, **9 of 11 test
examples got a single candidate** — ARC-AGI-2 allows two, so eight free attempts
were forfeited. Athanor has a mechanism for exactly this: the reflector's
EXPAND_CANDIDATES verdict, which fires when a rule is sound but ambiguity
remains between plausible branches. The CC variant has no equivalent, and on
`88e364bc` the cost was measurable.

**Shipped in response:** the generalization-audit directive now asks the solver
to examine *how* it ruled out an alternative — killing it with a training pair
it fails is a proof; killing it by extending a training-only regularity to the
test is a leap — and to spend the second candidate whenever an alternative
reproduced every training pair and died only to a leap.

That is guidance, not a mechanism, and it is weaker than what it replaces. The
honest conclusion is that this is the first measured cost of dropping the
reviewer.
