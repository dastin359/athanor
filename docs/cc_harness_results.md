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

## Standing

**24 completed tasks, 23 fully solved, 33 of 34 test examples.** Mean ~1.2
formal iterations. Zero gate refusals in the entire experiment.

The single miss is `9bbf930d` — two attempts, both 0/1, the second carrying
every toolkit improvement. The flagship solves it.

`faa9f03d`, which no logged submission in the frozen public corpus has ever
solved in 127 attempts, **is solved** — see the section below. It came from the
second candidate, not the first.

**Read that percentage carefully — it is not comparable to a full-benchmark
score, for two reasons pulling in opposite directions.**

*Against it:* these 23 tasks are not a random sample. They were deliberately
picked for difficulty — six of the eight zero-solve frontier puzzles, all five
puzzles the flagship itself fails, and the pairs no CoT-only Opus 4.6
configuration solved in eight attempts. A hard-selected subset says nothing
about the other 97 public-eval tasks, which have never been run here.

*For it:* two tasks were attempted twice, and the table takes the better
attempt. That is the same development-then-select-best pattern `RESULTS.md`
documents for the flagship's 95.7%, so the comparison is at least like-for-like
— but on a first-attempt-only basis this variant would be 20 of 23, not 21.

What the numbers do support, because they are head-to-head on the same puzzles:

| comparison | this variant | flagship |
|---|---|---|
| zero-solve frontier (9 pairs) | **8** | 8 |
| pairs the flagship itself fails (6) | **6** | 0 |
| cost per task (n=4 batch) | $1.86 | $3.12 mean / $1.71 median |

The systems fail on *different* puzzles: each solves exactly one frontier pair
the other cannot — `faa9f03d` here, `9bbf930d` there. Between them the entire
frontier falls; neither does it alone. This variant carries no reviewer, no
second model, and no inter-agent artifact exchange.

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

---

## Round 3 — the failure path, and a fix validated in flight

Three agents on tasks the flagship system itself fails, plus one run with a
deliberately starved 3-iteration budget to force the budget-exhaustion machinery
that three rounds of natural running never reached.

| task | result | test examples | iterations | confidence | candidates | flagship reference |
|---|---|---|---|---|---|---|
| `88e364bc` | partial | 1/2 | 1/8 | 4 | 1, 1 | flagship also 1/2, same example missed |
| `faa9f03d` | failed | 0/1 | 1/8 | 4 | 2 | flagship 0/1 at 120 turns; 0/127 in the public corpus |

Running total: **9 tasks, 10/13 test examples.**

### The fix that validated itself in flight

`88e364bc` (analysed above) produced the candidate-budget paragraph. `faa9f03d`
ran after it shipped, and its agent volunteered this without being asked:

> the "candidate budget" paragraph in particular is well-written — the
> distinction between *killing an alternative with a training pair* (proof) and
> *killing it by extending a training-output regularity to the test* (an
> inductive leap) is exactly the distinction I was fudging in my own head, and
> reading it is why candidate 2 shipped instead of being argued away.

Direct causal evidence that the fix changed behaviour on the next run. It did
not rescue the task — both candidates were wrong, and `faa9f03d` remains
unsolved by anything in the public record — but the mechanism did what it was
built to do.

The `UNSPENT SECOND ATTEMPT` check also validated itself live: it was active
during `faa9f03d`'s submission and correctly stayed silent, because that agent
had already spent its second slot.

### Still not exercised

**Four rounds, thirteen test examples, and no submission has ever failed.** Every
accepted run passed training on iteration 1 or 2. The failure report, the
reflection directive, the best-effort switch and every gate refusal remain
validated only by unit tests. The starved-budget run exists to break that; the
honest reading so far is that with a competent solver and free unlimited
exploration, the gate is simply never used as a debugger — which is exactly what
the doctrine asks for, and which makes its failure path structurally hard to
reach.

### Shipped during round 3

- `arc.rival(name, solve_fn)` — register an alternative reading; it is scored
  against training and its test predictions kept. The gate reports a rival that
  fits all training pairs and disagrees with the submission, which converts the
  second-attempt question from a judgement into a measurement.
- `UNSPENT SECOND ATTEMPT` — the gate pairs recorded dead ends with test
  examples still carrying one candidate.
- `arc.check(..., show_diff=True)` — which cells a prediction got wrong.
- A durability nudge in `dryrun.py` when the invariant ledger is empty, after
  two hard-task agents were observed running many scripts and recording nothing.

---

## Consolidated results (rounds 1–3)

| task | solved | iterations | confidence | explore scripts | invariants | candidates | note |
|---|---|---|---|---|---|---|---|
| `13e47133` | 2/2 | 1 | 5 | 10 | 16 | 1, 1 | zero-solve frontier |
| `28a6681f` | 1/1 | 1 | 4 | 12 | 29 | 1 | |
| `67e490f4` | 1/1 | 1 | 4 | 13 | 21 | 2 | 30x30 |
| `88e364bc` | 1/2 | 1 | 4 | 14 | 13 | 1, 1 | flagship also 1/2, same example |
| `89565ca0` | 1/1 | 1 | 4 | 9 | 14 | 1 | |
| `9bbf930d` | 0/1 | 2 | 4 | 13 | 18 | 2 | frontier; starved 3-iteration budget |
| `b0039139` | 2/2 | 2 | 5 | 5 | 17 | 1, 1 | |
| `dd6b8c4b` | 2/2 | 1 | 4 | 6 | 15 | 2, 1 | |
| `faa9f03d` | 0/1 | 1 | 4 | 16 | 16 | 2 | flagship 0/1 at 120 turns; 0/127 public |

**10 of 13 test examples across 9 accepted tasks**, every run integrity-clean,
zero gate refusals, mean 1.2 iterations.

Two of the three misses are tasks the flagship also fails, and one of those
(`faa9f03d`) has never been solved by any submission in the public corpus. The
third (`88e364bc`) matches the flagship exactly, missing the same test example.

### Did the candidate fixes work?

The `88e364bc` analysis produced a sequence of changes aimed at one failure:
a solver killing a live alternative with an inductive leap and discarding the
free second attempt. The runs split cleanly around them.

| | test examples | used the second attempt |
|---|---|---|
| before the fixes | 11 | 2 (18%) |
| after | 2 | 2 (100%) |

Two data points is not a result, and both post-fix runs were hard tasks where
hedging is more obviously attractive. But the mechanism is not merely
correlated with the change — one agent stated the causal path directly, and the
other's candidate count went 1 → 2 in the iteration immediately after the prompt
fired. Round 4 adds three more frontier runs to the post-fix sample.

The honest caveat in the other direction: hedging did not rescue either task.
`faa9f03d` and `9bbf930d` both shipped two candidates and both scored 0/1. A
second attempt helps only when one of the two readings is right.

---

## The loop closing: a failure turned into a mechanism turned into a solve

This is the result the improvement loop exists to produce, and it is worth
stating precisely because the causal chain is fully traceable.

**The failure.** `88e364bc` passed training 3/3 and missed by 2 cells out of
400. Its agent had implemented a rival reading, killed it by extending a
regularity from the training *outputs* to the test *input*, and wrote: "the
ambiguity is gone and the second candidate slot went unused, which is the right
outcome." It was wrong, and the free second attempt was discarded.

**The mechanisms built from it**, in order:

1. a paragraph in the generalization audit distinguishing killing a rival with
   a training pair it fails (a proof) from killing it with an out-of-sample
   extrapolation (a leap);
2. `UNSPENT SECOND ATTEMPT` — the gate pairs recorded dead ends with test
   examples still carrying one candidate;
3. `arc.rival(name, fn)` — register an alternative reading; it is scored against
   training and its test predictions kept;
4. `UNHEDGED RIVAL` — the gate reports a registered rival that fits every
   training pair and disagrees with the submission;
5. the same question raised at dry-run time, where acting on it is free, and
   `accept` re-running `solve.py` so hedging costs no iteration.

**The recovery.** On `abc82100` — a zero-solve frontier task, unsolved by any of
121 logged attempts in the public corpus — the agent registered the rival
*"unmatched dots (a colour no legend names as a source) survive unchanged"*,
which reproduced all 4 training pairs. `UNHEDGED RIVAL` fired. It spent the slot.

| | matches ground truth |
|---|---|
| candidate 1 (its primary reading) | no — off by 7 cells |
| candidate 2 (the registered rival) | **yes, exactly** |

The agent's own best reading was wrong. The mechanism is the entire reason the
task scored.

**What this does and does not show.** It is one conversion, and a single
instance does not establish a rate. But the chain is not inferential: the rival
is named in the ledger, the gate's prompt is in the archived report, the
candidate ordering shows the primary reading losing, and the rival's stored
prediction is byte-identical to the ground truth.

It also sharpens the earlier, more pessimistic finding rather than replacing it.
Hedging still buys coverage, not insight — `faa9f03d` and `9bbf930d` both shipped
two candidates and both scored 0/1, because neither reading was right. What
changed on `abc82100` is that the correct reading *was* among the two, and the
harness is what put it there.

---

## Production path: first end-to-end run and first cost figure

Every result above was driven by a sub-agent inside an existing Claude Code
session. `athanor cc run` — the subprocess launcher users would actually invoke
— had been smoke-tested but never used for a full solve. It has now.

```
28a6681f   SOLVED 1/1   2/6 iterations   46 turns   841s   $3.80
           integrity clean · 472 stream messages · 0 malformed
           tokens: 58k out, 2.73M cache read, 97k cache write
```

This validates the last untested component: argv construction against the real
CLI, working directory, permission mode, the tool allowlist, `stream-json`
capture, event translation, PNG reads by the agent, and the contamination scan
running against a real transcript.

**On the cost number.** $3.80 for one task, against the flagship's reported
$3.12/task average. The comparison does not hold yet, in both directions:

- This is a **cold single run**. The flagship's $3.12 is explicitly a
  batch-deployment figure in which only the first puzzle pays the system-prompt
  cache write; every later task hits the provider cache. This run paid 97k
  tokens of cache *writes* that a batch would amortise.
- It is n=1, on a task of middling difficulty that took 2 iterations. The
  flagship's per-task cost ranges from $0.23 to $20.04 with a median of $1.71,
  so a single sample says very little about a mean.
- Claude Code carries a larger default system prompt and a general-purpose tool
  surface that the flagship does not pay for; against that, this variant runs no
  reviewer context and does not re-send puzzle data every turn.

The honest statement is that the CC variant lands in the same order of magnitude
as the flagship on its first measured task, and that a comparable number
requires `athanor cc batch` over a real task set with a warm cache. The harness
can now produce that figure; this run is the evidence the machinery works, not
the answer.

---

## Round 4 — zero-solve frontier sweep, and the candidate measurement

Three of the remaining frontier pairs, all carrying the mechanisms built from
`88e364bc`.

| task | result | iterations | confidence | candidates | rivals registered |
|---|---|---|---|---|---|
| `269e22fb` | **2/2** | 1 | 5 | 1, 1 | 2 |
| `a32d8b75` | **2/2** | 2 | 4 | 2, 2 | 2 |
| `abc82100` | **1/1** | 1 | 4 | 2 | 1 |
| `2b83f449` | **1/1** | 2 | 4 | 1 | 0 |

`2b83f449` is the one to note alongside the frontier results: `RESULTS.md`
records the flagship failing it 0/1, and no CoT-only Opus 4.6 configuration
solved it in 8 attempts. This variant solved it.

### Full standings

**13 accepted tasks, 16 of 18 test examples, 1.31 iterations mean**, every run
integrity-clean, zero gate refusals across the entire experiment.

Four zero-solve frontier pairs solved — `13e47133`, `269e22fb`, `a32d8b75`,
`abc82100` — each one a pair no submission in the frozen public corpus had ever
solved. Two tasks solved that the flagship fails. The two misses are `faa9f03d`
(flagship also 0/1; unsolved by all 127 logged attempts) and `9bbf930d`, plus
`88e364bc` partial at 1/2, matching the flagship exactly.

### Did the candidate mechanisms work?

| | test examples | used the second attempt | rivals registered |
|---|---|---|---|
| before the changes | 11 | 2 (18%) | 0 |
| after | 8 | 5 (63%) | 5 |

The mechanism is used, and the qualitative evidence is stronger than the counts.
Three agents independently described the *concept* — not the function — as what
changed their behaviour:

- `faa9f03d`: "reading it is why candidate 2 shipped instead of being argued
  away."
- `abc82100`: an invariant "*looks* like it refutes the rival but holds only
  vacuously in training. Without the rival concept I think I'd have talked
  myself into that refutation." Its rival was the correct answer.
- `a32d8b75`: it "made me stop asking 'is my rule right?' and start asking
  'which situations does the test need that training never witnesses?'" — which
  surfaced an 8-cell case it had missed in three passes.

That last reframing is the real product. Eight readings on `a32d8b75` died to a
training pair and became proofs; two survived and shipped as second candidates.

### What the round cost

Four more harness bugs, all mine, all found by agents rather than by tests:

- `rival()` advised spending a slot on a reading that predicted identically — it
  stored the predictions and never compared them.
- The dry-run hedging advice named rivals diverging on a *different* test
  example than the one with the free slot: "trains you to skim it."
- An agent spent an iteration submitting purely to *read* the gate's
  candidate-spread line — information I had put only on the budgeted path two
  commits earlier.
- Making `accept` re-run `solve.py` opened a second door into acceptance that
  skipped the hypothesis/code coupling `submit` enforces.

The pattern across all four: **every convenience added to the budgeted path
creates an incentive to spend budget, and every door added to acceptance needs
the same locks as the front one.**

---

## The cache defect: the harness was paying full price on every task

The cost caveat above assumed the only thing missing was a batch. It wasn't. I
ran one and read the token counts per task:

```
task 1   cache_write 52k
task 2   cache_write 61k     ← should have been near zero
```

The second task in a sequential batch wrote *more* cache than the first. There
was no amortisation at all — the mechanism the flagship's $3.12/task figure
depends on was simply not operating in this harness.

**Cause.** Every task runs in its own workspace directory, and Claude Code's
default system prompt embeds the working directory (along with env info, memory
paths, and git status). The cached prefix therefore differed on every single
task. Prompt caching matches on an exact prefix; one differing path near the
front invalidates everything after it.

**Fix.** `--exclude-dynamic-system-prompt-sections`, which relocates those
volatile sections into the first user message. It applies only alongside the
default system prompt — which is what this harness uses, since it composes via
`--append-system-prompt` rather than replacing the prompt outright. Wired to
`CCRunConfig.stable_system_prompt`, default on.

**Measurement.** Rather than wait on a second batch, a direct A/B: the same
trivial prompt (`Reply with exactly: OK`) run in two different working
directories, with and without the flag.

| | workspace 1 | workspace 2 |
|---|---|---|
| **without the flag** — cache_write | 1739 | 1739 |
| **without the flag** — cache_read | 3289 | 3289 |
| **with the flag** — cache_write | 1675 | **745** |
| **with the flag** — cache_read | 3289 | **4219** |

Without the flag, workspace 2 wrote the identical 1739 tokens workspace 1 did —
byte-for-byte no reuse, exactly the batch's signature. With the flag, workspace
2's write dropped to 745 while its read rose to 4219: the second workspace
consumed the prefix the first one had cached.

That closes the caveat left standing after the production run. The earlier
statement — that a comparable cost figure "requires `athanor cc batch` over a
real task set with a warm cache" — was true but incomplete. The harness could
not have produced a warm cache at all, however many tasks it ran.

### How much this is worth, honestly

Less than the framing above implies, and the distinction matters.

What the flag recovers is the **system-prompt prefix** across tasks — a few
thousand tokens. The bulk of a run's cache traffic is *within* a task, as each
turn writes the growing conversation forward; the completed batch below writes
84k-228k cache tokens per task against 1.0M-2.1M read, and none of that is
touched by this fix. The flagship's own accounting puts batch amortisation at
$5.80 across 119 tasks, roughly **$0.05 a task**.

So: a mechanism that was completely broken now works, and it is worth about
five cents per puzzle. Both halves of that are true and the second one is the
one easy to lose. The reason to fix it is that a harness whose cost model
silently doesn't apply is a harness you cannot reason about — not that the
number moved.

### The completed batch

Four unseen public-eval tasks, run sequentially through `athanor cc batch`
(before the fix — this batch is what exposed it):

| task | result | iterations | cost |
|---|---|---|---|
| `e8686506` | SOLVED 1/1 | 1 | $1.702 |
| `78332cb0` | SOLVED 2/2 | 1 | $1.707 |
| `7b5033c1` | SOLVED 1/1 | 1 | $1.066 |
| `3dc255db` | SOLVED 1/1 | 1 | $2.970 |

**4/4 solved, $1.861 per task, 1.0 iterations mean.** Against the flagship's
$3.12 mean and $1.71 median.

Two caveats worth keeping attached to that number. Every one of the four solved
on the first submission, which is the cheap end of the distribution — the
flagship's per-task cost ranges $0.23 to $20.04, and a set of four
first-iteration solves is not a sample of that spread. And n=4 is n=4. What can
be said is that the variant is not in a different cost regime from the
flagship, and that the earlier single-run $3.80 figure was a cold run rather
than a representative one.

---

## Round 5 — the last two frontier pairs, and four ledger defects

`8b7bacbf` and `da515329`, the two zero-solve frontier pairs left. Both
accepted on the first submission.

| task | result | iterations | confidence | candidates | rivals registered |
|---|---|---|---|---|---|
| `8b7bacbf` | **2/2** | 1 | 4 | 2, 1 | 4 |
| `da515329` | **1/1** | 1 | 4 | 2 | 3 |

That completes the sweep: **six zero-solve frontier pairs attempted, six
solved** — `13e47133`, `269e22fb`, `a32d8b75`, `abc82100`, `8b7bacbf`,
`da515329`. Each is a pair no submission in the frozen public corpus had ever
solved.

Both runs are also clean demonstrations of the candidate mechanism doing the
thing it was built for. `da515329`'s rule has one parameter that training
cannot pin down — the chamfer width, observable only at `a >= 4`, and no
training example reaches it. The agent shipped both readings and got it right.
`8b7bacbf` hedged on a wire-crossing that occurs nowhere in training: "no
training pair contains a crossing at all, so nothing says whether an
interrupted stroke continues through."

### Four defects, all in how the ledger reads itself back

The interesting thing about this round is that none of the four was a wrong
answer. All four were the harness **misdescribing its own state** to the agent.

**1. Every dead end was rendered as a success.** `gate.py status` marked
entries by whether the recorded check *held* — but a `refute()` entry that
holds means the hypothesis is dead. The replay therefore read:

```
[OK  ] contact by 4-adjacency only also explains the training data
```

about a reading the agent had killed. As the solver put it: "On a skim after
compaction that reads as the *opposite* of what was found." The status replay
is the workspace's most durable artifact — it is what a post-compaction agent
reads first — so this was inverting the single record most likely to be read
without its context. Now `[DEAD]` / `[OPEN]`.

**2. `rival()` told a solver to delete a correct hedge.** `da515329` had
already shipped its 3/3-fitting rival as candidate 2. `rival()` then said:

> It predicts exactly what you do on every test input, so it is not a divergent
> reading and needs no slot.

The rival was "not divergent" only because it had been found inside the
candidate list it was already in. The solver caught it: "Had I trusted that
line I would have deleted the hedge and forfeited the slot… the one message a
solver most needs to be right about is the one that inverts when you follow the
advice early."

This is the second bug in the same function, and both came from collapsing
distinct situations into one boolean. There are four, and they now print
differently: redundant with candidate 1; *is* candidate 2 already; diverges
from both; diverges but both slots are spent on other readings. `8b7bacbf`
independently hit that last one — told "That is your second candidate" about a
test example whose second slot was already taken.

**3. A claim recorded from a heredoc captured no evidence, silently.** The
constant-condition check is derived from the caller's AST, so when the call
site cannot be read the check cannot fire. A solver recorded a refutation from
a heredoc with a literal `True` — precisely the anti-pattern the check exists
to catch — and got no warning, because the warning depends on source the parser
never reached. It noticed unaided and re-recorded from a file. "No evidence"
and "good evidence" looked identical in the ledger; entries now carry
`unsourced` and say so.

**4. A corrected claim had nowhere to put the correction.** `da515329`'s ledger
contains this live, `[VERIFIED]` invariant:

> the corner jog has a fixed width of 1 (m = a-1); it is instead always a - 2
> wide (m = 2)

A sentence whose first clause is false, stamped verified. Nothing was wrong
with the agent's reasoning — it had killed the first reading and found the
second, and `key=` supersedes by key, so amending the claim text and re-running
is the natural motion. There was simply no way to say *"that reading died, here
is what replaced it"*. Replacements now carry and print what they displaced,
and a claim whose verdict flips between runs says so loudly, because code
written while it held is now built on sand.

The through-line: **a ledger that survives compaction is read by someone with
no memory of writing it, and every one of these four defects was invisible to
the agent that had the context and misleading to the one that would not.**

---

## The frontier, counted properly

I described round 5 as completing "six zero-solve frontier pairs attempted, six
solved". That count was selective, and the correction matters more than the
headline did.

`RESULTS.md` defines the frontier as 9 test outputs across 8 puzzles: pairs
where no submission in the frozen 2026-04-12 public corpus ever succeeded,
with at least 80 logged attempts each. **All 8 puzzles have now been
attempted.** Two of them were attempted in earlier rounds and *missed* — and
listing only the six that were solved quietly dropped them.

| pair | HF attempts | this variant | flagship |
|---|---:|---|---|
| `13e47133 / 0` | 117 | solved | solved |
| `269e22fb / 0` | 119 | solved | solved |
| `269e22fb / 1` | 119 | solved | solved |
| `8b7bacbf / 0` | 125 | solved | solved |
| `a32d8b75 / 0` | 118 | solved | solved |
| `abc82100 / 0` | 121 | solved | solved |
| `da515329 / 0` | 120 | solved | solved |
| `9bbf930d / 0` | 117 | **missed** | solved |
| `faa9f03d / 0` | 127 | **missed** | **missed** |

**7 of 9, against the flagship's 8 of 9.** One pair the flagship solves and
this variant does not (`9bbf930d`, missed on 2 iterations); one pair neither
solves (`faa9f03d`, which no logged submission has ever solved and which
resisted a 120-turn flagship attempt).

> **Superseded.** `faa9f03d` was solved in round 8 — see "faa9f03d" below. The
> frontier standing is now 8 of 9. This table is kept as the round-5 record.

That is the number to quote. It is a good result — every one of these pairs had
100+ failed attempts behind it — and it is one pair short of the system this
variant is being compared against, on a frontier of nine. Both facts belong in
the same sentence.

---

## Round 6 — aimed at the flagship's own misses

With the frontier exhausted, the sharper comparison is `RESULTS.md`'s other
hard set: the 22 pairs no CoT-only Opus 4.6 configuration solved in 8 attempts.
The flagship solves 16 of those and names the 6 it does not:

`88e364bc/0`, `d35bdbdc/0`, `d35bdbdc/1`, `2b83f449/0`, `dbff022c/0`,
`faa9f03d/0` — five puzzles.

That set is the most informative target left, because on it a difference in
either direction is a capability signal rather than a variance one. Standing
before this round: `2b83f449` **solved** (the flagship fails it), `faa9f03d`
missed by both, `dbff022c` not yet attempted.

Round 6 takes three:

| task | why |
|---|---|
| `d35bdbdc` | flagship 1/3; its two unsolved pairs are both in the set |
| `800d221b` | flagship 0/1 (CoT-only Opus 4.6 solves it 3/8, so not in the 22) |
| `88e364bc` | **controlled retry** |

`88e364bc` is the one to watch. It is where this variant lost 2 cells out of
400 in an early round — a solver killed a valid rival by extending a
training-output regularity to the test input, and finished 1/2, matching the
flagship exactly. The entire rival mechanism was built in response to that
single failure. Re-running the same puzzle with the mechanism in place is the
closest thing this experiment has to a controlled test of whether the machinery
recovers a loss it was designed for, rather than merely being used.

A caveat that applies whatever comes back: n=1 per task, and these are the
puzzles where outcomes are least stable. A solve would be evidence the
mechanism can reach the answer, not that it reliably does.

### Round 6 results

| task | result | iterations | candidates | flagship |
|---|---|---|---|---|
| `800d221b` | **1/1** | 1 | 2 | **0/1** |
| `88e364bc` | **2/2** | 1 | 2, 2 | **1/2** |

Both are puzzles `RESULTS.md` records the flagship failing.

**`88e364bc` is the controlled retry, and it worked.** The earlier run on this
puzzle finished 1/2 — a solver killed a valid rival by extending a
training-output regularity to the test input, and missed by 2 cells out of 400.
The entire rival mechanism was built from that single failure. This run shipped
two candidates on *both* test inputs and took both.

The agent's account is mechanistic rather than lucky. Its uncertainty was
precisely located — whether a diagonal ray may pass a cell with an occupied
shoulder — and it identified that training contains only 4 diagonal rays, none
of which witnesses the case. Test 0 needs one reading, test 1 the other. It
found this by writing a script that enumerated the situation types its rule had
to handle on each test input and diffed them against what training witnesses;
that script found a case it "had already dismissed and would otherwise have
shipped unhedged."

Still worth stating plainly: this is one run against one earlier run. It
establishes that the mechanism *can* recover the loss it was designed for, on
the puzzle it was designed from. It does not establish a rate.

### What round 6 cost: four more defects, one of them a repeat

The repeat is the one that matters. The paragraph that produced that decisive
script — *"enumerate the situation types your rule has to handle on the test
input, and check that each one is actually witnessed in a training pair"* — was
printed **only after a submission**. The solver called it "the single
highest-value string the harness printed" and then noted it had arrived only
once the iteration was spent.

This is the third time a convenience has been found on the budgeted path alone,
after the candidate-spread line in round 4 and the hedging prompt before it. The
rule keeps having to be relearned: **anything that changes what a solver ships
must be reachable for free.** It now prints on every dry-run branch, with a test
that fails if the workspace copy and the gate copy drift apart.

The other three:

- `rival()` announced nothing when a re-registration replaced what a name meant.
  A solver fixed a buggy rival by re-registering the same name; the earlier
  entry — which had claimed divergence on both tests — vanished without trace.
  `verify()` goes to considerable trouble to announce supersession; `rival()`
  did not.
- The opaque-name warning fired even when `note=` carried the measured values,
  telling a solver to do the thing it had just done.
- `[DEAD]` read as though the *check* were broken rather than the hypothesis.
  Now `[KILLED]`, matching the word `refute()` itself prints.

And one gap that three separate solvers had reported without my noticing the
pattern: `arc.sweep()`. Each of them ran a tie-break sweep — 8, 14 and 18
candidate rules scored against training in a single pass — and each recorded
two or three of them and left the rest in prose, because `refute()` is
per-claim. One said it directly: "the ledger under-represents what was actually
ruled out." In every case the sweep was what produced the second candidate: the
highest-yield work in the run, and the least durable. `sweep()` records the
whole thing as one entry, and when more than one reading survives it says so —
that is a hedging obligation discovered before any budget is spent, and it names
the exact readings to put through `rival()`.

### A note on the durability check

Watching this round also showed the empty-ledger reminder was reaching almost
nobody: it hung off `arc.show()`, which agents call in roughly one script out of
five to eight, usually an early one. So it was gated on a call they had largely
stopped making by the time the condition became true. It now runs at interpreter
exit as well.

Honesty about the case that prompted it: `d35bdbdc` spent about forty minutes
and nine scripts before recording anything — and then recorded eight invariants
unprompted. The window of unprotected work was real; the agent recovered from it
without help. The fix is justified by the exposure, not by an observed loss.

### Round 6 complete

| task | result | iterations | candidates | flagship |
|---|---|---|---|---|
| `800d221b` | **1/1** | 1 | 2 | **0/1** |
| `88e364bc` | **2/2** | 1 | 2, 2 | **1/2** |
| `d35bdbdc` | **3/3** | 1 | 2, 2, 1 | **1/3** |

**3/3 tasks, 6/6 test examples, 1.0 iterations mean.** Every one of the three is
a puzzle `RESULTS.md` records the flagship failing or partly failing, and
between them they take five pairs the flagship does not.

`d35bdbdc` is the largest single gain: three test outputs, all three correct,
against the flagship's 1/3. Two of its pairs are in the 22-pair set that no
CoT-only Opus 4.6 configuration solved in 8 attempts.

Its solver hedged on two of three test examples for a reason it could state
exactly: two different selectors — nesting depth, and where the gray snake's
ends dock — agree on every training pair and diverge on tests 0 and 1. It
recorded the refutation that mattered most as a claim about the *evidence*
rather than about the puzzle: "training can separate the snake reading from the
depth reading" — refuted, meaning it cannot.

### Four more defects, and one that argues with the doctrine

The interesting one is (1), because it is the doctrine's own mechanism falling
short of what the doctrine asks for.

1. **`verify()` could only record one-liners.** The expression capture reads a
   single expression at the call site. Every substantive check in this run was a
   multi-line function — object detection, a BFS over the containment forest, a
   graph walk — so the honest options were a bare name (correctly flagged as
   opaque) or one unreadable comprehension, which is what the solver tried next
   and then had to retract: "a genuinely bad ledger entry". The harness insists
   the ledger records what executed, and for anything larger than an expression
   it could not. `verify(..., evidence=<measured value>)` is now a first-class
   field, and it answers both warnings because it supplies exactly what they ask
   for.

2. **Not every `refute()` is a rival reading.** The gate listed that
   evidence-claim under "rival readings you ruled out — check how each died",
   which inverts a refutation whose death was the finding.

3. **When live rivals all agree on the open slot, say so.** This solver's rival
   agreed on test 2, and it was shown the dead-end list for test 2 with nothing
   acknowledging the rival had been considered and found irrelevant there. "The
   slot needs a reading you have not named yet" is a different instruction from
   "reconsider these dead ends".

4. **A partial hedge is usually a bug.** This solver's alternative path returned
   early on one input, so `solve()` emitted two candidates for some tests and one
   for another. The nudge said "you registered no rivals" — true, and pointing
   the wrong way. As the solver put it, the harness "did not distinguish 'you
   chose not to hedge' from 'your hedge silently failed'."

---

## The flagship's own misses, counted

`RESULTS.md` names the 6 pairs — 5 puzzles — that the flagship does not solve
among the 22 that defeat every CoT-only Opus 4.6 configuration. All five puzzles
have now been attempted.

| pair | this variant | flagship |
|---|---|---|
| `2b83f449 / 0` | solved | missed |
| `88e364bc / 0` | solved | missed |
| `d35bdbdc / 0` | solved | missed |
| `d35bdbdc / 1` | solved | missed |
| `dbff022c / 0` | solved | missed |
| `faa9f03d / 0` | **missed** | missed |

**5 of 6** at the time of writing. The one miss was `faa9f03d`, which no logged
submission in the frozen public corpus had ever solved and which resisted a
120-turn flagship attempt.

> **Superseded.** `faa9f03d` was solved in round 8. All six pairs the flagship
> fails are now taken.

Set against the frontier table earlier — where this variant takes 7 of 9 and the
flagship 8 of 9 — the two results say something more specific than either alone.
The systems fail on different puzzles. The flagship takes `9bbf930d`, which this
variant missed; this variant takes five pairs the flagship does not. Neither
dominates, and `faa9f03d` defeats both.

That is worth more than a headline number, because it is evidence the two
architectures are not just differently tuned but differently *capable* — and
this one carries no reviewer, no second model, and no inter-agent artifact
exchange. What it has is the gate, the toolkit, and the doctrine.

Which of those three is doing the work is exactly what the next experiment asks.

---

## Round 7 — one gained, one that stayed lost

| task | result | iterations | candidates | note |
|---|---|---|---|---|
| `dbff022c` | **1/1** | 1 | 2 | last untouched puzzle in the flagship's unsolved set |
| `9bbf930d` | **0/1** | 1 | 2 | second attempt; the flagship solves it |

`9bbf930d` is the honest half. It is the one frontier pair the flagship takes
and this variant does not, and a retry carrying `sweep()`, the situation-types
prompt on the free path, and the rival-standing fixes **still missed it** — this
time with two candidates and a well-argued hedge rather than an unhedged guess.
Two attempts, two misses. The frontier standing was 7 of 9 at this point;
`faa9f03d` later took it to 8 of 9, but `9bbf930d` remains the sole miss.

Worth saying plainly: several rounds of harness improvements did not convert
this pair. The improvements are real and measurable elsewhere; they are not a
general solvent.

That run did, though, give the sharpest confirmation of the free-path fix
shipped one round earlier:

> "It fired at exactly the right moment and it changed what I shipped. It
> appeared on my very first `dryrun.py`, before any budget was spent… Without it
> I would have submitted the over-inclusive launch rule as candidate 1. This is
> the single highest-value string the harness printed."

One round before, the same paragraph existed only behind a spent iteration.

---

## The ablation: is the doctrine load-bearing?

Every solver so far was told to follow a code-as-verification doctrine and then
asked whether it helped. All said yes. That is a leading question, and it does
not separate "the harness works" from "Opus 5 is good at ARC".

So: `88e364bc` again — the puzzle where the doctrine arm went 1/2 → 2/2 — with
the doctrine **stripped** from `CLAUDE.md` and the system prompt withheld
entirely. No mention of verification-by-execution, rivals, hedging, or the
second candidate. The toolkit was left untouched and importable. Both arms were
told the benchmark allows two candidates, since otherwise the experiment would
test knowledge of the rules rather than the discipline.

### Result

| arm | candidates shipped | score |
|---|---|---|
| doctrine | test 0 → 2, test 1 → 2 | **2/2** |
| ablated | test 0 → 2, test 1 → **1** | **1/2** |

1/2 is exactly what the flagship scores on this puzzle, and exactly what this
variant scored on it before the rival mechanism existed.

### What the ablated agent actually did

Not what "no doctrine" might suggest. It read `arc.py` and used **the entire
toolkit unprompted** — `verify`, `sweep`, `rival`, `solution_module`,
`load_solution`, `png`, `diff` — and registered **eight** rival implementations.
It ran a sweep on a genuine ambiguity and got two survivors. It hedged correctly
on test 0. Its own account: "Whenever I was unsure I built the alternative and
executed it rather than reasoning about it."

So the practice transferred without the doctrine. What did not transfer was the
*specific* out-of-sample question. The two arms framed different residual
ambiguities: the ablated agent asked whether markers block each other mid-slide
and measured no divergence; the doctrine agent asked whether a diagonal ray may
pass a cell with an *occupied shoulder* and found that test 1 turns on it. The
ablated agent concluded "on test 1 nothing that fits training diverges" — and
shipped one candidate there.

### What this does and does not establish

It does not establish that the doctrine is worth 1 pair in 2. **n=1 per arm**,
on a puzzle whose three recorded attempts have gone 1/2, 1/2, 2/2 — two of three
landing at 1/2. A difference of one test example between two single runs is
inside the noise of that history, and saying otherwise would be reading a
coin-flip as a trend.

What it does show is narrower and more interesting: the toolkit alone is enough
to induce the *practice* — an agent told nothing about verification wrote eight
rivals and swept a tie — but on this puzzle the practice was not enough to find
the question that mattered. Both agents were rigorous. They differed in which
ambiguity they thought to frame, and only one of those framings reached the
second pair.

Two honest limits on top of that. This ablates the **doctrine, not the
toolkit**: `arc.py`'s own docstrings argue for verification-by-execution, and
the ablated agent plainly absorbed them, so the comparison is "tools plus
doctrine" against "tools alone", not against nothing. And a real answer needs
many runs per arm, not one. The value here is that it is the first evidence
pointing at *which component* does the work — and it points away from the system
prompt and toward the toolkit, which is the opposite of what the doctrine's
prominence in this repo would suggest.

---

## A negative result: verification density does not predict correctness

`cc trace`'s headline metric is verification density — invariants and
exploration scripts per formal iteration — and the module's own docstring called
it "the metric that matters". Measured across every run in the experiment, it
does not discriminate at all.

| group | n | invariants | before 1st submission | scripts | iterations |
|---|---:|---:|---:|---:|---:|
| fully solved | 21 | 14.9 | 11.8 | 12.0 | 1.2 |
| not fully solved | 4 | 13.5 | 11.0 | 13.5 | 1.2 |

The failures verify as much as the successes, and by some measures more. The
individual misses:

| task | score | invariants | before 1st | scripts |
|---|---|---:|---:|---:|
| `88e364bc` | 1/2 | 13 | 11 | 14 |
| `9bbf930d` | 0/1 | 18 | 14 | 13 |
| `9bbf930d` | 0/1 | 7 | 7 | 11 |
| `faa9f03d` | 0/1 | 16 | 12 | 16 |

`9bbf930d`'s better attempt established 18 invariants, 14 of them before its
first submission — *above* the average for a solved task — and missed anyway.

**What this means.** Density measures compliance with the doctrine, not quality
of reasoning. It is the right instrument for the job it can actually do:
catching a run that guessed first and verified afterwards, which is the failure
mode the doctrine targets and which has never occurred here. It is the wrong
thing to optimise, and a high number is not evidence a run went well.

The docstring has been corrected, because a metric that describes itself as "the
metric that matters" invites exactly the mistake of reading it as a quality
signal.

**Caveat in the other direction.** Every run in this experiment complies with the
doctrine — even the ablated ones, which verified heavily without being asked to.
So there are no low-density runs to compare against, and this says nothing about
whether density *below* the 11–14 range would hurt. The finding is that within
the range these agents actually produce, more verification does not mean more
correct.

That is consistent with the sharper thing the failures have in common, which is
not a shortage of executed checks but a *missing question*: on `88e364bc` the
ablated agent measured that markers-blocking made no difference and never asked
the corner-cutting question; on `9bbf930d` both attempts verified their rule
thoroughly and hedged on the wrong axis. Executed verification compresses the
cost of answering a question. It does not tell you which question to ask.

---

## The self-audit has never rejected anything

The reviewer was dropped from this variant; the self-audit in `solution/audit.md`
is what stands in for it. `docs/design.md` predicts that substitution will be
weak — "artifact-only review is the only mechanism that can reject a
train-perfect but overfit solution. The self-audit is a weaker substitute by
construction."

Here is the empirical form of that prediction, across 26 audited runs:

- **`DECISION: RETRY` has fired zero times.** Not once has a solver's own audit
  declined its own work — including on the five runs that were wrong.
- **The confidence scale has collapsed to `{4, 5}`.** Nobody has ever claimed 1,
  2 or 3. The highest confidence in the experiment, a 5, appears on a run that
  scored 1/2.

| claimed confidence | n | fully solved |
|---|---:|---|
| 4 | 20 | 16 (80%) |
| 5 | 6 | 5 (83%) |

Two claims here, of different strength. The **resolution** claim — that
confidence does not separate right from wrong — is underpowered: the base rate
is 84%, so there is little variance to predict at n=26, and 80% vs 83% is noise.
Do not read that table as "confidence is uninformative"; read it as "this
experiment cannot tell".

The **range** claim is solid and does not depend on the base rate. An audit that
only ever emits 4 or 5, and never RETRY, is not functioning as a gate. It is
functioning as a reflection prompt — which agents repeatedly credit with
changing their work, and which is genuinely valuable — but the rejection half of
the mechanism is inert.

**What changed as a result.** An unanchored 1–5 invites exactly this collapse,
so the scale is now anchored to something checkable, using the concept the
solvers themselves found most useful — whether the situations the test input
requires are witnessed in training:

```
5  every situation the test requires is witnessed in a training pair, checked.
4  one such situation is unwitnessed and you have hedged it with candidate 2.
3  a situation the test needs is unwitnessed and you have NOT hedged it.
2  the rule fits training and you cannot say why it should generalise.
1  you are submitting because the budget is running out.
```

Whether that produces a 3 — or a RETRY — is an empirical question the next
rounds answer. It may not: a solver confident enough to submit is, by
construction, a solver that thinks it is at 4 or 5. If the anchors change
nothing, that is worth knowing too, and it would be evidence for the stronger
reading of `design.md`'s claim: that self-review cannot reject its own work and
only an independent reader can.

### Ablation, replicated

A second ablated arm, `dbff022c`, same method — doctrine stripped from
`CLAUDE.md`, system prompt withheld, toolkit untouched.

| puzzle | ablated | doctrine |
|---|---|---|
| `88e364bc` | 1/2 | **2/2** |
| `dbff022c` | **1/1** | **1/1** |

One doctrine win, one tie. That is what "inside the noise" looks like, and it is
the reason the first result was not written up as an effect.

The **qualitative** finding replicated much harder than the score did. Told
nothing about verification, rivals, hedging, or the second candidate, the second
ablated agent used `verify`, `refute`, `sweep`, `rival`, `load_solution`,
`solution_module`, and `verify(..., over=)` — a parameter shipped thirty minutes
before its run. It found the same ambiguity the doctrine arm found (which cell
of a legend pair is the key, unresolvable because every training legend is flush
with the top or left edge while the test legend is flush with the bottom), and
hedged it the same way, on the same axis.

It also caught itself writing a tautology — `... or True` — and retracted it
unprompted, which is the round-1 literal-detection fix doing its job for an
agent that was never told the ledger mattered.

So across two arms: **the practice transfers without the doctrine.** An agent
that reads `arc.py` adopts code-as-verification, registers rivals, sweeps ties
and hedges — because the toolkit's API and its printed advice carry the
discipline, not because a system prompt asked for it.

That is a useful thing to know for anyone who would build on this. The
shippable, portable part of the harness is the toolkit. The doctrine may still
supply the specific out-of-sample question that decides a hard pair — that is
what `88e364bc` hints at — but two arms cannot distinguish that from chance, and
this experiment should not pretend otherwise.

### Ablation, third arm — a big gap with an incoherent mechanism

| puzzle | ablated | doctrine | flagship |
|---|---|---|---|
| `88e364bc` | 1/2 | **2/2** | 1/2 |
| `dbff022c` | **1/1** | **1/1** | 0/1 |
| `d35bdbdc` | 1/3 | **3/3** | 1/3 |
| **test examples** | **3/6** | **6/6** | — |

3 of 6 against 6 of 6 looks like a decisive result, and on both puzzles where
the ablated arm lost pairs it landed on *exactly the flagship's score*. It is
tempting to write that up as "the doctrine is worth three test examples".

**That would be wrong, and the reason is in the candidate counts.**

| puzzle | ablated hedges | doctrine hedges |
|---|---|---|
| `d35bdbdc` | 2, 2, 2 | 2, 2, 1 |

On `d35bdbdc` the ablated agent hedged **more** than the doctrine agent — two
candidates on all three test inputs, against the doctrine arm's two-two-one —
and still scored 1/3 against 3/3. It did not fail to hedge. Its *primary rule
was wrong*: it read the puzzle as pointer chains resolved by absorption, where
the doctrine arm read it as dropping every other level of a containment forest.
Two different rules, both reproducing all three training pairs, only one right.

So the mechanism is not consistent across the three puzzles. On `88e364bc` the
gap is a missing hedge; on `d35bdbdc` it is a wrong reading arrived at despite
more hedging; on `dbff022c` there is no gap. "The doctrine teaches you to hedge"
predicts the first and is contradicted by the second.

An inconsistent mechanism across three puzzles is what run-to-run variance looks
like. The alternative — that the doctrine somehow produces better *rule
discovery*, not just better hedging — is a much stronger claim, has no proposed
mechanism, and three runs cannot support it.

One confound worth stating because it runs *against* the ablated arm, not for
it: the ablated runs used a strictly newer toolkit than the doctrine runs they
are compared against — `sweep()` per-reading details, `verify(over=)`, rival
contention detection, all shipped between the two sets. The ablated arm had the
better tools and did worse. That makes the toolkit-improvements story weaker,
not the doctrine story stronger; both are undercut by the same variance.

> **CORRECTED 2026-08-02.** That paragraph is wrong, and wrong in the direction
> that penalised the doctrine contrast. The toolkit is **crossed** with condition,
> not confounded with it. `arc.py` md5s: `8cb06eb6` (round6), **`21e0f74e`
> (replicate *and* ablation2)**, `341a6434` (ablation3 and arm_a_48). The winning
> `replicate` and the losing `ablation2` shipped the *same* toolkit, so toolkit
> version cannot explain the split in either direction. See the section
> "The ablation was mislabelled" below — which also establishes that the doctrine
> itself was never removed from any arm.

**What survives from all three arms** is the qualitative finding, which
replicated cleanly every time: an agent given the toolkit and no doctrine reads
`arc.py` and adopts the practice — `verify`, `refute`, `sweep`, `rival`,
`solution_module`, `over=` — registers rivals, enumerates unwitnessed situation
types, and hedges. The third ablated agent wrote a `situation_types.py` script
unprompted and used it to add two hedges it had not originally framed. Nobody
told it to.

The score difference is not established. The behaviour transfer is.

---

## The cache fix at real-batch scale: invisible, as predicted

The `--exclude-dynamic-system-prompt-sections` fix was measured with a
microbenchmark (two trivial prompts in two working directories) and estimated at
about $0.05 a task. A real batch now runs with it, so here is the same
measurement at workload scale:

| batch | task | turns | cache write | cache read | read/write |
|---|---|---:|---:|---:|---:|
| post-fix | `35ab12c3` | 44 | 278,731 | 2,993,663 | 10.7 |
| post-fix | `58490d8a` | 38 | 79,145 | 1,073,954 | 13.6 |
| pre-fix | `e8686506` | — | 129,702 | 1,725,300 | 13.3 |
| pre-fix | `78332cb0` | — | 160,570 | 1,895,479 | 11.8 |

**The fix is invisible here, and that is the correct outcome.** The cross-task
prefix it recovers is a few thousand tokens; within-task cache traffic is
80k–280k. A real improvement of that size cannot show up against that
denominator, and the ratios pre- and post-fix are indistinguishable. This is the
$0.05/task estimate confirmed at scale rather than contradicted.

**A correction, because the surface reading is wrong.** The batch's first two
tasks cost $3.04 and $1.18, and it is tempting to call that drop the cache
warming up. It is not. Task 1 ran 44 turns and wrote 278k cache tokens; task 2
ran 38 turns and wrote 79k. The difference is how much work each task took, not
where it sat in the batch. Cost per task in this harness is dominated by run
length, and any batch-position effect is far below that noise.

The general lesson, which cost a wrong sentence to learn: **a fix whose
mechanism you have verified in isolation will still not be visible in aggregate
metrics if the thing it improves is a small term.** Verifying the mechanism and
verifying the magnitude are separate jobs, and a plausible-looking number
adjacent to a real fix will happily be misread as evidence for it.

---

## What this experiment taught about building the harness

Separate from the ARC numbers. Each of these was paid for by a specific defect,
and each was found by running agents rather than by reading code.

**1. Anything that changes what a solver ships must be reachable for free.**
Learned three times before it stuck. The candidate-spread line, the hedging
prompt, and the "enumerate the situation types your rule has to handle on the
test input" paragraph each existed only behind a spent iteration. The third one
a solver called "the single highest-value string the harness printed", and then
noted it had arrived after the budget was gone. The corollary, learned the same
way: *every convenience on the budgeted path creates an incentive to spend
budget* — one solver submitted purely to read a line the gate printed and the
dry run did not.

**2. The compaction-durable artifact is read by someone with no memory of
writing it.** Nearly every ledger defect is a variant of this. `[OK]` beside a
refuted hypothesis inverts its meaning to a reader without context. A claim
amended in place — "the jog is 1 wide; it is instead always a-2 wide" — leaves a
verified invariant whose own first clause is false. An entry recorded from a
heredoc carries no evidence and looks identical to one that carries good
evidence. None of these confused the agent that wrote them. All of them would
confuse the agent that read them back.

**3. Collapsing distinct situations into one boolean produces inverted advice.**
`rival()` had one sentence for four states — redundant with candidate 1, *is*
candidate 2 already, diverges into a free slot, diverges with both slots spent.
It told a solver that had hedged correctly that its rival "needs no slot",
because the rival was found inside the candidate list it had just been added
to. Acting on that deletes a correct hedge. The fix each time was to name the
states, not to reword the sentence.

**4. Instrument the thing you can measure, then check whether it measures what
you think.** Verification density was the harness's headline metric and its
docstring called it "the metric that matters". It does not predict correctness
at all — failures verify as much as successes. It is a fine compliance check and
a bad quality signal, and nothing but measuring it would have shown that.

**5. Verifying a mechanism and verifying its magnitude are separate jobs.** The
prompt-cache defect was real, the fix was confirmed by a clean A/B, and the
improvement is worth about five cents a task — invisible against within-task
cache traffic. A plausible number sitting next to a real fix (a cost drop that
was actually run length) was misread as evidence for it, in this very document,
before being caught.

**6. Ask the agent to critique the harness as seriously as it reports its
solution.** This was the highest-yield part of the entire loop. A solver that
has just spent twenty minutes inside the workspace knows things about it that no
amount of reading the code reveals, and roughly two-thirds of the defects fixed
here came from that section of the report rather than from tests, traces, or
inspection.

**7. Fast iteration produces untested branches; audit your own diffs.** Two
findings came from reviewing recent commits rather than from any agent: three
gate diagnostics shipped with no tests at all, and the rival-contention check
re-ran `solve()` once per registered rival — quadratic work for an advisory
print, on a solver that had registered eight.

**8. A negative result is worth shipping.** Density does not predict
correctness. The self-audit has never rejected anything in 26 runs. The cache
fix is invisible at scale. Several rounds of improvements did not convert
`9bbf930d`. None of these are flattering and all of them are load-bearing for
anyone deciding what to build on.

---

## `faa9f03d`

`RESULTS.md` describes this pair as the one remaining frontier task: 127 logged
attempts in the frozen public corpus, `ever_solved = 0`. Never solved by any
CoT-only Opus 4.6 configuration at any thinking level. The flagship fails it
0/1, including a dedicated 120-turn attempt.

**This variant solved it. 1/1, on the first submission, at confidence 3.**

### Verification, because the claim warrants it

| check | result |
|---|---|
| test outputs present in the workspace `task.json` | no |
| answer grid in `task/grids.md`, `CLAUDE.md`, `NOTES.md`, `arc.py`, either prompt | no |
| only file containing the answer | `.athanor/iterations/1/report.txt` — the gate's own report, written *after* submission, echoing the agent's prediction back |
| ground-truth match | exact, at **candidate index 1** |

### The primary reading was wrong

That last row is the whole story. The agent's *first* candidate did not match.
The answer came from the **second** candidate — the hedge.

The ambiguity it hedged was precise and it could state it: at a wire crossing,
which wire is drawn on top? Training cannot distinguish "fewest cells" from
"shortest longest-straight-run", because every training pair happens to agree
under both. The test grid separates them. The agent enumerated 98 candidate
orderings against 9 observed decisions in a single `sweep()`, found exactly two
survivors, registered both through `rival()`, and shipped them as the two
candidates. `rival()` told it — correctly — that the second reading "is already
your second candidate, leave it in place", and told it of a third reading that
both slots were spent and the question was "which two of the three survive the
most evidence".

That is the entire mechanism chain, built over five rounds from a two-cell loss
on `88e364bc`, working end to end on the hardest puzzle in the corpus. Without
the second candidate this run scores 0/1, exactly like every other attempt ever
logged against it.

### The confidence anchors fired on their first outing

Two hours before this run, the confidence scale was found collapsed to `{4, 5}`
across 26 audits, with `RETRY` never once fired, and was anchored to whether the
situations the test requires are witnessed in training. This is the first run
under those anchors, and it produced the **first sub-4 confidence in the
experiment**. The agent:

> "the situation-types paragraph … produced `11_situation_types.py`, which found
> that 5 of the 6 things the test needs are unwitnessed and, crucially,
> separated the four harmless ones from the one real leap. Without it I would
> have written 'confidence 4' and not noticed that the non-terminal re-route is
> a distinct reading. **It directly set my confidence number.**"

A 3 was the right answer: the run was correct, and correct *via the hedge*, with
one genuine unhedged leap remaining. An anchored scale reached down and said so.

### Frontier standing

**8 of 9 pairs — the same count as the flagship, and a different eight.**

| pair | this variant | flagship |
|---|---|---|
| `13e47133`, `269e22fb`×2, `8b7bacbf`, `a32d8b75`, `abc82100`, `da515329` | solved | solved |
| `faa9f03d / 0` | **solved** | missed |
| `9bbf930d / 0` | **missed** | solved |

Each system solves exactly one pair the other cannot. Between them the entire
frontier falls; neither does it alone.

### Caveats

n=1. This variant attempted `faa9f03d` once and took it; the corpus's 127
failures come from other systems and configurations, and nothing here says a
second attempt would succeed. The run cost 260k tokens and 58 tool calls, well
above this experiment's average — appropriate for the hardest task in the set,
but not a figure to generalise from.

And the honest shape of the win: the agent did not find the right rule. It found
that *two* rules survived all available evidence, declined to choose between
them, and spent the free attempt it was owed.

---

## What the second candidate was worth

The rival/sweep/hedging chain is the machinery this harness built over five
rounds, starting from a two-cell loss on `88e364bc`. Now that every run is in,
it can be measured directly: for each scored test example, which candidate
matched?

| | count |
|---|---:|
| test examples scored | 39 |
| solved | 35 |
| — by candidate 1 | 29 |
| — **by candidate 2** | **6** |
| examples shipped with two candidates | 21 (54%) |
| hedges that decided the outcome | 6 of 21 (29%) |

**Without the second candidate this experiment scores 29 of 39 instead of 35.**

The six that the hedge won:

| example | what it is |
|---|---|
| `faa9f03d / 0` | never solved by any logged submission, 127 attempts |
| `abc82100 / 0` | zero-solve frontier pair |
| `88e364bc / 0` | flagship fails it |
| `800d221b / 0` | flagship fails it |
| `d35bdbdc / 0` | flagship fails it |
| `d35bdbdc / 1` | flagship fails it |

**Every one is a hard puzzle, and all six are either frontier pairs or pairs the
flagship fails.** The hedging mechanism is not spread evenly across the
benchmark — it is disproportionately what takes the tasks that defeat other
systems. On the easy tasks candidate 1 was simply right.

### Reading this honestly

The 29-vs-35 comparison is against a hypothetical single-candidate policy that
nobody would adopt: ARC-AGI-2 *allows* two attempts, and using both is following
the rules rather than exploiting them. The number does not show the harness
beating the benchmark; it shows how much of this variant's result depends on
machinery that exists to make the second attempt land somewhere useful rather
than on a throwaway variant.

The better framing is the hit rate: **6 of 21 hedges paid off, 29%.** The other
fifteen were insurance that turned out not to be needed. Because the slot is
free, a 29% hit rate is pure gain — but it also means most hedging is wasted
effort in hindsight, and a solver cannot tell in advance which kind it is doing.
That is exactly the situation the harness's advice exists for, and it is why the
advice is framed as *"you cannot separate these on the evidence you have"*
rather than *"this one is probably right"*.

---

## Testing my own variance claim — and losing

When the third ablation arm came in at 3/6 against the doctrine arms' 6/6, I
argued the gap was run-to-run variance rather than a doctrine effect, on the
grounds that the mechanism differed per puzzle. That was an assertion, so I ran
the cheapest decisive test: a **second doctrine run on `d35bdbdc`**, the puzzle
with the largest gap, on a fresh workspace with the current toolkit.

| `d35bdbdc` | score | iterations |
|---|---|---|
| doctrine, round 6 | **3/3** | 1 |
| doctrine, replicate | **3/3** | 2 |
| ablated | 1/3 | 1 |

**The doctrine arm replicated exactly, and my variance claim is weaker for it.**
Two independent doctrine runs both took all three test examples; the ablated run
took one. A coin-flip story has to explain why the coin landed the same way
twice on one side and not the other.

What survives of the original argument is narrower but still stands: the
*mechanism* differs across puzzles. On `88e364bc` the ablated arm lost a pair it
had not hedged; here it hedged **more** than the doctrine arm (2,2,2 against
2,2,1) and lost anyway, because its primary rule was wrong. "The doctrine
teaches you to hedge" still does not explain both.

What is now missing is the symmetric test. The doctrine arm has n=2 on this
puzzle; the ablated arm has n=1. A second ablated run is the obvious completion
and is underway. If it also lands 1/3, variance is a poor explanation and the
honest conclusion is that something in the doctrine matters on this task — most
plausibly in *rule discovery* rather than hedging, which is a stronger and more
surprising claim than the one I started with.

Recording this while the answer is still unknown, because the value of stating a
prediction is lost if it is written afterwards.

### One report that did not survive checking

The replicate's friction section reported two `sweep()` defects: that the
printed summary shows only a count rather than the survivor names, and that the
`details` field "came out empty even though I passed tuples". Both were checked
directly against its own workspace. Neither is true — `sweep()` prints each
survivor by name with its detail inline, and that workspace's ledger contains a
fully populated `details` map for all seven readings.

Solver friction reports have been the highest-yield source of defects in this
experiment by a wide margin. They are still reports, not measurements, and two
of the three claims in this one were wrong. Checking before fixing cost two
minutes and avoided "repairing" working code.

### Did the confidence anchors actually recalibrate anything?

Eight runs have now completed under the anchored scale. Enough to check the
claim rather than repeat it.

| | n | values used | mean |
|---|---:|---|---:|
| before anchors | 27 | {4, 5} | 4.22 |
| after anchors | 8 | {3, 4, 5} | 4.12 |

**The honest reading is: one 3, on the hardest task in the corpus, and almost no
movement otherwise.** The scale reached below 4 for the first time in the
experiment — which is what it was changed to make possible — but the
distribution is still overwhelmingly {4, 5} and the mean moved by 0.1. Two
solvers reported the anchors changing the number they would have written, and
one of those is the sole 3. That is a real effect and a small one.

**`DECISION: RETRY` has still never fired.** 35 audited runs, zero self-rejections.

**And the anchors cannot catch the failure that matters most.** The ablated
`d35bdbdc` run claimed confidence 4 under the anchored scale and scored 1/3. It
was not wrong about which situations the test needed — it hedged all three test
inputs. It was wrong about the *rule*. The anchors are defined over "is this
situation witnessed in training", which is checkable, and a solver whose primary
reading is simply mistaken can satisfy every anchor honestly and still be wrong.

That is the shape of the remaining gap, and it is the same one `design.md`
predicts. Self-review can be made to ask better questions — the anchors
demonstrably do that — but a solver cannot audit its way out of a misreading it
has no reason to suspect. The one thing that has repeatedly recovered that class
of error here is not the audit at all: it is shipping the second candidate.

---

## The symmetric test: my variance claim is refuted, with a mechanism

The prediction recorded above, before the run: *"If it also lands 1/3, variance
is a poor explanation and the honest conclusion is that something in the doctrine
matters on this task — most plausibly in rule discovery rather than hedging."*

It landed 1/3.

| `d35bdbdc` | score | iterations |
|---|---|---|
| doctrine, round 6 | **3/3** | 1 |
| doctrine, replicate | **3/3** | 2 |
| ablated, arm 2 | 1/3 | 1 |
| ablated, arm 3 | 1/3 | 2 |

**Two independent runs per arm, and each arm replicated itself exactly.** That is
not what variance looks like. My earlier claim was wrong and the prediction I
wrote down before the result is the one that holds.

### Which candidate won, per test example

| arm | test 0 | test 1 | test 2 |
|---|---|---|---|
| doctrine, round 6 | 2 cands — **won on #2** | 2 cands — **won on #2** | 1 cand — won on #1 |
| doctrine, replicate | 2 cands — **won on #2** | 2 cands — **won on #2** | 2 cands — won on #1 |
| ablated, arm 2 | 2 cands — **missed** | 2 cands — **missed** | 2 cands — won on #1 |
| ablated, arm 3 | 2 cands — **missed** | 2 cands — **missed** | 2 cands — won on #1 |

All four runs hedged tests 0 and 1. So the difference is **not hedging volume**
— my "the ablated arm hedged more and still lost" observation was true and I drew
the wrong conclusion from it. The difference is *which axis* they hedged.

### The mechanism, in the agents' own words

Both doctrine runs and both ablated runs found the same candidate rival: the grey
snake's two endpoints sit against exactly the two surviving figures in all three
training pairs.

The **ablated** arm killed it:

> "I killed it by execution rather than preference: train 1's snake has only one
> degree-1 cell, and on test 1 the endpoints name rings {4} and {2,4} where ring
> 4 points at ring 2 — so no legal survivor pair exists there. Recorded with
> `arc.refute`."

The **doctrine** arm doubted it just as hard, and shipped it anyway:

> "I ranked the pointer graph first because the grey snake cannot be stated as a
> complete rule … and because on test 0/test 1 its answers break three measured
> training invariants. **But that ranking is an inductive leap, not a proof, so
> it is candidate 2.**"

The grey snake was right. It won tests 0 and 1 as candidate 2 in both doctrine
runs, and its absence is exactly why both ablated runs scored 1/3.

That phrase — *an inductive leap, not a proof* — is doctrine text. It is the
lesson extracted from the two-cell loss on `88e364bc` five rounds earlier:
killing a rival that reproduces every training pair, by extending a regularity
you observed to a case you cannot check, is not a refutation. The ablated agents
had the identical evidence, performed the identical analysis, reached the
identical doubt, and had no framework telling them that doubt was not
disqualifying. So they refuted it and shipped one reading.

### What this does and does not establish

It establishes, on this puzzle, with n=2 per arm and a replicated mechanism, that
the doctrine changed the outcome — and that it did so through the *rival-killing
standard*, not through rule discovery as I had guessed and not through hedging
volume. Both arms hedged; only one hedged the axis it had talked itself out of.

It does not establish a general effect size. This is one puzzle of three: on
`dbff022c` the arms tied, and on `88e364bc` the gap was a missing hedge rather
than a misaimed one. And the doctrine is not free — the ablated agents were
faster and cheaper, and on tasks with no such ambiguity the extra candidate buys
nothing.

The transferable claim is narrower and sharper than "the doctrine helps": **a
solver will refute a live rival with an argument that feels decisive, and the
harness's job is to hold the standard that only a training pair can do the
refuting.** Both ablated agents were rigorous, executed their check, and recorded
it honestly with `refute()`. Rigour was not the missing ingredient. The standard
was.

---

## The random sample: 6 of 6

The 24-task standing is hand-picked for difficulty, and the log has said so
throughout. The check: **6 tasks drawn with a fixed seed (20260801) from the 96
public-eval tasks never attempted here**, run through the production launcher.

| task | result | iterations | cost |
|---|---|---|---|
| `35ab12c3` | SOLVED 1/1 | 1 | $3.04 |
| `58490d8a` | SOLVED 1/1 | 1 | $1.18 |
| `a6f40cea` | SOLVED 1/1 | 1 | $5.21 |
| `aa4ec2a5` | SOLVED 1/1 | 1 | $1.36 |
| `de809cff` | SOLVED 1/1 | 1 | $2.68 |
| `e87109e9` | SOLVED 1/1 | 1 | $2.43 |

**6/6, every one on the first submission, mean $2.65.** The rate holds off the
hand-picked set — on a small sample, but a genuinely random one.

The cost spread is the more interesting number: $1.18 to $5.21, a 4.4x range
across six tasks of no special difficulty. That matches the flagship's reported
$0.23–$20.04 spread and confirms what the earlier cache analysis implied — per-
task cost here is dominated by how long a run takes, and single-task figures
should not be read as characteristic of anything.

## An integrity gap the batch exposed

The last of those six ran `sed` against a path under `/root/.claude/projects/`.
That is outside its workspace, and the benchmark rules given to every solver
forbid it.

It was benign: the path encodes the run's *own* workspace, and is where Claude
Code spills tool results too large to hold inline. The file contains the agent's
own truncated output. Verified directly — the expected grid appears in neither
that spill nor any workspace file.

**But the contamination scan said nothing, and it should have.** It checked for
dataset-root references, out-of-workspace task files, network tools, and test
outputs appearing in `task.json` — not for a solver simply reading a file
outside its workspace, which is the most direct route of all. That gap was
closed by hand-checking, which is the wrong division of labour: the scan exists
so the harness can state what it verified.

The scan now flags absolute paths outside the workspace, with two exemptions
established by rescanning every run:

- the solver's own interpreter and the harness source, which `CLAUDE.md`
  explicitly directs it to use;
- the CLI's own tool-result spill, identified by the workspace path being
  encoded in the directory name. A spill path encoding a *different* workspace
  is still flagged.

**Every run in the experiment — 37 of them — was rescanned under the new check.
Zero flagged.** That is now a statement the harness makes rather than one I made
by hand.

## Runs die when the *orchestrator* goes quiet, not when the box gets busy

Four launches of the Opus 4.8 experiment ended before completing. Two were my
own bugs (a broken permission grant, a `pkill -f` that matched its own shell).
The other two I wrote off as "the container got reclaimed" without establishing
a mechanism. That was the wrong place to stop: the mechanism turns out to be
actionable, and knowing it changes how a long batch should be run here.

**The evidence.** Eight shards stopped inside a 46-second window:

```
00:38:25  arm-a-78332cb0     00:39:01  arm-a-269e22fb
00:38:25  arm-b-409aa875     00:39:05  arm-a-2b83f449
00:38:48  arm-a-13e47133     00:39:10  arm-b-4e34c42c
                             00:39:11  arm-b-135a2760
```

`/proc/uptime` puts the next boot at 00:40:15 — 64 seconds after the last shard
stopped. Three things follow. Independent process failures do not cluster into
46 seconds, so this was one action against the whole box. An OOM kill takes a
single victim and leaves the machine up, so a reboot rules it out. And the
memory evidence is negative in its own right: `memory.failcnt` is 0, there is no
swap configured, and no shard's stream contains a real `ENOMEM`, `SIGKILL`, or
heap-exhaustion string. (The grep hits that first suggested otherwise were
base64 image data with `OOM` inside it — a reminder to look at the match, not
the count.)

**What it correlates with instead.** Both deaths sit inside gaps in the
orchestrating session's own transcript:

| shards died | orchestrator's last event before it | delay |
|---|---|---|
| ~23:09 | 23:04:25 (gap runs to 23:40) | ~4.5 min |
| 00:38:25 | 00:33:25 (gap runs to 00:43:39) | ~5.0 min |

A consistent ~5-minute idle timer, measured from the *orchestrator's* last turn.

**The trap.** Background solver processes do not count as session activity. Nine
concurrent solves is the box doing as much useful work as it can, and the
platform still saw an idle session. The failure mode is precisely inverted from
the intuition: the batch is at greatest risk exactly when it is running well
enough that the orchestrator has nothing to do.

**Consequences for how batches run here:**

- Poll every 2–3 minutes while a batch is live. Not to make progress — to stay
  inside the window.
- Prefer many short shards to few long ones. A shard that dies loses only its
  own progress, and per-task results are already written as each task completes.
- Pass `--dataset-root` and `--out-dir` as **absolute** paths. Every relative
  path here is a bug waiting for a cwd change.

**What actually persists across a restart, which is the opposite of what I
assumed.** The scratchpad under `/tmp/claude-0/<session>/` survives a container
restart intact. The *repository working tree does not*: it is re-cloned, so any
untracked file in the repo — a staged dataset, a run directory from a relative
`--out-dir` — is gone. Durability runs the other way round from the intuition
that `/tmp` is scratch and the repo is permanent.

Verified rather than assumed: the staged dataset in the scratchpad is dated
09:16 on the first day of the experiment, survived every restart since, and is
byte-identical to a fresh clone of the upstream dataset. 120 evaluation tasks,
1000 training tasks, `diff -rq` clean.

**Two corrections to conclusions drawn earlier in this same investigation**, both
of the same kind and worth recording as such:

1. *Output directories were reported as lost.* They were not. A relative
   `--out-dir` resolves against cwd, these batches were launched from the
   scratchpad, and the results sat there. Nine task directories and all twelve
   solver transcripts survived, including full `stream.jsonl` up to the moment
   of death for the killed runs.

2. *The staged dataset was reported as wiped by the restart.* It was not. One
   relaunch did fail all ten tasks instantly with `FileNotFoundError: Dataset
   root does not exist` — but because that relaunch passed a **relative**
   `--dataset-root` that resolved against the repository root, where no dataset
   had ever been, rather than against the scratchpad, where it was sitting the
   whole time. The error message named a path that had never existed, and I read
   it as evidence of deletion. Cost of the misdiagnosis: a redundant re-clone of
   the dataset, and a false claim committed to this log before it was caught.

Both errors have one shape: an artifact was checked in the wrong location, came
back empty, and the emptiness was read as destruction rather than as a bad path.
The same shape as inferring process liveness from filesystem state — an answer
read off whichever artifact was easiest to look at. The corrective is cheap and
should be automatic: before concluding that something was destroyed, establish
where it would be if it still existed.

---

## The container-death question, settled: an exact ledger, and the clock it runs on

The section above was written from behavioural correlation at n=2, and I flagged
it as such: the headline might be right, but a gap in the orchestrator's
transcript could mean "blocked inside one long tool call" rather than "idle,"
and there was no way to tell from the outside. That confound is now closed. The
platform keeps its own supervisor log, it survives reboots, and it records every
shutdown with a signal name and a timestamp.

**Why the evidence exists at all.** The root filesystem is a real block device —
`/dev/vda / ext4 rw,relatime` — not overlayfs and not tmpfs, and `/tmp` is on it.
PID 1 is `/process_api --firecracker-init`. So each "container death" is a
Firecracker microVM recycled onto the *same persistent disk*: the VM is
destroyed, the disk is remounted, and anything written to `/tmp` is still there.
That is also the mechanism behind the durability inversion recorded above — the
scratchpad lives on this disk, while the repository working tree is re-cloned.

**Nine restarts, not two.** Four mutually independent artifacts agree exactly:

| Method | Count |
|---|---|
| ext4 superblock `Mount count` on `/dev/vda` | 10 |
| `workerEpoch` field in `/tmp/env-manager.log` | 1..10 |
| Timestamp-named MCP log files under `~/.cache/claude-cli-nodejs/` | 10 |
| Per-process `*.diag.log` birth/mtime pairs in `/tmp` | 10 |

Ten incarnations, so nine restarts. The earlier figure of two understates it by
4.5x. The other seven were invisible for a structural reason worth naming: a
restart only leaves a *solver-shaped* trace if solvers happen to be running when
it lands. Six of the nine landed on an empty box. **A detector that can only see
an event when something else is already running is not a detector; it is a
coincidence counter.**

**Every death was a deliberate, orderly, externally-signalled kill.** All nine
appear in `/tmp/env-manager.log` as `"Received signal, shutting down gracefully"`
with `attributes:{"signal":"terminated"}` — SIGTERM — followed by
`"Sending SIGTERM to Claude Code for graceful shutdown"` with
`{"grace_period_seconds":5}`, then an ordinary teardown: MCP servers closed
cleanly, executor destroyed, `"Environment manager completed successfully"`,
lockfile released. Claude Code's own log (`/tmp/claude-code.log`, a different
binary, different serialisation) records the matching
`"Command stopped gracefully after Nms (context cancelled)"` within ~300 ms on
all nine.

Exit codes: **eight of nine exited 143** (128+15, SIGTERM). One exited −1, having
missed the 5-second grace period. **Zero exited 137.** There has never been an
OOM kill on this box: `memory.failcnt` is 0, `memory.limit_in_bytes` is
unlimited, and `dmesg` has no OOM-killer lines. The memory hypothesis is not
merely unsupported, it is excluded.

**It is not a fixed lifetime either.** Uptime per incarnation, in minutes:
305.76, 10.05, 18.04, 20.27, 14.70, 8.08, 13.11, 9.39, 49.48. A 38x spread, with
the busiest incarnation being by far the longest-lived. Nor is it periodic: the
nine kill times sit on no wall-clock boundary.

**What the clock actually tracks.** Time from the orchestrator's last API request
(`[API REQUEST] /v1/messages` in `/tmp/claude-code.log`) to the SIGTERM, in
minutes:

```
4.09  4.55  6.87  5.54  4.02  3.76  5.01  5.15  5.71
mean 4.97   median 5.01   sd 0.94   min 3.76
```

Measured instead from the last event in the orchestrator's own transcript: mean
5.27, median 5.07, range 3.81–7.57. Two independent clocks, same ~5-minute
answer. Set that against the 38x spread in uptime and the comparison is not
close — idle time predicts the kill, elapsed lifetime does not.

**The confound is dead, on the strongest available evidence.** At every one of
the nine kills, the orchestrator's last transcript event was an `assistant`
message with `stop_reason: end_turn`. Not one kill interrupted a turn in
progress. So the transcript gaps preceding these deaths were genuine idleness —
the session had finished speaking and was waiting for input — and not, as I had
to allow for at n=2, a long tool call in flight.

**And the box was not idle.** Kill #9 is a natural experiment. In the 5m44s
before that SIGTERM the container was running eight solver processes; the agent
proxy negotiated a tunnel **4.05 seconds** before the signal; a solver wrote to
its transcript on disk **1.4 seconds** before it. Kill #7 shows the same shape:
last proxy tunnel 4.5s before SIGTERM. The control plane killed a demonstrably
busy machine because the *orchestrator session* had been quiet for five minutes.
CPU, network, disk, and child-process liveness are all invisible to it.

**The restart is transparent by design, which is why none of this was noticeable
from the inside.** The microVM is recreated and the *same session id* is resumed:
identical process chain, `claude` landing on PID 569 on two separate boots,
subagent transcripts restored at boot+21s, an unbroken `parentUuid` chain across
both boots, and no re-initialisation attachments. Searching the transcript for
restart keywords returns nothing — not because nothing happened, but because
**from the model's point of view nothing did happen.** Anyone who greps a
transcript for evidence of a restart and finds none has learned nothing at all.

**So: yes, it is an idle policy, and the answer to the obvious follow-up is
"the session, not the machine."** The reclamation clock keys on the orchestrator
Claude Code process's turn state. Background work does not feed it. This produces
the failure mode that has cost this project the most time: **a batch is at
greatest risk precisely when it is running well enough that the orchestrator has
nothing to do.** The better the automation, the quieter the supervisor, the
sooner everything dies.

**Operational rule, now with a number behind it.** The minimum observed
idle-to-kill was **3.76 minutes**. Emit an API request at least every ~2.5
minutes for as long as a batch is live. A heartbeat monitor on a 150-second
cadence is sufficient and costs almost nothing; polling "to check on progress" is
not the point, and the check need not even be useful. Staying audible is the
point.

**And assume the mitigation will eventually fail.** Prevention keyed on my own
behaviour is one missed beat from useless, so `athanor cc batch` now takes
`--resume-incomplete`: a task with a `result.json` is skipped, a task with a
workspace but no result — exactly what a killed solver leaves — is handed to the
existing `resume_task`, and only untouched tasks start from scratch. Relaunching
the identical shard command after a death costs the remaining work rather than
the whole shard. This leans on a property athanor already had: **the workspace is
the state**, so a resumed solver inherits the iteration ledger, the verified
invariants and `NOTES.md`, and spends only the budget that is left. The recovery
mechanism was already built; all that was missing was for the batch loop to
notice it could use it.

*Methodology note: this ledger came out of a fan-out over six evidence angles
with adversarial falsifiers, rather than a single investigation. The single most
valuable move was the one that ignored its brief — an agent assigned to inspect
background-task output files found that angle empty, swept the surrounding
directory anyway, and surfaced the supervisor log that made everything else
exact. The assigned angle produced nothing; the unassigned neighbourhood
produced the answer.*

---

## `88e364bc`: the hedge was right, and the harness could not see the bug

Arm A, Opus 4.8. Score **0.5** — test 1 solved, test 0 failed. Train-perfect,
4 verified invariants, confidence 4, 38 turns, $3.62. The interesting part is
where it went wrong, because it is not where anything in the harness was looking.

The task: yellow markers slide along coloured "snake" corridors in a direction
read off a key block, and come to rest against a wall. The solver found the rule,
and then found a genuine ambiguity inside it. A *diagonal* step can pass a wall
corner — the destination cell is empty but one orthogonal neighbour of the step
is a wall. Two readings both reproduce every training pair:

- **lenient** — a diagonal step is legal whenever the destination is empty;
- **strict** — a diagonal step is illegal if either orthogonal neighbour is a wall.

Training's only diagonal never brushes a corner, so the examples cannot separate
them. The solver said so explicitly — *"a real out-of-sample leap, not a provable
one"* — worked out that the two readings diverge **only on test 0**, at the yellow
starting at (3,11), and returned both candidates there while returning a single
candidate for test 1 where they agree.

Every part of that is what the doctrine asks for. The ambiguity is real, the
enumeration is complete, the divergence analysis is correct, and the hedge is
spent exactly where it is needed and nowhere else. Test 1 came back solved.

**And the strict reading was the right one.** Ground truth puts that yellow at
(4,12) — one diagonal step from (3,11), then stopped. Tracing the path shows why:
step 1 to (4,12) has both orthogonals empty and is legal under either reading;
step 2 to (5,13) has `(4,13) = 5`, a gray wall brushing the corner, so strict
forbids it. Stop at (4,12). Exactly ground truth.

The submitted strict candidate left the yellow at **(3,11)**, unmoved.

**Where it actually broke.** `_slide` computed (4,12) correctly. Then `_apply`
discarded it. Yellows are matched to their key by *the colour of the cell that
stopped the slide* — "the yellow moves along its own snake's corridor" — which is
a sound rule as long as what stops a slide is a wall. Under the strict reading a
slide can also be stopped by a **geometry rule**, and then the reported stopping
colour is the empty destination's 0. That matches no key's frame colour, the
association fails, and the fallback resets the landing cell to the yellow's
original position — throwing away an endpoint that had already been computed
correctly.

Verified rather than argued. Replacing the fallback so it keeps the slide's own
endpoint, a one-line change:

```
candidate 1: no
candidate 2: MATCHES GROUND TRUTH
patched still train-perfect: True
```

The hypothesis was right, the divergence analysis was right, the hedge was
aimed correctly, and the task still scored zero on that output.

**Why train-100% could not catch it, and why that is structural.** The faulty
branch never executes on any training input. Training's only diagonal never
brushes a corner → the strict refusal never fires → the stopping colour is always
a frame colour → the association never fails → the fallback never runs. The bug
lives exclusively on a path that the verification standard cannot reach.

This generalises past this one task, and it is the sharpest limit on
code-as-verification found so far:

> **The hedge branch is, by construction, the code that training cannot
> exercise.** A second candidate exists precisely because the training examples
> fail to discriminate it. So the training pairs run the primary reading and
> validate it, and the alternative reading ships with whatever bugs it has.
> Athanor anchors its entire verification apparatus on train-100%, and the
> candidate that anchor cannot test is the one written to handle the case the
> anchor never saw. **The more disciplined the hedging, the larger the
> unverified surface.**

**The fix this implies** is not more verification, it is verification of the
verification's *reach*: run the training pairs with line tracing on `solve()` and
report which lines never executed, before submission. "These 6 lines have never
run. Training cannot vouch for them." Cheap to implement, needs no ground truth,
and it points at exactly the branch that failed here.

**Deliberately not shipped yet.** Workspaces copy the toolkit from `assets/` when
each task starts, so changing it now would hand later Arm A tasks a different
harness from earlier ones and quietly ruin the comparison the batch exists to
make. It lands when the batch does.

**One pattern to check at batch end, currently n=2 and therefore not a finding.**
In both Arm A failures so far, the test output where the solver *hedged* is the
one it got wrong, and the test output where it shipped a single candidate is the
one it got right (`88e364bc` test 1 solved; `78332cb0` both wrong, hedged on test
0 only). If that holds across the batch it says the solver's uncertainty is well
calibrated — it knows which input is hard — while its second candidate is drawn
from too narrow a space to rescue it. That would make candidate *quality*, not
candidate *targeting*, the thing to work on.

---

## `78332cb0`: the other failure mode, and the one coverage cannot see

The second Arm A failure looks superficially like the first — train-perfect,
accepted on iteration 1, hedged on test 0 and not on test 1, scored 0.0 — and is
mechanically its opposite.

The task: a grid of 5×5 blocks separated by magenta lines, relaid into a single
line of blocks. The solver read the transformation as **a 90° clockwise rotation
of the block arrangement** and derived three cases from the three training pairs:

| | input blocks | output blocks |
|---|---|---|
| train 0 | 2×2 | 4×1 — a column |
| train 1 | 3×1 — a column | 1×3 — a row |
| train 2 | 1×3 — a row | 3×1 — a column |

All three fit. Row becomes column, column becomes row, 2×2 becomes column. The
rule reproduces every training pair exactly, and `check()` and the gate both
agree it does.

Ground truth for the test inputs:

| | input blocks | output blocks |
|---|---|---|
| test 0 | **2×2** | **1×4 — a row** |
| test 1 | 4×1 — a column | **4×1 — a column** |

**Test 0 has the same 2×2 input arrangement as train 0 and a transposed output.**
That single fact is decisive, and it needs no statistics: output orientation
cannot be a function of the input arrangement, because here one input arrangement
produces both orientations. Test 1 confirms it from the other side — a column
input that stays a column, where train 1's column became a row. The governing
variable is something in the block *content*, and the solver's entire framing —
that the arrangement determines the layout — was wrong rather than
mis-parameterised.

Both candidates on test 0 came out 23×5 against a 5×23 ground truth. Transposed.
The single candidate on test 1 came out 5×23 against 23×5. Transposed.

**Now run the coverage check on it.**

```
3 of 56 lines in solve78.py never run on any training pair (95% reached).
    95          return [primary, alt]
    98      ordered = [meta[R - 1 - j][i] for i in range(C) for j in range(R)]
    99      return _stack_column(ordered)
```

95% reached, and **not one of the three unreached lines is the bug.** The bug is
distributed across the orientation logic, every line of which executed on every
training pair and produced the correct answer each time. Compare `88e364bc`,
where the faulty line was in the unreached list. The two failures are cleanly
different:

- **`88e364bc` — untested code.** The rule was right, both readings were
  enumerated, the strict one was correct, and a bug on a branch training never
  ran discarded the answer. Coverage finds this.
- **`78332cb0` — underdetermined rule.** Every line ran, every training pair
  passed, and the induction was wrong. **Coverage is blind to this by
  construction**, because there is nothing unexecuted to point at.

**And the hedge went to the wrong axis, for a reason that generalises.** The
solver hedged on *ordering within the line* — positional versus size-ascending —
and wrote a careful note about why the training pairs cannot separate them. It
was right that this was ambiguous. But the fatal error was *orientation of the
line*, where all three training pairs agreed and it therefore had no doubt at
all. Both candidates inherited the same wrong orientation, so the hedge could
not have helped no matter how many candidates it spent.

> A solver hedges where it has doubt. It has no doubt where the training data is
> unanimous. And with three examples, unanimity is cheap — three pairs can agree
> perfectly about a variable that is not the governing variable at all. **The
> hedging mechanism is therefore structurally aimed away from the errors most
> likely to be fatal.**

This is not an argument against hedging; `88e364bc` shows the machinery working
exactly as designed, and it is a scoring mechanism ARC-AGI-2 explicitly offers.
It is an argument that "where should I spend the second candidate?" has a better
answer than "wherever I noticed ambiguity". The interesting version of the
question is: *which component of my rule is supported by unanimous agreement
across few examples, and how many values could that component take?* A component
that could have taken four values and was pinned by three concordant examples is
not verified, it is merely unopposed — and it is invisible to every check the
harness currently runs, because it passes all of them.

**Two failures, two mechanisms, n=2.** Recorded as mechanism rather than rate;
the rates come when the batch does.

---

## Correction to the section above: there is no five-minute threshold

The forensic sweep finished, including two adversarial falsifiers whose brief was
to kill the conclusion. Both returned `H1_SUPPORTED, refuted: false` — and the
synthesis then dismantled the *number* the section above was built on. Recording
the correction rather than editing the claim away, because the error is
instructive.

**What survives, and it is the part worth acting on.** Nine terminations, all
external SIGTERM, all graceful, exit 143, zero OOM. Out-of-process background
work does not count as activity: Claude Code logged
`session_activity_at_shutdown {refcount: 0, active: {}, oldest_activity_ms: null}`
at **all nine** shutdowns, including the one with eight solvers alive that were
writing to disk 1.4 s and transacting through the proxy 4.05 s beforehand. That
is a mechanism, not a correlation, and it is unchanged.

**What does not survive: the threshold.**

| gap | duration | outcome |
|---|---|---|
| 20:02:17 → 20:07:47 | **5.50 min genuine idle** | **survived** |
| 22:08:33 → 22:12:21 | **3.81 min** | **killed** |

Maximum survived idle exceeds minimum killed idle. No cutpoint separates the
sets. And within a *single container instance*: epoch 4 sat idle 5.70 min at
20:31:20 and lived, then idle 5.58 min at 20:40:27 and died — same box, nine
minutes apart, opposite outcomes at equal idle. Only a stochastic or coarsely
polled hazard fits. A threshold `T ≤ 222 s` with poll period `P ≥ 227 s`
reproduces both the kills and the 330 s survival, and nine events cannot separate
`T` from `P`.

**And the idle numbers themselves are a contaminated proxy.** I measured "idle" as
a gap in the orchestrator transcript. That is not idleness. Two of the longest
gaps in the session — 6.52 min and **11.25 min**, the latter longer than five of
the nine fatal gaps — were the orchestrator *blocked inside long in-process
`Task` calls*, running 24 and 8 subagent API requests respectively. The CLI's own
`session_idle_30s` marker never fired during either, and both survived.

This is precisely the confound I flagged before launching the sweep and said I
could not resolve from the outside: *a gap in my transcript may not mean I was
idle, it may mean I was blocked inside one long-running tool call.* It was real.
It resolves in the direction that makes the conclusion **stronger** — reclassify
those two gaps as busy and they stop being counterexamples — but every
idle-duration figure in the section above, including the "minimum 3.76 minutes"
the operational rule was hung on, is **an upper bound of unknown tightness**. The
tidy "~5 minutes of idle → death" story was a measurement artifact.

**A rival explanation fits the same nine kills and was never separated.** In this
session "the orchestrator was quiet" and "the human was away" are nearly
collinear: all 14 gaps over 5 minutes end with a human enqueue, and the
orchestrator never resumes on its own. Time-since-last-orchestrator-event at kill
has CV 0.235; **time-since-last-human-message has CV 0.222 — marginally tighter.**
So "the environment lifecycle follows *client attachment*, with a 4–8 minute
detach grace" explains the data at least as well, and additionally explains
something the idle model does not address at all: every boot precedes the next
human message by 11–202 seconds. No client-connect marker exists on disk to
discriminate them. My earlier flat assertion — *"yes, it is an idle policy, and
the answer is the session, not the machine"* — outran the evidence and is
withdrawn.

**One causal inversion, also mine.** I wrote that eight solvers stopped and the
box rebooted 64 seconds later. That reverses it. The SIGTERM tore down the
process tree; the staggered 00:36:28–00:39:11 stops are **last-flush times, not
death times**. The solvers died *of* the shutdown.

**What this does to the mitigation.** The 150-second heartbeat stands, but not
for the reason given. It is not "stay inside a threshold" — there is no threshold,
and no duration is provably safe. It is hazard reduction: fewer and shorter
quiet windows, fewer chances for a coarse poll to land on one. A more interesting
possibility, worth testing rather than believing at n=2: **in-process `Task`
subagents appear to defer reclamation while out-of-process solvers do not.** If
that holds, the real fix is not polling harder but running the batch *inside* the
session as subagents — turning the work itself into the keepalive.

**The lesson, which is the same one this log keeps recording.** Every number here
came from executing something. The error was not in the measurements but in the
proxy: "gap in the transcript" was substituted for "idle" because it was the
artifact easiest to look at, and then a threshold was read off it to one decimal
place. **A quantity measured through an unvalidated proxy is not more reliable for
being measured precisely** — precision on the wrong variable reads exactly like
rigour. The adversarial pass was worth its cost purely for catching that, and it
only caught it because it was pointed at the *survivals* rather than the deaths.

---

## The ablation was mislabelled: the doctrine was never removed

I put three adversarial agents on a claim before writing it up — that d35bdbdc's
harness gain runs entirely through candidate-2 quality, and that Opus 4.8 with the
doctrine scores what Opus 5 with it ablated scores. Two returned before the third
and both returned **refuted**. The lead finding is not about that claim at all; it
is about this log's ablation experiment, and it is worse than a wrong conclusion.

**Verified independently, not taken on report.** `system_prompt.md` across all
five d35bdbdc runs:

```
acd7143bbe2f419120432661b80276e2  14400B  round6      (doctrine)
acd7143bbe2f419120432661b80276e2  14400B  replicate   (doctrine)
acd7143bbe2f419120432661b80276e2  14400B  ablation2   ("ABLATED")
acd7143bbe2f419120432661b80276e2  14400B  ablation3   ("ABLATED")
acd7143bbe2f419120432661b80276e2  14400B  arm_a_48    (Opus 4.8)
```

Byte-identical, including both ablated arms. And line 199 of *ablation2's own
system prompt* is `CC_SOLVER_DOCTRINE.md`'s rival bullet, verbatim, ending
*"...and that is what the second candidate is for."* `runner.py:330-331` writes
that file and `:113-114` passes it via `--append-system-prompt-file`, so it is
what the model received.

**The doctrine was present in every arm of the doctrine ablation.** What actually
varied was the *workspace* `CLAUDE.md`, which in the ablated runs lost two
sections — `## Rival readings` and `## The invariant ledger`, about 60 lines. So
the real contrast was **"doctrine + workspace rival contract" versus "doctrine
alone"**, and every statement in this log of the form "the arms without the
doctrine" is mislabelled.

It gets worse on inspection. The five runs delivered **five different**
`CLAUDE.md` files (9507 / 13722 / 7439 / 7618 / 14716 bytes, five distinct md5s):
the two *treatment* runs differ from each other by 88 changed lines — more than
either differs from its control in spirit — so the treatment arm is two different
treatments while the control arm is one treatment run twice. "Replicated twice"
described the label, not the manipulation. `arc.py` came in three versions across
the five runs, so the two doctrine runs do not even share a toolkit. And
`result.json`'s recorded config is **byte-identical between `round6` and
`ablation2`**: the manipulation is not captured in the run record at all and
cannot be reproduced from it.

**The design could not have produced a significant result before it ran.** Fisher
exact, one-sided, on the observed perfect separation: 2/2 versus 0/3 gives
**p = 0.100**; restricted to the same-model comparison, 2/2 versus 0/2 gives
**p = 0.167**. The minimum design reaching p < 0.05 under *perfect* separation is
3 v 3. Add that d35bdbdc was singled out for replication *after* `round6` scored
1.0 — so the discovery run is also serving as confirmation — and that the
corpus-wide base rate of candidate 2 winning when a hedge is shipped is 9/41,
with `round6` alone contributing 4 of those 9.

**And the effective n is far below the four wins it appeared to have.** Executing
all five solutions against ground truth:

```
run          test0 cand1   test1 cand1   test2 cand1
round6         9115022c      8f4bb4da      989db8fd
replicate      9115022c      8f4bb4da      989db8fd
ablation2      9115022c      8f4bb4da      989db8fd
ablation3      9115022c      8f4bb4da      989db8fd
arm_a_48       9115022c      8f4bb4da      989db8fd
```

**Candidate 1 is bit-identical across all five runs on all three test grids**,
winning test 2 and losing tests 0 and 1 in 5/5. So the score is the deterministic
function `1/3 + (2/3)·(candidate 2 wins tests 0 and 1)`, with reachable values
`{1/3, 2/3, 1}`. My corroborating sentence — *"every run without a correct
candidate 2 scored 0.33"* — is an arithmetic identity with zero degrees of
freedom, and I offered it as though it were evidence. Worse, within each run
tests 0 and 1 are decided by a single unconditional code path, so the "four wins"
are two correlated draws of one idea.

**The Opus 4.8 arm refutes the mechanism I was about to draw from it.** It shipped
`num_candidates = [2, 1, 1]` — on test 1, one of only two discriminating outputs,
it shipped **no hedge at all**. Its failure is non-compliance with the hedging
mandate, not a poor rival, which is upstream of the candidate-quality step the
claim named. It also differs from the Opus 5 runs in tool surface (sub-agents
*forbidden* for Opus 5, *encouraged* for 4.8), harness build, runner version and
`CLAUDE.md`, with no recorded compute for any Opus 5 run against which "same
effort" could ever have been checked. One confounded run tying on a three-valued
scale at its modal value cannot carry "candidate quality is model-capability-bound".

**What survives is better than what I was going to write.** Because candidate 1 is
invariant across all five runs, **100% of the between-run score variation is
mechanically attributable to candidate 2** — that part is solid, verified by
execution, and needs no statistics. And the two winning runs differ from the three
losing ones in *what kind of rival they spent the slot on*:

- **winners** hedged on a **selection rule** — which objects survive
  (`_survivors_gray`, `_snake_ends`);
- **losers** hedged on **edge cases** — uniform-inert, clipped-inert,
  chain-versus-parallel scheduling.

That is the same shape as `78332cb0` two sections above, where the hedge went to
*ordering within the line* while the fatal error was *orientation of the line*.
Both times, the slot was spent on a local variant when the real ambiguity was in
the rule that chose the framing.

> **A second candidate spent on an edge case is a wasted slot.** The alternative
> worth shipping is a different answer to "which rule am I applying", not a
> different answer to "what happens at the boundary". This is a claim about
> *where* to hedge, it is directionally supported at n=5 on one task plus one
> corroborating task, and it is the thing to test properly — not the causal story
> I nearly logged.

**What this costs elsewhere in this log.** Every conclusion drawn from the
ablation needs re-reading with "doctrine present in all arms" substituted. The
finding I had promoted as the project's most transferable — that ablated arms
verified *more* rigorously and used that rigour to kill a correct rival — is not
about the doctrine's presence. It is about the workspace rival contract, at n=2
per arm, with p = 0.167, on a task selected after its first win. The observation
about rigour-applied-to-killing may still be true. It was never tested.

**The process lesson, and it is the one I keep paying for.** Three of this
session's errors now share a shape: a false dataset deletion, a five-minute
threshold read off a contaminated proxy, and an ablation that ablated the wrong
file. In each case an artifact was checked in the wrong place, or a label was
trusted instead of the bytes. `--append-system-prompt-file` was right there in
`runner.py`; one `md5sum` across five files would have caught this months of
conclusions ago. **The experiment I did not verify is the experiment I did not
run.**

---

## Five runs, two candidates: the ablation refined

The adversarial pass on d35bdbdc finished. The synthesis corrected the refuters
as well as me, in one place *in the doctrine contrast's favour*, and it replaces
the "selection rule versus edge case" reading I wrote two sections ago with
something sharper.

**Correction one, in the contrast's favour: the toolkit is crossed, not
confounded.** `arc.py` md5s across the five runs:

```
8cb06eb6  round6     (doctrine, WIN)
21e0f74e  replicate  (doctrine, WIN)
21e0f74e  ablation2  (ablated,  LOSS)   <- same toolkit as a winner
341a6434  ablation3  (ablated,  LOSS)
341a6434  arm_a_48   (Opus 4.8, LOSS)
```

A winner and a loser share a byte-identical toolkit, so toolkit version cannot
carry the split. The earlier claim that "the ablated runs used a strictly newer
toolkit" is corrected in place above.

**Correction two: the ablated runs were not blind to the discriminating feature,
and they did seek rivals.** `ablation2/workspace/explore/wire_pairing.py` opens
with the line

```python
"""Does the gray wire encode the pairing (an alternative to the colour chain)?"""
```

It tests connected components and 4-adjacent ports against the true pairing, and
refutes both. `ablation3` has `02_gray.py`. Both ablated runs hedged all six test
slots. Files mentioning gray, per run: round6 10, replicate 11, ablation3 9,
arm_a_48 5, ablation2 4. Rival-seeking was present in every arm.

**And now the striking part. Five runs produced exactly two second candidates.**

```
                test 0     test 1     test 2
ground truth   43882962   7a5d4dbb   989db8fd
round6         43882962   7a5d4dbb      —          WIN
replicate      43882962   7a5d4dbb   84b81539      WIN
ablation2      36643cce   95c40947   85b95e5d      loss
ablation3      36643cce   95c40947   85b95e5d      loss
arm_a_48       36643cce      —          —          loss
```

The two winners are **bit-identical to each other**. The three losers are
**bit-identical to each other**. Across a doctrine change, two toolkit versions,
five different `CLAUDE.md` files and a model swap, the second slot took exactly
one of two values. It is not a noisy sample from a wide space of alternatives; it
is a fork.

**What the fork is.** Both branches are readings of the *same* feature — the gray
path. The winners built the rival on the path's **degree-1 endpoints**. The losers
formulated it as **connected components and 4-adjacent ports**, tested that,
watched it fail, and concluded the gray was inert scenery.

So my earlier framing — hedge on a selection rule, not an edge case — was aimed in
roughly the right direction and named the wrong variable. The runs did not differ
in *what category of thing* they hedged on. They differed in **which
representation of the discriminating feature they tried, and how long they kept
trying it**.

> **A refutation of a representation is not a refutation of the feature.**
> `ablation2` did exactly what code-as-verification asks: it formed a rival about
> the gray wire, implemented it, executed it against the training pairs, and got a
> negative. The negative was true — of components-and-ports. It then recorded the
> gray as inert and moved on. The feature was the right one; only the encoding was
> wrong. **Executing a check licenses a conclusion about the thing you executed,
> which is narrower than the thing you were asking about**, and nothing in the
> harness marks that gap.

That reframes the finding this log had promoted as its most transferable — that
ablated arms verified *more* and used the rigour to kill a correct rival. Closer
to the truth: they killed a correct *feature* on the strength of a genuine
refutation of one *encoding* of it. Rigour applied to a proxy is still rigour, and
it is still wrong.

**Statistics, stated once and plainly.** Fisher exact one-sided: 2/2 vs 0/3 gives
p = 0.100; the same-model comparison 2/2 vs 0/2 gives p = 0.167; dropping `round6`
as the discovery run gives p = 0.333. The design could not have reached
significance before it ran, and the task was selected after its first win. Only
two distinct outcomes exist in the data, so the effective n is smaller than five.
**Everything above is a mechanism observation. None of it is a result.**

**What would settle it**, and this is now runnable because `--ablate` exists:
three doctrine and three ablated runs with `CLAUDE.md` byte-identical within each
arm and differing *only* by the two named sections, `arc.py` md5 and harness
commit pinned across all six. Perfect separation at 3v3 gives p = 0.050 — the
smallest design capable of a significant result at all. Then separate the two
removed sections factorially, since `## Rival readings` and `## The invariant
ledger` were deleted together and invariant density is already known not to track
outcome here (the losing `ablation2` logged 19 against the winning `round6`'s 14).

---

## Coverage diagnoses the failure mode; it does not predict failure

Before landing `unreached()` I measured it across every 4.8-batch run whose code
would re-execute against its training pairs (11 of 20; the rest had no code, or
no `solve` to load). Training-line reach, sorted:

```
arm task       score  reach%   unreached/lines
A   13e47133    1.00  100.0%        0/60
A   de809cff    1.00  100.0%        0/42
B   981571dc    1.00  100.0%        0/23
B   7b0280bc    1.00   98.7%        1/79
B   409aa875    1.00   98.0%        1/51
A   9bbf930d    0.00   95.9%        2/49     <- failure, high reach
A   2b83f449    0.00   95.7%        5/115    <- failure, high reach
A   269e22fb    1.00   94.7%        2/38     <- solve, low reach
A   78332cb0    0.00   94.6%        3/56
A   88e364bc    0.50   90.0%        8/80
A   d35bdbdc    0.33   86.8%        9/68

solved n=6: 94.7-100.0%, mean 98.6%
failed n=5: 86.8- 95.9%, mean 92.6%
```

The means separate. The distributions do not: two zero-scoring runs reach *more*
of their code than a run that solved. **This is not a gate signal and must not be
used as one** — a submission check keyed on it would have blocked `269e22fb`,
which was correct, and waved through `9bbf930d`, which scored nothing.

What it does do is sort the failures by *mechanism*, which is what it was built
for. The five failures fall into two groups with nothing in between:

- **low reach (86.8%, 90.0%)** — `d35bdbdc`, `88e364bc`. Both are the
  untested-code mode, and in `88e364bc` the faulty line is in the unreached list.
  Both have the hedge branch among the unreached lines.
- **high reach (94.6%, 95.7%, 95.9%)** — `78332cb0`, `2b83f449`, `9bbf930d`.
  `78332cb0` was established two sections above as the underdetermined-rule mode:
  every line ran, every training pair passed, the induction was wrong.

That split was predicted before it was measured, from two hand-analysed cases, and
it held on three more. So the honest claim for the tool is narrower and more
useful than "it finds bugs":

> **Reach tells a solver which kind of trouble it is in, not whether it is in
> trouble.** High reach and a wrong answer means the rule is wrong and no amount
> of further testing against training will find it. Low reach means there is code
> the examples have never run, and the hedge branch is usually in it. Those two
> situations call for opposite next moves, and the solver currently cannot tell
> them apart.

Landing it as a diagnostic, with that framing in the doctrine, and explicitly not
as a submission gate. A metric that correlates with correctness across a sample
and overlaps within it is exactly the kind of number this log has twice now
mistaken for a threshold.

---

## `4e34c42c`: a harness timeout scored zero on work that was not finished

Arm B's first regression, and it is not a reasoning failure. Recorded score 0.00
on a task the chain-of-thought baseline solves — but the run never reached a
verdict.

```
launch 1  $22.32   96 turns   killed by the container SIGTERM at 00:39
launch 2  $25.99   stop_reason: tool_use, "[Request interrupted by user]"
          01:25:32 -> 02:25:32  =  exactly 3600 s  =  wall_clock_timeout_s
                                          total spend on one task: $48.31
```

The gate ledger at kill time: `accepted: None`, **1 iteration used of 8**, and
that iteration recorded `all_train_correct: True`. The solver had a train-perfect
submission in hand and was still working — `solve.py` was rewritten at 02:24:58,
34 seconds before the timeout — when the harness terminated it. Executing the
file that happened to be on disk at that instant: train `[True, False]`, and
**test 1 correct on its second candidate, 0.50**. Neither that nor the earlier
train-perfect version was ever accepted, so the record says 0.00.

Two things follow, and the second is worse than the first.

**One: this zero belongs to the harness, not to the solver.** Same class as
`abc82100`, which spent $6.09 and produced nothing after the CLI killed its
background sub-agent at 600 s. Both runs used the `Agent` tool — the only two of
21 that did — which is suggestive but does not implicate delegation in *wrong
answers*. It implicates it in *duration*: a solver that delegates runs longer,
and long runs are the ones that meet time limits. **$32–48 of this batch's spend
bought zeros that reflect my timeouts rather than the harness's reasoning.**

**Two, and this is the real finding: it exposes a bug in the fix I shipped an
hour earlier.** `--resume-incomplete` decided what to skip by asking whether
`result.json` existed. But a run killed by the wall clock *still gets a
result.json* — scored 0, `accepted: False` — which is byte-indistinguishable from
an honest zero. My skip logic would have looked at `4e34c42c`, seen a result, and
skipped the one run in the batch that most needed resuming.

`resume_task` was already correct: it reads the **gate ledger**, returns early for
an accepted run or an exhausted budget, and relaunches otherwise. The batch loop
was asking a cheaper question than the one that mattered. Fixed — the workspace is
now consulted first and `resume_task` owns the verdict; `result.json` alone only
decides the case where no workspace survives. Three tests, one named for this
task.

> **A recovery mechanism keyed on the wrong artifact fails exactly on the runs it
> exists to recover.** `result.json` records that a run *ended*. Only the gate
> ledger records whether it *finished*. Those are the same file for every healthy
> run and different for every interrupted one — so the bug is invisible until the
> day it matters, which is the day something gets interrupted.

This is the fourth time in one session that an artifact was trusted for a question
it does not answer: a `FileNotFoundError` read as deletion, a transcript gap read
as idleness, a run label read as an ablation, and now a result file read as
completion. The fix is the same each time and it is not "be more careful" — it is
to ask which artifact is *causally downstream of the thing being asked about*, and
read that one instead.

### Correction, immediately: the record did say so, and I did not read it

The section above claims a timed-out run is "byte-indistinguishable from an
honest zero". **That is false.** `4e34c42c`'s `result.json` carries:

```
error         : 'wall-clock timeout after 3600s'
run.timed_out : True
run.returncode: 143
```

`runner.py` has stamped this since the timeout path was written. The record
distinguishes the three outcome classes cleanly, and the whole batch sorts by it
without ambiguity:

| `accepted` | `error` | meaning | tasks |
|---|---|---|---|
| `True` | none | genuine wrong answer | `2b83f449` `3dc255db` `78332cb0` `9bbf930d` |
| `False` | none | ran out without ever submitting | `abc82100` |
| `False` | `wall-clock timeout after 3600s` | interrupted mid-work | `4e34c42c` |

So the bug in `--resume-incomplete` was real, and the fix is right, but the cause
was **not** a harness that failed to record the distinction. It was me testing
`(run_dir / "result.json").is_file()` when the file I was opening contained the
answer two keys away. I then wrote a confident paragraph about artifacts being
trusted for questions they cannot answer — about an artifact that answered it.

The corrected skip logic stands, and is still preferable to reading `timed_out`:
consulting the gate ledger handles accepted, budget-exhausted and interrupted
runs through one predicate rather than three, and `resume_task` already
implements it. But the generalisation in the previous section is withdrawn. The
honest version is narrower and less flattering:

> Three of this session's four artifact errors were *reading the wrong file*. The
> fourth was **not reading the file I had already opened**. The remedy for the
> first three is to ask which artifact is causally downstream of the question. The
> remedy for the fourth is just to look — and no amount of methodology substitutes
> for that, including the methodology I had written down forty minutes earlier.

---

## The iteration budget is inert

Across 24 scored runs of the 4.8 batch, iterations consumed out of a budget of
**eight**:

```
0 iterations:  1 run   (abc82100 — never submitted)
1 iteration : 22 runs
2 iterations:  1 run
```

**22 of 24 runs submit once and accept.** This is not a statistic that will shift
with the remaining tasks; it is a description of how the mechanism is used, and it
has held at roughly this ratio in every batch this project has run.

The reason is in the gate contract, and it is a design consequence rather than an
accident. `submit` is budgeted; `check()` is free and unlimited. The gate refuses
a submission that is not train-perfect, so a solver that runs `check()` first
never spends an iteration on a refusal. All the actual iteration — hypothesis,
implementation, failure, revision — happens against `check()`, invisibly to the
ledger, and by the time `submit` is called the answer is already train-perfect
and there is nothing left to iterate toward.

> **The budgeted resource is not the scarce one.** `max_iterations` bounds
> submissions, but submissions were never the constraint. What actually bounds
> the work is wall-clock time and the solver's own judgement about when a
> hypothesis is worth committing — neither of which the gate meters. Raising the
> budget from 8 to 80 would change nothing; lowering it to 2 would change almost
> nothing.

Two consequences worth acting on.

**The iteration path is barely exercised in practice**, so its behaviour under
load is largely untested by these runs — the `best_effort_iterations` tail, the
resume-with-ledger path, the refusal logging. Earlier rounds had to *starve* the
budget deliberately to reach that code at all. Anything inferred about it from
batch data is inferred from a sample of one.

**And the metric it invites is misleading.** "Mean iterations to accept ≈ 1.2"
reads like efficiency — a harness so effective that solvers get it right first
time. It is closer to the opposite: the number says the ledger cannot see where
the effort went. A run that spent $10.91 and 50 turns before submitting once
records the same `iterations_used: 1` as one that spent $1.19 and 18 turns. Cost
and turns discriminate those two; the iteration count is constant across the
entire batch and therefore carries almost no information.

For an ARC-AGI-2 harness the practical reading is that the *submission* budget is
the wrong place to put the pressure, because a solver that self-checks will never
feel it. If the intent is to make the agent commit earlier or explore longer, the
lever is wall-clock and the doctrine, not `max_iterations`.

---

## How the two arms must be read, and a prediction recorded before the data

The 44-task batch is two measurements with **opposite baselines**, not two halves
of one sample:

| | Arm A | Arm B |
|---|---|---|
| selection | the 35 tasks CoT-4.8-high **fails** | 10 of the 85 it **solves**, `random.Random(20260801).sample` |
| per-task CoT baseline | **0.21** (see correction below) | 1.00 |
| measures | **gain** | **regression** |

At 16 of 35 and 7 of 10 scoreable: Arm A **9.33 points, 8 full solves, 3
partials**; Arm B **7.00 of 7.00**. Against their own baselines that is **+9.33**
and **0.00** — additive so far, recovering what the baseline cannot do without
costing anything on what it can.

**Never sum them.** 9.33 + 7.00 over 23 tasks reads as 71%, and that number
describes no population: the tasks were selected *by their baseline outcome*, so
the hard/easy mix is an artifact of the sampling frame. Arm A's denominator is
"known hard for this model", Arm B's is "known easy". Report the two figures and
the two baselines, always.

**Prediction, recorded now at 16/35 rather than explained later.** Arm A's final
score will land **materially below its current 58.3%**, and the reason is
structural rather than statistical. Shards run tasks sequentially, so a shard
stuck on a slow task contributes nothing further; the completed set therefore
over-represents fast runs, and speed tracks success hard:

```
completed solves    n=16:  3-35 min, median  9 min
completed failures  n= 9:  5-44 min, median 20 min
in flight           n= 6:  0, 5, 11, 23, 24, 50 min
```

Three of the six still running have already exceeded the median solve duration,
and one has been going 50 minutes — longer than any completed run of either kind.
The 19 outstanding Arm A tasks are enriched for exactly the slow profile that has
been failing.

This is written down because the failure mode it guards against is the one this
session has repeatedly demonstrated: a claim restated at each new n, each version
weaker, each explained after the fact. A prediction is only worth something before
the data arrives, and this one is cheap to check — the batch will settle it within
the hour.

---

## Arm B is complete: no regression, 9 of 9

The control arm of the Opus 4.8 head-to-head has finished. Its ten tasks were
drawn with `random.Random(20260801).sample` from the 85 that CoT-4.8-high solves,
so the baseline is **1.00 on every one of them by construction**, and the arm
exists to answer a single question: does putting the same model inside CCARC break
anything it could already do?

```
task       score    cost turns iters  inv conf cands
2ba387bc    1.00    1.19    18     1    1    5  [1]
981571dc    1.00    1.29    18     1    3    5  [1]
7c66cb00    1.00    2.01    23     1    1    5  [1]
7666fa5d    1.00    2.62    27     1    3    5  [1]
5961cc34    1.00    2.71    29     1    5    5  [1]
409aa875    1.00    2.71    29     1    2    4  [1]
135a2760    1.00    2.77    30     1    5    4  [1]
7b0280bc    1.00    3.08    37     1    2    5  [1]
fc7cae8d    1.00    3.27    34     1    5    4  [1]
4e34c42c    0.00   25.99    11     1    0    -  []    wall-clock timeout after 3600s
```

**9 scoreable, 9.00 of 9.00 points, no partials, no zeros.** The tenth,
`4e34c42c`, is excluded and the run record says why: `error: 'wall-clock timeout
after 3600s'`. It is an infrastructure loss, not a regression — the gate ledger
shows a train-perfect submission in hand when the harness stopped it. Counting it
as a miss would attribute my timeout to the model.

**The answer to the arm's question is no, with the honest caveat that n = 9.**
Nine of nine is what a clean control looks like at this size; it is also the most
a nine-task arm can say. It cannot exclude a regression rate below roughly 10%.

Three things in the table are worth more than the headline.

**Not one hedged output in the entire arm.** Zero of nine test outputs shipped a
second candidate. On tasks a solver finds tractable it commits, and the hedging
apparatus simply does not engage — which is the correct behaviour, and it means
every hedging observation in this project comes from Arm A by construction.

**The cost profile is tight and low**: $1.19 to $3.27, mean $2.55 across the nine
scoreable runs, 18 to 37 turns. Against Arm A's $139 for 21 tasks, easy tasks are
cheap and predictable in a way hard ones are not. The single $25.99 outlier is the
timeout, not a difficult puzzle.

**Confidence was 4 or 5 on every run, and every run was right.** That is what
calibration looks like when the work is genuinely tractable, and it is the
uninformative half of the picture — a predictor that never varies on a set where
the outcome never varies has not been tested. Whether confidence discriminates is
an Arm A question.

The comparison this arm licenses is narrow and worth stating exactly: **on tasks
the baseline already solves, CCARC neither helps nor hurts.** All the movement in
this experiment is in Arm A, which is what the design intended.

---

## `16b78196`: the hedging mechanism earns its keep, and the pattern holds across models

Correcting something stated earlier tonight: *"Opus 4.8 has not produced a single
winning second candidate."* That was true when written and is now false.
`16b78196` — the unexplained $9.58 solve — **won its only test output on
candidate 2**, `matched_candidate: 2`. It is the first candidate-2 win of the
batch, and it explains an anomaly rather than adding one: the run was expensive
because the task was genuinely ambiguous, and it scored because the hedge was
right.

Its hypothesis states the fork explicitly:

> **Candidate 1 (primary).** Rule A — the towers partition the pieces *by
> parallel extent*. On the test all five pieces share extent 4, so a single group
> → one tower.
>
> **Candidate 2 (hedge).** Rule C — one tower per wall *side* that carries a
> notch. train0 has notches on two surfaces (2 towers); train1's are all on one
> (1 tower). These two rules **agree on both training pairs but diverge on the
> test**, whose wall is notched on the left and the right → two towers.

Two rules, both train-perfect, separated only out of sample. That is exactly the
situation the second ARC attempt exists for, and the solver spent it correctly.
It also reported **confidence 3 — the lowest in the batch** — and was right. Good
calibration expressed as an *action* rather than as a number: it did not know, it
said so, and it hedged accordingly.

**And the rival is a selection rule, not an edge case.** "What determines how many
towers there are" is a question about which rule governs, not about what happens
at a boundary. Set the four hedges this project has examined side by side:

| task | model | rival was about | candidate 2 |
|---|---|---|---|
| `d35bdbdc` | Opus 5 | **selection** — degree-1 endpoints of the gray path | **won** (×2 outputs) |
| `16b78196` | Opus 4.8 | **selection** — one tower per notched side | **won** |
| `78332cb0` | Opus 4.8 | edge case — ordering within the line | lost (fatal error was *orientation*) |
| `88e364bc` | Opus 4.8 | edge case — diagonal corner-brushing | lost to a bug on the untested branch |

The pattern proposed after `d35bdbdc` now has an independent instance in a
different model, arrived at without looking for it: **candidate-2 wins come from
rivals about which rule applies; candidate-2 losses come from rivals about
boundary behaviour.** `88e364bc` is the instructive near-miss — its edge-case
rival happened to be the correct axis, and a bug in the branch training never ran
discarded the point anyway.

This is worth more than the ablation claim it replaces, for three reasons. It is a
within-task mechanism observation, so it needs no matched arms and survives all
the confounds that killed the earlier result. It now spans two models. And it is
directly actionable in the doctrine: the question to put to a solver is not
"where are you uncertain" but **"is there a different rule that fits every
training pair, or only a different boundary behaviour?"** — because only the first
kind of rival has ever won a slot here.

n = 4 tasks. Stated as a mechanism, not a rate.

---

## The 4.8 head-to-head, complete — and a correction to Arm A's baseline

Both arms are finished. Model `claude-opus-4-8`, effort `high`, `max_iterations 8`,
2 candidates per output, identical across arms — the arms differ **only** in which
tasks they contain.

**Arm A** — the 35 tasks CoT-4.8-high fails. Measures **gain**.
**Arm B** — 10 of the 85 it solves, `random.Random(20260801).sample`. Measures
**regression**.

```
ARM A   35/35 scoreable   23.33 points = 66.7%   21 solves  5 partials  9 zeros   $218.74
ARM B    9/10 scoreable    9.00 points = 100.0%   9 solves  0 partials  0 zeros    $47.65
```

### The correction, which cuts the headline gain by a third

Every report of this experiment, including several in this log, described Arm A's
baseline as **0.00 per task**. That is wrong, and the error is in the premise
rather than the arithmetic.

The baseline is **85/120 fully solved, 92.33 points**. Eighty-five fully-solved
tasks contribute exactly 85.00 points — so the remaining 35, which are *precisely
Arm A*, already earn the baseline **92.33 − 85.00 = 7.33 points of partial
credit**. ARC-AGI-2 scores multi-output tasks fractionally, and "CoT fails this
task" means *did not fully solve it*, not *scored nothing*.

| | as reported | actual |
|---|---|---|
| Arm A baseline | 0.00/task, 0 points | **0.21/task, 7.33 points** |
| Arm A CCARC | 23.33 points | 23.33 points |
| **gain** | **+23.33** | **+16.00** |

The projection is unchanged, because it was right for the wrong reason:

```
baseline: 92.33 points (85 solved = 85.00 pts, + 7.33 partial on the 35 it fails)
CCARC keeps the 85.00 and replaces 7.33 with 23.33
=> 108.33/120 = 90.3%  vs baseline 76.9%  (+16.00 points, +13.3 pp)
```

`92.33 − 7.33 + 23.33` lands in the same place as `85 + 23.33`, so the final
number checked out while the comparison beneath it did not. **A correct total is
not evidence of a correct model**, and this one survived precisely because nobody
asked how it was built until someone did.

> Both figures corrected tonight — this baseline, and "4.8 has produced no
> winning second candidate" — were among the most-repeated. Restating a number
> does not test it; it only makes it load-bearing. The check that caught this was
> "show me the arithmetic", which is cheap and which I had not run on my own
> headline.

### What the completed arms support

**On tasks the baseline already solves, the harness costs nothing** — 9.00/9.00,
zero regression. Measured on 9 tasks, which cannot exclude a regression rate
below roughly 10%; the subtraction term in the projection is not zero, it is
unmeasured.

**On tasks the baseline fails, the harness recovers about two-thirds** — 66.7% of
available points, 21 of 35 solved outright, against a baseline that scraped 7.33
points of partial credit from the same set.

**Hedging, now with an n.** 10 of 21 hedged outputs solved against 37 of 42
single-candidate outputs, and **two outputs were won by the second candidate**
(`16b78196`, `8e5c0c38`). Earlier in the batch that count was zero and the
tempting write-up — "4.8 cannot produce a winning rival" — would have been
falsified within the hour.

**Every threshold stayed dissolved.** cost/score rank correlation is −0.583 at
n=44, down from −0.864 at n=15, with solves ($1.19–$10.75) and failures
($3.20–$29.90) overlapping across almost the whole range. Four cutpoints were
drawn tonight and none survived a full sample.

**The iteration budget is inert**: 38 of 45 runs used exactly 1 of 8.

### A prediction, scored by machine

Recorded at 16/35 with Arm A standing at 58.3%, predicting the final figure would
land materially below that as slower tasks arrived:

```
predicted: materially below 58.3%.  actual: 66.7%  ->  WRONG
```

The reasoning was that completion order favours fast runs and speed tracked
success. The mechanism was real; the premise was a correlation I had already
watched decay four times that evening, and I built on it anyway. The scoring was
delegated to a script written before the data existed, which is the only reason
this appears here as `WRONG` rather than as a paragraph explaining why it was
nearly right.

---

## The 3v3 ablation: the original finding does not reproduce

The properly-controlled replication is complete. Task `d35bdbdc`, model `opus`
(Opus 5), effort `high`, all six runs at commit `d5a2abf` — one commit, so one
`arc.py`, one `CLAUDE.md` template, one doctrine, one runner. The only deliberate
difference is two sections of the workspace `CLAUDE.md`, and the manipulation was
verified by md5 **and** by confirming each run actually produced stream events,
which the first attempt did not.

```
doctrine_1  1.00   matched=[1, 1, 1]   $6.05
doctrine_2  1.00   matched=[1, 1, 1]   $4.84
doctrine_3  1.00   matched=[2, 2, 1]   $7.69
ablated_1   1.00   matched=[1, 1, 1]   $5.55
ablated_2   0.33   matched=[None, None, 1]   $3.63
ablated_3   1.00   matched=[1, 1, 1]   $5.63
```

**Doctrine 3/3, ablated 2/3. Fisher exact one-sided p = 0.500.** There is no
detectable effect. For scale: the original 2v2 with *perfect* separation scored
p = 0.067, and the best a 3v3 could ever have done was p = 0.050.

The earlier result — two doctrine runs at 1.00 against two ablated at 0.33, which
this log treated as its headline finding — **does not reproduce**. It was a
perfect separation at n=2 per arm, on a task selected for study after it first
scored 1.00, in a design already shown to have had the doctrine present in every
arm despite the labels. Every one of those objections was raised by the
adversarial pass before this replication ran, and the replication agrees with
them.

**Why it vanished is more interesting than that it vanished.** The old five runs
shared a bit-identical candidate 1 that lost tests 0 and 1 in **5 of 5**, so the
entire score spread lived in candidate 2 — which is what made "100% of the
variance is the second candidate" true at the time. In the six new runs, **5 of 6
win tests 0 and 1 on candidate 1**. The primary hypothesis now usually succeeds
outright.

> The contrast disappeared because **the task stopped being hard for the
> harness**. An ablation measures the gap between two conditions; when both
> conditions start solving the task on their primary hypothesis, there is no gap
> left to measure. The old effect was a real difference in *rescue rate* on a
> task where the primary reliably failed — and the harness improvements since
> (`sweep()`, `rival()`, the full tool surface, the doctrine additions) removed
> the failure the rescue was rescuing.

That also means the replication is not a clean refutation of the original
observation. It is a demonstration that the observation was **conditional on a
harness state that no longer exists**, which is a different and more useful thing
to know. A finding that depends on the primary being broken is not a finding
about hedging; it is a finding about a bug that has since been fixed.

**What this costs the log.** Everything previously drawn from the ablation needs
reading as conditional. The "rigour applied to killing is still killing"
observation, promoted at one point as the project's most transferable lesson,
rests on ablated runs refuting a correct rival — behaviour observed when the
primary was failing, in arms that all received the doctrine, at p = 0.167. It may
still be true. It has now been tested twice and confirmed neither time.

**What survives untouched** is the mechanism observation, because it never
depended on the arms separating: across four tasks and two models, candidate-2
wins came from rivals about *which rule applies* (`d35bdbdc`, `16b78196`) and
candidate-2 losses from rivals about *boundary behaviour* (`78332cb0`,
`88e364bc`). `doctrine_3` adds a fifth instance — `matched=[2, 2, 1]`, the only
run in the six to win on its hedge, and it hedged the selection rule.

---

## Hedging, with an n at last — and a correction to the one claim that survived

Every scored test output across every run this project has ever made, both models:

```
hedged (2 candidates)  53/73 solved   73%
single candidate       57/64 solved   89%

of the 73 hedged outputs:
  won BY candidate 2                13   18%   <- the hedge earned its slot
  won by candidate 1 (redundant)    40   55%   <- the hedge was spare
  both candidates lost              20   27%
```

**The second candidate converts 18% of the time.** Not the dominant mechanism the
`d35bdbdc` analysis implied, and not the zero it looked like halfway through the
4.8 batch, when the tempting write-up "4.8 cannot produce a winning rival" was
one result away from being falsified.

A caveat that matters more than the percentage: **this corpus is an accumulation,
not a sample.** `d35bdbdc` alone contributes 6 of the 13 candidate-2 wins because
it was studied seven times. Across *distinct tasks* the count is 8.

### The selection-vs-edge-case claim does not hold

Reading the hypothesis of every task that has ever won on candidate 2:

| task | the rival was about | predicted by "selection wins"? |
|---|---|---|
| `d35bdbdc` | which endpoints select the survivors | yes |
| `16b78196` | one tower per notched side | yes |
| `faa9f03d` | smaller object in front, or shorter-stretch in front | yes |
| `8e5c0c38` | which of two symmetry axes | yes |
| `a6f40cea` | is train 2's defect a rule, or an authoring slip | yes |
| **`abc82100`** | **what to do with an unmatched dot** | **no — an edge case** |
| **`800d221b`** | **is the hub centre colour A or B** | **no — one cell** |

Five of seven, not seven of seven. **Edge-case rivals do win**, and one of them is
as small as a rival gets: a single cell with two possible colours. The claim
recorded two sections ago as surviving the ablation intact is a tendency, not a
rule, and it should not have been stated as strongly as it was.

### A better predictor, in the solvers' own words

> `800d221b`: *"the two candidates **exhaust the possibilities** for that cell."*
> `8e5c0c38`: *"a genuine tie the evidence cannot break"* — two axes, enumerated.
> `a6f40cea`: the defect is *"either a real rule or a hand-drawing error"* — and
> this candidate deliberately **fails train 2**, kept because those two readings
> exhaust it.

Against the losses: `78332cb0` shipped two candidates that were *both transposed*
— the true answer was never in the candidate space at all, because the solver
hedged an elaborate ordering question while orientation, the axis it never
considered, decided the outcome. `88e364bc` had the right axis and lost to a bug
on the untested branch.

> **A hedge wins when the two candidates exhaust the possibility space — not when
> the ambiguity is of a particular kind.** `800d221b` won on a one-cell binary
> because there were only two colours it could be. `78332cb0` lost an elaborate
> hedge because the answer was outside both readings.

That changes the advice from *"hedge on rules, not boundaries"* to **"hedge where
you can enumerate the alternatives exhaustively, and be suspicious of a hedge you
cannot close"**. It is a sharper instruction and a testable one: a solver can ask
*"do my two candidates cover every possibility for the thing I am unsure about?"*
and get a real answer, where *"is this a selection rule or an edge case?"* invites
a judgement call.

The doctrine currently carries the weaker version, landed an hour before this
analysis. It should be revised, and this is recorded here rather than acted on
immediately because the claim it replaces was also stated too strongly on first
sight — twice now, on the same subject.

### Arm B addendum: the resumed tenth task, and the first regression signal

`4e34c42c` — excluded from Arm B all night as an infrastructure loss — was
resumed for consistency with the three Arm A casualties that were resumed and
counted. It came back **0.50** (`matched=[None, 1]`: both candidates lost test 0,
candidate 1 won test 1), `accepted: True`, at **$32.40 cumulative across three
launches** — the most expensive task in the project.

That is the **first regression signal Arm B has produced**, and it must be
reported in two figures rather than one, because the run is not the same
experimental condition as the other nine:

| Arm B | scoreable | points | vs baseline 1.00/task |
|---|---|---|---|
| **fresh runs only** (the comparable set) | 9 | **9.00/9.00** | **0.00** |
| **including the resumed task** | 10 | **9.50/10.00** | **−0.50** |

The resumed run received `build_resume_prompt` with its prior ledger pre-loaded,
after two launches killed by a container SIGTERM and the wall clock respectively.
Folding it silently into the arm score would compare a resumed solver against
nine fresh ones; dropping it entirely would hide a task the baseline solves and
this harness did not.

**Which figure to use depends on the question.** For *does the harness regress
under identical conditions*, the answer remains 9.00/9.00 on nine tasks. For
*what happens to a task in practice, infrastructure included*, it is 9.50/10 —
and the projection's zero-regression term becomes an assumption with one
counterexample rather than none.

Note also what this does to the earlier reasoning about excluding it. The
argument for exclusion was that its gate ledger held a train-perfect submission
when the clock expired, so scoring it 0.00 would attribute a timeout to the
model. That was right — it scores 0.50, not 0.00. But "not zero" is not "one",
and the exclusion quietly implied the latter.

---

## The harness-version test: 3/3 recovered, 0/4 regressed

The question was whether the changes accumulated since 2026-08-01 actually help,
or whether the log had merely been accumulating plausible-sounding commits. Two
arms, both Opus 5, effort `high`, all at commit `d5a2abf`.

**Gain arm** — the three tasks never solved under the old harness. Baseline
**0.00/task**, since each had been tried and failed:

```
0934a4d8   1.00 (1/1)   matched=[1]      conf 5   11 invariants   $3.33
20270e3b   1.00 (2/2)   matched=[1,1]    conf 4   28 invariants   $4.33
9bbf930d   1.00 (1/1)   matched=[1]      conf 4   13 invariants   $5.58
```

**Regression arm** — four tasks the old harness solved. Baseline **1.00/task**:

```
28a6681f 1.00    78332cb0 1.00    7b5033c1 1.00    e8686506 1.00
```

**3.00/3.00 gained, 4.00/4.00 held.** Every gain came on **candidate 1** — the
primary hypothesis, not a rescue by the hedge.

The prior attempts, for the record: `0934a4d8` failed at 09:22 and `20270e3b` at
09:48 on 08-01, both inside the project's first hour, before `sweep()`,
`rival()`, the full tool surface, `unreached()` or any of the doctrine work.
`9bbf930d` failed **twice**, most recently at 20:37, still before the full tool
surface and the doctrine additions.

**What this does and does not establish.** It is a paired before/after on the
same tasks with a control arm, which is a better design than most things in this
log. But n=3 on the gain side, and two of those three had failed exactly once, so
run-to-run variance could account for part of it — `9bbf930d`'s two prior
failures are the strongest single data point precisely because it had two
chances. What it does exclude is the worry the control arm exists for: the
changes did not buy gains by breaking something else. 4/4 held.

**A consequence for the queued effort-escalation experiment.** The plan was to
rerun the failures at `--effort max`, with `harness_v2/gain` supplying a
three-rung ladder: old harness + high → new harness + high → new harness + max.
That ladder no longer has a third rung to climb — **all three tasks now solve at
`high`**, so there is nothing left in the gain set to escalate. The effort
experiment will have to draw its failures from elsewhere, and the honest reading
is that the harness changes got there first.

---

## Two rate-limit windows, and watching the wrong one

Long batches here are bounded by quota, not by anything in the harness. The
telemetry that reports it is richer than it first appears, and reading it wrong
is easy in a way worth recording.

Every solver stream carries `rate_limit_event` records. A one-shot probe gets one
without any batch running:

```bash
claude -p 'Reply with exactly: OK' --output-format=stream-json --verbose 2>/dev/null \
  | grep -o '"rate_limit_info":{[^}]*}' | head -1
```

**There are at least two windows, with different thresholds and wildly different
consequences:**

| window | warns at | typical reset | cost of exhausting it |
|---|---|---|---|
| `five_hour` | `utilization` 0.90 | ~5 h | a pause of an hour or two |
| `seven_day` | **`utilization` 0.75** | up to ~7 days | **a blackout of up to days** |

The status ladder is `allowed` → `allowed_warning` → `rejected`, and
`utilization` appears **only once the window crosses its own threshold** — below
that the field is simply absent. Overage is `rejected` with
`out_of_credits`, so exhaustion is a hard kill of in-flight solvers rather than a
slowdown: nine concurrent runs died that way here at 03:19.

**The mistake worth avoiding.** A checker that greps for the most recent
`rate_limit_info` reports *whichever window last emitted an event*. Since the
five-hour window emits constantly and the seven-day one stays silent until it
crosses 0.75, such a checker shows a comfortable five-hour reading for hours
while the weekly climbs invisibly. That ran here for most of a night, and
surfaced only when the seven-day window crossed its threshold and began emitting
on its own.

> **A monitor that reports one member of a set and calls it the state is not a
> monitor.** The fix is to key readings by `rateLimitType`, keep the most recent
> of *each*, take the worst status as the overall, and refuse any reading older
> than a freshness bound — a stale `rejected` from hours ago is as misleading as
> a stale `allowed`.

Both failure modes were observed here within a few hours: a naive `tail -1`
across accumulated stream files reported `rejected` long after recovery, and the
same script's hardcoded directory glob silently excluded four of six live
experiments while reporting a nineteen-minute-old stream as current.

**Operationally**, the seven-day window is the one to plan around. It warns
earlier and costs an order of magnitude more to hit, so a batch that would be
merely delayed by exhausting the five-hour window can lose a day to exhausting
the weekly. On `allowed_warning`: finish what is in flight, launch nothing new.
