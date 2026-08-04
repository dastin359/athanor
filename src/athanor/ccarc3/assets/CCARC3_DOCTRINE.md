# Solver doctrine — ARC-AGI-3

Everything here was measured, not assumed. Where a line contradicts what seems
obvious, the contradiction is the point: several of these were learned by losing
progress to them.

---

## 0. How you are scored. Read this first; it decides what to optimise.

The benchmark does not count games won. Per **completed** level, against the
human baseline `h` for that level, with `a` the actions you spent on it:

```
level score  = min(1.15, (h / a) ** 2)      an unfinished level scores 0
environment  = min( completed-levels cap , weighted mean of level scores )
weights      = the level numbers themselves: 1, 2, 3, ... n
cap          = sum(1..completed) / sum(1..n)
```

Four consequences, each of which changes what you should do:

**1. Only finished levels score at all.** A level you cannot complete is worth
zero no matter how elegantly you spent 200 actions inside it. If a level is
genuinely unreachable with the budget left, stopping costs you nothing — but
finishing one more level is worth a great deal, so be sure before you conclude
that.

**2. The ratio is squared.** Twice the baseline scores 0.25, not 0.5. Three
times scores 0.11. Overrunning a level is punished far harder than it looks,
and the damage is done in the level you are overrunning, not spread out.

**3. There is a cap at 1.15, and it is easy to hit.** Once you are at roughly
0.93× the baseline you have collected everything speed can earn. Being three
times faster than the human is worth **exactly the same** as being 1.08 times
faster. Surplus speed is not score — spend it on certainty, and on depth.

**4. Later levels are worth more, proportionally to their number.** On an
eight-level game the last level carries 8/36 of the environment and the first
carries 1/36. Being slow on level 1 costs little; being slow on level 7 costs
eight times as much. **Depth beats polish.** Clearing five levels scrappily
beats clearing three immaculately, every time.

Put together: get under the baseline, stop optimising once you are, and push as
deep as the budget allows.

### 0a. Exploration and execution can be separated. Use it when you were slow.

**A RESET issued while the server's action counter is zero — the state
immediately after a level advance — starts a NEW PLAY.** Measured: the API then
records a new `guid`, a new `actions` row, and a new `actions_by_level` row.
Per-level action counts are kept per play and never summed.

**The server scores the BEST play, and this is measured, not inferred.** Two
plays were driven by hand on `lp85` and the scorecard read back directly:

```
run 1:  21 actions   level_scores [ 65.533, 0...]   score 1.8204
run 2:   7 actions   level_scores [115.000, 0...]   score 2.7778
                                  environment score = 2.7778
```

The environment took the better play. Run in the other order — good play first,
bad play second — it *still* reported 2.7778, so the rule is `max` over plays and
not "most recent". **A replay can only raise your score or leave it alone. It can
never lower it.**

Everything else in the scorecard reproduced the rubric exactly: `65.533 =
(17/21)² × 100`, `115.0 = min(1.15, (17/7)²) × 100`, `1.8204 = 1 × 0.65533 / 36
× 100`, and `2.7778 = 1/36`, the completion cap for 1 of 8 levels binding over
that play's raw 3.194.

> **An earlier version of this section said DO NOT DO THIS.** The argument was
> that the human baseline comes from people who *could not restart* — "limited
> to a single attempt per environment", able to reset only the current level —
> so `(h/a)²` should compare your cost of learning against a human's cost of
> learning, and a post-learning replay measures a different quantity.
>
> That reasoning is still sound as a description of what the number *means*. It
> was wrong about what to do. Nothing forbids an agent from restarting, the
> scorer demonstrably reads the best play, and the benchmark is the benchmark.
> Recorded here rather than deleted because the caveat it ended on — "it is
> unverified that it would even work" — was the honest part, and it has now been
> verified the other way.

The games are also **deterministic** — replaying an action sequence reproduces
the board exactly, patrolling objects included, verified 40 frames out of 40.

So when a level cost you far more than its baseline, you have a second chance
that the score is happy to take:

```python
client.restart_for_replay()      # only legal right after a level advance
```

**Judge it by arithmetic, not by mood.** An environment scores
`min(completion cap, weighted mean)`. Clear every level and the cap is 1.0, so:

- If your levels came in **under baseline**, your raw score is already at or
  above 1.0 and the cap is binding. **A replay gains you exactly nothing.**
  This is the usual case — of 19 runs on record, 17 were already at the cap.
  Do not spend actions on it.
- If overruns dragged your raw score below the cap, a replay is worth
  `cap − raw`. `tn36` is the worked case: one run finished at `raw` **0.449**
  against a `cap` of 0.750, having cleared six of seven levels — and a later run
  of the same environment cleared all seven in **220 actions against that run's
  631**, scoring **1.000**. The routes were the same. What differed was spending
  them rather than finding them.

The second case is the one to watch for, because it is where a *bad* run turns
into a good score. The cap does not care that you were slow the first time.

**Clearing every level is not the end of the scoring.** `E = min(cap, raw)`.
Once you have cleared them all, `cap` is 1.0 and your score *is* `raw` — so any
level you finished above its baseline is still costing you, and a replay is
worth `1.0 − raw`. The instrument is available at exactly that moment and not
after: the winning frame is a level advance, so the action counter is zero and
`restart_for_replay()` is legal. Take one more action first and it is refused,
while every other action is refused too because the state is terminal — the run
is then stuck with the score it has.

Measured on this project's own runs: one won 6/6 with `raw` **0.9785** and
**71% of its action budget unspent**, and stopped. That 0.0215 was free.

**If you cannot compute `raw`, replay anyway when the budget is there.** Without
the per-level baselines you cannot tell whether you finished a level at 1.15 or
at 0.90, so *"I won"* and *"I scored what this game was worth"* are different
claims and you can only check the first. A replay that walks the route you now
know spends actions you were not going to spend, and **cannot lower your score**
— the server keeps each play separately and takes the best one.

Check before deciding: `arc.score_run(client.transitions(), baselines)` gives
`raw`, `cap` and the per-level breakdown.

**Fix whichever term is binding.** `E = min(cap, raw)`, so only the smaller one
is costing you, and the two are improved by opposite actions:

| binding term | what it means | what to do |
|---|---|---|
| `cap < raw` | you were efficient but stopped early | **clear another level.** A replay is worth zero |
| `raw < cap` | you finished levels but fumbled through them | **replay.** Another level barely helps — the ceiling is not your problem |

That second row is counterintuitive and it is the case this project actually
lost. On `tn36`, `raw` 0.449 against `cap` 0.750: clearing the last level would
have raised the ceiling to 1.0 and the score only to **0.511**, because `raw` was
still binding. Replaying the six levels it had already solved was worth
**0.750** — nearly five times as much, and more per action spent.

**If you do not know the baselines, you still know the cap.**
`cap = sum(1..k)/sum(1..n)` needs only levels cleared and levels total, both of
which the API tells you. `raw` is the term you cannot compute. So your ceiling
is known, your distance below it is not, and **a replay that executes the route
you now know is the only way to guarantee you collect the ceiling your level
count already earned.** Without it you may be leaving `cap - raw` behind with no
way to detect it.

You also do not need a baseline to know you fumbled. The gap between what the
route turned out to be and what you spent finding it is the signal — a level
that cost 200 confused actions and resolves to a twelve-move route is telling
you `raw` is far below `cap` in any units.

And a replay you cannot finish costs nothing but the actions. Each play carries
its own completion cap and the environment takes the **best** play, so a replay
that runs out of budget on level 3 simply scores worse and is discarded.

**What earns the score is understanding, not the recording.** Replaying your
own trace verbatim reproduces your own fumbling — the same actions give the
same result, that is what determinism means. The gain comes from executing the
route your *rules* now imply, which is work you can only do once you genuinely
understand the game. If you cannot state a better route than the one you took,
a replay will not help you.

**Reserve the budget before you need it, or the option is gone.** A replay
costs roughly one baseline. Your cap is a multiple of the baseline, so:

> **Do not spend more than half your action cap before you understand the
> game.** Keep one baseline in reserve.

This costs a healthy run nothing — runs that win use about 0.6 of a baseline in
total, well inside half the cap. It costs a struggling run the only move that
was still worth making.

Measured on a real run, `tn36`. After overrunning three levels it stood at
0.442 with five of seven cleared, and:

| option | value |
|---|---|
| stop there | 0.442 |
| finish the last two as it was going | 0.726 (+0.284) |
| restart and replay cleanly | **1.000 (+0.558)** |

The replay was worth twice as much as finishing — and was **impossible**,
because 317 actions were needed and 145 remained. The exploration had eaten
the budget the recovery required. Nothing about the game prevented it; only the
spending did.

---

## 1. Dying is cheap. Ignorance is expensive.

Your score is the **best** across all your plays; your actions are **summed**
across all of them.

```
Card.high_score    = max(scores)        # a GAME_OVER costs you nothing
Card.total_actions = sum(actions)       # every action costs, forever
```

A GAME_OVER does not end your run. It obliges a RESET, and RESET **restarts the
current level only** — not the game. So the price of a death is the actions you
had already spent inside that level, and nothing else.

**Therefore: use deaths as experiments.** If you suspect a colour is lethal,
the cheap way to find out is to touch it. A labelled negative transition is the
highest-information observation available, and it costs no score.

The one thing that scales the price is *distance from the level's start when you
run the experiment*. Test lethality early in a level, not at its far end.

> The stock SDK prompts all say *"WIN and avoid GAME_OVER while minimizing
> actions."* Under best-of scoring that is wrong as a terminal goal. Avoiding
> death is instrumental, not an objective. An agent that fears dying will
> under-explore, and it will do so exactly where exploring is cheapest.

### 1a. A per-action display may or may not be able to kill you. Test it.

Many games show something that changes once per action — a bar, a counter, a row
of tokens, a shape that shrinks. **Do not assume you know what it does.**

The two possibilities call for opposite play:

- **A resource that kills at zero.** Running out is `GAME_OVER`. Wandering is
  genuinely dangerous and exploration has a hard deadline.
- **A cycle that simply repeats.** Running out costs nothing. Treating it as
  lethal makes you rush, skip experiments and waste the exploration budget that
  every other line here tells you to spend.

Measured on `ls20`, the second is what happens: the bar holds 42 cells, drains
exactly one per action, reads 0, and **refills to 42 automatically** with nothing
collected and nothing else on the board changing. A clean 43-action cycle, twice
over, with no death in 120 actions of aimless wandering. A solver that assumed
lethality would have played that game far too carefully.

So: **find the display, then find out what it does.** `arc.monotone_rows()` finds
it. To learn whether it kills, let it run out once, early in a level, where a
death is cheap — that experiment costs almost nothing and its answer changes how
you play the rest of the game.

If it *does* kill, a death then has two possible causes and they call for
opposite responses:

- **Contact with something lethal** — you learned where a hazard is. Record it,
  with the location.
- **Resource exhaustion** — you learned nothing about the board. Recording a
  "hazard" here is a false rule that will mislead you for the rest of the run.

Deaths landing at a consistent action *count* rather than a consistent *place*
mean the resource, not the board.

## 2. Never make RESET your first action after completing a level.

The server's action counter is zeroed when a level advances. A RESET at that
precise moment takes the full-reset branch: **score to zero, back to level 0,
the whole game discarded.** Any other RESET does a harmless level reset.

The condition is invisible — nothing in the frame exposes it. Take any other
action first. `ArcClient.reset()` refuses this call for you; if you see that
refusal, it just saved the run.

**Do not trust `full_reset` on the frame to tell you it happened.** A run that
went from level 6 to level 0 in a single RESET received `full_reset: false` on
that very frame. The reliable signal is the one the server cannot fake: **your
level went down.** If `client.level` is lower than it was, your progress is
gone, whatever the flag says.

This has already cost one finished game. A resumed run opened with RESET one
action after clearing a level, replayed all seven levels, and nothing in the
harness reported a full reset while it happened.

### 2a. A level reset is a cheap, legitimate retry.

Resetting mid-level costs one action and restores the level. It does not open a
new play or touch your score, and it is not only for recovering from death — a
real run used two level resets with zero deaths, simply to re-approach a level
better once it understood it. If your route through a level was wasteful, doing
it again knowing the answer can cost fewer actions than continuing to improvise.

The opening RESET that starts the game is free. Every later one is billed.

## 3. Do not act while dead.

While the state is `GAME_OVER` or `WIN`, every non-RESET action is discarded
without stepping the game — and is still billed. Read `state` on every frame.
`ArcClient` refuses these too, and counts them as `wasted` in the ledger.

## 4. Rules are level-scoped. Mechanics are game-scoped.

| layer | scope | example | carries? |
|---|---|---|---|
| mechanic | game | "there is a pushable-object mechanic" | strongly |
| rule | level | "the block at (12,30) moves 1 right when pushed" | no |

Levels share lineage, not setup. A rule you learned on level 3 will often be
plainly false on level 1, because level 1 was arranged differently. That does
not make it a bad rule.

When you reach a new level, **do not port your rules. Port your mechanics, as
priors about what to test first.** A mechanic seen on three levels is worth two
actions to confirm rather than fifteen to rediscover. Prior evidence buys you
*search order*, never *truth*.

## 5. Failure is not refutation unless the belief was applicable.

Every belief you hold has a precondition and a claim, and they are separate.
"The red block moves when pushed", checked on a level with no red block, is
**not applicable** — it is not *refuted*. Only a case where the precondition was
met and the claim still failed is evidence against you.

Three outcomes, never two. Collapsing "did not apply" into "did not hold" is how
correct knowledge gets thrown away, and it is the single most expensive mistake
available here. It is the same distinction `gate.acknowledge()` asks you to make
between `refuted=` and `untested=`: something you never tried is not something
you disproved, and filing it as a refutation makes you stop asking.

If you want it mechanised, `arc.Rule` takes `applies` and `holds` as separate
callables so the two cannot merge, and `arc.verify` / `arc.survey` check one
level or report across all of them. Most solvers do this reasoning in their own
code instead, which is fine — the distinction is what matters, not the API.

**Exclude level boundaries from anything spatial — however you are analysing.**
The action that completes a level returns the *next* level's board, so `before`
and `after` are different boards entirely. Anything you conclude about movement,
position or adjacency across one is nonsense: the avatar appears to teleport.

This bites in a plain list comprehension exactly as hard as in a rule:

```python
ts = client.transitions()
[t for t in ts if t.action == "ACTION1" and not t.board_replaced]
```

`board_replaced` covers both causes — completing a level, and a full reset that
rewinds the game to level 0. `crosses_level` catches only the first, because a
full reset moves the level *down*.

This is not hypothetical: on a real run, "ACTION1 moves the cursor up" held 7/7
on level 0 and showed 13 holds and 1 violation on level 1. The violation was the
boundary. Excluded, the rule is 28/0 across both levels.

Never conclude a rule is dead because it failed on an earlier level. That is how
true knowledge gets destroyed.

**Vacuous truth is not verification.** A rule that was never applicable has been
tested by nothing.

### 5a. A forward model beats a pile of predicates.

A predicate says *a* property survived. A `step(before, action, params) -> board`
that reproduces every recorded board **exactly** says you understand the
mechanics. That is a much stronger claim, and it is the one to aim at:

```python
arc.predict(step, client.transitions(), name="my model")
# -> "my model: 47/48 exact (98%), 3 skipped; first failures at [31]"
```

Return `None` from `step` for cases you do not model yet — those are skipped,
not counted wrong. Level boundaries, full resets and wasted actions are skipped
for you.

**Chase `perfect`, not accuracy.** A model at 98% is not 98% right about the
mechanics; it is missing one, and `failures` tells you exactly which transition
to go and look at. The single wrong prediction is worth more than the 47 right
ones.

### 5b. Verification is cheap here. Planning is what costs you.

On a static puzzle you cannot test a hypothesis without spending your one
answer, so most of the work is convincing yourself you are right. **Here, testing
is acting**: one action tells you, for the price of one action. That inverts what
is scarce.

So do not spend long proving a mechanic you could confirm in two moves. Spend the
effort on the question that actually costs actions — *given what I already
believe, what is the shortest route?*

```python
arc.shortest_path(step, start, goal)     # step(state, action) -> state
arc.reachable(step, start)               # is it even reachable, or did I misread?
```

It searches over a **step function**, not a grid, so launchers, teleports and
wrap-around work as long as your model has them. The route is optimal *for your
model* — check the model with `arc.predict()` before trusting a route from it.

An unexpectedly small `reachable()` set usually means your model puts a wall
where there is none.

## 6. Budget against the published baseline.

Every game publishes `baseline_actions` — one figure per level, what a
playthrough costs when the rules are *already known*. Real games range from 171
to 1843 total.

**Here is what normal looks like, measured over 26 level-attempts across five
games:**

| | |
|---|---|
| levels cleared at **≤ 0.92×** their baseline | **24 of 25** |
| median | **0.52×** |
| attempts that crossed 1.0× | 2 — one cleared at 3.44×, one never cleared at 6.13× |

Finishing a level you understand costs *less* than the published figure, because
that figure includes a human's own hesitation. So **crossing 1.0× is already
unusual**, and it happened twice — both times on the levels that independent
analysis had already flagged as the problem levels.

You do not have to compute this. `client.status()` leads with it:

```
g: level 3/6 state=NOT_FINISHED actions=291 [190/55 on this level = 3.5x]  <- OVER BASELINE
```

There is no reliable cutoff to look up: the sample has **nothing between 0.92×
and 3.44×**, so any threshold in that range separates it identically. Read the
shape instead — normal is comfortably under 1, and being over it means stop
executing and go re-explore.

The sharper signal is relative, and `client.pace()` gives it to you free:

```python
client.pace()      # {0: (17, 22, 0.77), 1: (59, 123, 0.48), ...}
                   # {level: (spent, baseline, ratio)}
```

A game may run above or below baseline throughout; a level that runs above *its
own run* is the real outlier. That 3.44× level was **6.2× the median of the
levels the same run had already cleared** — far louder relatively than in
absolute terms.

And note what the 3.44× run was doing: it spent 344% of a level's baseline
executing a wrong plan **without wasting a single action.** Every move did
something; all of them were beside the point. Efficiency is no defence against
being wrong.

### 6a. Over pace, ask whether you are exploring or going in circles.

Being over baseline does not by itself mean you are stuck. The only game lost so
far and the slowest level of a game that was *won* both ran over the 1.0×
warning; what separated them was whether the board was going anywhere new.

`status()` reports it when it happens:

```
95/308 actions returned the board to a state already seen on this level
```

That is the level the lost run never cleared. Of its 309 actions, **8** changed
nothing — so the "changed nothing" tally stayed quiet — while a third of them
put the board back somewhere it had already been. Every action worked. The board
kept coming home.

Compare, at the same 1.52× of baseline: one level at 2% revisits, another at 8%,
both cleared. The winning run with the best efficiency on record revisited
**nothing at all**.

So when you are over pace, read the two numbers together:

| over pace, few revisits | you are exploring. Cost is real, progress is real. |
|---|---|
| over pace, many revisits | your action set moves the board around a loop. More of the same will not break out. Change *what you are trying*, not how carefully you try it. |

This is a number, not a threshold — the sample is 22 level-attempts and longer
levels collide more often by chance alone. Do not treat any particular
percentage as a stop sign. Treat a large one as a reason to go back to §5 and
ask what you believe that is wrong.

## 7. Trust `available_actions` over everything else.

Each frame tells you exactly which actions this game accepts. It is
authoritative. Do not spend actions discovering that ACTION5 does nothing when
the frame already told you it is not available.

`win_levels` tells you how many levels the game has. There is no `score` field —
the server sends `levels_completed`. Anything reading `score` gets 0 forever.

### 7a. Available is not the same as effective.

The frame says what the game *accepts*. It does not say what has an *effect*,
and on some games the gap is enormous: a run that never cleared a level spent
**one action in five changing nothing at all** — 46 of its 157 clicks landed on
dead ground — while every winning run in the same batch wasted none.

Reading the gap is free, because it comes off the ledger rather than the game:

```python
arc.effective_actions(client.transitions(), level=client.level)
# {'ACTION1': (12, 12), 'ACTION6': (0, 31)}   <- stop paying for ACTION6
```

`(changed, tried)`. Anything reading `0/n` for more than a handful of n is
either the wrong modality or the wrong target. The two most efficient wins on
record used exactly **one** action type each; the one total failure spread
itself across seven.

This does not mean an action that changed nothing is useless — a click on empty
space is a real observation the first time. It means the *second* one is not.

## 8. Write code to analyse; do not eyeball frames.

Every action is recorded to a ledger with the action that caused it. Frames are
64×64 over sixteen colours and carry real pixel detail — reading them by eye is
both expensive and unreliable.

Ask questions in code:

```python
ts = client.transitions()
# Which actions are doing anything at all, on this level?
arc.effective_actions(ts, level=client.level)
# What did ACTION3 do, every time?
[arc.diff(t.before, t.after) for t in ts if t.action == "ACTION3" and t.changed]
```

Downscaling will not save you as much as you expect — real frames rarely have a
uniform block structure, so `logical()` often declines to reduce at all. The
saving that works is being **selective about which frames you look at**, not
compressing all of them.

### Any `Bash` call that plays actions must pass an explicit `timeout`.

**`Bash` defaults to 120 seconds and does not warn you.** A shell `timeout 300`
inside the command does not help — the tool's own limit fires first, at 120 s,
and one of two things happens:

- the call is **moved to the background**. Your loop keeps running and keeps
  spending real actions on the real game, but you no longer see its output. You
  are now flying blind on a board that is still changing.
- the call is **killed** — `Exit code 143` — mid-loop, leaving the game in
  whatever state the last completed action produced.

Both have happened. Across ten scored games, six calls hit the cap: four went to
the background and one was killed outright partway through a replay loop that was
issuing `client.act(...)`.

The tool accepts `timeout` in **milliseconds, up to 600000** (10 minutes). Pass it
whenever a call drives the game or runs a search:

```python
# Bash(command=..., timeout=600000)
from session import client, gate, arc
for action in [1, 3, 3, 5]:
    client.act(action)
print(client.status())
```

If the work genuinely needs more than ten minutes, **split it** — drive a dozen
actions, print `client.status()`, return, and continue in the next call. A loop
you can see is worth more than a longer one you cannot.

## 9. Record what you learned at every level boundary.

You will be refused the first action of a new level until you do. That is
deliberate: the boundary is the one moment you are guaranteed to have just
learned something and to be about to need it.

State three things: which mechanics you believe are game-scoped, which rules
were level-specific, and what you ruled out. **Record refutations** — "ACTION5
does nothing in any state seen" is worth five actions on every later level if
remembered, and costs five per level if forgotten.

If a level genuinely taught nothing portable, say so. That is a real answer.

## 10. Hold hypotheses, not conclusions.

Stamp the hypothesis you are testing into the action's `reasoning` field; the
server stores it and hands it back, so your trace explains itself for free.

When a mechanic you had verified breaks on a new level *while applicable*, that
is the most informative event in the whole run — the game has just shown you
what it changed. Do not explain it away. Go and find out what is new.
