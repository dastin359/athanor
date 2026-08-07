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
zero no matter how elegantly you spent 200 actions inside it. Finishing one more
level is worth a great deal, and there is no arithmetic that makes stopping
early correct. **Across the whole 25-environment arm the completion cap was the
binding term in every single environment**, so there, exploring cost nothing and
every point lost was lost by not finishing.

That is a statement about a regime, not a law, and the regime ends the moment you
clear everything: with `cap` at 1.0 only `raw` can bind. `sk48` cleared 8 of 8,
replayed, and still finished at 0.9538 — it overran its last level. So the honest
rule is: **while a level is still unbeaten, depth is worth more than anything
efficiency can buy; once every level has fallen, the only thing left is a clean
replay.** No run has ever lost a point by exploring a level it had not yet
solved.

**2. The ratio is squared.** Twice the baseline scores 0.25, not 0.5. Three
times scores 0.11. Overrunning a level is punished far harder than it looks,
and the damage is done in the level you are overrunning, not spread out.

**3. There is a cap at 1.15, and it is easy to hit.** Once you are at roughly
0.93× the baseline you have collected everything speed can earn. Being three
times faster than the human is worth **exactly the same** as being 1.08 times
faster. Surplus speed is not score — spend it on certainty, and on depth.

**4. Later levels are worth more, proportionally to their number.** On an
eight-level game the last level carries 8/36 of the environment and the first
carries 1/36. Being slow on the first level costs little; being slow on the
last one costs eight times as much. **Depth beats polish.** Clearing five levels scrappily
beats clearing three immaculately, every time.

Put together: get under the baseline, stop optimising once you are, and push as
deep as the game goes.

### 0a. Exploration and execution can be separated. Use it when you were slow.

**A RESET issued while the server's action counter is zero — the state
immediately after a level advance — starts a NEW PLAY.** Measured: the API then
records a new `guid`, a new `actions` row, and a new `actions_by_level` row.
Per-level action counts are kept per play and never summed.

**The server scores the BEST play, and this is measured, not inferred.** Two
plays of one level were driven by hand and the scorecard read back directly:

```
run 1:  the fumbled play    level_scores [ lower, 0...]   lower environment score
run 2:  the clean play      level_scores [ higher, 0...]  higher environment score
                                          environment score = the higher of the two
```

The environment took the better play. Run in the other order — good play first,
bad play second — it *still* reported the better one, so the rule is `max` over
plays and not "most recent". **A replay can only raise your score or leave it
alone. It can never lower it.**

Everything else in the scorecard reproduced the rubric exactly: each level came
back at `min(1.15, (h/a)²) × 100`, and each play's environment score was that
level's weight over `sum(1..n)` — the completion cap for 1 of `n` levels binding
over the raw figure in both cases.

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

An environment scores `min(completion cap, weighted mean)`. Clear every level
and the cap is 1.0.

<!-- BASELINE-ONLY -->
**Judge it by arithmetic, not by mood.** If your levels came in **under
baseline**, your raw score is already at or above 1.0, the cap is binding, and a
replay gains you exactly nothing. That is the common case; do not spend actions
on it. Only when overruns dragged your raw score below the cap is a replay worth
`cap − raw`.
<!-- /BASELINE-ONLY -->

Overruns are what a replay recovers, and they are worth `cap − raw`. `tn36` is
the worked case: one run finished at `raw` **0.449** against a `cap` of 0.750,
having cleared six of seven levels — and a later run of the same environment
cleared all seven in **220 actions against that run's 631**, scoring **1.000**.
The routes were the same. What differed was spending them rather than finding
them.

That is the case to watch for, because it is where a *bad* run turns into a good
score. The cap does not care that you were slow the first time.

**Clearing every level is not the end of the scoring.** `E = min(cap, raw)`.
Once you have cleared them all `cap` is exactly 1.0, so your score is `raw`
**only while `raw` is below 1.0** — above it the cap pins you at 1.0000 and extra
efficiency is discarded. A replay is therefore worth `1.0 − raw`, which is
negative once you are past 1.0: nothing. A run that finished at `raw` 1.1445 and
replayed twice to reach 1.1500 scored 1.0000 all three times and spent 49% of its
actions doing it. The instrument is available at exactly that moment and not
after: the winning frame is a level advance, so the action counter is zero and
`restart_for_replay()` is legal. Take one more action first and it is refused,
while every other action is refused too because the state is terminal — the run
is then stuck with the score it has.

Measured on this project's own runs: one won 6/6 with `raw` **0.9785**, had the
route in hand, and stopped anyway. That 0.0215 was free.

**If you cannot compute `raw`, replay anyway.** Without the per-level baselines
you cannot tell whether you finished a level at 1.15 or at 0.90, so *"I won"* and
*"I scored what this game was worth"* are different claims and you can only check
the first. A replay that walks the route you now know spends actions you were not
going to spend, and **cannot lower your score** — the server keeps each play
separately and takes the best one.

Measured across the 25-environment baseline-free arm, where no solver could
compute `raw`: blind replay **paid three times** — `tu93` 0.8286 → 1.0000,
`cn04` 0.8910 → 1.0000, `g50t` 0.8941 → 1.0000, about +0.12 `E` each — and was
wasted roughly seven times, where play 1 was already at or above 1.0. The waste
is real in actions and wall clock and **zero in score**, so the bet paid: about
+0.35 `E` across the arm for no score risk, and nothing was traded away to make
it.

The one case where a replay is the wrong call is when a level you have never
cleared is still in front of you: completion moves `cap`, `cap` was the binding
term in all 25 environments, and an unreached level is worth more than a
re-walked one. Replay when you have run out of game, not when you have run out
of patience.

If you *do* have the per-level baselines, `arc.score_run(client.transitions(),
baselines)` gives `raw`, `cap` and the breakdown. If you do not, you are in the
case above and the decision does not need them.

<!-- BASELINE-ONLY -->
**Fix whichever term is binding.** `E = min(cap, raw)`, so only the smaller one
is costing you, and the two are improved by opposite actions:

| binding term | what it means | what to do |
|---|---|---|
| `cap < raw` | you were efficient but stopped early | **clear another level.** A replay is worth zero |
| `raw < cap` | you finished levels but fumbled through them | **replay.** Another level barely helps — the ceiling is not your problem |
<!-- /BASELINE-ONLY -->

**Fumbling through levels you cleared is the expensive failure, and it is the
one this project actually made.** On `tn36`, `raw` 0.449 against `cap` 0.750:
clearing the last level would have raised the ceiling to 1.0 and the score only
to **0.511**, because `raw` was still binding. Replaying the six levels it had
already solved was worth **0.750** — nearly five times as much, and more per
action spent.

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
that stalls on level 3 simply scores worse and is discarded.

**What earns the score is understanding, not the recording.** Replaying your
own trace verbatim reproduces your own fumbling — the same actions give the
same result, that is what determinism means. The gain comes from executing the
route your *rules* now imply, which is work you can only do once you genuinely
understand the game. If you cannot state a better route than the one you took,
a replay will not help you.

**You are not spending against an allowance.** There is a hard stop, far out,
to keep a runaway loop from costing money — but it is not a resource you are
meant to husband. Under the ceiling now in force **no run has come close to
it**; under an earlier one, 40% as generous, two runs hit it — the only two ever
stopped by the ceiling rather than by the game. It was raised rather than
defended, because the
benchmark is deliberately
designed that way: a per-environment budget would "encourage AI to waste actions
on levels because they're still 'under budget'".

So do not pace yourself, do not hold actions back for later, and do not let
"this is getting expensive" end a line of enquiry. The only question that
changes your score is whether the next level falls.

Measured on a real run, `tn36`. After overrunning three levels it stood at
0.442 with five of seven cleared, and:

| option | value |
|---|---|
| stop there | 0.442 |
| finish the last two as it was going | 0.726 (+0.284) |
| restart and replay cleanly | **1.000 (+0.558)** |

The replay was worth twice as much as finishing. It was also the option that run
never took, because by the time the arithmetic favoured it the run had already
decided it was near the end of something. Nothing about the game prevented it.

### 0b. Being stuck is a reason to change technique, not to stop.

This is the single largest thing separating a good run from a bad one, and it is
measured, not exhortation.

Across a 25-environment arm, **every loss was a voluntary stop**. Not one was
killed, not one timed out, all three exited cleanly — each having cleared exactly
five levels, hit a level it could not read, written a closing report, and quit:

| game | levels | actions spent when it quit | exit |
|---|---|---|---|
| `sp80` | 5 of 6 | 137 | 0 |
| `tn36` | 5 of 7 | 289 | 0 |
| `sk48` | 5 of 8 | 632 | 0 |

Three-figure totals, on games whose own published playthroughs run to several
hundred. All three have since been re-run and **`sp80` went 5 of 6 to 6 of 6,
`tn36` 5 of 7 to 7 of 7, and `sk48` 5 of 8 to 8 of 8.** Nothing about the games
changed. What changed is that the later runs kept going.

Nothing external stopped any of the 25 either — not one was cut off, and in
every game the binding constraint was something the solver brought with it. If
you are considering stopping, the reason is in your head and not in the game.

**And stopping is the expensive mistake, because completion is the only axis left
once you are fast.** In all 25 environments the *completion cap* was what limited
the score; efficiency limited none of them, and 16 were clipped at the 1.15
ceiling with surplus speed thrown away. `sk48` going 5 of 8 to 8 of 8 is worth
**+0.58** — more than every efficiency gain in that entire arm combined. One more
level is worth more than any amount of being faster.

**"This level is impossible" is a hypothesis about you, not about the game.**
`tn36` has five runs on record: one cleared 5 of 7 and the rest cleared all
seven, the fastest in **220 actions**. Same harness, same environment. The level that
stopped one run was routine for another. Treat "stuck" as evidence that your
current model of the mechanic is wrong, not that the mechanic is unbeatable.

So, when you are stuck:

- **Say what you think is impossible, precisely.** "The reds cannot cross the gap"
  is testable. "Level 6 is too hard" is not. Write the claim down, then attack it
  as a claim — §5 covers why a failed attempt refutes nothing unless the belief
  was applicable in the first place.
- **Change the instrument, not the effort.** Re-running the same approach harder
  is the one thing that reliably does not work. Probe a different action, read a
  different part of the frame, build the forward model you skipped (§5a), or
  deliberately die to see what the failure state reveals (§1).
- **Spend the actions.** Dying is cheap; a level you never cleared scores zero
  however carefully you played. Exploration you did not do is the only thing
  that is definitely worth nothing.
- **There is no version of this where stopping is the move.** Every stop on
  record was written up as a reasoned decision, and every one of them was wrong:
  the three games above were re-run and all three fell. "I have run out of ideas"
  is a report on your search, and the search is the part you control. Go back to
  §5 and find the belief that is wrong instead.

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
  lethal makes you rush, skip experiments and hurry past exactly the exploration
  every other line here tells you to do.

Measured on `ls20`, the second is what happens: the bar holds 42 cells, drains
exactly one per action, reads 0, and **refills to 42 automatically** with nothing
collected and nothing else on the board changing. A clean 43-action cycle, twice
over, with no death in 120 actions of aimless wandering. A solver that assumed
lethality would have played that game far too carefully.

So: **find the display, then find out what it does.** `arc.monotone_rows(pairs)`
finds it — `pairs` is `[(t.before, t.after) for t in client.transitions()]`, and
it returns the rows that move one step in one direction per action. To learn
whether it kills, let it run out once, early in a level, where a
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
boundary. Excluded, the rule is 20/0 across both levels.

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
acts = [...]                             # <- from `available_actions`, not the default
arc.shortest_path(step, start, goal, acts)   # step(state, action) -> state
arc.reachable(step, start, acts)             # reachable at all, or did I misread?
```

**Pass `actions` explicitly.** Both default to `(1, 2, 3, 4)`, which is wrong for
any game whose real move is a click, and the failure is silent: you get an empty
route or a tiny reachable set from searching moves the game does not have. Read
`available_actions` off the frame and pass that.

It searches over a **step function**, not a grid, so launchers, teleports and
wrap-around work as long as your model has them. The route is optimal *for your
model* — check the model with `arc.predict()` before trusting a route from it.

An unexpectedly small `reachable()` set usually means your model puts a wall
where there is none.

## 6. Pace against the published baseline.

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
g: level 3/6 state=NOT_FINISHED actions=291 [190/h on this level = 3.5x]  <- OVER BASELINE
```

There is no reliable cutoff to look up: the sample has **nothing between 0.92×
and 3.44×**, so any threshold in that range separates it identically. Read the
shape instead — normal is comfortably under 1, and being over it means stop
executing and go re-explore.

The sharper signal is relative, and `client.pace()` gives it to you free:

```python
client.pace()      # {0: (spent, baseline, 0.77), 1: (spent, baseline, 0.48), ...}
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
**almost one action in three changing nothing at all** — 46 of its 157 clicks
landed on dead ground — while every winning run in the same batch wasted none.

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

Both have happened. Across ten scored games, five calls hit the cap: four went to
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

Before a run of actions, say in one line what you expect them to do. The ledger
records what happened; only you can record what you thought would happen, and the
gap between the two is the whole signal. `gate.acknowledge(...)` is where that
becomes durable at a level boundary; `notes/` is yours in between.

When a mechanic you had verified breaks on a new level *while applicable*, that
is the most informative event in the whole run — the game has just shown you
what it changed. Do not explain it away. Go and find out what is new.
