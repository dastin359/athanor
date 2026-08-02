# Solver doctrine — ARC-AGI-3

Everything here was measured, not assumed. Where a line contradicts what seems
obvious, the contradiction is the point: several of these were learned by losing
progress to them.

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
