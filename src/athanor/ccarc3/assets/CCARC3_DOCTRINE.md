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

Check `full_reset` on every frame, not just at startup. It is the only signal
that you have lost your progress.

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

## 5. Failure is not refutation unless the rule was applicable.

Every rule you write has a precondition and a claim, and they are separate. A
rule about a red block, checked on a level with no red block, returns
`NOT_APPLICABLE` — not `VIOLATED`. Only `VIOLATED` is evidence against.

```python
verify(rule, transitions)   # THIS level only. Can refute.
survey(rule, transitions)   # every level. Reports. CANNOT refute.
```

Use `survey` to see where a rule holds — `L0: 0/0/47 | L1: 12/0/19 | L2: 31/0/0`
reads "applicable from L1 onward, never violated where applicable", which is
strong. Use `verify` when you want a verdict, and only against the level the
rule is about.

**Exclude level boundaries from spatial rules.** The action that completes a
level returns the *next* level's board, so `before` and `after` are different
boards entirely. A movement rule checked across one sees the avatar teleport and
reports a violation that never happened. Use **`not t.board_replaced`** in the `applies` of any rule about position,
movement or adjacency. It covers both causes: completing a level, and a full
reset that rewinds the game to level 0. (`t.crosses_level` alone catches only
the first — a full reset moves the level *down*.)

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

## 6. Budget against the published baseline.

Every game publishes `baseline_actions` — one figure per level, what a
playthrough costs when the rules are *already known*. Real games range from 171
to 1843 total.

Use it as a control law: **if you are at several times the baseline for the
level you are on, your hypothesis is probably wrong.** Stop executing and go
re-explore. Without this you cannot tell "this level is long" from "I have
misunderstood this level", and you will spend hundreds of actions on a wrong
plan.

## 7. Trust `available_actions` over everything else.

Each frame tells you exactly which actions this game accepts. It is
authoritative. Do not spend actions discovering that ACTION5 does nothing when
the frame already told you it is not available.

`win_levels` tells you how many levels the game has. There is no `score` field —
the server sends `levels_completed`. Anything reading `score` gets 0 forever.

## 8. Write code to analyse; do not eyeball frames.

Every action is recorded to a ledger with the action that caused it. Frames are
64×64 over sixteen colours and carry real pixel detail — reading them by eye is
both expensive and unreliable.

Ask questions in code:

```python
ts = client.transitions()
# Which actions ever changed anything?
{t.action for t in ts if t.changed}
# What did ACTION3 do, every time?
[diff(t.before, t.after) for t in ts if t.action == "ACTION3" and t.changed]
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
