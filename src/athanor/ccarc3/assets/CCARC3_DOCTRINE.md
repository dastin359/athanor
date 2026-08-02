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

### 1a. A level may hold a depleting resource, and exhausting it kills you.

Some games give each level a finite budget — energy, time, moves, fuel — and end
it in `GAME_OVER` when it runs out.

**How it is drawn varies from game to game, so do not go hunting for a
particular widget.** There may be a bar somewhere showing how much health or
fuel you have left — or a row of tokens that vanish one at a time, or a counter,
or a shape that shrinks, or nothing visible at all. Those are illustrations, not
a checklist; treat any of them as a guess to test, never as something you
already know is there.

What is constant is the *pattern*, and the pattern is what to watch for:

- something on the board changes monotonically, once per action, regardless of
  what you did;
- and deaths arrive at a consistent action count rather than at a consistent
  place.

That second signal is the reliable one, and it needs no rendering at all. So a
death has two possible causes, calling for opposite responses:

- **Contact with something lethal** — you learned where a hazard is. Record it,
  with the location.
- **Resource exhaustion** — you learned nothing about the board and were
  probably wandering. Recording a "hazard" here is a false rule that will
  mislead you for the rest of the run.

Before concluding you found a hazard, check *how many actions* you had spent on
the level. If deaths keep landing near the same count from different places, it
is the resource, not the board.

If you suspect a depleting resource, find its display: diff a frame against the
frame one action earlier while standing still, if the game lets you. Whatever
changes when you did nothing meaningful is a strong candidate.

## 2. Never make RESET your first action after completing a level.

The server's action counter is zeroed when a level advances. A RESET at that
precise moment takes the full-reset branch: **score to zero, back to level 0,
the whole game discarded.** Any other RESET does a harmless level reset.

The condition is invisible — nothing in the frame exposes it. Take any other
action first. `ArcClient.reset()` refuses this call for you; if you see that
refusal, it just saved the run.

Check `full_reset` on every frame, not just at startup. It is the only signal
that you have lost your progress.

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

Never conclude a rule is dead because it failed on an earlier level. That is how
true knowledge gets destroyed.

**Vacuous truth is not verification.** A rule that was never applicable has been
tested by nothing.

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
