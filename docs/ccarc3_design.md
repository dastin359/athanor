# CCARC3 — a Claude-Code-as-harness design for ARC-AGI-3

**Status: design, pre-implementation.** No games have been played against the
live API (no `ARC_API_KEY` yet). Everything in §1 and §2 is verified by reading
the installed SDK; everything in §3 onward is design, and §7 lists what must be
measured before the design is trusted.

Provenance is marked throughout, because the two kinds of claim have very
different strength:

- **[SDK]** — read directly out of `arc_agi_3` 0.0.1 / `arcengine` 0.9.3.
- **[PLAY]** — the operator's direct experience playing games in the public set.
- **[DESIGN]** — proposed here, not yet validated.

---

## 1. What the SDK actually exposes [SDK]

### 1.1 The agent contract

```python
class Agent(ABC):
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool
```

`main()` is a plain loop: while not done and under budget, choose an action,
POST it, append the returned frame. `frames` is the **entire history**, handed
back on every call. There is no hidden state and no server-side memory of what
the agent "knows".

### 1.2 `FrameData`

```python
class FrameData(BaseModel):
    game_id: str
    frame: list[list[list[int]]]      # a SEQUENCE of 64x64 int grids per action
    state: GameState                   # NOT_PLAYED | NOT_FINISHED | WIN | GAME_OVER
    score: int                         # 0..254
    action_input: ActionInput          # the action that PRODUCED this frame
    guid: str | None
    full_reset: bool
    available_actions: list[GameAction]
```

Three things matter more than they look:

1. **`frame` is a list of grids, not a grid.** The engine renders every frame
   until the action completes, so one action can return several 64x64 grids.
2. **`action_input` carries the causing action.** `id` is the `GameAction`,
   `data` the x/y params. The transition `(before, action) -> after` is fully
   recoverable from the recorded frames; no extra instrumentation is needed.
3. **`ActionInput.reasoning` is an opaque client blob, stored and echoed back
   verbatim** (size-capped by `MAX_REASONING_BYTES`). This is a free slot to
   stamp each action with the hypothesis it was testing, and get it back in the
   frame. Use it — it makes the trace self-describing at zero cost.

Colours are **0-15 (sixteen)**, not ARC-AGI-2's ten.

### 1.3 The action space

`RESET`, `ACTION1`-`ACTION5`, `ACTION7` are parameterless; `ACTION6` carries
`x, y` with `Field(ge=0, le=63)`. `RESET` is itself a counted action, and
`do_action_request` attaches `card_id` to it.

### 1.4 Recording

`append_frame` writes `json.loads(frame.model_dump_json())` per frame. Disk is
already JSON, and already includes `action_input`. `pretty_print_3d` is the
SDK's text rendering: raw `[5, 5, 5, ...]` rows, 12,416 characters for a single
64x64 grid plus a header and two-space indent per grid — and one action returns
several. Unusable in-context at scale; see §4.2.

---

## 2. The two budgets [SDK]

### 2.1 The action counter is client-side and advisory

```python
# _agent.py:31
MAX_ACTIONS: int = 80  # Avoid looping forever if agent doesn't exit

# _agent.py:95-98 -- the ONLY enforcement anywhere
while (not self.is_done(self.frames, self.frames[-1])
       and self.action_counter <= self.MAX_ACTIONS):
```

Nothing server-side rejects action N+1. `do_action_request` POSTs and parses
whatever comes back. The SDK's own templates disagree on the value, which is
the tell:

| template | `MAX_ACTIONS` |
|---|---|
| `RandomAgent` | 80 |
| `LangGraph` variants | 80 / 20 |
| `smolagents` | 100 |
| **`ReasoningAgent`** (hypothesis-driven, o4-mini, effort=high) | **400** |
| `Playback` | 1,000,000 |

Two details:

- **Off-by-one.** `<=` against a 0-initialised counter yields `MAX_ACTIONS + 1`
  actions. The default is 81, not 80.
- **The counter never resets across levels.** It is per-game. A five-level
  game's RESETs and re-explorations all draw on one pool. [PLAY] confirms some
  public-set games genuinely need hundreds of actions across all levels.

**[DESIGN] Do not set a global constant at all — derive it per game.** See §2.6:
the API publishes a per-level baseline action count for every game, so the cap
is a multiple of a known quantity rather than a guess. Log `action_counter` per
game and report it honestly — `Card.actions` is a scorecard-visible metric.

This supersedes an earlier recommendation here of a flat `MAX_ACTIONS = 1000`.
That number was reasoned from the SDK's own template spread (20 to 400) with no
knowledge of real game lengths, and §2.6 shows it would truncate 5 of the 25
public games outright. The reasoning behind it survives — a guessed cap must not
be the thing that ends a run, which is the direct lesson from ARC-AGI-2's inert
iteration budget (38 of 45 runs used 1 of 8) — but the number was wrong.

### 2.6 The API publishes per-level baselines and action-type tags [SDK]

`GET /api/games` returns, for each of the **25** public games, a `baseline_actions`
list — one entry per level — and usually a `tags` field. Measured:

| | min | median | max |
|---|---|---|---|
| levels per game | 6 | 7 | 10 |
| baseline actions per game | 171 | 638 | **1843** |

Two things follow immediately.

**The stock `MAX_ACTIONS = 80` is not merely conservative, it is unusable.** The
*shortest* game in the set needs 171 baseline actions. Several individual levels
exceed 80 on their own, and `dc22-fdcac232` has a single level with a baseline of
**578**. An 80-action agent is not playing these games; it is sampling their
opening moves. This is the operator's play experience confirmed with numbers.

**`baseline_actions` is a planning signal, not just a budget.** It is per level
and known before you start, so it says roughly how long a level *should* take
when you already understand it. That gives a control law the solver otherwise
lacks: **at a large multiple of the level's baseline, the working hypothesis is
probably wrong — stop grinding and go re-explore.** Without it, a solver has no
way to distinguish "this level is long" from "I have misunderstood it", and will
happily burn a thousand actions executing a wrong plan.

**`tags` prunes the action space for free.** Of 25 games: 13 `keyboard_click`,
7 `click`, 4 `keyboard`, 1 untagged. A `click` game is telling you `ACTION6(x,y)`
is what matters; a `keyboard` game is telling you it is not. Discovering that by
experiment costs actions that the tag gives away.

### 2.2 Retry economics: score is best-of, actions are cumulative

Retries are first-class. `Card`'s docstring: *"A game can be played more than
once, we track each play with lists of card properties (scores, states,
actions)."*

```python
Card.high_score    = max(self.scores)
Card.total_actions = sum(self.actions)                          # across plays
Scorecard.score    = sum(g.high_score for g in self.cards.values())
Scorecard.won      = sum(GameState.WIN in g.states for g in self.cards.values())
```

So:

- A game counts as **won if any play won**.
- Total score is the sum of **per-game bests**.
- **Actions are summed across every play.**

**A GAME_OVER costs nothing in score.** And it does not end the run: every
template's `is_done` returns `state is GameState.WIN` only; GAME_OVER merely
obliges a RESET (`_random.py:37`, `_langgraph.py:70`, `_llm.py:288`). Die,
RESET, continue, counter still climbing.

[PLAY] Humans routinely hit several GAME_OVERs before finding the rules. The SDK
agrees: `_llm.py:580` carries the game-specific hint *"start each level with
limited energy. you GAME_OVER if you run out."*

**[DESIGN] This is a licence to die on purpose, and our doctrine must say so
explicitly, because the SDK's own prompts say the opposite** — all of them read
*"Your objective is to WIN and avoid GAME_OVER while minimizing actions."* Under
max-semantics that is wrong as a terminal goal. Death is only *instrumentally*
costly, through walk-back. An agent told to avoid death will under-explore, and
will do it on level 1, where exploration is cheapest and most valuable.

A death is also a **labelled negative transition** — the highest-information
kind — and feeds §5's refutation log nearly for free.

### 2.3 RESET after GAME_OVER restarts the level, not the game [PLAY]

This is what makes the licence in §2.2 usable rather than theoretical.

The obvious worry about exploring by dying is that walk-back cost compounds with
depth: die on level 4 at action 200 and you would replay all 200. That does not
happen. **Death costs only the actions already spent inside the current level.**

Three consequences, and they are load-bearing:

1. **Death stays cheap at every depth**, not only on level 1. Exploring by dying
   is a globally valid strategy, not an opening gambit.
2. **The cost of a death scales with the current level's traversal length, not
   its index.** A long level is expensive to die in; a deep but short one is not.
   So the thing to economise is *distance from the level's start when you run the
   experiment*, which is directly controllable — test lethality hypotheses early
   in a level, not at its far end.
3. **The level is the natural unit of the experiment loop**, which is
   independent support for the level-boundary gate in §6.1.

**Corroborated independently by the engine source** [SDK]. `ARCBaseGame` treats
the two as different operations, and `handle_reset` picks between them:

```python
# arcengine/base_game.py
def handle_reset(self) -> None:
    if os.getenv("ONLY_RESET_LEVELS") == "true" and self._state != GameState.WIN:
        self.level_reset()
    elif self._action_count == 0 or self._state == GameState.WIN:
        self.full_reset()
    else:
        self.level_reset()
```

So a mid-game RESET — which is every RESET after a GAME_OVER — takes the
`level_reset()` branch. And `level_reset()` restores the current level from a
clean clone while leaving `_score` and `_action_count` untouched, whereas
`full_reset()` zeroes both, returns to level 0, and sets the `_full_reset` flag
that surfaces as `FrameData.full_reset`.

That resolves the residual question outright: **`full_reset` on the frame is the
reliable discriminator** between a level restart and a game restart. There is
also an `ONLY_RESET_LEVELS` environment switch, so the behaviour is deployment
configurable — which is presumably why the public games behave as observed.

**But there is a trap in that `elif`, and it is expensive.** `_action_count` is
zeroed when a level advances. So a RESET issued as the *very next action after
completing a level* meets `_action_count == 0` and takes the `full_reset()`
branch — scoring back to zero, back to level 0, whole game discarded. Every
other RESET at that same moment in the game does a harmless level reset.

The condition is invisible from the outside: nothing in `FrameData` exposes
`_action_count`, so the solver cannot query it. It can only be inferred ("did I
just advance a level?") or read back after the fact from `full_reset`.

Found by the local bench, which lost a completed level to it before the cause
was clear. Two consequences:

- **Doctrine: never make RESET the first action after a level advance.** If a
  level reset is wanted at that moment, take any other action first.
- **`full_reset` must be checked on every frame, not just at startup.** It is the
  only signal that the run just lost its progress.

### 2.4 Actions issued after a death are silently wasted [SDK]

A budget trap worth designing against. `perform_action` short-circuits:

```python
if action_input.id == GameAction.RESET:
    self.handle_reset()
elif self._state == GameState.GAME_OVER or self._state == GameState.WIN:
    return FrameData(..., frame=[], ...)   # no step, no frames
```

Any non-RESET action while dead returns an **empty frame** without advancing the
game — and still costs an action. A solver that does not notice it has died
burns budget at full rate producing nothing.

`ledger.load()` therefore keeps these records rather than dropping them, flagged
`Transition.wasted`, so "actions burned between a death and noticing it" is a
number readable straight off a trace. It is exactly the kind of quiet overhead
that a pooled success rate would never show.

### 2.5 Score is a count of completed levels [SDK]

`next_level()` is the only thing that touches the score and it adds exactly 1.
The engine's own `FrameData` renames the field `levels_completed` to say so.

Note the version skew: `arcengine` 0.9.3 sends `levels_completed`, while
`arc_agi_3` 0.0.1 sends `score`. A reader that knows only one of them records
zero for every frame produced by the other, silently. `TraceWriter` accepts
both.

---

## 3. Memory: two tiers with different lifetimes [DESIGN]

[PLAY] Once a level is cleared, its frame-by-frame trace stops mattering for
play. But the rules and priors discovered on earlier levels of the **same game**
matter enormously.

The correct reading of that is: **frames should leave context; they must not
leave disk.**

| tier | lifetime in context | lifetime on disk | volume | file |
|---|---|---|---|---|
| transition ledger | current level | whole game | high | `trace.jsonl` |
| rule book | whole game | whole game | low | `rules.json` |

This split is something a harness can *enforce* and a prompt instruction cannot:
the agent physically cannot hold 300 frames, but a function call can read all of
them.

---

## 4. What the harness ships, and what the agent writes [DESIGN]

**The harness ships the converter. The agent must not write it.** Three reasons:
every run would re-derive the same loader differently, destroying cross-run
comparability (the 43-function `arc_toolkit` precedent); it is plumbing, not
thinking — code-as-thinking means the agent writes *analysis*, not
`np.asarray`; and conversion is exactly where a silent bug (dropping
intermediate frames of a multi-frame action, mis-inferring the block size)
poisons every downstream conclusion invisibly.

### 4.1 Ledger and loader

- `trace.jsonl` — append-only, one record per action, a *view* over the frames
  the SDK already records. Same shape as athanor's `.athanor/invariants.jsonl`.
- `arc3.trace()` — returns `Transition` objects with `.before` / `.after` as
  `np.ndarray(64, 64)`, plus `.action`, `.params`, `.score_delta`, `.level`.

### 4.2 Representation primitives

- `arc3.logical(grid)` — infer the render block size and collapse 64x64 to the
  logical board. Games are *drawn* at 64x64 but the board is far coarser. This
  is the single largest token win available, and the agent should not be
  eyeballing the block size.
- `arc3.render(grid)` — one char per cell (16 colours -> `0-9a-f`), so a grid is
  64 lines of 64 chars (~1k tokens) instead of ~6k as bracketed int rows.
- `arc3.diff(a, b)`, `arc3.objects(grid)` — changed cells, connected components.

---

## 5. Mechanics vs rules, and three-valued predicates [DESIGN]

### 5.1 The scope split

[PLAY] Rules do **not** transfer 1:1 across levels — a level-3 rule often fails
on level 1 because level 1's setup differs. But the levels carry strong logical
lineage.

What transfers is one abstraction level up from the rule:

| layer | scope | example | transfers? |
|---|---|---|---|
| mechanic | game | "there is a pushable-object mechanic" | strongly |
| rule | level | "the block at (12,30) moves 1 right when pushed" | no |

Entering level 4 you do not port level 3's *rules*. You port its *mechanics*, as
**priors about what to test first**.

**Prior-level evidence buys prioritisation, not verification.** A mechanic seen
instantiated on L1, L2 and L3 is a high-prior hypothesis on L4 — worth two
actions to confirm rather than fifteen to discover. It sets the *search order*,
never the *truth value*. That is the honest form of the carry-over payoff.

Promotion runs the other way and is earned: when the same mechanic is
independently instantiated on k levels, confidence in the mechanic rises. It is
measured, not declared.

### 5.2 Predicates return three values, not two

`HOLDS` / `VIOLATED` / `NOT_APPLICABLE`. **Only `VIOLATED` is evidence against.**

Most cross-level "failures" are vacuous: the rule references a red block and
this level has no red block. That is an unmet precondition, not a refutation,
and a boolean predicate cannot tell them apart — it returns `False` for both and
a true rule dies.

This is the same bug `arc.unreached()` exists to catch in the ARC-AGI-2 toolkit,
where a branch that never executed was passing itself off as a branch that
passed. The general form is already in the doctrine as *"unanimity is not
verification."* Its ARC-AGI-3 sibling is **"failure is not refutation unless the
rule was applicable."**

### 5.3 Refutation authority is level-local

Two functions, because they have different powers, and conflating them is
exactly the error that destroys knowledge:

```python
arc3.verify(rule)   # current level only. CAN refute.
arc3.survey(rule)   # every level. reports per-level HOLDS/VIOLATED/N-A. CANNOT refute.
```

`survey` output reads like `L1: 0/0/47 | L2: 12/0/19 | L3: 31/0/0` — "applicable
from L2 onward, never violated where applicable", a strong rule. Whereas
`L1: 40/7/0` is a genuine violation and deserves the run's full attention.

`verify` is the ARC-AGI-2 `check()` analogue: a rule graduates from hypothesis
to verified not because the agent believes it, but because it reproduces
recorded history. Cheap, callable without limit. Everything else in this
document exists so that calling it is one line.

### 5.4 Continuous regression, and what a break means

Re-check every verified rule against each new transition automatically — a
regression suite for the world model, running continuously. Levels escalate by
introducing roughly one new mechanic, so a game-scoped mechanic that is
**applicable and violated** on a new level is the highest-signal event available:
the game deliberately changed something, and it has just told you where.

A mechanic that is merely `NOT_APPLICABLE` on the new level should be silent.
Boolean predicates would fire on both and train us to ignore the alarm.

### 5.5 The refutation log

`rules.json` needs a `refuted` section, not only `verified`. "ACTION5 has no
observable effect in any state seen" is worth ~5 actions on every subsequent
level if remembered, and costs ~5 per level if forgotten. Negative knowledge is
cheaper to carry than positive knowledge and transfers at least as well.

---

## 6. Gates and doctrine [DESIGN]

### 6.1 The level boundary is a free forcing function

It is the one moment the agent is guaranteed to have just learned something and
to be about to need it. Make it structural: **block the first action of a new
level until the rule book has been updated.** Same shape as the ARC-AGI-2
submission gate — you do not proceed until you have stated what you know. We
have evidence that shape works; it is what the train-100% requirement was doing.

### 6.2 Over-explore level 1, deliberately

Because the action counter is per-game and never resets, and score is best-of
while actions are cumulative, **level 1 is where you buy knowledge and levels
2..N are where you spend it.** The optimal policy explores level 1 well past
what is needed to clear it.

This belongs in the doctrine explicitly, because every instinct pushes the other
way: a myopic agent clears L1 in 12 actions and then burns 300 relearning across
L2-L5.

The corollary from §2.2 and §2.3: **use deaths as experiments, on every level.**
Dying costs no score, and RESET replays only the current level, so the price is
bounded by how far into the level you were. Run lethality experiments near a
level's start, where that price is near zero.

---

## 7. Measure before trusting [DESIGN]

Ordered by how much of the above they invalidate if they come out wrong.

*Resolved:* whether RESET after GAME_OVER restarts the game or the level. It
restarts **the level** [PLAY] — see §2.3, which is now the basis for §6.2 rather
than an open risk to it.

1. **Are levels really independent in state?** `score` is cumulative across
   levels; inventory, position conventions or palette may also persist. Cheap to
   check via `full_reset` and score continuity. §3's eviction policy assumes
   independence and should not until this is measured.
3. **Does a RESET after GAME_OVER open a new `Card` play row, or continue the
   current one?** `do_action_request` attaches `card_id` to RESET and forwards
   any existing `guid`; the resolution is server-side. Affects how
   `Card.scores` / `Card.actions` should be read.
4. **What is the actual distribution of actions-to-solve?** With `MAX_ACTIONS`
   at 1000 this becomes observable instead of truncated. It is the number that
   should set every later budget.
5. **What block size do real games render at?** `arc3.logical()`'s inference
   needs validating against actual games, not the toy.

---

## 8. Where this lives

Undecided, and deliberately so until §7.1-§7.2 are measured. The seam analysis
is in `cc_harness_results.md`. What is already clear: the *submission gate*
concept survives the port (as §6.1's level gate), the *toolkit* concept survives
(as §4), the *doctrine* concept survives (as §6.2), and the *scoring* module
does not — ARC-AGI-3 scores are server-side and best-of, with no local
ground truth to compare against. That is a large enough shared spine to argue
for the same repo and a different package.

---

## Appendix: verified local loop

`scratchpad/arc3/toy_game.py` drives a minimal `arcengine` game RESET -> actions
-> WIN with no API key, confirming that the engine can be driven standalone and
that frames come back as the integer grids `FrameData` promises. `arcengine`
allows local authoring without a key; the live API returns 401 without
`ARC_API_KEY`.
