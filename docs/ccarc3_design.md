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
SDK's text rendering: raw `[5, 5, 5, ...]` rows, roughly 6k tokens for a single
64x64 grid. Unusable in-context at scale; see §4.2.

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

**[DESIGN] Set `MAX_ACTIONS = 1000`** — not to buy headroom, but so the counter
stops being what ends a run. The real budget is then explicit and ours
(wall-clock, tokens), and the action count becomes a *measured* quantity rather
than a guessed cap. This is the direct lesson from ARC-AGI-2, where the
iteration budget turned out to be inert (38 of 45 runs used 1 of 8) because the
binding constraint was somewhere else entirely. Log `action_counter` per game
and report it honestly — `Card.actions` is a scorecard-visible metric.

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

The corollary from §2.2: front-load cheap deaths. Dying costs no score, and on
level 1 the walk-back is short.

---

## 7. Measure before trusting [DESIGN]

Ordered by how much of the above they invalidate if they come out wrong.

1. **After GAME_OVER, does RESET restart the game or the level?** This decides
   whether "explore by dying" stays cheap past level 1. Dying at L1 step 3 is
   nearly free; dying at L4 step 200 could cost 200 actions of re-traversal.
   `full_reset: bool` implies the distinction exists, but the semantics are
   server-side. **Everything in §6.2 depends on this.**
2. **Are levels really independent in state?** `score` is cumulative across
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
