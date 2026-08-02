# CCARC3 — a Claude-Code-as-harness design for ARC-AGI-3

**Status: implemented and running.** `athanor.ccarc3` is built and tested, and
games have been driven against the live API. The design below is kept in the
order it was reasoned, including the parts that were **wrong** — §4.2 and §2.5
carry corrections rather than quiet rewrites, because what the reasoning missed
is more useful than a tidy document.

Provenance is marked throughout, because the kinds of claim have very different
strength:

- **[SDK]** — read directly out of `arc_agi_3` 0.0.1 / `arcengine` 0.9.3.
- **[PLAY]** — the operator's direct experience playing games in the public set.
- **[LIVE]** — measured against the live API with a real key.
- **[DESIGN]** — proposed here; most is now built, see §8.

## What is actually established

Enough claims here have been overturned by their own follow-up measurements
that the strength of each is worth stating plainly. Read this before quoting
anything below.

**Well supported.**

| claim | evidence |
|---|---|
| The games are deterministic — a recorded action sequence replays frame-for-frame | [LIVE] 40/40 identical on `ls20`, including its patrolling items |
| A RESET with the action counter at zero starts a **new play**; per-level actions are recorded per play and never summed | [LIVE] scorecard probe: `plays` 1→2, new guid, new `actions_by_level` row |
| The leaderboard metric is RHAE, not the SDK's `sum(high_score)` | [LIVE] **7 of the 25 published Opus 5 scores land exactly on RHAE completion-cap fractions** — 47.6% = 10/21 (twice), 77.8% = 28/36, 28.6% = 6/21, 8.3% = 3/36, 3.6% = 1/28, 1.8% = 1/55. A levels-completed metric cannot produce them |
| Score is best-of across plays, actions are summed. Dying costs no score. | [SDK] read from source; the whole doctrine turns on it |
| RESET as the first action after a level advance discards the entire game | [LIVE] observed, and it cost a won game (§9.9) |
| A solver that understands a level finishes it under the published baseline | [LIVE] 24 of 25 cleared levels at ≤0.92×, median 0.52× |
| Solvers do not use the rule engine, the forward model, or the planner | [LIVE] **seven consecutive runs, zero calls**, all advertised (§9.8a) |
| A guard's arming state must persist, or it silently stands down | [LIVE] one occurrence, one lost game, mechanism fully traced |

**Corroborated externally** — by ARC's own technical report, not by this project.

| claim | evidence |
|---|---|
| Code-over-the-trace is the right shape for a harness | The report describes a Duke University harness built on the same idea — *"allowing the model to execute arbitrary Python code to selectively retrieve and transform information from its action history"* — noting that *"maintaining a naive rolling window of observations quickly exhausts a model's context budget"*, and that it *"was able to solve all three public environments with action counts comparable to human performance."* CCARC3 arrived at the same design independently, from measuring what solvers actually did |
| The engine is deterministic and traces re-execute faithfully | *"Known-good recordings are replayed under both win and loss conditions, confirming that the engine can serialize and faithfully re-execute action traces."* Matches the 40/40 frame test here, from the engine's side |
| **Our wins are not luck** | ARC fuzz-tests every environment: a **1,000,000-step** uninformed-random sweep, with the constraint that *"non-tutorial levels must remain unbeaten under uninformed random play."* A level cleared here was cleared by understanding it |

**Provisional — one observation, or a rank order without a cutpoint.**

| claim | why it is weak |
|---|---|
| The server's `full_reset` flag can read false when a full reset happened | n=1. Distrusting it is the safe reading regardless |
| A post-`GAME_OVER` RESET resets the level, not the game, and opens no new play row | n=1 game (`r11l`) |
| Wasted actions track failure | rank-ordered across 7 runs, but a run won at 93% effective. **No threshold.** |
| Deaths are unreachable by wandering on some games | 4 games probed with a random policy |

**Not established, and stated here because the tempting reading is wrong.**

- **Which play the scorer uses.** That plays are stored separately, each with
  its own `actions_by_level`, is measured. That the *best* one is scored is an
  inference from `Card.high_score = max(scores)` — a property of the SDK's
  client-side bookkeeping, not of RHAE. The whole explore-then-replay strategy
  turns on it: best-play makes a replay worth `1.0 − raw`, first-play makes it
  worth nothing. **Nothing in this project relies on the strategy**, and it
  should stay that way until this is measured.

- **That the harness changes caused any measured improvement.** `cd82` went from
  0/6 in 337 actions to 6/6 in 121 across a harness change — and the winning run
  called the new function *twice* in 83 tool blocks. Confounded, n=1.
- **That run-to-run variance is small.** The only estimate (`ls20` replaying
  levels 0–5 in exactly 369 actions, twice) came from re-walking a route already
  known, which is the lowest-variance case obtainable and says nothing about a
  novel exploration.

**Refuted, having once been stated here as findings.**

- *A wide action space is what breaks the solver.* `cd82` won using six action
  types.
- *Exhausting a per-action display kills you.* On `ls20` it is a 43-action cycle
  that refills itself.
- *The click space is harder to search than the keyboard.* Click-only games are
  among the most efficient runs on record.
- *Downscaling frames is the big token saving.* Real frames rarely have uniform
  block structure; `logical()` usually declines to reduce at all.

## What RHAE actually measures — skill acquisition, not mastery

**Corrected.** An earlier version of this section argued that RHAE measures
mastery rather than learning speed, on the grounds that `h_l` came from humans
who had already worked the game out. That is wrong, and the methodology says so
in as many words:

> "Human baselines are established through controlled testing where
> participants play each ARC-AGI-3 game **for the first time (having never seen
> the game before)**."
> — [docs.arcprize.org/methodology](https://docs.arcprize.org/methodology)

The baseline is a **first-time** player's action count. It contains that
human's own fumbling, wrong turns and discovery. So `(h_l / a_l)²` compares
*your* cost of learning the level against *a human's* cost of learning the same
level. RHAE measures **skill-acquisition efficiency**, which is what the
benchmark's own summary claims: *"Skill-acquisition efficiency over time"*, and
*"A 100% score means AI agents can beat every game as efficiently as humans."*

### What that does to the replay strategy

ARC settles it directly. Their technical report says they *built* the thing:
*"we are releasing an open-source 'harness' which scores 100% on all public
environments, using human replay"* — offered as a demonstration that public-set
scores are meaningless, not as a technique. And the human protocol they
baselined against allowed level resets but **"participants were limited to a
single attempt per environment and could not revisit previously completed
levels."**

The metric's own definition is first-exposure: *"Counting the total number of
actions taken on **first exposure** to beat an environment accounts for both"*
exploration and execution. Both are supposed to be counted.

It removes its justification. Explore expensively, restart, then execute a
clean route, and you are comparing a **post-learning** agent run against a
**during-learning** human baseline. That is not a like-for-like comparison; it
is the metric's central quantity replaced with something else.

Whether the scorer would even reward it is still unmeasured — which play it
reads is unknown (see the scoreboard above). But the intent is no longer
ambiguous, and that is sufficient. **Nothing in this project uses it, the
capability stays behind an explicit call the solver has to choose to make, and
the doctrine now says not to.**

The honest form of the benchmark is the one already being run: one play, learn
from nothing, and let the action count fall where it falls.

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
    score: int                         # 0..254 -- NOT SENT BY THE LIVE API, see 2.5
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
`x, y` with `Field(ge=0, le=63)`. `do_action_request` attaches `card_id` to `RESET`. The SDK's code reads as
though every RESET is a counted action; live, the *opening* one is free and
later ones are billed (§7.2).

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

### 2.5 The score field does not exist — and the stock SDK is stale [SDK, LIVE]

`next_level()` is the only thing that touches the score and it adds exactly 1,
so the score *is* a count of completed levels. Both the engine and the live API
name the field accordingly.

**The installed `arc_agi_3` 0.0.1 does not.** Verified against the live API, a
frame comes back with exactly these keys:

```
action_input  available_actions  frame  full_reset
game_id  guid  levels_completed  state  win_levels
```

There is no `score`. But `FrameData.score` is declared `int = Field(0, ...)`, so
pydantic fills the default and drops the unrecognised `levels_completed`.
**Anything reading `agent.score` or `frame.score` against the live API gets 0
forever, silently.** The stock templates survive only because they key `is_done`
on `state is WIN` rather than on score. Any score-based progress logic is
already broken before it is written.

`TraceWriter` reads `score` or `levels_completed`, whichever is present. That
started as defensiveness about an `arcengine` version skew and turns out to be
load-bearing against the live server.

Two more fields worth using, both per frame and both free:

- **`win_levels`** — the total number of levels in the game. Progress is
  `levels_completed / win_levels` without a lookup.
- **`available_actions`** — the actions this game accepts, e.g. `[1, 2, 3, 4]`
  for `ls20`. This is *authoritative* and strictly better than inferring from
  the `tags` field in §2.6. Use the tag for planning before a run; use this
  during one.

The scorecard adds **`actions_by_level`**, which is directly comparable to the
published `baseline_actions` — so the §2.6 control law ("am I over budget for
this level?") can be evaluated live rather than reconstructed.

### 2.6 The API publishes per-level baselines and action-type tags [LIVE]

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

- `arc3.render(grid)` — one char per cell (16 colours -> `0-9a-f`). 4,159
  characters against 12,416 for bracketed int rows, measured. This is the token
  win that actually survived contact with real games.
- `arc3.diff(a, b)`, `arc3.objects(grid)` — changed cells, connected components.
- `arc3.logical(grid)` / `arc3.collapse(grid)` / `arc3.cell_boundaries(grids)` —
  downscaling. **Read the correction below before relying on any of them.**

#### Correction: real frames are not coarse boards scaled up [LIVE]

This section previously claimed that collapsing 64x64 to "the logical board" was
*the single largest token win available*. Measured against a real game
(`ls20-9607627b`, 50 scripted actions), that is **wrong**:

| | toy 10x10 game | real `ls20` frame |
|---|---|---|
| `block_size()` | 2 | 1 |
| `logical()` | 32x32 | 64x64 (no-op) |
| `collapse()` | 5x5 | 27x35 |
| `cell_boundaries()` | 12 x 12 | **28 x 57** |

57 distinct column boundaries out of a possible 64 means adjacent columns almost
always differ. Real ARC-AGI-3 frames carry genuine pixel-level detail — 9
distinct colours and 18 connected components in the opening frame alone — rather
than a coarse board painted in blocks.

So the downscaling family is **not** the headline saving. `block_size()` and
`logical()` are exact and will simply decline to reduce a real frame, which is
the correct behaviour and also means they earn little. `collapse()` still halves
each axis and preserves structure, but destroys metric position, so it is a
summarising view and not a substitute for the frame.

What survives: `render()`'s 3x, and being selective about *which* frames enter
context at all (§3). The design's memory tiering matters more than its
compression, which is the opposite of what this section originally assumed.

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

## 7. Measure before trusting

**Resolved since this list was written:**

- *Does RESET after GAME_OVER restart the game or the level?* **The level**
  [PLAY], corroborated by `handle_reset` in the engine source. §2.3. This was
  the item everything in §6.2 depended on.
- *What block size do real games render at?* **None** [LIVE]. Real frames carry
  pixel-level detail and have no uniform block factor, so `logical()` correctly
  declines to reduce them. See the correction in §4.2 — this one came out
  against the design rather than for it.
- *What is the distribution of actions-to-solve?* Published, not inferred:
  `baseline_actions` per level for every game, 171 to 1843 per game. §2.6.

**Still open, in the order they would hurt:**

1. ~~**Are levels really independent in state?**~~ **Answered** [LIVE], from a
   real four-level `ls20` trace:

   - **The board is 64-76% identical across a boundary.** What changes is the
     playfield; what persists is chrome — panels, borders, the resource display.
     So a boundary transition is not a total replacement, but the part a spatial
     rule cares about *is* replaced, and `board_replaced` is still the right
     guard.
   - **The per-level resource resets exactly.** Row 61 holds 42 yellow cells at
     the first frame of every one of levels 0-4, and fewer at each level's last
     frame. So the energy budget is per level, refilled at the boundary — which
     is what makes §6.2's "explore early in a level" advice correct rather than
     merely plausible.
   - **A new colour appears at level 2** (14, green) and persists thereafter.
     Levels escalating by introducing roughly one new element is not just an
     assumption about the benchmark; it is visible in the palette.

   §3's eviction policy assumed independence and is close enough: frames may
   leave context at a boundary, because the playfield is genuinely new.

2. ~~**Does a RESET after GAME_OVER open a new `Card` play row?**~~ **Answered
   [LIVE]. No.** A scripted probe of `r11l-495a7899` died at action 38 having
   completed level 1. Across the death and the RESET that followed it:

   | | before RESET | after RESET |
   |---|---|---|
   | `total_plays` | 1 | **1** |
   | `levels_completed` | — | **[1]** |
   | `full_reset` on the frame | — | **not set** |

   Three things follow. A play row belongs to a *scorecard*, not to a life, so
   `Card.scores` and `Card.actions` are per-play and dying inside a play adds no
   entry. The RESET was a **level** reset — §2.3 replicated on a second game and
   a different tag. And **the completed level survived the death**, which is the
   direct confirmation of §2.2's central claim that dying costs progress
   nothing.

   Also established while probing: deaths are genuinely reachable on some games
   (38 random clicks sufficed on `r11l`) and apparently not on others — 71
   random actions each on `cd82`, `ft09` and `sb26` produced none.

   Residual: the scorecard's `states` still read `['GAME_OVER']` after the
   RESET. One sample, not worth a conclusion — but do not read `states` as the
   live state.

3. **How is a depleting per-level resource represented?** Confirmed to exist
   [PLAY], and confirmed to vary between games, so the doctrine deliberately
   describes the *pattern* (something changing monotonically per action; deaths
   at a consistent action count rather than a consistent place) instead of any
   widget. Whether the pattern is reliably detectable from frames alone is
   untested.

---

## 8. Where this lives — decided

`athanor.ccarc3`, in this repo, as a separate package from `cc_harness`. The
seam analysis argued for it and building it confirmed the shape:

| CCARC (ARC-AGI-2) | CCARC3 | |
|---|---|---|
| submission gate | `LevelGate` | survives — §6.1 |
| 43-function toolkit | `grids`, `ledger`, `rules` | survives — §4 |
| solver doctrine | `CCARC3_DOCTRINE.md` | survives — §6.2 |
| workspace + runner | `session.py` | survives |
| `scoring` against ground truth | *(none)* | does not port |

Scoring is the one piece that does not survive: ARC-AGI-3 scores are
server-side and best-of, with no local ground truth to check against. Its role
— making a claimed result and a real one the same thing — is taken over by the
trace, which is why `collect_outcome` reads the ledger and never the solver's
own account.

One thing the port did *not* inherit, and should not: `arc_agi_3` itself. The
SDK reads a `score` field the live server does not send (§2.5), so speaking
HTTP directly is both more correct and keeps the package free of any runtime
dependency beyond numpy.

---

## 9. What building it taught — findings about the harness, not the game

Everything above §8 is about ARC-AGI-3. These are about *Claude Code as a
harness*, and they generalise past this benchmark. Each was found by a real
solver run failing, and none of them would have appeared in a unit test.

### 9.1 A CC solver is a sequence of processes, not a process

The single most important structural fact. An agent works in one-shot
`python -c` commands, so **every action arrives in a brand new interpreter**.
Any harness object holding a session, a connection, a counter or an open handle
must persist to disk and resume, or it silently resets on every call.

This harness had two failure modes stacked on it before the fix. Each
construction deleted the trace file as a stale-run guard, and each `open()`
requested a fresh scorecard — so the agent restarted the game on every command
and truncated the record as fast as it wrote it, while every individual command
looked like it worked.

The general rule: **a harness API for a CC agent should be idempotent to
construct and resumable by default.** Design for "the caller has no memory",
because it does not.

### 9.2 `--allowedTools` grants permission; the permission mode alone does not

`acceptEdits` approves file writes but not Bash. A run configured with only a
permission mode reaches the model, thinks, writes files, and then takes **zero**
actions, because everything that touches the outside world is denied. It does
not look like a permissions failure; it looks like a solver that would not act.

CCARC had already hit this and documented it in `cc_harness/config.py`. It was
re-derived here anyway. The fix was to *import* those lists rather than restate
them — a second copy of a hard-won constant is a second chance to get it wrong.

### 9.3 Refusals are a better interface than instructions

Three things in this harness are enforced rather than advised: the action
budget, the RESET that would discard the game, and acting while dead. All three
began as documentation, and the documentation was correct and insufficient.

A refusal costs no budget, arrives at exactly the moment it is relevant, and
carries its reason. An instruction in `CLAUDE.md` competes with everything else
in context at the moment it matters most. Where a mistake is unrecoverable —
`full_reset` discards an entire game with no undo — advice is not an
appropriate mechanism.

This is the ARC-AGI-2 gate's lesson in a new setting: the submission gate worked
not because it told solvers to check their work, but because it declined to
accept work that had not been checked.

### 9.4 Cleanup must never mask the failure it follows

A 404 from `scorecard/close` in `__exit__` replaced the real exception from the
run, twice, before it was noticed. The information asymmetry is total: the body
error is why the run failed, and the cleanup error is trivia. Cleanup paths in a
harness should swallow and record, never raise.

The same instinct applies to error text. `HTTP Error 400: Bad Request` cost more
time than any other single message here; the server was putting the reason in
the response body, and urllib discards it unless you read it off the exception
before the handle closes. Once surfaced, the actual message — `game <id> not
found` — pointed straight at the bug in one step.

### 9.5 Advice that survives contact should become a function

The doctrine started as prose. Three pieces of it turned out to be mechanisable,
and each became better as code than as instruction:

| doctrine line | became | what it bought |
|---|---|---|
| "failure is not refutation unless applicable" | three-valued `Rule` | the solver cannot collapse the cases by accident |
| "watch for something ticking once per action" | `monotone_rows()` | found `ls20`'s energy bar in one call, unaided |
| "understand the mechanics" | `predict()` | a testable claim instead of a feeling |

The `monotone_rows()` case is the clearest. As prose it asked the solver to
notice a subtle statistical pattern across a hundred frames while also playing
the game. As a function it is one call that returns rows 61 and 62 — the energy
bar — without being told it existed or what it looked like.

The rule that emerges: **write the doctrine first, try it on real data, and
promote whatever survives into the toolkit.** Prose that cannot be mechanised is
usually prose about judgement, which is what the doctrine should be left holding.

### 9.6 The most damaging bug is a false positive in your best signal

§5.4 designates one event as the highest-signal thing in a run: a game-scoped
mechanic that is *applicable and violated*. That designation is what made a
false one so expensive.

The ledger labelled level-boundary transitions like any other, so a movement
rule checked across a boundary saw the avatar teleport and reported exactly that
event. `"ACTION1 moves the cursor up"` read 7/0 on level 0 and 13/1 on level 1;
the single violation was the board being swapped wholesale, 1467 cells at once.

The general form: **whatever your system treats as its strongest evidence needs
the strictest guard against manufacturing it.** A weak signal that misfires is
noise. A strong signal that misfires sends you somewhere specific and wrong.

### 9.7 Measure which of your abstractions the solver actually used

Across a 370-action run the solver used `Rule`, `verify` and `survey` **zero**
times. All three were documented in the workspace it was given. §5 is the most
carefully-reasoned part of this design and it went untouched.

What it wrote by hand instead, in its own scratch directory: domain parsers, a
forward simulator, and breadth-first search — twice, the second time after
discovering launchers made its grid version wrong.

The reason is structural, and it is the single most useful thing this build
taught. On ARC-AGI-2 a hypothesis cannot be tested without spending the one
submission, so verification machinery *is* the game. On ARC-AGI-3 **testing is
acting**: one action settles a question for the price of one action. Verification
becomes cheap and **planning becomes expensive** — the hard question stops being
"is my rule correct" and becomes "given rules I already believe, what is the
shortest route".

§5 stays: it is cheap, correct, and the three-valued distinction is still the
right one when a rule *is* worth checking. But it is no longer the centre of
gravity, and presenting it as such sent the solver's attention to the wrong
place.

The general lesson is cheap to apply and was available all along: **check the
run for uses of your own API**. An abstraction with zero uses is a design error,
not a solver error — the same shape as ARC-AGI-2's finding that the iteration
budget was inert because `check()` was free.

> **Do not do it by grepping the stream, which is what this section originally
> advised.** The stream carries tool *results*, so a single `Read DOCTRINE.md`
> echoes every example in the doctrine back into it — `verify(` greps as 65 hits
> across five runs and is really zero. Parse `tool_use` blocks and read only
> their inputs. §9.8a has the method and the numbers at n=5, including the part
> this section could not see from one run: the planner built in response to the
> hand-rolled BFS below was never adopted either.

### 9.8 Only a real agent finds the interface bugs

Every offline test passed while the harness was unusable, three separate times
and for three unrelated reasons. The tests were not wrong; they were written by
the same author as the code and encoded the same assumption about how it would
be called.

Two of the three bugs were found within minutes of a live solver touching it,
and the third by driving a real game with a scripted policy. **Run the thing
before trusting it**, and treat the first real run as an experiment about the
harness rather than about the model.

### 9.8a What solvers actually called — the measurement, done properly

§9.8 says run the thing before trusting it. §9.7 says measure which of your
abstractions the solver used. Here is that measurement over five runs, and it is
the most uncomfortable result in this document.

**Method matters, because the obvious method is wrong.** Grepping the stream for
`verify(` reports 65 hits across all five runs and means nothing: the stream
contains tool *results*, so every `Read DOCTRINE.md` echoes the doctrine's own
examples back into it. Parsing `tool_use` blocks and reading only the *inputs*
gives what the solver actually executed. The two answers are not close.

Across **539 tool calls** in seven runs, every workspace advertising all of these
in both `CLAUDE.md` and `DOCTRINE.md`:

| called | times |
|---|---|
| `client.status()` | 157 |
| `arc.render()` | 106 |
| `arc.diff()` | 63 |
| `arc.png()` | 35 |
| `arc.objects()` | 8 |
| `arc.effective_actions()` | 2 |
| `client.pace()` | 1 |

| never called, in any run | |
|---|---|
| `Rule`, `verify()`, `survey()`, `regressions()` | the entire three-valued rule engine |
| `predict()` | the forward model |
| `shortest_path()`, `reachable()` | the planner |
| `monotone_rows()`, `logical()`, `collapse()`, `block_size()` | |

**Six of those seven runs won.** §5 — the longest and most-argued section of this
design note, the three-valued predicate core the `rules` module opens by calling
"the single most important constraint in this module" — has never been exercised
by a solver. Not once, in seven runs, did any of them construct a `Rule`.

The two newest entries are the ones to watch rather than to celebrate:
`effective_actions()` and `pace()` were added *because* of this finding, and
between them have three calls. `pace()` was used on its first outing, which is
weak support for the placement argument below — it is a `client.` method, and
`client.` is where solvers look — but three calls is three calls.

Three claims elsewhere in this project are corrected by it:

- `planning.py` was built because "a solver hand-rolled BFS twice". It did — and
  it kept hand-rolling it. Supplying the tool did not cause adoption.
- `monotone_rows()` is described as having found `ls20`'s energy bar unaided.
  That was a **scripted probe of mine**, not a solver. No solver has called it.
- `predict()` was described as something "solvers reach for unaided". They wrote
  their own step functions; they did not call this.

What the used five have in common is that they answer *what do I see right now*
— status, diff, render, png, objects. What the unused ones have in common is
that they answer *what do I believe, and does it hold up*. A solver working a
live game apparently does the second in its head and only outsources the first.

**The lesson for the harness is not "delete the rule engine" at n=5.** It is
about placement: `status()` is called 82 times and a new `arc.*` function has
been called zero times, so anything the harness genuinely needs a solver to see
belongs *in a call it already makes*. That is why the pace ratio and the
no-effect warning went into `status()` rather than shipping as the functions
they started as, and why `pace()` is a client method.

### 9.9 A refusal is only as durable as the state that arms it

§9.1 and §9.3 have an intersection, and it is the worst place in the design to
be wrong. Refusals beat instructions — but a refusal has *state*, and in a
harness whose caller is a sequence of processes that state has to survive to
disk like everything else.

The RESET-after-advance refusal is armed by a single boolean recording that the
last action completed a level. It was a plain in-memory field. Across the
process boundary that §9.1 says is the defining fact of this architecture, it
defaulted to `False`, and the guard protecting the one unrecoverable mistake in
the game stood down. A resumed run opened with RESET, discarded a won game, and
replayed all seven levels.

The failure is strictly worse than having no refusal at all, for a reason worth
stating plainly: **a guard that is present but unarmed removes the vigilance
that its absence would have preserved.** The prompt for that run did say "Do not
RESET to 'start clean'; that discards real progress" — advice which by §9.3 was
never going to be sufficient, and which nothing was left to back up.

Two rules follow, both mechanical enough to check:

- **Every field that arms a refusal belongs in the persisted state, and there
  should be a test that a fresh process still refuses.** Not a test that the
  refusal fires — that one passed throughout.
- **Do not let a server flag be the only witness to an irreversible event.** The
  same run received `full_reset: false` on the transition that took it from
  level 6 to level 0. Derive the fact from state you own — the level went down —
  and treat the flag as corroboration. Two of this harness's counters, and the
  `board_replaced` property documented as the check spatial rules want, were all
  reading the flag and all reported nothing.

---

## Appendix: what has actually been run

- **Local loop, no key.** `scratchpad/arc3/toy_game.py` and
  `athanor.ccarc3.bench.lineage` drive `arcengine` games standalone. The bench
  game is built so rules and mechanics come apart across levels, which is what
  makes §5's central claim falsifiable rather than merely stated.
- **Live API.** `ls20-9607627b` driven with scripted policies, and a full solver
  workspace generated and run against it. Everything marked [LIVE] came from
  these, including the three corrections the design needed.
- **Tests.** 473, covering the grid primitives, the ledger, the three-valued
  rule core, the gate's refusals, the client's refusals, cross-process
  resumption, and workspace/outcome construction.
