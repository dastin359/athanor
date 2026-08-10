# Porting CCARC3 to another agent harness

**Audience: an agent asked to read this repository and rebuild its ARC-AGI-3
logic on a different harness.** This is the document to start from. It says what
is portable, what is Claude-Code-specific and must be rewritten, and — the part
worth the most — which invariants were expensive to discover and will silently
produce wrong numbers if you get them backwards.

Everything here is measured unless it says otherwise. Where something is
assumed, it says so.

---

## 0. What this is

CCARC3 runs an agent as the *player* of an ARC-AGI-3 environment. ARC-AGI-3 is
not the grid-transduction task: it is a set of small interactive games, played
over an HTTP API, one action at a time, where the agent must infer the mechanics
from what the board does. Scoring is RHAE — a per-level, human-relative
efficiency measure described in §3.

The harness here does four things:

1. **Plays the game** — an HTTP client that sends actions and records frames.
2. **Writes a ledger** — an append-only record of `(before, action) -> after`,
   which every downstream number is derived from.
3. **Gives the agent tools to think with** — grid primitives, three-valued rule
   checking, a forward-model checker.
4. **Decides whether a run may be banked** — a clean-rollout discipline with
   guards that discard runs that are not results.

The fourth is the part most likely to be underestimated. Most of the defects
found here were in code that decides whether a number is trustworthy, not in
code that computes it.

---

## 1. What ports, and what does not

### Portable — pure ARC-AGI-3 domain logic, no harness dependency

| module | what it is | notes |
|---|---|---|
| `src/athanor/ccarc3/grids.py` | grid representation, diffing, objects, cell-grid recovery | imports only numpy; no SDK, no API key |
| `src/athanor/ccarc3/ledger.py` | the transition ledger: write and read back | plain dicts; testable without the SDK |
| `src/athanor/ccarc3/rules.py` | three-valued rule checking, forward-model checking | pure functions over transitions |
| `src/athanor/ccarc3/scoring.py` | RHAE, per-level accounting, card corroboration | the arithmetic ARC actually uses |
| `src/athanor/ccarc3/client.py` | the HTTP client, action accounting, state persistence | stdlib `urllib`; the only network code |

These five are the port. They contain no reference to Claude Code, subprocesses,
or prompts. If you take nothing else, take `scoring.py` and `ledger.py` — they
encode the two things that are easiest to get wrong and hardest to notice.

### Harness-specific — rewrite for your runtime

| module | what it does | why it does not port |
|---|---|---|
| `src/athanor/ccarc3/session.py` | builds a per-game workspace, launches the agent as a subprocess, collects the outcome | Claude-Code CLI flags, stream-json parsing, session-resume semantics |
| `tools/clean_rollouts.py` | the driver: which game next, bank or discard, retries | orchestration policy, not domain logic — but read §5, the *policy* ports even if the code does not |
| `src/athanor/ccarc3/arc_proxy.py` | an allowlisting HTTP proxy that gates what the player may reach | your sandbox will differ |
| `tools/*.sh` | supervisor, heartbeat, evidence preservation, daemon watchdog | operational scaffolding for a long-running box |

### Assets

`src/athanor/ccarc3/assets/CCARC3_DOCTRINE.md` is the standing instruction given
to the player. It is worth reading in full before porting: much of what looks
like harness behaviour is actually taught there, and a port that copies the code
but not the doctrine will behave differently for reasons that are hard to trace.

---

## 2. The API's shape, and four things about it that surprise people

**Frames.** `FrameData.frame` is a **list of grids**, not one grid. The engine
renders every intermediate frame until the action completes, and the mechanics
are often visible only in the intermediate ones. Code that takes `frame[0]` or
`frame[-1]` silently discards them. `ledger.load` keeps all of them in
`Transition.intermediate` and uses the **last** as the resting state.

**Grids** are 64×64 over a sixteen-colour palette — not ARC-AGI-2's ten. Rendered
as bracketed integer rows a single grid costs roughly 6k tokens; one character
per cell over `0-9a-f` is about a third of that, measured.

**The score field has two names.** `arc_agi_3` 0.0.1 sends `score`; `arcengine`
0.9.3 renamed it `levels_completed`. Read whichever is present *by value, not by
key presence* — a frame carrying `score: None` beside a real `levels_completed`
must not record zero. The score is a count of completed levels, and level
boundaries are inferred from its increments, so a zeroed score collapses an
entire trace into level 0.

**A refused action still costs budget.** When the state is `GAME_OVER` or `WIN`,
the server short-circuits any non-RESET action and returns `frame: []` without
stepping the game. The action is still billed. These are recorded as `wasted`
rather than dropped, because "actions burned after a death, before the player
noticed it had to RESET" is a number worth being able to read off a trace.

---

## 3. RHAE scoring — the invariants, in the order they bite

This is the section to read twice. Every item was established against the live
API or a real trace, and each one produces a plausible-looking wrong number if
inverted.

**3.1 An environment is scored on its BEST play, not its last.** A scorecard
holds one row per playthrough, and the environment takes the maximum. Verified
with a deliberate probe: play 1 cleared a level cleanly, play 2 fumbled the same
level, and the environment score came back as the higher of the two. The two
conventions agree whenever a run's last play is also its best, which was true of
every game scored here before it was checked — so this is exactly the kind of
thing that passes unnoticed until it doesn't.

**3.2 `actions_by_level` is CUMULATIVE.** The field name says "by level" and the
content is `[level, actions spent when that level fell]`. Difference consecutive
entries to get per-level counts. Summing the raw values inflates every level
after the first and scores a run at roughly a third of what it earned. A
one-level game cannot reveal this — cumulative and per-level coincide there — so
test it on a game with at least three levels.

**3.3 Attribute an action to the level it was taken FROM.** The trace records
`level` as levels-completed *after* the action, so the action that finishes level
3 is stamped 4. Grouping by the recorded level credits every level's decisive
action to the next one, shortening each level by one and inventing a phantom
level at the end.

**3.4 The play-starting RESET is not an action; a level RESET is.** Measured
against a live scorecard: seven actions preceded by the opening RESET were
reported as 7, not 8. A RESET taken *after a death*, while the action counter is
non-zero, is a level reset and **is** counted. The exclusion is precisely the
play-starting RESET: the first transition of a trace, and any RESET that performs
a full reset.

**3.5 Replayed levels are not summed.** A RESET when the counter is zero — the
state immediately after a level advance — starts a **new play**: new guid, new
`actions` row, new `actions_by_level` row. Score each play separately and take
the best (3.1); do not sum across plays.

**3.6 A level cleared at zero cost takes the cap, and that is a deliberate
refusal.** A multi-level advance can credit a level that cost no actions of its
own. Scoring `(human/0)**2` raises; scoring it as the per-level cap is the
documented choice. Note the asymmetry that makes this worth stating: accepting a
zero from a *parsing fault* awards a defect the best possible score and the run
reads as excellent, while refusing costs a crash and a look. This repo's
`score_environment` refuses a non-positive completed-level count outright and
makes the caller decide.

**3.7 Levels are sequential.** Once one level is incomplete every later one must
be too. A completed level after an incomplete one is an error to raise, not a
shape to score.

**3.8 The completion cap and the efficiency score are different limits.** The
per-environment score is `min(weighted-mean-of-level-scores, weighted-fraction-
of-levels-completed)`. The second term is pure structure — which levels fell, not
how fast — so it is computable without knowing any human baseline. That matters
if you withhold baselines from the player (§6): the cap is the one number they
can still be told honestly.

---

## 4. The ledger, and why it is the source of truth

Every downstream number — scores, pace, waste, level boundaries — is derived from
`trace.jsonl` and nothing re-reads the server. So a field the writer drops is a
field nobody can recover.

- **Chain `before` from the previous transition's LAST frame**, not its first.
  They agree until an action renders more than one frame, and then every
  subsequent `before` is an intermediate state that was never a resting state.
- **A full reset moves the level DOWN.** The API has been observed reporting
  `full_reset: False` on a transition that took a game from level 6 to level 0.
  A check that looks only for a level *increase* misses the largest board
  replacement in the trace. Treat "the recorded level went down" as a full reset
  regardless of the flag, and give rules a `board_replaced` predicate that is
  `crosses_level or full_reset` — that, not `crosses_level`, is what a spatial
  rule wants.
- **Only the final line of a live trace may be forgiven.** A report is most
  useful *during* a run, which is exactly when the last line may be mid-write. A
  malformed line anywhere earlier is corruption: dropping it silently shortens a
  level's action count and changes a score.

---

## 5. Deciding whether a run counts

The policy ports even though the code does not.

**A run that gave up is not a result.** An agent that stops with most of its
action budget unspent has produced an artefact of its own quitting, not a
measurement of the game. Banking it contaminates the corpus in the direction
that flatters the harness. This fired on a live run for the first time during a
validation game: the player stopped at level 6 with 80% of its budget unspent,
the guard refused to bank it, and the re-run cleared level 6 and reached 7 — so
the first stop was a genuine give-up and not a capability wall. Without the
guard, "level 6 is hard" would have entered the record as a fact.

**Corroborate against the server's own card.** Our ledger is not independent
evidence about our ledger. A proxy fault once let a run finish 8/8 in 124 actions
against a card frozen at level 3, with no other symptom, because actions carry a
guid and are not card-scoped: the game plays perfectly and the numbers you score
from stop moving. Compare levels, not action counts — the two ledgers have always
differed by a few for documented reasons.

**Under a shared scorecard, scope the comparison to this attempt.** The server
appends every attempt of a game to the same card entry, so a plain `max()` over
its rows carries the high-water mark of attempts you threw away. Snapshot how
many rows the card held when the attempt opened, and compare only the rows after
that boundary. Otherwise a discarded attempt corroborates a frozen one.

**A scorecard is state on one backend instance.** Session cookies are the only
route back to it. Losing them loses the card, and a card is an artifact a
submission points at — opening one is not free bookkeeping.

---

## 6. Withholding, if you want the measurement to mean anything

The published human baselines are available from the API. If the player can read
them, its behaviour is conditioned on them and the run stops being a measurement
of unaided play.

The mechanism here is a two-level split: an *enforced* value the harness uses for
budgets and limits, and a *visible* value the player can read, which is `None`
when baselines are withheld. **Every limit must read the enforced value.** Reading
the visible one means hiding a number silently switches the limit off — which is
the failure mode, not a hypothetical: that exact swap passed the entire test suite
here until a mutant flushed it out.

A subtler leak class is worth knowing before you write any prose: a *derived*
number leaks as surely as a raw one. A score beside an action count inverts to a
baseline; a ratio beside a count does the same. Eight such leaks were found in
this repo's own comments, several of them added *by* the commit that fixed a
previous leak.

---

## 7. The method, which matters more than any single fix

**The recurring defect class here is a check that names a thing and reads a proxy
for it. The signature is that it passes by not running.** Reading a test does not
reveal this. Mutating the code the test claims to protect does.

`tools/mutation_check.py` applies a mutant, runs the tests, restores, and reports
which mutants survive. `tools/mutation_battery_ccarc3.py` holds every mutant the
five domain modules were audited with — **147 across five modules** — and exits
non-zero on any survivor it is not expecting.

    .venv/bin/python tools/mutation_battery_ccarc3.py            # all
    .venv/bin/python tools/mutation_battery_ccarc3.py scoring    # one module

Three things that battery taught, which generalise past this repo:

1. **A survivor is either a gap or an equivalence, and assuming the second is how
   a suite acquires tests that assert coincidences.** Six equivalences are
   recorded in `docs/ccarc3_open_findings.md` with an argument each. Two of them
   are properties of the *environment* rather than the code, and are marked as
   such, because they hold only while something else does.

2. **Run the battery against new tests too.** Three tests written here to kill a
   specific mutant passed against it. The instructive one: a default threshold of
   0.9 with a nine-of-ten fixture sits exactly on the boundary, so the test agreed
   with the mutant it was written to refute.

3. **A guard has two directions.** A single-instance guard was tested only for
   "a second instance refuses" and never for "the first one starts" — so a state
   where *nothing* could ever start satisfied the whole file, while the status
   command cheerfully reported everything healthy.

If you port only one habit, port this one.

---

## 8. What is NOT established

Stated plainly, because a port that inherits these as facts will build on sand.

- **The rule-checking layer has never been used by a player.** Across seven runs
  and 539 commands, `rules.py` was called zero times; players wrote their own
  models inline. The abstraction may be right and it is not validated by use.
  `client.status()` — one string, called 82 times in a single run — is the
  surface that actually got used. Offering an abstraction is not the same as it
  being adopted.
- **Whether real games render on a uniform block grid is an open question.** The
  block-size recovery is exact — it never claims a factor that does not hold —
  but it has only been checked against locally authored games. On real frames a
  board scaled into the 64×64 viewport usually has *no* integer factor, so the
  function correctly declines and the useful tool is the multi-frame cell-boundary
  recovery instead.
- **Score-increment level inference is a heuristic.** It was verified against one
  engine version where `next_level()` is the only thing touching the score, and
  adds exactly 1. The live API types the field as a range, so a server-side game
  could in principle award points within a level. Prefer an explicit level from
  the caller when one is available.
- **No claim is made here about how well any agent plays these games.** This
  document is about the harness.

---

## 9. Where to look

    src/athanor/ccarc3/
      grids.py      grid primitives                        440 lines
      ledger.py     transitions, read and write            341 lines
      rules.py      three-valued checking, forward models  480 lines
      scoring.py    RHAE and card corroboration            558 lines
      client.py     HTTP client, accounting, state        1570 lines
      session.py    workspace + subprocess (harness-specific)
      assets/CCARC3_DOCTRINE.md   what the player is told

    tools/
      mutation_check.py             the mutation runner
      mutation_battery_ccarc3.py    147 mutants, re-runnable
      clean_rollouts.py             the driver (policy in §5)

    docs/
      ccarc3_design.md          design rationale, open questions
      ccarc3_open_findings.md   every audit finding, with the argument
      ccarc3_withholding.md     why baselines are withheld and how
      ccarc3_port_guide.md      this file

Run the suite with `.venv/bin/pytest -q`. It is green at the time of writing and
the count is stated in `docs/ccarc3_memory.md`; a module's tests are the fastest
way to learn what its edge cases actually are, because most of them were written
in response to a specific defect and say so.

**One request, if you port this.** The docstrings and comments carry the evidence
— what was measured, on which run, and what was believed beforehand and turned
out wrong. That is the expensive part. Code without it will look cleaner and will
lose the reasons.
