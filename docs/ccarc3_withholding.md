# How CCARC3 withholds the human baselines, and what each layer is for

**This file exists because its contents cannot live in `client.py`.** The solver
imports that module, and `inspect.getsource(ArcClient)` returns the whole class
body — every `#` comment included. `docs/` is not on the solver's `PYTHONPATH`,
so this is the one place the reasoning can be written down in full.

That took two attempts to get right, and the second failure is the more
instructive one.

- **First**, the rationale lived in **docstrings**. `help(client)`,
  `pydoc.render_doc(ArcClient)` and `arc.gate.__doc__` all render docstrings, so
  an orienting solver got a complete briefing: `help(arc.GameInfo)` printed *"real
  games span 171 to 1843 baseline actions"*, `baseline_here` announced that a
  median exists and named the private attribute holding it, and `arc_proxy`'s
  module docstring described the entire defence. Fixed in `f66cc91` by moving it
  all into `#` comments.
- **Second**, the rationale lived in **comments in the same file**, on the
  argument that `help()` cannot see comments. True — and irrelevant.
  `inspect.getsource` on a *class* spans the whole block, comments and all. It is
  as standard a call as `help()`, and on 2026-08-07 a solver ran
  `dir(client)` followed by `inspect.getsource(client.restart_for_replay)` on its
  first orientation turn, one command short of the class.

  The near-miss is why this was easy to get wrong: `getsource` on a single
  *method* starts at its `def` and excludes preceding comments, so spot-checking
  a property returned clean output and looked like proof.

## The layers, in order of how much they actually carry

1. **The strip** (`tools/ablate_baselines.py`). Rewrites the workspace so the
   numbers are not in the solver's process at all: `baseline_actions=()` in
   `session.py`, the array and `action_budget` gone from `meta.json`, §6 removed
   from the doctrine, and `CLAUDE.md`'s baseline paragraph dropped whole. **This
   is the load-bearing layer.** Everything below it is defence in depth. With the
   array empty, `info.baseline_for(level)` returns `None`, so even
   `client._baseline_here_enforced` — the private attribute a determined solver
   would reach for — yields nothing.
2. **The proxy** (`src/athanor/ccarc3/arc_proxy.py`). Holds the ARC key and
   forwards four endpoints, refusing `/api/games`. Closes the deliberate route: a
   solver with a key can otherwise fetch every environment's baselines in nine
   lines of `urllib`, and two runs did exactly that before this existed.
3. **`CCARC3_HIDE_BASELINES`**. Closes the in-process route — `baselines_for`
   raises, `list_games` filters. Added after `cd82` ran `dir(arc)` on its first
   orientation turn and read `baselines_for` straight out of the listing, having
   gone looking for nothing at all.
4. **`quiet_pace`** (renamed from `hide_baselines`, 2026-08-07). Silences the
   pace ratio, `pace()` and the `raw`/ceiling half of the score block. Renamed
   because the workspace `session.py` passes the keyword literally and the
   doctrine tells the solver to import that file — so the old name announced in
   one word that per-level medians exist and are being kept from it.
5. **`redact_self_reference`**. Removes doctrine passages naming the game being
   played, so no solver reads about its own prior attempts. Only rewrites
   `DOCTRINE.md`; anything else that names environments has to be handled where
   it lives.

## What is still reachable, stated plainly

`dir(client)` lists `baseline_here`, `level_budget`, `max_actions` and `pace`.
Those names disclose that a baseline concept exists. They are not renamed because
`baseline_here` is the honest name for what the property returns when a run *is*
allowed to see it, and a euphemism in the API would be a worse cost than the
signpost. The numbers behind them are empty under the strip, which is layer 1's
job and is verified per-run by `tools/proofread_trace.py`.

**The distinction that matters, and the standard to hold new code to:** a *map* is
whether information is being withheld, by what, and where it lives; the *treasure*
is the numbers. Both matter, but only the treasure changes a score. Every finding
above was a map leak. None of them ever handed over a number, because layer 1 held
throughout.
