# CCARC3 Codex port

**Current continuation handoff:**
[`docs/ccarc3_codex_handoff_20260812.md`](ccarc3_codex_handoff_20260812.md).
It records the verified branch state, setup commands, strict live-run result,
evidence archive, and the clean-run safety boundary.

This branch runs the ARC-AGI-3 harness with `codex exec`; it is not an
ARC-AGI-2 port.

## Port boundary

The five domain modules — `grids.py`, `ledger.py`, `rules.py`, `scoring.py`, and
the HTTP behaviour in `client.py` — are unchanged. Their ledger and RHAE
semantics remain the source of truth. None of those five files changed.

`session.py` is runtime-specific and was rewritten at its CLI seam:

* workspaces contain `AGENTS.md`, which Codex discovers from its working
  directory;
* launches use `codex exec --json` with an explicit model, reasoning effort,
  and workspace sandbox;
* the Codex-generated thread id is read from `thread.started` and is used by
  bounded give-up nudges through `codex exec resume`;
* `turn.completed.usage` supplies token accounting. Codex JSONL does not expose
  Claude Code's dollar-cost or duration fields, so the port records no invented
  substitutes;
* the ARC credential is still removed from the child whenever the allowlisting
  proxy is active. The proxy, ledger-derived outcome, scorecard corroboration,
  clean-rollout rejection, shared-card attempt scoping, and action budget policy
  remain in force.

Operational scripts now default to this branch, a Codex-specific temporary run
root, `codex` process matching, and the `AGENTS.md` evidence surface.

## Scoring invariants

All eight invariants in `ccarc3_port_guide.md` §3 are runtime-independent and
are honoured without fallback. In particular, the port did not rewrite the
scoring or ledger modules: best-play selection, cumulative action differencing,
from-level attribution, RESET treatment, replay separation, zero-cost caps,
sequential completion, and the completion ceiling retain the audited code.

## Guide audit

No §3 claim was contradicted by the Codex runtime work. One runtime assumption
does not transfer: a session id cannot be assigned before launch. Codex emits
the authoritative thread id after launch, so the port captures that event
before attempting a nudge. This changes session mechanics, not any scoring
invariant.
