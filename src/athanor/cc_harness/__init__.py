"""Claude Code as harness — a single-agent variant of the Athanor solver.

The flagship Athanor system owns its agent loop: `solver/orchestrator.py` drives
turns, executes tools, injects reflection prompts, compresses context, and
consults an independent reflector.

This variant hands the agent loop to Claude Code and keeps only the parts of
Athanor that encode its research thesis:

* **Code as verification.** Every claim about the puzzle must be established by
  executing something, not by prose enumeration. The workspace toolkit
  (`arc.verify`) makes that the literal API for asserting a fact.
* **Artifact separation.** A natural-language hypothesis and a `solve()`
  implementation are separate, auditable artifacts, and the hypothesis must be
  written before the code it explains.
* **Budgeted formal iterations.** Exploration is unlimited and free; formal
  submission through the gate is counted and recorded.

There is no reviewer agent in this variant — the quality gate reduces to
`train 100%` plus a mandatory generalization self-audit.

See `docs/cc_harness.md` for the full mapping.
"""

from .config import CCRunConfig

__all__ = ["CCRunConfig"]
