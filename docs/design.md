# Design

System design rationale, architecture, and competitive context.

## Contents

1. [Thesis](#thesis)
2. [Three Mechanisms](#three-mechanisms)
3. [Architecture](#architecture)
4. [Competitive Context](#competitive-context)
5. [Terminology](#terminology)
6. [Threat Model](#threat-model)

---

## Thesis

Public ARC-AGI-2 systems already demonstrate that frontier models can search over executable Python solutions. Confluence, Imbue, and Squeeze-Evolve all achieve high accuracy through various forms of parallel search, evolutionary mutation, or recombination.

This work takes a different path. The contribution is an **organizing principle**: three interlocking mechanisms — code-verified hypothesis refinement, an independent artifact-only reviewer, and inter-agent artifact exchange — that together achieve **95.7% on ARC-AGI-2 public eval at $3.12/task**, the lowest cost per task among systems scoring above 95%.

## Three Mechanisms

### 1. Code as Verification

Exploratory Python execution (`run_code`, `run_code_in_previous_runtime`) is a first-class reasoning tool, separated from final solution submission. The solver tests sub-rules, inspects intermediate structure, and verifies invariants before committing to a `solve()` implementation.

The angle is not "LLMs can write Python" — Confluence and Imbue already demonstrate that. It is that **code execution compresses verification into fewer tokens**. A short Python snippet that executes and returns a concrete result replaces lengthy chain-of-thought enumeration. That token efficiency compounds across iterations into measurably lower cost.

Squeeze-Evolve (97.5% at $5.93/task) proves execution is not *necessary* for high ARC accuracy. The ~2× cost gap relative to this work is the efficiency price of verifying hypotheses in pure reasoning.

### 2. Independent Artifact-Only Reviewer

A separate model context (the independent reflector) reviews the solver's output with no access to the solver's reasoning chain. It sees: hypothesis text, `solve()` code, candidate test predictions, and training accuracy.

It issues a three-way verdict:

- **APPROVE** — hypothesis explains all training data, code implements it truthfully, rule appears to generalize.
- **REJECT** — rule is wrong, overfit, or unsupported. The rejection atomically triggers context compression + feedback injection, forcing the solver to distill what it learned and start fresh with the critique.
- **EXPAND_CANDIDATES** — rule is sound but ambiguity remains between plausible branches; the solver is asked to emit a second candidate for test outputs.

This is the only mechanism among top ARC systems that can reject a solution that passes training 100%. Confluence uses post-hoc voting; Imbue uses fitness with corroboration; Squeeze-Evolve uses intrinsic confidence routing — none has an artifact-level independent review.

The reflector can maintain a multi-turn review conversation (early turns with full puzzle data, later turns with only the changes since last review) to avoid false-confidence anchoring.

### 3. Inter-Agent Artifact Exchange (IAAE)

Active, threshold-triggered compression paired with artifact-based cross-referencing between the solver's and reflector's contexts. Three properties matter:

**Active compression.** Compression is agent-initiated when context pressure is measured, not time-based truncation. The solver writes a memory checkpoint with a specific schema:

- current best hypothesis
- results from exploratory experiments
- verified invariants (e.g., "output is always 20×20", "color mapping preserves adjacency")
- last submitted solver code
- failure analysis and reflection
- concise records of prior rejected attempts
- reflector critique (when IAAE was triggered by REJECT)

**Artifact-based cross-referencing.** Solver and reflector each maintain their own contexts and exchange only *artifacts* — hypothesis text, `solve()` code, candidate outputs, reflector verdict. Neither agent sees the other's reasoning chain. The reflector's multi-turn context anchors on solver artifacts across rounds; the solver's compression includes the reflector's critique after REJECT.

**Atomic rejection+compression.** When the reflector issues REJECT, feedback injection and context compression happen in a single solver turn. This prevents the solver from retrying in stale context.

The next context window resumes from this exchange of artifacts plus the original puzzle data, not from raw transcript. This is more specific than generic context summarization, and structurally distinct from population-based memory (Imbue, Squeeze-Evolve) or agent-local state (Confluence).

## Architecture

### Three Solver Artifacts

The solver loop tracks three distinct artifacts:

1. **Exploration code** (`run_code`, `run_code_in_previous_runtime`)
   - Temporary experiments used to test hypotheses and verify invariants.
   - Not the final submission artifact.
   - Unlimited calls; does not count toward iteration budget.

2. **Transform hypothesis** (`submit_transform_hypothesis`)
   - Natural-language externalization of the current rule.
   - Explicitly separated from final code submission.
   - Reused by the generalization gate and the independent reflector.

3. **Transform code** (`execute_python_solution`)
   - Candidate final `solve(grid)` implementation.
   - Evaluated against training pairs and used for test prediction generation.
   - Each call counts as one iteration.

### Pipeline

```text
Web UI
  |
  v
Orchestrator (solver/orchestrator.py)
  |- tool: run_code (exploration experiments)
  |- tool: submit_transform_hypothesis
  |- tool: execute_python_solution (final transform validation)
  |- reflection + inter-agent artifact exchange
  |
  v
Iteration state + distilled research state
  |
  +--> Independent Reflector
  |      |- APPROVE -> accept
  |      |- REJECT -> IAAE + feedback + retry
  |      \- EXPAND_CANDIDATES -> emit second candidate
  |
  v
Checkpoint (full conversation, config, tool calls, reflector verdicts)
```

### Iteration Flow

1. Build prompt from train/test inputs (text + rendered PNG grids).
2. Agent proposes hypothesis and runs exploratory `run_code` experiments to test sub-rules and inspect structure.
3. Agent externalizes the current rule via `submit_transform_hypothesis`.
4. Agent submits final transform candidate via `execute_python_solution`.
5. System runs the candidate against training pairs.
6. If training fails: self-reflection prompt + next iteration (IAAE may fire if context is large).
7. If training passes 100%: run test-generalization self-audit, then independent reflector.
8. On APPROVE: accept. On REJECT: IAAE + feedback, retry. On EXPAND_CANDIDATES: emit second candidate.

### Auditing Surface

- Event-level traces and tool-call records are saved in each run's checkpoint JSON under `state.history`.
- Release checkpoints are in `release_runs/` (read-only reference set) and `src/athanor/web_demo/saved_runs/` (user-writable). The web demo loads both.
- Each checkpoint embeds the exact config used (`state.config`), so loading one restores the run's settings.

## Competitive Context

### Headline Comparison

| System | Public eval | Cost / task | Execution | Quality gate | State continuity |
|---|---:|---:|---|---|---|
| [Confluence](https://github.com/confluence-labs/arc-agi-2) | 97.9% | $11.77 | Program synthesis | Vote aggregation | Agent-local |
| [Squeeze-Evolve](https://arxiv.org/abs/2604.07725) | 97.5% | $5.93 | None | Confidence routing | Population |
| [Darwinian Evolver (Imbue, Gemini 3.1 Pro)](https://github.com/imbue-ai/darwinian_evolver) | 95.1% | $8.71 | Program scoring | Fitness + corroboration | Evolutionary |
| **This work** | **95.7%** | **$3.12** | **Verification tool** | **Independent reflector** | **Inter-Agent Artifact Exchange** |
| [Darwinian Evolver (Imbue, Gemini 3 Flash)](https://github.com/imbue-ai/darwinian_evolver) | 61.4% | $2.42 | Program scoring | Fitness + corroboration | Evolutionary |

Public eval scores are not directly comparable to semi-private or private eval scores. None of the four systems above has a published verified semi-private score. The highest verified semi-private score is Poetiq at 54%.

### Why Cost Per Task Matters

The four systems scoring above 95% on public eval span a nearly 4× cost range ($3.12 to $11.77). That cost gap is the systems-level differentiator.

The mechanism is token efficiency of verification:

- **Pure LLM reasoning** (Squeeze-Evolve): checking "does every row contain exactly one non-zero cell" in token space requires careful enumeration — hundreds of tokens, occasional arithmetic errors, and confidence built by evolutionary consensus over many candidates.
- **Code execution as a verification tool** (this repo): the same check is a short Python snippet that executes and returns a deterministic boolean. The solver can then commit to or discard the sub-hypothesis with no further reasoning tokens spent.

That per-check efficiency compounds across iterations into measurably lower total cost. Squeeze-Evolve proves that verification in pure reasoning *works* (97.5% accuracy). The measured 2× cost gap is the price of doing verification without code.

### What Each Top System Does Well

- **Confluence** — parallel agent synthesis with vote-based aggregation scales to very high accuracy (97.9%). Cost is the trade-off.
- **Squeeze-Evolve** — pure LLM evolutionary recombination, properly routed across model tiers, can reach 97.5% without code execution. Forces a sharper framing of why code execution helps (efficiency, not capability).
- **Imbue** — evolutionary search over candidate programs with mutation and corroboration-based stopping is model-agnostic and works across cost tiers (61% with Flash at $2.42, 95% with Pro at $8.71).
- **This work** — a single-thread agentic discovery loop with independent review and inter-agent artifact exchange matches 95%+ accuracy at substantially lower cost, and solves hard pairs that baseline configs never reach.

## Terminology

**Exploration Code.** Short-lived code executed via `run_code` for analysis, diagnostics, or hypothesis testing. Unlimited calls; does not count toward iteration budget.

**Code as Verification.** Use of executable experiments as a token-efficient verification tool for testing sub-rules, inspecting intermediate structure, and refining hypotheses before committing to final solver code. A short Python snippet that executes and returns a concrete result replaces lengthy chain-of-thought enumeration — the primary cost-efficiency mechanism of the system.

**Transform Hypothesis.** Natural-language externalization of the current rule, submitted via `submit_transform_hypothesis`. Explicitly separated from the code submission so the hypothesis can be reviewed independently of its implementation.

**Transform Code.** The candidate final `solve(grid)` program submitted via `execute_python_solution`.

**Execution as Scoring.** Use of execution primarily to validate, score, rank, or aggregate candidate final programs rather than as a separate exploratory reasoning workspace. This is the default pattern in Confluence and Imbue; it is distinct from "code as verification" in this system.

**Independent Reflector.** A separate model context that reviews the solver's deliverables with no access to the solver's reasoning chain. Sees the hypothesis text, `solve()` code, candidate test predictions, and training accuracy. Issues APPROVE / REJECT / EXPAND_CANDIDATES.

**Inter-Agent Artifact Exchange (IAAE).** Active, threshold-triggered compression paired with artifact-based cross-referencing between the solver's and reflector's contexts. Three distinguishing properties: active (agent-initiated), artifact-based (no reasoning chain crosses the boundary), and atomic on REJECT (feedback + compression in one turn). Fires on context-size threshold or reflector REJECT.

**Distilled Research State.** The compact state bundle produced by IAAE. Not just a dialogue summary; it is the subset of information needed to continue the search from a fresh context window.

**Bounded-Context Solving.** An orchestration setting in which the solver cannot rely on carrying full raw interaction history forever and therefore needs an explicit continuity mechanism. IAAE is this system's answer.

**Quality Gate.** The broader acceptance layer applied after a train-perfect candidate is found, including the test-generalization self-audit and the independent reflector's verdict.

**Overfit Prevention.** Mechanisms that reduce train-only brittle solutions:
- test-generalization self-audit after training passes 100%
- independent reflector reject/approve stage

## Threat Model

### Why arbitrary execution exists

The research claim requires executable hypothesis testing. `run_code` and `execute_python_solution` are central to the method.

### Risk

When enabled, model-generated Python executes locally and may:
- read local files
- use network resources
- consume CPU/memory unexpectedly
- exfiltrate sensitive local data

### Boundary

- The web UI enables code execution by default (`unsafe_local_exec = true`). When disabled, `run_code` and `execute_python_solution` are blocked.
- When enabled, model-generated code runs in the local Python process with no sandbox.

### Recommended operation

1. Use an isolated container / VM.
2. Mount only minimal task/output directories.
3. Run under a non-privileged account.
4. Avoid secrets in environment and filesystem.
5. Monitor and cap resource usage at the environment level.

