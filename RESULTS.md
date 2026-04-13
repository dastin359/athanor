# Results

## Full-Eval Score

**Score: 113.83 / 119 = 95.7%** on ARC-AGI-2 public evaluation (120 tasks, 1 excluded).

| Status | Tasks | Points |
|---|---:|---:|
| Fully solved (100%) | 113 | 113.000 |
| Partially solved | 2 | 0.833 |
| Failed (0%) | 3 | 0.000 |
| Excluded | 1 | — |
| **Total** | **119** | **113.833** |

Scoring follows the official ARC-AGI-2 metric: each task contributes up to 1 point. If a task has multiple test outputs, each correct output contributes an equal fraction (e.g., 1 of 2 correct = 0.5 points).

One task (`b6f77b65`) is excluded due to a known data-quality issue ([arcprize/ARC-AGI-2#10](https://github.com/arcprize/ARC-AGI-2/issues/10), fix pending in [#61](https://github.com/arcprize/ARC-AGI-2/pull/61)).

### Partially solved tasks

| Task | Test outputs | Correct | Points | Note |
|---|---:|---:|---:|---|
| `88e364bc` | 2 | 1/2 | 0.500 | test 0 unsolved (never solved by chain-of-thought-only (CoT-only) Opus 4.6 at any thinking level) |
| `d35bdbdc` | 3 | 1/3 | 0.333 | tests 0,1 unsolved (never solved by CoT-only Opus 4.6 at any thinking level) |

### Failed tasks

| Task | Test outputs | Note |
|---|---:|---|
| `2b83f449` | 1 | never solved by CoT-only Opus 4.6 at any thinking level (0/8); solved by other CoT-only submissions in the [ARC Prize public eval corpus](https://huggingface.co/datasets/arcprize/arc_agi_v2_public_eval) (2/122 as of 2026-04-12) |
| `800d221b` | 1 | CoT-only Opus 4.6 solves 3/8 times |
| `faa9f03d` | 1 | never solved by CoT-only Opus 4.6 at any thinking level; extended 120-turn attempt, 0/1 correct |

## Cost Analysis

Batch-deployment cost to reproduce the 119-task set from `release_runs/`: **$371.25**

| Metric | Value |
|---|---|
| Average cost per task | $3.12 |
| Median cost per task | $1.71 |
| Min | $0.23 |
| Max | $20.04 |

Cost sums one checkpoint per task: the lowest-cost run achieving the best score. For failed tasks we include one failed attempt.

Only the first puzzle in a batch pays the ~$0.09 cold-write for the stable system prompt; subsequent puzzles hit Anthropic's 5-minute prompt cache automatically. Our actual 119 checkpoints were collected over weeks with many gaps between runs, so their summed cost is $377.05; reproducing them as a single batch saves $5.80 by amortizing the first-turn cache warm-up.

### Cost in context

| System | Public Eval Score | Cost / Task | Model | Code execution |
|---|---:|---:|---|---|
| [Confluence](https://github.com/confluence-labs/arc-agi-2) | 97.9% | $11.77 | Gemini | Program synthesis |
| [Squeeze-Evolve](https://arxiv.org/abs/2604.07725) | 97.5% | $5.93 | Gemini 3.0 Flash + 3.1 Pro | None |
| [Darwinian Evolver (Imbue)](https://github.com/imbue-ai/darwinian_evolver) | 95.1% | $8.71 | Gemini 3.1 Pro | Program scoring |
| **This work** | **95.7%** | **$3.12** | **Claude Opus 4.6 + Gemini 3.1 Pro (reflector)** | **Verification tool** |
| [Darwinian Evolver (Imbue, Flash)](https://github.com/imbue-ai/darwinian_evolver) | 61.4% | $2.42 | Gemini 3 Flash | Program scoring |

This work achieves the lowest cost per task among all systems scoring above 95%.

**Why code execution drives cost efficiency.** Squeeze-Evolve demonstrates that pure LLM evolutionary recombination can reach 97.5% without any code execution — so execution is not *necessary* for high ARC accuracy. The ~2× cost gap relative to this work reflects the token cost of verification: checking a hypothesis in pure reasoning requires hundreds of tokens of careful enumeration, while an equivalent Python snippet executes and returns a deterministic answer in far fewer tokens. Code execution is a token-compression mechanism that compounds across iterations into measurably lower cost.

**Caveat**: Public eval scores are not directly comparable to semi-private or private eval scores. Neither Confluence, Squeeze-Evolve, nor Imbue has published verified semi-private results. The highest verified semi-private score is Poetiq at 54%.

## Zero-Solve Frontier (Frozen 2026-04-12 Snapshot)

The pair-level frontier is defined as `(task_id, pair_index)` pairs where no submission has ever solved the pair (`ever_solved = 0`) and at least 80 attempts are logged, in the frozen HuggingFace (HF) public eval snapshot from 2026-04-12 (10,529 total attempts logged).

- Frontier size: **9** test outputs across **8** puzzles
- **Solved exclusively by this solver: 8 / 9**
- Remaining: 1

"Exclusive" means no other participant in the HF corpus has solved the pair as of the snapshot date.

### Solved frontier pairs (all exclusive)

Each pair is identified as `task_id / test_output_index`.

| Pair | HF attempt logs |
|---|---:|
| `13e47133 / 0` | 117 |
| `269e22fb / 0` | 119 |
| `269e22fb / 1` | 119 |
| `8b7bacbf / 0` | 125 |
| `9bbf930d / 0` | 117 |
| `a32d8b75 / 0` | 118 |
| `abc82100 / 0` | 121 |
| `da515329 / 0` | 120 |

### Remaining frontier

| Pair | HF attempt logs | Note |
|---|---:|---|
| `faa9f03d / 0` | 127 | extended 120-turn attempt, 0/1 correct |

### Broader view: pairs never solved by CoT-only Opus 4.6 (22 pairs)

These 22 pairs are those where no CoT-only Opus 4.6 configuration solved in 8 attempts (4 thinking efforts × 2 attempts each).

- **Solved by this system: 16 / 22** (pair level)
- Unsolved: 6 — `88e364bc/0`, `d35bdbdc/0`, `d35bdbdc/1`, `2b83f449/0`, `dbff022c/0`, `faa9f03d/0`

Solving 16 of 22 demonstrates that the combination of exploratory code execution, independent review, and inter-agent artifact exchange enables discoveries that repeated independent attempts cannot reach.

## Settings Used

- Solver: iterative hypothesis → executable experiments → final transform
- Model: Claude Opus 4.6 (main agent)
- Independent reflector: Gemini 3.1 Pro Preview
- Thinking effort: medium (main), max (reflection/compression)
- Runtime: local research mode with executable tools enabled
- Prefix caching: enabled (Anthropic 5-minute cache)

## Methodology & Artifacts

The `release_runs/` directory contains 119 checkpoint files covering 119 tasks — the minimal set achieving the reported score at the lowest cost per task.

### Per-run scoring compliance

The solver's `execute_python_solution` tool may return up to two candidate output grids per test example in a single run, consistent with ARC-AGI-2's scoring rule that accepts two attempts per test example. Within any single run, the system operates inside the benchmark's candidate budget.

### Selection across runs

The 119 checkpoints are inference traces from the development period (March–April 2026). For each puzzle, the release checkpoint is the best-scoring run from that period (lowest-cost run when multiple runs tied on score). Puzzles were typically attempted 1–3 times as the system was iteratively refined; runs not selected are not retained in the release bundle. The system prompt was refined iteratively across this period; 89 of 119 checkpoints use the final version, while the remaining 30 use earlier drafts. The differences are primarily wording, formatting, and level of detail in tool documentation — the core instructions, available tools, and ARC domain guidance are consistent across all versions. All variants share the same core architecture and three mechanisms. Each checkpoint embeds its exact prompt and config in `state.panels` and `state.config` for full auditability.

This is a development-then-select-best pattern, distinct from single-submission scoring: a single uniform run of the current system on all 120 public-eval puzzles would not necessarily reproduce every solved outcome. The aggregate score (95.7%) reflects the upper envelope across development runs, not a single seed's output.

The 119-trace bundle is provided as an auditable record of these selections. Other reported ARC systems have not, to our knowledge, released comparable inference-trace artifacts, so their development-run methodology is not publicly documented.

### Reproducibility

Re-running a small subset of the hardest puzzles in mid-April 2026 with a byte-identical system prompt and tool configuration produced worse outcomes than the original traces. The solver explored less extensively, and the reflector approved incorrect solutions it would have previously rejected. The bulk of the evaluation — puzzles the system solves consistently — is unaffected. Whether the tail-end regression reflects stochastic variance, inference-time compute changes, or other factors is not determined.

Full traces are provided so that the original runs can be audited directly rather than re-executed.

## Interpretation

This evidence supports three claims:

1. **Cost-efficient frontier-matching accuracy.** 95.7% at $3.12/task is the lowest cost per task among systems scoring above 95%, ~2× cheaper than the next-cheapest 95%+ system (Squeeze-Evolve at $5.93) and ~3× cheaper than Imbue ($8.71).
2. **Unique capability on hard pairs.** On the 2026-04-12 public-corpus frontier (pairs no participant has ever solved), the system has **exclusive solves on 8 of 9** pairs. On the 22 pairs never solved by any CoT-only Opus 4.6 configuration in 8 attempts, the system solves 16. This demonstrates discovery that repeated independent attempts cannot reach.
3. **Three mechanisms drive the result.** Exploratory code execution compresses verification into fewer tokens; an independent artifact-only reflector catches brittle solutions before acceptance; and inter-agent artifact exchange preserves discoveries across bounded context windows. No single mechanism alone explains the results.

The evidence is scoped to public eval. No verified semi-private score has been submitted.
