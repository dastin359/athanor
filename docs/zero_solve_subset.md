# Frozen Zero-Solve Public Eval Subset

This document freezes a dated definition of a hard ARC-AGI-2 public-eval subset so the repo's narrative does not depend on a live dashboard.

## Snapshot

- Date: `2026-04-12`
- Source dataset: `arcprize/arc_agi_v2_public_eval`
- Config / split: `attempts` / `test`
- Dataset revision: `e321bbebaae20fa6432ca13c77ba21d9031c25e1`
- Dataset last modified: `2026-03-19T19:47:29.000Z`
- Number of rows in the frozen attempts parquet: `10,529`
- Machine-readable snapshot: `artifacts/public_eval_zero_solve/2026-04-12_snapshot.json`
- Reproduction queries:
  - `artifacts/public_eval_zero_solve/hf_attempts_zero_solve_all_models.sql`
  - `artifacts/public_eval_zero_solve/hf_attempts_zero_solve_claude_opus_4_6.sql`
  - `artifacts/public_eval_zero_solve/hf_attempts_zero_solve_pairs_min80.sql`

## Protocol

The frozen subset is defined from the public attempts corpus, not from private benchmark labels.

For each non-null `attempt_1` and `attempt_2` row in the frozen parquet:
- extract `task_id` from the JSON payload
- extract `model` from the JSON payload when a model-specific slice is needed
- count an attempt as solved iff the payload contains `"correct":true`

Important claim boundary:
- "zero-solve" here means zero successful attempts in this frozen public corpus
- it does **not** mean the task is universally impossible

## Global Zero-Solve Subset

Five tasks had zero successful attempts in the frozen public attempts corpus:

- `269e22fb` (`0 / 238`)
- `faa9f03d` (`0 / 133`)
- `abc82100` (`0 / 127`)
- `da515329` (`0 / 120`)
- `9bbf930d` (`0 / 117`)

## Pair-Level Frontier

Using the pair-level query with the additional coverage filter `num_attempt_logs >= 80`, the frontier is:

- **9** unsolved test outputs
- across **8** puzzles

Rows from the frozen corpus (formatted as `task_id / test_output_index`):

- `13e47133`, pair `0` (`117` attempt logs)
- `269e22fb`, pair `0` (`119`)
- `269e22fb`, pair `1` (`119`)
- `8b7bacbf`, pair `0` (`125`)
- `9bbf930d`, pair `0` (`117`)
- `a32d8b75`, pair `0` (`118`)
- `abc82100`, pair `0` (`121`)
- `da515329`, pair `0` (`120`)
- `faa9f03d`, pair `0` (`127`)

This is the precise benchmark-target view for agentic approaches, because it tracks unresolved **test outputs**, not only unresolved puzzles. Only `269e22fb` contributes more than one unresolved test output.

## `claude-opus-4-6` Zero-Solve Slice

Filtering the same frozen corpus to exact `model == "claude-opus-4-6"` gives a ten-task zero-solve slice:

- `269e22fb` (`0 / 16`)
- `62593bfd` (`0 / 16`)
- `16b78196` (`0 / 8`)
- `2b83f449` (`0 / 8`)
- `5545f144` (`0 / 8`)
- `9bbf930d` (`0 / 8`)
- `abc82100` (`0 / 8`)
- `da515329` (`0 / 8`)
- `dbff022c` (`0 / 8`)
- `faa9f03d` (`0 / 8`)

Compared with the global five-task subset, the `claude-opus-4-6` slice adds five extra zero-solve tasks:

- `62593bfd`
- `16b78196`
- `2b83f449`
- `5545f144`
- `dbff022c`

## Evidence In This Repo

Using the pair-level frontier as the primary view, this solver has exclusive solves on **8 of the 9** frontier pairs:

- `13e47133 / 0`
  - `release_runs/13e47133.json`
  - saved run reports `Task fully solved (ARC task score: 100.0% (2/2 test outputs correct))`
- `269e22fb / 0`
  - `release_runs/269e22fb.json`
  - saved run reports `Task fully solved (ARC task score: 100.0% (2/2 test outputs correct))`
- `269e22fb / 1`
  - `release_runs/269e22fb.json`
  - same saved run solves both evaluation examples for that task
- `8b7bacbf / 0`
  - `release_runs/8b7bacbf.json`
  - saved run reports `Task fully solved (ARC task score: 100.0% (2/2 test outputs correct))`
- `9bbf930d / 0`
  - `release_runs/9bbf930d.json`
  - saved run reports `Task fully solved (ARC task score: 100.0% (1/1 test outputs correct))`
- `a32d8b75 / 0`
  - `release_runs/a32d8b75.json`
  - saved run reports `Task fully solved (ARC task score: 100.0% (2/2 test outputs correct))`
- `abc82100 / 0`
  - `release_runs/abc82100.json`
  - saved run reports `Task fully solved (ARC task score: 100.0% (1/1 test outputs correct))`
- `da515329 / 0`
  - `release_runs/da515329.json`
  - saved run reports `Task fully solved (ARC task score: 100.0% (1/1 test outputs correct))`

Remaining pair-level frontier target:

- `faa9f03d / 0`
  - `release_runs/faa9f03d.json`
  - extended 120-turn attempt reporting `ARC task score: 0.0% (0/1 test outputs correct)`

**All 8 solved pairs are exclusive to this solver** — no other participant in the HuggingFace corpus has solved them as of the 2026-04-12 snapshot.

These are selected-case successes, not a claim of benchmark-total leadership.

## Claim Boundaries

- "Zero-solve" means zero successful attempts in the frozen public corpus. It does not imply the task is universally impossible — some of these pairs have been solved by this system and by other participants not represented in the HuggingFace corpus.
- The `≥80 attempt logs` filter exists because the HF corpus contains 5 phantom pairs (`abc82100/1`, `4a21e3da/1`, `f560132c/1`, `faa9f03d/1`, `b6f77b65/2`) with only 6 logged attempts each. These pairs do not exist in the official ARC-AGI-2 dataset — they reference test indices beyond what the puzzle actually has — and appear to originate from a single erroneous submission. Every legitimate pair has ≥115 attempts. The filter ensures the frontier includes only well-attempted pairs where many participants tried and none succeeded.
- Exclusive solves show that this system reached results that logged HuggingFace participants have not, on this specific slice of the public corpus. That is not the same as "state of the art on ARC-AGI-2" — semi-private and private eval scores remain the authoritative benchmarks.
