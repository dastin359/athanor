"""Aggregate ARC-AGI-2 public-eval attempts from the HF corpus into per-config summaries.

Downloads `arcprize/arc_agi_v2_public_eval` (attempts split), groups by `test_id`,
and writes `hf_per_config.json` alongside this script. Each row reports the config's
task-level accuracy (ARC-AGI-2 metric) and mean cost per task.

Task-level scoring:
- Per pair: solved if either attempt_1 or attempt_2 is correct
- Per task: score = solved_pairs / total_pairs
- Overall: mean task score across attempted tasks

Usage:
    pip install datasets
    python tools/aggregate_hf_corpus.py
"""
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH   = SCRIPT_DIR / "hf_per_config.json"

ds = load_dataset("arcprize/arc_agi_v2_public_eval", "attempts", split="test")

# test_id → {(task_id, pair_idx): {"correct_any": bool, "cost_sum": float, "attempts": int}}
pair_data = defaultdict(lambda: defaultdict(lambda: {"correct_any": False, "cost_sum": 0.0, "attempts": 0}))

for row in ds:
    for col in ("attempt_1", "attempt_2"):
        raw = row[col]
        if not raw:
            continue
        try:
            a = json.loads(raw)
        except Exception:
            continue
        meta = a.get("metadata", {}) or {}
        tid = meta.get("test_id") or meta.get("model") or "UNK"
        task_id = meta.get("task_id")
        pair_idx = meta.get("pair_index")
        cost = (meta.get("cost") or {}).get("total_cost", 0.0) or 0.0
        correct = bool(a.get("correct"))
        if task_id is None or pair_idx is None:
            continue
        key = (task_id, pair_idx)
        p = pair_data[tid][key]
        if correct:
            p["correct_any"] = True
        p["cost_sum"] += float(cost)
        p["attempts"] += 1

summary = []
for tid, pairs in pair_data.items():
    tasks = defaultdict(lambda: {"pairs": 0, "solved": 0, "cost": 0.0})
    for (task_id, pair_idx), p in pairs.items():
        t = tasks[task_id]
        t["pairs"] += 1
        if p["correct_any"]:
            t["solved"] += 1
        t["cost"] += p["cost_sum"]
    if not tasks:
        continue
    n_tasks = len(tasks)
    task_scores = [t["solved"] / t["pairs"] for t in tasks.values()]
    total_task_score = sum(task_scores)
    task_accuracy_pct = (total_task_score / n_tasks) * 100
    total_cost = sum(t["cost"] for t in tasks.values())
    cost_per_task = total_cost / n_tasks
    summary.append({
        "test_id": tid,
        "n_tasks": n_tasks,
        "task_score_sum": round(total_task_score, 3),
        "task_accuracy_pct": round(task_accuracy_pct, 2),
        "cost_per_task": round(cost_per_task, 4),
        "total_cost": round(total_cost, 2),
    })

summary.sort(key=lambda x: -x["task_accuracy_pct"])

# Filter to well-attempted configs with non-zero cost (skip runs missing cost data)
filtered = [s for s in summary if s["n_tasks"] >= 100 and s["cost_per_task"] > 0]

print(f"\n{len(filtered)} well-attempted configs with cost data:\n")
print(f"{'test_id':<48} {'tasks':>6} {'acc':>6} {'$/task':>8}")
print("-" * 75)
for s in filtered[:50]:
    print(f"{s['test_id'][:47]:<48} {s['n_tasks']:>6} {s['task_accuracy_pct']:>5.1f}% ${s['cost_per_task']:>7.3f}")

with open(OUT_PATH, "w") as f:
    json.dump(filtered, f, indent=2)
print(f"\nSaved {len(filtered)} configs to {OUT_PATH}")
