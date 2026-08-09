"""Append a compact snapshot of every finished run to `results.jsonl`.

Written after a review subagent overwrote `runs/` and `runs2/`, destroying the
traces for six won environments. The *figures* had to be reconstructed by hand
from a committed markdown table. This makes that reconstruction unnecessary next
time: every heartbeat, each finished run's derived numbers are appended to one
file that no analysis ever needs to open a run directory to read.

It is deliberately **append-only and idempotent-by-content**: a run already
recorded with identical figures is not appended again, so it can be called every
few minutes forever without growing without bound.

This is not a backup of the traces -- those are 14 MB each and the point of the
exercise is cheapness. It preserves what a result *is*, not what it was derived
from. Keeping the traces safe is a matter of not pointing agents at them.

**This file used to live only in the scratchpad, and that was the bug.** The one
store that reverts to an image snapshot when the container is replaced is the
worst possible home for the script whose entire purpose is surviving data loss.
CLAUDE.md records the general form ("a rule that lives only there is a rule that
expires") and this project has already paid it once, when `heartbeat.sh` was a
scratchpad copy that drifted five days behind the repo. It is a repo file now,
symlinked into the scratchpad by `rehydrate_box.sh`, because a symlink cannot
drift.

**And it was not recording the current arm.** The batch list read
`("runs", "runs2", "runs3", "ablate_nobaseline")` -- written when those were the
arms, never updated when `clean_rollouts` became the sweep. Measured 2026-08-09:
25 banked games on disk under `clean_rollouts`, **zero** rows for it in
`results.jsonl`. Two independent causes, either of which alone was enough:
the name was missing from the list, AND `glob("*/result.json")` cannot reach a
sweep result, which lives at `<game>/attempt_N/<gid>/result.json`.

The list stays explicit -- see the fabricated-fixtures note below -- but an
unlisted directory holding results is now REPORTED rather than skipped in
silence. A ledger that quietly covers three arms out of four looks exactly like
one that covers everything.
"""
import json
import os
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from athanor.ccarc3 import load, score_run                      # noqa: E402
from athanor.ccarc3.session import ledger_facts, run_cost       # noqa: E402

SP = pathlib.Path(
    os.environ.get("CCARC3_SCRATCH")
    or "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)
LEDGER = SP / "results.jsonl"

# Named explicitly, not globbed. `runs*` also matched `runs4/`, a directory of
# synthetic fixtures a review agent left behind, and a ledger that mixes
# fabricated rows with real ones is worse than no ledger.
#
# `ablate_nobaseline` is the baseline-free arm. It is recorded in the same
# ledger but MUST NOT be summed with the rest: those runs saw
# `baseline_actions`, these did not, so mixing them would silently blend two
# different experiments into one headline number. The `batch` field is the
# discriminator -- every reader must filter on it. `clean_rollouts` and the
# submission sweep are baseline-free too, and are likewise their own batches.
BATCHES = [
    "runs", "runs2", "runs3", "ablate_nobaseline", "clean_rollouts",
    # **Added 2026-08-09 after the structural check surfaced them.** All hold
    # real ARC-AGI-3 runs whose figures exist nowhere else once the scratchpad
    # reverts, which is the one thing this file is for. They are RECORDED rather
    # than silenced: `batch` is the discriminator and every reader must filter on
    # it, so preserving a voided or quarantined run under its own name loses
    # nothing and keeps the evidence.
    #
    # `clean_rollouts_void` and `clean_rollouts_stale_card` are sweeps that were
    # deliberately voided -- their batch names say so. `quarantine` is one
    # quarantined run. `rerun_bp35_fixed` is the bp35 re-run. `repro_report`,
    # `rev_old` and `rev_runs` are single ls20 reproductions.
    "clean_rollouts_void", "clean_rollouts_stale_card", "quarantine",
    "rerun_bp35_fixed", "repro_report", "rev_old", "rev_runs",
]
_sweep = os.environ.get("CCARC3_SWEEP_DIR")
if _sweep and _sweep not in BATCHES:
    BATCHES.append(_sweep)

#: Two layouts, because the sweep driver nests per attempt. Both patterns end at
#: a `result.json` whose `trace.jsonl`, `meta.json` and `stream.jsonl` are
#: siblings, which is what the enrichment below requires -- so the deep pattern
#: is not merely "one level further", it is the level where the trace lives.
#: `clean_result.json` (the sweep's per-game banked marker) is deliberately NOT
#: read: it has no trace beside it, so a row built from it would carry none of
#: the derived figures this ledger exists to preserve.
RESULT_PATTERNS = ("*/result.json", "*/attempt_*/*/result.json")


#: A real ARC-AGI-3 game id: four lowercase alphanumerics, a dash, eight hex.
#: Used to decide whether an unlisted directory is even a candidate arm.
GAME_ID = re.compile(r"^[a-z0-9]{4}-[0-9a-f]{8}$")

#: ccarc3 probes already adjudicated as not-arms. These DO hold real game ids --
#: they are genuine ARC-AGI-3 runs -- so only a decision can exclude them.
NOT_ARMS = frozenset({"smoke", "strat_easy", "strat_hard", "verify", "runs4"})


def _known_batch_dirs() -> set[str]:
    return set(BATCHES) | set(NOT_ARMS)


def unlisted_result_dirs() -> list[str]:
    """Scratchpad directories holding results that no batch name covers.

    The explicit list is right, and stale by default: it was written for the
    arms of the day and nothing made its omissions visible. This is the cheap
    half of the fix -- the reader still decides what is real, but a new arm
    cannot go unrecorded in silence.

    ``NOT_ARMS`` is the other half: directories already adjudicated as probes
    stay quiet, so a name appearing here means something new has arrived rather
    than that the warning exists.
    """
    known = _known_batch_dirs()
    out = []
    if not SP.is_dir():
        return out
    for child in sorted(SP.iterdir()):
        if not child.is_dir() or child.name in known:
            continue
        if _holds_arc3_results(child):
            out.append(child.name)
    return out


def _holds_arc3_results(root: pathlib.Path) -> bool:
    """Does this directory hold results from THIS benchmark?

    **Structural, because a name list could not stop growing.** The first
    version reported any directory containing a `result.json`, which on the live
    scratchpad meant `round6`/`round7`/`round8` (cc_harness, i.e. ARC-AGI-2 --
    they carry `task_id` and `hypothesis`, not `game_id`), `runs4` (synthetic
    fixtures whose ids are `g01`-style), and a tail of `rev_runs*` probes. Each
    one I named made the next one appear; enumerating them is the same
    stale-list defect this warning was added to fix, one level up.

    A directory qualifies only if some result carries a game id of the real
    shape. That excludes another benchmark's output and fabricated fixtures by
    construction rather than by memory.
    """
    for pattern in (*RESULT_PATTERNS, "*/clean_result.json"):
        for f in root.glob(pattern):
            try:
                gid = json.loads(f.read_text()).get("game_id")
            except Exception:                      # noqa: BLE001
                continue
            if isinstance(gid, str) and GAME_ID.match(gid):
                return True
    return False


def main() -> int:
    seen = set()
    if LEDGER.exists():
        for line in LEDGER.open():
            try:
                seen.add(json.dumps(json.loads(line), sort_keys=True))
            except json.JSONDecodeError:
                continue

    added = 0
    with LEDGER.open("a") as out:
        for batch in [SP / name for name in BATCHES]:
            if not batch.is_dir():
                continue
            results = []
            for pattern in RESULT_PATTERNS:
                results.extend(batch.glob(pattern))
            for result in sorted(set(results)):
                try:
                    stored = json.loads(result.read_text())
                except json.JSONDecodeError:
                    continue
                row = {"batch": batch.name, **stored}
                trace = result.parent / "trace.jsonl"
                meta = result.parent / "meta.json"
                if trace.exists():
                    row.update(ledger_facts(trace))
                    row.update(run_cost(result.parent / "stream.jsonl"))
                    if meta.exists():
                        baselines = json.loads(meta.read_text()).get("baseline_actions")
                        if baselines:
                            try:
                                scored = score_run(load(trace), baselines)
                                row["rhae"] = scored.score
                                row["per_level_actions"] = [
                                    level.agent for level in scored.levels
                                ]
                                row["baselines"] = list(baselines)
                            except ValueError:
                                pass
                key = json.dumps(row, sort_keys=True)
                if key not in seen:
                    out.write(key + "\n")
                    seen.add(key)
                    added += 1

    print(f"{added} new snapshot row(s); {LEDGER} now has {len(seen)} unique rows")
    for name in unlisted_result_dirs():
        print(f"UNRECORDED: {name}/ holds results but is not in BATCHES — "
              f"add it or it stays out of the ledger silently")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
