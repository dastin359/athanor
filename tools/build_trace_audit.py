"""Rebuild the CCARC3 trace-audit page from the preserved span store.

The page is a span tree per game, reconstructed from the `claude -p` NDJSON that
each solver was recorded with. It is published as artifact
`7447856a-b587-4d52-9c5c-a839de3eb6ee`.

**This is a merge, not a regeneration, and that is deliberate.** The NDJSON
streams lived only in the scratchpad and are now gone: of the 25 game directories
the page covers, 12 no longer exist and every one of the 13 survivors has since
been overwritten by a later run of the same game. `lp85-305b61c3` still has a
stream with exactly the stored line count, 1157, and even that is a different run
-- none of its message ids match. So a build that scanned the scratchpad would
quietly emit a smaller page full of different runs wearing the right names.

`evidence/ccarc3/trace_audit/spans.json.gz` is therefore the seed. Games already
in it are served from it untouched; `--ingest` adds new ones from their run
directories. Nothing is ever re-derived from a source that no longer exists.

**On the timing model.** Because no original stream survives, the span builder
below could not be calibrated against known-good output -- it reproduces the
stored *schema*, and its timings follow the model stated in `build_spans`, which
is not verified to be identical to the one that produced the stored spans. New
games are internally consistent; a hypothetical millisecond-level diff against
the 2026-08-05 build is not something this can check, and is not claimed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import pathlib
import re
import hashlib
import statistics
import subprocess
import tempfile
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
STORE = REPO / "evidence" / "ccarc3" / "trace_audit"
TEMPLATE = REPO / "tools" / "trace_audit_template.html"

# The artifact runtime rejects a rendered page over 16 MB. The 2026-08-05 build
# was already 15.1 MB, so a new game does not fit without trimming payloads --
# see `fit()`. Target leaves room for the host's own <head> wrapper.
CEILING = 16 * 1024 * 1024
TARGET = int(15.2 * 1024 * 1024)


def _ts(value: str) -> float:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def build_spans(stream: pathlib.Path) -> tuple[list[dict], int, int]:
    """Reconstruct the span tree for one recorded `claude -p` stream.

    Timing model, stated because it could not be calibrated (see module docstring):

    * `t0` is the first row carrying a `timestamp`; every `t` is relative to it.
    * Assistant rows sharing a `message.id` are one **turn**. The turn starts at
      its first row and ends when its last child ends.
    * Each content block becomes a child in row order -- `text` -> SAY,
      `thinking` -> THINK, `tool_use` -> TOOL.
    * A TOOL span ends when the `user` row carrying its `tool_result` arrives.
      SAY and THINK end when the next row of the same turn begins, since the
      stream gives no other end signal for them.
    * `root` spans the whole session; its cost, turn count and usage come from
      the terminal `result` row rather than being summed, so they match what the
      CLI itself reported.
    """
    rows, unparseable = [], 0
    for line in stream.open():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            unparseable += 1

    stamped = [r for r in rows if r.get("timestamp")]
    if not stamped:
        return [], unparseable, len(rows)
    t0 = _ts(stamped[0]["timestamp"])

    # When a tool's result came back, keyed by tool_use_id.
    result_at: dict[str, float] = {}
    for r in rows:
        if r.get("type") != "user" or not r.get("timestamp"):
            continue
        content = (r.get("message") or {}).get("content")
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("tool_use_id"):
                result_at[block["tool_use_id"]] = _ts(r["timestamp"]) - t0

    result_row = next((r for r in rows if r.get("type") == "result"), {})
    usage = result_row.get("usage") or {}

    spans: list[dict] = [{
        "id": "root", "parent": None, "kind": "CHAIN", "name": "session",
        "t": 0.0, "dur": _ts(stamped[-1]["timestamp"]) - t0, "depth": 0,
        "cost": result_row.get("total_cost_usd"),
        "subtype": result_row.get("subtype"),
        "is_error": bool(result_row.get("is_error")),
        "num_turns": result_row.get("num_turns"),
        "usage": {
            "input": usage.get("input_tokens"),
            "cache_create": usage.get("cache_creation_input_tokens"),
            "cache_read": usage.get("cache_read_input_tokens"),
            "output": usage.get("output_tokens"),
        },
    }]

    # Assistant rows arrive one block at a time but share a message id; that id
    # is the only thing tying a turn's blocks together.
    turns: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        if r.get("type") != "assistant" or not r.get("timestamp"):
            continue
        mid = (r.get("message") or {}).get("id")
        if mid is None:
            continue
        if mid not in turns:
            turns[mid] = []
            order.append(mid)
        turns[mid].append(r)

    think_total = 0
    think_peak = 0
    for n, mid in enumerate(order, start=1):
        group = turns[mid]
        turn_id = f"turn:{n}"
        start = _ts(group[0]["timestamp"]) - t0
        message = group[0].get("message") or {}

        children: list[dict] = []
        calls: list[str] = []
        blocks = 0
        think_blocks = 0
        text = None
        for row in group:
            at = _ts(row["timestamp"]) - t0
            content = (row.get("message") or {}).get("content") or []
            for i, block in enumerate(content, start=1):
                if not isinstance(block, dict):
                    continue
                blocks += 1
                kind = block.get("type")
                if kind == "text":
                    body = block.get("text") or ""
                    text = text or body
                    children.append({
                        "id": f"say:{mid}:{i}", "parent": turn_id, "kind": "SAY",
                        "name": "text", "t": at, "dur": 0.0, "depth": 2,
                        "out": body, "chars": len(body),
                    })
                elif kind == "thinking":
                    body = block.get("thinking") or ""
                    think_blocks += 1
                    est = len(body) // 4
                    think_total += est
                    think_peak = max(think_peak, est)
                    children.append({
                        "id": f"think:{mid}:{i}", "parent": turn_id, "kind": "THINK",
                        "name": "thinking", "t": at, "dur": 0.0, "depth": 2,
                        "out": body, "chars": len(body), "est": est,
                        "sig": len(block.get("signature") or ""),
                        "redacted": not body,
                    })
                elif kind == "tool_use":
                    name = block.get("name") or "?"
                    calls.append(name)
                    payload = block.get("input") or {}
                    head = payload.get("command") or payload.get("file_path") or ""
                    if len(head) > 90:
                        head = head[:89] + "…"
                    tid = block.get("id") or f"{mid}:{i}"
                    children.append({
                        "id": f"tool:{tid}", "parent": turn_id, "kind": "TOOL",
                        "name": name, "t": at,
                        "dur": max(0.0, result_at.get(tid, at) - at), "depth": 2,
                        "input": json.dumps(payload, indent=1), "head": head,
                        "out": "", "err": False,
                    })

        # SAY/THINK have no end event -- close them at the next sibling's start,
        # or at the turn's last known instant when they are last.
        end = max((c["t"] + c["dur"] for c in children), default=start)
        for i, child in enumerate(children):
            if child["kind"] in ("SAY", "THINK"):
                nxt = children[i + 1]["t"] if i + 1 < len(children) else end
                child["dur"] = max(0.0, nxt - child["t"])

        ttype = "say+do" if text and calls else ("say" if text else "do")
        turn = {
            "id": turn_id, "parent": "root", "kind": "LLM", "name": f"turn {n}",
            "t": start, "dur": max(0.0, end - start), "depth": 1,
            "model": (message.get("model") or "").replace("claude-", "") or None,
            "calls": calls, "think_blocks": think_blocks,
            "sub": group[0].get("parent_tool_use_id") is not None,
            "blocks": blocks, "ttype": ttype,
        }
        if text:
            turn["text"] = text[:400]
        spans.append(turn)
        spans.extend(children)

    # Attach tool output to its span, so the detail pane has something to show.
    by_id = {s["id"]: s for s in spans}
    for r in rows:
        if r.get("type") != "user":
            continue
        content = (r.get("message") or {}).get("content")
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            span = by_id.get(f"tool:{block.get('tool_use_id')}")
            if span is None:
                continue
            body = block.get("content")
            span["out"] = body if isinstance(body, str) else json.dumps(body)
            span["err"] = bool(block.get("is_error"))

    spans[0]["think_tokens"] = think_total
    spans[0]["think_peak"] = think_peak
    return spans, unparseable, len(rows)


def stream_start(stream: pathlib.Path) -> str:
    """The first timestamp in a solver stream, or "" if it carries none.

    Tiles for ingested runs said "start time unknown" -- every rollout and
    re-run, twelve of thirty-five. The value was never missing: `build_spans`
    reads exactly this field to compute `t0`, then throws the absolute time away
    and keeps only offsets. So the page was reporting an absence that was really
    a discard, and on the one axis you would use to line runs up against a
    container event.
    """
    try:
        with stream.open() as fh:
            for line in fh:
                if '"timestamp"' not in line:
                    continue
                try:
                    stamp = json.loads(line).get("timestamp")
                except ValueError:
                    continue
                if stamp:
                    return stamp
    except OSError:
        pass
    return ""


def ingest(game_dir: pathlib.Path) -> dict | None:
    """Read one finished run directory into a store entry."""
    result_file = game_dir / "result.json"
    if not result_file.exists():
        return None
    result = json.loads(result_file.read_text())
    result.setdefault("_batch", game_dir.parent.name)
    result.setdefault("_dir", game_dir.name)

    attempts = []
    # stream.1.jsonl is the earliest attempt; stream.jsonl the latest. Sort
    # numerically -- sp80 reached stream.11, and lexicographically that lands
    # between stream.1 and stream.2, silently scrambling the attempt order.
    numbered = sorted(
        (p for p in game_dir.glob("stream.*.jsonl") if p.name != "stream.jsonl"),
        key=lambda p: int(p.name.split(".")[1]),
    )
    streams = numbered + [game_dir / "stream.jsonl"]
    for stream in streams:
        if not stream.exists():
            continue
        spans, unparseable, lines = build_spans(stream)
        if spans:
            attempts.append({
                "file": stream.name, "spans": spans,
                "unparseable": unparseable, "lines": lines,
            })
    if not attempts:
        return None
    # Earliest attempt wins: the run started when its first solver did, not when
    # the one that happened to finish was launched.
    for stream in streams:
        if stream.exists() and (stamp := stream_start(stream)):
            result.setdefault("_started", stamp)
            break
    return {"result": result, "attempts": attempts}


def arc_actions_per_level(game_dir: pathlib.Path, game_id: str) -> list[int] | None:
    """Per-level action counts as **ARC** counted them, from `scorecard.json`.

    Preferred over the local ledger on purpose. Both re-runs produced the
    project's first non-empty `disagreements_with_server` -- ARC counted 12 more
    actions than our trace on `sp80` and 22 more on `tn36`, from responses that
    never reached `trace.jsonl` when the container died mid-action. Our trace
    under-counts, which makes trace-derived scores optimistic, so the server's
    numbers are the ones that go on the page.

    ARC reports the level rows cumulatively (`[[1, 64], [2, 196], ...]`), so the
    per-level cost is the difference between consecutive entries.

    **Differencing is delegated to `scoring.server_actions_per_level`** rather than
    repeated here. An earlier version of this function open-coded the cumulative
    differencing, which duplicated the canonical implementation and silently
    dropped its guard against level rows arriving out of order -- two copies of the
    same arithmetic that can drift apart, in the one place the project treats as
    authoritative. Only the play *selection* lives here, because the canonical
    helper defaults to the most recent play while ARC scores the best one.
    """
    card_file = game_dir / "scorecard.json"
    if not card_file.exists():
        return None
    try:
        scorecard = json.loads(card_file.read_text())
        plays = scorecard["cards"][game_id]["actions_by_level"] or []
    except (ValueError, KeyError, TypeError):
        return None
    if not plays:
        return None

    sys.path.insert(0, str(REPO / "src"))
    from athanor.ccarc3 import scoring  # noqa: PLC0415

    # ARC scores the best play: furthest first, then cheapest at that depth.
    best = max(range(len(plays)),
               key=lambda i: (len(plays[i]), -(plays[i][-1][1] if plays[i] else 0)))
    try:
        return scoring.server_actions_per_level(scorecard, game_id, play=best)
    except (KeyError, ValueError) as exc:
        print(f"    {game_id}: server actions unusable ({exc})", file=sys.stderr)
        return None


def runs_row(result: dict, game_dir: pathlib.Path | None = None) -> dict:
    """Summary row for the per-game tiles.

    `E`/`raw`/`cap` are filled from the game's baselines when they are reachable
    and left null otherwise -- a tile that cannot be scored says so rather than
    showing a plausible zero.
    """
    total = result.get("levels_total") or 0
    reached = result.get("levels_reached") or 0
    cap = (sum(range(1, reached + 1)) / sum(range(1, total + 1))) if total else None
    row = {
        "id": result.get("game_id"),
        "game": (result.get("game_id") or "").split("-")[0].upper(),
        "status": "scored", "E": None, "raw": None, "cap": cap,
        "levels": f"{reached}/{total}",
        "actions": result.get("actions_used"),
        "wall": result.get("duration_s"),
        "cost": result.get("cost_usd"),
        "plays": result.get("playthroughs"),
        "baseline": result.get("baseline_total"),
        "won": result.get("won"),
        # **Derived, not assumed.** These were hardcoded to
        # `{nobase: True, wall4h: False, think: False, arcn: False}` for every
        # ingested run, so `sb26` and `ft09` were published as having neither
        # thinking capture nor ARC-notation counting. Both had both. A default
        # that renders as a struck-through chip is not a default, it is a claim.
        #
        # `think` and `arcn` are recoverable from the run itself. `wall4h` is
        # not -- the configured limit leaves no trace in a run that finished
        # inside it -- so it is left None and the chip is omitted rather than
        # asserted either way.
        "cfg": {"nobase": True, "wall4h": None, "think": None, "arcn": None},
        "tier": "clean" if not result.get("error") else "crashed",
    }
    if started := result.get("_started"):
        row["started"] = started
        try:
            when = dt.datetime.fromisoformat(started.replace("Z", "+00:00"))
            # The existing rows read UTC-7; keep the two comparable rather than
            # having half the grid in one zone and half in another.
            row["started_local"] = (when - dt.timedelta(hours=7)).strftime("%b %-d %H:%M")
        except ValueError:
            pass
    # ARC bills every action except the opening RESET of each playthrough, so a
    # trace one row longer than the billed count is the notation working.
    rows, used = result.get("trace_rows"), result.get("actions_used")
    if isinstance(rows, int) and isinstance(used, int):
        row["cfg"]["arcn"] = rows != used

    try:  # optional: only possible when the API is reachable
        sys.path.insert(0, str(REPO / "src"))
        from athanor.ccarc3.client import baselines_for  # noqa: PLC0415
        from athanor.ccarc3 import scoring  # noqa: PLC0415

        per = arc_actions_per_level(game_dir, result["game_id"]) if game_dir else None
        base = baselines_for(result["game_id"])
        if base and per:
            # Unreached levels score zero; pad so the arrays line up.
            per = (per + [None] * len(base))[:len(base)]
            s = scoring.score_environment(base, per)
            row.update(E=s.score, raw=s.raw, cap=s.cap)
    except Exception as exc:  # noqa: BLE001 -- scoring is a bonus, not the job
        print(f"    baselines unavailable ({exc.__class__.__name__}); "
              f"E/raw left null for {result.get('game_id')}", file=sys.stderr)
    return row


def mark_contaminated(data: dict, runs: list) -> list[str]:
    """Void every run whose solver could read its own game's baselines.

    **The tier was a label; this makes it a measurement.** `clean` on these tiles
    only ever meant "one solver process, no container restart" -- an
    infrastructure property. It was never a contamination check, so when the
    baseline strip turned out not to be installed for the re-runs and rollouts,
    ten void runs sat on the page wearing a green badge that read as
    endorsement.

    The test is the per-level array itself, searched for in everything the solver
    saw or said. Not the total and not the budget: both are numbers that occur by
    coincidence in a 700 kB span dump, and a false void is as bad as a false
    clean. The array is not a coincidence -- if it is in the transcript, the
    solver was handed it.

    Needs the live `/api/games` to know the arrays. Without it nothing is marked,
    and the page says so rather than quietly downgrading the check.
    """
    try:
        sys.path.insert(0, str(REPO / "src"))
        from athanor.ccarc3 import list_games  # noqa: PLC0415

        base = {g.game_id: list(g.baseline_actions) for g in list_games()}
    except Exception as exc:  # noqa: BLE001
        print(f"    contamination check SKIPPED ({exc.__class__.__name__}): "
              f"baselines unreachable, tiers left as they were", file=sys.stderr)
        return []

    voided = []
    for rid, entry in data.items():
        arr = base.get(rid.split("@")[0])
        if not arr:
            continue
        pattern = r"[\[(]\s*" + r"\s*,\s*".join(str(n) for n in arr) + r"\s*[\])]"
        if re.search(pattern, json.dumps(entry.get("attempts", []))):
            voided.append(rid)
    for row in runs:
        if row.get("id") in voided and row.get("tier") != "excluded":
            row["tier"] = "void"
    mark_generation(runs)
    mark_clean(data, runs)
    return sorted(voided)


def backfill_start_times(runs: list) -> list[str]:
    """Recover start times for rows ingested before `stream_start` existed.

    Ten of thirty-five tiles read "start time unknown" — every rollout and
    re-run. The time was never unknown; the ingest that built those rows simply
    did not keep it. The preserved streams under `evidence/ccarc3/` still have
    it, so this reads it back rather than leaving the page asserting an absence.
    """
    evidence = REPO / "evidence" / "ccarc3"
    filled = []
    for row in runs:
        if row.get("started_local") or not row.get("id"):
            continue
        gid = row["id"].split("@")[0]
        # Earliest stream across every batch that holds this game: attempt order
        # is numeric, and `stream.11` sorts before `stream.2` as a string.
        cands = sorted(
            (p for batch in evidence.iterdir() if batch.is_dir()
             for p in batch.glob(f"{gid}/**/stream*.jsonl.gz")),
            key=lambda p: (0 if ".jsonl" == p.name[-11:-3] else 1, p.name),
        )
        for path in cands:
            stamp = ""
            try:
                with gzip.open(path, "rt") as fh:
                    for line in fh:
                        if '"timestamp"' not in line:
                            continue
                        try:
                            stamp = json.loads(line).get("timestamp") or ""
                        except ValueError:
                            continue
                        if stamp:
                            break
            except OSError:
                continue
            if not stamp:
                continue
            try:
                when = dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            row["started"] = stamp
            row["started_local"] = (when - dt.timedelta(hours=7)).strftime("%b %-d %H:%M")
            filled.append(f"{row['id']} {row['started_local']}")
            break
    return filled


# What a solver reads, plus what decides whether it may read it. The doctrine
# lives under `src/athanor/ccarc3/assets/`, so one path covers the prompt, the
# client and the scoring; `ablate_baselines.py` is the strip and the proxy wiring.
HARNESS_PATHS = ("src/athanor/ccarc3/", "tools/ablate_baselines.py")


# Exactly what a solver can see: the four files it reads, and the API surface it
# is allowed to reach. Everything else in the harness is invisible to it by
# construction.
SOLVER_FILES = ("CLAUDE.md", "DOCTRINE.md", "session.py", "meta.json")


def surface_digest(root: pathlib.Path) -> str:
    """Digest of everything the run's solver could observe.

    **This replaces counting commits, which counted diffs rather than
    differences.** `sb26` was marked superseded by a commit that moved the
    proxy's counter and cookie jar from module globals onto an instance -- 289
    lines across three files, and byte-for-byte no change to anything the solver
    reads. A refactor is not a new experiment.

    The digest covers the four files in the workspace plus the proxy's allowlist
    and the fields it strips from responses, which together define what the API
    will do when the solver asks.

    **Not the environment**, though that is solver-visible and matters more than
    any of them -- whether `ARC_API_KEY` and `CCARC3_MAX_ACTIONS` reach the
    child. It is excluded because it cannot be recovered from a finished run, so
    including it on the reference side only would make every digest mismatch by
    construction. (It did, briefly: both clean runs read `superseded` against a
    workspace proven byte-identical.) The environment is checked directly
    instead, per run, by `proofread_trace.py`, which reads it from the live
    process rather than inferring it.

    Two runs with the same digest saw the same harness, whatever happened to the
    source in between.
    """
    h = hashlib.sha256()
    for name in SOLVER_FILES:
        path = root / name
        h.update(name.encode() + b"\0")
        h.update((path.read_bytes() if path.exists() else b"") + b"\0")
    try:
        sys.path.insert(0, str(REPO / "src"))
        from athanor.ccarc3 import arc_proxy  # noqa: PLC0415

        h.update(b"allow:" + ",".join(p.pattern for p in arc_proxy.ALLOW).encode())
        h.update(b"hidden:" + ",".join(sorted(arc_proxy.HIDDEN_FIELDS)).encode())
    except Exception:  # noqa: BLE001 -- a digest without it is still comparable
        pass
    return h.hexdigest()[:16]


def reference_digest(game_id: str) -> str:
    """What today's code would ship to this game. Empty if it cannot be built."""
    try:
        sys.path.insert(0, str(REPO / "src"))
        sys.path.insert(0, str(REPO / "tools"))
        import ablate_baselines as ab  # noqa: PLC0415

        from athanor.ccarc3 import Ccarc3Config, list_games  # noqa: PLC0415
        from athanor.ccarc3 import session as sess  # noqa: PLC0415

        ab.install()
        info = next((g for g in list_games() if g.game_id == game_id), None)
        if info is None:
            return ""
        with tempfile.TemporaryDirectory() as tmp:
            ws = sess.build_workspace(
                Ccarc3Config(game_id, out_dir=pathlib.Path(tmp), budget_multiple=5.0), info)
            digest = surface_digest(ws.root)
        ab.release_proxy(game_id)
        return digest
    except Exception as exc:  # noqa: BLE001
        print(f"    reference surface unavailable for {game_id} ({exc.__class__.__name__})",
              file=sys.stderr)
        return ""


def harness_commits() -> list[str]:
    """Commit timestamps that changed the harness, newest first."""
    try:
        out = subprocess.run(
            ["git", "log", "--format=%cI", "--", *HARNESS_PATHS],
            cwd=REPO, capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.splitlines() if line.strip()]


def mark_generation(runs: list) -> int:
    """Tier each run by whether the harness it saw still matches today's.

    **Only a change the solver could see supersedes a run.** This counted
    commits to the package before, which counted diffs rather than differences:
    `sb26` was superseded by a 289-line refactor that left every byte the solver
    reads identical. So where a run recorded its solver surface, that digest is
    compared against what today's code would ship to the same game, and the
    commit count is reported alongside as context rather than as the verdict.

    Runs with no recorded surface -- the 25 arm games, whose workspaces were
    never preserved -- keep the commit-based estimate and are marked
    `estimated`, because a guess must not read as a measurement.

    `void` and `excluded` outrank both: a contaminated run's generation is not
    the interesting fact about it.
    """
    commits = harness_commits()
    refs: dict[str, str] = {}
    moved = 0
    for row in runs:
        if row.get("tier") in {"void", "excluded"} or not row.get("started"):
            continue
        try:
            when = dt.datetime.fromisoformat(row["started"].replace("Z", "+00:00"))
        except ValueError:
            continue
        row["behind"] = sum(1 for c in commits if dt.datetime.fromisoformat(c) > when)

        surface = row.get("surface")
        if surface:
            gid = row["id"].split("@")[0]
            if gid not in refs:
                refs[gid] = reference_digest(gid)
            if refs[gid]:
                was = row.get("tier")
                row["tier"] = "current" if surface == refs[gid] else "superseded"
                row["estimated"] = False
                moved += row["tier"] != was
                continue
        # No surface to compare: the commit count is all there is.
        row["tier"] = "current" if row["behind"] == 0 else "superseded"
        row["estimated"] = True
    return moved


def mark_clean(data: dict, runs: list) -> int:
    """Flag the runs that meet the strict criterion, for the page's filter.

    **Three things have all been called "clean" here and they are not the same.**
    The `clean` *tier* means "valid score, older instrumentation". The word in
    conversation means "no container restart". And after the baseline leak it
    also has to mean "the solver could not read its own answer". A filter is only
    useful if it means one thing, so this is the conjunction — a run is clean when
    every one of these holds:

    * not `void` and not `excluded` — no baselines reached the solver;
    * one attempt — a resumed run inherited `rules.json` from its own earlier
      self, which is the confound the rollouts exist to remove;
    * no `error` — an interrupted run is discarded, never scored.

    Stored as a boolean rather than computed in the page, so the definition lives
    next to the evidence it is derived from.
    """
    n = 0
    for row in runs:
        entry = data.get(row.get("id")) or {}
        result = entry.get("result") or {}
        row["clean"] = bool(
            row.get("tier") not in {"void", "excluded", "crashed"}
            and (result.get("attempts") or 1) == 1
            and not result.get("error")
        )
        n += row["clean"]
    return n


def summarise(data: dict) -> dict:
    """Headline figures, derived from the spans actually on the page."""
    bash: list[float] = []
    llm = tool = 0.0
    for entry in data.values():
        for attempt in entry["attempts"]:
            for span in attempt["spans"]:
                dur = span.get("dur") or 0.0
                if span["kind"] == "TOOL":
                    tool += dur
                    if span.get("name") == "Bash":
                        bash.append(dur)
                elif span["kind"] == "LLM":
                    llm += dur
    return {
        "games": len(data),
        "bash": len(bash),
        "median": statistics.median(bash) if bash else 0.0,
        "llm_pct": 100 * llm / (llm + tool) if (llm + tool) else 0.0,
    }


def fit(data: dict, runs: list, template: str) -> tuple[str, int, int | None]:
    """Render, trimming tool payloads only as far as the size ceiling demands.

    Returns the page plus how many payloads were trimmed and at what cap, so the
    caller can report it. A page that silently dropped half its evidence would
    still look like a complete audit.

    The headline figures are derived here rather than written into the template.
    The hand-written ones went stale the moment the page grew past the build they
    were computed on, and nothing in the page said so.
    """
    stats = summarise(data)
    # **A run on the current harness is never trimmed.** Trimming is a size
    # concession, and it should be paid by the runs nobody is going to read
    # closely -- superseded and void ones are kept for the record, not for
    # study. The runs that describe the harness as it stands are the ones worth
    # reading a 40 kB tool payload out of, so they ship whole and the older ones
    # absorb the ceiling.
    keep_whole = {r["id"] for r in runs if r.get("tier") == "current"}

    # **A void run's spans do not ship at all.** Ten of them cost 9.8 MB of a
    # 16 MB budget, and not one is a result -- they are the contaminated runs,
    # kept for the record rather than for reading. Their tiles stay, so the page
    # still says what happened and why; only the tree is dropped. The full spans
    # remain in the store, which is where anyone auditing a void run should be
    # looking anyway.
    #
    # This is what pays for the untrimmed `current` runs. Without it the ladder
    # was already at 500 chars with four of them, one rung from failing outright.
    drop_spans = {r["id"] for r in runs if r.get("tier") == "void"}

    def render(cap: int | None) -> str:
        payload = {k: v for k, v in data.items() if k not in drop_spans}
        if cap is not None:
            payload = json.loads(json.dumps(payload))
            for rid, entry in payload.items():
                if rid in keep_whole:
                    continue
                for attempt in entry["attempts"]:
                    for span in attempt["spans"]:
                        for field in ("out", "input", "text"):
                            body = span.get(field)
                            if isinstance(body, str) and len(body) > cap:
                                span[field] = body[:cap] + "…[trimmed]"
        return (template
                .replace("__DATA_JSON__", json.dumps(payload))
                .replace("__RUNS_JSON__", json.dumps(runs))
                .replace("__NGAMES__", str(stats["games"]))
                .replace("__BASH_CALLS__", f"{stats['bash']:,}")
                .replace("__BASH_MEDIAN__", f"{stats['median']:.2f}")
                .replace("__LLM_PCT__", f"{stats['llm_pct']:.0f}"))

    page = render(None)
    if len(page.encode()) <= TARGET:
        return page, 0, None

    for cap in (8000, 4000, 2000, 1000, 500):
        page = render(cap)
        if len(page.encode()) <= TARGET:
            trimmed = sum(
                1 for rid, e in data.items() if rid not in keep_whole
                for a in e["attempts"] for s in a["spans"]
                for f in ("out", "input", "text")
                if isinstance(s.get(f), str) and len(s[f]) > cap
            )
            return page, trimmed, cap
    raise SystemExit("cannot fit the page under the artifact ceiling even at 500 chars")


class _Ingest(argparse.Action):
    """Collect `--ingest DIR [--as ID]` pairs in the order they were given.

    A re-run and the arm run it supersedes share a `game_id`, so the store key
    has to be settable per ingest -- otherwise merging a re-run silently evicts
    the original from the page.
    """

    def __call__(self, parser, ns, value, option_string=None):
        pairs = getattr(ns, "pairs", None) or []
        if option_string == "--ingest":
            pairs.append([value, None])
        else:
            if not pairs:
                parser.error("--as must follow an --ingest")
            pairs[-1][1] = value
        ns.pairs = pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest", action=_Ingest, metavar="DIR",
                    help="run directory of a finished game to merge in")
    ap.add_argument("--as", action=_Ingest, dest="as_id", metavar="ID",
                    help="store the preceding --ingest under this id")
    ap.add_argument("--out", default=str(STORE / "trace_audit.html"))
    ap.add_argument("--no-save", action="store_true",
                    help="render without writing the merged store back")
    args = ap.parse_args()
    args.ingest = getattr(args, "pairs", []) or []

    data = json.load(gzip.open(STORE / "spans.json.gz"))
    runs = json.load(gzip.open(STORE / "runs.json.gz"))
    print(f"store: {len(data)} games, {len(runs)} run rows")

    added = []
    for raw, label in args.ingest:
        game_dir = pathlib.Path(raw)
        entry = ingest(game_dir)
        if entry is None:
            print(f"  {game_dir.name}: no result.json or no spans, skipped")
            continue
        gid = label or entry["result"].get("game_id") or game_dir.name
        verb = "replaced" if gid in data else "added"
        data[gid] = entry
        row = runs_row(entry["result"], game_dir)
        row["id"] = gid
        # Captured now, because the workspace is on disk now. A digest stored in
        # the row outlives the directory it was computed from, so a run stays
        # judgeable long after its scratchpad copy is gone.
        row["surface"] = surface_digest(game_dir)
        # Non-empty thinking means the stream carried it; every run before
        # `b5f4651` has `redacted: true` on all of them.
        row["cfg"]["think"] = any(
            s["kind"] == "THINK" and not s.get("redacted")
            for a in entry["attempts"] for s in a["spans"])
        if label and "@" in label:
            row["game"] = f"{row['game']} {label.split('@')[1]}"
        runs = [r for r in runs if r.get("id") != gid] + [row]
        spans = sum(len(a["spans"]) for a in entry["attempts"])
        print(f"  {gid}: {verb}, {len(entry['attempts'])} attempt(s), {spans} spans")
        added.append(gid)

    filled = backfill_start_times(runs)
    if filled:
        print(f"start times recovered for {len(filled)}: {', '.join(filled)}")

    voided = mark_contaminated(data, runs)
    if voided:
        print(f"VOID: {len(voided)} run(s) could read their own baselines — "
              f"{', '.join(voided)}")

    if (added or voided or filled) and not args.no_save:
        with gzip.open(STORE / "spans.json.gz", "wt") as fh:
            json.dump(data, fh)
        with gzip.open(STORE / "runs.json.gz", "wt") as fh:
            json.dump(runs, fh)
        print(f"store updated: {len(data)} games")

    page, trimmed, cap = fit(data, runs, TEMPLATE.read_text())
    out = pathlib.Path(args.out)
    out.write_text(page)
    size = len(page.encode())
    print(f"wrote {out} — {size/1048576:.2f} MB of a {CEILING/1048576:.0f} MB ceiling")
    if trimmed:
        print(f"NOTE: {trimmed} payloads trimmed to {cap} chars to fit the ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
