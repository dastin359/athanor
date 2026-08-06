"""Proofread one finished run's trace before its score is believed.

**Pattern-matching a run for known leak shapes is the weaker check.** It finds
the numbers you thought to look for. It cannot find a solver that reached
outside its workspace in a way nobody anticipated, and it cannot tell a solver
*inferring* a hidden baseline from one *reading* it — the two produce nearly
identical prose. `ka59` was caught because it said what it had done, not because
a regex matched.

So this does the mechanical half exhaustively and then hands back a focused
digest for the half that needs reading. Three passes:

1. **Reach.** Every command the solver ran, flagged when it leaves the workspace
   or touches the network. This is the pass that would have caught `ka59`:
   nine `urllib` calls to `/api/games`.
2. **Inbound.** Every tool *result* — what the environment handed back, which is
   where a leak actually arrives. Scanned for this game's own per-level array,
   its budget, the API surface, and the pace line `status()` prints when
   baselines are visible. Solver text is scanned too, but a solver quoting a
   number it was given is a symptom; the tool result is the cause.
3. **Corroboration.** ARC's own card against the claimed result, because a run
   only we can confirm is not a result. A proxy bug once left `sb26` finishing
   8/8 with its card frozen at level 3.

What it cannot decide, it does not pretend to: every passage mentioning a
baseline, a budget or a prior run is printed with context under READ THIS, and
the exit code says so. Exit 0 means the mechanical checks passed and there is
nothing to read; 1 means passages need a human or a model to judge; 2 means a
leak was found and the run is void.

    python tools/proofread_trace.py <workspace>          # live run directory
    python tools/proofread_trace.py <preserved> --gz     # evidence/*/attempt_N/<game>/
"""
from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# A command is fine if it stays in the workspace and talks to nothing. These are
# the ways out: the network, the repo, the key, and the endpoint that carries
# every environment's medians.
ESCAPES = (
    (re.compile(r"\burllib\b|\brequests\b|\bcurl\b|\bwget\b|http\.client|socket\."), "network"),
    (re.compile(r"/api/games|scorecard/"), "ARC endpoint by hand"),
    (re.compile(r"ARC_API_KEY|ARCPRIZE_API_KEY|X-API-Key"), "API key"),
    (re.compile(r"list_games|baselines_for"), "baseline accessor"),
    (re.compile(r"/home/user/athanor|\bgit\b|site-packages"), "outside the workspace"),
    (re.compile(r"os\.environ|printenv|\benv\b\s*\|"), "own environment"),
)

# Passages a person has to judge. Not evidence of anything on their own -- the
# doctrine teaches the scoring formula on purpose, so "baseline" appears in every
# clean run.
JUDGE = re.compile(
    r"baseline|human median|budget of|action cap|previous (?:run|attempt|loss)"
    r"|prior (?:run|attempt)|last time|earlier run|/api/",
    re.I,
)


def read(path: pathlib.Path, gz: bool) -> str:
    target = path.with_suffix(path.suffix + ".gz") if gz else path
    if not target.exists():
        return ""
    try:
        return gzip.open(target, "rt", errors="ignore").read() if gz else \
            target.read_text(errors="ignore")
    except OSError:
        return ""


def blocks(stream_text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """(commands, tool results, thinking, text) from a recorded solver stream."""
    cmds, results, think, say = [], [], [], []
    for line in stream_text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        kind = row.get("type")
        content = (row.get("message") or {}).get("content") or []
        if kind == "assistant":
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "thinking":
                    think.append(b.get("thinking") or "")
                elif b.get("type") == "text":
                    say.append(b.get("text") or "")
                elif b.get("type") == "tool_use":
                    i = b.get("input") or {}
                    cmds.append(str(i.get("command") or i.get("file_path")
                                    or i.get("pattern") or json.dumps(i)))
        elif kind == "user":
            for b in content:
                if isinstance(b, dict) and b.get("tool_use_id"):
                    c = b.get("content")
                    results.append(c if isinstance(c, str) else json.dumps(c))
    return cmds, results, think, say


def baselines(game_id: str) -> list[int]:
    sys.path.insert(0, str(REPO / "src"))
    from athanor.ccarc3.client import baselines_for  # noqa: PLC0415

    return list(baselines_for(game_id) or [])


def context(blob: str, rx: re.Pattern, width: int = 200, limit: int = 40) -> list[str]:
    out, seen = [], set()
    for m in rx.finditer(blob):
        frag = blob[max(0, m.start() - width):m.end() + width].replace("\n", " ")
        key = re.sub(r"\s+", " ", frag)[width // 2:width + 60]
        if key in seen:
            continue
        seen.add(key)
        out.append(re.sub(r"\s+", " ", frag))
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workspace")
    ap.add_argument("--gz", action="store_true", help="preserved evidence, files are .gz")
    ap.add_argument("--quiet", action="store_true", help="only findings, no digest")
    args = ap.parse_args()
    ws = pathlib.Path(args.workspace)

    result = json.loads(read(ws / "result.json", args.gz) or "{}")
    gid = result.get("game_id") or ws.name
    stream = read(ws / "stream.jsonl", args.gz)
    if not stream:
        print(f"{gid}: no stream to proofread", file=sys.stderr)
        return 2

    cmds, results, think, say = blocks(stream)
    inbound = "\n".join(results)
    outbound = "\n".join(think + say)
    print(f"=== {gid} — {len(cmds)} commands, {len(results)} tool results, "
          f"{len(think)} thinking blocks ({len(inbound):,} + {len(outbound):,} chars)")

    verdicts: list[str] = []

    # 1. reach
    escaped = [(c, why) for c in cmds for rx, why in ESCAPES if rx.search(c)]
    if escaped:
        verdicts.append(f"REACH: {len(escaped)} command(s) left the workspace")
        for cmd, why in escaped[:12]:
            print(f"  ! {why:24} {re.sub(r'[[:space:]]+', ' ', cmd)[:150]}")
    else:
        print(f"  reach          OK — all {len(cmds)} commands stayed in the workspace")

    # 2. inbound
    base = baselines(gid)
    checks: list[tuple[str, re.Pattern]] = []
    if base:
        arr = r"[\[(]\s*" + r"\s*,\s*".join(str(n) for n in base) + r"\s*[\])]"
        checks.append(("own per-level array", re.compile(arr)))
        checks.append((f"budget {sum(base) * 5}", re.compile(rf"\b{sum(base) * 5}\b")))
    checks += [
        ("api/games or key", re.compile(r"/api/games|ARC_API_KEY|ARCPRIZE_API_KEY")),
        # `status()` prints the pace ratio only when baselines are visible.
        #
        # **Anchored to this game's id, because the doctrine quotes the format.**
        # §6 shows a worked example -- `g: level 3/6 ... [190/55 on this level =
        # 3.5x] <- OVER BASELINE` -- so an unanchored pattern fires on every run
        # that reads its own doctrine, which is every run. It made the check
        # report a leak on a workspace that had none, and a check that always
        # fails is a check nobody reads.
        ("pace line", re.compile(rf"{re.escape(gid)}: level .*\[\d+/\d+ on this level")),
    ]
    for name, rx in checks:
        where = [w for w, blob in (("tool results", inbound), ("solver", outbound))
                 if rx.search(blob)]
        if where:
            verdicts.append(f"LEAK: {name} present in {' and '.join(where)}")
            for frag in context(inbound if "tool results" in where else outbound, rx, 160, 3):
                print(f"  ! {name}: ...{frag}...")
        else:
            print(f"  {name:14} absent")

    # 3. corroboration
    card = json.loads(read(ws / "scorecard.json", args.gz) or "{}")
    entry = (card.get("cards") or {}).get(gid) or {}
    done = entry.get("levels_completed") or []
    best, reached = (max(done) if done else 0), result.get("levels_reached") or 0
    if not card:
        verdicts.append("CARD: no scorecard preserved")
    elif best < reached:
        verdicts.append(f"CARD: shows {best} levels, result claims {reached}")
    else:
        print(f"  card           OK — {best} levels, {entry.get('total_actions')} actions "
              f"vs result {result.get('actions_used')}")
    if (result.get("attempts") or 1) != 1 or result.get("error"):
        verdicts.append(f"NOT ONE-SHOT: attempts={result.get('attempts')} "
                        f"error={result.get('error')}")

    # what a person still has to read
    passages = context(outbound, JUDGE)
    if not args.quiet and passages:
        print(f"\n=== READ THIS — {len(passages)} passage(s) mentioning a baseline, "
              f"a budget or a prior run.\nInference is fine; reading is not. Judge each.\n")
        for p in passages:
            print(f"  · ...{p}...\n")

    if any(v.startswith(("LEAK", "REACH", "CARD", "NOT")) for v in verdicts):
        print("\n" + "\n".join(f"FAIL {v}" for v in verdicts))
        return 2
    if passages:
        print(f"\nMechanical checks passed. {len(passages)} passage(s) need reading.")
        return 1
    print("\nClean, and nothing to read.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
