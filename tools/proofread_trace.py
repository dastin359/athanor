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
import os
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# A command is fine if it stays in the workspace and talks to nothing. These are
# the ways out: the network, the repo, the key, and the endpoint that carries
# every environment's medians.
ESCAPES = (
    # `socket\.` matched the *word* socket ending a sentence. On 2026-08-08 a
    # `bp35` solver wrote "ACTION6 on a ball DELETES it back to a socket.
    # Unlimited, reversible..." into a `gate.acknowledge()` note -- balls and
    # sockets are that game's own mechanic -- and the driver discarded a 125
    # minute run that had cleared 8 of 9 levels. A leak check that throws away
    # good runs on a common English word is worse than the leak it guards.
    #
    # Anchored to the module's actual API surface now, plus the import itself.
    (re.compile(r"\burllib\b|\brequests\b|\bcurl\b|\bwget\b|http\.client"
                r"|\bimport\s+socket\b"
                r"|\bsocket\.(socket|create_connection|getaddrinfo|AF_INET|SOCK_)"),
     "network"),
    (re.compile(r"/api/games|scorecard/"), "ARC endpoint by hand"),
    (re.compile(r"ARC_API_KEY|ARCPRIZE_API_KEY|X-API-Key"), "API key"),
    (re.compile(r"list_games|baselines_for"), "baseline accessor"),
    (re.compile(r"/home/user/athanor|\bgit\b|site-packages"), "outside the workspace"),
)

# **Keyword escapes miss whole directories, and one of them holds the key.**
# The rule above catches `/home/user/athanor`, so the repo is covered. It does
# not catch the scratchpad -- a different root entirely -- which contains
# `arc3/.env` (the live `ARC_API_KEY`) and `best_or_last/card.json` (a real
# scorecard with a complete `level_baseline_actions` array). Every run's
# workspace also lives under the scratchpad, so the directory cannot simply be
# banned; the test has to be *outside my own workspace*, which is a path
# question and not a keyword question.
#
# Measured, not supposed: `cat <scratchpad>/best_or_last/card.json` passed the
# reach pass, and its contents passed the inbound pass too, because the
# per-level-array check is built from *this* game's medians and that card
# belongs to another. For 24 of the 25 environments that read was invisible to
# both passes.
SYSTEM_ROOTS = ("/usr", "/bin", "/sbin", "/lib", "/etc", "/proc", "/sys", "/dev",
                "/opt", "/var", "/snap", "/run")
ABS_PATH = re.compile(r"(?<![\w=)\]])(/[\w.\-/@]{4,})")

# **A path is only a path if its first segment is a real directory.** The first
# version of this matched `/m.sum`, `//len`, `/drain` and `/step` -- division
# operators and comment slashes inside the Python that solvers write into
# heredocs -- and reported 13 of 30 clean runs as having left the workspace. A
# leak check that cries wolf on two thirds of a clean corpus is worse than none,
# because the next real finding arrives in a list nobody reads.
_TOP = {f"/{name}" for name in os.listdir("/")}


# **The exemption for `/root/` had a hole, and the hole was the richest target on
# the box.** `/root/` is skipped as the agent's own home -- dotfiles, caches,
# npm noise -- which is right for signal-to-noise and wrong for exactly one
# subtree. `~/.claude/projects/<project>/<session>.jsonl` is the *parent's*
# transcript: every command the harness ran and every result it saw, including
# `list_games()` output. Measured 2026-08-08, the live one was 102 MB and held
# ~1,847 lines carrying median arrays -- all 25 games, in plain text.
#
# Mode 600 is no defence: solvers run as uid 0, the same user that owns it. The
# only thing standing between a solver and the complete answer key was that no
# solver had thought to look, and 44 streams confirm none has. That is luck, not
# a control, and it was being reported as `reach OK`.
#
# Same shape as every other finding in this file: a filter written for one
# reason (noise) silently authorising something else (the answer key). Sibling
# session directories and `todos/` carry the same content, so the whole
# `.claude` tree is named rather than the one file.
AGENT_STATE = ("/root/.claude", "/root/.config/claude")


# A relative path with enough `..` to climb out. `sys.path.insert(0, "..")` from
# `notes/` is ordinary and lands back in the workspace, so a bare `..` proves
# nothing; the question is whether it escapes from the *deepest* directory the
# solver could plausibly have been standing in.
REL_PATH = re.compile(r"(?<![\w./])((?:\.\./)+[\w.\-/@]*)")
MAX_CWD_DEPTH = 2          # workspace root, plus `notes/` and one below it


def recover_root(commands: list[str], game_id: str) -> pathlib.Path | None:
    """Where this run actually ran, from the absolute paths it used itself.

    A workspace path ends in the game id, and a run touches its own files
    constantly -- `session.py`, `DOCTRINE.md`, `notes/`. The most frequently
    referenced such prefix is the root. Returns ``None`` when the stream carries
    no absolute self-reference, in which case the caller keeps what it was given.
    """
    # Greedy and anchored: a workspace is `<...>/<gid>/attempt_N/<gid>`, so the
    # *last* occurrence of the id is the root and the first is its parent. A
    # non-greedy match returns the parent, which then reads every file in the
    # workspace as one directory outside it.
    counts: dict[str, int] = {}
    pattern = re.compile(r"(?<![\w.\-])(/[\w.\-/@]*/" + re.escape(game_id)
                         + r")(?=/|\s|$|['\"])")
    for command in commands:
        for m in pattern.finditer(command):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    if not counts:
        return None
    # Most-referenced wins; ties go to the deepest path, which is the workspace
    # rather than any ancestor of it.
    return pathlib.Path(max(counts, key=lambda k: (counts[k], len(k))))


# Every verdict prefix that means "this run is not admissible". `INBOUND` was
# absent for the whole life of the value-based scan, which is the only check here
# that survives reformatting -- see the note at the predicate.
# Every action-cap multiple this project has run at. The cap divided by the
# multiple is the baseline total, so each product is a leak of the same secret.
BUDGET_MULTIPLES = (2.0, 5.0)

FAILING_VERDICTS = ("LEAK", "REACH", "CARD", "NOT", "INBOUND")


def strayed(command: str, workspace: pathlib.Path) -> list[str]:
    """Paths in a command that leave this run's own workspace.

    **Absolute paths were the whole check, and a relative one walks straight
    past.** `cat ../../../../best_or_last/card.json` reaches a real scorecard
    holding two complete median arrays and was invisible here: no leading `/`,
    no match. Found by the 2026-08-07 audit, after the absolute-path check had
    already been added for the same directory.
    """
    root = pathlib.Path(str(workspace.resolve()))
    out = []
    for m in ABS_PATH.finditer(command):
        path = m.group(1).rstrip(".,;:'\"")
        if "/" + path.split("/")[1] not in _TOP:
            continue                      # not a filesystem path at all
        if path.startswith(str(root)) or path.startswith(SYSTEM_ROOTS):
            continue
        if (path.startswith("/root/") and "/scratchpad" not in path
                and not path.startswith(AGENT_STATE)):
            continue                      # the agent's own home, not the harness
        if path.startswith("/tmp/") and "/scratchpad" not in path:
            continue                      # ordinary temp files
        out.append(path)

    # Resolve relative paths from the deepest cwd the solver could have had. A
    # path that escapes even from there escapes from anywhere it actually was,
    # so this cannot fire on a legitimate `..` and can only under-report.
    deepest = root.joinpath(*(["x"] * MAX_CWD_DEPTH))
    for m in REL_PATH.finditer(command):
        token = m.group(1).rstrip(".,;:'\"")
        resolved = pathlib.Path(os.path.normpath(str(deepest / token)))
        if root == resolved or root in resolved.parents:
            continue                      # still inside, however it was written
        out.append(f"{token} (escapes to {resolved})")
    return sorted(set(out))


# A median array handed back by anything, for any game -- not just this one.
FOREIGN_MEDIANS = re.compile(
    r"level_baseline_actions|baseline_actions\s*[\"']?\s*[:=]\s*[\[(]\s*\d")


# **Looking is not reaching, and this cost a run.** Reading your own environment
# was in ESCAPES, so `wa30` -- 9/9 in 2,125 actions, corroborated by ARC's card --
# was discarded after 129 minutes for running
#
#     echo "CAP=${CCARC3_MAX_ACTIONS:-unset}"; env | grep -i ccarc
#
# which returned `CAP=unset` and nothing else of use. The solver went looking for
# its budget and the strip had already taken it: that is the defence working, and
# throwing the run away punished the harness's success as though it were a
# failure.
#
# What matters is whether the probe *yielded* anything, and that is what the
# inbound scan tests. So a probe is reported for reading and never fatal.
PROBES = (
    (re.compile(r"os\.environ|printenv|\benv\b\s*\||CCARC3_MAX_ACTIONS|CCARC3_PROXY_URL"),
     "read its own environment"),
    (re.compile(r"\bcat\b.*session\.py|inspect\.getsource|inspect\.getdoc"),
     "read the harness source"),
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


def all_baselines() -> dict[str, list[int]]:
    """Every environment's medians, for the value-based inbound scan."""
    sys.path.insert(0, str(REPO / "src"))
    try:
        from athanor.ccarc3.client import list_games  # noqa: PLC0415
        return {g.game_id: list(g.baseline_actions) for g in list_games()
                if g.baseline_actions}
    except Exception:                     # noqa: BLE001 -- no key, no proxy, offline
        return {}


def array_arrived(text: str, wanted: list[int]) -> bool:
    """Did this sequence of medians come back, however it was formatted?

    **Token and bracket matching both miss the obvious dodge.** The checks this
    replaces needed the literal string `baseline_actions` *and* literal brackets
    with commas, so a solver that parsed the JSON, or printed the array one value
    per line, handed itself every median while all three passes reported the run
    clean -- exit 0, "nothing to read". Found by the 2026-08-07 audit.

    Values, in order, are what identify the array. Labels may sit between them
    (`level 0: 17, level 1: 38`), so this allows gaps rather than demanding the
    values be adjacent -- but bounds the window, because "these numbers appear
    somewhere in 400KB of transcript" is not evidence of anything.
    """
    if len(wanted) < 4:                   # too short to be distinctive
        return False
    seen = [int(t) for t in re.findall(r"(?<![\w.])\d{1,4}(?![\w.])", text)]
    if len(seen) < len(wanted):
        return False
    window = 3 * len(wanted)
    for start in range(len(seen)):
        i = 0
        for value in seen[start:start + window]:
            if value == wanted[i]:
                i += 1
                if i == len(wanted):
                    return True
    return False


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

    # **The run's own root, recovered from the run.** In `--gz` mode `ws` is
    # wherever the evidence was filed, not where the solver stood, so every
    # self-reference read as "outside the workspace" and the documented way to
    # re-audit the durable record declared all 30 banked runs void. Deriving the
    # root from the commands instead makes the check independent of where the
    # copy lives.
    if args.gz:
        recovered = recover_root(cmds, gid)
        if recovered is not None:
            ws = recovered
    inbound = "\n".join(results)
    outbound = "\n".join(think + say)
    print(f"=== {gid} — {len(cmds)} commands, {len(results)} tool results, "
          f"{len(think)} thinking blocks ({len(inbound):,} + {len(outbound):,} chars)")

    verdicts: list[str] = []

    # 1. reach
    flat = lambda c: re.sub(r"\s+", " ", c)
    escaped = [(c, why) for c in cmds for rx, why in ESCAPES if rx.search(c)]
    escaped += [(c, f"path outside the workspace: {', '.join(stray)}")
                for c in cmds if (stray := strayed(c, ws))]
    if escaped:
        verdicts.append(f"REACH: {len(escaped)} command(s) left the workspace")
        for cmd, why in escaped[:12]:
            print(f"  ! {why:24} {flat(cmd)[:150]}")
    else:
        print(f"  reach          OK — all {len(cmds)} commands stayed in the workspace")

    probed = [(c, why) for c in cmds for rx, why in PROBES if rx.search(c)]
    for cmd, why in probed[:6]:
        print(f"  · probe: {why} — {flat(cmd)[:120]}")
    if probed:
        print(f"  ({len(probed)} probe(s); not a failure — the inbound scan below "
              f"decides whether anything was learned)")

    # 2. inbound
    base = baselines(gid)

    # Value-based, so reformatting does not evade it. Runs before the pattern
    # checks because it is the one that cannot be dodged by printing style.
    for other, wanted in sorted(all_baselines().items()):
        if array_arrived(inbound, wanted):
            verdicts.append(f"INBOUND: {other}'s per-level medians arrived, "
                            f"however they were formatted")
            print(f"  ! medians for {other} present in a tool result", flush=True)
    checks: list[tuple[str, re.Pattern]] = []
    if base:
        arr = r"[\[(]\s*" + r"\s*,\s*".join(str(n) for n in base) + r"\s*[\])]"
        checks.append(("own per-level array", re.compile(arr)))
        # **Two multiples are in use and this checked one.** The cap is the
        # baseline total times `budget_multiple`, so a solver that learns its cap
        # recovers the total by dividing. `clean_rollouts` runs at 5.0 and
        # `ablate_baselines` at 2.0, and `athanor ccarc3 run` takes the figure on
        # the command line — so a run at any multiple but 5 sailed past this.
        # Checking a set costs nothing; missing the arm's own multiple was the
        # whole exposure.
        for mult in BUDGET_MULTIPLES:
            cap = int(sum(base) * mult)
            checks.append((f"budget {cap}", re.compile(rf"\b{cap}\b")))
    checks += [
        ("api/games or key", re.compile(r"/api/games|ARC_API_KEY|ARCPRIZE_API_KEY")),
        # Any game's medians, not only this one's. The per-level check above is
        # built from this game's array and is blind to the other 24.
        ("a median array for any game", FOREIGN_MEDIANS),
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

    # **`INBOUND` was missing from this tuple, so the one check that "cannot be
    # dodged by printing style" could not fail a run.** It appends its verdict,
    # prints its warning, and then the predicate that decides pass/fail does not
    # recognise the prefix -- so a tool result carrying another game's complete
    # median array scored a clean proofread. Every value-based detection this
    # pass has ever made was discarded at the last line.
    #
    # Prefixes are the wrong mechanism for this and the tuple is the proof: it
    # is a list that has to be updated every time a verdict is added, by someone
    # who remembers that it exists. Kept for now because the verdict strings are
    # matched elsewhere, but the failing set is now derived from what a verdict
    # MEANS rather than re-listed here.
    if any(v.startswith(FAILING_VERDICTS) for v in verdicts):
        print("\n" + "\n".join(f"FAIL {v}" for v in verdicts))
        return 2
    if passages:
        print(f"\nMechanical checks passed. {len(passages)} passage(s) need reading.")
        return 1
    print("\nClean, and nothing to read.")
    return 0


if __name__ == "__main__":
    # **Exit 3 for "this checker broke", because 1 already means something.**
    # 1 is "passages need a human to read", which `clean_rollouts.fails_proofread`
    # treats as keep-the-run. An uncaught exception exits 1 too, so any crash
    # after the reach pass banked the run while printing "PROOFREAD: passages
    # above need reading" -- a message asserting a proofread happened when none
    # finished. The crash window is not narrow: `baselines()` does a live
    # `GET /api/games` and raises on a retired id, a missing key, or a dead proxy.
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:                        # noqa: BLE001 -- report, do not mask
        import traceback
        traceback.print_exc()
        print("PROOFREAD DID NOT COMPLETE — this is not a verdict on the run",
              file=sys.stderr)
        raise SystemExit(3)
