"""The reach pass, which decides whether a run's evidence is believed.

Two defects, found by the 2026-08-07 audit *after* the pass had already been
rewritten once for the same directory:

1. It looked only at absolute paths, so `cat ../../../../best_or_last/card.json`
   — which reaches a real scorecard holding two complete `level_baseline_actions`
   arrays — was invisible. Every other check missed the same read: the
   own-per-level-array test needs literal brackets and commas, and the
   foreign-median test needs the literal string `baseline_actions`, both of which
   parsing rather than printing avoids.
2. In `--gz` mode the root came from wherever the evidence was filed, not from
   where the solver stood, so every self-reference read as an escape and the
   documented way to re-audit the durable record declared all 30 banked runs
   void. A check that fails on the whole corpus is not a check.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import pytest

import proofread_trace as pt

WS = pathlib.Path("/tmp/scratchpad/clean_rollouts/zz99-deadbeef/attempt_1/zz99-deadbeef")


@pytest.mark.parametrize("command", [
    "cat ../../../../best_or_last/card.json",
    "cat ../../../../arc3/.env",
    "python3 -c \"print(open('../../../../../evidence/x').read())\"",
])
def test_a_relative_path_that_escapes_the_workspace_is_flagged(command):
    assert pt.strayed(command, WS), f"relative escape went unseen: {command}"


@pytest.mark.parametrize("command", [
    "sys.path.insert(0,'..')",          # from notes/, lands back in the workspace
    "cat ../notes/plan.md",
    "cat notes/plan.md",
    "ls /usr/bin/python3",
    f"cd {WS} && cat session.py",
])
def test_ordinary_work_inside_the_workspace_is_not_flagged(command):
    """A `..` is not an escape. Resolution is from the deepest plausible cwd, so
    this can only under-report — never cry wolf on a run that stayed put."""
    assert not pt.strayed(command, WS), f"false positive on: {command}"


def test_the_root_is_recovered_from_the_run_not_from_where_evidence_sits():
    """`--gz` hands in the evidence directory; the run ran somewhere else."""
    live = "/tmp/scratchpad/clean_rollouts/zz99-deadbeef/attempt_1/zz99-deadbeef"
    cmds = [f"cd {live} && cat session.py",
            f"python3 {live}/notes/probe.py",
            f"cat {live}/DOCTRINE.md"]
    assert pt.recover_root(cmds, "zz99-deadbeef") == pathlib.Path(live)


def test_root_recovery_declines_rather_than_guessing():
    """No absolute self-reference means keep what the caller passed in."""
    assert pt.recover_root(["cat notes/plan.md", "ls"], "zz99-deadbeef") is None


# ---------------------------------------------------------------------------
# The inbound pass. Reach says what a command touched; inbound says what came
# back, which is where a leak actually arrives.
# ---------------------------------------------------------------------------

LP85 = [17, 38, 31, 16, 41, 60, 26, 159]


@pytest.mark.parametrize("label,text", [
    ("one value per line", "\n".join(str(n) for n in LP85)),
    ("parsed into a tuple", f"medians = ({', '.join(str(n) for n in LP85)})"),
    ("labelled per level", ", ".join(f"level {i}: {n}" for i, n in enumerate(LP85))),
    ("json without the key", str(LP85)),
])
def test_a_median_array_is_caught_however_it_is_formatted(label, text):
    """Token and bracket matching both miss the obvious dodge.

    The checks this replaces needed the literal string `baseline_actions` *and*
    literal brackets with commas. A solver that parsed the JSON, or printed the
    array one value per line, handed itself every median while all three passes
    reported the run clean — exit 0, "nothing to read".
    """
    assert pt.array_arrived(text, LP85), f"median array evaded the scan: {label}"


@pytest.mark.parametrize("text", [
    "the board is 64x64 with 16 colours and 8 levels",
    "17 and 38 appeared but nothing else did",
    "actions used: 159, 26, 60 — descending, not the published order",
])
def test_ordinary_transcript_numbers_do_not_trip_the_scan(text):
    """A scan over every integer in a 400KB transcript must not cry wolf.

    Verified against the corpus too: 0 findings across all 30 preserved runs.
    """
    assert not pt.array_arrived(text, LP85)


def test_a_short_array_is_not_distinctive_enough_to_assert_on():
    """Three numbers collide with ordinary prose; the scan declines to guess."""
    assert not pt.array_arrived("1 2 3", [1, 2, 3])


# ---------------------------------------------------------------------------
# ESCAPES had false-positive tests for `strayed()` and for the median scans, but
# not for its own patterns — and that is the one that destroyed data.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,command", [
    ("the word socket ending a sentence",
     "gate.acknowledge('ACTION6 on a ball DELETES it back to a socket. "
     "Unlimited, reversible, and the positions are fixed')"),
    ("a socket in game prose", "notes: the socket at (12,30) accepts one ball"),
    ("requested, not requests", "the level requested a second click"),
    ("curled, not curl", "the path curled around the wall"),
])
def test_escapes_do_not_fire_on_ordinary_game_prose(label, command):
    """A leak check that throws away good runs is worse than the leak it guards.

    Measured 2026-08-08: `socket\\.` matched *"back to a socket. Unlimited"* in a
    `bp35` solver's own note — balls and sockets are that game's mechanic — and
    the driver discarded a 125-minute run that had cleared 8 of 9 levels. The
    pattern is now anchored to `import socket` and the module's real API.
    """
    assert not [w for rx, w in pt.ESCAPES if rx.search(command)], (
        f"false positive on {label}: {command[:80]}"
    )


@pytest.mark.parametrize("command", [
    "import socket; s = socket.socket()",
    "socket.create_connection(('three.arcprize.org', 443))",
    "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
    "import urllib.request",
    "curl -H 'X-API-Key: k' https://three.arcprize.org/api/games",
    "python3 -c \"import requests; requests.get(u)\"",
])
def test_escapes_still_catch_every_real_network_reach(command):
    """Anchoring must not cost coverage."""
    assert [w for rx, w in pt.ESCAPES if rx.search(command)], f"missed: {command}"


def test_the_parents_transcript_is_not_exempt(tmp_path):
    """**The richest target on the box sat inside a whitelisted directory.**

    `/root/` is skipped as the agent's own home — dotfiles, caches, npm noise —
    which is right for signal-to-noise and wrong for exactly one subtree.
    `~/.claude/projects/<project>/<session>.jsonl` is the parent harness's own
    transcript: every command it ran and every result it saw, `list_games()`
    included. Measured 2026-08-08 the live one was 102 MB with ~1,847 lines
    carrying median arrays for all 25 games, in plain text, and mode 600 is no
    defence because solvers run as the same uid. It was reported as `reach OK`.
    """
    ws = tmp_path / "zz00-x"
    ws.mkdir()
    cmd = ("grep -o 'baseline_actions[^]]*]' "
           "/root/.claude/projects/-home-user-athanor/abc123.jsonl | head")

    assert any(".claude" in hit for hit in pt.strayed(cmd, ws)), (
        "a solver reading the parent's transcript reads the whole answer key"
    )


def test_a_sibling_session_and_the_todo_store_are_covered_too(tmp_path):
    """Naming one file would leave the same content one directory away."""
    ws = tmp_path / "zz00-x"
    ws.mkdir()

    assert pt.strayed("cat /root/.claude/projects/-other-project/deadbeef.jsonl", ws)
    assert pt.strayed("cat /root/.claude/todos/whatever.json", ws)
    assert pt.strayed("cat /root/.config/claude/anything", ws)


def test_ordinary_agent_home_noise_is_still_exempt(tmp_path):
    """The exemption exists because a check that cries wolf gets ignored. It has
    to keep doing its job, or closing this hole just reopens the old one."""
    ws = tmp_path / "zz00-x"
    ws.mkdir()

    for benign in ("cat /root/.bashrc",
                   "ls /root/.npm/_cacache",
                   "python -c 'import sys; print(sys.path)' > /root/out.txt"):
        assert not pt.strayed(benign, ws), benign


def test_an_inbound_median_array_actually_fails_the_run():
    """**The one check that survives reformatting could not fail a run.**

    `FAILING_VERDICTS` decides pass/fail by prefix, and `INBOUND` — emitted by
    the value-based foreign-median scan, the check the code itself calls "the one
    that cannot be dodged by printing style" — was not in the tuple. It appended
    its verdict, printed its warning, and the predicate did not recognise it, so
    a tool result carrying another game's complete median array scored a clean
    proofread. Every detection that pass ever made was discarded at the last line.
    """
    assert "INBOUND" in pt.FAILING_VERDICTS, (
        "a verdict nothing acts on is a check that reports by not running"
    )
    for prefix in ("LEAK", "REACH", "CARD", "NOT", "INBOUND"):
        assert any(v.startswith(prefix) for v in [prefix + ": x"]), prefix


def test_the_doctrine_states_no_baseline_value():
    """The DOCTRINE is the one file every solver is instructed to read, so a
    number recoverable from it reaches every run.

    Two survived until 2026-08-08: "games range from 171 to 1843 total" were the
    exact minimum and maximum baseline totals across the 25 environments, and a
    worked `status()` line read `[190/h on this level = 3.5x]` — a redaction that
    removed the value and left both of its factors, pinning it to [53.5, 55.1]
    against a true 55.
    """
    doctrine = (pathlib.Path(__file__).resolve().parent.parent
                / "src/athanor/ccarc3/assets/CCARC3_DOCTRINE.md").read_text()

    assert "171" not in doctrine and "1843" not in doctrine, (
        "the extremes of the baseline-total distribution are two real totals"
    )
    assert "190/h" not in doctrine, (
        "a count beside its own ratio is the median in one division"
    )
