"""No tool may reach this box's clone path or scratchpad UUID unconditionally.

The point of the harness is that it runs somewhere else -- another account,
another branch, another container -- and every path it needs is either derived
from its own location or overridable by environment. Two lists already assert
that, in `test_ccarc3_scratch_path_is_overridable.py`, and both are enumerations:
six shell tools and four importable modules, hand-maintained. `tools/` holds 23
files, and the omissions include `refresh_audit.sh`, `rehydrate_box.sh`,
`proofread_trace.py`, `build_trace_audit.py`, `snapshot_results.py` and
`rerun_losses.py`.

That is the third or fourth instance of one defect tonight: a hand-maintained
list of the things of the day, going stale with no symptom. `snapshot_results`'s
batch list silently omitted 25 banked games; the live panel's arm allowlist
rendered "nothing running" rather than "I cannot see it"; `rehydrate_box.sh`'s
symlink list left a pre-fix baseline strip on a replaced container. Each was
fixed by replacing the enumeration with a rule. This is that rule for the
portability check, and it covers a tool added tomorrow without anyone
remembering.

**A literal is fine as a fallback.** `os.environ.get("CCARC3_SCRATCH") or
"/tmp/claude-0/…"` is the intended shape -- it makes the common case work with no
setup and the foreign case work with one variable. What must not exist is a use
that no variable can displace.

Verified live on 2026-08-09 against a second clone at a different path, on a
different branch, with a different scratchpad: `rehydrate_box.sh --links-only`
pointed every symlink into the second clone, all five importable tools derived
their paths there, `preserve_evidence.sh --once` wrote and pushed to that clone's
own origin, `restore_clean_rollouts.py` read it back -- and this repository was
untouched throughout: nothing staged, no foreign branch, HEAD unchanged.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"

# This container's two identities. Neither is a property of the code.
SCRATCH_UUID = "a3375e8f-271e-5133-96a4-a40a6a06a752"
CLONE_PATH = "/home/user/athanor"

# What makes a literal a fallback rather than a pin, in either language.
OVERRIDE_MARKERS = ("environ.get", "getenv", "CCARC3_", ":-")


def _tools() -> list[pathlib.Path]:
    return [p for p in sorted(TOOLS.iterdir())
            if p.is_file() and p.suffix in (".sh", ".py")]


def _is_prose(line: str) -> bool:
    s = line.strip()
    return s.startswith("#") or s.startswith('"""') or s.startswith("*") or not s


def _unguarded(path: pathlib.Path) -> list[str]:
    """Lines using a container-specific literal with nothing able to displace it."""
    lines = path.read_text(errors="ignore").splitlines()
    out = []
    for i, line in enumerate(lines, start=1):
        if SCRATCH_UUID not in line and CLONE_PATH not in line:
            continue
        if _is_prose(line):
            continue
        # The override may sit a line or three above the literal, which is how
        # a wrapped `os.environ.get(...)\n or "<default>"` reads.
        context = "\n".join(lines[max(0, i - 4):i])
        if any(m in context for m in OVERRIDE_MARKERS):
            continue
        out.append(f"{path.name}:{i}: {line.strip()[:110]}")
    return out


def test_the_scan_actually_sees_the_tools():
    """A rule over an empty set passes by not running."""
    found = _tools()
    assert len(found) >= 20, f"only {len(found)} tools found; the scan is looking in the wrong place"
    names = {p.name for p in found}
    for expected in ("clean_rollouts.py", "preserve_evidence.sh", "supervisor.sh",
                     "rehydrate_box.sh", "proofread_trace.py"):
        assert expected in names, f"{expected} is not in the scanned set"


@pytest.mark.parametrize("tool", _tools(), ids=lambda p: p.name)
def test_no_container_specific_literal_is_unconditional(tool):
    pinned = _unguarded(tool)
    assert pinned == [], (
        "this box's clone path or scratchpad UUID is used with nothing able to "
        "displace it, so the tool cannot run on another account:\n  "
        + "\n  ".join(pinned)
    )


def test_the_scan_would_catch_a_pin(tmp_path):
    """The rule has to be able to fail, or it asserts nothing."""
    planted = tmp_path / "planted.py"
    planted.write_text(
        f'SP = pathlib.Path("/tmp/claude-0/-home-user-athanor/{SCRATCH_UUID}/scratchpad")\n',
        encoding="utf-8")
    assert _unguarded(planted), "a hard pin was not detected"

    guarded = tmp_path / "guarded.py"
    guarded.write_text(
        'SP = pathlib.Path(\n'
        '    os.environ.get("CCARC3_SCRATCH")\n'
        f'    or "/tmp/claude-0/-home-user-athanor/{SCRATCH_UUID}/scratchpad"\n'
        ')\n', encoding="utf-8")
    assert not _unguarded(guarded), "a correct fallback was reported as a pin"

    prose = tmp_path / "prose.py"
    prose.write_text(
        f'# The hard-coded {CLONE_PATH} is this container\'s clone location.\n',
        encoding="utf-8")
    assert not _unguarded(prose), "a comment explaining the literal was flagged"
