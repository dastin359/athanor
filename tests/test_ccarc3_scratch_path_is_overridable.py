"""Every autopilot tool must resolve its scratchpad from ``CCARC3_SCRATCH``.

The default path encodes a session UUID and the container is recycled every
10-50 minutes, so a hard-coded copy points at a directory that stops existing.
Ten tools share the path; only ``quota.sh`` honoured the override, and the
other nine read the wrong tree under any fixture or any fresh container. The
restore tools matter most: re-hydrating banked evidence into the new scratchpad
is the first bring-up step on a replacement container, and it was the step that
would have written to a dead path.

These assert the tools resolve to the OVERRIDE, never merely that they mention
the variable -- a grep for ``CCARC3_SCRATCH`` passes on a file that reads it
into a variable nothing uses. Mutation-verified: reverting any tool's line to
the bare default makes its case fail.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
UUID = "a3375e8f-271e-5133-96a4-a40a6a06a752"

SHELL_TOOLS = [
    "quota.sh",
    "heartbeat.sh",
    "heartbeat_watch.sh",
    "supervisor.sh",
    "preserve_evidence.sh",
    "box_fingerprint.sh",
]

# The variable is not spelled the same way twice: `quota.sh` calls it `S`, the
# daemons call it `SP`, the restore tools call it `SCRATCH`. Matching only one
# spelling is how the first draft of this test "passed" `quota.sh` -- it found
# no `SP=` line, and `grep` exiting 1 raised rather than failing the assertion,
# which is a different bug wearing the same red.
SCRATCH_VAR = r"^(SP|S|SCRATCH)="

# Imported for real by the runtime cases below. `rerun_losses` is deliberately
# absent: it calls the ARC API at import, so importing it in a unit test is a
# network call, not a path check. It is covered by the whole-surface case.
IMPORTABLE = [
    "clean_rollouts",
    "ablate_baselines",
    "restore_banked_results",
    "restore_clean_rollouts",
]


def _sp_line(tool: str) -> str:
    """The tool's real scratch assignment, taken from the file.

    Restating the line here would test the restatement.
    """
    proc = subprocess.run(
        ["grep", "-m1", "-E", SCRATCH_VAR, str(TOOLS / tool)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"{tool}: no scratch assignment matching {SCRATCH_VAR}"
    return proc.stdout.strip()


def _eval_sp(line: str, env: dict[str, str]) -> str:
    var = line.split("=", 1)[0]
    out = subprocess.run(
        ["bash", "-c", f'{line}\nprintf "%s" "${var}"'],
        capture_output=True, text=True, env=env, check=True,
    )
    return out.stdout.strip()


@pytest.mark.parametrize("tool", SHELL_TOOLS)
def test_shell_tool_honours_override(tool: str, tmp_path: pathlib.Path) -> None:
    override = str(tmp_path / "scratch")
    got = _eval_sp(_sp_line(tool), {**os.environ, "CCARC3_SCRATCH": override})
    assert got == override


@pytest.mark.parametrize("tool", SHELL_TOOLS)
def test_shell_tool_falls_back_when_unset(tool: str) -> None:
    """Unset must yield the live default, not an empty string.

    An empty ``SP`` makes every ``"$SP/x"`` an absolute ``/x`` -- the daemons
    would read and write the container root. Absence of the variable is the
    normal case in production, so it is the case most worth pinning.
    """
    env = {k: v for k, v in os.environ.items() if k != "CCARC3_SCRATCH"}
    got = _eval_sp(_sp_line(tool), env)
    assert got.startswith("/tmp/"), f"{tool}: fallback resolved to {got!r}"
    assert got.endswith("/scratchpad"), f"{tool}: fallback resolved to {got!r}"


@pytest.mark.parametrize("mod", IMPORTABLE)
def test_python_tool_honours_override(mod: str, tmp_path: pathlib.Path) -> None:
    """Resolved at import, so each runs in a fresh process."""
    override = tmp_path / "scratch"
    override.mkdir()
    code = (
        "import sys; "
        f"sys.path[:0] = [{str(TOOLS)!r}, {str(REPO / 'src')!r}]; "
        f"m = __import__({mod!r}); "
        "print(getattr(m, 'SP', None) or getattr(m, 'SCRATCH', None))"
    )
    out = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True, text=True,
        env={**os.environ, "CCARC3_SCRATCH": str(override), "ARC_API_KEY": "dummy"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == str(override)


def test_clean_rollouts_out_follows_the_override(tmp_path: pathlib.Path) -> None:
    """``SP`` alone is a proxy; ``OUT`` is what the sweep actually reads.

    The driver's skip-if-banked check, the heartbeat's ``work_pending`` and the
    progress counts all glob under ``OUT``. An ``SP`` that moved while ``OUT``
    stayed pinned would pass the case above and still read the dead tree.
    """
    override = tmp_path / "scratch"
    override.mkdir()
    code = (
        "import sys; "
        f"sys.path[:0] = [{str(TOOLS)!r}, {str(REPO / 'src')!r}]; "
        "import clean_rollouts as cr; print(cr.OUT)"
    )
    out = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True, text=True,
        env={**os.environ, "CCARC3_SCRATCH": str(override), "ARC_API_KEY": "dummy"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1].startswith(str(override))


def test_no_tool_hardcodes_the_path_without_the_override() -> None:
    """The session UUID may appear only inside a defaulted expansion.

    This is the case that catches a NEW tool copying the literal in, which the
    per-tool cases cannot: they only know the ten that exist today. It found
    five beyond the enumerated set on its first run, including both restore
    tools -- so it is load-bearing, not belt-and-braces.

    Scoped to executable tools. The path also appears inside
    ``trace_audit_template.html`` as prose in a rendered finding: that is a
    report describing a path, not code resolving one.
    """
    offenders = []
    for path in sorted(TOOLS.iterdir()):
        if not path.is_file() or path.suffix not in (".sh", ".py"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for n, line in enumerate(text.splitlines(), 1):
            if UUID not in line or line.lstrip().startswith("#"):
                continue
            if "CCARC3_SCRATCH" in line:
                continue                       # defaulted, on one line
            if line.lstrip().startswith('or "') and "CCARC3_SCRATCH" in text:
                continue                       # defaulted, wrapped over lines
            offenders.append(f"{path.name}:{n}")
    assert not offenders, "hardcoded scratchpad path: " + ", ".join(offenders)
