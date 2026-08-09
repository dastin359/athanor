"""Every autopilot tool must resolve its roots from where it actually is.

Two roots, the same defect: the scratchpad (``CCARC3_SCRATCH``) and the repo
(derived from the script's own location). Both were hard-coded copies of
this container's paths.

Part one -- the scratchpad.

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


# ==========================================================================
# part two -- the repo path
#
# `/home/user/athanor` is this container's clone location. A fresh container, or
# the same repo checked out by a different account, gets a different one, and a
# hard-coded copy then names a directory that does not exist. In heartbeat.sh
# every consequence was silent by construction: `2>/dev/null` on the python
# calls, `|| true` on the snapshot, a default for RUNNER_NAME. It would have
# gone on printing tidy status lines having executed none of its checks.
#
# These relocate the tool and read what it resolves THERE. Asserting the value
# in place cannot work -- a frozen literal is correct here, which is the whole
# reason the bug was invisible.
# ==========================================================================

DERIVING_SHELL_TOOLS = [
    "heartbeat.sh",
    "heartbeat_watch.sh",
    "supervisor.sh",
    "preserve_evidence.sh",
    "box_fingerprint.sh",
]


@pytest.mark.parametrize("tool", DERIVING_SHELL_TOOLS)
def test_shell_tool_derives_repo_from_its_own_location(
    tool: str, tmp_path: pathlib.Path
) -> None:
    """Copy the tool to a new repo root; its derived path must follow.

    The probe is a FILE in the copy's `tools/`, never `bash -c`: the derivation
    reads `${BASH_SOURCE[0]}`, which is empty under `-c` and silently resolves
    against the caller's cwd instead.
    """
    dest_tools = tmp_path / "relocated" / "tools"
    dest_tools.mkdir(parents=True)
    src = (TOOLS / tool).read_text(encoding="utf-8")
    (dest_tools / tool).write_text(src, encoding="utf-8")

    # heartbeat_watch.sh derives the heartbeat's path rather than a REPO var.
    var = "HB" if tool == "heartbeat_watch.sh" else "REPO"
    line = next(ln for ln in src.splitlines() if ln.startswith(f"{var}="))
    probe = dest_tools / "_probe.sh"
    probe.write_text(f'{line}\nprintf "%s" "${var}"\n', encoding="utf-8")

    got = subprocess.run(
        ["/bin/bash", str(probe)], capture_output=True, text=True, check=True,
    ).stdout.strip()

    expected = (str(dest_tools / "heartbeat.sh") if var == "HB"
                else str(tmp_path / "relocated"))
    assert got == expected, (
        f"{tool} relocated to {tmp_path / 'relocated'} still resolves {var} to "
        f"{got!r} -- it is a frozen literal and names a directory that will not "
        f"exist on a fresh container"
    )


def test_python_tool_derives_repo_from_its_own_location(
    tmp_path: pathlib.Path
) -> None:
    """`leak_exposure` stands in for the Python tools that import cleanly.

    `rerun_losses` and `ablate_baselines` cannot be imported from a copy --
    they pull in siblings and the `athanor` package at import time -- so the
    whole-surface case below is what guards those two. Naming that here rather
    than quietly covering four tools with one that happens to be importable.
    """
    dest_tools = tmp_path / "relocated" / "tools"
    dest_tools.mkdir(parents=True)
    (dest_tools / "leak_exposure.py").write_text(
        (TOOLS / "leak_exposure.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(dest_tools)!r}); "
        "import leak_exposure as m; print(m.REPO)"
    )
    out = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"), "-c", code],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == str(tmp_path / "relocated")


def test_no_tool_hardcodes_the_repo_path() -> None:
    """The repo literal may appear only in prose, or as a documented default.

    This is what guards the tools the relocation cases cannot reach, and what
    catches a new tool pasting the path in.

    `rehydrate_bootstrap.sh` is the one legitimate exception: it exists to be
    COPIED into the scratchpad, so deriving from `${BASH_SOURCE[0]}` would
    resolve to wherever the copy sits rather than to the repo. It takes an
    explicit `CCARC3_REPO` override instead.
    """
    literal = "/home/user/athanor"
    offenders = []
    for base in (TOOLS, REPO / "src"):
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in (".sh", ".py"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for n, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                if literal not in line:
                    continue
                if stripped.startswith("#"):
                    continue                      # prose
                if "CCARC3_REPO" in line:
                    continue                      # documented override default
                offenders.append(f"{path.relative_to(REPO)}:{n}")
    assert not offenders, "hardcoded repo path: " + ", ".join(offenders)


# ==========================================================================
# part three -- the branch
#
# The branch is not a property of the code. It was a literal in four tools, and
# a session on a different account works on a different branch:
# preserve_evidence.sh's HEAD-vs-BRANCH guard then refuses every cycle, so
# nothing is preserved and the only symptom is a line in a log nobody reads.
#
# Deliberately NOT derived from `git rev-parse --abbrev-ref HEAD`. That guard
# exists to catch "HEAD moved and the push would go somewhere unintended", and a
# branch read from HEAD always agrees with HEAD -- deriving it would leave the
# check passing by construction, which is the defect class this suite is about.
# ==========================================================================

BRANCH_TOOLS = [
    "preserve_evidence.sh",
    "box_fingerprint.sh",
    "rehydrate_bootstrap.sh",
    "rehydrate_box.sh",
]
DEFAULT_BRANCH = "claude/athanor-cc-harness-variant-jpqw7t"


def _branch_line(tool: str) -> str:
    proc = subprocess.run(
        ["grep", "-m1", "-E", "^BRANCH=", str(TOOLS / tool)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"{tool}: no BRANCH assignment"
    return proc.stdout.strip()


@pytest.mark.parametrize("tool", BRANCH_TOOLS)
def test_branch_honours_the_override(tool: str) -> None:
    got = subprocess.run(
        ["bash", "-c", _branch_line(tool) + '\nprintf "%s" "$BRANCH"'],
        capture_output=True, text=True, check=True,
        env={**os.environ, "CCARC3_BRANCH": "feature/somewhere-else"},
    ).stdout.strip()
    assert got == "feature/somewhere-else"


@pytest.mark.parametrize("tool", BRANCH_TOOLS)
def test_branch_falls_back_to_the_working_branch(tool: str) -> None:
    env = {k: v for k, v in os.environ.items() if k != "CCARC3_BRANCH"}
    got = subprocess.run(
        ["bash", "-c", _branch_line(tool) + '\nprintf "%s" "$BRANCH"'],
        capture_output=True, text=True, check=True, env=env,
    ).stdout.strip()
    assert got == DEFAULT_BRANCH, f"{tool} defaults to {got!r}"


def test_the_branch_is_not_read_from_head() -> None:
    """Deriving it would make preserve_evidence's own guard vacuous.

    That guard compares HEAD against BRANCH to catch a push aimed somewhere
    unintended. A BRANCH read from HEAD always matches HEAD, so the comparison
    could never fail -- a check that passes by construction.
    """
    for tool in BRANCH_TOOLS:
        line = _branch_line(tool)
        assert "rev-parse" not in line and "symbolic-ref" not in line, (
            f"{tool} derives BRANCH from git state: {line!r} -- this disarms "
            f"the HEAD-vs-BRANCH guard in preserve_evidence.sh"
        )


# ==========================================================================
# part four -- the fingerprint's work count
#
# `scratch_dirs` counted `ablate_nobaseline` alone, so through a submission
# sweep it would sit frozen at 25 while saying nothing about the work at risk --
# in the one row whose purpose is noticing that a replacement took the disk.
# ==========================================================================


def _fingerprint_count(scratch: pathlib.Path, sweep: str | None) -> int:
    """Evaluate the real counting block from box_fingerprint.sh."""
    src = (TOOLS / "box_fingerprint.sh").read_text(encoding="utf-8")
    start = src.index("dirs=0")
    end = src.index("done", start) + len("done")
    env = {k: v for k, v in os.environ.items() if k != "CCARC3_SWEEP_DIR"}
    if sweep:
        env["CCARC3_SWEEP_DIR"] = sweep
    out = subprocess.run(
        ["/bin/bash", "-c", f'SP={scratch}\n' + src[start:end] + '\nprintf "%s" "$dirs"'],
        capture_output=True, text=True, check=True, env=env,
    )
    return int(out.stdout.strip())


def test_the_fingerprint_counts_the_live_sweep(tmp_path: pathlib.Path) -> None:
    for arm, n in (("ablate_nobaseline", 2), ("clean_rollouts", 3)):
        for i in range(n):
            (tmp_path / arm / f"g{i}").mkdir(parents=True)
    assert _fingerprint_count(tmp_path, None) == 5, (
        "the live sweep's games are not counted"
    )


def test_the_fingerprint_follows_a_renamed_sweep(tmp_path: pathlib.Path) -> None:
    (tmp_path / "ablate_nobaseline" / "g0").mkdir(parents=True)
    for i in range(4):
        (tmp_path / "clean_rollouts_submission" / f"g{i}").mkdir(parents=True)
    assert _fingerprint_count(tmp_path, "clean_rollouts_submission") == 5, (
        "the submission sweep's games are invisible to the fingerprint"
    )


def test_the_fingerprint_counts_rerun_directories(tmp_path: pathlib.Path) -> None:
    (tmp_path / "ablate_nobaseline" / "g0").mkdir(parents=True)
    (tmp_path / "rerun_losses" / "g1").mkdir(parents=True)
    (tmp_path / "rerun_bp35_fixed" / "g2").mkdir(parents=True)
    assert _fingerprint_count(tmp_path, None) == 3


def test_the_fingerprint_is_zero_on_an_empty_box(tmp_path: pathlib.Path) -> None:
    """A replacement that lost everything must read 0, not fail or inflate."""
    assert _fingerprint_count(tmp_path, None) == 0
