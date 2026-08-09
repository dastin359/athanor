"""A replaced container must not be left running a pre-fix copy of anything.

The scratchpad reverts to an image snapshot when the container is replaced, and
that snapshot holds whatever the tools looked like when it was taken.
`rehydrate_box.sh` exists to re-point those copies at the repo, and its list of
what to re-point was written by hand:

    for tool in refresh_audit.sh heartbeat.sh snapshot_results.py; do

Measured on the 02:44 PDT replacement of 2026-08-09, *after* that script ran and
reported success, three of five shadowing copies were repaired and two were not,
because nobody had added them:

    supervisor.sh          5,464 bytes  vs  16,369 in the repo
    ablate_baselines.py   21,367 bytes  vs  36,205 in the repo

The second is the baseline strip. A 21 kB copy predates `install()` and
`assert_installed()` — the fix for the defect that shipped real per-level medians
into eight rollouts and three re-runs. The first is the launcher without its key
guard, its argv-exact orphan sweep, or its sweep-directory awareness. And
`AUTOPILOT.md`'s launch block still reads `bash scratchpad/supervisor.sh`, so
following the standing instruction verbatim on a replaced box starts the pre-fix
launcher against the pre-fix strip.

Third instance of one defect: a hand-maintained list of the things of the day,
going stale with no symptom. The other two were `snapshot_results.py`'s batch
list, which silently omitted 25 banked games, and the live panel's arm allowlist,
which rendered "nothing running" rather than "I cannot see it". Both were fixed
by replacing the enumeration with a rule. This is that fix, plus the test that
would have caught it — the rule being: **any scratchpad entry sharing a basename
with a file in `tools/` must be a symlink to it.**

Run through `--links-only`, which does step 2 and stops: no fetch, no
fingerprint, no restores, no network.
"""

from __future__ import annotations

import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "rehydrate_box.sh"


def _rehydrate(scratch: pathlib.Path) -> str:
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--links-only"],
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": "/root",
             "CCARC3_SCRATCH": str(scratch)},
    )
    assert proc.returncode == 0, f"--links-only failed: {proc.stderr[-2000:]}"
    return proc.stdout


def _repo_tools() -> list[pathlib.Path]:
    return [p for p in sorted((REPO / "tools").iterdir()) if p.is_file()]


def test_every_shadowing_copy_is_repointed_not_just_the_listed_ones(tmp_path):
    """One copy of *each* repo tool, all stale. None may be left behind."""
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    tools = _repo_tools()
    for tool in tools:
        (scratch / tool.name).write_text("stale copy\n", encoding="utf-8")

    _rehydrate(scratch)

    left = [t.name for t in tools if not (scratch / t.name).is_symlink()]
    assert left == [], (
        f"{len(left)} shadowing copies survived rehydration: {left}. A "
        f"hand-maintained list is what left the baseline strip behind."
    )
    for tool in tools:
        assert (scratch / tool.name).resolve() == tool.resolve()


def test_the_baseline_strip_and_the_launcher_are_covered(tmp_path):
    """Named explicitly, because these two are what a stale copy actually costs."""
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    for name in ("ablate_baselines.py", "supervisor.sh"):
        (scratch / name).write_text("# a version from before the fixes\n", encoding="utf-8")

    _rehydrate(scratch)

    for name in ("ablate_baselines.py", "supervisor.sh"):
        link = scratch / name
        assert link.is_symlink(), f"{name} is still a real file after rehydration"
        assert link.resolve() == (REPO / "tools" / name).resolve()


def test_a_scratchpad_only_file_is_left_alone(tmp_path):
    """The rule discriminates on basename; everything else is the box's own."""
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    keep = {
        "batch6.py": "# a retired driver that never lived in the repo\n",
        "ablate_baselines.py.pre-guard": "# a deliberate backup\n",
        "supervisor.log": "17:00 something happened\n",
        "notes.md": "scratch notes\n",
    }
    for name, body in keep.items():
        (scratch / name).write_text(body, encoding="utf-8")

    _rehydrate(scratch)

    for name, body in keep.items():
        path = scratch / name
        assert not path.is_symlink(), f"{name} was replaced by a symlink"
        assert path.read_text(encoding="utf-8") == body, f"{name} was overwritten"


def test_a_tool_with_no_scratchpad_copy_is_not_created(tmp_path):
    """Re-pointing shadowing copies is not the same as populating the scratchpad.

    Only the four names the standing instructions call by their scratchpad path
    are created unconditionally; the rest appear only if something shadowed them.
    """
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()

    _rehydrate(scratch)

    always = {"refresh_audit.sh", "heartbeat.sh", "snapshot_results.py",
              "quota.sh", "AUTOPILOT.md", "MEMORY.md"}
    created = {p.name for p in scratch.iterdir()}
    assert always <= created, f"a standing-instruction path is missing: {always - created}"
    assert created == always, f"rehydration created unexpected entries: {created - always}"


def test_an_already_correct_symlink_is_left_quiet(tmp_path):
    """A healthy box must not report a repair it did not make."""
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    (scratch / "supervisor.sh").symlink_to(REPO / "tools" / "supervisor.sh")

    out = _rehydrate(scratch)

    assert "supervisor.sh was a real file" not in out, (
        "an already-correct symlink was reported as a repair; an always-on "
        "report is one people stop reading"
    )
    assert (scratch / "supervisor.sh").is_symlink()


def test_the_repair_names_the_drift_rather_than_fixing_it_silently(tmp_path):
    """The size delta is evidence about the box, not noise."""
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    (scratch / "ablate_baselines.py").write_text("x" * 21367, encoding="utf-8")

    out = _rehydrate(scratch)

    assert "ablate_baselines.py" in out and "21367" in out, (
        f"the drift was repaired silently; stdout was:\n{out}"
    )


def test_a_symlink_pointing_somewhere_else_is_repointed_but_not_called_a_file(tmp_path):
    """A stale link is a third state, between a correct link and a real copy.

    It arises the way this box's did: an older bring-up pointed the scratchpad at
    a path that has since moved. Re-pointing it is right; describing it as "was a
    real file" is not, and a report that miscategorises what it fixed is how the
    next reader misdiagnoses the box.
    """
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    elsewhere = tmp_path / "old_location.sh"
    elsewhere.write_text("# an earlier bring-up's copy\n", encoding="utf-8")
    (scratch / "supervisor.sh").symlink_to(elsewhere)

    out = _rehydrate(scratch)

    link = scratch / "supervisor.sh"
    assert link.is_symlink() and link.resolve() == (REPO / "tools" / "supervisor.sh").resolve(), (
        "a symlink pointing outside the repo was left alone"
    )
    assert "supervisor.sh was a real file" not in out, (
        f"a stale symlink was reported as a real file:\n{out}"
    )


# --- the proxy endpoint ----------------------------------------------------- #
#
# Outbound HTTPS leaves this box through an agent proxy on a loopback port, and
# that port changes when the session worker restarts -- so it is always wrong on
# a replaced box, and it is the one value only a live session can see. The
# supervisor validates it before every launch and refuses when it does not
# answer; before that guard existed the staleness was silent, and every driver
# the supervisor spawned got `[Errno 111] Connection refused` on its first
# `list_games()`.

def _proxy_line(scratch: pathlib.Path) -> str:
    for line in (scratch / "proxy_env").read_text(encoding="utf-8").splitlines():
        if line.startswith("export HTTPS_PROXY="):
            return line.split("=", 1)[1].strip('"')
    return ""


def _rehydrate_env(scratch: pathlib.Path, env: dict) -> str:
    proc = subprocess.run(
        ["bash", str(SCRIPT), "--links-only"],
        capture_output=True, text=True, timeout=120,
        env={"PATH": "/usr/bin:/bin", "HOME": "/root",
             "CCARC3_SCRATCH": str(scratch), **env},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def _seed_proxy(scratch: pathlib.Path, url: str) -> None:
    (scratch / "proxy_env").write_text(
        f'# the header explaining why this file exists\n'
        f'export HTTPS_PROXY="{url}"\nexport https_proxy="{url}"\n', encoding="utf-8")


def test_a_stale_proxy_port_is_refreshed_from_the_session(tmp_path):
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    _seed_proxy(scratch, "http://127.0.0.1:11111")

    out = _rehydrate_env(scratch, {"HTTPS_PROXY": "http://127.0.0.1:40983"})

    assert _proxy_line(scratch) == "http://127.0.0.1:40983", (
        "the replaced box kept the previous container's port"
    )
    assert "proxy_env ->" in out, "the refresh happened silently"


def test_an_already_current_proxy_is_left_quiet(tmp_path):
    """An always-on report is one people stop reading."""
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    _seed_proxy(scratch, "http://127.0.0.1:40983")

    out = _rehydrate_env(scratch, {"HTTPS_PROXY": "http://127.0.0.1:40983"})

    assert "proxy_env ->" not in out, "a no-op refresh was reported as a repair"
    assert _proxy_line(scratch) == "http://127.0.0.1:40983"


def test_no_proxy_in_the_environment_does_not_clobber_a_good_value(tmp_path):
    """Run outside a session there is nothing to inherit, and an empty value is worse."""
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    _seed_proxy(scratch, "http://127.0.0.1:40983")

    out = _rehydrate_env(scratch, {})

    assert _proxy_line(scratch) == "http://127.0.0.1:40983", (
        "a rehydrate with no proxy to inherit overwrote a working value"
    )
    assert "left as it is" in out, "the skip was silent"


def test_the_files_explanation_survives_the_refresh(tmp_path):
    """The header is why the next reader knows not to hard-code the port."""
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    _seed_proxy(scratch, "http://127.0.0.1:11111")

    _rehydrate_env(scratch, {"HTTPS_PROXY": "http://127.0.0.1:40983"})

    body = (scratch / "proxy_env").read_text(encoding="utf-8")
    assert body.startswith("# the header explaining why this file exists"), (
        f"the refresh ate the file's explanation:\n{body}"
    )
    assert body.count("export https_proxy=") == 1, "the lowercase alias was duplicated"
