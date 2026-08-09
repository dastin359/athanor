"""The supervisor must not launch a runner that cannot authenticate.

`start()` sourced `$SP/arc3/.env` unchecked, in a file with no `set -e`. A
missing env file printed one "No such file or directory" and the runner started
anyway with no ARC_API_KEY, 401-ing every action.

That is the GUARANTEED state of a fresh container: the scratchpad reverts to an
image snapshot and the key is deliberately not in the repo. So a replacement box
would launch a sweep that cannot play a single game -- and relaunch it every ten
minutes, because the caller retries.

`key_ok` is extracted from the real file and run for real, against three
fixtures: absent, present-but-empty, and valid.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SUPERVISOR = REPO / "tools" / "supervisor.sh"
SRC = SUPERVISOR.read_text(encoding="utf-8")

_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LC_ALL": "C",
}


def _key_ok_fn() -> str:
    m = re.search(r"^key_ok\(\) \{.*?^\}$", SRC, re.S | re.M)
    assert m, "supervisor.sh no longer defines key_ok(); this test needs updating"
    return m.group(0)


def _run(sp: pathlib.Path) -> subprocess.CompletedProcess:
    script = f'SP={sp}\n' + _key_ok_fn() + '\nkey_ok; echo "rc=$?"\n'
    return subprocess.run(
        ["/bin/bash", "-c", script],
        capture_output=True, text=True, env=_ENV, timeout=30,
    )


def test_a_missing_env_file_refuses_the_launch(tmp_path: pathlib.Path) -> None:
    out = _run(tmp_path)
    assert "rc=1" in out.stdout, f"missing .env did not stop the launch: {out.stdout!r}"
    assert "SKIPPING LAUNCH" in out.stdout


def test_an_empty_env_file_refuses_the_launch(tmp_path: pathlib.Path) -> None:
    """Presence is not the test -- the VALUE is.

    A truncated write, or a copy that lost its contents, leaves a file that
    exists and sets nothing. Checking `-r` alone would pass it straight through.
    """
    (tmp_path / "arc3").mkdir()
    (tmp_path / "arc3" / ".env").write_text("# nothing here\n", encoding="utf-8")
    out = _run(tmp_path)
    assert "rc=1" in out.stdout, f"empty .env did not stop the launch: {out.stdout!r}"
    assert "sets no ARC_API_KEY" in out.stdout


def test_a_blank_key_refuses_the_launch(tmp_path: pathlib.Path) -> None:
    (tmp_path / "arc3").mkdir()
    (tmp_path / "arc3" / ".env").write_text('ARC_API_KEY=""\n', encoding="utf-8")
    out = _run(tmp_path)
    assert "rc=1" in out.stdout, f"blank key did not stop the launch: {out.stdout!r}"


def test_a_real_key_allows_the_launch(tmp_path: pathlib.Path) -> None:
    """The other direction, or the guard could be `return 1` and pass everything."""
    (tmp_path / "arc3").mkdir()
    (tmp_path / "arc3" / ".env").write_text("ARC_API_KEY=sk-fixture-not-real\n",
                                            encoding="utf-8")
    out = _run(tmp_path)
    assert "rc=0" in out.stdout, f"a valid .env was refused: {out.stdout!r}"
    assert "SKIPPING LAUNCH" not in out.stdout


def test_the_key_is_exported_for_the_runner(tmp_path: pathlib.Path) -> None:
    """`set -a` around the source is what puts the key in the child's env.

    Drop it and the guard still passes -- the variable is set in the shell --
    while the runner it launches inherits nothing and 401s exactly as before.
    """
    (tmp_path / "arc3").mkdir()
    (tmp_path / "arc3" / ".env").write_text("ARC_API_KEY=sk-fixture-not-real\n",
                                            encoding="utf-8")
    script = (
        f"SP={tmp_path}\n" + _key_ok_fn()
        + '\nkey_ok || exit 9\n'
        + 'printenv ARC_API_KEY\n'          # a CHILD process, not the shell
    )
    out = subprocess.run(
        ["/bin/bash", "-c", script],
        capture_output=True, text=True, env=_ENV, timeout=30,
    )
    assert out.stdout.strip().endswith("sk-fixture-not-real"), (
        "ARC_API_KEY did not reach a child process -- the runner would launch "
        f"without it. Got {out.stdout!r}"
    )


def test_start_checks_the_key_before_announcing_the_launch() -> None:
    """A log that announces a launch it then abandons is worse than silence."""
    m = re.search(r"^start\(\) \{.*?^\}$", SRC, re.S | re.M)
    assert m, "supervisor.sh no longer defines start()"
    body = m.group(0)
    key_at = body.find("key_ok")
    announce_at = body.find("starting $(basename")
    assert key_at != -1, "start() no longer calls key_ok"
    assert announce_at != -1, "start() no longer announces the launch"
    assert key_at < announce_at, (
        "start() announces the launch before checking the key"
    )


def test_start_does_not_source_the_env_unchecked() -> None:
    """The original bug, pinned: a bare `. "$SP/arc3/.env"` inside start()."""
    m = re.search(r"^start\(\) \{.*?^\}$", SRC, re.S | re.M)
    body = m.group(0)
    assert '. "$SP/arc3/.env"' not in body, (
        "start() sources the env file directly again -- a missing file will "
        "launch a keyless runner"
    )
