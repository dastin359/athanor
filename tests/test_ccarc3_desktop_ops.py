"""Offline checks for the Codex desktop bootstrap and external resume guard."""

from __future__ import annotations

import os
import pathlib
import subprocess
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO / "tools" / "desktop_bootstrap.sh"
RESUME = REPO / "tools" / "ccarc3_resume.sh"


def _env(scratch: pathlib.Path, **extra: str) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(scratch.parent),
        "CCARC3_SCRATCH": str(scratch),
    }
    env.update(extra)
    return env


def _run_resume(scratch: pathlib.Path, **extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(RESUME)],
        cwd=REPO,
        env=_env(scratch, **extra),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_desktop_bootstrap_probes_the_codex_isolation_surface() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'CODEX_BINARY' in text
    assert 'command -v codex' in text
    for flag in (
        "--ignore-user-config",
        "--ignore-rules",
        "--disable memories",
        "--skip-git-repo-check",
    ):
        assert flag in text
    assert "CLAUDE_BINARY" not in text
    assert "command -v claude" not in text


def test_resume_with_a_closed_brake_is_a_quiet_noop(tmp_path: pathlib.Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "concurrency").write_text("0\n", encoding="utf-8")

    result = _run_resume(scratch)

    assert result.returncode == 0, result.stderr
    assert (scratch / "resume.heartbeat").is_file()
    assert not (scratch / "resume.log").exists()


def test_resume_refuses_to_guess_the_sweep_identity(tmp_path: pathlib.Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "concurrency").write_text("1\n", encoding="utf-8")

    result = _run_resume(scratch)

    assert result.returncode == 1
    assert "sweep.env is missing" in (scratch / "resume.log").read_text()


def test_resume_refuses_a_sanctioned_sweep_without_a_key(tmp_path: pathlib.Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "concurrency").write_text("1\n", encoding="utf-8")
    (scratch / "sweep.env").write_text(
        "CCARC3_ONLY=cd82\nCCARC3_SWEEP_DIR=clean_rollouts_fixture\n",
        encoding="utf-8",
    )

    result = _run_resume(scratch)

    assert result.returncode == 1
    assert "no ARC_API_KEY" in (scratch / "resume.log").read_text()


def test_resume_launches_only_after_every_guard_passes(tmp_path: pathlib.Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "concurrency").write_text("1\n", encoding="utf-8")
    (scratch / "sweep.env").write_text(
        "CCARC3_ONLY=cd82\nCCARC3_SWEEP_DIR=clean_rollouts_fixture\n",
        encoding="utf-8",
    )
    env_dir = scratch / "arc3"
    env_dir.mkdir()
    (env_dir / ".env").write_text("ARC_API_KEY=fixture-only\n", encoding="utf-8")

    driver_marker = tmp_path / "driver.started"
    supervisor_marker = tmp_path / "supervisor.started"
    driver = tmp_path / "stub_driver.py"
    driver.write_text(
        "import os, pathlib\n"
        f"pathlib.Path({str(driver_marker)!r}).write_text("
        "os.environ.get('CCARC3_ONLY', '') + '|' + "
        "os.environ.get('CCARC3_SWEEP_DIR', ''))\n",
        encoding="utf-8",
    )
    supervisor = tmp_path / "stub_supervisor.sh"
    supervisor.write_text(
        "#!/usr/bin/env bash\n"
        f"printf started > {str(supervisor_marker)!r}\n",
        encoding="utf-8",
    )
    supervisor.chmod(0o755)

    result = _run_resume(
        scratch,
        CCARC3_RESUME_DRIVER=str(driver),
        CCARC3_RESUME_SUPERVISOR=str(supervisor),
    )
    assert result.returncode == 0, result.stderr

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if driver_marker.exists() and supervisor_marker.exists():
            break
        time.sleep(0.05)
    assert driver_marker.read_text() == "cd82|clean_rollouts_fixture"
    assert supervisor_marker.read_text() == "started"


@pytest.mark.parametrize("script", [BOOTSTRAP, RESUME])
def test_desktop_scripts_are_valid_bash(script: pathlib.Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
