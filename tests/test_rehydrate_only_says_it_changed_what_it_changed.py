"""What `rehydrate_box.sh` reports, and what it refuses to do on a stray argument.

Two holes from the mutation pass, both in the reporting rather than the repair —
which is the half that decides whether an operator believes the box recovered.

* **`proxy_env` up to date was rewritten and announced anyway.** The message is
  "proxy_env -> … (was stale or absent)", printed on a run where it was neither.
  A recovery script that claims a repair it did not make is the same defect class
  as an always-on alarm: the line stops meaning anything, and the run where the
  port really had moved reads identically to the twenty where it had not.
* **Any argument was accepted as `--links-only`.** A typo or a `--help` would
  silently skip the fingerprint and banked-results restore while exiting 0, and
  the caller would believe a full recovery had happened.

The stale port is not hypothetical. Measured on this box at 22:5x PDT: `proxy_env`
held `:41083` while the live proxy had moved to `:42797`. The supervisor validates
the port before every launch and refuses when it does not answer, so the sweep
would have been blocked by a value only a live session can repair.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REHYDRATE = REPO / "tools" / "rehydrate_box.sh"


def _run(scratch: Path, *args: str, proxy: str | None) -> subprocess.CompletedProcess:
    env = {**os.environ, "CCARC3_SCRATCH": str(scratch)}
    if proxy is None:
        env.pop("HTTPS_PROXY", None)
        env.pop("https_proxy", None)
    else:
        env["HTTPS_PROXY"] = proxy
    return subprocess.run(["bash", str(REHYDRATE), *args], cwd=REPO,
                          capture_output=True, text=True, timeout=120, env=env)


@pytest.fixture()
def scratch(tmp_path):
    d = tmp_path / "scratchpad"
    d.mkdir()
    return d


def test_an_up_to_date_proxy_is_left_alone_and_not_announced(scratch):
    """The second run must be silent about the proxy. Announcing a repair that
    did not happen makes the line that matters indistinguishable from noise."""
    first = _run(scratch, "--links-only", proxy="http://127.0.0.1:9999")
    assert "proxy_env ->" in first.stdout, "the first run should have written it"

    second = _run(scratch, "--links-only", proxy="http://127.0.0.1:9999")
    assert "proxy_env ->" not in second.stdout, (
        f"a second run announced a repair it did not make: {second.stdout!r}"
    )


def test_a_moved_proxy_is_still_announced(scratch):
    """The positive control: silence must come from "nothing changed", not from
    the report being switched off."""
    _run(scratch, "--links-only", proxy="http://127.0.0.1:9999")
    moved = _run(scratch, "--links-only", proxy="http://127.0.0.1:8888")
    assert "proxy_env -> http://127.0.0.1:8888" in moved.stdout


def test_the_written_proxy_is_the_live_one(scratch):
    _run(scratch, "--links-only", proxy="http://127.0.0.1:9999")
    body = (scratch / "proxy_env").read_text()
    assert 'export HTTPS_PROXY="http://127.0.0.1:9999"' in body
    assert 'export https_proxy="http://127.0.0.1:9999"' in body


def test_a_missing_proxy_does_not_clobber_a_good_value(scratch):
    """A rehydrate run outside a session has nothing to inherit, and must not
    overwrite a working port with an empty one."""
    _run(scratch, "--links-only", proxy="http://127.0.0.1:9999")
    out = _run(scratch, "--links-only", proxy=None)
    assert "left as it is" in out.stdout
    assert 'http://127.0.0.1:9999' in (scratch / "proxy_env").read_text()


# --------------------------------------------------------------------------- #
# --links-only is a specific flag, not "any argument"
# --------------------------------------------------------------------------- #


def _flag_parse(arg: str | None) -> str:
    """Run just the flag-parsing lines. Running the whole script for this takes
    a minute and drags a network fetch in behind it — and a slow test that
    exercises the wrong thing is worse than no test."""
    body = REHYDRATE.read_text()
    marker = 'LINKS_ONLY=0'
    assert body.count(marker) == 1
    start = body.index(marker)
    end = body.index("\n", body.index("--links-only", start)) + 1
    snippet = body[start:end]
    assert "LINKS_ONLY=1" in snippet, "extraction lost the flag assignment"
    script = f"set -u\n{snippet}\necho \"LINKS_ONLY=$LINKS_ONLY\""
    argv = [] if arg is None else [arg]
    out = subprocess.run(["bash", "-c", script, "sh", *argv],
                         capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.mark.parametrize("arg", ["--help", "--linksonly", "-l", "links-only", "--dry-run", ""])
def test_a_stray_argument_is_not_treated_as_links_only(arg):
    """Accepting anything means a typo silently skips the fingerprint and
    banked-results restore while exiting 0 — a recovery that reports success and
    did half the job."""
    assert _flag_parse(arg) == "LINKS_ONLY=0", f"{arg!r} was accepted as the flag"


def test_no_argument_is_a_full_restore():
    assert _flag_parse(None) == "LINKS_ONLY=0"


def test_the_real_flag_is_accepted():
    """The positive control: if the extraction were broken, every case above
    would read 0 and the test would pass having checked nothing."""
    assert _flag_parse("--links-only") == "LINKS_ONLY=1"


def test_links_only_really_stops_after_the_links(scratch):
    out = _run(scratch, "--links-only", proxy="http://127.0.0.1:9999")
    assert out.returncode == 0
    assert "fingerprint" not in out.stdout.lower()
