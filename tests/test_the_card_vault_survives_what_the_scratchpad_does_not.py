"""The card pin must outlive the container that holds it.

A scorecard is reachable only via its `AWSALBAPP-*` stickiness cookies, which
live in `scratchpad/shared_card.json`. The scratchpad reverts to an image
snapshot when the container is replaced. On 2026-08-11 that stranded card
`e6e6a61c` two hours into a sweep -- seven games on it, and it could no longer be
read, played onto, or **closed**, which is what publishes. ~$90; could have been
$652.

The cookies cannot simply be committed: `evidence/` is pushed to a public remote.
So the pin is sealed under a key derived from `ARC_API_KEY` and only the
ciphertext is committed. The two stores fail in complementary ways -- the repo is
durable and public, while `scratchpad/arc3/.env` is private and (measured, mtime
2026-08-02) part of the image snapshot. The snapshot that destroys the cookies
restores the key that decrypts them.

These tests pin the properties that make that safe: the sealed file leaks
nothing, a wrong key fails closed, and a restore never overwrites a live pin with
a stale one.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import card_vault as cv


# **Realistic in SHAPE, deliberately not in LENGTH.**
# `tests/test_no_session_cookie_reaches_the_repository.py` scans every tracked
# file for a cookie name followed by a 20+ character value, because that is what
# a credential looks like and value-matching has failed twice in this project.
# The first draft of this fixture used a 20-character value and tripped it -- a
# true positive on the shape, and the guard was right to fire: nothing reading a
# repo can tell a convincing fixture from a live cookie, which is exactly why
# `preserve_evidence.sh` bans the shape rather than any particular value.
#
# The fix belongs here, not in the guard. An allowance for `tests/` would punch a
# hole in the one check that caught 79 leaked files, and the fixture loses
# nothing: what these tests need is the cookie *structure*, so the seal has
# something structural to hide, and a short canary is just as findable in the
# ciphertext as a long one. Keep these under 20 characters and obviously fake.
COOKIES = {
    "card_id": "50e3871e-c684-404f-a29a-f87023788f8f",
    "cookies": [
        {"name": "GAMESESSION", "value": "fake-session", "domain": ".arcprize.org"},
        {"name": "AWSALBAPP-0", "value": "fake-sticky", "domain": ".arcprize.org"},
    ],
}


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("ARC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("CCARC3_CARD_VAULT", "1")
    monkeypatch.setattr(cv, "SP", tmp_path / "scratch")
    monkeypatch.setattr(cv, "PIN", tmp_path / "scratch" / "shared_card.json")
    monkeypatch.setattr(cv, "VAULT_DIR", tmp_path / "vault")
    monkeypatch.setattr(cv, "ITERATIONS", 1000)      # keep the tests quick
    (tmp_path / "scratch").mkdir()
    (tmp_path / "vault").mkdir()
    return cv


def _write_pin(vault):
    vault.PIN.write_text(json.dumps(COOKIES))


def test_a_sealed_pin_round_trips(vault):
    _write_pin(vault)
    assert vault.save() == 0
    vault.PIN.unlink()                                # the container replacement
    assert vault.restore(COOKIES["card_id"]) == 0
    assert json.loads(vault.PIN.read_text()) == COOKIES


def test_the_sealed_file_leaks_no_cookie(vault):
    """The whole point: this lands in a public repo."""
    _write_pin(vault)
    vault.save()
    blob = vault.vault_path(COOKIES["card_id"]).read_bytes()
    for secret in (b"s3cr3t-session-value", b"st1ck1ness-value",
                   b"GAMESESSION", b"AWSALBAPP"):
        assert secret not in blob, f"{secret!r} is readable in the sealed file"


def test_a_wrong_key_fails_closed(vault, monkeypatch):
    """A rotated key must not half-restore something and call it a pin."""
    _write_pin(vault)
    vault.save()
    vault.PIN.unlink()
    monkeypatch.setenv("ARC_API_KEY", "a-different-key")
    assert vault.restore(COOKIES["card_id"]) == 1
    assert not vault.PIN.exists(), "a pin was written from a failed decrypt"


def test_restore_never_overwrites_a_live_pin(vault):
    """A stale vault would replace working cookies with dead ones."""
    _write_pin(vault)
    vault.save()
    live = dict(COOKIES, card_id="a-newer-card")
    vault.PIN.write_text(json.dumps(live))
    assert vault.restore(COOKIES["card_id"]) == 0
    assert json.loads(vault.PIN.read_text()) == live, "the live pin was clobbered"


def test_saving_without_a_key_refuses(vault, monkeypatch):
    _write_pin(vault)
    monkeypatch.delenv("ARC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        vault.save()


def test_it_is_off_by_default(tmp_path, monkeypatch):
    """Sealing publishes an encrypted credential; that is an operator decision."""
    # VAULT_DIR is redirected because `enabled()` also consults a committed
    # marker there, and a test of the env var must not read the real repo -- a
    # decision the operator makes tomorrow would otherwise start failing a test
    # about something else.
    monkeypatch.setattr(cv, "VAULT_DIR", tmp_path / "vault")
    monkeypatch.delenv("CCARC3_CARD_VAULT", raising=False)
    assert cv.enabled() is False
    monkeypatch.setenv("CCARC3_CARD_VAULT", "0")
    assert cv.enabled() is False
    monkeypatch.setenv("CCARC3_CARD_VAULT", "1")
    assert cv.enabled() is True


def test_the_switch_survives_the_replacement_it_exists_for(tmp_path, monkeypatch):
    """An env-var-only switch turns itself off at the moment it is needed.

    `scratchpad/arc3/.env` is durable only by predating the image snapshot. A
    line appended today is reverted by the next replacement -- so the vault would
    seal the first card, lose the switch along with the scratchpad, and leave the
    second card unsealed. The committed marker comes back from origin instead.
    """
    monkeypatch.setattr(cv, "VAULT_DIR", tmp_path / "vault")
    monkeypatch.delenv("CCARC3_CARD_VAULT", raising=False)
    assert cv.enabled() is False

    (tmp_path / "vault").mkdir()
    (tmp_path / "vault" / cv.MARKER).write_text("armed by the operator\n")
    assert cv.enabled() is True, "a committed marker must arm sealing on its own"

    # A local operator can still override the committed decision downward.
    monkeypatch.setenv("CCARC3_CARD_VAULT", "0")
    assert cv.enabled() is False


def test_resealing_an_unchanged_pin_writes_nothing(vault):
    """Fernet is nondeterministic, and this daemon commits whatever changed.

    `preserve_evidence.sh` calls `save` every 300s and commits `evidence/`. If an
    unchanged pin produced fresh ciphertext each time, that is ~288 commits a day
    all claiming the card pin moved.
    """
    _write_pin(vault)
    assert vault.save() == 0
    blob = vault.vault_path(COOKIES["card_id"]).read_bytes()

    assert vault.save() == 0
    assert vault.vault_path(COOKIES["card_id"]).read_bytes() == blob, (
        "an unchanged pin must not produce a new ciphertext")

    # A pin that really did change must still be resealed.
    moved = dict(COOKIES, cookies=[{"name": "GAMESESSION", "value": "fake-rotated"}])
    vault.PIN.write_text(json.dumps(moved))
    assert vault.save() == 0
    assert vault.vault_path(COOKIES["card_id"]).read_bytes() != blob


def test_a_restore_is_not_gated_on_the_publish_switch(vault, monkeypatch):
    """The switch decides whether to publish, not whether to recover.

    Once a vault exists the publication has already happened. Refusing to read it
    during a recovery forfeits the sweep and buys back no secrecy.
    """
    _write_pin(vault)
    assert vault.save() == 0
    vault.PIN.unlink()                       # the replacement

    monkeypatch.delenv("CCARC3_CARD_VAULT", raising=False)
    monkeypatch.setattr(cv, "VAULT_DIR", vault.VAULT_DIR)
    assert vault.enabled() is False
    assert vault.main(["card_vault.py", "restore", COOKIES["card_id"]]) == 0
    assert json.loads(vault.PIN.read_text())["card_id"] == COOKIES["card_id"]

    # Saving, which is the half that publishes, stays refused.
    vault.PIN.unlink()
    _write_pin(vault)
    vault.vault_path(COOKIES["card_id"]).unlink()
    assert vault.main(["card_vault.py", "save"]) == 0
    assert not vault.vault_path(COOKIES["card_id"]).exists()


def test_restoring_a_card_that_was_never_sealed_says_so(vault):
    assert vault.restore("00000000-0000-0000-0000-000000000000") == 1
    assert not vault.PIN.exists()


def test_quiet_silences_the_no_ops_and_nothing_else(vault, monkeypatch, capsys):
    """A daemon line every 300s is a log an operator stops reading.

    `preserve_evidence.sh` forwards this tool's stdout into its own log. The two
    routine outcomes -- not armed, and no pin yet -- are the common case on any
    box between sweeps, so they must be silent. A refusal must not be.
    """
    monkeypatch.delenv("CCARC3_CARD_VAULT", raising=False)
    monkeypatch.setattr(cv, "VAULT_DIR", vault.VAULT_DIR)

    assert cv.main(["card_vault.py", "save", "--quiet"]) == 0
    assert capsys.readouterr().out == "", "an unarmed vault must not natter"

    monkeypatch.setenv("CCARC3_CARD_VAULT", "1")
    assert cv.main(["card_vault.py", "save", "--quiet"]) == 0
    assert capsys.readouterr().out == "", "no pin yet is not news"

    # Armed, with a pin that cannot be read: that IS news, quiet or not.
    vault.PIN.write_text("{ truncated")
    assert cv.main(["card_vault.py", "save", "--quiet"]) == 1
    assert "unreadable" in capsys.readouterr().out

    # And a real seal announces itself, because it changed the committed tree.
    _write_pin(vault)
    assert cv.main(["card_vault.py", "save", "--quiet"]) == 0
    assert "sealed" in capsys.readouterr().out


def _an_interpreter_that_cannot_do_crypto() -> str:
    """A python on this box whose `cryptography` import fails, or ""."""
    import shutil
    import subprocess
    # **Probe each candidate; never compare paths.** `.venv/bin/python` and
    # `/usr/bin/python3` on this image are the SAME BINARY -- `resolve()` maps
    # both to `/usr/bin/python3.11` -- and they behave completely differently,
    # because what differs is `sys.prefix` and the site-packages behind it. An
    # earlier version of this helper excluded candidates by resolved path,
    # concluded every interpreter was itself, and skipped: the test reported
    # green by declining to run, which is this project's signature defect
    # (a check that passes by not running) reproduced inside the test written to
    # catch it. Capability is the only thing worth asking about, so ask it.
    for cand in ("/usr/local/bin/python3", "/usr/bin/python3", "/usr/bin/python3.11",
                 shutil.which("python3")):
        if not cand or not pathlib.Path(cand).exists():
            continue
        probe = subprocess.run(
            [cand, "-c", "from cryptography.fernet import Fernet"],
            capture_output=True, text=True)
        if probe.returncode != 0:
            return cand
    return ""


def test_a_crippled_interpreter_still_seals_and_says_nothing_about_it(tmp_path):
    """Existence is not capability, and the fallback is where that bites.

    Callers pick `$REPO/.venv/bin/python` if it exists and `python3` otherwise.
    On the real box the first branch always wins, so the second was never taken
    -- and this image's system `python3` ships `cryptography` 41.0.7 with no
    `_cffi_backend`, which dies inside a Rust extension as a twenty-line
    `pyo3_runtime.PanicException`. That branch is taken on exactly one occasion:
    a fresh container, where the restore that recovers a stranded card runs.

    Two properties, and the second is not decoration: `preserve_evidence.sh`
    captures this with `2>&1` into a log an operator reads, so a backtrace on a
    cycle that *succeeded* is 288 lines of false alarm a day.
    """
    import subprocess
    exe = _an_interpreter_that_cannot_do_crypto()
    if not exe:
        pytest.skip("every interpreter on this box can import cryptography")

    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "evidence" / "ccarc3" / "sweep_card").mkdir(parents=True)
    (repo / "tools" / "card_vault.py").write_text(pathlib.Path(cv.__file__).read_text())
    # The venv it must find its way into. A fresh container has one; the point of
    # the test is that the caller does not have to know that.
    (repo / ".venv").symlink_to(pathlib.Path(sys.executable).parent.parent)
    (repo / "evidence" / "ccarc3" / "sweep_card" / cv.MARKER).touch()

    sp = tmp_path / "sp"
    sp.mkdir()
    (sp / "shared_card.json").write_text(json.dumps(COOKIES))

    env = dict(os.environ, CCARC3_SCRATCH=str(sp), ARC_API_KEY="test-key-not-real")
    env.pop("CCARC3_CARD_VAULT", None)          # armed by the marker alone
    env.pop("CCARC3_CARD_VAULT_REEXEC", None)
    run = subprocess.run([exe, str(repo / "tools" / "card_vault.py"), "save", "--quiet"],
                         capture_output=True, text=True, env=env)

    sealed = repo / "evidence" / "ccarc3" / "sweep_card" / f"{COOKIES['card_id']}.vault"
    assert run.returncode == 0, run.stderr
    assert sealed.exists(), "the tool must find an interpreter that can encrypt"
    assert "sealed" in run.stdout
    assert "PanicException" not in run.stderr and "backtrace" not in run.stderr, (
        f"a recovered cycle must not leave a backtrace in the daemon log:\n{run.stderr}")

    # And a second cycle, still through the crippled interpreter, is silent.
    again = subprocess.run([exe, str(repo / "tools" / "card_vault.py"), "save", "--quiet"],
                           capture_output=True, text=True, env=env)
    assert again.returncode == 0
    assert again.stdout == "" and again.stderr == "", (
        f"stdout={again.stdout!r} stderr={again.stderr!r}")


# --------------------------------------------------------------------------- #
# the wiring, against a fixture shaped like the real .env
# --------------------------------------------------------------------------- #


def _env_fixture(sp: pathlib.Path) -> None:
    """A `.env` shaped like the real one: an assignment, and **no `export`**.

    This single detail is the whole point of the two tests below. `arc3/.env` on
    the box is one line, `ARC_API_KEY=...`, with no `export` — so `. file` makes
    it a SHELL variable and a python child cannot see it. Every existing consumer
    in `preserve_evidence.sh` reads it as `${ARC_API_KEY:-}` from the same shell,
    so nothing had ever needed it exported.

    The card vault wiring was drilled end to end before shipping and the drill
    passed, because the fixture it wrote said `export ARC_API_KEY=...`. The first
    real invocation failed with "no ARC_API_KEY in the environment". Testing the
    wiring against my own idea of the file rather than against the file.
    """
    (sp / "arc3").mkdir(parents=True, exist_ok=True)
    (sp / "arc3" / ".env").write_text("ARC_API_KEY=fixture-key-not-real\n")


def _shell_slice(path: pathlib.Path, start: str, end: str) -> str:
    text = path.read_text(encoding="utf-8")
    i = text.index(start)
    return text[i:text.index(end, i)]


def _fake_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "evidence" / "ccarc3" / "sweep_card").mkdir(parents=True)
    (repo / "tools" / "card_vault.py").write_text(pathlib.Path(cv.__file__).read_text())
    (repo / ".venv").symlink_to(pathlib.Path(sys.executable).parent.parent)
    (repo / "evidence" / "ccarc3" / "sweep_card" / cv.MARKER).touch()
    return repo


def test_the_preserver_hands_the_key_to_the_vault(tmp_path):
    """The daemon's save block, lifted out and run against a real `.env` shape."""
    import subprocess
    repo, sp = _fake_repo(tmp_path), tmp_path / "sp"
    sp.mkdir()
    _env_fixture(sp)
    (sp / "shared_card.json").write_text(json.dumps(COOKIES))

    block = _shell_slice(REPO / "tools" / "preserve_evidence.sh",
                         "    vpy=", "    # **Drain a push backlog")
    assert "card_vault.py" in block, "extraction missed the vault call"
    script = "\n".join([
        'log() { echo "$*"; }', f'REPO="{repo}"', f'SP="{sp}"',
        '. "$SP/arc3/.env"',          # exactly what the daemon does at startup
        block,
    ])
    run = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)

    sealed = repo / "evidence" / "ccarc3" / "sweep_card" / f"{COOKIES['card_id']}.vault"
    assert sealed.exists(), (
        "the daemon did not seal the pin — the key never reached the child:\n"
        + run.stdout + run.stderr)
    assert "failed" not in run.stdout.lower(), run.stdout


def test_rehydrate_hands_the_key_to_the_vault(tmp_path):
    """Step 3b, the half that runs on a replaced box — the one that matters."""
    import subprocess
    repo, sp = _fake_repo(tmp_path), tmp_path / "sp"
    sp.mkdir()
    _env_fixture(sp)
    (sp / "shared_card.json").write_text(json.dumps(COOKIES))
    (repo / "evidence" / "ccarc3" / "sweep_card" / "history.jsonl").write_text(
        json.dumps({"card_id": COOKIES["card_id"], "event": "opened", "at": 1}) + "\n")

    env = dict(os.environ, ARC_API_KEY="fixture-key-not-real",
               CCARC3_SCRATCH=str(sp))
    env.pop("CCARC3_CARD_VAULT_REEXEC", None)
    subprocess.run([sys.executable, str(repo / "tools" / "card_vault.py"), "save"],
                   check=True, capture_output=True, env=env, timeout=120)

    (sp / "shared_card.json").unlink()          # the replacement

    block = _shell_slice(REPO / "tools" / "rehydrate_box.sh",
                         'if [ -f "$SP/arc3/.env" ]; then', "\nfi\n")
    script = "\n".join([f'REPO="{repo}"', f'SP="{sp}"', block, "fi"])
    run = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=120)

    assert (sp / "shared_card.json").exists(), (
        "a replaced box could not recover its card pin:\n" + run.stdout + run.stderr)
    assert json.loads((sp / "shared_card.json").read_text()) == COOKIES
    assert "restored" in run.stdout, run.stdout


def test_a_closed_pipe_is_not_a_stack_trace(tmp_path):
    """`card_vault.py status | head` must exit quietly, not traceback.

    `preserve_evidence.sh` captures this tool's stdout AND stderr into a log an
    operator reads. A Python traceback on a cycle that succeeded is exactly the
    noise that trains people to stop reading the channel — which this repo has
    already paid for twice, once when a proxy had moved for two hours and once
    when a `tr` error reached an alert.
    """
    import subprocess
    repo = _fake_repo(tmp_path)
    sp = tmp_path / "sp"
    sp.mkdir()
    (repo / "evidence" / "ccarc3" / "sweep_card" / "history.jsonl").write_text(
        "".join(json.dumps({"card_id": f"{i}" * 8, "event": "opened", "at": 1}) + "\n"
                for i in range(9)))
    env = dict(os.environ, CCARC3_SCRATCH=str(sp), ARC_API_KEY="fixture-key-not-real")

    proc = subprocess.run(
        f'{sys.executable} {repo / "tools" / "card_vault.py"} status | head -2',
        shell=True, capture_output=True, text=True, timeout=120, env=env)

    assert "BrokenPipeError" not in proc.stderr, proc.stderr
    assert "Traceback" not in proc.stderr, proc.stderr
    assert "Exception ignored" not in proc.stderr, proc.stderr
