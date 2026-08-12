#!/usr/bin/env python3
"""Keep a scorecard's session cookies recoverable across a container replacement.

**The problem.** A scorecard is state on one backend instance, reachable only by
the load balancer's ``AWSALBAPP-*`` stickiness cookies. Those live in
``scratchpad/shared_card.json``, and the scratchpad reverts to an image snapshot
when the container is replaced. On 2026-08-11 that stranded card ``e6e6a61c``
two hours into a sweep: seven games on it, ``sb26`` and ``ft09`` both at 100.0,
and the card could no longer be read, played onto, or **closed** -- and closing
is what publishes. It cost roughly $90 and could as easily have cost $652.

**Why the cookies cannot simply be committed.** ``evidence/`` is pushed to a
public GitHub remote. ``preserve_evidence.sh`` strips cookies at copy time and
must keep doing so.

**The construction.** Encrypt the pin with a key derived from ``ARC_API_KEY`` and
commit only the ciphertext. The two stores have complementary failure modes:

* the repo is durable and public;
* ``scratchpad/arc3/.env`` is private and, measured, survives a replacement --
  its mtime is 2026-08-02, so it is part of the image snapshot, which is the same
  mechanism that *destroys* the card pin.

So the snapshot that loses the cookies restores the key that decrypts them.

**What this does and does not add to the exposure.** Anyone who can decrypt this
file already holds ``ARC_API_KEY``, and with that key they can already spend the
account's quota, open cards and play games. What they gain is the ability to
reach one specific card. That is a real but modest increment over what the key
alone confers, and it is the operator's call -- hence ``CCARC3_CARD_VAULT``,
which is **off by default**.

**Not a general secret store.** It protects one short-lived value whose loss has
a measured price. Do not extend it to the API key itself: encrypting a secret
under itself stores nothing.

Usage::

    CCARC3_CARD_VAULT=1 python3 tools/card_vault.py save
    CCARC3_CARD_VAULT=1 python3 tools/card_vault.py restore
    python3 tools/card_vault.py status
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
# Overridable via `CCARC3_SCRATCH`, in the same `or`-continuation form the other
# tools use -- the repo-wide guard in
# tests/test_ccarc3_scratch_path_is_overridable.py reads the line the literal is
# on, and it caught this file writing the default as a second positional
# argument to `os.environ.get`. Equivalent behaviour, but the guard cannot see
# the override from the offending line, and a guard that has to infer intent
# across lines is one that will miss the next tool that really does hardcode it.
SP = pathlib.Path(
    os.environ.get("CCARC3_SCRATCH")
    or "/tmp/claude-0/-home-user-athanor/a3375e8f-271e-5133-96a4-a40a6a06a752/scratchpad"
)
PIN = SP / "shared_card.json"
VAULT_DIR = REPO / "evidence" / "ccarc3" / "sweep_card"
# Fixed, and that is deliberate: a random salt would have to be stored beside the
# ciphertext, which is one more thing to lose. The secret is the API key; the
# salt only domain-separates this use of it from any other.
_REEXEC = "CCARC3_CARD_VAULT_REEXEC"
SALT = b"ccarc3-card-vault-v1"
ITERATIONS = 480_000


# **The switch has to be as durable as the thing it arms, and an env var is not.**
#
# The obvious place for `CCARC3_CARD_VAULT=1` is `scratchpad/arc3/.env`, which
# `preserve_evidence.sh` sources at start. But that file is durable only because
# it predates the image snapshot -- its mtime is 2026-08-02 -- and a line appended
# today is written *after* the snapshot, so the very replacement this vault exists
# for would revert the file and turn sealing off. The pin for the next card then
# never gets sealed, and the second replacement costs exactly what the first one
# did. A guard that switches itself off at the moment it is needed is the same
# shape as the guard whose evidence is destroyed by its own trigger.
#
# So the durable form of the switch is a committed file. Creating it is a
# deliberate act with a commit attached, which is the right ceremony for a
# decision to publish an encrypted credential, and it comes back from origin on
# every rehydrate.
MARKER = "ENABLED"


def enabled() -> bool:
    """Whether sealing is armed. Off unless the operator says otherwise.

    Env first so a single box can be overridden either way without a commit; an
    explicit ``0`` beats the committed marker, because the local operator is
    closer to the situation than the file is.
    """
    env = os.environ.get("CCARC3_CARD_VAULT", "")
    if env in ("0", "false", "no"):
        return False
    if env:
        return True
    return (VAULT_DIR / MARKER).exists()


def _key() -> bytes:
    """A Fernet key from ARC_API_KEY. Refuses rather than falling back."""
    import hashlib
    secret = os.environ.get("ARC_API_KEY", "")
    if not secret:
        raise SystemExit(
            "card_vault: no ARC_API_KEY in the environment. The vault is keyed on "
            "it, so without it a saved pin cannot be written or read. Source "
            "scratchpad/arc3/.env first.")
    raw = hashlib.pbkdf2_hmac("sha256", secret.encode(), SALT, ITERATIONS, dklen=32)
    return base64.urlsafe_b64encode(raw)


def _crypto():
    """``(Fernet, InvalidToken)``, or an explanation of which python has them.

    **Existence is not capability, and this box proves it.** The system
    ``python3`` ships ``cryptography`` 41.0.7 with no ``_cffi_backend``, so the
    import dies inside a Rust extension and surfaces as
    ``pyo3_runtime.PanicException`` over twenty lines of backtrace -- which is a
    ``BaseException``, so an ``except ImportError`` would not catch it and an
    operator reading a rehydrate log learns nothing from it. ``.venv/bin/python``
    has a working 50.0.0.

    Caught by drilling the wiring end to end in a fake repo with no ``.venv``,
    which is the only configuration where the callers' `[ -x venv ] || python3`
    fallback actually takes its fallback branch. On the real box the first branch
    always wins, so the broken one would have sat there untaken until a
    replacement -- when a fresh container has no ``.venv`` yet and the restore
    that recovers a stranded card is the first thing to need it.
    """
    # **The panic is written to fd 2 by Rust, so Python cannot catch it -- it has
    # to be closed off at the file-descriptor level.** Catching the exception is
    # enough to control the *program*, and not enough to control the *output*:
    # the six-line `pyo3_runtime` backtrace is already on stderr by the time the
    # `except` runs, and `preserve_evidence.sh` captures this with `2>&1` every
    # 300s. So a cycle that recovers perfectly -- re-execs, seals the pin, exits
    # 0 -- would still deposit a Rust backtrace in the operator's log 288 times a
    # day. This file already carries the reason that matters: an alarm channel
    # that emits noise is one people stop reading, and the next thing it says is
    # the one that counted. Silence spans the import attempt only, and is
    # restored before anything of ours is written.
    devnull, saved = os.open(os.devnull, os.O_WRONLY), os.dup(2)
    try:
        os.dup2(devnull, 2)
        try:
            from cryptography.fernet import Fernet, InvalidToken
        except BaseException as exc:                          # noqa: BLE001
            failed = exc
        else:
            failed = None
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)

    if failed is None:
        return Fernet, InvalidToken
    exc = failed
    # **Re-exec rather than instruct.** Two daemons and a session invoke this,
    # and making each one carry "which python can do crypto" is the same
    # inheritance that left the supervisor exporting a proxy port that had
    # moved hours earlier. The tool knows where the venv is; it can just go
    # there. `execv` replaces the process image, so a wedged interpreter is
    # not a problem -- nothing of it survives the call.
    alt = REPO / ".venv" / "bin" / "python"
    if (not os.environ.get(_REEXEC)
            and alt.exists() and str(alt) != sys.executable):
        os.environ[_REEXEC] = "1"                # once, so this cannot loop
        me = str(pathlib.Path(__file__).resolve())
        os.execv(str(alt), [str(alt), me, *sys.argv[1:]])
    raise SystemExit(
        f"card_vault: this interpreter ({sys.executable}) cannot load "
        f"cryptography ({type(exc).__name__}: {exc}), and there is no working "
        f"{alt} to fall back to. The sealed pin cannot be read or written "
        f"until one exists — on a fresh container, build the venv first."
    ) from None


def _fernet():
    Fernet, _ = _crypto()
    return Fernet(_key())


def vault_path(card_id: str) -> pathlib.Path:
    return VAULT_DIR / f"{card_id}.vault"


def save(quiet: bool = False) -> int:
    """Encrypt the live pin into the repo. Idempotent; safe to call repeatedly.

    ``quiet`` suppresses the routine "nothing to do" lines and nothing else.
    `preserve_evidence.sh` calls this every 300s and forwards whatever it says to
    a log an operator reads, so a per-cycle "no live pin" would be 288 lines a day
    of a daemon reporting that it is working correctly. Real failures still speak.
    """
    if not PIN.exists():
        if not quiet:
            print("card_vault: no live pin to save", flush=True)
        return 0
    try:
        pin = json.loads(PIN.read_text())
    except (OSError, ValueError) as exc:
        print(f"card_vault: pin unreadable ({exc}); not saving", flush=True)
        return 1
    cid = pin.get("card_id")
    if not cid:
        print("card_vault: pin carries no card_id; not saving", flush=True)
        return 1
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    out = vault_path(cid)
    plain = PIN.read_bytes()
    # **Re-sealing unchanged bytes must not produce a new file, and Fernet makes
    # that a real hazard rather than a nicety.** Every ciphertext carries a random
    # IV and a timestamp, so `encrypt` on identical input returns different bytes
    # every call. `preserve_evidence.sh` calls this every 300s and commits
    # whatever changed under `evidence/`, so an unconditional write would push a
    # commit every five minutes for the life of a sweep -- ~288 a day, all of them
    # saying the pin changed when it did not. Compare the plaintext, which is the
    # thing that actually has to be current.
    if out.exists():
        _, InvalidToken = _crypto()
        try:
            if _fernet().decrypt(out.read_bytes()) == plain:
                return 0
        except InvalidToken:
            pass                    # unreadable or keyed differently: reseal it
    blob = _fernet().encrypt(plain)
    tmp = out.with_suffix(".vault.tmp")
    tmp.write_bytes(blob)
    tmp.replace(out)          # atomic: a half-written vault is worse than none
    # **A log line must not be able to fail the save.** `relative_to(REPO)`
    # raises when the vault directory is not under the repo -- which it is in
    # every test, and would be for anyone pointing this at another tree. The
    # cosmetic path shortening was aborting the operation it was describing.
    try:
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out
    print(f"card_vault: sealed {cid[:8]} -> {shown} ({len(blob)} bytes)", flush=True)
    return 0


def restore(card_id: str = "") -> int:
    """Decrypt a sealed pin back into the scratchpad.

    **Refuses to overwrite a live pin.** If one exists it is the session's own
    and is authoritative; a stale vault would replace working cookies with dead
    ones, turning a healthy card into a stranded one.
    """
    if PIN.exists():
        print(f"card_vault: {PIN.name} already exists — refusing to overwrite a "
              f"live pin with a stored one", flush=True)
        return 0
    if not card_id:
        cards = _cards_from_history()
        if not cards:
            print("card_vault: no card history; nothing to restore", flush=True)
            return 1
        card_id = cards[-1]
    p = vault_path(card_id)
    if not p.exists():
        print(f"card_vault: no vault for {card_id[:8]} — that card's cookies were "
              f"never sealed and cannot be recovered", flush=True)
        return 1
    _, InvalidToken = _crypto()
    try:
        plain = _fernet().decrypt(p.read_bytes())
    except InvalidToken:
        print(f"card_vault: {p.name} will not decrypt with this ARC_API_KEY. The "
              f"key changed, or the vault is corrupt. NOT writing a pin.", flush=True)
        return 1
    SP.mkdir(parents=True, exist_ok=True)
    PIN.write_bytes(plain)
    PIN.chmod(0o600)
    print(f"card_vault: restored {card_id[:8]} -> {PIN}", flush=True)
    return 0


def _cards_from_history() -> list[str]:
    out: list[str] = []
    hist = VAULT_DIR / "history.jsonl"
    try:
        text = hist.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        try:
            cid = json.loads(line).get("card_id")
        except ValueError:
            continue
        if cid and cid not in out:
            out.append(cid)
    return out


def status() -> int:
    # Say WHICH switch is answering. "enabled: True" with no source sent an
    # operator to the wrong file twice over on the proxy port; naming the source
    # is the difference between a status line and a status line you can act on.
    env = os.environ.get("CCARC3_CARD_VAULT", "")
    if env in ("0", "false", "no"):
        why = f"CCARC3_CARD_VAULT={env} — explicitly off, overriding any marker"
    elif env:
        why = f"CCARC3_CARD_VAULT={env}"
    elif (VAULT_DIR / MARKER).exists():
        why = f"committed marker {VAULT_DIR.name}/{MARKER}"
    else:
        why = "neither CCARC3_CARD_VAULT nor a committed marker"
    print(f"  enabled     : {enabled()}  ({why})")
    print(f"  live pin    : {'present' if PIN.exists() else 'absent'}")
    for cid in _cards_from_history():
        p = vault_path(cid)
        print(f"  {cid[:8]}    : {'SEALED' if p.exists() else 'not sealed'}")
    return 0


def main(argv: list[str]) -> int:
    argv = [a for a in argv]
    quiet = False
    for flag in ("-q", "--quiet"):
        if flag in argv:
            quiet = True
            argv.remove(flag)
    cmd = argv[1] if len(argv) > 1 else "status"
    if cmd == "status":
        return status()
    # **The switch gates SAVING, not restoring, and the asymmetry is the point.**
    # What the operator is deciding is whether to *publish* an encrypted
    # credential to a public repo. Nothing about that decision is undone by
    # declining to read a file that is already there: if a vault exists, the
    # publication has already happened, and refusing to decrypt it during a
    # recovery forfeits the sweep it was written to save while buying back no
    # secrecy at all. A restore with the switch off is a no-op anyway -- there is
    # no vault to find -- so the only behaviour this changes is the one case
    # where being strict would be pure loss.
    if cmd == "restore":
        return restore(argv[2] if len(argv) > 2 else "")
    if not enabled():
        if not quiet:
            print("card_vault: not armed; doing nothing. This is the default — "
                  "sealing publishes an encrypted credential to a public repo and "
                  f"is an operator decision. Set CCARC3_CARD_VAULT=1, or commit "
                  f"{VAULT_DIR.name}/{MARKER} to arm it durably.", flush=True)
        return 0
    if cmd == "save":
        return save(quiet=quiet)
    sys.exit(f"usage: {argv[0]} [status|save|restore [card_id]]")


if __name__ == "__main__":
    try:
        _rc = main(sys.argv)
    except BrokenPipeError:
        # `card_vault.py status | head` closes the pipe while we are still
        # writing. Python then flushes again at interpreter shutdown, raises a
        # second time, and prints "Exception ignored" over whatever the operator
        # was reading. Pointing stdout at /dev/null first is the fix CPython
        # documents. Not cosmetic for long: `preserve_evidence.sh` captures this
        # tool's output, and a daemon that logs a stack trace on a successful
        # cycle is a log people stop reading -- the failure mode this repo has
        # already paid for twice.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        _rc = 0
    raise SystemExit(_rc)
