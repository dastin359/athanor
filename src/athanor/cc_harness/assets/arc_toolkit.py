"""arc — observation and verification helpers for this puzzle workspace.

Import it the same way from anywhere in the workspace::

    from arc import train_samples, test_samples, verify, check, show, diff

That works from a script under ``explore/`` (``python explore/foo.py``), from a
one-liner at the workspace root (``python -c "from arc import ..."``), and from
a module invocation (``python -m explore.foo``). No ``sys.path`` boilerplate is
needed in any of them.

Design rule, and it is deliberate: **this module contains no transformation
primitives.** No rotate, no flood-fill, no connected components. Building those
is the reasoning work, and handing them over would change what is being
measured. What you get here is the ability to *look* and to *check*.

The important function is :func:`verify`. Anything you believe about this puzzle
should pass through it, because a claim that has been executed is worth more
than a claim that has been asserted — and it costs a fraction of the tokens.
Every call is appended to ``.athanor/invariants.jsonl``, which survives context
compaction and is replayed by ``python gate.py status``.
"""

from __future__ import annotations

import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "WORKSPACE",
    "train_samples",
    "test_samples",
    "verify",
    "refute",
    "check",
    "load_solution",
    "solution_module",
    "show",
    "diff",
    "shape",
    "colors",
    "histogram",
    "png",
    "invariants",
]

Grid = list[list[int]]


def _find_workspace() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / ".athanor" / "state.json").is_file():
            return candidate
    return here


#: Root of this run's workspace.
WORKSPACE: Path = _find_workspace()

_TASK = json.loads((WORKSPACE / "task" / "task.json").read_text(encoding="utf-8"))

#: Training pairs — ``{'input': grid, 'output': grid}``.
train_samples: list[dict[str, Grid]] = _TASK.get("train") or []

#: Test inputs — ``{'input': grid}``. Test outputs are not in this workspace.
test_samples: list[dict[str, Grid]] = _TASK.get("test") or []

#: ARC colour names, for readable reports.
COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "lightblue", 9: "maroon",
}

_PALETTE = {
    0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64), 4: (255, 220, 0),
    5: (170, 170, 170), 6: (240, 18, 190), 7: (255, 133, 27), 8: (127, 219, 255), 9: (135, 12, 37),
}


# ── verification ─────────────────────────────────────────────────────────────

def _caller_frame() -> Any:
    """The frame of the script that called into this module."""
    for frame in inspect.stack()[1:]:
        filename = frame.filename or ""
        if filename == __file__ or filename.startswith("<"):
            continue
        return frame
    return None


def _caller_source() -> str:
    """Best-effort workspace-relative path of the script that called verify()."""
    frame = _caller_frame()
    if frame is None:
        return "<unknown>"
    try:
        return str(Path(frame.filename).resolve().relative_to(WORKSPACE))
    except ValueError:
        return os.path.basename(frame.filename)


_AST_CACHE: dict[str, Any] = {}


def _parse_caller(filename: str) -> Any:
    key = f"{filename}:{os.path.getmtime(filename)}" if os.path.exists(filename) else filename
    if key not in _AST_CACHE:
        try:
            import ast

            _AST_CACHE[key] = ast.parse(Path(filename).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            _AST_CACHE[key] = None
    return _AST_CACHE[key]


def _condition_evidence() -> tuple[str, bool, bool, bool]:
    """Recover the *expression* the caller passed as ``condition``.

    Returns ``(source_text, is_literal, is_opaque, is_unsourced)``. The ledger
    records what was actually executed, not just what was claimed — and a
    condition that is a compile-time constant is an assertion wearing a
    verification's clothes, which is exactly what the whole discipline exists to
    prevent.

    ``is_unsourced`` marks the case where the call site could not be read at all:
    a ``python -c`` one-liner, a heredoc piped to stdin, a REPL. That used to be
    indistinguishable from a clean capture — the entry simply carried no
    expression, and the constant-condition warning could not fire because the
    warning is derived from source the parser never reached. A solver recorded a
    refutation from a heredoc with a literal ``True`` condition, the exact
    anti-pattern the check exists to catch, and got no warning; it noticed
    unaided. "No evidence" and "good evidence" must not look alike.
    """
    import ast

    frame = _caller_frame()
    if frame is None:
        return "", False, False, True
    tree = _parse_caller(frame.filename)
    if tree is None:
        return "", False, False, True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name not in {"verify", "refute"} or node.lineno != frame.lineno:
            continue

        condition = None
        if len(node.args) >= 2:
            condition = node.args[1]
        else:
            for keyword in node.keywords:
                if keyword.arg == "condition":
                    condition = keyword.value
                    break
        if condition is None:
            return "", False, False, False

        body = condition.body if isinstance(condition, ast.Lambda) else condition
        literal = isinstance(body, ast.Constant)
        # A bare name records "allok" as the evidence, which tells a future
        # context nothing about what was executed — the ledger's whole value is
        # that it says what ran.
        opaque = isinstance(body, ast.Name)
        # `True is (...)` and `... == False` slip past the constant check while
        # being the same anti-pattern: a boolean literal standing in for a
        # measurement. A solver shipped exactly that and caught it unaided.
        if isinstance(body, ast.Compare):
            operands = [body.left, *body.comparators]
            if any(isinstance(o, ast.Constant) and isinstance(o.value, bool) for o in operands):
                literal = True
        try:
            return ast.unparse(condition), literal, opaque, False
        except Exception:  # noqa: BLE001 - unparse is a convenience, not a contract
            return "", literal, opaque, False
    return "", False, False, False


def refute(claim: str, condition: bool | Callable[[], Any] = False, note: str | None = None,
           *, key: str | None = None, retract: bool = False) -> bool:
    """Record a hypothesis as *ruled out* by an executed check.

    ``verify()`` has one channel for two different findings, and a false one
    leaves a `[REFUTED]` line that reads like a defect in your own work rather
    than like the discovery it is. Ruling something out is a result::

        refute("8-connectivity explains the selection",
               selected_under_8 != expected)

    Returns True when the hypothesis was successfully ruled out. Dead ends are
    worth as much as live ones — re-deriving a hypothesis you already killed is
    the most common way to burn an iteration budget.

    Withdraw a mistaken dead end with ``refute(claim, retract=True)``.
    """
    return verify(claim, condition, note, key=key, retract=retract, _mode="ruled_out")


def _previous_entry(entry_key: str) -> dict[str, Any] | None:
    """The most recent ledger entry filed under ``entry_key``, if any."""
    ledger = WORKSPACE / ".athanor" / "invariants.jsonl"
    if not ledger.is_file():
        return None
    found: dict[str, Any] | None = None
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(entry.get("key") or entry.get("claim")) == entry_key:
            found = entry
    return found


def _verdict_word(entry: dict[str, Any]) -> str:
    if entry.get("retracted"):
        return "RETRACTED"
    if entry.get("mode") == "ruled_out":
        return "RULED OUT" if entry.get("holds") else "STILL OPEN"
    return "VERIFIED" if entry.get("holds") else "REFUTED"


def verify(
    claim: str,
    condition: bool | Callable[[], Any] = False,
    note: str | None = None,
    *,
    retract: bool = False,
    key: str | None = None,
    _mode: str = "invariant",
) -> bool:
    """Establish a claim about this puzzle by executing it.

    ``condition`` is either a boolean or a zero-argument callable (use a
    callable when the check itself might raise — the exception is recorded as a
    refutation rather than crashing your script)::

        verify("every output is 20x20",
               all(len(s['output']) == 20 and len(s['output'][0]) == 20
                   for s in train_samples))

        verify("background colour is always 8",
               lambda: {s['input'][0][0] for s in train_samples} == {8})

    Prints ``[VERIFIED]`` or ``[REFUTED]``, records the result, and returns the
    boolean so you can branch on it.

    The ledger is append-only and **the most recent entry for a claim wins**, so
    re-verifying a claim supersedes the earlier record. If a check was wrong —
    a condition that was accidentally a tautology, say — withdraw it::

        verify("outputs are always square", retract=True)

    A retracted claim disappears from ``invariants()`` and from
    ``gate.py status``. Withdrawing a bad invariant matters: the whole point of
    the ledger is that everything in it has been executed, and one claim that
    only looks verified poisons the rest.

    Supersession is keyed on the claim string, which silently fails the moment
    you reword a claim while correcting it. Pass ``key=`` to make it explicit::

        verify("output height equals input height", ..., key="height-relation")
        verify("output height equals input height times 2", ..., key="height-relation")

    When a record replaces an earlier one, the replacement carries the claim it
    displaced and prints it. So **state only what is currently true** — do not
    write the correction into the claim text ("X is 1; it is instead 2"), which
    leaves a live invariant whose own first clause is false. The ledger keeps
    the history; the claim should carry the finding.

    A claim whose verdict changes between runs prints ``CHANGED VERDICT``, which
    is worth stopping for: code written while it held is now built on sand.

    Use :func:`refute` rather than a false ``verify`` when the finding is that a
    hypothesis is dead.
    """
    error = ""
    if retract:
        holds = False
    elif callable(condition):
        try:
            holds = bool(condition())
        except Exception as exc:  # noqa: BLE001 - a check that blows up has not held
            holds, error = False, f"{type(exc).__name__}: {exc}"
    else:
        holds = bool(condition)

    expression, literal, opaque, unsourced = (
        ("", False, False, False) if retract else _condition_evidence()
    )

    entry = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim": str(claim),
        "holds": holds,
        "source": _caller_source(),
        "key": str(key) if key else str(claim),
    }

    prior = None if retract else _previous_entry(entry["key"])
    if prior is not None and not prior.get("retracted"):
        reworded = str(prior.get("claim", "")) != str(claim)
        flipped = bool(prior.get("holds")) != holds
        if reworded or flipped:
            entry["supersedes"] = {
                "claim": str(prior.get("claim", "")),
                "verdict": _verdict_word(prior),
                "at": prior.get("at", ""),
            }
    if _mode == "ruled_out":
        entry["mode"] = "ruled_out"
    if expression:
        entry["expression"] = expression
    if literal:
        entry["literal"] = True
    if opaque:
        entry["opaque"] = True
    if unsourced:
        entry["unsourced"] = True
    if retract:
        entry["retracted"] = True
    if note:
        entry["note"] = str(note)
    if error:
        entry["error"] = error

    ledger = WORKSPACE / ".athanor" / "invariants.jsonl"
    try:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # a read-only workspace must not break exploration

    if retract:
        label = "RETRACTED"
    elif _mode == "ruled_out":
        label = "RULED OUT" if holds else "STILL OPEN"
    else:
        label = "VERIFIED " if holds else "REFUTED  "
    suffix = f"  ({note})" if note else ""
    if error:
        suffix += f"  [{error}]"
    print(f"[{label}] {claim}{suffix}")
    superseded = entry.get("supersedes")
    if superseded:
        was = superseded["verdict"]
        old_claim = superseded["claim"]
        if old_claim != str(claim):
            print(
                f"           ^ supersedes under key `{entry['key']}`: "
                f'"{old_claim}" [{was}]'
            )
            if was == "REFUTED":
                print(
                    "           The ledger now carries both the dead reading and its "
                    "replacement, so you do not need to encode the correction in the "
                    "claim text. State only what is true."
                )
        else:
            print(
                f"           ^ CHANGED VERDICT: this same claim was [{was}] at "
                f"{superseded['at']}. Anything you built on the old verdict is now "
                "suspect — re-check the code that assumed it."
            )
    if opaque and not literal:
        print(
            f"           ^ the recorded evidence is just the name `{expression}`. After a "
            "compaction that says nothing about what ran — inline the check, or pass "
            "note= with the measured value."
        )
    if literal:
        print(
            "           ^ WARNING: that condition is a compile-time constant "
            f"({expression or 'literal'}). Nothing was measured, so this records an "
            "assertion, not a verification. Re-run it with a real check, or retract it."
        )
    if unsourced:
        print(
            "           ^ NO EVIDENCE CAPTURED: this ran from a -c one-liner, a heredoc "
            "or a REPL, so the condition's source could not be read. The ledger entry "
            "carries the claim but nothing about what was executed, and the "
            "constant-condition check could not run at all. Put the check in a file "
            "under explore/ — or at minimum pass note= with the measured value."
        )
    return holds


def invariants() -> list[dict[str, Any]]:
    """Live invariants: most recent entry per claim, retractions removed."""
    ledger = WORKSPACE / ".athanor" / "invariants.jsonl"
    if not ledger.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[str(entry.get("key") or entry.get("claim"))] = entry
    return [entry for entry in latest.values() if not entry.get("retracted")]


# ── free dry-run ─────────────────────────────────────────────────────────────

def check(
    solve_fn: Callable[[Grid], Any], *, verbose: bool = True, show_diff: bool = False
) -> dict[str, Any]:
    """Score a candidate ``solve`` against the training pairs. Costs nothing.

    Use this as many times as you like while iterating. It does **not** consume
    an iteration and does **not** record anything — ``python gate.py submit`` is
    the formal checkpoint, and this exists so you never spend one on a bug you
    could have caught here::

        from arc import check
        def solve(grid): ...
        check(solve)
    """
    import copy

    rows: list[dict[str, Any]] = []
    correct = 0
    total_pixel = 0.0

    for idx, sample in enumerate(train_samples):
        expected = sample["output"]
        row: dict[str, Any] = {"index": idx, "correct": False, "pixel_accuracy": 0.0}
        try:
            predicted = _first_candidate(solve_fn(copy.deepcopy(sample["input"])))
            row["predicted"] = predicted
            row["correct"] = predicted == expected
            row["pixel_accuracy"] = _pixel_accuracy(predicted, expected)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        correct += 1 if row["correct"] else 0
        total_pixel += row["pixel_accuracy"]
        rows.append(row)

    test_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(test_samples):
        row = {"index": idx}
        try:
            raw = solve_fn(copy.deepcopy(sample["input"]))
            candidates = raw if _looks_like_candidate_list(raw) else [raw]
            row["candidates"] = candidates
            row["shapes"] = [shape(c) for c in candidates]
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        test_rows.append(row)

    summary = {
        "train": rows,
        "test": test_rows,
        "train_correct": correct,
        "train_total": len(train_samples),
        "train_pixel_accuracy": total_pixel / len(train_samples) if train_samples else 0.0,
        "all_train_correct": bool(train_samples) and correct == len(train_samples),
    }

    if verbose:
        for row in rows:
            if row.get("error"):
                print(f"train {row['index']}  ERROR  {row['error']}")
            elif row["correct"]:
                print(f"train {row['index']}  PASS   {shape(row.get('predicted'))}")
            else:
                expected = train_samples[row["index"]]["output"]
                wrong = _count_wrong(row.get("predicted"), expected)
                print(
                    f"train {row['index']}  FAIL   got {shape(row.get('predicted'))} "
                    f"want {shape(expected)}  "
                    f"wrong={wrong}  pixel={row['pixel_accuracy']:.3f}"
                )
                if show_diff:
                    predicted = row.get("predicted")
                    if isinstance(predicted, list) and shape(predicted) == shape(expected):
                        cells = [
                            (r, c, expected[r][c], predicted[r][c])
                            for r in range(len(expected))
                            for c in range(len(expected[0]))
                            if expected[r][c] != predicted[r][c]
                        ]
                        listing = " ".join(f"({r},{c}) want {w} got {g}" for r, c, w, g in cells[:30])
                        print(f"           {listing}")
                        if len(cells) > 30:
                            print(f"           … {len(cells) - 30} more")
        for row in test_rows:
            if row.get("error"):
                print(f"test  {row['index']}  ERROR  {row['error']}")
            else:
                candidates = row.get("candidates") or []
                print(f"test  {row['index']}  -> {len(candidates)} candidate(s) "
                      f"{row.get('shapes')}")
                # Otherwise the only way to see how far apart two candidates are
                # is the gate's report, which costs an iteration — a solver spent
                # one using the gate as a viewer.
                if len(candidates) > 1:
                    first, second = candidates[0], candidates[1]
                    if shape(first) != shape(second):
                        print("           candidates differ in shape")
                    else:
                        n = sum(
                            1
                            for r in range(len(first))
                            for c in range(len(first[0]))
                            if first[r][c] != second[r][c]
                        )
                        print(f"           candidates differ in {n} cell(s)")
        print(
            f"== {summary['train_correct']}/{summary['train_total']} training examples reproduced "
            f"(mean pixel {summary['train_pixel_accuracy']:.3f})"
        )
        if summary["all_train_correct"]:
            _hedging_advice(test_rows)
            hypothesis = WORKSPACE / "solution" / "hypothesis.md"
            if hypothesis.is_file() and hypothesis.read_text(encoding="utf-8").strip():
                print("== ready for `python gate.py submit`")
            else:
                print("== write solution/hypothesis.md, then `python gate.py submit`")
    return summary


def _hedging_advice(test_rows: list[dict[str, Any]]) -> None:
    """Raise the second-attempt question here, where acting on it is free.

    The gate asks it too, but a candidate comes out of ``solve()``, so acting on
    the gate's version costs a whole iteration to resubmit. One solver on a
    3-iteration budget spent a third of it doing exactly that. Asking at dry-run
    time means the hedge lands in the submission that was going to happen anyway.
    """
    unspent = [
        row["index"]
        for row in test_rows
        if not row.get("error") and 0 < len(row.get("candidates") or []) < 2
    ]
    if not unspent:
        return

    # A rival only belongs in the advice for a test example where it actually
    # predicts something different. Naming a rival that diverges on test 1 while
    # test 0 is the one with the free slot trains the solver to skim the nudge.
    live_rivals = []
    for entry in rivals():
        if not entry.get("fits_training"):
            continue
        predictions = entry.get("predictions") or []
        relevant = [
            index for index in unspent
            if index < len(predictions) and predictions[index] is not None
        ]
        divergent = _rival_divergence(predictions)
        if divergent is not None:
            relevant = [index for index in relevant if index in divergent]
        if relevant:
            live_rivals.append({**entry, "relevant": relevant})
    dead_ends = [e for e in invariants() if e.get("mode") == "ruled_out" and e.get("holds")]

    where = ", ".join(f"test {index}" for index in unspent)
    print(f"\n== {where} carries one candidate, and ARC-AGI-2 scores two.")

    if not live_rivals and not dead_ends:
        # An empty rival ledger used to mean silence — which skipped the prompt
        # on the very first dry run, the one moment before the first submission
        # when acting on it is free. A solver reported exactly that: it
        # registered its rivals afterwards and only saw the question from the
        # gate, after the iteration was spent.
        print("   You have registered no rival readings, which is not the same as there being")
        print("   none. Name the interpretation you rejected on the way here and run it through")
        print("   arc.rival(name, fn): if it reproduces every training pair and predicts")
        print("   something different, it is the best possible use of the second slot.")
        return

    if live_rivals:
        for entry in live_rivals[:3]:
            where = ", ".join(f"test {i}" for i in entry["relevant"])
            print(f"   rival fitting every training pair, differing on {where}: {entry['name']}")
        print("   Training cannot separate it from your reading. Unless you can point at")
        print("   evidence that rules it out, return it as the second candidate.")
    else:
        print(f"   You ruled out {len(dead_ends)} rival reading(s). Check how each died:")
        print("   a training pair it fails is a proof; extending a training-output regularity")
        print("   to the test input is not. Hedge here — after submitting it costs an iteration.")


def rival(name: str, solve_fn: Callable[[Grid], Any]) -> dict[str, Any]:
    """Register an alternative reading of the puzzle, scored against training.

    Use this the moment you implement a rival interpretation in order to
    *compare* it — which is usually the moment you are about to discard it::

        def strict(grid): ...        # the reading you suspect is wrong
        arc.rival("diagonals may not brush a wall corner", strict)

    If the rival reproduces every training pair and predicts something different
    from your own solution on a test input, the gate will say so at submission
    time. ARC-AGI-2 scores two attempts per test example, and a rival that fits
    all the evidence you have is the single best use of the second one.

    This exists because of a measured loss: a solver ruled out exactly such a
    rival by extending a regularity from the training outputs to the test input,
    discarded the free second attempt, and missed by two cells out of four
    hundred. It had the rival implemented at the time.
    """
    summary = check(solve_fn, verbose=False)
    predictions: list[Any] = []
    for sample in test_samples:
        try:
            raw = solve_fn([row[:] for row in sample["input"]])
            predictions.append(_first_candidate(raw))
        except Exception:  # noqa: BLE001 - a rival that crashes on test is still a record
            predictions.append(None)

    entry = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": str(name),
        "fits_training": bool(summary["all_train_correct"]),
        "train_correct": summary["train_correct"],
        "train_total": summary["train_total"],
        "predictions": predictions,
        "source": _caller_source(),
    }
    ledger = WORKSPACE / ".athanor" / "rivals.jsonl"
    try:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass

    if entry["fits_training"]:
        print(
            f"[RIVAL   ] {name} — reproduces all "
            f"{entry['train_total']} training pairs. Training cannot separate it from yours."
        )
        # The whole point is to stop a slot being forfeited, so the payload is
        # whether spending it would change anything. This function holds the
        # rival's predictions; comparing them costs nothing.
        standing = _rival_standing(predictions)
        if standing is None:
            print(
                "           No solution/solve.py to compare against yet — the gate will check "
                "at submission time."
            )
        else:
            differs = [i for i, s in standing.items() if s == "differs"]
            hedged = [i for i, s in standing.items() if s == "hedged"]
            contested = [i for i, s in standing.items() if s == "contested"]
            if contested:
                where = ", ".join(f"test {i}" for i in contested)
                print(
                    f"           On {where} both slots are already spent, on readings that are "
                    "not this one. Taking this rival means dropping one of them — so the "
                    "question is which two of the three survive the most evidence, not "
                    "whether to add a third."
                )
            if differs:
                where = ", ".join(f"test {i}" for i in differs)
                print(
                    f"           It predicts differently on {where}. That is your second candidate."
                )
            if hedged:
                where = ", ".join(f"test {i}" for i in hedged)
                print(
                    f"           On {where} it is already your second candidate — the slot is "
                    "spent on it and the hedge is doing its job. Leave it in place."
                )
            if not differs and not hedged and not contested:
                print(
                    "           It predicts exactly what your first candidate does on every test "
                    "input, so spending a slot on it would change nothing."
                )
    else:
        print(
            f"[RIVAL   ] {name} — fails training "
            f"({entry['train_correct']}/{entry['train_total']}). Ruled out by evidence, not by "
            "assumption."
        )
    return entry


def _rival_standing(predictions: list[Any]) -> dict[int, str] | None:
    """How a rival stands against the current solution, per test index.

    ``"differs"``  — predicts something neither of your candidates does.
    ``"hedged"``   — it *is* one of your later candidates already.
    ``"same"``     — identical to your first candidate; spending a slot is a no-op.

    The three were once collapsed into "diverges or not", which produced
    actively inverted advice: a solver hedged correctly with a rival as its
    second candidate, and ``rival()`` then told it the reading "is not a
    divergent reading and needs no slot" — because the rival was found inside
    the candidate list it had just been added to. Following that line would have
    deleted a correct hedge. "Already your second candidate" and "redundant with
    your first" are opposite situations and must not print the same sentence.

    Returns None when there is no loadable solution to compare against.
    """
    try:
        solve = load_solution()
    except Exception:  # noqa: BLE001 - no solution yet, or it does not load
        return None

    standing: dict[int, str] = {}
    for index, prediction in enumerate(predictions):
        if prediction is None or index >= len(test_samples):
            continue
        try:
            raw = solve([row[:] for row in test_samples[index]["input"]])
        except Exception:  # noqa: BLE001
            continue
        mine = raw if _looks_like_candidate_list(raw) else [raw]
        if not mine:
            standing[index] = "differs"
        elif prediction == mine[0]:
            standing[index] = "same"
        elif prediction in mine[1:]:
            standing[index] = "hedged"
        elif len(mine) >= 2:
            # Divergent, but there is no free slot to put it in. "That is your
            # second candidate" is wrong here — the second candidate exists and
            # is a different reading, so this is a swap, not an addition. Both
            # round-5 solvers reported being told to add what could only replace.
            standing[index] = "contested"
        else:
            standing[index] = "differs"
    return standing


def _rival_divergence(predictions: list[Any]) -> list[int] | None:
    """Test indices where a rival's prediction is not among the shipped candidates."""
    standing = _rival_standing(predictions)
    if standing is None:
        return None
    return [index for index, status in standing.items() if status == "differs"]


def rivals() -> list[dict[str, Any]]:
    """Alternative readings registered so far, most recent per name."""
    ledger = WORKSPACE / ".athanor" / "rivals.jsonl"
    if not ledger.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[str(entry.get("name"))] = entry
    return list(latest.values())


def solution_module(path: str | os.PathLike[str] | None = None) -> Any:
    """Import ``solution/solve.py`` and return the whole module.

    ``load_solution()`` hands back only ``solve``, which is not enough when you
    want to build a rival that shares the shipped parse — a solver needing the
    module's helpers had to hand-roll ``importlib`` to get them, which is the
    boilerplate this toolkit promises you never need::

        from arc import solution_module, rival
        shipped = solution_module()
        def alt(grid):
            parsed = shipped._parse(grid)     # reuse, do not re-derive
            ...
        rival("unmatched dots survive unchanged", alt)

    Reusing the shipped helpers is the point: a rival that re-implements the
    parse is testing two changes at once.
    """
    import importlib.util

    target = Path(path) if path else (WORKSPACE / "solution" / "solve.py")
    if not target.is_absolute():
        target = WORKSPACE / target
    if not target.is_file():
        raise FileNotFoundError(f"No solution at {target}")

    spec = importlib.util.spec_from_file_location("athanor_candidate_solution", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_solution(path: str | os.PathLike[str] | None = None) -> Callable[[Grid], Any]:
    """Return the ``solve`` function from ``solution/solve.py``.

    The doctrine asks you to run your verified invariants against your own test
    predictions before accepting — which means loading the solution you just
    wrote into an exploration script. Do it with this, from anywhere::

        from arc import load_solution, test_samples
        solve = load_solution()
        prediction = solve(test_samples[0]['input'])

    Resolves relative to the workspace, so the calling script works from any
    directory.
    """
    module = solution_module(path)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise AttributeError("solution/solve.py does not define a callable solve(grid)")
    return solve


def _looks_like_candidate_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and value
        and all(isinstance(v, list) and v and isinstance(v[0], list) for v in value)
    )


def _first_candidate(value: Any) -> Any:
    return value[0] if _looks_like_candidate_list(value) else value


def _count_wrong(predicted: Any, expected: Grid) -> int | str:
    if not isinstance(predicted, list) or shape(predicted) != shape(expected):
        return "n/a (shape mismatch)"
    return sum(
        1
        for r in range(len(expected))
        for c in range(len(expected[0]))
        if predicted[r][c] != expected[r][c]
    )


def _pixel_accuracy(predicted: Any, expected: Grid) -> float:
    try:
        exp_rows, exp_cols = len(expected), len(expected[0])
        total = exp_rows * exp_cols
        if not total:
            return 0.0
        pred_rows = len(predicted)
        pred_cols = len(predicted[0]) if pred_rows else 0
        return sum(
            1
            for r in range(min(pred_rows, exp_rows))
            for c in range(min(pred_cols, exp_cols))
            if predicted[r][c] == expected[r][c]
        ) / total
    except Exception:  # noqa: BLE001
        return 0.0


# ── observation ──────────────────────────────────────────────────────────────

def shape(grid: Any) -> str:
    """``"HxW"`` for a grid, ``"?"`` for anything that is not one."""
    try:
        return f"{len(grid)}x{len(grid[0])}"
    except Exception:  # noqa: BLE001
        return "?"


def colors(grid: Grid) -> list[int]:
    """Sorted distinct colours present in a grid."""
    return sorted({int(cell) for row in grid for cell in row})


def histogram(grid: Grid) -> dict[int, int]:
    """``{colour: count}``, most useful for spotting the background."""
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[int(cell)] = counts.get(int(cell), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


#: Fired at most once per process, so a long exploration session gets the
#: reminder without every call nagging.
_DURABILITY_REMINDED = False

#: Exploration scripts written before an empty ledger becomes worth mentioning.
_UNRECORDED_SCRIPT_THRESHOLD = 4


def _exploration_durability_check() -> None:
    """Mention an empty ledger during exploration, not only at dry-run time.

    The dry-run nudges only reach a solver that already has a candidate
    solution. Two agents were observed forty minutes into a hard task with six
    exploration scripts, no `solve.py`, and nothing recorded — the exact window
    where a compaction costs the most and where nothing was saying so.

    Hung off ``show()`` alone, this reached almost nobody: measured across a
    round, agents called ``show()`` in one script out of five to eight, usually
    an early one — so the reminder was gated on a function they had mostly
    stopped calling by the time the condition became true. It now also runs at
    interpreter exit, which is both broader and more precise: a script that
    records something during its run clears the condition and stays silent, and
    one that records nothing says so where the agent is already reading.
    """
    global _DURABILITY_REMINDED
    if _DURABILITY_REMINDED:
        return
    try:
        explore = WORKSPACE / "explore"
        scripts = [
            p for p in explore.glob("*.py") if p.name != "arc.py"
        ] if explore.is_dir() else []
        if len(scripts) < _UNRECORDED_SCRIPT_THRESHOLD or invariants():
            return
    except Exception:  # noqa: BLE001 - a reminder must never break exploration
        return
    _DURABILITY_REMINDED = True
    print(
        f"\n[note] {len(scripts)} exploration scripts, nothing recorded with arc.verify().\n"
        "       A compaction now would discard everything you have worked out; the ledger\n"
        "       and NOTES.md are what `gate.py status` replays. Record the facts you are\n"
        "       already relying on.\n"
    )


def _register_durability_atexit() -> None:
    import atexit

    def _at_exit() -> None:
        try:
            _exploration_durability_check()
        except Exception:  # noqa: BLE001 - never let a reminder break a run
            pass

    atexit.register(_at_exit)


_register_durability_atexit()


def show(grid: Grid, title: str | None = None, ruler: bool = True) -> None:
    """Print a grid with row and column indices."""
    _exploration_durability_check()
    if title:
        print(title)
    if not grid:
        print("<empty>")
        return
    height, width = len(grid), len(grid[0])
    pad = len(str(height - 1))
    if ruler:
        tens = "".join(str((c // 10) % 10) if c >= 10 else " " for c in range(width))
        ones = "".join(str(c % 10) for c in range(width))
        if width > 10:
            print(" " * (pad + 1) + tens)
        print(" " * (pad + 1) + ones)
    for r, row in enumerate(grid):
        print(f"{r:>{pad}} " + "".join(str(cell) for cell in row))
    print(f"({height}x{width}, colours {colors(grid)})")


def diff(a: Grid, b: Grid, *, limit: int = 40, labels: tuple[str, str] = ("a", "b")) -> None:
    """Print where two grids disagree."""
    if shape(a) != shape(b):
        print(f"shape mismatch: {labels[0]}={shape(a)} {labels[1]}={shape(b)}")
        return
    entries = [
        (r, c, a[r][c], b[r][c])
        for r in range(len(a))
        for c in range(len(a[0]))
        if a[r][c] != b[r][c]
    ]
    if not entries:
        print(f"identical ({shape(a)})")
        return
    rows = sorted({e[0] for e in entries})
    cols = sorted({e[1] for e in entries})
    print(
        f"{len(entries)} differing cells in {shape(a)}; "
        f"rows {rows[0]}-{rows[-1]}, cols {cols[0]}-{cols[-1]}"
    )
    for r, c, va, vb in entries[:limit]:
        print(f"  ({r},{c}) {labels[0]}={va} ({COLOR_NAMES.get(va, '?')}) "
              f"{labels[1]}={vb} ({COLOR_NAMES.get(vb, '?')})")
    if len(entries) > limit:
        print(f"  … {len(entries) - limit} more")


def png(grid: Grid, path: str | os.PathLike[str], cell: int = 24) -> str | None:
    """Render a grid to a PNG you can then open with the Read tool.

    Returns the path, or None when Pillow is unavailable. Useful for looking at
    a predicted output the way you looked at the puzzle: as a picture.
    """
    if not grid:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # The interpreter running this script has no Pillow, but the one that
        # built the workspace did — it rendered task/images/. Borrow it rather
        # than making the doctrine's "look at your prediction" advice a dead end.
        return _png_via_harness_python(grid, path, cell)

    height, width = len(grid), len(grid[0])
    image = Image.new("RGB", (width * cell, height * cell), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for r in range(height):
        for c in range(width):
            draw.rectangle(
                [c * cell, r * cell, c * cell + cell, r * cell + cell],
                fill=_PALETTE.get(int(grid[r][c]), (128, 128, 128)),
                outline=(50, 50, 50),
            )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG", compress_level=1)
    return str(target)


def _png_via_harness_python(grid: Grid, path: str | os.PathLike[str], cell: int) -> str | None:
    """Render through the interpreter that built this workspace."""
    import subprocess

    recorded = WORKSPACE / ".athanor" / "harness_python"
    if not recorded.is_file():
        print("Pillow is not installed here and no fallback interpreter was recorded.")
        return None
    executable = recorded.read_text(encoding="utf-8").strip()
    if not executable:
        print("Pillow is not installed here and no fallback interpreter was recorded.")
        return None

    payload = json.dumps({"grid": grid, "path": str(path), "cell": int(cell), "palette": _PALETTE})
    source = (
        "import json,sys\n"
        "from PIL import Image, ImageDraw\n"
        "a=json.loads(sys.stdin.read())\n"
        "g=a['grid']; c=a['cell']; pal={int(k):tuple(v) for k,v in a['palette'].items()}\n"
        "h,w=len(g),len(g[0])\n"
        "im=Image.new('RGB',(w*c,h*c),(255,255,255)); d=ImageDraw.Draw(im)\n"
        "for r in range(h):\n"
        "    for x in range(w):\n"
        "        d.rectangle([x*c,r*c,x*c+c,r*c+c],fill=pal.get(int(g[r][x]),(128,128,128)),outline=(50,50,50))\n"
        "import os\n"
        "os.makedirs(os.path.dirname(os.path.abspath(a['path'])) or '.',exist_ok=True)\n"
        "im.save(a['path'],format='PNG',compress_level=1)\n"
    )
    try:
        done = subprocess.run(
            [executable, "-c", source], input=payload, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Could not render PNG via {executable}: {exc}")
        return None
    if done.returncode != 0:
        print(f"Could not render PNG via {executable}: {(done.stderr or '').strip()[:200]}")
        return None
    return str(path)


def _pairs() -> Iterable[tuple[Grid, Grid]]:
    for sample in train_samples:
        yield sample["input"], sample["output"]


if __name__ == "__main__":  # `python arc.py` prints a quick orientation dump
    print(f"workspace: {WORKSPACE}")
    print(f"{len(train_samples)} training pairs, {len(test_samples)} test input(s)\n")
    for i, (inp, out) in enumerate(_pairs()):
        print(f"train {i}: {shape(inp)} -> {shape(out)}   "
              f"colours in {colors(inp)} -> out {colors(out)}")
    for i, sample in enumerate(test_samples):
        print(f"test  {i}: {shape(sample['input'])}          colours {colors(sample['input'])}")
