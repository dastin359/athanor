"""Docstrings are a solver-visible surface, and nobody was treating them as one.

The harness spends real machinery keeping the human baselines away from the
solver: a strip that rewrites every workspace file, a proxy that holds the API
key so `GET /api/games` is unreachable, and a redaction pass over the doctrine.
All of it guards *files*.

**`help()` is not a file.** The solver is handed `client`, `gate` and `arc` by
name in `session.py`, and `help(client)` / `dir(arc)` / `arc.gate.__doc__` are
the standard way to find out what an unfamiliar object does — exactly what an
orienting agent does first. `pydoc` renders every public docstring in the
package, and those docstrings were written for the operator. They carried real
published baseline totals, the name of the private attribute holding the
withheld number, the design of the strip itself, and the allowance framing the
doctrine was rewritten to remove.

This is not hypothetical browsing: `client.py`'s own note records `cd82` reading
`baselines_for` straight out of `dir(arc)` "having gone looking for nothing at
all".

So the rule these tests enforce: **anything `help()` can reach is solver-facing
text.** Rationale, measured figures from other environments, and the reasoning
behind the withholding all belong in `#` comments, which `help()` cannot see.
Docstrings state what the thing does and nothing else.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

import pytest

import athanor.ccarc3 as arc

# Real published per-game baseline totals and per-level medians that have
# appeared in this package's prose. Any of them in a reachable docstring is a
# direct disclosure of the quantity the whole strip exists to withhold.
BASELINE_NUMBERS = re.compile(r"\b(1843|171|388|361|317|218|518|2590)\b")

# Allowance vocabulary. `cost`/`cheap` are deliberately absent: the doctrine
# argues from cost all the time ("dying is cheap") and that framing is correct.
# What is banned is an *account* — a quantity that depletes.
ALLOWANCE = re.compile(
    r"\bbudget\b|\ballowance\b|\bunaffordable\b|\bafford\b|\baction cap\b"
    r"|\bactions allowed\b|\bration\b",
    re.I,
)

# Naming the machinery that hides things is itself a map to it.
STRIP_MAP = re.compile(
    r"_baseline_here_enforced|hide_baselines|arc_proxy|baselines_for"
    r"|withh(?:old|eld)|deliberately not the same|may not see",
    re.I,
)


def _modules():
    mods = [arc]
    for info in pkgutil.iter_modules(arc.__path__):
        mods.append(importlib.import_module(f"athanor.ccarc3.{info.name}"))
    return mods


def reachable_docs() -> list[tuple[str, str]]:
    """Every ``(where, docstring)`` a solver can reach through introspection.

    Modules, then their public classes and functions, then those classes' public
    attributes — which is what ``pydoc.render_doc`` walks, and therefore what
    ``help()`` prints.
    """
    out: list[tuple[str, str]] = []
    seen: set[int] = set()
    for mod in _modules():
        if mod.__doc__:
            out.append((f"{mod.__name__}:<module>", mod.__doc__))
        for name, obj in vars(mod).items():
            if name.startswith("_") or id(obj) in seen:
                continue
            seen.add(id(obj))
            own = getattr(obj, "__module__", "") or ""
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and own.startswith("athanor"):
                if obj.__doc__:
                    out.append((f"{mod.__name__}:{name}", obj.__doc__))
                if inspect.isclass(obj):
                    for attr, value in vars(obj).items():
                        if attr.startswith("_"):
                            continue
                        doc = getattr(value, "__doc__", None)
                        # property/staticmethod wrappers inherit object.__doc__
                        # variants; only count text this class actually wrote.
                        if doc and doc not in (type(value).__doc__,):
                            out.append((f"{mod.__name__}:{name}.{attr}", doc))
    return out


def _hits(pattern: re.Pattern) -> list[str]:
    found = []
    for where, doc in reachable_docs():
        for m in pattern.finditer(doc):
            frag = re.sub(r"\s+", " ", doc[max(0, m.start() - 70):m.end() + 70])
            found.append(f"{where} [{m.group(0)}] ...{frag}...")
    return found


def test_no_reachable_docstring_prints_a_real_baseline():
    """`help(arc.GameInfo)` printed the envelope of every game's baseline total.

    It read "real games span 171 to 1843 baseline actions" — the range of the
    secret, across the public set, from the standard introspection path on a
    class the workspace `session.py` imports by name. `gate.py`'s module
    docstring named 1843 outright and called re-derivation "unaffordable".
    """
    hits = _hits(BASELINE_NUMBERS)
    assert not hits, "reachable docstrings disclose real baselines:\n" + "\n".join(hits)


def test_no_reachable_docstring_frames_actions_as_an_allowance():
    """The doctrine reframe has to reach the docstrings too, or it is cosmetic.

    A solver that reads "no budget was spent" in `help(arc.ActionRefused)` has
    been told a budget exists, by the harness, in its own voice — whatever the
    doctrine says three files away.
    """
    hits = _hits(ALLOWANCE)
    assert not hits, "reachable docstrings frame actions as an allowance:\n" + "\n".join(hits)


def test_no_reachable_docstring_maps_the_withholding():
    """Explaining *why* a number is hidden tells the solver it is hidden.

    `baseline_here`'s docstring said the property returns None "when the solver
    may not see it" and named `_baseline_here_enforced` as the attribute holding
    the real value — a pointer straight to `info.baseline_actions`.
    """
    hits = _hits(STRIP_MAP)
    assert not hits, "reachable docstrings describe the strip:\n" + "\n".join(hits)


@pytest.mark.parametrize("name", ["ArcClient", "GameInfo", "LevelGate"])
def test_the_objects_the_solver_holds_by_name_render_clean(name):
    """End-to-end through pydoc, the way a solver would actually reach it.

    The walk above enumerates docstrings; this renders the real page. If pydoc
    ever surfaces text the walk misses — inherited members, dataclass field
    docs — this catches it on the three objects `session.py` binds for the
    solver.
    """
    import pydoc  # noqa: PLC0415

    page = pydoc.render_doc(getattr(arc, name))
    for pattern, label in (
        (BASELINE_NUMBERS, "a real baseline"),
        (ALLOWANCE, "allowance framing"),
        (STRIP_MAP, "a map to the withholding"),
    ):
        found = [m.group(0) for m in pattern.finditer(page)]
        assert not found, f"help({name}) shows {label}: {sorted(set(found))}"
