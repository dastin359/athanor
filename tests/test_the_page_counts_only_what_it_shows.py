"""The headline figures and the trim note must describe the rendered page.

`fit()` drops the spans of every `void` run before rendering -- ten contaminated
runs cost 9.8 MB of a 16 MB budget, and their tiles stay while their trees go.
Two things then counted the dropped spans anyway:

* `summarise()`, whose own docstring says it derives "the spans actually on the
  page", was handed the unfiltered store. The bash-call count, the median Bash
  duration and the LLM share all absorbed runs that no reader can find.
* the "N payloads trimmed to C chars" note counted payloads inside dropped runs
  as trimmed, when they were not trimmed -- they were removed.

Both agree with the truth on the default path, because `current_generation_only`
has already discarded the void runs by then, so `drop_spans` is empty. They part
company under `--all-generations`, which is the flag you reach for precisely when
you want to see the contaminated runs. A figure that is right only when the thing
it corrects for is absent is the recurring shape in this repo.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import build_trace_audit as bta  # noqa: E402

TEMPLATE = (
    "<html><body>__DATA_JSON__|__RUNS_JSON__|games=__NGAMES__|"
    "bash=__BASH_CALLS__|median=__BASH_MEDIAN__|llm=__LLM_PCT__|"
    "live=__LIVE_JSON__|built=__BUILT_AT__|excluded=__EXCLUDED__</body></html>"
)


def _entry(bash_durs: list[float], payload_chars: int = 10) -> dict:
    spans = [{"id": "root", "kind": "CHAIN", "name": "session", "dur": 1.0}]
    for i, dur in enumerate(bash_durs):
        spans.append({
            "id": f"tool:{i}", "kind": "TOOL", "name": "Bash", "dur": dur,
            "out": "x" * payload_chars, "input": "{}", "err": False,
        })
    spans.append({"id": "turn:1", "kind": "LLM", "name": "turn 1", "dur": 5.0})
    return {"result": {"game_id": "zz00-00000000"},
            "attempts": [{"file": "stream.jsonl", "spans": spans,
                          "unparseable": 0, "lines": len(spans)}]}


def test_a_void_run_is_absent_from_the_headline_figures(monkeypatch):
    """Its spans are not on the page, so they must not be in the numbers."""
    monkeypatch.setattr(bta, "live_games", lambda: [])
    data = {"good": _entry([2.0]), "bad": _entry([100.0, 100.0, 100.0])}
    runs = [{"id": "good", "tier": "current"}, {"id": "bad", "tier": "void"}]

    page, _trimmed, _cap = bta.fit(data, runs, TEMPLATE)

    assert "games=1" in page, (
        "the void run was counted as a game whose spans the page does not carry"
    )
    assert "bash=1" in page, "the void run's three Bash calls reached the count"
    assert "median=2.00" in page, (
        "the median was taken over spans that were dropped before rendering"
    )
    body = page.split("|")[0]
    assert '"bad"' not in body, "the void run's spans were rendered after all"


def test_the_trim_note_does_not_count_payloads_that_were_dropped(monkeypatch):
    """A payload inside a dropped run was not trimmed; it was removed."""
    monkeypatch.setattr(bta, "live_games", lambda: [])
    monkeypatch.setattr(bta, "TARGET", 4096)

    # One superseded run big enough to force the ladder, and one void run whose
    # payloads are the same size -- so a count over the unfiltered store doubles.
    data = {"old": _entry([1.0] * 3, payload_chars=9000),
            "bad": _entry([1.0] * 3, payload_chars=9000)}
    runs = [{"id": "old", "tier": "superseded"}, {"id": "bad", "tier": "void"}]

    page, trimmed, cap = bta.fit(data, runs, TEMPLATE)

    assert cap is not None, "the fixture did not force the trim ladder at all"
    rendered = json.loads(page.split("|")[0].replace("<html><body>", ""))
    really_trimmed = sum(
        1 for entry in rendered.values() for a in entry["attempts"]
        for s in a["spans"] for f in ("out", "input", "text")
        if isinstance(s.get(f), str) and s[f].endswith("…[trimmed]")
    )
    assert trimmed == really_trimmed, (
        f"reported {trimmed} payloads trimmed; the page carries "
        f"{really_trimmed} trimmed payloads"
    )


def test_the_default_path_is_unchanged(monkeypatch):
    """With no void runs, `shown` is the store and every figure is as before."""
    monkeypatch.setattr(bta, "live_games", lambda: [])
    data = {"a": _entry([2.0]), "b": _entry([4.0])}
    runs = [{"id": "a", "tier": "current"}, {"id": "b", "tier": "superseded"}]

    page, trimmed, cap = bta.fit(data, runs, TEMPLATE)

    assert "games=2" in page and "bash=2" in page and "median=3.00" in page
    assert (trimmed, cap) == (0, None)
