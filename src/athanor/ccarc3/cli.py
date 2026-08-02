"""Command line for CCARC3.

    athanor ccarc3 games                     # what is available, with baselines
    athanor ccarc3 run  --game ls20-9607627b # one solver session
    athanor ccarc3 batch --games a,b,c       # several, sequentially
    athanor ccarc3 report --out-dir runs/    # what finished, read off the traces
    athanor ccarc3 trace --run runs/ls20-…   # what actually happened in one run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import list_games
from .ledger import load
from .session import Ccarc3Config, build_workspace, collect_outcome, run_game

__all__ = ["add_arguments", "build_parser", "main"]


def _config(args: argparse.Namespace, game_id: str) -> Ccarc3Config:
    return Ccarc3Config(
        game_id=game_id,
        out_dir=Path(args.out_dir),
        model=args.model,
        effort=args.effort,
        budget_multiple=args.budget_multiple,
        wall_clock_timeout_s=args.timeout,
        fresh=getattr(args, "fresh", False),
    )


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", default="runs/ccarc3")
    p.add_argument("--model", default=Ccarc3Config("x").model)
    p.add_argument("--effort", default=Ccarc3Config("x").effort)
    p.add_argument("--budget-multiple", type=float, default=4.0,
                   help="action cap as a multiple of the game's published baseline")
    p.add_argument("--timeout", type=float, default=7200.0)
    p.add_argument("--fresh", action="store_true",
                   help="discard any existing trace and start over "
                        "(default is to resume an interrupted run)")


def cmd_games(args: argparse.Namespace) -> int:
    games = sorted(list_games(), key=lambda g: -g.baseline_total)
    print(f"{'game':24}{'tags':17}{'lvls':>5}{'baseline':>10}{'budget':>9}")
    for g in games:
        print(
            f"{g.game_id:24}{','.join(g.tags) or '-':17}{g.levels:5}"
            f"{g.baseline_total:10}{g.suggested_budget(args.budget_multiple):9}"
        )
    print(f"\n{len(games)} games. Baselines are actions for a playthrough that "
          f"already knows the rules.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    out = run_game(_config(args, args.game))
    print(json.dumps(out, indent=2))
    return 0 if out.get("levels_reached", 0) > 0 else 1


def cmd_batch(args: argparse.Namespace) -> int:
    ids = [g.strip() for g in args.games.split(",") if g.strip()]
    results = []
    for i, game_id in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}] {game_id}", flush=True)
        try:
            results.append(run_game(_config(args, game_id)))
        except Exception as exc:  # noqa: BLE001 -- one bad game must not end the batch
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
            results.append({"game_id": game_id, "error": str(exc)})
    _summarise(results)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Summarise finished runs by **re-deriving** from their traces.

    `result.json` is what the run computed at the time, and a later harness fix
    can make it wrong: every stored `ls20` figure still read 860 actions and
    zero full resets after full resets became detectable. The trace is the
    record, so anything derivable is derived again here and the stored file
    supplies only what the ledger cannot know — exit code, timeout, rule counts.
    """
    results = _read_runs(Path(args.out_dir))
    if not results:
        print(f"no finished runs under {args.out_dir}")
        return 1
    _summarise(results)
    if getattr(args, "against", None):
        _compare(_read_runs(Path(args.against)), results, Path(args.against))
    return 0


def _read_runs(root: Path) -> list[dict]:
    from .session import ledger_facts, run_cost

    out = []
    for p in sorted(root.glob("*/result.json")):
        stored = json.loads(p.read_text(encoding="utf-8"))
        trace = p.parent / "trace.jsonl"
        fresh = {**ledger_facts(trace), **run_cost(p.parent / "stream.jsonl")} if trace.exists() else {}
        meta = p.parent / "meta.json"
        if trace.exists() and meta.exists():
            baselines = json.loads(meta.read_text(encoding="utf-8")).get("baseline_actions")
            if baselines:
                from .ledger import load as _load
                from .scoring import score_run
                try:
                    fresh["rhae"] = score_run(_load(trace), baselines).score
                except ValueError:
                    pass          # a trace the rubric cannot score is not a crash
        out.append({**stored, **fresh})
    return out


def _compare(before: list[dict], after: list[dict], label: Path) -> None:
    """Pair games present in both directories and show the change.

    Operator tooling, not solver tooling. The same comparison was written by
    hand three times in one session -- levels, actions, and effectiveness for
    one game across two batches -- which is the demand signal that justifies it.
    Nothing here is offered to a solver; §9.8a is about what *they* ignore.
    """
    b = {r.get("game_id"): r for r in before if "error" not in r}
    pairs = [(b[r["game_id"]], r) for r in after
             if "error" not in r and r.get("game_id") in b]
    if not pairs:
        print(f"\nnothing in {label} to compare against.")
        return
    print(f"\npaired against {label} — before -> after")
    print(f"{'game':24}{'levels':>14}{'actions':>16}{'vs base':>16}")
    for x, y in sorted(pairs, key=lambda p: p[1]["game_id"]):
        base = y.get("baseline_total") or 0
        cx = x.get("actions_final_playthrough", x.get("actions_used", 0))
        cy = y.get("actions_final_playthrough", y.get("actions_used", 0))
        # Same rule as _summarise: levels and actions must describe one playthrough.
        lx = x.get("levels_reached_final_playthrough", x.get("levels_reached", 0))
        ly = y.get("levels_reached_final_playthrough", y.get("levels_reached", 0))
        lv = f"{lx}->{ly}/{y.get('levels_total',0)}"
        act = f"{cx}->{cy}"
        rat = f"{cx/base:.2f}x->{cy/base:.2f}x" if base else "-"
        print(f"{y['game_id']:24}{lv:>14}{act:>16}{rat:>16}")
    only = [r["game_id"] for r in after if "error" not in r and r.get("game_id") not in b]
    if only:
        print(f"  not in {label}, so unpaired: {', '.join(sorted(only))}")


def _summarise(results: list[dict]) -> None:
    """Report per game, and never a bare pooled rate.

    Games differ enormously in length -- 171 to 1843 baseline actions -- so a
    pooled "levels solved" number mostly reports which games were in the batch.
    Levels reached against levels available is the comparable figure.
    """
    ok = [r for r in results if "error" not in r]
    print(f"\n{'game':24}{'levels':>10}{'actions':>9}{'vs base':>9}{'deaths':>7}"
          f"{'turns':>7}{'cost':>8}  flags")
    for r in sorted(ok, key=lambda x: x.get("game_id", "")):
        total = r.get("levels_total", 0)
        used, base = r.get("actions_used", 0), r.get("baseline_total", 0)
        # Against the baseline, only the final playthrough is comparable: a full
        # reset means earlier actions bought progress that was then discarded,
        # and charging them to the result overstates the cost by whatever was
        # replayed. The total is still shown -- it is what the budget paid.
        charged = r.get("actions_final_playthrough", used)
        ratio = f"{charged / base:.2f}x" if base else "-"
        # **Levels must come from the same playthrough as the actions.** Pairing
        # `levels_reached` (the maximum over *every* playthrough) with a cost
        # restricted to the final one credits a run with progress a full reset
        # destroyed, at the price of the progress that survived -- the flattering
        # half of each. `ledger_facts` has computed the right figure since full
        # resets became detectable and nothing read it.
        reached = r.get("levels_reached_final_playthrough", r.get("levels_reached", 0))
        flags = " ".join(
            f
            for f, on in (
                ("WON", r.get("won")),
                ("TIMEOUT", r.get("timed_out")),
                (f"wasted={r.get('wasted_actions')}", r.get("wasted_actions")),
                (
                    f"FULLRESET={r.get('full_resets')} (ratio is the last "
                    f"{charged} of {used})",
                    r.get("full_resets"),
                ),
            )
            if on
        )
        spend = f"${r['cost_usd']:.2f}" if r.get("cost_usd") else "-"
        print(f"{r.get('game_id',''):24}{f'{reached}/{total}':>10}{used:9}{ratio:>9}"
              f"{r.get('deaths',0):7}{r.get('turns') or '-':>7}{spend:>8}  {flags}")

    failed = [r for r in results if "error" in r]
    for r in failed:
        print(f"{r['game_id']:24}{'ERROR':>10}  {r['error'][:60]}")

    solved = sum(
        r.get("levels_reached_final_playthrough", r.get("levels_reached", 0)) for r in ok
    )
    avail = sum(r.get("levels_total", 0) for r in ok)
    print(f"\n{len(ok)} runs, {solved}/{avail} levels reached, "
          f"{sum(1 for r in ok if r.get('won'))} games won.")
    if failed:
        print(f"{len(failed)} run(s) failed outright and are excluded above.")
    _rhae(ok)


PUBLIC_ENVIRONMENTS = 25
"""Size of the ARC-AGI-3 public demo set — the benchmark's denominator."""

OPUS5_PUBLISHED = 30.16
"""Claude Opus 5 on the ARC-AGI-3 leaderboard, 24 Jul 2026, High effort."""


def _rhae(results: list[dict]) -> None:
    """Print the official RHAE score, so it is never recomputed by hand.

    Unattempted environments count as zero: the benchmark divides by the whole
    set, and scoring only what was played is the easiest way to overstate a
    result here.
    """
    from .scoring import total_score

    scores = {}
    for r in results:
        score = r.get("rhae")
        if score is None:
            continue
        scores[r["game_id"]] = max(score, scores.get(r["game_id"], 0.0))
    if not scores:
        return
    padded = list(scores.values()) + [0.0] * max(0, PUBLIC_ENVIRONMENTS - len(scores))
    total = total_score(padded)
    played = 100.0 * sum(scores.values()) / len(scores)
    print(f"RHAE: {total:.2f}% over all {PUBLIC_ENVIRONMENTS} public environments "
          f"({len(scores)} scored, {played:.2f}% mean on those). "
          f"Opus 5 published: {OPUS5_PUBLISHED}%.")
    gap = OPUS5_PUBLISHED / 100 * PUBLIC_ENVIRONMENTS - sum(scores.values())
    if gap > 0:
        print(f"      {gap:.3f} environment-units behind — about "
              f"{gap:.2f} more full wins.")
    else:
        print(f"      ahead by {-gap:.3f} environment-units.")


def cmd_trace(args: argparse.Namespace) -> int:
    run = Path(args.run)
    transitions = load(run / "trace.jsonl")
    if not transitions:
        print(f"no trace in {run}")
        return 1
    print(f"{len(transitions)} actions")
    per_level: dict[int, int] = {}
    for t in transitions:
        per_level[t.level] = per_level.get(t.level, 0) + 1
    print("actions per level:", dict(sorted(per_level.items())))
    print("action mix:", {
        a: sum(1 for t in transitions if t.action == a)
        for a in sorted({t.action for t in transitions})
    })
    print("changed the board:", sum(t.changed for t in transitions))
    print("wasted (issued while dead):", sum(t.wasted for t in transitions))
    rules = run / "rules.json"
    if rules.exists():
        book = json.loads(rules.read_text(encoding="utf-8"))
        print(f"rule book: {len(book.get('verified', []))} mechanics, "
              f"{len(book.get('refuted', []))} refutations, "
              f"{len(book.get('open_questions', []))} open")
    return 0


def cmd_workspace(args: argparse.Namespace) -> int:
    ws = build_workspace(_config(args, args.game))
    print(ws.root)
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="ccarc3_command", required=True)

    games = sub.add_parser("games", help="List public games with baselines.")
    _common(games)
    games.set_defaults(func=cmd_games)

    run = sub.add_parser("run", help="Play one game with Claude Code as the agent loop.")
    run.add_argument("--game", required=True)
    _common(run)
    run.set_defaults(func=cmd_run)

    batch = sub.add_parser("batch", help="Play several games sequentially.")
    batch.add_argument("--games", required=True, help="comma-separated game ids")
    _common(batch)
    batch.set_defaults(func=cmd_batch)

    report = sub.add_parser("report", help="Summarise finished runs from their traces.")
    report.add_argument("--out-dir", default="runs/ccarc3")
    report.add_argument(
        "--against",
        help="an earlier run directory; games in both are shown paired, with deltas",
    )
    report.set_defaults(func=cmd_report)

    trace = sub.add_parser("trace", help="Describe what happened in one run.")
    trace.add_argument("--run", required=True)
    trace.set_defaults(func=cmd_trace)

    workspace = sub.add_parser("workspace", help="Build a workspace without running it.")
    workspace.add_argument("--game", required=True)
    _common(workspace)
    workspace.set_defaults(func=cmd_workspace)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="athanor ccarc3", description=__doc__)
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
