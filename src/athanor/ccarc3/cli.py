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
    root = Path(args.out_dir)
    results = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(root.glob("*/result.json"))
    ]
    if not results:
        print(f"no finished runs under {root}")
        return 1
    _summarise(results)
    return 0


def _summarise(results: list[dict]) -> None:
    """Report per game, and never a bare pooled rate.

    Games differ enormously in length -- 171 to 1843 baseline actions -- so a
    pooled "levels solved" number mostly reports which games were in the batch.
    Levels reached against levels available is the comparable figure.
    """
    ok = [r for r in results if "error" not in r]
    print(f"\n{'game':24}{'levels':>10}{'actions':>9}{'vs base':>9}{'deaths':>8}  flags")
    for r in sorted(ok, key=lambda x: x.get("game_id", "")):
        reached, total = r.get("levels_reached", 0), r.get("levels_total", 0)
        used, base = r.get("actions_used", 0), r.get("baseline_total", 0)
        ratio = f"{used / base:.2f}x" if base else "-"
        flags = " ".join(
            f
            for f, on in (
                ("WON", r.get("won")),
                ("TIMEOUT", r.get("timed_out")),
                (f"wasted={r.get('wasted_actions')}", r.get("wasted_actions")),
                (f"FULLRESET={r.get('full_resets')}", r.get("full_resets")),
            )
            if on
        )
        print(f"{r.get('game_id',''):24}{f'{reached}/{total}':>10}{used:9}{ratio:>9}"
              f"{r.get('deaths',0):8}  {flags}")

    failed = [r for r in results if "error" in r]
    for r in failed:
        print(f"{r['game_id']:24}{'ERROR':>10}  {r['error'][:60]}")

    solved = sum(r.get("levels_reached", 0) for r in ok)
    avail = sum(r.get("levels_total", 0) for r in ok)
    print(f"\n{len(ok)} runs, {solved}/{avail} levels reached, "
          f"{sum(1 for r in ok if r.get('won'))} games won.")
    if failed:
        print(f"{len(failed)} run(s) failed outright and are excluded above.")


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
