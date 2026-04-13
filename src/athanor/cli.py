"""Command-line interface for Athanor.

The `web` subcommand launches the batch dashboard from
``athanor.web_demo.batch_launcher``. The dashboard starts
empty when no `--tasks` are given and supports adding more puzzle
instances interactively via the "+" button.
"""

from __future__ import annotations

import argparse


def _cmd_web(args: argparse.Namespace) -> int:
    from athanor.web_demo import batch_launcher
    return batch_launcher.run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="athanor",
        description="ARC solver based on iterative scientific discovery with executable experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from athanor.web_demo.batch_launcher import add_arguments as _add_batch_arguments

    web = subparsers.add_parser(
        "web",
        help="Run the batch dashboard (empty dashboard when --tasks is omitted)",
    )
    _add_batch_arguments(web)
    web.set_defaults(func=_cmd_web)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
