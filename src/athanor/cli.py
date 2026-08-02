"""Command-line interface for Athanor.

The `web` subcommand launches the batch dashboard from
``athanor.web_demo.batch_launcher``. The dashboard starts
empty when no `--tasks` are given and supports adding more puzzle
instances interactively via the "+" button.

The `cc` subcommand runs the Claude Code harness variant, where Claude Code
owns the agent loop and Athanor supplies only the workspace and the
verification gate. See ``docs/cc_harness.md``.
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

    try:
        from athanor.web_demo.batch_launcher import add_arguments as _add_batch_arguments
    except ImportError:
        # The web dashboard pulls in the full model/serving stack. The `cc`
        # subcommand needs none of it, so a partial install still gets a usable CLI.
        _add_batch_arguments = None

    if _add_batch_arguments is not None:
        web = subparsers.add_parser(
            "web",
            help="Run the batch dashboard (empty dashboard when --tasks is omitted)",
        )
        _add_batch_arguments(web)
        web.set_defaults(func=_cmd_web)

    from athanor.cc_harness.cli import add_arguments as _add_cc_arguments

    cc = subparsers.add_parser(
        "cc",
        help="Claude Code harness variant (single solver agent, no reviewer)",
    )
    _add_cc_arguments(cc)

    from athanor.ccarc3.cli import add_arguments as _add_ccarc3_arguments

    ccarc3 = subparsers.add_parser(
        "ccarc3",
        help="ARC-AGI-3 harness variant (interactive games, level gate, no reviewer)",
    )
    _add_ccarc3_arguments(ccarc3)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
