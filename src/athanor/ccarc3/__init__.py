"""CCARC3 — harness primitives for a Claude-Code-as-agent ARC-AGI-3 solver.

Design: ``docs/ccarc3_design.md``.

The split this package exists to enforce: **the harness ships the converter, the
solver writes the analysis.** Every run would otherwise re-derive the same
loader differently, destroying cross-run comparability, and conversion is where
a silent bug (dropping the intermediate frames of a multi-frame action,
mis-inferring the render block size) poisons every downstream conclusion
invisibly.

Nothing here imports ``arc_agi_3``, so all of it is usable and testable without
the SDK or an API key. The seam is a plain dict of the shape
``FrameData.model_dump()`` produces.
"""

from .grids import (
    DEFAULT_BACKGROUND,
    PALETTE,
    PALETTE_SIZE,
    Change,
    Object,
    as_grid,
    block_size,
    cell_boundaries,
    collapse,
    counts,
    diff,
    flatten_frames,
    logical,
    objects,
    png,
    render,
)
from .client import (
    ROOT_URL,
    ActionRefused,
    ArcClient,
    GameInfo,
    list_games,
)
from .gate import GateRefusal, LevelGate
from .ledger import (
    ACTION_NAMES,
    TraceWriter,
    Transition,
    action_name,
    infer_levels,
    load,
)
from .session import Ccarc3Config, build_workspace, run_game
from .rules import (
    Counts,
    Outcome,
    Rule,
    RuleBook,
    Survey,
    VerifyResult,
    regressions,
    survey,
    verify,
)

__all__ = [
    # grids
    "PALETTE_SIZE",
    "DEFAULT_BACKGROUND",
    "Change",
    "Object",
    "as_grid",
    "render",
    "diff",
    "block_size",
    "logical",
    "collapse",
    "cell_boundaries",
    "png",
    "PALETTE",
    "objects",
    "counts",
    "flatten_frames",
    # client
    "ROOT_URL",
    "ArcClient",
    "GameInfo",
    "ActionRefused",
    "list_games",
    # gate
    "LevelGate",
    "GateRefusal",
    # session
    "Ccarc3Config",
    "build_workspace",
    "run_game",
    # ledger
    "ACTION_NAMES",
    "Transition",
    "TraceWriter",
    "load",
    "action_name",
    "infer_levels",
    # rules
    "Outcome",
    "Rule",
    "Counts",
    "VerifyResult",
    "Survey",
    "verify",
    "survey",
    "regressions",
    "RuleBook",
]
