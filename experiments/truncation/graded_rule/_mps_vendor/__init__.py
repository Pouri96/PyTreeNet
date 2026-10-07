"""RAGE_MPS and BUG_MPS, copied here so the experiments run without the ``rage`` package.

The files are ``rage.integrators.{BUG_MPS,RAGE_MPS}`` and the ``rage`` modules they use, flattened into
one package with their relative imports rewritten. The one other change is in ``bug_util``, where the
open-system truncation policies (absent from this branch's ``pytreenet``) are local stand-ins.
"""

import _pytreenet_root  # noqa: F401  (side-effecting: puts the right pytreenet first)

from .BUG_MPS import BUG_MPS
from .RAGE_MPS import RAGE_MPS
from .bug_util import RAGEMPSConfig
from .time_evolution import TimeEvoMode

__all__ = ["BUG_MPS", "RAGE_MPS", "RAGEMPSConfig", "TimeEvoMode"]
