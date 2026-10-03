"""RAGE's BUG integrator, copied here so the experiments run without the ``rage`` package.

The files are ``rage.integrators.BUG`` and the ``rage`` modules it uses, unchanged except for
their imports. pytreenet's own ``time_evolution/bug.py`` has no ``TruncationEngine``, so it has
no parameter budget or warm-up.
"""

import _pytreenet_root  # noqa: F401  (side-effecting: puts the right pytreenet first)

from .BUG import BUG
from .bug_util import BUGConfig
from .graph_ttn import ttns_from_tree
from pytreenet.time_evolution.time_evolution import TimeEvoMode

__all__ = ["BUG", "BUGConfig", "TimeEvoMode", "ttns_from_tree"]
