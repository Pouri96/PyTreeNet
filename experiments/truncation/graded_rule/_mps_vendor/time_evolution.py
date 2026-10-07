"""
The time-evolution driver and the local-solver selection, re-exported from PyTreeNet.

Only the four names the package actually builds on: ``TimeEvolution`` and ``TimeEvoConfig``
are what :mod:`rage.solvers.ttn_time_evolution` extends, ``TimeEvoMode`` selects the local
solver and ``EvoDirection`` its direction. The solver's own entry points
(``time_evolve``, ``krylov_exp_action``) are not re-exported: nothing here calls them, and
they remain importable from their own module.
"""
from pytreenet.time_evolution.time_evolution import (
    TimeEvolution,
    TimeEvoConfig,
    EvoDirection,
    TimeEvoMode,
)

__all__ = ["TimeEvolution", "TimeEvoConfig", "EvoDirection", "TimeEvoMode"]
