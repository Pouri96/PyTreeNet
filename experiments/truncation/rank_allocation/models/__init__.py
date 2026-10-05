"""The physical models: tree, Hamiltonian (TTNO and sparse, from one definition), product
start and dense reference.

    spin_boson       ohmic spin-boson cells (used by :mod:`cells`)
    spread_models    pyrazine and Holstein cells (used by :mod:`cells`)
"""
import _pytreenet_root  # noqa: F401  (side-effecting: puts the right pytreenet first)

from . import spin_boson, spread_models

__all__ = ["spin_boson", "spread_models"]
