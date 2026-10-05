"""
Truncation policies of the fused Pauli frame.

A policy decides which directions a fixed bond budget keeps. Plain SVD keeps the largest
Frobenius weight. :class:`DMT` reserves the trace and the local Pauli strings so that
they survive every truncation exactly. :class:`ReweightedSVD` biases the SVD against
high-weight strings. The two are alternatives, so an integrator takes one policy object
and never both.

The classes hold data only, and the integrators in RAGE and GCG read them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

__all__ = ["TruncationPolicy", "PlainSVD", "DMT", "ReweightedSVD", "PLAIN"]


class TruncationPolicy:
    """Base class of the truncation policies."""


@dataclass(frozen=True)
class PlainSVD(TruncationPolicy):
    """Truncate by the singular values of each bond. The default."""


@dataclass(frozen=True)
class DMT(TruncationPolicy):
    """Density-matrix truncation (White, Zaletel, Mong, Refael, arXiv:1707.01506).

    The trace and the Pauli strings within ``radius`` qubits of each cut are reserved, so every
    operator on at most ``2 * radius + 1`` contiguous qubits survives every truncation exactly.
    On a tree the distance is the tree distance. The rest of the bond budget goes to the
    largest singular values of what remains. It needs a finite ``max_bond_dim`` above the
    reserved count.

    Attributes:
        radius: Qubits reserved on each side of a cut. 0 keeps the trace and the single-qubit
            marginals, 1 also every interaction of range 2. The reserved count is
            ``2 * 4**radius`` per bond on a chain and grows with the node degree on a tree.
        min_bond_dim: Smallest ``max_bond_dim`` the integrator accepts. ``None`` takes the
            reserved count plus two complement directions.
        respect_degenerate: Never split a degenerate singular-value multiplet at the cut.
        conserved_terms: ``{"bond_terms": ..., "site_terms": ...}``, the dictionaries that
            ``pauli_lindbladian_ttno`` takes. The observable they define is reserved at every
            cut, which makes its expectation exact at any interaction range. Chain only.
    """

    radius: int = 1
    min_bond_dim: Optional[int] = None
    respect_degenerate: bool = True
    conserved_terms: Optional[dict] = None

    def __post_init__(self):
        if isinstance(self.radius, bool) or not isinstance(self.radius, int) or self.radius < 0:
            raise ValueError(f"DMT radius must be an integer >= 0, got {self.radius!r}.")
        if self.min_bond_dim is not None and (
            isinstance(self.min_bond_dim, bool)
            or not isinstance(self.min_bond_dim, int)
            or self.min_bond_dim < 1
        ):
            raise ValueError(
                f"DMT min_bond_dim must be None or an integer >= 1, got {self.min_bond_dim!r}."
            )
        terms = self.conserved_terms
        if terms is not None:
            if not isinstance(terms, dict) or not terms:
                raise ValueError(
                    "DMT conserved_terms must be a non-empty dict with the keys "
                    "'bond_terms' and 'site_terms'."
                )
            unknown = set(terms) - {"bond_terms", "site_terms"}
            if unknown:
                raise ValueError(
                    f"DMT conserved_terms has unknown keys {sorted(unknown)}, "
                    "expected 'bond_terms' and 'site_terms'."
                )


@dataclass(frozen=True)
class ReweightedSVD(TruncationPolicy):
    """Reweighted truncation (Guha Roy and Slagle, arXiv:2412.08730).

    Every Pauli coefficient of weight ``n`` is stored divided by ``gamma**n``, so the SVD
    deprioritises high-weight strings. It guarantees nothing exactly. It applies at any bond
    budget, and needs an effective generator that is not symmetrised, so ``hermitian`` must be
    False in the config.

    ``gamma`` has no default, because the best value depends on the workload.

    Attributes:
        gamma: The reweighting factor, ``>= 1``. 1 is plain SVD.
    """

    gamma: float

    def __post_init__(self):
        if isinstance(self.gamma, bool) or not isinstance(self.gamma, (int, float)):
            raise ValueError(
                f"ReweightedSVD gamma must be a number, got {self.gamma!r}."
            )
        if not math.isfinite(self.gamma) or self.gamma <= 0.0:
            raise ValueError(
                f"ReweightedSVD gamma must be positive and finite, got {self.gamma}."
            )
        if self.gamma < 1.0:
            raise ValueError(
                f"ReweightedSVD gamma must be >= 1, got {self.gamma}: a smaller value "
                "prioritises high-weight strings."
            )


#: The default policy.
PLAIN = PlainSVD()
