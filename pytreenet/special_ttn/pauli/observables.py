"""
Expectation values of a fused Pauli chain without building the dense matrix.

``Tr(O rho)`` is the bilinear contraction of the coefficient chain of ``rho`` with the
coefficient chain of ``O``, and a one-qubit expectation value is the same contraction with the
trace cap and one site vector swapped. Both cost ``O(N chi^2)`` and need no canonical form.
Both read the chain by its neighbour legs, so any root and any leg order is accepted. Neither
accepts a reweighted state, take ``physical_state()`` first.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .dmt import _obs_left_envs, _read_chain, pauli_operator_tensors
from .frame import PAULIS, _refuse_if_gauged, pauli_site_ids

__all__ = ["pauli_expectation", "pauli_local_expectation_sweep"]

_SQ2 = np.sqrt(2.0)
_CAP = _SQ2 * np.array([1.0, 0.0, 0.0, 0.0])
_TRACE_FLOOR = 1e-12


def _chain(net, site_ids: Optional[Sequence[str]], what: str):
    """The site tensors as ``(left, right, phys)`` in qubit order."""
    _refuse_if_gauged(net, what)
    site_ids = pauli_site_ids(net) if site_ids is None else list(site_ids)
    return _read_chain(net, site_ids)


def _trace_environments(tensors):
    """Left and right trace environments: ``left[k]`` covers sites ``< k``, ``right[k]`` covers
    sites ``>= k``, both as vectors on the bond between sites ``k - 1`` and ``k``."""
    left = [np.ones(1)]
    for tensor in tensors:
        left.append(np.einsum("l,lrp,p->r", left[-1], tensor, _CAP))
    right = [np.ones(1)]
    for tensor in reversed(tensors):
        right.append(np.einsum("lrp,p,r->l", tensor, _CAP, right[-1]))
    return left, right[::-1]


def _normalise(value, tensors, normalized: bool):
    """Divide by ``Tr(rho)`` when asked.

    Raises:
        ValueError: The trace is too small to divide by.
    """
    if not normalized:
        return value
    trace = _trace_environments(tensors)[0][-1][0]
    if abs(trace) < _TRACE_FLOOR:
        raise ValueError(f"Tr(rho) = {trace} is too small to normalise by.")
    return value / trace


def _real_if_real(value) -> Union[float, complex, np.ndarray]:
    """Drop the dtype of an exactly real result, so a real network gives floats."""
    return value.real if np.isrealobj(value) else value


def pauli_expectation(net, bond_terms: Optional[dict] = None, site_terms: Optional[dict] = None,
                      normalized: bool = False,
                      site_ids: Optional[Sequence[str]] = None) -> Union[float, complex]:
    """``Tr(O rho)`` for ``O = sum_q h_q + sum_(i,j) h_ij``, on a chain, without the dense matrix.

    The terms are the dictionaries that ``pauli_lindbladian_ttno`` and ``DMT(conserved_terms=...)``
    take, so the energy of a model is read with the terms that define it. A two-point correlator
    ``<Z_i Z_j>`` is the single bond term ``{(i, j): kron(Z, Z)}``.

    Args:
        net: A fused Pauli chain.
        bond_terms: ``{(i, j): h}`` with ``i < j`` and ``h`` a Hermitian 4x4 matrix in
            ``kron(qubit i, qubit j)`` order. The qubits need not be adjacent.
        site_terms: ``{q: h}`` with ``h`` a Hermitian 2x2 matrix.
        normalized: Divide by ``Tr(rho)``. Use it when the trace is not pinned.
        site_ids: The qubit nodes in qubit order. Found from the node names when omitted.

    Returns:
        The value, a float for a real network.

    Raises:
        ValueError: The network is not a chain in qubit order, carries a reweighting gauge, a
            term acts outside the chain or is not Hermitian, or ``Tr(rho)`` is too small to
            normalise by.
    """
    tensors = _chain(net, site_ids, "An expectation value")
    observable = pauli_operator_tensors(len(tensors), bond_terms, site_terms)
    value = _obs_left_envs(tensors, observable)[-1][0, 0]
    return _real_if_real(_normalise(value, tensors, normalized))


def pauli_local_expectation_sweep(net, op: np.ndarray, normalized: bool = False,
                                  site_ids: Optional[Sequence[str]] = None) -> np.ndarray:
    """``Tr(op_q rho)`` for every qubit ``q`` of a chain in one ``O(N chi^2)`` pass.

    Args:
        net: A fused Pauli chain.
        op: A one-qubit operator, a 2x2 matrix.
        normalized: Divide by ``Tr(rho)``. Use it when the trace is not pinned.
        site_ids: The qubit nodes in qubit order. Found from the node names when omitted.

    Returns:
        The ``(N,)`` array of values, real for a real network and a Hermitian ``op``.

    Raises:
        ValueError: ``op`` is not 2x2, the network is not a chain in qubit order or carries a
            reweighting gauge, or ``Tr(rho)`` is too small to normalise by.
    """
    op = np.asarray(op, dtype=complex)
    if op.shape != (2, 2):
        raise ValueError(f"op must be a 2x2 matrix, got shape {op.shape}.")
    tensors = _chain(net, site_ids, "A local expectation value")
    vec = np.array([np.trace(op @ P) / _SQ2 for P in PAULIS])
    vec = vec.real if np.max(np.abs(vec.imag)) < 1e-14 else vec
    left, right = _trace_environments(tensors)
    values = np.array([np.einsum("l,lrp,p,r->", left[q], tensors[q], vec, right[q + 1])
                       for q in range(len(tensors))])
    return _real_if_real(_normalise(values, tensors, normalized))
