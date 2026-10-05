"""
The fused Pauli frame: one site per qubit of dimension 4 holding the coefficients
``c_mu = Tr(sigma~_mu rho)`` of a density operator in the normalised Pauli basis.

``rho`` is Hermitian iff the network is real, ``Tr(rho)`` is a bilinear contraction with a bond-1
product cap, and ``sum_mu |c_mu|^2 = Tr(rho^dag rho)``. Every function here works on a chain
(``MatrixProductState``) and on a tree (``TreeTensorNetworkState``) alike. Qubit ``q`` sits on the
node ``f"{prefix}{q}"``, and nodes with a trivial open leg (virtual interior) carry no qubit.
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import List, Optional, Sequence

import numpy as np

from ...contractions.state_state_contraction import contract_two_ttns

__all__ = [
    "PAULIS", "U_PAULI", "pauli_rho_to_coeffs", "pauli_coeffs_to_rho", "pauli_site_vector",
    "pauli_site_ids", "assert_fused_pauli", "network_is_real", "realify_network",
    "pauli_trace_cap", "pauli_trace", "pauli_rescale_trace", "pauli_purity",
    "pauli_hermiticity_defect", "pauli_to_coeffs", "pauli_to_dense", "pauli_local_expectation",
    "pauli_gauge", "pauli_physical_state",
]

_I2 = np.eye(2, dtype=complex)
_PX = np.array([[0, 1], [1, 0]], dtype=complex)
_PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = (_I2, _PX, _PY, _PZ)
_SQ2 = np.sqrt(2.0)

#: ``U[mu, 2 i + j] = sigma~_mu[j, i]``: the per-qubit change of basis from the row-major
#: ``(ket, bra)`` pair index to the Pauli index. Unitary.
U_PAULI = np.array([[P[j, i] / _SQ2 for i in range(2) for j in range(2)] for P in PAULIS],
                   dtype=complex)

#: The identity-Pauli basis vector. ``Tr(rho) = (sqrt2)^N c_{I...I}``.
_CAP_VEC = np.array([1.0, 0.0, 0.0, 0.0])

_SUFFIX = re.compile(r"^(.*?)(\d+)$")

#: Attribute that records the reweighting gauge a state carries (see ``reweight.py``).
GAUGE_ATTR = "_pauli_gauge"


def pauli_gauge(net) -> float:
    """The reweighting gauge ``gamma`` that ``net`` carries, 1.0 for physical coordinates."""
    return float(getattr(net, GAUGE_ATTR, 1.0))


def _refuse_if_gauged(net, what: str) -> None:
    """Raise if ``net`` carries a reweighting gauge, whose coefficients are not physical.

    The trace, the Hermiticity defect and the bond dimensions are the same in both
    coordinates, and the other readouts are not.
    """
    gamma = pauli_gauge(net)
    if gamma != 1.0:
        raise ValueError(
            f"{what} is not defined on a network in reweighted coordinates (gamma={gamma:g}). "
            "Read it from the integrator's physical_state(), or pass the network through "
            "pauli_physical_state()."
        )


def pauli_physical_state(net):
    """A copy of ``net`` in physical coordinates, with the reweighting gauge removed."""
    from .reweight import pauli_reweight_network

    out = deepcopy(net)
    gamma = pauli_gauge(out)
    if gamma != 1.0:
        pauli_reweight_network(out, gamma, inverse=True)
    return out


# ---- dense <-> coefficient conversions (small N)

def pauli_rho_to_coeffs(rho: np.ndarray, num_qubits: int) -> np.ndarray:
    """Dense ``rho`` (``2^N x 2^N``, qubit 0 most significant) to the flat ``(4^N,)`` Pauli
    coefficients. Real up to rounding iff ``rho`` is Hermitian."""
    t = np.asarray(rho, dtype=complex).reshape((2,) * (2 * num_qubits))
    t = np.transpose(t, [x for q in range(num_qubits) for x in (q, q + num_qubits)])
    t = t.reshape((4,) * num_qubits)
    for q in range(num_qubits):
        t = np.moveaxis(np.tensordot(U_PAULI, t, axes=([1], [q])), 0, q)
    return t.reshape(-1)


def pauli_coeffs_to_rho(coeffs: np.ndarray, num_qubits: int) -> np.ndarray:
    """Inverse of :func:`pauli_rho_to_coeffs`: the dense ``2^N x 2^N`` density matrix."""
    t = np.asarray(coeffs, dtype=complex).reshape((4,) * num_qubits)
    for q in range(num_qubits):
        t = np.moveaxis(np.tensordot(U_PAULI.conj().T, t, axes=([1], [q])), 0, q)
    t = t.reshape((2,) * (2 * num_qubits))
    perm = [2 * q for q in range(num_qubits)] + [2 * q + 1 for q in range(num_qubits)]
    return np.transpose(t, perm).reshape(2 ** num_qubits, 2 ** num_qubits)


def pauli_site_vector(ket: np.ndarray) -> np.ndarray:
    """Coefficients ``<psi| sigma~_mu |psi>`` of the pure one-qubit state ``|psi><psi|``."""
    ket = np.asarray(ket, dtype=complex).reshape(2)
    ket = ket / np.linalg.norm(ket)
    return np.array([np.vdot(ket, pauli @ ket).real / _SQ2 for pauli in PAULIS])


# ---- structure

def _open_dim(net, node_id: str) -> int:
    dim = 1
    for leg in net.nodes[node_id].open_legs:
        dim *= int(net.tensors[node_id].shape[leg])
    return dim


def pauli_site_ids(net, prefix: Optional[str] = None) -> List[str]:
    """Identifiers of the qubit nodes of ``net`` in qubit order.

    The qubit nodes are the ones with open dimension 4. Their identifiers must be
    ``f"{prefix}{q}"`` for ``q = 0..N-1``.

    Args:
        net: A fused Pauli network.
        prefix: The expected identifier prefix. Read from the nodes when ``None``.

    Raises:
        ValueError: There is no dimension-4 node, or the identifiers do not follow the scheme.
    """
    ids = [nid for nid in net.nodes if _open_dim(net, nid) == 4]
    if not ids:
        raise ValueError("no dimension-4 open leg found: this is not a fused Pauli network.")
    parsed = {}
    for nid in ids:
        match = _SUFFIX.match(nid)
        if match is None:
            raise ValueError(f"qubit node {nid!r} does not end in a qubit index.")
        parsed[nid] = (match.group(1), int(match.group(2)))
    prefixes = {pfx for pfx, _ in parsed.values()}
    if len(prefixes) != 1 or (prefix is not None and prefixes != {prefix}):
        raise ValueError(f"qubit nodes must share one identifier prefix"
                         f"{'' if prefix is None else f' {prefix!r}'}, got {sorted(prefixes)}.")
    pfx = prefixes.pop()
    ordered = sorted(ids, key=lambda nid: parsed[nid][1])
    if ordered != [f"{pfx}{q}" for q in range(len(ordered))]:
        raise ValueError(f"qubit nodes must be {pfx}0..{pfx}{len(ordered) - 1}, got {ordered}.")
    return ordered


def assert_fused_pauli(net) -> List[str]:
    """Check that ``net`` is a fused Pauli state and return its qubit node identifiers.

    Every node has exactly one open leg, of dimension 4 on a qubit node and 1 on a virtual node.

    Raises:
        ValueError: A node has the wrong number of open legs, or an open dimension other than 1
            or 4. Dimension 2 means an ordinary pure state or a ket/bra network, not vec(rho).
    """
    for node_id in net.nodes:
        n_open = len(net.nodes[node_id].open_legs)
        if n_open != 1:
            raise ValueError(f"node {node_id!r} has {n_open} open legs, a fused Pauli state "
                             "has exactly one per node.")
        dim = _open_dim(net, node_id)
        if dim not in (1, 4):
            raise ValueError(
                f"node {node_id!r} has open dimension {dim}, a fused Pauli state carries 4 "
                "(one qubit's Pauli index) on a qubit node and 1 on a virtual one. Dimension 2 "
                "means a pure state or a ket/bra network, not a fused Pauli vec(rho).")
    return pauli_site_ids(net)


def network_is_real(net, tol: float = 1e-12) -> bool:
    """True iff every tensor of ``net`` is real to ``tol``. Works on states and operators."""
    return all(not np.iscomplexobj(t) or not np.asarray(t).size
               or np.max(np.abs(np.asarray(t).imag)) <= tol
               for t in net.tensors.values())


def realify_network(net, what: str, tol: float = 1e-12):
    """Cast every tensor of ``net`` to real in place and return it.

    Args:
        net: A state or an operator.
        what: Named in the error message.
        tol: Largest imaginary part accepted.

    Raises:
        ValueError: A tensor has an imaginary part above ``tol``.
    """
    bad = [(nid, float(np.max(np.abs(np.asarray(t).imag))))
           for nid, t in net.tensors.items()
           if np.iscomplexobj(t) and np.asarray(t).size
           and np.max(np.abs(np.asarray(t).imag)) > tol]
    if bad:
        worst = max(bad, key=lambda kv: kv[1])
        raise ValueError(
            f"{type(net).__name__} must be real-valued here ({what}): rho is Hermitian iff every "
            f"c_mu is real, but node {worst[0]} has max|Im| = {worst[1]:.3e} > {tol:.1e} "
            f"({len(bad)} node(s) affected).")
    for node_id, tensor in list(net.tensors.items()):
        tensor = np.asarray(tensor)
        if np.iscomplexobj(tensor):
            net.replace_tensor(node_id, tensor.real.copy(), new_shape=False)
    return net


# ---- trace, purity, Hermiticity

def pauli_trace_cap(net):
    """The trace cap on the topology of ``net``: ``sqrt2 * |I>`` on every dimension-4 open leg
    and 1 on every trivial one, all bonds of dimension 1. Contracting a state with it gives
    ``Tr(rho)``. Build it once and reuse it."""
    cap = deepcopy(net)
    if hasattr(cap, GAUGE_ATTR):
        delattr(cap, GAUGE_ATTR)
    for node_id in cap.nodes:
        node = cap.nodes[node_id]
        open_legs = list(node.open_legs)
        open_dim = _open_dim(cap, node_id)
        shape = [1] * (node.nneighbours() + len(open_legs))
        for leg in open_legs:
            shape[leg] = open_dim
        vec = _SQ2 * _CAP_VEC if open_dim == 4 else np.ones(open_dim)
        cap.replace_tensor(node_id, vec.reshape(shape), new_shape=True)
    return cap


def pauli_trace(net, cap=None) -> complex:
    """``Tr(rho)`` as the bilinear contraction with the trace cap. Assumes a real-valued state."""
    return complex(contract_two_ttns(net, pauli_trace_cap(net) if cap is None else cap))


def pauli_rescale_trace(net, trace: complex, floor: float = 1e-12) -> bool:
    """Divide one tensor of ``net`` by the real part of ``trace``, making ``Tr(rho) = 1``.

    The trace is linear in each tensor, so no canonical form is needed. The orthogonality centre
    is rescaled when set, the root otherwise.

    Returns:
        False, leaving ``net`` unchanged, when ``|trace|`` is non-finite or below ``floor``.
    """
    if not np.isfinite(trace) or abs(trace) < floor:
        return False
    scale = trace.real if abs(trace.real) >= 1e-300 else trace
    centre = net.orthogonality_center_id or net.root_id
    net.replace_tensor(centre, net.tensors[centre] / scale, new_shape=False)
    return True


def pauli_purity(net) -> float:
    """``sum_mu |c_mu|^2 = Tr(rho^dag rho)``, the squared norm of the network.

    Raises:
        ValueError: ``net`` carries a reweighting gauge.
    """
    _refuse_if_gauged(net, "The purity")
    return float(np.real(net.scalar_product()))


def pauli_hermiticity_defect(net) -> float:
    """``||rho - rho^dag||_F``. Exactly 0 for a real network. On a complex-typed network the
    value is resolved only to about ``1e-8``."""
    if all(np.isrealobj(tensor) for tensor in net.tensors.values()):
        return 0.0
    bilinear = complex(contract_two_ttns(net, deepcopy(net)))
    return float(np.sqrt(max(0.0, 2.0 * (pauli_purity(net) - bilinear.real))))


# ---- dense readout (small N)

def pauli_to_coeffs(net, site_ids: Optional[Sequence[str]] = None) -> np.ndarray:
    """The flat ``(4^N,)`` coefficient vector in qubit order. Exponential, small ``N`` only.

    Raises:
        ValueError: ``net`` carries a reweighting gauge.
    """
    _refuse_if_gauged(net, "The coefficient vector")
    site_ids = pauli_site_ids(net) if site_ids is None else list(site_ids)
    full, order = net.completely_contract_tree(to_copy=True)
    arr = np.asarray(full)
    sites = set(site_ids)
    phys_axes = [i for i, node_id in enumerate(order) if node_id in sites]
    other_axes = [i for i in range(arr.ndim) if i not in phys_axes]
    bad = [(order[i], arr.shape[i]) for i in other_axes if arr.shape[i] != 1]
    if bad:
        raise ValueError(f"non-trivial open legs outside the qubit nodes: {bad}")
    arr = np.transpose(arr, phys_axes + other_axes).reshape((4,) * len(site_ids))
    names = [order[i] for i in phys_axes]
    perm = [names.index(site_ids[q]) for q in range(len(site_ids))]
    return np.transpose(arr, perm).reshape(-1)


def pauli_to_dense(net, site_ids: Optional[Sequence[str]] = None) -> np.ndarray:
    """The dense ``2^N x 2^N`` density matrix. Exponential, small ``N`` only."""
    site_ids = pauli_site_ids(net) if site_ids is None else list(site_ids)
    return pauli_coeffs_to_rho(pauli_to_coeffs(net, site_ids), len(site_ids))


def pauli_local_expectation(net, op: np.ndarray, qubit: int, cap=None,
                            site_ids: Optional[Sequence[str]] = None) -> complex:
    """``Tr(op_qubit rho)`` for a one-qubit ``op``, by swapping one tensor of the trace cap.

    Raises:
        ValueError: ``net`` carries a reweighting gauge.
    """
    _refuse_if_gauged(net, "A local expectation value")
    site_ids = pauli_site_ids(net) if site_ids is None else list(site_ids)
    probe = deepcopy(pauli_trace_cap(net) if cap is None else cap)
    node_id = site_ids[qubit]
    op = np.asarray(op, dtype=complex)
    vec = np.array([np.trace(op @ P) / _SQ2 for P in PAULIS])
    probe.replace_tensor(node_id, vec.reshape(probe.tensors[node_id].shape), new_shape=False)
    return complex(contract_two_ttns(net, probe))
