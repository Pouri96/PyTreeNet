"""
The ReweightedSVD gauge on the fused Pauli frame.

Guha Roy and Slagle, arXiv:2412.08730.

``G = diag(1, 1/gamma, 1/gamma, 1/gamma)`` acts on every Pauli leg, so a coefficient of weight
``n`` is stored divided by ``gamma**n``. The gauge is real, diagonal and a product over sites,
so it changes no bond dimension and keeps a real network real. The identity entry is 1, which
leaves the trace and the trace cap unchanged.

The state transforms as ``c -> G c`` and a generator as ``W -> G W G^-1``, so the pair
reproduces the physical dynamics: ``(G W G^-1)(G c) = G (W c)``. A state carries its gauge as an
attribute, and the readouts that depend on it refuse, see ``frame.py``.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .frame import GAUGE_ATTR, pauli_gauge

__all__ = ["pauli_reweight_matrix", "pauli_reweight_network", "pauli_reweight_ptm",
           "pauli_reweight_channels", "pauli_reweight_mpo"]


def pauli_reweight_matrix(gamma: float, inverse: bool = False) -> np.ndarray:
    """``G = diag(1, 1/gamma, 1/gamma, 1/gamma)`` on one Pauli leg, or ``G^-1``.

    Raises:
        ValueError: ``gamma`` is not positive.
    """
    if not gamma > 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}.")
    scale = gamma if inverse else 1.0 / gamma
    return np.diag([1.0, scale, scale, scale])


def pauli_reweight_network(net, gamma: float, inverse: bool = False):
    """Apply the gauge to every dimension-4 Pauli leg of ``net`` in place and return it.

    A state has one open leg per node and goes to ``G c``, and records the accumulated gauge.
    An operator has an ``(out, in)`` pair of open legs per node and goes to ``G W G^-1``. Open
    legs of dimension 1, on virtual nodes, are skipped.

    Args:
        net: A fused Pauli state or a superoperator TTNO.
        gamma: The reweighting factor, positive. 1 changes nothing.
        inverse: Apply ``G^-1`` on a state and ``G^-1 W G`` on an operator.

    Raises:
        ValueError: A node has more than two open legs.
    """
    mat = pauli_reweight_matrix(gamma, inverse)
    inv = pauli_reweight_matrix(gamma, not inverse)
    is_state = True
    for node_id, node in net.nodes.items():
        tensor = np.asarray(net.tensors[node_id])
        open_legs = list(node.open_legs)
        is_state = is_state and len(open_legs) == 1
        if not any(tensor.shape[ax] == 4 for ax in open_legs):
            continue
        if len(open_legs) == 1:
            axis = open_legs[0]
            tensor = np.moveaxis(np.tensordot(tensor, mat, axes=([axis], [1])), -1, axis)
        elif len(open_legs) == 2:
            # PyTreeNet orders the open legs (output, input): G acts on the output leg and
            # G^-1 on the input leg. Crossing them builds G^-1 W G, a wrong generator.
            out_ax, in_ax = open_legs
            tensor = np.moveaxis(np.tensordot(tensor, mat, axes=([out_ax], [1])), -1, out_ax)
            tensor = np.moveaxis(np.tensordot(tensor, inv, axes=([in_ax], [0])), -1, in_ax)
        else:
            raise ValueError(
                f"node {node_id} has {len(open_legs)} open legs, the gauge is defined for a "
                "state (one Pauli leg) and a superoperator (an (out, in) pair)."
            )
        net.replace_tensor(node_id, tensor, new_shape=False)
    if is_state:
        total = pauli_gauge(net) * (1.0 / gamma if inverse else gamma)
        if abs(total - 1.0) < 1e-12:
            if hasattr(net, GAUGE_ATTR):
                delattr(net, GAUGE_ATTR)
        else:
            setattr(net, GAUGE_ATTR, total)
    return net


def pauli_reweight_ptm(ptm: np.ndarray, num_sites: int, gamma: float) -> np.ndarray:
    """Conjugate the PTM of a ``num_sites``-qubit channel into the gauge,
    ``R -> G^(x)k R (G^-1)^(x)k``.

    The result is real, has the operator-Schmidt rank of ``R``, and is not orthogonal even for a
    unitary channel.
    """
    mat = pauli_reweight_matrix(gamma)
    inv = pauli_reweight_matrix(gamma, inverse=True)
    left, right = mat, inv
    for _ in range(int(num_sites) - 1):
        left, right = np.kron(left, mat), np.kron(right, inv)
    return left @ np.asarray(ptm) @ right


def pauli_reweight_channels(channels: Sequence[Tuple[np.ndarray, np.ndarray]], gamma: float
                            ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Conjugate operator-Schmidt channels ``[(A_s, B_s), ...]`` into the gauge.

    Each factor goes to ``G A G^-1``, so the sum still reproduces the conjugated PTM and the
    number of channels is unchanged.
    """
    mat = pauli_reweight_matrix(gamma)
    inv = pauli_reweight_matrix(gamma, inverse=True)
    return [(mat @ np.asarray(a_s) @ inv, mat @ np.asarray(b_s) @ inv) for a_s, b_s in channels]


def pauli_reweight_mpo(tensors: Sequence[np.ndarray], gamma: float) -> List[np.ndarray]:
    """Conjugate the site tensors of a channel MPO into the gauge, ``W -> G W G^-1``.

    Each tensor has legs ``(w_l, w_r, p_out, p_in)`` and the gauge acts on the physical pair
    alone, ``G`` on the output leg and ``G^-1`` on the input leg, so the contraction of the
    result is ``G^(x)k R (G^-1)^(x)k`` for the operator ``R`` of the input. The bonds, the
    dtype and the operator-Schmidt rank are unchanged, and the inputs are not modified.

    Raises:
        ValueError: A tensor is not of rank 4 with two dimension-4 physical legs.
    """
    out_scale = np.diag(pauli_reweight_matrix(gamma))
    in_scale = np.diag(pauli_reweight_matrix(gamma, inverse=True))
    result = []
    for tensor in tensors:
        w = np.asarray(tensor)
        if w.ndim != 4 or w.shape[2:] != (4, 4):
            raise ValueError(
                f"an MPO tensor must have legs (w_l, w_r, 4, 4), got shape {w.shape}."
            )
        result.append(w * out_scale[None, None, :, None] * in_scale[None, None, None, :])
    return result
