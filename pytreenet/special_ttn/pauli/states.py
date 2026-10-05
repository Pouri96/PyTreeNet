"""
Builders of fused Pauli states: product states on a chain and on two binary-tree layouts, and the
tensor-train decomposition of a dense density matrix. All tensors are real.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ...core.node import Node
from ...ttns.ttns import TreeTensorNetworkState
from ..binary import generate_binary_ttns
from ..mps import MatrixProductState
from .frame import pauli_rho_to_coeffs, pauli_site_vector

__all__ = ["pauli_product_mps", "pauli_from_dense", "pauli_product_tree_allphys",
           "pauli_product_tree_virtnode"]

PHYS_PREFIX_DEFAULT = "qubit"


def _mps_from_3leg(tensors: Sequence[np.ndarray], node_prefix: str) -> MatrixProductState:
    """Chain from ``(left, right, phys)`` tensors, dropping the trivial boundary legs."""
    ns = len(tensors)
    out = []
    for k, tensor in enumerate(tensors):
        if ns == 1:
            out.append(tensor.reshape(tensor.shape[2]))
        elif k == 0:
            out.append(tensor.reshape(tensor.shape[1], tensor.shape[2]))
        elif k == ns - 1:
            out.append(tensor.reshape(tensor.shape[0], tensor.shape[2]))
        else:
            out.append(tensor)
    return MatrixProductState.from_tensor_list(out, node_prefix=node_prefix)


def pauli_product_mps(local_kets: Sequence[np.ndarray],
                      node_prefix: str = PHYS_PREFIX_DEFAULT) -> MatrixProductState:
    """The chain of ``rho = (x)_q |psi_q><psi_q|``: one site per qubit, all bonds 1.

    Args:
        local_kets: One length-2 state vector per qubit.
        node_prefix: Node ``q`` is named ``f"{node_prefix}{q}"``.
    """
    tensors = [pauli_site_vector(ket).reshape(1, 1, 4) for ket in local_kets]
    return _mps_from_3leg(tensors, node_prefix)


def pauli_from_dense(rho: np.ndarray, node_prefix: str = PHYS_PREFIX_DEFAULT,
                     tol: float = 1e-12) -> MatrixProductState:
    """Chain of a dense density matrix by TT-SVD, dropping singular values below
    ``tol * s[0]`` at each bond. The tensors are real when ``rho`` is Hermitian."""
    num_qubits = int(round(np.log2(np.asarray(rho).shape[0])))
    coeffs = pauli_rho_to_coeffs(rho, num_qubits)
    if np.max(np.abs(coeffs.imag)) < 1e-12:
        coeffs = coeffs.real
    rest = np.asarray(coeffs).reshape(1, -1)
    tensors: List[np.ndarray] = []
    left = 1
    for _ in range(num_qubits - 1):
        u, s, vh = np.linalg.svd(rest.reshape(left * 4, -1), full_matrices=False)
        keep = max(int((s > tol * s[0]).sum()), 1) if s.size and s[0] > 0 else 1
        tensors.append(np.transpose(u[:, :keep].reshape(left, 4, keep), (0, 2, 1)))
        rest, left = (s[:keep, None] * vh[:keep]), keep
    tensors.append(rest.reshape(left, 1, 4))
    return _mps_from_3leg(tensors, node_prefix)


def _leaf_ordered_ids(num_phys: int, order: str, prefix: str) -> List[str]:
    """Qubit identifier at each heap position of the all-physical tree."""
    if order == "heap":
        return [f"{prefix}{i}" for i in range(num_phys)]
    if order != "inorder":
        raise ValueError(f"order must be 'inorder' or 'heap', got {order!r}")
    positions: List[int] = []

    def visit(i: int) -> None:
        if i >= num_phys:
            return
        visit(2 * i + 1)
        positions.append(i)
        visit(2 * i + 2)

    visit(0)
    ids: List[Optional[str]] = [None] * num_phys
    for chain_pos, heap_pos in enumerate(positions):
        ids[heap_pos] = f"{prefix}{chain_pos}"
    return ids  # type: ignore[return-value]


def pauli_product_tree_virtnode(local_kets: Sequence[np.ndarray],
                                phys_prefix: str = PHYS_PREFIX_DEFAULT,
                                depth: Optional[int] = None) -> TreeTensorNetworkState:
    """``rho = (x)_q |psi_q><psi_q|`` on a binary tree whose qubits sit at the leaves.

    The interior nodes are virtual, with a trivial open leg of dimension 1. All bonds are 1.

    Args:
        local_kets: One length-2 state vector per qubit.
        phys_prefix: Qubit ``q`` is named ``f"{phys_prefix}{q}"``.
        depth: Tree depth. ``None`` gives a balanced tree.
    """
    phys = np.zeros((1, 4))
    phys[0, 0] = 1.0
    tree = generate_binary_ttns(len(local_kets), 1, phys, depth=depth,
                                phys_prefix=phys_prefix, dtype=float)
    for q, ket in enumerate(local_kets):
        node_id = f"{phys_prefix}{q}"
        tree.replace_tensor(node_id, pauli_site_vector(ket).reshape(tree.nodes[node_id].shape),
                            new_shape=False)
    return tree


def pauli_product_tree_allphys(local_kets: Sequence[np.ndarray], order: str = "inorder",
                               phys_prefix: str = PHYS_PREFIX_DEFAULT
                               ) -> TreeTensorNetworkState:
    """``rho = (x)_q |psi_q><psi_q|`` on a binary tree where every node carries a qubit.

    Node ``i`` of the heap layout has children ``2i+1`` and ``2i+2``. All bonds are 1.

    Args:
        local_kets: One length-2 state vector per qubit.
        order: ``"inorder"`` makes the in-order traversal the qubit order, so a chain model's
            neighbours stay close in the tree. ``"heap"`` numbers the qubits by heap position.
        phys_prefix: Qubit ``q`` is named ``f"{phys_prefix}{q}"``.
    """
    num_qubits = len(local_kets)
    ids = _leaf_ordered_ids(num_qubits, order, phys_prefix)
    qubit_at = {nid: int(nid[len(phys_prefix):]) for nid in ids}

    def n_children(i: int) -> int:
        return sum(1 for c in (2 * i + 1, 2 * i + 2) if c < num_qubits)

    tree = TreeTensorNetworkState()
    tree.add_root(Node(identifier=ids[0]),
                  pauli_site_vector(local_kets[qubit_at[ids[0]]])
                  .reshape((1,) * n_children(0) + (4,)))
    for i in range(1, num_qubits):
        parent = (i - 1) // 2
        child_order = 0 if i == 2 * parent + 1 else 1
        shape = (1,) + (1,) * n_children(i) + (4,)
        parent_leg = child_order if parent == 0 else 1 + child_order
        tree.add_child_to_parent(Node(identifier=ids[i]),
                                 pauli_site_vector(local_kets[qubit_at[ids[i]]]).reshape(shape),
                                 0, ids[parent], parent_leg)
    return tree
