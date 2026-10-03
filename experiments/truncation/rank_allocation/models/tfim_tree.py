"""The Table I system: ``H = -sum_i (X_i X_{i+1} + Z_i)`` on an open chain, Neel start.

The qubits ``qubit0 .. qubit{n-1}`` sit in chain order on the leaves of a balanced binary tree
whose interior nodes have no physical leg.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.sim_operators import (create_nearest_neighbour_hamiltonian,
                                               create_single_site_hamiltonian)
from pytreenet.special_ttn.binary import generate_binary_ttns
from pytreenet.ttns.ttns import TreeTensorNetworkState

X, Y, Z = pauli_matrices()
_SP = {"X": sp.csr_matrix(X), "Z": sp.csr_matrix(Z),
       "I": sp.eye(2, format="csr", dtype=complex)}

#: The single coefficient of both terms, carrying the sign of ``H``.
COUPLING = -1.0
#: The two basis states the Neel state alternates between.
UP = np.array([1.0, 0.0], dtype=complex)
DOWN = np.array([0.0, 1.0], dtype=complex)


def tfim_hamiltonian(n: int) -> Hamiltonian:
    """The TTNO-ready Hamiltonian on identifiers ``qubit0 .. qubit{n-1}``."""
    conv = {"X": X, "Z": Z}
    ham = Hamiltonian()
    ham.add_hamiltonian(create_nearest_neighbour_hamiltonian(
        [(f"qubit{i}", f"qubit{i + 1}") for i in range(n - 1)], "X",
        factor=(Fraction(1), "j"), local_operator2="X",
        conversion_dict=conv, coeffs_mapping={"j": complex(COUPLING)}))
    ham.add_hamiltonian(create_single_site_hamiltonian(
        [f"qubit{i}" for i in range(n)], "Z", factor=(Fraction(1), "h"),
        conversion_dict=conv, coeffs_mapping={"h": complex(COUPLING)}))
    ham.include_identities([1, 2])
    return ham


def tfim_sparse(n: int) -> sp.csr_matrix:
    """The same Hamiltonian as a dense-index sparse matrix, for the exact reference."""
    def term(ops):
        mat = _SP[ops.get(0, "I")]
        for i in range(1, n):
            mat = sp.kron(mat, _SP[ops.get(i, "I")], format="csr")
        return mat

    out = sp.csr_matrix((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        out = out + COUPLING * term({i: "X", i + 1: "X"})
    for i in range(n):
        out = out + COUPLING * term({i: "Z"})
    return out


def neel_state(n: int) -> TreeTensorNetworkState:
    """``|0101...>`` on the binary tree, every bond of dimension one."""
    state = generate_binary_ttns(n, 1, UP.reshape(1, 2), phys_prefix="qubit")
    for i in range(n):
        nid = f"qubit{i}"
        ket = UP if i % 2 == 0 else DOWN
        state.replace_tensor(nid, ket.reshape(state.tensors[nid].shape))
    state.canonical_form(state.root_id)
    state.normalize()
    return state


def aligned_vector(state: TreeTensorNetworkState, n: int) -> np.ndarray:
    """The state as a dense vector in ``qubit0 .. qubit{n-1}`` order."""
    vec, order = state.to_vector(to_copy=True)
    dims = [int(np.prod([state.tensors[nid].shape[leg]
                         for leg in state.nodes[nid].open_legs])) for nid in order]
    tensor = np.asarray(vec).reshape(dims)
    position = {nid: i for i, nid in enumerate(order)}
    physical = [position[f"qubit{i}"] for i in range(n)]
    rest = [i for i in range(len(order)) if i not in set(physical)]
    return np.transpose(tensor, physical + rest).reshape(-1)


def infidelity(state: TreeTensorNetworkState, exact: np.ndarray, n: int) -> float:
    """``1 - |<exact|psi>|^2``, both sides normalised."""
    vec = aligned_vector(state, n)
    overlap = np.vdot(exact, vec)
    return float(abs(1.0 - abs(overlap) ** 2
                     / abs(np.vdot(exact, exact) * np.vdot(vec, vec))))
