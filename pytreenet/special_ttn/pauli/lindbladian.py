"""
The Lindbladian ``drho/dt = -i[H, rho] + sum_q D_q(rho)`` as a TTNO on the fused Pauli frame, and a
dense reference of the same generator for small-N validation.

Every local block is a real 4x4 Pauli-transfer matrix, because a Hermiticity-preserving
superoperator has a real PTM and every Lindbladian is one. The terms are therefore built from the
two Hermiticity-preserving maps ``C_A: rho -> -i[A, rho]`` and ``J_A: rho -> (1/2){A, rho}``.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...operators.hamiltonian import Hamiltonian
from ...operators.tensorproduct import TensorProduct
from ...ttno.ttno_class import TreeTensorNetworkOperator
from .channels import dissipator_generator
from .frame import PAULIS, U_PAULI, pauli_rho_to_coeffs, pauli_site_ids

__all__ = ["pauli_lindbladian_ttno", "pauli_dense_lindbladian"]

_I2 = np.eye(2, dtype=complex)


def _real_block(sup: np.ndarray, what: str) -> np.ndarray:
    """Real PTM of a one-site superoperator given on the row-major ``(ket, bra)`` pair."""
    block = U_PAULI @ np.asarray(sup, dtype=complex) @ U_PAULI.conj().T
    defect = float(np.max(np.abs(block.imag)))
    if defect > 1e-10:
        raise ValueError(f"{what}: the Pauli-transfer block is not real (max|Im| = "
                         f"{defect:.2e}), the generator is not Hermiticity-preserving.")
    return block.real.copy()


def _commutator_block(op: np.ndarray) -> np.ndarray:
    """``C_A``, the real PTM of ``rho -> -i[A, rho]`` for Hermitian ``A``."""
    op = np.asarray(op, dtype=complex)
    return _real_block(-1j * (np.kron(op, _I2) - np.kron(_I2, op.T)), "commutator block")


def _anticommutator_block(op: np.ndarray) -> np.ndarray:
    """``J_A``, the real PTM of ``rho -> (1/2){A, rho}`` for Hermitian ``A``."""
    op = np.asarray(op, dtype=complex)
    return _real_block(0.5 * (np.kron(op, _I2) + np.kron(_I2, op.T)), "anticommutator block")


def _pauli_op_schmidt(h_ij: np.ndarray) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    """Operator-Schmidt decomposition ``h = sum_k s_k A_k (x) B_k`` taken in the Pauli basis, so
    that the weights are real and the factors Hermitian. At most 4 components."""
    h_ij = np.asarray(h_ij, dtype=complex)
    coeff = np.array([[np.trace(np.kron(a, b).conj().T @ h_ij) / 4.0 for b in PAULIS]
                      for a in PAULIS])
    defect = float(np.max(np.abs(coeff.imag)))
    if defect > 1e-10:
        raise ValueError(f"bond term is not Hermitian (max|Im| of its Pauli coefficients = "
                         f"{defect:.2e}).")
    u, s, vh = np.linalg.svd(coeff.real)
    out = []
    for k in range(4):
        if s[k] <= 1e-12:
            continue
        out.append((float(s[k]), sum(u[a, k] * PAULIS[a] for a in range(4)),
                    sum(vh[k, b] * PAULIS[b] for b in range(4))))
    return out


def pauli_lindbladian_ttno(reference, bond_terms: Dict[Tuple[int, int], np.ndarray],
                           site_terms: Dict[int, np.ndarray],
                           jump_ops: Dict[int, Sequence[Tuple[np.ndarray, float]]],
                           generator: str = "L",
                           site_ids: Optional[Sequence[str]] = None
                           ) -> TreeTensorNetworkOperator:
    """The Lindbladian as a TTNO on the topology of ``reference``, a chain or a tree.

    Args:
        reference: A fused Pauli state whose structure the TTNO follows.
        bond_terms: ``{(i, j): h}`` with ``i < j`` and ``h`` a Hermitian 4x4 matrix in
            ``np.kron`` order. The qubits need not be adjacent.
        site_terms: ``{q: h}`` with ``h`` a Hermitian 2x2 matrix.
        jump_ops: ``{q: [(L, gamma), ...]}`` with ``gamma`` the Lindblad rate.
        generator: ``"L"`` gives the real Liouvillian, to be propagated as ``exp(+dt L)``.
            ``"H_L"`` gives ``H_L = i L``, complex, to be propagated as ``exp(-i dt H_L)``. The
            two differ by the factor ``i``, so pairing the wrong propagator is silently wrong.
        site_ids: Node of each qubit. Read from ``reference`` when ``None``.

    Returns:
        The TTNO, of dtype float for ``"L"`` and complex for ``"H_L"``.
    """
    if generator not in ("H_L", "L"):
        raise ValueError(f"generator must be 'H_L' or 'L', got {generator!r}.")
    site_ids = pauli_site_ids(reference) if site_ids is None else list(site_ids)
    gamma = "iL" if generator == "H_L" else "reL"
    conv: Dict[str, np.ndarray] = {}
    sym_cache: Dict[bytes, str] = {}

    def sym_of(mat) -> str:
        mat = np.asarray(mat, dtype=float)
        key = np.round(mat, 12).tobytes()
        name = sym_cache.get(key)
        if name is None:
            name = f"o{len(sym_cache)}"
            sym_cache[key] = name
            conv[name] = mat
        return name

    # "1" is reserved for the identity padding and is applied once per site, so the imaginary
    # unit of H_L rides on its own symbol.
    ham = Hamiltonian(conversion_dictionary=conv,
                      coeffs_mapping={"iL": 1j} if generator == "H_L" else {"reL": 1.0})
    for q, h_q in site_terms.items():
        ham.add_term((Fraction(1), gamma,
                      TensorProduct({site_ids[q]: sym_of(_commutator_block(h_q))})))
    for (i, j), h_ij in bond_terms.items():
        if not i < j:
            raise ValueError(f"bond term ({i}, {j}) must have i < j.")
        for weight, op_a, op_b in _pauli_op_schmidt(h_ij):
            c_a, j_a = _commutator_block(op_a), _anticommutator_block(op_a)
            c_b, j_b = _commutator_block(op_b), _anticommutator_block(op_b)
            for left, right in ((c_a, j_b), (j_a, c_b)):
                ham.add_term((Fraction(1), gamma,
                              TensorProduct({site_ids[i]: sym_of(weight * left),
                                             site_ids[j]: sym_of(right)})))
    for q, jops in jump_ops.items():
        if not jops:
            continue
        block = _real_block(dissipator_generator(jops), f"dissipator on qubit {q}")
        ham.add_term((Fraction(1), gamma, TensorProduct({site_ids[q]: sym_of(block)})))
    ham.include_identities(reference, ident_creation=lambda d: np.eye(d))
    return TreeTensorNetworkOperator.from_hamiltonian(
        ham, reference, dtype=complex if generator == "H_L" else float)


# ---- dense reference (small N)

def _embed(op: np.ndarray, sites: Sequence[int], num_sites: int, dim: int) -> np.ndarray:
    """``op`` acting on ``sites`` (in that order) of ``num_sites`` sites of dimension ``dim``,
    identity elsewhere."""
    k = len(sites)
    rest = [q for q in range(num_sites) if q not in sites]
    order = list(sites) + rest
    full = np.kron(op, np.eye(dim ** (num_sites - k))).reshape((dim,) * (2 * num_sites))
    inv = np.argsort(order)
    perm = list(inv) + [num_sites + p for p in inv]
    return full.transpose(perm).reshape(dim ** num_sites, dim ** num_sites)


def pauli_dense_lindbladian(num_qubits: int,
                            bond_terms: Dict[Tuple[int, int], np.ndarray],
                            site_terms: Dict[int, np.ndarray],
                            jump_ops: Dict[int, Sequence[Tuple[np.ndarray, float]]]
                            ) -> np.ndarray:
    """The real ``4^N x 4^N`` Lindbladian on the Pauli coefficients, built densely.

    ``expm(t * L) @ c`` evolves the coefficients ``c`` of ``rho`` for time ``t``. The generator is
    assembled on ``vec(rho)`` with ``vec(A rho B) = (A (x) B^T) vec(rho)`` and then rotated into
    the Pauli basis. Same arguments as :func:`pauli_lindbladian_ttno`. Small N only.
    """
    n, dim = num_qubits, 2 ** num_qubits
    ident = np.eye(dim)

    def hamiltonian_part(h_full: np.ndarray) -> np.ndarray:
        return -1j * (np.kron(h_full, ident) - np.kron(ident, h_full.T))

    sup = np.zeros((dim ** 2, dim ** 2), dtype=complex)
    for q, h_q in site_terms.items():
        sup += hamiltonian_part(_embed(np.asarray(h_q, dtype=complex), [q], n, 2))
    for (i, j), h_ij in bond_terms.items():
        sup += hamiltonian_part(_embed(np.asarray(h_ij, dtype=complex), [i, j], n, 2))
    for q, jops in jump_ops.items():
        for jump_op, rate in jops:
            l_full = _embed(np.asarray(jump_op, dtype=complex), [q], n, 2)
            ldl = l_full.conj().T @ l_full
            sup += rate * (np.kron(l_full, l_full.conj()) - 0.5 * np.kron(ldl, ident)
                           - 0.5 * np.kron(ident, ldl.T))
    # P maps vec(rho) to the Pauli coefficients, columns being the images of the unit matrices.
    to_pauli = np.array([pauli_rho_to_coeffs(col.reshape(dim, dim), n)
                         for col in np.eye(dim ** 2)]).T
    block = to_pauli @ sup @ to_pauli.conj().T
    defect = float(np.max(np.abs(block.imag)))
    if defect > 1e-10:
        raise ValueError(f"the dense Lindbladian is not real in the Pauli basis (max|Im| = "
                         f"{defect:.2e}), some term is not Hermiticity-preserving.")
    return block.real
