"""
Pauli-transfer matrices (PTMs) of gates and noise channels on fused Pauli sites.

A ``k``-qubit channel acts on the coefficients as one real ``4^k x 4^k`` matrix on ``k``
consecutive fused sites, because ``U_PAULI`` is a one-site change of basis. A one-sided map such
as ``rho -> U rho`` is not Hermiticity-preserving and has no real PTM, so the whole two-sided
channel is always passed at once.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .frame import U_PAULI

__all__ = ["dissipator_generator", "pauli_gate_ptm", "pauli_channel_ptm",
           "pauli_dissipator_channel_ptm", "pauli_gate_channels"]

_ID2 = np.eye(2, dtype=complex)


def dissipator_generator(jump_ops: Sequence[Tuple[np.ndarray, float]]) -> np.ndarray:
    """The 4x4 vectorised dissipator of one qubit on its row-major ``(ket, bra)`` pair,
    ``D = sum_j gamma_j [L (x) conj(L) - 1/2 (L^dag L) (x) I - 1/2 I (x) (L^dag L)^T]``.

    Args:
        jump_ops: ``(L, gamma)`` pairs with ``L`` a 2x2 jump operator and ``gamma`` the Lindblad
            rate, ``D[L] rho = gamma (L rho L^dag - 1/2 {L^dag L, rho})``.
    """
    d = np.zeros((4, 4), dtype=complex)
    for jump_op, gamma in jump_ops:
        jump_op = np.asarray(jump_op, dtype=complex)
        if jump_op.shape != (2, 2):
            raise ValueError(f"jump operator must be 2x2, got {jump_op.shape}.")
        ldl = jump_op.conj().T @ jump_op
        d += gamma * (np.kron(jump_op, jump_op.conj())
                      - 0.5 * np.kron(ldl, _ID2)
                      - 0.5 * np.kron(_ID2, ldl.T))
    return d


def _fuse_ket_bra_pairs(sup: np.ndarray, num_qubits: int) -> np.ndarray:
    """Reorder a superoperator from the ``(ket_0..ket_{k-1}, bra_0..bra_{k-1})`` index order
    that ``np.kron`` of two ``2^k`` operators gives into the per-qubit ``(ket_q, bra_q)`` order."""
    k = int(num_qubits)
    tensor = np.asarray(sup).reshape((2,) * (4 * k))
    row = [q for pair in zip(range(k), range(k, 2 * k)) for q in pair]
    perm = row + [2 * k + p for p in row]
    return np.transpose(tensor, perm).reshape(4 ** k, 4 ** k)


def _ptm_multi(sup: np.ndarray, num_qubits: int, what: str) -> np.ndarray:
    """Real PTM of a ``num_qubits``-qubit superoperator given in the ``(ket_all, bra_all)``
    basis. Raises if it does not map Hermitian operators to Hermitian operators."""
    k = int(num_qubits)
    rot = U_PAULI
    for _ in range(k - 1):
        rot = np.kron(rot, U_PAULI)
    block = rot @ _fuse_ket_bra_pairs(sup, k) @ rot.conj().T
    defect = float(np.max(np.abs(block.imag))) if block.size else 0.0
    if defect > 1e-10:
        raise ValueError(
            f"{what}: the Pauli-transfer matrix is not real (max|Im| = {defect:.2e}), so the "
            "superoperator is not Hermiticity-preserving. A one-sided factor such as "
            "`rho -> U rho` is the usual cause, pass the whole channel.")
    return block.real.copy()


def pauli_gate_ptm(gate: np.ndarray, num_qubits: Optional[int] = None) -> np.ndarray:
    """Real ``4^k x 4^k`` PTM of the unitary channel ``rho -> U rho U^dag`` on ``k`` consecutive
    fused sites.

    Args:
        gate: The ``2^k x 2^k`` unitary in ``np.kron`` order, the first qubit being the leftmost
            fused site.
        num_qubits: ``k``. Inferred from ``gate`` when omitted.
    """
    gate = np.asarray(gate, dtype=complex)
    if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
        raise ValueError(f"gate must be a square matrix, got shape {gate.shape}.")
    k = int(round(np.log2(gate.shape[0]))) if num_qubits is None else int(num_qubits)
    if gate.shape[0] != 2 ** k:
        raise ValueError(f"gate must be {2 ** k}x{2 ** k} for {k} qubit(s), got {gate.shape}.")
    return _ptm_multi(np.kron(gate, gate.conj()), k, "pauli_gate_ptm")


def pauli_channel_ptm(kraus_ops: Sequence[np.ndarray], num_qubits: Optional[int] = None,
                      check_trace_preserving: bool = True) -> np.ndarray:
    """Real ``4^k x 4^k`` PTM of the Kraus channel ``rho -> sum_a K_a rho K_a^dag``.

    Args:
        kraus_ops: The complete set of Kraus operators, each ``2^k x 2^k``.
        num_qubits: ``k``. Inferred from the operator shapes when omitted.
        check_trace_preserving: Warn when ``sum_a K_a^dag K_a != I``, which an incomplete Kraus
            set would otherwise hide, since the PTM is real for any set.
    """
    ops = [np.asarray(k_a, dtype=complex) for k_a in kraus_ops]
    if not ops:
        raise ValueError("pauli_channel_ptm: no Kraus operators given.")
    dim = ops[0].shape[0]
    for k_a in ops:
        if k_a.shape != (dim, dim):
            raise ValueError("pauli_channel_ptm: all Kraus operators must be square and of the "
                             f"same size, got {[o.shape for o in ops]}.")
    k = int(round(np.log2(dim))) if num_qubits is None else int(num_qubits)
    if dim != 2 ** k:
        raise ValueError(f"Kraus operators must be {2 ** k}x{2 ** k} for {k} qubit(s), "
                         f"got {dim}x{dim}.")
    block = _ptm_multi(sum(np.kron(k_a, k_a.conj()) for k_a in ops), k, "pauli_channel_ptm")
    if check_trace_preserving:
        identity_row = np.zeros(4 ** k)
        identity_row[0] = 1.0
        drift = float(np.max(np.abs(block[0] - identity_row)))
        if drift > 1e-10:
            warnings.warn(f"pauli_channel_ptm: the channel is not trace-preserving (identity "
                          f"row deviates by {drift:.2e} from e_0).", RuntimeWarning)
    return block


def pauli_dissipator_channel_ptm(jump_ops: Sequence[Tuple[np.ndarray, float]],
                                 time_step: float) -> np.ndarray:
    """Real 4x4 on-site PTM of the exact one-qubit channel ``exp(time_step * D)``, with ``D``
    the dissipator of :func:`dissipator_generator`."""
    from scipy.linalg import expm
    generator = dissipator_generator(jump_ops)
    return _ptm_multi(expm(float(time_step) * generator), 1, "pauli_dissipator_channel_ptm")


def pauli_gate_channels(ptm: np.ndarray, tol: float = 1e-12
                        ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Operator-Schmidt channels ``R = sum_s A_s (x) B_s`` of a two-site PTM, with real 4x4
    factors and ``sqrt(sigma_s)`` absorbed into each.

    The count is the square of the gate's own operator-Schmidt rank: 16 for a generic two-qubit
    gate, 4 for CNOT and CZ, 1 for a product gate.

    Args:
        ptm: The 16x16 PTM of a two-qubit channel.
        tol: Channel ``s`` is kept iff ``sigma_s > tol * sigma_0``.
    """
    mat = np.asarray(ptm)
    if mat.shape != (16, 16):
        raise ValueError("pauli_gate_channels expects the 16x16 PTM of a two-qubit channel, "
                         f"got shape {mat.shape}. Build it with pauli_gate_ptm or "
                         "pauli_channel_ptm.")
    defect = float(np.max(np.abs(mat.imag))) if np.iscomplexobj(mat) else 0.0
    if defect > 1e-10:
        raise ValueError(f"pauli_gate_channels: the PTM is not real (max|Im| = {defect:.2e}).")
    reshuffled = np.real(mat).reshape(4, 4, 4, 4).transpose(0, 2, 1, 3).reshape(16, 16)
    u_mat, sing, vh_mat = np.linalg.svd(reshuffled)
    terms: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in range(len(sing)):
        if sing[s] <= tol * sing[0]:
            break
        scale = np.sqrt(sing[s])
        terms.append(((u_mat[:, s] * scale).reshape(4, 4), (scale * vh_mat[s, :]).reshape(4, 4)))
    return terms
