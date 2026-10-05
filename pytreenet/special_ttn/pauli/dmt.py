"""
Density-matrix truncation (DMT) of a fused Pauli chain (White, Zaletel, Mong, Refael,
arXiv:1707.01506).

At every truncated bond a set of directions is reserved so that chosen functionals of ``rho``
survive exactly, and the rest of the bond budget goes to the largest singular values of what
remains. The reserved functionals are the trace and the Pauli strings within ``radius`` qubits of
the cut, optionally extended by the channels of an observable.

The sweep is a backward sweep over the chain. Its precondition is a state that is left-canonical
up to the right end of the window with the orthogonality centre there. The map is the one-sided
reserved projector: the reserved span and the complement are collected in the left bond index of
each site and the bond is cut by the projector onto them.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .frame import PAULIS, _refuse_if_gauged, pauli_site_ids
from .lindbladian import _pauli_op_schmidt

__all__ = ["DMTReport", "dmt_min_chi_for_radius", "orient_observable", "pauli_dmt_truncate",
           "pauli_operator_tensors"]

_SQ2 = np.sqrt(2.0)
_CAP_VEC = np.array([1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class DMTReport:
    """What a DMT truncation did.

    Attributes:
        discarded: The ledger ``sum_i sqrt(eps_i)`` over the truncated bonds, ``eps_i`` the
            discarded weight of bond ``i`` relative to the norm at that point. It bounds the
            relative Frobenius error. A lossless pass reports rounding noise, about 1e-15, and
            exactly 0 when no bond exceeds the budget.
        discarded_weights: The weight ``eps_i`` discarded at each truncated bond, relative to
            the squared norm at that point, in the order the bonds were cut. ``discarded`` is
            the sum of their square roots.
        reserved_ranks: Number of reserved directions at each bond that needed a cut. On a chain
            it is keyed by the position of the bond's right site, on a tree by the child node of
            the edge.
        clipped_bonds: Bonds whose reservation did not fit the budget and was cut back.
    """

    discarded: float
    discarded_weights: List[float] = field(default_factory=list)
    reserved_ranks: Dict[Union[int, str], int] = field(default_factory=dict)
    clipped_bonds: List[Union[int, str]] = field(default_factory=list)


def dmt_min_chi_for_radius(radius: int) -> int:
    """The smallest ``max_bond_dim`` that leaves two complement directions after a radius-``radius``
    reservation: ``2 * 4**radius + 2``. It is 4, 10 and 34 for radius 0, 1 and 2."""
    return 2 * 4 ** int(radius) + 2


# ---- reading and writing the chain as (left, right, phys) tensors

def _read_chain(net, site_ids: Sequence[str]) -> List[np.ndarray]:
    """The site tensors in ``site_ids`` order as ``(left, right, phys)``, with a dummy leg of
    dimension 1 at each end. Any leg order of the nodes is accepted."""
    n = len(site_ids)
    out = []
    for k, sid in enumerate(site_ids):
        node = net.nodes[sid]
        expected = {site_ids[j] for j in (k - 1, k + 1) if 0 <= j < n}
        if set(node.neighbouring_nodes()) != expected:
            raise ValueError(
                f"the sites are not a chain in the order given: {sid!r} is joined to "
                f"{sorted(node.neighbouring_nodes())}, expected {sorted(expected)}.")
        axes = [node.neighbour_index(site_ids[j]) for j in (k - 1, k + 1) if 0 <= j < n]
        axes.append(node.open_legs[0])
        tensor = np.transpose(np.asarray(net.tensors[sid]), axes)
        if k == 0:
            tensor = np.expand_dims(tensor, 0)
        if k == n - 1:
            tensor = np.expand_dims(tensor, 1)
        out.append(tensor)
    return out


def _write_chain(net, site_ids: Sequence[str], tensors: Sequence[np.ndarray],
                 lo: int, hi: int) -> None:
    """Write ``tensors[lo..hi]`` back into ``net`` in each node's own leg order."""
    n = len(site_ids)
    for k in range(lo, hi + 1):
        node = net.nodes[site_ids[k]]
        axes = [node.neighbour_index(site_ids[j]) for j in (k - 1, k + 1) if 0 <= j < n]
        axes.append(node.open_legs[0])
        dummy = tuple(ax for ax, at_end in ((0, k == 0), (1, k == n - 1)) if at_end)
        tensor = np.squeeze(tensors[k], axis=dummy)
        net.replace_tensor(site_ids[k], np.transpose(tensor, np.argsort(axes)), new_shape=True)


def _left_canonicalize(tensors: Sequence[np.ndarray]) -> List[np.ndarray]:
    """QR sweep making every tensor but the last left-isometric. Keeps a real dtype."""
    out = [t.copy() for t in tensors]
    for k in range(len(out) - 1):
        ld, r, p = out[k].shape
        q, rmat = np.linalg.qr(np.transpose(out[k], (0, 2, 1)).reshape(ld * p, r))
        out[k] = np.transpose(q.reshape(ld, p, -1), (0, 2, 1))
        out[k + 1] = np.tensordot(rmat, out[k + 1], axes=([1], [0]))
    return out


# The helpers below run O(N) times per truncation on tensors so small that the call overhead is
# the cost, so the contractions are explicit matmul and tensordot, not einsum with a path search.

def _lenv(env: np.ndarray, tensor: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """One cap environment step to the right, ``sum_{a,p} env_a tensor_{arp} sel_p``."""
    return env @ (tensor @ sel)


def _rvec(tensor: np.ndarray, sel: np.ndarray, renv: np.ndarray) -> np.ndarray:
    """One cap environment step to the left, ``sum_{r,p} tensor_{arp} sel_p renv_r``."""
    return (tensor @ sel) @ renv


def _ordered_orthonormal_basis(cols: np.ndarray, tol: float) -> np.ndarray:
    """Orthonormal basis of the column span by re-orthogonalised modified Gram-Schmidt in column
    order, dropping a column whose residual is below ``tol`` times its norm.

    The order is the priority. The first ``m`` basis vectors are the best rank-``m`` prefix of the
    columns as given, so a reservation that does not fit the budget degrades from the outside in.
    """
    basis: List[np.ndarray] = []
    for j in range(cols.shape[1]):
        vec = np.array(cols[:, j], dtype=cols.dtype, copy=True)
        norm0 = float(np.linalg.norm(vec))
        if norm0 <= 0.0:
            continue
        for _ in range(2):
            for basis_vec in basis:
                vec = vec - basis_vec * np.vdot(basis_vec, vec)
        norm = float(np.linalg.norm(vec))
        if norm <= tol * norm0:
            continue
        basis.append(vec / norm)
    if not basis:
        return np.zeros((cols.shape[0], 0), dtype=cols.dtype)
    return np.stack(basis, axis=1)


def _radius_block_left(lefts: Sequence[np.ndarray], tensors: Sequence[np.ndarray],
                       k: int, radius: int) -> np.ndarray:
    """``(l_k, 4**radius)`` matrix of the left cap environment of bond ``k`` with an open Pauli
    leg on each of the ``radius`` qubits left of the cut, everything further left traced out.
    Column 0 is the all-identity string, the trace."""
    env = lefts[k - radius]
    for i in range(k - radius, k):
        env = np.swapaxes(np.tensordot(env, tensors[i], axes=([-1], [0])), -1, -2)
    return env.reshape(-1, env.shape[-1]).T


def _radius_block_right(capenv: Dict[int, np.ndarray], tensors: Sequence[np.ndarray],
                        k: int, radius: int) -> np.ndarray:
    """``(l_k, 4**radius)`` matrix of the column images of the right reserved functionals: the
    chain from site ``k`` through ``k + radius - 1`` with an open Pauli leg on each, capped by the
    trace environment from ``k + radius``. Column 0 is the all-identity string."""
    env = capenv[k + radius]
    for i in range(k + radius - 1, k - 1, -1):
        env = np.swapaxes(np.tensordot(env, tensors[i], axes=([-1], [1])), -1, -2)
    return env.reshape(-1, env.shape[-1]).T


def _radius_levels(radius: int) -> np.ndarray:
    """For each flat column of a radius block the smallest radius that already contains it, 0 for
    the trace and ``r`` for a string whose outermost non-identity Pauli is ``r`` qubits from the
    cut. The blocks are laid out outermost leg first."""
    if radius == 0:
        return np.zeros(1, dtype=int)
    digits = np.stack(np.meshgrid(*[np.arange(4)] * radius, indexing="ij")).reshape(radius, -1)
    nonzero = digits != 0
    first = np.where(nonzero.any(axis=0), np.argmax(nonzero, axis=0), radius)
    return radius - first


def _priority_ordered_reserved(blocks: Sequence[Tuple[int, int, np.ndarray]]) -> np.ndarray:
    """Concatenate ``(priority, side, columns)`` blocks in priority order: the trace, then the
    nearest-neighbour marginals, then the observables, then the outer radii, alternating the
    left and right side at each level."""
    order = sorted(range(len(blocks)), key=lambda i: (blocks[i][0], blocks[i][1], i))
    return np.concatenate([blocks[i][2] for i in order], axis=1)


def _obs_left_envs(tensors: Sequence[np.ndarray], obs: Sequence[np.ndarray],
                   stop: Optional[int] = None) -> List[np.ndarray]:
    """``E[j]`` of shape ``(l_j, a_j)``: the bilinear environment of the state and the observable
    left of bond ``j``, with the observable's virtual index open. Only ``E[0..stop]`` is built."""
    envs = [np.ones((1, 1), dtype=np.result_type(tensors[0].dtype, obs[0].dtype))]
    for k in range(len(tensors) if stop is None else stop):
        state_env = np.tensordot(envs[k], tensors[k], axes=([0], [0]))
        envs.append(np.tensordot(state_env, obs[k], axes=([0, 2], [0, 2])))
    return envs


def _tol_rank(sing: np.ndarray, rel_tol: float, total_tol: float) -> int:
    """Number of singular values above both the relative and the absolute floor."""
    if sing.size == 0 or sing[0] <= 0:
        return 0
    return int((sing > max(rel_tol * sing[0], total_tol)).sum())


def _respect_degenerate(sing: np.ndarray, keep: int, tol: float) -> int:
    """Pull ``keep`` back to the start of the degenerate multiplet it would split. If the whole
    retained block is one multiplet the original count stands."""
    if keep <= 0 or keep >= sing.size:
        return keep
    reduced = keep
    while reduced > 0 and abs(sing[reduced - 1] - sing[reduced]) <= tol * max(sing[0], 1e-300):
        reduced -= 1
    return reduced if reduced > 0 else keep


# ---- observables

def _op_cap_vector(op: np.ndarray) -> np.ndarray:
    """The site vector ``Tr(op sigma~_mu)``. For the identity it is the trace cap tensor."""
    op = np.asarray(op, dtype=complex)
    return np.array([np.trace(op @ P) / _SQ2 for P in PAULIS])


def _direct_sum(left: List[np.ndarray], right: List[np.ndarray]) -> List[np.ndarray]:
    """Sum of two chains of ``(left, right, phys)`` tensors: block-diagonal in the bonds."""
    n = len(left)
    if n == 1:
        return [left[0] + right[0]]
    out: List[np.ndarray] = []
    dtype = np.result_type(left[0].dtype, right[0].dtype)
    for k in range(n):
        la, ra, phys = left[k].shape
        lb, rb, _ = right[k].shape
        block = np.zeros((la if k == 0 else la + lb, rb if k == n - 1 else ra + rb, phys),
                         dtype=dtype)
        if k == 0:
            block[:, :ra] = left[k]
            block[:, ra:] = right[k]
        elif k == n - 1:
            block[:la] = left[k]
            block[la:] = right[k]
        else:
            block[:la, :ra] = left[k]
            block[la:, ra:] = right[k]
        out.append(block)
    return out


def _compress(tensors: Sequence[np.ndarray], tol: float) -> List[np.ndarray]:
    """Left-canonicalise, then a truncating backward SVD to the minimal bond up to ``tol``."""
    out = _left_canonicalize(tensors)
    for k in range(len(out) - 1, 0, -1):
        ld, r, phys = out[k].shape
        u_mat, sing, vh_mat = np.linalg.svd(out[k].reshape(ld, r * phys), full_matrices=False)
        keep = max(int((sing > tol * sing[0]).sum()), 1) if sing.size and sing[0] > 0 else 1
        out[k] = vh_mat[:keep].reshape(keep, r, phys)
        out[k - 1] = np.transpose(
            np.tensordot(out[k - 1], u_mat[:, :keep] * sing[:keep], axes=([1], [0])), (0, 2, 1))
    return out


def pauli_operator_tensors(num_qubits: int, bond_terms: Optional[dict] = None,
                           site_terms: Optional[dict] = None, tol: float = 1e-12,
                           compress_every: int = 8) -> List[np.ndarray]:
    """The Pauli coefficients ``Tr(sigma~_mu O)`` of a Hermitian observable
    ``O = sum_q h_q + sum_{(i,j)} h_ij`` as a chain of ``(left, right, 4)`` tensors.

    It takes the dictionaries that ``pauli_lindbladian_ttno`` takes. Contracted bilinearly with a
    state it gives ``Tr(O rho)``, and it is the observable that :func:`pauli_dmt_truncate` reserves.
    A term on ``(i, j)`` with ``j > i + 1`` carries identities between the two qubits. The sum is
    compressed to the minimal bond dimension, which is the operator-Schmidt rank of ``O``.

    Args:
        num_qubits: Length of the chain.
        bond_terms: ``{(i, j): h}`` with ``i < j`` and ``h`` a Hermitian 4x4 matrix.
        site_terms: ``{q: h}`` with ``h`` a Hermitian 2x2 matrix.
        tol: Relative singular-value tolerance of the compression.
        compress_every: Compress after this many added terms, so a long-range model does not
            build a bond that grows with the number of terms.

    Returns:
        The tensors, real when the observable is Hermitian. The trace cap if there are no terms.

    Raises:
        ValueError: A term acts on a qubit outside ``range(num_qubits)``, a bond term has
            ``i >= j``, or a term is not Hermitian.
    """
    bond_terms = bond_terms or {}
    site_terms = site_terms or {}
    outside = sorted({q for q in site_terms if not 0 <= q < num_qubits}
                     | {q for pair in bond_terms for q in pair if not 0 <= q < num_qubits})
    if outside:
        raise ValueError(f"terms act on qubits {outside} outside range({num_qubits}).")
    ident = _op_cap_vector(PAULIS[0])

    def product_term(local: Dict[int, np.ndarray], scale: float = 1.0) -> List[np.ndarray]:
        first = min(local)
        return [((scale if q == first else 1.0) * local.get(q, ident)).reshape(1, 1, 4)
                for q in range(num_qubits)]

    terms: List[List[np.ndarray]] = []
    for q, h_q in site_terms.items():
        vec = _op_cap_vector(h_q)
        defect = float(np.max(np.abs(vec.imag)))
        if defect > 1e-10:
            raise ValueError(f"site term on qubit {q} is not Hermitian (max|Im| of its "
                             f"Pauli coefficients = {defect:.2e}).")
        terms.append(product_term({q: vec}))
    for (i, j), h_ij in bond_terms.items():
        if not i < j:
            raise ValueError(f"bond term ({i}, {j}) must have i < j.")
        for weight, op_a, op_b in _pauli_op_schmidt(h_ij):
            terms.append(product_term({i: _op_cap_vector(op_a), j: _op_cap_vector(op_b)}, weight))
    if not terms:
        return [(_SQ2 * _CAP_VEC).reshape(1, 1, 4) for _ in range(num_qubits)]
    acc = terms[0]
    for idx, term in enumerate(terms[1:], start=1):
        acc = _direct_sum(acc, term)
        if idx % compress_every == 0:
            acc = _compress(acc, tol)
    acc = _compress(acc, tol)
    if all(np.max(np.abs(t.imag)) < 1e-12 for t in acc if np.iscomplexobj(t)):
        acc = [t.real.copy() if np.iscomplexobj(t) else t for t in acc]
    return acc


def orient_observable(tensors: Sequence[np.ndarray], qubit_ids: Sequence[str],
                      path: Sequence[str]) -> List[np.ndarray]:
    """The observable chain of :func:`pauli_operator_tensors` in the order of ``path``.

    The tensors follow ``qubit_ids``. An integrator sweeps along its own update path, which is
    the qubit order or its reverse, so a reversed path needs the chain reversed with its left
    and right legs swapped.

    Raises:
        ValueError: ``path`` is neither ``qubit_ids`` nor its reverse.
    """
    qubit_ids, path = list(qubit_ids), list(path)
    if path == qubit_ids:
        return list(tensors)
    if path == qubit_ids[::-1]:
        return [np.transpose(t, (1, 0, 2)) for t in tensors[::-1]]
    raise ValueError("an observable can only be reserved along the chain in qubit order or its "
                     "reverse.")


# ---- the truncation

def pauli_dmt_truncate(net, chi: int, site_ids: Optional[Sequence[str]] = None,
                       window: Optional[Tuple[int, int]] = None, radius: int = 1,
                       observables: Optional[Sequence[Sequence[np.ndarray]]] = None,
                       rel_tol: float = 1e-12, total_tol: float = 1e-12,
                       respect_degenerate: bool = True, degeneracy_tol: float = 1e-10,
                       reserve_tol: float = 1e-12, in_place: bool = False
                       ) -> Tuple[object, DMTReport]:
    """Truncate a fused Pauli chain to bond dimension ``chi`` with the DMT reservation.

    At bond ``k`` the reserved set is the trace and the Pauli strings on the ``radius`` qubits
    on each side of the cut. Every operator on at most ``2 * radius + 1`` contiguous qubits then
    survives every truncation exactly. Radius 1 covers every interaction of range 2.

    The reserved set is rank-revealed in priority order, and if it does not fit under ``chi`` it
    is cut back from the outside in and a ``RuntimeWarning`` names the bonds. The bond never
    exceeds ``chi``.

    Args:
        net: A fused Pauli chain with real tensors, left-canonical up to the window's right end
            with the orthogonality centre there. With ``window=None`` any gauge is accepted.
        chi: The bond budget.
        site_ids: The qubit nodes in chain order. Read from ``net`` when ``None``.
        window: ``(lo, hi)`` positions in ``site_ids``. Only the bonds from ``lo`` to ``hi`` are
            truncated, and the orthogonality centre ends at ``lo``. ``None`` re-canonicalises and
            truncates the whole chain.
        radius: Qubits reserved on each side of a cut.
        observables: Chains of ``(left, right, 4)`` tensors from :func:`pauli_operator_tensors`.
            Each is reserved with its virtual index open, which makes its expectation exact at
            any range for ``2 * D`` directions at a cut of operator-Schmidt rank ``D``.
        rel_tol: Complement singular values below ``rel_tol * s[0]`` are dropped.
        total_tol: Complement singular values below ``total_tol`` are dropped.
        respect_degenerate: Never split a degenerate singular-value multiplet.
        degeneracy_tol: Relative tolerance that makes two singular values degenerate.
        reserve_tol: Relative tolerance of the rank reveal of the reserved set.
        in_place: Write into ``net`` and return it, instead of a copy. Only the window changes.

    Returns:
        ``(chain, report)``, with the orthogonality centre at ``site_ids[lo]``.

    Raises:
        ValueError: A negative radius, a chain with virtual nodes, a state in reweighted
            coordinates, or an observable that does not match the chain.
    """
    _refuse_if_gauged(net, "DMT truncation")
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}.")
    site_ids = pauli_site_ids(net) if site_ids is None else list(site_ids)
    ns = len(site_ids)
    if ns != len(net.nodes):
        raise ValueError("pauli_dmt_truncate acts on a chain with one qubit on every node, but "
                         "the network has virtual nodes. Use pauli_dmt_truncate_tree.")
    lo, hi = (0, ns - 1) if window is None else window
    tensors = _read_chain(net, site_ids)
    if window is None:
        tensors = _left_canonicalize(tensors)
    dtype = tensors[0].dtype

    obs_tensors = [[np.asarray(t) for t in obs] for obs in (observables or [])]
    for idx, otens in enumerate(obs_tensors):
        if len(otens) != ns:
            raise ValueError(f"observables[{idx}] has {len(otens)} sites but the chain has {ns}.")
        if any(t.ndim != 3 or t.shape[2] != 4 for t in otens):
            raise ValueError(f"observables[{idx}] must be (left, right, 4) tensors.")
        if (all(np.isrealobj(t) for t in tensors)
                and any(np.iscomplexobj(t) and np.max(np.abs(t.imag)) > 1e-12 for t in otens)):
            raise ValueError(
                f"observables[{idx}] has an imaginary part but the chain is real: the "
                "observable is not Hermitian.")

    # The left environments stay valid for the whole backward sweep, because truncating bond k
    # rewrites only the right bond of tensors[k-1], and lefts[m] reads tensors[0..m-1].
    lefts = [np.ones(1, dtype=dtype)]
    for k in range(hi):
        lefts.append(_lenv(lefts[k], tensors[k], _CAP_VEC))
    obs_lefts = [_obs_left_envs(tensors, otens, stop=hi) for otens in obs_tensors]

    # Radius R needs capenv[k + R], so the right environments are kept per site, not folded.
    capenv: Dict[int, np.ndarray] = {ns: np.ones(1, dtype=dtype)}
    obs_rights: List[Dict[int, np.ndarray]] = [
        {ns: np.ones((1, 1), dtype=np.result_type(dtype, otens[0].dtype))} for otens in obs_tensors]

    def fill_right(site: int) -> None:
        capenv[site] = _rvec(tensors[site], _CAP_VEC, capenv[site + 1])
        for oi, otens in enumerate(obs_tensors):
            obs_env = np.tensordot(otens[site], obs_rights[oi][site + 1], axes=([1], [1]))
            obs_rights[oi][site] = np.tensordot(tensors[site], obs_env, axes=([1, 2], [2, 1]))

    for k in range(ns - 1, hi, -1):
        fill_right(k)
    # The radius blocks at bond k read capenv[k + radius], which the previous bond left fresh for
    # radius >= 1. Only the radius-0 trace column and the observable images sit at site k itself.
    refresh_on_entry = radius == 0 or bool(obs_tensors)

    eta = 0.0
    weights: List[float] = []
    clipped: List[int] = []
    reserved_ranks: Dict[int, int] = {}
    for k in range(hi, lo, -1):
        if refresh_on_entry:
            fill_right(k)
        ld, r, p = tensors[k].shape
        mat = tensors[k].reshape(ld, r * p)
        discarded = 0.0
        if ld > chi:
            rad_l, rad_r = min(radius, k), min(radius, ns - k)
            # Split each radius block by level so that a clipped reservation falls back to a
            # smaller radius. The observables sit after the nearest-neighbour marginals.
            blocks: List[Tuple[int, int, np.ndarray]] = []
            for side, cols, rad in ((0, _radius_block_left(lefts, tensors, k, rad_l).conj(), rad_l),
                                    (1, _radius_block_right(capenv, tensors, k, rad_r), rad_r)):
                levels = _radius_levels(rad)
                for level in np.unique(levels):
                    blocks.append((int(level) if level <= 1 else int(level) + 1, side,
                                   cols[:, levels == level]))
            for oi in range(len(obs_tensors)):
                blocks.append((2, 0, obs_lefts[oi][k].conj()))
                blocks.append((2, 1, obs_rights[oi][k]))
            q_res = _ordered_orthonormal_basis(_priority_ordered_reserved(blocks), reserve_tol)
            if q_res.shape[1] > chi:
                q_res = q_res[:, :chi]
                clipped.append(k)
            rank = q_res.shape[1]
            reserved_ranks[k] = rank
            resid = mat - q_res @ (q_res.conj().T @ mat)
            u_c, s_c, _ = np.linalg.svd(resid, full_matrices=False)
            keep = min(max(min(chi, ld) - rank, 0), _tol_rank(s_c, rel_tol, total_tol))
            if respect_degenerate:
                keep = _respect_degenerate(s_c, keep, degeneracy_tol)
            # A near-null left singular vector of the residual is an arbitrary completion that
            # need not be orthogonal to q_res. Concatenated as it is, it makes proj non-orthonormal
            # and proj proj^dag not a projector, which corrupts a rank-limited bond at O(1).
            # Re-project off q_res, drop collapsed columns and re-orthonormalise.
            comp = u_c[:, :keep]
            comp = comp - q_res @ (q_res.conj().T @ comp)
            comp = comp[:, np.linalg.norm(comp, axis=0) > 1e-8]
            if comp.shape[1]:
                comp, _ = np.linalg.qr(comp)
            proj = np.concatenate([q_res, comp], axis=1) if comp.shape[1] else q_res
            # The bond splits orthogonally into q_res q_res^dag mat, kept in full, and resid, of
            # which the leading singular directions are kept, so the deficit is the tail.
            discarded = float(s_c[keep:] @ s_c[keep:])
        else:
            proj = np.eye(ld, dtype=dtype)
        total = float(np.vdot(mat, mat).real)
        kept = proj.conj().T @ mat
        if total > 0 and discarded > 0.0:
            weights.append(discarded / total)
            eta += float(np.sqrt(discarded / total))
        q_t, r_t = np.linalg.qr(kept.conj().T)
        tensors[k] = (q_t.conj().T).reshape(-1, r, p)
        tensors[k - 1] = np.transpose(
            np.tensordot(tensors[k - 1], proj @ (r_t.conj().T), axes=([1], [0])), (0, 2, 1))
        fill_right(k)

    out = net if in_place else deepcopy(net)
    _write_chain(out, site_ids, tensors, 0 if window is None else lo,
                 ns - 1 if window is None else hi)
    out.orthogonality_center_id = site_ids[lo]
    if clipped:
        warnings.warn(
            f"pauli_dmt_truncate: the reserved set exceeds chi={chi} at bonds {clipped[:8]}"
            f"{'...' if len(clipped) > 8 else ''} and was cut back from the outside in, so the "
            f"diameter-{2 * radius + 1} guarantee does not hold there. Raise chi (at least "
            f"{dmt_min_chi_for_radius(radius)} for radius {radius}) or lower the radius.",
            RuntimeWarning)
    return out, DMTReport(discarded=eta, discarded_weights=weights,
                          reserved_ranks=reserved_ranks, clipped_bonds=clipped)
