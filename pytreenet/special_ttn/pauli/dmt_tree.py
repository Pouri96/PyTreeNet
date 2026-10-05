"""
Density-matrix truncation (DMT) of a fused Pauli tree.

The chain construction ports to a tree because its two ingredients do not depend on a linear
order. The reserved functionals at a cut are bilinear cap contractions of one side of the cut with
a few open Pauli legs, and a tree edge has two sides. What the linear order supplied was the
``radius`` qubits nearest the cut, which on a tree is the ball of nodes within distance ``radius``
of the edge inside each component.

The guarantee is the chain's in tree distance. Every operator on a connected node set of
tree-diameter at most ``2 * radius`` survives every truncation exactly. Radius 0 reserves the
trace alone. A virtual node has no Pauli leg, so an edge between two virtual nodes reserves the
trace alone at any radius, and the statement above holds for trees with a qubit on every node.

The walk is a post-order recursion with the orthogonality centre on the child whose bond to its
parent is cut, which is the DMT precondition: every other node is then isometric toward it.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np

from ...core.canonical_form import split_svd_contract_sv_to_neighbour
from ...util.tensor_splitting import SVDParameters
from .dmt import (DMTReport, _ordered_orthonormal_basis, _respect_degenerate, _tol_rank)
from .frame import _refuse_if_gauged

__all__ = ["dmt_tree_reserved_count", "dmt_min_chi_for_tree_radius", "pauli_dmt_truncate_tree"]

Memo = Dict[Tuple[str, Optional[str]], np.ndarray]


def _cap_vector(dim: int) -> np.ndarray:
    """The trace cap on one open leg: ``e_0`` on a dimension-4 Pauli leg, ``[1]`` on the trivial
    leg of a virtual node. The ``sqrt(2)`` factor is left out because only spans matter."""
    vec = np.zeros(dim)
    vec[0] = 1.0
    return vec


def _contract_open_legs(tree, node_id: str, selectors: Dict[int, np.ndarray]) -> np.ndarray:
    """The tensor of ``node_id`` with each open axis contracted against its selector. The
    neighbour axes keep their positions, because the open legs are the trailing axes and are
    contracted in descending order."""
    tensor = np.asarray(tree.tensors[node_id])
    for axis in sorted(tree.nodes[node_id].open_legs, reverse=True):
        tensor = np.tensordot(tensor, selectors[axis], axes=([axis], [0]))
    return tensor


def _is_qubit(tree, node_id: str) -> bool:
    """Whether the node carries a Pauli leg of dimension 4."""
    return any(np.asarray(tree.tensors[node_id]).shape[ax] == 4
               for ax in tree.nodes[node_id].open_legs)


def _neighbours_but(node, from_id: Optional[str]) -> List[str]:
    """The neighbours of ``node`` except ``from_id``, highest axis first, so that contracting
    them in this order keeps the remaining axis positions."""
    return sorted((nb for nb in node.neighbouring_nodes() if nb != from_id),
                  key=node.neighbour_index, reverse=True)


def _cap_env(tree, node_id: str, from_id: Optional[str], memo: Memo) -> np.ndarray:
    """The all-capped contraction of the component of ``node_id`` once the edge to ``from_id`` is
    cut, as a vector on that edge's bond. Memoised per edge, never across cuts, because a
    truncation changes the tensors."""
    key = (node_id, from_id)
    if key in memo:
        return memo[key]
    node = tree.nodes[node_id]
    shape = np.asarray(tree.tensors[node_id]).shape
    caps = {ax: _cap_vector(shape[ax]) for ax in node.open_legs}
    tensor = _contract_open_legs(tree, node_id, caps)
    for neighbour in _neighbours_but(node, from_id):
        tensor = np.tensordot(tensor, _cap_env(tree, neighbour, node_id, memo),
                              axes=([node.neighbour_index(neighbour)], [0]))
    memo[key] = tensor
    return tensor


def _string_env(tree, node_id: str, from_id: str, ball: set, sigma: Dict[str, int],
                memo: Memo) -> np.ndarray:
    """:func:`_cap_env` with the nodes of ``ball`` carrying the selector ``e_{sigma[node]}`` on
    their Pauli leg. It is the functional ``Tr(prod_n sigma_{mu_n}^{(n)} rho)`` with the rest of
    the component traced out, left open on the cut bond. The ball is connected and contains
    ``node_id``, so every other branch is served by the memoised all-capped environment."""
    node = tree.nodes[node_id]
    shape = np.asarray(tree.tensors[node_id]).shape
    selectors = {}
    for axis in node.open_legs:
        selector = _cap_vector(shape[axis])
        if shape[axis] == 4 and sigma.get(node_id):
            selector = np.zeros(4)
            selector[sigma[node_id]] = 1.0
        selectors[axis] = selector
    tensor = _contract_open_legs(tree, node_id, selectors)
    for neighbour in _neighbours_but(node, from_id):
        sub = (_string_env(tree, neighbour, node_id, ball, sigma, memo) if neighbour in ball
               else _cap_env(tree, neighbour, node_id, memo))
        tensor = np.tensordot(tensor, sub, axes=([node.neighbour_index(neighbour)], [0]))
    return tensor


def _ball(tree, endpoint: str, other_end: str, radius: int) -> List[Tuple[int, str]]:
    """``(depth, node_id)`` of the nodes within ``radius`` of the edge on ``endpoint``'s side, by
    BFS inside that component with ``endpoint`` at depth 0. Empty at radius 0."""
    if radius <= 0:
        return []
    out: List[Tuple[int, str]] = []
    seen = {other_end}
    frontier = [(endpoint, 0)]
    while frontier:
        node_id, depth = frontier.pop(0)
        if node_id in seen or depth >= radius:
            continue
        seen.add(node_id)
        out.append((depth, node_id))
        frontier.extend((nb, depth + 1) for nb in tree.nodes[node_id].neighbouring_nodes()
                        if nb not in seen)
    return out


def _reserved_columns(tree, endpoint: str, other_end: str, radius: int, memo: Memo
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """``(columns, levels)`` for one side of the edge: the reserved directions in the bond
    index and, for each, the smallest radius whose reservation contains it, 0 for the trace."""
    ball_pairs = _ball(tree, endpoint, other_end, radius)
    ball = {node_id for _, node_id in ball_pairs}
    variables = [(depth, node_id) for depth, node_id in ball_pairs if _is_qubit(tree, node_id)]
    columns, levels = [], []
    for assignment in product(range(4), repeat=len(variables)):
        sigma = {node_id: value for (_, node_id), value in zip(variables, assignment)}
        levels.append(max((depth + 1 for (depth, _), value in zip(variables, assignment) if value),
                          default=0))
        columns.append(np.asarray(_string_env(tree, endpoint, other_end, ball, sigma,
                                              memo)).reshape(-1))
    return np.stack(columns, axis=1), np.asarray(levels, dtype=int)


def dmt_tree_reserved_count(tree, radius: int) -> int:
    """The largest number of directions any edge of ``tree`` reserves at this radius.

    On a chain this is the constant ``2 * 4**radius``. On a tree it grows with the node degree,
    because the ball around an edge holds the endpoint and its other neighbours. Only the qubit
    nodes in a ball are variables, so a virtual node contributes the identity alone.
    """
    worst = 0
    for node_id, node in tree.nodes.items():
        if node.parent is None:
            continue
        total = 0
        for endpoint, other in ((node_id, node.parent), (node.parent, node_id)):
            variables = sum(1 for _, nid in _ball(tree, endpoint, other, radius)
                            if _is_qubit(tree, nid))
            total += 4 ** variables
        worst = max(worst, total)
    return worst


def dmt_min_chi_for_tree_radius(tree, radius: int) -> int:
    """The smallest ``max_bond_dim`` that leaves two complement directions on this tree,
    :func:`dmt_tree_reserved_count` plus two. It is 10 at radius 1 with a qubit on every node, and
    grows with the node degree as well as the radius."""
    return dmt_tree_reserved_count(tree, int(radius)) + 2


def _projector(tree, child_id: str, parent_id: str, mat: np.ndarray, chi: int, radius: int,
               rel_tol: float, total_tol: float, respect_degenerate: bool, degeneracy_tol: float,
               reserve_tol: float) -> Tuple[np.ndarray, float, int, bool]:
    """The reserved projector at the edge, its ``sqrt(eps)`` ledger value, the reserved rank and
    whether the reservation was clipped.

    ``mat`` is the child tensor with the parent bond as the row index and the centre on the child.
    The map is the one-sided reserved projector of the chain: an orthonormal basis of the reserved
    span, completed by the SVD of what that span does not explain.
    """
    memo: Memo = {}
    parent_cols, parent_lvl = _reserved_columns(tree, parent_id, child_id, radius, memo)
    child_cols, child_lvl = _reserved_columns(tree, child_id, parent_id, radius, memo)
    # Priority order: the trace, then the two sides alternating at each level, outer shells last.
    tagged = sorted([(int(lvl), 0, j) for j, lvl in enumerate(parent_lvl)]
                    + [(int(lvl), 1, j) for j, lvl in enumerate(child_lvl)])
    ordered = np.stack([(parent_cols if side == 0 else child_cols)[:, j] for _, side, j in tagged],
                       axis=1)
    basis = _ordered_orthonormal_basis(ordered, reserve_tol)
    room_for_reserved = max(1, min(chi - 1, mat.shape[0]))
    clipped = basis.shape[1] > room_for_reserved
    basis = basis[:, :room_for_reserved]
    residual = mat - basis @ (basis.conj().T @ mat)
    u_c, sing, _ = np.linalg.svd(residual, full_matrices=False)
    room = max(0, min(chi, mat.shape[0]) - basis.shape[1])
    keep = min(room, _tol_rank(sing, rel_tol, total_tol))
    if respect_degenerate:
        keep = min(room, _respect_degenerate(sing, keep, degeneracy_tol))
    # Re-project the complement off the reserved basis, drop collapsed columns and
    # re-orthonormalise. The singular vectors of the small singular values are arbitrary
    # completions that need not be orthogonal to the basis, and concatenated as they are they make
    # the projector wrong at O(1) on a rank-limited bond, which an evolved tree has.
    comp = u_c[:, :keep]
    comp = comp - basis @ (basis.conj().T @ comp)
    comp = comp[:, np.linalg.norm(comp, axis=0) > 1e-8]
    if comp.shape[1]:
        comp, _ = np.linalg.qr(comp)
    proj = np.concatenate([basis, comp], axis=1) if comp.shape[1] else basis
    total = float(np.vdot(mat, mat).real)
    dropped = float(np.sum(sing[keep:] ** 2))
    eta = float(np.sqrt(dropped / total)) if total > 0.0 and dropped > 0.0 else 0.0
    return proj, eta, basis.shape[1], clipped


def _truncate_subtree(node_id: str, tree, chi: int, radius: int, lossless: SVDParameters,
                      options: dict, ledger: dict) -> None:
    """Truncate every bond inside the subtree of ``node_id`` with the centre in and centre out.
    The recursion, the gauge transport and the centre invariant are those of the plain post-order
    truncation, and only the map on each bond differs."""
    for child_id in list(tree.nodes[node_id].children):
        tree.move_orthogonalization_center(child_id, preserve_legs_order=True)
        _truncate_subtree(child_id, tree, chi, radius, lossless, options, ledger)
        node = tree.nodes[child_id]
        tensor = np.asarray(tree.tensors[child_id])
        parent_axis = node.neighbour_index(node_id)
        moved = np.moveaxis(tensor, parent_axis, 0)
        mat = moved.reshape(moved.shape[0], -1)
        if mat.shape[0] > chi:
            proj, eta, rank, clipped = _projector(tree, child_id, node_id, mat, chi, radius,
                                                  **options)
            ledger["eta"] += eta
            if eta > 0.0:
                ledger["weights"].append(eta ** 2)
            ledger["ranks"][child_id] = rank
            if clipped:
                ledger["clipped"].append(child_id)
            projected = (proj @ (proj.conj().T @ mat)).reshape(moved.shape)
            tree.replace_tensor(child_id, np.moveaxis(projected, 0, parent_axis), new_shape=False)
        # The projected tensor has rank at most chi in the bond index, so this split only
        # re-factorises it into canonical form. A bond already within budget is split the same way.
        split_svd_contract_sv_to_neighbour(tree, child_id, node_id, lossless,
                                           preserve_legs_order=True)
        tree.orthogonality_center_id = node_id


def pauli_dmt_truncate_tree(tree, chi: int, radius: int = 1, rel_tol: float = 1e-12,
                            total_tol: float = 1e-12, respect_degenerate: bool = True,
                            degeneracy_tol: float = 1e-10, reserve_tol: float = 1e-12,
                            in_place: bool = False) -> Tuple[object, DMTReport]:
    """Truncate a fused Pauli tree to bond dimension ``chi`` with the DMT reservation.

    Every operator on a connected node set of tree-diameter at most ``2 * radius`` survives every
    truncation exactly, on a tree with a qubit on every node. A reservation that does not fit under
    ``chi`` is cut back from the outside in with a ``RuntimeWarning``, and no bond exceeds ``chi``.

    The reserved blocks are rebuilt at every edge, so the cost is ``O(N**2 chi**2)`` against the
    chain's ``O(N chi**2)``.

    Args:
        tree: A fused Pauli tree with real tensors.
        chi: The bond budget.
        radius: Nodes reserved around each edge, in tree distance.
        rel_tol: Complement singular values below ``rel_tol * s[0]`` are dropped.
        total_tol: Complement singular values below ``total_tol`` are dropped.
        respect_degenerate: Never split a degenerate singular-value multiplet.
        degeneracy_tol: Relative tolerance that makes two singular values degenerate.
        reserve_tol: Relative tolerance of the rank reveal of the reserved set.
        in_place: Truncate ``tree`` itself and return it, instead of a copy.

    Returns:
        ``(tree, report)``, with the orthogonality centre at the root. The report keys the
        reserved ranks and the clipped edges by the child node of the edge.

    Raises:
        ValueError: A negative radius or a state in reweighted coordinates.
    """
    _refuse_if_gauged(tree, "DMT truncation")
    chi, radius = int(chi), int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}.")
    out = tree if in_place else deepcopy(tree)
    root_id = out.root_id
    if out.orthogonality_center_id is None:
        out.canonical_form(root_id, preserve_legs_order=True)
    elif out.orthogonality_center_id != root_id:
        out.move_orthogonalization_center(root_id, preserve_legs_order=True)
    # DMT has already chosen the rank, so the re-factorisation must not truncate: a tolerance here
    # would drop reserved directions whose singular value is small, which they are meant to keep.
    lossless = SVDParameters(max_bond_dim=chi, rel_tol=1e-15, total_tol=0.0)
    options = dict(rel_tol=rel_tol, total_tol=total_tol, respect_degenerate=respect_degenerate,
                   degeneracy_tol=degeneracy_tol, reserve_tol=reserve_tol)
    ledger = {"eta": 0.0, "weights": [], "ranks": {}, "clipped": []}
    _truncate_subtree(root_id, out, chi, radius, lossless, options, ledger)
    out.orthogonality_center_id = root_id
    if ledger["clipped"]:
        warnings.warn(
            f"pauli_dmt_truncate_tree: the radius-{radius} reservation does not fit under chi="
            f"{chi} at the edges of {ledger['clipped'][:8]} and was cut back from the outside in, "
            "so the guarantee does not hold there. Raise chi or lower the radius. A tree ball "
            "grows with the node degree, so a radius that fits on a chain need not fit here.",
            RuntimeWarning)
    return out, DMTReport(discarded=ledger["eta"], discarded_weights=ledger["weights"],
                          reserved_ranks=ledger["ranks"], clipped_bonds=ledger["clipped"])
