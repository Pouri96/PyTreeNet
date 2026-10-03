"""Spin-boson on an ML-MCTDH tree, each bath mode with its own local dimension.

.. math::

    H = \\frac{\\varepsilon}{2}\\sigma_z + \\frac{\\Delta}{2}\\sigma_x
        + \\sum_k \\omega_k a_k^\\dagger a_k
        + \\sigma_z \\sum_k g_k (a_k + a_k^\\dagger)

One list of terms builds both the TTNO and the sparse matrix of the exact reference. With
unequal local dimensions:

* each dimension gets its own operator symbols (``"n16"``, ``"x16"``), since the conversion
  dictionary is global;
* ``include_identities`` is given the tree, which registers the ``"I1"`` of virtual nodes;
* the dense reference uses an explicit ordered list of physical nodes.
"""
from __future__ import annotations

import hashlib
import os
import time
from fractions import Fraction
from math import prod
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from pytreenet.operators.common_operators import bosonic_operators, pauli_matrices
from pytreenet.operators.hamiltonian import Hamiltonian
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.ttno.ttno_class import TTNO

from _rage_vendor import ttns_from_tree

SPIN = "s"


# ---------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------


def ohmic_modes(n_modes: int, alpha: float, w_c: float = 1.0, w_max: float = 4.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Equidistant star discretisation of ``J(w) = 2 alpha w exp(-w / w_c)``.

    Returns ``(w, g)`` with ``w_q`` at the midpoint of bin ``q`` (a mode at ``w = 0`` would
    need an infinite local dimension) and ``g_q = sqrt(J(w_q) dw)``.
    """
    dw = w_max / n_modes
    w = (np.arange(n_modes) + 0.5) * dw
    j = 2.0 * alpha * w * np.exp(-w / w_c)
    return w, np.sqrt(j * dw)


class SpinBoson:
    """The model, its TTNO and its sparse matrix, from one list of terms.

    ``site_order`` is the dense index order: entry 0 is the most significant index, matching the
    leftmost Kronecker factor in :meth:`sparse` and the first axis of :func:`aligned_vector`.
    """

    def __init__(self, w: Sequence[float], g: Sequence[float], dims: Sequence[int],
                 delta: float = 0.2, eps: float = 0.0):
        if not len(w) == len(g) == len(dims):
            raise ValueError(f"w, g and dims must agree in length; got "
                             f"{len(w)}, {len(g)}, {len(dims)}.")
        self.w = list(map(float, w))
        self.g = list(map(float, g))
        self.dims = [int(d) for d in dims]
        self.delta = float(delta)
        self.eps = float(eps)
        self.n_modes = len(self.w)
        self.mode_labels = [f"m{k}" for k in range(self.n_modes)]
        self.site_order = [SPIN] + self.mode_labels
        self.site_dims = {SPIN: 2, **dict(zip(self.mode_labels, self.dims))}

    @staticmethod
    def _mode_ops(d: int) -> Dict[str, np.ndarray]:
        """Number operator and position quadrature at dimension ``d``, symbols tagged by ``d``."""
        creation, annihilation, number = bosonic_operators(dimension=d)
        return {f"n{d}": np.asarray(number, dtype=complex),
                f"x{d}": np.asarray(creation + annihilation, dtype=complex)}

    def operators(self) -> Dict[str, np.ndarray]:
        """Every symbol this model uses, with its matrix."""
        sx, _, sz = pauli_matrices()
        ops = {"sx": np.asarray(sx, dtype=complex), "sz": np.asarray(sz, dtype=complex)}
        for d in set(self.dims):
            ops.update(self._mode_ops(d))
        return ops

    def terms(self) -> List[Tuple[float, str, Dict[str, str]]]:
        """``(value, coefficient symbol, {site: operator symbol})`` for every term of ``H``.
        Prefactors live in the coefficient, so modes of equal dimension share operator symbols."""
        out: List[Tuple[float, str, Dict[str, str]]] = []
        if self.eps != 0.0:
            out.append((self.eps / 2.0, "hz", {SPIN: "sz"}))
        out.append((self.delta / 2.0, "hx", {SPIN: "sx"}))
        for k, (wk, gk, dk) in enumerate(zip(self.w, self.g, self.dims)):
            lab = self.mode_labels[k]
            out.append((wk, f"w{k}", {lab: f"n{dk}"}))
            out.append((gk, f"g{k}", {SPIN: "sz", lab: f"x{dk}"}))
        return out

    def hamiltonian(self, reference_state) -> Hamiltonian:
        """The PyTreeNet ``Hamiltonian``, with identities registered from the tree."""
        ham = Hamiltonian()
        ham.conversion_dictionary.update(self.operators())
        for value, coeff, ops in self.terms():
            ham.add_term((Fraction(1), coeff, TensorProduct(dict(ops))))
            ham.coeffs_mapping[coeff] = complex(value)
        ham.include_identities(reference_state)
        return ham

    def sparse(self) -> sp.csr_matrix:
        """``H`` as a sparse matrix in ``site_order``, site 0 the leftmost Kronecker factor."""
        ops = self.operators()
        dim = self.total_dim()
        out = sp.csr_matrix((dim, dim), dtype=complex)
        for value, _, term_ops in self.terms():
            mat = None
            for lab in self.site_order:
                local = (sp.csr_matrix(ops[term_ops[lab]]) if lab in term_ops
                         else sp.identity(self.site_dims[lab], dtype=complex, format="csr"))
                mat = local if mat is None else sp.kron(mat, local, format="csr")
            out = out + value * mat
        return out.tocsr()

    def total_dim(self) -> int:
        return prod(self.site_dims[lab] for lab in self.site_order)


# ---------------------------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------------------------


def mode_tree(mode_labels: Sequence[str]) -> Tuple[List[Tuple[str, str]], str]:
    """The ``m = 3`` mode-combination tree with the spin at the root.

    Four modes: ``root = [spin, v1, v2]``, ``v1 = [a, b]``, ``v2 = [c, d]`` (8 nodes). Six
    modes: ``v1 = [a, [b, c]]``, ``v2 = [d, [e, f]]`` (12 nodes). Eight modes: four pairs.
    """
    labels = list(mode_labels)
    if len(labels) == 4:
        a, b, c, d = labels
        return ([("r", SPIN), ("r", "v1"), ("r", "v2"),
                 ("v1", a), ("v1", b), ("v2", c), ("v2", d)], "r")
    if len(labels) == 6:
        a, b, c, d, e, f = labels
        return ([("r", SPIN), ("r", "v1"), ("r", "v2"),
                 ("v1", a), ("v1", "v3"), ("v3", b), ("v3", c),
                 ("v2", d), ("v2", "v4"), ("v4", e), ("v4", f)], "r")
    if len(labels) == 8:
        a, b, c, d, e, f, g, h = labels
        return ([("r", SPIN), ("r", "v1"), ("r", "v2"),
                 ("v1", "v3"), ("v1", "v4"), ("v3", a), ("v3", b), ("v4", c), ("v4", d),
                 ("v2", "v5"), ("v2", "v6"), ("v5", e), ("v5", f), ("v6", g), ("v6", h)], "r")
    raise ValueError(f"the m=3 tree is built for four, six or eight modes; got {len(labels)}.")


def balanced_assignment(model: SpinBoson) -> List[str]:
    """The mode at each position of :func:`mode_tree`; the Hamiltonian is unchanged.

    Large and small dimensions are paired so the two subtrees have similar dimension products.
    With ``d`` sorted descending, six modes pair ``(d0, d5)`` and ``(d1, d4)``; four modes keep
    their order.
    """
    if model.n_modes == 4:
        return list(model.mode_labels)
    order = sorted(range(model.n_modes), key=lambda k: -model.dims[k])
    if model.n_modes == 8:
        pos = [order[0], order[7], order[3], order[4], order[1], order[6], order[2], order[5]]
    else:
        pos = [order[2], order[0], order[5], order[3], order[1], order[4]]
    return [model.mode_labels[k] for k in pos]


def build_state(edges, root, site_dims: Dict[str, int]):
    """The bond-1 product state ``spin-up (x) vacuum``; nodes not in ``site_dims`` are virtual."""
    labels = {u for e in edges for u in e}
    phys_dims = {lab: site_dims.get(lab, 1) for lab in labels}
    state, _ = ttns_from_tree(edges, root, node_ids={lab: lab for lab in labels},
                              phys_dims=phys_dims, bond_dim=1)
    state.canonical_form(state.root_id)
    state.normalize()
    return state


def make_ttno(model: SpinBoson, edges, root):
    """A TTNO built on a fresh state, never on the object that will be evolved."""
    template = build_state(edges, root, model.site_dims)
    return TTNO.from_hamiltonian(model.hamiltonian(template), template)


# ---------------------------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------------------------


def open_dims(state) -> Dict[str, int]:
    """Physical dimension carried by each node's own open legs; 1 where it has none."""
    out = {}
    for nid in state.nodes:
        node = state.nodes[nid]
        shape = state.tensors[nid].shape
        out[nid] = int(prod(shape[i] for i in node.open_legs)) if node.open_legs else 1
    return out


def aligned_vector(state, site_order: Sequence[str]) -> np.ndarray:
    """Dense state vector with ``site_order[0]`` as the most significant index.

    Nodes outside ``site_order`` must have dimension 1.
    """
    vec, order = state.to_vector(to_copy=True)
    dims = open_dims(state)
    tensor = np.asarray(vec).reshape([dims[nid] for nid in order])
    pos = {nid: i for i, nid in enumerate(order)}
    phys_axes = [pos[nid] for nid in site_order]
    rest = [i for i in range(len(order)) if i not in set(phys_axes)]
    bad = [(order[i], dims[order[i]]) for i in rest if dims[order[i]] != 1]
    if bad:
        raise ValueError(f"nodes outside site_order carry open dimensions {bad}; "
                         f"site_order is incomplete.")
    return np.transpose(tensor, phys_axes + rest).reshape(-1)


def infidelity(vec: np.ndarray, reference: np.ndarray) -> float:
    """``1 - |<ref|vec>|^2 / (<ref|ref><vec|vec>)``, insensitive to norm and global phase."""
    overlap = np.vdot(reference, vec)
    denom = float(np.vdot(reference, reference).real * np.vdot(vec, vec).real)
    return float(abs(1.0 - abs(overlap) ** 2 / denom))


def parameters(state) -> int:
    """Total stored entries."""
    return int(sum(np.asarray(state.tensors[nid]).size for nid in state.nodes))


def widest_bond(state) -> int:
    """Largest bond dimension."""
    return max((np.asarray(state.tensors[nid]).shape[state.nodes[nid].neighbour_index(x)]
                for nid in state.nodes
                for x in state.nodes[nid].neighbouring_nodes()), default=1)


def exact_reference(model: SpinBoson, edges, root, t_final: float,
                    cache_dir: str = "_refcache") -> Tuple[np.ndarray, float]:
    """``exp(-i T H)`` applied to the start vector with ``expm_multiply``, cached on disk.

    Returns ``(vector, seconds)``; seconds is 0 when read from the cache.
    """
    initial = aligned_vector(build_state(edges, root, model.site_dims), model.site_order)
    key = hashlib.sha1(repr((model.w, model.g, model.dims, model.delta, model.eps,
                             list(model.site_order), t_final)).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"ref_{key}.npy")
    if os.path.exists(path):
        return np.load(path), 0.0
    start = time.perf_counter()
    vec = expm_multiply(-1j * t_final * model.sparse(), initial)
    secs = time.perf_counter() - start
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, vec)
    return vec, secs
