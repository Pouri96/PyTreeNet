"""The two chemistry cells: pyrazine and the Holstein polaron. Same interface as ``SpinBoson``.

PYRAZINE: the four-mode S1/S2 linear vibronic coupling model of Raab, Worth, Meyer and
Cederbaum, J. Chem. Phys. 110, 936 (1999). Values in eV (verify against the paper before
citing): omega = 0.1139 (10a), 0.0739 (6a), 0.1258 (1), 0.1525 (9a); kappa^(1)/kappa^(2) =
-0.0964/0.1194 (6a), 0.0470/0.2012 (1), 0.1594/0.0484 (9a); lambda = 0.1825 (10a); E1 = 3.94,
E2 = 4.84. Units hbar = 1. Cutoffs [24, 20, 16, 12] keep the dense reference under 2e5
amplitudes. Start: vertical excitation with a 1% S1 admixture on the hub.

HOLSTEIN: J = 1, omega_0 = 1, g = 1.2, L = 4, phonon cutoff 10, electron starting on site 1, on
the star-chain tree.
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

from . import spin_boson as sb

HUB = sb.SPIN            # the two-level site keeps the spin-boson label so mode_tree applies


class _ModeModel:
    """Shared machinery: the Hamiltonian as term groups, the TTNO and the sparse matrix."""

    site_order: List[str]
    site_dims: Dict[str, int]

    def operators(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def terms(self) -> List[Tuple[float, str, Dict[str, str]]]:
        raise NotImplementedError

    def cache_key(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _mode_ops(d: int) -> Dict[str, np.ndarray]:
        creation, annihilation, number = bosonic_operators(dimension=d)
        return {f"n{d}": np.asarray(number, dtype=complex),
                f"x{d}": np.asarray(creation + annihilation, dtype=complex)}

    def hamiltonian(self, reference_state) -> Hamiltonian:
        ham = Hamiltonian()
        ham.conversion_dictionary.update(self.operators())
        for value, coeff, ops in self.terms():
            ham.add_term((Fraction(1), coeff, TensorProduct(dict(ops))))
            ham.coeffs_mapping[coeff] = complex(value)
        ham.include_identities(reference_state)
        return ham

    def sparse(self) -> sp.csr_matrix:
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

    def tree(self) -> Tuple[List[Tuple[str, str]], str]:
        raise NotImplementedError

    def initial_state(self):
        """The bond-1 product start on this model's tree."""
        edges, root = self.tree()
        return sb.build_state(edges, root, self.site_dims)


class Pyrazine(_ModeModel):
    """Four-mode LVC model of the pyrazine S1/S2 conical intersection.

    Hub basis: index 0 = S2 (the bright state the wavepacket starts in), index 1 = S1, so the
    default ``|0>`` product start of ``build_state`` IS the vertical excitation. With that basis
    ``sigma_z = diag(+1, -1)`` and the state-dependent couplings decompose as
    ``diag(k2, k1) = (k2 + k1)/2 * I + (k2 - k1)/2 * sigma_z``; the identity part is a genuine
    displacement term on the mode alone and is kept. Normal coordinates ``Q = (a + a^dag)/sqrt 2``.
    """

    OMEGA = {"10a": 0.1139, "6a": 0.0739, "1": 0.1258, "9a": 0.1525}
    KAPPA = {"6a": (-0.0964, 0.1194), "1": (0.0470, 0.2012), "9a": (0.1594, 0.0484)}   # (S1, S2)
    LAMBDA = 0.1825
    E1, E2 = 3.94, 4.84
    MODE_ORDER = ("10a", "6a", "1", "9a")

    #: :meth:`terms` returns ``TIME_SCALE * H``, so evolving to t = 1 equals evolving ``H`` to
    #: t = 30, where the S2 -> S1 transfer happens.
    TIME_SCALE = 30.0

    def __init__(self, dims: Sequence[int] = (24, 20, 16, 12), s1_admixture: float = 0.01):
        if len(dims) != 4:
            raise ValueError("pyrazine has four modes")
        self.dims = [int(d) for d in dims]
        self.s1_admixture = float(s1_admixture)
        self.mode_labels = [f"m{k}" for k in range(4)]
        self.names = dict(zip(self.mode_labels, self.MODE_ORDER))
        self.site_order = [HUB] + self.mode_labels
        self.site_dims = {HUB: 2, **dict(zip(self.mode_labels, self.dims))}

    def operators(self):
        sx, _, sz = pauli_matrices()
        ops = {"sx": np.asarray(sx, dtype=complex), "sz": np.asarray(sz, dtype=complex)}
        for d in set(self.dims):
            ops.update(self._mode_ops(d))
        return ops

    def terms(self):
        r = 1.0 / np.sqrt(2.0)
        out = [((self.E2 - self.E1) / 2.0, "gap", {HUB: "sz"})]
        for lab, d in zip(self.mode_labels, self.dims):
            name = self.names[lab]
            out.append((self.OMEGA[name], f"w_{name}", {lab: f"n{d}"}))
            if name in self.KAPPA:
                k1, k2 = self.KAPPA[name]
                out.append(((k2 + k1) / 2.0 * r, f"kmean_{name}", {lab: f"x{d}"}))
                out.append(((k2 - k1) / 2.0 * r, f"kdiff_{name}", {HUB: "sz", lab: f"x{d}"}))
            else:
                out.append((self.LAMBDA * r, f"lam_{name}", {HUB: "sx", lab: f"x{d}"}))
        return [(self.TIME_SCALE * value, name, ops) for value, name, ops in out]

    def tree(self):
        return sb.mode_tree(self.mode_labels)

    def initial_state(self):
        """Vertical excitation with an S1 admixture: ``cos(t/2)|S2> + sin(t/2)|S1>`` on the hub.

        The admixture is required: from the pure ``|S2> (x) |0>`` product, the only transfer term
        ``lambda sigma_x (x) Q_10a`` has zero expectation in both single-site projections, so BUG
        never leaves the product state.
        """
        state = super().initial_state()
        if self.s1_admixture > 0.0:
            node, tensor = state.nodes[HUB], state.tensors[HUB]
            leg = node.open_legs[0]
            moved = np.moveaxis(np.array(tensor), leg, -1)
            theta = 2.0 * np.arcsin(np.sqrt(self.s1_admixture))
            s2 = moved[..., 0].copy()
            moved[..., 0], moved[..., 1] = np.cos(theta / 2) * s2, np.sin(theta / 2) * s2
            state.tensors[HUB] = np.moveaxis(moved, -1, leg)
            state.canonical_form(state.root_id)
            state.normalize()
        return state

    def cache_key(self) -> str:
        return repr(("pyrazine", self.dims, self.s1_admixture, self.OMEGA, self.KAPPA,
                     self.LAMBDA, self.E1, self.E2, self.TIME_SCALE))


class Holstein(_ModeModel):
    """Spinless Holstein polaron: ``L`` electron sites on a chain, one phonon per site.

    ``H = -J sum (c_i^dag c_{i+1} + h.c.) + omega_0 sum b_i^dag b_i + g sum n_i (b_i + b_i^dag)``.
    One electron, so fermionic signs never enter; on a chain the nearest-neighbour hop is
    ``|1><0|_i (x) |0><1|_{i+1} + h.c.`` with no string either way. Tree: the star chain, electron
    sites on the spine, each site's phonon its leaf, rooted at the site the electron starts on.
    """

    def __init__(self, length: int = 4, d_ph: int = 10, J: float = 1.0, omega0: float = 1.0,
                 g: float = 1.2, start: int = 1):
        self.length, self.d_ph, self.J, self.omega0, self.g = length, int(d_ph), J, omega0, g
        self.start = start
        self.e = [f"e{i}" for i in range(length)]
        self.p = [f"p{i}" for i in range(length)]
        self.site_order = self.e + self.p
        self.site_dims = {**{s: 2 for s in self.e}, **{s: self.d_ph for s in self.p}}
        self.dims = [2] * length + [self.d_ph] * length

    def operators(self):
        ops = {"num": np.diag([0.0, 1.0]).astype(complex),
               "cdg": np.array([[0, 0], [1, 0]], dtype=complex),     # |1><0|
               "c":   np.array([[0, 1], [0, 0]], dtype=complex)}     # |0><1|
        ops.update(self._mode_ops(self.d_ph))
        return ops

    def terms(self):
        d = self.d_ph
        out = []
        for i in range(self.length):
            out.append((self.omega0, f"w{i}", {self.p[i]: f"n{d}"}))
            out.append((self.g, f"g{i}", {self.e[i]: "num", self.p[i]: f"x{d}"}))
        for i in range(self.length - 1):
            out.append((-self.J, f"hopA{i}", {self.e[i]: "cdg", self.e[i + 1]: "c"}))
            out.append((-self.J, f"hopB{i}", {self.e[i]: "c", self.e[i + 1]: "cdg"}))
        return out

    def tree(self):
        edges = [(self.e[i], self.e[i + 1]) for i in range(self.length - 1)]
        edges += [(self.e[i], self.p[i]) for i in range(self.length)]
        return edges, self.e[self.start]

    def initial_state(self):
        state = super().initial_state()
        # The electron on its start site: rotate that node's open leg from |0> to |1>.
        node, tensor = state.nodes[self.e[self.start]], state.tensors[self.e[self.start]]
        leg = node.open_legs[0]
        moved = np.moveaxis(np.array(tensor), leg, -1)
        moved[..., 1], moved[..., 0] = moved[..., 0], 0.0
        state.tensors[self.e[self.start]] = np.moveaxis(moved, -1, leg)
        state.canonical_form(state.root_id)
        state.normalize()
        return state

    def cache_key(self) -> str:
        return repr(("holstein", self.length, self.d_ph, self.J, self.omega0, self.g, self.start))


def exact_reference(model: _ModeModel, t_final: float, cache_dir: str = "_refcache"
                    ) -> Tuple[np.ndarray, float]:
    """``expm_multiply(-i T H)`` from the same initial state the tensor-network runs use."""
    initial = sb.aligned_vector(model.initial_state(), model.site_order)
    key = hashlib.sha1(repr((model.cache_key(), t_final)).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"ref_{key}.npy")
    if os.path.exists(path):
        return np.load(path), 0.0
    start = time.perf_counter()
    vec = expm_multiply(-1j * t_final * model.sparse(), initial)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, vec)
    return vec, time.perf_counter() - start


#: name -> (constructor, dt, t_final, chi grid, label). ``cells.py`` reads only the constructor
#: and the label; the other entries are unused.
MODELS = {
    "pyrazine": (lambda: Pyrazine(), 0.1, 30.0, (4, 6, 8, 10, 12, 16, 20, 24, 32),
                 r"pyrazine 4-mode, $d = [24,20,16,12]$, 1% S$_1$ admixture"),
    "holstein": (lambda: Holstein(), 0.02, 4.0, (4, 6, 8, 10, 12, 16, 20, 24, 32),
                 r"Holstein $L=4$, $d_{\rm ph}=10$"),
}
