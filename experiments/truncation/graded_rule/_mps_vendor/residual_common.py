"""The residual-selection core shared by the RAGE integrators, chain and tree alike.

Topology-free: it works on a bond matricisation ``(m_full, ...)``. Every augmentation goes
through :meth:`ResidualSelectionMixin._augment_basis`.
"""

import warnings
from typing import Callable, Optional

import numpy as np
from numpy.linalg import qr
from scipy.linalg import expm

from .bug_util import warmup_passes

__all__ = ["ResidualSelectionMixin", "col_norms_sq", "pivoted_complement",
           "KRYLOV_REL_BREAK"]

#: Relative Arnoldi-breakdown floor for the two-site Krylov recurrence, which lets
#: ``residual_krylov_tol=0`` stop at the noise floor rather than normalise noise.
KRYLOV_REL_BREAK = 1e-10


def col_norms_sq(mat: np.ndarray) -> np.ndarray:
    """Squared 2-norm of each column of ``mat``, ``np.linalg.norm(mat, axis=0) ** 2``.

    Args:
        mat: Matrix whose columns are measured.

    Returns:
        Real vector of length ``mat.shape[1]``.
    """
    if mat.dtype.kind == "c":
        return (np.einsum("ij,ij->j", mat.real, mat.real)
                + np.einsum("ij,ij->j", mat.imag, mat.imag))
    return np.einsum("ij,ij->j", mat, mat)


def pivoted_complement(resid: np.ndarray, n_max: int, aug_tol: float,
                       ref_scale: Optional[float] = None,
                       dtype: type = complex) -> np.ndarray:
    """Select the dominant left subspace of a projected residual block.

    Column-pivoted modified Gram-Schmidt with an early stop at the floor
    ``max(aug_tol, max(m, K) * eps) * scale`` or at ``n_max`` directions. A Gram
    eigendecomposition would square the spectrum and resolve only to ``~sqrt(eps) * sigma_1``,
    above ``rel_tol``.

    Args:
        resid: ``(m_full, K)`` candidate block, already projected off the floor.
        n_max: Largest number of directions to return.
        aug_tol: Relative significance floor for keeping a direction.
        ref_scale: Leading scale of the unprojected candidates; it can only raise the floor,
            so rounding noise at a near-saturated bond is not admitted.
        dtype: Working dtype of the buffers.

    Returns:
        ``(m_full, k)`` orthonormal with ``k <= n_max``, not re-orthogonalised against
        the floor (the caller does that).
    """
    m, k_cols = resid.shape
    kmax = min(int(n_max), m, k_cols)
    if kmax <= 0:
        return np.zeros((m, 0), dtype=resid.dtype)
    eps = np.finfo(float).eps
    r_mat = np.ascontiguousarray(resid, dtype=dtype)
    cn2 = col_norms_sq(r_mat)                         # squared column norms, no Gram
    n0 = float(np.sqrt(cn2.max())) if k_cols else 0.0
    scale = n0 if ref_scale is None else max(n0, float(ref_scale))
    floor2 = max(scale * aug_tol, scale * max(m, k_cols) * eps) ** 2
    keep = cn2 > floor2                               # exact prefilter
    if not keep.any():
        return np.zeros((m, 0), dtype=dtype)
    r_mat = r_mat[:, keep]
    cn2 = cn2[keep]
    q_sel = np.empty((m, kmax), dtype=dtype)
    k = 0
    while k < kmax:
        p = int(np.argmax(cn2))
        if cn2[p] <= floor2:                          # significance plateau
            break
        c = r_mat[:, p].copy()
        if k:                                         # reorthogonalise against Q
            c -= q_sel[:, :k] @ (q_sel[:, :k].conj().T @ c)
        nrm = float(np.linalg.norm(c))
        if nrm * nrm <= floor2:                       # column lost to rounding
            cn2[p] = -1.0
            continue
        q = c / nrm
        q_sel[:, k] = q
        if np.iscomplexobj(r_mat):
            w = q.conj() @ r_mat                      # one O(m K) matvec
            cn2 = cn2 - (w.real ** 2 + w.imag ** 2)   # O(K) norm downdate
        else:
            w = q @ r_mat
            cn2 = cn2 - w * w
        cn2[p] = -1.0
        k += 1
    return q_sel[:, :k]


class ResidualSelectionMixin:
    """The per-instance half of the residual-selection core. Mix in first, as in
    ``class RAGE_MPS(ResidualSelectionMixin, BUG_MPS)``."""

    #: Factor in the propagator ``exp(_evo_exp_factor * dt * T)``. ``-1j`` is the
    #: Schroedinger convention; ``1.0`` keeps a real generator's selection real.
    _evo_exp_factor = -1j
    #: Working dtype of the Krylov and selection buffers. ``float`` only makes sense
    #: together with a real ``_evo_exp_factor`` and a real generator.
    _work_dtype = complex

    def _warn_degenerate_aug_cap(self) -> None:
        """Warn if ``max_aug_bond_dim <= max_bond_dim``.

        Once a bond reaches ``max_bond_dim`` the cap leaves no room for a complement, so
        the bond stops growing and the state cannot follow the dynamics.
        """
        aug = self.config.max_aug_bond_dim
        budget = self.config.max_bond_dim
        if aug is None or budget is None or float("inf") in (aug, budget):
            return
        if aug <= budget:
            warnings.warn(
                f"max_aug_bond_dim ({aug}) <= max_bond_dim ({budget}): once a bond reaches "
                "max_bond_dim no complement direction can be added, so the bond stops "
                "growing and the state cannot follow the dynamics. Leave max_aug_bond_dim "
                "unset (the default). A cap above max_bond_dim can also cost accuracy, "
                "below a threshold that depends on the problem.",
                stacklevel=3,
            )

    def _bond_budget(self, aug_dim: int, m: int) -> int:
        """How many bond directions to keep at a bond of full dimension ``m``.

        The floor ``aug_dim`` is always kept; the complement may grow it up to
        ``config.max_aug_bond_dim``. A cap of ``None`` or infinity imposes no ceiling,
        leaving the complement to ``config.rel_tol`` alone.
        """
        cap = self.config.max_aug_bond_dim
        if cap is None:
            cap = float("inf")
        return int(min(m, max(aug_dim, cap)))

    def _propagator_first_col(self, t_mat: np.ndarray, dt: float) -> np.ndarray:
        """First column ``exp(f dt T) e_1`` of the Arnoldi-projected propagator, by
        eigendecomposition if ``config.hermitian``, else by dense ``expm``."""
        factor = self._evo_exp_factor
        if self.config.hermitian:
            herm = 0.5 * (t_mat + t_mat.conj().T)
            evals, vecs = np.linalg.eigh(herm)
            return vecs @ (np.exp(factor * dt * evals) * vecs[0].conj())
        return expm(factor * dt * t_mat)[:, 0]

    def _warmup_passes(self) -> int:
        """``config.warmup_sweeps`` when a warm-up applies at this step, otherwise 0."""
        return warmup_passes(self.config, self._steps_done)

    def _pivoted_complement(self, resid: np.ndarray, n_max: int,
                            aug_tol: float,
                            ref_scale: Optional[float] = None) -> np.ndarray:
        """pivoted_complement, bound to this instance's working dtype."""
        return pivoted_complement(resid, n_max, aug_tol, ref_scale=ref_scale,
                                  dtype=self._work_dtype)

    def _augment_basis(self, q1: np.ndarray, blocks: np.ndarray, m_full: int,
                       aug_tol: Optional[float] = None) -> np.ndarray:
        """The augmented bond basis: the floor plus the candidates clearing the tolerance.

        Args:
            q1: ``(m_full, r)`` orthonormal floor, typically ``QR(old)``. Returned
                untouched when no complement survives.
            blocks: ``(m_full, K)`` candidate directions, the propagator-weighted
                Krylov blocks.
            m_full: Dimension of the bond space the complement is drawn from.
            aug_tol: Selection threshold; defaults to ``config.rel_tol`` -- add a
                direction exactly when it would survive truncation.

        Returns:
            ``(m_full, k)`` orthonormal basis with ``q1`` in its leading columns.
        """
        aug_dim = q1.shape[1]
        ceiling = self._bond_budget(aug_dim, m_full)
        if ceiling <= aug_dim or blocks.shape[1] == 0:
            return q1
        # Reference scale: the largest candidate column norm before the floor is
        # projected off. See the ref_scale note in pivoted_complement.
        ref_scale = float(np.sqrt(col_norms_sq(blocks).max()))
        q1_h = q1.conj().T
        resid = blocks - q1 @ (q1_h @ blocks)
        complement = self._pivoted_complement(
            resid, ceiling - aug_dim,
            float(self.config.rel_tol) if aug_tol is None else float(aug_tol),
            ref_scale=ref_scale)
        if complement.shape[1] == 0:
            return q1
        # Projecting off q1 here restores the orthogonality pivoted_complement leaves out.
        complement = complement - q1 @ (q1_h @ complement)
        complement, _ = qr(complement)
        return np.concatenate([q1, complement], axis=1)

    def _arnoldi_blocks(self, action: Callable[[np.ndarray], np.ndarray],
                        theta: np.ndarray, m_full: int) -> np.ndarray:
        """Candidate directions for widening one bond, weighted by the propagator.

        The Krylov directions of the two-site generator from the current two-site tensor,
        scaled by their coefficient in ``exp(f dt H_eff) Theta``. The recurrence stops once
        the next block's estimated contribution is below ``config.residual_krylov_tol``
        (``0`` runs to the noise floor, which long-range generators need), and
        re-orthogonalises twice for conditioning.

        Args:
            action: The two-site effective-Hamiltonian action, as a map from an array
                of ``theta``'s shape to the same shape, so it can be iterated.
            theta: The two-site tensor seeding the Krylov space.
            m_full: Dimension of the bond space the complement is drawn from. Also
                caps the depth -- past ``m_full`` steps the retained column space
                cannot grow.

        Returns:
            ``(m_full, K)`` matrix of weighted Krylov blocks. ``K = 0`` when the seed
            vanishes or is already invariant, i.e. when there is nothing to add.
        """
        dt = self.time_step_size
        rows = m_full
        shape = theta.shape
        v0 = theta.reshape(rows, -1)
        cols = v0.shape[1]
        beta0 = float(np.linalg.norm(v0))
        if beta0 == 0.0:
            return np.zeros((rows, 0), dtype=self._work_dtype)
        tol = float(self.config.residual_krylov_tol)
        max_steps = rows                                  # column space saturates here
        breakdown = np.finfo(float).eps * max(rows, 1)
        # One Fortran-ordered buffer, so each projection is a single BLAS-3 call.
        qmat = np.empty((rows * cols, 4), dtype=self._work_dtype, order="F")
        qmat[:, 0] = v0.reshape(-1) / beta0                # seed
        hess = np.zeros((4, 4), dtype=self._work_dtype)   # Hessenberg, grows on demand
        m = 0                                             # completed Galerkin columns
        first_col = None                                  # cached exp column
        for j in range(max_steps + 1):
            if j + 2 > hess.shape[0]:                     # double the buffers as needed
                grown = np.zeros((2 * hess.shape[0], 2 * hess.shape[0]),
                                 dtype=self._work_dtype)
                grown[:hess.shape[0], :hess.shape[0]] = hess
                hess = grown
            if j + 2 > qmat.shape[1]:
                grown_q = np.empty((qmat.shape[0], 2 * qmat.shape[1]),
                                   dtype=self._work_dtype, order="F")
                grown_q[:, :qmat.shape[1]] = qmat
                qmat = grown_q
            w = np.asarray(action(qmat[:, j].reshape(shape))).reshape(-1)
            w_pre = float(np.linalg.norm(w))              # cancellation reference
            basis = qmat[:, :j + 1]
            basis_h = basis.conj().T if np.iscomplexobj(basis) else basis.T
            h = basis_h @ w
            w = w - basis @ h
            h_corr = basis_h @ w                          # second pass corrects drift
            w = w - basis @ h_corr
            hess[:j + 1, j] = h + h_corr
            beta = float(np.linalg.norm(w))
            hess[j + 1, j] = beta
            m = j + 1
            first_col = None                              # hess[:m, :m] just changed
            # Breakdown floor relative to w_pre: below it w is pure cancellation noise,
            # and normalising it would corrupt the complement.
            if beta <= breakdown or beta <= KRYLOV_REL_BREAK * w_pre:
                break
            if tol > 0.0:                                 # adaptive Lanczos-exp stop
                first_col = self._propagator_first_col(hess[:m, :m], dt)
                if beta0 * beta * abs(first_col[m - 1]) < tol:
                    break
            if j < max_steps:
                qmat[:, j + 1] = w / beta
            else:
                break
        # Reuse the stopping estimate's column when the loop exited through the adaptive
        # stop; only the breakdown exit needs a fresh one.
        if first_col is None:
            first_col = self._propagator_first_col(hess[:m, :m], dt)
        coeffs = beta0 * first_col                        # c_0 .. c_{m-1}
        if m < 2:
            return np.zeros((rows, 0), dtype=self._work_dtype)
        return np.concatenate([abs(coeffs[n]) * qmat[:, n].reshape(rows, cols)
                               for n in range(1, m)], axis=1)
