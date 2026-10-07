"""Local-region lookahead: window marginals after evolving a small region under its own open-chain Hamiltonian.

For a window of k sites (start i) the region R is the L_R contiguous sites that contain it, with a sites of margin on
each side where the chain allows (otherwise the region is anchored at the chain end). The reduced state rho_R of the
candidate is evolved exactly by U_R(tau) = exp(-i H_R tau), H_R the sum of the bond terms inside R, and the window
marginal is the partial trace of U_R rho_R U_R^dag. For any tau the cost is that of one L_R-site marginal, and the
propagator has an exact finite light cone, so the horizon is not limited by operator entanglement.
"""
import os
import sys
if os.environ.get('PYLIBS'):
    sys.path.insert(0, os.environ['PYLIBS'])
import numpy as np
import jax
import jax.numpy as jnp
import mpsenh as M
import mfc


def _chain(Li, As, Rend):
    """Unnormalised reduced density matrix of the sites As, given the left norm environment Li and the right one Rend."""
    X = Li[:, :, None, None]
    for A in As:
        X = jnp.einsum('abST,bsd,atc->cdSsTt', X, A, jnp.conj(A))
        c, d, S, s_, T_, t_ = X.shape
        X = X.reshape(c, d, S * s_, T_ * t_)
    return jnp.einsum('cdST,cd->ST', X, Rend)


_chain_ckpt = jax.checkpoint(_chain)


def region_rhos(Ts, L):
    """rho of every L-site window of the MPS, normalised. One rematerialised chain per window start keeps the peak memory
    at a single start instead of all of them."""
    N = len(Ts)
    Le, Re = mfc._env_left(Ts), mfc._env_right(Ts)
    nrm = jnp.real(Le[N][0, 0])
    return jnp.stack([_chain_ckpt(Le[i], tuple(Ts[i:i + L]), Re[i + L]) for i in range(N - L + 1)]) / nrm


class RegionLookahead:
    def __init__(self, model, N, L_R, a, taus, ks=(1, 2)):
        self.N, self.L, self.a, self.taus, self.ks = N, L_R, a, tuple(taus), tuple(ks)
        self.starts = np.arange(N - L_R + 1)
        # propagators: the region starting at r0 only differs through the last-bond field of the chain end
        self.U = {}
        for r0 in (0, N - L_R):
            Hm = np.zeros((2 ** L_R, 2 ** L_R), dtype=complex)
            for b in range(r0, r0 + L_R - 1):
                Hm += np.kron(np.kron(np.eye(2 ** (b - r0)), M.h_bond(model, b, N)), np.eye(2 ** (r0 + L_R - b - 2)))
            ev, V = np.linalg.eigh(Hm)
            self.U[r0] = [jnp.asarray((V * np.exp(-1j * t * ev)) @ V.conj().T) for t in self.taus]
        self.windows = {k: [(i, int(np.clip(i - a, 0, N - L_R))) for i in range(N - k + 1)] for k in self.ks}

    def values(self, Ts):
        """dict (j, k) -> (windows, 2^k, 2^k) for tau index j. Vectorised over regions and times, grouped by window offset."""
        L, N = self.L, self.N
        rho = region_rhos(Ts, L)                                  # (N - L + 1, 2^L, 2^L)
        Ub = jnp.stack(self.U[0])
        Ul = jnp.stack(self.U[N - L])
        ev = jnp.einsum('tij,rjk,tlk->rtil', Ub, rho[:-1], jnp.conj(Ub))
        ev_last = jnp.einsum('tij,jk,tlk->til', Ul, rho[-1], jnp.conj(Ul))[None]
        ev = jnp.concatenate([ev, ev_last], axis=0)               # (regions, times, 2^L, 2^L)
        out = {}
        T = len(self.taus)
        for k in self.ks:
            groups = {}
            for idx, (i, r0) in enumerate(self.windows[k]):
                groups.setdefault(i - r0, []).append((idx, r0))
            pieces, order = [], []
            for off, lst in groups.items():
                r = L - off - k
                r0s = np.array([r0 for _, r0 in lst])
                X = ev[r0s].reshape(len(lst), T, 2 ** off, 2 ** k, 2 ** r, 2 ** off, 2 ** k, 2 ** r)
                pieces.append(jnp.einsum('ntasbaub->ntsu', X))
                order += [idx for idx, _ in lst]
            allv = jnp.concatenate(pieces, axis=0)                # (windows in group order, T, 2^k, 2^k)
            inv = np.argsort(np.array(order))
            allv = allv[inv]
            for j in range(T):
                out[(j, k)] = allv[:, j]
        return out

    def loss(self, vals, tgt, weights=None):
        tot = 0.0
        for key, v in vals.items():
            w = 1.0 if weights is None else weights.get(key, 1.0)
            tot = tot + w * jnp.mean(jnp.sum(jnp.abs(v - tgt[key]) ** 2, axis=(1, 2)))
        return tot
