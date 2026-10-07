"""Hamiltonian environments for a sweep, so the GLOBAL energy is a quadratic form in the two-site tensor.

H = sum_b h_b as a finite-state-machine MPO.  In the mixed canonical form of a sweep the two-site tensor
theta (l,2,2,r) has orthonormal left and right blocks, so
    <psi|H|psi> = <theta| L W_b W_{b+1} R |theta>
exactly, with L the environment of sites < b and R the environment of sites > b+1.  Left environments
are extended after a right-sweep cut and right environments after a left-sweep cut, one update per cut.
"""
import numpy as np
import mpsenh as M


def mpo_bulk(model):
    I, X, Y, Z = M.I2, M.X, M.Y, M.Z
    if model in M.FIELDS:
        hx, hz = M.FIELDS[model]
        W = np.zeros((3, 3, 2, 2), dtype=complex)
        W[0, 0], W[0, 1], W[0, 2] = I, Z, hx * X + hz * Z
        W[1, 2], W[2, 2] = Z, I
        return W
    if model == 'heis':
        W = np.zeros((5, 5, 2, 2), dtype=complex)
        W[0, 0], W[0, 1], W[0, 2], W[0, 3] = I, X, Y, Z
        W[1, 4], W[2, 4], W[3, 4], W[4, 4] = X, Y, Z, I
        return W
    raise ValueError(model)


class HEnv:
    def __init__(self, model, N):
        self.N = N
        Wb = mpo_bulk(model)
        D = Wb.shape[0]
        self.W = [Wb[0:1] if i == 0 else Wb for i in range(N)]
        self.W[N - 1] = self.W[N - 1][:, D - 1:D]
        self.L = [None] * (N + 1)
        self.R = [None] * (N + 1)
        self.L[0] = np.ones((1, 1, 1), dtype=complex)
        self.R[N] = np.ones((1, 1, 1), dtype=complex)

    def init(self, T):
        """Right environments of an all-right-isometric chain (the start of the first right sweep)."""
        for i in range(self.N - 1, 0, -1):
            self.R[i] = self.up_right(self.R[i + 1], T[i], i)

    def up_left(self, L, A, i):
        T = np.tensordot(L, A, axes=([1], [0]))                    # (a,j,s,k)
        T = np.tensordot(T, self.W[i], axes=([0, 2], [0, 3]))      # (j,k,b,t)
        T = np.tensordot(T, A.conj(), axes=([0, 3], [0, 1]))       # (k,b,l)
        return T.transpose(1, 0, 2)

    def up_right(self, R, A, i):
        T = np.tensordot(A, R, axes=([2], [1]))                    # (i,s,b,l)
        T = np.tensordot(T, self.W[i], axes=([1, 2], [3, 1]))      # (i,l,a,t)
        T = np.tensordot(T, A.conj(), axes=([1, 3], [2, 1]))       # (i,a,j)
        return T.transpose(1, 0, 2)

    def prep(self, b):
        """Fix the left and right pieces of H_eff for the bond (b, b+1): one cut applies it to several tensors."""
        self._P = np.tensordot(self.L[b], self.W[b], axes=([0], [0]))            # (i,j,c,t,s)
        self._Q = np.tensordot(self.W[b + 1], self.R[b + 2], axes=([1], [0]))    # (c,v,u,k,l)

    def apply1(self, C, L, i, R):
        """Single-site H_eff on a tensor C (l,2,r) at site i, with left/right environments L and R."""
        T = np.tensordot(L, C, axes=([1], [0]))                    # (a,j,s,k)
        T = np.tensordot(T, self.W[i], axes=([0, 2], [0, 3]))      # (j,k,b,t)
        return np.tensordot(T, R, axes=([1, 2], [1, 0]))           # (j,t,l)

    def apply_many(self, ths):
        X = np.stack(ths)                                                         # (n,i,s,u,k)
        T = np.tensordot(X, self._P, axes=([1, 2], [0, 4]))                      # (n,u,k,j,c,t)
        out = np.tensordot(T, self._Q, axes=([1, 2, 4], [2, 3, 0]))              # (n,j,t,v,l)
        return list(out)

    def after_right_cut(self, b, Tb):
        self.L[b + 1] = self.up_left(self.L[b], Tb, b)

    def after_left_cut(self, b, Tb1):
        self.R[b + 1] = self.up_right(self.R[b + 2], Tb1, b + 1)

    def apply(self, th, b):
        """H_eff theta for the two-site tensor on sites (b, b+1)."""
        return np.einsum('aij,isuk,acts,cdvu,dkl->jtvl', self.L[b], th, self.W[b], self.W[b + 1],
                         self.R[b + 2], optimize=True)

    def full_energy(self, T):
        """<psi|H|psi> / <psi|psi> of an arbitrary MPS, by a full left-to-right contraction."""
        E = np.ones((1, 1, 1), dtype=complex)
        n = np.ones((1, 1), dtype=complex)
        for i, A in enumerate(T):
            E = self.up_left(E, A, i)
            n = np.einsum('ij,isk,jsl->kl', n, A, A.conj())
        return float(E.real.reshape(-1)[0] / n.real.reshape(-1)[0])
