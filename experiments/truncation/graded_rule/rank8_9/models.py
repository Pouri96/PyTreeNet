"""Proxy Hamiltonians and their 1+2-body RDM element sets, for the ranks 8/9 first move.

Qubit convention is that of ``mpsenh``: site 0 is the most significant bit of the dense index, |1> = occupied.
A *model* object offers

    N                number of qubits
    matvec(x)        H x on the full 2^N space (independent of the element bookkeeping, used to cross-check energies)
    ground_state()   dense normalised ground state (in the symmetry sector for the fermionic models)
    elem             an ``Elements`` object: values(v) -> vector R of 1+2-body expectation values, ``span`` (max - min + 1 of the
                     qubit indices an element touches), ``coef`` (so that E = sum_e coef_e R_e + const), ``kind`` (1 or 2)

Pauli models: elements are <sigma^a_i> and <sigma^a_i sigma^b_j> (a, b in X, Y, Z), 3N + 9 N (N-1) / 2 of them.
Fermionic (PPP) models: elements are gamma_PQ = <a+_P a_Q> and Gamma_{PQ,RS} = <a+_P a+_Q a_S a_R> (P<Q, R<S) over spin orbitals
P = 2 * position + spin, interleaved Jordan-Wigner (up, down per orbital), so a fermionic element whose indices lie in a window of w
qubits is a Pauli string inside that window.
"""
from __future__ import annotations

import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

PAULI = {'X': np.array([[0, 1], [1, 0]], dtype=complex), 'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
         'Z': np.array([[1, 0], [0, -1]], dtype=complex)}


# ---------------------------------------------------------------------------------------------- generic helpers
def popcount_table(N):
    x = np.arange(2 ** N, dtype=np.int64)
    c = np.zeros(2 ** N, dtype=np.int64)
    for q in range(N):
        c += (x >> q) & 1
    return c


def bit(N, q):
    """array of the occupation (0/1) of qubit q (site 0 = MSB) on the computational basis"""
    return (np.arange(2 ** N, dtype=np.int64) >> (N - 1 - q)) & 1


def pauli_term_sparse(N, term):
    """term = ((site, 'X'|'Y'|'Z'), ...): sparse matrix of the Pauli string on N qubits"""
    x = np.arange(2 ** N, dtype=np.int64)
    flip = 0
    phase = np.ones(2 ** N, dtype=complex)
    for s, p in term:
        b = (x >> (N - 1 - s)) & 1
        if p == 'X':
            flip |= 1 << (N - 1 - s)
        elif p == 'Y':
            flip |= 1 << (N - 1 - s)
            phase = phase * (1j * (1 - 2 * b))
        elif p == 'Z':
            phase = phase * (1 - 2 * b)
        else:
            raise ValueError(p)
    return sp.csr_matrix((phase, (x ^ flip, x)), shape=(2 ** N, 2 ** N))


def rdm2(v, N, i, j):
    t = np.moveaxis(v.reshape([2] * N), [i, j], [0, 1]).reshape(4, -1)
    return t @ t.conj().T


def rdm_sites(v, N, sites):
    """reduced density matrix on the given sites (sorted order = tensor order)"""
    t = np.moveaxis(v.reshape([2] * N), list(sites), list(range(len(sites)))).reshape(2 ** len(sites), -1)
    return t @ t.conj().T


def entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-14]
    return float(-np.sum(w * np.log(w)))


# ---------------------------------------------------------------------------------------------- Pauli models
class PauliElements:
    """1- and 2-body Pauli expectation values with coefficients of a Pauli-sum Hamiltonian."""
    def __init__(self, N, terms):
        self.N = N
        self.names, spans, kinds, coefs = [], [], [], []
        self.idx = {}
        for i in range(N):
            for a in 'XYZ':
                self.idx[((i, a),)] = len(self.names)
                self.names.append(((i, a),))
                spans.append(1)
                kinds.append(1)
        for i in range(N):
            for j in range(i + 1, N):
                for a in 'XYZ':
                    for b in 'XYZ':
                        key = ((i, a), (j, b))
                        self.idx[key] = len(self.names)
                        self.names.append(key)
                        spans.append(j - i + 1)
                        kinds.append(2)
        self.span = np.array(spans)
        self.kind = np.array(kinds)
        self.coef = np.zeros(len(self.names), dtype=complex)
        self.const = 0.0
        for c, term in terms:
            term = tuple(sorted(term))
            if len(term) == 0:
                self.const += c
                continue
            self.coef[self.idx[term]] += c
        self._pm = {k: [np.kron(PAULI[a], PAULI[b]) for a in 'XYZ' for b in 'XYZ'] for k in [0]}

    def values(self, v):
        N = self.N
        v = v / np.linalg.norm(v)
        out = np.zeros(len(self.names), dtype=complex)
        t = v.reshape([2] * N)
        # single site
        for i in range(N):
            r = np.moveaxis(t, i, 0).reshape(2, -1)
            rho = r @ r.conj().T
            for k, a in enumerate('XYZ'):
                out[self.idx[((i, a),)]] = np.trace(rho @ PAULI[a])
        pm = self._pm[0]
        for i in range(N):
            for j in range(i + 1, N):
                rho = rdm2(v, N, i, j)
                base = self.idx[((i, 'X'), (j, 'X'))]
                out[base:base + 9] = [np.trace(rho @ P) for P in pm]
        return out


class PauliModel:
    def __init__(self, N, terms, name):
        self.N, self.name = N, name
        self.terms = [(c, tuple(sorted(t))) for c, t in terms]
        self.elem = PauliElements(N, self.terms)
        # diagonal part (only Z strings) kept as a vector, off-diagonal part as sparse matrices
        diag = np.zeros(2 ** N)
        offd = {}
        zb = {}
        for c, t in self.terms:
            if all(p == 'Z' for _, p in t):
                d = np.full(2 ** N, float(c))
                for s, _ in t:
                    if s not in zb:
                        zb[s] = 1 - 2 * bit(N, s)
                    d = d * zb[s]
                diag += d
            else:
                offd[t] = offd.get(t, 0.0) + c
        H = sp.diags(diag).tocsr().astype(complex)
        for t, c in offd.items():
            H = H + c * pauli_term_sparse(N, t)
        self.H = H.tocsr()
        self._gs = None

    def matvec(self, x):
        return self.H @ x

    def ground_state(self):
        if self._gs is None:
            w, V = spla.eigsh(self.H, k=2, which='SA', tol=1e-13, ncv=40)
            o = np.argsort(w)
            self.e0, self.gap = float(w[o[0]]), float(w[o[1]] - w[o[0]])
            v = V[:, o[0]]
            v = v / np.linalg.norm(v)
            r = np.linalg.norm(self.H @ v - self.e0 * v)
            self.resid = float(r)
            self._gs = v.astype(complex)
        return self._gs


def tfim_terms(N, hx, hz, J=1.0):
    """the ``mpsenh`` convention: sum_b Z_b Z_{b+1} + hx X + hz Z on every site"""
    t = [(J, ((i, 'Z'), (i + 1, 'Z'))) for i in range(N - 1)]
    t += [(hx, ((i, 'X'),)) for i in range(N)]
    if hz != 0.0:
        t += [(hz, ((i, 'Z'),)) for i in range(N)]
    return t


def heis_terms(N):
    return [(1.0, ((i, a), (i + 1, a))) for i in range(N - 1) for a in 'XYZ']


def lr_ising_terms(N, alpha, h, perm=None):
    """H = sum_{i<j} |pi(i) - pi(j)|^-alpha Z_i Z_j + h sum X_i, where pi is the physical label of MPS site i (perm=None: identity)"""
    perm = np.arange(N) if perm is None else np.asarray(perm)
    t = []
    for i in range(N):
        for j in range(i + 1, N):
            t.append((float(abs(perm[i] - perm[j])) ** (-alpha), ((i, 'Z'), (j, 'Z'))))
    t += [(h, ((i, 'X'),)) for i in range(N)]
    return t


# ---------------------------------------------------------------------------------------------- fermionic (PPP) models
def jw_ops(N):
    """annihilation operators a_q = (prod_{k<q} Z_k) sigma^-_q as sparse matrices on the 2^N space (|1> occupied, site 0 = MSB)"""
    x = np.arange(2 ** N, dtype=np.int64)
    ops = []
    for q in range(N):
        bp = N - 1 - q
        occ = ((x >> bp) & 1) == 1
        xs = x[occ]
        # parity of occupied qubits k < q (bits above bp)
        hi = xs >> (bp + 1)
        par = np.zeros(len(xs), dtype=np.int64)
        h = hi.copy()
        while h.any():
            par ^= (h & 1)
            h >>= 1
        sign = 1.0 - 2.0 * par
        ops.append(sp.csr_matrix((sign.astype(complex), (xs ^ (1 << bp), xs)), shape=(2 ** N, 2 ** N)))
    return ops


class FermionElements:
    """gamma_PQ (all ordered (P, Q)) and Gamma_{(P<Q),(R<S)} = <a+_P a+_Q a_S a_R> (all ordered pair-pairs) of an N-spin-orbital system"""
    def __init__(self, N, h1, A, ops):
        self.N, self.ops = N, ops
        self.pairs = [(p, q) for p in range(N) for q in range(p + 1, N)]
        npair = len(self.pairs)
        self.npair = npair
        P, Q = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        g_span = (np.maximum(P, Q) - np.minimum(P, Q) + 1).ravel()
        pa = np.array(self.pairs)
        lo = np.minimum(pa[:, None, 0], pa[None, :, 0])
        hi = np.maximum(pa[:, None, 1], pa[None, :, 1])
        G_span = (hi - lo + 1).ravel()
        # index span must use all four indices: min over the four, max over the four
        self.span = np.concatenate([g_span, G_span])
        self.kind = np.concatenate([np.ones(N * N, int), 2 * np.ones(npair * npair, int)])
        # coefficient of gamma_PQ = <a+_P a_Q> is h1[P, Q]; of Gamma_{PQ,RS} is (1/2) * A~_{PQ,RS}
        c1 = h1.reshape(-1).astype(complex)
        pi = pa[:, 0]
        qi = pa[:, 1]
        Pm, Qm = pi[:, None], qi[:, None]
        Rm, Sm = pi[None, :], qi[None, :]
        At = A[Pm, Qm, Rm, Sm] - A[Qm, Pm, Rm, Sm] - A[Pm, Qm, Sm, Rm] + A[Qm, Pm, Sm, Rm]
        self.coef = np.concatenate([c1, 0.5 * At.reshape(-1)])
        self.const = 0.0
        self.n1 = N * N

    def values(self, v):
        v = v / np.linalg.norm(v)
        N = self.N
        av = np.stack([a @ v for a in self.ops])                       # (N, 2^N)
        gam = av.conj() @ av.T                                          # <a_P v | a_Q v> = <a+_P a_Q>
        phi = []
        for k, l in self.pairs:                                         # Phi_{kl} = a_l a_k v
            phi.append(self.ops[l] @ av[k])
        phi = np.stack(phi)
        Gam = phi.conj() @ phi.T                                        # <Phi_PQ | Phi_RS> = <a+_P a+_Q a_S a_R>
        return np.concatenate([gam.reshape(-1), Gam.reshape(-1)])


class PPPModel:
    """Pariser-Parr-Pople chain of L sites (2L qubits, interleaved up/down), half filled singlet-sector ground state.

    H = sum_{ij sigma} t_ij a+_i a_j + U sum n_i+ n_i- + 1/2 sum_{i!=j} V_ij (n_i - 1)(n_j - 1)    (Ohno V_ij),
    transformed to the orthonormal orbitals ``C`` (columns), placed in the order ``perm`` of the MPS.
    """
    def __init__(self, L=8, basis='site', perm=None, tpar=(2.6, 2.2), U=11.26, R=1.4, ne=None):
        self.L, self.N = L, 2 * L
        N = self.N
        t = np.zeros((L, L))
        for i in range(L - 1):
            t[i, i + 1] = t[i + 1, i] = -tpar[i % 2]
        Rij = R * np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
        V = 14.397 / np.sqrt((14.397 / U) ** 2 + Rij ** 2)
        np.fill_diagonal(V, 0.0)
        if basis == 'site':
            C = np.eye(L)
        elif basis == 'mo':
            e, C = np.linalg.eigh(t)
            # fix gauge: largest component positive
            for k in range(L):
                if C[np.argmax(np.abs(C[:, k])), k] < 0:
                    C[:, k] *= -1
        else:
            raise ValueError(basis)
        h_ao = t - np.diag(V.sum(1))
        g_ao = np.zeros((L, L, L, L))
        for a in range(L):
            g_ao[a, a, a, a] = U
            for c in range(L):
                if c != a:
                    g_ao[a, a, c, c] = V[a, c]
        perm = np.arange(L) if perm is None else np.asarray(perm)
        C = C[:, perm]
        self.C, self.perm, self.basis = C, perm, basis
        self.h = C.T @ h_ao @ C
        self.g = np.einsum('abcd,ap,bq,cr,ds->pqrs', g_ao, C, C, C, C, optimize=True)
        self.const = 0.5 * V.sum()
        self.ne = L if ne is None else ne
        self.name = f'ppp{L}_{basis}'
        # spin-orbital arrays
        P = np.arange(N)
        orb, spin = P // 2, P % 2
        h1 = np.where(spin[:, None] == spin[None, :], self.h[orb[:, None], orb[None, :]], 0.0)
        # A[P,Q,R,S] = (orb P orb R | orb Q orb S) delta(spin P, spin R) delta(spin Q, spin S)
        A = self.g[orb[:, None, None, None], orb[None, None, :, None], orb[None, :, None, None], orb[None, None, None, :]]
        A = A * (spin[:, None, None, None] == spin[None, None, :, None]) * (spin[None, :, None, None] == spin[None, None, None, :])
        self.h1, self.A = h1, A
        self.ops = jw_ops(N)
        self.opsH = [o.getH().tocsr() for o in self.ops]
        self.elem = FermionElements(N, h1, A, self.ops)
        self.elem.const = self.const
        # E_pq (spin summed) on the full space, for the matrix-free H
        self.E = {}
        for p in range(L):
            for q in range(L):
                m = None
                for s in (0, 1):
                    x = self.opsH[2 * p + s] @ self.ops[2 * q + s]
                    m = x if m is None else m + x
                self.E[p, q] = m.tocsr()
        self.pop = popcount_table(N)
        self._gs = None
        # sector: n_up = n_dn = ne/2
        nu = sum(bit(N, 2 * p) for p in range(L))
        nd = sum(bit(N, 2 * p + 1) for p in range(L))
        self.sector = np.nonzero((nu == self.ne // 2) & (nd == self.ne - self.ne // 2))[0]

    def matvec(self, x):
        """H x on the full space (the constant is included)"""
        L = self.L
        x = np.asarray(x, dtype=complex)
        Ex = np.stack([np.stack([self.E[p, q] @ x for q in range(L)]) for p in range(L)])      # (p, q, n)
        y = np.einsum('pq,pqn->n', self.h, Ex) - 0.5 * np.einsum('ps,psn->n', np.einsum('pqqs->ps', self.g), Ex)
        Z = np.tensordot(self.g.reshape(L * L, L * L), Ex.reshape(L * L, -1), axes=1).reshape(L, L, -1)
        for p in range(L):
            for q in range(L):
                y = y + 0.5 * (self.E[p, q] @ Z[p, q])
        return y + self.const * x

    def ground_state(self):
        if self._gs is not None:
            return self._gs
        sec = self.sector
        n = len(sec)
        full = np.zeros(2 ** self.N, dtype=complex)
        # build the sector Hamiltonian by restricting E_pq; the matvec of the sector is done through the restricted matrices
        Es = {k: m[sec][:, sec].tocsr() for k, m in self.E.items()}
        L = self.L
        g, h = self.g, self.h
        gs_ = np.zeros((L, L, L, L)) + g
        corr = np.einsum('pqqs->ps', g)

        def mv(x):
            x = np.asarray(x).reshape(-1)
            Ex = np.stack([np.stack([Es[p, q] @ x for q in range(L)]) for p in range(L)])   # (p, q, n)
            y = np.einsum('pq,pqn->n', h, Ex) - 0.5 * np.einsum('ps,psn->n', corr, Ex)
            Z = np.einsum('pqrs,rsn->pqn', g, Ex)
            for p in range(L):
                for q in range(L):
                    y = y + 0.5 * (Es[p, q] @ Z[p, q])
            return y + self.const * x
        op = spla.LinearOperator((n, n), matvec=mv, dtype=complex)
        w, V = spla.eigsh(op, k=2, which='SA', tol=1e-13, ncv=40)
        o = np.argsort(w)
        self.e0, self.gap = float(w[o[0]]), float(w[o[1]] - w[o[0]])
        full[sec] = V[:, o[0]]
        full /= np.linalg.norm(full)
        self.resid = float(np.linalg.norm(self.matvec(full) - self.e0 * full))
        self._gs = full
        return full

    def number_stats(self, v):
        v = v / np.linalg.norm(v)
        p = np.abs(v) ** 2
        n = self.pop
        mean = float(p @ n)
        var = float(p @ (n.astype(float) ** 2) - mean ** 2)
        szop = sum((bit(self.N, 2 * k) - bit(self.N, 2 * k + 1)) for k in range(self.L)) / 2.0
        smean = float(p @ szop)
        svar = float(p @ (szop ** 2) - smean ** 2)
        return mean, var, smean, svar


def orbital_mutual_information(v, L):
    """I_ij between spatial orbitals (qubit pairs 2i, 2i+1) of the dense state"""
    N = 2 * L
    v = v / np.linalg.norm(v)
    s1 = np.array([entropy(rdm_sites(v, N, [2 * i, 2 * i + 1])) for i in range(L)])
    I = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            s2 = entropy(rdm_sites(v, N, [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]))
            I[i, j] = I[j, i] = s1[i] + s1[j] - s2
    return I


def fiedler_order(I):
    L = I.shape[0]
    Lap = np.diag(I.sum(1)) - I
    w, V = np.linalg.eigh(Lap)
    f = V[:, 1]
    return np.argsort(f)
