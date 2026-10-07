"""Heisenberg-evolved local operators (U^dag)^m P U^m as MPOs, for the Strang step U used by every bench here.

U is the right sweep g_0 ... g_{N-2} followed by the left sweep g_{N-2} ... g_0 (g_b applied to sites b, b+1).
<psi| (U^dag)^m P U^m |psi> is the expectation of P in U^m |psi>, so the lookahead marginals of a state are plain
MPO expectation values and the operators can be built once and reused at every step.

An MPO tensor has shape (Dl, s_out, s_in, Dr). Conjugating by a two-site gate contracts two neighbouring tensors,
applies g^dag . g on the physical legs, and splits them again by an SVD truncated at `eps` (relative) and `dmax`.
"""
import itertools
import numpy as np
import mpsenh as M

PAULI = {'I': M.I2, 'X': M.X, 'Y': M.Y, 'Z': M.Z}


def identity_mpo(N):
    return [np.eye(2, dtype=complex).reshape(1, 2, 2, 1) for _ in range(N)]


def local_op(N, sites, labels):
    """MPO of the Pauli string labels[i] on sites[i], identity elsewhere."""
    W = identity_mpo(N)
    for s, lab in zip(sites, labels):
        W[s] = PAULI[lab].reshape(1, 2, 2, 1).astype(complex)
    return W


def conj_gate(W, b, g, eps, dmax):
    """W <- g^dag W g on sites (b, b+1); g has shape (2,2,2,2) acting as g[o1,o2,i1,i2]."""
    A, B = W[b], W[b + 1]
    Dl, Dr = A.shape[0], B.shape[3]
    X = np.tensordot(A, B, axes=([3], [0]))                       # (Dl, o1, i1, o2, i2, Dr)
    X = X.transpose(0, 1, 3, 2, 4, 5)                             # (Dl, o1, o2, i1, i2, Dr)
    G = g.reshape(4, 4)
    X = X.reshape(Dl, 4, 4, Dr)
    X = np.einsum('ab,lbcr,cd->ladr', G.conj().T, X, G)          # g^dag X g
    X = X.reshape(Dl, 2, 2, 2, 2, Dr).transpose(0, 1, 3, 2, 4, 5)  # (Dl, o1, i1, o2, i2, Dr)
    Mx = X.reshape(Dl * 4, 4 * Dr)
    U, s, Vh = np.linalg.svd(Mx, full_matrices=False)
    k = int(np.sum(s > eps * s[0]))
    k = max(1, min(k, dmax))
    W[b] = (U[:, :k] * s[:k]).reshape(Dl, 2, 2, k)
    W[b + 1] = Vh[:k].reshape(k, 2, 2, Dr)
    return W


def one_step(W, gates, N, eps, dmax):
    """P -> U^dag P U : conjugate by the gates in REVERSE order of application."""
    for b in range(N - 1):                 # undo the left sweep: it applied N-2 ... 0, so reversed is 0 ... N-2
        conj_gate(W, b, gates[b], eps, dmax)
    for b in range(N - 2, -1, -1):         # then the right sweep: it applied 0 ... N-2, reversed is N-2 ... 0
        conj_gate(W, b, gates[b], eps, dmax)
    return W


def damp(W, g):
    """Multiply every non-identity Pauli component on every site by exp(-g) (dissipation-assisted operator evolution)."""
    if g == 0:
        return W
    f = np.exp(-g)
    eye = np.eye(2, dtype=complex)
    for j, t in enumerate(W):
        c0 = (t[:, 0, 0, :] + t[:, 1, 1, :]) / 2
        W[j] = f * t + (1 - f) * c0[:, None, None, :] * eye[None, :, :, None]
    return W


def brick_stepper(GE, GO):
    """Stepper for the coarse brickwork Strang step V = L_even(GE) L_odd(GO) L_even(GE); the layer sequence is a palindrome,
    so conjugating by the layers in reverse order is the same sequence."""
    def step(W, gates, N, eps, dmax):
        for b in range(0, N - 1, 2):
            conj_gate(W, b, GE[b], eps, dmax)
        for b in range(1, N - 1, 2):
            conj_gate(W, b, GO[b], eps, dmax)
        for b in range(0, N - 1, 2):
            conj_gate(W, b, GE[b], eps, dmax)
        return W
    return step


def evolve(W0, gates, N, M_steps, times, eps=1e-10, dmax=64, gamma=0.0, stepper=None):
    """dict m -> MPO of (U^dag)^m P U^m for m in `times` (m = 0 is P itself).

    gamma > 0 damps the Pauli weight of the operator after every step (a per-site factor exp(-gamma) on X, Y, Z).
    """
    out, W = {}, [w.copy() for w in W0]
    if 0 in times:
        out[0] = [w.copy() for w in W]
    for m in range(1, M_steps + 1):
        W = damp((stepper or one_step)(W, gates, N, eps, dmax), gamma)
        if m in times:
            out[m] = [w.copy() for w in W]
    return out


def operator_set(N, kmax=2):
    """All Pauli strings on contiguous windows of 1..kmax sites with no identity at the window ends:
    (sites, labels). Windows of two or more sites include strings like XIX only when the middle is I."""
    ops = []
    for k in range(1, kmax + 1):
        for i in range(N - k + 1):
            for lab in itertools.product('IXYZ', repeat=k):
                if lab[0] == 'I' or lab[-1] == 'I':
                    continue
                ops.append((tuple(range(i, i + k)), ''.join(lab)))
    return ops


def mpo_to_dense(W):
    N = len(W)
    X = W[0].reshape(2, 2, -1)
    for j in range(1, N):
        w = W[j]
        X = np.einsum('abk,kcdl->acbdl', X, w).reshape(X.shape[0] * 2, X.shape[1] * 2, -1)
    return X[:, :, 0]
