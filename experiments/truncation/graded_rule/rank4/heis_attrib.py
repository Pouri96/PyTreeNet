"""Defect attribution for the Heisenberg-MPO run (Test 2).

Exact identity (cf. 2609.18246, App. A Eq. 6, here for a Heisenberg circuit with per-cut defects):
  O_K = Gamma_K ... Gamma_1 O_0 + sum_k Gamma_K ... Gamma_{k+1} delta_k,   delta_k = Cut_k(Gamma_k O_{k-1}) - Gamma_k O_{k-1}
so  <sigma_0|O_K|sigma_0> - exact = sum_k <phi_{K-k}| delta_k |phi_{K-k}>,
phi_j = the Schroedinger state after the first j gates of the forward TEBD sequence (the gate sequence is a palindrome).
The per-cut terms c_k are summed per Strang step n (k in step n) and plotted against s = T - n dt.
"""
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import heis_mpo as H  # noqa: E402
import mpsenh as M  # noqa: E402

_PA = np.array([np.eye(2), M.X, M.Y, M.Z], dtype=complex)        # (4,2,2)


def forward_states(model, N, nsteps, dt):
    """List phi_j, j = 0..K, dense states after j gates of the forward sequence."""
    Gs = M.make_gates(model, N, dt)
    one = list(range(N - 1)) + list(range(N - 2, -1, -1))
    v = M.mps_to_dense(M.initial_mps(model, N))
    out = [v]
    for _ in range(nsteps):
        for b in one:
            v = M.dense_gate(v, Gs[b], b, N)
            out.append(v)
    return out


def apply_pauli_mpo(T, phi):
    """O |phi> for a Pauli-basis MPO (list of (l,4,r) real tensors) acting on a dense state; returns dense vector."""
    N = len(T)
    X = phi.reshape(1, -1)
    for i, t in enumerate(T):
        l, _, m = t.shape
        W = np.einsum('lpm,pts->lmts', t, _PA)                       # (l, m, 2, 2)  [bond in, bond out, out idx, in idx]
        A = 2 ** i
        B = 2 ** (N - i - 1)
        X4 = X.reshape(l, A, 2, B)
        X = np.einsum('lmts,lasb->matb', W, X4).reshape(m, A * 2 * B)
    return X.reshape(-1)


def expect(T, phi):
    return float(np.vdot(phi, apply_pauli_mpo(T, phi)).real)


def run_attribution(model, N, i0, chi, T, dt, cut, ref, pauli=3):
    """Run the Heisenberg MPO with `cut` and attribute the final error to the individual cuts.
    Returns dict: per-cut c (K,), per-step c_n (nsteps,), s_n, total, exact-minus-run check."""
    nsteps = int(round(T / dt))
    per = 2 * (N - 1)
    K = nsteps * per
    phis = forward_states(model, N, nsteps, dt)
    c = np.zeros(K)
    counter = [0]

    def hook(b, dirn, step, snap, th, A, B):
        k = counter[0] + 1
        counter[0] = k
        phi = phis[K - k]
        Tb = list(snap)
        Ta = list(snap)
        # before: theta split into two sites would change the bond; contract directly with a merged two-site tensor
        c[k - 1] = _delta_expect(snap, b, th, A, B, phi, N)

    res = H.run_heis(model, N, i0, chi, nsteps, dt, cut, ref, pauli=pauli, hook=hook)
    return dict(c=c, per=per, nsteps=nsteps, res=res, phis=None)


def _delta_expect(snap, b, th, A, B, phi, N):
    """<phi| (O_after - O_before) |phi> where only sites b,b+1 differ: apply the left chain once, then the two variants."""
    X = phi.reshape(1, -1)
    for i in range(b):
        X = _apply_site(X, snap[i], i, N)
    # two-site tensor th (l,4,4,r) applied to sites b,b+1 directly
    th2 = np.tensordot(A, B, axes=([2], [0])) - th
    l = th2.shape[0]
    r = th2.shape[3]
    W = np.einsum('lpqr,pts,qvu->lrtvsu', th2, _PA, _PA)                 # (l, r, t_b, t_b1, s_b, s_b1)
    Aa = 2 ** b
    Bb = 2 ** (N - b - 2)
    X6 = X.reshape(l, Aa, 2, 2, Bb)
    Y = np.einsum('lrtvsu,lasub->ratvb', W, X6).reshape(r, -1)
    for i in range(b + 2, N):
        Y = _apply_site(Y, snap[i], i, N)
    return float(np.vdot(phi, Y.reshape(-1)).real)


def _apply_site(X, t, i, N):
    l, _, m = t.shape
    W = np.einsum('lpm,pts->lmts', t, _PA)
    A = 2 ** i
    B = 2 ** (N - i - 1)
    return np.einsum('lmts,lasb->matb', W, X.reshape(l, A, 2, B)).reshape(m, A * 2 * B)
