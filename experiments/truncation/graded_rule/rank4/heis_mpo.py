"""Heisenberg-picture MPO (Pauli-basis MPS, d = 4, basis I,X,Y,Z) Strang evolution of a local observable, with
interchangeable cuts: SVD-raw, SVD-renorm, DMT with a swappable reference covector (DMT-I, DMT-sigma0, DMT-sigma(s)).

The operator is O = sum_P c_P P with real c.  One Strang step is the palindrome [0..N-2, N-2..0] of gate
conjugations G^dag . G, which is the same bond order as the Schroedinger TEBD in mpsenh.run_tebd, so a single
run gives O(n dt) for every n:  O(n) = U_step^dag O(n-1) U_step,  <Neel|O(n)|Neel> = <Z_i0>(n dt).
Because the gate superoperator R is real orthogonal, the Frobenius norm of c is conserved by the exact dynamics and
the usual mixed-canonical SVD cut is optimal in the Frobenius (Hilbert-Schmidt) norm.

DMT cut at the bond between sites b and b+1 (theta over sites b, b+1; left tensors left-isometric, right tensors
right-isometric).  With covectors v_i = (1, <X_i>, <Y_i>, <Z_i>) of a product reference state sigma_ref:
    kL[l]  = contraction of sites < b with v      (so K_L[(l,p),p'] = kL[l] delta_{pp'})
    kR[r]  = contraction of sites > b+1 with v    (so K_R[(q,r),q'] = delta_{qq'} kR[r])
The cut preserves exactly  Tr[(X_{<=b+1} (x) sigma_ref^{>b+1}) O]  and  Tr[(sigma_ref^{<b} (x) X_{>=b}) O]  for all X.
(1707.01506 Sec. III.B with the identity covector replaced by v.)  No connected-correlator subtraction.
"""
import sys
from pathlib import Path
import time
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
import _paths  # noqa: F401,E402  (this repository's pytreenet and rule/ onto sys.path)
import mpsenh as M  # noqa: E402
import pauli_prop as PP  # noqa: E402

EPS_S = 1e-14   # relative singular-value floor, as in mpsenh.EPS_KEEP-style cuts


# ----------------------------------------------------------------------------------------------- gates and refs
def pauli_gates(model, N, dt):
    """Real (4,4,4,4) superoperators S[q_b, q_b1, p_b, p_b1] with  G^dag P_p G = sum_q S[q, p] P_q, per bond b."""
    out = []
    for G in M.make_gates(model, N, dt):
        R = PP.transfer(G.reshape(4, 4))                    # R[c_in, c_out], c = p_b + 4 p_b1
        out.append(np.ascontiguousarray(R.reshape(4, 4, 4, 4).transpose(3, 2, 1, 0)))
    return out


def initial_covectors(model, N):
    """v_i = (1, <X>, <Y>, <Z>) of the initial product state (Neel for all models used here)."""
    v = np.zeros((N, 4))
    for i, t in enumerate(M.initial_mps(model, N)):
        psi = t.reshape(2)
        v[i, 0] = 1.0
        for k, P in enumerate((M.X, M.Y, M.Z)):
            v[i, k + 1] = np.vdot(psi, P @ psi).real
    return v


def identity_covectors(N):
    v = np.zeros((N, 4))
    v[:, 0] = 1.0
    return v


def initial_operator(N, i0, pauli):
    T = []
    for i in range(N):
        t = np.zeros((1, 4, 1))
        t[0, pauli if i == i0 else 0, 0] = 1.0
        T.append(t)
    return T


def contract_cov(T, v):
    """<sigma_ref | O> = sum_P c_P prod_i v_i[p_i]."""
    x = np.ones(1)
    for i, t in enumerate(T):
        x = np.einsum('l,lpa,p->a', x, t, v[i])
    return float(x[0])


# ----------------------------------------------------------------------------------------------- cuts
def _split(U, s, Vh, dirn, l, r):
    k = len(s)
    if dirn == 'R':
        return U.reshape(l, 4, k), (s[:, None] * Vh).reshape(k, 4, r)
    return (U * s).reshape(l, 4, k), Vh.reshape(k, 4, r)


def _thin_svd(theta):
    l, r = theta.shape[0], theta.shape[3]
    U, s, Vh = np.linalg.svd(theta.reshape(4 * l, 4 * r), full_matrices=False)
    k = max(1, int(np.sum(s > EPS_S * s[0]))) if s[0] > 0 else 1
    return U[:, :k], s[:k], Vh[:k], s[k:]


class SVDCut:
    """Plain Frobenius truncation.  renorm=False: leave the norm drop; renorm=True: restore ||theta||."""
    needs_env = False

    def __init__(self, renorm):
        self.renorm = renorm
        self.name = 'svd_renorm' if renorm else 'svd_raw'

    def __call__(self, theta, chi, dirn, ctx):
        l, r = theta.shape[0], theta.shape[3]
        U, s, Vh, tail = _thin_svd(theta)
        full = np.sqrt(np.sum(s ** 2) + np.sum(tail ** 2))
        k = min(chi, len(s))
        disc = float(np.sum(s[k:] ** 2) + np.sum(tail ** 2))
        s = s[:k]
        if self.renorm:
            s = s * (full / np.linalg.norm(s))
        A, B = _split(U[:, :k], s, Vh[:k], dirn, l, r)
        return A, B, disc


def _basis(Rm, tol):
    """Orthogonal Q (k x k) whose first rk columns span the column space of Rm (k x 4); rk = numerical rank."""
    Q, sv, _ = np.linalg.svd(Rm, full_matrices=True)
    if sv[0] <= 1e-300:
        return Q, 0
    return Q, int(np.sum(sv > tol * sv[0]))


class DMTCut:
    """Reference-conditioned DMT cut.  Needs ctx['kL'], ctx['kR'] (environment contractions with the reference
    covectors); with check=True also ctx['vb'], ctx['vb1'] and the violation of every preserved functional is logged."""
    needs_env = True

    def __init__(self, name, tol=1e-13, check=False):
        self.name, self.tol, self.check = name, tol, check
        self.viol = dict(a=0.0, b=0.0, full=0.0)       # worst absolute violation over all cuts so far
        self.reserved = []
        self.ntrunc = 0

    def __call__(self, theta, chi, dirn, ctx):
        l, r = theta.shape[0], theta.shape[3]
        U, s, Vh, tail = _thin_svd(theta)
        k = len(s)
        if k <= chi:
            A, B = _split(U, s, Vh, dirn, l, r)
            return A, B, float(np.sum(tail ** 2))
        self.ntrunc += 1
        kL, kR = ctx['kL'], ctx['kR']
        RL = np.tensordot(kL, U.reshape(l, 4, k), axes=([0], [0])).T        # (k,4): U^T (kL (x) I4)
        RR = np.tensordot(Vh.reshape(k, 4, r), kR, axes=([2], [0]))          # (k,4): Vh (I4 (x) kR)
        QL, rL = _basis(RL, self.tol)
        QR, rR = _basis(RR, self.tol)
        rD = max(chi - rL - rR, 0)
        Mt = (QL.T * s) @ QR                                                 # bond matrix in the rotated bases
        D = Mt[rL:, rR:]
        Ud, sd, Vd = np.linalg.svd(D, full_matrices=False)
        Mt2 = Mt.copy()
        Mt2[rL:, rR:] = (Ud[:, :rD] * sd[:rD]) @ Vd[:rD]
        disc = float(np.sum(sd[rD:] ** 2) + np.sum(tail ** 2))
        U2, s2, V2 = np.linalg.svd(Mt2, full_matrices=False)
        k2 = max(1, min(chi, int(np.sum(s2 > EPS_S * s2[0])))) if s2[0] > 0 else 1
        Un = U @ (QL @ U2[:, :k2])
        Vn = (V2[:k2] @ QR.T) @ Vh
        A, B = _split(Un, s2[:k2], Vn, dirn, l, r)
        self.reserved.append((rL, rR))
        if self.check:
            self._check(theta, A, B, dirn, ctx)
        return A, B, disc

    def _check(self, theta, A, B, dirn, ctx):
        th2 = np.tensordot(A, B, axes=([2], [0]))
        d = th2 - theta
        kL, kR = ctx['kL'], ctx['kR']
        fa = np.linalg.norm(np.tensordot(d, kR, axes=([3], [0])))
        fb = np.linalg.norm(np.tensordot(kL, d, axes=([0], [0])))
        ff = abs(np.einsum('l,lpqr,p,q,r->', kL, d, ctx['vb'], ctx['vb1'], kR))
        self.viol['a'] = max(self.viol['a'], fa)
        self.viol['b'] = max(self.viol['b'], fb)
        self.viol['full'] = max(self.viol['full'], ff)


# ----------------------------------------------------------------------------------------------- the sweep
def make_cut(name, **kw):
    if name == 'svd_raw':
        return SVDCut(False)
    if name == 'svd_renorm':
        return SVDCut(True)
    if name.startswith('dmt'):
        return DMTCut(name, **kw)
    raise ValueError(name)


def ref_provider(model, N, kind):
    """Return f(step_index) -> (N,4) covectors for the cut reference, or None for SVD arms.
    kind: 'I' (identity), 'neel' (initial product state, static) or a callable."""
    if kind == 'I':
        v = identity_covectors(N)
        return lambda n: v
    if kind == 'neel':
        v = initial_covectors(model, N)
        return lambda n: v
    if callable(kind):
        return kind
    raise ValueError(kind)


def run_heis(model, N, i0, chi, nsteps, dt, cut, ref=None, pauli=3, gates=None, hook=None, verbose=False):
    """Strang sweeps for the Heisenberg MPO.  cut(theta, chi, dirn, ctx) -> (A, B, discarded weight^2).
    ref(n) -> (N,4) covectors used by the cut at step n (n = 1..nsteps); the READOUT always uses the initial state.
    Returns dict with val[n] (n = 0..nsteps), params[n], maxbond[n], norm[n], disc[n]; and the final tensors."""
    S = gates or pauli_gates(model, N, dt)
    T = initial_operator(N, i0, pauli)
    vro = initial_covectors(model, N)
    needs_env = getattr(cut, 'needs_env', False)
    val = [contract_cov(T, vro)]
    params, maxbond, nrm, disc = [sum(t.size for t in T)], [1], [1.0], [0.0]
    one = np.ones(1)
    t0 = time.time()
    for n in range(1, nsteps + 1):
        stepdisc = 0.0
        if needs_env:
            v = ref(n)
            Lr = [None] * (N + 1)
            Rr = [None] * (N + 1)
            Lr[0] = one
            Rr[N] = one
            for j in range(N - 1, 0, -1):
                Rr[j] = np.einsum('lpa,p,a->l', T[j], v[j], Rr[j + 1])
        for sweep in ('R', 'L'):
            bonds = range(N - 1) if sweep == 'R' else range(N - 2, -1, -1)
            for b in bonds:
                th = np.tensordot(T[b], T[b + 1], axes=([2], [0]))
                th = np.einsum('abst,lstr->labr', S[b], th)
                ctx = None
                if needs_env:
                    ctx = dict(kL=Lr[b], kR=Rr[b + 2], vb=v[b], vb1=v[b + 1], b=b, step=n)
                snap = list(T) if hook is not None else None
                A, B, d = cut(th, chi, sweep, ctx)
                if hook is not None:
                    hook(b, sweep, n, snap, th, A, B)
                T[b], T[b + 1] = A, B
                stepdisc += d
                if needs_env:
                    if sweep == 'R':
                        Lr[b + 1] = np.einsum('l,lpa,p->a', Lr[b], A, v[b])
                    else:
                        Rr[b + 1] = np.einsum('lpa,p,a->l', B, v[b + 1], Rr[b + 2])
        val.append(contract_cov(T, vro))
        params.append(sum(t.size for t in T))
        maxbond.append(max(t.shape[2] for t in T[:-1]))
        nrm.append(float(np.linalg.norm(T[0])))
        disc.append(stepdisc)
        if verbose and n % 10 == 0:
            print(f'  step {n}/{nsteps} val={val[-1]:+.6f} bond={maxbond[-1]} {time.time() - t0:.1f}s', flush=True)
    return dict(val=np.array(val), params=np.array(params), maxbond=np.array(maxbond), norm=np.array(nrm),
                disc=np.array(disc), wall=time.time() - t0, T=T)


def dense_op(T):
    """Dense coefficient tensor (4,)*N of the MPO (tests only)."""
    x = T[0][0]                       # (4, r)
    for t in T[1:]:
        x = np.tensordot(x, t, axes=([-1], [0]))
    return x[..., 0]
